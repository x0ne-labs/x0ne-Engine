//! `image import`: once per project, the CVM image built by x0ne-OS
//! (`disk.tar.gz`), through a transit bucket that is deleted at the end, on
//! failure and on Ctrl-C.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::color::{paint, Style};
use crate::provider::{gcp, ImageRef, ImportRequest, Leftover, LeftoverKind, Provider};
use crate::{Failure, DEFAULT_IMAGE};

pub const REFERENCE_SCHEMA: &str = "x0ne-engine/reference/v2";
/// Written before RTMR1 pinned the disk's GPT: the verifier refuses it, so does `image import`.
const REFERENCE_SCHEMA_V1: &str = "x0ne-engine/reference/v1";
const MAX_REFERENCE: u64 = 1 << 20;
const MAX_DISK: u64 = 64 << 30;
/// Image storage, USD per GiB per month: an estimate (unverified), not read from the catalog.
pub const IMAGE_STORAGE_USD_GIB_MONTH: f64 = 0.05;

/// The GCP image to boot, and the disk digest an image reference names.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageSpec {
    pub gcp_image: String,
    pub disk_tar_sha256: Option<String>,
    /// `--reference`; None for an installed reference, which the relay finds itself.
    pub reference: Option<PathBuf>,
    /// Where the name comes from, for messages.
    pub from: String,
}

pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

/// Reads `image.gcp_image` and `image.disk_tar_sha256` of an image reference.
pub fn reference_image(path: &Path) -> Result<ImageSpec, Failure> {
    let (gcp_image, disk_tar_sha256) = read_image(path)
        .map_err(|why| Failure::Usage(format!("--reference {}: {why}", path.display())))?;
    Ok(ImageSpec {
        gcp_image,
        disk_tar_sha256,
        reference: Some(path.to_path_buf()),
        from: format!("--reference {}", path.display()),
    })
}

fn read_image(path: &Path) -> Result<(String, Option<String>), String> {
    let bad = |why: &str| why.to_string();
    let f = File::open(path).map_err(|e| bad(&e.to_string()))?;
    let mut raw = Vec::new();
    f.take(MAX_REFERENCE + 1)
        .read_to_end(&mut raw)
        .map_err(|e| bad(&e.to_string()))?;
    if raw.len() as u64 > MAX_REFERENCE {
        return Err(bad("larger than 1 MiB"));
    }
    let v: Value = serde_json::from_slice(&raw).map_err(|_| bad("not JSON"))?;
    match v["schema"].as_str() {
        Some(REFERENCE_SCHEMA) => {}
        Some(REFERENCE_SCHEMA_V1) => {
            return Err(bad(&format!(
                "schema {REFERENCE_SCHEMA_V1}: written before RTMR1 pinned the disk's GPT, refused \
                 by this version; use the {REFERENCE_SCHEMA} reference.json built with the image"
            )))
        }
        _ => return Err(bad(&format!("schema is not {REFERENCE_SCHEMA}"))),
    }
    if v["provider"].as_str() != Some("gcp") {
        return Err(bad("not a reference for provider gcp"));
    }
    let name = v["image"]["gcp_image"]
        .as_str()
        .filter(|n| crate::valid_resource_name(n))
        .ok_or_else(|| bad("image.gcp_image missing or not an image name"))?;
    let digest = match v["image"]["disk_tar_sha256"].as_str() {
        Some(d) if is_sha256_hex(d) => Some(d.to_string()),
        Some(_) => return Err(bad("image.disk_tar_sha256 is not 64 lowercase hex digits")),
        None => None,
    };
    Ok((name.to_string(), digest))
}

/// The version of `x0ne-cvm-<version>.json`, digits and dots only: x0ne-infer
/// `reference::version_key`, the relay's choice. A manifest next to it is not one.
fn version_key(name: &str) -> Option<Vec<u64>> {
    name.strip_prefix("x0ne-cvm-")?
        .strip_suffix(".json")?
        .split('.')
        .map(|p| {
            p.bytes()
                .all(|b| b.is_ascii_digit())
                .then(|| p.parse().ok())
                .flatten()
        })
        .collect()
}

/// The newest installed `x0ne-cvm-<version>.json` in `dir`, as the relay picks it.
pub fn installed_reference_in(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter_map(|n| version_key(&n).map(|v| (v, n)))
        .max()
        .map(|(_, n)| dir.join(n))
}

/// `--reference`, else the reference the relay will take in `installed`, else
/// [`DEFAULT_IMAGE`] (and the relay, finding none, refuses: 89). The relay
/// names the image of the reference it verified; `up` refuses another.
pub fn image_spec(reference: Option<&Path>, installed: &Path) -> Result<ImageSpec, Failure> {
    if let Some(p) = reference {
        return reference_image(p);
    }
    let Some(p) = installed_reference_in(installed) else {
        return Ok(ImageSpec {
            gcp_image: DEFAULT_IMAGE.into(),
            disk_tar_sha256: None,
            reference: None,
            from: format!("DEFAULT_IMAGE, no reference in {}", installed.display()),
        });
    };
    let (gcp_image, disk_tar_sha256) = read_image(&p)
        .map_err(|why| Failure::Preflight(format!("installed reference {}: {why}", p.display())))?;
    Ok(ImageSpec {
        gcp_image,
        disk_tar_sha256,
        reference: None,
        from: format!("installed reference {}", p.display()),
    })
}

/// A READY image under the reference's name may be another build's: 0.2.0-beta.4
/// and beta.5 both build x0ne-cvm-0-2 (rev10, 2026-10-17). Without a
/// reference there is nothing to compare with.
pub fn same_build(image: &ImageRef, want: Option<&str>) -> Result<(), Failure> {
    let Some(want) = want else {
        return Ok(());
    };
    if image.disk_sha256.as_deref() == Some(want) {
        return Ok(());
    }
    Err(Failure::Preflight(format!(
        "image {} in project {} holds disk.tar.gz sha256 {}, the reference names {want}: \
         another build under the same name. Delete it \
         (x0ne-engine cloud image delete {} --project {}), then x0ne-engine cloud image import",
        image.name,
        image.project,
        image.disk_sha256.as_deref().unwrap_or("(none recorded)"),
        image.name,
        image.project
    )))
}

pub fn sha256_file(path: &Path) -> io::Result<(u64, String)> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        total += n as u64;
    }
    Ok((total, crate::hex(&h.finalize())))
}

pub async fn import(
    provider: &dyn Provider,
    project: &str,
    spec: &ImageSpec,
    file: &Path,
    region: &str,
    yes: bool,
) -> Result<(), Failure> {
    let usage = |why: String| Failure::Usage(format!("{}: {why}", file.display()));
    let meta = std::fs::metadata(file).map_err(|e| usage(e.to_string()))?;
    if !meta.is_file() || meta.len() == 0 || meta.len() > MAX_DISK {
        return Err(usage(
            "not a disk archive (a non-empty file up to 64 GiB)".into(),
        ));
    }
    if !file.to_string_lossy().ends_with(".tar.gz") {
        return Err(usage(
            "Compute Engine imports a .tar.gz holding disk.raw (x0ne-OS: make cvm)".into(),
        ));
    }
    let existing = provider.image(&spec.gcp_image).await?;
    match existing.status.as_str() {
        "READY" => {
            same_build(&existing, spec.disk_tar_sha256.as_deref())?;
            println!(
                "Image {} already exists in project {project}: nothing to do.",
                spec.gcp_image
            );
            return Ok(());
        }
        "MISSING" => {}
        other => {
            return Err(Failure::Preflight(format!(
                "image {} exists in state {other}: wait for it, or delete it \
                 (x0ne-engine cloud image delete {} --project {project})",
                spec.gcp_image, spec.gcp_image
            )))
        }
    }
    sayln!("Reading {} ...", file.display());
    let owned = file.to_path_buf();
    let (size, sha) = tokio::task::spawn_blocking(move || sha256_file(&owned))
        .await
        .map_err(|e| Failure::Preflight(e.to_string()))?
        .map_err(|e| usage(e.to_string()))?;
    if let Some(want) = &spec.disk_tar_sha256 {
        if *want != sha {
            return Err(Failure::Preflight(format!(
                "{} has sha256 {sha}, the reference names {want}: not the reviewed image. \
                 Nothing was uploaded",
                file.display()
            )));
        }
    }
    let bucket = gcp::import_bucket_name(
        &crate::random_hex(6).map_err(|e| Failure::Preflight(format!("/dev/urandom: {e}")))?,
    );
    let gib = size as f64 / f64::from(1u32 << 30);
    let on = crate::color::stdout_on();
    println!(
        "Image import: {} ({gib:.2} GiB, sha256 {sha}, {})",
        file.display(),
        if spec.disk_tar_sha256.is_some() {
            paint(on, Style::Pass, "matches the reference")
        } else {
            paint(on, Style::Warn, "no reference to compare with")
        }
    );
    println!(
        "  creates image {} in project {project}: {} \
         (estimate, {IMAGE_STORAGE_USD_GIB_MONTH} USD/GiB/month, not read from the catalog)",
        spec.gcp_image,
        paint(
            on,
            Style::Warn,
            &format!(
                "about {:.2} USD per month of storage",
                crate::quote::ceil_cents(gib * IMAGE_STORAGE_USD_GIB_MONTH)
            )
        )
    );
    println!(
        "  through the transit bucket {bucket} in {region}, deleted at the end \
         (Google also drops its object after one day)"
    );
    crate::up::confirm(yes, "import the image")?;
    let req = ImportRequest {
        project: project.to_string(),
        bucket: bucket.clone(),
        region: region.to_string(),
        image: spec.gcp_image.clone(),
        file: file.to_path_buf(),
        size,
        sha256_hex: sha,
    };
    // The signals that stop `up` cleanly: any of them must not leave the bucket.
    let (tx, mut stops) = tokio::sync::mpsc::unbounded_channel();
    crate::up::forward_stop_signals(&tx, |name| name)?;
    let interrupted = tokio::select! {
        r = provider.import_image(&req) => {
            let img = r?;
            if img.status == "FAILED" {
                return Err(Failure::Api(format!("image {} ended in state FAILED", img.name)));
            }
            let status = match img.status.as_str() {
                "READY" => paint(crate::color::stdout_on(), Style::Pass, &img.status),
                s => s.to_string(),
            };
            println!(
                "Image {} is {status} in project {project}. Transit bucket deleted.",
                img.name
            );
            return Ok(());
        }
        s = stops.recv() => s.unwrap_or("shutdown"),
    };
    // The import was dropped midway: remove what it may have created.
    sayln!(
        "\n{}",
        paint(
            crate::color::stderr_on(),
            Style::Warn,
            &format!("{interrupted}: deleting the transit bucket {bucket} ...")
        )
    );
    let l = Leftover {
        kind: LeftoverKind::Bucket,
        name: bucket.clone(),
        location: region.to_string(),
        detail: String::new(),
    };
    match provider.delete_leftover(&l).await {
        Ok(()) => Err(Failure::Interrupted(format!(
            "image import stopped by {interrupted}; transit bucket deleted. If the image creation \
             had started, check it: gcloud compute images describe {} --project {project}",
            spec.gcp_image
        ))),
        Err(e) => Err(Failure::DeleteUnconfirmed(format!(
            "transit bucket {bucket} not deleted ({}): gcloud storage rm -r gs://{bucket} --project {project}",
            e.message()
        ))),
    }
}

/// `image delete`: an image `image import` made, after "yes", and never one
/// a VM still boots from. A missing image is said and left at that.
pub async fn delete(
    provider: &dyn Provider,
    project: &str,
    name: &str,
    yes: bool,
) -> Result<(), Failure> {
    let img = provider.image(name).await?;
    if img.status == "MISSING" {
        println!("No image {name} in project {project}: nothing to delete.");
        return Ok(());
    }
    if !img.ours {
        return Err(Failure::Preflight(format!(
            "image {name} in project {project} does not carry the label x0ne-engine=image: \
             not made by x0ne-engine cloud image import, so not deleted here"
        )));
    }
    let users = provider.image_users(name).await?;
    if !users.is_empty() {
        return Err(Failure::Preflight(format!(
            "image {name} is the source of a disk attached to {}: not deleted while a VM \
             boots from it (x0ne-engine cloud down ends this machine's session)",
            users.join(", ")
        )));
    }
    let on = crate::color::stdout_on();
    println!(
        "Image {name} in project {project}: {}, disk.tar.gz sha256 {}",
        img.status,
        img.disk_sha256.as_deref().unwrap_or("(none recorded)")
    );
    println!(
        "  {}; image import makes it again from its disk.tar.gz",
        paint(on, Style::Warn, "deleting it cannot be undone")
    );
    if !yes {
        match crate::up::ask_yes("delete the image") {
            Ok(true) => {}
            Ok(false) => {
                println!("Nothing was deleted.");
                return Ok(());
            }
            Err(e) => {
                return Err(Failure::Usage(format!(
                    "{e}: run it in a terminal, or pass --yes. Nothing was deleted"
                )))
            }
        }
    }
    provider.delete_image(name).await?;
    println!("Image {name} deleted from project {project}.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, body: &str) -> PathBuf {
        let p = dir.join("ref.json");
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn a_reference_names_the_image_and_its_digest() {
        let d = tempfile::tempdir().unwrap();
        let sha = "ab".repeat(32);
        let p = write(
            d.path(),
            &format!(
                r#"{{"schema":"x0ne-engine/reference/v2","provider":"gcp","tee":"intel-tdx",
                    "image":{{"name":"x0ne-cvm","version":"0.1.0","gcp_image":"x0ne-cvm-0-1-0","disk_tar_sha256":"{sha}"}}}}"#
            ),
        );
        let s = reference_image(&p).unwrap();
        assert_eq!(s.gcp_image, "x0ne-cvm-0-1-0");
        assert_eq!(s.disk_tar_sha256.as_deref(), Some(sha.as_str()));
    }

    #[test]
    fn a_reference_for_another_provider_or_name_is_refused() {
        let d = tempfile::tempdir().unwrap();
        for body in [
            r#"{"schema":"x0ne-engine/reference/v2","provider":"azure","image":{"gcp_image":"x0ne-cvm-0-1-0"}}"#,
            r#"{"schema":"x0ne-engine/reference/v2","provider":"gcp","image":{"gcp_image":"../x"}}"#,
            r#"{"schema":"other","provider":"gcp","image":{"gcp_image":"x0ne-cvm-0-1-0"}}"#,
            r#"{"schema":"x0ne-engine/reference/v2","provider":"gcp","image":{"gcp_image":"x0ne-cvm-0-1-0","disk_tar_sha256":"…"}}"#,
        ] {
            assert!(reference_image(&write(d.path(), body)).is_err(), "{body}");
        }
        let none = tempfile::tempdir().unwrap();
        assert_eq!(
            image_spec(None, none.path()).unwrap().gcp_image,
            DEFAULT_IMAGE
        );
    }

    /// The relay's rule (x0ne-infer `reference::default_path_in`): the newest
    /// `x0ne-cvm-<digits and dots>.json`; `up` reads its image, and does not
    /// pass it as `--reference`, which would skip the relay's owner check.
    #[test]
    fn without_reference_the_installed_one_names_the_image() {
        let d = tempfile::tempdir().unwrap();
        let reference = |file: &str, image: &str, sha: &str| {
            std::fs::write(
                d.path().join(file),
                format!(
                    r#"{{"schema":"x0ne-engine/reference/v2","provider":"gcp","tee":"intel-tdx",
                        "image":{{"gcp_image":"{image}","disk_tar_sha256":"{sha}"}}}}"#
                ),
            )
            .unwrap();
        };
        let (a, b) = ("aa".repeat(32), "bb".repeat(32));
        reference("x0ne-cvm-0.2.1.json", "x0ne-cvm-0-2-1", &a);
        reference("x0ne-cvm-0.2.10.json", "x0ne-cvm-0-2-10", &b);
        reference("x0ne-cvm-0.2.10.manifest.json", "x0ne-cvm-manifest", &a);
        reference("x0ne-cvm-0.3.json.sig", "x0ne-cvm-sig", &a);
        let s = image_spec(None, d.path()).unwrap();
        assert_eq!(s.gcp_image, "x0ne-cvm-0-2-10");
        assert_eq!(s.disk_tar_sha256.as_deref(), Some(b.as_str()));
        assert_eq!(s.reference, None);
        assert!(s.from.contains("x0ne-cvm-0.2.10.json"), "{}", s.from);
        // `--reference` wins; a broken installed one is refused, not skipped.
        let given = d.path().join("x0ne-cvm-0.2.1.json");
        assert_eq!(
            image_spec(Some(&given), d.path()).unwrap().gcp_image,
            "x0ne-cvm-0-2-1"
        );
        std::fs::write(d.path().join("x0ne-cvm-0.9.json"), "not json").unwrap();
        assert!(matches!(
            image_spec(None, d.path()),
            Err(Failure::Preflight(m)) if m.contains("x0ne-cvm-0.9.json")
        ));
    }

    /// The image of x0ne OS ImageVersion 0.2.1 (2026-10-17), named as
    /// x0ne-OS `cvm/reference.py` names it: dots as dashes.
    #[test]
    fn with_no_reference_anywhere_the_image_is_x0ne_cvm_0_2_1() {
        let none = tempfile::tempdir().unwrap();
        let s = image_spec(None, none.path()).unwrap();
        assert_eq!(s.gcp_image, "x0ne-cvm-0-2-1");
        assert_eq!(
            s.gcp_image,
            format!("x0ne-cvm-{}", "0.2.1".replace('.', "-"))
        );
        assert!(crate::valid_resource_name(&s.gcp_image));
    }

    /// A v1 reference is refused by name before anything is uploaded or started.
    #[test]
    fn a_v1_reference_is_refused_with_what_to_use() {
        let d = tempfile::tempdir().unwrap();
        let p = write(
            d.path(),
            r#"{"schema":"x0ne-engine/reference/v1","provider":"gcp","tee":"intel-tdx",
                "image":{"name":"x0ne-cvm","version":"0.2","gcp_image":"x0ne-cvm-0-2"}}"#,
        );
        let Err(Failure::Usage(msg)) = reference_image(&p) else {
            panic!("a v1 reference was accepted")
        };
        assert!(
            msg.contains("x0ne-engine/reference/v1")
                && msg.contains(REFERENCE_SCHEMA)
                && msg.contains("GPT"),
            "{msg}"
        );
    }

    #[test]
    fn the_file_digest_is_streamed() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("disk.tar.gz");
        std::fs::write(&p, b"abc").unwrap();
        let (n, h) = sha256_file(&p).unwrap();
        assert_eq!(n, 3);
        assert_eq!(
            h,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
