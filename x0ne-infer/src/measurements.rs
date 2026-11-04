//! Reference measurements published with a release: `measurements/<version>.json`
//! at the repository root, embedded at compile time for the binary's own
//! version. Server and clients pin them by default; `--expect-measurement`
//! is an explicit override, `--measurements <file>` another file for the
//! same version (an image rebuilt before the next release).
//!
//! ```json
//! {
//!   "format": 1,
//!   "version": "0.2.0-beta.7",
//!   "algorithm": "sha384",
//!   "image": { "name": "…", "built": "YYYY-MM-DD", "source": "…" },
//!   "provenance": { "reference_signed_by": "…", "release": true },
//!   "registers": { "rtmr1": "<96 hex>", "rtmr2": "<96 hex>" },
//!   "signature": null
//! }
//! ```
//!
//! `provenance`, optional, is written by `run_fixtures`: the archive key that
//! signed the image reference the registers were checked against. A release
//! build refuses `"release": false` (the TEST key); others warn.
//! `registers` names what is pinned (`mrtd`, `rtmr0..rtmr3`, SHA-384 hex);
//! empty means nothing has been published for this version. `signature` is
//! reserved for a detached signature by the release key: this version checks
//! that it is absent and refuses a file that claims one it cannot verify.
//!
//! A release build with nothing pinned refuses to serve, and a release
//! client refuses to verify: "verified" would otherwise mean genuine
//! hardware running an unknown image. A debug build warns instead.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use x0ne_confidential::policy::ExpectedMeasurement;

/// The only file format this version reads.
pub const FORMAT: u64 = 1;
/// The version the embedded file must name: the binary's.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// `measurements/<VERSION>.json`, embedded. A release without the file does
/// not build.
pub const BUILTIN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../measurements/",
    env!("CARGO_PKG_VERSION"),
    ".json"
));
/// Exit code of `x0ne-infer` when no reference measurement can be pinned.
pub const EXIT_UNPINNED: i32 = 87;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Image {
    pub name: String,
    #[serde(default)]
    pub built: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
}

/// `reference_signed_by` for a reference the TEST archive key signed.
pub const TEST_ARCHIVE_KEY: &str = "test-archive-key";

/// The archive key behind the reference a run was checked against:
/// [`TEST_ARCHIVE_KEY`] or the release signing subkey's fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub reference_signed_by: String,
    pub release: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    format: u64,
    version: String,
    algorithm: String,
    #[serde(default)]
    image: Option<Image>,
    #[serde(default)]
    provenance: Option<Provenance>,
    registers: BTreeMap<String, String>,
    #[serde(default)]
    signature: Option<serde_json::Value>,
}

/// A measurements file that parsed, for the version asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Published {
    pub version: String,
    pub image: Option<Image>,
    pub provenance: Option<Provenance>,
    /// One pin per register, in register order.
    pub pins: Vec<ExpectedMeasurement>,
    /// Where it came from, for messages.
    pub source: String,
}

/// Parse `json` as the measurements file of `version`. Refuses another
/// format, another version, another algorithm, an unknown register, a
/// malformed value, or a signature this version cannot verify.
pub fn parse(json: &str, version: &str, source: &str) -> Result<Published, String> {
    let file: File = serde_json::from_str(json).map_err(|e| format!("{source}: {e}"))?;
    if file.format != FORMAT {
        return Err(format!(
            "{source}: format {} is not {FORMAT}, the only one this version reads",
            file.format
        ));
    }
    if file.version != version {
        return Err(format!(
            "{source}: measurements for version {}, this binary is {version}",
            file.version
        ));
    }
    if file.algorithm != "sha384" {
        return Err(format!(
            "{source}: algorithm {:?} is not sha384 (TDX registers)",
            file.algorithm
        ));
    }
    if file.signature.is_some() {
        return Err(format!(
            "{source}: carries a signature, and this version cannot verify one: refused rather \
             than taken as signed"
        ));
    }
    let mut pins = Vec::with_capacity(file.registers.len());
    for (register, value) in &file.registers {
        pins.push(
            crate::verify::parse_expected_measurement(&format!("{register}={value}"))
                .map_err(|e| format!("{source}: {e}"))?,
        );
    }
    Ok(Published {
        version: file.version,
        image: file.image,
        provenance: file.provenance,
        pins,
        source: source.to_string(),
    })
}

/// The file embedded for this binary's version.
pub fn builtin() -> Result<Published, String> {
    parse(
        BUILTIN,
        VERSION,
        &format!("measurements/{VERSION}.json (built in)"),
    )
}

/// A file passed by the operator, for this binary's version.
pub fn load(path: &Path) -> Result<Published, String> {
    let source = path.display().to_string();
    // Regular file, capped: a FIFO or /dev/zero here must not block the start.
    let bytes = crate::fsread::read_capped(path, crate::fsread::CONFIG_CAP)
        .map_err(|e| format!("{source}: {e}"))?;
    let json = String::from_utf8(bytes).map_err(|e| format!("{source}: {e}"))?;
    parse(&json, VERSION, &source)
}

/// What a server or client will pin, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    pub pins: Vec<ExpectedMeasurement>,
    /// `--expect-measurement`, or the file the pins were read from.
    pub origin: String,
    /// Set when nothing is pinned and the build allowed it.
    pub warning: Option<String>,
}

/// Choose the pins. `overrides` (`--expect-measurement`) win and the file is
/// not read; otherwise `published` (the file) must name at least one
/// register, unless the build is not a release.
pub fn resolve_from(
    overrides: &[ExpectedMeasurement],
    published: Result<Published, String>,
    release: bool,
) -> Result<Resolved, String> {
    if !overrides.is_empty() {
        return Ok(Resolved {
            pins: overrides.to_vec(),
            origin: "--expect-measurement".into(),
            warning: None,
        });
    }
    let published = published?;
    let not_release = published
        .provenance
        .as_ref()
        .filter(|p| !p.release)
        .map(|p| {
            format!(
                "{} was measured against a reference signed by {}, not by the release archive key",
                published.source, p.reference_signed_by
            )
        });
    if let (Some(why), true) = (&not_release, release) {
        return Err(format!(
            "{why}: refused by a release build. Pin with --expect-measurement <reg>=<hex>, or \
             --measurements <file> of a release run"
        ));
    }
    if !published.pins.is_empty() {
        return Ok(Resolved {
            pins: published.pins,
            origin: published.source,
            warning: not_release.map(|why| format!("{why}; allowed because this is a debug build")),
        });
    }
    let unpinned = format!(
        "no reference measurement is published for version {} ({} names no register): a \
         verified quote would prove genuine hardware, not a particular image",
        published.version, published.source
    );
    if release {
        return Err(format!(
            "{unpinned}. Pin one: --expect-measurement <reg>=<hex>, --measurements <file>, or \
             publish measurements/{}.json (docs/pinning.md)",
            published.version
        ));
    }
    Ok(Resolved {
        pins: Vec::new(),
        origin: published.source,
        warning: Some(format!("{unpinned}; allowed because this is a debug build")),
    })
}

/// [`resolve_from`] with the built-in file, or `file` when given, for THIS
/// build: a release unless compiled with debug assertions.
pub fn resolve(overrides: &[ExpectedMeasurement], file: Option<&Path>) -> Result<Resolved, String> {
    let published = match file {
        Some(path) => load(path),
        None => builtin(),
    };
    resolve_from(overrides, published, !cfg!(debug_assertions))
}

/// The measurements file for this version, from the TDX quote of an
/// `/attestation` document (`evidence.tdx_quote_b64`), pinning `registers`.
/// The quote is parsed, not verified: run this on a document whose quote
/// you verified.
pub fn render(
    attestation: &serde_json::Value,
    registers: &[&str],
    image: Option<&str>,
    provenance: Option<&Provenance>,
) -> Result<String, String> {
    use base64::Engine as _;
    let quote = base64::engine::general_purpose::STANDARD
        .decode(
            attestation["evidence"]["tdx_quote_b64"]
                .as_str()
                .ok_or("attestation has no evidence.tdx_quote_b64")?,
        )
        .map_err(|e| format!("tdx_quote_b64: {e}"))?;
    let measured = crate::verify::tdx_measurements(&quote)?;
    let mut out = serde_json::Map::new();
    for name in registers {
        let m = measured
            .iter()
            .find(|m| m.register == *name)
            .ok_or_else(|| {
                "a register named is not a TDX register (mrtd, rtmr0..rtmr3); \
                 the name given is not printed"
                    .to_string()
            })?;
        out.insert(
            m.register.clone(),
            serde_json::Value::String(m.value_hex.clone()),
        );
    }
    let mut file = serde_json::json!({
        "format": FORMAT,
        "version": VERSION,
        "algorithm": "sha384",
        "image": image.map(|name| serde_json::json!({ "name": name })),
        "registers": out,
        "signature": serde_json::Value::Null,
    });
    if let Some(p) = provenance {
        file["provenance"] = serde_json::json!(p);
    }
    let json = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
    // Round trip: what is written must be what this version reads.
    parse(&json, VERSION, "rendered")?;
    Ok(json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use x0ne_confidential::model::{AttestationEvidence, EvidenceFormat, Verdict};
    use x0ne_confidential::policy::VerificationPolicy;

    const FIXTURE: &str = include_str!("../tests/measurements/fixture.json");

    fn file(registers: &str) -> String {
        format!(
            r#"{{"format":1,"version":"{VERSION}","algorithm":"sha384","image":null,"registers":{{{registers}}},"signature":null}}"#
        )
    }

    fn pin(register: &str, byte: &str) -> ExpectedMeasurement {
        crate::verify::parse_expected_measurement(&format!("{register}={}", byte.repeat(48)))
            .expect("well-formed")
    }

    fn published(pins: Vec<ExpectedMeasurement>) -> Published {
        Published {
            version: VERSION.into(),
            image: None,
            provenance: None,
            pins,
            source: "test".into(),
        }
    }

    /// SECURITY: a release that pins nothing would verify any image on
    /// genuine hardware. A failure here means a release client or server
    /// starts without a pin.
    #[test]
    fn a_release_without_a_pin_refuses_and_says_how_to_pin() {
        let err = resolve_from(&[], Ok(published(vec![])), true).unwrap_err();
        assert!(
            err.contains("no reference measurement is published"),
            "{err}"
        );
        assert!(err.contains("--expect-measurement"), "{err}");
        assert!(
            err.contains(&format!("measurements/{VERSION}.json")),
            "{err}"
        );
    }

    #[test]
    fn a_debug_build_without_a_pin_is_allowed_with_a_warning() {
        let r = resolve_from(&[], Ok(published(vec![])), false).expect("allowed");
        assert!(r.pins.is_empty());
        let warning = r.warning.expect("warns");
        assert!(warning.contains("no reference measurement"), "{warning}");
        assert!(
            resolve_from(&[], Ok(published(vec![pin("rtmr1", "b2")])), true)
                .expect("pinned")
                .warning
                .is_none()
        );
    }

    /// `--expect-measurement` is an override: the file is not consulted,
    /// not even when it is broken.
    #[test]
    fn an_explicit_pin_overrides_the_published_file() {
        let ours = pin("rtmr1", "b2");
        let r = resolve_from(std::slice::from_ref(&ours), Err("unreadable".into()), true)
            .expect("the override wins");
        assert_eq!(r.pins, vec![ours]);
        assert_eq!(r.origin, "--expect-measurement");
    }

    /// A file that cannot be read is a refusal in every build: a missing or
    /// malformed file must never quietly become "nothing pinned".
    #[test]
    fn an_absent_or_malformed_file_is_refused_in_every_build() {
        let dir = tempfile::tempdir().expect("tempdir");
        let absent = dir.path().join("absent.json");
        for release in [true, false] {
            let err = resolve_from(&[], load(&absent), release).unwrap_err();
            assert!(err.contains("absent.json"), "{err}");
        }
        let malformed = dir.path().join("malformed.json");
        std::fs::write(&malformed, "{\"format\": 1,").expect("write");
        assert!(load(&malformed).is_err());
        let unknown_field = file(r#""rtmr1":"x""#).replace("\"image\"", "\"images\"");
        assert!(parse(&unknown_field, VERSION, "t").is_err());
    }

    #[test]
    fn another_version_format_algorithm_register_or_signature_is_refused() {
        let good = file(&format!(r#""rtmr1":"{}""#, "b2".repeat(48)));
        assert!(parse(&good, VERSION, "t").is_ok());
        let err = parse(&good, "9.9.9", "t").unwrap_err();
        assert!(err.contains("9.9.9") && err.contains(VERSION), "{err}");
        assert!(parse(&good.replace("\"format\":1", "\"format\":2"), VERSION, "t").is_err());
        assert!(parse(&good.replace("sha384", "sha256"), VERSION, "t").is_err());
        let err = parse(
            &good.replace("\"signature\":null", "\"signature\":\"abc\""),
            VERSION,
            "t",
        )
        .unwrap_err();
        assert!(err.contains("signature"), "{err}");
        for bad in [
            format!(r#""rtmr4":"{}""#, "b2".repeat(48)),
            format!(r#""rtmr1":"{}""#, "b2".repeat(32)),
            format!(r#""rtmr1":"{}""#, "zz".repeat(48)),
        ] {
            assert!(parse(&file(&bad), VERSION, "t").is_err(), "{bad}");
        }
    }

    /// SECURITY: registers measured under a reference the TEST
    /// archive key signed are refused by a release; a debug build takes them
    /// and says so.
    #[test]
    fn a_release_refuses_a_file_of_test_key_provenance() {
        let test = file(&format!(r#""rtmr1":"{}""#, "b2".repeat(48))).replace(
            "\"registers\"",
            r#""provenance":{"reference_signed_by":"test-archive-key","release":false},"registers""#,
        );
        let err = resolve_from(&[], parse(&test, VERSION, "t"), true).unwrap_err();
        assert!(err.contains("test-archive-key"), "{err}");
        assert!(err.contains("release archive key"), "{err}");
        let r = resolve_from(&[], parse(&test, VERSION, "t"), false).expect("allowed");
        assert_eq!(r.pins.len(), 1);
        let warning = r.warning.expect("warns");
        assert!(warning.contains("test-archive-key"), "{warning}");
        let released = test.replace(
            r#""reference_signed_by":"test-archive-key","release":false"#,
            r#""reference_signed_by":"0123456789ABCDEF0123456789ABCDEF01234567","release":true"#,
        );
        let r = resolve_from(&[], parse(&released, VERSION, "t"), true).expect("release");
        assert!(r.warning.is_none());
    }

    /// The embedded file parses for this binary's version. Its registers
    /// may be empty until an image is built and measured.
    #[test]
    fn the_builtin_file_is_for_this_version() {
        let p = builtin().expect("measurements/<version>.json parses");
        assert_eq!(p.version, VERSION);
        assert!(p.source.contains(VERSION));
    }

    /// The fixture file, through the file format, is enforced on the
    /// cryptographic path exactly like `--expect-measurement`.
    #[test]
    fn a_published_pin_is_enforced_on_the_cryptographic_path() {
        let fx = crate::pcs::tests::fixture_for_composite();
        let ev = AttestationEvidence {
            format: EvidenceFormat::TdxQuote,
            payload: fx.quote.clone(),
            collected_at_unix: 1_000,
        };
        let anchor = crate::pcs::TrustAnchor::RootDer(fx.root_der.clone());
        let policy = |pins: Vec<ExpectedMeasurement>| VerificationPolicy {
            workload_name: "x0ne-infer".into(),
            accepted_cpu: vec![x0ne_confidential::model::TeeTechnology::IntelTdx],
            require_gpu_cc_on: false,
            expected_measurements: pins,
        };
        let verdict = |p: &VerificationPolicy| {
            crate::verify::cryptographic_tdx(
                &ev,
                p,
                &anchor,
                Some(&[0x5a; 64]),
                crate::pcs::tests::NOW,
            )
        };

        let ours = resolve_from(&[], parse(FIXTURE, VERSION, "fixture"), true)
            .expect("the fixture pins mrtd and rtmr1");
        assert_eq!(ours.pins.len(), 2);
        let Verdict::Verified { verifier, .. } = verdict(&policy(ours.pins)) else {
            panic!("the image the file names must verify");
        };
        assert!(
            verifier.contains("pinned and matched: mrtd, rtmr1"),
            "{verifier}"
        );

        let theirs = parse(
            &file(&format!(r#""rtmr1":"{}""#, "c3".repeat(48))),
            VERSION,
            "theirs",
        )
        .expect("well-formed");
        let Verdict::Rejected { reason, .. } = verdict(&policy(theirs.pins)) else {
            panic!("another image must be refused");
        };
        assert!(
            reason.contains("rtmr1") && reason.contains("mismatch"),
            "{reason}"
        );
    }

    /// `render` writes, from a quote, what `parse` reads.
    #[test]
    fn render_produces_a_file_this_version_reads_back() {
        use base64::Engine as _;
        let fx = crate::pcs::tests::fixture_for_composite();
        let doc = serde_json::json!({ "evidence": { "tdx_quote_b64":
            base64::engine::general_purpose::STANDARD.encode(&fx.quote) } });
        let json = render(&doc, &["rtmr1", "rtmr2"], Some("fixture"), None).expect("renders");
        let p = parse(&json, VERSION, "rendered").expect("reads back");
        assert_eq!(p.pins[0], pin("rtmr1", "b2"));
        assert_eq!(p.pins.len(), 2);
        assert_eq!(p.image.map(|i| i.name).as_deref(), Some("fixture"));
        assert!(render(&doc, &["rtmr9"], None, None).is_err());
        assert!(render(&serde_json::json!({}), &["rtmr1"], None, None).is_err());
    }
}
