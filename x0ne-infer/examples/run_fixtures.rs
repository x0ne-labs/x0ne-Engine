//! The fixtures of a real GCE boot from its run (`x0ne_infer::run_fixtures`).
//! Needs `--features probe`; `make run-fixtures` runs it in the CI container.
//!
//!   run_fixtures --attestation FILE --reference FILE --date YYYY-MM-DD
//!                --image NAME [--machine a3-highgpu-1g] --out DIR [--check REPO]
//!                [--test-key]
//!
//! Writes each file under DIR at its path in the repository. `--check REPO`
//! compares each with REPO's: exit 1 if one differs or is absent, 3 if the
//! run is refused (nothing written), 2 on a usage or I/O error. The quote is
//! verified to Intel's root at this machine's clock; `<reference>.sig` is read
//! if present. `--test-key` accepts a reference the release key did not sign.

use std::path::{Path, PathBuf};
use x0ne_infer::pcs::TrustAnchor;
use x0ne_infer::run_fixtures::{build, Inputs};

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let mut args = std::env::args().skip(1);
    let (mut attestation, mut reference, mut date, mut image, mut out, mut check) =
        (None, None, None, None, None, None);
    let mut machine = "a3-highgpu-1g".to_string();
    let mut test_key = false;
    while let Some(a) = args.next() {
        if a == "--test-key" {
            test_key = true;
            continue;
        }
        let v = args.next();
        let slot = match a.as_str() {
            "--attestation" => &mut attestation,
            "--reference" => &mut reference,
            "--date" => &mut date,
            "--image" => &mut image,
            "--out" => &mut out,
            "--check" => &mut check,
            "--machine" => {
                machine = v.unwrap_or_default();
                continue;
            }
            _ => {
                eprintln!("run_fixtures: unknown argument {a}");
                return 2;
            }
        };
        *slot = v;
    }
    let (Some(attestation), Some(reference), Some(date), Some(image), Some(out)) =
        (attestation, reference, date, image, out)
    else {
        eprintln!(
            "usage: run_fixtures --attestation FILE --reference FILE --date YYYY-MM-DD \
             --image NAME [--machine TYPE] --out DIR [--check REPO] [--test-key]"
        );
        return 2;
    };
    let read = |p: &str| std::fs::read(p).map_err(|e| format!("{p}: {e}"));
    // No .sig reads as no signature: refused unless --test-key.
    let signature = std::fs::read(x0ne_infer::reference::signature_path(Path::new(&reference)))
        .unwrap_or_default();
    let now_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let intel = TrustAnchor::intel_production();
    let built = read(&attestation).and_then(|a| {
        let r = read(&reference)?;
        build(&Inputs {
            attestation: &a,
            reference: &r,
            signature: &signature,
            date: &date,
            image: &image,
            machine: &machine,
            tdx: &intel,
            now_unix,
            test_key,
        })
    });
    let f = match built {
        Ok(f) => f,
        Err(e) => {
            eprintln!("run_fixtures: refused, nothing written: {e}");
            return 3;
        }
    };
    for n in &f.notes {
        println!("  {n}");
    }
    let mut rc = 0;
    for (rel, bytes) in &f.files {
        let path = PathBuf::from(&out).join(rel);
        let written = std::fs::create_dir_all(path.parent().unwrap())
            .and_then(|_| std::fs::write(&path, bytes));
        if let Err(e) = written {
            eprintln!("run_fixtures: {}: {e}", path.display());
            return 2;
        }
        let verdict = match &check {
            None => String::new(),
            Some(repo) => match std::fs::read(PathBuf::from(repo).join(rel)) {
                Ok(b) if &b == bytes => "  identical".into(),
                Ok(_) => {
                    rc = 1;
                    "  DIFFERS".into()
                }
                Err(_) => {
                    rc = 1;
                    "  ABSENT".into()
                }
            },
        };
        println!("{rel} ({} bytes){verdict}", bytes.len());
    }
    rc
}
