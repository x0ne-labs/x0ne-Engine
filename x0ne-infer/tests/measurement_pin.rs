//! The published measurements are the default pin of `x0ne-infer`, judged
//! before any evidence is collected: a refused file is exit 87 whatever the
//! build; a release with nothing to pin is exit 87; a debug build warns.

use std::path::Path;
use std::process::Command;

const EXIT_UNPINNED: i32 = 87;
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/measurements/fixture.json"
);

fn infer(args: &[&str]) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_x0ne-infer"))
        // cargo sets LD_LIBRARY_PATH for tests; a release x0ne-infer refuses it (73).
        .env_remove("LD_LIBRARY_PATH")
        .args(["--model-dir", "/nonexistent", "--listen", "127.0.0.1:0"])
        .args(args)
        .env("X0NE_INFER_KEY_HEX", "00".repeat(32))
        .output()
        .expect("the binary runs");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into(),
    )
}

fn confidential_machine() -> bool {
    Path::new("/sys/kernel/config/tsm").exists()
}

/// SECURITY: a missing or malformed file must never become "nothing pinned".
#[test]
fn an_absent_or_malformed_measurements_file_is_exit_87() {
    let dir = tempfile::tempdir().expect("tempdir");
    let absent = dir.path().join("absent.json");
    let (code, err) = infer(&["--measurements", absent.to_str().unwrap()]);
    assert_eq!(code, Some(EXIT_UNPINNED), "{err}");
    assert!(err.contains("absent.json"), "{err}");

    let malformed = dir.path().join("malformed.json");
    std::fs::write(&malformed, "{\"format\":1,").expect("write");
    let (code, err) = infer(&["--measurements", malformed.to_str().unwrap()]);
    assert_eq!(code, Some(EXIT_UNPINNED), "{err}");

    let other_version = dir.path().join("other.json");
    std::fs::write(
        &other_version,
        std::fs::read_to_string(FIXTURE)
            .expect("fixture")
            .replace(env!("CARGO_PKG_VERSION"), "0.0.0"),
    )
    .expect("write");
    let (code, err) = infer(&["--measurements", other_version.to_str().unwrap()]);
    assert_eq!(code, Some(EXIT_UNPINNED), "{err}");
    assert!(err.contains("0.0.0"), "{err}");
}

/// A well-formed file is pinned, and said so, before the evidence refusal.
#[test]
fn a_well_formed_file_is_pinned_before_evidence_is_collected() {
    if confidential_machine() {
        return;
    }
    let (code, err) = infer(&["--measurements", FIXTURE]);
    assert_eq!(code, Some(64), "{err}");
    assert!(err.contains("pinning mrtd, rtmr1"), "{err}");
}

/// `--expect-measurement` overrides the file, even an unreadable one.
#[test]
fn an_explicit_pin_overrides_the_file() {
    if confidential_machine() {
        return;
    }
    let spec = format!("rtmr1={}", "c3".repeat(48));
    let (code, err) = infer(&[
        "--expect-measurement",
        &spec,
        "--measurements",
        "/nonexistent",
    ]);
    assert_eq!(code, Some(64), "{err}");
    assert!(
        err.contains("pinning rtmr1 from --expect-measurement"),
        "{err}"
    );
}

/// A debug build with nothing to pin serves (here: reaches the evidence
/// refusal) and warns. The release refusal is the test below.
#[cfg(debug_assertions)]
#[test]
fn a_debug_build_with_nothing_to_pin_warns() {
    if confidential_machine() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let empty = dir.path().join("empty.json");
    std::fs::write(
        &empty,
        format!(
            r#"{{"format":1,"version":"{}","algorithm":"sha384","registers":{{}}}}"#,
            env!("CARGO_PKG_VERSION")
        ),
    )
    .expect("write");
    let (code, err) = infer(&["--measurements", empty.to_str().unwrap()]);
    assert_eq!(code, Some(64), "{err}");
    assert!(
        err.contains("WARNING") && err.contains("no reference measurement"),
        "{err}"
    );
}

/// SECURITY: a release build with nothing to pin refuses before any
/// evidence. Observable only in a release test build (`cargo test --release -p x0ne-infer`).
#[cfg(not(debug_assertions))]
#[test]
fn a_release_build_with_nothing_to_pin_is_exit_87() {
    let dir = tempfile::tempdir().expect("tempdir");
    let empty = dir.path().join("empty.json");
    std::fs::write(
        &empty,
        format!(
            r#"{{"format":1,"version":"{}","algorithm":"sha384","registers":{{}}}}"#,
            env!("CARGO_PKG_VERSION")
        ),
    )
    .expect("write");
    let (code, err) = infer(&["--measurements", empty.to_str().unwrap()]);
    assert_eq!(code, Some(EXIT_UNPINNED), "{err}");
    assert!(err.contains("--expect-measurement"), "{err}");
}

/// While this version's published file names no register or has TEST
/// provenance, a release build off a TEE exits 87 before it looks for
/// evidence, not 64: the "no development bypass" paragraph of docs/running.md
/// must say so.
#[test]
fn the_running_doc_names_the_exit_of_an_unpinned_release_off_a_tee() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace");
    let file = root.join(format!("measurements/{}.json", env!("CARGO_PKG_VERSION")));
    let published: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&file).expect("published measurements"))
            .expect("JSON");
    let pinned = published["registers"]
        .as_object()
        .is_some_and(|r| !r.is_empty());
    if pinned && published["provenance"]["release"] != false {
        return;
    }
    let running = std::fs::read_to_string(root.join("docs/running.md")).expect("docs/running.md");
    let paragraph = running
        .split("\n\n")
        .find(|p| p.contains("has no development bypass"))
        .expect("the \"no development bypass\" paragraph");
    assert!(
        paragraph.contains("87"),
        "{} names no register or has TEST provenance, so a release exits 87 first: {paragraph}",
        file.display()
    );
}

/// `x0ne-infer measurements` refuses what is not an attestation document.
#[test]
fn the_measurements_subcommand_refuses_a_document_without_a_quote() {
    let out = Command::new(env!("CARGO_BIN_EXE_x0ne-infer"))
        .env_remove("LD_LIBRARY_PATH")
        .args(["measurements", FIXTURE])
        .output()
        .expect("the binary runs");
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("tdx_quote_b64"));
    let out = Command::new(env!("CARGO_BIN_EXE_x0ne-infer"))
        .env_remove("LD_LIBRARY_PATH")
        .args(["measurements"])
        .output()
        .expect("the binary runs");
    assert_eq!(out.status.code(), Some(2));
}
