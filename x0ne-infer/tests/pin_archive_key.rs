//! scripts/pin-archive-key.sh on a copy of the files it edits, with keyrings
//! made here (never a real key): it pins a v4 key whose fingerprint is the one
//! given and whose user IDs say neither TEST nor REHEARSAL (unless
//! ALLOW_REHEARSAL_KEY=1), rewrites every "no pin" text, and writes nothing when
//! it refuses. Once this tree pins a key, the script refuses it. The pinned
//! tree's build is checked by rehearsal (`make ci` on a throwaway clone); here,
//! the files it writes pass rustfmt and doc_claims' rule (support/pinned_tree.rs).

use std::path::{Path, PathBuf};
use std::process::Command;
use x0ne_infer::openpgp::build::{primary_block, subkey_block, KeyPair};

#[path = "support/pinned_tree.rs"]
mod pinned_tree;

const EDITED: &[&str] = &[
    "scripts/pin-archive-key.sh",
    "x0ne-infer/src/archive_key.rs",
    "x0ne-engine-cloud/src/up.rs",
    "x0ne-engine-cloud/tests/golden/colors_forced.ansi",
    "docs/cloud.md",
    "docs/attested-channel.md",
    "x0ne-infer/THREAT_MODEL.md",
    "CHANGELOG.md",
    "README.md",
];
const RING: &str = "x0ne-infer/archive-key/x0ne-archive-keyring.gpg";
/// The user ID of the rehearsal key of 2026-10-17 (x0ne-OS docs/archive-key.md).
const REHEARSAL_UID: &str = "x0ne OS archive key REHEARSAL throwaway <rehearsal@x0ne.invalid>";

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn tree() -> tempfile::TempDir {
    let t = tempfile::tempdir().unwrap();
    for rel in EDITED {
        let to = t.path().join(rel);
        std::fs::create_dir_all(to.parent().unwrap()).unwrap();
        std::fs::copy(workspace().join(rel), &to).unwrap();
    }
    t
}

fn snapshot(t: &Path) -> Vec<(String, Option<Vec<u8>>)> {
    EDITED
        .iter()
        .chain([&RING])
        .map(|r| (r.to_string(), std::fs::read(t.join(r)).ok()))
        .collect()
}

/// A certify-only primary and a signing subkey, as the ceremony makes them.
fn keyring(uid: &str, seed: u8) -> (Vec<u8>, String) {
    let primary = KeyPair::from_seed([seed; 32], 1_792_108_800);
    let sub = KeyPair::from_seed([seed ^ 0x5a; 32], 1_792_108_800);
    let mut ring = primary_block(&primary, uid.as_bytes(), 0);
    ring.extend(subkey_block(&primary, &primary, &sub, 0x02, 0, true));
    let fpr: String = primary
        .fingerprint()
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect();
    (ring, fpr)
}

fn pin(t: &Path, ring: &[u8], fpr: &str) -> (i32, String) {
    pin_with(t, ring, fpr, None)
}

/// `allow`: the value of ALLOW_REHEARSAL_KEY, unset when None.
fn pin_with(t: &Path, ring: &[u8], fpr: &str, allow: Option<&str>) -> (i32, String) {
    let file = t.join("keyring.gpg");
    std::fs::write(&file, ring).unwrap();
    let mut cmd = Command::new("bash");
    cmd.arg(t.join("scripts/pin-archive-key.sh"))
        .arg(&file)
        .arg(fpr)
        .env("PIN_DATE", "2026-10-17")
        .env_remove("ALLOW_REHEARSAL_KEY");
    if let Some(v) = allow {
        cmd.env("ALLOW_REHEARSAL_KEY", v);
    }
    let out = cmd.output().expect("bash");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code().unwrap_or(-1), text)
}

/// archive_key::RELEASE is Some: the script runs once, at the ceremony.
fn already_pinned() -> bool {
    !std::fs::read_to_string(workspace().join("x0ne-infer/src/archive_key.rs"))
        .unwrap()
        .contains("pub const RELEASE: Option<Pin> = None;")
}

fn log_pinned() -> bool {
    !std::fs::read_to_string(workspace().join("x0ne-infer/src/transparency.rs"))
        .unwrap()
        .contains("pub const RELEASE: Option<Pin> = None;")
}

/// CHANGELOG.md split before its first released version: (Unreleased, the rest).
fn changelog_parts(text: &str) -> (String, String) {
    let at = text
        .find("\n## 0.")
        .expect("CHANGELOG.md: a released version");
    (text[..at].to_string(), text[at..].to_string())
}

/// The first paragraph of `text` that names `what`, flattened.
fn first_paragraph_naming(text: &str, what: &str) -> String {
    text.split("\n\n")
        .map(flat)
        .find(|p| p.contains(what))
        .unwrap_or_default()
}

fn flat(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[test]
fn the_script_pins_the_ceremony_key_and_rewrites_every_no_pin_text() {
    let t = tree();
    let (ring, fpr) = keyring("x0ne archive key (fixture) <archive@invalid>", 7);
    if already_pinned() {
        let before = snapshot(t.path());
        let (code, out) = pin(t.path(), &ring, &fpr);
        assert_eq!(code, 1, "{out}");
        assert!(out.contains("already pins a key"), "{out}");
        assert_eq!(snapshot(t.path()), before);
        return;
    }
    let (code, out) = pin(t.path(), &ring, &fpr.to_lowercase());
    assert_eq!(code, 0, "{out}");
    let read = |r: &str| std::fs::read_to_string(t.path().join(r)).unwrap();
    assert_eq!(std::fs::read(t.path().join(RING)).unwrap(), ring);
    let key = read("x0ne-infer/src/archive_key.rs");
    assert!(key.contains(&format!(
        "pub const RELEASE: Option<Pin> = Some(Pin {{\n    primary_fingerprint: \"{fpr}\",\n    \
         keyring: include_bytes!(\"../archive-key/x0ne-archive-keyring.gpg\"),\n}});"
    )));
    assert!(!key.contains("fn no_release_key_refuses_everything"));
    assert!(key.contains("fn the_release_pin_is_the_ceremony_key"));
    assert!(
        key.contains("built before the signing ceremony"),
        "NO_KEY stays"
    );
    let up = read("x0ne-engine-cloud/src/up.rs");
    let limits = &up[up.find("pub const LIMITS").unwrap()..];
    let limits = &limits[..limits.find("];").unwrap()];
    assert!(!limits.contains("signing ceremony"), "{limits}");
    assert!(limits.contains(&format!("pinned in its build, primary {fpr}; \\\n")));
    let golden = read("x0ne-engine-cloud/tests/golden/colors_forced.ansi");
    assert!(golden.contains(&format!(
        "pinned in its build, primary {fpr}; whoever holds"
    )));
    for rel in [
        "docs/cloud.md",
        "docs/attested-channel.md",
        "x0ne-infer/THREAT_MODEL.md",
        "README.md",
    ] {
        let text = flat(&read(rel));
        assert!(!text.contains("pins no archive key"), "{rel}");
        assert!(text.contains(&fpr.to_lowercase()), "{rel}");
    }
    // Released CHANGELOG entries are history: the pin is a new Unreleased
    // entry, and every released section stays byte for byte.
    let (unreleased, released) = changelog_parts(&read("CHANGELOG.md"));
    let (_, released_before) =
        changelog_parts(&std::fs::read_to_string(workspace().join("CHANGELOG.md")).unwrap());
    assert_eq!(released, released_before, "a released entry was rewritten");
    let unreleased = flat(&unreleased);
    assert!(unreleased.contains(&fpr.to_lowercase()), "{unreleased}");
    assert!(unreleased.contains("pin-archive-key.sh"), "{unreleased}");
    // What `make ci` holds the pinned tree to: doc_claims' rule, and rustfmt.
    if let Err(e) = pinned_tree::archive_key_status(&read) {
        panic!("{e}");
    }
    for rel in [
        "x0ne-infer/src/archive_key.rs",
        "x0ne-engine-cloud/src/up.rs",
    ] {
        if let Err(e) = pinned_tree::rustfmt_clean(&t.path().join(rel)) {
            panic!("{e}");
        }
    }
    // The log is pinned apart: while it is not, each rewritten status line
    // still says that a release relay refuses every reference with 100.
    if !log_pinned() {
        for rel in ["README.md", "docs/cloud.md", "docs/attested-channel.md"] {
            let status = first_paragraph_naming(&read(rel), "archive key");
            assert!(
                status.contains("no transparency log") && status.contains("100"),
                "{rel}: the pinned status hides the 100 of the missing log: {status}"
            );
        }
    }
    // Twice is refused: the tree pins a key now.
    let before = snapshot(t.path());
    let (code, out) = pin(t.path(), &ring, &fpr);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("already pins a key"), "{out}");
    assert_eq!(snapshot(t.path()), before);
}

#[test]
fn the_script_refuses_and_writes_nothing() {
    let (good, fpr) = keyring("x0ne archive key (fixture) <archive@invalid>", 7);
    let (test, test_fpr) = keyring("x0ne TEST archive key <test@invalid>", 9);
    let (rehearsal, rehearsal_fpr) = keyring(REHEARSAL_UID, 11);
    let (lower, lower_fpr) = keyring("x0ne archive key (rehearsal) <archive@invalid>", 13);
    let mut armored = b"-----BEGIN PGP PUBLIC KEY BLOCK-----\n".to_vec();
    armored.extend_from_slice(&good);
    let mut secret = good.clone();
    secret[0] = 0xC5; // a secret-key packet in place of the primary
    let cases: [(&[u8], &str, &str); 8] = [
        (&test, &test_fpr, "says TEST"),
        (&rehearsal, &rehearsal_fpr, "says REHEARSAL"),
        (&lower, &lower_fpr, "says REHEARSAL"),
        (&good, &test_fpr, "its primary is"),
        (&good, "0123", "40 hex digits"),
        (&armored, &fpr, "ASCII-armored"),
        (&secret, &fpr, "secret key material"),
        (b"", &fpr, "is empty"),
    ];
    for (ring, f, why) in cases {
        let t = tree();
        let before = snapshot(t.path());
        let (code, out) = pin(t.path(), ring, f);
        assert_eq!(code, 1, "{why}: {out}");
        assert!(
            out.contains(why) && out.contains("nothing written"),
            "{why}: {out}"
        );
        assert_eq!(snapshot(t.path()), before, "{why}");
    }
    // A text it must rewrite is gone: nothing is written, not even the keyring.
    if already_pinned() {
        return;
    }
    let t = tree();
    let tm = t.path().join("x0ne-infer/THREAT_MODEL.md");
    let text = std::fs::read_to_string(&tm).unwrap();
    std::fs::write(
        &tm,
        text.replace("This source tree pins no archive key", "x"),
    )
    .unwrap();
    let before = snapshot(t.path());
    let (code, out) = pin(t.path(), &good, &fpr);
    assert_eq!(code, 1, "{out}");
    assert!(
        out.contains("x0ne-infer/THREAT_MODEL.md: the text to replace is there 0 times"),
        "{out}"
    );
    assert_eq!(snapshot(t.path()), before);
}

/// A rehearsal pins its throwaway key only with ALLOW_REHEARSAL_KEY=1, and says
/// so; any other value is refused.
#[test]
fn a_rehearsal_key_needs_allow_rehearsal_key() {
    let (ring, fpr) = keyring(REHEARSAL_UID, 11);
    for v in ["", "0", "yes"] {
        let t = tree();
        let before = snapshot(t.path());
        let (code, out) = pin_with(t.path(), &ring, &fpr, Some(v));
        assert_eq!(code, 1, "ALLOW_REHEARSAL_KEY={v:?}: {out}");
        assert!(out.contains("says REHEARSAL"), "{v:?}: {out}");
        assert_eq!(snapshot(t.path()), before, "{v:?}");
    }
    if already_pinned() {
        return;
    }
    let t = tree();
    let (code, out) = pin_with(t.path(), &ring, &fpr, Some("1"));
    assert_eq!(code, 0, "{out}");
    assert!(
        out.contains("ALLOW_REHEARSAL_KEY=1: REHEARSAL key pinned"),
        "{out}"
    );
    assert_eq!(std::fs::read(t.path().join(RING)).unwrap(), ring);
}
