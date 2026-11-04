//! scripts/pin-transparency-log.sh on a copy of the files it edits, with log
//! and witness keys made here (never a real key): it pins the log a pin file
//! names, after the archive key, rewrites every "no log" text, and writes
//! nothing when it refuses. Once this tree pins a log, the script refuses it.
//! The pinned tree's build is checked by rehearsal (`make ci` on a throwaway
//! clone); here the values it writes pass `transparency::pinned`, and the
//! file passes rustfmt (support/pinned_tree.rs).

use ed25519_dalek::SigningKey;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;
use x0ne_infer::openpgp::build::{primary_block, subkey_block, KeyPair};
use x0ne_infer::tlog::{KeyKind, NoteKey};
use x0ne_infer::transparency::{pinned, Pin};

#[path = "support/pinned_tree.rs"]
mod pinned_tree;

const EDITED: &[&str] = &[
    "scripts/pin-transparency-log.sh",
    "scripts/pin-archive-key.sh",
    "x0ne-infer/src/archive_key.rs",
    "x0ne-infer/src/transparency.rs",
    "x0ne-engine-cloud/src/up.rs",
    "x0ne-engine-cloud/tests/golden/colors_forced.ansi",
    "docs/cloud.md",
    "docs/attested-channel.md",
    "docs/pinning.md",
    "x0ne-infer/THREAT_MODEL.md",
    "CHANGELOG.md",
    "README.md",
];
const RING: &str = "x0ne-infer/archive-key/x0ne-archive-keyring.gpg";
const ORIGIN: &str = "log.x0ne.example/engine/image-references";
const TILES: &str = "https://log.x0ne.example/engine/";

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn read_ws(rel: &str) -> String {
    std::fs::read_to_string(workspace().join(rel)).unwrap()
}

fn pins_none(rel: &str) -> bool {
    read_ws(rel).contains("pub const RELEASE: Option<Pin> = None;")
}

/// The files the script edits, copied; the archive key pinned first, as at
/// the ceremony (by pin-archive-key.sh, unless this tree already pins one).
fn tree() -> tempfile::TempDir {
    let t = bare_tree();
    if pins_none("x0ne-infer/src/archive_key.rs") {
        let primary = KeyPair::from_seed([7; 32], 1_792_108_800);
        let sub = KeyPair::from_seed([7 ^ 0x5a; 32], 1_792_108_800);
        let mut ring = primary_block(&primary, b"x0ne archive key (fixture) <a@invalid>", 0);
        ring.extend(subkey_block(&primary, &primary, &sub, 0x02, 0, true));
        let fpr: String = primary
            .fingerprint()
            .iter()
            .map(|b| format!("{b:02X}"))
            .collect();
        let file = t.path().join("keyring.gpg");
        std::fs::write(&file, ring).unwrap();
        let out = Command::new("bash")
            .arg(t.path().join("scripts/pin-archive-key.sh"))
            .arg(&file)
            .arg(&fpr)
            .env("PIN_DATE", "2026-10-24")
            .env_remove("ALLOW_REHEARSAL_KEY")
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        std::fs::remove_file(&file).unwrap();
    }
    t
}

/// The files copied as they are in this tree.
fn bare_tree() -> tempfile::TempDir {
    let t = tempfile::tempdir().unwrap();
    for rel in EDITED.iter().chain([&RING]) {
        let from = workspace().join(rel);
        if !from.exists() {
            continue;
        }
        let to = t.path().join(rel);
        std::fs::create_dir_all(to.parent().unwrap()).unwrap();
        std::fs::copy(from, &to).unwrap();
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

fn vkey(name: &str, kind: KeyKind, seed: u8) -> String {
    let pk = SigningKey::from_bytes(&[seed; 32])
        .verifying_key()
        .to_bytes();
    NoteKey::new(name, kind, pk).unwrap().vkey()
}

fn log_vkey() -> String {
    vkey(ORIGIN, KeyKind::Ed25519, 1)
}

fn witnesses() -> Vec<String> {
    vec![
        vkey("witness.one.example", KeyKind::CosignatureV1, 2),
        vkey("witness.two.example", KeyKind::CosignatureV1, 3),
    ]
}

/// A pin file: origin, log, witnesses, quorum, max-age, tiles.
fn pin_file(
    origin: &str,
    log: &str,
    ws: &[String],
    quorum: &str,
    age: &str,
    tiles: &str,
) -> String {
    let mut s = format!("# the log of image references\norigin {origin}\nlog {log}\n");
    for w in ws {
        s.push_str(&format!("witness {w}\n"));
    }
    s.push_str(&format!("quorum {quorum}\nmax-age {age}\ntiles {tiles}\n"));
    s
}

fn good_pin() -> String {
    pin_file(ORIGIN, &log_vkey(), &witnesses(), "1", "604800", TILES)
}

/// `allow`: the value of ALLOW_REHEARSAL_LOG, unset when None.
fn pin_with(t: &Path, file: &str, allow: Option<&str>) -> (i32, String) {
    let p = t.join("log.pin");
    std::fs::write(&p, file).unwrap();
    let mut cmd = Command::new("bash");
    cmd.arg(t.join("scripts/pin-transparency-log.sh"))
        .arg(&p)
        .env("PIN_DATE", "2026-10-25")
        .env_remove("ALLOW_REHEARSAL_LOG");
    if let Some(v) = allow {
        cmd.env("ALLOW_REHEARSAL_LOG", v);
    }
    let out = cmd.output().expect("bash");
    std::fs::remove_file(&p).unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code().unwrap_or(-1), text)
}

fn pin(t: &Path, file: &str) -> (i32, String) {
    pin_with(t, file, None)
}

fn flat(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// The first paragraph of `text` that names `what`, flattened.
fn first_paragraph_naming(text: &str, what: &str) -> String {
    text.split("\n\n")
        .map(flat)
        .find(|p| p.contains(what))
        .unwrap_or_default()
}

/// CHANGELOG.md split before its first released version: (Unreleased, the rest).
fn changelog_parts(text: &str) -> (String, String) {
    let at = text
        .find("\n## 0.")
        .expect("CHANGELOG.md: a released version");
    (text[..at].to_string(), text[at..].to_string())
}

/// The fingerprint the script prints for a key: SHA-256 of the vkey line,
/// first 20 bytes, in 10 groups of 4 hex digits, as written on paper.
fn fingerprint(vkey: &str) -> String {
    let d = Sha256::digest(vkey.as_bytes());
    d[..20]
        .chunks(2)
        .map(|c| format!("{:02X}{:02X}", c[0], c[1]))
        .collect::<Vec<_>>()
        .join(" ")
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

#[test]
fn the_script_pins_the_log_and_rewrites_every_no_log_text() {
    let t = tree();
    if !pins_none("x0ne-infer/src/transparency.rs") {
        let before = snapshot(t.path());
        let (code, out) = pin(t.path(), &good_pin());
        assert_eq!(code, 1, "{out}");
        assert!(out.contains("already pins a log"), "{out}");
        assert_eq!(snapshot(t.path()), before);
        return;
    }
    let (_, released_before) = changelog_parts(&read_ws("CHANGELOG.md"));
    let before = snapshot(t.path());
    let (code, out) = pin(t.path(), &good_pin());
    assert_eq!(code, 0, "{out}");
    // The files it names are the files it changed, no other: the list is
    // read against `git diff --stat` at the ceremony.
    let named: Vec<&str> = out
        .lines()
        .filter_map(|l| l.strip_prefix("pinned: "))
        .collect();
    let changed: Vec<String> = snapshot(t.path())
        .into_iter()
        .zip(before)
        .filter(|(after, before)| after != before)
        .map(|(after, _)| after.0)
        .collect();
    let mut changed: Vec<&str> = changed.iter().map(String::as_str).collect();
    changed.sort_unstable();
    assert_eq!(named, changed, "{out}");
    let read = |r: &str| std::fs::read_to_string(t.path().join(r)).unwrap();
    let (log, ws) = (log_vkey(), witnesses());
    let src = read("x0ne-infer/src/transparency.rs");
    assert!(
        src.contains(&format!(
            "pub const RELEASE: Option<Pin> = Some(Pin {{\n    origin: \"{ORIGIN}\",\n    \
             log_key:\n        \"{log}\",\n    witnesses: &[\n        \"{}\",\n        \"{}\",\n    ],\n    \
             quorum: 1,\n    max_age_s: 604800,\n    tiles: \"{TILES}\",\n}});",
            ws[0], ws[1]
        )),
        "{src}"
    );
    if let Err(e) = pinned_tree::rustfmt_clean(&t.path().join("x0ne-infer/src/transparency.rs")) {
        panic!("{e}");
    }
    assert!(!src.contains("fn no_release_log_refuses_everything_and_pins_are_checked"));
    assert!(src.contains("fn the_release_pin_is_the_ceremony_log"));
    assert!(
        src.contains("pins no transparency log: no public log"),
        "NO_LOG stays"
    );
    // The values written are a pin the relay takes.
    let p = Pin {
        origin: ORIGIN,
        log_key: leak(log.clone()),
        witnesses: Box::leak(vec![leak(ws[0].clone()), leak(ws[1].clone())].into_boxed_slice()),
        quorum: 1,
        max_age_s: 604_800,
        tiles: TILES,
    };
    let policy = pinned(&p).unwrap();
    assert_eq!(
        (policy.origin.as_str(), policy.witnesses.len()),
        (ORIGIN, 2)
    );
    for rel in [
        "README.md",
        "docs/cloud.md",
        "docs/attested-channel.md",
        "docs/pinning.md",
        "x0ne-infer/THREAT_MODEL.md",
    ] {
        let text = flat(&read(rel));
        for gone in [
            "this source tree pins no log",
            "this tree pins no log",
            "pins no transparency log",
            "no transparency log (exit 100)",
            "x0ne runs no public log",
            "x0ne runs no log",
        ] {
            assert!(!text.contains(gone), "{rel}: {gone}");
        }
        assert!(text.contains(ORIGIN), "{rel}");
    }
    for rel in ["README.md", "docs/cloud.md", "docs/attested-channel.md"] {
        let status = first_paragraph_naming(&read(rel), "archive key");
        // "h100" stays: the refusal's exit code goes.
        for refusal in ["exit 100", "with 100"] {
            assert!(!status.contains(refusal), "{rel}: {status}");
        }
        assert!(status.contains(ORIGIN), "{rel}: {status}");
    }
    let cloud = read("docs/cloud.md");
    for k in [&log, &ws[0], &ws[1]] {
        assert!(cloud.contains(k.as_str()), "docs/cloud.md names {k}");
    }
    let (unreleased, released) = changelog_parts(&read("CHANGELOG.md"));
    assert_eq!(released, released_before, "a released entry was rewritten");
    let unreleased = flat(&unreleased);
    assert!(unreleased.contains(ORIGIN), "{unreleased}");
    assert!(
        unreleased.contains("pin-transparency-log.sh"),
        "{unreleased}"
    );
    assert!(
        !unreleased.contains("this tree pins no log"),
        "{unreleased}"
    );
    // What the key holder compares with the paper of the key generation.
    assert!(
        out.contains(&format!("log key fingerprint {}", fingerprint(&log))),
        "{out}"
    );
    // Twice is refused: the tree pins a log now.
    let before = snapshot(t.path());
    let (code, out) = pin(t.path(), &good_pin());
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("already pins a log"), "{out}");
    assert_eq!(snapshot(t.path()), before);
}

#[test]
fn the_script_refuses_and_writes_nothing() {
    let (log, ws) = (log_vkey(), witnesses());
    let other_named = vkey("other.example/log", KeyKind::Ed25519, 1);
    let cosig_log = vkey(ORIGIN, KeyKind::CosignatureV1, 1);
    let ed_witness = vkey("witness.three.example", KeyKind::Ed25519, 4);
    let same_key = vkey("witness.alias.example", KeyKind::CosignatureV1, 2);
    let bad_id = {
        let (name, rest) = log.split_once('+').unwrap();
        let (id, key) = rest.split_once('+').unwrap();
        let flipped = if id.starts_with('0') { "1" } else { "0" };
        format!("{name}+{flipped}{}+{key}", &id[1..])
    };
    let test_origin = "test.invalid/x0ne-engine/tlog";
    let test_log = vkey(test_origin, KeyKind::Ed25519, 1);
    let rehearsal_origin = "rehearsal.x0ne.example/engine";
    let rehearsal_log = vkey(rehearsal_origin, KeyKind::Ed25519, 1);
    let one = |w: &str| vec![w.to_string()];
    let cases: Vec<(String, &str)> = vec![
        (
            pin_file(ORIGIN, &other_named, &ws, "1", "604800", TILES),
            "named",
        ),
        (
            pin_file(ORIGIN, &cosig_log, &ws, "1", "604800", TILES),
            "Ed25519",
        ),
        (
            pin_file(ORIGIN, &log, &one(&ed_witness), "1", "604800", TILES),
            "cosignature/v1",
        ),
        (
            pin_file(
                ORIGIN,
                &log,
                &[ws[0].clone(), ws[0].clone()],
                "1",
                "604800",
                TILES,
            ),
            "twice",
        ),
        (
            pin_file(
                ORIGIN,
                &log,
                &[ws[0].clone(), same_key],
                "1",
                "604800",
                TILES,
            ),
            "same key",
        ),
        (
            pin_file(ORIGIN, &log, &[], "1", "604800", TILES),
            "no witness",
        ),
        (pin_file(ORIGIN, &log, &ws, "0", "604800", TILES), "quorum"),
        (pin_file(ORIGIN, &log, &ws, "3", "604800", TILES), "quorum"),
        (pin_file(ORIGIN, &log, &ws, "1", "0", TILES), "max-age"),
        (
            pin_file(ORIGIN, &log, &ws, "1", "31536001", TILES),
            "max-age",
        ),
        (
            pin_file(ORIGIN, &log, &ws, "1", "604800", "http://log.x0ne.example/"),
            "https",
        ),
        (
            pin_file(
                ORIGIN,
                &log,
                &ws,
                "1",
                "604800",
                "https://u:p@log.x0ne.example/",
            ),
            "https",
        ),
        (
            pin_file(ORIGIN, &bad_id, &ws, "1", "604800", TILES),
            "key ID",
        ),
        (
            pin_file(test_origin, &test_log, &ws, "1", "604800", TILES),
            "TEST",
        ),
        (
            pin_file(rehearsal_origin, &rehearsal_log, &ws, "1", "604800", TILES),
            "REHEARSAL",
        ),
        (format!("{}tiles {TILES}\n", good_pin()), "twice"),
        (format!("{}colour blue\n", good_pin()), "unknown line"),
        (
            good_pin().replace(&format!("tiles {TILES}\n"), ""),
            "no tiles",
        ),
    ];
    for (file, why) in &cases {
        let t = tree();
        let before = snapshot(t.path());
        let (code, out) = pin(t.path(), file);
        assert_eq!(code, 1, "{why}: {out}");
        assert!(
            out.contains(why) && out.contains("nothing written"),
            "{why}: {out}"
        );
        assert_eq!(snapshot(t.path()), before, "{why}");
    }
    if !pins_none("x0ne-infer/src/transparency.rs") {
        return;
    }
    // The order of the ceremony: the archive key first.
    if pins_none("x0ne-infer/src/archive_key.rs") {
        let t = bare_tree();
        let before = snapshot(t.path());
        let (code, out) = pin(t.path(), &good_pin());
        assert_eq!(code, 1, "{out}");
        assert!(out.contains("pin the archive key first"), "{out}");
        assert_eq!(snapshot(t.path()), before);
    }
    // A text it must rewrite is gone: nothing is written.
    let t = tree();
    let tm = t.path().join("docs/pinning.md");
    let text = std::fs::read_to_string(&tm).unwrap();
    std::fs::write(&tm, text.replace("This tree pins no log:", "x")).unwrap();
    let before = snapshot(t.path());
    let (code, out) = pin(t.path(), &good_pin());
    assert_eq!(code, 1, "{out}");
    assert!(
        out.contains("docs/pinning.md: the text to replace is there 0 times"),
        "{out}"
    );
    assert_eq!(snapshot(t.path()), before);
}

/// A rehearsal pins its throwaway log only with ALLOW_REHEARSAL_LOG=1, and
/// says so; any other value is refused.
#[test]
fn a_rehearsal_log_needs_allow_rehearsal_log() {
    let origin = "rehearsal.x0ne.example/engine";
    let file = pin_file(
        origin,
        &vkey(origin, KeyKind::Ed25519, 5),
        &[vkey("witness.REHEARSAL.example", KeyKind::CosignatureV1, 6)],
        "1",
        "604800",
        TILES,
    );
    for v in ["", "0", "yes"] {
        let t = tree();
        let before = snapshot(t.path());
        let (code, out) = pin_with(t.path(), &file, Some(v));
        assert_eq!(code, 1, "ALLOW_REHEARSAL_LOG={v:?}: {out}");
        assert!(out.contains("says REHEARSAL"), "{v:?}: {out}");
        assert_eq!(snapshot(t.path()), before, "{v:?}");
    }
    if !pins_none("x0ne-infer/src/transparency.rs") {
        return;
    }
    let t = tree();
    let (code, out) = pin_with(t.path(), &file, Some("1"));
    assert_eq!(code, 0, "{out}");
    assert!(
        out.contains("ALLOW_REHEARSAL_LOG=1: REHEARSAL log pinned"),
        "{out}"
    );
}

/// rustfmt keeps a field and its string on one line up to 100 columns, and
/// puts the string on the next line past that: `make ci` checks the pinned
/// tree with `cargo fmt --check`. Origins of 30 and 31 characters put the
/// `log_key` line at 100 and 101 columns.
#[test]
fn the_pin_is_written_as_rustfmt_writes_it() {
    if !pins_none("x0ne-infer/src/transparency.rs") {
        return;
    }
    for origin in [
        "log.x0ne.example/engine/tlog-a",
        "log.x0ne.example/engine/tlog-ab",
    ] {
        let t = tree();
        let log = vkey(origin, KeyKind::Ed25519, 1);
        let file = pin_file(origin, &log, &witnesses(), "2", "604800", TILES);
        let (code, out) = pin(t.path(), &file);
        assert_eq!(code, 0, "{out}");
        let src = std::fs::read_to_string(t.path().join("x0ne-infer/src/transparency.rs")).unwrap();
        let one_line = format!("    log_key: \"{log}\",\n");
        assert_eq!(src.contains(&one_line), one_line.len() <= 101, "{origin}");
        if let Err(e) = pinned_tree::rustfmt_clean(&t.path().join("x0ne-infer/src/transparency.rs"))
        {
            panic!("{origin}: {e}");
        }
    }
}
