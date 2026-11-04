//! Checks shared by doc_claims.rs and the pin scripts' tests, run on this tree
//! or on a pinned copy of it. The rehearsal of 2026-10-24 pinned a copy with
//! both scripts: their tests passed and `make ci` failed on it (fmt, and
//! doc_claims), because each test held its own rule.
#![allow(dead_code)]

use std::path::Path;
use std::process::Command;

/// Whitespace flattened, lower case: prose wraps differently from file to file.
fn flat(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// While no archive key is pinned, docs/cloud.md, THREAT_MODEL.md and
/// CHANGELOG.md say so; once one is, the first two say it no more. CHANGELOG.md
/// keeps its released entries as they were published: the pin is said by its
/// Unreleased part. `read` reads a file of the tree by its relative path.
pub fn archive_key_status(read: &dyn Fn(&str) -> String) -> Result<(), String> {
    const NONE: &str = "this source tree pins no archive key";
    const PINNED: &str = "the x0ne archive key is pinned since";
    let key = read("x0ne-infer/src/archive_key.rs");
    let unpinned = key.contains("pub const RELEASE: Option<Pin> = None;");
    if !unpinned && !key.contains("pub const RELEASE: Option<Pin> = Some(") {
        return Err("archive_key.rs: RELEASE is neither None nor Some(...)".into());
    }
    for rel in ["docs/cloud.md", "x0ne-infer/THREAT_MODEL.md"] {
        if flat(&read(rel)).contains(NONE) != unpinned {
            return Err(format!(
                "{rel}: \"{NONE}\" must be there exactly while archive_key::RELEASE is None"
            ));
        }
    }
    let changelog = read("CHANGELOG.md");
    let at = changelog
        .find("\n## 0.")
        .ok_or("CHANGELOG.md: no released version")?;
    let unreleased = flat(&changelog[..at]);
    if unpinned && !flat(&changelog).contains(NONE) {
        return Err(format!(
            "CHANGELOG.md: \"{NONE}\" is gone while RELEASE is None"
        ));
    }
    if !unpinned && unreleased.contains(NONE) {
        return Err(format!(
            "CHANGELOG.md: its Unreleased part says \"{NONE}\" while RELEASE is Some"
        ));
    }
    if unreleased.contains(PINNED) == unpinned {
        return Err(format!(
            "CHANGELOG.md: its Unreleased part says \"{PINNED}\" exactly while RELEASE is Some"
        ));
    }
    Ok(())
}

/// `rustfmt --check` on one file, with the workspace's edition: what `make ci`
/// (`cargo fmt -- --check`) says of it. rustfmt 1.9 exits 0 on a diff
/// read from stdin, so the file is given by its path.
pub fn rustfmt_clean(file: &Path) -> Result<(), String> {
    let manifest =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../Cargo.toml"))
            .map_err(|e| format!("workspace Cargo.toml: {e}"))?;
    let edition = manifest
        .lines()
        .find_map(|l| l.trim().strip_prefix("edition = \""))
        .and_then(|l| l.strip_suffix('"'))
        .ok_or("workspace Cargo.toml: no edition")?;
    let out = Command::new("rustfmt")
        .args(["--edition", edition, "--check"])
        .arg(file)
        .output()
        .map_err(|e| format!("rustfmt: {e} (make ci runs it in its container)"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{}: not as rustfmt formats it\n{}{}",
            file.display(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}
