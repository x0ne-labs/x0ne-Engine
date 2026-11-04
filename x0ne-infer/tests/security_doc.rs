//! The audit ignore list and the document that justifies it must agree.
//!
//! `cargo audit` is the first command many readers of a security project
//! run. Silencing an advisory is sometimes right — RUSTSEC-2023-0071 has no
//! fixed upgrade and does not apply to public-key verification — but an
//! ignore whose reason has been forgotten is worse than a red audit: it is a
//! red audit that stopped being read.
//!
//! So the ignore list is not allowed to drift away from SECURITY.md. This
//! file is a test rather than a comment because a doc-comment describing the
//! rule is exactly what this repository has repeatedly mistaken for enforcing
//! it.

use std::path::{Path, PathBuf};

/// `.cargo/audit.toml` sits at the workspace root in the monorepo and at the
/// tree root in the published crate. Both are one or two levels above this
/// crate's manifest, and the test must work in either.
fn find(relative: &str) -> Option<PathBuf> {
    let mut dir: &Path = Path::new(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        let candidate = dir.join(relative);
        if candidate.exists() {
            return Some(candidate);
        }
        dir = dir.parent()?;
    }
    None
}

/// Every advisory silenced in `.cargo/audit.toml` is explained in SECURITY.md.
///
/// A failure means either an ignore was added without writing down why, or
/// the paragraph explaining one was deleted while the ignore stayed. Both
/// leave a reader unable to tell a considered exemption from an oversight,
/// which is the only thing that makes the exemption acceptable at all.
#[test]
fn every_silenced_advisory_is_justified_in_the_security_policy() {
    let Some(audit) = find(".cargo/audit.toml") else {
        panic!("no .cargo/audit.toml found next to this crate — if the ignore list moved, this test must move with it");
    };
    let policy_path = find("SECURITY.md")
        .or_else(|| find("x0ne-infer/SECURITY.md"))
        .expect("SECURITY.md must ship beside the code it describes");
    let policy = std::fs::read_to_string(&policy_path).expect("SECURITY.md is readable");
    let mut text = std::fs::read_to_string(&audit).expect("audit.toml is readable");
    // deny.toml carries the same list for `cargo deny`; its ignores are held
    // to the same rule.
    if let Some(deny) = find("deny.toml") {
        text.push('\n');
        text.push_str(&std::fs::read_to_string(deny).expect("deny.toml is readable"));
    }

    // Every RUSTSEC id that appears OUTSIDE a comment line is an active
    // ignore. Ids inside comments are prose about ones already handled.
    let mut silenced: Vec<String> = Vec::new();
    for line in text.lines() {
        let code = line.split('#').next().unwrap_or("");
        let mut rest = code;
        while let Some(at) = rest.find("RUSTSEC-") {
            let id: String = rest[at..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '-')
                .collect();
            silenced.push(id);
            rest = &rest[at + 8..];
        }
    }
    assert!(
        !silenced.is_empty(),
        "this test proves nothing if nothing is silenced; if the list is now \
         empty, delete the test with the last ignore"
    );

    let missing: Vec<&String> = silenced.iter().filter(|id| !policy.contains(*id)).collect();
    assert!(
        missing.is_empty(),
        "silenced in {} but not explained in {}: {missing:?}",
        audit.display(),
        policy_path.display()
    );
}

/// The justification has to say what would make the exemption wrong again.
///
/// "Not applicable" is an assertion about today's code. Without the condition
/// that would end it, the next person to add an RSA private key has nothing
/// to trip over — and the ignore quietly becomes false.
#[test]
fn the_justification_states_what_would_make_it_expire() {
    let policy_path = find("SECURITY.md")
        .or_else(|| find("x0ne-infer/SECURITY.md"))
        .expect("SECURITY.md exists");
    let policy = std::fs::read_to_string(policy_path).expect("readable");
    let section = policy
        .split("## Known advisories")
        .nth(1)
        .expect("SECURITY.md must carry a Known advisories section");
    assert!(
        section.contains("What would make this live again"),
        "each exemption must name its own expiry condition"
    );
}

/// RUSTSEC-2026-0367/0368/0369/0429 (zip extraction) are ignored because nothing here opens a zip.
/// Measured 2026-10-21: zip comes from candle-core 0.11.0 alone, no workspace source names it.
/// A failure means that measurement no longer holds: re-measure, then fix or drop the ignores.
#[test]
fn the_zip_ignores_hold_only_while_nothing_opens_an_archive() {
    let lock_path = find("Cargo.lock").expect("Cargo.lock at the workspace root");
    let lock: toml::Value =
        toml::from_str(&std::fs::read_to_string(&lock_path).expect("Cargo.lock is readable"))
            .expect("Cargo.lock parses");
    let mut through: Vec<String> = Vec::new();
    for pkg in lock["package"].as_array().expect("a [[package]] list") {
        let deps = pkg.get("dependencies").and_then(|d| d.as_array());
        let needs_zip = deps.is_some_and(|d| {
            d.iter()
                .any(|d| d.as_str().and_then(|d| d.split(' ').next()) == Some("zip"))
        });
        if needs_zip {
            through.push(format!(
                "{} {}",
                pkg["name"].as_str().expect("name"),
                pkg["version"].as_str().expect("version")
            ));
        }
    }
    assert_eq!(
        through,
        ["candle-core 0.11.0"],
        "zip now enters through {through:?}: re-measure what calls it (SECURITY.md)"
    );

    let root = lock_path.parent().expect("the workspace root");
    let manifest: toml::Value = toml::from_str(
        &std::fs::read_to_string(root.join("Cargo.toml")).expect("Cargo.toml is readable"),
    )
    .expect("Cargo.toml parses");
    let mut files = Vec::new();
    for member in manifest["workspace"]["members"]
        .as_array()
        .expect("workspace members")
    {
        rs_files(
            &root.join(member.as_str().expect("member path")),
            &mut files,
        );
    }
    assert!(
        !files.is_empty(),
        "no source found under {}",
        root.display()
    );
    const NAMES: [&str; 13] = [
        "zip::",
        "ZipArchive",
        "ZipStreamReader",
        "ZipWriter",
        "npy::",
        "pickle::",
        "from_npz",
        "from_pth",
        "read_npz",
        "read_pth",
        "write_npz",
        "NpzTensors",
        "PthTensors",
    ];
    let mut hits = Vec::new();
    for f in files
        .iter()
        .filter(|f| !f.ends_with("tests/security_doc.rs"))
    {
        let text = std::fs::read_to_string(f).expect("source is readable");
        hits.extend(
            NAMES
                .iter()
                .filter(|n| text.contains(*n))
                .map(|n| format!("{}: {n}", f.display())),
        );
    }
    assert!(
        hits.is_empty(),
        "the zip ignores assume no code opens an archive; these sources name zip or \
         candle's npz/pth loaders: {hits:?}"
    );
}

/// Every `.rs` under `dir`, hidden directories and `target` left out.
fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let entry = entry.expect("directory entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || name == "target" {
            continue;
        }
        let path = entry.path();
        if entry.file_type().expect("file type").is_dir() {
            rs_files(&path, out);
        } else if std::path::Path::new(&name)
            .extension()
            .is_some_and(|e| e == "rs")
        {
            out.push(path);
        }
    }
}

/// Active `RUSTSEC-` ids of a TOML file, with the text of their line.
fn ignores(toml: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in toml.lines() {
        let code = line.split('#').next().unwrap_or("");
        if let Some(at) = code.find("RUSTSEC-") {
            let id: String = code[at..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '-')
                .collect();
            out.push((id, code.to_string()));
        }
    }
    out
}

/// cargo-deny (the CI job) and cargo audit (what readers run) ignore the same
/// advisories, except where both deny.toml's reason and SECURITY.md say that
/// cargo audit does not; deny.toml's header then may not claim they agree.
#[test]
fn the_deny_and_audit_ignore_lists_differ_only_where_said() {
    let deny = std::fs::read_to_string(find("deny.toml").expect("deny.toml")).unwrap();
    let audit = std::fs::read_to_string(find(".cargo/audit.toml").expect("audit.toml")).unwrap();
    let policy = std::fs::read_to_string(find("SECURITY.md").expect("SECURITY.md")).unwrap();
    let denied = ignores(&deny);
    let audited: Vec<String> = ignores(&audit).into_iter().map(|(id, _)| id).collect();
    for id in &audited {
        assert!(
            denied.iter().any(|(d, _)| d == id),
            "{id} is ignored by cargo audit but not by cargo deny"
        );
    }
    let deny_only: Vec<&(String, String)> = denied
        .iter()
        .filter(|(id, _)| !audited.contains(id))
        .collect();
    for (id, line) in &deny_only {
        assert!(
            line.contains("not ignored by cargo audit"),
            "deny.toml ignores {id} alone and its reason does not say so: {line}"
        );
        let at = policy
            .find(&format!("**{id}"))
            .expect("SECURITY.md paragraph");
        let paragraph = &policy[at..policy[at..].find("\n\n").map_or(policy.len(), |e| at + e)];
        assert!(
            paragraph.contains("Not\nignored by `cargo audit`")
                || paragraph.contains("Not ignored by `cargo audit`"),
            "SECURITY.md does not say cargo audit keeps {id}"
        );
    }
    let header: String = deny
        .lines()
        .take_while(|l| l.starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        deny_only.is_empty() || !header.contains("is also ignored in .cargo/audit.toml"),
        "deny.toml's header says every ignore is also in .cargo/audit.toml; {:?} are not",
        deny_only.iter().map(|(id, _)| id).collect::<Vec<_>>()
    );
}
