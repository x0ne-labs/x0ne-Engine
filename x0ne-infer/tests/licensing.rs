//! The license map of the repository, `LICENSING.md`, held to the manifests
//! and to the license files.
//!
//! The server crates are under AGPL-3.0-only, with an additional permission
//! (section 7) for NVIDIA's libraries, both in the root `LICENSE`. What an
//! application builds in (`x0ne-sdk`, `x0ne-confidential`, `x0ne-ohttp`) is
//! under Apache-2.0, each crate with its own `LICENSE`. x0ne-infer's files
//! that `x0ne-sdk` compiles by `#[path]` are Apache-2.0 too, each saying so
//! on its first line.
//!
//! A failure means that a manifest, a license file and `LICENSING.md` no
//! longer say the same thing: a reader would be told one license and given
//! another, or an application that builds in the SDK would take in code
//! under the AGPL.

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

const AGPL: &str = "AGPL-3.0-only";
const APACHE: &str = "Apache-2.0";

/// The crates an application builds in: Apache-2.0. Every other crate is
/// under AGPL-3.0-only (decision of 2026-11-02).
const APACHE_CRATES: &[&str] = &["x0ne-confidential", "x0ne-ohttp", "x0ne-sdk"];

/// The GNU AGPL version 3 as the FSF publishes it, the root `LICENSE`'s
/// first bytes: Debian's `/usr/share/common-licenses/AGPL-3.0` (base-files),
/// 34,523 bytes, read on 2026-11-02.
const AGPL_TEXT_LEN: usize = 34_523;
const AGPL_TEXT_SHA256: &str = "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0";

/// The Apache License 2.0 with its appendix filled in ("Copyright 2025-2026
/// x0ne"): the root `LICENSE` until 2026-11-02, the text the publication
/// gate checked word for word.
const APACHE_FILE_SHA256: &str = "4525674a90673ce56460081da4c3f1d07b3ead0eb230bfd8994d8330c727dad8";

/// What follows the AGPL's text in the root `LICENSE`.
const PERMISSION_HEADING: &str = "Additional permission under GNU AGPL version 3 section 7";

/// The NVIDIA libraries the permission at the end of `LICENSE` names.
const NVIDIA_LIBRARIES: &[&str] = &["libcudart", "libcublas", "libcublasLt", "libcurand"];

/// The NVIDIA libraries the image carries: `CUDA_LIBS` of the Containerfile,
/// without their `.so.N` suffix (2026-11-04: cuBLAS and cuBLASLt, which x0ne
/// Core opens with dlopen).
fn image_nvidia_libraries() -> Vec<String> {
    let containerfile = read("Containerfile");
    let line = containerfile
        .lines()
        .find_map(|l| l.strip_prefix("ENV CUDA_LIBS=\""))
        .expect("Containerfile: no ENV CUDA_LIBS line");
    line.trim_end_matches('"')
        .split_whitespace()
        .map(|l| l.split(".so").next().unwrap_or(l).to_string())
        .collect()
}

/// `text` names `lib` as a whole name: `libcublasLt` does not name `libcublas`.
fn names_library(text: &str, lib: &str) -> bool {
    text.match_indices(lib)
        .any(|(i, _)| !text[i + lib.len()..].starts_with(|c: char| c.is_ascii_alphanumeric()))
}

/// The first line of a file under Apache-2.0 outside an Apache crate.
const SPDX_APACHE: &str = "// SPDX-License-Identifier: Apache-2.0";

/// Directories no crate lives in: build outputs and tool state.
const SKIPPED_DIRS: &[&str] = &[".git", "target", "out", "vendor", "node_modules"];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("x0ne-infer sits in the repository's root")
        .to_path_buf()
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `a/b/../c` as `a/c`, relative to the repository's root, `/`-separated.
fn normalize(path: &Path) -> String {
    let mut parts: Vec<String> = Vec::new();
    for c in path.components() {
        match c {
            Component::ParentDir => {
                parts
                    .pop()
                    .unwrap_or_else(|| panic!("{} leaves the repository", path.display()));
            }
            Component::Normal(p) => parts.push(p.to_string_lossy().into_owned()),
            Component::CurDir => {}
            other => panic!("{}: unexpected component {other:?}", path.display()),
        }
    }
    parts.join("/")
}

/// `rel` joined to `dir` (both relative to the root), normalized.
fn join(dir: &str, rel: &str) -> String {
    normalize(&Path::new(dir).join(rel))
}

fn manifest(dir: &str) -> toml::Table {
    let rel = join(dir, "Cargo.toml");
    read(&rel).parse().unwrap_or_else(|e| panic!("{rel}: {e}"))
}

struct Crate {
    name: String,
    /// Relative to the root; "" for the root.
    dir: String,
    license: String,
    manifest: toml::Table,
}

/// The `license` of a manifest's `[package]`, from its workspace when it
/// says `license.workspace = true`.
fn license_of(dir: &str, package: &toml::Table, workspace_root: &toml::Table) -> String {
    match package.get("license") {
        Some(toml::Value::String(s)) => s.clone(),
        Some(toml::Value::Table(t)) if t.get("workspace") == Some(&toml::Value::Boolean(true)) => {
            workspace_root
                .get("workspace")
                .and_then(|w| w.get("package"))
                .and_then(|p| p.get("license"))
                .and_then(toml::Value::as_str)
                .unwrap_or_else(|| {
                    panic!("{dir}/Cargo.toml inherits a license its workspace does not set")
                })
                .to_string()
        }
        _ => panic!("{dir}/Cargo.toml declares no license"),
    }
}

fn package_crate(dir: &str, manifest: toml::Table, workspace_root: &toml::Table) -> Crate {
    let package = manifest
        .get("package")
        .and_then(toml::Value::as_table)
        .unwrap_or_else(|| panic!("{dir}/Cargo.toml has no [package]"))
        .clone();
    let name = package["name"].as_str().unwrap().to_string();
    let license = license_of(dir, &package, workspace_root);
    Crate {
        name,
        dir: dir.to_string(),
        license,
        manifest,
    }
}

/// The crates of the workspace at `dir`: its own package if it has one, its
/// members, and the workspaces it excludes, each with its own `[workspace]`.
fn workspace(dir: &str, out: &mut Vec<Crate>) {
    let root_manifest = manifest(dir);
    let ws = root_manifest
        .get("workspace")
        .and_then(toml::Value::as_table)
        .unwrap_or_else(|| panic!("{dir}/Cargo.toml: an excluded crate is a workspace of its own"));
    let list = |key: &str| -> Vec<String> {
        ws.get(key)
            .and_then(toml::Value::as_array)
            .map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect())
            .unwrap_or_default()
    };
    if root_manifest.contains_key("package") {
        out.push(package_crate(dir, root_manifest.clone(), &root_manifest));
    }
    for member in list("members") {
        let d = join(dir, &member);
        out.push(package_crate(&d, manifest(&d), &root_manifest));
    }
    for excluded in list("exclude") {
        workspace(&join(dir, &excluded), out);
    }
}

/// The root and the workspaces it excludes, recursively, as `workspace` walks them.
fn workspace_dirs(dir: &str, out: &mut BTreeSet<String>) {
    out.insert(dir.to_string());
    let excluded: Vec<String> = manifest(dir)
        .get("workspace")
        .and_then(|w| w.get("exclude"))
        .and_then(toml::Value::as_array)
        .map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect())
        .unwrap_or_default();
    for e in excluded {
        workspace_dirs(&join(dir, &e), out);
    }
}

/// The workspaces a tree may add, which the root's walk does not reach: each directory whose
/// `Cargo.toml` has a `[workspace]` table, the root neither listing nor excluding it. Each
/// lists its crates in its own `LICENSING.md`, the same table.
fn other_workspaces() -> BTreeSet<String> {
    let mut reached = BTreeSet::new();
    workspace_dirs("", &mut reached);
    let mut dirs = BTreeSet::new();
    manifest_dirs(&root(), "", &mut dirs);
    let mut out = BTreeSet::new();
    for d in dirs {
        if manifest(&d).contains_key("workspace") && !reached.contains(&d) {
            workspace_dirs(&d, &mut reached);
            out.insert(d);
        }
    }
    out
}

fn crates() -> Vec<Crate> {
    let mut out = Vec::new();
    workspace("", &mut out);
    for w in other_workspaces() {
        workspace(&w, &mut out);
    }
    out
}

/// Every `Cargo.toml` under the root, as directories relative to it.
fn manifest_dirs(dir: &Path, rel: &str, out: &mut BTreeSet<String>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = entry.path();
        let ty = entry.file_type().unwrap();
        if ty.is_dir() && !SKIPPED_DIRS.contains(&name.as_str()) {
            let sub = if rel.is_empty() {
                name.clone()
            } else {
                format!("{rel}/{name}")
            };
            manifest_dirs(&path, &sub, out);
        } else if ty.is_file() && name == "Cargo.toml" {
            out.insert(rel.to_string());
        }
    }
}

/// The body of `## heading` in `text`, up to the next `## `.
fn section<'a>(text: &'a str, heading: &str) -> &'a str {
    let marker = format!("\n## {heading}\n");
    let start = text
        .find(&marker)
        .unwrap_or_else(|| panic!("LICENSING.md has no section \"## {heading}\""));
    let body = &text[start + marker.len()..];
    match body.find("\n## ") {
        Some(end) => &body[..end],
        None => body,
    }
}

/// The rows of LICENSING.md's crate table, and of the `LICENSING.md` of each other workspace:
/// name -> (directory, license).
fn licensing_map() -> BTreeMap<String, (String, String)> {
    let mut rows = BTreeMap::new();
    licensing_rows("LICENSING.md", &mut rows);
    for w in other_workspaces() {
        licensing_rows(&join(&w, "LICENSING.md"), &mut rows);
    }
    rows
}

/// The rows of the crate table of the map at `rel` into `rows`, each name once in all of them.
fn licensing_rows(rel: &str, rows: &mut BTreeMap<String, (String, String)>) {
    let text = read(rel);
    for line in section(&text, "The crates")
        .lines()
        .filter(|l| l.starts_with("| `"))
    {
        let cells: Vec<&str> = line
            .trim()
            .trim_matches('|')
            .split('|')
            .map(str::trim)
            .collect();
        assert!(cells.len() >= 3, "{rel}: a short row: {line}");
        let name = cells[0].trim_matches('`').to_string();
        let dir = cells[1].trim_matches('`').to_string();
        let license = cells[2].trim_matches('`').to_string();
        assert!(
            rows.insert(name.clone(), (dir, license)).is_none(),
            "{rel}: {name} is listed twice"
        );
    }
}

/// The files LICENSING.md lists as x0ne-infer's files compiled into
/// x0ne-sdk.
fn licensing_shared_files() -> BTreeSet<String> {
    let text = read("LICENSING.md");
    section(&text, "Files of x0ne-infer compiled into x0ne-sdk")
        .lines()
        .filter_map(|l| l.strip_prefix("- `"))
        .map(|l| l.split('`').next().unwrap().to_string())
        .collect()
}

/// A crate's own dependencies that are linked into it: `[dependencies]` and
/// `[build-dependencies]`, plain and per target. Dev-dependencies are not.
fn linked_dependencies(manifest: &toml::Table) -> Vec<(String, toml::Value)> {
    let mut tables: Vec<&toml::Table> = Vec::new();
    for key in ["dependencies", "build-dependencies"] {
        if let Some(t) = manifest.get(key).and_then(toml::Value::as_table) {
            tables.push(t);
        }
    }
    if let Some(targets) = manifest.get("target").and_then(toml::Value::as_table) {
        for target in targets.values().filter_map(toml::Value::as_table) {
            for key in ["dependencies", "build-dependencies"] {
                if let Some(t) = target.get(key).and_then(toml::Value::as_table) {
                    tables.push(t);
                }
            }
        }
    }
    tables
        .into_iter()
        .flat_map(|t| t.iter().map(|(k, v)| (k.clone(), v.clone())))
        .collect()
}

/// The `.rs` files under `dir` (relative to the root), `/`-separated.
fn rust_files(dir: &str, out: &mut Vec<String>) {
    let full = root().join(dir);
    let Ok(entries) = std::fs::read_dir(&full) else {
        return;
    };
    for entry in entries {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        let rel = format!("{dir}/{name}");
        let ty = entry.file_type().unwrap();
        if ty.is_dir() && !SKIPPED_DIRS.contains(&name.as_str()) {
            rust_files(&rel, out);
        } else if ty.is_file() && Path::new(&name).extension().is_some_and(|e| e == "rs") {
            out.push(rel);
        }
    }
}

/// The source files an Apache crate's library and build script compile
/// from outside the crate's directory: `#[path = "…"]` and `include!("…")`.
fn compiled_from_outside(krate: &Crate) -> BTreeSet<String> {
    let mut files = Vec::new();
    rust_files(&format!("{}/src", krate.dir), &mut files);
    files.push(format!("{}/build.rs", krate.dir));
    let mut outside = BTreeSet::new();
    for file in files {
        let Ok(text) = std::fs::read_to_string(root().join(&file)) else {
            continue;
        };
        let dir = Path::new(&file)
            .parent()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        for opener in ["#[path = \"", "include!(\""] {
            let mut rest = text.as_str();
            while let Some(at) = rest.find(opener) {
                rest = &rest[at + opener.len()..];
                let target = &rest[..rest.find('"').unwrap()];
                let resolved = join(&dir, target);
                if !resolved.starts_with(&format!("{}/", krate.dir)) {
                    outside.insert(resolved);
                }
            }
        }
    }
    outside
}

/// The crate table of LICENSING.md names every crate of the tree, by its
/// directory, with the license its manifest declares. A crate it does not
/// list would have a license nobody wrote down.
#[test]
fn licensing_md_lists_every_crate_with_the_license_its_manifest_declares() {
    let crates = crates();
    let map = licensing_map();
    let found: BTreeMap<String, (String, String)> = crates
        .iter()
        .map(|c| (c.name.clone(), (c.dir.clone(), c.license.clone())))
        .collect();
    assert_eq!(
        map, found,
        "LICENSING.md's crate table (left) differs from the manifests (right)"
    );
    let mut dirs = BTreeSet::new();
    manifest_dirs(&root(), "", &mut dirs);
    // The workspaces the walk reached without a crate of their own: the root, x0ne-core.
    let mut roots = BTreeSet::new();
    workspace_dirs("", &mut roots);
    roots.extend(other_workspaces());
    for w in roots {
        if !manifest(&w).contains_key("package") {
            dirs.remove(&w);
        }
    }
    let known: BTreeSet<String> = crates.iter().map(|c| c.dir.clone()).collect();
    assert_eq!(
        dirs, known,
        "a Cargo.toml that no workspace reaches (left) has no place in the map"
    );
}

/// The split decided on 2026-11-02: what an application builds in is
/// Apache-2.0, the server and everything else AGPL-3.0-only. Changing it is
/// a decision, written here as well.
#[test]
fn the_crates_an_application_builds_in_are_apache_and_the_server_agpl() {
    for c in crates() {
        let want = if APACHE_CRATES.contains(&c.name.as_str()) {
            APACHE
        } else {
            AGPL
        };
        assert_eq!(
            c.license, want,
            "{} ({}) declares {}",
            c.name, c.dir, c.license
        );
    }
    let names: BTreeSet<String> = crates().into_iter().map(|c| c.name).collect();
    for name in APACHE_CRATES {
        assert!(names.contains(*name), "no crate {name}");
    }
}

/// The root LICENSE is the AGPL's text, unchanged, then x0ne's additional
/// permission for NVIDIA's libraries. Without the permission, a build that
/// links CUDA could not be conveyed under the AGPL alone. NOTICE names each
/// NVIDIA library the image carries.
#[test]
fn the_root_license_is_the_agpl_then_the_permission_for_nvidia_libraries() {
    let bytes = std::fs::read(root().join("LICENSE")).unwrap();
    assert!(
        bytes.len() > AGPL_TEXT_LEN,
        "LICENSE ({} bytes) is not the AGPL's text followed by the permission",
        bytes.len()
    );
    let (agpl, rest) = bytes.split_at(AGPL_TEXT_LEN);
    assert_eq!(
        sha256_hex(agpl),
        AGPL_TEXT_SHA256,
        "LICENSE does not open with the AGPL's text, unchanged"
    );
    let rest = std::str::from_utf8(rest).unwrap();
    let permission = &rest[rest
        .find(PERMISSION_HEADING)
        .expect("LICENSE: no additional permission after the AGPL's text")..];
    assert!(
        rest.contains("Copyright 2025-2026 x0ne"),
        "LICENSE: no copyright line"
    );
    for lib in NVIDIA_LIBRARIES {
        assert!(
            permission.contains(lib),
            "the permission does not name {lib}"
        );
    }
    let notice = read("NOTICE");
    let carried = image_nvidia_libraries();
    assert!(!carried.is_empty(), "Containerfile: CUDA_LIBS is empty");
    for lib in &carried {
        assert!(
            names_library(&notice, lib),
            "NOTICE does not name {lib}, which the image carries"
        );
        assert!(
            NVIDIA_LIBRARIES.contains(&lib.as_str()),
            "the permission does not cover {lib}, which the image carries"
        );
    }
    assert!(permission.contains("CUDA") && permission.contains("x0ne"));
}

/// Each Apache crate carries the Apache License 2.0, with x0ne's appendix,
/// in its own directory: a copy of the crate alone keeps its license.
#[test]
fn every_apache_crate_carries_the_apache_license() {
    for c in crates().iter().filter(|c| c.license == APACHE) {
        let rel = join(&c.dir, "LICENSE");
        let bytes = std::fs::read(root().join(&rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
        assert_eq!(
            sha256_hex(&bytes),
            APACHE_FILE_SHA256,
            "{rel} is not the Apache License 2.0 with x0ne's appendix"
        );
    }
}

/// No Apache crate links an AGPL crate: an application that builds in the
/// SDK would otherwise be built of AGPL code. Dev-dependencies (tests only)
/// are not linked into what ships.
#[test]
fn no_apache_crate_depends_on_an_agpl_crate() {
    let crates = crates();
    let by_dir: BTreeMap<&str, &Crate> = crates.iter().map(|c| (c.dir.as_str(), c)).collect();
    let by_name: BTreeMap<&str, &Crate> = crates.iter().map(|c| (c.name.as_str(), c)).collect();
    let mut checked = 0;
    for c in crates.iter().filter(|c| c.license == APACHE) {
        for (key, value) in linked_dependencies(&c.manifest) {
            let name = value
                .get("package")
                .and_then(toml::Value::as_str)
                .unwrap_or(&key);
            let dep = match value.get("path").and_then(toml::Value::as_str) {
                Some(path) => {
                    let dir = join(&c.dir, path);
                    *by_dir
                        .get(dir.as_str())
                        .unwrap_or_else(|| panic!("{}: path {path} is no crate", c.name))
                }
                None => match by_name.get(name) {
                    Some(d) => *d,
                    None => continue,
                },
            };
            checked += 1;
            assert_ne!(
                dep.license, AGPL,
                "{} ({}) depends on {} ({})",
                c.name, c.license, dep.name, dep.license
            );
        }
    }
    assert!(checked > 0, "no dependency between the crates was read");
}

/// What an Apache crate compiles from x0ne-infer by `#[path]` is Apache-2.0
/// too: each such file opens with the SPDX line and LICENSING.md lists it,
/// and no other file outside the Apache crates claims Apache-2.0.
#[test]
fn what_the_sdk_compiles_from_the_server_is_apache_and_listed() {
    let crates = crates();
    let apache: Vec<&Crate> = crates.iter().filter(|c| c.license == APACHE).collect();
    let mut compiled = BTreeSet::new();
    for c in &apache {
        compiled.extend(compiled_from_outside(c));
    }
    assert!(
        compiled.len() >= 10,
        "x0ne-sdk compiles {} files of x0ne-infer: the scan lost them",
        compiled.len()
    );
    for file in &compiled {
        let text = read(file);
        assert_eq!(
            text.lines().next(),
            Some(SPDX_APACHE),
            "{file} is compiled into an Apache crate but does not open with \"{SPDX_APACHE}\""
        );
    }
    assert_eq!(
        licensing_shared_files(),
        compiled,
        "LICENSING.md's list (left) differs from what the Apache crates compile (right)"
    );
    let mut claimed = BTreeSet::new();
    for c in crates.iter().filter(|c| c.license != APACHE) {
        let mut files = Vec::new();
        rust_files(&c.dir, &mut files);
        for f in files {
            let text = std::fs::read_to_string(root().join(&f)).unwrap();
            if text.lines().next() == Some(SPDX_APACHE) {
                claimed.insert(f);
            }
        }
    }
    assert_eq!(
        claimed, compiled,
        "files outside the Apache crates that claim Apache-2.0 (left) differ from what they compile (right)"
    );
}
