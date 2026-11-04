//! The compiler is pinned in four places; they must name the same one.
//!
//! `rust-toolchain.toml` (what `cargo` runs locally), the Containerfile's
//! `RUST_IMAGE` (what the image is built with), `ci.yml`'s `RUST_TOOLCHAIN`
//! (what CI runs) and `Cargo.toml`'s `rust-version` (the declared MSRV). A
//! drift means the MSRV is claimed by a compiler that never tested it.

use std::path::{Path, PathBuf};

/// Workspace files sit one level above this crate's manifest.
fn find(relative: &str) -> PathBuf {
    let mut dir: &Path = Path::new(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        let candidate = dir.join(relative);
        if candidate.exists() {
            return candidate;
        }
        dir = dir.parent().expect("workspace root reachable");
    }
    panic!("{relative} not found above {}", env!("CARGO_MANIFEST_DIR"));
}

fn read(relative: &str) -> String {
    std::fs::read_to_string(find(relative)).expect("readable")
}

/// The manifest that carries `[workspace]`; the crate's own only inherits.
fn workspace_manifest() -> String {
    let mut dir: &Path = Path::new(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..3 {
        if let Ok(text) = std::fs::read_to_string(dir.join("Cargo.toml")) {
            if text.contains("[workspace]") {
                return text;
            }
        }
        dir = dir.parent().expect("workspace root reachable");
    }
    panic!(
        "no workspace Cargo.toml above {}",
        env!("CARGO_MANIFEST_DIR")
    );
}

/// `key = "value"` or `key: value`, first occurrence, quotes stripped.
fn value_after(text: &str, key: &str) -> String {
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with(key))
        .unwrap_or_else(|| panic!("no line starting with {key:?}"));
    let rest = line[key.len()..].trim_start_matches([' ', '=', ':']).trim();
    rest.trim_matches('"').to_string()
}

/// A failure means one pin was bumped without the others: the MSRV, the CI
/// compiler and the image compiler no longer describe the same build.
#[test]
fn the_four_compiler_pins_name_the_same_toolchain() {
    let channel = value_after(&read("rust-toolchain.toml"), "channel");
    let ci = value_after(&read(".github/workflows/ci.yml"), "RUST_TOOLCHAIN");
    let msrv = value_after(&workspace_manifest(), "rust-version");

    let containerfile = read("Containerfile");
    let image = value_after(&containerfile, "ARG RUST_IMAGE=");
    let image_version = image
        .split("rust:")
        .nth(1)
        .and_then(|s| s.split('-').next())
        .unwrap_or_else(|| panic!("RUST_IMAGE is not rust:<version>-…: {image}"));

    assert_eq!(
        ci, channel,
        "ci.yml RUST_TOOLCHAIN vs rust-toolchain.toml channel"
    );
    assert_eq!(
        image_version, channel,
        "Containerfile RUST_IMAGE vs rust-toolchain.toml channel"
    );
    assert!(
        channel.starts_with(&format!("{msrv}.")) || channel == msrv,
        "Cargo.toml rust-version {msrv} is not the major.minor of the toolchain {channel}"
    );
}
