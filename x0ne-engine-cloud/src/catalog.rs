//! The models a cloud session can serve: the relay's catalog, read at build
//! time from the same file (`x0ne-infer/catalog/models.toml`). The relay
//! still judges the TD against it; this only refuses a name before any cost.

use serde::Deserialize;

use crate::Failure;

const CATALOG: &str = include_str!("../../x0ne-infer/catalog/models.toml");

#[derive(Deserialize)]
struct Raw {
    #[serde(default)]
    model: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    name: String,
    /// The served-model digest; the relay refuses a TD without it (77).
    #[serde(default)]
    digest_sha256: Option<String>,
    #[serde(default)]
    files: Vec<File>,
}

#[derive(Deserialize)]
struct File {
    size: u64,
}

impl Entry {
    /// Single file or sharded: the TD maps the files the catalog lists, then
    /// serves only on the pinned digest (96 without one, 77 on another).
    fn served(&self) -> bool {
        self.digest_sha256.is_some()
    }
}

fn entries(text: &str) -> Result<Vec<Entry>, Failure> {
    toml::from_str::<Raw>(text)
        .map(|r| r.model)
        .map_err(|e| Failure::Preflight(format!("the embedded model catalog: {e}")))
}

fn servable(all: &[Entry]) -> Vec<&str> {
    all.iter()
        .filter(|e| e.served())
        .map(|e| e.name.as_str())
        .collect()
}

fn check_in(text: &str, model: &str) -> Result<(), Failure> {
    let all = entries(text)?;
    let list = servable(&all).join(", ");
    match all.iter().find(|e| e.name == model) {
        None => Err(Failure::Usage(format!(
            "{model} is not in this build's catalog (x0ne-engine catalog); \
             cloud sessions serve: {list}"
        ))),
        Some(e) if !e.served() => Err(Failure::Usage(format!(
            "{model}: no served digest pinned yet, so neither the TD nor the relay can \
             check what would be served; cloud sessions serve: {list}"
        ))),
        Some(_) => Ok(()),
    }
}

fn bytes_in(text: &str, model: &str) -> Option<u64> {
    let all = entries(text).ok()?;
    let e = all.into_iter().find(|e| e.name == model)?;
    Some(e.files.iter().map(|f| f.size).sum())
}

/// Bytes of a model's files in the embedded catalog; `None` for a name it does not list.
pub fn model_bytes(model: &str) -> Option<u64> {
    bytes_in(CATALOG, model)
}

/// Exit 2 for a name the relay or the TD would refuse: absent, or without a
/// pinned served digest.
pub fn check(model: &str) -> Result<(), Failure> {
    check_in(CATALOG, model)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
schema = 1
[[model]]
name = "a:1b"
digest_sha256 = "00"
files = []
[[model]]
name = "b:2b"
files = []
[[model]]
name = "c:32b"
digest_sha256 = "00"
files = [{ path = "model.safetensors.index.json", size = 1, sha256 = "00" }]
[[model]]
name = "d:14b"
files = [{ path = "model.safetensors.index.json", size = 1, sha256 = "00" }]
"#;

    #[test]
    fn the_embedded_catalog_parses_and_serves_the_default_model() {
        assert!(entries(CATALOG).unwrap().len() >= 2);
        assert!(check(crate::DEFAULT_MODEL).is_ok());
        for m in ["qwen3:1.7b", "qwen3:8b", "qwen3:14b", "qwen3:32b"] {
            assert!(check(m).is_ok(), "{m}: pinned 2026-10-17 (v2)");
        }
    }

    #[test]
    fn a_name_outside_the_catalog_is_a_usage_error() {
        let e = check_in(SAMPLE, "d:3b").unwrap_err();
        assert_eq!(e.code(), 2);
        assert!(e.message().contains("not in this build's catalog"), "{e}");
        assert!(
            e.message().ends_with("cloud sessions serve: a:1b, c:32b"),
            "{e}"
        );
        assert!(check("qwen3:999b").is_err());
    }

    #[test]
    fn an_entry_without_a_served_digest_is_a_usage_error() {
        let e = check_in(SAMPLE, "b:2b").unwrap_err();
        assert_eq!(e.code(), 2);
        assert!(e.message().contains("no served digest pinned yet"), "{e}");
        assert!(check_in(SAMPLE, "a:1b").is_ok());
    }

    /// One TOML parser for the check and the boot timeout. The former line scan
    /// found no entry in this valid TOML (no spaces around `=`): 12 min of boot.
    #[test]
    fn model_bytes_read_the_catalog_as_toml() {
        let compact = r#"
schema=1
[[model]]
name="c:32b"
files=[{path="a",size=40000000000,sha256="00"},{path="b",size=25000000000,sha256="00"}]
"#;
        assert_eq!(bytes_in(compact, "c:32b"), Some(65_000_000_000));
        assert_eq!(bytes_in(SAMPLE, "c:32b"), Some(1));
        assert_eq!(bytes_in(SAMPLE, "d:3b"), None);
        assert_eq!(model_bytes("qwen3:1.7b"), Some(4_074_975_920));
    }

    /// Sharded or not, the pin decides: the TD verifies every shard, maps them,
    /// and serves only on the pinned served digest.
    #[test]
    fn a_sharded_entry_is_served_only_with_a_pinned_digest() {
        assert!(check_in(SAMPLE, "c:32b").is_ok());
        let e = check_in(SAMPLE, "d:14b").unwrap_err();
        assert_eq!(e.code(), 2);
        assert!(
            e.message().contains("d:14b: no served digest pinned yet"),
            "{e}"
        );
        assert!(
            e.message().ends_with("cloud sessions serve: a:1b, c:32b"),
            "{e}"
        );
    }
}
