//! The model catalog: a name maps to one pinned set of bytes.
//!
//! Same idea as a `Cargo.lock` checksum: the name is only a handle, what is
//! trusted is the revision and the sha256 of every file. The catalog is
//! untrusted input when it comes from `--catalog`, so every field is checked
//! here, and the embedded one goes through the same checks.

use std::collections::BTreeSet;

use serde::Deserialize;

/// The catalog shipped inside the binary.
pub const EMBEDDED: &str = include_str!("../../catalog/models.toml");

/// Families the engine serves: Qwen3, the one family x0ne Core runs
/// (2026-11-04). A catalog entry for anything else would download
/// bytes the server then refuses. `arch.rs` still reads OLMo and Ministral 3.
pub const ARCHITECTURES: &[&str] = &["qwen3"];

const MAX_FILE_BYTES: u64 = 256 << 30;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Catalog {
    pub models: Vec<Entry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub name: String,
    pub repo: String,
    pub revision: String,
    pub architecture: String,
    pub license: String,
    #[serde(default)]
    pub openness: Option<String>,
    #[serde(default)]
    pub measured: Option<String>,
    /// The served-model digest (v2, `model_digest`) of [`Entry::served_files`]:
    /// what the engine reports and every receipt carries; the TD and the client
    /// refuse any other. Must equal [`Entry::catalog_digest`]. Absent: not served in cloud mode.
    #[serde(default)]
    pub digest_sha256: Option<String>,
    pub files: Vec<FileSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct FileSpec {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    schema: u32,
    #[serde(default)]
    model: Vec<Entry>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CatalogError {
    #[error("catalog is not valid TOML for schema 1: {0}")]
    Syntax(String),
    #[error("catalog schema {0} is not supported (this build reads schema 1)")]
    Schema(u32),
    #[error("catalog entry {name:?}: {reason}")]
    Entry { name: String, reason: String },
}

impl Entry {
    /// `qwen3:0.6b` → (`qwen3`, `0.6b`). Valid after `Catalog::parse`.
    pub fn family_tag(&self) -> (&str, &str) {
        split_name(&self.name).unwrap_or((&self.name, ""))
    }

    /// True when the weights come in shards and their index.
    pub fn is_sharded(&self) -> bool {
        self.files.iter().any(|f| f.path == INDEX)
    }

    /// What a served model is made of: every file but LICENSE and NOTICE,
    /// which the engine never reads.
    pub fn served_files(&self) -> impl Iterator<Item = &FileSpec> {
        self.files
            .iter()
            .filter(|f| !matches!(f.path.as_str(), "LICENSE" | "NOTICE"))
    }

    /// The weight files: `model.safetensors`, or every shard.
    pub fn weight_files(&self) -> impl Iterator<Item = &FileSpec> {
        self.files
            .iter()
            .filter(|f| f.path == "model.safetensors" || is_shard(&f.path))
    }

    /// The served-model digest the catalog's own sha256s give: what `pull`
    /// installs, file by file, can only be served under this value.
    pub fn catalog_digest(&self) -> Result<String, String> {
        crate::model_digest::from_hashes(
            self.served_files()
                .map(|f| (f.path.as_str(), f.sha256.as_str())),
        )
    }

    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// Why cloud mode does not serve this entry, if it does not: no pinned
    /// served digest, single file or sharded. One test for the TD (exit 96)
    /// and the relay, which asks it before any VM exists.
    pub fn cloud_refusal(&self) -> Option<String> {
        self.digest_sha256.is_none().then(|| {
            format!(
                "{}: no served digest pinned yet, not served in cloud mode",
                self.name
            )
        })
    }

    /// The TD's last check before it serves: the digest of what `pull`
    /// installed (the merge, for shards) is the catalog's pin.
    pub fn check_served_digest(&self, got: &str) -> Result<(), String> {
        match self.digest_sha256.as_deref() {
            Some(pin) if pin == got => Ok(()),
            Some(pin) => Err(format!(
                "{}: served digest {got} is not the catalog's pin {pin}; not served",
                self.name
            )),
            None => Err(self.cloud_refusal().unwrap_or_default()),
        }
    }
}

const INDEX: &str = "model.safetensors.index.json";

/// `family:tag`, each `[a-z0-9][a-z0-9._-]{0,31}`. Starting with an
/// alphanumeric means neither part can be `.` or `..`: both become path
/// components of the cache.
pub fn split_name(name: &str) -> Option<(&str, &str)> {
    let (family, tag) = name.split_once(':')?;
    let ok = |s: &str| {
        (1..=32).contains(&s.len())
            && s.as_bytes()[0].is_ascii_alphanumeric()
            && s.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._-".contains(&b))
    };
    (ok(family) && ok(tag)).then_some((family, tag))
}

fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_repo(s: &str) -> bool {
    let Some((owner, name)) = s.split_once('/') else {
        return false;
    };
    let ok = |p: &str| {
        (1..=96).contains(&p.len())
            && p.as_bytes()[0].is_ascii_alphanumeric()
            && p.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    };
    ok(owner) && ok(name) && !s.contains("..")
}

fn is_shard(p: &str) -> bool {
    // model-00001-of-00002.safetensors
    let Some(mid) = p
        .strip_prefix("model-")
        .and_then(|r| r.strip_suffix(".safetensors"))
    else {
        return false;
    };
    let Some((a, b)) = mid.split_once("-of-") else {
        return false;
    };
    !a.is_empty() && !b.is_empty() && a.bytes().chain(b.bytes()).all(|c| c.is_ascii_digit())
}

/// Allow-list of file names. Everything not on it is refused, which is how
/// pickle checkpoints and repository Python stay out without a deny-list.
fn check_path(p: &str) -> Result<(), String> {
    const NAMED: &[&str] = &[
        "config.json",
        "tokenizer.json",
        "model.safetensors",
        INDEX,
        "LICENSE",
        "NOTICE",
    ];
    if NAMED.contains(&p) || is_shard(p) {
        return Ok(());
    }
    let ext = std::path::Path::new(p)
        .extension()
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    let ext = ext.to_str().unwrap_or("");
    let why = if ["bin", "pt", "pth", "pkl", "pickle", "ckpt"].contains(&ext) {
        "pickle checkpoints can run code when loaded; only safetensors weights are accepted"
    } else if ext == "py" {
        "repository Python is remote code; this engine never runs it"
    } else {
        "not a file the engine reads (config.json, tokenizer.json, model.safetensors, \
         model-N-of-M.safetensors + model.safetensors.index.json, LICENSE, NOTICE)"
    };
    Err(format!("file {p:?} refused: {why}"))
}

fn check_entry(e: &Entry) -> Result<(), String> {
    split_name(&e.name)
        .ok_or("name must be family:tag, lowercase [a-z0-9._-], each part at most 32")?;
    if !is_repo(&e.repo) {
        return Err(format!("repo {:?} is not owner/name", e.repo));
    }
    if !is_hex(&e.revision, 40) {
        return Err(format!(
            "revision {:?} must be a full commit sha (40 lowercase hex): a branch or tag can move",
            e.revision
        ));
    }
    if !ARCHITECTURES.contains(&e.architecture.as_str()) {
        return Err(format!(
            "architecture {:?} is not served by this engine ({})",
            e.architecture,
            ARCHITECTURES.join(", ")
        ));
    }
    if e.license.trim().is_empty() || e.license.len() > 64 {
        return Err("license must be a short non-empty SPDX-like name".into());
    }
    if let Some(d) = &e.digest_sha256 {
        if !is_hex(d, 64) {
            return Err("digest_sha256 must be 64 lowercase hex".into());
        }
    }
    if e.files.is_empty() {
        return Err("files is empty".into());
    }
    check_files(e)?;
    if let Some(d) = &e.digest_sha256 {
        let want = e.catalog_digest()?;
        if *d != want {
            return Err(format!(
                "digest_sha256 {d} is not the served-model digest of the files listed ({want})"
            ));
        }
    }
    Ok(())
}

fn check_files(e: &Entry) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for f in &e.files {
        check_path(&f.path)?;
        if !seen.insert(f.path.as_str()) {
            return Err(format!("file {:?} listed twice", f.path));
        }
        if !is_hex(&f.sha256, 64) {
            return Err(format!(
                "file {:?}: sha256 must be 64 lowercase hex",
                f.path
            ));
        }
        if f.size == 0 || f.size > MAX_FILE_BYTES {
            return Err(format!("file {:?}: size {} out of range", f.path, f.size));
        }
    }
    for needed in ["config.json", "tokenizer.json"] {
        if !seen.contains(needed) {
            return Err(format!("{needed} is missing"));
        }
    }
    let single = seen.contains("model.safetensors");
    let shards = e.files.iter().filter(|f| is_shard(&f.path)).count();
    match (single, seen.contains(INDEX), shards) {
        (true, false, 0) | (false, true, 1..) => Ok(()),
        (true, _, _) => Err("model.safetensors and shards: list one or the other".into()),
        _ => Err(
            "weights missing: model.safetensors, or shards with model.safetensors.index.json"
                .into(),
        ),
    }
}

impl Catalog {
    pub fn parse(text: &str) -> Result<Self, CatalogError> {
        let raw: Raw = toml::from_str(text).map_err(|e| CatalogError::Syntax(e.to_string()))?;
        if raw.schema != 1 {
            return Err(CatalogError::Schema(raw.schema));
        }
        let mut names = BTreeSet::new();
        for e in &raw.model {
            check_entry(e).map_err(|reason| CatalogError::Entry {
                name: e.name.clone(),
                reason,
            })?;
            if !names.insert(e.name.as_str()) {
                return Err(CatalogError::Entry {
                    name: e.name.clone(),
                    reason: "name listed twice".into(),
                });
            }
        }
        Ok(Self { models: raw.model })
    }

    pub fn embedded() -> Self {
        Self::parse(EMBEDDED).expect("the embedded catalog is validated by the tests")
    }

    /// The entry for `name`, or a sentence saying what exists instead.
    pub fn get(&self, name: &str) -> Result<&Entry, String> {
        if let Some(e) = self.models.iter().find(|e| e.name == name) {
            return Ok(e);
        }
        let family = name.split(':').next().unwrap_or(name);
        let tags: Vec<&str> = self
            .models
            .iter()
            .filter(|e| e.family_tag().0 == family)
            .map(|e| e.name.as_str())
            .collect();
        Err(if tags.is_empty() {
            let all: Vec<&str> = self.models.iter().map(|e| e.name.as_str()).collect();
            format!(
                "{name:?} is not in the catalog. Available: {}",
                all.join(", ")
            )
        } else {
            format!(
                "{name:?} is not in the catalog. Name a tag: {}",
                tags.join(", ")
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
    const REV: &str = "c1899de289a04d12100db370d81485cdf75e47ca";

    fn entry(files: &str) -> String {
        format!(
            "schema = 1\n[[model]]\nname = \"m:1\"\nrepo = \"o/m\"\nrevision = \"{REV}\"\n\
             architecture = \"qwen3\"\nlicense = \"Apache-2.0\"\nfiles = [{files}]\n"
        )
    }
    fn f(path: &str) -> String {
        format!("{{ path = \"{path}\", size = 1, sha256 = \"{SHA}\" }},")
    }
    fn base() -> String {
        f("config.json") + &f("tokenizer.json")
    }
    /// `text` with its one entry pinned to the digest of its own files.
    fn pinned(text: &str) -> (String, String) {
        let d = Catalog::parse(text).unwrap().models[0]
            .catalog_digest()
            .unwrap();
        let t = text.replace(
            "license = ",
            &format!("digest_sha256 = \"{d}\"\nlicense = "),
        );
        (t, d)
    }
    fn reason(text: &str) -> String {
        match Catalog::parse(text) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("accepted:\n{text}"),
        }
    }

    #[test]
    fn the_embedded_catalog_is_valid_and_has_the_reference_model() {
        let c = Catalog::embedded();
        let q = c.get("qwen3:0.6b").unwrap();
        assert_eq!(q.architecture, "qwen3");
        assert!(c.models.iter().all(|e| e.architecture == "qwen3"));
    }

    /// Two independent measurements of the same bytes must agree: the
    /// openness table (2026-09-25) and the catalog (2026-10-10).
    #[test]
    fn catalog_weights_agree_with_the_openness_table() {
        for e in Catalog::embedded().models {
            let Some(w) = e.files.iter().find(|f| f.path == "model.safetensors") else {
                continue;
            };
            let o = crate::openness::lookup(&w.sha256);
            if let Some(level) = &e.openness {
                assert_eq!(o.level.as_str(), level, "{}", e.name);
            }
        }
    }

    /// Cloud mode serves an entry, sharded or not, only when the catalog pins
    /// its served digest (the TD checks it before serving, the relay refuses a
    /// TD whose digest differs): every entry it serves pins one.
    #[test]
    fn every_cloud_servable_entry_pins_its_served_digest() {
        let mut served = Vec::new();
        for e in Catalog::embedded().models {
            match e.cloud_refusal() {
                None => {
                    assert!(
                        e.digest_sha256.as_deref().is_some_and(|d| is_hex(d, 64)),
                        "{}: served in cloud mode without a served digest",
                        e.name
                    );
                    served.push(e.name.clone());
                }
                Some(why) => {
                    assert!(e.digest_sha256.is_none(), "{}: pinned yet refused", e.name);
                    assert!(why.contains("no served digest pinned yet"), "{why}");
                }
            }
        }
        assert_eq!(
            served,
            [
                "qwen3:0.6b",
                "qwen3:1.7b",
                "qwen3:8b",
                "qwen3:14b",
                "qwen3:32b"
            ]
        );
    }

    /// The pins, frozen. Each is the v2 digest of the entry's own files
    /// (`catalog_digest`; the coreutils command of docs/cloud.md gives the same).
    /// A failure means a served model changed, or the definition did.
    #[test]
    fn the_pins_are_the_v2_digests_of_the_catalog_files() {
        let c = Catalog::embedded();
        for (name, pin) in [
            (
                "qwen3:0.6b",
                "b430888f3024072a1e7dd46e501bca046ed5fdd8e74397e3068a9a66d575d2a4",
            ),
            (
                "qwen3:1.7b",
                "24d5d12d410a39c0349f16d8f491bc826bf5779e284dfff04fced644383b7dba",
            ),
            (
                "qwen3:8b",
                "e1776f7c45e778bbca09ccad38b8b57508da04d862019453ad287cf6715a3b81",
            ),
            (
                "qwen3:14b",
                "a59e04e0949a1f57b3c602fc728a5c8577e9ad06ebc96455259d1fd68452ea1f",
            ),
            (
                "qwen3:32b",
                "745f026c07a33ff5c1b704b04efff392168f4312244e9d1027f26b0c9ae10f51",
            ),
        ] {
            let e = c.get(name).unwrap();
            assert_eq!(e.digest_sha256.as_deref(), Some(pin), "{name}");
            assert_eq!(e.catalog_digest().unwrap(), pin, "{name}");
            // LICENSE is listed, downloaded, and outside the digest.
            let has_licence = e.files.iter().any(|f| f.path == "LICENSE");
            assert_eq!(
                has_licence,
                e.served_files().count() < e.files.len(),
                "{name}"
            );
        }
    }

    /// A pin that is not the digest of the files listed could never be served:
    /// every served file is checked against those sha256s. Refused at parse.
    #[test]
    fn a_pin_other_than_the_files_digest_is_refused() {
        let (ok, d) = pinned(&entry(&(base() + &f("model.safetensors"))));
        assert!(Catalog::parse(&ok).is_ok());
        let other = "1".repeat(64);
        let r = reason(&ok.replace(&d, &other));
        assert!(
            r.contains("is not the served-model digest") && r.contains(&d),
            "{r}"
        );
    }

    /// Frozen 2026-10-16 from the Hub API tree at each commit. A failure means
    /// the catalog changed what `qwen3:8b`, `:14b` or `:32b` downloads.
    #[test]
    fn the_large_qwen3_entries_are_pinned() {
        let c = Catalog::embedded();
        let sha = |e: &Entry, p: &str| {
            e.files
                .iter()
                .find(|f| f.path == p)
                .map(|f| f.sha256.clone())
        };
        let small = c.get("qwen3:0.6b").unwrap();
        let d8b = "e1776f7c45e778bbca09ccad38b8b57508da04d862019453ad287cf6715a3b81";
        let d14b = "a59e04e0949a1f57b3c602fc728a5c8577e9ad06ebc96455259d1fd68452ea1f";
        let d32b = "745f026c07a33ff5c1b704b04efff392168f4312244e9d1027f26b0c9ae10f51";
        for (name, repo, rev, files, shards, bytes, served) in [
            (
                "qwen3:8b",
                "Qwen/Qwen3-8B",
                "b968826d9c46dd6066d109eabc6255188de91218",
                9,
                5,
                16_392_984_379u64,
                Some(d8b),
            ),
            (
                "qwen3:14b",
                "Qwen/Qwen3-14B",
                "40c069824f4251a91eefaf281ebe4c544efd3e18",
                12,
                8,
                29_548_136_879,
                Some(d14b),
            ),
            (
                "qwen3:32b",
                "Qwen/Qwen3-32B",
                "9216db5781bf21249d130ec9da846c4624c16137",
                21,
                17,
                65_535_821_615,
                Some(d32b),
            ),
        ] {
            let e = c.get(name).unwrap();
            assert_eq!(
                (
                    e.repo.as_str(),
                    e.revision.as_str(),
                    e.architecture.as_str()
                ),
                (repo, rev, "qwen3"),
                "{name}"
            );
            assert_eq!(e.files.len(), files, "{name}");
            assert_eq!(e.total_bytes(), bytes, "{name}");
            let mut got: Vec<&str> = e
                .files
                .iter()
                .map(|f| f.path.as_str())
                .filter(|p| is_shard(p))
                .collect();
            got.sort_unstable();
            let want: Vec<String> = (1..=shards)
                .map(|i| format!("model-{i:05}-of-{shards:05}.safetensors"))
                .collect();
            assert_eq!(got, want, "{name}: every shard, once");
            assert!(e.is_sharded(), "{name}");
            assert_eq!(e.digest_sha256.as_deref(), served, "{name}");
            // Same bytes as qwen3:0.6b's, which were downloaded and hashed on 2026-10-10.
            for p in ["tokenizer.json", "LICENSE"] {
                assert_eq!(sha(e, p), sha(small, p), "{name}: {p}");
            }
        }
    }

    /// Each entry's config.json as published (tests/arch/real) is the catalog's
    /// file and parses as qwen3. KV bytes per token, bf16: 2 x layers x KV heads
    /// x head_dim x 2, the figure the H100 budget in docs/pull-run.md uses.
    #[test]
    fn the_large_qwen3_configs_are_the_catalogs_and_parse() {
        use sha2::{Digest, Sha256};
        let c = Catalog::embedded();
        let configs: [(&str, &[u8], usize, usize); 3] = [
            (
                "qwen3:8b",
                include_bytes!("../../tests/arch/real/qwen3-8b.json"),
                36,
                147_456,
            ),
            (
                "qwen3:14b",
                include_bytes!("../../tests/arch/real/qwen3-14b.json"),
                40,
                163_840,
            ),
            (
                "qwen3:32b",
                include_bytes!("../../tests/arch/real/qwen3-32b.json"),
                64,
                262_144,
            ),
        ];
        for (name, json, layers, kv_bytes_per_token) in configs {
            let e = c.get(name).unwrap();
            let want = e.files.iter().find(|f| f.path == "config.json").unwrap();
            let got: String = Sha256::digest(json)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(
                (got, json.len() as u64),
                (want.sha256.clone(), want.size),
                "{name}"
            );
            let d = crate::arch::DecoderConfig::parse(json).unwrap();
            assert_eq!(d.arch, crate::arch::Architecture::Qwen3, "{name}");
            assert_eq!(
                (
                    d.num_layers,
                    d.num_kv_heads,
                    d.head_dim,
                    d.max_position_embeddings
                ),
                (layers, 8, 128, 40_960),
                "{name}"
            );
            assert!(!d.tie_word_embeddings, "{name}: lm_head is its own tensor");
            assert_eq!(
                2 * d.num_layers * d.num_kv_heads * d.head_dim * 2,
                kv_bytes_per_token,
                "{name}"
            );
        }
    }

    /// What the relay and the TD refuse: an entry, single file or sharded,
    /// whose served digest the catalog does not pin; a pinned one passes.
    #[test]
    fn cloud_mode_refuses_exactly_the_entries_without_a_served_digest() {
        for e in Catalog::embedded().models {
            assert_eq!(
                e.cloud_refusal().is_some(),
                e.digest_sha256.is_none(),
                "{}",
                e.name
            );
        }
        let unpinned = Catalog::parse(&entry(&(base() + &f("model.safetensors")))).unwrap();
        let why = unpinned.models[0].cloud_refusal().unwrap();
        assert!(why.contains("no served digest pinned yet"), "{why}");
        let shards = base()
            + &f("model.safetensors.index.json")
            + &f("model-00001-of-00002.safetensors")
            + &f("model-00002-of-00002.safetensors");
        let c = Catalog::parse(&pinned(&entry(&shards)).0).unwrap();
        assert!(c.models[0].is_sharded());
        assert_eq!(c.models[0].cloud_refusal(), None);
    }

    /// The TD's last check before it serves (main.rs, `registry::pull_served`).
    #[test]
    fn the_served_digest_must_be_the_pinned_one() {
        let (text, pin) = pinned(&entry(&(base() + &f("model.safetensors"))));
        let e = &Catalog::parse(&text).unwrap().models[0];
        assert_eq!(e.check_served_digest(&pin), Ok(()));
        let other = "1".repeat(64);
        let why = e.check_served_digest(&other).unwrap_err();
        assert!(why.contains(&other) && why.contains(&pin), "{why}");
        let unpinned = Catalog::parse(&entry(&(base() + &f("model.safetensors")))).unwrap();
        let why = unpinned.models[0].check_served_digest(SHA).unwrap_err();
        assert!(why.contains("no served digest pinned yet"), "{why}");
    }

    #[test]
    fn a_malformed_served_digest_is_refused() {
        let text = entry(&(base() + &f("model.safetensors")))
            .replace("license = ", "digest_sha256 = \"ABC\"\nlicense = ");
        assert!(reason(&text).contains("digest_sha256"));
    }

    #[test]
    fn a_valid_single_file_entry_parses() {
        let c = Catalog::parse(&entry(&(base() + &f("model.safetensors")))).unwrap();
        assert_eq!(c.models[0].family_tag(), ("m", "1"));
        assert!(!c.models[0].is_sharded());
    }

    #[test]
    fn a_valid_sharded_entry_parses() {
        let files = base()
            + &f("model.safetensors.index.json")
            + &f("model-00001-of-00002.safetensors")
            + &f("model-00002-of-00002.safetensors");
        assert!(Catalog::parse(&entry(&files)).unwrap().models[0].is_sharded());
    }

    #[test]
    fn pickle_python_and_unknown_files_are_refused() {
        for (file, why) in [
            ("pytorch_model.bin", "pickle"),
            ("model.pt", "pickle"),
            ("modeling_qwen.py", "remote code"),
            ("../config.json", "not a file the engine reads"),
            ("sub/config.json", "not a file the engine reads"),
            ("vocab.json", "not a file the engine reads"),
        ] {
            let r = reason(&entry(&(base() + &f("model.safetensors") + &f(file))));
            assert!(r.contains(why), "{file}: {r}");
        }
    }

    #[test]
    fn malformed_fields_are_refused() {
        let ok = entry(&(base() + &f("model.safetensors")));
        let cases = [
            (ok.replace(REV, "main"), "full commit sha"),
            (ok.replace(REV, &REV.to_uppercase()), "full commit sha"),
            (ok.replace(SHA, "abc"), "64 lowercase hex"),
            (ok.replace("size = 1,", "size = 0,"), "out of range"),
            (ok.replace("\"qwen3\"", "\"llama\""), "not served"),
            (ok.replace("\"m:1\"", "\"M:1\""), "family:tag"),
            (ok.replace("\"m:1\"", "\"m\""), "family:tag"),
            (ok.replace("\"m:1\"", "\"m:..\""), "family:tag"),
            (ok.replace("\"o/m\"", "\"o/../m\""), "owner/name"),
            (ok.replace("\"o/m\"", "\"m\""), "owner/name"),
            (ok.replace("schema = 1", "schema = 2"), "schema 2"),
            (
                ok.replace("license = \"Apache-2.0\"", "license = \"\""),
                "license",
            ),
            (ok.replace("license", "licence"), "not valid TOML"),
        ];
        for (text, why) in cases {
            let r = reason(&text);
            assert!(r.contains(why), "expected {why:?} in {r:?}");
        }
    }

    #[test]
    fn incomplete_or_ambiguous_file_sets_are_refused() {
        assert!(reason(&entry(&base())).contains("weights missing"));
        assert!(
            reason(&entry(&(f("config.json") + &f("model.safetensors"))))
                .contains("tokenizer.json is missing")
        );
        let both = base() + &f("model.safetensors") + &f("model-00001-of-00001.safetensors");
        assert!(reason(&entry(&both)).contains("one or the other"));
        let no_index = base() + &f("model-00001-of-00001.safetensors");
        assert!(reason(&entry(&no_index)).contains("weights missing"));
        let twice = base() + &f("model.safetensors") + &f("model.safetensors");
        assert!(reason(&entry(&twice)).contains("listed twice"));
    }

    #[test]
    fn duplicate_names_are_refused() {
        let one = entry(&(base() + &f("model.safetensors")));
        let model = one.split_once("[[model]]").unwrap().1;
        assert!(reason(&format!("{one}[[model]]{model}")).contains("name listed twice"));
    }

    #[test]
    fn an_unknown_name_says_what_exists() {
        let c = Catalog::embedded();
        assert!(c.get("qwen3").unwrap_err().contains("qwen3:0.6b"));
        assert!(c.get("mistral:7b").unwrap_err().contains("Available"));
    }
}
