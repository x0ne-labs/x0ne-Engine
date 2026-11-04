//! `x0ne-engine pull`: fetch a catalog model, verify it, then install it.
//!
//! Order matters, and is the point of this module:
//!
//! 1. download the files to `partial/<sha256>.part`, several at once and a large
//!    one in byte ranges (`transfer.rs`), resuming what a previous attempt left;
//! 2. hash the whole file from disk and compare to the catalog; a mismatch
//!    deletes the partial file and fails the pull;
//! 3. refuse a `config.json` that asks for remote code (`auto_map`) or names
//!    another architecture than the catalog;
//! 4. keep shards as published: the engine maps each file (`shards.rs`);
//! 5. write the record, then rename the staging directory into `models/`.
//!
//! Nothing under `models/` is ever partially written or unverified. Nothing
//! downloaded is executed: the catalog admits safetensors, JSON and licences.
//!
//! Precedents: Cargo (`Cargo.lock` checksums checked before a crate is
//! unpacked), `huggingface-cli download --revision <sha>` (pinned commit,
//! XDG cache), Ollama (`pull`/`run`/`list`/`rm`, digest-addressed blobs).

pub mod catalog;
pub mod store;
mod transfer;

use std::io::Write;
use std::path::Path;
use std::time::Duration;

use sha2::{Digest, Sha256};

pub use catalog::{Catalog, Entry, FileSpec};
pub use store::{Installed, Lock, Record, Store};
pub use transfer::Transfer;

pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

#[derive(Debug, thiserror::Error)]
pub enum PullError {
    /// The bytes are not the ones the catalog pins. Exit 65.
    #[error("{0}")]
    Integrity(String),
    /// Network or server trouble; a retry resumes. Exit 69.
    #[error("{0}")]
    Network(String),
    /// Local file system or configuration. Exit 74.
    #[error("{0}")]
    Local(String),
}

/// The download source. `https://huggingface.co` unless overridden; plain
/// HTTP only towards the loopback (tests, a local mirror).
#[derive(Debug, Clone)]
pub struct Source {
    pub endpoint: String,
    client: reqwest::Client,
    transfer: Transfer,
}

/// Parsed, not prefix-matched: `http://127.0.0.1:x@evil.example` starts like
/// the loopback but its host is `evil.example` (userinfo, 2026-10-10).
fn is_loopback_http(url: &str) -> bool {
    reqwest::Url::parse(url).is_ok_and(|u| {
        u.scheme() == "http"
            && u.username().is_empty()
            && u.password().is_none()
            && matches!(u.host_str(), Some("127.0.0.1" | "[::1]" | "localhost"))
    })
}

/// A URL as a message shows it: without its userinfo, which may be a
/// credential. A string that does not parse loses what would be userinfo.
pub fn shown_url(url: &str) -> String {
    if let Ok(mut u) = reqwest::Url::parse(url) {
        if u.username().is_empty() && u.password().is_none() {
            return url.to_string();
        }
        if u.set_username("").is_ok() && u.set_password(None).is_ok() {
            return u.to_string();
        }
    }
    let (scheme, rest) = url.split_at(url.find("://").map_or(0, |i| i + 3));
    let end = rest.find(['/', '\\', '?', '#']).unwrap_or(rest.len());
    match rest[..end].rfind('@') {
        Some(at) => format!("{scheme}{}", &rest[at + 1..]),
        None => url.to_string(),
    }
}

/// Metadata services answer on these names inside a cloud VM (GCE, EC2).
pub const METADATA_HOSTS: &[&str] = &["metadata.google.internal", "metadata", "instance-data"];

/// Why a TD may not fetch from `host` (a URL's `host_str`): a literal
/// loopback, link-local, private (RFC 1918, ULA) or unspecified address, or a
/// metadata or loopback name. Names are not resolved: a public name may still
/// lead to such an address.
fn non_public_host(host: &str) -> Option<&'static str> {
    use std::net::{Ipv4Addr, Ipv6Addr};
    let v4 = |a: Ipv4Addr| {
        if a.is_loopback() {
            Some("a loopback address")
        } else if a.is_link_local() {
            Some("a link-local address")
        } else if a.is_private() {
            Some("a private address (RFC 1918)")
        } else if a.octets()[0] == 0 {
            Some("an unspecified address")
        } else {
            None
        }
    };
    if let Some(v6) = host.strip_prefix('[').and_then(|h| h.strip_suffix(']')) {
        let a: Ipv6Addr = v6.parse().ok()?;
        return if let Some(m) = a.to_ipv4_mapped() {
            v4(m)
        } else if a.is_loopback() {
            Some("a loopback address")
        } else if a.is_unspecified() {
            Some("an unspecified address")
        } else if a.is_unicast_link_local() {
            Some("a link-local address")
        } else if a.is_unique_local() {
            Some("a unique local address (fc00::/7)")
        } else {
            None
        };
    }
    if let Ok(a) = host.parse::<Ipv4Addr>() {
        return v4(a);
    }
    let name = host.trim_end_matches('.').to_ascii_lowercase();
    if METADATA_HOSTS.contains(&name.as_str()) {
        Some("a cloud metadata name")
    } else if name == "localhost" || name.ends_with(".localhost") {
        Some("a loopback name")
    } else {
        None
    }
}

/// A URL a TD may fetch one file from: https to a host that is not a literal
/// loopback, link-local, private or unspecified address nor a metadata name,
/// or plain http towards the loopback (tests, a bench), with no userinfo
/// anywhere (a signed URL carries its token in the query). DNS is not
/// resolved here: a name that resolves to such an address passes.
pub fn check_source_url(url: &str) -> Result<(), String> {
    let shown = shown_url(url);
    let parsed = reqwest::Url::parse(url).map_err(|e| format!("{shown:?}: {e}; https only"))?;
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(format!("{shown:?} carries userinfo: refused"));
    }
    if parsed.scheme() == "https" {
        if let Some(host) = parsed.host_str() {
            return match non_public_host(host) {
                None => Ok(()),
                Some(why) => Err(format!(
                    "{shown:?} refused: {host} is {why}; the source must be a public https host \
                     (names are not resolved here)"
                )),
            };
        }
    }
    if is_loopback_http(url) {
        return Ok(());
    }
    Err(format!(
        "{shown:?} refused: https only (plain http is allowed to the loopback)"
    ))
}

/// One file by its URL into `store`'s partial area, verified against `spec`
/// (`size`, `sha256`) as a catalog file is: ranges, resumes and the hash of
/// the whole, by the same `transfer` code. Returns the verified file's path,
/// `partial/<sha256>.part`, which a later call finds complete and does not
/// fetch again. `source` was made from this URL (its https rule applies to
/// redirects); `name` names the download in messages.
pub async fn fetch_file(
    store: &Store,
    source: &Source,
    name: &str,
    url: &str,
    spec: &FileSpec,
    progress: &mut dyn Progress,
) -> Result<std::path::PathBuf, PullError> {
    let _lock = store.lock_file(&spec.sha256).map_err(PullError::Local)?;
    let partial = store.root.join("partial");
    std::fs::create_dir_all(&partial).map_err(local(partial.display()))?;
    transfer::fetch_one(store, source, name, url, spec, progress).await?;
    Ok(store.partial(&spec.sha256))
}

impl Source {
    pub fn new(endpoint: &str) -> Result<Self, String> {
        let endpoint = endpoint.trim_end_matches('/').to_string();
        let loopback = is_loopback_http(&endpoint);
        if !endpoint.starts_with("https://") && !loopback {
            return Err(format!(
                "endpoint {:?} refused: https only (plain http is allowed to the loopback)",
                shown_url(&endpoint)
            ));
        }
        // Redirects (the Hub sends LFS files to a CDN) must stay on https,
        // except within a loopback test server.
        let policy = reqwest::redirect::Policy::custom(move |attempt| {
            let url = attempt.url().to_string();
            if attempt.previous().len() >= 10 {
                attempt.error("more than 10 redirects")
            } else if url.starts_with("https://") || (loopback && is_loopback_http(&url)) {
                attempt.follow()
            } else {
                attempt.error(format!("redirect to a non-https URL refused: {url}"))
            }
        });
        let client = reqwest::Client::builder()
            .user_agent(concat!("x0ne-engine/", env!("CARGO_PKG_VERSION")))
            .redirect(policy)
            .https_only(!loopback)
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| format!("http client: {e}"))?;
        Ok(Self {
            endpoint,
            client,
            transfer: Transfer::default(),
        })
    }

    /// Other bounds than [`Transfer::default`].
    #[must_use]
    pub fn with_transfer(mut self, transfer: Transfer) -> Self {
        self.transfer = transfer;
        self
    }

    fn url(&self, entry: &Entry, file: &str) -> String {
        format!(
            "{}/{}/resolve/{}/{}",
            self.endpoint, entry.repo, entry.revision, file
        )
    }
}

/// Progress lines on stderr. Kept out of the logic so tests stay quiet.
pub trait Progress {
    fn file(&mut self, _path: &str, _done: u64, _total: u64) {}
    fn note(&mut self, _msg: &str) {}
}
pub struct Quiet;
impl Progress for Quiet {}

/// What `pull` did.
#[derive(Debug)]
pub struct Pulled {
    pub installed: Installed,
    pub already: bool,
    pub downloaded_bytes: u64,
}

fn local(ctx: impl std::fmt::Display) -> impl FnOnce(std::io::Error) -> PullError {
    move |e| PullError::Local(format!("{ctx}: {e}"))
}

/// Pull `entry` into `store`. A model already installed at the same pinned
/// revision and digests is left untouched (`already`).
pub async fn pull(
    store: &Store,
    source: &Source,
    entry: &Entry,
    progress: &mut dyn Progress,
) -> Result<Pulled, PullError> {
    let (family, tag) = entry.family_tag();
    if let Some(i) = store.installed(family, tag).map_err(PullError::Local)? {
        if i.record.is_merged() {
            progress.note(
                "installed copy has its shards merged (pulled before 0.2.0-beta.6): replacing",
            );
        } else if !i.record.matches(entry) {
            progress.note(&format!(
                "installed revision {} differs from the catalog's {}: replacing",
                short(&i.record.revision),
                short(&entry.revision)
            ));
        } else {
            // The record says what was installed, not what is on disk now.
            match store.verify(&i) {
                Ok(()) => {
                    return Ok(Pulled {
                        installed: i,
                        already: true,
                        downloaded_bytes: 0,
                    })
                }
                Err(e) => progress.note(&format!("installed copy is damaged ({e}): replacing")),
            }
        }
    }
    let _lock = store.lock(family, tag).map_err(PullError::Local)?;
    // partial/<sha256>.part is shared by every name listing that file.
    let _file_locks = entry
        .files
        .iter()
        .map(|f| store.lock_file(&f.sha256))
        .collect::<Result<Vec<_>, _>>()
        .map_err(PullError::Local)?;

    for d in ["partial", "staging"] {
        let p = store.root.join(d);
        std::fs::create_dir_all(&p).map_err(local(p.display()))?;
    }
    let staging = store.staging().join(format!("{family}-{tag}")); // unique: we hold the lock
    if staging.exists() {
        std::fs::remove_dir_all(&staging).map_err(local(staging.display()))?;
    }
    std::fs::create_dir(&staging).map_err(local(staging.display()))?;
    let result = assemble(store, source, entry, &staging, progress).await;
    let downloaded_bytes = match result {
        Ok(v) => v,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(e);
        }
    };

    let record = Record {
        name: entry.name.clone(),
        repo: entry.repo.clone(),
        revision: entry.revision.clone(),
        architecture: entry.architecture.clone(),
        license: entry.license.clone(),
        downloaded: Vec::new(),
        files: entry.files.clone(),
        pulled_at_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    let rec_path = staging.join(store::RECORD);
    let json = serde_json::to_vec_pretty(&record).expect("record serializes");
    write_synced(&rec_path, &json).map_err(local(rec_path.display()))?;
    store::restrict(&staging).map_err(local(staging.display()))?;
    store::sync(&staging).map_err(local(staging.display()))?;
    let dir = store
        .install(&staging, family, tag)
        .map_err(local("install"))?;
    Ok(Pulled {
        installed: Installed { dir, record },
        already: false,
        downloaded_bytes,
    })
}

/// What cloud mode serves: the files the served digest covers, the weights
/// mapped read-only, and that digest, already checked against the catalog's pin.
pub struct Served {
    pub config_json: Vec<u8>,
    pub tokenizer_json: Vec<u8>,
    pub index_json: Option<Vec<u8>>,
    /// The catalog's weight files, by name, in the catalog's order.
    pub weights: Vec<(String, memmap2::Mmap)>,
    pub digest_hex: String,
}

impl Served {
    pub fn weight_files(&self) -> crate::backend::WeightFiles<'_> {
        crate::backend::WeightFiles {
            index_json: self.index_json.as_deref(),
            files: self
                .weights
                .iter()
                .map(|(n, m)| (n.as_str(), &m[..]))
                .collect(),
        }
    }
}

/// Cloud mode's intake: `pull` checks every file; the CATALOG's weight files are
/// mapped, never a directory listing; index and headers agree (72); the v2 served
/// digest is the pin (77). 96 unpinned or pull failed, 71 unreadable.
pub async fn pull_served(
    store: &Store,
    source: &Source,
    entry: &Entry,
    progress: &mut dyn Progress,
) -> Result<Served, (i32, String)> {
    if let Some(why) = entry.cloud_refusal() {
        return Err((96, why));
    }
    let pulled = pull(store, source, entry, progress)
        .await
        .map_err(|e| (96, format!("pull {}: {e}", entry.name)))?;
    let dir = pulled.installed.dir;
    let unreadable = |f: &str, e: std::io::Error| (71, format!("{}: {e}", dir.join(f).display()));
    let read = |f: &str| crate::fsread::read_model_file(&dir, f).map_err(|e| unreadable(f, e));
    let config_json = read("config.json")?;
    let tokenizer_json = read("tokenizer.json")?;
    let index_json = if entry.is_sharded() {
        Some(read(crate::shards::INDEX_FILE)?)
    } else {
        None
    };
    let weights = entry
        .weight_files()
        .map(|f| {
            crate::fsread::map_model_file(&dir, &f.path)
                .map(|m| (f.path.clone(), m))
                .map_err(|e| unreadable(&f.path, e))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut served = Served {
        config_json,
        tokenizer_json,
        index_json,
        weights,
        digest_hex: String::new(),
    };
    let digest_hex = {
        let files = served.weight_files();
        crate::shards::Routing::check(files.index_json, &files.files)
            .map_err(|e| (72, format!("{}: {e:#}", entry.name)))?;
        crate::model_spec::served_digest(&served.config_json, &served.tokenizer_json, &files)
            .map_err(|e| (77, format!("{e:#}")))?
    };
    entry
        .check_served_digest(&digest_hex)
        .map_err(|e| (77, e))?;
    served.digest_hex = digest_hex;
    Ok(served)
}

fn write_synced(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(bytes)?;
    f.sync_all()
}

fn short(rev: &str) -> &str {
    &rev[..rev.len().min(12)]
}

/// Download and verify every file into `staging`, shards as published.
/// Returns the bytes transferred.
async fn assemble(
    store: &Store,
    source: &Source,
    entry: &Entry,
    staging: &Path,
    progress: &mut dyn Progress,
) -> Result<u64, PullError> {
    // Every file verified in partial/ first: an interrupted pull leaves only
    // resumable partials behind, never half a staging directory.
    let downloaded = transfer::fetch_all(store, source, entry, progress).await?;
    for f in &entry.files {
        let dest = staging.join(&f.path);
        // Same file system (both under the root): a rename, not a copy.
        std::fs::rename(store.partial(&f.sha256), &dest).map_err(local(dest.display()))?;
    }
    check_config(entry, &staging.join("config.json"))?;
    Ok(downloaded)
}

/// The architecture check and the remote-code check, on the verified file.
fn check_config(entry: &Entry, path: &Path) -> Result<(), PullError> {
    let bytes = crate::fsread::read_capped(path, crate::fsread::CONFIG_CAP)
        .map_err(local(path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| PullError::Integrity(format!("config.json: {e}")))?;
    if v.get("auto_map").is_some() {
        return Err(PullError::Integrity(
            "config.json declares auto_map (code shipped with the model, trust_remote_code): \
             refused, this engine never runs repository code"
                .into(),
        ));
    }
    let arch = crate::arch::Architecture::detect(&bytes)
        .map_err(|e| PullError::Integrity(format!("{e:#}")))?;
    if arch.name() != entry.architecture {
        return Err(PullError::Integrity(format!(
            "config.json says {}, the catalog says {}",
            arch.name(),
            entry.architecture
        )));
    }
    Ok(())
}

fn short_source(name: &str, path: &str, size: u64, got: u64) -> PullError {
    PullError::Integrity(format!(
        "{name} {path}: the source's file ends at {got} bytes, the catalog says {size}; the \
         partial download was deleted. Either the source serves another file or the catalog is wrong"
    ))
}

/// The full size the server says the file has: `Content-Range: bytes a-b/N`
/// on a 206, `Content-Length` on a 200. `None` when it does not say.
fn declared_total(resp: &reqwest::Response, status: u16) -> Option<u64> {
    match status {
        206 => resp
            .headers()
            .get(reqwest::header::CONTENT_RANGE)?
            .to_str()
            .ok()?
            .rsplit_once('/')?
            .1
            .parse()
            .ok(),
        200 => resp.content_length(),
        _ => None,
    }
}

/// sha256 of a byte slice, lowercase hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    store::hex(&Sha256::digest(bytes))
}

#[cfg(test)]
mod tests;
