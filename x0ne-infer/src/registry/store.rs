//! The local model cache.
//!
//! ```text
//! <root>/models/<family>/<tag>/      a verified model, ready to serve
//!        ├── config.json, tokenizer.json, model.safetensors or shards + index, [LICENSE]
//!        └── x0ne-model.json         what was installed, with every sha256
//! <root>/partial/<sha256>.part       a download in progress (resumable)
//! <root>/staging/                    a model being assembled
//! <root>/locks/<family>-<tag>.lock   one pull of a name at a time
//! <root>/locks/<sha256>.lock         one download of a file at a time
//! ```
//!
//! A directory under `models/` exists only once every file in it has been
//! verified: it is assembled in `staging/` and renamed into place.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::catalog::{Entry, FileSpec};

pub const RECORD: &str = "x0ne-model.json";

/// Where the cache lives: `$X0NE_ENGINE_HOME`, else `$XDG_CACHE_HOME/x0ne-engine`,
/// else `$HOME/.cache/x0ne-engine`. Relative values are ignored, as the XDG
/// Base Directory specification asks.
pub fn resolve_root(env: impl Fn(&str) -> Option<String>) -> Result<PathBuf, String> {
    let abs = |k: &str| {
        env(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
    };
    if let Some(home) = abs("X0NE_ENGINE_HOME") {
        return Ok(home);
    }
    if let Some(xdg) = abs("XDG_CACHE_HOME") {
        return Ok(xdg.join("x0ne-engine"));
    }
    if let Some(home) = abs("HOME") {
        return Ok(home.join(".cache").join("x0ne-engine"));
    }
    Err("no cache directory: set X0NE_ENGINE_HOME (an absolute path)".into())
}

/// What `x0ne-model.json` holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub name: String,
    pub repo: String,
    pub revision: String,
    pub architecture: String,
    pub license: String,
    /// The files installed, each with its sha256: the catalog's, shards as published.
    pub files: Vec<FileSpec>,
    /// Written by pulls before 0.2.0-beta.6, which merged shards into one file
    /// and listed the downloaded shards here. Such an install is replaced.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub downloaded: Vec<FileSpec>,
    pub pulled_at_unix: u64,
}

impl Record {
    /// An install whose shards were merged (before 0.2.0-beta.6).
    pub fn is_merged(&self) -> bool {
        !self.downloaded.is_empty()
    }

    /// Same pinned bytes as `entry`: the catalog has not moved since the pull.
    /// A merged install lists its merged file, so it never matches.
    pub fn matches(&self, entry: &Entry) -> bool {
        self.repo == entry.repo && self.revision == entry.revision && self.files == entry.files
    }
}

/// A held lock; its file is removed on release, so `locks/` stays empty.
#[derive(Debug)]
pub struct Lock {
    _file: std::fs::File,
    path: PathBuf,
}

impl Drop for Lock {
    fn drop(&mut self) {
        // Unlinked while the flock is still held; the descriptor closes after.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Installed files and directories: owner and group only (were 0644).
pub const FILE_MODE: u32 = 0o640;
pub const DIR_MODE: u32 = 0o750;

/// Set [`FILE_MODE`] / [`DIR_MODE`] on `dir` and everything in it.
pub fn restrict(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(DIR_MODE))?;
        for e in std::fs::read_dir(dir)? {
            let e = e?;
            let t = e.file_type()?;
            if t.is_dir() {
                restrict(&e.path())?;
            } else if t.is_file() {
                std::fs::set_permissions(e.path(), std::fs::Permissions::from_mode(FILE_MODE))?;
            }
        }
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct Store {
    pub root: PathBuf,
}

#[derive(Debug, Clone)]
pub struct Installed {
    pub dir: PathBuf,
    pub record: Record,
}

pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    use std::io::Read;
    // A link to /dev/zero or a FIFO would hash forever.
    let (mut f, _) = crate::fsread::open_regular(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// fsync a file or directory; a rename is durable only once its parent is.
pub fn sync(path: &Path) -> std::io::Result<()> {
    std::fs::File::open(path)?.sync_all()
}

impl Store {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn model_dir(&self, family: &str, tag: &str) -> PathBuf {
        self.root.join("models").join(family).join(tag)
    }
    pub fn partial(&self, sha256: &str) -> PathBuf {
        self.root.join("partial").join(format!("{sha256}.part"))
    }
    /// The byte ranges of a ranged partial already written (`transfer.rs`).
    pub fn partial_ranges(&self, sha256: &str) -> PathBuf {
        self.root.join("partial").join(format!("{sha256}.ranges"))
    }
    pub fn staging(&self) -> PathBuf {
        self.root.join("staging")
    }
    fn trash(&self) -> PathBuf {
        self.root.join("trash")
    }

    /// The installed copy of `name`, if any. `family`/`tag` come from a
    /// validated name, so they are plain path components.
    pub fn installed(&self, family: &str, tag: &str) -> Result<Option<Installed>, String> {
        let dir = self.model_dir(family, tag);
        let path = dir.join(RECORD);
        let bytes = match crate::fsread::read_capped(&path, crate::fsread::CONFIG_CAP) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        let record: Record = serde_json::from_slice(&bytes)
            .map_err(|e| format!("{}: {e} (remove it with x0ne-engine rm)", path.display()))?;
        Ok(Some(Installed { dir, record }))
    }

    /// Every installed model, sorted by name. Unreadable records are listed
    /// as errors rather than hidden.
    pub fn list(&self) -> Vec<Result<Installed, String>> {
        let mut out = Vec::new();
        let Ok(families) = std::fs::read_dir(self.root.join("models")) else {
            return out;
        };
        for fam in families.flatten() {
            let Ok(tags) = std::fs::read_dir(fam.path()) else {
                continue;
            };
            for tag in tags.flatten() {
                let (f, t) = (fam.file_name(), tag.file_name());
                match self.installed(&f.to_string_lossy(), &t.to_string_lossy()) {
                    Ok(Some(i)) => out.push(Ok(i)),
                    Ok(None) => out.push(Err(format!(
                        "{}: no {RECORD}, not a verified model",
                        tag.path().display()
                    ))),
                    Err(e) => out.push(Err(e)),
                }
            }
        }
        out.sort_by(|a, b| match (a, b) {
            (Ok(a), Ok(b)) => a.record.name.cmp(&b.record.name),
            (Ok(_), Err(_)) => std::cmp::Ordering::Less,
            (Err(_), Ok(_)) => std::cmp::Ordering::Greater,
            (Err(a), Err(b)) => a.cmp(b),
        });
        out
    }

    /// Every file present, regular, and of its recorded size: what `list`
    /// can afford. Only [`Store::verify`] proves the bytes.
    pub fn check_sizes(&self, installed: &Installed) -> Result<(), String> {
        for f in &installed.record.files {
            let p = installed.dir.join(&f.path);
            let (_, len) =
                crate::fsread::open_regular(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            if len != f.size {
                return Err(format!(
                    "{}: {len} bytes, the record says {}",
                    p.display(),
                    f.size
                ));
            }
        }
        Ok(())
    }

    /// Re-hash every file of an installed model against its record.
    pub fn verify(&self, installed: &Installed) -> Result<(), String> {
        self.check_sizes(installed)?;
        for f in &installed.record.files {
            let p = installed.dir.join(&f.path);
            let got = sha256_file(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            if got != f.sha256 {
                return Err(format!(
                    "{}: sha256 {got}, the record says {}",
                    p.display(),
                    f.sha256
                ));
            }
        }
        Ok(())
    }

    /// Move `staging` into place as `family/tag`, replacing what was there.
    /// Each step is a rename: a reader sees the old model or the new one.
    pub fn install(&self, staging: &Path, family: &str, tag: &str) -> std::io::Result<PathBuf> {
        let dest = self.model_dir(family, tag);
        let parent = dest.parent().expect("model dir has a parent");
        std::fs::create_dir_all(parent)?;
        if dest.exists() {
            let old = self.trash_slot(family, tag)?;
            std::fs::rename(&dest, &old)?;
            std::fs::rename(staging, &dest)?;
            std::fs::remove_dir_all(&old)?;
        } else {
            std::fs::rename(staging, &dest)?;
        }
        sync(parent)?;
        Ok(dest)
    }

    fn trash_slot(&self, family: &str, tag: &str) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(self.trash())?;
        Ok(self.trash().join(format!(
            "{family}-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        )))
    }

    /// Remove an installed model and the partial downloads of its files.
    /// Returns false when nothing was installed.
    pub fn remove(&self, family: &str, tag: &str, entry: Option<&Entry>) -> std::io::Result<bool> {
        if let Some(e) = entry {
            for f in &e.files {
                for p in [self.partial(&f.sha256), self.partial_ranges(&f.sha256)] {
                    match std::fs::remove_file(p) {
                        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e),
                        _ => {}
                    }
                }
            }
        }
        let dest = self.model_dir(family, tag);
        if !dest.exists() {
            return Ok(false);
        }
        let old = self.trash_slot(family, tag)?;
        std::fs::rename(&dest, &old)?;
        std::fs::remove_dir_all(&old)?;
        let _ = std::fs::remove_dir(dest.parent().expect("parent")); // only if empty
        Ok(true)
    }

    /// An exclusive, non-blocking lock on `name`, released when the file is
    /// dropped. Two pulls of one name would write the same partial files.
    pub fn lock(&self, family: &str, tag: &str) -> Result<Lock, String> {
        self.flock(
            &format!("{family}-{tag}"),
            &format!("pulling {family}:{tag}"),
        )
    }

    /// An exclusive, non-blocking lock on one file's `partial/<sha256>.part`.
    /// A sha256 (64 hex) never equals a `family-tag` lock name.
    pub fn lock_file(&self, sha256: &str) -> Result<Lock, String> {
        self.flock(
            sha256,
            &format!("downloading the file with sha256 {sha256}"),
        )
    }

    #[cfg(unix)]
    fn flock(&self, name: &str, what: &str) -> Result<Lock, String> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
        let dir = self.root.join("locks");
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = dir.join(format!("{name}.lock"));
        let err = |e: std::io::Error| format!("{}: {e}", path.display());
        // The holder unlinks on release: a lock taken on an unlinked inode
        // is retried, or two pulls would each hold "the" lock.
        for _ in 0..8 {
            let f = std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(&path)
                .map_err(err)?;
            // SAFETY: flock on a descriptor we own; no memory is passed.
            let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if rc != 0 {
                return Err(format!(
                    "another x0ne-engine is {what} (lock {})",
                    path.display()
                ));
            }
            let held = f.metadata().map_err(err)?;
            match std::fs::symlink_metadata(&path) {
                Ok(m) if m.dev() == held.dev() && m.ino() == held.ino() => {
                    return Ok(Lock { _file: f, path });
                }
                _ => {}
            }
        }
        Err(format!("{}: lock file keeps changing", path.display()))
    }

    #[cfg(not(unix))]
    fn flock(&self, _name: &str, _what: &str) -> Result<Lock, String> {
        Err("x0ne-engine pull needs a Unix system (file locks)".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn cache_root_follows_the_override_then_xdg_then_home() {
        let all = [
            ("X0NE_ENGINE_HOME", "/srv/models"),
            ("XDG_CACHE_HOME", "/c"),
            ("HOME", "/h"),
        ];
        assert_eq!(
            resolve_root(env(&all)).unwrap(),
            PathBuf::from("/srv/models")
        );
        assert_eq!(
            resolve_root(env(&all[1..])).unwrap(),
            PathBuf::from("/c/x0ne-engine")
        );
        assert_eq!(
            resolve_root(env(&all[2..])).unwrap(),
            PathBuf::from("/h/.cache/x0ne-engine")
        );
    }

    #[test]
    fn relative_or_empty_cache_variables_are_ignored() {
        let e = [
            ("X0NE_ENGINE_HOME", "models"),
            ("XDG_CACHE_HOME", ""),
            ("HOME", "/h"),
        ];
        assert_eq!(
            resolve_root(env(&e)).unwrap(),
            PathBuf::from("/h/.cache/x0ne-engine")
        );
        assert!(resolve_root(env(&[("HOME", "rel")])).is_err());
        assert!(resolve_root(env(&[])).is_err());
    }

    #[test]
    fn model_paths_stay_under_the_root() {
        let s = Store::new(PathBuf::from("/r"));
        assert_eq!(
            s.model_dir("qwen3", "0.6b"),
            PathBuf::from("/r/models/qwen3/0.6b")
        );
        assert_eq!(s.partial("ab"), PathBuf::from("/r/partial/ab.part"));
    }

    #[test]
    fn a_second_lock_on_the_same_name_is_refused() {
        let d = tempfile::tempdir().unwrap();
        let s = Store::new(d.path().into());
        let held = s.lock("m", "1").unwrap();
        assert!(s
            .lock("m", "1")
            .unwrap_err()
            .contains("another x0ne-engine"));
        assert!(s.lock("m", "2").is_ok());
        drop(held);
        assert!(s.lock("m", "1").is_ok());
    }
}
