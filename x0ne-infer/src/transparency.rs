// SPDX-License-Identifier: Apache-2.0
//! The transparency log of image references: the relay accepts an image
//! reference only if it is in the log this build pins, shown by a
//! `c2sp.org/tlog-proof@v1` file next to it (`<reference>.tlog-proof`):
//!
//! - the proof's checkpoint is signed by the pinned log key and names the
//!   pinned origin;
//! - it is cosigned by at least `quorum` of the pinned witnesses within
//!   `max_age_s` of this machine's clock; an older one is replaced by the
//!   log's current checkpoint, read from its tiles, shown consistent with it,
//!   and the reference shown included in it; a log that cannot be read then
//!   refuses the reference;
//! - the inclusion proof leads from the leaf of THIS reference's bytes
//!   (`entry`) to that checkpoint's root: a valid proof of another entry
//!   is refused (Cosign accepted one, CVE-2026-22703);
//! - the checkpoint is consistent with the last one this machine verified
//!   (kept under the relay's state directory): two signed heads that are not
//!   a prefix of each other are a fork, refused, and both are kept as evidence.
//!
//! Everything refused here exits 100. A build pins its log the way it pins
//! the archive key (`archive_key.rs`): `RELEASE` is None until x0ne runs a
//! log and witnesses agree to cosign it, and until then every reference is
//! refused; a debug build with `test-archive-key` pins the TEST log
//! (`test_log`), whose keys anyone can derive. No flag and no variable adds
//! a log or a witness or skips a check.
//!
//! x0ne-sdk compiles this file too (`#[path]`): what reads a file or the
//! network is left out there, and its check is `check_offline`, the steps
//! of `check` up to the refresh, on the proof alone; then `remember`, the
//! caller's memory of the log in place of this machine's state directory:
//! the last checkpoint it verified, handed back to it to keep, and a
//! newer one admitted only with a consistency proof it supplies
//! (tlog-witness `add-checkpoint` body), as a witness admits one.
//!
//! What a log proves: the reference was published where anyone can see it,
//! so an image made for one client is in the same list as every other. It
//! does not prove the image is good, and with fewer honest witnesses than
//! the quorum a log can still show a client its own view.

use crate::tlog::{self, Checkpoint, KeyKind, NoteKey, TileSource};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The relay's exit when a reference is not shown to be in the pinned log.
pub const EXIT_NOT_LOGGED: i32 = 100;
/// The first line of a logged entry: one image reference, by its SHA-256.
pub const ENTRY_HEADER: &str = "x0ne-engine/image-reference/v1";
pub const PROOF_SUFFIX: &str = ".tlog-proof";
pub const MAX_PROOF_BYTES: u64 = (tlog::MAX_NOTE_BYTES + 4096) as u64;
/// A cosignature dated more than this ahead of the local clock is not counted.
pub const FUTURE_SKEW_S: u64 = 300;
const MAX_TILE_BYTES: u64 = tlog::TILE_WIDTH * 32;

/// The logged entry of a reference: what the inclusion proof must lead from.
pub fn entry(reference: &[u8]) -> Vec<u8> {
    use sha2::Digest as _;
    let digest = tlog::hex(&sha2::Sha256::digest(reference));
    format!("{ENTRY_HEADER}\nsha256 {digest}\n").into_bytes()
}

/// `<reference>.tlog-proof`, next to the file.
pub fn proof_path(reference: &Path) -> PathBuf {
    let mut p = reference.as_os_str().to_owned();
    p.push(PROOF_SUFFIX);
    PathBuf::from(p)
}

/// A log as a build pins it.
pub struct Pin {
    pub origin: &'static str,
    /// The log's Ed25519 verifier key (signed-note vkey), named by the origin.
    pub log_key: &'static str,
    /// Witness `cosignature/v1` verifier keys.
    pub witnesses: &'static [&'static str],
    pub quorum: usize,
    pub max_age_s: u64,
    /// `https://` prefix of the log's tiles and checkpoint.
    pub tiles: &'static str,
}

/// The release pin: None until x0ne runs a public log with witnesses.
pub const RELEASE: Option<Pin> = None;

#[cfg_attr(all(feature = "test-archive-key", debug_assertions), allow(dead_code))]
const NO_LOG: &str = "this build pins no transparency log: no public log of x0ne image \
     references exists yet, so it accepts no image reference; a build made once the log runs \
     pins it, and no flag replaces it";

/// Where the log's files are read when a checkpoint must be refreshed or
/// two checkpoints compared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tiles {
    Https(String),
    /// The `tlog/` directory next to the reference (the TEST log).
    BesideReference,
}

#[derive(Debug, Clone)]
pub struct Policy {
    pub origin: String,
    pub log: NoteKey,
    pub witnesses: Vec<NoteKey>,
    pub quorum: usize,
    pub max_age_s: u64,
    pub tiles: Tiles,
}

/// The log every reference must be in, or why this build has none.
pub struct Anchor(Result<Policy, String>);

impl Anchor {
    pub fn production() -> Self {
        // Only a debug build may pin the TEST log, as the TEST archive key.
        #[cfg(all(feature = "test-archive-key", debug_assertions))]
        {
            Anchor(Ok(test_log::policy()))
        }
        #[cfg(not(all(feature = "test-archive-key", debug_assertions)))]
        {
            Anchor(match &RELEASE {
                Some(pin) => pinned(pin),
                None => Err(NO_LOG.into()),
            })
        }
    }

    pub fn policy(&self) -> Result<&Policy, String> {
        self.0.as_ref().map_err(Clone::clone)
    }
}

/// A pin is a log key named by its origin, witnesses of distinct names and
/// keys, none of them the log's key, a quorum of at least one of them, a
/// maximum age, and an HTTPS location.
pub fn pinned(pin: &Pin) -> Result<Policy, String> {
    let log = NoteKey::parse(pin.log_key).map_err(|e| format!("pinned log: {e}"))?;
    if log.kind != KeyKind::Ed25519 || log.name != pin.origin {
        return Err(format!(
            "pinned log: the key must be an Ed25519 note key named {:?}",
            pin.origin
        ));
    }
    let witnesses = pin
        .witnesses
        .iter()
        .map(|w| NoteKey::parse(w).map_err(|e| format!("pinned witness: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    for (i, w) in witnesses.iter().enumerate() {
        if w.kind != KeyKind::CosignatureV1 {
            return Err(format!(
                "pinned witness {}: not a cosignature/v1 key",
                w.name
            ));
        }
        if witnesses[..i].iter().any(|o| o.name == w.name) {
            return Err(format!("pinned witness {} twice", w.name));
        }
        // A cosignature/v1 does not sign its cosigner's name (C2SP tlog-cosignature):
        // one key under two names would count twice toward the quorum (tlog-policy).
        if let Some(o) = witnesses[..i]
            .iter()
            .find(|o| o.public_key() == w.public_key())
        {
            return Err(format!(
                "pinned witness {}: the key of pinned witness {}",
                w.name, o.name
            ));
        }
        if w.public_key() == log.public_key() {
            return Err(format!("pinned witness {}: the log's own key", w.name));
        }
    }
    if pin.quorum == 0 || pin.quorum > witnesses.len() {
        return Err(format!(
            "pinned log: quorum {} of {} witnesses; at least one is required",
            pin.quorum,
            witnesses.len()
        ));
    }
    if pin.max_age_s == 0 || !pin.tiles.starts_with("https://") {
        return Err("pinned log: a maximum age and an https:// tile location are required".into());
    }
    Ok(Policy {
        origin: pin.origin.into(),
        log,
        witnesses,
        quorum: pin.quorum,
        max_age_s: pin.max_age_s,
        tiles: Tiles::Https(pin.tiles.into()),
    })
}

/// What the log check established, for the verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Logged {
    pub origin: String,
    pub index: u64,
    pub size: u64,
    pub root_hex: String,
    /// The witnesses whose fresh cosignature counted, with its time.
    pub cosigned_by: Vec<(String, u64)>,
    /// The proof's checkpoint was too old: the log's current one was used.
    pub refreshed: bool,
}

/// The log's files in a directory. Paths are the log's own, nothing else.
#[cfg(not(x0ne_sdk))]
pub struct DirTiles(pub PathBuf);

#[cfg(not(x0ne_sdk))]
fn safe_tile_path(path: &str) -> bool {
    path.starts_with("tile/")
        && !path.contains("..")
        && path
            .bytes()
            .all(|b| b.is_ascii_digit() || b.is_ascii_lowercase() || b == b'/' || b == b'.')
}

#[cfg(not(x0ne_sdk))]
impl TileSource for DirTiles {
    fn checkpoint(&self) -> Result<String, String> {
        let p = self.0.join("checkpoint");
        let b = crate::fsread::read_capped(&p, tlog::MAX_NOTE_BYTES as u64)
            .map_err(|e| format!("{}: {e}", p.display()))?;
        String::from_utf8(b).map_err(|_| format!("{}: not UTF-8", p.display()))
    }

    fn read(&self, path: &str) -> Result<Vec<u8>, String> {
        if !safe_tile_path(path) {
            return Err(format!("{path:?} is not a tile path"));
        }
        let p = self.0.join(path);
        crate::fsread::read_capped(&p, MAX_TILE_BYTES).map_err(|e| format!("{}: {e}", p.display()))
    }
}

/// A log that cannot be read: every refresh or comparison fails closed.
pub struct Unreadable(pub String);

impl TileSource for Unreadable {
    fn checkpoint(&self) -> Result<String, String> {
        Err(self.0.clone())
    }
    fn read(&self, _: &str) -> Result<Vec<u8>, String> {
        Err(self.0.clone())
    }
}

/// The time one request to the log may take, connection included.
pub const HTTP_TIMEOUT: Duration = Duration::from_secs(20);

/// `http://` to 127.0.0.1, `[::1]` or localhost, with no credentials: a
/// local instance of the log, for tests and rehearsals.
#[cfg(not(x0ne_sdk))]
fn is_loopback_http(url: &str) -> bool {
    reqwest::Url::parse(url).is_ok_and(|u| {
        u.scheme() == "http"
            && u.username().is_empty()
            && u.password().is_none()
            && matches!(u.host_str(), Some("127.0.0.1" | "[::1]" | "localhost"))
    })
}

/// The log's files over HTTPS (plain HTTP on the loopback alone). One GET
/// per file, no redirect followed, one time limit for the whole request,
/// the body capped at the file's own cap; any other status, a longer body
/// or a failure of the transport is an error, and the check fails closed.
/// Each request runs on a thread of its own with its own runtime, so the
/// synchronous check works inside or outside an async runtime.
#[cfg(not(x0ne_sdk))]
#[derive(Debug)]
pub struct HttpTiles {
    base: String,
    loopback: bool,
    timeout: Duration,
}

#[cfg(not(x0ne_sdk))]
impl HttpTiles {
    pub fn new(base: &str) -> Result<Self, String> {
        Self::with_timeout(base, HTTP_TIMEOUT)
    }

    pub fn with_timeout(base: &str, timeout: Duration) -> Result<Self, String> {
        let loopback = is_loopback_http(base);
        let https = reqwest::Url::parse(base).is_ok_and(|u| {
            u.scheme() == "https"
                && u.host_str().is_some()
                && u.username().is_empty()
                && u.password().is_none()
        });
        if !https && !loopback {
            return Err(format!(
                "log location {base:?} refused: https only (plain http is allowed to the loopback)"
            ));
        }
        Ok(HttpTiles {
            base: base.trim_end_matches('/').to_string(),
            loopback,
            timeout,
        })
    }

    fn get(&self, path: &str, cap: u64) -> Result<Vec<u8>, String> {
        let url = format!("{}/{path}", self.base);
        let (loopback, timeout) = (self.loopback, self.timeout);
        std::thread::scope(|s| {
            s.spawn(|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| format!("{url}: runtime: {e}"))?;
                rt.block_on(fetch(&url, loopback, timeout, cap))
            })
            .join()
            .unwrap_or_else(|_| Err(format!("{url}: the request thread panicked")))
        })
    }
}

/// A reqwest error with its causes, and the time limit named on a timeout.
#[cfg(not(x0ne_sdk))]
fn describe(e: &reqwest::Error, timeout: Duration) -> String {
    if e.is_timeout() {
        return format!("timed out after {} s", timeout.as_secs_f64());
    }
    let mut out = e.to_string();
    let mut cause = std::error::Error::source(e);
    while let Some(c) = cause {
        out.push_str(": ");
        out.push_str(&c.to_string());
        cause = std::error::Error::source(c);
    }
    out
}

#[cfg(not(x0ne_sdk))]
async fn fetch(url: &str, loopback: bool, timeout: Duration, cap: u64) -> Result<Vec<u8>, String> {
    let client = reqwest::Client::builder()
        .user_agent(concat!("x0ne-engine/", env!("CARGO_PKG_VERSION")))
        .redirect(reqwest::redirect::Policy::none())
        .https_only(!loopback)
        .connect_timeout(timeout)
        .timeout(timeout)
        .build()
        .map_err(|e| format!("{url}: http client: {e}"))?;
    let mut resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("{url}: {}", describe(&e, timeout)))?;
    if resp.status() != reqwest::StatusCode::OK {
        return Err(format!("{url}: HTTP {}", resp.status().as_u16()));
    }
    if resp.content_length().is_some_and(|l| l > cap) {
        return Err(format!("{url}: larger than the {cap}-byte cap"));
    }
    let mut body = Vec::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| format!("{url}: {}", describe(&e, timeout)))?
    {
        if (body.len() + chunk.len()) as u64 > cap {
            return Err(format!("{url}: larger than the {cap}-byte cap"));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[cfg(not(x0ne_sdk))]
impl TileSource for HttpTiles {
    fn checkpoint(&self) -> Result<String, String> {
        let b = self.get("checkpoint", tlog::MAX_NOTE_BYTES as u64)?;
        String::from_utf8(b).map_err(|_| format!("{}/checkpoint: not UTF-8", self.base))
    }

    fn read(&self, path: &str) -> Result<Vec<u8>, String> {
        if !safe_tile_path(path) {
            return Err(format!("{path:?} is not a tile path"));
        }
        self.get(path, MAX_TILE_BYTES)
    }
}

#[cfg(not(x0ne_sdk))]
impl Policy {
    /// The log's files for a reference at `reference`: the `tlog/`
    /// directory next to it for the TEST log, the pinned location over
    /// HTTPS otherwise.
    pub fn source(&self, reference: &Path) -> Box<dyn TileSource> {
        match &self.tiles {
            Tiles::BesideReference => Box::new(DirTiles(
                reference.parent().unwrap_or(Path::new(".")).join("tlog"),
            )),
            Tiles::Https(url) => match HttpTiles::new(url) {
                Ok(h) => Box::new(h),
                Err(e) => Box::new(Unreadable(e)),
            },
        }
    }
}

impl Policy {
    fn keys(&self) -> Vec<NoteKey> {
        std::iter::once(self.log.clone())
            .chain(self.witnesses.iter().cloned())
            .collect()
    }

    /// A signed checkpoint of this log: its log signature verified, its
    /// origin pinned. Returns the checkpoint, its note and the cosignatures.
    fn open(&self, note: &str) -> Result<(Checkpoint, Vec<(String, u64)>), String> {
        let n = tlog::parse_note(note).map_err(|e| format!("checkpoint: {e}"))?;
        let verified = tlog::open_note(&n, &self.keys())
            .map_err(|e| format!("checkpoint not signed by the pinned log: {e}"))?;
        if !verified.iter().any(|v| v.key == self.log) {
            return Err(format!(
                "checkpoint not signed by the pinned log {}",
                self.log.name
            ));
        }
        let cp = tlog::parse_checkpoint(&n.text)?;
        if cp.origin != self.origin {
            return Err(format!(
                "checkpoint of {:?}, the pinned log is {:?}",
                cp.origin, self.origin
            ));
        }
        let cosigned = verified
            .into_iter()
            .filter_map(|v| v.time.map(|t| (v.key.name, t)))
            .collect();
        Ok((cp, cosigned))
    }

    /// The cosignatures that count at `now`: not older than `max_age_s`,
    /// not ahead of the clock by more than `FUTURE_SKEW_S`.
    fn fresh(&self, cosigned: &[(String, u64)], now: u64) -> Vec<(String, u64)> {
        cosigned
            .iter()
            .filter(|(_, t)| {
                t.saturating_add(self.max_age_s) >= now && *t <= now.saturating_add(FUTURE_SKEW_S)
            })
            .cloned()
            .collect()
    }
}

fn root_line(cp: &Checkpoint) -> String {
    format!("size {} root {}", cp.size, tlog::hex(&cp.root))
}

/// What a proof shows before its freshness is judged: its checkpoint signed
/// by the pinned log under the pinned origin, and the leaf of THIS
/// reference's bytes included in it.
struct Proven {
    proof: tlog::TlogProof,
    cp: Checkpoint,
    cosigned: Vec<(String, u64)>,
    leaf: tlog::Hash,
}

fn proven(reference: &[u8], proof_text: &str, policy: &Policy) -> Result<Proven, String> {
    let proof = tlog::parse_tlog_proof(proof_text)?;
    let (cp, cosigned) = policy.open(&proof.checkpoint)?;
    let leaf = tlog::leaf_hash(&entry(reference));
    tlog::verify_inclusion(&leaf, proof.index, cp.size, &proof.hashes, &cp.root)
        .map_err(|e| format!("this reference is not shown in the log: {e}"))?;
    Ok(Proven {
        proof,
        cp,
        cosigned,
        leaf,
    })
}

/// Why `cosigned`, of which `fresh` count now, is not a fresh quorum.
fn stale(policy: &Policy, cosigned: &[(String, u64)], fresh: usize) -> String {
    if cosigned.len() < policy.quorum {
        format!(
            "checkpoint cosigned by {} of the pinned witnesses, {} required",
            cosigned.len(),
            policy.quorum
        )
    } else {
        format!(
            "checkpoint stale: {fresh} of its cosignatures are within {} s of now, {} required",
            policy.max_age_s, policy.quorum
        )
    }
}

/// The log check of a reference's bytes and its proof alone, for a verifier
/// that reads no file and opens no socket (x0ne-sdk): [`check`]'s steps up
/// to its refresh, the same functions. A checkpoint without a fresh quorum
/// of cosignatures is refused here, where [`check`] reads the log's current
/// one; nothing is kept here: [`remember`] then holds the checkpoint to the
/// caller's memory of the log.
pub fn check_offline(
    reference: &[u8],
    proof_text: &str,
    policy: &Policy,
    now: u64,
) -> Result<Logged, String> {
    let Proven {
        proof,
        cp,
        cosigned,
        ..
    } = proven(reference, proof_text, policy)?;
    let fresh = policy.fresh(&cosigned, now);
    if fresh.len() < policy.quorum {
        return Err(format!(
            "{}; this verifier reads no log to refresh it",
            stale(policy, &cosigned, fresh.len())
        ));
    }
    Ok(Logged {
        origin: cp.origin.clone(),
        index: proof.index,
        size: cp.size,
        root_hex: tlog::hex(&cp.root),
        cosigned_by: fresh,
        refreshed: false,
    })
}

/// Why a proof's checkpoint is refused against the caller's memory of the
/// log (`remember`). Each names its step, and the relay's exit for a log
/// refusal is theirs ([`EXIT_NOT_LOGGED`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Inconsistent {
    /// The remembered note is not a checkpoint of the pinned log: not
    /// signed by it, another origin, unreadable. Nothing is admitted on it.
    Memory(String),
    /// The checkpoint is smaller than the one remembered: the log went
    /// back, or the proof is older than what this caller verified.
    Rollback { seen: u64, presented: u64 },
    /// Two checkpoints of one size, signed by the log, with two roots.
    Fork {
        size: u64,
        seen_root_hex: String,
        presented_root_hex: String,
    },
    /// The log grew: the consistency proof is missing, malformed, from
    /// another size, to another checkpoint, or false (a fork, or a proof
    /// made up).
    Proof(String),
}

impl Inconsistent {
    pub fn step(&self) -> &'static str {
        match self {
            Inconsistent::Memory(_) => "log-memory",
            Inconsistent::Rollback { .. } => "log-rollback",
            Inconsistent::Fork { .. } => "log-fork",
            Inconsistent::Proof(_) => "log-consistency",
        }
    }
}

impl std::fmt::Display for Inconsistent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Inconsistent::Memory(e) => write!(f, "the remembered checkpoint: {e}"),
            Inconsistent::Rollback { seen, presented } => write!(
                f,
                "the log went back: the proof's checkpoint has size {presented}, the last one \
                 verified here has size {seen}; a proof against the log's current checkpoint \
                 is needed"
            ),
            Inconsistent::Fork {
                size,
                seen_root_hex,
                presented_root_hex,
            } => write!(
                f,
                "the log forked: two signed checkpoints of size {size}, root {seen_root_hex} \
                 seen before, root {presented_root_hex} now"
            ),
            Inconsistent::Proof(e) => {
                write!(f, "consistency with the last checkpoint verified here: {e}")
            }
        }
    }
}

/// What `remember` settled: the checkpoint to keep, as its signed note, and
/// the size remembered before (None the first time).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Remembered {
    pub note: String,
    pub size: u64,
    pub root_hex: String,
    pub before: Option<u64>,
}

/// The caller's memory of the log, for a verifier that reads no log
/// (x0ne-sdk): `memory` is the signed checkpoint note it kept from its last
/// admission (None the first time), `proof_text` the reference's proof,
/// already accepted by [`check_offline`], `consistency` a tlog-witness
/// `add-checkpoint` body from the remembered size to the proof's
/// checkpoint, needed once the log has grown. As a witness (tlog-witness):
/// the same size must show the same root (a fork otherwise), a smaller
/// one is refused (the log went back, or the proof is stale), a larger one
/// is admitted on a consistency proof whose `old` is the remembered size and
/// whose checkpoint is the proof's; from no memory, any checkpoint of the
/// pinned log is kept, and a proof given then must be empty. The caller
/// keeps the note returned and hands it back next time; this function
/// reads and writes nothing.
pub fn remember(
    policy: &Policy,
    memory: Option<&str>,
    proof_text: &str,
    consistency: Option<&str>,
) -> Result<Remembered, Inconsistent> {
    let proof = tlog::parse_tlog_proof(proof_text).map_err(Inconsistent::Proof)?;
    let (cp, _) = policy
        .open(&proof.checkpoint)
        .map_err(|e| Inconsistent::Proof(format!("the proof's checkpoint: {e}")))?;
    let seen = match memory {
        None => None,
        Some(note) => {
            if note.len() > tlog::MAX_NOTE_BYTES {
                return Err(Inconsistent::Memory(format!(
                    "{} bytes, cap {}",
                    note.len(),
                    tlog::MAX_NOTE_BYTES
                )));
            }
            Some(policy.open(note).map_err(Inconsistent::Memory)?.0)
        }
    };
    let kept = |before: Option<u64>| Remembered {
        note: proof.checkpoint.clone(),
        size: cp.size,
        root_hex: tlog::hex(&cp.root),
        before,
    };
    // The consistency body, when given: from the remembered size (0 with no
    // memory), to the proof's checkpoint, verified with RFC 9162 §2.1.4.2.
    let (old_size, old_root) = match &seen {
        None => (0, tlog::empty_root()),
        Some(s) => (s.size, s.root),
    };
    let check_body = |body: &str| -> Result<(), Inconsistent> {
        let b = tlog::parse_add_checkpoint(body).map_err(Inconsistent::Proof)?;
        if b.old != old_size {
            return Err(Inconsistent::Proof(format!(
                "a proof from size {}, the last checkpoint verified here has size {old_size}",
                b.old
            )));
        }
        let (to, _) = policy
            .open(&b.checkpoint)
            .map_err(|e| Inconsistent::Proof(format!("the proof's own checkpoint: {e}")))?;
        if to.size != cp.size || to.root != cp.root {
            return Err(Inconsistent::Proof(format!(
                "a proof to size {} root {}, the reference's checkpoint is size {} root {}",
                to.size,
                tlog::hex(&to.root),
                cp.size,
                tlog::hex(&cp.root)
            )));
        }
        tlog::verify_consistency(old_size, cp.size, &b.hashes, &old_root, &cp.root)
            .map_err(Inconsistent::Proof)
    };
    let Some(seen) = seen else {
        if let Some(body) = consistency {
            check_body(body)?;
        }
        return Ok(kept(None));
    };
    if cp.size < seen.size {
        return Err(Inconsistent::Rollback {
            seen: seen.size,
            presented: cp.size,
        });
    }
    if cp.size == seen.size {
        if cp.root != seen.root {
            return Err(Inconsistent::Fork {
                size: cp.size,
                seen_root_hex: tlog::hex(&seen.root),
                presented_root_hex: tlog::hex(&cp.root),
            });
        }
        if let Some(body) = consistency {
            check_body(body)?;
        }
        return Ok(kept(Some(seen.size)));
    }
    let Some(body) = consistency else {
        return Err(Inconsistent::Proof(format!(
            "the log grew from {} to {}: a consistency proof from {} is required, none was given",
            seen.size, cp.size, seen.size
        )));
    };
    check_body(body)?;
    Ok(kept(Some(seen.size)))
}

/// The log check of one reference's bytes, its proof, the log's files and
/// this machine's last verified checkpoint under `state_dir`.
#[cfg(not(x0ne_sdk))]
pub fn check(
    reference: &[u8],
    proof_text: &str,
    policy: &Policy,
    log: &dyn TileSource,
    state_dir: &Path,
    now: u64,
) -> Result<Logged, String> {
    let Proven {
        proof,
        mut cp,
        cosigned,
        leaf,
    } = proven(reference, proof_text, policy)?;
    let mut fresh = policy.fresh(&cosigned, now);
    let mut note = proof.checkpoint.clone();
    let mut refreshed = false;
    if fresh.len() < policy.quorum {
        let why = stale(policy, &cosigned, fresh.len());
        let current = log
            .checkpoint()
            .map_err(|e| format!("{why}, and the log's current checkpoint is unreadable: {e}"))?;
        let (newer, newer_cosigned) = policy
            .open(&current)
            .map_err(|e| format!("{why}; the log's current checkpoint: {e}"))?;
        fresh = policy.fresh(&newer_cosigned, now);
        if fresh.len() < policy.quorum {
            return Err(format!(
                "{why}; the log's current checkpoint has {} fresh cosignatures",
                fresh.len()
            ));
        }
        if newer.size < cp.size {
            return Err(format!(
                "the log's current checkpoint ({}) is smaller than the proof's ({})",
                root_line(&newer),
                root_line(&cp)
            ));
        }
        let mut t = tlog::TileHashes::new(log, newer.size);
        let consistency = tlog::consistency_proof(&mut t, cp.size, newer.size)?;
        tlog::verify_consistency(cp.size, newer.size, &consistency, &cp.root, &newer.root)?;
        let path = tlog::inclusion_proof(&mut t, proof.index, newer.size)?;
        tlog::verify_inclusion(&leaf, proof.index, newer.size, &path, &newer.root)
            .map_err(|e| format!("this reference is not shown in the log: {e}"))?;
        cp = newer;
        note = current;
        refreshed = true;
    }
    consistent_with_last_seen(policy, &cp, &note, log, state_dir, now)?;
    Ok(Logged {
        origin: cp.origin.clone(),
        index: proof.index,
        size: cp.size,
        root_hex: tlog::hex(&cp.root),
        cosigned_by: fresh,
        refreshed,
    })
}

/// The file of the last checkpoint this machine verified for a log.
#[cfg(not(x0ne_sdk))]
pub fn state_file(state_dir: &Path, origin: &str) -> PathBuf {
    let id = crate::cloud_config::hex(&crate::cloud_config::sha256(origin.as_bytes()));
    state_dir
        .join("tlog")
        .join(format!("{}.checkpoint", &id[..16]))
}

#[cfg(not(x0ne_sdk))]
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let dir = path.parent().ok_or("state file without a directory")?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

/// The checkpoint against the last verified one: same size, same root;
/// otherwise a consistency proof from the log's tiles. A fork keeps both
/// signed notes as evidence and refuses; the larger checkpoint is kept.
#[cfg(not(x0ne_sdk))]
fn consistent_with_last_seen(
    policy: &Policy,
    cp: &Checkpoint,
    note: &str,
    log: &dyn TileSource,
    state_dir: &Path,
    now: u64,
) -> Result<(), String> {
    let file = state_file(state_dir, &policy.origin);
    let last = match crate::fsread::read_capped(&file, tlog::MAX_NOTE_BYTES as u64) {
        Ok(b) => Some(String::from_utf8(b).map_err(|_| format!("{}: not UTF-8", file.display()))?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(format!("{}: {e}", file.display())),
    };
    let Some(last_note) = last else {
        return write_atomic(&file, note.as_bytes());
    };
    let (seen, _) = policy
        .open(&last_note)
        .map_err(|e| format!("{}: {e}", file.display()))?;
    // Two signed heads of one size and two roots prove a fork; a consistency
    // proof from the log's tiles that fails shows a fork or tiles that do not
    // match the log's own signed heads. Refused either way, both heads kept.
    let fork = |lead: &str, why: String| -> String {
        let evidence = file.with_file_name(format!("fork-{now}.txt"));
        let body = format!("{lead}: {why}\n\n# last verified\n{last_note}\n# presented\n{note}");
        let kept = write_atomic(&evidence, body.as_bytes())
            .map(|_| format!("; both signed checkpoints kept in {}", evidence.display()))
            .unwrap_or_default();
        format!("{lead}: {why}{kept}")
    };
    if seen.size == cp.size {
        if seen.root != cp.root {
            return Err(fork(
                "the log forked",
                format!(
                    "two signed checkpoints of size {}: root {} seen before, root {} now",
                    cp.size,
                    tlog::hex(&seen.root),
                    tlog::hex(&cp.root)
                ),
            ));
        }
        return Ok(());
    }
    let (small, large) = if seen.size < cp.size {
        (&seen, cp)
    } else {
        (cp, &seen)
    };
    let mut t = tlog::TileHashes::new(log, large.size);
    let proof = tlog::consistency_proof(&mut t, small.size, large.size).map_err(|e| {
        format!(
            "this checkpoint ({}) cannot be compared with the last one verified here ({}): {e}",
            root_line(cp),
            root_line(&seen)
        )
    })?;
    tlog::verify_consistency(small.size, large.size, &proof, &small.root, &large.root).map_err(
        |e| {
            fork(
                "the log forked, or its tiles do not match its signed checkpoints",
                format!("{} then {}: {e}", root_line(&seen), root_line(cp)),
            )
        },
    )?;
    if cp.size > seen.size {
        write_atomic(&file, note.as_bytes())?;
    }
    Ok(())
}

/// The log check of a reference file: its `.tlog-proof`, read next to it.
#[cfg(not(x0ne_sdk))]
pub fn check_file(
    reference_path: &Path,
    reference: &[u8],
    anchor: &Anchor,
    state_dir: &Path,
    now: u64,
) -> Result<Logged, String> {
    let policy = anchor.policy()?;
    let proof_path = proof_path(reference_path);
    let proof = crate::fsread::read_capped(&proof_path, MAX_PROOF_BYTES)
        .map_err(|e| format!("no proof of inclusion, {}: {e}", proof_path.display()))?;
    let proof =
        String::from_utf8(proof).map_err(|_| format!("{}: not UTF-8", proof_path.display()))?;
    let source = policy.source(reference_path);
    check(reference, &proof, policy, source.as_ref(), state_dir, now)
        .map_err(|e| format!("{}: {e}", proof_path.display()))
}

#[cfg(test)]
impl Anchor {
    pub(crate) fn of(policy: Policy) -> Self {
        Anchor(Ok(policy))
    }
    pub(crate) fn none() -> Self {
        Anchor(Err(NO_LOG.into()))
    }
}

/// The TEST log, pinned on a bench: x0ne-sdk's `test-anchor` builds only.
#[cfg(x0ne_sdk_test_anchor)]
impl Anchor {
    pub(crate) fn test() -> Self {
        Anchor(Ok(test_log::policy()))
    }
}

/// The TEST log and its TEST witness: keys derived from public strings, so
/// anyone can sign with them. Compiled in every `test-archive-key` build,
/// pinned only by a debug one (`Anchor::production`). `LocalLog` is a log on
/// disk in the tlog-tiles layout, for tests and local rehearsals.
#[cfg(any(test, feature = "test-archive-key"))]
pub mod test_log {
    #[cfg(not(x0ne_sdk))]
    pub use super::test_log_files::{cosign, log_reference, LocalLog};
    use super::*;
    use ed25519_dalek::SigningKey;
    use sha2::{Digest, Sha256};

    pub const ORIGIN: &str = "test.invalid/x0ne-engine/tlog";
    pub const WITNESS: &str = "test.invalid/x0ne-engine/witness";
    /// Ten years: the TEST proofs committed as fixtures stay fresh. The
    /// freshness check is tested on a one-day policy (`tests`).
    pub const MAX_AGE_S: u64 = 3650 * 24 * 3600;

    pub fn secret(label: &str) -> SigningKey {
        SigningKey::from_bytes(
            &Sha256::digest(format!(
                "x0ne-engine TEST transparency log, not for release: {label}"
            ))
            .into(),
        )
    }

    pub fn log_key() -> NoteKey {
        NoteKey::new(
            ORIGIN,
            KeyKind::Ed25519,
            secret("log").verifying_key().to_bytes(),
        )
        .expect("TEST log key")
    }

    pub fn witness_key() -> NoteKey {
        NoteKey::new(
            WITNESS,
            KeyKind::CosignatureV1,
            secret("witness").verifying_key().to_bytes(),
        )
        .expect("TEST witness key")
    }

    pub fn policy() -> Policy {
        Policy {
            origin: ORIGIN.into(),
            log: log_key(),
            witnesses: vec![witness_key()],
            quorum: 1,
            max_age_s: MAX_AGE_S,
            tiles: Tiles::BesideReference,
        }
    }

    /// A proof of `reference` in a TEST log that holds it alone: the
    /// checkpoint signed by the TEST log and cosigned by the TEST witness at
    /// `time`, the empty inclusion path. What `log_reference` writes for a
    /// log's first entry, in memory: x0ne-sdk's benches read no file.
    pub fn proof_of_one(reference: &[u8], time: u64) -> String {
        let mut log = MemoryLog::new();
        log.append(reference);
        log.proof(0, time)
    }

    /// The TEST log in memory: its leaves, and what it signs at any size.
    /// x0ne-sdk's tests grow it, fork it (a second one with other entries,
    /// under the same TEST key) and take its consistency proofs.
    #[derive(Debug, Clone, Default)]
    pub struct MemoryLog {
        leaves: Vec<tlog::Hash>,
    }

    impl MemoryLog {
        pub fn new() -> Self {
            Self::default()
        }

        /// Logs a reference's bytes; its index.
        pub fn append(&mut self, reference: &[u8]) -> u64 {
            self.leaves.push(tlog::leaf_hash(&entry(reference)));
            self.leaves.len() as u64 - 1
        }

        pub fn size(&self) -> u64 {
            self.leaves.len() as u64
        }

        /// The checkpoint of the current size, signed by the TEST log and
        /// cosigned by the TEST witness at `time`.
        pub fn checkpoint_note(&self, time: u64) -> String {
            let text = Checkpoint {
                origin: ORIGIN.into(),
                size: self.size(),
                root: tlog::mth_of(&self.leaves),
                extensions: vec![],
            }
            .text();
            format!(
                "{text}\n{}{}",
                tlog::sign_line(&text, &log_key(), &secret("log"), 0),
                tlog::sign_line(&text, &witness_key(), &secret("witness"), time)
            )
        }

        /// `<reference>.tlog-proof` of the entry at `index`, against the
        /// current checkpoint.
        pub fn proof(&self, index: u64, time: u64) -> String {
            let hashes = tlog::inclusion_proof(&mut tlog::Leaves(&self.leaves), index, self.size())
                .expect("an index of this log");
            tlog::format_tlog_proof(index, &hashes, &self.checkpoint_note(time))
        }

        /// The tlog-witness `add-checkpoint` body from size `old` to the
        /// current checkpoint, cosigned at `time`.
        pub fn consistency(&self, old: u64, time: u64) -> String {
            let hashes = tlog::consistency_proof(&mut tlog::Leaves(&self.leaves), old, self.size())
                .expect("a size this log passed");
            tlog::format_add_checkpoint(old, &hashes, &self.checkpoint_note(time))
        }
    }
}

/// The TEST log on disk: what `test_log` re-exports outside x0ne-sdk,
/// which reads and writes no file.
#[cfg(all(any(test, feature = "test-archive-key"), not(x0ne_sdk)))]
pub mod test_log_files {
    use super::test_log::{secret, witness_key, ORIGIN};
    use super::*;
    use ed25519_dalek::SigningKey;

    /// A log in `dir`: `leaves` (32-byte leaf hashes, appended), `tile/`,
    /// `checkpoint`. Signed by `log_secret` under `origin`.
    pub struct LocalLog {
        pub dir: PathBuf,
        pub origin: String,
        pub log: NoteKey,
        log_secret: SigningKey,
        pub leaves: Vec<tlog::Hash>,
    }

    impl LocalLog {
        /// The TEST log in `dir`, with what it already holds.
        pub fn open(dir: &Path) -> Result<Self, String> {
            Self::open_as(dir, ORIGIN, secret("log"))
        }

        pub fn open_as(dir: &Path, origin: &str, log_secret: SigningKey) -> Result<Self, String> {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            let leaves = match std::fs::read(dir.join("leaves")) {
                Ok(b) if b.len().is_multiple_of(32) => b.as_chunks::<32>().0.to_vec(),
                Ok(_) => return Err("leaves: not a list of 32-byte hashes".into()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
                Err(e) => return Err(format!("leaves: {e}")),
            };
            let log = NoteKey::new(
                origin,
                KeyKind::Ed25519,
                log_secret.verifying_key().to_bytes(),
            )?;
            Ok(LocalLog {
                dir: dir.to_path_buf(),
                origin: origin.into(),
                log,
                log_secret,
                leaves,
            })
        }

        pub fn append(&mut self, entry: &[u8]) -> Result<u64, String> {
            self.leaves.push(tlog::leaf_hash(entry));
            let bytes: Vec<u8> = self.leaves.iter().flatten().copied().collect();
            std::fs::write(self.dir.join("leaves"), bytes).map_err(|e| format!("leaves: {e}"))?;
            Ok(self.leaves.len() as u64 - 1)
        }

        pub fn size(&self) -> u64 {
            self.leaves.len() as u64
        }

        /// The tiles of the current size (partial ones included), and the
        /// checkpoint text, signed by the log alone.
        pub fn publish(&self) -> Result<String, String> {
            let mut level: Vec<tlog::Hash> = self.leaves.clone();
            let mut lvl = 0u32;
            while !level.is_empty() {
                for (n, chunk) in level.chunks(tlog::TILE_WIDTH as usize).enumerate() {
                    let p = self
                        .dir
                        .join(tlog::tile_path(lvl, n as u64, chunk.len() as u64));
                    std::fs::create_dir_all(p.parent().expect("tile dir"))
                        .map_err(|e| format!("{}: {e}", p.display()))?;
                    let bytes: Vec<u8> = chunk.iter().flatten().copied().collect();
                    std::fs::write(&p, bytes).map_err(|e| format!("{}: {e}", p.display()))?;
                }
                level = level
                    .as_chunks::<{ tlog::TILE_WIDTH as usize }>()
                    .0
                    .iter()
                    .map(|c| tlog::mth_of(c.as_slice()))
                    .collect();
                lvl += 1;
            }
            let cp = Checkpoint {
                origin: self.origin.clone(),
                size: self.size(),
                root: tlog::mth_of(&self.leaves),
                extensions: vec![],
            };
            let text = cp.text();
            let note = format!(
                "{text}\n{}",
                tlog::sign_line(&text, &self.log, &self.log_secret, 0)
            );
            write_atomic(&self.dir.join("checkpoint"), note.as_bytes())?;
            Ok(note)
        }

        pub fn proof(&self, index: u64, checkpoint_note: &str) -> Result<String, String> {
            let n = tlog::parse_note(checkpoint_note)?;
            let cp = tlog::parse_checkpoint(&n.text)?;
            let leaves = &self.leaves[..cp.size as usize];
            let hashes = tlog::inclusion_proof(&mut tlog::Leaves(leaves), index, cp.size)?;
            Ok(tlog::format_tlog_proof(index, &hashes, checkpoint_note))
        }
    }

    /// The TEST witness adds its cosignature at `time` to a checkpoint note.
    /// It verifies the log's signature, and consistency with the last
    /// checkpoint it cosigned (`<witness_dir>/witness-state`) from the tiles
    /// the log serves: a fork is refused.
    pub fn cosign(
        witness_dir: &Path,
        log_tiles: &dyn TileSource,
        note: &str,
        log: &NoteKey,
        time: u64,
    ) -> Result<String, String> {
        let n = tlog::parse_note(note)?;
        tlog::open_note(&n, std::slice::from_ref(log)).map_err(|e| format!("witness: {e}"))?;
        let cp = tlog::parse_checkpoint(&n.text)?;
        let state = witness_dir.join("witness-state");
        if let Ok(prev) = std::fs::read_to_string(&state) {
            let prev = tlog::parse_checkpoint(&prev)?;
            if prev.size > cp.size {
                return Err("witness: refused, the log went back".into());
            }
            let mut t = tlog::TileHashes::new(log_tiles, cp.size);
            tlog::consistency_proof(&mut t, prev.size, cp.size)
                .and_then(|p| {
                    tlog::verify_consistency(prev.size, cp.size, &p, &prev.root, &cp.root)
                })
                .map_err(|e| format!("witness: refused, {e}"))?;
        }
        write_atomic(&state, n.text.as_bytes())?;
        Ok(format!(
            "{note}{}",
            tlog::sign_line(&n.text, &witness_key(), &secret("witness"), time)
        ))
    }

    /// Log a reference file in the TEST log next to it (`<dir>/tlog/`),
    /// cosign at `time`, and write `<reference>.tlog-proof`.
    pub fn log_reference(reference: &Path, time: u64) -> Result<u64, String> {
        let bytes =
            std::fs::read(reference).map_err(|e| format!("{}: {e}", reference.display()))?;
        let dir = reference.parent().unwrap_or(Path::new(".")).join("tlog");
        let mut log = LocalLog::open(&dir)?;
        let index = log.append(&entry(&bytes))?;
        let note = log.publish()?;
        let note = cosign(&dir, &DirTiles(dir.clone()), &note, &log.log, time)?;
        write_atomic(&dir.join("checkpoint"), note.as_bytes())?;
        write_atomic(&proof_path(reference), log.proof(index, &note)?.as_bytes())?;
        Ok(index)
    }
}

#[cfg(all(test, not(x0ne_sdk)))]
mod tests {
    use super::test_log::*;
    use super::*;

    const NOW: u64 = 1_792_627_200; // 2026-10-22T00:00:00Z
    const DAY: u64 = 24 * 3600;

    /// The TEST log with a one-day freshness window.
    fn day() -> Policy {
        Policy {
            max_age_s: DAY,
            ..policy()
        }
    }

    fn setup(n: usize) -> (tempfile::TempDir, Vec<PathBuf>) {
        let dir = tempfile::tempdir().unwrap();
        let refs = (0..n)
            .map(|i| {
                let p = dir.path().join(format!("x0ne-cvm-0.{i}.0.json"));
                std::fs::write(&p, format!("{{\"reference\": {i}}}")).unwrap();
                log_reference(&p, NOW - 60).unwrap();
                p
            })
            .collect();
        (dir, refs)
    }

    fn state(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("state")
    }

    fn checked(p: &Path, dir: &tempfile::TempDir, now: u64) -> Result<Logged, String> {
        let bytes = std::fs::read(p).unwrap();
        check_file(p, &bytes, &Anchor::of(day()), &state(dir), now)
    }

    #[test]
    fn a_logged_reference_passes_and_names_its_checkpoint() {
        let (dir, refs) = setup(3);
        let l = checked(&refs[1], &dir, NOW).unwrap();
        assert_eq!((l.index, l.size, l.refreshed), (1, 2, false), "{l:?}");
        assert_eq!(l.cosigned_by, [(WITNESS.to_string(), NOW - 60)]);
        // The proof of the third holds the size-3 checkpoint, consistent with size 2.
        let l = checked(&refs[2], &dir, NOW).unwrap();
        assert_eq!((l.index, l.size), (2, 3));
        // An older proof after a newer one: proven consistent from the tiles.
        checked(&refs[0], &dir, NOW).unwrap();
    }

    /// SECURITY: no proof, a proof of another reference (a valid proof of
    /// another entry, CVE-2026-22703 in Cosign), changed bytes, a changed
    /// index or hash: refused.
    #[test]
    fn a_reference_absent_from_the_log_is_refused() {
        let (dir, refs) = setup(2);
        let unlogged = dir.path().join("x0ne-cvm-9.9.9.json");
        std::fs::write(&unlogged, b"{\"reference\": \"targeted\"}").unwrap();
        let err = checked(&unlogged, &dir, NOW).unwrap_err();
        assert!(err.contains("no proof of inclusion"), "{err}");
        std::fs::copy(proof_path(&refs[0]), proof_path(&unlogged)).unwrap();
        let err = checked(&unlogged, &dir, NOW).unwrap_err();
        assert!(err.contains("not shown in the log"), "{err}");
        // The logged file, one byte changed.
        let mut b = std::fs::read(&refs[1]).unwrap();
        b[3] ^= 1;
        std::fs::write(&refs[1], &b).unwrap();
        assert!(checked(&refs[1], &dir, NOW)
            .unwrap_err()
            .contains("not shown in the log"));
    }

    #[test]
    fn a_false_inclusion_proof_is_refused() {
        let (dir, refs) = setup(5);
        let good = std::fs::read_to_string(proof_path(&refs[3])).unwrap();
        for (what, bad) in [
            ("index", good.replacen("index 3", "index 2", 1)),
            ("hash", {
                let lines: Vec<&str> = good.lines().collect();
                good.replacen(lines[2], "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=", 1)
            }),
            ("short", {
                let lines: Vec<&str> = good.lines().collect();
                good.replacen(&format!("{}\n", lines[3]), "", 1)
            }),
        ] {
            std::fs::write(proof_path(&refs[3]), &bad).unwrap();
            let err = checked(&refs[3], &dir, NOW).unwrap_err();
            assert!(err.contains("not shown in the log"), "{what}: {err}");
        }
    }

    /// SECURITY: a checkpoint without the log's signature, signed by another
    /// key under the log's name, or for another origin: refused.
    #[test]
    fn an_unsigned_or_foreign_checkpoint_is_refused() {
        let (dir, refs) = setup(1);
        let good = std::fs::read_to_string(proof_path(&refs[0])).unwrap();
        let log_line = good
            .lines()
            .find(|l| l.starts_with(&format!("\u{2014} {ORIGIN} ")))
            .unwrap();
        let unsigned = good.replacen(&format!("{log_line}\n"), "", 1);
        std::fs::write(proof_path(&refs[0]), &unsigned).unwrap();
        let err = checked(&refs[0], &dir, NOW).unwrap_err();
        assert!(err.contains("not signed by the pinned log"), "{err}");
        // A rogue log key, same origin, same witness: the impostor's own log.
        let rogue_dir = dir.path().join("rogue");
        let mut rogue = LocalLog::open_as(&rogue_dir, ORIGIN, secret("rogue")).unwrap();
        let bytes = std::fs::read(&refs[0]).unwrap();
        rogue.append(&entry(&bytes)).unwrap();
        let note = rogue.publish().unwrap();
        let note = format!(
            "{note}{}",
            tlog::sign_line(
                &tlog::parse_note(&note).unwrap().text,
                &witness_key(),
                &secret("witness"),
                NOW
            )
        );
        std::fs::write(proof_path(&refs[0]), rogue.proof(0, &note).unwrap()).unwrap();
        let err = checked(&refs[0], &dir, NOW).unwrap_err();
        assert!(err.contains("not signed by the pinned log"), "{err}");
        // The TEST log's key, but another origin.
        let other = Policy {
            origin: "test.invalid/other".into(),
            ..day()
        };
        std::fs::write(proof_path(&refs[0]), &good).unwrap();
        let err = check_file(&refs[0], &bytes, &Anchor::of(other), &state(&dir), NOW).unwrap_err();
        assert!(err.contains("pinned log"), "{err}");
    }

    /// SECURITY: a checkpoint cosigned by no pinned witness, or only long
    /// ago, with no newer one to be read from the log: refused.
    #[test]
    fn an_uncosigned_or_stale_checkpoint_is_refused_unless_the_log_shows_a_fresh_one() {
        let (dir, refs) = setup(2);
        let good = std::fs::read_to_string(proof_path(&refs[1])).unwrap();
        let witness_line = good
            .lines()
            .find(|l| l.starts_with(&format!("\u{2014} {WITNESS} ")))
            .unwrap();
        let tlog = dir.path().join("tlog");
        let log_only = std::fs::read_to_string(tlog.join("checkpoint"))
            .unwrap()
            .replacen(&format!("{witness_line}\n"), "", 1);
        // Not cosigned, and the log serves no cosigned checkpoint either.
        std::fs::write(
            proof_path(&refs[1]),
            good.replacen(&format!("{witness_line}\n"), "", 1),
        )
        .unwrap();
        std::fs::write(tlog.join("checkpoint"), &log_only).unwrap();
        let err = checked(&refs[1], &dir, NOW).unwrap_err();
        assert!(
            err.contains("cosigned by 0 of the pinned witnesses"),
            "{err}"
        );
        // Cosigned two days ago, the log unreadable.
        std::fs::write(proof_path(&refs[1]), &good).unwrap();
        std::fs::remove_file(tlog.join("checkpoint")).unwrap();
        let later = NOW + 2 * DAY;
        let err = checked(&refs[1], &dir, later).unwrap_err();
        assert!(err.contains("checkpoint stale"), "{err}");
        assert!(err.contains("unreadable"), "{err}");
        // Cosigned in the future beyond the skew: not counted either.
        let err = checked(&refs[1], &dir, NOW - 60 - FUTURE_SKEW_S - 1).unwrap_err();
        assert!(err.contains("stale"), "{err}");
        // The log shows a checkpoint cosigned now: the reference passes on it.
        let log = LocalLog::open(&tlog).unwrap();
        let note = cosign(
            &tlog,
            &DirTiles(tlog.clone()),
            &log.publish().unwrap(),
            &log.log,
            later,
        )
        .unwrap();
        std::fs::write(tlog.join("checkpoint"), note).unwrap();
        let l = checked(&refs[1], &dir, later).unwrap();
        assert!(l.refreshed, "{l:?}");
        assert_eq!(l.cosigned_by, [(WITNESS.to_string(), later)]);
    }

    /// SECURITY: two signed heads that are not a prefix of each other. Same
    /// size, two roots; or a larger head whose log rewrote what this
    /// machine saw: refused, both kept as evidence.
    #[test]
    fn a_forked_log_is_refused_and_both_heads_are_kept() {
        let (dir, refs) = setup(2);
        checked(&refs[1], &dir, NOW).unwrap();
        // The log's key signs another tree of size 2 for a targeted reference.
        let target = dir.path().join("x0ne-cvm-6.6.6.json");
        std::fs::write(&target, b"{\"reference\": \"targeted\"}").unwrap();
        let fork_dir = dir.path().join("fork");
        let mut fork = LocalLog::open(&fork_dir).unwrap();
        fork.append(&entry(&std::fs::read(&refs[0]).unwrap()))
            .unwrap();
        fork.append(&entry(&std::fs::read(&target).unwrap()))
            .unwrap();
        let note = fork.publish().unwrap();
        let note = format!(
            "{note}{}",
            tlog::sign_line(
                &tlog::parse_note(&note).unwrap().text,
                &witness_key(),
                &secret("witness"),
                NOW
            )
        );
        std::fs::write(proof_path(&target), fork.proof(1, &note).unwrap()).unwrap();
        let err = checked(&target, &dir, NOW).unwrap_err();
        assert!(
            err.contains("the log forked: two signed checkpoints of size 2"),
            "{err}"
        );
        let evidence: Vec<_> = std::fs::read_dir(state(&dir).join("tlog"))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.starts_with("fork-"))
            .collect();
        assert_eq!(evidence.len(), 1, "{evidence:?}");
        // A larger forked head (size 3, rewriting index 1), its tiles served.
        fork.append(b"padding").unwrap();
        let note = fork.publish().unwrap();
        let note = format!(
            "{note}{}",
            tlog::sign_line(
                &tlog::parse_note(&note).unwrap().text,
                &witness_key(),
                &secret("witness"),
                NOW
            )
        );
        std::fs::write(proof_path(&target), fork.proof(1, &note).unwrap()).unwrap();
        let tlog_dir = dir.path().join("tlog");
        std::fs::rename(&tlog_dir, dir.path().join("honest")).unwrap();
        std::fs::rename(&fork_dir, &tlog_dir).unwrap();
        let err = checked(&target, &dir, NOW + 1).unwrap_err();
        assert!(
            err.contains(
                "the log forked, or its tiles do not match its signed checkpoints: size 2"
            ),
            "{err}"
        );
        // The TEST witness itself refuses to cosign that fork.
        let honest = dir.path().join("honest");
        let err = cosign(&honest, &DirTiles(tlog_dir.clone()), &note, &log_key(), NOW).unwrap_err();
        assert!(err.contains("witness: refused"), "{err}");
    }

    /// SECURITY: this machine remembers the largest head it verified. Seen
    /// at size 3, a size-4 head that keeps entries 0 and 1 but rewrites
    /// entry 2 is a fork, though it extends the size-2 head seen before it.
    #[test]
    fn a_rewrite_of_what_the_largest_seen_head_holds_is_refused() {
        let (dir, refs) = setup(3);
        checked(&refs[1], &dir, NOW).unwrap();
        checked(&refs[2], &dir, NOW).unwrap();
        let target = dir.path().join("x0ne-cvm-6.6.6.json");
        std::fs::write(&target, b"{\"reference\": \"targeted\"}").unwrap();
        let fork_dir = dir.path().join("fork");
        let mut fork = LocalLog::open(&fork_dir).unwrap();
        for r in &refs[..2] {
            fork.append(&entry(&std::fs::read(r).unwrap())).unwrap();
        }
        fork.append(&entry(&std::fs::read(&target).unwrap()))
            .unwrap();
        fork.append(b"padding").unwrap();
        let note = fork.publish().unwrap();
        let note = format!(
            "{note}{}",
            tlog::sign_line(
                &tlog::parse_note(&note).unwrap().text,
                &witness_key(),
                &secret("witness"),
                NOW
            )
        );
        std::fs::write(proof_path(&target), fork.proof(2, &note).unwrap()).unwrap();
        let tlog_dir = dir.path().join("tlog");
        std::fs::rename(&tlog_dir, dir.path().join("honest")).unwrap();
        std::fs::rename(&fork_dir, &tlog_dir).unwrap();
        let err = checked(&target, &dir, NOW + 1).unwrap_err();
        assert!(
            err.contains(
                "the log forked, or its tiles do not match its signed checkpoints: size 3"
            ),
            "{err}"
        );
    }

    /// SECURITY: a stale proof gives way to the log's current checkpoint only
    /// if that one extends it: a stale fork head that holds this reference at
    /// the same index is refused, on a machine that verified no head before.
    #[test]
    fn a_stale_head_of_a_fork_is_refused_though_the_log_holds_the_reference() {
        let (dir, refs) = setup(4);
        let fork_dir = dir.path().join("fork");
        let mut fork = LocalLog::open(&fork_dir).unwrap();
        for r in &refs[..2] {
            fork.append(&entry(&std::fs::read(r).unwrap())).unwrap();
        }
        fork.append(b"rewritten").unwrap();
        let note = fork.publish().unwrap();
        let note = format!(
            "{note}{}",
            tlog::sign_line(
                &tlog::parse_note(&note).unwrap().text,
                &witness_key(),
                &secret("witness"),
                NOW - 2 * DAY
            )
        );
        std::fs::write(proof_path(&refs[1]), fork.proof(1, &note).unwrap()).unwrap();
        let err = checked(&refs[1], &dir, NOW).unwrap_err();
        assert!(
            err.contains("size 3 root") && err.contains("is not a prefix of size 4 root"),
            "{err}"
        );
    }

    /// SECURITY: a quorum is of distinct witnesses. One witness that cosigns
    /// twice, at two times, must not count twice: the note parser refuses a
    /// second signature line of one key (sumdb/note skips it; neither counts
    /// it), so a quorum of two is not reached with one witness.
    #[test]
    fn a_quorum_counts_each_witness_once() {
        let (dir, refs) = setup(1);
        let second = NoteKey::new(
            "test.invalid/x0ne-engine/witness-2",
            KeyKind::CosignatureV1,
            secret("witness-2").verifying_key().to_bytes(),
        )
        .unwrap();
        let two = Policy {
            witnesses: vec![witness_key(), second.clone()],
            quorum: 2,
            ..day()
        };
        let bytes = std::fs::read(&refs[0]).unwrap();
        let proof = std::fs::read_to_string(proof_path(&refs[0])).unwrap();
        let text = tlog::parse_note(&tlog::parse_tlog_proof(&proof).unwrap().checkpoint)
            .unwrap()
            .text;
        // The first witness again, 30 s later: still one witness of two.
        let twice = format!(
            "{proof}{}",
            tlog::sign_line(&text, &witness_key(), &secret("witness"), NOW - 30)
        );
        std::fs::write(proof_path(&refs[0]), &twice).unwrap();
        let checked_by_two = |dir: &tempfile::TempDir| {
            check_file(&refs[0], &bytes, &Anchor::of(two.clone()), &state(dir), NOW)
        };
        let err = checked_by_two(&dir).unwrap_err();
        assert!(
            err.contains("two signature lines of test.invalid/x0ne-engine/witness"),
            "{err}"
        );
        // The first witness alone, once: one of two.
        std::fs::write(proof_path(&refs[0]), &proof).unwrap();
        let err = checked_by_two(&dir).unwrap_err();
        assert!(
            err.contains("cosigned by 1 of the pinned witnesses, 2 required"),
            "{err}"
        );
        // The second witness: the quorum, each named once.
        let both = format!(
            "{proof}{}",
            tlog::sign_line(&text, &second, &secret("witness-2"), NOW - 30)
        );
        std::fs::write(proof_path(&refs[0]), &both).unwrap();
        let l = check_file(&refs[0], &bytes, &Anchor::of(two), &state(&dir), NOW).unwrap();
        assert_eq!(l.cosigned_by.len(), 2, "{l:?}");
        assert_ne!(l.cosigned_by[0].0, l.cosigned_by[1].0, "{l:?}");
    }

    /// The release pin: none yet, and every reference is refused saying why.
    /// A pin without a witness, with a log key of another name, or with
    /// a plain HTTP location is refused.
    #[test]
    fn no_release_log_refuses_everything_and_pins_are_checked() {
        assert!(
            RELEASE.is_none(),
            "a pin landed: replace this test by one on it"
        );
        let (dir, refs) = setup(1);
        let bytes = std::fs::read(&refs[0]).unwrap();
        let err = check_file(&refs[0], &bytes, &Anchor::none(), &state(&dir), NOW).unwrap_err();
        assert!(err.contains("pins no transparency log"), "{err}");
        let log = log_key().vkey();
        let w = witness_key().vkey();
        let ws = [w.as_str()];
        let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
        let pin =
            |origin: &str, quorum: usize, tiles: &str, witnesses: &'static [&'static str]| Pin {
                origin: leak(origin.into()),
                log_key: leak(log.clone()),
                witnesses,
                quorum,
                max_age_s: 3600,
                tiles: leak(tiles.into()),
            };
        let ws: &'static [&'static str] = Box::leak(vec![leak(ws[0].into())].into_boxed_slice());
        assert!(pinned(&pin(ORIGIN, 1, "https://log.example/", ws)).is_ok());
        assert!(pinned(&pin(ORIGIN, 0, "https://log.example/", ws))
            .unwrap_err()
            .contains("quorum"));
        assert!(pinned(&pin(ORIGIN, 1, "https://log.example/", &[])).is_err());
        assert!(pinned(&pin(ORIGIN, 1, "http://log.example/", ws)).is_err());
        assert!(pinned(&pin("other.invalid/log", 1, "https://log.example/", ws)).is_err());
        // A pinned HTTPS log is read at its location and nowhere else; here
        // nothing answers (no network in the CI), and the refresh fails closed.
        let p = pinned(&pin(ORIGIN, 1, "https://log.example/", ws)).unwrap();
        let err = p.source(&refs[0]).checkpoint().unwrap_err();
        assert!(err.starts_with("https://log.example/checkpoint: "), "{err}");
        // A policy whose location is not https and not the loopback reads nothing.
        let p = Policy {
            tiles: Tiles::Https("http://log.example/".into()),
            ..day()
        };
        let err = p.source(&refs[0]).checkpoint().unwrap_err();
        assert!(err.contains("https only"), "{err}");
    }

    /// A pin of the TEST log, its HTTPS location, one hour, these witnesses.
    fn test_pin(witnesses: &[String], quorum: usize) -> Pin {
        let leak = |s: &str| -> &'static str { Box::leak(s.to_owned().into_boxed_str()) };
        let witnesses: Vec<&'static str> = witnesses.iter().map(|w| leak(w)).collect();
        Pin {
            origin: ORIGIN,
            log_key: leak(&log_key().vkey()),
            witnesses: Box::leak(witnesses.into_boxed_slice()),
            quorum,
            max_age_s: 3600,
            tiles: "https://log.example/",
        }
    }

    /// SECURITY: a cosignature/v1 does not sign its cosigner's name (C2SP
    /// tlog-cosignature): one key pinned under two names would count twice
    /// toward the quorum. tlog-policy forbids it; the pin is refused.
    #[test]
    fn a_witness_key_pinned_under_two_names_is_refused() {
        let key = |name: &str, label: &str| {
            NoteKey::new(
                name,
                KeyKind::CosignatureV1,
                secret(label).verifying_key().to_bytes(),
            )
            .unwrap()
            .vkey()
        };
        let alias = key("test.invalid/x0ne-engine/witness-alias", "witness");
        let other = key("test.invalid/x0ne-engine/witness-2", "witness-2");
        assert!(pinned(&test_pin(&[witness_key().vkey(), other], 2)).is_ok());
        let err = pinned(&test_pin(&[witness_key().vkey(), alias], 2)).unwrap_err();
        assert!(
            err.contains("witness-alias") && err.contains(WITNESS),
            "{err}"
        );
    }

    /// SECURITY: the log's own key pinned as a witness lets the log cosign
    /// its own checkpoints, a quorum of one met by the log alone. Refused.
    #[test]
    fn the_log_key_pinned_as_a_witness_is_refused() {
        let log_as_witness = NoteKey::new(
            WITNESS,
            KeyKind::CosignatureV1,
            secret("log").verifying_key().to_bytes(),
        )
        .unwrap();
        let err = pinned(&test_pin(&[log_as_witness.vkey()], 1)).unwrap_err();
        assert!(err.contains("the log's own key"), "{err}");
    }

    /// The lifecycle's real_relay fixture is in the TEST log, cosigned
    /// within the TEST window: the debug relay it runs accepts it.
    #[test]
    fn the_real_relay_fixture_is_in_the_test_log() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../x0ne-engine-cloud/tests/real_relay/x0ne-cvm-0.1.0.json");
        let bytes = std::fs::read(&p).unwrap();
        let state = tempfile::tempdir().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let l = check_file(&p, &bytes, &Anchor::of(policy()), state.path(), now).unwrap();
        assert_eq!((l.index, l.size, l.refreshed), (0, 1, false), "{l:?}");
    }

    #[test]
    fn only_tile_paths_are_read_from_a_log_directory() {
        let dir = tempfile::tempdir().unwrap();
        let src = DirTiles(dir.path().to_path_buf());
        for bad in [
            "../etc/passwd",
            "tile/../../x",
            "checkpoint",
            "/etc/passwd",
            "tile/0/A",
        ] {
            assert!(
                src.read(bad).unwrap_err().contains("not a tile path"),
                "{bad}"
            );
        }
    }
}
