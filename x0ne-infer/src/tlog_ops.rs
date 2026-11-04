//! Running the log of image references (`x0ne-engine-tlog`): the log's and
//! a witness's keys, a log of static files in the tlog-tiles layout signed
//! by one key, the tlog-witness `add-checkpoint` request and a witness that
//! answers it, the cosignatures added to the checkpoint, and the
//! `c2sp.org/tlog-proof@v1` file of each reference.
//!
//! The log's directory is its whole state: `checkpoint`, `tile/<L>/...` and
//! `tile/entries/...`, served as they are. Opening it reads every entry
//! again and refuses one whose root is not the checkpoint's. A tile already
//! written is never rewritten with other bytes, and the checkpoint is
//! written last, so a reader never sees a checkpoint without its tiles.
//!
//! Secret keys are files of mode 0600, in the signed-note form
//! `PRIVATE+KEY+<name>+<id>+<base64(type || seed)>` (Go's `note` package
//! for type 0x01; the same form with type 0x04 for a witness). The tool
//! refuses one that others can read and never prints one.

use crate::tlog::{self, Checkpoint, Hash, KeyKind, NoteKey};
use crate::transparency::{self, DirTiles};
use base64::Engine as _;
use ed25519_dalek::SigningKey;
use sha2::{Digest, Sha256};
use std::os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _};
use std::path::{Path, PathBuf};
use zeroize::Zeroizing;

const SECRET_PREFIX: &str = "PRIVATE+KEY+";
const MAX_SECRET_BYTES: u64 = 1024;
const MAX_ENTRY_BYTES: usize = u16::MAX as usize;

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// A key name the pin script takes: letters, digits and `._~:/-`.
pub fn check_name(name: &str) -> Result<(), String> {
    let ok = (1..=255).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._~:/-".contains(&b));
    if ok {
        Ok(())
    } else {
        Err(format!(
            "key name {name:?}: 1 to 255 letters, digits and ._~:/- only"
        ))
    }
}

/// SHA-256 of the verifier key's line, its first 40 hex digits in groups of
/// four: what is written on paper when the key is made, and compared when it
/// is pinned (`scripts/pin-transparency-log.sh` prints the same).
pub fn fingerprint(vkey: &str) -> String {
    let h = tlog::hex(&Sha256::digest(vkey.as_bytes())).to_uppercase();
    h.as_bytes()[..40]
        .chunks(4)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect::<Vec<_>>()
        .join(" ")
}

/// A secret key and its verifier key.
pub struct Secret {
    pub key: NoteKey,
    signing: SigningKey,
}

impl Secret {
    /// A new key from the kernel CSPRNG.
    pub fn generate(name: &str, kind: KeyKind) -> Result<Self, String> {
        use std::io::Read as _;
        check_name(name)?;
        let mut seed = Zeroizing::new([0u8; 32]);
        std::fs::File::open("/dev/urandom")
            .and_then(|mut f| f.read_exact(seed.as_mut()))
            .map_err(|e| format!("/dev/urandom: {e}"))?;
        let signing = SigningKey::from_bytes(&seed);
        let key = NoteKey::new(name, kind, signing.verifying_key().to_bytes())?;
        Ok(Secret { key, signing })
    }

    /// `PRIVATE+KEY+<name>+<id>+<base64(type || seed)>`.
    fn text(&self) -> Zeroizing<String> {
        let mut raw = Zeroizing::new(vec![match self.key.kind {
            KeyKind::Ed25519 => tlog::SIG_ED25519,
            KeyKind::CosignatureV1 => tlog::SIG_COSIGNATURE_V1,
        }]);
        raw.extend_from_slice(self.signing.as_bytes());
        Zeroizing::new(format!(
            "{SECRET_PREFIX}{}+{:08x}+{}\n",
            self.key.name,
            self.key.id,
            b64(&raw)
        ))
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let body = text.strip_suffix('\n').unwrap_or(text);
        let rest = body
            .strip_prefix(SECRET_PREFIX)
            .ok_or("secret key: not PRIVATE+KEY+<name>+<id>+<key>")?;
        let mut parts = rest.splitn(3, '+');
        let (Some(name), Some(id), Some(key)) = (parts.next(), parts.next(), parts.next()) else {
            return Err("secret key: not PRIVATE+KEY+<name>+<id>+<key>".into());
        };
        let raw = Zeroizing::new(tlog::b64_canonical(key).map_err(|_| "secret key: bad base64")?);
        let kind = match raw.first() {
            Some(&tlog::SIG_ED25519) => KeyKind::Ed25519,
            Some(&tlog::SIG_COSIGNATURE_V1) => KeyKind::CosignatureV1,
            _ => return Err(format!("secret key {name}: unknown type")),
        };
        let seed: Zeroizing<[u8; 32]> = Zeroizing::new(
            raw[1..]
                .try_into()
                .map_err(|_| format!("secret key {name}: not 32 bytes"))?,
        );
        let signing = SigningKey::from_bytes(&seed);
        let key = NoteKey::new(name, kind, signing.verifying_key().to_bytes())?;
        if format!("{:08x}", key.id) != id {
            return Err(format!("secret key {name}: key ID {id} is not the key's"));
        }
        Ok(Secret { key, signing })
    }

    /// Read from a file only its owner can read.
    pub fn read(path: &Path) -> Result<Self, String> {
        let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if meta.permissions().mode() & 0o077 != 0 {
            return Err(format!(
                "{}: readable by others (mode {:o}); chmod 600 it, and treat the key as exposed",
                path.display(),
                meta.permissions().mode() & 0o777
            ));
        }
        let bytes = Zeroizing::new(
            crate::fsread::read_capped(path, MAX_SECRET_BYTES)
                .map_err(|e| format!("{}: {e}", path.display()))?,
        );
        let text =
            std::str::from_utf8(&bytes).map_err(|_| format!("{}: not UTF-8", path.display()))?;
        Self::parse(text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Written to a new file of mode 0600; an existing file is never replaced.
    pub fn write_new(&self, path: &Path) -> Result<(), String> {
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::AlreadyExists => {
                    format!(
                        "{}: already exists; a key is never overwritten",
                        path.display()
                    )
                }
                _ => format!("{}: {e}", path.display()),
            })?;
        f.write_all(self.text().as_bytes())
            .and_then(|()| f.sync_all())
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    fn sign(&self, text: &str, time: u64) -> String {
        tlog::sign_line(text, &self.key, &self.signing, time)
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let dir = path.parent().ok_or("a file without a directory")?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

/// The time a cosignature/v1 signature carries (its first 8 bytes).
fn cosigned_at(sig: &tlog::NoteSig) -> u64 {
    sig.bytes
        .first_chunk::<8>()
        .map_or(0, |t| u64::from_be_bytes(*t))
}

/// A tile is written once: the same bytes again are fine, others are refused.
fn write_tile(path: &Path, bytes: &[u8]) -> Result<(), String> {
    match std::fs::read(path) {
        Ok(old) if old == bytes => Ok(()),
        Ok(_) => Err(format!(
            "{}: already written with other bytes; the log would rewrite its history",
            path.display()
        )),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => write_atomic(path, bytes),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// `tile/entries/<N>[.p/<W>]`.
fn bundle_path(index: u64, width: u64) -> String {
    let base = format!("tile/entries/{}", tlog::tile_index_path(index));
    if width == tlog::TILE_WIDTH {
        base
    } else {
        format!("{base}.p/{width}")
    }
}

/// A log of static files: its entries, their leaf hashes, its checkpoint.
pub struct Log {
    pub dir: PathBuf,
    /// The origin of its checkpoint; None for a log never published.
    pub origin: Option<String>,
    pub entries: Vec<Vec<u8>>,
    pub leaves: Vec<Hash>,
    /// The checkpoint as a signed note, cosignatures included.
    pub note: Option<String>,
}

impl Log {
    /// The log in `dir`, every entry read again and checked against the
    /// checkpoint's root. A directory without a checkpoint is an empty log.
    pub fn open(dir: &Path) -> Result<Self, String> {
        let cp_path = dir.join("checkpoint");
        let note = match crate::fsread::read_capped(&cp_path, tlog::MAX_NOTE_BYTES as u64) {
            Ok(b) => Some(
                String::from_utf8(b).map_err(|_| format!("{}: not UTF-8", cp_path.display()))?,
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(format!("{}: {e}", cp_path.display())),
        };
        let Some(note) = note else {
            if dir.join("tile").exists() {
                return Err(format!(
                    "{}: tiles without a checkpoint; not a log this tool wrote",
                    dir.display()
                ));
            }
            return Ok(Log {
                dir: dir.into(),
                origin: None,
                entries: Vec::new(),
                leaves: Vec::new(),
                note: None,
            });
        };
        let n = tlog::parse_note(&note).map_err(|e| format!("{}: {e}", cp_path.display()))?;
        let cp =
            tlog::parse_checkpoint(&n.text).map_err(|e| format!("{}: {e}", cp_path.display()))?;
        let mut entries = Vec::with_capacity(cp.size as usize);
        let mut index = 0u64;
        while (entries.len() as u64) < cp.size {
            let width = (cp.size - index * tlog::TILE_WIDTH).min(tlog::TILE_WIDTH);
            let rel = bundle_path(index, width);
            let path = dir.join(&rel);
            let bytes =
                crate::fsread::read_capped(&path, tlog::TILE_WIDTH * (2 + MAX_ENTRY_BYTES as u64))
                    .map_err(|e| format!("{}: {e}", path.display()))?;
            let mut rest = bytes.as_slice();
            for _ in 0..width {
                let (len, tail) = rest
                    .split_first_chunk::<2>()
                    .ok_or_else(|| format!("{rel}: fewer than {width} entries"))?;
                let len = u16::from_be_bytes(*len) as usize;
                if tail.len() < len {
                    return Err(format!("{rel}: an entry runs past the bundle"));
                }
                entries.push(tail[..len].to_vec());
                rest = &tail[len..];
            }
            if !rest.is_empty() {
                return Err(format!("{rel}: bytes after its {width} entries"));
            }
            index += 1;
        }
        let leaves: Vec<Hash> = entries.iter().map(|e| tlog::leaf_hash(e)).collect();
        let root = tlog::mth_of(&leaves);
        if root != cp.root {
            return Err(format!(
                "{}: its entries give the root {}, the checkpoint says {}; the log is not used",
                dir.display(),
                tlog::hex(&root),
                tlog::hex(&cp.root)
            ));
        }
        Ok(Log {
            dir: dir.into(),
            origin: Some(cp.origin),
            entries,
            leaves,
            note: Some(note),
        })
    }

    pub fn size(&self) -> u64 {
        self.leaves.len() as u64
    }

    /// The index of an entry, if the log holds it.
    pub fn find(&self, entry: &[u8]) -> Option<u64> {
        let leaf = tlog::leaf_hash(entry);
        self.leaves
            .iter()
            .position(|l| *l == leaf)
            .map(|i| i as u64)
    }

    /// Append entries, write their bundles and tiles, then the checkpoint
    /// signed by the log key alone. Returns the indexes and the note.
    pub fn append(
        &mut self,
        secret: &Secret,
        new: &[Vec<u8>],
    ) -> Result<(Vec<u64>, String), String> {
        if secret.key.kind != KeyKind::Ed25519 {
            return Err(format!(
                "{} is not a log key (a witness's cosignature/v1 key)",
                secret.key.name
            ));
        }
        if let Some(origin) = &self.origin {
            if *origin != secret.key.name {
                return Err(format!(
                    "the log's origin is {origin:?}, this key is named {:?}: a log is signed by the key named by its origin",
                    secret.key.name
                ));
            }
        }
        if new.is_empty() {
            return Err("nothing to append".into());
        }
        for (i, e) in new.iter().enumerate() {
            if e.len() > MAX_ENTRY_BYTES {
                return Err(format!("an entry of {} bytes", e.len()));
            }
            if let Some(at) = self.find(e) {
                return Err(format!(
                    "entry {} is already in the log at index {at}",
                    i + 1
                ));
            }
            if new[..i].contains(e) {
                return Err(format!("entry {} is given twice", i + 1));
            }
        }
        let first = self.size();
        let mut entries = self.entries.clone();
        entries.extend(new.iter().cloned());
        let leaves: Vec<Hash> = entries.iter().map(|e| tlog::leaf_hash(e)).collect();
        // Everything is computed before anything is written.
        let mut files: Vec<(String, Vec<u8>)> = Vec::new();
        for (n, chunk) in entries.chunks(tlog::TILE_WIDTH as usize).enumerate() {
            let mut b = Vec::new();
            for e in chunk {
                b.extend_from_slice(&(e.len() as u16).to_be_bytes());
                b.extend_from_slice(e);
            }
            files.push((bundle_path(n as u64, chunk.len() as u64), b));
        }
        let mut level = leaves.clone();
        let mut lvl = 0u32;
        while !level.is_empty() {
            for (n, chunk) in level.chunks(tlog::TILE_WIDTH as usize).enumerate() {
                files.push((
                    tlog::tile_path(lvl, n as u64, chunk.len() as u64),
                    chunk.iter().flatten().copied().collect(),
                ));
            }
            level = level
                .as_chunks::<{ tlog::TILE_WIDTH as usize }>()
                .0
                .iter()
                .map(|c| tlog::mth_of(c.as_slice()))
                .collect();
            lvl += 1;
        }
        for (rel, bytes) in &files {
            let path = self.dir.join(rel);
            if let Ok(old) = std::fs::read(&path) {
                if old != *bytes {
                    return Err(format!(
                        "{}: already written with other bytes; the log would rewrite its history",
                        path.display()
                    ));
                }
            }
        }
        for (rel, bytes) in &files {
            write_tile(&self.dir.join(rel), bytes)?;
        }
        let text = Checkpoint {
            origin: secret.key.name.clone(),
            size: leaves.len() as u64,
            root: tlog::mth_of(&leaves),
            extensions: vec![],
        }
        .text();
        let note = format!("{text}\n{}", secret.sign(&text, 0));
        write_atomic(&self.dir.join("checkpoint"), note.as_bytes())?;
        self.entries = entries;
        self.leaves = leaves;
        self.origin = Some(secret.key.name.clone());
        self.note = Some(note.clone());
        Ok(((first..self.size()).collect(), note))
    }

    fn checkpoint(&self) -> Result<(String, tlog::Note, Checkpoint), String> {
        let note = self
            .note
            .clone()
            .ok_or_else(|| format!("{}: an empty log, nothing appended yet", self.dir.display()))?;
        let n = tlog::parse_note(&note)?;
        let cp = tlog::parse_checkpoint(&n.text)?;
        Ok((note, n, cp))
    }

    /// The tlog-witness `add-checkpoint` body: `old <size>`, the consistency
    /// proof from that size to the checkpoint's, a blank line, the
    /// checkpoint as the log signed it (cosignatures left out).
    pub fn witness_request(&self, old: u64) -> Result<String, String> {
        let (_, n, cp) = self.checkpoint()?;
        if old > cp.size {
            return Err(format!(
                "old size {old} is past the checkpoint's {}",
                cp.size
            ));
        }
        let origin = self.origin.as_deref().unwrap_or_default();
        let log_sigs: Vec<&tlog::NoteSig> = n.sigs.iter().filter(|s| s.name == origin).collect();
        if log_sigs.is_empty() {
            return Err("the checkpoint has no signature of its log".into());
        }
        let proof = if old == 0 || old == cp.size {
            Vec::new()
        } else {
            tlog::consistency_proof(&mut tlog::Leaves(&self.leaves), old, cp.size)?
        };
        let mut s = format!("old {old}\n");
        for h in &proof {
            s.push_str(&b64(h));
            s.push('\n');
        }
        s.push('\n');
        s.push_str(&n.text);
        s.push('\n');
        for sig in log_sigs {
            let mut raw = sig.id.to_be_bytes().to_vec();
            raw.extend_from_slice(&sig.bytes);
            s.push_str(&format!("\u{2014} {} {}\n", sig.name, b64(&raw)));
        }
        Ok(s)
    }

    /// Add witness signature lines to the checkpoint: each must verify under
    /// one of `witnesses`, for this checkpoint's text, or nothing is added.
    pub fn add_cosignatures(
        &mut self,
        witnesses: &[NoteKey],
        lines: &str,
    ) -> Result<usize, String> {
        let (_, n, _) = self.checkpoint()?;
        let mut sigs = n.sigs.clone();
        let mut count = 0;
        for line in lines.lines().filter(|l| !l.trim().is_empty()) {
            let one = tlog::parse_note(&format!("{}\n{line}\n", n.text))
                .map_err(|e| format!("cosignature line {line:?}: {e}"))?;
            let sig = &one.sigs[0];
            let Some(key) = witnesses
                .iter()
                .find(|k| k.name == sig.name && k.id == sig.id)
            else {
                return Err(format!(
                    "a cosignature of {} {:08x}, a witness not named on the command line",
                    sig.name, sig.id
                ));
            };
            if key.kind != KeyKind::CosignatureV1 {
                return Err(format!("{}: not a cosignature/v1 key", key.name));
            }
            tlog::open_note(&one, std::slice::from_ref(key))
                .map_err(|e| format!("the cosignature of {} does not verify: {e}", key.name))?;
            // One line per witness: a newer cosignature replaces an older one.
            match sigs
                .iter_mut()
                .find(|s| s.name == sig.name && s.id == sig.id)
            {
                Some(have) if cosigned_at(have) < cosigned_at(sig) => {
                    *have = sig.clone();
                    count += 1;
                }
                Some(_) => {}
                None => {
                    sigs.push(sig.clone());
                    count += 1;
                }
            }
        }
        if count > 0 {
            let mut note = format!("{}\n", n.text);
            for s in &sigs {
                let mut raw = s.id.to_be_bytes().to_vec();
                raw.extend_from_slice(&s.bytes);
                note.push_str(&format!("\u{2014} {} {}\n", s.name, b64(&raw)));
            }
            tlog::parse_note(&note)?;
            write_atomic(&self.dir.join("checkpoint"), note.as_bytes())?;
            self.note = Some(note);
        }
        Ok(count)
    }

    /// The `c2sp.org/tlog-proof@v1` file of an entry against the current
    /// checkpoint, which must carry at least one cosignature.
    pub fn proof(&self, entry: &[u8]) -> Result<(u64, String), String> {
        let (note, n, cp) = self.checkpoint()?;
        let index = self
            .find(entry)
            .ok_or("this reference is not in the log: append it first")?;
        let origin = self.origin.as_deref().unwrap_or_default();
        if !n.sigs.iter().any(|s| s.name != origin) {
            return Err(
                "the checkpoint has no cosignature yet: witness-request, cosign, add-cosignatures first"
                    .into(),
            );
        }
        let hashes = tlog::inclusion_proof(&mut tlog::Leaves(&self.leaves), index, cp.size)?;
        Ok((index, tlog::format_tlog_proof(index, &hashes, &note)))
    }
}

/// A witness answering a tlog-witness `add-checkpoint` request for one log:
/// the checkpoint signed by `log`, `old` equal to the size it last cosigned
/// (kept in `state`, the checkpoint text), the consistency proof from that
/// checkpoint to this one. Returns its cosignature line at `time`.
pub fn cosign(
    witness: &Secret,
    log: &NoteKey,
    state: &Path,
    request: &str,
    time: u64,
) -> Result<String, String> {
    if witness.key.kind != KeyKind::CosignatureV1 {
        return Err(format!("{} is not a witness key", witness.key.name));
    }
    let tlog::AddCheckpoint {
        old,
        hashes: proof,
        checkpoint: note,
    } = tlog::parse_add_checkpoint(request)?;
    let n = tlog::parse_note(&note)?;
    tlog::open_note(&n, std::slice::from_ref(log)).map_err(|e| {
        format!(
            "refused: the checkpoint is not signed by the log {}: {e}",
            log.name
        )
    })?;
    let cp = tlog::parse_checkpoint(&n.text)?;
    if cp.origin != log.name {
        return Err(format!(
            "refused: a checkpoint of {:?}, this witness serves {:?}",
            cp.origin, log.name
        ));
    }
    let last = match std::fs::read_to_string(state) {
        Ok(t) => Some(tlog::parse_checkpoint(&t).map_err(|e| format!("{}: {e}", state.display()))?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(format!("{}: {e}", state.display())),
    };
    let last_size = last.as_ref().map_or(0, |c| c.size);
    if let Some(l) = &last {
        if l.origin != cp.origin {
            return Err(format!(
                "{}: the state of another log, {:?}",
                state.display(),
                l.origin
            ));
        }
    }
    if old != last_size {
        return Err(format!(
            "refused (409 Conflict): old size {old}, this witness's last cosigned size is {last_size}"
        ));
    }
    if cp.size < old {
        return Err(format!(
            "refused: the log went back from {old} to {}",
            cp.size
        ));
    }
    match &last {
        None if !proof.is_empty() => {
            return Err("refused: a consistency proof from the empty tree is not empty".into())
        }
        None => {}
        Some(l) => tlog::verify_consistency(l.size, cp.size, &proof, &l.root, &cp.root)
            .map_err(|e| format!("refused (422): consistency proof: {e}"))?,
    }
    write_atomic(state, n.text.as_bytes())?;
    Ok(witness.sign(&n.text, time))
}

/// A pin file (`scripts/pin-transparency-log.sh`): origin, log, witnesses,
/// quorum, max-age, tiles; '#' comments.
pub struct PinFile {
    pub origin: String,
    pub log: String,
    pub witnesses: Vec<String>,
    pub quorum: usize,
    pub max_age_s: u64,
    pub tiles: String,
}

impl PinFile {
    pub fn parse(text: &str) -> Result<Self, String> {
        let (mut origin, mut log, mut quorum, mut age, mut tiles) = (None, None, None, None, None);
        let mut witnesses = Vec::new();
        for line in text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
        {
            let (k, v) = line
                .split_once(' ')
                .ok_or_else(|| format!("pin line {line:?}"))?;
            let v = v.trim().to_string();
            let slot = match k {
                "origin" => &mut origin,
                "log" => &mut log,
                "quorum" => &mut quorum,
                "max-age" => &mut age,
                "tiles" => &mut tiles,
                "witness" => {
                    witnesses.push(v);
                    continue;
                }
                _ => return Err(format!("pin: unknown line {k:?}")),
            };
            if slot.replace(v).is_some() {
                return Err(format!("pin: {k} twice"));
            }
        }
        let need = |v: Option<String>, k: &str| v.ok_or_else(|| format!("pin: no {k} line"));
        Ok(PinFile {
            origin: need(origin, "origin")?,
            log: need(log, "log")?,
            witnesses,
            quorum: need(quorum, "quorum")?
                .parse()
                .map_err(|_| "pin: quorum is not a number")?,
            max_age_s: tlog::decimal(&need(age, "max-age")?)
                .map_err(|e| format!("pin: max-age {e}"))?,
            tiles: need(tiles, "tiles")?,
        })
    }

    /// The policy a build that pins this file runs (`transparency::pinned`).
    pub fn policy(&self) -> Result<transparency::Policy, String> {
        fn leak(s: &str) -> &'static str {
            Box::leak(s.to_string().into_boxed_str())
        }
        let witnesses: Vec<&'static str> = self.witnesses.iter().map(|w| leak(w)).collect();
        transparency::pinned(&transparency::Pin {
            origin: leak(&self.origin),
            log_key: leak(&self.log),
            witnesses: Box::leak(witnesses.into_boxed_slice()),
            quorum: self.quorum,
            max_age_s: self.max_age_s,
            tiles: leak(&self.tiles),
        })
    }
}

/// The relay's check of a reference and its `.tlog-proof` against a pin
/// file, the log's files read from `tiles_dir` or from the pinned location.
/// `state_dir` holds the last checkpoint verified, as the relay keeps it.
pub fn verify(
    pin: &PinFile,
    reference: &Path,
    tiles_dir: Option<&Path>,
    state_dir: &Path,
    now: u64,
) -> Result<transparency::Logged, String> {
    let policy = pin.policy()?;
    let bytes = crate::fsread::read_capped(reference, crate::reference::MAX_BYTES)
        .map_err(|e| format!("{}: {e}", reference.display()))?;
    let proof_path = transparency::proof_path(reference);
    let proof = crate::fsread::read_capped(&proof_path, transparency::MAX_PROOF_BYTES)
        .map_err(|e| format!("{}: {e}", proof_path.display()))?;
    let proof =
        String::from_utf8(proof).map_err(|_| format!("{}: not UTF-8", proof_path.display()))?;
    let source: Box<dyn tlog::TileSource> = match tiles_dir {
        Some(d) => Box::new(DirTiles(d.into())),
        None => policy.source(reference),
    };
    transparency::check(&bytes, &proof, &policy, source.as_ref(), state_dir, now)
}
