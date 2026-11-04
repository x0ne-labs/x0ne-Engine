// SPDX-License-Identifier: Apache-2.0
//! Transparency-log verification in the C2SP formats (editor's copies read
//! 2026-10-22): signed notes (c2sp.org/signed-note), checkpoints
//! (c2sp.org/tlog-checkpoint), Ed25519 cosignatures `cosignature/v1`
//! (c2sp.org/tlog-cosignature), offline proofs `c2sp.org/tlog-proof@v1`
//! (c2sp.org/tlog-proof) and tiles (c2sp.org/tlog-tiles). Merkle proofs as
//! RFC 6962 §2.1 builds them and RFC 9162 §2.1.3.2 and §2.1.4.2 verify them.
//! Rekor v2 and Tessera logs serve the same tiles and checkpoints.
//!
//! No policy and no file system here: `transparency.rs` decides which keys,
//! which witnesses and how fresh, and reads the files.

use base64::Engine as _;
use ed25519_dalek::{Signature, VerifyingKey};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub type Hash = [u8; 32];

/// Signature type of an Ed25519 note signature (signed-note).
pub const SIG_ED25519: u8 = 0x01;
/// Signature type of an Ed25519 checkpoint cosignature (tlog-checkpoint).
pub const SIG_COSIGNATURE_V1: u8 = 0x04;
/// signed-note: verifiers must accept at least 16 signatures. Past this, refused.
pub const MAX_SIGNATURES: usize = 32;
pub const MAX_NOTE_BYTES: usize = 64 * 1024;
/// A proof longer than the height of a 2^63-leaf tree is malformed.
pub const MAX_PROOF_HASHES: usize = 63;
pub const PROOF_HEADER: &str = "c2sp.org/tlog-proof@v1";
/// Hashes in a full tile (tlog-tiles).
pub const TILE_WIDTH: u64 = 256;
const TILE_HEIGHT: u32 = 8;

/// RFC 6962 leaf hash: SHA-256(0x00 || entry).
pub fn leaf_hash(entry: &[u8]) -> Hash {
    let mut h = Sha256::new();
    h.update([0u8]);
    h.update(entry);
    h.finalize().into()
}

/// RFC 6962 interior node: SHA-256(0x01 || left || right).
pub fn node_hash(left: &Hash, right: &Hash) -> Hash {
    let mut h = Sha256::new();
    h.update([1u8]);
    h.update(left);
    h.update(right);
    h.finalize().into()
}

/// Root of the empty tree: SHA-256 of nothing.
pub fn empty_root() -> Hash {
    Sha256::digest([]).into()
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Standard base64 with padding, canonical only: decoders MUST reject
/// non-canonical encodings (tlog-checkpoint, tlog-proof).
pub fn b64_canonical(s: &str) -> Result<Vec<u8>, String> {
    let v = base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| format!("base64 {s:?}: {e}"))?;
    if b64(&v) != s {
        return Err(format!("base64 {s:?} is not canonical"));
    }
    Ok(v)
}

fn hash_of(s: &str) -> Result<Hash, String> {
    let v = b64_canonical(s)?;
    v.as_slice()
        .try_into()
        .map_err(|_| format!("hash {s:?}: {} bytes, expected 32", v.len()))
}

/// ASCII decimal without leading zeros ("0" is zero), as checkpoints and proofs write it.
pub fn decimal(s: &str) -> Result<u64, String> {
    let ok =
        !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) && (s == "0" || !s.starts_with('0'));
    if !ok {
        return Err(format!("{s:?} is not a decimal without leading zeros"));
    }
    s.parse().map_err(|e| format!("{s:?}: {e}"))
}

fn largest_power_of_two_below(n: u64) -> u64 {
    debug_assert!(n > 1);
    1u64 << (n - 1).ilog2()
}

// ---- Merkle proofs: verification (RFC 9162).

/// RFC 9162 §2.1.3.2: `proof` leads from `leaf` at `index` to `root` of a
/// tree of `size` leaves.
pub fn verify_inclusion(
    leaf: &Hash,
    index: u64,
    size: u64,
    proof: &[Hash],
    root: &Hash,
) -> Result<(), String> {
    if index >= size {
        return Err(format!("leaf index {index} is outside a tree of {size}"));
    }
    if proof.len() > MAX_PROOF_HASHES {
        return Err(format!("inclusion proof of {} hashes", proof.len()));
    }
    let (mut fnode, mut snode, mut r) = (index, size - 1, *leaf);
    for p in proof {
        if snode == 0 {
            return Err("inclusion proof longer than the path to the root".into());
        }
        if fnode & 1 == 1 || fnode == snode {
            r = node_hash(p, &r);
            if fnode & 1 == 0 {
                while fnode & 1 == 0 && fnode != 0 {
                    fnode >>= 1;
                    snode >>= 1;
                }
            }
        } else {
            r = node_hash(&r, p);
        }
        fnode >>= 1;
        snode >>= 1;
    }
    if snode != 0 {
        return Err("inclusion proof shorter than the path to the root".into());
    }
    if &r != root {
        return Err(format!(
            "inclusion proof leads to {}, the checkpoint's root is {}",
            hex(&r),
            hex(root)
        ));
    }
    Ok(())
}

/// RFC 9162 §2.1.4.2: the tree of `size1` leaves and root `root1` is a
/// prefix of the tree of `size2` leaves and root `root2`.
pub fn verify_consistency(
    size1: u64,
    size2: u64,
    proof: &[Hash],
    root1: &Hash,
    root2: &Hash,
) -> Result<(), String> {
    if size1 > size2 {
        return Err(format!("consistency from {size1} to the smaller {size2}"));
    }
    if proof.len() > MAX_PROOF_HASHES + 1 {
        return Err(format!("consistency proof of {} hashes", proof.len()));
    }
    if size1 == size2 {
        if !proof.is_empty() {
            return Err("consistency proof between equal sizes is not empty".into());
        }
        return if root1 == root2 {
            Ok(())
        } else {
            Err(format!(
                "two roots for size {size1}: {} and {}",
                hex(root1),
                hex(root2)
            ))
        };
    }
    if size1 == 0 {
        return if proof.is_empty() {
            Ok(())
        } else {
            Err("consistency proof from the empty tree is not empty".into())
        };
    }
    let mut path: Vec<Hash> = Vec::with_capacity(proof.len() + 1);
    if size1.is_power_of_two() {
        path.push(*root1);
    }
    path.extend_from_slice(proof);
    let Some((first, rest)) = path.split_first() else {
        return Err("empty consistency proof".into());
    };
    let (mut fnode, mut snode) = (size1 - 1, size2 - 1);
    while fnode & 1 == 1 {
        fnode >>= 1;
        snode >>= 1;
    }
    let (mut fr, mut sr) = (*first, *first);
    for c in rest {
        if snode == 0 {
            return Err("consistency proof longer than the trees".into());
        }
        if fnode & 1 == 1 || fnode == snode {
            fr = node_hash(c, &fr);
            sr = node_hash(c, &sr);
            if fnode & 1 == 0 {
                while fnode & 1 == 0 && fnode != 0 {
                    fnode >>= 1;
                    snode >>= 1;
                }
            }
        } else {
            sr = node_hash(&sr, c);
        }
        fnode >>= 1;
        snode >>= 1;
    }
    if snode != 0 {
        return Err("consistency proof shorter than the trees".into());
    }
    if &fr != root1 || &sr != root2 {
        return Err(format!(
            "size {size1} root {} is not a prefix of size {size2} root {}",
            hex(root1),
            hex(root2)
        ));
    }
    Ok(())
}

// ---- Merkle proofs: construction (RFC 6962 §2.1.1, §2.1.2), from any
// source of subtree hashes: the leaves of a local log, or tiles.

/// MTH(D\[lo:hi\]), the root of the leaves `lo..hi`.
pub trait Subtrees {
    fn mth(&mut self, lo: u64, hi: u64) -> Result<Hash, String>;
}

/// The leaf hashes of a whole tree, in memory (a local log, tests).
pub struct Leaves<'a>(pub &'a [Hash]);

impl Subtrees for Leaves<'_> {
    fn mth(&mut self, lo: u64, hi: u64) -> Result<Hash, String> {
        if lo > hi || hi > self.0.len() as u64 {
            return Err(format!(
                "subtree {lo}..{hi} outside {} leaves",
                self.0.len()
            ));
        }
        Ok(mth_of(&self.0[lo as usize..hi as usize]))
    }
}

/// The root of a list of leaf hashes (RFC 6962 MTH).
pub fn mth_of(leaves: &[Hash]) -> Hash {
    match leaves.len() {
        0 => empty_root(),
        1 => leaves[0],
        n => {
            let k = largest_power_of_two_below(n as u64) as usize;
            node_hash(&mth_of(&leaves[..k]), &mth_of(&leaves[k..]))
        }
    }
}

/// RFC 6962 PATH(index, D\[0:size\]).
pub fn inclusion_proof(t: &mut dyn Subtrees, index: u64, size: u64) -> Result<Vec<Hash>, String> {
    if index >= size {
        return Err(format!("leaf index {index} is outside a tree of {size}"));
    }
    let mut out = Vec::new();
    path(t, index, 0, size, &mut out)?;
    Ok(out)
}

fn path(t: &mut dyn Subtrees, m: u64, lo: u64, hi: u64, out: &mut Vec<Hash>) -> Result<(), String> {
    let n = hi - lo;
    if n <= 1 {
        return Ok(());
    }
    let k = largest_power_of_two_below(n);
    if m < k {
        path(t, m, lo, lo + k, out)?;
        out.push(t.mth(lo + k, hi)?);
    } else {
        path(t, m - k, lo + k, hi, out)?;
        out.push(t.mth(lo, lo + k)?);
    }
    Ok(())
}

/// RFC 6962 PROOF(size1, D\[0:size2\]).
pub fn consistency_proof(
    t: &mut dyn Subtrees,
    size1: u64,
    size2: u64,
) -> Result<Vec<Hash>, String> {
    if size1 > size2 {
        return Err(format!("consistency from {size1} to the smaller {size2}"));
    }
    let mut out = Vec::new();
    if size1 == 0 || size1 == size2 {
        return Ok(out);
    }
    subproof(t, size1, 0, size2, true, &mut out)?;
    Ok(out)
}

fn subproof(
    t: &mut dyn Subtrees,
    m: u64,
    lo: u64,
    hi: u64,
    b: bool,
    out: &mut Vec<Hash>,
) -> Result<(), String> {
    let n = hi - lo;
    if m == n {
        if !b {
            out.push(t.mth(lo, hi)?);
        }
        return Ok(());
    }
    let k = largest_power_of_two_below(n);
    if m <= k {
        subproof(t, m, lo, lo + k, b, out)?;
        out.push(t.mth(lo + k, hi)?);
    } else {
        subproof(t, m - k, lo + k, hi, false, out)?;
        out.push(t.mth(lo, lo + k)?);
    }
    Ok(())
}

// ---- Tiles (tlog-tiles): subtree hashes read from a log's static files.

/// Where a log's files are read: a directory, a mirror. What it returns is
/// not trusted: every proof built from it is verified against signed roots.
pub trait TileSource {
    /// `<prefix>/checkpoint`
    fn checkpoint(&self) -> Result<String, String>;
    /// `<prefix>/<path>`, for a `tile/...` path.
    fn read(&self, path: &str) -> Result<Vec<u8>, String>;
}

/// `<N>` of a tile path: groups of three digits, all but the last
/// prefixed `x` (1234067 is `x001/x234/067`).
pub fn tile_index_path(mut n: u64) -> String {
    let mut parts = vec![format!("{:03}", n % 1000)];
    n /= 1000;
    while n > 0 {
        parts.push(format!("x{:03}", n % 1000));
        n /= 1000;
    }
    parts.reverse();
    parts.join("/")
}

/// `tile/<L>/<N>[.p/<W>]`: a full tile has 256 hashes, a partial one `W`.
pub fn tile_path(level: u32, index: u64, width: u64) -> String {
    let base = format!("tile/{level}/{}", tile_index_path(index));
    if width == TILE_WIDTH {
        base
    } else {
        format!("{base}.p/{width}")
    }
}

/// The subtree hashes of a tree of `size` leaves, from its tiles.
pub struct TileHashes<'a> {
    src: &'a dyn TileSource,
    size: u64,
    cache: HashMap<(u32, u64), Vec<Hash>>,
}

impl<'a> TileHashes<'a> {
    pub fn new(src: &'a dyn TileSource, size: u64) -> Self {
        TileHashes {
            src,
            size,
            cache: HashMap::new(),
        }
    }

    /// The hashes of tile `index` at tile level `level`, as wide as this tree makes it.
    fn tile(&mut self, level: u32, index: u64) -> Result<&Vec<Hash>, String> {
        if !self.cache.contains_key(&(level, index)) {
            // Complete height-8L subtrees in the tree: the entries of level L.
            let entries = self.size >> (TILE_HEIGHT * level);
            let width = entries.saturating_sub(index * TILE_WIDTH).min(TILE_WIDTH);
            if width == 0 {
                return Err(format!(
                    "tile {level}/{index} is past a tree of {}",
                    self.size
                ));
            }
            let p = tile_path(level, index, width);
            let read = |path: &str, want: u64| -> Result<Vec<u8>, String> {
                let bytes = self.src.read(path)?;
                if bytes.len() as u64 != want * 32 {
                    return Err(format!(
                        "{path}: {} bytes, expected {}",
                        bytes.len(),
                        want * 32
                    ));
                }
                Ok(bytes)
            };
            // A partial tile is replaced by the full one as the log grows:
            // its prefix is the same hashes (tiles are append-only).
            let bytes = match read(&p, width) {
                Ok(b) => b,
                Err(e) if width < TILE_WIDTH => {
                    let full = tile_path(level, index, TILE_WIDTH);
                    let mut b = read(&full, TILE_WIDTH).map_err(|f| format!("{e}; {f}"))?;
                    b.truncate((width * 32) as usize);
                    b
                }
                Err(e) => return Err(e),
            };
            let hashes = bytes.as_chunks::<32>().0.to_vec();
            self.cache.insert((level, index), hashes);
        }
        Ok(&self.cache[&(level, index)])
    }

    /// The complete subtree of height `height` whose first leaf is `k << height`.
    fn complete(&mut self, height: u32, k: u64) -> Result<Hash, String> {
        let (level, rest) = (height / TILE_HEIGHT, height % TILE_HEIGHT);
        let first = k << rest;
        let (tile, offset) = (first / TILE_WIDTH, (first % TILE_WIDTH) as usize);
        let count = 1usize << rest;
        let hashes = self.tile(level, tile)?;
        if offset + count > hashes.len() {
            return Err(format!(
                "tile {level}/{tile} has {} hashes, needs {}",
                hashes.len(),
                offset + count
            ));
        }
        let mut row: Vec<Hash> = hashes[offset..offset + count].to_vec();
        while row.len() > 1 {
            row = row
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| node_hash(&p[0], &p[1]))
                .collect();
        }
        Ok(row[0])
    }
}

impl Subtrees for TileHashes<'_> {
    fn mth(&mut self, lo: u64, hi: u64) -> Result<Hash, String> {
        if lo >= hi || hi > self.size {
            return Err(format!(
                "subtree {lo}..{hi} outside a tree of {}",
                self.size
            ));
        }
        let n = hi - lo;
        if n.is_power_of_two() && lo.is_multiple_of(n) {
            return self.complete(n.trailing_zeros(), lo / n);
        }
        let k = largest_power_of_two_below(n);
        let left = self.mth(lo, lo + k)?;
        let right = self.mth(lo + k, hi)?;
        Ok(node_hash(&left, &right))
    }
}

// ---- Signed notes and keys (signed-note, tlog-checkpoint, tlog-cosignature).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyKind {
    /// A log's Ed25519 note signature (type 0x01).
    Ed25519,
    /// A witness's timestamped Ed25519 checkpoint cosignature (type 0x04).
    CosignatureV1,
}

impl KeyKind {
    fn byte(self) -> u8 {
        match self {
            KeyKind::Ed25519 => SIG_ED25519,
            KeyKind::CosignatureV1 => SIG_COSIGNATURE_V1,
        }
    }
}

/// A verifier key: name, key ID, type, Ed25519 public key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoteKey {
    pub name: String,
    pub id: u32,
    pub kind: KeyKind,
    key: [u8; 32],
}

fn check_key_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.contains('+') || name.chars().any(char::is_whitespace) {
        return Err(format!(
            "key name {name:?}: empty, or with a space or a plus sign"
        ));
    }
    Ok(())
}

/// SHA-256(name || "\n" || type || public key)\[:4\], big-endian.
pub fn key_id(name: &str, kind: KeyKind, key: &[u8; 32]) -> u32 {
    let mut h = Sha256::new();
    h.update(name.as_bytes());
    h.update(b"\n");
    h.update([kind.byte()]);
    h.update(key);
    let d = h.finalize();
    u32::from_be_bytes([d[0], d[1], d[2], d[3]])
}

impl NoteKey {
    pub fn new(name: &str, kind: KeyKind, key: [u8; 32]) -> Result<Self, String> {
        check_key_name(name)?;
        VerifyingKey::from_bytes(&key).map_err(|e| format!("key {name}: {e}"))?;
        Ok(NoteKey {
            name: name.to_string(),
            id: key_id(name, kind, &key),
            kind,
            key,
        })
    }

    /// `<name>+<8 hex>+<base64(type || key)>`; the ID must be the one the key gives.
    pub fn parse(vkey: &str) -> Result<Self, String> {
        let mut parts = vkey.splitn(3, '+');
        let (Some(name), Some(id), Some(rest)) = (parts.next(), parts.next(), parts.next()) else {
            return Err(format!("verifier key {vkey:?}: not <name>+<id>+<key>"));
        };
        let well_formed_id = id.len() == 8
            && id
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if !well_formed_id {
            return Err(format!(
                "verifier key {name}: key ID {id:?} is not 8 lowercase hex"
            ));
        }
        let raw = b64_canonical(rest)?;
        let (kind, key) = match raw.split_first() {
            Some((&SIG_ED25519, k)) => (KeyKind::Ed25519, k),
            Some((&SIG_COSIGNATURE_V1, k)) => (KeyKind::CosignatureV1, k),
            Some((t, _)) => {
                return Err(format!("verifier key {name}: type {t:#04x} not supported"))
            }
            None => return Err(format!("verifier key {name}: empty")),
        };
        let key: [u8; 32] = key
            .try_into()
            .map_err(|_| format!("verifier key {name}: {} key bytes, expected 32", key.len()))?;
        let k = NoteKey::new(name, kind, key)?;
        if format!("{:08x}", k.id) != id {
            return Err(format!(
                "verifier key {name}: key ID {id} is not the key's ({:08x})",
                k.id
            ));
        }
        Ok(k)
    }

    pub fn vkey(&self) -> String {
        let mut raw = vec![self.kind.byte()];
        raw.extend_from_slice(&self.key);
        format!("{}+{:08x}+{}", self.name, self.id, b64(&raw))
    }

    pub fn public_key(&self) -> [u8; 32] {
        self.key
    }
}

/// One signature line: `— <name> <base64(id || signature)>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoteSig {
    pub name: String,
    pub id: u32,
    pub bytes: Vec<u8>,
}

/// A note split in text and signatures; nothing verified yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Note {
    /// The signed text, ending with a newline.
    pub text: String,
    pub sigs: Vec<NoteSig>,
}

const SIG_PREFIX: &str = "\u{2014} ";

pub fn parse_note(s: &str) -> Result<Note, String> {
    if s.len() > MAX_NOTE_BYTES {
        return Err(format!("note of {} bytes, cap {MAX_NOTE_BYTES}", s.len()));
    }
    if s.chars().any(|c| c < ' ' && c != '\n') {
        return Err("note with a control character".into());
    }
    if !s.ends_with('\n') {
        return Err("note does not end with a newline".into());
    }
    // The separator is the last blank line.
    let split = s
        .rfind("\n\n")
        .ok_or("note without a blank line before its signatures")?;
    let (text, sigs) = (&s[..split + 1], &s[split + 2..]);
    if text.len() <= 1 {
        return Err("note with an empty text".into());
    }
    let mut out: Vec<NoteSig> = Vec::new();
    for line in sigs.lines() {
        if out.len() == MAX_SIGNATURES {
            return Err(format!("note with more than {MAX_SIGNATURES} signatures"));
        }
        let rest = line
            .strip_prefix(SIG_PREFIX)
            .ok_or_else(|| format!("signature line {line:?} does not start with an em dash"))?;
        let (name, sig) = rest
            .split_once(' ')
            .ok_or_else(|| format!("signature line {line:?}: no key name"))?;
        check_key_name(name)?;
        let raw = b64_canonical(sig)?;
        if raw.len() < 5 {
            return Err(format!("signature line of {name}: {} bytes", raw.len()));
        }
        let id = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
        if out.iter().any(|o| o.name == name && o.id == id) {
            return Err(format!("two signature lines of {name} {id:08x}"));
        }
        out.push(NoteSig {
            name: name.to_string(),
            id,
            bytes: raw[4..].to_vec(),
        });
    }
    if out.is_empty() {
        return Err("note without a signature".into());
    }
    Ok(Note {
        text: text.to_string(),
        sigs: out,
    })
}

/// The message a witness signs: `cosignature/v1`, `time <t>`, then the checkpoint text.
pub fn cosignature_message(text: &str, time: u64) -> String {
    format!("cosignature/v1\ntime {time}\n{text}")
}

/// A signature of a known key that verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verified {
    pub key: NoteKey,
    /// The cosignature's timestamp; None for a log signature.
    pub time: Option<u64>,
}

/// Verify a note against known keys (signed-note): signatures of unknown
/// keys are ignored, a known key whose signature fails rejects the whole
/// note, and at least one known key must verify.
pub fn open_note(note: &Note, known: &[NoteKey]) -> Result<Vec<Verified>, String> {
    let mut out = Vec::new();
    for sig in &note.sigs {
        let Some(key) = known.iter().find(|k| k.name == sig.name && k.id == sig.id) else {
            continue;
        };
        let vk = VerifyingKey::from_bytes(&key.key).map_err(|e| format!("{}: {e}", key.name))?;
        let (time, signature, message) = match key.kind {
            KeyKind::Ed25519 => (None, sig.bytes.as_slice(), note.text.clone()),
            KeyKind::CosignatureV1 => {
                if sig.bytes.len() != 8 + 64 {
                    return Err(format!(
                        "cosignature of {}: {} bytes, expected 72",
                        key.name,
                        sig.bytes.len()
                    ));
                }
                let t = u64::from_be_bytes(sig.bytes[..8].try_into().expect("8 bytes"));
                (Some(t), &sig.bytes[8..], cosignature_message(&note.text, t))
            }
        };
        let signature = Signature::from_slice(signature)
            .map_err(|e| format!("signature of {}: {e}", key.name))?;
        vk.verify_strict(message.as_bytes(), &signature)
            .map_err(|_| format!("the signature of {} does not verify", key.name))?;
        out.push(Verified {
            key: key.clone(),
            time,
        });
    }
    if out.is_empty() {
        return Err("no signature of a known key".into());
    }
    Ok(out)
}

/// A checkpoint's text (tlog-checkpoint).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    pub origin: String,
    pub size: u64,
    pub root: Hash,
    pub extensions: Vec<String>,
}

pub fn parse_checkpoint(text: &str) -> Result<Checkpoint, String> {
    let body = text
        .strip_suffix('\n')
        .ok_or("checkpoint text does not end with a newline")?;
    let lines: Vec<&str> = body.split('\n').collect();
    if lines.len() < 3 || lines.iter().any(|l| l.is_empty()) {
        return Err("checkpoint: fewer than three lines, or an empty line".into());
    }
    Ok(Checkpoint {
        origin: lines[0].to_string(),
        size: decimal(lines[1]).map_err(|e| format!("checkpoint size: {e}"))?,
        root: hash_of(lines[2]).map_err(|e| format!("checkpoint root: {e}"))?,
        extensions: lines[3..].iter().map(|l| l.to_string()).collect(),
    })
}

impl Checkpoint {
    pub fn text(&self) -> String {
        let mut s = format!("{}\n{}\n{}\n", self.origin, self.size, b64(&self.root));
        for e in &self.extensions {
            s.push_str(e);
            s.push('\n');
        }
        s
    }
}

/// A `c2sp.org/tlog-proof@v1` file: leaf index, inclusion proof, signed checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlogProof {
    /// Opaque and not authenticated: never trusted.
    pub extra: Option<Vec<u8>>,
    pub index: u64,
    pub hashes: Vec<Hash>,
    /// The checkpoint as a signed note, verbatim.
    pub checkpoint: String,
}

pub fn parse_tlog_proof(s: &str) -> Result<TlogProof, String> {
    let (head, checkpoint) = s
        .split_once("\n\n")
        .ok_or("tlog-proof without the blank line before its checkpoint")?;
    let mut lines = head.split('\n');
    if lines.next() != Some(PROOF_HEADER) {
        return Err(format!("tlog-proof: the first line is not {PROOF_HEADER}"));
    }
    let mut line = lines.next().ok_or("tlog-proof without an index")?;
    let mut extra = None;
    if let Some(e) = line.strip_prefix("extra ") {
        extra = Some(b64_canonical(e)?);
        line = lines.next().ok_or("tlog-proof without an index")?;
    }
    let index = decimal(
        line.strip_prefix("index ")
            .ok_or_else(|| format!("tlog-proof: {line:?} is not the index line"))?,
    )?;
    let hashes = lines.map(hash_of).collect::<Result<Vec<_>, _>>()?;
    if hashes.len() > MAX_PROOF_HASHES {
        return Err(format!("tlog-proof of {} hashes", hashes.len()));
    }
    Ok(TlogProof {
        extra,
        index,
        hashes,
        checkpoint: checkpoint.to_string(),
    })
}

pub fn format_tlog_proof(index: u64, hashes: &[Hash], checkpoint_note: &str) -> String {
    let mut s = format!("{PROOF_HEADER}\nindex {index}\n");
    for h in hashes {
        s.push_str(&b64(h));
        s.push('\n');
    }
    s.push('\n');
    s.push_str(checkpoint_note);
    s
}

/// A tlog-witness `add-checkpoint` request body (c2sp.org/tlog-witness):
/// `old <size>`, the consistency proof from that size to the checkpoint's,
/// a blank line, the signed checkpoint. The one standard encoding of a
/// consistency proof: what a verifier that keeps its last checkpoint takes
/// to move to a newer one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddCheckpoint {
    pub old: u64,
    pub hashes: Vec<Hash>,
    /// The checkpoint as a signed note, verbatim.
    pub checkpoint: String,
}

/// tlog-witness: a client sends at most 63 consistency proof lines.
pub fn parse_add_checkpoint(body: &str) -> Result<AddCheckpoint, String> {
    let (head, checkpoint) = body
        .split_once("\n\n")
        .ok_or("add-checkpoint request without the blank line before its checkpoint")?;
    let mut lines = head.split('\n');
    let old = lines
        .next()
        .and_then(|l| l.strip_prefix("old "))
        .ok_or("add-checkpoint request: the first line is not old <size>")?;
    let old = decimal(old).map_err(|e| format!("add-checkpoint request: old size {e}"))?;
    let hashes = lines
        .map(|l| hash_of(l).map_err(|e| format!("proof line {l:?}: {e}")))
        .collect::<Result<Vec<Hash>, String>>()?;
    if hashes.len() > MAX_PROOF_HASHES {
        return Err(format!(
            "add-checkpoint request of {} proof lines, at most {MAX_PROOF_HASHES}",
            hashes.len()
        ));
    }
    Ok(AddCheckpoint {
        old,
        hashes,
        checkpoint: checkpoint.to_string(),
    })
}

pub fn format_add_checkpoint(old: u64, hashes: &[Hash], checkpoint_note: &str) -> String {
    let mut s = format!("old {old}\n");
    for h in hashes {
        s.push_str(&b64(h));
        s.push('\n');
    }
    s.push('\n');
    s.push_str(checkpoint_note);
    s
}

/// A signature line for `text` (log, type 0x01) or its cosignature at `time` (type 0x04).
pub fn sign_line(
    text: &str,
    key: &NoteKey,
    secret: &ed25519_dalek::SigningKey,
    time: u64,
) -> String {
    use ed25519_dalek::Signer as _;
    let mut raw = key.id.to_be_bytes().to_vec();
    match key.kind {
        KeyKind::Ed25519 => raw.extend_from_slice(&secret.sign(text.as_bytes()).to_bytes()),
        KeyKind::CosignatureV1 => {
            raw.extend_from_slice(&time.to_be_bytes());
            raw.extend_from_slice(
                &secret
                    .sign(cosignature_message(text, time).as_bytes())
                    .to_bytes(),
            );
        }
    }
    format!("{SIG_PREFIX}{} {}\n", key.name, b64(&raw))
}

#[cfg(all(test, not(x0ne_sdk)))]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn leaves(n: u64) -> Vec<Hash> {
        (0..n)
            .map(|i| leaf_hash(format!("entry {i}").as_bytes()))
            .collect()
    }

    fn sk(label: &str) -> SigningKey {
        SigningKey::from_bytes(&Sha256::digest(label.as_bytes()).into())
    }

    fn key(name: &str, kind: KeyKind, s: &SigningKey) -> NoteKey {
        NoteKey::new(name, kind, s.verifying_key().to_bytes()).unwrap()
    }

    /// RFC 6962 hashes: the empty root, a leaf, a node (values of the
    /// Certificate Transparency test vectors, merkle_tree_test.cc).
    #[test]
    fn hashes_are_rfc_6962() {
        assert_eq!(
            hex(&empty_root()),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&leaf_hash(b"")),
            "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"
        );
        // The 8-leaf tree of the CT vectors: leaves "", 00, 10, 2021, 3031, 40414243,
        // 5051525354555657, 606162636465666768696a6b6c6d6e6f.
        let data: [&[u8]; 8] = [
            b"",
            &[0x00],
            &[0x10],
            &[0x20, 0x21],
            &[0x30, 0x31],
            &[0x40, 0x41, 0x42, 0x43],
            &[0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57],
            &[
                0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d,
                0x6e, 0x6f,
            ],
        ];
        let l: Vec<Hash> = data.iter().map(|d| leaf_hash(d)).collect();
        assert_eq!(
            hex(&mth_of(&l)),
            "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328"
        );
        assert_eq!(
            hex(&mth_of(&l[..3])),
            "aeb6bcfe274b70a14fb067a5e5578264db0fa9b51af5e0ba159158f329e06e77"
        );
    }

    #[test]
    fn every_inclusion_and_consistency_proof_of_small_trees_verifies() {
        let all = leaves(70);
        for size in 1..=70u64 {
            let l = &all[..size as usize];
            let root = mth_of(l);
            for i in 0..size {
                let p = inclusion_proof(&mut Leaves(l), i, size).unwrap();
                verify_inclusion(&l[i as usize], i, size, &p, &root).unwrap();
            }
            for size1 in 0..=size {
                let p = consistency_proof(&mut Leaves(l), size1, size).unwrap();
                verify_consistency(size1, size, &p, &mth_of(&l[..size1 as usize]), &root).unwrap();
            }
        }
    }

    /// SECURITY: a proof for another leaf, another index, a changed hash, a
    /// truncated or extended path, or another root: refused.
    #[test]
    fn a_false_inclusion_proof_is_refused() {
        let l = leaves(13);
        let root = mth_of(&l);
        let p = inclusion_proof(&mut Leaves(&l), 5, 13).unwrap();
        assert!(
            verify_inclusion(&l[6], 5, 13, &p, &root).is_err(),
            "other leaf"
        );
        assert!(
            verify_inclusion(&l[5], 4, 13, &p, &root).is_err(),
            "other index"
        );
        assert!(
            verify_inclusion(&l[5], 5, 6, &p, &root).is_err(),
            "other size"
        );
        // Sizes whose path to index 5 has the same shape (13 and 14) accept
        // the same path to the same root: the size is bound by the signed
        // checkpoint, which carries it with the root.
        assert!(verify_inclusion(&l[5], 5, 14, &p, &root).is_ok());
        assert!(
            verify_inclusion(&l[5], 13, 13, &p, &root).is_err(),
            "index past size"
        );
        let mut bad = p.clone();
        bad[1][0] ^= 1;
        assert!(
            verify_inclusion(&l[5], 5, 13, &bad, &root).is_err(),
            "changed hash"
        );
        assert!(
            verify_inclusion(&l[5], 5, 13, &p[..p.len() - 1], &root).is_err(),
            "short"
        );
        let mut long = p.clone();
        long.push(root);
        assert!(
            verify_inclusion(&l[5], 5, 13, &long, &root).is_err(),
            "long"
        );
        assert!(
            verify_inclusion(&l[5], 5, 13, &p, &mth_of(&l[..12])).is_err(),
            "other root"
        );
    }

    /// SECURITY: two heads of one log that are not a prefix of each other
    /// (a fork) fail every consistency proof, and two roots for one size are refused.
    #[test]
    fn a_forked_log_fails_consistency() {
        let honest = leaves(20);
        let mut forked = honest.clone();
        forked[9] = leaf_hash(b"the targeted image");
        let (r10, f20) = (mth_of(&honest[..10]), mth_of(&forked));
        // The fork's own proof, against the honest head.
        let p = consistency_proof(&mut Leaves(&forked), 10, 20).unwrap();
        let err = verify_consistency(10, 20, &p, &r10, &f20).unwrap_err();
        assert!(err.contains("is not a prefix"), "{err}");
        let p = consistency_proof(&mut Leaves(&honest), 10, 20).unwrap();
        assert!(verify_consistency(10, 20, &p, &r10, &f20).is_err());
        let err = verify_consistency(20, 20, &[], &mth_of(&honest), &f20).unwrap_err();
        assert!(err.contains("two roots for size 20"), "{err}");
        assert!(verify_consistency(21, 20, &[], &f20, &f20).is_err());
    }

    struct MemTiles(HashMap<String, Vec<u8>>);
    impl TileSource for MemTiles {
        fn checkpoint(&self) -> Result<String, String> {
            Err("none".into())
        }
        fn read(&self, path: &str) -> Result<Vec<u8>, String> {
            self.0
                .get(path)
                .cloned()
                .ok_or_else(|| format!("{path}: absent"))
        }
    }

    /// The tiles of a tree as tlog-tiles lays them out (levels of 256).
    fn tiles_of(l: &[Hash]) -> MemTiles {
        let mut out = HashMap::new();
        let mut level: Vec<Hash> = l.to_vec();
        let mut lvl = 0u32;
        while !level.is_empty() {
            for (n, chunk) in level.chunks(256).enumerate() {
                let bytes: Vec<u8> = chunk.iter().flatten().copied().collect();
                out.insert(tile_path(lvl, n as u64, chunk.len() as u64), bytes);
            }
            level = level
                .as_chunks::<256>()
                .0
                .iter()
                .map(|c| mth_of(c.as_slice()))
                .collect();
            lvl += 1;
        }
        MemTiles(out)
    }

    #[test]
    fn tile_paths_are_the_c2sp_ones() {
        assert_eq!(tile_index_path(1_234_067), "x001/x234/067");
        assert_eq!(tile_index_path(0), "000");
        assert_eq!(tile_index_path(1000), "x001/000");
        assert_eq!(tile_path(0, 5, 256), "tile/0/005");
        assert_eq!(tile_path(1, 0, 3), "tile/1/000.p/3");
    }

    /// Proofs built from tiles are the proofs of the tree, over three tile levels.
    #[test]
    fn proofs_from_tiles_equal_proofs_from_leaves() {
        let l = leaves(70_000);
        let tiles = tiles_of(&l);
        for size in [1u64, 2, 255, 256, 257, 511, 65_536, 65_537, 70_000] {
            let sl = &l[..size as usize];
            let mut t = TileHashes::new(&tiles, size);
            for i in [0, size / 2, size - 1] {
                let want = inclusion_proof(&mut Leaves(sl), i, size).unwrap();
                assert_eq!(
                    inclusion_proof(&mut t, i, size).unwrap(),
                    want,
                    "{size} {i}"
                );
            }
            for size1 in [1u64, 7, 256, 300, 65_536] {
                if size1 > size {
                    continue;
                }
                let want = consistency_proof(&mut Leaves(sl), size1, size).unwrap();
                assert_eq!(consistency_proof(&mut t, size1, size).unwrap(), want);
            }
            assert_eq!(t.mth(0, size).unwrap(), mth_of(sl));
        }
    }

    /// A tile of the wrong width, or missing: an error, never a hash.
    #[test]
    fn a_short_or_missing_tile_is_an_error() {
        let l = leaves(300);
        let mut tiles = tiles_of(&l);
        tiles.0.get_mut("tile/0/001.p/44").unwrap().pop();
        let mut t = TileHashes::new(&tiles, 300);
        assert!(t.mth(0, 300).is_err());
        let mut t = TileHashes::new(&tiles, 301);
        assert!(t.mth(0, 301).is_err(), "tile/0/001.p/45 does not exist");
        // An older size reads the prefix of the full tile when its partial
        // one is gone (a log MAY delete it once the full tile exists).
        let mut full = tiles_of(&leaves(512));
        let at_300 = tiles_of(&leaves(300));
        full.0
            .insert("tile/1/000.p/1".into(), at_300.0["tile/1/000.p/1"].clone());
        assert!(!full.0.contains_key("tile/0/001.p/44"));
        let mut t = TileHashes::new(&full, 300);
        assert_eq!(t.mth(0, 300).unwrap(), mth_of(&leaves(300)));
    }

    #[test]
    fn vkeys_round_trip_and_ids_are_checked() {
        let s = sk("log");
        let k = key("example.com/log", KeyKind::Ed25519, &s);
        assert_eq!(NoteKey::parse(&k.vkey()).unwrap(), k);
        let w = key("witness.example", KeyKind::CosignatureV1, &sk("w"));
        assert_eq!(NoteKey::parse(&w.vkey()).unwrap(), w);
        // Another name with the same key: another ID.
        let renamed = k.vkey().replacen("example.com/log", "example.com/other", 1);
        assert!(NoteKey::parse(&renamed)
            .unwrap_err()
            .contains("is not the key's"));
        assert!(NoteKey::parse("a+b").is_err());
        assert!(NoteKey::parse("bad name+00000000+AQ==").is_err());
    }

    fn signed(text: &str, signers: &[(&NoteKey, &SigningKey, u64)]) -> String {
        let mut s = format!("{text}\n");
        for (k, secret, t) in signers {
            s.push_str(&sign_line(text, k, secret, *t));
        }
        s
    }

    #[test]
    fn a_checkpoint_signed_by_its_log_and_cosigned_opens() {
        let (ls, ws) = (sk("log"), sk("witness"));
        let (lk, wk) = (
            key("example.com/log", KeyKind::Ed25519, &ls),
            key("witness.example", KeyKind::CosignatureV1, &ws),
        );
        let cp = Checkpoint {
            origin: "example.com/log".into(),
            size: 9,
            root: mth_of(&leaves(9)),
            extensions: vec![],
        };
        let note = signed(&cp.text(), &[(&lk, &ls, 0), (&wk, &ws, 1_792_000_000)]);
        let n = parse_note(&note).unwrap();
        assert_eq!(parse_checkpoint(&n.text).unwrap(), cp);
        let v = open_note(&n, &[lk.clone(), wk.clone()]).unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[1].time, Some(1_792_000_000));
        // Unknown keys are ignored: the log key alone still opens it.
        assert_eq!(open_note(&n, std::slice::from_ref(&lk)).unwrap().len(), 1);
    }

    /// SECURITY: not signed, signed by another key under the same name, a
    /// changed text, or a known key whose signature fails: refused.
    #[test]
    fn an_unsigned_or_falsely_signed_checkpoint_is_refused() {
        let (ls, other) = (sk("log"), sk("someone else"));
        let lk = key("example.com/log", KeyKind::Ed25519, &ls);
        let text = "example.com/log\n9\n47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=\n";
        assert!(parse_note(text).is_err(), "no signature at all");
        // Same name, other key: another ID, so unknown, so nothing verifies.
        let impostor = key("example.com/log", KeyKind::Ed25519, &other);
        let note = signed(text, &[(&impostor, &other, 0)]);
        let err = open_note(&parse_note(&note).unwrap(), std::slice::from_ref(&lk)).unwrap_err();
        assert!(err.contains("no signature of a known key"), "{err}");
        // The log's signature over another text.
        let mut note = signed(text, &[(&lk, &ls, 0)]);
        note = note.replacen("\n9\n", "\n10\n", 1);
        let err = open_note(&parse_note(&note).unwrap(), std::slice::from_ref(&lk)).unwrap_err();
        assert!(err.contains("does not verify"), "{err}");
        // A known witness whose cosignature fails rejects the note, even with a good log signature.
        let (ws, wk) = (sk("w"), key("w.example", KeyKind::CosignatureV1, &sk("w")));
        let good = signed(text, &[(&lk, &ls, 0), (&wk, &ws, 5)]);
        let tampered = {
            let mut n = parse_note(&good).unwrap();
            n.sigs[1].bytes[7] ^= 1; // the timestamp, covered by the signature
            n
        };
        assert!(open_note(&tampered, &[lk.clone(), wk.clone()]).is_err());
        // Duplicate lines of one key.
        let dup = format!("{good}{}", sign_line(text, &lk, &ls, 0));
        assert!(parse_note(&dup)
            .unwrap_err()
            .contains("two signature lines"));
    }

    #[test]
    fn checkpoints_and_proofs_parse_strictly() {
        assert!(parse_checkpoint("o\n01\n47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=\n").is_err());
        assert!(parse_checkpoint("o\n1\n47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFV=\n").is_err());
        assert!(parse_checkpoint("o\n1\n").is_err());
        let cp =
            parse_checkpoint("o\n0\n47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=\next\n").unwrap();
        assert_eq!(cp.extensions, ["ext"]);
        assert_eq!(cp.root, empty_root());
        let l = leaves(3);
        let p = format_tlog_proof(
            1,
            &inclusion_proof(&mut Leaves(&l), 1, 3).unwrap(),
            "o\n3\nr\n\n— k AAAA\n",
        );
        let parsed = parse_tlog_proof(&p).unwrap();
        assert_eq!(parsed.index, 1);
        assert_eq!(parsed.hashes.len(), 2);
        assert_eq!(parsed.checkpoint, "o\n3\nr\n\n— k AAAA\n");
        assert!(parse_tlog_proof(&p.replacen("@v1", "@v2", 1)).is_err());
        assert!(parse_tlog_proof(&p.replacen("index 1", "index 01", 1)).is_err());
        let with_extra = p.replacen("index 1", "extra AQID\nindex 1", 1);
        assert_eq!(
            parse_tlog_proof(&with_extra).unwrap().extra,
            Some(vec![1, 2, 3])
        );
    }
}
