// SPDX-License-Identifier: Apache-2.0
//! A closed OpenPGP reader for one job: a detached signature over an image
//! reference, made by a signing subkey bound to a pinned primary key
//! (RFC 9580). The profile is the one x0ne-OS `archive-key.sh` generates and
//! `release-sign.sh` signs with, read on GnuPG 2.4.7 output
//! (`tests/archive_key/`, 2026-10-16): version 4 keys and signatures, EdDSA
//! (algorithm 22) on Ed25519, SHA-256/384/512, binary packets. Anything else
//! is refused, not approximated: no RSA, no v6, no armor, no partial length,
//! no unknown critical subpacket. Ed25519 and the hashes are RustCrypto's.

use ed25519_dalek::{Signature, VerifyingKey};
use sha1::Sha1;
use sha2::{Digest, Sha256, Sha384, Sha512};

/// An exported public key is a few hundred bytes; the cap bounds a mistake.
pub const MAX_KEYRING_BYTES: usize = 64 * 1024;
/// A detached Ed25519 signature from GnuPG 2.4.7 is 119 bytes.
pub const MAX_SIGNATURE_BYTES: u64 = 4096;
/// A signature dated this far ahead of the clock is refused.
pub const CLOCK_SKEW_SECS: u64 = 300;

const TAG_SIGNATURE: u8 = 2;
const TAG_PUBLIC_KEY: u8 = 6;
const TAG_USER_ID: u8 = 13;
const TAG_PUBLIC_SUBKEY: u8 = 14;

const PK_EDDSA: u8 = 22;
/// 1.3.6.1.4.1.11591.15.1, Ed25519 (RFC 9580 §9.2).
const OID_ED25519: [u8; 9] = [0x2b, 0x06, 0x01, 0x04, 0x01, 0xda, 0x47, 0x0f, 0x01];
const HASHES: [u8; 3] = [8, 9, 10];

const SIG_BINARY: u8 = 0x00;
const SIG_CERT_FIRST: u8 = 0x10;
const SIG_CERT_LAST: u8 = 0x13;
const SIG_SUBKEY_BINDING: u8 = 0x18;
const SIG_PRIMARY_BINDING: u8 = 0x19;
const SIG_DIRECT_KEY: u8 = 0x1f;
const SIG_KEY_REVOCATION: u8 = 0x20;
const SIG_SUBKEY_REVOCATION: u8 = 0x28;
const SIG_CERT_REVOCATION: u8 = 0x30;

const SUB_CREATED: u8 = 2;
const SUB_SIG_EXPIRES: u8 = 3;
const SUB_KEY_EXPIRES: u8 = 9;
const SUB_ISSUER: u8 = 16;
const SUB_KEY_FLAGS: u8 = 27;
const SUB_EMBEDDED: u8 = 32;
const SUB_ISSUER_FPR: u8 = 33;
/// Understood: the ones above, revocation reason, and preferences that do
/// not bear on a verification. A critical subpacket outside this list voids
/// the signature (RFC 9580 §5.2.3.7).
const KNOWN: [u8; 16] = [2, 3, 9, 11, 16, 21, 22, 23, 25, 27, 29, 30, 32, 33, 34, 39];
const FLAG_SIGN: u8 = 0x02;

#[derive(Clone, Copy, Debug)]
struct Packet<'a> {
    tag: u8,
    body: &'a [u8],
}

/// Old and new packet formats, definite lengths only.
fn packets(mut b: &[u8]) -> Result<Vec<Packet<'_>>, String> {
    let mut out = Vec::new();
    while let Some(&ctb) = b.first() {
        if ctb & 0x80 == 0 {
            return Err(format!("byte {ctb:#04x} does not start an OpenPGP packet"));
        }
        let at = |i: usize| {
            b.get(i)
                .map(|&x| usize::from(x))
                .ok_or_else(|| "truncated packet header".to_string())
        };
        let (tag, header, len): (u8, usize, usize) = if ctb & 0x40 != 0 {
            let tag = ctb & 0x3f;
            match at(1)? {
                n @ 0..=191 => (tag, 2, n),
                n @ 192..=223 => (tag, 3, ((n - 192) << 8) + at(2)? + 192),
                255 => (
                    tag,
                    6,
                    (at(2)? << 24) | (at(3)? << 16) | (at(4)? << 8) | at(5)?,
                ),
                _ => return Err("partial body length: refused outside literal data".into()),
            }
        } else {
            let tag = (ctb >> 2) & 0x0f;
            match ctb & 0x03 {
                0 => (tag, 2, at(1)?),
                1 => (tag, 3, (at(1)? << 8) | at(2)?),
                2 => (
                    tag,
                    5,
                    (at(1)? << 24) | (at(2)? << 16) | (at(3)? << 8) | at(4)?,
                ),
                _ => return Err("indeterminate packet length".into()),
            }
        };
        let end = header
            .checked_add(len)
            .filter(|&e| e <= b.len())
            .ok_or("truncated packet")?;
        out.push(Packet {
            tag,
            body: &b[header..end],
        });
        b = &b[end..];
    }
    Ok(out)
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, i: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .i
            .checked_add(n)
            .filter(|&e| e <= self.b.len())
            .ok_or("truncated")?;
        let s = &self.b[self.i..end];
        self.i = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<usize, String> {
        let s = self.take(2)?;
        Ok((usize::from(s[0]) << 8) | usize::from(s[1]))
    }
    fn u32(&mut self) -> Result<u32, String> {
        let s = self.take(4)?;
        Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn mpi(&mut self) -> Result<(usize, &'a [u8]), String> {
        let bits = self.u16()?;
        Ok((bits, self.take(bits.div_ceil(8))?))
    }
    fn done(&self) -> bool {
        self.i == self.b.len()
    }
}

fn be32(s: &[u8]) -> Result<u32, String> {
    <[u8; 4]>::try_from(s)
        .map(u32::from_be_bytes)
        .map_err(|_| format!("a {}-byte time field, 4 expected", s.len()))
}

pub fn hex_upper(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02X}")).collect()
}

/// `YYYY-MM-DD` of a Unix time, UTC (H. Hinnant, civil_from_days).
pub fn utc_date(secs: u64) -> String {
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

#[derive(Clone, Debug)]
pub(crate) struct Key {
    body: Vec<u8>,
    created: u32,
    point: [u8; 32],
    fingerprint: [u8; 20],
}

impl Key {
    /// `0x99 ‖ len16 ‖ body`: how a key enters a hash and its fingerprint.
    fn framing(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(3 + self.body.len());
        v.push(0x99);
        v.extend_from_slice(&(self.body.len() as u16).to_be_bytes());
        v.extend_from_slice(&self.body);
        v
    }
    fn fpr(&self) -> String {
        hex_upper(&self.fingerprint)
    }
}

fn parse_key(body: &[u8]) -> Result<Key, String> {
    let mut r = Reader::new(body);
    let version = r.u8()?;
    if version != 4 {
        return Err(format!("key version {version}: only version 4 is accepted"));
    }
    let created = r.u32()?;
    let algo = r.u8()?;
    if algo != PK_EDDSA {
        return Err(format!(
            "key algorithm {algo}: only EdDSA on Ed25519 (22) is accepted"
        ));
    }
    let oid_len = usize::from(r.u8()?);
    if r.take(oid_len)? != OID_ED25519 {
        return Err("key curve is not Ed25519".into());
    }
    let (bits, q) = r.mpi()?;
    if bits != 263 || q.len() != 33 || q[0] != 0x40 {
        return Err("key point is not a native Ed25519 point (0x40, 263 bits)".into());
    }
    if !r.done() {
        return Err("trailing bytes in a key packet".into());
    }
    let mut point = [0u8; 32];
    point.copy_from_slice(&q[1..]);
    VerifyingKey::from_bytes(&point).map_err(|_| "key point is not on Ed25519".to_string())?;
    let mut h = Sha1::new();
    h.update([0x99]);
    h.update((body.len() as u16).to_be_bytes());
    h.update(body);
    Ok(Key {
        body: body.to_vec(),
        created,
        point,
        fingerprint: h.finalize().into(),
    })
}

#[derive(Clone, Debug)]
struct Sig {
    sig_type: u8,
    hash: u8,
    /// Version through the hashed subpackets: what the trailer counts.
    hashed_prefix: Vec<u8>,
    left16: [u8; 2],
    rs: [u8; 64],
    created: u32,
    sig_expires: Option<u32>,
    key_expires: Option<u32>,
    key_flags: Option<u8>,
    embedded: Option<Vec<u8>>,
    issuer_fpr: Option<[u8; 20]>,
    issuer_id: Option<[u8; 8]>,
}

fn subpackets(mut b: &[u8]) -> Result<Vec<(bool, u8, &[u8])>, String> {
    let mut out = Vec::new();
    while let Some(&o1) = b.first() {
        let at = |i: usize| {
            b.get(i)
                .map(|&x| usize::from(x))
                .ok_or_else(|| "truncated subpacket".to_string())
        };
        let (header, len): (usize, usize) = match usize::from(o1) {
            n @ 0..=191 => (1, n),
            n @ 192..=254 => (2, ((n - 192) << 8) + at(1)? + 192),
            _ => (5, (at(1)? << 24) | (at(2)? << 16) | (at(3)? << 8) | at(4)?),
        };
        if len == 0 {
            return Err("empty subpacket".into());
        }
        let end = header
            .checked_add(len)
            .filter(|&e| e <= b.len())
            .ok_or("truncated subpacket")?;
        let t = b[header];
        out.push((t & 0x80 != 0, t & 0x7f, &b[header + 1..end]));
        b = &b[end..];
    }
    Ok(out)
}

/// Once per signature; an identical repeat (issuer in both areas) is allowed.
fn set_once<T: PartialEq>(slot: &mut Option<T>, v: T, what: &str) -> Result<(), String> {
    match slot {
        Some(old) if *old != v => Err(format!("two different {what} subpackets")),
        Some(_) => Ok(()),
        None => {
            *slot = Some(v);
            Ok(())
        }
    }
}

fn parse_sig(body: &[u8]) -> Result<Sig, String> {
    let mut r = Reader::new(body);
    let version = r.u8()?;
    if version != 4 {
        return Err(format!(
            "signature version {version}: only version 4 is accepted"
        ));
    }
    let sig_type = r.u8()?;
    let pk = r.u8()?;
    if pk != PK_EDDSA {
        return Err(format!(
            "signature algorithm {pk}: only EdDSA on Ed25519 (22) is accepted"
        ));
    }
    let hash = r.u8()?;
    if !HASHES.contains(&hash) {
        return Err(format!(
            "hash algorithm {hash}: only SHA-256, SHA-384 and SHA-512 are accepted"
        ));
    }
    let hl = r.u16()?;
    let hashed = r.take(hl)?;
    let hashed_prefix = body[..6 + hl].to_vec();
    let ul = r.u16()?;
    let unhashed = r.take(ul)?;
    let left = r.take(2)?;
    let (_, sr) = r.mpi()?;
    let (_, ss) = r.mpi()?;
    if !r.done() {
        return Err("trailing bytes in a signature packet".into());
    }
    if sr.len() > 32 || ss.len() > 32 {
        return Err("signature value longer than Ed25519's".into());
    }
    let mut rs = [0u8; 64];
    rs[32 - sr.len()..32].copy_from_slice(sr);
    rs[64 - ss.len()..].copy_from_slice(ss);
    let mut s = Sig {
        sig_type,
        hash,
        hashed_prefix,
        left16: [left[0], left[1]],
        rs,
        created: 0,
        sig_expires: None,
        key_expires: None,
        key_flags: None,
        embedded: None,
        issuer_fpr: None,
        issuer_id: None,
    };
    let mut created = None;
    for (in_hashed, area) in [(true, hashed), (false, unhashed)] {
        for (critical, t, data) in subpackets(area)? {
            if critical && !KNOWN.contains(&t) {
                return Err(format!("unknown critical subpacket {t}"));
            }
            match (in_hashed, t) {
                (true, SUB_CREATED) => set_once(&mut created, be32(data)?, "creation time")?,
                (true, SUB_SIG_EXPIRES) => {
                    set_once(&mut s.sig_expires, be32(data)?, "signature expiration")?
                }
                (true, SUB_KEY_EXPIRES) => {
                    set_once(&mut s.key_expires, be32(data)?, "key expiration")?
                }
                (true, SUB_KEY_FLAGS) => set_once(
                    &mut s.key_flags,
                    *data.first().ok_or("empty key flags")?,
                    "key flags",
                )?,
                (_, SUB_EMBEDDED) => {
                    set_once(&mut s.embedded, data.to_vec(), "embedded signature")?
                }
                (_, SUB_ISSUER_FPR) => {
                    let fpr = match data {
                        [4, rest @ ..] => <[u8; 20]>::try_from(rest).ok(),
                        _ => None,
                    }
                    .ok_or("issuer fingerprint is not a version 4 fingerprint")?;
                    set_once(&mut s.issuer_fpr, fpr, "issuer fingerprint")?
                }
                (_, SUB_ISSUER) => {
                    let id =
                        <[u8; 8]>::try_from(data).map_err(|_| "issuer key ID is not 8 bytes")?;
                    set_once(&mut s.issuer_id, id, "issuer key ID")?
                }
                _ => {}
            }
        }
    }
    s.created = created.ok_or("no signature creation time in the hashed area")?;
    Ok(s)
}

impl Sig {
    fn names(&self, k: &Key) -> bool {
        match (self.issuer_fpr, self.issuer_id) {
            (Some(f), _) => f == k.fingerprint,
            (None, Some(id)) => id == k.fingerprint[12..],
            (None, None) => true,
        }
    }
    fn issuer_text(&self) -> String {
        match (self.issuer_fpr, self.issuer_id) {
            (Some(f), _) => format!("key {}", hex_upper(&f)),
            (None, Some(id)) => format!("key ID {}", hex_upper(&id)),
            (None, None) => "an unnamed key".into(),
        }
    }
    fn expired_at(&self) -> Option<u64> {
        self.sig_expires
            .filter(|&e| e != 0)
            .map(|e| u64::from(self.created) + u64::from(e))
    }
}

fn digest<D: Digest>(parts: &[&[u8]], prefix: &[u8], trailer: &[u8]) -> Vec<u8> {
    let mut d = D::new();
    for p in parts {
        d.update(p);
    }
    d.update(prefix);
    d.update(trailer);
    d.finalize().to_vec()
}

/// RFC 9580 §5.2.4: the signed data, the hashed part of the signature, then
/// `04 FF ‖ len32`; Ed25519 signs that digest.
fn verify(s: &Sig, key: &Key, parts: &[&[u8]]) -> Result<(), String> {
    let mut trailer = vec![0x04, 0xff];
    trailer.extend_from_slice(&(s.hashed_prefix.len() as u32).to_be_bytes());
    let d = match s.hash {
        8 => digest::<Sha256>(parts, &s.hashed_prefix, &trailer),
        9 => digest::<Sha384>(parts, &s.hashed_prefix, &trailer),
        _ => digest::<Sha512>(parts, &s.hashed_prefix, &trailer),
    };
    if d[..2] != s.left16 {
        return Err("the signed data differs (hash prefix)".into());
    }
    let vk = VerifyingKey::from_bytes(&key.point).map_err(|_| "key point is not on Ed25519")?;
    vk.verify_strict(&d, &Signature::from_bytes(&s.rs))
        .map_err(|_| "the Ed25519 signature does not verify".to_string())
}

#[derive(Clone, Debug)]
struct Binding {
    created: u32,
    flags: u8,
    expires_at: Option<u64>,
    sig_expires_at: Option<u64>,
}

#[derive(Clone, Debug)]
struct Subkey {
    key: Key,
    binding: Option<Binding>,
    revoked: bool,
}

#[derive(Clone, Debug, Default)]
struct UserId {
    raw: Vec<u8>,
    /// The newest self-signature: (created, key expiration seconds).
    newest: Option<(u32, Option<u32>)>,
    revoked: bool,
}

/// What signed: named in every verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signer {
    pub primary_fingerprint: String,
    pub subkey_fingerprint: String,
    pub signed_at_unix: u64,
    /// A user ID of the key says TEST: a test build's key, never a release
    /// pin (`archive_key::pinned` refuses it).
    pub test_key: bool,
}

impl Signer {
    /// The subkey's long key ID: the last 16 hex digits of its v4
    /// fingerprint (RFC 9580 §5.5.4.2), as `gpg --keyid-format long` prints it.
    pub fn subkey_key_id(&self) -> &str {
        let f = &self.subkey_fingerprint;
        f.get(f.len().saturating_sub(16)..).unwrap_or(f)
    }
}

/// A public key as `gpg --export` writes it, every self-signature verified.
#[derive(Clone, Debug)]
pub struct Cert {
    primary: Key,
    primary_expires_at: Option<u64>,
    primary_revoked: bool,
    user_ids: Vec<String>,
    subkeys: Vec<Subkey>,
}

impl Cert {
    pub fn parse(bytes: &[u8]) -> Result<Cert, String> {
        if bytes.len() > MAX_KEYRING_BYTES {
            return Err(format!(
                "keyring of {} bytes, cap {MAX_KEYRING_BYTES}",
                bytes.len()
            ));
        }
        let ps = packets(bytes)?;
        let (first, rest) = ps.split_first().ok_or("empty keyring")?;
        if first.tag != TAG_PUBLIC_KEY {
            return Err(format!(
                "a keyring starts with a public key packet, not tag {}",
                first.tag
            ));
        }
        let primary = parse_key(first.body)?;
        let pf = primary.framing();
        enum At {
            Primary,
            Uid(usize),
            Sub(usize),
        }
        let mut at = At::Primary;
        let mut uids: Vec<UserId> = Vec::new();
        let mut subkeys: Vec<Subkey> = Vec::new();
        let mut direct: Option<(u32, Option<u32>)> = None;
        let mut primary_revoked = false;
        for p in rest {
            match p.tag {
                TAG_USER_ID => {
                    uids.push(UserId {
                        raw: p.body.to_vec(),
                        ..UserId::default()
                    });
                    at = At::Uid(uids.len() - 1);
                }
                TAG_PUBLIC_SUBKEY => {
                    let key = parse_key(p.body)?;
                    // A second copy could carry a binding without the revocation.
                    if key.fingerprint == primary.fingerprint
                        || subkeys.iter().any(|k| k.key.fingerprint == key.fingerprint)
                    {
                        return Err(format!("subkey {} appears twice", key.fpr()));
                    }
                    subkeys.push(Subkey {
                        key,
                        binding: None,
                        revoked: false,
                    });
                    at = At::Sub(subkeys.len() - 1);
                }
                TAG_SIGNATURE => {
                    let s = parse_sig(p.body)?;
                    match (&at, s.sig_type) {
                        (At::Primary, SIG_KEY_REVOCATION) => {
                            verify(&s, &primary, &[&pf])
                                .map_err(|e| format!("primary key revocation: {e}"))?;
                            primary_revoked = true;
                        }
                        (At::Primary, SIG_DIRECT_KEY) => {
                            verify(&s, &primary, &[&pf])
                                .map_err(|e| format!("direct-key signature: {e}"))?;
                            newest(&mut direct, &s);
                        }
                        (At::Uid(i), SIG_CERT_FIRST..=SIG_CERT_LAST | SIG_CERT_REVOCATION) => {
                            let u = &mut uids[*i];
                            let mut uf = vec![0xb4];
                            uf.extend_from_slice(&(u.raw.len() as u32).to_be_bytes());
                            uf.extend_from_slice(&u.raw);
                            // Not by the primary: a third party's certification, ignored.
                            if !s.names(&primary) || verify(&s, &primary, &[&pf, &uf]).is_err() {
                                continue;
                            }
                            if s.sig_type == SIG_CERT_REVOCATION {
                                u.revoked = true;
                            } else {
                                newest(&mut u.newest, &s);
                            }
                        }
                        (At::Sub(i), SIG_SUBKEY_BINDING) => {
                            let b = binding(&primary, &subkeys[*i].key, &s)?;
                            let sub = &mut subkeys[*i];
                            if sub
                                .binding
                                .as_ref()
                                .is_none_or(|old| old.created <= b.created)
                            {
                                sub.binding = Some(b);
                            }
                        }
                        (At::Sub(i), SIG_SUBKEY_REVOCATION) => {
                            let sub = &mut subkeys[*i];
                            verify(&s, &primary, &[&pf, &sub.key.framing()]).map_err(|e| {
                                format!("revocation of subkey {}: {e}", sub.key.fpr())
                            })?;
                            sub.revoked = true;
                        }
                        (_, t) => {
                            return Err(format!(
                                "signature type {t:#04x} where an exported key has none"
                            ))
                        }
                    }
                }
                t => {
                    return Err(format!(
                        "packet tag {t} is not part of an exported public key"
                    ))
                }
            }
        }
        // The primary's validity: its newest self-signature, on a live user ID or direct.
        let newest_self = uids
            .iter()
            .filter(|u| !u.revoked)
            .filter_map(|u| u.newest)
            .chain(direct)
            .max_by_key(|(c, _)| *c)
            .ok_or("no valid self-signature by the primary key")?;
        Ok(Cert {
            primary_expires_at: newest_self
                .1
                .filter(|&e| e != 0)
                .map(|e| u64::from(primary.created) + u64::from(e)),
            user_ids: uids
                .into_iter()
                .filter(|u| !u.revoked && u.newest.is_some())
                .map(|u| String::from_utf8_lossy(&u.raw).into_owned())
                .collect(),
            primary,
            primary_revoked,
            subkeys,
        })
    }

    /// Upper-case hex, as `gpg` and `FINGERPRINTS` print it.
    pub fn primary_fingerprint(&self) -> String {
        self.primary.fpr()
    }

    pub fn user_ids(&self) -> &[String] {
        &self.user_ids
    }

    /// A detached signature (binary, one packet, type 0x00) over `data`, by
    /// a live signing subkey of this key, judged at `now`.
    pub fn verify_detached(
        &self,
        data: &[u8],
        signature: &[u8],
        now: u64,
    ) -> Result<Signer, String> {
        let primary = self.primary.fpr();
        if self.primary_revoked {
            return Err(format!("the archive key {primary} is revoked"));
        }
        if let Some(e) = self.primary_expires_at.filter(|&e| now >= e) {
            return Err(format!(
                "the archive key {primary} expired on {}",
                utc_date(e)
            ));
        }
        if signature.starts_with(b"-----BEGIN PGP") {
            return Err("an armored signature: the signature is binary (gpg --detach-sign, without --armor)".into());
        }
        let ps = packets(signature)?;
        let [p] = ps.as_slice() else {
            return Err(format!(
                "{} packets: a detached signature is one signature packet",
                ps.len()
            ));
        };
        if p.tag != TAG_SIGNATURE {
            return Err(format!("packet tag {}, not a signature", p.tag));
        }
        let s = parse_sig(p.body)?;
        if s.sig_type != SIG_BINARY {
            return Err(format!(
                "signature type {:#04x}: a reference is signed as binary data (0x00)",
                s.sig_type
            ));
        }
        let created = u64::from(s.created);
        if created > now.saturating_add(CLOCK_SKEW_SECS) {
            return Err(format!(
                "signature dated {}, after this machine's clock",
                utc_date(created)
            ));
        }
        if let Some(e) = s.expired_at().filter(|&e| now >= e) {
            return Err(format!("signature expired on {}", utc_date(e)));
        }
        if s.names(&self.primary) && (s.issuer_fpr.is_some() || s.issuer_id.is_some()) {
            return Err(format!(
                "signed by the primary key {primary}: a reference is signed by a signing subkey bound to it, never by the primary"
            ));
        }
        let named: Vec<&Subkey> = self.subkeys.iter().filter(|k| s.names(&k.key)).collect();
        if named.is_empty() {
            return Err(format!(
                "signed by {}, which is not a subkey of the archive key {primary}",
                s.issuer_text()
            ));
        }
        let mut why = String::new();
        for k in named {
            let fpr = k.key.fpr();
            if let Err(e) = usable(k, &s, now) {
                why = format!("subkey {fpr}: {e}");
                continue;
            }
            match verify(&s, &k.key, &[data]) {
                Ok(()) => {
                    return Ok(Signer {
                        primary_fingerprint: primary,
                        subkey_fingerprint: fpr,
                        signed_at_unix: created,
                        test_key: self.user_ids.iter().any(|u| u.contains("TEST")),
                    })
                }
                Err(e) => why = format!("subkey {fpr}: {e}"),
            }
        }
        Err(why)
    }
}

fn newest(slot: &mut Option<(u32, Option<u32>)>, s: &Sig) {
    if slot.is_none_or(|(c, _)| c <= s.created) {
        *slot = Some((s.created, s.key_expires));
    }
}

/// A subkey binding by the primary; a signing subkey must sign back
/// (primary key binding, 0x19), or a stolen signature could adopt it.
fn binding(primary: &Key, sub: &Key, s: &Sig) -> Result<Binding, String> {
    let (pf, sf) = (primary.framing(), sub.framing());
    verify(s, primary, &[&pf, &sf])
        .map_err(|e| format!("binding of subkey {} to the primary: {e}", sub.fpr()))?;
    let flags = s.key_flags.unwrap_or(0);
    if flags & FLAG_SIGN != 0 {
        let body = s.embedded.as_ref().ok_or_else(|| {
            format!(
                "signing subkey {} has no back-signature (primary key binding)",
                sub.fpr()
            )
        })?;
        let back = parse_sig(body)?;
        if back.sig_type != SIG_PRIMARY_BINDING {
            return Err(format!(
                "subkey {}: embedded signature of type {:#04x}, 0x19 expected",
                sub.fpr(),
                back.sig_type
            ));
        }
        verify(&back, sub, &[&pf, &sf])
            .map_err(|e| format!("back-signature of subkey {}: {e}", sub.fpr()))?;
    }
    if s.created < sub.created {
        return Err(format!(
            "binding of subkey {} predates the subkey",
            sub.fpr()
        ));
    }
    Ok(Binding {
        created: s.created,
        flags,
        expires_at: s
            .key_expires
            .filter(|&e| e != 0)
            .map(|e| u64::from(sub.created) + u64::from(e)),
        sig_expires_at: s.expired_at(),
    })
}

fn usable(k: &Subkey, s: &Sig, now: u64) -> Result<(), String> {
    let b = k
        .binding
        .as_ref()
        .ok_or("not bound to the primary (no binding signature)")?;
    if k.revoked {
        return Err("revoked".into());
    }
    if b.flags & FLAG_SIGN == 0 {
        return Err("not a signing subkey (key flags)".into());
    }
    if let Some(e) = b.sig_expires_at.filter(|&e| now >= e) {
        return Err(format!("its binding signature expired on {}", utc_date(e)));
    }
    if let Some(e) = b.expires_at {
        if now >= e {
            return Err(format!("expired on {}", utc_date(e)));
        }
        if u64::from(s.created) >= e {
            return Err(format!(
                "the signature was made after it expired ({})",
                utc_date(e)
            ));
        }
    }
    if s.created < k.key.created {
        return Err("the signature predates the subkey".into());
    }
    Ok(())
}

/// Keys and signatures in the same profile, for tests and the TEST key of a
/// `test-archive-key` build (tests and examples, a self dev-dependency).
/// The shipped binaries never compile it (`tests/cloud_source_sweep.rs`).
#[cfg(any(test, feature = "test-archive-key"))]
pub mod build {
    use ed25519_dalek::{Signer as _, SigningKey};

    pub struct KeyPair {
        pub sk: SigningKey,
        pub created: u32,
    }

    impl KeyPair {
        pub fn from_seed(seed: [u8; 32], created: u32) -> Self {
            Self {
                sk: SigningKey::from_bytes(&seed),
                created,
            }
        }
        pub fn body(&self) -> Vec<u8> {
            let mut b = vec![4];
            b.extend_from_slice(&self.created.to_be_bytes());
            b.push(super::PK_EDDSA);
            b.push(super::OID_ED25519.len() as u8);
            b.extend_from_slice(&super::OID_ED25519);
            b.extend_from_slice(&[0x01, 0x07, 0x40]);
            b.extend_from_slice(self.sk.verifying_key().as_bytes());
            b
        }
        pub fn framing(&self) -> Vec<u8> {
            let body = self.body();
            let mut v = vec![0x99];
            v.extend_from_slice(&(body.len() as u16).to_be_bytes());
            v.extend_from_slice(&body);
            v
        }
        pub fn fingerprint(&self) -> [u8; 20] {
            super::parse_key(&self.body()).unwrap().fingerprint
        }
    }

    /// A new-format packet (GnuPG writes the old format: both are read).
    pub fn packet(tag: u8, body: &[u8]) -> Vec<u8> {
        let mut v = vec![0xc0 | tag];
        match body.len() {
            n @ 0..=191 => v.push(n as u8),
            n @ 192..=8383 => {
                let m = n - 192;
                v.extend_from_slice(&[(m >> 8) as u8 + 192, m as u8]);
            }
            n => {
                v.push(255);
                v.extend_from_slice(&(n as u32).to_be_bytes());
            }
        }
        v.extend_from_slice(body);
        v
    }

    pub fn subpacket(t: u8, data: &[u8]) -> Vec<u8> {
        assert!(data.len() < 191);
        let mut v = vec![(data.len() + 1) as u8, t];
        v.extend_from_slice(data);
        v
    }

    fn mpi(b: &[u8]) -> Vec<u8> {
        let lead = b.iter().take_while(|&&x| x == 0).count();
        let b = &b[lead..];
        let bits = match b.first() {
            Some(f) => (b.len() - 1) * 8 + (8 - f.leading_zeros() as usize),
            None => 0,
        };
        let mut v = (bits as u16).to_be_bytes().to_vec();
        v.extend_from_slice(b);
        v
    }

    pub struct Spec {
        pub sig_type: u8,
        pub hash: u8,
        pub created: u32,
        /// After creation time and issuer fingerprint.
        pub hashed: Vec<u8>,
        pub unhashed: Vec<u8>,
    }

    impl Spec {
        pub fn new(sig_type: u8, created: u32) -> Self {
            Self {
                sig_type,
                hash: 10,
                created,
                hashed: Vec::new(),
                unhashed: Vec::new(),
            }
        }
    }

    /// The body of a signature packet by `by` over `parts`.
    pub fn sig_body(by: &KeyPair, spec: &Spec, parts: &[&[u8]]) -> Vec<u8> {
        let mut issuer = vec![4];
        issuer.extend_from_slice(&by.fingerprint());
        let mut hashed = subpacket(super::SUB_ISSUER_FPR, &issuer);
        hashed.extend(subpacket(super::SUB_CREATED, &spec.created.to_be_bytes()));
        hashed.extend_from_slice(&spec.hashed);
        let mut prefix = vec![4, spec.sig_type, super::PK_EDDSA, spec.hash];
        prefix.extend_from_slice(&(hashed.len() as u16).to_be_bytes());
        prefix.extend_from_slice(&hashed);
        let mut trailer = vec![0x04, 0xff];
        trailer.extend_from_slice(&(prefix.len() as u32).to_be_bytes());
        let d = match spec.hash {
            8 => super::digest::<super::Sha256>(parts, &prefix, &trailer),
            9 => super::digest::<super::Sha384>(parts, &prefix, &trailer),
            _ => super::digest::<super::Sha512>(parts, &prefix, &trailer),
        };
        let sig = by.sk.sign(&d).to_bytes();
        let mut body = prefix;
        body.extend_from_slice(&(spec.unhashed.len() as u16).to_be_bytes());
        body.extend_from_slice(&spec.unhashed);
        body.extend_from_slice(&d[..2]);
        body.extend(mpi(&sig[..32]));
        body.extend(mpi(&sig[32..]));
        body
    }

    pub fn uid_framing(uid: &[u8]) -> Vec<u8> {
        let mut v = vec![0xb4];
        v.extend_from_slice(&(uid.len() as u32).to_be_bytes());
        v.extend_from_slice(uid);
        v
    }

    /// Primary, user ID, its self-signature (certify only), as archive-key.sh.
    pub fn primary_block(primary: &KeyPair, uid: &[u8], key_expires: u32) -> Vec<u8> {
        let mut spec = Spec::new(0x13, primary.created);
        spec.hashed = subpacket(super::SUB_KEY_FLAGS, &[0x01]);
        if key_expires != 0 {
            spec.hashed.extend(subpacket(
                super::SUB_KEY_EXPIRES,
                &key_expires.to_be_bytes(),
            ));
        }
        let mut v = packet(super::TAG_PUBLIC_KEY, &primary.body());
        v.extend(packet(super::TAG_USER_ID, uid));
        v.extend(packet(
            super::TAG_SIGNATURE,
            &sig_body(primary, &spec, &[&primary.framing(), &uid_framing(uid)]),
        ));
        v
    }

    /// A subkey and its binding by `binder`; `back` adds the subkey's back-signature.
    pub fn subkey_block(
        binder: &KeyPair,
        primary: &KeyPair,
        sub: &KeyPair,
        flags: u8,
        key_expires: u32,
        back: bool,
    ) -> Vec<u8> {
        let parts = [primary.framing(), sub.framing()];
        let parts: [&[u8]; 2] = [&parts[0], &parts[1]];
        let mut spec = Spec::new(0x18, sub.created);
        spec.hashed = subpacket(super::SUB_KEY_FLAGS, &[flags]);
        if key_expires != 0 {
            spec.hashed.extend(subpacket(
                super::SUB_KEY_EXPIRES,
                &key_expires.to_be_bytes(),
            ));
        }
        if back {
            let backsig = sig_body(sub, &Spec::new(0x19, sub.created), &parts);
            spec.unhashed = subpacket_long(super::SUB_EMBEDDED, &backsig);
        }
        let mut v = packet(super::TAG_PUBLIC_SUBKEY, &sub.body());
        v.extend(packet(
            super::TAG_SIGNATURE,
            &sig_body(binder, &spec, &parts),
        ));
        v
    }

    fn subpacket_long(t: u8, data: &[u8]) -> Vec<u8> {
        let n = data.len() + 1;
        let mut v = if n < 192 {
            vec![n as u8]
        } else {
            let m = n - 192;
            vec![(m >> 8) as u8 + 192, m as u8]
        };
        v.push(t);
        v.extend_from_slice(data);
        v
    }

    /// A subkey revocation (0x28) by the primary.
    pub fn subkey_revocation(primary: &KeyPair, sub: &KeyPair, created: u32) -> Vec<u8> {
        packet(
            super::TAG_SIGNATURE,
            &sig_body(
                primary,
                &Spec::new(0x28, created),
                &[&primary.framing(), &sub.framing()],
            ),
        )
    }

    /// A detached binary signature (0x00), as release-sign.sh makes it.
    pub fn detached(by: &KeyPair, data: &[u8], created: u32) -> Vec<u8> {
        packet(
            super::TAG_SIGNATURE,
            &sig_body(by, &Spec::new(0x00, created), &[data]),
        )
    }
}

#[cfg(all(test, not(x0ne_sdk)))]
mod tests {
    use super::build::*;
    use super::*;

    /// 2026-10-16T00:00:00Z, the GnuPG fixtures' creation time.
    const T0: u32 = 1_792_108_800;
    const DAY: u32 = 86_400;
    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/archive_key/");

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!("{FIXTURES}{name}")).unwrap()
    }

    fn kp(label: u8, created: u32) -> KeyPair {
        KeyPair::from_seed([label; 32], created)
    }

    fn keyring(primary: &KeyPair, sub: &KeyPair, sub_expires: u32) -> Vec<u8> {
        let mut v = primary_block(primary, b"x0ne test <test@invalid>", 5 * 365 * DAY);
        v.extend(subkey_block(
            primary,
            primary,
            sub,
            FLAG_SIGN,
            sub_expires,
            true,
        ));
        v
    }

    fn now() -> u64 {
        u64::from(T0 + DAY)
    }

    // ── GnuPG 2.4.7 output, x0ne OS procedure of 2026-10-16 (tests/archive_key/README.md) ──

    #[test]
    fn gnupg_keyring_and_signature_verify() {
        let cert = Cert::parse(&fixture("test-archive-keyring.gpg")).unwrap();
        assert_eq!(
            cert.primary_fingerprint(),
            "2F957D30FAE5CD616C23FD44AAC12AC5E4D7FDCB"
        );
        assert_eq!(
            cert.user_ids(),
            ["x0ne Engine TEST archive key, not for release <test@invalid>"]
        );
        let signer = cert
            .verify_detached(
                &fixture("reference.json"),
                &fixture("reference.json.sig"),
                now(),
            )
            .unwrap();
        assert_eq!(
            signer.subkey_fingerprint,
            "43B2C8EB81620E1C19492F2C86D2CD081726C748"
        );
        // gpg 2.4.7 --show-keys --keyid-format long: "sub ed25519/86D2CD081726C748".
        assert_eq!(signer.subkey_key_id(), "86D2CD081726C748");
        assert_eq!(signer.signed_at_unix, u64::from(T0) + 3600);
        assert!(signer.test_key, "its user ID says TEST");
    }

    /// SECURITY: one byte of the reference changed, anywhere, is refused.
    #[test]
    fn a_reference_changed_by_one_byte_is_refused() {
        let cert = Cert::parse(&fixture("test-archive-keyring.gpg")).unwrap();
        let sig = fixture("reference.json.sig");
        let good = fixture("reference.json");
        for i in [0, good.len() / 2, good.len() - 1] {
            let mut bad = good.clone();
            bad[i] ^= 0x01;
            let err = cert.verify_detached(&bad, &sig, now()).unwrap_err();
            assert!(
                err.contains("differs") || err.contains("does not verify"),
                "{err}"
            );
        }
        let mut longer = good.clone();
        longer.push(b'\n');
        assert!(cert.verify_detached(&longer, &sig, now()).is_err());
    }

    /// SECURITY: a signature by another key, even well formed, is refused.
    #[test]
    fn a_signature_by_another_key_is_refused() {
        let cert = Cert::parse(&fixture("test-archive-keyring.gpg")).unwrap();
        let err = cert
            .verify_detached(
                &fixture("reference.json"),
                &fixture("reference.json.rogue.sig"),
                now(),
            )
            .unwrap_err();
        assert!(err.contains("not a subkey of the archive key"), "{err}");
        // Its own key accepts it: the refusal above is the key, not the bytes.
        let rogue = Cert::parse(&fixture("rogue-keyring.gpg")).unwrap();
        assert!(rogue
            .verify_detached(
                &fixture("reference.json"),
                &fixture("reference.json.rogue.sig"),
                now()
            )
            .is_ok());
    }

    /// SECURITY: GnuPG's expired subkey: valid the day it signed, refused after.
    #[test]
    fn a_gnupg_subkey_past_its_expiry_is_refused() {
        let cert = Cert::parse(&fixture("expired-subkey-keyring.gpg")).unwrap();
        let (r, s) = (
            fixture("reference.json"),
            fixture("reference.json.expired.sig"),
        );
        // 2025-01-01T02:00:00Z, within the day the subkey lives.
        assert!(cert.verify_detached(&r, &s, 1_735_696_800).is_ok());
        let err = cert.verify_detached(&r, &s, now()).unwrap_err();
        assert!(err.contains("expired on 2025-01-02"), "{err}");
    }

    /// SECURITY: a revoked subkey (archive-key.sh revoke-subkey) refuses what
    /// it signed, before its revocation too; the rotated subkey signs.
    #[test]
    fn a_revoked_subkey_is_refused_and_its_successor_accepted() {
        let cert = Cert::parse(&fixture("revoked-subkey-keyring.gpg")).unwrap();
        let r = fixture("reference.json");
        let err = cert
            .verify_detached(&r, &fixture("reference.json.sig"), now())
            .unwrap_err();
        assert!(
            err.contains("43B2C8EB81620E1C19492F2C86D2CD081726C748: revoked"),
            "{err}"
        );
        let ok = cert
            .verify_detached(&r, &fixture("reference.json.rotated.sig"), now())
            .unwrap();
        assert_eq!(
            ok.subkey_fingerprint,
            "E0486A12674E4FBF617FD3A5B1D55A0047986463"
        );
    }

    /// SECURITY: a subkey bound by another primary, spliced into the archive
    /// key's packets, is refused with the whole keyring.
    #[test]
    fn a_gnupg_subkey_from_another_key_is_not_bound() {
        let good = fixture("test-archive-keyring.gpg");
        let rogue = fixture("rogue-keyring.gpg");
        let gp = packets(&good).unwrap();
        let rp = packets(&rogue).unwrap();
        let start = |ps: &[Packet]| ps.iter().position(|p| p.tag == TAG_PUBLIC_SUBKEY).unwrap();
        let offset = |all: &[u8], p: &Packet| p.body.as_ptr() as usize - all.as_ptr() as usize;
        // Header of the first subkey packet: 2 bytes (old format, one-byte length).
        let cut = offset(&good, &gp[start(&gp)]) - 2;
        let from = offset(&rogue, &rp[start(&rp)]) - 2;
        let mut spliced = good[..cut].to_vec();
        spliced.extend_from_slice(&rogue[from..]);
        let err = Cert::parse(&spliced).unwrap_err();
        assert!(
            err.contains("binding of subkey") && err.contains("to the primary"),
            "{err}"
        );
    }

    #[test]
    fn a_missing_or_empty_signature_is_refused() {
        let cert = Cert::parse(&fixture("test-archive-keyring.gpg")).unwrap();
        let r = fixture("reference.json");
        assert!(cert
            .verify_detached(&r, b"", now())
            .unwrap_err()
            .contains("0 packets"));
        let armored = b"-----BEGIN PGP SIGNATURE-----\n\n-----END PGP SIGNATURE-----\n";
        assert!(cert
            .verify_detached(&r, armored, now())
            .unwrap_err()
            .contains("armored"));
        let mut two = fixture("reference.json.sig");
        two.extend(fixture("reference.json.sig"));
        assert!(cert
            .verify_detached(&r, &two, now())
            .unwrap_err()
            .contains("2 packets"));
        // The keyring in place of a signature.
        assert!(cert
            .verify_detached(&r, &fixture("test-archive-keyring.gpg"), now())
            .is_err());
    }

    // ── The same refusals on keys built here, one property at a time ──

    #[test]
    fn a_built_keyring_verifies_and_reads_both_packet_formats() {
        let (p, s) = (kp(1, T0), kp(2, T0));
        let cert = Cert::parse(&keyring(&p, &s, 2 * 365 * DAY)).unwrap();
        let sig = detached(&s, b"reference", T0 + 60);
        let signer = cert.verify_detached(b"reference", &sig, now()).unwrap();
        assert_eq!(signer.subkey_fingerprint, hex_upper(&s.fingerprint()));
        assert_eq!(signer.primary_fingerprint, hex_upper(&p.fingerprint()));
        // "x0ne test <test@invalid>": no TEST in capitals, not a TEST key.
        assert!(!signer.test_key);
    }

    /// SECURITY: a subkey with no binding, or bound by another primary.
    #[test]
    fn an_unbound_subkey_is_refused() {
        let (p, s, other) = (kp(1, T0), kp(2, T0), kp(3, T0));
        let sig = detached(&s, b"reference", T0 + 60);
        // No binding signature at all.
        let mut bare = primary_block(&p, b"x0ne test <test@invalid>", 0);
        bare.extend(packet(TAG_PUBLIC_SUBKEY, &s.body()));
        let err = Cert::parse(&bare)
            .unwrap()
            .verify_detached(b"reference", &sig, now())
            .unwrap_err();
        assert!(err.contains("not bound"), "{err}");
        // Bound by another key.
        let mut foreign = primary_block(&p, b"x0ne test <test@invalid>", 0);
        foreign.extend(subkey_block(&other, &p, &s, FLAG_SIGN, 0, true));
        assert!(Cert::parse(&foreign)
            .unwrap_err()
            .contains("binding of subkey"));
    }

    /// SECURITY: a signing subkey without its back-signature can be adopted
    /// by anyone's primary; refused.
    #[test]
    fn a_signing_subkey_without_back_signature_is_refused() {
        let (p, s) = (kp(1, T0), kp(2, T0));
        let mut v = primary_block(&p, b"x0ne test <test@invalid>", 0);
        v.extend(subkey_block(&p, &p, &s, FLAG_SIGN, 0, false));
        assert!(Cert::parse(&v).unwrap_err().contains("no back-signature"));
    }

    /// A signing subkey bound by `p` under `binding` (key flags added), with
    /// a back-signature of type `back_type` made by the subkey.
    fn bound(p: &KeyPair, s: &KeyPair, mut binding: Spec, back_type: u8) -> Vec<u8> {
        let (pf, sf) = (p.framing(), s.framing());
        let parts: [&[u8]; 2] = [&pf, &sf];
        binding
            .hashed
            .extend(subpacket(SUB_KEY_FLAGS, &[FLAG_SIGN]));
        let back = sig_body(s, &Spec::new(back_type, s.created), &parts);
        binding.unhashed = subpacket(SUB_EMBEDDED, &back);
        let mut v = primary_block(p, b"x0ne test <test@invalid>", 0);
        v.extend(packet(TAG_PUBLIC_SUBKEY, &s.body()));
        v.extend(packet(TAG_SIGNATURE, &sig_body(p, &binding, &parts)));
        v
    }

    /// SECURITY: the embedded back-signature is a primary key binding
    /// (0x19). One of another type, even made by the subkey over the same
    /// keys, does not bind it.
    #[test]
    fn a_back_signature_of_another_type_is_refused() {
        let (p, s) = (kp(1, T0), kp(2, T0));
        let sig = detached(&s, b"reference", T0 + 60);
        let cert = Cert::parse(&bound(&p, &s, Spec::new(0x18, T0), 0x19)).unwrap();
        assert!(cert.verify_detached(b"reference", &sig, now()).is_ok());
        for back_type in [SIG_BINARY, SIG_SUBKEY_BINDING] {
            let err = Cert::parse(&bound(&p, &s, Spec::new(0x18, T0), back_type)).unwrap_err();
            assert!(err.contains("0x19 expected"), "{back_type:#04x}: {err}");
        }
    }

    /// The binding signature's own expiry (subpacket 3) ends the subkey,
    /// even with no key expiry.
    #[test]
    fn a_subkey_whose_binding_signature_expired_is_refused() {
        let (p, s) = (kp(1, T0), kp(2, T0));
        let mut binding = Spec::new(0x18, T0);
        binding.hashed = subpacket(SUB_SIG_EXPIRES, &DAY.to_be_bytes());
        let cert = Cert::parse(&bound(&p, &s, binding, 0x19)).unwrap();
        let sig = detached(&s, b"reference", T0 + 60);
        assert!(cert
            .verify_detached(b"reference", &sig, u64::from(T0 + DAY) - 1)
            .is_ok());
        let err = cert
            .verify_detached(b"reference", &sig, u64::from(T0 + DAY))
            .unwrap_err();
        assert!(
            err.contains("binding signature expired on 2026-10-17"),
            "{err}"
        );
    }

    /// A binding dated before the subkey it binds voids the keyring.
    #[test]
    fn a_binding_older_than_its_subkey_is_refused() {
        let (p, s) = (kp(1, T0), kp(2, T0 + DAY));
        assert!(Cert::parse(&bound(&p, &s, Spec::new(0x18, T0 + DAY), 0x19)).is_ok());
        let err = Cert::parse(&bound(&p, &s, Spec::new(0x18, T0 + DAY - 1), 0x19)).unwrap_err();
        assert!(err.contains("predates the subkey"), "{err}");
    }

    /// A signature dated before its subkey existed is refused.
    #[test]
    fn a_signature_older_than_its_subkey_is_refused() {
        let (p, s) = (kp(1, T0), kp(2, T0 + 3600));
        let cert = Cert::parse(&keyring(&p, &s, 0)).unwrap();
        let at_creation = detached(&s, b"reference", T0 + 3600);
        assert!(cert
            .verify_detached(b"reference", &at_creation, now())
            .is_ok());
        let early = detached(&s, b"reference", T0 + 3599);
        let err = cert
            .verify_detached(b"reference", &early, now())
            .unwrap_err();
        assert!(err.ends_with("the signature predates the subkey"), "{err}");
    }

    /// SECURITY: an expired subkey refuses now, even for a signature it made in time.
    #[test]
    fn an_expired_subkey_is_refused() {
        let (p, s) = (kp(1, T0), kp(2, T0));
        let cert = Cert::parse(&keyring(&p, &s, DAY)).unwrap();
        let sig = detached(&s, b"reference", T0 + 60);
        assert!(cert
            .verify_detached(b"reference", &sig, u64::from(T0) + 3600)
            .is_ok());
        let err = cert
            .verify_detached(b"reference", &sig, u64::from(T0 + DAY))
            .unwrap_err();
        assert!(err.contains("expired on 2026-10-17"), "{err}");
        // Dated at the expiry, judged 100 s before it (inside the clock skew).
        let late = detached(&s, b"reference", T0 + DAY);
        let err = cert
            .verify_detached(b"reference", &late, u64::from(T0 + DAY) - 100)
            .unwrap_err();
        assert!(err.contains("made after it expired"), "{err}");
    }

    #[test]
    fn an_expired_or_revoked_primary_refuses_everything() {
        let (p, s) = (kp(1, T0), kp(2, T0));
        let mut v = primary_block(&p, b"x0ne test <test@invalid>", 10 * DAY);
        v.extend(subkey_block(&p, &p, &s, FLAG_SIGN, 0, true));
        let cert = Cert::parse(&v).unwrap();
        let sig = detached(&s, b"reference", T0 + 60);
        assert!(cert.verify_detached(b"reference", &sig, now()).is_ok());
        assert!(cert
            .verify_detached(b"reference", &sig, u64::from(T0 + 10 * DAY))
            .unwrap_err()
            .contains("archive key"));
        let mut revoked = packet(TAG_PUBLIC_KEY, &p.body());
        revoked.extend(packet(
            TAG_SIGNATURE,
            &sig_body(&p, &Spec::new(0x20, T0 + 1), &[&p.framing()]),
        ));
        revoked.extend_from_slice(&v[packet(TAG_PUBLIC_KEY, &p.body()).len()..]);
        assert!(Cert::parse(&revoked)
            .unwrap()
            .verify_detached(b"reference", &sig, now())
            .unwrap_err()
            .contains("revoked"));
    }

    #[test]
    fn a_revocation_by_the_primary_revokes_the_subkey() {
        let (p, s) = (kp(1, T0), kp(2, T0));
        let mut v = keyring(&p, &s, 0);
        v.extend(subkey_revocation(&p, &s, T0 + 10));
        let sig = detached(&s, b"reference", T0 + 5);
        let err = Cert::parse(&v)
            .unwrap()
            .verify_detached(b"reference", &sig, now())
            .unwrap_err();
        assert!(err.ends_with("revoked"), "{err}");
        // A revocation that does not verify voids the keyring.
        let mut forged = keyring(&p, &s, 0);
        forged.extend(subkey_revocation(&kp(9, T0), &s, T0 + 10));
        assert!(Cert::parse(&forged)
            .unwrap_err()
            .contains("revocation of subkey"));
    }

    /// SECURITY: a revoked subkey cannot come back as a second, unrevoked
    /// copy of itself, nor the primary as its own subkey.
    #[test]
    fn a_subkey_listed_twice_voids_the_keyring() {
        let (p, s) = (kp(1, T0), kp(2, T0));
        let mut v = keyring(&p, &s, 0);
        v.extend(subkey_revocation(&p, &s, T0 + 10));
        v.extend(subkey_block(&p, &p, &s, FLAG_SIGN, 0, true));
        let err = Cert::parse(&v).unwrap_err();
        assert!(err.contains("appears twice"), "{err}");
        let mut v = primary_block(&p, b"x0ne test <test@invalid>", 0);
        v.extend(subkey_block(&p, &p, &p, FLAG_SIGN, 0, true));
        assert!(Cert::parse(&v).unwrap_err().contains("appears twice"));
    }

    #[test]
    fn a_subkey_that_may_not_sign_is_refused() {
        let (p, s) = (kp(1, T0), kp(2, T0));
        let mut v = primary_block(&p, b"x0ne test <test@invalid>", 0);
        v.extend(subkey_block(&p, &p, &s, 0x0c, 0, false));
        let sig = detached(&s, b"reference", T0 + 60);
        assert!(Cert::parse(&v)
            .unwrap()
            .verify_detached(b"reference", &sig, now())
            .unwrap_err()
            .contains("not a signing subkey"));
    }

    /// The primary certifies only: a document it signs is refused.
    #[test]
    fn a_signature_by_the_primary_is_refused() {
        let (p, s) = (kp(1, T0), kp(2, T0));
        let cert = Cert::parse(&keyring(&p, &s, 0)).unwrap();
        let sig = detached(&p, b"reference", T0 + 60);
        let err = cert.verify_detached(b"reference", &sig, now()).unwrap_err();
        assert!(err.contains("never by the primary"), "{err}");
    }

    /// SECURITY: the shape of x0ne-OS gen-test-key.sh, as `make cvm-sign`
    /// signs: one primary that may sign, no subkey. Refused, whatever the
    /// primary's key flags; the message does not call it certify-only.
    #[test]
    fn a_gnupg_primary_that_signs_alone_is_refused() {
        let cert = Cert::parse(&fixture("primary-only-keyring.gpg")).unwrap();
        assert_eq!(
            cert.primary_fingerprint(),
            "8023EEA58F1ECFCB4F9210365086A8DF9D53A766"
        );
        let err = cert
            .verify_detached(
                &fixture("reference.json"),
                &fixture("reference.json.primary-only.sig"),
                now(),
            )
            .unwrap_err();
        assert!(
            err.contains("signed by the primary key 8023EEA58F1ECFCB4F9210365086A8DF9D53A766"),
            "{err}"
        );
        assert!(err.contains("never by the primary"), "{err}");
        assert!(!err.contains("certifies only"), "{err}");
    }

    #[test]
    fn signature_shape_and_time_are_judged() {
        let (p, s) = (kp(1, T0), kp(2, T0));
        let cert = Cert::parse(&keyring(&p, &s, 0)).unwrap();
        let sign = |spec: &Spec| packet(TAG_SIGNATURE, &sig_body(&s, spec, &[b"reference"]));
        // From the future, beyond the skew.
        let future = detached(&s, b"reference", T0 + 2 * DAY);
        assert!(cert
            .verify_detached(b"reference", &future, now())
            .unwrap_err()
            .contains("after this machine's clock"));
        // Text mode (0x01) is not what release-sign.sh makes.
        assert!(cert
            .verify_detached(b"reference", &sign(&Spec::new(0x01, T0)), now())
            .unwrap_err()
            .contains("binary data"));
        // SHA-1 and MD5 are refused before any hashing.
        for weak in [1u8, 2] {
            let mut body = sig_body(&s, &Spec::new(0x00, T0), &[b"reference"]);
            body[3] = weak;
            assert!(cert
                .verify_detached(b"reference", &packet(TAG_SIGNATURE, &body), now())
                .unwrap_err()
                .contains("hash algorithm"));
        }
        // SHA-256 and SHA-384 are accepted.
        for h in [8u8, 9] {
            let mut spec = Spec::new(0x00, T0);
            spec.hash = h;
            assert!(cert
                .verify_detached(b"reference", &sign(&spec), now())
                .is_ok());
        }
        // An unknown critical subpacket voids it; a known one does not.
        let mut spec = Spec::new(0x00, T0);
        spec.hashed = subpacket(0x80 | 31, &[0; 4]);
        assert!(cert
            .verify_detached(b"reference", &sign(&spec), now())
            .unwrap_err()
            .contains("critical subpacket 31"));
        // An expired signature.
        let mut spec = Spec::new(0x00, T0);
        spec.hashed = subpacket(SUB_SIG_EXPIRES, &60u32.to_be_bytes());
        assert!(cert
            .verify_detached(b"reference", &sign(&spec), now())
            .unwrap_err()
            .contains("signature expired"));
        // Two creation times.
        let mut spec = Spec::new(0x00, T0);
        spec.hashed = subpacket(SUB_CREATED, &(T0 + 1).to_be_bytes());
        assert!(cert
            .verify_detached(b"reference", &sign(&spec), now())
            .unwrap_err()
            .contains("two different creation time"));
    }

    /// SECURITY: the hashed area is under the signature; a byte of it changed
    /// (here the creation time) is refused.
    #[test]
    fn a_changed_hashed_subpacket_is_refused() {
        let (p, s) = (kp(1, T0), kp(2, T0));
        let cert = Cert::parse(&keyring(&p, &s, 0)).unwrap();
        let mut sig = detached(&s, b"reference", T0 + 60);
        // Packet header 2, signature header 6, issuer fingerprint subpacket 23.
        let created = 2 + 6 + 23;
        assert_eq!(sig[created..created + 2], [5, SUB_CREATED]);
        sig[created + 5] ^= 1;
        let err = cert.verify_detached(b"reference", &sig, now()).unwrap_err();
        assert!(
            err.contains("differs") || err.contains("does not verify"),
            "{err}"
        );
    }

    #[test]
    fn malformed_packets_are_refused_not_panicked() {
        let good = fixture("reference.json.sig");
        let cert = Cert::parse(&fixture("test-archive-keyring.gpg")).unwrap();
        let r = fixture("reference.json");
        // The unhashed area (issuer key ID) is outside the signature by design
        // (RFC 9580 §5.2.3): a change there must not panic, the rest must refuse.
        let hl = usize::from(good[6]) << 8 | usize::from(good[7]);
        let ul_at = 2 + 6 + hl;
        let ul = usize::from(good[ul_at]) << 8 | usize::from(good[ul_at + 1]);
        let unhashed = ul_at + 2..ul_at + 2 + ul;
        for n in 0..good.len() {
            assert!(cert.verify_detached(&r, &good[..n], now()).is_err());
            for bit in [0x01u8, 0x80] {
                let mut bad = good.clone();
                bad[n] ^= bit;
                let verdict = cert.verify_detached(&r, &bad, now());
                if !unhashed.contains(&n) {
                    assert!(verdict.is_err(), "byte {n} bit {bit:#x}");
                }
            }
        }
        let ring = fixture("test-archive-keyring.gpg");
        for n in 0..ring.len() {
            let _ = Cert::parse(&ring[..n]);
        }
        // Partial and indeterminate lengths.
        assert!(packets(&[0xc2, 0xe0, 0]).unwrap_err().contains("partial"));
        assert!(packets(&[0x8b, 0]).unwrap_err().contains("indeterminate"));
    }

    #[test]
    fn utc_dates() {
        assert_eq!(utc_date(0), "1970-01-01");
        assert_eq!(utc_date(u64::from(T0)), "2026-10-16");
        assert_eq!(utc_date(951_782_400), "2000-02-29");
    }
}
