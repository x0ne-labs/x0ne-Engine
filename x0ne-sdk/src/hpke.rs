//! HPKE (RFC 9180), base mode, one suite: DHKEM(X25519, HKDF-SHA256),
//! HKDF-SHA256, AES-128-GCM — the suite of RFC 9180 Appendix A.1 and of the
//! RFC 9458 example, whose vectors `tests/rfc_vectors.rs` checks byte for
//! byte.
//!
//! Built on the primitives already in the workspace lock (curve25519-dalek,
//! hmac, sha2, aes-gcm): this crate adds no dependency to the lock. A vetted
//! HPKE crate is the alternative; the vectors are what stands for this one.
//!
//! The caller brings the randomness ([`setup_base_s`] takes the ephemeral
//! key's input keying material): this module never draws, so a browser
//! passes `crypto.getRandomValues` output and a test is deterministic.

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes128Gcm, Key, Nonce};
use curve25519_dalek::montgomery::MontgomeryPoint;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use zeroize::{Zeroize, Zeroizing};

pub const KEM_X25519_HKDF_SHA256: u16 = 0x0020;
pub const KDF_HKDF_SHA256: u16 = 0x0001;
pub const AEAD_AES_128_GCM: u16 = 0x0001;
/// Encapsulated key, public key and private key lengths of X25519.
pub const N_ENC: usize = 32;
pub const N_PK: usize = 32;
pub const N_SK: usize = 32;
/// AES-128-GCM key and nonce lengths; HKDF-SHA256 output length.
pub const N_K: usize = 16;
pub const N_N: usize = 12;
pub const N_H: usize = 32;
/// HKDF-Expand can give at most 255 blocks.
const MAX_EXPAND: usize = 255 * N_H;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HpkeError {
    #[error("X25519 output is all zero: the public key has low order (RFC 9180 §7.1.4)")]
    ZeroSharedSecret,
    #[error("AEAD authentication failed")]
    Open,
    #[error("the context's sequence number is exhausted")]
    MessageLimit,
    #[error("export of {0} bytes is longer than HKDF-SHA256 gives")]
    ExportLength(usize),
    #[error("{0} bytes of ephemeral keying material: RFC 9180 §7.1.3 wants at least {N_SK}")]
    IkmShort(usize),
    #[error("a held private key of {0} bytes; X25519 takes {N_SK}")]
    KeyLength(usize),
}

/// Memory that holds a private scalar in place of this crate's own zeroed
/// array: the TD keeps its gateway key in `memfd_secret` memory, else
/// `mlock`ed.
pub trait SecretScalar: Send + Sync {
    /// The scalar's bytes, read for one X25519 operation: 32 of them.
    fn expose(&self) -> &[u8];
}

/// An X25519 private key, zeroed when dropped, or held by a [`SecretScalar`].
pub struct PrivateKey {
    scalar: Scalar,
    public: [u8; N_PK],
}

enum Scalar {
    Own(Zeroizing<[u8; N_SK]>),
    Held(Box<dyn SecretScalar>),
}

impl PrivateKey {
    pub fn from_bytes(bytes: [u8; N_SK]) -> Self {
        let scalar = Zeroizing::new(bytes);
        let public = MontgomeryPoint::mul_base_clamped(*scalar).to_bytes();
        Self {
            scalar: Scalar::Own(scalar),
            public,
        }
    }

    /// This key, its scalar moved into the memory `store` returns, which
    /// reads it once; the copy this value held is wiped. Memory that gives
    /// back other than 32 bytes fails each operation ([`HpkeError::KeyLength`]).
    ///
    /// # Errors
    /// `store`'s.
    pub fn held_by<S: SecretScalar + 'static, E>(
        self,
        store: impl FnOnce(&[u8]) -> Result<S, E>,
    ) -> Result<Self, E> {
        let held = store(self.expose())?;
        Ok(Self {
            scalar: Scalar::Held(Box::new(held)),
            public: self.public,
        })
    }

    fn expose(&self) -> &[u8] {
        match &self.scalar {
            Scalar::Own(s) => &s[..],
            Scalar::Held(h) => h.expose(),
        }
    }

    pub fn public_key(&self) -> [u8; N_PK] {
        self.public
    }

    fn dh(&self, public: &[u8; N_PK]) -> Result<Zeroizing<[u8; 32]>, HpkeError> {
        let bytes = self.expose();
        let scalar = Zeroizing::new(
            <[u8; N_SK]>::try_from(bytes).map_err(|_| HpkeError::KeyLength(bytes.len()))?,
        );
        let shared = Zeroizing::new(MontgomeryPoint(*public).mul_clamped(*scalar).to_bytes());
        // Constant-time enough for an all-zero test: the value is about to be
        // discarded either way, and which way leaks only "low order".
        if shared.iter().fold(0u8, |acc, b| acc | b) == 0 {
            return Err(HpkeError::ZeroSharedSecret);
        }
        Ok(shared)
    }
}

/// Whether `public` has low order (RFC 7748 §6.1, RFC 9180 §7.1.4): every
/// X25519 output with it is all zero, so anyone would share a sender's
/// "secret". A clamped scalar is a multiple of 8, so one fixed probe answers
/// for every point of order 1, 2, 4 or 8, canonical encoding or not; the
/// probe is not a multiple of the group order, so no honest key reads as low.
pub fn is_low_order(public: &[u8; N_PK]) -> bool {
    let mut probe = [0u8; N_SK];
    probe[0] = 0x08;
    probe[31] = 0x40;
    let out = MontgomeryPoint(*public).mul_clamped(probe).to_bytes();
    out.iter().fold(0u8, |acc, b| acc | b) == 0
}

impl std::fmt::Debug for PrivateKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PrivateKey(..)")
    }
}

/// HKDF-SHA256 (RFC 5869), as RFC 9180 and RFC 9458 use it.
pub mod kdf {
    use super::*;

    pub fn extract(salt: &[u8], ikm: &[u8]) -> Zeroizing<[u8; N_H]> {
        let salt = if salt.is_empty() {
            &[0u8; N_H][..]
        } else {
            salt
        };
        let mut mac = <Hmac<Sha256> as hmac::KeyInit>::new_from_slice(salt)
            .expect("HMAC takes a key of any length");
        mac.update(ikm);
        Zeroizing::new(mac.finalize().into_bytes().into())
    }

    /// `len` bytes of HKDF-Expand; at most 255 × 32.
    pub fn expand(prk: &[u8], info: &[u8], len: usize) -> Result<Zeroizing<Vec<u8>>, HpkeError> {
        if len > MAX_EXPAND {
            return Err(HpkeError::ExportLength(len));
        }
        let mut out = Zeroizing::new(Vec::with_capacity(len));
        let mut previous: Zeroizing<Vec<u8>> = Zeroizing::new(Vec::new());
        let mut counter = 1u8;
        while out.len() < len {
            let mut mac = <Hmac<Sha256> as hmac::KeyInit>::new_from_slice(prk)
                .expect("HMAC takes a key of any length");
            mac.update(&previous);
            mac.update(info);
            mac.update(&[counter]);
            let block: [u8; N_H] = mac.finalize().into_bytes().into();
            let take = (len - out.len()).min(N_H);
            out.extend_from_slice(&block[..take]);
            previous = Zeroizing::new(block.to_vec());
            counter = counter.wrapping_add(1);
        }
        Ok(out)
    }
}

const HPKE_V1: &[u8] = b"HPKE-v1";

fn kem_suite_id() -> [u8; 5] {
    let mut id = *b"KEM\0\0";
    id[3..].copy_from_slice(&KEM_X25519_HKDF_SHA256.to_be_bytes());
    id
}

fn hpke_suite_id() -> [u8; 10] {
    let mut id = [0u8; 10];
    id[..4].copy_from_slice(b"HPKE");
    id[4..6].copy_from_slice(&KEM_X25519_HKDF_SHA256.to_be_bytes());
    id[6..8].copy_from_slice(&KDF_HKDF_SHA256.to_be_bytes());
    id[8..10].copy_from_slice(&AEAD_AES_128_GCM.to_be_bytes());
    id
}

fn labeled_extract(suite: &[u8], salt: &[u8], label: &[u8], ikm: &[u8]) -> Zeroizing<[u8; N_H]> {
    let mut labeled = Zeroizing::new(Vec::with_capacity(
        7 + suite.len() + label.len() + ikm.len(),
    ));
    labeled.extend_from_slice(HPKE_V1);
    labeled.extend_from_slice(suite);
    labeled.extend_from_slice(label);
    labeled.extend_from_slice(ikm);
    kdf::extract(salt, &labeled)
}

fn labeled_expand(
    suite: &[u8],
    prk: &[u8],
    label: &[u8],
    info: &[u8],
    len: usize,
) -> Result<Zeroizing<Vec<u8>>, HpkeError> {
    let l = u16::try_from(len).map_err(|_| HpkeError::ExportLength(len))?;
    let mut labeled = Vec::with_capacity(9 + suite.len() + label.len() + info.len());
    labeled.extend_from_slice(&l.to_be_bytes());
    labeled.extend_from_slice(HPKE_V1);
    labeled.extend_from_slice(suite);
    labeled.extend_from_slice(label);
    labeled.extend_from_slice(info);
    kdf::expand(prk, &labeled, len)
}

/// RFC 9180 §7.1.3 `DeriveKeyPair` for X25519.
pub fn derive_key_pair(ikm: &[u8]) -> (PrivateKey, [u8; N_PK]) {
    let suite = kem_suite_id();
    let prk = labeled_extract(&suite, b"", b"dkp_prk", ikm);
    let sk = labeled_expand(&suite, &*prk, b"sk", b"", N_SK).expect("32 bytes expand");
    let mut bytes = [0u8; N_SK];
    bytes.copy_from_slice(&sk);
    let key = PrivateKey::from_bytes(bytes);
    bytes.zeroize();
    let pk = key.public_key();
    (key, pk)
}

fn extract_and_expand(dh: &[u8], kem_context: &[u8]) -> Zeroizing<Vec<u8>> {
    let suite = kem_suite_id();
    let eae_prk = labeled_extract(&suite, b"", b"eae_prk", dh);
    labeled_expand(&suite, &*eae_prk, b"shared_secret", kem_context, N_H).expect("32 bytes expand")
}

fn encap_with(
    sk_e: &PrivateKey,
    pk_r: &[u8; N_PK],
) -> Result<(Zeroizing<Vec<u8>>, [u8; N_ENC]), HpkeError> {
    let dh = sk_e.dh(pk_r)?;
    let enc = sk_e.public_key();
    let mut kem_context = [0u8; N_ENC + N_PK];
    kem_context[..N_ENC].copy_from_slice(&enc);
    kem_context[N_ENC..].copy_from_slice(pk_r);
    Ok((extract_and_expand(&*dh, &kem_context), enc))
}

fn decap(enc: &[u8; N_ENC], sk_r: &PrivateKey) -> Result<Zeroizing<Vec<u8>>, HpkeError> {
    let dh = sk_r.dh(enc)?;
    let mut kem_context = [0u8; N_ENC + N_PK];
    kem_context[..N_ENC].copy_from_slice(enc);
    kem_context[N_ENC..].copy_from_slice(&sk_r.public_key());
    Ok(extract_and_expand(&*dh, &kem_context))
}

/// One direction of an HPKE context: the key, the base nonce, the exporter
/// secret and the sequence number (RFC 9180 §5.2).
pub struct Context {
    key: Zeroizing<[u8; N_K]>,
    base_nonce: [u8; N_N],
    exporter_secret: Zeroizing<[u8; N_H]>,
    seq: u64,
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("seq", &self.seq)
            .finish_non_exhaustive()
    }
}

fn key_schedule_base(shared_secret: &[u8], info: &[u8]) -> Context {
    const MODE_BASE: u8 = 0;
    let suite = hpke_suite_id();
    let psk_id_hash = labeled_extract(&suite, b"", b"psk_id_hash", b"");
    let info_hash = labeled_extract(&suite, b"", b"info_hash", info);
    let mut context = Vec::with_capacity(1 + 2 * N_H);
    context.push(MODE_BASE);
    context.extend_from_slice(&*psk_id_hash);
    context.extend_from_slice(&*info_hash);
    let secret = labeled_extract(&suite, shared_secret, b"secret", b"");
    let key = labeled_expand(&suite, &*secret, b"key", &context, N_K).expect("16 bytes");
    let nonce = labeled_expand(&suite, &*secret, b"base_nonce", &context, N_N).expect("12 bytes");
    let exp = labeled_expand(&suite, &*secret, b"exp", &context, N_H).expect("32 bytes");
    let mut c = Context {
        key: Zeroizing::new([0u8; N_K]),
        base_nonce: [0u8; N_N],
        exporter_secret: Zeroizing::new([0u8; N_H]),
        seq: 0,
    };
    c.key.copy_from_slice(&key);
    c.base_nonce.copy_from_slice(&nonce);
    c.exporter_secret.copy_from_slice(&exp);
    c
}

/// The sender's context to `pk_r`: `(enc, context)`. The ephemeral key is
/// `DeriveKeyPair(ikm_e)`; `ikm_e` must be fresh random bytes, at least 32.
pub fn setup_base_s(
    pk_r: &[u8; N_PK],
    info: &[u8],
    ikm_e: &[u8],
) -> Result<([u8; N_ENC], Context), HpkeError> {
    let sk_e = ephemeral_key(ikm_e)?;
    setup_base_s_with_ephemeral(pk_r, info, &sk_e)
}

/// The sender's ephemeral key from `ikm_e`, the caller's fresh random bytes:
/// refused under `Nsk` of them (RFC 9180 §7.1.3). Every path that seals
/// derives its ephemeral key here, so none seals with a weaker key.
pub fn ephemeral_key(ikm_e: &[u8]) -> Result<PrivateKey, HpkeError> {
    if ikm_e.len() < N_SK {
        return Err(HpkeError::IkmShort(ikm_e.len()));
    }
    Ok(derive_key_pair(ikm_e).0)
}

/// [`setup_base_s`] with the ephemeral private key given: for vectors that
/// state the key rather than its input keying material (RFC 9458 Appendix A).
pub fn setup_base_s_with_ephemeral(
    pk_r: &[u8; N_PK],
    info: &[u8],
    sk_e: &PrivateKey,
) -> Result<([u8; N_ENC], Context), HpkeError> {
    let (shared_secret, enc) = encap_with(sk_e, pk_r)?;
    Ok((enc, key_schedule_base(&shared_secret, info)))
}

/// The receiver's context for `enc`.
pub fn setup_base_r(
    enc: &[u8; N_ENC],
    sk_r: &PrivateKey,
    info: &[u8],
) -> Result<Context, HpkeError> {
    let shared_secret = decap(enc, sk_r)?;
    Ok(key_schedule_base(&shared_secret, info))
}

/// AES-128-GCM seal under a key and nonce given (RFC 9458 §4.4 derives its
/// own for the response).
pub fn aead_seal(key: &[u8; N_K], nonce: &[u8; N_N], aad: &[u8], pt: &[u8]) -> Vec<u8> {
    Aes128Gcm::new(&Key::<Aes128Gcm>::from(*key))
        .encrypt(&Nonce::from(*nonce), Payload { msg: pt, aad })
        .expect("AES-GCM seal of an in-memory buffer")
}

pub fn aead_open(
    key: &[u8; N_K],
    nonce: &[u8; N_N],
    aad: &[u8],
    ct: &[u8],
) -> Result<Vec<u8>, HpkeError> {
    Aes128Gcm::new(&Key::<Aes128Gcm>::from(*key))
        .decrypt(&Nonce::from(*nonce), Payload { msg: ct, aad })
        .map_err(|_| HpkeError::Open)
}

impl Context {
    fn nonce(&self) -> [u8; N_N] {
        let mut nonce = self.base_nonce;
        for (n, s) in nonce[N_N - 8..].iter_mut().zip(self.seq.to_be_bytes()) {
            *n ^= s;
        }
        nonce
    }

    fn step(&mut self) -> Result<(), HpkeError> {
        self.seq = self.seq.checked_add(1).ok_or(HpkeError::MessageLimit)?;
        Ok(())
    }

    pub fn seal(&mut self, aad: &[u8], pt: &[u8]) -> Result<Vec<u8>, HpkeError> {
        if self.seq == u64::MAX {
            return Err(HpkeError::MessageLimit);
        }
        let ct = aead_seal(&self.key, &self.nonce(), aad, pt);
        self.step()?;
        Ok(ct)
    }

    /// A failed open does not move the sequence number (RFC 9180 §5.2).
    pub fn open(&mut self, aad: &[u8], ct: &[u8]) -> Result<Vec<u8>, HpkeError> {
        if self.seq == u64::MAX {
            return Err(HpkeError::MessageLimit);
        }
        let pt = aead_open(&self.key, &self.nonce(), aad, ct)?;
        self.step()?;
        Ok(pt)
    }

    /// RFC 9180 §5.3 `Context.Export`.
    pub fn export(
        &self,
        exporter_context: &[u8],
        len: usize,
    ) -> Result<Zeroizing<Vec<u8>>, HpkeError> {
        labeled_expand(
            &hpke_suite_id(),
            &*self.exporter_secret,
            b"sec",
            exporter_context,
            len,
        )
    }
}
