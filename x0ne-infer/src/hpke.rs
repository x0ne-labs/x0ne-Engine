//! HPKE (RFC 9180), base mode, one message: DHKEM(X25519, HKDF-SHA256),
//! HKDF-SHA256, AES-256-GCM. The model key travels to a TD under it
//! (`model_key.rs`): sealed to the X25519 key the TD's quote names, it opens
//! only where that private key is.
//!
//! Built from primitives already in the lock: X25519 from `curve25519-dalek`,
//! HKDF from `hmac` and `sha2`, the AEAD from `aes-gcm`. Checked against the
//! RFC's own vectors (A.1.1, AES-128-GCM) and the CFRG vector of this exact
//! suite (AES-256-GCM), byte for byte: a roundtrip cannot see a construction
//! that is wrong the same way on both sides.
//!
//! Single shot only: sequence number 0, so the nonce is `base_nonce`. No PSK,
//! no Auth mode, no exporter: nothing here needs them.

use aes_gcm::aead::{Aead as _, KeyInit as _, Payload};
use aes_gcm::{Aes128Gcm, Aes256Gcm, Key, Nonce};
use curve25519_dalek::montgomery::MontgomeryPoint;
use hmac::{Hmac, Mac as _};
use sha2::Sha256;
use zeroize::Zeroizing;

/// DHKEM(X25519, HKDF-SHA256).
pub const KEM_ID: u16 = 0x0020;
/// HKDF-SHA256.
pub const KDF_ID: u16 = 0x0001;
/// AES-256-GCM: the suite's AEAD.
pub const AEAD_ID: u16 = 0x0002;
/// The suite, as `RecipientDoc::suite` names it.
pub const SUITE: &str = "DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, AES-256-GCM";
/// Bytes of an encapsulated key and of an X25519 public key.
pub const N_ENC: usize = 32;
const N_SECRET: usize = 32;
const N_NONCE: usize = 12;
const MODE_BASE: u8 = 0x00;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HpkeError {
    /// RFC 9180 §7.1.4: an all-zero X25519 output (a low-order point).
    #[error("X25519 shared secret is all zero: the public key is of low order")]
    ZeroSharedSecret,
    /// The AEAD refused: another key, another `info`, another `aad`, or a
    /// changed ciphertext. Never a wrong plaintext.
    #[error("HPKE open failed: another recipient key, context or ciphertext")]
    Open,
}

/// The AEADs this module knows: the suite's, and AES-128-GCM for the RFC's
/// own vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Aead {
    #[cfg_attr(not(test), allow(dead_code))] // the RFC's own vectors (A.1.1)
    Aes128Gcm,
    Aes256Gcm,
}

impl Aead {
    fn id(self) -> u16 {
        match self {
            Aead::Aes128Gcm => 0x0001,
            Aead::Aes256Gcm => AEAD_ID,
        }
    }
    fn key_len(self) -> usize {
        match self {
            Aead::Aes128Gcm => 16,
            Aead::Aes256Gcm => 32,
        }
    }
}

/// `X25519(sk, pk)`, clamped as RFC 7748 says; all zero is refused.
fn dh(sk: &[u8; 32], pk: &[u8; 32]) -> Result<Zeroizing<[u8; 32]>, HpkeError> {
    let shared = Zeroizing::new(MontgomeryPoint(*pk).mul_clamped(*sk).to_bytes());
    // Constant time is not needed to compare with zero here: the check
    // refuses, it never leaks which bytes of a valid secret are what.
    if shared.iter().all(|b| *b == 0) {
        return Err(HpkeError::ZeroSharedSecret);
    }
    Ok(shared)
}

/// The public key of an X25519 private key.
pub fn public_key(sk: &[u8; 32]) -> [u8; 32] {
    MontgomeryPoint::mul_base_clamped(*sk).to_bytes()
}

/// RFC 9180 §7.1.3 `DeriveKeyPair` for X25519: the private key from 32
/// bytes or more of input keying material, and its public key.
pub fn derive_key_pair(ikm: &[u8]) -> (Zeroizing<[u8; 32]>, [u8; 32]) {
    let suite = kem_suite_id();
    let prk = labeled_extract(&suite, b"", b"dkp_prk", ikm);
    let sk: Zeroizing<[u8; 32]> = Zeroizing::new(
        labeled_expand(&suite, &prk[..], b"sk", b"", 32)[..]
            .try_into()
            .expect("32 bytes asked"),
    );
    let pk = public_key(&sk);
    (sk, pk)
}

// ── HKDF-SHA256 (RFC 5869) and the labeled forms of RFC 9180 §4 ──────────────

fn hmac(key: &[u8], parts: &[&[u8]]) -> Zeroizing<[u8; 32]> {
    let mut m = <Hmac<Sha256> as hmac::KeyInit>::new_from_slice(key)
        .expect("HMAC takes a key of any length");
    for p in parts {
        m.update(p);
    }
    Zeroizing::new(m.finalize().into_bytes().into())
}

fn extract(salt: &[u8], ikm: &[u8]) -> Zeroizing<[u8; 32]> {
    hmac(salt, &[ikm])
}

/// `L` is at most 255 × 32 by RFC 5869; this module asks for 32 at most.
fn expand(prk: &[u8], info: &[u8], len: usize) -> Zeroizing<Vec<u8>> {
    assert!(len <= 255 * 32, "HKDF-Expand: L too large");
    let mut out = Zeroizing::new(Vec::with_capacity(len));
    let mut t: Zeroizing<Vec<u8>> = Zeroizing::new(Vec::new());
    let mut i = 1u8;
    while out.len() < len {
        let block = hmac(prk, &[&t[..], info, &[i]]);
        t = Zeroizing::new(block.to_vec());
        let take = (len - out.len()).min(32);
        out.extend_from_slice(&block[..take]);
        i = i.wrapping_add(1);
    }
    out
}

fn labeled_extract(suite: &[u8], salt: &[u8], label: &[u8], ikm: &[u8]) -> Zeroizing<[u8; 32]> {
    let mut labeled = Zeroizing::new(Vec::with_capacity(
        7 + suite.len() + label.len() + ikm.len(),
    ));
    labeled.extend_from_slice(b"HPKE-v1");
    labeled.extend_from_slice(suite);
    labeled.extend_from_slice(label);
    labeled.extend_from_slice(ikm);
    extract(salt, &labeled)
}

fn labeled_expand(
    suite: &[u8],
    prk: &[u8],
    label: &[u8],
    info: &[u8],
    len: usize,
) -> Zeroizing<Vec<u8>> {
    let mut labeled = Vec::with_capacity(2 + 7 + suite.len() + label.len() + info.len());
    labeled.extend_from_slice(&(len as u16).to_be_bytes());
    labeled.extend_from_slice(b"HPKE-v1");
    labeled.extend_from_slice(suite);
    labeled.extend_from_slice(label);
    labeled.extend_from_slice(info);
    expand(prk, &labeled, len)
}

/// `"KEM" ‖ I2OSP(kem_id, 2)`.
fn kem_suite_id() -> Vec<u8> {
    let mut s = b"KEM".to_vec();
    s.extend_from_slice(&KEM_ID.to_be_bytes());
    s
}

/// `"HPKE" ‖ kem_id ‖ kdf_id ‖ aead_id`.
fn hpke_suite_id(aead: Aead) -> Vec<u8> {
    let mut s = b"HPKE".to_vec();
    s.extend_from_slice(&KEM_ID.to_be_bytes());
    s.extend_from_slice(&KDF_ID.to_be_bytes());
    s.extend_from_slice(&aead.id().to_be_bytes());
    s
}

/// DHKEM `ExtractAndExpand(dh, enc ‖ pkRm)`.
fn shared_secret(dh: &[u8; 32], enc: &[u8; 32], pk_r: &[u8; 32]) -> Zeroizing<Vec<u8>> {
    let suite = kem_suite_id();
    let prk = labeled_extract(&suite, b"", b"eae_prk", dh);
    let mut kem_context = [0u8; 64];
    kem_context[..32].copy_from_slice(enc);
    kem_context[32..].copy_from_slice(pk_r);
    labeled_expand(&suite, &prk[..], b"shared_secret", &kem_context, N_SECRET)
}

/// `KeySchedule(mode_base, shared_secret, info, "", "")`: the key and the
/// base nonce (the exporter secret is not derived: nothing exports).
fn key_schedule(aead: Aead, shared: &[u8], info: &[u8]) -> (Zeroizing<Vec<u8>>, [u8; N_NONCE]) {
    let suite = hpke_suite_id(aead);
    let psk_id_hash = labeled_extract(&suite, b"", b"psk_id_hash", b"");
    let info_hash = labeled_extract(&suite, b"", b"info_hash", info);
    let mut context = Vec::with_capacity(1 + 64);
    context.push(MODE_BASE);
    context.extend_from_slice(&psk_id_hash[..]);
    context.extend_from_slice(&info_hash[..]);
    let secret = labeled_extract(&suite, shared, b"secret", b"");
    let key = labeled_expand(&suite, &secret[..], b"key", &context, aead.key_len());
    let nonce = labeled_expand(&suite, &secret[..], b"base_nonce", &context, N_NONCE);
    (key, nonce[..].try_into().expect("12 bytes asked"))
}

fn aead_seal(aead: Aead, key: &[u8], nonce: &[u8; N_NONCE], aad: &[u8], pt: &[u8]) -> Vec<u8> {
    let payload = Payload { msg: pt, aad };
    let nonce = Nonce::from(*nonce);
    match aead {
        Aead::Aes128Gcm => {
            let key: [u8; 16] = key.try_into().expect("16-byte key");
            Aes128Gcm::new(&Key::<Aes128Gcm>::from(key)).encrypt(&nonce, payload)
        }
        Aead::Aes256Gcm => {
            let key: [u8; 32] = key.try_into().expect("32-byte key");
            Aes256Gcm::new(&Key::<Aes256Gcm>::from(key)).encrypt(&nonce, payload)
        }
    }
    .expect("AES-GCM encrypt is infallible for in-memory buffers")
}

fn aead_open(
    aead: Aead,
    key: &[u8],
    nonce: &[u8; N_NONCE],
    aad: &[u8],
    ct: &[u8],
) -> Result<Zeroizing<Vec<u8>>, HpkeError> {
    let payload = Payload { msg: ct, aad };
    let nonce = Nonce::from(*nonce);
    match aead {
        Aead::Aes128Gcm => {
            let key: [u8; 16] = key.try_into().expect("16-byte key");
            Aes128Gcm::new(&Key::<Aes128Gcm>::from(key)).decrypt(&nonce, payload)
        }
        Aead::Aes256Gcm => {
            let key: [u8; 32] = key.try_into().expect("32-byte key");
            Aes256Gcm::new(&Key::<Aes256Gcm>::from(key)).decrypt(&nonce, payload)
        }
    }
    .map(Zeroizing::new)
    .map_err(|_| HpkeError::Open)
}

/// Seal `pt` to `pk_r` under `info` and `aad`. `ikm_e`: 32 fresh random
/// bytes, the ephemeral key's seed (the caller owns randomness, so the
/// vectors can pin the output). Returns `(enc, ciphertext ‖ tag)`.
pub fn seal(
    pk_r: &[u8; 32],
    info: &[u8],
    aad: &[u8],
    pt: &[u8],
    ikm_e: &[u8; 32],
) -> Result<([u8; N_ENC], Vec<u8>), HpkeError> {
    seal_with(Aead::Aes256Gcm, pk_r, info, aad, pt, ikm_e)
}

/// Open what [`seal`] made for the key pair of `sk_r`.
pub fn open(
    sk_r: &[u8; 32],
    enc: &[u8; N_ENC],
    info: &[u8],
    aad: &[u8],
    ct: &[u8],
) -> Result<Zeroizing<Vec<u8>>, HpkeError> {
    open_with(Aead::Aes256Gcm, sk_r, enc, info, aad, ct)
}

pub(crate) fn seal_with(
    aead: Aead,
    pk_r: &[u8; 32],
    info: &[u8],
    aad: &[u8],
    pt: &[u8],
    ikm_e: &[u8; 32],
) -> Result<([u8; N_ENC], Vec<u8>), HpkeError> {
    let (sk_e, enc) = derive_key_pair(ikm_e);
    let dh = dh(&sk_e, pk_r)?;
    let shared = shared_secret(&dh, &enc, pk_r);
    let (key, nonce) = key_schedule(aead, &shared, info);
    Ok((enc, aead_seal(aead, &key, &nonce, aad, pt)))
}

pub(crate) fn open_with(
    aead: Aead,
    sk_r: &[u8; 32],
    enc: &[u8; N_ENC],
    info: &[u8],
    aad: &[u8],
    ct: &[u8],
) -> Result<Zeroizing<Vec<u8>>, HpkeError> {
    let dh = dh(sk_r, enc)?;
    let shared = shared_secret(&dh, enc, &public_key(sk_r));
    let (key, nonce) = key_schedule(aead, &shared, info);
    aead_open(aead, &key, &nonce, aad, ct)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        let s: String = s.split_whitespace().collect();
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }
    fn a32(s: &str) -> [u8; 32] {
        unhex(s).try_into().unwrap()
    }

    /// RFC 9180, Appendix A.1.1 (DHKEM(X25519, HKDF-SHA256), HKDF-SHA256,
    /// AES-128-GCM), base mode, sequence number 0, copied from the RFC text
    /// on 2026-10-22 (https://www.rfc-editor.org/rfc/rfc9180.txt).
    #[test]
    fn rfc9180_a_1_1_x25519_sha256_aes128gcm_byte_for_byte() {
        let info = unhex("4f6465206f6e2061204772656369616e2055726e");
        let ikm_e = a32("7268600d403fce431561aef583ee1613527cff655c1343f29812e66706df3234");
        let ikm_r = unhex("6db9df30aa07dd42ee5e8181afdb977e538f5e1fec8a06223f33f7013e525037");
        let (sk_e, pk_e) = derive_key_pair(&ikm_e);
        assert_eq!(
            &sk_e[..],
            &unhex("52c4a758a802cd8b936eceea314432798d5baf2d7e9235dc084ab1b9cfa2f736")[..]
        );
        assert_eq!(
            pk_e.to_vec(),
            unhex("37fda3567bdbd628e88668c3c8d7e97d1d1253b6d4ea6d44c150f741f1bf4431")
        );
        let (sk_r, pk_r) = derive_key_pair(&ikm_r);
        assert_eq!(
            &sk_r[..],
            &unhex("4612c550263fc8ad58375df3f557aac531d26850903e55a9f23f21d8534e8ac8")[..]
        );
        assert_eq!(
            pk_r.to_vec(),
            unhex("3948cfe0ad1ddb695d780e59077195da6c56506b027329794ab02bca80815c4d")
        );
        assert_eq!(public_key(&sk_r), pk_r);
        let pt = unhex("4265617574792069732074727574682c20747275746820626561757479");
        let aad = unhex("436f756e742d30");
        let (enc, ct) = seal_with(Aead::Aes128Gcm, &pk_r, &info, &aad, &pt, &ikm_e).unwrap();
        assert_eq!(enc, pk_e, "enc is the ephemeral public key");
        assert_eq!(
            ct,
            unhex(
                "f938558b5d72f1a23810b4be2ab4f84331acc02fc97babc53a52ae8218a355a9
                 6d8770ac83d07bea87e13c512a"
            )
        );
        let opened = open_with(Aead::Aes128Gcm, &sk_r, &enc, &info, &aad, &ct).unwrap();
        assert_eq!(&opened[..], &pt[..]);
    }

    /// The suite this module serves, AES-256-GCM (aead_id 2), from the CFRG
    /// test vectors the RFC cites ([TestVectors], commit 5f503c5,
    /// test-vectors.json sha256 61fc662f…), mode 0, kem 32, kdf 1, aead 2,
    /// first encryption.
    #[test]
    fn cfrg_vector_x25519_sha256_aes256gcm_byte_for_byte() {
        let info = unhex("4f6465206f6e2061204772656369616e2055726e");
        let ikm_e = a32("2cd7c601cefb3d42a62b04b7a9041494c06c7843818e0ce28a8f704ae7ab20f9");
        let ikm_r = unhex("dac33b0e9db1b59dbbea58d59a14e7b5896e9bdf98fad6891e99d1686492b9ee");
        let (sk_r, pk_r) = derive_key_pair(&ikm_r);
        assert_eq!(
            &sk_r[..],
            &unhex("497b4502664cfea5d5af0b39934dac72242a74f8480451e1aee7d6a53320333d")[..]
        );
        assert_eq!(
            pk_r.to_vec(),
            unhex("430f4b9859665145a6b1ba274024487bd66f03a2dd577d7753c68d7d7d00c00c")
        );
        let pt = unhex("4265617574792069732074727574682c20747275746820626561757479");
        let aad = unhex("436f756e742d30");
        let (enc, ct) = seal(&pk_r, &info, &aad, &pt, &ikm_e).unwrap();
        assert_eq!(
            enc.to_vec(),
            unhex("6c93e09869df3402d7bf231bf540fadd35cd56be14f97178f0954db94b7fc256")
        );
        assert_eq!(
            ct,
            unhex(
                "e5d84cd531cfb583096e7cfa9641bd3079cf3a91cda813c52deb5f512be9931980
                 a41de125a925cdad859d5b7a"
            )
        );
        assert_eq!(&open(&sk_r, &enc, &info, &aad, &ct).unwrap()[..], &pt[..]);
    }

    /// SECURITY: every input of the context is bound. Another recipient key,
    /// another `info`, another `aad`, a changed `enc` or ciphertext: the open
    /// fails, and never yields a wrong plaintext.
    #[test]
    fn another_key_info_aad_enc_or_ciphertext_does_not_open() {
        let (sk_r, pk_r) = derive_key_pair(&[1u8; 32]);
        let (sk_other, _) = derive_key_pair(&[2u8; 32]);
        let (enc, ct) = seal(&pk_r, b"info", b"aad", b"the model key", &[3u8; 32]).unwrap();
        assert_eq!(
            &open(&sk_r, &enc, b"info", b"aad", &ct).unwrap()[..],
            b"the model key"
        );
        assert_eq!(
            open(&sk_other, &enc, b"info", b"aad", &ct).unwrap_err(),
            HpkeError::Open,
            "another recipient"
        );
        assert!(open(&sk_r, &enc, b"infO", b"aad", &ct).is_err(), "info");
        assert!(open(&sk_r, &enc, b"info", b"aaD", &ct).is_err(), "aad");
        let mut enc2 = enc;
        enc2[0] ^= 1;
        assert!(open(&sk_r, &enc2, b"info", b"aad", &ct).is_err(), "enc");
        for i in [0, ct.len() - 1] {
            let mut ct2 = ct.clone();
            ct2[i] ^= 1;
            assert!(
                open(&sk_r, &enc, b"info", b"aad", &ct2).is_err(),
                "byte {i}"
            );
        }
        assert!(open(&sk_r, &enc, b"info", b"aad", &ct[..ct.len() - 1]).is_err());
        assert!(open(&sk_r, &enc, b"info", b"aad", &[]).is_err());
    }

    /// SECURITY (RFC 9180 §7.1.4): a low-order public key gives an all-zero
    /// shared secret, whoever holds what; both ends refuse it.
    #[test]
    fn a_low_order_public_key_is_refused_at_both_ends() {
        let (sk_r, _) = derive_key_pair(&[1u8; 32]);
        // u = 0, and a point of order 8 (RFC 7748 §6.1 asks for the check;
        // the value is the one libsodium's blocklist starts with).
        for low in [
            [0u8; 32],
            a32("e0eb7a7c3b41b8ae1656e3faf19fc46ada098deb9c32b1fd866205165f49b800"),
        ] {
            assert_eq!(
                seal(&low, b"", b"", b"k", &[3u8; 32]).unwrap_err(),
                HpkeError::ZeroSharedSecret
            );
            assert_eq!(
                open(&sk_r, &low, b"", b"", &[0u8; 32]).unwrap_err(),
                HpkeError::ZeroSharedSecret
            );
        }
    }

    /// Two seals of one key to one recipient never repeat: the ephemeral
    /// seed is the caller's, and it changes the encapsulation.
    #[test]
    fn two_seeds_give_two_encapsulations() {
        let (_, pk_r) = derive_key_pair(&[1u8; 32]);
        let a = seal(&pk_r, b"", b"", b"k", &[3u8; 32]).unwrap();
        let b = seal(&pk_r, b"", b"", b"k", &[4u8; 32]).unwrap();
        assert_ne!(a.0, b.0);
        assert_ne!(a.1, b.1);
    }
}
