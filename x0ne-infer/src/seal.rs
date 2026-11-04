//! The weight envelope — AES-256-GCM under the released key.
//!
//! This module is what makes the key release LOAD-BEARING. Until it existed,
//! `load()` took plaintext safetensors and the released key was required by
//! signature but never used — stated in the code, honest, and exactly the
//! gap an open-source confidential engine cannot ship with: "keys released
//! after attestation" must mean the weights are unreadable without them.
//!
//! Format (all bytes, no framing dependencies):
//!
//! ```text
//! [0..8)   magic  b"X0NESEAL"
//! [8]      version 0x01
//! [9..21)  nonce (12 bytes, caller-supplied — the sealer's RNG, not ours)
//! [21..]   AES-256-GCM ciphertext ‖ tag(16)
//! ```
//!
//! The magic+version prefix is authenticated as AAD: a sealed file whose
//! header was rewritten fails the tag, it does not decrypt under another
//! interpretation. Decrypted bytes come back `Zeroizing` — the plaintext
//! model exists in TDX-protected RAM and nowhere else.
//!
//! What this module does NOT decide: whether plaintext weights are allowed.
//! Public open-weights models legitimately load in the clear; the caller
//! (`core_backend`'s `CoreBackend::load`) branches on [`is_sealed`] and says so.

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use zeroize::Zeroizing;

pub const MAGIC: &[u8; 8] = b"X0NESEAL";
pub const VERSION: u8 = 1;
const NONCE_LEN: usize = 12;
const HEADER_LEN: usize = 8 + 1 + NONCE_LEN;
/// GCM tag length — a sealed payload can never be shorter than header + tag.
const TAG_LEN: usize = 16;
/// The most one GCM message holds, 2^36 - 32 bytes (NIST SP 800-38D; aes-gcm
/// 0.11's `P_MAX`): under `fsread::WEIGHTS_CAP`, 128 GiB, which reads one file.
pub const MAX_PLAINTEXT: u64 = (1 << 36) - 32;

/// Does this byte stream claim to be a sealed envelope? (Claim only — the
/// tag decides whether it IS one.)
pub fn is_sealed(bytes: &[u8]) -> bool {
    bytes.len() >= HEADER_LEN && &bytes[..8] == MAGIC
}

/// Seal `plaintext` under `key`. The nonce is the CALLER's: this function is
/// deterministic and testable, and the one production caller (the `seal`
/// subcommand) draws it from the OS. Reusing a nonce under the same key
/// breaks GCM — the caller's contract, stated here because it is the only
/// way to misuse this function. More than [`MAX_PLAINTEXT`] bytes is refused
/// before any encryption: AES-GCM cannot seal them in one message.
pub fn seal(plaintext: &[u8], key: &[u8; 32], nonce: &[u8; NONCE_LEN]) -> anyhow::Result<Vec<u8>> {
    if plaintext.len() as u64 > MAX_PLAINTEXT {
        anyhow::bail!(
            "{} bytes to seal: AES-GCM seals at most {MAX_PLAINTEXT} in one envelope",
            plaintext.len()
        );
    }
    let cipher = Aes256Gcm::new(&Key::<Aes256Gcm>::from(*key));
    let mut out = Vec::with_capacity(HEADER_LEN + plaintext.len() + TAG_LEN);
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.extend_from_slice(nonce);
    let aad = &out[..9]; // magic ‖ version — authenticated, not secret
    let ct = cipher
        .encrypt(
            &Nonce::from(*nonce),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|_| anyhow::anyhow!("AES-GCM refused to encrypt"))?;
    out.extend_from_slice(&ct);
    Ok(out)
}

/// Open a sealed envelope. Every refusal names what failed; a tampered
/// byte anywhere (header included, via AAD) is "authentication failed",
/// never a silent wrong-plaintext.
pub fn unseal(sealed: &[u8], key: &[u8; 32]) -> anyhow::Result<Zeroizing<Vec<u8>>> {
    if sealed.len() < HEADER_LEN + TAG_LEN {
        anyhow::bail!(
            "sealed payload is {} bytes, shorter than header+tag ({})",
            sealed.len(),
            HEADER_LEN + TAG_LEN
        );
    }
    if &sealed[..8] != MAGIC {
        anyhow::bail!("not a sealed payload (magic mismatch)");
    }
    let version = sealed[8];
    if version != VERSION {
        // SECURITY: an unknown version is a refusal, never a best-effort
        // parse with today's layout — the tdx_quote v5 lesson.
        anyhow::bail!(
            "sealed payload version {version} is not supported (this build reads {VERSION})"
        );
    }
    // The header length was checked above, so this slice is NONCE_LEN bytes;
    // the conversion is fallible only in the type system.
    let nonce: [u8; NONCE_LEN] = sealed[9..HEADER_LEN]
        .try_into()
        .map_err(|_| anyhow::anyhow!("sealed payload: nonce is not {NONCE_LEN} bytes"))?;
    let nonce = Nonce::from(nonce);
    let aad = &sealed[..9];
    let cipher = Aes256Gcm::new(&Key::<Aes256Gcm>::from(*key));
    let pt = cipher
        .decrypt(
            &nonce,
            Payload {
                msg: &sealed[HEADER_LEN..],
                aad,
            },
        )
        .map_err(|_| {
            anyhow::anyhow!("seal authentication failed — wrong key, or the payload was modified")
        })?;
    Ok(Zeroizing::new(pt))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> [u8; 32] {
        [7u8; 32]
    }
    fn nonce() -> [u8; 12] {
        [9u8; 12]
    }

    /// A known answer, computed OUTSIDE this crate (Python `cryptography`,
    /// AES-256-GCM, aad = magic ‖ version) on 2026-09-18, the day `aes-gcm`
    /// went 0.10 → 0.11. A roundtrip test cannot see a library that
    /// changes what it writes: both halves change together. This is what says
    /// weights sealed by an older build still open, and that the envelope is
    /// the standard construction and not an artefact of one crate version.
    #[test]
    fn the_envelope_is_standard_aes_gcm_byte_for_byte() {
        let key: [u8; 32] = std::array::from_fn(|i| i as u8);
        let nonce: [u8; 12] = std::array::from_fn(|i| 100 + i as u8);
        let want = "58304e455345414c016465666768696a6b6c6d6e6f2574ba0315c921fb5705379c\
                    a9494acd6cf3362ae41ed34289e09d611570262b01770eb19e5ad9fa01ab74";
        let got: String = seal(b"model weights, 0.10 or 0.11", &key, &nonce)
            .unwrap()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(got, want);
        let sealed: Vec<u8> = (0..want.len() / 2)
            .map(|i| u8::from_str_radix(&want[i * 2..i * 2 + 2], 16).unwrap())
            .collect();
        assert_eq!(
            unseal(&sealed, &key).unwrap().as_slice(),
            b"model weights, 0.10 or 0.11"
        );
    }

    /// The envelope gives back exactly what was sealed, and only under the
    /// key that sealed it.
    #[test]
    fn a_sealed_payload_roundtrips_under_its_key_and_only_its_key() {
        let sealed = seal(b"weights", &key(), &nonce()).unwrap();
        assert!(is_sealed(&sealed));
        assert_eq!(unseal(&sealed, &key()).unwrap().as_slice(), b"weights");
        let mut other = key();
        other[0] ^= 1;
        let err = unseal(&sealed, &other).unwrap_err().to_string();
        assert!(err.contains("authentication failed"), "{err}");
    }

    /// SECURITY: one flipped bit anywhere — ciphertext, tag, or the
    /// authenticated header — is a refusal, never a wrong plaintext. A
    /// failure would mean a modified model loading as if intact.
    #[test]
    fn any_flipped_bit_is_a_refusal_never_a_wrong_plaintext() {
        let sealed = seal(b"weights", &key(), &nonce()).unwrap();
        for i in [8usize, 10, HEADER_LEN + 1, sealed.len() - 1] {
            let mut bad = sealed.clone();
            bad[i] ^= 1;
            assert!(
                unseal(&bad, &key()).is_err(),
                "byte {i} flipped and the envelope still opened"
            );
        }
    }

    /// Truncation and unknown versions are named refusals (the v5 lesson:
    /// refuse what you do not implement, never parse it with today's layout).
    #[test]
    fn truncated_or_future_versions_are_refused_by_name() {
        let sealed = seal(b"weights", &key(), &nonce()).unwrap();
        let err = unseal(&sealed[..HEADER_LEN + 3], &key())
            .unwrap_err()
            .to_string();
        assert!(err.contains("shorter than"), "{err}");
        let mut future = sealed.clone();
        future[8] = 2;
        let err = unseal(&future, &key()).unwrap_err().to_string();
        assert!(err.contains("version 2"), "{err}");
        assert!(unseal(b"X0NESEALX", &key()).is_err());
        assert!(!is_sealed(b"plaintext safetensors"));
    }

    /// More than AES-GCM's 2^36 - 32 bytes is refused before encryption,
    /// where aes-gcm returned an error the `.expect` turned into a panic
    /// (2026-11-04). The bytes are an anonymous mapping never touched.
    #[cfg(target_os = "linux")]
    #[test]
    fn more_than_aes_gcm_holds_is_refused_before_encryption() {
        let len = MAX_PLAINTEXT as usize + 1;
        // SAFETY: a private anonymous mapping of `len` bytes, read-only and
        // never read: `seal` checks the length first. Unmapped below.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED, "{}", std::io::Error::last_os_error());
        // SAFETY: `ptr` maps `len` readable bytes until the munmap below.
        let big = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len) };
        let err = seal(big, &key(), &nonce()).unwrap_err().to_string();
        // SAFETY: the mapping made above, no longer borrowed.
        unsafe { libc::munmap(ptr, len) };
        assert!(err.contains("AES-GCM seals at most 68719476704"), "{err}");
    }
}
