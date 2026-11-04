// SPDX-License-Identifier: Apache-2.0
//! The x0ne archive key, pinned in this binary: its signing subkey signs
//! every image reference the relay accepts. The maintainer generates it
//! offline (x0ne-OS `docs/archive-key.md`, not public yet); a build carries the public part
//! and the primary's fingerprint, and nothing at run time adds a key.
//!
//! A build made before the signing ceremony pins no key (`RELEASE` is
//! None): its relay refuses every reference with 99 and says so. A debug
//! build with `test-archive-key` pins a TEST key instead, and the verdict
//! names it TEST. Without debug assertions the feature pins nothing: the
//! release test build and the release `cloud_probe` keep `RELEASE`.

use crate::openpgp::{Cert, Signer};

/// A public key and the fingerprint written down at its generation.
pub struct Pin {
    /// The `primary` line of the ceremony's `public/FINGERPRINTS`.
    pub primary_fingerprint: &'static str,
    /// The ceremony's `public/x0ne-archive-keyring.gpg`, binary.
    pub keyring: &'static [u8],
}

/// The release pin: None in a tree made before the signing ceremony.
pub const RELEASE: Option<Pin> = None;

#[cfg_attr(all(feature = "test-archive-key", debug_assertions), allow(dead_code))]
const NO_KEY: &str = "this build pins no x0ne archive key: it was built before the signing \
     ceremony, so it verifies no image reference; a build made after the ceremony pins the \
     key, and no flag replaces it";

/// The key every reference must be signed by, or why there is none.
pub struct Anchor(Result<Cert, String>);

impl Anchor {
    pub fn production() -> Self {
        // `cargo test --release` and release examples carry the feature too
        // (self dev-dependency): only a debug build may pin the TEST key.
        #[cfg(all(feature = "test-archive-key", debug_assertions))]
        {
            Anchor(Cert::parse(&test_key::keyring()))
        }
        #[cfg(not(all(feature = "test-archive-key", debug_assertions)))]
        {
            Anchor(match &RELEASE {
                Some(pin) => pinned(pin),
                None => Err(NO_KEY.into()),
            })
        }
    }

    /// The pinned key, or why this build has none (no pin, or a broken one):
    /// then every reference is refused, whatever its signature.
    pub fn key(&self) -> Result<&Cert, String> {
        self.0.as_ref().map_err(Clone::clone)
    }

    /// `data` signed by a live signing subkey of the pinned key, at `now`.
    pub fn verify(&self, data: &[u8], signature: &[u8], now_unix: u64) -> Result<Signer, String> {
        let cert = self.key()?;
        cert.verify_detached(data, signature, now_unix)
            .map_err(|e| format!("{e} (pinned archive key {})", cert.primary_fingerprint()))
    }
}

/// The keyring parses, its primary is the fingerprint written down, and it
/// is not a TEST key (as x0ne-OS `archive-key.sh check` and `dist-check.sh`).
pub fn pinned(pin: &Pin) -> Result<Cert, String> {
    let cert = Cert::parse(pin.keyring).map_err(|e| format!("pinned archive key: {e}"))?;
    let want = pin.primary_fingerprint.to_ascii_uppercase();
    if cert.primary_fingerprint() != want {
        return Err(format!(
            "pinned archive key: the keyring's primary is {}, the pin says {want}",
            cert.primary_fingerprint()
        ));
    }
    if let Some(uid) = cert.user_ids().iter().find(|u| u.contains("TEST")) {
        return Err(format!("pinned archive key: TEST user ID {uid:?}"));
    }
    Ok(cert)
}

#[cfg(test)]
impl Anchor {
    pub(crate) fn of(cert: Result<Cert, String>) -> Self {
        Anchor(cert)
    }
    pub(crate) fn none() -> Self {
        Anchor(Err(NO_KEY.into()))
    }
}

/// The TEST key, pinned on a bench: x0ne-sdk's `test-anchor` builds only,
/// debug or release (its wasm smoke build is a release one).
#[cfg(x0ne_sdk_test_anchor)]
impl Anchor {
    pub(crate) fn test() -> Self {
        Anchor(Cert::parse(&test_key::keyring()))
    }
}

/// The TEST key: derived from public strings, so anyone can sign with it.
/// Same shape as the archive key. Compiled in every `test-archive-key`
/// build, pinned only by a debug one (`Anchor::production`).
#[cfg(any(test, feature = "test-archive-key"))]
pub mod test_key {
    use crate::openpgp::build::{detached, primary_block, subkey_block, KeyPair};
    use sha2::{Digest, Sha256};

    pub const USER_ID: &str = "x0ne Engine TEST archive key, not for release <test@invalid>";
    /// 2026-10-16T00:00:00Z. No expiry: the tests run on the real clock.
    pub const CREATED: u32 = 1_792_108_800;

    fn pair(label: &str) -> KeyPair {
        let seed: [u8; 32] = Sha256::digest(format!(
            "x0ne-engine TEST archive key, not for release: {label}"
        ))
        .into();
        KeyPair::from_seed(seed, CREATED)
    }

    pub fn keyring() -> Vec<u8> {
        let (p, s) = (pair("primary"), pair("signing subkey"));
        let mut v = primary_block(&p, USER_ID.as_bytes(), 0);
        v.extend(subkey_block(&p, &p, &s, 0x02, 0, true));
        v
    }

    /// A detached signature over `data`, as `<reference>.sig`.
    pub fn sign(data: &[u8], created_unix: u32) -> Vec<u8> {
        detached(&pair("signing subkey"), data, created_unix)
    }
}

#[cfg(all(test, not(x0ne_sdk)))]
mod tests {
    use super::*;

    const GNUPG_TEST: &[u8] = include_bytes!("../tests/archive_key/test-archive-keyring.gpg");
    const GNUPG_TEST_FPR: &str = "2F957D30FAE5CD616C23FD44AAC12AC5E4D7FDCB";

    #[test]
    fn no_release_key_refuses_everything() {
        assert!(
            RELEASE.is_none(),
            "a pin landed: replace this test by one on its fingerprint"
        );
        let err = Anchor::none().verify(b"x", b"", 0).unwrap_err();
        assert!(err.contains("pins no x0ne archive key"), "{err}");
        assert!(err.contains("built before the signing ceremony"), "{err}");
    }

    /// SECURITY: a release pin is refused when the keyring is not the key
    /// written down, or when it is a TEST key.
    #[test]
    fn a_release_pin_must_match_its_fingerprint_and_not_be_a_test_key() {
        let err = pinned(&Pin {
            primary_fingerprint: GNUPG_TEST_FPR,
            keyring: GNUPG_TEST,
        })
        .unwrap_err();
        assert!(err.contains("TEST user ID"), "{err}");
        let err = pinned(&Pin {
            primary_fingerprint: "0000000000000000000000000000000000000000",
            keyring: GNUPG_TEST,
        })
        .unwrap_err();
        assert!(err.contains("the pin says"), "{err}");
        let err = pinned(&Pin {
            primary_fingerprint: GNUPG_TEST_FPR,
            keyring: b"not a key",
        })
        .unwrap_err();
        assert!(err.starts_with("pinned archive key:"), "{err}");
    }

    #[test]
    fn the_test_key_signs_and_verifies_and_names_itself() {
        let anchor = Anchor::of(Cert::parse(&test_key::keyring()));
        let sig = test_key::sign(b"reference", test_key::CREATED + 60);
        let signer = anchor
            .verify(b"reference", &sig, u64::from(test_key::CREATED) + 3600)
            .unwrap();
        assert_eq!(signer.signed_at_unix, u64::from(test_key::CREATED) + 60);
        let err = anchor
            .verify(b"other", &sig, u64::from(test_key::CREATED) + 3600)
            .unwrap_err();
        assert!(err.contains("pinned archive key"), "{err}");
        let cert = Cert::parse(&test_key::keyring()).unwrap();
        assert_eq!(cert.user_ids(), [test_key::USER_ID]);
    }
}
