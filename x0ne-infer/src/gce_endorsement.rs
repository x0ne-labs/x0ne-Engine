// SPDX-License-Identifier: Apache-2.0
//! Google's signed launch endorsement for GCE confidential-VM firmware:
//! the MRTD check without trusting Google's bucket or TLS.
//!
//! Object `ovmf_x64_csm/tdx/<MRTD hex>.binarypb` in the public bucket
//! `gce_tcb_integrity` is a `VMLaunchEndorsement { serialized_uefi_golden
//! = 1, signature = 2 }`; the golden `VMGoldenMeasurement` carries the
//! signer certificate (field 4) and `VMTdx { svn = 1, measurements = 2 }`
//! with `{ ram_gib = 1, early_accept = 2, mrtd = 3 }`
//! (google/gce-tcb-verifier `proto/endorsement.proto`, read 2026-10-12).
//!
//! Measured on the 2026-10-05 endorsement with openssl (2026-10-12): the
//! signature is RSASSA-PSS, SHA-256, MGF1-SHA-256, salt 32, over the golden
//! bytes; the signer `GCE-uefi-signer` is signed the same way by the pinned
//! root `GCE-cc-tcb-root` serial 1. Verified here with a 60-line protobuf
//! reader, the `rsa` crate and x509-parser — no protobuf library.

use rsa::pkcs8::DecodePublicKey;
use rsa::signature::Verifier as _;
use rsa::RsaPublicKey;
use x509_parser::certificate::X509Certificate;
use x509_parser::oid_registry::{OID_NIST_HASH_SHA256, OID_PKCS1_RSASSAPSS};
use x509_parser::prelude::FromDer;
use x509_parser::signature_algorithm::SignatureAlgorithm;

pub const BUCKET_BASE_URL: &str = "https://storage.googleapis.com/gce_tcb_integrity";
/// `https://pki.goog/cloud_integrity/GCE-cc-tcb-root_1.crt` (tests/VENDORED.md).
pub const ROOT_1_DER: &[u8] = include_bytes!("../tests/gce_tcb_root_1.der");
/// An endorsement is ~5 KiB; the cap only bounds a hostile answer.
pub const MAX_BLOB_BYTES: usize = 1024 * 1024;
const PSS_SALT_LEN: usize = 32;
/// gce-tcb-verifier `verify.go`: goldens after 2024-08-02 must carry provenance.
const PROVENANCE_REQUIRED_AFTER_UNIX: u64 = 1_722_556_800;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdxMeasurement {
    pub ram_gib: u32,
    pub early_accept: bool,
    pub mrtd: [u8; 48],
}

#[derive(Debug, Clone)]
pub struct Endorsement {
    pub timestamp_unix: u64,
    pub cl_spec: u64,
    pub commit_hex: String,
    pub uefi_sha384_hex: String,
    pub tdx_svn: u32,
    pub tdx_measurements: Vec<TdxMeasurement>,
    pub signer_subject: String,
}

impl Endorsement {
    pub fn tdx_measurement(&self, mrtd: &[u8; 48]) -> Option<&TdxMeasurement> {
        self.tdx_measurements.iter().find(|m| &m.mrtd == mrtd)
    }
}

pub fn object_name(mrtd: &[u8; 48]) -> String {
    format!("ovmf_x64_csm/tdx/{}.binarypb", hex(mrtd))
}

pub fn url(base: &str, mrtd: &[u8; 48]) -> String {
    format!("{}/{}", base.trim_end_matches('/'), object_name(mrtd))
}

/// `Ok(None)` when the bucket has no endorsement for this MRTD (404).
// Not in x0ne-sdk (its build.rs sets `x0ne_sdk`): it opens no socket, the
// caller relays these bytes.
#[cfg(not(x0ne_sdk))]
pub async fn fetch(
    http: &reqwest::Client,
    base_url: &str,
    mrtd: &[u8; 48],
) -> Result<Option<Vec<u8>>, String> {
    let url = url(base_url, mrtd);
    let resp = http
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("GET {url}: {e}"))?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !resp.status().is_success() {
        return Err(format!("GET {url}: {}", resp.status()));
    }
    let bytes = crate::http_body::read_capped(resp, MAX_BLOB_BYTES)
        .await
        .map_err(|e| format!("GET {url}: endorsement {e}"))?;
    Ok(Some(bytes))
}

// ── A protobuf reader for exactly this message ───────────────────────────────

enum Field<'a> {
    Varint(u64),
    Len(&'a [u8]),
    Fixed,
}

fn varint(b: &[u8], i: &mut usize) -> Result<u64, String> {
    let mut out: u64 = 0;
    for shift in (0..64).step_by(7) {
        let byte = *b.get(*i).ok_or("truncated varint")?;
        *i += 1;
        out |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(out);
        }
    }
    Err("varint longer than 64 bits".into())
}

fn fields(b: &[u8]) -> Result<Vec<(u32, Field<'_>)>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let key = varint(b, &mut i)?;
        let number = u32::try_from(key >> 3).map_err(|_| "field number overflow")?;
        match key & 7 {
            0 => out.push((number, Field::Varint(varint(b, &mut i)?))),
            1 => {
                i = i
                    .checked_add(8)
                    .filter(|e| *e <= b.len())
                    .ok_or("truncated fixed64")?;
                out.push((number, Field::Fixed));
            }
            2 => {
                let len = usize::try_from(varint(b, &mut i)?).map_err(|_| "length overflow")?;
                let end = i
                    .checked_add(len)
                    .filter(|e| *e <= b.len())
                    .ok_or("truncated bytes")?;
                out.push((number, Field::Len(&b[i..end])));
                i = end;
            }
            5 => {
                i = i
                    .checked_add(4)
                    .filter(|e| *e <= b.len())
                    .ok_or("truncated fixed32")?;
                out.push((number, Field::Fixed));
            }
            wt => return Err(format!("unsupported wire type {wt}")),
        }
    }
    Ok(out)
}

fn bytes_field<'a>(fs: &[(u32, Field<'a>)], n: u32) -> Option<&'a [u8]> {
    fs.iter().rev().find_map(|(k, f)| match f {
        Field::Len(b) if *k == n => Some(*b),
        _ => None,
    })
}

fn varint_field(fs: &[(u32, Field<'_>)], n: u32) -> Option<u64> {
    fs.iter().rev().find_map(|(k, f)| match f {
        Field::Varint(v) if *k == n => Some(*v),
        _ => None,
    })
}

fn parse_tdx(b: &[u8]) -> Result<(u32, Vec<TdxMeasurement>), String> {
    let fs = fields(b)?;
    let svn = u32::try_from(varint_field(&fs, 1).unwrap_or(0)).map_err(|_| "svn overflow")?;
    let mut measurements = Vec::new();
    for (k, f) in &fs {
        if let (2, Field::Len(m)) = (k, f) {
            let mf = fields(m)?;
            let mrtd = bytes_field(&mf, 3).ok_or("TDX measurement without mrtd")?;
            let mrtd: [u8; 48] = mrtd
                .try_into()
                .map_err(|_| format!("TDX mrtd is {} bytes", mrtd.len()))?;
            measurements.push(TdxMeasurement {
                ram_gib: u32::try_from(varint_field(&mf, 1).unwrap_or(0)).map_err(|_| "ram_gib")?,
                early_accept: varint_field(&mf, 2).unwrap_or(0) != 0,
                mrtd,
            });
        }
    }
    Ok((svn, measurements))
}

// ── Signatures ───────────────────────────────────────────────────────────────

fn pss_verify(
    key: &RsaPublicKey,
    message: &[u8],
    signature: &[u8],
    what: &str,
) -> Result<(), String> {
    let vk =
        rsa::pss::VerifyingKey::<rsa::sha2::Sha256>::new_with_salt_len(key.clone(), PSS_SALT_LEN);
    let sig =
        rsa::pss::Signature::try_from(signature).map_err(|e| format!("{what} signature: {e}"))?;
    vk.verify(message, &sig)
        .map_err(|_| format!("{what} signature does not verify (RSASSA-PSS SHA-256)"))
}

/// The certificate's own signature algorithm must be the one we verify.
fn require_pss_sha256(cert: &X509Certificate<'_>) -> Result<(), String> {
    if cert.signature_algorithm.algorithm != OID_PKCS1_RSASSAPSS {
        return Err(format!(
            "signer certificate algorithm {} is not RSASSA-PSS",
            cert.signature_algorithm.algorithm
        ));
    }
    let alg = SignatureAlgorithm::try_from(&cert.signature_algorithm)
        .map_err(|e| format!("signer certificate PSS parameters: {e}"))?;
    let SignatureAlgorithm::RSASSA_PSS(p) = alg else {
        return Err("signer certificate parameters are not PSS".into());
    };
    let mgf = p
        .mask_gen_algorithm()
        .map_err(|e| format!("signer certificate MGF: {e}"))?;
    if *p.hash_algorithm_oid() != OID_NIST_HASH_SHA256
        || mgf.hash != OID_NIST_HASH_SHA256
        || p.salt_length() as usize != PSS_SALT_LEN
    {
        return Err(
            "signer certificate PSS parameters are not SHA-256/MGF1-SHA-256/salt 32".into(),
        );
    }
    Ok(())
}

fn verify_signer(
    cert_der: &[u8],
    root_der: &[u8],
    now_unix: u64,
) -> Result<(RsaPublicKey, String), String> {
    let (_, root) = X509Certificate::from_der(root_der).map_err(|e| format!("pinned root: {e}"))?;
    let (_, cert) =
        X509Certificate::from_der(cert_der).map_err(|e| format!("signer certificate: {e}"))?;
    if cert.issuer().as_raw() != root.subject().as_raw() {
        return Err(format!(
            "signer {} is not issued by the pinned root ({})",
            cert.subject(),
            root.subject()
        ));
    }
    require_pss_sha256(&cert)?;
    let root_key = RsaPublicKey::from_public_key_der(root.public_key().raw)
        .map_err(|e| format!("pinned root key: {e}"))?;
    pss_verify(
        &root_key,
        cert.tbs_certificate.as_ref(),
        &cert.signature_value.data,
        "signer certificate",
    )?;
    let at =
        x509_parser::time::ASN1Time::from_timestamp(i64::try_from(now_unix).unwrap_or(i64::MAX))
            .map_err(|e| format!("clock: {e}"))?;
    if !cert.validity().is_valid_at(at) {
        return Err(format!(
            "signer certificate validity window [{}, {}] does not contain now={now_unix}",
            cert.validity().not_before.timestamp(),
            cert.validity().not_after.timestamp()
        ));
    }
    if let Ok(Some(bc)) = cert.basic_constraints() {
        if bc.value.ca {
            return Err("signer certificate is a CA, not an end entity".into());
        }
    }
    match cert.key_usage() {
        Ok(Some(ku)) if !ku.value.digital_signature() => {
            return Err("signer certificate keyUsage does not allow digitalSignature".into())
        }
        Err(e) => return Err(format!("signer certificate keyUsage: {e}")),
        _ => {}
    }
    let key = RsaPublicKey::from_public_key_der(cert.public_key().raw)
        .map_err(|e| format!("signer key: {e}"))?;
    Ok((key, cert.subject().to_string()))
}

/// Verify a `.binarypb` under `root_der` at `now_unix`. Returns what it endorses.
pub fn verify(blob: &[u8], root_der: &[u8], now_unix: u64) -> Result<Endorsement, String> {
    if blob.len() > MAX_BLOB_BYTES {
        return Err(format!(
            "endorsement is {} bytes, cap {MAX_BLOB_BYTES}",
            blob.len()
        ));
    }
    let outer = fields(blob).map_err(|e| format!("VMLaunchEndorsement: {e}"))?;
    let golden =
        bytes_field(&outer, 1).ok_or("VMLaunchEndorsement without serialized_uefi_golden")?;
    let signature = bytes_field(&outer, 2).ok_or("VMLaunchEndorsement without signature")?;
    let g = fields(golden).map_err(|e| format!("VMGoldenMeasurement: {e}"))?;
    let cert_der = bytes_field(&g, 4).ok_or("golden measurement without signer certificate")?;
    // Signer first, then the signature: a valid signature under an unknown
    // key proves nothing.
    let (signer_key, signer_subject) = verify_signer(cert_der, root_der, now_unix)?;
    pss_verify(&signer_key, golden, signature, "endorsement")?;

    let timestamp_unix = match bytes_field(&g, 1) {
        Some(ts) => varint_field(&fields(ts)?, 1).unwrap_or(0),
        None => 0,
    };
    let cl_spec = varint_field(&g, 2).unwrap_or(0);
    let commit = bytes_field(&g, 3).unwrap_or(&[]);
    if timestamp_unix > PROVENANCE_REQUIRED_AFTER_UNIX && cl_spec == 0 && commit.is_empty() {
        return Err("endorsement carries no provenance (cl_spec or commit)".into());
    }
    let tdx = bytes_field(&g, 8).ok_or("endorsement has no TDX section: not a TDX firmware")?;
    let (tdx_svn, tdx_measurements) = parse_tdx(tdx)?;
    if tdx_measurements.is_empty() {
        return Err("endorsement lists no TDX measurement".into());
    }
    Ok(Endorsement {
        timestamp_unix,
        cl_spec,
        commit_hex: hex(commit),
        uefi_sha384_hex: hex(bytes_field(&g, 5).unwrap_or(&[])),
        tdx_svn,
        tdx_measurements,
        signer_subject,
    })
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The SHA-384 the blob names as its firmware binary, hex, read without
/// checking the signature: which object a carrier fetches next
/// (`tdvf::object_name`). Nothing is believed from it; [`verify`] decides.
pub fn uefi_sha384_hex_unverified(blob: &[u8]) -> Result<String, String> {
    if blob.len() > MAX_BLOB_BYTES {
        return Err(format!(
            "endorsement is {} bytes, cap {MAX_BLOB_BYTES}",
            blob.len()
        ));
    }
    let outer = fields(blob).map_err(|e| format!("VMLaunchEndorsement: {e}"))?;
    let golden =
        bytes_field(&outer, 1).ok_or("VMLaunchEndorsement without serialized_uefi_golden")?;
    let g = fields(golden).map_err(|e| format!("VMGoldenMeasurement: {e}"))?;
    Ok(hex(bytes_field(&g, 5).unwrap_or(&[])))
}

// Not when compiled into x0ne-sdk (its build.rs sets `x0ne_sdk`): these tests
// need this crate's fixtures and dev-dependencies.
#[cfg(all(test, not(x0ne_sdk)))]
pub(crate) mod tests {
    use super::*;

    pub(crate) const VENDORED: &[u8] =
        include_bytes!("../tests/gce_endorsement_tdx_85aefc5d.binarypb");
    pub(crate) const VENDORED_MRTD_HEX: &str = "85aefc5d443857c9df7fdcb2a52819268aaa5f094a40601fad6d525d18651c005fe23f4efb9c10d02e3fa27edf147b7b";
    /// 2026-09-10, inside the signer's window (2026-08-04 to 2031-08-04).
    pub(crate) const NOW: u64 = 1_789_000_000;

    pub(crate) fn vendored_mrtd() -> [u8; 48] {
        crate::cloud_config::unhex::<48>(VENDORED_MRTD_HEX).unwrap()
    }

    /// The real object verifies under the real root and says what its name says.
    #[test]
    fn the_vendored_endorsement_verifies_under_the_pinned_root() {
        let e = verify(VENDORED, ROOT_1_DER, NOW).expect("verifies");
        assert_eq!(e.tdx_svn, 2);
        assert_eq!(e.tdx_measurements.len(), 1);
        let m = e.tdx_measurement(&vendored_mrtd()).expect("the named MRTD");
        assert_eq!((m.ram_gib, m.early_accept), (0, false));
        assert_eq!(e.cl_spec, 972_529_506);
        assert!(e.signer_subject.contains("GCE-uefi-signer"));
        assert_eq!(
            object_name(&vendored_mrtd()),
            format!("ovmf_x64_csm/tdx/{VENDORED_MRTD_HEX}.binarypb")
        );
    }

    /// SECURITY: one flipped signature byte, or one flipped golden byte, refuses.
    #[test]
    fn a_changed_byte_anywhere_refuses() {
        let mut sig = VENDORED.to_vec();
        let last = sig.len() - 1;
        sig[last] ^= 1;
        assert!(verify(&sig, ROOT_1_DER, NOW)
            .unwrap_err()
            .contains("endorsement signature"));
        let mut golden = VENDORED.to_vec();
        golden[3] ^= 1; // inside serialized_uefi_golden (timestamp)
        assert!(verify(&golden, ROOT_1_DER, NOW).is_err());
    }

    /// SECURITY: another root — even a genuine one — is not Google's.
    #[test]
    fn another_root_is_refused() {
        let intel = include_bytes!("../tests/intel_sgx_root_ca.der");
        assert!(verify(VENDORED, intel, NOW)
            .unwrap_err()
            .contains("not issued by the pinned root"));
    }

    #[test]
    fn the_signer_window_is_enforced() {
        assert!(verify(VENDORED, ROOT_1_DER, 1_700_000_000)
            .unwrap_err()
            .contains("validity window"));
    }

    pub(crate) const VENDORED_C1EE9C16: &[u8] =
        include_bytes!("../tests/gce_endorsement_tdx_c1ee9c16.binarypb");
    /// The MRTD of the GCE `a3-highgpu-1g` boots of 2026-10-17 and -18 (their
    /// TD reports and quote) [M].
    pub(crate) const C1EE9C16_MRTD_HEX: &str = "c1ee9c16e3afc506cfe042c5b846a368528f3b37618eafb27469bc114cf914e9222c91618470e7f2b28ac360968270a5";
    /// The binary Google's endorsement of c1ee9c16 names.
    pub(crate) const C1EE9C16_UEFI_SHA384: &str = "53c558fe06714203d7428484c46ee7c0b9df748a9789e173421c28d097115b30efb8476304617532488ab40c3122db88";

    /// The firmware the real boots ran: Google signed its MRTD and named its
    /// binary by SHA-384, the binary RTMR0 pinning fetches.
    #[test]
    fn the_endorsement_of_the_real_boots_names_its_binary() {
        let e = verify(VENDORED_C1EE9C16, ROOT_1_DER, NOW).expect("verifies");
        let mrtd = crate::cloud_config::unhex::<48>(C1EE9C16_MRTD_HEX).unwrap();
        let m = e.tdx_measurement(&mrtd).expect("the named MRTD");
        assert_eq!((m.ram_gib, m.early_accept, e.tdx_svn), (0, false, 2));
        assert_eq!(e.uefi_sha384_hex, C1EE9C16_UEFI_SHA384);
        assert_eq!(e.timestamp_unix, 1_779_407_565);
    }

    // ── A Google of our own, to sign what the tests need ────────────────────

    /// A root and a `GCE-uefi-signer` of our own, RSA-2048 with RSASSA-PSS
    /// SHA-256 salt 32 as Google's (the shape `verify` requires), and an
    /// endorsement encoder. TEST keys: their primes are written below, drawn
    /// from SHA-256 counters over the labels "x0ne TEST Google root" and
    /// "x0ne TEST Google uefi signer" (first probable prime of 1024 bits,
    /// top two bits set, 40 Miller-Rabin rounds).
    pub(crate) mod fake_google {
        use rsa::pkcs8::EncodePublicKey as _;
        use rsa::signature::{RandomizedSigner as _, SignatureEncoding as _};
        use rsa::{BigUint, RsaPrivateKey};
        use sha2::Digest as _;
        use std::sync::OnceLock;

        const ROOT_P: &str = "db568018965673d7ccab8374dad0036001323bace3bf0058542a4892bf60a02119a0b9d0d3ccebc6ec10428ab61e7fd60ae39e029c8ba010b1616faee2e3c5015272b4e9859f5dc1722e2df6f2aa642ea6ef11de2d6e2d9ebfb75d4e667caaf634d958a02afc31cd7e3cefa86856b7cf31490bdc83f6d209e2a3a202afc9a22d";
        const ROOT_Q: &str = "f05f811b001bad6b507990c98227001b30075fde10ebe93bd956d2185b97a519b5029815576a2fb9ae11e55341275e11a0c26a1e134b9d8e338ab6e75f550871eeed73ca9f56c206ac482356a47159d8cad2acca483369d3e3ebceb3c8b941c348af336499a614cd9ff6982de24a3d5cffa777536cd8bdfa7efdcbbd3f9b9b01";
        const SIGNER_P: &str = "c4e00af76247eea39377ef77b26bd2a27d4f2197407901e983dd9e10363bfae9e2ae376838a7ce38efac7534cc1467a618921385dcd962f62b835d3a2e0550a927afa4551959a25992dc56f44776b808ea9bd6c3e497d102cceda1bfda1b941dc32d47edcab1eb1ef1e6599430789c8a78993f7242910873a7ccc6df374be429";
        const SIGNER_Q: &str = "de1a05b82d6dffe3269195fc61872cf1b3df061d6c91f57fc3fce572f3cecc8c05e9a9c05bef4559b8db14b27fe7543eca3cabf42f8bb355a3d3d2b17002f2318880fc46051dbbdea772cf590628619798d63c4f6db5c5b69b71def51a809d4720fbca44ca2156125caae2632cbe7aff6017b72c0e6589aed5f0cb52aaed48d7";

        pub(crate) struct FakeGoogle {
            pub(crate) root_der: Vec<u8>,
            signer_der: Vec<u8>,
            signer_key: RsaPrivateKey,
        }

        /// A deterministic source for the PSS salts: SHA-256 over a counter.
        struct Counter(u64);

        impl rsa::rand_core::RngCore for Counter {
            fn next_u32(&mut self) -> u32 {
                let mut b = [0; 4];
                self.fill_bytes(&mut b);
                u32::from_le_bytes(b)
            }
            fn next_u64(&mut self) -> u64 {
                let mut b = [0; 8];
                self.fill_bytes(&mut b);
                u64::from_le_bytes(b)
            }
            fn fill_bytes(&mut self, dest: &mut [u8]) {
                for chunk in dest.chunks_mut(32) {
                    self.0 += 1;
                    let d = sha2::Sha256::digest(self.0.to_le_bytes());
                    chunk.copy_from_slice(&d[..chunk.len()]);
                }
            }
            fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rsa::rand_core::Error> {
                self.fill_bytes(dest);
                Ok(())
            }
        }

        impl rsa::rand_core::CryptoRng for Counter {}

        fn key(p: &str, q: &str) -> RsaPrivateKey {
            let n = |h: &str| BigUint::parse_bytes(h.as_bytes(), 16).unwrap();
            RsaPrivateKey::from_p_q(n(p), n(q), BigUint::from(65_537u32)).unwrap()
        }

        fn der(tag: u8, body: &[u8]) -> Vec<u8> {
            let n = body.len();
            let mut out = vec![tag];
            match n {
                0..=0x7f => out.push(n as u8),
                0x80..=0xff => out.extend([0x81, n as u8]),
                _ => out.extend([0x82, (n >> 8) as u8, n as u8]),
            }
            out.extend_from_slice(body);
            out
        }

        fn seq(parts: &[&[u8]]) -> Vec<u8> {
            der(0x30, &parts.concat())
        }

        const OID_PSS: &[u8] = &[
            0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a,
        ];
        const OID_MGF1: &[u8] = &[
            0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x08,
        ];
        const OID_SHA256: &[u8] = &[
            0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01,
        ];
        const OID_CN: &[u8] = &[0x06, 0x03, 0x55, 0x04, 0x03];
        const OID_BASIC_CONSTRAINTS: &[u8] = &[0x06, 0x03, 0x55, 0x1d, 0x13];
        const OID_KEY_USAGE: &[u8] = &[0x06, 0x03, 0x55, 0x1d, 0x0f];

        /// RSASSA-PSS, SHA-256, MGF1-SHA-256, salt 32 (RFC 4055).
        fn pss_sha256() -> Vec<u8> {
            let sha256 = seq(&[OID_SHA256, &[0x05, 0x00]]);
            let params = seq(&[
                &der(0xa0, &sha256),
                &der(0xa1, &seq(&[OID_MGF1, &sha256])),
                &der(0xa2, &der(0x02, &[32])),
            ]);
            seq(&[OID_PSS, &params])
        }

        fn name(cn: &str) -> Vec<u8> {
            seq(&[&der(0x31, &seq(&[OID_CN, &der(0x0c, cn.as_bytes())]))])
        }

        fn pss_sign(key: &RsaPrivateKey, message: &[u8], seed: u64) -> Vec<u8> {
            rsa::pss::SigningKey::<rsa::sha2::Sha256>::new_with_salt_len(key.clone(), 32)
                .sign_with_rng(&mut Counter(seed), message)
                .to_vec()
        }

        fn certificate(
            serial: u8,
            issuer: &str,
            subject: &str,
            subject_key: &RsaPrivateKey,
            issuer_key: &RsaPrivateKey,
            ca: bool,
        ) -> Vec<u8> {
            let spki = subject_key
                .to_public_key()
                .to_public_key_der()
                .unwrap()
                .as_bytes()
                .to_vec();
            let (constraints, usage) = if ca {
                (seq(&[&der(0x01, &[0xff])]), der(0x03, &[0x02, 0x04]))
            } else {
                (seq(&[]), der(0x03, &[0x07, 0x80]))
            };
            let critical = der(0x01, &[0xff]);
            let extensions = seq(&[
                &seq(&[OID_BASIC_CONSTRAINTS, &critical, &der(0x04, &constraints)]),
                &seq(&[OID_KEY_USAGE, &critical, &der(0x04, &usage)]),
            ]);
            let tbs = seq(&[
                &der(0xa0, &der(0x02, &[2])),
                &der(0x02, &[serial]),
                &pss_sha256(),
                &name(issuer),
                &seq(&[&der(0x17, b"250101000000Z"), &der(0x17, b"491231235959Z")]),
                &name(subject),
                &spki,
                &der(0xa3, &extensions),
            ]);
            let sig = pss_sign(issuer_key, &tbs, u64::from(serial) << 32);
            seq(&[
                &tbs,
                &pss_sha256(),
                &der(0x03, &[&[0u8][..], &sig].concat()),
            ])
        }

        pub(crate) fn fake_google() -> &'static FakeGoogle {
            static G: OnceLock<FakeGoogle> = OnceLock::new();
            G.get_or_init(|| {
                let root = key(ROOT_P, ROOT_Q);
                let signer_key = key(SIGNER_P, SIGNER_Q);
                const ROOT: &str = "x0ne TEST GCE-cc-tcb-root";
                FakeGoogle {
                    root_der: certificate(1, ROOT, ROOT, &root, &root, true),
                    signer_der: certificate(
                        2,
                        ROOT,
                        "x0ne TEST GCE-uefi-signer",
                        &signer_key,
                        &root,
                        false,
                    ),
                    signer_key,
                }
            })
        }

        fn varint(mut v: u64, out: &mut Vec<u8>) {
            while v >= 0x80 {
                out.push((v as u8) | 0x80);
                v >>= 7;
            }
            out.push(v as u8);
        }

        fn field_varint(n: u32, v: u64, out: &mut Vec<u8>) {
            varint(u64::from(n) << 3, out);
            varint(v, out);
        }

        fn field_bytes(n: u32, b: &[u8], out: &mut Vec<u8>) {
            varint((u64::from(n) << 3) | 2, out);
            varint(b.len() as u64, out);
            out.extend_from_slice(b);
        }

        impl FakeGoogle {
            /// A `VMLaunchEndorsement` for the binary of SHA-384 `uefi_sha384`
            /// listing `mrtds` (ram_gib 0), signed now by our signer.
            pub(crate) fn endorse(&self, uefi_sha384: &[u8], mrtds: &[[u8; 48]]) -> Vec<u8> {
                let mut golden = Vec::new();
                let mut ts = Vec::new();
                field_varint(1, 1_789_000_000, &mut ts);
                field_bytes(1, &ts, &mut golden);
                field_varint(2, 1, &mut golden);
                field_bytes(4, &self.signer_der, &mut golden);
                field_bytes(5, uefi_sha384, &mut golden);
                let mut tdx = Vec::new();
                field_varint(1, 2, &mut tdx);
                for m in mrtds {
                    let mut one = Vec::new();
                    field_varint(1, 0, &mut one);
                    field_bytes(3, m, &mut one);
                    field_bytes(2, &one, &mut tdx);
                }
                field_bytes(8, &tdx, &mut golden);
                let mut out = Vec::new();
                field_bytes(1, &golden, &mut out);
                field_bytes(2, &pss_sign(&self.signer_key, &golden, 7), &mut out);
                out
            }

            /// The endorsement of `firmware` (its SHA-384) for `mrtds`.
            pub(crate) fn endorse_binary(&self, firmware: &[u8], mrtds: &[[u8; 48]]) -> Vec<u8> {
                self.endorse(&sha2::Sha384::digest(firmware), mrtds)
            }
        }
    }

    /// Our Google signs the way `verify` requires, and only under its root.
    #[test]
    fn the_fake_google_endorses_under_its_own_root_only() {
        let g = fake_google::fake_google();
        let blob = g.endorse(&[0xab; 48], &[[0x5; 48]]);
        let e = verify(&blob, &g.root_der, NOW).expect("verifies under the test root");
        assert_eq!(e.uefi_sha384_hex, "ab".repeat(48));
        assert_eq!(e.tdx_measurements[0].mrtd, [0x5; 48]);
        assert!(e.signer_subject.contains("x0ne TEST GCE-uefi-signer"));
        assert!(verify(&blob, ROOT_1_DER, NOW)
            .unwrap_err()
            .contains("not issued by the pinned root"));
        let mut flipped = blob.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 1;
        assert!(verify(&flipped, &g.root_der, NOW)
            .unwrap_err()
            .contains("endorsement signature"));
    }
}
