// SPDX-License-Identifier: Apache-2.0
//! Cryptographic verification of an Intel TDX quote — the DCAP chain.
//!
//! What "attested" must mean before this server may serve (spec, decision 1):
//! the quote's signature chain verifies down to a **pinned Intel root**, not
//! merely that the bytes have the right shape. The chain, exactly as Intel
//! defines it for TD Quote v4 (ECDSA-P256 everywhere):
//!
//! ```text
//! Intel SGX Root CA (pinned here)
//!   └─ signs → PCK Platform/Processor CA
//!        └─ signs → PCK certificate (leaf, one per platform)
//!             └─ signs → QE report (384 bytes)
//!                  binds → attestation key (SHA-256 in qe_report.report_data)
//!                       └─ signs → the quote (header ‖ TD report)
//! ```
//!
//! Every arrow is checked here, in order, fail-closed: one broken link and the
//! verdict is `Rejected` with the link named. What this module does NOT do
//! yet, stated so the caller cannot over-trust it (`Verified::limitations`):
//! TCB status / revocation against the live Intel PCS API, and QE identity —
//! the chain proves *genuine Intel hardware signed this*, not *its firmware
//! is current*. Those checks live in `crate::tcb`, against live Intel PCS.
//!
//! The pinned root is injected (`TrustAnchor`), never fetched: production pins
//! the Intel SGX Root CA public key; tests pin their own generated root and
//! prove the whole path both ways (verify, and refuse when tampered).

use p256::ecdsa::signature::Verifier;
use p256::ecdsa::{Signature, VerifyingKey};
use sha2::{Digest, Sha256};
use x509_parser::prelude::{FromDer, Pem, X509Certificate};

/// Offsets from the quote layout (kept in lockstep with
/// `x0ne_confidential::tdx_quote` — same numbers, one format).
const HEADER: usize = 48;
const BODY: usize = 584;
const SIGNED_LEN: usize = HEADER + BODY;

/// SHA-256 of the DER-encoded SubjectPublicKeyInfo of the Intel SGX Root CA
/// (the root that anchors every PCK chain, TDX included).
///
/// MEASURED, not recalled: fetched from
/// `https://certificates.trustedservices.intel.com/Intel_SGX_Provisioning_Certification_RootCA.cer`
/// on 2026-08-20 (659-byte DER) and digested with
/// `openssl x509 -pubkey | openssl pkey -pubin -outform der | sha256sum`.
/// Pinned as a digest so the certificate bytes need not ship in the binary.
///
/// SECURITY: verify-before-trust for this constant happens out of band, at
/// review time — re-run the two commands above against Intel's URL. If Intel
/// rotates the root (unchanged since 2018), this is the single place to
/// update.
pub const INTEL_SGX_ROOT_CA_SPKI_SHA256_HEX: &str =
    "a0af031289f5d5d4132f9186068a7fc13628633ba235777472e29b6b6c67a49e";

/// The root the chain must end on.
pub enum TrustAnchor {
    /// Pin by SPKI digest — production: the Intel constant above.
    SpkiSha256Hex(String),
    /// Pin by full root certificate DER — tests, and operators who prefer
    /// shipping the certificate.
    RootDer(Vec<u8>),
}

impl TrustAnchor {
    pub fn intel_production() -> Self {
        TrustAnchor::SpkiSha256Hex(INTEL_SGX_ROOT_CA_SPKI_SHA256_HEX.into())
    }
}

/// Why the chain did not verify. One variant per arrow of the diagram, so a
/// rejection names the exact link that broke.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PcsRejected {
    #[error("quote too short for a signature section: {0} bytes")]
    TooShort(usize),
    #[error("signature section malformed: {0}")]
    Malformed(String),
    #[error("certification data type {0} unsupported (expected 6: QE report certification data)")]
    UnsupportedCertData(u16),
    #[error("PCK chain: {0}")]
    Chain(String),
    #[error("root does not match the pinned trust anchor")]
    RootNotPinned,
    #[error("QE report signature does not verify under the PCK key")]
    QeSignature,
    #[error("attestation key is not bound by the QE report (report_data mismatch)")]
    AttestationKeyBinding,
    #[error("the QE report's report_data[32..64] is not zero")]
    QeReportDataTail,
    #[error("quote signature does not verify under the attestation key")]
    QuoteSignature,
    #[error("TD report: {0}")]
    TdReport(String),
    #[error("TDATTRIBUTES.DEBUG is set: the host can read and write this TD's memory")]
    DebugTd,
    #[error("TDATTRIBUTES carries {0}: not a production TD")]
    TdAttributes(String),
}

/// What a verified chain established, with its stated limitations.
#[derive(Debug, Clone)]
pub struct VerifiedTdx {
    /// Lowercase-hex SHA-256 of the leaf (PCK) certificate DER — names the
    /// platform in receipts without shipping the certificate.
    pub pck_sha256_hex: String,
    /// The TD report's `report_data`, as the hardware wrote it — 64 bytes at
    /// quote[568..632], covered by the quote signature this function just
    /// verified.
    ///
    /// Returned rather than judged: this module verifies a CHAIN, it does not
    /// know what nonce the caller asked for. The comparison that turns this
    /// into an anti-replay is `verify::cryptographic_with_anchors`, which
    /// holds the session identity. A verified chain whose `report_data` is
    /// not the caller's challenge proves a genuine platform SOMEWHERE, not
    /// this process — see the security note there.
    pub report_data: [u8; 64],
    /// True limitations of this verdict, for the caller to surface.
    pub limitations: &'static [&'static str],
}

/// What this module DOES check, said once so a reader need not infer it from
/// the code: every certificate of the chain is verified against its issuer's
/// key AND against its own validity window (2026-08-21) — the window
/// being how an issuer retires a certificate whose key it no longer trusts.
pub const LIMITATIONS: &[&str] = &[
    "TCB status and revocation not checked against live Intel PCS",
    "QE identity not checked against Intel's published QE identity",
];

/// The parsed signature section of a TD Quote v4.
pub(crate) struct SigSection<'a> {
    pub(crate) quote_signature: &'a [u8], // 64: r‖s over quote[0..632]
    pub(crate) attestation_key: &'a [u8], // 64: P-256 x‖y (no 0x04 prefix)
    pub(crate) qe_report: &'a [u8],       // 384
    pub(crate) qe_report_signature: &'a [u8], // 64: r‖s over qe_report, by PCK key
    pub(crate) qe_auth_data: &'a [u8],
    pub(crate) pck_chain_pem: &'a [u8],
}

pub(crate) fn parse_sig_section(quote: &[u8]) -> Result<SigSection<'_>, PcsRejected> {
    if quote.len() < SIGNED_LEN + 4 {
        return Err(PcsRejected::TooShort(quote.len()));
    }
    let mut off = SIGNED_LEN;
    // Checked: a u32 length plus an offset wraps a 32-bit usize (wasm32).
    let take = |off: &mut usize, n: usize, what: &str| -> Result<&[u8], PcsRejected> {
        let s = off
            .checked_add(n)
            .and_then(|end| quote.get(*off..end))
            .ok_or_else(|| PcsRejected::Malformed(format!("{what}: need {n} bytes at {off}")))?;
        *off += n;
        Ok(s)
    };
    let sig_len_b = take(&mut off, 4, "sig_data_len")?;
    let sig_len = u32::from_le_bytes(sig_len_b.try_into().unwrap()) as usize;
    if quote.len() - SIGNED_LEN - 4 < sig_len {
        return Err(PcsRejected::Malformed(format!(
            "sig_data_len says {sig_len}, quote has {}",
            quote.len() - SIGNED_LEN - 4
        )));
    }
    let quote_signature = take(&mut off, 64, "quote_signature")?;
    let attestation_key = take(&mut off, 64, "attestation_key")?;
    let cert_type_b = take(&mut off, 2, "cert_data_type")?;
    let cert_type = u16::from_le_bytes(cert_type_b.try_into().unwrap());
    if cert_type != 6 {
        return Err(PcsRejected::UnsupportedCertData(cert_type));
    }
    let _cert_size = take(&mut off, 4, "cert_data_size")?;
    let qe_report = take(&mut off, 384, "qe_report")?;
    let qe_report_signature = take(&mut off, 64, "qe_report_signature")?;
    let auth_len_b = take(&mut off, 2, "qe_auth_data_size")?;
    let auth_len = u16::from_le_bytes(auth_len_b.try_into().unwrap()) as usize;
    let qe_auth_data = take(&mut off, auth_len, "qe_auth_data")?;
    let inner_type_b = take(&mut off, 2, "inner cert_data_type")?;
    let inner_type = u16::from_le_bytes(inner_type_b.try_into().unwrap());
    if inner_type != 5 {
        return Err(PcsRejected::Malformed(format!(
            "inner certification data type {inner_type}, expected 5 (PCK chain PEM)"
        )));
    }
    let inner_len_b = take(&mut off, 4, "inner cert_data_size")?;
    let inner_len = u32::from_le_bytes(inner_len_b.try_into().unwrap()) as usize;
    let pck_chain_pem = take(&mut off, inner_len, "PCK chain PEM")?;
    Ok(SigSection {
        quote_signature,
        attestation_key,
        qe_report,
        qe_report_signature,
        qe_auth_data,
        pck_chain_pem,
    })
}

fn p256_key_from_xy(xy: &[u8]) -> Result<VerifyingKey, PcsRejected> {
    let mut sec1 = Vec::with_capacity(65);
    sec1.push(0x04);
    sec1.extend_from_slice(xy);
    VerifyingKey::from_sec1_bytes(&sec1)
        .map_err(|e| PcsRejected::Malformed(format!("attestation key not on P-256: {e}")))
}

pub(crate) fn p256_key_from_cert(cert: &X509Certificate<'_>) -> Result<VerifyingKey, PcsRejected> {
    VerifyingKey::from_sec1_bytes(&cert.public_key().subject_public_key.data)
        .map_err(|e| PcsRejected::Chain(format!("certificate key not P-256: {e}")))
}

pub(crate) fn verify_cert_signed_by(
    cert: &X509Certificate<'_>,
    issuer: &VerifyingKey,
    what: &str,
    now_unix: u64,
) -> Result<(), PcsRejected> {
    // SECURITY: a certificate's validity window is part of its
    // signature's meaning. Checking only that the maths works accepts a
    // certificate its issuer has already retired — an NRAS signing leaf
    // whose key leaked would sign forever, the pin still matching and the
    // chain still verifying, because expiry would be nobody's job. Audit of
    // 2026-08-21: no window was checked on ANY of the three chains (PCK,
    // Intel collateral, NRAS leaf), and none of the stated limitations said
    // so. Checked here, once, for every chain that walks through this.
    let (not_before, not_after) = (
        cert.validity().not_before.timestamp(),
        cert.validity().not_after.timestamp(),
    );
    let now = now_unix as i64;
    if now < not_before || now > not_after {
        return Err(PcsRejected::Chain(format!(
            "{what}: outside its validity window [{not_before}, {not_after}] at {now}"
        )));
    }
    let sig = Signature::from_der(cert.signature_value.data.as_ref())
        .map_err(|e| PcsRejected::Chain(format!("{what}: signature not DER ECDSA: {e}")))?;
    issuer
        .verify(cert.tbs_certificate.as_ref(), &sig)
        .map_err(|_| PcsRejected::Chain(format!("{what}: signature does not verify")))
}

/// Does this root certificate satisfy the anchor? Shared with the live-TCB
/// verifier (`tcb.rs`), which anchors Intel's TCB signing chain on the SAME
/// pinned root as the PCK chain.
pub(crate) fn root_matches(root: &X509Certificate<'_>, anchor: &TrustAnchor) -> bool {
    match anchor {
        TrustAnchor::SpkiSha256Hex(expect) => {
            let got: String = Sha256::digest(root.public_key().raw)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            &got == expect
        }
        TrustAnchor::RootDer(der) => {
            // Compare by SPKI, not raw DER: a re-serialized certificate with
            // the same key is the same trust decision.
            Sha256::digest(root.public_key().raw).as_slice()
                == X509Certificate::from_der(der)
                    .map(|(_, c)| Sha256::digest(c.public_key().raw))
                    .map(|d| d.to_vec())
                    .unwrap_or_default()
                    .as_slice()
        }
    }
}

/// The quote's PCK chain (leaf first), every link verified and the root
/// pinned: what [`verify_tdx_chain`] walks, and what the PCK CRL check
/// (`tcb::verify_pck_crl`) needs the leaf's issuer from.
pub(crate) fn verified_pck_chain(
    pck_chain_pem: &[u8],
    anchor: &TrustAnchor,
    now_unix: u64,
) -> Result<Vec<Vec<u8>>, PcsRejected> {
    let mut certs_der: Vec<Vec<u8>> = Vec::new();
    for pem in Pem::iter_from_buffer(pck_chain_pem) {
        let pem = pem.map_err(|e| PcsRejected::Chain(format!("PEM: {e}")))?;
        certs_der.push(pem.contents);
    }
    // An Intel PCK chain is the PCK, the CA that issued it (Platform or
    // Processor) and the root. A longer chain is a leaf issued under a PCK,
    // whose key a host can extract (TEE.fail).
    if certs_der.len() != PCK_CHAIN_LEN {
        return Err(PcsRejected::Chain(format!(
            "chain has {} certificate(s); an Intel PCK chain is the PCK, its CA and the root",
            certs_der.len()
        )));
    }
    {
        let certs: Vec<X509Certificate<'_>> = certs_der
            .iter()
            .map(|der| {
                X509Certificate::from_der(der)
                    .map(|(_, c)| c)
                    .map_err(|e| PcsRejected::Chain(format!("X.509: {e}")))
            })
            .collect::<Result<_, _>>()?;

        // Root: last in the chain. Pin check first — a beautiful chain to the
        // wrong root is exactly the attack.
        let root = certs.last().expect("len checked");
        match anchor {
            TrustAnchor::SpkiSha256Hex(expect) => {
                let got: String = Sha256::digest(root.public_key().raw)
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect();
                if &got != expect {
                    return Err(PcsRejected::RootNotPinned);
                }
            }
            TrustAnchor::RootDer(expect) => {
                if certs_der.last().expect("len checked") != expect {
                    return Err(PcsRejected::RootNotPinned);
                }
            }
        }
        // Root is self-signed; verify it against its own key (integrity of the
        // pinned object), then each cert against its issuer, leaf-ward.
        let root_key = p256_key_from_cert(root)?;
        verify_cert_signed_by(root, &root_key, "root (self-signature)", now_unix)?;
        for i in (0..certs.len() - 1).rev() {
            let issuer_key = p256_key_from_cert(&certs[i + 1])?;
            verify_cert_signed_by(&certs[i], &issuer_key, &format!("chain link {i}"), now_unix)?;
        }
        intel_shape(&certs, &["the PCK", "the PCK's issuer", "the root"])
            .map_err(PcsRejected::Chain)?;
    }
    Ok(certs_der)
}

/// The PCK, its CA, the root: Intel's PCK chain in a TD Quote v4 (the 18/10
/// GCE quote carries those three, `x0ne-sdk/tests/real_quote_2026_10_18.rs`).
const PCK_CHAIN_LEN: usize = 3;

/// Intel's chain shape beyond the signatures, leaf first: each certificate
/// after the leaf a CA, the leaf not, each naming the next as its issuer.
/// `names` says what each one is. PCK chain here, collateral chains in `tcb`.
pub(crate) fn intel_shape(certs: &[X509Certificate<'_>], names: &[&str]) -> Result<(), String> {
    let name = |i: usize| names.get(i).copied().unwrap_or("a certificate");
    for i in (1..certs.len()).rev() {
        if !is_ca(&certs[i], name(i))? {
            return Err(format!("{} is not a CA (basicConstraints)", name(i)));
        }
    }
    if let Some(leaf) = certs.first() {
        if is_ca(leaf, name(0))? {
            return Err(format!("{} is a CA (basicConstraints cA true)", name(0)));
        }
    }
    for i in 0..certs.len().saturating_sub(1) {
        if certs[i].issuer().as_raw() != certs[i + 1].subject().as_raw() {
            return Err(format!(
                "chain link {i}: its issuer {} is not the subject {} of the next certificate",
                certs[i].issuer(),
                certs[i + 1].subject()
            ));
        }
    }
    Ok(())
}

/// basicConstraints cA, absent meaning false (RFC 5280 4.2.1.9).
fn is_ca(cert: &X509Certificate<'_>, what: &str) -> Result<bool, String> {
    cert.basic_constraints()
        .map(|bc| bc.is_some_and(|bc| bc.value.ca))
        .map_err(|e| format!("{what}: basicConstraints: {e}"))
}

/// The part of a TDX v4 quote the attestation key signs: header and TD report
/// body, `report_data` last. `None` when the quote is shorter. Read it from a
/// quote that passed [`verify_tdx_chain`].
pub fn signed_td_report(quote: &[u8]) -> Option<&[u8]> {
    quote.get(..SIGNED_LEN)
}

/// Verify the full chain. `quote` is the raw quote (header ‖ body ‖ signature
/// section); `anchor` is the pinned root.
pub fn verify_tdx_chain(
    quote: &[u8],
    anchor: &TrustAnchor,
    now_unix: u64,
) -> Result<VerifiedTdx, PcsRejected> {
    let sig = parse_sig_section(quote)?;

    // ── The PCK chain: leaf → … → root, root pinned ─────────────────────
    let certs_der = verified_pck_chain(sig.pck_chain_pem, anchor, now_unix)?;
    let (_, leaf) = X509Certificate::from_der(&certs_der[0])
        .map_err(|e| PcsRejected::Chain(format!("X.509: {e}")))?;
    let pck_key = p256_key_from_cert(&leaf)?;

    // ── QE report, signed by the PCK key ────────────────────────────────
    let qe_sig = Signature::from_slice(sig.qe_report_signature)
        .map_err(|e| PcsRejected::Malformed(format!("QE signature: {e}")))?;
    pck_key
        .verify(sig.qe_report, &qe_sig)
        .map_err(|_| PcsRejected::QeSignature)?;

    // ── The QE report binds the attestation key ─────────────────────────
    // qe_report.report_data == SHA-256(attestation_key ‖ qe_auth_data) ‖ 32 zero bytes
    let report_data = &sig.qe_report[320..384];
    let mut h = Sha256::new();
    h.update(sig.attestation_key);
    h.update(sig.qe_auth_data);
    let expect: [u8; 32] = h.finalize().into();
    if report_data[..32] != expect {
        return Err(PcsRejected::AttestationKeyBinding);
    }
    // SECURITY: Intel's TD QE zeroes these bytes (DCAP 72f800e, quoting_enclave_tdqe.cpp:851,
    // 986). go-tdx-guest requires it, Intel's QVL does not (2026-11-05); no real quote differs.
    if report_data[32..] != [0u8; 32] {
        return Err(PcsRejected::QeReportDataTail);
    }

    // ── The quote, signed by the attestation key ────────────────────────
    let att_key = p256_key_from_xy(sig.attestation_key)?;
    let quote_sig = Signature::from_slice(sig.quote_signature)
        .map_err(|e| PcsRejected::Malformed(format!("quote signature: {e}")))?;
    att_key
        .verify(&quote[..SIGNED_LEN], &quote_sig)
        .map_err(|_| PcsRejected::QuoteSignature)?;
    // The attributes are inside the signed body, checked once signed.
    let td = x0ne_confidential::tdx_quote::parse(quote)
        .map_err(|e| PcsRejected::TdReport(format!("{e:?}")))?;
    if td.is_debug() {
        return Err(PcsRejected::DebugTd);
    }
    let refused = td.refused_attributes();
    if refused != 0 {
        return Err(PcsRejected::TdAttributes(
            x0ne_confidential::tdx_quote::td_attribute_names(refused),
        ));
    }

    let pck_sha256_hex = Sha256::digest(&certs_der[0])
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let mut report_data = [0u8; 64];
    report_data.copy_from_slice(&quote[SIGNED_LEN - 64..SIGNED_LEN]);
    Ok(VerifiedTdx {
        pck_sha256_hex,
        report_data,
        limitations: LIMITATIONS,
    })
}

// Not when compiled into x0ne-sdk (its build.rs sets `x0ne_sdk`): these tests
// need this crate's fixtures and dev-dependencies.
#[cfg(all(test, not(x0ne_sdk)))]
pub(crate) mod tests {
    /// A fixed instant inside every fixture certificate's validity window
    /// (2026-02-12). Fixtures are minted by rcgen with its default window
    /// (1975..4096), so any realistic epoch second works; a constant keeps
    /// the tests deterministic now that the window is checked.
    pub(crate) const NOW: u64 = 1_770_000_000;

    use super::*;
    use p256::ecdsa::signature::Signer;
    use p256::ecdsa::SigningKey;
    use rcgen::{
        date_time_ymd, BasicConstraints, CertificateParams, DnType, IsCa, Issuer, KeyPair,
        PKCS_ECDSA_P256_SHA256,
    };

    /// A complete synthetic chain — our own "Intel" — so the WHOLE path is
    /// proven both ways: verifies when every link holds, and refuses when any
    /// single link is tampered. The production pin differs only by the anchor
    /// constant.
    pub(crate) struct Fixture {
        pub(crate) quote: Vec<u8>,
        pub(crate) root_der: Vec<u8>,
    }

    /// The composite test in `verify` uses the same fixture.
    pub(crate) fn fixture_for_composite() -> Fixture {
        build_fixture()
    }

    fn build_fixture() -> Fixture {
        build_fixture_with_leaf_window(None)
    }

    /// `leaf_window`: an explicit (notBefore, notAfter) year for the PCK leaf,
    /// so a test can mint a chain whose maths is perfect and whose leaf is
    /// retired. `None` keeps rcgen's default 1975..4096.
    fn build_fixture_with_leaf_window(leaf_window: Option<(i32, i32)>) -> Fixture {
        build_fixture_full(leaf_window, [0x5a; 64], [0; 8])
    }

    /// A genuine-looking quote answering a challenge of the caller's choice —
    /// what a client-side test needs to bind a quote to a session key.
    pub(crate) fn fixture_with_report_data(report_data: [u8; 64]) -> Fixture {
        build_fixture_full(None, report_data, [0; 8])
    }

    /// Same quote, TDATTRIBUTES bit 0 (DEBUG) set, signed like any other.
    pub(crate) fn fixture_debug_td(report_data: [u8; 64]) -> Fixture {
        build_fixture_full(None, report_data, [1, 0, 0, 0, 0, 0, 0, 0])
    }

    fn build_fixture_full(
        leaf_window: Option<(i32, i32)>,
        report_data: [u8; 64],
        td_attributes: [u8; 8],
    ) -> Fixture {
        // The quote: header (v4, TDX) + body; `signed` adds the signature.
        let mut quote = vec![0u8; SIGNED_LEN];
        quote[0] = 4; // version
        quote[4] = 0x81; // tee_type = TDX
        quote[SIGNED_LEN - 64..].copy_from_slice(&report_data);
        // Distinct, non-zero registers: an all-zero body would let a
        // pinning test pass against a parser that read nothing at all.
        // TD report body starts at 48; mr_td at +136, rtmr0 at +328, 48 B each.
        quote[48 + 136..48 + 184].copy_from_slice(&[0xa1; 48]); // mr_td
        quote[48 + 376..48 + 424].copy_from_slice(&[0xb2; 48]); // rtmr1
        quote[48 + 120..48 + 128].copy_from_slice(&td_attributes);
        signed(leaf_window, quote)
    }

    /// The header and TD report of `quote`, signed under this module's root:
    /// a real TD report with a signature the tests can make.
    pub(crate) fn fixture_signing(quote: &[u8]) -> Fixture {
        signed(None, quote[..SIGNED_LEN].to_vec())
    }

    fn root_params() -> CertificateParams {
        let mut p = CertificateParams::new(vec![]).unwrap();
        p.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        // 1970, not rcgen's 1975 default: several tests speak in small epoch
        // seconds (now = 1_000) about things other than time, and the window check would
        // make them fail for the wrong reason.
        p.not_before = date_time_ymd(1970, 1, 1);
        p
    }

    /// One root per test process, (key DER, certificate DER): a server test
    /// answers a client's challenge with a SECOND quote, and the client pins
    /// ONE root.
    fn root() -> &'static (Vec<u8>, Vec<u8>) {
        static ROOT: std::sync::OnceLock<(Vec<u8>, Vec<u8>)> = std::sync::OnceLock::new();
        ROOT.get_or_init(|| {
            let kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
            let cert = root_params().self_signed(&kp).unwrap();
            (kp.serialize_der(), cert.der().to_vec())
        })
    }

    fn root_issuer() -> Issuer<'static, KeyPair> {
        Issuer::new(
            root_params(),
            KeyPair::try_from(root().0.as_slice()).unwrap(),
        )
    }

    fn p256_of(kp: &KeyPair) -> SigningKey {
        p256::SecretKey::from_sec1_pem(&kp.serialize_pem())
            .map(SigningKey::from)
            .or_else(|_| {
                p256::pkcs8::DecodePrivateKey::from_pkcs8_pem(&kp.serialize_pem())
                    .map(|sk: p256::SecretKey| SigningKey::from(sk))
            })
            .expect("rcgen private key readable")
    }

    fn signed(leaf_window: Option<(i32, i32)>, quote: Vec<u8>) -> Fixture {
        // Root → intermediate → PCK leaf, all ECDSA-P256.
        let root_cert_der = &root().1;
        let root_issuer = root_issuer();

        let inter_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
        let mut inter_params = CertificateParams::new(vec![]).unwrap();
        inter_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        inter_params.not_before = date_time_ymd(1970, 1, 1);
        let inter_cert = inter_params
            .clone()
            .signed_by(&inter_kp, &root_issuer)
            .unwrap();
        let inter_issuer = Issuer::new(inter_params, inter_kp);

        let pck_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
        let mut pck_params = CertificateParams::new(vec![]).unwrap();
        let (from, to) = leaf_window.unwrap_or((1970, 4096));
        pck_params.not_before = date_time_ymd(from, 1, 1);
        pck_params.not_after = date_time_ymd(to, 1, 1);
        let pck_cert = pck_params.signed_by(&pck_kp, &inter_issuer).unwrap();

        let root_pem = pem_of(root_cert_der);
        let chain_pem = format!("{}{}{}", pck_cert.pem(), inter_cert.pem(), root_pem);
        Fixture {
            quote: with_signature(quote, &chain_pem, &p256_of(&pck_kp)),
            root_der: root_cert_der.clone(),
        }
    }

    /// `quote`'s header and TD report signed by an attestation key that a
    /// QE report binds, that report signed by `pck_sk`, `chain_pem` as the
    /// certification data.
    fn with_signature(mut quote: Vec<u8>, chain_pem: &str, pck_sk: &SigningKey) -> Vec<u8> {
        // Attestation key (raw P-256), QE report binding it, PCK-signed.
        // Deterministic test key: no RNG trait gymnastics, same key every run.
        let att_sk = SigningKey::from_slice(&[0x42u8; 32]).expect("valid scalar");
        let att_vk_xy = {
            // SEC1 uncompressed = 0x04 ‖ x ‖ y; the quote carries x ‖ y.
            let sec1 = att_sk.verifying_key().to_sec1_bytes();
            sec1[1..].to_vec()
        };
        let qe_auth: &[u8] = b"synthetic-qe-auth";
        let mut qe_report = vec![0u8; 384];
        let mut h = Sha256::new();
        h.update(&att_vk_xy);
        h.update(qe_auth);
        qe_report[320..352].copy_from_slice(&h.finalize());
        let qe_sig: Signature = pck_sk.sign(&qe_report);

        let quote_sig: Signature = att_sk.sign(&quote[..SIGNED_LEN]);

        // Signature section.
        let mut sig_data = Vec::new();
        sig_data.extend_from_slice(&quote_sig.to_bytes());
        sig_data.extend_from_slice(&att_vk_xy);
        sig_data.extend_from_slice(&6u16.to_le_bytes());
        sig_data.extend_from_slice(&0u32.to_le_bytes()); // outer size: unused by parser
        sig_data.extend_from_slice(&qe_report);
        sig_data.extend_from_slice(&qe_sig.to_bytes());
        sig_data.extend_from_slice(&(qe_auth.len() as u16).to_le_bytes());
        sig_data.extend_from_slice(qe_auth);
        sig_data.extend_from_slice(&5u16.to_le_bytes());
        sig_data.extend_from_slice(&(chain_pem.len() as u32).to_le_bytes());
        sig_data.extend_from_slice(chain_pem.as_bytes());

        quote.extend_from_slice(&(sig_data.len() as u32).to_le_bytes());
        quote.extend_from_slice(&sig_data);
        quote
    }

    /// A certificate named `cn`, issued by `issuer`: `ca` sets basicConstraints
    /// (`None`: no extension). Its PEM, parameters and key (DER).
    fn issued(
        cn: &str,
        ca: Option<bool>,
        issuer: &Issuer<'_, KeyPair>,
    ) -> (String, CertificateParams, Vec<u8>) {
        let kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
        let mut p = CertificateParams::new(vec![]).unwrap();
        p.distinguished_name.push(DnType::CommonName, cn);
        p.not_before = date_time_ymd(1970, 1, 1);
        p.is_ca = match ca {
            Some(true) => IsCa::Ca(BasicConstraints::Unconstrained),
            Some(false) => IsCa::ExplicitNoCa,
            None => IsCa::NoCa,
        };
        let pem = p.clone().signed_by(&kp, issuer).unwrap().pem();
        (pem, p, kp.serialize_der())
    }

    fn issuer_of(params: &CertificateParams, key_der: &[u8]) -> Issuer<'static, KeyPair> {
        Issuer::new(params.clone(), KeyPair::try_from(key_der).unwrap())
    }

    /// A quote whose QE report the key `signer_der` signs, `chain` and this
    /// module's root as its PCK chain.
    fn quote_with_chain(chain: &str, signer_der: &[u8]) -> Fixture {
        let mut quote = vec![0u8; SIGNED_LEN];
        quote[0] = 4;
        quote[4] = 0x81;
        let chain_pem = format!("{chain}{}", pem_of(&root().1));
        let signer = p256_of(&KeyPair::try_from(signer_der).unwrap());
        Fixture {
            quote: with_signature(quote, &chain_pem, &signer),
            root_der: root().1.clone(),
        }
    }

    fn chain_refusal(fx: &Fixture) -> String {
        match verify_tdx_chain(&fx.quote, &TrustAnchor::RootDer(fx.root_der.clone()), NOW) {
            Ok(_) => "verified".into(),
            Err(e) => e.to_string(),
        }
    }

    /// SECURITY: whoever holds one PCK's private
    /// key (TEE.fail) issues a leaf under it and signs the QE report with
    /// that leaf: every signature verifies, and the chain is one certificate
    /// longer than Intel's (the PCK, its CA, the root). Refused here, in the
    /// TD, the relay and the SDK alike, before a byte of that leaf is read
    /// as the platform's identity. Shorter chains are refused as well.
    #[test]
    fn a_pck_chain_of_other_than_three_certificates_is_refused() {
        let root = root_issuer();
        let (ca_pem, ca, ca_key) = issued("Intel SGX PCK Platform CA", Some(true), &root);
        let (pck_pem, pck, pck_key) = issued(
            "Intel SGX PCK Certificate",
            Some(false),
            &issuer_of(&ca, &ca_key),
        );
        let control = quote_with_chain(&format!("{pck_pem}{ca_pem}"), &pck_key);
        assert_eq!(chain_refusal(&control), "verified");
        let (forged_pem, _, forged_key) = issued(
            "Intel SGX PCK Certificate",
            None,
            &issuer_of(&pck, &pck_key),
        );
        let forged = quote_with_chain(&format!("{forged_pem}{pck_pem}{ca_pem}"), &forged_key);
        let e = chain_refusal(&forged);
        assert!(e.contains("chain has 4 certificate(s)"), "{e}");
        let (direct_pem, _, direct_key) = issued("Intel SGX PCK Certificate", None, &root);
        let short = quote_with_chain(&direct_pem, &direct_key);
        let e = chain_refusal(&short);
        assert!(e.contains("chain has 2 certificate(s)"), "{e}");
    }

    /// SECURITY: the chain's shape, beyond its signatures: the PCK's issuer
    /// and the root are CAs, the PCK is not, and each certificate names the
    /// next one's subject as its issuer.
    #[test]
    fn a_pck_chain_of_another_shape_than_intels_is_refused() {
        let root = root_issuer();
        let (ca_pem, ca, ca_key) = issued("Intel SGX PCK Platform CA", Some(true), &root);
        let ca_issuer = issuer_of(&ca, &ca_key);
        let leaf = |issuer: &Issuer<'_, KeyPair>, is_ca: Option<bool>| {
            let (pem, _, key) = issued("Intel SGX PCK Certificate", is_ca, issuer);
            (pem, key)
        };
        // A leaf without basicConstraints, as rcgen issues one: not a CA.
        let (pem, key) = leaf(&ca_issuer, None);
        let ok = quote_with_chain(&format!("{pem}{ca_pem}"), &key);
        assert_eq!(chain_refusal(&ok), "verified");
        let mut cases: Vec<(&str, Fixture, &str)> = Vec::new();
        for (what, flag) in [("no basicConstraints", None), ("cA false", Some(false))] {
            let (not_ca_pem, not_ca, not_ca_key) = issued("Intel SGX PCK Platform CA", flag, &root);
            let (pem, key) = leaf(&issuer_of(&not_ca, &not_ca_key), None);
            let fx = quote_with_chain(&format!("{pem}{not_ca_pem}"), &key);
            cases.push((what, fx, "the PCK's issuer is not a CA"));
        }
        let (pem, key) = leaf(&ca_issuer, Some(true));
        cases.push((
            "a PCK that is a CA",
            quote_with_chain(&format!("{pem}{ca_pem}"), &key),
            "the PCK is a CA",
        ));
        let mut renamed = ca.clone();
        renamed.distinguished_name = rcgen::DistinguishedName::new();
        renamed
            .distinguished_name
            .push(DnType::CommonName, "Intel SGX PCK Processor CA");
        let (pem, key) = leaf(&issuer_of(&renamed, &ca_key), None);
        cases.push((
            "a PCK naming another issuer, signed by its CA's key",
            quote_with_chain(&format!("{pem}{ca_pem}"), &key),
            "is not the subject",
        ));
        for (what, fx, says) in cases {
            let e = chain_refusal(&fx);
            assert!(e.contains(says), "{what}: {e}");
        }
    }

    /// SECURITY: a pinned root that is no CA does not issue a PCK chain.
    #[test]
    fn a_pinned_root_that_is_not_a_ca_is_refused() {
        let kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
        let mut p = CertificateParams::new(vec![]).unwrap();
        p.not_before = date_time_ymd(1970, 1, 1);
        let root_der = p.clone().self_signed(&kp).unwrap().der().to_vec();
        let root = Issuer::new(p, kp);
        let (ca_pem, ca, ca_key) = issued("Intel SGX PCK Platform CA", Some(true), &root);
        let (pem, _, key) = issued("Intel SGX PCK Certificate", None, &issuer_of(&ca, &ca_key));
        let mut quote = vec![0u8; SIGNED_LEN];
        quote[0] = 4;
        quote[4] = 0x81;
        let chain = format!("{pem}{ca_pem}{}", pem_of(&root_der));
        let quote = with_signature(
            quote,
            &chain,
            &p256_of(&KeyPair::try_from(key.as_slice()).unwrap()),
        );
        let e = verify_tdx_chain(&quote, &TrustAnchor::RootDer(root_der), NOW)
            .unwrap_err()
            .to_string();
        assert!(e.contains("the root is not a CA"), "{e}");
    }

    fn pem_of(der: &[u8]) -> String {
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(der);
        let mut out = String::from("-----BEGIN CERTIFICATE-----\n");
        for line in b64.as_bytes().chunks(64) {
            out.push_str(std::str::from_utf8(line).unwrap());
            out.push('\n');
        }
        out.push_str("-----END CERTIFICATE-----\n");
        out
    }

    /// SECURITY: a certificate whose validity window has closed must
    /// be refused even though every signature in the chain still verifies.
    /// A failure here means an issuer can no longer retire a certificate: a
    /// PCK leaf whose key leaked in 2020 would still open the gate today,
    /// because the maths does not expire and nobody else was checking.
    #[test]
    fn a_chain_whose_leaf_expired_is_refused_though_every_signature_verifies() {
        let fx = build_fixture_with_leaf_window(Some((2015, 2020)));
        // The window closed on 2020-01-01; NOW is 2026-02-12.
        let r = verify_tdx_chain(&fx.quote, &TrustAnchor::RootDer(fx.root_der), NOW);
        let msg = format!("{:?}", r.expect_err("an expired leaf must be refused"));
        assert!(
            msg.contains("validity window"),
            "the refusal must name the window, got: {msg}"
        );
    }

    /// SECURITY: the other end of the same window. A certificate
    /// post-dated into the future is not usable yet; accepting it would let
    /// a compromised CA mint credentials that verify before their own issuer
    /// intended them to.
    #[test]
    fn a_chain_whose_leaf_is_not_yet_valid_is_refused() {
        let fx = build_fixture_with_leaf_window(Some((2090, 2095)));
        let r = verify_tdx_chain(&fx.quote, &TrustAnchor::RootDer(fx.root_der), NOW);
        let msg = format!("{:?}", r.expect_err("a not-yet-valid leaf must be refused"));
        assert!(
            msg.contains("validity window"),
            "the refusal must name the window, got: {msg}"
        );
    }

    /// SECURITY: a debug TD lets the host read and write its memory.
    /// A genuine, correctly signed quote from one must not verify.
    #[test]
    fn a_debug_td_is_refused_though_its_chain_verifies() {
        let fx = fixture_debug_td([0x5a; 64]);
        let r = verify_tdx_chain(&fx.quote, &TrustAnchor::RootDer(fx.root_der), NOW);
        let msg = r.expect_err("a debug TD must be refused").to_string();
        assert!(msg.contains("DEBUG"), "the refusal must name DEBUG: {msg}");
    }

    /// SECURITY: the attribute policy holds on a correctly signed quote, the
    /// check the TD, the relay and the SSH client all run: a profiling bit,
    /// MIGRATABLE or a reserved bit refuses, by name.
    #[test]
    fn a_profiled_or_migratable_td_is_refused_though_its_chain_verifies() {
        for (attributes, name) in [
            (1u64 << 5, "PERF_PROF"),
            ((1 << 28) | (1 << 29), "MIGRATABLE"),
            ((1 << 28) | (1 << 24), "bit 24"),
            ((1 << 28) | (1 << 40), "bit 40"),
        ] {
            let fx = build_fixture_full(None, [0x5a; 64], attributes.to_le_bytes());
            let r = verify_tdx_chain(&fx.quote, &TrustAnchor::RootDer(fx.root_der), NOW);
            let msg = r.expect_err(&format!("{name} must be refused")).to_string();
            assert!(msg.contains(name), "the refusal must name {name}: {msg}");
        }
    }

    /// The control: GCE's attributes (SEPT_VE_DISABLE alone, 2026-10-17 and
    /// 10-18), and with the positive ICSSD and LASS and PERFMON, verify signed.
    #[test]
    fn the_td_attributes_measured_on_gce_verify_on_a_signed_quote() {
        for attributes in [1u64 << 28, (1 << 28) | (1 << 16) | (1 << 27) | (1 << 63)] {
            let fx = build_fixture_full(None, [0x5a; 64], attributes.to_le_bytes());
            verify_tdx_chain(&fx.quote, &TrustAnchor::RootDer(fx.root_der), NOW)
                .unwrap_or_else(|e| panic!("{attributes:#x}: {e}"));
        }
    }

    /// The full chain verifies against its own pinned root — and against the
    /// SPKI-digest form of the same pin.
    #[test]
    fn a_correct_chain_verifies_under_its_pinned_root() {
        let fx = build_fixture();
        let v = verify_tdx_chain(&fx.quote, &TrustAnchor::RootDer(fx.root_der.clone()), NOW)
            .expect("verifies");
        assert_eq!(v.pck_sha256_hex.len(), 64);
        assert!(
            !v.limitations.is_empty(),
            "limitations are stated, not hidden"
        );
        // Same chain, pinned by SPKI digest.
        let (_, root) = X509Certificate::from_der(&fx.root_der).unwrap();
        let spki: String = Sha256::digest(root.public_key().raw)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        verify_tdx_chain(&fx.quote, &TrustAnchor::SpkiSha256Hex(spki), NOW)
            .expect("verifies by SPKI");
    }

    /// A perfect chain to the WRONG root is the attack this module exists to
    /// stop: pinned to production Intel, our synthetic chain is refused.
    #[test]
    fn a_chain_to_an_unpinned_root_is_refused() {
        let fx = build_fixture();
        let r = verify_tdx_chain(&fx.quote, &TrustAnchor::intel_production(), NOW);
        assert_eq!(r.unwrap_err(), PcsRejected::RootNotPinned);
    }

    /// Flipping one byte of the signed region breaks the quote signature.
    #[test]
    fn a_tampered_quote_body_is_refused() {
        let mut fx = build_fixture();
        fx.quote[HEADER + 10] ^= 1;
        let r = verify_tdx_chain(&fx.quote, &TrustAnchor::RootDer(fx.root_der), NOW);
        assert_eq!(r.unwrap_err(), PcsRejected::QuoteSignature);
    }

    /// SECURITY: a failure means a QE report signed by the PCK may carry 32
    /// bytes no check reads, where Intel's TD QE writes zeros.
    #[test]
    fn a_qe_report_whose_report_data_does_not_end_in_zeros_is_refused() {
        let root = root_issuer();
        let (ca_pem, ca, ca_key) = issued("Intel SGX PCK Platform CA", Some(true), &root);
        let (pck_pem, _, pck_key) = issued(
            "Intel SGX PCK Certificate",
            Some(false),
            &issuer_of(&ca, &ca_key),
        );
        let control = quote_with_chain(&format!("{pck_pem}{ca_pem}"), &pck_key);
        assert_eq!(chain_refusal(&control), "verified");
        let pck = p256_of(&KeyPair::try_from(pck_key.as_slice()).unwrap());
        // Quote signature, attestation key, data type and size, then the QE report.
        let report = SIGNED_LEN + 4 + 64 + 64 + 2 + 4;
        for at in [320 + 32, 320 + 63] {
            let mut fx = quote_with_chain(&format!("{pck_pem}{ca_pem}"), &pck_key);
            fx.quote[report + at] = 1;
            let sig: Signature = pck.sign(&fx.quote[report..report + 384]);
            fx.quote[report + 384..report + 448].copy_from_slice(&sig.to_bytes());
            let e = chain_refusal(&fx);
            assert!(e.contains("is not zero"), "byte {at}: {e}");
        }
    }

    /// A swapped attestation key fails the QE binding — the exact spot a
    /// forged key would try to slip through.
    #[test]
    fn a_swapped_attestation_key_fails_the_qe_binding() {
        let mut fx = build_fixture();
        // attestation key sits right after the 4-byte length and 64-byte sig
        let off = SIGNED_LEN + 4 + 64;
        fx.quote[off] ^= 1;
        let r = verify_tdx_chain(&fx.quote, &TrustAnchor::RootDer(fx.root_der), NOW);
        assert!(matches!(
            r.unwrap_err(),
            PcsRejected::AttestationKeyBinding | PcsRejected::Malformed(_)
        ));
    }

    /// End to end: a chain the fixture root pins yields a CRYPTOGRAPHIC
    /// verdict, and THAT verdict opens the gate — the first time in this
    /// codebase that the door can open on evidence rather than on a
    /// hand-built verdict. The composite (`verify::cryptographic`) still
    /// refuses (GPU half), so serving remains impossible: exactly the
    /// intended state until NRAS lands.
    #[test]
    fn a_pinned_chain_yields_a_cryptographic_verdict_that_opens_the_gate() {
        use crate::gate::Gate;
        use x0ne_confidential::model::{
            AttestationEvidence, EvidenceFormat, ExecutionEnvironment, GpuCcMode, TeeTechnology,
        };
        let fx = build_fixture();
        let policy = x0ne_confidential::policy::VerificationPolicy {
            workload_name: "x0ne-infer".into(),
            accepted_cpu: vec![TeeTechnology::IntelTdx],
            // A TDX-only verdict cannot carry `GpuCcMode::On`, and
            // asking for it here is refused rather than granted.
            require_gpu_cc_on: false,
            expected_measurements: vec![],
        };
        let evidence = AttestationEvidence {
            format: EvidenceFormat::TdxQuote,
            payload: fx.quote.clone(),
            collected_at_unix: 1_000,
        };
        let verdict = crate::verify::cryptographic_tdx(
            &evidence,
            &policy,
            &TrustAnchor::RootDer(fx.root_der),
            // Chain only: this test anchors the ROOT pin, not a challenge.
            None,
            NOW,
        );
        assert!(verdict.is_verified(), "{verdict:?}");
        let proof = Gate::attest(
            ExecutionEnvironment::ConfidentialVmWithGpu {
                cpu: TeeTechnology::IntelTdx,
                gpu_mode: GpuCcMode::On,
            },
            &evidence,
            verdict,
            1_000,
        )
        .expect("a cryptographic verdict opens the gate");
        assert_eq!(proof.identity().name, "x0ne-infer");
    }

    /// The pinned production constant matches Intel's published root — the
    /// certificate is vendored under tests/ (fetched 2026-08-20, 659 bytes)
    /// and digested through the SAME code path the verifier uses
    /// (x509-parser SPKI `.raw`). A failure means either the vendored file
    /// or the constant changed: both are review events, never silent.
    #[test]
    fn the_production_pin_matches_intels_published_root() {
        let der = include_bytes!("../tests/intel_sgx_root_ca.der");
        let (_, cert) = X509Certificate::from_der(der).expect("Intel root parses");
        let got: String = Sha256::digest(cert.public_key().raw)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(got, INTEL_SGX_ROOT_CA_SPKI_SHA256_HEX);
    }

    /// A truncated signature section is malformed, never a panic.
    #[test]
    fn a_truncated_signature_section_is_malformed_not_a_panic() {
        let fx = build_fixture();
        for cut in [SIGNED_LEN + 3, SIGNED_LEN + 70, fx.quote.len() - 9] {
            let r = verify_tdx_chain(
                &fx.quote[..cut],
                &TrustAnchor::RootDer(fx.root_der.clone()),
                NOW,
            );
            assert!(r.is_err(), "cut at {cut} must refuse");
        }
    }

    /// Lengths at u32::MAX are refused as malformed: added to an offset they
    /// would wrap a 32-bit usize (the SDK's wasm32 build) without checks.
    #[test]
    fn a_length_at_u32_max_is_malformed_not_a_wrap() {
        let fx = build_fixture();
        let anchor = || TrustAnchor::RootDer(fx.root_der.clone());
        let mut q = fx.quote.clone();
        q[SIGNED_LEN..SIGNED_LEN + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let e = verify_tdx_chain(&q, &anchor(), NOW)
            .unwrap_err()
            .to_string();
        assert!(e.contains("sig_data_len says 4294967295"), "{e}");
        let auth = SIGNED_LEN + 4 + 64 + 64 + 2 + 4 + 384 + 64;
        let auth_len = u16::from_le_bytes([fx.quote[auth], fx.quote[auth + 1]]) as usize;
        let inner = auth + 2 + auth_len + 2;
        let mut q = fx.quote.clone();
        q[inner..inner + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let e = verify_tdx_chain(&q, &anchor(), NOW)
            .unwrap_err()
            .to_string();
        assert!(e.contains("PCK chain PEM: need 4294967295 bytes"), "{e}");
    }
}
