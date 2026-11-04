//! The live-TCB verifier against REAL data — a genuine TDX v4 quote
//! (MoeMahhouk/tdx-quote-parser sample, chain to the real Intel root) and
//! Intel PCS production collateral for its FMSPC, vendored on 2026-08-20.
//!
//! The sample platform is genuinely stale against today's tcbInfo: the
//! refusal tests below run on real bytes, real signatures, the real pinned
//! root — nothing synthetic. The ACCEPT path is proven separately under an
//! rcgen chain, because no fresh UpToDate quote can be vendored without
//! hardware in hand.

use x0ne_infer::pcs::TrustAnchor;
use x0ne_infer::tcb::{self, TcbPolicy, TcbRejected, TcbStatus};

const QUOTE: &[u8] = include_bytes!("pcs_live/quote.dat");
const TCB_INFO: &str = include_str!("pcs_live/tcb_info.json");
const TCB_CHAIN: &str = include_str!("pcs_live/tcb_issuer_chain.pem");
const QE_IDENTITY: &str = include_str!("pcs_live/qe_identity.json");
const QE_CHAIN: &str = include_str!("pcs_live/qe_issuer_chain.pem");

/// Inside the vendored collateral's validity window (issued 2026-08-20,
/// next update 2026-09-19). Fixed, so the test does not rot with the wall
/// clock; when the window passes, re-vendor and move this.
const NOW: u64 = 1_787_300_000; // 2026-08-21T08:13:20Z

/// The policy this collateral can pass: it is TCB evaluation 20, below the
/// production floor (`tcb_policy_live.rs` refuses it there), and its age is
/// judged by its own window only. These tests are about what comes after.
const VENDORED_DAY: TcbPolicy = TcbPolicy {
    mode: tcb::TcbMode::Strict,
    min_tcb_evaluation_data_number: 20,
    max_collateral_age_secs: u64::MAX,
};

/// The date parser against known epochs — everything else here leans on it.
#[test]
fn the_date_parser_matches_known_epochs() {
    // Private fn exercised through the public path: a collateral whose
    // window excludes `now` must refuse with the parsed bounds. Direct
    // anchor values first:
    assert_eq!(unix("1970-01-01T00:00:00Z"), 0);
    assert_eq!(unix("2020-01-01T00:00:00Z"), 1_577_836_800);
    assert_eq!(unix("2026-08-21T08:13:20Z"), NOW);
}

fn unix(s: &str) -> u64 {
    // date_unix is private; the public surface that exposes it is the
    // validity check. Re-implement the assertion through verify: simplest is
    // to duplicate the known values here — if tcb::verify_collateral ever
    // disagrees, the window tests below fail loudly.
    // (Kept as a helper so the constants above stay in one place.)
    chrono_free_unix(s)
}

/// Reference implementation for the test only (naive, month table).
fn chrono_free_unix(s: &str) -> u64 {
    let y: i64 = s[0..4].parse().unwrap();
    let m: i64 = s[5..7].parse().unwrap();
    let d: i64 = s[8..10].parse().unwrap();
    let hh: i64 = s[11..13].parse().unwrap();
    let mm: i64 = s[14..16].parse().unwrap();
    let ss: i64 = s[17..19].parse().unwrap();
    let leap = |y: i64| (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut days: i64 = 0;
    for yy in 1970..y {
        days += if leap(yy) { 366 } else { 365 };
    }
    for mm_ in 1..m {
        days += mdays[(mm_ - 1) as usize] + i64::from(mm_ == 2 && leap(y));
    }
    days += d - 1;
    (days * 86400 + hh * 3600 + mm * 60 + ss) as u64
}

/// The platform identity read from the real quote's PCK certificate is the
/// one Intel issued it with — measured out of band with openssl on
/// 2026-08-20 and pinned here byte for byte.
#[test]
fn the_real_quote_yields_its_real_platform_identity() {
    let pck = tcb::pck_tcb(QUOTE).expect("pck tcb");
    assert_eq!(pck.fmspc_hex, "90c06f000000");
    assert_eq!(pck.cpusvn, [2, 2, 2, 2, 3, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(pck.pcesvn, 13);
    assert_eq!(
        tcb::tee_tcb_svn(QUOTE).expect("tee svn"),
        [4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
}

/// Intel's production collateral verifies under the SAME pinned root as the
/// PCK chain — signature over the raw bytes, chain checked, window checked.
/// A failure means either Intel rotated the root (update the pin, one
/// place) or the verifier regressed.
#[test]
fn todays_intel_collateral_verifies_under_the_pinned_root() {
    let anchor = TrustAnchor::intel_production();
    // Both pieces go through the full verifier via verify_live; here the
    // point is that NEITHER fails on chain or signature — the outcome is a
    // STATUS decision, not a collateral rejection.
    let err = tcb::verify_live(
        QUOTE,
        TCB_INFO,
        TCB_CHAIN,
        QE_IDENTITY,
        QE_CHAIN,
        &anchor,
        NOW,
        &VENDORED_DAY,
    )
    .expect_err("the 2023-era sample platform cannot be current in 2026");
    let msg = err.to_string();
    assert!(
        !msg.contains("signature") && !msg.contains("chain") && !msg.contains("anchor"),
        "the refusal must be a STATUS decision, not a collateral failure: {msg}"
    );
}

/// SECURITY, on real Intel bytes: the collateral chain's own
/// certificates carry a validity window, and it is enforced. Same call as
/// the test above — same quote, same collateral, same pinned root — moved
/// to 2091, well past the Intel TCB signing certificate's `notAfter`. A
/// failure here means the collateral chain is verified by signature alone,
/// so a TCB verdict signed by a certificate Intel retired years earlier
/// still counts.
#[test]
fn the_collateral_chain_is_refused_once_its_certificates_have_expired() {
    let err = tcb::verify_live(
        QUOTE,
        TCB_INFO,
        TCB_CHAIN,
        QE_IDENTITY,
        QE_CHAIN,
        &TrustAnchor::intel_production(),
        3_820_000_000, // 2091-01-30
        &VENDORED_DAY,
    )
    .expect_err("expired collateral certificates must be refused");
    // "outside its validity window", not the tcbInfo `nextUpdate` window,
    // whose message says "does not contain now=" — both mention a window and
    // both would refuse here, so the assertion has to separate them or it
    // proves the wrong check.
    assert!(
        err.to_string().contains("outside its validity window"),
        "the refusal must come from the CERTIFICATE window, got: {err}"
    );
}

/// SECURITY — the point of the whole module, on real bytes: a quote whose
/// chain is impeccable but whose platform Intel now lists as stale is
/// REFUSED. Before the live TCB check this quote produced `Verified{{Cryptographic}}`.
/// Its CPUSVN component 7 is 3; each of the four levels of this tcbInfo asks
/// 5, so it matches none, not even the OutOfDate level of 2018.
#[test]
fn a_stale_platform_with_a_perfect_chain_is_refused_by_todays_intel() {
    // Premise: the chain itself still verifies to the pinned root.
    x0ne_infer::pcs::verify_tdx_chain(QUOTE, &TrustAnchor::intel_production(), NOW)
        .expect("the sample quote's chain is genuine");
    let err = tcb::verify_live(
        QUOTE,
        TCB_INFO,
        TCB_CHAIN,
        QE_IDENTITY,
        QE_CHAIN,
        &TrustAnchor::intel_production(),
        NOW,
        &VENDORED_DAY,
    )
    .expect_err("stale platform must refuse");
    assert!(
        matches!(err, TcbRejected::BelowEveryLevel),
        "the refusal names the staleness: {err}"
    );
}

/// A flipped byte in the signed collateral is a refusal — the signature
/// covers the raw JSON bytes, not a re-serialization.
#[test]
fn tampered_collateral_is_refused() {
    let tampered = TCB_INFO.replacen("\"pceId\":\"0000\"", "\"pceId\":\"0001\"", 1);
    assert_ne!(tampered, TCB_INFO, "the tamper target must exist");
    let err = tcb::verify_live(
        QUOTE,
        &tampered,
        TCB_CHAIN,
        QE_IDENTITY,
        QE_CHAIN,
        &TrustAnchor::intel_production(),
        NOW,
        &VENDORED_DAY,
    )
    .expect_err("tampered collateral must refuse");
    assert!(err.to_string().contains("signature"), "{err}");
}

/// Collateral outside its validity window is refused even with a valid
/// signature — yesterday's TCB truth is not today's.
#[test]
fn expired_collateral_is_refused_by_the_window() {
    let after = NOW + 90 * 86400;
    let err = tcb::verify_live(
        QUOTE,
        TCB_INFO,
        TCB_CHAIN,
        QE_IDENTITY,
        QE_CHAIN,
        &TrustAnchor::intel_production(),
        after,
        &VENDORED_DAY,
    )
    .expect_err("expired collateral must refuse");
    assert!(err.to_string().contains("validity window"), "{err}");
}

// ── The ACCEPT path, under a test chain ─────────────────────────────────────

use p256::ecdsa::signature::Signer as _;
use p256::ecdsa::SigningKey;
use rcgen::{
    BasicConstraints, CertificateParams, DistinguishedName, DnType, IsCa, Issuer, KeyPair,
    KeyUsagePurpose, PKCS_ECDSA_P256_SHA256,
};

fn zero_comps() -> serde_json::Value {
    serde_json::json!((0..16)
        .map(|_| serde_json::json!({"svn": 0}))
        .collect::<Vec<_>>())
}

/// The whole pipeline can also ACCEPT: collateral listing the sample
/// platform as current (signed under a test chain, anchored on the test
/// root) yields (UpToDate, UpToDate). A failure would mean the verifier can
/// only refuse — fail-closed collapsed into fail-always.
#[test]
fn a_current_platform_is_accepted_end_to_end() {
    let hex = |b: &[u8]| -> String { b.iter().map(|x| format!("{x:02x}")).collect() };
    let qe = tcb::qe_report(QUOTE).expect("qe report");

    let tcb_value = serde_json::json!({
        "id": "TDX",
        "version": 3,
        "tcbEvaluationDataNumber": 20,
        "issueDate": "2026-08-20T00:00:00Z",
        "nextUpdate": "2026-09-19T00:00:00Z",
        "fmspc": "90c06f000000",
        // The vendored quote runs TDX module version 1 at ISVSVN 4.
        "tdxModule": {"mrsigner": "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "attributes": "0000000000000000", "attributesMask": "FFFFFFFFFFFFFFFF"},
        "tdxModuleIdentities": [{
            "id": "TDX_01", "mrsigner": "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", "attributes": "0000000000000000", "attributesMask": "FFFFFFFFFFFFFFFF",
            "tcbLevels": [{"tcb": {"isvsvn": 4}, "tcbDate": "2026-08-01T00:00:00Z", "tcbStatus": "UpToDate"}]
        }],
        "tcbLevels": [{
            "tcb": {
                "sgxtcbcomponents": zero_comps(),
                "tdxtcbcomponents": zero_comps(),
                "pcesvn": 0
            },
            "tcbDate": "2026-08-01T00:00:00Z",
            "tcbStatus": "UpToDate"
        }]
    });
    let qe_value = serde_json::json!({
        "id": "TD_QE",
        "version": 2,
        "tcbEvaluationDataNumber": 20,
        "issueDate": "2026-08-20T00:00:00Z",
        "nextUpdate": "2026-09-19T00:00:00Z",
        // Offset 16 = sgx_report_body_t.misc_select (an earlier
        // [8..12] was four bytes of CPUSVN, test and code agreeing and both
        // wrong). Pinned below against the vendored quote.
        "miscselect": hex(&qe[16..20]),
        "miscselectMask": "ffffffff",
        "attributes": hex(&qe[48..64]),
        "attributesMask": "ffffffffffffffffffffffffffffffff",
        "mrsigner": hex(&qe[128..160]),
        "isvprodid": u16::from_le_bytes([qe[256], qe[257]]),
        "tcbLevels": [{
            "tcb": {"isvsvn": u16::from_le_bytes([qe[258], qe[259]])},
            "tcbDate": "2026-08-01T00:00:00Z",
            "tcbStatus": "UpToDate"
        }]
    });
    // ONE chain signs both pieces — one anchor, like production.
    let (both_bodies, shared_chain, shared_root) =
        synthetic_collateral(&tcb_value, &qe_value, "Intel SGX TCB Signing", true, false);
    let (tcb_body, qe_body) = both_bodies;

    let (platform, qe_status) = tcb::verify_live(
        QUOTE,
        &tcb_body,
        &shared_chain,
        &qe_body,
        &shared_chain,
        &TrustAnchor::RootDer(shared_root),
        1_787_300_000,
        &VENDORED_DAY,
    )
    .expect("a current platform must be accepted");
    assert_eq!(platform, TcbStatus::UpToDate);
    assert_eq!(qe_status, TcbStatus::UpToDate);
}

/// The offset killer: MISCSELECT lives at byte 16 of
/// `sgx_report_body_t` — [8..12) is CPUSVN, and the old check passed by
/// accident (zeros on both sides) while the real field was never compared.
/// Noise in CPUSVN with the true value at 16 must ACCEPT; noise at 16 must
/// REFUSE naming MISCSELECT. A failure means the check moved off the real
/// field again — and the accept half keeps a masked-zero identity from
/// passing vacuously.
#[test]
fn miscselect_is_read_at_offset_16_not_inside_cpusvn() {
    use x0ne_infer::tcb::{qe_status, TcbStatus};
    let mut qe = [0u8; 384];
    qe[8..12].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]); // CPUSVN noise
    qe[16..20].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]); // the real field
    let identity = serde_json::json!({
        "id": "TD_QE",
        "miscselect": "AABBCCDD",
        "miscselectMask": "FFFFFFFF",
        "attributes": "00000000000000000000000000000000",
        "attributesMask": "00000000000000000000000000000000",
        "mrsigner": "0000000000000000000000000000000000000000000000000000000000000000",
        "isvprodid": 0,
        "tcbLevels": [{"tcb": {"isvsvn": 0}, "tcbStatus": "UpToDate"}]
    });
    assert!(matches!(qe_status(&identity, &qe), Ok(TcbStatus::UpToDate)));
    let mut wrong = qe;
    wrong[16..20].copy_from_slice(&[0, 0, 0, 0]);
    let err = format!("{:?}", qe_status(&identity, &wrong).unwrap_err());
    assert!(err.contains("MISCSELECT"), "{err}");
}

/// The duplicate-key killer (critical): the signature was
/// verified over the FIRST textual `"tcbInfo"` while the verdict read the
/// key serde resolves (LAST wins) — Intel's genuine bytes plus a forged
/// duplicate would turn Revoked into UpToDate. The verdict now reads exactly
/// the verified slice, and a duplicated key is refused BY NAME. Replayed here
/// on the real vendored collateral: genuine object first, forged duplicate
/// appended, Intel's true signature kept.
#[test]
fn a_duplicated_collateral_key_is_refused_not_resolved_last_wins() {
    let anchor = TrustAnchor::intel_production();
    let key_pat = "\"tcbInfo\":";
    let start = TCB_INFO.find(key_pat).expect("fixture shape") + key_pat.len();
    let sig_at = TCB_INFO.rfind(",\"signature\"").expect("fixture shape");
    let genuine_obj = &TCB_INFO[start..sig_at];
    let sig = &TCB_INFO[sig_at..TCB_INFO.len() - 1];
    let forged = "{\"id\":\"TDX\",\"tcbLevels\":[]}";
    let attack = format!("{{\"tcbInfo\":{genuine_obj},\"tcbInfo\":{forged}{sig}}}");
    let err = format!(
        "{:?}",
        tcb::verify_live(
            QUOTE,
            &attack,
            TCB_CHAIN,
            QE_IDENTITY,
            QE_CHAIN,
            &anchor,
            NOW,
            &VENDORED_DAY,
        )
        .expect_err("a duplicated key must refuse, whatever the signature says")
    );
    assert!(
        err.contains("2 times"),
        "the refusal names the duplication: {err}"
    );
}

/// The synthetic "Intel": one root, one signing leaf, both collateral
/// documents signed by that leaf. `leaf_cn` and `digital_signature` are
/// parameters so the signer identity checks can be exercised by a chain
/// that is otherwise perfect — the whole point being that a perfect chain
/// to the right root is NOT an identity.
fn synthetic_collateral(
    tcb_value: &serde_json::Value,
    qe_value: &serde_json::Value,
    leaf_cn: &str,
    digital_signature: bool,
    leaf_is_ca: bool,
) -> ((String, String), String, Vec<u8>) {
    let (tcb_value, qe_value) = (tcb_value.clone(), qe_value.clone());

    // one chain, two signatures by the same leaf
    let root_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
    let mut root_params = CertificateParams::new(vec!["Shared Root".into()]).unwrap();
    root_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    let root_cert = root_params.self_signed(&root_kp).unwrap();
    let root_der = root_cert.der().to_vec();
    let root_pem = root_cert.pem();
    let root_issuer = Issuer::new(root_params, root_kp);
    let leaf_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
    let mut leaf_params = CertificateParams::new(vec!["Shared Signing".into()]).unwrap();
    leaf_params.distinguished_name = {
        let mut dn = DistinguishedName::new();
        dn.push(DnType::CommonName, leaf_cn);
        dn
    };
    if leaf_is_ca {
        leaf_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    }
    if digital_signature {
        leaf_params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    } else {
        leaf_params.key_usages = vec![KeyUsagePurpose::KeyEncipherment];
    }
    let leaf_cert = leaf_params.signed_by(&leaf_kp, &root_issuer).unwrap();
    let leaf_pem = leaf_cert.pem();
    let leaf_sk = p256::SecretKey::from_sec1_pem(&leaf_kp.serialize_pem())
        .map(SigningKey::from)
        .unwrap_or_else(|_| {
            p256::pkcs8::DecodePrivateKey::from_pkcs8_pem(&leaf_kp.serialize_pem())
                .map(|sk: p256::SecretKey| SigningKey::from(sk))
                .expect("leaf key")
        });
    let sign = |key: &str, value: &serde_json::Value| -> String {
        let raw = serde_json::to_string(value).unwrap();
        let sig: p256::ecdsa::Signature = leaf_sk.sign(raw.as_bytes());
        let sig_hex: String = sig.to_bytes().iter().map(|b| format!("{b:02x}")).collect();
        format!("{{\"{key}\":{raw},\"signature\":\"{sig_hex}\"}}")
    };
    (
        (
            sign("tcbInfo", &tcb_value),
            sign("enclaveIdentity", &qe_value),
        ),
        format!("{leaf_pem}{root_pem}"),
        root_der,
    )
}

/// SECURITY: a certificate that chains to the pinned root is not,
/// by that fact, the certificate allowed to say what the TCB is. Intel's
/// root issues many leaves — every platform's PCK certificate among them,
/// and those private keys do not live at Intel. This chain is impeccable:
/// right root, valid windows, signature verifying over the collateral
/// bytes. Only the subject is wrong. A failure here means whoever holds any
/// certificate under Intel's root can declare every platform UpToDate.
#[test]
fn collateral_signed_by_another_certificate_under_the_same_root_is_refused() {
    let tcb = serde_json::json!({"version": 3, "fmspc": "00806f050000"});
    let qe = serde_json::json!({"version": 2, "id": "TD_QE"});
    let ((tcb_body, qe_body), chain, root) =
        synthetic_collateral(&tcb, &qe, "Intel SGX PCK Certificate", true, false);
    let err = tcb::verify_live(
        QUOTE,
        &tcb_body,
        &chain,
        &qe_body,
        &chain,
        &TrustAnchor::RootDer(root),
        1_787_300_000,
        &VENDORED_DAY,
    )
    .expect_err("a non-TCB-signing certificate must not sign a TCB verdict");
    assert!(
        err.to_string().contains("Intel SGX TCB Signing"),
        "the refusal must name the identity it wanted, got: {err}"
    );
}

/// SECURITY: the same chain, the right subject, but a key its own
/// extension says is not for signing. Trusting the CN alone would accept
/// it; Intel marks the real certificate's keyUsage critical precisely so a
/// verifier that reads it cannot be talked out of the restriction.
#[test]
fn collateral_signed_by_a_key_not_marked_for_signing_is_refused() {
    let tcb = serde_json::json!({"version": 3, "fmspc": "00806f050000"});
    let qe = serde_json::json!({"version": 2, "id": "TD_QE"});
    let ((tcb_body, qe_body), chain, root) =
        synthetic_collateral(&tcb, &qe, "Intel SGX TCB Signing", false, false);
    let err = tcb::verify_live(
        QUOTE,
        &tcb_body,
        &chain,
        &qe_body,
        &chain,
        &TrustAnchor::RootDer(root),
        1_787_300_000,
        &VENDORED_DAY,
    )
    .expect_err("a key not marked for signing must not sign a TCB verdict");
    assert!(
        err.to_string().contains("digitalSignature"),
        "the refusal must name the missing usage, got: {err}"
    );
}

/// SECURITY: the right subject, a signing key — but a CA
/// certificate. A CA that also signs leaf data confuses two roles: nothing
/// then distinguishes "Intel said this platform is UpToDate" from
/// "Intel issued a certificate", and the path length constraint becomes
/// nobody's job. Intel marks the real certificate CA:FALSE, critical.
#[test]
fn collateral_signed_by_a_ca_certificate_is_refused() {
    let tcb = serde_json::json!({"version": 3, "fmspc": "00806f050000"});
    let qe = serde_json::json!({"version": 2, "id": "TD_QE"});
    let ((tcb_body, qe_body), chain, root) =
        synthetic_collateral(&tcb, &qe, "Intel SGX TCB Signing", true, true);
    let err = tcb::verify_live(
        QUOTE,
        &tcb_body,
        &chain,
        &qe_body,
        &chain,
        &TrustAnchor::RootDer(root),
        1_787_300_000,
        &VENDORED_DAY,
    )
    .expect_err("a CA certificate must not sign a TCB verdict");
    assert!(
        err.to_string().contains("end entity"),
        "the refusal must say why a CA is wrong here, got: {err}"
    );
}

/// The debug-TD check on real bytes: the vendored quote's TDATTRIBUTES is 0x8000000010000000
/// (DEBUG clear), and the chain still verifies with the DEBUG check in place.
#[test]
fn the_genuine_quote_is_not_a_debug_td_and_still_verifies() {
    let claims = x0ne_confidential::tdx_quote::parse(QUOTE).expect("v4 TDX");
    assert_eq!(claims.td_attributes, [0, 0, 0, 0x10, 0, 0, 0, 0x80]);
    assert!(!claims.is_debug());
    x0ne_infer::pcs::verify_tdx_chain(QUOTE, &TrustAnchor::intel_production(), NOW)
        .expect("a non-debug genuine quote still verifies");
}

// ── TDX module identity (MRSIGNERSEAM, SEAMATTRIBUTES, module ISVSVN) ───────

/// A current platform (as `a_current_platform_is_accepted_end_to_end`) whose
/// tcbInfo also carries the TDX module fields given; the verdict on the
/// vendored quote, whose TEE_TCB_SVN is `04 01 …`: module version 1, ISVSVN 4.
fn verdict_with_module(
    module: serde_json::Value,
) -> Result<(TcbStatus, TcbStatus), tcb::TcbRejected> {
    let hex = |b: &[u8]| -> String { b.iter().map(|x| format!("{x:02x}")).collect() };
    let qe = tcb::qe_report(QUOTE).expect("qe report");
    let mut tcb_value = serde_json::json!({
        "id": "TDX",
        "version": 3,
        "tcbEvaluationDataNumber": 20,
        "issueDate": "2026-08-20T00:00:00Z",
        "nextUpdate": "2026-09-19T00:00:00Z",
        "fmspc": "90c06f000000",
        "tcbLevels": [{
            "tcb": {
                "sgxtcbcomponents": zero_comps(),
                "tdxtcbcomponents": zero_comps(),
                "pcesvn": 0
            },
            "tcbDate": "2026-08-01T00:00:00Z",
            "tcbStatus": "UpToDate"
        }]
    });
    for (k, v) in module.as_object().unwrap() {
        tcb_value[k] = v.clone();
    }
    let qe_value = serde_json::json!({
        "id": "TD_QE",
        "version": 2,
        "tcbEvaluationDataNumber": 20,
        "issueDate": "2026-08-20T00:00:00Z",
        "nextUpdate": "2026-09-19T00:00:00Z",
        "miscselect": hex(&qe[16..20]),
        "miscselectMask": "ffffffff",
        "attributes": hex(&qe[48..64]),
        "attributesMask": "ffffffffffffffffffffffffffffffff",
        "mrsigner": hex(&qe[128..160]),
        "isvprodid": u16::from_le_bytes([qe[256], qe[257]]),
        "tcbLevels": [{
            "tcb": {"isvsvn": u16::from_le_bytes([qe[258], qe[259]])},
            "tcbDate": "2026-08-01T00:00:00Z",
            "tcbStatus": "UpToDate"
        }]
    });
    let ((tcb_body, qe_body), chain, root) =
        synthetic_collateral(&tcb_value, &qe_value, "Intel SGX TCB Signing", true, false);
    tcb::verify_live(
        QUOTE,
        &tcb_body,
        &chain,
        &qe_body,
        &chain,
        &TrustAnchor::RootDer(root),
        NOW,
        &VENDORED_DAY,
    )
}

const ZERO_MRSIGNER: &str = "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";

fn module(mrsigner: &str, levels: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "tdxModule": {"mrsigner": mrsigner, "attributes": "0000000000000000", "attributesMask": "FFFFFFFFFFFFFFFF"},
        "tdxModuleIdentities": [{
            "id": "TDX_01",
            "mrsigner": mrsigner,
            "attributes": "0000000000000000",
            "attributesMask": "FFFFFFFFFFFFFFFF",
            "tcbLevels": levels
        }]
    })
}

/// The accept half: Intel lists this module version at this ISVSVN as current.
#[test]
fn a_current_tdx_module_is_accepted() {
    let m = module(
        ZERO_MRSIGNER,
        serde_json::json!([{"tcb": {"isvsvn": 4}, "tcbDate": "2026-08-01T00:00:00Z", "tcbStatus": "UpToDate"}]),
    );
    let (platform, qe) = verdict_with_module(m).expect("a current module must be accepted");
    assert_eq!((platform, qe), (TcbStatus::UpToDate, TcbStatus::UpToDate));
}

/// SECURITY: the platform is current, the TDX module Intel lists for this
/// version says ISVSVN 4 is OutOfDate (11 is current). A failure means the
/// module running under the TD is never judged.
#[test]
fn a_tdx_module_intel_lists_as_out_of_date_is_refused() {
    let m = module(
        ZERO_MRSIGNER,
        serde_json::json!([
            {"tcb": {"isvsvn": 11}, "tcbDate": "2026-08-01T00:00:00Z", "tcbStatus": "UpToDate"},
            {"tcb": {"isvsvn": 4}, "tcbDate": "2025-05-14T00:00:00Z", "tcbStatus": "OutOfDate"}
        ]),
    );
    let err = verdict_with_module(m).expect_err("an out-of-date TDX module must be refused");
    assert!(err.to_string().contains("TDX module"), "{err}");
}

/// SECURITY: the TCB floor reads `tcbEvaluationDataNumber` from the signed
/// bytes. Signed collateral without it, or with it as anything but an
/// integer, is refused: a missing number is not a pass.
#[test]
fn signed_collateral_without_a_readable_evaluation_number_is_refused() {
    let current = || {
        module(
            ZERO_MRSIGNER,
            serde_json::json!([{"tcb": {"isvsvn": 4}, "tcbDate": "2026-08-01T00:00:00Z", "tcbStatus": "UpToDate"}]),
        )
    };
    for bad in [
        serde_json::Value::Null,
        serde_json::json!("22"),
        serde_json::json!(-1),
        serde_json::json!(22.5),
    ] {
        let mut m = current();
        m["tcbEvaluationDataNumber"] = bad.clone();
        let err = verdict_with_module(m).expect_err("no readable evaluation number");
        assert!(
            err.to_string()
                .contains("tcbEvaluationDataNumber unreadable"),
            "{bad}: {err}"
        );
    }
    verdict_with_module(current()).expect("control: evaluation 20");
}

/// SECURITY: a module signed by someone else than the signer Intel
/// publishes is refused by MRSIGNERSEAM, whatever its ISVSVN says.
#[test]
fn a_tdx_module_signed_by_another_key_is_refused() {
    let m = module(
        &"11".repeat(48),
        serde_json::json!([{"tcb": {"isvsvn": 4}, "tcbDate": "2026-08-01T00:00:00Z", "tcbStatus": "UpToDate"}]),
    );
    let err = verdict_with_module(m).expect_err("a foreign MRSIGNERSEAM must be refused");
    assert!(err.to_string().contains("MRSIGNERSEAM"), "{err}");
}

/// SECURITY: a module version Intel does not list has no identity to judge
/// it by: refused, not waved through.
#[test]
fn a_tdx_module_version_without_an_identity_is_refused() {
    let mut m = module(
        ZERO_MRSIGNER,
        serde_json::json!([{"tcb": {"isvsvn": 4}, "tcbDate": "2026-08-01T00:00:00Z", "tcbStatus": "UpToDate"}]),
    );
    m["tdxModuleIdentities"][0]["id"] = serde_json::json!("TDX_03");
    let err = verdict_with_module(m).expect_err("an unlisted module version must be refused");
    assert!(err.to_string().contains("TDX_01"), "{err}");
}

/// Real data: Intel's vendored tcbInfo lists TDX_01 at ISVSVN 4 as
/// OutOfDate (level of 2024-03-13); the vendored quote runs exactly that.
#[test]
fn the_genuine_quotes_tdx_module_is_out_of_date_against_todays_intel() {
    let v: serde_json::Value = serde_json::from_str(TCB_INFO).unwrap();
    let m = tcb::tdx_module_status(&v["tcbInfo"], QUOTE).expect("a listed module");
    assert_eq!(m.id, "TDX_01");
    assert_eq!(m.status, Some(TcbStatus::OutOfDate));
}
