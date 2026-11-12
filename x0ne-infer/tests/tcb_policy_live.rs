//! The TCB policy (`tcb::TcbPolicy`) on real bytes: the TD Quote of a GCE
//! a3-highgpu-1g boot of 2026-10-18 (`run_fixtures/`), and Intel PCS
//! production collateral fetched on 2026-10-22 at 10:55 UTC
//! (`pcs_live/2026-10-22/`, `VENDORED.md`).
//!
//! That day Intel's `standard` feed (no `update`, what PCS answers by
//! default) was TCB evaluation 20, the recovery of 2025-08-12; `early` was
//! evaluation 22, the recovery of 2026-08-11 (INTEL-SA-01436). Evaluated at
//! a fixed `NOW`, so the tests do not rot with the wall clock.

use x0ne_infer::pcs::TrustAnchor;
use x0ne_infer::tcb::{self, TcbPolicy, TcbRejected, TcbStatus};

const GCE_QUOTE: &[u8] = include_bytes!("run_fixtures/gce-a3-tdx-2026-10-18.quote");
const OLD_SAMPLE_QUOTE: &[u8] = include_bytes!("pcs_live/quote.dat");
const TCB_EARLY: &str = include_str!("pcs_live/2026-10-22/tcb_info_00806f050000_early.json");
const TCB_STANDARD: &str = include_str!("pcs_live/2026-10-22/tcb_info_00806f050000_standard.json");
const TCB_OLD_SAMPLE_EARLY: &str =
    include_str!("pcs_live/2026-10-22/tcb_info_90c06f000000_early.json");
const QE_EARLY: &str = include_str!("pcs_live/2026-10-22/qe_identity_early.json");
const QE_STANDARD: &str = include_str!("pcs_live/2026-10-22/qe_identity_standard.json");
/// Intel's TCB signing chain: the same bytes as on 2026-08-20.
const TCB_CHAIN: &str = include_str!("pcs_live/tcb_issuer_chain.pem");
const QE_CHAIN: &str = include_str!("pcs_live/qe_issuer_chain.pem");
const CRL: &[u8] = include_bytes!("pcs_live/2026-10-22/pck_crl_platform.der");

/// 2026-10-22T10:55:40Z, the `Date` of the last of the six answers.
const NOW: u64 = 1_792_666_540;
const PROD: TcbPolicy = TcbPolicy::production();

fn live(
    quote: &[u8],
    tcb: &str,
    qe: &str,
    now: u64,
    policy: &TcbPolicy,
) -> Result<(TcbStatus, TcbStatus), TcbRejected> {
    tcb::verify_live(
        quote,
        tcb,
        TCB_CHAIN,
        qe,
        QE_CHAIN,
        &TrustAnchor::intel_production(),
        now,
        policy,
    )
}

/// The platform the quote names, read out of band with openssl on 2026-10-22.
#[test]
fn the_gce_quote_names_a_sapphire_rapids_platform_with_tdx_module_1_svn_15() {
    let pck = tcb::pck_tcb(GCE_QUOTE).expect("pck tcb");
    assert_eq!(pck.fmspc_hex, "00806f050000");
    assert_eq!(pck.pcesvn, 13);
    assert_eq!(
        pck.cpusvn,
        [10, 10, 2, 2, 6, 1, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        tcb::tee_tcb_svn(GCE_QUOTE).unwrap()[..3],
        [15, 1, 10],
        "TEE_TCB_SVN: module SVN 15, module version 1 (TDX_01), late microcode 10"
    );
}

/// The accept path on real data: a GCE host of 2026-10-18 is UpToDate under
/// the recovery of 2026-08-11, platform and QE, with the production policy.
#[test]
fn the_gce_host_passes_the_production_policy_on_intels_latest_recovery() {
    x0ne_infer::pcs::verify_tdx_chain(GCE_QUOTE, &TrustAnchor::intel_production(), NOW)
        .expect("the chain is genuine");
    let (platform, qe) =
        live(GCE_QUOTE, TCB_EARLY, QE_EARLY, NOW, &PROD).expect("evaluation 22, fresh");
    assert_eq!((platform, qe), (TcbStatus::UpToDate, TcbStatus::UpToDate));
    let module = tcb::tdx_module_status(
        &serde_json::from_str::<serde_json::Value>(TCB_EARLY).unwrap()["tcbInfo"],
        GCE_QUOTE,
    )
    .unwrap();
    assert_eq!(module.id, "TDX_01");
    assert_eq!(module.status, Some(TcbStatus::UpToDate));
}

/// A QE identity rewritten to name another quoting enclave is refused at its
/// signature, on the very quote and collateral that pass when it is intact:
/// the identity is never read before Intel's signature over its raw bytes
/// holds (CVE-2026-22696, in another verifier, read it unsigned).
#[test]
fn a_qe_identity_rewritten_to_name_another_enclave_is_refused_at_its_signature() {
    live(GCE_QUOTE, TCB_EARLY, QE_EARLY, NOW, &PROD).expect("control: intact, accepted");
    for field in ["\"mrsigner\":\"", "\"isvprodid\":", "\"tcbStatus\":\""] {
        let at = QE_EARLY.find(field).unwrap_or_else(|| panic!("{field}")) + field.len();
        let mut forged = QE_EARLY.to_string();
        let other = if &forged[at..=at] == "1" { "2" } else { "1" };
        forged.replace_range(at..=at, other);
        assert_ne!(forged, QE_EARLY);
        let e = live(GCE_QUOTE, TCB_EARLY, &forged, NOW, &PROD)
            .expect_err("a forged QE identity must refuse");
        assert!(e.to_string().contains("signature"), "{field}: {e}");
    }
}

/// SECURITY (TOB-WAPI-8): Intel's default answer is evaluation 20, under
/// which a module two recoveries behind still reads UpToDate. Below the
/// floor it is refused, real signature and all; only a policy lowered to 20
/// takes it.
#[test]
fn intels_default_standard_collateral_is_refused_below_the_minimum_evaluation() {
    let e = live(GCE_QUOTE, TCB_STANDARD, QE_EARLY, NOW, &PROD).unwrap_err();
    assert!(
        matches!(
            e,
            TcbRejected::EvaluationBelowMinimum {
                what: "tcbInfo",
                have: 20,
                min: 22
            }
        ),
        "{e}"
    );
    let e = live(GCE_QUOTE, TCB_EARLY, QE_STANDARD, NOW, &PROD).unwrap_err();
    assert!(
        matches!(
            e,
            TcbRejected::EvaluationBelowMinimum {
                what: "qeIdentity",
                have: 20,
                min: 22
            }
        ),
        "{e}"
    );
    let lowered = TcbPolicy {
        min_tcb_evaluation_data_number: 20,
        ..PROD
    };
    live(GCE_QUOTE, TCB_STANDARD, QE_STANDARD, NOW, &lowered).expect("the same bytes at 20");
}

/// Under evaluation 22 Intel rates TDX_01 ISVSVN 11 OutOfDate, naming
/// INTEL-SA-01436; under evaluation 20 the same ISVSVN is UpToDate.
#[test]
fn evaluation_22_names_sa_01436_against_the_module_evaluation_20_accepts() {
    let level = |body: &str, svn: u64| {
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        let ids = v["tcbInfo"]["tdxModuleIdentities"]
            .as_array()
            .unwrap()
            .clone();
        let m = ids.into_iter().find(|m| m["id"] == "TDX_01").unwrap();
        m["tcbLevels"]
            .as_array()
            .unwrap()
            .iter()
            .find(|l| svn >= l["tcb"]["isvsvn"].as_u64().unwrap())
            .cloned()
            .unwrap()
    };
    assert_eq!(level(TCB_STANDARD, 11)["tcbStatus"], "UpToDate");
    let early = level(TCB_EARLY, 11);
    assert_eq!(early["tcbStatus"], "OutOfDate");
    assert!(
        early["advisoryIDs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a == "INTEL-SA-01436"),
        "{early}"
    );
}

/// Under evaluation 22, for this FMSPC, every level of both TDX module
/// identities but the newest is OutOfDate and names INTEL-SA-01436 (TDX
/// module 1.5.28 and earlier): the recovery the floor names covers that
/// advisory by Intel's own data, not by a version comparison here. One
/// FMSPC, one day (2026-10-22).
#[test]
fn under_evaluation_22_every_module_level_but_the_newest_names_sa_01436() {
    let v: serde_json::Value = serde_json::from_str(TCB_EARLY).unwrap();
    let ids = v["tcbInfo"]["tdxModuleIdentities"].as_array().unwrap();
    let mut seen = Vec::new();
    for m in ids {
        let levels = m["tcbLevels"].as_array().unwrap();
        let (newest, older) = levels.split_first().unwrap();
        assert_eq!(newest["tcbStatus"], "UpToDate", "{}: {newest}", m["id"]);
        assert!(!older.is_empty(), "{}", m["id"]);
        for level in older {
            assert_eq!(level["tcbStatus"], "OutOfDate", "{}: {level}", m["id"]);
            assert!(
                level["advisoryIDs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|a| a == "INTEL-SA-01436"),
                "{}: {level}",
                m["id"]
            );
        }
        seen.push((
            m["id"].as_str().unwrap().to_string(),
            newest["tcb"]["isvsvn"].as_u64().unwrap(),
        ));
    }
    assert_eq!(
        seen,
        [("TDX_03".to_string(), 9), ("TDX_01".to_string(), 15)]
    );
}

/// SECURITY: a platform behind the latest recovery is refused on real data
/// under the production policy: the public sample of `tcb_live.rs`. Its
/// CPUSVN component 7 is 3 where every level Intel publishes for its FMSPC
/// under evaluation 22 (five levels, read 2026-10-22) asks 5: it matches no
/// level at all, not even the OutOfDate one of 2018, and is refused as
/// `BelowEveryLevel`, not as `StatusRefused { OutOfDate }`; so it was under
/// Intel's collateral of 2026-08-20 (`tcb_live.rs`).
#[test]
fn a_platform_below_every_level_of_intels_latest_recovery_is_refused() {
    let pck = tcb::pck_tcb(OLD_SAMPLE_QUOTE).expect("pck tcb");
    assert_eq!(pck.fmspc_hex, "90c06f000000");
    assert_eq!(pck.cpusvn[7], 3);
    let info: serde_json::Value = serde_json::from_str(TCB_OLD_SAMPLE_EARLY).unwrap();
    let levels = info["tcbInfo"]["tcbLevels"].as_array().unwrap();
    assert_eq!(levels.len(), 5);
    for level in levels {
        assert_eq!(
            level["tcb"]["sgxtcbcomponents"][7]["svn"], 5,
            "every level asks more than the sample's 3: {level}"
        );
    }
    let e = live(OLD_SAMPLE_QUOTE, TCB_OLD_SAMPLE_EARLY, QE_EARLY, NOW, &PROD).unwrap_err();
    assert!(matches!(e, TcbRejected::BelowEveryLevel), "{e}");
}

/// SECURITY: the same collateral two days on is inside its 30-day window
/// and refused for its age; past its `nextUpdate`, refused as expired.
#[test]
fn todays_collateral_is_refused_once_older_than_a_day_and_once_expired() {
    let later = NOW + 2 * 86_400;
    let e = live(GCE_QUOTE, TCB_EARLY, QE_EARLY, later, &PROD).unwrap_err();
    assert!(
        matches!(
            e,
            TcbRejected::CollateralTooOld {
                what: "tcbInfo",
                max_secs: 86_400,
                ..
            }
        ),
        "{e}"
    );
    let loose = TcbPolicy {
        max_collateral_age_secs: u64::MAX,
        ..PROD
    };
    live(GCE_QUOTE, TCB_EARLY, QE_EARLY, later, &loose).expect("inside the window");
    // nextUpdate of the tcbInfo: 2026-11-21T10:32:14Z.
    let expired = 1_795_257_134 + 1;
    let e = live(GCE_QUOTE, TCB_EARLY, QE_EARLY, expired, &loose).unwrap_err();
    assert!(e.to_string().contains("validity window"), "{e}");
}

/// SECURITY: today's PCK CRL passes at `NOW`; two days on it is refused for
/// its age, and past its `nextUpdate` (2026-11-21T10:37:17Z) as expired.
#[test]
fn todays_pck_crl_is_refused_once_older_than_a_day_and_once_expired() {
    let anchor = TrustAnchor::intel_production();
    tcb::verify_pck_crl(GCE_QUOTE, CRL, &anchor, NOW, &PROD).expect("today");
    let e = tcb::verify_pck_crl(GCE_QUOTE, CRL, &anchor, NOW + 2 * 86_400, &PROD).unwrap_err();
    assert!(
        matches!(
            e,
            TcbRejected::CollateralTooOld {
                what: "PCK CRL",
                ..
            }
        ),
        "{e}"
    );
    let loose = TcbPolicy {
        max_collateral_age_secs: u64::MAX,
        ..PROD
    };
    let next_update = 1_795_257_437;
    tcb::verify_pck_crl(GCE_QUOTE, CRL, &anchor, next_update, &loose).expect("last second");
    let e = tcb::verify_pck_crl(GCE_QUOTE, CRL, &anchor, next_update + 1, &loose).unwrap_err();
    assert!(
        matches!(&e, TcbRejected::PckCrl(m) if m.contains("validity window")),
        "{e}"
    );
}
