//! The PCK CRL check against REAL data: Intel's PCK Platform CA CRL as PCS
//! served it on 2026-10-18 at 19:08:58 UTC (tests/VENDORED.md), and the quote
//! of the GCE a3-highgpu-1g run of the same day (tests/run_fixtures/).

use x0ne_infer::pcs::TrustAnchor;
use x0ne_infer::tcb::{self, TcbPolicy, TcbRejected};
use x509_parser::prelude::FromDer;
use x509_parser::revocation_list::CertificateRevocationList;

const CRL: &[u8] = include_bytes!("pcs_live/pck_crl_platform_2026-10-18.der");
const QUOTE: &[u8] = include_bytes!("run_fixtures/gce-a3-tdx-2026-10-18.quote");
/// PCS's Date header on the GET; thisUpdate and nextUpdate of the CRL.
const FETCHED: u64 = 1_792_350_538; // 2026-10-18T19:08:58Z
const THIS_UPDATE: u64 = 1_792_348_462; // 2026-10-18T18:34:22Z
const NEXT_UPDATE: u64 = 1_794_940_462; // 2026-11-17T18:34:22Z
/// The PCK serial of the 18/10 quote (openssl x509 -serial).
const RUN_PCK_SERIAL: &str = "862c5fffb445d164c5785b90acd0c9b88e5694bc";
/// The CRL judged by its own window only; the age limit has its own test.
const WINDOW_ONLY: TcbPolicy = TcbPolicy {
    max_collateral_age_secs: u64::MAX,
    ..TcbPolicy::production()
};

/// Not vacuous: the CRL lists real revoked PCKs, and the run's is not one.
#[test]
fn intels_crl_lists_57_pcks_and_not_the_one_of_the_18_10_run() {
    let (_, crl) = CertificateRevocationList::from_der(CRL).expect("Intel's CRL parses");
    let serials: Vec<String> = crl
        .iter_revoked_certificates()
        .map(|r| format!("{:x}", r.serial()))
        .collect();
    assert_eq!(serials.len(), 57);
    assert!(!serials.iter().any(|s| s == RUN_PCK_SERIAL));
    assert_eq!(crl.last_update().timestamp() as u64, THIS_UPDATE);
    assert_eq!(crl.next_update().unwrap().timestamp() as u64, NEXT_UPDATE);
}

#[test]
fn the_pck_of_the_18_10_run_passes_intels_crl_of_that_day() {
    assert_eq!(tcb::pck_crl_ca(QUOTE).unwrap(), "platform");
    let anchor = TrustAnchor::intel_production();
    for now in [FETCHED, THIS_UPDATE, NEXT_UPDATE] {
        tcb::verify_pck_crl(QUOTE, CRL, &anchor, now, &WINDOW_ONLY)
            .unwrap_or_else(|e| panic!("now={now}: {e}"));
    }
}

/// The freshness rule: the CRL counts only inside its own window.
#[test]
fn the_same_crl_is_refused_outside_its_window() {
    let anchor = TrustAnchor::intel_production();
    for now in [THIS_UPDATE - 1, NEXT_UPDATE + 1] {
        let e = tcb::verify_pck_crl(QUOTE, CRL, &anchor, now, &WINDOW_ONLY).unwrap_err();
        assert!(
            matches!(&e, TcbRejected::PckCrl(m) if m.contains("validity window")),
            "now={now}: {e}"
        );
    }
}

/// SECURITY: inside its window, the CRL is still refused once issued more
/// than a day before `now`: at the GET it passes, at its `nextUpdate` not.
#[test]
fn the_same_crl_is_refused_once_older_than_the_production_maximum() {
    let anchor = TrustAnchor::intel_production();
    let prod = TcbPolicy::production();
    tcb::verify_pck_crl(QUOTE, CRL, &anchor, FETCHED, &prod).expect("35 minutes old");
    let e = tcb::verify_pck_crl(QUOTE, CRL, &anchor, NEXT_UPDATE, &prod).unwrap_err();
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
}

/// One flipped bit anywhere in the signed list or its signature refuses.
#[test]
fn a_tampered_crl_is_refused() {
    let anchor = TrustAnchor::intel_production();
    for at in [CRL.len() - 3, 200] {
        let mut crl = CRL.to_vec();
        crl[at] ^= 0x01;
        let e = tcb::verify_pck_crl(QUOTE, &crl, &anchor, FETCHED, &WINDOW_ONLY).unwrap_err();
        assert!(matches!(e, TcbRejected::PckCrl(_)), "byte {at}: {e}");
    }
}
