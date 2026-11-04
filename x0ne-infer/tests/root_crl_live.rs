//! Intel's root CA CRL on REAL bytes: the CRL the Intel SGX Root CA publishes
//! at its own CRL distribution point, fetched on 2026-10-23 at 06:11:57 UTC
//! (`pcs_live/2026-10-23/root_ca_crl.der`, `VENDORED.md`), the quote of the
//! GCE a3-highgpu-1g run of 2026-10-18 (`run_fixtures/`) and Intel's TCB
//! signing chain (`pcs_live/*_issuer_chain.pem`).
//!
//! That CRL is number 1, thisUpdate 2026-02-26, nextUpdate 2027-02-26, no
//! revoked certificate: Intel issues it for a year, so the verifier judges
//! it by its own window and not by the collateral age limit.

use x0ne_infer::pcs::TrustAnchor;
use x0ne_infer::tcb::{self, TcbRejected};
use x509_parser::extensions::{DistributionPointName, GeneralName, ParsedExtension};
use x509_parser::pem::Pem;
use x509_parser::prelude::{FromDer, X509Certificate};
use x509_parser::revocation_list::CertificateRevocationList;

const ROOT_CRL: &[u8] = include_bytes!("pcs_live/2026-10-23/root_ca_crl.der");
const QUOTE: &[u8] = include_bytes!("run_fixtures/gce-a3-tdx-2026-10-18.quote");
const TCB_CHAIN: &str = include_str!("pcs_live/tcb_issuer_chain.pem");
const QE_CHAIN: &str = include_str!("pcs_live/qe_issuer_chain.pem");
/// The `Date` of the GET; thisUpdate and nextUpdate of the CRL.
const FETCHED: u64 = 1_792_735_917; // 2026-10-23T06:11:57Z
const THIS_UPDATE: u64 = 1_772_111_040; // 2026-02-26T13:04:00Z
const NEXT_UPDATE: u64 = 1_803_647_040; // 2027-02-26T13:04:00Z
/// notBefore of the 18/10 host's PCK (read from its chain by `pcs`): the
/// earliest instant at which this quote verifies at all. It is five and a
/// half months after the CRL's thisUpdate, so the real data cannot exercise
/// "root CRL not yet valid"; the synthetic test in `tcb.rs` does.
const PCK_NOT_BEFORE: u64 = 1_786_625_234; // 2026-08-13T12:47:14Z
/// Serials read with openssl on 2026-10-23: the root (also the CRL's
/// authority key identifier) and the TCB signing certificate, the same in
/// both collateral chains.
const ROOT_SERIAL: &str = "22650cd65a9d3489f383b49552bf501b392706ac";
const TCB_SIGNING_SERIAL: &str = "7e3882d5fb55294a40498e458403e91491bdf455";

fn nth_cert(chain_pem: &str, n: usize) -> Vec<u8> {
    Pem::iter_from_buffer(chain_pem.as_bytes())
        .nth(n)
        .expect("a certificate at that index")
        .expect("PEM")
        .contents
}

/// Not vacuous: the bytes are Intel's root CRL of that day, number 1, empty.
#[test]
fn intels_root_crl_is_number_1_valid_a_year_and_lists_nothing() {
    let (rest, crl) = CertificateRevocationList::from_der(ROOT_CRL).expect("Intel's CRL parses");
    assert!(rest.is_empty());
    assert_eq!(crl.last_update().timestamp() as u64, THIS_UPDATE);
    assert_eq!(crl.next_update().unwrap().timestamp() as u64, NEXT_UPDATE);
    assert_eq!(crl.iter_revoked_certificates().count(), 0);
    assert_eq!(
        crl.crl_number().map(|n| n.to_string()).as_deref(),
        Some("1")
    );
    assert_eq!(
        crl.issuer().to_string(),
        "CN=Intel SGX Root CA, O=Intel Corporation, L=Santa Clara, ST=CA, C=US"
    );
    let root = nth_cert(TCB_CHAIN, 1);
    let (_, root) = X509Certificate::from_der(&root).unwrap();
    assert_eq!(format!("{:x}", root.tbs_certificate.serial), ROOT_SERIAL);
    let tcb = nth_cert(TCB_CHAIN, 0);
    let (_, tcb) = X509Certificate::from_der(&tcb).unwrap();
    assert_eq!(
        format!("{:x}", tcb.tbs_certificate.serial),
        TCB_SIGNING_SERIAL
    );
    assert_eq!(nth_cert(QE_CHAIN, 0), nth_cert(TCB_CHAIN, 0));
}

/// The accept path on real data, at the GET, at the first second the PCK
/// exists and at the CRL's last: the 18/10 host's PCK CA and Intel's TCB
/// signer are not revoked.
#[test]
fn the_18_10_hosts_pck_ca_and_intels_tcb_signer_pass_the_root_crl() {
    let anchor = TrustAnchor::intel_production();
    assert!(THIS_UPDATE < PCK_NOT_BEFORE && PCK_NOT_BEFORE < FETCHED);
    for now in [FETCHED, PCK_NOT_BEFORE, NEXT_UPDATE] {
        tcb::verify_root_crl(QUOTE, &[TCB_CHAIN, QE_CHAIN], ROOT_CRL, &anchor, now)
            .unwrap_or_else(|e| panic!("now={now}: {e}"));
    }
}

/// The root CRL counts only inside its own window; inside it, its age is
/// not judged: at the GET it was 239 days old and is the current one. A
/// second past nextUpdate, the same bytes refuse, and the refusal is the
/// CRL's (the PCK is valid until 2033). Before thisUpdate the PCK chain
/// refuses first, so that side is left to the synthetic test.
#[test]
fn the_same_root_crl_is_refused_past_its_next_update() {
    let anchor = TrustAnchor::intel_production();
    let e = tcb::verify_root_crl(
        QUOTE,
        &[TCB_CHAIN, QE_CHAIN],
        ROOT_CRL,
        &anchor,
        NEXT_UPDATE + 1,
    )
    .unwrap_err();
    assert!(
        matches!(&e, TcbRejected::RootCrl(m) if m.contains("does not contain now")),
        "{e}"
    );
    let e = tcb::verify_root_crl(
        QUOTE,
        &[TCB_CHAIN, QE_CHAIN],
        ROOT_CRL,
        &anchor,
        THIS_UPDATE - 1,
    )
    .unwrap_err();
    assert!(
        matches!(&e, TcbRejected::RootCrl(m) if m.contains("PCK chain")),
        "{e}"
    );
    assert!(FETCHED - THIS_UPDATE > 230 * 86_400);
}

/// One flipped bit anywhere in the signed list or its signature refuses.
#[test]
fn a_tampered_root_crl_is_refused() {
    let anchor = TrustAnchor::intel_production();
    for at in [ROOT_CRL.len() - 3, 120] {
        let mut crl = ROOT_CRL.to_vec();
        crl[at] ^= 0x01;
        let e = tcb::verify_root_crl(QUOTE, &[TCB_CHAIN, QE_CHAIN], &crl, &anchor, FETCHED)
            .unwrap_err();
        assert!(matches!(e, TcbRejected::RootCrl(_)), "byte {at}: {e}");
    }
}

/// The production URL is the root certificate's own CRL distribution point,
/// read from the certificate, not recalled.
#[test]
fn the_root_crl_url_is_the_pinned_roots_distribution_point() {
    let root = nth_cert(TCB_CHAIN, 1);
    let (_, root) = X509Certificate::from_der(&root).unwrap();
    let points = root
        .extensions()
        .iter()
        .find_map(|x| match x.parsed_extension() {
            ParsedExtension::CRLDistributionPoints(p) => Some(p),
            _ => None,
        })
        .expect("the root names its CRL distribution point");
    let uris: Vec<&str> = points
        .points
        .iter()
        .filter_map(|p| match &p.distribution_point {
            Some(DistributionPointName::FullName(names)) => Some(names),
            _ => None,
        })
        .flatten()
        .filter_map(|n| match n {
            GeneralName::URI(u) => Some(*u),
            _ => None,
        })
        .collect();
    assert_eq!(uris, vec![tcb::INTEL_SGX_ROOT_CA_CRL_URL]);
}
