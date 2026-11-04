//! The two vendored Google files are the ones `tests/VENDORED.md` names, and
//! the real endorsement verifies under the real root with the production
//! code path. A failure means a file changed under the document, or the
//! verifier no longer accepts what Google publishes.

use sha2::{Digest, Sha256};
use x0ne_infer::gce_endorsement;

const ROOT_SHA256: &str = "e876bc6978bf4f3da445f98a0a82363c8c0bae5a1fc033c6df65846a6cb0f18c";
const BLOB_SHA256: &str = "addd7155dbfe2d68192e8467630a4a753e90a31501f0d8c753cfee444bbd94cb";
const MRTD_HEX: &str = "85aefc5d443857c9df7fdcb2a52819268aaa5f094a40601fad6d525d18651c005fe23f4efb9c10d02e3fa27edf147b7b";
/// Inside the signer's window (2026-08-04 to 2031-08-04).
const AT: u64 = 1_789_000_000;

fn root() -> Vec<u8> {
    std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/gce_tcb_root_1.der"
    ))
    .unwrap()
}

fn blob() -> Vec<u8> {
    std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/gce_endorsement_tdx_85aefc5d.binarypb"
    ))
    .unwrap()
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn the_vendored_files_are_the_ones_the_document_names() {
    assert_eq!(sha256_hex(&root()), ROOT_SHA256);
    assert_eq!(sha256_hex(&blob()), BLOB_SHA256);
    assert_eq!(
        gce_endorsement::ROOT_1_DER,
        root().as_slice(),
        "the compiled-in root is the vendored file"
    );
    let doc =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/VENDORED.md")).unwrap();
    for needle in [
        ROOT_SHA256,
        BLOB_SHA256,
        MRTD_HEX,
        "pki.goog/cloud_integrity/GCE-cc-tcb-root_1.crt",
    ] {
        assert!(doc.contains(needle), "VENDORED.md must name {needle}");
    }
}

#[test]
fn the_root_is_googles_cc_tcb_root_1() {
    use x509_parser::prelude::{FromDer, X509Certificate};
    let der = root();
    let (_, cert) = X509Certificate::from_der(&der).unwrap();
    let subject = cert.subject().to_string();
    assert!(subject.contains("CN=GCE-cc-tcb-root"), "{subject}");
    assert!(subject.contains("O=Google"), "{subject}");
    assert!(cert.is_ca());
    assert_eq!(
        cert.validity().not_before.timestamp(),
        1_697_493_265,
        "2023-10-16T21:54:25Z"
    );
    assert_eq!(
        cert.validity().not_after.timestamp(),
        2_486_411_665,
        "2048-10-15T21:54:25Z"
    );
}

#[test]
fn the_real_endorsement_verifies_under_the_real_root() {
    let e = gce_endorsement::verify(&blob(), &root(), AT).expect("verifies");
    assert_eq!(e.tdx_svn, 2);
    assert!(
        e.signer_subject.contains("CN=GCE-uefi-signer"),
        "{}",
        e.signer_subject
    );
    let mrtd: [u8; 48] = (0..48)
        .map(|i| u8::from_str_radix(&MRTD_HEX[2 * i..2 * i + 2], 16).unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let m = e
        .tdx_measurement(&mrtd)
        .expect("the named MRTD is endorsed");
    assert_eq!(m.ram_gib, 0);
    assert_eq!(
        gce_endorsement::url(gce_endorsement::BUCKET_BASE_URL, &mrtd),
        format!(
            "https://storage.googleapis.com/gce_tcb_integrity/ovmf_x64_csm/tdx/{MRTD_HEX}.binarypb"
        )
    );
}

/// SECURITY: one changed byte in the signature refuses; the Intel root does
/// not stand in for Google's; outside the signer's window refuses.
#[test]
fn the_endorsement_is_refused_when_tampered_misrooted_or_out_of_window() {
    let mut tampered = blob();
    let last = tampered.len() - 1;
    tampered[last] ^= 1;
    assert!(gce_endorsement::verify(&tampered, &root(), AT).is_err());
    let intel = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/intel_sgx_root_ca.der"
    ))
    .unwrap();
    assert!(gce_endorsement::verify(&blob(), &intel, AT).is_err());
    let err = gce_endorsement::verify(&blob(), &root(), 1_700_000_000).unwrap_err();
    assert!(err.contains("validity window"), "{err}");
}
