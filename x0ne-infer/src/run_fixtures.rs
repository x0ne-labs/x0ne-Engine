//! The fixtures of one real GCE boot, rebuilt from the attestation document
//! the relay serves for the run (`GET /x0ne/v1/attestation`) and the image's v2
//! reference: `measurements/<version>.json` and, under `x0ne-infer/tests/`,
//! `ccel/gce-<family>-tdx-<date>.bin`, `ccel/tdreport-gce-<family>-tdx-<date>.bin`
//! and `reference/x0ne-cvm-<image version>.json`. Nothing is written unless the
//! relay's report in the document says VERIFIED with this very reference, the
//! quote verifies to the pinned Intel root, the log replays to the quote, the
//! reference passes that log (`reference::check_image_rtmr0_reported`, the
//! lab's replay: the run's reference reported RTMR0) and the release archive
//! key signed it; `test_key` accepts a TEST-key run and says so in the notes.
//! `make run-fixtures` (ci/ci.sh), `examples/run_fixtures.rs`.

use crate::pcs::TrustAnchor;
use crate::{ccel, measurements, reference};
use base64::Engine as _;
use sha2::{Digest, Sha256};

/// 0xFF bytes kept after the last event: the cut of the 2026-10-14 and
/// 2026-10-17 logs (`ccel.rs`, `reference.rs` tests).
pub const PADDING_KEPT: usize = 256;
/// The TD report body of a v4 TDX quote.
pub const TD_REPORT: std::ops::Range<usize> = 48..632;

pub struct Inputs<'a> {
    /// The JSON the relay serves: `document.tdx.{quote_b64,ccel_b64}`, `report`.
    pub attestation: &'a [u8],
    /// The image's reference, schema v2, as published with the image.
    pub reference: &'a [u8],
    /// `<reference>.sig`, its detached signature; empty when there is none.
    pub signature: &'a [u8],
    /// The run's date, `YYYY-MM-DD`: it names the log and the TD report.
    pub date: &'a str,
    /// `image.name` of the measurements file.
    pub image: &'a str,
    pub machine: &'a str,
    /// The root the quote's PCK chain must end on: Intel's, outside tests.
    pub tdx: &'a TrustAnchor,
    /// The clock the quote's certificates and the signature are judged at.
    pub now_unix: u64,
    /// Accept a reference the release archive key did not sign (TEST key).
    pub test_key: bool,
}

#[derive(Debug)]
pub struct Fixtures {
    /// Path from the repository root, bytes.
    pub files: Vec<(String, Vec<u8>)>,
    pub notes: Vec<String>,
}

fn b64(doc: &serde_json::Value, field: &str) -> Result<Vec<u8>, String> {
    let s = doc["document"]["tdx"][field]
        .as_str()
        .ok_or_else(|| format!("no document.tdx.{field}"))?;
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| format!("document.tdx.{field}: {e}"))
}

fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

pub fn build(i: &Inputs) -> Result<Fixtures, String> {
    let d = i.date.as_bytes();
    if d.len() != 10
        || d.iter().enumerate().any(|(n, &c)| {
            if n == 4 || n == 7 {
                c != b'-'
            } else {
                !c.is_ascii_digit()
            }
        })
    {
        return Err(format!("date {:?}: expected YYYY-MM-DD", i.date));
    }
    let family = i.machine.split('-').next().unwrap_or("");
    if family.is_empty()
        || !family
            .bytes()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
    {
        return Err(format!("machine type {:?}", i.machine));
    }
    let doc: serde_json::Value =
        serde_json::from_slice(i.attestation).map_err(|e| format!("attestation: {e}"))?;
    if doc["report"]["verdict"].get("Verified").is_none() {
        return Err("the relay's report in this document is not VERIFIED".into());
    }
    let used = doc["report"]["reference_sha256_hex"]
        .as_str()
        .ok_or("the relay's report names no reference (reference_sha256_hex)")?;
    if used != sha256_hex(i.reference) {
        return Err(format!(
            "the run's relay took another reference: sha256 {used}, this one is {}",
            sha256_hex(i.reference)
        ));
    }
    let quote = b64(&doc, "quote_b64")?;
    if quote.len() < TD_REPORT.end || quote[..2] != [4, 0] {
        return Err("document.tdx.quote_b64 is not a v4 TDX quote".into());
    }
    // The relay's VERIFIED is its own word: the quote is verified again here.
    crate::pcs::verify_tdx_chain(&quote, i.tdx, i.now_unix)
        .map_err(|e| format!("quote: DCAP to the pinned root: {e}"))?;
    let claims =
        x0ne_confidential::tdx_quote::parse(&quote).map_err(|e| format!("quote: {e:?}"))?;
    let log = b64(&doc, "ccel_b64")?;
    let (events, end) = ccel::parse_with_end(&log).map_err(|e| format!("CCEL: {e}"))?;
    let tail = &log[end..];
    if tail.len() < PADDING_KEPT || tail.iter().any(|&b| b != 0xFF) {
        return Err(format!(
            "CCEL: {} bytes after the events, not all 0xFF or fewer than {PADDING_KEPT}",
            tail.len()
        ));
    }
    let r = reference::parse(i.reference).map_err(|e| format!("reference: {e}"))?;
    let report = reference::check_image_rtmr0_reported(&r, i.machine, &claims, &events)
        .map_err(|e| format!("reference against this run: {e}"))?;
    // A relay pinning the TEST key says VERIFIED too (remote_verify.rs).
    let release = crate::archive_key::RELEASE
        .as_ref()
        .ok_or_else(|| "this build pins no release archive key".to_string())
        .and_then(crate::archive_key::pinned)
        .and_then(|key| key.verify_detached(i.reference, i.signature, i.now_unix));
    let (provenance, test_note) = match release {
        Ok(signer) => (
            measurements::Provenance {
                reference_signed_by: signer.subkey_fingerprint,
                release: true,
            },
            None,
        ),
        Err(e) if i.test_key => (
            measurements::Provenance {
                reference_signed_by: measurements::TEST_ARCHIVE_KEY.into(),
                release: false,
            },
            Some(format!(
                "TEST key run, not for release: the reference is not signed by the release \
                 archive key ({e}); the relay's report names signing subkey {}",
                doc["report"]["reference_signer_subkey"]
                    .as_str()
                    .unwrap_or("none")
            )),
        ),
        Err(e) => {
            return Err(format!(
                "the reference is not signed by the release archive key: {e}; a TEST-key \
                 run is written only with --test-key (make run-fixtures RUN_TEST_KEY=1)"
            ))
        }
    };
    let quote_b64 = doc["document"]["tdx"]["quote_b64"].clone();
    let m = measurements::render(
        &serde_json::json!({ "evidence": { "tdx_quote_b64": quote_b64 } }),
        &["rtmr1", "rtmr2"],
        Some(i.image),
        Some(&provenance),
    )?;
    let mut notes = vec![
        format!("RTMR1 {}", ccel::hex(&claims.rtmr[1])),
        format!("RTMR2 {}", ccel::hex(&claims.rtmr[2])),
        format!(
            "{} events, {end} bytes, cut at {} of {}",
            events.len(),
            end + PADDING_KEPT,
            log.len()
        ),
        format!("reference sha256 {}", sha256_hex(i.reference)),
    ];
    if let Some(n) = test_note {
        notes.insert(0, n);
    }
    notes.extend(report.warnings.iter().map(|w| format!("warning: {w}")));
    let at = format!("gce-{family}-tdx-{}", i.date);
    Ok(Fixtures {
        files: vec![
            (
                format!("measurements/{}.json", measurements::VERSION),
                format!("{m}\n").into_bytes(),
            ),
            (
                format!("x0ne-infer/tests/ccel/{at}.bin"),
                log[..end + PADDING_KEPT].to_vec(),
            ),
            (
                format!("x0ne-infer/tests/ccel/tdreport-{at}.bin"),
                quote[TD_REPORT].to_vec(),
            ),
            (
                format!(
                    "x0ne-infer/tests/reference/x0ne-cvm-{}.json",
                    r.image.version
                ),
                i.reference.to_vec(),
            ),
        ],
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG: &[u8] = include_bytes!("../tests/ccel/gce-a3-tdx-2026-10-17.bin");
    const TDREPORT: &[u8] = include_bytes!("../tests/ccel/tdreport-gce-a3-tdx-2026-10-17.bin");
    const REFERENCE: &[u8] = include_bytes!("../tests/reference/x0ne-cvm-0.2.json");
    const MEASUREMENTS: &str = include_str!("../../measurements/0.2.0-beta.5.json");
    const IMAGE: &str = "x0ne-cvm 0.2 beta5c (tar 48180d78, run 2026-10-17)";
    // The 2026-10-18 run of x0ne-cvm 0.2.3, from the relay's document (sha256
    // 8b8f6c6c…): whole quote, log cut 256 bytes into its padding, reference
    // and its .sig by the TEST archive key.
    const QUOTE_1018: &[u8] = include_bytes!("../tests/run_fixtures/gce-a3-tdx-2026-10-18.quote");
    const LOG_1018: &[u8] = include_bytes!("../tests/run_fixtures/gce-a3-tdx-2026-10-18.ccel");
    const REFERENCE_023: &[u8] = include_bytes!("../tests/run_fixtures/x0ne-cvm-0.2.3.json");
    const SIG_023: &[u8] = include_bytes!("../tests/run_fixtures/x0ne-cvm-0.2.3.json.sig");
    /// `document.server.attested_at_unix` of that document.
    const NOW: u64 = 1_792_346_090;

    /// A document as far as the fixtures read it: the quote, the log padded
    /// to 256 KiB with 0xFF as GCE serves it, the relay's report.
    fn document(quote: &[u8], log: &[u8], report: serde_json::Value) -> Vec<u8> {
        let mut ccel = log.to_vec();
        ccel.resize(262_144, 0xFF);
        let e = base64::engine::general_purpose::STANDARD;
        serde_json::to_vec(&serde_json::json!({
            "document": { "tdx": { "quote_b64": e.encode(quote), "ccel_b64": e.encode(&ccel) } },
            "report": report
        }))
        .unwrap()
    }

    fn report(verdict: &str, reference: &[u8]) -> serde_json::Value {
        serde_json::json!({
            "verdict": { verdict: {} },
            "reference_sha256_hex": sha256_hex(reference)
        })
    }

    /// The 2026-10-17 TD report (the fixtures keep no signature), signed
    /// under the root of the `pcs` tests.
    fn quote_1017() -> (Vec<u8>, TrustAnchor) {
        let mut quote = vec![0u8; 48];
        quote[0] = 4;
        quote[4] = 0x81;
        quote.extend_from_slice(TDREPORT);
        let fx = crate::pcs::tests::fixture_signing(&quote);
        (fx.quote, TrustAnchor::RootDer(fx.root_der))
    }

    fn inputs<'a>(attestation: &'a [u8], reference: &'a [u8], tdx: &'a TrustAnchor) -> Inputs<'a> {
        Inputs {
            attestation,
            reference,
            signature: b"",
            date: "2026-10-17",
            image: IMAGE,
            machine: "a3-highgpu-1g",
            tdx,
            now_unix: NOW,
            test_key: true,
        }
    }

    fn run_1018<'a>(attestation: &'a [u8], tdx: &'a TrustAnchor) -> Inputs<'a> {
        Inputs {
            signature: SIG_023,
            date: "2026-10-18",
            image: "x0ne-cvm 0.2.3 (tar 9537364d, run 2026-10-18)",
            ..inputs(attestation, REFERENCE_023, tdx)
        }
    }

    /// The reference of the 2026-10-18 run with RTMR0 pinned
    /// (`x0ne-cvm-0.2.3-rtmr0-pinned.json`, the SDK's test reference since
    /// 2026-10-31): the published 0.2.3 in every field but `rtmr0`, whose
    /// list is `rtmr0_template` of that run's log under Google's binary
    /// 53c558fe, and the run passes it with RTMR0 pinned.
    #[test]
    fn the_pinned_0_2_3_reference_is_the_published_one_with_the_runs_rtmr0_events() {
        const PINNED: &[u8] =
            include_bytes!("../tests/run_fixtures/x0ne-cvm-0.2.3-rtmr0-pinned.json");
        let mut published: serde_json::Value = serde_json::from_slice(REFERENCE_023).unwrap();
        let pinned: serde_json::Value = serde_json::from_slice(PINNED).unwrap();
        assert_eq!(published["rtmr0"]["policy"], "report");
        published["rtmr0"] = pinned["rtmr0"].clone();
        assert_eq!(published, pinned, "a field other than rtmr0 differs");
        let (events, _) = ccel::parse_with_end(LOG_1018).unwrap();
        let cfv = crate::tdvf::tests::unhex48(crate::reference::tests::CFV_53C558FE);
        let template = reference::rtmr0_template(&events, &cfv).unwrap();
        assert_eq!(
            pinned["rtmr0"]["events_sha384"]["a3-highgpu-1g"],
            serde_json::json!(template)
        );
        let reference = reference::parse(PINNED).unwrap();
        assert_eq!(reference.rtmr0.policy, reference::RTMR0_GCE_ENDORSED);
        let claims = x0ne_confidential::tdx_quote::parse(QUOTE_1018).unwrap();
        let firmware = crate::reference::tests::firmware_53c558fe(claims.mr_td);
        let report =
            reference::check_image_with(&reference, "a3-highgpu-1g", &claims, &events, &firmware)
                .expect("the run of 2026-10-18 under its reference with RTMR0 pinned");
        assert!(
            report.rtmr0_pinned && report.warnings.is_empty(),
            "{report:?}"
        );
        // The published reference reports RTMR0: refused by the attested path.
        let published = reference::parse(REFERENCE_023).unwrap();
        let e =
            reference::check_image_with(&published, "a3-highgpu-1g", &claims, &events, &firmware)
                .unwrap_err();
        assert!(e.contains("RTMR0") && e.contains("\"report\""), "{e}");
    }

    #[test]
    fn the_2026_10_17_document_gives_back_the_committed_fixtures() {
        let (quote, root) = quote_1017();
        let doc = document(&quote, LOG, report("Verified", REFERENCE));
        let f = build(&inputs(&doc, REFERENCE, &root)).expect("the real run");
        // The committed file predates `provenance`: a TEST-key run adds it.
        let want = MEASUREMENTS
            .replace(
                "\"version\": \"0.2.0-beta.5\"",
                &format!("\"version\": \"{}\"", measurements::VERSION),
            )
            .replace(
                "  \"registers\"",
                "  \"provenance\": {\n    \"reference_signed_by\": \"test-archive-key\",\n    \
                 \"release\": false\n  },\n  \"registers\"",
            );
        let expected: Vec<(String, Vec<u8>)> = vec![
            (
                format!("measurements/{}.json", measurements::VERSION),
                want.into_bytes(),
            ),
            (
                "x0ne-infer/tests/ccel/gce-a3-tdx-2026-10-17.bin".into(),
                LOG.to_vec(),
            ),
            (
                "x0ne-infer/tests/ccel/tdreport-gce-a3-tdx-2026-10-17.bin".into(),
                TDREPORT.to_vec(),
            ),
            (
                "x0ne-infer/tests/reference/x0ne-cvm-0.2.json".into(),
                REFERENCE.to_vec(),
            ),
        ];
        for ((p, b), (q, c)) in f.files.iter().zip(&expected) {
            assert_eq!(p, q);
            assert!(b == c, "{p} differs from the committed file");
        }
        assert_eq!(f.files.len(), expected.len());
    }

    /// The input of `make run-fixtures` for x0ne-cvm 0.2.3: its registers are
    /// the ones the image build predicted.
    #[test]
    fn the_2026_10_18_document_gives_its_fixtures() {
        let intel = TrustAnchor::intel_production();
        let doc = document(QUOTE_1018, LOG_1018, report("Verified", REFERENCE_023));
        let f = build(&run_1018(&doc, &intel)).expect("the real run");
        let paths: Vec<&str> = f.files.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(
            paths,
            [
                format!("measurements/{}.json", measurements::VERSION).as_str(),
                "x0ne-infer/tests/ccel/gce-a3-tdx-2026-10-18.bin",
                "x0ne-infer/tests/ccel/tdreport-gce-a3-tdx-2026-10-18.bin",
                "x0ne-infer/tests/reference/x0ne-cvm-0.2.3.json",
            ]
        );
        assert_eq!(f.files[1].1, LOG_1018);
        assert_eq!(f.files[2].1, QUOTE_1018[TD_REPORT]);
        assert_eq!(f.files[3].1, REFERENCE_023);
        for rtmr in ["RTMR1 758a634e6bf5d06a", "RTMR2 8b1d6ce2e48f55ac"] {
            assert!(f.notes.iter().any(|n| n.starts_with(rtmr)), "{:?}", f.notes);
        }
    }

    /// SECURITY: the reference copied is the one the run's relay
    /// checked the log against, named in its report by its digest.
    #[test]
    fn a_reference_the_relay_did_not_take_is_refused() {
        let intel = TrustAnchor::intel_production();
        let mut other = report("Verified", REFERENCE_023);
        other["reference_sha256_hex"] = "d1589ae8".into();
        let doc = document(QUOTE_1018, LOG_1018, other);
        let e = build(&run_1018(&doc, &intel)).unwrap_err();
        assert!(e.contains("another reference"), "{e}");
        let mut none = report("Verified", REFERENCE_023);
        none.as_object_mut().unwrap().remove("reference_sha256_hex");
        let doc = document(QUOTE_1018, LOG_1018, none);
        let e = build(&run_1018(&doc, &intel)).unwrap_err();
        assert!(e.contains("names no reference"), "{e}");
    }

    /// SECURITY: the relay's VERIFIED is its word; the quote is
    /// verified again, to the pinned Intel root, before a byte is written.
    #[test]
    fn a_quote_that_does_not_verify_to_intel_is_refused() {
        let intel = TrustAnchor::intel_production();
        // One byte of the quote's ECDSA signature, after the signed TD report.
        let mut forged = QUOTE_1018.to_vec();
        forged[TD_REPORT.end + 4] ^= 1;
        let doc = document(&forged, LOG_1018, report("Verified", REFERENCE_023));
        let e = build(&run_1018(&doc, &intel)).unwrap_err();
        assert!(e.contains("DCAP"), "{e}");
        // The real quote under another root than Intel's.
        let other = TrustAnchor::RootDer(crate::pcs::tests::fixture_signing(QUOTE_1018).root_der);
        let doc = document(QUOTE_1018, LOG_1018, report("Verified", REFERENCE_023));
        let e = build(&run_1018(&doc, &other)).unwrap_err();
        assert!(e.contains("DCAP"), "{e}");
    }

    /// SECURITY: the relay says VERIFIED under the TEST archive key
    /// too. Without the release key's signature, nothing unless `test_key`,
    /// and then the notes say TEST.
    #[test]
    fn a_reference_not_signed_by_the_release_key_needs_test_key() {
        let intel = TrustAnchor::intel_production();
        let doc = document(QUOTE_1018, LOG_1018, report("Verified", REFERENCE_023));
        let mut i = run_1018(&doc, &intel);
        i.test_key = false;
        let e = build(&i).unwrap_err();
        assert!(e.contains("release archive key"), "{e}");
        i.test_key = true;
        let f = build(&i).expect("a TEST-key run, accepted as such");
        assert!(
            f.notes.iter().any(|n| n.starts_with("TEST")),
            "{:?}",
            f.notes
        );
    }

    /// SECURITY: the provenance is in the measurements file itself,
    /// where a release build reads it (`measurements::resolve_from`).
    #[test]
    fn a_test_key_run_writes_its_provenance_into_the_measurements_file() {
        let intel = TrustAnchor::intel_production();
        let doc = document(QUOTE_1018, LOG_1018, report("Verified", REFERENCE_023));
        let f = build(&run_1018(&doc, &intel)).expect("a TEST-key run");
        let m: serde_json::Value = serde_json::from_slice(&f.files[0].1).unwrap();
        assert_eq!(
            m["provenance"],
            serde_json::json!({ "reference_signed_by": "test-archive-key", "release": false })
        );
    }

    #[test]
    fn nothing_is_written_for_a_run_the_fixtures_do_not_explain() {
        let (quote, root) = quote_1017();
        let good = document(&quote, LOG, report("Verified", REFERENCE));
        let refused =
            |doc: &[u8], reference: &[u8]| build(&inputs(doc, reference, &root)).unwrap_err();
        assert!(refused(
            &document(&quote, LOG, report("Refused", REFERENCE)),
            REFERENCE
        )
        .contains("not VERIFIED"));
        // A log that does not replay to the quote: one RTMR1 digest changed.
        let mut log = LOG.to_vec();
        let events = ccel::parse(LOG).unwrap();
        let gpt = events
            .iter()
            .find(|e| e.event_type == ccel::EV_EFI_GPT_EVENT)
            .unwrap()
            .sha384()
            .unwrap();
        let at = log.windows(48).position(|w| w == gpt).unwrap();
        log[at] ^= 1;
        assert!(refused(
            &document(&quote, &log, report("Verified", REFERENCE)),
            REFERENCE
        )
        .contains("differs from the quote"));
        // A reference of another image: its UKI digest changed.
        let other = String::from_utf8(REFERENCE.to_vec())
            .unwrap()
            .replacen("27a2435a", "27a2435b", 1);
        let doc = document(&quote, LOG, report("Verified", other.as_bytes()));
        assert!(refused(&doc, other.as_bytes()).starts_with("reference against this run"));
        // A log cut inside its events: the padding is not all 0xFF.
        let cut = document(
            &quote,
            &LOG[..LOG.len() - 300],
            report("Verified", REFERENCE),
        );
        assert!(build(&inputs(&cut, REFERENCE, &root)).is_err());
        let mut i = inputs(&good, REFERENCE, &root);
        i.date = "17/10/2026";
        assert!(build(&i).unwrap_err().contains("YYYY-MM-DD"));
    }
}
