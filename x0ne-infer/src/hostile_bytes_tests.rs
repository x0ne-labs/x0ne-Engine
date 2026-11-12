//! Hostile bytes against the binary formats a third party hands this crate:
//! TD quotes and their PCK chain, Intel's signed collateral and CRLs, SNP
//! reports, the CC event log, the TDVF and OVMF firmware layouts, Google's
//! launch endorsement, OpenPGP keys and signatures, OCSP responses, sealed
//! blobs. Seeded from the real fixtures of `tests/`, mutated by
//! `x0ne-sdk/tests/support/hostile.rs`, which says what every case is held
//! to. The textual formats are in `hostile_text_tests.rs`.
//!
//! Three statements recur, each on its own inputs:
//!
//! - a truncation is refused: every strict prefix of an input, or what it
//!   still yields is said and checked (a log cut between two events replays
//!   to other registers, a quote cut in its trailing padding is the quote);
//! - a signed object whose signed bytes changed never verifies: every
//!   single-bit flip of the signed region is refused, and a mutated input
//!   that verifies gives the verdict of the intact one, bit for bit (it
//!   changed bytes no signature covers);
//! - what parses and has a serializer reads back the same.
//!
//! A failure lists one case per kind of finding, each written so it can be
//! redone by hand on the fixture it names.

use crate::hostile::{bit_flips, cases, flipped, lengths, prefixes, show, Campaign};
use crate::pcs::{self, TrustAnchor};
use crate::tcb::{self, TcbPolicy};
use x0ne_confidential::{snp_report, tdx_quote};
use x509_parser::prelude::{FromDer, X509Certificate};

/// A sample TDX v4 quote of 2023 (`tests/VENDORED.md`), chained to Intel's root.
const SAMPLE_QUOTE: &[u8] = include_bytes!("../tests/pcs_live/quote.dat");
/// 2026-08-21: inside the sample quote's certificates.
const SAMPLE_NOW: u64 = 1_787_300_000;
/// The quote of the GCE run of 2026-10-18.
const RUN_QUOTE: &[u8] = include_bytes!("../tests/run_fixtures/gce-a3-tdx-2026-10-18.quote");
/// 2026-10-22, the day Intel's answers below were fetched.
const RUN_NOW: u64 = 1_792_666_540;
/// Header and TD report body: what the attestation key signs.
const SIGNED: usize = 632;

const TCB_INFO: &str =
    include_str!("../tests/pcs_live/2026-10-22/tcb_info_00806f050000_early.json");
const QE_IDENTITY: &str = include_str!("../tests/pcs_live/2026-10-22/qe_identity_early.json");
const TCB_CHAIN: &str = include_str!("../tests/pcs_live/tcb_issuer_chain.pem");
const QE_CHAIN: &str = include_str!("../tests/pcs_live/qe_issuer_chain.pem");
const OTHER_TCB_INFO: &str =
    include_str!("../tests/pcs_live/2026-10-22/tcb_info_90c06f000000_early.json");

/// Where the last certificate of the quote's PEM chain ends: a prefix that
/// stops before it lacks a byte of the chain.
fn chain_end(quote: &[u8]) -> usize {
    let marker = b"-----END";
    (0..quote.len().saturating_sub(marker.len()))
        .rev()
        .find(|&i| &quote[i..i + marker.len()] == marker)
        .expect("a PEM chain in the quote")
}

/// What a quote verifies to. The PCK's fingerprint is left out: it is the
/// digest of the whole certificate, whose bytes after the signed part can be
/// written in more than one way (the parser reads an OID under another tag,
/// and an ECDSA signature has two forms), so it is compared for equality by
/// its callers and proves nothing by differing.
fn chain_verdict(quote: &[u8], now: u64) -> Result<[u8; 64], pcs::PcsRejected> {
    pcs::verify_tdx_chain(quote, &TrustAnchor::intel_production(), now).map(|v| v.report_data)
}

#[test]
fn a_td_quote_cut_or_changed_never_panics_the_layout_reader_and_a_cut_one_is_refused() {
    let mut c = Campaign::new("TD quote layout", 0x7d51);
    for (seed, donor) in [(SAMPLE_QUOTE, RUN_QUOTE), (RUN_QUOTE, SAMPLE_QUOTE)] {
        let control = tdx_quote::parse(seed).expect("the intact quote parses");
        for p in prefixes(&seed[..SIGNED]) {
            let what = format!("the first {} bytes of a quote", p.len());
            let r = c.run(&what, p.len(), || tdx_quote::parse(p));
            c.hold(
                matches!(r, Some(Err(_))),
                "a quote cut inside its TD report is read",
                &what,
            );
        }
        // Every bit of the 48-byte header: version, attestation key type,
        // TEE type and QE vendor are refused, no other header bit (the two
        // reserved words, the user data) changes what is read from the body.
        for (at, bit, m) in bit_flips(seed, 0..48) {
            let what = format!("bit {bit} of header byte {at} flipped");
            let r = c.run(&what, m.len(), || tdx_quote::parse(&m));
            let names_the_format = at < 8 || (12..28).contains(&at);
            match r {
                Some(Ok(claims)) => {
                    c.hold(
                        !names_the_format,
                        "another version, key type, TEE type or QE vendor is read as TDX v4",
                        &what,
                    );
                    c.hold(
                        claims == control,
                        "a header bit changes the claims read from the body",
                        &what,
                    );
                }
                Some(Err(_)) => c.hold(
                    names_the_format,
                    "a header bit the layout does not read refuses the quote",
                    &what,
                ),
                None => {}
            }
        }
        for _ in 0..cases(4000) {
            let (m, what) = c.mutant(seed, donor);
            c.run(&what, m.len(), || {
                let _ = tdx_quote::parse(&m);
                let _ = tdx_quote::verify(&m, &[0x5a; 64]);
                let _ = pcs::signed_td_report(&m);
            });
        }
    }
    c.finish();
}

#[test]
fn a_td_quote_whose_signed_bytes_changed_never_verifies_to_intels_root() {
    let mut c = Campaign::new("TD quote chain", 0x9c5);
    let control = chain_verdict(RUN_QUOTE, RUN_NOW).expect("the run's quote verifies");
    assert_eq!(
        chain_verdict(SAMPLE_QUOTE, SAMPLE_NOW),
        Ok(tdx_quote::parse(SAMPLE_QUOTE).unwrap().report_data),
        "the sample quote verifies"
    );

    // Truncation: nothing shorter than the whole PEM chain verifies, and
    // what is cut after it is padding (the same verdict).
    let end = chain_end(RUN_QUOTE);
    for n in lengths(RUN_QUOTE.len(), 61, 4) {
        let p = &RUN_QUOTE[..n];
        let what = format!("the first {n} bytes of the run's quote");
        if let Some(Ok(v)) = c.run(&what, n, || chain_verdict(p, RUN_NOW)) {
            c.hold(n >= end, "a quote cut inside its PCK chain verifies", &what);
            c.hold(
                v == control,
                "a cut quote verifies to another verdict",
                &what,
            );
            c.accepted_unchanged();
        }
    }
    for n in lengths(end, 1, 0) {
        // Every length up to the signature section, then the chain sampled:
        // these are refused before any signature is computed.
        if n > SIGNED + 600 && n % 7 != 0 {
            continue;
        }
        let p = &RUN_QUOTE[..n];
        let what = format!("the first {n} bytes of the run's quote");
        let r = c.run(&what, n, || chain_verdict(p, RUN_NOW));
        c.hold(
            matches!(r, Some(Err(_))),
            "a quote cut inside its PCK chain verifies",
            &what,
        );
    }

    // The signed region: every bit of the header, one bit of every byte of
    // the TD report (every bit from X0NE_FUZZ_ITERS=10).
    let every_bit = crate::hostile::depth() >= 10;
    for at in 0..SIGNED {
        for bit in 0..8 {
            if at >= 48 && !every_bit && bit != (at * 5 + 3) % 8 {
                continue;
            }
            let m = flipped(RUN_QUOTE, at, bit);
            let what = format!("bit {bit} of signed byte {at} flipped");
            let r = c.run(&what, m.len(), || chain_verdict(&m, RUN_NOW));
            c.hold(
                matches!(r, Some(Err(_))),
                "a quote with a changed signed byte verifies",
                &what,
            );
        }
    }
    // The signature section up to the chain: lengths, the two signatures,
    // the attestation key, the QE report and its authentication data.
    let section_end = RUN_QUOTE
        .windows(10)
        .position(|w| w == b"-----BEGIN")
        .expect("a PEM chain");
    for at in (SIGNED..section_end).step_by(if every_bit { 1 } else { 3 }) {
        let bit = (at * 3 + 1) % 8;
        let m = flipped(RUN_QUOTE, at, bit);
        let what = format!("bit {bit} of signature-section byte {at} flipped");
        if let Some(Ok(v)) = c.run(&what, m.len(), || chain_verdict(&m, RUN_NOW)) {
            c.hold(
                v == control,
                "a changed signature section verifies to another verdict",
                &what,
            );
            c.accepted_unchanged();
        }
    }
    for _ in 0..cases(500) {
        let (m, what) = c.mutant(RUN_QUOTE, SAMPLE_QUOTE);
        let changed_signed = m.get(..SIGNED) != Some(&RUN_QUOTE[..SIGNED]);
        let r = c.run(&what, m.len(), || {
            let _ = tcb::pck_tcb(&m);
            let _ = tcb::qe_report(&m);
            let _ = tcb::tee_tcb_svn(&m);
            let _ = tcb::pck_crl_ca(&m);
            chain_verdict(&m, RUN_NOW)
        });
        if let Some(Ok(v)) = r {
            c.hold(
                !changed_signed,
                "a quote with changed signed bytes verifies",
                &what,
            );
            c.hold(
                v == control,
                "a mutated quote verifies to another verdict",
                &what,
            );
            c.accepted_unchanged();
        }
    }
    c.finish();
}

fn live(tcb_body: &str, tcb_chain: &str, qe_body: &str, qe_chain: &str) -> Result<String, String> {
    tcb::verify_live(
        RUN_QUOTE,
        tcb_body,
        tcb_chain,
        qe_body,
        qe_chain,
        &TrustAnchor::intel_production(),
        RUN_NOW,
        &TcbPolicy::strict(),
    )
    .map(|(platform, qe)| format!("{platform:?} {qe:?}"))
    .map_err(|e| e.to_string())
}

#[test]
fn intels_collateral_cut_or_changed_is_refused_or_rates_the_platform_the_same() {
    let mut c = Campaign::new("Intel collateral", 0x7cb1);
    let control = live(TCB_INFO, TCB_CHAIN, QE_IDENTITY, QE_CHAIN)
        .expect("Intel's answers of 2026-10-22 rate the run's platform");

    // A cut answer: refused, unless the cut took only what follows the JSON.
    for (which, body) in [("tcbInfo", TCB_INFO), ("QE identity", QE_IDENTITY)] {
        for n in lengths(body.len(), 53, 12) {
            if !body.is_char_boundary(n) {
                continue;
            }
            let p = &body[..n];
            let what = format!("the first {n} bytes of the {which} answer");
            let r = c.run(&what, n, || match which {
                "tcbInfo" => live(p, TCB_CHAIN, QE_IDENTITY, QE_CHAIN),
                _ => live(TCB_INFO, TCB_CHAIN, p, QE_CHAIN),
            });
            if let Some(Ok(_)) = r {
                c.hold(
                    p.trim_end() == body.trim_end(),
                    "a cut collateral answer verifies",
                    &what,
                );
            }
        }
    }

    // Every status Intel signed, rewritten to UpToDate in place: the bytes
    // under the signature changed.
    for (i, _) in TCB_INFO.match_indices("\"tcbStatus\":\"") {
        let from = i + "\"tcbStatus\":\"".len();
        let to = from + TCB_INFO[from..].find('"').expect("a closing quote");
        if &TCB_INFO[from..to] == "UpToDate" {
            continue;
        }
        let forged = format!("{}UpToDate{}", &TCB_INFO[..from], &TCB_INFO[to..]);
        let what = format!("the tcbStatus at byte {from} rewritten to UpToDate");
        let r = c.run(&what, forged.len(), || {
            live(&forged, TCB_CHAIN, QE_IDENTITY, QE_CHAIN)
        });
        c.hold(
            matches!(r, Some(Err(_))),
            "a tcbInfo with a rewritten status verifies",
            &what,
        );
    }

    for _ in 0..cases(220) {
        let (m, what) = c.text_mutant(TCB_INFO, OTHER_TCB_INFO);
        let what = format!("tcbInfo answer: {what}");
        if let Some(Ok(v)) = c.run(&what, m.len(), || {
            live(&m, TCB_CHAIN, QE_IDENTITY, QE_CHAIN)
        }) {
            c.hold(
                v == control,
                "a mutated tcbInfo rates the platform otherwise",
                &what,
            );
            c.accepted_unchanged();
        }
        let (m, what) = c.text_mutant(QE_IDENTITY, TCB_INFO);
        let what = format!("QE identity answer: {what}");
        if let Some(Ok(v)) = c.run(&what, m.len(), || live(TCB_INFO, TCB_CHAIN, &m, QE_CHAIN)) {
            c.hold(
                v == control,
                "a mutated QE identity rates the QE otherwise",
                &what,
            );
            c.accepted_unchanged();
        }
    }
    for _ in 0..cases(120) {
        let (m, what) = c.text_mutant(TCB_CHAIN, QE_CHAIN);
        let what = format!("tcbInfo issuer chain: {what}");
        if let Some(Ok(v)) = c.run(&what, m.len(), || live(TCB_INFO, &m, QE_IDENTITY, QE_CHAIN)) {
            c.hold(
                v == control,
                "a mutated issuer chain rates the platform otherwise",
                &what,
            );
            c.accepted_unchanged();
        }
    }
    c.finish();
}

#[test]
fn intels_crls_cut_or_changed_are_refused() {
    const PCK_CRL: &[u8] = include_bytes!("../tests/pcs_live/pck_crl_platform_2026-10-18.der");
    const PCK_CRL_FETCHED: u64 = 1_792_350_538;
    const ROOT_CRL: &[u8] = include_bytes!("../tests/pcs_live/2026-10-23/root_ca_crl.der");
    const ROOT_CRL_FETCHED: u64 = 1_792_735_917;
    let anchor = TrustAnchor::intel_production();
    // Fetched 35 minutes after the CRL's thisUpdate: inside production's age limit.
    let window_only = TcbPolicy::production();
    let pck = |crl: &[u8]| {
        tcb::verify_pck_crl(RUN_QUOTE, crl, &anchor, PCK_CRL_FETCHED, &window_only)
            .map_err(|e| e.to_string())
    };
    let root = |crl: &[u8]| {
        tcb::verify_root_crl(
            RUN_QUOTE,
            &[TCB_CHAIN, QE_CHAIN],
            crl,
            &anchor,
            ROOT_CRL_FETCHED,
        )
        .map_err(|e| e.to_string())
    };
    pck(PCK_CRL).expect("Intel's PCK CRL of 2026-10-18 passes the run's PCK");
    root(ROOT_CRL).expect("Intel's root CRL passes the run's chain");

    let mut c = Campaign::new("Intel CRLs", 0xc71);
    for n in lengths(PCK_CRL.len(), 29, 8) {
        let what = format!("the first {n} bytes of the PCK CRL");
        let r = c.run(&what, n, || pck(&PCK_CRL[..n]));
        c.hold(matches!(r, Some(Err(_))), "a cut CRL is accepted", &what);
    }
    for p in prefixes(ROOT_CRL) {
        let what = format!("the first {} bytes of the root CRL", p.len());
        let r = c.run(&what, p.len(), || root(p));
        c.hold(matches!(r, Some(Err(_))), "a cut CRL is accepted", &what);
    }
    for _ in 0..cases(250) {
        let (m, what) = c.mutant(PCK_CRL, ROOT_CRL);
        let what = format!("PCK CRL: {what}");
        if let Some(Ok(())) = c.run(&what, m.len(), || pck(&m)) {
            c.hold(
                holds(&m, signed_part(PCK_CRL)),
                "a PCK CRL is accepted without its signed bytes",
                &what,
            );
            c.accepted_unchanged();
        }
        let (m, what) = c.mutant(ROOT_CRL, PCK_CRL);
        let what = format!("root CRL: {what}");
        if let Some(Ok(())) = c.run(&what, m.len(), || root(&m)) {
            c.hold(
                holds(&m, signed_part(ROOT_CRL)),
                "a root CRL is accepted without its signed bytes",
                &what,
            );
            c.accepted_unchanged();
        }
    }
    c.finish();
}

/// The signed part of a CRL of Intel: what follows the outer header, up to
/// the outer `signatureAlgorithm`.
fn signed_part(crl: &[u8]) -> &[u8] {
    let end = crl
        .windows(ECDSA_SHA256.len())
        .rposition(|w| w == ECDSA_SHA256)
        .expect("an ecdsa-with-SHA256 AlgorithmIdentifier");
    &crl[4..end]
}

/// Whether `part` is in `bytes`, whole and in order.
fn holds(bytes: &[u8], part: &[u8]) -> bool {
    bytes.windows(part.len()).any(|w| w == part)
}

/// DER of `AlgorithmIdentifier { ecdsa-with-SHA256 }`: in a certificate or a
/// CRL of Intel, once inside the signed part and once after it.
const ECDSA_SHA256: [u8; 12] = [
    0x30, 0x0a, 0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02,
];

/// `der` with its outer `signatureAlgorithm`, the last one, rewritten to
/// ecdsa-with-SHA384: bytes no signature covers.
fn with_another_outer_algorithm(der: &[u8]) -> Vec<u8> {
    let at = der
        .windows(ECDSA_SHA256.len())
        .rposition(|w| w == ECDSA_SHA256)
        .expect("an ecdsa-with-SHA256 AlgorithmIdentifier");
    let mut out = der.to_vec();
    out[at + ECDSA_SHA256.len() - 1] = 0x03;
    out
}

/// RFC 5280, sections 4.1.1.2 and 5.1.1.2: the algorithm written after the
/// signed part MUST be the one written inside it. Found by the campaigns
/// above at `X0NE_FUZZ_ITERS=100`: such a certificate or CRL verified, under
/// other bytes and, for the PCK, another fingerprint.
#[test]
fn a_certificate_or_a_crl_whose_outer_algorithm_is_not_the_signed_one_is_refused() {
    use base64::Engine as _;
    const PCK_CRL: &[u8] = include_bytes!("../tests/pcs_live/pck_crl_platform_2026-10-18.der");
    const ROOT_CRL: &[u8] = include_bytes!("../tests/pcs_live/2026-10-23/root_ca_crl.der");
    let anchor = TrustAnchor::intel_production();
    let policy = TcbPolicy::production();

    tcb::verify_pck_crl(RUN_QUOTE, PCK_CRL, &anchor, 1_792_350_538, &policy).expect("intact");
    let err = tcb::verify_pck_crl(
        RUN_QUOTE,
        &with_another_outer_algorithm(PCK_CRL),
        &anchor,
        1_792_350_538,
        &policy,
    )
    .expect_err("a PCK CRL with another outer algorithm is accepted");
    assert!(err.to_string().contains("algorithm"), "{err}");

    let chains = [TCB_CHAIN, QE_CHAIN];
    tcb::verify_root_crl(RUN_QUOTE, &chains, ROOT_CRL, &anchor, 1_792_735_917).expect("intact");
    let err = tcb::verify_root_crl(
        RUN_QUOTE,
        &chains,
        &with_another_outer_algorithm(ROOT_CRL),
        &anchor,
        1_792_735_917,
    )
    .expect_err("a root CRL with another outer algorithm is accepted");
    assert!(err.to_string().contains("algorithm"), "{err}");

    // The PCK certificate, first of the quote's PEM chain: decoded, changed,
    // encoded back on the same lines.
    const BEGIN: &[u8] = b"-----BEGIN CERTIFICATE-----\n";
    let from = RUN_QUOTE
        .windows(BEGIN.len())
        .position(|w| w == BEGIN)
        .expect("a PEM chain")
        + BEGIN.len();
    let to = from
        + RUN_QUOTE[from..]
            .windows(5)
            .position(|w| w == b"-----")
            .expect("the end of the first certificate");
    let body = &RUN_QUOTE[from..to];
    let b64: Vec<u8> = body.iter().copied().filter(|b| *b != b'\n').collect();
    let der = base64::engine::general_purpose::STANDARD
        .decode(&b64)
        .expect("base64");
    let changed = base64::engine::general_purpose::STANDARD
        .encode(with_another_outer_algorithm(&der))
        .into_bytes();
    assert_eq!(changed.len(), b64.len());
    let mut chars = changed.into_iter();
    let mut quote = RUN_QUOTE.to_vec();
    for b in &mut quote[from..to] {
        if *b != b'\n' {
            *b = chars.next().expect("as many characters");
        }
    }
    assert_ne!(quote, RUN_QUOTE);
    assert_eq!(quote[..from], RUN_QUOTE[..from]);
    chain_verdict(RUN_QUOTE, RUN_NOW).expect("intact");
    let err = chain_verdict(&quote, RUN_NOW)
        .expect_err("a PCK certificate with another outer algorithm is accepted");
    assert!(err.to_string().contains("algorithm"), "{err}");
}

#[test]
fn an_snp_report_cut_or_changed_never_panics_and_never_verifies_to_amds_root() {
    use crate::snp::tests::{
        AMD_GENOA_CRL, AZURE_GENOA_REPORT, AZURE_GENOA_VCEK, NOW_REAL, SB3019_GENOA,
    };
    use crate::snp::{self, SnpAnchors, SnpPolicy, SnpSmt};
    let payload = snp::evidence_payload(AZURE_GENOA_REPORT, AZURE_GENOA_VCEK);
    let anchors = SnpAnchors::production();
    let policy = SnpPolicy {
        minimum: Some(SB3019_GENOA),
        smt: SnpSmt::Allowed,
        ..SnpPolicy::default()
    };
    let verify = |p: &[u8]| {
        // Without the key certificate's fingerprint: see `chain_verdict`.
        snp::verify_snp_report(p, AMD_GENOA_CRL, &anchors, &policy, NOW_REAL).map(|mut v| {
            v.key_sha256_hex.clear();
            format!("{v:?}")
        })
    };
    let control = verify(&payload).expect("the real Genoa report verifies under AMD's root");

    let mut c = Campaign::new("SNP report", 0x5e9);
    // The report alone: its length is exact, so every cut is refused.
    for p in prefixes(AZURE_GENOA_REPORT) {
        let what = format!("the first {} bytes of a report", p.len());
        let r = c.run(&what, p.len(), || snp_report::parse(p));
        c.hold(matches!(r, Some(Err(_))), "a cut report is read", &what);
    }
    for _ in 0..cases(4000) {
        let (m, what) = c.mutant(AZURE_GENOA_REPORT, AZURE_GENOA_VCEK);
        c.run(&what, m.len(), || {
            let _ = snp_report::parse(&m);
            let _ = snp::signed_body_without_report_data(&m);
        });
    }
    // The evidence: report then key certificate.
    for n in lengths(payload.len(), 17, 8) {
        let what = format!("the first {n} bytes of the evidence");
        let r = c.run(&what, n, || verify(&payload[..n]));
        c.hold(matches!(r, Some(Err(_))), "cut evidence verifies", &what);
    }
    let every_bit = crate::hostile::depth() >= 10;
    for at in 0..snp_report::SIGNED_LEN {
        for bit in 0..8 {
            if !every_bit && bit != (at * 5 + 1) % 8 {
                continue;
            }
            let m = flipped(&payload, at, bit);
            let what = format!("bit {bit} of signed report byte {at:#x} flipped");
            let r = c.run(&what, m.len(), || verify(&m));
            c.hold(
                matches!(r, Some(Err(_))),
                "a report with a changed signed byte verifies",
                &what,
            );
        }
    }
    for _ in 0..cases(300) {
        let (m, what) = c.mutant(&payload, AZURE_GENOA_VCEK);
        let changed_signed =
            m.get(..snp_report::SIGNED_LEN) != Some(&payload[..snp_report::SIGNED_LEN]);
        let r = c.run(&what, m.len(), || {
            let _ = snp::split_evidence(&m);
            let _ = snp::evidence_product(&m);
            verify(&m)
        });
        if let Some(Ok(v)) = r {
            c.hold(
                !changed_signed,
                "evidence with changed signed bytes verifies",
                &what,
            );
            c.hold(
                v == control,
                "mutated evidence verifies to another verdict",
                &what,
            );
            c.accepted_unchanged();
        }
    }
    c.finish();
}

#[test]
fn an_event_log_cut_or_changed_never_panics_and_a_cut_one_replays_to_other_registers() {
    use crate::ccel;
    const LOG: &[u8] = include_bytes!("../tests/ccel/gce-a3-tdx-2026-10-18.bin");
    const OTHER: &[u8] = include_bytes!("../tests/ccel/gce-a3-tdx-2026-10-14.bin");
    let (events, end) = ccel::parse_with_end(LOG).expect("the run's log parses");
    let registers = ccel::replay(&events).expect("the run's log replays");

    let mut c = Campaign::new("CC event log", 0xcce1);
    // A log is cut between two events without a trace in its own bytes: what
    // refuses it is the quote, whose registers the shorter log no longer
    // replays to. Checked here for every cut before the end of the events.
    for p in prefixes(&LOG[..end]) {
        let what = format!("the first {} bytes of the log", p.len());
        let r = c.run(&what, p.len(), || {
            ccel::parse(p).and_then(|evs| ccel::replay(&evs))
        });
        c.hold(
            !matches!(r, Some(Ok(got)) if got == registers),
            "a cut log replays to the registers of the whole log",
            &what,
        );
    }
    for _ in 0..cases(4000) {
        let (m, what) = c.mutant(LOG, OTHER);
        let parsed = c.run(&what, m.len(), || {
            let parsed = ccel::parse_with_end(&m);
            if let Ok((evs, _)) = &parsed {
                let _ = ccel::replay(evs);
                for e in evs {
                    let _ = (e.sha384(), e.rtmr(), e.name());
                }
            }
            parsed
        });
        // What parsed is written back and read again as the same events,
        // unless it names more algorithms than a header may declare.
        if let Some(Ok((evs, at))) = parsed {
            c.hold(at <= m.len(), "the events end after the log", &what);
            let what = format!("{what}, then encoded");
            let again = c.run(&what, m.len(), || ccel::parse(&ccel::encode(&evs)));
            match again {
                Some(Ok(back)) => c.hold(back == evs, "a log does not read back the same", &what),
                Some(Err(e)) => c.hold(
                    e.contains("algorithms"),
                    "an encoded log is refused",
                    &format!("{what}: {e}"),
                ),
                None => {}
            }
        }
    }
    c.finish();
}

/// The fields of the synthetic TDVF's metadata (`tdvf::tests::synthetic`):
/// the descriptor at 0x2F10, then four 32-byte sections.
const TDVF_DESCRIPTOR: usize = 0x2F10;
const TDVF_OFFSET_BLOCK: usize = 0x3FB8;

#[test]
fn a_tdvf_image_cut_or_changed_never_panics_its_reader_nor_its_measurement() {
    use crate::tdvf::{self, Image};
    let image = tdvf::tests::synthetic();
    let control = Image::parse(image).expect("the synthetic image parses");
    let (mrtd, cfv) = (control.mrtd(), control.cfv_sha384());
    let read = |b: &[u8]| Image::parse(b).map(|i| (i.mrtd(), i.cfv_sha384()));

    let mut c = Campaign::new("TDVF image", 0x7df);
    // The volumes must add up to the image: no prefix is one.
    for p in prefixes(image) {
        let what = format!("the first {} bytes of the image", p.len());
        let r = c.run(&what, p.len(), || read(p));
        c.hold(matches!(r, Some(Err(_))), "a cut image is measured", &what);
    }
    // Every integer of the metadata set to the values a bound is wrong at.
    let mut fields: Vec<(usize, usize)> = vec![
        (TDVF_DESCRIPTOR + 4, 4),
        (TDVF_DESCRIPTOR + 8, 4),
        (TDVF_DESCRIPTOR + 12, 4),
        (TDVF_OFFSET_BLOCK, 4),
        (TDVF_OFFSET_BLOCK + 4, 2),
        (image.len() - 0x20 - 18, 2),
    ];
    for k in 0..4 {
        let at = TDVF_DESCRIPTOR + 16 + 32 * k;
        fields.extend([
            (at, 4),
            (at + 4, 4),
            (at + 8, 8),
            (at + 16, 8),
            (at + 24, 4),
            (at + 28, 4),
        ]);
    }
    let len = image.len() as u64;
    for &(at, width) in &fields {
        let mut cur = [0u8; 8];
        cur[..width].copy_from_slice(&image[at..at + width]);
        let cur = u64::from_le_bytes(cur);
        let ones = if width == 8 {
            u64::MAX
        } else {
            (1 << (8 * width)) - 1
        };
        let values = [
            0,
            1,
            0xff,
            0x100,
            0xfff,
            0x1000,
            0x1001,
            cur.wrapping_sub(1),
            cur.wrapping_add(1),
            cur.wrapping_add(0x1000),
            cur.wrapping_sub(0x1000),
            len - 1,
            len,
            len + 1,
            0x7fff_ffff,
            0x8000_0000,
            0xffff_f000,
            0xffff_ffff,
            0x1_0000_0000,
            1 << 26,
            (1 << 26) + 0x1000,
            u64::MAX - 0xfff,
            ones >> 1,
            (ones >> 1) + 1,
            ones - 1,
            ones,
        ];
        for v in values {
            let v = v & ones;
            let mut m = image.to_vec();
            m[at..at + width].copy_from_slice(&v.to_le_bytes()[..width]);
            let what = format!("the {width}-byte field at {at:#x} set to {v:#x}");
            if let Some(Ok(got)) = c.run(&what, m.len(), || read(&m)) {
                // The first section is measured byte for byte, the others by
                // their addresses: a changed field that still parses is the
                // same image only if it is the same value.
                c.hold(
                    got != (mrtd, cfv) || v == cur,
                    "a changed metadata field leaves the measurement unchanged",
                    &what,
                );
            }
        }
    }
    for _ in 0..cases(6000) {
        let (m, what) = c.mutant(image, &image[TDVF_DESCRIPTOR - 16..TDVF_DESCRIPTOR + 144]);
        c.run(&what, m.len(), || read(&m));
    }
    c.finish();
}

#[test]
fn an_ovmf_image_cut_or_changed_never_panics_nor_stalls_the_launch_measurement() {
    use crate::snp_launch::{launch_digest, Launch, Ovmf};
    let image = ovmf_fixture();
    let digest = |b: &[u8]| {
        Ovmf::parse(b).map(|_| ())?;
        launch_digest(&Launch {
            ovmf: b,
            kernel: None,
            initrd: None,
            append: None,
            vcpus: 2,
            vcpu_sig: 0x00a1_0f10,
            guest_features: 1,
        })
    };
    let control = digest(&image).expect("the synthetic OVMF is measured");

    let mut c = Campaign::new("OVMF image", 0x0f3f);
    for n in lengths(image.len(), 1, 0) {
        let what = format!("the first {n} bytes of the image");
        let r = c.run(&what, n, || digest(&image[..n]));
        c.hold(
            !matches!(r, Some(Ok(d)) if d == control),
            "a cut image has the measurement of the whole one",
            &what,
        );
    }
    for _ in 0..cases(3000) {
        // Mutations that keep the size a multiple of a page reach the tables.
        let (mut m, what) = c.mutant(&image, &image[image.len() - 0x200..]);
        if c.rng.below(4) != 0 {
            m.resize(image.len(), 0);
        }
        c.run(&what, m.len(), || digest(&m));
    }
    c.finish();
}

/// Two pages in OVMF's layout: the footer table 32 bytes before the end, with
/// the SEV-ES reset block and the SEV metadata entry, and an `ASEV`
/// descriptor of three sections (memory, secrets, CPUID).
fn ovmf_fixture() -> Vec<u8> {
    fn guid(s: &str) -> [u8; 16] {
        let h: Vec<u8> = s
            .replace('-', "")
            .as_bytes()
            .chunks(2)
            .map(|c| u8::from_str_radix(std::str::from_utf8(c).unwrap(), 16).unwrap())
            .collect();
        let mut g = [0u8; 16];
        g[0..4].copy_from_slice(&[h[3], h[2], h[1], h[0]]);
        g[4..6].copy_from_slice(&[h[5], h[4]]);
        g[6..8].copy_from_slice(&[h[7], h[6]]);
        g[8..16].copy_from_slice(&h[8..16]);
        g
    }
    let n = 0x2000usize;
    let mut b: Vec<u8> = (0..n).map(|i| (i * 13 + 5) as u8).collect();
    // The ASEV descriptor, 0x400 bytes before the end.
    let meta = n - 0x400;
    b[meta..meta + 4].copy_from_slice(b"ASEV");
    b[meta + 4..meta + 8].copy_from_slice(&(16u32 + 3 * 12).to_le_bytes());
    b[meta + 8..meta + 12].copy_from_slice(&1u32.to_le_bytes());
    b[meta + 12..meta + 16].copy_from_slice(&3u32.to_le_bytes());
    for (k, (gpa, size, kind)) in [
        (0x80_0000u32, 0x9000u32, 1u32),
        (0x80_d000, 0x1000, 2),
        (0x80_e000, 0x1000, 3),
    ]
    .into_iter()
    .enumerate()
    {
        let at = meta + 16 + 12 * k;
        b[at..at + 4].copy_from_slice(&gpa.to_le_bytes());
        b[at + 4..at + 8].copy_from_slice(&size.to_le_bytes());
        b[at + 8..at + 12].copy_from_slice(&kind.to_le_bytes());
    }
    // The footer table, entries growing downwards: data, u16 size, GUID.
    let mut table = Vec::new();
    for (g, data) in [
        ("dc886566-984a-4798-a75e-5585a7bf67cc", 0x400u32),
        ("00f771de-1a7e-4fcb-890e-68c77e2fb44e", 0xffff_fff0),
    ] {
        table.extend_from_slice(&data.to_le_bytes());
        table.extend_from_slice(&(4u16 + 18).to_le_bytes());
        table.extend_from_slice(&guid(g));
    }
    let foot = n - 32 - 18;
    let start = foot - table.len();
    b[start..foot].copy_from_slice(&table);
    b[foot..foot + 2].copy_from_slice(&((table.len() + 18) as u16).to_le_bytes());
    b[foot + 2..foot + 18].copy_from_slice(&guid("96b582de-1fb2-45f7-baea-a366c55a082d"));
    b
}

#[test]
fn googles_launch_endorsement_cut_or_changed_is_refused_or_endorses_the_same() {
    use crate::gce_endorsement::{
        self as gce, tests::NOW, tests::VENDORED, tests::VENDORED_C1EE9C16,
    };
    let verify = |b: &[u8]| gce::verify(b, gce::ROOT_1_DER, NOW).map(|e| format!("{e:?}"));
    let mut c = Campaign::new("GCE launch endorsement", 0x6ce);
    for (seed, donor) in [(VENDORED, VENDORED_C1EE9C16), (VENDORED_C1EE9C16, VENDORED)] {
        let control = verify(seed).expect("Google's endorsement verifies under Google's root");
        for n in lengths(seed.len(), 23, 8) {
            let what = format!("the first {n} bytes of the endorsement");
            let r = c.run(&what, n, || verify(&seed[..n]));
            c.hold(
                matches!(r, Some(Err(_))),
                "a cut endorsement verifies",
                &what,
            );
        }
        for _ in 0..cases(250) {
            let (m, what) = c.mutant(seed, donor);
            let r = c.run(&what, m.len(), || {
                let _ = gce::uefi_sha384_hex_unverified(&m);
                verify(&m)
            });
            if let Some(Ok(v)) = r {
                c.hold(
                    v == control,
                    "a mutated endorsement verifies and endorses something else",
                    &what,
                );
                c.accepted_unchanged();
            }
        }
    }
    c.finish();
}

#[test]
fn an_openpgp_key_or_signature_cut_or_changed_is_refused_or_names_the_same_signer() {
    use crate::openpgp::Cert;
    const KEYRING: &[u8] = include_bytes!("../tests/archive_key/test-archive-keyring.gpg");
    const REVOKED: &[u8] = include_bytes!("../tests/archive_key/revoked-subkey-keyring.gpg");
    const DATA: &[u8] = include_bytes!("../tests/archive_key/reference.json");
    const SIG: &[u8] = include_bytes!("../tests/archive_key/reference.json.sig");
    const OTHER_SIG: &[u8] = include_bytes!("../tests/archive_key/reference.json.rogue.sig");
    /// 2026-10-17, the day after the fixtures were made.
    const NOW: u64 = 1_792_108_800 + 86_400;
    let cert = Cert::parse(KEYRING).expect("GnuPG's keyring parses");
    let control = cert
        .verify_detached(DATA, SIG, NOW)
        .expect("GnuPG's signature verifies");

    let mut c = Campaign::new("OpenPGP", 0x9690);
    for p in prefixes(SIG) {
        let what = format!("the first {} bytes of the signature", p.len());
        let r = c.run(&what, p.len(), || cert.verify_detached(DATA, p, NOW));
        c.hold(matches!(r, Some(Err(_))), "a cut signature verifies", &what);
    }
    for p in prefixes(DATA) {
        let what = format!("the first {} bytes of the signed file", p.len());
        let r = c.run(&what, p.len(), || cert.verify_detached(p, SIG, NOW));
        c.hold(matches!(r, Some(Err(_))), "a cut file verifies", &what);
    }
    for (at, bit, m) in bit_flips(SIG, 0..SIG.len()) {
        let what = format!("bit {bit} of signature byte {at} flipped");
        if let Some(Ok(s)) = c.run(&what, m.len(), || cert.verify_detached(DATA, &m, NOW)) {
            c.hold(
                s == control,
                "a changed signature names another signer",
                &what,
            );
            c.accepted_unchanged();
        }
    }
    for _ in 0..cases(1500) {
        let (m, what) = c.mutant(DATA, KEYRING);
        let what = format!("signed file: {what}");
        let r = c.run(&what, m.len(), || cert.verify_detached(&m, SIG, NOW));
        c.hold(
            !matches!(r, Some(Ok(_))) || m == DATA,
            "a changed file verifies under the intact signature",
            &what,
        );
        let (m, what) = c.mutant(SIG, OTHER_SIG);
        let what = format!("signature: {what}");
        if let Some(Ok(s)) = c.run(&what, m.len(), || cert.verify_detached(DATA, &m, NOW)) {
            c.hold(
                s == control,
                "a mutated signature names another signer",
                &what,
            );
            c.accepted_unchanged();
        }
    }
    // The keyring is pinned by the build, not handed by a peer: it is read
    // here for panics, and a mutated one that still verifies the signature
    // holds the same two keys.
    for p in prefixes(KEYRING) {
        let what = format!("the first {} bytes of the keyring", p.len());
        c.run(&what, p.len(), || {
            Cert::parse(p).map(|k| k.verify_detached(DATA, SIG, NOW))
        });
    }
    for _ in 0..cases(1500) {
        let (m, what) = c.mutant(KEYRING, REVOKED);
        let what = format!("keyring: {what}");
        let r = c.run(&what, m.len(), || {
            Cert::parse(&m).map(|k| k.verify_detached(DATA, SIG, NOW))
        });
        if let Some(Ok(Ok(s))) = r {
            c.hold(
                (
                    s.primary_fingerprint.as_str(),
                    s.subkey_fingerprint.as_str(),
                ) == (
                    control.primary_fingerprint.as_str(),
                    control.subkey_fingerprint.as_str(),
                ),
                "a mutated keyring verifies the signature under other keys",
                &what,
            );
            c.accepted_unchanged();
        }
    }
    c.finish();
}

#[test]
fn an_ocsp_response_cut_or_changed_is_refused_or_says_the_same() {
    use crate::nras_ocsp::tests::{bench, respond, Response, Status, NOW};
    use crate::nras_ocsp::{verify_response, NrasOcspPolicy};
    let b = bench();
    let (_, leaf) = X509Certificate::from_der(&b.leaf_der).expect("leaf");
    let (_, issuer) = X509Certificate::from_der(&b.inter.der).expect("issuer");
    let policy = NrasOcspPolicy::required();
    let good = respond(&Response::good_at(NOW), &b.leaf_der, &b.inter);
    let revoked = respond(
        &Response {
            status: Status::Revoked(NOW - 3600, Some(1)),
            ..Response::good_at(NOW)
        },
        &b.leaf_der,
        &b.inter,
    );
    let verify = |der: &[u8]| verify_response(der, &leaf, &issuer, NOW, &policy);

    let mut c = Campaign::new("OCSP response", 0x0c59);
    for (seed, donor) in [(&good, &revoked), (&revoked, &good)] {
        let control = verify(seed).expect("the responder's answer verifies");
        // A splice can be the donor, whole: the responder signed that too.
        let other = verify(donor).expect("the responder's answer verifies");
        for p in prefixes(seed) {
            let what = format!("the first {} bytes of a response", p.len());
            let r = c.run(&what, p.len(), || verify(p));
            c.hold(
                matches!(r, Some(Err(_))),
                "a cut response is believed",
                &what,
            );
        }
        for (at, bit, m) in bit_flips(seed, 0..seed.len()) {
            if bit != (at * 3) % 8 && crate::hostile::depth() < 10 {
                continue;
            }
            let what = format!("bit {bit} of response byte {at} flipped");
            if let Some(Ok(s)) = c.run(&what, m.len(), || verify(&m)) {
                c.hold(
                    s == control,
                    "a changed response says something else",
                    &what,
                );
                c.accepted_unchanged();
            }
        }
        for _ in 0..cases(700) {
            let (m, what) = c.mutant(seed, donor);
            if let Some(Ok(s)) = c.run(&what, m.len(), || verify(&m)) {
                c.hold(
                    s == control || s == other,
                    "a mutated response says what the responder never signed",
                    &format!("{what}: {}", show(&m)),
                );
                c.accepted_unchanged();
            }
        }
    }
    c.finish();
}

/// A certificate as it is written: where its signed part starts and ends, and
/// what follows it (the outer `signatureAlgorithm`, then the signature).
fn signed_part_of(cert: &[u8]) -> std::ops::Range<usize> {
    let (_, c) = X509Certificate::from_der(cert).expect("a certificate");
    let tbs = c.tbs_certificate.as_ref();
    let at = cert
        .windows(tbs.len())
        .position(|w| w == tbs)
        .expect("the signed part is in the certificate");
    at..at + tbs.len()
}

/// The first `AlgorithmIdentifier` of a certificate's signed part: the one
/// its outer `signatureAlgorithm` must repeat, byte for byte.
fn signed_algorithm_of(cert: &[u8]) -> Vec<u8> {
    let (_, c) = X509Certificate::from_der(cert).expect("a certificate");
    let (tbs, oid) = (
        c.tbs_certificate.as_ref(),
        c.tbs_certificate.signature.algorithm.as_bytes(),
    );
    let oid_at = tbs
        .windows(oid.len())
        .position(|w| w == oid)
        .expect("the signed algorithm's OID");
    // SEQUENCE and its length, OBJECT IDENTIFIER and its length, before it.
    let at = oid_at - 4;
    assert_eq!((tbs[at], tbs[at + 2]), (0x30, 0x06), "short lengths");
    tbs[at..at + 2 + usize::from(tbs[at + 1])].to_vec()
}

/// Whether `tail`, what follows a certificate's signed part, starts with
/// `algorithm`, byte for byte: the tag of its SEQUENCE and the tag of the OID
/// inside it included, which x509-parser 0.18 does not read (measured on
/// 2026-11-10) and the verifiers compare all the same.
fn repeats(tail: &[u8], algorithm: &[u8]) -> bool {
    tail.starts_with(algorithm)
}

/// RFC 5280, section 4.1.1.2, on the certificate a delegated OCSP responder
/// brings: it travels in `certs`, which the response's signature does not
/// cover, and nothing signs its outer algorithm. A response that verifies
/// carries the certificate's signed part and the same algorithm after it,
/// byte for byte.
#[test]
fn a_delegated_ocsp_responders_certificate_changed_is_refused_or_says_the_same() {
    use crate::nras_ocsp::tests::{
        bench, der_oid, end_entity, null, respond, seq, with_outer_algorithm, Response, Signer, NOW,
    };
    use crate::nras_ocsp::{verify_response, NrasOcspPolicy};
    use rcgen::{ExtendedKeyUsagePurpose, KeyPair};
    const ECDSA_SHA384: &[u64] = &[1, 2, 840, 10045, 4, 3, 3];
    const ECDSA_SHA256_ARCS: &[u64] = &[1, 2, 840, 10045, 4, 3, 2];
    const SHA256_RSA: &[u64] = &[1, 2, 840, 113549, 1, 1, 11];
    const P384: &[u64] = &[1, 3, 132, 0, 34];

    let b = bench();
    let kp = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).expect("a key");
    let responder = end_entity(
        &b.inter,
        &kp,
        &[7],
        &[ExtendedKeyUsagePurpose::OcspSigning],
        (1970, 4096),
    );
    let (_, leaf) = X509Certificate::from_der(&b.leaf_der).expect("leaf");
    let (_, issuer) = X509Certificate::from_der(&b.inter.der).expect("issuer");
    let policy = NrasOcspPolicy::required();
    let answer = |responder: &[u8]| {
        let r = Response {
            delegated: Some((responder.to_vec(), Signer::of(&kp))),
            ..Response::good_at(NOW)
        };
        respond(&r, &b.leaf_der, &b.inter)
    };
    let verify = |der: &[u8]| verify_response(der, &leaf, &issuer, NOW, &policy);
    let seed = answer(&responder);
    let control = verify(&seed).expect("the delegated responder's answer verifies");

    let signed = signed_part_of(&responder);
    let algorithm = signed_algorithm_of(&responder);
    assert_eq!(algorithm, seq(&[&der_oid(ECDSA_SHA384)]));
    let outer = signed.end..signed.end + algorithm.len();
    assert_eq!(responder[outer.clone()], algorithm[..]);
    let cert_at = seed
        .windows(responder.len())
        .position(|w| w == responder)
        .expect("the responder certificate is in the response");

    let mut c = Campaign::new("OCSP responder certificate", 0x0c5a);
    // One bit of the certificate, in the response as it travels: every bit
    // of the outer algorithm, and of the rest as deep as asked.
    for (at, bit, m) in bit_flips(&seed, cert_at..cert_at + responder.len()) {
        let in_cert = at - cert_at;
        if !outer.contains(&in_cert) && bit != (at * 3) % 8 && crate::hostile::depth() < 10 {
            continue;
        }
        let what = format!("bit {bit} of byte {in_cert} of the responder certificate flipped");
        if let Some(Ok(s)) = c.run(&what, m.len(), || verify(&m)) {
            c.hold(
                s == control,
                "a changed responder certificate says something else",
                &what,
            );
            c.hold(
                !signed.contains(&in_cert) && !outer.contains(&in_cert),
                "a responder certificate verifies with its signed part or its outer algorithm changed",
                &what,
            );
            c.accepted_unchanged();
        }
    }
    // The outer algorithm written in other ways, the response made again
    // around the certificate: none of it is signed.
    for (name, written) in [
        ("with a NULL", seq(&[&der_oid(ECDSA_SHA384), &null()])),
        (
            "with the curve as its parameter",
            seq(&[&der_oid(ECDSA_SHA384), &der_oid(P384)]),
        ),
        (
            "with an empty SEQUENCE as its parameter",
            seq(&[&der_oid(ECDSA_SHA384), &seq(&[])]),
        ),
        ("as ecdsa-with-SHA256", seq(&[&der_oid(ECDSA_SHA256_ARCS)])),
        (
            "as sha256WithRSAEncryption",
            seq(&[&der_oid(SHA256_RSA), &null()]),
        ),
    ] {
        let what = format!("the responder certificate's outer algorithm written {name}");
        let m = answer(&with_outer_algorithm(&responder, &written));
        let r = c.run(&what, m.len(), || verify(&m));
        c.hold(
            matches!(r, Some(Err(_))),
            "a responder certificate verifies with an outer algorithm that is not the signed one",
            &what,
        );
    }
    // What follows the signed part, mutated, the certificate and the
    // response made again around it.
    let tail = &responder[signed.end..];
    for _ in 0..cases(400) {
        let (t, what) = c.mutant(tail, &b.leaf_der);
        let cert = seq(&[&responder[signed.clone()], &t]);
        if X509Certificate::from_der(&cert).is_err() {
            // The responder of the tests reads the certificate to name it.
            continue;
        }
        let what = format!("after the responder certificate's signed part: {what}");
        let m = answer(&cert);
        if let Some(Ok(s)) = c.run(&what, m.len(), || verify(&m)) {
            c.hold(
                s == control,
                "a changed responder certificate says something else",
                &what,
            );
            c.hold(
                repeats(&t, &algorithm),
                "a responder certificate verifies with an outer algorithm that is not the signed one",
                &format!("{what}: {}", show(&t)),
            );
            c.accepted_unchanged();
        }
    }
    c.finish();
}

/// The same rule on the signer certificate of a launch endorsement, under a
/// Google of the tests (the real signer's key is Google's: nothing of a real
/// endorsement changes and still verifies). The endorsement is signed again
/// around each certificate, as a signer would who wrote its own otherwise.
#[test]
fn a_launch_endorsement_whose_signer_certificate_is_written_otherwise_is_refused() {
    use crate::gce_endorsement::tests::fake_google::{
        fake_google, pkcs1_sha256, pss_sha256, pss_sha256_with_its_trailer, pss_sha384,
    };
    use crate::gce_endorsement::{self as gce, tests::NOW};
    use crate::nras_ocsp::tests::seq;
    let g = fake_google();
    let pss = pss_sha256();
    let signer = g.signer_naming(&pss, &pss);
    let verify = |signer: &[u8]| {
        let blob = g.endorse_under(signer, &[0xab; 48], &[[0x5; 48]]);
        gce::verify(&blob, &g.root_der, NOW).map(|e| format!("{e:?}"))
    };
    let control = verify(&signer).expect("the test Google's endorsement verifies");
    let signed = signed_part_of(&signer);
    assert_eq!(signed_algorithm_of(&signer), pss);
    let outer = signed.end..signed.end + pss.len();
    assert_eq!(signer[outer.clone()], pss[..]);

    let mut c = Campaign::new("GCE signer certificate", 0x6cf);
    for (at, bit, m) in bit_flips(&signer, outer.clone()) {
        let what = format!("bit {bit} of byte {at} of the signer certificate flipped");
        let r = c.run(&what, m.len(), || verify(&m));
        c.hold(
            matches!(r, Some(Err(_))),
            "an endorsement verifies under a signer certificate whose outer algorithm changed",
            &what,
        );
    }
    for (name, inside, after) in [
        ("RSASSA-PSS SHA-384 inside", pss_sha384(), pss.clone()),
        (
            "sha256WithRSAEncryption inside",
            pkcs1_sha256(),
            pss.clone(),
        ),
        (
            "its trailer field written out after",
            pss.clone(),
            pss_sha256_with_its_trailer(),
        ),
        (
            "its trailer field written out inside",
            pss_sha256_with_its_trailer(),
            pss.clone(),
        ),
        ("RSASSA-PSS SHA-384 after", pss.clone(), pss_sha384()),
        (
            "RSASSA-PSS SHA-384 on both sides",
            pss_sha384(),
            pss_sha384(),
        ),
        (
            "sha256WithRSAEncryption on both sides",
            pkcs1_sha256(),
            pkcs1_sha256(),
        ),
    ] {
        let what = format!("a signer certificate signed as always, naming {name}");
        let m = g.signer_naming(&inside, &after);
        let r = c.run(&what, m.len(), || verify(&m));
        c.hold(
            matches!(r, Some(Err(_))),
            "an endorsement verifies under a signer certificate that names another algorithm than the one checked",
            &what,
        );
    }
    let tail = &signer[signed.end..];
    for _ in 0..cases(60) {
        let (t, what) = c.mutant(tail, &g.root_der);
        let m = seq(&[&signer[signed.clone()], &t]);
        let what = format!("after the signer certificate's signed part: {what}");
        if let Some(Ok(v)) = c.run(&what, m.len(), || verify(&m)) {
            c.hold(
                v == control,
                "an endorsement under a changed signer certificate endorses something else",
                &what,
            );
            c.hold(
                repeats(&t, &pss),
                "an endorsement verifies under a signer certificate whose outer algorithm changed",
                &format!("{what}: {}", show(&t)),
            );
            c.accepted_unchanged();
        }
    }
    c.finish();
}

/// The same rule on NVIDIA's two real NRAS signing leaves, under the
/// production pins: one bit of a leaf's signed part or of the algorithm after
/// it, and the leaf gives no key; the outer algorithm written another way
/// gives none either.
#[test]
fn nvidias_real_leaves_changed_in_or_after_their_signed_part_give_no_key() {
    use crate::nras::tests::{real_leaf_key, REAL_LEAVES};
    use crate::nras_ocsp::tests::{null, seq, with_outer_algorithm};
    let mut c = Campaign::new("NRAS signing leaf", 0x4a5);
    for (leaf, inter, name) in REAL_LEAVES {
        let control = real_leaf_key(leaf, inter).expect("NVIDIA's leaf gives its key");
        let signed = signed_part_of(leaf);
        let algorithm = signed_algorithm_of(leaf);
        let outer = signed.end..signed.end + algorithm.len();
        assert_eq!(leaf[outer.clone()], algorithm[..], "{name}");
        for (at, bit, m) in bit_flips(leaf, 0..leaf.len()) {
            if !outer.contains(&at) && bit != (at * 3) % 8 && crate::hostile::depth() < 10 {
                continue;
            }
            let what = format!("bit {bit} of byte {at} of the leaf under {name} flipped");
            if let Some(Ok(k)) = c.run(&what, m.len(), || real_leaf_key(&m, inter)) {
                c.hold(k == control, "a changed leaf gives another key", &what);
                c.hold(
                    !signed.contains(&at) && !outer.contains(&at),
                    "a leaf gives a key with its signed part or its outer algorithm changed",
                    &what,
                );
                c.accepted_unchanged();
            }
        }
        // The same OID with a NULL added, or with its NULL dropped.
        let oid = &algorithm[2..4 + usize::from(algorithm[3])];
        let other = if oid.len() == algorithm.len() - 2 {
            seq(&[oid, &null()])
        } else {
            seq(&[oid])
        };
        assert_ne!(other, algorithm, "{name}");
        let what = format!("the outer algorithm of the leaf under {name} written otherwise");
        let m = with_outer_algorithm(leaf, &other);
        let r = c.run(&what, m.len(), || real_leaf_key(&m, inter));
        c.hold(
            matches!(r, Some(Err(_))),
            "a leaf gives a key with an outer algorithm that is not the signed one",
            &what,
        );
        let tail = &leaf[signed.end..];
        for _ in 0..cases(300) {
            let (t, what) = c.mutant(tail, inter);
            let m = seq(&[&leaf[signed.clone()], &t]);
            let what = format!("after the signed part of the leaf under {name}: {what}");
            if let Some(Ok(k)) = c.run(&what, m.len(), || real_leaf_key(&m, inter)) {
                c.hold(k == control, "a changed leaf gives another key", &what);
                c.hold(
                    repeats(&t, &algorithm),
                    "a leaf gives a key with an outer algorithm that is not the signed one",
                    &format!("{what}: {}", show(&t)),
                );
                c.accepted_unchanged();
            }
        }
    }
    c.finish();
}

/// The count and the sizes of an OVMF `ASEV` descriptor are the file's: a
/// firmware is measured within the bounds of `snp_launch` and refused past
/// them, at once. Before the bounds, the last case hashed 4 GiB of pages per
/// section for as many sections as the file holds.
#[test]
fn an_ovmf_descriptor_past_the_bounds_is_refused_at_once() {
    use crate::snp_launch::tests::{fixture_listing, LISTING_AT, LISTING_ROOM};
    use crate::snp_launch::{launch_digest, Launch, Ovmf, MAX_SECTIONS, MAX_SECTION_MEMORY};
    const MEMORY: u32 = 1;
    let digest = |b: &[u8]| {
        launch_digest(&Launch {
            ovmf: b,
            kernel: None,
            initrd: None,
            append: None,
            vcpus: 2,
            vcpu_sig: 0x00a1_0f10,
            guest_features: 1,
        })
    };
    let page = |i: usize| (0x0080_0000 + 0x1000 * i as u32, 0x1000, MEMORY);
    let room = LISTING_ROOM as u32;
    let largest = u32::try_from(MAX_SECTION_MEMORY).expect("under 4 GiB");

    let mut c = Campaign::new("OVMF descriptor bounds", 0x0f40);
    for count in [
        0,
        1,
        7,
        MAX_SECTIONS - 1,
        MAX_SECTIONS,
        MAX_SECTIONS + 1,
        0xff,
        0x100,
        0xffff,
        room - 1,
        room,
        room + 1,
        0x7fff_ffff,
        0x8000_0000,
        0xffff_ffff,
    ] {
        let what = format!("a descriptor of {count} sections of one page");
        let m = fixture_listing(count, page);
        let r = c.run(&what, m.len(), || digest(&m));
        c.hold(
            matches!(r, Some(Ok(_))) == (count <= MAX_SECTIONS),
            "a count of sections is not judged by the bound",
            &what,
        );
    }
    for size in [
        0,
        1,
        0xfff,
        0x1000,
        0x1001,
        0x1_0000,
        largest - 1,
        largest,
        largest + 1,
        largest + 0x1000,
        0x7fff_ffff,
        0x8000_0000,
        0xffff_f000,
        0xffff_ffff,
    ] {
        for count in [1, MAX_SECTIONS] {
            // The most the bounds let through, a million pages, is measured
            // once: under `largest`.
            if count == MAX_SECTIONS && size == largest - 1 {
                continue;
            }
            let what = format!("{count} sections of {size:#x} bytes each");
            let m = fixture_listing(count, |_| (0x0080_0000, size, MEMORY));
            let r = c.run(&what, m.len(), || digest(&m));
            c.hold(
                matches!(r, Some(Ok(_))) == (size <= largest),
                "a section's size is not judged by the bound",
                &what,
            );
        }
    }
    // As many sections as the file holds, each of the largest size a u32
    // says, under each count that reads them all.
    for count in [room, 0xffff_ffff] {
        let what = format!("{count} sections of 0xfffff000 bytes each");
        let m = fixture_listing(count, |_| (0, 0xffff_f000, MEMORY));
        let r = c.run(&what, m.len(), || digest(&m));
        c.hold(
            matches!(r, Some(Err(_))),
            "a firmware past the bounds is measured",
            &what,
        );
    }
    // The descriptor mutated: what is still measured is within the bounds.
    let seed = fixture_listing(7, page);
    let region = LISTING_AT..LISTING_AT + 16 + 12 * 8;
    for _ in 0..cases(1500) {
        let (part, what) = c.mutant(&seed[region.clone()], &seed[seed.len() - 0x100..]);
        let mut m = seed.clone();
        let n = part.len().min(region.len());
        m[region.start..region.start + n].copy_from_slice(&part[..n]);
        let what = format!("the descriptor: {what}");
        if let Some(Ok(_)) = c.run(&what, m.len(), || digest(&m)) {
            let o = Ovmf::parse(&m).expect("measured, so parsed");
            c.hold(
                o.sections().len() <= MAX_SECTIONS as usize
                    && o.sections()
                        .iter()
                        .all(|s| u64::from(s.size) <= MAX_SECTION_MEMORY),
                "a firmware past the bounds is measured",
                &format!("{what}: {}", show(&m[region.clone()])),
            );
        }
    }
    c.finish();
}

#[test]
fn sealed_and_tabled_blobs_cut_or_changed_never_panic_and_never_open() {
    use crate::{evidence, hpke, seal};
    let mut c = Campaign::new("sealed blobs and tables", 0x5ea1);

    // A sealed model key: AEAD, so any change is refused.
    let key = [0x42u8; 32];
    let sealed = seal::seal(b"thirty-two bytes of a model key!", &key, &[9u8; 12]).expect("seals");
    assert!(seal::unseal(&sealed, &key).is_ok());
    for p in prefixes(&sealed) {
        let what = format!("the first {} bytes of a sealed blob", p.len());
        let r = c.run(&what, p.len(), || seal::unseal(p, &key).is_ok());
        c.hold(r != Some(true), "a cut sealed blob opens", &what);
    }
    for (at, bit, m) in bit_flips(&sealed, 0..sealed.len()) {
        let what = format!("bit {bit} of sealed byte {at} flipped");
        let r = c.run(&what, m.len(), || seal::unseal(&m, &key).is_ok());
        c.hold(r != Some(true), "a changed sealed blob opens", &what);
    }
    for _ in 0..cases(2000) {
        let (m, what) = c.mutant(&sealed, &key);
        let r = c.run(&what, m.len(), || seal::unseal(&m, &key).is_ok());
        c.hold(
            r != Some(true) || m == sealed,
            "a mutated sealed blob opens",
            &what,
        );
    }

    // HPKE, the relay's channel to the TD: the encapsulated key, then the
    // ciphertext and its tag.
    let (sk, pk) = hpke::derive_key_pair(&[7u8; 32]);
    let (enc, ct) =
        hpke::seal(&pk, b"info", b"aad", b"a model key on its way", &[3u8; 32]).expect("seals");
    assert!(hpke::open(&sk, &enc, b"info", b"aad", &ct).is_ok());
    for p in prefixes(&ct) {
        let what = format!("the first {} bytes of an HPKE ciphertext", p.len());
        let r = c.run(&what, p.len(), || {
            hpke::open(&sk, &enc, b"info", b"aad", p).is_ok()
        });
        c.hold(r != Some(true), "a cut HPKE ciphertext opens", &what);
    }
    for (at, bit, m) in bit_flips(&ct, 0..ct.len()) {
        let what = format!("bit {bit} of HPKE ciphertext byte {at} flipped");
        let r = c.run(&what, m.len(), || {
            hpke::open(&sk, &enc, b"info", b"aad", &m).is_ok()
        });
        c.hold(r != Some(true), "a changed HPKE ciphertext opens", &what);
    }
    for (at, bit, m) in bit_flips(&enc, 0..enc.len()) {
        let what = format!("bit {bit} of HPKE encapsulated-key byte {at} flipped");
        let mut e = enc;
        e.copy_from_slice(&m);
        let r = c.run(&what, m.len(), || {
            hpke::open(&sk, &e, b"info", b"aad", &ct).is_ok()
        });
        // X25519 ignores the top bit of the last byte (RFC 7748, section 5).
        c.hold(
            r != Some(true) || (at, bit) == (31, 7),
            "a changed HPKE encapsulated key opens",
            &what,
        );
    }

    // The certificate table the SNP guest device returns: GUID, offset, length.
    const VCEK_GUID: &str = "63da758d-e664-4564-adc5-f4b93be8accd";
    let mut table = Vec::new();
    table.extend_from_slice(&[
        0x8d, 0x75, 0xda, 0x63, 0x64, 0xe6, 0x64, 0x45, 0xad, 0xc5, 0xf4, 0xb9, 0x3b, 0xe8, 0xac,
        0xcd,
    ]);
    table.extend_from_slice(&48u32.to_le_bytes());
    table.extend_from_slice(&16u32.to_le_bytes());
    table.extend_from_slice(&[0u8; 24]);
    table.extend_from_slice(&[0xc5; 16]);
    for p in prefixes(&table) {
        let what = format!("the first {} bytes of a certificate table", p.len());
        c.run(&what, p.len(), || {
            let _ = evidence::snp_cert_from_table(p, VCEK_GUID);
        });
    }
    for _ in 0..cases(4000) {
        let (m, what) = c.mutant(&table, &sealed);
        let what = format!("certificate table {}: {what}", show(&m));
        let r = c.run(&what, m.len(), || {
            evidence::snp_cert_from_table(&m, VCEK_GUID).map(|c| c.map(<[u8]>::len))
        });
        if let Some(Ok(Some(n))) = r {
            c.hold(n <= m.len(), "a certificate longer than its table", &what);
        }
    }
    c.finish();
}
