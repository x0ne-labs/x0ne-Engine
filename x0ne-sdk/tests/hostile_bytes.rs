//! Hostile bytes against what a client of this crate is handed by a party it
//! does not trust: the quote, the event log and Intel's collateral that
//! `attest::verify_quote` judges together, an OHTTP gateway's key
//! configuration, sealed requests and answers, binary HTTP messages.
//!
//! The statements, the mutations and the bounds are those of
//! `tests/support/hostile.rs`; x0ne-infer's `hostile_bytes_tests.rs` holds
//! each parser alone to them. Here the whole verdict is held: a mutated input
//! is refused, or the verdict is the intact one's, field for field.

#[path = "support/hostile.rs"]
mod hostile;

use base64::Engine as _;
use hostile::{bit_flips, cases, depth, flipped, lengths, prefixes, Campaign};
use sha2::{Digest, Sha256, Sha512};
use x0ne_sdk::attest::{self, Collateral, Policy, SignedReference};
use x0ne_sdk::bhttp;
use x0ne_sdk::hpke::PrivateKey;
use x0ne_sdk::ohttp::{self, KeyConfig};

const QUOTE: &[u8] =
    include_bytes!("../../x0ne-infer/tests/run_fixtures/gce-a3-tdx-2026-10-18.quote");
const CCEL: &[u8] =
    include_bytes!("../../x0ne-infer/tests/run_fixtures/gce-a3-tdx-2026-10-18.ccel");
const OTHER_QUOTE: &[u8] = include_bytes!("../../x0ne-infer/tests/pcs_live/quote.dat");
const REFERENCE_PINNED: &[u8] =
    include_bytes!("../../x0ne-infer/tests/run_fixtures/x0ne-cvm-0.2.3-rtmr0-pinned.json");
const FIRMWARE_SHA384: &str = "53c558fe06714203d7428484c46ee7c0b9df748a9789e173421c28d097115b30efb8476304617532488ab40c3122db88";
const CFV_SHA384: &str = "4e1221cf6f7815beec39a994bff99142ed305210e2735b1134fc8ef363923cbdcca91fa71e2e25ee8e689fb7d2c075bf";
const MACHINE: &str = "a3-highgpu-1g";
const ENDORSEMENT: &[u8] =
    include_bytes!("../../x0ne-infer/tests/gce_endorsement_tdx_c1ee9c16.binarypb");
const TCB: &str =
    include_str!("../../x0ne-infer/tests/pcs_live/2026-10-22/tcb_info_00806f050000_early.json");
const QE: &str = include_str!("../../x0ne-infer/tests/pcs_live/2026-10-22/qe_identity_early.json");
const CRL: &[u8] =
    include_bytes!("../../x0ne-infer/tests/pcs_live/2026-10-22/pck_crl_platform.der");
const TCB_CHAIN: &str = include_str!("../../x0ne-infer/tests/pcs_live/tcb_issuer_chain.pem");
const QE_CHAIN: &str = include_str!("../../x0ne-infer/tests/pcs_live/qe_issuer_chain.pem");
const ROOT_CA_CRL: &[u8] =
    include_bytes!("../../x0ne-infer/tests/pcs_live/2026-10-23/root_ca_crl.der");
/// 2026-10-22, the day the collateral was fetched.
const NOW: u64 = 1_792_666_540;

// What the run's quote binds (`tests/real_quote_2026_10_18.rs`).
const SESSION_NONCE: &str = "0e4309d4f93f2f8b74bdc1bd4485ef5b37d83778aabad6e08e6d2caa56d0ddd5";
const SESSION_PUBKEY: &str = "96dbaeb08eca9927f7816c66ceaa0c0047fdbeced5b87b7098273f85f8fcf58b";
const TLS_SPKI_SHA256: &str = "10220d5ab17f39e4a3782aa2c44c986799886aec5750c0c310e6a831fdfe4538";
const CONFIG: &str = r#"{"schema":"x0ne-engine/cloud-config/v1","model":"qwen3:32b","client_spki_sha256":"55cd175f360c4ffc1a784d79cf991b95c6f03b637c9ee08ff4102a7116e0c9a6","client_nonce":"04025646154239a270373a4a6cfac613fd918844793a0824cfab93326a4f077e","expires_at_unix":1792348209,"attest":"cpu+gpu","listen_port":443,"max_completion_tokens":4096}"#;

/// Header and TD report body: what the attestation key signs.
const SIGNED: usize = 632;

fn unhex<const N: usize>(s: &str) -> [u8; N] {
    let v: Vec<u8> = (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect();
    v.try_into().unwrap()
}

fn unhex_vec(s: &str) -> Vec<u8> {
    let s: String = s.split_whitespace().collect();
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn b64(b: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(b)
}

fn run_report_data() -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(b"x0ne-infer/v2");
    h.update(unhex::<32>(SESSION_NONCE));
    h.update(unhex::<32>(SESSION_PUBKEY));
    h.update(unhex::<32>(TLS_SPKI_SHA256));
    h.update(Sha256::digest(CONFIG.as_bytes()));
    h.finalize().into()
}

fn collateral() -> Collateral {
    Collateral {
        tcb_info: TCB.into(),
        tcb_info_chain_pem: TCB_CHAIN.into(),
        qe_identity: QE.into(),
        qe_identity_chain_pem: QE_CHAIN.into(),
        pck_crl_der_b64: b64(CRL),
        root_ca_crl_der_b64: b64(ROOT_CA_CRL),
        gce_endorsement_b64: b64(ENDORSEMENT),
        ..Collateral::default()
    }
}

fn policy() -> Policy {
    let (signature, tlog_proof) = attest::sign_reference_under_test_keys(REFERENCE_PINNED, NOW);
    let reference = attest::admit_reference_under_test_keys(
        &SignedReference {
            json: REFERENCE_PINNED,
            signature: &signature,
            tlog_proof: &tlog_proof,
        },
        &mut attest::LogMemory::new(),
        None,
        NOW,
    )
    .expect("signed and logged");
    Policy::production(reference, MACHINE)
        .with_strict_tcb()
        .with_firmware_measured_elsewhere(attest::FirmwareMeasuredElsewhere {
            sha384_hex: FIRMWARE_SHA384.into(),
            cfv_sha384: unhex(CFV_SHA384),
        })
}

#[test]
fn the_run_s_evidence_cut_or_changed_is_refused_or_gives_the_verdict_of_the_intact_one() {
    let (policy, col, bound) = (policy(), collateral(), run_report_data());
    let verify = |quote: &[u8], ccel: &[u8], col: &Collateral| {
        attest::verify_quote(quote, ccel, &bound, col, &policy, NOW)
    };
    let control = verify(QUOTE, CCEL, &col).expect("the run's evidence verifies");
    // The PCK's fingerprint is the digest of the whole certificate, whose
    // bytes after its signed part can be written in more than one way: it
    // is left out of the comparison.
    let same = |v: &attest::VerifiedQuote| {
        let mut v = v.clone();
        v.pck_sha256_hex.clone_from(&control.pck_sha256_hex);
        v == control
    };

    let mut c = Campaign::new("the whole verdict", 0x5d6);
    // Every bit of the quote's header, one bit of every byte of its TD
    // report (every bit from X0NE_FUZZ_ITERS=10): all under the signature.
    for at in 0..SIGNED {
        for bit in 0..8 {
            if at >= 48 && depth() < 10 && bit != (at * 3 + 5) % 8 {
                continue;
            }
            let m = flipped(QUOTE, at, bit);
            let what = format!("bit {bit} of signed quote byte {at} flipped");
            let r = c.run(&what, m.len(), || verify(&m, CCEL, &col));
            c.hold(
                matches!(r, Some(Err(_))),
                "a quote with a changed signed byte is believed",
                &what,
            );
        }
    }
    let chain_ends = QUOTE
        .windows(8)
        .rposition(|w| w == b"-----END")
        .expect("a PEM chain");
    for n in lengths(QUOTE.len(), 97, 4) {
        let what = format!("the first {n} bytes of the quote");
        if let Some(Ok(v)) = c.run(&what, n, || verify(&QUOTE[..n], CCEL, &col)) {
            c.hold(
                n > chain_ends,
                "a quote cut inside its PCK chain is believed",
                &what,
            );
            c.hold(same(&v), "a cut quote gives another verdict", &what);
            c.accepted_unchanged();
        }
    }
    for n in lengths(CCEL.len(), 331, 4) {
        let what = format!("the first {n} bytes of the event log");
        if let Some(Ok(v)) = c.run(&what, n, || verify(QUOTE, &CCEL[..n], &col)) {
            c.hold(v == control, "a cut event log gives another verdict", &what);
            c.accepted_unchanged();
        }
    }
    for _ in 0..cases(120) {
        let (m, what) = c.mutant(QUOTE, OTHER_QUOTE);
        let changed_signed = m.get(..SIGNED) != Some(&QUOTE[..SIGNED]);
        if let Some(Ok(v)) = c.run(&what, m.len(), || verify(&m, CCEL, &col)) {
            c.hold(
                !changed_signed,
                "a quote with changed signed bytes is believed",
                &what,
            );
            c.hold(same(&v), "a mutated quote gives another verdict", &what);
            c.accepted_unchanged();
        }
        let (m, what) = c.mutant(CCEL, QUOTE);
        let what = format!("event log: {what}");
        if let Some(Ok(v)) = c.run(&what, m.len(), || verify(QUOTE, &m, &col)) {
            c.hold(
                v == control,
                "a mutated event log gives another verdict",
                &what,
            );
            c.accepted_unchanged();
        }
    }
    // Each answer of Intel and Google, alone.
    type Field = fn(&mut Collateral) -> &mut String;
    let fields: [(&str, Field); 7] = [
        ("tcbInfo", |c| &mut c.tcb_info),
        ("tcbInfo issuer chain", |c| &mut c.tcb_info_chain_pem),
        ("QE identity", |c| &mut c.qe_identity),
        ("QE identity issuer chain", |c| &mut c.qe_identity_chain_pem),
        ("PCK CRL", |c| &mut c.pck_crl_der_b64),
        ("root CA CRL", |c| &mut c.root_ca_crl_der_b64),
        ("launch endorsement", |c| &mut c.gce_endorsement_b64),
    ];
    for (name, field) in fields {
        let intact = field(&mut col.clone()).clone();
        for _ in 0..cases(40) {
            let mut mutated = col.clone();
            let what = if c.rng.below(3) == 0 && !intact.starts_with(['{', '-']) {
                // The bytes under the base64, not its spelling.
                let raw = base64::engine::general_purpose::STANDARD
                    .decode(&intact)
                    .expect("base64");
                let (m, what) = c.mutant(&raw, QUOTE);
                *field(&mut mutated) = b64(&m);
                what
            } else {
                let (m, what) = c.text_mutant(&intact, TCB_CHAIN);
                *field(&mut mutated) = m;
                what
            };
            let what = format!("{name}: {what}");
            if let Some(Ok(v)) = c.run(&what, intact.len(), || verify(QUOTE, CCEL, &mutated)) {
                c.hold(
                    v == control,
                    "mutated collateral gives another verdict",
                    &what,
                );
                c.accepted_unchanged();
            }
        }
        let mut emptied = col.clone();
        field(&mut emptied).clear();
        let what = format!("{name}: absent");
        let r = c.run(&what, 0, || verify(QUOTE, CCEL, &emptied));
        c.hold(
            matches!(r, Some(Err(_))),
            "evidence is believed without a piece of its collateral",
            &what,
        );
    }
    c.finish();
}

// RFC 9458 and draft-ietf-ohai-chunked-ohttp, appendix A (`tests/chunked_ohttp.rs`).
const SK_R: &str = "1c190d72acdbe4dbc69e680503bb781a932c70a12c8f3754434c67d8640d8698";
const KEY_CONFIG: &str = "010020668eb21aace159803974a4c67f08b4152d29bed10735fd08f98ccdd6fe
                          09570800080001000100010003";
const REQUEST: &str = "00034745540568747470730b6578616d706c652e636f6d012f";
const SK_E: &str = "b26d565f3f875ed480d1abced3d665159650c99174fd0b124ac4bda0c64ae324";
const ENCAPSULATED_REQUEST: &str = "01002000010001
    8811eb457e100811c40a0aa71340a1b81d804bb986f736f2f566a7199761a032
    1c2ad24942d4d692563012f2980c8fef437a336b9b2fc938ef77a5834f
    1d2e33d8fd25577afe31bd1c79d094f76b6250ae6549b473ecd950501311
    001c6c1395d0ef7c1022297966307b8a7f";

#[test]
fn a_key_configuration_or_a_sealed_request_cut_or_changed_does_not_open() {
    let mut c = Campaign::new("OHTTP", 0x0477);
    let key_config = unhex_vec(KEY_CONFIG);
    let intact = KeyConfig::parse_any_suite(&key_config).expect("the draft's key configuration");
    for p in prefixes(&key_config) {
        let what = format!("the first {} bytes of a key configuration", p.len());
        let r = c.run(&what, p.len(), || {
            (
                KeyConfig::parse(p).is_ok(),
                KeyConfig::parse_any_suite(p).is_ok(),
            )
        });
        c.hold(
            r == Some((false, false)),
            "a cut key configuration is read",
            &what,
        );
    }
    let own = intact.encode();
    assert_eq!(KeyConfig::parse(&own), Ok(intact.clone()));
    for _ in 0..cases(8000) {
        let seed = if c.rng.below(2) == 0 {
            &key_config
        } else {
            &own
        };
        let (m, what) = c.mutant(seed, &own);
        let r = c.run(&what, m.len(), || {
            (KeyConfig::parse(&m), KeyConfig::parse_any_suite(&m))
        });
        if let Some((strict, any)) = r {
            if let Ok(k) = &strict {
                c.hold(
                    KeyConfig::parse(&k.encode()).as_ref() == Ok(k),
                    "a key configuration does not read back the same",
                    &what,
                );
                c.hold(
                    any.as_ref() == Ok(k),
                    "the two readers disagree on a key",
                    &what,
                );
            }
        }
    }

    // A sealed request: the header, the encapsulated key, the chunks.
    let sk = PrivateKey::from_bytes(unhex(SK_R));
    let sealed = unhex_vec(ENCAPSULATED_REQUEST);
    let open = |b: &[u8]| {
        ohttp::decapsulate_chunked_request(&sk, 1, b, 1 << 20)
            .map(|(request, _)| request.to_vec())
            .ok()
    };
    let request = open(&sealed).expect("the draft's request opens");
    for p in prefixes(&sealed) {
        let what = format!("the first {} bytes of a chunked request", p.len());
        let r = c.run(&what, p.len(), || open(p));
        c.hold(r == Some(None), "a cut chunked request opens", &what);
    }
    for (at, bit, m) in bit_flips(&sealed, 0..sealed.len()) {
        let what = format!("bit {bit} of chunked request byte {at} flipped");
        let r = c.run(&what, m.len(), || open(&m));
        // X25519 ignores the top bit of the encapsulated key's last byte
        // (RFC 7748, section 5): the same key, the same request.
        let same_key = (at, bit) == (7 + 31, 7);
        c.hold(
            r == Some(None) || (same_key && r == Some(Some(request.clone()))),
            "a changed chunked request opens",
            &what,
        );
    }
    // The single-shot form, sealed here.
    let (single, _) =
        ohttp::encapsulate_request(&intact, b"a whole request", &[5u8; 32]).expect("seals");
    let open_single = |b: &[u8]| {
        ohttp::decapsulate_request(&sk, 1, b)
            .map(|(request, _)| request.to_vec())
            .ok()
    };
    assert_eq!(
        open_single(&single).as_deref(),
        Some(&b"a whole request"[..])
    );
    for p in prefixes(&single) {
        let what = format!("the first {} bytes of a request", p.len());
        let r = c.run(&what, p.len(), || open_single(p));
        c.hold(r == Some(None), "a cut request opens", &what);
    }
    for _ in 0..cases(3000) {
        let (m, what) = c.mutant(&sealed, &single);
        let r = c.run(&what, m.len(), || (open(&m), open_single(&m)));
        if let Some((chunked, whole)) = r {
            c.hold(
                chunked.is_none() || chunked == Some(request.clone()),
                "a mutated chunked request opens to another request",
                &what,
            );
            c.hold(
                // A splice can be the donor, whole.
                whole.is_none() || m == single,
                "a chunked request opens as a whole one",
                &what,
            );
        }
        let (m, what) = c.mutant(&single, &sealed);
        let r = c.run(&what, m.len(), || (open_single(&m), open(&m)));
        if let Some((whole, chunked)) = r {
            c.hold(
                whole.is_none() || whole.as_deref() == Some(&b"a whole request"[..]),
                "a mutated request opens to another request",
                &what,
            );
            c.hold(
                chunked.is_none() || m == sealed,
                "a whole request opens as a chunked one",
                &what,
            );
        }
    }

    // The chunked answer, read as the network hands it.
    let (_, server) = ohttp::decapsulate_chunked_request(&sk, 1, &sealed, 1 << 20).expect("opens");
    let mut w = server.writer(&[0x11; 16]).expect("a writer");
    let mut answer = w.chunk(b"first").expect("chunk");
    answer.extend(w.chunk(b"second chunk").expect("chunk"));
    answer.extend(w.last(b"last").expect("last"));
    let read = |wire: &[u8], cut: usize| {
        // The reader's secret is the request's: the draft's request, sealed
        // again under the draft's ephemeral key for each read.
        let request = unhex_vec(REQUEST);
        let (_, client) = ohttp::encapsulate_chunked_request_with_ephemeral(
            &intact,
            &[&request[..12], &request[12..], b""],
            &PrivateKey::from_bytes(unhex(SK_E)),
        )
        .ok()?;
        let mut reader = client.reader();
        let (a, b) = wire.split_at(cut.min(wire.len()));
        let mut out: Vec<Vec<u8>> = Vec::new();
        for piece in [a, b] {
            out.extend(reader.push(piece).ok()?.into_iter().map(|z| z.to_vec()));
        }
        out.push(reader.finish().ok()?.to_vec());
        Some(out)
    };
    let whole = read(&answer, answer.len());
    assert_eq!(
        whole,
        Some(vec![
            b"first".to_vec(),
            b"second chunk".to_vec(),
            b"last".to_vec()
        ]),
        "the answer opens under the draft's request"
    );
    for cut in 0..=answer.len() {
        let what = format!("the answer handed in two pieces, cut at byte {cut}");
        let r = c.run(&what, answer.len(), || read(&answer, cut));
        c.hold(
            r == Some(whole.clone()),
            "an answer reads otherwise when cut elsewhere",
            &what,
        );
        let what = format!("the first {cut} bytes of the answer");
        if cut < answer.len() {
            let r = c.run(&what, cut, || read(&answer[..cut], cut));
            c.hold(
                r == Some(None),
                "a cut answer is read as a whole one",
                &what,
            );
        }
    }
    for _ in 0..cases(4000) {
        let (m, what) = c.mutant(&answer, &sealed);
        let cut = c.rng.below(m.len() + 1);
        let r = c.run(&what, m.len(), || read(&m, cut));
        c.hold(
            // A chunk's length is a QUIC varint, which has longer spellings
            // of one value (RFC 9000, section 16): the same chunks.
            r == Some(None) || r == Some(whole.clone()),
            "a mutated answer is read as another answer",
            &what,
        );
    }
    c.finish();
}

#[test]
fn a_binary_http_message_cut_or_changed_never_panics_and_reads_back_the_same() {
    let request = bhttp::Request {
        method: "POST".into(),
        scheme: "https".into(),
        authority: "td.example".into(),
        path: "/v1/chat/completions".into(),
        headers: vec![
            ("content-type".into(), "application/json".into()),
            ("x-request-id".into(), "0123456789abcdef".into()),
        ],
        content: br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#.to_vec(),
    };
    let response = bhttp::Response {
        status: 200,
        headers: vec![("content-type".into(), "application/json".into())],
        content: br#"{"choices":[{"message":{"content":"hello"}}]}"#.to_vec(),
    };
    let (req, resp) = (request.encode(0), response.encode(0));
    assert_eq!(bhttp::Request::decode(&req), Ok(request.clone()));
    assert_eq!(bhttp::Response::decode(&resp), Ok(response.clone()));

    let mut c = Campaign::new("binary HTTP", 0xb477);
    // RFC 9292, section 3.8: a message cut at the end of a section is one
    // whose later sections are empty. The last byte is the empty trailer
    // section: without it the message is the same. What a cut before it
    // never does is read as the whole message.
    for p in prefixes(&req[..req.len() - 1]) {
        let what = format!("the first {} bytes of a request", p.len());
        let r = c.run(&what, p.len(), || bhttp::Request::decode(p));
        c.hold(
            r != Some(Ok(request.clone())),
            "a cut request reads as the whole one",
            &what,
        );
    }
    for p in prefixes(&resp[..resp.len() - 1]) {
        let what = format!("the first {} bytes of a response", p.len());
        let r = c.run(&what, p.len(), || bhttp::Response::decode(p));
        c.hold(
            r != Some(Ok(response.clone())),
            "a cut response reads as the whole one",
            &what,
        );
    }
    for _ in 0..cases(8000) {
        let (m, what) = c.mutant(&req, &resp);
        if let Some(Ok(r)) = c.run(&what, m.len(), || bhttp::Request::decode(&m)) {
            let again = bhttp::Request::decode(&r.encode(0));
            c.hold(
                again == Ok(r),
                "a request does not read back the same",
                &what,
            );
        }
        let (m, what) = c.mutant(&resp, &req);
        if let Some(Ok(r)) = c.run(&what, m.len(), || bhttp::Response::decode(&m)) {
            let again = bhttp::Response::decode(&r.encode(0));
            c.hold(
                again == Ok(r),
                "a response does not read back the same",
                &what,
            );
        }
    }
    // The indeterminate-length response of a stream, in pieces.
    let mut stream = bhttp::indeterminate::head(200, &[("content-type", "text/event-stream")]);
    bhttp::indeterminate::chunk(&mut stream, b"data: one\n\n");
    bhttp::indeterminate::chunk(&mut stream, b"data: [DONE]\n\n");
    stream.extend_from_slice(&bhttp::indeterminate::END);
    let read = |wire: &[u8], cut: usize| {
        let mut reader = bhttp::ResponseReader::new(1 << 16);
        let (a, b) = wire.split_at(cut.min(wire.len()));
        let mut head = None;
        let mut content = Vec::new();
        for piece in [a, b] {
            for part in reader.push(piece)? {
                match part {
                    bhttp::Part::Head { status, headers } => head = Some((status, headers)),
                    bhttp::Part::Content(bytes) => content.extend(bytes),
                }
            }
        }
        reader.finish()?;
        Ok::<_, bhttp::BhttpError>((head, content))
    };
    let whole = read(&stream, stream.len());
    assert!(
        matches!(&whole, Ok((Some((200, _)), content)) if content.len() == 25),
        "the stream reads: {whole:?}"
    );
    for cut in 0..=stream.len() {
        let what = format!("the stream handed in two pieces, cut at byte {cut}");
        let r = c.run(&what, stream.len(), || read(&stream, cut));
        c.hold(
            r.as_ref() == Some(&whole),
            "a stream reads otherwise when cut elsewhere",
            &what,
        );
    }
    for _ in 0..cases(6000) {
        let (m, what) = c.mutant(&stream, &resp);
        let cut = c.rng.below(m.len() + 1);
        let at_once = c.run(&what, m.len(), || read(&m, m.len()));
        let in_two = c.run(&what, m.len(), || read(&m, cut));
        c.hold(
            at_once.as_ref().map(|r| r.as_ref().ok()) == in_two.as_ref().map(|r| r.as_ref().ok()),
            "a mutated stream reads otherwise when cut elsewhere",
            &format!("{what}, cut at byte {cut}"),
        );
    }
    c.finish();
}
