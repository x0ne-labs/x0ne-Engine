//! The two ends of the encrypted path without the attestation (that half is
//! tested against the simulated TD in x0ne-infer, `e2e_sdk_tests.rs`):
//! padding, the gateway's refusals, the binding's inputs, Binary HTTP bounds.

use x0ne_sdk::bhttp::{self, BhttpError};
use x0ne_sdk::e2e::{self, E2eError, Gateway};
use x0ne_sdk::ohttp::{self, KeyConfig};

const SEED: [u8; 32] = [7; 32];
/// The caller's clock when it seals (Unix seconds).
const NOW: u64 = 1_800_000_000;

fn config(g: &Gateway) -> KeyConfig {
    KeyConfig::parse(&g.key_config()).unwrap()
}

fn chat(text: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "model": "qwen3:0.6b",
        "messages": [{ "role": "user", "content": text }],
    }))
    .unwrap()
}

#[test]
fn a_question_and_its_answer_cross_the_gateway() {
    let g = Gateway::from_seed(&SEED);
    let inner = e2e::chat_request(&chat("what is 2+2?"), NOW).unwrap();
    let (request, client) = ohttp::encapsulate_request(&config(&g), &inner, &[1; 32]).unwrap();
    let (opened, responder) = g.open(&request).unwrap();
    assert_eq!(*opened.body, chat("what is 2+2?"));
    assert_eq!(opened.sent_at, NOW);
    let response = responder.seal(200, b"{\"answer\":4}", &[9; 16]).unwrap();
    let inner = client.decapsulate(&response).unwrap();
    let resp = bhttp::Response::decode(&inner).unwrap();
    assert_eq!(
        (resp.status, resp.content.as_slice()),
        (200, &b"{\"answer\":4}"[..])
    );
}

/// SECURITY: the carrier sees one size per bucket, not the question's length.
#[test]
fn questions_of_different_lengths_in_one_bucket_have_one_size_on_the_wire() {
    let g = Gateway::from_seed(&SEED);
    let short = e2e::chat_request(&chat("hi"), NOW).unwrap();
    let long = e2e::chat_request(&chat(&"x".repeat(3000)), NOW).unwrap();
    assert_eq!(short.len(), e2e::MIN_PADDED);
    assert_eq!(long.len(), e2e::MIN_PADDED);
    let (a, _) = ohttp::encapsulate_request(&config(&g), &short, &[1; 32]).unwrap();
    let (b, _) = ohttp::encapsulate_request(&config(&g), &long, &[2; 32]).unwrap();
    assert_eq!(a.len(), b.len());
    let bigger = e2e::chat_request(&chat(&"x".repeat(5000)), NOW).unwrap();
    assert_eq!(bigger.len(), 2 * e2e::MIN_PADDED);
    // Answers too.
    let (_, r1) = g.open(&a).unwrap();
    let (_, r2) = g.open(&b).unwrap();
    assert_eq!(
        r1.seal(200, b"4", &[3; 16]).unwrap().len(),
        r2.seal(200, &[b'a'; 3000], &[3; 16]).unwrap().len()
    );
}

#[test]
fn the_buckets_are_powers_of_two_from_4_kib_to_4_mib() {
    assert_eq!(e2e::padded_len(0).unwrap(), 4096);
    assert_eq!(e2e::padded_len(4096).unwrap(), 4096);
    assert_eq!(e2e::padded_len(4097).unwrap(), 8192);
    assert_eq!(e2e::padded_len(4 << 20).unwrap(), 4 << 20);
    assert_eq!(e2e::MAX_PADDED, 4 << 20);
    assert!(matches!(
        e2e::padded_len((4 << 20) + 1),
        Err(E2eError::TooLarge(_))
    ));
}

/// The largest sealed request is the 4 MiB bucket sealed, 55 bytes more, and
/// the client seals nothing larger: what the TD's route reads at most.
#[test]
fn the_largest_sealed_request_is_the_largest_bucket_sealed() {
    assert_eq!(e2e::MAX_SEALED_REQUEST, (4 << 20) + 55);
    let g = Gateway::from_seed(&SEED);
    let at = e2e::chat_request(&chat(&"a".repeat(3 << 20)), NOW).unwrap();
    assert_eq!(at.len(), e2e::MAX_PADDED);
    let (sealed, _) = ohttp::encapsulate_request(&config(&g), &at, &[1; 32]).unwrap();
    assert_eq!(sealed.len(), e2e::MAX_SEALED_REQUEST);
    assert!(g.open(&sealed).is_ok());
    assert!(matches!(
        e2e::chat_request(&chat(&"a".repeat(4 << 20)), NOW),
        Err(E2eError::TooLarge(_))
    ));
}

/// An unpadded message, sealed by something other than this SDK, is refused:
/// its length would say more than a bucket.
#[test]
fn the_gateway_refuses_an_unpadded_request() {
    let g = Gateway::from_seed(&SEED);
    let req = bhttp::Request {
        method: "POST".into(),
        scheme: "https".into(),
        authority: String::new(),
        path: e2e::CHAT_PATH.into(),
        headers: vec![
            ("content-type".into(), "application/json".into()),
            (e2e::SENT_AT_HEADER.into(), e2e::http_date(NOW)),
        ],
        content: chat("hi"),
    };
    let (request, _) = ohttp::encapsulate_request(&config(&g), &req.encode(0), &[1; 32]).unwrap();
    assert!(matches!(g.open(&request), Err(E2eError::NotPadded { .. })));
}

#[test]
fn the_gateway_forwards_the_chat_route_and_nothing_else() {
    let g = Gateway::from_seed(&SEED);
    for (method, path) in [
        ("GET", "/v2/attestation"),
        ("POST", "/v2/key"),
        ("GET", e2e::CHAT_PATH),
        ("POST", "/v1/chat/completions/../../v2/key"),
    ] {
        let req = bhttp::Request {
            method: method.into(),
            scheme: "https".into(),
            authority: String::new(),
            path: path.into(),
            headers: vec![
                ("content-type".into(), "application/json".into()),
                (e2e::SENT_AT_HEADER.into(), e2e::http_date(NOW)),
            ],
            content: b"{}".to_vec(),
        };
        let (request, _) =
            ohttp::encapsulate_request(&config(&g), &req.encode(e2e::MIN_PADDED), &[1; 32])
                .unwrap();
        assert!(
            matches!(g.open(&request), Err(E2eError::Refused(_))),
            "{method} {path}"
        );
    }
}

/// The time a question was sealed travels inside the encryption, where the
/// carrier can neither read nor move it; without it the gateway refuses.
#[test]
fn a_request_without_a_valid_date_is_refused() {
    let good = e2e::http_date(NOW);
    assert_eq!(good, "Fri, 15 Jan 2027 08:00:00 GMT");
    for date in [
        vec![],
        vec!["soon"],
        vec!["1800000000"],
        vec![" Fri, 15 Jan 2027 08:00:00 GMT"],
        vec!["Fri, 15 Jan 2027 08:00:00 UTC"],
        vec!["Thu, 15 Jan 2027 08:00:00 GMT"],
        vec!["Fri, 32 Jan 2027 08:00:00 GMT"],
        vec!["Mon, 29 Feb 2027 08:00:00 GMT"],
        vec!["Fri, 15 Jan 2027 24:00:00 GMT"],
        vec!["Friday, 15-Jan-27 08:00:00 GMT"],
        vec!["Fri Jan 15 08:00:00 2027"],
        vec![good.as_str(), good.as_str()],
    ] {
        let mut headers = vec![("content-type".to_string(), "application/json".to_string())];
        for v in &date {
            headers.push((e2e::SENT_AT_HEADER.into(), v.to_string()));
        }
        let req = bhttp::Request {
            method: "POST".into(),
            scheme: "https".into(),
            authority: String::new(),
            path: e2e::CHAT_PATH.into(),
            headers,
            content: chat("hi"),
        };
        let (request, _) =
            ohttp::encapsulate_request(&config(&g()), &req.encode(e2e::MIN_PADDED), &[1; 32])
                .unwrap();
        assert!(
            matches!(g().open(&request), Err(E2eError::Refused(_))),
            "{date:?}"
        );
    }
}

fn g() -> Gateway {
    Gateway::from_seed(&SEED)
}

#[test]
fn http_dates_round_trip_and_name_their_weekday() {
    for (unix, text) in [
        (0, "Thu, 01 Jan 1970 00:00:00 GMT"),
        (784_111_777, "Sun, 06 Nov 1994 08:49:37 GMT"),
        (951_782_400, "Tue, 29 Feb 2000 00:00:00 GMT"),
        (4_102_444_799, "Thu, 31 Dec 2099 23:59:59 GMT"),
        (4_107_542_399, "Sun, 28 Feb 2100 23:59:59 GMT"),
    ] {
        assert_eq!(e2e::http_date(unix), text);
        assert_eq!(e2e::parse_http_date(text), Some(unix), "{text}");
    }
    for unix in (0..4_000_000_000u64).step_by(7_919_993) {
        assert_eq!(e2e::parse_http_date(&e2e::http_date(unix)), Some(unix));
    }
    assert_eq!(e2e::parse_http_date("Sat, 01 Jan 1969 00:00:00 GMT"), None);
    assert_eq!(e2e::parse_http_date("Fri, 1a Jan 2027 08:00:00 GMT"), None);
    assert_eq!(
        e2e::parse_http_date("Fri, 15 Jan 2027 08:00:00 GMT\n"),
        None
    );
}

/// SECURITY: a carrier that keeps a sealed question and sends it again gets
/// no second answer: the gateway answers one `enc` once, and only inside
/// its window of the `Date` sealed in the request.
#[test]
fn the_replay_guard_answers_each_request_once_and_only_in_its_window() {
    let gw = g();
    let inner = e2e::chat_request(&chat("hi"), NOW).unwrap();
    let (request, _) = ohttp::encapsulate_request(&config(&gw), &inner, &[1; 32]).unwrap();
    let mut guard = e2e::ReplayGuard::new(120, 8);
    let admit = |guard: &mut e2e::ReplayGuard, now: u64| {
        let (opened, responder) = gw.open(&request).unwrap();
        guard.admit(responder.replay_id(), opened.sent_at, now)
    };
    assert_eq!(admit(&mut guard, NOW + 5), Ok(()));
    assert!(matches!(
        admit(&mut guard, NOW + 6),
        Err(E2eError::Replay(_))
    ));
    // Past the window the entry is forgotten, and the date alone refuses it.
    assert!(matches!(
        admit(&mut guard, NOW + 121),
        Err(E2eError::Replay(_))
    ));
    // A question dated ahead of the gateway's clock by more than the window.
    let mut fresh = e2e::ReplayGuard::new(120, 8);
    assert!(matches!(
        admit(&mut fresh, NOW - 121),
        Err(E2eError::Replay(_))
    ));
    assert_eq!(admit(&mut fresh, NOW - 120), Ok(()));
}

/// SECURITY: the host sets the TD's wall clock. Pushed past a request's
/// window, the guard drops the request's entry; set back inside it, the same
/// request was answered a second time until 2026-10-30. The window is judged
/// on the highest time the guard was given: what it dropped stays refused.
#[test]
fn a_clock_moved_back_does_not_reopen_a_dropped_request() {
    let gw = g();
    let inner = e2e::chat_request(&chat("hi"), NOW).unwrap();
    let (request, _) = ohttp::encapsulate_request(&config(&gw), &inner, &[1; 32]).unwrap();
    let mut guard = e2e::ReplayGuard::new(120, 8);
    let admit = |guard: &mut e2e::ReplayGuard, now: u64| {
        let (opened, responder) = gw.open(&request).unwrap();
        guard.admit(responder.replay_id(), opened.sent_at, now)
    };
    assert_eq!(admit(&mut guard, NOW + 5), Ok(()));
    // The clock pushed 130 s on: a request admitted then drops the first entry.
    guard.admit([7; 32], NOW + 130, NOW + 130).unwrap();
    assert_eq!(guard.len(), 1, "the first request's entry is dropped");
    // The clock set back inside the first request's window.
    for back in [NOW + 10, NOW + 119, NOW] {
        assert!(
            matches!(admit(&mut guard, back), Err(E2eError::Replay(_))),
            "replayed with the clock at NOW + {}",
            back - NOW
        );
    }
    // Nor does the window go back with the clock for a request never seen.
    assert!(matches!(
        guard.admit([8; 32], NOW + 9, NOW + 10),
        Err(E2eError::Replay(_))
    ));
    // Inside the window of the highest time, and not ahead of this clock: once.
    assert_eq!(guard.admit([9; 32], NOW + 20, NOW + 10), Ok(()));
    assert!(matches!(
        guard.admit([9; 32], NOW + 20, NOW + 10),
        Err(E2eError::Replay(_))
    ));
}

#[test]
fn the_replay_guard_forgets_only_what_its_window_refuses_and_stays_bounded() {
    let mut guard = e2e::ReplayGuard::new(60, 3);
    for i in 0..3u8 {
        guard.admit([i; 32], NOW, NOW).unwrap();
    }
    // Full, every entry still in the window: refused, nothing forgotten.
    assert!(matches!(
        guard.admit([9; 32], NOW, NOW + 1),
        Err(E2eError::Replay(_))
    ));
    assert_eq!(guard.len(), 3);
    assert!(matches!(
        guard.admit([0; 32], NOW, NOW + 1),
        Err(E2eError::Replay(_))
    ));
    // Once they age out, room again; the table never grows past capacity.
    guard.admit([9; 32], NOW + 61, NOW + 61).unwrap();
    assert_eq!(guard.len(), 1);
}

#[test]
fn a_request_to_another_gateway_key_does_not_open() {
    let g = Gateway::from_seed(&SEED);
    let other = Gateway::from_seed(&[8; 32]);
    let inner = e2e::chat_request(&chat("hi"), NOW).unwrap();
    let (request, _) = ohttp::encapsulate_request(&config(&other), &inner, &[1; 32]).unwrap();
    assert!(g.open(&request).is_err());
}

/// The response of one request does not open as the response of another,
/// even to the same gateway: each is keyed by its own request's context.
#[test]
fn a_response_opens_for_its_own_request_only() {
    let g = Gateway::from_seed(&SEED);
    let inner = e2e::chat_request(&chat("hi"), NOW).unwrap();
    let (a, client_a) = ohttp::encapsulate_request(&config(&g), &inner, &[1; 32]).unwrap();
    let (b, client_b) = ohttp::encapsulate_request(&config(&g), &inner, &[2; 32]).unwrap();
    let (_, ra) = g.open(&a).unwrap();
    let (_, rb) = g.open(&b).unwrap();
    let resp_a = ra.seal(200, b"for a", &[5; 16]).unwrap();
    let resp_b = rb.seal(200, b"for b", &[5; 16]).unwrap();
    assert!(client_a.decapsulate(&resp_b).is_err());
    assert!(client_b.decapsulate(&resp_a).is_err());
    assert!(client_a.decapsulate(&resp_a).is_ok());
}

/// The same ephemeral input twice gives the same `enc`: the replay id a
/// gateway remembers is the request's, not the caller's.
#[test]
fn the_replay_id_is_the_requests_encapsulated_key() {
    let g = Gateway::from_seed(&SEED);
    let inner = e2e::chat_request(&chat("hi"), NOW).unwrap();
    let (a, _) = ohttp::encapsulate_request(&config(&g), &inner, &[1; 32]).unwrap();
    let (b, _) = ohttp::encapsulate_request(&config(&g), &inner, &[1; 32]).unwrap();
    let (c, _) = ohttp::encapsulate_request(&config(&g), &inner, &[2; 32]).unwrap();
    let id = |r: &[u8]| g.open(r).unwrap().1.replay_id();
    assert_eq!(id(&a), id(&b));
    assert_ne!(id(&a), id(&c));
    assert_eq!(
        id(&a).as_slice(),
        &a[ohttp::HEADER_LEN..ohttp::HEADER_LEN + 32]
    );
}

/// The TD's boot binding, as a key document names it.
const BOOT: e2e::Boot = e2e::Boot {
    nonce: [4; 32],
    tls_spki_sha256: [5; 32],
    config_sha256: [6; 32],
};

/// SECURITY (step 4 of docs/sdk-e2e.md): each input moves the binding, the
/// boot binding's fields too, so a key document naming another session
/// config, boot nonce or TLS key does not match the TD's quote.
#[test]
fn the_binding_changes_with_each_of_its_inputs() {
    let cfg = Gateway::from_seed(&SEED).key_config();
    let other_cfg = Gateway::from_seed(&[8; 32]).key_config();
    let bind = |cfg: &[u8], nonce: [u8; 32], pk: [u8; 32], boot: e2e::Boot| {
        e2e::key_binding(cfg, &nonce, &pk, &boot)
    };
    let base = bind(&cfg, [1; 32], [2; 32], BOOT);
    assert_ne!(base, bind(&other_cfg, [1; 32], [2; 32], BOOT));
    assert_ne!(base, bind(&cfg, [3; 32], [2; 32], BOOT));
    assert_ne!(base, bind(&cfg, [1; 32], [3; 32], BOOT));
    for (boot, what) in [
        (
            e2e::Boot {
                nonce: [7; 32],
                ..BOOT
            },
            "boot nonce",
        ),
        (
            e2e::Boot {
                tls_spki_sha256: [7; 32],
                ..BOOT
            },
            "TLS key",
        ),
        (
            e2e::Boot {
                config_sha256: [7; 32],
                ..BOOT
            },
            "session config",
        ),
    ] {
        assert_ne!(base, bind(&cfg, [1; 32], [2; 32], boot), "{what}");
    }
    assert_eq!(base, bind(&cfg, [1; 32], [2; 32], BOOT));
}

/// The key binding spelled out, as a verifier in another language hashes it.
#[test]
fn the_key_binding_is_sha512_over_its_spelled_layout() {
    use sha2::{Digest, Sha512};
    let cfg = Gateway::from_seed(&SEED).key_config();
    let mut h = Sha512::new();
    h.update(b"x0ne-e2e/v2/ohttp-key\0");
    h.update((cfg.len() as u16).to_be_bytes());
    h.update(&cfg);
    h.update([1; 32]);
    h.update([2; 32]);
    h.update(BOOT.nonce);
    h.update(BOOT.tls_spki_sha256);
    h.update(BOOT.config_sha256);
    let want: [u8; 64] = h.finalize().into();
    assert_eq!(e2e::BINDING_LABEL, b"x0ne-e2e/v2/ohttp-key");
    assert_eq!(e2e::key_binding(&cfg, &[1; 32], &[2; 32], &BOOT), want);
}

#[test]
fn binary_http_refuses_what_it_does_not_speak() {
    // Indeterminate-length request (framing 2).
    assert_eq!(bhttp::Request::decode(&[2]), Err(BhttpError::Framing(2)));
    // A request where a response is expected.
    assert_eq!(bhttp::Response::decode(&[0]), Err(BhttpError::Framing(0)));
    // Informational status.
    assert_eq!(
        bhttp::Response::decode(&[1, 0x40, 100]),
        Err(BhttpError::Status(100))
    );
    // Trailers.
    let mut with_trailer = bhttp::Response {
        status: 200,
        headers: vec![],
        content: b"x".to_vec(),
    }
    .encode(0);
    with_trailer.pop();
    with_trailer.extend_from_slice(&[4, 1, b'a', 1, b'b']);
    assert_eq!(
        bhttp::Response::decode(&with_trailer),
        Err(BhttpError::Trailers)
    );
    // A non-zero byte in the padding.
    let mut padded = bhttp::Response {
        status: 200,
        headers: vec![],
        content: b"x".to_vec(),
    }
    .encode(64);
    padded[60] = 1;
    assert_eq!(
        bhttp::Response::decode(&padded),
        Err(BhttpError::Padding(60))
    );
    // A content length past the message.
    assert_eq!(
        bhttp::Response::decode(&[1, 0x40, 200, 0, 0x44, 0]),
        Err(BhttpError::Truncated("content"))
    );
    // A content length past the bound, refused before any allocation.
    let mut huge = vec![1, 0x40, 200, 0];
    bhttp::put_varint(&mut huge, (bhttp::MAX_CONTENT_BYTES + 1) as u64);
    assert!(matches!(
        bhttp::Response::decode(&huge),
        Err(BhttpError::TooLong {
            what: "content",
            ..
        })
    ));
    // Too many fields.
    let many: Vec<(String, String)> = (0..=bhttp::MAX_FIELDS)
        .map(|i| (format!("h{i}"), "v".into()))
        .collect();
    let r = bhttp::Response {
        status: 200,
        headers: many,
        content: vec![],
    }
    .encode(0);
    assert_eq!(
        bhttp::Response::decode(&r),
        Err(BhttpError::Fields(bhttp::MAX_FIELDS))
    );
}

#[test]
fn binary_http_round_trips_with_padding() {
    let req = bhttp::Request {
        method: "POST".into(),
        scheme: "https".into(),
        authority: String::new(),
        path: "/v1/chat/completions".into(),
        headers: vec![("content-type".into(), "application/json".into())],
        content: vec![0x7b; 70_000],
    };
    let bytes = req.encode(1 << 17);
    assert_eq!(bytes.len(), 1 << 17);
    assert_eq!(bhttp::Request::decode(&bytes).unwrap(), req);
    for v in [
        0u64,
        63,
        64,
        16383,
        16384,
        (1 << 30) - 1,
        1 << 30,
        (1 << 62) - 1,
    ] {
        let mut b = Vec::new();
        bhttp::put_varint(&mut b, v);
        assert_eq!(
            b.len(),
            match v {
                0..=63 => 1,
                64..=16383 => 2,
                16384..=0x3fff_ffff => 4,
                _ => 8,
            }
        );
    }
}

/// SECURITY: a key configuration whose public key has low order would give
/// every sender the same shared secret (RFC 9180 §7.1.4). HPKE refuses it
/// at seal time; the SDK must refuse it when it reads the configuration, so
/// a verdict never says "verified" about such a key.
#[test]
fn a_key_configuration_with_a_low_order_public_key_is_refused() {
    let mut one = [0u8; 32];
    one[0] = 1;
    let mut minus_one = [0xff_u8; 32];
    minus_one[0] = 0xec;
    minus_one[31] = 0x7f; // p - 1, order 4
    for low in [[0u8; 32], one, minus_one] {
        let cfg = KeyConfig {
            key_id: 1,
            public_key: low,
        };
        assert!(
            KeyConfig::parse(&cfg.encode()).is_err(),
            "{:02x?}",
            &low[..4]
        );
    }
    let honest = Gateway::from_seed(&SEED).key_config();
    assert!(KeyConfig::parse(&honest).is_ok());
}

fn sealed_chat(g: &Gateway, body: &[u8]) -> Vec<u8> {
    let inner = e2e::chat_request(body, NOW).unwrap();
    ohttp::encapsulate_request(&config(g), &inner, &[1; 32])
        .unwrap()
        .0
}

/// The gateway forwards a JSON object, as its documentation says, whoever
/// sealed it: the public OHTTP API seals any bytes, not only `Client`'s.
#[test]
fn the_gateway_refuses_a_body_that_is_not_a_json_object() {
    let g = Gateway::from_seed(&SEED);
    assert!(g.open(&sealed_chat(&g, &chat("hi"))).is_ok(), "control");
    for body in [
        &b"this is not json"[..],
        b"",
        b"{",
        b"{} {}",
        b"[]",
        b"\"hi\"",
        br#"{"stream":false,"stream":true}"#,
    ] {
        assert!(
            matches!(g.open(&sealed_chat(&g, body)), Err(E2eError::Refused(_))),
            "{}",
            String::from_utf8_lossy(body)
        );
    }
}

/// A string the JSON grammar allows but no text holds, a lone surrogate
/// (`\ud800`), is refused at opening, by `Client::seal_chat`'s rule: the TD
/// never opens a question it then cannot read.
/// A surrogate pair is text, and opens.
#[test]
fn the_gateway_refuses_a_lone_surrogate_as_the_client_does() {
    let g = Gateway::from_seed(&SEED);
    for content in [r"\ud800", r"\udc00", r"a\ud800b", r"\ud800\u0041"] {
        let body = format!(r#"{{"messages":[{{"role":"user","content":"{content}"}}]}}"#);
        assert!(
            matches!(
                g.open(&sealed_chat(&g, body.as_bytes())),
                Err(E2eError::Refused(_))
            ),
            "{content} opened"
        );
    }
    let pair = br#"{"messages":[{"role":"user","content":"\ud83d\ude00"}]}"#;
    assert!(g.open(&sealed_chat(&g, pair)).is_ok(), "a pair is text");
}

/// One sealed answer per question: `Client` refuses `stream: true`, and so
/// does the gateway, for a request sealed without `Client`.
#[test]
fn the_gateway_refuses_a_streamed_request_as_the_client_does() {
    let g = Gateway::from_seed(&SEED);
    let refused = sealed_chat(&g, br#"{"stream":true,"messages":[]}"#);
    assert!(
        matches!(g.open(&refused), Err(E2eError::Refused(why)) if why.contains("stream")),
        "stream: true forwarded"
    );
    assert!(g
        .open(&sealed_chat(&g, br#"{"stream":false,"messages":[]}"#))
        .is_ok());
}

/// A refusal leaves the TD in clear, to the carrier: it names no byte of
/// the question it refused.
#[test]
fn a_refused_body_is_not_echoed_in_the_refusal() {
    let g = Gateway::from_seed(&SEED);
    for body in [
        &br#"{"messages":[{"role":"user","content":"SECRET-WORD"}],"stream":"SECRET-WORD"}"#[..],
        br#"{"messages":"SECRET-WORD" SECRET-WORD"#,
    ] {
        let Err(e) = g.open(&sealed_chat(&g, body)) else {
            panic!("opened {}", String::from_utf8_lossy(body))
        };
        assert!(!e.to_string().contains("SECRET"), "{e}");
    }
}

/// The whole path and the streamed path open their own requests only: a
/// chunked request asks to stream, a whole one does not, and neither
/// opens under the other's labels.
#[test]
fn the_whole_and_the_streamed_paths_open_their_own_requests_only() {
    let g = Gateway::from_seed(&SEED);
    let streamed = e2e::chat_request(br#"{"stream":true,"messages":[]}"#, NOW).unwrap();
    let whole = e2e::chat_request(br#"{"messages":[]}"#, NOW).unwrap();
    let chunked = |inner: &[u8], ikm: u8| {
        ohttp::encapsulate_chunked_request(&config(&g), &[inner], &[ikm; 32])
            .unwrap()
            .0
    };
    let chunked_streamed = chunked(&streamed, 1);
    assert!(g.open_stream(&chunked_streamed).is_ok());
    assert!(matches!(
        g.open_stream(&chunked(&whole, 2)),
        Err(E2eError::Refused(why)) if why.contains("stream")
    ));
    let (whole_streamed, _) = ohttp::encapsulate_request(&config(&g), &streamed, &[3; 32]).unwrap();
    assert!(g.open_stream(&whole_streamed).is_err());
    assert!(g.open(&chunked_streamed).is_err());
    // Unpadded, refused as on the whole path.
    let mut short = streamed.clone();
    short.truncate(streamed.len() - 1);
    assert!(matches!(
        g.open_stream(&chunked(&short, 4)),
        Err(E2eError::NotPadded { .. })
    ));
}

/// The largest chunked request is the 4 MiB bucket in 256 chunks of 16384
/// bytes: 255 non-final ones, 20 bytes more each, and the final one, 1 byte
/// more. Larger chunks would be refused at 16384 (draft-08 §3).
#[test]
fn the_largest_chunked_request_is_the_largest_bucket_in_chunks_of_16_kib() {
    let g = Gateway::from_seed(&SEED);
    let body = format!(
        r#"{{"stream":true,"messages":[{{"role":"user","content":"{}"}}]}}"#,
        "a".repeat(3 << 20)
    );
    let inner = e2e::chat_request(body.as_bytes(), NOW).unwrap();
    assert_eq!(inner.len(), e2e::MAX_PADDED);
    assert!(ohttp::encapsulate_chunked_request(&config(&g), &[&inner], &[1; 32]).is_err());
    let chunks: Vec<&[u8]> = inner.chunks(ohttp::MAX_CHUNK_DATA).collect();
    assert_eq!(chunks.len(), 256);
    let (sealed, _) = ohttp::encapsulate_chunked_request(&config(&g), &chunks, &[1; 32]).unwrap();
    assert_eq!(sealed.len(), e2e::MAX_SEALED_STREAM_REQUEST);
    assert_eq!(
        e2e::MAX_SEALED_STREAM_REQUEST,
        e2e::MAX_SEALED_REQUEST + 1 + 255 * 20
    );
    assert!(g.open_stream(&sealed).is_ok());
}

/// SECURITY, cost: the client seals a chunked request in chunks of 16384
/// bytes, the final one excepted (`Client::seal_chat_stream`). A request in
/// smaller non-final chunks is refused before any of them is opened: one
/// of one-byte chunks within the route's bound held 233 300 of them, opened
/// in 67.9 ms against 9.4 ms for the largest honest one (an audit, 2026-10-30). A
/// failure is a gateway that decrypts chunks no client writes.
#[test]
fn a_chunked_request_in_chunks_under_16_kib_is_refused_unopened() {
    let g = Gateway::from_seed(&SEED);
    let body = format!(
        r#"{{"stream":true,"messages":[{{"role":"user","content":"{}"}}]}}"#,
        "a".repeat(20_000)
    );
    let inner = e2e::chat_request(body.as_bytes(), NOW).unwrap();
    assert_eq!(inner.len(), 32 * 1024);
    let seal = |chunks: &[&[u8]]| {
        ohttp::encapsulate_chunked_request(&config(&g), chunks, &[1; 32])
            .unwrap()
            .0
    };
    let honest: Vec<&[u8]> = inner.chunks(ohttp::MAX_CHUNK_DATA).collect();
    assert!(g.open_stream(&seal(&honest)).is_ok());
    for size in [1, 100, ohttp::MAX_CHUNK_DATA - 1] {
        let chunks: Vec<&[u8]> = inner.chunks(size).collect();
        let e = g.open_stream(&seal(&chunks)).err().expect("refused");
        assert!(e.to_string().contains("16384"), "{size}: {e}");
    }
    // One short chunk after a full one, its tag broken: refused for its
    // size, so before its tag is checked.
    let (a, rest) = inner.split_at(ohttp::MAX_CHUNK_DATA + 100);
    let mut wire = seal(&[
        &a[..ohttp::MAX_CHUNK_DATA],
        &a[ohttp::MAX_CHUNK_DATA..],
        rest,
    ]);
    // Header, encapsulated key, the first chunk (4-byte length), the second's length.
    let second = ohttp::HEADER_LEN + 32 + 4 + ohttp::MAX_CHUNK_DATA + ohttp::CHUNK_TAG + 2;
    wire[second] ^= 1;
    let e = g.open_stream(&wire).err().expect("refused");
    assert!(e.to_string().contains("16384"), "{e}");
}
