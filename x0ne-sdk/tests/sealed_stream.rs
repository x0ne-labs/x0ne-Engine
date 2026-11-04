//! The caller's reader of a sealed stream (`sealed_stream.rs`), on streams
//! written here with the TD's [`Framer`] and a chunked OHTTP writer: what
//! it accepts, and every way a carrier or a wrong TD can bend a stream,
//! refused. The TD's cadence (when chunks leave, how many) is tested in
//! x0ne-infer (`e2e_stream.rs`, `e2e_td_tests.rs`).

use base64::Engine as _;
use ed25519_dalek::{Signer, SigningKey};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use x0ne_sdk::attest::E2eStream;
use x0ne_sdk::e2e::{self, Gateway};
use x0ne_sdk::ohttp::{self, KeyConfig};
use x0ne_sdk::receipt::{self, RECEIPT_DOMAIN};
use x0ne_sdk::sealed_stream::{self, Framer, PendingStream, FINAL_CHUNK_BYTES, MIN_CHUNKS};

const DIGEST: &str = "4242424242424242424242424242424242424242424242424242424242424242";
const NOW: u64 = 1_800_000_000;
const L: E2eStream = E2eStream::DEFAULT;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn session() -> SigningKey {
    SigningKey::from_bytes(&[5; 32])
}

/// The final chunk's event as the TD writes it, its receipt over `text`
/// signed by `signer`.
fn last_event(text: &str, signer: &SigningKey) -> Value {
    last_event_for(text, signer, "stop", &|_| {})
}

/// [`last_event`] ending on `reason`, its signed payload bent by `f`.
fn last_event_for(text: &str, signer: &SigningKey, reason: &str, f: &dyn Fn(&mut Value)) -> Value {
    let salt = [9u8; 32];
    let mut h = Sha256::new();
    h.update(receipt::OUTPUT_LABEL);
    h.update(salt);
    h.update(text.as_bytes());
    let mut payload = json!({
        "model_digest_hex": DIGEST,
        "output_commitment_hex": hex(&h.finalize()),
        "compute_device": "cpu",
        "session_public_key_hex": hex(session().verifying_key().as_bytes()),
    });
    f(&mut payload);
    let payload = serde_json::to_vec(&payload).unwrap();
    let sig = signer.sign(&[RECEIPT_DOMAIN, &payload].concat());
    json!({
        "id": "chatcmpl-000000000000000000000000",
        "object": "chat.completion.chunk",
        "created": NOW,
        "model": "qwen3:0.6b",
        "choices": [{ "index": 0, "delta": {}, "finish_reason": reason }],
        "x0ne": {
            "receipt": { "signature_hex": hex(&sig.to_bytes()) },
            "opening": { "salt_hex": hex(&salt) },
            "signed_payload_b64": base64::engine::general_purpose::STANDARD.encode(&payload),
        },
    })
}

fn sse(v: &Value) -> Vec<u8> {
    format!("data: {v}\n\n").into_bytes()
}

fn delta(text: &str) -> Vec<u8> {
    sse(&json!({
        "id": "chatcmpl-000000000000000000000000",
        "object": "chat.completion.chunk",
        "created": NOW,
        "model": "qwen3:0.6b",
        "choices": [{ "index": 0, "delta": { "content": text }, "finish_reason": null }],
    }))
}

/// A sealed exchange: the caller's reader and the TD's writer.
fn exchange(cadence: E2eStream) -> (PendingStream, ohttp::ChunkedResponseWriter) {
    exchange_for(
        cadence,
        br#"{"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
    )
}

/// [`exchange`] of the question `body`.
fn exchange_for(cadence: E2eStream, body: &[u8]) -> (PendingStream, ohttp::ChunkedResponseWriter) {
    let (pending, w) = exchange_in(cadence, body, NOW + 3600, None);
    (pending.unwrap(), w)
}

/// [`exchange_for`] in a session ending at `ends`, under `strict` if given.
fn exchange_in(
    cadence: E2eStream,
    body: &[u8],
    ends: u64,
    strict: Option<x0ne_sdk::TcbDisclosure>,
) -> (
    Result<PendingStream, e2e::E2eError>,
    ohttp::ChunkedResponseWriter,
) {
    let g = Gateway::from_seed(&[7; 32]);
    let config = KeyConfig::parse(&g.key_config()).unwrap();
    let inner = e2e::chat_request(body, NOW).unwrap();
    let (request, client) =
        ohttp::encapsulate_chunked_request(&config, &[&inner[..]], &[1; 32]).unwrap();
    let (opened, responder) = g.open_stream(&request).unwrap();
    assert_eq!(*opened.body, body.to_vec());
    let key = session().verifying_key().to_bytes();
    let pending = PendingStream::for_tests(client, cadence, key, body, NOW, ends, strict);
    (pending, responder.writer(&[3; 16]).unwrap())
}

/// The plaintexts of a stream of `chunks` chunks: `events` queued at the
/// first chunk, `tail` in the last.
fn plaintexts(
    cadence: E2eStream,
    status: u16,
    ctype: &str,
    events: &[Vec<u8>],
    tail: &[u8],
    chunks: u64,
) -> Vec<Vec<u8>> {
    let mut f = Framer::new(cadence, status, ctype, &[]);
    for e in events {
        f.push(e);
    }
    let mut out: Vec<Vec<u8>> = (1..chunks).map(|_| f.chunk().unwrap()).collect();
    out.push(f.last(tail).unwrap());
    out
}

/// Each chunk as it leaves the TD: the response nonce leads the first.
fn sealed(w: &mut Option<ohttp::ChunkedResponseWriter>, pts: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let (last, rest) = pts.split_last().unwrap();
    let mut out: Vec<Vec<u8>> = rest
        .iter()
        .map(|p| w.as_mut().unwrap().chunk(p).unwrap())
        .collect();
    out.push(w.take().unwrap().last(last).unwrap());
    out
}

fn answer_events(text: &[&str]) -> Vec<Vec<u8>> {
    text.iter().map(|t| delta(t)).collect()
}

fn tail_ok(text: &str) -> Vec<u8> {
    let mut t = sse(&last_event(text, &session()));
    t.extend_from_slice(b"data: [DONE]\n\n");
    t
}

/// A whole stream of `chunks` chunks for `parts`, as wire pieces.
fn stream(parts: &[&str], chunks: u64) -> (PendingStream, Vec<Vec<u8>>) {
    let (pending, w) = exchange(L);
    let text: String = parts.concat();
    let pts = plaintexts(
        L,
        200,
        "text/event-stream",
        &answer_events(parts),
        &tail_ok(&text),
        chunks,
    );
    (pending, sealed(&mut Some(w), &pts))
}

fn read_all(
    mut p: PendingStream,
    wire: &[Vec<u8>],
) -> Result<
    (
        sealed_stream::StreamAnswer,
        Option<receipt::Checked>,
        Vec<String>,
    ),
    e2e::E2eError,
> {
    let mut deltas = Vec::new();
    for piece in wire {
        deltas.extend(
            p.push(piece)?
                .into_iter()
                .map(sealed_stream::UnverifiedDelta::into_unverified_text),
        );
    }
    let (a, c) = p.finish(DIGEST, NOW)?;
    Ok((a, c, deltas))
}

#[test]
fn a_stream_opens_delta_by_delta_and_its_receipt_checks_over_the_whole_text() {
    let (p, wire) = stream(&["Hel", "lo", ", world"], MIN_CHUNKS);
    let (a, checked, deltas) = read_all(p, &wire).unwrap();
    assert_eq!(deltas, ["Hel", "lo", ", world"]);
    assert_eq!(a.status, 200);
    assert_eq!(a.text, "Hello, world");
    assert_eq!(a.finish_reason.as_deref(), Some("stop"));
    assert_eq!(a.chunks, MIN_CHUNKS);
    assert_eq!(checked.unwrap().text, "Hello, world");
}

/// SECURITY: the sizes a carrier reads are the cadence's, whatever the
/// text: every non-final chunk `2 + L + 16` bytes (its varint length, its
/// plaintext, its tag), the first 16 more (the response nonce), the final
/// `1 + 16384 + 16`. A failure is a size that follows the answer.
#[test]
fn every_chunk_has_one_size_on_the_wire_whatever_the_text() {
    let (_, short) = stream(&["4"], MIN_CHUNKS);
    let (_, long) = stream(&[&"x".repeat(1500), &"é".repeat(300)], MIN_CHUNKS);
    let sizes = |w: &[Vec<u8>]| w.iter().map(Vec::len).collect::<Vec<_>>();
    assert_eq!(sizes(&short), sizes(&long));
    let l = L.chunk_bytes as usize;
    assert_eq!(short[0].len(), 16 + 2 + l + 16);
    for c in &short[1..short.len() - 1] {
        assert_eq!(c.len(), 2 + l + 16);
    }
    assert_eq!(short.last().unwrap().len(), 1 + FINAL_CHUNK_BYTES + 16);
}

/// SECURITY (CVE-2026-48480): a stream cut anywhere, at a chunk's edge or
/// inside one, never reads as whole. A failure is a cut answer taken for
/// the TD's whole one.
#[test]
fn a_stream_cut_anywhere_is_refused() {
    let (_, wire) = stream(&["Hello"], MIN_CHUNKS);
    let all = wire.concat();
    let edges: Vec<usize> = wire
        .iter()
        .scan(0, |at, c| {
            *at += c.len();
            Some(*at)
        })
        .collect();
    let mut cuts: Vec<usize> = edges[..edges.len() - 1].to_vec();
    cuts.extend([0, 1, 15, 16, 17, all.len() - 1, all.len() - 16]);
    for cut in cuts {
        let (p, _) = stream(&["Hello"], MIN_CHUNKS);
        let r = read_all(p, &[all[..cut].to_vec()]);
        assert!(r.is_err(), "cut at {cut} read as whole");
    }
}

/// SECURITY: chunks swapped, one replayed in place of another, one
/// dropped, one bit flipped in any chunk: refused, at the chunk or at the
/// end. A failure is a stream the TD did not write read as its own.
#[test]
fn reordered_replayed_dropped_or_altered_chunks_are_refused() {
    let (_, wire) = stream(&["Hello"], MIN_CHUNKS);
    let refused = |what: &str, w: Vec<Vec<u8>>| {
        let (p, _) = stream(&["Hello"], MIN_CHUNKS);
        assert!(read_all(p, &w).is_err(), "{what}");
    };
    let mut swapped = wire.clone();
    swapped.swap(2, 3);
    refused("swapped", swapped);
    let mut replayed = wire.clone();
    replayed[3] = wire[2].clone();
    refused("replayed", replayed);
    let mut dropped = wire.clone();
    dropped.remove(5);
    refused("dropped", dropped);
    let mut doubled = wire.clone();
    doubled.insert(5, wire[4].clone());
    refused("one chunk twice", doubled);
    for i in [0, 1, 7, wire.len() - 1] {
        for at in [0, 3, 20, wire[i].len() - 1] {
            let mut w = wire.clone();
            w[i][at] ^= 0x20;
            refused(&format!("chunk {i} byte {at}"), w);
        }
    }
}

/// The count of chunks is a power of two from 16 to the cadence's cap:
/// 17 chunks, or 15, or the cap doubled, is not a stream the TD writes.
#[test]
fn a_count_of_chunks_off_the_powers_of_two_is_refused() {
    let max = sealed_stream::max_chunks(&L);
    assert_eq!(
        max, 8192,
        "870.4 s at 100 ms: 8704 ticks, 8192 the power below"
    );
    for chunks in [MIN_CHUNKS * 2, 64] {
        let (p, wire) = stream(&["ok"], chunks);
        assert!(read_all(p, &wire).is_ok(), "{chunks}");
    }
    for chunks in [MIN_CHUNKS - 1, MIN_CHUNKS + 1, 24, 2, 1] {
        let (p, wire) = stream(&["ok"], chunks);
        assert!(read_all(p, &wire).is_err(), "{chunks} chunks read as whole");
    }
    // Past the cap, refused as the chunk that goes past it arrives.
    let cap = E2eStream {
        chunk_bytes: 256,
        tick_ms: 1000,
    };
    assert_eq!(sealed_stream::max_chunks(&cap), 512);
    let (mut p, w) = exchange(cap);
    let pts = plaintexts(cap, 200, "text/event-stream", &[], &tail_ok(""), 1024);
    let wire = sealed(&mut Some(w), &pts);
    let pushed: Result<Vec<_>, _> = wire.iter().map(|c| p.push(c)).collect();
    assert!(pushed.is_err());
}

/// A chunk of another size than the session's, or a final chunk not of
/// 16384 bytes: refused. A TD that wrote either would show the carrier
/// sizes the session config, which the quote binds, does not.
#[test]
fn chunks_of_another_size_are_refused() {
    let other = E2eStream {
        chunk_bytes: 512,
        tick_ms: 100,
    };
    let (p, w) = exchange(L);
    let pts = plaintexts(other, 200, "text/event-stream", &[], &tail_ok(""), 16);
    assert!(read_all(p, &sealed(&mut Some(w), &pts)).is_err());
    let (p, w) = exchange(L);
    let mut pts = plaintexts(L, 200, "text/event-stream", &[], &tail_ok(""), 16);
    pts.last_mut().unwrap().truncate(FINAL_CHUNK_BYTES - 1);
    assert!(read_all(p, &sealed(&mut Some(w), &pts)).is_err());
}

/// What a TD must end a 200 with: the final chunk with its receipt over
/// the text sent, then `[DONE]`. Without either, or with a receipt over
/// other text, or text after the final chunk: refused.
#[test]
fn a_stream_without_its_final_event_or_with_a_wrong_receipt_is_refused() {
    let cases: Vec<(&str, Vec<Vec<u8>>, Vec<u8>)> = vec![
        (
            "no [DONE]",
            answer_events(&["a"]),
            sse(&last_event("a", &session())),
        ),
        (
            "no final chunk",
            answer_events(&["a"]),
            b"data: [DONE]\n\n".to_vec(),
        ),
        (
            "receipt over other text",
            answer_events(&["a"]),
            tail_ok("b"),
        ),
        (
            "receipt by another key",
            answer_events(&["a"]),
            [
                sse(&last_event("a", &SigningKey::from_bytes(&[6; 32]))),
                b"data: [DONE]\n\n".to_vec(),
            ]
            .concat(),
        ),
        (
            "text after the final chunk",
            answer_events(&["a"]),
            [
                sse(&last_event("a", &session())),
                delta("b"),
                b"data: [DONE]\n\n".to_vec(),
            ]
            .concat(),
        ),
        (
            "an event after [DONE]",
            answer_events(&["a"]),
            [tail_ok("a"), delta("b")].concat(),
        ),
        (
            "the content ends inside an event",
            answer_events(&["a"]),
            [tail_ok("a"), b"data: {".to_vec()].concat(),
        ),
    ];
    for (what, events, tail) in cases {
        let (p, w) = exchange(L);
        let pts = plaintexts(L, 200, "text/event-stream", &events, &tail, MIN_CHUNKS);
        assert!(read_all(p, &sealed(&mut Some(w), &pts)).is_err(), "{what}");
    }
}

/// An error after the 200 is an error event in the final chunk: read as
/// the stream's failure, with no receipt. A refusal before it is a status
/// and its JSON error, padded by line feeds JSON skips.
#[test]
fn a_failure_and_a_refusal_come_back_with_no_receipt() {
    let (p, w) = exchange(L);
    let error = sse(&json!({ "error": { "message": "generation stalled" } }));
    let pts = plaintexts(
        L,
        200,
        "text/event-stream",
        &answer_events(&["par"]),
        &error,
        32,
    );
    let (a, checked, deltas) = read_all(p, &sealed(&mut Some(w), &pts)).unwrap();
    assert_eq!((a.status, a.text.as_str(), deltas.len()), (200, "par", 1));
    assert_eq!(a.error.unwrap()["message"], "generation stalled");
    assert!(checked.is_none());

    let (p, w) = exchange(L);
    let body = br#"{"error":{"message":"busy","type":"server_error"}}"#.to_vec();
    let pts = plaintexts(L, 503, "application/json", &[body], b"", MIN_CHUNKS);
    let (a, checked, _) = read_all(p, &sealed(&mut Some(w), &pts)).unwrap();
    assert_eq!(a.status, 503);
    assert_eq!(a.error.unwrap()["message"], "busy");
    assert!(checked.is_none());
}

/// The plaintext is one Binary HTTP response of indeterminate length (RFC
/// 9292 §3.2), the draft's content: its head, content chunks, the end of
/// content and trailers, zeros. Any BHTTP reader reads it whole.
#[test]
fn the_plaintext_is_one_indeterminate_length_binary_http_response() {
    let pts = plaintexts(
        L,
        200,
        "text/event-stream",
        &answer_events(&["hi"]),
        &tail_ok("hi"),
        16,
    );
    let mut r = x0ne_sdk::bhttp::ResponseReader::new(1 << 20);
    let mut content = Vec::new();
    for pt in &pts {
        for part in r.push(pt).unwrap() {
            match part {
                x0ne_sdk::bhttp::Part::Head { status, headers } => {
                    assert_eq!(status, 200);
                    assert_eq!(
                        headers,
                        [("content-type".into(), "text/event-stream".into())]
                    );
                }
                x0ne_sdk::bhttp::Part::Content(c) => content.extend(c),
            }
        }
    }
    r.finish().unwrap();
    let text = String::from_utf8(content).unwrap();
    assert!(text.starts_with("data: {"), "{text:.40}");
    assert!(text.trim_end().ends_with("data: [DONE]"));
    // Padding is line feeds, after whole events only.
    assert!(!text.contains("\"\n"), "a line feed inside an event");
}

/// A stream of `parts` for the question `body`, its final event ending on
/// `reason` with a receipt whose payload `f` bends.
fn stream_for(
    body: &[u8],
    parts: &[&str],
    reason: &str,
    f: &dyn Fn(&mut Value),
) -> Result<
    (
        sealed_stream::StreamAnswer,
        Option<receipt::Checked>,
        Vec<String>,
    ),
    e2e::E2eError,
> {
    let text: String = parts.concat();
    stream_ending(body, parts, &last_event_for(&text, &session(), reason, f))
}

/// [`stream_for`] with the final chunk's event given whole, `last`.
fn stream_ending(
    body: &[u8],
    parts: &[&str],
    last: &Value,
) -> Result<
    (
        sealed_stream::StreamAnswer,
        Option<receipt::Checked>,
        Vec<String>,
    ),
    e2e::E2eError,
> {
    let (pending, w) = exchange_for(L, body);
    let mut tail = sse(last);
    tail.extend_from_slice(b"data: [DONE]\n\n");
    let pts = plaintexts(
        L,
        200,
        "text/event-stream",
        &answer_events(parts),
        &tail,
        MIN_CHUNKS,
    );
    read_all(pending, &sealed(&mut Some(w), &pts))
}

/// SECURITY: the receipt of a stream's final chunk must sign the context
/// the question's conversation implies, as a whole answer's
/// (`receipt::check_answer_for`): a conversation asked whose receipt signs
/// no context or hides the KV-cache flag, none asked whose receipt signs
/// one, and a handle handed back that the signed commitment does not open
/// to are refused. A failure: a caller told its history was computed
/// afresh when it was reused, or the reverse, or a handle the TD did not
/// commit to taken as the next turn's.
#[test]
fn a_stream_whose_receipt_hides_or_invents_a_conversations_context_is_refused() {
    use x0ne_sdk::context::{Handle, SignedContext};
    // `last_event_for`'s salt.
    let salt = [9u8; 32];
    let issued = Handle::from_bytes([0xab; 32]);
    let sent = Handle::from_bytes([0xcd; 32]);
    let opened = serde_json::to_value(SignedContext::commit(&salt, 1, &[], None, &issued)).unwrap();
    let continued = serde_json::to_value(SignedContext::commit(
        &salt,
        2,
        &[1, 2, 3],
        Some(&sent),
        &issued,
    ))
    .unwrap();
    let signs = |context: Option<&Value>, shared: bool| {
        let context = context.cloned();
        move |p: &mut Value| {
            p["prompt_tokens"] = json!(10);
            p["isolation"] = json!({
                "kv_cache_shared_between_requests": shared,
                "prefix_cache_shared_between_sessions": false,
                "requests_batched_across_clients": false,
                "prompt_text_logged": false,
            });
            if let Some(c) = &context {
                p["context"] = c.clone();
            }
        }
    };
    let text = "4";
    let event = |f: &dyn Fn(&mut Value), handle: Option<&Handle>| {
        let mut e = last_event_for(text, &session(), "stop", f);
        if let Some(h) = handle {
            e["x0ne"]["conversation"] = json!({"handle": h.hex()});
        }
        e
    };
    let question = json!({"role": "user", "content": "hi"});
    let new = serde_json::to_vec(&json!({"stream": true, "messages": [question],
        "x0ne": {"conversation": "new"}}))
    .unwrap();
    let next = serde_json::to_vec(&json!({"stream": true, "messages": [question],
        "x0ne": {"conversation": sent.hex()}}))
    .unwrap();
    let free = br#"{"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;

    // Honest: the turn that opens, the one that continues; the handle comes
    // back from `finish`, checked against the commitment; none without a
    // conversation.
    let (a, checked, _) = stream_ending(
        &new,
        &[text],
        &event(&signs(Some(&opened), true), Some(&issued)),
    )
    .unwrap();
    assert_eq!(checked.unwrap().signed["context"]["turn"], json!(1));
    assert_eq!(a.conversation.unwrap().handle, issued);
    let (a, checked, _) = stream_ending(
        &next,
        &[text],
        &event(&signs(Some(&continued), true), Some(&issued)),
    )
    .unwrap();
    assert_eq!(
        checked.unwrap().signed["context"]["reused_tokens"],
        json!(3)
    );
    assert_eq!(a.conversation.unwrap().handle, issued);
    let (a, checked, _) = stream_ending(free, &[text], &event(&|_| {}, None)).unwrap();
    assert!(checked.is_some() && a.conversation.is_none());

    let refused = |r: Result<_, e2e::E2eError>, word: &str| match r {
        Err(e2e::E2eError::Receipt(e)) => {
            assert!(
                e.starts_with("conversation: ") && e.contains(word),
                "{word}: {e}"
            );
        }
        Err(e) => panic!("{word}: {e}"),
        Ok(_) => panic!("{word}: taken"),
    };
    // A conversation asked: no context signed (a TD that dropped the ask),
    // the flag hidden, no handle, a handle the commitment does not open to,
    // "new" answered as a later turn.
    refused(
        stream_ending(&new, &[text], &event(&|_| {}, None)),
        "signs no context",
    );
    refused(
        stream_ending(
            &new,
            &[text],
            &event(&signs(Some(&opened), false), Some(&issued)),
        ),
        "false",
    );
    refused(
        stream_ending(&new, &[text], &event(&signs(Some(&opened), true), None)),
        "no handle",
    );
    refused(
        stream_ending(
            &new,
            &[text],
            &event(&signs(Some(&opened), true), Some(&sent)),
        ),
        "commitment",
    );
    refused(
        stream_ending(
            &new,
            &[text],
            &event(&signs(Some(&continued), true), Some(&issued)),
        ),
        "later turn",
    );
    // None asked: a context, the flag or a handle invented.
    refused(
        stream_ending(
            free,
            &[text],
            &event(&signs(Some(&opened), true), Some(&issued)),
        ),
        "named no conversation",
    );
    refused(
        stream_ending(free, &[text], &event(&|_| {}, Some(&issued))),
        "did not ask",
    );
}

/// SECURITY: the receipt of a stream's final chunk must sign the decoding
/// the question asked, as a whole answer's: sampling, seed, schema digest,
/// the text a document of the schema, a start only under `length`. A
/// failure: a caller takes a stream drawn or shaped otherwise as asked.
#[test]
fn a_stream_whose_receipt_signs_another_decoding_is_refused() {
    let schema =
        json!({"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]});
    let digest = x0ne_sdk::json_schema::schema_sha256_hex(&schema);
    let body = serde_json::to_vec(&json!({
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.5,
        "seed": 7,
        "response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": schema}},
    }))
    .unwrap();
    let signs = |seed: Option<u64>, schema: bool| {
        let digest = digest.clone();
        move |p: &mut Value| {
            if let Some(seed) = seed {
                p["sampling"] = json!({"temperature": 0.5, "top_p": 1.0, "seed": seed});
            }
            if schema {
                p["json_schema_sha256_hex"] = json!(digest);
            }
        }
    };
    let honest = signs(Some(7), true);
    let (a, checked, _) = stream_for(&body, &[r#"{"ok":"#, "false}"], "stop", &honest).unwrap();
    assert_eq!(a.text, r#"{"ok":false}"#);
    let checked = checked.unwrap();
    assert_eq!(checked.signed["sampling"]["seed"], 7);
    assert_eq!(checked.signed["json_schema_sha256_hex"], json!(digest));
    let (a, _, _) = stream_for(&body, &[r#"{"ok":fa"#], "length", &honest).unwrap();
    assert_eq!(a.finish_reason.as_deref(), Some("length"));

    let refused = |r: Result<_, e2e::E2eError>, word: &str| match r {
        Err(e2e::E2eError::Receipt(e)) => {
            assert!(
                e.starts_with("decoding: ") && e.contains(word),
                "{word}: {e}"
            );
        }
        Err(e) => panic!("{word}: {e}"),
        Ok(_) => panic!("{word}: taken"),
    };
    refused(
        stream_for(&body, &[r#"{"ok":0}"#], "stop", &honest),
        "not a document",
    );
    refused(
        stream_for(&body, &[r#"{"ok":fa"#], "stop", &honest),
        "stops before",
    );
    let other_seed = signs(Some(8), true);
    refused(
        stream_for(&body, &[r#"{"ok":true}"#], "stop", &other_seed),
        "another seed",
    );
    let greedy = signs(None, true);
    refused(
        stream_for(&body, &[r#"{"ok":true}"#], "stop", &greedy),
        "greedy",
    );
    let no_schema = signs(Some(7), false);
    refused(
        stream_for(&body, &[r#"{"ok":true}"#], "stop", &no_schema),
        "signs none",
    );
    // A greedy, free question: a receipt that signs a schema or draws is refused.
    let free = br#"{"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    stream_for(free, &["4"], "stop", &|_| {}).unwrap();
    refused(
        stream_for(free, &[r#"{"ok":true}"#], "stop", &honest),
        "sampling",
    );
}

/// SECURITY: the deltas `push` hands out are marked unverified in their
/// type: their text is reached by `unverified_text` alone, and a stream
/// whose receipt is over another text hands out the same deltas before
/// `finish` refuses it. A failure: a delta a caller takes for checked text.
#[test]
fn deltas_are_marked_unverified_until_finish() {
    let (mut p, w) = exchange(L);
    let pts = plaintexts(
        L,
        200,
        "text/event-stream",
        &answer_events(&["Hel", "lo"]),
        &tail_ok("Bye"),
        MIN_CHUNKS,
    );
    let mut deltas: Vec<sealed_stream::UnverifiedDelta> = Vec::new();
    for piece in sealed(&mut Some(w), &pts) {
        deltas.extend(p.push(&piece).unwrap());
    }
    let text: Vec<&str> = deltas
        .iter()
        .map(sealed_stream::UnverifiedDelta::unverified_text)
        .collect();
    assert_eq!(text, ["Hel", "lo"]);
    assert!(matches!(
        p.finish(DIGEST, NOW),
        Err(e2e::E2eError::Receipt(_))
    ));
}

/// The refusal `read_all` met, as text.
fn refusal(wire: (PendingStream, Vec<Vec<u8>>)) -> String {
    match read_all(wire.0, &wire.1) {
        Err(e) => e.to_string(),
        Ok((a, ..)) => panic!("read as whole: {a:?}"),
    }
}

/// A stream of `events`, then `tail` in its final chunk, at status 200.
fn stream_of(events: &[Vec<u8>], tail: &[u8]) -> (PendingStream, Vec<Vec<u8>>) {
    let (p, w) = exchange(L);
    let pts = plaintexts(L, 200, "text/event-stream", events, tail, MIN_CHUNKS);
    (p, sealed(&mut Some(w), &pts))
}

/// SECURITY: once a chunk is refused, the stream stays refused: the next
/// intact chunks are refused, and so is `finish`. A failure: a reader that
/// resumes after a carrier's alteration and returns the rest as an answer.
#[test]
fn a_refused_stream_stays_refused() {
    let (mut p, mut wire) = stream(&["a", "b"], MIN_CHUNKS);
    let last = wire.len() - 1;
    wire[2][5] ^= 1;
    p.push(&wire[0]).unwrap();
    p.push(&wire[1]).unwrap();
    assert!(p.push(&wire[2]).is_err());
    for piece in &wire[3..last] {
        let e = p.push(piece).unwrap_err().to_string();
        assert!(e.contains("refused already"), "{e}");
    }
    let e = p.finish(DIGEST, NOW).unwrap_err().to_string();
    assert!(e.contains("refused already"), "{e}");
}

/// SECURITY: a 200 is an event stream and a refusal a JSON body: a head
/// that says otherwise is refused, so a refusal cannot pass for a stream
/// nor a stream for a refusal. A failure: content read under a type the
/// status does not allow.
#[test]
fn a_head_whose_content_type_does_not_match_its_status_is_refused() {
    let (p, w) = exchange(L);
    let pts = plaintexts(L, 200, "application/json", &[], b"{}", MIN_CHUNKS);
    let e = refusal((p, sealed(&mut Some(w), &pts)));
    assert!(
        e.contains("a 200 whose content is not text/event-stream"),
        "{e}"
    );
    let (p, w) = exchange(L);
    let pts = plaintexts(L, 400, "text/event-stream", &[], b"{}", MIN_CHUNKS);
    let e = refusal((p, sealed(&mut Some(w), &pts)));
    assert!(
        e.contains("a 400 whose content is not application/json"),
        "{e}"
    );
}

/// SSE as the standard writes it (WHATWG HTML §9.2.6): an event over two
/// `data:` lines is their join with a line feed, CRLF ends a line as LF
/// does, the space after `data:` is optional, comments and other fields
/// are skipped. A failure: a TD on another SSE stack read otherwise.
#[test]
fn sse_events_split_over_data_lines_or_ended_by_crlf_read_as_one() {
    let event = String::from_utf8(delta("Hel")).unwrap();
    let json = event.trim_end().strip_prefix("data: ").unwrap();
    let cut = json.find(",\"object\"").unwrap();
    let split = format!(
        ": a comment\r\nevent: delta\r\ndata: {}\r\ndata:{}\r\n\r\n",
        &json[..cut],
        &json[cut..]
    );
    let (a, checked, deltas) = read_all_of(stream_of(
        &[split.into_bytes(), delta("lo")],
        &tail_ok("Hello"),
    ));
    assert_eq!(deltas, ["Hel", "lo"]);
    assert_eq!(a.text, "Hello");
    assert!(checked.is_some());
}

fn read_all_of(
    wire: (PendingStream, Vec<Vec<u8>>),
) -> (
    sealed_stream::StreamAnswer,
    Option<receipt::Checked>,
    Vec<String>,
) {
    read_all(wire.0, &wire.1).unwrap()
}

/// SECURITY: what an OpenAI stream may not carry is refused: an event
/// after the error event, a usage chunk before the final chunk or twice,
/// a chunk with no choice and no usage, a finish_reason that is no string,
/// two choices. A failure: a TD's stream read past what it committed to.
#[test]
fn events_out_of_their_place_are_refused() {
    let error = sse(&json!({"error": {"message": "x", "code": "generation_failed"}}));
    let usage = sse(&json!({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}));
    let empty = sse(&json!({"choices": []}));
    let mut bad_reason = last_event("a", &session());
    bad_reason["choices"][0]["finish_reason"] = json!(1);
    let mut two = last_event("a", &session());
    let choice = two["choices"][0].clone();
    two["choices"] = json!([choice.clone(), choice]);
    let done = b"data: [DONE]\n\n".to_vec();
    let cases: Vec<(&str, Vec<Vec<u8>>, Vec<u8>)> = vec![
        ("after the error event", vec![delta("a"), error], delta("b")),
        (
            "usage chunk out of place",
            vec![delta("a"), usage.clone()],
            tail_ok("a"),
        ),
        (
            "usage chunk out of place",
            vec![delta("a")],
            [
                sse(&last_event("a", &session())),
                usage.clone(),
                usage.clone(),
                done.clone(),
            ]
            .concat(),
        ),
        ("an empty chunk", vec![delta("a"), empty], tail_ok("a")),
        (
            "finish_reason is not a string",
            vec![delta("a")],
            [sse(&bad_reason), done.clone()].concat(),
        ),
        (
            "2 choices: one is committed",
            vec![delta("a")],
            [sse(&two), done.clone()].concat(),
        ),
    ];
    for (what, events, tail) in cases {
        let e = refusal(stream_of(&events, &tail));
        assert!(e.contains(what), "{what}: {e}");
    }
    // The usage chunk in its place, once, after the final chunk.
    let (a, checked, _) = read_all_of(stream_of(
        &[delta("a")],
        &[sse(&last_event("a", &session())), usage, done].concat(),
    ));
    assert_eq!(a.usage.unwrap()["completion_tokens"], 1);
    assert!(checked.is_some());
}

/// SECURITY: an SSE line is held to 1 MiB: a TD or a carrier with the key
/// cannot make the caller buffer an unbounded line. The stream is at the
/// largest cadence, 4096 bytes every second, 512 chunks at most.
#[test]
fn an_sse_line_over_1_mib_is_refused() {
    let big = E2eStream {
        chunk_bytes: 4096,
        tick_ms: 1000,
    };
    let (mut p, w) = exchange(big);
    let mut f = Framer::new(big, 200, "text/event-stream", &[]);
    f.push(&vec![b'x'; (1 << 20) + 1]);
    let mut w = Some(w);
    let mut refused = None;
    for _ in 0..sealed_stream::max_chunks(&big) - 1 {
        let piece = w.as_mut().unwrap().chunk(&f.chunk().unwrap()).unwrap();
        if let Err(e) = p.push(&piece) {
            refused = Some(e.to_string());
            break;
        }
    }
    let e = refused.expect("a line of 1 MiB and one byte was held");
    assert!(e.contains("an SSE line over 1 MiB"), "{e}");
}

/// The TD's framer writes no final chunk that would drop queued text: with
/// more queued than the final chunk holds, `last` is refused and
/// `fits_last` says so; once non-final chunks drained the queue, the final
/// chunk carries the rest and the caller reads every byte. A failure: an
/// answer cut silently at its end.
#[test]
fn the_framer_writes_no_final_chunk_that_drops_text() {
    let text = "y".repeat(20_000);
    let event = delta(&text);
    let mut f = Framer::new(L, 200, "text/event-stream", &[]);
    f.push(&event);
    assert!(!f.fits_last(0));
    let e = Framer::last(
        {
            let mut g = Framer::new(L, 200, "text/event-stream", &[]);
            g.push(&event);
            g
        },
        b"",
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("cannot hold what is left"), "{e}");
    let tail = tail_ok(&text);
    let mut pts = Vec::new();
    while !f.fits_last(tail.len()) {
        pts.push(f.chunk().unwrap());
    }
    let chunks = (pts.len() as u64 + 1).next_power_of_two().max(MIN_CHUNKS);
    while (pts.len() as u64) < chunks - 1 {
        pts.push(f.chunk().unwrap());
    }
    pts.push(f.last(&tail).unwrap());
    let (p, w) = exchange(L);
    let (a, checked, _) = read_all_of((p, sealed(&mut Some(w), &pts)));
    assert_eq!(a.text, text);
    assert!(checked.is_some());
}

/// SECURITY: the TD drains its queue into the final chunk: the deltas it
/// carries are part of the text the receipt signs. The caller must get them
/// (an audit found 116 bytes handed out of 410 signed). A failure is a text
/// shown cut while its receipt covers it whole.
#[test]
fn the_deltas_of_the_final_chunk_reach_the_caller() {
    let parts: Vec<String> = (0..30)
        .map(|i| format!("{i:02} {} ", "w".repeat(96)))
        .collect();
    let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
    let (p, wire) = stream(&parts, MIN_CHUNKS);
    let (a, checked, pushed) = read_all(p, &wire).unwrap();
    assert!(checked.is_some());
    assert_eq!(a.text, parts.concat());
    assert!(
        !a.final_deltas.is_empty(),
        "the queue ends in the final chunk"
    );
    let shown: Vec<String> = pushed.into_iter().chain(a.final_deltas).collect();
    assert_eq!(shown, parts, "every delta reaches the caller, in order");
}

/// A stream of `chunks` chunks for `text` in a session ending at `ends`,
/// under `strict` if given, its receipt's payload bent by `f`.
fn stream_in(
    text: &str,
    chunks: u64,
    ends: u64,
    strict: Option<x0ne_sdk::TcbDisclosure>,
    f: &dyn Fn(&mut Value),
) -> Result<(PendingStream, Vec<Vec<u8>>), e2e::E2eError> {
    let (pending, w) = exchange_in(
        L,
        br#"{"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        ends,
        strict,
    );
    let mut tail = sse(&last_event_for(text, &session(), "stop", f));
    tail.extend_from_slice(b"data: [DONE]\n\n");
    let pts = plaintexts(L, 200, "text/event-stream", &[delta(text)], &tail, chunks);
    Ok((pending?, sealed(&mut Some(w), &pts)))
}

/// SECURITY: the caller bounds a stream by its session's end as the TD
/// does (`sealed_stream::cap`): asked 65 s before it, 32 chunks at most,
/// and within 60 s of it, no stream. A failure: a receipt taken from a
/// stream that outlived the session whose key signed it.
#[test]
fn a_stream_is_bounded_by_its_sessions_end() {
    let ends = NOW + 60 + 5;
    assert_eq!(sealed_stream::cap(&L, NOW, ends), Some(32), "49 ticks fit");
    let (p, wire) = stream_in("a", 32, ends, None, &|_| {}).unwrap();
    assert_eq!(read_all(p, &wire).unwrap().0.chunks, 32);
    let (p, wire) = stream_in("a", 64, ends, None, &|_| {}).unwrap();
    let Err(e) = read_all(p, &wire) else {
        panic!("past the session's end");
    };
    assert!(e.to_string().contains("more than 32 chunks"), "{e}");
    let (p, wire) = stream_in("a", 64, NOW + 3600, None, &|_| {}).unwrap();
    assert_eq!(read_all(p, &wire).unwrap().0.chunks, 64);
    assert_eq!(sealed_stream::cap(&L, NOW, NOW + 61), None);
    let e = stream_in("a", 16, NOW + 61, None, &|_| {})
        .err()
        .expect("no stream fits");
    assert!(e.to_string().contains("session"), "{e}");
}

/// The relay's view of an up-to-date TDX host under `strict`: Intel's
/// latest collateral at evaluation 19, judged 10 s before [`NOW`].
fn strict_view() -> x0ne_sdk::TcbDisclosure {
    serde_json::from_value(json!({
        "schema": x0ne_sdk::TCB_DISCLOSURE_SCHEMA,
        "policy": "strict",
        "as_of_unix": NOW - 10,
        "intel": {
            "accepted_under": "standard", "accepted_evaluation": 19,
            "platform": "UpToDate", "platform_tcb_date": "2025-11-12T00:00:00Z",
            "qe": "UpToDate", "latest_floor_evaluation": 19, "latest_age_limit_secs": 86_400,
            "latest": {
                "judged_at_unix": NOW - 10, "tcb_evaluation": 19, "qe_evaluation": 19,
                "issued_unix": NOW - 3600, "platform": "UpToDate",
                "platform_tcb_date": "2025-11-12T00:00:00Z", "tdx_module": "TDX_03",
                "qe": "UpToDate", "advisories": [],
            },
        },
        "line": "TCB: UpToDate at evaluation 19",
    }))
    .unwrap()
}

/// SECURITY: under `strict`, a stream's receipt must carry the TD's TCB
/// view at the key's floor, as a whole answer's (`e2e::strict_check`): one
/// that carries none is refused at `finish`; the same stream with its view
/// signed is whole. A failure: a stream from a host off its floor taken
/// under `strict`.
#[test]
fn under_strict_a_stream_whose_receipt_has_no_tcb_is_refused_at_finish() {
    let view = strict_view();
    let (p, wire) =
        stream_in("Hello", MIN_CHUNKS, NOW + 3600, Some(view.clone()), &|_| {}).unwrap();
    let Err(e) = read_all(p, &wire) else {
        panic!("no view signed");
    };
    assert!(e.to_string().contains("no TCB disclosure"), "{e}");
    let signed = serde_json::to_value(&view).unwrap();
    let with = |p: &mut Value| p["tcb"] = signed.clone();
    let (p, wire) = stream_in("Hello", MIN_CHUNKS, NOW + 3600, Some(view), &with).unwrap();
    let (a, checked, _) = read_all(p, &wire).expect("its view signed");
    assert_eq!((a.text.as_str(), checked.is_some()), ("Hello", true));
    let (p, wire) = stream_in("Hello", MIN_CHUNKS, NOW + 3600, None, &|_| {}).unwrap();
    read_all(p, &wire).expect("transparent: no view asked");
}

/// The most bytes a stream of a cadence puts on the wire, which a carrier
/// may hold a stream to without opening it: the response nonce, `max_chunks`
/// chunks of L bytes with their two-byte length and tag, and the final
/// chunk. At the default cadence, 2 261 025 bytes.
#[test]
fn the_wire_bound_of_a_stream_follows_its_cadence() {
    let last = 1 + FINAL_CHUNK_BYTES + 16;
    assert_eq!(sealed_stream::max_chunks(&E2eStream::DEFAULT), 8192);
    assert_eq!(
        sealed_stream::max_wire_bytes(&E2eStream::DEFAULT),
        16 + 8191 * (2 + 256 + 16) + last
    );
    assert_eq!(
        sealed_stream::max_wire_bytes(&E2eStream::DEFAULT),
        2_260_751
    );
    let fast = E2eStream {
        chunk_bytes: 4096,
        tick_ms: 20,
    };
    assert_eq!(sealed_stream::max_chunks(&fast), 32_768);
    assert_eq!(
        sealed_stream::max_wire_bytes(&fast),
        16 + 32_767 * (2 + 4096 + 16) + last
    );
    // A real stream of the shortest count is within the bound.
    let (_p, chunks) = stream(&["a"], MIN_CHUNKS);
    let wire: usize = chunks.iter().map(Vec::len).sum();
    assert!(wire >= 15 * (2 + 256 + 16) + last, "{wire}");
    assert!(wire <= sealed_stream::max_wire_bytes(&L));
}
