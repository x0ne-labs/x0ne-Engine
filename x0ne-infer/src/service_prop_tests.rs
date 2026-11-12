//! Properties of the HTTP service, on the router `serve_with` serves
//! (`server::app`), reached over in-memory connections with the clock paused:
//! a turn's second of padding costs no time, and what a burst of callers is
//! answered is decided by the order they arrive in, not by a scheduler.
//!
//! The model of admission is written here, apart from the server: `waiting +
//! 1` queue places; a request takes one when its head is read, before its
//! body, which has 30 s to be whole; a place comes back when the answer's turn
//! ends, or a second after a refusal, never within the instant of a burst. The backend is a scripted
//! model (`prop_kit_tests::ScriptedModel`) whose answer to a request is a
//! function of the request: a second instance is the reference.
//!
//! This router has no credential to get wrong: the `Host` it answers is its
//! access rule (421 otherwise), and stands in for "wrong auth" below.

use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt, DuplexStream};

use crate::backend::{
    Cancelled, Completion, GenerationParams, Isolation, LoadedModel, NoSink, Sampling, TokenSink,
};
use crate::binding::SessionIdentity;
use crate::padding::{
    is_size, pad_json, padme, release_after, size_for, MAX_TICKS, MIN_TICKS, MIN_WHOLE_BYTES,
};
use crate::prompt::Prompt;
use crate::prop_kit_tests::{cases, check, gen_schema, nasty_text, Named, Rng, ScriptedModel};
use crate::server::tests::{challenge_fixture, live_lease, proof};
use crate::server::{app, Admission, ChatMessage, RawEvidence, ServerState, Serving};
use crate::stream::FINAL_FRAME_BYTES;

/// The server's ceiling on one answer's tokens.
const CEILING: usize = 24;
const HOST: &str = "127.0.0.1";
/// axum's default bound on a request body.
const BODY_BOUND: usize = 2 * 1024 * 1024;

// ── The engine under test and its callers ───────────────────────────────────

struct Engine {
    app: Router,
    serving: Arc<Serving>,
}

fn state(model: Box<dyn LoadedModel>, ceiling: usize) -> ServerState {
    ServerState {
        raw_evidence: RawEvidence::default(),
        proof: proof(),
        identity: SessionIdentity::generate([7u8; 32], [9u8; 32]),
        model,
        max_completion_tokens: ceiling,
        lease: live_lease(),
        salt_source: Box::new(|| [3u8; 32]),
        challenge_source: challenge_fixture(),
    }
}

fn engine(model: Box<dyn LoadedModel>, waiting: usize, ceiling: usize) -> Engine {
    let state = state(model, ceiling);
    let serving = Arc::new(Serving {
        admission: Admission::new(waiting),
        ..Serving::default()
    });
    Engine {
        app: app(state, serving.clone()),
        serving,
    }
}

/// A connection to `app`, served as `axum::serve` serves one: hyper over the
/// router, HTTP/1.1. The caller's end is returned; dropping it hangs up.
fn connect(app: &Router) -> DuplexStream {
    let (caller, server) = tokio::io::duplex(64 * 1024);
    let service = hyper_util::service::TowerToHyperService::new(app.clone());
    tokio::spawn(async move {
        let io = hyper_util::rt::TokioIo::new(server);
        // An error here is a caller that hung up: the test's doing.
        let _ = hyper::server::conn::http1::Builder::new()
            .serve_connection(io, service)
            .await;
    });
    caller
}

/// Lets every task run until it waits on the clock or on a caller: what was
/// written is read, and a request whose head is whole has reached the queue.
async fn settle() {
    for _ in 0..200 {
        tokio::task::yield_now().await;
    }
}

fn head(host: &str, length: &str) -> String {
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\n\
         Content-Type: application/json\r\n{length}\r\n\r\n"
    )
}

fn request(host: &str, body: &[u8]) -> (Vec<u8>, usize) {
    let mut wire = head(host, &format!("Content-Length: {}", body.len())).into_bytes();
    let head_len = wire.len();
    wire.extend_from_slice(body);
    (wire, head_len)
}

/// One answer as it came on the wire.
#[derive(Debug)]
struct Answer {
    status: u16,
    /// Bytes of the status line and the headers, the blank line included.
    head_len: usize,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    /// The first chunk of a chunked body, in bytes.
    first_chunk: usize,
}

impl Answer {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }

    fn json(&self) -> Value {
        serde_json::from_slice(&self.body)
            .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&self.body)))
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Reads one answer; `pace`: this many bytes at a time, 3 ms apart. `None`
/// when the connection closes first.
async fn read_answer<R: AsyncRead + Unpin>(io: &mut R, pace: Option<usize>) -> Option<Answer> {
    let mut seen: Vec<u8> = Vec::new();
    let mut buf = vec![0u8; pace.unwrap_or(16 * 1024)];
    // More bytes, or `false` at the end of the connection.
    macro_rules! more {
        () => {{
            let n = io.read(&mut buf).await.unwrap_or(0);
            seen.extend_from_slice(&buf[..n]);
            if pace.is_some() {
                tokio::time::sleep(Duration::from_millis(3)).await;
            }
            n > 0
        }};
    }
    let head_len = loop {
        if let Some(at) = find(&seen, b"\r\n\r\n") {
            break at + 4;
        }
        if !more!() {
            return None;
        }
    };
    let head = String::from_utf8_lossy(&seen[..head_len]).into_owned();
    let mut lines = head.split("\r\n");
    let status: u16 = lines.next()?.split(' ').nth(1)?.parse().ok()?;
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(": "))
        .map(|(n, v)| (n.to_ascii_lowercase(), v.to_string()))
        .collect();
    let named = |name: &str| {
        headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
    };
    let mut body = Vec::new();
    let mut first_chunk = 0;
    let mut at = head_len;
    if named("transfer-encoding").as_deref() == Some("chunked") {
        loop {
            let line_end = loop {
                if let Some(end) = find(&seen[at..], b"\r\n") {
                    break at + end;
                }
                if !more!() {
                    return None;
                }
            };
            let size = usize::from_str_radix(std::str::from_utf8(&seen[at..line_end]).ok()?, 16);
            let size = size.ok()?;
            at = line_end + 2;
            while seen.len() < at + size + 2 {
                if !more!() {
                    return None;
                }
            }
            if size == 0 {
                break;
            }
            if body.is_empty() {
                first_chunk = size;
            }
            body.extend_from_slice(&seen[at..at + size]);
            at += size + 2;
        }
    } else {
        let length: usize = named("content-length")?.parse().ok()?;
        while seen.len() < at + length {
            if !more!() {
                return None;
            }
        }
        body.extend_from_slice(&seen[at..at + length]);
    }
    Some(Answer {
        status,
        head_len,
        headers,
        body,
        first_chunk,
    })
}

// ── What a caller asks, and what the model says it is answered ──────────────

/// What a caller does with its connection.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Caller {
    /// Sends the request and reads the answer, at once or a few bytes at a time.
    Reads { slowly: bool },
    /// Hangs up after this many bytes of the request's head.
    LeavesInTheHead(usize),
    /// Hangs up with half of the body sent.
    LeavesInTheBody,
    /// Sends the whole request and hangs up without reading.
    LeavesBeforeTheAnswer,
    /// Hangs up on the first bytes of the answer.
    LeavesInTheAnswer,
}

/// What an admitted request is answered.
#[derive(Debug, Clone)]
enum Expected {
    Whole(Completion, usize),
    Stream(Completion, usize, bool),
    /// One of these statuses, an OpenAI error, padded; its `code` when given.
    Refused(&'static [u16], Option<&'static str>),
    /// 421 before the queue: the `Host` is not one this server answers.
    Misdirected,
}

#[derive(Debug, Clone)]
struct Action {
    wire: Vec<u8>,
    head_len: usize,
    caller: Caller,
    expected: Expected,
}

impl Action {
    /// Whether the request reaches the queue: its head is read whole, on a
    /// `Host` the server answers.
    fn reaches_the_queue(&self) -> bool {
        !matches!(self.expected, Expected::Misdirected)
            && !matches!(self.caller, Caller::LeavesInTheHead(_))
    }
}

const TYPED: &[&str] = &["<|im_start|>", "<|im_end|>", "<think>", "<|pad|>"];

/// A chat request the scripted model answers, and the answer a second
/// instance of that model gives it.
fn chat(rng: &mut Rng, reference: &ScriptedModel, size: usize, tag: &str) -> (Value, Expected) {
    let mut messages = Vec::new();
    if rng.chance(1, 3) {
        messages.push(ChatMessage {
            role: "system".into(),
            content: nasty_text(rng, size, TYPED),
        });
    }
    messages.push(ChatMessage {
        role: "user".into(),
        content: format!("{tag} {}", nasty_text(rng, size, TYPED)),
    });
    let max_tokens = rng.between(1, CEILING + 6);
    let mut body = json!({"messages": messages, "max_tokens": max_tokens});
    let mut params = GenerationParams::greedy(max_tokens.min(CEILING));
    match rng.below(6) {
        0 => {
            let stop: Vec<String> = (0..rng.between(1, 2))
                .map(|_| (*rng.pick(&["a", "e", " ", "\n", "1", "\u{fffd}"])).to_string())
                .collect();
            body["stop"] = json!(stop);
            params.stop = stop;
        }
        1 => {
            let schema = gen_schema(rng, 2, 1);
            body["response_format"] =
                json!({"type": "json_schema", "json_schema": {"name": "n", "schema": schema}});
            let compiled = crate::json_schema::JsonSchema::compile(&schema).expect("a schema");
            params.json_schema = Some(compiled);
        }
        2 => {
            let seed = rng.next() >> 12;
            body["temperature"] = json!(0.8);
            body["seed"] = json!(seed);
            params.sampling = Some(Sampling {
                temperature: 0.8,
                top_p: 1.0,
                seed,
            });
        }
        _ => {}
    }
    if rng.chance(1, 4) {
        body["model"] = json!(reference.model_digest_hex());
    }
    let prompt = reference.chat_prompt(&messages);
    let completion = reference
        .generate_with(&prompt, &params)
        .unwrap_or_else(|e| panic!("the reference refuses {body}: {e:#}"));
    let budget = params.max_tokens;
    let expected = if rng.chance(1, 2) {
        body["stream"] = json!(true);
        let usage = rng.chance(1, 2);
        if usage {
            body["stream_options"] = json!({"include_usage": true});
        }
        Expected::Stream(completion, budget, usage)
    } else {
        Expected::Whole(completion, budget)
    };
    (body, expected)
}

/// A request the server refuses once it has read it.
fn refused(rng: &mut Rng) -> (Vec<u8>, Expected) {
    let user = json!([{"role": "user", "content": "x"}]);
    let (body, statuses, code): (Vec<u8>, &'static [u16], Option<&'static str>) =
        match rng.below(11) {
            0 => (b"{\"messages\": [".to_vec(), &[400, 422], None),
            1 => (b"not json".to_vec(), &[400, 422], None),
            2 => (Vec::new(), &[400, 422], None),
            3 => (b"[]".to_vec(), &[400, 422], None),
            // A lone surrogate, escaped: no string holds it.
            4 => (
                br#"{"messages":[{"role":"user","content":"\ud800"}]}"#.to_vec(),
                &[400, 422],
                None,
            ),
            5 => (
                json!({"messages": "x"}).to_string().into_bytes(),
                &[400, 422],
                None,
            ),
            6 => (
                json!({"messages": user, "top_k": 5})
                    .to_string()
                    .into_bytes(),
                &[400],
                None,
            ),
            7 => (
                json!({"messages": []}).to_string().into_bytes(),
                &[400],
                None,
            ),
            8 => (
                json!({"messages": user, "model": "another"})
                    .to_string()
                    .into_bytes(),
                &[404],
                Some("model_not_found"),
            ),
            9 => (
                json!({"messages": user, "tools": []})
                    .to_string()
                    .into_bytes(),
                &[400],
                None,
            ),
            // One kibibyte past the bound, whole: refused, not parsed.
            _ => {
                let mut long = br#"{"messages":[{"role":"user","content":""#.to_vec();
                long.resize(BODY_BOUND + 1024, b'a');
                long.extend_from_slice(br#""}]}"#);
                (long, &[413], None)
            }
        };
    (body, Expected::Refused(statuses, code))
}

fn action(rng: &mut Rng, reference: &ScriptedModel, size: usize, tag: &str) -> Action {
    let (body, expected) = match rng.below(10) {
        0 | 1 => refused(rng),
        _ => {
            let (body, expected) = chat(rng, reference, size, tag);
            (body.to_string().into_bytes(), expected)
        }
    };
    let misdirected = rng.chance(1, 12);
    let host = if misdirected { "engine.example" } else { HOST };
    let (wire, head_len) = request(host, &body);
    let streamed = matches!(expected, Expected::Stream(..));
    let caller = match rng.below(12) {
        0 => Caller::LeavesInTheHead(rng.below(head_len)),
        1 if !body.is_empty() => Caller::LeavesInTheBody,
        2 => Caller::LeavesBeforeTheAnswer,
        3 if !misdirected => Caller::LeavesInTheAnswer,
        4 | 5 => Caller::Reads { slowly: true },
        _ => Caller::Reads { slowly: false },
    };
    let _ = streamed;
    Action {
        wire,
        head_len,
        caller,
        expected: if misdirected {
            Expected::Misdirected
        } else {
            expected
        },
    }
}

/// Plays `action` on `io`; the answer, for a caller that reads it.
async fn call(action: Action, mut io: DuplexStream) -> Option<Answer> {
    let Action {
        wire,
        head_len,
        caller,
        ..
    } = action;
    match caller {
        Caller::Reads { slowly } => {
            // Past the bound the server stops reading: the rest is not taken.
            let _ = io.write_all(&wire).await;
            let pace = slowly.then_some(613);
            let answer = tokio::time::timeout(Duration::from_hours(2), read_answer(&mut io, pace));
            Some(
                answer
                    .await
                    .expect("an answer within two hours")
                    .expect("an answer"),
            )
        }
        Caller::LeavesInTheHead(sent) => {
            io.write_all(&wire[..sent]).await.expect("written");
            settle().await;
            None
        }
        Caller::LeavesInTheBody => {
            let half = head_len + (wire.len() - head_len) / 2;
            let _ = io.write_all(&wire[..half]).await;
            settle().await;
            None
        }
        Caller::LeavesBeforeTheAnswer => {
            let _ = io.write_all(&wire).await;
            settle().await;
            None
        }
        Caller::LeavesInTheAnswer => {
            let _ = io.write_all(&wire).await;
            let mut first = [0u8; 64];
            let _ = io.read(&mut first).await;
            None
        }
    }
}

/// An error in OpenAI's shape, of the engine's type: `message`, `type`,
/// `param` and `code`, nothing else.
fn is_an_error(v: &Value) -> &serde_json::Map<String, Value> {
    let outer = v.as_object().expect("an object");
    assert_eq!(outer.keys().collect::<Vec<_>>(), ["error"], "{v}");
    let error = outer["error"].as_object().expect("an error object");
    let mut keys: Vec<&str> = error.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["code", "message", "param", "type"], "{v}");
    assert!(
        error["message"].as_str().is_some_and(|m| !m.is_empty()),
        "{v}"
    );
    assert_eq!(error["type"], "x0ne_engine", "{v}");
    error
}

/// A whole answer or a refusal in its turn: JSON, then spaces up to a size.
fn is_padded(a: &Answer) {
    assert_eq!(a.header("content-type"), Some("application/json"));
    assert!(a.header("x0ne-pad").is_some(), "{:?}", a.headers);
    assert!(
        is_size(a.body.len() as u64, MIN_WHOLE_BYTES),
        "{} bytes is not a size",
        a.body.len()
    );
    assert_eq!(
        a.header("content-length"),
        Some(a.body.len().to_string().as_str())
    );
}

/// The text, the finish, the usage and the end of a stream's events.
struct Streamed {
    text: String,
    finish: Option<String>,
    usage: Option<Value>,
    done: bool,
}

fn streamed(a: &Answer) -> Streamed {
    let mut out = Streamed {
        text: String::new(),
        finish: None,
        usage: None,
        done: false,
    };
    let body = std::str::from_utf8(&a.body).expect("UTF-8");
    for event in body.split("\n\n") {
        let Some(data) = event.strip_prefix("data: ") else {
            assert!(event.is_empty() || event.starts_with(':'), "{event}");
            continue;
        };
        assert!(!out.done, "an event after [DONE]");
        if data == "[DONE]" {
            out.done = true;
            continue;
        }
        let v: Value = serde_json::from_str(data).unwrap_or_else(|e| panic!("{e}: {data}"));
        assert!(v.get("error").is_none(), "{data}");
        if let Some(choice) = v["choices"].get(0) {
            if let Some(text) = choice["delta"]["content"].as_str() {
                out.text.push_str(text);
            }
            if let Some(finish) = choice["finish_reason"].as_str() {
                out.finish = Some(finish.to_string());
            }
        }
        if v.get("usage").is_some_and(|u| !u.is_null()) {
            out.usage = Some(v["usage"].clone());
        }
    }
    out
}

fn usage_of(c: &Completion) -> Value {
    json!({
        "prompt_tokens": c.prompt_tokens,
        "completion_tokens": c.completion_tokens,
        "total_tokens": c.prompt_tokens + c.completion_tokens,
    })
}

/// Holds `answer` to what the model expects of an admitted request.
fn judge(expected: &Expected, a: &Answer, digest: &str) {
    match expected {
        Expected::Misdirected => {
            assert_eq!(a.status, 421);
            is_an_error(&a.json());
        }
        Expected::Refused(statuses, code) => {
            assert!(
                statuses.contains(&a.status),
                "{} for {statuses:?}",
                a.status
            );
            is_padded(a);
            let v = a.json();
            let error = is_an_error(&v);
            if let Some(code) = code {
                assert_eq!(error["code"], *code, "{v}");
            }
        }
        Expected::Whole(c, budget) => {
            assert_eq!(a.status, 200, "{}", String::from_utf8_lossy(&a.body));
            is_padded(a);
            let v = a.json();
            assert_eq!(v["object"], "chat.completion");
            assert_eq!(v["model"], digest);
            assert_eq!(v["choices"].as_array().map(Vec::len), Some(1));
            let choice = &v["choices"][0];
            assert_eq!(choice["message"]["role"], "assistant");
            assert_eq!(
                choice["message"]["content"],
                c.text.as_str(),
                "another's answer"
            );
            assert_eq!(choice["finish_reason"], c.finish_reason(*budget));
            assert_eq!(v["usage"], usage_of(c));
            assert!(v["x0ne"]["receipt"].is_object(), "no receipt");
        }
        Expected::Stream(c, budget, usage) => {
            assert_eq!(a.status, 200, "{}", String::from_utf8_lossy(&a.body));
            assert_eq!(a.header("content-type"), Some("text/event-stream"));
            let s = streamed(a);
            assert!(s.done, "no [DONE]");
            assert_eq!(s.text, c.text, "another's answer");
            assert_eq!(s.finish.as_deref(), Some(c.finish_reason(*budget)));
            assert_eq!(s.usage, usage.then(|| usage_of(c)));
            // Frames of one size, then the last one; their number is a size.
            let frame = a.first_chunk;
            assert!(frame > 0 && a.body.len() >= FINAL_FRAME_BYTES);
            let before = a.body.len() - FINAL_FRAME_BYTES;
            assert_eq!(
                before % frame,
                0,
                "{} bytes in frames of {frame}",
                a.body.len()
            );
            let frames = (before / frame) as u64 + 1;
            assert!(is_size(frames, MIN_TICKS), "{frames} frames is not a size");
        }
    }
}

/// Item 4. Random bursts of callers against the real router: whole and
/// streamed requests, slow readers, callers that hang up in the head, in the
/// body, before the answer and in it, bodies that are not JSON, a body past
/// the bound, requests the parser refuses, a `Host` the server does not
/// answer. Each caller's request reaches the server before the next one's.
///
/// Against the model: the first `waiting + 1` requests whose head was read
/// take the places, whatever they turn out to be; the others are told 503
/// with `Retry-After`, at once. An admitted request is answered what a second
/// instance of the model answers it, text, finish and usage, whole or as a
/// stream: never another caller's answer. Every refusal is OpenAI's error
/// shape, padded as an answer. Once the burst is over no place is left taken
/// and no stream left open, hang-ups included.
#[tokio::test(start_paused = true)]
async fn the_service_follows_the_model_of_admission_for_any_burst_of_callers() {
    let reference = ScriptedModel::new();
    let digest = reference.model_digest_hex();
    let engines: Vec<Engine> = (0..3)
        .map(|waiting| engine(Box::new(ScriptedModel::new()), waiting, CEILING))
        .collect();
    let mut frame_sizes = std::collections::BTreeSet::new();
    for (seed, size) in cases("http_admission", 36, 6) {
        let _named = Named("http_admission", seed, size);
        let mut rng = Rng::new(seed);
        let waiting = rng.below(engines.len());
        let Engine { app, serving } = &engines[waiting];
        let places = waiting + 1;
        for round in 0..rng.between(1, 3) {
            let burst: Vec<Action> = (0..rng.between(1, places + 3))
                .map(|i| {
                    action(
                        &mut rng,
                        &reference,
                        size,
                        &format!("#{seed:x}.{round}.{i}"),
                    )
                })
                .collect();
            // The model: places are taken in the order of arrival.
            let mut free = places;
            let admitted: Vec<bool> = burst
                .iter()
                .map(|a| {
                    let taken = a.reaches_the_queue() && free > 0;
                    free -= usize::from(taken);
                    taken
                })
                .collect();
            let mut callers = Vec::new();
            for a in &burst {
                callers.push(tokio::spawn(call(a.clone(), connect(app))));
                settle().await;
            }
            assert_eq!(
                serving.admission.in_flight(),
                places - free,
                "places taken within the instant of the burst"
            );
            let mut heads = std::collections::BTreeMap::new();
            for ((a, admitted), caller) in burst.iter().zip(admitted).zip(callers) {
                let Some(answer) = caller.await.expect("a caller") else {
                    continue;
                };
                let said = format!("{:?} {:?}", a.caller, a.expected);
                if a.reaches_the_queue() && !admitted {
                    assert_eq!(answer.status, 503, "{said}");
                    assert_eq!(answer.header("retry-after"), Some("5"), "{said}");
                    let v = answer.json();
                    let message = is_an_error(&v)["message"].as_str().unwrap_or_default();
                    assert!(message.contains("in flight"), "{message}");
                    continue;
                }
                judge(&a.expected, &answer, &digest);
                match a.expected {
                    Expected::Stream(..) => {
                        frame_sizes.insert(answer.first_chunk);
                    }
                    Expected::Misdirected => {}
                    // A head of one length for every status of one body size.
                    Expected::Whole(..) | Expected::Refused(..) => {
                        let head = *heads.entry(answer.body.len()).or_insert(answer.head_len);
                        assert_eq!(answer.head_len, head, "{said}: {:?}", answer.headers);
                    }
                }
            }
            // Every turn ends, every body too: nothing is left taken.
            tokio::time::sleep(Duration::from_mins(30)).await;
            assert_eq!(serving.admission.in_flight(), 0, "a queue place leaked");
            assert_eq!(serving.admission.streams_open(), 0, "a stream place leaked");
        }
    }
    assert!(
        frame_sizes.len() <= 1,
        "frames of several sizes: {frame_sizes:?}"
    );
}

/// Item 4, memory. A body that never ends is refused once it passes the
/// bound, in OpenAI's shape, padded: the server has then read the bound and
/// what its buffers hold, not what the caller was ready to send.
#[tokio::test(start_paused = true)]
async fn a_body_without_end_is_refused_at_the_bound_and_read_no_further() {
    let e = engine(Box::new(ScriptedModel::new()), 0, CEILING);
    let (mut reads, mut writes) = tokio::io::split(connect(&e.app));
    let sent = Arc::new(AtomicUsize::new(0));
    let writer = {
        let sent = sent.clone();
        tokio::spawn(async move {
            let head = head(HOST, "Transfer-Encoding: chunked");
            if writes.write_all(head.as_bytes()).await.is_err() {
                return;
            }
            // 64 KiB of whitespace a chunk: JSON that never begins.
            let mut chunk = b"10000\r\n".to_vec();
            chunk.resize(chunk.len() + 0x10000, b' ');
            chunk.extend_from_slice(b"\r\n");
            while writes.write_all(&chunk).await.is_ok() {
                sent.fetch_add(0x10000, SeqCst);
            }
        })
    };
    let answer = read_answer(&mut reads, None).await.expect("an answer");
    writer.abort();
    assert_eq!(answer.status, 413);
    is_padded(&answer);
    is_an_error(&answer.json());
    let sent = sent.load(SeqCst);
    assert!(sent >= BODY_BOUND, "refused after {sent} bytes");
    assert!(
        sent <= 2 * BODY_BOUND,
        "{sent} bytes taken from a body bounded at {BODY_BOUND}"
    );
    tokio::time::sleep(Duration::from_mins(1)).await;
    assert_eq!(e.serving.admission.in_flight(), 0);
}

/// What a queued request has to send its whole body, counted from its head:
/// the server's `BODY_TIMEOUT`, written here apart from it.
const BODY_WAIT: Duration = Duration::from_secs(30);
/// A second before it ends.
const BODY_WAIT_ALMOST: Duration = Duration::from_secs(29);

/// Item 4, the wait for a body. A queue place is taken when the head is
/// read, and the body has [`BODY_WAIT`] from then to be whole. Until then a
/// caller that sent a head and no body, or a body a byte at a time, holds
/// its place: with `waiting + 1` of them the next caller is told 503. Past
/// it the caller is told 408, padded as any refusal, the place is free
/// again and the connection is closed: the next caller is answered. A
/// caller that hangs up before then gives its place back as it leaves.
///
/// Until 2026-11-10 nothing bounded that wait: the place was still held an
/// hour later.
#[tokio::test(start_paused = true)]
async fn a_request_whose_body_does_not_come_gives_its_queue_place_back() {
    let body = json!({"messages": [{"role": "user", "content": "hi"}]}).to_string();
    for drips in [false, true] {
        let e = engine(Box::new(ScriptedModel::new()), 0, CEILING);
        let (mut reads, mut writes) = tokio::io::split(connect(&e.app));
        let head = head(HOST, "Content-Length: 64");
        writes.write_all(head.as_bytes()).await.expect("written");
        // One byte of the body every 5 s, for as long as the server reads.
        let dripper = tokio::spawn(async move {
            while drips && writes.write_all(b" ").await.is_ok() {
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            writes
        });
        tokio::time::sleep(BODY_WAIT_ALMOST).await;
        assert_eq!(e.serving.admission.in_flight(), 1, "held, drips: {drips}");
        let mut io = connect(&e.app);
        io.write_all(&request(HOST, body.as_bytes()).0)
            .await
            .expect("written");
        let answer = read_answer(&mut io, None).await.expect("an answer");
        assert_eq!(answer.status, 503, "drips: {drips}");

        // The refusal leaves a second after the wait ends, as any other.
        let told = tokio::time::timeout(BODY_WAIT, read_answer(&mut reads, None)).await;
        let told = told
            .unwrap_or_else(|_| panic!("no answer {BODY_WAIT:?} past the wait, drips: {drips}"))
            .expect("an answer");
        assert_eq!(told.status, 408, "drips: {drips}");
        is_padded(&told);
        is_an_error(&told.json());
        assert_eq!(e.serving.admission.in_flight(), 0, "drips: {drips}");
        // Nothing more is read from that caller: its connection is closed.
        assert!(read_answer(&mut reads, None).await.is_none());
        dripper.abort();

        let mut io = connect(&e.app);
        io.write_all(&request(HOST, body.as_bytes()).0)
            .await
            .expect("written");
        let answer = read_answer(&mut io, None).await.expect("an answer");
        assert_eq!(answer.status, 200, "drips: {drips}");
    }

    // A caller that leaves before the wait ends: its place comes back then.
    let e = engine(Box::new(ScriptedModel::new()), 0, CEILING);
    let mut idle = connect(&e.app);
    let head = head(HOST, "Content-Length: 64");
    idle.write_all(head.as_bytes()).await.expect("written");
    tokio::time::sleep(BODY_WAIT / 2).await;
    assert_eq!(e.serving.admission.in_flight(), 1);
    drop(idle);
    tokio::time::sleep(Duration::from_mins(1)).await;
    assert_eq!(
        e.serving.admission.in_flight(),
        0,
        "given back at the hang-up"
    );
}

/// A body that is whole within [`BODY_WAIT`] is answered, however it came:
/// in pieces a second apart, the last one a second before the wait ends.
#[tokio::test(start_paused = true)]
async fn a_body_that_comes_slowly_but_in_time_is_answered() {
    let e = engine(Box::new(ScriptedModel::new()), 0, CEILING);
    let mut io = connect(&e.app);
    let body = json!({"messages": [{"role": "user", "content": "hi"}]}).to_string();
    let (wire, head_len) = request(HOST, body.as_bytes());
    io.write_all(&wire[..head_len]).await.expect("written");
    let pieces = u32::try_from(BODY_WAIT.as_secs()).expect("seconds") - 1;
    let piece = body.len().div_ceil(pieces as usize);
    for part in wire[head_len..].chunks(piece) {
        tokio::time::sleep(Duration::from_secs(1)).await;
        io.write_all(part).await.expect("written");
    }
    let answer = read_answer(&mut io, None).await.expect("an answer");
    assert_eq!(answer.status, 200);
    tokio::time::sleep(Duration::from_mins(1)).await;
    assert_eq!(e.serving.admission.in_flight(), 0);
}

/// What a connection has to send a whole request head, counted from its
/// accept or from the end of the answer before: the server's `HEAD_TIMEOUT`,
/// written here apart from it.
const HEAD_WAIT: Duration = Duration::from_mins(61);

/// Item 4, connections. Over the listener `serve_with` serves: a connection
/// that sends nothing, half a request line, or a head without its end is
/// closed [`HEAD_WAIT`] after it was accepted, not before, and the server
/// still answers. Until 2026-11-10 nothing closed it. The clock is paused
/// once the connections are open.
#[tokio::test]
async fn a_connection_that_sends_no_whole_head_is_closed_after_61_minutes() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bound");
    let addr = listener.local_addr().expect("an address");
    let state = state(Box::new(ScriptedModel::new()), CEILING);
    let server = tokio::spawn(crate::server::serve_with(
        listener,
        state,
        Serving::default(),
    ));
    let opened = tokio::time::Instant::now();
    let mut silent = Vec::new();
    for sent in ["", "GET /v1/mod", "GET /v1/models HTTP/1.1\r\nHost: "] {
        let mut sock = tokio::net::TcpStream::connect(addr).await.expect("open");
        sock.write_all(sent.as_bytes()).await.expect("written");
        silent.push((sent, sock));
    }
    // One whole exchange: the server has read what was sent before the clock stops.
    let models = format!("GET /v1/models HTTP/1.1\r\nHost: {HOST}\r\n\r\n");
    let mut sock = tokio::net::TcpStream::connect(addr).await.expect("open");
    sock.write_all(models.as_bytes()).await.expect("written");
    let answer = read_answer(&mut sock, None).await.expect("an answer");
    assert_eq!(answer.status, 200);
    drop(sock);

    tokio::time::pause();
    for (sent, sock) in &mut silent {
        let mut buf = [0u8; 256];
        let closed = tokio::time::timeout(HEAD_WAIT + Duration::from_mins(1), async {
            // Whatever the server says first, its end is what counts.
            while matches!(sock.read(&mut buf).await, Ok(n) if n > 0) {}
        })
        .await;
        assert!(
            closed.is_ok(),
            "{sent:?}: still open a minute past the wait"
        );
        let after = opened.elapsed();
        assert!(after >= HEAD_WAIT, "{sent:?}: {after:?}");
    }
    tokio::time::resume();
    let mut sock = tokio::net::TcpStream::connect(addr).await.expect("open");
    sock.write_all(models.as_bytes()).await.expect("written");
    let answer = read_answer(&mut sock, None).await.expect("an answer");
    assert_eq!(answer.status, 200);
    server.abort();
}

/// A connection to `app` served as `serve_with` serves one
/// (`server::serve_connection`), its wait for a head `head_wait`. The second
/// value keeps the server from stopping: to be held.
fn connect_served(
    app: &Router,
    head_wait: Duration,
) -> (DuplexStream, tokio::sync::watch::Receiver<()>) {
    let (caller, server) = tokio::io::duplex(64 * 1024);
    let (stopping, never) = tokio::sync::watch::channel(());
    let (_, open) = tokio::sync::watch::channel(());
    tokio::spawn(crate::server::serve_connection(
        server,
        app.clone(),
        head_wait,
        stopping,
        open,
    ));
    (caller, never)
}

/// Everything the server still writes, to the end of the connection.
async fn to_the_close<R: AsyncRead + Unpin>(io: &mut R) -> Vec<u8> {
    let mut rest = Vec::new();
    let _ = io.read_to_end(&mut rest).await;
    rest
}

/// The wait for a head times a head and nothing else, here 500 ms. A stream
/// that lasts longer is whole; a connection that asks again within the wait,
/// counted from the end of the answer before, is answered, and again; left
/// idle past it, it is closed without a byte. A head that comes a byte at a
/// time is not a head in time: closed at the wait, counted from the accept.
#[tokio::test(start_paused = true)]
async fn the_wait_for_a_head_cuts_no_answer_and_no_connection_that_asks_in_time() {
    let wait = Duration::from_millis(500);
    let within = Duration::from_millis(400);
    let e = engine(Box::new(ScriptedModel::new()), 0, CEILING);
    let (mut io, _held) = connect_served(&e.app, wait);
    let ask = |stream: bool| {
        let body = json!({"messages": [{"role": "user", "content": "hi"}], "stream": stream});
        request(HOST, body.to_string().as_bytes()).0
    };

    let asked = tokio::time::Instant::now();
    io.write_all(&ask(true)).await.expect("written");
    let answer = read_answer(&mut io, None).await.expect("a stream");
    assert_eq!(answer.status, 200);
    let text = String::from_utf8_lossy(&answer.body);
    assert!(text.contains("data: [DONE]"), "{text}");
    assert!(asked.elapsed() > wait, "{:?}", asked.elapsed());

    for stream in [false, true, false] {
        tokio::time::sleep(within).await;
        io.write_all(&ask(stream)).await.expect("written");
        let answer = read_answer(&mut io, None).await.expect("an answer");
        assert_eq!(answer.status, 200, "stream: {stream}");
    }

    let idle = tokio::time::Instant::now();
    assert_eq!(to_the_close(&mut io).await, b"");
    let after = idle.elapsed();
    assert!(after >= wait && after < wait * 2, "{after:?}");

    let (io, _held) = connect_served(&e.app, wait);
    let opened = tokio::time::Instant::now();
    let (mut reads, mut writes) = tokio::io::split(io);
    let dripper = tokio::spawn(async move {
        let _ = writes.write_all(b"GET /v1/models HTTP/1.1\r\nX: ").await;
        while writes.write_all(b"a").await.is_ok() {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });
    assert_eq!(to_the_close(&mut reads).await, b"");
    let after = opened.elapsed();
    assert!(after >= wait && after < wait * 2, "{after:?}");
    dripper.abort();
    assert_eq!(e.serving.admission.in_flight(), 0);
}

// ── 5. A caller who hangs up ────────────────────────────────────────────────

/// A model that takes 5 ms a step and asks at each one whether anyone still
/// waits, as the candle and Core loops do; a whole answer goes through that
/// loop too. `ran`: the steps run, all requests together.
struct Plodding {
    ran: Arc<AtomicUsize>,
}

impl LoadedModel for Plodding {
    fn generate(&self, prompt: &Prompt, max_tokens: usize) -> anyhow::Result<(String, (u32, u32))> {
        let c = self.generate_with(prompt, &GenerationParams::greedy(max_tokens))?;
        Ok((c.text, (c.prompt_tokens, c.completion_tokens)))
    }
    fn generate_with(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
    ) -> anyhow::Result<Completion> {
        self.generate_streaming(prompt, params, &mut NoSink)
    }
    fn generate_whole(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        self.generate_streaming(prompt, params, sink)
    }
    fn generate_streaming(
        &self,
        _prompt: &Prompt,
        params: &GenerationParams,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        if sink.started().is_break() {
            return Err(Cancelled.into());
        }
        for _ in 0..params.max_tokens {
            if !sink.alive() {
                return Err(Cancelled.into());
            }
            std::thread::sleep(Duration::from_millis(5));
            self.ran.fetch_add(1, SeqCst);
        }
        Ok(Completion {
            text: "t".into(),
            prompt_tokens: 1,
            completion_tokens: params.max_tokens as u32,
            stopped: false,
            length: true,
        })
    }
    fn model_digest_hex(&self) -> String {
        "c0ffee".into()
    }
    fn compute_device(&self) -> String {
        "cpu".into()
    }
    fn isolation(&self) -> Isolation {
        Isolation::NONE_OF_IT
    }
}

/// Real time, which the paused clock does not count: the model's thread runs in it.
async fn real_pause(ms: u64) {
    tokio::task::spawn_blocking(move || std::thread::sleep(Duration::from_millis(ms)))
        .await
        .expect("slept");
}

/// The steps run once the model has stopped: no step in 100 ms.
async fn steps_once_still(ran: &AtomicUsize) -> usize {
    for _ in 0..200 {
        let before = ran.load(SeqCst);
        real_pause(100).await;
        if ran.load(SeqCst) == before {
            return before;
        }
    }
    panic!("still generating after 20 s");
}

fn whole(max_tokens: usize) -> Vec<u8> {
    let body = json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": max_tokens});
    request(HOST, body.to_string().as_bytes()).0
}

/// Item 5. A caller who hangs up on a whole answer stops its generation at
/// the next step, as a stream's reader does (`server.rs`, T11), and its queue
/// place comes back. Until 2026-11-10 a whole answer was generated to its
/// last token for a caller who had left: a GPU computing for nobody, for as
/// long as `max_tokens` allowed.
#[tokio::test(start_paused = true)]
async fn a_caller_who_hangs_up_on_a_whole_answer_stops_its_generation() {
    const STEPS: usize = 400;
    let ran = Arc::new(AtomicUsize::new(0));
    let e = engine(Box::new(Plodding { ran: ran.clone() }), 0, STEPS);
    let mut io = connect(&e.app);
    io.write_all(&whole(STEPS)).await.expect("written");
    for _ in 0..2000 {
        if ran.load(SeqCst) >= 3 {
            break;
        }
        real_pause(5).await;
    }
    assert!(ran.load(SeqCst) >= 3, "the generation never began");
    drop(io);
    let steps = steps_once_still(&ran).await;
    assert!(
        steps < STEPS / 2,
        "{steps} of {STEPS} steps ran for a caller who had left"
    );
    tokio::time::sleep(Duration::from_mins(1)).await;
    assert_eq!(
        e.serving.admission.in_flight(),
        0,
        "its place did not come back"
    );
    let mut io = connect(&e.app);
    io.write_all(&whole(2)).await.expect("written");
    let answer = read_answer(&mut io, None).await.expect("an answer");
    assert_eq!(
        answer.status,
        200,
        "{}",
        String::from_utf8_lossy(&answer.body)
    );
    assert_eq!(answer.json()["usage"]["completion_tokens"], 2);
}

/// Item 5, in the queue. A caller who hangs up while another's answer is
/// being generated never has its own begun: the model runs the first
/// caller's steps and none for the one who left.
#[tokio::test(start_paused = true)]
async fn a_caller_who_hangs_up_in_the_queue_is_never_generated_for() {
    const STEPS: usize = 40;
    let ran = Arc::new(AtomicUsize::new(0));
    let e = engine(Box::new(Plodding { ran: ran.clone() }), 1, STEPS);
    let mut first = connect(&e.app);
    first.write_all(&whole(STEPS)).await.expect("written");
    settle().await;
    let mut second = connect(&e.app);
    second.write_all(&whole(STEPS)).await.expect("written");
    settle().await;
    assert_eq!(e.serving.admission.in_flight(), 2);
    drop(second);
    let answer = read_answer(&mut first, None).await.expect("an answer");
    assert_eq!(answer.status, 200);
    tokio::time::sleep(Duration::from_mins(1)).await;
    assert_eq!(
        steps_once_still(&ran).await,
        STEPS,
        "steps ran for the caller who left"
    );
    assert_eq!(e.serving.admission.in_flight(), 0);
}

/// [`Plodding`] with a prefill: 20 ms once the generation has started, before
/// its first step, as the candle and Core loops read the prompt after
/// `started()` and ask `alive()` only at the first token. `prefills`: the
/// prompts read, all requests together.
struct Prefilling {
    prefills: Arc<AtomicUsize>,
    ran: Arc<AtomicUsize>,
}

impl LoadedModel for Prefilling {
    fn generate(&self, prompt: &Prompt, max_tokens: usize) -> anyhow::Result<(String, (u32, u32))> {
        let c = self.generate_with(prompt, &GenerationParams::greedy(max_tokens))?;
        Ok((c.text, (c.prompt_tokens, c.completion_tokens)))
    }
    fn generate_with(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
    ) -> anyhow::Result<Completion> {
        self.generate_streaming(prompt, params, &mut NoSink)
    }
    fn generate_whole(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        self.generate_streaming(prompt, params, sink)
    }
    fn generate_streaming(
        &self,
        _prompt: &Prompt,
        params: &GenerationParams,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        if sink.started().is_break() {
            return Err(Cancelled.into());
        }
        self.prefills.fetch_add(1, SeqCst);
        std::thread::sleep(Duration::from_millis(20));
        for _ in 0..params.max_tokens {
            if !sink.alive() {
                return Err(Cancelled.into());
            }
            std::thread::sleep(Duration::from_millis(5));
            self.ran.fetch_add(1, SeqCst);
        }
        Ok(Completion {
            text: "t".into(),
            prompt_tokens: 1,
            completion_tokens: params.max_tokens as u32,
            stopped: false,
            length: true,
        })
    }
    fn model_digest_hex(&self) -> String {
        "c0ffee".into()
    }
    fn compute_device(&self) -> String {
        "cpu".into()
    }
    fn isolation(&self) -> Isolation {
        Isolation::NONE_OF_IT
    }
}

fn asked_as_a_stream(max_tokens: usize) -> Vec<u8> {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": max_tokens,
        "stream": true,
    });
    request(HOST, body.to_string().as_bytes()).0
}

/// Item 5, in the queue, for a stream. A caller who asked for a stream and
/// hangs up while another's answer is being generated never has its prompt
/// read: until 2026-11-10 its request's task, which outlives it, was the
/// stream's reader until the header, so the model prefilled the prompt of a
/// caller who had left and stopped only at the first token.
#[tokio::test(start_paused = true)]
async fn a_stream_caller_who_hangs_up_in_the_queue_is_never_prefilled_for() {
    const STEPS: usize = 40;
    let prefills = Arc::new(AtomicUsize::new(0));
    let ran = Arc::new(AtomicUsize::new(0));
    let model = Prefilling {
        prefills: prefills.clone(),
        ran: ran.clone(),
    };
    let e = engine(Box::new(model), 1, STEPS);
    let mut first = connect(&e.app);
    first.write_all(&whole(STEPS)).await.expect("written");
    settle().await;
    let mut second = connect(&e.app);
    second
        .write_all(&asked_as_a_stream(STEPS))
        .await
        .expect("written");
    settle().await;
    assert_eq!(e.serving.admission.in_flight(), 2);
    drop(second);
    let answer = read_answer(&mut first, None).await.expect("an answer");
    assert_eq!(answer.status, 200);
    tokio::time::sleep(Duration::from_mins(1)).await;
    assert_eq!(
        steps_once_still(&ran).await,
        STEPS,
        "steps ran for the caller who left"
    );
    assert_eq!(
        prefills.load(SeqCst),
        1,
        "a prompt was read for the caller who left"
    );
    assert_eq!(e.serving.admission.in_flight(), 0);
}

// ── 7. Sizes ────────────────────────────────────────────────────────────────

/// Item 7. A padded length is a function of its bucket: every length from a
/// size down to the one before it pads to that size, a size is its own, the
/// padding is less than `len / 2^S` (Padme: under 12 % from 16 bytes on), and
/// the size has the low bits its definition clears.
#[test]
fn a_padded_length_depends_on_its_bucket_alone() {
    check("padme", 4000, 31, |rng, size| {
        let n = (rng.next() >> (63 - 2 * size as u32)).max(2);
        let size = padme(n);
        let e = u64::from(n.ilog2());
        let s = u64::from(e.ilog2()) + 1;
        let cleared = e - s;
        assert!(size >= n);
        assert_eq!(size % (1 << cleared), 0, "{n} -> {size}");
        assert!(size - n < 1 << cleared, "{n} -> {size}");
        assert!((size - n) << s < n, "{n} -> {size}: more than len / 2^{s}");
        assert_eq!(padme(size), size, "a size is its own");
        let other = n + rng.next() % (size - n + 1);
        assert_eq!(padme(other), size, "{n} and {other} share a bucket");
        if n > 2 {
            assert!(padme(n - 1) <= size, "not monotone at {n}");
        }
    });
}

/// Item 7. A whole answer's body: the same JSON, spaces after it, a size of
/// 16 KiB at least; and any other body of that bucket, whatever it holds, is
/// as many bytes.
#[test]
fn a_whole_body_is_padded_to_a_size_its_content_does_not_change() {
    check("pad_json", 400, 16, |rng, size| {
        let len = match rng.below(4) {
            0 => rng.below(64),
            1 => rng.between(16_300, 16_500),
            2 => rng.below(4096 * size),
            _ => rng.below(40_000),
        };
        let text: String = (0..len)
            .map(|_| *rng.pick(&['a', 'z', ' ', '"', '\\', '\n', '\u{e9}', '\u{1f600}']))
            .collect();
        let answer = json!({"content": text});
        let body = serde_json::to_vec(&answer).expect("json");
        let padded = pad_json(body.clone());
        assert!(padded.starts_with(&body));
        assert!(padded[body.len()..].iter().all(|&b| b == b' '));
        assert_eq!(
            padded.len() as u64,
            size_for(body.len() as u64, MIN_WHOLE_BYTES)
        );
        assert!(is_size(padded.len() as u64, MIN_WHOLE_BYTES));
        let read: Value = serde_json::from_slice(&padded).expect("the same JSON");
        assert_eq!(read, answer);
        // Buckets are 1 KiB wide from 16 KiB on: 64 bytes less is the same one.
        let other = vec![rng.below(256) as u8; padded.len() - rng.below(64)];
        assert_eq!(pad_json(other).len(), padded.len());
    });
}

/// Item 7. When a whole answer leaves: a whole number of ticks, a size of
/// them, never before the answer is ready, the same for every duration of a
/// bucket; past the longest size, the ticks it took.
#[test]
fn a_turn_ends_a_size_of_ticks_after_it_began() {
    let tick = crate::stream::TICK;
    let tick_ns = tick.as_nanos();
    check("release_after", 4000, 16, |rng, _| {
        let ms = match rng.below(4) {
            0 => rng.below(1_000),
            1 => rng.below(60_000),
            2 => rng.below(900_000),
            _ => rng.below(2_000_000),
        };
        let elapsed = Duration::from_micros(1000 * ms as u64 + rng.below(1000) as u64);
        let leaves = release_after(elapsed, tick);
        assert!(leaves >= elapsed);
        assert_eq!(leaves.as_nanos() % tick_ns, 0);
        let ticks = u64::try_from(leaves.as_nanos() / tick_ns).expect("ticks");
        let took = u64::try_from(elapsed.as_nanos().div_ceil(tick_ns)).expect("ticks");
        if took > MAX_TICKS {
            assert_eq!(ticks, took);
            return;
        }
        assert!(is_size(ticks, MIN_TICKS), "{ticks} ticks is not a size");
        assert_eq!(ticks, size_for(took, MIN_TICKS));
        assert_eq!(release_after(leaves, tick), leaves, "a size is its own");
        let within = leaves.saturating_sub(Duration::from_nanos(rng.below(1_000_000) as u64));
        if within >= elapsed {
            assert_eq!(release_after(within, tick), leaves, "one bucket, two times");
        }
    });
}
