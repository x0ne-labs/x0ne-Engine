//! The request table of the OpenAI surface, played against the three routers
//! that answer `POST /v1/chat/completions`: the TD served directly
//! (`server::serve_with`), the cloud TD (`cloud_serve::router`) and the relay
//! (`local_api::router`).
//!
//! The table states what each router answers today, divergences included:
//! status, body shape, content type, and the headers that differ. One engine
//! ([`play`]) runs it; each router has its own test. A change to what a
//! router answers edits its row, so the change shows in the diff.
//!
//! The cloud TD is called without its TLS channel, as `call_td` in
//! `cloud_e2e_tests` reaches it: its routes, queue and panic layer are the
//! production ones; in production mutual TLS stands in front of them.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use crate::backend::{Cancelled, Completion, GenerationParams, Isolation, LoadedModel, TokenSink};
use crate::openai::models::NOT_FOUND;
use crate::server::{Admission, Serving};

/// A prompt holding this waits in the model until the test lets it go.
const HOLD: &str = "HOLD_THE_QUEUE";
/// A prompt holding this panics in the model, with [`PANIC_PAYLOAD`].
const PANIC: &str = "PANIC_IN_THE_MODEL";
const PANIC_PAYLOAD: &str = "contract panic payload";
/// A prompt holding this fails in the model after its first delta, streamed.
const FAIL: &str = "FAIL_AFTER_THE_FIRST_DELTA";
/// What the TD served directly names as `model` when the request names none.
const TD_DIGEST: &str = "e0e0e0e0";

/// The keys of the `x0ne` block of a whole answer: the TD's and the relay's
/// differ.
const TD_X0NE: &[&str] = &["opening", "receipt", "signed_payload_b64"];
const RELAY_X0NE: &[&str] = &[
    "opening",
    "receipt",
    "receipt_verified",
    "session",
    "signed_payload_b64",
];

// ── The model behind the TD routers ─────────────────────────────────────────

/// Opened by the test or when its guard drops: a failed assertion never
/// leaves a blocking thread parked for the runtime to wait on.
#[derive(Clone, Default)]
struct Latch(Arc<(Mutex<bool>, Condvar)>);

impl Latch {
    fn open(&self) {
        let (open, cv) = &*self.0;
        *open
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        cv.notify_all();
    }

    fn wait(&self) {
        let (open, cv) = &*self.0;
        let mut is_open = open
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while !*is_open {
            is_open = cv
                .wait(is_open)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

struct OpensOnDrop(Latch);

impl Drop for OpensOnDrop {
    fn drop(&mut self) {
        self.0.open();
    }
}

/// Answers "ok". [`HOLD`] waits on `latch`, counted in `entered`; [`PANIC`]
/// panics. Streamed as the trait's default streams, the whole text as one
/// delta; [`FAIL`] then fails.
#[derive(Clone, Default)]
struct Scripted {
    latch: Latch,
    entered: Arc<AtomicUsize>,
    /// Its digest, and so the id it is served under; [`TD_DIGEST`] if none.
    digest: Option<Arc<str>>,
}

impl LoadedModel for Scripted {
    // The panic is the scripted answer to PANIC, not a check.
    #[allow(clippy::manual_assert)]
    fn generate(
        &self,
        prompt: &crate::prompt::Prompt,
        _max_tokens: usize,
    ) -> anyhow::Result<(String, (u32, u32))> {
        if prompt.contains(PANIC) {
            panic!("{PANIC_PAYLOAD}");
        }
        if prompt.contains(HOLD) {
            self.entered.fetch_add(1, SeqCst);
            self.latch.wait();
        }
        Ok(("ok".into(), (1, 1)))
    }
    /// The trait's default, then [`FAIL`]'s error after the delta.
    fn generate_streaming(
        &self,
        prompt: &crate::prompt::Prompt,
        params: &GenerationParams,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        let c = self.generate_with(prompt, params)?;
        if sink.started().is_break() || !sink.alive() || sink.delta(&c.text).is_break() {
            return Err(Cancelled.into());
        }
        if prompt.contains(FAIL) {
            anyhow::bail!("device lost");
        }
        Ok(c)
    }
    fn model_digest_hex(&self) -> String {
        self.digest.as_deref().unwrap_or(TD_DIGEST).into()
    }
    fn compute_device(&self) -> String {
        "cpu".into()
    }
    fn isolation(&self) -> Isolation {
        Isolation::NONE_OF_IT
    }
}

impl Scripted {
    fn maker(&self) -> Box<dyn Fn() -> Box<dyn LoadedModel> + Send + Sync> {
        let model = self.clone();
        Box::new(move || Box::new(model.clone()) as Box<dyn LoadedModel>)
    }

    /// Waits until `n` held prompts are in the model, 10 s at most.
    async fn until_entered(&self, n: usize) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while self.entered.load(SeqCst) < n {
            assert!(std::time::Instant::now() < deadline, "no held prompt ran");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
}

fn says(content: &str) -> Value {
    json!({"messages": [{"role": "user", "content": content}]})
}

// ── The table ───────────────────────────────────────────────────────────────

/// The state of the router a row is asked of.
#[derive(Clone, Copy)]
enum When {
    Ready,
    /// Its queue is full: one place, taken.
    Full,
    /// Its generation panics, by what the router's fixture adds to the body.
    Panicking,
    /// Its generation fails after its first delta, by what the router's
    /// fixture adds to the body.
    Failing,
}

enum Body {
    /// Added to `{"model": <served>, "messages": [user "hi"]}`, key by key.
    Chat(Value),
    /// Sent as is.
    Raw(&'static str),
}

struct Asked {
    body: Body,
    content_type: bool,
    host: Option<&'static str>,
    bearer: bool,
}

fn chat(extra: Value) -> Asked {
    Asked {
        body: Body::Chat(extra),
        content_type: true,
        host: None,
        bearer: true,
    }
}

fn raw(body: &'static str) -> Asked {
    Asked {
        body: Body::Raw(body),
        ..chat(json!({}))
    }
}

impl Asked {
    fn without_content_type(self) -> Self {
        Self {
            content_type: false,
            ..self
        }
    }
    fn host(self, host: &'static str) -> Self {
        Self {
            host: Some(host),
            ..self
        }
    }
    fn without_bearer(self) -> Self {
        Self {
            bearer: false,
            ..self
        }
    }

    /// The bytes sent.
    fn bytes(&self, served: &str, situation: &Value) -> Vec<u8> {
        match &self.body {
            Body::Raw(text) => text.as_bytes().to_vec(),
            Body::Chat(extra) => {
                let mut body = json!({"model": served});
                for part in [&says("hi"), extra, situation] {
                    for (k, v) in part.as_object().expect("an object") {
                        body[k] = v.clone();
                    }
                }
                body.to_string().into_bytes()
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Ct {
    Json,
    EventStream,
}

impl Ct {
    fn prefix(self) -> &'static str {
        match self {
            Self::Json => "application/json",
            Self::EventStream => "text/event-stream",
        }
    }
}

/// The usage chunk of a stream.
#[derive(Clone, Copy)]
enum Usage {
    /// None, and no `usage` key in any chunk.
    None,
    /// Last before `[DONE]`; no `usage` key in the other chunks.
    Chunk,
    /// Last before `[DONE]`; `usage: null` in every other chunk.
    ChunkAndNulls,
}

#[derive(Clone, Copy)]
enum Shape {
    /// `chat.completion` with `id` (`chatcmpl-` and 24 hex digits) and
    /// `created`, `model` the id the surface serves, its `x0ne` block by its
    /// keys.
    Completion(&'static [&'static str]),
    /// Chunks then `[DONE]`; `obfuscation` in every chunk but the usage
    /// chunk, or in none.
    Chunks { obfuscation: bool, usage: Usage },
    /// Chunks, none finishing, then the error event
    /// `{"error": {"message", "type": "x0ne_engine", "code": "generation_failed"}}`
    /// and the end: no `[DONE]`. How many chunks went out before is not
    /// checked: on the TD it is a race (`stream::SseBody::prepare` drops what
    /// a failure finds pending, the role chunk included).
    FailedStream { obfuscation: bool },
    /// OpenAI's shape: `{"error": {"message", "type": "x0ne_engine",
    /// "param", "code"}}`, `param` a string or null, `code` this one or null.
    Api(Option<&'static str>),
}

#[derive(Clone, Copy)]
struct Expect {
    status: u16,
    ct: Ct,
    shape: Shape,
    /// In the error's message or the text; empty: not checked.
    says: &'static str,
    /// Padded as a whole answer: a Padme size of 16 KiB or more, `x0ne-pad`.
    padded: bool,
    retry_after: Option<&'static str>,
    /// `x-should-retry: false`.
    no_retry: bool,
    /// `WWW-Authenticate: Bearer`.
    bearer_challenge: bool,
}

fn expect(status: u16, ct: Ct, shape: Shape, says: &'static str) -> Expect {
    Expect {
        status,
        ct,
        shape,
        says,
        padded: false,
        retry_after: None,
        no_retry: false,
        bearer_challenge: false,
    }
}

fn answered(x0ne: &'static [&'static str]) -> Expect {
    expect(200, Ct::Json, Shape::Completion(x0ne), "")
}

fn chunks(obfuscation: bool, usage: Usage) -> Expect {
    expect(
        200,
        Ct::EventStream,
        Shape::Chunks { obfuscation, usage },
        "",
    )
}

/// A 200, then an error event that says `says`: the failure came after the
/// status was sent.
fn failed_stream(obfuscation: bool, says: &'static str) -> Expect {
    expect(
        200,
        Ct::EventStream,
        Shape::FailedStream { obfuscation },
        says,
    )
}

/// An error in OpenAI's shape, as the TD and the relay send it.
fn api_error(status: u16, says: &'static str) -> Expect {
    expect(status, Ct::Json, Shape::Api(None), says)
}

impl Expect {
    /// An error with this `code`.
    fn coded(self, code: &'static str) -> Self {
        let shape = match self.shape {
            Shape::Api(_) => Shape::Api(Some(code)),
            other => other,
        };
        Self { shape, ..self }
    }
    fn padded(self) -> Self {
        Self {
            padded: true,
            ..self
        }
    }
    fn retry_after(self, seconds: &'static str) -> Self {
        Self {
            retry_after: Some(seconds),
            ..self
        }
    }
    fn no_retry(self) -> Self {
        Self {
            no_retry: true,
            ..self
        }
    }
    fn bearer_challenge(self) -> Self {
        Self {
            bearer_challenge: true,
            ..self
        }
    }
}

struct Row {
    name: &'static str,
    when: When,
    asked: Asked,
    td: Expect,
    cloud: Expect,
    relay: Expect,
}

const TOOLS: &str = "tool calling is not supported yet";
const LONG_MODEL: &str = "model: at most 256 bytes once JSON-escaped with stream: true";

fn weather_tool() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}])
}

/// What each router answers today. The TD columns carry the padding of a
/// whole answer, which is meant to stay.
fn rows() -> Vec<Row> {
    let td_ok = answered(TD_X0NE).padded();
    let relay_ok = answered(RELAY_X0NE);
    vec![
        Row {
            name: "a whole answer",
            when: When::Ready,
            asked: chat(json!({})),
            td: td_ok,
            cloud: td_ok,
            relay: relay_ok,
        },
        Row {
            name: "stream: true",
            when: When::Ready,
            asked: chat(json!({"stream": true})),
            td: chunks(true, Usage::None),
            cloud: chunks(true, Usage::None),
            // The relay drops `obfuscation`, and the usage chunk it always asks
            // of the TD.
            relay: chunks(false, Usage::None),
        },
        // The relay puts `usage: null` in the other chunks; the TD leaves the
        // key out.
        Row {
            name: "stream: true, include_usage",
            when: When::Ready,
            asked: chat(json!({"stream": true, "stream_options": {"include_usage": true}})),
            td: chunks(true, Usage::Chunk),
            cloud: chunks(true, Usage::Chunk),
            relay: chunks(false, Usage::ChunkAndNulls),
        },
        // The bound of the TD's frame size, in the shared validator.
        Row {
            name: "a model name too long for a stream",
            when: When::Ready,
            asked: chat(json!({"stream": true,
                "model": "m".repeat(crate::server::MAX_STREAM_MODEL_BYTES + 1)})),
            td: api_error(400, LONG_MODEL).padded(),
            cloud: api_error(400, LONG_MODEL).padded(),
            relay: api_error(400, LONG_MODEL),
        },
        Row {
            name: "stream that is not a boolean",
            when: When::Ready,
            asked: chat(json!({"stream": "yes"})),
            td: api_error(400, "stream: must be true, false or null").padded(),
            cloud: api_error(400, "stream: must be true, false or null").padded(),
            relay: api_error(400, "stream: must be true, false or null"),
        },
        Row {
            name: "a field refused by name",
            when: When::Ready,
            asked: chat(json!({"logprobs": true})),
            td: api_error(400, "logprobs: not supported by this server").padded(),
            cloud: api_error(400, "logprobs: not supported by this server").padded(),
            relay: api_error(400, "logprobs: not supported by this server"),
        },
        Row {
            name: "a field refused by name, streamed",
            when: When::Ready,
            asked: chat(json!({"logprobs": true, "stream": true})),
            // A stream refused before its 200 gets a JSON status, padded on the
            // TD as a whole answer's: its sentence names a field.
            td: api_error(400, "logprobs: not supported by this server").padded(),
            cloud: api_error(400, "logprobs: not supported by this server").padded(),
            relay: api_error(400, "logprobs: not supported by this server"),
        },
        Row {
            name: "n other than 1",
            when: When::Ready,
            asked: chat(json!({"n": 2})),
            td: api_error(400, "n: only 1 is supported").padded(),
            cloud: api_error(400, "n: only 1 is supported").padded(),
            relay: api_error(400, "n: only 1 is supported"),
        },
        // Tools are refused by every router, streamed or not.
        Row {
            name: "tools",
            when: When::Ready,
            asked: chat(json!({"tools": weather_tool()})),
            td: api_error(400, TOOLS).padded(),
            cloud: api_error(400, TOOLS).padded(),
            relay: api_error(400, TOOLS),
        },
        Row {
            name: "tools, streamed",
            when: When::Ready,
            asked: chat(json!({"tools": weather_tool(), "stream": true})),
            td: api_error(400, TOOLS).padded(),
            cloud: api_error(400, TOOLS).padded(),
            relay: api_error(400, TOOLS),
        },
        Row {
            name: "a tool result in the messages",
            when: When::Ready,
            asked: chat(json!({"messages": [{"role": "user", "content": "weather?"},
                {"role": "tool", "tool_call_id": "call_1", "content": "18 C"}]})),
            td: api_error(400, "messages[].tool_call_id").padded(),
            cloud: api_error(400, "messages[].tool_call_id").padded(),
            relay: api_error(400, "messages[].tool_call_id"),
        },
        // What axum's JSON extractor rejects, as JSON on every router, padded on
        // the TD: its sentence names where the JSON broke.
        Row {
            name: "malformed JSON",
            when: When::Ready,
            asked: raw(r#"{"messages": ["#),
            td: api_error(400, "Failed to parse the request body as JSON").padded(),
            cloud: api_error(400, "Failed to parse the request body as JSON").padded(),
            relay: api_error(400, "Failed to parse the request body as JSON"),
        },
        Row {
            name: "no content type",
            when: When::Ready,
            asked: chat(json!({})).without_content_type(),
            td: api_error(415, "Content-Type: application/json").padded(),
            cloud: api_error(415, "Content-Type: application/json").padded(),
            relay: api_error(415, "Content-Type: application/json"),
        },
        // serde's type errors come from the validator, not the extractor:
        // padded on the TD.
        Row {
            name: "messages that is not a list",
            when: When::Ready,
            asked: chat(json!({"messages": 5})),
            td: api_error(422, "invalid type").padded(),
            cloud: api_error(422, "invalid type").padded(),
            relay: api_error(422, "invalid type"),
        },
        // Content given as a list of parts is refused by serde on every router.
        Row {
            name: "content in parts",
            when: When::Ready,
            asked: chat(json!({"messages": [{"role": "user",
                "content": [{"type": "text", "text": "hi"}]}]})),
            td: api_error(422, "expected a string").padded(),
            cloud: api_error(422, "expected a string").padded(),
            relay: api_error(422, "expected a string"),
        },
        // A model name not served is a 404 that quotes nothing of it, padded on
        // the TD as any whole refusal.
        Row {
            name: "a model it does not serve",
            when: When::Ready,
            asked: chat(json!({"model": "gpt-4o"})),
            td: api_error(404, "not served here").coded(NOT_FOUND).padded(),
            cloud: api_error(404, "not served here").coded(NOT_FOUND).padded(),
            relay: api_error(404, "not served here").coded(NOT_FOUND),
        },
        // The same streamed: refused before the 200, padded on the TD.
        Row {
            name: "a stream for a model it does not serve",
            when: When::Ready,
            asked: chat(json!({"model": "gpt-4o", "stream": true})),
            td: api_error(404, "not served here").coded(NOT_FOUND).padded(),
            cloud: api_error(404, "not served here").coded(NOT_FOUND).padded(),
            relay: api_error(404, "not served here").coded(NOT_FOUND),
        },
        // A foreign Host is refused, except by the cloud TD, which reads no
        // Host: mutual TLS stands in front of it.
        Row {
            name: "a foreign Host",
            when: When::Ready,
            asked: chat(json!({})).host("evil.example"),
            // In OpenAI's shape, unpadded.
            td: api_error(421, crate::server::MISDIRECTED),
            cloud: td_ok,
            relay: api_error(403, "is not this endpoint"),
        },
        // Only the relay asks for a bearer, by design.
        Row {
            name: "no bearer",
            when: When::Ready,
            asked: chat(json!({})).without_bearer(),
            td: td_ok,
            cloud: td_ok,
            relay: api_error(401, "missing or wrong bearer").bearer_challenge(),
        },
        // Empty messages: the relay passes them to the TD, which refuses them.
        Row {
            name: "empty messages",
            when: When::Ready,
            asked: chat(json!({"messages": []})),
            td: api_error(400, "messages is empty").padded(),
            cloud: api_error(400, "messages is empty").padded(),
            relay: api_error(400, "messages is empty"),
        },
        // A per-request adapter is refused by design where one attested model
        // is served.
        Row {
            name: "a per-request adapter",
            when: When::Ready,
            asked: chat(json!({"x0ne": {"adapter": "a"}})),
            td: api_error(400, "x0ne.adapter").padded(),
            cloud: api_error(400, "x0ne.adapter").padded(),
            relay: api_error(400, "x0ne.adapter"),
        },
        // A full queue is a JSON 503 with Retry-After on every router; the
        // relay passes its Retry-After on.
        Row {
            name: "a full queue",
            when: When::Full,
            asked: chat(json!({})),
            td: api_error(503, "busy").retry_after("5"),
            cloud: api_error(503, "busy").retry_after("5"),
            relay: api_error(503, "busy").retry_after("5"),
        },
        // A stream refused before its 200 gets the status a whole answer gets,
        // unpadded.
        Row {
            name: "a full queue, streamed",
            when: When::Full,
            asked: chat(json!({"stream": true})),
            td: api_error(503, "busy").retry_after("5"),
            cloud: api_error(503, "busy").retry_after("5"),
            relay: api_error(503, "busy").retry_after("5"),
        },
        // A panic's payload is quoted by no router, and the relay says not to
        // retry. Padded on the TD as any whole failure.
        Row {
            name: "a panic in the generation",
            when: When::Panicking,
            asked: chat(json!({})),
            td: api_error(500, "internal error: this request failed").padded(),
            cloud: api_error(500, "internal error: this request failed").padded(),
            relay: api_error(500, "internal error: this request failed").no_retry(),
        },
        // Before the first token: a 500, not a 200, padded on the TD as a
        // whole refusal once its turn began.
        Row {
            name: "a panic in the generation, streamed",
            when: When::Panicking,
            asked: chat(json!({"stream": true})),
            td: api_error(500, "internal error: this request failed").padded(),
            cloud: api_error(500, "internal error: this request failed").padded(),
            relay: api_error(500, "internal error: this request failed").no_retry(),
        },
        // After the 200, the error event ends the body with a fixed message;
        // the relay's own (`local_api::GENERATION`).
        Row {
            name: "a generation that fails after its first token, streamed",
            when: When::Failing,
            asked: chat(json!({"stream": true})),
            td: failed_stream(true, "generation failed"),
            cloud: failed_stream(true, "generation failed"),
            relay: failed_stream(false, "the attested VM reported a failed generation"),
        },
    ]
}

// ── The engine ──────────────────────────────────────────────────────────────

enum Target {
    /// `<base>/v1/chat/completions` over HTTP, with this bearer.
    Http {
        base: String,
        bearer: Option<String>,
    },
    /// The routes in process.
    Routes(axum::Router),
}

fn http(base: String) -> Target {
    Target::Http { base, bearer: None }
}

/// One router under the table, in the four states a row may ask for.
struct Surface {
    name: &'static str,
    column: fn(&Row) -> &Expect,
    /// The id the surface serves: a row's `model` unless it names another,
    /// and every answer's.
    served: String,
    panic_payload: &'static str,
    ready: Target,
    full: Target,
    /// The router whose generation panics, and what the body adds to make it.
    panicking: (Target, Value),
    /// What the body adds to make the generation of `ready` fail after its
    /// first delta.
    failing: Value,
    /// Whether `ready` answers `GET /v1/models`; the cloud TD is reached
    /// through the relay, which lists for it.
    lists_models: bool,
}

struct Answer {
    status: u16,
    content_type: String,
    retry_after: Option<String>,
    should_retry: Option<String>,
    pad_header: bool,
    bearer_challenge: Option<String>,
    body: Vec<u8>,
}

impl Answer {
    fn of(status: u16, headers: &axum::http::HeaderMap, body: Vec<u8>) -> Self {
        let header = |name: &str| {
            headers
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
        };
        Self {
            status,
            content_type: header("content-type").unwrap_or_default(),
            retry_after: header("retry-after"),
            should_retry: header("x-should-retry"),
            pad_header: headers.contains_key("x0ne-pad"),
            bearer_challenge: header("www-authenticate"),
            body,
        }
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    fn excerpt(&self) -> String {
        self.text().trim_end().chars().take(300).collect()
    }
}

async fn send(target: &Target, asked: &Asked, body: Vec<u8>) -> Answer {
    let call = async {
        match target {
            Target::Http { base, bearer } => {
                let mut req = reqwest::Client::builder()
                    .no_proxy()
                    .build()
                    .unwrap()
                    .post(format!("{base}/v1/chat/completions"))
                    .body(body);
                if asked.content_type {
                    req = req.header(reqwest::header::CONTENT_TYPE, "application/json");
                }
                if let Some(host) = asked.host {
                    req = req.header(reqwest::header::HOST, host);
                }
                if let (true, Some(bearer)) = (asked.bearer, bearer) {
                    req = req.bearer_auth(bearer);
                }
                let r = req.send().await.expect("an answer, not a reset");
                let (status, headers) = (r.status().as_u16(), r.headers().clone());
                let body = r.bytes().await.expect("a body that ends").to_vec();
                Answer::of(status, &headers, body)
            }
            Target::Routes(routes) => {
                use hyper::service::Service as _;
                let svc = hyper_util::service::TowerToHyperService::new(routes.clone());
                let mut req = axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions");
                if asked.content_type {
                    req = req.header(axum::http::header::CONTENT_TYPE, "application/json");
                }
                if let Some(host) = asked.host {
                    req = req.header(axum::http::header::HOST, host);
                }
                let resp = svc
                    .call(req.body(axum::body::Body::from(body)).unwrap())
                    .await
                    .unwrap();
                let (status, headers) = (resp.status().as_u16(), resp.headers().clone());
                let body = axum::body::to_bytes(resp.into_body(), 1 << 22)
                    .await
                    .expect("a body that ends")
                    .to_vec();
                Answer::of(status, &headers, body)
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(60), call)
        .await
        .expect("an answer within 60 s")
}

/// `GET /v1/models` on `target`.
async fn models_of(target: &Target) -> Answer {
    let call = async {
        match target {
            Target::Http { base, bearer } => {
                let mut req = reqwest::Client::builder()
                    .no_proxy()
                    .build()
                    .unwrap()
                    .get(format!("{base}/v1/models"));
                if let Some(bearer) = bearer {
                    req = req.bearer_auth(bearer);
                }
                let r = req.send().await.expect("an answer, not a reset");
                let (status, headers) = (r.status().as_u16(), r.headers().clone());
                let body = r.bytes().await.expect("a body that ends").to_vec();
                Answer::of(status, &headers, body)
            }
            Target::Routes(routes) => {
                use hyper::service::Service as _;
                let svc = hyper_util::service::TowerToHyperService::new(routes.clone());
                let req = axum::http::Request::builder()
                    .method("GET")
                    .uri("/v1/models")
                    .body(axum::body::Body::empty())
                    .unwrap();
                let resp = svc.call(req).await.unwrap();
                let (status, headers) = (resp.status().as_u16(), resp.headers().clone());
                let body = axum::body::to_bytes(resp.into_body(), 1 << 22)
                    .await
                    .expect("a body that ends")
                    .to_vec();
                Answer::of(status, &headers, body)
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(60), call)
        .await
        .expect("an answer within 60 s")
}

/// `GET /v1/models` lists the one model the surface serves, by the id its
/// answers name, as OpenAI's `Model` objects with the engine's fields under
/// `x0ne`.
fn check_models(a: &Answer, s: &Surface) -> Result<(), String> {
    if !s.lists_models {
        return match a.status {
            404 => Ok(()),
            other => Err(format!("status {other}, not 404: no list here")),
        };
    }
    if a.status != 200 || !a.content_type.starts_with("application/json") {
        return Err(format!("status {} {:?}", a.status, a.content_type));
    }
    let v: Value = serde_json::from_slice(&a.body).map_err(|e| e.to_string())?;
    exactly("the list", &v, &["data", "object"])?;
    if v["object"] != "list" {
        return Err(format!("object {}", v["object"]));
    }
    let [model] = v["data"].as_array().map(Vec::as_slice).unwrap_or_default() else {
        return Err(format!("data is not one model: {}", v["data"]));
    };
    exactly(
        "the model",
        model,
        &["created", "id", "object", "owned_by", "x0ne"],
    )?;
    if model["id"] != s.served.as_str() {
        return Err(format!("id {}, not {}", model["id"], s.served));
    }
    if model["object"] != "model" || model["owned_by"] != "x0ne-engine" {
        return Err(format!("object or owned_by: {model}"));
    }
    if !model["created"].is_u64() || !model["x0ne"].is_object() {
        return Err(format!("created or x0ne: {model}"));
    }
    Ok(())
}

fn keys(v: &Value) -> BTreeSet<String> {
    v.as_object()
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default()
}

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|n| (*n).to_string()).collect()
}

/// `Ok` when `v` has exactly the keys `names`.
fn exactly(what: &str, v: &Value, names: &[&str]) -> Result<(), String> {
    if keys(v) == set(names) {
        Ok(())
    } else {
        Err(format!("{what} has keys {:?}, not {names:?}", keys(v)))
    }
}

fn message_says(message: &Value, says: &str) -> Result<(), String> {
    match message.as_str() {
        Some(m) if m.contains(says) => Ok(()),
        _ => Err(format!("the message {message} does not say {says:?}")),
    }
}

impl Expect {
    fn check(&self, a: &Answer, s: &Surface) -> Result<(), String> {
        if a.status != self.status {
            return Err(format!("status {}, not {}", a.status, self.status));
        }
        if !a.content_type.starts_with(self.ct.prefix()) {
            return Err(format!(
                "content type {:?}, not {}",
                a.content_type,
                self.ct.prefix()
            ));
        }
        let size = u64::try_from(a.body.len()).unwrap_or(u64::MAX);
        let padded = a.pad_header && crate::padding::is_size(size, crate::padding::MIN_WHOLE_BYTES);
        if padded != self.padded || a.pad_header != self.padded {
            return Err(format!(
                "padded {padded} ({} bytes, x0ne-pad {}), not {}",
                a.body.len(),
                a.pad_header,
                self.padded
            ));
        }
        if a.retry_after.as_deref() != self.retry_after {
            return Err(format!(
                "Retry-After {:?}, not {:?}",
                a.retry_after, self.retry_after
            ));
        }
        if (a.should_retry.as_deref() == Some("false")) != self.no_retry
            || (a.should_retry.is_some() && !self.no_retry)
        {
            return Err(format!("x-should-retry {:?}", a.should_retry));
        }
        if (a.bearer_challenge.as_deref() == Some("Bearer")) != self.bearer_challenge
            || (a.bearer_challenge.is_some() && !self.bearer_challenge)
        {
            return Err(format!("WWW-Authenticate {:?}", a.bearer_challenge));
        }
        // A panic's payload may quote the prompt; no surface returns it.
        if a.text().contains(s.panic_payload) {
            return Err("the panic payload is quoted".into());
        }
        if let Shape::Chunks { obfuscation, usage } = self.shape {
            return check_chunks(a, obfuscation, usage);
        }
        if let Shape::FailedStream { obfuscation } = self.shape {
            return check_failed_stream(a, obfuscation, self.says);
        }
        let v: Value = serde_json::from_slice(&a.body).map_err(|e| format!("not JSON: {e}"))?;
        match self.shape {
            Shape::Completion(x0ne) => {
                exactly(
                    "the answer",
                    &v,
                    &[
                        "choices", "created", "id", "model", "object", "usage", "x0ne",
                    ],
                )?;
                let id = v["id"].as_str().unwrap_or_default();
                let hex = id.strip_prefix("chatcmpl-").unwrap_or_default();
                if hex.len() != 24 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err(format!("id {}", v["id"]));
                }
                if !v["created"].is_u64() {
                    return Err(format!("created {}", v["created"]));
                }
                if v["object"] != "chat.completion" {
                    return Err(format!("object {}", v["object"]));
                }
                let choices = v["choices"].as_array().ok_or("choices not a list")?;
                if choices.len() != 1 {
                    return Err(format!("{} choices", choices.len()));
                }
                exactly(
                    "the choice",
                    &choices[0],
                    &["finish_reason", "index", "message"],
                )?;
                exactly("the message", &choices[0]["message"], &["content", "role"])?;
                if choices[0]["message"]["role"] != "assistant"
                    || !choices[0]["message"]["content"].is_string()
                {
                    return Err(format!("message {}", choices[0]["message"]));
                }
                exactly(
                    "usage",
                    &v["usage"],
                    &["completion_tokens", "prompt_tokens", "total_tokens"],
                )?;
                exactly("x0ne", &v["x0ne"], x0ne)?;
                let want = s.served.as_str();
                if v["model"] != want {
                    return Err(format!("model {}, not {want:?}", v["model"]));
                }
                Ok(())
            }
            Shape::Api(code) => {
                exactly("the error", &v, &["error"])?;
                openai_error(&v["error"], code, self.says)
            }
            Shape::Chunks { .. } | Shape::FailedStream { .. } => {
                unreachable!("checked above")
            }
        }
    }
}

/// OpenAI's `error` object: its four keys, our `type`, `code` as `code`.
fn openai_error(e: &Value, code: Option<&str>, says: &str) -> Result<(), String> {
    exactly("error", e, &["code", "message", "param", "type"])?;
    if e["type"] != crate::openai::error::TYPE {
        return Err(format!("error.type {}", e["type"]));
    }
    if !(e["param"].is_null() || e["param"].is_string()) {
        return Err(format!("error.param {}", e["param"]));
    }
    if e["code"] != code.map_or(Value::Null, Value::from) {
        return Err(format!("error.code {}, not {code:?}", e["code"]));
    }
    message_says(&e["message"], says)
}

/// The events of a stream, by the strict reader the relay uses, ending at an
/// event's end: every one but the last a `chat.completion.chunk`, then the last.
fn read_stream(a: &Answer) -> Result<(Vec<Value>, crate::sse::Data), String> {
    let mut reader = crate::sse::SseReader::new();
    let mut events = Vec::new();
    reader
        .push(&a.body, &mut events)
        .map_err(|e| format!("not strict SSE: {e}"))?;
    if !reader.at_boundary() {
        return Err("a partial event at the end".into());
    }
    let last = events.pop().ok_or("no event")?;
    let mut chunks = Vec::new();
    for event in events {
        match event {
            crate::sse::Data::Json(c) if c["object"] == "chat.completion.chunk" => chunks.push(c),
            other => return Err(format!("{other:?} before the last event")),
        }
    }
    Ok((chunks, last))
}

fn obfuscated(chunks: &[Value], obfuscation: bool) -> Result<(), String> {
    if chunks
        .iter()
        .any(|c| c.get("obfuscation").is_some() != obfuscation)
    {
        return Err(format!(
            "obfuscation not in every chunk or in none: {obfuscation}"
        ));
    }
    Ok(())
}

fn finishing(chunks: &[Value]) -> Vec<&Value> {
    chunks
        .iter()
        .filter(|c| c["choices"][0]["finish_reason"].is_string())
        .collect()
}

/// `[DONE]` last, the usage chunk before it as `usage` says, one finishing
/// chunk that carries `x0ne`, `obfuscation` in every other chunk or in none.
fn check_chunks(a: &Answer, obfuscation: bool, usage: Usage) -> Result<(), String> {
    let (mut chunks, last) = read_stream(a)?;
    if last != crate::sse::Data::Done {
        return Err("no [DONE] last".into());
    }
    let nulls = match usage {
        Usage::None => false,
        Usage::Chunk | Usage::ChunkAndNulls => {
            let u = chunks.pop().ok_or("no usage chunk")?;
            if u["choices"] != json!([]) || u.get("obfuscation").is_some() {
                return Err(format!("the usage chunk {u}"));
            }
            exactly(
                "usage",
                &u["usage"],
                &["completion_tokens", "prompt_tokens", "total_tokens"],
            )?;
            matches!(usage, Usage::ChunkAndNulls)
        }
    };
    if chunks
        .iter()
        .any(|c| c.get("usage").is_some() != nulls || !c["usage"].is_null())
    {
        return Err(format!(
            "usage: null not in every other chunk or in none: {nulls}"
        ));
    }
    obfuscated(&chunks, obfuscation)?;
    let finishing = finishing(&chunks);
    if finishing.len() != 1 || finishing[0].get("x0ne").is_none() {
        return Err(format!("{} finishing chunks, x0ne on it", finishing.len()));
    }
    Ok(())
}

/// Chunks, none finishing, then `generation_failed` saying `says`, last.
/// Measured 2026-10-25, one run: no chunk before the error from the TD,
/// the cloud TD and the relay.
fn check_failed_stream(a: &Answer, obfuscation: bool, says: &str) -> Result<(), String> {
    let (chunks, last) = read_stream(a)?;
    let crate::sse::Data::Json(event) = last else {
        return Err("[DONE] last".into());
    };
    exactly("the last event", &event, &["error"])?;
    exactly("its error", &event["error"], &["code", "message", "type"])?;
    if event["error"]["type"] != "x0ne_engine" || event["error"]["code"] != "generation_failed" {
        return Err(format!("the error {}", event["error"]));
    }
    message_says(&event["error"]["message"], says)?;
    if !finishing(&chunks).is_empty() {
        return Err("a finishing chunk before the error".into());
    }
    obfuscated(&chunks, obfuscation)
}

/// Every row of [`rows`] against `s`; one failure lists every row that differs.
async fn play(s: &Surface) {
    let rows = rows();
    let mut wrong = Vec::new();
    for row in &rows {
        let expect = (s.column)(row);
        let none = json!({});
        let (target, situation) = match row.when {
            When::Ready => (&s.ready, &none),
            When::Full => (&s.full, &none),
            When::Panicking => (&s.panicking.0, &s.panicking.1),
            When::Failing => (&s.ready, &s.failing),
        };
        let body = row.asked.bytes(&s.served, situation);
        let answer = send(target, &row.asked, body).await;
        if let Err(why) = expect.check(&answer, s) {
            wrong.push(format!(
                "  {}: {why}\n    got {} {:?} {}",
                row.name,
                answer.status,
                answer.content_type,
                answer.excerpt()
            ));
        }
    }
    let listed = models_of(&s.ready).await;
    if let Err(why) = check_models(&listed, s) {
        wrong.push(format!(
            "  GET /v1/models: {why}\n    got {} {:?} {}",
            listed.status,
            listed.content_type,
            listed.excerpt()
        ));
    }
    assert!(
        wrong.is_empty(),
        "{} of {} rows differ on {}:\n{}",
        wrong.len(),
        rows.len(),
        s.name,
        wrong.join("\n")
    );
}

// ── The four routers ────────────────────────────────────────────────────────

/// Every row of the table against the TD served directly
/// (`server::serve_with`): the console's `Host` list, the queue, the padding.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_td_served_directly_answers_the_table() {
    use crate::server::tests::spawn_serving;
    let model = Scripted::default();
    let ready = spawn_serving(Box::new(model.clone()), Serving::default()).await;
    let one_place = Serving {
        admission: Admission::new(0),
        ..Serving::default()
    };
    let _taken = one_place.admission.try_enter().expect("the one place");
    let full = spawn_serving(Box::new(model), one_place).await;
    play(&Surface {
        name: "the TD served directly",
        column: |r| &r.td,
        served: TD_DIGEST.into(),
        panic_payload: PANIC_PAYLOAD,
        ready: http(format!("http://{ready}")),
        full: http(format!("http://{full}")),
        panicking: (http(format!("http://{ready}")), says(PANIC)),
        failing: says(FAIL),
        lists_models: true,
    })
    .await;
}

/// The relay and its TD on the CPU product, ready and with one queue place
/// that a held question takes (through the TD's routes, not the relay).
async fn cloud_pair() -> (
    crate::cloud_e2e_tests::ContractUp,
    crate::cloud_e2e_tests::ContractUp,
    OpensOnDrop,
) {
    use crate::cloud_e2e_tests::contract_up;
    let ready = contract_up(Scripted::default().maker(), None).await;
    let held = Scripted::default();
    let full = contract_up(held.maker(), Some(0)).await;
    let routes = full.td.clone();
    let hold = raw("").without_content_type();
    tokio::spawn(async move {
        let body = says(HOLD).to_string().into_bytes();
        send(
            &Target::Routes(routes),
            &Asked {
                content_type: true,
                ..hold
            },
            body,
        )
        .await
    });
    held.until_entered(1).await;
    (ready, full, OpensOnDrop(held.latch.clone()))
}

/// Every row of the table against the cloud TD (`cloud_serve::router`), in
/// its phase `Ready`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_cloud_td_answers_the_table() {
    let (ready, full, _open) = cloud_pair().await;
    play(&Surface {
        name: "the cloud TD",
        column: |r| &r.cloud,
        served: ready.served.into(),
        panic_payload: PANIC_PAYLOAD,
        ready: Target::Routes(ready.td.clone()),
        full: Target::Routes(full.td.clone()),
        panicking: (Target::Routes(ready.td.clone()), says(PANIC)),
        failing: says(FAIL),
        lists_models: false,
    })
    .await;
}

/// Every row of the table against the relay (`local_api::router`), in front
/// of the cloud TD.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_relay_answers_the_table() {
    let (ready, full, _open) = cloud_pair().await;
    let relay = |up: &crate::cloud_e2e_tests::ContractUp| Target::Http {
        base: up.relay.clone(),
        bearer: Some(up.bearer.clone()),
    };
    play(&Surface {
        name: "the relay",
        column: |r| &r.relay,
        served: ready.served.into(),
        panic_payload: PANIC_PAYLOAD,
        ready: relay(&ready),
        full: relay(&full),
        panicking: (relay(&ready), says(PANIC)),
        failing: says(FAIL),
        lists_models: true,
    })
    .await;
}

// ── What the table leaves to the TD's own tests ─────────────────────────────

/// The bytes hyper writes before a body: status line, headers, blank line.
fn head_len(r: &reqwest::Response) -> usize {
    let reason = r.status().canonical_reason().map_or(0, str::len);
    let headers: usize = r
        .headers()
        .iter()
        .map(|(name, value)| name.as_str().len() + 2 + value.len() + 2)
        .sum();
    "HTTP/1.1 200 ".len() + reason + 2 + headers + 2
}

/// What a client of the TD served directly reads of one answer.
struct Timed {
    status: u16,
    head: usize,
    content_type: String,
    pad_header: bool,
    body: Vec<u8>,
    took: Duration,
}

async fn post_td(addr: std::net::SocketAddr, body: &str, content_type: Option<&str>) -> Timed {
    let asked = std::time::Instant::now();
    let mut req = reqwest::Client::new()
        .post(format!("http://{addr}/v1/chat/completions"))
        .body(body.to_string());
    if let Some(ct) = content_type {
        req = req.header(reqwest::header::CONTENT_TYPE, ct);
    }
    let r = req.send().await.unwrap();
    let (status, head) = (r.status().as_u16(), head_len(&r));
    let content_type = r
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let pad_header = r.headers().contains_key("x0ne-pad");
    let body = r.bytes().await.unwrap().to_vec();
    Timed {
        status,
        head,
        content_type,
        pad_header,
        body,
        took: asked.elapsed(),
    }
}

fn floor() -> Duration {
    crate::stream::TICK * u32::try_from(crate::padding::MIN_TICKS).unwrap()
}

/// A whole answer the validator refuses, or one naming a model the TD does
/// not serve, leaves as an answer does: the body of a 200's size,
/// a head of a 200's length, a second after the request at the earliest.
/// `server.rs`'s own test covers the refusals made after the validator
/// (empty messages, `x0ne.adapter`, the 404) and the 500. A failure is a
/// refusal the host tells from an answer by its size or time.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_whole_answer_the_validator_refuses_is_padded_as_an_answer() {
    let addr =
        crate::server::tests::spawn_serving(Box::new(Scripted::default()), Serving::default())
            .await;
    let json = Some("application/json");
    let ok = post_td(addr, &says("hi").to_string(), json).await;
    assert_eq!(ok.status, 200);
    let mut cases = Vec::new();
    for (extra, status, says_) in [
        (json!({"logprobs": true}), 400, "logprobs"),
        (json!({"n": 2}), 400, "n: only 1"),
        (json!({"tools": weather_tool()}), 400, TOOLS),
        (json!({"stream": "yes"}), 400, "stream: must be"),
        (
            json!({"stream_options": {"include_usage": true}}),
            400,
            "only with stream: true",
        ),
        (json!({"temperature": 3}), 400, "temperature"),
        (json!({"messages": 5}), 422, "invalid type"),
        (
            json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}),
            422,
            "expected a string",
        ),
        (json!({"model": "gpt-4o"}), 404, "not served here"),
    ] {
        let mut body = says("hi");
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        cases.push((body.to_string(), status, says_));
    }
    cases.push(("[]".into(), 422, "the body must be a JSON object"));
    for (body, status, says_) in cases {
        let got = post_td(addr, &body, json).await;
        assert_eq!(got.status, status, "{body}");
        assert_eq!(
            got.body.len(),
            ok.body.len(),
            "{body}: not padded as an answer"
        );
        assert_eq!(got.head, ok.head, "{body}: a head of its own length");
        assert!(got.pad_header, "{body}: no x0ne-pad");
        assert!(got.took >= floor(), "{body}: left after {:?}", got.took);
        let error: Value = serde_json::from_slice(&got.body).unwrap();
        assert!(
            error["error"]["message"]
                .as_str()
                .is_some_and(|e| e.contains(says_)),
            "{body}: {error}"
        );
        assert_eq!(error["error"]["type"], crate::openai::error::TYPE, "{body}");
    }
}

/// A body axum's JSON extractor rejects never reaches the validator: it
/// leaves by a status of its own, in OpenAI's shape, padded as a whole
/// answer's refusal
/// (its sentence names where the JSON broke). The relay never sends such a
/// body to its TD. A failure is one the host tells from an answer.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_body_the_extractor_rejects_leaves_the_td_as_a_whole_refusal() {
    let addr =
        crate::server::tests::spawn_serving(Box::new(Scripted::default()), Serving::default())
            .await;
    let hi = says("hi").to_string();
    let ok = post_td(addr, &hi, Some("application/json")).await;
    assert_eq!(ok.status, 200);
    for (body, content_type, status) in [
        (r#"{"messages": ["#, Some("application/json"), 400),
        (hi.as_str(), None, 415),
        (hi.as_str(), Some("text/plain"), 415),
    ] {
        let got = post_td(addr, body, content_type).await;
        assert_eq!(got.status, status, "{body} {content_type:?}");
        assert!(
            got.content_type.starts_with("application/json"),
            "{content_type:?}: {}",
            got.content_type
        );
        let error: Value = serde_json::from_slice(&got.body).unwrap();
        assert_eq!(error["error"]["type"], crate::openai::error::TYPE);
        assert!(got.pad_header, "{content_type:?}: no x0ne-pad");
        assert_eq!(got.body.len(), ok.body.len(), "{content_type:?}");
        assert_eq!(
            got.head, ok.head,
            "{content_type:?}: a head of its own length"
        );
        assert!(
            got.took >= floor(),
            "{content_type:?}: left after {:?}",
            got.took
        );
    }
}

/// A refusal that names part of the request, a field
/// the validator does not know in a stream or where the extractor found the
/// JSON broken, leaves the TD as a whole answer's refusal, on both TD
/// routers: its status and text kept, padded to a size, at the floor; so
/// does a body without a JSON content type. They left at once, unpadded,
/// their size the length of the name they quote. A failure is a refusal
/// whose size or time follows the request.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refusal_naming_part_of_the_request_leaves_as_a_whole_refusal() {
    let addr =
        crate::server::tests::spawn_serving(Box::new(Scripted::default()), Serving::default())
            .await;
    let cloud = crate::cloud_e2e_tests::contract_up(Scripted::default().maker(), None).await;
    let json = Some("application/json");
    let ok = post_td(addr, &says("hi").to_string(), json).await;
    assert_eq!(ok.status, 200);
    let mut cases = Vec::new();
    for n in [4, 400] {
        let name = "k".repeat(n);
        let mut streamed = says("hi");
        streamed["stream"] = json!(true);
        streamed[name.as_str()] = json!(1);
        let unknown = format!("{}: not supported", crate::openai::error::quoted(&name));
        cases.push((streamed.to_string(), json, 400, unknown, n));
        let broken = format!(r#"{{"{name}": ["#);
        let parse = "Failed to parse the request body as JSON".to_string();
        cases.push((broken, json, 400, parse, n));
    }
    let typed = "Content-Type: application/json".to_string();
    cases.push((says("hi").to_string(), None, 415, typed, 0));
    let mut failed = Vec::new();
    for (body, content_type, status, says_, n) in cases {
        let td = post_td(addr, &body, content_type).await;
        let asked = chat(json!({}));
        let asked = if content_type.is_some() {
            asked
        } else {
            asked.without_content_type()
        };
        let t0 = std::time::Instant::now();
        let routes = Target::Routes(cloud.td.clone());
        let routed = send(&routes, &asked, body.clone().into_bytes()).await;
        let routed_took = t0.elapsed();
        eprintln!(
            "{status} ({says_:.40}, a name of {n} bytes): TD {} bytes, head {} \
             (a 200's: {} and {}), x0ne-pad {}, after {:?}; cloud TD {} bytes, after {routed_took:?}",
            td.body.len(),
            td.head,
            ok.body.len(),
            ok.head,
            td.pad_header,
            td.took,
            routed.body.len()
        );
        let error: Value = serde_json::from_slice(&td.body).unwrap();
        assert_eq!(td.status, status, "{error}");
        assert_eq!(routed.status, status, "{}", routed.text());
        assert!(
            error["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains(&says_)),
            "{error}"
        );
        if td.body.len() != ok.body.len()
            || td.head != ok.head
            || !td.pad_header
            || td.took < floor()
            || routed.body.len() != ok.body.len()
            || !routed.pad_header
            || routed_took < floor()
        {
            failed.push(format!("{status} {says_:.40} ({n})"));
        }
    }
    assert!(failed.is_empty(), "not as a whole refusal: {failed:?}");
}

/// A refusal quotes at most 64 bytes of a name the
/// request chose, cut on a character, on both TD routers: a name past
/// 16 KiB made it leave at the next Padme size. A failure is a refusal
/// larger than a 200, or one quoting more of the name.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_name_past_16_kib_is_quoted_cut_and_leaves_at_one_size() {
    let addr =
        crate::server::tests::spawn_serving(Box::new(Scripted::default()), Serving::default())
            .await;
    let cloud = crate::cloud_e2e_tests::contract_up(Scripted::default().maker(), None).await;
    let json = Some("application/json");
    let ok = post_td(addr, &says("hi").to_string(), json).await;
    assert_eq!(ok.status, 200);
    // 20001 bytes; the 64th byte falls inside an é, so 63 are quoted.
    let long = format!("k{}", "é".repeat(10_000));
    let cut = format!("k{}", "é".repeat(31));
    let too_much = format!("k{}", "é".repeat(32));
    let named = || {
        let mut named = serde_json::Map::new();
        named.insert(long.clone(), json!(1));
        named
    };
    let mut bodies = Vec::new();
    for stream in [false, true] {
        let with = |key: &str, value: Value| {
            let mut body = says("hi");
            body["stream"] = json!(stream);
            body[key] = value;
            body.to_string()
        };
        bodies.push(with(&long, json!(1)));
        bodies.push(with("x0ne", Value::Object(named())));
        let mut message = named();
        message.insert("role".into(), json!("user"));
        message.insert("content".into(), json!("hi"));
        bodies.push(with("messages", json!([message])));
    }
    let mut options = says("hi");
    options["stream"] = json!(true);
    options["stream_options"] = Value::Object(named());
    bodies.push(options.to_string());
    bodies.push(format!(r#"{{"{long}": ["#));
    let routes = Target::Routes(cloud.td.clone());
    let mut failed = Vec::new();
    for body in &bodies {
        let td = post_td(addr, body, json).await;
        let routed = send(&routes, &chat(json!({})), body.clone().into_bytes()).await;
        for (on, status, len, said) in [
            (
                "TD",
                td.status,
                td.body.len(),
                String::from_utf8_lossy(&td.body).into_owned(),
            ),
            ("cloud TD", routed.status, routed.body.len(), routed.text()),
        ] {
            let error: Value = serde_json::from_str(&said).unwrap();
            let message = error["error"]["message"].as_str().unwrap_or_default();
            let shown: String = message.chars().filter(|c| *c != 'é').take(60).collect();
            eprintln!(
                "{on}: {status}, {len} bytes (a 200: {}), message of {} bytes: {shown}",
                ok.body.len(),
                message.len()
            );
            assert_eq!(status, 400, "{on}: {shown}");
            if len != ok.body.len() || message.contains(&too_much) || !message.contains(&cut) {
                failed.push(format!("{on}: {len} bytes, {shown}"));
            }
        }
    }
    assert!(
        failed.is_empty(),
        "{} of {}: {failed:#?}",
        failed.len(),
        2 * bodies.len()
    );
}

/// The chunks of the TD's stream, by their exact keys, and the size of
/// their frames for a given id served (that id, not the request's `model`),
/// with the padding left in the first. The
/// size tests compare frames with each other: a field added to the skeleton
/// would change the size of all of them and pass. Here it changes the
/// numbers below. A failure is a chunk that changed shape: say so in
/// `docs/api.md`, then update the numbers.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_td_stream_chunks_keep_their_keys_and_their_frame_size() {
    let widest = "m".repeat(crate::server::MAX_STREAM_MODEL_BYTES);
    for (model, frame, role_padding) in [("m", 512, 292), (widest.as_str(), 768, 293)] {
        let served = Scripted {
            digest: Some(model.into()),
            ..Scripted::default()
        };
        let addr = crate::server::tests::spawn_serving(Box::new(served), Serving::default()).await;
        let mut body = says("hi");
        body["model"] = json!(model);
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage": true});
        let got = post_td(addr, &body.to_string(), Some("application/json")).await;
        assert_eq!(got.status, 200);
        assert_eq!(got.content_type, "text/event-stream");
        let first = got.body.windows(2).position(|w| w == b"\n\n").unwrap() + 2;
        assert_eq!(
            first,
            frame,
            "the frame size for a model of {} bytes",
            model.len()
        );
        let mut reader = crate::sse::SseReader::new();
        let mut events = Vec::new();
        reader.push(&got.body, &mut events).unwrap();
        assert_eq!(events.last(), Some(&crate::sse::Data::Done));
        let chunks: Vec<&Value> = events
            .iter()
            .filter_map(|e| match e {
                crate::sse::Data::Json(v) => Some(v),
                crate::sse::Data::Done => None,
            })
            .collect();
        assert_eq!(chunks.len(), 4, "role, text, final, usage: {chunks:?}");
        let (role, text, last, usage) = (chunks[0], chunks[1], chunks[2], chunks[3]);
        let skeleton = ["choices", "created", "id", "model", "object"];
        let with = |more: &[&'static str]| {
            let mut all = skeleton.to_vec();
            all.extend_from_slice(more);
            all
        };
        exactly("the role chunk", role, &with(&["obfuscation"])).unwrap();
        exactly("the text chunk", text, &with(&["obfuscation"])).unwrap();
        exactly("the final chunk", last, &with(&["obfuscation", "x0ne"])).unwrap();
        exactly("the usage chunk", usage, &with(&["usage"])).unwrap();
        for c in [role, text, last] {
            exactly(
                "a choice",
                &c["choices"][0],
                &["delta", "finish_reason", "index"],
            )
            .unwrap();
            assert_eq!(c["choices"].as_array().unwrap().len(), 1, "{c}");
        }
        assert_eq!(
            role["choices"][0]["delta"],
            json!({"role": "assistant", "content": ""})
        );
        assert_eq!(text["choices"][0]["delta"], json!({"content": "ok"}));
        assert_eq!(last["choices"][0]["delta"], json!({}));
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        exactly("x0ne", &last["x0ne"], TD_X0NE).unwrap();
        assert_eq!(usage["choices"], json!([]));
        exactly(
            "usage",
            &usage["usage"],
            &["completion_tokens", "prompt_tokens", "total_tokens"],
        )
        .unwrap();
        assert_eq!(
            role["obfuscation"].as_str().unwrap().len(),
            role_padding,
            "the padding left in the role frame for a model of {} bytes",
            model.len()
        );
    }
}
