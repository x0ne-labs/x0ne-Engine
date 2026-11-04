//! I6 — the local OpenAI endpoint the relay opens once the TD is verified.
//!
//! Loopback only, bearer from a 0600 file, `Host` pinned, every request
//! validated here with the server's own rules before anything leaves the
//! machine. A whole answer is returned once its receipt verifies. A stream
//! passes its text on as it comes; its final chunk, usage and `[DONE]` go out
//! once the receipt, the usage and the clean end of the TD's body all check,
//! and an error event takes their place otherwise. Either ends the body clean.
//!
//! The SDK's sealed path rides on the same endpoint (`/v2/e2e/key`,
//! `/v2/e2e/chat`, `/v2/e2e/collateral`, behind the same guard): the relay
//! carries a key document it does not verify and a question and an answer
//! sealed to the TD's key, which it cannot open. What it sees of them: the
//! sizes, which the SDK pads to buckets; the timing; the caller's requests
//! and the TD's statuses. The model is the session's, named by the caller.
//!
//! No log macro in this file: it handles the text of the answer.

use crate::attest_v2::Document;
use crate::openai::error::ApiError;
use crate::receipt::{self, CommitmentKind, CommitmentOpening, InferenceReceipt};
use crate::remote_verify::{
    ChatStream, FrameSource, Report, SealedStream, TdAnswer, TdClient, TdError, CHAT_TIMEOUT,
};
use crate::sse::{Chunks, Data, SseReader};
use axum::body::Body;
use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use hyper::body::{Bytes, Frame};
use serde_json::{json, Value};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use tokio::time::Instant;
use x0ne_sdk::attest::E2eStream;

pub const DEFAULT_LISTEN: &str = "127.0.0.1:8478";
pub const KEY_FILE_NAME: &str = "local.key";
pub const SUBDIR: &str = "x0ne-engine";
/// New requests are refused this long before `expires_at_unix`.
pub const EXPIRY_GUARD_SECS: u64 = 60;
/// No byte from the TD for this long cuts a stream: the TD writes every 100 ms.
pub const FRAME_SILENCE: Duration = Duration::from_secs(30);
/// No event for this long cuts it too: padding alone keeps nothing open.
pub const CONTENT_SILENCE: Duration = Duration::from_secs(120);
/// The local client took nothing for this long: its stream is abandoned.
pub const WRITE_STALL: Duration = Duration::from_secs(30);
/// [`LocalServer::stop`] waits this long for open connections, then drops them.
pub const STOP_WAIT: Duration = Duration::from_secs(5);
/// A client that leaves after this long without a byte from the TD, ten
/// frames missed at 100 ms, leaves a frozen stream: a cut, not a departure.
pub const FROZEN: Duration = Duration::from_secs(1);
/// A TD's body ends at once after its error event; past this, its
/// connection is closed rather than kept for the next request.
pub const ERROR_END: Duration = Duration::from_secs(1);
/// Items queued for the local client before the relay waits on it.
const CLIENT_QUEUE: usize = 4;

/// The verified session the endpoint fronts.
pub struct Session {
    pub client: TdClient,
    pub document: Arc<Document>,
    pub report: Report,
    pub provider: String,
    pub machine_type: String,
    pub expires_at_unix: u64,
    pub started_at_unix: u64,
    /// The TD's context bound from /v2/ready; None from a TD before 0.2.0-beta.6.
    pub context_tokens: Option<u64>,
    /// /v2/ready says the TD streams; false, a stream is answered whole and emulated here.
    pub streaming: bool,
    /// The TD's prefill chunk at the full context, and by prompt length.
    pub prefill_chunk: Option<u64>,
    pub prefill_chunks: Option<Vec<(u64, u64)>>,
    /// The SDK's sealed path, carried; None: its routes answer 404.
    pub e2e: Option<E2eCarrier>,
}

/// What the relay needs to carry an SDK's sealed path: where Intel's and
/// Google's signed collateral comes from, the last one fetched, and the
/// cadence of the session's sealed streams, which bounds their bytes.
pub struct E2eCarrier {
    pub http: reqwest::Client,
    pub sources: crate::remote_verify::CollateralSources,
    /// The session config's `e2e_stream`, as the TD and the SDK read it.
    pub cadence: E2eStream,
    collateral: tokio::sync::Mutex<Option<(u64, Value)>>,
}

impl E2eCarrier {
    pub fn new(
        http: reqwest::Client,
        sources: crate::remote_verify::CollateralSources,
        cadence: E2eStream,
    ) -> Self {
        Self {
            http,
            sources,
            cadence,
            collateral: tokio::sync::Mutex::new(None),
        }
    }
}

// ── The sealed stream, carried frame by frame ────────────────────────────────

/// The `code` of a sealed stream the relay cut, in `/x0ne/v1/status`
/// (`sealed_streams.cut`): the TD wrote more bytes than the session's
/// cadence allows (the session stops, 65).
pub const SEALED_STREAM_TOO_LARGE: &str = "sealed_stream_too_large";
/// No frame for [`FRAME_SILENCE`] from a TD that writes every tick: lost.
pub const SEALED_STREAM_SILENT: &str = "sealed_stream_silent";
/// Still open [`CHAT_TIMEOUT`] after its request: lost.
pub const SEALED_STREAM_TOO_LONG: &str = "sealed_stream_too_long";
/// The local client left, or took nothing for [`WRITE_STALL`].
pub const SEALED_STREAM_CLIENT_GONE: &str = "sealed_stream_client_gone";
/// The TD's body broke short of its clean end: lost.
pub const SEALED_STREAM_TD_BROKE: &str = "sealed_stream_td_broke";
/// A sealed request over the SDK's largest, refused before the TD (413).
pub const SEALED_REQUEST_TOO_LARGE: &str = "sealed_request_too_large";

/// How a sealed stream ended short of its clean end.
enum SealedExit {
    /// Bytes read when the cadence's bound was passed.
    TooLarge(usize),
    Silent,
    TooLong,
    ClientGone,
    TdBroke,
    Stopped,
}

impl SealedExit {
    fn code(&self) -> &'static str {
        match self {
            SealedExit::TooLarge(_) => SEALED_STREAM_TOO_LARGE,
            SealedExit::Silent => SEALED_STREAM_SILENT,
            SealedExit::TooLong => SEALED_STREAM_TOO_LONG,
            SealedExit::ClientGone => SEALED_STREAM_CLIENT_GONE,
            SealedExit::TdBroke => SEALED_STREAM_TD_BROKE,
            SealedExit::Stopped => "session_stopped",
        }
    }
}

/// The session's sealed streams: carried to their clean end, or cut and why.
#[derive(Default)]
pub struct SealedCounts {
    carried: std::sync::atomic::AtomicU64,
    cuts: std::sync::Mutex<std::collections::BTreeMap<&'static str, u64>>,
    last_cut: std::sync::Mutex<Option<(&'static str, u64)>>,
}

impl SealedCounts {
    fn carried(&self) {
        self.carried.fetch_add(1, Ordering::SeqCst);
    }

    fn cut(&self, code: &'static str, at_unix: u64) {
        *self
            .cuts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(code)
            .or_insert(0) += 1;
        *self.last_cut.lock().unwrap_or_else(|e| e.into_inner()) = Some((code, at_unix));
    }

    /// `sealed_streams` in `/x0ne/v1/status`.
    fn json(&self) -> Value {
        let cuts: serde_json::Map<String, Value> = self
            .cuts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|(code, n)| ((*code).to_string(), json!(n)))
            .collect();
        let last = *self.last_cut.lock().unwrap_or_else(|e| e.into_inner());
        json!({
            "carried": self.carried.load(Ordering::SeqCst),
            "cut": cuts,
            "last_cut": last.map(|(code, at_unix)| json!({ "code": code, "at_unix": at_unix })),
        })
    }
}

/// A collateral fetched for an SDK is served again for this long. Intel
/// issues it about hourly; the SDK refuses one older than a day.
pub const E2E_COLLATERAL_REUSE_SECS: u64 = 600;

pub struct LocalState {
    pub session: Session,
    bearer_hex: zeroize::Zeroizing<String>,
    /// Set by [`start`] once bound: `--listen 127.0.0.1:0` is legal.
    port: AtomicU16,
    lost: AtomicBool,
    /// Set by [`Self::fail`] and [`LocalServer::stop`]: every stream ends, `chat` refuses.
    stopping: watch::Sender<bool>,
    fatal: watch::Sender<Option<(i32, String)>>,
    clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    /// The TCB disclosure of the last receipt that verified: what the TD's
    /// standing said about the platform's patch level at its last answer.
    last_receipt_tcb: std::sync::Mutex<Option<crate::tcb_disclosure::TcbDisclosure>>,
    /// The sealed streams carried and cut, for `/x0ne/v1/status`.
    pub sealed: SealedCounts,
}

impl LocalState {
    pub fn new(
        session: Session,
        bearer: &[u8; 32],
        clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    ) -> (Arc<Self>, watch::Receiver<Option<(i32, String)>>) {
        let (fatal, rx) = watch::channel(None);
        (
            Arc::new(Self {
                session,
                bearer_hex: zeroize::Zeroizing::new(crate::cloud_config::hex(bearer)),
                port: AtomicU16::new(0),
                lost: AtomicBool::new(false),
                stopping: watch::channel(false).0,
                fatal,
                clock,
                last_receipt_tcb: std::sync::Mutex::new(None),
                sealed: SealedCounts::default(),
            }),
            rx,
        )
    }

    pub fn mark_lost(&self) {
        self.lost.store(true, Ordering::SeqCst);
    }

    pub fn is_stopping(&self) -> bool {
        *self.stopping.borrow()
    }

    /// Every stream in flight ends with `session_stopped`; new requests are refused.
    pub fn stop_streams(&self) {
        self.stopping.send_replace(true);
    }

    fn fail(&self, code: i32, reason: String) {
        self.stop_streams();
        let _ = self.fatal.send(Some((code, reason)));
    }
}

/// Loopback only: anything else is refused before a socket exists.
pub fn parse_listen(spec: &str) -> Result<std::net::SocketAddr, String> {
    let addr: std::net::SocketAddr = spec
        .parse()
        .map_err(|e| format!("--listen {spec:?}: {e}"))?;
    if !addr.ip().is_loopback() {
        return Err(format!(
            "--listen {spec:?}: the local endpoint binds the loopback only"
        ));
    }
    Ok(addr)
}

/// `<runtime_dir>/x0ne-engine/local.key`, 0600 in a 0700 directory, the
/// 32-byte key as 64 hex characters and a newline. Both directories are
/// created 0700 when absent; one that exists must be the caller's own (a
/// directory, not a link, owned by the caller's uid, closed to other
/// accounts), or nothing is written: a directory another account made ahead
/// (`<temp>/x0ne-engine-<uid>`, `bin/proxy.rs`) gets no key.
pub fn write_key_file(runtime_dir: &Path, key: &[u8; 32]) -> Result<PathBuf, String> {
    let dir = runtime_dir.join(SUBDIR);
    #[cfg(unix)]
    {
        private_dir(runtime_dir, false)?;
        private_dir(&dir, true)?;
    }
    #[cfg(not(unix))]
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(KEY_FILE_NAME);
    let _ = std::fs::remove_file(&path);
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    use std::io::Write as _;
    let mut f = opts
        .open(&path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    f.write_all(format!("{}\n", crate::cloud_config::hex(key)).as_bytes())
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

/// `dir` created 0700 when absent (its parents too), then held to
/// [`foreign_dir`], read without following a link.
#[cfg(unix)]
fn private_dir(dir: &Path, leaf: bool) -> Result<(), String> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?;
    let meta = std::fs::symlink_metadata(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    // SAFETY: getuid has no preconditions.
    let uid = unsafe { libc::getuid() };
    foreign_dir(dir, &meta, uid, leaf).map_or(Ok(()), Err)
}

/// Why `meta` (from `lstat`) is not a directory of `uid`'s own to hold the
/// key: a symbolic link, not a directory, another uid's, or a mode that lets
/// others write in it (`leaf`, the key's own directory: in at all). None
/// when it is one.
#[cfg(unix)]
fn foreign_dir(dir: &Path, meta: &std::fs::Metadata, uid: u32, leaf: bool) -> Option<String> {
    use std::os::unix::fs::MetadataExt;
    let d = dir.display();
    if meta.file_type().is_symlink() {
        return Some(format!(
            "{d}: a symbolic link, refused: the key's directories must be your own"
        ));
    }
    if !meta.is_dir() {
        return Some(format!("{d}: not a directory"));
    }
    if meta.uid() != uid {
        return Some(format!(
            "{d}: owned by uid {}, not by you (uid {uid}): another account's directory, refused",
            meta.uid()
        ));
    }
    let others = if leaf { 0o077 } else { 0o022 };
    if meta.mode() & others != 0 {
        return Some(format!(
            "{d}: mode {:o} lets other accounts in, refused: chmod 700 it, or remove it",
            meta.mode() & 0o7777
        ));
    }
    None
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// OpenAI's shape (`crate::openai::error`), `param` and `code` null.
fn error(status: StatusCode, message: impl Into<String>) -> Response {
    ApiError::new(status, message).into_response()
}

/// [`error`] with the `code` a stream's error event would carry: the session is over.
fn error_coded(status: StatusCode, message: impl Into<String>, code: &'static str) -> Response {
    ApiError::new(status, message).code(code).into_response()
}

/// `Host` pinned to the loopback names, then the bearer. Everything else
/// is refused before a handler runs.
async fn guard(
    State(st): State<Arc<LocalState>>,
    headers: HeaderMap,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let host = headers
        .get("host")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let port = st.port.load(Ordering::SeqCst);
    let allowed = [
        format!("127.0.0.1:{port}"),
        format!("localhost:{port}"),
        format!("[::1]:{port}"),
    ];
    if !allowed.iter().any(|a| a == host) {
        return error(
            StatusCode::FORBIDDEN,
            format!("Host {host:?} is not this endpoint"),
        );
    }
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    if !constant_time_eq(token.as_bytes(), st.bearer_hex.as_bytes()) {
        let mut refused = error(
            StatusCode::UNAUTHORIZED,
            "missing or wrong bearer: read the key file",
        );
        refused.headers_mut().insert(
            axum::http::header::WWW_AUTHENTICATE,
            axum::http::HeaderValue::from_static("Bearer"),
        );
        return refused;
    }
    next.run(req).await
}

/// `GET /v1/models`: the session's model, `created` when the session began.
async fn models(State(st): State<Arc<LocalState>>) -> Json<Value> {
    use crate::openai::models::{list, ModelCard};
    let report = &st.session.report;
    Json(list(&[ModelCard {
        id: report.model_name.clone(),
        created: st.session.started_at_unix,
        x0ne: json!({ "digest_hex": report.model_digest_hex, "attested": true }),
    }]))
}

async fn status(State(st): State<Arc<LocalState>>) -> Json<Value> {
    let now = (st.clock)();
    let s = &st.session;
    let state = if st.lost.load(Ordering::SeqCst) {
        "lost"
    } else if now + EXPIRY_GUARD_SECS >= s.expires_at_unix {
        "expiring"
    } else {
        "serving"
    };
    Json(json!({
        "state": state,
        "verified": true,
        "provider": s.provider,
        "machine_type": s.machine_type,
        "model": {
            "name": s.report.model_name,
            "digest_hex": s.report.model_digest_hex,
            "context_tokens": s.context_tokens,
            "prefill_chunk": s.prefill_chunk,
            "prefill_chunks": s.prefill_chunks,
        },
        "session_public_key_hex": s.report.session_pubkey_hex,
        "started_at_unix": s.started_at_unix,
        "expires_at_unix": s.expires_at_unix,
        "seconds_left": s.expires_at_unix.saturating_sub(now),
        "summary": s.report.summary,
        "limitations": s.report.limitations,
        "verdict": s.report.verdict,
        // The verdict's TCB disclosure, then the last receipt's (None before
        // the first answer): the patch level the platform was named at.
        "tcb": s.report.tcb,
        "tcb_last_receipt": *st.last_receipt_tcb.lock().unwrap_or_else(|e| e.into_inner()),
        // The sealed streams carried to their end, and every cut by its code.
        "sealed_streams": st.sealed.json(),
        "estimated_cost_usd": Value::Null,
    }))
}

/// Why the session takes no new request now, if it does not.
fn not_serving(st: &LocalState) -> Option<Response> {
    if st.lost.load(Ordering::SeqCst) {
        return Some(error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the attested VM is lost; the session is over",
        ));
    }
    if st.is_stopping() {
        return Some(error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the session is stopping: no new requests",
        ));
    }
    let now = (st.clock)();
    if now + EXPIRY_GUARD_SECS >= st.session.expires_at_unix {
        return Some(error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "the session expires at {}: no new requests",
                st.session.expires_at_unix
            ),
        ));
    }
    None
}

fn no_sealed_path() -> Response {
    error(
        StatusCode::NOT_FOUND,
        "this relay carries no sealed path for this session",
    )
}

/// `GET /v2/e2e/key?nonce=<64 hex>`: the TD's key document for the SDK's
/// nonce, as the TD sent it, on a new connection (one quote each). Not
/// verified here: the SDK verifies it, for its own nonce.
async fn e2e_key(
    State(st): State<Arc<LocalState>>,
    axum::extract::RawQuery(query): axum::extract::RawQuery,
) -> Response {
    if st.session.e2e.is_none() {
        return no_sealed_path();
    }
    if let Some(refused) = not_serving(&st) {
        return refused;
    }
    let nonce = match query
        .as_deref()
        .and_then(|q| q.strip_prefix("nonce="))
        .map(crate::cloud_config::unhex32)
    {
        Some(Ok(n)) => n,
        _ => return error(StatusCode::BAD_REQUEST, "expected ?nonce=<64 hex>"),
    };
    match st.session.client.e2e_key(&nonce).await {
        Ok((status, bytes)) => (
            status,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            bytes,
        )
            .into_response(),
        Err(e) => unreachable_td(&st, e),
    }
}

/// `POST /v2/e2e/chat`: the SDK's sealed request, carried to the TD as it
/// came, and the TD's sealed answer as it came; the relay holds no key that
/// opens either. A whole request (`message/ohttp-req`, at most the SDK's
/// largest) gets the whole answer; a chunked one
/// (`message/ohttp-chunked-req`, at most the SDK's largest of that kind)
/// gets the TD's chunked answer frame by frame ([`relay_sealed_stream`]).
/// Over its largest: 413, [`SEALED_REQUEST_TOO_LARGE`], before the TD.
async fn e2e_chat(
    State(st): State<Arc<LocalState>>,
    headers: HeaderMap,
    body: Result<Bytes, axum::extract::rejection::BytesRejection>,
) -> Response {
    if st.session.e2e.is_none() {
        return no_sealed_path();
    }
    let media = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok());
    let chunked = match media {
        Some(x0ne_sdk::ohttp::MEDIA_TYPE_REQUEST) => false,
        Some(x0ne_sdk::ohttp::MEDIA_TYPE_CHUNKED_REQUEST) => true,
        _ => {
            return error(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                format!(
                    "a sealed request is {} or {}",
                    x0ne_sdk::ohttp::MEDIA_TYPE_REQUEST,
                    x0ne_sdk::ohttp::MEDIA_TYPE_CHUNKED_REQUEST
                ),
            )
        }
    };
    let largest = if chunked {
        x0ne_sdk::e2e::MAX_SEALED_STREAM_REQUEST
    } else {
        x0ne_sdk::e2e::MAX_SEALED_REQUEST
    };
    let too_large = || {
        error_coded(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("a sealed request of this kind is at most {largest} bytes"),
            SEALED_REQUEST_TOO_LARGE,
        )
    };
    let body = match body {
        Ok(b) if b.len() <= largest => b,
        Ok(_) => return too_large(),
        Err(rejected) if rejected.status() == StatusCode::PAYLOAD_TOO_LARGE => return too_large(),
        Err(rejected) => return error(rejected.status(), rejected.body_text()),
    };
    if let Some(refused) = not_serving(&st) {
        return refused;
    }
    if chunked {
        return e2e_chat_streamed(st, body).await;
    }
    match st.session.client.e2e_chat(body.to_vec()).await {
        Ok((status, retry_after, bytes)) => {
            let media = if status.is_success() {
                x0ne_sdk::ohttp::MEDIA_TYPE_RESPONSE
            } else {
                "application/json"
            };
            let mut r =
                (status, [(axum::http::header::CONTENT_TYPE, media)], bytes).into_response();
            if let Some(secs) = retry_after {
                r.headers_mut()
                    .insert(axum::http::header::RETRY_AFTER, secs.into());
            }
            r
        }
        Err(e) => unreachable_td(&st, e),
    }
}

/// A chunked sealed request to the TD: the 200 once the TD's is in, with
/// its content type, then its frames as they come, by
/// [`relay_sealed_stream`]. The TD's refusal is carried whole, as it came.
async fn e2e_chat_streamed(st: Arc<LocalState>, sealed: Bytes) -> Response {
    let started = Instant::now();
    let source = match st
        .session
        .client
        .e2e_chat_stream(sealed.to_vec(), started + CHAT_TIMEOUT)
        .await
    {
        Ok(SealedStream::Open {
            chunked: true,
            body,
        }) => body,
        Ok(SealedStream::Open { .. }) => {
            st.fail(
                65,
                format!(
                    "sealed stream protocol: a 200 that is not {}",
                    x0ne_sdk::ohttp::MEDIA_TYPE_CHUNKED_RESPONSE
                ),
            );
            return error_coded(
                StatusCode::BAD_GATEWAY,
                "the attested VM answered a sealed stream with something else; the session is being stopped",
                "stream_protocol",
            );
        }
        Ok(SealedStream::Refused {
            status,
            retry_after,
            body,
        }) => {
            let mut r = (
                status,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                body,
            )
                .into_response();
            if let Some(secs) = retry_after {
                r.headers_mut()
                    .insert(axum::http::header::RETRY_AFTER, secs.into());
            }
            return r;
        }
        Err(e) => return unreachable_td(&st, e),
    };
    let (tx, rx) = mpsc::channel(CLIENT_QUEUE);
    tokio::spawn(relay_sealed_stream(
        st,
        source,
        tx,
        Arc::new(TokioClock),
        started,
    ));
    (
        StatusCode::OK,
        [
            (
                axum::http::header::CONTENT_TYPE,
                x0ne_sdk::ohttp::MEDIA_TYPE_CHUNKED_RESPONSE,
            ),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
        ],
        Body::new(RelayBody {
            rx,
            flow: Flow::Open,
        }),
    )
        .into_response()
}

/// `GET /v2/e2e/collateral`: Intel's and Google's signed collateral for
/// this session's platform and firmware, as the SDK reads it
/// (`x0ne_sdk::attest::Collateral`), fetched by the relay as it fetches its
/// own and reused for [`E2E_COLLATERAL_REUSE_SECS`]. The SDK checks every
/// signature and age; nothing here is trusted by it.
async fn e2e_collateral(State(st): State<Arc<LocalState>>) -> Response {
    let Some(carrier) = &st.session.e2e else {
        return no_sealed_path();
    };
    let now = (st.clock)();
    let mut cached = carrier.collateral.lock().await;
    if let Some((at, v)) = &*cached {
        if now.saturating_sub(*at) < E2E_COLLATERAL_REUSE_SECS {
            return Json(v.clone()).into_response();
        }
    }
    let quote = match crate::attest_v2::unb64(&st.session.document.tdx.quote_b64) {
        Ok(q) => q,
        Err(e) => return error(StatusCode::BAD_GATEWAY, format!("session quote: {e}")),
    };
    match crate::remote_verify::sdk_collateral(&carrier.http, &carrier.sources, &quote).await {
        Ok(c) => {
            let v = serde_json::to_value(c).unwrap_or(Value::Null);
            *cached = Some((now, v.clone()));
            Json(v).into_response()
        }
        Err(e) => error(StatusCode::BAD_GATEWAY, format!("collateral: {e}")),
    }
}

async fn attestation(State(st): State<Arc<LocalState>>) -> Json<Value> {
    Json(json!({
        "document": *st.session.document,
        "report": st.session.report,
    }))
}

/// The exit code a receipt refusal ends the session with: 76 for the strict
/// TCB policy's (a live TCB refusal, as at start-up), 65 otherwise.
fn receipt_exit(why: &str) -> i32 {
    if why.starts_with(crate::tcb_disclosure::STRICT_REFUSAL) {
        76
    } else {
        65
    }
}

/// A receipt that does not verify ends the session: the TD answered with
/// something the attested key did not sign, or about another model.
/// `asked`: the decoding the request asked for; `cut`: `finish_reason` is
/// `"length"`.
fn verify_answer(
    st: &LocalState,
    answer: &Value,
    text: &str,
    asked: &crate::decoding::Asked,
    cut: bool,
) -> Result<(), String> {
    let x = &answer["x0ne"];
    let receipt: InferenceReceipt =
        serde_json::from_value(x["receipt"].clone()).map_err(|e| format!("receipt: {e}"))?;
    let payload = crate::attest_v2::unb64(x["signed_payload_b64"].as_str().unwrap_or(""))
        .map_err(|e| format!("signed_payload_b64: {e}"))?;
    if !receipt::verify_signed_payload(&payload, &receipt) {
        return Err("the receipt's signature does not verify over the signed payload".into());
    }
    if !receipt
        .session_public_key_hex
        .eq_ignore_ascii_case(&st.session.report.session_pubkey_hex)
    {
        return Err("the receipt is signed by a key the hardware did not attest".into());
    }
    if !receipt
        .model_digest_hex
        .eq_ignore_ascii_case(&st.session.report.model_digest_hex)
    {
        return Err(format!(
            "the receipt names model {} and the verified session {}",
            receipt.model_digest_hex, st.session.report.model_digest_hex
        ));
    }
    let opening: CommitmentOpening =
        serde_json::from_value(x["opening"].clone()).map_err(|e| format!("opening: {e}"))?;
    if !receipt::verify_commitment(
        CommitmentKind::Output,
        &receipt.output_commitment_hex,
        &opening,
        text,
    ) {
        return Err("the output commitment does not open to the answer text".into());
    }
    // SECURITY: the decoding asked, as the signed bytes say it was done: the
    // sampling and its seed, the schema's digest, the text a document of it.
    // A TD that decoded otherwise is caught here, not marked verified.
    let signed: Value =
        serde_json::from_slice(&payload).map_err(|e| format!("signed payload: {e}"))?;
    asked.check_signed(&signed, text, cut)?;
    asked.check_context(&signed, x)?;
    // Strict TCB policy: the TD serves under `transparent` and names Intel's
    // latest view in each receipt; this relay stops on one it would refuse.
    if let Some(verdict_tcb) = st.session.report.tcb.as_ref() {
        if verdict_tcb.policy == crate::tcb::TcbMode::Strict.as_str() {
            crate::tcb_disclosure::strict_receipt_check(
                verdict_tcb,
                receipt.tcb.as_ref(),
                (st.clock)(),
            )?;
        }
    }
    st.last_receipt_tcb
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone_from(&receipt.tcb);
    Ok(())
}

/// The TD's whole answer as the local client reads it, rebuilt from what was verified as
/// [`Reading`] rebuilds a stream. Until 2026-11-02 the TD's JSON went out as written, marked
/// verified: a field added in the receipt, a second choice, a usage of its own.
fn whole_answer(
    st: &LocalState,
    answer: &Value,
    asked: &crate::decoding::Asked,
) -> Result<Value, String> {
    let [choice] = answer["choices"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default()
    else {
        return Err("the answer does not hold exactly one choice".into());
    };
    if choice["index"] != 0 || choice["message"]["role"] != "assistant" {
        return Err("the answer's choice is not the assistant's, at index 0".into());
    }
    let Some(text) = choice["message"]["content"].as_str() else {
        return Err("the answer's content is not a string".into());
    };
    let finish = match choice["finish_reason"].as_str() {
        Some(reason @ ("stop" | "length")) => reason,
        _ => return Err("a finish_reason that is not stop or length".into()),
    };
    verify_answer(st, answer, text, asked, finish == "length")?;
    let id = answer["id"].as_str().filter(|id| is_chunk_id(id));
    let created = answer["created"].as_u64();
    let model = answer["model"].as_str();
    let (Some(id), Some(created), Some(model)) = (id, created, model) else {
        return Err("an answer without a valid id, created or model".into());
    };
    if answer["object"] != "chat.completion" {
        return Err("an answer whose object is not chat.completion".into());
    }
    // Passed on from the types: a field the TD added inside them does not reach the client.
    let x = &answer["x0ne"];
    let (Ok(receipt), Ok(opening), Some(payload)) = (
        serde_json::from_value::<InferenceReceipt>(x["receipt"].clone()),
        serde_json::from_value::<CommitmentOpening>(x["opening"].clone()),
        x["signed_payload_b64"].as_str(),
    ) else {
        return Err("a receipt that verified and did not parse".into());
    };
    let (p, c) = (
        u64::from(receipt.prompt_tokens),
        u64::from(receipt.completion_tokens),
    );
    let n = |k: &str| answer["usage"][k].as_u64();
    if (
        n("prompt_tokens"),
        n("completion_tokens"),
        n("total_tokens"),
    ) != (Some(p), Some(c), Some(p + c))
    {
        return Err("the answer's usage differs from the receipt's token counts".into());
    }
    let mut x0ne = json!({
        "receipt": receipt,
        "opening": opening,
        "signed_payload_b64": payload,
    });
    // The next turn's handle, which the receipt verified above commits to.
    if let Ok(issued) = serde_json::from_value::<crate::context::Issued>(x["conversation"].clone())
    {
        x0ne["conversation"] = json!(issued);
    }
    Ok(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": finish,
        }],
        "usage": { "prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c },
        "x0ne": x0ne,
    }))
}

/// A whole verified answer as SSE, for a TD that does not stream.
fn sse(answer: &Value, include_usage: bool, created: u64) -> Response {
    let id = format!(
        "chatcmpl-{}",
        answer["x0ne"]["receipt"]["output_commitment_hex"]
            .as_str()
            .map(|s| &s[..s.len().min(24)])
            .unwrap_or("0")
    );
    let model = answer["model"].clone();
    let text = answer["choices"][0]["message"]["content"].clone();
    let finish = answer["choices"][0]["finish_reason"].clone();
    let chunk = |choices: Value, usage: Option<Value>| {
        let mut c = json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": choices, "x0ne": answer["x0ne"].clone(),
        });
        if let Some(u) = usage {
            c["usage"] = u;
        }
        format!("data: {c}\n\n")
    };
    let mut body = String::new();
    body.push_str(&chunk(
        json!([{ "index": 0, "delta": { "role": "assistant", "content": text }, "finish_reason": null }]),
        None,
    ));
    body.push_str(&chunk(
        json!([{ "index": 0, "delta": {}, "finish_reason": finish }]),
        None,
    ));
    if include_usage {
        body.push_str(&chunk(json!([]), Some(answer["usage"].clone())));
    }
    body.push_str("data: [DONE]\n\n");
    (
        StatusCode::OK,
        [
            (axum::http::header::CONTENT_TYPE, "text/event-stream"),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
        ],
        Body::from(body),
    )
        .into_response()
}

/// An exchange that failed short of an answer: a refused connection stops
/// the session; the TD's quote limits on a new connection are a 503, as its
/// chat queue is; anything else loses it.
fn unreachable_td(st: &LocalState, e: TdError) -> Response {
    if let Some(r) = st.session.client.refusal() {
        st.fail(r.code, r.reason.clone());
        return error(
            StatusCode::BAD_GATEWAY,
            format!("{r}; the session is being stopped"),
        );
    }
    // Eight local requests at once against a TD at --max-queue 2 lost the
    // session on a 429 of its quote limits (2026-10-22): busy, not lost.
    if let TdError::Busy(why) = &e {
        let mut r = error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("the attested VM is busy, retry: {why}"),
        );
        r.headers_mut().insert(
            axum::http::header::RETRY_AFTER,
            axum::http::HeaderValue::from_static("1"),
        );
        return r;
    }
    st.mark_lost();
    error(
        StatusCode::BAD_GATEWAY,
        format!("attested VM unreachable: {e}"),
    )
}

/// The TD's own refusal, with its status, its `code` and `param`, and its
/// `Retry-After`: a full queue says when to come back. Read in OpenAI's shape
/// or in the bare `{"error": "<message>"}` of a TD before it.
fn refused_by_td(answer: &TdAnswer) -> Response {
    let refused = ApiError::read(answer.status, &answer.body)
        .unwrap_or_else(|| ApiError::new(answer.status, answer.body.to_string()));
    let mut r = refused.into_response();
    if let Some(secs) = answer.retry_after {
        r.headers_mut()
            .insert(axum::http::header::RETRY_AFTER, secs.into());
    }
    r
}

/// A 5xx without `Retry-After` (a failed generation, a lost or stopped
/// session) says `x-should-retry: false`: the OpenAI SDKs would otherwise
/// send it twice more, and run a failed generation again.
async fn no_blind_retry(mut resp: Response) -> Response {
    if resp.status().is_server_error()
        && !resp.headers().contains_key(axum::http::header::RETRY_AFTER)
    {
        resp.headers_mut().insert(
            axum::http::HeaderName::from_static("x-should-retry"),
            axum::http::HeaderValue::from_static("false"),
        );
    }
    resp
}

async fn chat(
    State(st): State<Arc<LocalState>>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let mut body = match body {
        Ok(Json(body)) => body,
        Err(rejected) => return ApiError::from(rejected).into_response(),
    };
    if st.lost.load(Ordering::SeqCst) {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the attested VM is lost; the session is over",
        );
    }
    if st.is_stopping() {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the session is stopping: no new requests",
        );
    }
    let now = (st.clock)();
    if now + EXPIRY_GUARD_SECS >= st.session.expires_at_unix {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "the session expires at {}: no new requests",
                st.session.expires_at_unix
            ),
        );
    }
    // The server's own rules, applied here: an invalid request never leaves.
    // Tool calling is one of them (`server::TOOL_FIELDS`), stream's types too.
    let (stream, include_usage, asked) = match crate::server::parse_chat_request(body.clone()) {
        Ok(req) => {
            // A request names the session's model or none; the TD answers under its name.
            let served = &st.session.report.model_name;
            if let Err(refused) = crate::openai::models::served(&req.model, served) {
                return refused.into_response();
            }
            (req.stream, req.include_usage, req.asked())
        }
        Err((status, message)) => return error(status, message),
    };
    if let Some(obj) = body.as_object_mut() {
        obj.remove("stream");
        obj.remove("stream_options");
    }
    if stream && st.session.streaming {
        return chat_streamed(st, body, include_usage, asked).await;
    }
    let td_answer = match st.session.client.chat(&body).await {
        Ok(a) if a.status.is_success() => a.body,
        Ok(a) => return refused_by_td(&a),
        Err(e) => return unreachable_td(&st, e),
    };
    let mut answer = match whole_answer(&st, &td_answer, &asked) {
        Ok(answer) => answer,
        Err(why) => {
            st.fail(
                receipt_exit(&why),
                format!("receipt verification failed: {why}"),
            );
            return error_coded(
                StatusCode::BAD_GATEWAY,
                format!("receipt verification failed: {why}; the session is being stopped"),
                "receipt_verification_failed",
            );
        }
    };
    // Stopped while the TD answered: nothing is marked verified after the stop.
    if st.is_stopping() {
        return error_coded(
            StatusCode::SERVICE_UNAVAILABLE,
            "the session is stopping: this answer is not returned",
            "session_stopped",
        );
    }
    answer["x0ne"]["receipt_verified"] = json!(true);
    answer["x0ne"]["session"] = json!(st.session.report.session_pubkey_hex);
    if stream {
        answer["x0ne"]["stream_emulated"] = json!(true);
        sse(&answer, include_usage, now)
    } else {
        Json(answer).into_response()
    }
}

/// `stream: true` to a TD that streams: the 200 once the TD's is in, then
/// the TD's chunks as they are checked, by [`relay_stream`].
async fn chat_streamed(
    st: Arc<LocalState>,
    mut body: Value,
    include_usage: bool,
    asked: crate::decoding::Asked,
) -> Response {
    body["stream"] = json!(true);
    // Always asked of the TD: its usage is checked against the receipt, then
    // dropped when the client did not ask for it.
    body["stream_options"] = json!({ "include_usage": true });
    let started = Instant::now();
    let source = match st
        .session
        .client
        .chat_stream(&body, started + CHAT_TIMEOUT)
        .await
    {
        Ok(ChatStream::Open {
            event_stream: true,
            body: source,
        }) => source,
        Ok(ChatStream::Open { .. }) => {
            st.fail(
                65,
                "stream protocol: a 200 that is not text/event-stream".into(),
            );
            return error_coded(
                StatusCode::BAD_GATEWAY,
                "the attested VM answered a stream with something else; the session is being stopped",
                "stream_protocol",
            );
        }
        Ok(ChatStream::Status(answer)) => return refused_by_td(&answer),
        Err(e) => return unreachable_td(&st, e),
    };
    let (tx, rx) = mpsc::channel(CLIENT_QUEUE);
    tokio::spawn(relay_stream(
        st,
        source,
        tx,
        include_usage,
        asked,
        Arc::new(TokioClock),
        started,
    ));
    (
        StatusCode::OK,
        [
            (axum::http::header::CONTENT_TYPE, "text/event-stream"),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
        ],
        Body::new(RelayBody {
            rx,
            flow: Flow::Open,
        }),
    )
        .into_response()
}

// ── Streamed answers ─────────────────────────────────────────────────────────

/// Time for the deadlines of a stream. Injected: tests move it by hand.
trait StreamClock: Send + Sync + 'static {
    fn now(&self) -> Instant;
    fn sleep_until(&self, at: Instant) -> Pin<Box<dyn Future<Output = ()> + Send>>;
}

struct TokioClock;

impl StreamClock for TokioClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
    fn sleep_until(&self, at: Instant) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(tokio::time::sleep_until(at))
    }
}

/// What the local client's body is handed. Closed before `Last`, `Failed`
/// or `End`, it breaks.
enum Out {
    Bytes(Bytes),
    /// The final chunk, the usage chunk if asked and `[DONE]`, then the end.
    Last(Bytes),
    /// The error event that ends the stream, then the end: the event says it failed.
    Failed(Bytes),
    /// The clean end of a sealed stream, whose every byte was a `Bytes`.
    End,
}

#[derive(Debug, thiserror::Error)]
#[error("the stream ended in error")]
struct StreamBroken;

enum Flow {
    Open,
    Ending,
    Breaking,
    Done,
}

/// The body of a stream to the local client: no clean end without `Last` or `Failed`.
struct RelayBody {
    rx: mpsc::Receiver<Out>,
    flow: Flow,
}

impl hyper::body::Body for RelayBody {
    type Data = Bytes;
    type Error = StreamBroken;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, StreamBroken>>> {
        let this = self.get_mut();
        match this.flow {
            Flow::Done => return Poll::Ready(None),
            Flow::Ending => {
                this.flow = Flow::Done;
                return Poll::Ready(None);
            }
            Flow::Breaking => {
                this.flow = Flow::Done;
                return Poll::Ready(Some(Err(StreamBroken)));
            }
            Flow::Open => {}
        }
        match std::task::ready!(this.rx.poll_recv(cx)) {
            Some(Out::Bytes(bytes)) => Poll::Ready(Some(Ok(Frame::data(bytes)))),
            Some(Out::Last(bytes) | Out::Failed(bytes)) => {
                this.flow = Flow::Ending;
                Poll::Ready(Some(Ok(Frame::data(bytes))))
            }
            Some(Out::End) => {
                this.flow = Flow::Done;
                Poll::Ready(None)
            }
            None => {
                // Pending once: hyper 1.11.1 flushes a frame when the body is
                // Pending, and drops it unsent when the body errs in the same poll.
                this.flow = Flow::Breaking;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

/// How a stream ended short of its verified final.
enum Exit {
    /// The local client left, or took nothing for [`WRITE_STALL`].
    Gone,
    /// `verify_answer`'s reason: fixed sentences, digests, a serde error on the receipt.
    Receipt(String),
    Protocol(&'static str),
    Cut,
    Generation,
    Stopped,
}

const CUT: &str =
    "the stream from the attested VM ended before its receipt: the text above is not verified";
const GENERATION: &str =
    "the attested VM reported a failed generation: the text above is not verified";
const STOPPED: &str = "the session is stopping: the text above is not verified";

impl Exit {
    /// The error event the client reads: a fixed message, nothing from the TD.
    fn event(&self) -> Option<Bytes> {
        let (code, message) = match self {
            Exit::Gone => return None,
            Exit::Receipt(why) => (
                "receipt_verification_failed",
                format!(
                    "receipt verification failed: {why}; the text streamed above is NOT \
                     verified and must be discarded; the session is being stopped"
                ),
            ),
            Exit::Protocol(why) => (
                "stream_protocol",
                format!(
                    "the attested VM broke the stream protocol ({why}): the text above is \
                     not verified; the session is being stopped"
                ),
            ),
            Exit::Cut => ("stream_cut", CUT.to_string()),
            Exit::Generation => ("generation_failed", GENERATION.to_string()),
            Exit::Stopped => ("session_stopped", STOPPED.to_string()),
        };
        Some(Bytes::from(crate::sse::error_event(code, &message)))
    }
}

fn protocol<T>(why: &'static str) -> Result<T, Exit> {
    Err(Exit::Protocol(why))
}

type Map = serde_json::Map<String, Value>;

fn only(map: &Map, allowed: &[&str]) -> Result<(), Exit> {
    if map.keys().all(|k| allowed.contains(&k.as_str())) {
        Ok(())
    } else {
        protocol("a field outside the whitelist")
    }
}

fn is_chunk_id(id: &str) -> bool {
    id.strip_prefix("chatcmpl-").is_some_and(|h| {
        h.len() == 24
            && h.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

const NON_FINAL: &[&str] = &["id", "object", "created", "model", "choices", "obfuscation"];
const FINAL: &[&str] = &[
    "id",
    "object",
    "created",
    "model",
    "choices",
    "obfuscation",
    "x0ne",
];
const USAGE: &[&str] = &[
    "id",
    "object",
    "created",
    "model",
    "choices",
    "obfuscation",
    "usage",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Role,
    Content,
    Usage,
    Done,
    Over,
}

/// What every chunk of one stream repeats, as its first chunk gave it.
#[derive(PartialEq, Eq)]
struct Shape {
    id: String,
    created: u64,
    model: String,
}

/// The check of one TD stream, chunk by chunk: the whitelist, the order, the
/// receipt. Every chunk the client reads is built here, never copied.
struct Reading {
    include_usage: bool,
    phase: Phase,
    shape: Option<Shape>,
    /// The client's chunks, of the shape the first chunk gave.
    chunks: Option<Chunks>,
    text: String,
    receipt: Option<InferenceReceipt>,
    /// Final, usage and `[DONE]` for the client, written once all is checked.
    last: String,
    /// The TD's error event was the last byte read: its body may end there.
    error_last: bool,
    /// The decoding the request asked for, checked against the receipt.
    asked: crate::decoding::Asked,
}

impl Reading {
    fn new(include_usage: bool, asked: crate::decoding::Asked) -> Self {
        Self {
            asked,
            include_usage,
            phase: Phase::Role,
            shape: None,
            chunks: None,
            text: String::new(),
            receipt: None,
            last: String::new(),
            error_last: false,
        }
    }

    /// One event of the TD: the chunk to pass on now, if any.
    fn take(&mut self, st: &LocalState, data: Data) -> Result<Option<Bytes>, Exit> {
        let chunk = match data {
            Data::Done => return self.done(),
            Data::Json(chunk) => chunk,
        };
        let Some(map) = chunk.as_object() else {
            return protocol("an event that is not an object");
        };
        if map.contains_key("error") {
            let lone = map.len() == 1 && map.get("error").is_some_and(Value::is_object);
            return match self.phase {
                Phase::Role | Phase::Content if lone => Err(Exit::Generation),
                _ => protocol("an error event out of place"),
            };
        }
        match self.phase {
            Phase::Role | Phase::Content | Phase::Usage => {}
            Phase::Done => return protocol("an event between the usage chunk and [DONE]"),
            Phase::Over => return protocol("an event after [DONE]"),
        }
        self.head(map)?;
        if self.phase == Phase::Usage {
            return self.usage(map).map(|()| None);
        }
        self.choice(st, &chunk)
    }

    fn head(&mut self, map: &Map) -> Result<(), Exit> {
        if map.get("object").and_then(Value::as_str) != Some("chat.completion.chunk") {
            return protocol("a chunk whose object is not chat.completion.chunk");
        }
        let id = map
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| is_chunk_id(id));
        let created = map.get("created").and_then(Value::as_u64);
        let model = map.get("model").and_then(Value::as_str);
        let (Some(id), Some(created), Some(model)) = (id, created, model) else {
            return protocol("a chunk without a valid id, created or model");
        };
        let shape = Shape {
            id: id.to_string(),
            created,
            model: model.to_string(),
        };
        match &self.shape {
            None => {
                self.chunks = Some(Chunks::new(
                    &shape.id,
                    shape.created,
                    &shape.model,
                    self.include_usage,
                ));
                self.shape = Some(shape);
            }
            Some(first) if *first == shape => {}
            Some(_) => return protocol("chunks of one stream that differ in id, created or model"),
        }
        if map.get("obfuscation").is_some_and(|o| !o.is_string()) {
            return protocol("an obfuscation field that is not a string");
        }
        Ok(())
    }

    fn choice(&mut self, st: &LocalState, chunk: &Value) -> Result<Option<Bytes>, Exit> {
        let Some(map) = chunk.as_object() else {
            return protocol("an event that is not an object");
        };
        let Some(Value::Array(choices)) = map.get("choices") else {
            return protocol("choices that is not an array");
        };
        let [Value::Object(choice)] = choices.as_slice() else {
            return protocol("choices that is not exactly one choice");
        };
        only(choice, &["index", "delta", "finish_reason"])?;
        if choice.get("index").and_then(Value::as_u64) != Some(0) {
            return protocol("a choice whose index is not 0");
        }
        let Some(Value::Object(delta)) = choice.get("delta") else {
            return protocol("a delta that is not an object");
        };
        let finish = match choice.get("finish_reason") {
            Some(Value::Null) => None,
            Some(Value::String(s)) if s == "stop" => Some("stop"),
            Some(Value::String(s)) if s == "length" => Some("length"),
            _ => return protocol("a finish_reason that is not null, stop or length"),
        };
        match (self.phase, finish) {
            (Phase::Role, None) => {
                only(map, NON_FINAL)?;
                self.role(delta).map(Some)
            }
            (Phase::Content, None) => {
                only(map, NON_FINAL)?;
                self.delta(delta).map(Some)
            }
            (Phase::Content, Some(reason)) => {
                only(map, FINAL)?;
                self.fin(st, chunk, delta, reason).map(|()| None)
            }
            _ => protocol("a finish_reason before the role"),
        }
    }

    fn role(&mut self, delta: &Map) -> Result<Bytes, Exit> {
        only(delta, &["role", "content"])?;
        if delta.get("role").and_then(Value::as_str) != Some("assistant") {
            return protocol("a first chunk without the assistant role");
        }
        let content = match delta.get("content") {
            None => "",
            Some(Value::String(s)) => s.as_str(),
            Some(_) => return protocol("content that is not a string"),
        };
        self.text.push_str(content);
        self.phase = Phase::Content;
        Ok(Bytes::from(self.chunks()?.role(content, None)))
    }

    fn delta(&mut self, delta: &Map) -> Result<Bytes, Exit> {
        only(delta, &["content"])?;
        let Some(Value::String(text)) = delta.get("content") else {
            return protocol("a delta without its text");
        };
        self.text.push_str(text);
        Ok(Bytes::from(self.chunks()?.content(text)))
    }

    /// The final chunk: its receipt verified over the text read so far, held.
    fn fin(
        &mut self,
        st: &LocalState,
        chunk: &Value,
        delta: &Map,
        reason: &'static str,
    ) -> Result<(), Exit> {
        if !delta.is_empty() {
            return protocol("a final chunk with a delta");
        }
        let Some(Value::Object(x0ne)) = chunk.get("x0ne") else {
            return protocol("a final chunk without its x0ne block");
        };
        only(
            x0ne,
            &["receipt", "opening", "signed_payload_b64", "conversation"],
        )?;
        verify_answer(st, chunk, &self.text, &self.asked, reason == "length")
            .map_err(Exit::Receipt)?;
        // Passed on from the types: a field the TD added inside them does not reach the client.
        let field = |k: &str| x0ne.get(k).cloned().unwrap_or(Value::Null);
        let (Ok(receipt), Ok(opening)) = (
            serde_json::from_value::<InferenceReceipt>(field("receipt")),
            serde_json::from_value::<CommitmentOpening>(field("opening")),
        ) else {
            return protocol("a receipt that verified and did not parse");
        };
        let mut x0ne = json!({
            "receipt": receipt,
            "opening": opening,
            "signed_payload_b64": field("signed_payload_b64"),
            "receipt_verified": true,
            "session": st.session.report.session_pubkey_hex,
        });
        // The next turn's handle, which the receipt verified above commits to.
        if let Ok(issued) = serde_json::from_value::<crate::context::Issued>(field("conversation"))
        {
            x0ne["conversation"] = json!(issued);
        }
        let final_chunk = self.chunks()?.finish(reason, x0ne);
        self.last.push_str(&final_chunk);
        self.receipt = Some(receipt);
        self.phase = Phase::Usage;
        Ok(())
    }

    fn usage(&mut self, map: &Map) -> Result<(), Exit> {
        only(map, USAGE)?;
        if map
            .get("choices")
            .and_then(Value::as_array)
            .is_none_or(|c| !c.is_empty())
        {
            return protocol("a chunk after the final that is not the usage chunk");
        }
        let Some(Value::Object(usage)) = map.get("usage") else {
            return protocol("a usage chunk without its usage");
        };
        only(
            usage,
            &["prompt_tokens", "completion_tokens", "total_tokens"],
        )?;
        let Some(receipt) = &self.receipt else {
            return protocol("a usage chunk before the receipt");
        };
        let (p, c) = (
            u64::from(receipt.prompt_tokens),
            u64::from(receipt.completion_tokens),
        );
        let n = |k: &str| usage.get(k).and_then(Value::as_u64);
        if (
            n("prompt_tokens"),
            n("completion_tokens"),
            n("total_tokens"),
        ) != (Some(p), Some(c), Some(p + c))
        {
            return protocol("a usage that differs from the receipt");
        }
        if self.include_usage {
            let usage_chunk = self.chunks()?.usage(p, c);
            self.last.push_str(&usage_chunk);
        }
        self.phase = Phase::Done;
        Ok(())
    }

    fn done(&mut self) -> Result<Option<Bytes>, Exit> {
        match self.phase {
            Phase::Done => {
                self.phase = Phase::Over;
                self.last.push_str(crate::sse::DONE);
                Ok(None)
            }
            // No receipt before [DONE]: a stream cut short, the session is lost.
            Phase::Role | Phase::Content => Err(Exit::Cut),
            Phase::Usage => protocol("[DONE] before the usage chunk"),
            Phase::Over => protocol("an event after [DONE]"),
        }
    }

    /// The TD's body ended clean: the held end, if the stream is whole.
    fn end(&mut self, sse: &SseReader) -> Result<Bytes, Exit> {
        if self.phase != Phase::Over || !sse.at_boundary() {
            return Err(Exit::Cut);
        }
        Ok(Bytes::from(std::mem::take(&mut self.last)))
    }

    /// The writer of this stream's chunks: set by [`Reading::head`], which
    /// every chunk passes before it is written.
    fn chunks(&self) -> Result<&Chunks, Exit> {
        match &self.chunks {
            Some(chunks) => Ok(chunks),
            None => protocol("a chunk before its head"),
        }
    }
}

/// Resolves once the session stops. The watch guard is dropped here, not held.
async fn stopped(stop: &mut watch::Receiver<bool>) {
    let _ = stop.wait_for(|s| *s).await;
}

/// One chunk to the client, unless the session stops, the client stalls or
/// the stream's `deadline` passes: past it, a cut like the TD's silence.
async fn forward(
    out: &mpsc::Sender<Out>,
    chunk: Bytes,
    clock: &dyn StreamClock,
    stop: &mut watch::Receiver<bool>,
    deadline: Instant,
) -> Result<(), Exit> {
    let stall = clock.now() + WRITE_STALL;
    let wait = clock.sleep_until(stall.min(deadline));
    tokio::select! {
        biased;
        () = stopped(stop) => Err(Exit::Stopped),
        sent = out.send(Out::Bytes(chunk)) => sent.map_err(|_| Exit::Gone),
        () = wait => Err(if stall < deadline { Exit::Gone } else { Exit::Cut }),
    }
}

/// The last item of a stream, given up after [`WRITE_STALL`] or at `deadline`.
async fn deliver(out: &mpsc::Sender<Out>, item: Out, clock: &dyn StreamClock, deadline: Instant) {
    let wait = clock.sleep_until((clock.now() + WRITE_STALL).min(deadline));
    tokio::select! {
        biased;
        _ = out.send(item) => {}
        () = wait => {}
    }
}

/// The client left: after [`FROZEN`] without a byte from the TD, the stream
/// it left was frozen by the host, and the session is lost with it.
fn left(now: Instant, heard: Instant) -> Exit {
    if now.saturating_duration_since(heard) > FROZEN {
        Exit::Cut
    } else {
        Exit::Gone
    }
}

/// Reads the TD's body, passing each chunk on once checked. `Ok`: final,
/// usage and `[DONE]` for the client, after the TD's body ended clean.
async fn read_stream<S: FrameSource>(
    st: &LocalState,
    source: &mut S,
    out: &mpsc::Sender<Out>,
    reading: &mut Reading,
    clock: &dyn StreamClock,
    deadline: Instant,
) -> Result<Bytes, Exit> {
    let mut stop = st.stopping.subscribe();
    let mut sse = SseReader::new();
    let mut events = Vec::new();
    let mut heard = clock.now();
    let mut content = heard;
    loop {
        let silence = (heard + FRAME_SILENCE)
            .min(content + CONTENT_SILENCE)
            .min(deadline);
        let read = tokio::select! {
            biased;
            () = stopped(&mut stop) => return Err(Exit::Stopped),
            () = out.closed() => return Err(left(clock.now(), heard)),
            () = clock.sleep_until(silence) => return Err(Exit::Cut),
            read = source.next_bytes() => read,
        };
        let Some(bytes) = read.map_err(|_| Exit::Cut)? else {
            return reading.end(&sse);
        };
        sse.push(&bytes, &mut events)
            .map_err(|e| Exit::Protocol(e.0))?;
        let any = !events.is_empty();
        let mut batch = events.drain(..);
        while let Some(data) = batch.next() {
            match reading.take(st, data) {
                Ok(Some(chunk)) => forward(out, chunk, clock, &mut stop, deadline).await?,
                Ok(None) => {}
                Err(Exit::Generation) => {
                    reading.error_last = batch.next().is_none() && !sse.read_past_event();
                    return Err(Exit::Generation);
                }
                Err(exit) => return Err(exit),
            }
        }
        // Time spent waiting on the local client is not the TD's silence.
        heard = clock.now();
        if any {
            content = heard;
        }
        if reading.phase == Phase::Over && sse.read_past_event() {
            return protocol("bytes after [DONE]");
        }
    }
}

/// One streamed answer, from the TD's body to the local client's. A failed
/// receipt or protocol stops the session after its error event is queued; a
/// cut loses it. The TD's connection is kept after a whole stream, and after
/// its error event when its body ends there, before the client's last item
/// is queued; it is closed on every other end. Nothing is waited for past
/// `started` + [`CHAT_TIMEOUT`].
async fn relay_stream<S: FrameSource>(
    st: Arc<LocalState>,
    mut source: S,
    out: mpsc::Sender<Out>,
    include_usage: bool,
    asked: crate::decoding::Asked,
    clock: Arc<dyn StreamClock>,
    started: Instant,
) {
    let deadline = started + CHAT_TIMEOUT;
    let mut reading = Reading::new(include_usage, asked);
    let read = read_stream(&st, &mut source, &out, &mut reading, &*clock, deadline).await;
    let exit = match read {
        Ok(last) if !st.is_stopping() => {
            source.keep();
            // A final still waiting for its client when the session stops is not sent.
            let mut stop = st.stopping.subscribe();
            tokio::select! {
                biased;
                () = stopped(&mut stop) => {}
                () = deliver(&out, Out::Last(last), &*clock, deadline) => return,
            }
            Exit::Stopped
        }
        Ok(_) => {
            drop(source);
            Exit::Stopped
        }
        Err(Exit::Generation) if reading.error_last => {
            // As on success: the connection is back in the pool before the
            // client reads the end, so its next request can take it.
            keep_if_ended(&st, source, &*clock, deadline).await;
            Exit::Generation
        }
        Err(exit) => {
            drop(source);
            exit
        }
    };
    match &exit {
        Exit::Receipt(_) | Exit::Protocol(_) => st.stop_streams(),
        Exit::Cut => st.mark_lost(),
        Exit::Gone | Exit::Generation | Exit::Stopped => {}
    }
    if let Some(event) = exit.event() {
        deliver(&out, Out::Failed(event), &*clock, deadline).await;
    }
    match exit {
        Exit::Receipt(why) => st.fail(
            receipt_exit(&why),
            format!("receipt verification failed: {why}"),
        ),
        Exit::Protocol(why) => st.fail(65, format!("stream protocol: {why}")),
        _ => {}
    }
}

/// One sealed stream, from the TD's body to the local client's, frame by
/// frame and opened by neither. The relay holds it to the bytes the
/// session's cadence allows (`sealed_stream::max_wire_bytes`), to
/// [`FRAME_SILENCE`] between frames and to [`CHAT_TIMEOUT`] in all. A body
/// ended clean is passed on whole and its connection kept; every other end
/// breaks the client's body, so the SDK refuses the stream, and is counted
/// by its code in `/x0ne/v1/status`. Over the bound the session stops
/// (65); a TD silent, broken or still writing at the deadline loses it.
async fn relay_sealed_stream<S: FrameSource>(
    st: Arc<LocalState>,
    mut source: S,
    out: mpsc::Sender<Out>,
    clock: Arc<dyn StreamClock>,
    started: Instant,
) {
    let deadline = started + CHAT_TIMEOUT;
    let cadence = st
        .session
        .e2e
        .as_ref()
        .map_or(E2eStream::DEFAULT, |c| c.cadence);
    let bound = x0ne_sdk::sealed_stream::max_wire_bytes(&cadence);
    match read_sealed(&st, &mut source, &out, &*clock, deadline, bound).await {
        Ok(()) => {
            source.keep();
            st.sealed.carried();
            deliver(&out, Out::End, &*clock, deadline).await;
        }
        Err(exit) => {
            drop(source);
            st.sealed.cut(exit.code(), (st.clock)());
            match exit {
                SealedExit::TooLarge(read) => st.fail(
                    65,
                    format!(
                        "sealed stream: {read} bytes from the attested VM, over the {bound} \
                         its cadence allows ({SEALED_STREAM_TOO_LARGE})"
                    ),
                ),
                SealedExit::Silent | SealedExit::TooLong | SealedExit::TdBroke => st.mark_lost(),
                SealedExit::ClientGone | SealedExit::Stopped => {}
            }
        }
    }
}

/// Reads the TD's sealed body, passing each frame on as it came, up to
/// `bound` bytes. `Ok`: the body ended clean.
async fn read_sealed<S: FrameSource>(
    st: &LocalState,
    source: &mut S,
    out: &mpsc::Sender<Out>,
    clock: &dyn StreamClock,
    deadline: Instant,
    bound: usize,
) -> Result<(), SealedExit> {
    let mut stop = st.stopping.subscribe();
    let mut heard = clock.now();
    let mut read = 0usize;
    loop {
        let silence = (heard + FRAME_SILENCE).min(deadline);
        let next = tokio::select! {
            biased;
            () = stopped(&mut stop) => return Err(SealedExit::Stopped),
            () = out.closed() => {
                // Frozen by the host when the client left: lost, as a plain stream.
                return Err(if clock.now().saturating_duration_since(heard) > FROZEN {
                    SealedExit::Silent
                } else {
                    SealedExit::ClientGone
                });
            }
            () = clock.sleep_until(silence) => {
                return Err(if silence >= deadline {
                    SealedExit::TooLong
                } else {
                    SealedExit::Silent
                });
            }
            next = source.next_bytes() => next,
        };
        let Some(bytes) = next.map_err(|_| SealedExit::TdBroke)? else {
            return Ok(());
        };
        read = read.saturating_add(bytes.len());
        if read > bound {
            return Err(SealedExit::TooLarge(read));
        }
        forward(out, bytes, clock, &mut stop, deadline)
            .await
            .map_err(|e| match e {
                Exit::Stopped => SealedExit::Stopped,
                Exit::Cut => SealedExit::TooLong,
                _ => SealedExit::ClientGone,
            })?;
        heard = clock.now();
    }
}

/// The TD's body after its error event: kept for the next request if it
/// ends within [`ERROR_END`] and by `deadline`, closed if anything else
/// comes first.
async fn keep_if_ended<S: FrameSource>(
    st: &LocalState,
    mut source: S,
    clock: &dyn StreamClock,
    deadline: Instant,
) {
    let mut stop = st.stopping.subscribe();
    let ended = tokio::select! {
        biased;
        () = stopped(&mut stop) => false,
        () = clock.sleep_until((clock.now() + ERROR_END).min(deadline)) => false,
        read = source.next_bytes() => matches!(read, Ok(None)),
    };
    if ended {
        source.keep();
    }
}

pub fn router(state: Arc<LocalState>) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/x0ne/v1/status", get(status))
        .route("/x0ne/v1/attestation", get(attestation))
        .route("/v2/e2e/key", get(e2e_key))
        .route(
            "/v2/e2e/chat",
            post(e2e_chat).layer(axum::extract::DefaultBodyLimit::max(
                x0ne_sdk::e2e::MAX_SEALED_STREAM_REQUEST,
            )),
        )
        .route("/v2/e2e/collateral", get(e2e_collateral))
        .layer(axum::middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
        .layer(axum::middleware::from_fn(crate::server::isolate_panics))
        .layer(axum::middleware::map_response(no_blind_retry))
}

pub struct LocalServer {
    pub listen: std::net::SocketAddr,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
    state: Option<Arc<LocalState>>,
}

impl LocalServer {
    /// Streams told to stop, then the server; connections still open after
    /// [`STOP_WAIT`] (a client that does not read) are no longer waited for.
    pub async fn stop(mut self) {
        if let Some(st) = &self.state {
            st.stop_streams();
        }
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if tokio::time::timeout(STOP_WAIT, &mut self.task)
            .await
            .is_err()
        {
            self.task.abort();
        }
    }
}

/// Bind `listen` (loopback only) and serve until stopped.
pub async fn start(state: Arc<LocalState>, listen: &str) -> Result<LocalServer, String> {
    let addr = parse_listen(listen)?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("bind {addr}: {e}"))?;
    let listen = listener.local_addr().map_err(|e| e.to_string())?;
    state.port.store(listen.port(), Ordering::SeqCst);
    let app = router(state.clone());
    Ok(serve(listener, listen, app, Some(state)))
}

fn serve(
    listener: tokio::net::TcpListener,
    listen: std::net::SocketAddr,
    app: Router,
    state: Option<Arc<LocalState>>,
) -> LocalServer {
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });
    LocalServer {
        listen,
        shutdown: Some(tx),
        task,
        state,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_loopback_binds() {
        assert!(parse_listen("127.0.0.1:8478").is_ok());
        assert!(parse_listen("[::1]:8478").is_ok());
        assert!(parse_listen("0.0.0.0:8478")
            .unwrap_err()
            .contains("loopback"));
        assert!(parse_listen("203.0.113.2:8478")
            .unwrap_err()
            .contains("loopback"));
        assert!(
            parse_listen("localhost:8478").is_err(),
            "a name is not an address"
        );
    }

    #[cfg(unix)]
    #[test]
    fn the_key_file_is_0600_in_a_0700_directory() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let p = write_key_file(dir.path(), &[0xab; 32]).unwrap();
        assert_eq!(
            std::fs::metadata(&p).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            std::fs::metadata(p.parent().unwrap())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            format!("{}\n", "ab".repeat(32))
        );
        // Written again: replaced, same mode.
        let p2 = write_key_file(dir.path(), &[0xcd; 32]).unwrap();
        assert_eq!(p, p2);
        assert_eq!(
            std::fs::read_to_string(&p2).unwrap(),
            format!("{}\n", "cd".repeat(32))
        );
    }

    /// SECURITY: the key's directories are the caller's own, or nothing is
    /// written: a directory that lets others in, a symbolic link (to a
    /// directory the caller owns, even), a runtime directory that is a link,
    /// and one owned by another account are refused before the key exists. A
    /// failure: the bearer written where another account can read or swap it
    /// (`/tmp/x0ne-engine-<uid>` made ahead of the relay).
    #[cfg(unix)]
    #[test]
    fn a_key_directory_that_is_not_the_callers_own_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let refused = |runtime: &Path, says: &str| {
            let e = write_key_file(runtime, &[0xab; 32]).expect_err("a key written");
            assert!(e.contains(says), "{e}");
            assert!(!runtime.join(SUBDIR).join(KEY_FILE_NAME).exists(), "{e}");
        };
        // A symbolic link to a 0700 directory of the caller's.
        let base = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(elsewhere.path(), base.path().join(SUBDIR)).unwrap();
        refused(base.path(), "symbolic link");
        assert!(!elsewhere.path().join(KEY_FILE_NAME).exists());
        // Made ahead with a mode that lets others in.
        let base = tempfile::tempdir().unwrap();
        std::fs::create_dir(base.path().join(SUBDIR)).unwrap();
        std::fs::set_permissions(
            base.path().join(SUBDIR),
            std::fs::Permissions::from_mode(0o777),
        )
        .unwrap();
        refused(base.path(), "mode 777");
        // The runtime directory itself a link (the fallback under /tmp).
        let base = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(elsewhere.path(), base.path().join("run")).unwrap();
        refused(&base.path().join("run"), "symbolic link");
        // Writable by others: another account could swap what is inside.
        let base = tempfile::tempdir().unwrap();
        std::fs::set_permissions(base.path(), std::fs::Permissions::from_mode(0o777)).unwrap();
        refused(base.path(), "mode 777");
        // Another account's: the same checks under a uid that is not the owner's.
        let base = tempfile::tempdir().unwrap();
        std::fs::set_permissions(base.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let meta = std::fs::symlink_metadata(base.path()).unwrap();
        let mine = std::os::unix::fs::MetadataExt::uid(&meta);
        let e = foreign_dir(base.path(), &meta, mine.wrapping_add(1), true)
            .expect("another account's directory accepted");
        assert!(e.contains("owned by uid"), "{e}");
        assert!(foreign_dir(base.path(), &meta, mine, true).is_none());
    }

    #[test]
    fn constant_time_eq_compares_lengths_and_bytes() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
    }

    mod streams {
        use super::super::*;
        use crate::binding::SessionIdentity;
        use crate::remote_verify::BodyBroke;
        use crate::stream::tests::ByHand;
        use crate::stream::{Event, Finale, SseBody};
        use std::collections::VecDeque;
        use std::sync::atomic::AtomicUsize;
        use std::sync::atomic::Ordering::SeqCst;
        use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
        use tokio::sync::Notify;

        // ── doubles ──────────────────────────────────────────────────────

        /// Moves only when told.
        struct FakeClock {
            base: Instant,
            offset: watch::Sender<Duration>,
        }

        impl FakeClock {
            fn new() -> Arc<Self> {
                Arc::new(Self {
                    base: Instant::now(),
                    offset: watch::channel(Duration::ZERO).0,
                })
            }
            fn advance(&self, by: Duration) {
                self.offset.send_modify(|o| *o += by);
            }
        }

        impl StreamClock for FakeClock {
            fn now(&self) -> Instant {
                self.base + *self.offset.borrow()
            }
            fn sleep_until(&self, at: Instant) -> Pin<Box<dyn Future<Output = ()> + Send>> {
                let mut offset = self.offset.subscribe();
                let base = self.base;
                Box::pin(async move {
                    loop {
                        let now = base + *offset.borrow_and_update();
                        if now >= at {
                            return;
                        }
                        if offset.changed().await.is_err() {
                            std::future::pending::<()>().await;
                        }
                    }
                })
            }
        }

        enum Step {
            Bytes(Vec<u8>),
            Advance(Duration),
            Gate(Arc<Notify>),
            Hang,
            Break,
            End,
        }

        #[derive(Default)]
        struct Seen {
            kept: AtomicBool,
            dropped: AtomicBool,
            reads: AtomicUsize,
        }

        impl Seen {
            /// Dropped without keep: the TD's connection was closed.
            fn closed(&self) -> bool {
                self.dropped.load(SeqCst) && !self.kept.load(SeqCst)
            }
        }

        /// A TD body played from a script.
        struct Scripted {
            steps: VecDeque<Step>,
            clock: Arc<FakeClock>,
            seen: Arc<Seen>,
        }

        impl FrameSource for Scripted {
            async fn next_bytes(&mut self) -> Result<Option<Bytes>, BodyBroke> {
                loop {
                    match self.steps.pop_front() {
                        Some(Step::Bytes(b)) => {
                            self.seen.reads.fetch_add(1, SeqCst);
                            return Ok(Some(Bytes::from(b)));
                        }
                        Some(Step::Advance(by)) => self.clock.advance(by),
                        Some(Step::Gate(gate)) => gate.notified().await,
                        Some(Step::Break) => return Err(BodyBroke),
                        Some(Step::End) => return Ok(None),
                        Some(Step::Hang) | None => std::future::pending::<()>().await,
                    }
                }
            }
            fn keep(self) {
                self.seen.kept.store(true, SeqCst);
            }
        }

        impl Drop for Scripted {
            fn drop(&mut self) {
                self.seen.dropped.store(true, SeqCst);
            }
        }

        fn digest() -> String {
            "ab".repeat(32)
        }

        fn identity() -> SessionIdentity {
            SessionIdentity::generate([21u8; 32], [22u8; 32])
        }

        type Fatal = watch::Receiver<Option<(i32, String)>>;

        /// A verified session over a TD at an address nothing listens on.
        fn state(id: &SessionIdentity) -> (Arc<LocalState>, Fatal) {
            state_with(id, None)
        }

        /// [`state`], the session carrying the sealed path as `e2e` says.
        fn state_with(id: &SessionIdentity, e2e: Option<E2eCarrier>) -> (Arc<LocalState>, Fatal) {
            let report: Report = serde_json::from_value(json!({
                "verdict": {}, "environment": "test",
                "session_pubkey_hex": crate::cloud_config::hex(id.public_key().as_bytes()),
                "model_name": "m", "model_digest_hex": digest(),
                "tcb_platform": "", "tcb_qe": "", "endorsement_signer": "",
                "endorsement_timestamp_unix": 0, "firmware_svn": 0, "mrtd_hex": "",
                "image_name": "", "image_version": "", "rtmr0_hex": "", "rtmr0_known": null,
                "sections": [], "gpu_count": 0, "reference_path": "",
                "reference_sha256_hex": "", "reference_explicit": false,
                "summary": [], "limitations": [],
            }))
            .unwrap();
            let document: Document = serde_json::from_value(json!({
                "schema": "",
                "binding": { "context": "", "session_nonce_hex": "", "session_pubkey_hex": "",
                             "tls_spki_sha256_hex": "", "config_sha256_hex": "" },
                "config_b64": "",
                "tdx": { "quote_b64": "", "ccel_b64": "" },
                "gpu": [],
                "nras": { "overall_token": "", "jwks": {} },
                "model": { "name": "", "digest_hex": "", "files": [] },
                "server": { "version": "", "attested_at_unix": 0, "verdict": {}, "live_tcb": ["", ""] },
            }))
            .unwrap();
            let tls = crate::tls::TlsIdentity::generate("td.test").unwrap();
            let session = Session {
                client: TdClient::new(&tls, "127.0.0.1:9", [0u8; 32]).unwrap(),
                document: Arc::new(document),
                report,
                provider: "test".into(),
                machine_type: "test".into(),
                expires_at_unix: u64::MAX / 2,
                started_at_unix: 0,
                context_tokens: None,
                streaming: true,
                prefill_chunk: None,
                prefill_chunks: None,
                e2e,
            };
            LocalState::new(session, &[1u8; 32], Arc::new(|| 1_000u64))
        }

        fn proof() -> crate::gate::Attested {
            use x0ne_confidential::model::{
                AttestationEvidence, EvidenceFormat, ExecutionEnvironment, GpuCcMode,
                TeeTechnology, Verdict, VerifierStrength, WorkloadIdentity,
            };
            let env = ExecutionEnvironment::ConfidentialVmWithGpu {
                cpu: TeeTechnology::IntelTdx,
                gpu_mode: GpuCcMode::On,
            };
            crate::gate::Gate::attest(
                env.clone(),
                &AttestationEvidence {
                    format: EvidenceFormat::StructuredClaims,
                    payload: vec![],
                    collected_at_unix: 1_000,
                },
                Verdict::Verified {
                    identity: WorkloadIdentity {
                        name: "x0ne-infer".into(),
                        environment: env,
                        measurements: vec![],
                    },
                    verifier: "test".into(),
                    strength: VerifierStrength::Cryptographic,
                },
                1_000,
            )
            .unwrap()
        }

        /// The x0ne block a TD signs over `text`, as the server serializes it.
        fn x0ne_block(id: &SessionIdentity, text: &str, tokens: (u32, u32)) -> String {
            let receipt::Signed {
                receipt,
                opening,
                payload,
            } = receipt::build_signed(
                &proof(),
                id,
                [5u8; 32],
                &digest(),
                "a prompt",
                text,
                tokens,
                "cpu",
                crate::backend::Isolation::NONE_OF_IT,
                1_000,
                1_001,
            )
            .unwrap();
            json!({
                "receipt": receipt,
                "opening": opening,
                "signed_payload_b64": crate::attest_v2::b64(&payload),
            })
            .to_string()
        }

        /// [`x0ne_block`], the receipt signing `claims` about the decoding.
        fn x0ne_block_claiming(
            id: &SessionIdentity,
            text: &str,
            tokens: (u32, u32),
            claims: receipt::DecodingClaims,
        ) -> String {
            let receipt::Signed {
                receipt,
                opening,
                payload,
            } = receipt::build_signed_decoding(
                &proof(),
                id,
                [5u8; 32],
                &digest(),
                "a prompt",
                text,
                tokens,
                "cpu",
                crate::backend::Isolation::NONE_OF_IT,
                1_000,
                1_001,
                None,
                claims,
            )
            .unwrap();
            json!({
                "receipt": receipt,
                "opening": opening,
                "signed_payload_b64": crate::attest_v2::b64(&payload),
            })
            .to_string()
        }

        /// The TD's own stream body, ticked by hand: its frames, byte for byte.
        struct Td {
            body: SseBody<ByHand>,
            events: UnboundedSender<Event>,
            ticks: UnboundedSender<Instant>,
        }

        impl Td {
            fn new() -> Self {
                let (events, rx) = unbounded_channel();
                let (ticks, by_hand) = unbounded_channel();
                let shape = crate::stream::tests::shape("m", true);
                Self {
                    body: SseBody::new(rx, shape, ByHand(by_hand), Instant::now()),
                    events,
                    ticks,
                }
            }
            fn frame(&mut self) -> Step {
                self.ticks.send(Instant::now()).unwrap();
                let mut cx = Context::from_waker(std::task::Waker::noop());
                use hyper::body::Body as _;
                match Pin::new(&mut self.body).poll_frame(&mut cx) {
                    Poll::Ready(Some(Ok(frame))) => {
                        Step::Bytes(frame.into_data().unwrap().to_vec())
                    }
                    _ => panic!("the TD body gave no frame"),
                }
            }
            /// The first frame: the role chunk.
            fn role(&mut self) -> Step {
                self.frame()
            }
            fn delta(&mut self, text: &str) -> Step {
                self.events.send(Event::Delta(text.into())).unwrap();
                self.frame()
            }
            fn padding(&mut self) -> Step {
                self.frame()
            }
            /// Frames until the body ends: the padding comments the frame
            /// count asks for (`crate::padding`), then the last frame, as
            /// the TD writes them, in one step.
            fn through_the_last(&mut self) -> Step {
                let mut bytes = Vec::new();
                let mut cx = Context::from_waker(std::task::Waker::noop());
                use hyper::body::Body as _;
                loop {
                    bytes.extend(bytes_of(self.frame()));
                    // After its last frame the body ends without a tick.
                    if let Poll::Ready(None) = Pin::new(&mut self.body).poll_frame(&mut cx) {
                        return Step::Bytes(bytes);
                    }
                    assert!(bytes.len() < 1 << 24, "the TD body has no last frame");
                }
            }
            fn last(&mut self, x0ne_json: String, tokens: (u32, u32)) -> Step {
                self.events
                    .send(Event::Final(Box::new(Finale {
                        finish_reason: "stop",
                        prompt_tokens: tokens.0,
                        completion_tokens: tokens.1,
                        x0ne_json,
                    })))
                    .unwrap();
                self.through_the_last()
            }
            fn error(&mut self) -> Step {
                self.events
                    .send(Event::Error(crate::stream::GENERATION_FAILED))
                    .unwrap();
                self.through_the_last()
            }
        }

        fn bytes_of(step: Step) -> Vec<u8> {
            match step {
                Step::Bytes(b) => b,
                _ => panic!("not bytes"),
            }
        }

        const ID: &str = "chatcmpl-abababababababababababab";

        /// A chunk as the TD writes it, unpadded.
        fn td_chunk(delta: Value, finish: Value) -> Value {
            json!({
                "id": ID, "object": "chat.completion.chunk", "created": 1_760_000_000u64,
                "model": "m",
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
            })
        }

        fn data(v: &Value) -> Step {
            Step::Bytes(format!("data: {v}\n\n").into_bytes())
        }

        /// One relayed stream on `st`, driven by `steps`.
        struct Run {
            rx: mpsc::Receiver<Out>,
            task: tokio::task::JoinHandle<()>,
            seen: Arc<Seen>,
        }

        fn launch(
            st: &Arc<LocalState>,
            clock: &Arc<FakeClock>,
            steps: Vec<Step>,
            include_usage: bool,
        ) -> Run {
            launch_asking(
                st,
                clock,
                steps,
                include_usage,
                crate::decoding::Asked::default(),
            )
        }

        /// [`launch`] for a request that asked `asked` of the decoding.
        fn launch_asking(
            st: &Arc<LocalState>,
            clock: &Arc<FakeClock>,
            steps: Vec<Step>,
            include_usage: bool,
            asked: crate::decoding::Asked,
        ) -> Run {
            let seen = Arc::new(Seen::default());
            let source = Scripted {
                steps: steps.into(),
                clock: clock.clone(),
                seen: seen.clone(),
            };
            let (tx, rx) = mpsc::channel(CLIENT_QUEUE);
            let task = tokio::spawn(relay_stream(
                st.clone(),
                source,
                tx,
                include_usage,
                asked,
                clock.clone(),
                clock.now(),
            ));
            Run { rx, task, seen }
        }

        /// What the client read: each item's text, whether `Last` came, and
        /// whether the last item was `Failed`.
        struct Read {
            items: Vec<String>,
            last: bool,
            failed: bool,
        }

        impl Read {
            fn text(&self) -> String {
                self.items.concat()
            }
            /// The JSON of every `data:` event, `[DONE]` as a string.
            fn events(&self) -> Vec<Value> {
                self.text()
                    .split("\n\n")
                    .filter_map(|e| e.strip_prefix("data: "))
                    .map(|d| {
                        if d == "[DONE]" {
                            Value::String(d.into())
                        } else {
                            serde_json::from_str(d).unwrap()
                        }
                    })
                    .collect()
            }
            /// The error event that ended the stream: code and message.
            fn error(&self) -> (String, String) {
                assert!(self.failed, "the error event is the last item, `Failed`");
                let events = self.events();
                let e = &events.last().expect("an event")["error"];
                (
                    e["code"].as_str().unwrap_or_default().into(),
                    e["message"].as_str().unwrap_or_default().into(),
                )
            }
        }

        fn item(out: Out) -> (String, bool) {
            match out {
                Out::Bytes(b) | Out::Failed(b) => (String::from_utf8(b.to_vec()).unwrap(), false),
                Out::Last(b) => (String::from_utf8(b.to_vec()).unwrap(), true),
                Out::End => panic!("a plain stream ends with its last item, not bare"),
            }
        }

        async fn read_all(run: &mut Run) -> Read {
            let mut read = Read {
                items: Vec::new(),
                last: false,
                failed: false,
            };
            while let Some(out) = run.rx.recv().await {
                read.failed = matches!(out, Out::Failed(_));
                let (text, last) = item(out);
                read.items.push(text);
                read.last |= last;
            }
            read
        }

        async fn settle() {
            for _ in 0..50 {
                tokio::task::yield_now().await;
            }
        }

        fn chat_body() -> Value {
            json!({ "model": "m", "messages": [{ "role": "user", "content": "hi" }] })
        }

        async fn status_of(st: &Arc<LocalState>) -> (StatusCode, String) {
            let resp = chat(State(st.clone()), Ok(Json(chat_body()))).await;
            let status = resp.status();
            let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
                .await
                .unwrap();
            (status, String::from_utf8_lossy(&body).into_owned())
        }

        /// A whole stream ending in `x0ne_json`, deltas `pieces`, from the TD's body.
        fn td_stream(pieces: &[&str], x0ne_json: String, tokens: (u32, u32)) -> Vec<Step> {
            let mut td = Td::new();
            let mut steps = vec![td.role()];
            for p in pieces {
                steps.push(td.padding());
                steps.push(td.delta(p));
            }
            steps.push(td.last(x0ne_json, tokens));
            steps.push(Step::End);
            steps
        }

        // ── V1 to V10 ────────────────────────────────────────────────────

        /// V1. The TD's own frames: each delta reaches the client before the
        /// next frame is read; final, usage and `[DONE]` come as one item, and
        /// only once the TD's body has ended; every chunk is rebuilt from the
        /// whitelist. A failure is text held back, or a chunk copied as sent.
        #[tokio::test]
        async fn a_valid_stream_forwards_deltas_then_writes_final_usage_done_at_once() {
            let id = identity();
            for include_usage in [true, false] {
                let (st, fatal) = state(&id);
                let clock = FakeClock::new();
                let mut td = Td::new();
                let (g1, g2, g3) = (
                    Arc::new(Notify::new()),
                    Arc::new(Notify::new()),
                    Arc::new(Notify::new()),
                );
                let steps = vec![
                    td.role(),
                    Step::Gate(g1.clone()),
                    td.padding(),
                    td.delta("Bon"),
                    Step::Gate(g2.clone()),
                    td.delta("jour"),
                    td.padding(),
                    td.last(x0ne_block(&id, "Bonjour", (3, 2)), (3, 2)),
                    Step::Gate(g3.clone()),
                    Step::End,
                ];
                let mut run = launch(&st, &clock, steps, include_usage);
                let mut items = Vec::new();
                let (role, last) = item(run.rx.recv().await.unwrap());
                assert!(!last && role.contains("\"role\":\"assistant\""), "{role}");
                settle().await;
                assert!(run.rx.try_recv().is_err(), "nothing before the next frame");
                items.push(role);
                g1.notify_one();
                let (bon, _) = item(run.rx.recv().await.unwrap());
                assert!(bon.contains("\"content\":\"Bon\""), "{bon}");
                items.push(bon);
                g2.notify_one();
                let (jour, _) = item(run.rx.recv().await.unwrap());
                assert!(jour.contains("\"content\":\"jour\""), "{jour}");
                items.push(jour);
                settle().await;
                assert!(
                    run.rx.try_recv().is_err(),
                    "the final is held until the TD's body ends"
                );
                assert!(!run.seen.kept.load(SeqCst));
                g3.notify_one();
                let (end, last) = item(run.rx.recv().await.unwrap());
                assert!(last, "final, usage and [DONE] in one item: {end}");
                assert!(end.ends_with("data: [DONE]\n\n"), "{end}");
                items.push(end);
                assert!(run.rx.recv().await.is_none());
                run.task.await.unwrap();
                assert!(run.seen.kept.load(SeqCst), "the connection is kept");
                assert!(fatal.borrow().is_none() && !st.lost.load(SeqCst));

                let read = Read {
                    items,
                    last: true,
                    failed: false,
                };
                let events = read.events();
                let n = events.len();
                assert_eq!(events[n - 1], "[DONE]");
                let chunks = &events[..n - 1];
                assert_eq!(chunks.len(), 4 + usize::from(include_usage), "{events:?}");
                for c in chunks {
                    let keys: Vec<&str> =
                        c.as_object().unwrap().keys().map(String::as_str).collect();
                    for k in &keys {
                        assert!(
                            ["id", "object", "created", "model", "choices", "usage", "x0ne"]
                                .contains(k),
                            "{c}"
                        );
                    }
                    assert_eq!(c["id"], ID);
                    assert_eq!(c["created"], 1_760_000_000u64);
                    assert_eq!(c["model"], "m");
                    assert_eq!(c["object"], "chat.completion.chunk");
                    assert_eq!(c.get("usage").is_some(), include_usage, "{c}");
                }
                assert_eq!(
                    chunks[0]["choices"][0]["delta"],
                    json!({"role": "assistant", "content": ""})
                );
                let text: String = chunks[1..3]
                    .iter()
                    .map(|c| c["choices"][0]["delta"]["content"].as_str().unwrap())
                    .collect();
                assert_eq!(text, "Bonjour");
                for c in &chunks[..3] {
                    assert!(c.get("x0ne").is_none(), "x0ne on the final only: {c}");
                    assert_eq!(c["choices"][0]["finish_reason"], Value::Null);
                    if include_usage {
                        assert_eq!(c["usage"], Value::Null);
                    }
                }
                let fin = &chunks[3];
                assert_eq!(fin["choices"][0]["finish_reason"], "stop");
                assert_eq!(fin["choices"][0]["delta"], json!({}));
                let x = fin["x0ne"].as_object().unwrap();
                let mut x_keys: Vec<&str> = x.keys().map(String::as_str).collect();
                x_keys.sort_unstable();
                assert_eq!(
                    x_keys,
                    [
                        "opening",
                        "receipt",
                        "receipt_verified",
                        "session",
                        "signed_payload_b64"
                    ]
                );
                assert_eq!(fin["x0ne"]["receipt_verified"], json!(true));
                assert_eq!(fin["x0ne"]["session"], st.session.report.session_pubkey_hex);
                assert!(receipt::verify_commitment(
                    CommitmentKind::Output,
                    fin["x0ne"]["receipt"]["output_commitment_hex"]
                        .as_str()
                        .unwrap(),
                    &serde_json::from_value(fin["x0ne"]["opening"].clone()).unwrap(),
                    "Bonjour",
                ));
                if include_usage {
                    assert_eq!(chunks[4]["choices"], json!([]));
                    assert_eq!(
                        chunks[4]["usage"],
                        json!({"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
                    );
                }
                assert!(!read.text().contains("obfuscation"));
            }
        }

        /// V2. A delta that is not what the TD signed: the error event takes
        /// the final's place, and fail(65) comes only once that event is
        /// queued. The client reads one item at a time behind a full queue:
        /// fatal seen while the event is not yet queued fails this test.
        #[tokio::test]
        async fn a_bad_receipt_replaces_the_final_chunk_and_stops_the_session_after_the_event() {
            let id = identity();
            let (st, mut fatal) = state(&id);
            let clock = FakeClock::new();
            let pieces = ["a", "b", "c", "d", "e", "f", "g"];
            let steps = td_stream(&pieces, x0ne_block(&id, "abcdefX", (3, 7)), (3, 7));
            let mut run = launch(&st, &clock, steps, true);
            let mut items: Vec<(String, bool)> = Vec::new();
            let mut failed = false;
            loop {
                settle().await;
                if fatal.borrow_and_update().is_some() {
                    while let Ok(out) = run.rx.try_recv() {
                        failed = matches!(out, Out::Failed(_));
                        items.push(item(out));
                    }
                    break;
                }
                match run.rx.recv().await {
                    Some(out) => {
                        failed = matches!(out, Out::Failed(_));
                        items.push(item(out));
                    }
                    None => break,
                }
                assert!(items.len() < 50, "no end");
            }
            let event = items
                .iter()
                .position(|(t, _)| t.contains("receipt_verification_failed"));
            assert!(
                event.is_some(),
                "fail(65) before the event was queued: {items:?}"
            );
            while let Some(out) = run.rx.recv().await {
                failed = matches!(out, Out::Failed(_));
                items.push(item(out));
            }
            run.task.await.unwrap();
            let read = Read {
                items: items.iter().map(|(t, _)| t.clone()).collect(),
                last: items.iter().any(|(_, l)| *l),
                failed,
            };
            assert!(!read.last, "the final never goes out");
            assert!(!read.text().contains("receipt_verified"));
            assert!(!read.text().contains("finish_reason\":\"stop"));
            let (code, message) = read.error();
            assert_eq!(code, "receipt_verification_failed");
            assert!(
                message.contains("the output commitment does not open to the answer text")
                    && message.contains("NOT verified and must be discarded"),
                "{message}"
            );
            let (code, reason) = fatal.borrow().clone().unwrap();
            assert_eq!(code, 65);
            assert!(
                reason.starts_with("receipt verification failed: "),
                "{reason}"
            );
            assert!(st.is_stopping());
            assert!(run.seen.closed(), "the TD's connection is closed");
        }

        /// SECURITY: a receipt that signs another decoding than the request's
        /// (digest, draws, text outside the schema) stops the session, 65,
        /// whole or streamed. A failure: such an answer marked verified.
        #[tokio::test]
        async fn a_receipt_that_says_another_decoding_stops_the_session() {
            use crate::decoding::{Asked, SignedSampling};
            use receipt::DecodingClaims;
            let id = identity();
            let schema =
                crate::json_schema::JsonSchema::compile(&json!({"type": "boolean"})).unwrap();
            let digest = schema.sha256_hex().to_string();
            let asked = Asked {
                sampling: None,
                json_schema: Some(schema),
                ..Asked::default()
            };
            let honest = DecodingClaims {
                sampling: None,
                json_schema_sha256_hex: Some(digest.clone()),
                context: None,
            };
            let answer = |text: &str, claims: DecodingClaims| -> Value {
                let block: Value =
                    serde_json::from_str(&x0ne_block_claiming(&id, text, (1, 1), claims)).unwrap();
                json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
                    "x0ne": block})
            };
            let (st, _) = state(&id);
            verify_answer(&st, &answer("true", honest.clone()), "true", &asked, false).unwrap();
            verify_answer(&st, &answer("tr", honest.clone()), "tr", &asked, true).unwrap();
            let draws = SignedSampling {
                temperature: 1.0,
                top_p: 1.0,
                seed: 1,
            };
            for (text, claims, word) in [
                ("true", DecodingClaims::default(), "signs none"),
                (
                    "true",
                    DecodingClaims {
                        sampling: None,
                        json_schema_sha256_hex: Some("00".repeat(32)),
                        context: None,
                    },
                    "another schema",
                ),
                (
                    "true",
                    DecodingClaims {
                        sampling: Some(draws),
                        json_schema_sha256_hex: Some(digest.clone()),
                        context: None,
                    },
                    "greedy",
                ),
                ("tr", honest.clone(), "stops before"),
                ("1", honest.clone(), "not a document"),
            ] {
                let e = verify_answer(&st, &answer(text, claims), text, &asked, false).unwrap_err();
                assert!(e.contains(word), "{word}: {e}");
            }

            // Streamed, honest: released.
            let clock = FakeClock::new();
            let steps = td_stream(
                &["tr", "ue"],
                x0ne_block_claiming(&id, "true", (3, 2), honest),
                (3, 2),
            );
            let mut run = launch_asking(&st, &clock, steps, true, asked.clone());
            let read = read_all(&mut run).await;
            run.task.await.unwrap();
            assert!(read.last, "{}", read.text());

            // Streamed, without the schema's digest: refused, the session stopped.
            let (st, fatal) = state(&id);
            let steps = td_stream(
                &["tr", "ue"],
                x0ne_block_claiming(&id, "true", (3, 2), DecodingClaims::default()),
                (3, 2),
            );
            let mut run = launch_asking(&st, &clock, steps, true, asked);
            let read = read_all(&mut run).await;
            run.task.await.unwrap();
            assert!(!read.last);
            let (code, message) = read.error();
            assert_eq!(code, "receipt_verification_failed");
            assert!(message.contains("json_schema"), "{message}");
            assert_eq!(fatal.borrow().clone().map(|f| f.0), Some(65));
        }

        /// SECURITY: an answer to a turn of a conversation counts only if its
        /// receipt signs the context it reused, the cache flag, and a
        /// commitment that opens to the handles; a receipt that hides it, or
        /// signs one the request did not ask for, is refused. A failure: a
        /// reuse the caller's verifier never sees.
        #[test]
        fn a_receipt_that_hides_a_conversations_context_is_refused() {
            use crate::context::{AskedConversation, Handle, Issued, SignedContext};
            use crate::decoding::Asked;
            let id = identity();
            let (st, _) = state(&id);
            let sent = Handle::from_bytes([6; 32]);
            let issued = Handle::from_bytes([4; 32]);
            let answer = |context: Option<SignedContext>, kv: bool, handle: Option<&Handle>| {
                let isolation = crate::backend::Isolation {
                    kv_cache_shared_between_requests: kv,
                    ..crate::backend::Isolation::NONE_OF_IT
                };
                let claims = receipt::DecodingClaims {
                    context,
                    ..receipt::DecodingClaims::default()
                };
                let receipt::Signed {
                    receipt,
                    opening,
                    payload,
                } = receipt::build_signed_decoding(
                    &proof(),
                    &id,
                    [5u8; 32],
                    &digest(),
                    "a prompt",
                    "ok",
                    (3, 1),
                    "cpu",
                    isolation,
                    1_000,
                    1_001,
                    None,
                    claims,
                )
                .unwrap();
                let mut x0ne = json!({"receipt": receipt, "opening": opening,
                    "signed_payload_b64": crate::attest_v2::b64(&payload)});
                if let Some(h) = handle {
                    x0ne["conversation"] = json!(Issued { handle: h.clone() });
                }
                json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                    "x0ne": x0ne})
            };
            let asking = |conversation| Asked {
                conversation,
                ..Asked::default()
            };
            let new = asking(AskedConversation::New { end: false });
            let next = asking(AskedConversation::Continue {
                handle: sent.clone(),
                end: false,
            });
            let salt = [5u8; 32];
            let opened = SignedContext::commit(&salt, 1, &[], None, &issued);
            let continued = SignedContext::commit(&salt, 2, &[7, 8], Some(&sent), &issued);
            let ok = |a: &Asked, v: Value| verify_answer(&st, &v, "ok", a, false);
            ok(&new, answer(Some(opened.clone()), true, Some(&issued))).unwrap();
            ok(&next, answer(Some(continued.clone()), true, Some(&issued))).unwrap();
            // A handle the TD no longer held: turn 1, from token 0.
            let restarted = SignedContext::commit(&salt, 1, &[], Some(&sent), &issued);
            ok(&next, answer(Some(restarted), true, Some(&issued))).unwrap();
            let none = Asked::default();
            ok(&none, answer(None, false, None)).unwrap();

            let mut whole = continued.clone();
            whole.reused_tokens = 3;
            let mut later = opened.clone();
            later.turn = 2;
            let mut reused_first = opened.clone();
            reused_first.reused_tokens = 1;
            for (a, v, word) in [
                (&new, answer(None, true, Some(&issued)), "signs no context"),
                (
                    &new,
                    answer(Some(opened.clone()), false, Some(&issued)),
                    "false",
                ),
                (&new, answer(Some(opened.clone()), true, None), "no handle"),
                (
                    &new,
                    answer(Some(opened.clone()), true, Some(&sent)),
                    "does not open",
                ),
                (
                    &next,
                    answer(Some(opened.clone()), true, Some(&issued)),
                    "does not open",
                ),
                (&new, answer(Some(later), true, Some(&issued)), "later turn"),
                (&next, answer(Some(whole), true, Some(&issued)), "prompt"),
                (
                    &new,
                    answer(Some(reused_first), true, Some(&issued)),
                    "opens",
                ),
                (
                    &none,
                    answer(Some(opened.clone()), true, Some(&issued)),
                    "no conversation",
                ),
                (&none, answer(None, true, None), "no conversation"),
                (&none, answer(None, false, Some(&issued)), "did not ask"),
            ] {
                let e = ok(a, v).unwrap_err();
                assert!(e.contains(word), "{word}: {e}");
            }
        }

        /// A whole answer over `text` as `server::chat_whole` writes it.
        fn td_whole(id: &SessionIdentity, text: &str, tokens: (u32, u32)) -> Value {
            let x0ne: Value = serde_json::from_str(&x0ne_block(id, text, tokens)).unwrap();
            json!({
                "id": "chatcmpl-00112233445566778899aabb",
                "object": "chat.completion",
                "created": 1_000,
                "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"}],
                "usage": {"prompt_tokens": tokens.0, "completion_tokens": tokens.1,
                    "total_tokens": tokens.0 + tokens.1},
                "x0ne": x0ne,
            })
        }

        /// An honest whole answer, cut at its budget or not, goes out as the TD wrote it.
        #[test]
        fn a_whole_answer_that_verifies_is_passed_on_as_written() {
            let id = identity();
            let (st, _) = state(&id);
            let asked = crate::decoding::Asked::default();
            let td = td_whole(&id, "hello", (3, 1));
            assert_eq!(whole_answer(&st, &td, &asked).unwrap(), td);
            let mut cut = td.clone();
            cut["choices"][0]["finish_reason"] = json!("length");
            assert_eq!(whole_answer(&st, &cut, &asked).unwrap(), cut);
        }

        /// SECURITY (audit of 2026-11-02): a field added in the receipt is outside the signed
        /// bytes, one added elsewhere outside the receipt. Neither goes out; the receipt that
        /// does still verifies. A failure: the TD's own word in an answer marked verified.
        #[test]
        fn fields_the_receipt_does_not_sign_are_not_passed_on() {
            let id = identity();
            let (st, _) = state(&id);
            let asked = crate::decoding::Asked::default();
            let honest = td_whole(&id, "hello", (3, 1));
            for at in [
                "/x0ne/receipt",
                "/x0ne/receipt/execution",
                "/x0ne/opening",
                "/x0ne",
                "",
                "/choices/0",
                "/choices/0/message",
                "/usage",
            ] {
                let mut td = honest.clone();
                td.pointer_mut(at).unwrap()["note"] = json!("SECRET");
                let out = whole_answer(&st, &td, &asked).unwrap_or_else(|e| panic!("{at}: {e}"));
                assert!(!out.to_string().contains("SECRET"), "{at}: {out}");
                assert_eq!(out, honest, "{at}");
            }
            let mut td = honest.clone();
            td["x0ne"]["receipt"]["note"] = json!("SECRET");
            let out = whole_answer(&st, &td, &asked).unwrap();
            let receipt: InferenceReceipt =
                serde_json::from_value(out["x0ne"]["receipt"].clone()).unwrap();
            let payload =
                crate::attest_v2::unb64(out["x0ne"]["signed_payload_b64"].as_str().unwrap())
                    .unwrap();
            assert!(receipt::verify_signed_payload(&payload, &receipt));
        }

        /// SECURITY (audit of 2026-11-02): the receipt signs one text. Two choices, or none,
        /// are refused as a receipt that does not verify, quoting nothing. A failure: an
        /// unchecked text beside the verified one.
        #[test]
        fn a_whole_answer_without_exactly_one_choice_is_refused() {
            let id = identity();
            let (st, _) = state(&id);
            let asked = crate::decoding::Asked::default();
            let honest = td_whole(&id, "hello", (3, 1));
            let mut two = honest.clone();
            two["choices"]
                .as_array_mut()
                .unwrap()
                .push(json!({"index": 1,
                "message": {"role": "assistant", "content": "SECRET"}, "finish_reason": "stop"}));
            let mut none = honest.clone();
            none["choices"] = json!([]);
            let mut missing = honest.clone();
            missing.as_object_mut().unwrap().remove("choices");
            for (what, td) in [("two", two), ("none", none), ("missing", missing)] {
                let e = whole_answer(&st, &td, &asked).unwrap_err();
                assert!(e.contains("exactly one choice"), "{what}: {e}");
                assert!(!e.contains("SECRET"), "{what}: {e}");
            }
        }

        /// SECURITY (audit of 2026-11-02): the usage is the receipt's token counts. A whole
        /// answer whose usage says otherwise, or nothing, is refused, as a stream's usage chunk is.
        #[test]
        fn a_whole_answer_whose_usage_is_not_the_receipts_is_refused() {
            let id = identity();
            let (st, _) = state(&id);
            let asked = crate::decoding::Asked::default();
            let honest = td_whole(&id, "hello", (3, 1));
            for (key, value) in [
                ("prompt_tokens", json!(4)),
                ("completion_tokens", json!(2)),
                ("total_tokens", json!(5)),
                ("completion_tokens", json!("1")),
                ("prompt_tokens", Value::Null),
            ] {
                let mut td = honest.clone();
                td["usage"][key] = value;
                let e = whole_answer(&st, &td, &asked).unwrap_err();
                assert!(e.contains("usage"), "{key}: {e}");
            }
            let mut td = honest.clone();
            td.as_object_mut().unwrap().remove("usage");
            let e = whole_answer(&st, &td, &asked).unwrap_err();
            assert!(e.contains("usage"), "{e}");
        }

        /// The rest of a whole answer's shape, held as a stream's chunk is: refused otherwise,
        /// quoting nothing of it.
        #[test]
        fn a_whole_answer_out_of_shape_is_refused() {
            let id = identity();
            let (st, _) = state(&id);
            let asked = crate::decoding::Asked::default();
            let honest = td_whole(&id, "hello", (3, 1));
            for (at, value) in [
                ("/id", json!("chatcmpl-SECRET")),
                ("/created", json!(-1)),
                ("/model", json!(["SECRET"])),
                ("/object", json!("chat.completion.chunk")),
                ("/choices/0/index", json!(1)),
                ("/choices/0/message/role", json!("user")),
                ("/choices/0/message/content", Value::Null),
                ("/choices/0/finish_reason", json!("tool_calls")),
                ("/choices/0/finish_reason", Value::Null),
                ("/x0ne/signed_payload_b64", json!(["SECRET"])),
            ] {
                let mut td = honest.clone();
                *td.pointer_mut(at).unwrap() = value;
                let e = whole_answer(&st, &td, &asked).unwrap_err();
                assert!(!e.contains("SECRET"), "{at}: {e}");
            }
        }

        /// Runs `bad` after a role and one delta; the session must stop on a
        /// fixed reason that quotes nothing of `bad`.
        async fn refused_as_protocol(bad: Vec<Step>, secret: &str) {
            let id = identity();
            let (st, fatal) = state(&id);
            let clock = FakeClock::new();
            let mut td = Td::new();
            let mut steps = vec![td.role(), td.delta("a")];
            steps.extend(bad);
            steps.push(Step::Hang);
            let mut run = launch(&st, &clock, steps, true);
            let read = read_all(&mut run).await;
            run.task.await.unwrap();
            assert!(!read.last);
            let (code, message) = read.error();
            assert_eq!(code, "stream_protocol", "{}", read.text());
            assert!(!read.text().contains(secret), "{}", read.text());
            assert!(!read.text().contains("receipt_verified"));
            let (code, reason) = fatal.borrow().clone().expect("the session stops");
            assert_eq!(code, 65);
            assert!(reason.starts_with("stream protocol: "), "{reason}");
            assert!(!reason.contains(secret) && !message.contains(secret));
            assert!(st.is_stopping() && run.seen.closed());
        }

        /// V3. Anything the whitelist does not name is a protocol error, the
        /// session stopped, the value never quoted.
        #[tokio::test]
        async fn fields_outside_the_whitelist_are_a_protocol_error() {
            let mut cases: Vec<(Value, &str)> = vec![
                (
                    td_chunk(
                        json!({"content": "x", "reasoning_content": "SECRET1"}),
                        Value::Null,
                    ),
                    "SECRET1",
                ),
                (
                    td_chunk(json!({"content": {"t": "SECRET4"}}), Value::Null),
                    "SECRET4",
                ),
                (
                    td_chunk(
                        json!({"role": "assistant", "content": "SECRET5"}),
                        Value::Null,
                    ),
                    "SECRET5",
                ),
            ];
            let mut x0ne_mid = td_chunk(json!({"content": "x"}), Value::Null);
            x0ne_mid["x0ne"] = json!({"receipt_verified": true, "note": "SECRET2"});
            cases.push((x0ne_mid, "SECRET2"));
            let mut two = td_chunk(json!({"content": "SECRET3"}), Value::Null);
            let one = two["choices"][0].clone();
            two["choices"].as_array_mut().unwrap().push(one);
            cases.push((two, "SECRET3"));
            for key in [
                "event",
                "type",
                "error",
                "timings",
                "selected_model_id",
                "usage",
            ] {
                let mut c = td_chunk(json!({"content": "x"}), Value::Null);
                c[key] = json!(format!("SECRET-{key}"));
                cases.push((c, "SECRET-"));
            }
            for (chunk, secret) in cases {
                refused_as_protocol(vec![data(&chunk)], secret).await;
            }
        }

        /// V4. What follows the final is checked before it is released: a
        /// usage other than the receipt's, bytes after `[DONE]`, no `[DONE]`.
        #[tokio::test]
        async fn what_follows_the_final_chunk_is_checked_before_it_is_released() {
            let id = identity();
            let block = || x0ne_block(&id, "ab", (3, 2));

            // Usage other than the receipt: the session stops.
            let (st, fatal) = state(&id);
            let clock = FakeClock::new();
            let mut run = launch(&st, &clock, td_stream(&["a", "b"], block(), (3, 9)), true);
            let read = read_all(&mut run).await;
            assert!(!read.last && !read.text().contains("receipt_verified"));
            assert_eq!(read.error().0, "stream_protocol");
            assert_eq!(fatal.borrow().as_ref().unwrap().0, 65);

            // Bytes after [DONE], in the last frame or after it.
            let mut td = Td::new();
            let (role, a) = (td.role(), td.delta("ab"));
            let mut last = bytes_of(td.last(block(), (3, 2)));
            for split in [false, true] {
                let (st, fatal) = state(&id);
                let clock = FakeClock::new();
                let mut steps = vec![
                    Step::Bytes(bytes_of_ref(&role)),
                    Step::Bytes(bytes_of_ref(&a)),
                ];
                if split {
                    steps.push(Step::Bytes(last.clone()));
                    steps.push(Step::Bytes(b": late\n\n".to_vec()));
                } else {
                    let mut joined = last.clone();
                    joined.extend_from_slice(b": late\n\n");
                    steps.push(Step::Bytes(joined));
                }
                steps.push(Step::End);
                let mut run = launch(&st, &clock, steps, true);
                let read = read_all(&mut run).await;
                assert!(!read.last && !read.text().contains("receipt_verified"));
                assert_eq!(read.error().0, "stream_protocol", "split {split}");
                assert_eq!(fatal.borrow().as_ref().unwrap().0, 65);
            }

            // No [DONE], then a clean end: a stream cut short, the session lost.
            let (st, fatal) = state(&id);
            let clock = FakeClock::new();
            let cut = last.len() - b"data: [DONE]\n\n".len();
            assert!(last.ends_with(b"data: [DONE]\n\n"));
            last.truncate(cut);
            let steps = vec![
                Step::Bytes(bytes_of_ref(&role)),
                Step::Bytes(bytes_of_ref(&a)),
                Step::Bytes(last),
                Step::End,
            ];
            let mut run = launch(&st, &clock, steps, true);
            let read = read_all(&mut run).await;
            assert!(!read.last && !read.text().contains("receipt_verified"));
            assert_eq!(read.error().0, "stream_cut");
            assert!(st.lost.load(SeqCst) && fatal.borrow().is_none());
            assert!(run.seen.closed());
        }

        fn bytes_of_ref(step: &Step) -> Vec<u8> {
            match step {
                Step::Bytes(b) => b.clone(),
                _ => panic!("not bytes"),
            }
        }

        /// V5. The parser refuses a comment over 64 KiB, two `data:` lines,
        /// a carriage return and a lone surrogate: each one stops the session.
        #[tokio::test]
        async fn the_parser_refuses_long_lines_multiline_data_cr_and_lone_surrogates() {
            let chunk = td_chunk(json!({"content": "x"}), Value::Null).to_string();
            let long = format!(":{}\n\n", " ".repeat(64 * 1024));
            let two = format!("data: {chunk}\ndata: {chunk}\n\n");
            let cr = format!("data: {chunk}\r\n\r\n");
            let surrogate = chunk.replace("\"x\"", "\"\\ud800\"");
            assert!(surrogate.contains("\\ud800"));
            let surrogate = format!("data: {surrogate}\n\n");
            for bad in [long, two, cr, surrogate] {
                refused_as_protocol(vec![Step::Bytes(bad.into_bytes())], "SECRET").await;
            }
        }

        /// V6. A body that breaks, or ends in the middle of an event or before
        /// its receipt: `stream_cut`, the session lost, `chat` a 503 after.
        #[tokio::test]
        async fn a_cut_stream_is_stream_cut_and_the_session_is_lost() {
            let id = identity();
            let mut td = Td::new();
            let role = bytes_of(td.role());
            let a = bytes_of(td.delta("a"));
            let last = bytes_of(td.last(x0ne_block(&id, "a", (3, 1)), (3, 1)));
            let ends: [Vec<Step>; 4] = [
                vec![Step::Break],
                vec![Step::Bytes(a[..a.len() / 2].to_vec()), Step::End],
                vec![Step::Bytes(last.clone()), Step::Break],
                vec![Step::Bytes(b"data: [DONE]\n\n".to_vec()), Step::End],
            ];
            for end in ends {
                let (st, fatal) = state(&id);
                let clock = FakeClock::new();
                let mut steps = vec![Step::Bytes(role.clone()), Step::Bytes(a.clone())];
                steps.extend(end);
                let mut run = launch(&st, &clock, steps, true);
                let read = read_all(&mut run).await;
                run.task.await.unwrap();
                assert!(!read.last && !read.text().contains("receipt_verified"));
                let (code, message) = read.error();
                assert_eq!(code, "stream_cut", "{}", read.text());
                assert_eq!(message, CUT);
                assert!(st.lost.load(SeqCst), "lost");
                assert!(fatal.borrow().is_none(), "lost, not refused");
                assert!(run.seen.closed());
                let (status, body) = status_of(&st).await;
                assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
                assert!(body.contains("lost"), "{body}");
            }
        }

        /// V7. On the injected clock: 30 s without a byte, 120 s of padding
        /// without an event, 15 min in all. Each one cuts, the session lost;
        /// padding alone still holds the stream at 119 s.
        #[tokio::test]
        async fn three_deadlines_end_a_stuck_stream() {
            let id = identity();
            let secs = Duration::from_secs;

            // Padding every 29 s keeps a stream open; 30 s of nothing cuts it.
            let mut td = Td::new();
            let mut frames = vec![td.role()];
            for _ in 0..2 {
                frames.push(Step::Advance(secs(29)));
                frames.push(td.padding());
            }
            frames.push(td.delta("a"));
            frames.push(Step::Advance(FRAME_SILENCE));
            frames.push(Step::Hang);

            // Padding every 10 s, no event after "a": open at 119 s, cut at 120 s.
            let at_119 = Arc::new(Notify::new());
            let mut td = Td::new();
            let mut padding = vec![td.role(), td.delta("a")];
            for _ in 0..11 {
                padding.push(Step::Advance(secs(10)));
                padding.push(td.padding());
            }
            padding.push(Step::Advance(secs(9)));
            padding.push(td.padding());
            padding.push(Step::Gate(at_119.clone()));
            padding.push(Step::Advance(secs(1)));
            padding.push(td.padding());
            padding.push(Step::Hang);

            // A delta every 20 s: cut at 15 min whatever comes.
            let mut td = Td::new();
            let mut total = vec![td.role()];
            for _ in 0..60 {
                total.push(Step::Advance(secs(20)));
                total.push(td.delta("x"));
            }
            total.push(Step::Hang);

            for (name, steps, min_deltas, max_deltas) in [
                ("frame", frames, 1, 1),
                ("content", padding, 1, 1),
                ("total", total, 44, 45),
            ] {
                let (st, fatal) = state(&id);
                let clock = FakeClock::new();
                let mut run = launch(&st, &clock, steps, false);
                if name == "content" {
                    settle().await;
                    assert_eq!(clock.now(), clock.base + secs(119));
                    assert!(!run.task.is_finished(), "content: cut before 120 s");
                    assert!(!st.lost.load(SeqCst), "content: lost before 120 s");
                    at_119.notify_one();
                }
                let read = read_all(&mut run).await;
                run.task.await.unwrap();
                assert_eq!(read.error().0, "stream_cut", "{name}: {}", read.text());
                let deltas = read
                    .events()
                    .iter()
                    .filter(|e| {
                        e["choices"][0]["delta"]["content"]
                            .as_str()
                            .is_some_and(|c| !c.is_empty())
                    })
                    .count();
                assert!(
                    (min_deltas..=max_deltas).contains(&deltas),
                    "{name}: {deltas} deltas"
                );
                assert!(st.lost.load(SeqCst) && fatal.borrow().is_none(), "{name}");
                assert!(run.seen.closed(), "{name}");
            }
        }

        /// V8. Two streams on one session; the first has a bad receipt. The
        /// second ends with `session_stopped` and never a verified final,
        /// though the TD sends one; `chat` answers 503.
        #[tokio::test]
        async fn a_failed_session_stops_every_concurrent_stream() {
            let id = identity();
            let (st, fatal) = state(&id);
            let clock = FakeClock::new();
            let gate = Arc::new(Notify::new());
            let mut td = Td::new();
            let second = vec![
                td.role(),
                td.delta("bb"),
                Step::Gate(gate.clone()),
                td.last(x0ne_block(&id, "bb", (3, 1)), (3, 1)),
                Step::End,
            ];
            let mut b = launch(&st, &clock, second, true);
            let (role, _) = item(b.rx.recv().await.unwrap());
            let (bb, _) = item(b.rx.recv().await.unwrap());
            assert!(role.contains("assistant") && bb.contains("\"bb\""));

            let first = td_stream(&["a"], x0ne_block(&id, "not a", (3, 1)), (3, 1));
            let mut a = launch(&st, &clock, first, true);
            let read_a = read_all(&mut a).await;
            assert_eq!(read_a.error().0, "receipt_verification_failed");
            gate.notify_one();

            let read_b = read_all(&mut b).await;
            b.task.await.unwrap();
            assert!(!read_b.last && !read_b.text().contains("receipt_verified"));
            let (code, message) = read_b.error();
            assert_eq!(code, "session_stopped");
            assert_eq!(message, STOPPED);
            assert!(b.seen.closed());
            assert_eq!(fatal.borrow().as_ref().unwrap().0, 65);
            let (status, body) = status_of(&st).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert!(body.contains("stopping"), "{body}");
        }

        /// Frames read in one piece, a client that takes 25 s per item: the
        /// stream still ends whole. Waiting on the client is not the TD's
        /// silence, nor a reason to lose the session.
        #[tokio::test]
        async fn a_slow_local_client_does_not_cut_the_stream() {
            let id = identity();
            let (st, fatal) = state(&id);
            let clock = FakeClock::new();
            let mut td = Td::new();
            let role = td.role();
            let mut burst = Vec::new();
            let mut text = String::new();
            for i in 0..(CLIENT_QUEUE + 2) {
                let piece = format!("p{i}");
                burst.extend(bytes_of(td.delta(&piece)));
                text.push_str(&piece);
            }
            let last = td.last(x0ne_block(&id, &text, (3, 6)), (3, 6));
            let steps = vec![role, Step::Bytes(burst), last, Step::End];
            let mut run = launch(&st, &clock, steps, false);
            let mut items = Vec::new();
            for _ in 0..4 {
                settle().await;
                clock.advance(Duration::from_secs(25));
                settle().await;
                items.push(item(run.rx.recv().await.unwrap()).0);
            }
            let rest = read_all(&mut run).await;
            run.task.await.unwrap();
            items.extend(rest.items);
            let read = Read {
                items,
                last: rest.last,
                failed: rest.failed,
            };
            assert!(read.last, "{}", read.text());
            assert!(!read.text().contains("\"error\""), "{}", read.text());
            let streamed: String = read
                .events()
                .iter()
                .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
                .collect();
            assert_eq!(streamed, text);
            assert!(!st.lost.load(SeqCst) && fatal.borrow().is_none());
            assert!(run.seen.kept.load(SeqCst));
        }

        /// A body that never ends: what a stream to a client that stopped
        /// reading leaves hyper writing into a full socket.
        struct Endless;

        impl hyper::body::Body for Endless {
            type Data = Bytes;
            type Error = std::convert::Infallible;
            fn poll_frame(
                self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
            ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
                static CHUNK: [u8; 64 * 1024] = [b'x'; 64 * 1024];
                Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(&CHUNK)))))
            }
        }

        /// V9. A client that no longer reads: the 30 s write stall abandons
        /// its stream and closes the TD's connection; `LocalServer::stop`
        /// returns after 5 s instead of waiting for that connection forever.
        #[tokio::test]
        async fn stopping_never_waits_more_than_5_s_for_a_client_that_does_not_read() {
            let id = identity();
            let (st, fatal) = state(&id);
            let clock = FakeClock::new();
            let mut td = Td::new();
            let mut steps = vec![td.role()];
            for _ in 0..(CLIENT_QUEUE + 4) {
                steps.push(td.delta("z"));
            }
            steps.push(Step::Hang);
            let run = launch(&st, &clock, steps, false);
            settle().await;
            let reads = run.seen.reads.load(SeqCst);
            assert!(
                reads <= CLIENT_QUEUE + 2,
                "the relay waits on the client: {reads}"
            );
            assert!(!run.task.is_finished());
            clock.advance(WRITE_STALL.checked_sub(Duration::from_secs(1)).unwrap());
            settle().await;
            assert!(!run.task.is_finished(), "not before 30 s");
            clock.advance(Duration::from_secs(1));
            settle().await;
            assert!(run.task.is_finished(), "abandoned at 30 s");
            assert!(run.seen.closed(), "the TD's connection is closed");
            assert!(!st.lost.load(SeqCst) && fatal.borrow().is_none());

            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let app = Router::new().route("/", get(|| async { Body::new(Endless) }));
            let server = serve(listener, addr, app, Some(st.clone()));
            let mut client = tokio::net::TcpStream::connect(addr).await.unwrap();
            use tokio::io::AsyncWriteExt as _;
            client
                .write_all(format!("GET / HTTP/1.1\r\nHost: {addr}\r\n\r\n").as_bytes())
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(300)).await;
            let t0 = std::time::Instant::now();
            server.stop().await;
            let waited = t0.elapsed();
            assert!(
                waited >= STOP_WAIT.checked_sub(Duration::from_millis(100)).unwrap(),
                "the connection was still open: {waited:?}"
            );
            assert!(waited < STOP_WAIT + Duration::from_secs(2), "{waited:?}");
            assert!(st.is_stopping(), "stop tells the streams first");
            assert_eq!(STOP_WAIT, Duration::from_secs(5));
            drop(client);
        }

        /// V10. An error event from the TD: `generation_failed` with the
        /// relay's own message, the TD's not copied; the session goes on.
        #[tokio::test]
        async fn a_td_error_event_is_relayed_with_a_fixed_message() {
            let id = identity();
            let mut td = Td::new();
            let role = td.role();
            let a = td.delta("a");
            let own = td.error();
            let forged = data(&json!({"error": {
                "message": "SECRET from the TD", "type": "x0ne_engine", "code": "generation_failed"}}));
            for error in [own, forged] {
                let (st, fatal) = state(&id);
                let clock = FakeClock::new();
                let steps = vec![
                    Step::Bytes(bytes_of_ref(&role)),
                    Step::Bytes(bytes_of_ref(&a)),
                    error,
                    Step::Break,
                ];
                let mut run = launch(&st, &clock, steps, true);
                let read = read_all(&mut run).await;
                run.task.await.unwrap();
                assert!(!read.last);
                let (code, message) = read.error();
                assert_eq!(code, "generation_failed");
                assert_eq!(message, GENERATION);
                assert!(!read.text().contains("SECRET"), "{}", read.text());
                assert!(!st.lost.load(SeqCst) && fatal.borrow().is_none() && !st.is_stopping());
                assert!(run.seen.closed());
            }
        }

        /// A TD body that ends right after its error event keeps its
        /// connection for the next request, as a whole stream does; one that
        /// breaks, sends more or stays silent is closed. The client reads the
        /// same event and the session goes on, whichever it is.
        #[tokio::test]
        async fn a_td_body_that_ends_after_its_error_event_keeps_its_connection() {
            let id = identity();
            let mut td = Td::new();
            let role = bytes_of(td.role());
            let a = bytes_of(td.delta("a"));
            let error = bytes_of(td.error());
            let mut joined = error.clone();
            joined.extend_from_slice(b": \n\n");
            let ends: [(&str, Vec<Step>, bool); 5] = [
                ("end", vec![Step::Bytes(error.clone()), Step::End], true),
                (
                    "break",
                    vec![Step::Bytes(error.clone()), Step::Break],
                    false,
                ),
                ("more", vec![Step::Bytes(joined), Step::End], false),
                (
                    "more after",
                    vec![Step::Bytes(error.clone()), Step::Bytes(b": \n\n".to_vec())],
                    false,
                ),
                (
                    "silent",
                    vec![
                        Step::Bytes(error.clone()),
                        Step::Advance(ERROR_END),
                        Step::Hang,
                    ],
                    false,
                ),
            ];
            for (name, end, kept) in ends {
                let (st, fatal) = state(&id);
                let clock = FakeClock::new();
                let mut steps = vec![Step::Bytes(role.clone()), Step::Bytes(a.clone())];
                steps.extend(end);
                let mut run = launch(&st, &clock, steps, false);
                let read = read_all(&mut run).await;
                run.task.await.unwrap();
                assert_eq!(
                    read.error(),
                    ("generation_failed".into(), GENERATION.into()),
                    "{name}"
                );
                assert!(!st.lost.load(SeqCst) && fatal.borrow().is_none() && !st.is_stopping());
                assert_eq!(run.seen.kept.load(SeqCst), kept, "{name}");
                assert!(run.seen.dropped.load(SeqCst), "{name}");
            }
        }

        /// Whether each item queued for the client so far is `Failed`.
        fn queued(run: &mut Run) -> Vec<bool> {
            let mut items = Vec::new();
            while let Ok(out) = run.rx.try_recv() {
                items.push(matches!(out, Out::Failed(_)));
            }
            items
        }

        /// The TD's body end after its error event is read before the event
        /// is queued for the client, as a whole stream's end is before its
        /// final: the connection is back in the pool when the client's body
        /// ends. A TD body still open holds the event until the stream's
        /// deadline at most, even when that comes before [`ERROR_END`].
        #[tokio::test]
        async fn a_failed_streams_td_connection_is_kept_before_its_event_is_queued() {
            let id = identity();
            let mut td = Td::new();
            let role = bytes_of(td.role());
            let a = bytes_of(td.delta("a"));
            let error = bytes_of(td.error());

            let (st, _fatal) = state(&id);
            let clock = FakeClock::new();
            let gate = Arc::new(Notify::new());
            let steps = vec![
                Step::Bytes(role.clone()),
                Step::Bytes(a.clone()),
                Step::Bytes(error.clone()),
                Step::Gate(gate.clone()),
                Step::End,
            ];
            let mut run = launch(&st, &clock, steps, false);
            settle().await;
            let early = queued(&mut run);
            assert!(
                !early.is_empty() && !early.contains(&true),
                "the event waits for the TD's body end: {early:?}"
            );
            gate.notify_one();
            let mut kept_at_event = None;
            while let Some(out) = run.rx.recv().await {
                if matches!(out, Out::Failed(_)) {
                    kept_at_event = Some(run.seen.kept.load(SeqCst));
                }
            }
            run.task.await.unwrap();
            assert_eq!(kept_at_event, Some(true), "kept before the event");

            let (st, _fatal) = state(&id);
            let clock = FakeClock::new();
            let steps = vec![
                Step::Bytes(role),
                Step::Bytes(a),
                Step::Advance(CHAT_TIMEOUT.checked_sub(ERROR_END / 2).unwrap()),
                Step::Bytes(error),
                Step::Hang,
            ];
            let mut run = launch(&st, &clock, steps, false);
            settle().await;
            assert!(!queued(&mut run).contains(&true), "the TD's end awaited");
            clock.advance(ERROR_END / 2);
            settle().await;
            assert_eq!(queued(&mut run), [true], "the event at the deadline");
            run.task.await.unwrap();
            assert!(run.seen.closed(), "a TD body still open is closed");
        }

        /// A client that leaves a stream the TD froze, over 1 s without a
        /// byte, loses the session; one that leaves a live stream does not.
        /// A failure is a frozen stream that ends with no trace.
        #[tokio::test]
        async fn a_client_that_leaves_a_frozen_stream_loses_the_session() {
            let id = identity();
            let ms = Duration::from_millis;
            for (silence, lost) in [(ms(900), false), (FROZEN + ms(100), true)] {
                let (st, fatal) = state(&id);
                let clock = FakeClock::new();
                let mut td = Td::new();
                let steps = vec![td.role(), td.delta("a"), Step::Advance(silence), Step::Hang];
                let mut run = launch(&st, &clock, steps, false);
                let (role, _) = item(run.rx.recv().await.unwrap());
                let (a, _) = item(run.rx.recv().await.unwrap());
                assert!(role.contains("assistant") && a.contains("\"a\""));
                settle().await;
                assert_eq!(clock.now(), clock.base + silence);
                drop(run.rx);
                run.task.await.unwrap();
                assert_eq!(st.lost.load(SeqCst), lost, "{silence:?}");
                assert!(fatal.borrow().is_none() && !st.is_stopping(), "{silence:?}");
                assert!(run.seen.closed(), "{silence:?}");
            }
        }

        /// The session stops while a whole stream's final waits for its
        /// client: `session_stopped` goes out instead. A failure is a final
        /// marked verified after the session was declared failed.
        #[tokio::test]
        async fn a_final_still_waiting_for_its_client_is_not_sent_once_the_session_stops() {
            let id = identity();
            let (st, fatal) = state(&id);
            let clock = FakeClock::new();
            // With the role chunk, these deltas fill the client's queue.
            let pieces: Vec<String> = (1..CLIENT_QUEUE).map(|i| format!("p{i}")).collect();
            let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
            let steps = td_stream(&refs, x0ne_block(&id, &pieces.concat(), (3, 3)), (3, 3));
            let mut run = launch(&st, &clock, steps, false);
            settle().await;
            assert!(!run.task.is_finished(), "the final waits for the client");
            st.stop_streams();
            settle().await;
            let read = read_all(&mut run).await;
            run.task.await.unwrap();
            assert!(
                !read.last && !read.text().contains("receipt_verified"),
                "{}",
                read.text()
            );
            assert_eq!(
                read.error(),
                ("session_stopped".to_string(), STOPPED.to_string())
            );
            assert!(!st.lost.load(SeqCst) && fatal.borrow().is_none());
        }

        /// A client that stops reading near the end of the 15 min: the
        /// stream is cut at 15 min, not 30 s after its last item. A failure
        /// is a stream that outlives its bound by the write stall.
        #[tokio::test]
        async fn the_15_min_bound_holds_while_the_client_is_waited_on() {
            let id = identity();
            let (st, fatal) = state(&id);
            let clock = FakeClock::new();
            let secs = Duration::from_secs;
            let mut td = Td::new();
            let mut steps = vec![td.role()];
            for _ in 0..44 {
                steps.push(Step::Advance(secs(20)));
                steps.push(td.delta("x"));
            }
            steps.push(Step::Advance(secs(15)));
            let mut burst = Vec::new();
            for _ in 0..=CLIENT_QUEUE {
                burst.extend(bytes_of(td.delta("y")));
            }
            steps.push(Step::Bytes(burst));
            steps.push(Step::Hang);
            let mut run = launch(&st, &clock, steps, false);
            for _ in 0..45 {
                run.rx.recv().await.unwrap();
            }
            settle().await;
            assert_eq!(clock.now(), clock.base + secs(895));
            assert!(!run.task.is_finished(), "the client is waited on");
            clock.advance(secs(4));
            settle().await;
            assert!(!run.task.is_finished(), "not before 15 min");
            clock.advance(secs(1));
            settle().await;
            assert!(run.task.is_finished(), "cut at 15 min, not at 925 s");
            assert!(st.lost.load(SeqCst) && fatal.borrow().is_none());
            assert!(run.seen.closed());
            assert!(!read_all(&mut run).await.last);
        }

        // ── the body to the local client ─────────────────────────────────

        /// Over real HTTP: `Last` and `Failed` end the body clean; a body
        /// closed without either still delivers its bytes, then a body error.
        /// hyper 1.11.1 drops a frame unsent when the body errs in the same poll.
        #[tokio::test]
        async fn the_client_body_ends_clean_only_after_its_last_item_or_its_error_event() {
            for (end, clean) in [("bytes", false), ("failed", true), ("last", true)] {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let addr = listener.local_addr().unwrap();
                let app = Router::new().route(
                    "/",
                    get(move || async move {
                        let (tx, rx) = mpsc::channel(CLIENT_QUEUE);
                        tokio::spawn(async move {
                            let _ = tx
                                .send(Out::Bytes(Bytes::from_static(b"data: 1\n\n")))
                                .await;
                            let bytes = Bytes::from_static(b"data: {\"error\":{}}\n\n");
                            let _ = tx
                                .send(match end {
                                    "last" => Out::Last(bytes),
                                    "failed" => Out::Failed(bytes),
                                    _ => Out::Bytes(bytes),
                                })
                                .await;
                        });
                        Body::new(RelayBody {
                            rx,
                            flow: Flow::Open,
                        })
                    }),
                );
                let server = serve(listener, addr, app, None);
                let mut resp = reqwest::Client::new()
                    .get(format!("http://{addr}/"))
                    .send()
                    .await
                    .unwrap();
                let mut bytes = Vec::new();
                let ended_clean = loop {
                    match resp.chunk().await {
                        Ok(Some(chunk)) => bytes.extend_from_slice(&chunk),
                        Ok(None) => break true,
                        Err(_) => break false,
                    }
                };
                assert_eq!(
                    String::from_utf8_lossy(&bytes),
                    "data: 1\n\ndata: {\"error\":{}}\n\n",
                    "{end}"
                );
                assert_eq!(ended_clean, clean, "{end}");
                server.stop().await;
            }
        }

        /// One HTTP/1.1 response read off `sock`: up to the chunked end, or
        /// to the connection's close when it never comes.
        async fn chunked_response(sock: &mut tokio::net::TcpStream) -> Vec<u8> {
            use tokio::io::AsyncReadExt;
            let mut got = Vec::new();
            let mut buf = [0u8; 4096];
            let read = async {
                while !got.ends_with(b"\r\n0\r\n\r\n") {
                    match sock.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => got.extend_from_slice(&buf[..n]),
                    }
                }
            };
            tokio::time::timeout(Duration::from_secs(10), read)
                .await
                .expect("an end or a close within 10 s");
            got
        }

        /// Over a raw socket: a relayed stream that the TD ends in a failed
        /// generation writes its error event, then the chunked end, and the
        /// connection carries the next request. A failure is a cut after the
        /// event, which HTTP clients report in place of the event.
        #[tokio::test]
        async fn an_error_event_ends_the_client_body_clean_and_keeps_its_connection() {
            use tokio::io::AsyncWriteExt;
            let id = identity();
            let (st, _fatal) = state(&id);
            let clock = FakeClock::new();
            let app = Router::new().route(
                "/",
                get(move || {
                    let (st, clock) = (st.clone(), clock.clone());
                    async move {
                        let mut td = Td::new();
                        let steps = vec![td.role(), td.delta("a"), td.error(), Step::Break];
                        let source = Scripted {
                            steps: steps.into(),
                            clock: clock.clone(),
                            seen: Arc::default(),
                        };
                        let (tx, rx) = mpsc::channel(CLIENT_QUEUE);
                        let started = clock.now();
                        tokio::spawn(relay_stream(
                            st,
                            source,
                            tx,
                            false,
                            crate::decoding::Asked::default(),
                            clock,
                            started,
                        ));
                        Body::new(RelayBody {
                            rx,
                            flow: Flow::Open,
                        })
                    }
                }),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let server = serve(listener, addr, app, None);
            let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
            let event = format!(
                "data: {}\n\n\r\n0\r\n\r\n",
                json!({ "error": { "message": GENERATION, "type": "x0ne_engine",
                                   "code": "generation_failed" } })
            );
            for n in 0..2 {
                sock.write_all(b"GET / HTTP/1.1\r\nHost: t\r\n\r\n")
                    .await
                    .unwrap();
                let got = chunked_response(&mut sock).await;
                let text = String::from_utf8_lossy(&got);
                assert!(text.starts_with("HTTP/1.1 200"), "request {n}: {text}");
                assert!(text.contains("\"a\""), "request {n}: {text}");
                assert!(text.ends_with(&event), "request {n}: {text}");
            }
            server.stop().await;
        }

        // ── retry headers ────────────────────────────────────────────────

        /// Over the real router: the relay's own 5xx, a TD it cannot reach
        /// (502) then the lost session (503), say `x-should-retry: false`
        /// and no `Retry-After`; a 400 and a 200 carry neither.
        #[tokio::test]
        async fn the_relays_own_5xx_say_not_to_retry() {
            let (st, _fatal) = state(&identity());
            let server = start(st.clone(), "127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", server.listen);
            let bearer = crate::cloud_config::hex(&[1u8; 32]);
            let http = reqwest::Client::builder().no_proxy().build().unwrap();
            let mut tools = chat_body();
            tools["tools"] = json!([]);
            for (body, want) in [(tools, 400), (chat_body(), 502), (chat_body(), 503)] {
                let resp = http
                    .post(format!("{base}/v1/chat/completions"))
                    .bearer_auth(&bearer)
                    .json(&body)
                    .send()
                    .await
                    .unwrap();
                let (status, headers) = (resp.status().as_u16(), resp.headers().clone());
                let text = resp.text().await.unwrap();
                assert_eq!(status, want, "{text}");
                assert_eq!(headers.get("retry-after"), None, "{text}");
                if want == 400 {
                    assert!(!headers.contains_key("x-should-retry"), "{text}");
                } else {
                    assert_eq!(headers["x-should-retry"], "false", "{text}");
                }
            }
            assert!(st.lost.load(SeqCst));
            let resp = http
                .get(format!("{base}/x0ne/v1/status"))
                .bearer_auth(&bearer)
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            assert!(!resp.headers().contains_key("x-should-retry"));
            server.stop().await;
        }

        // ── The sealed stream, carried frame by frame ─────────────────────

        mod sealed {
            use super::*;
            use x0ne_sdk::attest::E2eStream;

            fn carrier(cadence: E2eStream) -> E2eCarrier {
                E2eCarrier::new(
                    reqwest::Client::new(),
                    crate::remote_verify::CollateralSources {
                        pcs_base_url: String::new(),
                        root_crl_url: String::new(),
                        gce_bucket_base_url: String::new(),
                    },
                    cadence,
                )
            }

            /// A session carrying the sealed path at `cadence`.
            fn sealed_state_at(cadence: E2eStream) -> (Arc<LocalState>, Fatal) {
                state_with(&identity(), Some(carrier(cadence)))
            }

            fn sealed_state() -> (Arc<LocalState>, Fatal) {
                sealed_state_at(E2eStream::DEFAULT)
            }

            /// One sealed stream relayed on `st`, the TD's body played from `steps`.
            struct Sealed {
                rx: mpsc::Receiver<Out>,
                task: tokio::task::JoinHandle<()>,
                seen: Arc<Seen>,
            }

            fn launch(st: &Arc<LocalState>, clock: &Arc<FakeClock>, steps: Vec<Step>) -> Sealed {
                let seen = Arc::new(Seen::default());
                let source = Scripted {
                    steps: steps.into(),
                    clock: clock.clone(),
                    seen: seen.clone(),
                };
                let (tx, rx) = mpsc::channel(CLIENT_QUEUE);
                let task = tokio::spawn(relay_sealed_stream(
                    st.clone(),
                    source,
                    tx,
                    clock.clone(),
                    clock.now(),
                ));
                Sealed { rx, task, seen }
            }

            /// What the client read: each frame as it came, and whether the
            /// body ended clean (`End`) or broke.
            async fn read(run: &mut Sealed) -> (Vec<Vec<u8>>, bool) {
                let mut frames = Vec::new();
                while let Some(out) = run.rx.recv().await {
                    match out {
                        Out::Bytes(b) => frames.push(b.to_vec()),
                        Out::End => return (frames, true),
                        Out::Last(_) | Out::Failed(_) => {
                            panic!("a sealed stream carries no item of the relay's")
                        }
                    }
                }
                (frames, false)
            }

            /// `/x0ne/v1/status`'s account of the sealed streams.
            async fn sealed_streams(st: &Arc<LocalState>) -> Value {
                status(State(st.clone())).await.0["sealed_streams"].clone()
            }

            /// Three frames in, three out as they came, then the clean end:
            /// no byte opened or held back, the TD's connection kept, the
            /// stream counted.
            #[tokio::test]
            async fn a_sealed_stream_passes_frame_by_frame_and_ends_clean() {
                let (st, _fatal) = sealed_state();
                let clock = FakeClock::new();
                let frames = vec![vec![1u8; 290], vec![2u8; 274], vec![3u8; 16_401]];
                let steps = frames
                    .iter()
                    .cloned()
                    .map(Step::Bytes)
                    .chain([Step::End])
                    .collect();
                let mut run = launch(&st, &clock, steps);
                let (read, clean) = read(&mut run).await;
                assert_eq!(read, frames);
                assert!(clean);
                run.task.await.unwrap();
                assert!(run.seen.kept.load(SeqCst), "the TD's connection is kept");
                assert!(!st.lost.load(SeqCst));
                let s = sealed_streams(&st).await;
                assert_eq!(s["carried"], 1, "{s}");
                assert_eq!(s["cut"], json!({}), "{s}");
                assert_eq!(s["last_cut"], Value::Null, "{s}");
            }

            /// Past the bytes any stream of the session's cadence can put on
            /// the wire (`sealed_stream::max_wire_bytes`), the body is cut,
            /// the cut named, and the session stopped (65): a TD, or a host
            /// between, writing what the cadence does not allow.
            #[tokio::test]
            async fn a_sealed_stream_past_its_byte_bound_is_cut_and_the_session_stopped() {
                let cadence = E2eStream {
                    chunk_bytes: 128,
                    tick_ms: 1000,
                };
                let bound = x0ne_sdk::sealed_stream::max_wire_bytes(&cadence);
                let (st, fatal) = sealed_state_at(cadence);
                let clock = FakeClock::new();
                let frame = vec![7u8; bound / 2 + 1];
                let mut run = launch(
                    &st,
                    &clock,
                    vec![
                        Step::Bytes(frame.clone()),
                        Step::Bytes(frame.clone()),
                        Step::Hang,
                    ],
                );
                let (read, clean) = read(&mut run).await;
                assert_eq!(read, vec![frame]);
                assert!(!clean);
                run.task.await.unwrap();
                assert!(run.seen.closed(), "the TD's connection is closed");
                let s = sealed_streams(&st).await;
                assert_eq!(s["carried"], 0, "{s}");
                assert_eq!(s["cut"][SEALED_STREAM_TOO_LARGE], 1, "{s}");
                assert_eq!(s["last_cut"]["code"], SEALED_STREAM_TOO_LARGE, "{s}");
                let (code, reason) = fatal.borrow().clone().expect("the session stops");
                assert_eq!(code, 65);
                assert!(reason.contains(SEALED_STREAM_TOO_LARGE), "{reason}");
            }

            /// [`FRAME_SILENCE`] without a byte from a TD that writes every
            /// tick: cut, named, and the session lost, as a plain stream's.
            #[tokio::test]
            async fn a_sealed_stream_silent_for_30_s_is_cut_and_the_session_lost() {
                let (st, _fatal) = sealed_state();
                let clock = FakeClock::new();
                let mut run = launch(
                    &st,
                    &clock,
                    vec![
                        Step::Bytes(vec![1; 290]),
                        Step::Advance(FRAME_SILENCE + Duration::from_secs(1)),
                        Step::Hang,
                    ],
                );
                let (read, clean) = read(&mut run).await;
                assert_eq!(read.len(), 1);
                assert!(!clean);
                run.task.await.unwrap();
                assert!(st.lost.load(SeqCst), "the TD went silent: lost");
                assert!(run.seen.closed());
                let s = sealed_streams(&st).await;
                assert_eq!(s["cut"][SEALED_STREAM_SILENT], 1, "{s}");
            }

            /// A stream still open [`CHAT_TIMEOUT`] after its request, frames
            /// coming every 20 s: cut there, named, the session lost. The
            /// SDK's own cap ends every stream before it.
            #[tokio::test]
            async fn a_sealed_stream_open_past_the_relays_15_minutes_is_cut() {
                let (st, _fatal) = sealed_state();
                let clock = FakeClock::new();
                let mut steps = Vec::new();
                for _ in 0..=(CHAT_TIMEOUT.as_secs() / 20) {
                    steps.push(Step::Advance(Duration::from_secs(20)));
                    steps.push(Step::Bytes(vec![1; 274]));
                }
                steps.push(Step::Hang);
                let mut run = launch(&st, &clock, steps);
                let (read, clean) = read(&mut run).await;
                assert!(!clean);
                assert!(
                    read.len() <= 45,
                    "{} frames: carried past the relay's 15 minutes",
                    read.len()
                );
                run.task.await.unwrap();
                assert!(st.lost.load(SeqCst));
                let s = sealed_streams(&st).await;
                assert_eq!(s["cut"][SEALED_STREAM_TOO_LONG], 1, "{s}");
            }

            /// The client leaves: the TD's connection is closed, so the TD
            /// sees its reader gone and stops generating; nothing is lost;
            /// the cut is named.
            #[tokio::test]
            async fn a_client_that_leaves_a_sealed_stream_closes_the_tds_connection() {
                let (st, _fatal) = sealed_state();
                let clock = FakeClock::new();
                let gate = Arc::new(Notify::new());
                let mut run = launch(
                    &st,
                    &clock,
                    vec![
                        Step::Bytes(vec![1; 290]),
                        Step::Gate(gate.clone()),
                        Step::Bytes(vec![2; 274]),
                        Step::Hang,
                    ],
                );
                let first = run.rx.recv().await;
                assert!(matches!(first, Some(Out::Bytes(_))));
                drop(run.rx);
                run.task.await.unwrap();
                assert!(run.seen.closed(), "the TD's connection is closed");
                assert!(!st.lost.load(SeqCst));
                let s = sealed_streams(&st).await;
                assert_eq!(s["cut"][SEALED_STREAM_CLIENT_GONE], 1, "{s}");
                drop(gate);
            }
        }
    }
}
