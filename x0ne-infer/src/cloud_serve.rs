//! `x0ne-infer --cloud-config`: the TD's HTTP surface over the attested
//! channel (`tls`). Nine routes, one state machine:
//!
//! ```text
//! Attested --POST /v2/key (client verdict first)--> Loading --> Ready | Failed
//! Attested --POST /v2/model-key (sealed weights)--> Loading --> Ready | Failed
//! ```
//!
//! `GET /v2/attestation` serves the attestation document (re-attested when the
//! evidence ages past [`REFRESH_AFTER_SECS`](crate::cloud_serve::REFRESH_AFTER_SECS) and no key was released yet,
//! so the key can still be released inside the 300 s freshness window).
//! `GET /v2/attestation/session` takes a quote over `BindingV3`: the TLS
//! exporter of the connection it is asked on, derived here, never sent.
//! `POST /v2/key` is accepted once. `POST /v1/chat/completions` is the
//! existing handler (`server::chat`), reachable only in `Ready`.
//!
//! A sealed config (`model_key.rs`) takes no `/v2/key`: the model owner's
//! relay asks `GET /v2/model-key/recipient?nonce=N`, a quote over the
//! X25519 key born at boot, and posts the owner's key wrapped to it on
//! `POST /v2/model-key`. The private half opens one key and is dropped.
//!
//! The caller-sealed path (docs/sdk-e2e.md), once `Ready`: `GET
//! /v2/e2e/key?nonce=N` is one quote over this boot's OHTTP key, the receipt
//! key and `N` (`e2e_gateway`), under the session quote's gate; on a
//! GPU-attested session, over each GPU's report for the nonce derived from
//! them and its certificate chain too, read first, outside the quote lock.
//! `POST /v2/e2e/chat` takes an OHTTP request sealed to that key and answers
//! it sealed, through the chat handler above.

use crate::attest_v2::{Attestation, BoxFuture, Document, Failure, ModelReport};
use crate::binding::{BindingV2, SessionIdentity};
use crate::channel_key::{ClientKeyBroker, KeyRequest, KEY_ID};
use crate::cloud_config::Loaded;
use crate::keys::ReleasedKey;
use crate::server::{RawEvidence, ServerState, Serving};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use std::collections::{BTreeMap, HashMap};
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Re-attest on `GET /v2/attestation` once the evidence is this old and no
/// key was released: `keys::release` refuses evidence older than 300 s.
pub const REFRESH_AFTER_SECS: u64 = 120;
/// Header carrying the in-TD quote time, `quote;dur=<ms>` (W3C Server Timing).
pub const SERVER_TIMING: &str = "server-timing";
/// `nvmlSystemGetConfComputeGpusReadyState`: 1 = accepting client requests.
pub const NVML_CC_ACCEPTING_CLIENT_REQUESTS_TRUE: u32 = 1;

/// The bounds on quote taking, per gate: session quotes
/// (`GET /v2/attestation/session`, and the document refresh) and key
/// documents (`GET /v2/e2e/key`) each have their own. See
/// [`QuoteLimits::for_queue`] for the numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuoteLimits {
    /// Session quotes one TLS connection may take.
    pub per_connection: u32,
    /// Global bucket: quotes that may start back to back.
    pub burst: u32,
    /// Global bucket: one quote more per this interval.
    pub refill_every: Duration,
    /// Requests that may wait behind the quote in flight; the next gets 503.
    pub waiting: usize,
    /// How long one of them waits before it gets 503.
    pub wait_at_most: Duration,
}

impl QuoteLimits {
    /// Sized on the chat queue (`--max-queue`): the TD runs at most
    /// `max_queue + 1` chats at once, so an honest relay never needs more
    /// fresh connections at once than that, plus the document's.
    /// Default queue 8: burst 10, so a burst of 8 is served.
    #[must_use]
    pub fn for_queue(max_queue: usize) -> Self {
        let burst = u32::try_from(max_queue.saturating_add(2)).unwrap_or(u32::MAX);
        Self {
            per_connection: 1,
            burst,
            refill_every: Duration::from_secs(1),
            waiting: max_queue,
            wait_at_most: Duration::from_secs(10),
        }
    }
}

impl Default for QuoteLimits {
    fn default() -> Self {
        Self::for_queue(crate::server::DEFAULT_MAX_QUEUE)
    }
}

/// `--max-handshakes` when the operator names none.
pub const DEFAULT_MAX_HANDSHAKES: usize = 256;

/// hyper 1.11 arms this on an idle connection too, and a relay's request leaving as that
/// connection closes fails and loses its session (287 of 1000 on a swept gap, 2026-11-05):
/// long, and off the hour a client might poll at.
pub const DEFAULT_HEAD_TIMEOUT: Duration = Duration::from_secs(61 * 60);

/// The bounds on TLS handshakes: the only work the TD does for a peer
/// before mutual TLS has said who it is. And the bound on what follows the
/// handshake until a request head is whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandshakeLimits {
    /// A handshake not finished by then is dropped.
    pub timeout: Duration,
    /// Handshakes in flight. One more drops the oldest handshake of the
    /// source holding the most, never the newcomer.
    pub max_in_flight: usize,
    /// A connection whose next request head is not whole by then, counted from
    /// the handshake or the previous answer's end, is closed: slow or idle.
    pub head_timeout: Duration,
}

impl Default for HandshakeLimits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(10),
            max_in_flight: DEFAULT_MAX_HANDSHAKES,
            head_timeout: DEFAULT_HEAD_TIMEOUT,
        }
    }
}

/// Where a handshake comes from: the IPv4 address, or the /64 of an IPv6
/// one (a single host commonly holds a whole /64).
fn source_of(peer: SocketAddr) -> IpAddr {
    match peer.ip() {
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => IpAddr::V4(v4),
            None => IpAddr::V6(Ipv6Addr::from_bits(v6.to_bits() & !u128::from(u64::MAX))),
        },
        v4 @ IpAddr::V4(_) => v4,
    }
}

struct Pending {
    source: IpAddr,
    cancel: Arc<tokio::sync::Notify>,
}

/// TLS handshakes in flight, oldest first, counted per source.
#[derive(Default)]
struct HandshakeTable {
    next: u64,
    by_age: BTreeMap<u64, Pending>,
    per_source: HashMap<IpAddr, usize>,
}

impl HandshakeTable {
    fn remove(&mut self, id: u64) -> Option<Pending> {
        let gone = self.by_age.remove(&id)?;
        if let Some(n) = self.per_source.get_mut(&gone.source) {
            *n -= 1;
            if *n == 0 {
                self.per_source.remove(&gone.source);
            }
        }
        Some(gone)
    }

    /// The oldest handshake of the source with the most in flight: a few
    /// addresses flooding silent sockets never push out another peer's.
    fn victim(&self) -> Option<u64> {
        let most = self.per_source.values().copied().max()?;
        self.by_age
            .iter()
            .find(|(_, p)| self.per_source.get(&p.source) == Some(&most))
            .map(|(id, _)| *id)
    }
}

/// The handshake table under its [`HandshakeLimits`].
struct Handshakes {
    limits: HandshakeLimits,
    table: Mutex<HandshakeTable>,
}

impl Handshakes {
    fn lock(&self) -> std::sync::MutexGuard<'_, HandshakeTable> {
        self.table
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Admits one handshake from `source`. Full: cancels the victim first
    /// and returns its source.
    fn enter(&self, source: IpAddr) -> (u64, Arc<tokio::sync::Notify>, Option<IpAddr>) {
        let mut t = self.lock();
        let mut dropped = None;
        if t.by_age.len() >= self.limits.max_in_flight.max(1) {
            if let Some(gone) = t.victim().and_then(|id| t.remove(id)) {
                gone.cancel.notify_one();
                dropped = Some(gone.source);
            }
        }
        let id = t.next;
        t.next += 1;
        let cancel = Arc::new(tokio::sync::Notify::new());
        t.by_age.insert(
            id,
            Pending {
                source,
                cancel: cancel.clone(),
            },
        );
        *t.per_source.entry(source).or_default() += 1;
        (id, cancel, dropped)
    }
}

/// One handshake's place in the table, given back however it ends.
struct Slot {
    id: u64,
    state: Arc<CloudState>,
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.state.handshakes.lock().remove(self.id);
    }
}

/// Counts one event and logs the 1st, 2nd, 4th, 8th…: a flood cannot fill
/// the journal.
fn count_and_log(counter: &AtomicU64, line: impl FnOnce(u64) -> String) {
    let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
    if n.is_power_of_two() {
        eprintln!("x0ne-infer: {}", line(n));
    }
}

/// Why a quote was not taken. Bodies say which limit, never more.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteRefusal {
    /// 429: this connection already took its quote.
    Connection,
    /// 429: the global bucket is empty; retry after this many seconds.
    Rate(u64),
    /// 503: one quote in flight and the queue behind it is full.
    Busy,
}

/// One quote in flight, a bounded queue behind it, a global token bucket.
pub struct QuoteGate {
    limits: QuoteLimits,
    in_flight: Arc<tokio::sync::Semaphore>,
    waiting: AtomicUsize,
    bucket: Mutex<(f64, Instant)>,
}

impl QuoteGate {
    #[must_use]
    pub fn new(limits: QuoteLimits) -> Self {
        Self {
            limits,
            in_flight: Arc::new(tokio::sync::Semaphore::new(1)),
            waiting: AtomicUsize::new(0),
            bucket: Mutex::new((f64::from(limits.burst), Instant::now())),
        }
    }

    #[must_use]
    pub fn limits(&self) -> QuoteLimits {
        self.limits
    }

    fn take_token(&self) -> Result<(), QuoteRefusal> {
        let mut b = self
            .bucket
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = Instant::now();
        let every = self.limits.refill_every.as_secs_f64();
        if every > 0.0 {
            let gained = now.duration_since(b.1).as_secs_f64() / every;
            b.0 = (b.0 + gained).min(f64::from(self.limits.burst));
        }
        b.1 = now;
        if b.0 >= 1.0 {
            b.0 -= 1.0;
            return Ok(());
        }
        let wait = ((1.0 - b.0) * every).ceil().max(1.0);
        // Bounded by `refill_every`, a few seconds: no truncation that matters.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Err(QuoteRefusal::Rate(wait as u64))
    }

    /// The single in-flight permit, waiting in the bounded queue if needed.
    async fn permit(&self) -> Result<tokio::sync::OwnedSemaphorePermit, QuoteRefusal> {
        if let Ok(p) = self.in_flight.clone().try_acquire_owned() {
            return Ok(p);
        }
        let queued = self.waiting.fetch_add(1, Ordering::SeqCst);
        let outcome = if queued < self.limits.waiting {
            tokio::time::timeout(
                self.limits.wait_at_most,
                self.in_flight.clone().acquire_owned(),
            )
            .await
            .ok()
            .and_then(Result::ok)
            .ok_or(QuoteRefusal::Busy)
        } else {
            Err(QuoteRefusal::Busy)
        };
        self.waiting.fetch_sub(1, Ordering::SeqCst);
        outcome
    }

    /// Admission of one session quote on a connection that took `taken`.
    ///
    /// # Errors
    /// The [`QuoteRefusal`] to answer: per connection, rate, or busy.
    pub async fn admit(
        &self,
        taken: &AtomicU32,
    ) -> Result<tokio::sync::OwnedSemaphorePermit, QuoteRefusal> {
        if taken.load(Ordering::SeqCst) >= self.limits.per_connection {
            return Err(QuoteRefusal::Connection);
        }
        self.take_token()?;
        let permit = self.permit().await?;
        // Counted once the permit is held: a 503 costs the connection nothing.
        if taken.fetch_add(1, Ordering::SeqCst) >= self.limits.per_connection {
            return Err(QuoteRefusal::Connection);
        }
        Ok(permit)
    }

    /// The TD's own quotes (document refresh): wait for the slot, no limit.
    async fn own(&self) -> tokio::sync::OwnedSemaphorePermit {
        self.in_flight
            .clone()
            .acquire_owned()
            .await
            .expect("the quote semaphore is never closed")
    }
}

pub type Refresh = Box<dyn Fn() -> BoxFuture<'static, Result<Attestation, Failure>> + Send + Sync>;
pub type Loader =
    Box<dyn Fn(&ReleasedKey) -> anyhow::Result<Box<dyn crate::backend::LoadedModel>> + Send + Sync>;
pub type GpuReady = Box<dyn Fn() -> Option<u32> + Send + Sync>;

pub enum Phase {
    Attested,
    Loading,
    Ready(Arc<ServerState>),
    Failed { code: i32, detail: String },
}

impl Phase {
    fn name(&self) -> &'static str {
        match self {
            Phase::Attested => "attested",
            Phase::Loading => "loading",
            Phase::Ready(_) => "ready",
            Phase::Failed { .. } => "failed",
        }
    }
}

/// What the TD has received, for the tests' "neither a question nor a key".
#[derive(Default)]
pub struct Stats {
    pub attestation_gets: AtomicU64,
    /// `GET /v2/attestation/session`: quotes taken, one per connection asked.
    pub session_quotes: AtomicU64,
    /// Session quote requests refused by their [`QuoteGate`] (429 or 503).
    pub session_refused: AtomicU64,
    /// Full TLS handshakes accepted; a resumed one is dropped unserved.
    pub handshakes: AtomicU64,
    /// Handshakes not finished within [`HandshakeLimits::timeout`].
    pub handshake_timeouts: AtomicU64,
    /// Handshakes dropped for a newer one past [`HandshakeLimits::max_in_flight`].
    pub handshakes_dropped: AtomicU64,
    /// Connections closed for [`HandshakeLimits::head_timeout`].
    pub head_timeouts: AtomicU64,
    pub key_posts: AtomicU64,
    /// `GET /v2/model-key/recipient` answered with a quote.
    pub recipient_quotes: AtomicU64,
    /// `POST /v2/model-key`, whatever the answer.
    pub model_key_posts: AtomicU64,
    pub chat_posts: AtomicU64,
    /// `GET /v2/e2e/key` answered with a quote.
    pub e2e_key_quotes: AtomicU64,
    /// Key document requests refused by their [`QuoteGate`] (429 or 503).
    pub e2e_key_refused: AtomicU64,
    /// `POST /v2/e2e/chat`, whatever the answer.
    pub e2e_chat_posts: AtomicU64,
}

/// Recipient quotes a TD gives per boot: a relay asks once; a few retries
/// are allowed, never a quote oracle.
pub const MAX_RECIPIENT_QUOTES: usize = 4;
/// Wrapped keys that do not open before the TD stops taking any.
pub const MAX_MODEL_KEY_FAILURES: u32 = 3;

/// Everything `CloudState::new` needs.
pub struct CloudParts {
    pub identity: SessionIdentity,
    pub binding: BindingV2,
    pub config: Loaded,
    pub model: ModelReport,
    pub version: String,
    pub max_completion_tokens: usize,
    pub attestation: Attestation,
    /// The source the gate was opened on; quotes taken after boot use it.
    pub evidence: Arc<dyn crate::attest_v2::TdEvidence>,
    pub refresh: Refresh,
    pub loader: Loader,
    pub gpu_ready: GpuReady,
    pub serving: Serving,
    pub salt_source: Arc<dyn Fn() -> [u8; 32] + Send + Sync>,
    pub clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    pub quote_limits: QuoteLimits,
    pub handshake_limits: HandshakeLimits,
    /// The platform's standing under Intel's live verdict, renewed by
    /// `tcb_standing::revalidate` while serving; the served model's lease
    /// is judged with it.
    pub standing: Arc<crate::tcb_standing::PlatformStanding>,
    /// For a sealed config: the X25519 key the model key is wrapped to,
    /// born at boot. `None` for public weights.
    pub recipient: Option<crate::model_key::Recipient>,
    /// The OHTTP gateway of the caller-sealed path, its key born at boot.
    pub e2e: crate::e2e_gateway::E2e,
}

pub struct CloudState {
    pub identity: SessionIdentity,
    pub binding: BindingV2,
    pub config: Loaded,
    pub model: ModelReport,
    pub version: String,
    pub max_completion_tokens: usize,
    pub stats: Stats,
    pub e2e: crate::e2e_gateway::E2e,
    attestation: Mutex<Attestation>,
    evidence: Arc<dyn crate::attest_v2::TdEvidence>,
    document: RwLock<Arc<Document>>,
    refresh: Refresh,
    refreshing: tokio::sync::Mutex<()>,
    loader: Loader,
    gpu_ready: GpuReady,
    phase: RwLock<Phase>,
    key_taken: Mutex<bool>,
    pub(crate) serving: Arc<Serving>,
    salt_source: Arc<dyn Fn() -> [u8; 32] + Send + Sync>,
    clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    quotes: QuoteGate,
    /// Key documents' own gate: a flood of them leaves session quotes, and
    /// so new `/v1` connections, their bucket.
    e2e_quotes: QuoteGate,
    handshakes: Handshakes,
    standing: Arc<crate::tcb_standing::PlatformStanding>,
    recipient: Mutex<Option<crate::model_key::Recipient>>,
    /// Release nonces this TD quoted for, in order, and whether a wrapped key
    /// used it up: a wrapped key names one not yet used.
    issued: Mutex<Vec<([u8; 32], bool)>>,
    model_key_failures: AtomicU32,
}

impl CloudState {
    /// The TDX quote of the attestation served now (refreshed before the
    /// key release, the boot quote after): what the live re-check judges.
    pub fn quote(&self) -> Vec<u8> {
        self.attestation
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .quote()
            .to_vec()
    }

    /// The served attestation is an SEV-SNP report, not a TDX quote.
    pub fn is_snp(&self) -> bool {
        self.attestation
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_snp()
    }

    /// The event log of the attestation served now, as its document carries
    /// it for the relay; e2e key documents carry it too.
    pub fn ccel(&self) -> Vec<u8> {
        self.attestation
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .ccel
            .clone()
    }

    pub fn new(p: CloudParts) -> Arc<Self> {
        let document = Document::build(
            &p.attestation,
            &p.identity,
            &p.binding,
            &p.config,
            &p.model,
            &p.version,
        );
        Arc::new(Self {
            identity: p.identity,
            binding: p.binding,
            config: p.config,
            model: p.model,
            version: p.version,
            max_completion_tokens: p.max_completion_tokens,
            stats: Stats::default(),
            e2e: p.e2e,
            attestation: Mutex::new(p.attestation),
            evidence: p.evidence,
            document: RwLock::new(Arc::new(document)),
            refresh: p.refresh,
            refreshing: tokio::sync::Mutex::new(()),
            loader: p.loader,
            gpu_ready: p.gpu_ready,
            phase: RwLock::new(Phase::Attested),
            key_taken: Mutex::new(false),
            serving: Arc::new(p.serving),
            salt_source: p.salt_source,
            clock: p.clock,
            quotes: QuoteGate::new(p.quote_limits),
            e2e_quotes: QuoteGate::new(p.quote_limits),
            standing: p.standing,
            handshakes: Handshakes {
                limits: p.handshake_limits,
                table: Mutex::default(),
            },
            recipient: Mutex::new(p.recipient),
            issued: Mutex::new(Vec::new()),
            model_key_failures: AtomicU32::new(0),
        })
    }

    /// TLS handshakes started and not yet finished, dropped or timed out.
    pub fn handshakes_in_flight(&self) -> usize {
        self.handshakes.lock().by_age.len()
    }

    pub fn document(&self) -> Arc<Document> {
        self.document
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn phase_name(&self) -> &'static str {
        self.phase.read().unwrap_or_else(|e| e.into_inner()).name()
    }

    pub fn attestation_gets(&self) -> u64 {
        self.stats.attestation_gets.load(Ordering::SeqCst)
    }

    /// The chat server, once `Ready`; otherwise the phase's name.
    pub(crate) fn serving_state(&self) -> Result<Arc<ServerState>, &'static str> {
        match &*self.phase.read().unwrap_or_else(|e| e.into_inner()) {
            Phase::Ready(s) => Ok(s.clone()),
            other => Err(other.name()),
        }
    }

    fn set_phase(&self, phase: Phase) {
        *self.phase.write().unwrap_or_else(|e| e.into_inner()) = phase;
    }

    fn evidence_age(&self) -> u64 {
        let collected = self
            .attestation
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .evidence
            .collected_at_unix;
        (self.clock)().saturating_sub(collected)
    }

    /// Re-run the pipeline for the same binding and publish the new document.
    async fn refresh(&self) -> Result<(), Failure> {
        let _one_at_a_time = self.refreshing.lock().await;
        if self.evidence_age() <= REFRESH_AFTER_SECS {
            return Ok(()); // another request refreshed meanwhile
        }
        let fresh = {
            let _slot = self.quotes.own().await;
            (self.refresh)().await?
        };
        let doc = Document::build(
            &fresh,
            &self.identity,
            &self.binding,
            &self.config,
            &self.model,
            &self.version,
        );
        *self.attestation.lock().unwrap_or_else(|e| e.into_inner()) = fresh;
        *self.document.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(doc);
        Ok(())
    }

    fn snapshot(
        &self,
    ) -> (
        crate::gate::Attested,
        x0ne_confidential::model::AttestationEvidence,
        RawEvidence,
    ) {
        let a = self.attestation.lock().unwrap_or_else(|e| e.into_inner());
        (
            a.proof.clone(),
            a.evidence.clone(),
            RawEvidence {
                tdx_quote: a.quote().to_vec(),
                nras_token: a
                    .gpu
                    .as_ref()
                    .map(|g| g.overall_token.clone())
                    .unwrap_or_default(),
            },
        )
    }
}

fn error(status: StatusCode, message: String) -> Response {
    (status, Json(serde_json::json!({ "error": message }))).into_response()
}

async fn attestation(State(st): State<Arc<CloudState>>) -> Response {
    st.stats.attestation_gets.fetch_add(1, Ordering::SeqCst);
    let no_key_yet = matches!(
        *st.phase.read().unwrap_or_else(|e| e.into_inner()),
        Phase::Attested
    );
    if no_key_yet && st.evidence_age() > REFRESH_AFTER_SECS {
        if let Err(f) = st.refresh().await {
            eprintln!("x0ne-infer: re-attestation failed: {f}");
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "re-attestation failed".into(),
            );
        }
    }
    let doc = st.document();
    Json(&*doc).into_response()
}

/// The exporter of the TLS connection a request arrived on, set per
/// connection by [`serve_tls`]. Never read from the request.
#[derive(Clone, Copy)]
pub struct ConnectionExporter(pub [u8; 32]);

/// Quotes taken on the TLS connection a request arrived on, set per
/// connection by [`serve_tls`].
#[derive(Clone, Default)]
pub struct ConnectionQuotes(pub Arc<AtomicU32>);

fn refused(st: &CloudState, why: QuoteRefusal) -> Response {
    quote_refused(&st.stats.session_refused, why)
}

/// A quote refused by a gate, counted in `counter`. One quote per
/// connection, whichever route took it.
fn quote_refused(counter: &AtomicU64, why: QuoteRefusal) -> Response {
    counter.fetch_add(1, Ordering::SeqCst);
    let (status, retry_after, message) = match why {
        QuoteRefusal::Connection => (
            StatusCode::TOO_MANY_REQUESTS,
            None,
            "one session quote per connection",
        ),
        QuoteRefusal::Rate(s) => (StatusCode::TOO_MANY_REQUESTS, Some(s), "too many quotes"),
        QuoteRefusal::Busy => (StatusCode::SERVICE_UNAVAILABLE, Some(1), "busy"),
    };
    let mut r = error(status, message.into());
    if let Some(s) = retry_after {
        if let Ok(v) = axum::http::HeaderValue::from_str(&s.to_string()) {
            r.headers_mut().insert(axum::http::header::RETRY_AFTER, v);
        }
    }
    r
}

/// One quote over `BindingV3` for the connection this request came on.
/// Nothing from the request enters `report_data` (no oracle): the exporter
/// is the connection's, the rest is the boot binding. With the GPU product,
/// then the GPU evidence of this connection, for the nonce derived from its
/// exporter and that quote (`binding::session_gpu_nonce`), under the same
/// quote slot: one attestation at a time, GPU included.
async fn session_attestation(
    State(st): State<Arc<CloudState>>,
    exporter: Option<axum::Extension<ConnectionExporter>>,
    quotes: Option<axum::Extension<ConnectionQuotes>>,
) -> Response {
    let (
        Some(axum::Extension(ConnectionExporter(exporter))),
        Some(axum::Extension(ConnectionQuotes(taken))),
    ) = (exporter, quotes)
    else {
        return error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "no TLS exporter for this connection".into(),
        );
    };
    let _slot = match st.quotes.admit(&taken).await {
        Ok(p) => p,
        Err(why) => return refused(&st, why),
    };
    st.stats.session_quotes.fetch_add(1, Ordering::SeqCst);
    let report_data = crate::binding::BindingV3::of(&st.binding, exporter).tdx_report_data();
    let source = st.evidence.clone();
    let now = (st.clock)();
    let collected = tokio::task::spawn_blocking(move || {
        let _one_at_a_time = crate::server::CHALLENGE_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let t = Instant::now();
        source.tdx(&report_data, now).map(|c| (c, t.elapsed()))
    })
    .await;
    match collected {
        Ok(Ok((c, took))) => {
            let ms = took.as_secs_f64() * 1000.0;
            eprintln!("x0ne-infer: session quote generated in {ms:.1} ms");
            let snp = c.evidence.format == x0ne_confidential::model::EvidenceFormat::SevSnpReport;
            let quote = c.evidence.payload;
            let mut timing = format!("quote;dur={ms:.1}");
            let gpu = if st.config.config.gpu_required() {
                let t = Instant::now();
                match session_gpu(&*st.evidence, &exporter, snp, &quote).await {
                    Ok(g) => {
                        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;
                        eprintln!("x0ne-infer: session GPU evidence attested in {gpu_ms:.1} ms");
                        timing.push_str(&format!(", gpu;dur={gpu_ms:.1}"));
                        Some(g)
                    }
                    // SECURITY: the detail (NVML, NRAS) stays in the TD's log.
                    Err(e) => {
                        eprintln!("x0ne-infer: session GPU evidence failed: {e}");
                        return error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "session GPU attestation unavailable".into(),
                        );
                    }
                }
            } else {
                None
            };
            let mut r = Json(crate::attest_v2::SessionDoc {
                schema: crate::attest_v2::SESSION_SCHEMA.into(),
                quote_b64: if snp {
                    String::new()
                } else {
                    crate::attest_v2::b64(&quote)
                },
                snp: snp.then(|| crate::attest_v2::SnpDoc {
                    evidence_b64: crate::attest_v2::b64(&quote),
                }),
                gpu,
            })
            .into_response();
            // Read by the probe (no shell in the TD): generation times only, no content.
            if let Ok(v) = axum::http::HeaderValue::from_str(&timing) {
                r.headers_mut().insert(SERVER_TIMING, v);
            }
            r
        }
        // SECURITY: the detail (paths, QGS state) stays in the TD's log.
        Ok(Err(e)) => {
            eprintln!("x0ne-infer: session quote failed: {e:#}");
            error(
                StatusCode::SERVICE_UNAVAILABLE,
                "session quote unavailable".into(),
            )
        }
        Err(_) => {
            eprintln!("x0ne-infer: session quote task failed");
            error(
                StatusCode::SERVICE_UNAVAILABLE,
                "session quote unavailable".into(),
            )
        }
    }
}

/// The GPU evidence of the connection that exported `exporter`, whose session
/// quote is `quote`: NRAS asked to echo `session_gpu_nonce(exporter, signed
/// TD report)`. On SEV-SNP the signed bytes are the report's first 0x2A0
/// (`snp_report::SIGNED_LEN`). The client verifies it; the TD only collects
/// and forwards.
async fn session_gpu(
    source: &dyn crate::attest_v2::TdEvidence,
    exporter: &[u8; 32],
    snp: bool,
    quote: &[u8],
) -> Result<crate::attest_v2::SessionGpuDoc, String> {
    let report = if snp {
        quote.get(..x0ne_confidential::snp_report::SIGNED_LEN)
    } else {
        crate::pcs::signed_td_report(quote)
    }
    .ok_or_else(|| {
        format!(
            "session quote of {} bytes has no signed report",
            quote.len()
        )
    })?;
    let nonce = crate::binding::session_gpu_nonce(exporter, report);
    let attested = source
        .gpu(&nonce)
        .await
        .map_err(|e| format!("{} (code {})", e.message, e.code))?;
    Ok(crate::attest_v2::SessionGpuDoc::of(&attested))
}

async fn key(State(st): State<Arc<CloudState>>, body: axum::body::Bytes) -> Response {
    st.stats.key_posts.fetch_add(1, Ordering::SeqCst);
    if st.config.config.is_sealed() {
        return error(
            StatusCode::CONFLICT,
            "this session's weights are sealed: the model key comes through /v2/model-key, \
             wrapped to this TD's attested key"
                .into(),
        );
    }
    let material = match KeyRequest::parse(&body) {
        Ok(m) => m,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    {
        let mut taken = st.key_taken.lock().unwrap_or_else(|e| e.into_inner());
        if *taken {
            return error(
                StatusCode::CONFLICT,
                "a session key was already released".into(),
            );
        }
        *taken = true;
    }
    let broker = match ClientKeyBroker::new(material) {
        Ok(b) => b,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let (proof, evidence, raw) = st.snapshot();
    let key = match crate::keys::release(&proof, &evidence, KEY_ID, &broker, (st.clock)()) {
        Ok(k) => k,
        Err(e) => {
            let detail = format!("key refused: {e:#}");
            st.set_phase(Phase::Failed {
                code: 70,
                detail: detail.clone(),
            });
            return error(StatusCode::FORBIDDEN, detail);
        }
    };
    start_loading(&st, key, proof, raw, None)
}

fn start_loading(
    st: &Arc<CloudState>,
    key: ReleasedKey,
    proof: crate::gate::Attested,
    raw: RawEvidence,
    not_after_unix: Option<u64>,
) -> Response {
    st.set_phase(Phase::Loading);
    let st2 = st.clone();
    tokio::spawn(async move {
        let outcome = load(st2.clone(), key, proof, raw, not_after_unix).await;
        if let Err((code, detail)) = outcome {
            st2.set_phase(Phase::Failed { code, detail });
        }
    });
    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "accepted": true })),
    )
        .into_response()
}

/// `nonce=<64 lowercase hex>`, and nothing else.
fn release_nonce(query: Option<&str>) -> Result<[u8; 32], String> {
    let hex = query
        .and_then(|q| q.strip_prefix("nonce="))
        .ok_or_else(|| "expected ?nonce=<64 hex>".to_string())?;
    crate::cloud_config::unhex32(hex).map_err(|e| format!("nonce: {e}"))
}

/// A quote over the recipient binding: the relay's release nonce, this
/// boot's X25519 key, the session. The nonce enters `report_data` hashed
/// under its own domain (`model_key::CONTEXT`), never as chosen bytes.
async fn model_key_recipient(
    State(st): State<Arc<CloudState>>,
    axum::extract::RawQuery(query): axum::extract::RawQuery,
) -> Response {
    if !st.config.config.is_sealed() {
        return error(
            StatusCode::NOT_FOUND,
            "this session serves public weights: there is no model key".into(),
        );
    }
    // SECURITY: the owner's relay verifies a TDX quote here; an SNP report
    // in its place would be refused there, so it is refused here first.
    if st.is_snp() {
        return error(
            StatusCode::NOT_IMPLEMENTED,
            "sealed weights are released to TDX guests only: this guest is AMD SEV-SNP".into(),
        );
    }
    let nonce = match release_nonce(query.as_deref()) {
        Ok(n) => n,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let attested = matches!(
        *st.phase.read().unwrap_or_else(|e| e.into_inner()),
        Phase::Attested
    );
    let taken = *st.key_taken.lock().unwrap_or_else(|e| e.into_inner());
    let kem_pk = st
        .recipient
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(crate::model_key::Recipient::public_key);
    let Some(kem_pk) = kem_pk.filter(|_| attested && !taken) else {
        return error(
            StatusCode::CONFLICT,
            "no recipient key: the model key was released or refused".into(),
        );
    };
    {
        let mut issued = st.issued.lock().unwrap_or_else(|e| e.into_inner());
        if issued.len() >= MAX_RECIPIENT_QUOTES {
            return error(
                StatusCode::TOO_MANY_REQUESTS,
                "recipient quotes exhausted for this boot".into(),
            );
        }
        issued.push((nonce, false));
    }
    let binding = crate::model_key::RecipientBinding::of(&st.binding, nonce, kem_pk);
    let report_data = binding.tdx_report_data();
    let source = st.evidence.clone();
    let now = (st.clock)();
    let collected = {
        let _slot = st.quotes.own().await;
        tokio::task::spawn_blocking(move || {
            let _one_at_a_time = crate::server::CHALLENGE_LOCK
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            source.tdx(&report_data, now)
        })
        .await
    };
    match collected {
        Ok(Ok(c)) => {
            st.stats.recipient_quotes.fetch_add(1, Ordering::SeqCst);
            Json(crate::model_key::RecipientDoc {
                schema: crate::model_key::RECIPIENT_SCHEMA.into(),
                suite: crate::hpke::SUITE.into(),
                release_nonce_hex: crate::cloud_config::hex(&nonce),
                kem_pk_hex: crate::cloud_config::hex(&kem_pk),
                quote_b64: crate::attest_v2::b64(&c.evidence.payload),
            })
            .into_response()
        }
        // SECURITY: the detail stays in the TD's log, as for session quotes.
        Ok(Err(e)) => {
            eprintln!("x0ne-infer: recipient quote failed: {e:#}");
            error(
                StatusCode::SERVICE_UNAVAILABLE,
                "recipient quote unavailable".into(),
            )
        }
        Err(_) => error(
            StatusCode::SERVICE_UNAVAILABLE,
            "recipient quote unavailable".into(),
        ),
    }
}

/// The model owner's key, wrapped to this boot's X25519 key for a nonce
/// this TD issued. It opens here or nowhere; the grant must name this
/// session's sealed file; then the same release decision as `/v2/key`.
async fn model_key(State(st): State<Arc<CloudState>>, body: axum::body::Bytes) -> Response {
    st.stats.model_key_posts.fetch_add(1, Ordering::SeqCst);
    if !st.config.config.is_sealed() {
        return error(
            StatusCode::NOT_FOUND,
            "this session serves public weights: there is no model key".into(),
        );
    }
    let parts = match crate::model_key::Wrapped::parse(&body) {
        Ok(p) => p,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let opened = {
        let mut taken = st.key_taken.lock().unwrap_or_else(|e| e.into_inner());
        let mut recipient = st.recipient.lock().unwrap_or_else(|e| e.into_inner());
        let attested = matches!(
            *st.phase.read().unwrap_or_else(|e| e.into_inner()),
            Phase::Attested
        );
        let Some(r) = recipient.as_ref().filter(|_| attested && !*taken) else {
            return error(
                StatusCode::CONFLICT,
                "no recipient key: the model key was released or refused".into(),
            );
        };
        // One attempt per nonce: the first wrapped key that names it uses
        // it up, whether it opens or not.
        let issued = st
            .issued
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter_mut()
            .find(|(n, used)| *n == parts.release_nonce && !*used)
            .map(|(_, used)| *used = true)
            .is_some();
        if !issued {
            return error(
                StatusCode::FORBIDDEN,
                "release nonce not issued by this TD, or already used".into(),
            );
        }
        let binding = crate::model_key::RecipientBinding::of(
            &st.binding,
            parts.release_nonce,
            r.public_key(),
        );
        let opened = r
            .open(&binding, &parts)
            .and_then(|o| o.grant.check(&st.config.config, (st.clock)()).map(|()| o));
        match opened {
            Ok(o) => {
                *taken = true;
                *recipient = None; // the private half: one key, then gone
                o
            }
            Err(e) => {
                let n = st.model_key_failures.fetch_add(1, Ordering::SeqCst) + 1;
                let detail = format!("the model key does not open here: {e}");
                if n >= MAX_MODEL_KEY_FAILURES {
                    *recipient = None;
                    st.set_phase(Phase::Failed {
                        code: 70,
                        detail: format!("{n} wrapped keys did not open; the last: {e}"),
                    });
                }
                return error(StatusCode::FORBIDDEN, detail);
            }
        }
    };
    let not_after = opened.grant.not_after_unix;
    let broker = match ClientKeyBroker::for_id(opened.key, crate::model_key::KEY_ID) {
        Ok(b) => b,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let (proof, evidence, raw) = st.snapshot();
    let key = match crate::keys::release(
        &proof,
        &evidence,
        crate::model_key::KEY_ID,
        &broker,
        (st.clock)(),
    ) {
        Ok(k) => k,
        Err(e) => {
            let detail = format!("model key refused: {e:#}");
            st.set_phase(Phase::Failed {
                code: 70,
                detail: detail.clone(),
            });
            return error(StatusCode::FORBIDDEN, detail);
        }
    };
    drop(broker);
    start_loading(&st, key, proof, raw, Some(not_after))
}

/// One generation before ready, so no question pays the first one's setup.
/// Core ships cubins, no PTX to JIT (x0ne-core/kernels/build.rs, 2026-11-04).
/// Fixed text, output dropped; the KV cache is cleared after every generation.
const WARM_UP_PROMPT: &str = "Hello";
const WARM_UP_TOKENS: usize = 2;

/// The model load, off the reactor. The digest the document promised is
/// enforced on what the backend reports: a mismatch never serves.
async fn load(
    st: Arc<CloudState>,
    key: ReleasedKey,
    proof: crate::gate::Attested,
    raw: RawEvidence,
    not_after_unix: Option<u64>,
) -> Result<(), (i32, String)> {
    if st.config.config.gpu_required() {
        match (st.gpu_ready)() {
            Some(NVML_CC_ACCEPTING_CLIENT_REQUESTS_TRUE) => {}
            other => {
                return Err((
                    79,
                    format!("GPU is not accepting client requests (ready state {other:?}): arm it"),
                ))
            }
        }
    }
    let st_load = st.clone();
    let loaded = tokio::task::spawn_blocking(move || {
        let model = (st_load.loader)(&key)?;
        Ok::<_, anyhow::Error>((model, key))
    })
    .await
    .map_err(|_| (72, "model load panicked".to_string()))?;
    let (model, key) = loaded.map_err(|e| (72, format!("model load failed: {e:#}")))?;
    crate::backend::enforce_expected_digest(Some(&st.model.digest_hex), &model.model_digest_hex())
        .map_err(|e| (77, format!("{e:#}")))?;
    let model = tokio::task::spawn_blocking(move || {
        model
            .generate(&crate::prompt::Prompt::text(WARM_UP_PROMPT), WARM_UP_TOKENS)
            .map(|_| model)
    })
    .await
    .map_err(|_| (72, "warm-up generation panicked".to_string()))?
    .map_err(|e| (72, format!("warm-up generation failed: {e:#}")))?;
    // The owner's grant ends the lease at its own expiry, if sooner.
    let lease = match not_after_unix {
        Some(t) => crate::model_key::bound_lease(key.lease(), t),
        None => key.lease().clone(),
    };
    let lease =
        crate::serving_lease::ServingLease::new(lease, key.released_at()).on(st.standing.clone());
    drop(key);
    let salt = st.salt_source.clone();
    let server = ServerState {
        proof,
        identity: st.identity.clone(),
        model,
        max_completion_tokens: st.max_completion_tokens,
        lease,
        raw_evidence: raw,
        salt_source: Box::new(move || salt()),
        challenge_source: Arc::new(crate::attest_v2::TdQuotes(st.evidence.clone())),
    };
    st.set_phase(Phase::Ready(Arc::new(server)));
    Ok(())
}

async fn ready(State(st): State<Arc<CloudState>>) -> Json<serde_json::Value> {
    let phase = st.phase.read().unwrap_or_else(|e| e.into_inner());
    let (ready, detail, code, digest, limits, chunks) = match &*phase {
        Phase::Ready(s) => (
            true,
            None,
            None,
            Some(s.model.model_digest_hex()),
            s.model.serving_limits(),
            crate::backend::prefill_chunks(&*s.model),
        ),
        Phase::Failed { code, detail } => {
            (false, Some(detail.clone()), Some(*code), None, None, None)
        }
        _ => (false, None, None, None, None, None),
    };
    // A prompt of context_tokens or more is a 413 and a completion stops at the
    // context (backend.rs): the client reads the bound here, not by failing.
    Json(serde_json::json!({
        "ready": ready,
        "state": phase.name(),
        "detail": detail,
        "code": code,
        "gpu_ready_state": (st.gpu_ready)(),
        "model_digest_hex": digest,
        "context_tokens": limits.map(|l| l.0),
        "prefill_chunk": limits.map(|l| l.1),
        // A relay streams only from a TD that says so; an older one is emulated.
        "streaming": true,
        "prefill_chunks": chunks,
    }))
}

/// The attested chat. `slot` is the queue place `admit` took: passed on, so
/// the answer's turn holds it until the turn ends here as on `serve_with`.
async fn chat(
    State(st): State<Arc<CloudState>>,
    slot: Option<axum::Extension<crate::server::AdmissionSlot>>,
    body: Result<Json<serde_json::Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    use crate::openai::error::ApiError;
    let asked = tokio::time::Instant::now();
    // Before the count, as when axum's extractor refused it; padded as the
    // chat handler's refusals: its sentence names where the JSON broke.
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejected) => return crate::server::whole_refused(asked, rejected).await,
    };
    st.stats.chat_posts.fetch_add(1, Ordering::SeqCst);
    let server = {
        let phase = st.phase.read().unwrap_or_else(|e| e.into_inner());
        match &*phase {
            Phase::Ready(s) => s.clone(),
            other => {
                return ApiError::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("not serving: state is {}", other.name()),
                )
                .into_response()
            }
        }
    };
    crate::server::chat_as(server, slot, Ok(Json(body)), &st.model.name).await
}

/// `GET /v2/e2e/key?nonce=N`, once `Ready`: one quote over this boot's
/// OHTTP key, `N` and this boot's binding (the receipt key, the boot nonce,
/// the TLS key, the session config), hashed under the e2e label
/// (`e2e_gateway::E2e::report_data`), never as chosen bytes; the document
/// carries the session config's bytes, and the event log of the
/// attestation document (`CloudState::ccel`). On a GPU-attested
/// session the GPUs' reports for `E2e::gpu_nonce` and their chains are read
/// first and bound too (`E2e::report_data_gpu`); without them, no document.
/// The quote passes its own gate: one quote per connection (a session quote
/// counts), its own bucket and queue; the TDX source takes one quote at a
/// time, under the lock session quotes share. The GPUs' evidence is read
/// before that lock, as the relay's session path reads it. Once quoted,
/// that evidence goes to NVIDIA's NRAS (`TdEvidence::nras`), outside the
/// quote's queue place, and the verdict rides in the document: the caller
/// checks it under its own pins. NRAS down: no document.
async fn e2e_key(
    State(st): State<Arc<CloudState>>,
    quotes: Option<axum::Extension<ConnectionQuotes>>,
    axum::extract::RawQuery(query): axum::extract::RawQuery,
) -> Response {
    // SECURITY: e2e key documents carry a TDX quote and its event log, which
    // the SDK verifies as such; an SEV-SNP guest has neither.
    if st.is_snp() {
        return error(
            StatusCode::NOT_IMPLEMENTED,
            "the caller-sealed path is TDX only: this guest is AMD SEV-SNP".into(),
        );
    }
    let Some(axum::Extension(ConnectionQuotes(taken))) = quotes else {
        return error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "no quote count for this connection".into(),
        );
    };
    let nonce = match release_nonce(query.as_deref()) {
        Ok(n) => n,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    if let Err(phase) = st.serving_state() {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("not serving: state is {phase}"),
        );
    }
    let slot = match st.e2e_quotes.admit(&taken).await {
        Ok(p) => p,
        Err(why) => return quote_refused(&st.stats.e2e_key_refused, why),
    };
    let gpu = st.config.config.gpu_required();
    let ccel = st.ccel();
    let source = st.evidence.clone();
    let now = (st.clock)();
    let td = st.clone();
    let collected = tokio::task::spawn_blocking(move || {
        // SECURITY: NVML is read before the quote lock, which session quotes
        // share: a slow NVML held them 2.8 s behind one document (review of
        // 2026-10-27, finding 1). Each document hashes its own evidence.
        let gpus = if gpu {
            let gpus = source
                .gpu_evidence(&td.e2e.gpu_nonce(&nonce))
                .map_err(|e| anyhow::anyhow!("GPU evidence, code {}: {}", e.code, e.message))?;
            Some(gpus)
        } else {
            None
        };
        let report_data = match &gpus {
            None => td.e2e.report_data(&nonce),
            Some(gpus) => td
                .e2e
                .report_data_gpu(&nonce, gpus)
                .map_err(|e| anyhow::anyhow!("GPU evidence: {e}"))?,
        };
        let _one_at_a_time = crate::server::CHALLENGE_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        source.tdx(&report_data, now).map(|c| (c, gpus))
    })
    .await;
    match collected {
        Ok(Ok((c, gpus))) => {
            st.stats.e2e_key_quotes.fetch_add(1, Ordering::SeqCst);
            drop(slot);
            // SECURITY: NRAS judges exactly the evidence the quote bound, for
            // the nonce it answered; the verdict is NVIDIA's signature, checked
            // by the caller, never by this TD on the caller's behalf.
            let verdict = match &gpus {
                None => None,
                Some(g) => {
                    let gpu_nonce = st.e2e.gpu_nonce(&nonce);
                    match st.evidence.nras(&gpu_nonce, g).await {
                        Ok(v) => Some(v),
                        Err(e) => {
                            eprintln!(
                                "x0ne-infer: e2e key GPU verdict failed, code {}: {}",
                                e.code, e.message
                            );
                            return error(
                                StatusCode::SERVICE_UNAVAILABLE,
                                "e2e key GPU verdict unavailable".into(),
                            );
                        }
                    }
                }
            };
            let quote = &c.evidence.payload;
            let doc = match gpus {
                None => st.e2e.document(quote, &ccel),
                Some(gpus) => st.e2e.document_gpu(quote, &ccel, gpus),
            };
            let doc = match verdict {
                Some(v) => crate::e2e_gateway::E2e::with_nras(doc, v),
                None => doc,
            };
            Json(doc).into_response()
        }
        // SECURITY: the detail stays in the TD's log, as for session quotes.
        Ok(Err(e)) => {
            eprintln!("x0ne-infer: e2e key quote failed: {e:#}");
            error(
                StatusCode::SERVICE_UNAVAILABLE,
                "e2e key quote unavailable".into(),
            )
        }
        Err(_) => error(
            StatusCode::SERVICE_UNAVAILABLE,
            "e2e key quote unavailable".into(),
        ),
    }
}

/// `POST /v2/e2e/chat`, once `Ready`: an OHTTP request sealed to this boot's
/// key, answered sealed (`e2e_gateway::answer`); a chunked one
/// (`message/ohttp-chunked-req`) answered as a chunked stream at the
/// session's cadence (`e2e_gateway::answer_stream`). Its queue place is the
/// chat's (`server::admit`); the response nonce is a fresh draw. A body past
/// its type's largest (`x0ne_sdk::e2e::MAX_SEALED_REQUEST`,
/// `MAX_SEALED_STREAM_REQUEST`) is refused as any other, unopened; past the
/// larger, unread.
async fn e2e_chat(
    State(st): State<Arc<CloudState>>,
    slot: Option<axum::Extension<crate::server::AdmissionSlot>>,
    headers: axum::http::HeaderMap,
    body: Result<axum::body::Bytes, axum::extract::rejection::BytesRejection>,
) -> Response {
    let asked = tokio::time::Instant::now();
    st.stats.e2e_chat_posts.fetch_add(1, Ordering::SeqCst);
    let server = match st.serving_state() {
        Ok(s) => s,
        Err(phase) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("not serving: state is {phase}"),
            )
        }
    };
    let ctype = headers
        .get(axum::http::header::CONTENT_TYPE)
        .map(axum::http::HeaderValue::as_bytes);
    let whole = ctype == Some(x0ne_sdk::ohttp::MEDIA_TYPE_REQUEST.as_bytes());
    let chunked = ctype == Some(x0ne_sdk::ohttp::MEDIA_TYPE_CHUNKED_REQUEST.as_bytes());
    let largest = if whole {
        x0ne_sdk::e2e::MAX_SEALED_REQUEST
    } else {
        x0ne_sdk::e2e::MAX_SEALED_STREAM_REQUEST
    };
    let sealed = match body {
        Ok(b) if (whole || chunked) && b.len() <= largest => b,
        _ => return crate::e2e_gateway::refused(asked).await,
    };
    let mut nonce = [0u8; x0ne_sdk::ohttp::RESPONSE_NONCE_LEN];
    nonce.copy_from_slice(&(st.salt_source)()[..x0ne_sdk::ohttp::RESPONSE_NONCE_LEN]);
    if chunked {
        let config = &st.config.config;
        let terms = crate::e2e_gateway::StreamTerms {
            cadence: crate::e2e_stream::cadence_of(config),
            ends: config.expires_at_unix,
        };
        return crate::e2e_gateway::answer_stream(
            &st.e2e,
            server,
            slot,
            &sealed,
            &st.model.name,
            nonce,
            (st.clock)(),
            asked,
            terms,
        )
        .await;
    }
    crate::e2e_gateway::answer(
        &st.e2e,
        server,
        slot,
        &sealed,
        &st.model.name,
        nonce,
        (st.clock)(),
        asked,
    )
    .await
}

pub fn router(state: Arc<CloudState>) -> Router {
    let admitted =
        axum::middleware::from_fn_with_state(state.serving.clone(), crate::server::admit);
    Router::new()
        .route("/v2/attestation", get(attestation))
        .route("/v2/attestation/session", get(session_attestation))
        .route("/v2/key", post(key))
        .route("/v2/model-key/recipient", get(model_key_recipient))
        .route("/v2/model-key", post(model_key))
        .route("/v2/ready", get(ready))
        .route("/v2/e2e/key", get(e2e_key))
        // The largest request the SDK seals, chunked, and no byte more: past
        // it, `e2e_chat`'s one refusal, unread.
        .route(
            "/v2/e2e/chat",
            post(e2e_chat)
                .layer(axum::extract::DefaultBodyLimit::max(
                    x0ne_sdk::e2e::MAX_SEALED_STREAM_REQUEST,
                ))
                .layer(admitted.clone()),
        )
        .route("/v1/chat/completions", post(chat).layer(admitted))
        .with_state(state)
        .layer(axum::middleware::from_fn(crate::server::isolate_panics))
}

/// Serve `router` behind TLS on the listener the gate returned. Each
/// connection shakes hands in its own task: a client the verifier refuses
/// costs one task, never the accept loop. Handshakes are bounded in time
/// and in number ([`HandshakeLimits`]): silent sockets cannot starve the relay.
pub async fn serve_tls(
    listener: tokio::net::TcpListener,
    tls: Arc<rustls::ServerConfig>,
    state: Arc<CloudState>,
) -> std::io::Result<()> {
    let app = router(state.clone());
    serve_tls_app(listener, tls, state, app).await
}

/// [`serve_tls`] with `app` in place of [`router`]: tests wrap the router.
pub(crate) async fn serve_tls_app(
    listener: tokio::net::TcpListener,
    tls: Arc<rustls::ServerConfig>,
    state: Arc<CloudState>,
    app: Router,
) -> std::io::Result<()> {
    let acceptor = tokio_rustls::TlsAcceptor::from(tls);
    let client_nonce = state.config.config.nonce();
    let limits = state.handshakes.limits;
    loop {
        let (tcp, peer) = match listener.accept().await {
            Ok(c) => c,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::ConnectionRefused
                        | std::io::ErrorKind::ConnectionAborted
                        | std::io::ErrorKind::ConnectionReset
                ) =>
            {
                continue
            }
            Err(e) => {
                // Out of descriptors and the like: wait, then go on.
                eprintln!("x0ne-infer: accept: {e}");
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                continue;
            }
        };
        let _ = tcp.set_nodelay(true);
        let source = source_of(peer);
        let (id, cancel, dropped) = state.handshakes.enter(source);
        if let Some(from) = dropped {
            count_and_log(&state.stats.handshakes_dropped, |n| {
                format!(
                    "{} TLS handshakes in flight: dropped the oldest from {from} ({n} so far)",
                    limits.max_in_flight
                )
            });
        }
        let slot = Slot {
            id,
            state: state.clone(),
        };
        let acceptor = acceptor.clone();
        let app = app.clone();
        let stats_state = state.clone();
        tokio::spawn(async move {
            let shake = tokio::time::timeout(limits.timeout, acceptor.accept(tcp));
            let done = tokio::select! {
                biased;
                () = cancel.notified() => return, // dropped for a newer handshake
                done = shake => done,
            };
            // Out of the table before any count: whoever reads a count sees
            // the table already updated (the tests rely on it).
            drop(slot);
            let tls = match done {
                Ok(Ok(tls)) => tls,
                Ok(Err(_)) => return, // refused at the handshake: nothing was served
                Err(_) => {
                    count_and_log(&stats_state.stats.handshake_timeouts, |n| {
                        format!(
                            "TLS handshake from {source} not finished in {:?} ({n} so far)",
                            limits.timeout
                        )
                    });
                    return;
                }
            };
            // The config sends no ticket (`tls::server_config`); a resumption
            // would mean it changed, and it would skip the client check.
            if tls.get_ref().1.handshake_kind() == Some(rustls::HandshakeKind::Resumed) {
                return;
            }
            stats_state.stats.handshakes.fetch_add(1, Ordering::SeqCst);
            let Ok(exporter) = crate::tls::exporter(tls.get_ref().1, &client_nonce) else {
                return;
            };
            let app = app
                .layer(axum::Extension(ConnectionExporter(exporter)))
                .layer(axum::Extension(ConnectionQuotes::default()));
            let io = hyper_util::rt::TokioIo::new(tls);
            let service = hyper_util::service::TowerToHyperService::new(app);
            // SECURITY: without a timer, hyper 1.11 drops its 30 s default in silence.
            let served = hyper::server::conn::http1::Builder::new()
                .timer(hyper_util::rt::TokioTimer::new())
                .header_read_timeout(limits.head_timeout)
                .serve_connection(io, service)
                .await;
            if served.is_err_and(|e| e.is_timeout()) {
                count_and_log(&stats_state.stats.head_timeouts, |n| {
                    format!(
                        "no whole request head from {source} in {:?}, slow or idle: closed ({n} so far)",
                        limits.head_timeout
                    )
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(s: &str) -> IpAddr {
        source_of(SocketAddr::new(s.parse().unwrap(), 443))
    }

    /// One IPv6 host holds a /64: it counts as one source, not 2^64.
    #[test]
    fn an_ipv6_source_is_its_64_and_a_mapped_v4_is_the_v4() {
        assert_eq!(
            source("2001:db8:1:2:aaaa::1"),
            source("2001:db8:1:2:ffff::9")
        );
        assert_ne!(source("2001:db8:1:2::1"), source("2001:db8:1:3::1"));
        assert_eq!(source("::ffff:203.0.113.7"), source("203.0.113.7"));
        assert_ne!(source("203.0.113.7"), source("203.0.113.8"));
    }

    /// Full: the victim is the oldest handshake of the busiest source, and a
    /// victim's departure is never counted twice.
    #[test]
    fn the_victim_is_the_oldest_of_the_busiest_source() {
        let shakes = Handshakes {
            limits: HandshakeLimits {
                timeout: Duration::from_secs(10),
                max_in_flight: 3,
                head_timeout: DEFAULT_HEAD_TIMEOUT,
            },
            table: Mutex::default(),
        };
        let (a, b) = (source("192.0.2.1"), source("192.0.2.2"));
        let (first_a, _, none) = shakes.enter(a);
        assert_eq!(none, None);
        let (first_b, cancel_b, _) = shakes.enter(b);
        let (_, _, none) = shakes.enter(b);
        assert_eq!(none, None);
        // b holds two: its oldest goes, although a's is older.
        let (_, _, dropped) = shakes.enter(a);
        assert_eq!(dropped, Some(b));
        assert!(shakes.lock().by_age.contains_key(&first_a));
        assert!(!shakes.lock().by_age.contains_key(&first_b));
        // The victim was told, and its own departure is a no-op.
        assert!(ready_at_once(cancel_b.notified()));
        assert!(shakes.lock().remove(first_b).is_none());
        let t = shakes.lock();
        assert_eq!(
            (t.by_age.len(), t.per_source[&a], t.per_source[&b]),
            (3, 2, 1)
        );
    }

    fn ready_at_once(f: impl std::future::Future<Output = ()>) -> bool {
        let mut f = std::pin::pin!(f);
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        f.as_mut().poll(&mut cx).is_ready()
    }

    /// The GPU nonce of a session hashes the signed bytes of its CPU
    /// evidence: the TD report of a TDX quote, the first 0x2A0 bytes of an
    /// SEV-SNP report (its certificate is not signed by the chip, not hashed).
    #[tokio::test]
    async fn the_session_gpu_nonce_hashes_the_signed_report_of_either_tee() {
        struct Seen(std::sync::Mutex<Vec<[u8; 32]>>);
        impl crate::attest_v2::TdEvidence for Seen {
            fn tdx(&self, _: &[u8; 64], _: u64) -> anyhow::Result<crate::evidence::Collected> {
                anyhow::bail!("not asked")
            }
            fn ccel(&self) -> anyhow::Result<Vec<u8>> {
                anyhow::bail!("not asked")
            }
            fn gpu<'a>(
                &'a self,
                nonce: &'a [u8; 32],
            ) -> crate::attest_v2::BoxFuture<
                'a,
                Result<crate::attest_v2::GpuAttested, crate::attest_v2::GpuError>,
            > {
                self.0.lock().unwrap().push(*nonce);
                Box::pin(async {
                    Err(crate::attest_v2::GpuError {
                        code: 74,
                        message: "recorded".into(),
                    })
                })
            }
        }
        let seen = Seen(std::sync::Mutex::new(Vec::new()));
        let exporter = [7u8; 32];
        let snp = crate::snp::tests::genoa_evidence(|_| {});
        assert!(session_gpu(&seen, &exporter, true, &snp).await.is_err());
        let signed = &snp[..x0ne_confidential::snp_report::SIGNED_LEN];
        assert_eq!(
            seen.0.lock().unwrap().pop(),
            Some(crate::binding::session_gpu_nonce(&exporter, signed))
        );
        let tdx = vec![0x5Au8; 1200];
        assert!(session_gpu(&seen, &exporter, false, &tdx).await.is_err());
        assert_eq!(
            seen.0.lock().unwrap().pop(),
            Some(crate::binding::session_gpu_nonce(
                &exporter,
                crate::pcs::signed_td_report(&tdx).unwrap()
            ))
        );
        // Too short to hold a signed report: refused before any GPU call.
        let e = session_gpu(&seen, &exporter, true, &snp[..100])
            .await
            .unwrap_err();
        assert!(e.contains("no signed report"), "{e}");
        assert!(seen.0.lock().unwrap().is_empty());
    }
}
