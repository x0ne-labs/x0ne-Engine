//! The gateway: opens encapsulated requests, hands them to one target, seals the
//! answers. In the design it runs inside each TD, with a key generated there that
//! never leaves it.
//!
//! Errors before decapsulation are plain 4xx; after it, they travel inside the
//! encapsulation, so the relay learns nothing from them (RFC 9458 section 5.2).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use ohttp::hpke::{Aead, Kdf};
use sha2::{Digest, Sha256};

use crate::config::join_keys_media;
use crate::message::{self, InnerRequest, InnerResponse};
use crate::{KEYS_MEDIA_TYPE, REQUEST_MEDIA_TYPE, RESPONSE_MEDIA_TYPE};

/// Key id, KEM, KDF and AEAD ids in front of every encapsulated request.
const REQUEST_HEADER_LEN: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kem {
    /// DHKEM(X25519, HKDF-SHA256), 0x0020.
    X25519,
    /// X-Wing (ML-KEM-768 + X25519), 0x647a: hybrid post-quantum.
    XWing,
}

impl Kem {
    fn ohttp(self) -> ohttp::hpke::Kem {
        match self {
            Kem::X25519 => ohttp::hpke::Kem::X25519Sha256,
            Kem::XWing => ohttp::hpke::Kem::XWing,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("key generation failed")]
    Key,
}

/// A key pair generated here, with its public configuration. The private key
/// has no export path.
pub struct GatewayKey {
    config: ohttp::KeyConfig,
    kem: Kem,
    encoded: Vec<u8>,
}

impl GatewayKey {
    /// One suite only (HKDF-SHA256, AES-128-GCM): every client encapsulates the
    /// same way.
    pub fn generate(key_id: u8, kem: Kem) -> Result<Self, GatewayError> {
        let suites = vec![ohttp::SymmetricSuite::new(Kdf::HkdfSha256, Aead::Aes128Gcm)];
        let config =
            ohttp::KeyConfig::new(key_id, kem.ohttp(), suites).map_err(|_| GatewayError::Key)?;
        let encoded = config.encode().map_err(|_| GatewayError::Key)?;
        Ok(Self {
            config,
            kem,
            encoded,
        })
    }

    pub fn encoded_config(&self) -> &[u8] {
        &self.encoded
    }

    /// The `application/ohttp-keys` body: this configuration alone.
    pub fn keys_media(&self) -> Vec<u8> {
        join_keys_media(&[&self.encoded]).unwrap_or_default()
    }
}

/// The inference server behind the gateway.
pub type Target =
    Arc<dyn Fn(InnerRequest) -> Pin<Box<dyn Future<Output = InnerResponse> + Send>> + Send + Sync>;

/// The gateway's clock; [`Gateway::new`] uses the system's.
pub type Clock = Arc<dyn Fn() -> SystemTime + Send + Sync>;

#[derive(Clone, Debug)]
pub struct GatewayPolicy {
    /// The one authority served; other names get 403.
    pub authority: String,
    /// The paths served; others get 404.
    pub paths: Vec<String>,
    /// Largest encapsulated request (413 above).
    pub max_body: usize,
    /// Requests remembered against replay, at most. A request is remembered once
    /// it passed every other check, until its date leaves the window. When full,
    /// requests are refused (503) until the oldest leave it: a flood of valid
    /// requests still stops the node for that long.
    pub replay_capacity: usize,
    /// Largest distance between a request's date and the gateway's clock (RFC 9458
    /// section 6.5.1); farther gets 400. It must cover the minute the client
    /// rounds down to, clock skew and transit.
    pub replay_window: Duration,
}

/// Refusals before decapsulation, sent in the clear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reject {
    Malformed,
    TooLarge,
}

/// Encapsulated keys seen, each kept until the date window has passed it.
#[derive(Default)]
struct ReplayCache {
    seen: HashSet<[u8; 32]>,
    expiry: BinaryHeap<Reverse<(SystemTime, [u8; 32])>>,
}

impl ReplayCache {
    /// `Some(true)` the first time `h` is seen, `Some(false)` on a replay, `None`
    /// when full. Entries whose `until` is before `now` are dropped first: a request
    /// dated that early is refused by the window anyway.
    fn first_use(
        &mut self,
        h: [u8; 32],
        until: SystemTime,
        now: SystemTime,
        capacity: usize,
    ) -> Option<bool> {
        while let Some(&Reverse((t, old))) = self.expiry.peek() {
            if t >= now {
                break;
            }
            self.expiry.pop();
            self.seen.remove(&old);
        }
        if self.seen.contains(&h) {
            return Some(false);
        }
        if self.seen.len() >= capacity {
            return None;
        }
        self.seen.insert(h);
        self.expiry.push(Reverse((until, h)));
        Some(true)
    }
}

pub struct Gateway {
    server: ohttp::Server,
    n_enc: usize,
    keys_media: Vec<u8>,
    policy: GatewayPolicy,
    target: Target,
    clock: Clock,
    replay: Mutex<ReplayCache>,
}

impl Gateway {
    pub fn new(key: GatewayKey, policy: GatewayPolicy, target: Target) -> Arc<Self> {
        Self::with_clock(key, policy, target, Arc::new(SystemTime::now))
    }

    pub fn with_clock(
        key: GatewayKey,
        policy: GatewayPolicy,
        target: Target,
        clock: Clock,
    ) -> Arc<Self> {
        let keys_media = key.keys_media();
        let n_enc = key.kem.ohttp().n_enc();
        // `Server::new` only checks that the private key is there, and a generated
        // key always has it.
        let server = ohttp::Server::new(key.config).expect("generated key has its private key");
        Arc::new(Self {
            server,
            n_enc,
            keys_media,
            policy,
            target,
            clock,
            replay: Mutex::new(ReplayCache::default()),
        })
    }

    /// One encapsulated request in, one encapsulated response out.
    pub async fn handle(&self, enc_request: &[u8]) -> Result<Vec<u8>, Reject> {
        if enc_request.len() > self.policy.max_body {
            return Err(Reject::TooLarge);
        }
        let (bhttp, sealer) = self
            .server
            .decapsulate(enc_request)
            .map_err(|_| Reject::Malformed)?;
        let inner = self.serve(enc_request, &bhttp).await;
        let plain = message::encode_response(&inner)
            .or_else(|_| message::encode_response(&InnerResponse::status_only(500)))
            .map_err(|_| Reject::Malformed)?;
        sealer.encapsulate(&plain).map_err(|_| Reject::Malformed)
    }

    async fn serve(&self, enc_request: &[u8], bhttp: &[u8]) -> InnerResponse {
        let Some(enc) = enc_request.get(REQUEST_HEADER_LEN..REQUEST_HEADER_LEN + self.n_enc) else {
            return InnerResponse::status_only(400);
        };
        let Ok(decoded) = message::decode_request(bhttp) else {
            return InnerResponse::status_only(400);
        };
        if decoded.scheme != "https" || decoded.authority != self.policy.authority {
            return InnerResponse::status_only(403);
        }
        let mut req = decoded.request;
        if !self.policy.paths.contains(&req.path) {
            return InnerResponse::status_only(404);
        }
        let Ok(date) = message::check_fields(&req.headers) else {
            return InnerResponse::status_only(400);
        };
        let now = (self.clock)();
        let distance = now.duration_since(date).unwrap_or_else(|e| e.duration());
        let Some(until) = date.checked_add(self.policy.replay_window) else {
            return InnerResponse::status_only(400);
        };
        if distance > self.policy.replay_window {
            return InnerResponse::status_only(400);
        }
        // Remembered only now: a request refused above cannot fill the cache.
        match self.first_use(enc, until, now) {
            Some(true) => {}
            Some(false) => return InnerResponse::status_only(409),
            None => return InnerResponse::status_only(503),
        }
        // The date served the replay check; the target does not need it.
        req.headers.pop();
        (self.target)(req).await
    }

    fn first_use(&self, enc: &[u8], until: SystemTime, now: SystemTime) -> Option<bool> {
        let h: [u8; 32] = Sha256::digest(enc).into();
        self.replay
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .first_use(h, until, now, self.policy.replay_capacity)
    }

    pub fn router(self: Arc<Self>) -> Router {
        let limit = self.policy.max_body;
        Router::new()
            .route("/", post(ohttp_post))
            .route("/keys", get(keys))
            .layer(DefaultBodyLimit::max(limit))
            .with_state(self)
    }
}

async fn ohttp_post(State(g): State<Arc<Gateway>>, headers: HeaderMap, body: Bytes) -> Response {
    let typed = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case(REQUEST_MEDIA_TYPE));
    if !typed {
        return StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response();
    }
    match g.handle(&body).await {
        Ok(enc) => typed_body(enc, RESPONSE_MEDIA_TYPE),
        Err(Reject::TooLarge) => StatusCode::PAYLOAD_TOO_LARGE.into_response(),
        Err(Reject::Malformed) => StatusCode::BAD_REQUEST.into_response(),
    }
}

async fn keys(State(g): State<Arc<Gateway>>) -> Response {
    typed_body(g.keys_media.clone(), KEYS_MEDIA_TYPE)
}

fn typed_body(body: Vec<u8>, media_type: &'static str) -> Response {
    let mut out = Response::new(Body::from(body));
    out.headers_mut()
        .insert("content-type", HeaderValue::from_static(media_type));
    out
}
