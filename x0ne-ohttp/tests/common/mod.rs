//! Loopback topology for the tests: client -> relay -> spy -> gateway -> target.
//!
//! The spy stands where the relay's upstream connection lands, so what it records
//! is exactly what the relay sends and receives. It can also misbehave the way a
//! hostile relay could: flip bytes, replay, swap responses, fail.
#![allow(dead_code)]

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use x0ne_ohttp::gateway::{Gateway, GatewayKey, GatewayPolicy, Kem, Target};
use x0ne_ohttp::message::{InnerRequest, InnerResponse};
use x0ne_ohttp::relay::{self, RelayConfig};

pub const AUTHORITY: &str = "engine.x0ne.internal";
pub const PATH: &str = "/v1/chat/completions";
/// Accepted distance between a request's date and the gateway's clock.
pub const WINDOW: Duration = Duration::from_secs(300);

/// What the spy does to the traffic it forwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Pass,
    FlipRequestByte,
    FlipResponseByte,
    /// Answers with the response of the previous request.
    ReplayPreviousResponse,
    /// Sends the request to the gateway twice, answers with the first response.
    DuplicateRequest,
    /// Answers 502 without forwarding.
    Fail,
}

#[derive(Clone, Debug)]
pub struct Seen {
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

pub struct Spy {
    pub upstream: String,
    pub mode: Mutex<Mode>,
    pub requests: Mutex<Vec<Seen>>,
    pub responses: Mutex<Vec<Vec<u8>>>,
    http: reqwest::Client,
}

impl Spy {
    pub fn set_mode(&self, mode: Mode) {
        *self.mode.lock().unwrap() = mode;
    }
    pub fn all_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for s in self.requests.lock().unwrap().iter() {
            for (k, v) in &s.headers {
                out.extend_from_slice(k.as_bytes());
                out.extend_from_slice(v.as_bytes());
            }
            out.extend_from_slice(&s.body);
        }
        for r in self.responses.lock().unwrap().iter() {
            out.extend_from_slice(r);
        }
        out
    }
}

fn header_list(h: &HeaderMap) -> Vec<(String, String)> {
    h.iter()
        .map(|(k, v)| {
            (
                k.as_str().to_string(),
                String::from_utf8_lossy(v.as_bytes()).into_owned(),
            )
        })
        .collect()
}

async fn forward(spy: &Spy, body: Vec<u8>) -> Result<(u16, String, Vec<u8>), String> {
    let resp = spy
        .http
        .post(format!("{}/", spy.upstream))
        .header("content-type", "message/ohttp-req")
        .body(body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = resp.status().as_u16();
    let ct = resp
        .headers()
        .get("content-type")
        .map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned())
        .unwrap_or_default();
    let bytes = resp.bytes().await.map_err(|e| e.to_string())?.to_vec();
    Ok((status, ct, bytes))
}

async fn spy_post(State(spy): State<Arc<Spy>>, headers: HeaderMap, body: Bytes) -> Response {
    let mode = *spy.mode.lock().unwrap();
    let mut body = body.to_vec();
    spy.requests.lock().unwrap().push(Seen {
        headers: header_list(&headers),
        body: body.clone(),
    });
    if mode == Mode::Fail {
        return StatusCode::BAD_GATEWAY.into_response();
    }
    if mode == Mode::FlipRequestByte {
        let last = body.len() - 1;
        body[last] ^= 0x01;
    }
    let previous = spy.responses.lock().unwrap().last().cloned();
    let (status, ct, mut resp) = match forward(&spy, body.clone()).await {
        Ok(r) => r,
        Err(_) => return StatusCode::BAD_GATEWAY.into_response(),
    };
    if mode == Mode::DuplicateRequest {
        if let Ok((_, _, second)) = forward(&spy, body).await {
            spy.responses.lock().unwrap().push(second);
        }
    }
    spy.responses.lock().unwrap().push(resp.clone());
    if mode == Mode::FlipResponseByte && !resp.is_empty() {
        let last = resp.len() - 1;
        resp[last] ^= 0x01;
    }
    if mode == Mode::ReplayPreviousResponse {
        if let Some(p) = previous {
            resp = p;
        }
    }
    let mut out = Response::new(axum::body::Body::from(resp));
    *out.status_mut() = StatusCode::from_u16(status).unwrap();
    if !ct.is_empty() {
        out.headers_mut()
            .insert("content-type", ct.parse().expect("content type"));
    }
    out
}

async fn spy_keys(State(spy): State<Arc<Spy>>, headers: HeaderMap) -> Response {
    spy.requests.lock().unwrap().push(Seen {
        headers: header_list(&headers),
        body: Vec::new(),
    });
    let resp = match spy.http.get(format!("{}/keys", spy.upstream)).send().await {
        Ok(r) => r,
        Err(_) => return StatusCode::BAD_GATEWAY.into_response(),
    };
    let status = resp.status().as_u16();
    let bytes = resp.bytes().await.unwrap_or_default().to_vec();
    let mut out = Response::new(axum::body::Body::from(bytes));
    *out.status_mut() = StatusCode::from_u16(status).unwrap();
    out.headers_mut()
        .insert("content-type", "application/ohttp-keys".parse().unwrap());
    out
}

pub async fn serve(router: Router) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve");
    });
    addr
}

/// Records every request that reaches the target.
#[derive(Default)]
pub struct TargetLog {
    pub calls: AtomicUsize,
    pub requests: Mutex<Vec<InnerRequest>>,
}

impl TargetLog {
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

/// A stand-in for the inference server: answers "echo:" + the request body.
pub fn echo_target(log: Arc<TargetLog>) -> Target {
    Arc::new(move |req: InnerRequest| {
        let log = log.clone();
        Box::pin(async move {
            log.calls.fetch_add(1, Ordering::SeqCst);
            let mut body = b"echo:".to_vec();
            body.extend_from_slice(&req.body);
            log.requests.lock().unwrap().push(req);
            InnerResponse {
                status: 200,
                headers: vec![("content-type".into(), "application/json".into())],
                body,
            }
        })
    })
}

pub struct Topology {
    pub relay_url: String,
    pub gateway_url: String,
    pub spy: Arc<Spy>,
    pub target: Arc<TargetLog>,
    pub config: Vec<u8>,
    pub key_id: u8,
}

pub fn policy() -> GatewayPolicy {
    GatewayPolicy {
        authority: AUTHORITY.to_string(),
        paths: vec![PATH.to_string()],
        max_body: 1 << 20,
        replay_capacity: 1024,
        replay_window: WINDOW,
    }
}

pub async fn topology_with(key_id: u8, kem: Kem) -> Topology {
    let key = GatewayKey::generate(key_id, kem).expect("gateway key");
    let config = key.encoded_config().to_vec();
    let target = Arc::new(TargetLog::default());
    let gateway = Gateway::new(key, policy(), echo_target(target.clone()));
    let gateway_addr = serve(gateway.router()).await;
    let gateway_url = format!("http://{gateway_addr}");

    let spy = Arc::new(Spy {
        upstream: gateway_url.clone(),
        mode: Mutex::new(Mode::Pass),
        requests: Mutex::new(Vec::new()),
        responses: Mutex::new(Vec::new()),
        http: reqwest::Client::new(),
    });
    let spy_router = Router::new()
        .route("/", post(spy_post))
        .route("/keys", get(spy_keys))
        .with_state(spy.clone());
    let spy_addr = serve(spy_router).await;

    let relay_router = relay::router(RelayConfig {
        gateway_url: format!("http://{spy_addr}"),
        max_body: 1 << 20,
    })
    .expect("relay router");
    let relay_addr = serve(relay_router).await;

    Topology {
        relay_url: format!("http://{relay_addr}"),
        gateway_url,
        spy,
        target,
        config,
        key_id,
    }
}

pub async fn topology() -> Topology {
    topology_with(7, Kem::X25519).await
}

pub fn chat(body: &str) -> InnerRequest {
    InnerRequest {
        method: "POST".into(),
        path: PATH.into(),
        headers: vec![("content-type".into(), "application/json".into())],
        body: body.as_bytes().to_vec(),
    }
}

pub fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

/// A syntactically valid RFC 9578 token of type 0x0002 (354 bytes): "AAIA" is 00 02 00.
/// Nothing verifies its signature yet.
pub fn private_token() -> String {
    format!("PrivateToken token=\"AAIA{}\"", "A".repeat(468))
}

/// A request encoded the way a client that skips this crate's checks would: these
/// fields, in this order, then encapsulated to `config` with the `ohttp` crate.
pub fn hostile_request(config: &[u8], fields: &[(&str, &str)]) -> (Vec<u8>, ohttp::ClientResponse) {
    let mut m = bhttp::Message::request(
        b"POST".to_vec(),
        b"https".to_vec(),
        AUTHORITY.as_bytes().to_vec(),
        PATH.as_bytes().to_vec(),
    );
    for (k, v) in fields {
        m.put_header(k.as_bytes().to_vec(), v.as_bytes().to_vec());
    }
    m.write_content(b"x");
    let mut out = Vec::new();
    m.write_bhttp(bhttp::Mode::KnownLength, &mut out)
        .expect("bhttp");
    ohttp::ClientRequest::from_encoded_config(config)
        .expect("config")
        .encapsulate(&out)
        .expect("encapsulate")
}

/// The status inside an encapsulated answer.
pub fn status_of(pending: ohttp::ClientResponse, enc_response: &[u8]) -> u16 {
    let bhttp = pending.decapsulate(enc_response).expect("answer opens");
    x0ne_ohttp::message::decode_response(&bhttp)
        .expect("answer decodes")
        .status
}
