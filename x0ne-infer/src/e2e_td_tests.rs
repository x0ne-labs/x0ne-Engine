//! The TD's `/v2/e2e/*` routes, caller to model on one host, no network: the
//! cloud TD's own server (`cloud_serve::serve_tls`, mutual TLS, the chat
//! handler with its padding, cadence and receipts) on the synthetic Intel of
//! `attest_v2::fixtures`; a carrier that holds the client's TLS key, relays
//! Intel's collateral from the (fake) PCS and, for a GPU-attested TD,
//! NVIDIA's verdict from the (fake) NRAS, and records every byte it moves;
//! and x0ne-sdk on the caller's side, which verifies the key document,
//! seals the question, opens the answer and checks its receipt.
//!
//! The TD binary has no simulated evidence source (`evidence.rs`): what runs
//! here is the library code `run_cloud` assembles, on `FakeTd` evidence, as
//! in `cloud_e2e_tests`. The model is scripted (`Scripted`), under the
//! catalog's digest for qwen3:0.6b; `qwen3_*` runs the real weights on CPU
//! and `wasm_*` the wasm32 build in Node (`scripts/e2e-local.sh`).

use crate::attest_v2::fixtures::{
    bucket_with_endorsed, fake_nras, fake_pcs, FakeIntel, FakeNvidia, FakePcs, FakeTd, NrasLie,
};
use crate::attest_v2::{attest_td, catalog_files, unb64, ModelReport, TdAnchors};
use crate::backend::{GenerationParams, LoadedModel};
use crate::binding::{BindingV2, SessionIdentity};
use crate::ccel::{Event, EV_IPL};
use crate::cloud_config::{CloudConfig, Loaded};
use crate::cloud_serve::{self, CloudParts, CloudState};
use crate::e2e_gateway::REFUSED;
use crate::keys::ReleasedKey;
use crate::tls::TlsIdentity;
use axum::body::Bytes;
use axum::extract::{RawQuery, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use serde_json::{json, Value};
use std::sync::atomic::Ordering::SeqCst;
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use x0ne_sdk::attest::{self, Collateral, KeyDocument, Policy};
use x0ne_sdk::e2e::{Client, Gateway, Pending};
use x0ne_sdk::ohttp;

const MODEL: &str = "qwen3:0.6b";
/// The machine type the image reference names and Google's vendored
/// endorsement lists.
const MACHINE: &str = "a3-highgpu-1g";

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn b64(b: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(b)
}

/// Deterministic but distinct 32-byte values, one per call.
fn test_random() -> [u8; 32] {
    use sha2::{Digest, Sha256};
    static N: AtomicU64 = AtomicU64::new(1 << 40);
    Sha256::digest(N.fetch_add(1, SeqCst).to_le_bytes()).into()
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

/// The served-model digest the catalog pins for [`MODEL`].
fn catalog_digest() -> String {
    crate::registry::Catalog::embedded()
        .get(MODEL)
        .unwrap()
        .digest_sha256
        .clone()
        .expect("the catalog pins the served digest")
}

// ── The model ────────────────────────────────────────────────────────────────

/// The question the caller asks, and the word of it a carrier would look for.
const QUESTION: &str = "Which quay of Lyon-Perrache does the 7:42 to Grenoble leave from?";
const QUESTION_NEEDLE: &[u8] = b"Perrache";
/// The scripted model's answer to [`QUESTION`]; no word of it is in the question.
const ANSWER: &str = "Quai Fourvirelle, voie onze.";
const ANSWER_NEEDLE: &[u8] = b"Fourvirelle";

/// A question of another subject whose answer is twenty times longer.
const LONG_QUESTION: &str = "How did the Thirty Years War end, and who signed for whom?";
const LONG_NEEDLE: &[u8] = b"Osnabruck";

/// The scripted model's answer to [`LONG_QUESTION`]: about 600 bytes.
fn long_answer() -> String {
    "Two treaties signed at Munster and Osnabruck in October 1648 ended it:      the Emperor, Sweden and France settled their claims, the princes kept      their own churches, and the Dutch Republic left the Empire for good. "
        .repeat(3)
}

/// A question whose answer [`Scripted`] streams in 34 words of 63 bytes:
/// at L = 256 its last event is queued at the 16th tick, and its rest
/// leaves in the final chunk.
const FULL_QUESTION: &str = "Where did the Empire's diet sit for good after 1663?";
const FULL_NEEDLE: &[u8] = b"Ratisbonne";

fn full_answer() -> String {
    (0..34)
        .map(|i| format!("{i:02}{} ", "Ratisbonne".repeat(6)))
        .collect()
}

/// Questions that make [`Scripted`] panic: in its chat template, before
/// the model's turn, or in its generation, during it.
const PANIC_IN_TEMPLATE: &str = "scripted panic in the template";
const PANIC_IN_GENERATION: &str = "scripted panic in the generation";

fn question(text: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({ "messages": [{ "role": "user", "content": text }] })).unwrap()
}

/// The model of `make ci`: no weights, the catalog's digest for [`MODEL`].
/// It answers [`ANSWER`] to a prompt holding [`QUESTION`], "?" to any other,
/// counts its generations, and panics on [`PANIC_IN_TEMPLATE`] and
/// [`PANIC_IN_GENERATION`].
struct Scripted {
    runs: Arc<AtomicUsize>,
}

impl LoadedModel for Scripted {
    fn generate(
        &self,
        prompt: &crate::prompt::Prompt,
        _max_tokens: usize,
    ) -> anyhow::Result<(String, (u32, u32))> {
        self.runs.fetch_add(1, SeqCst);
        assert!(
            !prompt.as_str().contains(PANIC_IN_GENERATION),
            "scripted panic"
        );
        let text = if prompt.as_str().contains(QUESTION) {
            ANSWER.to_string()
        } else if prompt.as_str().contains(LONG_QUESTION) {
            long_answer()
        } else if prompt.as_str().contains(FULL_QUESTION) {
            full_answer()
        } else {
            "?".to_string()
        };
        Ok((text, (12, 6)))
    }
    /// Word by word, 5 ms apart: a stream whose length and pace follow its text.
    fn generate_streaming(
        &self,
        prompt: &crate::prompt::Prompt,
        params: &GenerationParams,
        sink: &mut dyn crate::backend::TokenSink,
    ) -> anyhow::Result<crate::backend::Completion> {
        let c = self.generate_with(prompt, params)?;
        if sink.started().is_break() || !sink.alive() {
            return Err(crate::backend::Cancelled.into());
        }
        for word in c.text.split_inclusive(' ') {
            std::thread::sleep(Duration::from_millis(5));
            if sink.delta(word).is_break() {
                return Err(crate::backend::Cancelled.into());
            }
        }
        Ok(c)
    }
    fn chat_prompt(&self, messages: &[crate::server::ChatMessage]) -> crate::prompt::Prompt {
        assert!(
            !messages
                .iter()
                .any(|m| m.content.contains(PANIC_IN_TEMPLATE)),
            "scripted panic"
        );
        crate::server::chat_prompt(messages)
    }
    fn model_digest_hex(&self) -> String {
        catalog_digest()
    }
    fn compute_device(&self) -> String {
        "cpu".into()
    }
    fn isolation(&self) -> crate::backend::Isolation {
        crate::backend::Isolation::NONE_OF_IT
    }
}

type Maker = Arc<dyn Fn(&ReleasedKey) -> anyhow::Result<Box<dyn LoadedModel>> + Send + Sync>;

fn scripted(runs: &Arc<AtomicUsize>) -> Maker {
    let runs = runs.clone();
    Arc::new(move |_| Ok(Box::new(Scripted { runs: runs.clone() }) as Box<dyn LoadedModel>))
}

// ── The TD ───────────────────────────────────────────────────────────────────

/// The synthetic Intel, NVIDIA and PCS of one test.
/// Our Google's root, the signer of the honest TD's endorsement.
fn google_root() -> Vec<u8> {
    crate::gce_endorsement::tests::fake_google::fake_google()
        .root_der
        .clone()
}

/// The image reference of `reference::tests`, RTMR0 pinned to the boot of
/// the synthetic firmware (`FakeTd::honest`), our Google's root named.
fn reference_json() -> Vec<u8> {
    let spki = crate::tls::spki_sha256_of_cert(&google_root()).unwrap();
    serde_json::to_vec(&crate::reference::tests::endorsed_reference(
        &hex(&spki),
        MACHINE,
        crate::reference::tests::rtmr0_pins(),
    ))
    .unwrap()
}

struct Lab {
    intel: Arc<FakeIntel>,
    nvidia: Arc<FakeNvidia>,
    pcs: FakePcs,
    /// Google's bucket, mirrored: our Google's endorsement of the honest
    /// TD's firmware, and the binary (`bucket_with_endorsed`).
    bucket: String,
    /// NVIDIA's NRAS, simulated: what the carrier asks for a GPU TD's documents.
    nras: String,
    /// Answers that NRAS gave.
    nras_answers: Arc<AtomicUsize>,
}

impl Lab {
    async fn new() -> Self {
        let intel = Arc::new(FakeIntel::new());
        let pcs = fake_pcs(&intel, Some(now())).await;
        let nvidia = Arc::new(FakeNvidia::new());
        let (nras, nras_answers) = fake_nras(nvidia.clone(), NrasLie::None).await;
        Self {
            intel,
            nvidia,
            pcs,
            bucket: bucket_with_endorsed().await,
            nras,
            nras_answers,
        }
    }

    fn td_anchors(&self) -> TdAnchors {
        TdAnchors {
            tdx: self.intel.anchor(),
            nras_intermediate_pins: self.nvidia.intermediate_pins(),
            tcb_policy: crate::tcb::TcbPolicy::production(),
            root_crl_url: self.pcs.root_crl_url.clone(),
            snp: crate::snp::SnpAnchors::production(),
            snp_policy: crate::snp::SnpPolicy::default(),
        }
    }

    fn honest(&self) -> FakeTd {
        FakeTd::honest(self.intel.clone(), self.nvidia.clone(), now())
    }

    /// The caller's policy for the honest image: the synthetic Intel root,
    /// Google's root, the image reference of `reference::tests` (the log
    /// `FakeTd::honest` replays), signed by the TEST archive key and shown
    /// in the TEST log, and the machine type; `transparent`; no pin.
    fn reference_only(&self) -> Policy {
        let json = reference_json();
        let (signature, tlog_proof) = attest::sign_reference_under_test_keys(&json, now());
        let reference = attest::admit_reference_under_test_keys(
            &attest::SignedReference {
                json: &json,
                signature: &signature,
                tlog_proof: &tlog_proof,
            },
            &mut attest::LogMemory::new(),
            None,
            now(),
        )
        .expect("the lab's reference, signed and logged");
        Policy {
            anchor: x0ne_sdk::TrustAnchor::RootDer(self.intel.root_der.clone()),
            gce_root_der: google_root(),
            // A simulated TD serves on the CPU: the tests' `attest=cpu` (x0ne-sdk `test-anchor`).
            attest: x0ne_sdk::attest::Attest::Cpu,
            ..Policy::production(reference, MACHINE)
        }
    }

    /// [`Self::reference_only`], `td`'s MRTD and RTMR1..3 pinned on top.
    fn policy(&self, td: &FakeTd) -> Policy {
        let r = td.rtmr();
        self.reference_only().pinned(td.mrtd, [r[1], r[2], r[3]])
    }

    /// [`Self::policy`] under `attest=cpu+gpu`, this lab's NVIDIA
    /// intermediate pinned in place of NVIDIA's two.
    fn gpu_policy(&self, td: &FakeTd) -> Policy {
        let mut p = self.policy(td);
        p.attest = attest::Attest::CpuGpu;
        p.nras_pins = vec![x0ne_sdk::nras_test_pin(&self.nvidia.intermediate_der()).unwrap()];
        p
    }
}

/// What a test sets on its TD.
struct Spec {
    td: FakeTd,
    gpu: bool,
    model: Maker,
    max_completion_tokens: usize,
    quote_limits: cloud_serve::QuoteLimits,
}

/// A running TD and the client TLS identity its config pins.
struct Td {
    state: Arc<CloudState>,
    endpoint: String,
    carrier_tls: Arc<TlsIdentity>,
    /// Seconds the host moves the TD's wall clock by (`CloudState::clock`).
    clock_skew: Arc<AtomicI64>,
}

/// The TD `run_cloud` assembles, on `spec.td`'s synthetic evidence, serving
/// `spec.model` under the catalog's name and digest for [`MODEL`].
async fn start_td(lab: &Lab, spec: Spec) -> Td {
    let carrier_tls = TlsIdentity::generate("carrier").unwrap();
    let attest = if spec.gpu { "cpu+gpu" } else { "cpu" };
    // A day: a caller whose clock is an hour ahead still seals (the SDK
    // refuses within a minute of the end), and the TD judges its Date.
    let config: Loaded = CloudConfig::new(
        MODEL,
        carrier_tls.spki_sha256(),
        test_random(),
        now() + 86_400,
        attest,
        443,
        spec.max_completion_tokens,
    )
    .to_loaded()
    .unwrap();
    let catalog = crate::registry::Catalog::embedded();
    let model = ModelReport {
        name: MODEL.into(),
        digest_hex: catalog_digest(),
        files: catalog_files(catalog.get(MODEL).unwrap()),
    };
    let identity = SessionIdentity::generate(test_random(), test_random());
    let tls = TlsIdentity::generate("td").unwrap();
    let binding = BindingV2::of(&identity, tls.spki_sha256(), config.sha256);
    let (gpu, anchors, pcs) = (spec.gpu, lab.td_anchors(), lab.pcs.base_url.clone());
    let attestation = attest_td(&spec.td, &binding, gpu, &pcs, &anchors, &|| now())
        .await
        .unwrap_or_else(|e| panic!("the fake TD attests honestly: {e}"));
    let td = Arc::new(spec.td);
    let evidence: Arc<dyn crate::attest_v2::TdEvidence> = td.clone();
    let refresh: cloud_serve::Refresh = Box::new(move || {
        let (td, anchors, pcs) = (td.clone(), anchors.clone(), pcs.clone());
        Box::pin(async move { attest_td(&*td, &binding, gpu, &pcs, &anchors, &|| now()).await })
    });
    let make = spec.model;
    let loader: cloud_serve::Loader = Box::new(move |key: &ReleasedKey| make(key));
    let client_spki = config.config.client_spki();
    let clock_skew = Arc::new(AtomicI64::new(0));
    let skew = clock_skew.clone();
    let e2e = crate::e2e_gateway::E2e::from_seed(&test_random(), binding, &config.bytes).unwrap();
    let state = CloudState::new(CloudParts {
        identity,
        binding,
        config,
        model,
        version: "test".into(),
        max_completion_tokens: spec.max_completion_tokens,
        attestation,
        evidence,
        refresh,
        loader,
        gpu_ready: Box::new(|| Some(cloud_serve::NVML_CC_ACCEPTING_CLIENT_REQUESTS_TRUE)),
        serving: crate::server::Serving {
            admission: crate::server::Admission::new(8),
            ..crate::server::Serving::default()
        },
        salt_source: Arc::new(test_random),
        clock: Arc::new(move || now().saturating_add_signed(skew.load(SeqCst))),
        quote_limits: spec.quote_limits,
        handshake_limits: cloud_serve::HandshakeLimits::default(),
        standing: Arc::new(crate::tcb_standing::PlatformStanding::verified_at(
            now(),
            crate::tcb::TcbPolicy::production(),
        )),
        recipient: None,
        e2e,
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let tls_config = crate::tls::server_config(&tls, client_spki).unwrap();
    tokio::spawn(cloud_serve::serve_tls(listener, tls_config, state.clone()));
    Td {
        state,
        endpoint,
        carrier_tls: Arc::new(carrier_tls),
        clock_skew,
    }
}

/// An honest TD on the scripted model, its model key released and loaded.
async fn ready_td(lab: &Lab, runs: &Arc<AtomicUsize>) -> Td {
    let td = start_td(
        lab,
        Spec {
            td: lab.honest(),
            gpu: false,
            model: scripted(runs),
            max_completion_tokens: 64,
            quote_limits: cloud_serve::QuoteLimits::for_queue(8),
        },
    )
    .await;
    release_and_wait(&td).await;
    td
}

type Send1 = hyper::client::conn::http1::SendRequest<axum::body::Body>;

/// One mutual-TLS connection to `endpoint` under the client key `tls`.
async fn mtls(endpoint: &str, tls: &TlsIdentity) -> Send1 {
    let tcp = tokio::net::TcpStream::connect(endpoint).await.unwrap();
    let cfg = crate::tls::client_config(tls, crate::tls::ServerPin::new()).unwrap();
    let name = rustls::pki_types::ServerName::try_from("127.0.0.1").unwrap();
    let tls = tokio_rustls::TlsConnector::from(Arc::new(cfg))
        .connect(name, tcp)
        .await
        .unwrap();
    let (send, conn) = hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(tls))
        .await
        .unwrap();
    tokio::spawn(conn);
    send
}

/// What came back: status, its headers as sent, body.
struct Reply {
    status: u16,
    headers: HeaderMap,
    body: Vec<u8>,
}

impl Reply {
    fn content_type(&self) -> &str {
        self.headers
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
    }
}

/// One request on the connection `send`.
async fn request(
    send: &mut Send1,
    method: &str,
    path: &str,
    ctype: Option<&str>,
    body: Vec<u8>,
) -> Reply {
    let mut req = axum::http::Request::builder()
        .method(method)
        .uri(path)
        .header(header::HOST, "td");
    if let Some(ct) = ctype {
        req = req.header(header::CONTENT_TYPE, ct);
    }
    let req = req.body(axum::body::Body::from(body)).unwrap();
    send.ready().await.unwrap();
    let resp = send.send_request(req).await.unwrap();
    let status = resp.status().as_u16();
    let headers = resp.headers().clone();
    let body = axum::body::to_bytes(axum::body::Body::new(resp.into_body()), 8 << 20)
        .await
        .unwrap()
        .to_vec();
    Reply {
        status,
        headers,
        body,
    }
}

/// The floor of every refusal: a whole answer refused before its turn.
fn floor() -> Duration {
    crate::stream::TICK * u32::try_from(crate::padding::MIN_TICKS).unwrap()
}

/// How far past the floor a refusal may arrive: scheduling and the hops,
/// never a second floor nor a model run.
const CADENCE_SLACK: Duration = Duration::from_millis(500);

/// `took` is the refusals' cadence, bounded both ways.
fn assert_cadence(took: Duration, what: &str) {
    let floor = floor();
    assert!(
        took >= floor && took < floor + CADENCE_SLACK,
        "{what}: {took:?}, the floor is {floor:?}"
    );
}

/// One request on a fresh connection to the TD.
async fn call(td: &Td, method: &str, path: &str, ctype: Option<&str>, body: Vec<u8>) -> Reply {
    let mut send = mtls(&td.endpoint, &td.carrier_tls).await;
    request(&mut send, method, path, ctype, body).await
}

/// The key released, the model loaded: `/v2/ready` says so.
async fn release_and_wait(td: &Td) {
    let key = crate::channel_key::KeyRequest::to_json(&[9u8; 32]).into_bytes();
    let r = call(td, "POST", "/v2/key", Some("application/json"), key).await;
    assert_eq!(r.status, 202, "{}", String::from_utf8_lossy(&r.body));
    let deadline = Instant::now() + Duration::from_secs(600);
    loop {
        let r = call(td, "GET", "/v2/ready", None, vec![]).await;
        let v: Value = serde_json::from_slice(&r.body).unwrap();
        if v["ready"] == true {
            return;
        }
        assert!(v["state"] != "failed", "{v}");
        assert!(Instant::now() < deadline, "not ready: {v}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

// ── The carrier ──────────────────────────────────────────────────────────────

/// What the carrier does to what it carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tamper {
    None,
    /// One bit of the quote's MRTD flipped in the key document.
    FlipQuote,
    /// Its own OHTTP key in the document, next to the TD's quote.
    OwnKey,
    /// Another receipt key in the document.
    OtherReceiptKey,
    /// One bit of the sealed request flipped.
    FlipRequest,
    /// One bit of the sealed answer flipped.
    FlipResponse,
    /// The GPU evidence dropped from the key document, NRAS's verdict on it
    /// with it: a CPU document as far as its fields go.
    DropGpu,
}

/// One recorded byte stream: a direction of a TCP connection, or one
/// exchange the carrier had in clear with the TD.
type Streams = Arc<Mutex<Vec<(String, Arc<Mutex<Vec<u8>>>)>>>;

fn stream(seen: &Streams, label: String) -> Arc<Mutex<Vec<u8>>> {
    let bytes = Arc::new(Mutex::new(Vec::new()));
    seen.lock().unwrap().push((label, bytes.clone()));
    bytes
}

/// Everything recorded so far, one buffer per stream.
fn recorded(seen: &Streams) -> Vec<(String, Vec<u8>)> {
    seen.lock()
        .unwrap()
        .iter()
        .map(|(l, b)| (l.clone(), b.lock().unwrap().clone()))
        .collect()
}

async fn pump(
    mut from: tokio::net::tcp::OwnedReadHalf,
    mut to: tokio::net::tcp::OwnedWriteHalf,
    into: Arc<Mutex<Vec<u8>>>,
) {
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = match from.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        into.lock().unwrap().extend_from_slice(&buf[..n]);
        if to.write_all(&buf[..n]).await.is_err() {
            break;
        }
    }
    let _ = to.shutdown().await;
}

/// A TCP relay to `upstream` that records both directions of every
/// connection under `name`. Its address.
async fn tap(upstream: String, seen: Streams, name: &'static str) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    tokio::spawn(async move {
        let mut n = 0;
        while let Ok((client, _)) = listener.accept().await {
            n += 1;
            let Ok(server) = tokio::net::TcpStream::connect(&upstream).await else {
                continue;
            };
            let up = stream(&seen, format!("{name} #{n} up"));
            let down = stream(&seen, format!("{name} #{n} down"));
            let ((cr, cw), (sr, sw)) = (client.into_split(), server.into_split());
            tokio::spawn(pump(cr, sw, up));
            tokio::spawn(pump(sr, cw, down));
        }
    });
    addr
}

struct CarrierState {
    /// The TD, through the tap that records the TLS bytes.
    td: String,
    tls: Arc<TlsIdentity>,
    pcs: String,
    /// Where the carrier fetches Intel's root CA CRL (the relay's
    /// `attest::ROOT_CA_CRL_URL`, here the fake PCS's).
    root_crl: String,
    /// The mirror of Google's bucket the endorsement URL is rebased on.
    bucket: String,
    /// NRAS: where the carrier POSTs what the caller asks, and fetches the JWKS.
    nras: String,
    seen: Streams,
    tamper: Mutex<Tamper>,
    collateral: tokio::sync::Mutex<Option<Collateral>>,
}

/// The carrier: plain HTTP towards the caller, mutual TLS towards the TD
/// (one connection per request: one quote each), Intel's collateral relayed
/// from PCS. It reads all it carries; `seen` holds all of it.
struct Carrier {
    url: String,
    seen: Streams,
    state: Arc<CarrierState>,
}

impl Carrier {
    async fn start(lab: &Lab, td: &Td) -> Self {
        let seen: Streams = Arc::default();
        let td_tap = tap(td.endpoint.clone(), seen.clone(), "carrier-td TLS").await;
        let state = Arc::new(CarrierState {
            td: td_tap,
            tls: td.carrier_tls.clone(),
            pcs: lab.pcs.base_url.clone(),
            root_crl: lab.pcs.root_crl_url.clone(),
            bucket: lab.bucket.clone(),
            nras: lab.nras.clone(),
            seen: seen.clone(),
            tamper: Mutex::new(Tamper::None),
            collateral: tokio::sync::Mutex::new(None),
        });
        let app = Router::new()
            .route("/v2/e2e/key", get(carried_key))
            .route("/v2/e2e/collateral", get(carried_collateral))
            .route("/v2/e2e/nras", post(carried_nras))
            .route("/v2/e2e/chat", post(carried_chat))
            .route("/v1/chat/completions", post(carried_clear_chat))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let carrier = listener.local_addr().unwrap().to_string();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        let front = tap(carrier, seen.clone(), "caller-carrier HTTP").await;
        Self {
            url: format!("http://{front}"),
            seen,
            state,
        }
    }

    fn tamper(&self, t: Tamper) {
        *self.state.tamper.lock().unwrap() = t;
    }
}

/// The request to the TD, in clear as the carrier holds it, recorded with
/// the answer.
async fn forward(
    c: &CarrierState,
    method: &str,
    uri: &str,
    ctype: Option<&str>,
    body: Vec<u8>,
) -> Reply {
    let sent = stream(&c.seen, format!("carrier->td clear {method} {uri}"));
    {
        let mut s = sent.lock().unwrap();
        s.extend_from_slice(
            format!("{method} {uri}\r\ncontent-type: {ctype:?}\r\n\r\n").as_bytes(),
        );
        s.extend_from_slice(&body);
    }
    let mut send = mtls(&c.td, &c.tls).await;
    let reply = request(&mut send, method, uri, ctype, body).await;
    let got = stream(&c.seen, format!("td->carrier clear {method} {uri}"));
    {
        let mut g = got.lock().unwrap();
        g.extend_from_slice(format!("{}\r\n", reply.status).as_bytes());
        for (k, v) in &reply.headers {
            g.extend_from_slice(k.as_str().as_bytes());
            g.extend_from_slice(b": ");
            g.extend_from_slice(v.as_bytes());
            g.extend_from_slice(b"\r\n");
        }
        g.extend_from_slice(b"\r\n");
        g.extend_from_slice(&reply.body);
    }
    reply
}

fn relay(reply: Reply, body: Vec<u8>) -> Response {
    let ct = reply.content_type().to_string();
    (
        StatusCode::from_u16(reply.status).unwrap(),
        [(header::CONTENT_TYPE, ct)],
        body,
    )
        .into_response()
}

async fn carried_key(State(c): State<Arc<CarrierState>>, RawQuery(q): RawQuery) -> Response {
    let uri = format!("/v2/e2e/key?{}", q.unwrap_or_default());
    let reply = forward(&c, "GET", &uri, None, vec![]).await;
    let tamper = *c.tamper.lock().unwrap();
    if reply.status != 200 || tamper == Tamper::None {
        let body = reply.body.clone();
        return relay(reply, body);
    }
    let mut doc: KeyDocument = serde_json::from_slice(&reply.body).unwrap();
    match tamper {
        Tamper::FlipQuote => {
            let mut quote = unb64(&doc.quote_b64).unwrap();
            // The TD report's MRTD: header 48 bytes, then 136 into the body.
            quote[48 + 136] ^= 0x01;
            doc.quote_b64 = b64(&quote);
        }
        Tamper::OwnKey => doc.key_config_b64 = b64(&Gateway::from_seed(&[0xCA; 32]).key_config()),
        Tamper::OtherReceiptKey => {
            doc.session_pubkey_hex = hex(&SessionIdentity::generate([0xCB; 32], [0; 32])
                .public_key()
                .to_bytes());
        }
        Tamper::DropGpu => {
            doc.gpu_evidence_list = None;
            doc.nras_answer = None;
            doc.nras_jwks = None;
        }
        _ => {}
    }
    let body = serde_json::to_vec(&doc).unwrap();
    relay(reply, body)
}

/// Intel's collateral for the TD's platform, from PCS: the FMSPC and the
/// CRL's CA read from the quote of the TD's attestation document; Intel's
/// root CA CRL; Google's endorsement of the quote's MRTD, from the URL the
/// SDK names (`attest::gce_endorsement_url`) on the bucket's mirror, a 404
/// relayed empty as the SDK expects; the firmware binary that endorsement
/// names (`attest::gce_firmware_url`), the same way.
async fn carried_collateral(State(c): State<Arc<CarrierState>>) -> Json<Collateral> {
    let mut cached = c.collateral.lock().await;
    if let Some(col) = &*cached {
        return Json(col.clone());
    }
    let doc = forward(&c, "GET", "/v2/attestation", None, vec![]).await;
    let doc: Value = serde_json::from_slice(&doc.body).unwrap();
    let quote = unb64(doc["tdx"]["quote_b64"].as_str().unwrap()).unwrap();
    let fmspc = attest::fmspc_hex(&quote).unwrap();
    let ca = attest::pck_crl_ca(&quote).unwrap();
    let fetch = crate::nras_http::fetch_pcs_collateral;
    let (tcb_info, tcb_info_chain_pem) = fetch(&crate::tcb::pcs_tcb_url(&c.pcs, &fmspc))
        .await
        .unwrap();
    let (qe_identity, qe_identity_chain_pem) = fetch(&crate::tcb::pcs_qe_identity_url(&c.pcs))
        .await
        .unwrap();
    let standard = crate::tcb::PcsUpdate::Standard;
    let (tcb_info_standard, tcb_info_standard_chain_pem) =
        fetch(&crate::tcb::pcs_tcb_url_for(&c.pcs, &fmspc, standard))
            .await
            .unwrap();
    let (qe_identity_standard, qe_identity_standard_chain_pem) =
        fetch(&crate::tcb::pcs_qe_identity_url_for(&c.pcs, standard))
            .await
            .unwrap();
    let crl = crate::nras_http::fetch_pcs_der(&crate::tcb::pcs_pck_crl_url(&c.pcs, ca))
        .await
        .unwrap();
    let root_crl = crate::nras_http::fetch_pcs_der(&c.root_crl).await.unwrap();
    let mirrored = attest::gce_endorsement_url(&quote).unwrap();
    let http = reqwest::Client::builder().no_proxy().build().unwrap();
    let from_bucket = |url: String| {
        let http = http.clone();
        let bucket = c.bucket.clone();
        async move {
            let mirrored = url
                .strip_prefix(crate::gce_endorsement::BUCKET_BASE_URL)
                .map(|path| format!("{bucket}{path}"))
                .expect("the URL is Google's bucket");
            let resp = http.get(&mirrored).send().await.unwrap();
            match resp.status().as_u16() {
                200 => resp.bytes().await.unwrap().to_vec(),
                404 => Vec::new(),
                other => panic!("bucket answered {other}"),
            }
        }
    };
    let endorsement = from_bucket(mirrored).await;
    let firmware = if endorsement.is_empty() {
        Vec::new()
    } else {
        from_bucket(attest::gce_firmware_url(&endorsement).unwrap()).await
    };
    let jwks = http
        .get(format!("{}/.well-known/jwks.json", c.nras))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let col = Collateral {
        tcb_info,
        tcb_info_chain_pem,
        qe_identity,
        qe_identity_chain_pem,
        tcb_info_standard,
        tcb_info_standard_chain_pem,
        qe_identity_standard,
        qe_identity_standard_chain_pem,
        pck_crl_der_b64: b64(&crl),
        root_ca_crl_der_b64: b64(&root_crl),
        gce_endorsement_b64: b64(&endorsement),
        gce_firmware_b64: b64(&firmware),
        nras_answer: String::new(),
        nras_jwks: jwks,
    };
    *cached = Some(col.clone());
    Json(col)
}

/// The caller's NRAS request (`attest::nras_request`), POSTed to NRAS as it
/// came, NRAS's answer returned as it came: NVIDIA signs it, the caller
/// checks it. Recorded, as all the carrier moves.
async fn carried_nras(State(c): State<Arc<CarrierState>>, body: Bytes) -> Response {
    stream(&c.seen, "carrier->nras".into())
        .lock()
        .unwrap()
        .extend_from_slice(&body);
    let answer = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap()
        .post(format!("{}/v4/attest/gpu", c.nras))
        .header("content-type", "application/json")
        .body(body.to_vec())
        .send()
        .await
        .unwrap();
    let status = StatusCode::from_u16(answer.status().as_u16()).unwrap();
    let body = answer.bytes().await.unwrap().to_vec();
    stream(&c.seen, "nras->carrier".into())
        .lock()
        .unwrap()
        .extend_from_slice(&body);
    (status, [(header::CONTENT_TYPE, "application/json")], body).into_response()
}

async fn carried_chat(
    State(c): State<Arc<CarrierState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let ctype = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let tamper = *c.tamper.lock().unwrap();
    let mut body = body.to_vec();
    if tamper == Tamper::FlipRequest && body.len() > 32 {
        // Inside the AEAD ciphertext, past the header and the encapsulated key.
        let i = body.len() - 32;
        body[i] ^= 0x01;
    }
    let reply = forward(&c, "POST", "/v2/e2e/chat", ctype.as_deref(), body).await;
    let mut out = reply.body.clone();
    if tamper == Tamper::FlipResponse && reply.status == 200 {
        let i = out.len() / 2;
        out[i] ^= 0x01;
    }
    relay(reply, out)
}

/// The control: the plain chat route, carried the same way.
async fn carried_clear_chat(State(c): State<Arc<CarrierState>>, body: Bytes) -> Response {
    let reply = forward(
        &c,
        "POST",
        "/v1/chat/completions",
        Some("application/json"),
        body.to_vec(),
    )
    .await;
    let out = reply.body.clone();
    relay(reply, out)
}

// ── The caller ───────────────────────────────────────────────────────────────

struct Caller {
    url: String,
    http: reqwest::Client,
}

impl Caller {
    fn of(carrier: &Carrier) -> Self {
        Self {
            url: carrier.url.clone(),
            http: reqwest::Client::builder().no_proxy().build().unwrap(),
        }
    }

    async fn document(&self, nonce: &[u8; 32]) -> KeyDocument {
        let resp = self
            .http
            .get(format!("{}/v2/e2e/key?nonce={}", self.url, hex(nonce)))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        resp.json().await.unwrap()
    }

    async fn collateral(&self) -> Collateral {
        self.http
            .get(format!("{}/v2/e2e/collateral", self.url))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    /// NRAS's answer to `doc`, through the carrier: the body as it came.
    async fn nras(&self, doc: &KeyDocument, nonce: &[u8; 32]) -> String {
        let Ok(request) = attest::nras_request(doc, nonce) else {
            return String::new();
        };
        let resp = self
            .http
            .post(format!("{}/v2/e2e/nras", self.url))
            .json(&request)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        resp.text().await.unwrap()
    }

    /// Fetch, verify: the SDK's client, or the step it refused at. A key
    /// document with GPU evidence carries NRAS's verdict; one that does not
    /// (a TD before 2026-10-28) gets it through the carrier.
    async fn verified(&self, policy: &Policy) -> Result<Client, attest::Refusal> {
        let nonce = test_random();
        let doc = self.document(&nonce).await;
        let mut col = self.collateral().await;
        if doc.nras_answer.is_none() {
            col.nras_answer = self.nras(&doc, &nonce).await;
        }
        attest::verify_key(&doc, &col, policy, &nonce, now()).map(Client::new)
    }

    /// POST the sealed request: status, body, how long it took.
    async fn send(&self, request: Vec<u8>, ctype: &str) -> (u16, Vec<u8>, Duration) {
        let t = Instant::now();
        let resp = self
            .http
            .post(format!("{}/v2/e2e/chat", self.url))
            .header("content-type", ctype)
            .body(request)
            .send()
            .await
            .unwrap();
        let status = resp.status().as_u16();
        let body = resp.bytes().await.unwrap().to_vec();
        (status, body, t.elapsed())
    }

    // Async like the other requests of Caller: every call site awaits it.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    async fn sealed(&self, client: &Client, text: &str, sent_at: u64) -> (Vec<u8>, Pending) {
        client
            .seal_chat(&question(text), &test_random(), sent_at)
            .unwrap()
    }
}

/// No recorded stream holds a byte of `needles`; and there were streams.
fn assert_carried_blind(seen: &Streams, needles: &[&[u8]]) {
    let streams = recorded(seen);
    assert!(streams.len() >= 4, "nothing was recorded: {streams:?}");
    for (label, bytes) in &streams {
        for needle in needles {
            assert!(
                !contains(bytes, needle),
                "{label} carried {:?} in clear",
                String::from_utf8_lossy(needle)
            );
        }
    }
}

// ── The accepting path ───────────────────────────────────────────────────────

/// SECURITY, (a) (b) (d): the SDK verifies the TD's key document through the
/// carrier, seals the question, and opens the model's answer, whose receipt
/// it checks under the key the quote bound. Neither the question nor the
/// answer is in any byte the carrier moved or held in clear; the same
/// recorder sees both in the plain route's traffic.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sealed_question_crosses_the_carrier_and_its_answer_and_receipt_check() {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = ready_td(&lab, &runs).await;
    let carrier = Carrier::start(&lab, &td).await;
    let caller = Caller::of(&carrier);
    let client = caller
        .verified(&lab.policy(&lab.honest()))
        .await
        .expect("the honest TD's key verifies");
    assert_eq!(client.key().tcb_platform, "UpToDate");
    assert_eq!(
        client.key().session_pubkey(),
        &td.state.identity.public_key().to_bytes(),
        "the quote binds the TD's receipt key"
    );
    // Before sealing: the session config and the boot binding the quote
    // bound, the TD's own (the model, its end, the relay's TLS key).
    let key = client.key();
    assert_eq!(
        serde_json::to_value(&key.session_config).unwrap(),
        serde_json::to_value(&td.state.config.config).unwrap()
    );
    assert_eq!(key.session_config.model, MODEL);
    assert_eq!(
        key.session_config.client_spki_sha256,
        hex(&td.carrier_tls.spki_sha256())
    );
    assert_eq!(
        key.session_config_sha256_hex,
        hex(&td.state.binding.config_sha256)
    );
    assert_eq!(key.boot_nonce_hex, hex(&td.state.binding.session_nonce));
    assert_eq!(
        key.tls_spki_sha256_hex,
        hex(&td.state.binding.tls_spki_sha256)
    );
    let ran = runs.load(SeqCst);
    let (request, pending) = caller.sealed(&client, QUESTION, now()).await;
    let (status, response, _) = caller.send(request, ohttp::MEDIA_TYPE_REQUEST).await;
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&response));
    assert_eq!(runs.load(SeqCst), ran + 1, "the model ran once");

    // (a) The model's answer, as the TD's chat handler gave it.
    let digest = catalog_digest();
    let (answer, checked) = pending
        .open_verified(&response, &digest, now())
        .expect("the answer opens and its receipt checks");
    assert_eq!(answer.status, 200);
    let shown: Value = serde_json::from_slice(&answer.body).unwrap();
    assert_eq!(shown["choices"][0]["message"]["content"], ANSWER);
    assert_eq!(shown["model"], MODEL);

    // (d) The receipt: signed by the bound key over the text shown, for the
    // pinned model; under another key or another pin it does not check.
    let checked = checked.expect("a 200 carries a receipt");
    assert_eq!(checked.text, ANSWER);
    assert_eq!(checked.model_digest_hex, digest);
    assert_eq!(checked.compute_device, "cpu");
    let bound = client.key().session_pubkey();
    assert!(x0ne_sdk::receipt::check_answer(&answer.body, bound, Some(&digest)).is_ok());
    let other = SessionIdentity::generate([0xCC; 32], [0; 32])
        .public_key()
        .to_bytes();
    assert!(x0ne_sdk::receipt::check_answer(&answer.body, &other, None).is_err());
    assert!(x0ne_sdk::receipt::check_answer(&answer.body, bound, Some(&"00".repeat(32))).is_err());

    // (b) What the carrier moved and held: no word of either.
    assert_carried_blind(&carrier.seen, &[QUESTION_NEEDLE, ANSWER_NEEDLE]);
    let e2e = &td.state.e2e.counts;
    assert_eq!((e2e.opened.load(SeqCst), e2e.sealed.load(SeqCst)), (1, 1));
    assert_eq!(td.state.stats.chat_posts.load(SeqCst), 0, "no clear chat");
    assert_eq!(td.state.stats.e2e_key_quotes.load(SeqCst), 1);

    // The control: the same recorder, the plain route, the same question.
    let clear = Carrier::start(&lab, &td).await;
    let resp = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap()
        .post(format!("{}/v1/chat/completions", clear.url))
        .header("content-type", "application/json")
        .body(question(QUESTION))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let streams = recorded(&clear.seen);
    for needle in [QUESTION_NEEDLE, ANSWER_NEEDLE] {
        assert!(
            streams.iter().any(|(_, b)| contains(b, needle)),
            "the recorder misses {:?} in clear",
            String::from_utf8_lossy(needle)
        );
    }
    // TLS hides them from the wire, not from the carrier that holds the key.
    for (label, bytes) in &streams {
        if label.starts_with("carrier-td TLS") {
            assert!(!contains(bytes, QUESTION_NEEDLE), "{label}");
        }
    }
}

// ── (c) The caller's refusals ────────────────────────────────────────────────

/// SECURITY (c): a carrier that alters the proof, puts its own key or another
/// receipt key in the document is refused before anything is sealed; so is
/// a TD whose measurement is not the pinned one, and a pin that is not the
/// TD's. Each refusal names its step.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_caller_refuses_an_altered_proof_key_or_measurement() {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = ready_td(&lab, &runs).await;
    let carrier = Carrier::start(&lab, &td).await;
    let caller = Caller::of(&carrier);
    let policy = lab.policy(&lab.honest());
    for (tamper, step) in [
        (Tamper::FlipQuote, "quote"),
        (Tamper::OwnKey, "binding"),
        (Tamper::OtherReceiptKey, "binding"),
    ] {
        carrier.tamper(tamper);
        match caller.verified(&policy).await {
            Ok(_) => panic!("{tamper:?}: accepted"),
            Err(r) => assert_eq!(r.step, step, "{tamper:?}: {r:?}"),
        }
    }
    carrier.tamper(Tamper::None);
    let nonce = test_random();
    let doc = caller.document(&nonce).await;
    let col = caller.collateral().await;
    assert!(attest::verify_key(&doc, &col, &policy, &nonce, now()).is_ok());
    assert!(attest::verify_key(&doc, &col, &lab.reference_only(), &nonce, now()).is_ok());
    let mut pin = lab.policy(&lab.honest());
    pin.mrtd_pin.as_mut().unwrap()[0] ^= 1;
    assert_eq!(
        attest::verify_key(&doc, &col, &pin, &nonce, now())
            .unwrap_err()
            .step,
        "measurement"
    );
    for i in 0..3 {
        let mut pin = lab.policy(&lab.honest());
        pin.rtmr_pins[i].as_mut().unwrap()[47] ^= 1;
        let e = attest::verify_key(&doc, &col, &pin, &nonce, now()).unwrap_err();
        assert_eq!(e.step, "measurement", "RTMR{}", i + 1);
    }
    let late = now() + 2 * 86_400;
    assert!(attest::verify_key(&doc, &col, &policy, &nonce, late).is_err());
    assert_eq!(
        attest::verify_key(&doc, &Collateral::default(), &policy, &nonce, now())
            .unwrap_err()
            .step,
        "tcb"
    );
    assert_eq!(
        td.state.stats.e2e_chat_posts.load(SeqCst),
        0,
        "nothing sealed"
    );

    // Another image (an RTMR2 event the pins do not list) on the same Intel.
    let mut other = lab.honest();
    other.events.push(Event::new_sha384(
        2,
        EV_IPL,
        [0x42; 48],
        b"another kernel".to_vec(),
    ));
    let runs = Arc::new(AtomicUsize::new(0));
    let other_td = start_td(
        &lab,
        Spec {
            td: other,
            gpu: false,
            model: scripted(&runs),
            max_completion_tokens: 64,
            quote_limits: cloud_serve::QuoteLimits::for_queue(8),
        },
    )
    .await;
    release_and_wait(&other_td).await;
    let other_carrier = Carrier::start(&lab, &other_td).await;
    let e = Caller::of(&other_carrier)
        .verified(&policy)
        .await
        .err()
        .expect("another image is refused");
    assert_eq!(e.step, "measurement", "{e:?}");
}

/// SECURITY: the key document carries the TD's event log, the bytes its
/// attestation document carries for the relay, which the SDK replays into
/// the quote's RTMRs and judges against the image reference. Withheld, or
/// one event more, it is refused at `measurement`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_key_document_carries_the_event_log_the_relay_reads() {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = ready_td(&lab, &runs).await;
    let carrier = Carrier::start(&lab, &td).await;
    let caller = Caller::of(&carrier);
    let nonce = test_random();
    let doc = caller.document(&nonce).await;
    let attested = forward(&carrier.state, "GET", "/v2/attestation", None, vec![]).await;
    let attested: Value = serde_json::from_slice(&attested.body).unwrap();
    assert_eq!(
        doc.ccel_b64,
        attested["tdx"]["ccel_b64"].as_str().unwrap(),
        "the log the relay reads"
    );
    let events = lab.honest().events;
    assert_eq!(unb64(&doc.ccel_b64).unwrap(), crate::ccel::encode(&events));
    let col = caller.collateral().await;
    let policy = lab.reference_only();
    assert!(attest::verify_key(&doc, &col, &policy, &nonce, now()).is_ok());
    let mut withheld = doc.clone();
    withheld.ccel_b64 = String::new();
    let mut longer = events.clone();
    longer.push(Event::new_sha384(
        2,
        EV_IPL,
        [0x42; 48],
        b"another kernel".to_vec(),
    ));
    let mut other = doc.clone();
    other.ccel_b64 = b64(&crate::ccel::encode(&longer));
    for (what, d) in [("withheld", withheld), ("one event more", other)] {
        let e = attest::verify_key(&d, &col, &policy, &nonce, now()).unwrap_err();
        assert_eq!(e.step, "measurement", "{what}: {e:?}");
    }
}

/// SECURITY (c): a carrier that flips a bit of the sealed request, replays
/// it, sends it under another type, or one sealed an hour ago, gets one
/// answer from the TD: 400, the same body, after the same second; the
/// model never runs. The request it altered still opens afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_altered_replayed_or_stale_sealed_request_gets_one_uniform_refusal() {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = ready_td(&lab, &runs).await;
    let carrier = Carrier::start(&lab, &td).await;
    let caller = Caller::of(&carrier);
    let client = caller.verified(&lab.policy(&lab.honest())).await.unwrap();
    let ran = runs.load(SeqCst);
    let (request, pending) = caller.sealed(&client, QUESTION, now()).await;
    let mut bodies = Vec::new();

    carrier.tamper(Tamper::FlipRequest);
    let (status, body, took) = caller
        .send(request.clone(), ohttp::MEDIA_TYPE_REQUEST)
        .await;
    assert_eq!(status, 400);
    assert_cadence(took, "flipped");
    bodies.push(body);
    carrier.tamper(Tamper::None);

    let (status, response, _) = caller
        .send(request.clone(), ohttp::MEDIA_TYPE_REQUEST)
        .await;
    assert_eq!(status, 200, "the flip used nothing up");
    assert!(pending
        .open_verified(&response, &catalog_digest(), now())
        .is_ok());

    let (status, body, took) = caller
        .send(request.clone(), ohttp::MEDIA_TYPE_REQUEST)
        .await;
    assert_eq!(status, 400, "replayed");
    assert_cadence(took, "replayed");
    bodies.push(body);

    let (other, _) = caller.sealed(&client, QUESTION, now()).await;
    let (status, body, took) = caller.send(other, "application/json").await;
    assert_eq!(status, 400, "another type");
    assert_cadence(took, "another type");
    bodies.push(body);

    let (stale, _) = caller.sealed(&client, QUESTION, now() - 3600).await;
    let (status, body, took) = caller.send(stale, ohttp::MEDIA_TYPE_REQUEST).await;
    assert_eq!(status, 400, "stale");
    assert_cadence(took, "stale");
    bodies.push(body);

    let (ahead, _) = caller.sealed(&client, QUESTION, now() + 3600).await;
    let (status, body, took) = caller.send(ahead, ohttp::MEDIA_TYPE_REQUEST).await;
    assert_eq!(status, 400, "dated an hour ahead");
    assert_cadence(took, "dated an hour ahead");
    bodies.push(body);

    let expected = serde_json::to_vec(&json!({ "error": REFUSED })).unwrap();
    for body in &bodies {
        assert_eq!(body, &expected, "{}", String::from_utf8_lossy(body));
    }
    assert_eq!(runs.load(SeqCst), ran + 1, "only the honest request ran");
    let e2e = &td.state.e2e.counts;
    assert_eq!(e2e.refused_open.load(SeqCst), 1);
    assert_eq!(e2e.refused_replay.load(SeqCst), 3);
    assert_carried_blind(&carrier.seen, &[QUESTION_NEEDLE, ANSWER_NEEDLE]);
}

/// SECURITY: the host sets the TD's wall clock. It lets a sealed request be
/// answered, moves the TD's clock 130 s on, where an honest request answered
/// makes the gateway drop the first one's entry, then sets the clock back
/// inside the first request's window and replays it. Until 2026-10-30 the
/// model answered it a second time: one more size and timing sample for the
/// host. Now the route's one refusal, and the model does not run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_request_replayed_after_the_host_set_the_tds_clock_back_gets_no_answer() {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = ready_td(&lab, &runs).await;
    let carrier = Carrier::start(&lab, &td).await;
    let caller = Caller::of(&carrier);
    let client = caller.verified(&lab.policy(&lab.honest())).await.unwrap();
    let ran = runs.load(SeqCst);
    let (request, _) = caller.sealed(&client, QUESTION, now()).await;
    let (status, _, _) = caller
        .send(request.clone(), ohttp::MEDIA_TYPE_REQUEST)
        .await;
    assert_eq!(status, 200);

    td.clock_skew.store(130, SeqCst);
    let (later, _) = caller.sealed(&client, QUESTION, now() + 130).await;
    let (status, _, _) = caller.send(later, ohttp::MEDIA_TYPE_REQUEST).await;
    assert_eq!(status, 200, "a request dated at the TD's new time");

    td.clock_skew.store(10, SeqCst);
    let (status, body, took) = caller.send(request, ohttp::MEDIA_TYPE_REQUEST).await;
    assert_eq!(status, 400, "replayed with the TD's clock set back");
    assert_cadence(took, "replayed with the clock set back");
    assert_eq!(
        body,
        serde_json::to_vec(&json!({ "error": REFUSED })).unwrap()
    );
    assert_eq!(runs.load(SeqCst), ran + 2, "the two honest requests only");
    let e2e = &td.state.e2e.counts;
    assert_eq!(e2e.refused_replay.load(SeqCst), 1);
    assert_eq!(e2e.opened.load(SeqCst), 2);
}

/// SECURITY: requests sealed to the TD's key, so
/// past the AEAD, that are not a chat request the gateway forwards: each gets
/// the route's one refusal, at the floor and no later, and the model does not
/// run. Then an honest request is answered.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn opened_requests_that_are_no_chat_get_the_one_refusal_at_the_floor() {
    use x0ne_sdk::bhttp;
    use x0ne_sdk::e2e::{chat_request, http_date, padded_len, CHAT_PATH, SENT_AT_HEADER};
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = ready_td(&lab, &runs).await;
    let ran = runs.load(SeqCst);
    let config = ohttp::KeyConfig::parse(&td.state.e2e.key_config()).unwrap();
    let chat = question(QUESTION);
    let json = || ("content-type".to_string(), "application/json".to_string());
    let date = || (SENT_AT_HEADER.to_string(), http_date(now()));
    let inner =
        |method: &str, path: &str, headers: Vec<(String, String)>, content: &[u8]| bhttp::Request {
            method: method.into(),
            scheme: "https".into(),
            authority: String::new(),
            path: path.into(),
            headers,
            content: content.to_vec(),
        };
    let padded = |req: bhttp::Request| req.encode(padded_len(req.encode(0).len()).unwrap());
    let honest = chat_request(&chat, now()).unwrap();
    let mut framing = honest.clone();
    framing[0] = 1;
    let mut trailer = honest.clone();
    *trailer.last_mut().unwrap() = 1;
    let mut noise = vec![0u8; x0ne_sdk::e2e::MIN_PADDED];
    for (i, b) in noise.iter_mut().enumerate() {
        *b = (i.wrapping_mul(2_654_435_761) >> 11) as u8 | 0x40;
    }
    let hostile: Vec<(&str, Vec<u8>)> = vec![
        ("not Binary HTTP", noise),
        ("another framing", framing),
        ("padding not zero", trailer),
        (
            "unpadded",
            inner("POST", CHAT_PATH, vec![json(), date()], &chat).encode(0),
        ),
        (
            "GET",
            padded(inner("GET", CHAT_PATH, vec![json(), date()], &chat)),
        ),
        (
            "another path",
            padded(inner("POST", "/v1/models", vec![json(), date()], &chat)),
        ),
        (
            "text/plain",
            padded(inner(
                "POST",
                CHAT_PATH,
                vec![("content-type".into(), "text/plain".into()), date()],
                &chat,
            )),
        ),
        (
            "no content-type",
            padded(inner("POST", CHAT_PATH, vec![date()], &chat)),
        ),
        (
            "stream",
            padded(inner(
                "POST",
                CHAT_PATH,
                vec![json(), date()],
                br#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#,
            )),
        ),
        (
            "not JSON",
            padded(inner("POST", CHAT_PATH, vec![json(), date()], b"Perrache?")),
        ),
        (
            "an array",
            padded(inner("POST", CHAT_PATH, vec![json(), date()], b"[1]")),
        ),
        (
            "no date",
            padded(inner("POST", CHAT_PATH, vec![json()], &chat)),
        ),
        (
            "two dates",
            padded(inner(
                "POST",
                CHAT_PATH,
                vec![json(), date(), date()],
                &chat,
            )),
        ),
        (
            "not an IMF-fixdate",
            padded(inner(
                "POST",
                CHAT_PATH,
                vec![json(), (SENT_AT_HEADER.into(), "yesterday".into())],
                &chat,
            )),
        ),
    ];
    let expected = serde_json::to_vec(&json!({ "error": REFUSED })).unwrap();
    for (what, plaintext) in &hostile {
        let (sealed, _) = ohttp::encapsulate_request(&config, plaintext, &test_random()).unwrap();
        let t = Instant::now();
        let r = call(
            &td,
            "POST",
            "/v2/e2e/chat",
            Some(ohttp::MEDIA_TYPE_REQUEST),
            sealed,
        )
        .await;
        assert_eq!(r.status, 400, "{what}");
        assert_eq!(r.body, expected, "{what}");
        assert_cadence(t.elapsed(), what);
    }
    let counts = &td.state.e2e.counts;
    assert_eq!(counts.refused_open.load(SeqCst), hostile.len() as u64);
    assert_eq!(counts.opened.load(SeqCst), 0);
    assert_eq!(runs.load(SeqCst), ran, "the model did not run");
    let (sealed, _) = ohttp::encapsulate_request(&config, &honest, &test_random()).unwrap();
    let r = call(
        &td,
        "POST",
        "/v2/e2e/chat",
        Some(ohttp::MEDIA_TYPE_REQUEST),
        sealed,
    )
    .await;
    assert_eq!(r.status, 200);
    assert_eq!(runs.load(SeqCst), ran + 1);
}

/// SECURITY (c): an answer with one bit flipped, or another request's
/// answer, does not open: the caller is told nothing it did not get sealed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_caller_refuses_an_altered_or_swapped_answer() {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = ready_td(&lab, &runs).await;
    let carrier = Carrier::start(&lab, &td).await;
    let caller = Caller::of(&carrier);
    let client = caller.verified(&lab.policy(&lab.honest())).await.unwrap();

    carrier.tamper(Tamper::FlipResponse);
    let (request, pending) = caller.sealed(&client, QUESTION, now()).await;
    let (status, response, _) = caller.send(request, ohttp::MEDIA_TYPE_REQUEST).await;
    assert_eq!(status, 200);
    assert!(pending.open(&response).is_err(), "a flipped answer opened");
    carrier.tamper(Tamper::None);

    let (a, pending_a) = caller.sealed(&client, QUESTION, now()).await;
    let (b, pending_b) = caller.sealed(&client, QUESTION, now()).await;
    let (_, answer_a, _) = caller.send(a, ohttp::MEDIA_TYPE_REQUEST).await;
    let (_, answer_b, _) = caller.send(b, ohttp::MEDIA_TYPE_REQUEST).await;
    assert!(
        pending_a.open(&answer_b).is_err(),
        "B's answer opened as A's"
    );
    assert!(
        pending_b.open(&answer_a).is_err(),
        "A's answer opened as B's"
    );
    assert!(pending_a.open(&answer_a).is_ok() && pending_b.open(&answer_b).is_ok());
}

/// SECURITY: a model that panics on a question,
/// in its template before its turn or in its generation during it, answers
/// as a failed answer does: 200 sealed, an error padded to an answer's size
/// inside, after the floor of a whole answer; never a 500 in clear at the
/// instant of the panic, which would tell the carrier that this question
/// broke the model, and when. The TD answers the next question.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_panic_of_the_model_leaves_sealed_padded_and_at_the_cadence() {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = ready_td(&lab, &runs).await;
    let carrier = Carrier::start(&lab, &td).await;
    let caller = Caller::of(&carrier);
    let client = caller.verified(&lab.policy(&lab.honest())).await.unwrap();
    let floor = floor();
    let (honest, _) = caller.sealed(&client, QUESTION, now()).await;
    let (status, reference, _) = caller.send(honest, ohttp::MEDIA_TYPE_REQUEST).await;
    assert_eq!(status, 200);

    for marker in [PANIC_IN_TEMPLATE, PANIC_IN_GENERATION] {
        let (request, pending) = caller.sealed(&client, marker, now()).await;
        let (status, response, took) = caller.send(request, ohttp::MEDIA_TYPE_REQUEST).await;
        assert_eq!(
            status,
            200,
            "{marker}: {}",
            String::from_utf8_lossy(&response)
        );
        assert!(took >= floor, "{marker}: left after {took:?}");
        assert_eq!(
            response.len(),
            reference.len(),
            "{marker}: an answer's size"
        );
        let answer = pending.open(&response).expect("sealed for the caller");
        assert_eq!(answer.status, 500, "{marker}");
        assert!(
            crate::padding::is_size(answer.body.len() as u64, crate::padding::MIN_WHOLE_BYTES),
            "{marker}: padded"
        );
    }
    let (request, pending) = caller.sealed(&client, QUESTION, now()).await;
    let (status, response, _) = caller.send(request, ohttp::MEDIA_TYPE_REQUEST).await;
    assert_eq!(status, 200);
    let (answer, _) = pending
        .open_verified(&response, &catalog_digest(), now())
        .unwrap();
    assert_eq!(answer.status, 200, "the TD answers after the panics");
    assert_carried_blind(
        &carrier.seen,
        &[QUESTION_NEEDLE, ANSWER_NEEDLE, b"scripted panic"],
    );
}

// ── The TD's side of the routes ──────────────────────────────────────────────

/// The e2e routes answer once the model is ready, take one quote per
/// connection (a session quote counts), and refuse a nonce that is not 64
/// lowercase hex digits.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_routes_wait_for_the_model_and_take_one_quote_per_connection() {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = start_td(
        &lab,
        Spec {
            td: lab.honest(),
            gpu: false,
            model: scripted(&runs),
            max_completion_tokens: 64,
            quote_limits: cloud_serve::QuoteLimits::for_queue(8),
        },
    )
    .await;
    let key = format!("/v2/e2e/key?nonce={}", hex(&[0x51; 32]));
    assert_eq!(call(&td, "GET", &key, None, vec![]).await.status, 503);
    let chat = call(
        &td,
        "POST",
        "/v2/e2e/chat",
        Some(ohttp::MEDIA_TYPE_REQUEST),
        vec![0; 64],
    )
    .await;
    assert_eq!(chat.status, 503);
    release_and_wait(&td).await;

    let mut conn = mtls(&td.endpoint, &td.carrier_tls).await;
    let first = request(&mut conn, "GET", &key, None, vec![]).await;
    assert_eq!(
        first.status,
        200,
        "{}",
        String::from_utf8_lossy(&first.body)
    );
    assert_eq!(first.content_type(), "application/json");
    let second = request(&mut conn, "GET", &key, None, vec![]).await;
    assert_eq!(second.status, 429, "one quote per connection");

    let mut conn = mtls(&td.endpoint, &td.carrier_tls).await;
    let session = request(&mut conn, "GET", "/v2/attestation/session", None, vec![]).await;
    assert_eq!(session.status, 200);
    let after = request(&mut conn, "GET", &key, None, vec![]).await;
    assert_eq!(after.status, 429, "the session quote used the connection's");

    for bad in ["", "?nonce=11", &format!("?nonce={}", "AB".repeat(32))] {
        let r = call(&td, "GET", &format!("/v2/e2e/key{bad}"), None, vec![]).await;
        assert_eq!(r.status, 400, "{bad}");
    }
    assert_eq!(td.state.stats.e2e_key_quotes.load(SeqCst), 1);
}

/// SECURITY: key documents draw on a bucket of
/// their own. Once callers have emptied it, the relay still opens a session
/// on `/v1`: its quote and its chat answer; the e2e refusals are counted
/// apart from the session's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn key_documents_that_empty_their_bucket_leave_v1_sessions_open() {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let limits = cloud_serve::QuoteLimits {
        burst: 3,
        refill_every: Duration::from_secs(3600),
        ..cloud_serve::QuoteLimits::for_queue(8)
    };
    let td = start_td(
        &lab,
        Spec {
            td: lab.honest(),
            gpu: false,
            model: scripted(&runs),
            max_completion_tokens: 64,
            quote_limits: limits,
        },
    )
    .await;
    release_and_wait(&td).await;
    let key = format!("/v2/e2e/key?nonce={}", hex(&[0x52; 32]));
    for n in 0..3 {
        assert_eq!(
            call(&td, "GET", &key, None, vec![]).await.status,
            200,
            "{n}"
        );
    }
    assert_eq!(
        call(&td, "GET", &key, None, vec![]).await.status,
        429,
        "empty"
    );

    let mut conn = mtls(&td.endpoint, &td.carrier_tls).await;
    let session = request(&mut conn, "GET", "/v2/attestation/session", None, vec![]).await;
    assert_eq!(
        session.status,
        200,
        "{}",
        String::from_utf8_lossy(&session.body)
    );
    let chat = request(
        &mut conn,
        "POST",
        "/v1/chat/completions",
        Some("application/json"),
        question(QUESTION),
    )
    .await;
    assert_eq!(chat.status, 200);
    let stats = &td.state.stats;
    assert_eq!(stats.e2e_key_quotes.load(SeqCst), 3);
    assert_eq!(stats.e2e_key_refused.load(SeqCst), 1);
    assert_eq!(stats.session_refused.load(SeqCst), 0);
}

/// SECURITY: the gateway's private scalar is in
/// locked memory, as the model key's recipient key; on Linux, never the heap.
#[test]
fn the_gateway_key_is_in_locked_memory() {
    let config = CloudConfig::new(MODEL, [1; 32], [2; 32], now() + 3600, "cpu", 443, 64)
        .to_loaded()
        .unwrap();
    let boot = BindingV2::of(
        &SessionIdentity::generate(test_random(), test_random()),
        [3; 32],
        config.sha256,
    );
    let e2e = crate::e2e_gateway::E2e::from_seed(&test_random(), boot, &config.bytes)
        .expect("locked memory");
    if cfg!(target_os = "linux") {
        assert_ne!(e2e.key_backing(), crate::harden::Backing::Heap);
    }
}

/// The gateway's documents carry the config bytes its boot binding names,
/// never others: `run_cloud` stops (97) rather than serve a document whose
/// quote no caller would verify.
#[test]
fn the_gateway_carries_only_the_config_its_boot_binding_names() {
    let config = CloudConfig::new(MODEL, [1; 32], [2; 32], now() + 3600, "cpu", 443, 64)
        .to_loaded()
        .unwrap();
    let boot = BindingV2::of(
        &SessionIdentity::generate(test_random(), test_random()),
        [3; 32],
        config.sha256,
    );
    let mut other = config.bytes.clone();
    other.push(b'\n');
    let e = crate::e2e_gateway::E2e::from_seed(&test_random(), boot, &other)
        .err()
        .expect("a config the binding does not name");
    assert!(
        e.to_string().contains("not the one the boot binding names"),
        "{e}"
    );
}

/// SECURITY: the chat route reads the largest
/// request the SDK seals, its 4 MiB bucket, and refuses one byte more as it
/// refuses anything else, before trying to open it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_chat_route_reads_the_largest_sealed_request_and_no_byte_more() {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = ready_td(&lab, &runs).await;
    let carrier = Carrier::start(&lab, &td).await;
    let caller = Caller::of(&carrier);
    let client = caller.verified(&lab.policy(&lab.honest())).await.unwrap();
    let floor = floor();
    let counts = &td.state.e2e.counts;
    let (largest, pending) = client
        .seal_chat(&question(&"a".repeat(3 << 20)), &test_random(), now())
        .unwrap();
    assert_eq!(largest.len(), x0ne_sdk::e2e::MAX_SEALED_REQUEST);
    let ctype = Some(ohttp::MEDIA_TYPE_REQUEST);
    let r = call(&td, "POST", "/v2/e2e/chat", ctype, largest.clone()).await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    assert!(pending.open(&r.body).is_ok());
    assert_eq!(counts.opened.load(SeqCst), 1);

    let mut over = largest;
    over.push(0);
    let t = Instant::now();
    let r = call(&td, "POST", "/v2/e2e/chat", ctype, over).await;
    assert_eq!(r.status, 400);
    assert!(t.elapsed() >= floor, "{:?}", t.elapsed());
    assert_eq!(
        r.body,
        serde_json::to_vec(&json!({ "error": REFUSED })).unwrap()
    );
    assert_eq!(counts.refused_open.load(SeqCst), 0, "not read, not opened");
}

/// SECURITY: a question whose JSON holds a lone
/// surrogate (`\ud800`) is refused by the caller before anything leaves,
/// and, sealed by other code, by the gateway as it opens: the TD opens no
/// request it then cannot read, and the model does not run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lone_surrogate_is_refused_before_sending_and_at_opening() {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = ready_td(&lab, &runs).await;
    let carrier = Carrier::start(&lab, &td).await;
    let caller = Caller::of(&carrier);
    let client = caller.verified(&lab.policy(&lab.honest())).await.unwrap();
    let body = br#"{"messages":[{"role":"user","content":"\ud800"}]}"#;
    assert!(
        client.seal_chat(body, &test_random(), now()).is_err(),
        "the caller sealed a lone surrogate"
    );
    let config = ohttp::KeyConfig::parse(&td.state.e2e.key_config()).unwrap();
    let inner = x0ne_sdk::e2e::chat_request(body, now()).unwrap();
    let (sealed, _) = ohttp::encapsulate_request(&config, &inner, &test_random()).unwrap();
    let ran = runs.load(SeqCst);
    let r = call(
        &td,
        "POST",
        "/v2/e2e/chat",
        Some(ohttp::MEDIA_TYPE_REQUEST),
        sealed,
    )
    .await;
    assert_eq!(r.status, 400);
    assert_eq!(
        r.body,
        serde_json::to_vec(&json!({ "error": REFUSED })).unwrap()
    );
    let counts = &td.state.e2e.counts;
    assert_eq!(counts.opened.load(SeqCst), 0, "refused as it opened");
    assert_eq!(counts.refused_open.load(SeqCst), 1);
    assert_eq!(runs.load(SeqCst), ran);
}

/// A GPU-attested TD on the scripted model, released and loaded.
async fn ready_gpu_td(lab: &Lab, td: FakeTd, runs: &Arc<AtomicUsize>) -> Td {
    let td = start_td(
        lab,
        Spec {
            td,
            gpu: true,
            model: scripted(runs),
            max_completion_tokens: 64,
            quote_limits: cloud_serve::QuoteLimits::for_queue(8),
        },
    )
    .await;
    release_and_wait(&td).await;
    td
}

/// SECURITY: a GPU-attested session, caller to model. The key document
/// carries the H100's report and chain, bound by the quote, and NRAS's
/// verdict on them that the TD fetched for this document's nonce (the
/// relay's per-connection check, here per document); the caller checks it,
/// then seals the question, opens the answer and checks its receipt. The
/// carrier asks NRAS nothing. Nothing it moved holds a word of either. The
/// same document is refused under `attest=cpu`, and without its GPU
/// evidence.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_gpu_session_seals_through_the_carrier_with_its_gpu_attested() {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = ready_gpu_td(&lab, lab.honest(), &runs).await;
    let carrier = Carrier::start(&lab, &td).await;
    let caller = Caller::of(&carrier);
    let asked = lab.nras_answers.load(SeqCst);
    let doc = caller.document(&test_random()).await;
    let carried: Value = serde_json::from_str(doc.nras_answer.as_deref().unwrap()).unwrap();
    assert_eq!(
        carried[0][0], "JWT",
        "NRAS's unified answer, in the document"
    );
    assert!(doc.nras_jwks.is_some());
    let client = caller
        .verified(&lab.gpu_policy(&lab.honest()))
        .await
        .expect("the honest GPU TD's key verifies");
    assert_eq!(
        lab.nras_answers.load(SeqCst),
        asked,
        "the carrier asked NRAS nothing"
    );
    assert_eq!(
        client.key().gpu_ueids,
        vec![crate::nras::tests::GPU_SERIAL_DECIMAL.to_string()]
    );
    let (request, pending) = caller.sealed(&client, QUESTION, now()).await;
    let (status, response, _) = caller.send(request, ohttp::MEDIA_TYPE_REQUEST).await;
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&response));
    let (answer, checked) = pending
        .open_verified(&response, &catalog_digest(), now())
        .expect("the answer opens and its receipt checks");
    let shown: Value = serde_json::from_slice(&answer.body).unwrap();
    assert_eq!(shown["choices"][0]["message"]["content"], ANSWER);
    assert_eq!(checked.expect("a receipt").text, ANSWER);
    assert_carried_blind(&carrier.seen, &[QUESTION_NEEDLE, ANSWER_NEEDLE]);
    let streams = recorded(&carrier.seen);
    assert!(!streams.iter().any(|(l, _)| l.starts_with("nras")));

    let e = caller
        .verified(&lab.policy(&lab.honest()))
        .await
        .err()
        .expect("attest=cpu refuses a GPU TD");
    assert_eq!(e.step, "gpu", "{e}");
    carrier.tamper(Tamper::DropGpu);
    let e = caller
        .verified(&lab.gpu_policy(&lab.honest()))
        .await
        .err()
        .expect("the quote bound the GPU evidence");
    assert_eq!(e.step, "binding", "{e}");
    assert_eq!(td.state.stats.e2e_key_quotes.load(SeqCst), 4);
}

/// SECURITY: NRAS's verdict is the TD's to fetch and the caller's to judge.
/// A verdict for another nonce, or naming another GPU than the one the
/// quote bound, is refused at `gpu` with the relay's 75; NRAS down, the TD
/// serves no key document (503), and nothing is sealed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_nras_verdict_unbound_or_missing_gets_the_caller_no_key() {
    let lab = Lab::new().await;
    for (lie, says) in [
        (NrasLie::OtherNonce, "eat_nonce"),
        (NrasLie::OtherUeid, "ueid"),
    ] {
        let runs = Arc::new(AtomicUsize::new(0));
        let mut fake = lab.honest();
        fake.nras_lie = lie;
        let td = ready_gpu_td(&lab, fake, &runs).await;
        let carrier = Carrier::start(&lab, &td).await;
        let e = Caller::of(&carrier)
            .verified(&lab.gpu_policy(&lab.honest()))
            .await
            .err()
            .expect("an unbound verdict accepted");
        assert_eq!((e.step, e.code), ("gpu", 75), "{lie:?}: {e}");
        assert!(e.reason.contains(says), "{lie:?}: {e}");
    }
    let runs = Arc::new(AtomicUsize::new(0));
    let mut fake = lab.honest();
    fake.nras_down = true;
    let td = ready_gpu_td(&lab, fake, &runs).await;
    let key = format!("/v2/e2e/key?nonce={}", hex(&[0x52; 32]));
    let r = call(&td, "GET", &key, None, vec![]).await;
    assert_eq!(r.status, 503);
    let body = String::from_utf8_lossy(&r.body);
    assert!(body.contains("e2e key GPU verdict unavailable"), "{body}");
    assert!(!body.contains("NRAS"), "{body}");
}

/// SECURITY: a CPU-attested session's key is refused under
/// `attest=cpu+gpu`: its document carries no GPU evidence.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cpu_session_is_refused_under_attest_cpu_gpu() {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = ready_td(&lab, &runs).await;
    let carrier = Carrier::start(&lab, &td).await;
    let caller = Caller::of(&carrier);
    let e = caller
        .verified(&lab.gpu_policy(&lab.honest()))
        .await
        .err()
        .expect("refused");
    assert_eq!(e.step, "gpu", "{e}");
    assert!(e.reason.contains("no GPU evidence"), "{e}");
    caller
        .verified(&lab.policy(&lab.honest()))
        .await
        .expect("the control, under attest=cpu");
}

/// SECURITY: a GPU that gives the TD no report (outside CC mode, NVML
/// "Not Supported"), or whose NVML says CC is off, devtools on or the
/// environment not production (review of 2026-10-27, finding 3), gets the
/// caller no key document: one generic 503, the detail in the TD's log
/// only, no quote taken.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_gpu_td_without_gpu_evidence_serves_no_key_document() {
    let lab = Lab::new().await;
    let prod = crate::attest_v2::CcSystemState::PRODUCTION;
    let cases: [(&str, fn(&mut FakeTd)); 4] = [
        ("NVML: Not Supported", |f| f.gpu_cc_off = true),
        ("devtools on", |f| f.gpu_cc_state.dev_tools_mode = 1),
        ("CC feature off", |f| f.gpu_cc_state.cc_feature = 0),
        ("simulated environment", |f| f.gpu_cc_state.environment = 1),
    ];
    for (why, lie) in cases {
        let runs = Arc::new(AtomicUsize::new(0));
        let mut fake = lab.honest();
        assert_eq!(fake.gpu_cc_state, prod);
        lie(&mut fake);
        let td = start_td(
            &lab,
            Spec {
                td: fake,
                gpu: true,
                model: scripted(&runs),
                max_completion_tokens: 64,
                quote_limits: cloud_serve::QuoteLimits::for_queue(8),
            },
        )
        .await;
        release_and_wait(&td).await;
        let key = format!("/v2/e2e/key?nonce={}", hex(&[0x51; 32]));
        let r = call(&td, "GET", &key, None, vec![]).await;
        assert_eq!(r.status, 503, "{why}");
        let body = String::from_utf8_lossy(&r.body);
        assert!(body.contains("e2e key quote unavailable"), "{why}: {body}");
        assert!(
            !body.contains("NVML") && !body.contains("GPU"),
            "{why}: {body}"
        );
        assert_eq!(td.state.stats.e2e_key_quotes.load(SeqCst), 0, "{why}");
    }
}

/// SECURITY: a key document's NVML collection runs outside the quote lock
/// that session quotes share (review of 2026-10-27, finding 1): a session
/// quote asked while a GPU key document waits on a slow NVML (3 s here)
/// takes no longer than one asked alone, give or take a second.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_key_documents_nvml_collection_does_not_hold_the_session_quote() {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let mut fake = lab.honest();
    fake.gpu_delay = Duration::from_secs(3);
    let td = ready_gpu_td(&lab, fake, &runs).await;
    let t = Instant::now();
    let r = call(&td, "GET", "/v2/attestation/session", None, vec![]).await;
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let alone = t.elapsed();
    let key = format!("/v2/e2e/key?nonce={}", hex(&[0x51; 32]));
    let (endpoint, tls) = (td.endpoint.clone(), td.carrier_tls.clone());
    let document = tokio::spawn(async move {
        let mut send = mtls(&endpoint, &tls).await;
        request(&mut send, "GET", &key, None, vec![]).await.status
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    let t = Instant::now();
    let r = call(&td, "GET", "/v2/attestation/session", None, vec![]).await;
    let behind = t.elapsed();
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    assert_eq!(document.await.unwrap(), 200, "the slow document is served");
    assert!(
        behind < alone + Duration::from_secs(1),
        "the session quote waited {behind:?} behind a key document's NVML (alone {alone:?})"
    );
}

/// SECURITY: no byte string a carrier sends opens, panics or reaches the
/// model: truncations and bit flips of a real sealed request, and noise;
/// the route still answers afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_bytes_a_carrier_sends_panic_the_gateway_or_reach_the_model() {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = ready_td(&lab, &runs).await;
    let carrier = Carrier::start(&lab, &td).await;
    let caller = Caller::of(&carrier);
    let client = caller.verified(&lab.policy(&lab.honest())).await.unwrap();
    let (request, _) = caller.sealed(&client, QUESTION, now()).await;
    let e2e = &td.state.e2e;
    let mut tried = 0u64;
    for n in 0..request.len() {
        assert!(e2e.admit(&request[..n], now()).is_err(), "{n} bytes opened");
        tried += 1;
    }
    for i in 0..request.len() {
        let mut flipped = request.clone();
        flipped[i] ^= 0x80;
        assert!(
            e2e.admit(&flipped, now()).is_err(),
            "byte {i} flipped opened"
        );
        tried += 1;
    }
    let mut noise = vec![0u8; 3 * 4096];
    for (i, b) in noise.iter_mut().enumerate() {
        *b = (i.wrapping_mul(2_654_435_761) >> 13) as u8;
    }
    for len in [1, 7, 39, 55, 4151, noise.len()] {
        let mut probe = noise[..len].to_vec();
        if len > 7 {
            probe[..7].copy_from_slice(&request[..7]);
        }
        assert!(e2e.admit(&probe, now()).is_err());
        tried += 1;
    }
    assert_eq!(e2e.counts.refused_open.load(SeqCst), tried);
    assert_eq!(e2e.counts.opened.load(SeqCst), 0);
    for body in [vec![], vec![0u8], noise.clone()] {
        let r = call(
            &td,
            "POST",
            "/v2/e2e/chat",
            Some(ohttp::MEDIA_TYPE_REQUEST),
            body,
        )
        .await;
        assert_eq!(r.status, 400);
    }
    let (status, response, _) = caller.send(request, ohttp::MEDIA_TYPE_REQUEST).await;
    assert_eq!(status, 200);
    assert!(!response.is_empty());
}

// ── The real model, on CPU ───────────────────────────────────────────────────

/// A model that records each generation: prompt, decoding and text.
struct Recorded {
    inner: Box<dyn LoadedModel>,
    log: Arc<Mutex<Vec<(crate::prompt::Prompt, GenerationParams, String)>>>,
}

impl LoadedModel for Recorded {
    fn generate(
        &self,
        prompt: &crate::prompt::Prompt,
        max_tokens: usize,
    ) -> anyhow::Result<(String, (u32, u32))> {
        self.inner.generate(prompt, max_tokens)
    }
    fn generate_with(
        &self,
        prompt: &crate::prompt::Prompt,
        params: &GenerationParams,
    ) -> anyhow::Result<crate::backend::Completion> {
        let c = self.inner.generate_with(prompt, params)?;
        self.log
            .lock()
            .unwrap()
            .push((prompt.clone(), params.clone(), c.text.clone()));
        Ok(c)
    }
    fn chat_prompt(&self, messages: &[crate::server::ChatMessage]) -> crate::prompt::Prompt {
        self.inner.chat_prompt(messages)
    }
    fn model_digest_hex(&self) -> String {
        self.inner.model_digest_hex()
    }
    fn compute_device(&self) -> String {
        self.inner.compute_device()
    }
    fn isolation(&self) -> crate::backend::Isolation {
        self.inner.isolation()
    }
    fn serving_limits(&self) -> Option<(usize, usize)> {
        self.inner.serving_limits()
    }
    fn prefill_chunk_for(&self, prompt_tokens: usize) -> Option<usize> {
        self.inner.prefill_chunk_for(prompt_tokens)
    }
}

const QWEN3_QUESTION: &str = "What is the capital of France? Answer with one word.";

/// Qwen3-0.6B from `dir` (the catalog's files), through the candle backend.
fn qwen3(dir: &str, key: &ReleasedKey) -> anyhow::Result<Box<dyn LoadedModel>> {
    use crate::backend::InferenceBackend as _;
    let read = |f: &str| std::fs::read(format!("{dir}/{f}"));
    crate::candle_backend::CandleBackend::cpu(crate::candle_backend::ModelArtifacts {
        config_json: read("config.json")?,
        tokenizer_json: read("tokenizer.json")?,
    })
    .load(&read("model.safetensors")?, key)
}

/// (a) with the real model: Qwen3-0.6B's weights, the catalog's digest,
/// on CPU, behind the TD. The caller's text is what the model generated
/// for the question, and what a second load generates from the same prompt.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs X0NE_E2E_QWEN3_DIR (Qwen3-0.6B) and a release build: scripts/e2e-local.sh"]
async fn qwen3_answers_a_sealed_question_through_the_carrier() {
    let dir = std::env::var("X0NE_E2E_QWEN3_DIR").expect("X0NE_E2E_QWEN3_DIR");
    let log = Arc::new(Mutex::new(Vec::new()));
    let (d, l) = (dir.clone(), log.clone());
    let model: Maker = Arc::new(move |key| {
        Ok(Box::new(Recorded {
            inner: qwen3(&d, key)?,
            log: l.clone(),
        }) as Box<dyn LoadedModel>)
    });
    let lab = Lab::new().await;
    let t0 = Instant::now();
    let td = start_td(
        &lab,
        Spec {
            td: lab.honest(),
            gpu: false,
            model,
            max_completion_tokens: 32,
            quote_limits: cloud_serve::QuoteLimits::for_queue(8),
        },
    )
    .await;
    release_and_wait(&td).await;
    eprintln!("qwen3: loaded in {:.1} s", t0.elapsed().as_secs_f32());
    let carrier = Carrier::start(&lab, &td).await;
    let caller = Caller::of(&carrier);
    let client = caller.verified(&lab.policy(&lab.honest())).await.unwrap();
    let body = serde_json::to_vec(&json!({
        "messages": [{ "role": "user", "content": QWEN3_QUESTION }],
        "temperature": 0,
    }))
    .unwrap();
    let (request, pending) = client.seal_chat(&body, &test_random(), now()).unwrap();
    let t1 = Instant::now();
    let (status, response, _) = caller.send(request, ohttp::MEDIA_TYPE_REQUEST).await;
    eprintln!(
        "qwen3: sealed answer in {:.1} s",
        t1.elapsed().as_secs_f32()
    );
    assert_eq!(status, 200);
    let digest = catalog_digest();
    let (answer, checked) = pending.open_verified(&response, &digest, now()).unwrap();
    let checked = checked.expect("receipt");
    let (prompt, params, generated) = log.lock().unwrap().last().cloned().expect("a generation");
    assert!(prompt.as_str().contains(QWEN3_QUESTION));
    assert!(params.sampling.is_none(), "greedy");
    assert_eq!(checked.text, generated, "the text is the model's");
    assert_eq!(answer.status, 200);
    assert!(checked.text.contains("Paris"), "{:?}", checked.text);

    let key = crate::keys::release(
        &crate::server::tests::proof(),
        &x0ne_confidential::model::AttestationEvidence {
            format: x0ne_confidential::model::EvidenceFormat::StructuredClaims,
            payload: vec![],
            collected_at_unix: 1_000,
        },
        "model-key-1",
        &crate::server::tests::MemBroker,
        1_000,
    )
    .unwrap();
    let again = qwen3(&dir, &key)
        .unwrap()
        .generate_with(&prompt, &params)
        .unwrap();
    assert_eq!(again.text, generated, "another load, the same prompt");
    assert_carried_blind(&carrier.seen, &[b"France", b"Paris"]);
}

// ── The wasm32 build, in Node, through the carrier ───────────────────────────

/// The browser's path through the carrier: the wasm32 build (`test-anchor`,
/// the synthetic root) in Node verifies, seals, opens, checks the receipt.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the wasm32 build and Node: scripts/e2e-local.sh"]
async fn wasm_in_node_through_the_carrier() {
    wasm_through_the_carrier(false).await;
}

/// The same on a GPU-attested session: the page asks the carrier for
/// NRAS's verdict (`OP_NRAS_REQUEST`, `/v2/e2e/nras`), the wasm module
/// checks it, the bench's NVIDIA intermediate pinned.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the wasm32 build and Node: scripts/e2e-local.sh"]
async fn wasm_gpu_session_in_node_through_the_carrier() {
    wasm_through_the_carrier(true).await;
}

/// The C ABI's policy for the honest TD of `lab`: the lab's reference under
/// the TEST keys, its measurements pinned, the catalog's digest, the lab's
/// Intel root, and NVIDIA's intermediate for a GPU session.
fn abi_policy(lab: &Lab, gpu: bool) -> Value {
    let honest = lab.honest();
    let r = honest.rtmr();
    let json = String::from_utf8(reference_json()).unwrap();
    let (signature, proof) = attest::sign_reference_under_test_keys(json.as_bytes(), now());
    let mut policy = json!({
        "reference_json": json,
        "reference_sig_b64": b64(&signature),
        "reference_tlog_proof": proof,
        "reference_keys": "test",
        "machine_type": MACHINE,
        "mrtd_hex": hex(&honest.mrtd),
        "rtmr1_hex": hex(&r[1]),
        "rtmr2_hex": hex(&r[2]),
        "rtmr3_hex": hex(&r[3]),
        "model_digest_hex": catalog_digest(),
        "root_der_b64": b64(&lab.intel.root_der),
        "gce_root_der_b64": b64(&google_root()),
        "attest": "cpu",
    });
    if gpu {
        policy["attest"] = json!("cpu+gpu");
        policy["nras_intermediate_der_b64"] = json!(b64(&lab.nvidia.intermediate_der()));
    }
    policy
}

async fn wasm_through_the_carrier(gpu: bool) {
    let wasm = std::env::var("X0NE_SDK_WASM").expect("X0NE_SDK_WASM: the test-anchor build");
    let node = std::env::var("X0NE_NODE").unwrap_or_else(|_| "node".into());
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = if gpu {
        ready_gpu_td(&lab, lab.honest(), &runs).await
    } else {
        ready_td(&lab, &runs).await
    };
    let carrier = Carrier::start(&lab, &td).await;
    let policy = abi_policy(&lab, gpu);
    let asked = lab.nras_answers.load(SeqCst);
    let smoke = |streamed: bool| {
        let (node, wasm, url, policy) = (
            node.clone(),
            wasm.clone(),
            carrier.url.clone(),
            policy.to_string(),
        );
        async move {
            let out = tokio::task::spawn_blocking(move || {
                std::process::Command::new(node)
                    .arg(concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../x0ne-sdk/js/smoke.mjs"
                    ))
                    .arg(wasm)
                    .arg(url)
                    .arg(policy)
                    .arg(QUESTION)
                    .env("X0NE_SMOKE_STREAM", if streamed { "1" } else { "0" })
                    .output()
                    .expect("node runs")
            })
            .await
            .unwrap();
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            let v: Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
                panic!("{e}: {stdout} {}", String::from_utf8_lossy(&out.stderr))
            });
            assert!(out.status.success(), "{v}");
            v
        }
    };
    // The same question as a sealed stream: every frame pushed as it came,
    // the deltas shown unverified, the receipt checked at the end. This
    // carrier's HTTP route answers the TD's body whole, so Node may read it
    // in one frame; the frame-by-frame reading is the relay's test.
    let v = smoke(true).await;
    assert_eq!(v["status"], 200, "{v}");
    assert_eq!(v["receipt"]["checked"], true, "{v}");
    assert_eq!(v["text"], ANSWER, "{v}");
    assert_eq!(v["chunks"], 16, "{v}");
    assert!(v["streamed"]["frames"].as_u64().unwrap() >= 1, "{v}");
    assert!(v["streamed"]["deltas"].as_u64().unwrap() >= 1, "{v}");
    assert_eq!(v["streamed"]["unverified_matches"], true, "{v}");
    let v = smoke(false).await;
    assert_eq!(v["status"], 200, "{v}");
    assert_eq!(v["receipt"]["checked"], true, "{v}");
    assert_eq!(v["text"], ANSWER, "{v}");
    assert_eq!(v["wasm_imports"], 0, "{v}");
    assert_eq!(
        v["verified"]["session_config"],
        serde_json::to_value(&td.state.config.config).unwrap(),
        "the page reads the session config the quote bound: {v}"
    );
    if gpu {
        assert_eq!(
            v["verified"]["gpu_ueids"],
            json!([crate::nras::tests::GPU_SERIAL_DECIMAL]),
            "{v}"
        );
        assert_eq!(
            lab.nras_answers.load(SeqCst),
            asked,
            "NRAS's verdict came in the document"
        );
    } else {
        assert_eq!(v["verified"]["gpu_ueids"], json!([]), "{v}");
        assert_eq!(lab.nras_answers.load(SeqCst), asked, "no NRAS call");
    }
    assert_carried_blind(&carrier.seen, &[QUESTION_NEEDLE, ANSWER_NEEDLE]);
}

// ── The sealed stream (docs/sdk-e2e.md, step 6) ──────────────────────────────

fn stream_question(text: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "stream": true,
        "messages": [{ "role": "user", "content": text }],
    }))
    .unwrap()
}

/// One chunk of a sealed stream as the carrier saw it: when its last byte
/// came, after the request left, and its bytes; the response nonce leads
/// the first.
struct SeenChunk {
    at: Duration,
    bytes: Vec<u8>,
}

/// The TD's reply to a sealed request, read frame by frame as a carrier
/// reads it.
struct Streamed {
    status: u16,
    headers: HeaderMap,
    /// When the head came, after the request left.
    head: Duration,
    /// Each body frame: when it came, its bytes.
    frames: Vec<(Duration, Vec<u8>)>,
}

impl Streamed {
    fn bytes(&self) -> Vec<u8> {
        self.frames.iter().flat_map(|(_, b)| b.clone()).collect()
    }

    /// The body cut at its chunks (draft-08 §6.2): the 16-byte nonce and
    /// the first chunk, then one per varint length, the final one (a zero
    /// length) to the end.
    fn chunks(&self) -> Vec<SeenChunk> {
        let mut arrival = Vec::new();
        for (at, b) in &self.frames {
            arrival.extend(std::iter::repeat_n(*at, b.len()));
        }
        let wire = self.bytes();
        let mut out = Vec::new();
        let mut from = 0;
        for bytes in chunks_of(&wire) {
            from += bytes.len();
            out.push(SeenChunk {
                at: arrival[from - 1],
                bytes,
            });
        }
        out
    }
}

/// `wire`, a chunked response body, cut at its chunks (draft-08 §6.2): the
/// 16-byte nonce and the first chunk, then one per varint length, the final
/// one (a zero length) to the end.
pub(crate) fn chunks_of(wire: &[u8]) -> Vec<Vec<u8>> {
    let (mut out, mut from, mut at) = (Vec::new(), 0, 16);
    while at < wire.len() {
        let first = wire[at];
        let n = 1usize << (first >> 6);
        let mut len = u64::from(first & 0x3f);
        for b in &wire[at + 1..at + n] {
            len = (len << 8) | u64::from(*b);
        }
        let end = if len == 0 {
            wire.len()
        } else {
            at + n + len as usize
        };
        out.push(wire[from..end].to_vec());
        from = end;
        at = end;
    }
    out
}

/// `body` POSTed to the TD as the carrier does, with `ctype`; the answer
/// read frame by frame. Both recorded in clear, as the carrier holds them.
async fn carry_stream(c: &CarrierState, ctype: &str, body: Vec<u8>) -> Streamed {
    use hyper::body::Body as _;
    let sent = stream(
        &c.seen,
        "carrier->td clear POST /v2/e2e/chat (stream)".into(),
    );
    sent.lock().unwrap().extend_from_slice(&body);
    let mut send = mtls(&c.td, &c.tls).await;
    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v2/e2e/chat")
        .header(header::HOST, "td")
        .header(header::CONTENT_TYPE, ctype)
        .body(axum::body::Body::from(body))
        .unwrap();
    send.ready().await.unwrap();
    let t = Instant::now();
    let resp = send.send_request(req).await.unwrap();
    let head = t.elapsed();
    let (status, headers) = (resp.status().as_u16(), resp.headers().clone());
    let mut incoming = Box::pin(resp.into_body());
    let mut frames = Vec::new();
    while let Some(frame) = std::future::poll_fn(|cx| incoming.as_mut().poll_frame(cx)).await {
        let Ok(data) = frame.unwrap().into_data() else {
            continue;
        };
        frames.push((t.elapsed(), data.to_vec()));
    }
    let got = stream(
        &c.seen,
        "td->carrier clear POST /v2/e2e/chat (stream)".into(),
    );
    {
        let mut g = got.lock().unwrap();
        g.extend_from_slice(format!("{status}\r\n").as_bytes());
        for (k, v) in &headers {
            g.extend_from_slice(format!("{k}: {}\r\n", v.to_str().unwrap_or("")).as_bytes());
        }
        for (_, b) in &frames {
            g.extend_from_slice(b);
        }
    }
    Streamed {
        status,
        headers,
        head,
        frames,
    }
}

/// The text of deltas a stream handed out, still unverified.
fn unverified(deltas: Vec<x0ne_sdk::sealed_stream::UnverifiedDelta>) -> Vec<String> {
    deltas
        .into_iter()
        .map(x0ne_sdk::sealed_stream::UnverifiedDelta::into_unverified_text)
        .collect()
}

/// A verified client of a ready TD, and its carrier.
async fn stream_lab() -> (Lab, Td, Carrier, Client, Arc<AtomicUsize>) {
    let lab = Lab::new().await;
    let runs = Arc::new(AtomicUsize::new(0));
    let td = ready_td(&lab, &runs).await;
    let carrier = Carrier::start(&lab, &td).await;
    let client = Caller::of(&carrier)
        .verified(&lab.policy(&lab.honest()))
        .await
        .expect("the honest TD's key verifies");
    (lab, td, carrier, client, runs)
}

/// Sizes of a stream's chunks as the carrier sees them, at L = 256: the
/// first carries the nonce, the last 16384 bytes of plaintext.
pub(crate) fn stream_sizes(chunks: usize) -> Vec<usize> {
    let mut sizes = vec![16 + 2 + 256 + 16];
    sizes.extend(std::iter::repeat_n(2 + 256 + 16, chunks - 2));
    sizes.push(1 + 16_384 + 16);
    sizes
}

/// SECURITY, step 6: a question sealed as a chunked request crosses the
/// carrier; its answer comes back chunk by chunk, the caller opens each as
/// it arrives and checks the receipt of the last over the whole text. No
/// byte the carrier moved or held holds the question or the answer; every
/// chunk is of the cadence's size. A failure is a stream that tells a
/// carrier its text, or one the caller cannot read.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sealed_stream_crosses_the_carrier_chunk_by_chunk_and_its_receipt_checks() {
    let (_lab, td, carrier, client, runs) = stream_lab().await;
    let ran = runs.load(SeqCst);
    let (request, mut pending) = client
        .seal_chat_stream(&stream_question(QUESTION), &test_random(), now())
        .unwrap();
    let r = carry_stream(&carrier.state, ohttp::MEDIA_TYPE_CHUNKED_REQUEST, request).await;
    assert_eq!(r.status, 200);
    assert_eq!(
        r.headers.get(header::CONTENT_TYPE).unwrap(),
        ohttp::MEDIA_TYPE_CHUNKED_RESPONSE
    );
    assert_eq!(r.headers.get("incremental").unwrap(), "?1");
    let mut deltas = Vec::new();
    for (_, frame) in &r.frames {
        deltas.extend(unverified(pending.push(frame).unwrap()));
    }
    let (answer, checked) = pending.finish(&catalog_digest(), now()).unwrap();
    assert_eq!(answer.status, 200);
    assert_eq!(answer.text, ANSWER);
    // Words come 5 ms apart, ticks 100 ms: the text pending at a tick is one event.
    assert_eq!(deltas.concat(), ANSWER);
    assert_eq!(checked.unwrap().text, ANSWER);
    assert_eq!(runs.load(SeqCst), ran + 1, "the model ran once");
    let chunks = r.chunks();
    let sizes: Vec<usize> = chunks.iter().map(|c| c.bytes.len()).collect();
    assert_eq!(sizes, stream_sizes(16));
    assert_eq!(answer.chunks, 16);
    assert_carried_blind(&carrier.seen, &[QUESTION_NEEDLE, ANSWER_NEEDLE]);
    assert_eq!(td.state.e2e.counts.streams.load(SeqCst), 1);
}

/// `0` under the catalog's digest, so the TD serves it.
struct Cataloged(Box<dyn LoadedModel>);

impl LoadedModel for Cataloged {
    fn generate(
        &self,
        prompt: &crate::prompt::Prompt,
        max_tokens: usize,
    ) -> anyhow::Result<(String, (u32, u32))> {
        self.0.generate(prompt, max_tokens)
    }
    fn generate_with(
        &self,
        prompt: &crate::prompt::Prompt,
        params: &GenerationParams,
    ) -> anyhow::Result<crate::backend::Completion> {
        self.0.generate_with(prompt, params)
    }
    fn generate_streaming(
        &self,
        prompt: &crate::prompt::Prompt,
        params: &GenerationParams,
        sink: &mut dyn crate::backend::TokenSink,
    ) -> anyhow::Result<crate::backend::Completion> {
        self.0.generate_streaming(prompt, params, sink)
    }
    fn chat_prompt(&self, messages: &[crate::server::ChatMessage]) -> crate::prompt::Prompt {
        self.0.chat_prompt(messages)
    }
    fn generate_turn(
        &self,
        prompt: &crate::prompt::Prompt,
        params: &GenerationParams,
        conversation: Option<&crate::conversation::Turn>,
        sink: &mut dyn crate::backend::TokenSink,
    ) -> anyhow::Result<crate::backend::Completion> {
        self.0.generate_turn(prompt, params, conversation, sink)
    }
    fn model_digest_hex(&self) -> String {
        catalog_digest()
    }
    fn compute_device(&self) -> String {
        self.0.compute_device()
    }
    fn isolation(&self) -> crate::backend::Isolation {
        self.0.isolation()
    }
}

/// [`Scripted`] blind to the decoding asked: no sampling, no schema refused,
/// the scripted text streamed whatever the request, as a backend that
/// dropped them would.
struct Blind(Scripted);

impl LoadedModel for Blind {
    fn generate(
        &self,
        prompt: &crate::prompt::Prompt,
        max_tokens: usize,
    ) -> anyhow::Result<(String, (u32, u32))> {
        self.0.generate(prompt, max_tokens)
    }
    fn generate_streaming(
        &self,
        prompt: &crate::prompt::Prompt,
        params: &GenerationParams,
        sink: &mut dyn crate::backend::TokenSink,
    ) -> anyhow::Result<crate::backend::Completion> {
        let (text, (prompt_tokens, completion_tokens)) =
            self.0.generate(prompt, params.max_tokens)?;
        if sink.started().is_break() || sink.delta(&text).is_break() {
            return Err(crate::backend::Cancelled.into());
        }
        Ok(crate::backend::Completion {
            text,
            prompt_tokens,
            completion_tokens,
            stopped: false,
            length: false,
        })
    }
    fn chat_prompt(&self, messages: &[crate::server::ChatMessage]) -> crate::prompt::Prompt {
        self.0.chat_prompt(messages)
    }
    fn model_digest_hex(&self) -> String {
        self.0.model_digest_hex()
    }
    fn compute_device(&self) -> String {
        self.0.compute_device()
    }
    fn isolation(&self) -> crate::backend::Isolation {
        self.0.isolation()
    }
}

/// A verified client of a ready TD serving `model`, and its carrier.
async fn lab_serving(model: Maker) -> (Lab, Td, Carrier, Client) {
    let lab = Lab::new().await;
    let td = start_td(
        &lab,
        Spec {
            td: lab.honest(),
            gpu: false,
            model,
            max_completion_tokens: 64,
            quote_limits: cloud_serve::QuoteLimits::for_queue(8),
        },
    )
    .await;
    release_and_wait(&td).await;
    let carrier = Carrier::start(&lab, &td).await;
    let client = Caller::of(&carrier)
        .verified(&lab.policy(&lab.honest()))
        .await
        .expect("the honest TD's key verifies");
    (lab, td, carrier, client)
}

/// `body` sealed as a stream across `carrier`: the sizes the carrier saw,
/// and what the caller's `finish` made of it.
async fn stream_through(
    client: &Client,
    carrier: &Carrier,
    body: &[u8],
) -> (
    Vec<usize>,
    x0ne_sdk::sealed_stream::StreamAnswer,
    Option<x0ne_sdk::receipt::Checked>,
) {
    let (request, mut pending) = client
        .seal_chat_stream(body, &test_random(), now())
        .unwrap();
    let r = carry_stream(&carrier.state, ohttp::MEDIA_TYPE_CHUNKED_REQUEST, request).await;
    assert_eq!(r.status, 200);
    let sizes: Vec<usize> = r.chunks().iter().map(|c| c.bytes.len()).collect();
    for (_, frame) in &r.frames {
        pending.push(frame).unwrap();
    }
    let (answer, checked) = pending.finish(&catalog_digest(), now()).unwrap();
    (sizes, answer, checked)
}

/// SECURITY: a sealed stream's final receipt signs the decoding its
/// question asked, which the caller checks (`PendingStream::finish`): the
/// sampling and seed, the schema's digest, a text that is a document of it.
/// The model is the qwen3 fixture over JSON pieces, on CPU. A failure: a
/// stream drawn or shaped otherwise than asked taken as asked.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sealed_stream_signs_the_decoding_asked_and_holds_its_schema() {
    let maker: Maker =
        Arc::new(|_| Ok(Box::new(Cataloged(crate::e2e_sdk_tests::json_model())) as _));
    let (_lab, _td, carrier, client) = lab_serving(maker).await;
    let schema = json!({"type": "object", "properties": {"ok": {"type": "boolean"},
        "n": {"enum": [0, 1, 2, 10]}}, "required": ["ok", "n"], "additionalProperties": false});
    let body = serde_json::to_vec(&json!({
        "stream": true,
        "messages": [{"role": "user", "content": "x y"}],
        "temperature": 1.0, "seed": 4, "max_tokens": 40,
        "response_format": {"type": "json_schema",
            "json_schema": {"name": "r", "strict": true, "schema": schema}},
    }))
    .unwrap();
    let (sizes, answer, checked) = stream_through(&client, &carrier, &body).await;
    assert_eq!(answer.status, 200, "{answer:?}");
    assert_eq!(sizes, stream_sizes(sizes.len()));
    let checked = checked.expect("a 200 without an error carries a receipt");
    let v: Value = serde_json::from_str(&checked.text).unwrap();
    assert!(v["ok"].is_boolean() && v["n"].is_u64(), "{}", checked.text);
    assert_eq!(
        checked.signed["json_schema_sha256_hex"],
        json!(x0ne_sdk::json_schema::schema_sha256_hex(&schema))
    );
    assert_eq!(
        checked.signed["sampling"],
        json!({"temperature": 1.0, "top_p": 1.0, "seed": 4})
    );
}

/// SECURITY: a conversation crosses the sealed stream as it crosses the
/// whole path: the TD serves it with its signed context, the handle of the
/// next turn comes back from `finish` alone, checked against the receipt's
/// commitment, and the next turn, streamed, reuses the first one's keys
/// and values. No byte the carrier moved holds a handle. A failure: a turn
/// served on the sealed stream with no signed context, a handle the SDK
/// takes unchecked, or one the carrier can read.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_conversation_crosses_the_sealed_stream_and_the_sdk_checks_its_context() {
    let maker: Maker =
        Arc::new(|_| Ok(Box::new(Cataloged(crate::e2e_sdk_tests::chat_model())) as _));
    let (_lab, _td, carrier, client) = lab_serving(maker).await;
    let question = json!({"role": "user", "content": "t1 t2 t3 t4 t5"});
    let body = serde_json::to_vec(&json!({"stream": true, "messages": [question],
        "max_tokens": 2, "x0ne": {"conversation": "new"}}))
    .unwrap();
    let (_, answer, checked) = stream_through(&client, &carrier, &body).await;
    assert_eq!(answer.status, 200, "{answer:?}");
    let checked = checked.expect("a 200 without an error carries a receipt");
    assert_eq!(checked.signed["context"]["turn"], json!(1));
    assert_eq!(
        checked.signed["isolation"]["kv_cache_shared_between_requests"],
        json!(true)
    );
    let handle = answer.conversation.expect("the next turn's handle").handle;
    let body = serde_json::to_vec(&json!({"stream": true, "messages": [question,
        {"role": "assistant", "content": checked.text}, {"role": "user", "content": "t6"}],
        "max_tokens": 2, "x0ne": {"conversation": handle.hex()}}))
    .unwrap();
    let (_, answer, checked) = stream_through(&client, &carrier, &body).await;
    assert_eq!(answer.status, 200, "{answer:?}");
    let checked = checked.expect("a 200 without an error carries a receipt");
    assert_eq!(checked.signed["context"]["turn"], json!(2));
    assert!(checked.signed["context"]["reused_tokens"].as_u64().unwrap() > 0);
    let next = answer.conversation.expect("the next turn's handle").handle;
    assert_ne!(next, handle, "a handle serves one turn");
    let (handle, next) = (handle.hex(), next.hex());
    assert_carried_blind(&carrier.seen, &[handle.as_bytes(), next.as_bytes()]);
}

/// SECURITY: a model that keeps no conversation refuses one asked on the
/// sealed stream by name, inside the stream, as the whole path answers 400
/// (docs/api.md): never a stream served as if no conversation were asked,
/// whose receipt would sign no context. A failure: a conversation silently
/// dropped, the model run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_model_that_keeps_no_conversation_refuses_one_on_the_sealed_stream() {
    let (_lab, _td, carrier, client, runs) = stream_lab().await;
    let ran = runs.load(SeqCst);
    let body = serde_json::to_vec(&json!({"stream": true,
        "messages": [{"role": "user", "content": QUESTION}],
        "x0ne": {"conversation": "new"}}))
    .unwrap();
    let (sizes, answer, checked) = stream_through(&client, &carrier, &body).await;
    assert_eq!(sizes, stream_sizes(16));
    assert!(checked.is_none());
    let error = answer.error.expect("refused inside the stream");
    assert_eq!(error["status"], json!(400), "{error}");
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("x0ne.conversation"),
        "{error}"
    );
    assert_eq!(runs.load(SeqCst), ran, "the model did not run");
}

/// SECURITY: a streamed text outside the schema asked is not signed: the
/// TD checks it again at the end and ends the stream in an error event, at
/// a success's sizes. The deltas already handed out are not verified, as
/// for any stream (`PendingStream`). A failure: a receipt over a text the
/// backend did not hold to its schema.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_streamed_text_outside_its_schema_ends_unsigned() {
    let runs = Arc::new(AtomicUsize::new(0));
    let r = runs.clone();
    let maker: Maker = Arc::new(move |_| Ok(Box::new(Blind(Scripted { runs: r.clone() })) as _));
    let (_lab, _td, carrier, client) = lab_serving(maker).await;
    let body = serde_json::to_vec(&json!({
        "stream": true,
        "messages": [{"role": "user", "content": QUESTION}],
        "response_format": {"type": "json_schema",
            "json_schema": {"name": "r", "schema": {"type": "object"}}},
    }))
    .unwrap();
    let ran = runs.load(SeqCst);
    let (sizes, answer, checked) = stream_through(&client, &carrier, &body).await;
    assert!(checked.is_none(), "{answer:?}");
    assert_eq!(answer.status, 200);
    assert_eq!(answer.error.as_ref().unwrap()["code"], "generation_failed");
    assert_eq!(sizes, stream_sizes(16));
    assert_eq!(runs.load(SeqCst), ran + 1, "the model ran once");
    assert_carried_blind(&carrier.seen, &[QUESTION_NEEDLE, ANSWER_NEEDLE]);
}

/// SECURITY: through the C ABI, a stream that ends without a checked
/// receipt is an error, never a result: a refusal before the model (a 400
/// inside), and a 200 that ends in an error event, its text and final
/// deltas unsigned. What the TD said is apart, under `unverified`. A
/// failure: unsigned deltas handed out as the end of a stream.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_c_abi_ends_a_stream_without_a_receipt_in_an_error() {
    use x0ne_sdk::abi::{
        call, OP_SEAL_CHAT_STREAM, OP_STREAM_FINISH, OP_STREAM_PUSH, OP_VERIFY_KEY,
    };
    let runs = Arc::new(AtomicUsize::new(0));
    let counted = runs.clone();
    let maker: Maker = Arc::new(move |_| {
        Ok(Box::new(Blind(Scripted {
            runs: counted.clone(),
        })) as _)
    });
    let (lab, _td, carrier, _) = lab_serving(maker).await;
    let caller = Caller::of(&carrier);
    let nonce = test_random();
    let doc = caller.document(&nonce).await;
    let col = caller.collateral().await;
    let run = |op: u32, v: Value| call(op, &serde_json::to_vec(&v).unwrap());
    let v = run(
        OP_VERIFY_KEY,
        json!({ "document": doc, "collateral": col, "policy": abi_policy(&lab, false),
                "client_nonce_hex": hex(&nonce), "now": now() }),
    );
    assert_eq!(v["ok"], true, "{v}");
    for (schema, status) in [
        // Refused before the model: a 400 inside the stream.
        (json!({"type": "string", "pattern": "^a"}), 400),
        // Streamed, then outside its schema: an error event, unsigned.
        (json!({"type": "object"}), 200),
    ] {
        let body = json!({ "stream": true, "messages": [{ "role": "user", "content": QUESTION }],
            "response_format": { "type": "json_schema",
                "json_schema": { "name": "r", "schema": schema } } });
        let s = run(
            OP_SEAL_CHAT_STREAM,
            json!({ "key": v["key"], "body": body, "ikm_hex": hex(&test_random()), "now": now() }),
        );
        assert_eq!(s["ok"], true, "{s}");
        let request = base64::engine::general_purpose::STANDARD
            .decode(s["request_b64"].as_str().unwrap())
            .unwrap();
        let carried =
            carry_stream(&carrier.state, ohttp::MEDIA_TYPE_CHUNKED_REQUEST, request).await;
        assert_eq!(carried.status, 200);
        for (_, frame) in &carried.frames {
            let p = run(
                OP_STREAM_PUSH,
                json!({ "stream": s["stream"], "bytes_b64": b64(frame) }),
            );
            assert_eq!(p["ok"], true, "{p}");
        }
        let f = run(
            OP_STREAM_FINISH,
            json!({ "stream": s["stream"], "now": now() }),
        );
        assert_eq!(f["ok"], false, "{f}");
        assert!(
            f["error"].as_str().unwrap().contains("no checked receipt"),
            "{f}"
        );
        assert_eq!(f["unverified"]["status"], status, "{f}");
        assert!(f["unverified"]["error"].is_object(), "{f}");
        for unsigned in [
            "text",
            "final_deltas",
            "receipt",
            "finish_reason",
            "usage",
            "status",
        ] {
            assert!(
                f.get(unsigned).is_none(),
                "{unsigned} outside `unverified`: {f}"
            );
        }
    }
    assert_carried_blind(&carrier.seen, &[QUESTION_NEEDLE, ANSWER_NEEDLE]);
}

/// SECURITY: two answers twenty times apart in length, of two subjects,
/// streamed: the carrier records the same chunk sizes, and each chunk
/// leaves one tick after the one before whatever the text, measured here
/// on the real stack (the exact instants are `e2e_stream`'s paused-clock
/// tests). The offsets are printed (`--nocapture`). A failure is a size, an
/// early chunk, or a time that follows the answer in three pairs in a row.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_streamed_answers_of_different_lengths_have_one_shape_on_the_wire() {
    let (_lab, _td, carrier, client, _) = stream_lab().await;
    assert!(long_answer().len() / ANSWER.len() >= 20);
    // CI test-all, 30/10, load 23 on 24 cores: 1 pair in 15 drifted 101 ms
    // apart, the long one late from its third tick on. Wall-clock: measured again.
    let mut drifts = Vec::new();
    for _ in 0..3 {
        match one_pair_on_the_wire(&client, &carrier).await {
            Ok(()) => break,
            Err(drift) => drifts.push(drift),
        }
    }
    assert!(drifts.len() < 3, "every pair drifted: {drifts:?}");
    assert_carried_blind(
        &carrier.seen,
        &[QUESTION_NEEDLE, ANSWER_NEEDLE, LONG_NEEDLE],
    );
}

/// A short then a long answer streamed: their texts, receipts and sizes
/// asserted, and no chunk early; `Err` names the first chunk the two left
/// a tick or more apart.
async fn one_pair_on_the_wire(client: &Client, carrier: &Carrier) -> Result<(), String> {
    let mut traces = Vec::new();
    for (q, a) in [
        (QUESTION, ANSWER.to_string()),
        (LONG_QUESTION, long_answer()),
    ] {
        let (request, mut pending) = client
            .seal_chat_stream(&stream_question(q), &test_random(), now())
            .unwrap();
        let r = carry_stream(&carrier.state, ohttp::MEDIA_TYPE_CHUNKED_REQUEST, request).await;
        let mut deltas = Vec::new();
        for (_, frame) in &r.frames {
            deltas.extend(unverified(pending.push(frame).unwrap()));
        }
        let (answer, checked) = pending.finish(&catalog_digest(), now()).unwrap();
        assert_eq!(answer.text, a);
        assert_eq!(deltas.concat(), a);
        // One event of at most L bytes a tick: the long answer takes several.
        assert_eq!(deltas.len() > 1, a.len() > 256, "{deltas:?}");
        assert!(checked.is_some());
        // The head leaves with the first chunk: its time says nothing either.
        assert!(r.head >= Duration::from_millis(100), "{:?}", r.head);
        println!("{} bytes: head at {:?}", a.len(), r.head);
        traces.push(r.chunks());
    }
    let (short, long) = (&traces[0], &traces[1]);
    let sizes = |t: &[SeenChunk]| t.iter().map(|c| c.bytes.len()).collect::<Vec<_>>();
    assert_eq!(sizes(short), sizes(long));
    assert_eq!(sizes(short), stream_sizes(16));
    let tick = Duration::from_millis(100);
    for (k, (s, l)) in short.iter().zip(long).enumerate() {
        let due = tick * (k as u32 + 1);
        println!(
            "chunk {k:2}: {} bytes, short at {:?}, long at {:?}",
            s.bytes.len(),
            s.at,
            l.at
        );
        assert!(
            s.at >= due && l.at >= due,
            "chunk {k} early: {:?} {:?}",
            s.at,
            l.at
        );
        if s.at.abs_diff(l.at) >= tick {
            return Err(format!("chunk {k}: {:?} and {:?}", s.at, l.at));
        }
    }
    Ok(())
}

/// SECURITY: an answer whose last event the TD queues at its 16th tick
/// ends there: the final chunk carries the rest of the text, and `finish`
/// hands it out (`StreamAnswer::final_deltas`), so the caller shows the
/// whole text the receipt signs. A failure is a text shown cut (an audit
/// found 116 bytes of 410) while its receipt covers it whole.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_rest_of_an_answer_in_the_final_chunk_reaches_the_caller() {
    let (_lab, _td, carrier, client, _) = stream_lab().await;
    let (request, mut pending) = client
        .seal_chat_stream(&stream_question(FULL_QUESTION), &test_random(), now())
        .unwrap();
    let r = carry_stream(&carrier.state, ohttp::MEDIA_TYPE_CHUNKED_REQUEST, request).await;
    assert_eq!(r.status, 200);
    let mut pushed = Vec::new();
    for (_, frame) in &r.frames {
        pushed.extend(unverified(pending.push(frame).unwrap()));
    }
    let (answer, checked) = pending.finish(&catalog_digest(), now()).unwrap();
    assert!(checked.is_some());
    assert_eq!((answer.chunks, answer.text.clone()), (16, full_answer()));
    let pushed = pushed.concat();
    println!(
        "{} bytes: {} by push, {} by finish",
        answer.text.len(),
        pushed.len(),
        answer.final_deltas.concat().len()
    );
    assert!(
        !answer.final_deltas.is_empty(),
        "no rest in the final chunk"
    );
    assert_eq!(pushed + &answer.final_deltas.concat(), answer.text);
    assert_carried_blind(&carrier.seen, &[FULL_NEEDLE]);
}

/// [`Scripted`], each generation's start noted, the second one held
/// [`SECOND_GENERATION`] longer.
struct Clocked(Scripted, Arc<Mutex<Vec<Instant>>>);

// Its 20 ms alone flaked: a second request reaching the TD more than that after the first
// ended at 16 chunks (1 run in 11, loaded machine; a 100 ms gap forces it, 2026-11-05). With
// 800 ms a gap up to 700 ms passed and 900 ms failed: the measured margin.
const SECOND_GENERATION: Duration = Duration::from_millis(800);

impl LoadedModel for Clocked {
    fn generate(
        &self,
        prompt: &crate::prompt::Prompt,
        max_tokens: usize,
    ) -> anyhow::Result<(String, (u32, u32))> {
        self.0.generate(prompt, max_tokens)
    }
    fn generate_streaming(
        &self,
        prompt: &crate::prompt::Prompt,
        params: &GenerationParams,
        sink: &mut dyn crate::backend::TokenSink,
    ) -> anyhow::Result<crate::backend::Completion> {
        let nth = {
            let mut starts = self.1.lock().unwrap();
            starts.push(Instant::now());
            starts.len()
        };
        if nth == 2 {
            std::thread::sleep(SECOND_GENERATION);
        }
        self.0.generate_streaming(prompt, params, sink)
    }
    fn chat_prompt(&self, messages: &[crate::server::ChatMessage]) -> crate::prompt::Prompt {
        self.0.chat_prompt(messages)
    }
    fn model_digest_hex(&self) -> String {
        self.0.model_digest_hex()
    }
    fn compute_device(&self) -> String {
        self.0.compute_device()
    }
    fn isolation(&self) -> crate::backend::Isolation {
        self.0.isolation()
    }
}

/// SECURITY: a sealed stream's turn, and the queue place it holds, end at
/// a power of two of ticks after its request, as its chunks do: ended a
/// Padme size after its generation (12 % at most), a host that probes the
/// queue read the generation's time finer than the count of chunks tells
/// it (an audit, 2026-10-30). Two streams asked together: the second's
/// generation starts 16 ticks after the first's request, and the second
/// ends at 32 chunks, not 16: the cost, printed. Measured 2026-10-30 before
/// the change: generations at 48 ms and 1.05 s, 16 chunks each. A failure
/// is a turn that ends before the 16th tick.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sealed_streams_turn_ends_at_a_power_of_two_of_ticks() {
    let starts = Arc::new(Mutex::new(Vec::new()));
    let maker: Maker = {
        let (starts, runs) = (starts.clone(), Arc::new(AtomicUsize::new(0)));
        Arc::new(move |_| {
            let model = Scripted { runs: runs.clone() };
            Ok(Box::new(Clocked(model, starts.clone())) as Box<dyn LoadedModel>)
        })
    };
    let (_lab, _td, carrier, client) = lab_serving(maker).await;
    let ask = || {
        client
            .seal_chat_stream(&stream_question(QUESTION), &test_random(), now())
            .unwrap()
    };
    let ((a, pa), (b, pb)) = (ask(), ask());
    let t0 = Instant::now();
    let kind = ohttp::MEDIA_TYPE_CHUNKED_REQUEST;
    let (ra, rb) = tokio::join!(
        carry_stream(&carrier.state, kind, a),
        carry_stream(&carrier.state, kind, b)
    );
    let starts: Vec<Duration> = starts
        .lock()
        .unwrap()
        .iter()
        .map(|s| s.duration_since(t0))
        .collect();
    let mut ends: Vec<(usize, Duration)> = [&ra, &rb]
        .iter()
        .map(|r| {
            (
                r.chunks().len(),
                r.chunks().last().map_or(Duration::ZERO, |c| c.at),
            )
        })
        .collect();
    ends.sort();
    println!("generations at {starts:?}; streams of (chunks, last chunk at) {ends:?}");
    assert_eq!(starts.len(), 2);
    assert!(starts[1] >= Duration::from_millis(1600), "{starts:?}");
    assert_eq!((ends[0].0, ends[1].0), (16, 32), "{ends:?}");
    for (r, mut p) in [(ra, pa), (rb, pb)] {
        for (_, frame) in &r.frames {
            p.push(frame).unwrap();
        }
        let (answer, checked) = p.finish(&catalog_digest(), now()).unwrap();
        assert_eq!((answer.text.as_str(), checked.is_some()), (ANSWER, true));
    }
}

/// Measure, `--ignored --nocapture`: how long the TD
/// takes to admit the largest question (4 MiB, chunked) before its first
/// tick, against the shortest cadence (T = 20 ms). Past T, the head and the
/// first chunk leave at the admission's end and tell its time. Release,
/// 2026-10-30, load 33 on 24 cores: 7.5 to 12.2 ms, median 8.9 ms.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "a measure"]
async fn measure_admission_of_the_largest_question_against_a_20_ms_tick() {
    let (_lab, td, _carrier, client, _) = stream_lab().await;
    let body = serde_json::to_vec(&json!({
        "stream": true,
        "messages": [{"role": "user", "content": "a".repeat(3 << 20)}],
    }))
    .unwrap();
    let st = &td.state;
    let terms = crate::e2e_gateway::StreamTerms {
        cadence: x0ne_sdk::attest::E2eStream {
            chunk_bytes: 256,
            tick_ms: 20,
        },
        ends: st.config.config.expires_at_unix,
    };
    let mut took = Vec::new();
    for _ in 0..7 {
        let (request, _pending) = client
            .seal_chat_stream(&body, &test_random(), now())
            .unwrap();
        assert_eq!(request.len(), x0ne_sdk::e2e::MAX_SEALED_STREAM_REQUEST);
        let server = st.serving_state().unwrap();
        // Asked a second ago: the wait for the first tick is past, what is
        // timed is the admission alone.
        let start = tokio::time::Instant::now();
        let asked = start - Duration::from_secs(1);
        let response = crate::e2e_gateway::answer_stream(
            &st.e2e,
            server,
            None,
            &request,
            &st.model.name,
            [0; ohttp::RESPONSE_NONCE_LEN],
            now(),
            asked,
            terms,
        )
        .await;
        took.push(start.elapsed());
        assert_eq!(response.status(), 200);
    }
    took.sort();
    println!("admitted after (T = 20 ms): {took:?}; median {:?}", took[3]);
}

/// What the carrier can do to one live stream.
#[derive(Clone, Copy, Debug)]
enum Bend {
    /// The final chunk withheld: the stream ends one chunk short.
    DropLast,
    /// Cut inside the eighth chunk.
    CutInside,
    /// One bit flipped in the fifth chunk.
    Flip,
    /// The fourth and fifth chunks swapped.
    Swap,
    /// The fourth chunk sent again in place of the fifth.
    Replay,
}

/// SECURITY (CVE-2026-48480 and its kin): a carrier that withholds the final
/// chunk, cuts the stream, alters, swaps or replays a chunk gets a refusal
/// from the caller, never an answer. A failure is a stream the TD did not
/// write read as its whole answer.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cut_altered_swapped_or_replayed_stream_is_refused_by_the_caller() {
    let (_lab, _td, carrier, client, _) = stream_lab().await;
    for bend in [
        Bend::DropLast,
        Bend::CutInside,
        Bend::Flip,
        Bend::Swap,
        Bend::Replay,
    ] {
        let (request, mut pending) = client
            .seal_chat_stream(&stream_question(QUESTION), &test_random(), now())
            .unwrap();
        let r = carry_stream(&carrier.state, ohttp::MEDIA_TYPE_CHUNKED_REQUEST, request).await;
        let mut chunks: Vec<Vec<u8>> = r.chunks().into_iter().map(|c| c.bytes).collect();
        match bend {
            Bend::DropLast => {
                chunks.pop();
            }
            Bend::CutInside => {
                chunks.truncate(8);
                chunks[7].truncate(100);
            }
            Bend::Flip => chunks[4][40] ^= 0x01,
            Bend::Swap => chunks.swap(3, 4),
            Bend::Replay => chunks[4] = chunks[3].clone(),
        }
        let pushed: Result<Vec<_>, _> = chunks.iter().map(|c| pending.push(c)).collect();
        let finished = pushed.and_then(|_| pending.finish(&catalog_digest(), now()));
        assert!(finished.is_err(), "{bend:?}: read as whole");
    }
}

/// SECURITY: refused before the model (an empty conversation, a panic in
/// the template), failed in it (a panic in the generation): each answer
/// is a stream of the same sizes as a success's, 16 chunks, at the same
/// rhythm; the caller reads the refusal or the error inside.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refused_or_failed_stream_has_the_shape_of_a_success() {
    let (_lab, _td, carrier, client, _) = stream_lab().await;
    let empty = serde_json::to_vec(&json!({ "stream": true, "messages": [] })).unwrap();
    let cases = [
        (empty, 400),
        (stream_question(PANIC_IN_TEMPLATE), 500),
        (stream_question(PANIC_IN_GENERATION), 200),
    ];
    for (body, want) in cases {
        let (request, mut pending) = client
            .seal_chat_stream(&body, &test_random(), now())
            .unwrap();
        let r = carry_stream(&carrier.state, ohttp::MEDIA_TYPE_CHUNKED_REQUEST, request).await;
        assert_eq!(r.status, 200);
        let chunks = r.chunks();
        let sizes: Vec<usize> = chunks.iter().map(|c| c.bytes.len()).collect();
        assert_eq!(sizes, stream_sizes(16), "{want}");
        let last = chunks.last().unwrap().at;
        assert!(
            last >= Duration::from_millis(1600)
                && last < Duration::from_millis(1600) + CADENCE_SLACK,
            "{want}: {last:?}"
        );
        for (_, frame) in &r.frames {
            pending.push(frame).unwrap();
        }
        let (answer, checked) = pending.finish(&catalog_digest(), now()).unwrap();
        assert_eq!(answer.status, want);
        assert!(answer.error.is_some(), "{want}");
        assert!(checked.is_none());
    }
}

/// SECURITY: the places a stream holds on the TD, which a host can probe
/// with requests of its own (a full queue, a full set of open streams: a
/// 503 each), are held as long for a refusal as for an answer: the queue
/// place to the 16th tick (`e2e_stream::StreamTurn`), the open stream
/// place to the body's end. A failure is a refusal a probe tells from an
/// answer.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refused_stream_holds_its_places_as_long_as_an_answer() {
    let (_lab, td, carrier, client, _) = stream_lab().await;
    let admission = &td.state.serving.admission;
    let empty = serde_json::to_vec(&json!({ "stream": true, "messages": [] })).unwrap();
    let mut seen = Vec::new();
    for body in [stream_question(QUESTION), empty] {
        let (request, _) = client
            .seal_chat_stream(&body, &test_random(), now())
            .unwrap();
        let state = carrier.state.clone();
        let t = Instant::now();
        let carried = tokio::spawn(async move {
            carry_stream(&state, ohttp::MEDIA_TYPE_CHUNKED_REQUEST, request).await
        });
        let mut row = Vec::new();
        for at in [500, 1300] {
            tokio::time::sleep_until((t + Duration::from_millis(at)).into()).await;
            row.push((admission.in_flight(), admission.streams_open()));
        }
        assert_eq!(carried.await.unwrap().chunks().len(), 16);
        tokio::time::sleep(Duration::from_millis(100)).await;
        row.push((admission.in_flight(), admission.streams_open()));
        seen.push(row);
    }
    assert_eq!(seen[0], vec![(1, 1), (1, 1), (0, 0)], "the answer's places");
    assert_eq!(seen[1], seen[0], "the refusal's places");
}

/// A stream asked within the session's last minute plus its shortest
/// stream is refused before the model runs, at the rhythm of any refusal:
/// no receipt is signed past the session's end. The SDK refuses to send it
/// first; a caller whose clock is a second behind the TD's gets the TD's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stream_that_could_outlive_the_session_is_refused_before_the_model() {
    let (_lab, td, carrier, client, runs) = stream_lab().await;
    let ends = td.state.config.config.expires_at_unix;
    let skew = i64::try_from(ends - now()).unwrap() - 61;
    td.clock_skew.store(skew, SeqCst);
    let ran = runs.load(SeqCst);
    let at = now().saturating_add_signed(skew);
    let late = client.seal_chat_stream(&stream_question(QUESTION), &test_random(), at);
    assert!(
        matches!(late, Err(x0ne_sdk::e2e::E2eError::Refused(ref e)) if e.contains("session ends"))
    );
    // A caller whose clock is a second behind the TD's sends it anyway.
    let (request, mut pending) = client
        .seal_chat_stream(&stream_question(QUESTION), &test_random(), at - 1)
        .unwrap();
    let r = carry_stream(&carrier.state, ohttp::MEDIA_TYPE_CHUNKED_REQUEST, request).await;
    for (_, frame) in &r.frames {
        pending.push(frame).unwrap();
    }
    let (answer, _) = pending.finish(&catalog_digest(), at).unwrap();
    assert_eq!(answer.status, 503);
    assert_eq!(answer.chunks, 16);
    assert_eq!(runs.load(SeqCst), ran, "the model did not run");
}

/// SECURITY (RFC 9458 §6.5.1): a chunked request is answered once; sent
/// again, it gets the one refusal at its cadence and the model does not
/// run again. A failure is a second stream for one request.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_replayed_chunked_request_is_refused_and_not_run_again() {
    let (_lab, td, carrier, client, runs) = stream_lab().await;
    let (request, _) = client
        .seal_chat_stream(&stream_question(QUESTION), &test_random(), now())
        .unwrap();
    let ran = runs.load(SeqCst);
    let r = carry_stream(
        &carrier.state,
        ohttp::MEDIA_TYPE_CHUNKED_REQUEST,
        request.clone(),
    )
    .await;
    assert_eq!(r.chunks().len(), 16);
    let t = Instant::now();
    let again = call(
        &td,
        "POST",
        "/v2/e2e/chat",
        Some(ohttp::MEDIA_TYPE_CHUNKED_REQUEST),
        request,
    )
    .await;
    assert_eq!(again.status, 400);
    assert_eq!(
        again.body,
        serde_json::to_vec(&json!({ "error": REFUSED })).unwrap()
    );
    assert_cadence(t.elapsed(), "replayed");
    assert_eq!(td.state.e2e.counts.refused_replay.load(SeqCst), 1);
    assert_eq!(runs.load(SeqCst), ran + 1, "the model ran once");
}

/// The two paths do not cross: a chunked request sent as a whole one, or a
/// whole one sent as chunked, is the one refusal, unopened as a chat; the
/// largest chunked request is read, one byte more is not.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_paths_do_not_cross_and_the_chunked_route_reads_no_byte_more() {
    let (_lab, td, carrier, client, _) = stream_lab().await;
    let counts = &td.state.e2e.counts;
    let (chunked, _) = client
        .seal_chat_stream(&stream_question(QUESTION), &test_random(), now())
        .unwrap();
    let (whole, _) = client
        .seal_chat(&question(QUESTION), &test_random(), now())
        .unwrap();
    for (body, ctype) in [
        (chunked, ohttp::MEDIA_TYPE_REQUEST),
        (whole, ohttp::MEDIA_TYPE_CHUNKED_REQUEST),
    ] {
        let t = Instant::now();
        let r = call(&td, "POST", "/v2/e2e/chat", Some(ctype), body).await;
        assert_eq!(r.status, 400, "{ctype}");
        assert_eq!(
            r.body,
            serde_json::to_vec(&json!({ "error": REFUSED })).unwrap()
        );
        assert_cadence(t.elapsed(), ctype);
    }
    assert_eq!(counts.opened.load(SeqCst), 0);

    let (largest, mut pending) = client
        .seal_chat_stream(
            &stream_question(&"a".repeat(3 << 20)),
            &test_random(),
            now(),
        )
        .unwrap();
    assert_eq!(largest.len(), x0ne_sdk::e2e::MAX_SEALED_STREAM_REQUEST);
    let mut over = largest.clone();
    let r = carry_stream(&carrier.state, ohttp::MEDIA_TYPE_CHUNKED_REQUEST, largest).await;
    assert_eq!(r.status, 200);
    for (_, frame) in &r.frames {
        pending.push(frame).unwrap();
    }
    assert!(pending.finish(&catalog_digest(), now()).is_ok());
    let refused_before = counts.refused_open.load(SeqCst);
    over.push(0);
    let r = call(
        &td,
        "POST",
        "/v2/e2e/chat",
        Some(ohttp::MEDIA_TYPE_CHUNKED_REQUEST),
        over,
    )
    .await;
    assert_eq!(r.status, 400);
    assert_eq!(
        counts.refused_open.load(SeqCst),
        refused_before,
        "not read, not opened"
    );
}

/// The cost of a stream at the default cadence, measured on the wire next
/// to the whole answer to the same question: bytes the carrier moves, and
/// when the first text and the receipt reach the caller. Printed
/// (`--nocapture`) for docs/sdk-e2e.md.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_cost_of_a_sealed_stream_is_measured_against_the_whole_answer() {
    let (_lab, _td, carrier, client, _) = stream_lab().await;
    let caller = Caller::of(&carrier);
    let (request, pending) = caller.sealed(&client, QUESTION, now()).await;
    let whole_request = request.len();
    let (status, response, took) = caller.send(request, ohttp::MEDIA_TYPE_REQUEST).await;
    assert_eq!(status, 200);
    assert!(pending.open(&response).is_ok());

    let (request, mut pending) = client
        .seal_chat_stream(&stream_question(QUESTION), &test_random(), now())
        .unwrap();
    let stream_request = request.len();
    let r = carry_stream(&carrier.state, ohttp::MEDIA_TYPE_CHUNKED_REQUEST, request).await;
    let mut first_text = None;
    for (at, frame) in &r.frames {
        if !pending.push(frame).unwrap().is_empty() && first_text.is_none() {
            first_text = Some(*at);
        }
    }
    let (answer, _) = pending.finish(&catalog_digest(), now()).unwrap();
    assert_eq!(answer.text, ANSWER);
    let streamed = r.bytes().len();
    let end = r.frames.last().unwrap().0;
    println!(
        "whole: request {whole_request} B, answer {} B, answered in {took:?}",
        response.len()
    );
    println!(
        "stream: request {stream_request} B, answer {streamed} B in {} chunks, \
         first text at {:?}, receipt at {end:?}",
        answer.chunks,
        first_text.unwrap()
    );
    assert_eq!(streamed, stream_sizes(16).iter().sum::<usize>());
}

// ── The C ABI on the sealed stream (docs/sdk-e2e.md, step 6) ─────────────────

/// SECURITY, (f): the C ABI streams as the Rust SDK does, op by op: a stream
/// sealed to a verified key, every frame pushed as it came off the carrier
/// (its deltas handed out as unverified), the receipt of the final chunk
/// checked at `finish`, which takes the handle. A frame altered is refused
/// at `push` and the stream stays refused at `finish`; a body that does not
/// ask to stream is refused before anything is sealed; a handle never issued
/// serves nothing; `close` frees one whose answer never came. A failure: a
/// browser reading a stream the Rust SDK refuses, or text shown as verified
/// before its receipt.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_c_abi_streams_through_the_carrier_and_refuses_a_bent_frame() {
    use x0ne_sdk::abi::{
        call, OP_CLOSE, OP_SEAL_CHAT_STREAM, OP_STREAM_FINISH, OP_STREAM_PUSH, OP_VERIFY_KEY,
    };
    let (lab, _td, carrier, _, runs) = stream_lab().await;
    let caller = Caller::of(&carrier);
    let nonce = test_random();
    let doc = caller.document(&nonce).await;
    let col = caller.collateral().await;
    let run = |op: u32, v: Value| call(op, &serde_json::to_vec(&v).unwrap());
    let error_of = |v: &Value| v["error"].as_str().unwrap_or("").to_string();
    let v = run(
        OP_VERIFY_KEY,
        json!({ "document": doc, "collateral": col, "policy": abi_policy(&lab, false),
                "client_nonce_hex": hex(&nonce), "now": now() }),
    );
    assert_eq!(v["ok"], true, "{v}");
    let key = v["key"].clone();
    let seal = |body: Vec<u8>| {
        run(
            OP_SEAL_CHAT_STREAM,
            json!({ "key": key, "body": serde_json::from_slice::<Value>(&body).unwrap(),
                    "ikm_hex": hex(&test_random()), "now": now() }),
        )
    };
    let decode = |s: &Value| {
        base64::engine::general_purpose::STANDARD
            .decode(s["request_b64"].as_str().unwrap())
            .unwrap()
    };
    // A body that does not ask to stream is refused before anything is sealed.
    let whole = seal(question(QUESTION));
    assert!(error_of(&whole).contains("stream must be true"), "{whole}");

    let s = seal(stream_question(QUESTION));
    assert_eq!(s["ok"], true, "{s}");
    let ran = runs.load(SeqCst);
    let r = carry_stream(
        &carrier.state,
        ohttp::MEDIA_TYPE_CHUNKED_REQUEST,
        decode(&s),
    )
    .await;
    assert_eq!(r.status, 200);
    let mut unverified = String::new();
    for (_, frame) in &r.frames {
        let p = run(
            OP_STREAM_PUSH,
            json!({ "stream": s["stream"], "bytes_b64": b64(frame) }),
        );
        assert_eq!(p["ok"], true, "{p}");
        for d in p["unverified_deltas"].as_array().unwrap() {
            unverified.push_str(d.as_str().unwrap());
        }
    }
    let f = run(
        OP_STREAM_FINISH,
        json!({ "stream": s["stream"], "now": now() }),
    );
    assert_eq!(f["ok"], true, "{f}");
    assert_eq!(f["status"], 200, "{f}");
    assert_eq!(f["text"], ANSWER, "{f}");
    assert_eq!(unverified, ANSWER);
    assert_eq!(f["receipt"]["checked"], true, "{f}");
    assert_eq!(f["receipt"]["model_digest_hex"], catalog_digest(), "{f}");
    assert_eq!(f["chunks"], 16, "{f}");
    // What the receipt does not sign is apart, under `unverified`.
    assert_eq!(f["unverified"]["finish_reason"], "stop", "{f}");
    for unsigned in ["finish_reason", "usage", "error", "body"] {
        assert!(
            f.get(unsigned).is_none(),
            "{unsigned} beside the signed text: {f}"
        );
    }
    assert_eq!(runs.load(SeqCst), ran + 1, "the model ran once");
    // `finish` took the handle.
    let again = run(
        OP_STREAM_FINISH,
        json!({ "stream": s["stream"], "now": now() }),
    );
    assert!(error_of(&again).contains("no pending stream"), "{again}");

    // One bit flipped in the fifth chunk: refused at push, and at finish.
    let s = seal(stream_question(QUESTION));
    let r = carry_stream(
        &carrier.state,
        ohttp::MEDIA_TYPE_CHUNKED_REQUEST,
        decode(&s),
    )
    .await;
    let mut chunks: Vec<Vec<u8>> = r.chunks().into_iter().map(|c| c.bytes).collect();
    chunks[4][40] ^= 0x01;
    let mut refused = false;
    for c in &chunks {
        let p = run(
            OP_STREAM_PUSH,
            json!({ "stream": s["stream"], "bytes_b64": b64(c) }),
        );
        if p["ok"] == false {
            refused = true;
            break;
        }
    }
    assert!(refused, "a flipped frame is refused at push");
    let f = run(
        OP_STREAM_FINISH,
        json!({ "stream": s["stream"], "now": now() }),
    );
    assert!(error_of(&f).contains("refused already"), "{f}");

    // `close` frees a stream whose answer never came.
    let s = seal(stream_question(QUESTION));
    let closed = run(OP_CLOSE, json!({ "handle": s["stream"] }));
    assert_eq!(closed["ok"], true, "{closed}");
    let p = run(
        OP_STREAM_PUSH,
        json!({ "stream": s["stream"], "bytes_b64": "" }),
    );
    assert!(error_of(&p).contains("no pending stream"), "{p}");
    let closed = run(OP_CLOSE, json!({ "handle": key }));
    assert_eq!(closed["ok"], true, "{closed}");
    assert_carried_blind(&carrier.seen, &[QUESTION_NEEDLE, ANSWER_NEEDLE]);
}
