//! x0ne-sdk against the simulated TD: the caller checks the TD's attested
//! OHTTP key itself, seals the question to it, and checks the receipt of the
//! answer under the session key the quote bound. Nothing between the two
//! ends reads the text.
//!
//! The bench is this crate's synthetic Intel (`attest_v2::fixtures`): a real
//! DCAP chain, TCB info, QE identity, PCK CRL and root CA CRL under a test
//! root, so the SDK's verifier (x0ne-infer's own `pcs.rs`, `tcb.rs`,
//! `gce_endorsement.rs`, `ccel.rs` and `reference.rs`, compiled into the
//! SDK) runs unchanged and every refusal is a real one. The TD's MRTD is the
//! one Google's vendored endorsement names, under Google's real root; its
//! event log is the reference's test image (`reference::tests`). The TD half
//! is the TD binary's own (`e2e_gateway`: the gateway, the key document, the
//! sealed answer through `server::chat_as`, echo backend, real receipts),
//! with the lies below told around it. The TD's routes themselves, behind
//! mutual TLS and a carrier, are tested in `e2e_td_tests`.
//!
//! The refusals replay, on the SDK's side, the relay's
//! (docs/attested-channel.md): a measurement drift, a quote that answers another nonce,
//! a carrier holding a key of its own, a twin TD, a downgrade of the suite,
//! and a replayed request. `wasm_*` runs the same path through the wasm32
//! build in Node (`scripts/sdk-wasm.sh`).

use crate::attest_v2::fixtures::{
    fresh, synthetic_report, years, FakeIntel, FakeNvidia, NrasLie, OTHER_GPU_SERIAL_BYTES,
    STANDARD_EVALUATION,
};
use crate::backend::{EchoBackend, InferenceBackend as _};
use crate::binding::{BindingV2, SessionIdentity};
use crate::ccel::Event;
use crate::cloud_config::{CloudConfig, Loaded};
use crate::e2e_gateway::{E2e, REFUSED};
use crate::nras::tests::{gpu_chain_b64, GPU_SERIAL_BYTES, GPU_SERIAL_DECIMAL};
use crate::server::{self, RawEvidence, ServerState, DEFAULT_MAX_COMPLETION_TOKENS};
use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine as _;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use x0ne_confidential::model::{AttestationEvidence, EvidenceFormat};
use x0ne_sdk::attest::{self, Collateral, KeyDocument, Policy};
use x0ne_sdk::e2e::{self, Client, E2eError, Gateway};
use x0ne_sdk::ohttp::{self, KeyConfig};

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn b64(b: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(b)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

const MACHINE: &str = "a3-highgpu-1g";

/// The synthetic firmware our Google endorses ([`endorsement`]): its MRTD.
fn mrtd() -> [u8; 48] {
    crate::tdvf::tests::synthetic_mrtd()
}

/// The honest TD's event log: a GCE-shaped RTMR0 whose configuration
/// volume is the synthetic firmware's, then the test image of
/// `reference::tests`.
fn log() -> Vec<Event> {
    crate::reference::tests::gce_log(crate::tdvf::tests::synthetic_cfv())
}

/// Our Google's endorsement of the synthetic firmware for [`mrtd`].
fn endorsement() -> Vec<u8> {
    crate::gce_endorsement::tests::fake_google::fake_google()
        .endorse_binary(crate::tdvf::tests::synthetic(), &[mrtd()])
}

/// Our Google's root, which signed [`endorsement`].
fn google_root() -> Vec<u8> {
    crate::gce_endorsement::tests::fake_google::fake_google()
        .root_der
        .clone()
}

/// RTMR0..3 as the honest log replays them.
fn rtmr() -> [[u8; 48]; 4] {
    crate::ccel::replay(&log()).unwrap()
}

/// The reference of that image, as the caller holds the file: RTMR0 pinned
/// to the honest boot's host events (`gce-endorsed`), our Google's root
/// named by its SPKI digest.
fn reference_json() -> String {
    let spki = crate::tls::spki_sha256_of_cert(&google_root()).unwrap();
    serde_json::to_string(&crate::reference::tests::endorsed_reference(
        &hex(&spki),
        MACHINE,
        crate::reference::tests::rtmr0_pins(),
    ))
    .unwrap()
}

/// `json` as x0ne would publish it, under the TEST archive key and the TEST
/// log: its detached signature and its log proof, both made at `at`.
fn signed(json: &[u8], at: u64) -> (Vec<u8>, String) {
    attest::sign_reference_under_test_keys(json, at)
}

/// `json`, signed and logged at `now()`, admitted as the caller admits it.
fn admitted(json: &[u8]) -> attest::AdmittedReference {
    let (signature, tlog_proof) = signed(json, now());
    attest::admit_reference_under_test_keys(
        &attest::SignedReference {
            json,
            signature: &signature,
            tlog_proof: &tlog_proof,
        },
        &mut attest::LogMemory::new(),
        None,
        now(),
    )
    .expect("the bench's reference, signed and logged")
}

/// The policy fields of the C ABI that carry [`reference_json`], signed and
/// logged at `now()` under the TEST key and log.
fn reference_fields() -> Value {
    let json = reference_json();
    let (signature, proof) = signed(json.as_bytes(), now());
    json!({
        "reference_json": json,
        "reference_sig_b64": b64(&signature),
        "reference_tlog_proof": proof,
        "reference_keys": "test",
    })
}

/// [`reference_fields`] and `rest`, one policy object.
fn abi_policy(rest: Value) -> Value {
    let mut p = reference_fields();
    for (k, v) in rest.as_object().unwrap() {
        p[k] = v.clone();
    }
    p
}
const QUESTION: &str = "what colour is the sky over Lyon-Perrache tonight";
/// The word of [`QUESTION`] a carrier would look for.
const NEEDLE: &[u8] = b"Perrache";

/// What the simulated TD lies about. `None`: an honest TD.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lie {
    None,
    /// The key document carries a key the carrier holds, next to the TD's quote.
    CarrierKey,
    /// The document carries another receipt key than the one the quote bound.
    ReceiptKey,
    /// The quote answers another caller's nonce.
    OtherNonce,
    /// TDATTRIBUTES.DEBUG set.
    Debug,
    /// The quote's RTMR2 is not what its event log replays to.
    RtmrDrift,
    /// The host's ACPI tables differ: one more RTMR0 event, logged and
    /// quoted. Reported, not pinned.
    Rtmr0Drift,
    /// The chat handler signs receipts with a key the quote did not bind.
    UnboundSigner,
}

/// The catalog name the bench's session config asks for. The bench serves
/// the echo backend: the receipt's digest is that backend's, pinned apart.
const MODEL: &str = "qwen3:0.6b";
/// SHA-256 of the relay's TLS key: the one client the bench's TD accepts.
const RELAY_SPKI: [u8; 32] = [0x52; 32];
/// SHA-256 of the bench TD's TLS key.
const TD_TLS_SPKI: [u8; 32] = [0x54; 32];

/// The session config the relay wrote for the bench's TD: [`MODEL`],
/// [`RELAY_SPKI`], `attest`, ending an hour on.
fn session_config(attest: &str) -> Loaded {
    CloudConfig::new(MODEL, RELAY_SPKI, [0x43; 32], now() + 3600, attest, 443, 64)
        .to_loaded()
        .unwrap()
}

/// The binary's gateway of a TD booted under `config`, its receipts signed
/// by `identity`, its OHTTP key derived from `seed`.
fn gateway(seed: [u8; 32], identity: &SessionIdentity, config: &Loaded) -> E2e {
    let boot = BindingV2::of(identity, TD_TLS_SPKI, config.sha256);
    E2e::from_seed(&seed, boot, &config.bytes).unwrap()
}

/// The TD half: the binary's gateway (its key, its boot binding, its
/// session config), its registers.
struct Td {
    intel: Arc<FakeIntel>,
    e2e: E2e,
    lie: Lie,
}

impl Td {
    fn new(intel: Arc<FakeIntel>, seed: u8, identity: &SessionIdentity, lie: Lie) -> Self {
        Self::under(intel, seed, identity, lie, &session_config("cpu"))
    }

    /// [`Self::new`] booted under `config`.
    fn under(
        intel: Arc<FakeIntel>,
        seed: u8,
        identity: &SessionIdentity,
        lie: Lie,
        config: &Loaded,
    ) -> Self {
        Self {
            intel,
            e2e: gateway([seed; 32], identity, config),
            lie,
        }
    }

    /// `GET /v2/e2e/key?nonce=`: the binary's key document over a quote
    /// taken now, with this TD's lie told in it.
    fn key_document(&self, nonce: &[u8; 32]) -> KeyDocument {
        let bound_nonce = if self.lie == Lie::OtherNonce {
            [0xEE; 32]
        } else {
            *nonce
        };
        let report_data = self.e2e.report_data(&bound_nonce);
        let mut events = log();
        if self.lie == Lie::Rtmr0Drift {
            // EV_PLATFORM_CONFIG_FLAGS "ACPI DATA", as GCE logs its tables.
            events.insert(
                0,
                Event::new_sha384(1, 0xa, [0x99; 48], b"ACPI DATA".to_vec()),
            );
        }
        let mut rtmr = crate::ccel::replay(&events).unwrap();
        if self.lie == Lie::RtmrDrift {
            rtmr[2] = [0x99; 48];
        }
        let attributes = if self.lie == Lie::Debug {
            [1, 0, 0, 0, 0, 0, 0, 0]
        } else {
            [0; 8]
        };
        let quote = self.intel.quote(&report_data, &mrtd(), &rtmr, attributes);
        let mut doc = self.e2e.document(&quote, &crate::ccel::encode(&events));
        match self.lie {
            Lie::CarrierKey => {
                doc.key_config_b64 = b64(&Gateway::from_seed(&[0xCA; 32]).key_config());
            }
            Lie::ReceiptKey => {
                doc.session_pubkey_hex = hex(&SessionIdentity::generate([0xCB; 32], [0; 32])
                    .public_key()
                    .to_bytes());
            }
            _ => {}
        }
        doc
    }
}

/// Intel's signed collateral and Google's endorsement, as anyone may relay
/// them, issued as Intel serves it at `now()` (fixtures: an hour before,
/// next update 30 days on).
fn collateral(intel: &FakeIntel, revoked: &[Vec<u8>]) -> Collateral {
    collateral_at(intel, revoked, now())
}

/// The same, as Intel serves it at `at`.
fn collateral_at(intel: &FakeIntel, revoked: &[Vec<u8>], at: u64) -> Collateral {
    let window = fresh(at);
    Collateral {
        tcb_info: intel.tcb_info(window.0),
        tcb_info_chain_pem: intel.collateral_chain_pem(),
        qe_identity: intel.qe_identity(window.0),
        qe_identity_chain_pem: intel.collateral_chain_pem(),
        tcb_info_standard: intel.tcb_info_with(window.0, STANDARD_EVALUATION, "UpToDate"),
        tcb_info_standard_chain_pem: intel.collateral_chain_pem(),
        qe_identity_standard: intel.qe_identity_with(window.0, STANDARD_EVALUATION),
        qe_identity_standard_chain_pem: intel.collateral_chain_pem(),
        pck_crl_der_b64: b64(&intel.pck_crl(revoked, window)),
        root_ca_crl_der_b64: b64(&intel.root_crl(&[], window)),
        gce_endorsement_b64: b64(&endorsement()),
        gce_firmware_b64: b64(crate::tdvf::tests::synthetic()),
        ..Collateral::default()
    }
}

/// The caller's policy: the simulated Intel root, our Google's root, the
/// image reference signed and logged; the relay's `transparent` TCB policy;
/// no pin.
fn policy(intel: &FakeIntel) -> Policy {
    Policy {
        anchor: x0ne_sdk::TrustAnchor::RootDer(intel.root_der.clone()),
        gce_root_der: google_root(),
        // A simulated TD serves on the CPU: the tests' `attest=cpu` (x0ne-sdk `test-anchor`).
        attest: x0ne_sdk::attest::Attest::Cpu,
        ..Policy::production(admitted(reference_json().as_bytes()), MACHINE)
    }
}

/// The same, the image's MRTD and RTMR1..3 pinned on top.
fn pinned(intel: &FakeIntel) -> Policy {
    let r = rtmr();
    policy(intel).pinned(mrtd(), [r[1], r[2], r[3]])
}

/// The real chat handler's state, signing receipts with `identity`, over
/// `model`, else the echo backend.
fn chat_server_with(
    identity: SessionIdentity,
    model: Option<Box<dyn crate::backend::LoadedModel>>,
) -> (Arc<ServerState>, String) {
    let p = server::tests::proof();
    let key = crate::keys::release(
        &p,
        &AttestationEvidence {
            format: EvidenceFormat::StructuredClaims,
            payload: vec![],
            collected_at_unix: 1_000,
        },
        "model-key-1",
        &server::tests::MemBroker,
        1_000,
    )
    .unwrap();
    let model = match model {
        Some(model) => model,
        None => EchoBackend.load(b"encrypted-weights", &key).unwrap(),
    };
    let digest = model.model_digest_hex();
    let state = ServerState {
        raw_evidence: RawEvidence::default(),
        proof: p,
        identity,
        model,
        max_completion_tokens: DEFAULT_MAX_COMPLETION_TOKENS,
        lease: server::tests::live_lease(),
        salt_source: Box::new(|| [3u8; 32]),
        challenge_source: server::tests::challenge_fixture(),
    };
    (Arc::new(state), digest)
}

/// The TD's `/v2/e2e/*` surface, and what crossed it as the carrier saw it.
struct Bench {
    td: Td,
    collateral: Collateral,
    chat: Arc<ServerState>,
    /// The name the chat is served under: the model's digest, as `server::chat`.
    served: String,
    carried: Mutex<Vec<Vec<u8>>>,
    nonces: AtomicU64,
}

async fn key(State(b): State<Arc<Bench>>, Query(q): Query<HashMap<String, String>>) -> Response {
    let Some(nonce) = q
        .get("nonce")
        .and_then(|n| crate::cloud_config::unhex::<32>(n).ok())
    else {
        return (StatusCode::BAD_REQUEST, "nonce: 64 hex digits").into_response();
    };
    Json(b.td.key_document(&nonce)).into_response()
}

async fn relayed_collateral(State(b): State<Arc<Bench>>) -> Json<Collateral> {
    Json(b.collateral.clone())
}

/// `POST /v2/e2e/chat`: the binary's sealed answer (`e2e_gateway::answer`).
async fn chat(State(b): State<Arc<Bench>>, body: Bytes) -> Response {
    let asked = tokio::time::Instant::now();
    b.carried.lock().unwrap().push(body.to_vec());
    let n = b.nonces.fetch_add(1, Ordering::SeqCst);
    let mut nonce = [0u8; ohttp::RESPONSE_NONCE_LEN];
    nonce[..8].copy_from_slice(&n.to_be_bytes());
    let response = crate::e2e_gateway::answer(
        &b.td.e2e,
        b.chat.clone(),
        None,
        &body,
        &b.served,
        nonce,
        now(),
        asked,
    )
    .await;
    let (parts, body) = response.into_parts();
    let sealed = axum::body::to_bytes(body, 8 << 20).await.unwrap();
    if parts.status == StatusCode::OK {
        b.carried.lock().unwrap().push(sealed.to_vec());
    }
    Response::from_parts(parts, axum::body::Body::from(sealed))
}

struct Running {
    url: String,
    bench: Arc<Bench>,
    intel: Arc<FakeIntel>,
    digest: String,
}

/// A bench with one TD telling `lie`, its CRL listing `revoked`.
async fn bench(lie: Lie, revoked: &[Vec<u8>]) -> Running {
    bench_with(lie, revoked, None).await
}

/// [`bench`], the TD answering with `model`, else the echo backend.
async fn bench_with(
    lie: Lie,
    revoked: &[Vec<u8>],
    model: Option<Box<dyn crate::backend::LoadedModel>>,
) -> Running {
    let intel = Arc::new(FakeIntel::new());
    let bound = SessionIdentity::generate([7; 32], [9; 32]);
    let signer = if lie == Lie::UnboundSigner {
        SessionIdentity::generate([8; 32], [9; 32])
    } else {
        bound.clone()
    };
    let (chat_state, digest) = chat_server_with(signer, model);
    let collateral = collateral(&intel, revoked);
    let bench = Arc::new(Bench {
        td: Td::new(intel.clone(), 5, &bound, lie),
        collateral,
        chat: chat_state,
        served: digest.clone(),
        carried: Mutex::new(Vec::new()),
        nonces: AtomicU64::new(1),
    });
    let app = Router::new()
        .route("/v2/e2e/key", get(key))
        .route("/v2/e2e/collateral", get(relayed_collateral))
        .route("/v2/e2e/chat", post(chat))
        .with_state(bench.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Running {
        url: format!("http://{addr}"),
        bench,
        intel,
        digest,
    }
}

fn question() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "messages": [{ "role": "user", "content": QUESTION }],
    }))
    .unwrap()
}

/// The caller's side, natively: fetch, verify, seal, send, open, check.
async fn verified(r: &Running, nonce: [u8; 32]) -> Result<Client, attest::Refusal> {
    let http = reqwest::Client::new();
    let doc: KeyDocument = http
        .get(format!("{}/v2/e2e/key?nonce={}", r.url, hex(&nonce)))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let col: Collateral = http
        .get(format!("{}/v2/e2e/collateral", r.url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    attest::verify_key(&doc, &col, &policy(&r.intel), &nonce, now()).map(Client::new)
}

async fn send_sealed(r: &Running, request: Vec<u8>) -> (u16, Vec<u8>) {
    let resp = reqwest::Client::new()
        .post(format!("{}/v2/e2e/chat", r.url))
        .header("content-type", ohttp::MEDIA_TYPE_REQUEST)
        .body(request)
        .send()
        .await
        .unwrap();
    (resp.status().as_u16(), resp.bytes().await.unwrap().to_vec())
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

// ── The accepting path ───────────────────────────────────────────────────────

/// SECURITY: the whole path. The caller checks the key, seals, and reads an
/// answer whose receipt the bound key signed over the text shown; the
/// carrier saw two padded blobs, neither holding a word of the question.
#[tokio::test]
async fn a_question_crosses_sealed_and_its_answer_comes_back_with_a_checked_receipt() {
    let r = bench(Lie::None, &[]).await;
    let client = verified(&r, [0x51; 32]).await.expect("an honest TD's key");
    assert_eq!(client.key().tcb_platform, "UpToDate");
    assert_eq!(client.key().rtmr0_hex, hex(&rtmr()[0]));
    let (request, pending) = client.seal_chat(&question(), &[0x61; 32], now()).unwrap();
    assert_eq!(
        request.len(),
        ohttp::HEADER_LEN + 32 + e2e::MIN_PADDED + 16,
        "a 4 KiB bucket, sealed"
    );
    let (status, response) = send_sealed(&r, request).await;
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&response));
    let (answer, checked) = pending
        .open_verified(&response, &r.digest, now())
        .expect("the receipt checks under the bound key");
    let checked = checked.expect("a 200 carries a receipt");
    assert_eq!(answer.status, 200);
    let shown: Value = serde_json::from_slice(&answer.body).unwrap();
    assert_eq!(
        shown["choices"][0]["message"]["content"],
        checked.text.as_str()
    );
    assert!(checked.text.contains("Lyon-Perrache"), "{}", checked.text);
    assert_eq!(checked.model_digest_hex, r.digest);
    assert_eq!(checked.compute_device, "cpu");

    let carried = r.bench.carried.lock().unwrap();
    assert_eq!(carried.len(), 2);
    for bytes in carried.iter() {
        assert!(!contains(bytes, NEEDLE), "the carrier read the question");
        assert!(!contains(bytes, b"application/json"));
    }
    // The TD pads a whole answer's body to 16 KiB at least (padding.rs); with
    // its Binary HTTP framing that fills the next bucket, 32 KiB (2026-10-24).
    let bucket = e2e::padded_len(crate::padding::MIN_WHOLE_BYTES as usize + 1).unwrap();
    assert_eq!(carried[1].len(), ohttp::RESPONSE_NONCE_LEN + bucket + 16);
}

/// The candle model, prompted with its messages' text alone: the qwen3
/// fixture's 40 ids leave no room for the template's markers.
struct Untemplated(Box<dyn crate::backend::LoadedModel>);

impl crate::backend::LoadedModel for Untemplated {
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
        params: &crate::backend::GenerationParams,
    ) -> anyhow::Result<crate::backend::Completion> {
        self.0.generate_with(prompt, params)
    }
    fn generate_streaming(
        &self,
        prompt: &crate::prompt::Prompt,
        params: &crate::backend::GenerationParams,
        sink: &mut dyn crate::backend::TokenSink,
    ) -> anyhow::Result<crate::backend::Completion> {
        self.0.generate_streaming(prompt, params, sink)
    }
    fn chat_prompt(&self, messages: &[server::ChatMessage]) -> crate::prompt::Prompt {
        let words: Vec<&str> = messages.iter().map(|m| m.content.as_str()).collect();
        crate::prompt::Prompt::text(words.join(" "))
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

/// The qwen3 fixture over a vocabulary of JSON pieces (`candle_backend::tests`).
pub(crate) fn json_model() -> Box<dyn crate::backend::LoadedModel> {
    use crate::candle_backend::tests::{fixture, json_tokenizer};
    use crate::candle_backend::{CandleBackend, ModelArtifacts};
    let (config, weights, _) = fixture("qwen3");
    let artifacts = ModelArtifacts {
        config_json: config,
        tokenizer_json: json_tokenizer(),
    };
    let key = crate::keys::test_released_key(vec![7u8; 32]);
    Box::new(Untemplated(
        CandleBackend::cpu(artifacts).load(&weights, &key).unwrap(),
    ))
}

/// SECURITY: through the sealed path the TD holds the schema and signs its
/// digest and the seed, which the SDK checks; a bad schema is an OpenAI 400.
/// A failure: a schema not held, or a receipt that does not say so accepted.
#[tokio::test]
async fn a_json_schema_crosses_sealed_and_the_sdk_checks_what_the_receipt_signs() {
    let r = bench_with(Lie::None, &[], Some(json_model())).await;
    let client = verified(&r, [0x53; 32]).await.expect("an honest TD's key");
    let schema = json!({"type": "object", "properties": {"ok": {"type": "boolean"},
        "n": {"enum": [0, 1, 2, 10]}}, "required": ["ok", "n"], "additionalProperties": false});
    let ask = |schema: Value| {
        serde_json::to_vec(&json!({"messages": [{"role": "user", "content": "x y"}],
            "temperature": 1.0, "seed": 4, "max_tokens": 40,
            "response_format": {"type": "json_schema",
                "json_schema": {"name": "r", "strict": true, "schema": schema}}}))
        .unwrap()
    };
    let (request, pending) = client
        .seal_chat(&ask(schema.clone()), &[0x62; 32], now())
        .unwrap();
    let (status, response) = send_sealed(&r, request).await;
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&response));
    let (answer, checked) = pending
        .open_verified(&response, &r.digest, now())
        .expect("the receipt and the decoding check");
    assert_eq!(
        answer.status,
        200,
        "{}",
        String::from_utf8_lossy(&answer.body)
    );
    let checked = checked.expect("a 200 carries a receipt");
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

    let (request, pending) = client
        .seal_chat(
            &ask(json!({"type": "string", "pattern": "^a"})),
            &[0x63; 32],
            now(),
        )
        .unwrap();
    let (_, response) = send_sealed(&r, request).await;
    let (answer, checked) = pending.open_verified(&response, &r.digest, now()).unwrap();
    assert!(checked.is_none());
    assert_eq!(answer.status, 400);
    let refused: Value = serde_json::from_slice(&answer.body).unwrap();
    let message = refused["error"]["message"].as_str().unwrap();
    assert!(
        message.starts_with("response_format: json_schema.schema") && message.contains("pattern"),
        "{refused}"
    );
    for field in ["type", "param", "code"] {
        assert!(refused["error"].get(field).is_some(), "{refused}");
    }
}

/// The qwen3 fixture with ChatML's markers inside its 40 ids, as served.
pub(crate) fn chat_model() -> Box<dyn crate::backend::LoadedModel> {
    use crate::candle_backend::tests::fixture;
    use crate::candle_backend::{CandleBackend, ModelArtifacts};
    let (config, weights, _) = fixture("qwen3");
    let artifacts = ModelArtifacts {
        config_json: config,
        tokenizer_json: crate::prompt::tests::chatml_tokenizer_json(),
    };
    let key = crate::keys::test_released_key(vec![7u8; 32]);
    CandleBackend::cpu(artifacts).load(&weights, &key).unwrap()
}

/// SECURITY: a conversation crosses the sealed path: its handle comes back
/// inside the sealed answer only, the next turn reuses the first one's keys
/// and values, and the SDK checks the signed context against the handles it
/// sent and received. A failure: a reuse the caller cannot see, or a handle
/// the carrier can read.
#[tokio::test]
async fn a_conversation_crosses_sealed_and_the_sdk_checks_its_context() {
    let r = bench_with(Lie::None, &[], Some(chat_model())).await;
    let client = verified(&r, [0x54; 32]).await.expect("an honest TD's key");
    let question = json!({"role": "user", "content": "t1 t2 t3 t4 t5"});
    let turn = |body: Value, label: u8| {
        let (request, pending) = client
            .seal_chat(&serde_json::to_vec(&body).unwrap(), &[label; 32], now())
            .unwrap();
        let r = &r;
        async move {
            let (status, response) = send_sealed(r, request).await;
            assert_eq!(status, 200);
            let (answer, checked) = pending
                .open_verified(&response, &r.digest, now())
                .expect("the receipt and its context check");
            let shown: Value = serde_json::from_slice(&answer.body).unwrap();
            (shown, checked.expect("a 200 carries a receipt"))
        }
    };
    let (shown, checked) = turn(
        json!({"messages": [question], "max_tokens": 2, "x0ne": {"conversation": "new"}}),
        0x64,
    )
    .await;
    assert_eq!(checked.signed["context"]["turn"], json!(1));
    let handle = shown["x0ne"]["conversation"]["handle"]
        .as_str()
        .unwrap()
        .to_string();
    let (_, checked) = turn(
        json!({"messages": [question, {"role": "assistant", "content": checked.text},
            {"role": "user", "content": "t6"}], "max_tokens": 2,
            "x0ne": {"conversation": handle}}),
        0x65,
    )
    .await;
    assert_eq!(checked.signed["context"]["turn"], json!(2));
    assert!(checked.signed["context"]["reused_tokens"].as_u64().unwrap() > 0);
    let carried = r.bench.carried.lock().unwrap();
    assert_eq!(carried.len(), 4);
    for bytes in carried.iter() {
        assert!(
            !contains(bytes, handle.as_bytes()),
            "the carrier read the handle"
        );
    }
}

/// The C ABI the browser calls, natively, on the same bench: verified key
/// handle, sealed question, opened answer with its receipt checked.
#[tokio::test]
async fn the_c_abi_runs_the_same_path() {
    use x0ne_sdk::abi::{call, OP_OPEN, OP_SEAL_CHAT, OP_VERIFY_KEY};
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x52; 32];
    let http = reqwest::Client::new();
    let doc: Value = http
        .get(format!("{}/v2/e2e/key?nonce={}", r.url, hex(&nonce)))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let col: Value = http
        .get(format!("{}/v2/e2e/collateral", r.url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let pol = abi_policy(json!({
        "machine_type": MACHINE,
        "model_digest_hex": r.digest,
        "root_der_b64": b64(&r.intel.root_der),
        "gce_root_der_b64": b64(&google_root()),
        "attest": "cpu",
    }));
    let run = |op: u32, v: Value| call(op, &serde_json::to_vec(&v).unwrap());
    let v = run(
        OP_VERIFY_KEY,
        json!({ "document": doc, "collateral": col, "policy": pol,
                "client_nonce_hex": hex(&nonce), "now": now() }),
    );
    assert_eq!(v["ok"], true, "{v}");
    // The relay's disclosure and the reference's provenance, as the caller
    // shows them.
    let shown = &v["verified"];
    assert_eq!(
        shown["tcb"]["schema"], "x0ne-infer/tcb-disclosure/v1",
        "{v}"
    );
    assert_eq!(shown["tcb"]["policy"], "transparent", "{v}");
    assert_eq!(shown["reference"]["signed_by"]["test_key"], true, "{v}");
    assert_eq!(
        shown["reference"]["log"]["origin"], "test.invalid/x0ne-engine/tlog",
        "{v}"
    );
    let s = run(
        OP_SEAL_CHAT,
        json!({ "key": v["key"], "body": serde_json::from_slice::<Value>(&question()).unwrap(),
                "ikm_hex": hex(&[0x62; 32]), "now": now() }),
    );
    assert_eq!(s["ok"], true, "{s}");
    let request = base64::engine::general_purpose::STANDARD
        .decode(s["request_b64"].as_str().unwrap())
        .unwrap();
    let (status, response) = send_sealed(&r, request).await;
    assert_eq!(status, 200);
    let o = run(
        OP_OPEN,
        json!({ "pending": s["pending"], "response_b64": b64(&response), "now": now() }),
    );
    assert_eq!(o["ok"], true, "{o}");
    assert_eq!(o["receipt"]["checked"], true);
    assert!(o["text"].as_str().unwrap().contains("Lyon-Perrache"));
    // What the receipt does not sign is apart: the body, under `unverified`.
    assert!(
        o.get("body").is_none(),
        "the body beside the signed text: {o}"
    );
    assert!(o["unverified"]["body"]["choices"].is_array(), "{o}");
    // A pending handle opens once.
    let again = run(
        OP_OPEN,
        json!({ "pending": s["pending"], "response_b64": b64(&response), "now": now() }),
    );
    assert_eq!(again["ok"], false);
    // A question the TD refuses has no receipt: an error, never a result;
    // its status and body are the TD's word alone, under `unverified`.
    let refused = run(
        OP_SEAL_CHAT,
        json!({ "key": v["key"], "ikm_hex": hex(&[0x64; 32]), "now": now(),
                "body": { "messages": [{ "role": "user", "content": "x" }],
                          "response_format": { "type": "json_schema", "json_schema": {
                              "name": "r", "schema": { "type": "string", "pattern": "^a" } } } } }),
    );
    assert_eq!(refused["ok"], true, "{refused}");
    let request = base64::engine::general_purpose::STANDARD
        .decode(refused["request_b64"].as_str().unwrap())
        .unwrap();
    let (status, response) = send_sealed(&r, request).await;
    assert_eq!(status, 200);
    let o = run(
        OP_OPEN,
        json!({ "pending": refused["pending"], "response_b64": b64(&response), "now": now() }),
    );
    assert_eq!(o["ok"], false, "{o}");
    assert!(
        o["error"].as_str().unwrap().contains("no checked receipt"),
        "{o}"
    );
    assert_eq!(o["unverified"]["status"], 400, "{o}");
    assert!(o["unverified"]["body"]["error"].is_object(), "{o}");
    for signed in ["text", "receipt", "status", "body"] {
        assert!(
            o.get(signed).is_none(),
            "{signed} outside `unverified`: {o}"
        );
    }
    // Handles are closed by the caller (op 4), a
    // pending request whose answer never came as a key it is done with; the
    // 1024 slots are freed, and a closed handle serves nothing.
    let seal = || {
        run(
            OP_SEAL_CHAT,
            json!({ "key": v["key"], "body": serde_json::from_slice::<Value>(&question()).unwrap(),
                    "ikm_hex": hex(&[0x63; 32]), "now": now() }),
        )
    };
    let unanswered = seal();
    assert_eq!(unanswered["ok"], true, "{unanswered}");
    let closed = run(4, json!({ "handle": unanswered["pending"] }));
    assert_eq!(closed["ok"], true, "{closed}");
    let o = run(
        OP_OPEN,
        json!({ "pending": unanswered["pending"], "response_b64": b64(&response), "now": now() }),
    );
    assert!(
        o["error"].as_str().unwrap().contains("no pending request"),
        "{o}"
    );
    let closed = run(4, json!({ "handle": v["key"] }));
    assert_eq!(closed["ok"], true, "{closed}");
    let after = seal();
    assert!(
        after["error"].as_str().unwrap().contains("no verified key"),
        "{after}"
    );
    let twice = run(4, json!({ "handle": v["key"] }));
    assert_eq!(twice["ok"], false, "{twice}");
}

/// SECURITY: a panic in the C ABI drops every handle of the instance: the
/// key verified before it seals nothing more, its pending request opens
/// nothing, and the key verified after it gets a number never issued
/// before. In a process of its own (`X0NE_ABI_PANIC_CHILD`): the other
/// tests of this binary share the instance. A failure: a key or a request
/// served after a panic that may have left it half changed.
#[test]
fn a_panic_in_the_c_abi_drops_every_handle() {
    use x0ne_sdk::abi::{
        call, x0ne_call, x0ne_result_ptr, CODE_PANIC, OP_OPEN, OP_SEAL_CHAT, OP_TEST_PANIC,
        OP_VERIFY_KEY,
    };
    if std::env::var_os("X0NE_ABI_PANIC_CHILD").is_none() {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "e2e_sdk_tests::a_panic_in_the_c_abi_drops_every_handle",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("X0NE_ABI_PANIC_CHILD", "1")
            .output()
            .unwrap();
        let journal = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.status.success(), "{journal}");
        assert!(journal.contains("child: handles dropped"), "{journal}");
        return;
    }
    let run = |op: u32, v: Value| call(op, &serde_json::to_vec(&v).unwrap());
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let r = bench(Lie::None, &[]).await;
        let http = reqwest::Client::new();
        let verify = |nonce: [u8; 32]| {
            let (http, url, digest, root) = (
                http.clone(),
                r.url.clone(),
                r.digest.clone(),
                b64(&r.intel.root_der),
            );
            async move {
                let get = |path: String| {
                    let http = http.clone();
                    async move {
                        http.get(path)
                            .send()
                            .await
                            .unwrap()
                            .json::<Value>()
                            .await
                            .unwrap()
                    }
                };
                let doc = get(format!("{url}/v2/e2e/key?nonce={}", hex(&nonce))).await;
                let col = get(format!("{url}/v2/e2e/collateral")).await;
                let pol = abi_policy(json!({
                    "machine_type": MACHINE,
                    "model_digest_hex": digest,
                    "root_der_b64": root,
                    "gce_root_der_b64": b64(&google_root()),
                    "attest": "cpu",
                }));
                call(
                    OP_VERIFY_KEY,
                    &serde_json::to_vec(&json!({ "document": doc, "collateral": col,
                        "policy": pol, "client_nonce_hex": hex(&nonce), "now": now() }))
                    .unwrap(),
                )
            }
        };
        let v = verify([0x54; 32]).await;
        assert_eq!(v["ok"], true, "{v}");
        let s = run(
            OP_SEAL_CHAT,
            json!({ "key": v["key"], "body": serde_json::from_slice::<Value>(&question()).unwrap(),
                    "ikm_hex": hex(&[0x65; 32]), "now": now() }),
        );
        assert_eq!(s["ok"], true, "{s}");
        // The panic, through the export a host calls.
        // SAFETY: a null input of length 0 is the ABI's empty input.
        let n = unsafe { x0ne_call(OP_TEST_PANIC, std::ptr::null(), 0) };
        // SAFETY: n bytes at x0ne_result_ptr until this thread's next call.
        let out: Value =
            serde_json::from_slice(unsafe { std::slice::from_raw_parts(x0ne_result_ptr(), n) })
                .unwrap();
        assert_eq!(out["code"], CODE_PANIC, "{out}");
        let sealed = run(
            OP_SEAL_CHAT,
            json!({ "key": v["key"], "body": serde_json::from_slice::<Value>(&question()).unwrap(),
                    "ikm_hex": hex(&[0x66; 32]), "now": now() }),
        );
        assert!(
            sealed["error"]
                .as_str()
                .unwrap()
                .contains("no verified key"),
            "{sealed}"
        );
        let opened = run(
            OP_OPEN,
            json!({ "pending": s["pending"], "response_b64": "", "now": now() }),
        );
        assert!(
            opened["error"]
                .as_str()
                .unwrap()
                .contains("no pending request"),
            "{opened}"
        );
        // The instance serves again, under numbers never issued before.
        let again = verify([0x55; 32]).await;
        assert_eq!(again["ok"], true, "{again}");
        assert!(
            again["key"].as_u64() > s["pending"].as_u64(),
            "{again} after {s}"
        );
    });
    eprintln!("child: handles dropped");
}

/// Sizes are capped as the relay caps them, before
/// anything is decoded: a collateral field over 4 MiB (`remote_verify.rs`)
/// and the quote likewise, the event log over 16 MiB (`ccel::MAX_LOG_BYTES`),
/// each refused at the step that reads it.
#[tokio::test]
async fn a_field_over_the_relays_cap_is_refused_at_its_step() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x5d; 32];
    let doc = r.bench.td.key_document(&nonce);
    let p = policy(&r.intel);
    let over = |cap: usize| "A".repeat(cap / 3 * 4 + 8);
    let mib = 1024 * 1024;
    let docs = [
        (
            "quote",
            KeyDocument {
                quote_b64: over(4 * mib),
                ..doc.clone()
            },
        ),
        (
            "measurement",
            KeyDocument {
                ccel_b64: over(16 * mib),
                ..doc.clone()
            },
        ),
    ];
    for (step, d) in docs {
        let e = attest::verify_key(&d, &r.bench.collateral, &p, &nonce, now()).unwrap_err();
        assert_eq!(e.step, step, "{e}");
        assert!(e.reason.contains("cap"), "{e}");
    }
    let c = &r.bench.collateral;
    let big = over(4 * mib);
    let cols = [
        (
            "tcb",
            Collateral {
                tcb_info: big.clone(),
                ..c.clone()
            },
        ),
        (
            "tcb",
            Collateral {
                tcb_info_chain_pem: big.clone(),
                ..c.clone()
            },
        ),
        (
            "tcb",
            Collateral {
                qe_identity: big.clone(),
                ..c.clone()
            },
        ),
        (
            "tcb",
            Collateral {
                qe_identity_chain_pem: big.clone(),
                ..c.clone()
            },
        ),
        (
            "tcb",
            Collateral {
                tcb_info_standard: big.clone(),
                ..c.clone()
            },
        ),
        (
            "tcb",
            Collateral {
                tcb_info_standard_chain_pem: big.clone(),
                ..c.clone()
            },
        ),
        (
            "tcb",
            Collateral {
                qe_identity_standard: big.clone(),
                ..c.clone()
            },
        ),
        (
            "tcb",
            Collateral {
                qe_identity_standard_chain_pem: big.clone(),
                ..c.clone()
            },
        ),
        (
            "revocation",
            Collateral {
                pck_crl_der_b64: big.clone(),
                ..c.clone()
            },
        ),
        (
            "revocation",
            Collateral {
                root_ca_crl_der_b64: big.clone(),
                ..c.clone()
            },
        ),
        (
            "measurement",
            Collateral {
                gce_endorsement_b64: big.clone(),
                ..c.clone()
            },
        ),
    ];
    for (step, col) in cols {
        let e = attest::verify_key(&doc, &col, &p, &nonce, now()).unwrap_err();
        assert_eq!(e.step, step, "{e}");
        assert!(e.reason.contains("cap"), "{e}");
    }
    attest::verify_key(&doc, c, &p, &nonce, now()).expect("control");
}

// ── The caller's refusals ────────────────────────────────────────────────────

async fn refused(lie: Lie) -> attest::Refusal {
    let r = bench(lie, &[]).await;
    match verified(&r, [0x53; 32]).await {
        Ok(_) => panic!("{lie:?}: a lying TD's key was accepted"),
        Err(refusal) => refusal,
    }
}

/// SECURITY: a carrier that puts its own OHTTP key next to the TD's quote
/// would read every question. The quote binds the key configuration: refused.
#[tokio::test]
async fn a_carrier_key_next_to_the_tds_quote_is_refused() {
    assert_eq!(refused(Lie::CarrierKey).await.step, "binding");
}

/// SECURITY: receipts are checked under the receipt key the quote bound,
/// never under one the document names on its own.
#[tokio::test]
async fn a_receipt_key_the_quote_did_not_bind_is_refused() {
    assert_eq!(refused(Lie::ReceiptKey).await.step, "binding");
}

/// SECURITY: a quote taken for another caller, or replayed from an earlier
/// exchange, does not answer this caller's nonce.
#[tokio::test]
async fn a_quote_that_answers_another_nonce_is_refused() {
    assert_eq!(refused(Lie::OtherNonce).await.step, "binding");
}

/// One TD under `config`, its key document for `nonce`, what verifies it.
fn session_case(config: &Loaded, nonce: [u8; 32]) -> (KeyDocument, Collateral, Policy) {
    let intel = Arc::new(FakeIntel::new());
    let identity = SessionIdentity::generate([7; 32], [9; 32]);
    let td = Td::under(intel.clone(), 5, &identity, Lie::None, config);
    let doc = td.key_document(&nonce);
    (doc, collateral(&intel, &[]), policy(&intel))
}

/// SECURITY (step 4 of docs/sdk-e2e.md, 2026-10-30): the key document names
/// the session config the TD serves under (the model, when it ends, the
/// relay's TLS key) and the TD's boot binding (its boot nonce, its TLS
/// key). A carrier that swaps any of them next to the TD's quote, to show
/// the caller another model, a later end or its own relay key, is refused
/// at `binding`: the quote names the config's bytes, not a parse of them.
#[test]
fn a_key_document_naming_another_session_config_or_boot_is_refused_at_binding() {
    let nonce = [0x5f; 32];
    let served = session_config("cpu");
    let (honest, col, p) = session_case(&served, nonce);
    let key = attest::verify_key(&honest, &col, &p, &nonce, now()).expect("control");
    // The SDK's copy of the type (one source, two builds): compared as JSON.
    assert_eq!(
        serde_json::to_value(&key.session_config).unwrap(),
        serde_json::to_value(&served.config).unwrap()
    );
    assert_eq!(key.session_config.model, MODEL);
    assert_eq!(key.session_config.client_spki_sha256, hex(&RELAY_SPKI));
    assert_eq!(
        key.session_config.expires_at_unix,
        served.config.expires_at_unix
    );
    assert_eq!(key.session_config_sha256_hex, hex(&served.sha256));
    assert_eq!(key.boot_nonce_hex, hex(&[9; 32]));
    assert_eq!(key.tls_spki_sha256_hex, hex(&TD_TLS_SPKI));
    let swapped = |f: &dyn Fn(&mut CloudConfig)| {
        let mut c = served.config.clone();
        f(&mut c);
        KeyDocument {
            session_config_b64: b64(&serde_json::to_vec(&c).unwrap()),
            ..honest.clone()
        }
    };
    let mut respaced = served.bytes.clone();
    respaced.push(b'\n');
    for (doc, what) in [
        (
            swapped(&|c| c.client_spki_sha256 = hex(&[0x66; 32])),
            "the carrier's TLS key as the relay's",
        ),
        (swapped(&|c| c.model = "qwen3:8b".into()), "another model"),
        (swapped(&|c| c.expires_at_unix += 86_400), "a later expiry"),
        (
            KeyDocument {
                session_config_b64: b64(&respaced),
                ..honest.clone()
            },
            "the same config, other bytes",
        ),
        (
            KeyDocument {
                boot_nonce_hex: hex(&[0x99; 32]),
                ..honest.clone()
            },
            "another boot nonce",
        ),
        (
            KeyDocument {
                tls_spki_sha256_hex: hex(&[0x98; 32]),
                ..honest.clone()
            },
            "another TLS key",
        ),
    ] {
        match attest::verify_key(&doc, &col, &p, &nonce, now()) {
            Ok(k) => panic!(
                "{what}: accepted, the caller reads {:?} until {}",
                k.session_config.model, k.session_config.expires_at_unix
            ),
            Err(e) => assert_eq!(e.step, "binding", "{what}: {e}"),
        }
    }
}

/// A key document without its session config or boot binding (a TD before
/// v2 that relabelled its schema, or a carrier that strips them), or with
/// a config that is not one, is refused at `key`, by the field's name.
#[test]
fn a_key_document_without_a_readable_session_config_is_refused_at_key() {
    let nonce = [0x60; 32];
    let (honest, col, p) = session_case(&session_config("cpu"), nonce);
    let mut other_schema = session_config("cpu").config;
    other_schema.schema = "x0ne-engine/cloud-config/v2".into();
    for (doc, says) in [
        (
            KeyDocument {
                session_config_b64: String::new(),
                ..honest.clone()
            },
            "session_config_b64",
        ),
        (
            KeyDocument {
                session_config_b64: b64(b"{}"),
                ..honest.clone()
            },
            "not a session config",
        ),
        (
            KeyDocument {
                session_config_b64: b64(&serde_json::to_vec(&other_schema).unwrap()),
                ..honest.clone()
            },
            "cloud-config/v2",
        ),
        (
            KeyDocument {
                boot_nonce_hex: String::new(),
                ..honest.clone()
            },
            "boot_nonce_hex",
        ),
        (
            KeyDocument {
                tls_spki_sha256_hex: "aa".repeat(31),
                ..honest.clone()
            },
            "tls_spki_sha256_hex",
        ),
    ] {
        let e = attest::verify_key(&doc, &col, &p, &nonce, now()).unwrap_err();
        assert_eq!(e.step, "key", "{says}: {e}");
        assert!(e.reason.contains(says), "{says}: {e}");
    }
    let v1 = KeyDocument {
        schema: "x0ne-e2e/key/v1".into(),
        ..honest
    };
    let e = attest::verify_key(&v1, &col, &p, &nonce, now()).unwrap_err();
    assert_eq!(e.step, "schema", "{e}");
}

/// The relay takes no new request within `EXPIRY_GUARD_SECS` of the
/// session's end; the caller reads the same end, bound by the quote, and
/// stops at the same edge: refused at `session` to verify, and to seal.
#[test]
fn a_session_about_to_end_is_refused_before_anything_is_sealed() {
    assert_eq!(
        attest::SESSION_EXPIRY_GUARD_SECS,
        crate::local_api::EXPIRY_GUARD_SECS
    );
    let nonce = [0x61; 32];
    let t = now();
    let ends = t + 100;
    let config = CloudConfig::new(MODEL, RELAY_SPKI, [0x43; 32], ends, "cpu", 443, 64)
        .to_loaded()
        .unwrap();
    let (doc, col, p) = session_case(&config, nonce);
    let edge = ends - attest::SESSION_EXPIRY_GUARD_SECS;
    let e = attest::verify_key(&doc, &col, &p, &nonce, edge).unwrap_err();
    assert_eq!(e.step, "session", "{e}");
    assert!(e.reason.contains(&format!("expires at {ends}")), "{e}");
    let client = Client::new(attest::verify_key(&doc, &col, &p, &nonce, t).expect("control"));
    client
        .seal_chat(&question(), &[0x65; 32], edge - 1)
        .expect("a second before the edge");
    match client.seal_chat(&question(), &[0x66; 32], edge) {
        Err(E2eError::Refused(why)) => assert!(why.contains("expires at"), "{why}"),
        other => panic!("sealed at the edge: {:?}", other.map(|_| ())),
    }
}

/// The session config the quote bound asks for what the policy attests: a
/// TD booted under `attest=cpu+gpu` that serves a CPU-only key document is
/// refused at `session` by a CPU caller.
#[test]
fn a_session_config_that_attests_other_than_the_policy_is_refused() {
    let nonce = [0x62; 32];
    let (doc, col, p) = session_case(&session_config("cpu+gpu"), nonce);
    let e = attest::verify_key(&doc, &col, &p, &nonce, now()).unwrap_err();
    assert_eq!(e.step, "session", "{e}");
    assert!(e.reason.contains("attest=cpu+gpu"), "{e}");
}

#[tokio::test]
async fn a_debug_td_is_refused() {
    assert_eq!(refused(Lie::Debug).await.step, "quote");
}

/// SECURITY: a quote whose RTMR2 is not what its event log replays to is
/// refused with no pin set: the log must explain the quote. RTMR0 drift,
/// logged and quoted alike (a host table the reference does not pin), is
/// refused too, at the same step with the relay's 65: RTMR0 is pinned to
/// the reference's host events, not reported (until 2026-10-31 it was).
#[tokio::test]
async fn an_rtmr_drift_is_refused_and_so_is_an_rtmr0_drift() {
    let e = refused(Lie::RtmrDrift).await;
    assert_eq!(e.step, "measurement", "{e}");
    assert!(e.reason.contains("CCEL replay of RTMR2"), "{e}");
    let e = refused(Lie::Rtmr0Drift).await;
    assert_eq!((e.step, e.code), ("measurement", 65), "{e}");
    assert!(e.reason.contains("RTMR0 event 0"), "{e}");
}

#[tokio::test]
async fn a_measurement_other_than_the_pinned_one_is_refused() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x55; 32];
    let doc = r.bench.td.key_document(&nonce);
    let col = r.bench.collateral.clone();
    attest::verify_key(&doc, &col, &pinned(&r.intel), &nonce, now()).expect("the image's own pins");
    let mut p = pinned(&r.intel);
    p.mrtd_pin = Some([0x12; 48]);
    let e = attest::verify_key(&doc, &col, &p, &nonce, now()).unwrap_err();
    assert_eq!(e.step, "measurement");
    for i in 0..3 {
        let mut p = pinned(&r.intel);
        p.rtmr_pins[i] = Some([0x77; 48]);
        let e = attest::verify_key(&doc, &col, &p, &nonce, now()).unwrap_err();
        assert_eq!(e.step, "measurement", "RTMR{}", i + 1);
    }
}

/// A quote under a root other than the pinned one: another "Intel".
#[tokio::test]
async fn a_quote_under_another_root_is_refused() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x56; 32];
    let doc = r.bench.td.key_document(&nonce);
    let other = FakeIntel::new();
    let e =
        attest::verify_key(&doc, &r.bench.collateral, &policy(&other), &nonce, now()).unwrap_err();
    assert_eq!(e.step, "quote");
}

// ── The relay's TCB policy and the reference's admission, caller side ───────

/// Intel's two answers for the GCE H100 hosts of 2026-10-28 on this bench's
/// SVNs (`FakeIntel::tcb_info_of_2026_10_28`): UpToDate on `standard`
/// (evaluation 20), OutOfDate on `early` (evaluation 22, three advisories).
fn collateral_of_2026_10_28(intel: &FakeIntel) -> Collateral {
    let (standard, early) = intel.tcb_info_of_2026_10_28(fresh(now()).0);
    Collateral {
        tcb_info: early,
        tcb_info_standard: standard,
        ..collateral(intel, &[])
    }
}

/// SECURITY: under `transparent`, the relay's default, a host Intel's
/// `early` feed rates OutOfDate is accepted on `standard` and named, as the
/// relay names it (`x0ne-infer/tcb-disclosure/v1`): the policy, the
/// collateral accepted on, Intel's latest view and its advisories. Under
/// `strict` the same evidence is refused at `tcb`, with the relay's 76.
#[tokio::test]
async fn an_out_of_date_host_under_early_is_named_under_transparent_and_refused_under_strict() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x5a; 32];
    let doc = r.bench.td.key_document(&nonce);
    let col = collateral_of_2026_10_28(&r.intel);
    let key = attest::verify_key(&doc, &col, &policy(&r.intel), &nonce, now())
        .expect("transparent accepts on standard");
    assert_eq!(key.tcb.schema, x0ne_sdk::TCB_DISCLOSURE_SCHEMA);
    assert_eq!(key.tcb.policy, "transparent");
    let intel = key.tcb.intel.as_ref().expect("an Intel verdict");
    assert_eq!(
        (intel.accepted_under.as_str(), intel.accepted_evaluation),
        ("standard", STANDARD_EVALUATION)
    );
    assert_eq!(intel.platform, "UpToDate");
    let latest = intel.latest.as_ref().expect("Intel's early view, named");
    assert_eq!(
        (latest.platform.as_str(), latest.tcb_evaluation),
        (
            "OutOfDate",
            crate::tcb::TcbPolicy::MIN_TCB_EVALUATION_DATA_NUMBER
        )
    );
    assert_eq!(
        latest.advisories,
        crate::attest_v2::fixtures::ADVISORIES_2026_10_28
    );
    assert!(key.tcb.line.contains("OutOfDate"), "{}", key.tcb.line);

    let strict = policy(&r.intel).with_strict_tcb();
    let e = attest::verify_key(&doc, &col, &strict, &nonce, now()).unwrap_err();
    assert_eq!((e.step, e.code), ("tcb", 76), "{e}");
    assert!(e.reason.contains("OutOfDate"), "{e}");
    // The control: strict on an UpToDate `early`.
    let key = attest::verify_key(&doc, &r.bench.collateral, &strict, &nonce, now())
        .expect("strict on an UpToDate host");
    assert_eq!(key.tcb.policy, "strict");
    assert_eq!(key.tcb.intel.unwrap().accepted_under, "early");
}

/// A carrier that withholds Intel's `early` collateral: `transparent` accepts
/// on `standard` and says the view is missing, as the relay does when its
/// fetch fails; `strict` refuses. Withheld `standard`: both refuse, as
/// neither accepts on what it was not given.
#[tokio::test]
async fn early_withheld_is_named_under_transparent_and_refused_under_strict() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x5b; 32];
    let doc = r.bench.td.key_document(&nonce);
    let no_early = Collateral {
        tcb_info: String::new(),
        tcb_info_chain_pem: String::new(),
        qe_identity: String::new(),
        qe_identity_chain_pem: String::new(),
        ..r.bench.collateral.clone()
    };
    let key = attest::verify_key(&doc, &no_early, &policy(&r.intel), &nonce, now())
        .expect("transparent, early withheld");
    let intel = key.tcb.intel.unwrap();
    assert!(intel.latest.is_none());
    assert!(
        intel
            .latest_unavailable
            .as_deref()
            .is_some_and(|w| w.contains("not relayed")),
        "{intel:?}"
    );
    let strict = policy(&r.intel).with_strict_tcb();
    let e = attest::verify_key(&doc, &no_early, &strict, &nonce, now()).unwrap_err();
    assert_eq!((e.step, e.code), ("tcb", 76), "{e}");
    let no_standard = Collateral {
        tcb_info_standard: String::new(),
        tcb_info_standard_chain_pem: String::new(),
        qe_identity_standard: String::new(),
        qe_identity_standard_chain_pem: String::new(),
        ..r.bench.collateral.clone()
    };
    let e = attest::verify_key(&doc, &no_standard, &policy(&r.intel), &nonce, now()).unwrap_err();
    assert_eq!((e.step, e.code), ("tcb", 76), "{e}");
    assert!(e.reason.contains("update=standard"), "{e}");
    attest::verify_key(&doc, &no_standard, &strict, &nonce, now())
        .expect("strict reads early alone");
}

/// SECURITY: under `strict`, the answer's receipt must name Intel's latest
/// view as the relay requires it (`tcb_disclosure::strict_receipt_check`):
/// this bench's chat signs receipts without one, so the answer is refused.
/// Under `transparent` the same answer opens.
#[tokio::test]
async fn under_strict_an_answer_whose_receipt_names_no_tcb_view_is_refused() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x5e; 32];
    let http = reqwest::Client::new();
    let doc: KeyDocument = http
        .get(format!("{}/v2/e2e/key?nonce={}", r.url, hex(&nonce)))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let strict = policy(&r.intel).with_strict_tcb();
    let client = Client::new(
        attest::verify_key(&doc, &r.bench.collateral, &strict, &nonce, now())
            .expect("strict on an UpToDate host"),
    );
    let (request, pending) = client.seal_chat(&question(), &[0x64; 32], now()).unwrap();
    let (status, response) = send_sealed(&r, request).await;
    assert_eq!(status, 200);
    let e = pending
        .open_verified(&response, &r.digest, now())
        .unwrap_err();
    assert!(e.to_string().contains("strict TCB policy: "), "{e}");
    assert!(e.to_string().contains("no TCB disclosure"), "{e}");

    let client = verified(&r, [0x5e; 32]).await.unwrap();
    let (request, pending) = client.seal_chat(&question(), &[0x65; 32], now()).unwrap();
    let (_, response) = send_sealed(&r, request).await;
    pending
        .open_verified(&response, &r.digest, now())
        .expect("transparent: the receipt checks");
}

/// The reference of [`reference_json`] with another kernel: another image.
fn other_image_json() -> String {
    let mut v: Value = serde_json::from_str(&reference_json()).unwrap();
    v["image"]["version"] = json!("0.1.1");
    v["rtmr2"]["events_sha384"][".linux"] = json!(hex(&[0xc1; 48]));
    v.to_string()
}

/// SECURITY: the caller holds the reference of the image it accepts, signed
/// and logged. A key document from a TD of another image (its kernel not on
/// the reference's list) is refused at `measurement`, the relay's 65, with
/// no pin set.
#[tokio::test]
async fn a_key_document_from_another_image_is_refused() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x59; 32];
    let doc = r.bench.td.key_document(&nonce);
    let other = Policy {
        anchor: x0ne_sdk::TrustAnchor::RootDer(r.intel.root_der.clone()),
        gce_root_der: google_root(),
        // A simulated TD serves on the CPU: the tests' `attest=cpu` (x0ne-sdk `test-anchor`).
        attest: x0ne_sdk::attest::Attest::Cpu,
        ..Policy::production(admitted(other_image_json().as_bytes()), MACHINE)
    };
    let e = attest::verify_key(&doc, &r.bench.collateral, &other, &nonce, now()).unwrap_err();
    assert_eq!((e.step, e.code), ("measurement", 65), "{e}");
    assert!(e.reason.contains(".linux"), "{e}");
    attest::verify_key(&doc, &r.bench.collateral, &policy(&r.intel), &nonce, now())
        .expect("the control: the TD's own image");
}

fn admit(
    json: &[u8],
    signature: &[u8],
    proof: &str,
) -> Result<attest::AdmittedReference, attest::Refusal> {
    attest::admit_reference_under_test_keys(
        &attest::SignedReference {
            json,
            signature,
            tlog_proof: proof,
        },
        &mut attest::LogMemory::new(),
        None,
        now(),
    )
}

/// SECURITY: the reference is admitted as the relay admits it
/// (`reference::admit_signed`, the same code): without a valid signature by
/// the pinned archive key, refused with 99, before its log proof is read.
/// A carrier that hands the caller another reference gets nothing past it.
#[test]
fn a_reference_unsigned_or_signed_over_other_bytes_is_refused_with_99() {
    let json = reference_json();
    let other = other_image_json();
    let (signature, proof) = signed(json.as_bytes(), now());
    let (other_signature, _) = signed(other.as_bytes(), now());
    let mut flipped = signature.clone();
    let n = flipped.len();
    flipped[n - 3] ^= 1;
    for (what, sig) in [
        ("no signature", Vec::new()),
        ("another file's signature", other_signature),
        ("one bit flipped", flipped),
    ] {
        let e = admit(json.as_bytes(), &sig, &proof).unwrap_err();
        assert_eq!((e.step, e.code), ("reference", 99), "{what}: {e}");
    }
    let a = admit(json.as_bytes(), &signature, &proof).expect("the control");
    assert_eq!(a.provenance()["signed_by"]["test_key"], true);
}

/// SECURITY: signed but not shown in the pinned log, under a fresh quorum
/// of its witnesses: refused with 100, as the relay refuses it. A proof
/// for another file, none, or a checkpoint older than the log's freshness
/// window (this verifier reads no log to refresh it).
#[test]
fn a_reference_outside_the_log_or_under_a_stale_checkpoint_is_refused_with_100() {
    let json = reference_json();
    let (signature, proof) = signed(json.as_bytes(), now());
    let (_, other_proof) = signed(other_image_json().as_bytes(), now());
    let eleven_years = 11 * 365 * 86_400;
    let stale = crate::transparency::test_log::proof_of_one(json.as_bytes(), now() - eleven_years);
    for (what, p, says) in [
        ("another file's proof", other_proof.as_str(), ""),
        ("no proof", "", ""),
        (
            "a stale checkpoint",
            stale.as_str(),
            "reads no log to refresh it",
        ),
    ] {
        let e = admit(json.as_bytes(), &signature, p).unwrap_err();
        assert_eq!((e.step, e.code), ("reference", 100), "{what}: {e}");
        assert!(e.reason.contains(says), "{what}: {e}");
    }
    let a = admit(json.as_bytes(), &signature, &proof).expect("the control");
    assert_eq!(a.provenance()["log"]["size"], 1);
}

/// The SDK admits by the relay's own function: one `admit_signed` in
/// `reference.rs`, called by the relay's loading and the SDK's admission.
#[test]
fn the_sdk_admits_the_reference_by_the_relays_function() {
    let src = include_str!("reference.rs");
    assert_eq!(src.matches("pub(crate) fn admit_signed(").count(), 1);
    assert_eq!(
        src.matches("admit_signed(").count(),
        3,
        "the definition and its two callers"
    );
    let sdk = include_str!("../../x0ne-sdk/src/attest.rs");
    assert!(sdk.contains("crate::reference::admit_offline("));
    assert!(!sdk.contains("fn parse_reference"));
}

/// Compatibility: what a caller built before 2026-10-28 (its `KeyDocument`
/// and `Collateral`, both `deny_unknown_fields`) does with what this TD and
/// this relay serve. A key document, CPU or GPU, names the session config
/// (v2, 2026-10-30) and the collateral names Intel's `standard` answers,
/// each refused by its field's name: never read under the older, looser
/// rules. A caller of 2026-10-28 to 2026-10-30 refuses the same way.
#[test]
fn a_caller_from_before_refuses_a_v2_key_document_by_name() {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    #[allow(dead_code)]
    struct OldKeyDocument {
        schema: String,
        key_config_b64: String,
        session_pubkey_hex: String,
        quote_b64: String,
        ccel_b64: String,
        #[serde(default)]
        gpu_evidence_list: Option<Vec<Value>>,
    }
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    #[allow(dead_code)]
    struct OldCollateral {
        tcb_info: String,
        tcb_info_chain_pem: String,
        qe_identity: String,
        qe_identity_chain_pem: String,
        pck_crl_der_b64: String,
        root_ca_crl_der_b64: String,
        gce_endorsement_b64: String,
        #[serde(default)]
        nras_answer: String,
        #[serde(default)]
        nras_jwks: String,
    }
    let intel = FakeIntel::new();
    let td = Td::new(
        Arc::new(FakeIntel::new()),
        5,
        &SessionIdentity::generate([7; 32], [9; 32]),
        Lie::None,
    );
    let cpu = serde_json::to_string(&td.key_document(&[0x58; 32])).unwrap();
    assert!(!cpu.contains("nras"), "{cpu}");
    let c = gpu_case(GpuLie::None, NrasLie::None);
    let answer: Value = serde_json::from_str(&c.collateral.nras_answer).unwrap();
    let jwks: Value = serde_json::from_str(&c.collateral.nras_jwks).unwrap();
    let gpu = E2e::with_nras(c.doc, crate::attest_v2::NrasVerdict { answer, jwks });
    let gpu = serde_json::to_string(&gpu).unwrap();
    for (doc, what) in [(&cpu, "CPU"), (&gpu, "GPU")] {
        let e = serde_json::from_str::<OldKeyDocument>(doc)
            .err()
            .unwrap_or_else(|| panic!("an older caller read a {what} document without its config"));
        assert!(
            e.to_string().contains("unknown field `session_config_b64`"),
            "{what}: {e}"
        );
        let mut v1: Value = serde_json::from_str(doc).unwrap();
        for field in [
            "session_config_b64",
            "boot_nonce_hex",
            "tls_spki_sha256_hex",
        ] {
            v1.as_object_mut().unwrap().remove(field);
        }
        v1["schema"] = json!("x0ne-e2e/key/v1");
        let doc: KeyDocument = serde_json::from_value(v1).unwrap();
        let e = attest::verify_key(
            &doc,
            &collateral(&intel, &[]),
            &policy(&intel),
            &[0x58; 32],
            now(),
        )
        .unwrap_err();
        assert_eq!(e.step, "schema", "{what}: {e}");
    }
    let col = serde_json::to_string(&collateral(&intel, &[])).unwrap();
    let e = serde_json::from_str::<OldCollateral>(&col)
        .err()
        .expect("an older caller read the collateral without its standard half");
    assert!(
        e.to_string().contains("unknown field `tcb_info_standard`"),
        "{e}"
    );
}

/// The collateral is relayed: a carrier can withhold it or bring another
/// root's, never forge Intel's signature. Both refused, and a revoked PCK.
#[tokio::test]
async fn relayed_collateral_withheld_forged_or_revoking_is_refused() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x57; 32];
    let doc = r.bench.td.key_document(&nonce);
    let p = policy(&r.intel);
    let check = |col: &Collateral| attest::verify_key(&doc, col, &p, &nonce, now());

    assert_eq!(check(&Collateral::default()).unwrap_err().step, "tcb");
    let other = FakeIntel::new();
    let forged = collateral(&other, &[]);
    assert_eq!(check(&forged).unwrap_err().step, "tcb");

    let mut crl_by_other = r.bench.collateral.clone();
    crl_by_other.pck_crl_der_b64 = b64(&other.pck_crl(&[], fresh(now())));
    assert_eq!(check(&crl_by_other).unwrap_err().step, "revocation");

    let revoked = collateral(&r.intel, &[r.intel.pck_serial()]);
    assert_eq!(check(&revoked).unwrap_err().step, "revocation");

    // Issued an hour before now: judged on the caller's clock two days on, it
    // is older than the production policy's day.
    let two_days_on = now() + 2 * 86_400;
    assert!(attest::verify_key(&doc, &r.bench.collateral, &p, &nonce, two_days_on).is_err());
}

/// SECURITY: the collateral is relayed, so its
/// carrier picks which genuine collateral to relay: anything still in its
/// window and under the relay's day. A caller can shorten that day for
/// itself (`Policy::with_max_collateral_age`), never lengthen it: a longer
/// one is refused when built and, set by hand, at `tcb`.
#[tokio::test]
async fn collateral_replayed_within_the_day_is_taken_unless_the_caller_shortens_it() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x5f; 32];
    let doc = r.bench.td.key_document(&nonce);
    let t = now();
    // Issued an hour before `t` itself: the bench's, an hour before the bench
    // started, read 3601 s old at `t` when a second passed (CI, 2026-10-27).
    let current = collateral_at(&r.intel, &[], t);
    let issued = t - 2 * 3600;
    let window = (issued, t + 30 * 86_400);
    let two_hours = Collateral {
        tcb_info: r.intel.tcb_info(issued),
        qe_identity: r.intel.qe_identity(issued),
        tcb_info_standard: r
            .intel
            .tcb_info_with(issued, STANDARD_EVALUATION, "UpToDate"),
        qe_identity_standard: r.intel.qe_identity_with(issued, STANDARD_EVALUATION),
        pck_crl_der_b64: b64(&r.intel.pck_crl(&[], window)),
        ..current.clone()
    };
    attest::verify_key(&doc, &two_hours, &policy(&r.intel), &nonce, t)
        .expect("two hours old: within the relay's day");
    let hour = policy(&r.intel).with_max_collateral_age(3600).unwrap();
    let e = attest::verify_key(&doc, &two_hours, &hour, &nonce, t).unwrap_err();
    assert_eq!(e.step, "tcb", "{e}");
    assert!(e.reason.contains("accepts at most 3600 s"), "{e}");
    let crl_alone = Collateral {
        pck_crl_der_b64: b64(&r.intel.pck_crl(&[], window)),
        ..current.clone()
    };
    let e = attest::verify_key(&doc, &crl_alone, &hour, &nonce, t).unwrap_err();
    assert_eq!(e.step, "revocation", "{e}");
    assert!(e.reason.contains("accepts at most 3600 s"), "{e}");
    attest::verify_key(&doc, &current, &hour, &nonce, t).expect("issued within the hour");

    let longer = policy(&r.intel).with_max_collateral_age(2 * 86_400);
    assert!(longer.is_err());
    // The SDK's field set by hand, past its builder (not a TcbPolicy: the
    // SDK builds that with `tightened`, which refuses it).
    let mut by_hand = policy(&r.intel);
    by_hand.max_collateral_age_secs = 2 * 86_400;
    let e = attest::verify_key(&doc, &r.bench.collateral, &by_hand, &nonce, t).unwrap_err();
    assert_eq!(e.step, "tcb", "{e}");
    assert!(e.reason.contains("shortened, not lengthened"), "{e}");
}

/// SECURITY: Intel's root CA CRL comes with the relayed collateral and is
/// read as the relay reads it (`tcb::verify_root_crl`), on the relay's cases
/// (`cloud_e2e_tests`): the PCK CA or the TCB signer listed is a revocation
/// (the relay's 65); withheld, expired, signed by another key or altered, an
/// unusable CRL (its 76). Each refused at `revocation`; a leaf listed is no
/// revocation of what the root issued.
#[tokio::test]
async fn a_root_ca_crl_withheld_stale_foreign_altered_or_revoking_is_refused() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x5c; 32];
    let doc = r.bench.td.key_document(&nonce);
    let p = policy(&r.intel);
    let intel = &r.intel;
    let t = now();
    let year = (t - 200 * 86_400, t + 165 * 86_400);
    let with = |crl: &[u8]| Collateral {
        root_ca_crl_der_b64: b64(crl),
        ..r.bench.collateral.clone()
    };
    let mut altered = intel.root_crl(&[], year);
    let n = altered.len();
    altered[n - 3] ^= 1;
    let cases: Vec<(&str, Vec<u8>, Option<&str>)> = vec![
        ("empty", intel.root_crl(&[], year), None),
        (
            "the PCK leaf listed",
            intel.root_crl(&[intel.pck_serial()], year),
            None,
        ),
        (
            "over the years",
            intel.root_crl(&[], years(2020, 2040)),
            None,
        ),
        (
            "PCK CA listed",
            intel.root_crl(&[vec![0x77; 16], intel.pck_ca_serial()], year),
            Some("is revoked: Intel's root CA CRL lists it"),
        ),
        (
            "TCB signer listed",
            intel.root_crl(&[intel.tcb_signing_serial()], year),
            Some("is revoked: Intel's root CA CRL lists it"),
        ),
        ("withheld", vec![], Some("root CA CRL: not a DER CRL")),
        (
            "expired",
            intel.root_crl(&[], (t - 400 * 86_400, t - 86_400)),
            Some("root CA CRL: validity window"),
        ),
        (
            "signed by the PCK CA",
            intel.root_crl_with(intel.pck_ca_key_der(), &[], year, |_| {}),
            Some("root CA CRL: signature does not verify"),
        ),
        ("one bit", altered, Some("root CA CRL")),
    ];
    for (what, crl, want) in cases {
        let got = attest::verify_key(&doc, &with(&crl), &p, &nonce, t);
        match (got, want) {
            (Ok(_), None) => {}
            (Err(e), Some(says)) => {
                assert_eq!(e.step, "revocation", "{what}: {e}");
                assert!(e.reason.contains(says), "{what}: {e}");
            }
            (Ok(_), Some(_)) => panic!("{what}: verified"),
            (Err(e), None) => panic!("{what}: refused: {e}"),
        }
    }
}

/// SECURITY: with no pin set, the MRTD is judged by Google's launch
/// endorsement and the RTMRs by the TD's log, as the relay judges them: an
/// endorsement withheld or of another firmware, the endorsed binary withheld
/// or another served in its place, a log withheld or one event changed, is
/// refused at `measurement`.
#[tokio::test]
async fn an_endorsement_or_a_log_withheld_or_not_the_tds_is_refused() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x5d; 32];
    let doc = r.bench.td.key_document(&nonce);
    let p = policy(&r.intel);
    let check =
        |doc: &KeyDocument, col: &Collateral| attest::verify_key(doc, col, &p, &nonce, now());
    check(&doc, &r.bench.collateral).expect("control: the honest TD");
    let endorsed = |blob: &[u8]| Collateral {
        gce_endorsement_b64: b64(blob),
        ..r.bench.collateral.clone()
    };
    let mut other_app = log();
    let first_app = other_app
        .iter()
        .position(|e| e.event_type == crate::ccel::EV_EFI_BOOT_SERVICES_APPLICATION)
        .expect("the log shows a boot application");
    other_app[first_app] = Event::new_sha384(
        2,
        crate::ccel::EV_EFI_BOOT_SERVICES_APPLICATION,
        [0xee; 48],
        b"shim".to_vec(),
    );
    let logged = |events: Option<&[Event]>| KeyDocument {
        ccel_b64: events
            .map(|e| b64(&crate::ccel::encode(e)))
            .unwrap_or_default(),
        ..doc.clone()
    };
    for (what, d, col, says) in [
        (
            "endorsement withheld",
            doc.clone(),
            endorsed(&[]),
            "has no Google launch endorsement",
        ),
        (
            "another firmware's endorsement",
            doc.clone(),
            endorsed(
                &crate::gce_endorsement::tests::fake_google::fake_google()
                    .endorse_binary(b"another firmware", &[[0x66; 48]]),
            ),
            "is not in the Google endorsement",
        ),
        (
            "the endorsed binary withheld",
            doc.clone(),
            Collateral {
                gce_firmware_b64: String::new(),
                ..r.bench.collateral.clone()
            },
            "was not relayed",
        ),
        (
            "another binary under the endorsed name",
            doc.clone(),
            Collateral {
                gce_firmware_b64: b64(&{
                    let mut t = crate::tdvf::tests::synthetic().to_vec();
                    t[0x10] ^= 1;
                    t
                }),
                ..r.bench.collateral.clone()
            },
            "is not the firmware Google endorsed",
        ),
        (
            "log withheld",
            logged(None),
            r.bench.collateral.clone(),
            "CCEL",
        ),
        (
            "another boot application logged",
            logged(Some(&other_app)),
            r.bench.collateral.clone(),
            "CCEL replay of RTMR1",
        ),
    ] {
        let e = check(&d, &col).unwrap_err();
        assert_eq!(e.step, "measurement", "{what}: {e}");
        assert!(e.reason.contains(says), "{what}: {e}");
    }
}

/// SECURITY: whoever holds one PCK's private key
/// (TEE.fail) issues a leaf of its own under it, signs the QE report with
/// it, and relays a CRL it signed with the PCK's key. Every signature holds;
/// the chain is one certificate longer than Intel's. Refused at `quote`.
#[tokio::test]
async fn a_pck_chain_extended_under_an_extracted_pck_key_is_refused() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x5e; 32];
    let honest = r.bench.td.key_document(&nonce);
    let report_data = r.bench.td.e2e.report_data(&nonce);
    let forged = r
        .intel
        .quote_under_extracted_pck(&report_data, &mrtd(), &rtmr(), [0; 8]);
    let doc = KeyDocument {
        quote_b64: b64(&forged),
        ..honest.clone()
    };
    let col = Collateral {
        pck_crl_der_b64: b64(&r.intel.crl_signed_by_the_pck(fresh(now()))),
        ..r.bench.collateral.clone()
    };
    let p = policy(&r.intel);
    attest::verify_key(&honest, &r.bench.collateral, &p, &nonce, now()).expect("control");
    let e = attest::verify_key(&doc, &col, &p, &nonce, now()).unwrap_err();
    assert_eq!(e.step, "quote", "{e}");
    assert!(e.reason.contains("chain has 4 certificate(s)"), "{e}");
}

/// SECURITY: a TCB signer issued by a PCK whose
/// key was extracted signs the relayed collateral; every signature holds.
/// Refused at `tcb` in the collateral accepted on, where only the root CRL
/// stopped it until then; in `early` under `transparent`, refused at
/// `revocation`, as the relay checks every chain it read against the root
/// CA's CRL.
#[tokio::test]
async fn collateral_signed_under_an_extracted_pck_key_is_refused() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x5f; 32];
    let doc = r.bench.td.key_document(&nonce);
    let (tcb_info, qe_identity, chain) = r.intel.collateral_under_extracted_pck(fresh(now()).0);
    let early = Collateral {
        tcb_info: tcb_info.clone(),
        tcb_info_chain_pem: chain.clone(),
        qe_identity: qe_identity.clone(),
        qe_identity_chain_pem: chain.clone(),
        ..r.bench.collateral.clone()
    };
    let p = policy(&r.intel);
    attest::verify_key(&doc, &r.bench.collateral, &p, &nonce, now()).expect("control");
    let e = attest::verify_key(&doc, &early, &p, &nonce, now()).unwrap_err();
    assert_eq!(e.step, "revocation", "{e}");
    assert!(e.reason.contains("collateral signer"), "{e}");
    let strict = policy(&r.intel).with_strict_tcb();
    let e = attest::verify_key(&doc, &early, &strict, &nonce, now()).unwrap_err();
    assert_eq!(e.step, "tcb", "{e}");
    assert!(
        e.reason.contains("issuer chain has 4 certificate(s)"),
        "{e}"
    );
    let standard = Collateral {
        tcb_info_standard: tcb_info,
        tcb_info_standard_chain_pem: chain.clone(),
        qe_identity_standard: qe_identity,
        qe_identity_standard_chain_pem: chain,
        ..r.bench.collateral.clone()
    };
    let e = attest::verify_key(&doc, &standard, &p, &nonce, now()).unwrap_err();
    assert_eq!(e.step, "tcb", "{e}");
    assert!(
        e.reason.contains("issuer chain has 4 certificate(s)"),
        "{e}"
    );
}

/// SECURITY: a PCK under a CA of the root other than Intel's PCK CAs, in a
/// chain of Intel's shape, with that CA's empty CRL. Without
/// `tcb::pck_crl_ca`, as the relay calls it, this verified (2026-10-27).
#[tokio::test]
async fn a_pck_under_another_ca_of_the_root_is_refused_at_revocation() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x60; 32];
    let honest = r.bench.td.key_document(&nonce);
    let report_data = r.bench.td.e2e.report_data(&nonce);
    let (quote, crl) = r
        .intel
        .quote_under_another_ca(&report_data, &mrtd(), &rtmr(), fresh(now()));
    let doc = KeyDocument {
        quote_b64: b64(&quote),
        ..honest
    };
    let col = Collateral {
        pck_crl_der_b64: b64(&crl),
        ..r.bench.collateral.clone()
    };
    let e = attest::verify_key(&doc, &col, &policy(&r.intel), &nonce, now()).unwrap_err();
    assert_eq!(e.step, "revocation", "{e}");
    assert!(
        e.reason
            .contains("neither Intel SGX PCK Platform CA nor Processor CA"),
        "{e}"
    );
}

/// SECURITY: a key configuration that offers another suite is refused before
/// anything is sealed: no carrier-chosen downgrade.
#[tokio::test]
async fn a_key_configuration_offering_another_suite_is_refused() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x58; 32];
    let mut doc = r.bench.td.key_document(&nonce);
    let mut config = base64::engine::general_purpose::STANDARD
        .decode(&doc.key_config_b64)
        .unwrap();
    let last = config.len() - 1;
    config[last] = 3; // ChaCha20-Poly1305
    doc.key_config_b64 = b64(&config);
    let e = attest::verify_key(&doc, &r.bench.collateral, &policy(&r.intel), &nonce, now())
        .unwrap_err();
    assert_eq!(e.step, "key");
}

/// SECURITY: the SDK compiles x0ne-infer's `tcb.rs`, which decodes Intel's
/// hex through `cloud_config::hex_bytes`. A multibyte character in a relayed
/// signature is no panic (a trap in wasm): a refusal at `tcb` in the
/// collateral accepted on (early under `strict`, standard under
/// `transparent`); in `early` under `transparent`, the key is accepted and
/// the disclosure names why Intel's latest view was not read, as the relay's.
#[tokio::test]
async fn a_multibyte_character_in_the_collaterals_hex_is_refused_not_a_panic() {
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x5a; 32];
    let doc = r.bench.td.key_document(&nonce);
    let transparent = policy(&r.intel);
    let strict = policy(&r.intel).with_strict_tcb();
    // 128 bytes, a P-256 signature's hex length; "€" first: &s[0..2] panics.
    let euro = format!("€{}", "0".repeat(125));
    assert_eq!(euro.len(), 128);
    for field in [
        "tcbInfo",
        "enclaveIdentity",
        "tcbInfo standard",
        "enclaveIdentity standard",
    ] {
        let mut col = r.bench.collateral.clone();
        let body = match field {
            "tcbInfo" => &mut col.tcb_info,
            "enclaveIdentity" => &mut col.qe_identity,
            "tcbInfo standard" => &mut col.tcb_info_standard,
            _ => &mut col.qe_identity_standard,
        };
        let mut v: Value = serde_json::from_str(body).unwrap();
        v["signature"] = Value::String(euro.clone());
        *body = v.to_string();
        let run = |p: &attest::Policy| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                attest::verify_key(&doc, &col, p, &nonce, now())
            }))
            .expect("a refusal, not a panic")
        };
        let (judged, other) = if field.ends_with("standard") {
            (&transparent, &strict)
        } else {
            (&strict, &transparent)
        };
        let e = run(judged).unwrap_err();
        assert_eq!(e.step, "tcb", "{field}: {e}");
        assert!(e.reason.contains("signature is not hex"), "{field}: {e}");
        let key = run(other).expect("the other update's collateral, untouched");
        if !field.ends_with("standard") {
            let why = key.tcb.intel.unwrap().latest_unavailable.unwrap();
            assert!(why.contains("signature is not hex"), "{field}: {why}");
        }
    }
}

// ── What the TD and the carrier cannot do ────────────────────────────────────

/// SECURITY: a twin TD (same image, same Intel, its own boot key) does not
/// open what was sealed to the TD the caller verified: a carrier that routes
/// the question elsewhere gets nothing it can read or answer.
#[tokio::test]
async fn a_twin_td_cannot_open_what_was_sealed_to_the_verified_one() {
    let r = bench(Lie::None, &[]).await;
    let client = verified(&r, [0x59; 32]).await.unwrap();
    let (request, _) = client.seal_chat(&question(), &[0x63; 32], now()).unwrap();
    let twin = Gateway::from_seed(&[6; 32]);
    assert!(matches!(
        twin.open(&request),
        Err(E2eError::Ohttp(ohttp::OhttpError::Hpke(_)))
    ));
    assert_ne!(
        KeyConfig::parse(&twin.key_config()).unwrap(),
        *client.key().key_config()
    );
    let twin = gateway(
        [6; 32],
        &SessionIdentity::generate([7; 32], [9; 32]),
        &session_config("cpu"),
    );
    assert_eq!(
        twin.admit(&request, now()).err(),
        Some(crate::e2e_gateway::Refusal::Open)
    );
}

/// SECURITY: an answer signed by a key the quote did not bind (another
/// process, another TD) is refused, even sealed by the right gateway.
#[tokio::test]
async fn an_answer_signed_by_an_unbound_key_is_refused() {
    let r = bench(Lie::UnboundSigner, &[]).await;
    let client = verified(&r, [0x5a; 32]).await.unwrap();
    let (request, pending) = client.seal_chat(&question(), &[0x64; 32], now()).unwrap();
    let (status, response) = send_sealed(&r, request).await;
    assert_eq!(status, 200);
    let e = pending
        .open_verified(&response, &r.digest, now())
        .unwrap_err();
    assert!(e.to_string().contains("signature"), "{e}");
}

/// SECURITY: a carrier that keeps a sealed question and sends it again gets
/// no second answer, and a question sealed out of the window gets none. The
/// refusal says neither: one body for both, the class only counted.
#[tokio::test]
async fn a_replayed_or_stale_request_gets_no_answer() {
    let r = bench(Lie::None, &[]).await;
    let client = verified(&r, [0x5b; 32]).await.unwrap();
    let (request, _) = client.seal_chat(&question(), &[0x65; 32], now()).unwrap();
    assert_eq!(send_sealed(&r, request.clone()).await.0, 200);
    let refused = serde_json::to_vec(&json!({ "error": REFUSED })).unwrap();
    let (status, body) = send_sealed(&r, request).await;
    assert_eq!((status, &body), (400, &refused));

    let (stale, _) = client
        .seal_chat(&question(), &[0x66; 32], now() - 3600)
        .unwrap();
    let (status, body) = send_sealed(&r, stale).await;
    assert_eq!((status, &body), (400, &refused));
    let counts = &r.bench.td.e2e.counts;
    assert_eq!(counts.refused_replay.load(Ordering::SeqCst), 2);
    assert_eq!(counts.opened.load(Ordering::SeqCst), 1);
}

// ── The SDK and the TD speak the same receipts and bindings ──────────────────

#[test]
fn the_sdk_spells_the_tds_receipt_domain_and_commitment_labels() {
    assert_eq!(
        x0ne_sdk::receipt::RECEIPT_DOMAIN,
        crate::binding::RECEIPT_DOMAIN
    );
    assert_eq!(
        x0ne_sdk::receipt::OUTPUT_LABEL,
        crate::receipt::CommitmentKind::Output.label()
    );
    assert_eq!(
        x0ne_sdk::receipt::PROMPT_LABEL,
        crate::receipt::CommitmentKind::Prompt.label()
    );
}

/// The key binding's label is its own: no `report_data` context of this
/// crate is a prefix of it, nor it of one.
#[test]
fn the_key_bindings_label_is_no_other_report_data_context() {
    let ours = e2e::BINDING_LABEL;
    for theirs in [
        &b"x0ne-infer/v1"[..],
        crate::binding::CONTEXT_V2,
        crate::binding::GPU_CONTEXT_V2,
        crate::binding::CONTEXT_V3,
    ] {
        assert!(!ours.starts_with(theirs) && !theirs.starts_with(ours));
    }
}

// ── A TD that attests its GPU ────────────────────────────────────────────────

/// What a GPU-attested TD, or its carrier, does to the GPU evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuLie {
    None,
    /// The document drops the evidence the quote bound.
    Dropped,
    /// A CPU-attested TD's quote, GPU evidence added next to it.
    Added,
    /// The report replaced, after the quote, by another one for the same nonce.
    OtherReport,
    /// The chain replaced, after the quote, by another GPU's: the carrier's
    /// own H100, which NRAS then judges.
    OtherChain,
}

/// One key document of a GPU-attested TD (the binary's `E2e`), the
/// collateral with NRAS's answer to it as the carrier fetches it
/// (`attest::nras_request`, `FakeNvidia::answer`), the caller's nonce and
/// its `attest=cpu+gpu` policy, the bench's NVIDIA intermediate pinned.
struct GpuCase {
    doc: KeyDocument,
    collateral: Collateral,
    nonce: [u8; 32],
    policy: Policy,
}

fn gpu_case(gpu_lie: GpuLie, nras_lie: NrasLie) -> GpuCase {
    gpu_case_for([0x47; 32], gpu_lie, nras_lie)
}

/// One NVIDIA for every case: its RSA intermediate (as 004) is slow to draw,
/// and an answer replayed from one case to another must be NVIDIA's own.
static NVIDIA: std::sync::LazyLock<FakeNvidia> = std::sync::LazyLock::new(FakeNvidia::new);

/// The gateway of the GPU cases' TD, booted under `attest=cpu+gpu`.
fn gpu_gateway() -> E2e {
    gateway(
        [0x45; 32],
        &SessionIdentity::generate([7; 32], [9; 32]),
        &session_config("cpu+gpu"),
    )
}

/// [`gpu_case`] for the caller's `nonce`.
fn gpu_case_for(nonce: [u8; 32], gpu_lie: GpuLie, nras_lie: NrasLie) -> GpuCase {
    let intel = FakeIntel::new();
    let nvidia = &*NVIDIA;
    let e2e = gpu_gateway();
    let gpu_nonce = e2e.gpu_nonce(&nonce);
    let evidence = vec![crate::nras_client::GpuEvidence {
        evidence: b64(&synthetic_report(&hex(&gpu_nonce))),
        certificate: gpu_chain_b64(&GPU_SERIAL_BYTES),
    }];
    let report_data = match gpu_lie {
        GpuLie::Added => e2e.report_data(&nonce),
        _ => e2e.report_data_gpu(&nonce, &evidence).unwrap(),
    };
    let quote = intel.quote(&report_data, &mrtd(), &rtmr(), [0; 8]);
    let ccel = crate::ccel::encode(&log());
    let mut sent = evidence.clone();
    match gpu_lie {
        GpuLie::OtherReport => {
            sent[0].evidence = b64(&[synthetic_report(&hex(&gpu_nonce)), b"#2".to_vec()].concat());
        }
        GpuLie::OtherChain => sent[0].certificate = gpu_chain_b64(&OTHER_GPU_SERIAL_BYTES),
        _ => {}
    }
    let doc = if gpu_lie == GpuLie::Dropped {
        e2e.document(&quote, &ccel)
    } else {
        e2e.document_gpu(&quote, &ccel, sent)
    };
    let mut collateral = collateral(&intel, &[]);
    // What the carrier would ask NRAS: for a document without GPU evidence,
    // the honest one's request.
    let asked = attest::nras_request(&doc, &nonce).unwrap_or_else(|_| {
        json!({ "nonce": hex(&gpu_nonce), "evidence_list": evidence,
                "arch": "HOPPER", "claims_version": "3.0" })
    });
    collateral.nras_answer = nvidia.answer(&asked, now(), nras_lie).to_string();
    collateral.nras_jwks = nvidia.jwks().to_string();
    let mut policy = policy(&intel);
    policy.attest = attest::Attest::CpuGpu;
    policy.nras_pins = vec![x0ne_sdk::nras_test_pin(&nvidia.intermediate_der()).unwrap()];
    GpuCase {
        doc,
        collateral,
        nonce,
        policy,
    }
}

impl GpuCase {
    fn verify(&self) -> Result<attest::VerifiedKey, attest::Refusal> {
        attest::verify_key(
            &self.doc,
            &self.collateral,
            &self.policy,
            &self.nonce,
            now(),
        )
    }
}

#[track_caller]
fn refused_at(r: Result<attest::VerifiedKey, attest::Refusal>, step: &str, says: &str) {
    let e = r.expect_err("accepted");
    assert_eq!(e.step, step, "{e}");
    assert!(e.reason.contains(says), "{e}");
}

/// The GPU path accepted: the quote binds the H100's report and chain, NRAS
/// judged that report for this document's nonce, and its token names the
/// GPU whose chain the quote bound. The verdict names it, with the GPU
/// limitations.
#[test]
fn a_gpu_tds_key_verifies_and_names_its_gpu() {
    let key = gpu_case(GpuLie::None, NrasLie::None)
        .verify()
        .expect("the honest GPU TD's key verifies");
    assert_eq!(key.gpu_ueids, vec![GPU_SERIAL_DECIMAL.to_string()]);
    for line in x0ne_sdk::GPU_LIMITATIONS {
        assert!(key.limitations.iter().any(|l| l == line), "{line}");
    }
    assert!(!key
        .limitations
        .iter()
        .any(|l| x0ne_sdk::CPU_ONLY_LIMITATIONS.contains(&l.as_str())));
}

/// SECURITY: the quote binds the GPU evidence: dropped, added to a CPU
/// TD's quote, or either half swapped after the quote (the carrier's own
/// H100 answering the same nonce), it is refused before NRAS is read.
#[test]
fn gpu_evidence_dropped_added_or_swapped_after_the_quote_is_refused_at_binding() {
    for lie in [
        GpuLie::Dropped,
        GpuLie::Added,
        GpuLie::OtherReport,
        GpuLie::OtherChain,
    ] {
        let e = gpu_case(lie, NrasLie::None).verify().unwrap_err();
        assert_eq!(e.step, "binding", "{lie:?}: {e}");
    }
}

/// SECURITY: the policy says which product it expects. A GPU TD's key
/// under `attest=cpu`, and a CPU TD's key under `attest=cpu+gpu`, are
/// refused at `gpu`, each with a quote that binds what its document holds.
#[test]
fn attest_cpu_refuses_a_gpu_td_and_attest_cpu_gpu_a_cpu_td() {
    let mut c = gpu_case(GpuLie::None, NrasLie::None);
    c.policy.attest = attest::Attest::Cpu;
    refused_at(c.verify(), "gpu", "attest=cpu;");
    let mut c = gpu_case(GpuLie::None, NrasLie::None);
    let intel = FakeIntel::new();
    let td = Td::new(
        Arc::new(intel),
        5,
        &SessionIdentity::generate([7; 32], [9; 32]),
        Lie::None,
    );
    c.doc = td.key_document(&c.nonce);
    c.policy.anchor = x0ne_sdk::TrustAnchor::RootDer(td.intel.root_der.clone());
    c.collateral = Collateral {
        nras_answer: c.collateral.nras_answer.clone(),
        nras_jwks: c.collateral.nras_jwks.clone(),
        ..collateral(&td.intel, &[])
    };
    refused_at(c.verify(), "gpu", "attest=cpu+gpu;");
    c.policy.attest = attest::Attest::Cpu;
    c.verify()
        .expect("the control: the CPU TD under attest=cpu");
}

/// SECURITY: NRAS's verdict carried in the key document, fetched by the TD
/// for this document's `gpu_nonce`: accepted with no NRAS answer in the
/// collateral. The document's verdict is the one judged, never replaced by
/// the collateral's: one for another nonce, or naming another GPU, is
/// refused at `gpu` (75) even next to a good one; half of it, refused too.
#[test]
fn the_nras_verdict_the_key_document_carries_is_the_one_judged() {
    let carried = |lie: NrasLie| {
        let c = gpu_case(GpuLie::None, NrasLie::None);
        let asked = attest::nras_request(&c.doc, &c.nonce).unwrap();
        let verdict = crate::attest_v2::NrasVerdict {
            answer: NVIDIA.answer(&asked, now(), lie),
            jwks: NVIDIA.jwks(),
        };
        (E2e::with_nras(c.doc.clone(), verdict), c)
    };
    let (doc, mut c) = carried(NrasLie::None);
    c.doc = doc;
    c.collateral.nras_answer = String::new();
    c.collateral.nras_jwks = String::new();
    let key = c.verify().expect("the document's own verdict");
    assert_eq!(key.gpu_ueids, vec![GPU_SERIAL_DECIMAL.to_string()]);
    for (lie, says) in [
        (NrasLie::OtherNonce, "eat_nonce"),
        (NrasLie::OtherUeid, "ueid"),
        (NrasLie::OverallFalse, ""),
    ] {
        let (doc, mut c) = carried(lie);
        c.doc = doc;
        let e = c.verify().unwrap_err();
        assert_eq!((e.step, e.code), ("gpu", 75), "{lie:?}: {e}");
        assert!(e.reason.contains(says), "{lie:?}: {e}");
    }
    let (doc, mut c) = carried(NrasLie::None);
    c.doc = KeyDocument {
        nras_jwks: None,
        ..doc
    };
    refused_at(c.verify(), "gpu", "without its JWKS");
    // A CPU document that names a verdict: nothing for it to judge.
    let (doc, mut c) = carried(NrasLie::None);
    c.doc = KeyDocument {
        gpu_evidence_list: None,
        ..doc
    };
    refused_at(c.verify(), "key", "without GPU evidence");
}

/// SECURITY: NRAS's answer for another nonce (an earlier document's, or
/// any nonce but this document's `gpu_nonce`) is refused.
#[test]
fn an_nras_answer_for_another_nonce_is_refused() {
    refused_at(
        gpu_case(GpuLie::None, NrasLie::OtherNonce).verify(),
        "gpu",
        "eat_nonce",
    );
    // Another document's honest answer, replayed: another caller's nonce.
    let earlier = gpu_case_for([0x48; 32], GpuLie::None, NrasLie::None);
    let mut c = gpu_case(GpuLie::None, NrasLie::None);
    c.collateral.nras_answer = earlier.collateral.nras_answer;
    refused_at(c.verify(), "gpu", "eat_nonce");
}

/// SECURITY: a token is NVIDIA's only under a pinned intermediate: one
/// signed by another key under the genuine kid, one under a leaf another CA
/// issued next to the pinned intermediate, and the bench's NVIDIA under
/// NVIDIA's production pins (004, 005), are refused.
#[test]
fn an_nras_token_under_an_unpinned_key_or_intermediate_is_refused() {
    refused_at(
        gpu_case(GpuLie::None, NrasLie::Unpinned).verify(),
        "gpu",
        "signature",
    );
    // Another NRAS's answer for this very request, its leaf in x5c[0] and
    // the pinned intermediate in x5c[1]: only the leaf's signature betrays it.
    let mut c = gpu_case(GpuLie::None, NrasLie::None);
    let other = FakeNvidia::p384();
    let asked = attest::nras_request(&c.doc, &c.nonce).unwrap();
    c.collateral.nras_answer = other.answer(&asked, now(), NrasLie::None).to_string();
    let mut jwks = other.jwks();
    jwks["keys"][0]["x5c"][1] = NVIDIA.jwks()["keys"][0]["x5c"][1].clone();
    c.collateral.nras_jwks = jwks.to_string();
    refused_at(
        c.verify(),
        "gpu",
        "not signed by the pinned NVIDIA intermediate",
    );
    let mut c = gpu_case(GpuLie::None, NrasLie::None);
    c.policy.nras_pins = x0ne_sdk::NVIDIA_NRAS_GPU_INTERMEDIATE_PINS.to_vec();
    refused_at(c.verify(), "gpu", "matches no pin");
}

/// SECURITY: NVIDIA's verdict is read, the summary and the detail: overall
/// false, `measres` not success, `dbgstat` enabled (a GPU in debug mode)
/// are refused, each with every other claim genuine. A report that does
/// not echo the nonce (another GPU's, or a CC-off GPU's) gets NRAS's
/// nonce-match false.
#[test]
fn a_gpu_nras_judged_failing_or_in_debug_is_refused() {
    for (lie, says) in [
        (NrasLie::OverallFalse, "x-nvidia-overall-att-result"),
        (NrasLie::MeasresFail, "measres"),
        (NrasLie::DebugEnabled, "dbgstat"),
    ] {
        refused_at(gpu_case(GpuLie::None, lie).verify(), "gpu", says);
    }
}

/// SECURITY: a token for another GPU, the carrier's own H100 asked for the
/// same nonce, names another ueid than the chain the quote bound: refused.
#[test]
fn a_token_for_another_gpu_is_refused() {
    refused_at(
        gpu_case(GpuLie::None, NrasLie::OtherUeid).verify(),
        "gpu",
        "not the local one",
    );
}

/// SECURITY: per-GPU tokens and an overall verdict from two NRAS answers.
#[test]
fn per_gpu_tokens_from_another_answer_are_refused() {
    refused_at(
        gpu_case(GpuLie::None, NrasLie::ForeignSubmods).verify(),
        "gpu",
        "submods",
    );
}

/// SECURITY: an expired answer, and an answer the carrier withheld,
/// emptied or reshaped, are refused.
#[test]
fn an_nras_answer_expired_withheld_or_malformed_is_refused() {
    refused_at(
        gpu_case(GpuLie::None, NrasLie::Expired).verify(),
        "gpu",
        "expired",
    );
    // Not valid before an hour on (review of 2026-10-27, finding 5).
    refused_at(
        gpu_case(GpuLie::None, NrasLie::NotYetValid).verify(),
        "gpu",
        "nbf",
    );
    for (answer, jwks, says) in [
        ("", None, "no NRAS answer"),
        ("not json", None, "nras_answer"),
        ("[[\"JWT\", \"a.b.c\"]]", None, "nras_answer"),
        ("[[\"JWT\", \"a.b.c\"], {}]", None, "zero GPUs"),
        ("", Some(""), "no NRAS answer"),
        ("keep", Some("{\"keys\": []}"), "kid"),
    ] {
        let mut c = gpu_case(GpuLie::None, NrasLie::None);
        if answer != "keep" {
            c.collateral.nras_answer = answer.into();
        }
        if let Some(j) = jwks {
            c.collateral.nras_jwks = j.into();
        }
        refused_at(c.verify(), "gpu", says);
    }
}

/// The TD binds no GPU evidence the SDK would refuse at `key`: none, more
/// than `MAX_GPUS`, a report or a chain over its cap or not base64. It
/// serves no document then (a 503, `cloud_serve::e2e_key`).
#[test]
fn the_td_binds_no_gpu_evidence_the_sdk_would_refuse() {
    let e2e = gpu_gateway();
    let c = gpu_case(GpuLie::None, NrasLie::None);
    let entry = crate::nras_client::GpuEvidence {
        evidence: c.doc.gpu_evidence_list.as_ref().unwrap()[0]
            .evidence
            .clone(),
        certificate: c.doc.gpu_evidence_list.as_ref().unwrap()[0]
            .certificate
            .clone(),
    };
    let bind = |gpus: &[crate::nras_client::GpuEvidence]| e2e.report_data_gpu(&[0x47; 32], gpus);
    bind(std::slice::from_ref(&entry)).expect("the control: one GPU binds");
    bind(&vec![entry.clone(); x0ne_sdk::e2e::MAX_GPUS]).expect("the control: MAX_GPUS bind");
    let over = "A".repeat(attest::MAX_GPU_EVIDENCE_BYTES / 3 * 4 + 4);
    for (gpus, why) in [
        (vec![], "none"),
        (
            vec![entry.clone(); x0ne_sdk::e2e::MAX_GPUS + 1],
            "MAX_GPUS + 1",
        ),
        (
            vec![crate::nras_client::GpuEvidence {
                evidence: over.clone(),
                ..entry.clone()
            }],
            "a report over its cap",
        ),
        (
            vec![crate::nras_client::GpuEvidence {
                certificate: over.clone(),
                ..entry.clone()
            }],
            "a chain over its cap",
        ),
        (
            vec![crate::nras_client::GpuEvidence {
                certificate: format!("{}!", entry.certificate),
                ..entry.clone()
            }],
            "a chain not base64",
        ),
    ] {
        assert!(bind(&gpus).is_err(), "{why}");
    }
}

/// Caps before decoding: the GPU evidence at `key` (count and size), the
/// NRAS answer and JWKS at `gpu`.
#[test]
fn gpu_fields_over_their_caps_are_refused_at_their_step() {
    let c = gpu_case(GpuLie::None, NrasLie::None);
    let entry = c.doc.gpu_evidence_list.as_ref().unwrap()[0].clone();
    let over = |cap: usize| "A".repeat(cap / 3 * 4 + 8);
    let mut docs = Vec::new();
    for list in [
        vec![],
        vec![entry.clone(); x0ne_sdk::e2e::MAX_GPUS + 1],
        vec![x0ne_sdk::attest::GpuEvidence {
            evidence: over(attest::MAX_GPU_EVIDENCE_BYTES),
            ..entry.clone()
        }],
        vec![x0ne_sdk::attest::GpuEvidence {
            certificate: over(attest::MAX_GPU_EVIDENCE_BYTES),
            ..entry.clone()
        }],
        vec![x0ne_sdk::attest::GpuEvidence {
            evidence: format!("{} ", entry.evidence),
            ..entry.clone()
        }],
    ] {
        docs.push(KeyDocument {
            gpu_evidence_list: Some(list),
            ..c.doc.clone()
        });
    }
    for d in docs {
        let e = attest::verify_key(&d, &c.collateral, &c.policy, &c.nonce, now()).unwrap_err();
        assert_eq!(e.step, "key", "{e}");
        assert!(e.reason.contains("gpu_evidence_list"), "{e}");
    }
    let big = over(attest::MAX_COLLATERAL_BYTES);
    for col in [
        Collateral {
            nras_answer: big.clone(),
            ..c.collateral.clone()
        },
        Collateral {
            nras_jwks: big.clone(),
            ..c.collateral.clone()
        },
    ] {
        let e = attest::verify_key(&c.doc, &col, &c.policy, &c.nonce, now()).unwrap_err();
        assert_eq!(e.step, "gpu", "{e}");
        assert!(e.reason.contains("cap"), "{e}");
    }
    c.verify().expect("control");
}

/// The body the carrier POSTs is the relay's own (`nras_client`), over the
/// document's evidence and its `gpu_nonce`; and the SDK splits NRAS's
/// answer as the relay does.
#[test]
fn the_sdk_asks_nras_as_the_relay_does() {
    let c = gpu_case(GpuLie::None, NrasLie::None);
    let e2e = gpu_gateway();
    let evidence: Vec<crate::nras_client::GpuEvidence> = c
        .doc
        .gpu_evidence_list
        .as_ref()
        .unwrap()
        .iter()
        .map(|g| crate::nras_client::GpuEvidence {
            evidence: g.evidence.clone(),
            certificate: g.certificate.clone(),
        })
        .collect();
    let relays = crate::nras_client::build_attest_request(
        &hex(&e2e.gpu_nonce(&c.nonce)),
        evidence,
        "HOPPER",
    )
    .unwrap();
    assert_eq!(
        attest::nras_request(&c.doc, &c.nonce).unwrap(),
        serde_json::to_value(relays).unwrap()
    );
    let answer: Value = serde_json::from_str(&c.collateral.nras_answer).unwrap();
    let (overall, per_gpu) = crate::nras_client::split_attest_response(&answer).unwrap();
    assert_eq!(answer[0][1], overall);
    assert_eq!(per_gpu.len(), 1);
    assert!(attest::nras_request(&gpu_case(GpuLie::Dropped, NrasLie::None).doc, &c.nonce).is_err());
}

/// The GPU labels are their own: neither is a prefix of the other, of the
/// CPU key binding's, or of any `report_data` or GPU nonce context of the
/// relay's.
#[test]
fn the_gpu_labels_are_no_other_context() {
    let ours = [
        e2e::BINDING_LABEL,
        e2e::GPU_BINDING_LABEL,
        e2e::GPU_NONCE_LABEL,
    ];
    let theirs = [
        &b"x0ne-infer/v1"[..],
        crate::binding::CONTEXT_V2,
        crate::binding::GPU_CONTEXT_V2,
        crate::binding::CONTEXT_V3,
        crate::binding::GPU_CONTEXT_V4,
    ];
    for (i, a) in ours.iter().enumerate() {
        for b in ours.iter().skip(i + 1).chain(theirs.iter()) {
            assert!(!a.starts_with(b) && !b.starts_with(a), "{a:?} {b:?}");
        }
    }
}

// ── The wasm32 build, in Node ────────────────────────────────────────────────

/// The C ABI keeps no memory of the log of its own: the caller hands back
/// the `tlog_checkpoint` a verified key returned. The second verification
/// names the size remembered; a log grown since is refused without a
/// consistency proof (100, `log-consistency`) and admitted with one from
/// the remembered size, moving the memory; a history that does not extend
/// the remembered one is refused with the same code.
#[tokio::test]
async fn the_c_abi_takes_the_callers_log_checkpoint_and_returns_the_one_to_keep() {
    use x0ne_sdk::abi::{call, OP_VERIFY_KEY};
    use x0ne_sdk::attest::TestLog;
    let r = bench(Lie::None, &[]).await;
    let nonce = [0x53; 32];
    let http = reqwest::Client::new();
    let doc: Value = http
        .get(format!("{}/v2/e2e/key?nonce={}", r.url, hex(&nonce)))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let col: Value = http
        .get(format!("{}/v2/e2e/collateral", r.url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let json = reference_json();
    let (signature, _) = signed(json.as_bytes(), now());
    let policy_with = |proof: String| {
        let mut p = abi_policy(json!({
            "machine_type": MACHINE,
            "model_digest_hex": r.digest,
            "root_der_b64": b64(&r.intel.root_der),
            "gce_root_der_b64": b64(&google_root()),
            "attest": "cpu",
        }));
        p["reference_tlog_proof"] = json!(proof);
        p["reference_sig_b64"] = json!(b64(&signature));
        p
    };
    let verify = |pol: Value, checkpoint: Option<&str>, consistency: Option<&str>| {
        let mut input = json!({ "document": doc, "collateral": col, "policy": pol,
                "client_nonce_hex": hex(&nonce), "now": now() });
        if let Some(c) = checkpoint {
            input["tlog_checkpoint"] = json!(c);
        }
        if let Some(c) = consistency {
            input["tlog_consistency"] = json!(c);
        }
        call(OP_VERIFY_KEY, &serde_json::to_vec(&input).unwrap())
    };
    // One entry: the reference. The first sight is kept.
    let mut log = TestLog::new();
    log.append(json.as_bytes());
    // One instant for the two proofs of this size: the witness dates its cosignature, and the
    // note kept is compared whole below.
    let first = now();
    let v = verify(policy_with(log.proof(0, first)), None, None);
    assert_eq!(v["ok"], true, "{v}");
    let memory = &v["verified"]["reference"]["log"]["memory"];
    assert_eq!(
        (memory["before"].as_u64(), memory["kept"].as_u64()),
        (None, Some(1)),
        "{v}"
    );
    let kept = v["verified"]["tlog_checkpoint"]
        .as_str()
        .expect("a note to keep")
        .to_string();
    assert!(
        kept.starts_with("test.invalid/x0ne-engine/tlog\n1\n"),
        "{kept}"
    );
    // The same checkpoint, remembered: nothing to prove.
    let v = verify(policy_with(log.proof(0, first)), Some(&kept), None);
    assert_eq!(v["ok"], true, "{v}");
    let memory = &v["verified"]["reference"]["log"]["memory"];
    assert_eq!(
        (memory["before"].as_u64(), memory["kept"].as_u64()),
        (Some(1), Some(1)),
        "{v}"
    );
    assert_eq!(v["verified"]["tlog_checkpoint"], kept, "{v}");
    // The log grew: refused without the proof, admitted with it.
    log.append(b"another reference");
    let v = verify(policy_with(log.proof(0, now())), Some(&kept), None);
    assert_eq!(v["ok"], false, "{v}");
    assert_eq!(
        (v["code"].as_i64(), v["step"].as_str()),
        (Some(100), Some("log-consistency")),
        "{v}"
    );
    assert!(
        v["error"]
            .as_str()
            .unwrap()
            .contains("the log grew from 1 to 2"),
        "{v}"
    );
    let v = verify(
        policy_with(log.proof(0, now())),
        Some(&kept),
        Some(&log.consistency(1, now())),
    );
    assert_eq!(v["ok"], true, "{v}");
    let memory = &v["verified"]["reference"]["log"]["memory"];
    assert_eq!(
        (memory["before"].as_u64(), memory["kept"].as_u64()),
        (Some(1), Some(2)),
        "{v}"
    );
    let kept2 = v["verified"]["tlog_checkpoint"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        kept2.starts_with("test.invalid/x0ne-engine/tlog\n2\n"),
        "{kept2}"
    );
    // A history of size 2 that does not extend the one of size 1: its own
    // consistency body verifies against nothing this caller saw.
    let mut other = TestLog::new();
    other.append(b"another reference");
    other.append(json.as_bytes());
    let v = verify(
        policy_with(other.proof(1, now())),
        Some(&kept),
        Some(&other.consistency(1, now())),
    );
    assert_eq!(v["ok"], false, "{v}");
    assert_eq!(
        (v["code"].as_i64(), v["step"].as_str()),
        (Some(100), Some("log-consistency")),
        "{v}"
    );
    assert!(
        v["error"].as_str().unwrap().contains("is not a prefix of"),
        "{v}"
    );
    // And the memory of size 2 refuses the size-1 proof: the log went back.
    let mut one = TestLog::new();
    one.append(json.as_bytes());
    let v = verify(policy_with(one.proof(0, now())), Some(&kept2), None);
    assert_eq!(
        (v["code"].as_i64(), v["step"].as_str()),
        (Some(100), Some("log-rollback")),
        "{v}"
    );
}

/// The wasm32 build with `test-anchor` (the bench's root accepted).
fn wasm_bench() -> String {
    std::env::var("X0NE_SDK_WASM").expect("X0NE_SDK_WASM: the wasm32 build with test-anchor")
}

/// The wasm32 build a page would ship: no feature, Intel's root alone.
fn wasm_pinned() -> String {
    std::env::var("X0NE_SDK_WASM_PINNED")
        .expect("X0NE_SDK_WASM_PINNED: the wasm32 build without test-anchor")
}

/// Runs `x0ne-sdk/js/smoke.mjs` in Node on the wasm32 build at `wasm`,
/// against `r`. The JSON it prints, or its error.
fn node(wasm: &str, r: &Running, policy: Value) -> Result<Value, String> {
    node_with(wasm, r, policy, &[])
}

/// [`node`] with `env` set for the script (`X0NE_SMOKE_TLOG_FILE`: the file
/// the loader keeps the log's checkpoint in, as a page keeps it in storage).
fn node_with(
    wasm: &str,
    r: &Running,
    policy: Value,
    env: &[(&str, &str)],
) -> Result<Value, String> {
    let node = std::env::var("X0NE_NODE").unwrap_or_else(|_| "node".into());
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/../x0ne-sdk/js/smoke.mjs");
    let out = std::process::Command::new(node)
        .arg(script)
        .arg(wasm)
        .arg(&r.url)
        .arg(policy.to_string())
        .arg(QUESTION)
        .envs(env.iter().copied())
        .output()
        .expect("node runs");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let v: Value = serde_json::from_str(stdout.trim())
        .map_err(|e| format!("{e}: {stdout} {}", String::from_utf8_lossy(&out.stderr)))?;
    if out.status.success() {
        Ok(v)
    } else {
        Err(v["error"].as_str().unwrap_or("").to_string())
    }
}

fn wasm_policy(r: &Running) -> Value {
    abi_policy(json!({
        "machine_type": MACHINE,
        "model_digest_hex": r.digest,
        "root_der_b64": b64(&r.intel.root_der),
        "gce_root_der_b64": b64(&google_root()),
        "attest": "cpu",
    }))
}

/// The browser's path: the wasm32 build verifies the quote and the TCB,
/// seals with Node's `crypto.getRandomValues`, opens, checks the receipt.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the wasm32 build and Node: scripts/sdk-wasm.sh"]
async fn wasm_in_node_verifies_seals_opens_and_checks_the_receipt() {
    let r = bench(Lie::None, &[]).await;
    let pol = wasm_policy(&r);
    let v = tokio::task::spawn_blocking({
        let r = Running {
            url: r.url.clone(),
            bench: r.bench.clone(),
            intel: r.intel.clone(),
            digest: r.digest.clone(),
        };
        move || node(&wasm_bench(), &r, pol)
    })
    .await
    .unwrap()
    .expect("node: the wasm path");
    assert_eq!(v["status"], 200, "{v}");
    assert_eq!(v["receipt"]["checked"], true, "{v}");
    assert!(v["text"].as_str().unwrap().contains("Lyon-Perrache"), "{v}");
    assert_eq!(v["verified"]["tcb_platform"], "UpToDate");
    // A closed session seals nothing more (op 4).
    assert_eq!(v["closed"], true, "{v}");
    // The module imports nothing: no clock, no randomness, no network of its
    // own. Each such input is the caller's, where a page can see it given.
    assert_eq!(v["wasm_imports"], 0, "{v}");
    for bytes in r.bench.carried.lock().unwrap().iter() {
        assert!(!contains(bytes, NEEDLE), "the carrier read the question");
    }
}

/// The browser's memory of the log: the loader keeps the checkpoint the
/// module returned (here in a file, as a page keeps it in storage) and
/// hands it back; a second run names the size remembered, and a log grown
/// since is refused in Node as natively, the file unchanged.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the wasm32 build and Node: scripts/sdk-wasm.sh"]
async fn wasm_in_node_keeps_the_logs_checkpoint_between_runs() {
    use x0ne_sdk::attest::TestLog;
    let r = bench(Lie::None, &[]).await;
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("tlog-checkpoint.txt");
    let file_s = file.to_str().unwrap().to_string();
    let json = reference_json();
    let (signature, _) = signed(json.as_bytes(), now());
    let mut log = TestLog::new();
    log.append(json.as_bytes());
    let policy_with = |proof: String| {
        let mut p = wasm_policy(&r);
        p["reference_tlog_proof"] = json!(proof);
        p["reference_sig_b64"] = json!(b64(&signature));
        p
    };
    let run = |pol: Value| {
        let run = Running {
            url: r.url.clone(),
            bench: r.bench.clone(),
            intel: r.intel.clone(),
            digest: r.digest.clone(),
        };
        let file_s = file_s.clone();
        tokio::task::spawn_blocking(move || {
            node_with(
                &wasm_bench(),
                &run,
                pol,
                &[("X0NE_SMOKE_TLOG_FILE", &file_s)],
            )
        })
    };
    let v = run(policy_with(log.proof(0, now())))
        .await
        .unwrap()
        .expect("first run");
    assert_eq!(
        v["verified"]["reference"]["log"]["memory"]["before"],
        Value::Null,
        "{v}"
    );
    let kept = std::fs::read_to_string(&file).expect("the loader kept the checkpoint");
    assert_eq!(v["verified"]["tlog_checkpoint"], kept, "{v}");
    assert!(
        kept.starts_with("test.invalid/x0ne-engine/tlog\n1\n"),
        "{kept}"
    );
    let v = run(policy_with(log.proof(0, now())))
        .await
        .unwrap()
        .expect("second run");
    assert_eq!(
        v["verified"]["reference"]["log"]["memory"]["before"], 1,
        "{v}"
    );
    // The same checkpoint, signed again at this run's clock: the file holds
    // the newer note of the same tree head (origin, size, root).
    let kept2 = std::fs::read_to_string(&file).unwrap();
    assert_eq!(v["verified"]["tlog_checkpoint"], kept2, "{v}");
    let head = |n: &str| n.lines().take(3).collect::<Vec<_>>().join("\n");
    assert_eq!(head(&kept2), head(&kept), "{kept2}");
    log.append(b"another reference");
    let e = run(policy_with(log.proof(0, now())))
        .await
        .unwrap()
        .expect_err("grown, no proof");
    assert!(e.contains("refused at log-consistency"), "{e}");
    assert_eq!(
        std::fs::read_to_string(&file).unwrap(),
        kept2,
        "the memory moved on a refusal"
    );
}

/// The wasm32 build refuses as the native one does.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the wasm32 build and Node: scripts/sdk-wasm.sh"]
async fn wasm_in_node_refuses_a_carrier_key_and_a_measurement_drift() {
    for (lie, edit, step) in [
        (Lie::CarrierKey, None, "binding"),
        (Lie::RtmrDrift, None, "measurement"),
        (Lie::None, Some("mrtd_hex"), "measurement"),
    ] {
        let r = bench(lie, &[]).await;
        let mut pol = wasm_policy(&r);
        if let Some(field) = edit {
            pol[field] = json!("ab".repeat(48));
        }
        let run = Running {
            url: r.url.clone(),
            bench: r.bench.clone(),
            intel: r.intel.clone(),
            digest: r.digest.clone(),
        };
        let e = tokio::task::spawn_blocking(move || node(&wasm_bench(), &run, pol))
            .await
            .unwrap()
            .expect_err("refused");
        assert!(e.contains(&format!("refused at {step}")), "{lie:?}: {e}");
        assert!(
            r.bench.carried.lock().unwrap().is_empty(),
            "nothing was sealed"
        );
    }
}

/// SECURITY: the build a page would ship, without `test-anchor`, takes no
/// root but Intel's and Google's. Handed the bench's roots, it refuses before
/// it reads a quote; handed none, it judges the bench's quote under Intel's
/// root and refuses that. Either way nothing is sealed.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the wasm32 builds and Node: scripts/sdk-wasm.sh"]
async fn wasm_built_without_test_anchor_takes_no_root_but_intels() {
    // A build made before the signing ceremony pins no archive key: past the
    // bench's pins, it refuses the reference itself, with the relay's 99.
    for (drop_root, nvidia, test_keys, expected) in [
        (
            false,
            false,
            true,
            "root_der_b64: this build pins Intel's root and takes no other",
        ),
        (
            true,
            true,
            true,
            "nras_intermediate_der_b64: this build pins NVIDIA's intermediates and takes no other",
        ),
        (
            true,
            false,
            true,
            "gce_root_der_b64: this build pins Google's root and takes no other",
        ),
        (
            true,
            false,
            true,
            "reference_keys: this build pins x0ne's archive key and log and takes no other",
        ),
        (
            true,
            false,
            false,
            "refused at reference: this build pins no x0ne archive key",
        ),
    ] {
        let r = bench(Lie::None, &[]).await;
        let mut pol = wasm_policy(&r);
        if drop_root {
            pol.as_object_mut().unwrap().remove("root_der_b64");
        }
        // Past Intel's root, every case but Google's own drops Google's root:
        // the policy checks it before NVIDIA's intermediates and the keys.
        if drop_root && !expected.starts_with("gce_root_der_b64") {
            pol.as_object_mut().unwrap().remove("gce_root_der_b64");
        }
        if nvidia {
            pol["attest"] = json!("cpu+gpu");
            pol["nras_intermediate_der_b64"] = json!(b64(&FakeNvidia::new().intermediate_der()));
        }
        if !test_keys {
            pol.as_object_mut().unwrap().remove("reference_keys");
        }
        let run = Running {
            url: r.url.clone(),
            bench: r.bench.clone(),
            intel: r.intel.clone(),
            digest: r.digest.clone(),
        };
        let e = tokio::task::spawn_blocking(move || node(&wasm_pinned(), &run, pol))
            .await
            .unwrap()
            .expect_err("refused");
        assert!(e.contains(expected), "drop_root={drop_root}: {e}");
        assert!(
            r.bench.carried.lock().unwrap().is_empty(),
            "nothing was sealed"
        );
    }
}

/// The OCSP responses the key document staples about NRAS's signing chain
/// are judged under the SDK's policy, as the relay judges a document's: a
/// good one admits under `required` and the verified key names it; a revoked
/// one refuses at `revocation` (75) under both policies; one that is not
/// base64 refuses at `gpu`; `required` without any refuses at `revocation`.
#[test]
fn the_key_documents_stapled_ocsp_is_judged_under_the_sdk_policy() {
    use crate::nras_ocsp::tests::Status;
    let good = NVIDIA.ocsp_b64(Status::Good, now());
    let revoked = NVIDIA.ocsp_b64(Status::Revoked(now() - 3600, Some(1)), now());
    let verify = |ocsp: Option<Vec<String>>, required: bool| {
        let GpuCase {
            mut doc,
            collateral,
            nonce,
            policy,
        } = gpu_case(GpuLie::None, NrasLie::None);
        doc.nras_ocsp = ocsp;
        let policy = if required {
            policy.with_required_nras_ocsp()
        } else {
            policy
        };
        attest::verify_key(&doc, &collateral, &policy, &nonce, now())
    };
    let key = verify(Some(vec![good.clone()]), true).expect("a good stapled response");
    assert!(
        key.nras_revocation
            .iter()
            .any(|l| l.contains("x5c leaf") && l.contains("good")),
        "{:?}",
        key.nras_revocation
    );
    assert!(
        key.limitations.iter().any(|l| l.contains("required")),
        "{:?}",
        key.limitations
    );
    for required in [false, true] {
        let e = verify(Some(vec![revoked.clone()]), required).expect_err("revoked");
        assert_eq!((e.step, e.code), ("revocation", 75), "{e}");
        assert!(e.reason.contains("revoked"), "{e}");
    }
    refused_at(
        verify(Some(vec!["not base64!".into()]), false),
        "gpu",
        "nras_ocsp[0]",
    );
    refused_at(verify(None, true), "revocation", "no response stapled");
}
