//! The C ABI a browser calls: JSON in, JSON out, no wasm-bindgen. Built for
//! `wasm32-unknown-unknown`; compiled and tested natively too.
//!
//! ```text
//! ptr = x0ne_alloc(len); write the request JSON at ptr  // null over MAX_INPUT_BYTES
//! n   = x0ne_call(op, ptr, len)        // frees nothing; the result is kept
//! read n bytes at x0ne_result_ptr()    // {"ok":true,...} or {"ok":false,"error":...,"code"?}
//! x0ne_free(ptr, len)
//! ```
//!
//! Threads (a native host; a wasm instance has one): the result is the
//! calling thread's own, read at `x0ne_result_ptr()` on that thread and
//! valid until that thread's next `x0ne_call` (or its exit); another
//! thread's calls neither replace nor free it. The handles are the
//! instance's, shared by every thread: one thread can use or close a
//! handle another was given. Calls are serialized on them.
//!
//! Panics: no export unwinds into the host. A call that panics returns
//! `{"ok":false,"error":...,"code":101,"step":"panic"}` ([`CODE_PANIC`], the
//! code a panicking Rust process exits with), and drops every handle of the
//! instance (keys, pending requests, streams, on every thread), since the
//! state it left may be half changed: the host verifies a key again. Handle
//! numbers are not reused after it. In wasm32 a panic is a trap of the
//! instance instead (no unwinding): `js/x0ne-sdk.mjs` refuses every call
//! after one, and the page loads the module again.
//!
//! Ops: [`OP_VERIFY_KEY`] returns a key handle and what was verified, the
//! TD's session config the quote bound among it (`session_config`: the
//! model, `expires_at_unix`, the relay's TLS key as `client_spki_sha256`);
//! [`OP_SEAL_CHAT`] takes one, refuses within a minute of the session's end,
//! and returns the Encapsulated Request and a pending handle; [`OP_OPEN`]
//! takes the pending handle (once) and the Encapsulated Response, and for a
//! 200 returns the text only once its receipt checked under the session key
//! the quote bound (`receipt.rs`), for the model the policy names (any
//! other answer is an error, below);
//! [`OP_CLOSE`] drops a key or a pending handle, of [`MAX_HANDLES`] an
//! instance holds at once; [`OP_NRAS_REQUEST`] returns the body a carrier
//! POSTs to NVIDIA's NRAS for a key document with GPU evidence (checked by
//! nothing: the verdict is [`OP_VERIFY_KEY`]'s). Nothing is sealed to a key that did not pass
//! `verify_key` in this instance.
//!
//! The sealed stream: [`OP_SEAL_CHAT_STREAM`] takes a key handle and a body
//! with `stream: true`, and returns the chunked Encapsulated Request
//! (`message/ohttp-chunked-req`) and a stream handle; [`OP_STREAM_PUSH`]
//! takes the handle and the bytes of one frame of the chunked response as
//! they came off the transport, and returns the deltas they completed, NOT
//! VERIFIED (`unverified_deltas`); [`OP_STREAM_FINISH`] takes the handle
//! (once) at the end of the body, checks the receipt of the final chunk over
//! the whole text, and returns the text as [`OP_OPEN`] does, with the
//! deltas the final chunk carried (`final_deltas`) and the count of chunks.
//! A frame altered, dropped, reordered or replayed is refused at `push`, and
//! the stream stays refused at `finish`; [`OP_CLOSE`] frees a stream whose
//! end never came.
//!
//! What a receipt signs and what it does not are apart. [`OP_OPEN`] and
//! [`OP_STREAM_FINISH`] return `status`, `text` and `receipt` (a stream:
//! `final_deltas`, the end of `text`; `chunks`, counted here; `conversation`,
//! the handle the signed commitment opens); what the TD said that no
//! receipt signs is under `unverified`: the whole `body` of an answer, a
//! stream's `finish_reason` and `usage`. An answer without a checked
//! receipt (the TD's refusal, a stream that ended in an error event) is an
//! error, never a result: `{"ok":false,"error":"no checked receipt: ...",
//! "unverified":{"status":...,"body"|"error":...}}`, and the deltas a
//! stream pushed are to be discarded.
//!
//! The policy names the image reference (`reference_json`, the file's text,
//! with `reference_sig_b64`, its detached signature, and
//! `reference_tlog_proof`, its log proof: admitted only signed by x0ne's
//! archive key and shown in x0ne's log, as the relay admits it, so a
//! carrier that swaps it is refused with 99 or 100), the machine type and
//! the served model's digest (required: the relay checks its catalogue's);
//! MRTD and RTMR1..3 pins are optional, checked on top;
//! `max_collateral_age_secs` may shorten the relay's day; `attest`
//! (`"cpu+gpu"`, required; `"cpu"` refused by name) asks for the TD's GPUs;
//! `tcb` (`"transparent"`, the default, or `"strict"`) is the relay's TCB
//! policy. The roots are Intel's and Google's, the NRAS intermediates
//! NVIDIA's, the archive key and the log x0ne's. A `root_der_b64`, an
//! `nras_intermediate_der_b64` or `"reference_keys": "test"` in the policy
//! is read only by a build with the `test-anchor` feature (the simulated
//! bench's Intel root, NVIDIA intermediate, TEST archive key and TEST log)
//! and is refused otherwise. A refusal of [`OP_VERIFY_KEY`] carries the
//! relay's exit code for it (`code`) and the step that refused (`step`).
//!
//! The log's memory is the caller's, not the instance's: [`OP_VERIFY_KEY`]
//! takes `tlog_checkpoint`, the signed checkpoint note a verified key
//! returned last time (`verified.tlog_checkpoint`), and `tlog_consistency`,
//! a tlog-witness `add-checkpoint` body from that checkpoint's size to the
//! reference's, needed once the log has grown; the reference's checkpoint
//! is held to it as `attest::admit_reference` says (a fork, a smaller
//! checkpoint, a missing or false proof: 100, the step named). The page
//! keeps the note where it keeps its state (`js/x0ne-sdk.mjs`:
//! `localStorage`); the module has no storage and keeps none between calls.

#![allow(unsafe_code)]

use crate::attest::{self, Collateral, KeyDocument, Policy};
use crate::e2e::{Client, Pending};
use crate::sealed_stream::PendingStream;
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Mutex, MutexGuard};

pub const OP_VERIFY_KEY: u32 = 1;
pub const OP_SEAL_CHAT: u32 = 2;
pub const OP_OPEN: u32 = 3;
pub const OP_CLOSE: u32 = 4;
pub const OP_NRAS_REQUEST: u32 = 5;
pub const OP_SEAL_CHAT_STREAM: u32 = 6;
pub const OP_STREAM_PUSH: u32 = 7;
pub const OP_STREAM_FINISH: u32 = 8;
/// Panics holding the instance's state: a build with `test-anchor` only,
/// for the tests of what a panic leaves (`tests/abi_ffi.rs`).
#[cfg(feature = "test-anchor")]
pub const OP_TEST_PANIC: u32 = 0xdead;
/// Keys and requests one instance holds at once; [`OP_CLOSE`] frees one.
pub const MAX_HANDLES: usize = 1024;
/// The largest input [`x0ne_alloc`] gives a buffer for: a key document and
/// its collateral at the relay's caps fit (`attest::MAX_COLLATERAL_BYTES`).
pub const MAX_INPUT_BYTES: usize = 64 * 1024 * 1024;

/// A verified key, and the model its answers must be signed for.
struct Key {
    client: Client,
    model_digest_hex: String,
}

#[derive(Default)]
struct State {
    next: u32,
    keys: HashMap<u32, Key>,
    pending: HashMap<u32, (Pending, String)>,
    /// A stream in flight, and the model its receipt must be signed for.
    streams: HashMap<u32, (PendingStream, String)>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

thread_local! {
    /// The calling thread's last result: what [`x0ne_result_ptr`] points at.
    static RESULT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// The `code` of a call that panicked, the step `panic`: the exit code of a
/// Rust process that panics.
pub const CODE_PANIC: i32 = 101;

/// What a call that panicked returns (no payload: it may quote input).
const PANICKED: &[u8] = br#"{"ok":false,"error":"the call panicked: every handle of this instance was dropped, verify a key again","code":101,"step":"panic"}"#;

/// The state, its handles dropped first if a panic poisoned its lock: the
/// call that panicked may have left a key, a request or a stream half
/// changed, so none is served again.
fn lock_state() -> MutexGuard<'static, Option<State>> {
    match STATE.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            let mut guard = poisoned.into_inner();
            drop_handles(&mut guard);
            STATE.clear_poison();
            guard
        }
    }
}

/// Every key, pending request and stream dropped; the handle counter kept,
/// so a number the host still holds names nothing issued later.
fn drop_handles(st: &mut Option<State>) {
    let next = st.as_ref().map_or(0, |s| s.next);
    *st = Some(State {
        next,
        ..State::default()
    });
}

fn with_state<T>(f: impl FnOnce(&mut State) -> T) -> T {
    let mut guard = lock_state();
    f(guard.get_or_insert_with(State::default))
}

fn b64d(s: &str) -> Result<Vec<u8>, String> {
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| e.to_string())
}

fn b64e(b: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(b)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyIn {
    /// The image reference file (`x0ne-engine/reference/v2`), verbatim.
    reference_json: String,
    /// Its detached OpenPGP signature (`<file>.sig`), base64: required.
    reference_sig_b64: String,
    /// Its log proof (`<file>.tlog-proof`), the text: required.
    reference_tlog_proof: String,
    /// `"test"`: the TEST archive key and TEST log (`test-anchor` builds only).
    #[serde(default)]
    reference_keys: Option<String>,
    /// `"transparent"` (default) or `"strict"`.
    #[serde(default)]
    tcb: Option<String>,
    /// The GCE machine type the TD must run on, as the reference names it.
    machine_type: String,
    #[serde(default)]
    mrtd_hex: Option<String>,
    #[serde(default)]
    rtmr1_hex: Option<String>,
    #[serde(default)]
    rtmr2_hex: Option<String>,
    #[serde(default)]
    rtmr3_hex: Option<String>,
    /// Optional, on top of the reference's pin to the endorsed firmware.
    #[serde(default)]
    rtmr0_hex: Option<String>,
    /// The served model's digest the answers' receipts must sign, hex:
    /// required, as the relay requires its catalogue's.
    model_digest_hex: String,
    /// Shorter than the relay's day (`tcb::TcbPolicy::tightened`), never longer.
    #[serde(default)]
    max_collateral_age_secs: Option<u64>,
    #[serde(default)]
    root_der_b64: Option<String>,
    /// A Google root in place of the pinned one (`test-anchor` builds only).
    #[serde(default)]
    gce_root_der_b64: Option<String>,
    /// `"cpu+gpu"` (`attest::Attest`): required, never guessed; `"cpu"` is refused.
    attest: String,
    #[serde(default)]
    nras_intermediate_der_b64: Option<String>,
    /// `"transparent"` (default, the relay's) or `"required"`: how the OCSP
    /// responses the key document staples about the NRAS signing chain are
    /// judged (`Policy::nras_ocsp`).
    #[serde(default)]
    nras_ocsp: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VerifyIn {
    document: KeyDocument,
    collateral: Collateral,
    policy: PolicyIn,
    client_nonce_hex: String,
    now: u64,
    /// The caller's memory of the log: the `tlog_checkpoint` a verified key
    /// returned last time, kept by the caller (the module keeps none).
    /// Absent the first time, or when it was lost: the proof's checkpoint
    /// is then kept as the first seen.
    #[serde(default)]
    tlog_checkpoint: Option<String>,
    /// A tlog-witness `add-checkpoint` body from the remembered size to the
    /// reference's checkpoint: required once the log has grown since.
    #[serde(default)]
    tlog_consistency: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NrasRequestIn {
    document: KeyDocument,
    client_nonce_hex: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SealIn {
    key: u32,
    /// The chat completion request, as JSON.
    body: Value,
    /// 32 fresh random bytes, hex.
    ikm_hex: String,
    /// The caller's clock (Unix seconds), sealed inside as the `Date`.
    now: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenIn {
    pending: u32,
    response_b64: String,
    /// The caller's clock (Unix seconds): a strict key's receipts are
    /// judged at it. Required since 2026-10-28.
    now: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PushIn {
    stream: u32,
    /// One frame of the chunked response, as the transport handed it over.
    bytes_b64: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FinishIn {
    stream: u32,
    /// The caller's clock (Unix seconds): a strict key's receipts are
    /// judged at it.
    now: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CloseIn {
    handle: u32,
}

/// An optional 48-byte pin, its field named in the error.
fn pin(field: &str, hex: Option<&String>) -> Result<Option<[u8; 48]>, String> {
    hex.map(|h| crate::unhex::<48>(h).map_err(|e| format!("{field}: {e}")))
        .transpose()
}

/// Why a call failed; a refused key carries the relay's code for it, and
/// the step that refused.
struct Failure {
    error: String,
    code: Option<i32>,
    step: Option<&'static str>,
    /// What the TD said about an answer no receipt covers.
    unverified: Option<Value>,
}

impl From<String> for Failure {
    fn from(error: String) -> Self {
        Self {
            error,
            code: None,
            step: None,
            unverified: None,
        }
    }
}

impl From<&str> for Failure {
    fn from(error: &str) -> Self {
        error.to_string().into()
    }
}

impl From<attest::Refusal> for Failure {
    fn from(r: attest::Refusal) -> Self {
        Self {
            error: format!("refused at {r}"),
            code: Some(r.code),
            step: Some(r.step),
            unverified: None,
        }
    }
}

fn policy(
    p: PolicyIn,
    memory: &mut attest::LogMemory,
    consistency: Option<&str>,
    now: u64,
) -> Result<(Policy, String), Failure> {
    // What the build pins is settled before anything is admitted under it.
    let root = p.root_der_b64.as_deref().map(test_root).transpose()?;
    let gce_root = p
        .gce_root_der_b64
        .as_deref()
        .map(test_gce_root)
        .transpose()?;
    let nras_pin = p
        .nras_intermediate_der_b64
        .as_deref()
        .map(test_nras_pin)
        .transpose()?;
    let signature = b64d(&p.reference_sig_b64).map_err(|e| format!("reference_sig_b64: {e}"))?;
    let signed = attest::SignedReference {
        json: p.reference_json.as_bytes(),
        signature: &signature,
        tlog_proof: &p.reference_tlog_proof,
    };
    let reference = match p.reference_keys.as_deref() {
        None => attest::admit_reference(&signed, memory, consistency, now)?,
        Some("test") => test_reference(&signed, memory, consistency, now)?,
        Some(other) => {
            return Err(format!("reference_keys: {other:?}, only \"test\" is known").into())
        }
    };
    let mut policy = Policy::production(reference, &p.machine_type);
    match p.tcb.as_deref().map(crate::TcbMode::parse).transpose() {
        Ok(Some(crate::TcbMode::Strict)) => policy = policy.with_strict_tcb(),
        Ok(_) => {}
        Err(e) => return Err(format!("tcb: {e}").into()),
    }
    match p
        .nras_ocsp
        .as_deref()
        .map(crate::NrasOcspMode::parse)
        .transpose()
    {
        Ok(Some(crate::NrasOcspMode::Required)) => policy = policy.with_required_nras_ocsp(),
        Ok(_) => {}
        Err(e) => return Err(format!("nras_ocsp: {e}").into()),
    }
    policy.mrtd_pin = pin("mrtd_hex", p.mrtd_hex.as_ref())?;
    policy.rtmr_pins = [
        pin("rtmr1_hex", p.rtmr1_hex.as_ref())?,
        pin("rtmr2_hex", p.rtmr2_hex.as_ref())?,
        pin("rtmr3_hex", p.rtmr3_hex.as_ref())?,
    ];
    policy.rtmr0_pin = pin("rtmr0_hex", p.rtmr0_hex.as_ref())?;
    if let Some(secs) = p.max_collateral_age_secs {
        policy = policy
            .with_max_collateral_age(secs)
            .map_err(|e| format!("max_collateral_age_secs: {e}"))?;
    }
    if let Some(root) = root {
        policy.anchor = root;
    }
    if let Some(root) = gce_root {
        policy.gce_root_der = root;
    }
    policy.attest = attest::Attest::parse(&p.attest).map_err(|e| format!("attest: {e}"))?;
    if let Some(pin) = nras_pin {
        policy.nras_pins = vec![pin];
    }
    let model = p.model_digest_hex;
    if model.len() != 64 || !model.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("model_digest_hex: expected 64 hex digits".into());
    }
    Ok((policy, model))
}

#[cfg(feature = "test-anchor")]
fn test_reference(
    signed: &attest::SignedReference<'_>,
    memory: &mut attest::LogMemory,
    consistency: Option<&str>,
    now: u64,
) -> Result<attest::AdmittedReference, Failure> {
    Ok(attest::admit_reference_under_test_keys(
        signed,
        memory,
        consistency,
        now,
    )?)
}

#[cfg(not(feature = "test-anchor"))]
fn test_reference(
    _: &attest::SignedReference<'_>,
    _: &mut attest::LogMemory,
    _: Option<&str>,
    _: u64,
) -> Result<attest::AdmittedReference, Failure> {
    Err("reference_keys: this build pins x0ne's archive key and log and takes no other".into())
}

#[cfg(feature = "test-anchor")]
fn test_root(root_der_b64: &str) -> Result<crate::TrustAnchor, String> {
    Ok(crate::TrustAnchor::RootDer(b64d(root_der_b64)?))
}

#[cfg(not(feature = "test-anchor"))]
fn test_root(_: &str) -> Result<crate::TrustAnchor, String> {
    Err("root_der_b64: this build pins Intel's root and takes no other".into())
}

#[cfg(feature = "test-anchor")]
fn test_gce_root(root_der_b64: &str) -> Result<Vec<u8>, String> {
    b64d(root_der_b64).map_err(|e| format!("gce_root_der_b64: {e}"))
}

#[cfg(not(feature = "test-anchor"))]
fn test_gce_root(_: &str) -> Result<Vec<u8>, String> {
    Err("gce_root_der_b64: this build pins Google's root and takes no other".into())
}

#[cfg(feature = "test-anchor")]
fn test_nras_pin(der_b64: &str) -> Result<crate::NrasIntermediatePin, String> {
    crate::nras_test_pin(&b64d(der_b64)?).map_err(|e| format!("nras_intermediate_der_b64: {e}"))
}

#[cfg(not(feature = "test-anchor"))]
fn test_nras_pin(_: &str) -> Result<crate::NrasIntermediatePin, String> {
    Err(NO_TEST_NRAS_PIN.into())
}

#[cfg(not(feature = "test-anchor"))]
const NO_TEST_NRAS_PIN: &str =
    "nras_intermediate_der_b64: this build pins NVIDIA's intermediates and takes no other";

fn handle(st: &mut State) -> Result<u32, Failure> {
    if st.keys.len() + st.pending.len() + st.streams.len() >= MAX_HANDLES {
        return Err("too many open handles".into());
    }
    st.next = st.next.wrapping_add(1).max(1);
    Ok(st.next)
}

/// One call, in Rust: what [`x0ne_call`] runs on the bytes it was given.
pub fn call(op: u32, input: &[u8]) -> Value {
    let out = match op {
        OP_VERIFY_KEY => verify(input),
        OP_SEAL_CHAT => seal(input),
        OP_OPEN => open(input),
        OP_CLOSE => close(input),
        OP_NRAS_REQUEST => nras_request(input),
        OP_SEAL_CHAT_STREAM => seal_stream(input),
        OP_STREAM_PUSH => push(input),
        OP_STREAM_FINISH => finish(input),
        #[cfg(feature = "test-anchor")]
        OP_TEST_PANIC => with_state(|_| panic!("OP_TEST_PANIC")),
        other => Err(format!("unknown op {other}").into()),
    };
    match out {
        Ok(mut v) => {
            v["ok"] = json!(true);
            v
        }
        Err(f) => {
            let mut v = json!({ "ok": false, "error": f.error });
            if let Some(code) = f.code {
                v["code"] = json!(code);
                v["step"] = json!(f.step);
            }
            if let Some(unverified) = f.unverified {
                v["unverified"] = unverified;
            }
            v
        }
    }
}

/// An answer that came with no checked receipt: an error, what the TD said
/// apart.
fn unchecked(error: String, unverified: Value) -> Failure {
    Failure {
        error,
        code: None,
        step: None,
        unverified: Some(unverified),
    }
}

fn verify(input: &[u8]) -> Result<Value, Failure> {
    let v: VerifyIn = serde_json::from_slice(input).map_err(|e| format!("input: {e}"))?;
    let nonce =
        crate::unhex::<32>(&v.client_nonce_hex).map_err(|e| format!("client_nonce_hex: {e}"))?;
    let mut memory = match v.tlog_checkpoint {
        Some(note) => attest::LogMemory::from_note(note),
        None => attest::LogMemory::new(),
    };
    let (policy, model_digest_hex) =
        policy(v.policy, &mut memory, v.tlog_consistency.as_deref(), v.now)?;
    let key = attest::verify_key(&v.document, &v.collateral, &policy, &nonce, v.now)?;
    let summary = json!({
        "tcb_platform": key.tcb_platform,
        "tcb_qe": key.tcb_qe,
        "tcb": key.tcb,
        "mrtd_hex": key.mrtd_hex,
        "rtmr0_hex": key.rtmr0_hex,
        "firmware_sha384_hex": key.firmware_sha384_hex,
        "image": key.image,
        "reference": key.reference,
        "warnings": key.warnings,
        "gpu_ueids": key.gpu_ueids,
        "nras_revocation": key.nras_revocation,
        // The session config the quote bound: the model, expires_at_unix, the
        // relay's TLS key (client_spki_sha256), read before anything is sealed.
        "session_config": key.session_config,
        "session_config_sha256_hex": key.session_config_sha256_hex,
        "boot_nonce_hex": key.boot_nonce_hex,
        "tls_spki_sha256_hex": key.tls_spki_sha256_hex,
        "session_pubkey_hex": key.session_pubkey().iter().map(|b| format!("{b:02x}")).collect::<String>(),
        "limitations": key.limitations,
        // The caller's memory of the log after this admission: keep it, hand
        // it back as `tlog_checkpoint` next time.
        "tlog_checkpoint": memory.note(),
    });
    with_state(|st| {
        let h = handle(st)?;
        st.keys.insert(
            h,
            Key {
                client: Client::new(key),
                model_digest_hex,
            },
        );
        Ok(json!({ "key": h, "verified": summary }))
    })
}

fn nras_request(input: &[u8]) -> Result<Value, Failure> {
    let v: NrasRequestIn = serde_json::from_slice(input).map_err(|e| format!("input: {e}"))?;
    let nonce =
        crate::unhex::<32>(&v.client_nonce_hex).map_err(|e| format!("client_nonce_hex: {e}"))?;
    Ok(json!({ "request": attest::nras_request(&v.document, &nonce)? }))
}

fn seal(input: &[u8]) -> Result<Value, Failure> {
    let v: SealIn = serde_json::from_slice(input).map_err(|e| format!("input: {e}"))?;
    let ikm = crate::unhex::<32>(&v.ikm_hex).map_err(|e| format!("ikm_hex: {e}"))?;
    let body = serde_json::to_vec(&v.body).map_err(|e| e.to_string())?;
    with_state(|st| {
        let key = st
            .keys
            .get(&v.key)
            .ok_or_else(|| format!("no verified key {}", v.key))?;
        let (request, pending) = key
            .client
            .seal_chat(&body, &ikm, v.now)
            .map_err(|e| e.to_string())?;
        let pin = key.model_digest_hex.clone();
        let h = handle(st)?;
        st.pending.insert(h, (pending, pin));
        Ok(json!({ "request_b64": b64e(&request), "pending": h }))
    })
}

fn open(input: &[u8]) -> Result<Value, Failure> {
    let v: OpenIn = serde_json::from_slice(input).map_err(|e| format!("input: {e}"))?;
    let response = b64d(&v.response_b64).map_err(|e| format!("response_b64: {e}"))?;
    let (pending, pin) = with_state(|st| st.pending.remove(&v.pending))
        .ok_or_else(|| format!("no pending request {}", v.pending))?;
    let (answer, checked) = pending
        .open_verified(&response, &pin, v.now)
        .map_err(|e| e.to_string())?;
    let body: Value = serde_json::from_slice(&answer.body).unwrap_or(Value::Null);
    let Some(c) = checked else {
        return Err(unchecked(
            format!(
                "no checked receipt: the TD answered {}, its body unsigned (`unverified`)",
                answer.status
            ),
            json!({ "status": answer.status, "body": body }),
        ));
    };
    Ok(json!({
        "status": answer.status,
        "text": c.text,
        "receipt": receipt(&c),
        // The receipt signs the text: not the usage, the finish_reason, the rest.
        "unverified": { "body": body },
    }))
}

/// The receipt of a checked answer, as every op that opens one returns it.
fn receipt(c: &crate::receipt::Checked) -> Value {
    json!({
        "checked": true,
        "model_digest_hex": c.model_digest_hex,
        "compute_device": c.compute_device,
        "tcb": c.signed.get("tcb"),
        "signed": c.signed,
    })
}

fn seal_stream(input: &[u8]) -> Result<Value, Failure> {
    let v: SealIn = serde_json::from_slice(input).map_err(|e| format!("input: {e}"))?;
    let ikm = crate::unhex::<32>(&v.ikm_hex).map_err(|e| format!("ikm_hex: {e}"))?;
    let body = serde_json::to_vec(&v.body).map_err(|e| e.to_string())?;
    with_state(|st| {
        let key = st
            .keys
            .get(&v.key)
            .ok_or_else(|| format!("no verified key {}", v.key))?;
        let (request, pending) = key
            .client
            .seal_chat_stream(&body, &ikm, v.now)
            .map_err(|e| e.to_string())?;
        let pin = key.model_digest_hex.clone();
        let h = handle(st)?;
        st.streams.insert(h, (pending, pin));
        Ok(json!({ "request_b64": b64e(&request), "stream": h }))
    })
}

/// One frame to a stream: the deltas it completed, NOT VERIFIED. A refused
/// frame leaves the stream held and refused, until `finish` or `close`.
fn push(input: &[u8]) -> Result<Value, Failure> {
    let v: PushIn = serde_json::from_slice(input).map_err(|e| format!("input: {e}"))?;
    let bytes = b64d(&v.bytes_b64).map_err(|e| format!("bytes_b64: {e}"))?;
    with_state(|st| {
        let (pending, _) = st
            .streams
            .get_mut(&v.stream)
            .ok_or_else(|| format!("no pending stream {}", v.stream))?;
        let deltas = pending.push(&bytes).map_err(|e| e.to_string())?;
        let deltas: Vec<String> = deltas
            .into_iter()
            .map(crate::sealed_stream::UnverifiedDelta::into_unverified_text)
            .collect();
        Ok(json!({ "unverified_deltas": deltas }))
    })
}

/// The end of a stream's body: the handle is taken, the receipt of the
/// final chunk checked over the whole text.
fn finish(input: &[u8]) -> Result<Value, Failure> {
    let v: FinishIn = serde_json::from_slice(input).map_err(|e| format!("input: {e}"))?;
    let (pending, pin) = with_state(|st| st.streams.remove(&v.stream))
        .ok_or_else(|| format!("no pending stream {}", v.stream))?;
    let (answer, checked) = pending.finish(&pin, v.now).map_err(|e| e.to_string())?;
    let Some(c) = checked else {
        // A refusal, or a 200 ended by an error event: its text, the final
        // deltas among it, is unsigned; none of it comes back.
        return Err(unchecked(
            format!(
                "no checked receipt: the stream ended with {} and an error, unsigned \
                 (`unverified`); discard the deltas it pushed",
                answer.status
            ),
            json!({ "status": answer.status, "error": answer.error }),
        ));
    };
    let conversation = answer
        .conversation
        .as_ref()
        .map(|c| json!({ "handle": c.handle.hex() }));
    Ok(json!({
        "status": answer.status,
        "text": c.text,
        "final_deltas": answer.final_deltas,
        "receipt": receipt(&c),
        "chunks": answer.chunks,
        "conversation": conversation,
        // Not under the receipt (`receipt::check_answer_for`).
        "unverified": { "finish_reason": answer.finish_reason, "usage": answer.usage },
    }))
}

/// A key, a pending request or a stream the caller is done with, or whose
/// answer never came: its slot is free again.
fn close(input: &[u8]) -> Result<Value, Failure> {
    let v: CloseIn = serde_json::from_slice(input).map_err(|e| format!("input: {e}"))?;
    with_state(|st| {
        let key = st.keys.remove(&v.handle).is_some();
        if key || st.pending.remove(&v.handle).is_some() || st.streams.remove(&v.handle).is_some() {
            Ok(json!({}))
        } else {
            Err(format!("no handle {}", v.handle).into())
        }
    })
}

/// `len` bytes the caller writes its input into; [`x0ne_free`] returns them.
/// Null over [`MAX_INPUT_BYTES`]: an allocation that fails is a trap of the
/// instance, a null is the caller's error to handle. Null on a panic too.
#[no_mangle]
pub extern "C" fn x0ne_alloc(len: usize) -> *mut u8 {
    if len > MAX_INPUT_BYTES {
        return std::ptr::null_mut();
    }
    catch_unwind(|| {
        let mut buf = vec![0u8; len].into_boxed_slice();
        let ptr = buf.as_mut_ptr();
        std::mem::forget(buf);
        ptr
    })
    .unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `ptr` and `len` are exactly those of one [`x0ne_alloc`], not yet freed.
#[no_mangle]
pub unsafe extern "C" fn x0ne_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    // Dropping bytes cannot panic; caught all the same, as every export.
    let _ = catch_unwind(|| {
        // SAFETY: the caller's contract above: this box was leaked by x0ne_alloc(len).
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) })
    });
}

/// Runs `op` on the `len` bytes at `ptr`; returns the result's length, read
/// at [`x0ne_result_ptr`] on this thread until this thread's next call. A
/// panic is the result [`CODE_PANIC`] and drops every handle (module doc).
///
/// # Safety
/// `ptr` points to `len` readable bytes (an [`x0ne_alloc`] buffer).
#[no_mangle]
pub unsafe extern "C" fn x0ne_call(op: u32, ptr: *const u8, len: usize) -> usize {
    let input: &[u8] = if ptr.is_null() || len == 0 {
        &[]
    } else {
        // SAFETY: the caller's contract above.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    };
    let out = catch_unwind(AssertUnwindSafe(|| {
        serde_json::to_vec(&call(op, input)).unwrap_or_else(|_| b"{\"ok\":false}".to_vec())
    }))
    .unwrap_or_else(|_| {
        // Poisoned or not, the lock is taken back and the handles dropped.
        drop_handles(&mut lock_state());
        PANICKED.to_vec()
    });
    RESULT
        .try_with(|r| {
            let n = out.len();
            *r.borrow_mut() = out;
            n
        })
        // The thread is exiting: no result can be kept for it.
        .unwrap_or(0)
}

/// This thread's last result's bytes; valid until this thread's next
/// [`x0ne_call`]. Null when the thread has none to keep (it is exiting).
#[no_mangle]
pub extern "C" fn x0ne_result_ptr() -> *const u8 {
    catch_unwind(|| RESULT.try_with(|r| r.borrow().as_ptr()).ok())
        .ok()
        .flatten()
        .unwrap_or(std::ptr::null())
}
