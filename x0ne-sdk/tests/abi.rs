//! The C ABI's refusals, natively. Its accepting path (a verified key, a
//! sealed question, an opened answer) runs against the simulated TD in
//! x0ne-infer (`e2e_sdk_tests.rs`) and, built for wasm32, in Node
//! (`scripts/sdk-wasm.sh`).

use serde_json::{json, Value};
use x0ne_sdk::abi::{
    self, OP_NRAS_REQUEST, OP_OPEN, OP_SEAL_CHAT, OP_SEAL_CHAT_STREAM, OP_STREAM_FINISH,
    OP_STREAM_PUSH, OP_VERIFY_KEY,
};
use x0ne_sdk::attest::{CloudConfig, KEY_SCHEMA};

fn call(op: u32, v: Value) -> Value {
    abi::call(op, &serde_json::to_vec(&v).unwrap())
}

/// The image reference of the GCE run of 2026-10-18 (x0ne-cvm 0.2.3), and
/// its signature by the TEST archive key.
/// The reference of the 2026-10-18 run with RTMR0 pinned
/// (`real_quote_2026_10_18.rs`), signed here by the TEST key.
const REFERENCE: &str =
    include_str!("../../x0ne-infer/tests/run_fixtures/x0ne-cvm-0.2.3-rtmr0-pinned.json");
/// The run's published reference, which reports RTMR0, and its signature
/// by the TEST key: refused since 2026-10-31.
const REFERENCE_REPORT: &str =
    include_str!("../../x0ne-infer/tests/run_fixtures/x0ne-cvm-0.2.3.json");
const REFERENCE_REPORT_SIG: &[u8] =
    include_bytes!("../../x0ne-infer/tests/run_fixtures/x0ne-cvm-0.2.3.json.sig");
const MACHINE: &str = "a3-highgpu-1g";
/// The callers' clock in these tests.
const NOW: u64 = 1_800_000_000;

fn b64(b: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(b)
}

/// The policy fields that carry [`REFERENCE`]: its signature, a TEST log
/// that shows it, cosigned at [`NOW`], and the TEST keys named.
fn reference_fields() -> Value {
    let (sig, proof) = x0ne_sdk::attest::sign_reference_under_test_keys(REFERENCE.as_bytes(), NOW);
    json!({
        "reference_json": REFERENCE,
        "reference_sig_b64": b64(&sig),
        "reference_tlog_proof": proof,
        "reference_keys": "test",
    })
}

/// [`reference_fields`] and `rest`, one policy object.
fn with_reference(rest: Value) -> Value {
    let mut p = reference_fields();
    for (k, v) in rest.as_object().unwrap() {
        p[k] = v.clone();
    }
    p
}

/// The model digest the receipts must sign: required in the policy.
const MODEL: &str = "ab";

fn model() -> String {
    MODEL.repeat(32)
}

/// Op 4, as the JavaScript loader names the ops: by number.
const OP_CLOSE: u32 = 4;

fn policy() -> Value {
    with_reference(
        json!({ "machine_type": MACHINE, "model_digest_hex": model(), "attest": "cpu+gpu" }),
    )
}

/// A session config as the relay writes it: `x0ne-engine/cloud-config/v1`.
fn session_config() -> Vec<u8> {
    serde_json::to_vec(&CloudConfig {
        schema: x0ne_sdk::attest::SESSION_CONFIG_SCHEMA.into(),
        model: "qwen3:0.6b".into(),
        client_spki_sha256: "52".repeat(32),
        client_nonce: "43".repeat(32),
        expires_at_unix: NOW + 3600,
        attest: "cpu+gpu".into(),
        listen_port: 443,
        max_completion_tokens: 64,
        sealed: None,
        e2e_stream: None,
    })
    .unwrap()
}

/// A key document of the suite this SDK speaks, its receipt key `receipt_hex`,
/// a session config and a boot binding, with no quote: what passes `schema`
/// and `key` is refused at `quote`.
fn document(schema: &str, receipt_hex: &str) -> Value {
    use base64::Engine as _;
    let config = x0ne_sdk::e2e::Gateway::from_seed(&[0x31; 32]).key_config();
    json!({
        "schema": schema,
        "key_config_b64": base64::engine::general_purpose::STANDARD.encode(config),
        "session_pubkey_hex": receipt_hex,
        "session_config_b64": b64(&session_config()),
        "boot_nonce_hex": "09".repeat(32),
        "tls_spki_sha256_hex": "54".repeat(32),
        "quote_b64": "",
        "ccel_b64": "",
    })
}

fn verify(document: Value, policy: Value) -> Value {
    call(
        OP_VERIFY_KEY,
        json!({ "document": document, "collateral": empty_collateral(), "policy": policy,
                "client_nonce_hex": "00".repeat(32), "now": NOW }),
    )
}

fn error(out: &Value) -> &str {
    assert_eq!(out["ok"], false, "{out}");
    out["error"].as_str().unwrap()
}

/// The y-coordinate 2 decompresses to no Ed25519 point (x² is not a square
/// mod 2^255 - 19); 3 does.
fn not_a_point() -> String {
    format!("02{}", "00".repeat(31))
}

fn a_point() -> String {
    format!("03{}", "00".repeat(31))
}

/// SECURITY: a document of another schema is
/// refused at `schema`, before any of its fields is read.
#[test]
fn a_document_of_another_schema_is_refused_at_schema() {
    let out = verify(document("x0ne-e2e/key/v0", &a_point()), policy());
    assert!(error(&out).starts_with("refused at schema:"), "{out}");
    // v1 (before 2026-10-30) bound no session config: refused unread.
    let out = verify(document("x0ne-e2e/key/v1", &a_point()), policy());
    assert!(error(&out).starts_with("refused at schema:"), "{out}");
    assert_eq!(out["code"], 88, "{out}");
    let control = verify(document(KEY_SCHEMA, &a_point()), policy());
    assert!(
        error(&control).starts_with("refused at quote:"),
        "{control}"
    );
}

/// SECURITY: a valid key configuration next to a receipt key
/// that is no Ed25519 point is refused at `key`: no receipt could check.
#[test]
fn a_receipt_key_that_is_no_ed25519_point_is_refused_at_key() {
    let out = verify(document(KEY_SCHEMA, &not_a_point()), policy());
    assert!(
        error(&out).starts_with("refused at key: session_pubkey_hex"),
        "{out}"
    );
}

/// SECURITY: the relay refuses a model other than its
/// catalogue's; the SDK, a model other than the digest its caller names.
/// A policy without one is refused before anything is verified.
#[test]
fn a_policy_without_the_models_digest_is_refused() {
    let mut p = policy();
    p.as_object_mut().unwrap().remove("model_digest_hex");
    let out = verify(document(KEY_SCHEMA, &a_point()), p);
    assert!(error(&out).contains("model_digest_hex"), "{out}");
    let mut p = policy();
    p["model_digest_hex"] = json!("ab".repeat(31));
    let out = verify(document(KEY_SCHEMA, &a_point()), p);
    assert!(
        error(&out).contains("model_digest_hex: expected 64 hex digits"),
        "{out}"
    );
}

/// SECURITY: the caller may shorten the collateral's maximum
/// age (the relay's day), never lengthen it.
#[test]
fn the_collateral_age_can_be_shortened_not_lengthened() {
    let mut p = policy();
    p["max_collateral_age_secs"] = json!(2 * 86_400);
    let out = verify(document(KEY_SCHEMA, &a_point()), p);
    assert!(
        error(&out).contains("it can be shortened, not lengthened"),
        "{out}"
    );
    let mut p = policy();
    p["max_collateral_age_secs"] = json!(3600);
    let out = verify(document(KEY_SCHEMA, &a_point()), p);
    assert!(error(&out).starts_with("refused at quote:"), "{out}");
}

/// The reference is read under the relay's cap on its file.
#[test]
fn a_reference_over_the_relays_cap_is_refused() {
    let mut p = policy();
    p["reference_json"] = json!(" ".repeat(1024 * 1024 + 1));
    let out = verify(document(KEY_SCHEMA, &a_point()), p);
    assert!(
        error(&out).contains("refused at reference: 1048577 bytes, cap 1048576"),
        "{out}"
    );
    assert_eq!(out["code"], 89, "{out}");
}

/// SECURITY: the reference is admitted as the relay admits it, before any
/// key: not signed by the pinned archive key, refused with 99; not shown in
/// the pinned log, refused with 100. The refusal carries the relay's code.
#[test]
fn a_reference_unsigned_or_outside_the_log_is_refused_with_the_relays_codes() {
    let mut p = policy();
    let (mut sig, _) = x0ne_sdk::attest::sign_reference_under_test_keys(REFERENCE.as_bytes(), NOW);
    let n = sig.len();
    sig[n - 3] ^= 1;
    p["reference_sig_b64"] = json!(b64(&sig));
    let out = verify(document(KEY_SCHEMA, &a_point()), p);
    assert!(
        error(&out).starts_with("refused at reference: not signed by the x0ne archive key"),
        "{out}"
    );
    assert_eq!(out["code"], 99, "{out}");
    let mut p = policy();
    let (_, other) = x0ne_sdk::attest::sign_reference_under_test_keys(b"another file", NOW);
    p["reference_tlog_proof"] = json!(other);
    let out = verify(document(KEY_SCHEMA, &a_point()), p);
    assert!(
        error(&out).starts_with("refused at reference: not in the transparency log"),
        "{out}"
    );
    assert_eq!(out["code"], 100, "{out}");
}

/// SECURITY: the module keeps no memory of the log of its own; the caller
/// hands back `tlog_checkpoint`, and a reference whose checkpoint forks it,
/// or outgrows it without `tlog_consistency`, is refused with 100 at a step
/// named in the result (`step`), before any key is read.
#[test]
fn a_log_that_forks_or_outgrows_the_callers_checkpoint_is_refused_with_100_and_named() {
    use x0ne_sdk::attest::TestLog;
    let mut log = TestLog::new();
    log.append(b"entry 0");
    log.append(REFERENCE.as_bytes());
    let mut fork = TestLog::new();
    fork.append(b"another entry 0");
    fork.append(REFERENCE.as_bytes());
    let (sig, _) = x0ne_sdk::attest::sign_reference_under_test_keys(REFERENCE.as_bytes(), NOW);
    let proof_of = |l: &TestLog| {
        json!({
            "reference_json": REFERENCE,
            "reference_sig_b64": b64(&sig),
            "reference_tlog_proof": l.proof(1, NOW),
            "reference_keys": "test",
            "machine_type": MACHINE,
            "model_digest_hex": model(),
            "attest": "cpu+gpu",
        })
    };
    let remembered = log.checkpoint_note(NOW);
    let mut input = json!({
        "document": document(KEY_SCHEMA, &a_point()),
        "collateral": empty_collateral(),
        "policy": proof_of(&fork),
        "client_nonce_hex": "11".repeat(32),
        "now": NOW,
        "tlog_checkpoint": remembered,
    });
    let out = call(OP_VERIFY_KEY, input.clone());
    assert!(
        error(&out).starts_with("refused at log-fork: the log forked"),
        "{out}"
    );
    assert_eq!(
        (out["code"].as_i64(), out["step"].as_str()),
        (Some(100), Some("log-fork")),
        "{out}"
    );
    log.append(b"entry 2");
    input["policy"] = proof_of(&log);
    let out = call(OP_VERIFY_KEY, input.clone());
    assert!(
        error(&out).starts_with("refused at log-consistency: "),
        "{out}"
    );
    assert_eq!(
        (out["code"].as_i64(), out["step"].as_str()),
        (Some(100), Some("log-consistency")),
        "{out}"
    );
    // The control: with the proof from size 2, the reference passes and the
    // next refusal is the key document's, not the log's.
    input["tlog_consistency"] = json!(log.consistency(2, NOW));
    let out = call(OP_VERIFY_KEY, input);
    assert!(
        !error(&out).starts_with("refused at log-")
            && !error(&out).starts_with("refused at reference"),
        "{out}"
    );
}

/// Without `reference_keys`, the production anchors decide. A debug build
/// pins the TEST key and log there (`Anchor::production`): the reference
/// passes and the next refusal is the key document's. A release build made
/// before the signing ceremony pins nothing: refused at `reference` with 99,
/// before the log is read.
#[test]
fn without_reference_keys_the_production_anchors_decide() {
    let mut p = policy();
    p.as_object_mut().unwrap().remove("reference_keys");
    let out = verify(document(KEY_SCHEMA, &a_point()), p);
    if cfg!(debug_assertions) {
        assert!(
            !error(&out).starts_with("refused at reference")
                && !error(&out).starts_with("refused at log-"),
            "{out}"
        );
    } else {
        assert!(error(&out).starts_with("refused at reference: "), "{out}");
        assert_eq!(
            (out["code"].as_i64(), out["step"].as_str()),
            (Some(99), Some("reference")),
            "{out}"
        );
    }
}

/// SECURITY: a reference that reports RTMR0 instead of pinning it (the
/// published x0ne-cvm 0.2.3, under its own TEST signature of 2026-10-18)
/// passes its signature and log, then is refused with the relay's 104.
#[test]
fn a_reference_reporting_rtmr0_is_refused_with_104() {
    let mut p = policy();
    let (_, proof) =
        x0ne_sdk::attest::sign_reference_under_test_keys(REFERENCE_REPORT.as_bytes(), NOW);
    p["reference_json"] = json!(REFERENCE_REPORT);
    p["reference_sig_b64"] = json!(b64(REFERENCE_REPORT_SIG));
    p["reference_tlog_proof"] = json!(proof);
    let out = verify(document(KEY_SCHEMA, &a_point()), p);
    assert!(
        error(&out).starts_with("refused at reference: RTMR0 not pinned"),
        "{out}"
    );
    assert_eq!(out["code"], 104, "{out}");
}

/// A caller written before 2026-10-28 sends no signature, no log proof and
/// no clock to `open`: refused, each by the field's name, never admitted
/// on the older rules. Unknown spellings of the new fields are refused too.
#[test]
fn a_caller_from_before_is_refused_by_the_fields_it_lacks() {
    for field in ["reference_sig_b64", "reference_tlog_proof"] {
        let mut p = policy();
        p.as_object_mut().unwrap().remove(field);
        let out = verify(document(KEY_SCHEMA, &a_point()), p);
        assert!(
            error(&out).contains(&format!("missing field `{field}`")),
            "{out}"
        );
    }
    let out = call(OP_OPEN, json!({ "pending": 1, "response_b64": "" }));
    assert!(error(&out).contains("missing field `now`"), "{out}");
    let mut p = policy();
    p["reference_keys"] = json!("production");
    let out = verify(document(KEY_SCHEMA, &a_point()), p);
    assert!(error(&out).contains("only \"test\" is known"), "{out}");
    let mut p = policy();
    p["tcb"] = json!("lenient");
    let out = verify(document(KEY_SCHEMA, &a_point()), p);
    assert!(error(&out).starts_with("tcb: "), "{out}");
    let mut p = policy();
    p["tcb"] = json!("strict");
    let out = verify(document(KEY_SCHEMA, &a_point()), p);
    assert!(error(&out).starts_with("refused at quote:"), "{out}");
}

/// The TEST archive key and log are behind the `test-anchor` feature alone,
/// as the test root is: a build without it refuses `"reference_keys": "test"`.
#[test]
fn the_test_archive_key_and_log_are_behind_the_test_anchor_feature_alone() {
    let src = include_str!("../src/abi.rs");
    assert!(src.contains("#[cfg(feature = \"test-anchor\")]\nfn test_reference("));
    assert!(src.contains("#[cfg(not(feature = \"test-anchor\"))]\nfn test_reference("));
    assert!(src.contains(
        "Err(\"reference_keys: this build pins x0ne's archive key and log and takes no other\".into())"
    ));
    let attest = include_str!("../src/attest.rs");
    assert!(attest
        .contains("#[cfg(feature = \"test-anchor\")]\npub fn admit_reference_under_test_keys("));
    assert!(attest
        .contains("#[cfg(feature = \"test-anchor\")]\npub fn sign_reference_under_test_keys("));
    let key = include_str!("../../x0ne-infer/src/archive_key.rs");
    assert!(key.contains("#[cfg(x0ne_sdk_test_anchor)]\nimpl Anchor {"));
    let log = include_str!("../../x0ne-infer/src/transparency.rs");
    assert!(log.contains("#[cfg(x0ne_sdk_test_anchor)]\nimpl Anchor {"));
}

/// The instance allocates no input over its cap; the loader
/// throws on the null pointer instead of writing at address 0.
#[test]
fn an_input_over_the_cap_gets_no_buffer() {
    let cap = 64 * 1024 * 1024;
    let big = abi::x0ne_alloc(cap + 1);
    if !big.is_null() {
        // SAFETY: the pointer and length of the allocation just made.
        unsafe { abi::x0ne_free(big, cap + 1) };
    }
    assert!(big.is_null(), "{} bytes allocated", cap + 1);
    let small = abi::x0ne_alloc(16);
    assert!(!small.is_null());
    // SAFETY: as above.
    unsafe { abi::x0ne_free(small, 16) };
}

/// A handle that was never issued closes nothing; the bench
/// closes issued ones (`e2e_sdk_tests`).
#[test]
fn closing_a_handle_that_was_never_issued_is_an_error() {
    let out = call(OP_CLOSE, json!({ "handle": 4242 }));
    assert!(error(&out).contains("no handle 4242"), "{out}");
}

/// The limitations returned with every key
/// say what the carrier and the caller choose. A line was dropped once
/// (2026-10-27); this test keeps them.
#[test]
fn the_limitations_say_what_the_carrier_and_the_caller_choose() {
    let all = x0ne_sdk::LIMITATIONS.join("\n");
    for says in [
        "whoever supplies it chooses the image accepted",
        "never from the carrier",
        "can relay genuine collateral older than Intel's newest",
        "up to a day old",
        "root CA CRL until it expires",
        "the model is the one whose digest the caller names",
    ] {
        assert!(all.contains(says), "{says:?} missing from LIMITATIONS");
    }
}

fn empty_collateral() -> Value {
    json!({
        "tcb_info": "", "tcb_info_chain_pem": "", "qe_identity": "",
        "qe_identity_chain_pem": "", "pck_crl_der_b64": "",
        "root_ca_crl_der_b64": "", "gce_endorsement_b64": "",
    })
}

#[test]
fn nothing_is_sealed_without_a_key_this_instance_verified() {
    let out = call(
        OP_SEAL_CHAT,
        json!({ "key": 1, "body": {"messages": []}, "ikm_hex": "11".repeat(32), "now": 1_800_000_000u64 }),
    );
    assert_eq!(out["ok"], false, "{out}");
    assert!(
        out["error"].as_str().unwrap().contains("no verified key"),
        "{out}"
    );
    let streamed = call(
        OP_SEAL_CHAT_STREAM,
        json!({ "key": 1, "body": {"stream": true, "messages": []}, "ikm_hex": "11".repeat(32), "now": NOW }),
    );
    assert!(error(&streamed).contains("no verified key"), "{streamed}");
}

/// A stream handle this instance never issued takes no bytes and finishes
/// nothing; the TD's routes test issued ones (`e2e_td_tests`).
#[test]
fn a_stream_handle_that_was_never_issued_takes_and_finishes_nothing() {
    let pushed = call(OP_STREAM_PUSH, json!({ "stream": 7, "bytes_b64": "" }));
    assert!(error(&pushed).contains("no pending stream 7"), "{pushed}");
    let finished = call(OP_STREAM_FINISH, json!({ "stream": 7, "now": NOW }));
    assert!(
        error(&finished).contains("no pending stream 7"),
        "{finished}"
    );
}

#[test]
fn a_pending_handle_that_was_never_issued_opens_nothing() {
    let out = call(
        OP_OPEN,
        json!({ "pending": 77, "response_b64": "", "now": NOW }),
    );
    assert_eq!(out["ok"], false, "{out}");
    assert!(
        out["error"]
            .as_str()
            .unwrap()
            .contains("no pending request"),
        "{out}"
    );
}

#[test]
fn malformed_input_and_unknown_ops_are_errors_not_panics() {
    assert_eq!(abi::call(OP_VERIFY_KEY, b"not json")["ok"], false);
    assert_eq!(abi::call(OP_SEAL_CHAT, b"{}")["ok"], false);
    assert_eq!(abi::call(OP_SEAL_CHAT_STREAM, b"{}")["ok"], false);
    assert_eq!(abi::call(OP_STREAM_PUSH, b"not json")["ok"], false);
    assert_eq!(abi::call(OP_STREAM_FINISH, b"[]")["ok"], false);
    assert_eq!(abi::call(99, b"{}")["ok"], false);
    let unknown_field = call(
        OP_STREAM_PUSH,
        json!({ "stream": 1, "bytes_b64": "", "extra": 1 }),
    );
    assert_eq!(unknown_field["ok"], false);
    let unknown_field = call(
        OP_OPEN,
        json!({ "pending": 1, "response_b64": "", "now": NOW, "extra": 1 }),
    );
    assert_eq!(unknown_field["ok"], false);
}

#[test]
fn a_garbage_document_is_refused_at_its_step() {
    let out = call(
        OP_VERIFY_KEY,
        json!({
            "document": {
                "schema": KEY_SCHEMA,
                "key_config_b64": "AQAg",
                "session_pubkey_hex": "00".repeat(32),
                "quote_b64": "",
                "ccel_b64": "",
            },
            "collateral": empty_collateral(),
            "policy": with_reference(json!({
                "machine_type": MACHINE,
                "mrtd_hex": "00".repeat(48), "rtmr1_hex": "00".repeat(48),
                "rtmr2_hex": "00".repeat(48), "rtmr3_hex": "00".repeat(48),
                "model_digest_hex": model(),
                "attest": "cpu+gpu",
            })),
            "client_nonce_hex": "00".repeat(32),
            "now": NOW,
        }),
    );
    assert_eq!(out["ok"], false, "{out}");
    assert!(
        out["error"]
            .as_str()
            .unwrap()
            .starts_with("refused at key:"),
        "{out}"
    );
}

/// The test root is read only in a `test-anchor` build. This crate's own
/// tests enable it; the source keeps the reading behind the feature.
#[test]
fn the_test_root_is_behind_the_test_anchor_feature_alone() {
    let src = include_str!("../src/abi.rs");
    let gated = "#[cfg(feature = \"test-anchor\")]\nfn test_root(root_der_b64: &str)";
    assert!(src.contains(gated), "test_root lost its feature gate");
    assert_eq!(src.matches("TrustAnchor::RootDer").count(), 1);
    let refusing = "#[cfg(not(feature = \"test-anchor\"))]\nfn test_root(_: &str)";
    assert!(src.contains(refusing));
    assert!(src
        .contains("Err(\"root_der_b64: this build pins Intel's root and takes no other\".into())"));
    let manifest = include_str!("../Cargo.toml");
    assert!(manifest.contains("test-anchor = [\"test-archive-key\"]"));
    assert!(manifest.contains("test-archive-key = []"));
    assert!(!manifest.contains("default = [\"test-archive-key\"]"));
    assert!(
        !manifest.contains("default = [\"test-anchor\"]"),
        "test-anchor must never be a default"
    );
}

/// SECURITY: the policy says whether the TD's GPUs are attested, in the
/// relay's spelling; a policy that does not say is refused, unread.
#[test]
fn a_policy_without_attest_or_with_another_spelling_is_refused() {
    let mut p = policy();
    p.as_object_mut().unwrap().remove("attest");
    let out = verify(document(KEY_SCHEMA, &a_point()), p);
    assert!(error(&out).contains("attest"), "{out}");
    for other in ["gpu", "CPU", "cpu+gpu+tpm", ""] {
        let mut p = policy();
        p["attest"] = json!(other);
        let out = verify(document(KEY_SCHEMA, &a_point()), p);
        assert!(error(&out).starts_with("attest:"), "{other:?}: {out}");
    }
    // A TD serves only on its attested GPU (2026-11-04): a build without
    // test-anchor refuses `cpu` by name, never as an unknown value; this
    // test-anchor build keeps it for the simulated TDs.
    use x0ne_sdk::attest::{Attest, ATTEST_CPU_REFUSED};
    assert_eq!(
        Attest::parse_in("cpu", false),
        Err(ATTEST_CPU_REFUSED.into())
    );
    assert_eq!(Attest::parse_in("cpu+gpu", false), Ok(Attest::CpuGpu));
    assert_eq!(
        Attest::parse_in("gpu", false),
        Err("\"gpu\", expected \"cpu+gpu\"".into())
    );
    assert_eq!(Attest::parse_in("cpu", true), Ok(Attest::Cpu));
    let mut p = policy();
    p["attest"] = json!("cpu+gpu");
    let out = verify(document(KEY_SCHEMA, &a_point()), p);
    assert!(error(&out).starts_with("refused at quote:"), "{out}");
    let attest = include_str!("../src/attest.rs");
    assert!(attest.contains("    #[cfg(feature = \"test-anchor\")]\n    Cpu,\n"));
    assert!(attest.contains("        Self::parse_in(s, cfg!(feature = \"test-anchor\"))\n"));
    assert!(attest.contains(
        "            #[cfg(feature = \"test-anchor\")]\n            \"cpu\" if simulated_cpu => Ok(Self::Cpu),\n"
    ));
    assert!(attest.contains("            attest: Attest::CpuGpu,\n"));
}

/// The bench's NVIDIA intermediate, like the bench's Intel root, is read
/// only in a `test-anchor` build; an NRAS pin is built in `nras.rs` alone,
/// a bench's under the cfg this crate's build script sets for that feature.
#[test]
fn the_test_nras_pin_is_behind_the_test_anchor_feature_alone() {
    let src = include_str!("../src/abi.rs");
    assert!(src.contains("#[cfg(feature = \"test-anchor\")]\nfn test_nras_pin(der_b64: &str)"));
    assert!(src.contains("#[cfg(not(feature = \"test-anchor\"))]\nfn test_nras_pin(_: &str)"));
    assert!(src.contains("#[cfg(not(feature = \"test-anchor\"))]\nfn test_nras_pin(_: &str) -> Result<crate::NrasIntermediatePin, String> {\n    Err(NO_TEST_NRAS_PIN.into())\n}"));
    assert!(src.contains(
        "\"nras_intermediate_der_b64: this build pins NVIDIA's intermediates and takes no other\";"
    ));
    let lib = include_str!("../src/lib.rs");
    assert!(lib.contains("#[cfg(feature = \"test-anchor\")]\npub fn nras_test_pin("));
    let build = include_str!("../build.rs");
    assert!(build.contains(
        "if std::env::var_os(\"CARGO_FEATURE_TEST_ANCHOR\").is_some() {\n        println!(\"cargo::rustc-cfg=x0ne_sdk_test_anchor\");"
    ));
    let nras = include_str!("../../x0ne-infer/src/nras.rs");
    assert!(nras.contains("#[cfg(x0ne_sdk_test_anchor)]\npub fn throwaway_pin("));
    // In this build (test-anchor), a DER that is no certificate is an error naming the field.
    let mut p = policy();
    p["attest"] = json!("cpu+gpu");
    p["nras_intermediate_der_b64"] = json!("AAAA");
    let out = verify(document(KEY_SCHEMA, &a_point()), p);
    assert!(
        error(&out).starts_with("nras_intermediate_der_b64:"),
        "{out}"
    );
}

/// Op 5: the body a carrier POSTs to NRAS, over the document's GPU nonce
/// and its evidence verbatim; a document without GPU evidence has none.
#[test]
fn the_nras_request_is_nvidias_v4_body_over_the_documents_gpu_nonce() {
    let mut doc = document(KEY_SCHEMA, &a_point());
    let out = call(
        OP_NRAS_REQUEST,
        json!({ "document": doc.clone(), "client_nonce_hex": "07".repeat(32) }),
    );
    assert!(error(&out).contains("no GPU evidence"), "{out}");
    let list = json!([{ "evidence": "cmVwb3J0", "certificate": "Y2hhaW4=" }]);
    doc["gpu_evidence_list"] = list.clone();
    let out = call(
        OP_NRAS_REQUEST,
        json!({ "document": doc.clone(), "client_nonce_hex": "07".repeat(32) }),
    );
    assert_eq!(out["ok"], true, "{out}");
    use base64::Engine as _;
    let config = base64::engine::general_purpose::STANDARD
        .decode(doc["key_config_b64"].as_str().unwrap())
        .unwrap();
    let pk: [u8; 32] = [3]
        .into_iter()
        .chain(std::iter::repeat_n(0, 31))
        .collect::<Vec<u8>>()
        .try_into()
        .unwrap();
    let nonce: String = x0ne_sdk::e2e::gpu_nonce(&config, &[7; 32], &pk)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        out["request"],
        json!({ "nonce": nonce, "evidence_list": list, "arch": "HOPPER", "claims_version": "3.0" })
    );
}

/// A page loads a `.wasm` file: only a `cdylib` build gives one. Without it
/// `cargo build --target wasm32-unknown-unknown` leaves an rlib and nothing
/// a browser can instantiate.
#[test]
fn the_crate_is_built_as_a_cdylib_for_the_browser() {
    let manifest = include_str!("../Cargo.toml");
    let lib = manifest
        .split("[lib]")
        .nth(1)
        .and_then(|s| s.split("\n[").next())
        .expect("a [lib] table");
    assert!(
        lib.contains("crate-type") && lib.contains("\"cdylib\"") && lib.contains("\"rlib\""),
        "[lib] must declare crate-type = [\"cdylib\", \"rlib\"]: {lib}"
    );
}

/// 64 or 96 bytes of text whose third byte is inside a character: the
/// right byte length for a hex field, not hex. Slicing it as a str panics.
fn multibyte_hex(bytes: usize) -> String {
    let s = format!("a{}a", "é".repeat((bytes - 2) / 2));
    assert_eq!(s.len(), bytes);
    s
}

fn verify_input(nonce_hex: &str, mrtd_hex: &str, session_pubkey_hex: &str) -> Value {
    json!({
        "document": {
            "schema": x0ne_sdk::attest::KEY_SCHEMA,
            "key_config_b64": base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                x0ne_sdk::e2e::Gateway::from_seed(&[5; 32]).key_config(),
            ),
            "session_pubkey_hex": session_pubkey_hex,
            "session_config_b64": b64(&session_config()),
            "boot_nonce_hex": "09".repeat(32),
            "tls_spki_sha256_hex": "54".repeat(32),
            "quote_b64": "",
            "ccel_b64": "",
        },
        "collateral": empty_collateral(),
        "policy": with_reference(json!({
            "machine_type": MACHINE,
            "mrtd_hex": mrtd_hex, "rtmr1_hex": "00".repeat(48),
            "rtmr2_hex": "00".repeat(48), "rtmr3_hex": "00".repeat(48),
            "model_digest_hex": model(),
            "attest": "cpu+gpu",
        })),
        "client_nonce_hex": nonce_hex,
        "now": NOW,
    })
}

/// SECURITY: the key document is the carrier's until `verify_key` accepts
/// it, and the page writes the other fields. A hex field of the right byte
/// length holding a multibyte character is an error naming the field: a
/// panic is a trap in wasm32, the caller's instance gone, not a refusal.
#[test]
fn a_hex_field_with_a_multibyte_character_is_refused_not_a_panic() {
    let (h32, h48) = ("00".repeat(32), "00".repeat(48));
    for (field, input) in [
        (
            "client_nonce_hex",
            verify_input(&multibyte_hex(64), &h48, &h32),
        ),
        ("mrtd_hex", verify_input(&h32, &multibyte_hex(96), &h32)),
        (
            "session_pubkey_hex",
            verify_input(&h32, &h48, &multibyte_hex(64)),
        ),
    ] {
        let out = call(OP_VERIFY_KEY, input);
        assert_eq!(out["ok"], false, "{field}: {out}");
        assert!(
            out["error"].as_str().unwrap().contains(field),
            "{field}: {out}"
        );
    }
    for field in ["boot_nonce_hex", "tls_spki_sha256_hex"] {
        let mut input = verify_input(&h32, &h48, &a_point());
        input["document"][field] = json!(multibyte_hex(64));
        let out = call(OP_VERIFY_KEY, input);
        assert!(
            error(&out).starts_with(&format!("refused at key: {field}")),
            "{field}: {out}"
        );
    }
    let out = call(
        OP_SEAL_CHAT,
        json!({ "key": 1, "body": {}, "ikm_hex": multibyte_hex(64), "now": 0 }),
    );
    assert_eq!(out["ok"], false, "{out}");
    assert!(out["error"].as_str().unwrap().contains("ikm_hex"), "{out}");
}

/// `nras_ocsp` in the policy: `"transparent"` and `"required"` are read;
/// anything else is an error naming the field.
#[test]
fn the_nras_ocsp_policy_is_read_and_another_value_is_an_error() {
    let (h32, h48) = ("00".repeat(32), "00".repeat(48));
    for mode in ["transparent", "required"] {
        let mut input = verify_input(&h32, &h48, &a_point());
        input["policy"]["nras_ocsp"] = json!(mode);
        let out = call(OP_VERIFY_KEY, input);
        assert!(!error(&out).contains("nras_ocsp"), "{mode}: {out}");
    }
    let mut input = verify_input(&h32, &h48, &a_point());
    input["policy"]["nras_ocsp"] = json!("strict");
    let out = call(OP_VERIFY_KEY, input);
    assert!(error(&out).contains("nras_ocsp"), "{out}");
}
