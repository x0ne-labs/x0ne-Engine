//! The receipt check on the caller's side: an answer counts only if the key
//! the quote bound signed it, under the receipt domain, and its output
//! commitment opens on the text shown. Answers built here by hand; the TD's
//! own receipts pass the same check in x0ne-infer (`e2e_sdk_tests.rs`).

use base64::Engine as _;
use ed25519_dalek::{Signer, SigningKey};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use x0ne_sdk::receipt::{self, RECEIPT_DOMAIN};

const DIGEST: &str = "4242424242424242424242424242424242424242424242424242424242424242";

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn commit(label: &[u8], salt: &[u8; 32], text: &str) -> String {
    let mut h = Sha256::new();
    h.update(label);
    h.update(salt);
    h.update(text.as_bytes());
    hex(&h.finalize())
}

struct Made {
    key: [u8; 32],
    answer: Value,
}

/// An answer as the TD shapes it, `edit` applied to the signed payload
/// before signing, the payload signed by `signer` over `domain ‖ payload`.
fn answer(text: &str, signer: &SigningKey, domain: &[u8], edit: impl FnOnce(&mut Value)) -> Made {
    let attested = SigningKey::from_bytes(&[5; 32]);
    let salt = [9u8; 32];
    let mut payload = json!({
        "model_digest_hex": DIGEST,
        "prompt_commitment_hex": commit(b"x0ne-infer/commit/prompt/v1\0", &salt, "q"),
        "output_commitment_hex": commit(receipt::OUTPUT_LABEL, &salt, text),
        "compute_device": "cpu",
        "session_public_key_hex": hex(attested.verifying_key().as_bytes()),
    });
    edit(&mut payload);
    let payload = serde_json::to_vec(&payload).unwrap();
    let sig = signer.sign(&[domain, &payload].concat());
    let mut receipt = serde_json::from_slice::<Value>(&payload).unwrap();
    receipt["signature_hex"] = json!(hex(&sig.to_bytes()));
    Made {
        key: attested.verifying_key().to_bytes(),
        answer: json!({
            "object": "chat.completion",
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": text } }],
            "x0ne": {
                "receipt": receipt,
                "opening": { "salt_hex": hex(&salt) },
                "signed_payload_b64": base64::engine::general_purpose::STANDARD.encode(&payload),
            },
        }),
    }
}

fn attested() -> SigningKey {
    SigningKey::from_bytes(&[5; 32])
}

fn check(m: &Made, pin: Option<&str>) -> Result<receipt::Checked, String> {
    receipt::check_answer(&serde_json::to_vec(&m.answer).unwrap(), &m.key, pin)
}

#[test]
fn an_answer_signed_by_the_attested_key_passes_and_shows_its_signed_fields() {
    let m = answer("4", &attested(), RECEIPT_DOMAIN, |_| {});
    let c = check(&m, Some(DIGEST)).unwrap();
    assert_eq!(c.text, "4");
    assert_eq!(c.model_digest_hex, DIGEST);
    assert_eq!(c.compute_device, "cpu");
    assert_eq!(c.signed["session_public_key_hex"], hex(&m.key));
}

#[test]
fn a_signature_by_any_other_key_is_refused() {
    let m = answer(
        "4",
        &SigningKey::from_bytes(&[6; 32]),
        RECEIPT_DOMAIN,
        |_| {},
    );
    assert!(check(&m, None).unwrap_err().contains("signature"));
}

#[test]
fn a_signature_without_the_receipt_domain_is_refused() {
    let m = answer("4", &attested(), b"", |_| {});
    assert!(check(&m, None).unwrap_err().contains("signature"));
    let m = answer(
        "4",
        &attested(),
        b"x0ne-infer/decision-receipt/v1\0",
        |_| {},
    );
    assert!(check(&m, None).unwrap_err().contains("signature"));
}

/// Signed by the attested key, but naming another: what the TD signed is
/// not what it claims to be.
#[test]
fn a_payload_naming_another_session_key_is_refused() {
    let m = answer("4", &attested(), RECEIPT_DOMAIN, |p| {
        p["session_public_key_hex"] = json!("11".repeat(32));
    });
    assert!(check(&m, None).unwrap_err().contains("session key"));
}

/// SECURITY: the carrier cannot edit the answer (it is sealed), but the
/// check does not rely on that: a text other than the committed one fails.
#[test]
fn an_answer_text_other_than_the_committed_one_is_refused() {
    let mut m = answer("4", &attested(), RECEIPT_DOMAIN, |_| {});
    m.answer["choices"][0]["message"]["content"] = json!("5");
    assert!(check(&m, None).unwrap_err().contains("commitment"));
    let mut m = answer("4", &attested(), RECEIPT_DOMAIN, |_| {});
    m.answer["x0ne"]["opening"]["salt_hex"] = json!("00".repeat(32));
    assert!(check(&m, None).unwrap_err().contains("commitment"));
}

#[test]
fn a_model_digest_other_than_the_pinned_one_is_refused() {
    let m = answer("4", &attested(), RECEIPT_DOMAIN, |_| {});
    assert!(check(&m, Some(&"00".repeat(32)))
        .unwrap_err()
        .contains("model"));
}

/// The shown receipt is not what counts: the signed bytes are. A receipt
/// object edited after signing changes nothing the check returns.
#[test]
fn the_fields_returned_are_the_signed_ones_not_the_shown_ones() {
    let mut m = answer("4", &attested(), RECEIPT_DOMAIN, |_| {});
    m.answer["x0ne"]["receipt"]["compute_device"] = json!("gpu");
    let c = check(&m, None).unwrap();
    assert_eq!(c.compute_device, "cpu");
}

#[test]
fn an_answer_without_its_receipt_parts_is_refused() {
    for path in ["signed_payload_b64", "receipt", "opening"] {
        let mut m = answer("4", &attested(), RECEIPT_DOMAIN, |_| {});
        m.answer["x0ne"].as_object_mut().unwrap().remove(path);
        assert!(check(&m, None).is_err(), "{path}");
    }
    let mut m = answer("4", &attested(), RECEIPT_DOMAIN, |_| {});
    m.answer.as_object_mut().unwrap().remove("x0ne");
    assert!(check(&m, None).is_err());
    assert!(receipt::check_answer(b"not json", &[0; 32], None).is_err());
}

/// A request body for `extra`'s fields, read as the TD reads it.
fn asked(extra: Value) -> x0ne_sdk::decoding::Asked {
    let mut body = json!({"messages": [{"role": "user", "content": "q"}]});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    x0ne_sdk::decoding::Asked::from_body(&body).unwrap()
}

fn check_for(m: &Made, asked: &x0ne_sdk::decoding::Asked) -> Result<receipt::Checked, String> {
    receipt::check_answer_for(
        &serde_json::to_vec(&m.answer).unwrap(),
        &m.key,
        Some(DIGEST),
        asked,
    )
}

/// The receipt must sign the decoding asked: seed, parameters, schema digest,
/// the text a document of it (a start under `length`). A failure: a client
/// takes a receipt of another decoding, or of none.
#[test]
fn the_receipt_must_sign_the_decoding_the_request_asked() {
    let schema =
        json!({"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]});
    let digest = x0ne_sdk::json_schema::schema_sha256_hex(&schema);
    let request = json!({"temperature": 0.5, "seed": 7, "response_format": {"type": "json_schema",
        "json_schema": {"name": "a", "schema": schema}}});
    let want = asked(request.clone());
    let honest = |text: &str| {
        let digest = digest.clone();
        answer(text, &attested(), RECEIPT_DOMAIN, move |p| {
            p["sampling"] = json!({"temperature": 0.5, "top_p": 1.0, "seed": 7});
            p["json_schema_sha256_hex"] = json!(digest);
        })
    };
    let c = check_for(&honest(r#"{"ok":false}"#), &want).unwrap();
    assert_eq!(c.signed["sampling"]["seed"], 7);
    assert_eq!(c.signed["json_schema_sha256_hex"], json!(digest));

    let refused = |m: Made, word: &str| {
        let e = check_for(&m, &want).unwrap_err();
        assert!(
            e.starts_with("decoding: ") && e.contains(word),
            "{word}: {e}"
        );
    };
    refused(honest(r#"{"ok":0}"#), "not a document");
    refused(honest(r#"{"ok":fa"#), "stops before");
    let mut cut = honest(r#"{"ok":fa"#);
    cut.answer["choices"][0]["finish_reason"] = json!("length");
    check_for(&cut, &want).unwrap();
    refused(
        answer(r#"{"ok":true}"#, &attested(), RECEIPT_DOMAIN, |p| {
            p["sampling"] = json!({"temperature": 0.5, "top_p": 1.0, "seed": 8});
            p["json_schema_sha256_hex"] = json!(digest);
        }),
        "another seed",
    );
    refused(
        answer(r#"{"ok":true}"#, &attested(), RECEIPT_DOMAIN, |p| {
            p["json_schema_sha256_hex"] = json!(digest);
        }),
        "greedy",
    );
    refused(
        answer(r#"{"ok":true}"#, &attested(), RECEIPT_DOMAIN, |p| {
            p["sampling"] = json!({"temperature": 0.5, "top_p": 1.0, "seed": 7});
        }),
        "signs none",
    );
    // A greedy, free request: a receipt that signs a schema or draws is refused.
    let free = asked(json!({}));
    check_for(&answer("4", &attested(), RECEIPT_DOMAIN, |_| {}), &free).unwrap();
    let e = check_for(&honest(r#"{"ok":true}"#), &free).unwrap_err();
    assert!(e.contains("sampling"), "{e}");
}

/// A turn of a conversation counts only if its receipt signs the context it
/// reused, the cache flag, and a commitment that opens to the handles; an
/// answer with no conversation asked must sign neither. A failure: a reuse
/// the caller's verifier never sees.
#[test]
fn an_answer_to_a_conversation_must_sign_its_context() {
    use x0ne_sdk::context::{Handle, SignedContext};
    let issued = Handle::from_bytes([4; 32]);
    let made = |context: Option<&SignedContext>, kv: bool, handle: bool| {
        let mut m = answer("4", &attested(), RECEIPT_DOMAIN, |p| {
            p["prompt_tokens"] = json!(3);
            p["isolation"] = json!({"kv_cache_shared_between_requests": kv,
                "prefix_cache_shared_between_sessions": false,
                "requests_batched_across_clients": false, "prompt_text_logged": false});
            if let Some(c) = context {
                p["context"] = serde_json::to_value(c).unwrap();
            }
        });
        if handle {
            m.answer["x0ne"]["conversation"] = json!({"handle": issued.hex()});
        }
        m
    };
    let opened = SignedContext::commit(&[9; 32], 1, &[], None, &issued);
    let new = asked(json!({"x0ne": {"conversation": "new"}}));
    check_for(&made(Some(&opened), true, true), &new).unwrap();
    for (m, word) in [
        (made(None, true, true), "signs no context"),
        (made(Some(&opened), false, true), "false"),
        (made(Some(&opened), true, false), "no handle"),
    ] {
        let e = check_for(&m, &new).unwrap_err();
        assert!(
            e.starts_with("conversation: ") && e.contains(word),
            "{word}: {e}"
        );
    }
    let sent = Handle::from_bytes([6; 32]);
    let next = asked(json!({"x0ne": {"conversation": sent.hex()}}));
    let e = check_for(&made(Some(&opened), true, true), &next).unwrap_err();
    assert!(e.contains("does not open"), "{e}");
    let continued = SignedContext::commit(&[9; 32], 2, &[1, 2], Some(&sent), &issued);
    check_for(&made(Some(&continued), true, true), &next).unwrap();

    let none = asked(json!({}));
    check_for(&made(None, false, false), &none).unwrap();
    let e = check_for(&made(Some(&opened), true, true), &none).unwrap_err();
    assert!(e.contains("no conversation"), "{e}");
}
