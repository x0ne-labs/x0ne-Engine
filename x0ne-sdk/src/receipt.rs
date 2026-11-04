//! The receipt of an answer, checked on the caller's side.
//!
//! The TD signs each answer's receipt with its per-boot session key
//! (x0ne-infer `receipt.rs`): Ed25519 over [`RECEIPT_DOMAIN`] ‖ the exact
//! bytes it hands out as `x0ne.signed_payload_b64`. The caller checks, in
//! order, each refusal naming its check:
//!
//! 1. `signature`: those bytes are signed by the key the quote bound
//!    ([`crate::attest::VerifiedKey::session_pubkey`]), never by a key the
//!    answer names;
//! 2. `session key`: the signed payload names that same key;
//! 3. `commitment`: SHA-256([`OUTPUT_LABEL`] ‖ salt ‖ text) of the text shown
//!    is the signed output commitment;
//! 4. `model`: the signed model digest is the pinned one, when one is pinned.
//!
//! What it returns is read from the signed bytes, never from the receipt
//! object shown next to them.

use base64::Engine as _;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// The session key's domain for chat receipts (x0ne-infer `binding.rs`).
pub const RECEIPT_DOMAIN: &[u8] = b"x0ne-infer/receipt/v1\0";
/// The label of the output commitment (x0ne-infer `receipt.rs`).
pub const OUTPUT_LABEL: &[u8] = b"x0ne-infer/commit/output/v1\0";
/// The label of the prompt commitment. The caller cannot recompute it: the
/// TD commits to the prompt after its chat template.
pub const PROMPT_LABEL: &[u8] = b"x0ne-infer/commit/prompt/v1\0";

/// An answer whose receipt checked: the text, and the signed payload.
#[derive(Debug, Clone, PartialEq)]
pub struct Checked {
    /// `choices[0].message.content`, which the output commitment opens on.
    pub text: String,
    pub model_digest_hex: String,
    pub compute_device: String,
    /// Every signed field, as signed.
    pub signed: Value,
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn str_at<'a>(v: &'a Value, path: &[&str]) -> Result<&'a str, String> {
    let mut at = v;
    for p in path {
        at = at.get(*p).ok_or_else(|| format!("no {}", path.join(".")))?;
    }
    at.as_str()
        .ok_or_else(|| format!("{} is not a string", path.join(".")))
}

/// The check of one chat completion `answer` (the JSON the TD sent), under
/// `attested_key`, the session key the quote bound, for the model
/// `model_pin` names; `None` checks no model, which
/// [`crate::e2e::Pending::open_verified`] never passes.
pub fn check_answer(
    answer: &[u8],
    attested_key: &[u8; 32],
    model_pin: Option<&str>,
) -> Result<Checked, String> {
    let answer: Value = serde_json::from_slice(answer).map_err(|e| format!("answer: {e}"))?;
    let signed = signed_payload(&answer, attested_key)?;
    let choices = answer
        .get("choices")
        .and_then(Value::as_array)
        .ok_or("no choices")?;
    if choices.len() != 1 {
        return Err(format!("{} choices: one is committed", choices.len()));
    }
    let text = str_at(&choices[0], &["message", "content"])?.to_string();
    opened(text, &answer, signed, model_pin)
}

/// The check of a streamed answer: `text`, every delta as it came, and
/// `last`, the stream's final chunk, which carries the `x0ne` block. The
/// same checks, in the same order, as [`check_answer`].
pub fn check_text(
    text: &str,
    last: &Value,
    attested_key: &[u8; 32],
    model_pin: Option<&str>,
) -> Result<Checked, String> {
    let signed = signed_payload(last, attested_key)?;
    opened(text.to_string(), last, signed, model_pin)
}

/// Checks 1 and 2: the payload `holder.x0ne` carries, signed by the
/// attested key and naming it.
fn signed_payload(holder: &Value, attested_key: &[u8; 32]) -> Result<Value, String> {
    let payload = base64::engine::general_purpose::STANDARD
        .decode(str_at(holder, &["x0ne", "signed_payload_b64"])?)
        .map_err(|e| format!("signed_payload_b64: {e}"))?;
    let sig = crate::unhex::<64>(str_at(holder, &["x0ne", "receipt", "signature_hex"])?)
        .map_err(|e| format!("signature: {e}"))?;

    let key = ed25519_dalek::VerifyingKey::from_bytes(attested_key)
        .map_err(|e| format!("signature: attested key: {e}"))?;
    let mut message = Vec::with_capacity(RECEIPT_DOMAIN.len() + payload.len());
    message.extend_from_slice(RECEIPT_DOMAIN);
    message.extend_from_slice(&payload);
    key.verify_strict(&message, &ed25519_dalek::Signature::from_bytes(&sig))
        .map_err(|_| {
            "signature: not by the attested session key, under the receipt domain".to_string()
        })?;

    let signed: Value =
        serde_json::from_slice(&payload).map_err(|e| format!("signed payload: {e}"))?;
    if !signed.is_object() {
        return Err("signed payload is not an object".into());
    }
    if str_at(&signed, &["session_public_key_hex"])? != hex(attested_key) {
        return Err("session key: the signed payload names another key".into());
    }
    Ok(signed)
}

/// Checks 3 and 4: `text` opens the signed commitment with the salt
/// `holder.x0ne.opening` shows, for the pinned model.
fn opened(
    text: String,
    holder: &Value,
    signed: Value,
    model_pin: Option<&str>,
) -> Result<Checked, String> {
    let salt = crate::unhex::<32>(str_at(holder, &["x0ne", "opening", "salt_hex"])?)
        .map_err(|e| format!("commitment: salt: {e}"))?;
    let mut h = Sha256::new();
    h.update(OUTPUT_LABEL);
    h.update(salt);
    h.update(text.as_bytes());
    if hex(&h.finalize()) != str_at(&signed, &["output_commitment_hex"])? {
        return Err("commitment: the text shown is not the text the TD committed to".into());
    }

    let model_digest_hex = str_at(&signed, &["model_digest_hex"])?.to_string();
    if let Some(pin) = model_pin {
        if !model_digest_hex.eq_ignore_ascii_case(pin) {
            return Err(format!(
                "model: signed digest {model_digest_hex} is not the pinned {pin}"
            ));
        }
    }
    let compute_device = str_at(&signed, &["compute_device"])?.to_string();
    Ok(Checked {
        text,
        model_digest_hex,
        compute_device,
        signed,
    })
}

/// [`check_answer`], then the decoding `asked`: sampling, seed and schema
/// digest as signed, the text a document of it (a start under `length`):
/// [`crate::decoding::Asked::check_signed`]; and the context of the
/// conversation it names, if any: [`crate::decoding::Asked::check_context`].
pub fn check_answer_for(
    answer: &[u8],
    attested_key: &[u8; 32],
    model_pin: Option<&str>,
    asked: &crate::decoding::Asked,
) -> Result<Checked, String> {
    let checked = check_answer(answer, attested_key, model_pin)?;
    // finish_reason is not signed: `length` admits a begun document, never
    // another one, and the text is the committed one.
    let shown: Value = serde_json::from_slice(answer).map_err(|e| format!("answer: {e}"))?;
    let cut = shown["choices"][0]["finish_reason"] == "length";
    asked
        .check_signed(&checked.signed, &checked.text, cut)
        .map_err(|e| format!("decoding: {e}"))?;
    // The context a conversation's turn reused: signed, and only when asked.
    asked
        .check_context(&checked.signed, &shown["x0ne"])
        .map_err(|e| format!("conversation: {e}"))?;
    Ok(checked)
}
