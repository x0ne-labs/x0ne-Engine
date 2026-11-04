//! Receipts: the durable, client-verifiable record of where an inference ran.
//!
//! Built on [`x0ne_confidential::model::ExecutionReceipt`] (environment,
//! verdict, timings, action facts), extended with what an inference needs:
//! the model digest, token counts, COMMITMENTS to prompt and output — never
//! the text, and never a bare hash of it either: a bare SHA-256 of a short
//! or predictable prompt falls to a dictionary. The commitment is salted
//! per-request (`commit = SHA-256(label ‖ salt ‖ text)`, salt random,
//! returned to the CALLER only), so the client can prove what was said and
//! nobody else can test guesses against the receipt. The label says WHICH
//! text — see [`CommitmentKind`].
//!
//! Signed with the per-boot session key (`binding::SessionIdentity`), whose
//! public key the hardware attested (report_data) — so a receipt verifies
//! offline against the attestation the client already checked.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use x0ne_confidential::model::ExecutionReceipt;

use crate::binding::SessionIdentity;
use crate::cloud_config::hex_array;
use crate::gate::Attested;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceReceipt {
    pub execution: ExecutionReceipt,
    pub model_digest_hex: String,
    /// SHA-256(prompt label ‖ salt ‖ prompt), salt handed to the caller, not
    /// in the receipt.
    pub prompt_commitment_hex: String,
    /// SHA-256(output label ‖ salt ‖ output), same salt discipline.
    pub output_commitment_hex: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Where the arithmetic ran: `"cpu"` or `"cuda:N"`. The
    /// environment above says what the MACHINE proved; this says which of
    /// its processors did the work. `ConfidentialVmWithGpu { gpu_mode: On }`
    /// with `compute_device: "cpu"` is an honest and cheaper product — a
    /// TDX-confidential inference on an attested GPU machine — but a reader
    /// who cannot tell it from the other one is being misled by omission.
    pub compute_device: String,
    /// What this engine did NOT do with the text. Signed like the
    /// rest, so the statement is durable and attributable rather than a
    /// sentence on a marketing page. These are the properties that are
    /// impossible to claim without owning the generation loop.
    pub isolation: crate::backend::Isolation,
    /// On a turn of a conversation only: the context it reused, from an
    /// earlier turn of the same conversation in this TD (`crate::context`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<crate::context::SignedContext>,
    /// What the platform's last verdict said about its patch level when
    /// this answer was computed ([`crate::tcb_disclosure`], since
    /// 2026-10-28). Absent from older receipts, whose signed bytes are
    /// unchanged: absent is not serialized.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tcb: Option<crate::tcb_disclosure::TcbDisclosure>,
    /// The draws of a sampled answer: `temperature`, `top_p`, and the seed,
    /// the caller's or drawn by the TD (since 2026-10-30). Absent: greedy;
    /// a greedy receipt's signed bytes are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<crate::decoding::SignedSampling>,
    /// SHA-256 of the `response_format` schema that masked every token
    /// ([`crate::json_schema::canonical`]), since 2026-10-30. Absent:
    /// unconstrained.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema_sha256_hex: Option<String>,
    /// Hex of the session public key the hardware attested (report_data
    /// binding, `binding` module) — what the signature verifies under.
    pub session_public_key_hex: String,
    /// Hex ed25519 signature over the canonical JSON of every field above.
    pub signature_hex: String,
}

/// What the caller gets besides the receipt: the salt that opens the
/// commitments. Losing it only loses the ability to PROVE the text —
/// the receipt stays valid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentOpening {
    pub salt_hex: String,
}

/// Which of the two texts a commitment is about.
///
/// SECURITY (2026-09-18): both commitments share one salt, and until
/// this label existed they were the same function of (salt, text). So the two
/// fields were interchangeable: an opening that proved "this was the OUTPUT"
/// proved equally well "this was the PROMPT", and a receipt whose two fields
/// had been swapped still opened. The label makes each commitment say what it
/// is a commitment TO. NUL-terminated so no label is a prefix of another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitmentKind {
    Prompt,
    Output,
}

impl CommitmentKind {
    /// PROTOCOL: `console.html` spells the output label too — a test holds
    /// the two together.
    pub const fn label(self) -> &'static [u8] {
        match self {
            Self::Prompt => b"x0ne-infer/commit/prompt/v1\0",
            Self::Output => b"x0ne-infer/commit/output/v1\0",
        }
    }
}

fn commit(kind: CommitmentKind, salt: &[u8; 32], text: &str) -> String {
    let mut h = Sha256::new();
    h.update(kind.label());
    h.update(salt);
    h.update(text.as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// The bytes the signature covers: the receipt WITHOUT its signature field,
/// serialized canonically (serde_json preserves struct order).
///
/// PROTOCOL: a verifier in another language cannot be asked to reproduce
/// "whatever serde_json emitted" — key order, escaping and number
/// formatting are implementation details, and a signature scheme whose
/// message is not specifiable is not verifiable by anyone but us. So the
/// answer carries these exact bytes alongside the receipt
/// (`signed_payload_b64`, the JWS pattern): a client verifies the signature
/// over the bytes it was given, then checks that those bytes, parsed, say
/// what the receipt it is displaying says. See `verify_signed_payload`.
///
/// SECURITY: this returned `unwrap_or_default()` — a serialization
/// failure became the EMPTY message, signed without complaint, and every
/// receipt that failed the same way verified against the same signature.
/// A failure is now a failure: nothing is signed, and a verifier says no.
pub(crate) fn signable_bytes(r: &InferenceReceipt) -> Result<Vec<u8>, serde_json::Error> {
    let mut unsigned = r.clone();
    unsigned.signature_hex = String::new();
    serde_json::to_vec(&unsigned)
}

/// One inference, signed: the receipt, the salt that opens it, and the EXACT
/// bytes the signature covers — handed out as they were signed, never
/// re-serialized on the way to the client.
#[derive(Debug, Clone)]
pub struct Signed {
    pub receipt: InferenceReceipt,
    pub opening: CommitmentOpening,
    pub payload: Vec<u8>,
}

/// What a receipt signs about the decoding, besides the texts, and the
/// context a turn of a conversation reused.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecodingClaims {
    pub sampling: Option<crate::decoding::SignedSampling>,
    pub json_schema_sha256_hex: Option<String>,
    pub context: Option<crate::context::SignedContext>,
}

impl DecodingClaims {
    /// The claims of the decoding `params` asked for: the receipt says
    /// what the loop was handed, seed included.
    pub fn of(params: &crate::backend::GenerationParams) -> Self {
        Self {
            sampling: params.sampling.map(|s| crate::decoding::SignedSampling {
                temperature: s.temperature,
                top_p: s.top_p,
                seed: s.seed,
            }),
            json_schema_sha256_hex: params
                .json_schema
                .as_ref()
                .map(|s| s.sha256_hex().to_string()),
            context: None,
        }
    }
}

/// Build and SIGN the receipt for one inference under `proof`.
#[allow(clippy::too_many_arguments)]
pub fn build_signed(
    proof: &Attested,
    identity: &SessionIdentity,
    salt: [u8; 32],
    model_digest_hex: &str,
    prompt: &str,
    output: &str,
    tokens: (u32, u32),
    compute_device: &str,
    isolation: crate::backend::Isolation,
    started_at_unix: u64,
    finished_at_unix: u64,
) -> Result<Signed, serde_json::Error> {
    build_signed_disclosing(
        proof,
        identity,
        salt,
        model_digest_hex,
        prompt,
        output,
        tokens,
        compute_device,
        isolation,
        started_at_unix,
        finished_at_unix,
        None,
    )
}

/// [`build_signed`], the receipt carrying `tcb`: what the platform's last
/// verdict said about its patch level, signed with the rest.
#[allow(clippy::too_many_arguments)]
pub fn build_signed_disclosing(
    proof: &Attested,
    identity: &SessionIdentity,
    salt: [u8; 32],
    model_digest_hex: &str,
    prompt: &str,
    output: &str,
    tokens: (u32, u32),
    compute_device: &str,
    isolation: crate::backend::Isolation,
    started_at_unix: u64,
    finished_at_unix: u64,
    tcb: Option<crate::tcb_disclosure::TcbDisclosure>,
) -> Result<Signed, serde_json::Error> {
    build_signed_decoding(
        proof,
        identity,
        salt,
        model_digest_hex,
        prompt,
        output,
        tokens,
        compute_device,
        isolation,
        started_at_unix,
        finished_at_unix,
        tcb,
        DecodingClaims::default(),
    )
}

/// [`build_signed_disclosing`], the receipt carrying the decoding's
/// `claims`: the sampling and its seed, the schema's digest.
#[allow(clippy::too_many_arguments)]
pub fn build_signed_decoding(
    proof: &Attested,
    identity: &SessionIdentity,
    salt: [u8; 32],
    model_digest_hex: &str,
    prompt: &str,
    output: &str,
    tokens: (u32, u32),
    compute_device: &str,
    isolation: crate::backend::Isolation,
    started_at_unix: u64,
    finished_at_unix: u64,
    tcb: Option<crate::tcb_disclosure::TcbDisclosure>,
    claims: DecodingClaims,
) -> Result<Signed, serde_json::Error> {
    let mut receipt = InferenceReceipt {
        execution: ExecutionReceipt {
            workload: proof.identity().name.clone(),
            environment: proof.environment().clone(),
            verdict: Some(proof.verdict().clone()),
            started_at_unix,
            finished_at_unix,
            actions: vec![("Infer".into(), true)],
        },
        model_digest_hex: model_digest_hex.to_string(),
        prompt_commitment_hex: commit(CommitmentKind::Prompt, &salt, prompt),
        output_commitment_hex: commit(CommitmentKind::Output, &salt, output),
        prompt_tokens: tokens.0,
        completion_tokens: tokens.1,
        compute_device: compute_device.to_string(),
        isolation,
        context: claims.context,
        tcb,
        sampling: claims.sampling,
        json_schema_sha256_hex: claims.json_schema_sha256_hex,
        session_public_key_hex: identity
            .public_key()
            .as_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
        signature_hex: String::new(),
    };
    let payload = signable_bytes(&receipt)?;
    let sig = identity.sign_receipt(&payload);
    receipt.signature_hex = sig.to_bytes().iter().map(|b| format!("{b:02x}")).collect();
    Ok(Signed {
        receipt,
        opening: CommitmentOpening {
            salt_hex: salt.iter().map(|b| format!("{b:02x}")).collect(),
        },
        payload,
    })
}

/// Client-side verification: signature under the receipt's own key, and —
/// given the opening and the texts — the commitments. Checking that the KEY
/// is the attested one is `binding::verify_binding`, against the quote.
pub fn verify_signed(receipt: &InferenceReceipt) -> bool {
    let Ok(key_bytes) = hex_array::<32>(&receipt.session_public_key_hex) else {
        return false;
    };
    let Ok(sig_bytes) = hex_array::<64>(&receipt.signature_hex) else {
        return false;
    };
    let Ok(key) = ed25519_dalek::VerifyingKey::from_bytes(&key_bytes) else {
        return false;
    };
    let sig = ed25519_dalek::Signature::from_bytes(&sig_bytes);
    let Ok(payload) = signable_bytes(receipt) else {
        return false;
    };
    crate::binding::verify_receipt_signature(&key, &payload, &sig)
}

/// Open a commitment: does (salt, text) match what the receipt committed to?
/// Verify a signature over the EXACT bytes a client was handed, and confirm
/// those bytes are the receipt it is looking at.
///
/// This is what a third-party verifier runs. `payload` is what the answer
/// carried verbatim; `receipt` is the parsed object the caller is about to
/// trust. Both must agree, or the answer is showing one thing and proving
/// another — the substitution this function exists to catch.
pub fn verify_signed_payload(payload: &[u8], receipt: &InferenceReceipt) -> bool {
    let Ok(sig_bytes) = hex_array::<64>(&receipt.signature_hex) else {
        return false;
    };
    let Ok(key_bytes) = hex_array::<32>(&receipt.session_public_key_hex) else {
        return false;
    };
    let Ok(key) = ed25519_dalek::VerifyingKey::from_bytes(&key_bytes) else {
        return false;
    };
    let sig = ed25519_dalek::Signature::from_bytes(&sig_bytes);
    if !crate::binding::verify_receipt_signature(&key, payload, &sig) {
        return false;
    }
    // The signed bytes must BE this receipt, minus its signature.
    signable_bytes(receipt).is_ok_and(|b| b == payload)
}

pub fn verify_commitment(
    kind: CommitmentKind,
    commitment_hex: &str,
    opening: &CommitmentOpening,
    text: &str,
) -> bool {
    let Ok(salt) = hex_array::<32>(&opening.salt_hex) else {
        return false;
    };
    commit(kind, &salt, text) == commitment_hex
}

#[cfg(test)]
mod tests {

    /// A third-party verifier checks the signature over the bytes it was
    /// GIVEN, then that those bytes are the receipt it is displaying.
    ///
    /// SECURITY: this is what a browser console runs. A failure means either
    /// the two halves disagree (a client could verify a signature over one
    /// document while showing another — the substitution the payload exists
    /// to prevent) or the payload no longer matches what we sign.
    #[test]
    fn the_signed_payload_is_the_receipt_and_any_substitution_is_caught() {
        let id = SessionIdentity::generate([11u8; 32], [13u8; 32]);
        let Signed {
            receipt, payload, ..
        } = build_signed(
            &proof(),
            &id,
            [5u8; 32],
            "abcd",
            "a prompt",
            "an output",
            (2, 2),
            "cpu",
            crate::backend::Isolation::NONE_OF_IT,
            1_000,
            1_001,
        )
        .unwrap();
        // The bytes handed out ARE the bytes signed, not a second serialization.
        assert_eq!(payload, signable_bytes(&receipt).unwrap());
        assert!(verify_signed_payload(&payload, &receipt));

        // A payload that verifies under the signature but is NOT this
        // receipt cannot exist — but a client handed a DIFFERENT receipt
        // with the same payload must refuse.
        let mut swapped = receipt.clone();
        swapped.model_digest_hex = "0".repeat(64);
        assert!(
            !verify_signed_payload(&payload, &swapped),
            "the payload must be checked against the receipt, not just the signature"
        );

        // And a tampered payload fails the signature outright.
        let mut tampered = payload.clone();
        let last = tampered.len() - 2;
        tampered[last] ^= 0x20;
        assert!(!verify_signed_payload(&tampered, &receipt));
    }
    /// 2026-10-28: a receipt signed before the TCB disclosure existed (the
    /// build of that morning, tests/VENDORED.md) still parses, still verifies
    /// over its bytes, and has no `tcb`; one signed with a disclosure
    /// carries it under the signature.
    #[test]
    fn a_receipt_from_before_the_tcb_disclosure_still_verifies() {
        let payload =
            include_bytes!("../tests/receipts/receipt_before_tcb_disclosure.payload.json");
        let json = include_str!("../tests/receipts/receipt_before_tcb_disclosure.receipt.json");
        let old: InferenceReceipt = serde_json::from_str(json).unwrap();
        assert_eq!(old.tcb, None);
        assert!(verify_signed(&old));
        assert!(verify_signed_payload(payload, &old));
        assert!(!String::from_utf8_lossy(payload).contains("\"tcb\""));

        let id = SessionIdentity::generate([11u8; 32], [13u8; 32]);
        let tcb = crate::tcb_disclosure::TcbDisclosure {
            schema: crate::tcb_disclosure::SCHEMA.into(),
            policy: "transparent".into(),
            as_of_unix: 1_000,
            intel: None,
            amd: None,
            line: "Platform TCB 2026-02-11, OutOfDate under Intel's TCB evaluation 22".into(),
        };
        let signed = build_signed_disclosing(
            &proof(),
            &id,
            [5u8; 32],
            "abcd",
            "a prompt",
            "an output",
            (2, 2),
            "cpu",
            crate::backend::Isolation::NONE_OF_IT,
            1_000,
            1_001,
            Some(tcb.clone()),
        )
        .unwrap();
        assert_eq!(signed.receipt.tcb.as_ref(), Some(&tcb));
        assert!(verify_signed_payload(&signed.payload, &signed.receipt));
        // The disclosure is under the signature: another one breaks it.
        let mut swapped = signed.receipt.clone();
        swapped.tcb.as_mut().unwrap().line = "Platform TCB 2026-08-12, UpToDate".into();
        assert!(!verify_signed_payload(&signed.payload, &swapped));
        assert!(!verify_signed(&swapped));
        let mut dropped = signed.receipt;
        dropped.tcb = None;
        assert!(!verify_signed(&dropped));
    }

    use super::*;
    use crate::gate::Gate;
    use x0ne_confidential::model::{
        AttestationEvidence, EvidenceFormat, ExecutionEnvironment, GpuCcMode, TeeTechnology,
        Verdict, VerifierStrength, WorkloadIdentity,
    };

    fn proof() -> Attested {
        let env = ExecutionEnvironment::ConfidentialVmWithGpu {
            cpu: TeeTechnology::IntelTdx,
            gpu_mode: GpuCcMode::On,
        };
        Gate::attest(
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

    /// A receipt: signature verifies, tampering any field breaks it, the
    /// commitments open with the salt and text — and the raw text appears
    /// nowhere in the JSON (zero retention, dictionary-resistant).
    #[test]
    fn a_signed_receipt_verifies_commits_and_carries_no_text() {
        let id = SessionIdentity::generate([7u8; 32], [9u8; 32]);
        let Signed {
            receipt, opening, ..
        } = build_signed(
            &proof(),
            &id,
            [3u8; 32],
            "abcd",
            "secret prompt",
            "secret output",
            (2, 2),
            "cpu",
            crate::backend::Isolation::NONE_OF_IT,
            1_000,
            1_001,
        )
        .unwrap();
        assert!(verify_signed(&receipt));
        let json = serde_json::to_string(&receipt).unwrap();
        assert!(!json.contains("secret"));
        assert!(verify_commitment(
            CommitmentKind::Prompt,
            &receipt.prompt_commitment_hex,
            &opening,
            "secret prompt"
        ));
        assert!(!verify_commitment(
            CommitmentKind::Prompt,
            &receipt.prompt_commitment_hex,
            &opening,
            "guessed prompt"
        ));
        let mut tampered = receipt.clone();
        tampered.model_digest_hex = "beef".into();
        assert!(!verify_signed(&tampered), "tampering breaks the signature");
        // Same prompt, different salt: different commitment — no dictionary.
        let Signed { receipt: r2, .. } = build_signed(
            &proof(),
            &id,
            [4u8; 32],
            "abcd",
            "secret prompt",
            "secret output",
            (2, 2),
            "cpu",
            crate::backend::Isolation::NONE_OF_IT,
            1_000,
            1_001,
        )
        .unwrap();
        assert_ne!(r2.prompt_commitment_hex, receipt.prompt_commitment_hex);
    }

    /// SECURITY: the receipt says which processor did the work, and
    /// the signature covers it. A failure here means the field is decorative
    /// — an operator could serve CPU inference on an attested GPU machine
    /// and hand out a receipt saying `cuda:0`, or a proxy could rewrite the
    /// word in transit, and the client's own verification would still pass.
    #[test]
    fn the_compute_device_is_in_the_receipt_and_under_the_signature() {
        let id = SessionIdentity::generate([21u8; 32], [23u8; 32]);
        let Signed { receipt, .. } = build_signed(
            &proof(),
            &id,
            [7u8; 32],
            "abcd",
            "a prompt",
            "an output",
            (2, 2),
            "cuda:0",
            crate::backend::Isolation::NONE_OF_IT,
            1_000,
            1_001,
        )
        .unwrap();
        assert_eq!(receipt.compute_device, "cuda:0");
        assert!(verify_signed(&receipt));
        // The lie a receipt must not be able to tell.
        let mut lying = receipt.clone();
        lying.compute_device = "cpu".into();
        assert!(
            !verify_signed(&lying),
            "rewriting the device must break the signature"
        );
    }

    /// SECURITY: the negative attestations are under the signature.
    /// Their entire value is that they are durable and attributable — a
    /// claim a client can hold us to three weeks later. A field a proxy can
    /// rewrite in flight is a sentence on a marketing page, not an
    /// attestation.
    #[test]
    fn the_isolation_claims_are_in_the_receipt_and_under_the_signature() {
        let id = SessionIdentity::generate([31u8; 32], [33u8; 32]);
        let Signed { receipt, .. } = build_signed(
            &proof(),
            &id,
            [7u8; 32],
            "abcd",
            "a prompt",
            "an output",
            (2, 2),
            "cpu",
            crate::backend::Isolation::NONE_OF_IT,
            1_000,
            1_001,
        )
        .unwrap();
        assert!(!receipt.isolation.kv_cache_shared_between_requests);
        assert!(!receipt.isolation.prefix_cache_shared_between_sessions);
        assert!(!receipt.isolation.requests_batched_across_clients);
        assert!(!receipt.isolation.prompt_text_logged);
        assert!(verify_signed(&receipt));
        let mut lying = receipt.clone();
        lying.isolation.prefix_cache_shared_between_sessions = true;
        assert!(
            !verify_signed(&lying),
            "rewriting an isolation claim must break the signature"
        );
    }

    /// SECURITY: a failure means the two commitments are interchangeable
    /// again — an opening that proves "this was the OUTPUT" also proves "this
    /// was the PROMPT", and a receipt with its two fields swapped still opens.
    #[test]
    fn an_opening_for_the_output_does_not_open_the_prompt() {
        let id = SessionIdentity::generate([41u8; 32], [43u8; 32]);
        let Signed {
            receipt, opening, ..
        } = build_signed(
            &proof(),
            &id,
            [6u8; 32],
            "abcd",
            "the same words",
            "the same words",
            (3, 3),
            "cpu",
            crate::backend::Isolation::NONE_OF_IT,
            1_000,
            1_001,
        )
        .unwrap();
        assert_ne!(
            receipt.prompt_commitment_hex, receipt.output_commitment_hex,
            "same salt, same text, different field: the label is what differs"
        );
        assert!(verify_commitment(
            CommitmentKind::Output,
            &receipt.output_commitment_hex,
            &opening,
            "the same words"
        ));
        assert!(!verify_commitment(
            CommitmentKind::Prompt,
            &receipt.output_commitment_hex,
            &opening,
            "the same words"
        ));
    }

    /// PROTOCOL: the browser console recomputes the commitment and the signed
    /// message by itself, in JavaScript. If the bytes drift apart, every
    /// honest answer shows a red badge — or, worse, someone "fixes" the page
    /// by loosening the check. The page must spell exactly what this crate
    /// signs.
    #[test]
    fn the_browser_console_spells_the_same_domain_and_label() {
        let page = include_str!("console.html");
        for bytes in [
            crate::binding::RECEIPT_DOMAIN,
            CommitmentKind::Output.label(),
        ] {
            let text = std::str::from_utf8(&bytes[..bytes.len() - 1]).unwrap();
            assert_eq!(bytes[bytes.len() - 1], 0, "NUL-terminated");
            assert!(
                page.contains(&format!("\"{text}\\0\"")),
                "console.html must contain the literal \"{text}\\0\""
            );
        }
    }

    /// The page verifies the signature under the key the RECEIPT names;
    /// it may only call that key hardware-attested if it compares it to
    /// `/attestation`'s.
    #[test]
    fn the_console_does_not_call_an_unchecked_key_attested() {
        let page = include_str!("console.html");
        if page.contains("the hardware attested") {
            assert!(
                page.contains("attestation.session_public_key_hex"),
                "console.html calls the receipt key attested and never compares it"
            );
        }
    }

    /// SECURITY: the claim `prompt_text_logged: false` is checked
    /// against the crate, not trusted. Every receipt this engine signs
    /// asserts that no fragment of prompt or completion reaches a log; the
    /// cheapest way for that to become false is someone adding a `println!`
    /// while debugging and leaving it. A failure here means the receipt is
    /// making a statement the source no longer supports — fix the source, or
    /// stop making the claim.
    #[test]
    fn no_module_of_this_crate_logs_prompt_or_completion_text() {
        // Every source file of the crate, read at test time so a new module
        // cannot escape by not being listed.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut offenders: Vec<String> = Vec::new();
        let mut stack = vec![dir];
        while let Some(d) = stack.pop() {
            for entry in std::fs::read_dir(&d).expect("src is readable") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    // `src/bin` holds the CLIENTS: they run on the user's own
                    // machine and print the answer to the person who asked for
                    // it. The claim is about what the ENCLAVE does with the
                    // text, and the enclave is the library.
                    if path.file_name().and_then(|n| n.to_str()) != Some("bin") {
                        stack.push(path);
                    }
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                let text = std::fs::read_to_string(&path).expect("source is UTF-8");
                for (n, line) in text.lines().enumerate() {
                    let l = line.trim_start();
                    // Comments and doc-comments talk ABOUT prompts; code logs
                    // them. Only the latter is the defect.
                    if l.starts_with("//") {
                        continue;
                    }
                    let logs = [
                        "println!",
                        "eprintln!",
                        "info!",
                        "warn!",
                        "debug!",
                        "trace!",
                        "error!",
                        "dbg!",
                    ]
                    .iter()
                    .any(|m| l.contains(m));
                    if logs
                        && (l.contains("prompt") || l.contains("completion") || l.contains("text"))
                    {
                        offenders.push(format!("{}:{}: {}", path.display(), n + 1, line.trim()));
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "the receipt claims prompt_text_logged: false, but:\n{}",
            offenders.join("\n")
        );
    }

    /// T22.
    /// The streaming path holds the answer's text in every frame it builds:
    /// no log macro there at all, whatever the line names. A failure is one
    /// `dbg!(frame)` away from the text in the journal.
    #[test]
    fn streaming_code_has_no_log_macro() {
        const MACROS: [&str; 12] = [
            "println!",
            "eprintln!",
            "print!",
            "eprint!",
            "info!",
            "warn!",
            "debug!",
            "trace!",
            "error!",
            "dbg!",
            "log::",
            "tracing::",
        ];
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let candle = std::fs::read_to_string(src.join("candle_backend.rs")).expect("readable");
        let streaming = candle
            .split("fn generate_streaming(")
            .nth(1)
            .expect("generate_streaming exists")
            .split("\n    fn ")
            .next()
            .unwrap_or_default()
            .to_string();
        let mut checked = vec![("generate_streaming".to_string(), streaming)];
        // The TD's frames, the relay's reader of them and the relay's stream.
        for file in ["stream.rs", "sse.rs", "local_api.rs"] {
            match std::fs::read_to_string(src.join(file)) {
                Ok(text) => checked.push((file.to_string(), text)),
                Err(e) => panic!("{file}: {e}"),
            }
        }
        let mut offenders = Vec::new();
        for (name, text) in &checked {
            for (n, line) in text.lines().enumerate() {
                let l = line.trim_start();
                if !l.starts_with("//") && MACROS.iter().any(|m| l.contains(m)) {
                    offenders.push(format!("{name}:{}: {l}", n + 1));
                }
            }
        }
        assert!(offenders.is_empty(), "{}", offenders.join("\n"));
    }

    /// T8.
    /// SECURITY: the claim `kv_cache_shared_between_requests: false`
    /// rests on the candle loop clearing its cache on EVERY exit path. It did
    /// not until 2026-08-22 — the clear sat after the loop, and each `?`
    /// inside jumped over it, so a failed generation left one caller's keys
    /// and values in place for the next. A failure here means the clear has
    /// drifted back behind a fallible expression and the receipt is again
    /// asserting something the error path contradicts.
    #[test]
    fn the_kv_cache_is_cleared_on_every_exit_path_of_generate() {
        let src = include_str!("candle_backend.rs");
        let body_of = |name: &str| {
            src.split(&format!("fn {name}("))
                .nth(1)
                .unwrap_or_else(|| panic!("{name} exists"))
                .split("\n    fn ")
                .next()
                .unwrap_or_else(|| panic!("{name} has an end"))
        };
        // The loop is in generate_turn; the other three only delegate to it.
        for delegate in ["generate", "generate_with", "generate_streaming"] {
            assert!(
                !body_of(delegate).contains("forward("),
                "a second generation loop in {delegate} would bypass the clear below"
            );
        }
        let body = body_of("generate_turn");
        // Kept for a conversation's next turn, or zeroed (`Slot::settle`).
        let clear = body.find("model.settle(").expect("the cache is settled");
        // Nothing fallible may sit between the clear and the end of the
        // function, and nothing fallible may be able to return before it —
        // which is what the closure buys: its `?` returns into `run`, a
        // value, not out of `generate`. Since 2026-10-31 the call is under
        // `catch_unwind`: a panic inside the loop evicts the cache before it
        // goes on, so the panic path keeps nothing either.
        let call = body[..clear]
            .find("catch_unwind(std::panic::AssertUnwindSafe(run))")
            .expect(
                "the clear must follow a call that CAPTURES the failure instead \
                 of propagating it; if the closure is gone, every `?` skips the \
                 clear, and if catch_unwind is gone, a panic skips it",
            );
        let guarded = &body[call..clear];
        let evict = guarded
            .find("model.evict()")
            .expect("a panic inside the loop evicts the cache before it goes on");
        let resume = guarded
            .find("resume_unwind(")
            .expect("the panic goes on, after the eviction");
        assert!(
            evict < resume,
            "the eviction must come BEFORE the panic resumes, or it never runs"
        );
        assert!(
            body[clear..].contains("let generated = generated?;"),
            "the captured failure must be re-raised AFTER the clear, not before"
        );
    }

    /// SECURITY: a receipt is untrusted input. A key, signature or salt that
    /// is not hex, multibyte UTF-8 at the length check's exact byte count
    /// included, is refused. Before 2026-10-23 the decoder sliced the str by
    /// two bytes and panicked on "€" (end byte index 2 is not a char boundary).
    #[test]
    fn non_ascii_hex_in_a_receipt_is_refused_and_never_panics() {
        let id = SessionIdentity::generate([3u8; 32], [4u8; 32]);
        let Signed {
            receipt, payload, ..
        } = build_signed(
            &proof(),
            &id,
            [6u8; 32],
            "abcd",
            "a prompt",
            "an output",
            (1, 1),
            "cpu",
            crate::backend::Isolation::NONE_OF_IT,
            1_000,
            1_001,
        )
        .unwrap();
        assert!(verify_signed(&receipt));
        let euro64 = format!("€{}", "0".repeat(61)); // 64 bytes, 62 chars
        let euro128 = format!("€{}", "0".repeat(125)); // 128 bytes
        for (key, sig) in [
            (euro64.as_str(), receipt.signature_hex.as_str()),
            (receipt.session_public_key_hex.as_str(), euro128.as_str()),
            ("0".repeat(63).as_str(), receipt.signature_hex.as_str()), // odd length
            ("zz".repeat(32).as_str(), receipt.signature_hex.as_str()), // not hex
        ] {
            let mut bad = receipt.clone();
            bad.session_public_key_hex = key.to_string();
            bad.signature_hex = sig.to_string();
            assert!(!verify_signed(&bad), "{key} / {sig}");
            assert!(!verify_signed_payload(&payload, &bad), "{key} / {sig}");
        }
        let opening = CommitmentOpening { salt_hex: euro64 };
        assert!(!verify_commitment(
            CommitmentKind::Output,
            &receipt.output_commitment_hex,
            &opening,
            "an output"
        ));
    }
}
