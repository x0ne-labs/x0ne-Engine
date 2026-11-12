//! The compute backend — pluggable, behind a trait. The engine's "motto"
//! (owner, 2026-08-19): *every recent inference optimization* (continuous
//! batching, paged KV-cache, speculative decoding, quantization, prefix
//! caching) **and** *every confidentiality optimization* (encrypted weights
//! streamed straight into protected GPU memory, KV-cache never leaving it,
//! zero-retention buffers, no host-side plaintext longer than the request).
//! The trait is the seam where both families plug in. Since 2026-11-04 x0ne
//! Core serves (`core_backend`); candle (`candle_backend`, feature
//! `candle-reference`) is the tests' reference on the CPU.

use std::ops::ControlFlow;

use crate::keys::ReleasedKey;
use crate::prompt::Prompt;

/// A loaded model, ready to generate. Constructed only by a backend's `load`,
/// which takes the released key — so weights are decrypted only downstream of
/// the gate (non-negotiable 2).
/// The prompt does not fit the model's context window.
///
/// Its own type so a server can answer 413 with a sentence, instead of the 500
/// and the tensor-shape message a client got before 2026-09-19 — after the GPU
/// had run the whole prefill for nothing (measured on an RTX 5090: a 42 021
/// token prompt against a 40 960 context, 3.9 s, then `narrow invalid args`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("the prompt is {prompt_tokens} tokens and this model's context is {context_tokens}")]
pub struct PromptTooLong {
    pub prompt_tokens: usize,
    pub context_tokens: usize,
}

/// How many tokens may still be generated: `max_tokens`, cut to what is left
/// of the context. Judged BEFORE the model runs. A prompt that fills the
/// context leaves nothing to generate into, so it is refused too.
pub fn generation_budget(
    prompt_tokens: usize,
    max_tokens: usize,
    context_tokens: usize,
) -> Result<usize, PromptTooLong> {
    if prompt_tokens >= context_tokens {
        return Err(PromptTooLong {
            prompt_tokens,
            context_tokens,
        });
    }
    Ok(max_tokens.min(context_tokens - prompt_tokens))
}

/// How tokens are drawn when decoding is not greedy: OpenAI `temperature`,
/// `top_p` and `seed`, as the request gave them or the server drew the seed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
}

/// One request's decoding. `sampling: None` is greedy.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub sampling: Option<Sampling>,
    /// OpenAI `stop`: the answer ends before the first of these it contains.
    pub stop: Vec<String>,
    /// OpenAI `response_format: json_schema`: each token masked to keep the
    /// answer a document of this schema ([`crate::constrain`]).
    pub json_schema: Option<crate::json_schema::JsonSchema>,
}

impl GenerationParams {
    pub fn greedy(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            sampling: None,
            stop: Vec::new(),
            json_schema: None,
        }
    }
}

/// A completion, and whether a stop sequence ended it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// `text` ends where a stop sequence began, the sequence not in it, or
    /// where its schema's document closed: on the last token of the budget
    /// too, which the count of tokens alone reads as a cut.
    pub stopped: bool,
    /// The generation used its whole budget: `max_tokens`, or fewer when that
    /// is all the context window left. Compared with `max_tokens` alone, a cut
    /// by the window read as a model that stopped by itself.
    pub length: bool,
}

impl Completion {
    /// OpenAI's `finish_reason`: `"stop"` on a stop sequence, `"length"` on a
    /// spent budget, else as [`crate::server::finish_reason`] reads the ceiling.
    pub fn finish_reason(&self, max_tokens: usize) -> &'static str {
        if self.stopped {
            "stop"
        } else if self.length {
            "length"
        } else {
            crate::server::finish_reason(self.completion_tokens, max_tokens)
        }
    }
}

/// Sampling asked of a backend that only decodes greedily: a 400, not a
/// greedy answer under a temperature the caller set.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("temperature: this backend decodes greedily only; refused rather than ignored")]
pub struct SamplingUnsupported;

/// A schema asked of a backend that cannot mask its tokens, or of a
/// tokenizer whose tokens are not their text: a 400, never an answer that
/// was not held to it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("response_format: {0}; refused rather than ignored")]
pub struct JsonSchemaUnsupported(pub String);

/// A conversation asked of a backend that keeps none: a 400, not an answer
/// without the context block the caller's verifier then refuses.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("x0ne.conversation: this model keeps no conversation between requests; refused rather than ignored")]
pub struct ConversationUnsupported;

/// Messages in an order the model's own chat template raises on (a
/// conversation that starts with an assistant message, for one): a 400, not a
/// prompt the model was never trained on.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("messages: {0}; refused rather than rendered into a prompt the model was not trained on")]
pub struct ChatRefused(pub String);

/// `text` up to the earliest stop sequence it contains, and whether one was.
pub fn cut_at_stop(text: &str, stop: &[String]) -> (String, bool) {
    match stop
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
    {
        Some(at) => (text[..at].to_string(), true),
        None => (text.to_string(), false),
    }
}

/// What [`StopHoldback::push`] lets out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Held {
    /// No stop sequence can begin in this text any more. May be empty.
    Emit(String),
    /// The text before the first stop sequence met; the generation ends.
    Stop(String),
}

/// [`cut_at_stop`] for text that arrives in pieces: a tail that may still
/// become a stop sequence is held back, never emitted then retracted.
/// Searches bytes: a byte offset sliced as `str` panics off a char boundary.
#[derive(Debug, Clone)]
pub struct StopHoldback {
    stop: Vec<String>,
    acc: String,
    sent: usize,
    hold: usize,
    stopped: bool,
}

impl StopHoldback {
    pub fn new(stop: &[String]) -> Self {
        let stop: Vec<String> = stop.iter().filter(|s| !s.is_empty()).cloned().collect();
        let hold = stop
            .iter()
            .map(String::len)
            .max()
            .unwrap_or(0)
            .saturating_sub(1);
        Self {
            stop,
            acc: String::new(),
            sent: 0,
            hold,
            stopped: false,
        }
    }

    /// Adds `piece`; returns what may be sent now. After a `Stop`, nothing more.
    pub fn push(&mut self, piece: &str) -> Held {
        if self.stopped {
            return Held::Stop(String::new());
        }
        self.acc.push_str(piece);
        let bytes = self.acc.as_bytes();
        let from = self.sent.saturating_sub(self.hold);
        let hit = self
            .stop
            .iter()
            .filter_map(|s| find_bytes(&bytes[from..], s.as_bytes()).map(|at| from + at))
            .min();
        let (upto, ends) = match hit {
            Some(at) => (at, true),
            None => (self.acc.len() - self.held_tail(), false),
        };
        // Both offsets start a char: a stop sequence's first byte is a lead byte.
        let out = self.acc[self.sent..upto].to_string();
        self.sent = upto;
        if ends {
            self.stopped = true;
            Held::Stop(out)
        } else {
            Held::Emit(out)
        }
    }

    /// What is still held back once the text has ended without a stop.
    pub fn into_rest(self) -> String {
        if self.stopped {
            return String::new();
        }
        self.acc[self.sent..].to_string()
    }

    /// The longest tail of the text that is a strict prefix of a stop sequence.
    fn held_tail(&self) -> usize {
        let bytes = self.acc.as_bytes();
        (1..=self.hold.min(bytes.len()))
            .rev()
            .find(|&k| {
                let tail = &bytes[bytes.len() - k..];
                self.stop
                    .iter()
                    .any(|s| s.len() > k && s.as_bytes().starts_with(tail))
            })
            .unwrap_or(0)
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// What a generation hands out as it goes. `Break`, or `alive()` false: the
/// reader is gone, and the generation ends in [`Cancelled`].
pub trait TokenSink {
    /// Every refusal is past (budget, sampling, lease, model lock held).
    fn started(&mut self) -> ControlFlow<()>;
    /// Final text, never empty: never retracted, never cut later by a stop.
    fn delta(&mut self, text: &str) -> ControlFlow<()>;
    /// Read at every decode step, whether or not it emits.
    fn alive(&self) -> bool {
        true
    }
}

/// The reader left: the generation stopped early and nothing may be signed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("cancelled")]
pub struct Cancelled;

/// The streamed text is not a prefix of the decoded completion: no receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("decoder is not prefix-stable")]
pub struct NotPrefixStable;

/// A sink nobody reads: a whole answer through the streaming loop.
pub struct NoSink;

impl TokenSink for NoSink {
    fn started(&mut self) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
    fn delta(&mut self, _text: &str) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

/// A seed for a request that samples without one: 8 bytes of `/dev/urandom`,
/// cut by [`drawn_seed`].
pub fn fresh_seed() -> Result<u64, String> {
    use std::io::Read as _;
    let mut b = [0u8; 8];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .map_err(|e| format!("/dev/urandom: {e}"))?;
    Ok(drawn_seed(b))
}

/// A seed the engine draws, from 8 random bytes: under 2^53, so a receipt
/// read by a JavaScript client keeps it exact and the draws can be replayed.
pub fn drawn_seed(bytes: [u8; 8]) -> u64 {
    u64::from_le_bytes(bytes) & ((1 << 53) - 1)
}

pub trait LoadedModel: Send + Sync {
    /// Generate a completion for `prompt` (the chat template is the
    /// backend's business), bounded by `max_tokens`. Returns text and the
    /// (prompt_tokens, completion_tokens) the receipt reports.
    fn generate(&self, prompt: &Prompt, max_tokens: usize) -> anyhow::Result<(String, (u32, u32))>;
    /// [`Self::generate`] with the request's sampling and stop sequences, as
    /// the servers call it. This default serves greedy backends: sampling is
    /// refused ([`SamplingUnsupported`]), the text is cut once generated.
    fn generate_with(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
    ) -> anyhow::Result<Completion> {
        if params.sampling.is_some() {
            return Err(SamplingUnsupported.into());
        }
        if params.json_schema.is_some() {
            return Err(JsonSchemaUnsupported(
                "json_schema: this backend cannot mask its tokens".into(),
            )
            .into());
        }
        let (text, (prompt_tokens, completion_tokens)) =
            self.generate(prompt, params.max_tokens)?;
        let (text, stopped) = cut_at_stop(&text, &params.stop);
        Ok(Completion {
            text,
            prompt_tokens,
            completion_tokens,
            stopped,
            // generate() does not say its budget: finish_reason reads max_tokens.
            length: false,
        })
    }
    /// [`Self::generate_with`] for a caller who may leave before the answer:
    /// `sink.alive()` says, at every step, whether anyone still waits for it
    /// (`Err(Cancelled)` once nobody does). The text is the completion's;
    /// what `sink` is handed of it on the way is not read. This default
    /// cannot stop: it generates to the end. A backend that decodes step by
    /// step answers with its streaming loop.
    fn generate_whole(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        let _ = sink;
        self.generate_with(prompt, params)
    }
    /// [`Self::generate_with`], handing the text to `sink` as it is final.
    /// This default refuses first, then hands out the whole text at once.
    /// `Break` or `alive()` false: `Err(Cancelled)`, never a truncated `Ok`.
    fn generate_streaming(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        let c = self.generate_with(prompt, params)?;
        if sink.started().is_break() || !sink.alive() {
            return Err(Cancelled.into());
        }
        if !c.text.is_empty() && sink.delta(&c.text).is_break() {
            return Err(Cancelled.into());
        }
        Ok(c)
    }
    /// [`Self::generate_streaming`] as one turn of `conversation`
    /// (`crate::conversation`), when there is one: the backend may reuse the
    /// keys and values of its previous turn and records what it reused in
    /// it. This default keeps no conversation: one is refused
    /// ([`ConversationUnsupported`]).
    fn generate_turn(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
        conversation: Option<&crate::conversation::Turn>,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        if conversation.is_some() {
            return Err(ConversationUnsupported.into());
        }
        self.generate_streaming(prompt, params, sink)
    }
    /// The prompt this model was trained to answer. Qwen3's by default; a
    /// model of another family answers with its own template. The messages
    /// are text in it, never control tokens ([`crate::prompt`]).
    fn chat_prompt(&self, messages: &[crate::server::ChatMessage]) -> Prompt {
        crate::server::chat_prompt(messages)
    }
    /// A digest of the weights as loaded — goes into every receipt.
    fn model_digest_hex(&self) -> String;
    /// Where the arithmetic actually happens: `"cpu"` or `"cuda:N"`.
    ///
    /// SECURITY: the receipt attests a machine, not a chip. A reader
    /// who sees `ConfidentialVmWithGpu { gpu_mode: On }` concludes the data
    /// was in the GPU's protected region; if the model ran on the CPU, that
    /// conclusion is wrong and nothing in the receipt said so — the word
    /// "device" appeared nowhere in `receipt.rs` before 2026-08-22. Both
    /// devices are legitimate products (TDX-only inference is genuinely
    /// confidential, at a sixteenth of the price); what was not legitimate
    /// was leaving them indistinguishable.
    fn compute_device(&self) -> String;
    /// What this engine does NOT do with the caller's text.
    ///
    /// These are the properties an inference provider cannot honestly claim
    /// without owning the generation loop — vLLM ships shared prefix caching
    /// as a FEATURE, and six 2026 attack papers target exactly that surface.
    /// A backend declares them; the crate's tests hold each declaration to
    /// the code that is supposed to make it true, because a claim nobody
    /// checks is the defect this repository is named for.
    fn isolation(&self) -> Isolation;
    /// `(context tokens, prefill chunk)` the model was loaded with, when the
    /// backend computes them (`core_backend`'s `CoreModel`).
    fn serving_limits(&self) -> Option<(usize, usize)> {
        None
    }
    /// Tokens per prefill pass for a prompt of `prompt_tokens`, when the backend sizes
    /// it per request (`candle_backend::request_chunk`).
    fn prefill_chunk_for(&self, _prompt_tokens: usize) -> Option<usize> {
        None
    }
}

/// `(prompt tokens, prefill chunk)` at 1024, 2048, … below `ctx`, then at `ctx`. The
/// chunk does not grow with the length: two rows bound every length between them.
pub fn prefill_table(ctx: usize, chunk: impl Fn(usize) -> usize) -> Vec<(usize, usize)> {
    let mut lens: Vec<usize> = std::iter::successors(Some(1024usize), |n| n.checked_mul(2))
        .take_while(|&n| n < ctx)
        .collect();
    lens.push(ctx);
    lens.into_iter().map(|n| (n, chunk(n))).collect()
}

/// [`prefill_table`] of a loaded model, from its own limits; `None` when it has none.
pub fn prefill_chunks(m: &dyn LoadedModel) -> Option<Vec<(usize, usize)>> {
    let (ctx, _) = m.serving_limits()?;
    m.prefill_chunk_for(ctx)?;
    Some(prefill_table(ctx, |n| m.prefill_chunk_for(n).unwrap_or(0)))
}

/// The negative attestations, signed into every receipt. `false` everywhere
/// is the good answer: each field names something a shared inference server
/// normally does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Isolation {
    /// Keys and values from one request visible to the next.
    pub kv_cache_shared_between_requests: bool,
    /// A prompt-prefix cache surviving across sessions — the surface the
    /// 2026 timing papers attack.
    pub prefix_cache_shared_between_sessions: bool,
    /// Two clients' tokens in one forward pass.
    pub requests_batched_across_clients: bool,
    /// Any fragment of prompt or completion reaching a log.
    pub prompt_text_logged: bool,
}

impl Isolation {
    /// What this engine offers: none of the four. Named rather than written
    /// out at each site so adding a field cannot silently default to `false`
    /// in a backend that has not thought about it.
    pub const NONE_OF_IT: Self = Self {
        kv_cache_shared_between_requests: false,
        prefix_cache_shared_between_sessions: false,
        requests_batched_across_clients: false,
        prompt_text_logged: false,
    };
}

/// Open weights as files: the shard index when there is one, then every
/// weight file by the name the served-model digest lists (`model_digest`).
pub struct WeightFiles<'a> {
    pub index_json: Option<&'a [u8]>,
    pub files: Vec<(&'a str, &'a [u8])>,
}

/// A backend: turns an encrypted artifact + released key into a loaded model.
pub trait InferenceBackend {
    fn name(&self) -> &'static str;
    fn load(
        &self,
        encrypted_weights: &[u8],
        key: &ReleasedKey,
    ) -> anyhow::Result<Box<dyn LoadedModel>>;
    /// Open weights, one file or shards, never sealed: a sealed model is one
    /// blob and goes through [`Self::load`]. The key proves the gate released.
    fn load_files(
        &self,
        weights: &WeightFiles<'_>,
        key: &ReleasedKey,
    ) -> anyhow::Result<Box<dyn LoadedModel>> {
        let _ = (weights, key);
        anyhow::bail!(
            "the {} backend loads one weight blob, through load()",
            self.name()
        )
    }
}

/// Test backend: no weights, echoes the prompt. Lives in the crate (not only
/// in tests) so the HTTP surface can be exercised end-to-end without a GPU.
pub struct EchoBackend;

pub struct EchoModel {
    digest: String,
}

impl LoadedModel for EchoModel {
    fn generate(&self, prompt: &Prompt, max_tokens: usize) -> anyhow::Result<(String, (u32, u32))> {
        let words: Vec<&str> = prompt.split_whitespace().collect();
        let out: Vec<&str> = words
            .iter()
            .rev()
            .copied()
            .take(max_tokens.max(1))
            .collect();
        Ok((out.join(" "), (words.len() as u32, out.len() as u32)))
    }
    fn model_digest_hex(&self) -> String {
        self.digest.clone()
    }
    fn compute_device(&self) -> String {
        "cpu".into()
    }
    fn isolation(&self) -> Isolation {
        // True of the echo backend for the dullest reason: it has no cache,
        // no batch and no logger.
        Isolation::NONE_OF_IT
    }
}

impl InferenceBackend for EchoBackend {
    fn name(&self) -> &'static str {
        "echo"
    }
    fn load(
        &self,
        encrypted_weights: &[u8],
        key: &ReleasedKey,
    ) -> anyhow::Result<Box<dyn LoadedModel>> {
        use sha2::{Digest, Sha256};
        // The echo backend does not decrypt; it still proves the plumbing:
        // the digest covers the ciphertext and the key id, so a receipt from
        // a different artifact or key reads differently.
        let mut h = Sha256::new();
        h.update(encrypted_weights);
        h.update(key.key_id().as_bytes());
        let digest = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        Ok(Box::new(EchoModel { digest }))
    }
}

/// Enforce the operator's expected model digest, when one was given.
///
/// SECURITY: closes the "model substitution" line of THREAT_MODEL.md — the
/// digest was PUBLISHED in receipts but nothing ever compared it, so a
/// swapped weights file served happily and only the client could notice,
/// after the fact. `None` means the operator pinned nothing (their call,
/// and the receipts still expose what ran); `Some` must match exactly,
/// case-insensitively, and an empty expectation is a refusal, not a wildcard.
pub fn enforce_expected_digest(expected: Option<&str>, loaded_hex: &str) -> anyhow::Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    if expected.is_empty() {
        anyhow::bail!("--expected-model-digest is empty: pass the digest or drop the flag");
    }
    if !expected.eq_ignore_ascii_case(loaded_hex) {
        anyhow::bail!(
            "model digest mismatch: loaded {loaded_hex}, expected {expected} — \
             the weights on disk are not the model the operator pinned"
        );
    }
    Ok(())
}

/// [`enforce_expected_digest`] on weights not yet loaded: the served digest
/// of the bytes read (`model_spec::served_digest`), before any backend parses
/// them, so weights other than the pinned ones reach neither safetensors nor
/// a GPU. `Ok` when nothing is pinned, or when a file is sealed: the backend
/// opens it under the released key, and the pin is compared on the model it
/// loads. A digest that cannot be computed is a refusal.
pub fn enforce_expected_digest_before_load(
    expected: Option<&str>,
    artifacts: &crate::model_spec::ModelArtifacts,
    weights: &WeightFiles<'_>,
) -> anyhow::Result<()> {
    if expected.is_none() || weights.files.iter().any(|(_, b)| crate::seal::is_sealed(b)) {
        return Ok(());
    }
    let served = crate::model_spec::served_digest(
        &artifacts.config_json,
        &artifacts.tokenizer_json,
        weights,
    )
    .map_err(|e| anyhow::anyhow!("--expected-model-digest: not compared, refused: {e:#}"))?;
    enforce_expected_digest(expected, &served)
}

#[cfg(test)]
mod digest_tests {
    /// A failure means an oversized prompt reaches the GPU again, or a long
    /// generation runs off the end of the context half-way through an answer.
    #[test]
    fn the_context_window_is_enforced_before_the_model_runs() {
        use super::{generation_budget, PromptTooLong};
        assert_eq!(generation_budget(10, 100, 4096), Ok(100));
        assert_eq!(
            generation_budget(4000, 100, 4096),
            Ok(96),
            "clamped to what is left"
        );
        assert_eq!(generation_budget(4095, 100, 4096), Ok(1));
        for too_long in [4096, 42_021] {
            assert_eq!(
                generation_budget(too_long, 100, 4096),
                Err(PromptTooLong {
                    prompt_tokens: too_long,
                    context_tokens: 4096
                })
            );
        }
    }

    use super::*;

    /// A pinned digest is enforced; an unpinned load still serves. A failure
    /// here would mean the substitution check is decorative again — published
    /// in receipts, compared nowhere.
    #[test]
    fn a_pinned_digest_refuses_substitution_and_no_pin_means_no_check() {
        assert!(enforce_expected_digest(None, "abc123").is_ok());
        assert!(enforce_expected_digest(Some("ABC123"), "abc123").is_ok());
        let err = enforce_expected_digest(Some("abc124"), "abc123")
            .unwrap_err()
            .to_string();
        assert!(err.contains("mismatch") && err.contains("abc124"), "{err}");
        assert!(
            enforce_expected_digest(Some(""), "abc123").is_err(),
            "empty is not a wildcard"
        );
    }

    /// SECURITY: the pin is compared on the bytes read, before any backend
    /// parses them: bytes that are no safetensors file are refused for their
    /// digest alone, both digests named; the pinned bytes pass, with the
    /// digest a backend serves them under; with nothing pinned, or a sealed
    /// file, the backend has them and the pin is compared on its model.
    #[test]
    fn the_pin_is_compared_on_the_bytes_before_any_backend_reads_them() {
        let artifacts = crate::model_spec::ModelArtifacts {
            config_json: b"{}".to_vec(),
            tokenizer_json: b"{}".to_vec(),
        };
        let one = |bytes| WeightFiles {
            index_json: None,
            files: vec![(crate::shards::SINGLE_FILE, bytes)],
        };
        let hostile = one(b"not a safetensors file");
        let served = crate::model_spec::served_digest(
            &artifacts.config_json,
            &artifacts.tokenizer_json,
            &hostile,
        )
        .unwrap();
        assert!(enforce_expected_digest_before_load(Some(&served), &artifacts, &hostile).is_ok());
        let other = "0".repeat(64);
        let err = enforce_expected_digest_before_load(Some(&other), &artifacts, &hostile)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("mismatch") && err.contains(&served) && err.contains(&other),
            "{err}"
        );
        assert!(enforce_expected_digest_before_load(Some(""), &artifacts, &hostile).is_err());
        assert!(enforce_expected_digest_before_load(None, &artifacts, &hostile).is_ok());
        let sealed = crate::seal::seal(b"weights", &[7; 32], &[0; 12]).unwrap();
        assert!(
            enforce_expected_digest_before_load(Some(&other), &artifacts, &one(&sealed)).is_ok(),
            "a sealed file is pinned on the model the backend loads"
        );
    }

    /// The earliest stop sequence in the text wins, whatever its rank in the list.
    #[test]
    fn the_text_is_cut_before_the_earliest_stop_sequence() {
        let stop = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let t = "alpha beta STOP gamma";
        assert_eq!(
            cut_at_stop(t, &stop(&["gamma", "STOP"])),
            ("alpha beta ".into(), true)
        );
        assert_eq!(cut_at_stop(t, &stop(&["zzz"])), (t.into(), false));
        assert_eq!(cut_at_stop(t, &stop(&[])), (t.into(), false));
        assert_eq!(cut_at_stop(t, &stop(&[""])), (t.into(), false));
        assert_eq!(cut_at_stop(t, &stop(&["alpha"])), (String::new(), true));
    }

    /// A greedy-only backend refuses sampling instead of answering greedily
    /// under a temperature, and still honours stop sequences.
    #[test]
    fn a_greedy_backend_refuses_sampling_and_cuts_at_stop() {
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        let echo = EchoBackend.load(b"w", &key).unwrap();
        let mut params = GenerationParams::greedy(8);
        params.stop = vec!["c".into()];
        let c = echo.generate_with(&"a b c d".into(), &params).unwrap();
        assert_eq!((c.text.as_str(), c.stopped), ("d ", true));
        assert_eq!(c.finish_reason(8), "stop");
        params.sampling = Some(Sampling {
            temperature: 0.7,
            top_p: 1.0,
            seed: 1,
        });
        let e = echo.generate_with(&"a b c d".into(), &params).unwrap_err();
        assert!(e.downcast_ref::<SamplingUnsupported>().is_some(), "{e}");
        let (status, message) = crate::server::generate_error(e);
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST, "{message}");
        assert!(message.contains("temperature"), "{message}");
    }

    /// Messages the model's template refuses are the caller's error, a 400
    /// that says why, not a 500.
    #[test]
    fn a_conversation_the_template_refuses_is_a_400() {
        let e: anyhow::Error =
            ChatRefused("the conversation starts with an assistant".into()).into();
        let (status, message) = crate::server::generate_error(e);
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST, "{message}");
        assert!(message.contains("starts with an assistant"), "{message}");
    }

    /// Every split of `text` into pieces at char boundaries.
    fn splits(text: &str) -> Vec<Vec<&str>> {
        let cuts: Vec<usize> = text.char_indices().map(|(i, _)| i).skip(1).collect();
        (0u32..(1u32 << cuts.len()))
            .map(|mask| {
                let mut pieces = Vec::new();
                let mut from = 0;
                for (bit, &at) in cuts.iter().enumerate() {
                    if mask & (1u32 << bit) != 0 {
                        pieces.push(&text[from..at]);
                        from = at;
                    }
                }
                pieces.push(&text[from..]);
                pieces
            })
            .collect()
    }

    /// Feeds `pieces` as a generation would: what was emitted, whether a stop
    /// ended it, and the text consumed until then.
    fn feed(stop: &[String], pieces: &[&str]) -> (String, bool, String, Vec<String>) {
        let mut h = StopHoldback::new(stop);
        let (mut out, mut consumed, mut emissions) = (String::new(), String::new(), Vec::new());
        for piece in pieces {
            consumed.push_str(piece);
            match h.push(piece) {
                Held::Emit(s) => {
                    out.push_str(&s);
                    emissions.push(s);
                }
                Held::Stop(s) => {
                    out.push_str(&s);
                    emissions.push(s);
                    return (out, true, consumed, emissions);
                }
            }
        }
        let rest = h.into_rest();
        out.push_str(&rest);
        emissions.push(rest);
        (out, false, consumed, emissions)
    }

    fn owned(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    /// T1.
    /// The streamed text equals the text cut once generated, for any split.
    /// "STOP", "é" and "本語" give an odd hold; a failure is a wrong answer or a panic.
    #[test]
    fn stop_holdback_matches_cut_at_stop_on_every_split() {
        let sets: &[&[&str]] = &[
            &["STOP"],
            &["é"],
            &["abcd", "bc"],
            &["本語"],
            &["ST", "STOP"],
            &["zz"],
            &[],
            &[""],
            &["é ST"],
        ];
        let texts = ["aSTOPb", "éé STé", "xabcdy", "日本語のé", "ab", "STST", ""];
        for set in sets {
            let stop = owned(set);
            for text in texts {
                for pieces in splits(text) {
                    let (out, stopped, consumed, _) = feed(&stop, &pieces);
                    assert_eq!(
                        (out, stopped),
                        cut_at_stop(&consumed, &stop),
                        "{text:?} split {pieces:?} stop {stop:?}"
                    );
                    if !stopped {
                        assert_eq!(consumed, text);
                    }
                }
            }
        }
    }

    /// T2.
    /// A stop sequence completed by a later piece was never partly sent.
    #[test]
    fn stop_holdback_never_emits_the_start_of_a_stop_sequence() {
        let (out, stopped, _, emissions) = feed(&owned(&["STOP"]), &["aST", "OP", "b"]);
        assert_eq!((out.as_str(), stopped), ("a", true));
        assert!(emissions.iter().all(|e| !e.contains('S')), "{emissions:?}");

        let (out, stopped, _, emissions) = feed(&owned(&["éta"]), &["日本é", "t", "a"]);
        assert_eq!((out.as_str(), stopped), ("日本", true));
        assert_eq!(emissions, ["日本", "", ""]);

        // A start that does not complete is released, whole, once it cannot.
        let (out, stopped, _, emissions) = feed(&owned(&["éta"]), &["日本é", "t", "x"]);
        assert_eq!((out.as_str(), stopped), ("日本étx", false));
        assert_eq!(emissions, ["日本", "", "étx", ""]);
    }

    /// Records what a generation hands out; breaks where asked.
    #[derive(Default)]
    struct Recorder {
        started: usize,
        deltas: Vec<String>,
        break_at_start: bool,
        break_at_delta: Option<usize>,
    }

    impl TokenSink for Recorder {
        fn started(&mut self) -> ControlFlow<()> {
            self.started += 1;
            if self.break_at_start {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }
        fn delta(&mut self, text: &str) -> ControlFlow<()> {
            self.deltas.push(text.to_string());
            if self.break_at_delta == Some(self.deltas.len()) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }
    }

    /// T3.
    /// A refusal comes before `started`, so it can still be an HTTP status; a
    /// reader that breaks gets `Cancelled`, never a truncated answer to sign.
    #[test]
    fn the_default_streaming_refuses_first_then_emits_the_whole_text_once() {
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        let echo = EchoBackend.load(b"w", &key).unwrap();
        let mut sampled = GenerationParams::greedy(8);
        sampled.sampling = Some(Sampling {
            temperature: 0.7,
            top_p: 1.0,
            seed: 1,
        });
        let mut sink = Recorder::default();
        let e = echo
            .generate_streaming(&"a b c d".into(), &sampled, &mut sink)
            .unwrap_err();
        assert!(e.downcast_ref::<SamplingUnsupported>().is_some(), "{e}");
        assert_eq!((sink.started, sink.deltas.len()), (0, 0));

        let greedy = GenerationParams::greedy(8);
        let mut sink = Recorder::default();
        let c = echo
            .generate_streaming(&"a b c d".into(), &greedy, &mut sink)
            .unwrap();
        assert_eq!(c.text, "d c b a");
        assert_eq!(sink.started, 1);
        assert_eq!(sink.deltas, [c.text.as_str()]);

        let mut sink = Recorder::default();
        let c = echo
            .generate_streaming(&"".into(), &greedy, &mut sink)
            .unwrap();
        assert_eq!((c.text.as_str(), sink.started), ("", 1));
        assert!(sink.deltas.is_empty(), "no empty delta");

        for sink in [
            Recorder {
                break_at_start: true,
                ..Recorder::default()
            },
            Recorder {
                break_at_delta: Some(1),
                ..Recorder::default()
            },
        ] {
            let mut sink = sink;
            let e = echo
                .generate_streaming(&"a b c d".into(), &greedy, &mut sink)
                .unwrap_err();
            assert!(e.downcast_ref::<Cancelled>().is_some(), "{e}");
        }
    }
}
