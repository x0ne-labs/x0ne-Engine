//! The candle backend, the tests' reference on the CPU since 2026-11-04
//! (feature `candle-reference`, which `cargo build` never has): x0ne Core
//! serves (`core_backend`).
//!
//! Scope: Qwen3 dense by [`crate::qwen3`], a fork of candle's implementation,
//! OLMo 2 and OLMo 3 by [`crate::decoder`] (the family is read from
//! config.json, see [`crate::arch`]), greedy decoding, one request at a time,
//! on the CPU. No batching, no paged KV, no speculative decoding.
//!
//! Confidentiality shape: `load` receives the WEIGHT BYTES and the released
//! key — never a path it dereferences on its own. In production the bytes
//! arrive decrypted from the artifact layer inside the protected VM memory
//! and go to the device from there; the boundary (plaintext exists in CVM
//! RAM during load) is the one documented in THREAT_MODEL.md. v0 loads
//! sealed (`seal.rs`, AES-256-GCM under the released key) or, for public
//! open-weights models, plaintext safetensors — stated at the branch; the
//! digest in receipts covers what was actually loaded. Open weights may come in
//! shards ([`InferenceBackend::load_files`]), each file parsed in place.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::Config;
use tokenizers::Tokenizer;

use crate::arch::{Architecture, DecoderConfig};
use crate::backend::{
    Cancelled, Completion, GenerationParams, Held, InferenceBackend, LoadedModel, NoSink,
    NotPrefixStable, Sampling, TokenSink, WeightFiles,
};
use crate::conversation::{Resident, Reused, Turn};
use crate::decoder::{Decoder, KvCache};
use crate::keys::ReleasedKey;
use crate::ministral3::Ministral3;
use crate::prompt::Prompt;
use crate::qwen3::{Projections, Qwen3};

pub(crate) use crate::decode_stream::StreamDecoder;
pub use crate::model_spec::{served_digest, ModelArtifacts, ModelSpec};

pub struct CandleBackend {
    pub artifacts: ModelArtifacts,
    pub device: Device,
}

/// Former names, from when Qwen3 was the only family.
pub type Qwen3Artifacts = ModelArtifacts;
pub type CandleQwen3Backend = CandleBackend;

impl CandleBackend {
    pub fn cpu(artifacts: ModelArtifacts) -> Self {
        Self {
            artifacts,
            device: Device::Cpu,
        }
    }

    /// What `load` will hand the tensors.
    ///
    /// BF16 on the GPU, F32 on CPU. Not a tuning knob: Qwen3 ships BF16, and
    /// upconverting on a GPU doubles what has to cross the encrypted PCIe
    /// channel and sit in the protected region for no accuracy anyone asked
    /// for. On CPU, BF16 kernels are the slow path in candle, so F32 stays.
    fn dtype(&self) -> DType {
        if self.device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        }
    }
}

/// candle's own Qwen3 config, from Engine's reading of it
/// ([`crate::model_spec::parse_qwen3_config`]).
pub fn parse_qwen3_config(config_json: &[u8]) -> anyhow::Result<Config> {
    to_candle_qwen3(&crate::model_spec::parse_qwen3_config(config_json)?)
}

/// Field for field: both types read and write the same names.
fn to_candle_qwen3(c: &crate::model_spec::Qwen3Config) -> anyhow::Result<Config> {
    serde_json::to_value(c)
        .and_then(serde_json::from_value)
        .map_err(|e| anyhow::anyhow!("config.json: {e}"))
}

/// What only candle does with a [`ModelSpec`].
trait CandleSpec {
    /// `p`: `Fused` where the model owns its weights, `Split` where they are shared.
    fn build(&self, vb: VarBuilder, p: Projections) -> anyhow::Result<ArchModel>;
    /// Ministral 3: every tensor is the text model's or the vision tower's
    /// (`ministral3::check_tensor_names`). Other families: nothing to check.
    fn check_tensor_names<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
    ) -> anyhow::Result<()>;
}

impl CandleSpec for ModelSpec {
    fn build(&self, vb: VarBuilder, p: Projections) -> anyhow::Result<ArchModel> {
        let built = |e| anyhow::anyhow!("model build: {e}");
        Ok(match (self.qwen3(), self.arch) {
            (Some(cfg), _) => {
                ArchModel::Qwen3(Qwen3::new(&to_candle_qwen3(cfg)?, vb, p).map_err(built)?)
            }
            (None, Architecture::Ministral3) => {
                ArchModel::Ministral3(Ministral3::new(&self.decoder, vb, p).map_err(built)?)
            }
            (None, _) => ArchModel::Decoder(
                Decoder::new(&self.decoder, vb).map_err(built)?,
                KvCache::default(),
            ),
        })
    }

    fn check_tensor_names<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
    ) -> anyhow::Result<()> {
        match self.arch {
            Architecture::Ministral3 => crate::ministral3::check_tensor_names(&self.decoder, names),
            _ => Ok(()),
        }
    }
}

enum ArchModel {
    Qwen3(Qwen3),
    Ministral3(Ministral3),
    Decoder(Decoder, KvCache),
}

impl ArchModel {
    /// `input`: (1, tokens) u32. Returns the last position's logits.
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Qwen3(m) => m.forward(input, offset),
            Self::Ministral3(m) => m.forward(input, offset),
            Self::Decoder(d, cache) => d.forward(&input.flatten_all()?.to_vec1()?, offset, cache),
        }
    }
    /// Qwen3 and Ministral 3: zeros over the keys and values written, then
    /// dropped. OLMo's cache, grown by concatenation, is dropped.
    fn clear_kv_cache(&mut self) -> candle_core::Result<()> {
        match self {
            Self::Qwen3(m) => m.clear_kv_cache(),
            Self::Ministral3(m) => m.clear_kv_cache(),
            Self::Decoder(_, cache) => {
                cache.clear();
                Ok(())
            }
        }
    }
    /// Whether the cache can stay for a conversation's next turn: written in
    /// place. OLMo's leaves a freed copy at each step, never zeroed.
    fn keeps_conversations(&self) -> bool {
        !matches!(self, Self::Decoder(..))
    }
    /// Positions in the cache; 0 for OLMo, which keeps no conversation.
    fn kv_len(&self) -> usize {
        match self {
            Self::Qwen3(m) => m.kv_len(),
            Self::Ministral3(m) => m.kv_len(),
            Self::Decoder(..) => 0,
        }
    }
    fn truncate_kv_cache(&mut self, len: usize) {
        match self {
            Self::Qwen3(m) => m.truncate_kv_cache(len),
            Self::Ministral3(m) => m.truncate_kv_cache(len),
            Self::Decoder(..) => {}
        }
    }
    /// Positions this request may fill; the Qwen3 and Ministral 3 caches are
    /// allocated once at that size.
    fn reserve(&mut self, tokens: usize) {
        match self {
            Self::Qwen3(m) => m.reserve(tokens),
            Self::Ministral3(m) => m.reserve(tokens),
            Self::Decoder(..) => {}
        }
    }
}

/// The model and the conversation its cache holds between turns
/// (`crate::conversation`), under one lock: the cache and whose it is
/// cannot disagree. Derefs to the model.
struct Slot {
    model: ArchModel,
    resident: Option<Resident>,
    /// [`crate::conversation::KEEP_FOR`]; shorter in tests.
    keep_for: Duration,
    /// A [`reap`] thread waits on `resident`'s deadline.
    reaping: bool,
    /// Positions the last prefill computed: the tests' probe of a reuse.
    last_prefill: usize,
}

impl std::ops::Deref for Slot {
    type Target = ArchModel;
    fn deref(&self) -> &ArchModel {
        &self.model
    }
}

impl std::ops::DerefMut for Slot {
    fn deref_mut(&mut self) -> &mut ArchModel {
        &mut self.model
    }
}

impl Slot {
    fn shared(model: ArchModel) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            model,
            resident: None,
            keep_for: crate::conversation::KEEP_FOR,
            reaping: false,
            last_prefill: 0,
        }))
    }

    /// Zeroes the cache and forgets whose it was.
    fn evict(&mut self) -> candle_core::Result<()> {
        self.resident = None;
        self.model.clear_kv_cache()
    }

    /// Before a turn: `(tokens reused, number of the turn)`. A turn that
    /// continues the resident conversation keeps the prefix its prompt
    /// shares with it; every other request finds the cache zeroed. The
    /// resident handle is spent either way.
    fn admit(&mut self, turn: Option<&Turn>, prompt: &[u32]) -> candle_core::Result<(usize, u32)> {
        let resident = self.resident.take();
        let continued = match (turn, resident.as_ref()) {
            (Some(turn), Some(resident)) => resident.continued_by(turn, prompt),
            _ => None,
        };
        drop(resident);
        match continued {
            Some((reuse, number)) => {
                self.model.truncate_kv_cache(reuse);
                Ok((reuse, number))
            }
            None => self.evict().map(|()| (0, 1)),
        }
    }

    /// After a turn: the cache kept for the conversation's next turn when
    /// the turn asked it and ran to its end (`generated`), zeroed otherwise.
    /// Whether it was kept.
    fn settle(
        &mut self,
        turn: Option<&Turn>,
        number: u32,
        prompt: &[u32],
        generated: Option<&Vec<u32>>,
    ) -> candle_core::Result<bool> {
        let (Some(turn), Some(generated)) = (turn, generated) else {
            return self.evict().map(|()| false);
        };
        let len = self.model.kv_len();
        if turn.end || !self.model.keeps_conversations() || len > prompt.len() + generated.len() {
            return self.evict().map(|()| false);
        }
        let mut ids = Vec::with_capacity(len);
        ids.extend(prompt.iter().chain(generated).take(len));
        self.resident = Some(Resident::new(turn, number, ids));
        Ok(true)
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        // Nothing to return an error to: the buffers are dropped anyway.
        let _ = self.evict();
    }
}

/// Zeroes the conversation resident in `held` once its time passed without
/// a turn. Ends when none is resident, or the model is gone.
fn reap(held: &std::sync::Weak<Mutex<Slot>>) {
    loop {
        let Some(strong) = held.upgrade() else {
            return;
        };
        let wait = {
            let mut h = strong
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let now = Instant::now();
            match h.resident.as_ref().map(|r| r.deadline(h.keep_for)) {
                Some(deadline) if deadline > now => deadline - now,
                due => {
                    if due.is_some() {
                        // Dropped whatever the zeroing says: no caller to tell.
                        let _ = h.evict();
                    }
                    h.reaping = false;
                    return;
                }
            }
        };
        drop(strong);
        std::thread::sleep(wait);
    }
}

struct Loaded {
    /// candle's forward takes `&mut self` (KV cache); one request at a time
    /// in v0, made explicit by the lock rather than hidden by a refcell.
    model: Arc<Mutex<Slot>>,
    arch: Architecture,
    tokenizer: Arc<Tokenizer>,
    digest_hex: String,
    device: Device,
    eos_ids: Vec<u32>,
    /// `max_position_embeddings` of the loaded config.
    context_tokens: usize,
    /// Tokens per prefill forward at the full context; see `prefill_chunk`.
    prefill_chunk: usize,
    /// What sizes each request's chunk: `request_chunk`.
    decoder: DecoderConfig,
    dtype_bytes: u64,
    /// GPU bytes for one generation beyond the weights; `None` on the CPU.
    budget: Option<u64>,
    /// [`ModelSpec::default_system`].
    default_system: Option<&'static str>,
    /// The vocabulary as bytes, for `json_schema` masks: read once, at the
    /// first request that asks (79 ms for Qwen3, measured 2026-10-30).
    token_table: std::sync::OnceLock<Result<crate::constrain::TokenTable, String>>,
}

impl Loaded {
    /// [`Self::token_table`], read on first use; why it could not be, as text.
    fn token_table(&self) -> Result<&crate::constrain::TokenTable, String> {
        self.token_table
            .get_or_init(|| {
                crate::constrain::TokenTable::from_tokenizer(&self.tokenizer, &self.eos_ids)
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    fn chunk_for(&self, prompt_tokens: usize) -> usize {
        request_chunk(
            &self.decoder,
            self.dtype_bytes,
            self.context_tokens,
            self.budget,
            self.prefill_chunk,
            prompt_tokens,
        )
    }
}

/// candle-kernels 0.11 softmax indexes with `int` (reduce.cu:241,254,269): past
/// heads * q_len * kv_len = 2^31 it reads and writes out of bounds. Measured
/// 2026-10-10: Qwen3-4B (32 heads) breaks at 8193 tokens, Qwen3-0.6B (16) at 11586.
pub fn prefill_chunk(num_heads: usize, context_tokens: usize) -> usize {
    let per_query_row = num_heads.max(1) * context_tokens.max(1);
    (i32::MAX as usize / per_query_row).clamp(1, 2048)
}

/// Device bytes one generation needs beyond the weights, by the tensors
/// candle 0.11 allocates on the GPU path (qwen3.rs `forward_standard_attn`):
/// the KV cache, one layer's K/V copied by `ConcatKvCache::append`, K/V
/// repeated to every query head, the scores (two of `q·kᵀ`, scaled, masked,
/// softmax alive at once; three counted), the causal mask and one MLP.
pub fn serving_peak_bytes(d: &DecoderConfig, dtype_bytes: u64, ctx: usize, chunk: usize) -> u64 {
    prefill_peak_bytes(d, dtype_bytes, ctx, ctx, chunk)
}

/// [`serving_peak_bytes`] for one prefill pass: the KV cache still reserved for `ctx`
/// tokens, every other term over the `kv_len` keys the pass attends to. The MLP counts
/// four: the fork's fused gate+up output stays alive with the activation and the product.
pub fn prefill_peak_bytes(
    d: &DecoderConfig,
    dtype_bytes: u64,
    ctx: usize,
    kv_len: usize,
    chunk: usize,
) -> u64 {
    let (ctx, kv_len, chunk) = (ctx as u64, kv_len as u64, chunk as u64);
    let (heads, kv_heads, head_dim) =
        (d.num_heads as u64, d.num_kv_heads as u64, d.head_dim as u64);
    let layer_kv = |n: u64| 2 * kv_heads * head_dim * dtype_bytes * n;
    let kv_cache = d.num_layers as u64 * layer_kv(ctx);
    let repeated = 2 * heads * head_dim * dtype_bytes * kv_len;
    let scores = 3 * heads * chunk * kv_len * dtype_bytes;
    let mask = chunk * kv_len * (4 + dtype_bytes);
    let mlp = 4 * chunk * d.intermediate_size as u64 * dtype_bytes;
    kv_cache + layer_kv(kv_len) + repeated + scores + mask + mlp
}

/// Largest pass [`request_chunk`] grows to above `at_context`. On the RTX 5090 (Qwen3-8B,
/// 2026-10-20): with 2048-token passes the first token came 9 % later at 4k than with 1063.
pub const PASS_CAP: usize = 1024;

/// Tokens per prefill pass for a prompt of `prompt_tokens`: the largest chunk whose
/// [`prefill_peak_bytes`] fits `budget` over the prompt's keys, under the softmax's
/// i32 bound (`prefill_chunk`) and `at_context.max(PASS_CAP)`. Never below `at_context`,
/// the chunk computed at the full context: a model where it is 1024 or more keeps it.
pub fn request_chunk(
    d: &DecoderConfig,
    dtype_bytes: u64,
    ctx: usize,
    budget: Option<u64>,
    at_context: usize,
    prompt_tokens: usize,
) -> usize {
    let p = prompt_tokens.clamp(1, ctx.max(1));
    let cap = prefill_chunk(d.num_heads, p)
        .min(p)
        .min(at_context.max(PASS_CAP));
    let Some(budget) = budget else {
        return cap;
    };
    let fits = |c: usize| prefill_peak_bytes(d, dtype_bytes, ctx, p, c) <= budget;
    if fits(cap) {
        return cap;
    }
    // fits(lo) or lo == 1, !fits(hi): the peak grows with the chunk.
    let (mut lo, mut hi) = (1, cap);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// GPU bytes one generation may use beyond the weights: `free` less [`gpu_reserve`].
pub fn serving_budget(gpu: Option<(u64, u64)>) -> Option<u64> {
    gpu.map(|(free, total)| free.saturating_sub(gpu_reserve(total)))
}

/// Device bytes kept free besides the serving peak: cuBLAS workspaces, the
/// allocator's slack, NCCL-free CUDA context growth. 1 GiB or 2 % of the card.
pub fn gpu_reserve(total: u64) -> u64 {
    (1u64 << 30).max(total / 50)
}

/// Context and prefill chunk a model is served with. `gpu`: (free, total)
/// device bytes once the weights are resident; `None` on the CPU.
/// The context comes first: the largest multiple of 256 tokens, up to the
/// model's own, that fits with a 64-token chunk; then the largest chunk that
/// fits at that context. Under 1024 tokens of context, the load is refused.
pub fn serving_limits(
    d: &DecoderConfig,
    dtype_bytes: u64,
    gpu: Option<(u64, u64)>,
) -> Result<(usize, usize), String> {
    const STEP: usize = 256;
    const MIN_CHUNK: usize = 64;
    const MIN_CONTEXT: usize = 1024;
    let full = d.max_position_embeddings;
    let (Some((free, total)), Some(budget)) = (gpu, serving_budget(gpu)) else {
        return Ok((full, prefill_chunk(d.num_heads, full)));
    };
    let fits = |ctx: usize, chunk: usize| serving_peak_bytes(d, dtype_bytes, ctx, chunk) <= budget;
    let mut ctx = full;
    while ctx >= MIN_CONTEXT && !fits(ctx, MIN_CHUNK.min(ctx)) {
        ctx = if ctx == full && !full.is_multiple_of(STEP) {
            full / STEP * STEP
        } else {
            ctx - STEP
        };
    }
    if ctx < MIN_CONTEXT {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        return Err(format!(
            "{:.1} GiB free of {:.1} GiB on the GPU once the weights are resident: \
             not enough for a {MIN_CONTEXT}-token context ({:.1} GiB with the reserve)",
            gib(free),
            gib(total),
            gib(serving_peak_bytes(d, dtype_bytes, MIN_CONTEXT, MIN_CHUNK) + gpu_reserve(total)),
        ));
    }
    let mut chunk = prefill_chunk(d.num_heads, ctx).max(MIN_CHUNK.min(ctx));
    while chunk > MIN_CHUNK && !fits(ctx, chunk) {
        chunk = (chunk / 2).max(MIN_CHUNK);
    }
    // Back up from the halving: the largest chunk that still fits.
    let mut hi = (chunk * 2).min(prefill_chunk(d.num_heads, ctx));
    while hi > chunk && !fits(ctx, hi) {
        hi -= (hi - chunk).div_ceil(8).max(1);
    }
    Ok((ctx, hi.max(chunk)))
}

/// `(free, total)` bytes of the model's GPU; `None` on the CPU.
/// candle runs on the CPU only since 2026-11-04 (the GPU is x0ne Core's).
#[allow(clippy::unnecessary_wraps)]
fn gpu_memory(device: &Device) -> anyhow::Result<Option<(u64, u64)>> {
    let _ = device;
    Ok(None)
}

/// The token a masked host `row` picks on this backend: greedy, or candle's
/// draw over a CPU tensor of it (what `stepped::pick` did until
/// `crate::sampler`, whose draws are these). A row that holds a NaN picks none.
fn pick(
    row: &[f32],
    sampler: Option<&mut candle_transformers::generation::LogitsProcessor>,
) -> anyhow::Result<u32> {
    match sampler {
        None => crate::stepped::argmax(row),
        Some(s) if !row.iter().any(|v| v.is_nan()) => {
            Some(s.sample(&Tensor::from_slice(row, row.len(), &Device::Cpu)?)?)
        }
        Some(_) => None,
    }
    .ok_or_else(|| anyhow::anyhow!("the logits hold a NaN: no token"))
}

/// candle's sampler, one per request: no draw state outlives it. candle reads
/// `top_p` 0 as "no nucleus"; the nucleus of mass 0 is the top token, argmax.
pub(crate) fn logits_processor(s: Sampling) -> candle_transformers::generation::LogitsProcessor {
    use candle_transformers::generation::{LogitsProcessor, Sampling as Draw};
    let draw = if s.temperature < 1e-7 || s.top_p <= 0.0 {
        Draw::ArgMax
    } else if s.top_p >= 1.0 {
        Draw::All {
            temperature: s.temperature,
        }
    } else {
        Draw::TopP {
            p: s.top_p,
            temperature: s.temperature,
        }
    };
    LogitsProcessor::from_sampling(s.seed, draw)
}

impl LoadedModel for Loaded {
    fn generate(&self, prompt: &Prompt, max_tokens: usize) -> anyhow::Result<(String, (u32, u32))> {
        let c = self.generate_with(prompt, &GenerationParams::greedy(max_tokens))?;
        Ok((c.text, (c.prompt_tokens, c.completion_tokens)))
    }

    fn generate_with(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
    ) -> anyhow::Result<Completion> {
        self.generate_streaming(prompt, params, &mut NoSink)
    }

    fn generate_streaming(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        self.generate_turn(prompt, params, None, sink)
    }

    fn generate_turn(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
        conversation: Option<&Turn>,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        // SECURITY: the messages' text cannot become control tokens (crate::prompt).
        let prompt_ids = crate::prompt::encode(&self.tokenizer, prompt)?;
        let prompt_len = prompt_ids.len();
        // Before the lock and before the GPU: a prompt that cannot fit is a
        // refusal that costs nothing, and the budget keeps a long generation
        // from running off the end of the context in the middle of an answer.
        let max_tokens =
            crate::backend::generation_budget(prompt_len, params.max_tokens, self.context_tokens)?;
        // Before the lock too: a schema this vocabulary cannot hold is a 400.
        let mut constraint = match &params.json_schema {
            None => None,
            Some(schema) => {
                crate::stepped::refuse_stop(params)?;
                let table = self
                    .token_table()
                    .map_err(crate::backend::JsonSchemaUnsupported)?;
                Some(crate::constrain::Constraint::new(
                    schema,
                    table,
                    &self.eos_ids,
                ))
            }
        };
        let mut closed = false;
        let mut sampler = params.sampling.map(logits_processor);
        // A panic under this lock poisons it and skips the clear below. The
        // model is recovered and its cache cleared here: one bad request is
        // one 500, not every later one, and its KV never reaches the next.
        let mut model = match self.model.lock() {
            Ok(held) => held,
            Err(poisoned) => {
                let mut recovered = poisoned.into_inner();
                recovered.evict()?;
                // Left set, every later lock took this branch and evicted
                // the conversation kept: none continued until a restart
                // (read on 2026-10-31, a test holds it since).
                self.model.clear_poison();
                recovered
            }
        };
        if conversation.is_some() && !model.keeps_conversations() {
            return Err(crate::backend::ConversationUnsupported.into());
        }
        // SECURITY: whose cache this is, settled before anything runs. A turn
        // that presents the resident conversation's handle on its route keeps
        // the prefix its prompt shares with it; any other request finds the
        // cache zeroed (crate::conversation).
        let (reuse, number) = model.admit(conversation, &prompt_ids)?;
        // SECURITY: the KV cache is cleared on EVERY exit path, which
        // is why the loop lives in a closure whose `?` cannot skip the clear.
        // Before 2026-08-22 the clear sat after the loop and every `?` inside
        // it jumped over: a generation that failed mid-way left this request's
        // keys and values resident, and the next caller's prefill started at
        // index 0 against a non-empty cache. The receipt's
        // `kv_cache_shared_between_requests: false` would have been a false
        // statement on exactly the path nobody exercises.
        let mut emitted = String::new();
        let run = || -> anyhow::Result<Vec<u32>> {
            // Lock held, budget and lease past: the 200 may go before the prefill.
            if sink.started().is_break() {
                return Err(Cancelled.into());
            }
            let mut decoder = StreamDecoder::new(&self.tokenizer, &params.stop);
            let mut all_ids = prompt_ids.clone();
            let mut generated: Vec<u32> = Vec::new();
            // One token per step, greedy unless sampled; the prompt in chunks sized for this prompt.
            model.reserve(prompt_len + max_tokens);
            let chunk = self.chunk_for(prompt_len);
            let mut logits = None;
            model.last_prefill = prompt_len - reuse;
            for (i, part) in prompt_ids[reuse..].chunks(chunk).enumerate() {
                let input = Tensor::new(part, &self.device)?.unsqueeze(0)?;
                logits = Some(model.forward(&input, reuse + i * chunk)?);
            }
            let mut logits = logits.ok_or_else(|| anyhow::anyhow!("empty prompt"))?;
            for step in 0..max_tokens {
                if !sink.alive() {
                    return Err(Cancelled.into());
                }
                // forward narrows to the last position: logits are [1, 1, vocab]
                // (batch, seq=1, vocab) — flatten to [vocab] before argmax.
                let last = logits.flatten_all()?.to_dtype(DType::F32)?;
                let next = match (constraint.as_ref(), sampler.as_mut()) {
                    (None, None) => last.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?,
                    (None, Some(sampler)) => sampler.sample(&last)?,
                    // Masked on the host, then picked as `stepped` picks.
                    (Some(c), sampler) => {
                        let mut row = last.to_vec1::<f32>()?;
                        c.mask_logits(&mut row)?;
                        pick(&row, sampler)?
                    }
                };
                if self.eos_ids.contains(&next) {
                    break;
                }
                if let Some(c) = constraint.as_mut() {
                    c.advance(next)?;
                    closed = c.done();
                }
                generated.push(next);
                all_ids.push(next);
                let (piece, stop) = match decoder.push(next)? {
                    Held::Emit(piece) => (piece, false),
                    Held::Stop(piece) => (piece, true),
                };
                if !piece.is_empty() {
                    if sink.delta(&piece).is_break() {
                        return Err(Cancelled.into());
                    }
                    emitted.push_str(&piece);
                }
                if stop || closed || step + 1 == max_tokens {
                    break;
                }
                let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
                logits = model.forward(&input, all_ids.len() - 1)?;
            }
            Ok(generated)
        };
        // A panic inside the loop: the cache is zeroed here, before the panic
        // goes on, not when the next request takes the poisoned lock. Until
        // 2026-10-31 a lock nobody took kept the keys past the 300 s bound.
        let generated = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
            Ok(generated) => generated,
            Err(panic) => {
                let _ = model.evict();
                std::panic::resume_unwind(panic);
            }
        };
        // The answer checked before the cache is settled: a turn that fails
        // anywhere keeps nothing.
        let generated = generated.and_then(|ids| {
            let text = self
                .tokenizer
                .decode(&ids, true)
                .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
            let (text, stopped) = crate::backend::cut_at_stop(&text, &params.stop);
            // What was streamed must be what is signed: a decoder whose output is
            // not prefix-stable is an error here, never a receipt over other text.
            let Some(rest) = text.strip_prefix(emitted.as_str()) else {
                return Err(NotPrefixStable.into());
            };
            if constraint.as_ref().is_some_and(|c| !c.joins_to(&text)) {
                anyhow::bail!(
                    "json_schema: the decoded text is not the bytes the mask let through"
                );
            }
            if !rest.is_empty() && sink.delta(rest).is_break() {
                return Err(Cancelled.into());
            }
            Ok((ids, text, stopped))
        });
        // Unconditional: the `?` above cannot reach past this line. Kept for
        // the conversation's next turn, or zeroed.
        let kept = model.settle(
            conversation,
            number,
            &prompt_ids,
            generated.as_ref().ok().map(|(ids, ..)| ids),
        );
        let generated = generated?;
        if kept? && !model.reaping {
            let held = Arc::downgrade(&self.model);
            let spawned = std::thread::Builder::new()
                .name("x0ne-conversation".into())
                .spawn(move || reap(&held));
            model.reaping = spawned.is_ok();
            // No timer, nothing kept: 300 s is the bound the docs give.
            if spawned.is_err() {
                model.evict()?;
            }
        }
        if let Some(turn) = conversation {
            turn.record(Reused {
                turn: number,
                ids: prompt_ids[..reuse].to_vec(),
            });
        }
        let (generated, text, stopped) = generated;
        Ok(Completion {
            text,
            prompt_tokens: prompt_len as u32,
            completion_tokens: generated.len() as u32,
            stopped,
            // The budget, not params.max_tokens: the context window may have
            // cut it. A closed document was not cut.
            length: !closed && generated.len() >= max_tokens,
        })
    }

    fn model_digest_hex(&self) -> String {
        self.digest_hex.clone()
    }

    fn chat_prompt(&self, messages: &[crate::server::ChatMessage]) -> Prompt {
        self.arch.chat_prompt_with(messages, self.default_system)
    }

    /// Read from the tensors' own device, not from the `--device` argument:
    /// the receipt must say where the arithmetic HAPPENED. The two agree
    /// today; asking the argument would make them agree by construction,
    /// which is how a claim stops being a measurement.
    /// SECURITY: each `false` below is a property of the loop above,
    /// not a wish, for a request without a conversation.
    ///   - the KV cache is dropped on every exit path of `generate_turn`,
    ///     zeroed first on Qwen3 and Ministral 3 (OLMo's cache is dropped,
    ///     not zeroed); a turn of a conversation keeps it for the next
    ///     turn, and the server signs this flag `true` on it
    ///     (`server::Decoding::reused_context`);
    ///   - there is no prefix cache in this engine at all — nothing to share;
    ///   - `self.model` is behind a `Mutex` and `forward` is called with a
    ///     batch dimension of exactly 1, so no two callers ever share a pass;
    ///   - this crate contains no logging statement, in any module, that
    ///     takes prompt or completion text (a test greps for it).
    fn isolation(&self) -> crate::backend::Isolation {
        crate::backend::Isolation::NONE_OF_IT
    }

    fn serving_limits(&self) -> Option<(usize, usize)> {
        Some((self.context_tokens, self.prefill_chunk))
    }

    fn prefill_chunk_for(&self, prompt_tokens: usize) -> Option<usize> {
        Some(self.chunk_for(prompt_tokens))
    }

    fn compute_device(&self) -> String {
        // `location()`, not a `#[cfg(feature = "cuda")]` arm: that arm called
        // `CudaDevice::ordinal()`, which candle dropped, and since no machine
        // without the CUDA toolkit ever compiles the feature, nobody saw it
        // until the first build on an RTX 5090 (2026-09-19). This form is
        // compiled — and so kept honest — by every build, GPU or not.
        match self.device.location() {
            candle_core::DeviceLocation::Cpu => "cpu".into(),
            candle_core::DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
            candle_core::DeviceLocation::Metal { gpu_id } => format!("metal:{gpu_id}"),
        }
    }
}

impl InferenceBackend for CandleBackend {
    /// Named by the family it serves when that is Ministral 3; `candle-qwen3`
    /// otherwise, its name since v0.
    fn name(&self) -> &'static str {
        match Architecture::detect(&self.artifacts.config_json) {
            Ok(Architecture::Ministral3) => "candle-ministral3",
            _ => "candle-qwen3",
        }
    }

    fn load(
        &self,
        encrypted_weights: &[u8],
        key: &ReleasedKey,
    ) -> anyhow::Result<Box<dyn LoadedModel>> {
        // Sealed weights (`X0NESEAL` envelope) decrypt under the released
        // key — the release decision is load-bearing, not ceremonial. Bare
        // plaintext is still accepted, for one stated reason: public
        // open-weights models are legitimately unencrypted, and pretending
        // to decrypt them would be theatre. A CLIENT model must be sealed;
        // the operator tooling (`x0ne-infer seal`) exists for exactly that.
        //
        // Exactly the bytes given are loaded: no fallback to constructor
        // state (a caller passing empty bytes once silently loaded whatever
        // the backend held: input substitution). Empty is
        // an error, never a default.
        if encrypted_weights.is_empty() {
            anyhow::bail!("no weight bytes given to load()");
        }
        let unsealed;
        let bytes: &[u8] = if crate::seal::is_sealed(encrypted_weights) {
            unsealed = key.with_material(|m| -> anyhow::Result<_> {
                let m: &[u8; 32] = m
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("released key is not 32 bytes"))?;
                crate::seal::unseal(encrypted_weights, m)
            })?;
            &unsealed
        } else {
            encrypted_weights
        };
        self.load_bytes(bytes)
    }

    fn load_files(
        &self,
        weights: &WeightFiles<'_>,
        _key: &ReleasedKey,
    ) -> anyhow::Result<Box<dyn LoadedModel>> {
        if weights.files.iter().any(|(_, b)| crate::seal::is_sealed(b)) {
            anyhow::bail!("sealed weights are one file, opened through load()");
        }
        self.load_parts(weights)
    }
}

impl CandleBackend {
    fn load_bytes(&self, bytes: &[u8]) -> anyhow::Result<Box<dyn LoadedModel>> {
        self.load_parts(&single(bytes))
    }

    /// The digest covers everything that shapes behaviour: weights, config.json,
    /// tokenizer.json, the index. For a sealed model, the PLAINTEXT: one model,
    /// one digest, whatever key sealed it.
    fn load_parts(&self, weights: &WeightFiles<'_>) -> anyhow::Result<Box<dyn LoadedModel>> {
        if weights.files.is_empty() || weights.files.iter().any(|(_, b)| b.is_empty()) {
            anyhow::bail!("no weight bytes given to load()");
        }
        let a = &self.artifacts;
        let digest_hex = served_digest(&a.config_json, &a.tokenizer_json, weights)?;
        let spec = ModelSpec::parse(&self.artifacts.config_json)?.served_as(&digest_hex);
        eprintln!("{}", spec.default_system_line(&digest_hex));
        let tokenizer = Arc::new(crate::prompt::text_only(
            Tokenizer::from_bytes(&self.artifacts.tokenizer_json)
                .map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?,
        ));
        let slices = crate::candle_shards::ShardedSlices::new(weights.index_json, &weights.files)
            .map_err(|e| anyhow::anyhow!("safetensors: {e:#}"))?;
        spec.check_tensor_names(slices.names())?;
        let vb = VarBuilder::from_backend(Box::new(slices), self.dtype(), self.device.clone());
        let model = spec.build(vb, Projections::Fused)?;
        finish(model, tokenizer, digest_hex, &spec, &self.device)
    }
}

/// One `model.safetensors`, no index.
fn single(bytes: &[u8]) -> WeightFiles<'_> {
    WeightFiles {
        index_json: None,
        files: vec![(crate::shards::SINGLE_FILE, bytes)],
    }
}

/// EOS: `<|im_end|>` and `<|endoftext|>`, resolved from the tokenizer rather
/// than hardcoded ids — exactly the Qwen3 list since v0; `</s>` alone for
/// Ministral 3 (its tokenizer.json, id 2). Other families add config.json's
/// `eos_token_id`.
fn finish(
    model: ArchModel,
    tokenizer: Arc<Tokenizer>,
    digest_hex: String,
    spec: &ModelSpec,
    device: &Device,
) -> anyhow::Result<Box<dyn LoadedModel>> {
    // Weights are resident: what the GPU has left bounds the context, so a
    // long prompt is a 413 at the door, not an out-of-memory mid-generation.
    let dtype_bytes = if device.is_cpu() { 4 } else { 2 };
    // Frees of the build (fused projections) are stream-ordered: settle them first.
    device
        .synchronize()
        .map_err(|e| anyhow::anyhow!("device synchronize: {e}"))?;
    let gpu = gpu_memory(device)?;
    let (context_tokens, prefill_chunk) =
        serving_limits(&spec.decoder, dtype_bytes, gpu).map_err(|e| anyhow::anyhow!(e))?;
    let budget = serving_budget(gpu);
    if gpu.is_some() {
        let table = crate::backend::prefill_table(context_tokens, |p| {
            request_chunk(
                &spec.decoder,
                dtype_bytes,
                context_tokens,
                budget,
                prefill_chunk,
                p,
            )
        });
        eprintln!(
            "x0ne-infer: GPU serving limits: {context_tokens} tokens of context \
             (model: {}), prefill chunk {prefill_chunk}, by length {table:?}",
            spec.decoder.max_position_embeddings
        );
    }
    let special = crate::prompt::special_ids(&tokenizer);
    let missing: Vec<&str> = spec
        .arch
        .control_markers()
        .iter()
        .copied()
        .filter(|m| !special.contains_key(*m))
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "tokenizer.json: {} not a special added token: the {} template's markers \
             would be text a message can type",
            missing.join(", "),
            spec.arch.name()
        );
    }
    let eos_names: &[&str] = match spec.arch {
        Architecture::Ministral3 => &["</s>"],
        _ => &["<|im_end|>", "<|endoftext|>"],
    };
    let mut eos_ids: Vec<u32> = eos_names
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();
    if spec.arch != Architecture::Qwen3 {
        for id in &spec.decoder.eos_token_ids {
            if !eos_ids.contains(id) {
                eos_ids.push(*id);
            }
        }
    }
    // Ministral 3's config.json names no EOS (both releases, 2026-10-28):
    // `</s>`, a control marker, was required above.
    Ok(Box::new(Loaded {
        model: Slot::shared(model),
        arch: spec.arch,
        tokenizer,
        digest_hex,
        device: device.clone(),
        eos_ids,
        context_tokens,
        prefill_chunk,
        decoder: spec.decoder.clone(),
        dtype_bytes,
        budget,
        default_system: spec.default_system,
        token_table: std::sync::OnceLock::new(),
    }))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use candle_transformers::models::qwen3::ModelForCausalLM;

    /// Every prefill chunk keeps heads * q_len * kv_len under the softmax's i32 index.
    #[test]
    fn prefill_chunk_keeps_attention_under_i32() {
        for (heads, ctx) in [
            (16, 40960),
            (32, 40960),
            (64, 131072),
            (128, 1 << 20),
            (1, 1),
        ] {
            let c = prefill_chunk(heads, ctx);
            assert!(c >= 1);
            assert!(
                i32::try_from(heads * c * ctx).is_ok(),
                "{heads} heads, {ctx} ctx, chunk {c}"
            );
        }
        // Qwen3-4B: one prefill of 8193 tokens was past the bound (measured 2026-10-10).
        assert!(i32::try_from(32usize * 8193 * 8193).is_err());
        assert!(prefill_chunk(32, 40960) < 8193);
    }

    /// Qwen3-32B bf16 (config.json of Qwen/Qwen3-32B at 9216db57, sha256
    /// 97e295b6…83bb, fetched 2026-10-16) on an H100 counted as 80e9 bytes, under
    /// the 81 559 MiB an H100 80GB reports: 65 524 328 560 bytes of shards and a
    /// 512 MiB CUDA context leave about 13 GiB.
    #[test]
    fn qwen3_32b_fits_an_h100_with_its_context() {
        let d = DecoderConfig::parse(include_bytes!("../tests/arch/real/qwen3-32b.json")).unwrap();
        let total: u64 = 80_000_000_000;
        let free = total - 65_524_328_560 - (512 << 20);
        let (ctx, chunk) = serving_limits(&d, 2, Some((free, total))).unwrap();
        let peak = serving_peak_bytes(&d, 2, ctx, chunk);
        assert!(
            peak + gpu_reserve(total) <= free,
            "ctx {ctx}, chunk {chunk}: {peak} bytes + reserve > {free} free"
        );
        assert!(ctx >= 8192, "a useful context: {ctx}");
        assert!(i32::try_from(d.num_heads * chunk * ctx).is_ok());
        // The CPU keeps the model's own context.
        assert_eq!(serving_limits(&d, 2, None).unwrap().0, 40960);
    }

    /// The 18/10 run (validation-023, Qwen3-32B, 40 960 tokens of context) prefilled
    /// 3 921 tokens at 1 814 tok/s, about 40 passes of a chunk sized for the context. With
    /// 15.5 GB free, a chunk sized for the prompt is at least 900: 5 passes at most.
    #[test]
    fn a_3921_token_prompt_on_the_32b_gets_a_chunk_of_900_or_more() {
        let d = DecoderConfig::parse(include_bytes!("../tests/arch/real/qwen3-32b.json")).unwrap();
        let gpu = Some((15_500_000_000, 80_000_000_000));
        let (ctx, at_context) = serving_limits(&d, 2, gpu).unwrap();
        assert_eq!(ctx, 40960);
        let budget = serving_budget(gpu).unwrap();
        let c = request_chunk(&d, 2, ctx, Some(budget), at_context, 3921);
        assert!(
            c >= 900,
            "chunk {c} for 3921 tokens (at the context: {at_context})"
        );
        assert!(
            3921usize.div_ceil(c) <= 5,
            "{} passes",
            3921usize.div_ceil(c)
        );
        assert!(prefill_peak_bytes(&d, 2, ctx, 3921, c) <= budget);
        assert!(i32::try_from(d.num_heads * c * 3921).is_ok());
    }

    /// Over prompt lengths, cards and models: the chunk stays within the prompt, the
    /// softmax's i32 index and the budget, never falls under the chunk at the full
    /// context, and does not grow with the prompt. On the CPU only the i32 bound applies.
    #[test]
    fn the_request_chunk_keeps_every_bound() {
        let models: [&[u8]; 3] = [
            include_bytes!("../tests/arch/real/qwen3-8b.json"),
            include_bytes!("../tests/arch/real/qwen3-14b.json"),
            include_bytes!("../tests/arch/real/qwen3-32b.json"),
        ];
        for json in models {
            let d = DecoderConfig::parse(json).unwrap();
            for free in [6u64 << 30, 15_500_000_000, 40 << 30] {
                let gpu = Some((free, 80_000_000_000));
                let Ok((ctx, at_context)) = serving_limits(&d, 2, gpu) else {
                    continue;
                };
                let budget = serving_budget(gpu);
                let mut previous = usize::MAX;
                let lens = [1, 2, 63, 64, 900, 3921, 8192, 16349, ctx - 1, ctx];
                for p in lens.into_iter().filter(|&p| p <= ctx) {
                    let c = request_chunk(&d, 2, ctx, budget, at_context, p);
                    let case = format!(
                        "{} layers, {free} free, prompt {p}: chunk {c}",
                        d.num_layers
                    );
                    assert!(c >= 1 && c <= p, "{case}");
                    assert!(c >= at_context.min(p), "{case} < {at_context}");
                    assert!(c <= at_context.max(PASS_CAP), "{case} > {at_context}");
                    assert!(i32::try_from(d.num_heads * c * p).is_ok(), "{case}");
                    assert!(
                        prefill_peak_bytes(&d, 2, ctx, p, c) <= budget.unwrap(),
                        "{case}"
                    );
                    if c < p {
                        assert!(c <= previous, "{case} grew from {previous}");
                        previous = c;
                    }
                }
            }
            let (_, at_context) = serving_limits(&d, 4, None).unwrap();
            let cpu = request_chunk(&d, 4, 40960, None, at_context, 40959);
            assert_eq!(cpu, prefill_chunk(d.num_heads, 40959));
            assert_eq!(request_chunk(&d, 4, 40960, None, at_context, 100), 100);
        }
    }

    /// Where the chunk at the context is PASS_CAP or more (1063 for the 8B on the
    /// RTX 5090, 2026-10-20), every prompt is cut as at the context; below it (the 32B
    /// above), the chunk sized for the prompt grows to PASS_CAP at most.
    #[test]
    fn a_chunk_at_the_context_of_pass_cap_or_more_is_kept_at_every_length() {
        let models: [&[u8]; 3] = [
            include_bytes!("../tests/arch/real/qwen3-8b.json"),
            include_bytes!("../tests/arch/real/qwen3-14b.json"),
            include_bytes!("../tests/arch/real/qwen3-32b.json"),
        ];
        let (mut kept, mut grown) = (0, 0);
        for json in models {
            let d = DecoderConfig::parse(json).unwrap();
            for free in [6u64 << 30, 15_500_000_000, 17_000_000_000, 40 << 30] {
                let gpu = Some((free, 80_000_000_000));
                let Ok((ctx, at_context)) = serving_limits(&d, 2, gpu) else {
                    continue;
                };
                let budget = serving_budget(gpu);
                for p in [1, 900, 1063, 3827, 7926, 16120, ctx] {
                    let p = p.min(ctx);
                    let c = request_chunk(&d, 2, ctx, budget, at_context, p);
                    let case = format!(
                        "{} layers, {free} free, prompt {p}: chunk {c}",
                        d.num_layers
                    );
                    if at_context >= PASS_CAP {
                        assert_eq!(c, at_context.min(p), "{case}");
                        kept += 1;
                    } else {
                        assert!(c <= PASS_CAP && c >= at_context.min(p), "{case}");
                        grown += 1;
                    }
                }
            }
        }
        assert!(kept > 0 && grown > 0, "kept {kept}, grown {grown}");
    }

    /// Weights that leave less than a 1024-token context refuse the load, with numbers.
    #[test]
    fn a_gpu_too_small_for_any_context_refuses_with_numbers() {
        let d = DecoderConfig::parse(include_bytes!("../tests/arch/real/qwen3-32b.json")).unwrap();
        let err = serving_limits(&d, 2, Some((1 << 30, 80 << 30))).unwrap_err();
        assert!(err.contains("GiB free"), "{err}");
    }

    fn backend() -> CandleBackend {
        CandleBackend::cpu(ModelArtifacts {
            config_json: include_bytes!("../tests/arch/qwen3/config.json").to_vec(),
            tokenizer_json: b"{}".to_vec(),
        })
    }

    /// SECURITY: the wiring proof for the envelope. Sealed bytes given to
    /// `load` with the RIGHT key must fail PAST the seal (in safetensors
    /// parsing — proof the unseal ran and succeeded); with the WRONG key the
    /// refusal must name authentication and never reach the parser. A
    /// failure here would mean the release decision is ceremonial again —
    /// the exact gap this envelope exists to close.
    #[test]
    fn sealed_weights_are_opened_by_the_released_key_and_refused_by_any_other() {
        let sealed = crate::seal::seal(b"not a real model", &[7u8; 32], &[9u8; 12]).unwrap();

        let right = crate::keys::test_released_key(vec![7u8; 32]);
        let err = match backend().load(&sealed, &right) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("garbage plaintext must not build a model"),
        };
        assert!(
            !err.contains("authentication"),
            "the unseal must succeed under the released key: {err}"
        );

        let wrong = crate::keys::test_released_key(vec![8u8; 32]);
        let err = match backend().load(&sealed, &wrong) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a wrong key must refuse"),
        };
        assert!(
            err.contains("authentication failed"),
            "a wrong key must be refused BY THE SEAL, got: {err}"
        );
    }

    /// Both places a `config.json` may keep `rope_theta`. A failure means the
    /// engine refuses a model whose config a current `transformers` wrote —
    /// the load died with "missing field `rope_theta`" on 2026-09-19, and the
    /// workaround was to paste another model's config over it.
    #[test]
    fn rope_theta_is_read_whether_it_is_nested_or_not() {
        let nested = include_str!("../tests/qwen3_config_rope_parameters.json");
        let cfg = parse_qwen3_config(nested.as_bytes()).expect("nested rope_parameters");
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        assert_eq!(cfg.max_position_embeddings, 40_960);

        // The flat layout (transformers 4.51, the Qwen3-0.6B release) is
        // untouched, and an explicit top-level value wins over the nested one.
        let flat = nested.replace(
            r#""rope_parameters": {
    "rope_theta": 1000000,
    "rope_type": "default"
  }"#,
            r#""rope_theta": 5000"#,
        );
        assert_ne!(flat, nested, "the substitution must have applied");
        assert_eq!(
            parse_qwen3_config(flat.as_bytes())
                .expect("flat")
                .rope_theta,
            5000.0
        );
        let both = nested.replace(
            r#""rope_parameters": {"#,
            r#""rope_theta": 7000, "rope_parameters": {"#,
        );
        assert_eq!(
            parse_qwen3_config(both.as_bytes())
                .expect("both")
                .rope_theta,
            7000.0
        );

        // Neither layout: still a refusal, and it still names the field.
        let none = nested.replace("rope_theta", "rope_thetaa");
        let err = parse_qwen3_config(none.as_bytes()).unwrap_err().to_string();
        assert!(err.contains("rope_theta"), "{err}");
    }

    /// Engine's Qwen3 config reader is candle's: on real configs and on each
    /// way to break one, both accept with equal values or both refuse. A
    /// failure means `model_spec::Qwen3Config` drifted from candle's `Config`.
    #[test]
    fn the_qwen3_config_reader_is_candles() {
        // candle's reader, with the `rope_theta` lift Engine always applied.
        let candle = |json: &serde_json::Value| -> Option<Config> {
            let mut v = json.clone();
            if let Some(o) = v.as_object_mut() {
                if !o.contains_key("rope_theta") {
                    if let Some(t) = o
                        .get("rope_parameters")
                        .and_then(|p| p.get("rope_theta"))
                        .cloned()
                    {
                        o.insert("rope_theta".into(), t);
                    }
                }
            }
            serde_json::from_value(v).ok()
        };
        let ours = |json: &serde_json::Value| -> Option<Config> {
            let c =
                crate::model_spec::parse_qwen3_config(&serde_json::to_vec(json).unwrap()).ok()?;
            Some(to_candle_qwen3(&c).expect("an accepted config converts"))
        };
        let mut cases: Vec<serde_json::Value> = [
            &include_bytes!("../tests/arch/real/qwen3-8b.json")[..],
            &include_bytes!("../tests/arch/real/qwen3-14b.json")[..],
            &include_bytes!("../tests/arch/real/qwen3-32b.json")[..],
            &include_bytes!("../tests/qwen3_config_rope_parameters.json")[..],
            &include_bytes!("../tests/arch/qwen3/config.json")[..],
        ]
        .iter()
        .map(|b| serde_json::from_slice(b).unwrap())
        .collect();
        let base = cases[0].clone();
        for key in base.as_object().unwrap().keys() {
            let mut v = base.clone();
            v.as_object_mut().unwrap().remove(key);
            cases.push(v);
            let mut v = base.clone();
            v[key] = serde_json::json!("not a value");
            cases.push(v);
        }
        for act in [
            serde_json::json!("silu"),
            serde_json::json!("gelu"),
            serde_json::json!("gelu_new"),
            serde_json::json!("newgelu"),
            serde_json::json!("gelu_pytorch_tanh"),
            serde_json::json!("swiglu"),
            serde_json::json!("hardswish"),
            serde_json::json!("SiLU"),
            serde_json::json!("bogus"),
            serde_json::json!({"elu": 1.5}),
            serde_json::json!({"leakyrelu": 0.01}),
            serde_json::json!(null),
        ] {
            let mut v = base.clone();
            v["hidden_act"] = act;
            cases.push(v);
        }
        for (key, value) in [
            ("sliding_window", serde_json::json!(null)),
            ("sliding_window", serde_json::json!(4096)),
            ("rope_theta", serde_json::json!(5e5)),
            ("rms_norm_eps", serde_json::json!(1e-6)),
            ("vocab_size", serde_json::json!(-1)),
            ("unknown_key", serde_json::json!(1)),
        ] {
            let mut v = base.clone();
            v[key] = value;
            cases.push(v);
        }
        let mut accepted = 0;
        for (i, v) in cases.iter().enumerate() {
            let (a, b) = (ours(v), candle(v));
            assert_eq!(a, b, "case {i}: {v}");
            accepted += a.is_some() as usize;
        }
        assert!(
            accepted >= 12 && accepted < cases.len(),
            "{accepted} of {}",
            cases.len()
        );
    }

    /// Plaintext bytes still load without the envelope (public open-weights
    /// models are legitimately unencrypted) — the failure must come from the
    /// parser, never from a seal that was not there.
    #[test]
    fn plaintext_weights_still_take_the_direct_path() {
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        let err = match backend().load(b"plaintext bytes", &key) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("garbage plaintext must not build a model"),
        };
        assert!(!err.contains("authentication"), "{err}");
        assert!(!err.contains("sealed"), "{err}");
    }

    /// Every family's template markers, once each.
    pub(crate) fn all_markers() -> Vec<&'static str> {
        let mut all: Vec<&str> = Vec::new();
        for a in [
            Architecture::Qwen3,
            Architecture::Olmo2,
            Architecture::Olmo3,
            Architecture::Ministral3,
        ] {
            for m in a.control_markers() {
                if !all.contains(m) {
                    all.push(m);
                }
            }
        }
        all
    }

    /// A word-level tokenizer over `t0 … t{n-1}`: the tiny fixtures' vocabulary.
    /// Every family's template markers are special added tokens from id `n`,
    /// past the fixtures' vocabulary: a word's id and an answer's text are as
    /// before they were added (2026-10-28), and no answer ends on one.
    pub(crate) fn word_tokenizer(n: usize) -> Vec<u8> {
        let vocab: serde_json::Map<String, serde_json::Value> =
            (0..n).map(|i| (format!("t{i}"), i.into())).collect();
        let added: Vec<serde_json::Value> = all_markers()
            .iter()
            .enumerate()
            .map(|(i, m)| special_token(n + i, m))
            .collect();
        serde_json::to_vec(&serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
            "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"},
        }))
        .unwrap()
    }

    pub(crate) fn special_token(id: usize, content: &str) -> serde_json::Value {
        serde_json::json!({"id": id, "content": content, "single_word": false,
            "lstrip": false, "rstrip": false, "normalized": false, "special": true})
    }

    /// `tokenizer` with the added token `marker` dropped, or made a plain one
    /// (`special: false`).
    pub(crate) fn without_special(tokenizer: &[u8], marker: &str, keep_plain: bool) -> Vec<u8> {
        let mut t: serde_json::Value = serde_json::from_slice(tokenizer).unwrap();
        let added = t["added_tokens"].as_array_mut().unwrap();
        if keep_plain {
            for a in added.iter_mut().filter(|a| a["content"] == marker) {
                a["special"] = false.into();
            }
        } else {
            added.retain(|a| a["content"] != marker);
        }
        serde_json::to_vec(&t).unwrap()
    }

    /// A template marker that is not a special added token is refused at
    /// load, in every family: a message could type it and write a turn. Both
    /// forms: absent, and an added token with `special: false`.
    #[test]
    fn a_template_marker_that_is_not_a_special_token_is_refused() {
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        for family in ["qwen3", "olmo2", "olmo3", "ministral3"] {
            let (config, weights, _) = fixture(family);
            let arch = ModelSpec::parse(&config).unwrap().arch;
            let load = |tokenizer: Vec<u8>| {
                CandleBackend::cpu(ModelArtifacts {
                    config_json: config.clone(),
                    tokenizer_json: tokenizer,
                })
                .load(&weights, &key)
            };
            assert!(load(word_tokenizer(40)).is_ok(), "{family}");
            for marker in arch.control_markers() {
                for plain in [false, true] {
                    let err = match load(without_special(&word_tokenizer(40), marker, plain)) {
                        Err(e) => format!("{e:#}"),
                        Ok(_) => panic!("{family}: loaded without a special {marker}"),
                    };
                    assert!(err.contains(marker) && err.contains("special"), "{err}");
                }
            }
        }
    }

    /// Ministral 3's six markers as vocabulary words a Split isolates: a
    /// message typing them gave the ids of a real three-turn conversation
    /// (measured 2026-10-28). Such a tokenizer is refused.
    #[test]
    fn a_tokenizer_whose_markers_are_plain_words_is_refused() {
        let list = [
            "t0",
            "<s>",
            "</s>",
            "[INST]",
            "[/INST]",
            "[SYSTEM_PROMPT]",
            "[/SYSTEM_PROMPT]",
            "hi",
        ];
        let vocab: serde_json::Map<String, serde_json::Value> = list
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), i.into()))
            .collect();
        let plain = serde_json::to_vec(&serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Split",
                "pattern": {"Regex": "\\[/?INST\\]|</?s>|\\[/?SYSTEM_PROMPT\\]"},
                "behavior": "Isolated", "invert": false},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"},
        }))
        .unwrap();
        let (config, weights, _) = fixture("ministral3");
        let err = match CandleBackend::cpu(ModelArtifacts {
            config_json: config,
            tokenizer_json: plain,
        })
        .load(&weights, &crate::keys::test_released_key(vec![7u8; 32]))
        {
            Err(e) => format!("{e:#}"),
            Ok(_) => panic!("loaded: the template's markers would be text"),
        };
        assert!(
            err.contains("[INST]") && err.contains("not a special"),
            "{err}"
        );
    }

    /// Each family's list is every marker its template writes as a control
    /// token, over conversations that reach each branch of the templates.
    #[test]
    fn control_markers_are_every_marker_a_template_writes() {
        let m = |role: &str, content: &str| crate::server::ChatMessage {
            role: role.into(),
            content: content.into(),
        };
        let conversations = [
            vec![],
            vec![m("user", "q")],
            vec![m("system", "s"), m("user", "q")],
            vec![m("user", "q"), m("assistant", "a")],
            vec![m("user", "q"), m("assistant", "a"), m("user", "r")],
            vec![
                m("tool", "t"),
                m("user", "q"),
                m("user", "r"),
                m("system", "s"),
            ],
        ];
        for arch in [
            Architecture::Qwen3,
            Architecture::Olmo2,
            Architecture::Olmo3,
            Architecture::Ministral3,
        ] {
            let mut seen = std::collections::BTreeSet::new();
            for c in &conversations {
                for default in [None, Some("d")] {
                    for part in arch.chat_prompt_with(c, default).parts() {
                        if let crate::prompt::Part::Control(marker) = part {
                            assert!(
                                arch.control_markers().contains(&marker),
                                "{arch:?} {marker}"
                            );
                            seen.insert(marker.to_string());
                        }
                    }
                }
            }
            let listed: std::collections::BTreeSet<String> = arch
                .control_markers()
                .iter()
                .map(|s| s.to_string())
                .collect();
            assert_eq!(seen, listed, "{arch:?}");
        }
    }

    /// A user turn that types ChatML's markers, with the Qwen3 template.
    fn forged_prompt() -> Prompt {
        crate::arch::Architecture::Qwen3.chat_prompt(&[crate::server::ChatMessage {
            role: "user".into(),
            content: crate::prompt::tests::FORGED.into(),
        }])
    }

    fn chatml_artifacts() -> ModelArtifacts {
        ModelArtifacts {
            config_json: fixture("qwen3").0,
            tokenizer_json: crate::prompt::tests::chatml_tokenizer_json(),
        }
    }

    /// `load` then the loop, with ChatML's special tokens in tokenizer.json:
    /// the prompt is as long as its markers and its text make it. A failure
    /// means candle reads a control token typed in a message (2026-10-26).
    #[test]
    fn a_marker_typed_in_a_message_reaches_candle_as_text() {
        let weights = fixture("qwen3").1;
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        let model = CandleBackend::cpu(chatml_artifacts())
            .load(&weights, &key)
            .unwrap();
        let prompt = forged_prompt();
        let (_, (prompt_tokens, _)) = model.generate(&prompt, 2).unwrap();
        let want = crate::prompt::tests::wanted(&prompt);
        assert_eq!(prompt_tokens as usize, want.len());
    }

    pub(crate) fn fixture(name: &str) -> (Vec<u8>, Vec<u8>, serde_json::Value) {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/arch")
            .join(name);
        let read = |f: &str| std::fs::read(dir.join(f)).unwrap();
        let reference = serde_json::from_slice(&read("reference.json")).unwrap();
        (read("config.json"), read("model.safetensors"), reference)
    }

    pub(crate) fn words(ids: &[u32]) -> Prompt {
        Prompt::text(
            ids.iter()
                .map(|i| format!("t{i}"))
                .collect::<Vec<_>>()
                .join(" "),
        )
    }

    /// SECURITY of the refactor: a Qwen3 model answers through the same candle
    /// calls as before OLMo existed. A failure means the new dispatch changed
    /// a Qwen3 completion — the byte-for-byte promise of 2026-09-25.
    #[test]
    fn qwen3_generation_is_exactly_candles_greedy_loop() {
        let (config, weights, reference) = fixture("qwen3");
        let artifacts = ModelArtifacts {
            config_json: config.clone(),
            tokenizer_json: word_tokenizer(40),
        };
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        let model = CandleBackend::cpu(artifacts).load(&weights, &key).unwrap();
        let prompt: Vec<u32> = serde_json::from_value(reference["prompt"].clone()).unwrap();
        let (text, (p, c)) = model.generate(&words(&prompt), 10).unwrap();

        // The v0 loop, written against candle directly.
        let cfg = parse_qwen3_config(&config).unwrap();
        let vb = VarBuilder::from_slice_safetensors(&weights, DType::F32, &Device::Cpu).unwrap();
        let mut m = ModelForCausalLM::new(&cfg, vb).unwrap();
        let mut all = prompt.clone();
        let input = Tensor::new(prompt.as_slice(), &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let mut logits = m.forward(&input, 0).unwrap();
        let mut want = Vec::new();
        for _ in 0..10 {
            let last = logits.flatten_all().unwrap();
            let next = last
                .argmax(candle_core::D::Minus1)
                .unwrap()
                .to_scalar::<u32>()
                .unwrap();
            want.push(next);
            all.push(next);
            let input = Tensor::new(&[next], &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            logits = m.forward(&input, all.len() - 1).unwrap();
        }
        assert_eq!(text, words(&want).as_str());
        assert_eq!((p, c), (prompt.len() as u32, 10));
        assert_eq!(model.chat_prompt(&[]), crate::server::chat_prompt(&[]));
    }

    /// `weights` cut in two shards (tensors alternated by sorted name) and their index.
    fn split(weights: &[u8]) -> (Vec<u8>, Vec<(String, Vec<u8>)>) {
        let st = safetensors::SafeTensors::deserialize(weights).unwrap();
        let mut names: Vec<String> = st.names().into_iter().map(str::to_string).collect();
        names.sort();
        let mut map = serde_json::Map::new();
        let shards: Vec<(String, Vec<u8>)> = (0..2)
            .map(|k| {
                let file = format!("model-0000{}-of-00002.safetensors", k + 1);
                let mine: Vec<(String, safetensors::tensor::TensorView<'_>)> = names
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == k)
                    .map(|(_, n)| (n.clone(), st.tensor(n).unwrap()))
                    .collect();
                for (n, _) in &mine {
                    map.insert(n.clone(), file.clone().into());
                }
                (file, safetensors::serialize(mine, None).unwrap())
            })
            .collect();
        let index = serde_json::to_vec(&serde_json::json!({ "weight_map": map })).unwrap();
        (index, shards)
    }

    fn files<'a>(index: &'a [u8], shards: &'a [(String, Vec<u8>)]) -> WeightFiles<'a> {
        WeightFiles {
            index_json: Some(index),
            files: shards.iter().map(|(n, b)| (n.as_str(), &b[..])).collect(),
        }
    }

    /// The same tensors served from two shards answer token for token what the
    /// single file answers, in either file order, under one digest per file set.
    #[test]
    fn shards_generate_exactly_what_the_single_file_generates() {
        let (config, weights, reference) = fixture("qwen3");
        let backend = || {
            CandleBackend::cpu(ModelArtifacts {
                config_json: config.clone(),
                tokenizer_json: word_tokenizer(40),
            })
        };
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        let prompt: Vec<u32> = serde_json::from_value(reference["prompt"].clone()).unwrap();
        let single = backend().load(&weights, &key).unwrap();
        let want = single.generate(&words(&prompt), 10).unwrap();

        let (index, shards) = split(&weights);
        let mut reversed = shards.clone();
        reversed.reverse();
        let mut digests = Vec::new();
        for s in [&shards, &reversed] {
            let w = files(&index, s);
            let model = backend().load_files(&w, &key).unwrap();
            assert_eq!(model.generate(&words(&prompt), 10).unwrap(), want);
            let tok = word_tokenizer(40);
            let mut parts: Vec<(&str, &[u8])> = vec![
                ("config.json", &config),
                ("tokenizer.json", &tok),
                ("model.safetensors.index.json", &index),
            ];
            parts.extend(w.files.iter().copied());
            assert_eq!(
                model.model_digest_hex(),
                crate::model_digest::of_contents(&parts).unwrap()
            );
            digests.push(model.model_digest_hex());
        }
        assert_eq!(
            digests[0], digests[1],
            "file order is not part of the digest"
        );
        assert_ne!(
            digests[0],
            single.model_digest_hex(),
            "another file set, another digest"
        );
    }

    /// SECURITY: a shard with other bytes under the same name is another model:
    /// another digest, which the TD (77) and the client then refuse.
    #[test]
    fn a_substituted_shard_changes_the_digest_and_a_sealed_one_is_refused() {
        let (config, weights, _) = fixture("qwen3");
        let backend = CandleBackend::cpu(ModelArtifacts {
            config_json: config,
            tokenizer_json: word_tokenizer(40),
        });
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        let (index, shards) = split(&weights);
        let honest = backend.load_files(&files(&index, &shards), &key).unwrap();

        let mut other = shards.clone();
        let st = safetensors::SafeTensors::deserialize(&other[1].1).unwrap();
        let mut tensors: Vec<(String, Vec<u8>, safetensors::Dtype, Vec<usize>)> = st
            .tensors()
            .into_iter()
            .map(|(n, v)| (n, v.data().to_vec(), v.dtype(), v.shape().to_vec()))
            .collect();
        tensors[0].1[0] ^= 1;
        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(n, d, t, sh)| {
                (
                    n.clone(),
                    safetensors::tensor::TensorView::new(*t, sh.clone(), d).unwrap(),
                )
            })
            .collect();
        other[1].1 = safetensors::serialize(views, None).unwrap();
        let swapped = backend.load_files(&files(&index, &other), &key).unwrap();
        assert_ne!(swapped.model_digest_hex(), honest.model_digest_hex());

        let mut sealed = shards.clone();
        sealed[0].1 = crate::seal::seal(&sealed[0].1, &[7u8; 32], &[9u8; 12]).unwrap();
        let err = match backend.load_files(&files(&index, &sealed), &key) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a sealed shard must be refused"),
        };
        assert!(err.contains("sealed"), "{err}");
    }

    /// The whole serving path — detection, tokenizer, decoder, KV cache,
    /// sliding window — against transformers' greedy `generate` on OLMo 2/3.
    #[test]
    fn olmo_families_generate_what_transformers_generates() {
        for name in ["olmo2", "olmo3"] {
            let (config, weights, reference) = fixture(name);
            let artifacts = ModelArtifacts {
                config_json: config,
                tokenizer_json: word_tokenizer(40),
            };
            let key = crate::keys::test_released_key(vec![7u8; 32]);
            let model = CandleBackend::cpu(artifacts).load(&weights, &key).unwrap();
            let prompt: Vec<u32> = serde_json::from_value(reference["prompt"].clone()).unwrap();
            let want: Vec<u32> = serde_json::from_value(reference["generated"].clone()).unwrap();
            let (text, _) = model.generate(&words(&prompt), want.len()).unwrap();
            assert_eq!(text, words(&want).as_str(), "{name}");
            // KV cache cleared: the same request answers the same way twice.
            assert_eq!(model.generate(&words(&prompt), want.len()).unwrap().0, text);
            // A cache that grows by concatenation keeps no conversation: refused by name.
            let asked = crate::context::AskedConversation::New { end: false };
            let turn = Turn::admit(crate::conversation::Route::Chat, &asked, || [1; 32]).unwrap();
            let params = GenerationParams::greedy(1);
            let e = model
                .generate_turn(&words(&prompt), &params, Some(&turn), &mut NoSink)
                .unwrap_err();
            assert!(
                e.downcast_ref::<crate::backend::ConversationUnsupported>()
                    .is_some(),
                "{name}: {e:#}"
            );
        }
    }

    /// The qwen3 fixture as served, and the words of its greedy answer.
    fn fixture_model() -> (Box<dyn LoadedModel>, Prompt) {
        let (config, weights, reference) = fixture("qwen3");
        let artifacts = ModelArtifacts {
            config_json: config,
            tokenizer_json: word_tokenizer(40),
        };
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        let model = CandleBackend::cpu(artifacts).load(&weights, &key).unwrap();
        let prompt: Vec<u32> = serde_json::from_value(reference["prompt"].clone()).unwrap();
        (model, words(&prompt))
    }

    /// A WordLevel vocabulary of JSON pieces, fused on decode: the qwen3
    /// fixture's 40 ids, then every family's markers as special tokens.
    pub(crate) fn json_tokenizer() -> Vec<u8> {
        let pieces = [
            "{", "}", "[", "]", ":", ",", "\"", "\"ok\"", "\"n\"", "\"tags\"", "true", "false",
            "null", "0", "1", "2", "10", "a", "b", "ab", "\"a\"", "\"b\"", "}}", "],", "e", "\\",
            "-", ".", "x", "y", "ok", "n", "tags", "{\"", "\":", "<unk>", " ", "\n", "zz", "\",",
        ];
        let vocab: serde_json::Map<String, serde_json::Value> = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| ((*p).to_string(), i.into()))
            .collect();
        assert_eq!(vocab.len(), 40);
        let added: Vec<serde_json::Value> = all_markers()
            .iter()
            .enumerate()
            .map(|(i, m)| special_token(40 + i, m))
            .collect();
        serde_json::to_vec(&serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
            "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": null, "decoder": {"type": "Fuse"},
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
        }))
        .unwrap()
    }

    /// SECURITY: under a schema a real model writes a document of it, greedy
    /// or sampled, whole or streamed; cut by its budget, a start with `length`.
    /// Without the schema it writes none. A failure: a forbidden token picked.
    #[test]
    fn a_json_schema_holds_every_completion_of_a_real_model() {
        use crate::json_schema::{Accepted, JsonSchema};
        let (config, weights, _) = fixture("qwen3");
        let artifacts = ModelArtifacts {
            config_json: config,
            tokenizer_json: json_tokenizer(),
        };
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        let model = CandleBackend::cpu(artifacts).load(&weights, &key).unwrap();
        let raw = serde_json::json!({"type": "object", "properties": {
            "ok": {"type": "boolean"}, "n": {"enum": [0, 1, 2, 10]},
            "tags": {"type": "array", "items": {"enum": ["a", "b"]}, "maxItems": 3}},
            "required": ["ok", "n"]});
        let schema = JsonSchema::compile(&raw).unwrap();
        let prompt = Prompt::text("x y");
        let with = |sampling: Option<Sampling>, max_tokens: usize| GenerationParams {
            max_tokens,
            sampling,
            stop: vec![],
            json_schema: Some(schema.clone()),
        };
        let mut seen = std::collections::BTreeSet::new();
        for (t, seed) in [(0.0, 0), (1.0, 1), (1.0, 2), (1.5, 3), (1.5, 4), (2.0, 5)] {
            let sampling = (t > 0.0).then_some(Sampling {
                temperature: t,
                top_p: 1.0,
                seed,
            });
            let c = model.generate_with(&prompt, &with(sampling, 40)).unwrap();
            assert_eq!(
                schema.accepts(&c.text),
                Accepted::Complete,
                "({t}, {seed}): {c:?}"
            );
            let v: serde_json::Value = serde_json::from_str(&c.text).unwrap();
            assert!(v["ok"].is_boolean() && v["n"].is_u64(), "{}", c.text);
            assert!(!c.length, "a closed document is not cut: {c:?}");
            let mut sink = Collect::default();
            let streamed = model
                .generate_streaming(&prompt, &with(sampling, 40), &mut sink)
                .unwrap();
            assert_eq!(
                (sink.deltas.concat(), &streamed.text),
                (c.text.clone(), &c.text)
            );
            seen.insert(c.text);
        }
        assert!(
            seen.len() > 1,
            "sampling still samples under a mask: {seen:?}"
        );
        let free = model
            .generate_with(&prompt, &GenerationParams::greedy(40))
            .unwrap();
        assert_ne!(
            schema.accepts(&free.text),
            Accepted::Complete,
            "the model writes this document on its own; the test proves nothing: {free:?}"
        );
        let cut = model.generate_with(&prompt, &with(None, 3)).unwrap();
        assert!(cut.length, "{cut:?}");
        assert_eq!(schema.accepts(&cut.text), Accepted::Prefix, "{cut:?}");
    }

    /// A tokenizer whose decoder writes a token by its neighbours (here none:
    /// words joined by spaces) refuses a schema before the model runs.
    #[test]
    fn a_schema_on_a_tokenizer_that_reads_tokens_in_context_is_refused() {
        let (model, prompt) = fixture_model();
        let mut p = GenerationParams::greedy(4);
        p.json_schema = Some(
            crate::json_schema::JsonSchema::compile(&serde_json::json!({"type": "null"})).unwrap(),
        );
        let e = model.generate_with(&prompt, &p).unwrap_err();
        let refused = e
            .downcast_ref::<crate::backend::JsonSchemaUnsupported>()
            .unwrap_or_else(|| panic!("{e}"));
        assert!(refused.to_string().contains("no decoder"), "{refused}");
    }

    /// A seed repeats its draw, a request's draw does not reach the next, and
    /// `top_p` 0 or a nucleus below the top token's mass is argmax. A failure
    /// means sampling is ignored, unseeded, or leaks across requests.
    #[test]
    fn sampling_is_seeded_per_request_and_a_null_nucleus_is_greedy() {
        let (model, prompt) = fixture_model();
        let greedy = model.generate(&prompt, 10).unwrap();
        let g = model
            .generate_with(&prompt, &GenerationParams::greedy(10))
            .unwrap();
        assert_eq!(
            (g.text.clone(), (g.prompt_tokens, g.completion_tokens)),
            greedy
        );
        assert!(!g.stopped);
        let sampled = |temperature: f64, top_p: f64, seed: u64| {
            let mut p = GenerationParams::greedy(10);
            p.sampling = Some(Sampling {
                temperature,
                top_p,
                seed,
            });
            model.generate_with(&prompt, &p).unwrap().text
        };
        let draws: Vec<String> = (0..6).map(|seed| sampled(1.5, 1.0, seed)).collect();
        for (seed, draw) in draws.iter().enumerate() {
            assert_eq!(*draw, sampled(1.5, 1.0, seed as u64), "seed {seed} repeats");
        }
        assert!(
            draws.iter().any(|d| *d != greedy.0),
            "temperature 1.5 changed nothing"
        );
        assert_eq!(
            model.generate(&prompt, 10).unwrap(),
            greedy,
            "no draw reached the next request"
        );
        assert_eq!(sampled(1.5, 0.0, 3), greedy.0, "top_p 0 is the top token");
        assert_eq!(
            sampled(1.5, 1e-9, 3),
            greedy.0,
            "a tiny nucleus is the top token"
        );
    }

    /// A stop sequence ENDS the loop, not only the text: fewer tokens are paid
    /// for. A failure means the generation runs to the ceiling and is cut after.
    #[test]
    fn a_stop_sequence_ends_the_generation_loop() {
        let (model, prompt) = fixture_model();
        let (text, _) = model.generate(&prompt, 10).unwrap();
        let words: Vec<&str> = text.split(' ').collect();
        assert_eq!(words.len(), 10, "{text}");
        // The first word from the third on whose first occurrence is its own.
        let (k, at) = (2..words.len())
            .map(|k| (k, words[..k].iter().map(|w| w.len() + 1).sum::<usize>()))
            .find(|(k, at)| text.find(words[*k]) == Some(*at))
            .expect("a word not met earlier in the text");
        let mut p = GenerationParams::greedy(10);
        p.stop = vec!["zzz".into(), words[k].into()];
        let c = model.generate_with(&prompt, &p).unwrap();
        assert_eq!(c.text, text[..at], "cut before {}", words[k]);
        assert!(c.stopped);
        assert_eq!(
            c.completion_tokens as usize,
            k + 1,
            "stopped at the token that completed it"
        );
        assert_eq!(c.finish_reason(10), "stop");
    }

    /// A panic under the model lock poisons it. Before
    /// 2026-10-06 the next `generate` died on `expect("model lock")`, so one
    /// bad request made every later one a 500. The keys and values the
    /// panicking request left in the cache must not reach the next caller.
    #[test]
    fn a_panic_under_the_model_lock_leaves_the_model_serving_with_a_clean_cache() {
        let (config, weights, reference) = fixture("qwen3");
        let spec = ModelSpec::parse(&config).unwrap();
        let vb = VarBuilder::from_slice_safetensors(&weights, DType::F32, &Device::Cpu).unwrap();
        let loaded = Loaded {
            model: Slot::shared(spec.build(vb, Projections::Fused).unwrap()),
            arch: spec.arch,
            tokenizer: Arc::new(crate::prompt::text_only(
                Tokenizer::from_bytes(word_tokenizer(40)).unwrap(),
            )),
            digest_hex: String::new(),
            device: Device::Cpu,
            eos_ids: vec![],
            context_tokens: spec.decoder.max_position_embeddings,
            prefill_chunk: prefill_chunk(
                spec.decoder.num_heads,
                spec.decoder.max_position_embeddings,
            ),
            decoder: spec.decoder.clone(),
            dtype_bytes: 4,
            budget: None,
            default_system: None,
            token_table: std::sync::OnceLock::new(),
        };
        let prompt: Vec<u32> = serde_json::from_value(reference["prompt"].clone()).unwrap();
        let before = loaded.generate(&words(&prompt), 10).unwrap();

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut m = loaded.model.lock().unwrap();
            let input = Tensor::new(&[1u32, 2, 3][..], &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            m.forward(&input, 0).unwrap();
            panic!("mid-generation");
        }));
        assert!(panicked.is_err() && loaded.model.is_poisoned());

        let after = loaded
            .generate(&words(&prompt), 10)
            .expect("the model serves after a poisoning panic");
        assert_eq!(after, before, "the panicking request's cache was cleared");
        // The poison is cleared with the cache: left, every later lock took
        // the recovery branch and evicted the conversation kept, so none
        // continued until a restart.
        assert!(!loaded.model.is_poisoned(), "the poison is cleared");
        let asked = crate::context::AskedConversation::New { end: false };
        let first = Turn::admit(crate::conversation::Route::Chat, &asked, || [1; 32]).unwrap();
        let greedy = GenerationParams::greedy(3);
        loaded
            .generate_turn(&words(&prompt), &greedy, Some(&first), &mut NoSink)
            .unwrap();
        let asked = crate::context::AskedConversation::Continue {
            handle: crate::context::Handle::from_bytes([1; 32]),
            end: false,
        };
        let second = Turn::admit(crate::conversation::Route::Chat, &asked, || [2; 32]).unwrap();
        let mut longer = prompt.clone();
        longer.extend([30, 31]);
        loaded
            .generate_turn(&words(&longer), &greedy, Some(&second), &mut NoSink)
            .unwrap();
        assert_eq!(
            second.reused().unwrap().turn,
            2,
            "a conversation kept after the recovery"
        );
    }

    /// A context window that leaves 3 tokens cuts a 10-token request at 3,
    /// and that is "length". Until 2026-10-20 it read "stop": 3 < 10, as if
    /// the model had ended its answer by itself.
    #[test]
    fn a_generation_cut_by_the_context_window_is_length() {
        let (config, weights, reference) = fixture("qwen3");
        let spec = ModelSpec::parse(&config).unwrap();
        let vb = VarBuilder::from_slice_safetensors(&weights, DType::F32, &Device::Cpu).unwrap();
        let tokenizer = Arc::new(crate::prompt::text_only(
            Tokenizer::from_bytes(word_tokenizer(40)).unwrap(),
        ));
        let prompt: Vec<u32> = serde_json::from_value(reference["prompt"].clone()).unwrap();
        let prompt = words(&prompt);
        let prompt_len = tokenizer.encode(prompt.as_str(), true).unwrap().len();
        let context_tokens = prompt_len + 3;
        let loaded = Loaded {
            model: Slot::shared(spec.build(vb, Projections::Fused).unwrap()),
            arch: spec.arch,
            tokenizer,
            digest_hex: String::new(),
            device: Device::Cpu,
            eos_ids: vec![],
            context_tokens,
            prefill_chunk: prefill_chunk(spec.decoder.num_heads, context_tokens),
            decoder: spec.decoder.clone(),
            dtype_bytes: 4,
            budget: None,
            default_system: None,
            token_table: std::sync::OnceLock::new(),
        };
        let cut = loaded
            .generate_with(&prompt, &GenerationParams::greedy(10))
            .unwrap();
        assert_eq!(cut.completion_tokens, 3);
        assert!(cut.length && !cut.stopped);
        assert_eq!(cut.finish_reason(10), "length");
        let within = loaded
            .generate_with(&prompt, &GenerationParams::greedy(2))
            .unwrap();
        assert_eq!((within.completion_tokens, within.length), (2, true));
        assert!(cut.text.starts_with(&within.text), "the same greedy answer");
    }

    /// The qwen3 fixture behind the candle loop, with these EOS ids and this context.
    fn fixture_loaded(eos_ids: Vec<u32>, context: Option<usize>) -> Loaded {
        fixture_loaded_with(&word_tokenizer(40), eos_ids, context)
    }

    fn fixture_loaded_with(tokenizer: &[u8], eos_ids: Vec<u32>, context: Option<usize>) -> Loaded {
        let (config, weights, _) = fixture("qwen3");
        let spec = ModelSpec::parse(&config).unwrap();
        let vb = VarBuilder::from_slice_safetensors(&weights, DType::F32, &Device::Cpu).unwrap();
        let context_tokens = context.unwrap_or(spec.decoder.max_position_embeddings);
        Loaded {
            model: Slot::shared(spec.build(vb, Projections::Fused).unwrap()),
            arch: spec.arch,
            tokenizer: Arc::new(crate::prompt::text_only(
                Tokenizer::from_bytes(tokenizer).unwrap(),
            )),
            digest_hex: String::new(),
            device: Device::Cpu,
            eos_ids,
            context_tokens,
            prefill_chunk: prefill_chunk(spec.decoder.num_heads, context_tokens),
            decoder: spec.decoder.clone(),
            dtype_bytes: 4,
            budget: None,
            default_system: None,
            token_table: std::sync::OnceLock::new(),
        }
    }

    /// The candle model seen as a session: what `stepped::run` reads of it.
    struct CandleSteps<'m> {
        model: &'m mut ArchModel,
        pos: usize,
        logits: Vec<f32>,
    }

    impl crate::stepped::LogitsSession for CandleSteps<'_> {
        fn logits(&self) -> &[f32] {
            &self.logits
        }
        fn step(&mut self, token: u32) -> anyhow::Result<()> {
            let input = Tensor::new(&[token], &Device::Cpu)?.unsqueeze(0)?;
            let out = self.model.forward(&input, self.pos)?;
            self.logits = out.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
            self.pos += 1;
            Ok(())
        }
        fn finish(self) -> anyhow::Result<()> {
            self.model.clear_kv_cache()?;
            Ok(())
        }
    }

    /// `stepped::run` over the candle model, prefilled in the candle loop's chunks.
    fn stepped_answer(
        loaded: &Loaded,
        prompt: &Prompt,
        params: &GenerationParams,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        let req = crate::stepped::prepare(
            &loaded.tokenizer,
            prompt,
            params.max_tokens,
            loaded.context_tokens,
        )?;
        let chunk = loaded.chunk_for(req.prompt_ids.len());
        let budget = req.max_tokens;
        let mut guard = loaded.model.lock().unwrap();
        let model: &mut ArchModel = &mut guard;
        crate::stepped::run(
            &loaded.tokenizer,
            &loaded.eos_ids,
            &req,
            params,
            sink,
            move |ids: &[u32]| {
                let model = model;
                model.reserve(ids.len() + budget);
                let mut last = None;
                for (i, part) in ids.chunks(chunk).enumerate() {
                    let input = Tensor::new(part, &Device::Cpu)?.unsqueeze(0)?;
                    last = Some(model.forward(&input, i * chunk)?);
                }
                let logits = last
                    .ok_or_else(|| anyhow::anyhow!("empty prompt"))?
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1()?;
                Ok(CandleSteps {
                    model,
                    pos: ids.len(),
                    logits,
                })
            },
        )
    }

    /// The loop of `stepped`, fed the candle model's logits, answers what the
    /// candle loop answers, and streams the same pieces: greedy, sampled, stop
    /// sequences, EOS, the context's cut, a reader gone. A failure means a
    /// backend on `stepped` decides differently from candle on the same logits.
    #[test]
    fn the_stepped_loop_is_the_candle_loop_on_candle_logits() {
        use crate::stepped::tests::Recorder;
        let (_, _, reference) = fixture("qwen3");
        let prompt_ids: Vec<u32> = serde_json::from_value(reference["prompt"].clone()).unwrap();
        let prompt = words(&prompt_ids);
        let both =
            |loaded: &Loaded, params: &GenerationParams, break_at: Option<usize>, what: &str| {
                let mut a = Recorder {
                    break_at,
                    ..Recorder::default()
                };
                let mut b = Recorder {
                    break_at,
                    ..Recorder::default()
                };
                let want = loaded.generate_streaming(&prompt, params, &mut a);
                let got = stepped_answer(loaded, &prompt, params, &mut b);
                match (&want, &got) {
                    (Ok(w), Ok(g)) => assert_eq!(g, w, "{what}"),
                    (Err(w), Err(g)) => assert_eq!(
                        g.downcast_ref::<Cancelled>().is_some(),
                        w.downcast_ref::<Cancelled>().is_some(),
                        "{what}: {g:#} vs {w:#}"
                    ),
                    _ => panic!("{what}: candle {want:?}, stepped {got:?}"),
                }
                assert_eq!(b.deltas, a.deltas, "{what}: streamed pieces");
                want
            };
        let plain = fixture_loaded(vec![], None);
        let greedy = both(&plain, &GenerationParams::greedy(12), None, "greedy").unwrap();
        assert_eq!(greedy.completion_tokens, 12);
        let _ = both(&plain, &GenerationParams::greedy(1), None, "one token");
        for (seed, temperature, top_p) in
            [(0, 1.5, 1.0), (1, 1.5, 1.0), (2, 0.8, 0.9), (3, 1.0, 0.0)]
        {
            let mut p = GenerationParams::greedy(12);
            p.sampling = Some(Sampling {
                temperature,
                top_p,
                seed,
            });
            let _ = both(
                &plain,
                &p,
                None,
                &format!("sampled {seed} {temperature} {top_p}"),
            );
        }
        let words: Vec<&str> = greedy.text.split(' ').collect();
        let mut p = GenerationParams::greedy(12);
        p.stop = vec!["zzz".into(), words[5].into()];
        assert!(both(&plain, &p, None, "stop").unwrap().stopped);
        both(
            &plain,
            &GenerationParams::greedy(12),
            Some(2),
            "a reader gone",
        )
        .unwrap_err();

        let enc = plain.tokenizer.encode(greedy.text.as_str(), false).unwrap();
        let eos = fixture_loaded(vec![enc.get_ids()[4]], None);
        let ended = both(&eos, &GenerationParams::greedy(12), None, "eos").unwrap();
        assert!(ended.completion_tokens < 12);

        let cut = fixture_loaded(vec![], Some(prompt_ids.len() + 3));
        let c = both(&cut, &GenerationParams::greedy(12), None, "context").unwrap();
        assert_eq!((c.completion_tokens, c.length), (3, true));
    }

    /// Under a schema, `stepped` (Core's loop) picks the tokens, pieces and
    /// cut of the candle loop, greedy and sampled. A failure: Core holds a
    /// schema otherwise than candle, or not at all.
    #[test]
    fn the_stepped_loop_masks_as_the_candle_loop() {
        use crate::stepped::tests::Recorder;
        let loaded = fixture_loaded_with(&json_tokenizer(), vec![], None);
        let schema = crate::json_schema::JsonSchema::compile(&serde_json::json!({
            "type": "array", "items": {"type": ["boolean", "null", "string"]}, "maxItems": 4}))
        .unwrap();
        let prompt = Prompt::text("x y");
        for (sampling, max_tokens) in [
            (None, 30),
            (Some((1.5, 1)), 30),
            (Some((2.0, 2)), 30),
            (None, 2),
        ] {
            let mut p = GenerationParams::greedy(max_tokens);
            p.sampling = sampling.map(|(temperature, seed)| Sampling {
                temperature,
                top_p: 1.0,
                seed,
            });
            p.json_schema = Some(schema.clone());
            let (mut a, mut b) = (Recorder::default(), Recorder::default());
            let want = loaded.generate_streaming(&prompt, &p, &mut a).unwrap();
            let got = stepped_answer(&loaded, &prompt, &p, &mut b).unwrap();
            assert_eq!(got, want, "{sampling:?}");
            assert_eq!(b.deltas, a.deltas, "{sampling:?}");
            assert_ne!(
                schema.accepts(&want.text),
                crate::json_schema::Accepted::No,
                "{want:?}"
            );
        }
    }

    /// A failure means a model of an unknown family is built anyway.
    #[test]
    fn an_unknown_architecture_is_refused_at_load() {
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        let backend = CandleBackend::cpu(ModelArtifacts {
            config_json: br#"{"model_type":"llama","architectures":["LlamaForCausalLM"]}"#.to_vec(),
            tokenizer_json: word_tokenizer(4),
        });
        let err = match backend.load(b"weights", &key) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("llama must be refused"),
        };
        assert!(err.contains("\"llama\" is not supported"), "{err}");
    }

    /// OLMo chat templates, rendered and tokenized exactly as transformers
    /// renders the published templates (fixture from the real tokenizers).
    #[test]
    fn olmo_chat_templates_match_the_published_ones() {
        use crate::arch::Architecture;
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/arch/chat_templates.json")).unwrap();
        for (family, arch) in [
            ("olmo2", Architecture::Olmo2),
            ("olmo3", Architecture::Olmo3),
        ] {
            for case in fixture[family].as_array().unwrap() {
                let messages: Vec<crate::server::ChatMessage> =
                    serde_json::from_value(case["messages"].clone()).unwrap();
                assert_eq!(
                    arch.chat_prompt(&messages),
                    case["text"].as_str().unwrap(),
                    "{family}"
                );
            }
        }
    }

    /// The ministral3 fixture as Mistral publishes its checkpoints: inside the
    /// vision-language envelope, its tensors under `language_model.`, a vision
    /// tower beside them, and the `extra` tensors given (f32, one value).
    fn ministral3_envelope(extra: &[&str]) -> (Vec<u8>, Vec<u8>) {
        use safetensors::tensor::TensorView;
        let (config, weights, _) = fixture("ministral3");
        let text: serde_json::Value = serde_json::from_slice(&config).unwrap();
        let config = serde_json::to_vec(&serde_json::json!({
            "architectures": ["Mistral3ForConditionalGeneration"],
            "model_type": "mistral3",
            "text_config": text,
            "vision_config": {"model_type": "pixtral", "hidden_size": 4},
            "image_token_index": 10,
        }))
        .unwrap();
        let st = safetensors::SafeTensors::deserialize(&weights).unwrap();
        let mut owned: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> = st
            .names()
            .into_iter()
            .map(|n| {
                let t = st.tensor(n).unwrap();
                let name = format!("language_model.{n}");
                (name, t.dtype(), t.shape().to_vec(), t.data().to_vec())
            })
            .collect();
        let vision = [
            "vision_tower.transformer.layers.0.attention.q_proj.weight",
            "multi_modal_projector.linear_1.weight",
        ];
        for n in vision.iter().chain(extra) {
            let one = 1f32.to_le_bytes().to_vec();
            owned.push((n.to_string(), safetensors::Dtype::F32, vec![1], one));
        }
        let views: Vec<_> = owned
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), TensorView::new(*d, s.clone(), b).unwrap()))
            .collect();
        (config, safetensors::serialize(views, None).unwrap())
    }

    /// `family`'s fixture with `target` stored as `dtype` (`byte` repeated),
    /// under `prefix`.
    fn restored(
        family: &str,
        prefix: &str,
        target: &str,
        dtype: safetensors::Dtype,
        byte: u8,
    ) -> Vec<u8> {
        use safetensors::tensor::TensorView;
        let (_, weights, _) = fixture(family);
        let st = safetensors::SafeTensors::deserialize(&weights).unwrap();
        let owned: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> = st
            .names()
            .into_iter()
            .map(|n| {
                let t = st.tensor(n).unwrap();
                if n == target {
                    let count: usize = t.shape().iter().product();
                    let size = dtype.bitsize() / 8;
                    (
                        format!("{prefix}{n}"),
                        dtype,
                        t.shape().to_vec(),
                        vec![byte; count * size],
                    )
                } else {
                    (
                        format!("{prefix}{n}"),
                        t.dtype(),
                        t.shape().to_vec(),
                        t.data().to_vec(),
                    )
                }
            })
            .collect();
        let views: Vec<_> = owned
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), TensorView::new(*d, s.clone(), b).unwrap()))
            .collect();
        safetensors::serialize(views, None).unwrap()
    }

    /// A weight stored as an integer or FP8 is refused, in every family:
    /// quantized storage this engine does not dequantize, which a cast to
    /// F32 served as another model (measured 2026-10-28). F16 is read.
    #[test]
    fn a_weight_stored_as_an_integer_or_fp8_is_refused() {
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        let cases = [
            ("qwen3", "model.layers.0.self_attn.q_proj.weight"),
            ("olmo2", "model.layers.0.mlp.down_proj.weight"),
            ("olmo3", "model.embed_tokens.weight"),
            ("ministral3", "model.layers.0.self_attn.q_proj.weight"),
            ("ministral3", "model.layers.0.mlp.down_proj.weight"),
        ];
        for (family, target) in cases {
            let (config, _, _) = fixture(family);
            let tokenizer = if family == "ministral3" {
                ministral3_words()
            } else {
                word_tokenizer(40)
            };
            let load = |weights: Vec<u8>| {
                CandleBackend::cpu(ModelArtifacts {
                    config_json: config.clone(),
                    tokenizer_json: tokenizer.clone(),
                })
                .load(&weights, &key)
            };
            for (dtype, byte) in [
                (safetensors::Dtype::U8, 1u8),
                (safetensors::Dtype::F8_E4M3, 0x38),
                (safetensors::Dtype::I32, 1),
            ] {
                let err = match load(restored(family, "", target, dtype, byte)) {
                    Err(e) => format!("{e:#}"),
                    Ok(_) => panic!("{family}: a {dtype:?} {target} was loaded"),
                };
                assert!(err.contains(target) && err.contains("stored as"), "{err}");
            }
            assert!(
                load(restored(family, "", target, safetensors::Dtype::F16, 0)).is_ok(),
                "{family}: F16"
            );
        }
    }

    /// A 2^62 head_dim panicked at load in qwen3 and ministral3 (rotary
    /// tables sized from it); refused at parse now, in every family.
    #[test]
    fn a_hostile_head_dim_is_refused_at_load_not_a_panic() {
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        for family in ["qwen3", "olmo2", "olmo3", "ministral3"] {
            let (config, weights, _) = fixture(family);
            let mut v: serde_json::Value = serde_json::from_slice(&config).unwrap();
            v["head_dim"] = (1u64 << 62).into();
            let tokenizer = if family == "ministral3" {
                ministral3_words()
            } else {
                word_tokenizer(40)
            };
            let err = match CandleBackend::cpu(ModelArtifacts {
                config_json: serde_json::to_vec(&v).unwrap(),
                tokenizer_json: tokenizer,
            })
            .load(&weights, &key)
            {
                Err(e) => format!("{e:#}"),
                Ok(_) => panic!("{family}: loaded"),
            };
            assert!(err.contains("head_dim"), "{family}: {err}");
        }
    }

    /// Ministral 3 served end to end, the text model alone and inside its
    /// envelope: transformers' greedy tokens, twice (the KV cache is cleared).
    #[test]
    fn ministral3_generates_what_transformers_generates() {
        let (config, weights, reference) = fixture("ministral3");
        let (envelope, wrapped) = ministral3_envelope(&[]);
        let prompt: Vec<u32> = serde_json::from_value(reference["prompt"].clone()).unwrap();
        let want: Vec<u32> = serde_json::from_value(reference["generated"].clone()).unwrap();
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        for (config, weights) in [(config, weights), (envelope, wrapped)] {
            let backend = CandleBackend::cpu(ModelArtifacts {
                config_json: config,
                tokenizer_json: ministral3_words(),
            });
            assert_eq!(backend.name(), "candle-ministral3");
            let model = backend.load(&weights, &key).unwrap();
            let (text, _) = model.generate(&words(&prompt), want.len()).unwrap();
            assert_eq!(text, words(&want).as_str());
            assert_eq!(model.generate(&words(&prompt), want.len()).unwrap().0, text);
        }
    }

    /// A tensor beside the text model that it does not read is refused at
    /// load: an FP8 scale left unread would leave the weights unscaled.
    #[test]
    fn ministral3_refuses_a_tensor_it_does_not_compute() {
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        for extra in [
            "language_model.model.layers.0.self_attn.q_proj.weight_scale",
            "language_model.model.layers.0.self_attn.q_proj.bias",
        ] {
            let (config, weights) = ministral3_envelope(&[extra]);
            let backend = CandleBackend::cpu(ModelArtifacts {
                config_json: config,
                tokenizer_json: ministral3_words(),
            });
            let err = match backend.load(&weights, &key) {
                Err(e) => format!("{e:#}"),
                Ok(_) => panic!("{extra}: loaded"),
            };
            assert!(err.contains(extra), "{err}");
        }
    }

    /// `word_tokenizer(n)` with the word of `id` renamed `name`. A marker
    /// stays a special added token, of that id: tokenizers 0.23 gives an
    /// added token its vocabulary id and ignores the `id` written beside it.
    fn renamed(n: usize, id: u32, name: &str) -> Vec<u8> {
        let mut tokenizer: serde_json::Value = serde_json::from_slice(&word_tokenizer(n)).unwrap();
        let vocab = tokenizer["model"]["vocab"].as_object_mut().unwrap();
        vocab.remove(&format!("t{id}"));
        vocab.insert(name.into(), id.into());
        for a in tokenizer["added_tokens"].as_array_mut().unwrap() {
            if a["content"] == name {
                a["id"] = id.into();
            }
        }
        serde_json::to_vec(&tokenizer).unwrap()
    }

    /// The ministral3 fixture's words, `</s>` at 39: an id neither its prompt
    /// nor its greedy tokens hold.
    fn ministral3_words() -> Vec<u8> {
        renamed(40, 39, "</s>")
    }

    /// `</s>` ends a Ministral 3 answer (id 2 of the published tokenizer.json;
    /// its config.json names no EOS). The fixture's second greedy token is
    /// given each name in turn: Qwen's do not end it. A tokenizer without a
    /// special `</s>` is refused at load (control_markers).
    #[test]
    fn ministral3_stops_on_its_end_of_sequence_token() {
        let (config, weights, reference) = fixture("ministral3");
        let prompt: Vec<u32> = serde_json::from_value(reference["prompt"].clone()).unwrap();
        let want: Vec<u32> = serde_json::from_value(reference["generated"].clone()).unwrap();
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        let load = |name: &str| {
            CandleBackend::cpu(ModelArtifacts {
                config_json: config.clone(),
                tokenizer_json: renamed(40, want[1], name),
            })
            .load(&weights, &key)
        };
        let g = load("</s>")
            .unwrap()
            .generate_with(&words(&prompt), &GenerationParams::greedy(10))
            .unwrap();
        assert_eq!((g.completion_tokens, g.length), (1, false));
        for name in ["<|im_end|>", "<|endoftext|>"] {
            let g = load(name)
                .unwrap()
                .generate_with(&words(&prompt), &GenerationParams::greedy(10))
                .unwrap();
            assert_eq!((g.completion_tokens, g.length), (10, true), "{name}");
        }
        let err = match CandleBackend::cpu(ModelArtifacts {
            config_json: config.clone(),
            tokenizer_json: without_special(&renamed(40, want[1], "</s>"), "</s>", false),
        })
        .load(&weights, &key)
        {
            Err(e) => format!("{e:#}"),
            Ok(_) => panic!("loaded without </s>"),
        };
        assert!(err.contains("</s> not a special"), "{err}");
    }

    /// The default system message is the release's: the served digest picks
    /// it at load, and the loaded model prompts with it.
    #[test]
    fn a_pinned_ministral3_release_prompts_with_its_default_system_message() {
        let real = include_bytes!("../tests/arch/real/ministral3-14b-instruct-2512.json");
        let pin = "ba565c1c9abcd7ef89a0c9e9ffbf66fc7cfee4a0a7e7d39caba4857d6b2d169b";
        let spec = ModelSpec::parse(real).unwrap().served_as(pin);
        let d = spec.default_system.expect("the 14B's pin");
        assert!(
            d.starts_with("You are Ministral-3-14B-Instruct-2512, "),
            "{d}"
        );
        let other = ModelSpec::parse(real).unwrap().served_as(&"0".repeat(64));
        assert_eq!(other.default_system, None);
        let line = spec.default_system_line(pin);
        let sha: String = <sha2::Sha256 as sha2::Digest>::digest(d.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let sha = format!("sha256 {}...", &sha[..16]);
        assert!(
            line.contains("release ba565c1c9abc (") && line.contains(&sha),
            "{line}"
        );
        assert!(line.contains(&format!("{} bytes", d.len())), "{line}");
        let none = other.default_system_line(&"0".repeat(64));
        assert!(none.contains("none for release 000000000000;"), "{none}");
        let (config, weights, _) = fixture("ministral3");
        let mut spec = ModelSpec::parse(&config).unwrap().served_as(pin);
        assert_eq!(spec.default_system.map(|s| s.len()), Some(d.len()));
        let hi = [crate::server::ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let tokenizer = Arc::new(crate::prompt::text_only(
            Tokenizer::from_bytes(ministral3_words()).unwrap(),
        ));
        let mut rendered = Vec::new();
        for system in [None, Some("Be brief.")] {
            spec.default_system = system;
            let vb =
                VarBuilder::from_slice_safetensors(&weights, DType::F32, &Device::Cpu).unwrap();
            let model = spec.build(vb, Projections::Fused).unwrap();
            let loaded = finish(model, tokenizer.clone(), "d".into(), &spec, &Device::Cpu).unwrap();
            rendered.push(loaded.chat_prompt(&hi).as_str().to_string());
        }
        assert_eq!(
            rendered,
            [
                "<s>[INST]hi[/INST]",
                "<s>[SYSTEM_PROMPT]Be brief.[/SYSTEM_PROMPT][INST]hi[/INST]"
            ]
        );
        // The tiny fixture is no release: loaded, it has no default.
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        let backend = CandleBackend::cpu(ModelArtifacts {
            config_json: config,
            tokenizer_json: ministral3_words(),
        });
        let loaded = backend.load(&weights, &key).unwrap();
        assert_eq!(loaded.chat_prompt(&hi).as_str(), "<s>[INST]hi[/INST]");
    }

    use std::ops::ControlFlow;

    /// Records a generation's deltas; breaks at the `breaks_at`-th one, and
    /// reports the reader gone after `alive_for` reads of `alive()`.
    #[derive(Default)]
    struct Collect {
        started: usize,
        deltas: Vec<String>,
        breaks_at: Option<usize>,
        alive_for: Option<usize>,
        alive_reads: std::cell::Cell<usize>,
    }

    impl TokenSink for Collect {
        fn started(&mut self) -> ControlFlow<()> {
            self.started += 1;
            ControlFlow::Continue(())
        }
        fn delta(&mut self, text: &str) -> ControlFlow<()> {
            self.deltas.push(text.to_string());
            if Some(self.deltas.len()) == self.breaks_at {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }
        fn alive(&self) -> bool {
            self.alive_reads.set(self.alive_reads.get() + 1);
            self.alive_for.is_none_or(|n| self.alive_reads.get() <= n)
        }
    }

    /// The first `n`-word phrase, from word `min` on, whose first occurrence
    /// in `text` is its own: its first word's index, its byte offset, itself.
    fn own_phrase(text: &str, min: usize, n: usize) -> (usize, usize, String) {
        let words: Vec<&str> = text.split(' ').collect();
        (min..=words.len() - n)
            .map(|k| {
                let at = words[..k].iter().map(|w| w.len() + 1).sum::<usize>();
                (k, at, words[k..k + n].join(" "))
            })
            .find(|(_, at, phrase)| text.find(phrase.as_str()) == Some(*at))
            .expect("a phrase not met earlier in the text")
    }

    /// T4.
    /// What is streamed is what is returned, so what is signed: one delta per
    /// word-level token, and with a stop sequence no empty delta.
    #[test]
    fn streamed_deltas_concatenate_to_the_completion_text() {
        let (model, prompt) = fixture_model();
        let greedy = GenerationParams::greedy(10);
        let whole = model.generate_with(&prompt, &greedy).unwrap();
        let mut sink = Collect::default();
        let c = model
            .generate_streaming(&prompt, &greedy, &mut sink)
            .unwrap();
        assert_eq!(c, whole);
        assert_eq!(sink.started, 1);
        assert_eq!(sink.deltas.len(), 10, "one per token: {:?}", sink.deltas);
        assert_eq!(sink.deltas.concat(), c.text);

        let (_, at, phrase) = own_phrase(&whole.text, 2, 1);
        let mut p = GenerationParams::greedy(10);
        p.stop = vec![phrase];
        let mut sink = Collect::default();
        let c = model.generate_streaming(&prompt, &p, &mut sink).unwrap();
        assert!(c.stopped);
        assert_eq!(c.text, whole.text[..at]);
        assert!(
            sink.deltas.iter().all(|d| !d.is_empty()),
            "{:?}",
            sink.deltas
        );
        assert_eq!(sink.deltas.concat(), c.text);
    }

    /// A GPT-2 byte-level tokenizer: one token per byte, id = byte, no merge.
    fn byte_level_tokenizer() -> Tokenizer {
        let printable: Vec<u8> = (b'!'..=b'~')
            .chain(0xA1..=0xAC)
            .chain(0xAE..=0xFF)
            .collect();
        let mut shifted = 0u32;
        let vocab: serde_json::Map<String, serde_json::Value> = (0..=255u8)
            .map(|b| {
                let c = if printable.contains(&b) {
                    char::from(b)
                } else {
                    shifted += 1;
                    char::from_u32(255 + shifted).unwrap()
                };
                (c.to_string(), u32::from(b).into())
            })
            .collect();
        let byte_level = serde_json::json!({
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true,
        });
        let json = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": byte_level.clone(),
            "post_processor": null, "decoder": byte_level,
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
        });
        Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap()
    }

    /// T5.
    /// Byte-level tokens cut characters; the stream must not. A failure is a
    /// U+FFFD sent to the client, then a prefix check that refuses the answer.
    #[test]
    fn a_byte_level_stream_decoder_never_splits_a_character() {
        let tok = byte_level_tokenizer();
        let text = "é😀中";
        let ids: Vec<u32> = text.bytes().map(u32::from).collect();
        assert_eq!(
            tok.decode(&ids, true).unwrap(),
            text,
            "the fixture decodes bytes"
        );
        let mut d = StreamDecoder::new(&tok, &[]);
        let mut out = String::new();
        let mut deltas = 0;
        for &id in &ids {
            match d.push(id).unwrap() {
                Held::Emit(s) => {
                    assert!(!s.contains('\u{FFFD}'), "{s:?}");
                    deltas += usize::from(!s.is_empty());
                    out.push_str(&s);
                }
                Held::Stop(s) => panic!("no stop sequence was set: {s:?}"),
            }
        }
        assert_eq!(out, text);
        assert_eq!(deltas, 3, "one per character, none per byte");

        let mut d = StreamDecoder::new(&tok, &["😀".to_string()]);
        let mut sent = String::new();
        let stop_at = ids.iter().position(|&id| match d.push(id).unwrap() {
            Held::Emit(s) => {
                sent.push_str(&s);
                false
            }
            Held::Stop(s) => {
                sent.push_str(&s);
                true
            }
        });
        assert_eq!((sent.as_str(), stop_at), ("é", Some(5)));
    }

    /// T6.
    /// A reader that leaves ends the loop in `Cancelled`, and the next caller
    /// gets what a freshly loaded model answers: nothing stayed in the cache.
    #[test]
    fn a_sink_that_breaks_cancels_and_leaves_a_clean_cache() {
        let (model, prompt) = fixture_model();
        let greedy = GenerationParams::greedy(10);
        let mut sink = Collect {
            breaks_at: Some(3),
            ..Collect::default()
        };
        let e = model
            .generate_streaming(&prompt, &greedy, &mut sink)
            .unwrap_err();
        assert!(e.downcast_ref::<Cancelled>().is_some(), "{e}");
        assert_eq!(sink.deltas.len(), 3);

        let mut sink = Collect {
            alive_for: Some(4),
            ..Collect::default()
        };
        let e = model
            .generate_streaming(&prompt, &greedy, &mut sink)
            .unwrap_err();
        assert!(e.downcast_ref::<Cancelled>().is_some(), "{e}");
        assert_eq!(sink.deltas.len(), 4, "alive() is read before every step");

        let (fresh, _) = fixture_model();
        assert_eq!(
            model.generate_with(&prompt, &greedy).unwrap(),
            fresh.generate_with(&prompt, &greedy).unwrap()
        );
    }

    /// T7.
    /// A stop sequence over two tokens: its first token is held back, then
    /// dropped. A failure sends text the answer is later cut before.
    #[test]
    fn a_stop_sequence_spanning_two_tokens_is_never_streamed() {
        let (model, prompt) = fixture_model();
        let whole = model.generate(&prompt, 10).unwrap().0;
        let (k, at, phrase) = own_phrase(&whole, 2, 2);
        let mut p = GenerationParams::greedy(10);
        p.stop = vec![phrase];
        let mut sink = Collect::default();
        let c = model.generate_streaming(&prompt, &p, &mut sink).unwrap();
        assert!(c.stopped);
        assert_eq!(c.text, whole[..at]);
        assert_eq!(sink.deltas.concat(), c.text, "{:?}", sink.deltas);
        assert_eq!(
            c.completion_tokens as usize,
            k + 2,
            "stopped at its second token"
        );
    }

    mod conversation {
        use super::*;
        use crate::context::{AskedConversation, Handle};
        use crate::conversation::Route;

        fn loaded(keep_for: Duration) -> Loaded {
            let loaded = fixture_loaded(vec![], None);
            loaded.model.lock().unwrap().keep_for = keep_for;
            loaded
        }

        fn turn(route: Route, presented: Option<u8>, issued: u8, end: bool) -> Turn {
            let asked = match presented {
                None => AskedConversation::New { end },
                Some(b) => AskedConversation::Continue {
                    handle: Handle::from_bytes([b; 32]),
                    end,
                },
            };
            Turn::admit(route, &asked, || [issued; 32]).unwrap()
        }

        fn run(loaded: &Loaded, ids: &[u32], conversation: Option<&Turn>) -> Completion {
            let params = GenerationParams::greedy(3);
            loaded
                .generate_turn(&words(ids), &params, conversation, &mut NoSink)
                .unwrap()
        }

        /// Layer 0's K buffer, sharing its storage: what is left in it.
        fn probe(loaded: &Loaded) -> Option<Tensor> {
            match &loaded.model.lock().unwrap().model {
                ArchModel::Qwen3(m) => m.layer_cache(0).map(|(_, _, k)| k),
                _ => None,
            }
        }

        fn zeroed(t: &Tensor) -> bool {
            let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
            v.iter().all(|x| *x == 0.0)
        }

        fn prefilled(loaded: &Loaded) -> usize {
            loaded.model.lock().unwrap().last_prefill
        }

        fn reused(t: &Turn) -> (u32, usize) {
            let r = t.reused().expect("the backend recorded the turn");
            (r.turn, r.ids.len())
        }

        fn prompt() -> Vec<u32> {
            (1..=20).collect()
        }

        /// SECURITY: two conversations with the same prompt share nothing: the
        /// second computes every position, and the first's keys and values are
        /// zeroed before it runs; so are they before a request that names no
        /// conversation, or a handle presented on the other route. A failure:
        /// one caller's cache answering another's prompt.
        #[test]
        fn two_conversations_with_the_same_prefix_share_nothing() {
            let loaded = loaded(crate::conversation::KEEP_FOR);
            let p = prompt();
            let a = turn(Route::Chat, None, 1, false);
            let first = run(&loaded, &p, Some(&a));
            assert_eq!((reused(&a), prefilled(&loaded)), ((1, 0), 20));
            let left = probe(&loaded).expect("kept for its next turn");
            assert!(!zeroed(&left));

            let b = turn(Route::Chat, None, 2, false);
            assert_eq!(run(&loaded, &p, Some(&b)), first, "the same greedy answer");
            assert_eq!((reused(&b), prefilled(&loaded)), ((1, 0), 20));
            assert!(zeroed(&left), "the first conversation's keys, zeroed");

            // Its handle on the other route continues nothing.
            let left = probe(&loaded).unwrap();
            let other_route = turn(Route::Sealed, Some(2), 3, false);
            run(&loaded, &p, Some(&other_route));
            assert_eq!((reused(&other_route), prefilled(&loaded)), ((1, 0), 20));
            assert!(zeroed(&left));

            // A request without a conversation: zeroed before, nothing kept after.
            let left = probe(&loaded).unwrap();
            assert_eq!(run(&loaded, &p, None), first);
            assert_eq!(prefilled(&loaded), 20);
            assert!(zeroed(&left) && probe(&loaded).is_none());
        }

        /// A turn that presents its conversation's handle computes only what
        /// follows the prefix it shares with the cache, and answers what the
        /// same prompt answers from token 0. A failure: a reuse that changes
        /// the answer, or recomputes what it claims to reuse.
        #[test]
        fn a_continued_turn_computes_only_the_rest_and_answers_as_cold() {
            let loaded = loaded(crate::conversation::KEEP_FOR);
            let p1 = prompt();
            let first = turn(Route::Chat, None, 1, false);
            let c1 = run(&loaded, &p1, Some(&first));
            let g1 =
                crate::prompt::encode(&loaded.tokenizer, &Prompt::text(c1.text.clone())).unwrap();
            // The history as the next prompt carries it, then a new question
            // that does not begin as the cache's next token.
            let next = if g1.first() == Some(&21) { 22 } else { 21 };
            let mut p2 = p1.clone();
            p2.push(next);
            p2.extend([30, 31, 32]);
            let second = turn(Route::Chat, Some(1), 2, false);
            let warm = run(&loaded, &p2, Some(&second));
            assert_eq!(reused(&second), (2, 20));
            assert_eq!(prefilled(&loaded), p2.len() - 20);
            assert_eq!(warm, run(&loaded, &p2, None), "the answer from token 0");
            // The third turn: the second's prompt reused, under its handle.
            let mut p3 = p2.clone();
            p3.extend([33, 34]);
            let third = turn(Route::Chat, Some(2), 3, false);
            run(&loaded, &p3, Some(&third));
            assert_eq!(reused(&third), (1, 0), "the request in between evicted it");
            let fourth = turn(Route::Chat, Some(3), 4, false);
            let mut p4 = p3.clone();
            p4.push(35);
            let warm = run(&loaded, &p4, Some(&fourth));
            assert_eq!(reused(&fourth), (2, p3.len()));
            assert_eq!(warm, run(&loaded, &p4, None));
        }

        /// A prompt that differs from the cache at one position reuses exactly
        /// the positions before it, 0 at the first, and answers as from token
        /// 0. A failure: keys and values of a token the prompt no longer has.
        #[test]
        fn a_prefix_changed_by_one_token_invalidates_the_reuse_from_there() {
            let loaded = loaded(crate::conversation::KEEP_FOR);
            for (n, at) in [0usize, 7, 19].into_iter().enumerate() {
                let issued = 10 + 2 * n as u8;
                let p1 = prompt();
                run(
                    &loaded,
                    &p1,
                    Some(&turn(Route::Sealed, None, issued, false)),
                );
                let mut p2 = p1.clone();
                p2[at] = 39 - p2[at] % 5;
                p2.extend([25, 26]);
                let t = turn(Route::Sealed, Some(issued), issued + 1, false);
                let warm = run(&loaded, &p2, Some(&t));
                assert_eq!(reused(&t), (2, at), "changed at {at}");
                assert_eq!(prefilled(&loaded), p2.len() - at, "changed at {at}");
                assert_eq!(warm, run(&loaded, &p2, None), "changed at {at}");
            }
        }

        /// SECURITY: a kept conversation's keys and values are zeroed when it
        /// ends: after its time without a turn, after a turn that asks to end
        /// it, after a turn that fails, and with the model. A spent handle
        /// continues nothing. A failure: a cache left resident, or freed
        /// unzeroed, past the end the docs give.
        #[test]
        fn a_kept_conversation_is_zeroed_when_it_ends() {
            // Its time passes: the reaper zeroes it.
            let loaded = loaded(Duration::from_millis(50));
            let p = prompt();
            run(&loaded, &p, Some(&turn(Route::Chat, None, 1, false)));
            let left = probe(&loaded).unwrap();
            let deadline = Instant::now() + Duration::from_secs(10);
            while probe(&loaded).is_some() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            assert!(
                probe(&loaded).is_none() && zeroed(&left),
                "zeroed by its timer"
            );
            let late = turn(Route::Chat, Some(1), 2, false);
            run(&loaded, &p, Some(&late));
            assert_eq!(reused(&late), (1, 0));

            // A turn that asks to end it: nothing kept after it.
            let loaded = loaded_for_ever();
            run(&loaded, &p, Some(&turn(Route::Chat, None, 1, false)));
            let left = probe(&loaded).unwrap();
            let last = turn(Route::Chat, Some(1), 2, true);
            run(&loaded, &p, Some(&last));
            assert_eq!(reused(&last), (2, 19));
            assert!(probe(&loaded).is_none() && zeroed(&left));
            let after = turn(Route::Chat, Some(2), 3, false);
            run(&loaded, &p, Some(&after));
            assert_eq!(reused(&after), (1, 0), "ended");

            // A spent handle: its turn ran, the next handle is the live one.
            let again = turn(Route::Chat, Some(2), 4, false);
            run(&loaded, &p, Some(&again));
            assert_eq!(reused(&again), (1, 0), "spent");

            // A turn that fails: nothing kept, the cache zeroed.
            let left = probe(&loaded).unwrap();
            let mut gone = crate::stepped::tests::Recorder {
                break_at: Some(0),
                ..Default::default()
            };
            let failed = turn(Route::Chat, Some(4), 5, false);
            let e = loaded
                .generate_turn(
                    &words(&p),
                    &GenerationParams::greedy(3),
                    Some(&failed),
                    &mut gone,
                )
                .unwrap_err();
            assert!(e.downcast_ref::<Cancelled>().is_some(), "{e:#}");
            assert!(failed.reused().is_none());
            assert!(probe(&loaded).is_none() && zeroed(&left));

            // With the model.
            run(&loaded, &p, Some(&turn(Route::Chat, None, 6, false)));
            let left = probe(&loaded).unwrap();
            drop(loaded);
            // Dropped here, or by the timer thread if it held the model then.
            let deadline = Instant::now() + Duration::from_secs(10);
            while !zeroed(&left) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            assert!(zeroed(&left), "zeroed when the model is dropped");
        }

        /// A reader whose handler fails at the first delta.
        struct PanicsOnDelta;

        impl TokenSink for PanicsOnDelta {
            fn started(&mut self) -> ControlFlow<()> {
                ControlFlow::Continue(())
            }
            fn delta(&mut self, _: &str) -> ControlFlow<()> {
                panic!("mid-turn")
            }
        }

        /// SECURITY: a panic in the middle of a turn zeroes the cache then,
        /// before the panic goes on, not when the next request takes the
        /// poisoned lock: a lock nobody takes would keep the keys past the
        /// 300 s the docs give. That next request clears the poison and may
        /// keep a conversation. A failure: keys resident after a panic, or a
        /// model that forgets every conversation until it restarts.
        #[test]
        fn a_panic_during_a_turn_zeroes_the_cache_at_once() {
            let loaded = loaded_for_ever();
            let p = prompt();
            run(&loaded, &p, Some(&turn(Route::Chat, None, 1, false)));
            let left = probe(&loaded).unwrap();
            // The same length: the cache is written in place, not grown.
            let mut p2 = p.clone();
            p2[19] = 30;
            let panicking = turn(Route::Chat, Some(1), 2, false);
            let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                loaded.generate_turn(
                    &words(&p2),
                    &GenerationParams::greedy(3),
                    Some(&panicking),
                    &mut PanicsOnDelta,
                )
            }));
            assert!(unwound.is_err() && loaded.model.is_poisoned());
            // Read before any other request takes the lock.
            assert!(
                zeroed(&left),
                "zeroed at the panic, not at the next request"
            );
            assert!(panicking.reused().is_none(), "not signed");
            let next = turn(Route::Chat, None, 3, false);
            run(&loaded, &p, Some(&next));
            assert!(!loaded.model.is_poisoned(), "cleared by the next request");
            let continued = turn(Route::Chat, Some(3), 4, false);
            run(&loaded, &p2, Some(&continued));
            assert_eq!(reused(&continued), (2, 19));
        }

        fn loaded_for_ever() -> Loaded {
            loaded(crate::conversation::KEEP_FOR)
        }
    }
}
