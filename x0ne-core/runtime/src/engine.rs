//! E2 engine: Qwen3 dense bf16 on GPU 0, cuBLAS for every projection. One arena (weights,
//! KV cache, activations, logits, cuBLAS workspace) is frozen at load; client bytes are zeroed
//! after each request. The decode step (embedding, 36 layers, final norm, lm_head, argmax,
//! position advance) is captured once into a CUDA graph and replayed per token: the token
//! and the position stay on the device, the host receives 4 bytes per token.
//! Per layer: add_rmsnorm, qkv, qk_norm_rope_kv_write, attention, o, add_rmsnorm, gate_up, swiglu, down.
//! E3: the decode attention is split-K (P blocks per q head over interleaved 128-key tiles,
//! then a merge per q head in a fixed order); E2's per-head kernel stays selectable.
//! `Session`: a caller that picks each token itself (a server) gets the prefill's last logits
//! row, then per step sends one token up, replays the graph and gets one logits row back.
//! `Session::into_greedy`: the device picks and feeds each token itself, the host reads 4 bytes.
//! GM1, Qwen3-MoE: the feed-forward is the router GEMM (f32 logits), the routing kernel, the
//! [gate; up] and down GEMVs of the chosen experts (indices read on the device), SwiGLU and the
//! weighted combination. A MoE engine prefills through the decode route, one token per step.

use crate::arena::{Arena, Building, Class, Frozen, Slice};
use crate::config::{self, MoeConfig, Qwen3Config, Qwen3MoeConfig};
use crate::error::{format_err, Error, Result};
use crate::gpu::{DeviceBacking, Gpu, Graph, HostFloats, HostWords, Out, SplitPlan};
use crate::launch;
use crate::metrics;
use crate::ops::{
    as_bytes, attn_scale, inv_freq, launch_split, read_vec, split_state_bytes, CUBLAS_WORKSPACE,
};
use crate::st::{Safetensors, TensorSource, DEFAULT_FILE_CAP};
use crate::weights::{self, WeightSlices};
use std::path::Path;
use std::time::Instant;

pub use crate::gpu::Counts;
pub use crate::ops::{DecodeAttention, SplitMerge};

/// The split-K merge of a loaded engine (when its kernel is grouped) and of
/// `Kernels::attention_decode_split`.
pub const DEFAULT_SPLIT_MERGE: SplitMerge = SplitMerge::Kernel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineOptions {
    pub max_ctx: usize,
}

/// How logits are produced: one prefill over the sequence, or one decode step per token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Route {
    Prefill,
    Decode,
}

/// How a decode step is issued: every launch from the host (E1), or the graph captured at
/// load, replayed per token, token and position read on the device (E2, the default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecodeMode {
    Eager,
    Graph,
}

/// Per position i: argmax and log-sum-exp of the logits predicting token i + 1, and the
/// log-probability of the actual token i + 1 (NaN at the last position).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PosStats {
    pub argmax: u32,
    pub logsumexp: f32,
    pub next_logprob: f32,
}

/// Rows of one prefill launch sequence. Activations and logits are sized for it.
pub const PREFILL_CHUNK: usize = 512;

/// Decode steps in flight ahead of the host in graph mode: one host word (and its event) each.
const RING: usize = 16;

/// Most steps a greedy session runs ahead of its caller (`Session::into_greedy_ahead`).
pub const GREEDY_AHEAD_MAX: usize = RING - 1;

/// Steps `Session::into_greedy` runs ahead of its caller. X7 on the 5090, 2026-11-04: 8B decode
/// at 2 ahead is within 0.02 % of 15 ahead (82.10 against 82.08 tok/s at 16k) and a stop waits
/// 28 ms instead of 186 ms; 1 ahead loses 0.1 to 0.24 %.
pub const GREEDY_AHEAD: usize = 2;

/// Client buffers (all `Class::ClientData` but `inv_freq`).
struct Bufs {
    inv_freq: Slice,
    /// [layer][k|v][max_ctx][kv_dim] bf16.
    kv: Slice,
    tokens: Slice,
    pos: Slice,
    residual: Slice,
    delta: Slice,
    normed: Slice,
    qkv: Slice,
    q: Slice,
    attn: Slice,
    gate_up: Slice,
    act: Slice,
    logits: Slice,
    argmax: Slice,
    workspace: Slice,
    /// Split-K partial states of one decode step: [q_heads][P][SPLIT_STATE] f32.
    attn_state: Slice,
    /// One u32 per kv head for the last-block merge (option 7); the kernel leaves it at 0.
    attn_counter: Slice,
    moe: Option<MoeBufs>,
}

/// The MoE block's buffers. `gate_up` and `act` of `Bufs` hold the chosen experts' rows.
struct MoeBufs {
    /// `[chunk][num_experts]` f32.
    router_logits: Slice,
    /// `[layer][chunk][top_k]` u32: each layer keeps its routing until the next step.
    ids: Slice,
    /// `[layer][chunk][top_k]` f32.
    weights: Slice,
    /// `[chunk][top_k][hidden]` f32: the down outputs before the combination.
    y: Slice,
}

pub struct Engine {
    // Dropped before the arena and the device: they replay into it.
    graph: Option<Graph>,
    ring: Option<HostWords>,
    /// A session's logits row (vocab_size values) and its fed token, page-locked.
    host_logits: Option<HostFloats>,
    host_token: Option<HostWords>,
    /// A scrub failed: no session opens until one succeeds.
    poisoned: bool,
    cfg: Qwen3Config,
    /// `Some`: Qwen3-MoE, every layer's feed-forward is this sparse block.
    moe: Option<MoeConfig>,
    max_ctx: usize,
    chunk: usize,
    gpu: Gpu,
    arena: Arena<Frozen<DeviceBacking>>,
    w: WeightSlices,
    b: Bufs,
    mode: DecodeMode,
    attention: DecodeAttention,
    /// The split-K launch (kernel, grid, P), fixed at load from the GPU's SMs and occupancy.
    split: SplitPlan,
    merge: SplitMerge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Attn {
    Decode,
    Prefill,
}

fn u32_of(v: usize) -> u32 {
    u32::try_from(v).expect("checked at load: every size fits u32")
}

fn plan_bufs(
    c: &Qwen3Config,
    moe: Option<&MoeConfig>,
    max_ctx: usize,
    chunk: usize,
    splits: u32,
    a: &mut Arena<Building>,
) -> Result<Bufs> {
    let cd = Class::ClientData;
    let (h, n) = (c.hidden_size as u64, chunk as u64);
    let kv = c.num_hidden_layers as u64 * 2 * max_ctx as u64 * c.kv_dim() as u64 * 2;
    // Rows of the SwiGLU input and output: one per token, or one per (token, chosen expert).
    let (ffn_rows, inter) = match moe {
        None => (n, c.intermediate_size as u64),
        Some(m) => (
            n * m.num_experts_per_tok as u64,
            m.moe_intermediate_size as u64,
        ),
    };
    let moe_bufs = match moe {
        None => None,
        Some(m) => {
            let (e, k, l) = (
                m.num_experts as u64,
                m.num_experts_per_tok as u64,
                c.num_hidden_layers as u64,
            );
            Some(MoeBufs {
                router_logits: a.alloc("moe_router_logits", n * e * 4, cd)?,
                ids: a.alloc("moe_ids", l * n * k * 4, cd)?,
                weights: a.alloc("moe_weights", l * n * k * 4, cd)?,
                y: a.alloc("moe_y", n * k * h * 4, cd)?,
            })
        }
    };
    Ok(Bufs {
        inv_freq: a.alloc("inv_freq", (c.head_dim / 2 * 4) as u64, Class::Weights)?,
        kv: a.alloc("kv_cache", kv, cd)?,
        tokens: a.alloc("tokens", n * 4, cd)?,
        pos: a.alloc("pos", 4, cd)?,
        residual: a.alloc("residual", n * h * 4, cd)?,
        delta: a.alloc("delta", n * h * 2, cd)?,
        normed: a.alloc("normed", n * h * 2, cd)?,
        qkv: a.alloc("qkv", n * c.qkv_rows() as u64 * 2, cd)?,
        q: a.alloc("q", n * c.q_dim() as u64 * 2, cd)?,
        attn: a.alloc("attn", n * c.q_dim() as u64 * 2, cd)?,
        gate_up: a.alloc("gate_up", ffn_rows * 2 * inter * 2, cd)?,
        act: a.alloc("act", ffn_rows * inter * 2, cd)?,
        logits: a.alloc("logits", n * c.vocab_size as u64 * 4, cd)?,
        argmax: a.alloc("argmax", n * 4, cd)?,
        workspace: a.alloc("cublas_workspace", CUBLAS_WORKSPACE, cd)?,
        attn_state: a.alloc(
            "attention_split_state",
            split_state_bytes(c.num_attention_heads, splits),
            cd,
        )?,
        attn_counter: a.alloc(
            "attention_split_counter",
            c.num_key_value_heads as u64 * 4,
            cd,
        )?,
        moe: moe_bufs,
    })
}

/// Limits of the kernels and of the u32 sizes they take.
fn check_config(c: &Qwen3Config, moe: Option<&MoeConfig>, opts: EngineOptions) -> Result<()> {
    if let Some(m) = moe {
        if m.num_experts == 0 || m.num_experts > crate::ops::MOE_MAX_EXPERTS {
            return format_err(format!(
                "num_experts {} outside 1..={}",
                m.num_experts,
                crate::ops::MOE_MAX_EXPERTS
            ));
        }
        if m.num_experts_per_tok == 0 || m.num_experts_per_tok > 32 {
            return format_err(format!(
                "num_experts_per_tok {} outside 1..=32",
                m.num_experts_per_tok
            ));
        }
        if m.moe_intermediate_size == 0 || !m.moe_intermediate_size.is_multiple_of(8) {
            return format_err("moe_intermediate_size is not a positive multiple of 8");
        }
        let big = [
            m.num_experts * 2 * m.moe_intermediate_size,
            m.num_experts_per_tok * c.hidden_size,
        ];
        if big.iter().any(|&v| u32::try_from(v).is_err()) {
            return format_err("a MoE dimension exceeds u32");
        }
    }
    if c.head_dim != x0ne_core_kernels::HEAD_DIM {
        return format_err(format!(
            "head_dim {}: kernels are built for {}",
            c.head_dim,
            x0ne_core_kernels::HEAD_DIM
        ));
    }
    if !c.hidden_size.is_multiple_of(8) {
        return format_err("hidden_size is not a multiple of 8");
    }
    if opts.max_ctx == 0 || opts.max_ctx > c.max_position_embeddings {
        return format_err(format!(
            "max_ctx {} outside 1..={}",
            opts.max_ctx, c.max_position_embeddings
        ));
    }
    let big = [
        c.vocab_size,
        c.intermediate_size * 2,
        c.qkv_rows(),
        opts.max_ctx * c.kv_dim(),
    ];
    if big.iter().any(|&v| u32::try_from(v).is_err()) {
        return format_err("a dimension exceeds u32");
    }
    Ok(())
}

impl Engine {
    pub fn load(model_dir: &Path, opts: EngineOptions) -> Result<Engine> {
        let (cfg, moe) =
            config::read_any(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
        check_config(&cfg, moe.as_ref().map(|m| &m.moe), opts)?;
        let src = Safetensors::open_dir(model_dir, DEFAULT_FILE_CAP)?;
        Engine::build(cfg, moe, &src, opts)
    }

    /// `load` from a config text and tensors already in memory (`st::SliceSource`).
    pub fn load_from(
        config_json: &str,
        weights: &impl TensorSource,
        opts: EngineOptions,
    ) -> Result<Engine> {
        let (cfg, moe) = config::read_any(config_json)?;
        check_config(&cfg, moe.as_ref().map(|m| &m.moe), opts)?;
        Engine::build(cfg, moe, weights, opts)
    }

    fn build(
        cfg: Qwen3Config,
        moe: Option<Qwen3MoeConfig>,
        src: &impl TensorSource,
        opts: EngineOptions,
    ) -> Result<Engine> {
        match &moe {
            None => weights::check(&cfg, src)?,
            Some(m) => weights::check_moe(m, src)?,
        }
        let gpu = Gpu::new()?;
        // A MoE engine prefills through the decode route: one row per pass.
        let chunk = if moe.is_some() {
            1
        } else {
            PREFILL_CHUNK.min(opts.max_ctx)
        };
        let split = gpu.split_plan(cfg.num_attention_heads, cfg.num_key_value_heads)?;
        let mut a = Arena::new();
        let w = match &moe {
            None => weights::plan(&cfg, &mut a)?,
            Some(m) => weights::plan_moe(m, &mut a)?,
        };
        let moe_cfg = moe.as_ref().map(|m| m.moe.clone());
        let b = plan_bufs(
            &cfg,
            moe_cfg.as_ref(),
            opts.max_ctx,
            chunk,
            split.splits,
            &mut a,
        )?;
        let mut arena = a.freeze(|len| gpu.allocate(len))?;
        match &moe {
            None => weights::upload(&cfg, src, &w, &mut arena)?,
            Some(m) => weights::upload_moe(m, src, &w, &mut arena)?,
        };
        arena.write(
            b.inv_freq,
            0,
            as_bytes(&inv_freq(cfg.head_dim, cfg.rope_theta as f32)),
        )?;
        arena.scrub_client_data()?;
        // SAFETY: a CUBLAS_WORKSPACE-byte client slice of the arena, which the engine drops
        // after the Gpu's handle has no GEMM left (field order, `Gpu::drop` synchronizes).
        unsafe { gpu.set_cublas_workspace(arena.ptr(b.workspace)?, CUBLAS_WORKSPACE)? };
        let ring = gpu.host_words(RING)?;
        let host_logits = gpu.host_floats(cfg.vocab_size)?;
        let host_token = gpu.host_words(1)?;
        let mut e = Engine {
            graph: None,
            ring: Some(ring),
            host_logits: Some(host_logits),
            host_token: Some(host_token),
            poisoned: false,
            cfg,
            moe: moe_cfg,
            max_ctx: opts.max_ctx,
            chunk,
            gpu,
            arena,
            w,
            b,
            mode: DecodeMode::Graph,
            attention: DecodeAttention::SplitK,
            merge: if split.grouped {
                DEFAULT_SPLIT_MERGE
            } else {
                SplitMerge::Kernel
            },
            split,
        };
        e.capture_decode_step()?;
        Ok(e)
    }

    /// Opens one request: prefills `prompt` and holds the logits of its last position.
    /// A failed opening leaves the client bytes zeroed (or the engine poisoned).
    pub fn session(&mut self, prompt: &[u32]) -> Result<Session<'_>> {
        if self.poisoned {
            self.scrub()?;
        }
        self.check_ids(prompt)?;
        match self.open_session(prompt) {
            Ok(()) => Ok(Session {
                e: self,
                pos: prompt.len(),
                open: true,
                failed: false,
            }),
            Err(err) => {
                let _ = self.scrub();
                Err(err)
            }
        }
    }

    /// Prefill as `greedy_inner` does (last row's logits into row 0), the position for the
    /// decode graph, then that row on the host.
    fn open_session(&mut self, prompt: &[u32]) -> Result<()> {
        let mut start = 0;
        while start < prompt.len() {
            let len = self.chunk.min(prompt.len() - start);
            self.put_tokens(&prompt[start..start + len])?;
            let last = start + len == prompt.len();
            self.forward(
                start,
                len,
                self.prefill_attn(),
                last.then_some((len - 1, 1)),
            )?;
            start += len;
        }
        self.set_pos(prompt.len())?;
        self.pull_logits()
    }

    /// Logits row 0 into the host row: one asynchronous copy, then its event.
    fn pull_logits(&mut self) -> Result<()> {
        let host = self
            .host_logits
            .as_mut()
            .ok_or_else(|| Error::Gpu("no host logits row".into()))?;
        host.fetch(&self.arena, self.b.logits, 0, self.cfg.vocab_size)?;
        host.values().map(|_| ())
    }

    /// One decode step of an open session on `token` at the device's `pos`: the token goes up
    /// from a host word, the graph runs, the logits row comes back.
    fn session_step(&mut self, token: u32) -> Result<()> {
        let word = self
            .host_token
            .as_ref()
            .ok_or_else(|| Error::Gpu("no host token word".into()))?;
        word.set(0, token)?;
        word.send(0, &self.arena, self.b.tokens, 0)?;
        let g = self.graph()?;
        // SAFETY: the graph launches on `self.arena`, frozen at load and dropped after the
        // graph (field order).
        unsafe { self.gpu.launch_graph(g) }?;
        self.pull_logits()
    }

    /// Zeroes every client byte: the arena's, then the host buffers once the stream is drained.
    /// A failure poisons the engine.
    fn scrub(&mut self) -> Result<()> {
        self.poisoned = true;
        // Each region's memset is followed by a stream synchronization: no copy is in flight below.
        self.arena.scrub_client_data()?;
        for h in self.host_logits.iter_mut() {
            h.zero()?;
        }
        for w in self.host_token.iter().chain(self.ring.iter()) {
            w.zero()?;
        }
        self.poisoned = false;
        Ok(())
    }

    /// True when the host buffers that receive or send client data (a session's logits row
    /// and token, the greedy ring) hold only zeros.
    pub fn host_buffers_are_zero(&self) -> bool {
        self.host_logits.iter().all(HostFloats::is_zero)
            && self
                .host_token
                .iter()
                .chain(self.ring.iter())
                .all(HostWords::is_zero)
    }

    /// One eager decode step on the zeroed client buffers (cuBLAS and the driver load their
    /// code lazily, which a capture must not see), then the capture of the same launches.
    /// Nothing of it survives: the client buffers are scrubbed afterwards.
    fn capture_decode_step(&mut self) -> Result<()> {
        self.decode_step_launches()?;
        self.gpu.sync()?;
        let graph = self.gpu.capture(|| self.decode_step_launches())?;
        self.graph = Some(graph);
        self.arena.scrub_client_data()?;
        Ok(())
    }

    /// The launches of one decode step at the device's `pos` on the device's `tokens[0]`:
    /// forward, logits, argmax into `tokens[0]`, `pos += 1`. No host copy, no synchronization.
    fn decode_step_launches(&self) -> Result<()> {
        self.step_launches(1, Attn::Decode, Some((0, 1)))?;
        self.launch_argmax(1, self.b.tokens)?;
        let g = &self.gpu;
        // SAFETY: `pos` is a one-word client slice of the arena.
        unsafe { launch!(g, g.f.advance, (1, 1), 32; self.arena.ptr(self.b.pos)?) }
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.cfg
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn decode_mode(&self) -> DecodeMode {
        self.mode
    }

    /// Eager mode is the witness of the graph: same launches, issued one by one from the host.
    pub fn set_decode_mode(&mut self, mode: DecodeMode) -> Result<()> {
        if mode == DecodeMode::Graph && self.graph.is_none() {
            return Err(Error::Gpu("no decode graph captured".into()));
        }
        self.mode = mode;
        Ok(())
    }

    /// The decode attention kernel the step (eager and graph) uses.
    pub fn decode_attention(&self) -> DecodeAttention {
        self.attention
    }

    /// Selects the decode attention kernel (E3 split-K by default, E2 per head as the
    /// reference); the decode graph is captured again for it, which zeroes the client
    /// buffers: call it between requests. If the capture fails, no graph is left.
    pub fn set_decode_attention(&mut self, a: DecodeAttention) -> Result<()> {
        if a == self.attention && self.graph.is_some() {
            return Ok(());
        }
        self.attention = a;
        self.graph = None;
        self.capture_decode_step()
    }

    /// How the split-K states are merged.
    pub fn split_merge(&self) -> SplitMerge {
        self.merge
    }

    /// Selects the split-K merge (`LastBlock` needs a grouped kernel); the decode graph is
    /// captured again, which zeroes the client buffers: call it between requests.
    pub fn set_split_merge(&mut self, m: SplitMerge) -> Result<()> {
        if m == SplitMerge::LastBlock && !self.split.grouped {
            return Err(Error::Gpu(
                "the last-block merge needs a grouped split-K kernel".into(),
            ));
        }
        if m == self.merge && self.graph.is_some() {
            return Ok(());
        }
        self.merge = m;
        self.graph = None;
        self.capture_decode_step()
    }

    /// P, the split-K slices of the context.
    pub fn splits(&self) -> u32 {
        self.split.splits
    }

    /// Driver calls this engine's context has made so far (see `gpu::Counts`).
    pub fn counts(&self) -> Counts {
        self.gpu.counts()
    }

    /// Nodes of the captured decode graph.
    pub fn graph_nodes(&self) -> usize {
        self.graph.as_ref().map_or(0, Graph::nodes)
    }

    /// GPU name, cubin architecture, driver and cuBLAS versions, arena size, graph nodes.
    pub fn describe(&self) -> String {
        let g = &self.gpu;
        format!(
            "{} (cc {}.{}, {} SMs, cubin {}), driver {}, cuBLAS {}, arena {} bytes, decode graph {} nodes, decode attention {}, feed-forward {}",
            g.name,
            g.cc.0,
            g.cc.1,
            g.sms,
            g.arch,
            g.driver_version,
            g.cublas_version,
            crate::arena::Backing::len(self.arena.backing()),
            self.graph_nodes(),
            match self.attention {
                DecodeAttention::SplitK => format!(
                    "split-K P {} (grid {} x {}, {} q heads per block, {} blocks per SM, merge {:?})",
                    self.split.splits, self.split.grid_x, self.split.splits, self.split.group, self.split.per_sm, self.merge
                ),
                DecodeAttention::PerHead => "per head (E2)".to_string(),
            },
            match &self.moe {
                None => "dense".to_string(),
                Some(m) => format!(
                    "MoE, {} experts of width {}, {} per token (prefill through the decode route)",
                    m.num_experts, m.moe_intermediate_size, m.num_experts_per_tok
                ),
            }
        )
    }

    /// The sparse feed-forward of a Qwen3-MoE engine; `None` for a dense one.
    pub fn moe(&self) -> Option<&MoeConfig> {
        self.moe.as_ref()
    }

    /// GM1 diagnostic: the experts every layer chose at each position of `ids`, fed one token
    /// per step through the eager decode route; `[position][layer][top_k]`, flattened. Empty for
    /// a dense engine. Client bytes are zeroed afterwards.
    pub fn moe_routing(&mut self, ids: &[u32]) -> Result<Vec<u32>> {
        self.check_ids(ids)?;
        let result = self.moe_routing_inner(ids);
        self.scrub()?;
        result
    }

    fn moe_routing_inner(&mut self, ids: &[u32]) -> Result<Vec<u32>> {
        let (k, slice) = match (&self.moe, &self.b.moe) {
            (Some(m), Some(mb)) => (m.num_experts_per_tok, mb.ids),
            _ => return Ok(Vec::new()),
        };
        // chunk is 1 for a MoE engine: the ids slice is [layer][top_k].
        let per = self.cfg.num_hidden_layers * k;
        let mut out = Vec::with_capacity(ids.len() * per);
        for p in 0..ids.len() {
            self.put_tokens(&ids[p..p + 1])?;
            self.forward(p, 1, Attn::Decode, None)?;
            self.gpu.sync()?;
            out.extend(read_vec::<u32>(&self.arena, slice, 0, per)?);
        }
        Ok(out)
    }

    /// Free device bytes as cuMemGetInfo reports them (whole device, every process).
    pub fn device_free_bytes(&self) -> Result<u64> {
        Ok(self.gpu.mem_info()?.0)
    }

    /// True when every client byte of the arena (KV cache, activations, logits,
    /// cuBLAS workspace) reads back as zero.
    pub fn client_data_is_zero(&self) -> Result<bool> {
        use crate::arena::Backing;
        const STEP: u64 = 64 << 20;
        let mut buf = vec![0u8; STEP as usize];
        for r in self
            .arena
            .regions()
            .iter()
            .filter(|r| r.class == Class::ClientData)
        {
            let mut at = 0;
            while at < r.len {
                let n = STEP.min(r.len - at);
                self.arena
                    .backing()
                    .read(r.offset + at, &mut buf[..n as usize])?;
                if buf[..n as usize].iter().any(|&b| b != 0) {
                    return Ok(false);
                }
                at += n;
            }
        }
        Ok(true)
    }

    fn check_ids(&self, ids: &[u32]) -> Result<()> {
        if ids.is_empty() || ids.len() > self.max_ctx {
            return format_err(format!("{} tokens for max_ctx {}", ids.len(), self.max_ctx));
        }
        if let Some(t) = ids.iter().find(|&&t| t as usize >= self.cfg.vocab_size) {
            return format_err(format!("token {t} outside the vocabulary"));
        }
        Ok(())
    }

    /// The attention of a prefill pass: the decode kernel for a MoE engine (one row per pass).
    fn prefill_attn(&self) -> Attn {
        if self.moe.is_some() {
            Attn::Decode
        } else {
            Attn::Prefill
        }
    }

    fn put_tokens(&mut self, ids: &[u32]) -> Result<()> {
        self.arena.write(self.b.tokens, 0, as_bytes(ids))
    }

    fn set_pos(&mut self, pos: usize) -> Result<()> {
        self.arena.write(self.b.pos, 0, &u32_of(pos).to_le_bytes())
    }

    /// One pass over `n` tokens already in `tokens`, at positions pos0..pos0+n (the host
    /// writes `pos`). Writes their K and V into the cache; with `logits = Some((first, count))`,
    /// writes f32 logits of rows first..first+count into `logits`.
    fn forward(
        &mut self,
        pos0: usize,
        n: usize,
        attn: Attn,
        logits: Option<(usize, usize)>,
    ) -> Result<()> {
        if n == 0 || n > self.chunk || pos0 + n > self.max_ctx || (attn == Attn::Decode && n != 1) {
            return format_err(format!("forward: {n} rows at {pos0}"));
        }
        self.set_pos(pos0)?;
        self.step_launches(n, attn, logits)
    }

    /// The kernel and GEMM launches of one pass over `n` rows at the device's `pos`: no host
    /// copy and no synchronization, so the sequence can be captured into a graph.
    fn step_launches(&self, n: usize, attn: Attn, logits: Option<(usize, usize)>) -> Result<()> {
        let (c, g, a, b, w) = (&self.cfg, &self.gpu, &self.arena, &self.b, &self.w);
        let (h, qd, kvd, inter) = (c.hidden_size, c.q_dim(), c.kv_dim(), c.intermediate_size);
        let (qh, kvh) = (u32_of(c.num_attention_heads), u32_of(c.num_key_value_heads));
        let eps = c.rms_norm_eps as f32;
        let scale = attn_scale(c.head_dim);
        let nn = u32_of(n);
        let plane = (self.max_ctx * kvd * 2) as u64;
        let (tokens, pos) = (a.ptr(b.tokens)?, a.ptr(b.pos)?);
        let (residual, delta, normed) = (a.ptr(b.residual)?, a.ptr(b.delta)?, a.ptr(b.normed)?);
        let (qkv, q, att) = (a.ptr(b.qkv)?, a.ptr(b.q)?, a.ptr(b.attn)?);
        let (gate_up, act) = (a.ptr(b.gate_up)?, a.ptr(b.act)?);
        if n == 0 || n > self.chunk {
            return format_err(format!("step: {n} rows for chunks of {}", self.chunk));
        }
        if let Some((first, count)) = logits {
            if first + count > n {
                return format_err("forward: logits rows outside the pass");
            }
        }

        // SAFETY: every address is a slice of `self.arena` that `plan_bufs` and the weight plan
        // sized for `chunk` rows, `max_ctx` positions and this config; n <= chunk and the logits
        // rows lie in the pass (checked above). The arena outlives the launches: the engine
        // synchronizes before it drops, and a replayed graph runs on the same arena.
        unsafe {
            launch!(g, g.f.embedding, (nn, 1), 128; a.ptr(w.embed)?, tokens, delta, u32_of(h), u32_of(c.vocab_size))?;
            g.memset_u32(residual, 0, n * h)?;
            for (l, ls) in w.layers.iter().enumerate() {
                let kc = a.addr(b.kv, 2 * l as u64 * plane)?;
                let vc = a.addr(b.kv, (2 * l as u64 + 1) * plane)?;
                launch!(g, g.f.add_rmsnorm, (nn, 1), 256; residual, delta, a.ptr(ls.input_norm)?, normed, u32_of(h), eps)?;
                g.gemm(a.ptr(ls.qkv)?, c.qkv_rows(), h, normed, n, qkv, Out::Bf16)?;
                launch!(g, g.f.qk_norm_rope_kv_write, (qh + kvh, nn), 64;
                    qkv, a.ptr(ls.q_norm)?, a.ptr(ls.k_norm)?, a.ptr(b.inv_freq)?, q, kc, vc, pos, qh, kvh,
                    u32_of(self.max_ctx), eps)?;
                match attn {
                    Attn::Decode => match self.attention {
                        DecodeAttention::PerHead => launch!(g, g.f.attention_decode, (qh, 1), 256;
                            q, kc, vc, att, pos, qh, kvh, u32_of(self.max_ctx), scale)?,
                        DecodeAttention::SplitK => launch_split(
                            g,
                            &self.split,
                            self.merge,
                            q,
                            kc,
                            vc,
                            a.ptr(b.attn_state)?,
                            a.ptr(b.attn_counter)?,
                            att,
                            pos,
                            qh,
                            kvh,
                            u32_of(self.max_ctx),
                            scale,
                        )?,
                    },
                    Attn::Prefill => {
                        launch!(g, g.f.attention_prefill, (qh, u32_of(n.div_ceil(8))), 256;
                        q, kc, vc, att, pos, nn, qh, kvh, u32_of(self.max_ctx), scale)?
                    }
                }
                g.gemm(a.ptr(ls.o)?, h, qd, att, n, delta, Out::Bf16)?;
                launch!(g, g.f.add_rmsnorm, (nn, 1), 256; residual, delta, a.ptr(ls.post_norm)?, normed, u32_of(h), eps)?;
                match (&self.moe, &b.moe, ls.router) {
                    (None, _, _) => {
                        g.gemm(
                            a.ptr(ls.gate_up)?,
                            2 * inter,
                            h,
                            normed,
                            n,
                            gate_up,
                            Out::Bf16,
                        )?;
                        launch!(g, g.f.swiglu, (u32_of(inter.div_ceil(256)), nn), 256; gate_up, act, u32_of(inter))?;
                        g.gemm(a.ptr(ls.down)?, h, inter, act, n, delta, Out::Bf16)?;
                    }
                    (Some(m), Some(mb), Some(router)) => {
                        let (ne, k, mi) = (
                            m.num_experts,
                            m.num_experts_per_tok,
                            m.moe_intermediate_size,
                        );
                        let (ne32, k32, slots) = (u32_of(ne), u32_of(k), u32_of(n * k));
                        let logits_r = a.ptr(mb.router_logits)?;
                        let ids = a.addr(mb.ids, (l * self.chunk * k * 4) as u64)?;
                        let wts = a.addr(mb.weights, (l * self.chunk * k * 4) as u64)?;
                        let y = a.ptr(mb.y)?;
                        g.gemm(a.ptr(router)?, ne, h, normed, n, logits_r, Out::F32)?;
                        launch!(g, g.f.moe_route, (nn, 1), 128; logits_r, ne32, k32, ids, wts)?;
                        launch!(g, g.f.moe_gemv_bf16, (u32_of((2 * mi).div_ceil(8)), slots), 256;
                            a.ptr(ls.gate_up)?, ids, normed, gate_up, u32_of(2 * mi), u32_of(h), ne32, k32, u32_of(h))?;
                        launch!(g, g.f.swiglu, (u32_of(mi.div_ceil(256)), slots), 256; gate_up, act, u32_of(mi))?;
                        launch!(g, g.f.moe_gemv_f32, (u32_of(h.div_ceil(8)), slots), 256;
                            a.ptr(ls.down)?, ids, act, y, u32_of(h), u32_of(mi), ne32, 1u32, u32_of(mi))?;
                        launch!(g, g.f.moe_combine, (u32_of(h.div_ceil(256)), nn), 256; y, wts, delta, u32_of(h), k32)?;
                    }
                    _ => {
                        return Err(Error::Gpu(
                            "MoE config without its buffers or router".into(),
                        ))
                    }
                }
            }
            launch!(g, g.f.add_rmsnorm, (nn, 1), 256; residual, delta, a.ptr(w.final_norm)?, normed, u32_of(h), eps)?;
            if let Some((first, count)) = logits {
                let head = w.lm_head.unwrap_or(w.embed);
                let x = a.addr(b.normed, (first * h * 2) as u64)?;
                g.gemm(
                    a.ptr(head)?,
                    c.vocab_size,
                    h,
                    x,
                    count,
                    a.ptr(b.logits)?,
                    Out::F32,
                )?;
            }
        }
        Ok(())
    }

    /// Argmax of logits rows 0..count into `out` (device), launched, not waited for.
    fn launch_argmax(&self, count: usize, out: Slice) -> Result<()> {
        let (g, a) = (&self.gpu, &self.arena);
        if count == 0 || count > self.chunk || out.len() < count as u64 * 4 {
            return format_err(format!("argmax: {count} rows"));
        }
        // SAFETY: `logits` holds `chunk` rows of vocab_size f32 and `out` at least `count`
        // words (checked above), both slices of the arena.
        unsafe {
            launch!(g, g.f.argmax, (u32_of(count), 1), 1024; a.ptr(self.b.logits)?, u32_of(self.cfg.vocab_size), a.ptr(out)?)
        }
    }

    /// `count` argmax words read back from `out` after the stream is drained.
    fn read_argmax(&self, count: usize, out: Slice) -> Result<Vec<u32>> {
        self.gpu.sync()?;
        let v: Vec<u32> = read_vec(&self.arena, out, 0, count)?;
        if v.iter().any(|&t| t as usize >= self.cfg.vocab_size) {
            return Err(Error::Gpu(
                "argmax: a logits row has no finite maximum".into(),
            ));
        }
        Ok(v)
    }

    /// Argmax of logits rows 0..count, written to `out` (device) and returned.
    fn argmax_rows(&self, count: usize, out: Slice) -> Result<Vec<u32>> {
        self.launch_argmax(count, out)?;
        self.read_argmax(count, out)
    }

    /// Stats of logits rows 0..count for positions start..start+count; full rows kept for
    /// positions listed in `keep` (index into `rows`). `am` holds the argmax of each row.
    #[allow(clippy::too_many_arguments)]
    fn collect(
        &self,
        ids: &[u32],
        start: usize,
        count: usize,
        am: &[u32],
        keep: &[usize],
        stats: &mut Vec<PosStats>,
        rows: &mut [Option<Vec<f32>>],
    ) -> Result<()> {
        let vocab = self.cfg.vocab_size;
        let logits: Vec<f32> = read_vec(&self.arena, self.b.logits, 0, count * vocab)?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(count);
        let per = count.div_ceil(threads);
        let lse: Vec<f64> = std::thread::scope(|s| {
            let hs: Vec<_> = logits
                .chunks(per * vocab)
                .map(|part| {
                    s.spawn(move || {
                        part.chunks_exact(vocab)
                            .map(metrics::logsumexp)
                            .collect::<Vec<f64>>()
                    })
                })
                .collect();
            hs.into_iter()
                .flat_map(|h| h.join().expect("stats worker"))
                .collect()
        });
        for i in 0..count {
            let p = start + i;
            let row = &logits[i * vocab..(i + 1) * vocab];
            let next_logprob = match ids.get(p + 1) {
                Some(&t) => (row[t as usize] as f64 - lse[i]) as f32,
                None => f32::NAN,
            };
            stats.push(PosStats {
                argmax: am[i],
                logsumexp: lse[i] as f32,
                next_logprob,
            });
            for (k, &want) in keep.iter().enumerate() {
                if want == p {
                    rows[k] = Some(row.to_vec());
                }
            }
        }
        Ok(())
    }

    /// Teacher-forced pass over `ids`: stats at every position, full logits at `keep`.
    pub fn run(
        &mut self,
        ids: &[u32],
        route: Route,
        keep: &[usize],
    ) -> Result<(Vec<PosStats>, Vec<Vec<f32>>)> {
        self.check_ids(ids)?;
        if let Some(&p) = keep.iter().find(|&&p| p >= ids.len()) {
            return format_err(format!("position {p} outside {} tokens", ids.len()));
        }
        let result = self.run_inner(ids, route, keep);
        self.scrub()?;
        result
    }

    fn graph(&self) -> Result<&Graph> {
        self.graph
            .as_ref()
            .ok_or_else(|| Error::Gpu("no decode graph captured".into()))
    }

    fn run_inner(
        &mut self,
        ids: &[u32],
        route: Route,
        keep: &[usize],
    ) -> Result<(Vec<PosStats>, Vec<Vec<f32>>)> {
        let mut stats = Vec::with_capacity(ids.len());
        let mut rows: Vec<Option<Vec<f32>>> = vec![None; keep.len()];
        match route {
            Route::Prefill => {
                let mut start = 0;
                while start < ids.len() {
                    let n = self.chunk.min(ids.len() - start);
                    self.put_tokens(&ids[start..start + n])?;
                    self.forward(start, n, self.prefill_attn(), Some((0, n)))?;
                    let am = self.argmax_rows(n, self.b.argmax)?;
                    self.collect(ids, start, n, &am, keep, &mut stats, &mut rows)?;
                    start += n;
                }
            }
            Route::Decode => match self.mode {
                DecodeMode::Eager => {
                    for p in 0..ids.len() {
                        self.put_tokens(&ids[p..p + 1])?;
                        self.forward(p, 1, Attn::Decode, Some((0, 1)))?;
                        let am = self.argmax_rows(1, self.b.argmax)?;
                        self.collect(ids, p, 1, &am, keep, &mut stats, &mut rows)?;
                    }
                }
                DecodeMode::Graph => {
                    // The graph reads the token and the position on the device, and leaves
                    // the argmax in tokens[0] and pos + 1: the host feeds the next token only.
                    self.set_pos(0)?;
                    for p in 0..ids.len() {
                        self.put_tokens(&ids[p..p + 1])?;
                        let g = self.graph()?;
                        // SAFETY: the graph launches on `self.arena`, frozen at load and
                        // dropped after the graph (field order).
                        unsafe { self.gpu.launch_graph(g) }?;
                        let am = self.read_argmax(1, self.b.tokens)?;
                        self.collect(ids, p, 1, &am, keep, &mut stats, &mut rows)?;
                    }
                }
            },
        }
        Ok((
            stats,
            rows.into_iter()
                .map(|r| r.expect("every kept position was visited"))
                .collect(),
        ))
    }

    /// Stats for every position of `ids`, teacher-forced.
    pub fn stats(&mut self, ids: &[u32], route: Route) -> Result<Vec<PosStats>> {
        Ok(self.run(ids, route, &[])?.0)
    }

    /// Full f32 logits at `positions` of `ids`, teacher-forced.
    pub fn logits_at(
        &mut self,
        ids: &[u32],
        positions: &[usize],
        route: Route,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(self.run(ids, route, positions)?.1)
    }

    /// `n` greedy tokens after `prompt`.
    pub fn greedy(&mut self, prompt: &[u32], n: usize) -> Result<Vec<u32>> {
        Ok(self.greedy_timed(prompt, n)?.0)
    }

    /// `n` greedy tokens after `prompt`, with the wall time (seconds from the call) at which
    /// each token was back on the host. Each token goes from the argmax kernel to the next
    /// step's embedding through device memory; the host copy is only for the caller.
    pub fn greedy_timed(&mut self, prompt: &[u32], n: usize) -> Result<(Vec<u32>, Vec<f64>)> {
        self.check_ids(prompt)?;
        if n == 0 || prompt.len() + n - 1 > self.max_ctx {
            return format_err(format!(
                "{} + {n} tokens exceed max_ctx {}",
                prompt.len(),
                self.max_ctx
            ));
        }
        let result = self.greedy_inner(prompt, n);
        self.scrub()?;
        result
    }

    /// `n` greedy tokens after `prompt`, and the f32 logits row that gave the last one (the
    /// prefill's last row when n = 1, else the last decode step's).
    pub fn greedy_with_last_logits(
        &mut self,
        prompt: &[u32],
        n: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        self.check_ids(prompt)?;
        if n == 0 || prompt.len() + n - 1 > self.max_ctx {
            return format_err(format!(
                "{} + {n} tokens exceed max_ctx {}",
                prompt.len(),
                self.max_ctx
            ));
        }
        let result = self.greedy_inner(prompt, n).and_then(|(tokens, _)| {
            self.gpu.sync()?;
            Ok((
                tokens,
                read_vec(&self.arena, self.b.logits, 0, self.cfg.vocab_size)?,
            ))
        });
        self.scrub()?;
        result
    }

    fn greedy_inner(&mut self, prompt: &[u32], n: usize) -> Result<(Vec<u32>, Vec<f64>)> {
        let t0 = Instant::now();
        let (mut out, mut times) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let mut start = 0;
        while start < prompt.len() {
            let len = self.chunk.min(prompt.len() - start);
            self.put_tokens(&prompt[start..start + len])?;
            let last = start + len == prompt.len();
            self.forward(
                start,
                len,
                self.prefill_attn(),
                last.then_some((len - 1, 1)),
            )?;
            start += len;
        }
        // The first token comes from the prefill; its argmax lands in tokens[0] for the steps.
        let tok = self.argmax_rows(1, self.b.tokens)?;
        out.push(tok[0]);
        times.push(t0.elapsed().as_secs_f64());
        match self.mode {
            DecodeMode::Eager => {
                for i in 1..n {
                    self.forward(prompt.len() + i - 1, 1, Attn::Decode, Some((0, 1)))?;
                    let tok = self.argmax_rows(1, self.b.tokens)?;
                    out.push(tok[0]);
                    times.push(t0.elapsed().as_secs_f64());
                }
            }
            DecodeMode::Graph => {
                // One position write per request; then per token: one graph launch, one
                // 4-byte copy into a host word, one event. The host runs up to RING - 1
                // steps ahead and collects tokens in order as their events complete.
                self.set_pos(prompt.len())?;
                let ring = self
                    .ring
                    .as_ref()
                    .ok_or_else(|| Error::Gpu("no host ring".into()))?;
                let vocab = self.cfg.vocab_size;
                let mut next = 1;
                let collect = |j: usize, out: &mut Vec<u32>, times: &mut Vec<f64>| -> Result<()> {
                    let t = ring.take(j % RING)?;
                    if t as usize >= vocab {
                        return Err(Error::Gpu(
                            "argmax: a logits row has no finite maximum".into(),
                        ));
                    }
                    out.push(t);
                    times.push(t0.elapsed().as_secs_f64());
                    Ok(())
                };
                for i in 1..n {
                    let g = self.graph()?;
                    // SAFETY: the graph launches on `self.arena`, frozen at load and dropped
                    // after the graph (field order).
                    unsafe { self.gpu.launch_graph(g) }?;
                    ring.fetch(&self.arena, self.b.tokens, 0, i % RING)?;
                    if i + 1 - next >= RING - 1 {
                        collect(next, &mut out, &mut times)?;
                        next += 1;
                    }
                }
                while next < n {
                    collect(next, &mut out, &mut times)?;
                    next += 1;
                }
            }
        }
        Ok((out, times))
    }
}

/// One request on the engine: the prompt is prefilled when the session opens; then each
/// `step` feeds the token the caller chose and computes the logits of the next position.
/// Client bytes (device and host) are zeroed by `finish`, or on drop, panics included.
pub struct Session<'e> {
    e: &'e mut Engine,
    pos: usize,
    open: bool,
    /// A step failed on the device: the logits are no longer the next position's.
    failed: bool,
}

impl<'e> Session<'e> {
    /// f32 logits of the last position, `vocab_size` values (empty if the wait for their copy
    /// fails, as after a failed step).
    pub fn logits(&self) -> &[f32] {
        self.e
            .host_logits
            .as_ref()
            .and_then(|h| h.values().ok())
            .unwrap_or(&[])
    }

    /// Tokens in the KV cache.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// One decode step on `token`. A token outside the vocabulary or a full context is refused
    /// with the session unchanged; a device failure ends the session's steps.
    pub fn step(&mut self, token: u32) -> Result<()> {
        if self.failed {
            return Err(Error::Gpu("a previous step failed".into()));
        }
        if token as usize >= self.e.cfg.vocab_size {
            return format_err(format!("token {token} outside the vocabulary"));
        }
        if self.pos >= self.e.max_ctx {
            return format_err(format!(
                "position {} at max_ctx {}",
                self.pos, self.e.max_ctx
            ));
        }
        if let Err(err) = self.e.session_step(token) {
            self.failed = true;
            return Err(err);
        }
        self.pos += 1;
        Ok(())
    }

    /// Zeroes the client bytes and reports a failure to do so (the engine is then poisoned).
    pub fn finish(mut self) -> Result<()> {
        self.open = false;
        self.e.scrub()
    }

    /// Greedy decoding on the device from the current position, for at most `max_tokens`
    /// tokens: the argmax kernel picks each token and the graph feeds it, so the host reads one
    /// word per token and no logits row. Refused when `max_tokens - 1` steps would pass
    /// max_ctx; the session's client bytes are zeroed on every way out, as for `Session`.
    /// `GREEDY_AHEAD` steps run ahead of the caller (`into_greedy_ahead`).
    pub fn into_greedy(self, max_tokens: usize) -> Result<Greedy<'e>> {
        self.into_greedy_ahead(max_tokens, GREEDY_AHEAD)
    }

    /// `into_greedy` with at most `ahead` steps (1 to `GREEDY_AHEAD_MAX`) launched past the
    /// caller's last token. A caller that stops (EOS, a stop sequence, a cancel) leaves those
    /// steps computed for nothing, and `finish` waits for them.
    pub fn into_greedy_ahead(self, max_tokens: usize, ahead: usize) -> Result<Greedy<'e>> {
        if !(1..=GREEDY_AHEAD_MAX).contains(&ahead) {
            return format_err(format!(
                "{ahead} greedy steps ahead: 1 to {GREEDY_AHEAD_MAX}"
            ));
        }
        if self.failed {
            return Err(Error::Gpu("a previous step failed".into()));
        }
        if max_tokens == 0 {
            return format_err("0 greedy tokens asked".to_string());
        }
        let room = self.e.max_ctx.saturating_sub(self.pos);
        if max_tokens - 1 > room {
            return format_err(format!(
                "{max_tokens} greedy tokens at position {} pass max_ctx {}",
                self.pos, self.e.max_ctx
            ));
        }
        let mut g = Greedy {
            s: self,
            max_tokens,
            ahead,
            taken: 0,
            launched: 0,
            first: 0,
        };
        let tokens = g.s.e.b.tokens;
        match g.s.e.argmax_rows(1, tokens) {
            Ok(v) => g.first = v[0],
            Err(err) => {
                g.s.failed = true;
                return Err(err);
            }
        }
        g.fill()?;
        Ok(g)
    }
}

/// A session decoding greedily on the device (`Session::into_greedy`). Per token after the
/// first: one graph launch, one 4-byte device-to-host copy into the greedy ring, one event; no
/// host-to-device copy. Up to `ahead` steps run ahead of the caller, never past the tokens it
/// asked for: steps computed after its last token are zeroed with the client bytes.
pub struct Greedy<'e> {
    s: Session<'e>,
    /// Tokens the caller may take, the current position's argmax included.
    max_tokens: usize,
    /// Steps launched past the caller's last token, at most (1 to `GREEDY_AHEAD_MAX`).
    ahead: usize,
    /// Tokens returned so far.
    taken: usize,
    /// Graph steps launched; step j writes token j into ring word j % RING.
    launched: usize,
    /// The argmax of the position the session was at, returned first.
    first: u32,
}

impl Greedy<'_> {
    /// The next greedy token, already fed to the model; refused past `max_tokens` or after a
    /// failed step.
    pub fn next_token(&mut self) -> Result<u32> {
        if self.s.failed {
            return Err(Error::Gpu("a previous step failed".into()));
        }
        if self.taken == self.max_tokens {
            return format_err(format!("all {} greedy tokens taken", self.max_tokens));
        }
        let t = if self.taken == 0 {
            self.first
        } else {
            let ring = self
                .s
                .e
                .ring
                .as_ref()
                .ok_or_else(|| Error::Gpu("no host ring".into()));
            match ring.and_then(|r| r.take(self.taken % RING)) {
                Ok(t) => t,
                Err(err) => {
                    self.s.failed = true;
                    return Err(err);
                }
            }
        };
        if t as usize >= self.s.e.cfg.vocab_size {
            self.s.failed = true;
            return Err(Error::Gpu(
                "argmax: a logits row has no finite maximum".into(),
            ));
        }
        self.taken += 1;
        self.fill()?;
        Ok(t)
    }

    /// Tokens returned so far.
    pub fn taken(&self) -> usize {
        self.taken
    }

    /// Tokens in the KV cache, the steps run ahead included.
    pub fn position(&self) -> usize {
        self.s.pos
    }

    /// Zeroes the client bytes and reports a failure to do so (the engine is then poisoned).
    pub fn finish(self) -> Result<()> {
        self.s.finish()
    }

    /// Launches steps until `ahead` tokens are in flight or every token asked is launched.
    /// Token j (j >= 1) is the step j's argmax: its word is taken before step j + RING - 1
    /// reuses it.
    fn fill(&mut self) -> Result<()> {
        while self.launched + 1 < self.max_tokens
            && self.launched - self.taken.saturating_sub(1) < self.ahead
        {
            let j = self.launched + 1;
            let e = &*self.s.e;
            let launched = e.graph().and_then(|g| {
                // SAFETY: the graph launches on `e.arena`, frozen at load and dropped after the
                // graph (field order).
                unsafe { e.gpu.launch_graph(g) }?;
                e.ring
                    .as_ref()
                    .ok_or_else(|| Error::Gpu("no host ring".into()))?
                    .fetch(&e.arena, e.b.tokens, 0, j % RING)
            });
            if let Err(err) = launched {
                self.s.failed = true;
                return Err(err);
            }
            self.launched = j;
            self.s.pos += 1;
        }
        Ok(())
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        if self.open {
            self.open = false;
            let _ = self.e.scrub();
        }
    }
}
