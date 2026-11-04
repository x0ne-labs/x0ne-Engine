//! Ministral 3, the text model (transformers 5.19 `modeling_ministral3.py`), built on
//! the Qwen3 fork's in-place KV cache ([`crate::qwen3`]), not on [`crate::decoder`],
//! whose cache grows by concatenation. What differs from Qwen3: no q/k norm, YaRN
//! RoPE whose cos/sin carry the mscale factor, and Llama 4's query scale after RoPE.

use std::collections::HashSet;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Embedding, Linear, RmsNorm, VarBuilder};

use crate::arch::{Architecture, DecoderConfig};
use crate::qwen3::{cpu_flash, fused, grouped, weight, Attention, LayerCache, Projections};

/// What a vision-language checkpoint holds besides the text model: never read.
pub const VISION_PREFIXES: &[&str] = &["vision_tower.", "multi_modal_projector."];

const LAYER_TENSORS: [&str; 9] = [
    "input_layernorm",
    "post_attention_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

/// The tensors the text model reads, in a fixed order.
pub fn text_tensor_names(cfg: &DecoderConfig) -> Vec<String> {
    let p = cfg.weights_prefix;
    let mut names = vec![
        format!("{p}model.embed_tokens.weight"),
        format!("{p}model.norm.weight"),
        format!("{p}lm_head.weight"),
    ];
    for l in 0..cfg.num_layers {
        for t in LAYER_TENSORS {
            names.push(format!("{p}model.layers.{l}.{t}.weight"));
        }
    }
    names
}

/// Every tensor is one the text model reads or, inside the envelope, the vision
/// tower's. Anything else (a quantization scale, a bias, a layer past the
/// config's) is refused: left unread, it would be weights the model was trained
/// with and this engine does not apply.
pub fn check_tensor_names<'a>(
    cfg: &DecoderConfig,
    names: impl IntoIterator<Item = &'a str>,
) -> anyhow::Result<()> {
    let want = text_tensor_names(cfg);
    let wanted: HashSet<&str> = want.iter().map(String::as_str).collect();
    let mut seen = HashSet::new();
    for n in names {
        if wanted.contains(n) {
            seen.insert(n);
        } else if cfg.weights_prefix.is_empty() || !VISION_PREFIXES.iter().any(|p| n.starts_with(p))
        {
            anyhow::bail!("ministral3: tensor {n:?} is not one this engine computes: refused");
        }
    }
    if let Some(missing) = want.iter().find(|n| !seen.contains(n.as_str())) {
        anyhow::bail!("ministral3: tensor {missing:?} is missing");
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct Rotary {
    sin: Tensor,
    cos: Tensor,
    /// Llama 4's query scale at each position, (max_positions,).
    query: Tensor,
}

impl Rotary {
    fn new(dtype: DType, cfg: &DecoderConfig, dev: &Device) -> Result<Self> {
        let (Some(rope), Some(scale)) = (cfg.ropes.first(), cfg.query_scale) else {
            candle_core::bail!("ministral3: no rope or no query scale")
        };
        let (inv_freq, factor) = rope.inv_freq(cfg.head_dim);
        let n = inv_freq.len();
        let max = cfg.max_position_embeddings;
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), dev)?;
        let t = Tensor::arange(0u32, max as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        // transformers multiplies cos and sin by the factor in f32, then casts.
        let scaled = |x: Tensor| -> Result<Tensor> {
            if factor == 1.0 {
                Ok(x)
            } else {
                x * f64::from(factor)
            }
        };
        let query: Vec<f32> = (0..max).map(|p| scale.at(p)).collect();
        Ok(Self {
            sin: scaled(freqs.sin()?)?.to_dtype(dtype)?,
            cos: scaled(freqs.cos()?)?.to_dtype(dtype)?,
            query: Tensor::from_vec(query, max, dev)?.to_dtype(dtype)?,
        })
    }

    /// `x`: (B, H, L, D), contiguous.
    fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    /// `q`: (B, H, L, D) after RoPE; each position times its scale.
    fn scale_queries(&self, q: &Tensor, offset: usize) -> Result<Tensor> {
        let l = q.dim(2)?;
        let s = self.query.narrow(0, offset, l)?.reshape((1, 1, l, 1))?;
        q.broadcast_mul(&s)
    }
}

#[derive(Debug, Clone)]
enum Qkv {
    Fused(Linear),
    Split(Linear, Linear, Linear),
}

#[derive(Debug, Clone)]
enum GateUp {
    Fused(Linear),
    Split(Linear, Linear),
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_up: GateUp,
    down_proj: Linear,
    intermediate: usize,
}

impl Mlp {
    fn new(cfg: &DecoderConfig, vb: VarBuilder, p: Projections) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let gate = weight(&vb.pp("gate_proj"), h, i, false)?;
        let up = weight(&vb.pp("up_proj"), h, i, false)?;
        let gate_up = match p {
            Projections::Fused => GateUp::Fused(fused(vec![gate, up])?),
            Projections::Split => GateUp::Split(Linear::new(gate.0, None), Linear::new(up.0, None)),
        };
        let (down, _) = weight(&vb.pp("down_proj"), i, h, false)?;
        Ok(Self {
            gate_up,
            down_proj: Linear::new(down, None),
            intermediate: i,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (gate, up) = match &self.gate_up {
            GateUp::Fused(l) => {
                let gu = l.forward(x)?;
                let last = gu.rank() - 1;
                (
                    gu.narrow(last, 0, self.intermediate)?,
                    gu.narrow(last, self.intermediate, self.intermediate)?,
                )
            }
            GateUp::Split(g, u) => (g.forward(x)?, u.forward(x)?),
        };
        (gate.apply(&Activation::Silu)? * up)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct SelfAttention {
    qkv: Qkv,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cache: LayerCache,
}

/// One layer's attention probabilities, (B, H, L, T): what the tests compare
/// with transformers' `output_attentions`.
type Kept<'a> = Option<&'a mut Vec<Tensor>>;

impl SelfAttention {
    fn new(cfg: &DecoderConfig, vb: VarBuilder, p: Projections) -> Result<Self> {
        let (d, h, kvh) = (cfg.head_dim, cfg.num_heads, cfg.num_kv_heads);
        if kvh == 0 || !h.is_multiple_of(kvh) {
            candle_core::bail!("{h} attention heads are not a multiple of {kvh} KV heads");
        }
        let q = weight(&vb.pp("q_proj"), cfg.hidden_size, h * d, false)?;
        let k = weight(&vb.pp("k_proj"), cfg.hidden_size, kvh * d, false)?;
        let v = weight(&vb.pp("v_proj"), cfg.hidden_size, kvh * d, false)?;
        let qkv = match p {
            Projections::Fused => Qkv::Fused(fused(vec![q, k, v])?),
            Projections::Split => Qkv::Split(
                Linear::new(q.0, None),
                Linear::new(k.0, None),
                Linear::new(v.0, None),
            ),
        };
        let o = weight(&vb.pp("o_proj"), h * d, cfg.hidden_size, false)?;
        Ok(Self {
            qkv,
            o_proj: Linear::new(o.0, None),
            num_heads: h,
            num_kv_heads: kvh,
            head_dim: d,
            cache: LayerCache::default(),
        })
    }

    /// (B, L, heads * D) -> (B, heads, L, D), contiguous.
    fn heads(&self, x: Tensor, b: usize, l: usize, n: usize) -> Result<Tensor> {
        x.reshape((b, l, n, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
        rotary: &Rotary,
        attention: Attention,
        reserve: usize,
        max: usize,
        kept: Kept<'_>,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let (h, kvh, d) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let (q, k, v) = match &self.qkv {
            Qkv::Fused(p) => {
                let qkv = p.forward(x)?.reshape((b, l, h + 2 * kvh, d))?;
                let part = |at: usize, n: usize| -> Result<Tensor> {
                    qkv.narrow(2, at, n)?.transpose(1, 2)?.contiguous()
                };
                (part(0, h)?, part(h, kvh)?, part(h + kvh, kvh)?)
            }
            Qkv::Split(qp, kp, vp) => (
                self.heads(qp.forward(x)?, b, l, h)?,
                self.heads(kp.forward(x)?, b, l, kvh)?,
                self.heads(vp.forward(x)?, b, l, kvh)?,
            ),
        };
        let q = rotary.scale_queries(&rotary.apply(&q, offset)?, offset)?;
        let k = rotary.apply(&k, offset)?;
        let (k, v) = self.cache.append(&k, &v, reserve, max)?;
        let scale = 1.0 / (d as f64).sqrt();
        if let Some(kept) = kept {
            kept.push(probabilities(&q, &k, mask, scale)?);
        }
        let ctx = if attention == Attention::Auto && x.device().is_cpu() {
            cpu_flash(&q, &k, &v, scale as f32, offset)?
        } else {
            grouped(&q, &k, &v, mask, scale)?.transpose(1, 2)?
        };
        ctx.reshape((b, l, h * d))?.apply(&self.o_proj)
    }
}

/// softmax(q kᵀ · scale + mask), (B, H, L, T), head h on KV head h / (H / KVH).
fn probabilities(q: &Tensor, k: &Tensor, mask: Option<&Tensor>, scale: f64) -> Result<Tensor> {
    let (b, h, l, d) = q.dims4()?;
    let (kvh, t) = (k.dim(1)?, k.dim(2)?);
    let rep = h / kvh;
    let q = q.reshape((b, kvh, rep * l, d))?;
    let mut scores = (q.matmul(&k.t()?)? * scale)?.reshape((b, kvh, rep, l, t))?;
    if let Some(m) = mask {
        scores = scores.broadcast_add(m)?;
    }
    candle_nn::ops::softmax_last_dim(&scores.reshape((b, h, l, t))?)
}

#[derive(Debug, Clone)]
struct Layer {
    attn: SelfAttention,
    mlp: Mlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

/// Ministral 3 for causal LM: `forward` returns the last position's logits, (B, 1, vocab).
#[derive(Debug, Clone)]
pub struct Ministral3 {
    embed_tokens: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary: Rotary,
    device: Device,
    dtype: DType,
    max_positions: usize,
    attention: Attention,
    reserve: usize,
}

impl Ministral3 {
    /// Reads the text model's tensors, under `cfg.weights_prefix`; the
    /// checkpoint's own `lm_head` always (arch.rs, `parse_ministral3`).
    pub fn new(cfg: &DecoderConfig, vb: VarBuilder, p: Projections) -> Result<Self> {
        if cfg.arch != Architecture::Ministral3 {
            candle_core::bail!("{} is not ministral3", cfg.arch.name());
        }
        if cfg.ropes.len() != 1 || cfg.layers.iter().any(|l| l.window.is_some() || l.rope != 0) {
            candle_core::bail!("ministral3: one rope and full attention on every layer expected");
        }
        if vb.dtype() == DType::F64 {
            candle_core::bail!("ministral3 does not support f64; load weights as f32 or bf16");
        }
        let vb = match cfg.weights_prefix.strip_suffix('.') {
            Some(prefix) if !prefix.is_empty() => vb.pp(prefix),
            _ => vb,
        };
        let (hidden, eps) = (cfg.hidden_size, cfg.rms_norm_eps);
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, hidden, vb.pp("model.embed_tokens"))?;
        let rotary = Rotary::new(vb.dtype(), cfg, vb.device())?;
        let vb_l = vb.pp("model.layers");
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let vb = vb_l.pp(i);
            layers.push(Layer {
                attn: SelfAttention::new(cfg, vb.pp("self_attn"), p)?,
                mlp: Mlp::new(cfg, vb.pp("mlp"), p)?,
                ln1: candle_nn::rms_norm(hidden, eps, vb.pp("input_layernorm"))?,
                ln2: candle_nn::rms_norm(hidden, eps, vb.pp("post_attention_layernorm"))?,
            });
        }
        let lm_head = Linear::new(
            vb.pp("lm_head").get((cfg.vocab_size, hidden), "weight")?,
            None,
        );
        Ok(Self {
            embed_tokens,
            layers,
            norm: candle_nn::rms_norm(hidden, eps, vb.pp("model.norm"))?,
            lm_head,
            rotary,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            max_positions: cfg.max_position_embeddings,
            attention: Attention::Auto,
            reserve: 0,
        })
    }

    pub fn set_attention(&mut self, a: Attention) {
        self.attention = a;
    }

    /// Positions the next request may fill: the cache is allocated once at that size.
    pub fn reserve(&mut self, tokens: usize) {
        self.reserve = tokens.min(self.max_positions);
    }

    /// Zeroes every layer's keys and values, then drops them and the
    /// reservation; dropped even when a zeroing fails, whose error is returned.
    pub fn clear_kv_cache(&mut self) -> Result<()> {
        let mut zeroed = Ok(());
        for l in &mut self.layers {
            zeroed = zeroed.and(l.attn.cache.clear());
        }
        self.reserve = 0;
        zeroed
    }

    /// Keeps the first `len` positions of every layer: the next forward starts there.
    pub fn truncate_kv_cache(&mut self, len: usize) {
        for l in &mut self.layers {
            l.attn.cache.truncate(len);
        }
    }

    /// Positions in the cache, the same in every layer.
    pub fn kv_len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.attn.cache.len)
    }

    /// (len, capacity) of layer `i`'s cache, and its K buffer: tests look for copies.
    pub fn layer_cache(&self, i: usize) -> Option<(usize, usize, Tensor)> {
        let c = &self.layers.get(i)?.attn.cache;
        Some((c.len, c.capacity(), c.k.clone()?))
    }

    /// (L, offset + L) additive mask (0 or -inf), on the model's device.
    fn causal_mask(&self, l: usize, offset: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..l)
            .flat_map(|i| {
                (0..(l + offset)).map(move |j| {
                    if j <= i + offset {
                        0.
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (l, l + offset), &self.device)?.to_dtype(self.dtype)
    }

    /// `input`: (B, L) token ids at positions `offset..offset + L`.
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        self.run(input, offset, false, None)
    }

    /// Logits of every position, (B, L, vocab), and each layer's attention
    /// probabilities when `kept` is given.
    fn run(
        &mut self,
        input: &Tensor,
        offset: usize,
        all: bool,
        mut kept: Kept<'_>,
    ) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        let matmul = self.attention == Attention::Matmul || !self.device.is_cpu() || kept.is_some();
        let mask = if matmul && l > 1 {
            Some(self.causal_mask(l, offset)?)
        } else {
            None
        };
        for layer in &mut self.layers {
            let a = layer.attn.forward(
                &layer.ln1.forward(&h)?,
                mask.as_ref(),
                offset,
                &self.rotary,
                self.attention,
                self.reserve,
                self.max_positions,
                kept.as_deref_mut(),
            )?;
            h = (h + a)?;
            let m = layer.mlp.forward(&layer.ln2.forward(&h)?)?;
            h = (h + m)?;
        }
        let h = if all { h } else { h.narrow(1, l - 1, 1)? };
        self.norm.forward(&h)?.apply(&self.lm_head)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        cfg: DecoderConfig,
        weights: Vec<u8>,
        reference: serde_json::Value,
    }

    fn fixture(name: &str) -> Fixture {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/arch")
            .join(name);
        let read = |f: &str| std::fs::read(dir.join(f)).unwrap();
        Fixture {
            cfg: DecoderConfig::parse(&read("config.json")).unwrap(),
            weights: read("model.safetensors"),
            reference: serde_json::from_slice(&read("reference.json")).unwrap(),
        }
    }

    impl Fixture {
        fn model(&self, p: Projections, a: Attention) -> Ministral3 {
            let vb = VarBuilder::from_slice_safetensors(&self.weights, DType::F32, &Device::Cpu)
                .unwrap();
            let mut m = Ministral3::new(&self.cfg, vb, p).unwrap();
            m.set_attention(a);
            m
        }
        fn ids(&self, key: &str) -> Vec<u32> {
            serde_json::from_value(self.reference[key].clone()).unwrap()
        }
        fn floats(v: &serde_json::Value) -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        }
    }

    fn ids(t: &[u32]) -> Tensor {
        Tensor::new(t, &Device::Cpu).unwrap().unsqueeze(0).unwrap()
    }

    fn argmax(t: &Tensor) -> u32 {
        t.flatten_all()
            .unwrap()
            .argmax(0)
            .unwrap()
            .to_scalar::<u32>()
            .unwrap()
    }

    /// Worst |got - want| / (1 + |want|), the measure of decoder.rs's tests.
    fn worst(got: &Tensor, want: &[f32]) -> f32 {
        let got: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got.len(), want.len());
        got.iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs() / (1.0 + w.abs()))
            .fold(0f32, f32::max)
    }

    /// The fixture (tests/arch/ministral3): transformers 5.19 `Ministral3ForCausalLM`,
    /// GQA 3/1, YaRN with mscale 2 / mscale_all_dim 0.5 (cos/sin scale 1.1076), query
    /// scale from position 4. A failure means this fork does not compute Ministral 3.
    /// Measured 2026-10-28: worst 7.0e-6 (logits), 5.5e-6 (attention); 0.24 on the
    /// logits without the query scale, while the greedy tokens stayed the same.
    #[test]
    fn the_fork_matches_transformers_on_the_forward_pass() {
        let f = fixture("ministral3");
        let mut worst_logits = 0f32;
        let mut worst_attention = 0f32;
        for p in [Projections::Fused, Projections::Split] {
            let mut m = f.model(p, Attention::Matmul);
            let mut kept = Vec::new();
            let logits = m
                .run(&ids(&f.ids("ids")), 0, true, Some(&mut kept))
                .unwrap();
            worst_logits =
                worst_logits.max(worst(&logits, &Fixture::floats(&f.reference["logits"])));
            let want = f.reference["attentions"].as_array().unwrap();
            assert_eq!(kept.len(), want.len());
            for (got, want) in kept.iter().zip(want) {
                worst_attention = worst_attention.max(worst(got, &Fixture::floats(want)));
            }
        }
        eprintln!("ministral3: worst logits {worst_logits}, attention {worst_attention}");
        assert!(
            worst_logits < 1e-4,
            "logits: worst relative error {worst_logits}"
        );
        assert!(
            worst_attention < 1e-5,
            "attention: worst error {worst_attention}"
        );
    }

    /// Greedy tokens of transformers' `generate`, prefill in one pass or token by
    /// token, through both attention kernels: the KV cache and the offsets.
    #[test]
    fn the_fork_generates_what_transformers_generates() {
        let f = fixture("ministral3");
        let (prompt, want) = (f.ids("prompt"), f.ids("generated"));
        for p in [Projections::Fused, Projections::Split] {
            for a in [Attention::Auto, Attention::Matmul] {
                for chunk in [prompt.len(), 1] {
                    let mut m = f.model(p, a);
                    m.reserve(prompt.len() + want.len());
                    let mut logits = None;
                    for (i, c) in prompt.chunks(chunk).enumerate() {
                        logits = Some(m.forward(&ids(c), i * chunk).unwrap());
                    }
                    let mut logits = logits.unwrap();
                    let mut got = Vec::new();
                    for step in 0..want.len() {
                        let next = argmax(&logits);
                        got.push(next);
                        logits = m.forward(&ids(&[next]), prompt.len() + step).unwrap();
                    }
                    assert_eq!(got, want, "{p:?}, {a:?}, chunk {chunk}");
                }
            }
        }
    }

    /// The real RoPE and query scale of the 14B (theta 1e9, YaRN 16 from 16384,
    /// beta 0.1) on a tiny model, 16448 positions: the logits of the last 64, all
    /// past 16384, and the greedy continuation. transformers' logits without the
    /// query scale differ from these by `without_query_scale_worst` (0.18): a fork
    /// without it fails here. Measured 2026-10-28: worst 4.9e-5 (Auto), 1.2e-4 (Matmul).
    #[test]
    fn past_16384_positions_the_fork_gives_transformers_logits() {
        let f = fixture("ministral3-long");
        let len = 16_384 + 64;
        let all: Vec<u32> = (0..len as u32)
            .map(|i| (i * i * 7 + i * 13 + 5) % 40)
            .collect();
        let want = Fixture::floats(&f.reference["logits_last_64"]);
        let new = f.ids("generated");
        let unscaled = f.reference["without_query_scale_worst"].as_f64().unwrap();
        assert!(unscaled > 0.1, "{unscaled}");
        for a in [Attention::Auto, Attention::Matmul] {
            let mut m = f.model(Projections::Fused, a);
            m.reserve(len + new.len());
            for (i, c) in all[..len - 64].chunks(512).enumerate() {
                m.forward(&ids(c), i * 512).unwrap();
            }
            let logits = m.run(&ids(&all[len - 64..]), len - 64, true, None).unwrap();
            let w = worst(&logits, &want);
            eprintln!("ministral3-long {a:?}: worst relative error {w}");
            assert!(w < 1e-3, "{a:?}: worst relative error {w}");
            let mut next = argmax(&logits.narrow(1, 63, 1).unwrap());
            let mut got = vec![next];
            for step in 0..new.len() - 1 {
                let l = m.forward(&ids(&[next]), len + step).unwrap();
                next = argmax(&l);
                got.push(next);
            }
            assert_eq!(got, new, "{a:?}");
        }
    }

    /// A pulled release against transformers on the CPU, same ids, then
    /// logits and greedy tokens, in X0NE_INFER_MINISTRAL3_DTYPE (f32 or f16).
    /// Procedure and results: docs/learn-manual-test.md.
    #[test]
    #[ignore = "needs a pulled Ministral 3 and its transformers reference"]
    fn real_weights_match_transformers() {
        use crate::arch::Architecture;
        use crate::prompt::{encode, text_only, Prompt};
        use std::time::Instant;
        let var = |v: &str| std::env::var(v).unwrap_or_else(|_| panic!("{v}"));
        let dir = std::path::PathBuf::from(var("X0NE_INFER_MINISTRAL3_DIR"));
        let read = |f: &str| crate::fsread::read_model_file(&dir, f).unwrap();
        let (config, tokenizer, index) = (
            read("config.json"),
            read("tokenizer.json"),
            read(crate::shards::INDEX_FILE),
        );
        let map: serde_json::Value = serde_json::from_slice(&index).unwrap();
        let mut shards: Vec<String> = map["weight_map"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        shards.sort();
        shards.dedup();
        let maps: Vec<memmap2::Mmap> = shards
            .iter()
            .map(|s| crate::fsread::map_model_file(&dir, s).unwrap())
            .collect();
        let files: Vec<(&str, &[u8])> = shards
            .iter()
            .zip(&maps)
            .map(|(s, m)| (s.as_str(), &m[..]))
            .collect();
        let t = Instant::now();
        let weights = crate::backend::WeightFiles {
            index_json: Some(&index),
            files: files.clone(),
        };
        let digest = crate::candle_backend::served_digest(&config, &tokenizer, &weights).unwrap();
        eprintln!(
            "served digest {digest} ({:.0} s)",
            t.elapsed().as_secs_f32()
        );
        let default = Architecture::Ministral3.default_system(&digest);
        assert!(default.is_some(), "not a pinned release: {digest}");

        let reference =
            candle_core::safetensors::load(var("X0NE_INFER_MINISTRAL3_REFERENCE"), &Device::Cpu)
                .unwrap();
        let meta = {
            let bytes = std::fs::read(var("X0NE_INFER_MINISTRAL3_REFERENCE")).unwrap();
            let (_, m) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
            m.metadata().clone().unwrap()
        };
        let tok = text_only(tokenizers::Tokenizer::from_bytes(&tokenizer).unwrap());
        let chats: Vec<Vec<crate::server::ChatMessage>> =
            serde_json::from_str(&meta["conversations"]).unwrap();
        let ref_ids = |i: usize| -> Vec<u32> {
            reference[&format!("p{i}.ids")]
                .to_dtype(DType::U32)
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        for (i, chat) in chats.iter().enumerate() {
            let p = Architecture::Ministral3.chat_prompt_with(chat, default);
            assert_eq!(encode(&tok, &p).unwrap(), ref_ids(i), "prompt {i}");
        }
        let raw = chats.len();
        assert_eq!(
            encode(&tok, &Prompt::text(meta["raw"].clone())).unwrap(),
            ref_ids(raw)
        );
        // Optional "<chats.jsonl>:<rows.jsonl>" (id, messages; id, prompt_ids):
        // the script's --extra rows, rendered by the engine to the same ids.
        if let Ok(pair) = std::env::var("X0NE_INFER_MINISTRAL3_EXTRA") {
            let lines = |p: &str| -> Vec<serde_json::Value> {
                let text = std::fs::read_to_string(p).unwrap();
                text.lines()
                    .map(|l| serde_json::from_str(l).unwrap())
                    .collect()
            };
            let (chats, rows) = pair.split_once(':').expect("<chats>:<rows>");
            let (chats, rows) = (lines(chats), lines(rows));
            assert_eq!(chats.len(), rows.len());
            for chat in &chats {
                let row = rows.iter().find(|r| r["id"] == chat["id"]).unwrap();
                let messages: Vec<crate::server::ChatMessage> =
                    serde_json::from_value(chat["messages"].clone()).unwrap();
                let want: Vec<u32> = serde_json::from_value(row["prompt_ids"].clone()).unwrap();
                let p = Architecture::Ministral3.chat_prompt_with(&messages, default);
                assert_eq!(encode(&tok, &p).unwrap(), want, "{}", chat["id"]);
            }
            eprintln!("extra: {} conversations, the same ids", rows.len());
        }

        let cfg = DecoderConfig::parse(&config).unwrap();
        let names: Vec<String> = crate::candle_shards::ShardedSlices::new(Some(&index), &files)
            .unwrap()
            .names()
            .map(String::from)
            .collect();
        check_tensor_names(&cfg, names.iter().map(String::as_str)).unwrap();
        let slices = crate::candle_shards::ShardedSlices::new(Some(&index), &files).unwrap();
        let dtype = match std::env::var("X0NE_INFER_MINISTRAL3_DTYPE").as_deref() {
            Ok("f16") => DType::F16,
            Ok("f32") | Err(_) => DType::F32,
            Ok(other) => panic!("X0NE_INFER_MINISTRAL3_DTYPE={other}: f32 or f16"),
        };
        let vb = VarBuilder::from_backend(Box::new(slices), dtype, Device::Cpu);
        let t = Instant::now();
        let mut m = Ministral3::new(&cfg, vb, Projections::Fused).unwrap();
        eprintln!(
            "built in {:.0} s, {dtype:?}; reference {}, transformers {}",
            t.elapsed().as_secs_f32(),
            meta.get("dtype").map_or("bf16", String::as_str),
            meta["transformers"]
        );
        let mut prompts = 0;
        while reference.contains_key(&format!("p{prompts}.ids")) {
            prompts += 1;
        }
        let mut wrong = Vec::new();
        for i in 0..prompts {
            let all = ref_ids(i);
            let want_logits = &reference[&format!("p{i}.logits")];
            let want: Vec<u32> = reference[&format!("p{i}.greedy")]
                .to_dtype(DType::U32)
                .unwrap()
                .to_vec1()
                .unwrap();
            let (len, keep) = (all.len(), want_logits.dim(0).unwrap());
            m.clear_kv_cache().unwrap();
            m.reserve(len + want.len());
            let t = Instant::now();
            for (c, chunk) in all[..len - keep].chunks(512).enumerate() {
                m.forward(&ids(chunk), c * 512).unwrap();
            }
            let logits = m
                .run(&ids(&all[len - keep..]), len - keep, true, None)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .squeeze(0)
                .unwrap();
            let prefill = t.elapsed().as_secs_f32();
            let diff = (&logits - want_logits)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let scale = want_logits
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let top: Vec<u32> = logits.argmax(1).unwrap().to_vec1().unwrap();
            let ref_top: Vec<u32> = want_logits.argmax(1).unwrap().to_vec1().unwrap();
            let agree = top.iter().zip(&ref_top).filter(|(a, b)| a == b).count();
            let mut next = top[keep - 1];
            let mut got = vec![next];
            let t = Instant::now();
            for step in 0..want.len() - 1 {
                let l = m.forward(&ids(&[next]), len + step).unwrap();
                next = argmax(&l.to_dtype(DType::F32).unwrap());
                got.push(next);
            }
            let decode = t.elapsed().as_secs_f32();
            let same = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
            eprintln!(
                "p{i}: {len} ids; last {keep} positions: max |diff| {diff:.4} (max |logit| \
                 {scale:.2}), top-1 {agree}/{keep}; greedy: {same}/{} tokens equal; prefill \
                 {prefill:.1} s ({:.1} tok/s), decode {:.2} tok/s; transformers: prefill {} s, \
                 {} tokens in {} s",
                want.len(),
                len as f32 / prefill,
                (want.len() - 1) as f32 / decode,
                meta[&format!("p{i}.prefill_s")],
                want.len(),
                meta[&format!("p{i}.generate_s")],
            );
            eprintln!(
                "p{i}: engine {:?}",
                tok.decode(&got, false).unwrap_or_default()
            );
            if same < want.len() || agree * 100 < keep * 95 {
                wrong.push(format!(
                    "p{i}: greedy {same}/{}, top-1 {agree}/{keep}",
                    want.len()
                ));
            }
        }
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        let hwm = status.lines().find(|l| l.starts_with("VmHWM")).unwrap();
        eprintln!("{hwm}");
        assert!(wrong.is_empty(), "{wrong:?}");
    }

    fn buffer(t: &Tensor) -> usize {
        let (s, _) = t.storage_and_layout();
        match &*s {
            candle_core::Storage::Cpu(candle_core::CpuStorage::F32(v)) => v.as_ptr() as usize,
            _ => panic!("f32 CPU storage expected"),
        }
    }

    /// A decode step writes its token's K and V into the buffer reserved for the
    /// request. A failure means each token copies the whole cache again.
    #[test]
    fn a_decode_step_writes_the_kv_cache_in_place() {
        let f = fixture("ministral3");
        let mut m = f.model(Projections::Fused, Attention::Matmul);
        m.reserve(30);
        m.forward(&ids(&f.ids("prompt")), 0).unwrap();
        let first: Vec<usize> = (0..4)
            .map(|l| buffer(&m.layer_cache(l).unwrap().2))
            .collect();
        for step in 0..10 {
            m.forward(&ids(&[1]), 6 + step).unwrap();
            for (l, &b) in first.iter().enumerate() {
                let (len, cap, k) = m.layer_cache(l).unwrap();
                assert_eq!((len, cap), (6 + step + 1, 30), "layer {l}");
                assert_eq!(
                    buffer(&k),
                    b,
                    "layer {l}, step {step}: the cache was copied"
                );
            }
        }
        let left = m.layer_cache(0).unwrap().2;
        m.clear_kv_cache().unwrap();
        assert!(m.layer_cache(0).is_none());
        let left: Vec<f32> = left.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            left.iter().all(|x| *x == 0.0),
            "zeroed before it was dropped"
        );
    }

    /// The published checkpoints hold the vision tower next to the text model;
    /// anything else is refused, and so is a missing text tensor.
    #[test]
    fn only_the_text_model_and_the_vision_tower_are_accepted() {
        let real = DecoderConfig::parse(include_bytes!(
            "../tests/arch/real/ministral3-14b-instruct-2512.json"
        ))
        .unwrap();
        let text = text_tensor_names(&real);
        assert_eq!(text.len(), 363, "40 layers of 9, embed, norm, lm_head");
        let vision = [
            "vision_tower.transformer.layers.0.attention.q_proj.weight",
            "multi_modal_projector.linear_1.weight",
        ];
        let with = |extra: &[&str], drop: Option<&str>| -> anyhow::Result<()> {
            let names = text
                .iter()
                .map(String::as_str)
                .filter(|n| Some(*n) != drop)
                .chain(extra.iter().copied());
            check_tensor_names(&real, names)
        };
        with(&vision, None).unwrap();
        for (extra, drop, needle) in [
            (
                &["language_model.model.layers.0.self_attn.q_proj.weight_scale"][..],
                None,
                "weight_scale",
            ),
            (
                &["language_model.model.layers.40.mlp.up_proj.weight"][..],
                None,
                "layers.40",
            ),
            (&["lm_head.weight"][..], None, "\"lm_head.weight\""),
            (&[][..], Some("language_model.lm_head.weight"), "missing"),
        ] {
            let err = with(extra, drop).unwrap_err().to_string();
            assert!(err.contains(needle), "{err}");
        }
        // Without the envelope there is no vision tower to accept.
        let small = fixture("ministral3").cfg;
        let names = text_tensor_names(&small);
        check_tensor_names(&small, names.iter().map(String::as_str)).unwrap();
        let err = check_tensor_names(
            &small,
            names
                .iter()
                .map(String::as_str)
                .chain(vision.iter().copied()),
        )
        .unwrap_err();
        assert!(err.to_string().contains("vision_tower"), "{err}");
    }
}
