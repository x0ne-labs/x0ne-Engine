//! Qwen3 dense, forked from candle-transformers 0.11.0 `src/models/qwen3.rs` (NOTICE).
//! Same weights, config and arithmetic; the KV cache is written in place, grouped-query
//! attention reads K/V once per KV head, and q/k/v and gate/up can be one matmul each.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Embedding, Linear, RmsNorm, VarBuilder};
use candle_transformers::models::qwen3::Config;

/// How the projections are held. `Fused` concatenates q/k/v and gate/up at build time:
/// one matmul instead of three and two. `Split` keeps the loaded tensors by handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projections {
    Fused,
    Split,
}

/// Which kernel computes attention. `Auto` is candle's choice: its CPU flash kernel on
/// the CPU, `Matmul` elsewhere. `Matmul` on the CPU runs the GPU path (tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attention {
    Auto,
    Matmul,
}

#[derive(Debug, Clone)]
struct Rotary {
    sin: Tensor,
    cos: Tensor,
}

impl Rotary {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// `x`: (B, H, L, D), contiguous.
    fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }
}

/// `in_dim -> out_dim` with the weight read as `(out_dim, in_dim)`, like `candle_nn::linear_b`.
pub(crate) fn weight(
    vb: &VarBuilder,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    let w = vb.get((out_dim, in_dim), "weight")?;
    let b = if bias {
        Some(vb.get(out_dim, "bias")?)
    } else {
        None
    };
    Ok((w, b))
}

/// Rows of the parts' weights one after the other: the output of the fused linear is
/// the parts' outputs side by side, in order.
pub(crate) fn fused(parts: Vec<(Tensor, Option<Tensor>)>) -> Result<Linear> {
    let ws: Vec<&Tensor> = parts.iter().map(|(w, _)| w).collect();
    let w = Tensor::cat(&ws, 0)?;
    let bs: Option<Vec<&Tensor>> = parts.iter().map(|(_, b)| b.as_ref()).collect();
    let b = match bs {
        Some(bs) => Some(Tensor::cat(&bs, 0)?),
        None => None,
    };
    Ok(Linear::new(w, b))
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
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder, p: Projections) -> Result<Self> {
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
            act_fn: cfg.hidden_act,
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
        (gate.apply(&self.act_fn)? * up)?.apply(&self.down_proj)
    }
}

/// One layer's keys and values, (B, KVH, capacity, D) each, written in place. Allocated
/// at the first append, zeroed then dropped by `clear`. Between the turns of one
/// conversation it is kept, cut back to the prefix the next prompt shares (`truncate`).
#[derive(Debug, Clone, Default)]
pub(crate) struct LayerCache {
    pub(crate) k: Option<Tensor>,
    v: Option<Tensor>,
    pub(crate) len: usize,
    /// Positions written since the buffers were allocated: what `clear` zeroes.
    written: usize,
}

impl LayerCache {
    pub(crate) fn capacity(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.dims()[2])
    }

    /// Appends `k`, `v` (B, KVH, L, D) at the current length; returns views of the first
    /// `len + L` positions. `reserve` sizes the first allocation of a request.
    pub(crate) fn append(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        reserve: usize,
        max: usize,
    ) -> Result<(Tensor, Tensor)> {
        let l = k.dim(2)?;
        let need = self.len + l;
        if need > self.capacity() {
            // A reservation that covers the need is the size: a conversation's
            // kept cache grows to its turn's, bounded by the served context.
            let cap = if reserve >= need {
                reserve
            } else {
                need.max(self.capacity() * 2).min(max).max(need)
            };
            self.k = Some(grown(self.k.as_ref(), k, self.len, self.written, cap)?);
            self.v = Some(grown(self.v.as_ref(), v, self.len, self.written, cap)?);
            self.written = self.len;
        }
        let (kb, vb) = match (&self.k, &self.v) {
            (Some(kb), Some(vb)) => (kb, vb),
            _ => candle_core::bail!("KV cache not allocated"),
        };
        kb.slice_set(&k.contiguous()?, 2, self.len)?;
        vb.slice_set(&v.contiguous()?, 2, self.len)?;
        self.len = need;
        self.written = self.written.max(need);
        Ok((kb.narrow(2, 0, need)?, vb.narrow(2, 0, need)?))
    }

    /// SECURITY: zeroes every position written, then drops the buffers; dropped even
    /// when zeroing fails, and the error returned.
    pub(crate) fn clear(&mut self) -> Result<()> {
        let k = zeroed(self.k.as_ref(), self.written);
        let v = zeroed(self.v.as_ref(), self.written);
        *self = Self::default();
        k.and(v)
    }

    /// Keeps the first `len` positions, the next append writing from there. What lies
    /// past them is never read again, and `clear` zeroes it.
    pub(crate) fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }
}

/// Zeroes the first `written` positions of `t`, in place.
fn zeroed(t: Option<&Tensor>, written: usize) -> Result<()> {
    match t {
        Some(t) if written > 0 => t.narrow(2, 0, written)?.zero_set(),
        _ => Ok(()),
    }
}

/// A zeroed (B, KVH, cap, D) buffer holding the first `len` positions of `old`. The
/// `written` positions of `old`, and the copy made on the way, are zeroed.
fn grown(
    old: Option<&Tensor>,
    like: &Tensor,
    len: usize,
    written: usize,
    cap: usize,
) -> Result<Tensor> {
    let (b, h, _, d) = like.dims4()?;
    let buf = Tensor::zeros((b, h, cap, d), like.dtype(), like.device())?;
    if let Some(old) = old {
        if len > 0 {
            let kept = old.narrow(2, 0, len)?.contiguous()?;
            buf.slice_set(&kept, 2, 0)?;
            kept.zero_set()?;
        }
        zeroed(Some(old), written)?;
    }
    Ok(buf)
}

#[derive(Debug, Clone)]
struct SelfAttention {
    qkv: Qkv,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    cache: LayerCache,
}

impl SelfAttention {
    fn new(cfg: &Config, vb: VarBuilder, p: Projections) -> Result<Self> {
        if cfg.use_sliding_window {
            candle_core::bail!("sliding window is not supported")
        }
        let (d, h, kvh) = (
            cfg.head_dim,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
        );
        if kvh == 0 || !h.is_multiple_of(kvh) {
            candle_core::bail!("{h} attention heads are not a multiple of {kvh} KV heads");
        }
        let bias = cfg.attention_bias;
        let q = weight(&vb.pp("q_proj"), cfg.hidden_size, h * d, bias)?;
        let k = weight(&vb.pp("k_proj"), cfg.hidden_size, kvh * d, bias)?;
        let v = weight(&vb.pp("v_proj"), cfg.hidden_size, kvh * d, bias)?;
        let qkv = match p {
            Projections::Fused => Qkv::Fused(fused(vec![q, k, v])?),
            Projections::Split => Qkv::Split(
                Linear::new(q.0, q.1),
                Linear::new(k.0, k.1),
                Linear::new(v.0, v.1),
            ),
        };
        let o = weight(&vb.pp("o_proj"), h * d, cfg.hidden_size, bias)?;
        Ok(Self {
            qkv,
            o_proj: Linear::new(o.0, o.1),
            q_norm: candle_nn::rms_norm(d, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: candle_nn::rms_norm(d, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: h,
            num_kv_heads: kvh,
            head_dim: d,
            // candle's note: config.hidden_size is not always heads * head_dim.
            hidden_size: h * d,
            cache: LayerCache::default(),
        })
    }

    /// (B, L, heads * D) -> (B, heads, L, D), contiguous; a view when L == 1.
    fn heads(&self, x: Tensor, b: usize, l: usize, n: usize) -> Result<Tensor> {
        x.reshape((b, l, n, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// Per-head RMSNorm over D, as candle: on (B * heads * L, D).
    fn norm(&self, norm: &RmsNorm, x: &Tensor) -> Result<Tensor> {
        let (b, n, l, d) = x.dims4()?;
        norm.forward(&x.flatten(0, 2)?)?.reshape((b, n, l, d))
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
        let q = rotary.apply(&self.norm(&self.q_norm, &q)?, offset)?;
        let k = rotary.apply(&self.norm(&self.k_norm, &k)?, offset)?;
        let (k, v) = self.cache.append(&k, &v, reserve, max)?;
        let scale = 1.0 / (d as f64).sqrt();
        let ctx = if attention == Attention::Auto && x.device().is_cpu() {
            cpu_flash(&q, &k, &v, scale as f32, offset)?
        } else {
            grouped(&q, &k, &v, mask, scale)?.transpose(1, 2)?
        };
        ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj)
    }
}

/// q (B, H, L, D) against k, v (B, KVH, T, D): the H / KVH query heads of a group are rows
/// of one matmul, so K and V are read once per KV head. Head h uses KV head h / (H / KVH),
/// as `repeat_kv`. Returns (B, H, L, D).
pub(crate) fn grouped(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let (b, h, l, d) = q.dims4()?;
    let (kvh, t) = (k.dim(1)?, k.dim(2)?);
    let rep = h / kvh;
    let q = q.reshape((b, kvh, rep * l, d))?;
    let mut scores = (q.matmul(&k.t()?)? * scale)?;
    if let Some(m) = mask {
        scores = scores
            .reshape((b, kvh, rep, l, t))?
            .broadcast_add(m)?
            .reshape((b, kvh, rep * l, t))?;
    }
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    probs.matmul(v)?.reshape((b, h, l, d))
}

/// candle's CPU path, unchanged: its flash kernel on (B, S, H, D), f32 inside.
/// Returns (B, L, H, D).
pub(crate) fn cpu_flash(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    offset: usize,
) -> Result<Tensor> {
    use candle_nn::attention::{flash_attn, AttnMask};
    let q = q.transpose(1, 2)?.contiguous()?;
    let k = k.transpose(1, 2)?.contiguous()?;
    let v = v.transpose(1, 2)?.contiguous()?;
    let mask = || AttnMask::causal_with_offset(offset);
    let ctx = match q.dtype() {
        DType::F32 => flash_attn::<f32>(&q, &k, &v, scale, mask(), None, None)?,
        other => flash_attn::<f32>(
            &q.to_dtype(DType::F32)?,
            &k.to_dtype(DType::F32)?,
            &v.to_dtype(DType::F32)?,
            scale,
            mask(),
            None,
            None,
        )?
        .to_dtype(other)?,
    };
    ctx.transpose(1, 2)
}

#[derive(Debug, Clone)]
struct Layer {
    attn: SelfAttention,
    mlp: Mlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

/// Qwen3 for causal LM: `forward` returns the last position's logits, (B, 1, vocab).
#[derive(Debug, Clone)]
pub struct Qwen3 {
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

impl Qwen3 {
    pub fn new(cfg: &Config, vb: VarBuilder, p: Projections) -> Result<Self> {
        if vb.dtype() == DType::F64 {
            candle_core::bail!(
                "Qwen3 does not support f64; load weights as f32 or bf16 (CPU) or f16/bf16 (GPU)"
            );
        }
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Rotary::new(vb.dtype(), cfg, vb.device())?;
        let vb_l = vb.pp("model.layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let vb = vb_l.pp(i);
            layers.push(Layer {
                attn: SelfAttention::new(cfg, vb.pp("self_attn"), p)?,
                mlp: Mlp::new(cfg, vb.pp("mlp"), p)?,
                ln1: candle_nn::rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("input_layernorm"),
                )?,
                ln2: candle_nn::rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("post_attention_layernorm"),
                )?,
            });
        }
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            Linear::new(
                vb.pp("lm_head")
                    .get((cfg.vocab_size, cfg.hidden_size), "weight")?,
                None,
            )
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
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

    /// (L, offset + L) additive mask, as candle's (0 or -inf), on the model's device.
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
        let (_, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        let matmul = self.attention == Attention::Matmul || !self.device.is_cpu();
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
            )?;
            h = (h + a)?;
            let m = layer.mlp.forward(&layer.ln2.forward(&h)?)?;
            h = (h + m)?;
        }
        // RmsNorm is per position: the last one alone gives candle's values.
        self.norm
            .forward(&h.narrow(1, l - 1, 1)?)?
            .apply(&self.lm_head)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::models::qwen3::ModelForCausalLM;
    use std::collections::HashMap;

    /// splitmix64: the same weights on every machine, no RNG crate.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
        fn tensor(&mut self, shape: &[usize], scale: f32, center: f32) -> Tensor {
            let n = shape.iter().product();
            let v: Vec<f32> = (0..n).map(|_| center + scale * self.next()).collect();
            Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
        }
    }

    struct Shape {
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        hidden: usize,
        bias: bool,
        tie: bool,
    }

    fn config(s: &Shape) -> Config {
        serde_json::from_value(serde_json::json!({
            "vocab_size": 61, "hidden_size": s.hidden, "intermediate_size": 40,
            "num_hidden_layers": 3, "num_attention_heads": s.heads, "head_dim": s.head_dim,
            "attention_bias": s.bias, "num_key_value_heads": s.kv_heads,
            "max_position_embeddings": 96, "sliding_window": null, "max_window_layers": 3,
            "tie_word_embeddings": s.tie, "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
            "use_sliding_window": false, "hidden_act": "silu"
        }))
        .unwrap()
    }

    /// Every tensor candle's Qwen3 reads: name, shape, spread and center of its random
    /// values (weights in ±0.3, norms around 1).
    fn layout(c: &Config) -> Vec<(String, Vec<usize>, f32, f32)> {
        let (h, d) = (c.hidden_size, c.head_dim);
        let (q, kv, i) = (
            c.num_attention_heads * d,
            c.num_key_value_heads * d,
            c.intermediate_size,
        );
        let mut w = vec![
            (
                "model.embed_tokens.weight".to_string(),
                vec![c.vocab_size, h],
                1.0,
                0.0,
            ),
            ("model.norm.weight".to_string(), vec![h], 0.2, 1.0),
        ];
        if !c.tie_word_embeddings {
            w.push(("lm_head.weight".into(), vec![c.vocab_size, h], 0.3, 0.0));
        }
        for l in 0..c.num_hidden_layers {
            let mut put = |n: &str, shape: &[usize], scale: f32, center: f32| {
                w.push((
                    format!("model.layers.{l}.{n}"),
                    shape.to_vec(),
                    scale,
                    center,
                ));
            };
            for (n, out, inp) in [("q", q, h), ("k", kv, h), ("v", kv, h), ("o", h, q)] {
                put(&format!("self_attn.{n}_proj.weight"), &[out, inp], 0.3, 0.0);
                if c.attention_bias {
                    put(&format!("self_attn.{n}_proj.bias"), &[out], 0.3, 0.0);
                }
            }
            put("self_attn.q_norm.weight", &[d], 0.2, 1.0);
            put("self_attn.k_norm.weight", &[d], 0.2, 1.0);
            put("mlp.gate_proj.weight", &[i, h], 0.3, 0.0);
            put("mlp.up_proj.weight", &[i, h], 0.3, 0.0);
            put("mlp.down_proj.weight", &[h, i], 0.3, 0.0);
            put("input_layernorm.weight", &[h], 0.2, 1.0);
            put("post_attention_layernorm.weight", &[h], 0.2, 1.0);
        }
        w
    }

    fn weights(c: &Config, seed: u64) -> HashMap<String, Tensor> {
        let mut r = Rng(seed);
        layout(c)
            .into_iter()
            .map(|(n, shape, scale, center)| (n, r.tensor(&shape, scale, center)))
            .collect()
    }

    /// Worst |a - b| / (1 + |b|), the measure of decoder.rs's tests against transformers.
    fn worst(a: &Tensor, b: &Tensor) -> f32 {
        let a: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
            .fold(0f32, f32::max)
    }

    fn argmax(t: &Tensor) -> u32 {
        t.flatten_all()
            .unwrap()
            .argmax(0)
            .unwrap()
            .to_scalar::<u32>()
            .unwrap()
    }

    fn ids(t: &[u32]) -> Tensor {
        Tensor::new(t, &Device::Cpu).unwrap().unsqueeze(0).unwrap()
    }

    const SHAPES: [Shape; 3] = [
        // Two KV heads of two query heads: a wrong head-to-group map changes the logits.
        Shape {
            heads: 4,
            kv_heads: 2,
            head_dim: 16,
            hidden: 24,
            bias: false,
            tie: false,
        },
        Shape {
            heads: 6,
            kv_heads: 3,
            head_dim: 8,
            hidden: 48,
            bias: true,
            tie: true,
        },
        Shape {
            heads: 4,
            kv_heads: 4,
            head_dim: 8,
            hidden: 32,
            bias: false,
            tie: true,
        },
    ];

    /// f32 on the CPU. Measured 2026-10-18: worst 3.8e-6 over every case below.
    const TOLERANCE: f32 = 1e-5;
    const PROMPT: usize = 37;
    const STEPS: usize = 12;

    /// The fork against candle-transformers 0.11.0's Qwen3, same tensors, f32 on the CPU:
    /// logits after a prefill in one pass and in chunks, then at each greedy step, and the
    /// greedy tokens. A failure means the fork no longer computes candle's model.
    #[test]
    fn the_fork_gives_candles_logits_and_greedy_tokens() {
        let mut overall = 0f32;
        for (n, s) in SHAPES.iter().enumerate() {
            let cfg = config(s);
            let w = weights(&cfg, 7 + n as u64);
            let vb = || VarBuilder::from_tensors(w.clone(), DType::F32, &Device::Cpu);
            let prompt: Vec<u32> = (0..PROMPT as u32).map(|i| (i * 17 + 3) % 61).collect();

            let mut candle = ModelForCausalLM::new(&cfg, vb()).unwrap();
            let mut want = vec![candle.forward(&ids(&prompt), 0).unwrap()];
            let mut greedy = Vec::new();
            for step in 0..STEPS {
                let next = argmax(&want[step]);
                greedy.push(next);
                want.push(candle.forward(&ids(&[next]), PROMPT + step).unwrap());
            }

            for p in [Projections::Fused, Projections::Split] {
                for a in [Attention::Auto, Attention::Matmul] {
                    for chunk in [PROMPT, 5, 1] {
                        let mut fork = Qwen3::new(&cfg, vb(), p).unwrap();
                        fork.set_attention(a);
                        let reserve = if chunk == 5 { 0 } else { PROMPT + STEPS };
                        let case =
                            format!("shape {n}, {p:?}, {a:?}, chunk {chunk}, reserve {reserve}");
                        let run = |fork: &mut Qwen3| -> Vec<Tensor> {
                            fork.reserve(reserve);
                            let mut got = Vec::new();
                            let mut last = None;
                            for (i, c) in prompt.chunks(chunk).enumerate() {
                                last = Some(fork.forward(&ids(c), i * chunk).unwrap());
                            }
                            got.push(last.unwrap());
                            for (step, &t) in greedy.iter().enumerate() {
                                got.push(fork.forward(&ids(&[t]), PROMPT + step).unwrap());
                            }
                            fork.clear_kv_cache().unwrap();
                            got
                        };
                        let got = run(&mut fork);
                        for (step, (g, w)) in got.iter().zip(&want).enumerate() {
                            let e = worst(g, w);
                            overall = overall.max(e);
                            assert!(
                                e < TOLERANCE,
                                "{case}, step {step}: worst relative error {e}"
                            );
                            if step < STEPS {
                                assert_eq!(argmax(g), greedy[step], "{case}, step {step}");
                            }
                            // Same split projections and kernel as candle: the same bits.
                            if p == Projections::Split && a == Attention::Auto && chunk == PROMPT {
                                assert_eq!(e, 0.0, "{case}, step {step}");
                            }
                        }
                        // Cleared: the same request answers with the same bits.
                        let again = run(&mut fork);
                        for (g, a2) in got.iter().zip(&again) {
                            assert_eq!(worst(a2, g), 0.0, "{case}: after clear_kv_cache");
                        }
                        assert!(fork.layer_cache(0).is_none(), "{case}: cache freed");
                    }
                }
            }
        }
        eprintln!("worst relative error against candle: {overall:e} (tolerance {TOLERANCE:e})");
    }

    /// transformers' greedy `generate` on the qwen3 fixture (3 heads, 1 KV head), as
    /// decoder.rs checks it, through the GPU path run on the CPU.
    #[test]
    fn the_fork_generates_what_transformers_generates() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/arch/qwen3");
        let cfg = crate::candle_backend::parse_qwen3_config(
            &std::fs::read(dir.join("config.json")).unwrap(),
        )
        .unwrap();
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("reference.json")).unwrap()).unwrap();
        let weights = std::fs::read(dir.join("model.safetensors")).unwrap();
        let list = |k: &str| -> Vec<u32> { serde_json::from_value(reference[k].clone()).unwrap() };
        let (prompt, want) = (list("prompt"), list("generated"));
        for p in [Projections::Fused, Projections::Split] {
            let vb =
                VarBuilder::from_slice_safetensors(&weights, DType::F32, &Device::Cpu).unwrap();
            let mut fork = Qwen3::new(&cfg, vb, p).unwrap();
            fork.set_attention(Attention::Matmul);
            fork.reserve(prompt.len() + want.len());
            let mut logits = fork.forward(&ids(&prompt), 0).unwrap();
            let mut got = Vec::new();
            for step in 0..want.len() {
                let next = argmax(&logits);
                got.push(next);
                logits = fork.forward(&ids(&[next]), prompt.len() + step).unwrap();
            }
            assert_eq!(got, want, "{p:?}");
        }
    }

    fn buffer(t: &Tensor) -> usize {
        let (s, _) = t.storage_and_layout();
        match &*s {
            candle_core::Storage::Cpu(candle_core::CpuStorage::F32(v)) => v.as_ptr() as usize,
            _ => panic!("f32 CPU storage expected"),
        }
    }

    /// A decode step writes its token's K and V into the buffer reserved for the
    /// request, it does not build a new cache. A failure means each token copies the
    /// whole cache again (candle's ConcatKvCache: the slope of decode time with context).
    #[test]
    fn a_decode_step_writes_the_kv_cache_in_place() {
        let cfg = config(&SHAPES[0]);
        let vb = VarBuilder::from_tensors(weights(&cfg, 7), DType::F32, &Device::Cpu);
        let mut fork = Qwen3::new(&cfg, vb, Projections::Fused).unwrap();
        fork.set_attention(Attention::Matmul);
        fork.reserve(PROMPT + STEPS);
        let prompt: Vec<u32> = (0..PROMPT as u32).collect();
        fork.forward(&ids(&prompt), 0).unwrap();
        let first: Vec<usize> = (0..3)
            .map(|l| buffer(&fork.layer_cache(l).unwrap().2))
            .collect();
        for step in 0..STEPS {
            fork.forward(&ids(&[1]), PROMPT + step).unwrap();
            for (l, &b) in first.iter().enumerate() {
                let (len, cap, k) = fork.layer_cache(l).unwrap();
                assert_eq!(len, PROMPT + step + 1);
                assert_eq!(cap, PROMPT + STEPS, "layer {l}: one allocation per request");
                assert_eq!(
                    buffer(&k),
                    b,
                    "layer {l}, step {step}: the cache was copied"
                );
            }
        }
    }

    /// Layer `l`'s K and V buffers, sharing their storage: a probe of what is left in them.
    fn kv(fork: &Qwen3, l: usize) -> (Tensor, Tensor) {
        let c = &fork.layers[l].attn.cache;
        (c.k.clone().unwrap(), c.v.clone().unwrap())
    }

    fn zeroed(t: &Tensor) -> bool {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|x| *x == 0.0)
    }

    /// SECURITY: clearing the cache writes zeros over the keys and values a
    /// request wrote, in the buffers themselves, before it drops them; growing
    /// it zeroes the buffer it leaves. A failure: keys and values freed as
    /// they were, to the allocator the next request draws from.
    #[test]
    fn clearing_and_growing_the_kv_cache_zero_the_buffers_left() {
        let cfg = config(&SHAPES[0]);
        let vb = VarBuilder::from_tensors(weights(&cfg, 7), DType::F32, &Device::Cpu);
        let mut fork = Qwen3::new(&cfg, vb, Projections::Fused).unwrap();
        let layers = cfg.num_hidden_layers;
        let prompt: Vec<u32> = (0..PROMPT as u32).collect();
        fork.reserve(PROMPT);
        fork.forward(&ids(&prompt), 0).unwrap();
        let small: Vec<_> = (0..layers).map(|l| kv(&fork, l)).collect();
        assert!(
            small.iter().all(|(k, v)| !zeroed(k) && !zeroed(v)),
            "written"
        );
        // One decode step past the reservation: the cache grows into a new buffer.
        fork.forward(&ids(&[1]), PROMPT).unwrap();
        for (l, (k, v)) in small.iter().enumerate() {
            assert!(
                zeroed(k) && zeroed(v),
                "layer {l}: the buffer left by growth"
            );
        }
        let grown: Vec<_> = (0..layers).map(|l| kv(&fork, l)).collect();
        fork.clear_kv_cache().unwrap();
        for (l, (k, v)) in grown.iter().enumerate() {
            assert!(zeroed(k) && zeroed(v), "layer {l}: the buffer cleared");
        }
        assert!(fork.layer_cache(0).is_none(), "dropped");
    }

    /// A conversation's next turn that outgrows the cache kept grows it to
    /// the turn's reservation, which the served context bounds; decoding
    /// past a reservation still doubles. A failure: a kept cache doubled
    /// past the context the serving budget was sized for.
    #[test]
    fn a_reserved_turn_grows_the_cache_to_its_reservation() {
        let cfg = config(&SHAPES[0]);
        let vb = VarBuilder::from_tensors(weights(&cfg, 7), DType::F32, &Device::Cpu);
        let mut fork = Qwen3::new(&cfg, vb, Projections::Fused).unwrap();
        let prompt: Vec<u32> = (0..PROMPT as u32).collect();
        fork.reserve(PROMPT);
        fork.forward(&ids(&prompt), 0).unwrap();
        fork.reserve(PROMPT + 3);
        fork.forward(&ids(&[1, 2, 3]), PROMPT).unwrap();
        for l in 0..cfg.num_hidden_layers {
            let (len, cap, _) = fork.layer_cache(l).unwrap();
            assert_eq!((len, cap), (PROMPT + 3, PROMPT + 3), "layer {l}");
        }
        fork.forward(&ids(&[4]), PROMPT + 3).unwrap();
        assert_eq!(fork.layer_cache(0).unwrap().1, 2 * (PROMPT + 3));
    }

    /// A conversation's next turn: the cache truncated to the prefix the two
    /// prompts share, the rest computed at its offset, gives the logits of
    /// the whole prompt from token 0. Clearing then zeroes the positions past
    /// the truncation too. A failure: a reused prefix that answers otherwise
    /// than a cold one, or keys left past it.
    #[test]
    fn a_truncated_cache_continues_as_a_cold_prefill() {
        for (n, s) in SHAPES.iter().enumerate() {
            let cfg = config(s);
            let w = weights(&cfg, 7 + n as u64);
            let vb = || VarBuilder::from_tensors(w.clone(), DType::F32, &Device::Cpu);
            let first: Vec<u32> = (0..PROMPT as u32).map(|i| (i * 17 + 3) % 61).collect();
            let shared = 20;
            let mut second = first[..shared].to_vec();
            second.extend((0..10u32).map(|i| (i * 5 + 1) % 61));
            let mut cold = Qwen3::new(&cfg, vb(), Projections::Fused).unwrap();
            let want = cold.forward(&ids(&second), 0).unwrap();
            let mut warm = Qwen3::new(&cfg, vb(), Projections::Fused).unwrap();
            warm.forward(&ids(&first), 0).unwrap();
            warm.forward(&ids(&[5]), PROMPT).unwrap();
            assert_eq!(warm.kv_len(), PROMPT + 1);
            warm.truncate_kv_cache(shared);
            assert_eq!(warm.kv_len(), shared, "shape {n}");
            let got = warm.forward(&ids(&second[shared..]), shared).unwrap();
            let e = worst(&got, &want);
            assert!(e < TOLERANCE, "shape {n}: worst relative error {e}");
            assert_eq!(argmax(&got), argmax(&want), "shape {n}");
            let left = kv(&warm, 0);
            warm.clear_kv_cache().unwrap();
            assert!(
                zeroed(&left.0) && zeroed(&left.1),
                "shape {n}: past the truncation"
            );
        }
    }
}
