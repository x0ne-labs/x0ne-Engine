//! A decoder-only transformer written out in elementary tensor operations, for
//! every family of [`crate::arch`]: the candle reference of the tests computes
//! the OLMo families with it (candle 0.11 has `olmo2` without sliding window
//! or YaRN, and no `olmo3`); Qwen3 stays on candle's implementation.
//!
//! Checked against Hugging Face transformers on tiny random models
//! (`tests/arch/`, `scripts/make-arch-fixtures.py`) and against candle's
//! `qwen3` and `olmo2`.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;

use crate::arch::{Architecture, DecoderConfig, NormPlacement, QkNorm};

type Result<T> = candle_core::Result<T>;

/// A linear projection `y = x·Wᵀ + b`, `W` of shape (out, in).
#[derive(Clone)]
pub struct Proj {
    pub w: Tensor,
    pub b: Option<Tensor>,
}

impl Proj {
    fn load(vb: &VarBuilder, out: usize, inp: usize, bias: bool, name: &str) -> Result<Self> {
        let w = vb.get((out, inp), &format!("{name}.weight"))?;
        let b = if bias {
            Some(vb.get(out, &format!("{name}.bias"))?)
        } else {
            None
        };
        Ok(Self { w, b })
    }

    /// `x`: (tokens, in).
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.matmul(&self.w.t()?)?;
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

/// One block. `norm_a`/`norm_b` are Qwen3's `input_layernorm` /
/// `post_attention_layernorm`, OLMo's `post_attention_layernorm` /
/// `post_feedforward_layernorm`.
#[derive(Clone)]
pub struct LayerWeights {
    pub q: Proj,
    pub k: Proj,
    pub v: Proj,
    pub o: Proj,
    pub gate: Proj,
    pub up: Proj,
    pub down: Proj,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub norm_a: Tensor,
    pub norm_b: Tensor,
}

impl LayerWeights {
    fn load(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let (h, hd, nh, nkv) = (
            cfg.hidden_size,
            cfg.head_dim,
            cfg.num_heads,
            cfg.num_kv_heads,
        );
        let b = cfg.attention_bias;
        let a = vb.pp("self_attn");
        let m = vb.pp("mlp");
        let (qn, kn) = match cfg.qk_norm {
            QkNorm::PerHead => (hd, hd),
            QkNorm::FullProjection => (nh * hd, nkv * hd),
            QkNorm::None => candle_core::bail!("a layer without q/k norm is not implemented here"),
        };
        let (na, nb) = match cfg.norm {
            NormPlacement::Pre => ("input_layernorm", "post_attention_layernorm"),
            NormPlacement::Post => ("post_attention_layernorm", "post_feedforward_layernorm"),
        };
        Ok(Self {
            q: Proj::load(&a, nh * hd, h, b, "q_proj")?,
            k: Proj::load(&a, nkv * hd, h, b, "k_proj")?,
            v: Proj::load(&a, nkv * hd, h, b, "v_proj")?,
            o: Proj::load(&a, h, nh * hd, b, "o_proj")?,
            gate: Proj::load(&m, cfg.intermediate_size, h, false, "gate_proj")?,
            up: Proj::load(&m, cfg.intermediate_size, h, false, "up_proj")?,
            down: Proj::load(&m, h, cfg.intermediate_size, false, "down_proj")?,
            q_norm: a.get(qn, "q_norm.weight")?,
            k_norm: a.get(kn, "k_norm.weight")?,
            norm_a: vb.get(h, &format!("{na}.weight"))?,
            norm_b: vb.get(h, &format!("{nb}.weight"))?,
        })
    }

    /// Every tensor cast to `dtype` (a no-op when it already is).
    fn cast(&self, dtype: DType) -> Result<Self> {
        let p = |p: &Proj| -> Result<Proj> {
            Ok(Proj {
                w: p.w.to_dtype(dtype)?,
                b: p.b.as_ref().map(|b| b.to_dtype(dtype)).transpose()?,
            })
        };
        Ok(Self {
            q: p(&self.q)?,
            k: p(&self.k)?,
            v: p(&self.v)?,
            o: p(&self.o)?,
            gate: p(&self.gate)?,
            up: p(&self.up)?,
            down: p(&self.down)?,
            q_norm: self.q_norm.to_dtype(dtype)?,
            k_norm: self.k_norm.to_dtype(dtype)?,
            norm_a: self.norm_a.to_dtype(dtype)?,
            norm_b: self.norm_b.to_dtype(dtype)?,
        })
    }
}

/// RMSNorm over the last dimension, computed in f32 (as transformers does).
fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    x.broadcast_div(&rms)?
        .broadcast_mul(&w.to_dtype(DType::F32)?)?
        .to_dtype(dtype)
}

/// Softmax over the last dimension in f32. The max is a constant shift.
fn softmax(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?.detach();
    let e = x.broadcast_sub(&max)?.exp()?;
    e.broadcast_div(&e.sum_keepdim(D::Minus1)?)
}

/// Rotate-half RoPE. `x`: (heads, tokens, head_dim); `cos`/`sin`: (tokens, head_dim/2).
fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let a = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
    let b = (x2.broadcast_mul(cos)? + x1.broadcast_mul(sin)?)?;
    Tensor::cat(&[a, b], D::Minus1)
}

/// Keys and values of past positions, per layer, after RoPE: (kv_heads, t, head_dim).
/// `Clone` shares tensors by handle; `layer` replaces them, never writes in place.
#[derive(Default, Clone)]
pub struct KvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
}

impl KvCache {
    pub fn clear(&mut self) {
        self.layers.clear();
    }
    pub fn len(&self) -> usize {
        self.layers
            .first()
            .and_then(|l| l.as_ref())
            .map_or(0, |(k, _)| k.dim(1).unwrap_or(0))
    }
}

pub struct Decoder {
    pub cfg: DecoderConfig,
    embed: Tensor,
    layers: Vec<LayerWeights>,
    norm: Tensor,
    lm_head: Tensor,
    /// Per rope of `cfg.ropes`: inverse frequencies (f32, on device) and cos/sin scale.
    rope_freqs: Vec<(Tensor, f32)>,
    device: Device,
}

/// What [`Decoder::trace`] returns: everything in f32, detached.
pub struct Trace {
    /// (tokens, vocab).
    pub logits: Tensor,
    /// `hidden[0]` is the embedding output, `hidden[l+1]` the output of layer `l`.
    pub hidden: Vec<Tensor>,
    /// Per layer, (heads, tokens, tokens), when asked for.
    pub attention: Vec<Tensor>,
}

impl Decoder {
    pub fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        if cfg.arch == Architecture::Ministral3 {
            candle_core::bail!(
                "ministral3 is computed by crate::ministral3, not by this decoder \
                 (no query scale, no weights prefix here)"
            );
        }
        let device = vb.device().clone();
        let m = vb.pp("model");
        let embed = m.get((cfg.vocab_size, cfg.hidden_size), "embed_tokens.weight")?;
        let layers = (0..cfg.num_layers)
            .map(|i| LayerWeights::load(cfg, m.pp("layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let norm = m.get(cfg.hidden_size, "norm.weight")?;
        let lm_head = if cfg.tie_word_embeddings {
            embed.clone()
        } else {
            vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")?
        };
        let rope_freqs = cfg
            .ropes
            .iter()
            .map(|r| {
                let (inv, scale) = r.inv_freq(cfg.head_dim);
                Ok((Tensor::new(inv.as_slice(), &device)?, scale))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cfg: cfg.clone(),
            embed,
            layers,
            norm,
            lm_head,
            rope_freqs,
            device,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn cos_sin(&self, rope: usize, offset: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let (inv, scale) = &self.rope_freqs[rope];
        let pos: Vec<f32> = (offset..offset + len).map(|p| p as f32).collect();
        let pos = Tensor::new(pos.as_slice(), &self.device)?.reshape((len, 1))?;
        let freqs = pos.broadcast_mul(&inv.reshape((1, inv.dim(0)?))?)?;
        Ok((
            (freqs.cos()? * (*scale as f64))?,
            (freqs.sin()? * (*scale as f64))?,
        ))
    }

    /// One block. `x`: (tokens, hidden) at positions `offset..`.
    fn layer(
        &self,
        index: usize,
        w: &LayerWeights,
        x: &Tensor,
        offset: usize,
        cache: Option<&mut Option<(Tensor, Tensor)>>,
        attention: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let cfg = &self.cfg;
        let (l, _) = x.dims2()?;
        let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let eps = cfg.rms_norm_eps;
        let spec = &cfg.layers[index];
        let dtype = x.dtype();

        let h = match cfg.norm {
            NormPlacement::Pre => rms_norm(x, &w.norm_a, eps)?,
            NormPlacement::Post => x.clone(),
        };
        let (mut q, mut k, v) = (w.q.apply(&h)?, w.k.apply(&h)?, w.v.apply(&h)?);
        if cfg.qk_norm == QkNorm::FullProjection {
            q = rms_norm(&q, &w.q_norm, eps)?;
            k = rms_norm(&k, &w.k_norm, eps)?;
        }
        let heads = |t: Tensor, n: usize| -> Result<Tensor> {
            t.reshape((l, n, hd))?.transpose(0, 1)?.contiguous()
        };
        let (mut q, mut k, v) = (heads(q, nh)?, heads(k, nkv)?, heads(v, nkv)?);
        if cfg.qk_norm == QkNorm::PerHead {
            q = rms_norm(&q, &w.q_norm, eps)?;
            k = rms_norm(&k, &w.k_norm, eps)?;
        }
        let (cos, sin) = self.cos_sin(spec.rope, offset, l)?;
        let (cos, sin) = (cos.to_dtype(dtype)?, sin.to_dtype(dtype)?);
        let q = rope(&q, &cos, &sin)?;
        let k = rope(&k, &cos, &sin)?;

        let (k, v) = match cache {
            Some(slot) => {
                let kv = match slot.take() {
                    Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 1)?, Tensor::cat(&[&pv, &v], 1)?),
                    None => (k, v),
                };
                *slot = Some(kv.clone());
                kv
            }
            None => (k, v),
        };
        let t = k.dim(1)?;
        let groups = nh / nkv;
        let repeat = |x: &Tensor| -> Result<Tensor> {
            if groups == 1 {
                return Ok(x.clone());
            }
            x.unsqueeze(1)?
                .broadcast_as((nkv, groups, t, hd))?
                .reshape((nh, t, hd))
        };
        let (k, v) = (repeat(&k)?, repeat(&v)?);

        let scale = 1.0 / (hd as f64).sqrt();
        let mut scores = (q
            .to_dtype(DType::F32)?
            .matmul(&k.to_dtype(DType::F32)?.t()?)?
            * scale)?;
        let needs_mask = l > 1 || spec.window.is_some_and(|w| t > w);
        if needs_mask {
            let mask: Vec<f32> = (0..l)
                .flat_map(|i| {
                    let qpos = offset + i;
                    (0..t).map(move |j| {
                        let causal = j <= qpos;
                        let in_window = spec.window.is_none_or(|w| j + w > qpos);
                        if causal && in_window {
                            0.0
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect();
            let mask = Tensor::from_vec(mask, (1, l, t), &self.device)?;
            scores = scores.broadcast_add(&mask)?;
        }
        let probs = softmax(&scores)?;
        if let Some(out) = attention {
            out.push(probs.detach());
        }
        let ctx = probs
            .to_dtype(dtype)?
            .matmul(&v)?
            .transpose(0, 1)?
            .reshape((l, nh * hd))?;
        let attn = w.o.apply(&ctx)?;
        let x = match cfg.norm {
            NormPlacement::Pre => (x + attn)?,
            NormPlacement::Post => (x + rms_norm(&attn, &w.norm_a, eps)?)?,
        };
        let h = match cfg.norm {
            NormPlacement::Pre => rms_norm(&x, &w.norm_b, eps)?,
            NormPlacement::Post => x.clone(),
        };
        let m = w
            .down
            .apply(&(w.gate.apply(&h)?.silu()? * w.up.apply(&h)?)?)?;
        match cfg.norm {
            NormPlacement::Pre => x + m,
            NormPlacement::Post => x + rms_norm(&m, &w.norm_b, eps)?,
        }
    }

    fn embed(&self, ids: &[u32], dtype: DType) -> Result<Tensor> {
        let ids = Tensor::new(ids, &self.device)?;
        self.embed.index_select(&ids, 0)?.to_dtype(dtype)
    }

    /// Generation step: logits of the LAST position, shape (1, 1, vocab) like
    /// candle's models, with the KV cache advanced.
    pub fn forward(&self, ids: &[u32], offset: usize, cache: &mut KvCache) -> Result<Tensor> {
        let x = self.residual(ids, offset, cache)?;
        let last = x.narrow(0, ids.len() - 1, 1)?;
        let h = rms_norm(&last, &self.norm, self.cfg.rms_norm_eps)?;
        h.matmul(&self.lm_head.t()?)?.unsqueeze(0)
    }

    fn residual(&self, ids: &[u32], offset: usize, cache: &mut KvCache) -> Result<Tensor> {
        let mut x = self.embed(ids, self.embed.dtype())?;
        cache.layers.resize(self.layers.len(), None);
        for (i, w) in self.layers.iter().enumerate() {
            x = self.layer(i, w, &x, offset, Some(&mut cache.layers[i]), None)?;
        }
        Ok(x)
    }

    /// Whole-text forward in f32, no cache, keeping the residual stream and,
    /// if asked, every attention map: what the conformance tests compare.
    pub fn trace(&self, ids: &[u32], attention: bool) -> Result<Trace> {
        let mut x = self.embed(ids, DType::F32)?;
        let mut hidden = vec![x.clone()];
        let mut maps = Vec::new();
        for (i, w) in self.layers.iter().enumerate() {
            let w = w.cast(DType::F32)?;
            x = self.layer(i, &w, &x, 0, None, attention.then_some(&mut maps))?;
            hidden.push(x.clone());
        }
        let h = rms_norm(&x, &self.norm.to_dtype(DType::F32)?, self.cfg.rms_norm_eps)?;
        let logits = h.matmul(&self.lm_head.to_dtype(DType::F32)?.t()?)?;
        Ok(Trace {
            logits,
            hidden,
            attention: maps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::DecoderConfig;

    struct Fixture {
        cfg: DecoderConfig,
        weights: Vec<u8>,
        reference: serde_json::Value,
    }

    fn fixture(name: &str) -> Fixture {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/arch")
            .join(name);
        let config = std::fs::read(dir.join("config.json")).unwrap();
        Fixture {
            cfg: DecoderConfig::parse(&config).unwrap(),
            weights: std::fs::read(dir.join("model.safetensors")).unwrap(),
            reference: serde_json::from_slice(&std::fs::read(dir.join("reference.json")).unwrap())
                .unwrap(),
        }
    }

    impl Fixture {
        fn decoder(&self) -> Decoder {
            let vb = VarBuilder::from_slice_safetensors(&self.weights, DType::F32, &Device::Cpu)
                .unwrap();
            Decoder::new(&self.cfg, vb).unwrap()
        }
        fn ids(&self, key: &str) -> Vec<u32> {
            self.reference[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect()
        }
        fn floats(&self, v: &serde_json::Value) -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        }
    }

    fn close(what: &str, got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let worst = got
            .iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs() / (1.0 + w.abs()))
            .fold(0f32, f32::max);
        assert!(worst < tol, "{what}: worst relative error {worst}");
    }

    /// A failure means this implementation no longer computes what the
    /// family's reference implementation computes: logits, attention and the
    /// residual stream of every layer, on a text longer than the sliding window.
    #[test]
    fn every_family_matches_transformers_on_the_forward_pass() {
        for name in ["qwen3", "olmo2", "olmo3"] {
            let f = fixture(name);
            let d = f.decoder();
            let ids = f.ids("ids");
            let t = d.trace(&ids, true).unwrap();
            close(
                &format!("{name} logits"),
                &t.logits.flatten_all().unwrap().to_vec1().unwrap(),
                &f.floats(&f.reference["logits"]),
                1e-4,
            );
            for (l, want) in f.reference["attentions"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                close(
                    &format!("{name} attention {l}"),
                    &t.attention[l].flatten_all().unwrap().to_vec1().unwrap(),
                    &f.floats(want),
                    1e-5,
                );
            }
            for (l, want) in f.reference["hidden"].as_array().unwrap().iter().enumerate() {
                close(
                    &format!("{name} hidden {l}"),
                    &t.hidden[l + 1].flatten_all().unwrap().to_vec1().unwrap(),
                    &f.floats(want),
                    1e-4,
                );
            }
        }
    }

    /// Its query scale and weights prefix are not computed here: a ministral3
    /// built by this decoder would answer fluent nonsense.
    #[test]
    fn ministral3_is_refused_by_the_generic_decoder() {
        let f = fixture("ministral3");
        let vb = VarBuilder::from_slice_safetensors(&f.weights, DType::F32, &Device::Cpu).unwrap();
        let err = Decoder::new(&f.cfg, vb).err().expect("refused").to_string();
        assert!(err.contains("crate::ministral3"), "{err}");
    }

    /// A failure means the KV cache or the sliding window is wrong in the
    /// decode path: greedy tokens differ from transformers' `generate`.
    #[test]
    fn every_family_generates_what_transformers_generates() {
        for name in ["qwen3", "olmo2", "olmo3"] {
            let f = fixture(name);
            let d = f.decoder();
            let prompt = f.ids("prompt");
            let want = f.ids("generated");
            let mut cache = KvCache::default();
            let mut all = prompt.clone();
            let mut logits = d.forward(&prompt, 0, &mut cache).unwrap();
            let mut got = Vec::new();
            for _ in 0..want.len() {
                let next = logits
                    .flatten_all()
                    .unwrap()
                    .argmax(0)
                    .unwrap()
                    .to_scalar::<u32>()
                    .unwrap();
                got.push(next);
                all.push(next);
                logits = d.forward(&[next], all.len() - 1, &mut cache).unwrap();
            }
            assert_eq!(got, want, "{name}");
            assert_eq!(cache.len(), all.len());
        }
    }

    /// The decoder and candle's own models agree, so serving OLMo 2 with
    /// this file rather than candle's `olmo2` changes nothing on CPU/f32.
    #[test]
    fn the_decoder_agrees_with_candles_qwen3_and_olmo2() {
        for name in ["qwen3", "olmo2"] {
            let f = fixture(name);
            let d = f.decoder();
            let ids = f.ids("ids");
            let ours = d
                .forward(&ids, 0, &mut KvCache::default())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let vb =
                VarBuilder::from_slice_safetensors(&f.weights, DType::F32, &Device::Cpu).unwrap();
            let input = Tensor::new(ids.as_slice(), &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let theirs = if name == "qwen3" {
                let c = crate::candle_backend::parse_qwen3_config(
                    &std::fs::read(format!(
                        "{}/tests/arch/qwen3/config.json",
                        env!("CARGO_MANIFEST_DIR")
                    ))
                    .unwrap(),
                )
                .unwrap();
                let mut m =
                    candle_transformers::models::qwen3::ModelForCausalLM::new(&c, vb).unwrap();
                m.forward(&input, 0).unwrap()
            } else {
                // transformers 5 nests rope_theta; candle wants it flat.
                let mut v: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(format!(
                        "{}/tests/arch/olmo2/config.json",
                        env!("CARGO_MANIFEST_DIR")
                    ))
                    .unwrap(),
                )
                .unwrap();
                v["rope_theta"] = v["rope_parameters"]["rope_theta"].clone();
                let c: candle_transformers::models::olmo2::Config =
                    serde_json::from_value(v).unwrap();
                let mut m = candle_transformers::models::olmo2::Model::new(&c, vb).unwrap();
                m.forward(&input, 0).unwrap()
            };
            close(
                name,
                &ours,
                &theirs.flatten_all().unwrap().to_vec1().unwrap(),
                1e-4,
            );
        }
    }
}
