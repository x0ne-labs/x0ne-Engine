//! The Qwen3 dense forward, one token at a time over a KV cache, rounding to bf16 where the
//! engine stores bf16 (`runtime/src/engine.rs`, `step_launches`): the embedding row, every
//! normed row, every projection's output but the logits, q and k after RoPE, the attention's
//! output, the SwiGLU's. The residual stream stays in f32 and the logits are f32.
//!
//! One code path serves a prefill and a decode step: a causal prefill of n tokens is n steps.
//! The engine has two attention kernels for the two; the tests state the tolerance between
//! each of them and this path.

use crate::{add_rmsnorm, attention_decode, bf16, embedding, gemv, qk_norm_rope, swiglu, Heads};

/// The sizes of a Qwen3 dense model, as its `config.json` names them.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn heads(&self) -> Heads {
        Heads {
            q_heads: self.num_attention_heads,
            kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
        }
    }

    /// The `config.json` of this model, with the fields the runtime's strict reader asks for.
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\"model_type\":\"qwen3\",\"dtype\":\"bfloat16\",\"hidden_act\":\"silu\",",
                "\"attention_bias\":false,\"hidden_size\":{},\"intermediate_size\":{},",
                "\"num_hidden_layers\":{},\"num_attention_heads\":{},\"num_key_value_heads\":{},",
                "\"head_dim\":{},\"vocab_size\":{},\"max_position_embeddings\":{},",
                "\"rms_norm_eps\":{:e},\"rope_theta\":{:e},\"tie_word_embeddings\":{}}}"
            ),
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.max_position_embeddings,
            self.rms_norm_eps,
            self.rope_theta,
            self.tie_word_embeddings
        )
    }

    /// Every tensor of the checkpoint with its shape, in the order of the layers: the names
    /// and shapes of a Hugging Face Qwen3 checkpoint, written out here leaf by leaf.
    pub fn tensors(&self) -> Vec<(String, Vec<usize>)> {
        let (h, i, hd) = (self.hidden_size, self.intermediate_size, self.head_dim);
        let (qd, kvd) = (self.heads().q_dim(), self.heads().kv_dim());
        let mut v = vec![(
            "model.embed_tokens.weight".to_owned(),
            vec![self.vocab_size, h],
        )];
        for l in 0..self.num_hidden_layers {
            let p = format!("model.layers.{l}");
            v.push((format!("{p}.input_layernorm.weight"), vec![h]));
            v.push((format!("{p}.self_attn.q_proj.weight"), vec![qd, h]));
            v.push((format!("{p}.self_attn.k_proj.weight"), vec![kvd, h]));
            v.push((format!("{p}.self_attn.v_proj.weight"), vec![kvd, h]));
            v.push((format!("{p}.self_attn.q_norm.weight"), vec![hd]));
            v.push((format!("{p}.self_attn.k_norm.weight"), vec![hd]));
            v.push((format!("{p}.self_attn.o_proj.weight"), vec![h, qd]));
            v.push((format!("{p}.post_attention_layernorm.weight"), vec![h]));
            v.push((format!("{p}.mlp.gate_proj.weight"), vec![i, h]));
            v.push((format!("{p}.mlp.up_proj.weight"), vec![i, h]));
            v.push((format!("{p}.mlp.down_proj.weight"), vec![h, i]));
        }
        v.push(("model.norm.weight".to_owned(), vec![h]));
        if !self.tie_word_embeddings {
            v.push(("lm_head.weight".to_owned(), vec![self.vocab_size, h]));
        }
        v
    }
}

struct Layer {
    input_norm: Vec<f32>,
    /// `[q_dim + 2 kv_dim][hidden]`: q rows, k rows, v rows.
    qkv: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    o: Vec<f32>,
    post_norm: Vec<f32>,
    /// `[2 intermediate][hidden]`: gate rows, up rows.
    gate_up: Vec<f32>,
    down: Vec<f32>,
}

/// A model: its weights as f32 values of bf16.
pub struct Model {
    cfg: Config,
    embed: Vec<f32>,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    lm_head: Option<Vec<f32>>,
    inv_freq: Vec<f32>,
}

/// The K and V rows of every layer for the positions seen so far, `[layer][position][kv_dim]`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Cache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    len: usize,
}

impl Cache {
    /// Positions held.
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Forgets every position from `len` on.
    pub fn truncate(&mut self, len: usize) {
        assert!(len <= self.len);
        let per = self
            .k
            .first()
            .map_or(0, Vec::len)
            .checked_div(self.len)
            .unwrap_or(0);
        for rows in self.k.iter_mut().chain(self.v.iter_mut()) {
            rows.truncate(len * per);
        }
        self.len = len;
    }
    /// The K row and the V row of `layer` at `pos`.
    pub fn row(&self, layer: usize, pos: usize) -> (&[f32], &[f32]) {
        let per = self.k[layer].len() / self.len;
        (
            &self.k[layer][pos * per..(pos + 1) * per],
            &self.v[layer][pos * per..(pos + 1) * per],
        )
    }
}

impl Model {
    /// A model from its tensors: `get(name)` gives the values of a tensor of `cfg.tensors()`.
    /// `inv_freq` is what the host gives the device (`ops::inv_freq`).
    pub fn new(cfg: Config, inv_freq: Vec<f32>, mut get: impl FnMut(&str) -> Vec<f32>) -> Model {
        assert_eq!(inv_freq.len(), cfg.head_dim / 2);
        let layers = (0..cfg.num_hidden_layers)
            .map(|l| {
                let mut t = |leaf: &str| get(&format!("model.layers.{l}.{leaf}.weight"));
                let mut qkv = t("self_attn.q_proj");
                qkv.extend(t("self_attn.k_proj"));
                qkv.extend(t("self_attn.v_proj"));
                let mut gate_up = t("mlp.gate_proj");
                gate_up.extend(t("mlp.up_proj"));
                Layer {
                    input_norm: t("input_layernorm"),
                    qkv,
                    q_norm: t("self_attn.q_norm"),
                    k_norm: t("self_attn.k_norm"),
                    o: t("self_attn.o_proj"),
                    post_norm: t("post_attention_layernorm"),
                    gate_up,
                    down: t("mlp.down_proj"),
                }
            })
            .collect();
        Model {
            embed: get("model.embed_tokens.weight"),
            layers,
            final_norm: get("model.norm.weight"),
            lm_head: (!cfg.tie_word_embeddings).then(|| get("lm_head.weight")),
            inv_freq,
            cfg,
        }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Feeds `token` at position `cache.len()`: writes its K and V rows into the cache and
    /// returns the f32 logits of the next position.
    pub fn step(&self, cache: &mut Cache, token: u32) -> Vec<f32> {
        let c = &self.cfg;
        let (h, heads) = (c.hidden_size, c.heads());
        let (qd, kvd, inter) = (heads.q_dim(), heads.kv_dim(), c.intermediate_size);
        let pos = cache.len;
        assert!(
            pos < c.max_position_embeddings,
            "position {pos} past the model's"
        );
        if cache.k.is_empty() {
            cache.k = vec![Vec::new(); self.layers.len()];
            cache.v = vec![Vec::new(); self.layers.len()];
        }
        let mut delta = embedding(&self.embed, h, token);
        let mut residual = vec![0f32; h];
        for (l, w) in self.layers.iter().enumerate() {
            let normed = bf16::round_all(&add_rmsnorm(
                &mut residual,
                &delta,
                &w.input_norm,
                c.rms_norm_eps,
            ));
            let qkv = bf16::round_all(&gemv(&w.qkv, qd + 2 * kvd, h, &normed));
            let (q, k, v) = qk_norm_rope(
                &qkv,
                &w.q_norm,
                &w.k_norm,
                &self.inv_freq,
                heads,
                pos as u32,
                c.rms_norm_eps,
            );
            cache.k[l].extend(bf16::round_all(&k));
            cache.v[l].extend(v);
            let q = bf16::round_all(&q);
            let attn = bf16::round_all(&attention_decode(
                &q,
                &cache.k[l],
                &cache.v[l],
                pos + 1,
                heads,
            ));
            delta = bf16::round_all(&gemv(&w.o, h, qd, &attn));
            let normed = bf16::round_all(&add_rmsnorm(
                &mut residual,
                &delta,
                &w.post_norm,
                c.rms_norm_eps,
            ));
            let gate_up = bf16::round_all(&gemv(&w.gate_up, 2 * inter, h, &normed));
            let act = bf16::round_all(&swiglu(&gate_up));
            delta = bf16::round_all(&gemv(&w.down, h, inter, &act));
        }
        cache.len = pos + 1;
        let normed = bf16::round_all(&add_rmsnorm(
            &mut residual,
            &delta,
            &self.final_norm,
            c.rms_norm_eps,
        ));
        gemv(
            self.lm_head.as_ref().unwrap_or(&self.embed),
            c.vocab_size,
            h,
            &normed,
        )
    }

    /// Feeds `ids` in order; the logits after the last one. `ids` must not be empty.
    pub fn extend(&self, cache: &mut Cache, ids: &[u32]) -> Vec<f32> {
        let mut last = Vec::new();
        for &t in ids {
            last = self.step(cache, t);
        }
        assert!(!last.is_empty(), "no token");
        last
    }

    /// The logits after every token of `ids`, from an empty cache.
    pub fn logits(&self, ids: &[u32]) -> Vec<Vec<f32>> {
        let mut cache = Cache::default();
        ids.iter().map(|&t| self.step(&mut cache, t)).collect()
    }
}
