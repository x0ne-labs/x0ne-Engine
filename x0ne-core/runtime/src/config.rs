//! Qwen3 dense and Qwen3-MoE `config.json`, read strictly: anything this engine does not
//! implement is refused.

use crate::error::{format_err, Result};
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
}

/// A parsed config cannot overflow its derived sizes; the fields are public, so one built by
/// hand can, and panics here instead of wrapping (2^57 heads gave q_dim 0 in release, 2026-11-04).
const SIZES_FIT: &str = "a config's derived sizes overflow; from_json refuses such a config";

fn get<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
    v.get(key)
        .ok_or_else(|| crate::Error::Format(format!("config.json: missing {key}")))
}

fn usize_of(v: &Value, key: &str) -> Result<usize> {
    let n = get(v, key)?.as_u64().filter(|&n| n > 0);
    match n.and_then(|n| usize::try_from(n).ok()) {
        Some(n) => Ok(n),
        None => format_err(format!("config.json: {key} is not a positive integer")),
    }
}

fn f64_of(v: &Value, key: &str) -> Result<f64> {
    match get(v, key)?.as_f64() {
        Some(x) if x.is_finite() && x > 0.0 => Ok(x),
        _ => format_err(format!("config.json: {key} is not a positive number")),
    }
}

/// `true` when the key is absent, null, or false.
fn off(v: &Value, key: &str) -> bool {
    matches!(
        v.get(key),
        None | Some(Value::Null) | Some(Value::Bool(false))
    )
}

fn parse(text: &str) -> Result<Value> {
    serde_json::from_str(text).map_err(|e| crate::Error::Format(format!("config.json: {e}")))
}

impl Qwen3Config {
    pub fn from_json(text: &str) -> Result<Self> {
        Self::from_value(&parse(text)?, "qwen3")
    }

    /// The fields Qwen3 dense and Qwen3-MoE share, for a config of `model_type`.
    fn from_value(v: &Value, model_type: &str) -> Result<Self> {
        if get(v, "model_type")?.as_str() != Some(model_type) {
            return format_err(format!("config.json: model_type is not {model_type}"));
        }
        // Newer transformers write "dtype", older ones "torch_dtype".
        let dtype = v
            .get("dtype")
            .or_else(|| v.get("torch_dtype"))
            .and_then(Value::as_str);
        if dtype != Some("bfloat16") {
            return format_err(format!(
                "config.json: dtype {dtype:?}, only bfloat16 is served"
            ));
        }
        if get(v, "hidden_act")?.as_str() != Some("silu") {
            return format_err("config.json: hidden_act is not silu");
        }
        for key in ["attention_bias", "use_sliding_window", "rope_scaling"] {
            if !off(v, key) {
                return format_err(format!("config.json: {key} is set, not implemented"));
            }
        }
        let tie = match v.get("tie_word_embeddings") {
            Some(Value::Bool(b)) => *b,
            None => false,
            _ => return format_err("config.json: tie_word_embeddings is not a boolean"),
        };
        let c = Qwen3Config {
            hidden_size: usize_of(v, "hidden_size")?,
            intermediate_size: usize_of(v, "intermediate_size")?,
            num_hidden_layers: usize_of(v, "num_hidden_layers")?,
            num_attention_heads: usize_of(v, "num_attention_heads")?,
            num_key_value_heads: usize_of(v, "num_key_value_heads")?,
            head_dim: usize_of(v, "head_dim")?,
            vocab_size: usize_of(v, "vocab_size")?,
            max_position_embeddings: usize_of(v, "max_position_embeddings")?,
            rms_norm_eps: f64_of(v, "rms_norm_eps")?,
            rope_theta: f64_of(v, "rope_theta")?,
            tie_word_embeddings: tie,
        };
        if c.num_key_value_heads == 0
            || !c.num_attention_heads.is_multiple_of(c.num_key_value_heads)
        {
            return format_err("config.json: attention heads are not a multiple of KV heads");
        }
        if !c.head_dim.is_multiple_of(2) {
            return format_err("config.json: head_dim is odd, RoPE needs pairs");
        }
        if c.vocab_size > u32::MAX as usize {
            return format_err("config.json: vocab_size does not fit a u32 token id");
        }
        c.check_sizes()?;
        Ok(c)
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// Refuses a config one of whose derived sizes overflows: the rows of a projection, the
    /// parameters of a layer, the KV bytes of a position, the bytes of all the weights. A parsed
    /// config has passed it; the fields are public, so what takes a config checks it again.
    pub(crate) fn check_sizes(&self) -> Result<()> {
        let fit = self.linear_params().is_some()
            && self.norm_params().is_some()
            && self.kv_bytes().is_some()
            && self.weight_bytes().is_some();
        if !fit {
            return format_err("config.json: a size derived from the config overflows");
        }
        Ok(())
    }

    fn q_rows(&self) -> Option<usize> {
        self.num_attention_heads.checked_mul(self.head_dim)
    }

    fn kv_rows(&self) -> Option<usize> {
        self.num_key_value_heads.checked_mul(self.head_dim)
    }

    fn packed_rows(&self) -> Option<usize> {
        self.q_rows()?.checked_add(self.kv_rows()?.checked_mul(2)?)
    }

    fn linear_params(&self) -> Option<u64> {
        let h = self.hidden_size as u64;
        let attn = h
            .checked_mul(self.packed_rows()? as u64)?
            .checked_add((self.q_rows()? as u64).checked_mul(h)?)?;
        let mlp = h
            .checked_mul(3)?
            .checked_mul(self.intermediate_size as u64)?;
        attn.checked_add(mlp)
    }

    fn norm_params(&self) -> Option<u64> {
        (self.hidden_size as u64)
            .checked_mul(2)?
            .checked_add((self.head_dim as u64).checked_mul(2)?)
    }

    fn kv_bytes(&self) -> Option<u64> {
        let bytes = self.num_hidden_layers.checked_mul(4)?;
        Some(bytes.checked_mul(self.kv_rows()?)? as u64)
    }

    /// The embedding, the final norm and the LM head unless tied, in parameters.
    fn outer_params(&self) -> Option<u64> {
        let embed = (self.vocab_size as u64).checked_mul(self.hidden_size as u64)?;
        let copies = if self.tie_word_embeddings { 1 } else { 2 };
        embed
            .checked_mul(copies)?
            .checked_add(self.hidden_size as u64)
    }

    /// Bytes of every weight of the dense model, bf16.
    fn weight_bytes(&self) -> Option<u64> {
        let layer = self.linear_params()?.checked_add(self.norm_params()?)?;
        (self.num_hidden_layers as u64)
            .checked_mul(layer)?
            .checked_add(self.outer_params()?)?
            .checked_mul(2)
    }

    pub fn q_dim(&self) -> usize {
        self.q_rows().expect(SIZES_FIT)
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_rows().expect(SIZES_FIT)
    }

    /// Output rows of the packed [q; k; v] projection.
    pub fn qkv_rows(&self) -> usize {
        self.packed_rows().expect(SIZES_FIT)
    }

    /// Parameters of the seven projections of one layer (norms excluded).
    pub fn linear_params_per_layer(&self) -> u64 {
        self.linear_params().expect(SIZES_FIT)
    }

    /// Parameters of the norm weights of one layer.
    pub fn norm_params_per_layer(&self) -> u64 {
        self.norm_params().expect(SIZES_FIT)
    }

    /// K and V bytes stored per cached position, all layers, bf16.
    pub fn kv_bytes_per_position(&self) -> u64 {
        self.kv_bytes().expect(SIZES_FIT)
    }
}

/// The sparse feed-forward of Qwen3-MoE: every layer routes each token to
/// `num_experts_per_tok` of `num_experts` SwiGLU experts and renormalises their weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
}

/// Qwen3-MoE `config.json` (`model_type` qwen3_moe), read strictly. `base` holds what Qwen3
/// dense has (attention, norms, vocabulary, positions); its `intermediate_size` is unused.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3MoeConfig {
    pub base: Qwen3Config,
    pub moe: MoeConfig,
}

impl Qwen3MoeConfig {
    pub fn from_json(text: &str) -> Result<Self> {
        let v = parse(text)?;
        let base = Qwen3Config::from_value(&v, "qwen3_moe")?;
        // Shared experts (Qwen2-MoE, DeepSeek) are another block: GM2.
        for key in ["shared_expert_intermediate_size", "n_shared_experts"] {
            if !matches!(v.get(key), None | Some(Value::Null)) {
                return format_err(format!("config.json: {key} is set, not implemented"));
            }
        }
        if usize_of(&v, "decoder_sparse_step")? != 1 {
            return format_err(
                "config.json: decoder_sparse_step is not 1, dense layers are not implemented",
            );
        }
        match v.get("mlp_only_layers") {
            None | Some(Value::Null) => {}
            Some(Value::Array(a)) if a.is_empty() => {}
            _ => {
                return format_err(
                    "config.json: mlp_only_layers is not empty, dense layers are not implemented",
                )
            }
        }
        if v.get("norm_topk_prob") != Some(&Value::Bool(true)) {
            return format_err(
                "config.json: norm_topk_prob is not true, only renormalised top-k is implemented",
            );
        }
        let moe = MoeConfig {
            num_experts: usize_of(&v, "num_experts")?,
            num_experts_per_tok: usize_of(&v, "num_experts_per_tok")?,
            moe_intermediate_size: usize_of(&v, "moe_intermediate_size")?,
        };
        if moe.num_experts_per_tok > moe.num_experts {
            return format_err("config.json: num_experts_per_tok exceeds num_experts");
        }
        if moe.num_experts > u32::MAX as usize {
            return format_err("config.json: num_experts does not fit a u32 expert id");
        }
        let c = Qwen3MoeConfig { base, moe };
        c.check_sizes()?;
        Ok(c)
    }

    /// As `Qwen3Config::check_sizes`, with the router and the experts of every layer.
    pub(crate) fn check_sizes(&self) -> Result<()> {
        self.base.check_sizes()?;
        if self.weight_bytes().is_none() {
            return format_err("config.json: a size derived from the config overflows");
        }
        Ok(())
    }

    /// Bytes of every weight of the MoE model, bf16: per layer the attention, the norms, the
    /// router and three projections per expert.
    fn weight_bytes(&self) -> Option<u64> {
        let (b, m) = (&self.base, &self.moe);
        let (h, n) = (b.hidden_size as u64, m.num_experts as u64);
        let attn = h
            .checked_mul(b.packed_rows()? as u64)?
            .checked_add((b.q_rows()? as u64).checked_mul(h)?)?;
        let expert = h
            .checked_mul(3)?
            .checked_mul(m.moe_intermediate_size as u64)?;
        let layer = attn
            .checked_add(b.norm_params()?)?
            .checked_add(n.checked_mul(h)?)?
            .checked_add(n.checked_mul(expert)?)?;
        (b.num_hidden_layers as u64)
            .checked_mul(layer)?
            .checked_add(b.outer_params()?)?
            .checked_mul(2)
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// The first `layers` layers: embedding, final norm and LM head unchanged.
    pub fn truncated(&self, layers: usize) -> Result<Self> {
        if layers == 0 || layers > self.base.num_hidden_layers {
            return format_err(format!(
                "truncation to {layers} layers: the config has {}",
                self.base.num_hidden_layers
            ));
        }
        let mut c = self.clone();
        c.base.num_hidden_layers = layers;
        Ok(c)
    }
}

/// A `config.json` of Qwen3 dense (`model_type` qwen3) or Qwen3-MoE (qwen3_moe): the fields
/// both share, and the sparse block when there is one. Any other model_type is refused.
pub fn read_any(text: &str) -> Result<(Qwen3Config, Option<Qwen3MoeConfig>)> {
    if parse(text)?.get("model_type").and_then(Value::as_str) == Some("qwen3_moe") {
        let m = Qwen3MoeConfig::from_json(text)?;
        return Ok((m.base.clone(), Some(m)));
    }
    Ok((Qwen3Config::from_json(text)?, None))
}
