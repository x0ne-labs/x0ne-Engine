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
        Ok(c)
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Output rows of the packed [q; k; v] projection.
    pub fn qkv_rows(&self) -> usize {
        self.q_dim() + 2 * self.kv_dim()
    }

    /// Parameters of the seven projections of one layer (norms excluded).
    pub fn linear_params_per_layer(&self) -> u64 {
        let h = self.hidden_size as u64;
        let attn = h * self.qkv_rows() as u64 + self.q_dim() as u64 * h;
        let mlp = 3 * h * self.intermediate_size as u64;
        attn + mlp
    }

    /// Parameters of the norm weights of one layer.
    pub fn norm_params_per_layer(&self) -> u64 {
        2 * self.hidden_size as u64 + 2 * self.head_dim as u64
    }

    /// K and V bytes stored per cached position, all layers, bf16.
    pub fn kv_bytes_per_position(&self) -> u64 {
        (self.num_hidden_layers * 2 * self.kv_dim() * 2) as u64
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
        Ok(Qwen3MoeConfig { base, moe })
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
