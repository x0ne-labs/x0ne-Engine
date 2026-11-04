//! What a model's metadata says, read without candle: the artifacts a load
//! takes beside the weights, the Qwen3 `config.json` as candle 0.11 reads it,
//! the family and decoder view of a config ([`ModelSpec`]), and the served
//! digest of what is loaded. Every backend reads them here; `candle_backend`
//! re-exports them and converts the Qwen3 config into candle's own.
//!
//! [`ModelSpec`]: crate::model_spec::ModelSpec

use serde::{Deserialize, Serialize};

use crate::arch::{Architecture, DecoderConfig};
use crate::backend::WeightFiles;

/// What `load` needs beside the weights: the model's own metadata files,
/// as bytes — same discipline, no paths.
pub struct ModelArtifacts {
    pub config_json: Vec<u8>,
    pub tokenizer_json: Vec<u8>,
    // No weights here: `load` takes them as its argument, and a second copy
    // held by the constructor would double the resident model for nothing
    // (a 16 GB checkpoint was held twice, one copy never read, both resident
    // for the life of the server).
}

/// A Qwen3 `config.json`, field for field what candle 0.11's
/// `models::qwen3::Config` reads (the same names, the same required and
/// optional keys), so a config one refuses the other refuses. Proven against
/// candle's parser in `candle_backend`'s tests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
}

/// The `hidden_act` names candle 0.11's `Activation` reads, as it reads them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[serde(alias = "gelu")]
    Gelu,
    #[serde(alias = "gelu_new")]
    NewGelu,
    Relu,
    Relu2,
    Relu6,
    Silu,
    Sigmoid,
    HardSigmoid,
    Swiglu,
    Swish,
    Mish,
    HardSwish,
    Elu(f64),
    LeakyRelu(f64),
    #[serde(alias = "gelu_pytorch_tanh")]
    GeluPytorchTanh,
}

/// Accept `rope_theta` wherever the writer put it.
///
/// A `config.json` written by a recent `transformers` nests it under
/// `rope_parameters` and candle's `Config` requires it at the top level:
/// measured 2026-09-19 on a fine-tuned Qwen3-0.6B, the load died with
/// "missing field `rope_theta`" and the workaround was to paste the base
/// model's config back. A top-level value, when present, is left alone.
///
/// The model digest still covers the config bytes AS WRITTEN — this rewrite
/// feeds the parser, it does not change what is hashed.
pub fn parse_qwen3_config(config_json: &[u8]) -> anyhow::Result<Qwen3Config> {
    let mut v: serde_json::Value =
        serde_json::from_slice(config_json).map_err(|e| anyhow::anyhow!("config.json: {e}"))?;
    if let Some(obj) = v.as_object_mut() {
        if !obj.contains_key("rope_theta") {
            if let Some(theta) = obj
                .get("rope_parameters")
                .and_then(|p| p.get("rope_theta"))
                .cloned()
            {
                obj.insert("rope_theta".into(), theta);
            }
        }
    }
    serde_json::from_value(v).map_err(|e| anyhow::anyhow!("config.json: {e}"))
}

/// What a config.json describes, parsed once: the family, the Qwen3 config
/// when it is one, and the decoder's view of it.
#[derive(Clone)]
pub struct ModelSpec {
    pub arch: Architecture,
    qwen3: Option<Qwen3Config>,
    pub decoder: DecoderConfig,
    /// The release's default system message ([`Self::served_as`]).
    pub default_system: Option<&'static str>,
}

impl ModelSpec {
    pub fn parse(config_json: &[u8]) -> anyhow::Result<Self> {
        let arch = Architecture::detect(config_json)?;
        let qwen3 = match arch {
            Architecture::Qwen3 => Some(parse_qwen3_config(config_json)?),
            _ => None,
        };
        let decoder = DecoderConfig::parse(config_json)?;
        Ok(Self {
            arch,
            qwen3,
            decoder,
            default_system: None,
        })
    }

    /// The Qwen3 config, for a Qwen3 model.
    pub fn qwen3(&self) -> Option<&Qwen3Config> {
        self.qwen3.as_ref()
    }

    /// The spec of the release served under `digest_hex` (v2): its chat
    /// template's default system message ([`Architecture::default_system`]).
    #[must_use]
    pub fn served_as(mut self, digest_hex: &str) -> Self {
        self.default_system = self.arch.default_system(digest_hex);
        self
    }

    /// The load's line on the default system message: which one a chat
    /// without a system message gets, or that none is added. Not its text.
    pub fn default_system_line(&self, digest_hex: &str) -> String {
        use sha2::{Digest, Sha256};
        let release = digest_hex.get(..12).unwrap_or(digest_hex);
        match self.default_system {
            Some(text) => {
                let sha: String = Sha256::digest(text.as_bytes())[..8]
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect();
                format!(
                    "x0ne-infer: default system message: the one of release {release} \
                     ({} bytes, sha256 {sha}...), added to a chat that has none",
                    text.len()
                )
            }
            None => format!(
                "x0ne-infer: default system message: none for release {release}; \
                 a chat is prompted with the messages it holds"
            ),
        }
    }
}

/// The served-model digest (v2, `model_digest`) of what is loaded.
pub fn served_digest(
    config_json: &[u8],
    tokenizer_json: &[u8],
    weights: &WeightFiles<'_>,
) -> anyhow::Result<String> {
    let mut parts: Vec<(&str, &[u8])> = vec![
        ("config.json", config_json),
        ("tokenizer.json", tokenizer_json),
    ];
    if let Some(index) = weights.index_json {
        parts.push((crate::shards::INDEX_FILE, index));
    }
    parts.extend(weights.files.iter().copied());
    crate::model_digest::of_contents(&parts).map_err(|e| anyhow::anyhow!("served digest: {e}"))
}
