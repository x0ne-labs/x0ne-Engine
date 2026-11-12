//! x0ne Core as a backend: `--backend core`. Core computes, Engine decides:
//! Core prefills and returns the logits of each position (`Engine::session`),
//! and the token, EOS, stop sequences, budget, stream and cancellation are
//! Engine's ([`crate::stepped`]), the same code for every backend that hands
//! out logits. Greedy requests let Core pick each token on the GPU
//! ([`crate::stepped::DevicePick::Greedy`]): the host reads 4 bytes per token
//! instead of a row of logits, and gets the same token.
//!
//! Qwen3 dense, bf16, `head_dim` 128, on GPU 0 only; anything else is refused
//! at load. The context is the model's `max_position_embeddings`; a model
//! whose arena does not fit the GPU is refused at load. Core's crates are
//! `x0ne-core/` (its `PROVENANCE.md`), linked with the `cuda` feature only:
//! their kernels are compiled by nvcc. What reads a config or a device name
//! is compiled in every build.

use serde_json::Value;

use crate::arch::Architecture;
use crate::model_spec::ModelSpec;

/// The fingerprint `x0ne-core/PROVENANCE.md` gives the copy of Core's files.
pub const CORE_COPY_SHA256: &str =
    "10d4c8ea7890870c4239ed9981f367265458e63433ae8c43e15e98b035fafa8b";

/// `config.json` for Core's strict reader: a Qwen3 dense config, read first by
/// Engine's parser; `rope_theta` nested under `rope_parameters` (recent
/// `transformers`) is lifted to the top, any other RoPE type is refused.
pub fn core_config(config_json: &[u8]) -> anyhow::Result<String> {
    let spec = ModelSpec::parse(config_json)?;
    if spec.arch != Architecture::Qwen3 {
        anyhow::bail!(
            "--backend core serves Qwen3 dense only, not {}",
            spec.arch.name()
        );
    }
    let mut v: Value =
        serde_json::from_slice(config_json).map_err(|e| anyhow::anyhow!("config.json: {e}"))?;
    let obj = v
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("config.json: not an object"))?;
    if let Some(p) = obj.get("rope_parameters").cloned() {
        match p.get("rope_type").and_then(Value::as_str) {
            None | Some("default") => {}
            Some(t) => {
                anyhow::bail!("config.json: RoPE type {t:?}, Core implements the default only")
            }
        }
        if !obj.contains_key("rope_theta") {
            if let Some(theta) = p.get("rope_theta") {
                obj.insert("rope_theta".into(), theta.clone());
            }
        }
    }
    Ok(serde_json::to_string(&v)?)
}

/// GPU 0 only: `cuda` or `cuda:0`.
pub fn check_device(device: &str) -> anyhow::Result<()> {
    match device {
        "cuda" | "cuda:0" => Ok(()),
        _ => anyhow::bail!("--backend core runs on GPU 0 only (cuda or cuda:0), not {device:?}"),
    }
}

#[cfg(feature = "cuda")]
pub use linked::{CoreBackend, CoreEntry, CoreModel, EngineFacts, PREFILL_CHUNK};

#[cfg(feature = "cuda")]
mod linked {
    use std::sync::{Mutex, PoisonError};

    use tokenizers::Tokenizer;
    use x0ne_core_runtime::engine::{Engine, EngineOptions, Greedy, Session};
    use x0ne_core_runtime::st::SliceSource;

    use super::{check_device, core_config, CORE_COPY_SHA256};
    use crate::backend::{
        Completion, GenerationParams, InferenceBackend, Isolation, LoadedModel, NoSink, TokenSink,
        WeightFiles,
    };
    use crate::backends::NamedBackend;
    use crate::keys::ReleasedKey;
    use crate::model_spec::{served_digest, ModelArtifacts, ModelSpec};
    use crate::prompt::Prompt;
    use crate::stepped::{self, DevicePick, LogitsSession};

    /// Rows of one prefill pass in Core (`PREFILL_CHUNK`).
    pub const PREFILL_CHUNK: usize = x0ne_core_runtime::engine::PREFILL_CHUNK;

    /// What the loaded engine says of itself: the copy of Core, its kernels and
    /// their compiler, how it decodes. Printed at load; not in a receipt.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct EngineFacts {
        /// `x0ne-core/PROVENANCE.md`'s fingerprint of the copied files.
        pub core_copy_sha256: &'static str,
        /// Hash of the embedded cubins (`x0ne_core_kernels::KERNEL_SET_SHA256`).
        pub kernel_set_sha256: &'static str,
        /// `nvcc --version`'s release line at build time.
        pub nvcc_release: &'static str,
        /// `Engine::describe`: decode mode, attention, splits.
        pub decode: String,
    }

    /// A loaded model on Core.
    pub struct CoreModel {
        pub(crate) engine: Mutex<Engine>,
        pub(crate) tokenizer: Tokenizer,
        pub(crate) eos_ids: Vec<u32>,
        /// The vocabulary as bytes, for `json_schema` masks: read once.
        token_table: stepped::TableCache,
        digest_hex: String,
        pub(crate) context_tokens: usize,
        facts: EngineFacts,
    }

    impl CoreModel {
        /// Loads `weights` with the metadata `artifacts` on `device`.
        pub fn load(
            artifacts: &ModelArtifacts,
            weights: &WeightFiles<'_>,
            device: &str,
        ) -> anyhow::Result<Self> {
            check_device(device)?;
            let config = core_config(&artifacts.config_json)?;
            if weights.files.is_empty() || weights.files.iter().any(|(_, b)| b.is_empty()) {
                anyhow::bail!("no weight bytes given");
            }
            if weights.files.iter().any(|(_, b)| crate::seal::is_sealed(b)) {
                anyhow::bail!("sealed weights are opened only through load() and the released key");
            }
            // The index and the headers agree, as for every backend (shards.rs).
            crate::shards::Routing::check(weights.index_json, &weights.files)
                .map_err(|e| anyhow::anyhow!("safetensors: {e:#}"))?;
            let digest_hex =
                served_digest(&artifacts.config_json, &artifacts.tokenizer_json, weights)?;
            // A message's text never becomes a control token (crate::prompt).
            let tokenizer = crate::prompt::text_only(
                Tokenizer::from_bytes(&artifacts.tokenizer_json)
                    .map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?,
            );
            // As every backend's load: a marker that is a plain word could be
            // typed by a message, and an answer needs a token that ends it.
            crate::prompt::require_control_markers(&tokenizer, crate::arch::Architecture::Qwen3)
                .map_err(|e| anyhow::anyhow!(e))?;
            let eos_ids = crate::prompt::eos_ids(&tokenizer, &["<|im_end|>", "<|endoftext|>"])
                .map_err(|e| anyhow::anyhow!(e))?;
            let context_tokens = ModelSpec::parse(&artifacts.config_json)?
                .decoder
                .max_position_embeddings;
            let source = SliceSource::new(weights.index_json, &weights.files)?;
            let engine = Engine::load_from(
                &config,
                &source,
                EngineOptions {
                    max_ctx: context_tokens,
                },
            )?;
            let (kernel_set_sha256, nvcc_release) = x0ne_core_runtime::kernel_set();
            let facts = EngineFacts {
                core_copy_sha256: CORE_COPY_SHA256,
                kernel_set_sha256,
                nvcc_release,
                decode: engine.describe(),
            };
            eprintln!(
                "x0ne-infer: core: copy {}, kernels {}, {}; {}",
                &facts.core_copy_sha256[..12],
                &facts.kernel_set_sha256[..12.min(facts.kernel_set_sha256.len())],
                facts.nvcc_release,
                facts.decode
            );
            Ok(Self {
                engine: Mutex::new(engine),
                tokenizer,
                eos_ids,
                token_table: stepped::TableCache::default(),
                digest_hex,
                context_tokens,
                facts,
            })
        }

        /// What the loaded engine says of itself.
        pub fn facts(&self) -> &EngineFacts {
            &self.facts
        }
    }

    /// A Core session seen by Engine's loop: stepped by the host, or, once
    /// the loop asks, deciding greedy tokens on the GPU.
    pub(super) enum CoreSession<'e> {
        Stepping(Session<'e>),
        Device(Greedy<'e>),
        /// `into_greedy` refused: the session was dropped, and cleared by its `Drop`.
        Gone,
    }

    impl LogitsSession for CoreSession<'_> {
        fn logits(&self) -> &[f32] {
            match self {
                Self::Stepping(s) => s.logits(),
                // Never read on the device path (stepped::run): no row is copied.
                Self::Device(_) | Self::Gone => &[],
            }
        }
        fn step(&mut self, token: u32) -> anyhow::Result<()> {
            match self {
                Self::Stepping(s) => Ok(s.step(token)?),
                _ => anyhow::bail!("core: a step on a session that decides on the GPU"),
            }
        }
        fn finish(self) -> anyhow::Result<()> {
            match self {
                Self::Stepping(s) => Ok(s.finish()?),
                Self::Device(g) => Ok(g.finish()?),
                Self::Gone => Ok(()),
            }
        }
        fn picks_on_device(&self, rule: DevicePick) -> bool {
            matches!((self, rule), (Self::Stepping(_), DevicePick::Greedy))
        }
        fn next_on_device(&mut self, rule: DevicePick, budget: usize) -> anyhow::Result<u32> {
            let DevicePick::Greedy = rule;
            if let Self::Stepping(_) = self {
                let Self::Stepping(s) = std::mem::replace(self, Self::Gone) else {
                    unreachable!("matched above")
                };
                *self = Self::Device(s.into_greedy(budget)?);
            }
            match self {
                Self::Device(g) => Ok(g.next_token()?),
                _ => anyhow::bail!("core: the session was refused the GPU's greedy decoding"),
            }
        }
    }

    impl LoadedModel for CoreModel {
        fn generate(
            &self,
            prompt: &Prompt,
            max_tokens: usize,
        ) -> anyhow::Result<(String, (u32, u32))> {
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
        fn generate_whole(
            &self,
            prompt: &Prompt,
            params: &GenerationParams,
            sink: &mut dyn TokenSink,
        ) -> anyhow::Result<Completion> {
            self.generate_streaming(prompt, params, sink)
        }
        fn generate_streaming(
            &self,
            prompt: &Prompt,
            params: &GenerationParams,
            sink: &mut dyn TokenSink,
        ) -> anyhow::Result<Completion> {
            let req = stepped::prepare(
                &self.tokenizer,
                prompt,
                params.max_tokens,
                self.context_tokens,
            )?;
            // Before the lock: the table is read once, and a schema this
            // vocabulary cannot hold is refused without waiting for the engine.
            let table = self
                .token_table
                .get(&self.tokenizer, &self.eos_ids, params)?;
            // A panic under this lock dropped its session, and that drop cleared
            // the client bytes; when the clear fails Core poisons the engine and
            // `session` retries it before any prefill.
            let mut guard = self.engine.lock().unwrap_or_else(PoisonError::into_inner);
            let engine: &mut Engine = &mut guard;
            stepped::run_with_table(
                &self.tokenizer,
                &self.eos_ids,
                table,
                &req,
                params,
                sink,
                move |ids| {
                    let engine = engine;
                    Ok(CoreSession::Stepping(engine.session(ids)?))
                },
            )
        }
        fn model_digest_hex(&self) -> String {
            self.digest_hex.clone()
        }
        fn compute_device(&self) -> String {
            "cuda:0".into()
        }
        fn isolation(&self) -> Isolation {
            Isolation::NONE_OF_IT
        }
        fn serving_limits(&self) -> Option<(usize, usize)> {
            Some((self.context_tokens, PREFILL_CHUNK.min(self.context_tokens)))
        }
        fn prefill_chunk_for(&self, _prompt_tokens: usize) -> Option<usize> {
            Some(PREFILL_CHUNK.min(self.context_tokens))
        }
    }

    /// Core over a model's metadata, for loads through the released key.
    pub struct CoreBackend {
        artifacts: ModelArtifacts,
        device: String,
    }

    impl InferenceBackend for CoreBackend {
        fn name(&self) -> &'static str {
            "core-qwen3"
        }
        fn load(
            &self,
            encrypted_weights: &[u8],
            key: &ReleasedKey,
        ) -> anyhow::Result<Box<dyn LoadedModel>> {
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
            let files = WeightFiles {
                index_json: None,
                files: vec![(crate::shards::SINGLE_FILE, bytes)],
            };
            Ok(Box::new(CoreModel::load(
                &self.artifacts,
                &files,
                &self.device,
            )?))
        }
        fn load_files(
            &self,
            weights: &WeightFiles<'_>,
            key: &ReleasedKey,
        ) -> anyhow::Result<Box<dyn LoadedModel>> {
            let _ = key;
            if weights.files.iter().any(|(_, b)| crate::seal::is_sealed(b)) {
                anyhow::bail!("sealed weights are one file, opened through load()");
            }
            Ok(Box::new(CoreModel::load(
                &self.artifacts,
                weights,
                &self.device,
            )?))
        }
    }

    /// The registry entry: `--backend core`.
    pub struct CoreEntry;

    impl NamedBackend for CoreEntry {
        fn name(&self) -> &'static str {
            "core"
        }
        fn open(
            &self,
            artifacts: ModelArtifacts,
            device: &str,
        ) -> anyhow::Result<Box<dyn InferenceBackend + Send + Sync>> {
            check_device(device)?;
            core_config(&artifacts.config_json)?;
            Ok(Box::new(CoreBackend {
                artifacts,
                device: device.into(),
            }))
        }
    }
}

#[cfg(test)]
mod tests;
