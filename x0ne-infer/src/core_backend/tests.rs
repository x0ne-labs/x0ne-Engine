use super::*;
use std::path::{Path, PathBuf};

/// Qwen3-0.6B's `config.json` (transformers 4.51), with `edit` applied.
fn config(edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>)) -> Vec<u8> {
    let mut v = serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
        "bos_token_id": 151643, "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu",
        "hidden_size": 1024, "initializer_range": 0.02, "intermediate_size": 3072,
        "max_position_embeddings": 40960, "max_window_layers": 28, "model_type": "qwen3",
        "num_attention_heads": 16, "num_hidden_layers": 28, "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 1000000, "sliding_window": null,
        "tie_word_embeddings": true, "torch_dtype": "bfloat16", "transformers_version": "4.51.0",
        "use_cache": true, "use_sliding_window": false, "vocab_size": 151936
    });
    edit(v.as_object_mut().unwrap());
    serde_json::to_vec(&v).unwrap()
}

fn olmo(c: &mut serde_json::Map<String, serde_json::Value>) {
    c.insert("model_type".into(), "olmo2".into());
    c.insert(
        "architectures".into(),
        serde_json::json!(["Olmo2ForCausalLM"]),
    );
}

fn read(text: &str) -> serde_json::Value {
    serde_json::from_str(text).unwrap()
}

/// Engine's parser first, then Core's: a nested `rope_theta` is lifted, a top
/// one kept; another family, or a RoPE type Core does not implement, refused.
#[test]
fn the_config_is_read_by_engine_then_normalized_for_core() {
    let plain = read(&core_config(&config(|_| {})).unwrap());
    assert_eq!(plain["rope_theta"], 1000000);

    let nested = config(|c| {
        c.remove("rope_theta");
        c.insert(
            "rope_parameters".into(),
            serde_json::json!({"rope_theta": 5000000, "rope_type": "default"}),
        );
    });
    assert_eq!(read(&core_config(&nested).unwrap())["rope_theta"], 5000000);

    let both = config(|c| {
        c.insert(
            "rope_parameters".into(),
            serde_json::json!({"rope_theta": 7}),
        );
    });
    assert_eq!(
        read(&core_config(&both).unwrap())["rope_theta"],
        1000000,
        "a top-level value is kept"
    );

    let yarn = config(|c| {
        c.insert(
            "rope_parameters".into(),
            serde_json::json!({"rope_theta": 1000000, "rope_type": "yarn", "factor": 4.0}),
        );
    });
    assert!(core_config(&yarn).is_err(), "yarn");
    assert!(core_config(&config(olmo)).is_err(), "another family");
    assert!(core_config(b"{").is_err());
}

/// Core runs on GPU 0 only.
#[test]
fn the_device_is_gpu_zero() {
    assert!(check_device("cuda").is_ok());
    assert!(check_device("cuda:0").is_ok());
    for d in ["cpu", "cuda:1", "cuda:", "metal:0", ""] {
        assert!(check_device(d).is_err(), "{d:?}");
    }
}

fn core_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../x0ne-core")
}

/// The files `x0ne-core/PROVENANCE.md` lists as copied, in its order.
fn listed() -> Vec<String> {
    let text = std::fs::read_to_string(core_dir().join("PROVENANCE.md")).unwrap();
    let list = text
        .split_once("\n- `")
        .map(|(_, rest)| rest)
        .expect("a list of copied files");
    let list = list.split("\n\n").next().unwrap();
    std::iter::once(list)
        .flat_map(|l| l.split("\n- `"))
        .map(|item| item.trim_end_matches('`').to_string())
        .collect()
}

/// The copy of Core is the one PROVENANCE.md fingerprints, and the build
/// prints: `sha256sum <files> | sha256sum`, recomputed from the files.
#[test]
fn the_copy_of_core_is_the_one_provenance_fingerprints() {
    use sha2::{Digest, Sha256};
    let hex = |b: &[u8]| -> String { b.iter().map(|x| format!("{x:02x}")).collect() };
    let files = listed();
    assert_eq!(files.len(), 30, "{files:?}");
    let mut lines = String::new();
    for f in &files {
        let bytes = std::fs::read(core_dir().join(f)).unwrap_or_else(|e| panic!("{f}: {e}"));
        lines.push_str(&format!("{}  {f}\n", hex(&Sha256::digest(&bytes))));
    }
    assert_eq!(hex(&Sha256::digest(lines.as_bytes())), CORE_COPY_SHA256);
    let text = std::fs::read_to_string(core_dir().join("PROVENANCE.md")).unwrap();
    assert!(
        text.contains(CORE_COPY_SHA256),
        "PROVENANCE.md prints another fingerprint"
    );
}

/// Every file under `kernels/` and `runtime/` is a listed copy: nothing is
/// written in Core's crates here without being in the fingerprint.
#[test]
fn nothing_unlisted_lives_in_cores_crates() {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<String>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, root, out);
            } else {
                out.push(p.strip_prefix(root).unwrap().to_string_lossy().into_owned());
            }
        }
    }
    let root = core_dir();
    let mut found = Vec::new();
    for d in ["kernels", "runtime"] {
        walk(&root.join(d), &root, &mut found);
    }
    found.sort();
    let mut want = listed();
    want.sort();
    assert_eq!(found, want);
}

/// The primitives every backend shares compile without candle: no candle
/// crate, nor the candle backend, is named in their code (tests and comments
/// aside). candle stays the reference their tests compare against.
#[test]
fn the_shared_primitives_name_no_candle() {
    let sources = [
        ("core_backend.rs", include_str!("../core_backend.rs")),
        ("sampler.rs", include_str!("../sampler.rs")),
        ("stepped.rs", include_str!("../stepped.rs")),
        ("model_spec.rs", include_str!("../model_spec.rs")),
        ("decode_stream.rs", include_str!("../decode_stream.rs")),
        ("shards.rs", include_str!("../shards.rs")),
        ("arch.rs", include_str!("../arch.rs")),
        ("backend.rs", include_str!("../backend.rs")),
        ("prompt.rs", include_str!("../prompt.rs")),
        ("constrain.rs", include_str!("../constrain.rs")),
        ("model_digest.rs", include_str!("../model_digest.rs")),
        ("seal.rs", include_str!("../seal.rs")),
    ];
    let names = [
        "candle_core",
        "candle_nn",
        "candle_transformers",
        "candle_backend",
        "candle_shards",
        "candle::",
    ];
    for (file, src) in sources {
        let code = &src[..src.find("#[cfg(test)]").unwrap_or(src.len())];
        for (n, line) in code.lines().enumerate() {
            let l = line.trim_start();
            if l.starts_with("//") {
                continue;
            }
            if let Some(name) = names.iter().find(|w| l.contains(*w)) {
                panic!("{file}:{}: names {name}: {line}", n + 1);
            }
        }
    }
}

/// No log statement of the backend takes the caller's text: the rule
/// `x0ne-infer` holds for its own sources.
#[test]
fn no_log_statement_names_the_text() {
    let src = include_str!("../core_backend.rs");
    let macros = [
        "println!",
        "eprintln!",
        "print!",
        "eprint!",
        "dbg!",
        "tracing::",
        "log::",
    ];
    let lines: Vec<&str> = src.lines().collect();
    for (n, line) in lines.iter().enumerate() {
        if !macros.iter().any(|m| line.contains(m)) {
            continue;
        }
        // The statement up to its closing `);`.
        let end = (n..lines.len())
            .find(|&k| lines[k].trim_end().ends_with(");"))
            .unwrap_or(n);
        let stmt = lines[n..=end].join("\n").to_lowercase();
        for w in ["prompt", "text", "completion", "logits", "ids", "token"] {
            assert!(
                !stmt.contains(w),
                "core_backend.rs:{}: names {w}: {stmt}",
                n + 1
            );
        }
    }
}

#[cfg(feature = "cuda")]
mod linked_tests {
    use super::*;
    use crate::backend::WeightFiles;
    use crate::backends::{NamedBackend, Registry};
    use crate::model_spec::ModelArtifacts;

    /// `core` is the default, first in the list; candle, the tests'
    /// reference, follows it.
    #[test]
    fn core_is_the_default_before_candle() {
        let r = Registry::default();
        assert_eq!(r.names(), vec!["core", "candle"]);
        assert_eq!(
            r.get(crate::backends::DEFAULT).map(|b| b.name()),
            Some("core")
        );
        assert!(r.with(Box::new(CoreEntry)).is_err(), "a second core");
    }

    /// What is refused before the GPU is touched: by `open`, another device or
    /// family; by the loads, empty weights, and sealed ones through `load_files`.
    #[test]
    fn refusals_come_before_the_gpu() {
        let qwen3 = || ModelArtifacts {
            config_json: config(|_| {}),
            tokenizer_json: b"{}".to_vec(),
        };
        for d in ["cpu", "cuda:1"] {
            assert!(CoreEntry.open(qwen3(), d).is_err(), "{d}");
        }
        let other = ModelArtifacts {
            config_json: config(olmo),
            tokenizer_json: b"{}".to_vec(),
        };
        assert!(CoreEntry.open(other, "cuda").is_err(), "olmo2");

        let key = crate::keys::test_released_key(vec![7u8; 32]);
        let backend = CoreEntry.open(qwen3(), "cuda").unwrap();
        let sealed = crate::seal::seal(b"weights", &[1u8; 32], &[0u8; 12]).unwrap();
        let files = WeightFiles {
            index_json: None,
            files: vec![(crate::shards::SINGLE_FILE, &sealed[..])],
        };
        let e = backend
            .load_files(&files, &key)
            .err()
            .expect("sealed through load_files");
        assert!(format!("{e:#}").contains("sealed"), "{e:#}");
        assert!(backend.load(b"", &key).is_err(), "no bytes");
        let empty = WeightFiles {
            index_json: None,
            files: vec![],
        };
        assert!(backend.load_files(&empty, &key).is_err(), "no files");
    }

    /// On a GPU (`--ignored`, `make core-gpu`), with the models it names.
    mod gpu {
        use super::*;
        use crate::backend::{GenerationParams, LoadedModel, NoSink};
        use crate::prompt::Prompt;
        use crate::stepped::{self, LogitsSession};

        fn dir(var: &str) -> PathBuf {
            PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var}")))
        }

        fn artifacts(d: &Path) -> ModelArtifacts {
            let read = |f: &str| crate::fsread::read_model_file(d, f).unwrap();
            ModelArtifacts {
                config_json: read("config.json"),
                tokenizer_json: read("tokenizer.json"),
            }
        }

        /// Core passes the suite candle passes, loaded through the released key.
        fn suite(var: &str) {
            let d = dir(var);
            let weights = crate::shards::map_dir(&d).unwrap();
            let key = crate::keys::test_released_key(vec![7u8; 32]);
            let r = crate::conformance::check(
                &CoreEntry,
                artifacts(&d),
                &weights.as_files(),
                "cuda",
                &key,
                crate::conformance::Probe::chat,
            );
            assert_eq!(r, Ok(()));
        }

        #[test]
        #[ignore = "GPU: X0NE_INFER_CORE_QWEN3_0_6B"]
        fn core_passes_the_suite_0_6b() {
            suite("X0NE_INFER_CORE_QWEN3_0_6B");
        }

        #[test]
        #[ignore = "GPU: X0NE_INFER_CORE_QWEN3_8B"]
        fn core_passes_the_suite_8b() {
            suite("X0NE_INFER_CORE_QWEN3_8B");
        }

        /// The five bench prompts: short, 31 tokens, 4k, 8k, 16k.
        fn prompts(model: &CoreModel) -> Vec<(&'static str, String)> {
            let p = dir("X0NE_INFER_CORE_PROMPTS");
            let text =
                |s: &str| std::fs::read_to_string(p.join(format!("{s}.prompt.txt"))).unwrap();
            let four = text("4k");
            let ids = model
                .tokenizer
                .encode(four.as_str(), true)
                .unwrap()
                .get_ids()
                .to_vec();
            let s31 = model.tokenizer.decode(&ids[..31], false).unwrap();
            vec![
                ("s20", text("short")),
                ("s31", s31),
                ("4k", four),
                ("8k", text("8k")),
                ("16k", text("16k")),
            ]
        }

        fn load(var: &str) -> CoreModel {
            let d = dir(var);
            let weights = crate::shards::map_dir(&d).unwrap();
            CoreModel::load(&artifacts(&d), &weights.as_files(), "cuda").unwrap()
        }

        /// `CoreModel::generate` (greedy, 256: the device path) gives the text
        /// and the count of `Engine::greedy`'s tokens cut at the first EOS.
        #[test]
        #[ignore = "GPU: X0NE_INFER_CORE_QWEN3_8B, X0NE_INFER_CORE_PROMPTS"]
        fn generate_is_cores_greedy() {
            let model = load("X0NE_INFER_CORE_QWEN3_8B");
            let prompts = prompts(&model);
            let mut same = 0;
            for (name, prompt) in &prompts {
                let ids = model
                    .tokenizer
                    .encode(prompt.as_str(), true)
                    .unwrap()
                    .get_ids()
                    .to_vec();
                let mut want = model.engine.lock().unwrap().greedy(&ids, 256).unwrap();
                if let Some(at) = want.iter().position(|t| model.eos_ids.contains(t)) {
                    want.truncate(at);
                }
                let want_text = model.tokenizer.decode(&want, true).unwrap();
                let (got, (pt, ct)) = model.generate(&Prompt::text(prompt.as_str()), 256).unwrap();
                let ok = got == want_text && ct as usize == want.len() && pt as usize == ids.len();
                println!(
                    "MEASURE generate vs greedy {name}: prompt {} tokens, {} generated, identical {ok}",
                    ids.len(),
                    ct
                );
                same += ok as usize;
            }
            assert_eq!(same, prompts.len());
        }

        /// A Core session that never picks on the device: the host reads
        /// each row and feeds each token.
        struct HostOnly<'e>(crate::core_backend::linked::CoreSession<'e>);

        impl LogitsSession for HostOnly<'_> {
            fn logits(&self) -> &[f32] {
                self.0.logits()
            }
            fn step(&mut self, token: u32) -> anyhow::Result<()> {
                self.0.step(token)
            }
            fn finish(self) -> anyhow::Result<()> {
                self.0.finish()
            }
        }

        fn on_host(
            model: &CoreModel,
            prompt: &Prompt,
            max_tokens: usize,
        ) -> crate::backend::Completion {
            let req = stepped::prepare(&model.tokenizer, prompt, max_tokens, model.context_tokens)
                .unwrap();
            let mut guard = model.engine.lock().unwrap();
            let engine = &mut *guard;
            stepped::run(
                &model.tokenizer,
                &model.eos_ids,
                &req,
                &GenerationParams::greedy(max_tokens),
                &mut NoSink,
                move |ids| {
                    let engine = engine;
                    Ok(HostOnly(
                        crate::core_backend::linked::CoreSession::Stepping(engine.session(ids)?),
                    ))
                },
            )
            .unwrap()
        }

        /// The host path (each row read, argmax on the host) gives the device
        /// path's answer; the time of each, on the same engine.
        #[test]
        #[ignore = "GPU: X0NE_INFER_CORE_QWEN3_8B, X0NE_INFER_CORE_PROMPTS"]
        fn the_device_path_is_the_host_path() {
            let model = load("X0NE_INFER_CORE_QWEN3_8B");
            let mut same = 0;
            let prompts = prompts(&model);
            for (name, prompt) in &prompts {
                let prompt = Prompt::text(prompt.as_str());
                let device = GenerationParams::greedy(256);
                let t0 = std::time::Instant::now();
                let d = model
                    .generate_streaming(&prompt, &device, &mut NoSink)
                    .unwrap();
                let td = t0.elapsed().as_secs_f64();
                let t1 = std::time::Instant::now();
                let h = on_host(&model, &prompt, 256);
                let th = t1.elapsed().as_secs_f64();
                let ok = d.text == h.text && d.completion_tokens == h.completion_tokens;
                println!(
                    "MEASURE device vs host {name}: {} tokens, identical {ok}, device {:.3} s, host {:.3} s",
                    d.completion_tokens, td, th
                );
                same += ok as usize;
            }
            assert_eq!(same, prompts.len());
        }
    }
}
