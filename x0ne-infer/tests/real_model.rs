//! End-to-end with real models — ignored by default. Run:
//!
//! ```bash
//! X0NE_INFER_QWEN3_DIR=/path/to/qwen3-0.6b \
//!   X0NE_MEMCAP=16G scripts/capped cargo test -p x0ne-infer --test real_model -- --ignored --nocapture
//! ```
//!
//! `X0NE_INFER_QWEN3_DIR`: Qwen3-0.6B's `config.json`, `tokenizer.json`,
//! `model.safetensors` (≈1.4 GB, CPU, F32 — slow but real). The acceptance
//! test: candle loads the model behind the trait, generates real tokens, and
//! the digest is the catalog's pin for those files.
//!
//! `X0NE_INFER_QWEN3_1_7B_DIR` (a `pull` of qwen3:1.7b: two shards and their
//! index) and `X0NE_INFER_QWEN3_1_7B_MERGED` (`x0ne-infer merge-shards` of it):
//! the shards as served must generate what the one merged file generates.
//!
//! `X0NE_INFER_PIN_DIR` and `X0NE_INFER_PIN_MODEL`: the served-model digest of a
//! directory, as the engine maps and checks it, against the catalog's pin. Hashes
//! the files without building the model: how a pin is measured on real files.

use x0ne_infer::backend::InferenceBackend;
use x0ne_infer::candle_backend::{CandleBackend, ModelArtifacts};
use x0ne_infer::keys::ReleasedKey;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name}"))
}

fn backend(dir: &str) -> CandleBackend {
    let read = |f: &str| std::fs::read(format!("{dir}/{f}")).unwrap_or_else(|e| panic!("{f}: {e}"));
    CandleBackend::cpu(ModelArtifacts {
        config_json: read("config.json"),
        tokenizer_json: read("tokenizer.json"),
    })
}

fn pin(name: &str) -> String {
    let catalog = x0ne_infer::registry::Catalog::embedded();
    let entry = catalog.get(name).unwrap();
    entry.digest_sha256.clone().expect("pinned")
}

/// The key path is exercised through the same release flow as production.
fn released_key() -> ReleasedKey {
    use x0ne_confidential::model::{
        AttestationEvidence, EvidenceFormat, ExecutionEnvironment, GpuCcMode, TeeTechnology,
        Verdict, VerifierStrength, WorkloadIdentity,
    };
    use x0ne_infer::gate::Gate;
    struct MemBroker;
    impl x0ne_infer::keys::KeyBroker for MemBroker {
        fn unwrap_key(&self, _k: &str) -> anyhow::Result<zeroize::Zeroizing<Vec<u8>>> {
            Ok(zeroize::Zeroizing::new(vec![1u8; 32]))
        }
    }
    let env = ExecutionEnvironment::ConfidentialVmWithGpu {
        cpu: TeeTechnology::IntelTdx,
        gpu_mode: GpuCcMode::On,
    };
    let evidence = AttestationEvidence {
        format: EvidenceFormat::StructuredClaims,
        payload: vec![],
        collected_at_unix: 1_000,
    };
    let proof = Gate::attest(
        env.clone(),
        &evidence,
        Verdict::Verified {
            identity: WorkloadIdentity {
                name: "x0ne-infer".into(),
                environment: env,
                measurements: vec![],
            },
            verifier: "test".into(),
            strength: VerifierStrength::Cryptographic,
        },
        1_000,
    )
    .unwrap();
    x0ne_infer::keys::release(&proof, &evidence, "qwen3-key", &MemBroker, 1_000).unwrap()
}

/// `<|im_start|>user\nSay hello in one short sentence.<|im_end|>\n<|im_start|>assistant\n`,
/// its markers control tokens.
fn prompt() -> x0ne_infer::prompt::Prompt {
    let mut p = x0ne_infer::prompt::Prompt::new();
    p.push_control("<|im_start|>");
    p.push_text("user\nSay hello in one short sentence.");
    p.push_control("<|im_end|>");
    p.push_text("\n");
    p.push_control("<|im_start|>");
    p.push_text("assistant\n");
    p
}

#[test]
#[ignore = "needs X0NE_INFER_QWEN3_DIR with the 1.4 GB model files"]
fn qwen3_generates_real_tokens_on_cpu() {
    let dir = env("X0NE_INFER_QWEN3_DIR");
    let backend = backend(&dir);
    let key = released_key();
    let weights = std::fs::read(format!("{dir}/model.safetensors")).expect("weights");
    let t0 = std::time::Instant::now();
    let model = backend.load(&weights, &key).expect("model loads");
    let load_s = t0.elapsed().as_secs_f32();

    let t1 = std::time::Instant::now();
    let (text, (p, c)) = model.generate(&prompt(), 24).expect("generates");
    let gen_s = t1.elapsed().as_secs_f32();

    println!("load: {load_s:.1}s · generate: {gen_s:.1}s · prompt {p} tok · out {c} tok");
    println!("digest: {}", model.model_digest_hex());
    println!("text: {text:?}");
    assert!(c > 0, "at least one token came out");
    assert!(!text.trim().is_empty(), "the text is not empty");
    assert_eq!(
        model.model_digest_hex(),
        pin("qwen3:0.6b"),
        "files other than the catalog's?"
    );
}

/// A failure means the shards as served are not the model the merged file is:
/// a tensor routed to the wrong shard, or one read twice.
#[test]
#[ignore = "needs qwen3:1.7b (4 GB of shards) and its merge"]
fn qwen3_1_7b_shards_generate_what_the_merged_file_generates() {
    let dir = env("X0NE_INFER_QWEN3_1_7B_DIR");
    let backend = backend(&dir);
    let key = released_key();
    let sharded = {
        let mapped = x0ne_infer::shards::map_dir(std::path::Path::new(&dir)).unwrap();
        assert!(mapped.index_json.is_some() && mapped.files.len() == 2);
        let t0 = std::time::Instant::now();
        let model = backend
            .load_files(&mapped.as_files(), &key)
            .expect("shards load");
        println!("shards: load {:.1}s", t0.elapsed().as_secs_f32());
        assert_eq!(model.model_digest_hex(), pin("qwen3:1.7b"));
        model.generate(&prompt(), 48).unwrap()
    };
    let merged = {
        let bytes = std::fs::read(env("X0NE_INFER_QWEN3_1_7B_MERGED")).unwrap();
        let model = backend.load(&bytes, &key).expect("merged loads");
        assert_ne!(
            model.model_digest_hex(),
            pin("qwen3:1.7b"),
            "another file set"
        );
        model.generate(&prompt(), 48).unwrap()
    };
    println!("shards: {sharded:?}\nmerged: {merged:?}");
    assert!(sharded.1 .1 > 0);
    assert_eq!(sharded, merged);
}

/// A failure means the files are not the catalog's, or the engine digests
/// them otherwise than the pin was computed (`model_digest.rs`).
#[test]
#[ignore = "needs X0NE_INFER_PIN_DIR and X0NE_INFER_PIN_MODEL"]
fn the_engine_digest_of_a_model_dir_is_the_catalog_pin() {
    let dir = std::path::PathBuf::from(env("X0NE_INFER_PIN_DIR"));
    let name = env("X0NE_INFER_PIN_MODEL");
    let read = |f: &str| x0ne_infer::fsread::read_model_file(&dir, f).unwrap();
    let (config, tokenizer) = (read("config.json"), read("tokenizer.json"));
    let t0 = std::time::Instant::now();
    let mapped = x0ne_infer::shards::map_dir(&dir).unwrap();
    let files = mapped.as_files();
    x0ne_infer::shards::Routing::check(files.index_json, &files.files)
        .expect("index and headers agree");
    let digest = x0ne_infer::candle_backend::served_digest(&config, &tokenizer, &files).unwrap();
    println!(
        "{name}: {} weight files, digest {digest} in {:.1}s",
        files.files.len(),
        t0.elapsed().as_secs_f32()
    );
    let catalog = x0ne_infer::registry::Catalog::embedded();
    let entry = catalog.get(&name).unwrap();
    let mut listed: Vec<&str> = entry.weight_files().map(|f| f.path.as_str()).collect();
    let mut mapped_names: Vec<&str> = files.files.iter().map(|(n, _)| *n).collect();
    listed.sort_unstable();
    mapped_names.sort_unstable();
    assert_eq!(mapped_names, listed, "the weight files are the catalog's");
    assert_eq!(Some(digest), entry.digest_sha256.clone(), "{name}");
}
