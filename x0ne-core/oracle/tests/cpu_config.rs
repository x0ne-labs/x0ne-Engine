//! `config.json` read strictly, and what `Engine::load_from` refuses before it opens a GPU: a
//! config the kernels do not serve, a context out of range, a checkpoint that is not the
//! config's. No GPU: a load that passes every check ends on the missing driver, and on a machine
//! with a GPU it loads.

mod common;

use common::{file_of, model_tensors, tiny, Rng, TempDir};
use serde_json::{json, Value};
use x0ne_core_runtime::config::{read_any, Qwen3Config, Qwen3MoeConfig};
use x0ne_core_runtime::engine::{Engine, EngineOptions};
use x0ne_core_runtime::st::SliceSource;
use x0ne_core_runtime::Error;

fn dense() -> Value {
    json!({
        "model_type": "qwen3",
        "dtype": "bfloat16",
        "hidden_act": "silu",
        "attention_bias": false,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 384,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": false,
        "architectures": ["Qwen3ForCausalLM"],
        "use_cache": true
    })
}

fn moe() -> Value {
    let mut v = dense();
    v["model_type"] = json!("qwen3_moe");
    v["decoder_sparse_step"] = json!(1);
    v["norm_topk_prob"] = json!(true);
    v["num_experts"] = json!(8);
    v["num_experts_per_tok"] = json!(2);
    v["moe_intermediate_size"] = json!(16);
    v["mlp_only_layers"] = json!([]);
    v
}

fn with(mut v: Value, key: &str, value: Value) -> String {
    v[key] = value;
    v.to_string()
}

fn without(mut v: Value, key: &str) -> String {
    v.as_object_mut().unwrap().remove(key);
    v.to_string()
}

fn refused(what: &str, text: &str) {
    match Qwen3Config::from_json(text) {
        Err(Error::Format(_)) => {}
        other => panic!("{what}: {other:?}"),
    }
}

const SIZES: [&str; 8] = [
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "max_position_embeddings",
];

#[test]
fn a_qwen3_config_reads_into_its_fields() {
    let c = Qwen3Config::from_json(&dense().to_string()).expect("the config itself");
    let want = Qwen3Config {
        hidden_size: 64,
        intermediate_size: 96,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 128,
        vocab_size: 384,
        max_position_embeddings: 2048,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        tie_word_embeddings: false,
    };
    assert_eq!(c, want);
    assert_eq!((c.q_dim(), c.kv_dim(), c.qkv_rows()), (512, 256, 1024));
    assert_eq!(c.kv_bytes_per_position(), 2 * 2 * 256 * 2);
    assert_eq!(c.norm_params_per_layer(), 2 * 64 + 2 * 128);
    assert_eq!(
        c.linear_params_per_layer(),
        64 * 1024 + 512 * 64 + 3 * 64 * 96
    );
    // The spellings that mean the same.
    let older = with(
        serde_json::from_str(&without(dense(), "dtype")).unwrap(),
        "torch_dtype",
        json!("bfloat16"),
    );
    assert_eq!(Qwen3Config::from_json(&older).unwrap(), want);
    for key in ["attention_bias", "use_sliding_window", "rope_scaling"] {
        for off in [json!(false), Value::Null] {
            assert_eq!(
                Qwen3Config::from_json(&with(dense(), key, off)).unwrap(),
                want,
                "{key}"
            );
        }
        assert_eq!(
            Qwen3Config::from_json(&without(dense(), key)).unwrap(),
            want,
            "{key}"
        );
    }
    assert!(
        !Qwen3Config::from_json(&without(dense(), "tie_word_embeddings"))
            .unwrap()
            .tie_word_embeddings
    );
    assert!(
        Qwen3Config::from_json(&with(dense(), "tie_word_embeddings", json!(true)))
            .unwrap()
            .tie_word_embeddings
    );
    // An integer where a number is asked is a number.
    assert_eq!(
        Qwen3Config::from_json(&with(dense(), "rope_theta", json!(1000000))).unwrap(),
        want
    );
}

/// A size is a positive integer that fits the machine: nothing else is read as one.
#[test]
fn a_size_that_is_not_a_positive_integer_is_refused() {
    let bad = [
        "0",
        "-1",
        "-0",
        "1.0",
        "1.5",
        "1e3",
        "\"64\"",
        "null",
        "true",
        "[64]",
        "{}",
        "18446744073709551616",
        "1e400",
        "-9223372036854775808",
    ];
    for key in SIZES {
        refused(&format!("{key} missing"), &without(dense(), key));
        for value in bad {
            let text = dense().to_string().replacen(
                &format!("\"{key}\":{}", dense()[key]),
                &format!("\"{key}\":{value}"),
                1,
            );
            assert!(
                text.contains(&format!("\"{key}\":{value}")),
                "{key} not replaced"
            );
            refused(&format!("{key} = {value}"), &text);
        }
    }
    for key in ["rms_norm_eps", "rope_theta"] {
        refused(&format!("{key} missing"), &without(dense(), key));
        for value in [
            json!(0),
            json!(0.0),
            json!(-1e-6),
            json!("1e-6"),
            Value::Null,
            json!(false),
            json!([1.0]),
        ] {
            refused(&format!("{key} = {value}"), &with(dense(), key, value));
        }
        // serde_json refuses what is not a finite number before the config sees it.
        for value in ["NaN", "Infinity", "-Infinity", "1e999", "nan", "inf"] {
            let text = dense().to_string().replacen(
                &format!("\"{key}\":{}", dense()[key]),
                &format!("\"{key}\":{value}"),
                1,
            );
            assert!(text.contains(value), "{key} not replaced");
            refused(&format!("{key} = {value}"), &text);
        }
    }
}

/// What this engine does not implement is refused by name, not served wrong.
#[test]
fn a_model_this_engine_does_not_implement_is_refused() {
    for (key, values) in [
        (
            "model_type",
            vec![
                json!("qwen2"),
                json!("qwen3_moe"),
                json!("Qwen3"),
                json!(""),
                Value::Null,
                json!(3),
            ],
        ),
        (
            "dtype",
            vec![
                json!("float32"),
                json!("float16"),
                json!("bf16"),
                json!(""),
                Value::Null,
                json!(16),
            ],
        ),
        (
            "hidden_act",
            vec![
                json!("gelu"),
                json!("relu"),
                json!("SiLU"),
                Value::Null,
                json!(["silu"]),
            ],
        ),
        (
            "attention_bias",
            vec![json!(true), json!(1), json!("false"), json!(0)],
        ),
        ("use_sliding_window", vec![json!(true), json!(4096)]),
        (
            "rope_scaling",
            vec![
                json!({"type": "yarn", "factor": 4.0}),
                json!(true),
                json!({}),
            ],
        ),
        (
            "tie_word_embeddings",
            vec![json!("true"), json!(1), json!(0), Value::Null, json!([])],
        ),
    ] {
        for value in values {
            refused(&format!("{key} = {value}"), &with(dense(), key, value));
        }
    }
    refused("model_type missing", &without(dense(), "model_type"));
    refused("hidden_act missing", &without(dense(), "hidden_act"));
    refused("no dtype at all", &without(dense(), "dtype"));
    // "dtype" wins over "torch_dtype": a config that says both is read by the newer key.
    refused("dtype float32 over torch_dtype bfloat16", &{
        let mut v = dense();
        v["torch_dtype"] = json!("bfloat16");
        with(v, "dtype", json!("float32"))
    });
    for (q, kv) in [(4, 3), (2, 4), (5, 2), (1, 2), (7, 3)] {
        let v: Value =
            serde_json::from_str(&with(dense(), "num_attention_heads", json!(q))).unwrap();
        refused(
            &format!("{q} heads over {kv} kv heads"),
            &with(v, "num_key_value_heads", json!(kv)),
        );
    }
    for odd in [1, 3, 127, 129] {
        refused(
            &format!("head_dim {odd}"),
            &with(dense(), "head_dim", json!(odd)),
        );
    }
    refused(
        "a vocabulary past u32",
        &with(dense(), "vocab_size", json!(1u64 << 32)),
    );
    assert!(Qwen3Config::from_json(&with(dense(), "vocab_size", json!(u32::MAX))).is_ok());
    for text in [
        "",
        " ",
        "null",
        "[]",
        "\"qwen3\"",
        "42",
        "{",
        "{}",
        "{\"model_type\":\"qwen3\"}",
        "\u{feff}{}",
    ] {
        refused(&format!("{text:?}"), text);
    }
}

/// Cut anywhere, or with any one byte changed, a config is read or refused, never a panic; a
/// cut config is never read.
#[test]
fn a_cut_or_corrupted_config_never_panics() {
    let text = dense().to_string();
    for len in 0..text.len() {
        if text.is_char_boundary(len) {
            refused(&format!("cut at {len}"), &text[..len]);
        }
    }
    let mut rng = Rng::new(31);
    let (mut read, mut not) = (0, 0);
    for _ in 0..40_000 {
        let mut bytes = text.clone().into_bytes();
        for _ in 0..rng.range(1, 3) {
            let at = rng.below(bytes.len());
            bytes[at] = match rng.below(4) {
                0 => *rng.pick(b"0123456789"),
                1 => *rng.pick(b"{}[],:\"-+.eE "),
                2 => bytes[at] ^ (1 << rng.below(7)),
                _ => rng.range(0x20, 0x7E) as u8,
            };
        }
        let Ok(s) = String::from_utf8(bytes) else {
            continue;
        };
        for moe in [false, true] {
            let r = if moe {
                Qwen3MoeConfig::from_json(&s).map(|m| m.base)
            } else {
                Qwen3Config::from_json(&s)
            };
            match r {
                Ok(c) => {
                    read += 1;
                    // What was read satisfies what the reader promises.
                    assert!(
                        c.num_key_value_heads > 0
                            && c.num_attention_heads.is_multiple_of(c.num_key_value_heads)
                    );
                    assert!(c.head_dim.is_multiple_of(2) && c.vocab_size <= u32::MAX as usize);
                    assert!(c.rms_norm_eps > 0.0 && c.rope_theta > 0.0);
                    assert!([
                        c.hidden_size,
                        c.intermediate_size,
                        c.num_hidden_layers,
                        c.max_position_embeddings
                    ]
                    .iter()
                    .all(|&v| v > 0));
                }
                Err(Error::Format(_)) => not += 1,
                Err(e) => panic!("{s}: {e}"),
            }
        }
        let _ = read_any(&s);
    }
    eprintln!("MEASURE corrupted configs: {not} refused, {read} read and checked");
    assert!(read > 100 && not > 100);
}

#[test]
fn a_qwen3_moe_config_reads_strictly() {
    let c = Qwen3MoeConfig::from_json(&moe().to_string()).expect("the config itself");
    assert_eq!(
        (
            c.moe.num_experts,
            c.moe.num_experts_per_tok,
            c.moe.moe_intermediate_size
        ),
        (8, 2, 16)
    );
    assert_eq!(c.base.hidden_size, 64);
    let bad = |what: &str, text: &str| match Qwen3MoeConfig::from_json(text) {
        Err(Error::Format(_)) => {}
        other => panic!("{what}: {other:?}"),
    };
    bad("a dense config", &dense().to_string());
    for (key, values) in [
        (
            "shared_expert_intermediate_size",
            vec![json!(64), json!(0), json!(false)],
        ),
        ("n_shared_experts", vec![json!(1), json!(0)]),
        (
            "decoder_sparse_step",
            vec![json!(2), json!(0), Value::Null, json!("1")],
        ),
        (
            "mlp_only_layers",
            vec![json!([0]), json!(0), json!({}), json!(false)],
        ),
        (
            "norm_topk_prob",
            vec![json!(false), Value::Null, json!(1), json!("true")],
        ),
        (
            "num_experts",
            vec![
                json!(0),
                json!(1),
                json!(-8),
                json!(8.5),
                json!(1u64 << 32),
                Value::Null,
            ],
        ),
        (
            "num_experts_per_tok",
            vec![json!(0), json!(9), json!(-1), json!("2")],
        ),
        (
            "moe_intermediate_size",
            vec![json!(0), json!(-16), json!(16.5), Value::Null],
        ),
    ] {
        for value in values {
            bad(&format!("{key} = {value}"), &with(moe(), key, value));
        }
    }
    for key in [
        "decoder_sparse_step",
        "norm_topk_prob",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    ] {
        bad(&format!("{key} missing"), &without(moe(), key));
    }
    for (key, ok) in [
        ("mlp_only_layers", Value::Null),
        ("shared_expert_intermediate_size", Value::Null),
        ("n_shared_experts", Value::Null),
    ] {
        assert!(
            Qwen3MoeConfig::from_json(&with(moe(), key, ok)).is_ok(),
            "{key} null"
        );
        assert!(
            Qwen3MoeConfig::from_json(&without(moe(), key)).is_ok(),
            "{key} absent"
        );
    }
    // A truncation keeps everything but the layer count, and stays inside the model.
    assert_eq!(c.truncated(1).unwrap().base.num_hidden_layers, 1);
    assert_eq!(c.truncated(2).unwrap(), c);
    for layers in [0, 3, usize::MAX] {
        assert!(
            matches!(c.truncated(layers), Err(Error::Format(_))),
            "{layers} layers"
        );
    }
}

#[test]
fn read_any_tells_dense_from_moe_and_refuses_the_rest() {
    let (c, m) = read_any(&dense().to_string()).unwrap();
    assert!(m.is_none());
    assert_eq!(c, Qwen3Config::from_json(&dense().to_string()).unwrap());
    let (c, m) = read_any(&moe().to_string()).unwrap();
    assert_eq!(Some(&c), m.as_ref().map(|m| &m.base));
    for model_type in ["qwen2", "qwen2_moe", "llama", "", "qwen3_moe "] {
        assert!(
            matches!(
                read_any(&with(moe(), "model_type", json!(model_type))),
                Err(Error::Format(_))
            ),
            "{model_type:?}"
        );
    }
    assert!(
        read_any(&with(dense(), "model_type", json!("qwen3_moe"))).is_err(),
        "a MoE without its fields"
    );
    assert!(read_any("").is_err() && read_any("[]").is_err());
}

/// The checkpoint of the `tiny` model, and its config text.
fn tiny_checkpoint() -> (String, Vec<u8>) {
    let c = tiny(2, 4, 2, false);
    (c.to_json(), file_of(&model_tensors(&c, &mut Rng::new(41))))
}

fn load(config: &str, file: &[u8], max_ctx: usize) -> Result<Engine, Error> {
    let src = SliceSource::new(None, &[("model.safetensors", file)])?;
    Engine::load_from(config, &src, EngineOptions { max_ctx })
}

/// A load that passed every check: it failed on the driver or the device (no GPU here), or it
/// loaded (a GPU is here).
fn passed_the_checks(what: &str, r: Result<Engine, Error>) {
    match r {
        Ok(_) | Err(Error::Gpu(_)) => {}
        Err(e) => panic!("{what}: refused before the GPU: {e}"),
    }
}

/// What the kernels do not serve, and a checkpoint that is not the config's, are refused as a
/// format error whether or not a GPU is present: the checks run before the device is opened.
#[test]
fn a_load_is_refused_on_its_config_and_checkpoint_before_any_gpu() {
    let (config, file) = tiny_checkpoint();
    passed_the_checks("the model itself", load(&config, &file, 64));
    passed_the_checks("the whole context", load(&config, &file, 2048));
    passed_the_checks("a context of one", load(&config, &file, 1));
    let refused = |what: &str, r: Result<Engine, Error>| match r {
        Err(Error::Format(_)) => {}
        Err(e) => panic!("{what}: {e}"),
        Ok(_) => panic!("{what}: loaded"),
    };
    refused("max_ctx 0", load(&config, &file, 0));
    refused("max_ctx past the positions", load(&config, &file, 2049));
    refused("max_ctx usize::MAX", load(&config, &file, usize::MAX));
    let v: Value = serde_json::from_str(&config).unwrap();
    for (key, value) in [
        ("head_dim", json!(64)),
        ("head_dim", json!(256)),
        ("hidden_size", json!(60)),
        ("hidden_size", json!(68)),
        ("hidden_size", json!(72)),
        ("intermediate_size", json!(104)),
        ("num_hidden_layers", json!(1)),
        ("num_hidden_layers", json!(3)),
        ("num_attention_heads", json!(2)),
        ("num_attention_heads", json!(8)),
        ("vocab_size", json!(383)),
        ("vocab_size", json!(385)),
        ("tie_word_embeddings", json!(true)),
        ("model_type", json!("qwen2")),
        ("dtype", json!("float32")),
    ] {
        refused(
            &format!("{key} = {value}"),
            load(&with(v.clone(), key, value), &file, 64),
        );
    }
    // A checkpoint cut short, or of another model.
    refused(
        "a truncated checkpoint",
        load(&config, &file[..file.len() - 2], 64),
    );
    let other = file_of(&model_tensors(&tiny(1, 4, 2, false), &mut Rng::new(43)));
    refused(
        "the checkpoint of a one-layer model",
        load(&config, &other, 64),
    );
    refused("no checkpoint", load(&config, &[], 64));
    // A MoE config over a dense checkpoint.
    let mut m = moe();
    m["hidden_size"] = json!(64);
    refused(
        "a MoE config over dense weights",
        load(&m.to_string(), &file, 64),
    );
    for (key, value) in [
        ("num_experts", json!(1025)),
        ("num_experts_per_tok", json!(33)),
        ("moe_intermediate_size", json!(12)),
    ] {
        let mut big = moe();
        big["num_experts"] = json!(1024);
        refused(
            &format!("{key} = {value}"),
            load(&with(big, key, value), &file, 64),
        );
    }
}

/// `Engine::load` on a directory: the same refusals, read from disk.
#[test]
fn a_model_directory_is_refused_on_what_it_lacks() {
    let (config, file) = tiny_checkpoint();
    let opts = EngineOptions { max_ctx: 64 };
    let dir = TempDir::new("load");
    assert!(
        matches!(Engine::load(dir.path(), opts), Err(Error::Io(_))),
        "no config.json"
    );
    std::fs::write(dir.path().join("config.json"), "{").unwrap();
    assert!(
        matches!(Engine::load(dir.path(), opts), Err(Error::Format(_))),
        "a cut config.json"
    );
    std::fs::write(dir.path().join("config.json"), &config).unwrap();
    assert!(
        matches!(Engine::load(dir.path(), opts), Err(Error::Format(_))),
        "no weights"
    );
    std::fs::write(
        dir.path().join("model.safetensors"),
        &file[..file.len() / 2],
    )
    .unwrap();
    assert!(
        matches!(Engine::load(dir.path(), opts), Err(Error::Format(_))),
        "cut weights"
    );
    std::fs::write(dir.path().join("model.safetensors"), &file).unwrap();
    assert!(matches!(
        Engine::load(dir.path(), EngineOptions { max_ctx: 0 }),
        Err(Error::Format(_))
    ));
    passed_the_checks("the directory itself", Engine::load(dir.path(), opts));
    assert_eq!(
        Qwen3Config::from_file(&dir.path().join("config.json"))
            .unwrap()
            .hidden_size,
        64
    );
    assert!(matches!(
        Qwen3Config::from_file(&dir.path().join("absent.json")),
        Err(Error::Io(_))
    ));
}

/// Sizes a config can state and no machine can hold. The load computed `intermediate_size * 2`,
/// `num_attention_heads * head_dim` and `max_ctx * kv_dim` in usize before it compared them
/// with u32 (engine.rs `check_config`, config.rs `q_dim`): with overflow checks that was a
/// panic; without (the release profile of the server) the product wrapped and could pass.
/// Fixed in x0ne Core and copied on 2026-11-10: the products are checked.
fn overflowing() -> Vec<(&'static str, String, usize)> {
    vec![
        (
            "intermediate_size 2^63",
            with(dense(), "intermediate_size", json!(1u64 << 63)),
            64,
        ),
        (
            "num_attention_heads 2^57",
            {
                let v: Value =
                    serde_json::from_str(&with(dense(), "num_attention_heads", json!(1u64 << 57)))
                        .unwrap();
                with(v, "num_key_value_heads", json!(1u64 << 57))
            },
            64,
        ),
        (
            "num_key_value_heads 2^56 at max_ctx 2^20",
            {
                let mut v = dense();
                v["num_attention_heads"] = json!(1u64 << 56);
                v["num_key_value_heads"] = json!(1u64 << 56);
                with(v, "max_position_embeddings", json!(1u64 << 40))
            },
            1 << 20,
        ),
    ]
}

#[test]
fn a_config_whose_sizes_overflow_is_refused_not_a_panic() {
    let (_, file) = tiny_checkpoint();
    for (what, config, max_ctx) in overflowing() {
        let r = std::panic::catch_unwind(|| load(&config, &file, max_ctx).map(|_| ()));
        match r {
            Ok(Err(Error::Format(_))) => {}
            Ok(other) => panic!("{what}: {other:?}"),
            Err(_) => panic!("{what}: the load panicked instead of refusing"),
        }
    }
}

/// A config may state any number of layers. The list of expected tensors was built for all of
/// them before the first was looked up (weights.rs `expected`, called by `check`): the time
/// and memory of a refusal grew with a number the config's author chose, 1.3 s for 300 000
/// layers. Fixed in x0ne Core and copied on 2026-11-10: `check` stops at the first tensor the
/// checkpoint lacks.
#[test]
fn a_config_with_many_layers_is_refused_without_listing_them() {
    let (_, file) = tiny_checkpoint();
    let time = |layers: u64| {
        let config = with(dense(), "num_hidden_layers", json!(layers));
        let t = std::time::Instant::now();
        assert!(matches!(load(&config, &file, 64), Err(Error::Format(_))));
        t.elapsed().as_secs_f64()
    };
    let (small, large) = (time(3), time(300_000));
    eprintln!("MEASURE refusal of 3 layers {small:.6} s, of 300000 layers {large:.6} s");
    assert!(
        large < 0.01 + 20.0 * small,
        "a refusal took {large:.3} s for 300000 layers, {small:.6} s for 3"
    );
}
