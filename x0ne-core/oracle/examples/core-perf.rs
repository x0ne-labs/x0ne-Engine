//! A short measure of the engine on one model directory, for the performance gate
//! (`ci/core-perf.sh`): a prefill of 4096 tokens, 256 greedy tokens after it, and the time to
//! the first token of a short prompt. The prompt is token ids drawn from a fixed seed: no
//! tokenizer, the same ids on every run.
//!
//!   core-perf <model directory> <output.json> [--prompt N] [--decode N] [--rounds N]
//!
//! Needs the GPU. Each quantity is measured `rounds` times after one warm-up round and
//! reported as the median, with the smallest and the largest. Nothing here judges the numbers.

use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;
use x0ne_core_runtime::engine::{Engine, EngineOptions};

/// splitmix64, as in the tests.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % n as u64) as usize
    }
}

fn spread(mut v: Vec<f64>) -> serde_json::Value {
    v.sort_by(f64::total_cmp);
    json!({ "median": v[v.len() / 2], "min": v[0], "max": v[v.len() - 1] })
}

fn usage() -> ! {
    eprintln!(
        "usage: core-perf <model directory> <output.json> [--prompt N] [--decode N] [--rounds N]"
    );
    std::process::exit(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(model), Some(out)) = (args.next(), args.next()) else {
        usage()
    };
    let (model, out) = (PathBuf::from(model), PathBuf::from(out));
    let (mut prompt_len, mut decode, mut rounds) = (4096usize, 256usize, 3usize);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0);
        match (flag.as_str(), value) {
            ("--prompt", Some(v)) => prompt_len = v,
            ("--decode", Some(v)) if v > 1 => decode = v,
            ("--rounds", Some(v)) => rounds = v,
            _ => usage(),
        }
    }
    const SHORT: usize = 16;

    let t = Instant::now();
    let mut e = Engine::load(
        &model,
        EngineOptions {
            max_ctx: prompt_len + decode,
        },
    )?;
    let load_s = t.elapsed().as_secs_f64();
    let vocab = e.config().vocab_size;
    let mut rng = Rng(0x0C0E_5EED);
    let prompt: Vec<u32> = (0..prompt_len).map(|_| rng.below(vocab) as u32).collect();

    let (mut prefill, mut ttft, mut rate, mut ttft_short) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut tokens = Vec::new();
    for round in 0..=rounds {
        let (out_tokens, times) = e.greedy_timed(&prompt, decode)?;
        let (_, short) = e.greedy_timed(&prompt[..SHORT], 1)?;
        if round == 0 {
            // The warm-up round: its tokens are the ones every later round must give.
            tokens = out_tokens;
            continue;
        }
        if out_tokens != tokens {
            return Err(
                "two rounds gave different tokens: the measure is not of one computation".into(),
            );
        }
        ttft.push(times[0]);
        prefill.push(prompt_len as f64 / times[0]);
        rate.push((decode - 1) as f64 / (times[decode - 1] - times[0]));
        ttft_short.push(short[0]);
    }

    let (kernel_set, nvcc) = x0ne_core_runtime::kernel_set();
    let c = e.config();
    let report = json!({
        "schema": 1,
        "model": model.file_name().map(|n| n.to_string_lossy().into_owned()),
        "config": {
            "hidden_size": c.hidden_size,
            "num_hidden_layers": c.num_hidden_layers,
            "num_attention_heads": c.num_attention_heads,
            "num_key_value_heads": c.num_key_value_heads,
            "vocab_size": c.vocab_size,
        },
        "engine": e.describe(),
        "kernel_set": kernel_set,
        "nvcc": nvcc,
        "prompt_tokens": prompt_len,
        "decode_tokens": decode,
        "short_prompt_tokens": SHORT,
        "rounds": rounds,
        "load_s": load_s,
        "metrics": {
            "prefill_tok_s": spread(prefill),
            "decode_tok_s": spread(rate),
            "ttft_s": spread(ttft),
            "ttft_short_s": spread(ttft_short),
        },
    });
    let text = serde_json::to_string_pretty(&report)?;
    std::fs::write(&out, format!("{text}\n"))?;
    println!("{text}");
    Ok(())
}
