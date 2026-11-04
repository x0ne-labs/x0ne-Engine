//! Bench of 2026-11-04 (not part of x0ne Core; added in the bench kit's copy only): per-token timestamps of greedy
//! decode, for TTFT, ITL distribution and E2E. FILE is a JSON list of prompts (lists of token ids); prompt 0 is the
//! warm-up, recorded with "warmup": true. One JSON line per run to --out, flushed at once: a cut keeps finished runs.
//! times[i] is the time from the call to token i (Engine::greedy_timed); prefill is in times[0].

use std::io::Write;
use std::path::PathBuf;
use x0ne_core_runtime::engine::{DecodeMode, Engine, EngineOptions};
use x0ne_core_tools::{die, Flags};

const USAGE: &str =
    "x0ne-core-itl --model DIR --prompts LABEL=FILE... --gen N --out FILE.jsonl [--tag T] [--decode graph|eager]";

fn main() {
    let f = Flags::parse(USAGE);
    let model = PathBuf::from(f.req("model"));
    let gen = f.num("gen", 256);
    let out = f.req("out").to_owned();
    let tag = f.get("tag").unwrap_or("-").to_owned();
    let mut sets: Vec<(String, Vec<Vec<u32>>)> = Vec::new();
    for p in f.all("prompts") {
        let (label, file) = p.split_once('=').unwrap_or_else(|| die(USAGE));
        let text = std::fs::read_to_string(file).unwrap_or_else(|e| die(&format!("{file}: {e}")));
        let v: Vec<Vec<u32>> = serde_json::from_str(&text).unwrap_or_else(|e| die(&format!("{file}: {e}")));
        if v.len() < 2 {
            die(&format!("{file}: a warm-up and at least one measured prompt are needed"));
        }
        sets.push((label.to_owned(), v));
    }
    if sets.is_empty() || gen < 2 {
        die(USAGE);
    }
    let longest = sets.iter().flat_map(|(_, v)| v.iter().map(|p| p.len())).max().unwrap();
    let mode = match f.get("decode").unwrap_or("graph") {
        "graph" => DecodeMode::Graph,
        "eager" => DecodeMode::Eager,
        other => die(&format!("--decode {other}: graph or eager")),
    };
    let t0 = std::time::Instant::now();
    let mut e = Engine::load(&model, EngineOptions { max_ctx: longest + gen })
        .unwrap_or_else(|e| die(&e.to_string()));
    e.set_decode_mode(mode).unwrap_or_else(|e| die(&e.to_string()));
    let load_s = t0.elapsed().as_secs_f64();
    eprintln!("loaded in {load_s:.1} s: {}, decode {mode:?}", e.describe());
    let mut w = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out)
        .unwrap_or_else(|e| die(&format!("{out}: {e}")));
    let line = serde_json::json!({"engine": "x0ne-core", "tool": "x0ne-core-itl", "tag": tag, "event": "load", "load_s": load_s,
        "decode": format!("{mode:?}"), "describe": e.describe()});
    writeln!(w, "{line}").unwrap_or_else(|e| die(&e.to_string()));
    for (label, prompts) in &sets {
        for (i, ids) in prompts.iter().enumerate() {
            let (toks, times) = e.greedy_timed(ids, gen).unwrap_or_else(|e| die(&e.to_string()));
            let line = serde_json::json!({"engine": "x0ne-core", "tool": "x0ne-core-itl", "tag": tag, "label": label, "i": i,
                "warmup": i == 0, "prompt_len": ids.len(), "gen": toks.len(), "times": times});
            writeln!(w, "{line}").unwrap_or_else(|e| die(&e.to_string()));
            w.flush().unwrap_or_else(|e| die(&e.to_string()));
            let n = times.len();
            eprintln!("{label} run {i}{}: ttft {:.4} s, e2e {:.3} s, decode {:.2} tok/s", if i == 0 { " (warm-up)" } else { "" },
                times[0], times[n - 1], (n - 1) as f64 / (times[n - 1] - times[0]));
        }
    }
}
