//! Places decode measurements against the bandwidth ceiling of a model on a GPU (active bytes
//! for a qwen3_moe config).
//!
//! x0ne-core-roofline --config <config.json> --gpu <key|GB/s> --tsv <measurements.tsv>
//! x0ne-core-roofline --config <config.json> --gpu <key|GB/s> --prompt P --generated N [--tok-s X]

use std::process::ExitCode;
use x0ne_core_runtime::config;
use x0ne_core_runtime::roofline;

fn usage() -> ExitCode {
    let keys: Vec<&str> = roofline::GPUS.iter().map(|g| g.key).collect();
    eprintln!(
        "usage: x0ne-core-roofline --config <config.json> --gpu <{}|GB/s>",
        keys.join("|")
    );
    eprintln!("         (--tsv <label\\tprompt\\tgenerated\\ttok_s file> | --prompt P --generated N [--tok-s X])");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut opt = std::collections::HashMap::new();
    let mut it = args.iter();
    while let Some(k) = it.next() {
        let Some(v) = it.next() else { return usage() };
        if !k.starts_with("--") {
            return usage();
        }
        opt.insert(k.trim_start_matches("--").to_owned(), v.clone());
    }
    let (Some(config), Some(gpu)) = (opt.get("config"), opt.get("gpu")) else {
        return usage();
    };
    let read = std::fs::read_to_string(config)
        .map_err(|e| e.to_string())
        .and_then(|t| config::read_any(&t).map_err(|e| e.to_string()));
    let (cfg, moe) = match read {
        Ok(c) => c,
        Err(e) => {
            eprintln!("x0ne-core-roofline: {config}: {e}");
            return ExitCode::from(1);
        }
    };
    let (label, bw) = match roofline::gpu(gpu) {
        Some(g) => (format!("{} ({})", g.name, g.source), g.bandwidth_gb_s),
        None => match gpu.parse::<f64>() {
            Ok(x) if x > 0.0 => (format!("{x} GB/s given"), x),
            _ => return usage(),
        },
    };
    let rows = if let Some(path) = opt.get("tsv") {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("x0ne-core-roofline: {path}: {e}");
                return ExitCode::from(1);
            }
        };
        match roofline::parse_tsv(&text) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("x0ne-core-roofline: {path}: {e}");
                return ExitCode::from(1);
            }
        }
    } else {
        let num = |k: &str| opt.get(k).and_then(|v| v.parse::<u64>().ok());
        let (Some(p), Some(n)) = (num("prompt"), num("generated")) else {
            return usage();
        };
        let tok_s = opt
            .get("tok-s")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(f64::NAN);
        vec![("given".to_owned(), p, n, tok_s)]
    };
    let placed: Vec<_> = rows
        .iter()
        .map(|(l, p, n, t)| roofline::place_of(&cfg, moe.as_ref(), bw, l, *p, *n, *t))
        .collect();
    print!("{}", roofline::render(&label, bw, &placed));
    ExitCode::SUCCESS
}
