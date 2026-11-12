//! The core tier of `make bench` (`scripts/bench.sh`): x0ne Core alone on one model directory,
//! nothing between the caller and the engine. Per length of `--lengths`, the time to the first
//! token and the prefill and decode rates of `Engine::greedy_timed`, over `--reps` repetitions
//! after a first pass reported apart; the cold load, the arena and the VRAM taken.
//!
//!   core-bench <model directory> [--prompts DIR] [--lengths short,1k,4k,8k,16k,max]
//!              [--decode N] [--reps N]
//!
//! A length is the ids of `<name>.ids.json` in DIR (the prompt kit: its chat template applied).
//! Without that file, `1k` is the first 1024 ids of `4k.ids.json`, and `short` (20), `<n>k`
//! (n * 1024) and `<n>` are ids drawn from a fixed seed (splitmix64, as core-perf and
//! x0ne-infer's bench_tests.rs draw them). `max` is drawn too, as long as the largest context
//! the card holds allows: contexts are tried from `max_position_embeddings` down, and one the
//! card refuses for memory is written down, not a failure. When none fits, the result says
//! `refused`. Prints one line `x0ne-bench-result: <JSON>` (schema 1, kind "core") on stdout,
//! its progress on stderr. Needs the GPU; nothing here judges the numbers.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;
use x0ne_core_runtime::engine::{Engine, EngineOptions};

const USAGE: &str = "usage: core-bench <model directory> [--prompts DIR] [--lengths a,b,...] [--decode N] [--reps N]";

/// splitmix64, as in core-perf and the tests.
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

/// `n` ids below `vocab` from the fixed seed: the same on every run, here and in x0ne-infer's
/// bench_tests.rs, whose test pins the same first six.
fn synthetic(n: usize, vocab: usize) -> Vec<u32> {
    let mut rng = Rng(0x0C0E_5EED);
    (0..n).map(|_| rng.below(vocab) as u32).collect()
}

/// Median, smallest, largest, mean and standard deviation (n - 1) of the repetitions.
fn stats(v: &[f64]) -> Value {
    if v.is_empty() {
        return Value::Null;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    let median = if n % 2 == 1 {
        s[n / 2]
    } else {
        f64::midpoint(s[n / 2 - 1], s[n / 2])
    };
    let mean = s.iter().sum::<f64>() / n as f64;
    let stddev = if n > 1 {
        (s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
    } else {
        0.0
    };
    json!({"median": median, "min": s[0], "max": s[n - 1], "mean": mean, "stddev": stddev, "n": n})
}

/// (time to the first token, prefill rate, decode rate) of one `greedy_timed`: `times[i]` the
/// second token i was on the host, from the call; `prompt` ids before it.
fn rates(times: &[f64], prompt: usize) -> (f64, f64, f64) {
    let (first, last) = (times[0], times[times.len() - 1]);
    (
        first,
        prompt as f64 / first,
        (times.len() - 1) as f64 / (last - first),
    )
}

#[derive(Debug, PartialEq)]
struct Args {
    model: PathBuf,
    prompts: Option<PathBuf>,
    lengths: Vec<String>,
    decode: usize,
    reps: usize,
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut it = argv.iter();
    let model = it
        .next()
        .filter(|m| !m.starts_with("--"))
        .ok_or("no model directory")?;
    let mut a = Args {
        model: PathBuf::from(model),
        prompts: None,
        lengths: ["short", "1k", "4k", "8k", "16k", "max"]
            .map(String::from)
            .to_vec(),
        decode: 256,
        reps: 5,
    };
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag}: no value"))?;
        let number = |least: usize| {
            value
                .parse::<usize>()
                .ok()
                .filter(|&n| n >= least)
                .ok_or_else(|| format!("{flag} {value}: a whole number of at least {least}"))
        };
        match flag.as_str() {
            "--prompts" => a.prompts = Some(PathBuf::from(value)),
            "--lengths" => {
                let l: Vec<String> = value.split(',').map(|x| x.trim().to_owned()).collect();
                if l.iter().any(String::is_empty) {
                    return Err(format!("--lengths {value:?}: an empty name"));
                }
                a.lengths = l;
            }
            "--decode" => a.decode = number(2)?,
            "--reps" => a.reps = number(1)?,
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    Ok(a)
}

/// What a length's ids are, before the engine is loaded.
#[derive(Debug, PartialEq)]
enum Ids {
    Ready(Vec<u32>),
    /// `max`: drawn once the context is known.
    Max,
    /// Why there are none: the length's status is "error".
    Bad(String),
}

#[derive(Debug, PartialEq)]
struct Planned {
    name: String,
    ids: Ids,
    /// `<name>.ids.json`, `4k.ids.json[..1024]` or `synthetic`.
    source: String,
}

/// The ids of each length (`max` waits for the context the card holds).
fn plan(names: &[String], prompts: Option<&Path>, vocab: usize) -> Vec<Planned> {
    let has = |f: &str| prompts.is_some_and(|d| d.join(f).is_file());
    let read = |f: &str| read_ids(prompts.expect("a file of the kit"), f, vocab);
    let drawn = |n: usize| (Ids::Ready(synthetic(n, vocab)), "synthetic".to_owned());
    names
        .iter()
        .map(|name| {
            let file = format!("{name}.ids.json");
            let (ids, source) = if name == "max" {
                (Ids::Max, "synthetic".to_owned())
            } else if has(&file) {
                (read(&file).map_or_else(Ids::Bad, Ids::Ready), file)
            } else if name == "1k" && has("4k.ids.json") {
                match read("4k.ids.json") {
                    Ok(v) if v.len() >= 1024 => (
                        Ids::Ready(v[..1024].to_vec()),
                        "4k.ids.json[..1024]".to_owned(),
                    ),
                    Ok(_) => drawn(1024),
                    Err(e) => (Ids::Bad(e), "4k.ids.json[..1024]".to_owned()),
                }
            } else if let Some(n) = size_of(name) {
                drawn(n)
            } else {
                let why = match prompts {
                    Some(d) => format!("no {file} in {}", d.display()),
                    None => format!("no prompt kit (--prompts) for {name}"),
                };
                (Ids::Bad(why), file)
            };
            Planned {
                name: name.clone(),
                ids,
                source,
            }
        })
        .collect()
}

/// The ids a name says without a file: `short` 20, `<n>k` n * 1024, `<n>` n.
fn size_of(name: &str) -> Option<usize> {
    if name == "short" {
        return Some(20);
    }
    let n = match name.strip_suffix('k') {
        Some(k) => k.parse::<usize>().ok()?.checked_mul(1024)?,
        None => name.parse::<usize>().ok()?,
    };
    (n > 0).then_some(n)
}

/// `<dir>/<file>`: a JSON list of ids below `vocab`.
fn read_ids(dir: &Path, file: &str, vocab: usize) -> Result<Vec<u32>, String> {
    let path = dir.join(file);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let ids: Vec<u32> =
        serde_json::from_str(&text).map_err(|_| format!("{file}: not a JSON list of token ids"))?;
    match ids.iter().find(|&&i| i as usize >= vocab) {
        Some(i) => Err(format!(
            "{file}: id {i} is not below the vocabulary of {vocab}"
        )),
        None => Ok(ids),
    }
}

/// The contexts to try, largest first: `top`, then fractions of `mpe` below it down to
/// `least` (the shortest length's need), multiples of 256 but `least`.
fn contexts(top: usize, mpe: usize, least: usize) -> Vec<usize> {
    const FRACTIONS: [(usize, usize); 7] =
        [(3, 4), (1, 2), (3, 8), (1, 4), (3, 16), (1, 8), (1, 16)];
    let mut out = vec![top];
    out.extend(
        FRACTIONS
            .iter()
            .map(|&(num, den)| mpe * num / den / 256 * 256)
            .filter(|&c| c > least && c < top),
    );
    if least < top {
        out.push(least);
    }
    out
}

/// The arena's size, from `Engine::describe` ("..., arena N bytes, ...").
fn arena_bytes(describe: &str) -> Option<u64> {
    describe.split(", ").find_map(|p| {
        p.strip_prefix("arena ")?
            .strip_suffix(" bytes")?
            .parse()
            .ok()
    })
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("core-bench: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    match run(&args) {
        // A leading newline: a line printed before this one need not have ended.
        Ok(v) => println!("\nx0ne-bench-result: {v}"),
        Err(e) => {
            eprintln!("core-bench: {e}");
            std::process::exit(1);
        }
    }
}

fn run(a: &Args) -> Result<Value, Box<dyn std::error::Error>> {
    let raw: Value = serde_json::from_str(&std::fs::read_to_string(a.model.join("config.json"))?)?;
    let field = |k: &str| -> Result<usize, String> {
        raw[k]
            .as_u64()
            .map(|n| n as usize)
            .ok_or_else(|| format!("config.json: no {k}"))
    };
    let (vocab, mpe) = (field("vocab_size")?, field("max_position_embeddings")?);
    let name = a
        .model
        .file_name()
        .map(|n| n.to_string_lossy().into_owned());
    let planned = plan(&a.lengths, a.prompts.as_deref(), vocab);
    // A length needs its ids and every decoded token but the last (greedy_timed's bound).
    let need = |n: usize| n + a.decode - 1;
    let needs: Vec<usize> = planned
        .iter()
        .filter_map(|p| match &p.ids {
            Ids::Ready(v) if need(v.len()) <= mpe => Some(need(v.len())),
            _ => None,
        })
        .collect();
    let wants_max = planned.iter().any(|p| p.ids == Ids::Max);
    let least = needs
        .iter()
        .copied()
        .min()
        .unwrap_or_else(|| need(a.decode).min(mpe));
    let top = if wants_max {
        mpe
    } else {
        needs.iter().copied().max().unwrap_or(least)
    };

    // The CUDA context, held for the readings of the device's memory (the whole device).
    let gpu = x0ne_core_runtime::gpu::Gpu::new()?;
    let (free_before, total) = gpu.mem_info()?;
    let mut tried = Vec::new();
    let mut loaded = None;
    for ctx in contexts(top, mpe, least) {
        eprintln!(
            "core-bench: loading {} with a context of {ctx}",
            a.model.display()
        );
        let t = Instant::now();
        match Engine::load(&a.model, EngineOptions { max_ctx: ctx }) {
            Ok(e) => {
                loaded = Some((e, ctx, t.elapsed().as_secs_f64()));
                break;
            }
            Err(e) if e.to_string().contains("OUT_OF_MEMORY") => {
                eprintln!("core-bench: {e}");
                tried.push(json!({"max_ctx": ctx, "error": e.to_string()}));
            }
            Err(e) => return Err(e.into()),
        }
    }
    let refused_all = |why: String| -> Vec<Value> {
        planned
            .iter()
            .map(
                |p| json!({"name": p.name, "source": p.source, "status": "refused", "reason": why}),
            )
            .collect()
    };
    let Some((first, ctx, cold_s)) = loaded else {
        let why = format!("no context from {top} down to {least} fits on the card (out of memory)");
        return Ok(json!({
            "schema": 1, "kind": "core", "model": name, "decode_tokens": a.decode, "reps": a.reps,
            "refused": why, "load": {"tried": tried}, "lengths": refused_all(why.clone()),
        }));
    };
    // Again, the files now in the page cache: the load without the disk.
    drop(first);
    let t = Instant::now();
    let mut e = Engine::load(&a.model, EngineOptions { max_ctx: ctx })?;
    let reload_s = t.elapsed().as_secs_f64();
    let free_after_load = gpu.mem_info()?.0;
    let mut free_min = free_after_load;

    let mut lengths = Vec::new();
    for p in planned {
        let ids = match p.ids {
            Ids::Ready(v) => v,
            Ids::Max if ctx > a.decode => synthetic(ctx - a.decode, vocab),
            Ids::Max => Vec::new(),
            Ids::Bad(why) => {
                lengths.push(
                    json!({"name": p.name, "source": p.source, "status": "error", "reason": why}),
                );
                continue;
            }
        };
        let tokens = ids.len();
        if tokens == 0 || need(tokens) > ctx {
            let why = if need(tokens) > mpe {
                format!(
                    "{tokens} + {} tokens exceed the model's {mpe} positions",
                    a.decode
                )
            } else {
                format!(
                    "{tokens} + {} tokens exceed the context of {ctx} the card held",
                    a.decode
                )
            };
            lengths.push(json!({"name": p.name, "tokens": tokens, "source": p.source, "status": "refused", "reason": why}));
            continue;
        }
        eprintln!(
            "core-bench: {}: {tokens} ids, {} tokens, 1 + {} passes",
            p.name, a.decode, a.reps
        );
        let mut first: Option<(Vec<u32>, (f64, f64, f64))> = None;
        let (mut same, mut failed) = (true, None);
        let (mut ttft, mut prefill, mut decode) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..=a.reps {
            match e.greedy_timed(&ids, a.decode) {
                Ok((out, times)) => {
                    free_min = free_min.min(gpu.mem_info()?.0);
                    let r = rates(&times, tokens);
                    match &first {
                        None => first = Some((out, r)),
                        Some((f, _)) => {
                            same &= out == *f;
                            ttft.push(r.0);
                            prefill.push(r.1);
                            decode.push(r.2);
                        }
                    }
                }
                Err(err) => {
                    failed = Some(err.to_string());
                    break;
                }
            }
        }
        match (failed, first) {
            // The card's memory, not the engine: written down like a context it refuses.
            (Some(why), _) if why.contains("OUT_OF_MEMORY") => lengths.push(
                json!({"name": p.name, "tokens": tokens, "source": p.source, "status": "refused", "reason": why}),
            ),
            (Some(why), _) => lengths.push(
                json!({"name": p.name, "tokens": tokens, "source": p.source, "status": "error", "reason": why}),
            ),
            (None, None) => unreachable!("a pass that neither ran nor failed"),
            (None, Some((_, (t, pr, d)))) => lengths.push(json!({
                "name": p.name, "tokens": tokens, "source": p.source, "status": "ok", "same_tokens": same,
                "first": {"ttft_s": t, "prefill_tok_s": pr, "decode_tok_s": d},
                "ttft_s": stats(&ttft), "prefill_tok_s": stats(&prefill), "decode_tok_s": stats(&decode),
            })),
        }
    }

    let describe = e.describe();
    let (kernel_set, nvcc) = x0ne_core_runtime::kernel_set();
    let c = e.config();
    let moe = e.moe().map(|m| {
        json!({"experts": m.num_experts, "per_token": m.num_experts_per_tok, "width": m.moe_intermediate_size})
    });
    Ok(json!({
        "schema": 1, "kind": "core", "model": name, "max_ctx": ctx, "decode_tokens": a.decode,
        "reps": a.reps, "engine": describe, "kernel_set": kernel_set, "nvcc": nvcc,
        "config": {
            "model_type": raw["model_type"], "layers": c.num_hidden_layers, "moe": moe,
            "hidden_size": c.hidden_size, "vocab_size": c.vocab_size, "max_position_embeddings": mpe,
        },
        "load": {"cold_s": cold_s, "reload_s": reload_s, "tried": tried},
        "memory": {
            "device_total_bytes": total, "free_before_load_bytes": free_before,
            "free_after_load_bytes": free_after_load, "free_min_bytes": free_min,
            "arena_bytes": arena_bytes(&describe),
            "taken_at_load_bytes": free_before.saturating_sub(free_after_load),
            "peak_used_bytes": total.saturating_sub(free_min),
        },
        "lengths": lengths,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    /// The ids x0ne-infer's bench_tests.rs pins too: a failure means the baseline's synthetic
    /// lengths are no longer core's, and the comparison of the two is void.
    #[test]
    fn the_synthetic_ids_are_the_fixed_seeds() {
        assert_eq!(
            synthetic(6, 151_936),
            [126_819, 135_692, 59_481, 30_850, 145_817, 72_711]
        );
        assert_eq!(synthetic(40_000, 1000).len(), 40_000);
        assert!(synthetic(40_000, 1000).iter().all(|&i| i < 1000));
    }

    #[test]
    fn the_statistics_of_the_repetitions() {
        let v = stats(&[3.0, 1.0, 2.0]);
        assert_eq!(
            v,
            json!({"median": 2.0, "min": 1.0, "max": 3.0, "mean": 2.0, "stddev": 1.0, "n": 3})
        );
        assert_eq!(stats(&[1.0, 2.0, 3.0, 4.0])["median"], json!(2.5));
        assert_eq!(stats(&[5.0])["stddev"], json!(0.0));
        assert_eq!(stats(&[]), Value::Null);
    }

    /// The first token is the prefill's; the rate of decoding counts the tokens after it.
    #[test]
    fn the_rates_of_one_pass() {
        let (ttft, prefill, decode) = rates(&[0.1, 0.2, 0.3, 0.5], 50);
        assert_eq!(ttft, 0.1);
        assert!((prefill - 500.0).abs() < 1e-9);
        assert!((decode - 7.5).abs() < 1e-9, "{decode}");
    }

    #[test]
    fn the_arguments() {
        let a = parse(&s(&[
            "/m",
            "--prompts",
            "/p",
            "--lengths",
            "short,1k,max",
            "--decode",
            "64",
            "--reps",
            "3",
        ]))
        .unwrap();
        assert_eq!(
            a,
            Args {
                model: "/m".into(),
                prompts: Some("/p".into()),
                lengths: s(&["short", "1k", "max"]),
                decode: 64,
                reps: 3
            }
        );
        let d = parse(&s(&["/m"])).unwrap();
        assert_eq!(
            (d.lengths, d.decode, d.reps, d.prompts),
            (s(&["short", "1k", "4k", "8k", "16k", "max"]), 256, 5, None)
        );
        for bad in [
            &["/m", "--decode", "1"][..],
            &["/m", "--reps", "0"],
            &["/m", "--lengths", ""],
            &["/m", "--lengths", "a,,b"],
            &["/m", "--what", "1"],
            &["/m", "--reps"],
            &[],
        ] {
            assert!(parse(&s(bad)).is_err(), "{bad:?}");
        }
    }

    /// The kit's files first; `1k` from `4k`; drawn ids for the sizes a name says; `max`
    /// later; anything else, or a file that is not a list of ids of this vocabulary, an error.
    #[test]
    fn the_ids_of_each_length() {
        let dir = std::env::temp_dir().join(format!("core-bench-plan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("short.ids.json"), "[1, 2, 3]").unwrap();
        let four: Vec<u32> = (0..2000).collect();
        std::fs::write(
            dir.join("4k.ids.json"),
            serde_json::to_string(&four).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("big.ids.json"), "[1, 5000]").unwrap();
        std::fs::write(dir.join("text.ids.json"), "\"hello\"").unwrap();
        let p = plan(
            &s(&[
                "short", "1k", "4k", "2k", "300", "max", "weird", "big", "text",
            ]),
            Some(&dir),
            4096,
        );
        let got: Vec<(&str, &str, Option<usize>)> = p
            .iter()
            .map(|x| {
                let n = match &x.ids {
                    Ids::Ready(v) => Some(v.len()),
                    _ => None,
                };
                (x.name.as_str(), x.source.as_str(), n)
            })
            .collect();
        assert_eq!(
            got,
            [
                ("short", "short.ids.json", Some(3)),
                ("1k", "4k.ids.json[..1024]", Some(1024)),
                ("4k", "4k.ids.json", Some(2000)),
                ("2k", "synthetic", Some(2048)),
                ("300", "synthetic", Some(300)),
                ("max", "synthetic", None),
                ("weird", "weird.ids.json", None),
                ("big", "big.ids.json", None),
                ("text", "text.ids.json", None),
            ]
        );
        assert_eq!(p[1].ids, Ids::Ready((0..1024).collect()));
        assert_eq!(p[3].ids, Ids::Ready(synthetic(2048, 4096)));
        assert_eq!(p[5].ids, Ids::Max);
        for (i, want) in [
            (6, "no weird.ids.json in"),
            (7, "5000"),
            (8, "not a JSON list"),
        ] {
            match &p[i].ids {
                Ids::Bad(why) => assert!(why.contains(want), "{why}"),
                other => panic!("{}: {other:?}", p[i].name),
            }
        }
        // Without a kit: short is 20 drawn ids, 1k 1024.
        let q = plan(&s(&["short", "1k"]), None, 4096);
        assert_eq!(q[0].ids, Ids::Ready(synthetic(20, 4096)));
        assert_eq!(q[1].ids, Ids::Ready(synthetic(1024, 4096)));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// From the largest down, never below what the shortest length needs.
    #[test]
    fn the_contexts_tried() {
        assert_eq!(
            contexts(40_960, 40_960, 275),
            [40_960, 30_720, 20_480, 15_360, 10_240, 7_680, 5_120, 2_560, 275]
        );
        assert_eq!(contexts(4_351, 40_960, 275), [4_351, 2_560, 275]);
        assert_eq!(contexts(275, 40_960, 275), [275]);
    }

    #[test]
    fn the_arena_of_a_description() {
        let d = "NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs, cubin sm_120), driver 13000, cuBLAS 130100, arena 2900000000 bytes, decode graph 412 nodes";
        assert_eq!(arena_bytes(d), Some(2_900_000_000));
        assert_eq!(arena_bytes("no arena here"), None);
    }
}
