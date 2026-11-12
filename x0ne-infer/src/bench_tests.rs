//! make bench's tiers that go through x0ne-infer (`scripts/bench.sh`), and their CPU tests.
//!
//! On a GPU, ignored here, run one at a time by `make bench` from the release test binary
//! built with `--features cuda`:
//!
//! - `gpu::engine`: the kit's chat prompts (`<name>.chat.json`) through the server of this
//!   crate's unit tests on x0ne Core, streamed and whole, at the ticks of `X0NE_BENCH_TICKS`.
//! - `gpu::load`: levels of callers against that server (`bench_load_tests.rs`).
//! - `gpu::soak`: callers for a long while, throughput per window, host and device memory.
//! - `baseline_llama`: llama.cpp's llama-server on the same ids as core-bench (x0ne-core's
//!   `oracle/examples/core-bench.rs`), as a comparison, never a reference.
//!
//! The server is the unit tests' one: a test proof, no TDX, no relay, its client in the same
//! process over loopback. Each prints one line `x0ne-bench-result: <JSON>` (schema 1, kind: the
//! tier), or one with `refused` when the card cannot hold the model. Nothing here judges the
//! numbers: `scripts/bench.sh` does, against `ci/bench-baseline.json`.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::backend::{
    Cancelled, Completion, GenerationParams, Isolation, LoadedModel, NoSink, TokenSink,
};
use crate::bench_load_tests::{
    body, chat, client, level_summary, run_clients, soak, soak_summary, stats, Seen,
};
use crate::binding::SessionIdentity;
use crate::padding::Tick;
use crate::prompt::Prompt;
use crate::server::tests::{challenge_fixture, live_lease, proof};
use crate::server::{
    serve_with, Admission, ChatMessage, RawEvidence, ServerState, Serving,
    DEFAULT_MAX_COMPLETION_TOKENS,
};

/// The ceiling of the bench's server: the default of a served one.
const CEILING: usize = DEFAULT_MAX_COMPLETION_TOKENS;

/// What `scripts/bench.sh` passes, in `X0NE_BENCH_*` variables; lists comma-separated.
#[derive(Debug, Clone, PartialEq)]
struct Settings {
    model: Option<PathBuf>,
    prompts: Option<PathBuf>,
    reps: usize,
    decode: usize,
    engine_prompts: Vec<String>,
    ticks: Vec<String>,
    clients: Vec<usize>,
    level_secs: usize,
    load_prompt: String,
    load_tokens: usize,
    max_queue: usize,
    soak_secs: usize,
    soak_clients: usize,
    soak_window_secs: usize,
    llama_server: Option<PathBuf>,
    llama_lib: Option<PathBuf>,
    gguf: Option<PathBuf>,
    lengths: Vec<String>,
}

/// The settings `get` gives, defaults where it gives none.
fn settings(get: &dyn Fn(&str) -> Option<String>) -> Result<Settings, String> {
    let number = |k: &str, default: usize, least: usize| match get(k) {
        None => Ok(default),
        Some(v) => v
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|&n| n >= least)
            .ok_or_else(|| format!("{k}={v:?}: a whole number of at least {least}")),
    };
    let list = |k: &str, default: &str| {
        let v = get(k).unwrap_or_else(|| default.to_owned());
        let items: Vec<String> = v.split(',').map(|x| x.trim().to_owned()).collect();
        if items.iter().any(String::is_empty) {
            return Err(format!("{k}={v:?}: an empty name"));
        }
        Ok(items)
    };
    let path = |k: &str| get(k).map(PathBuf::from);
    let ticks = list("X0NE_BENCH_TICKS", "production,short")?;
    if let Some(t) = ticks
        .iter()
        .find(|t| !matches!(t.as_str(), "production" | "short"))
    {
        return Err(format!(
            "X0NE_BENCH_TICKS: {t:?} is neither production nor short"
        ));
    }
    let clients = list("X0NE_BENCH_CLIENTS", "1,2,4,8,16,32")?
        .iter()
        .map(|c| {
            c.parse::<usize>()
                .ok()
                .filter(|&n| n >= 1)
                .ok_or_else(|| format!("X0NE_BENCH_CLIENTS: {c:?} is not a number of clients"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let s = Settings {
        model: path("X0NE_BENCH_MODEL"),
        prompts: path("X0NE_BENCH_PROMPTS"),
        reps: number("X0NE_BENCH_REPS", 5, 1)?,
        decode: number("X0NE_BENCH_DECODE", 256, 2)?,
        engine_prompts: list("X0NE_BENCH_ENGINE_PROMPTS", "short,4k,8k,16k")?,
        ticks,
        clients,
        level_secs: number("X0NE_BENCH_LEVEL_SECS", 30, 1)?,
        load_prompt: get("X0NE_BENCH_LOAD_PROMPT").unwrap_or_else(|| "short".to_owned()),
        load_tokens: number("X0NE_BENCH_LOAD_TOKENS", 256, 1)?,
        max_queue: number("X0NE_BENCH_MAX_QUEUE", 8, 0)?,
        soak_secs: number("X0NE_BENCH_SOAK_SECS", 600, 1)?,
        soak_clients: number("X0NE_BENCH_SOAK_CLIENTS", 4, 1)?,
        soak_window_secs: number("X0NE_BENCH_SOAK_WINDOW_SECS", 60, 1)?,
        llama_server: path("X0NE_BENCH_LLAMA_SERVER"),
        llama_lib: path("X0NE_BENCH_LLAMA_LIB"),
        gguf: path("X0NE_BENCH_GGUF"),
        lengths: list("X0NE_BENCH_LENGTHS", "short,1k,4k,8k,16k,max")?,
    };
    if s.soak_window_secs > s.soak_secs {
        return Err(format!(
            "X0NE_BENCH_SOAK_WINDOW_SECS={} is longer than the soak ({} s)",
            s.soak_window_secs, s.soak_secs
        ));
    }
    Ok(s)
}

fn from_env() -> Settings {
    let get = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    settings(&get).unwrap_or_else(|e| panic!("{e}"))
}

/// The line `scripts/bench.sh` reads: one, after a newline (libtest's `--nocapture` may have
/// left one open).
fn result_line(v: &Value) -> String {
    format!("\nx0ne-bench-result: {v}")
}

fn print_result(v: &Value) {
    println!("{}", result_line(v));
}

// ---- the ids of each length, as core-bench draws them.

/// splitmix64, as core-bench and core-perf.
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

/// `n` ids below `vocab` from core-bench's fixed seed.
fn synthetic(n: usize, vocab: usize) -> Vec<u32> {
    let mut rng = Rng(0x0C0E_5EED);
    (0..n).map(|_| rng.below(vocab) as u32).collect()
}

/// What a length's ids are, before the context is known.
#[derive(Debug, PartialEq)]
enum Ids {
    Ready(Vec<u32>),
    /// `max`: drawn once the context is known.
    Max,
    /// Why there are none.
    Bad(String),
}

#[derive(Debug, PartialEq)]
struct Planned {
    name: String,
    ids: Ids,
    /// `<name>.ids.json`, `4k.ids.json[..1024]` or `synthetic`.
    source: String,
}

/// core-bench's plan: the kit's `<name>.ids.json`; `1k` the first 1024 ids of `4k.ids.json`;
/// `short` (20), `<n>k` and `<n>` drawn ids without a file; `max` later; anything else, or a
/// file that is not a list of ids of this vocabulary, why not.
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
                    None => format!("no prompt kit (X0NE_BENCH_PROMPTS) for {name}"),
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

// ---- the server.

/// One loaded model behind several servers, one per tick.
struct Shared(Arc<dyn LoadedModel>);

impl LoadedModel for Shared {
    fn generate(&self, prompt: &Prompt, max_tokens: usize) -> anyhow::Result<(String, (u32, u32))> {
        self.0.generate(prompt, max_tokens)
    }
    fn generate_with(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
    ) -> anyhow::Result<Completion> {
        self.0.generate_with(prompt, params)
    }
    fn generate_whole(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        self.0.generate_whole(prompt, params, sink)
    }
    fn generate_streaming(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        self.0.generate_streaming(prompt, params, sink)
    }
    fn generate_turn(
        &self,
        prompt: &Prompt,
        params: &GenerationParams,
        conversation: Option<&crate::conversation::Turn>,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        self.0.generate_turn(prompt, params, conversation, sink)
    }
    fn chat_prompt(&self, messages: &[ChatMessage]) -> Prompt {
        self.0.chat_prompt(messages)
    }
    fn model_digest_hex(&self) -> String {
        self.0.model_digest_hex()
    }
    fn compute_device(&self) -> String {
        self.0.compute_device()
    }
    fn isolation(&self) -> Isolation {
        self.0.isolation()
    }
    fn serving_limits(&self) -> Option<(usize, usize)> {
        self.0.serving_limits()
    }
    fn prefill_chunk_for(&self, prompt_tokens: usize) -> Option<usize> {
        self.0.prefill_chunk_for(prompt_tokens)
    }
}

/// The unit tests' server on `model`: a test proof, the ceiling of a served one.
async fn spawn(model: Arc<dyn LoadedModel>, tick: Tick, admission: Admission) -> SocketAddr {
    let p = proof();
    let listener = crate::gate::bind(&p, "127.0.0.1:0")
        .await
        .expect("a loopback port");
    let addr = listener.local_addr().expect("its address");
    tokio::spawn(async move {
        let _ = serve_with(
            listener,
            ServerState {
                raw_evidence: RawEvidence::default(),
                proof: p,
                identity: SessionIdentity::generate([7u8; 32], [9u8; 32]),
                model: Box::new(Shared(model)),
                max_completion_tokens: CEILING,
                lease: live_lease().ticking(tick),
                salt_source: Box::new(|| [3u8; 32]),
                challenge_source: challenge_fixture(),
            },
            Serving {
                admission,
                tick,
                ..Serving::default()
            },
        )
        .await;
    });
    addr
}

fn url(addr: SocketAddr) -> String {
    format!("http://{addr}/v1/chat/completions")
}

fn tick_of(name: &str) -> Tick {
    match name {
        "short" => Tick::SHORT,
        _ => Tick::PRODUCTION,
    }
}

fn tick_ms(tick: Tick) -> f64 {
    tick.period().as_micros() as f64 / 1000.0
}

/// The kit's `<name>.chat.json`: its messages.
fn chat_messages(prompts: &Path, name: &str) -> Option<Value> {
    let v: Value =
        serde_json::from_slice(&std::fs::read(prompts.join(format!("{name}.chat.json"))).ok()?)
            .ok()?;
    v.get("messages").filter(|m| m.is_array()).cloned()
}

/// The count of the kit's `<name>.ids.json`: the chat template applied, as the server does.
fn ids_count(prompts: &Path, name: &str) -> Option<usize> {
    let v: Value =
        serde_json::from_slice(&std::fs::read(prompts.join(format!("{name}.ids.json"))).ok()?)
            .ok()?;
    v.as_array().map(Vec::len)
}

/// Why a pass is not an answer, or nothing.
fn failure(s: &Seen) -> Option<String> {
    s.error
        .clone()
        .or_else(|| (s.status != 200).then(|| format!("HTTP {}", s.status)))
}

fn finishes(passes: &[Seen]) -> Vec<String> {
    let mut f: Vec<String> = passes.iter().filter_map(|s| s.finish.clone()).collect();
    f.sort();
    f.dedup();
    f
}

fn token_range(passes: &[Seen]) -> Value {
    let n: Vec<u64> = passes.iter().filter_map(|s| s.completion_tokens).collect();
    json!({"min": n.iter().min(), "max": n.iter().max()})
}

/// A stream's passes: the first apart, the others' times; its decode rate from the first
/// content to the last, which a tick quantizes (a frame carries what came during it).
fn stream_section(passes: &[Seen]) -> Value {
    let (first, rest) = passes.split_first().expect("one pass at least");
    let good: Vec<&Seen> = rest.iter().filter(|s| s.ok()).collect();
    let decode: Vec<f64> = good
        .iter()
        .filter_map(
            |s| match (s.ttft_s, s.last_content_s, s.completion_tokens) {
                (Some(a), Some(b), Some(n)) if b > a && n > 1 => Some((n - 1) as f64 / (b - a)),
                _ => None,
            },
        )
        .collect();
    json!({
        "first": {"ttft_s": first.ttft_s, "end_s": first.end_s},
        "ttft_s": stats(&good.iter().filter_map(|s| s.ttft_s).collect::<Vec<_>>()),
        "last_content_s": stats(&good.iter().filter_map(|s| s.last_content_s).collect::<Vec<_>>()),
        "end_s": stats(&good.iter().map(|s| s.end_s).collect::<Vec<_>>()),
        "decode_tok_s": stats(&decode),
        "completion_tokens": token_range(passes),
        "finish": finishes(passes),
        "errors": passes.iter().filter_map(failure).collect::<Vec<_>>(),
    })
}

fn whole_section(passes: &[Seen]) -> Value {
    let (first, rest) = passes.split_first().expect("one pass at least");
    let e2e: Vec<f64> = rest.iter().filter(|s| s.ok()).map(|s| s.end_s).collect();
    json!({
        "first": {"e2e_s": first.end_s},
        "e2e_s": stats(&e2e),
        "completion_tokens": token_range(passes),
        "finish": finishes(passes),
        "errors": passes.iter().filter_map(failure).collect::<Vec<_>>(),
    })
}

/// The prompt `name` of the kit at the server `addr`: `1 + reps` streams, then as many whole
/// answers, of `decode` tokens at most; skipped, with why, when the kit has no chat of that
/// name or its ids and the tokens exceed `context`.
async fn engine_case(
    addr: SocketAddr,
    prompts: &Path,
    name: &str,
    tick: &str,
    s: &Settings,
    context: Option<usize>,
) -> Value {
    let ids_tokens = ids_count(prompts, name);
    let head = json!({"prompt": name, "tick": tick, "tick_ms": tick_ms(tick_of(tick)), "ids_tokens": ids_tokens});
    let skipped = |why: String| {
        let mut v = head.clone();
        v["skipped"] = json!(why);
        v
    };
    let Some(messages) = chat_messages(prompts, name) else {
        return skipped(format!("no {name}.chat.json in the kit"));
    };
    if let (Some(n), Some(ctx)) = (ids_tokens, context) {
        if n + s.decode > ctx {
            return skipped(format!(
                "{n} + {} tokens exceed the context of {ctx}",
                s.decode
            ));
        }
    }
    let (pool, url) = (client(), url(addr));
    let mut out = head.clone();
    let mut prompt_tokens = None;
    for (key, stream) in [("stream", true), ("whole", false)] {
        let b = body(&messages, s.decode, stream);
        let mut passes = Vec::new();
        for _ in 0..=s.reps {
            passes.push(chat(&pool, &url, &b, Instant::now(), 0).await);
        }
        prompt_tokens = prompt_tokens.or(passes[0].prompt_tokens);
        out[key] = if stream {
            stream_section(&passes)
        } else {
            whole_section(&passes)
        };
    }
    out["prompt_tokens"] = json!(prompt_tokens);
    out
}

// ---- llama.cpp's llama-server.

/// (version, file type) from llama-server's log: `build: N (sha) with ...` (`build = ` in
/// older ones), `file type = BF16` (`model ftype = ` in older ones).
fn llama_facts(log: &str) -> (Option<String>, Option<String>) {
    let version = log
        .lines()
        .find_map(|l| {
            ["build: ", "build = "]
                .iter()
                .find_map(|k| l.find(k).map(|i| &l[i + k.len()..]))
        })
        .and_then(|v| v.split(" with ").next())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_owned);
    let file_type = log
        .lines()
        .find(|l| l.contains("file type") || l.contains("model ftype"))
        .and_then(|l| l.split_once('='))
        .map(|(_, v)| v.trim())
        .filter(|v| !v.is_empty())
        .map(str::to_owned);
    (version, file_type)
}

/// (prefill seconds, prefill rate, decode rate) as llama-server's `timings` give them; refused
/// when it read other than the `tokens` ids sent, or decoded other than `decode` tokens.
fn llama_rates(v: &Value, tokens: usize, decode: usize) -> Result<(f64, f64, f64), String> {
    let t = &v["timings"];
    let number = |k: &str| {
        t[k].as_f64()
            .filter(|x| *x > 0.0)
            .ok_or_else(|| format!("llama-server's timings: no {k} above 0"))
    };
    let (prompt_n, prompt_ms) = (number("prompt_n")?, number("prompt_ms")?);
    let (predicted_n, predicted_ms) = (number("predicted_n")?, number("predicted_ms")?);
    if prompt_n != tokens as f64 {
        return Err(format!(
            "llama-server read {prompt_n} of the {tokens} ids sent"
        ));
    }
    if predicted_n != decode as f64 {
        return Err(format!(
            "llama-server decoded {predicted_n} tokens, not {decode}"
        ));
    }
    Ok((
        prompt_ms / 1000.0,
        prompt_n * 1000.0 / prompt_ms,
        predicted_n * 1000.0 / predicted_ms,
    ))
}

/// A llama-server, killed when dropped.
struct Llama {
    child: std::process::Child,
    port: u16,
    log: PathBuf,
}

impl Drop for Llama {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Llama {
    /// `server -m gguf -c ctx`, every layer on the GPU, one slot; ready once `/health` says so.
    async fn start(
        server: &Path,
        lib: Option<&Path>,
        gguf: &Path,
        ctx: usize,
    ) -> Result<Llama, String> {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .and_then(|l| l.local_addr())
            .map_err(|e| format!("a loopback port: {e}"))?
            .port();
        let log = std::env::temp_dir().join(format!("x0ne-bench-llama-{}.log", std::process::id()));
        let file = std::fs::File::create(&log).map_err(|e| format!("{}: {e}", log.display()))?;
        let mut cmd = std::process::Command::new(server);
        cmd.arg("-m")
            .arg(gguf)
            .args(["-c", &ctx.to_string(), "-ngl", "999", "--host", "127.0.0.1"])
            .args(["--port", &port.to_string(), "-np", "1"])
            .stdin(std::process::Stdio::null())
            .stdout(file.try_clone().map_err(|e| e.to_string())?)
            .stderr(file);
        if let Some(lib) = lib {
            let old = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            let path = if old.is_empty() {
                lib.display().to_string()
            } else {
                format!("{}:{old}", lib.display())
            };
            cmd.env("LD_LIBRARY_PATH", path);
        }
        let child = cmd
            .spawn()
            .map_err(|e| format!("{}: {e}", server.display()))?;
        let mut l = Llama { child, port, log };
        let pool = client();
        let t = Instant::now();
        loop {
            if let Ok(Some(status)) = l.child.try_wait() {
                return Err(format!("llama-server left ({status}): {}", l.tail()));
            }
            if let Ok(r) = pool.get(l.at("/health")).send().await {
                if r.status() == 200 {
                    return Ok(l);
                }
            }
            if t.elapsed() > Duration::from_mins(15) {
                return Err(format!("llama-server not ready after 900 s: {}", l.tail()));
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    fn at(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }

    fn read_log(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap_or_default()
    }

    /// The end of its log, for a reason.
    fn tail(&self) -> String {
        let log = self.read_log();
        let lines: Vec<&str> = log.lines().collect();
        lines[lines.len().saturating_sub(8)..].join(" | ")
    }

    /// One greedy completion of `ids`, `decode` tokens whatever the end of sentence says.
    async fn complete(
        &self,
        pool: &reqwest::Client,
        ids: &[u32],
        decode: usize,
    ) -> Result<(f64, f64, f64, f64), String> {
        let b = json!({"prompt": ids, "n_predict": decode, "temperature": 0, "top_k": 1,
                       "ignore_eos": true, "cache_prompt": false, "stream": false});
        let t = Instant::now();
        let r = pool
            .post(self.at("/completion"))
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(b.to_string())
            .send()
            .await
            .map_err(|e| format!("no answer: {e}"))?;
        let status = r.status().as_u16();
        let bytes = r.bytes().await.map_err(|e| format!("body cut: {e}"))?;
        let e2e_s = t.elapsed().as_secs_f64();
        if status != 200 {
            let head: String = String::from_utf8_lossy(&bytes).chars().take(160).collect();
            return Err(format!("HTTP {status}: {head}"));
        }
        let v: Value = serde_json::from_slice(&bytes).map_err(|e| format!("not JSON: {e}"))?;
        let (prefill_s, prefill, rate) = llama_rates(&v, ids.len(), decode)?;
        Ok((prefill_s, prefill, rate, e2e_s))
    }
}

/// The settings' model directory and its name.
fn model_of(s: &Settings) -> (PathBuf, Option<String>) {
    let dir = s
        .model
        .clone()
        .expect("X0NE_BENCH_MODEL: the model's directory");
    let name = dir.file_name().map(|n| n.to_string_lossy().into_owned());
    (dir, name)
}

/// llama.cpp on the ids core-bench measures: `X0NE_BENCH_LENGTHS` of the kit or drawn from
/// its seed, `max` as long as the model's positions allow, `1 + reps` passes each. Its
/// rates are llama-server's own `timings`; the response time is this client's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "llama-server on a GPU: X0NE_BENCH_LLAMA_SERVER, X0NE_BENCH_GGUF (make bench)"]
async fn baseline_llama() {
    let s = from_env();
    let server = s
        .llama_server
        .clone()
        .expect("X0NE_BENCH_LLAMA_SERVER: a llama-server built for this GPU");
    let gguf = s
        .gguf
        .clone()
        .expect("X0NE_BENCH_GGUF: a GGUF of the model");
    let (dir, name) = model_of(&s);
    let config: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .expect("config.json is JSON");
    let field = |k: &str| {
        config[k]
            .as_u64()
            .unwrap_or_else(|| panic!("config.json: no {k}")) as usize
    };
    let (vocab, mpe) = (field("vocab_size"), field("max_position_embeddings"));
    let planned = plan(&s.lengths, s.prompts.as_deref(), vocab);
    let need = |n: usize| n + s.decode;
    let ctx = if planned.iter().any(|p| p.ids == Ids::Max) {
        mpe
    } else {
        planned
            .iter()
            .filter_map(|p| match &p.ids {
                Ids::Ready(v) if need(v.len()) <= mpe => Some(need(v.len())),
                _ => None,
            })
            .max()
            .unwrap_or_else(|| need(20).min(mpe))
    };
    let llama = match Llama::start(&server, s.llama_lib.as_deref(), &gguf, ctx).await {
        Ok(l) => l,
        Err(why) if why.to_lowercase().contains("out of memory") => {
            print_result(
                &json!({"schema": 1, "kind": "baseline", "engine": "llama.cpp",
                                 "model": name, "refused": why}),
            );
            return;
        }
        Err(why) => panic!("{why}"),
    };
    let (version, file_type) = llama_facts(&llama.read_log());
    let pool = client();
    let mut lengths = Vec::new();
    for p in planned {
        let ids = match p.ids {
            Ids::Ready(v) => v,
            Ids::Max => synthetic(ctx - s.decode, vocab),
            Ids::Bad(why) => {
                lengths.push(
                    json!({"name": p.name, "source": p.source, "status": "error", "reason": why}),
                );
                continue;
            }
        };
        let tokens = ids.len();
        if need(tokens) > ctx {
            lengths.push(json!({"name": p.name, "tokens": tokens, "source": p.source, "status": "refused",
                                "reason": format!("{tokens} + {} tokens exceed the context of {ctx}", s.decode)}));
            continue;
        }
        eprintln!("baseline: {}: {tokens} ids, 1 + {} passes", p.name, s.reps);
        let mut passes = Vec::new();
        let mut failed = None;
        for _ in 0..=s.reps {
            match llama.complete(&pool, &ids, s.decode).await {
                Ok(r) => passes.push(r),
                Err(why) => {
                    failed = Some(why);
                    break;
                }
            }
        }
        if let Some(why) = failed {
            lengths.push(json!({"name": p.name, "tokens": tokens, "source": p.source, "status": "error", "reason": why}));
            continue;
        }
        let (first, rest) = passes.split_first().expect("1 + reps passes");
        let col =
            |f: fn(&(f64, f64, f64, f64)) -> f64| stats(&rest.iter().map(f).collect::<Vec<_>>());
        lengths.push(json!({
            "name": p.name, "tokens": tokens, "source": p.source, "status": "ok",
            "first": {"prefill_s": first.0, "prefill_tok_s": first.1, "decode_tok_s": first.2, "e2e_s": first.3},
            "prefill_s": col(|r| r.0), "prefill_tok_s": col(|r| r.1), "decode_tok_s": col(|r| r.2),
            "e2e_s": col(|r| r.3),
        }));
    }
    print_result(&json!({
        "schema": 1, "kind": "baseline", "engine": "llama.cpp", "model": name,
        "server_version": version, "file_type": file_type, "ctx": ctx,
        "decode_tokens": s.decode, "reps": s.reps, "lengths": lengths,
    }));
}

/// On a GPU, through x0ne Core: `make bench` runs each alone (`--exact`).
mod gpu {
    use super::*;

    /// A model loaded through the registry as a server loads it, at its whole context.
    struct Loaded {
        model: Arc<dyn LoadedModel>,
        load_s: f64,
        context: Option<usize>,
        name: Option<String>,
    }

    /// The model of the settings; or, the card refusing it (out of memory), `None` and the
    /// refusal printed as the result of `kind`.
    fn loaded(s: &Settings, kind: &str) -> Option<Loaded> {
        let (dir, name) = model_of(s);
        eprintln!("bench: {kind}: loading {}", dir.display());
        let read = |f: &str| {
            crate::fsread::read_model_file(&dir, f)
                .unwrap_or_else(|e| panic!("{}: {e}", dir.join(f).display()))
        };
        let registry = crate::backends::Registry::default();
        let entry = registry
            .get("core")
            .expect("no core backend in this build: --features cuda");
        let t = Instant::now();
        let backend = entry
            .open(
                crate::model_spec::ModelArtifacts {
                    config_json: read("config.json"),
                    tokenizer_json: read("tokenizer.json"),
                },
                "cuda",
            )
            .unwrap_or_else(|e| panic!("core: {e:#}"));
        let weights =
            crate::shards::map_dir(&dir).unwrap_or_else(|e| panic!("{}: {e:#}", dir.display()));
        let key = crate::keys::test_released_key(vec![7u8; 32]);
        match backend.load_files(&weights.as_files(), &key) {
            Ok(m) => {
                let model: Arc<dyn LoadedModel> = Arc::from(m);
                let context = model.serving_limits().map(|(c, _)| c);
                Some(Loaded {
                    model,
                    load_s: t.elapsed().as_secs_f64(),
                    context,
                    name,
                })
            }
            Err(e) if format!("{e:#}").contains("OUT_OF_MEMORY") => {
                print_result(
                    &json!({"schema": 1, "kind": kind, "model": name, "refused": format!("{e:#}")}),
                );
                None
            }
            Err(e) => panic!("core: loading {}: {e:#}", dir.display()),
        }
    }

    fn prompts(s: &Settings) -> PathBuf {
        s.prompts
            .clone()
            .expect("X0NE_BENCH_PROMPTS: the prompt kit")
    }

    /// The device memory in use, MiB, read through a CUDA context of its own: the whole
    /// device, the engine's and this reading's.
    #[cfg(feature = "cuda")]
    fn vram_reader() -> Arc<dyn Fn() -> Option<u64> + Send + Sync> {
        match x0ne_core_runtime::gpu::Gpu::new() {
            Ok(g) => {
                let g = std::sync::Mutex::new(g);
                Arc::new(move || {
                    let (free, total) = g.lock().ok()?.mem_info().ok()?;
                    Some(total.saturating_sub(free) >> 20)
                })
            }
            Err(e) => {
                eprintln!("bench: soak: no reading of the device memory: {e}");
                Arc::new(|| None)
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn vram_reader() -> Arc<dyn Fn() -> Option<u64> + Send + Sync> {
        Arc::new(|| None)
    }

    /// One answer before the measures: a fresh server's first request is not one of them.
    async fn warm(url: &str, b: &Value) {
        let w = chat(&client(), url, b, Instant::now(), 0).await;
        assert!(w.ok(), "the warm-up request: {w:?}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "GPU: X0NE_BENCH_MODEL (make bench)"]
    async fn engine() {
        let s = from_env();
        let Some(Loaded {
            model,
            load_s,
            context,
            name,
        }) = loaded(&s, "engine")
        else {
            return;
        };
        let kit = prompts(&s);
        let mut cases = Vec::new();
        for tick in &s.ticks {
            let addr = spawn(model.clone(), tick_of(tick), Admission::new(s.max_queue)).await;
            for p in &s.engine_prompts {
                eprintln!("bench: engine: {p} at the {tick} tick");
                cases.push(engine_case(addr, &kit, p, tick, &s, context).await);
            }
        }
        print_result(&json!({
            "schema": 1, "kind": "engine", "model": name, "decode_tokens": s.decode, "reps": s.reps,
            "context_tokens": context, "load_s": load_s, "device": model.compute_device(), "cases": cases,
        }));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "GPU: X0NE_BENCH_MODEL (make bench)"]
    async fn load() {
        let s = from_env();
        let Some(Loaded {
            model,
            load_s,
            context,
            name,
        }) = loaded(&s, "load")
        else {
            return;
        };
        let kit = prompts(&s);
        let messages = chat_messages(&kit, &s.load_prompt)
            .unwrap_or_else(|| panic!("no {}.chat.json in {}", s.load_prompt, kit.display()));
        let addr = spawn(model, Tick::PRODUCTION, Admission::new(s.max_queue)).await;
        let (u, b) = (url(addr), body(&messages, s.load_tokens, true));
        warm(&u, &b).await;
        let mut levels = Vec::new();
        for &c in &s.clients {
            eprintln!("bench: load: {c} clients for {} s", s.level_secs);
            let run_for = Duration::from_secs(s.level_secs as u64);
            let (seen, seconds) = run_clients(&u, &b, c, Instant::now(), run_for).await;
            levels.push(level_summary(&seen, c, seconds));
        }
        print_result(&json!({
            "schema": 1, "kind": "load", "model": name, "tick": "production",
            "tick_ms": tick_ms(Tick::PRODUCTION), "max_queue": s.max_queue, "ceiling": CEILING,
            "prompt": s.load_prompt, "max_tokens": s.load_tokens, "level_s": s.level_secs,
            "context_tokens": context, "load_s": load_s, "levels": levels,
        }));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "GPU: X0NE_BENCH_MODEL (make bench)"]
    async fn soak() {
        let s = from_env();
        let Some(Loaded {
            model,
            load_s,
            context,
            name,
        }) = loaded(&s, "soak")
        else {
            return;
        };
        let kit = prompts(&s);
        let messages = chat_messages(&kit, &s.load_prompt)
            .unwrap_or_else(|| panic!("no {}.chat.json in {}", s.load_prompt, kit.display()));
        let addr = spawn(model, Tick::PRODUCTION, Admission::new(s.max_queue)).await;
        let (u, b) = (url(addr), body(&messages, s.load_tokens, true));
        warm(&u, &b).await;
        eprintln!(
            "bench: soak: {} clients for {} s",
            s.soak_clients, s.soak_secs
        );
        let (seen, samples, ran_s) = super::soak(
            &u,
            &b,
            s.soak_clients,
            Duration::from_secs(s.soak_secs as u64),
            Duration::from_secs(s.soak_window_secs as u64),
            vram_reader(),
        )
        .await;
        let (windows, summary) = soak_summary(
            &seen,
            &samples,
            s.soak_secs as f64,
            s.soak_window_secs as f64,
        );
        print_result(&json!({
            "schema": 1, "kind": "soak", "model": name, "tick": "production",
            "clients": s.soak_clients, "seconds": s.soak_secs, "window_s": s.soak_window_secs,
            "ran_s": ran_s, "prompt": s.load_prompt, "max_tokens": s.load_tokens,
            "max_queue": s.max_queue, "context_tokens": context, "load_s": load_s,
            "windows": windows, "summary": summary,
        }));
    }
}

// ---- the CPU tests.

/// A model that takes `step` a token and hands each to its sink: 7 prompt tokens, its whole
/// budget, `t` per token.
struct Paced {
    step: Duration,
}

impl LoadedModel for Paced {
    fn generate(&self, prompt: &Prompt, max_tokens: usize) -> anyhow::Result<(String, (u32, u32))> {
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
        _prompt: &Prompt,
        params: &GenerationParams,
        sink: &mut dyn TokenSink,
    ) -> anyhow::Result<Completion> {
        if sink.started().is_break() {
            return Err(Cancelled.into());
        }
        for _ in 0..params.max_tokens {
            if !sink.alive() {
                return Err(Cancelled.into());
            }
            std::thread::sleep(self.step);
            if sink.delta("t").is_break() {
                return Err(Cancelled.into());
            }
        }
        Ok(Completion {
            text: "t".repeat(params.max_tokens),
            prompt_tokens: 7,
            completion_tokens: params.max_tokens as u32,
            stopped: false,
            length: true,
        })
    }
    fn model_digest_hex(&self) -> String {
        "bench".into()
    }
    fn compute_device(&self) -> String {
        "cpu".into()
    }
    fn isolation(&self) -> Isolation {
        Isolation::NONE_OF_IT
    }
}

fn paced(ms: u64) -> Arc<dyn LoadedModel> {
    Arc::new(Paced {
        step: Duration::from_millis(ms),
    })
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| (*x).to_owned()).collect()
}

fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
    move |k| {
        pairs
            .iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| (*v).to_owned())
    }
}

/// A temporary prompt kit, removed when dropped.
struct Kit(PathBuf);

impl Kit {
    fn new(tag: &str) -> Kit {
        let dir = std::env::temp_dir().join(format!("x0ne-bench-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Kit(dir)
    }
    fn put(&self, file: &str, text: &str) {
        std::fs::write(self.0.join(file), text).unwrap();
    }
}

impl Drop for Kit {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The ids core-bench pins too (`x0ne-core/oracle/examples/core-bench.rs`): a failure means
/// the baseline's drawn lengths are no longer core's, and comparing the two is void.
#[test]
fn the_synthetic_ids_are_core_bench_s() {
    assert_eq!(
        synthetic(6, 151_936),
        [126_819, 135_692, 59_481, 30_850, 145_817, 72_711]
    );
    assert!(synthetic(40_000, 1000).iter().all(|&i| i < 1000));
}

/// The kit's files first; `1k` from `4k`; drawn ids for the sizes a name says; `max` later;
/// anything else, or a file that is not a list of ids of this vocabulary, an error.
#[test]
fn the_ids_of_each_length_are_core_bench_s() {
    let kit = Kit::new("plan");
    kit.put("short.ids.json", "[1, 2, 3]");
    let four: Vec<u32> = (0..2000).collect();
    kit.put("4k.ids.json", &serde_json::to_string(&four).unwrap());
    kit.put("big.ids.json", "[1, 5000]");
    kit.put("text.ids.json", "\"hello\"");
    let p = plan(
        &s(&[
            "short", "1k", "4k", "2k", "300", "max", "weird", "big", "text",
        ]),
        Some(&kit.0),
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
    let q = plan(&s(&["short", "1k", "weird"]), None, 4096);
    assert_eq!(q[0].ids, Ids::Ready(synthetic(20, 4096)));
    assert_eq!(q[1].ids, Ids::Ready(synthetic(1024, 4096)));
    assert!(matches!(&q[2].ids, Ids::Bad(why) if why.contains("no prompt kit")));
}

#[test]
fn the_settings_scripts_bench_sh_passes() {
    let d = settings(&env_of(&[])).unwrap();
    assert_eq!(
        (d.reps, d.decode, d.max_queue, d.level_secs, d.load_tokens),
        (5, 256, 8, 30, 256)
    );
    assert_eq!(
        (d.soak_secs, d.soak_clients, d.soak_window_secs),
        (600, 4, 60)
    );
    assert_eq!(d.clients, [1, 2, 4, 8, 16, 32]);
    assert_eq!(d.ticks, s(&["production", "short"]));
    assert_eq!(d.engine_prompts, s(&["short", "4k", "8k", "16k"]));
    assert_eq!(d.lengths, s(&["short", "1k", "4k", "8k", "16k", "max"]));
    assert_eq!(
        (d.model, d.prompts, d.load_prompt),
        (None, None, "short".into())
    );
    let set = settings(&env_of(&[
        ("X0NE_BENCH_MODEL", "/data/model"),
        ("X0NE_BENCH_CLIENTS", "1,3"),
        ("X0NE_BENCH_TICKS", "short"),
        ("X0NE_BENCH_MAX_QUEUE", "0"),
        ("X0NE_BENCH_SOAK_SECS", "120"),
        ("X0NE_BENCH_SOAK_WINDOW_SECS", "120"),
    ]))
    .unwrap();
    assert_eq!(set.model, Some(PathBuf::from("/data/model")));
    assert_eq!(
        (set.clients, set.ticks, set.max_queue),
        (vec![1, 3], s(&["short"]), 0)
    );
    for bad in [
        ("X0NE_BENCH_DECODE", "1"),
        ("X0NE_BENCH_REPS", "0"),
        ("X0NE_BENCH_LEVEL_SECS", "x"),
        ("X0NE_BENCH_CLIENTS", "1,,2"),
        ("X0NE_BENCH_CLIENTS", "0"),
        ("X0NE_BENCH_TICKS", "fast"),
        ("X0NE_BENCH_SOAK_WINDOW_SECS", "900"),
        ("X0NE_BENCH_LENGTHS", "short,"),
    ] {
        let pairs = [bad];
        assert!(settings(&env_of(&pairs)).is_err(), "{bad:?}");
    }
}

/// One line, after a newline, whatever the strings of the result hold.
#[test]
fn a_result_is_one_line() {
    let v = json!({"schema": 1, "kind": "engine", "reason": "two\nlines"});
    let out = result_line(&v);
    assert!(out.starts_with('\n'));
    let lines: Vec<&str> = out
        .lines()
        .filter(|l| l.contains("x0ne-bench-result: "))
        .collect();
    assert_eq!(lines.len(), 1, "{out}");
    let json = lines[0].strip_prefix("x0ne-bench-result: ").unwrap();
    assert_eq!(serde_json::from_str::<Value>(json).unwrap(), v);
}

#[test]
fn llama_server_s_version_file_type_and_rates() {
    let log = "ggml_cuda_init: found 1 CUDA devices\n\
               build: 6400 (abc1234) with cc (GCC) 12.2.0 for x86_64-linux-gnu\n\
               print_info: file type   = BF16\n";
    assert_eq!(
        llama_facts(log),
        (Some("6400 (abc1234)".into()), Some("BF16".into()))
    );
    let old = "main: build = 3000 (deadbee)\nllm_load_print_meta: model ftype      = BF16\n";
    assert_eq!(
        llama_facts(old),
        (Some("3000 (deadbee)".into()), Some("BF16".into()))
    );
    assert_eq!(llama_facts("nothing here\n"), (None, None));

    let v = json!({"timings": {"prompt_n": 100, "prompt_ms": 50.0, "predicted_n": 256, "predicted_ms": 1000.0}});
    assert_eq!(llama_rates(&v, 100, 256), Ok((0.05, 2000.0, 256.0)));
    let short = llama_rates(&v, 101, 256).unwrap_err();
    assert!(short.contains("100 of the 101"), "{short}");
    let fewer = llama_rates(&v, 100, 128).unwrap_err();
    assert!(fewer.contains("256"), "{fewer}");
    assert!(llama_rates(&json!({}), 100, 256).is_err());
}

/// A stream and a whole answer, a first pass apart and two repetitions, against the unit
/// tests' server; a prompt the kit lacks, or too long for the context, skipped with why.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_engine_case_measures_a_stream_and_a_whole_answer() {
    let kit = Kit::new("engine");
    kit.put(
        "short.chat.json",
        r#"{"messages": [{"role": "user", "content": "hi"}]}"#,
    );
    kit.put("short.ids.json", "[1, 2, 3]");
    let st = Settings {
        reps: 2,
        decode: 6,
        ..settings(&env_of(&[])).unwrap()
    };
    let addr = spawn(paced(2), Tick::SHORT, Admission::new(8)).await;
    let c = engine_case(addr, &kit.0, "short", "short", &st, Some(4096)).await;
    assert_eq!(
        (
            &c["prompt"],
            &c["tick"],
            &c["tick_ms"],
            &c["ids_tokens"],
            &c["prompt_tokens"]
        ),
        (
            &json!("short"),
            &json!("short"),
            &json!(5.0),
            &json!(3),
            &json!(7)
        ),
        "{c}"
    );
    for key in ["stream", "whole"] {
        assert_eq!(c[key]["errors"], json!([]), "{c}");
        assert_eq!(c[key]["finish"], json!(["length"]), "{c}");
        assert_eq!(
            c[key]["completion_tokens"],
            json!({"min": 6, "max": 6}),
            "{c}"
        );
    }
    assert_eq!(c["stream"]["ttft_s"]["n"], 2, "{c}");
    assert!(
        c["stream"]["first"]["ttft_s"]
            .as_f64()
            .is_some_and(|t| t > 0.0),
        "{c}"
    );
    assert_eq!(c["whole"]["e2e_s"]["n"], 2, "{c}");
    let rate = &c["stream"]["decode_tok_s"];
    assert!(
        rate.is_null() || rate["median"].as_f64().is_some_and(|r| r > 0.0),
        "{c}"
    );

    let none = engine_case(addr, &kit.0, "4k", "short", &st, Some(4096)).await;
    assert_eq!(none["skipped"], "no 4k.chat.json in the kit");
    let long = engine_case(addr, &kit.0, "short", "short", &st, Some(8)).await;
    assert_eq!(long["skipped"], "3 + 6 tokens exceed the context of 8");
}

/// One place and no waiting room: the callers beyond it are refused with a Retry-After of 1 s,
/// counted, never errors, and wait it out within the level.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_full_queue_is_counted_as_refusals_with_their_retry_after() {
    let mut admission = Admission::new(0);
    admission.retry_after_seconds = 1;
    let addr = spawn(paced(5), Tick::SHORT, admission).await;
    let b = body(&json!([{"role": "user", "content": "hi"}]), 4, true);
    let (seen, seconds) = run_clients(
        &url(addr),
        &b,
        4,
        Instant::now(),
        Duration::from_millis(300),
    )
    .await;
    let v = level_summary(&seen, 4, seconds);
    assert_eq!(v["errors"], 0, "{v}");
    assert!(v["ok"].as_u64().is_some_and(|n| n >= 1), "{v}");
    assert!(v["rejected"].as_u64().is_some_and(|n| n >= 1), "{v}");
    assert_eq!(v["rejected_without_retry_after"], 0, "{v}");
    assert_eq!(v["retry_after_s"], json!({"min": 1.0, "max": 1.0}), "{v}");
    assert_eq!(
        v["requests"].as_u64(),
        Some(v["ok"].as_u64().unwrap() + v["rejected"].as_u64().unwrap())
    );
}

/// A soak of three windows: four samples of memory, one before and one at each window's end.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_soak_samples_memory_at_each_window() {
    let addr = spawn(paced(2), Tick::SHORT, Admission::new(8)).await;
    let b = body(&json!([{"role": "user", "content": "hi"}]), 4, true);
    let vram: Arc<dyn Fn() -> Option<u64> + Send + Sync> = Arc::new(|| Some(42));
    let (seen, samples, _) = soak(
        &url(addr),
        &b,
        2,
        Duration::from_millis(300),
        Duration::from_millis(100),
        vram,
    )
    .await;
    assert_eq!(samples.len(), 4, "{samples:?}");
    assert!(samples
        .iter()
        .all(|x| x.rss_kib.is_some() && x.vram_used_mib == Some(42)));
    assert!(samples.windows(2).all(|w| w[0].at_s <= w[1].at_s));
    let (windows, summary) = soak_summary(&seen, &samples, 0.3, 0.1);
    assert_eq!(windows.as_array().map(Vec::len), Some(3));
    assert_eq!(summary["errors"], 0, "{summary}");
    assert!(summary["ok"].as_u64().is_some_and(|n| n >= 2), "{summary}");
}
