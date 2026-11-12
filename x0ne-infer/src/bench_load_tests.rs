//! The client of make bench's server tiers (`bench_tests.rs`, `scripts/bench.sh`): requests to
//! `/v1/chat/completions` as a caller sees them, streamed or whole; callers that send again as
//! soon as their answer ends; what a level or a soak of them comes to. Every time is the
//! client's, over loopback, in the process of the server: it holds the server's work and the
//! client's, and no network.

use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::sse::{Data, SseReader};

/// One request as its client saw it. `sent_s` from the start of its level, the other times
/// from its sending.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Seen {
    pub client: usize,
    pub sent_s: f64,
    /// 0: no answer came.
    pub status: u16,
    pub retry_after_s: Option<f64>,
    /// The first content of a stream.
    pub ttft_s: Option<f64>,
    pub last_content_s: Option<f64>,
    /// `[DONE]` read, or the whole body.
    pub end_s: f64,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub finish: Option<String>,
    pub error: Option<String>,
}

impl Seen {
    /// Answered, and nothing wrong in the answer.
    pub fn ok(&self) -> bool {
        self.status == 200 && self.error.is_none()
    }

    /// Refused for a full queue: 503, which is an answer, not an error.
    pub fn rejected(&self) -> bool {
        self.status == 503 && self.error.is_none()
    }

    fn first_token_at(&self) -> Option<f64> {
        self.ttft_s.map(|t| self.sent_s + t)
    }

    fn ended_at(&self) -> f64 {
        self.sent_s + self.end_s
    }
}

/// A chat request: greedy (no temperature), `max_tokens`, its usage asked for when streamed.
pub(crate) fn body(messages: &Value, max_tokens: usize, stream: bool) -> Value {
    let mut b = json!({"messages": messages, "max_tokens": max_tokens});
    if stream {
        b["stream"] = json!(true);
        b["stream_options"] = json!({"include_usage": true});
    }
    b
}

/// A client for the tiers: one pool of connections, ten minutes at most per answer.
pub(crate) fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .expect("a reqwest client")
}

/// Sends `body` to `url` and reads the answer to its end.
pub(crate) async fn chat(
    client: &reqwest::Client,
    url: &str,
    body: &Value,
    t0: Instant,
    who: usize,
) -> Seen {
    let sent = Instant::now();
    let mut seen = Seen {
        client: who,
        sent_s: sent.duration_since(t0).as_secs_f64(),
        ..Seen::default()
    };
    let answer = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body.to_string())
        .send()
        .await;
    let mut resp = match answer {
        Ok(r) => r,
        Err(e) => {
            seen.error = Some(format!("no answer: {e}"));
            seen.end_s = sent.elapsed().as_secs_f64();
            return seen;
        }
    };
    seen.status = resp.status().as_u16();
    seen.retry_after_s = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<f64>().ok());
    if seen.status != 200 {
        let text = resp.text().await.unwrap_or_default();
        if seen.status != 503 {
            let head: String = text.chars().take(160).collect();
            seen.error = Some(format!("HTTP {}: {head}", seen.status));
        }
    } else if body["stream"] == true {
        read_stream(&mut resp, sent, &mut seen).await;
    } else {
        match resp.bytes().await {
            Ok(b) => match serde_json::from_slice::<Value>(&b) {
                Ok(v) => {
                    let c = &v["choices"][0];
                    seen.finish = c["finish_reason"].as_str().map(str::to_owned);
                    usage(&v["usage"], &mut seen);
                    if c["message"]["content"].as_str().is_none() {
                        seen.error = Some("an answer without content".into());
                    }
                }
                Err(e) => seen.error = Some(format!("not JSON: {e}")),
            },
            Err(e) => seen.error = Some(format!("body cut: {e}")),
        }
    }
    seen.end_s = sent.elapsed().as_secs_f64();
    seen
}

async fn read_stream(resp: &mut reqwest::Response, sent: Instant, seen: &mut Seen) {
    let mut reader = SseReader::new();
    let mut events = Vec::new();
    let mut done = false;
    loop {
        match resp.chunk().await {
            Ok(Some(bytes)) => {
                if let Err(e) = reader.push(&bytes, &mut events) {
                    seen.error = Some(format!("not SSE: {e}"));
                    return;
                }
                let at = sent.elapsed().as_secs_f64();
                for event in events.drain(..) {
                    match event {
                        Data::Done => done = true,
                        Data::Json(v) => note(&v, at, seen),
                    }
                }
            }
            Ok(None) => break,
            Err(e) => {
                seen.error = Some(format!("stream cut: {e}"));
                return;
            }
        }
    }
    if !done && seen.error.is_none() {
        seen.error = Some("the stream ended without [DONE]".into());
    }
}

/// One event of a stream, read at `at`.
fn note(v: &Value, at: f64, seen: &mut Seen) {
    if let Some(e) = v.get("error") {
        seen.error
            .get_or_insert_with(|| format!("error event: {e}"));
    }
    let choice = &v["choices"][0];
    if choice["delta"]["content"]
        .as_str()
        .is_some_and(|c| !c.is_empty())
    {
        seen.ttft_s.get_or_insert(at);
        seen.last_content_s = Some(at);
    }
    if let Some(f) = choice["finish_reason"].as_str() {
        seen.finish = Some(f.to_owned());
    }
    usage(&v["usage"], seen);
}

fn usage(u: &Value, seen: &mut Seen) {
    if let Some(p) = u["prompt_tokens"].as_u64() {
        seen.prompt_tokens = Some(p);
    }
    if let Some(c) = u["completion_tokens"].as_u64() {
        seen.completion_tokens = Some(c);
    }
}

/// `clients` callers from `t0`, each sending `body` again as soon as its answer ends, until
/// `run_for` is past. A refused one waits its Retry-After (1 s without one), a failed one
/// 100 ms, never past `run_for`. Every request seen, and the seconds until the last answer.
pub(crate) async fn run_clients(
    url: &str,
    body: &Value,
    clients: usize,
    t0: Instant,
    run_for: Duration,
) -> (Vec<Seen>, f64) {
    let pool = client();
    let mut tasks = Vec::new();
    for who in 0..clients {
        let (pool, url, body) = (pool.clone(), url.to_owned(), body.clone());
        tasks.push(tokio::spawn(async move {
            let mut seen = Vec::new();
            while t0.elapsed() < run_for {
                let s = chat(&pool, &url, &body, t0, who).await;
                let pause = if s.rejected() {
                    s.retry_after_s.unwrap_or(1.0)
                } else if s.ok() {
                    0.0
                } else {
                    0.1
                };
                seen.push(s);
                let left = run_for.saturating_sub(t0.elapsed());
                if pause > 0.0 && !left.is_zero() {
                    tokio::time::sleep(Duration::from_secs_f64(pause).min(left)).await;
                }
            }
            seen
        }));
    }
    let mut all = Vec::new();
    for t in tasks {
        all.extend(t.await.expect("a client of the level panicked"));
    }
    (all, t0.elapsed().as_secs_f64())
}

/// The host memory and the device memory, at `at_s` from the start of a soak.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Sample {
    pub at_s: f64,
    pub rss_kib: Option<u64>,
    pub vram_used_mib: Option<u64>,
}

/// This process's resident memory (`VmRSS`), KiB: the server, its model's host side and the
/// clients together.
pub(crate) fn rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))?
        .trim()
        .strip_suffix("kB")?
        .trim()
        .parse()
        .ok()
}

/// A soak: [`run_clients`] for `run_for`, and a [`Sample`] before the first request, then at
/// the end of each whole window.
pub(crate) async fn soak(
    url: &str,
    body: &Value,
    clients: usize,
    run_for: Duration,
    window: Duration,
    vram_used_mib: std::sync::Arc<dyn Fn() -> Option<u64> + Send + Sync>,
) -> (Vec<Seen>, Vec<Sample>, f64) {
    let t0 = Instant::now();
    let windows = whole_windows(run_for, window);
    let sampler = tokio::spawn(async move {
        let mut samples = Vec::new();
        for k in 0..=windows {
            let at = window.saturating_mul(u32::try_from(k).unwrap_or(u32::MAX));
            tokio::time::sleep(at.saturating_sub(t0.elapsed())).await;
            samples.push(Sample {
                at_s: t0.elapsed().as_secs_f64(),
                rss_kib: rss_kib(),
                vram_used_mib: vram_used_mib(),
            });
        }
        samples
    });
    let (seen, seconds) = run_clients(url, body, clients, t0, run_for).await;
    let samples = sampler.await.expect("the sampler of the soak panicked");
    (seen, samples, seconds)
}

/// The whole windows in `run_for`.
pub(crate) fn whole_windows(run_for: Duration, window: Duration) -> usize {
    if window.is_zero() {
        return 0;
    }
    usize::try_from(run_for.as_nanos() / window.as_nanos()).unwrap_or(usize::MAX)
}

/// Median (the mean of the middle two of an even count), smallest, largest, mean, standard
/// deviation (n - 1) and count; null for none.
pub(crate) fn stats(v: &[f64]) -> Value {
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

/// p50, p95 and p99 by nearest rank, and the count; null percentiles for none.
pub(crate) fn percentiles(v: &[f64]) -> Value {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    // The smallest value with at least p percent of them at or below it.
    let at = |p: usize| (n > 0).then(|| s[(p * n).div_ceil(100).max(1) - 1]);
    json!({"p50": at(50), "p95": at(95), "p99": at(99), "n": n})
}

/// Jain's fairness index of `x`: 1 when every share is the same, 1/n when one takes all; none
/// when nothing was shared.
pub(crate) fn jain(x: &[f64]) -> Option<f64> {
    let sum: f64 = x.iter().sum();
    let squares: f64 = x.iter().map(|v| v * v).sum();
    (squares > 0.0).then(|| sum * sum / (x.len() as f64 * squares))
}

/// (inversions, pairs) over the answered requests with a first token: the pairs where the
/// later one was sent before the earlier one's first token came (they waited together), and
/// of them those whose first tokens came in the other order. Two sent at the same instant
/// have no order.
pub(crate) fn fifo(seen: &[Seen]) -> (usize, usize) {
    let mut v: Vec<(f64, f64)> = seen
        .iter()
        .filter(|s| s.ok())
        .filter_map(|s| s.first_token_at().map(|f| (s.sent_s, f)))
        .collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut inversions, mut pairs) = (0, 0);
    for (i, &(sent_a, first_a)) in v.iter().enumerate() {
        for &(sent_b, first_b) in &v[i + 1..] {
            if sent_b > sent_a && sent_b < first_a {
                pairs += 1;
                if first_b < first_a {
                    inversions += 1;
                }
            }
        }
    }
    (inversions, pairs)
}

/// What one level of `clients` callers over `seconds` came to.
pub(crate) fn level_summary(seen: &[Seen], clients: usize, seconds: f64) -> Value {
    let ok: Vec<&Seen> = seen.iter().filter(|s| s.ok()).collect();
    let rejected = seen.iter().filter(|s| s.rejected()).count();
    let retry: Vec<f64> = seen
        .iter()
        .filter(|s| s.rejected())
        .filter_map(|s| s.retry_after_s)
        .collect();
    let errors: Vec<&String> = seen.iter().filter_map(|s| s.error.as_ref()).collect();
    let mut samples: Vec<&String> = Vec::new();
    for e in &errors {
        if samples.len() < 3 && !samples.contains(e) {
            samples.push(e);
        }
    }
    let tokens: u64 = ok.iter().filter_map(|s| s.completion_tokens).sum();
    let per_client: Vec<f64> = (0..clients)
        .map(|c| ok.iter().filter(|s| s.client == c).count() as f64)
        .collect();
    let (inversions, pairs) = fifo(seen);
    let per_s = |x: f64| if seconds > 0.0 { x / seconds } else { 0.0 };
    json!({
        "clients": clients, "seconds": seconds, "requests": seen.len(), "ok": ok.len(),
        "rejected": rejected,
        "rejected_frac": if seen.is_empty() { 0.0 } else { rejected as f64 / seen.len() as f64 },
        "retry_after_s": if retry.is_empty() {
            Value::Null
        } else {
            json!({"min": retry.iter().copied().fold(f64::INFINITY, f64::min),
                   "max": retry.iter().copied().fold(f64::NEG_INFINITY, f64::max)})
        },
        "rejected_without_retry_after": rejected - retry.len(),
        "errors": errors.len(), "error_samples": samples,
        "ttft_s": percentiles(&ok.iter().filter_map(|s| s.ttft_s).collect::<Vec<_>>()),
        "e2e_s": percentiles(&ok.iter().map(|s| s.end_s).collect::<Vec<_>>()),
        "throughput_tok_s": per_s(tokens as f64), "ok_per_s": per_s(ok.len() as f64),
        "jain_index": jain(&per_client), "fifo_inversions": inversions, "fifo_pairs": pairs,
    })
}

/// The windows of a soak that sent requests for `run_s` seconds, and its summary. Window `i`
/// holds the answers that ended in `[i * window_s, (i + 1) * window_s)`, `samples[i + 1]` its
/// memory (`samples[0]` before the first request); only the whole windows of `run_s` count,
/// the answers that end after them only in the totals. Drift: the last window's throughput
/// against the first's, in percent. Slope: the least-squares line through every window's
/// throughput, per hour, in percent of their mean.
pub(crate) fn soak_summary(
    seen: &[Seen],
    samples: &[Sample],
    run_s: f64,
    window_s: f64,
) -> (Value, Value) {
    let n = if window_s > 0.0 {
        // 0.3 / 0.1 is 2.999...: three windows.
        (run_s / window_s + 1e-9).floor() as usize
    } else {
        0
    };
    let mut windows = Vec::new();
    let mut rates = Vec::new();
    let mut tokens_all = 0u64;
    for i in 0..n {
        let (from, to) = (i as f64 * window_s, (i + 1) as f64 * window_s);
        let here: Vec<&Seen> = seen
            .iter()
            .filter(|s| (from..to).contains(&s.ended_at()))
            .collect();
        let ok: Vec<&&Seen> = here.iter().filter(|s| s.ok()).collect();
        let tokens: u64 = ok.iter().filter_map(|s| s.completion_tokens).sum();
        tokens_all += tokens;
        let rate = tokens as f64 / window_s;
        rates.push(rate);
        let ttft: Vec<f64> = ok.iter().filter_map(|s| s.ttft_s).collect();
        let sample = samples.get(i + 1);
        windows.push(json!({
            "end_s": to, "ok": ok.len(), "tokens": tokens, "throughput_tok_s": rate,
            "errors": here.iter().filter(|s| s.error.is_some()).count(),
            "rejected": here.iter().filter(|s| s.rejected()).count(),
            "ttft_p50_s": percentiles(&ttft)["p50"],
            "rss_kib": sample.and_then(|x| x.rss_kib),
            "vram_used_mib": sample.and_then(|x| x.vram_used_mib),
        }));
    }
    let drift = match (rates.first(), rates.last()) {
        (Some(&first), Some(&last)) if n >= 2 && first > 0.0 => {
            Some((last - first) / first * 100.0)
        }
        _ => None,
    };
    let slope = (n >= 2).then(|| {
        let x: Vec<f64> = (0..n)
            .map(|i| (i as f64 + 0.5) * window_s / 3600.0)
            .collect();
        let (mx, my) = (
            x.iter().sum::<f64>() / n as f64,
            rates.iter().sum::<f64>() / n as f64,
        );
        let num: f64 = x.iter().zip(&rates).map(|(a, b)| (a - mx) * (b - my)).sum();
        let den: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
        (my > 0.0 && den > 0.0).then(|| num / den / my * 100.0)
    });
    let rss: Vec<u64> = samples.iter().filter_map(|x| x.rss_kib).collect();
    let summary = json!({
        "requests": seen.len(),
        "ok": seen.iter().filter(|s| s.ok()).count(),
        "errors": seen.iter().filter(|s| s.error.is_some()).count(),
        "rejected": seen.iter().filter(|s| s.rejected()).count(),
        "windows": n,
        "throughput_tok_s": (n > 0).then(|| tokens_all as f64 / (n as f64 * window_s)),
        "throughput_drift_pct": drift,
        "throughput_slope_pct_per_h": slope.flatten(),
        "rss_first_kib": samples.first().and_then(|x| x.rss_kib),
        "rss_last_kib": samples.last().and_then(|x| x.rss_kib),
        "rss_peak_kib": rss.iter().max(),
        "vram_first_mib": samples.first().and_then(|x| x.vram_used_mib),
        "vram_last_mib": samples.last().and_then(|x| x.vram_used_mib),
        "vram_peak_mib": samples.iter().filter_map(|x| x.vram_used_mib).max(),
    });
    (Value::Array(windows), summary)
}

fn answer(client: usize, sent_s: f64, ttft_s: f64, end_s: f64, tokens: u64) -> Seen {
    Seen {
        client,
        sent_s,
        status: 200,
        ttft_s: Some(ttft_s),
        last_content_s: Some(end_s),
        end_s,
        completion_tokens: Some(tokens),
        finish: Some("length".into()),
        ..Seen::default()
    }
}

#[test]
fn the_statistics_of_repetitions() {
    assert_eq!(
        stats(&[3.0, 1.0, 2.0]),
        json!({"median": 2.0, "min": 1.0, "max": 3.0, "mean": 2.0, "stddev": 1.0, "n": 3})
    );
    assert_eq!(stats(&[1.0, 2.0, 3.0, 4.0])["median"], json!(2.5));
    assert_eq!(stats(&[5.0])["stddev"], json!(0.0));
    assert_eq!(stats(&[]), Value::Null);
}

#[test]
fn percentiles_by_nearest_rank() {
    let hundred: Vec<f64> = (1..=100).map(f64::from).collect();
    assert_eq!(
        percentiles(&hundred),
        json!({"p50": 50.0, "p95": 95.0, "p99": 99.0, "n": 100})
    );
    assert_eq!(
        percentiles(&[3.0, 1.0, 2.0]),
        json!({"p50": 2.0, "p95": 3.0, "p99": 3.0, "n": 3})
    );
    assert_eq!(
        percentiles(&[]),
        json!({"p50": null, "p95": null, "p99": null, "n": 0})
    );
}

#[test]
fn jain_s_index_of_the_answers_per_client() {
    assert_eq!(jain(&[2.0, 2.0, 2.0, 2.0]), Some(1.0));
    assert_eq!(jain(&[4.0, 0.0, 0.0, 0.0]), Some(0.25));
    assert_eq!(jain(&[0.0, 0.0]), None);
    assert_eq!(jain(&[]), None);
}

/// b came first though sent after a, while a waited: one inversion. c was sent once a had
/// its first token: no pair. d waited with a and b and came last: two pairs, in order.
#[test]
fn fifo_inversions_count_only_requests_that_waited_together() {
    let a = answer(0, 0.0, 1.0, 2.0, 4);
    let b = answer(1, 0.5, 0.3, 1.0, 4);
    let c = answer(2, 2.0, 1.0, 1.5, 4);
    let d = answer(3, 0.6, 0.6, 1.0, 4);
    assert_eq!(fifo(&[a.clone(), b.clone(), c.clone()]), (1, 1));
    assert_eq!(fifo(&[c, d, b, a.clone()]), (1, 3));
    let refused = Seen {
        status: 503,
        sent_s: 0.1,
        ..Seen::default()
    };
    let twin = answer(4, 0.0, 0.5, 1.0, 4);
    assert_eq!(fifo(&[a, refused, twin]), (0, 0));
}

#[test]
fn a_level_sums_its_answers_refusals_and_errors() {
    let refused = |client, sent_s, retry: Option<f64>| Seen {
        client,
        sent_s,
        status: 503,
        retry_after_s: retry,
        end_s: 0.01,
        ..Seen::default()
    };
    let failed = |client, sent_s, why: &str| Seen {
        client,
        sent_s,
        status: 500,
        end_s: 0.01,
        error: Some(why.into()),
        ..Seen::default()
    };
    let seen = [
        answer(0, 0.0, 0.1, 1.0, 10),
        answer(0, 1.0, 0.1, 1.0, 10),
        answer(1, 0.0, 0.2, 2.0, 20),
        refused(2, 0.0, Some(5.0)),
        refused(2, 5.0, Some(1.0)),
        refused(3, 0.0, None),
        failed(3, 1.0, "HTTP 500: boom"),
        failed(3, 2.0, "HTTP 500: boom"),
        failed(3, 3.0, "no answer: reset"),
    ];
    let v = level_summary(&seen, 4, 10.0);
    assert_eq!(
        v,
        json!({
            "clients": 4, "seconds": 10.0, "requests": 9, "ok": 3, "rejected": 3,
            "rejected_frac": 3.0 / 9.0, "retry_after_s": {"min": 1.0, "max": 5.0},
            "rejected_without_retry_after": 1, "errors": 3,
            "error_samples": ["HTTP 500: boom", "no answer: reset"],
            "ttft_s": {"p50": 0.1, "p95": 0.2, "p99": 0.2, "n": 3},
            "e2e_s": {"p50": 1.0, "p95": 2.0, "p99": 2.0, "n": 3},
            "throughput_tok_s": 4.0, "ok_per_s": 0.3,
            "jain_index": 9.0 / 20.0, "fifo_inversions": 0, "fifo_pairs": 0,
        })
    );
    let empty = level_summary(&[], 2, 1.0);
    assert_eq!(
        (
            &empty["requests"],
            &empty["rejected_frac"],
            &empty["jain_index"]
        ),
        (&json!(0), &json!(0.0), &Value::Null)
    );
}

/// A soak of 30 s: three whole windows of 10 s at 10, 11 and 12 tok/s; an answer that ends
/// after them counts in the totals only.
#[test]
fn a_soak_comes_to_its_windows_drift_and_slope() {
    let mut seen = vec![
        answer(0, 1.0, 0.5, 2.0, 100),
        answer(0, 12.0, 0.5, 2.0, 110),
        answer(1, 21.0, 0.5, 2.0, 60),
        answer(1, 25.0, 0.5, 2.0, 60),
        answer(0, 29.0, 0.5, 2.0, 999),
    ];
    seen.push(Seen {
        status: 503,
        retry_after_s: Some(5.0),
        sent_s: 3.0,
        end_s: 0.1,
        ..Seen::default()
    });
    let sample = |at_s, rss| Sample {
        at_s,
        rss_kib: Some(rss),
        vram_used_mib: Some(9000),
    };
    let samples = [
        sample(0.0, 1000),
        sample(10.0, 1500),
        sample(20.0, 1200),
        sample(30.0, 1100),
    ];
    let (windows, summary) = soak_summary(&seen, &samples, 30.0, 10.0);
    let rates: Vec<f64> = windows
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w["throughput_tok_s"].as_f64().unwrap())
        .collect();
    assert_eq!(rates, [10.0, 11.0, 12.0]);
    assert_eq!(windows[0]["rejected"], 1);
    assert_eq!(windows[1]["rss_kib"], 1200);
    assert_eq!(windows[0]["ok"], 1);
    assert_eq!(windows[2]["ok"], 2);
    assert_eq!(summary["ok"], 5);
    assert_eq!(summary["rejected"], 1);
    assert_eq!(summary["errors"], 0);
    assert_eq!(summary["throughput_tok_s"], 11.0);
    assert!((summary["throughput_drift_pct"].as_f64().unwrap() - 20.0).abs() < 1e-9);
    // 1 tok/s more every 10 s: 360 per hour, over a mean of 11.
    let slope = summary["throughput_slope_pct_per_h"].as_f64().unwrap();
    assert!((slope - 360.0 / 11.0 * 100.0).abs() < 1e-6, "{slope}");
    assert_eq!(
        (
            &summary["rss_first_kib"],
            &summary["rss_last_kib"],
            &summary["rss_peak_kib"]
        ),
        (&json!(1000), &json!(1100), &json!(1500))
    );
    assert_eq!(summary["vram_first_mib"], 9000);
    // One window: no drift, no slope.
    let (one, s1) = soak_summary(&seen[..1], &samples[..2], 10.0, 10.0);
    assert_eq!(one.as_array().unwrap().len(), 1);
    assert_eq!(
        (
            &s1["throughput_drift_pct"],
            &s1["throughput_slope_pct_per_h"]
        ),
        (&Value::Null, &Value::Null)
    );
}

#[test]
fn the_whole_windows_of_a_soak() {
    let ms = Duration::from_millis;
    assert_eq!(whole_windows(ms(300), ms(100)), 3);
    assert_eq!(whole_windows(ms(299), ms(100)), 2);
    assert_eq!(whole_windows(ms(50), ms(100)), 0);
    assert_eq!(whole_windows(ms(50), ms(0)), 0);
}

#[test]
fn the_resident_memory_of_this_process_is_read() {
    assert!(rss_kib().is_some_and(|k| k > 0));
}
