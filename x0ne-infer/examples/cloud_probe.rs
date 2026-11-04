//! Real-hardware probe of the attested channel (scripts/e2e-cloud.sh, pass 2).
//! Plays the relay's client half with its own session, then measures what the
//! relay cannot report. Every case is `x0ne_infer::cloud_probe`, played first
//! against the fake TD (`src/cloud_e2e_tests.rs`). Not a product: no anchor or
//! check is relaxed; it calls the relay's code. Needs `--features probe`.
//!
//!   cloud_probe --out DIR --reference FILE [--measurements FILE]
//!               [--model qwen3:0.6b] [--machine-type a3-highgpu-1g] [--ttl-secs 3600]
//!               [--max-queue 8] [--usd-per-hour 6.60]
//! Writes DIR/config.json (the x0ne-config metadata), then reads "IP:PORT" on stdin.
//!
//! Phases, most information per VM second first, DIR/probe.json rewritten
//! after each so a stop at the cost cap keeps what was measured:
//!   1 attacks (~25 s): connections without the relay, C3 burst past the
//!     bucket, then the nominal rate
//!   2 quote time inside the TD, `Server-Timing` (~7 s)
//!   3 key, model ready, one long completion then two at once: tokens/s and
//!     $ per million completion tokens (~1 to 2 min)
//!   4 rechecks of 2026-10-14: C1 cross-connection replay, one quote per
//!     connection, C2 resumption (~10 s)

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use x0ne_infer::cloud_probe::{self as probe, ms};
use x0ne_infer::cloud_serve::QuoteLimits;
use x0ne_infer::{attest_v2, cloud_config, measurements, reference, registry, remote_verify, tls};

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn urandom32() -> [u8; 32] {
    use std::io::Read;
    let mut b = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .expect("urandom");
    b
}

fn log(msg: impl AsRef<str>) {
    eprintln!("{} probe: {}", now_unix(), msg.as_ref());
}

fn stats(v: &[f64]) -> serde_json::Value {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if s.is_empty() {
        return serde_json::json!(null);
    }
    serde_json::json!({ "n": s.len(), "min": s[0], "median": s[s.len() / 2], "max": s[s.len() - 1] })
}

struct Report {
    path: PathBuf,
    map: serde_json::Map<String, serde_json::Value>,
}

impl Report {
    fn put(&mut self, k: &str, v: serde_json::Value) {
        self.map.insert(k.into(), v);
    }
    /// Rewritten whole after each phase: the file is always a valid JSON.
    fn save(&self) {
        let tmp = self.path.with_extension("json.tmp");
        if std::fs::write(&tmp, serde_json::to_vec_pretty(&self.map).unwrap()).is_ok() {
            let _ = std::fs::rename(&tmp, &self.path);
        }
    }
}

fn answers(v: &[Result<probe::Answer, String>]) -> Vec<String> {
    v.iter()
        .map(|a| match a {
            Ok(a) => format!("{} in {:.0} ms", a.status, a.ms),
            Err(e) => format!("error: {e}"),
        })
        .collect()
}

#[tokio::main]
async fn main() {
    let mut out: Option<PathBuf> = None;
    let mut reference_path: Option<PathBuf> = None;
    let mut measurements_path: Option<PathBuf> = None;
    let mut model = "qwen3:0.6b".to_string();
    let mut machine_type = "a3-highgpu-1g".to_string();
    let mut ttl: u64 = 3600;
    let mut max_queue = x0ne_infer::server::DEFAULT_MAX_QUEUE;
    let mut usd_per_hour: Option<f64> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let v = it.next().expect("flag needs a value");
        match a.as_str() {
            "--out" => out = Some(v.into()),
            "--reference" => reference_path = Some(v.into()),
            "--measurements" => measurements_path = Some(v.into()),
            "--model" => model = v,
            "--machine-type" => machine_type = v,
            "--ttl-secs" => ttl = v.parse().expect("ttl"),
            "--max-queue" => max_queue = v.parse().expect("max-queue"),
            "--usd-per-hour" => usd_per_hour = Some(v.parse().expect("usd-per-hour")),
            o => panic!("unknown flag {o}"),
        }
    }
    let out = out.expect("--out");
    std::fs::create_dir_all(&out).expect("out dir");
    let mut report = Report {
        path: out.join("probe.json"),
        map: serde_json::Map::new(),
    };
    // The TD's limits, from its `--max-queue` (x0ne-infer.service).
    let limits = QuoteLimits::for_queue(max_queue);
    report.put(
        "quote_limits",
        serde_json::json!({
            "max_queue": max_queue, "burst": limits.burst,
            "refill_every_s": limits.refill_every.as_secs_f64(), "per_connection": limits.per_connection,
        }),
    );
    let reference = reference::load(
        &reference_path.expect("--reference"),
        true,
        &x0ne_infer::archive_key::Anchor::production(),
        &x0ne_infer::transparency::Anchor::production(),
        &out.join("tlog-state"),
        now_unix(),
    )
    .unwrap_or_else(|e| panic!("reference: {e}"));
    let catalog = registry::Catalog::embedded();
    let entry = catalog.get(&model).expect("model").clone();
    let identity = tls::TlsIdentity::generate("x0ne-engine-proxy").expect("identity");
    let config = cloud_config::CloudConfig::new(
        &model,
        identity.spki_sha256(),
        urandom32(),
        now_unix() + ttl,
        "cpu+gpu",
        443,
        256,
    )
    .to_loaded()
    .expect("config");
    std::fs::write(out.join("config.json"), &config.bytes).expect("write config");
    log(format!(
        "config written, sha256 {}",
        cloud_config::hex(&config.sha256)
    ));
    println!("CONFIG_READY {}", out.join("config.json").display());

    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line).expect("stdin");
    let endpoint = line.trim().to_string();
    let host = endpoint
        .rsplit_once(':')
        .map(|(h, _)| h.to_string())
        .expect("host:port");
    let nonce = config.config.nonce();
    let client = remote_verify::TdClient::new(&identity, &endpoint, nonce).expect("client");

    // ── document ──────────────────────────────────────────────────────────
    let t_wait = Instant::now();
    let doc = loop {
        match client.attestation().await {
            Ok(d) => break d,
            Err(e) => {
                if t_wait.elapsed() > Duration::from_mins(25) {
                    log(format!("no document after 25 min: {e}"));
                    std::process::exit(93);
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    };
    log(format!(
        "document received after {:.1} s of polling",
        t_wait.elapsed().as_secs_f64()
    ));
    std::fs::write(
        out.join("document.json"),
        serde_json::to_vec_pretty(&doc).unwrap(),
    )
    .ok();
    pins(&doc, measurements_path.as_deref(), &mut report);

    // ── verify on this machine ───────────────────────────────────────────
    let tls_spki = client.pinned_spki().expect("pinned");
    let exp = remote_verify::Expectations {
        config: &config,
        tls_spki_sha256: tls_spki,
        machine_type: &machine_type,
        reference: &reference,
        catalog_entry: &entry,
    };
    let anchors = remote_verify::Anchors::production();
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap();
    let t = Instant::now();
    let verdict = remote_verify::verify_document(&doc, &exp, &anchors, &http, now_unix()).await;
    let verify_ms = ms(t.elapsed());
    let rep = match verdict {
        Ok(r) => r,
        Err(refusal) => {
            log(format!("REFUSED {}: {}", refusal.code, refusal.reason));
            report.put(
                "refused",
                serde_json::json!({"code": refusal.code, "reason": refusal.reason}),
            );
            report.save();
            std::process::exit(refusal.code);
        }
    };
    log(format!(
        "document VERIFIED in {verify_ms:.0} ms (PCS, Google, NRAS fetched live)"
    ));
    for l in &rep.summary {
        log(format!("  {l}"));
    }
    report.put("verify_document_ms", serde_json::json!(verify_ms));
    report.put("verdict", serde_json::to_value(&rep).unwrap());

    let clock: Arc<dyn Fn() -> u64 + Send + Sync> = Arc::new(now_unix);
    let check_raw =
        remote_verify::SessionCheck::new(&doc, &exp, &anchors, clock.clone()).expect("check");
    let check =
        remote_verify::SessionCheck::new(&doc, &exp, &anchors, clock.clone()).expect("check");
    let t = Instant::now();
    if let Err(r) = client.establish(check).await {
        log(format!("establish REFUSED {}: {}", r.code, r.reason));
        std::process::exit(r.code);
    }
    report.put("establish_ms", serde_json::json!(ms(t.elapsed())));
    log(format!("session established in {:.0} ms", ms(t.elapsed())));
    report.save();
    let relay_cfg = Arc::new(tls::client_config(&identity, tls::ServerPin::new()).expect("cfg"));
    let every = limits.refill_every + Duration::from_millis(200);

    // ── 1. attacks: without the relay, C3 past the bucket, nominal rate ──
    {
        let d = probe::direct(&endpoint, &host).await;
        log(format!(
            "direct, no relay: {d:?}: {}",
            if d.all_refused() {
                "all refused"
            } else {
                "ANSWERED (bad)"
            }
        ));
        report.put(
            "direct_without_relay",
            serde_json::json!({ "cases": d, "all_refused": d.all_refused() }),
        );
        let n = 3 * usize::try_from(limits.burst).unwrap();
        let b = probe::quote_burst(relay_cfg.clone(), &endpoint, &host, nonce, n).await;
        let v = b.verdict(&limits);
        log(format!(
            "C3 burst: {}",
            match &v {
                Ok(s) => format!("held, {s}"),
                Err(s) => s.clone(),
            }
        ));
        report.put(
            "c3_burst",
            serde_json::json!({
                "n": n, "bound": b.bound(&limits), "tally": b.tally(), "window_ms": ms(b.window),
                "held": v.is_ok(), "verdict": match &v { Ok(s) | Err(s) => s },
                "answers": answers(&b.answers),
            }),
        );
        tokio::time::sleep(every).await;
        let nominal =
            probe::quote_nominal(relay_cfg.clone(), &endpoint, &host, nonce, 3, every).await;
        let served = nominal
            .iter()
            .all(|a| matches!(a, Ok(a) if a.status == 200));
        log(format!(
            "C3 nominal, one per {:.1} s: {:?}",
            every.as_secs_f64(),
            answers(&nominal)
        ));
        report.put(
            "c3_nominal",
            serde_json::json!({ "every_s": every.as_secs_f64(), "all_200": served, "answers": answers(&nominal) }),
        );
        report.save();
    }

    // ── 2. quote time inside the TD ──────────────────────────────────────
    {
        tokio::time::sleep(every).await;
        let (inside, round_trip) =
            probe::quote_times(relay_cfg.clone(), &endpoint, &host, nonce, 5, every).await;
        log(format!(
            "session quote inside the TD: {} ; round trip here: {}",
            stats(&inside),
            stats(&round_trip)
        ));
        report.put("quote_inside_td_ms", stats(&inside));
        report.put("quote_round_trip_ms", stats(&round_trip));
        report.save();
    }

    // ── 3. key, ready, completions, cost per token ───────────────────────
    let client = Arc::new(client);
    let t = Instant::now();
    let key = zeroize::Zeroizing::new(urandom32());
    if let Err(e) = client.post_key(&key[..]).await {
        log(format!("post_key: {e}"));
        report.save();
        std::process::exit(70);
    }
    drop(key);
    loop {
        match client.ready().await {
            Ok(s) if s.ready => break,
            Ok(s) if s.state == "failed" => {
                log(format!("TD failed to load: {:?} {:?}", s.code, s.detail));
                report.save();
                std::process::exit(72);
            }
            _ => tokio::time::sleep(Duration::from_secs(2)).await,
        }
        if t.elapsed() > Duration::from_secs(600) {
            log("not ready after 600 s");
            report.save();
            std::process::exit(72);
        }
    }
    log(format!(
        "key released, model ready after {:.1} s",
        t.elapsed().as_secs_f64()
    ));
    report.put(
        "key_to_ready_s",
        serde_json::json!(t.elapsed().as_secs_f64()),
    );
    let long = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Explain in detail how a TLS 1.3 handshake works, step by step."}],
        "max_tokens": 256
    });
    for parallel in [1usize, 2] {
        let tp = probe::chats(client.clone(), long.clone(), parallel).await;
        let usd = usd_per_hour.and_then(|p| tp.usd_per_million_tokens(p));
        log(format!(
            "{parallel} at once: {:?}, {} tokens in {:.1} s = {:.1} tokens/s{}",
            tp.statuses,
            tp.completion_tokens,
            tp.secs,
            tp.tokens_per_s(),
            usd.map(|u| format!(
                ", {u:.2} $ per million at {:.2} $/h",
                usd_per_hour.unwrap_or(0.0)
            ))
            .unwrap_or_default()
        ));
        report.put(
            &format!("chats_{parallel}_at_once"),
            serde_json::json!({
                "throughput": tp, "tokens_per_s": tp.tokens_per_s(), "usd_per_hour": usd_per_hour,
                "usd_per_million_completion_tokens": usd,
            }),
        );
        report.save();
    }

    // ── 4. rechecks of 2026-10-14: C1 cross replay, one quote per connection, C2
    {
        let mut sessions: Vec<(attest_v2::SessionDoc, [u8; 32])> = vec![];
        let mut second = vec![];
        for _ in 0..2 {
            tokio::time::sleep(every).await;
            let Ok(mut raw) = probe::open(relay_cfg.clone(), &endpoint, &host, &nonce).await else {
                continue;
            };
            let Ok(a) = probe::request(&mut raw, "GET", probe::SESSION, &endpoint, None).await
            else {
                continue;
            };
            if a.status != 200 {
                continue;
            }
            if let Ok(sdoc) = serde_json::from_slice::<attest_v2::SessionDoc>(&a.body) {
                sessions.push((sdoc, raw.exporter));
            }
            if let Ok(a2) = probe::request(&mut raw, "GET", probe::SESSION, &endpoint, None).await {
                second.push(a2.status);
            }
        }
        report.put(
            "second_session_quote_same_connection_status",
            serde_json::json!(second),
        );
        if sessions.len() == 2 {
            let s = match check_raw.verify(&sessions[0].0, &sessions[1].1) {
                Ok(()) => "ACCEPTED (bad)".to_string(),
                Err(x) => format!("refused {}: {}", x.code, x.reason),
            };
            log(format!(
                "C1 cross-connection replay: {s}; second quote on a connection: {second:?}"
            ));
            report.put("c1_cross_connection_replay", serde_json::json!(s));
        }
        let mut cfg = tls::client_config(&identity, tls::ServerPin::new()).expect("cfg");
        cfg.resumption = rustls::client::Resumption::in_memory_sessions(16);
        let cfg = Arc::new(cfg);
        let mut kinds = vec![];
        for _ in 0..3 {
            match probe::open(cfg.clone(), &endpoint, &host, &nonce).await {
                Ok(mut r) => {
                    let _ = probe::request(&mut r, "GET", "/v2/attestation", &endpoint, None).await;
                    kinds.push(if r.resumed { "resumed" } else { "full" }.to_string());
                }
                Err(e) => kinds.push(format!("error: {e}")),
            }
        }
        log(format!(
            "C2 resumption asked by the client, handshakes: {kinds:?}"
        ));
        report.put(
            "c2_handshakes_with_resumption_asked",
            serde_json::json!(kinds),
        );
        report.save();
    }
    log("done");
}

/// The published-file form of the measurements, read from the quote, and
/// the pinned comparison when `--measurements` is given.
fn pins(doc: &attest_v2::Document, measurements_path: Option<&Path>, report: &mut Report) {
    let wrapped = serde_json::json!({ "evidence": { "tdx_quote_b64": doc.tdx.quote_b64 } });
    match measurements::render(
        &wrapped,
        &["rtmr1", "rtmr2"],
        Some("x0ne-cvm (probe run)"),
        None,
    ) {
        Ok(json) => {
            std::fs::write(report.path.with_file_name("measurements.json"), &json).ok();
            report.put(
                "measurements_rendered",
                serde_json::from_str(&json).unwrap(),
            );
        }
        Err(e) => log(format!("measurements render: {e}")),
    }
    if let Some(p) = measurements_path {
        let published = measurements::load(p).expect("measurements file");
        let quote = {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD
                .decode(&doc.tdx.quote_b64)
                .unwrap()
        };
        let measured = x0ne_infer::verify::tdx_measurements(&quote).expect("registers");
        let mut all = true;
        for pin in &published.pins {
            let got = measured
                .iter()
                .find(|m| m.register == pin.register)
                .map(|m| m.value_hex.clone());
            let ok = got.as_deref() == Some(pin.value_hex.as_str());
            all &= ok;
            log(format!(
                "pin {} from {}: {}",
                pin.register,
                p.display(),
                if ok { "MATCH" } else { "MISMATCH" }
            ));
        }
        report.put(
            "published_pin_match",
            serde_json::json!(all && !published.pins.is_empty()),
        );
    }
}
