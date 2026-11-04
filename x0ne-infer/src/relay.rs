//! The relay. `x0ne-engine-proxy` is this module plus argv parsing:
//! the lifecycle (`x0ne-engine cloud`) drives it over stdin/stdout, one JSON object per
//! line; tests drive it over any pair of streams.
//!
//! ```text
//! proxy → {"v":1,"event":"config","config_b64":…,"config_sha256":…,"image":{"gcp_image":…,"disk_tar_sha256":…},
//!          "tcb_policy":{"mode":"transparent"|"strict","min_tcb_evaluation":N,"max_collateral_age_secs":N}}
//! cycle → {"v":1,"cmd":"connect","endpoint":"host:port","boot_deadline_unix":N}
//! proxy → {"v":1,"event":"progress","stage":"connecting|verifying|releasing|loading","detail":…}  (0..n)
//! proxy → {"v":1,"event":"verified","verdict":{…},"limitations":[…]}
//!      or {"v":1,"event":"refused","code":N,"reason":…,"evidence_file":…}   then exit N
//! proxy → {"v":1,"event":"serving","listen":…,"key_file":…,"context_tokens":N|null,
//!          "prefill_chunk":N|null,"prefill_chunks":[[tokens,chunk],…]|null}
//! cycle → {"v":1,"cmd":"stop"}        proxy → {"v":1,"event":"stopped"}     exit 0
//! ```
//!
//! Nothing but `GET /v2/attestation` reaches the TD before `verified`, and
//! the key is the first thing after it: `POST /v2/key`, or for sealed
//! weights the recipient quote and `POST /v2/model-key`. `stop` is honoured at every
//! step. An unexpected end of the control stream stops everything, exit 2.
//!
//! With the model owner's key (`Env::model_key`) the config names the sealed
//! weights, the owner's policy judges the verified TD (exit 102 if not), and
//! the key goes wrapped to the X25519 key a fresh recipient quote names
//! (`model_key.rs`): `GET /v2/model-key/recipient`, then `POST /v2/model-key`.

use crate::attest_v2::{b64, Document};
use crate::cloud_config::{hex, CloudConfig, Loaded};
use crate::local_api::{self, LocalState, Session};
use crate::local_chat;
use crate::remote_verify::{
    self, Anchors, Expectations, Pinned, Refusal, SnpExpectations, TdClient,
};
use crate::term_color;
use crate::tls::TlsIdentity;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc;
use zeroize::Zeroizing;

pub const PROTOCOL: u64 = 1;
pub const DEFAULT_TD_PORT: u16 = 443;
/// The control stream is a program's, not a human's: a protocol error,
/// including its unexpected end, exits with the shell's usage code.
pub const EXIT_PROTOCOL: i32 = 2;
/// The local endpoint could not be opened (key file, bind).
pub const EXIT_LOCAL_ENDPOINT: i32 = 98;

pub struct Args {
    pub provider: String,
    pub machine_type: String,
    pub model: String,
    pub expires_at_unix: u64,
    /// The local endpoint, loopback only (`local_api::parse_listen`).
    pub listen: String,
    pub attest: String,
    pub td_port: u16,
    pub max_completion_tokens: usize,
}

/// What the relay runs with; production values in `src/bin/proxy.rs`.
pub struct Env {
    pub anchors: Anchors,
    /// The image reference: Intel TDX or AMD SEV-SNP. The session verifies
    /// that TEE's evidence only.
    pub reference: Pinned,
    pub runtime_dir: PathBuf,
    pub state_dir: PathBuf,
    pub clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    pub random: Arc<dyn Fn() -> [u8; 32] + Send + Sync>,
    pub poll_interval: Duration,
    pub load_timeout: Duration,
    pub version: String,
    /// Colors on standard error (`term_color`); presentation only, decided by the caller.
    pub color: bool,
    /// `--quiet`: the progress, the verdict and the refusals go to the events
    /// only, which the caller prints (`x0ne-engine cloud`). Other lines stay.
    pub quiet: bool,
    /// The model owner's key and release policy (`--model-key`): the session
    /// serves the owner's sealed weights, and the key goes to the TD only
    /// after the verdict and the policy. `None`: a catalog model, public.
    pub model_key: Option<Arc<crate::model_key::OwnerKey>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cmd {
    Connect {
        endpoint: String,
        boot_deadline_unix: u64,
    },
    Stop,
    Eof,
    Bad(String),
}

pub fn parse_cmd(line: &str) -> Cmd {
    let v: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return Cmd::Bad(format!("not JSON: {e}")),
    };
    if v["v"].as_u64() != Some(PROTOCOL) {
        return Cmd::Bad(format!(
            "protocol {} (this relay speaks {PROTOCOL})",
            v["v"]
        ));
    }
    match v["cmd"].as_str() {
        Some("connect") => {
            let endpoint = v["endpoint"].as_str().unwrap_or("").to_string();
            let Some(boot_deadline_unix) = v["boot_deadline_unix"].as_u64() else {
                return Cmd::Bad("connect without boot_deadline_unix".into());
            };
            if endpoint.is_empty() {
                return Cmd::Bad("connect without endpoint".into());
            }
            Cmd::Connect {
                endpoint,
                boot_deadline_unix,
            }
        }
        Some("stop") => Cmd::Stop,
        other => Cmd::Bad(format!("unknown cmd {other:?}")),
    }
}

/// The control stream, and how standard error is written (`Env::color`, `Env::quiet`).
struct Out<'a, W> {
    w: &'a mut W,
    color: bool,
    quiet: bool,
}

impl<W: AsyncWrite + Unpin> Out<'_, W> {
    async fn emit(&mut self, mut v: serde_json::Value) -> Result<(), i32> {
        v["v"] = json!(PROTOCOL);
        let line = format!("{v}\n");
        self.w
            .write_all(line.as_bytes())
            .await
            .map_err(|_| EXIT_PROTOCOL)?;
        self.w.flush().await.map_err(|_| EXIT_PROTOCOL)
    }

    /// A line that is also an event: on standard error unless quiet. Text
    /// from the TD is made printable by the caller, before any color.
    fn say(&self, line: &str) {
        if !self.quiet {
            eprintln!("{line}");
        }
    }

    async fn progress(&mut self, stage: &str, detail: String) -> Result<(), i32> {
        let detail = local_chat::printable(&detail);
        self.say(&format!("x0ne-engine-proxy: {stage}: {detail}"));
        self.emit(json!({ "event": "progress", "stage": stage, "detail": detail }))
            .await
    }

    /// The refusal event, then the code to exit with.
    async fn refused(&mut self, code: i32, reason: String, evidence_file: Option<&Path>) -> i32 {
        // Reasons quote the TD (keys, error bodies): no escape sequence passes.
        let reason = local_chat::printable(&reason);
        self.say(&term_color::paint(
            self.color,
            term_color::Style::Fail,
            &format!("x0ne-engine-proxy: REFUSED ({code}): {reason}"),
        ));
        let _ = self
            .emit(json!({ "event": "refused", "code": code, "reason": reason, "evidence_file": evidence_file }))
            .await;
        code
    }

    async fn stopped(&mut self) -> i32 {
        let _ = self.emit(json!({ "event": "stopped" })).await;
        0
    }
}

/// One step of a loop that also listens to the control stream.
enum Step<T> {
    Got(T),
    Retry(String),
    Ignore,
    Stop,
    Eof,
}

fn control(cmd: Option<Cmd>) -> Step<()> {
    match cmd {
        Some(Cmd::Stop) => Step::Stop,
        None | Some(Cmd::Eof) => Step::Eof,
        Some(Cmd::Bad(why)) => {
            eprintln!("x0ne-engine-proxy: control: {why}, ignored");
            Step::Ignore
        }
        Some(Cmd::Connect { .. }) => {
            eprintln!("x0ne-engine-proxy: control: a second connect, ignored");
            Step::Ignore
        }
    }
}

/// Run one session over `input`/`output`; the process exit code.
pub async fn run<R, W>(args: Args, env: Env, input: R, output: &mut W) -> i32
where
    R: AsyncBufRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin,
{
    let (tx, rx) = mpsc::channel::<Cmd>(8);
    let reader = tokio::spawn(async move {
        let mut lines = input.lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    if line.trim().is_empty() {
                        continue;
                    }
                    if tx.send(parse_cmd(&line)).await.is_err() {
                        return;
                    }
                }
                Ok(None) | Err(_) => {
                    let _ = tx.send(Cmd::Eof).await;
                    return;
                }
            }
        }
    });
    let mut out = Out {
        w: output,
        color: env.color,
        quiet: env.quiet,
    };
    let code = match session(&args, &env, rx, &mut out).await {
        Ok(code) | Err(code) => code,
    };
    reader.abort();
    code
}

/// Judged before anything is sent anywhere.
fn prepare(args: &Args, env: &Env) -> Result<(TlsIdentity, Loaded), (i32, String)> {
    local_api::parse_listen(&args.listen).map_err(|e| (EXIT_PROTOCOL, e))?;
    let now = (env.clock)();
    if args.expires_at_unix <= now + local_api::EXPIRY_GUARD_SECS {
        return Err((
            EXIT_PROTOCOL,
            format!(
                "--expires-at-unix {} is not in the future (now {now})",
                args.expires_at_unix
            ),
        ));
    }
    if env.reference.provider() != args.provider {
        return Err((
            89,
            format!(
                "the image reference is for provider {:?}, this session is {:?}",
                env.reference.provider(),
                args.provider
            ),
        ));
    }
    // The owner's release is written for a TDX quote (`model_key.rs`): an
    // SEV-SNP guest gets no model key, refused before anything is sent.
    if env.model_key.is_some() && matches!(env.reference, Pinned::Snp(_)) {
        return Err((
            89,
            "--model-key: sealed weights are released to Intel TDX guests only, and the image reference is AMD SEV-SNP".into(),
        ));
    }
    let sealed = match &env.model_key {
        None => None,
        Some(owner) => Some(sealed_weights(&owner.policy, args, now)?),
    };
    let tls = TlsIdentity::generate("x0ne-engine-proxy")
        .map_err(|e| (97, format!("TLS identity: {e:#}")))?;
    let mut config = CloudConfig::new(
        &args.model,
        tls.spki_sha256(),
        (env.random)(),
        args.expires_at_unix,
        &args.attest,
        args.td_port,
        args.max_completion_tokens,
    );
    if let Some(s) = sealed {
        config = config.with_sealed(s);
    }
    let config = config
        .to_loaded()
        .map_err(|e| (EXIT_PROTOCOL, format!("config: {e}")))?;
    Ok((tls, config))
}

/// What the config says of the owner's sealed weights, from the policy;
/// nothing is created for a policy that cannot release.
fn sealed_weights(
    policy: &crate::model_key::ReleasePolicy,
    args: &Args,
    now: u64,
) -> Result<crate::cloud_config::SealedWeights, (i32, String)> {
    let no = |why: String| {
        (
            crate::model_key::EXIT_POLICY,
            format!("{why}; nothing was created"),
        )
    };
    if policy.model != args.model {
        return Err(no(format!(
            "the model key policy is for base model {:?}, this session asks for {:?}",
            policy.model, args.model
        )));
    }
    if now >= policy.not_after_unix {
        return Err(no(format!(
            "the model owner's key policy expired at {} (unix)",
            policy.not_after_unix
        )));
    }
    Ok(crate::cloud_config::SealedWeights {
        sealed_sha256: policy.sealed_sha256.clone(),
        digest_hex: policy.served_digest().map_err(no)?,
        source: policy.sealed_source.clone(),
    })
}

/// The key, once the verdict and (for sealed weights) the owner's policy
/// said yes: a random session key, or the owner's key wrapped to the TD.
/// The verdict on `doc` under the pinned reference, and the check every
/// connection's attestation will pass: TDX or SEV-SNP, never the other.
#[allow(clippy::too_many_arguments)]
async fn verify_pinned(
    doc: &Document,
    config: &Loaded,
    tls_spki: [u8; 32],
    machine_type: &str,
    entry: &crate::registry::Entry,
    env: &Env,
    http: &reqwest::Client,
    now: u64,
) -> Result<(remote_verify::Report, remote_verify::SessionCheck), Refusal> {
    match &env.reference {
        Pinned::Tdx(reference) => {
            let exp = Expectations {
                config,
                tls_spki_sha256: tls_spki,
                machine_type,
                reference,
                catalog_entry: entry,
            };
            let report = remote_verify::verify_document(doc, &exp, &env.anchors, http, now).await?;
            let check =
                remote_verify::SessionCheck::new(doc, &exp, &env.anchors, env.clock.clone())?;
            Ok((report, check))
        }
        Pinned::Snp(reference) => {
            let exp = SnpExpectations {
                config,
                tls_spki_sha256: tls_spki,
                reference,
                catalog_entry: entry,
            };
            let (report, crl) =
                remote_verify::verify_snp_document(doc, &exp, &env.anchors, http, now).await?;
            let check =
                remote_verify::SessionCheck::snp(doc, &exp, &env.anchors, &crl, env.clock.clone())?;
            Ok((report, check))
        }
    }
}

async fn send_key(client: &TdClient, env: &Env) -> Result<(), Refusal> {
    let refused = |code: i32, reason: String| Refusal { code, reason };
    let Some(owner) = &env.model_key else {
        let key = Zeroizing::new((env.random)());
        return client
            .post_key(&key[..])
            .await
            .map_err(|e| refused(70, format!("the TD refused the session key: {e}")));
    };
    let binding = client.model_key_recipient(&(env.random)()).await?;
    let grant = owner
        .grant()
        .map_err(|e| refused(crate::model_key::EXIT_POLICY, e))?;
    let wrapped = owner
        .wrap(&binding, &grant, &(env.random)())
        .map_err(|e| refused(70, e))?;
    client
        .post_model_key(&wrapped)
        .await
        .map_err(|e| refused(70, format!("the TD refused the model key: {e}")))
}

async fn session<W: AsyncWrite + Unpin>(
    args: &Args,
    env: &Env,
    mut rx: mpsc::Receiver<Cmd>,
    out: &mut Out<'_, W>,
) -> Result<i32, i32> {
    let (tls, config) = match prepare(args, env) {
        Ok(p) => p,
        Err((code, reason)) => return Ok(out.refused(code, reason, None).await),
    };
    let catalog = crate::registry::Catalog::embedded();
    let entry = catalog.get(&config.config.model).map_err(|e| {
        eprintln!("x0ne-engine-proxy: {e}");
        EXIT_PROTOCOL
    })?;
    // The TD would exit 96 on it after the VM is paid for. Sealed weights
    // carry their own digest, from the owner's policy.
    if let Some(why) = entry.cloud_refusal().filter(|_| !config.config.is_sealed()) {
        return Ok(out
            .refused(96, format!("{why}; nothing was created"), None)
            .await);
    }
    // The image of the reference verified above: the lifecycle refuses to
    // boot another, before anything is created. The TCB policy every verdict
    // will apply: the lifecycle refuses a looser one than it asked for.
    let tcb = &env.anchors.tcb_policy;
    let event = match &env.reference {
        Pinned::Tdx(r) => json!({
            "event": "config",
            "config_b64": b64(&config.bytes),
            "config_sha256": hex(&config.sha256),
            "image": {"gcp_image": r.reference.image.gcp_image, "disk_tar_sha256": r.reference.image.disk_tar_sha256},
            "tcb_policy": {
                "mode": tcb.mode.as_str(),
                "min_tcb_evaluation": tcb.min_tcb_evaluation_data_number,
                "max_collateral_age_secs": tcb.max_collateral_age_secs,
            },
        }),
        // No GCP image: the GCP lifecycle refuses this config event.
        Pinned::Snp(r) => json!({
            "event": "config",
            "config_b64": b64(&config.bytes),
            "config_sha256": hex(&config.sha256),
            "image": {
                "tee": crate::snp_reference::TEE,
                "name": r.reference.image.name,
                "version": r.reference.image.version,
                "launch_measurements_sha384": r.reference.launch.measurements_sha384,
            },
            "snp_policy": {
                "products": r.reference.products,
                "tcb_minimum": r.reference.tcb_minimum,
                "guest_policy": r.reference.guest_policy,
                "vlek_csp_ids": r.reference.signing_keys.vlek_csp_ids,
                "tcb_mode": tcb.mode.as_str(),
            },
        }),
    };
    out.emit(event).await?;

    // ── connect ──────────────────────────────────────────────────────────
    let (endpoint, boot_deadline) = loop {
        match control_or_connect(rx.recv().await) {
            Ok(c) => break c,
            Err(Step::Stop) => return Ok(out.stopped().await),
            Err(Step::Eof) => return Err(EXIT_PROTOCOL),
            Err(_) => {}
        }
    };
    let client = match TdClient::new(&tls, &endpoint, config.config.nonce()) {
        Ok(c) => c,
        Err(e) => {
            return Ok(out
                .refused(88, format!("endpoint {endpoint:?}: {e}"), None)
                .await)
        }
    };
    out.progress(
        "connecting",
        format!("waiting for the TD at {endpoint} (boot deadline {boot_deadline})"),
    )
    .await?;
    let started_at_unix = (env.clock)();
    let mut last_error = String::from("not tried yet");
    let doc: Document = loop {
        if (env.clock)() > boot_deadline {
            return Ok(out
                .refused(
                    88,
                    format!("the TD did not answer GET /v2/attestation before the boot deadline ({boot_deadline}); last error: {last_error}"),
                    None,
                )
                .await);
        }
        let step = tokio::select! {
            cmd = rx.recv() => control(cmd).map(|_| unreachable!()),
            r = client.attestation() => match r {
                Ok(d) => Step::Got(d),
                Err(e) => Step::Retry(e),
            },
        };
        match step {
            Step::Got(d) => break d,
            Step::Retry(e) => {
                last_error = e;
                tokio::time::sleep(env.poll_interval).await;
            }
            Step::Ignore => {}
            Step::Stop => return Ok(out.stopped().await),
            Step::Eof => return Err(EXIT_PROTOCOL),
        }
    };

    // ── verify, on this machine ──────────────────────────────────────────
    let Some(tls_spki) = client.pinned_spki() else {
        return Ok(out
            .refused(
                88,
                "no TLS server key was pinned on the connection".into(),
                None,
            )
            .await);
    };
    out.progress(
        "verifying",
        "attestation document received; checking it on this machine, no question and no session key sent yet"
            .into(),
    )
    .await?;
    let http = match reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(60))
        .build()
    {
        Ok(h) => h,
        Err(e) => return Ok(out.refused(88, format!("HTTP client: {e}"), None).await),
    };
    let now = (env.clock)();
    let verdict = loop {
        let step = tokio::select! {
            cmd = rx.recv() => control(cmd).map(|_| unreachable!()),
            r = verify_pinned(&doc, &config, tls_spki, &args.machine_type, entry, env, &http, now) => Step::Got(r),
        };
        match step {
            Step::Got(r) => break r,
            Step::Ignore => {}
            Step::Stop => return Ok(out.stopped().await),
            Step::Eof => return Err(EXIT_PROTOCOL),
            Step::Retry(_) => unreachable!(),
        }
    };
    // The session quote of the connection the document came on: the TLS
    // session itself is the TD's, not a relay's holding its key.
    let verdict = match verdict {
        Ok((report, check)) => client.establish(check).await.map(|()| report),
        Err(refusal) => Err(refusal),
    };
    // The model owner's own yes, on what the verdict established.
    let verdict = match (verdict, &env.model_key) {
        (Ok(report), Some(owner)) => {
            crate::model_key::Observed::from_verified(&doc, &report, &config.config)
                .map_err(|e| Refusal {
                    code: 65,
                    reason: format!("verified document: {e}"),
                })
                .and_then(|seen| owner.policy.judge(&seen, (env.clock)()))
                .map(|()| report)
        }
        (other, _) => other,
    };
    let report = match verdict {
        Ok(report) => report,
        Err(refusal) => {
            let file = save_evidence(&env.state_dir, now, &doc, &refusal, &config);
            return Ok(out
                .refused(refusal.code, refusal.reason, file.as_deref())
                .await);
        }
    };
    for line in &report.summary {
        out.say(&format!(
            "x0ne-engine-proxy:   {}",
            term_color::verdict_line(env.color, &local_chat::printable(line))
        ));
    }
    out.emit(json!({
        "event": "verified",
        "verdict": serde_json::to_value(&report).unwrap_or_default(),
        "limitations": report.limitations,
    }))
    .await?;

    // ── release the key, wait for the model ──────────────────────────────
    out.progress(
        "releasing",
        if env.model_key.is_some() {
            "verdict VERIFIED, the owner's policy agrees; sending the model key, wrapped to \
             the key the TD's fresh quote names"
                .into()
        } else {
            "verdict VERIFIED; sending the session key over the attested channel".into()
        },
    )
    .await?;
    let step = tokio::select! {
        cmd = rx.recv() => control(cmd).map(|_| unreachable!()),
        r = send_key(&client, env) => Step::Got(r),
    };
    match step {
        Step::Got(Ok(())) | Step::Ignore => {}
        Step::Got(Err(r)) => return Ok(out.refused(r.code, r.reason, None).await),
        Step::Retry(_) => unreachable!("send_key retries nothing"),
        Step::Stop => return Ok(out.stopped().await),
        Step::Eof => return Err(EXIT_PROTOCOL),
    }
    out.progress(
        "loading",
        "key accepted; waiting for the TD to load the model".into(),
    )
    .await?;
    let load_started = tokio::time::Instant::now();
    let (mut context_tokens, mut prefill_chunk, mut prefill_chunks) = (None, None, None);
    let mut streaming = false;
    loop {
        if load_started.elapsed() > env.load_timeout {
            return Ok(out
                .refused(
                    72,
                    format!(
                        "the TD did not become ready within {} s",
                        env.load_timeout.as_secs()
                    ),
                    None,
                )
                .await);
        }
        let step = tokio::select! {
            cmd = rx.recv() => control(cmd).map(|_| unreachable!()),
            r = client.ready() => match r {
                Ok(s) if s.ready => {
                    context_tokens = s.context_tokens;
                    streaming = s.streaming;
                    prefill_chunk = s.prefill_chunk;
                    prefill_chunks = s.prefill_chunks;
                    Step::Got(())
                }
                Ok(s) if s.state == "failed" => Step::Retry(format!(
                    "failed:{}:{}", s.code.unwrap_or(72), s.detail.unwrap_or_default())),
                Ok(s) => Step::Retry(format!("state {}", s.state)),
                Err(e) => Step::Retry(e),
            },
        };
        match step {
            Step::Got(()) => break,
            Step::Retry(e) => {
                if let Some(rest) = e.strip_prefix("failed:") {
                    let (code, detail) = rest.split_once(':').unwrap_or(("72", rest));
                    return Ok(out
                        .refused(
                            code.parse().unwrap_or(72),
                            format!("the TD failed to load: {detail}"),
                            None,
                        )
                        .await);
                }
                tokio::time::sleep(env.poll_interval).await;
            }
            Step::Ignore => {}
            Step::Stop => return Ok(out.stopped().await),
            Step::Eof => return Err(EXIT_PROTOCOL),
        }
    }

    // ── serve on the loopback ────────────────────────────────────────────
    let bearer = Zeroizing::new((env.random)());
    let key_file = match local_api::write_key_file(&env.runtime_dir, &bearer) {
        Ok(p) => p,
        Err(e) => {
            return Ok(out
                .refused(EXIT_LOCAL_ENDPOINT, format!("key file: {e}"), None)
                .await)
        }
    };
    // The TD's sealed path is TDX only (`cloud_serve::e2e_key`).
    let sealed_path = doc.snp.is_none();
    let session = Session {
        client,
        document: Arc::new(doc),
        report,
        provider: args.provider.clone(),
        machine_type: args.machine_type.clone(),
        expires_at_unix: args.expires_at_unix,
        started_at_unix,
        context_tokens,
        streaming,
        prefill_chunk,
        prefill_chunks: prefill_chunks.clone(),
        e2e: sealed_path.then(|| {
            local_api::E2eCarrier::new(
                http.clone(),
                remote_verify::CollateralSources {
                    pcs_base_url: env.anchors.pcs_base_url.clone(),
                    root_crl_url: env.anchors.root_crl_url.clone(),
                    gce_bucket_base_url: env.anchors.gce_bucket_base_url.clone(),
                },
                crate::e2e_stream::cadence_of(&config.config),
            )
        }),
    };
    let (state, mut fatal) = LocalState::new(session, &bearer, env.clock.clone());
    drop(bearer);
    let server = match local_api::start(state, &args.listen).await {
        Ok(s) => s,
        Err(e) => return Ok(out.refused(EXIT_LOCAL_ENDPOINT, e, None).await),
    };
    out.emit(json!({
        "event": "serving",
        "listen": server.listen.to_string(),
        "key_file": key_file,
        "context_tokens": context_tokens,
        "prefill_chunk": prefill_chunk,
        "prefill_chunks": prefill_chunks,
    }))
    .await?;
    enum End {
        Stop,
        Eof,
        Fatal(i32, String),
    }
    let end = loop {
        tokio::select! {
            cmd = rx.recv() => match control(cmd) {
                Step::Stop => break End::Stop,
                Step::Eof => break End::Eof,
                _ => {}
            },
            changed = fatal.changed() => {
                if changed.is_err() {
                    break End::Eof;
                }
                if let Some((code, reason)) = fatal.borrow_and_update().clone() {
                    break End::Fatal(code, reason);
                }
            }
        }
    };
    server.stop().await;
    match end {
        End::Stop => Ok(out.stopped().await),
        End::Eof => Err(EXIT_PROTOCOL),
        End::Fatal(code, reason) => Ok(out.refused(code, reason, None).await),
    }
}

fn control_or_connect(cmd: Option<Cmd>) -> Result<(String, u64), Step<()>> {
    match cmd {
        Some(Cmd::Connect {
            endpoint,
            boot_deadline_unix,
        }) => Ok((endpoint, boot_deadline_unix)),
        other => Err(control(other)),
    }
}

impl<T> Step<T> {
    fn map<U>(self, f: impl FnOnce(T) -> U) -> Step<U> {
        match self {
            Step::Got(t) => Step::Got(f(t)),
            Step::Retry(s) => Step::Retry(s),
            Step::Ignore => Step::Ignore,
            Step::Stop => Step::Stop,
            Step::Eof => Step::Eof,
        }
    }
}

/// The refused document, kept for review: public data only (the quote, the
/// log, the tokens, the config), 0600 in a 0700 directory. `None` when it
/// could not be written; the refusal stands either way.
pub fn save_evidence(
    dir: &Path,
    now: u64,
    doc: &Document,
    refusal: &Refusal,
    config: &Loaded,
) -> Option<PathBuf> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
            .ok()?;
    }
    #[cfg(not(unix))]
    std::fs::create_dir_all(dir).ok()?;
    let body = serde_json::to_vec_pretty(&json!({
        "schema": "x0ne-engine/refused-evidence/v1",
        "at_unix": now,
        "refusal": refusal,
        "config_b64": b64(&config.bytes),
        "document": doc,
    }))
    .ok()?;
    let stamp = remote_verify::file_stamp(now);
    for n in 0..100u32 {
        let name = if n == 0 {
            format!("refused-{stamp}.json")
        } else {
            format!("refused-{stamp}-{n}.json")
        };
        let path = dir.join(name);
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        match opts.open(&path) {
            Ok(mut f) => {
                use std::io::Write as _;
                return f.write_all(&body).ok().map(|_| path);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(_) => return None,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refusal reason carries text the TD chose (an attested key, an error
    /// body): written to a terminal raw, an escape sequence in it would run.
    #[tokio::test]
    async fn a_refusal_reason_reaches_no_one_with_control_characters() {
        let mut buf: Vec<u8> = Vec::new();
        let mut out = Out {
            w: &mut buf,
            color: false,
            quiet: true,
        };
        let reason = "the TD attested TLS key \x1b]0;owned\x07\x1b[2J but".to_string();
        assert_eq!(out.refused(88, reason, None).await, 88);
        out.progress("verify", "a\x1b[31mb".into()).await.unwrap();
        for line in String::from_utf8(buf).unwrap().lines() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            for key in ["reason", "detail"] {
                if let Some(text) = v[key].as_str() {
                    assert!(!text.chars().any(char::is_control), "{key}: {text:?}");
                }
            }
        }
    }

    #[test]
    fn commands_are_parsed_strictly() {
        assert_eq!(
            parse_cmd(
                r#"{"v":1,"cmd":"connect","endpoint":"203.0.113.7:443","boot_deadline_unix":5}"#
            ),
            Cmd::Connect {
                endpoint: "203.0.113.7:443".into(),
                boot_deadline_unix: 5
            }
        );
        assert_eq!(parse_cmd(r#"{"v":1,"cmd":"stop"}"#), Cmd::Stop);
        assert!(matches!(parse_cmd(r#"{"v":2,"cmd":"stop"}"#), Cmd::Bad(_)));
        assert!(matches!(
            parse_cmd(r#"{"v":1,"cmd":"connect","endpoint":"x"}"#),
            Cmd::Bad(_)
        ));
        assert!(matches!(
            parse_cmd(r#"{"v":1,"cmd":"connect","boot_deadline_unix":1}"#),
            Cmd::Bad(_)
        ));
        assert!(matches!(parse_cmd("nope"), Cmd::Bad(_)));
        assert!(matches!(
            parse_cmd(r#"{"v":1,"cmd":"delete"}"#),
            Cmd::Bad(_)
        ));
    }

    #[test]
    fn the_evidence_file_is_private_and_never_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let config = CloudConfig::new(
            "qwen3:0.6b",
            [1; 32],
            [2; 32],
            4_000_000_000,
            "cpu",
            443,
            64,
        )
        .to_loaded()
        .unwrap();
        let doc: Document = serde_json::from_value(serde_json::json!({
            "schema": "x", "config_b64": "",
            "binding": {"context":"","session_nonce_hex":"","session_pubkey_hex":"","tls_spki_sha256_hex":"","config_sha256_hex":""},
            "tdx": {"quote_b64":"","ccel_b64":""}, "gpu": [], "nras": {"overall_token":"","jwks":{}},
            "model": {"name":"","digest_hex":"","files":[]},
            "server": {"version":"","attested_at_unix":0,"verdict":{},"live_tcb":["",""]}
        }))
        .unwrap();
        let refusal = Refusal {
            code: 65,
            reason: "test".into(),
        };
        let a = save_evidence(dir.path(), 1_760_303_100, &doc, &refusal, &config).unwrap();
        let b = save_evidence(dir.path(), 1_760_303_100, &doc, &refusal, &config).unwrap();
        assert_ne!(a, b);
        assert!(a
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("refused-2025-10-12T21-05Z"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&a).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(&a).unwrap()).unwrap();
        assert_eq!(saved["refusal"]["code"], 65);
    }
}
