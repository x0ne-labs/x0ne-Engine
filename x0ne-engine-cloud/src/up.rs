//! `quote`, `up` (and `run`, which is `up` with a chat), `status`, `down`.
//! Every exit of `up` after the creation started goes through `cleanup`,
//! which deletes the VM and its firewall rule; a failed deletion is exit 94.

use std::os::unix::process::ExitStatusExt as _;
use std::process::{ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::{mpsc, watch};
use tokio::time::sleep_until;

use crate::cli::Opts;
use crate::color::{self, paint, Style};
use crate::provider::{Instance, Preflight, Provider, SerialChunk, SessionRequest, VmState};
use crate::proxy_link::{self, Event, Proxy, ProxyArgs};
use crate::quote::Quote;
use crate::session::{self, Done, Liveness, Paths, SessionState, StopReason, STATE_SCHEMA};
use crate::td_console;
use crate::token::Tty;
use crate::{
    hhmm_utc, human_secs, mins_down, now_unix, sanitize, Failure, Knobs, RunLimit, MACHINE_TYPE,
};

/// What `status --limits` prints, before the relay's own list.
pub const LIMITS: &[&str] = &[
    "this machine: malware in your session can read the local key and your questions;",
    "Google's firmware: the binary Google signed is fetched and measured, its source is not \
     published; RTMR0 (the host's TD HOB, Secure Boot variables, ACPI tables and boot \
     variables) is pinned to the reference's events as one boot showed them, not audited;",
    "revocation: the TD's PCK is checked against Intel's PCK CRL, its PCK CA and Intel's TCB \
     signing certificate against Intel's root CA CRL, both fetched live; the OCSP responses a \
     document staples about NVIDIA's signing chain are judged under the policy the verdict \
     names, a verified revoked refused either way; no TD staples them yet;",
    "the platform's patch level: under the default transparent TCB policy the host is \
     accepted on Intel's standard TCB collateral, and what Intel's latest collateral rates \
     not fixed is named on the VERIFIED line and in each receipt, not refused; \
     `--tcb-policy strict` refuses it;",
    "the GPU verdict comes from NVIDIA's NRAS service; each connection's GPU evidence answers \
     a nonce derived from that connection's TLS exporter and TDX quote, but CPU and GPU are \
     bound by the TD's software, not by the hardware (Hopper): the link falls with the CPU \
     TEE (TEE.fail);",
    "physical attacks on memory (TEE.fail, DDR5 interposer, 2025-10) and side channels;",
    "a compromise of Intel's, NVIDIA's or Google's roots;",
    "availability: Google can stop the VM (spot); it sees times, sizes, IP addresses and the bill;",
    "the image reference: the relay takes one only with a signature by the x0ne archive key \
     pinned in its build, and a relay built before the signing ceremony pins none and \
     refuses them all; whoever holds the signing subkey can sign any image, and a \
     revocation reaches you only with a new release;",
    "the behaviour of the model itself.",
];

pub struct Ctx<'a> {
    pub provider: &'a dyn Provider,
    pub paths: &'a Paths,
    pub knobs: &'a Knobs,
    pub project: String,
    pub image: String,
    /// The reference's `image.disk_tar_sha256`, when a reference names the image.
    pub image_sha256: Option<String>,
    /// Which reference named `image`, or none: for messages.
    pub image_from: String,
    /// gcloud can mint a new token; a pasted one cannot.
    pub token_refreshable: bool,
}

/// What faces the user once the session serves.
#[derive(Debug, Clone, PartialEq)]
pub enum Front {
    /// `up`: the local API only.
    Api,
    /// `run`: the API, and a chat child whose end ends the session.
    Chat {
        /// The token came on stdin: the chat reads `/dev/tty`.
        tty_input: bool,
        max_tokens: u32,
    },
}

/// True for "yes" typed on the controlling terminal. Err: no terminal.
pub fn ask_yes(what: &str) -> Result<bool, String> {
    let Some(mut tty) = Tty::open() else {
        return Err("no terminal to answer \"yes\"".into());
    };
    let a = tty
        .read_line(&format!("Type \"yes\" to {what}: "))
        .map_err(|e| format!("no answer ({e})"))?;
    Ok(a == "yes")
}

/// `--yes`, or "yes" typed. Anything else is exit 90 and nothing is created.
pub fn confirm(yes: bool, what: &str) -> Result<(), Failure> {
    if yes {
        return Ok(());
    }
    match ask_yes(what) {
        Ok(true) => Ok(()),
        Ok(false) => Err(Failure::QuoteRefused(
            "not confirmed; nothing was created".into(),
        )),
        Err(e) => Err(Failure::QuoteRefused(format!(
            "{e} to {what}: run it in a terminal, or pass --yes (with --max-cost <USD> for up). \
             Nothing was created"
        ))),
    }
}

fn token_line(id: &crate::provider::Identity) -> String {
    let who = id
        .account
        .clone()
        .unwrap_or_else(|| "account not shown (token without the email scope)".into());
    match id.expires_in_secs {
        Some(s) => format!("Token OK: {who}, expires in {} min.", s / 60),
        None => format!("Token OK: {who}."),
    }
}

/// When a token `up` cannot renew cut `--max-run` (`asked`): to what, and why.
fn cut_line(asked: u64, l: &RunLimit, margin: u64) -> Option<String> {
    let (Some(expires), Some(end_by)) = (l.token_expires_unix, l.end_by_unix) else {
        return None;
    };
    (l.secs < asked).then(|| {
        format!(
            "--max-run cut to {} (asked: {}): the token expires at {} and up cannot renew it, \
             so the session ends by {}, {} before, to delete the VM with it.",
            mins_down(l.secs),
            mins_down(asked),
            hhmm_utc(expires),
            hhmm_utc(end_by),
            mins_down(margin)
        )
    })
}

/// On standard error: a running VM and a missing image in yellow when `on`.
fn preflight_line(p: &Preflight, on: bool) -> String {
    let live = if p.live.is_empty() {
        "No x0ne-engine VM running.".to_string()
    } else {
        paint(
            on,
            Style::Warn,
            &format!(
                "x0ne-engine VM already running: {}.",
                p.live
                    .iter()
                    .map(|l| format!("{} ({}, {})", l.name, l.location, l.detail))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )
    };
    let image = match p.image.status.as_str() {
        "READY" => format!("Image {} found.", p.image.name),
        "MISSING" => paint(
            on,
            Style::Warn,
            &format!(
                "Image {} not found in project {} (x0ne-engine cloud image import).",
                p.image.name, p.image.project
            ),
        ),
        s => paint(on, Style::Warn, &format!("Image {} is {s}.", p.image.name)),
    };
    format!("Preflight: {}. {live} {image}", p.quota)
}

fn leftover_lines(p: &Preflight) -> Option<String> {
    if p.leftovers.is_empty() {
        return None;
    }
    let mut s = paint(
        color::stderr_on(),
        Style::Warn,
        "Leftovers from earlier sessions (remove: x0ne-engine cloud sweep):",
    );
    s.push('\n');
    for l in &p.leftovers {
        s.push_str(&format!(
            "  {:<14}{:<36}{:<16}{}\n",
            l.kind.label(),
            l.name,
            l.location,
            l.detail
        ));
    }
    Some(s)
}

fn request(o: &Opts, ctx: &Ctx<'_>, session_id: String) -> SessionRequest {
    SessionRequest {
        project: ctx.project.clone(),
        zone: o.zone.clone(),
        machine_type: MACHINE_TYPE.into(),
        model: o.model.clone(),
        max_run_secs: o.max_run_secs,
        image: ctx.image.clone(),
        session_id,
        source_ranges: Vec::new(),
        nras_key: o.nras_key.clone(),
    }
}

/// The firewall rule's sources: `--allow-from`, else this machine's public
/// IPv4 address as Google's name server sees it, /32. Never everyone unasked.
async fn firewall_sources(o: &Opts, ctx: &Ctx<'_>) -> Result<Vec<String>, Failure> {
    if !o.allow_from.is_empty() {
        let line = format!(
            "Firewall: tcp:443 from {} (--allow-from).",
            o.allow_from.join(", ")
        );
        if o.allow_from.iter().any(|r| r.ends_with("/0")) {
            let warn = format!(
                "{line} Anyone on the Internet can open a connection; mutual TLS alone keeps them out."
            );
            sayln!("{}", paint(color::stderr_on(), Style::Warn, &warn));
        } else {
            sayln!("{line}");
        }
        return Ok(o.allow_from.clone());
    }
    let ip = crate::myaddr::public_ipv4(&ctx.knobs.myaddr_server, crate::myaddr::WAIT)
        .await
        .map_err(|e| {
            Failure::Preflight(format!(
                "this machine's public IPv4 address could not be found ({e}): pass \
                 --allow-from <CIDR> with the addresses your connections to the VM come \
                 from; nothing was created"
            ))
        })?;
    sayln!(
        "Firewall: tcp:443 from {ip}/32 only, this machine's public address as Google's \
         name server sees it (--allow-from <CIDR> to widen)."
    );
    Ok(vec![format!("{ip}/32")])
}

/// `quote`: token, preflight, price. Creates nothing.
pub async fn quote(o: &Opts, ctx: &Ctx<'_>) -> Result<(), Failure> {
    sayln!(
        "x0ne Engine cloud: {}, project {}",
        ctx.provider.display_name(),
        ctx.project
    );
    let id = ctx.provider.identity().await?;
    sayln!("{}", token_line(&id));
    let req = request(o, ctx, "000000".into());
    let pf = ctx.provider.preflight(&req).await?;
    sayln!("{}", preflight_line(&pf, color::stderr_on()));
    if let Some(l) = leftover_lines(&pf) {
        say!("{l}");
    }
    let q = ctx.provider.quote(&req, &pf).await?;
    print!(
        "{}",
        q.render_styled("1x H100 80GB", now_unix(), None, color::stdout_on())
    );
    if let Some(max) = o.max_cost {
        if q.max_cost() > max + 1e-9 {
            return Err(Failure::QuoteRefused(format!(
                "the maximum cost {:.2} USD is above --max-cost {max:.2}",
                q.max_cost()
            )));
        }
    }
    Ok(())
}

// ── stop requests: signals and the control socket ───────────────────────────

pub struct Stops {
    rx: mpsc::UnboundedReceiver<StopReason>,
    done: watch::Sender<Option<Done>>,
    active: Arc<AtomicUsize>,
}

/// Every signal whose default action ends the process, except SIGKILL, SIGSTOP,
/// the faults and glibc's 32 and 33 (its sigaction refuses them; 33 still ends
/// `up`). Uncaught, each ended `up` with the VM left running (rev10).
fn stop_signal_name(n: libc::c_int) -> Option<&'static str> {
    Some(match n {
        libc::SIGINT => "Ctrl-C",
        libc::SIGTERM => "SIGTERM",
        libc::SIGHUP => "SIGHUP",
        libc::SIGQUIT => "Ctrl-\\",
        libc::SIGUSR1 => "SIGUSR1",
        libc::SIGUSR2 => "SIGUSR2",
        libc::SIGALRM => "SIGALRM",
        libc::SIGVTALRM => "SIGVTALRM",
        libc::SIGPROF => "SIGPROF",
        libc::SIGIO => "SIGIO",
        libc::SIGPWR => "SIGPWR",
        libc::SIGSTKFLT => "SIGSTKFLT",
        libc::SIGXCPU => "SIGXCPU",
        libc::SIGXFSZ => "SIGXFSZ",
        n if (libc::SIGRTMIN()..=libc::SIGRTMAX()).contains(&n) => "a real-time signal",
        _ => return None,
    })
}

/// From here on, every signal `stop_signal_name` names is caught and sent to
/// `tx` by its name, instead of ending the process. `up` and `image import`.
pub(crate) fn forward_stop_signals<T: Send + 'static>(
    tx: &mpsc::UnboundedSender<T>,
    wrap: fn(&'static str) -> T,
) -> Result<(), Failure> {
    for n in 1..=libc::SIGRTMAX() {
        let Some(name) = stop_signal_name(n) else {
            continue;
        };
        let mut s = signal(SignalKind::from_raw(n))
            .map_err(|e| Failure::Preflight(format!("signals: {name}: {e}")))?;
        let tx = tx.clone();
        tokio::spawn(async move {
            while s.recv().await.is_some() {
                if tx.send(wrap(name)).is_err() {
                    break;
                }
            }
        });
    }
    Ok(())
}

impl Stops {
    /// From here on, every signal `stop_signal_name` names ends the session cleanly,
    /// and the control socket answers for session `session_id`.
    pub fn install(paths: &Paths, session_id: &str) -> Result<Stops, Failure> {
        let (tx, rx) = mpsc::unbounded_channel();
        forward_stop_signals(&tx, StopReason::Signal)?;
        let (done, done_rx) = watch::channel(None);
        let active = Arc::new(AtomicUsize::new(0));
        session::serve_control(paths, session_id, tx, done_rx, active.clone())
            .map_err(|e| Failure::Preflight(format!("control socket: {e}")))?;
        Ok(Stops { rx, done, active })
    }

    pub async fn next(&mut self) -> StopReason {
        self.rx
            .recv()
            .await
            .unwrap_or(StopReason::Signal("shutdown"))
    }

    /// Hands the result to any waiting `down`, and gives it 2 s to be written.
    async fn finish(&self, code: u8, message: &str) {
        let _ = self.done.send(Some(Done {
            code,
            message: message.to_string(),
        }));
        let until = Instant::now() + Duration::from_secs(2);
        while self.active.load(Ordering::SeqCst) > 0 && Instant::now() < until {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

fn stop_text(r: &StopReason) -> String {
    match r {
        StopReason::Signal(s) => format!("stopped by {s}"),
        StopReason::Down => "stopped by x0ne-engine cloud down".into(),
    }
}

/// A relay that ended: its own refusal code, or 95.
fn relay_ended(code: Option<i32>, when: &str) -> Failure {
    match code {
        Some(c) if (64..=87).contains(&c) => {
            Failure::Relay(c as u8, format!("the relay exited with code {c} {when}"))
        }
        Some(c) => Failure::RelayFailed(format!("the relay exited with code {c} {when}")),
        None => Failure::RelayFailed(format!("the relay was killed {when}")),
    }
}

/// After the chat's [`crate::chat::CHAT_SESSION_STOPPED`], the relay's refusal
/// is waited for this long: twice the 5 s it gives open connections first.
const REFUSAL_WAIT: Duration = Duration::from_secs(10);

/// After a `refused` event: the code the relay refused with, its exit status
/// (`--quiet` keeps it from printing it), and `up`'s, 64-87 or the event's.
async fn refusal_codes(proxy: &mut Proxy, event_code: u8) -> (i32, u8) {
    let c = proxy.exit_code().await;
    let exit = c
        .filter(|c| (64..=87).contains(c))
        .map_or(event_code, |c| c as u8);
    (c.unwrap_or(i32::from(event_code)), exit)
}

/// A `refused` event while serving: said in red, and the session's failure.
async fn relay_refused(proxy: &mut Proxy, code: u8, reason: &str, on: bool) -> Failure {
    let (said, code) = refusal_codes(proxy, code).await;
    sayln!(
        "      {}",
        paint(on, Style::Fail, &format!("REFUSED ({said}): {reason}"))
    );
    Failure::Relay(code, format!("the relay ended the session: {reason}"))
}

/// The chat exited on an answer the relay stopped the session for: the
/// relay's refusal, read within [`REFUSAL_WAIT`], ends the session with its code.
async fn refusal_after_chat(proxy: &mut Proxy, on: bool) -> Option<Failure> {
    let until = tokio::time::Instant::now() + REFUSAL_WAIT;
    loop {
        let next = tokio::time::timeout_at(until, proxy.next_event());
        match next.await.ok()? {
            Ok(Some(Event::Refused { code, reason, .. })) => {
                return Some(relay_refused(proxy, code, &reason, on).await);
            }
            Ok(Some(Event::Stopped) | None) | Err(_) => return None,
            Ok(Some(_)) => {}
        }
    }
}

/// Characters kept of one line of the relay's verdict summary. The relay writes
/// these lines itself; the longest seen, the Image line of 2026-10-31, has 321,
/// and a cap of 300 cut it before its RTMR0 clause.
const VERDICT_LINE_CHARS: usize = 1000;

/// The relay's verdict as lines: its `summary` (the relay's report), else
/// `lines`, else every scalar field. Each line whole up to
/// `VERDICT_LINE_CHARS`, control characters replaced.
pub fn verdict_lines(v: &Value) -> Vec<String> {
    if let Some(lines) = v["summary"].as_array().or_else(|| v["lines"].as_array()) {
        return lines
            .iter()
            .filter_map(Value::as_str)
            .map(|l| sanitize(l, VERDICT_LINE_CHARS))
            .collect();
    }
    let mut out = Vec::new();
    for (k, val) in v.as_object().into_iter().flatten() {
        let text = match val {
            Value::String(s) => s.clone(),
            Value::Bool(b) => b.to_string(),
            Value::Number(n) => n.to_string(),
            _ => continue,
        };
        out.push(format!(
            "{:<6}{}",
            sanitize(k, 24),
            sanitize(&text, VERDICT_LINE_CHARS)
        ));
    }
    out
}

/// The verdict's TCB disclosure line (x0ne-infer `tcb_disclosure`, since
/// 2026-10-28): the platform's TCB date and what Intel's latest collateral
/// or AMD's bulletins say against it. None from an older relay.
pub fn tcb_line(v: &Value) -> Option<String> {
    v["tcb"]["line"].as_str().map(|l| sanitize(l, 600))
}

fn print_verdict(v: &Value) {
    let on = color::stderr_on();
    let tcb = tcb_line(v);
    for l in verdict_lines(v) {
        // The summary's TCB line is said once, after VERIFIED.
        if tcb.is_some() && l.starts_with("TCB   ") {
            continue;
        }
        sayln!("      {}", color::verdict_line(on, &l));
    }
    sayln!("      {}", verified_line(v, on));
}

/// The line after the verdict: VERIFIED, then the TCB disclosure when the
/// relay sent one, then where the limits are.
pub fn verified_line(v: &Value, on: bool) -> String {
    let verified = paint(on, Style::Verified, "VERIFIED");
    match tcb_line(v) {
        Some(t) => format!(
            "{verified}. {}. Limits: x0ne-engine cloud status --limits",
            color::claims(on, &t)
        ),
        None => format!("{verified}. Limits: x0ne-engine cloud status --limits"),
    }
}

/// `run`'s one-line verdict, from the relay's report fields.
pub fn verdict_line(v: &Value, model: &str) -> String {
    let field = |k: &str| v[k].as_str().map(crate::chat::short);
    let gpu = match v["gpu_count"].as_u64() {
        Some(0) => "CPU (Intel TDX) verified, GPU NOT attested".to_string(),
        Some(n) => {
            format!("CPU (Intel TDX) and {n} GPU (NVIDIA, confidential computing on) verified")
        }
        None => "CPU and GPU: see the verdict above".to_string(),
    };
    let model = v["model_name"]
        .as_str()
        .map(|m| sanitize(m, 64))
        .unwrap_or_else(|| model.to_string());
    format!(
        "Verdict, checked on this machine: {gpu}; model {model}, digest {}; session key {}.",
        field("model_digest_hex").unwrap_or_else(|| "?".into()),
        field("session_pubkey_hex").unwrap_or_else(|| "?".into()),
    )
}

/// `verdict_line` for a terminal: its head bold, its claims as `color::claims` colors them.
pub fn verdict_line_styled(v: &Value, model: &str, on: bool) -> String {
    let line = verdict_line(v, model);
    let head = "Verdict, checked on this machine:";
    match line.strip_prefix(head) {
        Some(rest) if on => format!(
            "{}{}",
            paint(true, Style::Bold, head),
            color::claims(true, rest)
        ),
        _ => line,
    }
}

/// How a serving session ended without error.
enum Ended {
    Stopped(StopReason),
    Expired,
    /// `run`: /exit, end of input.
    FromChat,
}

struct Live<'a> {
    ctx: &'a Ctx<'a>,
    front: &'a Front,
    proxy: Proxy,
    /// `run`'s chat, once the session serves: the relay's binary, `chat`.
    chat: Option<tokio::process::Child>,
    relay_bin: std::path::PathBuf,
    inst: Instance,
    st: SessionState,
    quote: Quote,
    created: Instant,
    /// The insert may still land after a failed or interrupted creation.
    pending_insert: bool,
    /// The expiry of the token that cut `--max-run`, if one did.
    token_cut: Option<u64>,
    /// The firewall rule's sources, as the user was told before "yes".
    source_ranges: Vec<String>,
    /// `--nras-key-file`, for the insert only.
    nras_key: Option<crate::nras_key::NrasKey>,
}

/// The chat's exit, or never when there is no chat.
async fn chat_exit(chat: &mut Option<tokio::process::Child>) -> std::io::Result<ExitStatus> {
    match chat {
        Some(c) => c.wait().await,
        None => std::future::pending().await,
    }
}

/// A chat that left cleanly ends the session; one that was killed by a
/// signal of [`stop_signal_name`] is that stop; anything else is exit 96.
fn chat_ended(st: std::io::Result<ExitStatus>) -> Result<Ended, Failure> {
    let st = st.map_err(|e| Failure::Chat(format!("waiting for the chat: {e}")))?;
    if st.success() {
        return Ok(Ended::FromChat);
    }
    if let Some(name) = st.signal().and_then(stop_signal_name) {
        return Ok(Ended::Stopped(StopReason::Signal(name)));
    }
    match (st.code(), st.signal()) {
        (Some(c), _) => Err(Failure::Chat(format!("the chat exited with code {c}"))),
        (None, s) => Err(Failure::Chat(format!(
            "the chat was killed by signal {}",
            s.unwrap_or(0)
        ))),
    }
}

impl Live<'_> {
    fn save(&mut self, phase: &str) {
        self.st.phase = phase.into();
        if let Err(e) = session::write_state(self.ctx.paths, &self.st) {
            sayln!(
                "{}",
                paint(
                    color::stderr_on(),
                    Style::Warn,
                    &format!("x0ne-engine cloud: warning: session state not saved: {e}")
                )
            );
        }
    }
}

/// `up`, and `run` with [`Front::Chat`]. Before "yes" nothing is created and
/// a signal just ends the process.
pub async fn up(o: &Opts, ctx: &Ctx<'_>, front: &Front) -> Result<(), Failure> {
    crate::check_max_run(o.max_run_secs, ctx.knobs.boot_timeout, &o.model)?;
    if matches!(
        front,
        Front::Chat {
            tty_input: true,
            ..
        }
    ) && Tty::open().is_none()
    {
        return Err(Failure::Usage(
            "run --auth stdin reads the token on standard input and your messages on the \
             terminal, and there is no terminal: use --auth gcloud, or a terminal"
                .into(),
        ));
    }
    let proxy_path = proxy_link::find_proxy()?;
    // A relay without its pins refuses every session, and one with the TEST
    // pins is refused off a test build's loopback fake: said before the token.
    let test_cloud = crate::http::Endpoints::resolve()?.loopback;
    proxy_link::check_pins(&proxy_path, test_cloud).await?;
    let _lock = session::try_lock(ctx.paths)
        .map_err(|e| Failure::Preflight(format!("session lock: {e}")))?
        .ok_or_else(|| {
            Failure::Preflight(
                "a session is already running on this machine (x0ne-engine cloud status, down)"
                    .into(),
            )
        })?;
    sayln!(
        "x0ne Engine cloud: {}, project {}",
        ctx.provider.display_name(),
        ctx.project
    );
    // Read before tokeninfo: the expiry computed is never later than Google's.
    let asked = now_unix();
    let id = ctx.provider.identity().await?;
    sayln!("{}", token_line(&id));
    let token_expires = id.expires_in_secs.map(|s| asked + s);
    let limit = |now| {
        crate::run_limit(
            o.max_run_secs,
            ctx.token_refreshable,
            token_expires,
            ctx.knobs,
            &o.model,
            now,
        )
    };
    let quoted = limit(now_unix())?;
    let margin = ctx.knobs.token_margin.as_secs();
    if let Some(l) = cut_line(o.max_run_secs, &quoted, margin) {
        sayln!("{}", paint(color::stderr_on(), Style::Warn, &l));
    }
    let session_id =
        crate::random_hex(3).map_err(|e| Failure::Preflight(format!("/dev/urandom: {e}")))?;
    let mut req = request(o, ctx, session_id);
    req.max_run_secs = quoted.secs;
    let pf = ctx.provider.preflight(&req).await?;
    sayln!("{}", preflight_line(&pf, color::stderr_on()));
    if let Some(l) = leftover_lines(&pf) {
        say!("{l}");
    }
    if !pf.live.is_empty() {
        return Err(Failure::Preflight(
            "one session at a time per project: stop the running one (x0ne-engine cloud down, \
             or x0ne-engine cloud sweep)"
                .into(),
        ));
    }
    if pf.image.status != "READY" {
        return Err(Failure::Preflight(format!(
            "image {} is {} in project {}: x0ne-engine cloud image import <disk.tar.gz>",
            pf.image.name, pf.image.status, pf.image.project
        )));
    }
    crate::image_import::same_build(&pf.image, ctx.image_sha256.as_deref())?;
    req.source_ranges = firewall_sources(o, ctx).await?;
    if req.nras_key.is_some() {
        sayln!(
            "NRAS: the key of --nras-key-file is handed to the VM (metadata item x0ne-nras-key); \
             x0ne-cvm 0.2.7 and later pass it to x0ne-infer, an older image calls NRAS without it."
        );
    }
    let q = ctx.provider.quote(&req, &pf).await?;
    say!(
        "\n{}",
        q.render_styled(
            "1x H100 80GB",
            now_unix(),
            quoted.end_by_unix,
            color::stderr_on()
        )
    );
    if let Some(max) = o.max_cost {
        if q.max_cost() > max + 1e-9 {
            return Err(Failure::QuoteRefused(format!(
                "the maximum cost {:.2} USD is above --max-cost {max:.2}; nothing was created",
                q.max_cost()
            )));
        }
    }
    confirm(o.yes, "create the VM")?;
    sayln!();
    // The time spent at the prompt came off the token.
    let run = limit(now_unix())?;
    let said = cut_line(o.max_run_secs, &quoted, margin);
    if let Some(l) = cut_line(o.max_run_secs, &run, margin).filter(|l| Some(l) != said.as_ref()) {
        sayln!("{}", paint(color::stderr_on(), Style::Warn, &l));
    }

    let mut stops = Stops::install(ctx.paths, &req.session_id)?;
    let result = session(o, ctx, front, &req, &pf, q, &proxy_path, &mut stops, &run).await;
    let _ = std::fs::remove_file(ctx.paths.socket());
    match &result {
        Ok(()) => stops.finish(0, "session ended; VM deleted").await,
        Err(f) => stops.finish(f.code(), f.message()).await,
    }
    result
}

async fn session(
    o: &Opts,
    ctx: &Ctx<'_>,
    front: &Front,
    req: &SessionRequest,
    pf: &Preflight,
    quote: Quote,
    proxy_path: &std::path::Path,
    stops: &mut Stops,
    run: &RunLimit,
) -> Result<(), Failure> {
    let expires_at = run.end(now_unix());
    // The token's expiry, when it cut --max-run: said at [5/5], at the end, in `status`.
    let token_cut = run.token_expires_unix.filter(|_| run.secs < o.max_run_secs);
    let args = ProxyArgs {
        provider: ctx.provider.name().into(),
        machine_type: req.machine_type.clone(),
        model: req.model.clone(),
        expires_at_unix: expires_at,
        listen: o.listen.clone(),
        reference: o.reference.clone(),
        tcb_policy: o.tcb_policy,
        min_tcb_evaluation: o.min_tcb_evaluation,
        max_collateral_age: o.max_collateral_age,
        model_key: o.model_key.clone(),
    };
    let mut proxy = Proxy::spawn(proxy_path, &args)?;
    let config = tokio::select! {
        ev = tokio::time::timeout(Duration::from_secs(30), proxy.next_event()) => match ev {
            Ok(Ok(Some(Event::Config { bytes, sha256_hex, image, tcb_policy }))) => {
                if let Err(e) = proxy_link::check_config(
                    &bytes, &sha256_hex, &req.model, expires_at, o.model_key.is_some(),
                ) {
                    proxy.kill();
                    return Err(Failure::RelayFailed(format!("relay config refused: {e}; nothing was created")));
                }
                if let Err(e) = proxy_link::check_image(
                    image.as_ref(), &ctx.image, ctx.image_sha256.as_deref(), &ctx.image_from,
                ) {
                    proxy.kill();
                    return Err(Failure::Preflight(format!("{e}; nothing was created")));
                }
                if let Err(e) = proxy_link::check_tcb_policy(
                    tcb_policy.as_ref(), o.tcb_strict(), o.min_tcb_evaluation, o.max_collateral_age,
                ) {
                    proxy.kill();
                    return Err(Failure::RelayFailed(format!("{e}; nothing was created")));
                }
                bytes
            }
            Ok(Ok(Some(Event::Refused { code, reason, .. }))) => {
                let (said, code) = refusal_codes(&mut proxy, code).await;
                let tail = if reason.ends_with("nothing was created") { "" } else { "; nothing was created" };
                return Err(Failure::Relay(code, format!("the relay refused ({said}): {reason}{tail}")));
            }
            Ok(Ok(Some(other))) => {
                proxy.kill();
                return Err(Failure::RelayFailed(format!("the relay sent {other:?} before its config; nothing was created")));
            }
            Ok(Ok(None)) => {
                let c = proxy.exit_code().await;
                return Err(relay_ended(c, "before sending its config; nothing was created"));
            }
            Ok(Err(e)) => {
                proxy.kill();
                return Err(Failure::RelayFailed(format!("{e}; nothing was created")));
            }
            Err(_) => {
                proxy.kill();
                return Err(Failure::RelayFailed("no config from the relay within 30 s; nothing was created".into()));
            }
        },
        r = stops.next() => {
            proxy.kill();
            return Err(Failure::Interrupted(format!("{}; nothing was created", stop_text(&r))));
        }
    };

    let inst = Instance {
        project: ctx.project.clone(),
        zone: pf.zone.clone(),
        name: req.vm_name(),
        firewall: req.vm_name(),
    };
    let st = SessionState {
        schema: STATE_SCHEMA.into(),
        provider: ctx.provider.name().into(),
        instance: inst.clone(),
        session_id: req.session_id.clone(),
        machine_type: req.machine_type.clone(),
        model: req.model.clone(),
        listen: o.listen.clone(),
        key_file: None,
        pid: std::process::id(),
        phase: "creating".into(),
        created_at_unix: now_unix(),
        expires_at_unix: expires_at,
        usd_per_hour: quote.hourly(),
        max_cost_usd: quote.max_cost(),
        price_source: quote.source_line(),
        verdict: None,
        limitations: Vec::new(),
        ended_at_unix: None,
        note: token_cut.map(|t| {
            format!(
                "--max-run cut to {}: the token expires at {}",
                mins_down(run.secs),
                hhmm_utc(t)
            )
        }),
    };
    let mut live = Live {
        ctx,
        front,
        proxy,
        chat: None,
        relay_bin: proxy_path.to_path_buf(),
        inst,
        st,
        quote,
        created: Instant::now(),
        pending_insert: true,
        token_cut,
        source_ranges: req.source_ranges.clone(),
        nras_key: req.nras_key.clone(),
    };
    live.save("creating");
    let outcome = drive(&mut live, &config, stops).await;
    let cleaned = cleanup(&mut live, stops).await;
    match (outcome, cleaned) {
        (Ok(Ended::Stopped(r)), Ok(())) => {
            sayln!("Session {}.", stop_text(&r));
            Ok(())
        }
        (Ok(Ended::Expired), Ok(())) => {
            match live.token_cut {
                Some(t) => sayln!(
                    "Session ended: its time ran out, {} before the token expires ({}).",
                    mins_down(ctx.knobs.token_margin.as_secs()),
                    hhmm_utc(t)
                ),
                None => sayln!("Session ended: its time ran out."),
            }
            Ok(())
        }
        (Ok(Ended::FromChat), Ok(())) => {
            sayln!("Session ended from the chat.");
            Ok(())
        }
        // A failed deletion wins over how the session ended.
        (_, Err(e)) | (Err(e), Ok(())) => Err(e),
    }
}

/// Everything from creation to the end of serving. Never deletes: the caller does.
async fn drive(live: &mut Live<'_>, config: &[u8], stops: &mut Stops) -> Result<Ended, Failure> {
    let ctx = live.ctx;
    let knobs = ctx.knobs;
    let req_model = live.st.model.clone();
    let req = SessionRequest {
        project: live.inst.project.clone(),
        zone: Some(live.inst.zone.clone()),
        machine_type: live.st.machine_type.clone(),
        model: req_model,
        max_run_secs: live
            .st
            .expires_at_unix
            .saturating_sub(live.st.created_at_unix),
        image: ctx.image.clone(),
        session_id: live.st.session_id.clone(),
        source_ranges: live.source_ranges.clone(),
        nras_key: live.nras_key.clone(),
    };

    // [1/5] The creation is not cut short: once it returns, delete finds what it made.
    let on = color::stderr_on();
    say!(
        "{} ... ",
        paint(
            on,
            Style::Bold,
            &format!("[1/5] Creating {} in {}", live.inst.name, live.inst.zone)
        )
    );
    let t0 = Instant::now();
    live.created = t0;
    let mut stop_seen = None;
    let created = {
        let create = ctx.provider.create(&req, &live.inst, config);
        tokio::pin!(create);
        loop {
            tokio::select! {
                r = &mut create => break r,
                s = stops.next() => {
                    if stop_seen.is_none() {
                        sayln!("\n      {}", paint(on, Style::Warn, "Stopping: the VM is being created; it is deleted as soon as the creation returns."));
                    }
                    stop_seen = Some(s);
                }
            }
        }
    };
    match created {
        Ok(()) => {
            live.pending_insert = false;
            sayln!("done ({})", human_secs(t0.elapsed().as_secs()));
        }
        Err(e) => {
            sayln!("{}", paint(on, Style::Fail, "failed"));
            return Err(e);
        }
    }
    if let Some(r) = stop_seen {
        return Err(Failure::Interrupted(stop_text(&r)));
    }

    // [2/5] Boot: RUNNING with an external address, within the boot timeout.
    live.save("booting");
    say!(
        "{} ... ",
        paint(on, Style::Bold, "[2/5] Waiting for boot and GPU setup")
    );
    let boot_deadline = t0 + knobs.boot_timeout;
    let boot_deadline_unix = live.st.created_at_unix + knobs.boot_timeout.as_secs().max(1);
    let t1 = Instant::now();
    let mut errors = 0;
    let ip = loop {
        let state = tokio::select! {
            s = ctx.provider.instance_state(&live.inst) => s,
            r = stops.next() => { sayln!(); return Err(Failure::Interrupted(stop_text(&r))) }
            _ = sleep_until(boot_deadline.into()) => { sayln!(); return Err(boot_timeout(live, "boot").await) }
        };
        match state {
            Ok(VmState::Running(ip)) => break ip,
            Ok(VmState::Pending(_)) => errors = 0,
            Ok(VmState::Stopped(s)) => {
                sayln!();
                return Err(Failure::VmLost(format!(
                    "the VM stopped while booting ({s}): spot preemption"
                )));
            }
            Ok(VmState::Gone) => {
                sayln!();
                return Err(Failure::VmLost(
                    "the VM disappeared while booting (spot preemption)".into(),
                ));
            }
            Err(e) => {
                errors += 1;
                if errors >= 3 {
                    sayln!();
                    return Err(e);
                }
            }
        }
        tokio::select! {
            _ = tokio::time::sleep(knobs.boot_poll) => {}
            r = stops.next() => { sayln!(); return Err(Failure::Interrupted(stop_text(&r))) }
            _ = sleep_until(boot_deadline.into()) => { sayln!(); return Err(boot_timeout(live, "boot").await) }
        }
    };
    sayln!("done ({})", human_secs(t1.elapsed().as_secs()));

    // [3/5]-[4/5] The relay checks the proof here. The VM's metadata already
    // holds the session config (and the NRAS key, if given); no question and
    // no session key leave before the verdict.
    live.save("verifying");
    sayln!(
        "{}",
        paint(
            on,
            Style::Bold,
            "[3/5] Checking the hardware proof on THIS machine. No question and no session key have been sent yet."
        )
    );
    let endpoint = std::net::SocketAddr::new(ip, 443).to_string();
    if let Err(e) = live.proxy.connect(&endpoint, boot_deadline_unix).await {
        return Err(Failure::RelayFailed(format!("sending connect: {e}")));
    }
    let mut verified = false;
    let mut deadline = boot_deadline + Duration::from_secs(30).min(knobs.boot_timeout);
    // Nothing is waited for past the session's end: the token may expire soon after it.
    let run_end =
        Instant::now() + Duration::from_secs(live.st.expires_at_unix.saturating_sub(now_unix()));
    let end_text = hhmm_utc(live.st.expires_at_unix);
    let mut next_poll = Instant::now() + knobs.serve_poll;
    // The TD's own refusal, on the serial console: it can only end this wait.
    let console_vm = live.inst.clone();
    let mut console = td_console::Reader::default();
    let mut console_read = Box::pin(read_console(
        ctx.provider,
        &console_vm,
        Instant::now() + knobs.console_poll,
        0,
    ));
    let (listen, key_file, context_tokens) = loop {
        tokio::select! {
            ev = live.proxy.next_event() => match ev {
                Ok(Some(Event::Progress { stage, detail })) => {
                    if stage == "releasing" {
                        sayln!("{}", paint(on, Style::Bold, "[4/5] Releasing the session key to the verified VM ..."));
                    }
                    if !detail.is_empty() {
                        sayln!("      {detail}");
                    }
                }
                Ok(Some(Event::Verified { verdict, limitations })) => {
                    print_verdict(&verdict);
                    verified = true;
                    live.st.verdict = Some(verdict);
                    live.st.limitations = limitations;
                    live.save("verified");
                    deadline = Instant::now() + knobs.boot_timeout;
                }
                Ok(Some(Event::Refused { code, reason, evidence_file })) => {
                    let (said, code) = refusal_codes(&mut live.proxy, code).await;
                    sayln!("      {}", paint(on, Style::Fail, &format!("REFUSED ({said}): {reason}")));
                    sayln!("      No question and no session key were sent to the VM.");
                    if let Some(f) = evidence_file {
                        sayln!("      Evidence saved for review: {f} (public data only)");
                    }
                    return Err(Failure::Relay(code, format!("proof refused: {reason}")));
                }
                Ok(Some(Event::Serving { listen, key_file, context_tokens })) => {
                    if !verified {
                        return Err(Failure::RelayFailed(
                            "the relay announced serving without a verdict".into(),
                        ));
                    }
                    break (listen, key_file, context_tokens);
                }
                Ok(Some(Event::Stopped)) => {
                    return Err(Failure::RelayFailed("the relay stopped on its own".into()));
                }
                Ok(Some(Event::Config { .. })) => {
                    return Err(Failure::RelayFailed("the relay sent a second config".into()));
                }
                Ok(Some(Event::Other(e))) => sayln!("      {}", paint(on, Style::Dim, &format!("(relay event {e:?} ignored)"))),
                Ok(None) => {
                    let c = live.proxy.exit_code().await;
                    return Err(relay_ended(c, "during the verification"));
                }
                Err(e) => return Err(Failure::RelayFailed(e)),
            },
            r = stops.next() => return Err(Failure::Interrupted(stop_text(&r))),
            _ = sleep_until(run_end.into()) => {
                return Err(Failure::VmLost(format!(
                    "the session reached its end ({end_text}) before the model was served"
                )));
            }
            _ = sleep_until(deadline.into()) => {
                return Err(if verified {
                    Failure::VmLost(format!(
                        "the model was not served within {} min of the verdict",
                        knobs.boot_timeout.as_secs() / 60
                    ))
                } else {
                    boot_timeout(live, "verdict").await
                });
            }
            _ = sleep_until(next_poll.into()) => {
                next_poll = Instant::now() + knobs.serve_poll;
                match ctx.provider.instance_state(&live.inst).await {
                    Ok(VmState::Gone | VmState::Stopped(_)) => {
                        return Err(Failure::VmLost("Google stopped the VM (spot preemption)".into()));
                    }
                    _ => {}
                }
            }
            r = &mut console_read, if !verified && console.reading() => {
                match console.take(r) {
                    td_console::Read::Nothing => {}
                    td_console::Read::Refused(t) => return Err(td_refused(live, &t, on).await),
                    td_console::Read::GaveUp(e) => sayln!("      {}", paint(on, Style::Dim, &format!(
                        "(serial console unread {} times in a row, {e}; the wait goes on without it)",
                        td_console::MAX_FAILED_READS
                    ))),
                }
                console_read.set(read_console(
                    ctx.provider,
                    &console_vm,
                    Instant::now() + knobs.console_poll,
                    console.start(),
                ));
            }
        }
    };

    // [5/5] Serving until Ctrl-C, `down`, the end of the run, or a lost VM.
    let key = key_file
        .clone()
        .unwrap_or_else(|| "see the relay's output".into());
    live.st.listen.clone_from(&listen);
    live.st.key_file = key_file;
    live.save("serving");
    sayln!(
        "{}  (key: {key})",
        paint(
            on,
            Style::Bold,
            &format!("[5/5] OpenAI API on http://{listen}/v1")
        )
    );
    if let Some(n) = context_tokens {
        sayln!("      Context: {n} tokens, prompt and answer together.");
    }
    let left = live.st.expires_at_unix.saturating_sub(now_unix());
    let why = live.token_cut.map_or(String::new(), |t| {
        format!(
            ": the session ends {} before the token expires ({})",
            mins_down(knobs.token_margin.as_secs()),
            hhmm_utc(t)
        )
    });
    sayln!(
        "      {} Ctrl-C stops the session and deletes the VM.",
        paint(on, Style::Warn, &format!("{} min left{why}.", left / 60))
    );
    if let Front::Chat {
        tty_input,
        max_tokens,
    } = live.front
    {
        let verdict = live.st.verdict.clone().unwrap_or_default();
        sayln!("\n{}", verdict_line_styled(&verdict, &live.st.model, on));
        sayln!("The local API stays open to other clients until the session ends.");
        sayln!("/exit, Ctrl-D or Ctrl-C ends the session and deletes the VM.\n");
        let input = if *tty_input {
            std::fs::File::open("/dev/tty")
                .map(Stdio::from)
                .map_err(|e| Failure::Chat(format!("/dev/tty: {e}")))?
        } else {
            Stdio::inherit()
        };
        let key_file = live.st.key_file.clone().ok_or_else(|| {
            Failure::Chat("the relay announced no key file: the chat cannot authenticate".into())
        })?;
        live.chat = Some(crate::chat::spawn(
            &live.relay_bin,
            &listen,
            &key_file,
            *max_tokens,
            input,
        )?);
    }
    let expiry = Instant::now() + Duration::from_secs(left);
    let mut failed_polls = 0;
    let mut next_poll = Instant::now() + knobs.serve_poll;
    loop {
        tokio::select! {
            r = stops.next() => return Ok(Ended::Stopped(r)),
            _ = sleep_until(expiry.into()) => return Ok(Ended::Expired),
            st = chat_exit(&mut live.chat) => {
                if matches!(&st, Ok(s) if s.code() == Some(crate::chat::CHAT_SESSION_STOPPED)) {
                    if let Some(refused) = refusal_after_chat(&mut live.proxy, on).await {
                        return Err(refused);
                    }
                }
                return chat_ended(st);
            }
            ev = live.proxy.next_event() => match ev {
                Ok(Some(Event::Progress { detail, .. })) if !detail.is_empty() => sayln!("      {detail}"),
                Ok(Some(Event::Refused { code, reason, .. })) => {
                    return Err(relay_refused(&mut live.proxy, code, &reason, on).await);
                }
                Ok(Some(Event::Stopped) | None) => {
                    let c = live.proxy.exit_code().await;
                    return Err(relay_ended(c, "while serving"));
                }
                Ok(Some(_)) => {}
                Err(e) => return Err(Failure::RelayFailed(e)),
            },
            _ = sleep_until(next_poll.into()), if failed_polls < 3 => {
                next_poll = Instant::now() + knobs.serve_poll;
                match ctx.provider.instance_state(&live.inst).await {
                    Ok(VmState::Gone | VmState::Stopped(_)) => {
                        sayln!("      {}", paint(on, Style::Fail, "The VM is gone: Google stopped it (spot preemption)."));
                        return Err(Failure::VmLost("Google stopped the VM (spot preemption)".into()));
                    }
                    Ok(_) => failed_polls = 0,
                    Err(_) => {
                        failed_polls += 1;
                        if failed_polls == 3 {
                            sayln!(
                                "      {}",
                                paint(on, Style::Warn, "(the provider no longer answers, the token may have expired: \
                                 preemption is not watched any more; the VM still deletes itself at the end)")
                            );
                        }
                    }
                }
            }
        }
    }
}

/// One read of the serial console at `at`. Kept across the wait's loop, so
/// relay events do not cancel it; its result changes nothing until `take`.
async fn read_console(
    provider: &dyn Provider,
    inst: &Instance,
    at: Instant,
    start: u64,
) -> Result<SerialChunk, String> {
    sleep_until(at.into()).await;
    provider
        .serial_since(inst, start)
        .await
        .map_err(|e| e.to_string())
}

/// x0ne-infer in the VM exits on a refusal its restarts will not change.
async fn td_refused(live: &mut Live<'_>, t: &td_console::Refusal, on: bool) -> Failure {
    // x0ne-cvm 0.2.7 (2026-10-30) prints systemd's fields only: the code's
    // meaning stands in for x0ne-infer's line, which stays in the VM's journal.
    let reason = t.reason();
    let read = match t.line {
        Some(_) => "unverified text",
        None => "an unverified exit code, x0ne-infer's own line is not on it",
    };
    sayln!(
        "      {}",
        paint(
            on,
            Style::Fail,
            &format!(
                "REFUSED by x0ne-infer in the VM (exit {}, {} restarts): {reason}",
                t.code, t.restarts
            )
        )
    );
    if t.gave_up && !reason.starts_with("systemd ") {
        sayln!(
            "      systemd stopped restarting x0ne-infer after {} restarts (ActiveState=failed).",
            t.restarts
        );
    }
    sayln!("      Read on the VM's serial console, {read}: it only ends the wait.");
    sayln!("      No question and no session key were sent to the VM.");
    let saved = crate::serial::save_boot_log(live.ctx.provider, &live.inst, live.ctx.paths).await;
    let where_ = match saved {
        Ok(p) => format!("serial console saved to {}", p.display()),
        Err(e) => format!("serial console not saved: {e}"),
    };
    Failure::TdRefused(format!(
        "x0ne-infer in the VM refused to serve (exit {}, {} restarts); {where_}",
        t.code, t.restarts
    ))
}

/// Past the boot timeout: keep the serial console for review, then fail.
async fn boot_timeout(live: &mut Live<'_>, what: &str) -> Failure {
    let mins = live.ctx.knobs.boot_timeout.as_secs() / 60;
    let saved = crate::serial::save_boot_log(live.ctx.provider, &live.inst, live.ctx.paths).await;
    let where_ = match saved {
        Ok(p) => format!("serial console saved to {}", p.display()),
        Err(e) => format!("serial console not saved: {e}"),
    };
    Failure::VmLost(format!(
        "no {what} within {mins} min of the creation; {where_}"
    ))
}

/// Stops the relay, deletes the VM and its firewall rule, records the end.
async fn cleanup(live: &mut Live<'_>, stops: &mut Stops) -> Result<(), Failure> {
    let grace = live.ctx.knobs.stop_grace;
    // The chat first: nothing reads the terminal while the VM is deleted.
    if let Some(mut c) = live.chat.take() {
        let _ = c.start_kill();
        let _ = c.wait().await;
        sayln!();
    }
    let _ = live.proxy.stop(grace).await;
    live.save("stopping");
    let on = color::stderr_on();
    say!("Deleting {} ... ", live.inst.name);
    let pending = live.pending_insert;
    let deleted = {
        let del = live.ctx.provider.delete(&live.inst, pending);
        tokio::pin!(del);
        loop {
            tokio::select! {
                r = &mut del => break r,
                _ = stops.next() => say!("{} ", paint(on, Style::Warn, "(still deleting: Ctrl-C does not interrupt the deletion)")),
            }
        }
    };
    let used = live.created.elapsed().as_secs();
    let cost = live.quote.cost_after(used);
    live.st.ended_at_unix = Some(now_unix());
    match deleted {
        Ok(()) => {
            sayln!(
                "done. {}",
                paint(on, Style::Warn, &format!("Estimated cost: {cost:.2} USD."))
            );
            live.st.note = Some(format!("estimated cost {cost:.2} USD"));
            live.save("deleted");
            Ok(())
        }
        Err(e) => {
            sayln!("{}", paint(on, Style::Critical, "NOT CONFIRMED"));
            let by = live.st.expires_at_unix + live.ctx.knobs.boot_timeout.as_secs();
            sayln!(
                "{}",
                paint(
                    on,
                    Style::Critical,
                    &format!(
                        "CRITICAL: the deletion of {} could not be confirmed: {}",
                        live.inst.name,
                        e.message()
                    )
                )
            );
            sayln!(
                "  {}",
                paint(
                    on,
                    Style::Warn,
                    &format!(
                        "It deletes itself by {} at the latest (max {:.2} USD). Delete it now:",
                        hhmm_utc(by),
                        live.quote.max_cost()
                    )
                )
            );
            for c in live.ctx.provider.manual_delete_commands(&live.inst) {
                sayln!("    {c}");
            }
            sayln!("  or: x0ne-engine cloud down (asks for a token), then x0ne-engine cloud sweep");
            live.st.note = Some(e.message().to_string());
            live.save("delete-unconfirmed");
            Err(Failure::DeleteUnconfirmed(format!(
                "deletion of {} not confirmed: {}",
                live.inst.name,
                e.message()
            )))
        }
    }
}

// ── status and down ─────────────────────────────────────────────────────────

pub fn status(paths: &Paths, limits: bool) -> Result<(), Failure> {
    let st = session::read_state(paths)
        .map_err(|e| Failure::Preflight(format!("session state: {e}")))?;
    let on = color::stdout_on();
    match &st {
        None => println!("No x0ne-engine cloud session on this machine."),
        Some(s) => {
            let now = now_unix();
            let who = match session::liveness(paths, s) {
                Liveness::Alive => format!("up process {} alive", s.pid),
                Liveness::Silent => paint(
                    on,
                    Style::Warn,
                    "the up process does not answer: stopped (Ctrl-Z) or busy",
                ),
                Liveness::Gone if s.live() => paint(
                    on,
                    Style::Fail,
                    "the up process is GONE: run x0ne-engine cloud down",
                ),
                Liveness::Gone => "ended".to_string(),
            };
            let phase = match s.phase.as_str() {
                "delete-unconfirmed" => paint(on, Style::Critical, &s.phase),
                p => p.to_string(),
            };
            println!(
                "{}: {phase} ({who})",
                paint(
                    on,
                    Style::Bold,
                    &format!("x0ne Engine cloud session {}", s.session_id)
                )
            );
            println!(
                "  provider  {}, project {}, zone {}",
                s.provider, s.instance.project, s.instance.zone
            );
            println!(
                "  VM        {} ({} spot, Intel TDX + 1x H100)",
                s.instance.name, s.machine_type
            );
            println!("  model     {}", s.model);
            if s.phase == "serving" {
                println!(
                    "  local API http://{}/v1 (key: {})",
                    s.listen,
                    s.key_file.as_deref().unwrap_or("?")
                );
            }
            match &s.verdict {
                Some(_) => println!(
                    "  verdict   {} (by the relay, on this machine)",
                    paint(on, Style::Verified, "VERIFIED")
                ),
                None => println!("  verdict   none yet"),
            }
            let end = s.ended_at_unix.unwrap_or(now).min(s.expires_at_unix);
            let spent = (s.usd_per_hour * end.saturating_sub(s.created_at_unix) as f64 / 3600.0)
                .min(s.max_cost_usd);
            if s.live() {
                println!(
                    "  expires   {}; the VM deletes itself then",
                    paint(
                        on,
                        Style::Warn,
                        &format!(
                            "{} ({} min left)",
                            hhmm_utc(s.expires_at_unix),
                            s.expires_at_unix.saturating_sub(now) / 60
                        )
                    )
                );
            }
            println!(
                "  cost      {}",
                paint(
                    on,
                    Style::Warn,
                    &format!(
                        "{:.2} USD so far (estimate, {:.2} USD/h); at most {:.2} USD",
                        crate::quote::ceil_cents(spent),
                        s.usd_per_hour,
                        s.max_cost_usd
                    )
                )
            );
            if let Some(n) = &s.note {
                println!("  note      {n}");
            }
        }
    }
    if limits {
        println!("\n{}", paint(on, Style::Bold, "Not protected:"));
        for l in LIMITS {
            println!("- {l}");
        }
        if let Some(s) = &st {
            if !s.limitations.is_empty() {
                println!(
                    "\n{}",
                    paint(on, Style::Bold, "Reported by the relay for this session:")
                );
                for l in &s.limitations {
                    println!("- {l}");
                }
            }
        }
    }
    Ok(())
}

/// `down`: ask a live `up` to stop, or delete by name after a crash.
pub async fn down<F>(paths: &Paths, make_provider: F) -> Result<(), Failure>
where
    F: FnOnce(&str) -> Result<Box<dyn Provider>, Failure>,
{
    let read = session::read_state(paths);
    if session::is_locked(paths) {
        // The lock may be another process's: a second `up` serves its socket
        // before it writes its state, and `stop` would end that one instead.
        let live = read
            .as_ref()
            .ok()
            .and_then(Option::as_ref)
            .filter(|s| s.live());
        match live.map(|s| (s, session::liveness(paths, s))) {
            Some((s, Liveness::Silent)) => {
                return Err(Failure::Preflight(format!(
                    "the up process of session {} does not answer on its control socket: \
                     stopped (Ctrl-Z, SIGSTOP) or busy. Resume it (fg in its terminal) and run \
                     x0ne-engine cloud down again, or press Ctrl-C there",
                    s.session_id
                )))
            }
            Some((s, Liveness::Gone)) => {
                return Err(Failure::Preflight(format!(
                    "the up process of session {} is gone, and another x0ne-engine cloud process \
                     holds the session (an up being prepared, or a down). Its VM {} deletes \
                     itself by {} at the latest (max {:.2} USD). To delete it now: x0ne-engine \
                     cloud sweep --project {}",
                    s.session_id,
                    s.instance.name,
                    hhmm_utc(s.expires_at_unix + Knobs::production().boot_timeout.as_secs()),
                    s.max_cost_usd,
                    s.instance.project
                )))
            }
            _ => {}
        }
        if !paths.socket().exists() {
            return Err(Failure::Preflight(
                "another x0ne-engine cloud process is preparing a session (nothing created yet): \
                 answer or stop it in its own terminal"
                    .into(),
            ));
        }
        let d = session::request_stop(paths, Duration::from_mins(15)).await?;
        let on = color::stderr_on();
        let line = format!("x0ne-engine cloud: {}", d.message);
        match d.code {
            0 | 130 => sayln!("{line}"),
            _ => sayln!("{}", paint(on, Style::Fail, &line)),
        }
        return match d.code {
            0 | 130 => Ok(()),
            94 => Err(Failure::DeleteUnconfirmed(d.message)),
            c => {
                sayln!(
                    "{}",
                    paint(
                        on,
                        Style::Fail,
                        &format!("x0ne-engine cloud: the session ended with code {c}")
                    )
                );
                Ok(())
            }
        };
    }
    let st = read.map_err(|e| Failure::Preflight(format!("session state: {e}")))?;
    let Some(mut st) = st.filter(|s| s.live()) else {
        sayln!("x0ne-engine cloud: no session to stop.");
        return Ok(());
    };
    let _lock = session::try_lock(paths)
        .map_err(|e| Failure::Preflight(format!("session lock: {e}")))?
        .ok_or_else(|| {
            Failure::Preflight("another x0ne-engine cloud process holds the session".into())
        })?;
    let by = st.expires_at_unix + Knobs::production().boot_timeout.as_secs();
    let on = color::stderr_on();
    sayln!(
        "{}",
        paint(
            on,
            Style::Warn,
            &format!(
                "x0ne-engine cloud: the up process of session {} is gone. Its VM {} deletes itself by {} \
                 at the latest (max {:.2} USD). Deleting it now needs a token.",
                st.session_id,
                st.instance.name,
                hhmm_utc(by),
                st.max_cost_usd
            )
        )
    );
    let provider = make_provider(&st.instance.project)?;
    say!("Deleting {} ... ", st.instance.name);
    match provider.delete(&st.instance, true).await {
        Ok(()) => {
            sayln!("done.");
            st.phase = "deleted".into();
            st.ended_at_unix = Some(now_unix());
            let _ = session::write_state(paths, &st);
            Ok(())
        }
        Err(e) => {
            sayln!("{}", paint(on, Style::Critical, "NOT CONFIRMED"));
            for c in provider.manual_delete_commands(&st.instance) {
                sayln!("    {c}");
            }
            st.phase = "delete-unconfirmed".into();
            st.note = Some(e.message().to_string());
            let _ = session::write_state(paths, &st);
            Err(Failure::DeleteUnconfirmed(e.message().to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ended(raw: i32) -> String {
        match chat_ended(Ok(ExitStatus::from_raw(raw))) {
            Ok(Ended::FromChat) => "chat".into(),
            Ok(Ended::Stopped(StopReason::Signal(s))) => s.into(),
            Ok(_) => "other".into(),
            Err(f) => format!("{}: {}", f.code(), f.message()),
        }
    }

    #[test]
    fn the_chat_ends_the_session_as_its_exit_says() {
        assert_eq!(ended(0), "chat");
        assert_eq!(ended(libc::SIGINT), "Ctrl-C");
        assert_eq!(ended(libc::SIGHUP), "SIGHUP");
        assert_eq!(ended(96 << 8), "96: the chat exited with code 96");
        assert_eq!(ended(libc::SIGKILL), "96: the chat was killed by signal 9");
        assert_eq!(ended(libc::SIGUSR1), "SIGUSR1");
        assert_eq!(ended(libc::SIGRTMAX()), "a real-time signal");
        assert_eq!(ended(libc::SIGSEGV), "96: the chat was killed by signal 11");
    }

    #[test]
    fn faults_kill_and_stop_are_not_stop_signals() {
        for n in [
            libc::SIGKILL,
            libc::SIGSTOP,
            libc::SIGSEGV,
            libc::SIGBUS,
            libc::SIGILL,
            libc::SIGFPE,
            libc::SIGTRAP,
            libc::SIGSYS,
            libc::SIGABRT,
            libc::SIGCHLD,
            libc::SIGCONT,
            libc::SIGWINCH,
            libc::SIGURG,
            libc::SIGTSTP,
        ] {
            assert_eq!(stop_signal_name(n), None, "signal {n}");
        }
        // The C library's own (32 and 33 under glibc).
        for n in 32..libc::SIGRTMIN() {
            assert_eq!(stop_signal_name(n), None, "signal {n}");
        }
        for n in libc::SIGRTMIN()..=libc::SIGRTMAX() {
            assert!(stop_signal_name(n).is_some(), "signal {n}");
        }
    }

    /// docs/cloud.md names every signal that ends `up` uncaught, faults aside:
    /// SIGTRAP and SIGSYS were missing from both places (found 2026-10-17).
    #[test]
    fn the_doc_names_every_signal_that_kills_up() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/cloud.md");
        let doc = std::fs::read_to_string(&path).unwrap();
        let row = doc
            .lines()
            .find(|l| l.starts_with("| `up` killed:"))
            .expect("docs/cloud.md: the killed row");
        let para = &doc[doc
            .find("- **If `up` itself is killed** (")
            .expect("docs/cloud.md: the killed paragraph")..];
        let para = &para[..para.find(')').unwrap()];
        let faults = [libc::SIGSEGV, libc::SIGBUS, libc::SIGILL, libc::SIGFPE];
        // Ignored, stopping or continuing by default; SIGPIPE: the Rust runtime ignores it.
        let never_ends = [
            libc::SIGCHLD,
            libc::SIGCONT,
            libc::SIGWINCH,
            libc::SIGURG,
            libc::SIGSTOP,
            libc::SIGTSTP,
            libc::SIGTTIN,
            libc::SIGTTOU,
            libc::SIGPIPE,
        ];
        let kills: Vec<i32> = (1..32)
            .filter(|n| stop_signal_name(*n).is_none())
            .filter(|n| !faults.contains(n) && !never_ends.contains(n))
            .collect();
        let named = [
            (libc::SIGKILL, "SIGKILL"),
            (libc::SIGABRT, "SIGABRT"),
            (libc::SIGTRAP, "SIGTRAP"),
            (libc::SIGSYS, "SIGSYS"),
        ];
        let mut want: Vec<i32> = named.iter().map(|(n, _)| *n).collect();
        want.sort_unstable();
        assert_eq!(
            kills, want,
            "a signal ends `up` uncaught and has no name here"
        );
        for text in [row, para] {
            for (_, name) in named {
                assert!(text.contains(name), "{name} missing: {text}");
            }
            assert!(
                text.contains("signal 33") && text.contains("a fault"),
                "{text}"
            );
        }
    }

    /// 2026-10-28: the relay's TCB disclosure follows VERIFIED, said once; an
    /// older relay's verdict, without one, prints as before.
    #[test]
    fn verified_is_followed_by_the_tcb_disclosure_when_the_relay_sends_one() {
        let line = "Platform TCB 2026-02-11, OutOfDate under Intel's TCB evaluation 22; Intel \
                    advisories not fixed: INTEL-DOC-10003, INTEL-SA-01439, INTEL-SA-01442 \
                    (transparent TCB policy)";
        let v = serde_json::json!({
            "summary": ["CPU   ok", format!("TCB   {line}")],
            "tcb": { "line": line },
        });
        assert_eq!(
            verified_line(&v, false),
            format!("VERIFIED. {line}. Limits: x0ne-engine cloud status --limits")
        );
        assert_eq!(tcb_line(&v).as_deref(), Some(line));
        let old = serde_json::json!({ "summary": ["CPU   ok"] });
        assert_eq!(
            verified_line(&old, false),
            "VERIFIED. Limits: x0ne-engine cloud status --limits"
        );
        assert_eq!(tcb_line(&old), None);
        // Colored: the clause naming what is not fixed is a warning.
        assert!(verified_line(&v, true).contains("\x1b[33mIntel advisories not fixed"));
    }

    /// 2026-10-31, H100 run: the relay's Image line, 321 characters, was cut at
    /// 300 by `up`, before its RTMR0 clause; a check reading it stopped the run.
    #[test]
    fn the_relays_summary_lines_are_shown_whole() {
        let cpu = "CPU   Intel TDX quote genuine, TCB UpToDate / QE UpToDate, TDX module \
                   identity matched, PCK not on Intel's CRL, its CA and Intel's TCB signer \
                   not on Intel's root CA CRL (Intel PCS standard collateral, fetched now: \
                   TCB evaluation 20, collateral at most 86400 s old; transparent TCB \
                   policy, Intel's latest collateral on the TCB line)";
        let image = "Image x0ne-cvm 0.2.8 (reference signed 2026-10-31 by archive subkey \
                     with key ID 010D0382FD71EF0C; TEST key, not for release; logged at \
                     index 0 of test.invalid/x0ne-engine/tlog, size 1): RTMR1, RTMR2 match \
                     the reference; RTMR3 empty; RTMR0 pinned (the endorsed firmware's \
                     configuration volume, the reference's host events)";
        assert_eq!((cpu.chars().count(), image.chars().count()), (331, 321));
        let v = serde_json::json!({ "summary": [cpu, image] });
        assert_eq!(verdict_lines(&v), [cpu, image]);
        // The field-by-field fallback shows them whole too.
        let v = serde_json::json!({ "image": image });
        assert_eq!(verdict_lines(&v), [format!("image {image}")]);
        // Still sanitized: a control character is `?`, and a line a hostile
        // relay pads past the cap is cut, the ellipsis saying so.
        let v = serde_json::json!({ "summary": ["ok\x1b[31mred"] });
        assert_eq!(verdict_lines(&v), ["ok?[31mred"]);
        let long = format!("Image {}", "x".repeat(5000));
        let cut = &verdict_lines(&serde_json::json!({ "summary": [long] }))[0];
        let n = cut.chars().count();
        assert!(cut.ends_with('…') && n <= 2001, "{n}");
    }

    #[test]
    fn the_verdict_line_names_what_was_checked() {
        let v = serde_json::json!({
            "summary": ["CPU   ok", "GPU   ok"],
            "gpu_count": 1,
            "model_name": "qwen3:32b",
            "model_digest_hex": "ab".repeat(32),
            "session_pubkey_hex": "cd".repeat(32),
        });
        assert_eq!(verdict_lines(&v), ["CPU   ok", "GPU   ok"]);
        assert_eq!(
            verdict_line(&v, "x"),
            "Verdict, checked on this machine: CPU (Intel TDX) and 1 GPU (NVIDIA, \
             confidential computing on) verified; model qwen3:32b, digest abababababababab; \
             session key cdcdcdcdcdcdcdcd."
        );
        let none = verdict_line(&serde_json::json!({"gpu_count": 0}), "qwen3:8b");
        assert!(
            none.contains("GPU NOT attested; model qwen3:8b, digest ?"),
            "{none}"
        );
    }
}
