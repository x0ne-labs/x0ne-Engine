//! `x0ne-engine cloud`: the lifecycle half of a confidential cloud session.
//! It holds the cloud token and never sees a question; the attested relay
//! verifies the VM on this machine and serves the local API. See docs/cloud.md.

/// `eprint!` that cannot panic. After a hangup (the terminal closed), stderr
/// fails with EIO and `eprint!` panics: the deletion that follows must run.
#[macro_export]
macro_rules! say {
    ($($t:tt)*) => {
        $crate::say_str(&format!($($t)*))
    };
}

/// `eprintln!` that cannot panic; see [`say!`].
#[macro_export]
macro_rules! sayln {
    () => {
        $crate::say_str("\n")
    };
    ($($t:tt)*) => {
        $crate::say_str(&format!("{}\n", format_args!($($t)*)))
    };
}

/// Writes to stderr and ignores a failure (a closed terminal).
pub fn say_str(s: &str) {
    use std::io::Write as _;
    let _ = std::io::stderr().write_all(s.as_bytes());
}

pub mod argv;
pub mod catalog;
pub mod chat;
pub mod cli;
pub mod color;
pub mod http;
pub mod image_import;
pub mod myaddr;
pub mod nras_key;
pub mod provider;
pub mod proxy_link;
pub mod quote;
pub mod serial;
pub mod session;
pub mod sweep;
pub mod td_console;
pub mod token;
pub mod up;

use std::time::Duration;

/// Every name this tool creates starts with it; `sweep` touches nothing else.
pub const NAME_PREFIX: &str = "x0ne-engine-";
/// Label key on every labelled resource this tool creates.
pub const LABEL_KEY: &str = "x0ne-engine";
pub const DEFAULT_MODEL: &str = "qwen3:0.6b";
pub const DEFAULT_LISTEN: &str = "127.0.0.1:8478";
pub const MACHINE_TYPE: &str = "a3-highgpu-1g";
/// Image name used when no reference names one: none given, none installed.
pub const DEFAULT_IMAGE: &str = "x0ne-cvm-0-2-1";
/// Where the relay takes its reference without `--reference`: x0ne-infer
/// `reference::DEFAULT_DIR` (`tests/source_sweep.rs` keeps them equal).
pub const REFERENCE_DIR: &str = "/usr/share/x0ne-engine/reference";
/// The relay's TCB floor: x0ne-infer `TcbPolicy::MIN_TCB_EVALUATION_DATA_NUMBER`
/// and `MAX_COLLATERAL_AGE_SECS` (x0ne-infer `tests/tcb_policy_claims.rs` keeps
/// them equal). `--min-tcb-evaluation` and `--max-collateral-age` may only be
/// stricter.
pub const TCB_EVALUATION_FLOOR: u64 = 22;
pub const MAX_COLLATERAL_AGE_SECS: u64 = 86400;
/// Test builds: the installed references `up` reads. Never reaches the relay.
pub const TEST_REFERENCE_DIR_ENV: &str = "X0NE_ENGINE_CLOUD_TEST_REFERENCE_DIR";

/// [`REFERENCE_DIR`], or [`TEST_REFERENCE_DIR_ENV`], refused unless the API
/// endpoint is the loopback, as the test timings are.
pub fn reference_dir(loopback: bool) -> Result<std::path::PathBuf, Failure> {
    let name = TEST_REFERENCE_DIR_ENV;
    match std::env::var_os(name) {
        None => Ok(REFERENCE_DIR.into()),
        Some(_) if !loopback => Err(Failure::Usage(format!(
            "{name} is a test setting: refused unless X0NE_ENGINE_GCP_ENDPOINT is the loopback"
        ))),
        Some(dir) => Ok(dir.into()),
    }
}
/// TDX + H100 zones, tried in this order when no --zone is given.
pub const TDX_H100_ZONES: [&str; 3] = ["europe-west4-c", "us-central1-a", "us-east5-a"];
pub const DEFAULT_MAX_RUN_SECS: u64 = 60 * 60;
pub const MAX_RUN_CAP_SECS: u64 = 8 * 60 * 60;
pub const MIN_RUN_SECS: u64 = 10 * 60;

/// The relay's exit when the model owner's key policy refuses the TD or the
/// session, or its files are unusable (x0ne-infer `model_key::EXIT_POLICY`).
pub const KEY_POLICY_REFUSED: u8 = 102;

/// The relay's exit codes `up` keeps as its own: the verdict's refusals,
/// 64-87, and [`KEY_POLICY_REFUSED`]. Any other is a relay failure, 95.
pub fn relay_refusal(code: i64) -> Option<u8> {
    u8::try_from(code)
        .ok()
        .filter(|c| (64..=87).contains(c) || *c == KEY_POLICY_REFUSED)
}

/// Why a command stopped. Each variant is one exit code (docs/cloud.md).
#[derive(Debug, Clone, PartialEq)]
pub enum Failure {
    /// 2: bad arguments; nothing was contacted.
    Usage(String),
    /// 90: quote not accepted (declined, above --max-cost, no price).
    QuoteRefused(String),
    /// 91: preflight (token, quota, live session, image, local setup).
    Preflight(String),
    /// 92: the provider's API refused or failed.
    Api(String),
    /// 93: the VM was lost (spot preemption) or never became reachable.
    VmLost(String),
    /// 94: deletion could not be confirmed. Critical: money may be running.
    DeleteUnconfirmed(String),
    /// 64..=87 or 102: the relay's own refusal code ([`relay_refusal`]).
    Relay(u8, String),
    /// 95: the relay crashed or broke the relay protocol.
    RelayFailed(String),
    /// 130: stopped by a signal or `down` before the session served.
    Interrupted(String),
    /// 96: `run`'s chat could not start or failed; the VM is deleted.
    Chat(String),
    /// 97: x0ne-infer in the VM refused to serve, a refusal its restarts
    /// will not change (serial console, `td_console`); the VM is deleted.
    TdRefused(String),
}

impl Failure {
    pub fn code(&self) -> u8 {
        match self {
            Failure::Usage(_) => 2,
            Failure::QuoteRefused(_) => 90,
            Failure::Preflight(_) => 91,
            Failure::Api(_) => 92,
            Failure::VmLost(_) => 93,
            Failure::DeleteUnconfirmed(_) => 94,
            Failure::Relay(c, _) => *c,
            Failure::RelayFailed(_) => 95,
            Failure::Interrupted(_) => 130,
            Failure::Chat(_) => 96,
            Failure::TdRefused(_) => 97,
        }
    }

    /// The same failure with its text passed through `f` (a redaction).
    #[must_use]
    pub fn map_message(self, f: impl FnOnce(&str) -> String) -> Failure {
        match self {
            Failure::Usage(m) => Failure::Usage(f(&m)),
            Failure::QuoteRefused(m) => Failure::QuoteRefused(f(&m)),
            Failure::Preflight(m) => Failure::Preflight(f(&m)),
            Failure::Api(m) => Failure::Api(f(&m)),
            Failure::VmLost(m) => Failure::VmLost(f(&m)),
            Failure::DeleteUnconfirmed(m) => Failure::DeleteUnconfirmed(f(&m)),
            Failure::Relay(c, m) => Failure::Relay(c, f(&m)),
            Failure::RelayFailed(m) => Failure::RelayFailed(f(&m)),
            Failure::Interrupted(m) => Failure::Interrupted(f(&m)),
            Failure::Chat(m) => Failure::Chat(f(&m)),
            Failure::TdRefused(m) => Failure::TdRefused(f(&m)),
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Failure::Usage(m)
            | Failure::QuoteRefused(m)
            | Failure::Preflight(m)
            | Failure::Api(m)
            | Failure::VmLost(m)
            | Failure::DeleteUnconfirmed(m)
            | Failure::Relay(_, m)
            | Failure::RelayFailed(m)
            | Failure::Interrupted(m)
            | Failure::Chat(m)
            | Failure::TdRefused(m) => m,
        }
    }
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

/// Timings. Production values unless the API endpoint is the loopback (tests).
#[derive(Debug, Clone)]
pub struct Knobs {
    /// Between two instance reads while the VM boots.
    pub boot_poll: Duration,
    /// Between two instance reads while serving (spot preemption).
    pub serve_poll: Duration,
    /// Between two serial console reads while waiting for the verdict.
    pub console_poll: Duration,
    /// From creation to a verdict; past it the VM is deleted.
    pub boot_timeout: Duration,
    /// After a delete answered 404 for an insert that may still land.
    pub settle: Duration,
    /// How long the relay gets to answer `stop`.
    pub stop_grace: Duration,
    /// Upper bound for one provider operation (insert, delete).
    pub op_timeout: Duration,
    /// Session time `up` must leave after the boot timeout ([`MIN_SERVING_SECS`]).
    pub min_serving: Duration,
    /// Kept between the end of a session and the expiry of a token `up`
    /// cannot renew ([`TOKEN_MARGIN_SECS`]).
    pub token_margin: Duration,
    /// The name server asked for this machine's public address ([`myaddr::SERVER`]).
    pub myaddr_server: String,
    /// Upper bound for one request to the provider's API, answer included.
    pub http_timeout: Duration,
    /// Test builds: the step of `up` that panics (`creating`, `booting`,
    /// `verifying`, `serving`), to show that its VM is deleted all the same.
    #[cfg(feature = "test-endpoints")]
    pub panic_at: Option<String>,
}

impl Knobs {
    /// Production timings for a model of `model_bytes`: see [`model_boot_allowance`].
    pub fn production_for(model_bytes: u64) -> Self {
        let mut k = Knobs::production();
        k.boot_timeout += model_boot_allowance(model_bytes);
        k
    }

    pub fn production() -> Self {
        Knobs {
            boot_poll: Duration::from_secs(5),
            serve_poll: Duration::from_secs(30),
            console_poll: Duration::from_secs(15),
            boot_timeout: Duration::from_mins(12),
            settle: Duration::from_secs(5),
            stop_grace: Duration::from_secs(10),
            op_timeout: Duration::from_mins(5),
            min_serving: Duration::from_secs(MIN_SERVING_SECS),
            token_margin: Duration::from_secs(TOKEN_MARGIN_SECS),
            myaddr_server: myaddr::SERVER.into(),
            http_timeout: Duration::from_mins(5),
            #[cfg(feature = "test-endpoints")]
            panic_at: None,
        }
    }

    /// Test timings come from `X0NE_ENGINE_CLOUD_TEST_*`, and only when the
    /// API endpoint is the loopback; anywhere else they are refused.
    pub fn from_env(loopback: bool, model_bytes: u64) -> Result<Self, Failure> {
        let mut k = Knobs::production_for(model_bytes);
        let ms = |name: &str| -> Result<Option<Duration>, Failure> {
            let Some(v) = std::env::var_os(name) else {
                return Ok(None);
            };
            if !loopback {
                return Err(Failure::Usage(format!(
                    "{name} is a test setting: refused unless X0NE_ENGINE_GCP_ENDPOINT is the loopback"
                )));
            }
            let n: u64 = v
                .to_str()
                .and_then(|s| s.parse().ok())
                .filter(|n| *n > 0 && *n <= 3_600_000)
                .ok_or_else(|| Failure::Usage(format!("{name}: milliseconds expected")))?;
            Ok(Some(Duration::from_millis(n)))
        };
        if let Some(p) = ms("X0NE_ENGINE_CLOUD_TEST_POLL_MS")? {
            k.boot_poll = p;
            k.serve_poll = p;
            k.console_poll = p;
            k.settle = p;
            k.stop_grace = Duration::from_secs(2);
            k.op_timeout = Duration::from_secs(20);
        }
        if let Some(b) = ms("X0NE_ENGINE_CLOUD_TEST_BOOT_TIMEOUT_MS")? {
            k.boot_timeout = b;
        }
        if let Some(t) = ms("X0NE_ENGINE_CLOUD_TEST_OP_TIMEOUT_MS")? {
            k.op_timeout = t;
        }
        if let Some(t) = ms("X0NE_ENGINE_CLOUD_TEST_HTTP_TIMEOUT_MS")? {
            k.http_timeout = t;
        }
        if let Some(m) = ms("X0NE_ENGINE_CLOUD_TEST_MIN_SERVING_MS")? {
            k.min_serving = m;
        }
        if let Some(m) = ms("X0NE_ENGINE_CLOUD_TEST_TOKEN_MARGIN_MS")? {
            k.token_margin = m;
        }
        let name = "X0NE_ENGINE_CLOUD_TEST_MYADDR_SERVER";
        if let Some(v) = std::env::var_os(name) {
            if !loopback {
                return Err(Failure::Usage(format!(
                    "{name} is a test setting: refused unless X0NE_ENGINE_GCP_ENDPOINT is the loopback"
                )));
            }
            k.myaddr_server = v
                .to_str()
                .filter(|s| {
                    s.parse::<std::net::SocketAddr>()
                        .is_ok_and(|a| a.ip().is_loopback())
                })
                .ok_or_else(|| Failure::Usage(format!("{name}: a loopback ip:port expected")))?
                .to_string();
        }
        #[cfg(feature = "test-endpoints")]
        {
            let name = "X0NE_ENGINE_CLOUD_TEST_PANIC_AT";
            if let Some(v) = std::env::var_os(name) {
                if !loopback {
                    return Err(Failure::Usage(format!(
                        "{name} is a test setting: refused unless X0NE_ENGINE_GCP_ENDPOINT is the loopback"
                    )));
                }
                k.panic_at = Some(v.to_string_lossy().into_owned());
            }
        }
        Ok(k)
    }
}

/// Bytes of a catalog model's files, from the catalog this build embeds (the
/// relay's), parsed by [`catalog`]. `None` for a name it does not list.
pub fn catalog_model_bytes(model: &str) -> Option<u64> {
    catalog::model_bytes(model)
}

/// A TD's work on a model before it answers: the download, `pull`'s sha256 of each
/// file, the served digest. 40 MB/s overall, not measured on a TD, set when a shard
/// merge came on top. qwen3:32b, 65.5 GB: 27 min on top of a small model's 12.
pub const MODEL_INTAKE_BYTES_PER_SEC: u64 = 40_000_000;

/// Boot time a model of `bytes` adds; see [`MODEL_INTAKE_BYTES_PER_SEC`].
pub fn model_boot_allowance(bytes: u64) -> Duration {
    Duration::from_secs(bytes.div_ceil(MODEL_INTAKE_BYTES_PER_SEC))
}

/// Session time `--max-run` must leave after the boot timeout.
pub const MIN_SERVING_SECS: u64 = 5 * 60;

/// What `up` keeps between the end of a session and the expiry of a token it
/// cannot renew (`--auth paste`, `stdin`), to delete the VM with that token.
/// 5 min: about four times one `down`, measured once (about 69 s to delete the
/// VM, 4 s its firewall rule); Google's Java client stops using a token 3 min
/// before expiry (`OAuth2Credentials.DEFAULT_EXPIRATION_MARGIN`).
pub const TOKEN_MARGIN_SECS: u64 = 5 * 60;

/// The session `up` asks for: `--max-run`, or less when a token it cannot
/// renew expires sooner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunLimit {
    /// `--max-run`, or what the token leaves of it.
    pub secs: u64,
    /// The expiry of a token `up` cannot renew, unix seconds.
    pub token_expires_unix: Option<u64>,
    /// The latest end of the session: that expiry less the margin.
    pub end_by_unix: Option<u64>,
}

impl RunLimit {
    /// The session's end for a start at `start`.
    pub fn end(&self, start: u64) -> u64 {
        let e = start + self.secs;
        self.end_by_unix.map_or(e, |b| e.min(b))
    }
}

/// `--max-run` as `up` keeps it at `now`. A token gcloud renews leaves it as
/// asked. Any other must expire [`Knobs::token_margin`] after the session's
/// end, so that `up` deletes the VM with it: `--max-run` is cut to fit, and a
/// token that leaves less than the boot timeout plus [`Knobs::min_serving`],
/// or whose expiry is unknown, is refused (91).
pub fn run_limit(
    max_run_secs: u64,
    refreshable: bool,
    token_expires_unix: Option<u64>,
    knobs: &Knobs,
    model: &str,
    now: u64,
) -> Result<RunLimit, Failure> {
    if refreshable {
        return Ok(RunLimit {
            secs: max_run_secs,
            token_expires_unix: None,
            end_by_unix: None,
        });
    }
    let Some(expires) = token_expires_unix else {
        return Err(Failure::Preflight(
            "Google's tokeninfo gave no expiry for this token, and up cannot renew it: it \
             cannot end the session while the token is still valid, to delete the VM with it. \
             Use --auth gcloud, which renews it; nothing was created"
                .into(),
        ));
    };
    let margin = knobs.token_margin.as_secs();
    let end_by = expires.saturating_sub(margin);
    let left = end_by.saturating_sub(now);
    let boot = knobs.boot_timeout.as_secs();
    let serving = knobs.min_serving.as_secs();
    if left < boot + serving {
        return Err(Failure::Preflight(format!(
            "the token expires at {}, in {}, and up cannot renew it: the session would end {} \
             before, to delete the VM with it, which leaves {}, and {model} needs {} (up to {} \
             to boot and load, then {} of serving). Paste a token with at least {} left, or \
             use --auth gcloud, which renews it; nothing was created",
            hhmm_utc(expires),
            mins_down(expires.saturating_sub(now)),
            mins_down(margin),
            mins_down(left),
            mins_up(boot + serving),
            mins_up(boot),
            mins_up(serving),
            mins_up(boot + serving + margin),
        )));
    }
    Ok(RunLimit {
        secs: max_run_secs.min(left),
        token_expires_unix: Some(expires),
        end_by_unix: Some(end_by),
    })
}

/// `54 min` for 3299 s, rounded down; `8 s` under a minute (the test timings).
pub fn mins_down(s: u64) -> String {
    if s < 60 {
        format!("{s} s")
    } else {
        format!("{} min", s / 60)
    }
}

/// `55 min` for 3299 s, rounded up; `8 s` under a minute.
pub fn mins_up(s: u64) -> String {
    if s < 60 {
        format!("{s} s")
    } else {
        format!("{} min", s.div_ceil(60))
    }
}

/// `--max-run` must outlast the boot: the VM deletes itself at `max_run`,
/// and a VM gone before the model loaded is paid for nothing. Exit 2.
pub fn check_max_run(
    max_run_secs: u64,
    boot_timeout: Duration,
    model: &str,
) -> Result<(), Failure> {
    let need = boot_timeout.as_secs() + MIN_SERVING_SECS;
    if max_run_secs < need {
        return Err(Failure::Usage(format!(
            "--max-run {} min: {model} may take up to {} min to boot and load; \
             ask for at least {} min",
            max_run_secs / 60,
            boot_timeout.as_secs().div_ceil(60),
            need.div_ceil(60)
        )));
    }
    Ok(())
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `n` random bytes from the kernel, as lowercase hex.
pub fn random_hex(n: usize) -> std::io::Result<String> {
    use std::io::Read;
    let mut b = vec![0u8; n];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut b)?;
    Ok(hex(&b))
}

pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Text from a server, a VM or the relay, made safe for a terminal: control
/// characters (escape sequences included) become `?`, length is capped.
pub fn sanitize(s: &str, max: usize) -> String {
    let mut out: String = s
        .chars()
        .take(max)
        .map(|c| if c.is_control() && c != '\n' { '?' } else { c })
        .collect();
    if s.chars().count() > max {
        out.push('…');
    }
    out
}

/// (year, month, day, hour, minute, second) in UTC.
pub fn utc_parts(unix: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (unix / 86_400) as i64;
    let rem = unix % 86_400;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (
        y,
        m,
        d,
        (rem / 3600) as u32,
        ((rem % 3600) / 60) as u32,
        (rem % 60) as u32,
    )
}

/// `22:05 UTC`
pub fn hhmm_utc(unix: u64) -> String {
    let (_, _, _, h, m, _) = utc_parts(unix);
    format!("{h:02}:{m:02} UTC")
}

/// `2026-10-12 21:05 UTC`
pub fn datetime_utc(unix: u64) -> String {
    let (y, mo, d, h, mi, _) = utc_parts(unix);
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02} UTC")
}

/// `2026-10-12T21-05-09Z`, for file names.
pub fn stamp_utc(unix: u64) -> String {
    let (y, mo, d, h, mi, s) = utc_parts(unix);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}-{mi:02}-{s:02}Z")
}

/// `41 s`, `3 min 12 s`
pub fn human_secs(s: u64) -> String {
    if s < 60 {
        format!("{s} s")
    } else {
        format!("{} min {} s", s / 60, s % 60)
    }
}

/// GCP project id: 6-30 chars, lowercase letter first, no trailing hyphen.
pub fn valid_project(p: &str) -> bool {
    let b = p.as_bytes();
    (6..=30).contains(&b.len())
        && b[0].is_ascii_lowercase()
        && *b.last().unwrap() != b'-'
        && b.iter()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'-')
}

/// `us-central1-a`
pub fn valid_zone(z: &str) -> bool {
    let parts: Vec<&str> = z.split('-').collect();
    parts.len() == 3
        && parts[0].len() >= 2
        && parts[0].bytes().all(|c| c.is_ascii_lowercase())
        && parts[1].len() >= 2
        && parts[1]
            .bytes()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        && parts[1].bytes().last().is_some_and(|c| c.is_ascii_digit())
        && parts[2].len() == 1
        && parts[2].bytes().all(|c| c.is_ascii_lowercase())
}

/// `us-central1-a` → `us-central1`
pub fn region_of(zone: &str) -> &str {
    zone.rsplit_once('-').map(|(r, _)| r).unwrap_or(zone)
}

/// Compute resource name: `[a-z]([-a-z0-9]{0,61}[a-z0-9])?`.
pub fn valid_resource_name(n: &str) -> bool {
    let b = n.as_bytes();
    !b.is_empty()
        && b.len() <= 63
        && b[0].is_ascii_lowercase()
        && b[b.len() - 1] != b'-'
        && b.iter()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'-')
}

/// Catalog model name `family:tag`.
pub fn valid_model(m: &str) -> bool {
    let ok = |s: &str| {
        !s.is_empty()
            && s.len() <= 64
            && s.bytes().all(|c| {
                c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, b'.' | b'_' | b'-')
            })
    };
    matches!(m.split_once(':'), Some((f, t)) if ok(f) && ok(t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_formatting_matches_known_instants() {
        assert_eq!(datetime_utc(0), "1970-01-01 00:00 UTC");
        assert_eq!(datetime_utc(1_760_303_109), "2025-10-12 21:05 UTC");
        assert_eq!(stamp_utc(1_791_839_109), "2026-10-12T21-05-09Z");
        assert_eq!(hhmm_utc(1_791_839_109), "21:05 UTC");
        // Leap day.
        assert_eq!(datetime_utc(1_709_164_800), "2024-02-29 00:00 UTC");
    }

    #[test]
    fn sanitize_removes_escape_sequences() {
        assert_eq!(sanitize("ok\x1b[2Jboom", 100), "ok?[2Jboom");
        assert_eq!(sanitize("abcdef", 3), "abc…");
        assert_eq!(sanitize("line\nnext", 100), "line\nnext");
    }

    #[test]
    fn names_are_validated_before_they_reach_a_url() {
        assert!(valid_project("demo-project-123456"));
        assert!(!valid_project("cc/../x"));
        assert!(!valid_project("Abcdef"));
        assert!(!valid_project("abcde-"));
        assert!(valid_zone("us-central1-a"));
        assert!(valid_zone("europe-west4-c"));
        assert!(!valid_zone("us-central1"));
        assert!(!valid_zone("us-central1-a/../b"));
        assert_eq!(region_of("us-central1-a"), "us-central1");
        assert!(valid_resource_name("x0ne-engine-7f3a9c"));
        assert!(!valid_resource_name("x0ne-engine-"));
        assert!(valid_model("qwen3:0.6b"));
        assert!(!valid_model("qwen3"));
        assert!(!valid_model("../x:y"));
    }

    /// qwen3:32b (61 GiB, catalog of 2026-10-16) on a TD: the 12 min of a
    /// small model are not enough to download and hash it; the default model
    /// keeps about 12 min. Sizes are the catalog's own.
    #[test]
    fn the_boot_timeout_grows_with_the_models_bytes() {
        let small =
            catalog_model_bytes(DEFAULT_MODEL).expect("the default model is in the catalog");
        assert_eq!(
            small, 1_514_735_051,
            "qwen3:0.6b: config, tokenizer, weights, licence"
        );
        assert_eq!(catalog_model_bytes("qwen3:1.7b"), Some(4_074_975_920));
        assert_eq!(catalog_model_bytes("nope:1b"), None);
        let b = Knobs::production_for(small).boot_timeout;
        assert!(
            b >= Duration::from_mins(12) && b <= Duration::from_mins(13),
            "{b:?}"
        );
        let b = Knobs::production_for(65_535_821_615).boot_timeout;
        assert!(b >= Duration::from_mins(39), "qwen3:32b: {b:?}");
    }

    /// A VM that deletes itself before the model could load is refused before it exists.
    #[test]
    fn a_max_run_shorter_than_the_boot_is_refused() {
        let boot = Knobs::production_for(65_535_821_615).boot_timeout;
        let e = check_max_run(40 * 60, boot, "qwen3:32b").unwrap_err();
        assert_eq!(e.code(), 2, "{e:?}");
        assert!(e.message().contains("at least"), "{e:?}");
        let boot = Knobs::production_for(1_514_735_051).boot_timeout;
        assert!(check_max_run(40 * 60, boot, DEFAULT_MODEL).is_ok());
    }

    /// 2026-10-22, 00:00 UTC.
    const T: u64 = 1_792_627_200;

    #[test]
    fn a_token_up_cannot_renew_cuts_the_run_to_its_expiry_less_the_margin() {
        let k = Knobs::production_for(1_514_735_051);
        // 59 min left: 54 min of session, which ends 5 min before the expiry.
        let l = run_limit(3600, false, Some(T + 3540), &k, DEFAULT_MODEL, T).unwrap();
        assert_eq!(
            l,
            RunLimit {
                secs: 3540 - 300,
                token_expires_unix: Some(T + 3540),
                end_by_unix: Some(T + 3240),
            }
        );
        assert_eq!(l.end(T), T + 3240);
        // A session that starts later (the "yes") still ends 5 min before.
        assert_eq!(l.end(T + 7), T + 3240);
        // A shorter --max-run is kept, and bounded all the same.
        let l = run_limit(1800, false, Some(T + 3540), &k, DEFAULT_MODEL, T).unwrap();
        assert_eq!((l.secs, l.end_by_unix), (1800, Some(T + 3240)));
        assert_eq!(l.end(T + 1500), T + 3240);
    }

    #[test]
    fn a_token_gcloud_renews_leaves_max_run_as_asked() {
        let k = Knobs::production_for(1_514_735_051);
        let l = run_limit(8 * 3600, true, Some(T + 600), &k, DEFAULT_MODEL, T).unwrap();
        assert_eq!(
            l,
            RunLimit {
                secs: 8 * 3600,
                token_expires_unix: None,
                end_by_unix: None,
            }
        );
        assert!(run_limit(3600, true, None, &k, DEFAULT_MODEL, T).is_ok());
    }

    /// qwen3:32b: 39.3 min of boot timeout, 5 min of serving, 5 min to delete.
    #[test]
    fn a_token_too_short_for_boot_serving_and_deletion_is_refused() {
        let k = Knobs::production_for(65_535_821_615);
        let e = run_limit(3600, false, Some(T + 49 * 60), &k, "qwen3:32b", T).unwrap_err();
        assert_eq!(e.code(), 91, "{e:?}");
        assert_eq!(
            e.message(),
            "the token expires at 00:49 UTC, in 49 min, and up cannot renew it: the session \
             would end 5 min before, to delete the VM with it, which leaves 44 min, and \
             qwen3:32b needs 45 min (up to 40 min to boot and load, then 5 min of serving). \
             Paste a token with at least 50 min left, or use --auth gcloud, which renews it; \
             nothing was created"
        );
        let l = run_limit(3600, false, Some(T + 50 * 60), &k, "qwen3:32b", T).unwrap();
        assert_eq!(l.secs, 45 * 60);
    }

    #[test]
    fn a_token_whose_expiry_google_did_not_give_is_refused_unless_renewed() {
        let k = Knobs::production_for(1_514_735_051);
        let e = run_limit(3600, false, None, &k, DEFAULT_MODEL, T).unwrap_err();
        assert_eq!(e.code(), 91, "{e:?}");
        assert!(
            e.message().contains("gave no expiry for this token")
                && e.message().contains("--auth gcloud")
                && e.message().ends_with("nothing was created"),
            "{e:?}"
        );
    }

    #[test]
    fn the_margin_outlasts_the_measured_down() {
        // 78 s on 2026-10-22, from the end of the session to the deletion confirmed.
        assert!(TOKEN_MARGIN_SECS >= 2 * 78);
        let k = Knobs::production();
        assert_eq!(k.token_margin, Duration::from_secs(TOKEN_MARGIN_SECS));
        assert_eq!(k.min_serving, Duration::from_secs(MIN_SERVING_SECS));
    }

    #[test]
    fn test_knobs_are_refused_off_the_loopback() {
        // Only the absence case is deterministic here (env is process-wide);
        // the refusal is covered end to end in tests/fake_gcp.rs.
        assert!(Knobs::from_env(true, 0).is_ok());
    }
}
