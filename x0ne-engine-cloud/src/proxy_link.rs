//! The lifecycle side of the relay protocol: the attested relay, started by absolute path and
//! spoken to in JSON lines over its stdin and stdout. It holds the client key;
//! this process holds the cloud token.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use base64::Engine as _;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use crate::Failure;

pub const PROXY_NAME: &str = "x0ne-engine-proxy";
pub const LIBEXEC_DIR: &str = "/usr/libexec/x0ne-engine";
pub const PROTOCOL: &str = "1";
/// The cloud config travels in instance metadata; it must stay small and
/// carry no secret.
pub const MAX_CONFIG: usize = 16 * 1024;
const MAX_LINE: usize = 1 << 20;

/// Next to this executable, then the libexec directory. Never `$PATH`.
pub fn proxy_candidates(exe: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(dir) = exe.parent() {
        v.push(dir.join(PROXY_NAME));
    }
    v.push(Path::new(LIBEXEC_DIR).join(PROXY_NAME));
    v
}

fn usable(p: &Path) -> bool {
    std::fs::metadata(p).is_ok_and(|m| {
        let mode = m.permissions().mode();
        m.is_file() && mode & 0o111 != 0 && mode & 0o022 == 0
    })
}

pub fn find_proxy() -> Result<PathBuf, Failure> {
    let exe = std::env::current_exe()
        .map_err(|e| Failure::Preflight(format!("cannot locate this executable: {e}")))?;
    let cands = proxy_candidates(&exe);
    cands.iter().find(|p| usable(p)).cloned().ok_or_else(|| {
        Failure::Preflight(format!(
            "{PROXY_NAME} not found at {} (never looked up in $PATH; it must not be writable by \
             group or others). Install the x0ne-engine package, or build both binaries into one directory",
            cands
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(" or ")
        ))
    })
}

/// How long `pins` may take: it reads nothing and connects nowhere.
const PINS_WAIT: Duration = Duration::from_secs(10);

/// The relay's `pins`, asked before the token: a build that pins no archive
/// key (99) or no transparency log (100) refuses every session, and a build
/// that pins the TEST anchors would accept a reference anyone can sign, so
/// `up` stops there; `test_cloud` (the API endpoint is a test build's
/// loopback fake, where nothing real is created) lets the TEST anchors
/// through. Any other answer (a relay older than `pins`, no answer) lets `up`
/// go on; the relay checks its pins again when the session starts.
pub async fn check_pins(proxy: &Path, test_cloud: bool) -> Result<(), Failure> {
    let mut cmd = Command::new(proxy);
    cmd.arg("pins")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    confine(&mut cmd);
    let Ok(mut child) = cmd.spawn() else {
        return Ok(());
    };
    let Some(out) = child.stdout.take() else {
        return Ok(());
    };
    let mut out = out.take(MAX_LINE as u64);
    let mut line = Vec::new();
    let read = tokio::time::timeout(PINS_WAIT, out.read_to_end(&mut line)).await;
    let _ = child.start_kill();
    let _ = child.wait().await;
    if !matches!(read, Ok(Ok(_))) {
        return Ok(());
    }
    let first = String::from_utf8_lossy(&line);
    let Some(v) = first
        .lines()
        .next()
        .and_then(|l| serde_json::from_str::<Value>(l).ok())
    else {
        return Ok(());
    };
    if v["v"] != json!(1) {
        return Ok(());
    }
    if v["event"] == "pins" && v["test"] == json!(true) && !test_cloud {
        return Err(Failure::Relay(
            95,
            "the relay pins the TEST archive key and the TEST transparency log, whose private \
             keys are derived from public strings: anyone can sign a reference it would accept. \
             Use a release build of x0ne-engine-proxy; no token was asked, nothing was created"
                .into(),
        ));
    }
    if v["event"] != "refused" {
        return Ok(());
    }
    match v["code"].as_u64() {
        Some(code @ (99 | 100)) => Err(Failure::Relay(
            95,
            format!(
                "the relay refused ({code}): {}; no token was asked, nothing was created",
                crate::sanitize(v["reason"].as_str().unwrap_or("no reason given"), 500)
            ),
        )),
        _ => Ok(()),
    }
}

/// The only variables a child of the lifecycle keeps: no token, no override.
/// The last four decide the colors of the relay and its chat (`color.rs`).
pub const KEPT_ENV: [&str; 14] = [
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "TZ",
    "XDG_RUNTIME_DIR",
    "XDG_STATE_HOME",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "TERM",
    "NO_COLOR",
    "CLICOLOR_FORCE",
    "CLICOLOR",
];

/// [`KEPT_ENV`] only, and SIGTERM when this process dies (relay and chat).
pub fn confine(cmd: &mut Command) {
    cmd.env_clear();
    for k in KEPT_ENV {
        if let Some(v) = std::env::var_os(k) {
            cmd.env(k, v);
        }
    }
    // SAFETY: prctl is async-signal-safe and touches no Rust state.
    unsafe {
        cmd.pre_exec(|| {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM, 0, 0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[derive(Debug, Clone)]
pub struct ProxyArgs {
    pub provider: String,
    pub machine_type: String,
    pub model: String,
    pub expires_at_unix: u64,
    pub listen: String,
    pub reference: Option<PathBuf>,
    /// `--tcb-policy`, `--min-tcb-evaluation`, `--max-collateral-age`:
    /// passed when given.
    pub tcb_policy: Option<&'static str>,
    pub min_tcb_evaluation: Option<u64>,
    pub max_collateral_age: Option<u64>,
    /// The model owner's key file and policy file, both or neither.
    pub model_key: Option<(PathBuf, PathBuf)>,
}

impl ProxyArgs {
    /// `--quiet`: `up` prints the progress, the verdict and the refusals from
    /// the events; the relay's own lines on the inherited stderr doubled them.
    pub fn argv(&self) -> Vec<String> {
        let mut v: Vec<String> = [
            "--protocol",
            PROTOCOL,
            "--provider",
            &self.provider,
            "--machine-type",
            &self.machine_type,
            "--model",
            &self.model,
            "--expires-at-unix",
            &self.expires_at_unix.to_string(),
            "--listen",
            &self.listen,
            "--quiet",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        if let Some(r) = &self.reference {
            v.push("--reference".into());
            v.push(r.display().to_string());
        }
        if let Some(m) = self.tcb_policy {
            v.push("--tcb-policy".into());
            v.push(m.into());
        }
        if let Some(n) = self.min_tcb_evaluation {
            v.push("--min-tcb-evaluation".into());
            v.push(n.to_string());
        }
        if let Some(n) = self.max_collateral_age {
            v.push("--max-collateral-age".into());
            v.push(n.to_string());
        }
        if let Some((key, policy)) = &self.model_key {
            v.push("--model-key".into());
            v.push(key.display().to_string());
            v.push("--key-policy".into());
            v.push(policy.display().to_string());
        }
        v
    }
}

/// The image of the reference the relay verified, as its config event names it.
#[derive(Debug, Clone, PartialEq)]
pub struct RelayImage {
    pub gcp_image: String,
    pub disk_tar_sha256: Option<String>,
}

/// The TCB policy the relay's verdict will apply, as its config event names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelayTcbFloor {
    /// `mode` is `strict`, or absent: a relay before 2026-10-28 knew no other.
    pub strict: bool,
    pub min_tcb_evaluation: u64,
    pub max_age_secs: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Config {
        bytes: Vec<u8>,
        sha256_hex: String,
        /// None from a relay before 0.2.0-beta.6: `up` refuses it.
        image: Option<RelayImage>,
        /// None from a relay older than this lifecycle: `up` refuses it.
        tcb_policy: Option<RelayTcbFloor>,
    },
    Progress {
        stage: String,
        detail: String,
    },
    Verified {
        verdict: Value,
        limitations: Vec<String>,
    },
    Refused {
        code: u8,
        reason: String,
        evidence_file: Option<String>,
    },
    Serving {
        listen: String,
        key_file: Option<String>,
        /// The TD's context bound; None from a relay before 0.2.0-beta.6.
        context_tokens: Option<u64>,
    },
    Stopped,
    Other(String),
}

/// Characters kept of one limitation the relay reports with its verdict. The
/// relay writes them itself; the longest seen, the `verdict verifier` line of
/// 2026-10-31, has 1513, and a cap of 500 cut it after its first clauses.
const LIMITATION_CHARS: usize = 2000;

pub fn parse_event(line: &str) -> Result<Event, String> {
    let v: Value = serde_json::from_str(line).map_err(|e| format!("not JSON: {e}"))?;
    if v["v"].as_u64() != Some(1) {
        return Err("not protocol version 1".into());
    }
    let s = |k: &str| v[k].as_str().map(|x| crate::sanitize(x, 2000));
    match v["event"].as_str() {
        Some("config") => {
            let b64 = v["config_b64"]
                .as_str()
                .ok_or("config without config_b64")?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|_| "config_b64 is not base64")?;
            let sha = v["config_sha256"]
                .as_str()
                .ok_or("config without config_sha256")?
                .to_ascii_lowercase();
            let image = match &v["image"] {
                Value::Null => None,
                i => Some(RelayImage {
                    gcp_image: i["gcp_image"]
                        .as_str()
                        .filter(|n| crate::valid_resource_name(n))
                        .ok_or("config image without a gcp_image name")?
                        .to_string(),
                    disk_tar_sha256: match &i["disk_tar_sha256"] {
                        Value::Null => None,
                        d => Some(
                            d.as_str()
                                .filter(|d| crate::image_import::is_sha256_hex(d))
                                .ok_or("config image disk_tar_sha256 is not a sha256")?
                                .to_string(),
                        ),
                    },
                }),
            };
            let tcb_policy = match &v["tcb_policy"] {
                Value::Null => None,
                t => Some(RelayTcbFloor {
                    strict: match &t["mode"] {
                        Value::Null => true,
                        m => match m.as_str() {
                            Some("strict") => true,
                            Some("transparent") => false,
                            _ => {
                                return Err(
                                    "config tcb_policy mode is neither transparent nor strict"
                                        .into(),
                                )
                            }
                        },
                    },
                    min_tcb_evaluation: t["min_tcb_evaluation"]
                        .as_u64()
                        .ok_or("config tcb_policy without min_tcb_evaluation")?,
                    max_age_secs: t["max_collateral_age_secs"]
                        .as_u64()
                        .ok_or("config tcb_policy without max_collateral_age_secs")?,
                }),
            };
            Ok(Event::Config {
                bytes,
                sha256_hex: sha,
                image,
                tcb_policy,
            })
        }
        Some("progress") => Ok(Event::Progress {
            stage: s("stage").unwrap_or_default(),
            detail: s("detail").unwrap_or_default(),
        }),
        Some("verified") => Ok(Event::Verified {
            verdict: v["verdict"].clone(),
            limitations: v["limitations"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|x| x.as_str().map(|x| crate::sanitize(x, LIMITATION_CHARS)))
                .collect(),
        }),
        Some("refused") => Ok(Event::Refused {
            code: v["code"]
                .as_u64()
                .filter(|c| (64..=87).contains(c))
                .map_or(95, |c| c as u8),
            reason: s("reason").unwrap_or_else(|| "no reason given".into()),
            evidence_file: s("evidence_file"),
        }),
        Some("serving") => Ok(Event::Serving {
            listen: s("listen").ok_or("serving without listen")?,
            key_file: s("key_file"),
            context_tokens: v["context_tokens"].as_u64(),
        }),
        Some("stopped") => Ok(Event::Stopped),
        Some(other) => Ok(Event::Other(crate::sanitize(other, 60))),
        None => Err("no event".into()),
    }
}

/// The cloud config as the relay produced it: hash matches, small, the schema, the model
/// and the expiry this process asked for. Bytes are not re-encoded. With a
/// model key (`model_key`) the config must name sealed weights and where the
/// TD fetches the file; without one it must name none: a session the TD could
/// not serve is refused before anything is created.
pub fn check_config(
    bytes: &[u8],
    sha256_hex: &str,
    model: &str,
    expires_at_unix: u64,
    model_key: bool,
) -> Result<(), String> {
    if bytes.is_empty() || bytes.len() > MAX_CONFIG {
        return Err(format!(
            "config size {} outside 1..={MAX_CONFIG}",
            bytes.len()
        ));
    }
    let got = crate::hex(&Sha256::digest(bytes));
    if got != sha256_hex {
        return Err("config_sha256 does not match config_b64".into());
    }
    let text = std::str::from_utf8(bytes).map_err(|_| "config is not UTF-8")?;
    let v: Value = serde_json::from_str(text).map_err(|_| "config is not JSON")?;
    if v["schema"].as_str() != Some("x0ne-engine/cloud-config/v1") {
        return Err("config schema is not x0ne-engine/cloud-config/v1".into());
    }
    if v["model"].as_str() != Some(model) {
        return Err("config names another model".into());
    }
    if v["expires_at_unix"].as_u64() != Some(expires_at_unix) {
        return Err("config carries another expiry".into());
    }
    match (model_key, v["sealed"].is_object()) {
        (true, false) => Err("config names no sealed weights, and a model key was given".into()),
        (false, true) => Err("config names sealed weights without a model key".into()),
        (true, true) if !v["sealed"]["source"].is_object() => Err(
            "config names sealed weights but no sealed_source: the TD would have no file to \
             fetch (set sealed_source in the key policy)"
                .into(),
        ),
        _ => Ok(()),
    }
}

/// The relay's reference names the image and disk `up` checked, or nothing
/// is created: else the TD of another build is refused after it is paid for.
pub fn check_image(
    relay: Option<&RelayImage>,
    image: &str,
    disk_tar_sha256: Option<&str>,
    from: &str,
) -> Result<(), String> {
    let Some(r) = relay else {
        return Err(format!(
            "the relay names no image (a relay before 0.2.0-beta.6?); up checked {image} ({from})"
        ));
    };
    if r.gcp_image == image && r.disk_tar_sha256.as_deref() == disk_tar_sha256 {
        return Ok(());
    }
    let sha = |s: Option<&str>| s.unwrap_or("none").to_string();
    Err(format!(
        "the relay's reference names image {} (disk.tar.gz sha256 {}), up checked {image} \
         (sha256 {}, {from}): give both the same reference with --reference",
        r.gcp_image,
        sha(r.disk_tar_sha256.as_deref()),
        sha(disk_tar_sha256)
    ))
}

/// The relay applies the TCB policy `up` asked for (`--tcb-policy`,
/// `--min-tcb-evaluation`, `--max-collateral-age`, else the build's floor),
/// or a stricter one. A relay that names none, or a looser one (transparent
/// where strict was asked), would judge the TD on less than the user was
/// told: refused before anything is created.
pub fn check_tcb_policy(
    relay: Option<&RelayTcbFloor>,
    strict: bool,
    min_tcb_evaluation: Option<u64>,
    max_collateral_age: Option<u64>,
) -> Result<(), String> {
    let min = min_tcb_evaluation.unwrap_or(crate::TCB_EVALUATION_FLOOR);
    let age = max_collateral_age.unwrap_or(crate::MAX_COLLATERAL_AGE_SECS);
    let Some(r) = relay else {
        return Err(format!(
            "the relay names no TCB policy (a relay older than this lifecycle?); up asked for \
             TCB evaluation {min} or later and collateral at most {age} s old"
        ));
    };
    if strict && !r.strict {
        return Err(
            "the relay applies the transparent TCB policy, looser than the strict one up asked \
             for (--tcb-policy strict, or --min-tcb-evaluation)"
                .into(),
        );
    }
    if r.min_tcb_evaluation >= min && r.max_age_secs <= age {
        return Ok(());
    }
    Err(format!(
        "the relay applies TCB evaluation {} or later and collateral at most {} s old, looser \
         than the evaluation {min} and {age} s up asked for (--min-tcb-evaluation, \
         --max-collateral-age)",
        r.min_tcb_evaluation, r.max_age_secs
    ))
}

pub struct Proxy {
    child: Child,
    stdin: Option<ChildStdin>,
    out: ChildStdout,
    partial: Vec<u8>,
    pub path: PathBuf,
}

impl Proxy {
    /// Own process group (Ctrl-C reaches the lifecycle, which sends `stop`),
    /// a parent-death signal, and an environment without anything secret.
    pub fn spawn(path: &Path, args: &ProxyArgs) -> Result<Proxy, Failure> {
        let mut cmd = Command::new(path);
        cmd.args(args.argv())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .process_group(0);
        confine(&mut cmd);
        let mut child = cmd
            .spawn()
            .map_err(|e| Failure::Preflight(format!("{}: {e}", path.display())))?;
        let stdin = child.stdin.take();
        let out = child.stdout.take().expect("piped");
        Ok(Proxy {
            child,
            stdin,
            out,
            partial: Vec::new(),
            path: path.to_path_buf(),
        })
    }

    /// Next event; Ok(None) at end of stream. Cancel-safe: a partial line
    /// stays in `self.partial` when a `select!` drops this future.
    pub async fn next_event(&mut self) -> Result<Option<Event>, String> {
        loop {
            if let Some(pos) = self.partial.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = self.partial.drain(..=pos).collect();
                let line =
                    String::from_utf8(line).map_err(|_| "relay line is not UTF-8".to_string())?;
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                return parse_event(line).map(Some);
            }
            if self.partial.len() > MAX_LINE {
                return Err("relay line longer than 1 MiB".into());
            }
            let mut chunk = [0u8; 8192];
            let n = self.out.read(&mut chunk).await.map_err(|e| e.to_string())?;
            if n == 0 {
                return if self.partial.is_empty() {
                    Ok(None)
                } else {
                    Err("relay output ended mid-line".into())
                };
            }
            self.partial.extend_from_slice(&chunk[..n]);
        }
    }

    async fn send(&mut self, v: Value) -> Result<(), String> {
        let stdin = self.stdin.as_mut().ok_or("relay stdin closed")?;
        let mut line = serde_json::to_vec(&v).map_err(|e| e.to_string())?;
        line.push(b'\n');
        stdin.write_all(&line).await.map_err(|e| e.to_string())?;
        stdin.flush().await.map_err(|e| e.to_string())
    }

    pub async fn connect(&mut self, endpoint: &str, boot_deadline_unix: u64) -> Result<(), String> {
        self.send(json!({"v": 1, "cmd": "connect", "endpoint": endpoint,
                         "boot_deadline_unix": boot_deadline_unix}))
            .await
    }

    /// `stop` then end of input; up to `grace` for the exit, then SIGKILL.
    pub async fn stop(&mut self, grace: Duration) -> Option<i32> {
        let _ = self.send(json!({"v": 1, "cmd": "stop"})).await;
        self.stdin = None;
        let wait = async {
            loop {
                if !matches!(self.next_event().await, Ok(Some(_))) {
                    break;
                }
            }
            self.child.wait().await.ok()
        };
        match tokio::time::timeout(grace, wait).await {
            Ok(Some(st)) => st.code(),
            _ => {
                let _ = self.child.start_kill();
                self.child.wait().await.ok().and_then(|s| s.code())
            }
        }
    }

    /// After end of stream: the relay's exit code, waiting a moment for it.
    pub async fn exit_code(&mut self) -> Option<i32> {
        self.stdin = None;
        match tokio::time::timeout(Duration::from_secs(5), self.child.wait()).await {
            Ok(Ok(s)) => s.code(),
            _ => {
                let _ = self.child.start_kill();
                None
            }
        }
    }

    pub fn kill(&mut self) {
        let _ = self.child.start_kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(model: &str, exp: u64) -> (Vec<u8>, String) {
        let b = format!(
            r#"{{"schema":"x0ne-engine/cloud-config/v1","model":"{model}","client_spki_sha256":"{}","client_nonce":"{}","expires_at_unix":{exp},"attest":"cpu+gpu","listen_port":443,"max_completion_tokens":4096}}"#,
            "a".repeat(64),
            "b".repeat(64)
        )
        .into_bytes();
        let h = crate::hex(&Sha256::digest(&b));
        (b, h)
    }

    #[test]
    fn the_relay_is_never_looked_up_in_path() {
        let c = proxy_candidates(Path::new("/opt/x/bin/x0ne-engine-cloud"));
        assert_eq!(
            c,
            vec![
                PathBuf::from("/opt/x/bin/x0ne-engine-proxy"),
                PathBuf::from("/usr/libexec/x0ne-engine/x0ne-engine-proxy")
            ]
        );
    }

    #[test]
    fn the_config_is_checked_as_bytes() {
        let (b, h) = config("qwen3:0.6b", 1_760_303_100);
        assert!(check_config(&b, &h, "qwen3:0.6b", 1_760_303_100, false).is_ok());
        assert!(check_config(&b, &"0".repeat(64), "qwen3:0.6b", 1_760_303_100, false).is_err());
        assert!(check_config(&b, &h, "other:1b", 1_760_303_100, false).is_err());
        assert!(check_config(&b, &h, "qwen3:0.6b", 1, false).is_err());
        let big = vec![b' '; MAX_CONFIG + 1];
        assert!(check_config(
            &big,
            &crate::hex(&Sha256::digest(&big)),
            "qwen3:0.6b",
            1,
            false
        )
        .is_err());
    }

    #[test]
    fn events_are_parsed_strictly_on_version_and_codes() {
        assert!(parse_event(r#"{"v":2,"event":"stopped"}"#).is_err());
        assert_eq!(
            parse_event(r#"{"v":1,"event":"stopped"}"#).unwrap(),
            Event::Stopped
        );
        match parse_event(r#"{"v":1,"event":"refused","code":65,"reason":"RTMR2 \u001b[1m"}"#)
            .unwrap()
        {
            Event::Refused { code, reason, .. } => {
                assert_eq!(code, 65);
                assert_eq!(reason, "RTMR2 ?[1m");
            }
            e => panic!("{e:?}"),
        }
        match parse_event(r#"{"v":1,"event":"refused","code":0}"#).unwrap() {
            Event::Refused { code, .. } => assert_eq!(code, 95),
            e => panic!("{e:?}"),
        }
        let serving = |extra: &str| {
            parse_event(&format!(
                r#"{{"v":1,"event":"serving","listen":"l"{extra}}}"#
            ))
            .unwrap()
        };
        let context = |e: Event| match e {
            Event::Serving { context_tokens, .. } => context_tokens,
            e => panic!("{e:?}"),
        };
        assert_eq!(context(serving(r#","context_tokens":40960"#)), Some(40960));
        assert_eq!(context(serving("")), None);
        assert_eq!(context(serving(r#","context_tokens":-1"#)), None);
    }

    /// 2026-10-31, H100 run: the relay's `verdict verifier` limitation, 1513
    /// characters, was cut at 500 by `status --limits` after its first clauses.
    #[test]
    fn the_relays_limitations_are_kept_whole() {
        let verifier = format!("verdict verifier: {}", "x".repeat(1495));
        assert_eq!(verifier.chars().count(), 1513);
        let line = serde_json::json!({
            "v": 1, "event": "verified", "verdict": {},
            "limitations": [verifier, "ok\u{1b}[31mred", "y".repeat(5000)]
        })
        .to_string();
        match parse_event(&line).unwrap() {
            Event::Verified { limitations, .. } => {
                assert_eq!(limitations[0], verifier);
                // Still sanitized: a control character is `?`, and a line a
                // hostile relay pads past the cap is cut, the ellipsis saying so.
                assert_eq!(limitations[1], "ok?[31mred");
                let n = limitations[2].chars().count();
                assert!(
                    limitations[2].ends_with('…') && n <= LIMITATION_CHARS + 1,
                    "{n}"
                );
            }
            e => panic!("{e:?}"),
        }
    }

    /// The image a config names, parsed strictly, and compared with what `up`
    /// checked: name and disk both, a missing image refused.
    #[test]
    fn the_relay_image_must_be_the_one_up_checked() {
        let (sha, other) = ("1".repeat(64), "2".repeat(64));
        let image = |member: &str| {
            parse_event(&format!(
                r#"{{"v":1,"event":"config","config_b64":"","config_sha256":"00"{member}}}"#
            ))
            .map(|e| match e {
                Event::Config { image, .. } => image,
                e => panic!("{e:?}"),
            })
        };
        let named = image(&format!(
            r#","image":{{"gcp_image":"x0ne-cvm-0-2","disk_tar_sha256":"{sha}"}}"#
        ))
        .unwrap();
        assert_eq!(image("").unwrap(), None);
        for bad in [
            r#","image":{"gcp_image":"../x","disk_tar_sha256":null}"#,
            r#","image":{"disk_tar_sha256":null}"#,
            r#","image":{"gcp_image":"x0ne-cvm-0-2","disk_tar_sha256":"AB"}"#,
        ] {
            assert!(image(bad).is_err(), "{bad}");
        }
        let check = |img: &str, s: Option<&str>| check_image(named.as_ref(), img, s, "test");
        assert!(check("x0ne-cvm-0-2", Some(&sha)).is_ok());
        assert!(check("x0ne-cvm-0-2-1", Some(&sha)).is_err());
        assert!(check("x0ne-cvm-0-2", Some(&other)).is_err());
        assert!(check("x0ne-cvm-0-2", None).is_err());
        assert!(check_image(None, "x0ne-cvm-0-2", Some(&sha), "test").is_err());
    }

    /// The TCB policy a config names, parsed strictly, and compared with
    /// what `up` asked for: equal or stricter passes, looser or none refused.
    #[test]
    fn the_relay_tcb_policy_must_be_at_least_what_up_asked() {
        let policy = |member: &str| {
            parse_event(&format!(
                r#"{{"v":1,"event":"config","config_b64":"","config_sha256":"00"{member}}}"#
            ))
            .map(|e| match e {
                Event::Config { tcb_policy, .. } => tcb_policy,
                e => panic!("{e:?}"),
            })
        };
        let named = |min: u64, age: u64| {
            policy(&format!(
                r#","tcb_policy":{{"min_tcb_evaluation":{min},"max_collateral_age_secs":{age}}}"#
            ))
            .unwrap()
        };
        assert_eq!(
            named(23, 3600),
            Some(RelayTcbFloor {
                strict: true,
                min_tcb_evaluation: 23,
                max_age_secs: 3600
            })
        );
        let moded = |mode: &str| {
            policy(&format!(
                r#","tcb_policy":{{"mode":"{mode}","min_tcb_evaluation":22,"max_collateral_age_secs":86400}}"#
            ))
            .unwrap()
        };
        let (transparent, strict) = (moded("transparent"), moded("strict"));
        assert_eq!(transparent.map(|t| t.strict), Some(false));
        assert_eq!(strict.map(|t| t.strict), Some(true));
        assert_eq!(policy("").unwrap(), None);
        for bad in [
            r#","tcb_policy":{"mode":"lenient","min_tcb_evaluation":22,"max_collateral_age_secs":86400}"#,
            r#","tcb_policy":{"max_collateral_age_secs":3600}"#,
            r#","tcb_policy":{"min_tcb_evaluation":"23","max_collateral_age_secs":3600}"#,
            r#","tcb_policy":{"min_tcb_evaluation":23}"#,
        ] {
            assert!(policy(bad).is_err(), "{bad}");
        }
        let floor = named(crate::TCB_EVALUATION_FLOOR, crate::MAX_COLLATERAL_AGE_SECS);
        assert!(check_tcb_policy(floor.as_ref(), false, None, None).is_ok());
        assert!(check_tcb_policy(named(23, 3600).as_ref(), true, Some(23), Some(3600)).is_ok());
        assert!(check_tcb_policy(named(24, 60).as_ref(), true, Some(23), Some(3600)).is_ok());
        let e = check_tcb_policy(floor.as_ref(), true, Some(23), None).unwrap_err();
        assert!(e.contains("looser") && e.contains("evaluation 23"), "{e}");
        let e = check_tcb_policy(floor.as_ref(), false, None, Some(3600)).unwrap_err();
        assert!(e.contains("looser") && e.contains("3600 s"), "{e}");
        assert!(check_tcb_policy(named(21, 86_400).as_ref(), false, None, None).is_err());
        assert!(check_tcb_policy(named(22, 86_401).as_ref(), false, None, None).is_err());
        let e = check_tcb_policy(None, false, None, None).unwrap_err();
        assert!(e.contains("names no TCB policy"), "{e}");
        // SECURITY (2026-10-28): strict asked, a transparent relay is looser;
        // transparent asked (the default), a strict relay is stricter.
        let e = check_tcb_policy(transparent.as_ref(), true, None, None).unwrap_err();
        assert!(
            e.contains("transparent TCB policy, looser than the strict one"),
            "{e}"
        );
        assert!(check_tcb_policy(transparent.as_ref(), true, Some(22), None).is_err());
        assert!(check_tcb_policy(strict.as_ref(), true, None, None).is_ok());
        assert!(check_tcb_policy(strict.as_ref(), false, None, None).is_ok());
        assert!(check_tcb_policy(transparent.as_ref(), false, None, None).is_ok());
    }

    #[test]
    fn the_relay_argv_is_the_i4_one() {
        let a = ProxyArgs {
            provider: "gcp".into(),
            machine_type: "a3-highgpu-1g".into(),
            model: "qwen3:0.6b".into(),
            expires_at_unix: 1_760_303_100,
            listen: "127.0.0.1:8478".into(),
            reference: None,
            tcb_policy: None,
            min_tcb_evaluation: None,
            max_collateral_age: None,
            model_key: None,
        };
        assert_eq!(
            a.argv().join(" "),
            "--protocol 1 --provider gcp --machine-type a3-highgpu-1g --model qwen3:0.6b \
             --expires-at-unix 1760303100 --listen 127.0.0.1:8478 --quiet"
        );
        let strict = ProxyArgs {
            tcb_policy: Some("strict"),
            min_tcb_evaluation: Some(23),
            max_collateral_age: Some(3600),
            ..a.clone()
        };
        assert!(
            strict.argv().join(" ").ends_with(
                "--quiet --tcb-policy strict --min-tcb-evaluation 23 --max-collateral-age 3600"
            ),
            "{:?}",
            strict.argv()
        );
        let owner = ProxyArgs {
            model_key: Some((PathBuf::from("/o/key"), PathBuf::from("/o/policy.json"))),
            ..a
        };
        assert_eq!(
            owner.argv().join(" "),
            "--protocol 1 --provider gcp --machine-type a3-highgpu-1g --model qwen3:0.6b \
             --expires-at-unix 1760303100 --listen 127.0.0.1:8478 --quiet \
             --model-key /o/key --key-policy /o/policy.json"
        );
    }

    /// The relay's config must name sealed weights exactly when a model key
    /// was given, and then say where the TD fetches the file: a session the
    /// TD could not serve is refused before anything is created.
    #[test]
    fn the_config_names_sealed_weights_exactly_when_a_model_key_is_given() {
        let (b, h) = config("qwen3:0.6b", 1_760_303_100);
        assert!(check_config(&b, &h, "qwen3:0.6b", 1_760_303_100, false).is_ok());
        let e = check_config(&b, &h, "qwen3:0.6b", 1_760_303_100, true).unwrap_err();
        assert!(e.contains("no sealed weights"), "{e}");
        let sealed = |source: &str| {
            let text = String::from_utf8(b.clone()).unwrap().replace(
                r#""max_completion_tokens":4096}"#,
                &format!(
                    r#""max_completion_tokens":4096,"sealed":{{"sealed_sha256":"{}","digest_hex":"{}"{source}}}}}"#,
                    "33".repeat(32),
                    "44".repeat(32)
                ),
            );
            let h = crate::hex(&Sha256::digest(text.as_bytes()));
            (text.into_bytes(), h)
        };
        let (sb, sh) = sealed(r#","source":{"url":"https://w.example/m.sealed","size":5}"#);
        assert!(check_config(&sb, &sh, "qwen3:0.6b", 1_760_303_100, true).is_ok());
        let e = check_config(&sb, &sh, "qwen3:0.6b", 1_760_303_100, false).unwrap_err();
        assert!(e.contains("without a model key"), "{e}");
        let (nb, nh) = sealed("");
        let e = check_config(&nb, &nh, "qwen3:0.6b", 1_760_303_100, true).unwrap_err();
        assert!(e.contains("sealed_source"), "{e}");
    }
}
