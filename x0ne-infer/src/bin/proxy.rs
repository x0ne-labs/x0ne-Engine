//! `x0ne-engine-proxy` — the cloud relay, on YOUR machine. The lifecycle
//! (`x0ne-engine cloud`) starts it by absolute path and talks to it
//! over stdin/stdout (`relay.rs`). It verifies the TD's proof itself
//! (`remote_verify.rs`), releases the session key only after that, and
//! serves the OpenAI API on the loopback (`local_api.rs`). With
//! `--model-key` and `--key-policy` the session serves the model owner's
//! sealed weights: the owner's key goes to the TD, wrapped to its attested
//! X25519 key, after the verdict and the owner's policy (`model_key.rs`).
//!
//! Exit codes: 0 stopped on request; 2 control-stream or usage error;
//! 65-77 the verdict's refusals, as `x0ne-infer` numbers them (65
//! measurement or verdict, 66 not cryptographic, 70 key refused, 72 the TD
//! failed to load, 75 NRAS, 76 Intel or Google unreachable, 77 model);
//! 73 the process could not be made non-dumpable (`harden::process`);
//! 87 this machine's clock is wrong; 88 the attested channel was not
//! established or not bound; 89 the image reference is unusable; 96 the
//! catalog does not serve this model in cloud mode (refused before anything
//! is created); 97 TLS identity; 98 the local endpoint could not be opened; 99 the image
//! reference is not signed by the x0ne archive key pinned in this build, or
//! this build pins none (built before the signing ceremony); 100 the image
//! reference is not shown in the transparency log pinned in this build
//! (`transparency.rs`), the log forked, or this build pins no log; 102 the
//! model owner's key policy refuses this TD or this session, or its files are
//! unusable; 104 the image reference reports RTMR0 instead of pinning it
//! (`rtmr0.policy` `report`: the relay takes `gce-endorsed` alone).
//!
//! No flag and no variable sets a trust anchor or skips a check
//! (`tests/cloud_source_sweep.rs`).
//!
//! `x0ne-engine-proxy chat` is a client of a serving relay's local API
//! (`local_chat.rs`), started by `x0ne-engine cloud run`: the questions stay
//! in the relay's binary, never in the lifecycle's. Exit 1: the chat failed.
//!
//! `x0ne-engine-proxy pins` prints the archive key and the transparency log
//! this build pins, as one `pins` event with `test` true when they are the
//! TEST ones, and exits 0; without them it refuses as a session would, 99 or
//! 100. `x0ne-engine cloud up` asks it before the token and refuses a relay
//! that pins the TEST anchors. It reads no file and writes nothing.

use std::path::PathBuf;
use std::time::Duration;
use x0ne_infer::relay;
use x0ne_infer::term_color::{self, Style};

const USAGE: &str = "\
x0ne-engine-proxy — the cloud relay: verifies the TD's proof on this machine,
then serves the OpenAI API on the loopback. Driven by `x0ne-engine cloud`.

USAGE:
  x0ne-engine-proxy --protocol 1 --provider <name> --machine-type <type> --model <name>
                    --expires-at-unix <N> [--listen 127.0.0.1:8478] [--reference FILE]
                    [--attest cpu+gpu] [--td-port 443] [--max-completion-tokens N]
                    [--tcb-policy transparent|strict] [--min-tcb-evaluation N]
                    [--max-collateral-age secs]
                    [--model-key FILE --key-policy FILE] [--quiet]

  --protocol <N>               the control protocol; this build speaks 1
  --provider <name>            the cloud provider the reference must name (gcp)
  --machine-type <type>        the machine type the MRTD endorsement is read for
  --model <name>               a catalog model name (x0ne-engine list)
  --expires-at-unix <N>        when the session ends (in the config, in the quote)
  --listen <addr>              the local endpoint, loopback only [default: 127.0.0.1:8478]
  --reference <file>           the image reference to pin instead of the installed one
                               (the verdict names the file and its SHA-256);
                               <file>.sig must be its detached signature by the
                               x0ne archive key, like the installed one's
  --attest <cpu+gpu>           the product asked of the TD, the CPU's proof and the
                               GPU's [default and only value]. `cpu` is refused: the
                               TD serves only on its attested GPU
  --td-port <port>             the TD's TLS port [default: 443]
  --max-completion-tokens <N>  ceiling asked of the TD [default: 4096]
  --tcb-policy <mode>          transparent: accept on Intel's standard TCB collateral and
                               name, in the verdict and every receipt, the platform's TCB
                               date and the advisories Intel's latest collateral lists
                               against it; on SEV-SNP, impose AMD bulletins a year old and
                               name newer ones. strict: refuse anything Intel's latest
                               collateral (evaluation 22 or --min-tcb-evaluation) does not
                               rate UpToDate or SWHardeningNeeded, and every AMD bulletin
                               this build knows [default: transparent]
  --min-tcb-evaluation <N>     strict only (implies it when --tcb-policy is not given):
                               the oldest Intel TCB evaluation (TCB recovery) the TD's
                               tcbInfo and QE identity may carry; this build's floor can
                               be raised, not lowered [default: 22, 2026-08-11]
  --max-collateral-age <secs>  how long before the verdict Intel may have issued the
                               tcbInfo, the QE identity and the PCK CRL; can be
                               shortened, not lengthened [default: 86400]
  --model-key <file>           the model owner's key: 64 hex characters, mode 0600.
                               The session serves the owner's sealed weights for
                               --model; the key leaves this machine only wrapped
                               to the key a fresh quote of the verified TD names
  --key-policy <file>          the owner's release policy (x0ne-engine/model-key-
                               policy/v1): the images, TCB, GPU and expiry the
                               key may go to; any other TD is refused (exit 102).
                               Its sealed_source (URL, size) tells the TD where
                               to fetch the sealed file after its attestation
  --quiet                      the progress, the verdict and the refusals go to the
                               control stream only: the caller prints them, as
                               `x0ne-engine cloud` does; other messages stay on stderr
  -h, --help                   print this and exit
  -V, --version                print the version and exit

  x0ne-engine-proxy pins

  the archive key and the transparency log this build pins, as one JSON
  line (\"test\": true when they are the TEST ones); without them, the
  refusal (99 or 100) every session would get

  x0ne-engine-proxy chat --listen <addr> --key-file <file> [--max-tokens N]

  a chat in this terminal on a serving session's local API, started by
  `x0ne-engine cloud run`; leaving it does not end the session
  --max-tokens <N>             the longest answer [default: 1024, max 4096]

ENVIRONMENT:
  XDG_RUNTIME_DIR   where x0ne-engine/local.key is written (0600); without it,
                    /run/user/<uid>, else <temp dir>/x0ne-engine-<uid>; a
                    directory that is not yours, or a link, is refused (98)
  XDG_STATE_HOME    where a refused attestation document is kept for review
";

struct Parsed {
    args: relay::Args,
    reference: Option<PathBuf>,
    quiet: bool,
    /// The production TCB policy, or a stricter one (`--min-tcb-evaluation`,
    /// `--max-collateral-age`).
    tcb_policy: x0ne_infer::tcb::TcbPolicy,
    /// `--model-key` and `--key-policy`, both or neither.
    model_key: Option<(PathBuf, PathBuf)>,
}

fn parse(argv: Vec<String>) -> Result<Option<Parsed>, String> {
    let mut protocol: Option<u64> = None;
    let mut provider = None;
    let mut machine_type = None;
    let mut model = None;
    let mut expires_at_unix = None;
    let mut listen = x0ne_infer::local_api::DEFAULT_LISTEN.to_string();
    let mut reference = None;
    let mut attest = "cpu+gpu".to_string();
    let mut td_port = relay::DEFAULT_TD_PORT;
    let mut max_completion_tokens = x0ne_infer::server::DEFAULT_MAX_COMPLETION_TOKENS;
    let mut quiet = false;
    let (mut min_tcb_evaluation, mut max_collateral_age) = (None, None);
    let mut tcb_mode = None;
    let mut model_key = None;
    let mut key_policy = None;
    let mut it = argv.into_iter();
    while let Some(a) = it.next() {
        let mut value = |flag: &str| it.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(None);
            }
            "-V" | "--version" => {
                println!(
                    "x0ne-engine-proxy {}{}",
                    env!("CARGO_PKG_VERSION"),
                    git_suffix()
                );
                return Ok(None);
            }
            "--protocol" => {
                protocol = Some(
                    value("--protocol")?
                        .parse()
                        .map_err(|e| format!("--protocol: {e}"))?,
                )
            }
            "--provider" => provider = Some(value("--provider")?),
            "--machine-type" => machine_type = Some(value("--machine-type")?),
            "--model" => model = Some(value("--model")?),
            "--expires-at-unix" => {
                expires_at_unix = Some(
                    value("--expires-at-unix")?
                        .parse()
                        .map_err(|e| format!("--expires-at-unix: {e}"))?,
                )
            }
            "--listen" => listen = value("--listen")?,
            "--reference" => reference = Some(PathBuf::from(value("--reference")?)),
            "--attest" => {
                attest = value("--attest")?;
                attest_asked(&attest)?;
            }
            "--quiet" => quiet = true,
            "--model-key" => model_key = Some(PathBuf::from(value("--model-key")?)),
            "--key-policy" => key_policy = Some(PathBuf::from(value("--key-policy")?)),
            "--td-port" => {
                td_port = value("--td-port")?
                    .parse()
                    .map_err(|e| format!("--td-port: {e}"))?
            }
            "--max-completion-tokens" => {
                max_completion_tokens = value("--max-completion-tokens")?
                    .parse()
                    .map_err(|e| format!("--max-completion-tokens: {e}"))?
            }
            "--tcb-policy" => {
                tcb_mode = Some(x0ne_infer::tcb::TcbMode::parse(&value("--tcb-policy")?)?)
            }
            "--min-tcb-evaluation" => {
                min_tcb_evaluation = Some(
                    value("--min-tcb-evaluation")?
                        .parse::<u64>()
                        .map_err(|e| format!("--min-tcb-evaluation: {e}"))?,
                )
            }
            "--max-collateral-age" => {
                max_collateral_age = Some(
                    value("--max-collateral-age")?
                        .parse::<u64>()
                        .map_err(|e| format!("--max-collateral-age: {e}"))?,
                )
            }
            other => {
                return Err(format!(
                    "unknown argument {}",
                    x0ne_infer::argv::shown(other)
                ))
            }
        }
    }
    match protocol {
        Some(p) if p == relay::PROTOCOL => {}
        Some(p) => {
            return Err(format!(
                "--protocol {p}: this build speaks {}",
                relay::PROTOCOL
            ))
        }
        None => return Err("--protocol is required".into()),
    }
    // Stricter than the build or refused: no flag lowers the floor. The floor
    // on Intel's latest collateral is what `strict` refuses on: given alone it
    // keeps that meaning (strict); beside `transparent` it is refused.
    let tcb_mode = match (tcb_mode, min_tcb_evaluation) {
        (Some(x0ne_infer::tcb::TcbMode::Transparent), Some(_)) => {
            return Err(
                "--min-tcb-evaluation refuses only under --tcb-policy strict: transparent names, it does not refuse"
                    .into(),
            )
        }
        (Some(m), _) => m,
        (None, Some(_)) => x0ne_infer::tcb::TcbMode::Strict,
        (None, None) => x0ne_infer::tcb::TcbPolicy::production().mode,
    };
    let tcb_policy = x0ne_infer::tcb::TcbPolicy::tightened(min_tcb_evaluation, max_collateral_age)?
        .with_mode(tcb_mode);
    let model_key = match (model_key, key_policy) {
        (Some(k), Some(p)) => Some((k, p)),
        (None, None) => None,
        _ => return Err("--model-key and --key-policy go together".into()),
    };
    let need = |what: &str, v: Option<String>| v.ok_or_else(|| format!("{what} is required"));
    Ok(Some(Parsed {
        args: relay::Args {
            provider: need("--provider", provider)?,
            machine_type: need("--machine-type", machine_type)?,
            model: need("--model", model)?,
            expires_at_unix: expires_at_unix.ok_or("--expires-at-unix is required")?,
            listen,
            attest,
            td_port,
            max_completion_tokens,
        },
        reference,
        quiet,
        tcb_policy,
        model_key,
    }))
}

/// `cpu+gpu`, the one product a TD serves. `cpu` is refused by name: x0ne-infer
/// serves only on its attested GPU and refuses a config that attests the CPU
/// alone (its exit 90), so the relay never asks for one. Usage error, exit 2.
fn attest_asked(attest: &str) -> Result<(), String> {
    match attest {
        "cpu+gpu" => Ok(()),
        "cpu" => Err(
            "--attest cpu: refused. The TD serves only on its attested GPU (cpu+gpu); \
             x0ne-infer refuses a CPU-only config, nothing falls back to one"
                .into(),
        ),
        _ => Err(x0ne_infer::argv::refused(
            "unknown --attest value",
            "cpu+gpu",
        )),
    }
}

/// `" (git <sha>)"` when the package build set `X0NE_GIT_SHA` (ci/inside-deb.sh), else empty.
fn git_suffix() -> String {
    option_env!("X0NE_GIT_SHA")
        .map(|s| format!(" (git {s})"))
        .unwrap_or_default()
}

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

/// `$XDG_RUNTIME_DIR` (absolute, as the spec asks), else `/run/user/<uid>`,
/// else `<temp dir>/x0ne-engine-<uid>` (macOS: `$TMPDIR`, which has no
/// `XDG_RUNTIME_DIR`). `local_api::write_key_file` creates what is missing
/// 0700 and refuses a directory that is not the caller's own.
fn runtime_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|d| d.is_absolute())
    {
        return d;
    }
    #[cfg(unix)]
    {
        // SAFETY: getuid has no preconditions.
        let uid = unsafe { libc::getuid() };
        let d = PathBuf::from(format!("/run/user/{uid}"));
        if d.is_dir() {
            return d;
        }
        std::env::temp_dir().join(format!("x0ne-engine-{uid}"))
    }
    #[cfg(not(unix))]
    std::env::temp_dir().join("x0ne-engine")
}

fn state_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("XDG_STATE_HOME").filter(|d| !d.is_empty()) {
        return PathBuf::from(d).join("x0ne-engine");
    }
    match std::env::var_os("HOME").filter(|d| !d.is_empty()) {
        Some(h) => PathBuf::from(h).join(".local/state/x0ne-engine"),
        None => std::env::temp_dir().join("x0ne-engine-state"),
    }
}

/// Before the relay runs, errors still speak the control protocol.
fn refuse_early(code: i32, reason: &str, quiet: bool) -> i32 {
    let reason = &x0ne_infer::local_chat::printable(reason);
    if !quiet {
        eprintln!(
            "{}",
            term_color::paint(
                term_color::stderr_on(),
                Style::Fail,
                &format!("x0ne-engine-proxy: {reason}")
            )
        );
    }
    println!(
        "{}",
        serde_json::json!({ "v": relay::PROTOCOL, "event": "refused", "code": code, "reason": reason, "evidence_file": null })
    );
    code
}

/// TEST anchors: a key whose user ID says TEST (`archive_key.rs`), or a log
/// whose origin is under `.invalid` (RFC 2606), as the TEST log's is.
fn test_anchors(cert: &x0ne_infer::openpgp::Cert, origin: &str) -> bool {
    cert.user_ids().iter().any(|u| u.contains("TEST"))
        || origin
            .split('/')
            .next()
            .is_some_and(|host| host.ends_with(".invalid"))
}

/// `pins`: 0 and a `pins` event naming what this build pins, `test` true when
/// they are the TEST anchors, or the refusal (99, 100) a session of this
/// build would get before it looks for a reference; 2 if anything follows.
fn pins(args: &[String]) -> i32 {
    if !args.is_empty() {
        eprintln!("x0ne-engine-proxy: pins takes no argument\n{USAGE}");
        return relay::EXIT_PROTOCOL;
    }
    let archive_key = x0ne_infer::archive_key::Anchor::production();
    let tlog = x0ne_infer::transparency::Anchor::production();
    let pinned = archive_key
        .key()
        .map_err(x0ne_infer::reference::LoadError::NoKey)
        .and_then(|cert| {
            tlog.policy()
                .map(|p| {
                    let test = test_anchors(cert, &p.origin);
                    (cert.primary_fingerprint(), p.origin.clone(), test)
                })
                .map_err(x0ne_infer::reference::LoadError::NotLogged)
        });
    match pinned {
        Ok((fingerprint, origin, test)) => {
            println!(
                "{}",
                serde_json::json!({ "v": relay::PROTOCOL, "event": "pins", "archive_key": fingerprint, "transparency_log": origin, "test": test })
            );
            0
        }
        Err(e) => refuse_early(e.exit_code(), &format!("image reference: {e}"), true),
    }
}

/// `chat …`: 0 when the user leaves, 3 when the relay stopped the session on
/// an answer (`local_chat::EXIT_SESSION_STOPPED`), 1 when the chat fails
/// otherwise, 2 on usage.
fn chat(args: &[String]) -> i32 {
    use std::io::Write as _;
    let o = match x0ne_infer::local_chat::parse(args) {
        Ok(o) => o,
        Err(e) => {
            let _ = writeln!(
                std::io::stderr(),
                "x0ne-engine-proxy: {e}\nUSAGE:\n{}",
                x0ne_infer::local_chat::USAGE
            );
            return relay::EXIT_PROTOCOL;
        }
    };
    let done = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| x0ne_infer::local_chat::ChatEnd::from(format!("tokio runtime: {e}")))
        .and_then(|rt| rt.block_on(x0ne_infer::local_chat::run(&o)));
    match done {
        Ok(()) => 0,
        Err(e) => {
            let _ = writeln!(
                std::io::stderr(),
                "{}",
                term_color::paint(
                    term_color::stderr_on(),
                    Style::Fail,
                    &format!("x0ne-engine-proxy chat: {e}")
                )
            );
            e.code
        }
    }
}

fn main() {
    // Peers of the same user cannot ptrace it or read its memory; no core file.
    // Its return was ignored until 2026-11-04: a failed prctl went on, dumpable.
    if let Err(e) = x0ne_infer::harden::process() {
        eprintln!("x0ne-engine-proxy: process hardening: {e}");
        std::process::exit(73);
    }
    // stderr is the lifecycle's journal: no panic payload in it.
    x0ne_infer::server::withhold_panic_payloads();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("chat") {
        std::process::exit(chat(&argv[1..]));
    }
    if argv.first().map(String::as_str) == Some("pins") {
        std::process::exit(pins(&argv[1..]));
    }
    let parsed = match parse(argv) {
        Ok(Some(p)) => p,
        Ok(None) => return,
        Err(e) => {
            eprintln!("x0ne-engine-proxy: {e}\n{USAGE}");
            std::process::exit(relay::EXIT_PROTOCOL);
        }
    };
    let archive_key = x0ne_infer::archive_key::Anchor::production();
    let tlog = x0ne_infer::transparency::Anchor::production();
    let tlog_state = state_dir();
    let load = |path: &std::path::Path, explicit| {
        x0ne_infer::remote_verify::Pinned::load(
            path,
            explicit,
            &archive_key,
            &tlog,
            &tlog_state,
            now_unix(),
        )
    };
    // No pin, no reference can pass: say it before looking for one.
    let reference = archive_key
        .key()
        .map_err(x0ne_infer::reference::LoadError::NoKey)
        .and_then(|_| {
            tlog.policy()
                .map(|_| ())
                .map_err(x0ne_infer::reference::LoadError::NotLogged)
        })
        .and_then(|_| match &parsed.reference {
            Some(path) => load(path, true),
            None => x0ne_infer::reference::default_path()
                .map_err(x0ne_infer::reference::LoadError::Unusable)
                .and_then(|p| load(&p, false)),
        });
    let reference = match reference {
        Ok(r) => r,
        Err(e) => std::process::exit(refuse_early(
            e.exit_code(),
            &format!("image reference: {e}"),
            parsed.quiet,
        )),
    };
    // The owner's key and policy, read before anything is created.
    let model_key = match &parsed.model_key {
        None => None,
        Some((key, policy)) => match x0ne_infer::model_key::OwnerKey::load(key, policy) {
            Ok(k) => Some(std::sync::Arc::new(k)),
            Err(e) => std::process::exit(refuse_early(
                x0ne_infer::model_key::EXIT_POLICY,
                &format!("model key: {e}"),
                parsed.quiet,
            )),
        },
    };
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => std::process::exit(refuse_early(
            relay::EXIT_LOCAL_ENDPOINT,
            &format!("tokio runtime: {e}"),
            parsed.quiet,
        )),
    };
    let mut anchors = x0ne_infer::remote_verify::Anchors::production();
    anchors.tcb_policy = parsed.tcb_policy;
    let env = relay::Env {
        anchors,
        reference,
        runtime_dir: runtime_dir(),
        state_dir: state_dir(),
        clock: std::sync::Arc::new(now_unix),
        random: std::sync::Arc::new(urandom32),
        poll_interval: Duration::from_secs(3),
        load_timeout: Duration::from_mins(20),
        version: env!("CARGO_PKG_VERSION").into(),
        color: term_color::stderr_on(),
        quiet: parsed.quiet,
        model_key,
    };
    let code = runtime.block_on(async move {
        let stdin = tokio::io::BufReader::new(tokio::io::stdin());
        let mut stdout = tokio::io::stdout();
        relay::run(parsed.args, env, stdin, &mut stdout).await
    });
    std::process::exit(code);
}
