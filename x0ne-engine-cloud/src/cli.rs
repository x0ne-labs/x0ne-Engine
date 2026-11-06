//! Arguments of `x0ne-engine cloud`. One text for `--help` and the parser.

use std::path::PathBuf;

use crate::chat::{ChatOpts, DEFAULT_MAX_TOKENS, MAX_TOKENS_CAP};
use crate::{
    Failure, DEFAULT_LISTEN, DEFAULT_MAX_RUN_SECS, DEFAULT_MODEL, MAX_COLLATERAL_AGE_SECS,
    MAX_RUN_CAP_SECS, MIN_RUN_SECS, TCB_EVALUATION_FLOOR,
};

pub const USAGE: &str = "\
x0ne-engine cloud — a confidential GPU VM for one session, proved on THIS machine.

USAGE:
  x0ne-engine cloud run <family:tag>     one model, one chat, on a confidential GPU VM:
                                         quote, \"yes\", create the VM, check its proof
                                         here, then chat in this terminal (the OpenAI
                                         API stays on the loopback for other clients);
                                         /exit, Ctrl-D or Ctrl-C ends the session and
                                         deletes the VM
  x0ne-engine cloud quote                quote only; creates nothing
  x0ne-engine cloud up                   quote, \"yes\", create the VM, check its hardware
                                         proof here, then serve an OpenAI API on the
                                         loopback; Ctrl-C stops and deletes the VM
  x0ne-engine cloud chat                 chat with the session running on this machine
                                         (another terminal); leaving does not end it
  x0ne-engine cloud status [--limits]    the local session (no token needed)
  x0ne-engine cloud down                 stop the session and delete its VM (another
                                         terminal, or after a crash)
  x0ne-engine cloud sweep                list x0ne-engine leftovers in the project,
                                         delete them after \"yes\"
  x0ne-engine cloud image import <disk.tar.gz>
                                         once per project: import the CVM image
  x0ne-engine cloud image delete <x0ne-cvm-…>
                                         delete an image `image import` made, after
                                         \"yes\"; refused while a VM boots from it
  x0ne-engine cloud setup gcp            how to install gcloud on x0ne OS
  x0ne-engine cloud login gcp            gcloud auth login --no-launch-browser

OPTIONS:
  --provider gcp|azure      [default: gcp; azure is refused: different guarantee]
  --project <id>            [default: $GOOGLE_CLOUD_PROJECT; with --auth gcloud,
                            gcloud's own]
  --zone <zone>             [default: the first of europe-west4-c, us-central1-a,
                            us-east5-a with H100 spot quota]
  --region <region>         image import only [default: us-central1]
  --model <family:tag>      up [default: qwen3:0.6b]; up and run serve a catalog
                            model with a pinned served-model digest, sharded or not
  --max-run <30m|2h>        the VM deletes itself after this [default: 60m, max 8h];
                            with --auth paste or stdin, the session ends 5 min
                            before the token expires, to delete the VM with it,
                            and a token too short for the model's boot is refused
  --auth paste|stdin|gcloud token source [default: paste: hidden, kept in RAM];
                            with stdin, run reads your messages on the terminal
  --listen <ip:port>        local API, loopback only [default: 127.0.0.1:8478]
  --reference <file>        image reference values you chose (named in the verdict)
  --model-key <file>        up and run: your model key (64 hex, mode 0600), with
  --key-policy <file>       your release policy; the relay reads both and the
                            TD serves your sealed weights (docs/model-key.md)
  --yes                     no question asked; `up` and `run` also need --max-cost
  --max-cost <USD>          refuse any quote above this
  --max-tokens <N>          run and chat: the longest answer [default: 1024, max 4096]
  --tcb-policy <mode>       up and run: transparent serves on Intel's standard
                            TCB collateral and names, after VERIFIED and in every
                            receipt, the platform's TCB date and the advisories
                            Intel's latest collateral lists against it; strict
                            refuses what that latest collateral does not rate
                            current [default: transparent]
  --min-tcb-evaluation <N>  up and run, strict only (implies it): the oldest
                            Intel TCB recovery the relay accepts in the TD's
                            collateral; raise only [default: 22, of 2026-08-11]
  --max-collateral-age <s>  up and run: how old Intel's tcbInfo, QE identity and
                            PCK CRL may be at the verdict; shorten only
                            [default: 86400]
  --key-file <file>         chat: the relay's key [default: the session's]
  --allow-from <CIDR>       up and run: who may open a connection to the VM's
                            tcp:443, an IPv4 address or network, repeatable
                            [default: this machine's public IPv4 address, /32,
                            as Google's name server sees it]
  --nras-key-file <file>    up and run: an NVIDIA API key for NRAS, in a file
                            you own of mode 0600 or 0400; handed to the VM in
                            the metadata item x0ne-nras-key, never printed
                            [default: none, NRAS is called without a key]

The token is never put in the environment, argv, a file or a log. The VM is
created with an automatic deletion at the end of its run, whatever happens here.

EXIT CODES:
  0 ok · 2 usage · 64-87 the relay's refusal (x0ne-infer codes, 65 measurement,
  75 NRAS, 76 Intel PCS) · 90 quote refused · 91 preflight · 92 provider API ·
  93 VM lost (preemption, boot timeout) · 94 DELETION NOT CONFIRMED · 95 relay
  failed · 96 the chat failed (run: the VM is deleted) · 97 x0ne-infer in the VM
  refused to serve, read on its serial console (the VM is deleted) · 102 the
  model owner's key policy refused the TD or the session (--key-policy) · 130
  interrupted before serving (what was created is deleted)
";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Auth {
    Paste,
    Stdin,
    Gcloud,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Opts {
    pub provider: String,
    pub project: Option<String>,
    pub zone: Option<String>,
    pub region: Option<String>,
    pub model: String,
    pub max_run_secs: u64,
    pub auth: Auth,
    pub listen: String,
    pub reference: Option<PathBuf>,
    pub yes: bool,
    pub max_cost: Option<f64>,
    /// `run`: the longest answer the chat asks for.
    pub max_tokens: u32,
    /// `--tcb-policy`: `transparent` or `strict`; `None`: the relay's
    /// default (transparent), strict when a minimum evaluation is given.
    pub tcb_policy: Option<&'static str>,
    /// The relay's TCB policy, stricter than its build's; `None`: the build's.
    pub min_tcb_evaluation: Option<u64>,
    pub max_collateral_age: Option<u64>,
    /// `--model-key <file> --key-policy <file>`: the model owner's key and
    /// release policy, both or neither; the relay reads them.
    pub model_key: Option<(PathBuf, PathBuf)>,
    /// `--allow-from`: the firewall rule's sources, canonical; empty: this
    /// machine's public address, /32.
    pub allow_from: Vec<String>,
    /// `--nras-key-file`: the key, read when the command line is.
    pub nras_key: Option<crate::nras_key::NrasKey>,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            provider: "gcp".into(),
            project: None,
            zone: None,
            region: None,
            model: DEFAULT_MODEL.into(),
            max_run_secs: DEFAULT_MAX_RUN_SECS,
            auth: Auth::Paste,
            listen: DEFAULT_LISTEN.into(),
            reference: None,
            yes: false,
            max_cost: None,
            max_tokens: DEFAULT_MAX_TOKENS,
            tcb_policy: None,
            min_tcb_evaluation: None,
            max_collateral_age: None,
            model_key: None,
            allow_from: Vec::new(),
            nras_key: None,
        }
    }
}

impl Opts {
    /// The relay must apply the strict TCB policy: asked for, or implied by
    /// a minimum evaluation (a floor to refuse on).
    pub fn tcb_strict(&self) -> bool {
        self.tcb_policy == Some("strict") || self.min_tcb_evaluation.is_some()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Cmd {
    Help,
    Version,
    Quote(Opts),
    Up(Opts),
    /// `up` with a chat in front; the model is `Opts::model`.
    Run(Opts),
    Chat(ChatOpts),
    Status {
        limits: bool,
    },
    Down(Opts),
    Sweep(Opts),
    ImageImport(Opts, PathBuf),
    /// The image to delete, an `x0ne-cvm-*` resource name.
    ImageDelete(Opts, String),
    SetupGcp,
    LoginGcp,
}

/// `30m`, `2h`, `90` (minutes).
pub fn parse_duration(s: &str) -> Option<u64> {
    let (num, mult) = if let Some(n) = s.strip_suffix('h') {
        (n, 3600)
    } else if let Some(n) = s.strip_suffix("min") {
        (n, 60)
    } else if let Some(n) = s.strip_suffix('m') {
        (n, 60)
    } else {
        (s, 60)
    };
    let n: u64 = num.parse().ok()?;
    n.checked_mul(mult)
}

fn usage(msg: impl Into<String>) -> Failure {
    Failure::Usage(msg.into())
}

pub fn parse(args: &[String]) -> Result<Cmd, Failure> {
    let Some(first) = args.first() else {
        return Ok(Cmd::Help);
    };
    if args.iter().any(|a| a == "-h" || a == "--help") || first == "help" {
        return Ok(Cmd::Help);
    }
    if first == "-V" || first == "--version" {
        return Ok(Cmd::Version);
    }
    let (cmd, rest): (&str, &[String]) = match first.as_str() {
        "image" => match args.get(1).map(String::as_str) {
            Some("import") => ("image import", &args[2..]),
            Some("delete") => ("image delete", &args[2..]),
            _ => {
                return Err(usage(
                    "image: the subcommands are `image import <disk.tar.gz>` and \
                     `image delete <x0ne-cvm-…>`",
                ))
            }
        },
        "setup" | "login" => match args.get(1).map(String::as_str) {
            Some("gcp") if args.len() == 2 => {
                return Ok(if first == "setup" {
                    Cmd::SetupGcp
                } else {
                    Cmd::LoginGcp
                })
            }
            _ => return Err(usage(format!("{first}: only `{first} gcp` exists"))),
        },
        c => (c, &args[1..]),
    };
    if cmd == "status" {
        return match rest {
            [] => Ok(Cmd::Status { limits: false }),
            [l] if l == "--limits" => Ok(Cmd::Status { limits: true }),
            _ => Err(usage("status takes only --limits")),
        };
    }
    if cmd == "chat" {
        return parse_chat(rest).map(Cmd::Chat);
    }
    let mut o = Opts::default();
    let mut max_tokens_given = false;
    // The relay's options, named for the refusal outside `up` and `run`.
    let mut relay_tcb_given: Option<&'static str> = None;
    let mut model_key: Option<PathBuf> = None;
    let mut key_policy: Option<PathBuf> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        let mut val = |name: &str| -> Result<String, Failure> {
            it.next()
                .cloned()
                .ok_or_else(|| usage(format!("{name} needs a value")))
        };
        match a.as_str() {
            "--provider" => {
                let v = val(a)?;
                if !matches!(v.as_str(), "gcp" | "azure") {
                    return Err(usage(crate::argv::refused("--provider", "gcp or azure")));
                }
                o.provider = v;
            }
            "--project" => {
                let v = val(a)?;
                if !crate::valid_project(&v) {
                    return Err(usage(crate::argv::refused("--project", "a GCP project id")));
                }
                o.project = Some(v);
            }
            "--zone" => {
                let v = val(a)?;
                if !crate::valid_zone(&v) {
                    return Err(usage(crate::argv::refused(
                        "--zone",
                        "a zone (us-central1-a)",
                    )));
                }
                o.zone = Some(v);
            }
            "--region" => {
                let v = val(a)?;
                if !crate::valid_zone(&format!("{v}-a")) {
                    return Err(usage(crate::argv::refused(
                        "--region",
                        "a region (us-central1)",
                    )));
                }
                o.region = Some(v);
            }
            "--model" => {
                let v = val(a)?;
                if !crate::valid_model(&v) {
                    return Err(usage(crate::argv::refused(
                        "--model",
                        "family:tag, e.g. qwen3:0.6b",
                    )));
                }
                o.model = v;
            }
            "--max-run" => {
                let v = val(a)?;
                let s = parse_duration(&v).ok_or_else(|| {
                    usage(crate::argv::refused("--max-run", "30m, 2h or minutes"))
                })?;
                if !(MIN_RUN_SECS..=MAX_RUN_CAP_SECS).contains(&s) {
                    return Err(usage(format!("--max-run {v}: between 10m and 8h")));
                }
                o.max_run_secs = s;
            }
            "--auth" => {
                o.auth = match val(a)?.as_str() {
                    "paste" => Auth::Paste,
                    "stdin" => Auth::Stdin,
                    "gcloud" => Auth::Gcloud,
                    _ => {
                        return Err(usage(crate::argv::refused(
                            "--auth",
                            "paste, stdin or gcloud",
                        )))
                    }
                };
            }
            "--listen" => {
                let v = val(a)?;
                let ok = v
                    .parse::<std::net::SocketAddr>()
                    .is_ok_and(|s| s.ip().is_loopback() && s.port() != 0);
                if !ok {
                    return Err(usage(crate::argv::refused(
                        "--listen",
                        "a loopback address and port (127.0.0.1:8478); nothing else is accepted",
                    )));
                }
                o.listen = v;
            }
            "--reference" => {
                let p = PathBuf::from(val(a)?);
                if !p.is_file() {
                    return Err(usage(format!(
                        "--reference {}: no such file",
                        crate::argv::path_shown(&p.to_string_lossy())
                    )));
                }
                o.reference = Some(std::fs::canonicalize(&p).unwrap_or(p));
            }
            "--model-key" => {
                let v = val(a)?;
                if crate::argv::is_hex_key(&v) {
                    return Err(usage(MODEL_KEY_ON_ARGV));
                }
                model_key = Some(existing_file("--model-key", v)?);
            }
            "--key-policy" => key_policy = Some(existing_file("--key-policy", val(a)?)?),
            "--max-tokens" => {
                o.max_tokens = max_tokens(&val(a)?)?;
                max_tokens_given = true;
            }
            "--yes" => o.yes = true,
            "--max-cost" => {
                let v = val(a)?;
                let c: f64 = v
                    .parse()
                    .ok()
                    .filter(|c: &f64| c.is_finite() && *c > 0.0 && *c <= 100_000.0)
                    .ok_or_else(|| {
                        usage(crate::argv::refused(
                            "--max-cost",
                            "a positive amount in USD",
                        ))
                    })?;
                o.max_cost = Some(c);
            }
            "--tcb-policy" => {
                o.tcb_policy = Some(match val(a)?.as_str() {
                    "transparent" => "transparent",
                    "strict" => "strict",
                    _ => {
                        return Err(usage(crate::argv::refused(
                            "--tcb-policy",
                            "transparent or strict",
                        )))
                    }
                });
                relay_tcb_given = Some("--tcb-policy");
            }
            "--min-tcb-evaluation" => {
                let n: u64 = val(a)?
                    .parse()
                    .map_err(|_| usage(crate::argv::refused("--min-tcb-evaluation", "a number")))?;
                if n < TCB_EVALUATION_FLOOR {
                    return Err(usage(format!(
                        "--min-tcb-evaluation {n}: the relay's floor is {TCB_EVALUATION_FLOOR}; it can be raised, not lowered"
                    )));
                }
                o.min_tcb_evaluation = Some(n);
                relay_tcb_given = Some("--min-tcb-evaluation");
            }
            "--max-collateral-age" => {
                let n: u64 = val(a)?
                    .parse()
                    .map_err(|_| usage(crate::argv::refused("--max-collateral-age", "seconds")))?;
                if n > MAX_COLLATERAL_AGE_SECS {
                    return Err(usage(format!(
                        "--max-collateral-age {n}: the relay accepts at most {MAX_COLLATERAL_AGE_SECS} s; it can be shortened, not lengthened"
                    )));
                }
                o.max_collateral_age = Some(n);
                relay_tcb_given = Some("--max-collateral-age");
            }
            "--allow-from" => {
                let v = val(a)?;
                o.allow_from.push(cidr(&v).ok_or_else(|| {
                    usage(crate::argv::refused(
                        "--allow-from",
                        "an IPv4 address or network, a.b.c.d or a.b.c.d/n with the host bits \
                         zero (the VM has an IPv4 address only)",
                    ))
                })?);
            }
            "--nras-key-file" => {
                let v = val(a)?;
                if v.starts_with("nvapi-") || crate::argv::is_hex_key(&v) {
                    return Err(usage(NRAS_KEY_ON_ARGV));
                }
                let key =
                    crate::nras_key::NrasKey::read(std::path::Path::new(&v)).map_err(|e| {
                        usage(format!("--nras-key-file: {e} (the path is not printed)"))
                    })?;
                o.nras_key = Some(key);
            }
            s if s.starts_with('-') => {
                return Err(usage(format!(
                    "unknown option {} (see --help)",
                    crate::argv::shown(s)
                )))
            }
            s => positional.push(s.to_string()),
        }
    }
    let no_positional = |c: Cmd| {
        if positional.is_empty() {
            Ok(c)
        } else {
            Err(usage(format!(
                "unexpected argument {}",
                crate::argv::shown(&positional[0])
            )))
        }
    };
    if max_tokens_given && cmd != "run" {
        return Err(usage("--max-tokens: run and chat only"));
    }
    // Only `up` and `run` start a relay: elsewhere the option would be taken and ignored.
    if let Some(flag) = relay_tcb_given.filter(|_| !matches!(cmd, "up" | "run")) {
        return Err(usage(format!(
            "{flag}: up and run only, the TCB policy of the relay they start; {cmd} starts none"
        )));
    }
    // A floor to refuse on beside the policy that only names: refused, not ignored.
    if o.tcb_policy == Some("transparent") && o.min_tcb_evaluation.is_some() {
        return Err(usage(
            "--min-tcb-evaluation refuses only under --tcb-policy strict: transparent names, it does not refuse",
        ));
    }
    o.model_key = match (model_key, key_policy) {
        (None, None) => None,
        (Some(k), Some(p)) => Some((k, p)),
        _ => return Err(usage(
            "--model-key and --key-policy go together: the owner's key file and its release policy",
        )),
    };
    if o.model_key.is_some() && !matches!(cmd, "up" | "run") {
        return Err(usage("--model-key and --key-policy: up and run only"));
    }
    if o.nras_key.is_some() && !matches!(cmd, "up" | "run") {
        return Err(usage(
            "--nras-key-file: up and run only, the VM they create calls NRAS",
        ));
    }
    if !o.allow_from.is_empty() && !matches!(cmd, "up" | "run") {
        return Err(usage(
            "--allow-from: up and run only, the firewall rule of the VM they create",
        ));
    }
    match cmd {
        "quote" => no_positional(Cmd::Quote(o)),
        "up" => {
            if o.yes && o.max_cost.is_none() {
                return Err(usage(
                    "up --yes needs --max-cost <USD>: without a terminal, the ceiling is what you agree to",
                ));
            }
            crate::catalog::check(&o.model)?;
            no_positional(Cmd::Up(o))
        }
        "run" => {
            if rest.iter().any(|a| a == "--model") {
                return Err(usage(
                    "run takes the model as its argument: x0ne-engine cloud run <family:tag>",
                ));
            }
            let [model] = positional.as_slice() else {
                return Err(usage(
                    "run takes one model: x0ne-engine cloud run <family:tag> (x0ne-engine catalog)",
                ));
            };
            if !crate::valid_model(model) {
                return Err(usage(crate::argv::refused(
                    "the model",
                    "family:tag, e.g. qwen3:0.6b",
                )));
            }
            if o.yes && o.max_cost.is_none() {
                return Err(usage(
                    "run --yes needs --max-cost <USD>: without a terminal, the ceiling is what you agree to",
                ));
            }
            crate::catalog::check(model)?;
            o.model.clone_from(model);
            Ok(Cmd::Run(o))
        }
        "down" => no_positional(Cmd::Down(o)),
        "sweep" => no_positional(Cmd::Sweep(o)),
        "image import" => match positional.as_slice() {
            [f] => Ok(Cmd::ImageImport(o, PathBuf::from(f))),
            _ => Err(usage(
                "image import takes one file: the disk.tar.gz built by x0ne-OS (make cvm)",
            )),
        },
        "image delete" => match positional.as_slice() {
            [name] if is_cvm_image(name) => Ok(Cmd::ImageDelete(o, name.clone())),
            [_] => Err(usage(crate::argv::refused(
                "image delete",
                "the name of an x0ne-cvm-* image, e.g. x0ne-cvm-0-2-1",
            ))),
            _ => Err(usage(
                "image delete takes one image: an x0ne-cvm-* name, e.g. x0ne-cvm-0-2-1",
            )),
        },
        other => Err(usage(format!(
            "unknown command {} (see --help)",
            crate::argv::shown(other)
        ))),
    }
}

/// The only images `image delete` names: those `image import` creates.
pub fn is_cvm_image(name: &str) -> bool {
    name.starts_with("x0ne-cvm-") && crate::valid_resource_name(name)
}

/// `--allow-from`: `a.b.c.d` (a /32) or `a.b.c.d/n` with the host bits zero,
/// written back canonical. IPv4 only: the VM has no IPv6 address.
pub fn cidr(v: &str) -> Option<String> {
    let (ip, len) = match v.split_once('/') {
        Some((ip, n)) if (1..=2).contains(&n.len()) && n.bytes().all(|b| b.is_ascii_digit()) => {
            (ip, n.parse::<u32>().ok()?)
        }
        Some(_) => return None,
        None => (v, 32),
    };
    let ip: std::net::Ipv4Addr = ip.parse().ok()?;
    let host = u32::MAX.checked_shr(len).unwrap_or(0);
    (len <= 32 && u32::from(ip) & host == 0).then(|| format!("{ip}/{len}"))
}

fn max_tokens(v: &str) -> Result<u32, Failure> {
    v.parse::<u32>()
        .ok()
        .filter(|n| (1..=MAX_TOKENS_CAP).contains(n))
        .ok_or_else(|| {
            usage(crate::argv::refused(
                "--max-tokens",
                &format!("from 1 to {MAX_TOKENS_CAP}"),
            ))
        })
}

/// The key itself typed after `--key-file`: refused, never read. A command
/// line is visible to every local process (`ps`) and kept by the shell.
const KEY_ON_ARGV: &str = "chat --key-file takes the path of the relay's key file, not the key: \
     a key on a command line is visible to every local process and kept in shell history";

/// The model key itself typed after `--model-key`: the same refusal.
const NRAS_KEY_ON_ARGV: &str = "--nras-key-file takes the path of the key file, not the key: \
     a key on a command line is visible to every local process and kept in shell history \
     (the value is not printed)";

const MODEL_KEY_ON_ARGV: &str = "--model-key takes the path of the owner's key file, not the key: \
     a key on a command line is visible to every local process and kept in shell history";

/// A flag that names a file the relay will read: the file must exist now, and
/// its path reaches the relay canonical. A missing one is said without the
/// path when that path looks like a key.
fn existing_file(flag: &str, value: String) -> Result<PathBuf, Failure> {
    let p = PathBuf::from(value);
    if !p.is_file() {
        return Err(usage(format!(
            "{flag} {}: no such file",
            crate::argv::path_shown(&p.to_string_lossy())
        )));
    }
    Ok(std::fs::canonicalize(&p).unwrap_or(p))
}

/// `chat [--listen <ip:port>] [--key-file <file>] [--max-tokens <N>]`.
fn parse_chat(rest: &[String]) -> Result<ChatOpts, Failure> {
    let mut c = ChatOpts::default();
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        // Named before its value is looked for: an unknown one is never printed.
        let flag = match a.as_str() {
            f @ ("--listen" | "--key-file" | "--max-tokens") => f,
            _ => {
                return Err(usage(format!(
                    "chat: unknown argument {} (chat takes --listen, --key-file and --max-tokens)",
                    crate::argv::shown(a)
                )))
            }
        };
        let v = it
            .next()
            .ok_or_else(|| usage(format!("chat: {flag} needs a value (see --help)")))?;
        match flag {
            "--listen" => {
                crate::chat::loopback(v).map_err(|_| {
                    usage(crate::argv::refused(
                        "--listen",
                        "a loopback address and port",
                    ))
                })?;
                c.listen = Some(v.clone());
            }
            "--key-file" if crate::argv::is_hex_key(v) => return Err(usage(KEY_ON_ARGV)),
            "--key-file" => c.key_file = Some(PathBuf::from(v)),
            _ => c.max_tokens = max_tokens(v)?,
        }
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_takes_one_catalog_model_and_a_ceiling_with_yes() {
        let Ok(Cmd::Run(o)) = p("run qwen3:0.6b --max-tokens 64") else {
            panic!("{:?}", p("run qwen3:0.6b --max-tokens 64"));
        };
        assert_eq!((o.model.as_str(), o.max_tokens), ("qwen3:0.6b", 64));
        assert_eq!(p("run").unwrap_err().code(), 2);
        assert_eq!(p("run qwen3:0.6b olmo2:1b").unwrap_err().code(), 2);
        assert_eq!(p("run --model qwen3:0.6b").unwrap_err().code(), 2);
        assert_eq!(p("run qwen3:0.6b --yes").unwrap_err().code(), 2);
        assert!(matches!(
            p("run qwen3:0.6b --yes --max-cost 15"),
            Ok(Cmd::Run(_))
        ));
        let e = p("run qwen3:999b").unwrap_err();
        assert!(
            e.code() == 2 && e.message().contains("not in this build's catalog"),
            "{e}"
        );
        assert!(p("up --model qwen3:999b").is_err());
        assert!(p("run qwen3:0.6b --max-tokens 0").is_err());
        assert!(p("run qwen3:0.6b --max-tokens 4097").is_err());
        assert!(p("up --max-tokens 64").is_err());
    }

    /// SECURITY: the TCB policy options reach the relay only stricter than
    /// its build (x0ne-infer `TcbPolicy`); looser or unreadable, a usage
    /// error before any quote. `up` and `run` take them.
    #[test]
    fn the_tcb_policy_options_are_stricter_than_the_relays_floor_or_refused() {
        let Ok(Cmd::Up(o)) = p("up --min-tcb-evaluation 23 --max-collateral-age 3600") else {
            panic!(
                "{:?}",
                p("up --min-tcb-evaluation 23 --max-collateral-age 3600")
            );
        };
        assert_eq!(
            (o.min_tcb_evaluation, o.max_collateral_age),
            (Some(23), Some(3600))
        );
        let Ok(Cmd::Run(o)) = p("run qwen3:0.6b --min-tcb-evaluation 22") else {
            panic!("run");
        };
        assert_eq!(
            (o.min_tcb_evaluation, o.max_collateral_age),
            (Some(22), None)
        );
        assert!(o.tcb_strict(), "a floor given is strict");
        let Ok(Cmd::Up(o)) = p("up") else {
            panic!("up")
        };
        assert_eq!((o.min_tcb_evaluation, o.max_collateral_age), (None, None));
        assert_eq!((o.tcb_policy, o.tcb_strict()), (None, false));
        let Ok(Cmd::Up(o)) = p("up --tcb-policy strict") else {
            panic!("strict")
        };
        assert_eq!((o.tcb_policy, o.tcb_strict()), (Some("strict"), true));
        let Ok(Cmd::Run(o)) = p("run qwen3:0.6b --tcb-policy transparent --max-collateral-age 60")
        else {
            panic!("transparent")
        };
        assert_eq!((o.tcb_policy, o.tcb_strict()), (Some("transparent"), false));
        for (args, says) in [
            ("up --tcb-policy lenient", "--tcb-policy"),
            (
                "up --tcb-policy transparent --min-tcb-evaluation 23",
                "--min-tcb-evaluation refuses only under --tcb-policy strict",
            ),
            ("up --min-tcb-evaluation 21", "22"),
            ("up --max-collateral-age 86401", "86400"),
            ("up --min-tcb-evaluation latest", "--min-tcb-evaluation"),
            (
                "run qwen3:0.6b --max-collateral-age 1d",
                "--max-collateral-age",
            ),
        ] {
            let e = p(args).unwrap_err();
            assert!(e.code() == 2 && e.message().contains(says), "{args}: {e}");
        }
        assert!(
            USAGE.contains("--min-tcb-evaluation <N>")
                && USAGE.contains("--max-collateral-age <s>")
                && USAGE.contains("--tcb-policy <mode>")
        );
    }

    /// The TCB policy options outside `up` and `run`, which start no relay,
    /// are a usage error: taken there, they would change nothing.
    #[test]
    fn the_tcb_policy_options_are_refused_where_no_relay_starts() {
        for cmd in ["quote", "down", "sweep", "image import disk.tar.gz"] {
            for (opt, says) in [
                ("--min-tcb-evaluation 23", "--min-tcb-evaluation"),
                ("--max-collateral-age 3600", "--max-collateral-age"),
                ("--tcb-policy strict", "--tcb-policy"),
            ] {
                let e = p(&format!("{cmd} {opt}")).unwrap_err();
                assert!(
                    e.code() == 2
                        && e.message().contains(says)
                        && e.message().contains("up and run only"),
                    "{cmd} {opt}: {e}"
                );
            }
            assert!(p(cmd).is_ok(), "control: {cmd}");
        }
    }

    /// `--allow-from`: IPv4 networks, canonical, repeatable, for `up` and
    /// `run` only, which create the rule; a refused value is never printed.
    #[test]
    fn allow_from_takes_ipv4_networks_on_up_and_run_only() {
        let Ok(Cmd::Up(o)) =
            p("up --allow-from 192.0.2.0/24 --allow-from 198.51.100.7 --allow-from 0.0.0.0/0")
        else {
            panic!("up")
        };
        assert_eq!(
            o.allow_from,
            vec!["192.0.2.0/24", "198.51.100.7/32", "0.0.0.0/0"]
        );
        let Ok(Cmd::Run(o)) = p("run qwen3:0.6b --allow-from 203.0.113.128/25") else {
            panic!("run")
        };
        assert_eq!(o.allow_from, vec!["203.0.113.128/25"]);
        let Ok(Cmd::Up(o)) = p("up") else {
            panic!("up")
        };
        assert!(o.allow_from.is_empty());
        for bad in [
            "192.0.2.1/24",
            "192.0.2.0/33",
            "2001:db8::/32",
            "192.0.2",
            "192.0.2.0/",
            "/24",
            "example.com",
            "192.0.2.0/-1",
            "5ec75ec75ec75ec75ec75ec75ec75ec7",
        ] {
            let e = p(&format!("up --allow-from {bad}")).unwrap_err();
            assert!(
                e.code() == 2 && e.message().contains("--allow-from") && !e.message().contains(bad),
                "{bad}: {e}"
            );
        }
        for cmd in ["quote", "down", "sweep"] {
            let e = p(&format!("{cmd} --allow-from 192.0.2.0/24")).unwrap_err();
            assert!(
                e.code() == 2 && e.message().contains("up and run only"),
                "{cmd}: {e}"
            );
        }
        assert!(p("up --allow-from").is_err());
        assert!(USAGE.contains("--allow-from <CIDR>"));
    }

    /// `--nras-key-file` is read with the command line, for `up` and `run`;
    /// the key it holds is in neither the options' Debug nor a refusal.
    #[test]
    fn the_nras_key_file_is_read_on_up_and_run() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("k");
        std::fs::write(&f, "nvapi-CANARY0123\n").unwrap();
        std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o600)).unwrap();
        let f = f.to_str().unwrap();
        for line in [
            format!("up --nras-key-file {f}"),
            format!("run qwen3:0.6b --nras-key-file {f}"),
        ] {
            let o = match p(&line) {
                Ok(Cmd::Up(o) | Cmd::Run(o)) => o,
                other => panic!("{line}: {other:?}"),
            };
            assert_eq!(o.nras_key.as_ref().unwrap().expose(), "nvapi-CANARY0123");
            assert!(!format!("{o:?}").contains("CANARY"));
        }
        let e = p(&format!("down --nras-key-file {f}")).unwrap_err();
        assert!(
            e.code() == 2 && e.message().contains("up and run only"),
            "{e}"
        );
        assert!(USAGE.contains("--nras-key-file <file>"));
    }

    #[test]
    fn chat_takes_its_three_options_only() {
        assert_eq!(p("chat").unwrap(), Cmd::Chat(ChatOpts::default()));
        let Ok(Cmd::Chat(c)) = p("chat --listen 127.0.0.1:9000 --key-file /k --max-tokens 9")
        else {
            panic!()
        };
        assert_eq!(c.listen.as_deref(), Some("127.0.0.1:9000"));
        assert_eq!(c.max_tokens, 9);
        assert!(p("chat --listen 203.0.113.7:8478").is_err());
        assert!(p("chat --yes").is_err());
        assert!(p("chat --listen").is_err());
    }

    fn p(s: &str) -> Result<Cmd, Failure> {
        parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>())
    }

    /// `--model-key` and `--key-policy` go together, name files, and are
    /// for `up` and `run` only; the paths reach the relay as given (canonical).
    #[test]
    fn a_model_key_and_its_policy_go_together_on_up_and_run() {
        let dir = tempfile::tempdir().unwrap();
        let (k, pol) = (dir.path().join("k"), dir.path().join("policy.json"));
        std::fs::write(&k, "00".repeat(32)).unwrap();
        std::fs::write(&pol, "{}").unwrap();
        let line = |cmd: &str| {
            format!(
                "{cmd} --model-key {} --key-policy {}",
                k.display(),
                pol.display()
            )
        };
        let Ok(Cmd::Up(o)) = p(&line("up")) else {
            panic!("{:?}", p(&line("up")));
        };
        assert_eq!(
            o.model_key,
            Some((
                std::fs::canonicalize(&k).unwrap(),
                std::fs::canonicalize(&pol).unwrap()
            ))
        );
        assert!(matches!(p(&line("run qwen3:0.6b")), Ok(Cmd::Run(o)) if o.model_key.is_some()));
        assert!(matches!(p("up"), Ok(Cmd::Up(o)) if o.model_key.is_none()));
        for bad in [
            format!("up --model-key {}", k.display()),
            format!("up --key-policy {}", pol.display()),
            format!("up --model-key {} --key-policy /nonexistent/p", k.display()),
            format!(
                "up --model-key /nonexistent/k --key-policy {}",
                pol.display()
            ),
            line("quote"),
            line("down"),
        ] {
            let e = p(&bad).unwrap_err();
            assert_eq!(e.code(), 2, "{bad}: {e}");
        }
        // The key itself typed after --model-key is refused, never read as a path.
        let e = p(&format!(
            "up --model-key {} --key-policy {}",
            "5ec7".repeat(16),
            pol.display()
        ))
        .unwrap_err();
        assert_eq!(e.code(), 2);
        assert!(e.message().contains("not the key"), "{e}");
        assert!(!format!("{e:?}").contains("5ec75ec7"), "{e:?}");
    }

    #[test]
    fn up_yes_without_a_ceiling_is_refused() {
        assert_eq!(p("up --yes").unwrap_err().code(), 2);
        assert!(matches!(p("up --yes --max-cost 10"), Ok(Cmd::Up(o)) if o.max_cost == Some(10.0)));
        assert!(matches!(p("sweep --yes"), Ok(Cmd::Sweep(o)) if o.yes));
    }

    /// --help says what a token `up` cannot renew does to --max-run, and why.
    #[test]
    fn the_help_says_a_pasted_token_cuts_max_run() {
        let row: String = USAGE
            .split("\n  --")
            .find(|r| r.starts_with("max-run "))
            .expect("the --max-run row")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let margin = crate::TOKEN_MARGIN_SECS / 60;
        assert!(
            row.contains(&format!(
                "with --auth paste or stdin, the session ends {margin} min before the token \
                 expires, to delete the VM with it"
            )),
            "{row}"
        );
        assert!(
            row.contains("too short for the model's boot is refused"),
            "{row}"
        );
    }

    #[test]
    fn the_local_api_stays_on_the_loopback() {
        assert!(p("up --listen 0.0.0.0:8478").is_err());
        assert!(p("up --listen 203.0.113.2:8478").is_err());
        assert!(p("up --listen localhost:8478").is_err());
        assert!(p("up --listen 127.0.0.1:0").is_err());
        assert!(p("up --listen [::1]:9000").is_ok());
    }

    #[test]
    fn durations_and_bounds() {
        assert_eq!(parse_duration("30m"), Some(1800));
        assert_eq!(parse_duration("2h"), Some(7200));
        assert_eq!(parse_duration("90"), Some(5400));
        assert!(p("up --max-run 5m").is_err());
        assert!(p("up --max-run 9h").is_err());
        assert!(matches!(p("quote --max-run 8h"), Ok(Cmd::Quote(o)) if o.max_run_secs == 28_800));
    }

    #[test]
    fn names_that_reach_urls_are_validated() {
        assert!(p("quote --project ../etc").is_err());
        assert!(p("quote --zone us-central1-a/x").is_err());
        assert!(p("quote --model qwen3").is_err());
        assert!(p("quote --provider aws").is_err());
        assert!(
            matches!(p("image import disk.tar.gz --region us-central1"), Ok(Cmd::ImageImport(o, _)) if o.region.as_deref() == Some("us-central1"))
        );
        assert_eq!(p("setup gcp").unwrap(), Cmd::SetupGcp);
        assert!(p("setup aws").is_err());
    }

    /// A key typed on the command line is refused, never echoed.
    #[test]
    fn an_unknown_argument_is_refused_without_its_value() {
        let key = "5ec7".repeat(16);
        for line in [
            format!("run qwen3:0.6b --key-hex={key}"),
            format!("run qwen3:0.6b --key-hex {key}"),
            format!("down {key}"),
            format!("X0NE_INFER_KEY_HEX={key}"),
            key.clone(),
            format!("chat {key}"),
            format!("chat --key-hex={key}"),
            format!("chat --key-hex {key}"),
            format!("chat X0NE_INFER_KEY_HEX={key}"),
            format!("chat {key} 1"),
            format!("chat --listen 127.0.0.1:8478 {key}"),
        ] {
            let e = p(&line).unwrap_err();
            assert_eq!(e.code(), 2, "{line}");
            assert!(!format!("{e:?}").contains("5ec75ec7"), "{line}: {e:?}");
        }
    }

    /// A refused value is not printed either: a key or a token pasted after
    /// the wrong flag, or where the model goes, stays off the terminal. The
    /// key itself after `chat --key-file` is refused as a key.
    #[test]
    fn a_refused_value_is_not_printed() {
        let key = "5ec7".repeat(16);
        for line in [
            format!("run {key}"),
            format!("up --model {key}"),
            format!("up --provider {key}"),
            format!("up --project {key}"),
            format!("up --zone {key}"),
            format!("up --region {key}"),
            format!("up --max-run {key}"),
            format!("up --auth {key}"),
            format!("up --listen {key}"),
            format!("up --reference /nonexistent/{key}"),
            format!("up --max-cost {key}"),
            format!("run qwen3:0.6b --max-tokens {key}"),
            format!("chat --listen {key}"),
            format!("chat --max-tokens {key}"),
            format!("chat --key-file {key}"),
        ] {
            let e = p(&line).unwrap_err();
            assert_eq!(e.code(), 2, "{line}");
            assert!(!format!("{e:?}").contains("5ec75ec7"), "{line}: {e:?}");
        }
        let e = p(&format!("chat --key-file {key}")).unwrap_err();
        assert!(e.message().contains("not the key"), "{e}");
    }

    /// --help says what the catalog serves: a model whose served digest is
    /// pinned, sharded or not (`catalog::a_sharded_entry_is_served_only_with_a_pinned_digest`).
    /// Until 2026-10-30 it said "in one weight file".
    #[test]
    fn the_help_says_a_sharded_model_is_served() {
        let row: String = USAGE
            .split("\n  --")
            .find(|r| r.starts_with("model "))
            .expect("the --model row")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        assert!(!row.contains("one weight file"), "{row}");
        assert!(row.contains("sharded or not"), "{row}");
        assert!(row.contains("pinned served-model digest"), "{row}");
    }
}
