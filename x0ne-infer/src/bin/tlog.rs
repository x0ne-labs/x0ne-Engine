//! `x0ne-engine-tlog`: runs the transparency log of image references on
//! the operator's machine (`x0ne_infer::tlog_ops`). Nothing here talks to
//! the network: the witness request is sent with any HTTP client, and its
//! answer read back on standard input.

use std::io::Read as _;
use std::os::unix::fs::DirBuilderExt as _;
use std::path::Path;
use x0ne_infer::tlog::{KeyKind, NoteKey};
use x0ne_infer::tlog_ops::{self, Log, PinFile, Secret};
use x0ne_infer::transparency;

const USAGE: &str = "\
x0ne-engine-tlog — run the transparency log of x0ne image references.

USAGE:
  x0ne-engine-tlog keygen log|witness <name> <secret-file>
      A new key: the secret goes to <secret-file> (created, mode 0600, never
      replaced); the verifier key and its fingerprint are printed, for paper.
  x0ne-engine-tlog fingerprint <verifier-key>
  x0ne-engine-tlog append <log-dir> <log-secret-file> <reference>...
      Logs each reference (its SHA-256, x0ne-engine/image-reference/v1),
      writes the tiles, then the checkpoint signed by the log alone.
  x0ne-engine-tlog witness-request <log-dir> <old-size>
      Prints the tlog-witness add-checkpoint body, from the size the witness
      last cosigned (0 the first time). POST it to <witness>/add-checkpoint.
  x0ne-engine-tlog cosign <witness-secret-file> <log-verifier-key> <state-file>
      A witness: reads an add-checkpoint body on standard input, prints its
      cosignature line. For a witness x0ne runs, or a rehearsal.
  x0ne-engine-tlog add-cosignatures <log-dir> <witness-verifier-key>...
      Reads cosignature lines on standard input; each must verify under one
      of the keys given, or nothing is added. A witness's newer cosignature
      replaces its older one (the refresh: witness-request <dir> <size>).
  x0ne-engine-tlog prove <log-dir> <reference>
      Writes <reference>.tlog-proof against the cosigned checkpoint.
  x0ne-engine-tlog verify <log.pin> <reference> [<log-dir>]
      The relay's check of <reference>.tlog-proof under the pin file, the
      log's files read from <log-dir> or from the pinned https location.

EXIT: 0 done, 1 refused (the reason on standard error, nothing written),
2 usage.
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    match run(&words) {
        Ok(out) => print!("{out}"),
        Err(Fail::Usage) => {
            eprint!("{USAGE}");
            std::process::exit(2);
        }
        Err(Fail::Refused(e)) => {
            eprintln!("x0ne-engine-tlog: {e}");
            std::process::exit(1);
        }
    }
}

enum Fail {
    Usage,
    Refused(String),
}

impl From<String> for Fail {
    fn from(e: String) -> Self {
        Fail::Refused(e)
    }
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn stdin() -> Result<String, String> {
    let mut s = String::new();
    std::io::stdin()
        .take(transparency::MAX_PROOF_BYTES)
        .read_to_string(&mut s)
        .map_err(|e| format!("standard input: {e}"))?;
    Ok(s)
}

fn reference_entry(path: &str) -> Result<Vec<u8>, String> {
    let bytes = x0ne_infer::fsread::read_capped(Path::new(path), x0ne_infer::reference::MAX_BYTES)
        .map_err(|e| format!("{path}: {e}"))?;
    Ok(transparency::entry(&bytes))
}

fn run(args: &[&str]) -> Result<String, Fail> {
    match args {
        ["keygen", kind, name, file] => {
            let kind = match *kind {
                "log" => KeyKind::Ed25519,
                "witness" => KeyKind::CosignatureV1,
                _ => return Err(Fail::Refused(format!("{kind:?}: log or witness"))),
            };
            let secret = Secret::generate(name, kind)?;
            secret.write_new(Path::new(file))?;
            let vkey = secret.key.vkey();
            Ok(format!(
                "vkey {vkey}\nfingerprint {}\nsecret {file} (mode 0600)\n",
                tlog_ops::fingerprint(&vkey)
            ))
        }
        ["fingerprint", vkey] => {
            NoteKey::parse(vkey)?;
            Ok(format!("{}\n", tlog_ops::fingerprint(vkey)))
        }
        ["append", dir, secret, refs @ ..] if !refs.is_empty() => {
            let secret = Secret::read(Path::new(secret))?;
            let mut log = Log::open(Path::new(dir))?;
            let entries = refs
                .iter()
                .map(|r| reference_entry(r))
                .collect::<Result<Vec<_>, _>>()?;
            let (indexes, note) = log.append(&secret, &entries).map_err(|e| {
                let named = refs.iter().enumerate().fold(e, |e, (i, r)| {
                    e.replace(&format!("entry {} ", i + 1), &format!("{r} "))
                });
                Fail::Refused(named)
            })?;
            let mut out = String::new();
            for (r, i) in refs.iter().zip(indexes) {
                out.push_str(&format!("{r}: index {i}\n"));
            }
            out.push_str(&format!("checkpoint, signed by the log alone:\n{note}"));
            Ok(out)
        }
        ["witness-request", dir, old] => {
            let old = x0ne_infer::tlog::decimal(old)?;
            Ok(Log::open(Path::new(dir))?.witness_request(old)?)
        }
        ["cosign", secret, log_vkey, state] => {
            let secret = Secret::read(Path::new(secret))?;
            let log = NoteKey::parse(log_vkey)?;
            Ok(tlog_ops::cosign(
                &secret,
                &log,
                Path::new(state),
                &stdin()?,
                now(),
            )?)
        }
        ["add-cosignatures", dir, keys @ ..] if !keys.is_empty() => {
            let keys = keys
                .iter()
                .map(|k| NoteKey::parse(k))
                .collect::<Result<Vec<_>, _>>()?;
            let mut log = Log::open(Path::new(dir))?;
            let n = log.add_cosignatures(&keys, &stdin()?)?;
            Ok(format!("{n} cosignature(s) added\n"))
        }
        ["prove", dir, reference] => {
            let log = Log::open(Path::new(dir))?;
            let (index, proof) = log.proof(&reference_entry(reference)?)?;
            let path = transparency::proof_path(Path::new(reference));
            std::fs::write(&path, proof).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(format!("{}: index {index}\n", path.display()))
        }
        ["verify", pin, reference, rest @ ..] if rest.len() <= 1 => {
            let text = x0ne_infer::fsread::read_capped(Path::new(pin), 64 * 1024)
                .map_err(|e| format!("{pin}: {e}"))?;
            let pin = PinFile::parse(&String::from_utf8_lossy(&text))?;
            // A new private directory: the relay's last-seen state, empty.
            let state = std::env::temp_dir().join(format!(
                "x0ne-engine-tlog-verify.{}.{}",
                std::process::id(),
                now()
            ));
            std::fs::DirBuilder::new()
                .mode(0o700)
                .create(&state)
                .map_err(|e| format!("{}: {e}", state.display()))?;
            let logged = tlog_ops::verify(
                &pin,
                Path::new(reference),
                rest.first().map(Path::new),
                &state,
                now(),
            );
            let _ = std::fs::remove_dir_all(&state);
            let l = logged.map_err(|e| format!("not logged: {e}"))?;
            Ok(format!(
                "logged: {} index {} of {}, root {}, cosigned by {}{}\n",
                l.origin,
                l.index,
                l.size,
                l.root_hex,
                l.cosigned_by
                    .iter()
                    .map(|(w, t)| format!("{w} at {t}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                if l.refreshed { " (refreshed)" } else { "" }
            ))
        }
        _ => Err(Fail::Usage),
    }
}
