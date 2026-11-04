//! `x0ne-engine`: pull, list, verify and remove models from the x0ne catalog,
//! and the `cloud` commands. This binary serves nothing itself.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use x0ne_infer::registry::{self, Catalog, Progress, PullError, Source, Store};

const USAGE: &str = "\
x0ne-engine — fetch verified open models; confidential GPU VMs.

USAGE:
  x0ne-engine pull <name>             download, verify (sha256), install
  x0ne-engine list                    installed models
  x0ne-engine catalog                 models this build can pull
  x0ne-engine verify <name>           re-hash an installed model
  x0ne-engine rm <name>               remove an installed model
  x0ne-engine path <name>             print an installed model's directory
  x0ne-engine cloud <command>         a confidential GPU VM for one session, its
                                      proof checked on this machine
                                      (x0ne-engine cloud --help)
  x0ne-engine cloud run <name>        the same, with a chat in this terminal
  x0ne-engine snp-measure --ovmf <f> --vcpus <n> [--kernel <f> ...]
                                      the SEV-SNP launch measurement of a QEMU
                                      guest, computed offline
                                      (x0ne-engine snp-measure --help)

GLOBAL OPTIONS (before the command):
  --catalog <file>     use this catalog instead of the embedded one
                       (same validation; also X0NE_ENGINE_CATALOG)
  -h, --help           this text
  -V, --version

ENVIRONMENT:
  X0NE_ENGINE_HOME           cache root [default: $XDG_CACHE_HOME/x0ne-engine,
                             else ~/.cache/x0ne-engine]
  X0NE_ENGINE_HF_ENDPOINT    download source [default: https://huggingface.co];
                             https only, plain http to the loopback only

NAMES are family:tag, e.g. qwen3:0.6b. Each pins a Hugging Face commit and the
sha256 of every file; a file whose bytes differ is deleted and nothing is
installed. Only safetensors weights, JSON and licence files are accepted.

EXIT CODES:
  0 ok · 1 not installed / not found · 2 usage · 65 integrity (sha256, catalog,
  remote code) · 69 network (pull again to resume) · 74 local I/O
";

fn fail(code: u8, msg: impl std::fmt::Display) -> ExitCode {
    eprintln!("x0ne-engine: {msg}");
    ExitCode::from(code)
}

fn pull_code(e: &PullError) -> u8 {
    match e {
        PullError::Integrity(_) => 65,
        PullError::Network(_) => 69,
        PullError::Local(_) => 74,
    }
}

/// Percent lines on stderr: in place on a terminal, every 10 % otherwise.
struct Bar {
    tty: bool,
    /// Last step shown, per file: files now report in turns.
    last: std::collections::HashMap<String, u64>,
}
impl Progress for Bar {
    fn file(&mut self, path: &str, done: u64, total: u64) {
        let pct = (done * 100).checked_div(total).unwrap_or(100);
        let step = if self.tty { pct } else { pct / 10 * 10 };
        if self.last.insert(path.to_string(), step) == Some(step) {
            return;
        }
        let line = format!("  {path:<36} {:>9} {pct:>3}%", human(total));
        if self.tty {
            eprint!("\r{line}");
            if done >= total {
                eprintln!();
            }
        } else {
            eprintln!("{line}");
        }
    }
    fn note(&mut self, msg: &str) {
        eprintln!("x0ne-engine: {msg}");
    }
}

fn human(b: u64) -> String {
    const U: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < U.len() {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{b} B")
    } else {
        format!("{v:.1} {}", U[i])
    }
}

struct Ctx {
    catalog: Catalog,
    store: Store,
}

fn context(catalog_path: Option<PathBuf>) -> Result<Ctx, ExitCode> {
    let catalog = match catalog_path {
        None => Catalog::embedded(),
        Some(p) => {
            let bytes = x0ne_infer::fsread::read_capped(&p, x0ne_infer::fsread::CONFIG_CAP)
                .map_err(|e| fail(74, format!("{}: {e}", p.display())))?;
            let text =
                String::from_utf8(bytes).map_err(|e| fail(74, format!("{}: {e}", p.display())))?;
            Catalog::parse(&text).map_err(|e| fail(65, format!("{}: {e}", p.display())))?
        }
    };
    let root = registry::store::resolve_root(|k| std::env::var(k).ok()).map_err(|e| fail(74, e))?;
    Ok(Ctx {
        catalog,
        store: Store::new(root),
    })
}

fn source() -> Result<Source, ExitCode> {
    let endpoint = std::env::var("X0NE_ENGINE_HF_ENDPOINT")
        .unwrap_or_else(|_| registry::DEFAULT_ENDPOINT.to_string());
    Source::new(&endpoint).map_err(|e| fail(2, e))
}

async fn do_pull(ctx: &Ctx, name: &str) -> Result<registry::Installed, ExitCode> {
    let entry = ctx.catalog.get(name).map_err(|e| fail(1, e))?;
    let source = source()?;
    eprintln!(
        "x0ne-engine: {} = {}@{} ({}, {}, {})",
        entry.name,
        entry.repo,
        &entry.revision[..12],
        entry.architecture,
        entry.license,
        human(entry.total_bytes())
    );
    let mut bar = Bar {
        tty: std::io::IsTerminal::is_terminal(&std::io::stderr()),
        last: std::collections::HashMap::default(),
    };
    match registry::pull(&ctx.store, &source, entry, &mut bar).await {
        Ok(p) => {
            if p.already {
                eprintln!(
                    "x0ne-engine: {} is up to date (sha256 of {} files re-checked)",
                    entry.name,
                    p.installed.record.files.len()
                );
            } else {
                eprintln!(
                    "x0ne-engine: {} verified (sha256 of {} files) and installed, {} downloaded",
                    entry.name,
                    entry.files.len(),
                    human(p.downloaded_bytes)
                );
            }
            Ok(p.installed)
        }
        Err(e) => Err(fail(pull_code(&e), format!("pull {}: {e}", entry.name))),
    }
}

/// `name` split, or a refusal that does not print it: a key typed where the
/// model name goes stays off the terminal.
fn model_name(name: &str) -> Result<(&str, &str), ExitCode> {
    registry::catalog::split_name(name).ok_or_else(|| {
        fail(
            2,
            x0ne_infer::argv::refused("the model name", "family:tag, e.g. qwen3:0.6b"),
        )
    })
}

fn installed(ctx: &Ctx, name: &str) -> Result<registry::Installed, ExitCode> {
    let (family, tag) = model_name(name)?;
    match ctx.store.installed(family, tag) {
        Ok(Some(i)) => Ok(i),
        Ok(None) => Err(fail(
            1,
            format!("{name} is not installed (x0ne-engine pull {name})"),
        )),
        Err(e) => Err(fail(74, e)),
    }
}

const CLOUD_BIN: &str = "x0ne-engine-cloud";
const CLOUD_LIBEXEC: &str = "/usr/libexec/x0ne-engine";

/// Next to this binary, then the libexec directory; never `$PATH`. Refused
/// if group or others can write it: it is handed the cloud token.
fn cloud_binary_in(exe_dir: Option<&Path>, libexec: &Path) -> Option<PathBuf> {
    exe_dir
        .map(|d| d.join(CLOUD_BIN))
        .into_iter()
        .chain([libexec.join(CLOUD_BIN)])
        .find(|p| {
            std::fs::metadata(p).is_ok_and(|m| {
                #[cfg(unix)]
                let ok = {
                    use std::os::unix::fs::PermissionsExt;
                    let mode = m.permissions().mode();
                    mode & 0o111 != 0 && mode & 0o022 == 0
                };
                #[cfg(not(unix))]
                let ok = true;
                m.is_file() && ok
            })
        })
}

/// `cloud …` runs in its own process (x0ne-engine-cloud), which holds the token.
fn cloud(rest: &[String]) -> ExitCode {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let Some(bin) = cloud_binary_in(exe_dir.as_deref(), Path::new(CLOUD_LIBEXEC)) else {
        return fail(
            91,
            format!(
                "cloud: {CLOUD_BIN} not found next to this binary nor in {CLOUD_LIBEXEC} \
                 (never looked up in $PATH)"
            ),
        );
    };
    let mut cmd = std::process::Command::new(&bin);
    cmd.args(rest);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let e = cmd.exec(); // returns only on failure
        fail(91, format!("{}: {e}", bin.display()))
    }
    #[cfg(not(unix))]
    match cmd.status() {
        Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(e) => fail(91, format!("{}: {e}", bin.display())),
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut catalog_path = std::env::var_os("X0NE_ENGINE_CATALOG").map(PathBuf::from);
    while let Some(first) = args.first().cloned() {
        match first.as_str() {
            "-h" | "--help" | "help" => {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "-V" | "--version" => {
                println!("x0ne-engine {}{}", env!("CARGO_PKG_VERSION"), git_suffix());
                return ExitCode::SUCCESS;
            }
            "--catalog" => {
                if args.len() < 2 {
                    return fail(2, "--catalog needs a file");
                }
                catalog_path = Some(PathBuf::from(&args[1]));
                args.drain(..2);
            }
            _ => break,
        }
    }
    let Some(cmd) = args.first().cloned() else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    let rest = &args[1..];
    if cmd == "cloud" {
        return cloud(rest);
    }
    if cmd == "snp-measure" {
        return snp_measure(rest);
    }
    if rest.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let ctx = match context(catalog_path) {
        Ok(c) => c,
        Err(code) => return code,
    };
    let one = |what: &str| -> Result<String, ExitCode> {
        match rest {
            [n] => model_name(n).map(|_| n.clone()),
            _ => Err(fail(2, format!("{what} takes exactly one model name"))),
        }
    };
    let result: Result<(), ExitCode> = async {
        match cmd.as_str() {
            "pull" => {
                let name = one("pull")?;
                let i = do_pull(&ctx, &name).await?;
                println!("{}", i.dir.display());
            }
            // Named refusal: the lab server it started left on 2026-11-04.
            "run" => {
                return Err(fail(
                    2,
                    "run left with the lab server; x0ne-engine cloud run <name> serves in a confidential GPU VM",
                ))
            }
            "list" | "ls" => {
                let all = ctx.store.list();
                if all.is_empty() {
                    eprintln!(
                        "x0ne-engine: nothing installed under {}",
                        ctx.store.root.display()
                    );
                }
                println!(
                    "{:<16} {:<14} {:>10}  {:<8} STATUS",
                    "NAME", "REVISION", "SIZE", "ARCH"
                );
                for item in all {
                    match item {
                        Ok(i) => {
                            let r = &i.record;
                            // Sizes only: re-hashing every model is `verify`'s job.
                            let status = match ctx.catalog.get(&r.name) {
                                _ if ctx.store.check_sizes(&i).is_err() => {
                                    "DAMAGED (x0ne-engine verify, then pull)"
                                }
                                Ok(e) if r.matches(e) => "ok (sizes; verify re-hashes)",
                                Ok(_) => "outdated (pull to update)",
                                Err(_) => "not in this catalog",
                            };
                            let size: u64 = r.files.iter().map(|f| f.size).sum();
                            println!(
                                "{:<16} {:<14} {:>10}  {:<8} {status}",
                                r.name,
                                &r.revision[..12.min(r.revision.len())],
                                human(size),
                                r.architecture
                            );
                        }
                        Err(e) => println!("?  {e}"),
                    }
                }
            }
            "catalog" => {
                println!(
                    "{:<16} {:<36} {:>10}  {:<8} LICENSE",
                    "NAME", "SOURCE", "SIZE", "ARCH"
                );
                for e in &ctx.catalog.models {
                    println!(
                        "{:<16} {:<36} {:>10}  {:<8} {}",
                        e.name,
                        e.repo,
                        human(e.total_bytes()),
                        e.architecture,
                        e.license
                    );
                }
            }
            "verify" => {
                let name = one("verify")?;
                let i = installed(&ctx, &name)?;
                ctx.store.verify(&i).map_err(|e| fail(65, e))?;
                if let Ok(e) = ctx.catalog.get(&name) {
                    if !i.record.matches(e) {
                        eprintln!("x0ne-engine: {name} is intact but outdated (pull to update)");
                    }
                }
                eprintln!(
                    "x0ne-engine: {name}: {} files match their sha256",
                    i.record.files.len()
                );
            }
            "rm" => {
                let name = one("rm")?;
                let (family, tag) = model_name(&name)?;
                let entry = ctx.catalog.get(&name).ok();
                match ctx.store.remove(family, tag, entry) {
                    Ok(true) => eprintln!("x0ne-engine: removed {name}"),
                    Ok(false) => return Err(fail(1, format!("{name} is not installed"))),
                    Err(e) => return Err(fail(74, format!("rm {name}: {e}"))),
                }
            }
            "path" => {
                let name = one("path")?;
                println!("{}", installed(&ctx, &name)?.dir.display());
            }
            other => {
                let other = x0ne_infer::argv::shown(other);
                return Err(fail(2, format!("unknown command {other} (see --help)")));
            }
        }
        Ok(())
    }
    .await;
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(c) => c,
    }
}

/// `snp-measure`: offline, no catalog, no network. 2 usage, 65 a firmware
/// that cannot measure what it is given, 74 a file unreadable.
fn snp_measure(rest: &[String]) -> ExitCode {
    use x0ne_infer::snp_launch;
    if rest.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", snp_launch::MEASURE_USAGE);
        return ExitCode::SUCCESS;
    }
    let a = match snp_launch::parse_measure_args(rest) {
        Ok(a) => a,
        Err(e) => return fail(2, format!("snp-measure: {e}")),
    };
    for p in [Some(&a.ovmf), a.kernel.as_ref(), a.initrd.as_ref()]
        .into_iter()
        .flatten()
    {
        if !p.is_file() {
            return fail(
                74,
                format!("snp-measure: {} is not a readable file", p.display()),
            );
        }
    }
    match snp_launch::measure_files(&a) {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(e) => fail(65, format!("snp-measure: {e}")),
    }
}

/// `" (git <sha>)"` when the package build set `X0NE_GIT_SHA` (ci/inside-deb.sh), else empty.
fn git_suffix() -> String {
    option_env!("X0NE_GIT_SHA")
        .map(|s| format!(" (git {s})"))
        .unwrap_or_default()
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn exe(dir: &Path, mode: u32) -> PathBuf {
        let p = dir.join(CLOUD_BIN);
        std::fs::write(&p, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
        p
    }

    #[test]
    fn the_cloud_binary_is_found_next_to_us_then_in_libexec_only() {
        let sibling = tempfile::tempdir().unwrap();
        let libexec = tempfile::tempdir().unwrap();
        assert_eq!(cloud_binary_in(Some(sibling.path()), libexec.path()), None);
        let l = exe(libexec.path(), 0o755);
        assert_eq!(
            cloud_binary_in(Some(sibling.path()), libexec.path()),
            Some(l.clone())
        );
        let s = exe(sibling.path(), 0o755);
        assert_eq!(
            cloud_binary_in(Some(sibling.path()), libexec.path()),
            Some(s)
        );
        assert_eq!(cloud_binary_in(None, libexec.path()), Some(l));
    }

    #[test]
    fn a_writable_or_non_executable_cloud_binary_is_refused() {
        let sibling = tempfile::tempdir().unwrap();
        let none = tempfile::tempdir().unwrap();
        exe(sibling.path(), 0o775);
        assert_eq!(cloud_binary_in(Some(sibling.path()), none.path()), None);
        exe(sibling.path(), 0o644);
        assert_eq!(cloud_binary_in(Some(sibling.path()), none.path()), None);
    }
}
