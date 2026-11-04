//! The cloud token: pasted (hidden), piped, or from gcloud at a fixed path.
//! Never in the environment, argv, a file or a log. This crate reads and
//! keeps it in `Zeroizing` buffers, and builds the `Authorization` line in
//! one; the copies made from them are not zeroized: the `HeaderValue`
//! (marked sensitive, which does not clear it), the tokeninfo form body
//! (`http.rs`, a plain `Vec`), and the buffers of reqwest, hyper and rustls.
//! No page is locked (no `mlock`): any of them can reach swap. The process
//! is made non-dumpable before any token exists.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::{Duration, Instant};

use zeroize::Zeroizing;

use crate::Failure;

/// The only places gcloud is looked for; `$PATH` is never searched.
pub const GCLOUD_PATHS: [&str; 2] = ["/usr/bin/gcloud", "/usr/lib/google-cloud-sdk/bin/gcloud"];
/// Longest token accepted: twice the 2048 bytes Google documents for access tokens.
pub const MAX_TOKEN: usize = 4096;

pub trait TokenSource {
    /// A bearer token for the provider's API. May prompt the first time.
    fn token(&mut self) -> Result<Zeroizing<String>, Failure>;
    /// Whether a fresh token can be had without the user (gcloud).
    fn refreshable(&self) -> bool;
    fn label(&self) -> &'static str;
}

/// prctl(PR_SET_DUMPABLE, 0) as ssh-agent does, no core file, umask 077.
pub fn harden_process() -> io::Result<()> {
    // SAFETY: constant arguments; no memory is shared with the kernel.
    unsafe {
        if libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0) != 0 {
            return Err(io::Error::last_os_error());
        }
        let none = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if libc::setrlimit(libc::RLIMIT_CORE, &none) != 0 {
            return Err(io::Error::last_os_error());
        }
        libc::umask(0o077);
    }
    Ok(())
}

pub fn is_dumpable() -> bool {
    // SAFETY: read-only query.
    unsafe { libc::prctl(libc::PR_GET_DUMPABLE, 0, 0, 0, 0) != 0 }
}

/// Bytes of a token: what Google's tokens use, nothing that could end a header.
fn token_from_bytes(raw: &[u8]) -> Result<Zeroizing<String>, Failure> {
    let start = raw
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(raw.len());
    let end = raw
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .map_or(start, |e| e + 1);
    let t = &raw[start..end];
    if t.len() < 20 {
        return Err(Failure::Preflight(
            "the token is empty or too short (gcloud auth print-access-token prints one)".into(),
        ));
    }
    if t.len() > MAX_TOKEN {
        return Err(Failure::Preflight(format!(
            "the token is longer than {MAX_TOKEN} bytes"
        )));
    }
    if !t.iter().all(|b| {
        b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'~' | b'+' | b'/' | b'=')
    }) {
        return Err(Failure::Preflight(
            "the token has characters an access token does not have".into(),
        ));
    }
    let mut s = Zeroizing::new(String::with_capacity(t.len()));
    for b in t {
        s.push(*b as char);
    }
    Ok(s)
}

/// Reads until EOF into a buffer that never reallocates (no stray copies).
fn read_capped(mut r: impl Read, cap: usize) -> io::Result<Zeroizing<Vec<u8>>> {
    let mut buf = Zeroizing::new(Vec::with_capacity(cap));
    let mut chunk = Zeroizing::new([0u8; 512]);
    loop {
        let n = match r.read(&mut chunk[..]) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        if buf.len() + n > cap {
            return Err(io::Error::other("input too long"));
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    Ok(buf)
}

// ── the terminal ────────────────────────────────────────────────────────────

/// The controlling terminal (`/dev/tty`), whatever stdin and stdout are.
pub struct Tty {
    file: File,
}

static RESTORE_FD: AtomicI32 = AtomicI32::new(-1);
static mut RESTORE_TERMIOS: std::mem::MaybeUninit<libc::termios> = std::mem::MaybeUninit::uninit();

extern "C" fn restore_and_reraise(sig: libc::c_int) {
    let fd = RESTORE_FD.load(Ordering::SeqCst);
    // SAFETY: tcsetattr, write, signal and raise are async-signal-safe;
    // RESTORE_TERMIOS is written before RESTORE_FD is published.
    unsafe {
        if fd >= 0 {
            libc::tcsetattr(
                fd,
                libc::TCSANOW,
                std::ptr::addr_of!(RESTORE_TERMIOS).cast(),
            );
            libc::write(fd, b"\n".as_ptr().cast(), 1);
        }
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }
}

const RESTORE_SIGNALS: [libc::c_int; 4] =
    [libc::SIGINT, libc::SIGTERM, libc::SIGHUP, libc::SIGQUIT];

/// Echo off for the time of one read; a signal restores the terminal first.
struct EchoOff {
    fd: i32,
    old: libc::termios,
    old_actions: Vec<(libc::c_int, libc::sigaction)>,
}

impl EchoOff {
    fn new(fd: i32) -> io::Result<Self> {
        // SAFETY: termios is plain data; tcgetattr fills it.
        let mut old: libc::termios = unsafe { std::mem::zeroed() };
        if unsafe { libc::tcgetattr(fd, &mut old) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: single writer, before the handlers can observe it.
        unsafe {
            std::ptr::addr_of_mut!(RESTORE_TERMIOS).write(std::mem::MaybeUninit::new(old));
        }
        RESTORE_FD.store(fd, Ordering::SeqCst);
        let mut old_actions = Vec::new();
        for sig in RESTORE_SIGNALS.into_iter().chain([libc::SIGTSTP]) {
            // SAFETY: sigaction with a valid handler or SIG_IGN.
            unsafe {
                let mut sa: libc::sigaction = std::mem::zeroed();
                sa.sa_sigaction = if sig == libc::SIGTSTP {
                    libc::SIG_IGN
                } else {
                    restore_and_reraise as *const () as usize
                };
                libc::sigemptyset(&mut sa.sa_mask);
                let mut prev: libc::sigaction = std::mem::zeroed();
                if libc::sigaction(sig, &sa, &mut prev) == 0 {
                    old_actions.push((sig, prev));
                }
            }
        }
        let mut new = old;
        new.c_lflag &= !(libc::ECHO | libc::ECHOE | libc::ECHOK);
        new.c_lflag |= libc::ECHONL;
        // TCSAFLUSH: what was typed before the prompt is dropped, as getpass does.
        if unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, &new) } != 0 {
            let e = io::Error::last_os_error();
            let _ = EchoOff {
                fd,
                old,
                old_actions,
            };
            return Err(e);
        }
        Ok(EchoOff {
            fd,
            old,
            old_actions,
        })
    }
}

impl Drop for EchoOff {
    fn drop(&mut self) {
        // SAFETY: restoring what tcgetattr/sigaction returned.
        unsafe {
            libc::tcsetattr(self.fd, libc::TCSANOW, &self.old);
            for (sig, prev) in &self.old_actions {
                libc::sigaction(*sig, prev, std::ptr::null_mut());
            }
        }
        RESTORE_FD.store(-1, Ordering::SeqCst);
    }
}

impl Tty {
    pub fn open() -> Option<Tty> {
        OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOCTTY)
            .open("/dev/tty")
            .ok()
            .map(|file| Tty { file })
    }

    pub fn say(&mut self, text: &str) {
        let _ = self.file.write_all(text.as_bytes());
        let _ = self.file.flush();
    }

    /// One line, echo off. The buffer never grows past `max` (no copies).
    pub fn read_hidden(&mut self, prompt: &str, max: usize) -> io::Result<Zeroizing<Vec<u8>>> {
        self.say(prompt);
        let fd = self.file.as_raw_fd();
        let guard = EchoOff::new(fd)?;
        let line = read_line_fd(fd, max);
        drop(guard);
        line
    }

    /// One visible line (the "yes").
    pub fn read_line(&mut self, prompt: &str) -> io::Result<String> {
        self.say(prompt);
        let line = read_line_fd(self.file.as_raw_fd(), 256)?;
        Ok(String::from_utf8_lossy(&line).trim().to_string())
    }
}

fn read_line_fd(fd: i32, max: usize) -> io::Result<Zeroizing<Vec<u8>>> {
    let mut buf = Zeroizing::new(Vec::with_capacity(max));
    let mut chunk = Zeroizing::new([0u8; 512]);
    loop {
        // SAFETY: chunk is a valid writable buffer of its length.
        let n = unsafe { libc::read(fd, chunk.as_mut_ptr().cast(), chunk.len()) };
        if n < 0 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(e);
        }
        if n == 0 {
            if buf.is_empty() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "end of input"));
            }
            break;
        }
        let part = &chunk[..n as usize];
        let (body, done) = match part.iter().position(|&b| b == b'\n') {
            Some(p) => (&part[..p], true),
            None => (part, false),
        };
        if buf.len() + body.len() > max {
            return Err(io::Error::other("line too long"));
        }
        buf.extend_from_slice(body);
        if done {
            break;
        }
    }
    Ok(buf)
}

// ── sources ─────────────────────────────────────────────────────────────────

/// Default: pasted on the terminal, hidden, kept in RAM for the session.
#[derive(Default)]
pub struct Paste {
    cached: Option<Zeroizing<String>>,
    prompt: Option<String>,
}

impl Paste {
    pub fn with_prompt(prompt: &str) -> Self {
        Paste {
            cached: None,
            prompt: Some(prompt.to_string()),
        }
    }
}

impl TokenSource for Paste {
    fn token(&mut self) -> Result<Zeroizing<String>, Failure> {
        if let Some(t) = &self.cached {
            return Ok(t.clone());
        }
        let mut tty = Tty::open().ok_or_else(|| {
            Failure::Preflight(
                "no terminal to paste a token on: use --auth stdin (gcloud auth print-access-token | …) \
                 or --auth gcloud"
                    .into(),
            )
        })?;
        let prompt = self.prompt.clone().unwrap_or_else(|| {
            "Paste a Google Cloud access token (hidden, kept in memory only): ".into()
        });
        let raw = tty
            .read_hidden(&prompt, MAX_TOKEN + 64)
            .map_err(|e| Failure::Preflight(format!("reading the token: {e}")))?;
        let t = token_from_bytes(&raw)?;
        self.cached = Some(t.clone());
        Ok(t)
    }
    fn refreshable(&self) -> bool {
        false
    }
    fn label(&self) -> &'static str {
        "pasted token"
    }
}

/// `--auth stdin`: a pipe, read to EOF once.
#[derive(Default)]
pub struct Stdin {
    cached: Option<Zeroizing<String>>,
}

impl TokenSource for Stdin {
    fn token(&mut self) -> Result<Zeroizing<String>, Failure> {
        if let Some(t) = &self.cached {
            return Ok(t.clone());
        }
        let stdin = io::stdin();
        if io::IsTerminal::is_terminal(&stdin) {
            return Err(Failure::Preflight(
                "--auth stdin reads a pipe (gcloud auth print-access-token | x0ne-engine cloud …); \
                 on a terminal, the default --auth paste hides what you type"
                    .into(),
            ));
        }
        let raw = read_capped(stdin.lock(), MAX_TOKEN + 64)
            .map_err(|e| Failure::Preflight(format!("reading the token from stdin: {e}")))?;
        let t = token_from_bytes(&raw)?;
        self.cached = Some(t.clone());
        Ok(t)
    }
    fn refreshable(&self) -> bool {
        false
    }
    fn label(&self) -> &'static str {
        "token from stdin"
    }
}

/// gcloud at one of [`GCLOUD_PATHS`], or nothing.
pub fn find_gcloud() -> Option<PathBuf> {
    find_gcloud_in(&GCLOUD_PATHS.map(Path::new))
}

pub fn find_gcloud_in(paths: &[&Path]) -> Option<PathBuf> {
    paths.iter().find(|p| p.is_file()).map(|p| p.to_path_buf())
}

/// gcloud runs with a fixed PATH and the user's own gcloud settings only.
fn gcloud_command(path: &Path) -> Command {
    let mut c = Command::new(path);
    c.env_clear();
    c.env("PATH", "/usr/bin:/bin");
    for (k, v) in std::env::vars_os() {
        let Some(k) = k.to_str() else { continue };
        let keep = matches!(
            k,
            "HOME"
                | "USER"
                | "LOGNAME"
                | "LANG"
                | "LC_ALL"
                | "LC_CTYPE"
                | "TERM"
                | "TZ"
                | "XDG_CONFIG_HOME"
        ) || k.starts_with("CLOUDSDK_");
        if keep {
            c.env(k, v);
        }
    }
    c
}

pub fn gcloud_missing() -> Failure {
    Failure::Preflight(format!(
        "gcloud not found at {} or {} (never looked up in $PATH). Install it: x0ne-engine cloud setup gcp",
        GCLOUD_PATHS[0], GCLOUD_PATHS[1]
    ))
}

/// `gcloud config get-value <key>`; None when unset.
pub fn gcloud_config_value(path: &Path, key: &str) -> Option<String> {
    let out = gcloud_command(path)
        .args(["config", "get-value", key])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    let v = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (out.status.success() && !v.is_empty() && !v.contains(char::is_whitespace)).then_some(v)
}

/// The exec for `x0ne-engine cloud login gcp`.
pub fn gcloud_login_command(path: &Path) -> Command {
    let mut c = gcloud_command(path);
    c.args(["auth", "login", "--no-launch-browser"]);
    c
}

/// `--auth gcloud`: explicit only. gcloud keeps a refresh token on disk,
/// the opposite of a short token; x0ne never reads gcloud's files itself.
pub struct Gcloud {
    path: PathBuf,
    cached: Option<(Zeroizing<String>, Instant)>,
}

impl Gcloud {
    pub fn new() -> Result<Self, Failure> {
        let path = find_gcloud().ok_or_else(gcloud_missing)?;
        Ok(Gcloud { path, cached: None })
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl TokenSource for Gcloud {
    fn token(&mut self) -> Result<Zeroizing<String>, Failure> {
        if let Some((t, at)) = &self.cached {
            if at.elapsed() < Duration::from_secs(300) {
                return Ok(t.clone());
            }
        }
        let mut child = gcloud_command(&self.path)
            .args(["auth", "print-access-token"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| Failure::Preflight(format!("{}: {e}", self.path.display())))?;
        let raw = child
            .stdout
            .take()
            .map(|o| read_capped(o, MAX_TOKEN + 64))
            .transpose()
            .map_err(|e| Failure::Preflight(format!("gcloud auth print-access-token: {e}")))?
            .unwrap_or_default();
        let status = child
            .wait()
            .map_err(|e| Failure::Preflight(format!("gcloud: {e}")))?;
        if !status.success() {
            return Err(Failure::Preflight(
                "gcloud auth print-access-token failed (x0ne-engine cloud login gcp)".into(),
            ));
        }
        let t = token_from_bytes(&raw)?;
        self.cached = Some((t.clone(), Instant::now()));
        Ok(t)
    }
    fn refreshable(&self) -> bool {
        true
    }
    fn label(&self) -> &'static str {
        "gcloud"
    }
}

/// A token given by the caller (library use and unit tests).
pub struct Fixed(pub Zeroizing<String>);

impl TokenSource for Fixed {
    fn token(&mut self) -> Result<Zeroizing<String>, Failure> {
        Ok(self.0.clone())
    }
    fn refreshable(&self) -> bool {
        false
    }
    fn label(&self) -> &'static str {
        "token"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardening_leaves_the_process_non_dumpable() {
        harden_process().unwrap();
        assert!(!is_dumpable());
    }

    #[test]
    fn tokens_are_trimmed_and_checked() {
        let t = token_from_bytes(b"  ya29.a0AfH6SMBx-fake_token_for_tests_only \n").unwrap();
        assert_eq!(t.as_str(), "ya29.a0AfH6SMBx-fake_token_for_tests_only");
        assert!(token_from_bytes(b"short").is_err());
        assert!(token_from_bytes(b"ya29.aaaaaaaaaaaaaaaaaaaa\r\nX-Evil: 1").is_err());
        assert!(token_from_bytes(&vec![b'a'; MAX_TOKEN + 1]).is_err());
    }

    #[test]
    fn capped_reads_refuse_long_input() {
        assert!(read_capped(&b"abc"[..], 2).is_err());
        assert_eq!(&read_capped(&b"abc"[..], 3).unwrap()[..], b"abc");
    }

    #[test]
    fn gcloud_is_looked_up_at_fixed_paths_only() {
        let dir = tempfile::tempdir().unwrap();
        let fake = dir.path().join("gcloud");
        std::fs::write(&fake, "#!/bin/sh\n").unwrap();
        let missing = dir.path().join("nope/gcloud");
        assert_eq!(find_gcloud_in(&[missing.as_path()]), None);
        assert_eq!(
            find_gcloud_in(&[missing.as_path(), fake.as_path()]),
            Some(fake.clone())
        );
    }
}
