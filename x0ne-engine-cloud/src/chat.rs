//! The chat of `run` and `chat`: `x0ne-engine-proxy chat`, the relay's own
//! binary, so no question or answer passes through this process, which
//! holds the cloud token. Here: its options, where the serving session is,
//! and how it starts (the relay's environment, a parent-death signal).

use std::net::SocketAddr;
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use tokio::process::{Child, Command};

use crate::{sanitize, Failure};

pub const DEFAULT_MAX_TOKENS: u32 = 1024;
/// `x0ne-engine-proxy chat`'s exit when the relay stopped the session on an
/// answer (`local_chat::EXIT_SESSION_STOPPED` there): the relay's refusal follows.
pub const CHAT_SESSION_STOPPED: i32 = 3;
/// The TD's ceiling: the relay's default `--max-completion-tokens`.
pub const MAX_TOKENS_CAP: u32 = 4096;

#[derive(Debug, Clone, PartialEq)]
pub struct ChatOpts {
    /// Default: the session state's, phase `serving`.
    pub listen: Option<String>,
    pub key_file: Option<PathBuf>,
    pub max_tokens: u32,
}

impl Default for ChatOpts {
    fn default() -> Self {
        ChatOpts {
            listen: None,
            key_file: None,
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }
}

fn fail(msg: impl Into<String>) -> Failure {
    Failure::Chat(msg.into())
}

/// Loopback only: the key goes nowhere else.
pub fn loopback(listen: &str) -> Result<SocketAddr, Failure> {
    listen
        .parse::<SocketAddr>()
        .ok()
        .filter(|a| a.ip().is_loopback() && a.port() != 0)
        .ok_or_else(|| fail(format!("{listen:?} is not a loopback address and port")))
}

/// First 16 hex digits of a digest or key, for one line.
pub fn short(hex: &str) -> String {
    sanitize(&hex.chars().take(16).collect::<String>(), 16)
}

/// The relay's arguments for a chat.
pub fn relay_args(listen: &str, key_file: &str, max_tokens: u32) -> Vec<String> {
    vec![
        "chat".into(),
        "--listen".into(),
        listen.into(),
        "--key-file".into(),
        key_file.into(),
        "--max-tokens".into(),
        max_tokens.to_string(),
    ]
}

/// `run`'s chat: in the foreground process group (it reads the terminal,
/// and Ctrl-C reaches it too), with the relay's environment.
pub fn spawn(
    relay: &Path,
    listen: &str,
    key_file: &str,
    max_tokens: u32,
    input: Stdio,
) -> Result<Child, Failure> {
    loopback(listen)?;
    let mut cmd = Command::new(relay);
    cmd.args(relay_args(listen, key_file, max_tokens))
        .stdin(input)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    crate::proxy_link::confine(&mut cmd);
    cmd.spawn()
        .map_err(|e| fail(format!("starting the chat ({}): {e}", relay.display())))
}

/// The serving session's address and key file, unless both are given.
fn resolve(o: &ChatOpts) -> Result<(String, PathBuf), Failure> {
    if let (Some(l), Some(k)) = (&o.listen, &o.key_file) {
        return Ok((l.clone(), k.clone()));
    }
    let paths = crate::session::Paths::resolve()?;
    let st = crate::session::read_state(&paths)
        .map_err(|e| fail(format!("session state: {e}")))?
        .filter(|s| s.phase == "serving")
        .ok_or_else(|| {
            fail("no session is serving on this machine (x0ne-engine cloud up, or run)")
        })?;
    // A dead `up` leaves "serving" behind; its port is then anyone's (rev10).
    if !crate::session::up_alive(&paths, &st) {
        return Err(fail(
            "the session's x0ne-engine cloud process is gone, and its relay with it \
             (x0ne-engine cloud status, down)",
        ));
    }
    let key = o
        .key_file
        .clone()
        .or_else(|| st.key_file.clone().map(PathBuf::from))
        .ok_or_else(|| fail("the session state names no key file"))?;
    Ok((o.listen.clone().unwrap_or(st.listen), key))
}

/// `chat`: this process becomes the relay's chat (exec), with the relay's
/// environment. Leaving it does not end the session.
pub fn exec(o: &ChatOpts) -> Result<(), Failure> {
    let relay = crate::proxy_link::find_proxy()?;
    let (listen, key_file) = resolve(o)?;
    loopback(&listen)?;
    let mut cmd = std::process::Command::new(&relay);
    cmd.args(relay_args(
        &listen,
        &key_file.to_string_lossy(),
        o.max_tokens,
    ))
    .env_clear();
    for k in crate::proxy_link::KEPT_ENV {
        if let Some(v) = std::env::var_os(k) {
            cmd.env(k, v);
        }
    }
    let e = cmd.exec();
    Err(fail(format!("{}: {e}", relay.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_goes_to_the_loopback_only() {
        assert!(loopback("127.0.0.1:8478").is_ok());
        assert!(loopback("[::1]:8478").is_ok());
        assert_eq!(loopback("203.0.113.7:8478").unwrap_err().code(), 96);
        assert!(loopback("localhost:8478").is_err());
        assert!(loopback("127.0.0.1:0").is_err());
    }

    #[test]
    fn the_relay_is_asked_for_a_chat_and_nothing_else() {
        assert_eq!(
            relay_args("127.0.0.1:8478", "/run/k", 64),
            [
                "chat",
                "--listen",
                "127.0.0.1:8478",
                "--key-file",
                "/run/k",
                "--max-tokens",
                "64"
            ]
        );
        assert_eq!(short(&"ab".repeat(32)), "abababababababab");
    }
}
