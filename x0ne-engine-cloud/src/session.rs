//! The local session: a state file with no secret, a lock held by `up` for
//! its whole life, and a control socket so `down` works from another
//! terminal. All of it under `$XDG_RUNTIME_DIR/x0ne-engine` (0700, tmpfs).

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{mpsc, watch};

use crate::provider::Instance;
use crate::Failure;

pub const STATE_SCHEMA: &str = "x0ne-engine/cloud-session/v1";

#[derive(Debug, Clone)]
pub struct Paths {
    /// `$XDG_RUNTIME_DIR/x0ne-engine`: state, lock, socket (the relay's key too).
    pub runtime: PathBuf,
    /// `$XDG_STATE_HOME/x0ne-engine`: boot logs kept for review (public data).
    pub state: PathBuf,
}

fn uid() -> u32 {
    // SAFETY: getuid cannot fail.
    unsafe { libc::getuid() }
}

/// Created 0700 if absent; refused if someone else owns it or can enter it.
pub fn private_dir(p: &Path) -> io::Result<()> {
    if !p.exists() {
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(p)?;
    }
    let m = fs::symlink_metadata(p)?;
    if !m.is_dir() || m.uid() != uid() || m.mode() & 0o077 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "{} must be a directory owned by you with mode 0700",
                p.display()
            ),
        ));
    }
    Ok(())
}

impl Paths {
    pub fn resolve() -> Result<Paths, Failure> {
        let runtime_base = std::env::var_os("XDG_RUNTIME_DIR")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| {
                let p = PathBuf::from(format!("/run/user/{}", uid()));
                fs::metadata(&p)
                    .ok()
                    .filter(|m| m.is_dir() && m.uid() == uid())
                    .map(|_| p)
            })
            .ok_or_else(|| {
                Failure::Preflight(
                    "no $XDG_RUNTIME_DIR (a per-user tmpfs): log in through a session manager, \
                     or set it to a 0700 directory you own"
                        .into(),
                )
            })?;
        let state_base = std::env::var_os("XDG_STATE_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .filter(|p| p.is_absolute())
                    .map(|h| h.join(".local/state"))
            })
            .ok_or_else(|| Failure::Preflight("neither $XDG_STATE_HOME nor $HOME is set".into()))?;
        let paths = Paths {
            runtime: runtime_base.join("x0ne-engine"),
            state: state_base.join("x0ne-engine"),
        };
        private_dir(&paths.runtime).map_err(|e| Failure::Preflight(e.to_string()))?;
        Ok(paths)
    }

    pub fn session_file(&self) -> PathBuf {
        self.runtime.join("cloud-session.json")
    }
    pub fn lock_file(&self) -> PathBuf {
        self.runtime.join("cloud-session.lock")
    }
    pub fn socket(&self) -> PathBuf {
        self.runtime.join("cloud.sock")
    }

    /// The state directory, created on first use.
    pub fn state_dir(&self) -> io::Result<&Path> {
        private_dir(&self.state)?;
        Ok(&self.state)
    }
}

/// What `status` and `down` need. Names, times and prices: no secret.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionState {
    pub schema: String,
    pub provider: String,
    pub instance: Instance,
    pub session_id: String,
    pub machine_type: String,
    pub model: String,
    pub listen: String,
    pub key_file: Option<String>,
    pub pid: u32,
    /// creating | booting | verifying | serving | stopping | deleted | delete-unconfirmed
    pub phase: String,
    pub created_at_unix: u64,
    pub expires_at_unix: u64,
    pub usd_per_hour: f64,
    pub max_cost_usd: f64,
    pub price_source: String,
    pub verdict: Option<Value>,
    pub limitations: Vec<String>,
    pub ended_at_unix: Option<u64>,
    pub note: Option<String>,
}

impl SessionState {
    pub fn live(&self) -> bool {
        !matches!(self.phase.as_str(), "deleted")
    }
}

/// Atomic: a temporary file, then rename. Mode 0600.
pub fn write_state(paths: &Paths, s: &SessionState) -> io::Result<()> {
    let body = serde_json::to_vec_pretty(s)?;
    let tmp = paths.runtime.join(".cloud-session.json.tmp");
    let _ = fs::remove_file(&tmp);
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&tmp)?;
    f.write_all(&body)?;
    f.sync_all()?;
    fs::rename(&tmp, paths.session_file())
}

pub fn read_state(paths: &Paths) -> io::Result<Option<SessionState>> {
    match fs::read(paths.session_file()) {
        Ok(b) => serde_json::from_slice(&b)
            .map(Some)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Held by `up` (or by `down` acting alone) until the process ends.
pub struct SessionLock {
    _file: File,
}

fn open_lock(paths: &Paths) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC)
        .open(paths.lock_file())
}

/// None when another process holds it.
pub fn try_lock(paths: &Paths) -> io::Result<Option<SessionLock>> {
    let f = open_lock(paths)?;
    // SAFETY: flock on a descriptor we own.
    if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        Ok(Some(SessionLock { _file: f }))
    } else {
        let e = io::Error::last_os_error();
        if e.raw_os_error() == Some(libc::EWOULDBLOCK) {
            Ok(None)
        } else {
            Err(e)
        }
    }
}

/// Is an `up` (or a `down`) alive and holding the session?
pub fn is_locked(paths: &Paths) -> bool {
    if !paths.lock_file().exists() {
        return false;
    }
    matches!(try_lock(paths), Ok(None))
}

/// The pid the kernel recorded at listen(2) on the control socket
/// (SO_PEERCRED), as seen from this pid namespace, 0 if not visible from
/// it; None for another user's.
fn peer_pid(s: &std::os::unix::net::UnixStream) -> Option<u32> {
    let mut c = libc::ucred {
        pid: 0,
        uid: 0,
        gid: 0,
    };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: getsockopt writes at most `len` bytes into `c`, which we own.
    let r = unsafe {
        libc::getsockopt(
            s.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            std::ptr::from_mut(&mut c).cast(),
            &mut len,
        )
    };
    (r == 0 && c.uid == uid()).then_some(c.pid as u32)
}

/// The process serving the control socket: its pid and its session id.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlPeer {
    pub pid: u32,
    pub session_id: String,
}

/// What the control socket says of the process serving it.
#[derive(Debug, Clone, PartialEq)]
pub enum Peer {
    /// No process of this user listens on it.
    Absent,
    /// It answered `{"cmd":"session"}` within 2 s.
    Session(ControlPeer),
    /// It accepted the connection but named no session within 2 s: stopped
    /// (Ctrl-Z, SIGSTOP), busy, or an `up` older than 0.2.0-beta.5.
    Silent,
}

pub fn peer(paths: &Paths) -> Peer {
    let Ok(s) = std::os::unix::net::UnixStream::connect(paths.socket()) else {
        return Peer::Absent;
    };
    let Some(pid) = peer_pid(&s) else {
        return Peer::Absent;
    };
    match session_of(&s) {
        Some(session_id) => Peer::Session(ControlPeer { pid, session_id }),
        None => Peer::Silent,
    }
}

fn session_of(mut s: &std::os::unix::net::UnixStream) -> Option<String> {
    use std::io::{BufRead as _, Read as _};
    s.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
    s.set_write_timeout(Some(Duration::from_secs(2))).ok()?;
    s.write_all(b"{\"cmd\":\"session\"}\n").ok()?;
    let mut line = String::new();
    io::BufReader::new(s.take(1024)).read_line(&mut line).ok()?;
    let v: Value = serde_json::from_str(line.trim()).ok()?;
    Some(v["session_id"].as_str()?.to_string())
}

/// None unless a process of this user serves the control socket and names
/// its session within 2 s.
pub fn control_peer(paths: &Paths) -> Option<ControlPeer> {
    match peer(paths) {
        Peer::Session(p) => Some(p),
        _ => None,
    }
}

/// `chat`'s proof that the `up` of `st` is alive: the lock is held and its
/// control socket is served by `st.pid` answering `st.session_id`. The lock
/// may be another process's, and a pid may be handed out again.
pub fn up_alive(paths: &Paths, st: &SessionState) -> bool {
    is_locked(paths)
        && control_peer(paths).is_some_and(|p| p.pid == st.pid && p.session_id == st.session_id)
}

/// What `status`, `sweep` and `down` can tell of the `up` of `st`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Liveness {
    /// The lock is held and the socket answers `st.session_id`. The pid is
    /// not compared: from another pid namespace it differs.
    Alive,
    /// The lock is held and a process listens without naming a session.
    Silent,
    /// No lock, no listener, or a listener of another session.
    Gone,
}

pub fn liveness(paths: &Paths, st: &SessionState) -> Liveness {
    if !is_locked(paths) {
        return Liveness::Gone;
    }
    match peer(paths) {
        Peer::Session(p) if p.session_id == st.session_id => Liveness::Alive,
        Peer::Silent => Liveness::Silent,
        Peer::Session(_) | Peer::Absent => Liveness::Gone,
    }
}

// ── control socket ──────────────────────────────────────────────────────────

/// Why `up` is asked to stop.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    Signal(&'static str),
    Down,
}

/// The end of a session, sent to a waiting `down`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Done {
    pub code: u8,
    pub message: String,
}

/// Serves `{"cmd":"stop"}` and `{"cmd":"session"}` (the id of the session
/// this process runs) to processes of the same user only.
/// `active` counts handlers still writing, so `up` can wait for them to flush.
pub fn serve_control(
    paths: &Paths,
    session_id: &str,
    stops: mpsc::UnboundedSender<StopReason>,
    done: watch::Receiver<Option<Done>>,
    active: Arc<AtomicUsize>,
) -> io::Result<()> {
    let sock = paths.socket();
    let _ = fs::remove_file(&sock);
    let listener = UnixListener::bind(&sock)?;
    fs::set_permissions(&sock, fs::Permissions::from_mode(0o600))?;
    let session_id: Arc<str> = session_id.into();
    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                continue;
            };
            let me = uid();
            if stream.peer_cred().map(|c| c.uid()).ok() != Some(me) {
                continue;
            }
            let stops = stops.clone();
            let done = done.clone();
            let active = active.clone();
            let session_id = session_id.clone();
            active.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(async move {
                let _ = handle_control(stream, &session_id, stops, done).await;
                active.fetch_sub(1, Ordering::SeqCst);
            });
        }
    });
    Ok(())
}

async fn handle_control(
    stream: UnixStream,
    session_id: &str,
    stops: mpsc::UnboundedSender<StopReason>,
    mut done: watch::Receiver<Option<Done>>,
) -> io::Result<()> {
    let (r, mut w) = stream.into_split();
    let mut line = String::new();
    let mut r = BufReader::new(r).take(1024);
    tokio::time::timeout(Duration::from_secs(5), r.read_line(&mut line))
        .await
        .map_err(|_| io::Error::from(io::ErrorKind::TimedOut))??;
    let cmd: Value = serde_json::from_str(line.trim()).unwrap_or(Value::Null);
    match cmd["cmd"].as_str() {
        Some("stop") => {}
        Some("session") => {
            let mut out = serde_json::to_vec(&serde_json::json!({ "session_id": session_id }))?;
            out.push(b'\n');
            return w.write_all(&out).await;
        }
        _ => {
            w.write_all(b"{\"error\":\"unknown command\"}\n").await?;
            return Ok(());
        }
    }
    let _ = stops.send(StopReason::Down);
    w.write_all(b"{\"ok\":true}\n").await?;
    let result = loop {
        if let Some(d) = done.borrow().clone() {
            break d;
        }
        if done.changed().await.is_err() {
            break Done {
                code: 95,
                message: "the up process ended without a result".into(),
            };
        }
    };
    let mut out = serde_json::to_vec(&result)?;
    out.push(b'\n');
    w.write_all(&out).await
}

/// `down` with `up` alive: ask it to stop, wait for its result.
pub async fn request_stop(paths: &Paths, wait: Duration) -> Result<Done, Failure> {
    let mut s = UnixStream::connect(paths.socket()).await.map_err(|e| {
        Failure::Preflight(format!(
            "the session is held by a running process but its control socket {} answers: {e}",
            paths.socket().display()
        ))
    })?;
    s.write_all(b"{\"cmd\":\"stop\"}\n")
        .await
        .map_err(|e| Failure::Preflight(format!("control socket: {e}")))?;
    let mut r = BufReader::new(s);
    let mut seen_ok = false;
    let read = async {
        loop {
            let mut line = String::new();
            let n = r.read_line(&mut line).await?;
            if n == 0 {
                return Ok::<Option<Done>, io::Error>(None);
            }
            let v: Value = serde_json::from_str(line.trim()).unwrap_or(Value::Null);
            if v["ok"] == Value::Bool(true) {
                seen_ok = true;
                sayln!("x0ne-engine cloud: the session is stopping; its VM is being deleted ...");
                continue;
            }
            if let Ok(d) = serde_json::from_value::<Done>(v) {
                return Ok(Some(d));
            }
        }
    };
    match tokio::time::timeout(wait, read).await {
        Ok(Ok(Some(d))) => Ok(d),
        Ok(Ok(None) | Err(_)) if seen_ok => Err(Failure::DeleteUnconfirmed(
            "the up process ended without reporting the deletion".into(),
        )),
        Ok(Ok(None) | Err(_)) => Err(Failure::Preflight(
            "the up process did not accept the stop request".into(),
        )),
        Err(_) => Err(Failure::DeleteUnconfirmed(format!(
            "no result from the up process after {} s",
            wait.as_secs()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(dir: &Path) -> Paths {
        let p = Paths {
            runtime: dir.join("rt/x0ne-engine"),
            state: dir.join("st/x0ne-engine"),
        };
        private_dir(&p.runtime).unwrap();
        p
    }

    fn sample() -> SessionState {
        SessionState {
            schema: STATE_SCHEMA.into(),
            provider: "gcp".into(),
            instance: Instance {
                project: "demo-project-123456".into(),
                zone: "us-central1-a".into(),
                name: "x0ne-engine-abc123".into(),
                firewall: "x0ne-engine-abc123".into(),
            },
            session_id: "abc123".into(),
            machine_type: crate::MACHINE_TYPE.into(),
            model: crate::DEFAULT_MODEL.into(),
            listen: crate::DEFAULT_LISTEN.into(),
            key_file: None,
            pid: 1,
            phase: "creating".into(),
            created_at_unix: 1,
            expires_at_unix: 3601,
            usd_per_hour: 6.3,
            max_cost_usd: 6.3,
            price_source: "table".into(),
            verdict: None,
            limitations: vec![],
            ended_at_unix: None,
            note: None,
        }
    }

    #[test]
    fn state_round_trips_with_mode_0600() {
        let d = tempfile::tempdir().unwrap();
        let p = paths(d.path());
        write_state(&p, &sample()).unwrap();
        let m = fs::metadata(p.session_file()).unwrap();
        assert_eq!(m.mode() & 0o777, 0o600);
        assert_eq!(read_state(&p).unwrap().unwrap(), sample());
    }

    #[test]
    fn a_group_readable_runtime_dir_is_refused() {
        let d = tempfile::tempdir().unwrap();
        let dir = d.path().join("open");
        fs::create_dir(&dir).unwrap();
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o750)).unwrap();
        assert!(private_dir(&dir).is_err());
    }

    #[test]
    fn the_lock_is_exclusive_while_held() {
        let d = tempfile::tempdir().unwrap();
        let p = paths(d.path());
        assert!(!is_locked(&p));
        let held = try_lock(&p).unwrap().expect("first lock");
        assert!(is_locked(&p));
        assert!(try_lock(&p).unwrap().is_none());
        drop(held);
        assert!(!is_locked(&p));
    }

    /// `serve_control` of this process for session `id`, until the runtime drops.
    fn served(p: &Paths, id: &str) -> tokio::runtime::Runtime {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let (tx, _) = mpsc::unbounded_channel();
        let (_, done) = watch::channel(None);
        rt.block_on(async { serve_control(p, id, tx, done, Arc::new(AtomicUsize::new(0))) })
            .unwrap();
        rt
    }

    #[test]
    fn the_up_is_alive_only_with_its_lock_its_pid_and_its_session_id() {
        let d = tempfile::tempdir().unwrap();
        let p = paths(d.path());
        let st = SessionState {
            pid: std::process::id(),
            ..sample()
        };
        assert_eq!(control_peer(&p), None);
        let lock = try_lock(&p).unwrap().expect("lock");
        let rt = served(&p, "abc123");
        assert_eq!(
            control_peer(&p),
            Some(ControlPeer {
                pid: std::process::id(),
                session_id: "abc123".into()
            })
        );
        assert!(up_alive(&p, &st));
        // Another session's up serves the socket (its window before "creating").
        let other = SessionState {
            session_id: "0dd0dd".into(),
            ..st.clone()
        };
        assert!(!up_alive(&p, &other));
        // The session's id from a process that did not write the state.
        let moved = SessionState {
            pid: st.pid + 1,
            ..st.clone()
        };
        assert!(!up_alive(&p, &moved));
        drop(lock);
        assert!(!up_alive(&p, &st), "no lock held");
        drop(rt);
        assert!(p.socket().exists(), "a dead up leaves its socket file");
        assert_eq!(control_peer(&p), None);
    }

    #[test]
    fn a_listener_that_does_not_name_its_session_is_not_an_up() {
        use std::io::Read as _;
        let d = tempfile::tempdir().unwrap();
        let p = paths(d.path());
        let l = std::os::unix::net::UnixListener::bind(p.socket()).unwrap();
        // beta.4's up answers any command but "stop" with an error.
        let old = std::thread::spawn(move || {
            let (mut c, _) = l.accept().unwrap();
            let mut b = [0u8; 64];
            let _ = c.read(&mut b);
            c.write_all(b"{\"error\":\"unknown command\"}\n").unwrap();
            l
        });
        assert_eq!(peer(&p), Peer::Silent);
        let _silent = old.join().unwrap();
        // Bound, never accepting: connect and write succeed, the read times out.
        let (tx, rx) = std::sync::mpsc::channel();
        let q = p.clone();
        std::thread::spawn(move || tx.send(peer(&q)));
        let r = rx.recv_timeout(Duration::from_secs(4));
        assert_eq!(r, Ok(Peer::Silent), "a silent listener held peer");
    }

    /// `status`, `sweep` and `down`: the session id decides, not the pid; a
    /// silent listener under the lock is not a dead `up`.
    #[test]
    fn liveness_is_the_session_id_under_the_lock_and_silence_is_not_death() {
        let d = tempfile::tempdir().unwrap();
        let p = paths(d.path());
        let st = SessionState {
            pid: std::process::id() + 1,
            ..sample()
        };
        assert_eq!(peer(&p), Peer::Absent);
        let lock = try_lock(&p).unwrap().expect("lock");
        assert_eq!(liveness(&p, &st), Liveness::Gone, "nobody listens");
        let rt = served(&p, "abc123");
        assert_eq!(liveness(&p, &st), Liveness::Alive, "another pid namespace");
        let other = SessionState {
            session_id: "0dd0dd".into(),
            ..st.clone()
        };
        assert_eq!(liveness(&p, &other), Liveness::Gone, "another session");
        drop(rt);
        let _l = std::os::unix::net::UnixListener::bind({
            let _ = fs::remove_file(p.socket());
            p.socket()
        })
        .unwrap();
        assert_eq!(liveness(&p, &st), Liveness::Silent);
        drop(lock);
        assert_eq!(liveness(&p, &st), Liveness::Gone, "no lock held");
    }
}
