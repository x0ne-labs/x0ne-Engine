//! `x0ne-infer` and `x0ne-engine-proxy` as launched: not dumpable, no core
//! file, their memory closed to processes of the same user. Measured from
//! outside, on the real binaries.
#![cfg(target_os = "linux")]

use std::fs::File;
use std::io::Read;
use std::os::fd::{FromRawFd, OwnedFd};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// A pipe whose buffer is already full: a child writing to it blocks, alive,
/// until the test drains it.
fn full_pipe() -> (File, OwnedFd) {
    let mut fds = [0; 2];
    // SAFETY: plain fd syscalls on descriptors created here and owned below.
    unsafe {
        assert_eq!(libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC), 0);
        let (r, w) = (fds[0], fds[1]);
        libc::fcntl(w, libc::F_SETPIPE_SZ, 4096);
        let flags = libc::fcntl(w, libc::F_GETFL);
        libc::fcntl(w, libc::F_SETFL, flags | libc::O_NONBLOCK);
        let junk = [b'x'; 4096];
        while libc::write(w, junk.as_ptr().cast(), junk.len()) > 0 {}
        libc::fcntl(w, libc::F_SETFL, flags);
        (File::from_raw_fd(r), OwnedFd::from_raw_fd(w))
    }
}

fn field(status: &str, name: &str) -> String {
    status
        .lines()
        .find_map(|l| l.strip_prefix(name)?.strip_prefix(':'))
        .unwrap_or_else(|| panic!("no {name} in {status}"))
        .trim()
        .to_string()
}

fn status(pid: u32) -> String {
    std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap()
}

/// Owner of `/proc/<pid>/status`: root for a non-dumpable task (proc(5),
/// `PR_SET_DUMPABLE`), the task's user otherwise. `status` has no
/// "Dumpable" line (Linux 6.12).
fn proc_owner(pid: u32) -> u32 {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(format!("/proc/{pid}/status"))
        .unwrap()
        .uid()
}

fn my_uid() -> u32 {
    // SAFETY: read-only query.
    unsafe { libc::getuid() }
}

/// Root opens any `/proc/<pid>/mem` and attaches to any process, so the
/// refusals cannot be observed: SKIPPED line and false. `X0NE_REQUIRE_NON_ROOT=1`
/// (ci/inside.sh) turns the skip into a failure.
fn ordinary_user_or_skip(test: &str) -> bool {
    if my_uid() != 0 {
        return true;
    }
    assert!(
        std::env::var_os("X0NE_REQUIRE_NON_ROOT").is_none(),
        "{test}: an ordinary user is required, this runs as root"
    );
    eprintln!("SKIPPED {test}: runs as root, which passes every check");
    false
}

/// `PTRACE_ATTACH` from this thread: `Ok` if it took (then detached), or errno.
fn attach(pid: u32) -> Result<(), i32> {
    let pid = pid as libc::pid_t;
    // SAFETY: ptrace and waitpid on a child of this process; detached at once.
    unsafe {
        if libc::ptrace(libc::PTRACE_ATTACH, pid, 0, 0) != 0 {
            return Err(std::io::Error::last_os_error().raw_os_error().unwrap());
        }
        let mut st = 0;
        libc::waitpid(pid, &mut st, libc::__WALL);
        libc::ptrace(libc::PTRACE_DETACH, pid, 0, 0);
    }
    Ok(())
}

#[test]
fn the_td_binary_is_not_dumpable_has_no_core_and_refuses_its_peers() {
    if !ordinary_user_or_skip("the_td_binary_is_not_dumpable_has_no_core_and_refuses_its_peers") {
        return;
    }
    let (mut drain, stdout) = full_pipe();
    let mut child = Command::new(env!("CARGO_BIN_EXE_x0ne-infer"))
        // cargo sets LD_LIBRARY_PATH for tests; a release x0ne-infer refuses it (73).
        .env_remove("LD_LIBRARY_PATH")
        .arg("--help")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let pid = child.id();
    // More than one thread: the runtime exists, so main's first lines ran;
    // the help text then blocks on the full pipe.
    let until = Instant::now() + Duration::from_secs(20);
    let seen = loop {
        let s = status(pid);
        if field(&s, "Threads") != "1" || Instant::now() > until {
            break s;
        }
        std::thread::yield_now();
    };
    let owner = proc_owner(pid);
    let limits = std::fs::read_to_string(format!("/proc/{pid}/limits")).unwrap();
    let mem = File::open(format!("/proc/{pid}/mem"));
    let traced = attach(pid);
    let mut rest = Vec::new();
    drain.read_to_end(&mut rest).unwrap();
    let exit = child.wait().unwrap();

    assert_ne!(
        field(&seen, "Threads"),
        "1",
        "the child never got past main"
    );
    assert_eq!(
        owner, 0,
        "/proc/<pid> of a non-dumpable task belongs to root"
    );
    let core = limits
        .lines()
        .find(|l| l.starts_with("Max core file size"))
        .unwrap();
    assert_eq!(
        core.split_whitespace().collect::<Vec<_>>(),
        ["Max", "core", "file", "size", "0", "0", "bytes"]
    );
    assert_eq!(
        mem.map(|_| ()).map_err(|e| e.raw_os_error()),
        Err(Some(libc::EACCES)),
        "/proc/<pid>/mem opened"
    );
    assert_eq!(traced, Err(libc::EPERM), "PTRACE_ATTACH took");
    assert!(exit.success(), "{exit:?}");
    assert!(String::from_utf8_lossy(&rest).contains("--cloud-config"));
}

/// Control: an ordinary child of the same user is dumpable, and its parent
/// opens its memory, so the refusals above come from the binary.
#[test]
fn control_an_ordinary_child_is_dumpable_and_open_to_its_parent() {
    let mut child = Command::new("/bin/sleep").arg("10").spawn().unwrap();
    let pid = child.id();
    let owner = proc_owner(pid);
    let mem = File::open(format!("/proc/{pid}/mem"));
    let traced = attach(pid);
    child.kill().unwrap();
    child.wait().unwrap();
    assert_eq!(owner, my_uid());
    assert!(mem.is_ok(), "{mem:?}");
    assert_eq!(traced, Ok(()));
}

/// `x0ne-engine-proxy` holds the session key on the caller's machine: not
/// dumpable, no core file, closed to its peers. Until 2026-11-04 it ignored
/// what `prctl` returned. `pins` writes to a full pipe and blocks, alive,
/// once `main` has hardened the process.
#[test]
fn the_relay_binary_is_not_dumpable_has_no_core_and_refuses_its_peers() {
    if !ordinary_user_or_skip("the_relay_binary_is_not_dumpable_has_no_core_and_refuses_its_peers")
    {
        return;
    }
    let (_drain_out, stdout) = full_pipe();
    let (_drain_err, stderr) = full_pipe();
    let mut child = Command::new(env!("CARGO_BIN_EXE_x0ne-engine-proxy"))
        .arg("pins")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .unwrap();
    let pid = child.id();
    let core_line = |pid: u32| {
        std::fs::read_to_string(format!("/proc/{pid}/limits"))
            .ok()
            .and_then(|l| {
                l.lines()
                    .find(|l| l.starts_with("Max core file size"))
                    .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
            })
    };
    let until = Instant::now() + Duration::from_secs(20);
    let (owner, core) = loop {
        let owner = proc_owner(pid);
        let core = core_line(pid);
        let hardened = owner == 0 && core.as_deref() == Some("Max core file size 0 0 bytes");
        if hardened || Instant::now() > until {
            break (owner, core);
        }
        std::thread::yield_now();
    };
    let mem = File::open(format!("/proc/{pid}/mem"));
    let traced = attach(pid);
    child.kill().unwrap();
    child.wait().unwrap();

    assert_eq!(
        owner, 0,
        "/proc/<pid> of a non-dumpable task belongs to root"
    );
    assert_eq!(core.as_deref(), Some("Max core file size 0 0 bytes"));
    assert_eq!(
        mem.map(|_| ()).map_err(|e| e.raw_os_error()),
        Err(Some(libc::EACCES)),
        "/proc/<pid>/mem opened"
    );
    assert_eq!(traced, Err(libc::EPERM), "PTRACE_ATTACH took");
}
