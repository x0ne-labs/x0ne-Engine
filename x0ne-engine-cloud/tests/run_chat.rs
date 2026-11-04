//! `run`: `up` with the relay's chat in the terminal, against the fake
//! Google Cloud and the fake relay (whose `chat` echoes). Every way the chat
//! ends deletes the VM. The chat itself is tested in x0ne-infer
//! (`local_chat.rs`, and against the real relay in `cloud_e2e_tests.rs`).

mod common;

use std::fs;
use std::time::{Duration, Instant};

use common::*;

const MODEL: &str = "qwen3:0.6b";
const RECEIPT: &str = "  [receipt verified by the relay (fake)]";

fn run_args<'a>(extra: &[&'a str]) -> Vec<&'a str> {
    let mut v = vec!["run", MODEL, "--auth", "stdin", "--yes", "--max-cost", "10"];
    v.extend_from_slice(extra);
    v
}

fn wait_count(run: &mut Run, needle: &str, n: usize) {
    let until = Instant::now() + WAIT;
    while run.stderr().matches(needle).count() < n {
        if let Ok(Some(st)) = run.child.try_wait() {
            panic!("exited ({st}) before {n} × {needle:?}:\n{}", run.stderr());
        }
        assert!(
            Instant::now() < until,
            "no {n} × {needle:?}:\n{}",
            run.stderr()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Children of `pid`, from /proc/<n>/stat.
fn children(pid: i32) -> Vec<i32> {
    let mut out = Vec::new();
    for e in fs::read_dir("/proc").unwrap().flatten() {
        let Ok(n) = e.file_name().to_string_lossy().parse::<i32>() else {
            continue;
        };
        let Ok(stat) = fs::read_to_string(format!("/proc/{n}/stat")) else {
            continue;
        };
        let after = &stat[stat.rfind(')').unwrap_or(0) + 1..];
        if after.split_whitespace().nth(1) == Some(&pid.to_string()) {
            out.push(n);
        }
    }
    out
}

/// A session up to the chat's first prompt.
fn started(gcp: &FakeGcp, env: &Env, pty: &Pty) -> Run {
    let mut cmd = env.command(&gcp.base, &run_args(&[]));
    let mut run = spawn(env, &mut cmd, Some(TOKEN), "run", Some(pty));
    run.wait_stderr("Chat with fake (29 min left). /exit or Ctrl-D to leave.");
    let until = Instant::now() + WAIT;
    while !run.stdout().contains(">>> ") {
        assert!(Instant::now() < until, "no prompt: {}", run.stdout());
        std::thread::sleep(Duration::from_millis(20));
    }
    run
}

#[test]
fn run_starts_the_relays_chat_without_the_token_and_exit_deletes_the_vm() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let pty = Pty::open();
    let mut run = started(&gcp, &env, &pty);
    let vm = vm_name(&gcp);
    let err = run.stderr();
    let (d16, e16) = ("d".repeat(16), "e".repeat(16));
    for line in [
        "      CPU   fake TDX verdict",
        "      GPU   1 GPU(s), fake NRAS verdict",
        &format!(
            "Verdict, checked on this machine: CPU (Intel TDX) and 1 GPU (NVIDIA, confidential computing on) verified; model {MODEL}, digest {d16}; session key {e16}."
        ),
        "The local API stays open to other clients until the session ends.",
        "/exit, Ctrl-D or Ctrl-C ends the session and deletes the VM.",
    ] {
        assert!(err.contains(line), "missing {line:?} in:\n{err}");
    }

    // The chat is the relay's binary, with the relay's environment: no
    // token, no endpoint override, nothing but the kept directories.
    let key = env.runtime.join("x0ne-engine/local.key");
    assert_eq!(
        env.relay_log("chat-argv"),
        format!(
            "chat\n--listen\n127.0.0.1:8478\n--key-file\n{}\n--max-tokens\n1024\n",
            key.display()
        )
    );
    let chat_env = env.relay_log("chat-env");
    for absent in [TOKEN, "X0NE_ENGINE", "GOOGLE_CLOUD_PROJECT"] {
        assert!(!chat_env.contains(absent), "{absent} in:\n{chat_env}");
    }
    assert!(chat_env.contains("XDG_RUNTIME_DIR="), "{chat_env}");
    let chat = children(run.pid())
        .into_iter()
        .find(|c| {
            fs::read(format!("/proc/{c}/cmdline"))
                .is_ok_and(|b| contains(&b, b"x0ne-engine-proxy\0chat\0"))
        })
        .expect("the chat is a child of the lifecycle");
    let cmdline = fs::read(format!("/proc/{chat}/cmdline")).unwrap();
    assert!(!contains(&cmdline, TOKEN.as_bytes()));

    pty.type_in(b"hello\n");
    wait_count(&mut run, RECEIPT, 1);
    assert!(run.stdout().contains("echo 1: hello\n"), "{}", run.stdout());

    // `chat` from another terminal finds the serving session and becomes
    // the relay's chat (exec); leaving it does not end the session.
    let mut other = env.command(&gcp.base, &["chat", "--max-tokens", "64"]);
    let o = spawn(&env, &mut other, Some("from elsewhere\n"), "chat", None).finish();
    assert_eq!(o.code, 0, "{}", o.err);
    assert_eq!(o.out, "echo 1: from elsewhere\n");
    assert!(o.err.contains(RECEIPT), "{}", o.err);
    assert_eq!(
        env.relay_log("chat-argv"),
        format!(
            "chat\n--listen\n127.0.0.1:8478\n--key-file\n{}\n--max-tokens\n64\n",
            key.display()
        )
    );
    assert_eq!(env.session_state().unwrap()["phase"], "serving");

    pty.type_in(b"/exit\n");
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    for line in [
        &format!("Deleting {vm} ... done. Estimated cost: "),
        "Session ended from the chat.",
    ] {
        assert!(d.err.contains(line), "missing {line:?} in:\n{}", d.err);
    }
    assert_cleaned(&gcp, &vm);
    assert!(env.relay_log("cmds").contains(r#""cmd":"stop""#));
    assert_eq!(env.session_state().unwrap()["phase"], "deleted");
    assert!(
        env.files_containing(TOKEN.as_bytes()).is_empty(),
        "{:?}",
        env.files_containing(TOKEN.as_bytes())
    );
    assert!(env.decoys_ran().is_empty(), "{:?}", env.decoys_ran());
}

#[test]
fn ctrl_d_in_the_chat_deletes_the_vm() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let pty = Pty::open();
    let run = started(&gcp, &env, &pty);
    let vm = vm_name(&gcp);
    pty.type_in(b"\x04");
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(d.err.contains("Session ended from the chat."), "{}", d.err);
    assert_cleaned(&gcp, &vm);
    assert_eq!(env.session_state().unwrap()["phase"], "deleted");
}

#[test]
fn ctrl_c_in_the_chat_deletes_the_vm() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let pty = Pty::open();
    let run = started(&gcp, &env, &pty);
    let vm = vm_name(&gcp);
    pty.type_in(b"half a quest\x03");
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(d.err.contains("Session stopped by Ctrl-C."), "{}", d.err);
    assert_cleaned(&gcp, &vm);
    assert!(!d.out.contains("echo"), "{}", d.out);
}

#[test]
fn a_chat_that_fails_is_exit_96_and_deletes_the_vm() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let pty = Pty::open();
    let run = started(&gcp, &env, &pty);
    let vm = vm_name(&gcp);
    let chat = children(run.pid())
        .into_iter()
        .find(|c| {
            fs::read(format!("/proc/{c}/cmdline"))
                .is_ok_and(|b| contains(&b, b"x0ne-engine-proxy\0chat\0"))
        })
        .expect("the chat");
    // SAFETY: plain kill(2) on a process of this test.
    unsafe {
        libc::kill(chat, libc::SIGKILL);
    }
    let d = run.finish();
    assert_eq!(d.code, 96, "{}", d.err);
    assert!(
        d.err.contains("the chat was killed by signal 9"),
        "{}",
        d.err
    );
    assert_cleaned(&gcp, &vm);
}

/// The chat ends on an answer the relay stopped the session for, a second
/// before the relay's refusal: `run` waits for it and exits 65, not 96. A
/// failure is an exit code decided by which of the two processes ends first.
#[test]
fn a_chat_stopped_by_the_relay_exits_with_the_relays_refusal() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::RefuseAfterChat(65)));
    let pty = Pty::open();
    let run = started(&gcp, &env, &pty);
    let vm = vm_name(&gcp);
    pty.type_in(b"stop-the-session\n");
    let d = run.finish();
    assert_eq!(d.code, 65, "{}", d.err);
    assert!(
        d.err
            .contains("REFUSED (65): receipt verification failed (fake)"),
        "{}",
        d.err
    );
    assert!(!d.err.contains("the chat exited with code"), "{}", d.err);
    assert_cleaned(&gcp, &vm);
}

#[test]
fn a_terminal_hangup_during_the_chat_deletes_the_vm() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let pty = Pty::open();
    let mut cmd = env.command(&gcp.base, &run_args(&[]));
    let run = spawn_on_tty(&env, &mut cmd, Some(TOKEN), "run", &pty);
    pty.wait_screen(">>> ");
    let vm = vm_name(&gcp);
    pty.hang_up();
    run.finish();
    assert_cleaned(&gcp, &vm);
    assert_eq!(env.session_state().unwrap()["phase"], "deleted");
}

/// Ctrl-\ is the terminal's other quit key: SIGQUIT to the foreground group,
/// chat and cloud run alike. Before 0.2.0-beta.5 it ended the run without a
/// DELETE (exit 131), the VM billed until --max-run.
#[test]
fn ctrl_backslash_in_the_chat_deletes_the_vm() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let pty = Pty::open();
    let run = started(&gcp, &env, &pty);
    let vm = vm_name(&gcp);
    pty.type_in(b"\x1c");
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(d.err.contains("Session stopped by Ctrl-\\."), "{}", d.err);
    assert_cleaned(&gcp, &vm);
}

// ── refused before anything is created ──────────────────────────────────────

/// Absent from the catalog: refused before any request, the servable listed.
/// Every embedded entry is pinned since 0.2.0-beta.6; an unpinned one is refused the
/// same way (`catalog.rs`, `an_entry_without_a_served_digest_is_a_usage_error`).
#[test]
fn a_model_no_cloud_session_serves_is_refused_before_any_request() {
    let (model, why) = ("qwen3:999b", " is not in this build's catalog");
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let pty = Pty::open();
    let args = ["run", model, "--auth", "stdin", "--yes", "--max-cost", "10"];
    let mut cmd = env.command(&gcp.base, &args);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "run", Some(&pty)).finish();
    assert_eq!(d.code, 2, "{model}: {}", d.err);
    assert!(d.err.contains(&format!("{model}{why}")), "{}", d.err);
    assert!(
        d.err.contains(&format!("cloud sessions serve: {MODEL}")) && d.err.contains("qwen3:32b"),
        "the servable models are listed: {}",
        d.err
    );
    assert!(gcp.requests().is_empty(), "{:?}", gcp.requests());
    assert!(env.relay_log("argv").is_empty(), "the relay was started");
}

/// qwen3:8b (16.4 GB) passes the catalog; its boot allowance makes a 20 min
/// --max-run too short, refused before any request.
#[test]
fn qwen3_8b_passes_the_catalog_and_needs_24_minutes() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let pty = Pty::open();
    let args = [
        "run",
        "qwen3:8b",
        "--auth",
        "stdin",
        "--yes",
        "--max-cost",
        "10",
        "--max-run",
        "20m",
    ];
    let mut cmd = env.command(&gcp.base, &args);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "run", Some(&pty)).finish();
    assert_eq!(d.code, 2, "{}", d.err);
    assert!(!d.err.contains("served digest"), "{}", d.err);
    assert!(d.err.contains("ask for at least 24 min"), "{}", d.err);
    assert!(gcp.requests().is_empty(), "{:?}", gcp.requests());
}

#[test]
fn run_yes_without_a_ceiling_is_refused_before_any_request() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let pty = Pty::open();
    let mut cmd = env.command(&gcp.base, &["run", MODEL, "--auth", "stdin", "--yes"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "run", Some(&pty)).finish();
    assert_eq!(d.code, 2, "{}", d.err);
    assert!(d.err.contains("run --yes needs --max-cost"), "{}", d.err);
    assert!(gcp.requests().is_empty(), "{:?}", gcp.requests());
}

#[test]
fn run_with_the_token_on_stdin_and_no_terminal_creates_nothing() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let mut cmd = env.command(&gcp.base, &run_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "run", None).finish();
    assert_eq!(d.code, 2, "{}", d.err);
    assert!(d.err.contains("and there is no terminal"), "{}", d.err);
    no_creation(&gcp);
}

#[test]
fn chat_without_a_serving_session_says_so() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let mut cmd = env.command(&gcp.base, &["chat"]);
    let d = spawn(&env, &mut cmd, Some("hello\n"), "chat", None).finish();
    assert_eq!(d.code, 96, "{}", d.err);
    assert!(
        d.err.contains("no session is serving on this machine"),
        "{}",
        d.err
    );
    assert!(env.relay_log("chat-argv").is_empty());
}

/// The `run` process is gone (SIGKILL, OOM): its state still says "serving",
/// its relay is dead, and any local process may now listen on its loopback
/// port. `chat` must not hand that port the session's key and the questions.
#[test]
fn chat_refuses_a_session_whose_process_is_gone() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let pty = Pty::open();
    let run = started(&gcp, &env, &pty);
    run.signal(libc::SIGKILL);
    let _ = run.finish();
    assert_eq!(env.session_state().unwrap()["phase"], "serving");
    let _ = fs::remove_file(env.path("relay.chat-argv"));
    let mut other = env.command(&gcp.base, &["chat"]);
    let o = spawn(&env, &mut other, Some("my secret question\n"), "chat", None).finish();
    assert!(
        env.relay_log("chat-argv").is_empty(),
        "chat started against a dead session: {}",
        env.relay_log("chat-argv")
    );
    assert!(!o.out.contains("my secret question"), "{}", o.out);
    assert_eq!(o.code, 96, "{}", o.err);
    assert!(o.err.contains("process is gone"), "{}", o.err);
}

/// The live session from another process: the relay's chat starts on its key.
#[test]
fn chat_reaches_a_live_session_from_another_process() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut up = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    up.wait_stderr("[5/5] OpenAI API");
    let vm = vm_name(&gcp);
    let mut other = env.command(&gcp.base, &["chat"]);
    let o = spawn(&env, &mut other, Some("hello\n"), "chat", None).finish();
    assert_eq!(o.code, 0, "{}", o.err);
    assert!(o.out.contains("echo 1: hello"), "{}", o.out);
    assert!(
        env.relay_log("chat-argv").contains("local.key"),
        "{}",
        env.relay_log("chat-argv")
    );
    assert_eq!(env.session_state().unwrap()["phase"], "serving");
    up.signal(libc::SIGINT);
    let d = up.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert_cleaned(&gcp, &vm);
}

/// The dead session's state, its pid replaced: a pid the kernel handed out again.
fn set_state_pid(env: &Env, pid: u32) {
    let p = env.runtime.join("x0ne-engine/cloud-session.json");
    let mut st = env.session_state().unwrap();
    st["pid"] = pid.into();
    let tmp = env.path("cloud-session.json.tmp");
    fs::write(&tmp, serde_json::to_vec(&st).unwrap()).unwrap();
    fs::rename(&tmp, &p).unwrap();
}

/// A second `up` waiting for "yes" holds the lock of a dead session that still
/// says "serving": chat sent its key and question to the dead port (rev10,
/// 2026-10-17). Also refused when the dead pid is alive again.
#[test]
fn chat_refuses_a_dead_session_while_another_up_holds_the_lock() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let pty = Pty::open();
    let run = started(&gcp, &env, &pty);
    run.signal(libc::SIGKILL);
    let _ = run.finish();
    let dead = env.session_state().unwrap();
    assert_eq!(dead["phase"], "serving");
    // The dead session's VM went at --max-run.
    gcp.with(|c| {
        c.vms.clear();
        c.firewalls.clear();
    });
    let pty2 = Pty::open();
    let mut b = env.command(&gcp.base, &["up", "--auth", "stdin", "--max-cost", "10"]);
    let b = spawn(&env, &mut b, Some(TOKEN), "up-b", Some(&pty2));
    pty2.wait_screen("Type \"yes\"");
    assert_eq!(env.session_state().unwrap(), dead);
    let mut leaks = Vec::new();
    // The dead pid; the same number given to the lock holder; to another process of ours.
    let dead_pid = dead["pid"].as_u64().unwrap() as u32;
    for pid in [dead_pid, b.pid() as u32, std::process::id()] {
        set_state_pid(&env, pid);
        let _ = fs::remove_file(env.path("relay.chat-argv"));
        let mut other = env.command(&gcp.base, &["chat"]);
        let o = spawn(&env, &mut other, Some("my secret question\n"), "chat", None).finish();
        let argv = env.relay_log("chat-argv");
        if !argv.is_empty() || o.code != 96 || !o.err.contains("process is gone") {
            leaks.push((pid, o.code, argv, o.err));
        }
    }
    pty2.type_in(b"no\n");
    let _ = b.finish();
    assert!(
        leaks.is_empty(),
        "(state pid, chat exit, relay chat argv, stderr): {leaks:?}"
    );
}

/// `chat` against the dead session while the second `up` is in that window,
/// no pid reused. The window lasts until the relay's config, up to 30 s.
#[test]
fn chat_refuses_a_dead_session_while_a_second_up_serves_its_socket() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let pty = Pty::open();
    let first = started(&gcp, &env, &pty);
    let (dead, b, _pty2) = second_up_in_its_window(&gcp, &env, first);
    let _ = fs::remove_file(env.path("relay.chat-argv"));
    let mut other = env.command(&gcp.base, &["chat"]);
    let o = spawn(&env, &mut other, Some("my secret question\n"), "chat", None).finish();
    let argv = env.relay_log("chat-argv");
    let still_dead = env.session_state().unwrap() == dead;
    b.signal(libc::SIGINT);
    let _ = b.finish();
    assert!(still_dead, "the window closed before chat ran");
    assert!(
        argv.is_empty(),
        "chat started against a dead session: {argv}"
    );
    assert_eq!(o.code, 96, "{}", o.err);
    assert!(o.err.contains("process is gone"), "{}", o.err);
}

/// The same window with the dead pid handed out again, to the second `up`:
/// SO_PEERCRED names the state's pid, the session it serves is another.
#[test]
fn chat_refuses_a_dead_session_whose_pid_serves_the_socket_again() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Serve));
    let pty = Pty::open();
    let first = started(&gcp, &env, &pty);
    let (dead, b, _pty2) = second_up_in_its_window(&gcp, &env, first);
    set_state_pid(&env, b.pid() as u32);
    let _ = fs::remove_file(env.path("relay.chat-argv"));
    let mut other = env.command(&gcp.base, &["chat"]);
    let o = spawn(&env, &mut other, Some("my secret question\n"), "chat", None).finish();
    let argv = env.relay_log("chat-argv");
    let st = env.session_state().unwrap();
    let still_dead = st["session_id"] == dead["session_id"] && st["phase"] == "serving";
    b.signal(libc::SIGINT);
    let _ = b.finish();
    assert!(still_dead, "the window closed before chat ran");
    assert!(
        argv.is_empty(),
        "chat started against a dead session: {argv}"
    );
    assert_eq!(o.code, 96, "{}", o.err);
    assert!(o.err.contains("process is gone"), "{}", o.err);
}
