//! Colors of `x0ne-engine cloud`, against the fake Google Cloud and the fake
//! relay: a whole session forced (`CLICOLOR_FORCE=1`) against its golden and,
//! colors stripped, against the same session printed to files; each stream
//! judged on its own on a terminal; `NO_COLOR` over everything. `demo`
//! (ignored) plays a session on this terminal, for a recording.

mod common;

use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use common::*;
use serde_json::json;
use x0ne_engine_cloud::color::strip;

const FORCE: &[(&str, &str)] = &[("CLICOLOR_FORCE", "1")];
const GOLDEN: &str = "tests/golden/colors_forced.ansi";

fn command(env: &Env, gcp: &FakeGcp, args: &[&str], vars: &[(&str, &str)]) -> Command {
    let mut c = env.command(&gcp.base, args);
    for (k, v) in vars {
        c.env(k, v);
    }
    c
}

fn record(t: &mut String, args: &str, d: &Done) {
    t.push_str(&format!("$ x0ne-engine cloud {args}\n"));
    if !d.out.is_empty() {
        t.push_str("[stdout]\n");
        t.push_str(&d.out);
    }
    if !d.err.is_empty() {
        t.push_str("[stderr]\n");
        t.push_str(&d.err);
    }
    t.push_str(&format!("[exit {}]\n\n", d.code));
}

// ── what varies from one run to the next ────────────────────────────────────

/// Start of the number that ends `head`: digits, `:`, `.`, `-`, and a space
/// between two of them (`2026-10-17 21:05`).
fn number_start(head: &str) -> usize {
    let b = head.as_bytes();
    let mut i = b.len();
    while i > 0 {
        let c = b[i - 1];
        let inner_space = c == b' ' && i >= 2 && b[i - 2].is_ascii_digit() && i < b.len();
        if c.is_ascii_digit() || matches!(c, b':' | b'.' | b'-') || inner_space {
            i -= 1;
        } else {
            break;
        }
    }
    while i < b.len() && !b[i].is_ascii_digit() {
        i += 1;
    }
    i
}

/// The number right before each `suffix` becomes `#`.
fn mask_before(s: &str, suffix: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find(suffix) {
        let head = &rest[..i];
        let n = number_start(head);
        out.push_str(&head[..n]);
        if n < head.len() {
            out.push('#');
        }
        out.push_str(suffix);
        rest = &rest[i + suffix.len()..];
    }
    out.push_str(rest);
    out
}

/// What follows each `prefix` up to the first character `keep` refuses becomes `#`.
fn mask_after(s: &str, prefix: &str, keep: impl Fn(char) -> bool) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find(prefix) {
        out.push_str(&rest[..i + prefix.len()]);
        rest = &rest[i + prefix.len()..];
        let n = rest.find(|c: char| !keep(c)).unwrap_or(rest.len());
        if n > 0 {
            out.push('#');
        }
        rest = &rest[n..];
    }
    out.push_str(rest);
    out
}

/// Directories, session ids, clock times, durations, costs so far, pids and
/// the transit bucket's random suffix become markers.
fn normalize(t: &str, dirs: &[&Path], vms: &[String]) -> String {
    let mut s = t.to_string();
    for d in dirs {
        s = s.replace(&d.display().to_string(), "<DIR>");
    }
    for vm in vms {
        let id = vm.trim_start_matches("x0ne-engine-");
        s = s.replace(vm, "x0ne-engine-<SESSION>");
        s = s.replace(&format!("session {id}"), "session <SESSION>");
    }
    for suffix in [" UTC", " min left", " USD so far"] {
        s = mask_before(&s, suffix);
    }
    let num = |c: char| c.is_ascii_digit() || c == '.';
    s = mask_after(&s, "Estimated cost: ", num);
    s = mask_after(&s, "estimated cost ", num);
    s = mask_after(&s, "up process ", num);
    s = mask_after(&s, "done (", |c| c != ')');
    mask_after(&s, "x0ne-engine-import-", |c| c.is_ascii_hexdigit())
}

// ── a whole session, every command ──────────────────────────────────────────

fn disk_tar(env: &Env) -> (std::path::PathBuf, std::path::PathBuf) {
    let disk = env.path("disk.tar.gz");
    fs::write(
        &disk,
        (0..300_000).map(|i| (i % 251) as u8).collect::<Vec<_>>(),
    )
    .unwrap();
    let (_, sha) = x0ne_engine_cloud::image_import::sha256_file(&disk).unwrap();
    let reference = env.path("x0ne-cvm-0.1.0.json");
    fs::write(
        &reference,
        json!({"schema": "x0ne-engine/reference/v2", "provider": "gcp", "tee": "intel-tdx",
               "image": {"name": "x0ne-cvm", "version": "0.1.0", "gcp_image": "x0ne-cvm-0-1-0",
                         "disk_tar_sha256": sha}})
        .to_string(),
    )
    .unwrap();
    (disk, reference)
}

fn leftovers() -> Cloud {
    let ours = json!({"x0ne-engine": "session"});
    let mut c = Cloud::default();
    c.vms.insert(
        "x0ne-engine-f00d42".into(),
        Vm {
            zone: "us-central1-a".into(),
            status: "RUNNING".into(),
            labels: ours.clone(),
            gets: 100,
        },
    );
    c.addresses = vec![(
        "us-central1".into(),
        json!({"name": "x0ne-engine-ip1", "status": "RESERVED", "labels": ours}),
    )];
    c
}

/// quote; up, status while it serves, down; status --limits; sweep; image
/// import; then three failures: a refused proof, a deletion not confirmed, a
/// quote above --max-cost. Every command with `vars` added to its environment.
fn session(vars: &[(&str, &str)]) -> String {
    let mut t = String::new();
    let mut dirs = Vec::new();
    let mut vms = Vec::new();

    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Proof));
    let mut c = command(&env, &gcp, &["quote", "--auth", "stdin"], vars);
    record(
        &mut t,
        "quote",
        &spawn(&env, &mut c, Some(TOKEN), "quote", None).finish(),
    );
    let mut c = command(&env, &gcp, &up_args(&[]), vars);
    let mut up = spawn(&env, &mut c, Some(TOKEN), "up", None);
    up.wait_stderr("(key: ");
    let mut c = command(&env, &gcp, &["status"], vars);
    let status = spawn(&env, &mut c, None, "status", None).finish();
    let mut c = command(&env, &gcp, &["down"], vars);
    let down = spawn(&env, &mut c, None, "down", None).finish();
    record(&mut t, "up --yes --max-cost 10", &up.finish());
    record(&mut t, "status   # while up serves", &status);
    record(&mut t, "down", &down);
    let mut c = command(&env, &gcp, &["status", "--limits"], vars);
    record(
        &mut t,
        "status --limits",
        &spawn(&env, &mut c, None, "limits", None).finish(),
    );
    vms.push(vm_name(&gcp));
    dirs.push(env.dir.path().to_path_buf());

    let gcp = FakeGcp::start(leftovers());
    let env = Env::new(None);
    let mut c = command(&env, &gcp, &["sweep", "--yes", "--auth", "stdin"], vars);
    record(
        &mut t,
        "sweep --yes",
        &spawn(&env, &mut c, Some(TOKEN), "sweep", None).finish(),
    );
    dirs.push(env.dir.path().to_path_buf());

    let mut cloud = Cloud::default();
    cloud.images.clear();
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(None);
    let (disk, reference) = disk_tar(&env);
    let (disk, reference) = (disk.to_str().unwrap(), reference.to_str().unwrap());
    let args = [
        "image",
        "import",
        disk,
        "--reference",
        reference,
        "--yes",
        "--auth",
        "stdin",
    ];
    let mut c = command(&env, &gcp, &args, vars);
    record(
        &mut t,
        "image import <DIR>/disk.tar.gz --reference <DIR>/x0ne-cvm-0.1.0.json --yes",
        &spawn(&env, &mut c, Some(TOKEN), "import", None).finish(),
    );
    dirs.push(env.dir.path().to_path_buf());

    // (relay, cloud, --max-cost, label, Ctrl-C once serving)
    for (relay, cloud, extra, name, interrupt) in [
        (Relay::Refuse(65), Cloud::default(), "10", "refused", false),
        (
            Relay::Ok,
            Cloud {
                fail_vm_delete: true,
                ..Cloud::default()
            },
            "10",
            "delete-fails, Ctrl-C",
            true,
        ),
        (Relay::Ok, Cloud::default(), "1", "above-max-cost", false),
    ] {
        let gcp = FakeGcp::start(cloud);
        let env = Env::new(Some(relay));
        let args = ["up", "--auth", "stdin", "--yes", "--max-cost", extra];
        let mut c = command(&env, &gcp, &args, vars);
        let mut up = spawn(&env, &mut c, Some(TOKEN), "up", None);
        if interrupt {
            up.wait_stderr("(key: ");
            up.signal(libc::SIGINT);
        }
        record(
            &mut t,
            &format!("up --yes --max-cost {extra}   # {name}"),
            &up.finish(),
        );
        if gcp.count("POST", "/zones/us-central1-a/instances") == 1 {
            vms.push(vm_name(&gcp));
        }
        dirs.push(env.dir.path().to_path_buf());
    }
    let dirs: Vec<&Path> = dirs.iter().map(|d| d.as_path()).collect();
    normalize(&t, &dirs, &vms)
}

/// The forced session is its golden, byte for byte once normalized; colors
/// stripped, it is the session printed to files without them.
#[test]
fn a_whole_session_forced_matches_its_golden_and_strips_to_the_plain_text() {
    let forced = session(FORCE);
    let plain = session(&[]);
    assert!(
        !plain.contains('\x1b'),
        "plain output carries escapes:\n{plain}"
    );
    assert_eq!(
        strip(&forced),
        plain,
        "the text changed, not only its colors"
    );
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(GOLDEN);
    if std::env::var_os("X0NE_UPDATE_GOLDEN").is_some() {
        fs::write(&path, &forced).unwrap();
    }
    let golden = fs::read_to_string(&path).unwrap_or_default();
    assert!(
        forced == golden,
        "{GOLDEN} differs (X0NE_UPDATE_GOLDEN=1 rewrites it):\n{forced}"
    );
    assert!(
        forced.contains(
            "\x1b[1m[1/5] Creating x0ne-engine-<SESSION> in us-central1-a\x1b[0m ... done"
        ),
        "{forced}"
    );
    assert!(
        forced.contains("      \x1b[1mCPU\x1b[0m   Intel TDX quote \x1b[32mgenuine\x1b[0m,"),
        "{forced}"
    );
    assert!(
        forced.contains("      \x1b[1;32mVERIFIED\x1b[0m. Limits:"),
        "{forced}"
    );
    assert!(
        forced.contains(
            "      \x1b[31mREFUSED (65): RTMR2 does not match the reference (fake)\x1b[0m"
        ),
        "{forced}"
    );
    assert!(
        forced.contains("\x1b[1;31mNOT CONFIRMED\x1b[0m"),
        "{forced}"
    );
    assert!(
        forced.contains("\x1b[31mx0ne-engine cloud: the maximum cost"),
        "{forced}"
    );
    assert!(
        forced.contains("done. \x1b[33mEstimated cost: # USD.\x1b[0m"),
        "{forced}"
    );
}

/// stdout and stderr each on a terminal (TERM set) while the other is a file:
/// only the terminal gets colors. `quote --max-cost 1` writes the quote on
/// stdout and its refusal (90) on stderr.
#[test]
fn each_stream_is_colored_only_when_it_is_a_terminal() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(None);
    for (out_tty, err_tty) in [(false, true), (true, false), (true, true)] {
        let pty = Pty::open();
        let args = ["quote", "--auth", "stdin", "--max-cost", "1"];
        let mut c = command(&env, &gcp, &args, &[("TERM", "xterm-256color")]);
        let d = spawn_split(&env, &mut c, Some(TOKEN), "quote", &pty, out_tty, err_tty).finish();
        assert_eq!(d.code, 90, "{}", d.err);
        let screen = String::from_utf8_lossy(&pty.screen()).into_owned();
        let quote_bold = "\x1b[1mQuote (Cloud Billing catalog, ";
        let refusal_red = "\x1b[31mx0ne-engine cloud: the maximum cost ";
        if out_tty {
            assert!(screen.contains(quote_bold), "{screen:?}");
        } else {
            assert!(
                d.out.starts_with("Quote (Cloud Billing catalog, "),
                "{:?}",
                d.out
            );
            assert!(!d.out.contains('\x1b') && !screen.contains(quote_bold));
        }
        if err_tty {
            assert!(screen.contains(refusal_red), "{screen:?}");
        } else {
            assert!(
                d.err.contains("x0ne-engine cloud: the maximum cost "),
                "{:?}",
                d.err
            );
            assert!(!d.err.contains('\x1b') && !screen.contains(refusal_red));
        }
    }
}

/// On a terminal: no TERM (as every other test here), TERM=dumb, NO_COLOR
/// (even with CLICOLOR_FORCE) or CLICOLOR=0 leave the bytes as they were.
#[test]
fn no_term_a_dumb_one_no_color_or_clicolor_0_stay_plain_on_a_terminal() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(None);
    let xterm = ("TERM", "xterm-256color");
    for vars in [
        vec![],
        vec![("TERM", "dumb")],
        vec![xterm, ("NO_COLOR", "1")],
        vec![xterm, ("NO_COLOR", "1"), ("CLICOLOR_FORCE", "1")],
        vec![xterm, ("CLICOLOR", "0")],
    ] {
        let pty = Pty::open();
        let mut c = command(
            &env,
            &gcp,
            &["quote", "--auth", "stdin", "--max-cost", "1"],
            &vars,
        );
        let d = spawn_on_tty(&env, &mut c, Some(TOKEN), "quote", &pty).finish();
        assert_eq!(d.code, 90, "{vars:?}");
        let screen = String::from_utf8_lossy(&pty.screen()).into_owned();
        assert!(screen.contains("the maximum cost"), "{vars:?}: {screen:?}");
        assert!(!screen.contains('\x1b'), "{vars:?}: {screen:?}");
    }
    // Files: NO_COLOR wins over CLICOLOR_FORCE there too.
    let mut c = command(
        &env,
        &gcp,
        &["quote", "--auth", "stdin", "--max-cost", "1"],
        &[("NO_COLOR", "1"), ("CLICOLOR_FORCE", "1")],
    );
    let d = spawn(&env, &mut c, Some(TOKEN), "quote", None).finish();
    assert_eq!(d.code, 90);
    assert!(!d.out.contains('\x1b') && !d.err.contains('\x1b'));
}

/// The relay and its chat decide their own colors: the four variables reach
/// them, and still no token and no override.
#[test]
fn the_relay_gets_the_color_variables_and_nothing_more() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let vars = [
        ("TERM", "xterm-256color"),
        ("NO_COLOR", "1"),
        ("CLICOLOR_FORCE", "1"),
        ("CLICOLOR", "1"),
    ];
    let mut c = command(&env, &gcp, &up_args(&[]), &vars);
    let mut up = spawn(&env, &mut c, Some(TOKEN), "up", None);
    up.wait_stderr("(key: ");
    up.signal(libc::SIGTERM);
    assert_eq!(up.finish().code, 0);
    let relay_env = env.relay_log("env");
    for (k, v) in vars {
        assert!(
            relay_env.contains(&format!("{k}={v}\n")),
            "{k}: {relay_env}"
        );
    }
    for absent in [TOKEN, "X0NE_ENGINE", "GOOGLE_CLOUD_PROJECT", "PATH="] {
        assert!(!relay_env.contains(absent), "{absent} in:\n{relay_env}");
    }
}

// ── the demo: a session on this terminal, for a recording ───────────────────

/// Copies to `tty` what the program wrote since `seen`.
fn forward(pty: &Pty, tty: &mut fs::File, seen: &mut usize) -> Vec<u8> {
    let screen = pty.screen();
    let _ = tty.write_all(&screen[*seen..]);
    *seen = screen.len();
    screen
}

/// Forwards until `needle` was written after `seen`.
fn follow(pty: &Pty, tty: &mut fs::File, seen: &mut usize, needle: &str) {
    let from = *seen;
    let until = Instant::now() + WAIT;
    while !contains(&forward(pty, tty, seen)[from..], needle.as_bytes()) {
        assert!(Instant::now() < until, "no {needle:?} on the terminal");
        std::thread::sleep(Duration::from_millis(15));
    }
}

/// Forwards until the program exits; its exit code.
fn play_out(mut run: Run, pty: &Pty, tty: &mut fs::File, seen: &mut usize) -> i32 {
    let until = Instant::now() + WAIT;
    let code = loop {
        forward(pty, tty, seen);
        if let Some(st) = run.child.try_wait().unwrap() {
            break st.code().unwrap_or(-1);
        }
        assert!(Instant::now() < until, "still running");
        std::thread::sleep(Duration::from_millis(15));
    };
    std::thread::sleep(Duration::from_millis(50));
    forward(pty, tty, seen);
    let _ = tty.write_all(b"\r\n");
    code
}

/// Typed at a human pace, echoed by the terminal.
fn type_slowly(pty: &Pty, text: &str) {
    for b in text.bytes() {
        pty.type_in(&[b]);
        std::thread::sleep(Duration::from_millis(45));
    }
}

fn shell_line(tty: &mut fs::File, line: &str) {
    let _ = tty.write_all(b"\x1b[1;34m$\x1b[0m ");
    for c in line.chars() {
        let _ = write!(tty, "{c}");
        std::thread::sleep(Duration::from_millis(35));
    }
    let _ = tty.write_all(b"\r\n");
}

/// The local API a serving relay offers, as the chat sees it: the model's
/// status, then answers with a verified receipt.
fn fake_local_api(listen: &str) {
    use axum::http::HeaderMap;
    use axum::routing::{get, post};
    let auth = format!("Bearer {}", "c".repeat(64));
    let check = move |h: &HeaderMap| {
        h.get("authorization").and_then(|v| v.to_str().ok()) == Some(auth.as_str())
    };
    let (c1, c2) = (check.clone(), check);
    let app = axum::Router::new()
        .route(
            "/x0ne/v1/status",
            get(move |h: HeaderMap| {
                let ok = c1(&h);
                async move {
                    assert!(ok);
                    axum::Json(json!({"model": {"name": "qwen3:0.6b"}, "seconds_left": 3540}))
                }
            }),
        )
        .route(
            "/v1/chat/completions",
            post(
                move |h: HeaderMap, axum::Json(b): axum::Json<serde_json::Value>| {
                    let ok = c2(&h);
                    async move {
                        assert!(ok);
                        let q = b["messages"]
                            .as_array()
                            .and_then(|m| m.last())
                            .and_then(|m| m["content"].as_str())
                            .unwrap_or("")
                            .to_string();
                        let a = if q.contains("France") { "Paris." } else { "7" };
                        tokio::time::sleep(Duration::from_millis(900)).await;
                        axum::Json(json!({
                            "choices": [{"message": {"role": "assistant", "content": a}, "finish_reason": "stop"}],
                            "usage": {"completion_tokens": 3},
                            "x0ne": {"receipt_verified": true, "receipt": {"model_digest_hex": "d".repeat(64)}},
                        }))
                    }
                },
            ),
        );
    let listen = listen.to_string();
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let l = tokio::net::TcpListener::bind(&listen).await.unwrap();
                axum::serve(l, app).await.unwrap();
            });
    });
}

/// One command at the prompt: `answers` are (what to wait for, what to type;
/// the token is pasted), then its output to the end.
fn scene(env: &Env, gcp: &FakeGcp, args: &[&str], answers: &[(&str, &str)]) -> i32 {
    let mut tty = fs::OpenOptions::new().write(true).open("/dev/tty").unwrap();
    shell_line(&mut tty, &format!("x0ne-engine cloud {}", args.join(" ")));
    let vars = [
        ("TERM", "xterm-256color"),
        ("CLICOLOR_FORCE", "1"),
        ("X0NE_ENGINE_CLOUD_TEST_POLL_MS", "350"),
    ];
    let pty = Pty::open();
    let mut c = command(env, gcp, args, &vars);
    let run = spawn_interactive(env, &mut c, "demo", &pty);
    let mut seen = 0;
    for (wait_for, typed) in answers {
        follow(&pty, &mut tty, &mut seen, wait_for);
        std::thread::sleep(Duration::from_millis(600));
        if *typed == TOKEN {
            pty.type_in(format!("{TOKEN}\n").as_bytes());
        } else {
            type_slowly(&pty, typed);
        }
    }
    let code = play_out(run, &pty, &mut tty, &mut seen);
    std::thread::sleep(Duration::from_millis(1500));
    code
}

/// `X0NE_DEMO_PROXY=<x0ne-engine-proxy> <this test binary> --ignored --exact demo`
/// on a terminal (asciinema): quote, run with its chat, status, then a proof
/// refused. Fake Google Cloud and fake relay of these tests, the relay's
/// real chat on a fake local API: no VM, no cost.
#[test]
#[ignore = "a recording, not a check: needs X0NE_DEMO_PROXY and a terminal"]
fn demo() {
    let Some(proxy) = std::env::var_os("X0NE_DEMO_PROXY") else {
        panic!("X0NE_DEMO_PROXY (the real x0ne-engine-proxy) is not set");
    };
    let mut tty = fs::OpenOptions::new().write(true).open("/dev/tty").unwrap();
    fake_local_api("127.0.0.1:8478");
    let _ = tty.write_all(
        b"\x1b[2m# x0ne-engine cloud, colors on. Fake Google Cloud and fake relay: no VM, no cost.\x1b[0m\r\n\r\n",
    );
    std::thread::sleep(Duration::from_millis(1000));
    let gcp = FakeGcp::start(Cloud {
        boot: Boot::After(4),
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Proof));
    // The fake relay speaks I4; its `chat` is the real relay's chat.
    let relay = env.bin.with_file_name("x0ne-engine-proxy");
    let script = fs::read_to_string(&relay).unwrap().replacen(
        "set -u\n",
        &format!(
            "set -u\n[ \"${{1:-}}\" = chat ] && exec '{}' \"$@\"\n",
            Path::new(&proxy).display()
        ),
        1,
    );
    write_exe(&relay, &script, 0o755);
    let paste = "(hidden, kept in memory only): ";
    let yes = "Type \"yes\" to create the VM: ";
    assert_eq!(scene(&env, &gcp, &["quote"], &[(paste, TOKEN)]), 0);
    let chat = [
        (paste, TOKEN),
        (yes, "yes\n"),
        (">>> ", "What is the capital of France? One word.\n"),
        (">>> ", "Pick a prime number between 5 and 10.\n"),
        (">>> ", "/exit\n"),
    ];
    let run = ["run", "qwen3:0.6b", "--max-cost", "10"];
    assert_eq!(scene(&env, &gcp, &run, &chat), 0);
    assert_eq!(scene(&env, &gcp, &["status"], &[]), 0);
    let _ = tty.write_all(b"\x1b[2m# the same with a relay that refuses the proof\x1b[0m\r\n");
    let refusing = Env::new(Some(Relay::Refuse(65)));
    let up = ["up", "--max-cost", "10"];
    assert_eq!(
        scene(&refusing, &gcp, &up, &[(paste, TOKEN), (yes, "yes\n")]),
        65
    );
}
