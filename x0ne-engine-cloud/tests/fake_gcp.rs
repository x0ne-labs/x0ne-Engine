//! `up`, `quote`, `down`, `status`, `sweep`, `image import` and `image delete`
//! against a fake Google Cloud on the loopback, with a fake relay. No real
//! cloud is called.

mod common;

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

use common::*;
use serde_json::json;
use x0ne_engine_cloud::DEFAULT_IMAGE;

// ── up: confirmation, golden bodies, a whole session ────────────────────────

#[test]
fn up_creates_only_after_yes_serves_and_deletes_on_ctrl_c() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let pty = Pty::open();
    let mut cmd = env.command(&gcp.base, &["up", "--auth", "stdin", "--max-run", "30m"]);
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", Some(&pty));

    pty.wait_screen("Type \"yes\" to create the VM: ");
    no_creation(&gcp);
    let err = run.stderr();
    assert!(
        err.contains("x0ne Engine cloud: Google Cloud, project x0ne-test-project"),
        "{err}"
    );
    assert!(
        err.contains("Token OK: tester@example.com, expires in 59 min."),
        "{err}"
    );
    assert!(
        err.contains(&format!("Preflight: H100 spot quota in us-central1: 1 (used 0). No x0ne-engine VM running. Image {DEFAULT_IMAGE} found.")),
        "{err}"
    );
    assert!(err.contains("Quote (Cloud Billing catalog, "), "{err}");
    assert!(err.contains("maximum cost if it runs to the end"), "{err}");

    pty.type_in(b"yes\n");
    run.wait_stderr("[5/5] OpenAI API on http://127.0.0.1:8478/v1  (key: /fake/local.key)");
    let vm = vm_name(&gcp);

    // While it serves: the token is in neither argv nor a readable environ,
    // and `status` answers from the state file without a token.
    let pid = run.pid();
    let cmdline = fs::read(format!("/proc/{pid}/cmdline")).unwrap();
    assert!(!contains(&cmdline, TOKEN.as_bytes()));
    let environ = fs::read(format!("/proc/{pid}/environ"));
    assert!(
        matches!(&environ, Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied),
        "the environ of a non-dumpable process must be unreadable: {environ:?}"
    );
    let mut st = env.command(&gcp.base, &["status"]);
    let s = spawn(&env, &mut st, None, "status", None).finish();
    assert_eq!(s.code, 0, "{}", s.err);
    assert!(
        s.out.contains(&format!("{vm} (a3-highgpu-1g spot")),
        "{}",
        s.out
    );
    assert!(s.out.contains("serving"), "{}", s.out);
    assert!(
        s.out.contains(&format!("up process {pid} alive")),
        "{}",
        s.out
    );
    assert!(s.out.contains("verdict   VERIFIED"), "{}", s.out);

    pty.type_in(b"\x03");
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    for line in [
        &format!("[1/5] Creating {vm} in us-central1-a ... done"),
        "[2/5] Waiting for boot and GPU setup ... done",
        "[3/5] Checking the hardware proof on THIS machine. No question and no session key have been sent yet.",
        "      CPU   fake verdict",
        "      VERIFIED. Limits: x0ne-engine cloud status --limits",
        "[4/5] Releasing the session key to the verified VM ...",
        &format!("Deleting {vm} ... done. Estimated cost: "),
        "Session stopped by Ctrl-C.",
    ] {
        assert!(d.err.contains(line), "missing {line:?} in:\n{}", d.err);
    }
    assert_cleaned(&gcp, &vm);
    assert!(env.relay_log("cmds").contains(r#""cmd":"stop""#));
    assert!(env
        .relay_log("connect")
        .contains(&format!("\"endpoint\":\"{VM_IP}:443\"")));
    assert_eq!(env.session_state().unwrap()["phase"], "deleted");

    // The token went to tokeninfo in a body and to the API in a header only.
    assert!(
        env.files_containing(TOKEN.as_bytes()).is_empty(),
        "{:?}",
        env.files_containing(TOKEN.as_bytes())
    );
    assert!(!contains(&pty.screen(), TOKEN.as_bytes()));
    for r in gcp.requests() {
        assert!(
            !r.path.contains(TOKEN) && !r.query.contains(TOKEN),
            "token in a URL: {}",
            r.path
        );
    }
    assert!(env.decoys_ran().is_empty(), "{:?}", env.decoys_ran());
}

#[test]
fn the_environ_of_an_ordinary_child_is_readable_so_the_refusal_is_dumpable_0() {
    // Control for the check above: same user, same container, dumpable child.
    let mut c = {
        let _g = fork_lock();
        Command::new("/bin/sleep")
            .arg("5")
            .env("PROBE", "1")
            .spawn()
            .unwrap()
    };
    // posix_spawn returns before the new image's environment is mapped: poll.
    let until = std::time::Instant::now() + WAIT;
    let seen = loop {
        let e = fs::read(format!("/proc/{}/environ", c.id())).unwrap();
        if contains(&e, b"PROBE=1") || std::time::Instant::now() > until {
            break e;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    let _ = c.kill();
    let _ = c.wait();
    assert!(
        contains(&seen, b"PROBE=1"),
        "{:?}",
        String::from_utf8_lossy(&seen)
    );
}

#[test]
fn the_insert_and_firewall_bodies_match_the_golden_reference() {
    insert_and_firewall_match_the_golden_reference(Relay::Ok);
}

#[test]
fn the_golden_bodies_hold_when_a_second_passes_between_the_two_clock_reads() {
    insert_and_firewall_match_the_golden_reference(Relay::SlowConfig);
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn insert_and_firewall_match_the_golden_reference(relay: Relay) {
    let t0 = unix_now();
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(relay));
    let mut cmd = env.command(&gcp.base, &up_args(&["--max-run", "30m"]));
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    run.wait_stderr("[5/5] OpenAI API");
    run.signal(libc::SIGTERM);
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    let t1 = unix_now();

    let vm = vm_name(&gcp);
    let session = vm.strip_prefix("x0ne-engine-").unwrap().to_string();
    assert_eq!(session.len(), 6);
    let mut insert = gcp.find("POST", "/zones/us-central1-a/instances")[0].json();
    // up reads the clock for the expiry before the relay starts, and for created_at after its
    // config: maxRunDuration is the gap, 1799 when a second passes between (make ci, 2026-10-16).
    // Checked exactly against the state file and I1, then set to 1800 for the golden comparison.
    let st = env.session_state().unwrap();
    let created = st["created_at_unix"].as_u64().unwrap();
    let expires = st["expires_at_unix"].as_u64().unwrap();
    let start = expires - 1800;
    assert!(
        t0 <= start && start <= created && created <= t1,
        "{t0} {start} {created} {t1}"
    );
    assert_eq!(
        insert["scheduling"]["maxRunDuration"]["seconds"],
        json!(expires - created)
    );
    if relay == Relay::SlowConfig {
        assert!(expires - created < 1800, "{created} {expires}");
    }
    insert["scheduling"]["maxRunDuration"]["seconds"] = json!(1800);
    assert_eq!(
        normalize(&insert, &session),
        golden("instances_insert.json"),
        "body:\n{}",
        serde_json::to_string_pretty(&insert).unwrap()
    );
    // I1 byte for byte as the relay emitted it.
    let config = fs::read(env.path("relay.config")).unwrap();
    assert_eq!(
        insert["metadata"]["items"][0]["value"]
            .as_str()
            .unwrap()
            .as_bytes(),
        &config[..]
    );
    let i1: serde_json::Value = serde_json::from_slice(&config).unwrap();
    assert_eq!(i1["expires_at_unix"], json!(expires));
    let fw = gcp.find("POST", "/global/firewalls")[0].json();
    assert_eq!(normalize(&fw, &session), golden("firewalls_insert.json"));
    // The rule admits this machine's address only, as Google's name server saw it.
    assert!(gcp.with(|c| c.dns_queries) >= 1);
    assert!(
        d.err.contains(&format!(
            "Firewall: tcp:443 from {CLIENT_IP}/32 only, this machine's public address"
        )),
        "{}",
        d.err
    );
    assert!(d.err.contains("Session stopped by SIGTERM."), "{}", d.err);
    assert_cleaned(&gcp, &vm);
}

/// `--allow-from` names the rule's sources itself: no address is asked of
/// anyone, and the rule admits exactly the ranges given, canonical.
#[test]
fn up_allow_from_names_the_firewall_sources_and_asks_no_address() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let args = up_args(&[
        "--allow-from",
        "192.0.2.0/24",
        "--allow-from",
        "198.51.100.77",
    ]);
    let mut run = spawn(
        &env,
        &mut env.command(&gcp.base, &args),
        Some(TOKEN),
        "up",
        None,
    );
    run.wait_stderr("[5/5] OpenAI API");
    run.signal(libc::SIGTERM);
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    let fw = gcp.find("POST", "/global/firewalls")[0].json();
    assert_eq!(
        fw["sourceRanges"],
        json!(["192.0.2.0/24", "198.51.100.77/32"]),
        "{fw}"
    );
    assert_eq!(gcp.with(|c| c.dns_queries), 0);
    assert!(
        d.err
            .contains("Firewall: tcp:443 from 192.0.2.0/24, 198.51.100.77/32 (--allow-from)."),
        "{}",
        d.err
    );
    assert_cleaned(&gcp, &vm_name(&gcp));
}

/// No public address for this machine: no answer, or one the firewall cannot
/// name (private, carrier-grade NAT). Exit 91 before anything is created,
/// never a rule open to everyone, and the way out named.
#[test]
fn up_without_a_public_address_creates_nothing_and_names_allow_from() {
    for (answer, says) in [
        (None, "no answer from"),
        (Some("172.20.1.3"), "172.20.1.3 is private"),
        (Some("100.64.7.8"), "100.64.7.8 is carrier-grade NAT"),
    ] {
        let gcp = FakeGcp::start(Cloud {
            public_addr: answer.map(String::from),
            ..Cloud::default()
        });
        let env = Env::new(Some(Relay::Ok));
        let d = spawn(
            &env,
            &mut env.command(&gcp.base, &up_args(&[])),
            Some(TOKEN),
            "up",
            None,
        )
        .finish();
        assert_eq!(d.code, 91, "{answer:?}: {}", d.err);
        no_creation(&gcp);
        assert!(gcp.with(|c| c.dns_queries) >= 1);
        for part in [
            says,
            "this machine's public IPv4 address",
            "--allow-from <CIDR>",
            "nothing was created",
        ] {
            assert!(d.err.contains(part), "{answer:?}: {part:?} in {}", d.err);
        }
        assert!(!d.err.contains("0.0.0.0/0"), "{}", d.err);
    }
}

/// qwen3:32b, 17 shards: the checks before any VM use its 65.5 GB
/// (12 min + 27.3 min of intake + 5 min serving), the VM is the one H100
/// type, and the context bound the TD reports is printed.
#[test]
fn up_model_qwen3_32b_passes_the_checks_before_the_vm_and_prints_the_context() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let short = up_args(&["--model", "qwen3:32b", "--max-run", "44m"]);
    let d = spawn(
        &env,
        &mut env.command(&gcp.base, &short),
        Some(TOKEN),
        "up",
        None,
    )
    .finish();
    assert_eq!(d.code, 2, "{}", d.err);
    assert!(
        d.err
            .contains("qwen3:32b may take up to 40 min to boot and load; ask for at least 45 min"),
        "{}",
        d.err
    );
    no_creation(&gcp);

    let ok = up_args(&["--model", "qwen3:32b", "--max-run", "45m"]);
    let mut run = spawn(
        &env,
        &mut env.command(&gcp.base, &ok),
        Some(TOKEN),
        "up",
        None,
    );
    run.wait_stderr("      Context: 40960 tokens, prompt and answer together.");
    run.signal(libc::SIGTERM);
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    let vm = vm_name(&gcp);
    let insert = gcp.find("POST", "/zones/us-central1-a/instances")[0].json();
    assert!(
        insert["machineType"]
            .as_str()
            .unwrap()
            .ends_with("/machineTypes/a3-highgpu-1g"),
        "{insert}"
    );
    let i1: serde_json::Value =
        serde_json::from_slice(&fs::read(env.path("relay.config")).unwrap()).unwrap();
    assert_eq!(i1["model"], "qwen3:32b");
    assert!(
        env.relay_log("argv").contains("--model\nqwen3:32b\n"),
        "{}",
        env.relay_log("argv")
    );
    assert_cleaned(&gcp, &vm);
}

// ── a token `up` cannot renew (--auth paste, stdin) ─────────────────────────

/// Test timings, in milliseconds: the boot timeout, the serving `up` must
/// leave after it, and the margin it keeps before the token's expiry.
fn token_timings(cmd: &mut Command, boot: u32, serving: u32, margin: u32) -> &mut Command {
    cmd.env("X0NE_ENGINE_CLOUD_TEST_BOOT_TIMEOUT_MS", boot.to_string())
        .env("X0NE_ENGINE_CLOUD_TEST_MIN_SERVING_MS", serving.to_string())
        .env("X0NE_ENGINE_CLOUD_TEST_TOKEN_MARGIN_MS", margin.to_string())
}

fn token_not_shown(d: &Done) {
    assert!(!d.err.contains(TOKEN), "the token on stderr");
    assert!(!d.out.contains(TOKEN), "the token on stdout");
}

/// The fake's token (3599 s) and the default --max-run (60 min), production
/// timings: the session is cut to 54 min, and every step says it and why.
#[test]
fn a_pasted_token_cuts_max_run_and_up_says_so_from_the_quote_to_the_end() {
    let t0 = unix_now();
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    run.wait_stderr("[5/5] OpenAI API");
    let s = spawn(
        &env,
        &mut env.command(&gcp.base, &["status"]),
        None,
        "status",
        None,
    )
    .finish();
    assert_eq!(s.code, 0, "{}", s.err);
    assert!(
        s.out
            .contains("note      --max-run cut to 54 min: the token expires at "),
        "{}",
        s.out
    );
    run.signal(libc::SIGTERM);
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    for part in [
        "Token OK: tester@example.com, expires in 59 min.",
        "--max-run cut to 54 min (asked: 60 min): the token expires at ",
        " and up cannot renew it, so the session ends by ",
        ", 5 min before, to delete the VM with it.",
        " (54 min)\n",
        "54 min left: the session ends 5 min before the token expires (",
        "Session stopped by SIGTERM.",
    ] {
        assert!(d.err.contains(part), "missing {part:?} in:\n{}", d.err);
    }
    assert!(!d.err.contains("Your token expires before"), "{}", d.err);
    // The quote comes after the cut was computed: it says the session's end.
    let at = |after: &str| d.err.split(after).nth(1).and_then(|s| s.get(..9));
    assert_eq!(
        at("the VM deletes itself at "),
        at("so the session ends by "),
        "{}",
        d.err
    );
    // Google's clock started the token at t0 at the earliest: 3599 s, less 300.
    let st = env.session_state().unwrap();
    let expires = st["expires_at_unix"].as_u64().unwrap();
    assert!(expires <= t0 + 3599 + 1 - 300, "{expires} {t0}");
    let insert = gcp.find("POST", "/zones/us-central1-a/instances")[0].json();
    let secs = insert["scheduling"]["maxRunDuration"]["seconds"]
        .as_u64()
        .unwrap();
    assert!((3200..=3299).contains(&secs), "maxRunDuration {secs}");
    assert_cleaned(&gcp, &vm_name(&gcp));
    token_not_shown(&d);
}

/// A token given 20 s before its expiry, test timings (8 s of margin): the
/// session ends on its own, and the VM is deleted while the token is valid.
#[test]
fn a_session_cut_by_the_token_ends_and_deletes_the_vm_while_the_token_is_valid() {
    let gcp = FakeGcp::start(Cloud {
        token_life: Some(std::time::Duration::from_secs(20)),
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut run = spawn(
        &env,
        token_timings(&mut cmd, 4000, 1000, 8000),
        Some(TOKEN),
        "up",
        None,
    );
    run.wait_stderr("[5/5] OpenAI API");
    let vm = vm_name(&gcp);
    let insert = gcp.find("POST", "/zones/us-central1-a/instances")[0].json();
    let secs = insert["scheduling"]["maxRunDuration"]["seconds"]
        .as_u64()
        .unwrap();
    if secs > 12 {
        run.signal(libc::SIGTERM);
        let d = run.finish();
        panic!(
            "maxRunDuration {secs} s, for a token that expires 20 s after it was given:\n{}",
            d.err
        );
    }
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    for part in [
        "--max-run cut to ",
        " (asked: 60 min): the token expires at ",
        ", 8 s before, to delete the VM with it.",
        "Session ended: its time ran out, 8 s before the token expires (",
        &format!("Deleting {vm} ... done."),
    ] {
        assert!(d.err.contains(part), "missing {part:?} in:\n{}", d.err);
    }
    assert_cleaned(&gcp, &vm);
    assert_eq!(
        gcp.with(|c| c.expired_uses),
        0,
        "a call used the token after its expiry"
    );
    token_not_shown(&d);
}

/// Production timings: 20 min of token leave 14 min of session, and
/// qwen3:0.6b needs 12.6 min of boot timeout and 5 min of serving.
#[test]
fn a_pasted_token_too_short_for_the_boot_is_refused_before_anything_else() {
    let gcp = FakeGcp::start(Cloud {
        token_life: Some(std::time::Duration::from_mins(20)),
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Ok));
    let d = spawn(
        &env,
        &mut env.command(&gcp.base, &up_args(&[])),
        Some(TOKEN),
        "up",
        None,
    )
    .finish();
    assert_eq!(d.code, 91, "{}", d.err);
    for part in [
        "Token OK: tester@example.com, expires in 19 min.",
        "x0ne-engine cloud: the token expires at ",
        ", in 19 min, and up cannot renew it: the session would end 5 min before, to \
         delete the VM with it, which leaves 14 min, and qwen3:0.6b needs 18 min (up to 13 \
         min to boot and load, then 5 min of serving). Paste a token with at least 23 min \
         left, or use --auth gcloud, which renews it; nothing was created",
    ] {
        assert!(d.err.contains(part), "missing {part:?} in:\n{}", d.err);
    }
    assert!(!d.err.contains("Quote ("), "{}", d.err);
    let asked: Vec<String> = gcp.requests().into_iter().map(|r| r.path).collect();
    assert_eq!(asked, ["/tokeninfo"], "only tokeninfo is read");
    no_creation(&gcp);
    token_not_shown(&d);
}

#[test]
fn a_pasted_token_without_an_expiry_is_refused_before_anything_else() {
    let gcp = FakeGcp::start(Cloud {
        token_life: None,
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Ok));
    let d = spawn(
        &env,
        &mut env.command(&gcp.base, &up_args(&[])),
        Some(TOKEN),
        "up",
        None,
    )
    .finish();
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(
        d.err.contains("Token OK: tester@example.com.\n"),
        "{}",
        d.err
    );
    assert!(
        d.err.contains(
            "x0ne-engine cloud: Google's tokeninfo gave no expiry for this token, and up \
             cannot renew it"
        ),
        "{}",
        d.err
    );
    let asked: Vec<String> = gcp.requests().into_iter().map(|r| r.path).collect();
    assert_eq!(asked, ["/tokeninfo"], "only tokeninfo is read");
    no_creation(&gcp);
    token_not_shown(&d);
}

/// The token is read before the quote, the session starts after "yes": time
/// spent at the prompt counts. Test timings: 10 s of boot, 1 s of serving and
/// 8 s of margin need 19 s of token; 24 s pass at the prompt, 17 s after 7 s.
#[test]
fn a_wait_at_the_prompt_that_leaves_too_little_of_the_token_creates_nothing() {
    let gcp = FakeGcp::start(Cloud {
        token_life: Some(std::time::Duration::from_secs(24)),
        ..Cloud::default()
    });
    let given = gcp.with(|c| c.token_start);
    let env = Env::new(Some(Relay::Ok));
    let pty = Pty::open();
    let mut cmd = env.command(&gcp.base, &["up", "--auth", "stdin", "--max-cost", "10"]);
    let run = spawn(
        &env,
        token_timings(&mut cmd, 10_000, 1000, 8000),
        Some(TOKEN),
        "up",
        Some(&pty),
    );
    pty.wait_screen("Type \"yes\" to create the VM: ");
    let at = given + std::time::Duration::from_secs(7);
    if let Some(d) = at.checked_duration_since(std::time::Instant::now()) {
        std::thread::sleep(d);
    }
    pty.type_in(b"yes\n");
    let d = run.finish();
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(
        d.err.contains(
            " and up cannot renew it: the session would end 8 s before, to delete the VM with it"
        ) && d.err.contains("nothing was created"),
        "{}",
        d.err
    );
    no_creation(&gcp);
    token_not_shown(&d);
}

/// A wait at the prompt that leaves enough: the session is shorter by that
/// wait, and the new cut is said after "yes". Test timings: 4 s of boot, 1 s
/// of serving, 8 s of margin, a 32 s token; "yes" 5 s after the prompt shows.
#[test]
fn a_wait_at_the_prompt_comes_off_the_session_and_is_said() {
    let gcp = FakeGcp::start(Cloud {
        token_life: Some(std::time::Duration::from_secs(32)),
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Ok));
    let pty = Pty::open();
    let mut cmd = env.command(&gcp.base, &["up", "--auth", "stdin", "--max-cost", "10"]);
    let mut run = spawn(
        &env,
        token_timings(&mut cmd, 4000, 1000, 8000),
        Some(TOKEN),
        "up",
        Some(&pty),
    );
    pty.wait_screen("Type \"yes\" to create the VM: ");
    std::thread::sleep(std::time::Duration::from_secs(5));
    pty.type_in(b"yes\n");
    run.wait_stderr("[5/5] OpenAI API");
    run.signal(libc::SIGTERM);
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    let cuts: Vec<&str> = d
        .err
        .lines()
        .filter(|l| l.starts_with("--max-run cut to "))
        .collect();
    assert_eq!(cuts.len(), 2, "{}", d.err);
    let secs = |l: &str| -> u64 {
        l["--max-run cut to ".len()..]
            .split(' ')
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    // Whole seconds of the wall clock, read more than 5 s apart: they differ by
    // 5 or more. 4 leaves 1 s whatever tokeninfo and the quote took.
    assert!(secs(cuts[1]) + 4 <= secs(cuts[0]), "{cuts:?}");
    let st = env.session_state().unwrap();
    let (created, expires) = (
        st["created_at_unix"].as_u64().unwrap(),
        st["expires_at_unix"].as_u64().unwrap(),
    );
    assert!(
        expires - created <= secs(cuts[1]),
        "{created} {expires} {cuts:?}"
    );
    token_not_shown(&d);
}

/// A verdict that comes late leaves less than the boot timeout before the
/// session's end: `up` waits for `serving` until that end, not past it.
/// Test timings: 6 s of boot, 1 s of serving, 3 s of margin, a 14 s token;
/// the verdict comes 5 s after `connect`, `serving` never.
#[test]
fn the_wait_for_serving_stops_at_the_end_of_the_session() {
    let gcp = FakeGcp::start(Cloud {
        token_life: Some(std::time::Duration::from_secs(14)),
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::LateVerdict(5)));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(
        &env,
        token_timings(&mut cmd, 6000, 1000, 3000),
        Some(TOKEN),
        "up",
        None,
    )
    .finish();
    assert_eq!(d.code, 93, "{}", d.err);
    assert!(d.err.contains("      VERIFIED."), "{}", d.err);
    assert!(
        d.err
            .contains("x0ne-engine cloud: the session reached its end (")
            && d.err.contains(") before the model was served"),
        "{}",
        d.err
    );
    assert_cleaned(&gcp, &vm_name(&gcp));
    assert_eq!(
        gcp.with(|c| c.expired_uses),
        0,
        "a call used the token after its expiry"
    );
    token_not_shown(&d);
}

#[test]
fn without_a_terminal_and_without_yes_nothing_is_created() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &["up", "--auth", "stdin"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 90, "{}", d.err);
    assert!(d.err.contains("no terminal to answer \"yes\""), "{}", d.err);
    assert!(d.err.contains("Nothing was created"), "{}", d.err);
    no_creation(&gcp);
}

#[test]
fn answering_anything_but_yes_creates_nothing() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let pty = Pty::open();
    let mut cmd = env.command(&gcp.base, &["up", "--auth", "stdin"]);
    let run = spawn(&env, &mut cmd, Some(TOKEN), "up", Some(&pty));
    pty.wait_screen("Type \"yes\" to create the VM: ");
    pty.type_in(b"y\n");
    let d = run.finish();
    assert_eq!(d.code, 90, "{}", d.err);
    assert!(
        d.err.contains("not confirmed; nothing was created"),
        "{}",
        d.err
    );
    no_creation(&gcp);
}

#[test]
fn ctrl_c_at_the_yes_prompt_creates_nothing() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let pty = Pty::open();
    let mut cmd = env.command(&gcp.base, &["up", "--auth", "stdin"]);
    let run = spawn(&env, &mut cmd, Some(TOKEN), "up", Some(&pty));
    pty.wait_screen("Type \"yes\" to create the VM: ");
    pty.type_in(b"\x03");
    let d = run.finish();
    assert_eq!(d.code, 128 + libc::SIGINT, "{}", d.err);
    no_creation(&gcp);
    assert!(gcp.deletes().is_empty());
}

// ── up: every way it can end deletes the VM ─────────────────────────────────

#[test]
fn spot_preemption_while_serving_is_exit_93_and_the_rule_is_deleted() {
    let gcp = FakeGcp::start(Cloud {
        boot: Boot::Preempted { after: 2, then: 6 },
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 93, "{}", d.err);
    assert!(d.err.contains("spot preemption"), "{}", d.err);
    assert_cleaned(&gcp, &vm_name(&gcp));
    assert!(env.relay_log("cmds").contains(r#""cmd":"stop""#));
}

#[test]
fn the_boot_timeout_saves_the_serial_console_privately_and_deletes() {
    let gcp = FakeGcp::start(Cloud {
        boot: Boot::Never,
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    cmd.env("X0NE_ENGINE_CLOUD_TEST_BOOT_TIMEOUT_MS", "1500");
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 93, "{}", d.err);
    assert!(
        d.err
            .contains("no boot within 0 min of the creation; serial console saved to "),
        "{}",
        d.err
    );
    let dir = env.state.join("x0ne-engine");
    let logs: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .collect();
    assert_eq!(logs.len(), 1, "{logs:?}");
    let name = logs[0].file_name().unwrap().to_string_lossy().to_string();
    assert!(
        name.starts_with("serial-x0ne-engine-")
            && Path::new(&name).extension().is_some_and(|e| e == "txt"),
        "{name}"
    );
    assert_eq!(
        fs::metadata(&logs[0]).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert!(fs::read_to_string(&logs[0])
        .unwrap()
        .contains("SERIAL-MARKER"));
    // Text from the VM is untrusted: saved, never printed.
    assert!(
        !d.err.contains("SERIAL-MARKER") && !d.err.contains("evil title"),
        "{}",
        d.err
    );
    assert_cleaned(&gcp, &vm_name(&gcp));
}

// ── the TD's refusal, read on the serial console while `up` waits ──────────

/// The serial console of the VM refused on 2026-10-25, cut to two status
/// blocks: 0 restarts, then 3, both exit 76 (Intel rates the host OutOfDate).
const RUN_B: &str = include_str!("serial/td-refused-76.txt");
const RUN_B_LINE: &str = "x0ne-infer: live TCB verification failed: TCB status OutOfDate \
    for FMSPC 00806f050000 is not accepted (level dated 2026-02-11T00:00:00Z)";

fn run_b_blocks() -> Vec<String> {
    let at = RUN_B.find("x0ne-cvm-status: begin uptime_s=61").unwrap();
    vec![RUN_B[..at].into(), RUN_B[at..].into()]
}

/// A status block of x0ne-OS after `restarts` restarts, each exit `code`
/// after x0ne-infer printed `said`.
fn status_block(code: i32, restarts: u64, said: &str) -> String {
    format!(
        "x0ne-cvm-status: begin uptime_s=91\r\n\
         x0ne-cvm-status: infer=Result=exit-code NRestarts={restarts} ExecMainStatus={code} ActiveState=activating SubState=auto-restart\r\n\
         x0ne-cvm-status: infer_tail=Started x0ne-infer.service - x0ne confidential inference server (I5).|{said}|\
         x0ne-infer.service: Main process exited, code=exited, status={code}/n/a|\
         x0ne-infer.service: Failed with result 'exit-code'.|\r\n\
         x0ne-cvm-status: end\r\n"
    )
}

fn serial_starts(gcp: &FakeGcp) -> Vec<String> {
    gcp.find("GET", "/serialPort")
        .iter()
        .map(|r| {
            r.query
                .split('&')
                .find_map(|kv| kv.strip_prefix("start="))
                .unwrap_or("")
                .to_string()
        })
        .collect()
}

fn saved_serial(env: &Env) -> Vec<std::path::PathBuf> {
    fs::read_dir(env.state.join("x0ne-engine"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("serial-"))
        })
        .collect()
}

/// Exit 76 at its 3rd restart, Intel's rating: `up` stops at the block that
/// says so, long before the verdict deadline (60 s here), and deletes the VM.
#[test]
fn a_final_refusal_on_the_serial_console_ends_the_wait_and_deletes_the_vm() {
    let gcp = FakeGcp::start(Cloud {
        serial: "SeaBIOS\r\n".into(),
        serial_more: run_b_blocks(),
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Hang));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    cmd.env("X0NE_ENGINE_CLOUD_TEST_BOOT_TIMEOUT_MS", "60000");
    let t = std::time::Instant::now();
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 97, "{}", d.err);
    assert!(t.elapsed().as_secs() < 40, "{:?}", t.elapsed());
    assert!(
        d.err.contains(&format!(
            "REFUSED by x0ne-infer in the VM (exit 76, 3 restarts): {RUN_B_LINE}"
        )),
        "{}",
        d.err
    );
    assert!(
        d.err
            .contains("No question and no session key were sent to the VM."),
        "{}",
        d.err
    );
    assert!(
        d.err.contains(
            "x0ne-infer in the VM refused to serve (exit 76, 3 restarts); serial console saved to "
        ),
        "{}",
        d.err
    );
    // Each read starts where the last one ended (GCE's `next`).
    let first = "SeaBIOS\r\n".len() + run_b_blocks()[0].len();
    let starts = serial_starts(&gcp);
    assert_eq!(
        starts[..2],
        ["0".to_string(), first.to_string()],
        "{starts:?}"
    );
    assert!(env.relay_log("cmds").contains(r#""cmd":"stop""#));
    assert_cleaned(&gcp, &vm_name(&gcp));
    let logs = saved_serial(&env);
    assert_eq!(logs.len(), 1, "{logs:?}");
    assert_eq!(
        fs::metadata(&logs[0]).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert!(fs::read_to_string(&logs[0])
        .unwrap()
        .contains("x0ne-cvm-status: begin uptime_s=61"));
}

/// Two status blocks of x0ne-OS 0.2.7 (the console of 2026-10-30, its
/// `infer=` lines set to an exit 65): systemd's fields, no journal line.
const EXIT_65_0_2_7: &str = include_str!("serial/cvm-0.2.7-exit-65.txt");

/// x0ne-OS 0.2.7 prints no `infer_tail=`: exit 65 at its 3rd restart ends
/// the wait on the code alone, long before the deadline, and deletes the VM.
/// Requiring the tail, `up` waited to its boot deadline and exited 93.
#[test]
fn a_final_exit_code_on_a_0_2_7_console_ends_the_wait_and_deletes_the_vm() {
    let at = EXIT_65_0_2_7
        .find("x0ne-cvm-status: begin uptime_s=62")
        .unwrap();
    let gcp = FakeGcp::start(Cloud {
        serial: "SeaBIOS\r\n".into(),
        serial_more: vec![EXIT_65_0_2_7[..at].into(), EXIT_65_0_2_7[at..].into()],
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Hang));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    cmd.env("X0NE_ENGINE_CLOUD_TEST_BOOT_TIMEOUT_MS", "60000");
    let t = std::time::Instant::now();
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 97, "{}", d.err);
    assert!(t.elapsed().as_secs() < 40, "{:?}", t.elapsed());
    assert!(
        d.err.contains(
            "REFUSED by x0ne-infer in the VM (exit 65, 3 restarts): exit 65 means: \
             the TD's own verifier rejected its evidence"
        ),
        "{}",
        d.err
    );
    assert!(
        d.err
            .contains("an unverified exit code, x0ne-infer's own line is not on it"),
        "{}",
        d.err
    );
    assert!(
        d.err
            .contains("No question and no session key were sent to the VM."),
        "{}",
        d.err
    );
    assert!(env.relay_log("cmds").contains(r#""cmd":"stop""#));
    assert_cleaned(&gcp, &vm_name(&gcp));
    assert_eq!(saved_serial(&env).len(), 1);
}

/// The status blocks of an image whose unit sets `RestartPreventExitStatus=103`:
/// x0ne-infer running, then its exit 103 that systemd does not restart.
const EXIT_103_NO_RESTART: &str = include_str!("serial/cvm-exit-103-no-restart.txt");

/// systemd does not restart x0ne-infer after exit 103: `NRestarts=0`, the
/// unit failed. `up` ends its wait on that block and deletes the VM; counting
/// restarts alone, it waited to its boot deadline and exited 93.
#[test]
fn an_exit_systemd_does_not_restart_ends_the_wait_and_deletes_the_vm() {
    let at = EXIT_103_NO_RESTART
        .find("x0ne-cvm-status: begin uptime_s=62")
        .unwrap();
    let gcp = FakeGcp::start(Cloud {
        serial: "SeaBIOS\r\n".into(),
        serial_more: vec![
            EXIT_103_NO_RESTART[..at].into(),
            EXIT_103_NO_RESTART[at..].into(),
        ],
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Hang));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    cmd.env("X0NE_ENGINE_CLOUD_TEST_BOOT_TIMEOUT_MS", "60000");
    let t = std::time::Instant::now();
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 97, "{}", d.err);
    assert!(t.elapsed().as_secs() < 40, "{:?}", t.elapsed());
    assert!(
        d.err.contains(
            "REFUSED by x0ne-infer in the VM (exit 103, 0 restarts): exit 103 means: \
             NRAS refused the key"
        ),
        "{}",
        d.err
    );
    assert!(
        d.err.contains(
            "systemd stopped restarting x0ne-infer after 0 restarts (ActiveState=failed)."
        ),
        "{}",
        d.err
    );
    assert!(
        d.err
            .contains("No question and no session key were sent to the VM."),
        "{}",
        d.err
    );
    assert!(env.relay_log("cmds").contains(r#""cmd":"stop""#));
    assert_cleaned(&gcp, &vm_name(&gcp));
    assert_eq!(saved_serial(&env).len(), 1);
}

/// The console unreadable (IAM without `compute.instances.getSerialPortOutput`):
/// one note after 3 failed reads, no more reads, and the wait ends as before.
#[test]
fn a_console_that_cannot_be_read_leaves_the_wait_as_it_was() {
    let gcp = FakeGcp::start(Cloud {
        serial_error: Some(403),
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Hang));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    cmd.env("X0NE_ENGINE_CLOUD_TEST_BOOT_TIMEOUT_MS", "1500");
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 93, "{}", d.err);
    assert!(
        d.err
            .contains("no verdict within 0 min of the creation; serial console not saved: "),
        "{}",
        d.err
    );
    let notes: Vec<&str> = d
        .err
        .lines()
        .filter(|l| l.contains("serial console unread 3 times in a row"))
        .collect();
    assert_eq!(notes.len(), 1, "{}", d.err);
    assert!(
        notes[0].contains("the wait goes on without it"),
        "{}",
        notes[0]
    );
    assert!(!notes[0].contains('\x1b'), "{:?}", notes[0]);
    // 3 reads while waiting, 1 to save the console at the timeout.
    assert_eq!(gcp.count("GET", "/serialPort"), 4);
    assert!(!d.err.contains("REFUSED"), "{}", d.err);
    assert_cleaned(&gcp, &vm_name(&gcp));
}

/// A console with no status block (another image, an older x0ne-OS): read
/// all along, nothing printed from it, the verdict deadline as before.
#[test]
fn a_console_without_status_blocks_leaves_the_wait_as_it_was() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Hang));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    cmd.env("X0NE_ENGINE_CLOUD_TEST_BOOT_TIMEOUT_MS", "1500");
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 93, "{}", d.err);
    assert!(
        d.err
            .contains("no verdict within 0 min of the creation; serial console saved to "),
        "{}",
        d.err
    );
    assert!(gcp.count("GET", "/serialPort") > 3);
    assert!(
        !d.err.contains("SERIAL-MARKER") && !d.err.contains("evil title"),
        "{}",
        d.err
    );
    assert!(!d.err.contains("REFUSED"), "{}", d.err);
    assert_cleaned(&gcp, &vm_name(&gcp));
}

/// What `up` prints from the console is one line, without control or
/// invisible characters, and bounded.
#[test]
fn a_hostile_refusal_line_is_printed_cleaned_and_bounded() {
    let said = format!(
        "x0ne-infer: gate refused, not serving\x1b[2J\x1b]0;evil title\x07\u{202e}\u{200b}\x08: {}",
        "Z".repeat(3000)
    );
    let gcp = FakeGcp::start(Cloud {
        serial: "SeaBIOS\r\n".into(),
        serial_more: vec![status_block(65, 2, &said)],
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Hang));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    cmd.env("X0NE_ENGINE_CLOUD_TEST_BOOT_TIMEOUT_MS", "60000");
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 97, "{}", d.err);
    let line = d
        .err
        .lines()
        .find(|l| l.contains("REFUSED by x0ne-infer in the VM (exit 65, 2 restarts): "))
        .unwrap_or_else(|| panic!("{}", d.err));
    assert!(
        line.contains("x0ne-infer: gate refused, not serving[2J]0;evil title: ZZZ"),
        "{line:?}"
    );
    assert!(
        line.ends_with('…') && line.chars().count() < 400,
        "{line:?}"
    );
    for bad in ['\x1b', '\x07', '\x08', '\u{202e}', '\u{200b}'] {
        assert!(!d.err.contains(bad), "{bad:?} in {}", d.err);
    }
    assert_cleaned(&gcp, &vm_name(&gcp));
}

/// NRAS unreachable (75) may pass on a later start: however many restarts,
/// the wait goes on to the verdict deadline.
#[test]
fn a_refusal_that_may_pass_does_not_end_the_wait() {
    let said = "x0ne-infer: NRAS attestation failed: NRAS POST \
                https://nras.attestation.nvidia.com/v4/attest/gpu: 503";
    let gcp = FakeGcp::start(Cloud {
        serial: "SeaBIOS\r\n".into(),
        serial_more: (2..8).map(|n| status_block(75, n, said)).collect(),
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Hang));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    cmd.env("X0NE_ENGINE_CLOUD_TEST_BOOT_TIMEOUT_MS", "1500");
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 93, "{}", d.err);
    assert!(d.err.contains("no verdict within 0 min"), "{}", d.err);
    assert!(
        gcp.with(|c| c.serial_more.is_empty()),
        "every block was read"
    );
    assert!(!d.err.contains("REFUSED"), "{}", d.err);
    assert!(!d.err.contains("NRAS attestation failed"), "{}", d.err);
    assert_cleaned(&gcp, &vm_name(&gcp));
}

/// The terminal closes while the session serves (window closed, ssh
/// dropped): SIGHUP, then every write to stderr fails with EIO.
#[test]
fn a_terminal_hangup_while_serving_still_deletes_the_vm() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let pty = Pty::open();
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let run = spawn_on_tty(&env, &mut cmd, Some(TOKEN), "up", &pty);
    pty.wait_screen("[5/5] OpenAI API on");
    let vm = vm_name(&gcp);
    pty.hang_up();
    let d = run.finish();
    assert_eq!(d.code, 0, "exit code after the hangup (101 is a panic)");
    assert_cleaned(&gcp, &vm);
    assert_eq!(env.session_state().unwrap()["phase"], "deleted");
}

/// A delete that Google answers 503 twice is asked again: the VM is deleted
/// and the session ends as one whose delete answered at once. One transient
/// fault at the end of a session must not leave a GPU VM billed.
#[test]
fn a_deletion_that_fails_twice_in_passing_is_asked_again_and_confirmed() {
    let gcp = FakeGcp::start(Cloud {
        vm_delete_faults: 2,
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    run.wait_stderr("[5/5] OpenAI API");
    run.signal(libc::SIGINT);
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(!d.err.contains("NOT CONFIRMED"), "{}", d.err);
    let vm = vm_name(&gcp);
    let asked = gcp
        .deletes()
        .iter()
        .filter(|p| p.ends_with(&format!("/instances/{vm}")))
        .count();
    assert_eq!(asked, 3, "two faults, then the delete that is served");
    assert!(gcp.with(|c| c.vms.is_empty()), "the VM is still there");
}

#[test]
fn a_deletion_that_fails_is_exit_94_with_the_commands_to_run() {
    let gcp = FakeGcp::start(Cloud {
        fail_vm_delete: true,
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    run.wait_stderr("[5/5] OpenAI API");
    run.signal(libc::SIGINT);
    let d = run.finish();
    assert_eq!(d.code, 94, "{}", d.err);
    let vm = vm_name(&gcp);
    assert!(d.err.contains("NOT CONFIRMED"), "{}", d.err);
    assert!(
        d.err.contains(&format!(
            "CRITICAL: the deletion of {vm} could not be confirmed"
        )),
        "{}",
        d.err
    );
    assert!(
        d.err.contains(&format!(
            "gcloud compute instances delete {vm} --zone us-central1-a --project x0ne-test-project --quiet"
        )),
        "{}",
        d.err
    );
    assert!(d.err.contains("x0ne-engine cloud sweep"), "{}", d.err);
    assert!(
        d.err.contains("Internal error ?[31m"),
        "server text is sanitized: {}",
        d.err
    );
    assert!(!d.err.contains('\u{1b}'));
    assert_eq!(env.session_state().unwrap()["phase"], "delete-unconfirmed");
    // The rule is still deleted when the VM is not.
    assert!(gcp
        .deletes()
        .iter()
        .any(|p| p.ends_with(&format!("/global/firewalls/{vm}"))));
}

#[test]
fn no_spot_capacity_is_exit_92_and_the_rule_is_deleted() {
    let gcp = FakeGcp::start(Cloud {
        insert_error: Some((
            "ZONE_RESOURCE_POOL_EXHAUSTED".into(),
            "The zone does not have enough resources".into(),
        )),
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 92, "{}", d.err);
    assert!(
        d.err.contains("no spot H100 capacity in us-central1-a"),
        "{}",
        d.err
    );
    let vm = vm_name(&gcp);
    assert_cleaned(&gcp, &vm);
    // An insert that failed may still land: absence is checked three times.
    let gets = gcp.count("GET", &format!("/zones/us-central1-a/instances/{vm}"));
    assert!(gets >= 3, "absence checked {gets} times");
}

#[test]
fn ctrl_c_while_the_vm_is_created_deletes_it_once_the_creation_returns() {
    let gcp = FakeGcp::start(Cloud {
        hold_vm_insert: true,
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    gcp.wait_request("POST", "/zones/us-central1-a/instances");
    run.signal(libc::SIGINT);
    run.wait_stderr(
        "Stopping: the VM is being created; it is deleted as soon as the creation returns.",
    );
    assert!(
        gcp.deletes().is_empty(),
        "nothing is deleted while the insert runs"
    );
    gcp.with(|c| c.hold_vm_insert = false);
    let d = run.finish();
    assert_eq!(d.code, 130, "{}", d.err);
    assert_cleaned(&gcp, &vm_name(&gcp));
}

#[test]
fn ctrl_c_during_the_boot_deletes_the_vm() {
    let gcp = FakeGcp::start(Cloud {
        boot: Boot::Never,
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    run.wait_stderr("[2/5] Waiting for boot");
    run.signal(libc::SIGINT);
    let d = run.finish();
    assert_eq!(d.code, 130, "{}", d.err);
    assert!(d.err.contains("stopped by Ctrl-C"), "{}", d.err);
    assert_cleaned(&gcp, &vm_name(&gcp));
}

// ── down and status ─────────────────────────────────────────────────────────

#[test]
fn down_from_another_terminal_stops_the_session_without_a_token() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut up = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    up.wait_stderr("[5/5] OpenAI API");
    let mut cmd = env.command(&gcp.base, &["down"]);
    let down = spawn(&env, &mut cmd, None, "down", None).finish();
    assert_eq!(down.code, 0, "{}", down.err);
    assert!(
        down.err.contains("session ended; VM deleted"),
        "{}",
        down.err
    );
    let d = up.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(
        d.err.contains("Session stopped by x0ne-engine cloud down."),
        "{}",
        d.err
    );
    assert_cleaned(&gcp, &vm_name(&gcp));
}

#[test]
fn down_after_up_was_killed_deletes_the_vm_by_name() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut up = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    up.wait_stderr("[5/5] OpenAI API");
    up.signal(libc::SIGKILL);
    assert_eq!(up.finish().code, 128 + libc::SIGKILL);
    let vm = vm_name(&gcp);
    assert!(
        gcp.with(|c| c.vms.contains_key(&vm)),
        "the VM outlives a killed up"
    );
    assert_eq!(env.session_state().unwrap()["phase"], "serving");

    let mut st = env.command(&gcp.base, &["status"]);
    let s = spawn(&env, &mut st, None, "status", None).finish();
    assert!(
        s.out
            .contains("the up process is GONE: run x0ne-engine cloud down"),
        "{}",
        s.out
    );

    let mut cmd = env.command(&gcp.base, &["down", "--auth", "stdin"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "down", None).finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(
        d.err.contains(&format!("Its VM {vm} deletes itself by ")),
        "{}",
        d.err
    );
    assert_cleaned(&gcp, &vm);
    assert_eq!(env.session_state().unwrap()["phase"], "deleted");
}

#[test]
fn status_without_a_session_and_its_limits() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(None);
    let mut cmd = env.command(&gcp.base, &["status", "--limits"]);
    let d = spawn(&env, &mut cmd, None, "status", None).finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(
        d.out
            .contains("No x0ne-engine cloud session on this machine."),
        "{}",
        d.out
    );
    assert!(d.out.contains("Not protected:"), "{}", d.out);
    assert!(
        d.out
            .contains("signature by the x0ne archive key pinned in"),
        "{}",
        d.out
    );
    assert!(!d.out.contains("TEST key"), "{}", d.out);
    // Every line of LIMITS. Its image reference line names the signing
    // ceremony or the pinned key, as archive_key::RELEASE is (doc_claims.rs).
    for l in x0ne_engine_cloud::up::LIMITS {
        assert!(d.out.contains(&format!("- {l}\n")), "{l}\n{}", d.out);
    }
    assert!(gcp.requests().is_empty(), "status needs no API call");
}

/// A killed `up` whose VM still runs, the lock held by another process.
fn killed_up_with_the_lock_held(gcp: &FakeGcp, env: &Env) -> (String, std::fs::File) {
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut up = spawn(env, &mut cmd, Some(TOKEN), "up", None);
    up.wait_stderr("[5/5] OpenAI API");
    up.signal(libc::SIGKILL);
    assert_eq!(up.finish().code, 128 + libc::SIGKILL);
    let vm = vm_name(gcp);
    assert!(gcp.with(|c| c.vms.contains_key(&vm)));
    assert_eq!(env.session_state().unwrap()["phase"], "serving");
    (vm, hold_lock(env))
}

/// status took the lock for the `up` and said "alive": nothing told the user
/// to run `down`, the VM ran to --max-run.
#[test]
fn status_says_a_killed_up_is_gone_while_another_process_holds_the_lock() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let (_vm, _lock) = killed_up_with_the_lock_held(&gcp, &env);
    let mut st = env.command(&gcp.base, &["status"]);
    let s = spawn(&env, &mut st, None, "status", None).finish();
    assert_eq!(s.code, 0, "{}", s.err);
    assert!(
        s.out
            .contains("the up process is GONE: run x0ne-engine cloud down"),
        "{}",
        s.out
    );
}

/// sweep protected the dead session's VM as the running one's.
#[test]
fn sweep_deletes_a_killed_ups_vm_while_another_process_holds_the_lock() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let (vm, _lock) = killed_up_with_the_lock_held(&gcp, &env);
    let mut cmd = env.command(&gcp.base, &["sweep", "--yes", "--auth", "stdin"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "sweep", None).finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(!d.err.contains("is not listed"), "{}", d.err);
    assert_cleaned(&gcp, &vm);
}

/// down stopped the second `up` (nothing created), said so with exit 0, and
/// left the dead session's state, its VM to --max-run.
#[test]
fn down_does_not_stop_a_second_up_in_place_of_a_dead_session() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut first = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    first.wait_stderr("[5/5] OpenAI API");
    let vm = vm_name(&gcp);
    let (dead, b, _pty) = second_up_in_its_window(&gcp, &env, first);
    let mut cmd = env.command(&gcp.base, &["down"]);
    let d = spawn(&env, &mut cmd, None, "down", None).finish();
    let state_then = env.session_state().unwrap();
    b.signal(libc::SIGINT);
    let bd = b.finish();
    assert_eq!(state_then, dead, "the window closed before down ran");
    assert!(
        !bd.err.contains("stopped by x0ne-engine cloud down"),
        "down stopped the second up: {}",
        bd.err
    );
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(
        d.err.contains(&format!(
            "the up process of session {} is gone",
            dead["session_id"].as_str().unwrap()
        )),
        "{}",
        d.err
    );
    assert!(
        d.err.contains(&format!("Its VM {vm} deletes itself by ")),
        "{}",
        d.err
    );
    // "down again" once that process ends found the second up's state.
    assert!(
        d.err
            .contains("x0ne-engine cloud sweep --project x0ne-test-project"),
        "{}",
        d.err
    );
}

/// status took any session id on the socket for the state's up.
#[test]
fn status_says_gone_while_a_second_up_serves_its_socket() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut first = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    first.wait_stderr("[5/5] OpenAI API");
    let (dead, b, _pty) = second_up_in_its_window(&gcp, &env, first);
    let mut st = env.command(&gcp.base, &["status"]);
    let s = spawn(&env, &mut st, None, "status", None).finish();
    let state_then = env.session_state().unwrap();
    b.signal(libc::SIGINT);
    let _ = b.finish();
    assert_eq!(state_then, dead, "the window closed before status ran");
    assert_eq!(s.code, 0, "{}", s.err);
    assert!(
        s.out
            .contains("the up process is GONE: run x0ne-engine cloud down"),
        "{}",
        s.out
    );
}

/// sweep took any session id on the socket for the state's up, and left
/// the dead session's VM out of its list.
#[test]
fn sweep_lists_a_dead_sessions_vm_while_a_second_up_serves_its_socket() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut first = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    first.wait_stderr("[5/5] OpenAI API");
    let vm = vm_name(&gcp);
    let (vms, rules) = gcp.with(|c| (c.vms.clone(), c.firewalls.clone()));
    let (dead, b, _pty) = second_up_in_its_window(&gcp, &env, first);
    gcp.with(|c| {
        c.vms.extend(vms);
        c.firewalls.extend(rules);
    });
    let mut cmd = env.command(&gcp.base, &["sweep", "--yes", "--auth", "stdin"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "sweep", None).finish();
    let state_then = env.session_state().unwrap();
    b.signal(libc::SIGINT);
    let _ = b.finish();
    assert_eq!(state_then, dead, "the window closed before sweep ran");
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(!d.err.contains("is not listed"), "{}", d.err);
    assert_cleaned(&gcp, &vm);
}

/// A serving `up` frozen by SIGSTOP, as Ctrl-Z freezes it: it holds the lock
/// and its socket accepts, but nothing answers. SIGCONT and SIGINT on drop.
struct Stopped(Option<Run>);

impl Stopped {
    fn resume(mut self) -> Run {
        let r = self.0.take().unwrap();
        r.signal(libc::SIGCONT);
        r
    }
}

impl Drop for Stopped {
    fn drop(&mut self) {
        if let Some(r) = &self.0 {
            r.signal(libc::SIGCONT);
            r.signal(libc::SIGINT);
        }
    }
}

fn stopped_up(gcp: &FakeGcp, env: &Env) -> (String, Stopped) {
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut up = spawn(env, &mut cmd, Some(TOKEN), "up", None);
    up.wait_stderr("[5/5] OpenAI API");
    up.signal(libc::SIGSTOP);
    (vm_name(gcp), Stopped(Some(up)))
}

/// Resumed, the `up` ends on SIGINT and deletes its VM.
fn resume_and_end(gcp: &FakeGcp, stopped: Stopped, vm: &str) {
    let up = stopped.resume();
    up.signal(libc::SIGINT);
    let d = up.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert_cleaned(gcp, vm);
}

/// status said "GONE: run down" of a stopped up; the same for
/// an up busy over 2 s, or older than the session id command.
#[test]
fn status_says_a_stopped_up_does_not_answer() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let (vm, stopped) = stopped_up(&gcp, &env);
    let mut st = env.command(&gcp.base, &["status"]);
    let s = spawn(&env, &mut st, None, "status", None).finish();
    assert_eq!(s.code, 0, "{}", s.err);
    assert!(
        s.out.contains("the up process does not answer"),
        "{}",
        s.out
    );
    assert!(!s.out.contains("GONE"), "{}", s.out);
    resume_and_end(&gcp, stopped, &vm);
}

/// sweep --yes deleted the VM of a stopped up as a dead session's.
#[test]
fn sweep_keeps_the_vm_of_a_stopped_up() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let (vm, stopped) = stopped_up(&gcp, &env);
    let mut cmd = env.command(&gcp.base, &["sweep", "--yes", "--auth", "stdin"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "sweep", None).finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(
        d.err.contains(&format!("({vm}) is not listed")),
        "{}",
        d.err
    );
    assert!(gcp.with(|c| c.vms.contains_key(&vm)), "the VM was swept");
    resume_and_end(&gcp, stopped, &vm);
}

/// down said a stopped up was gone and another process held the session.
#[test]
fn down_says_a_stopped_up_does_not_answer() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let (vm, stopped) = stopped_up(&gcp, &env);
    let mut cmd = env.command(&gcp.base, &["down"]);
    let d = spawn(&env, &mut cmd, None, "down", None).finish();
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(
        d.err.contains("does not answer on its control socket"),
        "{}",
        d.err
    );
    assert!(!d.err.contains("is gone"), "{}", d.err);
    assert!(gcp.with(|c| c.vms.contains_key(&vm)));
    let up = stopped.resume();
    let mut cmd = env.command(&gcp.base, &["down"]);
    let d = spawn(&env, &mut cmd, None, "down", None).finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert_eq!(up.finish().code, 0);
    assert_cleaned(&gcp, &vm);
}

// ── the token ───────────────────────────────────────────────────────────────

#[test]
fn a_pasted_token_is_never_echoed_and_the_terminal_is_restored() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(None);
    let pty = Pty::open();
    let mut cmd = env.command(&gcp.base, &["quote"]);
    let run = spawn(&env, &mut cmd, None, "quote", Some(&pty));
    pty.wait_screen("Paste a Google Cloud access token (hidden, kept in memory only): ");
    pty.wait_echo_off();
    pty.type_in(format!("{TOKEN}\n").as_bytes());
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(d.out.contains("Quote (Cloud Billing catalog"), "{}", d.out);
    assert!(
        !contains(&pty.screen(), TOKEN.as_bytes()),
        "echoed: {:?}",
        String::from_utf8_lossy(&pty.screen())
    );
    assert!(pty.echo_on(), "echo not restored");
    assert!(env.files_containing(TOKEN.as_bytes()).is_empty());
}

#[test]
fn ctrl_c_at_the_token_prompt_restores_the_terminal() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(None);
    let pty = Pty::open();
    let mut cmd = env.command(&gcp.base, &["quote"]);
    let run = spawn(&env, &mut cmd, None, "quote", Some(&pty));
    pty.wait_screen("Paste a Google Cloud access token");
    pty.wait_echo_off();
    pty.type_in(b"ya29.partial\x03");
    let d = run.finish();
    assert_eq!(d.code, 128 + libc::SIGINT, "{}", d.err);
    assert!(pty.echo_on(), "echo not restored after Ctrl-C");
    assert!(gcp.requests().is_empty());
}

#[test]
fn a_gcloud_in_path_is_never_run() {
    // A gcloud installed at the fixed paths is hidden from the binary only
    // (own mount namespace): the only gcloud left is the decoy in $PATH.
    let installed: Vec<&Path> = ["/usr/bin/gcloud", "/usr/lib/google-cloud-sdk/bin/gcloud"]
        .iter()
        .map(Path::new)
        .filter(|p| p.is_file())
        .collect();
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    for args in [vec!["quote", "--auth", "gcloud"], vec!["login", "gcp"]] {
        let mut cmd = env.command(&gcp.base, &args);
        if !installed.is_empty() {
            hide_files(&mut cmd, &installed);
        }
        let d = spawn(&env, &mut cmd, None, "gcloud", None).finish();
        assert_eq!(d.code, 91, "{args:?}: {}", d.err);
        assert!(d.err.contains("never looked up in $PATH"), "{}", d.err);
    }
    assert!(env.decoys_ran().is_empty(), "{:?}", env.decoys_ran());
    assert!(gcp.requests().is_empty());
}

#[test]
fn a_token_google_refuses_stops_at_preflight() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(
        &env,
        &mut cmd,
        Some("ya29.WRONG-token-0123456789abcdef"),
        "up",
        None,
    )
    .finish();
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(d.err.contains("Google refused the token"), "{}", d.err);
    no_creation(&gcp);
}

/// No answer from tokeninfo is not Google refusing the token: the message
/// says the network failed (read offline as a refusal until 2026-10-30).
#[test]
fn a_tokeninfo_that_cannot_be_reached_is_not_called_a_refusal() {
    // A loopback port nobody listens on: the connection is refused at once.
    let closed = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", l.local_addr().unwrap())
    };
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&closed, &["quote", "--auth", "stdin"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "offline", None).finish();
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(!d.err.contains("refused the token"), "{}", d.err);
    assert!(
        d.err.contains("no answer from Google's tokeninfo"),
        "{}",
        d.err
    );
    assert!(d.err.contains("not a verdict on the token"), "{}", d.err);
}

// ── where the token may go ──────────────────────────────────────────────────

#[test]
fn api_overrides_off_the_loopback_are_refused() {
    let env = Env::new(Some(Relay::Ok));
    for base in [
        "https://compute.googleapis.com.evil.example",
        "http://example.com:80",
        "http://127.0.0.1:1@evil.example:9",
        "http://127.0.0.1",
        "https://127.0.0.1:443",
    ] {
        let mut cmd = env.command(base, &["quote", "--auth", "stdin"]);
        let d = spawn(&env, &mut cmd, Some(TOKEN), "override", None).finish();
        assert_eq!(d.code, 2, "{base}: {}", d.err);
        assert!(
            d.err
                .contains("refused: test overrides go to http://127.0.0.1:<port> only"),
            "{}",
            d.err
        );
    }
}

#[test]
fn test_timings_without_the_loopback_override_are_refused() {
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command("unused", &["quote", "--auth", "stdin"]);
    cmd.env_remove("X0NE_ENGINE_GCP_ENDPOINT");
    let d = spawn(&env, &mut cmd, Some(TOKEN), "knobs", None).finish();
    assert_eq!(d.code, 2, "{}", d.err);
    assert!(
        d.err
            .contains("X0NE_ENGINE_CLOUD_TEST_POLL_MS is a test setting"),
        "{}",
        d.err
    );
}

#[test]
fn azure_is_refused_by_name_before_any_token() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &["quote", "--provider", "azure"]);
    let d = spawn(&env, &mut cmd, None, "azure", None).finish();
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(
        d.err.contains("SnpViaParavisor") && d.err.contains("TdxQuoteDirect"),
        "{}",
        d.err
    );
    assert!(gcp.requests().is_empty());
}

// ── the quote ───────────────────────────────────────────────────────────────

#[test]
fn the_quote_reads_the_billing_catalog() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(None);
    let mut cmd = env.command(&gcp.base, &["quote", "--auth", "stdin"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "quote", None).finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(
        d.out.contains("Quote (Cloud Billing catalog, "),
        "{}",
        d.out
    );
    assert!(
        d.out
            .contains("a3-highgpu-1g spot, 1x H100 80GB, Intel TDX"),
        "{}",
        d.out
    );
    assert!(
        d.out.contains("  6.06 USD/h"),
        "5.57 + 26 x 0.01 + 234 x 0.001: {}",
        d.out
    );
    assert!(
        d.out.contains("Confidential Computing on A3, spot") && d.out.contains("  0.44 USD/h"),
        "the confidential surcharge: {}",
        d.out
    );
    assert!(d.out.contains("  0.01 USD/h"), "{}", d.out);
    assert!(d.out.contains("(60 min)"), "{}", d.out);
    assert!(
        d.out.contains("  6.52 USD"),
        "6.064 + 0.4391592 + 0.01: {}",
        d.out
    );
    assert_eq!(
        gcp.count("GET", "/v1/services/6F81-5844-456A/skus"),
        2,
        "both catalog pages read"
    );
    no_creation(&gcp);
}

#[test]
fn without_the_catalog_the_dated_table_is_used_and_said() {
    let gcp = FakeGcp::start(Cloud {
        catalog: None,
        ..Cloud::default()
    });
    let env = Env::new(None);
    let mut cmd = env.command(&gcp.base, &["quote", "--auth", "stdin"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "quote", None).finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(
        d.out.contains("embedded table dated 2026-10-16"),
        "{}",
        d.out
    );
    assert!(d.out.contains("the catalog could not be read"), "{}", d.out);
    assert!(d.out.contains("  6.60 USD/h"), "{}", d.out);
    assert!(d.out.contains("  0.44 USD/h"), "{}", d.out);
    assert!(
        d.out.contains("  7.05 USD"),
        "6.60 + 0.4391592 + 0.01: {}",
        d.out
    );
}

#[test]
fn up_yes_without_max_cost_is_refused_before_any_request() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &["up", "--auth", "stdin", "--yes"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 2, "{}", d.err);
    assert!(
        d.err.contains("up --yes needs --max-cost <USD>"),
        "{}",
        d.err
    );
    assert!(gcp.requests().is_empty());
}

/// A token that outlives the hour, so that `up` quotes the whole default
/// --max-run: a pasted token of 59 min cuts it to 54.
fn hour_long_token() -> Cloud {
    Cloud {
        token_life: Some(std::time::Duration::from_hours(2)),
        ..Cloud::default()
    }
}

#[test]
fn a_quote_above_max_cost_is_refused() {
    let gcp = FakeGcp::start(hour_long_token());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(
        &gcp.base,
        &["up", "--auth", "stdin", "--yes", "--max-cost", "1"],
    );
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 90, "{}", d.err);
    assert!(
        d.err
            .contains("the maximum cost 6.52 USD is above --max-cost 1.00"),
        "{}",
        d.err
    );
    no_creation(&gcp);
    let mut cmd = env.command(&gcp.base, &["quote", "--auth", "stdin", "--max-cost", "1"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "quote", None).finish();
    assert_eq!(d.code, 90, "{}", d.err);
}

/// --max-cost holds the total: 6.08 USD without the confidential surcharge
/// would pass a 6.10 ceiling, 6.52 USD with it does not.
#[test]
fn max_cost_covers_the_confidential_surcharge() {
    let gcp = FakeGcp::start(hour_long_token());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(
        &gcp.base,
        &["up", "--auth", "stdin", "--yes", "--max-cost", "6.10"],
    );
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 90, "{}", d.err);
    assert!(
        d.err
            .contains("the maximum cost 6.52 USD is above --max-cost 6.10"),
        "{}",
        d.err
    );
    no_creation(&gcp);
}

#[test]
fn no_free_quota_in_any_tdx_zone_is_a_preflight_failure() {
    let mut cloud = Cloud::default();
    cloud.quotas.insert("us-central1".into(), (1.0, 1.0));
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(
        d.err
            .contains("no free H100 spot quota (PREEMPTIBLE_NVIDIA_H100_GPUS)"),
        "{}",
        d.err
    );
    assert!(
        d.err
            .contains("europe-west4: 0 (used 0), us-central1: 1 (used 1), us-east5: 0 (used 0)"),
        "{}",
        d.err
    );
    assert_eq!(gcp.count_containing("GET", "/quotaInfos/"), 0);
    no_creation(&gcp);
}

// ── spot quota: regions.get without the H100 metric, Cloud Quotas, vCPU, local SSD ──

/// regions.get listed no H100 metric in three regions of a real project (2026-10-17);
/// the quota was in Cloud Quotas only.
fn cloud_without_legacy_h100() -> Cloud {
    Cloud {
        legacy_h100: false,
        ..Cloud::default()
    }
}

const QUOTA_INFO_PATH: &str = "/v1/projects/x0ne-test-project/locations/global/services/compute.googleapis.com/quotaInfos/PREEMPTIBLE-NVIDIA-H100-GPUS-per-project-region";

#[test]
fn the_legacy_h100_metric_when_listed_is_read_and_cloud_quotas_is_not_asked() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &["quote", "--auth", "stdin"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "quote", None).finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(
        d.err
            .contains("Preflight: H100 spot quota in us-central1: 1 (used 0)."),
        "{}",
        d.err
    );
    assert_eq!(gcp.count_containing("GET", "/quotaInfos/"), 0);
}

#[test]
fn without_the_legacy_metric_the_cloud_quotas_value_of_the_region_picks_the_zone() {
    let gcp = FakeGcp::start(cloud_without_legacy_h100());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&["--max-run", "30m"]));
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    run.wait_stderr("[5/5] OpenAI API");
    let vm = vm_name(&gcp);
    run.signal(libc::SIGTERM);
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(
        d.err.contains(
            "Preflight: H100 spot quota in us-central1: 1 (Cloud Quotas API, \
             PREEMPTIBLE-NVIDIA-H100-GPUS-per-project-region, region entry), used 0 (H100 GPUs \
             of this project's VMs in us-central1: Cloud Quotas gives no usage)."
        ),
        "{}",
        d.err
    );
    assert!(gcp.count("GET", QUOTA_INFO_PATH) >= 1);
    assert_cleaned(&gcp, &vm);
}

#[test]
fn no_cloud_quotas_value_for_any_candidate_region_is_a_preflight_failure() {
    for dims in [
        vec![(None, None)],
        vec![(Some("asia-east1"), Some("8")), (None, Some("0"))],
    ] {
        let gcp = FakeGcp::start(Cloud {
            quota_info: QuotaInfo::Dims(dims.clone()),
            ..cloud_without_legacy_h100()
        });
        let env = Env::new(Some(Relay::Ok));
        let mut cmd = env.command(&gcp.base, &up_args(&[]));
        let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
        assert_eq!(d.code, 91, "{dims:?}: {}", d.err);
        assert!(
            d.err.contains(
                "no free H100 spot quota (PREEMPTIBLE_NVIDIA_H100_GPUS): europe-west4: 0 \
                 (Cloud Quotas API, PREEMPTIBLE-NVIDIA-H100-GPUS-per-project-region, default entry)"
            ),
            "{dims:?}: {}",
            d.err
        );
        assert!(d.err.contains("us-east5: 0 (Cloud Quotas API"), "{}", d.err);
        no_creation(&gcp);
    }
}

#[test]
fn h100_gpus_on_the_projects_vms_are_the_cloud_quotas_usage() {
    let mut cloud = cloud_without_legacy_h100();
    cloud.foreign_vms.push((
        "us-central1-b".into(),
        json!({"name": "someone-elses", "status": "RUNNING",
               "machineType": "https://www.googleapis.com/compute/v1/projects/x0ne-test-project/zones/us-central1-b/machineTypes/a3-highgpu-1g",
               "guestAccelerators": [{"acceleratorType": "https://www.googleapis.com/compute/v1/projects/x0ne-test-project/zones/us-central1-b/acceleratorTypes/nvidia-h100-80gb", "acceleratorCount": 1}],
               "scheduling": {"provisioningModel": "SPOT"}}),
    ));
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&["--zone", "us-central1-a"]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(
        d.err.contains(
            "us-central1: 1 (Cloud Quotas API, PREEMPTIBLE-NVIDIA-H100-GPUS-per-project-region, \
             region entry), used 1 (H100 GPUs of this project's VMs in us-central1"
        ),
        "{}",
        d.err
    );
    no_creation(&gcp);
}

#[test]
fn a_cloud_quotas_api_that_refuses_warns_and_lets_google_decide_at_create() {
    for status in [400, 403, 404] {
        let gcp = FakeGcp::start(Cloud {
            quota_info: QuotaInfo::Error(status),
            ..cloud_without_legacy_h100()
        });
        let env = Env::new(Some(Relay::Ok));
        let mut cmd = env.command(
            &gcp.base,
            &up_args(&["--zone", "us-central1-a", "--max-run", "30m"]),
        );
        let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
        run.wait_stderr("[5/5] OpenAI API");
        let vm = vm_name(&gcp);
        run.signal(libc::SIGTERM);
        let d = run.finish();
        assert_eq!(d.code, 0, "{status}: {}", d.err);
        assert!(
            d.err.contains(&format!(
                "Preflight: H100 spot quota in us-central1 not readable (Cloud Quotas API: HTTP {status}"
            )),
            "{status}: {}",
            d.err
        );
        assert!(
            d.err.contains(
                "Google refuses the create if there is none, nothing is billed. No x0ne-engine VM running."
            ),
            "{status}: {}",
            d.err
        );
        assert_cleaned(&gcp, &vm);
    }
}

#[test]
fn too_little_spot_vcpu_quota_is_a_preflight_failure_before_any_insert() {
    for (metric, limit, usage, need) in [
        ("PREEMPTIBLE_CPUS", 0.0, 0.0, 26),
        ("PREEMPTIBLE_CPUS", 26.0, 1.0, 26),
        ("PREEMPTIBLE_LOCAL_SSD_GB", 375.0, 0.0, 750),
    ] {
        let mut cloud = cloud_without_legacy_h100();
        let q = cloud.region_quotas.get_mut("us-central1").unwrap();
        q.retain(|(m, _, _)| *m != metric);
        q.push((metric, limit, usage));
        let gcp = FakeGcp::start(cloud);
        let env = Env::new(Some(Relay::Ok));
        let mut cmd = env.command(&gcp.base, &up_args(&["--zone", "us-central1-a"]));
        let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
        assert_eq!(d.code, 91, "{metric}: {}", d.err);
        assert!(
            d.err.contains(&format!(
                "no zone has the spot quota a3-highgpu-1g needs: us-central1: {metric} has {} \
                 free (limit {limit}, used {usage}), a3-highgpu-1g needs {need}",
                limit - usage
            )),
            "{metric}: {}",
            d.err
        );
        no_creation(&gcp);
    }
}

#[test]
fn a_zone_short_of_spot_vcpus_is_skipped_for_the_next_candidate() {
    // europe-west4 has H100 quota here but PREEMPTIBLE_CPUS 0: us-central1-a is picked.
    let mut cloud = Cloud::default();
    cloud.quotas.insert("europe-west4".into(), (1.0, 0.0));
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &["quote", "--auth", "stdin"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "quote", None).finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(
        d.err
            .contains("Preflight: H100 spot quota in us-central1: 1 (used 0)."),
        "{}",
        d.err
    );
}

#[test]
fn a_running_session_in_the_project_blocks_a_second_one() {
    let mut cloud = Cloud::default();
    cloud.vms.insert(
        "x0ne-engine-abcdef".into(),
        Vm {
            zone: "us-central1-a".into(),
            status: "RUNNING".into(),
            labels: json!({"x0ne-engine": "session"}),
            gets: 100,
        },
    );
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(
        d.err
            .contains("x0ne-engine VM already running: x0ne-engine-abcdef"),
        "{}",
        d.err
    );
    assert!(
        d.err.contains("one session at a time per project"),
        "{}",
        d.err
    );
    no_creation(&gcp);
}

#[test]
fn a_missing_image_is_a_preflight_failure() {
    let mut cloud = Cloud::default();
    cloud.images.clear();
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(
        d.err
            .contains("x0ne-engine cloud image import <disk.tar.gz>"),
        "{}",
        d.err
    );
    no_creation(&gcp);
}

// ── sweep ───────────────────────────────────────────────────────────────────

#[test]
fn sweep_deletes_only_what_carries_the_prefix_and_the_mark() {
    let gcp = FakeGcp::start(crowded_cloud());
    let env = Env::new(None);
    let mut cmd = env.command(&gcp.base, &["sweep", "--yes", "--auth", "stdin"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "sweep", None).finish();
    assert_eq!(d.code, 0, "{}\n{}", d.out, d.err);
    assert!(d.out.contains("Deleted 6 resources."), "{}", d.out);
    let mut deleted = gcp.deletes();
    deleted.sort();
    let p = |s: &str| format!("/compute/v1/projects/{PROJECT}/{s}");
    let mut want = vec![
        p("zones/us-central1-a/instances/x0ne-engine-a1b2c3"),
        p("global/firewalls/x0ne-engine-a1b2c3"),
        p("regions/us-central1/addresses/x0ne-engine-ip1"),
        p("global/addresses/x0ne-engine-gip"),
        p("zones/us-central1-a/disks/x0ne-engine-a1b2c3"),
        "/storage/v1/b/x0ne-engine-import-0123456789ab/o/x0ne-cvm-0-1-0.tar.gz".to_string(),
        "/storage/v1/b/x0ne-engine-import-0123456789ab".to_string(),
    ];
    want.sort();
    assert_eq!(deleted, want);
    gcp.with(|c| {
        assert!(c.vms.contains_key("x0ne-engine-nolabel") && c.vms.contains_key("web-1"));
        assert_eq!(c.disks.len(), 2);
        assert_eq!(c.addresses.len(), 1);
        assert_eq!(c.firewalls.len(), 3);
        assert!(c.buckets.contains_key("x0ne-engine-backups"));
    });
}

#[test]
fn sweep_without_yes_and_without_a_terminal_deletes_nothing() {
    let gcp = FakeGcp::start(crowded_cloud());
    let env = Env::new(None);
    let mut cmd = env.command(&gcp.base, &["sweep", "--auth", "stdin"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "sweep", None).finish();
    assert_eq!(d.code, 2, "{}", d.err);
    assert!(
        d.out.contains("x0ne-engine-a1b2c3"),
        "the list is shown: {}",
        d.out
    );
    assert!(
        !d.out.contains("web-1") && !d.out.contains("x0ne-engine-backups"),
        "{}",
        d.out
    );
    assert!(gcp.deletes().is_empty());
}

#[test]
fn sweep_leaves_the_session_this_machine_is_running() {
    let mut cloud = Cloud::default();
    cloud.vms.insert(
        "x0ne-engine-0dd0dd".into(),
        Vm {
            zone: "us-central1-a".into(),
            status: "TERMINATED".into(),
            labels: json!({"x0ne-engine": "session"}),
            gets: 100,
        },
    );
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut up = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    up.wait_stderr("[5/5] OpenAI API");
    let live = vm_name(&gcp);

    let mut cmd = env.command(&gcp.base, &["sweep", "--yes", "--auth", "stdin"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "sweep", None).finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(
        d.err.contains(&format!(
            "The running session of this machine ({live}) is not listed"
        )),
        "{}",
        d.err
    );
    assert_eq!(
        gcp.deletes(),
        vec![format!(
            "/compute/v1/projects/{PROJECT}/zones/us-central1-a/instances/x0ne-engine-0dd0dd"
        )]
    );
    up.signal(libc::SIGINT);
    assert_eq!(up.finish().code, 0);
    assert_cleaned(&gcp, &live);
}

// ── image import ────────────────────────────────────────────────────────────

#[test]
fn image_import_goes_through_a_transit_bucket_that_is_removed() {
    let mut cloud = Cloud::default();
    cloud.images.clear();
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(None);
    let disk = disk_tar(&env, 300_000);
    let mut cmd = env.command(
        &gcp.base,
        &[
            "image",
            "import",
            disk.to_str().unwrap(),
            "--yes",
            "--auth",
            "stdin",
        ],
    );
    let d = spawn(&env, &mut cmd, Some(TOKEN), "import", None).finish();
    assert_eq!(d.code, 0, "{}\n{}", d.out, d.err);
    assert!(
        d.out.contains(&format!(
            "Image {DEFAULT_IMAGE} is READY in project x0ne-test-project. Transit bucket deleted."
        )),
        "{}",
        d.out
    );
    let created = gcp.find("POST", "/storage/v1/b");
    assert_eq!(created.len(), 1);
    let bucket = created[0].json();
    let name = bucket["name"].as_str().unwrap().to_string();
    assert!(
        name.starts_with("x0ne-engine-import-") && name.len() == "x0ne-engine-import-".len() + 12,
        "{name}"
    );
    assert_eq!(bucket["labels"], json!({"x0ne-engine": "import"}));
    assert_eq!(
        bucket["iamConfiguration"]["publicAccessPrevention"],
        "enforced"
    );
    let image = gcp.find("POST", "/global/images")[0].json();
    assert_eq!(
        image["rawDisk"]["source"],
        format!("https://storage.googleapis.com/{name}/{DEFAULT_IMAGE}.tar.gz")
    );
    assert_eq!(image["guestOsFeatures"][0]["type"], "TDX_CAPABLE");
    gcp.with(|c| {
        assert!(c.buckets.is_empty(), "transit bucket left behind");
        assert_eq!(
            c.images.get(DEFAULT_IMAGE).map(String::as_str),
            Some("READY")
        );
    });
    // The upload session URL carries no token; only the start request does.
    for r in gcp
        .requests()
        .iter()
        .filter(|r| r.path.starts_with("/upload-session/"))
    {
        assert!(r.auth.is_none(), "token resent to the upload session");
    }
}

#[test]
fn image_import_does_nothing_when_the_image_is_there() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(None);
    let disk = disk_tar(&env, 1000);
    let mut cmd = env.command(
        &gcp.base,
        &[
            "image",
            "import",
            disk.to_str().unwrap(),
            "--yes",
            "--auth",
            "stdin",
        ],
    );
    let d = spawn(&env, &mut cmd, Some(TOKEN), "import", None).finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(
        d.out
            .contains("already exists in project x0ne-test-project: nothing to do"),
        "{}",
        d.out
    );
    assert_eq!(gcp.count("POST", "/storage/v1/b"), 0);
}

#[test]
fn image_import_refuses_a_disk_its_reference_does_not_name() {
    let mut cloud = Cloud::default();
    cloud.images.clear();
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(None);
    let disk = disk_tar(&env, 1000);
    let reference = env.path("x0ne-cvm-0.1.0.json");
    fs::write(
        &reference,
        format!(
            r#"{{"schema":"x0ne-engine/reference/v2","provider":"gcp","tee":"intel-tdx",
                "image":{{"name":"x0ne-cvm","version":"0.1.0","gcp_image":"x0ne-cvm-0-1-0","disk_tar_sha256":"{}"}}}}"#,
            "0".repeat(64)
        ),
    )
    .unwrap();
    let mut cmd = env.command(
        &gcp.base,
        &[
            "image",
            "import",
            disk.to_str().unwrap(),
            "--reference",
            reference.to_str().unwrap(),
            "--yes",
            "--auth",
            "stdin",
        ],
    );
    let d = spawn(&env, &mut cmd, Some(TOKEN), "import", None).finish();
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(
        d.err
            .contains("not the reviewed image. Nothing was uploaded"),
        "{}",
        d.err
    );
    assert_eq!(gcp.count("POST", "/storage/v1/b"), 0);
}

/// `image import` caught INT, TERM and HUP only: SIGQUIT and the other catchable
/// ending signals ended it with its transit bucket left in the project.
#[test]
fn every_catchable_ending_signal_during_image_import_deletes_the_transit_bucket() {
    let mut failed = Vec::new();
    for (sig, name) in [
        (libc::SIGINT, "Ctrl-C"),
        (libc::SIGTERM, "SIGTERM"),
        (libc::SIGHUP, "SIGHUP"),
        (libc::SIGQUIT, "Ctrl-\\"),
        (libc::SIGUSR1, "SIGUSR1"),
        (libc::SIGUSR2, "SIGUSR2"),
        (libc::SIGALRM, "SIGALRM"),
        (libc::SIGVTALRM, "SIGVTALRM"),
        (libc::SIGPROF, "SIGPROF"),
        (libc::SIGIO, "SIGIO"),
        (libc::SIGPWR, "SIGPWR"),
        (libc::SIGSTKFLT, "SIGSTKFLT"),
        (libc::SIGXCPU, "SIGXCPU"),
        (libc::SIGXFSZ, "SIGXFSZ"),
        (libc::SIGRTMIN(), "a real-time signal"),
        (libc::SIGRTMAX(), "a real-time signal"),
    ] {
        let mut cloud = Cloud::default();
        cloud.images.clear();
        cloud.hold_image_insert = true;
        let gcp = FakeGcp::start(cloud);
        let env = Env::new(None);
        let disk = disk_tar(&env, 1000);
        let mut cmd = env.command(
            &gcp.base,
            &[
                "image",
                "import",
                disk.to_str().unwrap(),
                "--yes",
                "--auth",
                "stdin",
            ],
        );
        let run = spawn(&env, &mut cmd, Some(TOKEN), "import", None);
        gcp.wait_request("POST", "/global/images");
        assert_eq!(gcp.with(|c| c.buckets.len()), 1);
        run.signal(sig);
        let d = run.finish();
        let left = gcp.with(|c| c.buckets.len());
        let said = d
            .err
            .contains(&format!("{name}: deleting the transit bucket"));
        if d.code != 130 || left != 0 || !said {
            failed.push((sig, d.code, left));
        }
    }
    assert!(
        failed.is_empty(),
        "(signal, exit, buckets left): {failed:?}"
    );
}

// ── image delete ────────────────────────────────────────────────────────────

/// A disk made from `image`, attached to `vm` or to nothing, as
/// disks.aggregatedList shapes it (`sourceImage`, `users`).
fn disk_from(image: &str, vm: Option<&str>) -> (String, serde_json::Value) {
    let api = "https://www.googleapis.com/compute/v1/projects";
    let mut d = json!({
        "name": vm.unwrap_or("spare-disk"),
        "sizeGb": "200",
        "sourceImage": format!("{api}/{PROJECT}/global/images/{image}"),
    });
    if let Some(vm) = vm {
        d["users"] = json!([format!(
            "{api}/{PROJECT}/zones/us-central1-a/instances/{vm}"
        )]);
    }
    ("us-central1-a".to_string(), d)
}

fn image_delete(gcp: &FakeGcp, env: &Env, args: &[&str]) -> Done {
    let mut all = vec!["image", "delete"];
    all.extend_from_slice(args);
    all.extend(["--auth", "stdin"]);
    let mut cmd = env.command(&gcp.base, &all);
    spawn(env, &mut cmd, Some(TOKEN), "image-delete", None).finish()
}

#[test]
fn image_delete_removes_an_image_that_image_import_made() {
    let mut cloud = Cloud::default();
    // A disk made from it and attached to nothing does not hold it.
    cloud.disks.push(disk_from("x0ne-cvm-0-1-0", None));
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(None);
    let d = image_delete(&gcp, &env, &["x0ne-cvm-0-1-0", "--yes"]);
    assert_eq!(d.code, 0, "{}\n{}", d.out, d.err);
    assert!(
        d.out
            .contains("Image x0ne-cvm-0-1-0 deleted from project x0ne-test-project."),
        "{}",
        d.out
    );
    assert_eq!(
        gcp.deletes(),
        vec![format!(
            "/compute/v1/projects/{PROJECT}/global/images/x0ne-cvm-0-1-0"
        )]
    );
    gcp.with(|c| {
        assert!(!c.images.contains_key("x0ne-cvm-0-1-0"));
        assert!(
            c.images.contains_key(DEFAULT_IMAGE),
            "the other image is kept"
        );
    });
}

#[test]
fn image_delete_without_yes_and_without_a_terminal_deletes_nothing() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(None);
    let d = image_delete(&gcp, &env, &["x0ne-cvm-0-1-0"]);
    assert_eq!(d.code, 2, "{}", d.err);
    assert!(d.err.contains("Nothing was deleted"), "{}", d.err);
    assert!(gcp.deletes().is_empty());
}

#[test]
fn answering_anything_but_yes_deletes_no_image() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(None);
    let pty = Pty::open();
    let mut cmd = env.command(
        &gcp.base,
        &["image", "delete", "x0ne-cvm-0-1-0", "--auth", "stdin"],
    );
    let run = spawn(&env, &mut cmd, Some(TOKEN), "image-delete", Some(&pty));
    pty.wait_screen("Type \"yes\" to delete the image: ");
    pty.type_in(b"y\n");
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(d.out.contains("Nothing was deleted."), "{}", d.out);
    assert!(gcp.deletes().is_empty());
}

/// An image a VM still boots from is not deleted: the VM is named, nothing
/// is asked, nothing is deleted.
#[test]
fn image_delete_refuses_an_image_a_vm_boots_from() {
    let mut cloud = Cloud::default();
    cloud
        .disks
        .push(disk_from("x0ne-cvm-0-1-0", Some("x0ne-engine-a1b2c3")));
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(None);
    let d = image_delete(&gcp, &env, &["x0ne-cvm-0-1-0", "--yes"]);
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(d.err.contains("x0ne-engine-a1b2c3"), "{}", d.err);
    assert!(gcp.deletes().is_empty());
}

/// Only what `image import` made: an image without its label is not ours,
/// whatever its name.
#[test]
fn image_delete_refuses_an_image_that_image_import_did_not_make() {
    let mut cloud = Cloud::default();
    cloud.image_labels.remove("x0ne-cvm-0-1-0");
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(None);
    let d = image_delete(&gcp, &env, &["x0ne-cvm-0-1-0", "--yes"]);
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(
        d.err.contains("not made by x0ne-engine cloud image import"),
        "{}",
        d.err
    );
    assert!(gcp.deletes().is_empty());
}

/// The name is judged before the token: an x0ne-cvm-* image, one of them.
#[test]
fn image_delete_names_one_x0ne_cvm_image() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(None);
    for args in [
        vec!["debian-12", "--yes"],
        vec!["--yes"],
        vec!["x0ne-cvm-0-1-0", "x0ne-cvm-0-2-1", "--yes"],
        vec!["X0NE-CVM-0-1-0", "--yes"],
        vec!["x0ne-cvm-../x", "--yes"],
    ] {
        let d = image_delete(&gcp, &env, &args);
        assert_eq!(d.code, 2, "{args:?}: {}", d.err);
        assert!(d.err.contains("x0ne-cvm-"), "{args:?}: {}", d.err);
    }
    assert!(gcp.requests().is_empty(), "{:?}", gcp.requests());
}

#[test]
fn image_delete_of_a_missing_image_deletes_nothing() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(None);
    let d = image_delete(&gcp, &env, &["x0ne-cvm-9-9-9", "--yes"]);
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(
        d.out
            .contains("No image x0ne-cvm-9-9-9 in project x0ne-test-project: nothing to delete."),
        "{}",
        d.out
    );
    assert!(gcp.deletes().is_empty());
}

#[test]
fn an_image_deletion_google_refuses_is_exit_94_with_the_command_to_run() {
    let mut cloud = Cloud::default();
    cloud.fail_image_delete = true;
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(None);
    let d = image_delete(&gcp, &env, &["x0ne-cvm-0-1-0", "--yes"]);
    assert_eq!(d.code, 94, "{}", d.err);
    assert!(
        d.err
            .contains("gcloud compute images delete x0ne-cvm-0-1-0 --project x0ne-test-project"),
        "{}",
        d.err
    );
    gcp.with(|c| assert!(c.images.contains_key("x0ne-cvm-0-1-0")));
}

// ── an image of another build under the same name ───────────────────────────

const REF_SHA: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const OTHER_SHA: &str = "2222222222222222222222222222222222222222222222222222222222222222";

/// A reference naming x0ne-cvm-0-1-0 and the disk sha256 `REF_SHA`.
fn sha_reference(env: &Env) -> std::path::PathBuf {
    let reference = env.path("x0ne-cvm-0.1.0.json");
    fs::write(
        &reference,
        format!(
            r#"{{"schema":"x0ne-engine/reference/v2","provider":"gcp","tee":"intel-tdx",
                "image":{{"name":"x0ne-cvm","version":"0.1.0","gcp_image":"x0ne-cvm-0-1-0","disk_tar_sha256":"{REF_SHA}"}}}}"#
        ),
    )
    .unwrap();
    reference
}

/// The READY image x0ne-cvm-0-1-0 as `image import` describes it, or with no description.
fn cloud_with_image_of(sha: Option<&str>) -> Cloud {
    let mut cloud = Cloud::default();
    if let Some(sha) = sha {
        cloud.image_descriptions.insert(
            "x0ne-cvm-0-1-0".into(),
            format!("x0ne CVM image; disk.tar.gz sha256 {sha}"),
        );
    }
    cloud
}

/// 0.2.0-beta.4 and beta.5 both build `x0ne-cvm-0-2` (rev10, 2026-10-17): a
/// READY image of the other build must not pass as this reference's.
#[test]
fn image_import_refuses_a_ready_image_of_another_build() {
    for described in [Some(OTHER_SHA), None] {
        let gcp = FakeGcp::start(cloud_with_image_of(described));
        let env = Env::new(None);
        let disk = disk_tar(&env, 1000);
        let reference = sha_reference(&env);
        let mut cmd = env.command(
            &gcp.base,
            &[
                "image",
                "import",
                disk.to_str().unwrap(),
                "--reference",
                reference.to_str().unwrap(),
                "--yes",
                "--auth",
                "stdin",
            ],
        );
        let d = spawn(&env, &mut cmd, Some(TOKEN), "import", None).finish();
        assert_eq!(d.code, 91, "{described:?}: {}\n{}", d.out, d.err);
        assert!(
            d.err.contains(
                "x0ne-engine cloud image delete x0ne-cvm-0-1-0 --project x0ne-test-project"
            ),
            "{}",
            d.err
        );
        assert_eq!(gcp.count("POST", "/storage/v1/b"), 0);
        assert_eq!(gcp.count("POST", "/global/images"), 0);
    }
}

#[test]
fn image_import_does_nothing_when_the_ready_image_is_the_references() {
    let gcp = FakeGcp::start(cloud_with_image_of(Some(REF_SHA)));
    let env = Env::new(None);
    let disk = disk_tar(&env, 1000);
    let reference = sha_reference(&env);
    let mut cmd = env.command(
        &gcp.base,
        &[
            "image",
            "import",
            disk.to_str().unwrap(),
            "--reference",
            reference.to_str().unwrap(),
            "--yes",
            "--auth",
            "stdin",
        ],
    );
    let d = spawn(&env, &mut cmd, Some(TOKEN), "import", None).finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(d.out.contains("nothing to do"), "{}", d.out);
    assert_eq!(gcp.count("POST", "/storage/v1/b"), 0);
}

/// The same check in `up`, before the quote and before any VM: the TD of
/// another build may never answer, and the relay then waits the boot timeout.
#[test]
fn up_refuses_a_ready_image_the_reference_does_not_name() {
    for described in [Some(OTHER_SHA), None] {
        let gcp = FakeGcp::start(cloud_with_image_of(described));
        let env = Env::new(Some(Relay::Ok));
        let reference = sha_reference(&env);
        let mut cmd = env.command(
            &gcp.base,
            &up_args(&["--reference", reference.to_str().unwrap()]),
        );
        let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
        assert_eq!(d.code, 91, "{described:?}: {}", d.err);
        assert!(d.err.contains(REF_SHA), "{}", d.err);
        no_creation(&gcp);
    }
}

#[test]
fn up_serves_on_the_image_its_reference_names() {
    let gcp = FakeGcp::start(cloud_with_image_of(Some(REF_SHA)));
    let env = Env::new(Some(Relay::Ok));
    let reference = sha_reference(&env);
    let mut cmd = env.command(
        &gcp.base,
        &up_args(&["--reference", reference.to_str().unwrap()]),
    );
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    run.wait_stderr("[5/5] OpenAI API");
    let vm = vm_name(&gcp);
    run.signal(libc::SIGINT);
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert_cleaned(&gcp, &vm);
}

// ── the image of the relay's reference ───────────────────────────────

/// The relay's config names the image of the reference it verified. Another
/// image or disk than the one `up` checked, or none, is refused before any
/// VM: a drift used to fail at verification, on a VM already paid for.
#[test]
fn up_refuses_a_relay_whose_reference_names_another_image() {
    let other = "x0ne-cvm-9-9-9";
    for (relay, with_reference, named) in [
        (Relay::Image(other, None), false, other),
        (
            Relay::Image("x0ne-cvm-0-1-0", Some(OTHER_SHA)),
            true,
            OTHER_SHA,
        ),
        (Relay::Image(DEFAULT_IMAGE, Some(REF_SHA)), false, REF_SHA),
        (Relay::NoImage, true, "names no image"),
    ] {
        let gcp = FakeGcp::start(cloud_with_image_of(Some(REF_SHA)));
        let env = Env::new(Some(relay));
        let reference = sha_reference(&env);
        let extra = ["--reference", reference.to_str().unwrap()];
        let args = up_args(if with_reference { &extra } else { &[] });
        let mut cmd = env.command(&gcp.base, &args);
        let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
        assert_eq!(d.code, 91, "{relay:?}: {}", d.err);
        assert!(d.err.contains(named), "{relay:?}: {}", d.err);
        assert!(
            d.err.contains("nothing was created"),
            "{relay:?}: {}",
            d.err
        );
        no_creation(&gcp);
    }
}

/// Without `--reference`, `up` and `quote` check the image of the reference
/// the relay takes: the newest installed `x0ne-cvm-<version>.json`, not a
/// manifest next to it, not `DEFAULT_IMAGE`. The relay finds it itself.
#[test]
fn without_reference_up_checks_the_image_of_the_installed_reference() {
    let gcp = FakeGcp::start(cloud_with_image_of(Some(REF_SHA)));
    let env = Env::new(Some(Relay::Image("x0ne-cvm-0-1-0", Some(REF_SHA))));
    let installed = |name: &str, image: &str| {
        fs::write(
            env.references.join(name),
            format!(
                r#"{{"schema":"x0ne-engine/reference/v2","provider":"gcp","tee":"intel-tdx",
                    "image":{{"name":"x0ne-cvm","version":"0","gcp_image":"{image}","disk_tar_sha256":"{REF_SHA}"}}}}"#
            ),
        )
        .unwrap();
    };
    installed("x0ne-cvm-0.0.9.json", "x0ne-cvm-0-0-9");
    installed("x0ne-cvm-0.1.0.json", "x0ne-cvm-0-1-0");
    installed("x0ne-cvm-0.1.0.manifest.json", "x0ne-cvm-manifest");

    let mut cmd = env.command(&gcp.base, &["quote", "--auth", "stdin"]);
    let d = spawn(&env, &mut cmd, Some(TOKEN), "quote", None).finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(d.err.contains("Image x0ne-cvm-0-1-0 found."), "{}", d.err);

    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    run.wait_stderr("[5/5] OpenAI API");
    let vm = vm_name(&gcp);
    run.signal(libc::SIGINT);
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(d.err.contains("Image x0ne-cvm-0-1-0 found."), "{}", d.err);
    assert!(
        !env.relay_log("argv").contains("--reference"),
        "{}",
        env.relay_log("argv")
    );
    assert_cleaned(&gcp, &vm);
}

/// The name server asked for this machine's address is Google's, unless a
/// test build talks to the loopback fake: off it, or not a loopback address,
/// the override is refused before anything is asked.
#[test]
fn the_test_name_server_off_the_loopback_is_refused() {
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command("unused", &up_args(&[]));
    for v in [
        "X0NE_ENGINE_GCP_ENDPOINT",
        "X0NE_ENGINE_CLOUD_TEST_POLL_MS",
        "X0NE_ENGINE_CLOUD_TEST_REFERENCE_DIR",
    ] {
        cmd.env_remove(v);
    }
    cmd.env("X0NE_ENGINE_CLOUD_TEST_MYADDR_SERVER", "127.0.0.1:53");
    let d = spawn(&env, &mut cmd, Some(TOKEN), "knobs", None).finish();
    assert_eq!(d.code, 2, "{}", d.err);
    assert!(
        d.err
            .contains("X0NE_ENGINE_CLOUD_TEST_MYADDR_SERVER is a test setting"),
        "{}",
        d.err
    );
    let gcp = FakeGcp::start(Cloud::default());
    for server in ["8.8.8.8:53", "ns1.google.com:53", "127.0.0.1"] {
        let mut cmd = env.command(&gcp.base, &up_args(&[]));
        cmd.env("X0NE_ENGINE_CLOUD_TEST_MYADDR_SERVER", server);
        let d = spawn(&env, &mut cmd, Some(TOKEN), "knobs", None).finish();
        assert_eq!(d.code, 2, "{server}: {}", d.err);
        assert!(d.err.contains("a loopback ip:port expected"), "{}", d.err);
    }
    no_creation(&gcp);
    assert_eq!(gcp.with(|c| c.dns_queries), 0);
}

// ── the NRAS key (--nras-key-file) ──────────────────────────────────────────

/// Shaped like an NGC key; a canary for every place it must not reach.
const NRAS_KEY: &str = "nvapi-CANARY0nras0key0abcdefghijklmnopqrstuvwxyz0123456789_-ABCDE";

/// The key in `name` under the test directory, with `mode`.
fn nras_key_file(env: &Env, name: &str, body: &str, mode: u32) -> String {
    let p = env.path(name);
    fs::write(&p, body).unwrap();
    fs::set_permissions(&p, fs::Permissions::from_mode(mode)).unwrap();
    p.to_str().unwrap().to_string()
}

/// The VM's metadata items, (key, value), in the order of the insert.
fn metadata_items(gcp: &FakeGcp) -> Vec<(String, String)> {
    let ins = gcp.find("POST", "/zones/us-central1-a/instances");
    ins[0].json()["metadata"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| {
            (
                i["key"].as_str().unwrap().to_string(),
                i["value"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

/// The key reaches the VM in its own metadata item, never in `x0ne-config`
/// (hashed into the quote, published in the attestation document), and
/// nowhere else: not the output, not the relay, not a file of the session.
#[test]
fn up_hands_the_nras_key_to_the_vm_alone_and_prints_it_nowhere() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let file = nras_key_file(&env, "nras.key", &format!("{NRAS_KEY}\n"), 0o600);
    let args = up_args(&["--nras-key-file", &file]);
    let mut run = spawn(
        &env,
        &mut env.command(&gcp.base, &args),
        Some(TOKEN),
        "up",
        None,
    );
    run.wait_stderr("[5/5] OpenAI API");
    let cmdline = fs::read(format!("/proc/{}/cmdline", run.pid())).unwrap();
    assert!(!contains(&cmdline, NRAS_KEY.as_bytes()));
    run.signal(libc::SIGTERM);
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    let items = metadata_items(&gcp);
    let keys: Vec<&str> = items.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        keys,
        ["x0ne-config", "x0ne-nras-key", "block-project-ssh-keys"]
    );
    assert_eq!(items[1].1, NRAS_KEY);
    assert!(!items[0].1.contains("CANARY"), "{}", items[0].1);
    assert!(
        d.err
            .contains("NRAS: the key of --nras-key-file is handed to the VM"),
        "{}",
        d.err
    );
    for text in [&d.out, &d.err] {
        assert!(!text.contains("CANARY"), "{text}");
    }
    assert_eq!(
        env.files_containing(NRAS_KEY.as_bytes()),
        vec![std::path::PathBuf::from(&file)]
    );
    for r in gcp.requests() {
        assert!(!r.path.contains("CANARY") && !r.query.contains("CANARY"));
    }
}

/// Without the option, the metadata is what it was: no key item.
#[test]
fn without_a_key_file_the_vm_gets_no_key_item() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut run = spawn(
        &env,
        &mut env.command(&gcp.base, &up_args(&[])),
        Some(TOKEN),
        "up",
        None,
    );
    run.wait_stderr("[5/5] OpenAI API");
    run.signal(libc::SIGTERM);
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    let keys: Vec<String> = metadata_items(&gcp).into_iter().map(|(k, _)| k).collect();
    assert_eq!(keys, ["x0ne-config", "block-project-ssh-keys"]);
    assert!(!d.err.contains("NRAS:"), "{}", d.err);
}

/// Google quotes a value it refuses in its error; that value is the key.
#[test]
fn a_google_error_that_quotes_the_nras_key_is_said_without_it() {
    let gcp = FakeGcp::start(Cloud {
        insert_error: Some((
            "INVALID_FIELD_VALUE".into(),
            "Invalid value '{metadata:x0ne-nras-key}' for field 'metadata'".into(),
        )),
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Ok));
    let file = nras_key_file(&env, "nras.key", NRAS_KEY, 0o600);
    let mut cmd = env.command(&gcp.base, &up_args(&["--nras-key-file", &file]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 92, "{}", d.err);
    assert!(
        d.err.contains("Invalid value '(key not printed)'"),
        "{}",
        d.err
    );
    assert!(
        !d.out.contains("CANARY") && !d.err.contains("CANARY"),
        "{}",
        d.err
    );
    assert_eq!(
        env.files_containing(NRAS_KEY.as_bytes()),
        vec![std::path::PathBuf::from(&file)]
    );
    assert_cleaned(&gcp, &vm_name(&gcp));
}

/// A key file anyone else can read, a link, a missing file, a key typed
/// where its path goes: refused before any call, naming neither the path
/// nor what it holds. `up` and `run` only.
#[test]
fn a_bad_nras_key_file_is_refused_before_anything_without_being_quoted() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let good = nras_key_file(&env, "good-nras.key", NRAS_KEY, 0o600);
    let shared = nras_key_file(&env, "shared-nras.key", NRAS_KEY, 0o644);
    let two = nras_key_file(
        &env,
        "two-nras.key",
        &format!("{NRAS_KEY} {NRAS_KEY}"),
        0o600,
    );
    let link = env.path("link-nras.key");
    std::os::unix::fs::symlink(&good, &link).unwrap();
    let link = link.to_str().unwrap().to_string();
    let absent = env.path("absent-nras.key").to_str().unwrap().to_string();
    for (args, says) in [
        (up_args(&["--nras-key-file", &shared]), "0600"),
        (up_args(&["--nras-key-file", &two]), "one token"),
        (up_args(&["--nras-key-file", &link]), "not a regular file"),
        (up_args(&["--nras-key-file", &absent]), "cannot be opened"),
        (
            up_args(&["--nras-key-file", NRAS_KEY]),
            "the path of the key file, not the key",
        ),
        (
            vec!["quote", "--auth", "stdin", "--nras-key-file", &good],
            "up and run only",
        ),
    ] {
        let mut cmd = env.command(&gcp.base, &args);
        let d = spawn(&env, &mut cmd, Some(TOKEN), "refused", None).finish();
        assert_eq!(d.code, 2, "{args:?}: {}", d.err);
        assert!(
            d.err.contains("--nras-key-file") && d.err.contains(says),
            "{args:?}: {}",
            d.err
        );
        assert!(
            !d.err.contains("CANARY") && !d.err.contains("nras.key"),
            "{args:?}: {}",
            d.err
        );
    }
    assert!(gcp.requests().is_empty(), "{:?}", gcp.requests().len());
}

#[test]
fn the_test_reference_dir_without_the_loopback_override_is_refused() {
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command("unused", &up_args(&[]));
    cmd.env_remove("X0NE_ENGINE_GCP_ENDPOINT");
    cmd.env_remove("X0NE_ENGINE_CLOUD_TEST_POLL_MS");
    let d = spawn(&env, &mut cmd, Some(TOKEN), "knobs", None).finish();
    assert_eq!(d.code, 2, "{}", d.err);
    assert!(
        d.err
            .contains("X0NE_ENGINE_CLOUD_TEST_REFERENCE_DIR is a test setting"),
        "{}",
        d.err
    );
}

/// `up`, no chat: SIGQUIT (Ctrl-\ in its terminal) while serving deletes the VM
/// like SIGINT, SIGTERM and SIGHUP do.
#[test]
fn sigquit_while_serving_deletes_the_vm() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    run.wait_stderr("[5/5] OpenAI API");
    let vm = vm_name(&gcp);
    run.signal(libc::SIGQUIT);
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert!(d.err.contains("Session stopped by Ctrl-\\."), "{}", d.err);
    assert_cleaned(&gcp, &vm);
}

/// Each signal below ended `up` at once while it served, the VM left running
/// until `--max-run` (exits 138 to 192, rev10, 2026-10-17).
#[test]
fn every_catchable_ending_signal_while_serving_deletes_the_vm() {
    let mut failed = Vec::new();
    for (sig, name) in [
        (libc::SIGUSR1, "SIGUSR1"),
        (libc::SIGUSR2, "SIGUSR2"),
        (libc::SIGALRM, "SIGALRM"),
        (libc::SIGVTALRM, "SIGVTALRM"),
        (libc::SIGPROF, "SIGPROF"),
        (libc::SIGIO, "SIGIO"),
        (libc::SIGPWR, "SIGPWR"),
        (libc::SIGSTKFLT, "SIGSTKFLT"),
        (libc::SIGXCPU, "SIGXCPU"),
        (libc::SIGXFSZ, "SIGXFSZ"),
        (libc::SIGRTMIN(), "SIGRTMIN"),
        (libc::SIGRTMAX(), "SIGRTMAX"),
    ] {
        let gcp = FakeGcp::start(Cloud::default());
        let env = Env::new(Some(Relay::Ok));
        let mut cmd = env.command(&gcp.base, &up_args(&[]));
        let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
        run.wait_stderr("[5/5] OpenAI API");
        let vm = vm_name(&gcp);
        run.signal(sig);
        let d = run.finish();
        let deletes = gcp.deletes();
        let deleted = deletes
            .iter()
            .any(|p| p.ends_with(&format!("/zones/us-central1-a/instances/{vm}")))
            && deletes
                .iter()
                .any(|p| p.ends_with(&format!("/global/firewalls/{vm}")));
        if d.code != 0 || !deleted || !d.err.contains("Session stopped by ") {
            failed.push((name, d.code, deleted));
        }
    }
    assert!(
        failed.is_empty(),
        "(signal, exit, VM and rule deleted): {failed:?}"
    );
}
