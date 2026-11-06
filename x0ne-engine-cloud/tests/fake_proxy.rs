//! The lifecycle side of I4 against fake relays: where the relay is looked up,
//! what it is given, and every way it can fail. The VM is deleted each time.

mod common;

use std::fs;

use common::*;

#[test]
fn the_relay_is_never_taken_from_path() {
    let gcp = FakeGcp::start(Cloud::default());
    // No relay next to the binary; a decoy one first in $PATH.
    let env = Env::new(None);
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(
        d.err.contains("x0ne-engine-proxy not found at ")
            && d.err.contains("/usr/libexec/x0ne-engine/x0ne-engine-proxy")
            && d.err.contains("never looked up in $PATH"),
        "{}",
        d.err
    );
    assert!(env.decoys_ran().is_empty(), "{:?}", env.decoys_ran());
    assert!(
        gcp.requests().is_empty(),
        "the relay is checked before the token is used"
    );
}

#[test]
fn a_relay_that_others_can_write_is_refused() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(None);
    write_exe(
        &env.bin.with_file_name("x0ne-engine-proxy"),
        &relay_script(&env.path("relay"), Relay::Ok),
        0o775,
    );
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 91, "{}", d.err);
    assert!(
        d.err.contains("must not be writable by group or others"),
        "{}",
        d.err
    );
    no_creation(&gcp);
}

#[test]
fn the_relay_gets_the_i4_argv_and_an_environment_without_secrets() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let reference = env.path("x0ne-cvm-0.1.0.json");
    fs::write(
        &reference,
        r#"{"schema":"x0ne-engine/reference/v2","provider":"gcp","tee":"intel-tdx",
            "image":{"name":"x0ne-cvm","version":"0.1.0","gcp_image":"x0ne-cvm-0-1-0"}}"#,
    )
    .unwrap();
    let mut cmd = env.command(
        &gcp.base,
        &up_args(&[
            "--max-run",
            "45m",
            "--reference",
            reference.to_str().unwrap(),
        ]),
    );
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    run.wait_stderr("[5/5] OpenAI API");
    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    run.signal(libc::SIGINT);
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);

    let argv: Vec<String> = env.relay_log("argv").lines().map(String::from).collect();
    let exp: u64 = argv[9].parse().unwrap();
    assert!(exp.abs_diff(started + 45 * 60) < 120, "expiry {exp}");
    let want: Vec<String> = [
        "--protocol",
        "1",
        "--provider",
        "gcp",
        "--machine-type",
        "a3-highgpu-1g",
        "--model",
        "qwen3:0.6b",
        "--expires-at-unix",
        &exp.to_string(),
        "--listen",
        "127.0.0.1:8478",
        "--quiet",
        "--reference",
        reference.to_str().unwrap(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(argv, want);
    // I1 carries the same expiry, and its bytes went to the VM unchanged.
    let config = env.relay_log("config");
    assert!(
        config.contains(&format!("\"expires_at_unix\":{exp}")),
        "{config}"
    );

    let relay_env = env.relay_log("env");
    for absent in [
        "X0NE_ENGINE_GCP_ENDPOINT",
        "X0NE_ENGINE_CLOUD_TEST_POLL_MS",
        "X0NE_ENGINE_CLOUD_TEST_REFERENCE_DIR",
        "GOOGLE_CLOUD_PROJECT",
        TOKEN,
    ] {
        assert!(
            !relay_env.contains(absent),
            "{absent} reached the relay:\n{relay_env}"
        );
    }
    assert!(
        relay_env.contains(&format!("XDG_RUNTIME_DIR={}", env.runtime.display())),
        "{relay_env}"
    );
    assert_cleaned(&gcp, &vm_name(&gcp));
}

/// SECURITY: `--tcb-policy`, `--min-tcb-evaluation` and
/// `--max-collateral-age` reach the relay's command line, and the relay's
/// config names them back as the policy it applies.
#[test]
fn up_passes_the_tcb_policy_options_to_the_relay() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(
        &gcp.base,
        &up_args(&[
            "--tcb-policy",
            "strict",
            "--min-tcb-evaluation",
            "23",
            "--max-collateral-age",
            "3600",
        ]),
    );
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    run.wait_stderr("[5/5] OpenAI API");
    run.signal(libc::SIGINT);
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    let argv: Vec<String> = env.relay_log("argv").lines().map(String::from).collect();
    assert_eq!(
        argv[argv.len() - 7..],
        [
            "--quiet",
            "--tcb-policy",
            "strict",
            "--min-tcb-evaluation",
            "23",
            "--max-collateral-age",
            "3600"
        ],
        "{argv:?}"
    );
    assert_cleaned(&gcp, &vm_name(&gcp));
}

/// SECURITY: a relay that takes the TCB policy options and applies the
/// build's floor instead is refused on its config, before the VM exists.
#[test]
fn a_relay_that_ignores_the_tcb_policy_options_is_refused_before_anything_is_created() {
    let strict = "the relay applies the transparent TCB policy, looser than the strict one";
    for (opts, says) in [
        (["--min-tcb-evaluation", "23"], strict),
        (["--tcb-policy", "strict"], strict),
        (
            ["--max-collateral-age", "3600"],
            "the relay applies TCB evaluation 22 or later",
        ),
    ] {
        let gcp = FakeGcp::start(Cloud::default());
        let env = Env::new(Some(Relay::IgnoresTcbOptions));
        let mut cmd = env.command(&gcp.base, &up_args(&opts));
        let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
        assert_eq!(d.code, 95, "{opts:?}: {}", d.err);
        assert!(
            d.err.contains(says)
                && d.err.contains("looser")
                && d.err.contains("nothing was created"),
            "{opts:?}: {}",
            d.err
        );
        no_creation(&gcp);
    }
}

#[test]
fn a_refusal_exits_with_the_relay_code_and_the_vm_is_deleted() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Refuse(65)));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 65, "{}", d.err);
    for line in [
        "      REFUSED (65): RTMR2 does not match the reference (fake)",
        "      No question and no session key were sent to the VM.",
        "      Evidence saved for review: /fake/evidence.json (public data only)",
    ] {
        assert!(d.err.contains(line), "missing {line:?}:\n{}", d.err);
    }
    assert!(!d.err.contains("[5/5]"));
    assert_cleaned(&gcp, &vm_name(&gcp));
}

/// The model owner's key policy refuses the TD after the verdict: the relay
/// exits 102 and so does `up` (2026-11-06, H100 run: `up` printed
/// "REFUSED (102)" and exited 95).
#[test]
fn a_key_policy_refusal_exits_102_and_the_vm_is_deleted() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Refuse(102)));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 102, "{}", d.err);
    assert!(d.err.contains("      REFUSED (102): "), "{}", d.err);
    assert_cleaned(&gcp, &vm_name(&gcp));
}

/// The same refusal before the config (an expired policy, unusable files):
/// 102, and nothing is created.
#[test]
fn a_key_policy_refusal_before_the_config_exits_102_and_creates_nothing() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::ExitBeforeConfig(102)));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 102, "{}", d.err);
    no_creation(&gcp);
}

/// Only the documented codes pass: a relay that exits 101 (a Rust panic)
/// after a `refused` event is a relay failure.
#[test]
fn a_refusal_with_an_undocumented_code_is_95() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Refuse(101)));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 95, "{}", d.err);
    assert!(d.err.contains("      REFUSED (101): "), "{}", d.err);
    assert_cleaned(&gcp, &vm_name(&gcp));
}

#[test]
fn a_relay_crash_during_verification_is_95_and_the_vm_is_deleted() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::CrashAfterConnect(1)));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 95, "{}", d.err);
    assert!(
        d.err
            .contains("the relay exited with code 1 during the verification"),
        "{}",
        d.err
    );
    assert_cleaned(&gcp, &vm_name(&gcp));
}

#[test]
fn a_relay_that_dies_with_a_refusal_code_keeps_that_code() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::CrashAfterConnect(75)));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 75, "{}", d.err);
    assert_cleaned(&gcp, &vm_name(&gcp));
}

#[test]
fn a_relay_crash_while_serving_is_95_and_the_vm_is_deleted() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::CrashWhileServing(3)));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 95, "{}", d.err);
    assert!(d.err.contains("[5/5] OpenAI API"), "{}", d.err);
    assert!(
        d.err.contains("the relay exited with code 3 while serving"),
        "{}",
        d.err
    );
    assert_cleaned(&gcp, &vm_name(&gcp));
}

#[test]
fn a_config_whose_hash_does_not_match_creates_nothing() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::BadConfig));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 95, "{}", d.err);
    assert!(
        d.err.contains(
            "relay config refused: config_sha256 does not match config_b64; nothing was created"
        ),
        "{}",
        d.err
    );
    no_creation(&gcp);
    assert!(gcp.deletes().is_empty());
}

#[test]
fn a_relay_that_exits_before_its_config_creates_nothing() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::ExitBeforeConfig(70)));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 70, "{}", d.err);
    assert!(
        d.err
            .contains("before sending its config; nothing was created"),
        "{}",
        d.err
    );
    no_creation(&gcp);
}

/// A relay built before the signing ceremony (99), or before a public log
/// (100), refuses every session: `up` asks its `pins` first and stops there,
/// before the token, the quote and "yes".
#[test]
fn a_relay_without_its_pins_is_refused_before_the_token() {
    for code in [99u8, 100] {
        let gcp = FakeGcp::start(Cloud::default());
        let env = Env::new(Some(Relay::NoPins(code)));
        let mut cmd = env.command(&gcp.base, &up_args(&[]));
        let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
        assert_eq!(d.code, 95, "{}", d.err);
        assert!(
            d.err.contains(&format!("the relay refused ({code}): ")),
            "{}",
            d.err
        );
        assert!(d.err.contains("no token was asked"), "{}", d.err);
        assert_eq!(env.relay_log("pins-argv"), "pins\n");
        assert!(
            gcp.requests().is_empty(),
            "{code}: the token was used: {:?}",
            gcp.requests()
        );
        assert!(env.relay_log("argv").is_empty(), "no session was started");
        no_creation(&gcp);
    }
}

/// A relay built for tests pins the TEST archive key and the TEST log, which
/// anyone can sign with (keys derived from public strings): `up` against
/// Google's endpoints refuses it from its `pins`, before the token, the quote
/// and "yes". No endpoint override and no token: a regression fails on the
/// missing token, offline. A failure means a reference signed by anyone can
/// again open a session after `cargo test`. The loopback fake lets it
/// through (`real_relay.rs` runs the real debug relay that way).
#[test]
fn a_relay_that_pins_the_test_anchors_is_refused_before_the_token() {
    let env = Env::new(Some(Relay::TestPins));
    let r = env.path("x0ne-cvm-0.1.0.json");
    std::fs::write(
        &r,
        r#"{"schema":"x0ne-engine/reference/v2","provider":"gcp","tee":"intel-tdx",
            "image":{"name":"x0ne-cvm","version":"0.1.0","gcp_image":"x0ne-cvm-0-1-0"}}"#,
    )
    .unwrap();
    let mut cmd = env.command("", &up_args(&["--reference", r.to_str().unwrap()]));
    for test_setting in [
        "X0NE_ENGINE_GCP_ENDPOINT",
        "X0NE_ENGINE_CLOUD_TEST_MYADDR_SERVER",
        "X0NE_ENGINE_CLOUD_TEST_POLL_MS",
        "X0NE_ENGINE_CLOUD_TEST_REFERENCE_DIR",
    ] {
        cmd.env_remove(test_setting);
    }
    let d = spawn(&env, &mut cmd, None, "up", None).finish();
    assert_eq!(d.code, 95, "{}", d.err);
    assert!(d.err.contains("TEST archive key"), "{}", d.err);
    assert!(d.err.contains("no token was asked"), "{}", d.err);
    assert_eq!(env.relay_log("pins-argv"), "pins\n");
    assert!(env.relay_log("argv").is_empty(), "no session was started");
}

/// A relay that pins both is asked once, with no token in its environment,
/// and the session goes on.
#[test]
fn the_pins_are_asked_once_and_the_session_goes_on() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    run.wait_stderr("[5/5] OpenAI API");
    run.signal(libc::SIGINT);
    let d = run.finish();
    assert_eq!(d.code, 0, "{}", d.err);
    assert_eq!(env.relay_log("pins-argv"), "pins\n");
    assert!(!env.relay_log("pins-env").contains(TOKEN));
}

#[test]
fn ctrl_c_during_the_verification_stops_the_relay_and_deletes() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Hang));
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    run.wait_stderr("waiting for the VM (fake)");
    run.signal(libc::SIGINT);
    let d = run.finish();
    assert_eq!(d.code, 130, "{}", d.err);
    assert!(env.relay_log("cmds").contains(r#""cmd":"stop""#));
    assert_cleaned(&gcp, &vm_name(&gcp));
}

#[test]
fn a_verdict_that_never_comes_ends_at_the_boot_timeout() {
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
    assert_cleaned(&gcp, &vm_name(&gcp));
}
