//! `x0ne-engine-proxy` as a process: usage errors exit 2 before anything,
//! a reference not signed by the pinned archive key (the TEST key in a
//! debug test build, `test-archive-key`) exits 99 speaking the protocol,
//! one absent from the pinned transparency log (the TEST log) exits 100,
//! an unusable reference exits 89 speaking the protocol, a non-loopback
//! `--listen` is refused before a config is announced, and a closed stdin
//! after the config stops the relay with exit 2 (I4: unexpected end of the
//! control stream is a failure). Colors go to stderr only, when forced or
//! on a terminal. A release test build pins `RELEASE`, not the TEST key:
//! there, only the usage tests and the 99 of an unpinned build run.

use std::process::{Command, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_x0ne-engine-proxy");

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A reference of the documented shape, written where the test says, and
/// its `.sig` by the TEST archive key this test build pins.
/// A v2 reference the relay admits, values fake, RTMR0 pinned to the
/// endorsed firmware (`gce-endorsed`), signed by the TEST key and logged.
fn reference_file(dir: &std::path::Path) -> std::path::PathBuf {
    reference_file_with(
        dir,
        serde_json::json!({
            "policy": "gce-endorsed",
            "events_sha384": {"a3-highgpu-1g": ["c0".repeat(48), "firmware:cfv", "c2".repeat(48)]},
        }),
    )
}

/// [`reference_file`] with `rtmr0` as given.
fn reference_file_with(dir: &std::path::Path, rtmr0: serde_json::Value) -> std::path::PathBuf {
    let root_spki = x0ne_infer::remote_verify::Anchors::production()
        .gce_root_spki_sha256_hex()
        .unwrap();
    let body = serde_json::json!({
        "schema": x0ne_infer::reference::SCHEMA, "provider": "gcp", "tee": "intel-tdx",
        "image": {"name": "x0ne-cvm", "version": "0.1.0", "gcp_image": "x0ne-cvm-0-1-0",
                  "disk_tar_sha256": "0".repeat(64), "source": {"x0ne_os": "test"}},
        "mrtd": {"policy": "gce-launch-endorsement", "root_spki_sha256": root_spki, "ram_gib": {"a3-highgpu-1g": 0}},
        "rtmr0": rtmr0,
        "rtmr1": {"policy": "eventlog", "gpt_event_sha384": "a1".repeat(48), "boot_applications_sha384": ["aa".repeat(48)]},
        "rtmr2": {"policy": "eventlog", "events_sha384": {".linux": "b1".repeat(48)}},
        "rtmr3": {"policy": "zero"},
        "td_attributes": {"debug": false}
    });
    let path = dir.join("x0ne-cvm-0.1.0.json");
    let bytes = serde_json::to_vec_pretty(&body).unwrap();
    let sig = x0ne_infer::archive_key::test_key::sign(&bytes, now() as u32 - 60);
    std::fs::write(&path, &bytes).unwrap();
    std::fs::write(x0ne_infer::reference::signature_path(&path), sig).unwrap();
    // In the TEST transparency log a debug relay pins, next to the file.
    x0ne_infer::transparency::test_log::log_reference(&path, now() - 60).unwrap();
    path
}

fn run(args: &[&str], stdin: Stdio, dir: &std::path::Path) -> (i32, String, String) {
    run_with(args, stdin, dir, &[])
}

/// As [`run`], with only `vars` and the two directories in its environment.
fn run_with(
    args: &[&str],
    stdin: Stdio,
    dir: &std::path::Path,
    vars: &[(&str, &str)],
) -> (i32, String, String) {
    let mut cmd = Command::new(BIN);
    if !vars.is_empty() {
        cmd.env_clear().envs(vars.iter().copied());
    }
    let child = cmd
        .args(args)
        .env("XDG_RUNTIME_DIR", dir.join("run"))
        .env("XDG_STATE_HOME", dir.join("state"))
        .stdin(stdin)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    // Both pipes at once: on 2026-10-28 a usage error (4.7 KB on stderr) hung
    // here, stdout read first, the pipe cut to one 4 KiB page under load.
    let o = child.wait_with_output().unwrap();
    let text = |b: Vec<u8>| String::from_utf8(b).unwrap();
    (
        o.status.code().unwrap_or(-1),
        text(o.stdout),
        text(o.stderr),
    )
}

fn valid_args(reference: &str, listen: &str) -> Vec<String> {
    [
        "--protocol",
        "1",
        "--provider",
        "gcp",
        "--machine-type",
        "a3-highgpu-1g",
        "--model",
        "qwen3:0.6b",
        "--expires-at-unix",
        &(now() + 3600).to_string(),
        "--listen",
        listen,
        "--reference",
        reference,
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

#[test]
fn help_and_version_answer_on_any_machine() {
    let dir = tempfile::tempdir().unwrap();
    let (code, out, _) = run(&["--help"], Stdio::null(), dir.path());
    assert_eq!(code, 0);
    assert!(
        out.contains("x0ne-engine-proxy") && out.contains("--protocol"),
        "{out}"
    );
    let (code, out, _) = run(&["--version"], Stdio::null(), dir.path());
    assert_eq!(code, 0);
    assert!(out.starts_with("x0ne-engine-proxy "), "{out}");
}

#[test]
fn usage_errors_exit_2_before_anything_runs() {
    let dir = tempfile::tempdir().unwrap();
    let (code, out, err) = run(&[], Stdio::null(), dir.path());
    assert_eq!(code, 2, "{err}");
    assert!(err.contains("--protocol is required"), "{err}");
    assert!(out.is_empty());
    let (code, _, err) = run(&["--protocol", "2"], Stdio::null(), dir.path());
    assert_eq!(code, 2);
    assert!(err.contains("speaks 1"), "{err}");
    let (code, _, err) = run(
        &["--protocol", "1", "--insecure"],
        Stdio::null(),
        dir.path(),
    );
    assert_eq!(code, 2);
    assert!(err.contains("unknown argument"), "{err}");
    assert!(!dir.path().join("run").exists() && !dir.path().join("state").exists());
}

/// `--attest cpu` is refused by name before anything runs: the TD serves only
/// on its attested GPU, so the relay never asks for a CPU-only proof.
#[test]
fn attest_cpu_is_refused_by_name_before_anything_runs() {
    let dir = tempfile::tempdir().unwrap();
    for (value, says) in [
        ("cpu", "--attest cpu: refused"),
        ("gpu", "unknown --attest value"),
    ] {
        let mut args = valid_args("/nonexistent/reference.json", "127.0.0.1:0");
        args.extend(["--attest".to_string(), value.to_string()]);
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let (code, out, err) = run(&args, Stdio::null(), dir.path());
        assert_eq!(code, 2, "{value}: {err}");
        assert!(err.contains(says), "{value}: {err}");
        assert!(out.is_empty(), "{out}");
    }
    assert!(!dir.path().join("run").exists() && !dir.path().join("state").exists());
}

/// SECURITY: the TCB policy flags tighten the build's floor and never
/// loosen it. A looser or unreadable value is a usage error (2) before
/// anything runs; `--help` names both flags.
#[test]
fn the_tcb_policy_flags_tighten_and_a_looser_value_is_a_usage_error() {
    let dir = tempfile::tempdir().unwrap();
    for (args, says) in [
        (
            ["--min-tcb-evaluation", "21"],
            "minimum TCB evaluation 21 is below this build's 22",
        ),
        (
            ["--max-collateral-age", "86401"],
            "maximum collateral age 86401 s is above this build's 86400 s",
        ),
        (["--min-tcb-evaluation", "latest"], "--min-tcb-evaluation"),
        (["--max-collateral-age", "-1"], "--max-collateral-age"),
        (["--tcb-policy", "lenient"], "--tcb-policy"),
    ] {
        let mut argv = vec!["--protocol", "1"];
        argv.extend(args);
        let (code, out, err) = run(&argv, Stdio::null(), dir.path());
        assert_eq!(code, 2, "{args:?}: {err}");
        assert!(err.contains(says), "{args:?}: {err}");
        assert!(out.is_empty(), "{args:?}: {out}");
    }
    // A refusal floor beside the mode that only names: refused, not ignored.
    let (code, out, err) = run(
        &[
            "--protocol",
            "1",
            "--tcb-policy",
            "transparent",
            "--min-tcb-evaluation",
            "23",
        ],
        Stdio::null(),
        dir.path(),
    );
    assert_eq!(code, 2, "{err}");
    assert!(
        err.contains("--min-tcb-evaluation refuses only under --tcb-policy strict"),
        "{err}"
    );
    assert!(out.is_empty(), "{out}");
    assert!(!dir.path().join("run").exists() && !dir.path().join("state").exists());
    let (_, out, _) = run(&["--help"], Stdio::null(), dir.path());
    assert!(
        out.contains("--min-tcb-evaluation <N>")
            && out.contains("--max-collateral-age <secs>")
            && out.contains("--tcb-policy <mode>"),
        "{out}"
    );
}

/// SECURITY: the policy the flags build is the one the relay applies: its
/// config names it (`up` refuses a looser one), the build's floor without
/// flags. A relay that parsed the flags and kept the production policy
/// fails here.
#[cfg(debug_assertions)]
#[test]
fn the_relay_config_names_the_tcb_policy_its_flags_set() {
    let dir = tempfile::tempdir().unwrap();
    let reference = reference_file(dir.path());
    for (flags, want) in [
        (&[][..], ("transparent", 22, 86_400)),
        (&["--tcb-policy", "strict"][..], ("strict", 22, 86_400)),
        (
            &["--tcb-policy", "transparent"][..],
            ("transparent", 22, 86_400),
        ),
        // A floor alone keeps its meaning of 2026-10-27: refuse below it.
        (&["--min-tcb-evaluation", "23"][..], ("strict", 23, 86_400)),
        (
            &["--max-collateral-age", "3600"][..],
            ("transparent", 22, 3600),
        ),
        (
            &[
                "--tcb-policy",
                "strict",
                "--min-tcb-evaluation",
                "24",
                "--max-collateral-age",
                "60",
            ][..],
            ("strict", 24, 60),
        ),
    ] {
        let mut args = valid_args(reference.to_str().unwrap(), "127.0.0.1:0");
        args.extend(flags.iter().map(|s| s.to_string()));
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let (code, out, err) = run(&args, Stdio::null(), dir.path());
        assert_eq!(code, 2, "{flags:?}: {out}{err}");
        let config: serde_json::Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
        assert_eq!(config["event"], "config", "{config}");
        assert_eq!(
            config["tcb_policy"],
            serde_json::json!({"mode": want.0, "min_tcb_evaluation": want.1, "max_collateral_age_secs": want.2}),
            "{flags:?}: {config}"
        );
    }
}

/// Stricter values pass the usage check: the relay goes on to the image
/// reference (here unusable, 89), the policy taken.
#[cfg(debug_assertions)]
#[test]
fn stricter_tcb_policy_flags_are_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let mut args = valid_args("/nonexistent/x0ne-cvm.json", "127.0.0.1:0");
    args.extend(["--min-tcb-evaluation", "23", "--max-collateral-age", "3600"].map(String::from));
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let (code, out, err) = run(&args, Stdio::null(), dir.path());
    assert_eq!(code, 89, "{out}{err}");
}

#[cfg(debug_assertions)]
#[test]
fn an_unusable_reference_is_refused_with_89_in_the_protocol() {
    let dir = tempfile::tempdir().unwrap();
    let args = valid_args("/nonexistent/x0ne-cvm.json", "127.0.0.1:0");
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let (code, out, _) = run(&args, Stdio::null(), dir.path());
    assert_eq!(code, 89, "{out}");
    let event: serde_json::Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    assert_eq!(event["v"], 1);
    assert_eq!(event["event"], "refused");
    assert_eq!(event["code"], 89);
    assert!(
        event["reason"]
            .as_str()
            .unwrap()
            .contains("image reference"),
        "{event}"
    );
}

/// A v1 reference, written before RTMR1 pinned the disk's GPT, is refused
/// with 89 and named, rather than read with the old RTMR1 policy.
#[cfg(debug_assertions)]
#[test]
fn a_v1_reference_is_refused_with_89_and_named() {
    let dir = tempfile::tempdir().unwrap();
    let path = reference_file(dir.path());
    let mut body: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    body["schema"] = "x0ne-engine/reference/v1".into();
    body["rtmr1"]
        .as_object_mut()
        .unwrap()
        .remove("gpt_event_sha384");
    // Signed and logged again: the TEST key and log pass, so 89 is the
    // schema's refusal.
    let bytes = serde_json::to_vec_pretty(&body).unwrap();
    let sig = x0ne_infer::archive_key::test_key::sign(&bytes, now() as u32 - 60);
    std::fs::write(&path, &bytes).unwrap();
    std::fs::write(x0ne_infer::reference::signature_path(&path), sig).unwrap();
    x0ne_infer::transparency::test_log::log_reference(&path, now() - 60).unwrap();
    let args = valid_args(path.to_str().unwrap(), "127.0.0.1:0");
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let (code, out, _) = run(&args, Stdio::null(), dir.path());
    assert_eq!(code, 89, "{out}");
    let event: serde_json::Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    let reason = event["reason"].as_str().unwrap();
    assert!(
        reason.contains("x0ne-engine/reference/v1") && reason.contains("GPT"),
        "{event}"
    );
}

/// A reference that reports RTMR0 instead of pinning it (`rtmr0.policy`
/// `report`, every x0ne-cvm 0.2.x reference published before 2026-10-31)
/// is signed and logged, then refused with 104 and named, before the relay
/// listens: nothing is created.
#[cfg(debug_assertions)]
#[test]
fn a_reference_reporting_rtmr0_is_refused_with_104_and_named() {
    let dir = tempfile::tempdir().unwrap();
    let path = reference_file_with(
        dir.path(),
        serde_json::json!({"policy": "report", "known": {"a3-highgpu-1g": []}}),
    );
    let args = valid_args(path.to_str().unwrap(), "127.0.0.1:0");
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let (code, out, _) = run(&args, Stdio::null(), dir.path());
    assert_eq!(code, 104, "{out}");
    let event: serde_json::Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    assert_eq!(event["event"], "refused", "{event}");
    let reason = event["reason"].as_str().unwrap();
    assert!(
        reason.contains("RTMR0")
            && reason.contains("\"report\"")
            && reason.contains("gce-endorsed"),
        "{event}"
    );
}

#[cfg(debug_assertions)]
#[test]
fn a_listen_address_off_the_loopback_is_refused_before_the_config() {
    let dir = tempfile::tempdir().unwrap();
    let reference = reference_file(dir.path());
    let args = valid_args(reference.to_str().unwrap(), "0.0.0.0:0");
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let (code, out, _) = run(&args, Stdio::null(), dir.path());
    assert_eq!(code, 2, "{out}");
    let event: serde_json::Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    assert_eq!(event["event"], "refused", "{event}");
    assert!(
        event["reason"].as_str().unwrap().contains("loopback"),
        "{event}"
    );
    assert!(!out.contains("\"config\""), "no config is announced: {out}");
}

#[cfg(debug_assertions)]
#[test]
fn a_closed_control_stream_after_the_config_stops_the_relay_with_2() {
    let dir = tempfile::tempdir().unwrap();
    let reference = reference_file(dir.path());
    let args = valid_args(reference.to_str().unwrap(), "127.0.0.1:0");
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let (code, out, _) = run(&args, Stdio::null(), dir.path());
    assert_eq!(code, 2, "{out}");
    let mut lines = out.lines();
    let config: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    assert_eq!(config["event"], "config", "{config}");
    assert_eq!(config["config_sha256"].as_str().unwrap().len(), 64);
    // The image of the reference it verified: `up` refuses another.
    assert_eq!(
        config["image"],
        serde_json::json!({"gcp_image": "x0ne-cvm-0-1-0", "disk_tar_sha256": "0".repeat(64)}),
        "{config}"
    );
    let bytes = {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .decode(config["config_b64"].as_str().unwrap())
            .unwrap()
    };
    let loaded = x0ne_infer::cloud_config::parse(&bytes).expect("a valid config");
    assert_eq!(loaded.config.model, "qwen3:0.6b");
    assert_eq!(loaded.config.attest, "cpu+gpu");
    assert_eq!(loaded.config.listen_port, 443);
    assert!(lines.next().is_none(), "nothing after the config: {out}");
    assert!(
        !dir.path().join("run/x0ne-engine/local.key").exists(),
        "no key file before a verdict"
    );
}

/// SECURITY: a reference signed by the TEST archive key but absent
/// from the TEST transparency log: exit 100 before a config exists.
#[cfg(debug_assertions)]
#[test]
fn a_reference_absent_from_the_log_is_refused_with_100() {
    let dir = tempfile::tempdir().unwrap();
    let reference = reference_file(dir.path());
    let _ = std::fs::remove_file(x0ne_infer::transparency::proof_path(&reference));
    let args = valid_args(reference.to_str().unwrap(), "127.0.0.1:0");
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let (code, out, _) = run(&args, Stdio::null(), dir.path());
    assert_eq!(code, 100, "{out}");
    let event: serde_json::Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    assert_eq!(event["event"], "refused", "{event}");
    assert_eq!(event["code"], 100, "{event}");
    assert!(!out.contains("\"config\""), "no config is announced: {out}");
}

/// Alone, the relay prints its progress and its refusal on stderr; with
/// `--quiet`, as `x0ne-engine cloud` starts it, stderr stays empty and
/// stdout carries the same events. A boot deadline in the past: 88 before
/// any connection.
#[cfg(debug_assertions)]
#[test]
fn quiet_leaves_the_progress_and_the_refusal_to_the_protocol() {
    let dir = tempfile::tempdir().unwrap();
    let reference = reference_file(dir.path());
    let connect = dir.path().join("connect.jsonl");
    std::fs::write(
        &connect,
        "{\"v\":1,\"cmd\":\"connect\",\"endpoint\":\"127.0.0.1:9\",\"boot_deadline_unix\":1}\n",
    )
    .unwrap();
    for quiet in [false, true] {
        let mut args = valid_args(reference.to_str().unwrap(), "127.0.0.1:0");
        if quiet {
            args.push("--quiet".into());
        }
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let stdin = Stdio::from(std::fs::File::open(&connect).unwrap());
        let (code, out, err) = run(&args, stdin, dir.path());
        assert_eq!(code, 88, "quiet {quiet}: {err}");
        let events: Vec<serde_json::Value> = out
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let names: Vec<&str> = events
            .iter()
            .map(|e| e["event"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["config", "progress", "refused"], "{out}");
        assert_eq!(events[1]["stage"], "connecting");
        assert_eq!(events[2]["code"], 88);
        if quiet {
            assert_eq!(err, "", "quiet");
        } else {
            assert_eq!(
                err.lines().collect::<Vec<_>>(),
                [
                    format!(
                        "x0ne-engine-proxy: connecting: {}",
                        events[1]["detail"].as_str().unwrap()
                    ),
                    format!(
                        "x0ne-engine-proxy: REFUSED (88): {}",
                        events[2]["reason"].as_str().unwrap()
                    ),
                ],
                "{err}"
            );
        }
    }
}

/// `--quiet` mutes only what is also an event: a bad control line and a
/// second connect are not events, so they stay on stderr.
#[cfg(debug_assertions)]
#[test]
fn quiet_keeps_the_control_stream_messages_on_stderr() {
    let dir = tempfile::tempdir().unwrap();
    let reference = reference_file(dir.path());
    let connect = format!(
        "{{\"v\":1,\"cmd\":\"connect\",\"endpoint\":\"127.0.0.1:9\",\"boot_deadline_unix\":{}}}\n",
        now() + 60
    );
    let input = dir.path().join("control.jsonl");
    std::fs::write(
        &input,
        format!("nope\n{connect}{connect}{{\"v\":1,\"cmd\":\"stop\"}}\n"),
    )
    .unwrap();
    let mut args = valid_args(reference.to_str().unwrap(), "127.0.0.1:0");
    args.push("--quiet".into());
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let stdin = Stdio::from(std::fs::File::open(&input).unwrap());
    let (code, out, err) = run(&args, stdin, dir.path());
    assert_eq!(code, 0, "{err}");
    let names: Vec<String> = out
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["event"].to_string())
        .collect();
    assert_eq!(
        names,
        ["\"config\"", "\"progress\"", "\"stopped\""],
        "{out}"
    );
    let lines: Vec<&str> = err.lines().collect();
    assert_eq!(lines.len(), 2, "{err}");
    assert!(
        lines[0].starts_with("x0ne-engine-proxy: control: not JSON: ")
            && lines[0].ends_with(", ignored"),
        "{err}"
    );
    assert_eq!(
        lines[1], "x0ne-engine-proxy: control: a second connect, ignored",
        "{err}"
    );
}

/// Colors on stderr only, never in the protocol on stdout; `NO_COLOR` wins
/// over `CLICOLOR_FORCE`. Pipes here, so color only when forced.
#[cfg(debug_assertions)]
#[test]
fn refusals_are_red_on_stderr_when_forced_and_the_protocol_never_is() {
    let dir = tempfile::tempdir().unwrap();
    let args = valid_args("/nonexistent/x0ne-cvm.json", "127.0.0.1:0");
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let forced = [("CLICOLOR_FORCE", "1")];
    let (code, out, err) = run_with(&args, Stdio::null(), dir.path(), &forced);
    assert_eq!(code, 89, "{err}");
    assert!(
        err.starts_with("\x1b[31mx0ne-engine-proxy: image reference: ")
            && err.ends_with("\x1b[0m\n"),
        "{err:?}"
    );
    assert!(!out.contains('\x1b'), "{out:?}");
    let event: serde_json::Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    assert_eq!(event["event"], "refused");
    assert!(!event["reason"].as_str().unwrap().contains('\x1b'));

    let key = dir.path().join("missing.key");
    let chat = [
        "chat",
        "--listen",
        "127.0.0.1:9",
        "--key-file",
        key.to_str().unwrap(),
    ];
    let (code, _, err) = run_with(&chat, Stdio::null(), dir.path(), &forced);
    assert_eq!(code, 1, "{err}");
    assert!(
        err.starts_with("\x1b[31mx0ne-engine-proxy chat: "),
        "{err:?}"
    );

    for vars in [
        &[("NO_COLOR", "1"), ("CLICOLOR_FORCE", "1")][..],
        &[("TERM", "xterm-256color")][..],
    ] {
        let (code, out, err) = run_with(&args, Stdio::null(), dir.path(), vars);
        assert_eq!(code, 89, "{vars:?}");
        assert!(
            !err.contains('\x1b') && !out.contains('\x1b'),
            "{vars:?}: {err:?}"
        );
        let (code, _, err) = run_with(&chat, Stdio::null(), dir.path(), vars);
        assert_eq!(code, 1, "{vars:?}");
        assert!(
            err.starts_with("x0ne-engine-proxy chat: "),
            "{vars:?}: {err:?}"
        );
    }
}

#[cfg(debug_assertions)]
fn refused_99(reference: &std::path::Path, dir: &std::path::Path, why: &str) {
    let args = valid_args(reference.to_str().unwrap(), "127.0.0.1:0");
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let (code, out, _) = run(&args, Stdio::null(), dir);
    assert_eq!(code, 99, "{why}: {out}");
    let mut lines = out.lines();
    let event: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    assert_eq!(event["event"], "refused", "{why}: {event}");
    assert_eq!(event["code"], 99, "{why}: {event}");
    let reason = event["reason"].as_str().unwrap();
    assert!(
        reason.contains("not signed by the x0ne archive key"),
        "{why}: {reason}"
    );
    assert!(
        lines.next().is_none(),
        "{why}: no config is announced: {out}"
    );
    assert!(!dir.join("run").exists(), "{why}: nothing written");
}

/// SECURITY: unsigned, signed by another key, or changed after signing:
/// exit 99 before a config exists, so `cloud up` creates nothing.
#[cfg(debug_assertions)]
#[test]
fn a_reference_not_signed_by_the_pinned_key_is_refused_with_99() {
    let dir = tempfile::tempdir().unwrap();
    let reference = reference_file(dir.path());
    let sig = x0ne_infer::reference::signature_path(&reference);
    std::fs::remove_file(&sig).unwrap();
    refused_99(&reference, dir.path(), "no .sig");

    let reference = reference_file(dir.path());
    let mut bytes = std::fs::read(&reference).unwrap();
    let at = bytes.iter().position(|&b| b == b'0').unwrap();
    bytes[at] = b'1';
    std::fs::write(&reference, bytes).unwrap();
    refused_99(&reference, dir.path(), "one byte changed after signing");

    // Signed by GnuPG with another TEST key (tests/archive_key/).
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/archive_key");
    let other = dir.path().join("other.json");
    std::fs::copy(fixtures.join("reference.json"), &other).unwrap();
    std::fs::copy(
        fixtures.join("reference.json.sig"),
        dir.path().join("other.json.sig"),
    )
    .unwrap();
    refused_99(&other, dir.path(), "another key");
}

/// SECURITY: `cargo test --release` and the release `cloud_probe` carry
/// `test-archive-key` too (self dev-dependency). Without debug assertions
/// the relay pins `RELEASE`: a reference signed by the TEST key gets 99.
#[cfg(not(debug_assertions))]
#[test]
fn a_release_build_never_pins_the_test_key() {
    let dir = tempfile::tempdir().unwrap();
    let reference = reference_file(dir.path());
    let args = valid_args(reference.to_str().unwrap(), "127.0.0.1:0");
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let (code, out, _) = run(&args, Stdio::null(), dir.path());
    assert_eq!(code, 99, "{out}");
    let mut lines = out.lines();
    let event: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    assert_eq!(event["event"], "refused", "{event}");
    let reason = event["reason"].as_str().unwrap();
    let want = match x0ne_infer::archive_key::RELEASE {
        None => "built before the signing ceremony",
        Some(_) => "not signed by the x0ne archive key",
    };
    assert!(reason.contains(want), "{reason}");
    assert!(lines.next().is_none(), "no config is announced: {out}");
    assert!(!dir.path().join("run").exists(), "nothing written");
}

/// `pins`: the anchors this build pins, asked by the lifecycle before the
/// token. A debug test build pins the TEST key and the TEST log, which anyone
/// can sign with, and says so (`"test": true`): the lifecycle refuses such a
/// relay. `pins` writes nothing and takes no argument.
#[cfg(debug_assertions)]
#[test]
fn pins_names_the_test_anchors_of_a_debug_build() {
    let dir = tempfile::tempdir().unwrap();
    let (code, out, err) = run(&["pins"], Stdio::null(), dir.path());
    assert_eq!(code, 0, "{out}{err}");
    let fpr = x0ne_infer::openpgp::Cert::parse(&x0ne_infer::archive_key::test_key::keyring())
        .unwrap()
        .primary_fingerprint();
    let origin = x0ne_infer::transparency::test_log::policy().origin;
    let event: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
    assert_eq!(
        event,
        serde_json::json!({"v": 1, "event": "pins", "archive_key": fpr, "transparency_log": origin, "test": true})
    );
    assert_eq!(out.lines().count(), 1, "{out}");
    assert!(err.is_empty(), "{err}");
    assert!(!dir.path().join("run").exists() && !dir.path().join("state").exists());
    let (code, _, err) = run(&["pins", "--quiet"], Stdio::null(), dir.path());
    assert_eq!(code, 2, "{err}");
}

/// In a release build `pins` refuses as a session would: 99 without the
/// archive key, 100 without the transparency log.
#[cfg(not(debug_assertions))]
#[test]
fn pins_refuses_as_a_session_would_in_a_release_build() {
    let dir = tempfile::tempdir().unwrap();
    let (code, out, _) = run(&["pins"], Stdio::null(), dir.path());
    let event: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
    let want = match (
        x0ne_infer::archive_key::RELEASE.is_some(),
        x0ne_infer::transparency::RELEASE.is_some(),
    ) {
        (false, _) => Some((99, "pins no x0ne archive key")),
        (true, false) => Some((100, "pins no transparency log")),
        (true, true) => None,
    };
    match want {
        Some((c, why)) => {
            assert_eq!(code, c, "{out}");
            assert_eq!(event["event"], "refused", "{event}");
            assert_eq!(event["code"], c, "{event}");
            assert!(event["reason"].as_str().unwrap().contains(why), "{event}");
        }
        None => {
            assert_eq!(code, 0, "{out}");
            assert_eq!(event["event"], "pins", "{event}");
            assert_eq!(event["test"], false, "{event}");
        }
    }
}
