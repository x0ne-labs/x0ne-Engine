//! No binary copies an argument it does not know to its output. The four
//! forms of a key on a command line measured on 2026-10-12, against each parser.

use std::process::{Command, Stdio};

fn key() -> String {
    "5ec7".repeat(16)
}

fn forms() -> Vec<Vec<String>> {
    let k = key();
    vec![
        vec!["--key-hex".into(), k.clone()],
        vec![format!("--key-hex={k}")],
        vec![k.clone()],
        vec![format!("X0NE_INFER_KEY_HEX={k}")],
    ]
}

/// Each form, after `prefix`: refused or ignored with "unknown", never printed.
fn assert_not_echoed(bin: &str, prefix: &[&str], env: &[(&str, &str)]) {
    for form in forms() {
        let out = Command::new(bin)
            // cargo sets LD_LIBRARY_PATH for tests; a release x0ne-infer refuses it (73).
            .env_remove("LD_LIBRARY_PATH")
            .args(prefix)
            .args(&form)
            .envs(env.iter().copied())
            .stdin(Stdio::null())
            .output()
            .expect("the binary runs");
        let all = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(!out.status.success(), "{bin} {form:?}: {all}");
        assert!(all.contains("unknown"), "{bin} {form:?}: {all}");
        assert!(
            !all.contains(&key()[..8]),
            "{bin} {form:?} printed it: {all}"
        );
    }
}

#[test]
fn no_parser_prints_an_unknown_argument() {
    let dir = tempfile::tempdir().expect("tempdir");
    let absent = dir.path().join("absent.json");
    let zero_key = "00".repeat(32);
    // Exit 87 on the absent file, after the arguments are read.
    assert_not_echoed(
        env!("CARGO_BIN_EXE_x0ne-infer"),
        &[
            "--model-dir",
            "/nonexistent",
            "--listen",
            "127.0.0.1:0",
            "--measurements",
            absent.to_str().expect("utf-8"),
        ],
        &[("X0NE_INFER_KEY_HEX", zero_key.as_str())],
    );
    assert_not_echoed(
        env!("CARGO_BIN_EXE_x0ne-engine-proxy"),
        &["--protocol", "1"],
        &[],
    );
    // The relay's chat has its own parser: an argument without a value was quoted.
    let proxy = env!("CARGO_BIN_EXE_x0ne-engine-proxy");
    assert_not_echoed(proxy, &["chat"], &[]);
    assert_not_echoed(proxy, &["chat", "--listen", "127.0.0.1:9"], &[]);
    let home = dir.path().join("home");
    let home = home.to_str().expect("utf-8");
    let env = [("X0NE_ENGINE_HOME", home), ("PATH", "")];
    let engine = env!("CARGO_BIN_EXE_x0ne-engine");
    assert_not_echoed(engine, &[], &env);
}

/// Runs `bin` with `args`: its exit code and everything it printed.
fn run(bin: &str, args: &[String], env: &[(&str, &str)]) -> (Option<i32>, String) {
    let out = Command::new(bin)
        .env_remove("LD_LIBRARY_PATH")
        .args(args)
        .envs(env.iter().copied())
        .stdin(Stdio::null())
        .output()
        .expect("the binary runs");
    let all = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code(), all)
}

/// Until 2026-10-22 `--expected-model-digest=<hex>` served with no pin and
/// `--max-queue 8o` with 8. Each is now exit 2, before the measurements file
/// is read (an absent one is 87), and the refused value is never printed.
#[test]
fn x0ne_infer_refuses_what_it_cannot_read_before_anything() {
    let dir = tempfile::tempdir().expect("tempdir");
    let absent = dir.path().join("absent.json");
    let prefix = [
        "--model-dir",
        "/nonexistent",
        "--listen",
        "127.0.0.1:0",
        "--measurements",
        absent.to_str().expect("utf-8"),
    ];
    let secret = key();
    let s = |v: &str| v.to_string();
    let mut cases: Vec<(Vec<String>, &str)> = vec![
        (
            vec![format!("--expected-model-digest={secret}")],
            "--expected-model-digest",
        ),
        (
            vec![s("--expected-model-digset"), secret.clone()],
            "--expected-model-digset",
        ),
        (vec![s("--max-queue"), format!("{secret}x")], "--max-queue"),
        (vec![s("--max-queue"), s("-1")], "--max-queue"),
        (
            vec![s("--max-completion-tokens"), secret.clone()],
            "--max-completion-tokens",
        ),
        (
            vec![s("--max-completion-tokens"), s("0")],
            "--max-completion-tokens",
        ),
        (vec![s("--max-handshakes"), s("0")], "--max-handshakes"),
        (
            vec![s("--max-handshakes"), secret.clone()],
            "--max-handshakes",
        ),
        (
            vec![s("--allow-host"), format!("{secret} !")],
            "--allow-host",
        ),
        (vec![s("--listen"), String::new()], "--listen"),
        (vec![s("--model-dir")], "--model-dir"),
    ];
    for form in forms() {
        cases.push((form, "unknown"));
    }
    let zero_key = "00".repeat(32);
    for (case, named) in cases {
        let args: Vec<String> = prefix
            .iter()
            .map(|a| a.to_string())
            .chain(case.clone())
            .collect();
        let (code, all) = run(
            env!("CARGO_BIN_EXE_x0ne-infer"),
            &args,
            &[("X0NE_INFER_KEY_HEX", zero_key.as_str())],
        );
        assert_eq!(code, Some(2), "{case:?}: {all}");
        assert!(all.contains(named), "{case:?} not named: {all}");
        assert!(!all.contains(&secret[..8]), "{case:?} printed it: {all}");
    }
    // A refused --device, --attest or --expect-measurement keeps its own
    // code, still before anything is collected, without its value.
    for (flag, value, want) in [
        ("--device", secret.clone(), 81),
        ("--device", format!("cuda:{secret}"), 81),
        ("--attest", secret.clone(), 82),
        ("--expect-measurement", secret.clone(), 84),
        ("--expect-measurement", format!("{secret}=00"), 84),
    ] {
        let args: Vec<String> = prefix
            .iter()
            .map(|a| a.to_string())
            .chain([flag.to_string(), value])
            .collect();
        let (code, all) = run(env!("CARGO_BIN_EXE_x0ne-infer"), &args, &[]);
        assert_eq!(code, Some(want), "{flag}: {all}");
        assert!(!all.contains(&secret[..8]), "{flag} printed it: {all}");
    }
}

/// `--nras-key-file`: the NVIDIA API key NRAS will require. The file is read
/// after the arguments and before anything else; what it holds is never
/// printed, whether it is taken, refused, or typed on the command line.
#[test]
fn the_nras_key_is_read_first_and_never_printed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let absent = dir.path().join("absent.json");
    let canary = "nvapi-CANARY5ec75ec75ec75ec75ec75ec7";
    let good = dir.path().join("nras.key");
    std::fs::write(&good, format!("{canary}\n")).expect("write");
    let two = dir.path().join("two.key");
    std::fs::write(&two, format!("{canary} {canary}")).expect("write");
    let link = dir.path().join("link.key");
    std::os::unix::fs::symlink(&good, &link).expect("symlink");
    let path = |p: &std::path::Path| p.to_str().expect("utf-8").to_string();
    let local = |key: &std::path::Path| {
        vec![
            "--model-dir".into(),
            "/nonexistent".into(),
            "--listen".into(),
            "127.0.0.1:0".into(),
            "--measurements".into(),
            path(&absent),
            "--nras-key-file".into(),
            path(key),
        ]
    };
    let zero_key = "00".repeat(32);
    let bin = env!("CARGO_BIN_EXE_x0ne-infer");
    let env = [("X0NE_INFER_KEY_HEX", zero_key.as_str())];
    // Taken: the run goes on to the absent measurements file (87).
    let (code, all) = run(bin, &local(&good), &env);
    assert_eq!(code, Some(87), "{all}");
    // Refused before the measurements: 101, the path named, not the content.
    for (file, says) in [
        (&two, "one token"),
        (&link, "not a regular file"),
        (&dir.path().join("absent.key"), "No such file"),
    ] {
        let (code, all) = run(bin, &local(file), &env);
        assert_eq!(code, Some(101), "{}: {all}", file.display());
        assert!(
            all.contains("--nras-key-file") && all.contains(says),
            "{}: {all}",
            file.display()
        );
        assert!(!all.contains("CANARY"), "{}: {all}", file.display());
    }
    // Cloud mode reads it too, before its config (absent: 95).
    let cloud = vec![
        "--cloud-config".to_string(),
        path(&absent),
        "--nras-key-file".into(),
        path(&good),
    ];
    let (code, all) = run(bin, &cloud, &[]);
    assert_eq!(code, Some(95), "{all}");
    assert!(!all.contains("CANARY"), "{all}");
    // On the command line it is an unknown argument, never quoted.
    for form in [
        vec!["--nras-key".to_string(), canary.into()],
        vec![format!("--nras-key={canary}")],
        vec![format!("--nras-key-file={canary}")],
    ] {
        let mut args = local(&good);
        args.extend(form.clone());
        let (code, all) = run(bin, &args, &env);
        assert_eq!(code, Some(2), "{form:?}: {all}");
        assert!(all.contains("unknown"), "{form:?}: {all}");
        assert!(!all.contains("CANARY"), "{form:?} printed it: {all}");
    }
}

/// A key typed where a model name goes was quoted back by x0ne-engine:
/// `x0ne-engine pull <key>` printed `"<key>" is not a model name`.
#[test]
fn x0ne_engine_does_not_print_a_refused_model_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    let home = dir.path().join("home");
    let home = home.to_str().expect("utf-8");
    let env = [("X0NE_ENGINE_HOME", home), ("PATH", "")];
    let secret = key();
    for cmd in ["pull", "verify", "rm", "path", "run"] {
        for form in forms() {
            let args: Vec<String> = std::iter::once(cmd.to_string()).chain(form).collect();
            let (code, all) = run(env!("CARGO_BIN_EXE_x0ne-engine"), &args, &env);
            assert_eq!(code, Some(2), "{args:?}: {all}");
            assert!(!all.contains(&secret[..8]), "{args:?} printed it: {all}");
        }
    }
}

/// The relay's chat reads its key from a file: the key itself typed after
/// `--key-file` is refused as a key, and no refused value is printed.
#[test]
fn the_relay_chat_refuses_a_key_on_its_command_line() {
    let proxy = env!("CARGO_BIN_EXE_x0ne-engine-proxy");
    let secret = key();
    for (flag, value) in [
        ("--listen", secret.clone()),
        ("--max-tokens", secret.clone()),
        ("--key-file", secret.clone()),
        ("--key-file", format!("/nonexistent/{secret}")),
    ] {
        let args = vec![
            "chat".to_string(),
            "--listen".into(),
            "127.0.0.1:9".into(),
            flag.into(),
            value,
        ];
        let (code, all) = run(proxy, &args, &[]);
        assert!(code != Some(0), "{args:?}: {all}");
        assert!(all.contains(flag), "{args:?} not named: {all}");
        assert!(!all.contains(&secret[..8]), "{args:?} printed it: {all}");
    }
    let args = vec!["chat".to_string(), "--key-file".into(), secret.clone()];
    let (_, all) = run(proxy, &args, &[]);
    assert!(all.contains("not the key"), "{all}");
}
