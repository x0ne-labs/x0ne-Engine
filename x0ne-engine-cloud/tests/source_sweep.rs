//! The shipped lifecycle binary reads no API endpoint from the environment:
//! the cloud token travels to `https://*.googleapis.com` and nowhere else.
//! The loopback override the integration tests use exists only under the
//! `test-endpoints` feature, which a release build refuses to compile with.
//!
//! Mirror of x0ne-infer's `cloud_source_sweep.rs`. A failure means a planted
//! variable can again send a live bearer token in cleartext to a local port.

use std::path::{Path, PathBuf};

const GATE: &str = "#[cfg(feature = \"test-endpoints\")]";

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn src(name: &str) -> String {
    let path = manifest_dir().join("src").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Everything before the first `#[cfg(test)]`: the shipped code.
fn shipped(name: &str) -> String {
    let source = src(name);
    source[..source.find("#[cfg(test)]").unwrap_or(source.len())].to_string()
}

fn is_fn_line(l: &str) -> bool {
    let t = l.trim_start();
    let t = t.strip_prefix("pub ").unwrap_or(t);
    let t = t.strip_prefix("pub(crate) ").unwrap_or(t);
    let t = t.strip_prefix("async ").unwrap_or(t);
    t.starts_with("fn ")
}

/// Lines of `body` that read the environment outside a function whose
/// attributes carry [`GATE`].
fn ungated_env_reads(body: &str) -> Vec<String> {
    let lines: Vec<&str> = body.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if !(line.contains("std::env") || line.contains("env::var")) {
            continue;
        }
        let Some(f) = (0..=i).rev().find(|&j| is_fn_line(lines[j])) else {
            out.push(format!("{}: {}", i + 1, line.trim()));
            continue;
        };
        let gated = (0..f)
            .rev()
            .map(|j| lines[j].trim())
            .take_while(|l| l.starts_with("#[") || l.starts_with("///") || l.starts_with("//"))
            .any(|l| l == GATE);
        if !gated {
            out.push(format!("{}: {}", i + 1, line.trim()));
        }
    }
    out
}

/// `http.rs`, the only module that names the API hosts: no override is
/// compiled into a release, and the client is HTTPS-only there.
#[test]
fn the_endpoint_override_exists_only_under_the_test_feature() {
    let body = shipped("http.rs");
    let ungated = ungated_env_reads(&body);
    assert!(
        ungated.is_empty(),
        "http.rs reads the environment outside {GATE}: {ungated:?}"
    );
    let guard = body
        .lines()
        .position(|l| l.contains("compile_error!"))
        .expect("http.rs carries a compile_error! guard");
    let attr = body.lines().nth(guard - 1).unwrap_or("");
    assert!(
        attr.contains("feature = \"test-endpoints\"") && attr.contains("not(debug_assertions)"),
        "the guard must refuse the feature outside debug builds, got: {attr}"
    );
    assert_eq!(
        body.matches("https_only(").count(),
        1,
        "one client, one https_only decision"
    );
    for forbidden in ["danger_accept_invalid", "no_proxy", "proxy("] {
        assert!(!body.contains(forbidden), "http.rs: {forbidden}");
    }
}

/// Every other module: the variables read are directories, the project id,
/// what gcloud needs, or test timings that `Knobs::from_env` refuses unless
/// the endpoint is the loopback (never, in a release).
#[test]
fn no_other_module_reads_an_endpoint_from_the_environment() {
    let allowed = [
        "\"GOOGLE_CLOUD_PROJECT\"",
        "\"XDG_RUNTIME_DIR\"",
        "\"XDG_STATE_HOME\"",
        "\"HOME\"",
        "var_os(name)", // lib.rs: X0NE_ENGINE_CLOUD_TEST_* (`ms`, `reference_dir`); color.rs: see below
        "var_os(k)",    // proxy_link.rs: the allow-listed relay environment
        "vars_os()",    // token.rs: the allow-listed gcloud environment
        "args_os()",
        "current_exe()",
    ];
    let mut names: Vec<String> = Vec::new();
    for dir in ["", "provider"] {
        let d = manifest_dir().join("src").join(dir);
        for entry in std::fs::read_dir(&d).unwrap() {
            let p = entry.unwrap().path();
            if p.extension().is_some_and(|e| e == "rs") {
                let rel = p.strip_prefix(manifest_dir().join("src")).unwrap();
                names.push(rel.to_string_lossy().into_owned());
            }
        }
    }
    names.sort();
    assert!(names.len() >= 12, "{names:?}");
    for name in names.iter().filter(|n| n.as_str() != "http.rs") {
        let body = shipped(name);
        for line in body.lines().filter(|l| l.contains("std::env")) {
            assert!(
                allowed.iter().any(|a| line.contains(a)),
                "{name} reads the environment for something new: {}",
                line.trim()
            );
        }
        assert!(
            !body.contains("X0NE_ENGINE_GCP_ENDPOINT") || name.as_str() == "lib.rs",
            "{name} names the endpoint override"
        );
    }
}

/// `color.rs`, compiled into the relay too (x0ne-infer's `term_color`): one
/// read, of the four color variables, which the relay receives (`KEPT_ENV`).
/// Without `--reference`, `up` names the image of the reference the relay
/// will take: the same directory and file-name rule as x0ne-infer
/// `reference.rs`. A drift would refuse every such launch (the relay names
/// its image), not boot a wrong one.
#[test]
fn the_installed_reference_up_reads_is_the_relays() {
    let path = manifest_dir().join("../x0ne-infer/src/reference.rs");
    let relay =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(
        relay.contains(&format!(
            "pub const DEFAULT_DIR: &str = \"{}\";",
            x0ne_engine_cloud::REFERENCE_DIR
        )),
        "REFERENCE_DIR is not x0ne-infer's DEFAULT_DIR"
    );
    let body = |s: &str| {
        let s = &s[s.find("fn version_key(").expect("fn version_key")..];
        s[..s.find("\n}\n").expect("end of version_key") + 2].to_string()
    };
    assert_eq!(body(&src("image_import.rs")), body(&relay));
}

#[test]
fn colors_read_four_variables_and_the_relay_gets_them() {
    let body = shipped("color.rs");
    let reads: Vec<&str> = body
        .lines()
        .filter(|l| l.contains("std::env") || l.contains("env::var"))
        .map(str::trim)
        .collect();
    assert_eq!(
        reads,
        [r#""NO_COLOR" | "CLICOLOR_FORCE" | "CLICOLOR" | "TERM" => std::env::var_os(name),"#]
    );
    let link = shipped("proxy_link.rs");
    let kept = &link[link.find("pub const KEPT_ENV").expect("KEPT_ENV")..];
    let kept = &kept[..kept.find("];").unwrap()];
    for v in ["NO_COLOR", "CLICOLOR_FORCE", "CLICOLOR", "TERM"] {
        assert!(kept.contains(&format!("\"{v}\"")), "KEPT_ENV lacks {v}");
    }
}

/// After a terminal hangup, stderr fails with EIO and `eprint!` panics:
/// measured exit 101 with no DELETE sent. Every module prints with `say!`.
#[test]
fn no_module_prints_to_stderr_with_a_macro_that_panics() {
    for dir in ["", "provider"] {
        let d = manifest_dir().join("src").join(dir);
        for entry in std::fs::read_dir(&d).unwrap() {
            let p = entry.unwrap().path();
            if p.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            let rel = p.strip_prefix(manifest_dir().join("src")).unwrap();
            let body = shipped(&rel.to_string_lossy());
            for line in body.lines() {
                let code = line.split("//").next().unwrap_or("");
                assert!(
                    !code.contains("eprint!(") && !code.contains("eprintln!("),
                    "{}: {}",
                    rel.display(),
                    line.trim()
                );
            }
        }
    }
}

/// The feature is never a default, and the crate enables it for its own
/// tests only (a dev-dependency on itself), so `cargo build --release`
/// cannot carry it.
#[test]
fn the_test_feature_is_not_a_default_and_is_enabled_for_tests_only() {
    let manifest = std::fs::read_to_string(manifest_dir().join("Cargo.toml")).unwrap();
    let section = |name: &str| -> String {
        let start = manifest
            .find(&format!("[{name}]"))
            .unwrap_or_else(|| panic!("Cargo.toml has no [{name}]"));
        let rest = &manifest[start + name.len() + 2..];
        rest[..rest.find("\n[").unwrap_or(rest.len())].to_string()
    };
    let features = section("features");
    assert!(features.contains("test-endpoints = []"), "{features}");
    assert!(
        !features
            .lines()
            .any(|l| l.trim_start().starts_with("default")),
        "a default feature list: {features}"
    );
    let dev = section("dev-dependencies");
    let me = dev
        .lines()
        .find(|l| l.trim_start().starts_with("x0ne-engine-cloud"))
        .expect("the crate is its own dev-dependency");
    assert!(
        me.contains("path = \".\"") && me.contains("features = [\"test-endpoints\"]"),
        "{me}"
    );
    for section_name in ["dependencies", "build-dependencies"] {
        if let Some(start) = manifest.find(&format!("[{section_name}]")) {
            let rest = &manifest[start..];
            let body = &rest[..rest[1..].find("\n[").map(|i| i + 1).unwrap_or(rest.len())];
            assert!(
                !body.contains("test-endpoints"),
                "[{section_name}] enables the test feature"
            );
        }
    }
    assert!(
        Path::new(&manifest_dir().join("src/http.rs")).exists(),
        "layout moved; update this sweep"
    );
}

// ── td_console: the words and codes it judges are x0ne-infer's ──────────────

fn infer_src(name: &str) -> String {
    let path = manifest_dir().join("../x0ne-infer/src").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Exit codes x0ne-infer's header names that a TD in cloud mode does not
/// exit with, and why `td_console::TD_EXITS` leaves them out.
const NOT_A_TD_EXIT: &[(u32, &str)] = &[
    (
        70,
        "key refused: the /v2/ready phase, not an exit of cloud mode",
    ),
    (
        84,
        "--expect-measurement: cloud mode refuses the flag with 95",
    ),
    (
        87,
        "no reference measurement: not cloud mode; 87 is also the relay's",
    ),
    (88, "the relay's, on the client's machine"),
    (89, "the relay's, on the client's machine"),
    (102, "the relay's, on the client's machine"),
];

/// A wording change in x0ne-infer would silently turn a final refusal back
/// into a 39-minute wait: the words `td_console` looks for must be its own.
#[test]
fn the_words_td_console_looks_for_are_printed_by_x0ne_infer() {
    use x0ne_engine_cloud::td_console::{NOT_THE_PIN, SAID, TCB_RATING};
    assert_eq!(SAID, "x0ne-infer: ");
    let main = infer_src("main.rs");
    assert!(main.contains(r#"eprintln!("x0ne-infer: {msg}");"#));
    let attest = infer_src("attest_v2.rs");
    assert!(attest.contains(r#"fail(76, format!("live TCB verification failed: {e}"))"#));
    let tcb = infer_src("tcb.rs");
    for w in TCB_RATING {
        assert!(tcb.contains(w), "tcb.rs no longer says {w:?}");
    }
    let catalog = infer_src("registry/catalog.rs");
    for w in NOT_THE_PIN {
        assert!(
            catalog.contains(w),
            "registry/catalog.rs no longer says {w:?}"
        );
    }
}

#[test]
fn td_console_judges_every_exit_code_of_x0ne_infer() {
    use x0ne_engine_cloud::td_console::TD_EXITS;
    let main = infer_src("main.rs");
    let from = main
        .find("//! Exit codes are distinct")
        .expect("the header");
    let to = from + main[from..].find("The `seal`").expect("the header's end");
    let codes: std::collections::BTreeSet<u32> = main[from..to]
        .split(|c: char| !c.is_ascii_digit())
        .filter_map(|n| n.parse().ok())
        .collect();
    assert!(
        codes.contains(&2) && codes.contains(&76) && codes.contains(&98),
        "{codes:?}"
    );
    for code in &codes {
        let judged = TD_EXITS.iter().any(|e| e.code as u32 == *code);
        let left_out = NOT_A_TD_EXIT.iter().any(|(c, _)| c == code);
        assert!(
            judged != left_out,
            "exit {code}: judged {judged}, left out {left_out}"
        );
    }
    for e in TD_EXITS {
        assert!(
            codes.contains(&(e.code as u32)),
            "{} is not x0ne-infer's",
            e.code
        );
    }
}
