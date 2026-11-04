//! `x0ne-infer` refuses to start under `LD_PRELOAD`, `LD_AUDIT` or
//! `GLIBC_TUNABLES`: through the first two the dynamic loader brings code the
//! image never measured into the TD process, before `main` runs; the third
//! steers that loader. An environment file injected `LD_PRELOAD`
//! into a CVM unit in Trail of Bits' review of Meta's Private Processing
//! (2025, finding 13). A release build also refuses `LD_LIBRARY_PATH`,
//! `CUDA_INJECTION64_PATH` and `NVTX_INJECTION64_PATH`, which choose the file a
//! library name opens. Measured on the real binary, from outside.
#![cfg(target_os = "linux")]

use std::process::{Command, Stdio};

/// Refused by every build.
const LOADER: [&str; 3] = ["LD_PRELOAD", "LD_AUDIT", "GLIBC_TUNABLES"];
/// Refused by a release build; cargo sets the first for every test.
const LIBRARY_CHOICE: [&str; 3] = [
    "LD_LIBRARY_PATH",
    "CUDA_INJECTION64_PATH",
    "NVTX_INJECTION64_PATH",
];

/// `x0ne-infer --help` with every variable above cleared, then `var` set.
fn run(var: Option<(&str, &str)>) -> (i32, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_x0ne-infer"));
    cmd.arg("--help")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    for v in LOADER.iter().chain(&LIBRARY_CHOICE) {
        cmd.env_remove(v);
    }
    if let Some((k, v)) = var {
        cmd.env(k, v);
    }
    let out = cmd.output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// SECURITY: each loader variable alone, even empty, ends the process with
/// the exit of the process hardening (73) and the name of the variable,
/// before any key is read.
#[test]
fn a_loader_variable_in_the_environment_refuses_to_start() {
    for var in LOADER {
        let (code, err) = run(Some((var, "")));
        assert_eq!(code, 73, "{var}: exit {code}, stderr: {err}");
        assert!(err.contains(var), "{var} not named: {err}");
        assert!(err.contains("process hardening"), "{var}: {err}");
    }
}

/// SECURITY: x0ne Core and NVML hand `dlopen` bare names after `main`
/// (`libcuda.so.1`, `libcublas.so.13`, `libnvidia-ml.so.1`); a
/// `libnvidia-ml.so.1` put on `LD_LIBRARY_PATH` ran its constructor in a
/// release `x0ne-infer` (2026-11-04). A release build refuses each variable
/// that chooses such a file, alone, even empty, with 73 and its name.
#[cfg(not(debug_assertions))]
#[test]
fn a_release_build_refuses_the_variables_that_choose_a_library() {
    for var in LIBRARY_CHOICE {
        let (code, err) = run(Some((var, "")));
        assert_eq!(code, 73, "{var}: exit {code}, stderr: {err}");
        assert!(err.contains(var), "{var} not named: {err}");
        assert!(err.contains("process hardening"), "{var}: {err}");
    }
}

/// A build with debug assertions takes them: cargo sets `LD_LIBRARY_PATH`
/// for every test and every `cargo run`.
#[cfg(debug_assertions)]
#[test]
fn a_debug_build_takes_the_variables_that_choose_a_library() {
    for var in LIBRARY_CHOICE {
        let (code, err) = run(Some((var, "/nonexistent")));
        assert_eq!(code, 0, "{var}: exit {code}, stderr: {err}");
    }
}

/// Control: without them `--help` is served, so the refusal above is the
/// variable's doing and not the binary's.
#[test]
fn control_without_loader_variables_help_is_served() {
    let (code, err) = run(None);
    assert_eq!(code, 0, "stderr: {err}");
}
