//! `x0ne-engine-cloud --version`: `<name> <version>`, plus ` (git <sha>)` when
//! the build set `X0NE_GIT_SHA` (ci/inside-deb.sh), like the other binaries
//! of the x0ne-engine package. `make deb-check` checks the installed .deb.

#[test]
fn version_prints_the_build_commit() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_x0ne-engine-cloud"))
        .arg("--version")
        .output()
        .expect("spawn");
    let git = option_env!("X0NE_GIT_SHA")
        .map(|s| format!(" (git {s})"))
        .unwrap_or_default();
    assert!(out.status.success(), "{:?}", out.status);
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("x0ne-engine-cloud {}{git}\n", env!("CARGO_PKG_VERSION"))
    );
}
