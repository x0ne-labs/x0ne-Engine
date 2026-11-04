//! `--version` of the binaries of this crate that ship: x0ne-engine and
//! x0ne-engine-proxy in the x0ne-engine package, x0ne-infer in the CVM image:
//! `<name> <version>`, plus ` (git <sha>)` when the build set `X0NE_GIT_SHA`
//! (ci/inside-deb.sh). `make deb-check` checks the same on the installed .deb.

use std::process::Command;

fn expected(name: &str) -> String {
    let git = option_env!("X0NE_GIT_SHA")
        .map(|s| format!(" (git {s})"))
        .unwrap_or_default();
    format!("{name} {}{git}\n", env!("CARGO_PKG_VERSION"))
}

#[test]
fn each_packaged_binary_prints_its_version_and_the_build_commit() {
    let mut wrong = Vec::new();
    for (name, bin) in [
        ("x0ne-infer", env!("CARGO_BIN_EXE_x0ne-infer")),
        ("x0ne-engine", env!("CARGO_BIN_EXE_x0ne-engine")),
        ("x0ne-engine-proxy", env!("CARGO_BIN_EXE_x0ne-engine-proxy")),
    ] {
        let out = Command::new(bin)
            // cargo sets LD_LIBRARY_PATH for tests; a release x0ne-infer refuses it (73).
            .env_remove("LD_LIBRARY_PATH")
            .arg("--version")
            .output()
            .expect("spawn");
        let line = String::from_utf8_lossy(&out.stdout).into_owned();
        if !out.status.success() || line != expected(name) {
            wrong.push(format!("{name}: {:?} {line:?}", out.status.code()));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
