// SPDX-License-Identifier: Apache-2.0
//! `test-anchor` on wasm32 only from a bench (2026-11-04): such a `.wasm`
//! takes a policy's own roots. The rule as the build script runs it, the
//! scripts that build it, and no `.wasm` in the tree.

#[path = "../build_guard.rs"]
mod build_guard;

use build_guard::{test_anchor_refusal, BENCH_VARIABLE};
use std::path::Path;

#[test]
fn a_wasm_with_test_anchor_is_refused_unless_a_bench_asks() {
    let refused = test_anchor_refusal(true, "wasm32", None).expect("refused");
    assert!(refused.contains("X0NE_SDK_BENCH=1"), "{refused}");
    assert!(test_anchor_refusal(true, "wasm32", Some("0")).is_some());
    assert!(test_anchor_refusal(true, "wasm32", Some("")).is_some());
    assert!(test_anchor_refusal(true, "wasm32", Some("1")).is_none());
    // The pinned .wasm, and the native builds of x0ne-infer's tests.
    assert!(test_anchor_refusal(false, "wasm32", None).is_none());
    assert!(test_anchor_refusal(true, "x86_64", None).is_none());
}

fn workspace() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}

/// Every command that builds x0ne-sdk with `test-anchor` for wasm32 sets the
/// variable on that command, and writes the `.wasm` under `out/` (ignored).
#[test]
fn the_scripts_that_build_the_bench_wasm_ask_for_it_and_keep_it_in_out() {
    let mut seen = 0;
    for rel in ["scripts/sdk-wasm.sh", "scripts/e2e-local.sh"] {
        let text = std::fs::read_to_string(workspace().join(rel)).unwrap();
        for line in text.lines() {
            if line.contains("wasm32-unknown-unknown") && line.contains("--features test-anchor") {
                seen += 1;
                assert!(
                    line.trim_start()
                        .starts_with(&format!("{BENCH_VARIABLE}=1 cargo build")),
                    "{rel}: {line}"
                );
            }
            if line.contains("x0ne_sdk-bench.wasm") && line.contains("cp ") {
                assert!(line.contains("/out/x0ne_sdk-bench.wasm"), "{rel}: {line}");
            }
        }
    }
    assert_eq!(seen, 2, "the two bench builds");
    let ignore = std::fs::read_to_string(workspace().join(".gitignore")).unwrap();
    assert!(ignore.lines().any(|l| l == "/out/"), ".gitignore: /out/");
}

/// No wasm module in the sources: a bench build cannot leave with them.
/// Build outputs (`target/`, `out/`), git's own directory and other checkouts
/// (a directory with its own `.git`) are not read.
#[test]
fn no_wasm_file_is_in_the_sources() {
    fn walk(dir: &Path, found: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().into_owned();
            let kind = entry.file_type().unwrap();
            if kind.is_dir() {
                let other_checkout = entry.path().join(".git").exists();
                if !other_checkout
                    && !matches!(name.as_str(), "target" | "out" | ".git" | "node_modules")
                {
                    walk(&entry.path(), found);
                }
            } else {
                // A wasm module by its magic: ci/Containerfile.wasm is a text file.
                let mut magic = [0u8; 4];
                let module = std::fs::File::open(entry.path())
                    .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic))
                    .is_ok_and(|()| magic == *b"\0asm");
                if module {
                    found.push(entry.path().display().to_string());
                }
            }
        }
    }
    let mut found = Vec::new();
    walk(workspace(), &mut found);
    assert!(found.is_empty(), "{found:?}");
}
