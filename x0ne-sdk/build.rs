// `pcs.rs`, `tcb.rs`, `gce_endorsement.rs`, `ccel.rs`, `reference.rs`,
// `tdvf.rs` and `nras.rs` are x0ne-infer's files, compiled into this crate by `#[path]`
// (src/lib.rs). Their `#[cfg(test)]` modules need x0ne-infer's fixtures and
// dev-dependencies, which this crate does not have, and `cargo test
// --all-targets` builds a library's test target even with `test = false`.
// The `x0ne_sdk` cfg lets those files leave out their test modules, and what
// opens a socket or reads a disk (the fetches, the reference's loading), when
// built here, and nowhere else. `x0ne_sdk_test_anchor`, with the `test-anchor`
// feature alone, lets `nras.rs` pin a bench's NVIDIA intermediate
// (`nras::throwaway_pin`); no other build can make an NRAS pin. On wasm32,
// `test-anchor` is refused unless a bench asks for it (`build_guard.rs`).
#[path = "build_guard.rs"]
mod build_guard;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(x0ne_sdk)");
    println!("cargo::rustc-check-cfg=cfg(x0ne_sdk_test_anchor)");
    println!("cargo::rustc-cfg=x0ne_sdk");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let bench = std::env::var(build_guard::BENCH_VARIABLE).ok();
    let feature = std::env::var_os("CARGO_FEATURE_TEST_ANCHOR").is_some();
    if let Some(why) = build_guard::test_anchor_refusal(feature, &arch, bench.as_deref()) {
        panic!("{why}");
    }
    if std::env::var_os("CARGO_FEATURE_TEST_ANCHOR").is_some() {
        println!("cargo::rustc-cfg=x0ne_sdk_test_anchor");
    }
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=build_guard.rs");
    println!(
        "cargo::rerun-if-env-changed={}",
        build_guard::BENCH_VARIABLE
    );
}
