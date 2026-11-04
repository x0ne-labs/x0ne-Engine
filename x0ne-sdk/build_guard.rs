// SPDX-License-Identifier: Apache-2.0
//! Who may build x0ne-sdk for wasm32 with `test-anchor`: the build script
//! runs it, `tests/test_anchor_guard.rs` tests it. Such a `.wasm` takes a
//! policy's own roots (Intel, Google, NVIDIA), the TEST archive key and the
//! TEST log; a page that loaded it would trust whatever the policy names.

/// The variable a bench sets to `1` to build that `.wasm`
/// (`scripts/sdk-wasm.sh`, `scripts/e2e-local.sh`).
pub const BENCH_VARIABLE: &str = "X0NE_SDK_BENCH";

/// Why this build is refused, or `None`. `test_anchor`: the feature is on;
/// `target_arch`: Cargo's `CARGO_CFG_TARGET_ARCH`; `bench`: the value of
/// [`BENCH_VARIABLE`], if set. Native builds pass: the feature reaches them
/// through x0ne-infer's dev-dependency, for its tests and examples.
pub fn test_anchor_refusal(
    test_anchor: bool,
    target_arch: &str,
    bench: Option<&str>,
) -> Option<String> {
    if !test_anchor || target_arch != "wasm32" || bench == Some("1") {
        return None;
    }
    Some(format!(
        "x0ne-sdk: `test-anchor` on wasm32 builds a .wasm that takes the roots a policy \
         names, the TEST archive key and the TEST log; only a bench builds it, with \
         {BENCH_VARIABLE}=1 (scripts/sdk-wasm.sh). A page ships the build without the feature."
    ))
}
