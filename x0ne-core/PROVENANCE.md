# Provenance of `x0ne-core/`

The crates `x0ne-core-kernels` (`kernels/`) and `x0ne-core-runtime`
(`runtime/`) are x0ne Core's, copied unchanged (2026-11-04: greedy decoding
on the device through a session, `Session::into_greedy`, 2 steps ahead of the
caller, and `Session::into_greedy_ahead`). The copy is named by its content,
not by a commit: the fingerprint below, whose first 12 digits `x0ne-infer`
prints when it loads a model (`core: copy ...`). Nothing in them is changed
here: a change is made in x0ne Core, then copied. x0ne Core's repository is
not public: the fingerprint below is the reference for this copy, and
`crates/<path>` names a path there that a reader cannot check.

Copied (`crates/<path>` there, `<path>` here). Run from `crates/` there or
from this directory here, `sha256sum <these files, in this order> | sha256sum`
prints `43c553fb05b888ddf5e42f522a0ecc35c0f427685000ba5c8ae14992fa673012`.

- `kernels/Cargo.toml`
- `kernels/build.rs`
- `kernels/cuda/advance.cu`
- `kernels/cuda/argmax.cu`
- `kernels/cuda/attention.cu`
- `kernels/cuda/common.cuh`
- `kernels/cuda/embedding.cu`
- `kernels/cuda/moe.cu`
- `kernels/cuda/probe.cu`
- `kernels/cuda/rmsnorm.cu`
- `kernels/cuda/rope.cu`
- `kernels/cuda/swiglu.cu`
- `kernels/cuda/x0ne_kernels.cu`
- `kernels/cuda/x0ne_kernels.h`
- `kernels/src/lib.rs`
- `runtime/Cargo.toml`
- `runtime/src/arena.rs`
- `runtime/src/bin/x0ne-core-roofline.rs`
- `runtime/src/config.rs`
- `runtime/src/cuda.rs`
- `runtime/src/engine.rs`
- `runtime/src/error.rs`
- `runtime/src/gpu.rs`
- `runtime/src/lib.rs`
- `runtime/src/metrics.rs`
- `runtime/src/ops.rs`
- `runtime/src/refs.rs`
- `runtime/src/roofline.rs`
- `runtime/src/st.rs`
- `runtime/src/weights.rs`

Written here, not copied: `Cargo.toml`, the workspace the two manifests
inherit from. Its values are x0ne Core's (`version`, `edition`,
`rust-version`, `authors`, `publish`, the pinned `half`, `memmap2`,
`safetensors`, `serde_json`, `sha2`), except `license`: here the two crates
are under AGPL-3.0-only, as the rest of the server (`LICENSING.md`,
2026-11-02); and its members are these two crates only.

Not copied, kept in x0ne Core with the tests that need them: the CPU oracle
(`crates/oracle`, a dev-dependency of the runtime that a build of Engine
never resolves), the tools (`crates/tools`: references, throughput benches),
the frozen references (`refs/`), the runtime's integration tests
(`crates/runtime/tests`), the build and test scripts. `x0ne-infer` serves
through `runtime/src/engine.rs` and what it uses; `refs.rs`, `roofline.rs` and
the `x0ne-core-roofline` binary are copied so that `runtime/src/lib.rs` stays
x0ne Core's, and serving uses none of them.
