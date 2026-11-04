#!/usr/bin/env bash
# make core-gpu: x0ne Core inside x0ne-infer (`--features cuda`), built and tested on the local
# NVIDIA GPU. make ci builds without CUDA (Core's kernels need nvcc, its tests a GPU): run this
# before merging a change of x0ne-core/ or of the code that serves through it.
#
#   build     x0ne-infer and its lib test binary, release, --features cuda (the tests add candle-reference)
#   test      core_backend's tests, the GPU ones included: the conformance suite on Qwen3-0.6B and
#             Qwen3-8B, generate() equal to Core's own greedy decoding on the five bench prompts,
#             the device's greedy path equal to the host's
#   sanitize  compute-sanitizer memcheck and initcheck over the Qwen3-0.6B suite
#
# Environment:
#   CORE_GPU_IMAGE        a CUDA devel image with nvcc and a Rust toolchain on PATH (required)
#   CORE_GPU_VENDOR       a `cargo vendor` directory of this Cargo.lock (required: no network)
#   CORE_GPU_QWEN3_0_6B   Qwen3-0.6B directory (config.json, tokenizer.json, *.safetensors)
#   CORE_GPU_QWEN3_8B     Qwen3-8B directory (test only)
#   CORE_GPU_PROMPTS      the bench prompts: short, 4k, 8k, 16k .prompt.txt (test only)
#   CORE_GPU_TARGET       build directory (default target/core-gpu)
#   CORE_GPU_STEPS        default "build test sanitize"
#   CORE_GPU_PODMAN_ARGS  extra podman flags, split on spaces (default --device nvidia.com/gpu=all)
# Each step runs in a container without network, removed by its name. Exit 0: every step green.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
image=${CORE_GPU_IMAGE:?set CORE_GPU_IMAGE to a CUDA devel image with nvcc and Rust}
vendor=${CORE_GPU_VENDOR:?set CORE_GPU_VENDOR to a cargo vendor directory of Cargo.lock}
build=${CORE_GPU_TARGET:-$ROOT/target/core-gpu}
steps=${CORE_GPU_STEPS:-build test sanitize}
podman_args=${CORE_GPU_PODMAN_ARGS:---device nvidia.com/gpu=all}
features=cuda
mkdir -p "$build/target" "$build/cargo-home"
cat > "$build/cargo-home/config.toml" <<'TOML'
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "/vendor"

[net]
offline = true
TOML
mounts=(-v "$ROOT:/src:ro" -v "$vendor:/vendor:ro" -v "$build/target:/target" -v "$build/cargo-home:/cargo-home")
envs=()
for v in QWEN3_0_6B QWEN3_8B PROMPTS; do
  var=CORE_GPU_$v
  if [ -n "${!var:-}" ]; then
    mounts+=(-v "${!var}:/data/$v:ro")
    envs+=(-e "X0NE_INFER_CORE_$v=/data/$v")
  fi
done
need() {
  local var
  for var in "$@"; do
    [ -n "${!var:-}" ] || { echo "core-gpu: $step needs $var" >&2; exit 2; }
  done
}
# run NAME CMD...: CMD in the image, the tree read-only.
run() {
  local name="x0ne-core-gpu-$1-$$"
  shift
  # shellcheck disable=SC2086
  podman run --rm --network=none --name "$name" $podman_args "${mounts[@]}" "${envs[@]}" -w /src \
    -e CARGO_HOME=/cargo-home -e CARGO_TARGET_DIR=/target -e CARGO_INCREMENTAL=0 "$image" "$@"
}
cargo_args=(--release --locked --offline -p x0ne-infer --features "$features")
# The lib test binary of x0ne-infer, as cargo names it (built by the build step).
test_bin() {
  run bin cargo test "${cargo_args[@]}" --lib --no-run --message-format=json \
    | sed -n 's/.*"executable":"\([^"]*\/x0ne_infer-[^"]*\)".*/\1/p' | tail -1
}
verdict=0
for step in $steps; do
  rc=0
  started=$(date +%s)
  case $step in
    build)
      run build cargo build "${cargo_args[@]}" --bin x0ne-infer || rc=$?
      [ "$rc" != 0 ] || run build cargo test "${cargo_args[@]}" --lib --no-run || rc=$?
      ;;
    test)
      need CORE_GPU_QWEN3_0_6B CORE_GPU_QWEN3_8B CORE_GPU_PROMPTS
      bin=$(test_bin)
      [ -n "$bin" ] || { echo "core-gpu: no test binary, run the build step" >&2; exit 2; }
      run tests "$bin" core_backend:: --include-ignored --test-threads=1 --nocapture || rc=$?
      ;;
    sanitize)
      need CORE_GPU_QWEN3_0_6B
      bin=$(test_bin)
      [ -n "$bin" ] || { echo "core-gpu: no test binary, run the build step" >&2; exit 2; }
      for tool in memcheck initcheck; do
        san=(compute-sanitizer --tool "$tool" --error-exitcode 9)
        # memcheck only: "explicit" reports the errors of the API calls made, not the driver's
        # log warnings (as x0ne Core's scripts/test.sh sanitize).
        if [ "$tool" = memcheck ]; then san+=(--report-api-errors explicit); fi
        echo "== $tool: the Qwen3-0.6B suite"
        run "$tool" "${san[@]}" "$bin" core_backend::tests::linked_tests::gpu::core_passes_the_suite_0_6b \
          --exact --include-ignored --test-threads=1 || rc=$?
      done
      ;;
    *)
      echo "core-gpu: unknown step $step (build, test, sanitize)" >&2
      exit 2
      ;;
  esac
  echo "core-gpu: $step rc=$rc in $(($(date +%s) - started)) s"
  [ "$rc" = 0 ] || verdict=1
done
exit "$verdict"
