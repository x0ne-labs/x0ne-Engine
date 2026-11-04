#!/usr/bin/env bash
# The caller-sealed path on this host, SDK to model (x0ne-infer src/e2e_td_tests.rs):
# the TD's server code on synthetic TDX evidence, a recording carrier, x0ne-sdk.
#
# Outside the container (the default): builds ci/Containerfile.wasm on the CI
# image when it is missing, the only step with network (as scripts/sdk-wasm.sh).
# Then, with no network, under the Rust build lock shared with make ci:
#   1. x0ne-sdk for wasm32-unknown-unknown, release, with `test-anchor` (and
#      X0NE_SDK_BENCH=1, without which x0ne-sdk's build.rs refuses it);
#   2. e2e_td_tests with the scripted model (what make ci runs);
#   3. e2e_td_tests::wasm, the wasm32 build in Node through the carrier;
#   4. with X0NE_E2E_QWEN3_DIR, e2e_td_tests::qwen3: Qwen3-0.6B's
#      config.json, tokenizer.json and model.safetensors (the catalog's files,
#      checked by the TD's digest) on CPU, release build. Without it, skipped.
# Writes out/e2e-local/{x0ne_sdk-bench.wasm,test.log,latest.json};
# exit 0 = green, 1 = red, 2 = setup.
#
# Environment, as ci/ci.sh, or from CI_ENV_FILE: CI_CACHE, CI_LOCK,
# CI_CGROUP_PARENT, CI_MEMORY (16g), CI_JOBS (nproc), CI_WAIT_MAX (3600 s),
# SCCACHE_BIN and SCCACHE_DISK (sccache when both exist). CI_VENDOR: the
# vendored crates of this Cargo.lock (make ci makes them). The target is
# $CI_CACHE/e2e-local/target, shared with no other script.
#
# `inside` is the half that runs in the container; not for the host.
set -euo pipefail
cmd=${1:-run}

inside() {
  export HOME=/tmp/home CARGO_HOME=/tmp/cargo-home CARGO_TARGET_DIR=/target
  mkdir -p "$HOME" "$CARGO_HOME"
  cp /vendor/config.toml "$CARGO_HOME/config.toml"
  if [ -x /usr/local/bin/sccache ] && [ -d /sccache ]; then
    export RUSTC_WRAPPER=/usr/local/bin/sccache SCCACHE_DIR=/sccache SCCACHE_CACHE_SIZE=${SCCACHE_CACHE_SIZE:-107374182400}
    sccache --start-server >/dev/null 2>&1 || true
  fi
  cd /src
  local built=/target/wasm32-unknown-unknown/release/x0ne_sdk.wasm
  rm -f "$built"
  X0NE_SDK_BENCH=1 cargo build --locked --offline -p x0ne-sdk --target wasm32-unknown-unknown --release --features test-anchor
  [ -s "$built" ] || { echo "no $built after the build" >&2; exit 2; }
  cp "$built" /out/x0ne_sdk-bench.wasm
  echo "== e2e_td_tests, scripted model, the SDK in Rust"
  cargo test --locked --offline -p x0ne-infer --lib -- e2e_td_tests
  echo "== e2e_td_tests::wasm, the wasm32 build in Node"
  X0NE_SDK_WASM=/out/x0ne_sdk-bench.wasm X0NE_NODE=/usr/local/bin/node \
    cargo test --locked --offline -p x0ne-infer --lib -- e2e_td_tests::wasm --ignored
  if [ -d /model ]; then
    echo "== e2e_td_tests::qwen3, Qwen3-0.6B on CPU, release"
    X0NE_E2E_QWEN3_DIR=/model \
      cargo test --locked --offline --release -p x0ne-infer --lib -- e2e_td_tests::qwen3 --ignored --nocapture
  else
    echo "== e2e_td_tests::qwen3: skipped, no X0NE_E2E_QWEN3_DIR"
  fi
}

if [ "$cmd" = inside ]; then inside; exit; fi

REPO=$(git rev-parse --show-toplevel)
CI_ENV_FILE=${CI_ENV_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/x0ne-engine/ci.env}
if [ -f "$CI_ENV_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in ''|'#'*) continue ;; esac
    key=${line%%=*}
    case $key in CI_CACHE|CI_LOCK|CI_WAIT_MAX|CI_CGROUP_PARENT|CI_MEMORY|CI_JOBS|SCCACHE_BIN|SCCACHE_DISK) ;; *) continue ;; esac
    [ -n "${!key+x}" ] || printf -v "$key" '%s' "${line#*=}"
  done < "$CI_ENV_FILE"
fi
CI_CACHE=${CI_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/x0ne-engine-ci}
LOCK=${CI_LOCK:-$CI_CACHE/build.lock}
CI_MEMORY=${CI_MEMORY:-16g}
CI_JOBS=${CI_JOBS:-$(nproc)}
CI_WAIT_MAX=${CI_WAIT_MAX:-3600}
SCCACHE_BIN=${SCCACHE_BIN:-$(command -v sccache || true)}
SCCACHE_DISK=${SCCACHE_DISK:-$CI_CACHE/sccache}
PODMAN_RUN="podman run --rm${CI_CGROUP_PARENT:+ --cgroup-parent=$CI_CGROUP_PARENT}"
IMG=localhost/x0ne-engine-ci:$(sha256sum "$REPO/ci/Containerfile" | cut -c1-12)
WIMG=localhost/x0ne-engine-sdk-wasm:$(sha256sum "$REPO/ci/Containerfile.wasm" | cut -c1-12)
VENDOR=${CI_VENDOR:-$CI_CACHE/vendor/$(sha256sum "$REPO/Cargo.lock" | cut -c1-16)}
TARGET=$CI_CACHE/e2e-local/target
OUT=$REPO/out/e2e-local

log() { printf 'e2e-local: %s\n' "$*" >&2; }
die() { log "$*"; exit 2; }

podman image exists "$IMG" || die "no CI image $IMG: make ci-image"
[ -s "$VENDOR/config.toml" ] || die "no vendored crates at $VENDOR: make ci once, or CI_VENDOR=..."
model=() model_name=scripted
if [ -n "${X0NE_E2E_QWEN3_DIR:-}" ]; then
  for f in config.json tokenizer.json model.safetensors; do
    p=$(readlink -f "$X0NE_E2E_QWEN3_DIR/$f") || die "X0NE_E2E_QWEN3_DIR: no $f"
    [ -f "$p" ] || die "X0NE_E2E_QWEN3_DIR: no $f"
    model+=(-v "$p:/model/$f:ro")
  done
  model_name=qwen3:0.6b
fi
sc=()
if [ -x "$SCCACHE_BIN" ] && [ -d "$SCCACHE_DISK" ]; then
  sc=(-v "$(readlink -f "$SCCACHE_BIN")":/usr/local/bin/sccache:ro -v "$SCCACHE_DISK":/sccache)
fi
if ! podman image exists "$WIMG"; then
  log "building $WIMG"
  podman build --pull=never --build-arg "CI_IMAGE=$IMG" -f "$REPO/ci/Containerfile.wasm" -t "$WIMG" "$REPO/ci" >&2
fi
mkdir -p "$OUT" "$TARGET" "$(dirname "$LOCK")"
rm -f "$OUT"/x0ne_sdk-bench.wasm

exec 9>"$LOCK"
t0=$SECONDS
flock -w "$CI_WAIT_MAX" 9 || die "build lock $LOCK still held after $CI_WAIT_MAX s"
log "lock: waited $((SECONDS - t0)) s; $PODMAN_RUN, image $WIMG, model $model_name"
started=$(date -Is); t0=$SECONDS
set +e
nice -n 10 $PODMAN_RUN --memory="$CI_MEMORY" --network=none --pids-limit=4096 --userns=keep-id \
  --name "engine-e2e-local-$$" \
  -v "$REPO":/src:ro -v "$VENDOR":/vendor:ro -v "$TARGET":/target -v "$OUT":/out \
  "${model[@]}" "${sc[@]}" \
  -e CARGO_PROFILE_DEV_DEBUG=line-tables-only -e CARGO_BUILD_JOBS="$CI_JOBS" \
  "$WIMG" bash /src/scripts/e2e-local.sh inside > "$OUT/test.log" 2>&1
rc=$?
set -e
exec 9>&-
status=red; [ $rc -eq 0 ] && status=green; [ $rc -eq 2 ] && status=setup
jq -n --arg status "$status" --arg sha "$(git -C "$REPO" rev-parse HEAD)" \
  --arg dirty "$(git -C "$REPO" status --porcelain --untracked-files=no | head -c1)" \
  --arg started "$started" --arg ended "$(date -Is)" --argjson duration $((SECONDS - t0)) \
  --arg image "$WIMG" --arg model "$model_name" --arg log "$OUT/test.log" \
  '{status: $status, sha: $sha, dirty: ($dirty != ""), started: $started, ended: $ended,
    duration_s: $duration, image: $image, model: $model, log: $log}' > "$OUT/latest.json.tmp"
mv "$OUT/latest.json.tmp" "$OUT/latest.json"
log "$status in $((SECONDS - t0)) s, model $model_name; $OUT/latest.json"
[ $rc -ne 0 ] && tail -n 30 "$OUT/test.log" >&2
exit $rc
