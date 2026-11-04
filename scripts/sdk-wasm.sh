#!/usr/bin/env bash
# The wasm32 build of x0ne-sdk, and its run in Node against the simulated TD.
#
# Outside the container (the default): builds ci/Containerfile.wasm on the CI
# image when it is missing, the only step with network (the wasm32 standard
# library of the pinned toolchain; Node comes from a local image, by digest).
# Then, with no network, under the Rust build lock shared with make ci:
#   1. x0ne-sdk for wasm32-unknown-unknown, release, twice: with `test-anchor`
#      (the bench's root is accepted; X0NE_SDK_BENCH=1, without which x0ne-sdk's
#      build.rs refuses it) and without (Intel's root, no other);
#   2. the `wasm_*` tests of x0ne-infer (src/e2e_sdk_tests.rs), which run
#      x0ne-sdk/js/smoke.mjs in Node on each build.
# Writes out/sdk-wasm/{x0ne_sdk.wasm,x0ne_sdk-bench.wasm,test.log,latest.json};
# exit 0 = green, 1 = red, 2 = setup.
#
# Environment, as ci/ci.sh, or from CI_ENV_FILE: CI_CACHE
# (${XDG_CACHE_HOME:-~/.cache}/x0ne-engine-ci), CI_LOCK ($CI_CACHE/build.lock),
# CI_CGROUP_PARENT, CI_MEMORY (16g), CI_JOBS (nproc), CI_WAIT_MAX (3600 s).
# CI_VENDOR: the vendored crates of this Cargo.lock; by default the directory
# make ci made for its hash (run make ci once on this tree first).
#
# `inside` is the half that runs in the container; not for the host.
set -euo pipefail
cmd=${1:-run}

inside() {
  export HOME=/tmp/home CARGO_HOME=/tmp/cargo-home CARGO_TARGET_DIR=/target
  mkdir -p "$HOME" "$CARGO_HOME"
  cp /vendor/config.toml "$CARGO_HOME/config.toml"
  cd /src
  local built=/target/wasm32-unknown-unknown/release/x0ne_sdk.wasm
  rm -f "$built"
  X0NE_SDK_BENCH=1 cargo build --locked --offline -p x0ne-sdk --target wasm32-unknown-unknown --release --features test-anchor
  [ -s "$built" ] || { echo "no $built after the build: is x0ne-sdk a cdylib?" >&2; exit 2; }
  cp "$built" /out/x0ne_sdk-bench.wasm
  rm -f "$built"
  cargo build --locked --offline -p x0ne-sdk --target wasm32-unknown-unknown --release
  [ -s "$built" ] || { echo "no $built after the pinned build" >&2; exit 2; }
  cp "$built" /out/x0ne_sdk.wasm
  X0NE_SDK_WASM=/out/x0ne_sdk-bench.wasm X0NE_SDK_WASM_PINNED=/out/x0ne_sdk.wasm \
    X0NE_NODE=/usr/local/bin/node \
    cargo test --locked --offline -p x0ne-infer --lib -- e2e_sdk_tests::wasm --ignored
}

if [ "$cmd" = inside ]; then inside; exit; fi

REPO=$(git rev-parse --show-toplevel)
CI_ENV_FILE=${CI_ENV_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/x0ne-engine/ci.env}
if [ -f "$CI_ENV_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in ''|'#'*) continue ;; esac
    key=${line%%=*}
    case $key in CI_CACHE|CI_LOCK|CI_WAIT_MAX|CI_CGROUP_PARENT|CI_MEMORY|CI_JOBS) ;; *) continue ;; esac
    [ -n "${!key+x}" ] || printf -v "$key" '%s' "${line#*=}"
  done < "$CI_ENV_FILE"
fi
CI_CACHE=${CI_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/x0ne-engine-ci}
LOCK=${CI_LOCK:-$CI_CACHE/build.lock}
CI_MEMORY=${CI_MEMORY:-16g}
CI_JOBS=${CI_JOBS:-$(nproc)}
CI_WAIT_MAX=${CI_WAIT_MAX:-3600}
PODMAN_RUN="podman run --rm${CI_CGROUP_PARENT:+ --cgroup-parent=$CI_CGROUP_PARENT}"
IMG=localhost/x0ne-engine-ci:$(sha256sum "$REPO/ci/Containerfile" | cut -c1-12)
WIMG=localhost/x0ne-engine-sdk-wasm:$(sha256sum "$REPO/ci/Containerfile.wasm" | cut -c1-12)
VENDOR=${CI_VENDOR:-$CI_CACHE/vendor/$(sha256sum "$REPO/Cargo.lock" | cut -c1-16)}
TARGET=$CI_CACHE/v2/target
OUT=$REPO/out/sdk-wasm

log() { printf 'sdk-wasm: %s\n' "$*" >&2; }
die() { log "$*"; exit 2; }

podman image exists "$IMG" || die "no CI image $IMG: make ci-image"
[ -s "$VENDOR/config.toml" ] || die "no vendored crates at $VENDOR: make ci once, or CI_VENDOR=..."
if ! podman image exists "$WIMG"; then
  log "building $WIMG"
  podman build --pull=never --build-arg "CI_IMAGE=$IMG" -f "$REPO/ci/Containerfile.wasm" -t "$WIMG" "$REPO/ci" >&2
fi
mkdir -p "$OUT" "$TARGET" "$(dirname "$LOCK")"
rm -f "$OUT"/x0ne_sdk.wasm "$OUT"/x0ne_sdk-bench.wasm

exec 9>"$LOCK"
t0=$SECONDS
flock -w "$CI_WAIT_MAX" 9 || die "build lock $LOCK still held after $CI_WAIT_MAX s"
log "lock: waited $((SECONDS - t0)) s; image $WIMG, vendor $VENDOR"
started=$(date -Is); t0=$SECONDS
set +e
nice -n 10 $PODMAN_RUN --memory="$CI_MEMORY" --network=none --pids-limit=4096 --userns=keep-id \
  --name "engine-sdk-wasm-$$" \
  -v "$REPO":/src:ro -v "$VENDOR":/vendor:ro -v "$TARGET":/target -v "$OUT":/out \
  -e CARGO_PROFILE_DEV_DEBUG=line-tables-only -e CARGO_BUILD_JOBS="$CI_JOBS" \
  "$WIMG" bash /src/scripts/sdk-wasm.sh inside > "$OUT/test.log" 2>&1
rc=$?
set -e
exec 9>&-
status=red; [ $rc -eq 0 ] && status=green; [ $rc -eq 2 ] && status=setup
size() { [ -s "$1" ] && stat -c %s "$1" || echo 0; }
jq -n --arg status "$status" --arg sha "$(git -C "$REPO" rev-parse HEAD)" \
  --arg started "$started" --arg ended "$(date -Is)" --argjson duration $((SECONDS - t0)) \
  --arg image "$WIMG" --argjson bench "$(size "$OUT/x0ne_sdk-bench.wasm")" \
  --argjson pinned "$(size "$OUT/x0ne_sdk.wasm")" --arg log "$OUT/test.log" \
  '{status: $status, sha: $sha, started: $started, ended: $ended, duration_s: $duration,
    image: $image, wasm_bytes: {bench: $bench, pinned: $pinned}, log: $log}' > "$OUT/latest.json.tmp"
mv "$OUT/latest.json.tmp" "$OUT/latest.json"
log "$status in $((SECONDS - t0)) s; bench $(size "$OUT/x0ne_sdk-bench.wasm") B, pinned $(size "$OUT/x0ne_sdk.wasm") B; $OUT/latest.json"
[ $rc -ne 0 ] && tail -n 30 "$OUT/test.log" >&2
exit $rc
