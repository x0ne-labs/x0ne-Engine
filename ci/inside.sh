#!/usr/bin/env bash
# Runs inside the CI container (ci/ci.sh), with no network. Mounts: /src (the
# commit, read-only), /vendor (crates), /advisory-db (RustSec, read-only),
# /target, /out (step logs and steps.tsv). Exit 0 = every step green.
#
# The steps run in six lanes at once (CI_LANES=0: one after the other):
#   light  fmt ci-sh features deny gitleaks        no cargo build
#   a      clippy test test-doc           default features, /target/debug
#   b      clippy-all doc test-all        every feature, /target/all/debug
#   c      build-cloud                    /target/release (the profile that ships)
#          comment-invariance             /target/release (the .deb binaries, built twice)
#          release-refusal                x0ne-infer, release profile, default features, run
#   d      test-release                   /target/test-release/release (the same profile)
#   e      core                           x0ne-core's own workspace, /target/core, no nvcc
# No two lanes write one profile directory: a test that runs the real binary
# (CARGO_BIN_EXE_*) would otherwise run the one another lane just built with other
# features (lab_option.rs went red that way). The build of one lane overlaps the tests
# of the others; CPU is shared, so a step's seconds move while the wall clock drops.
# test-release has its own target directory in both modes: x0ne-sdk is a cdylib, which Cargo
# writes without a hash in its file name, so its build with the tests' `test-anchor` and its
# build without replaced each other in /target/release, and each replacement rebuilt x0ne-infer:
# 56 to 78 s of test-release and 20 s of build-cloud on every run, 17 runs of 17, with no
# source changed (2026-11-10).
#
# A tree may add a workspace with steps of its own (LICENSING.md): its ci/steps.sh, sourced
# here, sets EXTRA_STEPS (added to the default list) and EXTRA_LANE_B, EXTRA_LANE_C and
# EXTRA_LANE_E (the lanes they run in), and defines extra_step <name>, which runs one of them
# with `step` and returns 1 for a name it does not know.
# At most one such file; a tree without one runs the steps above.
set -uo pipefail
ALL_FEATURES=${CI_ALL_FEATURES:-x0ne-infer/nvml,x0ne-infer/probe}
EXTRA_STEPS= EXTRA_LANE_B= EXTRA_LANE_C= EXTRA_LANE_E=
extra_step() { return 1; }
extra=0
for f in /src/*/ci/steps.sh; do
  [ -f "$f" ] || continue
  extra=$((extra + 1))
  [ "$extra" = 1 ] || { echo "inside.sh: more than one ci/steps.sh under /src" >&2; exit 2; }
  # shellcheck source=/dev/null
  . "$f" || { echo "inside.sh: $f failed" >&2; exit 2; }
done
STEPS=${CI_STEPS:-fmt ci-sh features clippy clippy-all test test-doc test-all test-release build-cloud comment-invariance release-refusal core doc deny gitleaks $EXTRA_STEPS}
LANES=${CI_LANES:-1}
LANE_LIGHT="fmt ci-sh features deny gitleaks"
LANE_A="clippy test test-doc"
LANE_B="clippy-all doc test-all $EXTRA_LANE_B"
LANE_C="build-cloud comment-invariance release-refusal $EXTRA_LANE_C"
LANE_D="test-release"
LANE_E="core $EXTRA_LANE_E"
# The target directory of test-release (see the lanes above).
TEST_RELEASE_TARGET=/target/test-release
# The steps test, test-all and test-release run each test binary through ci/test-runner.sh:
# libtest prints the time of each test in the step's log.
TEST_RUNNER=CARGO_TARGET_$(rustc -vV | sed -n 's/^host: //p' | tr 'a-z.-' 'A-Z__')_RUNNER="bash /src/ci/test-runner.sh"

export HOME=/tmp/home CARGO_HOME=/tmp/cargo-home CARGO_TARGET_DIR=/target
export XDG_CONFIG_HOME=$HOME/.config XDG_DATA_HOME=$HOME/.local/share XDG_CACHE_HOME=$HOME/.cache
# ci.sh runs this as an ordinary user (keep-id): tests/process_hardening.rs may not skip here.
export X0NE_REQUIRE_NON_ROOT=1
# The image ships jq: ci/tests/release-verify-tag.sh may not skip here either.
export X0NE_REQUIRE_JQ=1
# It ships cc, binutils, perl and dpkg-deb: ci/tests/strip-release.sh and deb-verify.sh may not skip.
export X0NE_REQUIRE_CI_TOOLS=1
mkdir -p "$HOME" "$CARGO_HOME" /out /out/steps.d
# Vendored crates replace crates.io; the config is the one `cargo vendor` printed.
cp /vendor/config.toml "$CARGO_HOME/config.toml"
# The read-only mount is copied: cargo-deny takes a write lock on the database even offline.
cp -r /advisory-db "$CARGO_HOME/advisory-dbs"
if [ -x /usr/local/bin/sccache ] && [ -d /sccache ]; then
  export RUSTC_WRAPPER=/usr/local/bin/sccache SCCACHE_DIR=/sccache SCCACHE_CACHE_SIZE=${SCCACHE_CACHE_SIZE:-107374182400}
  sccache --start-server >/dev/null 2>&1 || true
fi
cd /src

: > /out/steps.tsv
rm -f /out/steps.d/*
# step <name> <command...>: one log, one line in /out/steps.d/<name> (name, ok|fail, seconds,
# exit), one report printed in one write (lanes interleave otherwise).
step() {
  local name=$1; shift
  local t0=$SECONDS report
  if "$@" > "/out/$name.log" 2>&1; then
    printf '%s\tok\t%d\t0\n' "$name" $((SECONDS - t0)) > "/out/steps.d/$name"
    report="ok    $name ($((SECONDS - t0)) s)"
  else
    local e=$?
    printf '%s\tfail\t%d\t%d\n' "$name" $((SECONDS - t0)) "$e" > "/out/steps.d/$name"
    report="FAIL  $name ($((SECONDS - t0)) s, exit $e)"$'\n'"$(tail -n 40 "/out/$name.log" | sed 's/^/      /')"
  fi
  printf '%s\n' "$report"
}

run_step() {
  case $1 in
    # Not --all: x0ne-core/ is another workspace. Its two copied crates are formatted in x0ne
    # Core (PROVENANCE.md); the one written here, x0ne-core/oracle, by step core.
    fmt)        step fmt cargo fmt -- --check ;;
    ci-sh)      step ci-sh bash -c "bash /src/ci/tests/ci-sh.sh && bash /src/ci/tests/release-verify-tag.sh && bash /src/ci/tests/coverage.sh" ;;
    # The clippy of the GitHub `clippy` job: -D warnings, then its PEDANTIC list (clippy_ci).
    features)   step features bash -c "python3 -I /src/ci/features.py --check && bash /src/ci/tests/features.sh" ;;
    clippy)     step clippy clippy_ci ;;
    clippy-all) step clippy-all clippy_ci --features "$ALL_FEATURES" ;;
    test)       step test env "$TEST_RUNNER" cargo test --locked --offline --workspace --all-targets ;;
    # Not in the default list: every case of x0ne-engine-cloud/tests/chaos.rs (CI_STEPS=chaos).
    chaos)      step chaos env X0NE_ENGINE_CLOUD_CHAOS=full cargo test --locked --offline -p x0ne-engine-cloud --test chaos -- --nocapture ;;
    test-doc)   step test-doc cargo test --locked --offline --workspace --doc ;;
    test-all)   step test-all env "$TEST_RUNNER" cargo test --locked --offline -p x0ne-infer --all-targets --features "$ALL_FEATURES" ;;
    # Tests gated on not(debug_assertions), and the cloud_probe of e2e-cloud.sh, in the profile
    # that ships (release: Cargo.toml, line tables kept for ci/strip-release.sh), in their own
    # target directory: the builds of lane c have other features of x0ne-sdk.
    test-release) step test-release env CARGO_TARGET_DIR="$TEST_RELEASE_TARGET" "$TEST_RUNNER" cargo test --locked --offline --release -p x0ne-infer --all-targets --features probe ;;
    # The client build of docs/cloud.md, in /target/release, which no other step writes:
    # x0ne-engine-proxy without the TEST key, as a user builds it.
    build-cloud) step build-cloud bash -c 'cargo build --locked --release -p x0ne-infer --bin x0ne-engine --bin x0ne-engine-proxy -p x0ne-engine-cloud --bin x0ne-engine-cloud --offline && "$CARGO_TARGET_DIR"/release/x0ne-engine cloud --help >/dev/null' ;;
    # The five binaries of the .deb, built as make deb builds them, then again after a comment
    # appended at the end of three files: the same sha256 each (ci/comment-invariance.sh).
    comment-invariance) step comment-invariance bash /src/ci/comment-invariance.sh ;;
    # x0ne-infer built with the release profile, run with no argument, refuses as a release build.
    release-refusal) step release-refusal release_refusal ;;
    # x0ne-core's workspace without a GPU or nvcc (ci/core-cpu.sh): the CPU reference of the
    # kernels, the tests of the copied runtime that need no device, the same under
    # AddressSanitizer; the GPU tests are compiled. make core-gpu runs them on a GPU.
    core)       step core env CARGO_TARGET_DIR=/target/core bash /src/ci/core-cpu.sh ;;
    doc)        step doc env RUSTDOCFLAGS="-D warnings" cargo doc --locked --offline --workspace --no-deps --features "$ALL_FEATURES" ;;
    deny)       step deny cargo deny --offline --locked --workspace check ;;
    gitleaks)   step gitleaks gitleaks_all ;;
    *)          extra_step "$1" || { echo "unknown step: $1" >&2; printf '%s\tfail\t0\t2\n' "$1" > "/out/steps.d/$1"; } ;;
  esac
}
# clippy_ci [cargo args...]: clippy with -D warnings and the PEDANTIC list of the commit's
# .github/workflows/ci.yml (clippy::pedantic minus the lints it allows), read by
# ci/clippy-pedantic.sh: one list for both CIs. A list it cannot read fails the step.
clippy_ci() {
  local out lints
  out=$(bash /src/ci/clippy-pedantic.sh /src/.github/workflows/ci.yml) || return 2
  mapfile -t lints <<< "$out"
  echo "clippy_ci: -D warnings ${lints[*]}"
  cargo clippy --locked --offline --workspace --all-targets "$@" -- -D warnings "${lints[@]}"
}
# release_refusal: ci/release-refusal.sh on x0ne-infer built with the release profile and its default
# features. No .deb ships it since 2026-11-04: the CVM image builds it with cuda nvml (x0ne-OS
# scripts/pkg-infer-cuda.sh), which needs nvcc; the release switch is the same code without them.
release_refusal() {
  cargo build --release --locked --offline -p x0ne-infer --bin x0ne-infer || return 2
  bash /src/ci/release-refusal.sh "$CARGO_TARGET_DIR/release/x0ne-infer"
}
gitleaks_all() {
  gitleaks dir /src --redact --no-banner --exit-code 1 || return 1
  # The merged range, as patches (ci/ci.sh writes it): commit messages and diffs.
  if [ -s /out/range.patch ]; then gitleaks stdin --redact --no-banner --exit-code 1 < /out/range.patch || return 1; fi
}

# lane <steps of the lane...>: the requested steps among them, in the requested order.
lane() {
  local s l
  for s in $STEPS; do for l in "$@"; do [ "$s" = "$l" ] && run_step "$s"; done; done
}
in_lanes() {
  local s l
  for s in $STEPS; do
    for l in $LANE_LIGHT $LANE_A $LANE_B $LANE_C $LANE_D $LANE_E; do [ "$s" = "$l" ] && continue 2; done
    return 1
  done
}

if [ "$LANES" = 1 ] && in_lanes; then
  lane $LANE_LIGHT & light=$!
  lane $LANE_A & a=$!
  CARGO_TARGET_DIR=/target/all lane $LANE_B & b=$!
  lane $LANE_C & c=$!
  lane $LANE_D & d=$!
  lane $LANE_E & e=$!
  wait "$light" "$a" "$b" "$c" "$d" "$e"
else
  for s in $STEPS; do run_step "$s"; done
fi

# steps.tsv in the requested order, whatever the lanes' timing.
rc=0
for s in $STEPS; do
  if [ -s "/out/steps.d/$s" ]; then cat "/out/steps.d/$s"; else printf '%s\tfail\t0\t2\n' "$s"; fi
done > /out/steps.tsv
grep -qv $'\tok\t' /out/steps.tsv && rc=1
[ -n "${RUSTC_WRAPPER:-}" ] && sccache --show-stats > /out/sccache.log 2>&1 || true
exit $rc
