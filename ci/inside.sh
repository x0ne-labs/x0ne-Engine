#!/usr/bin/env bash
# Runs inside the CI container (ci/ci.sh), with no network. Mounts: /src (the
# commit, read-only), /vendor (crates), /advisory-db (RustSec, read-only),
# /target, /out (step logs and steps.tsv). Exit 0 = every step green.
#
# The steps run in four lanes at once (CI_LANES=0: one after the other):
#   light  fmt ci-sh deny gitleaks        no cargo build
#   a      clippy test test-doc           default features, /target/debug
#   b      clippy-all doc test-all        every feature, /target/all/debug
#   c      test-release                   /target/release (the profile that ships)
#          build-cloud                    /target/release
#          comment-invariance             /target/release (the .deb binaries, built twice)
#          release-refusal                x0ne-infer, release profile, default features, run
# No two lanes write one profile directory: a test that runs the real binary
# (CARGO_BIN_EXE_*) would otherwise run the one another lane just built with other
# features (lab_option.rs went red that way). The build of one lane overlaps the tests
# of the others; CPU is shared, so a step's seconds move while the wall clock drops.
set -uo pipefail
ALL_FEATURES=${CI_ALL_FEATURES:-x0ne-infer/nvml,x0ne-infer/probe}
STEPS=${CI_STEPS:-fmt ci-sh features clippy clippy-all test test-doc test-all test-release build-cloud comment-invariance release-refusal doc deny gitleaks}
LANES=${CI_LANES:-1}
LANE_LIGHT="fmt ci-sh features deny gitleaks"
LANE_A="clippy test test-doc"
LANE_B="clippy-all doc test-all"
LANE_C="test-release build-cloud comment-invariance release-refusal"

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
    # Not --all: it also loads x0ne-core's own workspace, whose runtime's dev-dependency (x0ne
    # Core's CPU oracle) is not copied. x0ne-core/ is formatted in x0ne Core (PROVENANCE.md).
    fmt)        step fmt cargo fmt -- --check ;;
    ci-sh)      step ci-sh bash -c "bash /src/ci/tests/ci-sh.sh && bash /src/ci/tests/release-verify-tag.sh && bash /src/ci/tests/coverage.sh" ;;
    # The clippy of the GitHub `clippy` job: -D warnings, then its PEDANTIC list (clippy_ci).
    features)   step features bash -c "python3 -I /src/ci/features.py --check && bash /src/ci/tests/features.sh" ;;
    clippy)     step clippy clippy_ci ;;
    clippy-all) step clippy-all clippy_ci --features "$ALL_FEATURES" ;;
    test)       step test cargo test --locked --offline --workspace --all-targets ;;
    test-doc)   step test-doc cargo test --locked --offline --workspace --doc ;;
    test-all)   step test-all cargo test --locked --offline -p x0ne-infer --all-targets --features "$ALL_FEATURES" ;;
    # Tests gated on not(debug_assertions), and the cloud_probe of e2e-cloud.sh, in the profile
    # that ships (release: Cargo.toml, line tables kept for ci/strip-release.sh).
    test-release) step test-release cargo test --locked --offline --release -p x0ne-infer --all-targets --features probe ;;
    # The client build of docs/cloud.md, in /target/release, which no other step writes:
    # x0ne-engine-proxy without the TEST key, as a user builds it.
    build-cloud) step build-cloud bash -c 'cargo build --locked --release -p x0ne-infer --bin x0ne-engine --bin x0ne-engine-proxy -p x0ne-engine-cloud --bin x0ne-engine-cloud --offline && "$CARGO_TARGET_DIR"/release/x0ne-engine cloud --help >/dev/null' ;;
    # The five binaries of the .deb, built as make deb builds them, then again after a comment
    # appended at the end of three files: the same sha256 each (ci/comment-invariance.sh).
    comment-invariance) step comment-invariance bash /src/ci/comment-invariance.sh ;;
    # x0ne-infer built with the release profile, run with no argument, refuses as a release build.
    release-refusal) step release-refusal release_refusal ;;
    doc)        step doc env RUSTDOCFLAGS="-D warnings" cargo doc --locked --offline --workspace --no-deps --features "$ALL_FEATURES" ;;
    deny)       step deny cargo deny --offline --locked --workspace check ;;
    gitleaks)   step gitleaks gitleaks_all ;;
    *)          echo "unknown step: $1" >&2; printf '%s\tfail\t0\t2\n' "$1" > "/out/steps.d/$1" ;;
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
    for l in $LANE_LIGHT $LANE_A $LANE_B $LANE_C; do [ "$s" = "$l" ] && continue 2; done
    return 1
  done
}

if [ "$LANES" = 1 ] && in_lanes; then
  lane $LANE_LIGHT & light=$!
  lane $LANE_A & a=$!
  CARGO_TARGET_DIR=/target/all lane $LANE_B & b=$!
  lane $LANE_C & c=$!
  wait "$light" "$a" "$b" "$c"
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
