#!/usr/bin/env bash
# Runs inside the coverage container (ci/coverage.sh run), with no network. Mounts: /src (the
# tree under test, an overlay thrown away at exit), /vendor (crates, read-only), /target, /out. Builds the tests of the
# workspace with -C instrument-coverage and COVERAGE_FEATURES, runs them, merges their profiles
# (llvm-profdata) and exports a summary per file (llvm-cov export --summary-only) to
# /out/llvm-cov.json. Also /out/members.tsv (package, directory), objects.txt, status.env, logs.
# Doctests are not measured: keeping their binaries needs rustdoc --persist-doctests, which
# 1.98.1 refuses outside nightly (measured 2026-10-30).
set -uo pipefail
FEATURES=${COVERAGE_FEATURES:?COVERAGE_FEATURES unset}
export HOME=/tmp/home CARGO_HOME=/tmp/cargo-home CARGO_TARGET_DIR=/target
export XDG_CONFIG_HOME=$HOME/.config XDG_DATA_HOME=$HOME/.local/share XDG_CACHE_HOME=$HOME/.cache
# The tests make ci may not skip (ci/inside.sh) may not skip here either.
export X0NE_REQUIRE_NON_ROOT=1 X0NE_REQUIRE_JQ=1 X0NE_REQUIRE_CI_TOOLS=1
mkdir -p "$HOME" "$CARGO_HOME" /out
cp /vendor/config.toml "$CARGO_HOME/config.toml"
cd /src
# -C instrument-coverage on the workspace members only, through cargo's RUSTC_WORKSPACE_WRAPPER:
# with RUSTFLAGS every dependency was instrumented too, and the x0ne-infer lib tests ran 325 of
# 1109 in 23 min against 95 s for all of them in make ci (2026-10-30). Not counted anyway.
# bash, not sh: dash drops the variables whose name has a hyphen, CARGO_BIN_EXE_x0ne-engine-cloud.
printf '#!/bin/bash\nexec "$@" -C instrument-coverage\n' > /tmp/rustc-coverage
chmod +x /tmp/rustc-coverage
export RUSTC_WORKSPACE_WRAPPER=/tmp/rustc-coverage
COV=/target/coverage
rm -rf "$COV"; mkdir -p "$COV/build" "$COV/run"
args=(--locked --offline --workspace --all-targets --features "$FEATURES")
: > /out/status.env
st() { printf '%s=%s\n' "$1" "$2" >> /out/status.env; }
st toolchain "$(rustc -V)"
st llvm "$(llvm-cov --version | sed -n 's/^ *LLVM version //p')"
cargo metadata --locked --offline --no-deps --format-version 1 2> /out/metadata.log \
  | jq -r '.workspace_members as $w | .packages[] | select(.id as $i | $w | index($i))
      | [.name, (.manifest_path | ltrimstr("/src/") | rtrimstr("/Cargo.toml"))] | @tsv' > /out/members.tsv

# Build first: the profiles of build scripts and proc-macros, run by cargo and rustc, land
# apart from those of the tests. The JSON lists every executable to read coverage from.
t=$SECONDS
LLVM_PROFILE_FILE="$COV/build/%p-%m.profraw" cargo test "${args[@]}" --no-run \
  --message-format=json-render-diagnostics > /out/build.jsonl 2> /out/build.log
st build_exit $?; st build_s $((SECONDS - t))
# Test binaries and the bins tests start (CARGO_BIN_EXE_*); not examples, not build scripts.
jq -r 'select(.reason == "compiler-artifact" and .executable != null
    and (.target.kind | index("example") | not)) | .executable' /out/build.jsonl | sort -u > /out/objects.txt
st objects "$(wc -l < /out/objects.txt)"

# %8m: per binary, at most 8 files merged online, whatever the number of processes.
t=$SECONDS
LLVM_PROFILE_FILE="$COV/run/%8m.profraw" cargo test "${args[@]}" --no-fail-fast > /out/test.log 2>&1
st test_exit $?; st test_s $((SECONDS - t))

# A process started with env_clear() loses LLVM_PROFILE_FILE and writes default_*.profraw in its
# working directory (/src is a throwaway overlay, ci/coverage.sh): those left under /src and /tmp
# are merged too; one written in a directory a test removed is lost.
t=$SECONDS
{ find "$COV/run" -name '*.profraw'; find /src /tmp -name 'default_*.profraw' 2>/dev/null; } > /out/profraw.txt
st profraw "$(wc -l < /out/profraw.txt)"
st stray_profraw "$(grep -vc "^$COV/run/" /out/profraw.txt)"
llvm-profdata merge -sparse -f /out/profraw.txt -o "$COV/coverage.profdata" > /out/merge.log 2>&1
st merge_exit $?; st merge_s $((SECONDS - t))

t=$SECONDS
mapfile -t objs < /out/objects.txt
more=()
for o in "${objs[@]:1}"; do more+=(-object "$o"); done
llvm-cov export --summary-only --instr-profile="$COV/coverage.profdata" \
  --ignore-filename-regex='^/(vendor|rustc|target|tmp|usr)/' "${objs[0]:-/nonexistent}" "${more[@]}" \
  > /out/llvm-cov.json 2> /out/export.log
st export_exit $?; st export_s $((SECONDS - t))
