#!/usr/bin/env bash
# The part of x0ne-core/ that needs no GPU, inside a container without network: step `core` of
# make ci (ci/inside.sh) and step `cpu` of make core-gpu (scripts/core-gpu.sh).
#
#   lock    every crate of x0ne-core/Cargo.lock is in the root Cargo.lock at the same version
#           (one vendor directory serves both workspaces)
#   fmt     x0ne-core/oracle, the crate written here (x0ne Core formats its own two)
#   clippy  x0ne-core/oracle, every target, -D warnings
#   test    the tests of x0ne-core/oracle and of the runtime that need no device; the GPU tests
#           are compiled and listed as ignored
#   asan    the same test binaries under AddressSanitizer
#
# Without nvcc the kernels crate is built on ci/nvcc-stub.sh: its cubins are placeholders, its
# own tests (which read real cubins) are left out, and no kernel can be loaded. With nvcc (NVCC,
# CUDA_HOME or PATH) the whole workspace is tested.
#
# Environment:
#   CORE_CPU_PARTS  default "lock fmt clippy test asan"
#   CORE_CPU_ASAN   auto (default: run when the toolchain ships the runtime, else say so and
#                   pass), 1 (its absence fails), 0 (never)
# fmt and clippy are skipped, and said so, in an image without them. Exit 0: every part green.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
parts=${CORE_CPU_PARTS:-lock fmt clippy test asan}
asan=${CORE_CPU_ASAN:-auto}
cd "$SRC/x0ne-core"

say() { echo "core-cpu: $*"; }

stub=$SRC/ci/nvcc-stub.sh
if [ "${NVCC:-}" != "$stub" ] && { [ -n "${NVCC:-}" ] || [ -n "${CUDA_HOME:-}" ] || command -v nvcc >/dev/null 2>&1; }; then
  packages=(--workspace)
  say "nvcc found: the whole workspace"
else
  export NVCC=$stub
  packages=(-p x0ne-core-oracle -p x0ne-core-runtime)
  say "no nvcc: kernels built on ci/nvcc-stub.sh, their own tests left out"
fi

# name and version of every registry package of a Cargo.lock, one per line.
locked() {
  awk '
    /^\[\[package\]\]/ { name = ""; version = ""; registry = 0 }
    /^name = /         { name = $3 }
    /^version = /      { version = $3 }
    /^source = /       { registry = 1 }
    /^$/               { if (registry) print name, version; registry = 0 }
    END                { if (registry) print name, version }
  ' "$1" | tr -d '"' | sort -u
}

check_lock() {
  local missing
  missing=$(comm -23 <(locked Cargo.lock) <(locked "$SRC/Cargo.lock"))
  if [ -n "$missing" ]; then
    say "x0ne-core/Cargo.lock names crates the root Cargo.lock does not (no vendor directory holds them):"
    printf '  %s\n' "${missing//$'\n'/$'\n'  }"
    return 1
  fi
  say "lock: $(locked Cargo.lock | wc -l) crates, all in the root Cargo.lock"
}

# The AddressSanitizer runtime of the toolchain in use, or nothing.
asan_runtime() {
  local sysroot
  sysroot=$(rustc --print sysroot) || return 0
  find "$sysroot/lib/rustlib/x86_64-unknown-linux-gnu/lib" -maxdepth 1 -name 'librustc*_rt.asan.a' 2>/dev/null | head -1
}

run_asan() {
  if [ "$asan" = 0 ]; then say "asan: off (CORE_CPU_ASAN=0)"; return 0; fi
  if [ -z "$(asan_runtime)" ]; then
    say "asan: this toolchain ships no AddressSanitizer runtime"
    [ "$asan" != 1 ]
    return
  fi
  # A stable toolchain takes -Z flags under RUSTC_BOOTSTRAP=1. The standard library is the
  # shipped one, not instrumented: an error is caught where this workspace's code makes it.
  # A target triple keeps the flags off build scripts and proc macros.
  RUSTC_BOOTSTRAP=1 RUSTFLAGS="-Zsanitizer=address" ASAN_OPTIONS=detect_leaks=1 \
    CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$SRC/x0ne-core/target}/asan" \
    cargo test --locked --offline --target x86_64-unknown-linux-gnu \
      -p x0ne-core-oracle -p x0ne-core-runtime --lib --tests
}

rc=0
for part in $parts; do
  t0=$SECONDS
  part_rc=0
  case $part in
    lock) check_lock || part_rc=$? ;;
    fmt)
      if cargo fmt --version >/dev/null 2>&1; then
        cargo fmt -p x0ne-core-oracle -- --check || part_rc=$?
      else
        say "fmt: no rustfmt in this image, skipped"
      fi
      ;;
    clippy)
      if cargo clippy --version >/dev/null 2>&1; then
        cargo clippy --locked --offline -p x0ne-core-oracle --all-targets -- -D warnings || part_rc=$?
      else
        say "clippy: no clippy in this image, skipped"
      fi
      ;;
    test) cargo test --locked --offline "${packages[@]}" || part_rc=$? ;;
    asan) run_asan || part_rc=$? ;;
    *) say "unknown part $part (lock, fmt, clippy, test, asan)"; exit 2 ;;
  esac
  say "$part rc=$part_rc in $((SECONDS - t0)) s"
  [ "$part_rc" = 0 ] || rc=1
done
exit "$rc"
