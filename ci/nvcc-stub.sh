#!/usr/bin/env bash
# Stands in for nvcc where there is none (the CI image): x0ne-core/kernels/build.rs runs
# `nvcc --version`, then `nvcc -cubin ... -o <file> cuda/x0ne_kernels.cu` once per architecture,
# and embeds the files. This writes a marker instead of each cubin, so that the Rust of x0ne Core
# compiles and its tests that need no GPU run (ci/core-cpu.sh). No kernel is compiled: a binary
# built with it loads nothing on a GPU, and the kernels crate's own tests, which read real
# cubins, are left to `make core-gpu`.
set -euo pipefail
if [ "${1:-}" = --version ]; then
  echo "nvcc stub of x0ne Engine's CI: release none, no kernel compiled"
  exit 0
fi
out=''
arch=''
while [ $# -gt 0 ]; do
  case $1 in
    -o) out=$2; shift 2 ;;
    --gpu-architecture=*) arch=${1#--gpu-architecture=}; shift ;;
    *) shift ;;
  esac
done
[ -n "$out" ] || { echo "nvcc stub: no -o" >&2; exit 2; }
printf 'x0ne Engine CI: not a cubin (%s), ci/nvcc-stub.sh\n' "$arch" > "$out"
