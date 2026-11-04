#!/usr/bin/env bash
# The release switches of a shipped x0ne-infer, tested on the bytes shipped. Each case runs the binary
# and requires an exit code and a text on stderr:
#   87  no argument: a release build refuses the built-in measurements file, of TEST provenance until
#       the signing ceremony (x0ne-infer/src/measurements.rs resolve_from, "refused by a release
#       build"); a build with debug assertions (CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true) accepts it;
#   90  the CPU asked to serve: --device cpu, --attest cpu, a cloud config that attests the CPU only,
#       --attest cpu or --device cpu with a GPU config, --device cpu with a pin (main.rs cpu_serving);
#   73  one variable, empty, of x0ne-infer/src/harden.rs: LD_PRELOAD, LD_AUDIT, GLIBC_TUNABLES, and
#       in a release build LD_LIBRARY_PATH, CUDA_INJECTION64_PATH, NVTX_INJECTION64_PATH;
#   81  --device cuda:1 (x0ne Core runs on GPU 0 only) and --device cpu:0; 82 --attest gpu and CPU;
#   2   the flags of the lab server, gone from this version; 95 a session key or a pin in cloud mode.
# Each is decided before any evidence: the table holds with or without a TD, a GPU, cuda or nvml, for
# the default build of make ci and the cuda nvml build of the CVM image alike.
#
#   ci/release-refusal.sh <x0ne-infer>...
#
# Each run: an empty directory, an empty environment but the case's variable, stdin /dev/null, at most
# 30 s (RELEASE_REFUSAL_TIMEOUT, for the tests); a binary that times out is not run again. One line per
# case. Exit 0 when every case of every binary holds, 1 otherwise, 2 on usage. Run by make ci (step
# release-refusal) on x0ne-infer built with the release profile, and by x0ne-OS
# scripts/pkg-infer-cuda.sh on the x0ne-infer of the CVM image.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: ci/release-refusal.sh <x0ne-infer>..." >&2; exit 2; }
for b in "$@"; do [ -f "$b" ] && [ -x "$b" ] || { echo "release-refusal: $b: not an executable file" >&2; exit 2; }; done
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
mkdir "$d/run"
# Two cloud configs that parse (x0ne-infer/src/cloud_config.rs), one per attest value; never served.
cfg() { printf '{"schema":"x0ne-engine/cloud-config/v1","model":"qwen3:0.6b","client_spki_sha256":"%s","client_nonce":"%s","expires_at_unix":4102444800,"attest":"%s","listen_port":443,"max_completion_tokens":64}\n' \
  "$(printf 'ab%.0s' $(seq 32))" "$(printf 'cd%.0s' $(seq 32))" "$1" > "$d/$2"; }
cfg cpu cpu.json
cfg cpu+gpu gpu.json
# <exit>|<text stderr must hold>|<VAR=value, or ->|<arguments>; @cpu @gpu: the configs, @pin: a
# 96-hex rtmr1, @key: a 64-hex key, all TEST values.
CASES='87|refused by a release build|-|
90|--device cpu: refused|-|--device cpu
90|--attest cpu: refused|-|--attest cpu
90|a cloud config that attests the CPU only: refused|-|--cloud-config @cpu
90|--attest cpu: refused|-|--cloud-config @gpu --attest cpu
90|--device cpu: refused|-|--cloud-config @gpu --device cpu
90|--device cpu: refused|-|--expect-measurement rtmr1=@pin --device cpu
73|process hardening: LD_PRELOAD|LD_PRELOAD=|
73|process hardening: LD_AUDIT|LD_AUDIT=|
73|process hardening: GLIBC_TUNABLES|GLIBC_TUNABLES=|
73|process hardening: LD_LIBRARY_PATH|LD_LIBRARY_PATH=|
73|process hardening: CUDA_INJECTION64_PATH|CUDA_INJECTION64_PATH=|
73|process hardening: NVTX_INJECTION64_PATH|NVTX_INJECTION64_PATH=|
81|expected cuda or cuda:0|-|--device cuda:1
81|expected cuda or cuda:0|-|--device cpu:0
82|unknown --attest value|-|--attest gpu
82|unknown --attest value|-|--attest CPU
2|unknown argument|-|--lab
2|unknown argument|-|--not-confidential
2|unknown argument|-|--decide
95|unset X0NE_INFER_KEY_HEX|X0NE_INFER_KEY_HEX=@key|--cloud-config @gpu
95|the client pins the image in cloud mode|-|--cloud-config @gpu --expect-measurement rtmr1=@pin'
pin=$(printf 'ab%.0s' $(seq 48)); key=$(printf '11%.0s' $(seq 32))
sub() { local w=$1; w=${w//@cpu/$d/cpu.json}; w=${w//@gpu/$d/gpu.json}; w=${w//@pin/$pin}; printf '%s' "${w//@key/$key}"; }
rc=0
for b in "$@"; do
  abs=$(cd "$(dirname "$b")" && pwd)/$(basename "$b")
  sha=$(sha256sum < "$b" | cut -c1-64)
  n=0 bad=0
  while IFS='|' read -r want text var args; do
    n=$((n + 1))
    envs=(PATH=/usr/bin:/bin); [ "$var" = - ] || envs+=("$(sub "$var")")
    read -r -a argv <<< "$(sub "$args")"
    case="${var/#-/}${var:+ }$args"; case=${case# }
    r=0
    (cd "$d/run" && exec env -i "${envs[@]}" timeout "${RELEASE_REFUSAL_TIMEOUT:-30}" "$abs" "${argv[@]}" < /dev/null > "$d/out" 2> "$d/err") || r=$?
    if [ "$r" = "$want" ] && grep -qF -- "$text" "$d/err"; then
      printf 'refused  %-3s %s  [%s]\n' "$r" "$b" "${case:-no argument}"
    else
      bad=$((bad + 1))
      printf 'NOT REFUSED  %s  %s (exit %s, want %s "%s") [%s]: %s\n' "$sha" "$b" "$r" "$want" "$text" "${case:-no argument}" "$(head -c 200 "$d/err" | tr '\n' ' ')"
      [ "$r" != 124 ] || { echo "release-refusal: $b timed out, its other cases are not run"; break; }
    fi
  done <<< "$CASES"
  if [ "$bad" = 0 ]; then printf 'refused  %s  %s (%s cases)\n' "$sha" "$b" "$n"; else rc=1; fi
done
exit "$rc"
