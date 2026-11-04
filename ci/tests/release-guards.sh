#!/usr/bin/env bash
# Tests of ci/release-env.sh (the build environment refused before cargo) and ci/release-refusal.sh
# (the shipped x0ne-infer must refuse each case of its table), on small programs built here (cc) and
# scripts that answer as x0ne-infer does. No container, no network. Run by ci/tests/ci-sh.sh, or by
# hand. Needs cc; without it the release-refusal cases skip, unless X0NE_REQUIRE_CI_TOOLS is set
# (ci/inside.sh does), then they fail.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
FAILS=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s: %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }

# ---- release-env.sh: env -i, then only the variables given.
# env_case <name> <want exit> <names stderr must give, space-separated|-> <VAR=value>...
env_case() {
  local name=$1 want=$2 msg=$3 rc=0 n missing=; shift 3
  env -i PATH=/usr/bin:/bin "$@" bash "$SRC/ci/release-env.sh" > "$T/env.out" 2> "$T/env.err" || rc=$?
  local err; err=$(tr '\n' ' ' < "$T/env.err")
  if [ "$msg" != - ]; then for n in $msg; do grep -qF -- " $n" "$T/env.err" || missing="$missing $n"; done; fi
  if [ "$rc" != "$want" ]; then fail "$name" "exit $rc, want $want: $err"
  elif [ -n "$missing" ]; then fail "$name" "no$missing in: $err"
  elif grep -q 'SECRETVALUE' "$T/env.err" "$T/env.out"; then fail "$name" "prints a value: $err"
  else pass "$name (exit $rc)"; fi
}
env_case env-clean 0 - CARGO_HOME=/tmp/c CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0 CARGO_NET_OFFLINE=true SOURCE_DATE_EPOCH=1 RUSTC_WRAPPER=/bin/true
env_case env-debug-assertions 2 CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=SECRETVALUE
env_case env-dev-profile 2 CARGO_PROFILE_DEV_DEBUG CARGO_PROFILE_DEV_DEBUG=SECRETVALUE
env_case env-rustflags 2 RUSTFLAGS RUSTFLAGS=SECRETVALUE
env_case env-encoded 2 CARGO_ENCODED_RUSTFLAGS CARGO_ENCODED_RUSTFLAGS=SECRETVALUE
env_case env-build-rustflags 2 CARGO_BUILD_RUSTFLAGS CARGO_BUILD_RUSTFLAGS=SECRETVALUE
env_case env-target-rustflags 2 CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS=SECRETVALUE
env_case env-dash-in-name 2 CARGO_PROFILE_RELEASE_PACKAGE_x0ne-infer_DEBUG_ASSERTIONS 'CARGO_PROFILE_RELEASE_PACKAGE_x0ne-infer_DEBUG_ASSERTIONS=SECRETVALUE'
env_case env-two-named 2 'RUSTFLAGS CARGO_PROFILE_RELEASE_OPT_LEVEL' RUSTFLAGS=SECRETVALUE CARGO_PROFILE_RELEASE_OPT_LEVEL=SECRETVALUE
env_case env-newline-in-value 0 - "X0NE_NOTE=a
RUSTFLAGS=b"

# ---- release-refusal.sh, on programs that imitate x0ne-infer.
if ! command -v cc >/dev/null; then
  if [ -n "${X0NE_REQUIRE_CI_TOOLS:-}" ]; then echo "FAIL  release-guards: cc not found, X0NE_REQUIRE_CI_TOOLS is set"; exit 1; fi
  echo "skip  release-refusal: cc not found"
else
  # prog <name> <exit> <stderr text> [stdout text]: a binary that writes and exits.
  prog() {
    printf '#include <stdio.h>\nint main(void){fputs("%s",stderr);fputs("%s",stdout);return %s;}\n' "$3" "${4:-}" "$2" > "$T/$1.c"
    cc -O1 -o "$T/$1" "$T/$1.c"
  }
  REL='x0ne-infer: measurements/0.2.0-beta.7.json (built in) was measured against a reference signed by test-archive-key, not by the release archive key: refused by a release build. Pin with --expect-measurement\\n'
  # infer <name> [case]: a script that decides as a release x0ne-infer (main.rs, harden.rs), in
  # its order; the case named, when one is, goes on to "no evidence" (64) instead of its refusal.
  infer() {
    cat > "$T/$1" <<'FAKE'
#!/bin/bash
WRONG=@WRONG@
s() { [ "$WRONG" = "$1" ] && { echo "x0ne-infer: no evidence, not serving" >&2; exit 64; }; echo "x0ne-infer: $3" >&2; exit "$2"; }
CPU="refused. x0ne-infer serves only on the attested GPU: the CPU serves nothing, and nothing falls back to it"
for v in LD_PRELOAD LD_AUDIT GLIBC_TUNABLES LD_LIBRARY_PATH CUDA_INJECTION64_PATH NVTX_INJECTION64_PATH; do
  [ -z "${!v+x}" ] || s "$v" 73 "process hardening: $v set: the process would load code or settings the image did not measure"
done
[ $# = 0 ] && s none 87 "measurements/0.2.0-beta.7.json (built in): refused by a release build"
dev= att= cfg= pin=
while [ $# -gt 0 ]; do
  case $1 in
    --device) dev=$2 ;; --attest) att=$2 ;; --cloud-config) cfg=$2 ;; --expect-measurement) pin=$2 ;;
    *) s unknown 2 "unknown argument \"$1\" (see --help)" ;;
  esac
  shift 2 || s value 2 "$1: a value is missing (see --help)"
done
case $att in '' | cpu+gpu) ;; cpu) s attest-cpu 90 "--attest cpu: $CPU" ;; *) s attest 82 "unknown --attest value: expected cpu+gpu (the value given is not printed)" ;; esac
case $dev in
  '' | cuda | cuda:0) ;; cpu) s device-cpu 90 "--device cpu: $CPU" ;;
  cuda:[0-9]*) s cuda-n 81 "--device: expected cuda or cuda:0, x0ne Core runs on GPU 0 only (the value given is not printed)" ;;
  *) s device 81 "unknown --device: expected cuda or cuda:0 (the value given is not printed)" ;;
esac
if [ -n "$cfg" ]; then
  [ -z "${X0NE_INFER_KEY_HEX+x}" ] || s key 95 "cloud mode takes the session key from the attested channel: unset X0NE_INFER_KEY_HEX"
  [ -z "$pin" ] || s pin 95 "the client pins the image in cloud mode (its reference): drop --expect-measurement and --measurements"
  grep -q '"attest":"cpu"' "$cfg" && s config-cpu 90 "a cloud config that attests the CPU only: $CPU"
fi
s none 64 "no evidence, not serving"
FAKE
    sed -i "s/@WRONG@/${2:-}/" "$T/$1"
    chmod 0755 "$T/$1"
  }
  infer release
  for w in LD_LIBRARY_PATH CUDA_INJECTION64_PATH NVTX_INJECTION64_PATH LD_PRELOAD device-cpu attest-cpu config-cpu cuda-n attest unknown key pin; do
    infer "takes-$w" "$w"
  done
  prog debug 64 'x0ne-infer: WARNING: ... allowed because this is a debug build\\nx0ne-infer: no evidence, not serving\\n'
  prog unpinned 87 'x0ne-infer: no reference measurement is published for version 0.2.0\\n'
  prog says-but-0 0 "$REL"
  prog on-stdout 87 '' "$REL"
  printf '#include <unistd.h>\nint main(void){sleep(100);return 87;}\n' > "$T/hang.c"; cc -o "$T/hang" "$T/hang.c"
  printf '#include <stdio.h>\n#include <stdlib.h>\nint main(void){if(getenv("X0NE_INFER_KEY_HEX")){fputs("refused by a release build\\n",stderr);return 87;}return 3;}\n' > "$T/env.c"
  cc -o "$T/envprobe" "$T/env.c"
  # ref_case <name> <want exit> <binary>...
  ref_case() {
    local name=$1 want=$2 rc=0; shift 2
    bash "$SRC/ci/release-refusal.sh" "$@" > "$T/ref.out" 2>&1 || rc=$?
    if [ "$rc" = "$want" ]; then pass "$name (exit $rc)"; else fail "$name" "exit $rc, want $want: $(tr '\n' ' ' < "$T/ref.out")"; fi
  }
  ref_case refusal-release 0 "$T/release"
  # A binary that takes one case of the table, each in turn (the 73 of LD_LIBRARY_PATH: 0.2.0-beta.7
  # until 2026-11-04; --device cuda:1, accepted then refused with 79 after the attestation).
  for w in LD_LIBRARY_PATH CUDA_INJECTION64_PATH NVTX_INJECTION64_PATH LD_PRELOAD device-cpu attest-cpu config-cpu cuda-n attest unknown key pin; do
    ref_case "refusal-takes-$w" 1 "$T/takes-$w"
  done
  ref_case refusal-debug-assertions 1 "$T/debug"
  ref_case refusal-other-87 1 "$T/unpinned"
  ref_case refusal-message-exit-0 1 "$T/says-but-0"
  ref_case refusal-on-stdout 1 "$T/on-stdout"
  ref_case refusal-one-of-two 1 "$T/release" "$T/debug"
  ref_case refusal-two 0 "$T/release" "$T/release"
  # The caller's environment does not reach the binary.
  X0NE_INFER_KEY_HEX=00 ref_case refusal-empty-env 1 "$T/envprobe"
  ref_case refusal-missing 2 "$T/nonexistent"
  ref_case refusal-no-argument 2
  chmod 0644 "$T/debug"; ref_case refusal-not-executable 2 "$T/debug"
  t0=$SECONDS; RELEASE_REFUSAL_TIMEOUT=2 ref_case refusal-hang 1 "$T/hang"
  [ $((SECONDS - t0)) -le 10 ] && pass "refusal-hang stops at the timeout ($((SECONDS - t0)) s)" || fail refusal-hang "took $((SECONDS - t0)) s"
fi

[ "$FAILS" = 0 ] || { echo "release-guards: $FAILS failed"; exit 1; }
echo "release-guards: all passed"
