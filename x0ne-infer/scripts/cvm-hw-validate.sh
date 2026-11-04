#!/usr/bin/env bash
# Hardware validation of a confidential VM — run ON the CVM (GCP a3-highgpu-1g,
# TDX + H100 CC, the shape `x0ne cc` boots). Every step prints PASS/FAIL and
# the script exits non-zero on the first failure. Nothing here is simulated:
# a step that cannot run on this machine fails, it does not pretend.
#
# Cost note: the CVM this runs on bills (the one estimate is in docs/verification.md — three different hourly figures used to live in three files). The run itself takes
# minutes; stop the VM after (x0ne cc does).
set -uo pipefail
fail() { echo "FAIL: $1" >&2; exit 1; }

echo "== 1. TDX guest?"
grep -q tdx_guest /proc/cpuinfo || fail "no tdx_guest in cpuinfo"
echo PASS

echo "== 2. configfs-tsm present?"
[ -d /sys/kernel/config/tsm/report ] || fail "no /sys/kernel/config/tsm/report (kernel < 6.7 or not mounted)"
echo PASS

echo "== 3. GPU CC on?"
nvidia-smi conf-compute -f 2>/dev/null | grep -qi "CC status: ON" || fail "GPU CC is not ON"
echo PASS

echo "== 4. x0ne-infer runs the full gate (expects: attested or a NAMED refusal)"
./x0ne-infer; code=$?
echo "x0ne-infer exit: $code"
case $code in
  0) echo "PASS (attested and would serve)";;
  64) fail "no evidence collected — configfs path broken on this machine";;
  65|66|67|68) echo "gate refused with a named reason (see stderr) — expected until NRAS collection lands";;
  *) fail "unexpected exit $code";;
esac

echo "== 5. NRAS reachability (JWKS)"
curl -fsS https://nras.attestation.nvidia.com/.well-known/jwks.json -o /dev/null || fail "JWKS unreachable"
echo PASS

echo "done — keep this transcript with the run's entry in docs/hardware-runs.md."
