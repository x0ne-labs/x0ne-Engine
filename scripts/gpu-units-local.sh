#!/usr/bin/env bash
# The CVM image's GPU units, replayed on a local GPU before any paid VM.
# On 2026-10-14, VMs 4 to 8 were paid for device nodes, the unit's device
# cgroup and DeviceAllow (x0ne-infer/RUN_2026-10-14_beta2.md). Those fail the
# same way on any NVIDIA GPU; only the CC ready state and the TDX quote need GCP.
#
#   scripts/gpu-units-local.sh DIR     DIR: x0ne OS cvm/mkosi.extra (units of 2026-10-14 or later)
#
# Units: x0ne-tsm-report (must skip off a TD), x0ne-nvidia-nodes (as is, then its
# sandbox around a plain mknod of the four nodes into an empty RuntimeDirectory,
# with and without its DeviceAllow: the 2026-10-14 runs did this on /dev, which
# TemporaryFileSystem= had left as the host's, nodes already there, mknod never
# called), x0ne-gpu-ready (helper built from DIR/../gpu-ready.c), and x0ne-infer's
# [Service] sandbox copied as is (DynamicUser stands in for the image's sysusers
# entry) around `nvidia-smi -q -d COMPUTE`: NVML init in the unit's device cgroup. No chat:
# x0ne-infer refuses outside a TD (exit 87 or 64), and the lab binary that chatted in this
# sandbox left on 2026-11-04.
# One sudo on the VM (E2E_GPU_VM, its libvirt domain, required), run with a tty so it may ask
# a password. Units in /run/systemd/system, files in /run/x0ne-unit-test and
# /var/lib/x0ne-unit-test: removed at the end. Aborts under 3 GB available RAM or
# above 85 C. Never starts the VM. Log: E2E_GPU_LOG (default ./gpu-units-<date>.log).
# E2E_GPU_DRY=1: stage, build, offline scores, no sudo. Exit 0 GREEN. gpu-ready and the flipped MemoryDenyWriteExecute are reported, not judged.
set -euo pipefail

src=${1:?usage: gpu-units-local.sh DIR (x0ne-OS cvm/mkosi.extra)}
vm=${E2E_GPU_VM:-}
[ -n "$vm" ] || { echo "gpu-units: set E2E_GPU_VM to the libvirt domain of the local GPU VM" >&2; exit 2; }
cc_image=${E2E_GPU_CC_IMAGE:-docker.io/library/rust:1.98.1-bookworm}
st=$(virsh -c qemu:///system domstate "$vm" 2>/dev/null | head -1 || true)
st=${st:-absent}
[ "$st" = running ] || { echo "gpu-units: VM $vm is $st. Start it first: virsh -c qemu:///system start $vm" >&2; exit 2; }

sys=$src/usr/lib/systemd/system
helper=$src/../gpu-ready.c
for f in "$sys/x0ne-tsm-report.service" "$sys/x0ne-nvidia-nodes.service" "$sys/x0ne-gpu-ready.service" "$sys/x0ne-infer.service" "$helper"; do
  [ -f "$f" ] || { echo "gpu-units: missing $f" >&2; exit 2; }
done

tmp=$(mktemp -d)
remote=/tmp/x0ne-unit-test-$$
cleanup() {
  rm -rf "$tmp"
  ssh -n "$vm" "rm -rf $remote" 2>/dev/null || true
}
trap cleanup EXIT

cp "$sys/x0ne-tsm-report.service" "$sys/x0ne-nvidia-nodes.service" "$tmp/"
# /var/lib, not /run: "Unable to locate executable '/run/x0ne-unit-test/gpu-ready': Permission denied" (2026-10-14).
sed 's#^ExecStart=/usr/lib/x0ne/gpu-ready#ExecStart=/var/lib/x0ne-unit-test/gpu-ready#' "$sys/x0ne-gpu-ready.service" > "$tmp/x0ne-gpu-ready.service"
grep -q '^ExecStart=/var/lib/x0ne-unit-test/gpu-ready$' "$tmp/x0ne-gpu-ready.service" || { echo "gpu-units: gpu-ready ExecStart not rewritten" >&2; exit 2; }

# The [Service] of a unit, with its own [Unit] header. $3: awk lines added after [Service].
service_copy() { # name description extra-lines source
  awk -v desc="$2" -v extra="$3" '
    /^\[Unit\]/ { print; print "Description=TEST: " desc; skip = 1; next }
    /^\[Service\]/ { skip = 0; print; if (extra != "") print extra; next }
    skip { next }
    /^\[Install\]/ { exit }
    { print }
  ' "$4" > "$tmp/$1.service"
}

# mknod of the four nodes in the unit's sandbox, into an empty RuntimeDirectory: the device
# cgroup checks mknod wherever the node is made (vfs_mknod -> devcgroup_inode_mknod). DeviceAllow
# "m" must let it through (fresh) and DevicePolicy=closed alone must refuse it (denied); the
# unit never opens the GPU in either case.
cat > "$tmp/nodes-probe.sh" <<'PROBE'
D=$1
for spec in nvidiactl:nvidia:255 nvidia0:nvidia:0 nvidia-uvm:nvidia-uvm:0 nvidia-uvm-tools:nvidia-uvm:1; do
  name=${spec%%:*}; rest=${spec#*:}; drv=${rest%%:*}; minor=${rest#*:}
  major=$(sed -n "s/^ *\([0-9][0-9]*\) $drv\$/\1/p" /proc/devices | head -1)
  if [ -z "$major" ]; then echo "mknod $name: no $drv in /proc/devices"; continue; fi
  if err=$(mknod "$D/$name" c "$major" "$minor" 2>&1); then echo "mknod $name $major:$minor ok"; else echo "mknod $name $major:$minor refused: $err"; fi
done
ls -l "$D" | grep '^c' || echo "no node"
if (exec 3<"$D/nvidiactl") 2>/dev/null; then echo "nvidiactl opened"; else echo "nvidiactl refused"; fi
PROBE
for v in fresh denied; do
  service_copy "x0ne-unit-test-nodes-$v" "x0ne-nvidia-nodes sandbox, mknod in an empty directory ($v)" \
    "RuntimeDirectory=x0ne-unit-test-nodes-$v" "$sys/x0ne-nvidia-nodes.service"
  sed -i -e "s#^ExecStart=.*#ExecStart=/bin/sh /run/x0ne-unit-test/nodes-probe.sh /run/x0ne-unit-test-nodes-$v#" "$tmp/x0ne-unit-test-nodes-$v.service"
done
sed -i '/^DeviceAllow=/d' "$tmp/x0ne-unit-test-nodes-denied.service"

# x0ne-infer's sandbox. Dropped: ReadWritePaths=/run/x0ne/models (absent here), Restart*.
infer_copy() { # name description exec-line type extra-lines
  service_copy "$1" "$2" "DynamicUser=yes${5:+
$5}" "$sys/x0ne-infer.service"
  sed -i -e "s#^ExecStart=.*#$3#" -e '/^ReadWritePaths=\/run\/x0ne\/models$/d' -e '/^Restart/d' -e "s#^Type=.*#Type=$4#" "$tmp/$1.service"
  grep -q "^Type=$4\$" "$tmp/$1.service" || { echo "gpu-units: $1 lost its Type" >&2; exit 2; }
  grep -q '^DevicePolicy=closed' "$tmp/$1.service" || { echo "gpu-units: $1 lost DevicePolicy" >&2; exit 2; }
  return 0
}
infer_copy x0ne-unit-test-infer-nvml "x0ne-infer sandbox, NVML init" \
  "ExecStart=/usr/bin/nvidia-smi -q -d COMPUTE" oneshot
cp "$helper" "$tmp/"

ssh -n "$vm" "mkdir -p $remote"
scp -q "$tmp"/* "$vm:$remote/"
# gpu-ready, built as the image builds it, in a throwaway container on the VM.
ssh -n "$vm" "podman run --rm --network=none -v $remote:/w:Z $cc_image gcc -O2 -Wall -Wextra -Werror -o /w/gpu-ready /w/gpu-ready.c"

cat > "$tmp/on-vm.sh" <<'ON_VM'
set -u
R=$1
guard() {
  local avail temp
  avail=$(free -g | awk '/^Mem:/ {print $7}')
  temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
  echo "guard: ${avail} GB available, GPU ${temp} C"
  if [ "$avail" -lt 3 ] || [ "$temp" -gt 85 ]; then echo "guard: ABORT"; teardown; exit 3; fi
}
units="x0ne-tsm-report x0ne-nvidia-nodes x0ne-unit-test-nodes-fresh x0ne-unit-test-nodes-denied x0ne-gpu-ready x0ne-unit-test-infer-nvml"
teardown() {
  for u in $units; do
    systemctl stop "$u.service" 2>/dev/null
    systemctl reset-failed "$u.service" 2>/dev/null
    rm -f "/run/systemd/system/$u.service"
  done
  rm -rf /run/x0ne-unit-test /var/lib/x0ne-unit-test "$R"
  systemctl daemon-reload
}
journal() { journalctl -b -u "$1.service" --no-pager -o cat | tail -n "${2:-12}" | sed "s/^/  $1| /"; }
show() { local p; for p in ConditionResult ActiveState Result ExecMainStatus; do printf '%s=%s ' "$p" "$(systemctl show -p "$p" --value "$1.service")"; done; }
# A unit of the same name in /etc or /usr would win over /run, or be the one tested: refuse.
for u in $units; do
  if systemctl cat "$u.service" >/dev/null 2>&1; then echo "unit $u already on this VM: not touched"; exit 2; fi
done
install -d /run/x0ne-unit-test /var/lib/x0ne-unit-test
install -m 0755 "$R/gpu-ready" /var/lib/x0ne-unit-test/
install -m 0644 "$R/nodes-probe.sh" /run/x0ne-unit-test/
for u in $units; do [ -f "$R/$u.service" ] && install -m 0644 "$R/$u.service" /run/systemd/system/; done
systemctl daemon-reload
echo "driver: $(head -1 /proc/driver/nvidia/version 2>/dev/null)"
echo "vram: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
echo "configfs-tsm: $([ -d /sys/kernel/config/tsm ] && echo present || echo absent)"
guard
rc=0
judge() { echo "check $1: $2"; [ "$2" = ok ] || rc=1; }

systemctl start x0ne-tsm-report.service
s=$(show x0ne-tsm-report); echo "x0ne-tsm-report: $s"
case "$s" in "ConditionResult=no ActiveState=inactive Result=success "*) judge tsm-report-skips ok ;; *) judge tsm-report-skips FAILED ;; esac
[ -e /sys/kernel/config/tsm/report/x0ne-infer ] && judge tsm-report-no-trace FAILED

if systemctl start x0ne-nvidia-nodes.service; then judge nvidia-nodes ok; else judge nvidia-nodes FAILED; fi
journal x0ne-nvidia-nodes 5
for v in fresh denied; do
  systemctl start "x0ne-unit-test-nodes-$v.service"
  journal "x0ne-unit-test-nodes-$v" 8
done
last() { journalctl -b _SYSTEMD_INVOCATION_ID="$(systemctl show -p InvocationID --value "$1.service")" -o cat; }
n=$(last x0ne-unit-test-nodes-fresh | grep -c '^mknod .* ok$')
[ "$n" = 4 ] && judge mknod-allowed-under-DeviceAllow-m ok || judge "mknod-allowed-under-DeviceAllow-m ($n of 4)" FAILED
n=$(last x0ne-unit-test-nodes-denied | grep -c '^mknod .* refused: ')
[ "$n" = 4 ] && judge mknod-refused-without-DeviceAllow ok || judge "mknod-refused-without-DeviceAllow ($n of 4)" FAILED
# The unit grants mknod only (DeviceAllow=... m): it must never open the GPU, with or without it.
last x0ne-unit-test-nodes-fresh | grep -q '^nvidiactl refused$' \
  && judge nodes-open-refused-under-DeviceAllow-m ok || judge nodes-open-refused-under-DeviceAllow-m FAILED
last x0ne-unit-test-nodes-denied | grep -q '^nvidiactl refused$' \
  && judge nodes-open-refused-without-DeviceAllow ok || judge nodes-open-refused-without-DeviceAllow FAILED

if systemctl start x0ne-gpu-ready.service; then echo "x0ne-gpu-ready: ok"; else echo "x0ne-gpu-ready: FAILED (reported, not judged: no CC ready state on this GPU)"; fi
journal x0ne-gpu-ready 5

if systemctl start x0ne-unit-test-infer-nvml.service; then judge infer-sandbox-nvml ok; else judge infer-sandbox-nvml FAILED; fi
journal x0ne-unit-test-infer-nvml 30 | grep -E 'Driver Version|CUDA Version|Compute Mode|Attached GPUs|rror' || true
guard

echo "systemd-analyze security (units as installed):"
for u in x0ne-tsm-report x0ne-nvidia-nodes x0ne-gpu-ready x0ne-unit-test-infer-nvml; do
  echo "  $u: $(systemd-analyze security --no-pager "$u.service" 2>&1 | tail -1 | sed 's/.*: //')"
done
teardown
echo "left: $(ls /run/systemd/system | grep -c -E 'x0ne-(tsm|nvidia|gpu-ready|unit-test)') units, $(ls -d /run/x0ne-unit-test /var/lib/x0ne-unit-test 2>/dev/null | wc -l) dirs"
echo "verdict: $([ $rc = 0 ] && echo GREEN || echo RED)"
exit $rc
ON_VM
scp -q "$tmp/on-vm.sh" "$vm:$remote/"

log=${E2E_GPU_LOG:-gpu-units-$(date -u +%m%d%H%M).log}
if [ "${E2E_GPU_DRY:-}" = 1 ]; then
  # No sudo: the staged units' offline security scores, then everything removed.
  ssh -n "$vm" "cd $remote && for u in *.service; do echo \"\$u: \$(systemd-analyze security --offline=yes --no-pager ./\$u 2>&1 | tail -1 | sed 's/.*: //')\"; done; ls -l gpu-ready 2>&1 | head -5" | tee "$log"
  exit 0
fi
set +e
ssh -t "$vm" "sudo bash $remote/on-vm.sh $remote" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
echo "gpu-units: log $log, exit $rc"
exit "$rc"
