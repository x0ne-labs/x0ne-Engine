#!/usr/bin/env bash
# End-to-end run of the attested cloud channel on GCP, by hand: the sequence
# `x0ne-engine cloud up` automates (docs/cloud.md), written step by step so
# that each link is timed and every resource is named and deleted.
# First executed 2026-10-14 (x0ne-infer/RUN_2026-10-14_beta2.md).
#
#   scripts/e2e-cloud.sh bench              local, no cloud: x0ne-infer tests with the probe feature (fake TD)
#   scripts/e2e-cloud.sh gpu-units DIR      local, no cloud: x0ne-OS cvm units (tsm-report, nvidia-nodes, gpu-ready,
#                                            x0ne-infer sandbox) on the local GPU VM E2E_GPU_VM (gpu-units-local.sh)
#   scripts/e2e-cloud.sh preflight          read-only: GCP empty, price, cost cap
#   scripts/e2e-cloud.sh relay              pass 1: the product relay, in the x0ne OS client container
#   scripts/e2e-cloud.sh probe [--pin F]    pass 2: same VM rebooted with the probe's config (examples/cloud_probe.rs)
#   scripts/e2e-cloud.sh down               delete VM and firewall rule, prove the project is empty
#   scripts/e2e-cloud.sh image-delete       delete the imported image
#
# The VM bills from `relay` to `down`. Three guards, any one is enough:
# maxRunDuration with instanceTerminationAction=DELETE on the VM itself, a
# systemd --user timer on this machine that deletes it, and `down`.
# gcloud runs on this machine with the operator's account; the client
# container (x0ne OS, x0ne-engine from apt) holds no cloud credential.
#
# One grouped run, cost cap E2E_CAP_USD (10 $): bench and gpu-units
# green first (gpu-units replays VMs 4 to 8 of 2026-10-14 on a local GPU;
# only the CC ready state and the TDX quote stay for GCP), then preflight,
# relay, probe, down. The probe's phases run most information per
# VM second first, probe.json rewritten after each (examples/cloud_probe.rs):
#   1 attacks: connections without the relay, C3 burst of 3 x burst, nominal rate
#   2 quote time inside the TD (Server-Timing)
#   3 key, ready, one long completion then two at once: tokens/s, $ per million
#   4 rechecks of 2026-10-14: C1, one quote per connection, C2
# E2E_PROBE_BIN: `cargo build --release -p x0ne-infer --features probe --example cloud_probe`.
# E2E_MAX_QUEUE must be the TD's --max-queue (x0ne-infer.service; default 8):
# the probe derives the C3 bound from it. Tokens/s mean GPU only if the
# image serves on the GPU (the 2026-10-14 test unit ran --device cpu).
set -euo pipefail

if [ "${1:-}" = bench ]; then
  # The cases the probe plays, on the fake TD, through the CI container (lock, slice, sccache).
  cd "$(dirname "$0")/.." && CI_WORKTREE=1 CI_STEPS="test-all" exec ci/ci.sh run
fi
if [ "${1:-}" = gpu-units ]; then
  exec "$(dirname "$0")/gpu-units-local.sh" "${2:?gpu-units DIR: x0ne-OS cvm/mkosi.extra}"
fi
E2E=${E2E_DIR:?E2E_DIR: run directory (inputs and logs)}
ID=${E2E_ID:-$(date -u +%m%d%H%M)}
NAME=${E2E_NAME:-x0ne-engine-e2e-$ID}
ZONE=${E2E_ZONE:-us-central1-a}
IMAGE=${E2E_IMAGE:?E2E_IMAGE: the imported CVM image name}
MAX_RUN_MIN=${E2E_MAX_RUN_MIN:-75}
USD_PER_HOUR=${E2E_USD_PER_HOUR:?E2E_USD_PER_HOUR: read from the Cloud Billing catalog the same day}
CLIENT_IMG=${E2E_CLIENT_IMG:-localhost/x0ne-e2e-client:r1014}
REFERENCE=${E2E_REFERENCE:?E2E_REFERENCE: the image reference of the image}
PROBE_BIN=${E2E_PROBE_BIN:-}
MODEL=${E2E_MODEL:-qwen3:0.6b}
MAX_QUEUE=${E2E_MAX_QUEUE:-8}
CAP_USD=${E2E_CAP_USD:-10}
MACHINE=a3-highgpu-1g
LOG=$E2E/logs
mkdir -p "$LOG"
STATE=$E2E/state.env
[ -f "$STATE" ] && . "$STATE"

ts() { while IFS= read -r l; do printf '%s %s\n' "$(date -u +%T.%3N)" "$l"; done; }
note() { printf '%s %s\n' "$(date -u +%T.%3N)" "$*" | tee -a "$LOG/e2e.log" >&2; }
save() { printf '%s=%q\n' "$1" "$2" >> "$STATE"; }

inventory() {
  gcloud compute instances list --format="table(name,zone,status,machineType)"
  gcloud compute disks list --format="table(name,sizeGb,zone,users)"
  gcloud compute addresses list --format="table(name,status)"
}

preflight() {
  note "preflight: project $(gcloud config get-value project 2>/dev/null)"
  [ -z "$(gcloud compute instances list --format='value(name)')" ] || { note "instances present: refusing"; exit 2; }
  inventory
  gcloud compute images describe "$IMAGE" --format='value(name,status,guestOsFeatures)'
  local most; most=$(awk -v m="$MAX_RUN_MIN" -v p="$USD_PER_HOUR" 'BEGIN{printf "%.2f", m/60*p}')
  note "plan: $NAME $MACHINE spot TDX in $ZONE, max run $MAX_RUN_MIN min at $USD_PER_HOUR \$/h = at most $most \$ (cap $CAP_USD \$)"
  awk -v a="$most" -v c="$CAP_USD" 'BEGIN{exit !(a <= c)}' || { note "at most $most \$ is over the cap $CAP_USD \$: lower E2E_MAX_RUN_MIN"; exit 2; }
}

# The firewall rule admits tcp:443 from this machine's address only.
create_vm() {
  local config=$1 myip
  myip=$(curl -4 -fsS --max-time 10 https://ifconfig.me)
  note "firewall: tcp:443 to tag $NAME from this machine only"
  gcloud compute firewall-rules create "$NAME" --network=default --direction=INGRESS \
    --allow=tcp:443 --source-ranges="$myip/32" --target-tags="$NAME" \
    --description="x0ne-engine session $ID: tcp:443 to this session's VM only" --format=none
  save FW_CREATED 1
  note "create: $NAME (meter starts)"
  local t0; t0=$(date +%s); save CREATED_UNIX "$t0"
  gcloud compute instances create "$NAME" --zone="$ZONE" --machine-type="$MACHINE" \
    --confidential-compute-type=TDX --maintenance-policy=TERMINATE --no-restart-on-failure \
    --provisioning-model=SPOT --instance-termination-action=DELETE \
    --max-run-duration="${MAX_RUN_MIN}m" \
    --accelerator=type=nvidia-h100-80gb,count=1 \
    --image="$IMAGE" --boot-disk-auto-delete --network-interface=nic-type=GVNIC \
    --no-shielded-secure-boot --no-shielded-vtpm --no-shielded-integrity-monitoring \
    --no-service-account --no-scopes --tags="$NAME" \
    --labels=x0ne-engine=session,x0ne-session="$ID" \
    --metadata=block-project-ssh-keys=true --metadata-from-file=x0ne-config="$config" \
    --format=none 2>&1 | tee -a "$LOG/create.log"
  note "create returned after $(( $(date +%s) - t0 )) s"
  # Local guard: deletes the VM at the cap even if this script and the session die.
  systemd-run --user --unit="x0ne-e2e-guard-$ID" --on-active="$(( MAX_RUN_MIN * 60 + 120 ))" \
    /usr/bin/gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet >/dev/null 2>&1 \
    && note "guard: systemd --user timer x0ne-e2e-guard-$ID armed at +$((MAX_RUN_MIN + 2)) min"
  gcloud compute instances describe "$NAME" --zone="$ZONE" \
    --format='yaml(status,confidentialInstanceConfig,scheduling)' > "$LOG/instance.yaml"
}

# Checked before relay creates the VM; never in down.
sig_present() { [ -f "$REFERENCE.sig" ] || { note "$REFERENCE.sig missing: the client would refuse the reference (99)"; exit 2; }; }
vm_ip() { gcloud compute instances describe "$NAME" --zone="$ZONE" --format='value(networkInterfaces[0].accessConfigs[0].natIP)'; }

# Pass 1: x0ne-engine-proxy from the apt package, in the x0ne OS container.
relay() {
  sig_present
  local expires=$(( $(date +%s) + MAX_RUN_MIN * 60 ))
  local ctr="x0ne-e2e-relay-$ID"
  mkdir -p "$E2E/relay-state"
  coproc RELAY { podman run -i --rm --userns=keep-id --name "$ctr" -e XDG_RUNTIME_DIR=/tmp/run -e XDG_STATE_HOME=/tmp/state \
      -v "$E2E/relay-state":/tmp/state \
      -v "$REFERENCE":/ref/reference.json:ro -v "$REFERENCE.sig":/ref/reference.json.sig:ro "$CLIENT_IMG" \
      bash -c 'mkdir -m 0700 -p /tmp/run; exec /usr/bin/x0ne-engine-proxy --protocol 1 --provider gcp --machine-type '"$MACHINE"' --model '"$MODEL"' --expires-at-unix '"$expires"' --reference /ref/reference.json' \
      2> >(ts > "$LOG/relay.stderr.log"); }
  local line event
  IFS= read -r line <&"${RELAY[0]}"
  printf '%s %s\n' "$(date -u +%T.%3N)" "$line" >> "$LOG/relay.events.log"
  python3 -I -c 'import json,sys,base64; e=json.loads(sys.argv[1]); assert e["event"]=="config"; open(sys.argv[2],"wb").write(base64.b64decode(e["config_b64"]))' "$line" "$E2E/config-relay.json"
  note "relay: config emitted, sha256 $(sha256sum "$E2E/config-relay.json" | cut -c1-16)…"
  create_vm "$E2E/config-relay.json"
  local ip; ip=$(vm_ip)
  printf '{"v":1,"cmd":"connect","endpoint":"%s:443","boot_deadline_unix":%s}\n' "$ip" "$(( $(date +%s) + 900 ))" >&"${RELAY[1]}"
  note "relay: connect sent (endpoint withheld from the record)"
  while IFS= read -r line <&"${RELAY[0]}"; do
    printf '%s %s\n' "$(date -u +%T.%3N)" "$line" >> "$LOG/relay.events.log"
    event=$(python3 -I -c 'import json,sys; print(json.loads(sys.argv[1]).get("event",""))' "$line")
    note "relay event: $event"
    case $event in
      serving) break ;;
      refused|stopped) note "relay ended: $line"; wait "$RELAY_PID" || true; return 1 ;;
    esac
  done
  # One question through the local API, inside the same container.
  local t0; t0=$(date +%s.%N)
  podman exec "$ctr" bash -c 'curl -sS --max-time 300 -H "Authorization: Bearer $(cat /tmp/run/x0ne-engine/local.key)" \
      -H "Content-Type: application/json" http://127.0.0.1:8478/v1/chat/completions \
      -d "{\"model\":\"'"$MODEL"'\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one word.\"}],\"max_tokens\":48}"' \
      > "$LOG/relay.chat.json" || true
  note "chat through the relay: $(python3 -c "print(round($(date +%s.%N)-$t0,1))") s"
  # A second question: the relay reuses its checked connection.
  t0=$(date +%s.%N)
  podman exec "$ctr" bash -c 'curl -sS --max-time 300 -H "Authorization: Bearer $(cat /tmp/run/x0ne-engine/local.key)" \
      -H "Content-Type: application/json" http://127.0.0.1:8478/v1/chat/completions \
      -d "{\"model\":\"'"$MODEL"'\",\"messages\":[{\"role\":\"user\",\"content\":\"Name one prime number.\"}],\"max_tokens\":24}"' \
      > "$LOG/relay.chat2.json" || true
  note "second chat through the relay: $(python3 -c "print(round($(date +%s.%N)-$t0,1))") s"
  printf '{"v":1,"cmd":"stop"}\n' >&"${RELAY[1]}"
  while IFS= read -r line <&"${RELAY[0]}"; do printf '%s %s\n' "$(date -u +%T.%3N)" "$line" >> "$LOG/relay.events.log"; done
  # bash unsets NAME_PID once the coprocess has exited.
  wait "${RELAY_PID:-}" 2>/dev/null || true
  note "relay stopped"
}

# Pass 2: same VM, new config (metadata), hard reset, the probe as the client.
probe() {
  sig_present
  local pin=() ctr="x0ne-e2e-probe-$ID"
  [ "${1:-}" = --pin ] && pin=(--measurements /ref/measurements.json) && local pinfile=$2
  [ -x "$PROBE_BIN" ] || { note "E2E_PROBE_BIN missing"; exit 2; }
  mkdir -p "$E2E/probe"
  coproc PROBE { podman run -i --rm --userns=keep-id --name "$ctr" -v "$REFERENCE":/ref/reference.json:ro \
      -v "$REFERENCE.sig":/ref/reference.json.sig:ro \
      ${pinfile:+-v "$pinfile":/ref/measurements.json:ro} -v "$PROBE_BIN":/usr/local/bin/cloud_probe:ro \
      -v "$E2E/probe":/out "$CLIENT_IMG" /usr/local/bin/cloud_probe --out /out --reference /ref/reference.json \
      --model "$MODEL" --machine-type "$MACHINE" --ttl-secs "$(( MAX_RUN_MIN * 60 ))" "${pin[@]}" \
      --max-queue "$MAX_QUEUE" --usd-per-hour "$USD_PER_HOUR" \
      2> >(tee "$LOG/probe.stderr.log" >&2); }
  local ppid=$PROBE_PID line; IFS= read -r line <&"${PROBE[0]}"
  note "probe: $line"
  gcloud compute instances add-metadata "$NAME" --zone="$ZONE" --metadata-from-file=x0ne-config="$E2E/probe/config.json" --format=none
  local t0; t0=$(date +%s)
  gcloud compute instances reset "$NAME" --zone="$ZONE" --format=none
  note "probe: VM reset with the probe's config ($(( $(date +%s) - t0 )) s)"
  # Not in a pipeline: bash closes coprocess descriptors in subshells.
  local ip; ip=$(vm_ip); printf '%s:443\n' "$ip" >&"${PROBE[1]}"
  wait "$ppid" && note "probe: exit 0" || note "probe: exit $?"
}

serial() { gcloud compute instances get-serial-port-output "$NAME" --zone="$ZONE" 2>/dev/null | grep -a 'x0ne-cvm-report' | grep -v -E 'natIP|listen_tcp' > "$LOG/serial-report-${1:-now}.txt" || true; }

down() {
  local t0; t0=$(date +%s)
  serial final
  gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet 2>&1 | tail -1 || true
  for _ in $(seq 1 30); do
    gcloud compute instances describe "$NAME" --zone="$ZONE" >/dev/null 2>&1 || break; sleep 5
  done
  gcloud compute firewall-rules delete "$NAME" --quiet 2>&1 | tail -1 || true
  systemctl --user stop "x0ne-e2e-guard-$ID.timer" 2>/dev/null || true
  local end; end=$(date +%s)
  note "down: VM and rule deleted in $(( end - t0 )) s"
  [ -n "${CREATED_UNIX:-}" ] && note "billed life ≈ $(( end - CREATED_UNIX )) s ≈ $(awk -v s=$(( end - CREATED_UNIX )) -v p="$USD_PER_HOUR" 'BEGIN{printf "%.2f", s/3600*p}') \$ at $USD_PER_HOUR \$/h"
  inventory | tee "$LOG/inventory-final.txt"
  gcloud compute firewall-rules list --format='value(name)' | tee -a "$LOG/inventory-final.txt"
}

image_delete() { gcloud compute images delete "$IMAGE" --quiet; }

case ${1:-} in
  preflight) preflight ;;
  relay) relay ;;
  probe) shift; probe "$@" ;;
  serial) serial "${2:-now}" ;;
  down) down ;;
  image-delete) image_delete ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
