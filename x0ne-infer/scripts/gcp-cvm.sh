#!/usr/bin/env bash
# Provision a confidential VM on GCP and run x0ne-infer on it, for real.
#
# The recipe for a full cryptographic verdict (Intel DCAP to the pinned
# root + live PCS TCB + NVIDIA NRAS) and a signed receipt; its run of
# 2026-08-20 took about ten minutes of billed VM.
#
# Phases are SEPARATE on purpose: you invoke them one at a time and read
# what each one says. There is no "do everything" mode, because every phase
# past `create` costs money and the one that matters most is `stop`.
#
#   ./gcp-cvm.sh create      # a3-highgpu-1g SPOT, TDX + 1×H100 — it BILLS while it runs; the one estimate is in docs/verification.md
#                            # X0NE_INFER_MAX_RUN (default 120m): the VM stops ITSELF at that deadline
#   ./gcp-cvm.sh wait-ssh
#   ./gcp-cvm.sh driver      # NVIDIA 580 open + LKCA modprobe rule
#   ./gcp-cvm.sh reboot      # proven by boot_id, LKCA needs a new initramfs
#   ./gcp-cvm.sh checks      # TDX guest? configfs-tsm? GPU CC status ON?
#   ./gcp-cvm.sh arm         # declare the CC GPU ready, so CUDA may init
#   ./gcp-cvm.sh build       # compile x0ne-infer ON the VM (glibc)
#                            # X0NE_INFER_CUDA=1 to build the CUDA backend
#   ./gcp-cvm.sh model       # fetch a small open-weights model
#   ./gcp-cvm.sh measure     # DEV ONLY, trust on first use: RTMR1/RTMR2 of a
#                            # fresh quote, printed as X0NE_INFER_EXPECT
#   ./gcp-cvm.sh run         # attest, serve, one completion, capture proofs
#                            # X0NE_INFER_EXPECT="rtmr1=<hex> rtmr2=<hex>" required
#   ./gcp-cvm.sh console     # serve + ssh tunnel: the chat UI in your browser
#   ./gcp-cvm.sh fetch       # bring attestation.json / completion.json back
#   ./gcp-cvm.sh stop        # TERMINATED, local SSD discarded
#   ./gcp-cvm.sh delete      # and gone
#
# COST: the meter starts at `create` and stops at `stop`. If anything goes
# wrong, run `stop` — it does not depend on any other phase having worked.
#
# PINS: the VM is Ubuntu, not an x0ne-cvm image. The measurements a release
# x0ne-infer pins by default are not its own, and with none given it exits
# 87 (measured on 2026-10-24, tests/gcp_cvm_recipe.rs). `run` and `console`
# pin what X0NE_INFER_EXPECT names. `measure` prints what this VM reports:
# trust on first use, for this development recipe on Ubuntu only. The
# product path, `x0ne-engine cloud up`, pins a signed release reference.
# X0NE_INFER_PORT (default 8477): the server's port on the VM's loopback.
set -uo pipefail

NAME=${X0NE_INFER_VM:-x0ne-infer-cvm}
ZONE=${X0NE_INFER_ZONE:-us-central1-a}
MODEL_REPO=${X0NE_INFER_MODEL:-Qwen/Qwen3-0.6B}
OUT=${X0NE_INFER_OUT:-./cvm-artifacts}
SSH="gcloud compute ssh $NAME --zone=$ZONE --command"
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PORT=${X0NE_INFER_PORT:-8477}

# The `--expect-measurement` flags X0NE_INFER_EXPECT names, or exit 2 before
# ssh: a release x0ne-infer with nothing pinned refuses to serve (87).
expect_flags() {
  local spec flags='' specs=()
  [[ $PORT =~ ^[0-9]{1,5}$ ]] || { echo "X0NE_INFER_PORT: not a port number" >&2; return 2; }
  read -ra specs <<< "${X0NE_INFER_EXPECT:-}"
  for spec in "${specs[@]}"; do
    [[ $spec =~ ^(mrtd|rtmr[0-3])=[0-9a-fA-F]{96}$ ]] || {
      echo "X0NE_INFER_EXPECT: '${spec:0:12}…' is not <mrtd|rtmr0..3>=<96 hex digits>" >&2
      return 2
    }
    flags+=" --expect-measurement $spec"
  done
  [ -n "$flags" ] || {
    echo "X0NE_INFER_EXPECT is empty: a release x0ne-infer would refuse to serve (87)." >&2
    echo "Development recipe: '$0 measure' prints what the VM reports (trust on first use)." >&2
    return 2
  }
  printf '%s' "$flags"
}

# The deadline is the one thing that does not depend on the operator: at
# X0NE_INFER_MAX_RUN the VM is STOPPED by GCP (SPOT needs the explicit action
# and, on an a3 with local SSD, the discard flag — `gcloud … create --help`).
create() {
  gcloud compute instances create "$NAME" \
    --zone="$ZONE" \
    --machine-type=a3-highgpu-1g \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --confidential-compute-type=TDX \
    --maintenance-policy=TERMINATE \
    --shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring \
    --accelerator=type=nvidia-h100-80gb,count=1 \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --max-run-duration="${X0NE_INFER_MAX_RUN:-120m}" \
    --discard-local-ssds-at-termination-timestamp=true
}

wait-ssh() {
  for i in $(seq 1 30); do
    if $SSH "true" -- -o ConnectTimeout=10 >/dev/null 2>&1; then echo "ssh OK ($i)"; return 0; fi
    sleep 10
  done
  echo "ssh never answered" >&2; return 1
}

# The confidential GPU needs the LKCA crypto modules present in the initramfs
# BEFORE the driver loads, or it comes up without its secure channel. `-k all`
# because the driver packages can pull a newer kernel than the running one.
#
# The persistenced switch to uvm-persistence-mode is NOT optional and NOT
# cosmetic: dropping it (which this script did on 2026-08-21) left a machine
# where `nvidia-smi` saw the H100 for a minute and then lost it entirely, and
# `conf-compute -f` answered "No devices were found" — measured, and the only
# difference from the recipe that worked on an earlier hardware run.
driver() {
  $SSH "sudo sh -c 'set -e; \
     export DEBIAN_FRONTEND=noninteractive; \
     apt-get update -q; \
     apt-get install -y -q linux-headers-\$(uname -r) build-essential pkg-config; \
     apt-get install -y -q nvidia-driver-580-server-open nvidia-utils-580-server \
       linux-modules-nvidia-580-server-open-gcp; \
     echo \"install nvidia /sbin/modprobe ecdsa_generic; /sbin/modprobe ecdh; \
/sbin/modprobe --ignore-install nvidia\" > /etc/modprobe.d/nvidia-lkca.conf; \
     update-initramfs -u -k all; \
     if test -f /usr/lib/systemd/system/nvidia-persistenced.service; then \
       sed -i s/no-persistence-mode/uvm-persistence-mode/g \
         /usr/lib/systemd/system/nvidia-persistenced.service; \
       systemctl daemon-reload; \
     fi'"
}

# Detached, so the ssh command we are inside exits 0 first; then PROVE the
# machine went down and came back rather than assuming it.
reboot() {
  local before after
  before=$($SSH "cat /proc/sys/kernel/random/boot_id" 2>/dev/null | tr -d '\r')
  [ -n "$before" ] || { echo "boot_id unreadable" >&2; return 1; }
  echo "boot_id before: $before"
  $SSH "sudo sh -c 'nohup sh -c \"sleep 2; reboot\" >/dev/null 2>&1 &'"
  sleep 20
  for _ in $(seq 1 30); do
    after=$($SSH "cat /proc/sys/kernel/random/boot_id" 2>/dev/null | tr -d '\r')
    if [ -n "$after" ] && [ "$after" != "$before" ]; then echo "reboot proven ($after)"; return 0; fi
    sleep 10
  done
  echo "no new boot_id — the machine may not have rebooted" >&2; return 1
}

checks() {
  $SSH "grep -q tdx_guest /proc/cpuinfo && echo TDX_OK; \
        test -d /sys/kernel/config/tsm/report && echo TSM_OK; \
        nvidia-smi --query-gpu=name --format=csv,noheader; \
        sudo nvidia-smi conf-compute -f"
}

# A confidential GPU refuses CUDA initialisation until it is declared ready.
# This is the step that makes `--device cuda` possible, and it comes AFTER
# attestation on purpose: arming a GPU we have not verified would be arming
# someone else's.
arm() {
  $SSH "sudo nvidia-smi conf-compute -srs 1 && sudo nvidia-smi conf-compute -grs"
}

# Compiled ON the VM: a binary built against a newer host glibc will not run
# on the CVM's 2.35, and static linking is a separate task.
#
# X0NE_INFER_CUDA=1 also installs the CUDA toolkit and builds the `cuda`
# feature — several minutes and ~3 GB more, which is why it is opt-in.
build() {
  # Extracted OVER the previous tree, never after removing it: `rm -rf ~/build`
  # threw away target/ — ten minutes of CUDA compilation to pay again on a
  # billing GPU — and failed anyway on a non-empty directory (2026-08-21).
  # Modules are declared in lib.rs, so a stale orphan file is not compiled.
  #
  # The whole workspace goes, tables included: a minimal `[workspace]
  # members = [...]` cost a build on a billing GPU (2026-08-21) because the
  # crates inherit edition/version/deps from the root manifest.
  local tar rc
  tar=$(mktemp "${TMPDIR:-/tmp}/x0ne-infer-src.XXXXXXXX") || return 1
  tar czf "$tar" --exclude=./target --exclude=./.git -C "$REPO_ROOT" .
  gcloud compute scp "$tar" "$NAME":~/src.tar.gz --zone="$ZONE"
  rc=$?
  rm -f "$tar"
  [ "$rc" -eq 0 ] || return "$rc"
  local feats="nvml"
  local cuda_env=""
  if [ "${X0NE_INFER_CUDA:-0}" = "1" ]; then
    feats="nvml,cuda"
    # NOT Ubuntu's `nvidia-cuda-toolkit`: 22.04 ships CUDA 11.5, and an H100
    # is compute capability 9.0, which needs 11.8+. Installing the distro
    # package here costs ten minutes and then fails to generate code for the
    # only GPU this script provisions. NVIDIA's own repo it is.
    $SSH "set -e; cd /tmp; \
      test -f cuda-keyring_1.1-1_all.deb || curl -fsSLO \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb; \
      sudo dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null; \
      sudo DEBIAN_FRONTEND=noninteractive apt-get update -q; \
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q cuda-toolkit-12-6; \
      /usr/local/cuda/bin/nvcc --version | tail -2"
    # CUDA_COMPUTE_CAP (90, the H100) was read by candle's build; x0ne Core's
    # build ignores it and compiles fixed architectures (x0ne-core/kernels/build.rs).
    cuda_env="export PATH=/usr/local/cuda/bin:\$PATH CUDA_COMPUTE_CAP=90;"
  fi
  $SSH "set -e; \
    command -v cargo >/dev/null || (curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal -q); \
    . ~/.cargo/env; $cuda_env \
    mkdir -p ~/build && cd ~/build && tar xzf ~/src.tar.gz; \
    cargo build --release --locked -p x0ne-infer --features $feats > ~/cargo.log 2>&1 \
      || { tail -20 ~/cargo.log; exit 1; }; \
    cp target/release/x0ne-infer ~/x0ne-infer && ls -la ~/x0ne-infer"
}

model() {
  local base="https://huggingface.co/${MODEL_REPO}/resolve/main"
  $SSH "mkdir -p ~/model && cd ~/model && \
        curl -fsSL -o config.json       $base/config.json && \
        curl -fsSL -o tokenizer.json    $base/tokenizer.json && \
        curl -fsSL -o model.safetensors $base/model.safetensors && ls -la"
}

# DEVELOPMENT RECIPE ONLY, trust on first use: what the VM's TDX module
# reports now, from a quote read through configfs-tsm (root), RTMR1 (kernel,
# command line) and RTMR2 (initrd). It names what this VM runs, not that it
# is what you meant to run; `x0ne-engine cloud up` never pins this way.
measure() {
  echo "gcp-cvm.sh measure: development recipe only, trust on first use: these are the values this VM reports, not a reference (docs/running.md)." >&2
  $SSH "set -e; \
    test -d /sys/kernel/config/tsm/report || { echo 'no configfs-tsm report directory: not a TDX guest, or a kernel without it' >&2; exit 64; }; \
    d=/sys/kernel/config/tsm/report/x0ne-measure-\$\$; \
    sudo mkdir \$d; \
    head -c64 /dev/urandom | sudo tee \$d/inblob >/dev/null; \
    q=\$(sudo base64 -w0 \$d/outblob); \
    sudo rmdir \$d; \
    printf '{\"evidence\":{\"tdx_quote_b64\":\"%s\"}}' \"\$q\" > ~/quote.json; \
    ~/x0ne-infer measurements ~/quote.json rtmr1 rtmr2 > ~/measured.json; \
    cat ~/measured.json; \
    printf 'export X0NE_INFER_EXPECT=\"%s\"\n' \"\$(sed -n 's/^ *\"\(rtmr[0-3]\)\": \"\([0-9a-f]*\)\".*/\1=\2/p' ~/measured.json | tr '\n' ' ' | sed 's/ \$//')\""
}

# The data key is generated ON the VM and never leaves it. In production this
# is a client-held KMS; the v0 broker reads X0NE_INFER_KEY_HEX, and the
# release DECISION path (proof, freshness, lease) is the real one either way.
run() {
  local expect
  expect=$(expect_flags) || return 2
  # The ready state does not survive a stop/start, and this script is used
  # across reboots — so arm before serving, exactly as the chat client does.
  # Measured 2026-08-21: without it, CUDA_ERROR_SYSTEM_NOT_READY, and the
  # server's own message names this command.
  if [ "${X0NE_INFER_DEVICE:-cpu}" != cpu ]; then
    $SSH "sudo nvidia-smi conf-compute -srs 1 >/dev/null 2>&1 || true"
  fi
  # The server is a job of this shell, not of a subshell: when it stops
  # before serving its attestation, its exit code is this phase's, at once.
  $SSH "set -e; \
    export X0NE_INFER_KEY_HEX=\$(head -c32 /dev/urandom | od -An -tx1 | tr -d ' \n'); \
    rm -f ~/attestation.json ~/completion.json; \
    sudo -E ~/x0ne-infer --model-dir ~/model --device ${X0NE_INFER_DEVICE:-cpu} \
       --listen 127.0.0.1:$PORT$expect > ~/infer.log 2>&1 & pid=\$!; echo \$pid > ~/infer.pid; \
    for _ in \$(seq 1 60); do \
      curl -fsS http://127.0.0.1:$PORT/attestation -o ~/attestation.json 2>/dev/null && break; \
      jobs -rp | grep -qx \$pid || break; \
      sleep 2; \
    done; \
    if ! test -s ~/attestation.json; then \
      if jobs -rp | grep -qx \$pid; then sudo kill \$pid 2>/dev/null || true; code=65; why='no attestation within 120 s'; \
      else code=0; wait \$pid || code=\$?; why=\"x0ne-infer exited \$code\"; fi; \
      echo \"=== infer.log (\$why) ===\"; cat ~/infer.log; \
      [ \$code -ne 0 ] || code=65; exit \$code; \
    fi; \
    curl -fsS -X POST http://127.0.0.1:$PORT/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{\"model\":\"\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello from the enclave in one short sentence.\"}],\"max_tokens\":32}' \
      -o ~/completion.json; \
    sudo kill \$(cat ~/infer.pid) 2>/dev/null || true; \
    echo '=== attestation ==='; cat ~/attestation.json; echo; \
    echo '=== completion ==='; cat ~/completion.json"
}

# The console, in YOUR browser, over an ssh tunnel — the server listens on
# the VM's loopback and nothing is ever exposed publicly. Ctrl+C ends the
# tunnel AND the server; the VM is left running, so `stop` is on you.
console() {
  local expect
  expect=$(expect_flags) || return 2
  echo "starting the enclave and tunnelling $PORT → your machine…"
  echo "open http://127.0.0.1:$PORT/  (Ctrl+C here when done, then: $0 stop)"
  gcloud compute ssh "$NAME" --zone="$ZONE" -- -L "$PORT:127.0.0.1:$PORT" -t "
    export X0NE_INFER_KEY_HEX=\$(head -c32 /dev/urandom | od -An -tx1 | tr -d ' \n')
    if [ \"${X0NE_INFER_DEVICE:-cpu}\" != cpu ]; then sudo nvidia-smi conf-compute -srs 1 >/dev/null 2>&1 || true; fi
    sudo -E ~/x0ne-infer --model-dir ~/model --device ${X0NE_INFER_DEVICE:-cpu} --listen 127.0.0.1:$PORT$expect
  "
}

fetch() {
  mkdir -p "$OUT"
  for f in attestation.json completion.json infer.log; do
    gcloud compute scp "$NAME":~/"$f" "$OUT/$f" --zone="$ZONE" || true
  done
  ls -la "$OUT"
}

# --discard-local-ssd is not a tuning knob: an a3 refuses to stop without it
# (measured 2026-08-14, HTTPError 400).
stop() {
  gcloud compute instances stop "$NAME" --zone="$ZONE" --discard-local-ssd=true
  gcloud compute instances list
}

delete() {
  gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet
  echo "── remaining billable resources ──"
  gcloud compute instances list
  gcloud compute disks list
  gcloud compute addresses list
}

case "${1:-}" in
  create|wait-ssh|driver|reboot|checks|arm|build|model|measure|run|console|fetch|stop|delete) "$1" ;;
  *) sed -n '2,40p' "$0"; exit 2 ;;
esac
