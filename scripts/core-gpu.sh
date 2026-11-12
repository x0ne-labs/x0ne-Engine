#!/usr/bin/env bash
# make core-gpu: x0ne Core inside x0ne-infer (`--features cuda`), built and tested on the local
# NVIDIA GPU. make ci builds without CUDA (Core's kernels need nvcc, its tests a GPU): run this
# before merging a change of x0ne-core/ or of the code that serves through it.
#
#   build      x0ne-infer and its lib test binary, release, --features cuda (the tests add
#              candle-reference); then x0ne-core's own workspace: the GPU tests of
#              x0ne-core/oracle and the core-perf example, compiled with the real nvcc
#   cpu        ci/core-cpu.sh in this image: the tests of x0ne-core that need no GPU, the kernels
#              crate's own included (real cubins), and the same under AddressSanitizer
#   test       core_backend's tests, the GPU ones included: the conformance suite on Qwen3-0.6B and
#              Qwen3-8B, generate() equal to Core's own greedy decoding on the five bench prompts,
#              the device's greedy path equal to the host's. Without Qwen3-8B or the prompts, the
#              GPU tests that name them (#[ignore = "GPU: ..."] in core_backend/tests.rs) are not
#              run and the step is skipped, reason "partial: ..." with their names       (GPU)
#   reference  x0ne-core/oracle's GPU tests: every kernel and the engine against the CPU
#              reference, determinism, the routes that must agree, sessions              (GPU)
#   sanitize   compute-sanitizer memcheck and initcheck over the Qwen3-0.6B suite        (GPU)
#   perf       the core-perf example on Qwen3-0.6B (and Qwen3-8B when given), held to
#              ci/core-perf-baseline.json by ci/core-perf.sh                            (GPU)
#
# The verdict is written to out/core-gpu/<sha>.json (<sha>+dirty.json when the tree is not its
# commit's), as make ci writes out/ci/<sha>.json: status, tree, image, GPU, and per step its
# status, seconds, exit code and reason. A step is ok, fail or skipped. The steps marked (GPU)
# are skipped, reason "no GPU", when the image sees none (nvidia-smi, with the podman flags
# given); perf is skipped, reason "no baseline", when ci/core-perf-baseline.json holds no run
# of this model on this GPU. A skip is not a pass: the status is then "skipped" and the exit
# code 3. ci/core-gpu-check.sh is the gate that reads the file.
#
# CORE_GPU_RUNNER: where the GPU is not the host's to give to a container (a card held by
# vfio-pci for a microVM), the steps marked (GPU) go through a launcher that runs ONE program on
# the card and hands back its stdout, stderr and exit code: x0ne OS's
# scripts/microvm-gpu-run.sh, or a script with its options (-f HOST:GUEST read-only input,
# -w DIR, -e NAME=VALUE, -m MIB, -T SECONDS, -o RESULTS, -g program of the guest's image) and its
# codes (the program's; 125 the run failed; 199 the card no longer answers). Then:
#   - build and cpu run in containers as above, without the device flag;
#   - the GPU is the one the guest's /usr/bin/nvidia-smi names (-g). A runner that refuses or
#     sees no card: the GPU steps are skipped, reason "no GPU: runner exit N: its last line".
#     A run with no GPU step does not call the runner;
#   - no cargo in the guest: each step runs the binaries the build step left, found by the
#     path cargo prints for them (--no-run --message-format=json, in a container). Two paths
#     for one binary, or none, fail the step. x0ne-infer/ and x0ne-core/ are bound read-only
#     under /src, the models under /data, as in the containers; CARGO_MANIFEST_DIR and the
#     working directory are the binary's package, as under cargo;
#   - sanitize is skipped, reason given, unless CORE_GPU_RUNNER_SANITIZER is set: the guest's
#     image has no compute-sanitizer;
#   - perf reads core-perf's JSON from the runner's stdout;
#   - 125 fails the step. 199 stops everything: the step is failed, reason "card lost: ...",
#     every step after it is skipped, reason "card lost: not run", nothing is run again, and
#     the verdict is red with the reason in "stopped". The host then needs a reboot.
# The verdict names the runner (runner.path, runner.sha256; null without one).
#
# Environment:
#   CORE_GPU_IMAGE        a CUDA devel image with nvcc and a Rust toolchain on PATH (required)
#   CORE_GPU_VENDOR       a `cargo vendor` directory of this Cargo.lock (required: no network)
#   CORE_GPU_QWEN3_0_6B   Qwen3-0.6B directory (config.json, tokenizer.json, *.safetensors)
#   CORE_GPU_QWEN3_8B     Qwen3-8B directory (test; perf when set)
#   CORE_GPU_PROMPTS      the bench prompts: short, 4k, 8k, 16k .prompt.txt (test only)
#   CORE_GPU_TARGET       build directory (default target/core-gpu)
#   CORE_GPU_OUT          where the verdict goes (default out/core-gpu)
#   CORE_GPU_STEPS        default "build cpu test reference sanitize perf"
#   CORE_GPU_PODMAN_ARGS  extra podman flags, split on spaces (default --device nvidia.com/gpu=all)
#   CORE_GPU_RUNNER       the launcher of the GPU steps (above); run by bash when not executable.
#                         What it reads itself (VMM_IMG, CUDAIMG, GPU_RUN_*) passes through
#   CORE_GPU_RUNNER_MEM, CORE_GPU_RUNNER_TIMEOUT   its -m (MiB, default 12288) and -T (seconds,
#                         default 1800) for each program
#   CORE_GPU_RUNNER_ARGS  more options for it, split on spaces (e.g. "-c 8")
#   CORE_GPU_RUNNER_SANITIZER  a host copy of the CUDA image's /usr/local/cuda/compute-sanitizer
#                         directory, bound at that path in the guest for the sanitize step
# Each step runs in a container without network, removed by its name.
# Exit 0: every step ok. 1: a step failed, or the card was lost. 3: none failed, at least one
# was skipped. 2: usage.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
image=${CORE_GPU_IMAGE:?set CORE_GPU_IMAGE to a CUDA devel image with nvcc and Rust}
vendor=${CORE_GPU_VENDOR:?set CORE_GPU_VENDOR to a cargo vendor directory of Cargo.lock}
build=${CORE_GPU_TARGET:-$ROOT/target/core-gpu}
out=${CORE_GPU_OUT:-$ROOT/out/core-gpu}
steps=${CORE_GPU_STEPS:-build cpu test reference sanitize perf}
podman_args=${CORE_GPU_PODMAN_ARGS:---device nvidia.com/gpu=all}
runner=${CORE_GPU_RUNNER:-}
runner_mem=${CORE_GPU_RUNNER_MEM:-12288}
runner_secs=${CORE_GPU_RUNNER_TIMEOUT:-1800}
features=cuda
for step in $steps; do
  case $step in
    build|cpu|test|reference|sanitize|perf) ;;
    *) echo "core-gpu: unknown step $step (build, cpu, test, reference, sanitize, perf)" >&2; exit 2 ;;
  esac
done
command -v jq >/dev/null || { echo "core-gpu: jq not found (the verdict is JSON)" >&2; exit 2; }
runner_cmd=() runner_sha=
if [ -n "$runner" ]; then
  [ -f "$runner" ] || { echo "core-gpu: CORE_GPU_RUNNER $runner is not a file" >&2; exit 2; }
  runner=$(readlink -f "$runner")
  runner_sha=$(sha256sum < "$runner" | cut -c1-64)
  runner_cmd=("$runner")
  [ -x "$runner" ] || runner_cmd=(bash "$runner")
  # The device flag is for a GPU the host gives to containers: with a runner, none gets it.
  podman_args=${podman_args//--device nvidia.com\/gpu=all/}
fi

# ---- the tree under test: its commit, whether it is that commit's, a hash of its files.
sha=$(git -C "$ROOT" rev-parse HEAD)
branch=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)
dirty=0
[ -z "$(git -C "$ROOT" status --porcelain)" ] || dirty=1
id=$sha
[ "$dirty" = 0 ] || id=$sha+dirty
# sha256 over the sha256 and path of every file git tracks or would track (not the ignored).
tree=$(cd "$ROOT" && git ls-files -z -c -o --exclude-standard --deduplicate \
  | while IFS= read -r -d '' f; do if [ -f "$f" ]; then printf '%s\0' "$f"; fi; done \
  | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -c1-16)
started=$(date -Is)
t_start=$(date +%s)
logs=$out/logs/$id
mkdir -p "$build/target" "$build/cargo-home" "$build/perf" "$logs"
rm -f "$logs"/*.reason "$logs/card-lost" "$build"/perf/*.json
rm -rf "$logs/runner" # the runner wants one empty results directory per program
: > "$logs/steps.tsv"

cat > "$build/cargo-home/config.toml" <<'TOML'
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "/vendor"

[net]
offline = true
TOML
mounts=(-v "$ROOT:/src:ro" -v "$vendor:/vendor:ro" -v "$build/target:/target" -v "$build/cargo-home:/cargo-home"
  -v "$build/perf:/perf")
envs=()
for v in QWEN3_0_6B QWEN3_8B PROMPTS; do
  var=CORE_GPU_$v
  if [ -n "${!var:-}" ]; then
    mounts+=(-v "${!var}:/data/$v:ro")
    envs+=(-e "X0NE_INFER_CORE_$v=/data/$v")
  fi
done

# on_card NAME [RUNNER OPTIONS...] PROGRAM [ARGS...]: one program on the card through the
# runner, once; its stdout, stderr and exit code. 199 is written down: the main loop stops on it.
on_card() {
  local name=$1 rc=0
  shift
  # shellcheck disable=SC2086
  "${runner_cmd[@]}" -m "$runner_mem" -T "$runner_secs" ${CORE_GPU_RUNNER_ARGS:-} -o "$logs/runner/$name" "$@" || rc=$?
  case $rc in
    199)
      echo "card lost: the runner exited 199 at $name, nothing was run after it; the host needs a reboot" > "$logs/card-lost"
      echo "core-gpu: CARD LOST at $name (runner exit 199): every remaining step is dropped, nothing is retried" >&2 ;;
    125)
      echo "core-gpu: the runner failed at $name (exit 125): no result from the card" >&2
      [ -z "${step:-}" ] || echo "runner exit 125 at $name: no result from the card" > "$logs/$step.reason" ;;
  esac
  return "$rc"
}

# ---- the GPU, as a container of this image sees it with the flags given. Without one, the
# steps that need none run without the default device flag (podman refuses a device it lacks).
# With a runner: as the guest's nvidia-smi sees it, asked only when a GPU step is to run.
gpu=
no_gpu="no GPU"
if [ -z "$runner" ]; then
  gpu=$(podman run --rm --network=none --name "x0ne-core-gpu-probe-$$" $podman_args "$image" \
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1) || gpu=
else
  echo "core-gpu: the GPU steps go through the runner $runner"
  case " $steps " in
    *" test "*|*" reference "*|*" sanitize "*|*" perf "*)
      prc=0
      step='' on_card probe -g /usr/bin/nvidia-smi --query-gpu=name,driver_version --format=csv,noheader \
        > "$logs/probe.out" 2> "$logs/probe.log" || prc=$?
      if [ "$prc" = 0 ]; then
        gpu=$(head -n 1 "$logs/probe.out")
        [ -n "$gpu" ] || no_gpu="no GPU: the runner's nvidia-smi named none"
      else
        no_gpu="no GPU: runner exit $prc: $(grep -v '^core-gpu: ' "$logs/probe.log" | tail -n 1 | tr -d '\t\r' | cut -c1-200)"
      fi
      ;;
    *) no_gpu="no GPU: the runner was not asked" ;;
  esac
fi
run_args=$podman_args
if [ -s "$logs/card-lost" ]; then
  echo "core-gpu: $(cat "$logs/card-lost")"
elif [ -z "$gpu" ]; then
  run_args=${podman_args//--device nvidia.com\/gpu=all/}
  if [ -z "$runner" ]; then echo "core-gpu: no GPU in $image with \"$podman_args\": the GPU steps are skipped, not passed"
  else echo "core-gpu: $no_gpu: the GPU steps are skipped, not passed"; fi
else
  echo "core-gpu: GPU $gpu"
fi

# need VAR...: the step's reason and exit 2 when one is not set.
need() {
  local var
  for var in "$@"; do
    if [ -z "${!var:-}" ]; then
      echo "core-gpu: $step needs $var" >&2
      echo "$var is not set" > "$logs/$step.reason"
      return 2
    fi
  done
}
# run NAME CMD...: CMD in the image, the tree read-only.
run() {
  local name="x0ne-core-gpu-$1-$$"
  shift
  # shellcheck disable=SC2086
  podman run --rm --network=none --name "$name" $run_args "${mounts[@]}" "${envs[@]}" -w /src \
    -e CARGO_HOME=/cargo-home -e CARGO_TARGET_DIR=/target -e CARGO_INCREMENTAL=0 "$image" "$@"
}
cargo_args=(--release --locked --offline -p x0ne-infer --features "$features")
# x0ne-core's own workspace: its lock, its target directory, the same vendored crates.
core_cargo='cd /src/x0ne-core && export CARGO_TARGET_DIR=/target/core && cargo'
# The lib test binary of x0ne-infer, as cargo names it (built by the build step).
test_bin() {
  run bin cargo test "${cargo_args[@]}" --lib --no-run --message-format=json \
    | sed -n 's/.*"executable":"\([^"]*\/x0ne_infer-[^"]*\)".*/\1/p' | tail -1
}
infer_bin() {
  if [ -n "$runner" ]; then
    bin=$(built lib x0ne_infer cargo test "${cargo_args[@]}" --lib --no-run --message-format=json)
    return
  fi
  bin=$(test_bin)
  [ -n "$bin" ] || { echo "core-gpu: no test binary, run the build step" >&2; echo "no test binary: run the build step" > "$logs/$step.reason"; return 2; }
}
# built KIND NAME CARGO...: the path in the containers (/target/...) of the one executable the
# cargo command given, with --message-format=json, names for the target NAME of kind KIND (lib:
# its test binary). Not a file pattern: an older binary of the same name stays in deps/.
# None, two, or one that is not an executable file of the build directory: the step's reason, 2.
built() {
  local kind=$1 name=$2 json paths n
  shift 2
  refuse() { echo "core-gpu: $1" >&2; echo "$1" > "$logs/$step.reason"; return 2; }
  json=$(run bin "$@") || { refuse "cargo did not name the $kind $name: run the build step"; return; }
  paths=$(jq -rR --arg k "$kind" --arg n "$name" 'fromjson? | select(type == "object")
    | select(.reason == "compiler-artifact" and (.executable | type) == "string"
        and ((.target.name // "") | gsub("-"; "_")) == ($n | gsub("-"; "_")) and ((.target.kind // []) | index($k)) != null)
    | .executable' <<<"$json" | sort -u)
  n=$(grep -c . <<<"$paths" || true)
  [ "$n" = 1 ] || { refuse "$n executables for the $kind $name, one expected"; return; }
  case $paths in
    /target/*) ;;
    *) refuse "the $kind $name is at $paths, outside /target"; return ;;
  esac
  [ -f "$build$paths" ] && [ -x "$build$paths" ] || { refuse "no executable file $build$paths"; return; }
  printf '%s\n' "$paths"
}
# What a program on the card reads, read-only, at the paths of the containers.
core_src=(-f "$ROOT/x0ne-core:/src/x0ne-core")
infer_src=(-f "$ROOT/x0ne-infer:/src/x0ne-infer" "${core_src[@]}")
infer_at=(-w /src/x0ne-infer -e CARGO_MANIFEST_DIR=/src/x0ne-infer)
oracle_at=(-w /src/x0ne-core/oracle -e CARGO_MANIFEST_DIR=/src/x0ne-core/oracle)
# data NAME...: in dargs, the model directories given among these and their variables.
data() {
  local v var
  dargs=()
  for v in "$@"; do
    var=CORE_GPU_$v
    [ -z "${!var:-}" ] || dargs+=(-f "${!var}:/data/$v" -e "X0NE_INFER_CORE_$v=/data/$v")
  done
}

# One function per step; its exit code is the step's. 3 with a .reason file: skipped.
step_build() {
  run build cargo build "${cargo_args[@]}" --bin x0ne-infer || return
  run build cargo test "${cargo_args[@]}" --lib --no-run || return
  run build bash -c "$core_cargo test --locked --offline -p x0ne-core-oracle --no-run \
    && cargo build --release --locked --offline -p x0ne-core-oracle --example core-perf"
}
step_cpu() {
  run cpu env CARGO_TARGET_DIR=/target/core bash /src/ci/core-cpu.sh
}
step_test() {
  need CORE_GPU_QWEN3_0_6B || return
  # Without Qwen3-8B or the prompts: the GPU tests whose #[ignore = "GPU: ..."] names what is
  # missing are not run, and the step is skipped with their names. Nothing to name: it fails.
  local v var missing=() skipped=() skips=() t
  for v in QWEN3_8B PROMPTS; do
    var=CORE_GPU_$v
    [ -n "${!var:-}" ] && continue
    missing+=("$var")
    while IFS= read -r t; do
      case " ${skipped[*]} " in *" $t "*) ;; *) skipped+=("$t"); skips+=(--skip "$t") ;; esac
    done < <(awk -v var="X0NE_INFER_CORE_$v" '
      /#\[ignore = "GPU: / { held = index($0, var) > 0; seen = 1; next }
      seen && match($0, /fn [a-z0-9_]+\(/) { if (held) print substr($0, RSTART + 3, RLENGTH - 4); seen = 0; held = 0 }' \
      "$ROOT/x0ne-infer/src/core_backend/tests.rs")
  done
  if [ "${#missing[@]}" -gt 0 ] && [ "${#skipped[@]}" = 0 ]; then need "${missing[@]}" || return; fi
  infer_bin || return
  local args=(core_backend:: --include-ignored --test-threads=1 --nocapture "${skips[@]}")
  if [ -n "$runner" ]; then
    data QWEN3_0_6B QWEN3_8B PROMPTS
    on_card tests "${infer_src[@]}" "${dargs[@]}" "${infer_at[@]}" "$build$bin" "${args[@]}" || return
  else
    run tests "$bin" "${args[@]}" || return
  fi
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "partial: $(printf '%s is not set, ' "${missing[@]}")not run: ${skipped[*]}" > "$logs/$step.reason"
    return 3
  fi
}
step_reference() {
  if [ -n "$runner" ]; then
    # Both binaries are found before the card is used; a failure of the first does not hide the second's.
    local t rc=0 c bins=()
    for t in gpu_kernels gpu_engine; do
      bin=$(built test "$t" bash -c "$core_cargo test --locked --offline -p x0ne-core-oracle --no-run --message-format=json") || return
      bins+=("$bin")
    done
    for bin in "${bins[@]}"; do
      t=${bin##*/gpu_}
      c=0
      on_card "${t%%-*}" "${core_src[@]}" "${oracle_at[@]}" "$build$bin" --include-ignored --test-threads=1 --nocapture || c=$?
      [ "$c" != 199 ] || return 199
      [ "$rc" != 0 ] || rc=$c
    done
    return "$rc"
  fi
  run reference bash -c "$core_cargo test --locked --offline -p x0ne-core-oracle \
    --test gpu_kernels --test gpu_engine -- --include-ignored --test-threads=1 --nocapture"
}
step_sanitize() {
  need CORE_GPU_QWEN3_0_6B || return
  local tool san rc=0 c tools=/usr/local/cuda/compute-sanitizer
  if [ -n "$runner" ] && [ -z "${CORE_GPU_RUNNER_SANITIZER:-}" ]; then
    echo "compute-sanitizer is not in the guest of the runner (CORE_GPU_RUNNER_SANITIZER: a copy of the CUDA image's $tools)" > "$logs/$step.reason"
    return 3
  fi
  if [ -n "$runner" ] && [ ! -x "$CORE_GPU_RUNNER_SANITIZER/compute-sanitizer" ]; then
    echo "no compute-sanitizer in $CORE_GPU_RUNNER_SANITIZER" > "$logs/$step.reason"
    return 2
  fi
  infer_bin || return
  for tool in memcheck initcheck; do
    san=(compute-sanitizer --tool "$tool" --error-exitcode 9)
    # memcheck only: "explicit" reports the errors of the API calls made, not the driver's
    # log warnings (as x0ne Core's scripts/test.sh sanitize).
    if [ "$tool" = memcheck ]; then san+=(--report-api-errors explicit); fi
    echo "== $tool: the Qwen3-0.6B suite"
    if [ -n "$runner" ]; then
      # The tool finds its libraries beside its own file: its directory is bound where the
      # image has it and run from there (-g), the test binary at its path of the containers.
      data QWEN3_0_6B
      c=0
      on_card "$tool" "${infer_src[@]}" "${dargs[@]}" -f "$CORE_GPU_RUNNER_SANITIZER:$tools" -f "$build$bin:$bin" \
        "${infer_at[@]}" -g "$tools/${san[0]}" "${san[@]:1}" "$bin" \
        core_backend::tests::linked_tests::gpu::core_passes_the_suite_0_6b --exact --include-ignored --test-threads=1 || c=$?
      [ "$c" != 199 ] || return 199
      [ "$c" = 0 ] || rc=$c
      continue
    fi
    run "$tool" "${san[@]}" "$bin" core_backend::tests::linked_tests::gpu::core_passes_the_suite_0_6b \
      --exact --include-ignored --test-threads=1 || rc=$?
  done
  return "$rc"
}
step_perf() {
  need CORE_GPU_QWEN3_0_6B || return
  local model rc=0 none=0 c
  if [ -n "$runner" ]; then
    bin=$(built example core-perf bash -c "$core_cargo build --release --locked --offline -p x0ne-core-oracle \
      --example core-perf --message-format=json") || return
  fi
  for model in QWEN3_0_6B QWEN3_8B; do
    var=CORE_GPU_$model
    [ -n "${!var:-}" ] || continue
    if [ -n "$runner" ]; then
      # core-perf prints the JSON it writes: the file stays in the guest, stdout comes back.
      on_card "perf-$model" "${core_src[@]}" "${oracle_at[@]}" -f "${!var}:/data/$model" "$build$bin" "/data/$model" /dev/null \
        > "$build/perf/$model.json" || return
    else
      run perf bash -c "$core_cargo run --release --locked --offline -p x0ne-core-oracle --example core-perf -- \
        /data/$model /perf/$model.json" || return
    fi
    cp "$build/perf/$model.json" "$out/perf-$id-$model.json" || return
    c=0
    bash "$ROOT/ci/core-perf.sh" compare "$out/perf-$id-$model.json" || c=$?
    case $c in
      0) ;;
      3) none=1 ;;
      *) rc=1 ;;
    esac
  done
  [ "$rc" = 0 ] || return "$rc"
  if [ "$none" = 1 ]; then echo "no baseline" > "$logs/$step.reason"; return 3; fi
}

for step in $steps; do
  rc=0
  t0=$(date +%s)
  status=ok
  if [ -s "$logs/card-lost" ]; then
    # The card no longer answers: nothing more is run, on it or beside it.
    status=skipped
    echo "card lost: not run" > "$logs/$step.reason"
  else
    case $step in
      test|reference|sanitize|perf)
        if [ -z "$gpu" ]; then
          status=skipped
          echo "$no_gpu" > "$logs/$step.reason"
        fi
        ;;
    esac
  fi
  if [ "$status" = ok ]; then
    "step_$step" > >(tee "$logs/$step.log") 2>&1 || rc=$?
    if [ -s "$logs/card-lost" ]; then
      status=fail
      [ "$rc" != 0 ] || rc=199
      cp "$logs/card-lost" "$logs/$step.reason"
    elif [ "$rc" = 3 ] && [ -s "$logs/$step.reason" ]; then status=skipped; rc=0
    elif [ "$rc" != 0 ]; then status=fail; fi
  fi
  reason=
  [ ! -s "$logs/$step.reason" ] || reason=$(head -n 1 "$logs/$step.reason")
  secs=$(($(date +%s) - t0))
  printf '%s\t%s\t%d\t%d\t%s\n' "$step" "$status" "$secs" "$rc" "$reason" >> "$logs/steps.tsv"
  echo "core-gpu: $step $status${reason:+: $reason} rc=$rc in $secs s"
done

jq -n --arg sha "$sha" --arg branch "$branch" --arg dirty "$dirty" --arg tree "$tree" --arg image "$image" \
  --arg gpu "$gpu" --arg started "$started" --arg ended "$(date -Is)" --argjson duration "$(($(date +%s) - t_start))" \
  --arg log "$logs" --arg podman_args "$podman_args" --rawfile steps "$logs/steps.tsv" \
  --arg runner "$runner" --arg runner_sha "$runner_sha" --arg stopped "$(cat "$logs/card-lost" 2>/dev/null || true)" '
  ($steps | split("\n") | map(select(length > 0) | split("\t")
     | {name: .[0], status: .[1], seconds: (.[2] | tonumber), exit: (.[3] | tonumber)}
       + (if (.[4] // "") == "" then {} else {reason: .[4]} end))) as $s
  | ($s | map(select(.status == "fail") | .name)) as $failed
  | ($s | map(select(.status == "skipped") | .name)) as $skipped
  | {status: (if ($failed | length) > 0 or $stopped != "" then "red" elif ($skipped | length) > 0 then "skipped" else "green" end),
     sha: $sha, short: ($sha[0:7] + (if $dirty == "1" then "+dirty" else "" end)), dirty: ($dirty == "1"),
     tree: $tree, branch: $branch, started: $started, ended: $ended, duration_s: $duration,
     image: $image, podman_args: $podman_args, gpu: (if $gpu == "" then null else $gpu end), log_dir: $log,
     runner: (if $runner == "" then null else {path: $runner, sha256: $runner_sha} end),
     stopped: (if $stopped == "" then null else $stopped end),
     failed: $failed, skipped: $skipped, steps: $s}' > "$out/$id.json.tmp"
mv "$out/$id.json.tmp" "$out/$id.json"
ln -sfn "$id.json" "$out/latest.json"
status=$(jq -r .status "$out/$id.json")
echo "core-gpu: $status $(jq -r .short "$out/$id.json") in $(($(date +%s) - t_start)) s$(jq -r '
  (if .stopped then " (" + .stopped + ")" else "" end)
  + (if (.failed | length) > 0 then " (failed: " + (.failed | join(", ")) + ")" else "" end)
  + (if (.skipped | length) > 0 then " (skipped, not passed: " + ([.steps[] | select(.status == "skipped") | "\(.name): \(.reason)"] | join(", ")) + ")" else "" end)' "$out/$id.json"); $out/$id.json"
case $status in
  green) exit 0 ;;
  skipped) exit 3 ;;
  *) exit 1 ;;
esac
