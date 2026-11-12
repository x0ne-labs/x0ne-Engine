#!/usr/bin/env bash
# Tests of scripts/core-gpu.sh (its verdict file, against a fake podman and a fake GPU runner:
# no container, no GPU, no microVM, no build), of ci/core-gpu-check.sh (the gate that reads it) and of ci/core-perf.sh (the
# performance ratchet). Every number below is made up for the test, in a throwaway directory.
# Run by ci/tests/ci-sh.sh, or by hand. Needs jq and git.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
FAILS=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s: %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1" "${3:-} exit ${RC:-?}: ${OUT:-}"; fi; }

# ---- fixture: a repository with the scripts under test, model directories, a fake podman.
REPO=$T/repo BIN=$T/bin
mkdir -p "$REPO/scripts" "$REPO/ci" "$BIN" "$T/vendor" "$T/m06" "$T/m8" "$T/prompts"
cp "$SRC/scripts/core-gpu.sh" "$REPO/scripts/"
cp "$SRC/ci/core-perf.sh" "$SRC/ci/core-gpu-check.sh" "$SRC/ci/core-perf-baseline.json" "$REPO/ci/"
# The tests of core_backend, as the tree has them: the test step reads which model each GPU test needs.
mkdir -p "$REPO/x0ne-infer/src/core_backend" "$REPO/x0ne-core"
cp "$SRC/x0ne-infer/src/core_backend/tests.rs" "$REPO/x0ne-infer/src/core_backend/"
printf '/out/\n/target\n' > "$REPO/.gitignore"
git -C "$REPO" init -q -b main
commit() { git -C "$REPO" add -A; git -C "$REPO" -c user.name=t -c user.email=t@t commit -qm "$1"; SHA=$(git -C "$REPO" rev-parse HEAD); }
commit fixture

# perf_json <model> <prefill tok/s> <decode tok/s> <ttft s>: a core-perf result, made up.
perf_json() {
  jq -n --arg model "$1" --argjson p "$2" --argjson d "$3" --argjson t "$4" '
    {schema: 1, model: $model, engine: "Fake GPU (cc 0.0, 1 SMs, cubin none), driver none", kernel_set: "none",
     prompt_tokens: 4096, decode_tokens: 256,
     metrics: {prefill_tok_s: {median: $p, min: $p, max: $p}, decode_tok_s: {median: $d, min: $d, max: $d},
               ttft_s: {median: $t, min: $t, max: $t}}}'
}
# podman: a GPU when FAKE_GPU is set; the step named FAKE_FAIL fails; a perf run writes FAKE_PERF.
cat > "$BIN/podman" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_LOG"
case " $* " in
  *" nvidia-smi "*)
    if [ -n "${FAKE_GPU:-}" ]; then echo "Fake GPU, 0.0"; exit 0; fi
    exit 125 ;;
esac
name= perf=
while [ $# -gt 0 ]; do
  case $1 in
    --name) name=$2; shift 2 ;;
    -v) case $2 in *:/perf) perf=${2%:/perf} ;; esac; shift 2 ;;
    *) shift ;;
  esac
done
[ -n "${FAKE_FAIL:-}" ] && case $name in x0ne-core-gpu-"$FAKE_FAIL"-*) echo "the fake $FAKE_FAIL fails"; exit 101 ;; esac
case $name in
  x0ne-core-gpu-bin-*)
    # What cargo --message-format=json prints: every artifact of the command, the executables among them.
    art() { printf '{"reason":"compiler-artifact","target":{"kind":["%s"],"name":"%s"},"executable":%s}\n' "$1" "$2" "$3"; }
    art lib serde null
    art lib x0ne_infer '"/target/release/deps/x0ne_infer-0000"'
    [ -z "${FAKE_TWO:-}" ] || art lib x0ne_infer '"/target/release/deps/x0ne_infer-1111"'
    art test cpu_oracle '"/target/core/debug/deps/cpu_oracle-0000"'
    art test gpu_kernels '"/target/core/debug/deps/gpu_kernels-0000"'
    art test gpu_engine '"/target/core/debug/deps/gpu_engine-0000"'
    art example core-perf '"/target/core/release/examples/core-perf"'
    echo '{"reason":"build-finished","success":true}' ;;
  x0ne-core-gpu-perf-*) for m in QWEN3_0_6B QWEN3_8B; do cp "$FAKE_PERF" "$perf/$m.json"; done ;;
esac
exit 0
EOF
chmod +x "$BIN/podman"
export FAKE_LOG=$T/podman.log FAKE_PERF=$T/perf.json
perf_json fake-model 1000 100 4 > "$FAKE_PERF"

# gpu [VAR=value...]: core-gpu.sh in the fixture. RC, OUT, V (the verdict file of HEAD).
gpu() {
  RC=0
  : > "$FAKE_LOG"
  OUT=$(env PATH="$BIN:$PATH" CORE_GPU_IMAGE=fake-image CORE_GPU_VENDOR="$T/vendor" CORE_GPU_TARGET="$T/build" \
    CORE_GPU_QWEN3_0_6B="$T/m06" CORE_GPU_QWEN3_8B="$T/m8" CORE_GPU_PROMPTS="$T/prompts" \
    "$@" bash "$REPO/scripts/core-gpu.sh" 2>&1) || RC=$?
  V=$REPO/out/core-gpu/$SHA.json
}
# shellcheck disable=SC2034  # GRC, GOUT, RC, OUT are read by the eval of check
gate() { GRC=0; GOUT=$(bash "$REPO/ci/core-gpu-check.sh" "$@" 2>&1) || GRC=$?; }
st() { jq -r --arg s "$1" '.steps[] | select(.name == $s) | "\(.status)|\(.reason // "")"' "$V"; }

# ---- no GPU: the steps that need one are skipped, said so, and nothing is green.
gpu
check "without a GPU the run exits 3, not 0" '[ "$RC" = 3 ]'
check "its verdict is status skipped, GPU null, for the clean commit" \
  '[ "$(jq -r "[.status, (.gpu == null), .dirty, .sha] | @tsv" "$V")" = "$(printf "skipped\ttrue\tfalse\t%s" "$SHA")" ]' "$(cat "$V" 2>&1)"
check "build and cpu ran and are ok" '[ "$(st build)" = "ok|" ] && [ "$(st cpu)" = "ok|" ]'
for s in test reference sanitize perf; do
  check "$s is skipped: no GPU" '[ "$(st '"$s"')" = "skipped|no GPU" ]' "$(st "$s")"
done
check "the skipped steps are listed, and none is listed as failed" \
  '[ "$(jq -c "[.skipped, .failed]" "$V")" = "[[\"test\",\"reference\",\"sanitize\",\"perf\"],[]]" ]'
check "the verdict carries a tree hash, the image and each step's seconds and exit code" \
  'jq -e "(.tree | test(\"^[0-9a-f]{16}$\")) and .image == \"fake-image\" and (.steps | all(has(\"seconds\") and has(\"exit\")))" "$V" >/dev/null'
check "no container of a skipped step was started" '! grep -q "x0ne-core-gpu-\(tests\|reference\|memcheck\|perf\)-" "$FAKE_LOG"'
check "the default device flag is dropped for the steps that still run" \
  'grep "x0ne-core-gpu-build-" "$FAKE_LOG" | grep -vq -- "--device"' "$(cat "$FAKE_LOG")"
check "the last line says skipped, not passed" 'grep -q "core-gpu: skipped .*skipped, not passed: test: no GPU" <<<"$OUT"'
gate
check "SECURITY: the gate refuses a run whose GPU steps were skipped, and names them" \
  '[ "$GRC" = 1 ] && grep -q "reference skipped: no GPU" <<<"$GOUT" && grep -q "no GPU was seen" <<<"$GOUT"' "$GOUT"

# ---- a GPU, no baseline: perf is skipped, the rest is ok, still not green.
gpu FAKE_GPU=1
check "with a GPU and no baseline the run exits 3" '[ "$RC" = 3 ]'
check "perf is skipped: no baseline; the GPU steps are ok" \
  '[ "$(st perf)" = "skipped|no baseline" ] && [ "$(st test)" = "ok|" ] && [ "$(st reference)" = "ok|" ] && [ "$(st sanitize)" = "ok|" ]' "$(jq -c .steps "$V")"
check "the GPU is recorded" '[ "$(jq -r .gpu "$V")" = "Fake GPU, 0.0" ]'
check "the perf results are kept beside the verdict" '[ -s "$REPO/out/core-gpu/perf-$SHA-QWEN3_0_6B.json" ] && [ -s "$REPO/out/core-gpu/perf-$SHA-QWEN3_8B.json" ]'
gate
check "SECURITY: the gate refuses a run with no performance baseline" '[ "$GRC" = 1 ] && grep -q "perf skipped: no baseline" <<<"$GOUT"' "$GOUT"

# ---- the baseline recorded and committed: green, and the gate opens for that commit only.
bash "$REPO/ci/core-perf.sh" update "$FAKE_PERF" > /dev/null
OLD=$SHA
commit baseline
gate
check "a commit without a verdict is refused" '[ "$GRC" = 1 ] && grep -q "no verdict" <<<"$GOUT"' "$GOUT"
gpu FAKE_GPU=1
check "every step ok: exit 0, status green" '[ "$RC" = 0 ] && [ "$(jq -r .status "$V")" = green ] && [ "$(jq -c "[.failed, .skipped]" "$V")" = "[[],[]]" ]' "$(jq -c .steps "$V" 2>&1)"
check "latest.json points at it" '[ "$(readlink "$REPO/out/core-gpu/latest.json")" = "$SHA.json" ]'
gate
check "the gate opens on a green verdict of HEAD" '[ "$GRC" = 0 ] && grep -q "ok ${SHA:0:7}: green on Fake GPU" <<<"$GOUT"' "$GOUT"
gate "$OLD"
check "SECURITY: and stays shut for another commit" '[ "$GRC" = 1 ]' "$GOUT"
export CORE_GPU_REQUIRED="build cpu test reference sanitize perf more"
gate
unset CORE_GPU_REQUIRED
check "SECURITY: a required step the run did not have refuses the gate" '[ "$GRC" = 1 ] && grep -q "more absent" <<<"$GOUT"' "$GOUT"
jq '.status = "green" | .steps |= map(if .name == "reference" then .status = "skipped" | .reason = "no GPU" else . end)' "$V" > "$T/v" && cp "$T/v" "$V"
gate
check "SECURITY: a verdict that says green over a skipped step is refused" '[ "$GRC" = 1 ] && grep -q "reference skipped" <<<"$GOUT"' "$GOUT"

# ---- a subset of the steps is green for itself, and not enough for the gate.
gpu FAKE_GPU=1 CORE_GPU_STEPS="build cpu"
check "a run of two steps is green" '[ "$RC" = 0 ] && [ "$(jq -r ".steps | length" "$V")" = 2 ]'
gate
check "SECURITY: the gate refuses it for the steps it lacks" '[ "$GRC" = 1 ] && grep -q "reference absent" <<<"$GOUT"' "$GOUT"

# ---- a failed step: red, exit 1, the others still run.
gpu FAKE_GPU=1 FAKE_FAIL=reference
check "a failed step makes the run red, exit 1" \
  '[ "$RC" = 1 ] && [ "$(jq -r .status "$V")" = red ] && [ "$(jq -c .failed "$V")" = "[\"reference\"]" ] && [ "$(st reference)" = "fail|" ] && [ "$(jq -r ".steps[] | select(.name == \"reference\") | .exit" "$V")" = 101 ]' "$(jq -c .steps "$V" 2>&1)"
check "the steps after it ran" '[ "$(st sanitize)" = "ok|" ] && [ "$(st perf)" = "ok|" ]'
check "its output is kept in the log directory" 'grep -q "the fake reference fails" "$(jq -r .log_dir "$V")/reference.log"'
gate
check "SECURITY: the gate refuses a red verdict" '[ "$GRC" = 1 ] && grep -q "reference fail" <<<"$GOUT"' "$GOUT"

# ---- a slower run than the baseline: perf fails.
perf_json fake-model 1000 80 4 > "$FAKE_PERF"
gpu FAKE_GPU=1
check "a decode rate 20 % under the baseline fails perf" '[ "$RC" = 1 ] && [ "$(st perf)" = "fail|" ] && grep -q "RED   decode_tok_s 80 against 100" <<<"$OUT"'
perf_json fake-model 1000 100 4 > "$FAKE_PERF"

# ---- a model directory not given, on a GPU: the step fails and says what it needs.
gpu FAKE_GPU=1 CORE_GPU_QWEN3_0_6B= CORE_GPU_STEPS="build sanitize"
check "a step without its model directory fails with the reason" '[ "$RC" = 1 ] && [ "$(st sanitize)" = "fail|CORE_GPU_QWEN3_0_6B is not set" ]' "$(jq -c .steps "$V" 2>&1)"

# ---- a tree that is not its commit's: another file, which the gate does not read.
echo change >> "$REPO/scripts/core-gpu.sh"
gpu FAKE_GPU=1
check "a changed tree writes <sha>+dirty.json, marked dirty" \
  '[ "$RC" = 0 ] && [ "$(jq -r "[.dirty, .short] | @tsv" "$REPO/out/core-gpu/$SHA+dirty.json")" = "$(printf "true\t%s+dirty" "${SHA:0:7}")" ]'
rm -f "$V"
gate
check "SECURITY: the gate refuses the verdict of a changed tree" '[ "$GRC" = 1 ] && grep -q "not the commit" <<<"$GOUT"' "$GOUT"
git -C "$REPO" checkout -q -- scripts/core-gpu.sh

# ---- the test step without Qwen3-8B: what needs Qwen3-0.6B alone runs, the step is skipped and says what was not run.
gpu FAKE_GPU=1 CORE_GPU_QWEN3_8B= CORE_GPU_STEPS="build test"
check "without Qwen3-8B the test step is skipped, not ok, and names the tests it did not run" \
  '[ "$RC" = 3 ] && [ "$(st test)" = "skipped|partial: CORE_GPU_QWEN3_8B is not set, not run: core_passes_the_suite_8b generate_is_cores_greedy the_device_path_is_the_host_path" ]' "$(st test)"
check "the tests that need Qwen3-0.6B alone ran, the others were skipped by name" \
  'grep "x0ne-core-gpu-tests-" "$FAKE_LOG" | grep -q -- "core_backend:: --include-ignored --test-threads=1 --nocapture --skip core_passes_the_suite_8b --skip generate_is_cores_greedy --skip the_device_path_is_the_host_path$"' "$(cat "$FAKE_LOG")"
gate
check "SECURITY: the gate refuses a partial test step" '[ "$GRC" = 1 ] && grep -q "test skipped: partial" <<<"$GOUT"' "$GOUT"
gpu FAKE_GPU=1 CORE_GPU_QWEN3_8B= CORE_GPU_STEPS="build test" FAKE_FAIL=tests
check "a partial test step that fails is failed, not skipped" '[ "$RC" = 1 ] && [ "$(st test)" = "fail|" ]' "$(st test)"
gpu FAKE_GPU=1 CORE_GPU_PROMPTS= CORE_GPU_STEPS="build test"
check "without the prompts, the two tests that read them are the ones not run" \
  '[ "$(st test)" = "skipped|partial: CORE_GPU_PROMPTS is not set, not run: generate_is_cores_greedy the_device_path_is_the_host_path" ]' "$(st test)"
gpu FAKE_GPU=1 CORE_GPU_QWEN3_0_6B= CORE_GPU_STEPS="build test"
check "without Qwen3-0.6B the test step fails" '[ "$RC" = 1 ] && [ "$(st test)" = "fail|CORE_GPU_QWEN3_0_6B is not set" ]' "$(st test)"
check "a run in containers names no runner" '[ "$(jq -c "[.runner, .stopped]" "$V")" = "[null,null]" ]'

# ---- CORE_GPU_RUNNER: the GPU steps through a launcher (one program on the card, its output and code back),
# faked here. It is not executable, as the real one: bash runs it. FAKE_RUNNER_RC="name:code,..." sets the code of a run.
cat > "$T/runner.sh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_RUNNER_LOG"
res= guest=0
while [ $# -gt 0 ]; do
  case $1 in
    -f|-e|-T|-m|-c|-d|-w) shift 2 ;;
    -o) res=$2; shift 2 ;;
    -g) guest=1; shift ;;
    *) break ;;
  esac
done
name=${res##*/}
[ -n "$res" ] && [ ! -e "$res" ] || { echo "fake-runner: -o $res is not empty: one results directory per run" >&2; exit 125; }
mkdir -p "$res"; : > "$res/summary.txt"
case ",${FAKE_RUNNER_RC:-}" in
  *",$name:"*) rc=${FAKE_RUNNER_RC#*"$name:"}; rc=${rc%%,*}; echo "fake-runner: refused: $name exits $rc" >&2; exit "$rc" ;;
esac
if [ "$guest" = 1 ]; then
  case $1 in */nvidia-smi) echo "Fake GPU, 0.0" ;; esac
else
  [ -f "$1" ] && [ -x "$1" ] || { echo "fake-runner: PROGRAM '$1' is not an executable file" >&2; exit 125; }
fi
case $name in perf-*) cat "$FAKE_PERF" ;; esac
exit 0
EOF
export FAKE_RUNNER_LOG=$T/runner.log
BT=$T/build/target
mkdir -p "$BT/release/deps" "$BT/core/debug/deps" "$BT/core/release/examples" "$T/san"
for f in release/deps/x0ne_infer-0000 release/deps/x0ne_infer-1111 core/debug/deps/gpu_kernels-0000 core/debug/deps/gpu_engine-0000 \
  core/release/examples/core-perf; do printf '#!/bin/sh\n' > "$BT/$f"; chmod +x "$BT/$f"; done
printf '#!/bin/sh\n' > "$T/san/compute-sanitizer"; chmod +x "$T/san/compute-sanitizer"
rgpu() { : > "$FAKE_RUNNER_LOG"; gpu CORE_GPU_RUNNER="$T/runner.sh" "$@"; RL=$REPO/out/core-gpu/logs/$SHA/runner; }
# called OPTIONS-AND-PROGRAM: the runner was called once with exactly this after its memory, limit and results directory.
called() { [ "$(grep -cxF -- "-m 12288 -T 1800 -o $RL/$1 $2" "$FAKE_RUNNER_LOG")" = 1 ]; }
SRC_INFER="-f $REPO/x0ne-infer:/src/x0ne-infer -f $REPO/x0ne-core:/src/x0ne-core"
M06="-f $T/m06:/data/QWEN3_0_6B -e X0NE_INFER_CORE_QWEN3_0_6B=/data/QWEN3_0_6B"
ORACLE="-f $REPO/x0ne-core:/src/x0ne-core -w /src/x0ne-core/oracle -e CARGO_MANIFEST_DIR=/src/x0ne-core/oracle"

rgpu
check "through a runner, without a sanitizer given: sanitize is skipped with its reason, the run exits 3" \
  '[ "$RC" = 3 ] && [ "$(jq -c .skipped "$V")" = "[\"sanitize\"]" ] && grep -q "^skipped|compute-sanitizer is not in the guest of the runner" <<<"$(st sanitize)"' "$OUT"
check "build, cpu, test, reference and perf are ok" \
  '[ "$(st build)$(st cpu)$(st test)$(st reference)$(st perf)" = "ok|ok|ok|ok|ok|" ]' "$(jq -c .steps "$V" 2>&1)"
check "the GPU is the one the runner's nvidia-smi names, from the guest's image" \
  '[ "$(jq -r .gpu "$V")" = "Fake GPU, 0.0" ] && called probe "-g /usr/bin/nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"' "$(cat "$FAKE_RUNNER_LOG")"
check "the verdict names the runner and its digest" \
  '[ "$(jq -r "[.runner.path, .runner.sha256, .stopped] | @tsv" "$V")" = "$(printf "%s\t%s\t" "$T/runner.sh" "$(sha256sum < "$T/runner.sh" | cut -c1-64)")" ]' "$(jq -c .runner "$V")"
check "test: the lib test binary cargo names, sources and models read-only at the container's paths" \
  'called tests "$SRC_INFER $M06 -f $T/m8:/data/QWEN3_8B -e X0NE_INFER_CORE_QWEN3_8B=/data/QWEN3_8B -f $T/prompts:/data/PROMPTS -e X0NE_INFER_CORE_PROMPTS=/data/PROMPTS -w /src/x0ne-infer -e CARGO_MANIFEST_DIR=/src/x0ne-infer $BT/release/deps/x0ne_infer-0000 core_backend:: --include-ignored --test-threads=1 --nocapture"' "$(cat "$FAKE_RUNNER_LOG")"
check "reference: gpu_kernels, then gpu_engine, each with the oracle's manifest directory" \
  'called kernels "$ORACLE $BT/core/debug/deps/gpu_kernels-0000 --include-ignored --test-threads=1 --nocapture" && called engine "$ORACLE $BT/core/debug/deps/gpu_engine-0000 --include-ignored --test-threads=1 --nocapture"' "$(cat "$FAKE_RUNNER_LOG")"
check "perf: core-perf on each model, its JSON read from the runner's stdout and held to the baseline" \
  'called perf-QWEN3_0_6B "$ORACLE -f $T/m06:/data/QWEN3_0_6B $BT/core/release/examples/core-perf /data/QWEN3_0_6B /dev/null" && called perf-QWEN3_8B "$ORACLE -f $T/m8:/data/QWEN3_8B $BT/core/release/examples/core-perf /data/QWEN3_8B /dev/null" && cmp -s "$FAKE_PERF" "$REPO/out/core-gpu/perf-$SHA-QWEN3_0_6B.json" && grep -q "holds its baseline" <<<"$OUT"' "$(cat "$FAKE_RUNNER_LOG")"
check "the runner was called six times, and for nothing else" '[ "$(wc -l < "$FAKE_RUNNER_LOG")" = 6 ]' "$(cat "$FAKE_RUNNER_LOG")"
check "build and cpu ran in containers, without the device flag; no GPU step did" \
  'grep -q "x0ne-core-gpu-build-" "$FAKE_LOG" && grep -q "x0ne-core-gpu-cpu-" "$FAKE_LOG" && ! grep -q -- "--device\|nvidia-smi\|x0ne-core-gpu-\(tests\|reference\|memcheck\|initcheck\|perf\)-" "$FAKE_LOG"' "$(cat "$FAKE_LOG")"
check "no cargo runs in the guest" '! grep -q "cargo" "$FAKE_RUNNER_LOG"'
gate
check "SECURITY: the gate refuses a run whose sanitize step was skipped" '[ "$GRC" = 1 ] && grep -q "sanitize skipped: compute-sanitizer is not in the guest" <<<"$GOUT"' "$GOUT"

rgpu CORE_GPU_RUNNER_SANITIZER="$T/san" CORE_GPU_RUNNER_MEM=12288
SAN="$SRC_INFER $M06 -f $T/san:/usr/local/cuda/compute-sanitizer -f $BT/release/deps/x0ne_infer-0000:/target/release/deps/x0ne_infer-0000 -w /src/x0ne-infer -e CARGO_MANIFEST_DIR=/src/x0ne-infer -g /usr/local/cuda/compute-sanitizer/compute-sanitizer --tool"
SUITE="/target/release/deps/x0ne_infer-0000 core_backend::tests::linked_tests::gpu::core_passes_the_suite_0_6b --exact --include-ignored --test-threads=1"
check "with a copy of the image's compute-sanitizer: both tools over the suite, the run green" \
  '[ "$RC" = 0 ] && [ "$(jq -r .status "$V")" = green ] && called memcheck "$SAN memcheck --error-exitcode 9 --report-api-errors explicit $SUITE" && called initcheck "$SAN initcheck --error-exitcode 9 $SUITE"' "$(cat "$FAKE_RUNNER_LOG"; jq -c .steps "$V")"
gate
check "the gate opens on a green verdict through a runner" '[ "$GRC" = 0 ]' "$GOUT"
rgpu CORE_GPU_RUNNER_SANITIZER="$T/nothing" CORE_GPU_STEPS="build sanitize"
check "a sanitizer directory without compute-sanitizer fails the step" '[ "$RC" = 1 ] && grep -q "^fail|no compute-sanitizer in" <<<"$(st sanitize)"' "$(st sanitize)"
rgpu CORE_GPU_RUNNER_MEM=16384 CORE_GPU_RUNNER_TIMEOUT=600 CORE_GPU_RUNNER_ARGS="-c 8" CORE_GPU_STEPS="reference"
check "memory, limit and extra options of the runner are settable" \
  'grep -qF -- "-m 16384 -T 600 -c 8 -o $RL/kernels $ORACLE" "$FAKE_RUNNER_LOG"' "$(cat "$FAKE_RUNNER_LOG")"

# ---- 125: the run failed, the step is red, the next ones still run.
rgpu FAKE_RUNNER_RC=engine:125
check "a run the runner fails (125) makes its step red, with the reason" \
  '[ "$RC" = 1 ] && [ "$(jq -c .failed "$V")" = "[\"reference\"]" ] && [ "$(st reference)" = "fail|runner exit 125 at engine: no result from the card" ] && [ "$(jq -r ".steps[] | select(.name == \"reference\") | .exit" "$V")" = 125 ]' "$(jq -c .steps "$V" 2>&1)"
check "the steps after it ran, and the card is not said lost" '[ "$(st perf)" = "ok|" ] && [ "$(jq -r .stopped "$V")" = null ]'
rgpu FAKE_RUNNER_RC=kernels:101
check "a program that fails on the card fails its step with its code; the next program still runs" \
  '[ "$RC" = 1 ] && [ "$(st reference)" = "fail|" ] && [ "$(jq -r ".steps[] | select(.name == \"reference\") | .exit" "$V")" = 101 ] && grep -q -- "-o $RL/engine " "$FAKE_RUNNER_LOG"' "$(jq -c .steps "$V" 2>&1)"

# ---- 199: the card is lost. Nothing else runs, on the card or not, and nothing is tried again.
rgpu FAKE_RUNNER_RC=kernels:199 CORE_GPU_RUNNER_SANITIZER="$T/san"
check "SECURITY: a lost card (199) stops the run: red, exit 1, the reason in the verdict" \
  '[ "$RC" = 1 ] && [ "$(jq -r .status "$V")" = red ] && grep -q "^card lost: the runner exited 199 at kernels" <<<"$(jq -r .stopped "$V")"' "$(cat "$V" 2>&1)"
check "the step that lost it is failed with that reason, the steps after it are not run and say why" \
  'grep -q "^fail|card lost: the runner exited 199 at kernels" <<<"$(st reference)" && [ "$(st sanitize)" = "skipped|card lost: not run" ] && [ "$(st perf)" = "skipped|card lost: not run" ] && [ "$(st test)" = "ok|" ]' "$(jq -c .steps "$V" 2>&1)"
check "SECURITY: the runner is not called again after a 199: kernels is its last call, made once" \
  '[ "$(tail -n 1 "$FAKE_RUNNER_LOG" | grep -c -- "-o $RL/kernels ")" = 1 ] && [ "$(grep -c -- "-o $RL/kernels " "$FAKE_RUNNER_LOG")" = 1 ] && [ "$(wc -l < "$FAKE_RUNNER_LOG")" = 3 ]' "$(cat "$FAKE_RUNNER_LOG")"
check "the last line says the card is lost" 'grep -q "core-gpu: red .*card lost: the runner exited 199 at kernels" <<<"$OUT"' "$OUT"
gate
check "SECURITY: the gate refuses a run stopped by a lost card" '[ "$GRC" = 1 ] && grep -q "reference fail: card lost" <<<"$GOUT"' "$GOUT"
rgpu FAKE_RUNNER_RC=probe:199
check "SECURITY: a card lost at the probe: red, no step runs, not even in a container" \
  '[ "$RC" = 1 ] && [ "$(jq -r .status "$V")" = red ] && [ "$(jq -c "[.steps[].status] | unique" "$V")" = "[\"skipped\"]" ] && [ "$(st build)" = "skipped|card lost: not run" ] && [ ! -s "$FAKE_LOG" ] && [ "$(wc -l < "$FAKE_RUNNER_LOG")" = 1 ]' "$(cat "$V" 2>&1)"
rgpu FAKE_RUNNER_RC=perf-QWEN3_0_6B:199
check "a card lost at the first model of perf: the second is not measured" \
  'grep -q "^fail|card lost" <<<"$(st perf)" && ! grep -q "perf-QWEN3_8B" "$FAKE_RUNNER_LOG"' "$(cat "$FAKE_RUNNER_LOG")"

# ---- no card, or a runner that refuses: the GPU steps are skipped with the runner's reason, never green.
rgpu FAKE_RUNNER_RC=probe:125
check "a runner that refuses the probe: the GPU steps are skipped, exit 3, GPU null" \
  '[ "$RC" = 3 ] && [ "$(jq -r "[.status, (.gpu == null)] | @tsv" "$V")" = "$(printf "skipped\ttrue")" ] && [ "$(jq -c .skipped "$V")" = "[\"test\",\"reference\",\"sanitize\",\"perf\"]" ]' "$(cat "$V" 2>&1)"
check "their reason is no GPU, with the runner's code and last words" \
  '[ "$(st reference)" = "skipped|no GPU: runner exit 125: fake-runner: refused: probe exits 125" ]' "$(st reference)"
check "build and cpu still ran, and the runner was called once" '[ "$(st build)$(st cpu)" = "ok|ok|" ] && [ "$(wc -l < "$FAKE_RUNNER_LOG")" = 1 ]'
gate
check "SECURITY: the gate refuses it" '[ "$GRC" = 1 ] && grep -q "no GPU was seen" <<<"$GOUT"' "$GOUT"
rgpu CORE_GPU_STEPS="build cpu"
check "a run without a GPU step does not boot the card: the runner is not called" '[ "$RC" = 0 ] && [ ! -s "$FAKE_RUNNER_LOG" ] && [ "$(jq -r .gpu "$V")" = null ]' "$(cat "$FAKE_RUNNER_LOG")"

# ---- the binaries: the one cargo names, or none.
rgpu FAKE_TWO=1 CORE_GPU_STEPS="build test reference"
check "SECURITY: two candidate test binaries: the step is refused, none is run" \
  '[ "$RC" = 1 ] && [ "$(st test)" = "fail|2 executables for the lib x0ne_infer, one expected" ] && ! grep -q "x0ne_infer-" "$FAKE_RUNNER_LOG" && [ "$(st reference)" = "ok|" ]' "$(st test) $(cat "$FAKE_RUNNER_LOG")"
rm "$BT/core/debug/deps/gpu_engine-0000"
rgpu CORE_GPU_STEPS="reference"
check "a binary cargo names and the build directory lacks: the step fails before the card" \
  '[ "$RC" = 1 ] && grep -q "^fail|no executable file $BT/core/debug/deps/gpu_engine-0000" <<<"$(st reference)" && ! grep -q "kernels\|engine" "$FAKE_RUNNER_LOG"' "$(st reference)"
printf '#!/bin/sh\n' > "$BT/core/debug/deps/gpu_engine-0000"; chmod +x "$BT/core/debug/deps/gpu_engine-0000"
RC=0; OUT=$(env PATH="$BIN:$PATH" CORE_GPU_IMAGE=fake-image CORE_GPU_VENDOR="$T/vendor" CORE_GPU_RUNNER="$T/absent.sh" bash "$REPO/scripts/core-gpu.sh" 2>&1) || RC=$?
check "a runner that is not a file is a usage error" '[ "$RC" = 2 ] && grep -q "CORE_GPU_RUNNER" <<<"$OUT"' "$OUT"

# ---- usage.
gpu CORE_GPU_STEPS="build nothing"
check "an unknown step is refused before anything runs" '[ "$RC" = 2 ] && [ ! -s "$FAKE_LOG" ]'

# ---- ci/core-perf.sh on its own: compare and update.
B=$T/baseline.json
# The tree's baseline without its runs: the cases below record their own.
jq '.runs = {}' "$SRC/ci/core-perf-baseline.json" > "$B"
perf() { RC=0; OUT=$(bash "$SRC/ci/core-perf.sh" "$@" 2>&1) || RC=$?; }
check "every run the tree's baseline holds is a recorded one: its kernel set, its date, positive metrics" 'jq -e ".tolerance_pct > 0 and ([.runs[] | (.kernel_set | test(\"^[0-9a-f]{64}$\")) and (.date | test(\"^[0-9]{4}-\")) and ((.metrics | length) > 0) and ([.metrics[] | . > 0] | all)] | all)" "$SRC/ci/core-perf-baseline.json" >/dev/null'
perf_json fake-model 1000 100 4 > "$T/r.json"
perf compare "$T/r.json" "$B"
check "no entry: exit 3, nothing compared" '[ "$RC" = 3 ] && grep -q "no baseline for \"fake-model on Fake GPU\"" <<<"$OUT"'
perf update "$T/r.json" "$B"
check "update records a first run" '[ "$RC" = 0 ] && [ "$(jq -c ".runs[\"fake-model on Fake GPU\"].metrics" "$B")" = "{\"prefill_tok_s\":1000,\"decode_tok_s\":100,\"ttft_s\":4}" ]' "$(cat "$B")"
perf compare "$T/r.json" "$B"
check "the same run holds" '[ "$RC" = 0 ] && grep -q "holds its baseline" <<<"$OUT"'
perf_json fake-model 960 96 4.19 > "$T/r.json"; perf compare "$T/r.json" "$B"
check "inside the tolerance on every metric: holds" '[ "$RC" = 0 ]'
perf_json fake-model 1000 94 4 > "$T/r.json"; perf compare "$T/r.json" "$B"
check "a rate 6 % lower: red, and named" '[ "$RC" = 1 ] && grep -q "RED   decode_tok_s 94 against 100 (-6 %)" <<<"$OUT"'
perf_json fake-model 1000 100 4.3 > "$T/r.json"; perf compare "$T/r.json" "$B"
check "a time 7.5 % longer: red" '[ "$RC" = 1 ] && grep -q "RED   ttft_s 4.3 against 4 (7.5 %)" <<<"$OUT"'
perf update "$T/r.json" "$B"
check "update does not record a run that regressed" '[ "$RC" = 1 ] && [ "$(jq ".runs[].metrics.ttft_s" "$B")" = 4 ]'
perf_json fake-model 1100 99 3.5 > "$T/r.json"; perf update "$T/r.json" "$B"
check "update keeps the better of each metric" '[ "$RC" = 0 ] && [ "$(jq -c ".runs[].metrics" "$B")" = "{\"prefill_tok_s\":1100,\"decode_tok_s\":100,\"ttft_s\":3.5}" ]' "$(cat "$B")"
perf_json fake-model 500 50 9 > "$T/r.json"; perf update "$T/r.json" "$B" --replace
check "--replace writes the run as it is" '[ "$RC" = 0 ] && [ "$(jq -c ".runs[].metrics" "$B")" = "{\"prefill_tok_s\":500,\"decode_tok_s\":50,\"ttft_s\":9}" ]'
perf_json other-model 1 1 1 > "$T/r.json"; perf compare "$T/r.json" "$B"
check "another model has no entry" '[ "$RC" = 3 ]'
jq 'del(.metrics.ttft_s)' <(perf_json fake-model 500 50 9) > "$T/r.json"; perf compare "$T/r.json" "$B"
check "a metric the baseline holds and the run lacks: red" '[ "$RC" = 1 ] && grep -q "ttft_s is in the baseline and not in the run" <<<"$OUT"'
for bad in '{}' 'not json' '{"schema":1,"model":"m","engine":"g","metrics":{"decode_tok_s":{"median":0}}}' '{"schema":2,"model":"m","engine":"g","metrics":{"a_s":{"median":1}}}'; do
  printf '%s\n' "$bad" > "$T/r.json"; perf compare "$T/r.json" "$B"
  check "a file that is not a result is refused: $bad" '[ "$RC" = 2 ]'
done
perf compare "$T/absent.json" "$B"
check "no result file: usage" '[ "$RC" = 2 ]'
perf_json fake-model 1 1 1 > "$T/r.json"; echo '{"schema":1}' > "$T/b.json"; perf compare "$T/r.json" "$T/b.json"
check "a baseline that is not one is refused" '[ "$RC" = 2 ]'

[ "$FAILS" = 0 ] || { echo "core-gpu: $FAILS failed"; exit 1; }
echo "core-gpu: all passed"
