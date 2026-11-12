#!/usr/bin/env bash
# Tests of scripts/bench.sh (make bench): its verdict file, its Markdown report and its ratchet
# against ci/bench-baseline.json, against a fake podman and a fake GPU runner: no container, no
# GPU, no microVM, no build. Every number below is made up for the test, in a throwaway
# directory. Run by ci/tests/ci-sh.sh, or by hand. Needs jq and git.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
FAILS=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s: %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1" "${3:-} exit ${RC:-?}: $(tail -n 5 <<<"${OUT:-}")"; fi; }

# ---- fixture: a repository with the script under test, model directories, a fake podman.
REPO=$T/repo BIN=$T/bin
mkdir -p "$REPO/scripts" "$REPO/ci" "$BIN" "$T/vendor" "$T/m06" "$T/moe" "$T/prompts" "$T/llama/bin" "$T/llama/lib"
cp "$SRC/scripts/bench.sh" "$REPO/scripts/"
cp "$SRC/ci/bench-baseline.json" "$REPO/ci/"
printf '/out/\n/target\n' > "$REPO/.gitignore"
git -C "$REPO" init -q -b main
commit() { git -C "$REPO" add -A; git -C "$REPO" -c user.name=t -c user.email=t@t commit -qm "$1"; SHA=$(git -C "$REPO" rev-parse HEAD); }
commit fixture
dense() { printf '{"model_type":"qwen3","num_hidden_layers":28,"vocab_size":151936,"max_position_embeddings":40960}\n' > "$1/config.json"; echo '{}' > "$1/tokenizer.json"; }
dense "$T/m06"
printf '{"model_type":"qwen3_moe","num_hidden_layers":4,"num_experts":128,"vocab_size":151936,"max_position_embeddings":40960}\n' > "$T/moe/config.json"
echo '{"what":"first 4 layers"}' > "$T/moe/CUT.json"
printf '#!/bin/sh\n' > "$T/llama/bin/llama-server"; chmod +x "$T/llama/bin/llama-server"
echo gguf > "$T/q.gguf"

# Results as the programs print them (made-up numbers): FAKE_SCALE multiplies every rate.
stats() { jq -nc --argjson v "$1" '{median: $v, min: $v, max: $v, mean: $v, stddev: 0, n: 5}'; }
results() {
  local s=${FAKE_SCALE:-1}
  jq -nc --argjson s "$s" --argjson same "${FAKE_SAME:-true}" '
    def st($v): {median: $v, min: $v, max: $v, mean: $v, stddev: 0, n: 5};
    {schema: 1, kind: "core", model: "m", max_ctx: 40960, decode_tokens: 256, reps: 5,
     engine: "Fake GPU (cc 0.0, 1 SMs, cubin none), driver 0, arena 7 bytes", kernel_set: "k", nvcc: "n",
     config: {model_type: "qwen3", layers: 28, moe: null},
     load: {cold_s: 2, reload_s: 1},
     memory: {device_total_bytes: 34359738368, free_before_load_bytes: 33000000000, free_after_load_bytes: 30000000000,
              free_min_bytes: 29000000000, arena_bytes: 2900000000, taken_at_load_bytes: 3000000000, peak_used_bytes: 5359738368},
     lengths: [
       {name: "short", tokens: 20, source: "short.ids.json", status: "ok", same_tokens: $same,
        first: {ttft_s: 0.01, prefill_tok_s: 2000, decode_tok_s: 590},
        ttft_s: st(0.002 / $s), prefill_tok_s: st(10000 * $s), decode_tok_s: st(600 * $s)},
       {name: "4k", tokens: 3827, source: "4k.ids.json", status: "ok", same_tokens: true,
        first: {ttft_s: 0.2, prefill_tok_s: 19000, decode_tok_s: 580},
        ttft_s: st(0.17 / $s), prefill_tok_s: st(23000 * $s), decode_tok_s: st(589 * $s)},
       {name: "max", tokens: 40704, source: "synthetic", status: "refused", reason: "made-up refusal"}
     ]}' > "$T/res-core.json"
  jq -nc --argjson s "$s" '
    def st($v): {median: $v, min: $v, max: $v, mean: $v, stddev: 0, n: 5};
    {schema: 1, kind: "engine", model: "m", decode_tokens: 256, reps: 5, context_tokens: 40960, load_s: 3,
     cases: [{prompt: "short", tick: "production", tick_ms: 100, ids_tokens: 20, prompt_tokens: 20,
       stream: {first: {ttft_s: 0.1}, ttft_s: st(0.1 / $s), last_content_s: st(0.5), end_s: st(1.0),
                decode_tok_s: st(550 * $s), completion_tokens: {min: 256, max: 256}, finish: ["length"], errors: []},
       whole: {first: {e2e_s: 1.0}, e2e_s: st(1.0 / $s), completion_tokens: {min: 256, max: 256}, finish: ["length"], errors: []}}]}' \
    > "$T/res-engine.json"
  jq -nc --argjson s "$s" --argjson errors "${FAKE_LOAD_ERRORS:-0}" --argjson bare "${FAKE_BARE_503:-0}" '
    {schema: 1, kind: "load", model: "m", tick: "production", max_queue: 8, ceiling: 4096, prompt: "short", max_tokens: 256, level_s: 30,
     levels: [{clients: 1, seconds: 30, requests: 20, ok: 20, rejected: 0, rejected_frac: 0, retry_after_s: null,
               rejected_without_retry_after: 0, errors: 0, error_samples: [],
               ttft_s: {p50: 0.1, p95: 0.1, p99: 0.1, n: 20}, e2e_s: {p50: 1, p95: 1, p99: 1, n: 20},
               throughput_tok_s: (170 * $s), ok_per_s: 0.66, jain_index: 1, fifo_inversions: 0, fifo_pairs: 0},
              {clients: 32, seconds: 31, requests: 90, ok: 50, rejected: 40, rejected_frac: 0.44, retry_after_s: {min: 1, max: 1},
               rejected_without_retry_after: $bare, errors: $errors, error_samples: (if $errors > 0 then ["made-up reset"] else [] end),
               ttft_s: {p50: 2, p95: 8, p99: 9, n: 50}, e2e_s: {p50: 3, p95: 9, p99: 10, n: 50},
               throughput_tok_s: (400 * $s), ok_per_s: 1.6, jain_index: 0.9, fifo_inversions: 1, fifo_pairs: 300}]}' \
    > "$T/res-load.json"
  jq -nc --argjson s "$s" --argjson errors "${FAKE_SOAK_ERRORS:-0}" '
    {schema: 1, kind: "soak", model: "m", tick: "production", clients: 4, seconds: 600, window_s: 60,
     windows: [{end_s: 60, ok: 100, tokens: 25600, throughput_tok_s: 426, errors: 0, rejected: 0, ttft_p50_s: 0.2, rss_kib: 100000, vram_used_mib: 9000}],
     summary: {ok: 1000, errors: $errors, rejected: 0, throughput_tok_s: (426 * $s), throughput_drift_pct: -0.5,
               throughput_slope_pct_per_h: -1, rss_first_kib: 100000, rss_last_kib: 100100, rss_peak_kib: 100200,
               vram_first_mib: 9000, vram_last_mib: 9000}}' > "$T/res-soak.json"
  jq -nc '
    def st($v): {median: $v, min: $v, max: $v, mean: $v, stddev: 0, n: 5};
    {schema: 1, kind: "baseline", engine: "llama.cpp", model: "m", server_version: "b0", file_type: "BF16", decode_tokens: 256, reps: 5,
     lengths: [{name: "short", tokens: 20, source: "short.ids.json", status: "ok", first: {prefill_tok_s: 1},
                prefill_s: st(0.003), prefill_tok_s: st(7000), decode_tok_s: st(400), e2e_s: st(0.7)}]}' > "$T/res-baseline.json"
}
results
# podman: a GPU when FAKE_GPU is set; the container named FAKE_FAIL fails; a program's container prints its result.
cat > "$BIN/podman" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_LOG"
case " $* " in
  *" nvidia-smi "*)
    if [ -n "${FAKE_GPU:-}" ]; then echo "Fake GPU, 0.0, 32607 MiB"; exit 0; fi
    exit 125 ;;
esac
name=
while [ $# -gt 0 ]; do
  case $1 in
    --name) name=$2; shift 2 ;;
    *) shift ;;
  esac
done
[ -n "${FAKE_FAIL:-}" ] && case $name in x0ne-bench-"$FAKE_FAIL"-*) echo "the fake $FAKE_FAIL fails"; exit 101 ;; esac
case $name in
  x0ne-bench-bin-*)
    art() { printf '{"reason":"compiler-artifact","target":{"kind":["%s"],"name":"%s"},"executable":%s}\n' "$1" "$2" "$3"; }
    art lib serde null
    art lib x0ne_infer '"/target/release/deps/x0ne_infer-0000"'
    art example core-bench '"/target/core/release/examples/core-bench"'
    art example core-perf '"/target/core/release/examples/core-perf"'
    echo '{"reason":"build-finished","success":true}' ;;
  x0ne-bench-core-*|x0ne-bench-engine-*|x0ne-bench-load-*|x0ne-bench-soak-*|x0ne-bench-baseline-*)
    t=${name#x0ne-bench-}; t=${t%%-*}
    echo "running"; printf 'x0ne-bench-result: %s\n' "$(cat "$FAKE_RES/res-$t.json")" ;;
esac
exit 0
EOF
chmod +x "$BIN/podman"
export FAKE_LOG=$T/podman.log FAKE_RES=$T

# The runner: one program on the card, faked. Not executable, as the real one: bash runs it.
# FAKE_RUNNER_RC="name:code,..." sets the code of a run; FAKE_NO_RESULT names runs that print none.
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
mkdir -p "$res"
case ",${FAKE_RUNNER_RC:-}" in
  *",$name:"*) rc=${FAKE_RUNNER_RC#*"$name:"}; rc=${rc%%,*}; echo "fake-runner: refused: $name exits $rc" >&2; exit "$rc" ;;
esac
if [ "$guest" = 1 ]; then
  case $1 in */nvidia-smi) echo "Fake GPU, 0.0, 32607 MiB" ;; esac
  exit 0
fi
[ -f "$1" ] && [ -x "$1" ] || { echo "fake-runner: PROGRAM '$1' is not an executable file" >&2; exit 125; }
case " ${FAKE_NO_RESULT:-} " in *" $name "*) echo "test result: ok"; exit 0 ;; esac
t=${name%%-*}
echo "running 1 test"
printf 'x0ne-bench-result: %s\n' "$(cat "$FAKE_RES/res-$t.json")"
echo "test result: ok"
EOF
export FAKE_RUNNER_LOG=$T/runner.log
BT=$T/build/target
mkdir -p "$BT/release/deps" "$BT/core/release/examples"
for f in release/deps/x0ne_infer-0000 core/release/examples/core-bench; do printf '#!/bin/sh\n' > "$BT/$f"; chmod +x "$BT/$f"; done

# bench [VAR=value...]: bench.sh in the fixture, through the runner. RC, OUT, V (the verdict of HEAD), RL.
# What a caller's environment may hold for make core-gpu, which bench.sh falls back on: never here.
CLEAN=(env -u CORE_GPU_IMAGE -u CORE_GPU_VENDOR -u CORE_GPU_TARGET -u CORE_GPU_PODMAN_ARGS -u CORE_GPU_RUNNER
  -u CORE_GPU_RUNNER_ARGS -u CORE_GPU_QWEN3_0_6B -u CORE_GPU_QWEN3_8B -u CORE_GPU_PROMPTS)
bench() {
  RC=0
  : > "$FAKE_LOG"; : > "$FAKE_RUNNER_LOG"
  OUT=$("${CLEAN[@]}" PATH="$BIN:$PATH" BENCH_IMAGE=fake-image BENCH_VENDOR="$T/vendor" BENCH_TARGET="$T/build" \
    BENCH_RUNNER="$T/runner.sh" BENCH_QWEN3_0_6B="$T/m06" BENCH_PROMPTS="$T/prompts" BENCH_SOAK_MIN=10 \
    "$@" bash "$REPO/scripts/bench.sh" 2>&1) || RC=$?
  V=$REPO/out/bench/$SHA.json
  RL=$REPO/out/bench/logs/$SHA/runner
}
# run TIER MODEL: "status|reason" of that run in the verdict.
run() { jq -r --arg t "$1" --arg m "$2" '.tiers[] | select(.name == $t) | .runs[] | select(.model == $m) | "\(.status)|\(.reason // "")"' "$V"; }
called() { grep -F -- "-o $RL/$1 " "$FAKE_RUNNER_LOG"; }

# ---- usage.
bench BENCH_STEPS="build nothing"
check "an unknown step is refused before anything runs" '[ "$RC" = 2 ] && [ ! -s "$FAKE_LOG" ] && grep -q "unknown step nothing" <<<"$OUT"'
bench BENCH_SOAK_MIN=0
check "a soak of no minutes is refused" '[ "$RC" = 2 ] && grep -q "BENCH_SOAK_MIN" <<<"$OUT"'
bench BENCH_REPS=x
check "repetitions that are not a positive number are refused" '[ "$RC" = 2 ] && grep -q "BENCH_REPS" <<<"$OUT"'
RC=0; OUT=$("${CLEAN[@]}" PATH="$BIN:$PATH" BENCH_VENDOR="$T/vendor" bash "$REPO/scripts/bench.sh" 2>&1) || RC=$?
check "no image: usage" '[ "$RC" = 2 ] && grep -q "BENCH_IMAGE" <<<"$OUT"'

# ---- the whole suite through the runner: Qwen3-0.6B measured, the models not given skipped, never green.
bench
check "a run with absent models exits 3 (partial), not 0" '[ "$RC" = 3 ] && [ "$(jq -r .status "$V")" = partial ]' "$(jq -c '[.status, .failed, .skipped]' "$V" 2>&1)"
check "the verdict says schema, commit, tree, machine, runner, GPU, driver and settings" \
  'jq -e ".schema == 1 and .sha == \"$SHA\" and .dirty == false and (.tree | test(\"^[0-9a-f]{16}$\"))
    and .machine.gpu.name == \"Fake GPU\" and .machine.gpu.driver == \"0.0\" and .machine.gpu.memory == \"32607 MiB\"
    and (.machine.host.cpu | type) == \"string\" and (.machine.host.threads > 0) and (.machine.host.mem_mib > 0)
    and .machine.runner.path == \"$T/runner.sh\" and (.machine.runner.sha256 | test(\"^[0-9a-f]{64}$\")) and .machine.runner.mem_mib == 16384
    and .settings.reps == 5 and .settings.decode_tokens == 256 and .settings.soak_s == 600
    and .settings.lengths == [\"short\",\"1k\",\"4k\",\"8k\",\"16k\",\"max\"] and .settings.clients == [1,2,4,8,16,32]" "$V" > /dev/null' \
  "$(jq -c '{machine, settings}' "$V" 2>&1)"
check "core: Qwen3-0.6B ok, its result kept whole" \
  '[ "$(run core qwen3-0.6b)" = "ok|" ] && [ "$(jq -c ".tiers[] | select(.name == \"core\") | .runs[] | select(.model == \"qwen3-0.6b\") | .result" "$V")" = "$(cat "$T/res-core.json")" ]'
for m in qwen3-8b qwen3-14b qwen3-32b qwen3-30b-a3b; do
  v=$(echo "$m" | tr 'a-z.-' 'A-Z__')
  check "core: $m is skipped, its variable named" '[ "$(run core '"$m"')" = "skipped|BENCH_'"$v"' is not set" ]' "$(run core "$m")"
done
check "every tier ran for Qwen3-0.6B, the baseline aside (no llama-server given)" \
  '[ "$(run engine qwen3-0.6b)$(run load qwen3-0.6b)$(run soak qwen3-0.6b)" = "ok|ok|ok|" ] && grep -q "^skipped|BENCH_LLAMA_SERVER is not set" <<<"$(run baseline qwen3-0.6b)"' \
  "$(jq -c '[.tiers[] | {name, runs: [.runs[] | [.model, .status, .reason]]}]' "$V")"
check "the baseline says what it needs: a llama-server built with CUDA for this GPU and a GGUF of the same weights" \
  'grep -q "built with CUDA" <<<"$(run baseline qwen3-0.6b)" && grep -q "BENCH_GGUF_QWEN3_0_6B" <<<"$(run baseline qwen3-0.6b)"' "$(run baseline qwen3-0.6b)"
check "no reference yet: each measured run says so, nothing is held" \
  '[ "$(jq -r "[.tiers[].runs[] | select(.status == \"ok\") | .ratchet.status] | unique | join(\",\")" "$V")" = "no reference" ]'
check "the GPU is the runner's nvidia-smi, from the guest image" \
  'grep -qF -- "-o $RL/probe -g /usr/bin/nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader" "$FAKE_RUNNER_LOG"' "$(cat "$FAKE_RUNNER_LOG")"
check "core: core-bench on the model and the prompts, read-only, with the lengths, decode and repetitions" \
  'called core-qwen3-0.6b | grep -qF -- "-m 16384 -T 3600 -o $RL/core-qwen3-0.6b -f $T/m06:/data/model -f $T/prompts:/data/prompts -w /tmp/x0ne-bench $BT/core/release/examples/core-bench /data/model --prompts /data/prompts --lengths short,1k,4k,8k,16k,max --decode 256 --reps 5"' \
  "$(cat "$FAKE_RUNNER_LOG")"
SETTINGS="-e X0NE_BENCH_MODEL=/data/model -e X0NE_BENCH_PROMPTS=/data/prompts -e X0NE_BENCH_REPS=5 -e X0NE_BENCH_DECODE=256"
check "engine: the test binary's engine case, the model and prompts bound, its settings passed" \
  'called engine-qwen3-0.6b | grep -qF -- "-o $RL/engine-qwen3-0.6b -f $T/m06:/data/model -f $T/prompts:/data/prompts $SETTINGS -e X0NE_BENCH_ENGINE_PROMPTS=short,4k,8k,16k -e X0NE_BENCH_TICKS=production,short -w /tmp/x0ne-bench $BT/release/deps/x0ne_infer-0000 bench_tests::gpu::engine --exact --include-ignored --test-threads=1 --nocapture"' \
  "$(cat "$FAKE_RUNNER_LOG")"
check "load: the clients, the seconds per level, the prompt and the queue" \
  'called load-qwen3-0.6b | grep -qF -- "-e X0NE_BENCH_CLIENTS=1,2,4,8,16,32 -e X0NE_BENCH_LEVEL_SECS=30 -e X0NE_BENCH_LOAD_PROMPT=short -e X0NE_BENCH_LOAD_TOKENS=256 -e X0NE_BENCH_MAX_QUEUE=8 -w /tmp/x0ne-bench $BT/release/deps/x0ne_infer-0000 bench_tests::gpu::load --exact"' \
  "$(cat "$FAKE_RUNNER_LOG")"
check "soak: ten minutes by default, the runner given the soak and half an hour more" \
  'called soak-qwen3-0.6b | grep -qF -- "-T 2400 -o $RL/soak-qwen3-0.6b" && called soak-qwen3-0.6b | grep -qF -- "-e X0NE_BENCH_SOAK_SECS=600 -e X0NE_BENCH_SOAK_CLIENTS=4 -e X0NE_BENCH_SOAK_WINDOW_SECS=60"' \
  "$(cat "$FAKE_RUNNER_LOG")"
check "the runner was called five times (probe, core, engine, load, soak), and for nothing else" '[ "$(wc -l < "$FAKE_RUNNER_LOG")" = 5 ]' "$(cat "$FAKE_RUNNER_LOG")"
check "no cargo runs in the guest; the binaries were found by cargo in a container" \
  '! grep -q cargo "$FAKE_RUNNER_LOG" && grep -q "x0ne-bench-bin-" "$FAKE_LOG" && ! grep -q -- "--device" "$FAKE_LOG"' "$(cat "$FAKE_LOG")"
check "the last line says partial and names what was skipped" 'tail -n 1 <<<"$OUT" | grep -q "bench: partial .*skipped: core/qwen3-8b"' "$(tail -n 1 <<<"$OUT")"
check "latest.json points at the verdict" '[ "$(readlink "$REPO/out/bench/latest.json")" = "$SHA.json" ]'

# ---- the Markdown report: generated from the verdict, beside it.
MD=$REPO/out/bench/$SHA.md
check "a Markdown report is written beside the verdict" '[ -s "$MD" ] && [ "$(readlink "$REPO/out/bench/latest.md")" = "$SHA.md" ]'
check "the report names the machine, the driver, the runner and the repetitions" \
  'grep -q "Fake GPU" "$MD" && grep -q "driver 0.0" "$MD" && grep -q "runner.sh" "$MD" && grep -q "5 repetitions" "$MD"' "$(head -n 30 "$MD")"
check "the report has the core table with medians and spreads, and the length that failed" \
  'grep -q "| qwen3-0.6b | short | 20 | 600" "$MD" && grep -q "max.*made-up refusal" "$MD"' "$(grep -n "qwen3-0.6b" "$MD" | head)"
check "the report says what the engine tier does not measure" 'grep -q "TDX" "$MD" && grep -q "relay" "$MD"'
check "the report gives the service overhead against core for the same prompt" 'grep -qi "overhead" "$MD" && grep -q "0.098" "$MD"' "$(grep -n -i overhead "$MD")"
check "the report lists every skip with its reason" 'grep -q "qwen3-8b.*BENCH_QWEN3_8B is not set" "$MD"'
check "the report of a verdict can be made again by hand" \
  'bash "$REPO/scripts/bench.sh" report "$V" | cmp -s - "$MD"'

# ---- the reference: recorded from the clean run, committed, then held.
RC=0; UOUT=$(bash "$REPO/scripts/bench.sh" update "$V" 2>&1) || RC=$?
check "update records the run's metrics for its GPU" \
  '[ "$RC" = 0 ] && [ "$(jq -r ".runs[\"Fake GPU\"].metrics[\"core/qwen3-0.6b/4k/decode_tok_s\"]" "$REPO/ci/bench-baseline.json")" = 589 ] && [ "$(jq -r ".runs[\"Fake GPU\"].sha" "$REPO/ci/bench-baseline.json")" = "$SHA" ]' "$UOUT"
check "every tier's headline metrics are recorded, the baseline's none" \
  'jq -e ".runs[\"Fake GPU\"].metrics | has(\"engine/qwen3-0.6b/short/production/stream_ttft_s\") and has(\"load/qwen3-0.6b/c32/throughput_tok_s\") and has(\"soak/qwen3-0.6b/throughput_tok_s\") and (keys | map(select(startswith(\"baseline\"))) | length == 0)" "$REPO/ci/bench-baseline.json" > /dev/null' \
  "$(jq -c '.runs[]?.metrics | keys' "$REPO/ci/bench-baseline.json")"
commit baseline
bench
check "the same numbers hold the reference: every measured run held" \
  '[ "$(jq -r "[.tiers[].runs[] | select(.status == \"ok\") | .ratchet.status] | unique | join(\",\")" "$V")" = held ]' "$(jq -c '[.tiers[].runs[] | .ratchet]' "$V")"

# ---- slower: the ratchet is red, the run fails, the others still ran.
FAKE_SCALE=0.8 results
bench
check "rates 20 % lower: red, exit 1, the regressed runs failed by the ratchet" \
  '[ "$RC" = 1 ] && [ "$(jq -r .status "$V")" = red ] && [ "$(run core qwen3-0.6b)" = "fail|below the reference" ]' "$(jq -c '[.status, .failed]' "$V")"
check "the red metric is named with its value, reference and change" \
  'jq -e "[.tiers[].runs[].ratchet.lines[]? | select(.metric == \"core/qwen3-0.6b/4k/decode_tok_s\" and .state == \"red\" and ((.value - 471.2) | fabs) < 0.000001 and .reference == 589)] | length == 1" "$V" > /dev/null && grep -q "RED.*core/qwen3-0.6b/4k/decode_tok_s" "$REPO/out/bench/$SHA.md"' \
  "$(jq -c '[.tiers[].runs[].ratchet.lines[]?] | .[0:3]' "$V")"
RC=0; UOUT=$(bash "$REPO/scripts/bench.sh" update "$V" 2>&1) || RC=$?
check "update refuses a run that regressed" '[ "$RC" = 1 ] && [ "$(jq -r ".runs[\"Fake GPU\"].metrics[\"core/qwen3-0.6b/4k/decode_tok_s\"]" "$REPO/ci/bench-baseline.json")" = 589 ]' "$UOUT"
FAKE_SCALE=0.97 results
bench
check "within the tolerance (5 % for core): held" '[ "$(jq -r ".tiers[] | select(.name == \"core\") | .runs[] | select(.model == \"qwen3-0.6b\") | .ratchet.status" "$V")" = held ]'
FAKE_SCALE=1 results

# ---- the checks that need no reference: they fail the run.
FAKE_SAME=false results; bench BENCH_STEPS="build core"
check "core: two repetitions that gave other tokens fail the run" '[ "$RC" = 1 ] && grep -q "^fail|short: the repetitions gave different tokens" <<<"$(run core qwen3-0.6b)"' "$(run core qwen3-0.6b)"
FAKE_LOAD_ERRORS=3 results; bench BENCH_STEPS="build load"
check "load: an error other than a 503 fails the run, with a sample" '[ "$RC" = 1 ] && grep -q "^fail|32 clients: 3 errors (made-up reset)" <<<"$(run load qwen3-0.6b)"' "$(run load qwen3-0.6b)"
FAKE_BARE_503=2 results; bench BENCH_STEPS="build load"
check "SECURITY: a 503 without Retry-After fails the run" '[ "$RC" = 1 ] && grep -q "^fail|32 clients: 2 refusals without Retry-After" <<<"$(run load qwen3-0.6b)"' "$(run load qwen3-0.6b)"
FAKE_SOAK_ERRORS=1 results; bench BENCH_STEPS="build soak"
check "soak: an error fails the run" '[ "$RC" = 1 ] && [ "$(run soak qwen3-0.6b)" = "fail|1 errors during the soak" ]' "$(run soak qwen3-0.6b)"
results

# ---- MoE and extra models: Core's tier with its own lengths, the server's tiers say why not.
bench BENCH_EXTRA_MODELS="$T/moe" BENCH_STEPS="build core engine"
check "an extra model is named by its directory and measured by core with the MoE lengths" \
  '[ "$(run core moe)" = "ok|" ] && called core-moe | grep -qF -- "-f $T/moe:/data/model" && called core-moe | grep -qF -- "--lengths short,1k,4k "' "$(cat "$FAKE_RUNNER_LOG")"
check "the verdict records what the model is: MoE, its layers, its truncation note" \
  '[ "$(jq -c ".tiers[] | select(.name == \"core\") | .runs[] | select(.model == \"moe\") | .about" "$V")" = "{\"model_type\":\"qwen3_moe\",\"layers\":4,\"experts\":128,\"truncated\":\"first 4 layers\"}" ]' \
  "$(jq -c '.tiers[] | select(.name == "core") | .runs[] | select(.model == "moe") | .about' "$V")"
check "the server's tier skips a MoE model and says why" '[ "$(run engine moe)" = "skipped|the server serves Qwen3 dense only (core_backend::core_config)" ]' "$(run engine moe)"
mkdir -p "$T/notok"; dense "$T/notok"; rm "$T/notok/tokenizer.json"
bench BENCH_EXTRA_MODELS="$T/notok" BENCH_STEPS="build engine"
check "a model without tokenizer.json is skipped by the server's tiers" '[ "$(run engine notok)" = "skipped|no tokenizer.json in $T/notok: the server needs one" ]' "$(run engine notok)"
bench BENCH_QWEN3_8B="$T/absent" BENCH_STEPS="build core"
check "a model variable naming no model directory fails, with the reason" '[ "$RC" = 1 ] && [ "$(run core qwen3-8b)" = "fail|BENCH_QWEN3_8B: no config.json in $T/absent" ]' "$(run core qwen3-8b)"

# ---- the baseline, given a llama-server and a GGUF.
bench BENCH_STEPS="build core baseline" BENCH_LLAMA_SERVER="$T/llama/bin/llama-server" BENCH_GGUF_QWEN3_0_6B="$T/q.gguf"
check "baseline: llama-server, its libraries, the GGUF and the model's config bound, the core lengths" \
  '[ "$(run baseline qwen3-0.6b)" = "ok|" ] && called baseline-qwen3-0.6b | grep -qF -- "-f $T/llama:/data/llama -f $T/q.gguf:/data/model.gguf -f $T/m06/config.json:/data/model/config.json -f $T/prompts:/data/prompts" && called baseline-qwen3-0.6b | grep -qF -- "-e X0NE_BENCH_LLAMA_SERVER=/data/llama/bin/llama-server -e X0NE_BENCH_LLAMA_LIB=/data/llama/lib -e X0NE_BENCH_GGUF=/data/model.gguf -e X0NE_BENCH_LENGTHS=short,1k,4k,8k,16k,max" && called baseline-qwen3-0.6b | grep -qF -- "bench_tests::baseline_llama --exact --include-ignored"' \
  "$(run baseline qwen3-0.6b) $(cat "$FAKE_RUNNER_LOG")"
check "baseline: the GGUF's digest is recorded; same weights not established without its source" \
  '[ "$(jq -r ".tiers[] | select(.name == \"baseline\") | .runs[0] | [.gguf.sha256 == \"$(sha256sum < "$T/q.gguf" | cut -c1-64)\", .comparable, .not_comparable_because[0]] | @tsv" "$V")" = "$(printf "true\tfalse\tthe GGUF'"'"'s source is not given (BENCH_GGUF_QWEN3_0_6B_FROM)")" ]' \
  "$(jq -c '.tiers[] | select(.name == "baseline") | .runs[0] | del(.result)' "$V")"
check "the report keeps the baseline apart, without a ratio" 'grep -q "not compared" "$REPO/out/bench/$SHA.md"'
bench BENCH_STEPS="build core baseline" BENCH_LLAMA_SERVER="$T/llama/bin/llama-server" BENCH_GGUF_QWEN3_0_6B="$T/q.gguf" BENCH_GGUF_QWEN3_0_6B_FROM="$T/m06"
check "baseline: comparable when the GGUF comes from the same directory, BF16, the same ids" \
  '[ "$(jq -r ".tiers[] | select(.name == \"baseline\") | .runs[0].comparable" "$V")" = true ] && grep -q "llama.cpp / x0ne Core" "$REPO/out/bench/$SHA.md"' \
  "$(jq -c '.tiers[] | select(.name == "baseline") | .runs[0] | del(.result)' "$V")"

# ---- failures: a program that fails, one with no result, a lost card.
bench FAKE_RUNNER_RC=engine-qwen3-0.6b:101 BENCH_STEPS="build core engine load"
check "a program that fails on the card fails its run with its code; the next runs still run" \
  '[ "$RC" = 1 ] && [ "$(run engine qwen3-0.6b)" = "fail|exit 101" ] && [ "$(run load qwen3-0.6b)" = "ok|" ]' "$(run engine qwen3-0.6b)"
bench FAKE_NO_RESULT=core-qwen3-0.6b BENCH_STEPS="build core"
check "a program that prints no result fails its run" '[ "$RC" = 1 ] && [ "$(run core qwen3-0.6b)" = "fail|no x0ne-bench-result line on its output" ]' "$(run core qwen3-0.6b)"
cp "$T/res-engine.json" "$T/res-core.json"
bench BENCH_STEPS="build core"
check "a result of another kind fails its run" '[ "$RC" = 1 ] && [ "$(run core qwen3-0.6b)" = "fail|not a core result (schema 1, kind core)" ]' "$(run core qwen3-0.6b)"
results
jq -c '. + {refused: "made-up out of memory at every context"}' "$T/res-core.json" > "$T/res-refused.json"
mv "$T/res-refused.json" "$T/res-core.json"
bench BENCH_STEPS="build core"
check "a model the card cannot hold is skipped with why, never failed nor passed" \
  '[ "$RC" = 3 ] && [ "$(run core qwen3-0.6b)" = "skipped|refused on this card: made-up out of memory at every context" ]' "$(run core qwen3-0.6b)"
results
bench FAKE_RUNNER_RC=core-qwen3-0.6b:199
check "SECURITY: a lost card (199) stops the run: red, every later run skipped, the runner not called again" \
  '[ "$RC" = 1 ] && grep -q "^card lost" <<<"$(jq -r .stopped "$V")" && grep -q "^fail|card lost" <<<"$(run core qwen3-0.6b)" && [ "$(run engine qwen3-0.6b)" = "skipped|card lost: not run" ] && [ "$(tail -n 1 "$FAKE_RUNNER_LOG" | grep -c -- "-o $RL/core-qwen3-0.6b ")" = 1 ] && [ "$(wc -l < "$FAKE_RUNNER_LOG")" = 2 ]' \
  "$(jq -c .stopped "$V") $(cat "$FAKE_RUNNER_LOG")"
bench FAKE_RUNNER_RC=probe:125
check "a runner that refuses the probe: every tier skipped, no GPU, never green" \
  '[ "$RC" = 3 ] && [ "$(jq -r ".machine.gpu" "$V")" = null ] && [ "$(run core qwen3-0.6b)" = "skipped|no GPU: runner exit 125: fake-runner: refused: probe exits 125" ] && [ "$(wc -l < "$FAKE_RUNNER_LOG")" = 1 ]' \
  "$(run core qwen3-0.6b)"
bench BENCH_STEPS="build"
check "a run without a GPU step does not boot the card" '[ "$RC" = 0 ] && [ ! -s "$FAKE_RUNNER_LOG" ] && [ "$(jq -r .status "$V")" = green ]' "$(cat "$FAKE_RUNNER_LOG")"
check "the build is recorded apart from the tiers" '[ "$(jq -r "[.build.status, (.tiers | length)] | @tsv" "$V")" = "$(printf "ok\t0")" ]' "$(jq -c '{build, tiers}' "$V")"

# ---- the soak's length and the runner's limits.
bench BENCH_STEPS="build soak" BENCH_SOAK_MIN=60 BENCH_RUNNER_MEM=12288 BENCH_RUNNER_ARGS="-c 8"
check "an hour of soak: the runner gets 3600 s and half an hour more, the memory and options given" \
  'called soak-qwen3-0.6b | grep -qF -- "-m 12288 -T 5400 -c 8 -o $RL/soak-qwen3-0.6b" && called soak-qwen3-0.6b | grep -qF -- "X0NE_BENCH_SOAK_SECS=3600"' "$(cat "$FAKE_RUNNER_LOG")"

# ---- without a runner: containers, the GPU flag only where a GPU is.
RC=0; : > "$FAKE_LOG"
OUT=$("${CLEAN[@]}" PATH="$BIN:$PATH" BENCH_IMAGE=fake-image BENCH_VENDOR="$T/vendor" BENCH_TARGET="$T/build" \
  BENCH_QWEN3_0_6B="$T/m06" BENCH_PROMPTS="$T/prompts" BENCH_STEPS="build core" bash "$REPO/scripts/bench.sh" 2>&1) || RC=$?
check "no runner and no GPU in the image: the GPU tiers are skipped, never green" \
  '[ "$RC" = 3 ] && [ "$(run core qwen3-0.6b)" = "skipped|no GPU" ] && ! grep -q "x0ne-bench-core-" "$FAKE_LOG"' "$(run core qwen3-0.6b)"
RC=0; : > "$FAKE_LOG"
OUT=$("${CLEAN[@]}" PATH="$BIN:$PATH" FAKE_GPU=1 BENCH_IMAGE=fake-image BENCH_VENDOR="$T/vendor" BENCH_TARGET="$T/build" \
  BENCH_QWEN3_0_6B="$T/m06" BENCH_PROMPTS="$T/prompts" BENCH_STEPS="build core" bash "$REPO/scripts/bench.sh" 2>&1) || RC=$?
check "a GPU in the image: core-bench runs in a container with the device, the model read-only" \
  '[ "$(run core qwen3-0.6b)" = "ok|" ] && grep "x0ne-bench-core-qwen3-0.6b-" "$FAKE_LOG" | grep -q -- "--device nvidia.com/gpu=all .*-v $T/m06:/data/model:ro .*/target/core/release/examples/core-bench /data/model"' \
  "$(grep x0ne-bench-core "$FAKE_LOG")"

# ---- a tree that is not its commit's.
echo change >> "$REPO/scripts/bench.sh"
bench BENCH_STEPS="build"
check "a changed tree writes <sha>+dirty.json, marked dirty" '[ "$(jq -r "[.dirty, .short] | @tsv" "$REPO/out/bench/$SHA+dirty.json")" = "$(printf "true\t%s+dirty" "${SHA:0:7}")" ]'
RC=0; UOUT=$(bash "$REPO/scripts/bench.sh" update "$REPO/out/bench/$SHA+dirty.json" 2>&1) || RC=$?
check "update refuses the verdict of a changed tree: a reference names a commit" '[ "$RC" = 2 ] && grep -q "dirty" <<<"$UOUT"' "$UOUT"
git -C "$REPO" checkout -q -- scripts/bench.sh

# ---- the reference file of the tree.
check "the tree's reference is a reference: tolerances per tier, runs recorded on a GPU only" \
  'jq -e ".schema == 1 and ([.tolerance_pct[]] | all(. > 0)) and (.tolerance_pct | keys) == [\"core\",\"engine\",\"load\",\"soak\"] and (.runs | type) == \"object\" and ([.runs[] | (.sha | test(\"^[0-9a-f]{40}$\")) and (.metrics | length > 0)] | all)" "$SRC/ci/bench-baseline.json" > /dev/null'

[ "$FAILS" = 0 ] || { echo "bench: $FAILS failed"; exit 1; }
echo "bench: all passed"
