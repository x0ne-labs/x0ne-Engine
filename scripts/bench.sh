#!/usr/bin/env bash
# make bench: x0ne Engine measured on the local NVIDIA GPU, tier by tier, into a verdict file and a
# Markdown report, held to the versioned reference ci/bench-baseline.json. docs/benchmarks.md
# ("The suite") describes the tiers, the verdict and what is not measured.
#
#   bench.sh [run]                        the steps of BENCH_STEPS
#   bench.sh report VERDICT               the Markdown report of a verdict, on stdout
#   bench.sh update VERDICT [--replace]   record a clean run's metrics in ci/bench-baseline.json
#
# Steps (BENCH_STEPS, default "build core engine load soak baseline"), each a tier of the verdict:
#   build     in a container, with the real nvcc: x0ne-infer's lib test binary (release, --features
#             cuda: the server, Core, bench_tests.rs and its client bench_load_tests.rs) and
#             Core's core-bench example. Nothing runs on the card.
#   core      x0ne Core alone (core-bench), per model: cold load, then per length of BENCH_LENGTHS
#             (BENCH_MOE_LENGTHS for a mixture of experts) the prefill and decode rates and the time
#             to the first token, "max" being the largest context the card accepts; the arena and
#             the VRAM taken. BENCH_REPS repetitions after a first pass reported apart.
#   engine    the same prompts through x0ne-infer's server on Core, in the test mode of its unit
#             tests (a test proof: no TDX), streamed and whole, at the ticks of BENCH_TICKS.
#   load      BENCH_CLIENTS clients against that server, BENCH_LOAD_SECS seconds each: TTFT and
#             response p50/p95/p99, throughput, 503 and Retry-After, fairness, FIFO inversions.
#   soak      BENCH_SOAK_CLIENTS clients for BENCH_SOAK_MIN minutes: throughput per window and its
#             drift, host memory, VRAM, errors.
#   baseline  llama.cpp's llama-server on the same ids (BENCH_LLAMA_SERVER, BENCH_GGUF_<MODEL>).
#             Compared with core only when the GGUF is known to come from the model's directory
#             (BENCH_GGUF_<MODEL>_FROM), is BF16 and was given the same ids. Never a reference.
# Every program on the card prints one line "x0ne-bench-result: <JSON>" (schema 1, kind: the tier).
#
# Models: BENCH_QWEN3_0_6B, BENCH_QWEN3_8B, BENCH_QWEN3_14B, BENCH_QWEN3_32B, BENCH_QWEN3_30B_A3B
# (directories: config.json, *.safetensors, tokenizer.json for the server), BENCH_EXTRA_MODELS
# (directories, each named by its last component). A model not given is skipped, reason "BENCH_X
# is not set". The server's tiers skip what the server cannot serve (MoE, no tokenizer.json), with
# why. load, soak and baseline measure BENCH_LOAD_MODELS, BENCH_SOAK_MODELS and
# BENCH_BASELINE_MODELS (names, default qwen3-0.6b).
#
# The verdict: out/bench/<sha>.json (<sha>+dirty.json for a tree that is not its commit's) and
# beside it <id>.md; latest.json and latest.md point at them. A run is ok, fail or skipped. A
# measured run is held to the reference of its GPU: a rate (*_tok_s, *_per_s) lower, or a time
# higher, than the reference by more than its tier's tolerance fails it, reason "below the
# reference". Status: red (a run failed, or the card was lost: exit 1), partial (a run skipped,
# or measured without a reference: exit 3; never green), green (exit 0). 2: usage.
#
# BENCH_RUNNER: where the card is held by vfio-pci for a microVM, each program goes through a
# launcher that runs ONE program on the card (x0ne OS's scripts/microvm-gpu-run.sh), as
# make core-gpu does with CORE_GPU_RUNNER: the GPU is what the guest's nvidia-smi names; 125 fails
# the run; 199 (the card no longer answers) stops everything, every later run skipped, nothing
# retried. Without a runner, each program runs in a container given the GPU by BENCH_PODMAN_ARGS.
#
# Environment (a CORE_GPU_* variable stands in for its BENCH_* twin when that one is not set):
#   BENCH_IMAGE (CORE_GPU_IMAGE)    a CUDA devel image with nvcc and Rust on PATH (required)
#   BENCH_VENDOR (CORE_GPU_VENDOR)  a `cargo vendor` directory of Cargo.lock (required: no network)
#   BENCH_TARGET (CORE_GPU_TARGET)  the build directory, default target/core-gpu: make core-gpu's
#   BENCH_OUT                       default out/bench
#   BENCH_PODMAN_ARGS (CORE_GPU_PODMAN_ARGS)   default --device nvidia.com/gpu=all
#   BENCH_RUNNER (CORE_GPU_RUNNER), BENCH_RUNNER_ARGS (CORE_GPU_RUNNER_ARGS)   the launcher above
#   BENCH_RUNNER_MEM                the guest's memory per program, MiB (16384)
#   BENCH_RUNNER_TIMEOUT            seconds per program (3600); the soak gets its length + 1800
#   BENCH_PROMPTS (CORE_GPU_PROMPTS)   the prompt kit: <name>.ids.json, <name>.chat.json
#   BENCH_REPS (5), BENCH_DECODE (256 tokens), BENCH_LENGTHS ("short 1k 4k 8k 16k max"),
#   BENCH_MOE_LENGTHS ("short 1k 4k"), BENCH_ENGINE_PROMPTS ("short 4k 8k 16k"),
#   BENCH_TICKS ("production short"), BENCH_CLIENTS ("1 2 4 8 16 32"), BENCH_LOAD_SECS (30),
#   BENCH_LOAD_PROMPT (short), BENCH_LOAD_TOKENS (256), BENCH_MAX_QUEUE (8), BENCH_SOAK_MIN (10),
#   BENCH_SOAK_CLIENTS (4), BENCH_SOAK_WINDOW (60 s)
#   BENCH_LLAMA_SERVER              a llama-server built with CUDA for this GPU; its directory's
#                                   parent (BENCH_LLAMA_ROOT) is bound, BENCH_LLAMA_LIB its libraries
#   BENCH_GGUF_<MODEL>, BENCH_GGUF_<MODEL>_FROM   a GGUF of that model, the directory it was made from
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REF=$ROOT/ci/bench-baseline.json

usage() { echo "bench: $*" >&2; exit 2; }
command -v jq > /dev/null || usage "jq not found (the verdict is JSON)"

# ---- the Markdown report, from the verdict alone.
read -r -d '' REPORT_JQ <<'JQ' || true
def num:
  if type != "number" then "–"
  elif . == 0 then "0"
  else fabs as $a
    | if $a >= 100 then round
      elif $a >= 10 then (. * 10 | round) / 10
      elif $a >= 1 then (. * 100 | round) / 100
      else pow(10; 2 - ($a | log10 | floor)) as $p | (. * $p | round) / $p
      end
    | tostring
  end;
def st: if type == "object" then "\(.median | num) (\(.min | num)–\(.max | num), sd \(.stddev | num))" else "–" end;
def gib: if type == "number" then "\(. / 1073741824 | num) GiB" else "–" end;
def runs($t): [.tiers[]? | select(.name == $t) | .runs[]];
def what: if . == null then "" else " (\(.model_type // "?"), \(.layers // "?") layers\(if .experts then ", \(.experts) experts" else "" end)\(if .truncated then ", cut: \(.truncated)" else "" end))" end;
. as $v
| (runs("core") | map(select(.status != "skipped" and .result != null) | {key: .model, value: .result}) | from_entries) as $coreres
| [
  "# make bench \(.short)",
  "",
  "Status: **\(.status)**\(if .stopped then ": " + .stopped else "" end). Commit `\(.sha)` on \(.branch), tree \(.tree)\(if .dirty then ", **changed from its commit**" else "" end); from \(.started) to \(.ended), \(.duration_s) s.",
  "",
  "## Machine",
  "",
  "- GPU: \(if .machine.gpu then "\(.machine.gpu.name), driver \(.machine.gpu.driver), \(.machine.gpu.memory)" else "none seen" end)",
  "- Host: \(.machine.host.cpu), \(.machine.host.threads) threads, \(.machine.host.mem_mib) MiB, Linux \(.machine.host.kernel)",
  (if .machine.runner then "- Runner: \(.machine.runner.path) (sha256 \(.machine.runner.sha256[0:16])), one program per microVM of \(.machine.runner.mem_mib) MiB"
   else "- No runner: each program in a container, podman \(.machine.podman_args)" end),
  "- Image: \(.machine.image)",
  "- Each number is the median of \(.settings.reps) repetitions, after a first pass reported apart, with the smallest and the largest and the standard deviation; \(.settings.decode_tokens) tokens decoded, greedy. Numbers of this machine and this run only.",
  (if .build then "- Build: \(.build.status)\(if .build.reason then ": " + .build.reason else "" end), \(.build.seconds) s" else empty end),
  (runs("core") | map(select(.result)) | if length > 0 then
    ["", "## Core: x0ne Core alone", "",
     "Rates in tokens per second, times in seconds. The ids are the kit's `<name>.ids.json` (`1k`: the first 1024 of `4k`); `max` and a length without a file: ids drawn from a fixed seed.",
     "", "| model | length | tokens | decode tok/s | prefill tok/s | TTFT s | first pass: decode / prefill / TTFT |", "|---|---|---|---|---|---|---|",
     (.[] | .model as $m | .result.lengths[]? |
       if .status == "ok" then "| \($m) | \(.name) | \(.tokens) | \(.decode_tok_s | st) | \(.prefill_tok_s | st) | \(.ttft_s | st) | \(.first.decode_tok_s | num) / \(.first.prefill_tok_s | num) / \(.first.ttft_s | num) |"
       else "| \($m) | \(.name) | \(.tokens // "–") | \(.status): \(.reason // "") | | | |" end),
     "",
     (.[] | .result as $r | "- \(.model)\(.about | what): loaded in \($r.load.cold_s | num) s cold, \($r.load.reload_s | num) s again; context of \($r.max_ctx) tokens; arena \($r.memory.arena_bytes | gib), \($r.memory.taken_at_load_bytes | gib) taken by the load, \($r.memory.peak_used_bytes | gib) used at the peak of \($r.memory.device_total_bytes | gib). \($r.engine)")]
   else [] end),
  (runs("engine") | map(select(.result)) | if length > 0 then
    ["", "## Engine: the server on Core, in its test mode", "",
     "The server of x0ne-infer's unit tests on x0ne Core, its client in the same guest over loopback: the kit's chat prompts, streamed and whole. A stream sends one frame per tick; a whole answer leaves at a tick.",
     "**Not measured here**: TDX (no trust domain, no quote, no private memory), the relay, the network between a client and the server.",
     "", "| model | prompt | tick | prompt tokens | stream TTFT s | stream decode tok/s | stream end s | whole response s | TTFT overhead over Core s |", "|---|---|---|---|---|---|---|---|---|",
     (.[] | .model as $m | .result.cases[]? | . as $c
       | if $c.skipped then "| \($m) | \($c.prompt) | \($c.tick) | \($c.prompt_tokens // "–") | skipped: \($c.skipped) | | | | |"
         else
           ((($coreres[$m].lengths // []) | map(select(.name == $c.prompt and .status == "ok")) | .[0].ttft_s.median)) as $ct
           | "| \($m) | \($c.prompt) | \($c.tick) (\($c.tick_ms) ms) | \($c.prompt_tokens // "–") | \($c.stream.ttft_s | st) | \($c.stream.decode_tok_s | st) | \($c.stream.end_s | st) | \($c.whole.e2e_s | st) | \(if ($ct | type) == "number" and ($c.stream.ttft_s.median | type) == "number" then ($c.stream.ttft_s.median - $ct | num) else "–" end) |"
         end),
     "", "TTFT overhead over Core: the stream's median time to its first token minus Core's for the prompt of the same name (its ids): HTTP, the chat template and the tokenizer, admission, and the wait for the next tick."]
   else [] end),
  (runs("load") | map(select(.result)) | if length > 0 then
    ["", "## Load", "",
     (.[0].result | "One server (\(.max_queue) waiting at most, 503 with Retry-After past them); each client sends its next request when its answer ends, for \(.level_s) s per level; prompt `\(.prompt)`, \(.max_tokens) tokens, tick \(.tick). Jain's index over the answers per client (1: even). FIFO inversions: pairs of admitted requests whose first tokens came in the other order than their sending."),
     "", "| model | clients | requests | answered | 503 (share) | Retry-After s | TTFT p50 / p95 / p99 s | response p50 / p95 / p99 s | tok/s | answers/s | Jain | FIFO inversions | errors |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
     (.[] | .model as $m | .result.levels[]? |
       "| \($m) | \(.clients) | \(.requests) | \(.ok) | \(.rejected) (\(.rejected_frac * 100 | num) %) | \(if .retry_after_s then "\(.retry_after_s.min | num)–\(.retry_after_s.max | num)" else "–" end) | \(.ttft_s.p50 | num) / \(.ttft_s.p95 | num) / \(.ttft_s.p99 | num) | \(.e2e_s.p50 | num) / \(.e2e_s.p95 | num) / \(.e2e_s.p99 | num) | \(.throughput_tok_s | num) | \(.ok_per_s | num) | \(.jain_index | num) | \(.fifo_inversions) of \(.fifo_pairs) | \(.errors) |")]
   else [] end),
  (runs("soak") | map(select(.result)) | if length > 0 then
    ["", "## Soak", "",
     "| model | minutes | clients | answers | errors | 503 | tok/s | drift % | slope %/h | RSS first / last / peak MiB | VRAM first / last MiB |", "|---|---|---|---|---|---|---|---|---|---|---|",
     (.[] | .model as $m | .result as $r | $r.summary |
       "| \($m) | \($r.seconds / 60 | num) | \($r.clients) | \(.ok) | \(.errors) | \(.rejected) | \(.throughput_tok_s | num) | \(.throughput_drift_pct | num) | \(.throughput_slope_pct_per_h | num) | \(.rss_first_kib / 1024 | num) / \(.rss_last_kib / 1024 | num) / \(.rss_peak_kib / 1024 | num) | \(.vram_first_mib | num) / \(.vram_last_mib | num) |"),
     "", "Drift: the last window's throughput against the first's (windows of \(.[0].result.window_s) s). Slope: the least-squares line through every window, per hour, as a share of their mean."]
   else [] end),
  (runs("baseline") | map(select(.result)) | if length > 0 then
    ["", "## Baseline: llama.cpp", "",
     (.[] | . as $b | ($coreres[$b.model].lengths // []) as $cl
       | "- \($b.model): llama-server \($b.result.server_version // "?"), GGUF \($b.result.file_type // "?") (sha256 \($b.gguf.sha256[0:16]))"
           + (if $b.comparable then ": the same weights, precision and ids as core." else ": **not compared**: \($b.not_comparable_because | join("; "))." end),
         "",
         (if $b.comparable then "| model | length | tokens | llama.cpp decode tok/s | llama.cpp prefill tok/s | llama.cpp / x0ne Core: decode | llama.cpp / x0ne Core: prefill |"
          else "| model | length | tokens | llama.cpp decode tok/s | llama.cpp prefill tok/s |" end),
         (if $b.comparable then "|---|---|---|---|---|---|---|" else "|---|---|---|---|---|" end),
         ($b.result.lengths[]? | . as $l
           | if $l.status != "ok" then "| \($b.model) | \($l.name) | \($l.tokens // "–") | \($l.status): \($l.reason // "") | |\(if $b.comparable then " | |" else "" end)"
             else "| \($b.model) | \($l.name) | \($l.tokens) | \($l.decode_tok_s | st) | \($l.prefill_tok_s | st) |"
               + (if $b.comparable then
                    (($cl | map(select(.name == $l.name and .status == "ok" and .tokens == $l.tokens and .source == $l.source)) | .[0]) as $c
                    | if $c then " \($l.decode_tok_s.median / $c.decode_tok_s.median | num) | \($l.prefill_tok_s.median / $c.prefill_tok_s.median | num) |"
                      else " other ids than core's | |" end)
                  else "" end)
             end),
         "")]
   else [] end),
  "",
  "## Reference: ci/bench-baseline.json",
  "",
  ([.tiers[] | .name as $t | .runs[] | select(.ratchet and (.ratchet.lines | type) == "array") | . as $r
    | "- \($t)/\(.model): \(.ratchet.status) (\(.ratchet.lines | map(select(.state == "held")) | length) held, \(.ratchet.lines | map(select(.state == "new")) | length) without a reference, tolerance \(.ratchet.tolerance_pct) %)",
      (.ratchet.lines[] | select(.state == "red") | "  - RED \(.metric): \(.value | num) against \(.reference | num) (\(.pct | num) %)")]
   | if length == 0 then ["Nothing measured to hold."] else . end),
  ([(if .build and .build.status != "ok" then "- build, \(.build.status): \(.build.reason // "")" else empty end),
    (.tiers[] | .name as $t | .runs[] | select(.status != "ok") | "- \($t)/\(.model), \(.status): \(.reason // "")")]
   | if length > 0 then ["", "## Not passed", ""] + . else [] end)
  ] | flatten | join("\n")
JQ

# ---- the checks a result must pass whatever the reference: a reason, empty when none failed.
read -r -d '' CHECKS_JQ <<'JQ' || true
if $t == "core" then
  [.lengths[]? | if .status == "error" then "\(.name): \(.reason // "error")"
                 elif .status == "ok" and .same_tokens != true then "\(.name): the repetitions gave different tokens"
                 else empty end]
elif $t == "engine" then
  [.cases[]? | . as $c | ((.stream.errors // []) + (.whole.errors // [])) | select(length > 0)
   | "\($c.prompt)/\($c.tick): \(length) errors (\(.[0:3] | join(", ")))"]
elif $t == "load" then
  [.levels[]? | (if (.errors // 0) > 0 then "\(.clients) clients: \(.errors) errors (\(.error_samples[0:3] | join(", ")))" else empty end),
                (if (.rejected_without_retry_after // 0) > 0 then "\(.clients) clients: \(.rejected_without_retry_after) refusals without Retry-After" else empty end)]
elif $t == "soak" then
  [if (.summary.errors // 0) > 0 then "\(.summary.errors) errors during the soak" else empty end]
elif $t == "baseline" then
  [.lengths[]? | select(.status == "error") | "\(.name): \(.reason // "error")"]
else [] end | join("; ")
JQ

# ---- the verdict, from the runs recorded, the reference and the machine.
read -r -d '' VERDICT_JQ <<'JQ' || true
def higher: test("_(tok|per)_s$");
def nm: "\(.tier)\(if .model then "/" + .model else "" end)";
def metrics($t; $m):
  if $t == "core" then
    [.lengths[]? | select(.status == "ok") | . as $l
     | ("decode_tok_s", "prefill_tok_s", "ttft_s") | {key: "core/\($m)/\($l.name)/\(.)", value: $l[.].median}]
  elif $t == "engine" then
    [.cases[]? | select(.skipped == null) | . as $c | "engine/\($m)/\($c.prompt)/\($c.tick)/" as $p
     | {key: ($p + "stream_ttft_s"), value: $c.stream.ttft_s.median},
       {key: ($p + "stream_decode_tok_s"), value: $c.stream.decode_tok_s.median},
       {key: ($p + "whole_e2e_s"), value: $c.whole.e2e_s.median}]
  elif $t == "load" then
    [.levels[]? | {key: "load/\($m)/c\(.clients)/throughput_tok_s", value: .throughput_tok_s},
                  {key: "load/\($m)/c\(.clients)/ttft_p50_s", value: .ttft_s.p50}]
  elif $t == "soak" then [{key: "soak/\($m)/throughput_tok_s", value: .summary.throughput_tok_s}]
  else [] end
  | map(select(.value | type == "number")) | from_entries;
def ratchet($ms; $held; $tol):
  [$ms | to_entries[] | .key as $k | .value as $x | ($held // {})[$k] as $r
   | if ($r | type) != "number" then {metric: $k, state: "new", value: $x}
     else (if $r == 0 then 0 else ($x - $r) / $r * 100 end) as $pct
       | {metric: $k,
          state: (if ($k | higher) then (if $pct < (0 - $tol) then "red" else "held" end)
                  else (if $pct > $tol then "red" else "held" end) end),
          value: $x, reference: $r, pct: ($pct * 100 | round / 100)}
     end] as $lines
  | {status: (if ($lines | length) == 0 then "nothing measured"
              elif any($lines[]; .state == "red") then "red"
              elif all($lines[]; .state == "new") then "no reference"
              elif any($lines[]; .state == "new") then "partly new"
              else "held" end),
     tolerance_pct: $tol, lines: $lines};
def comparable($all):
  . as $b
  | ($all | map(select(.tier == "core" and .model == $b.model and .status != "skipped" and .result != null)) | .[0].result) as $core
  | [(if $b.gguf.from == null then "the GGUF's source is not given (\($b.gguf.var)_FROM)"
      elif $b.gguf.from_real != $b.model_dir then "the GGUF was made from \($b.gguf.from), not from \($b.model_dir)"
      else empty end),
     (if ($b.result.file_type // null) != "BF16" then "the GGUF is \($b.result.file_type // "of a type not read"), not BF16" else empty end),
     (if $b.weights_dtype != null and $b.weights_dtype != "bfloat16" then "the model's weights are \($b.weights_dtype), not bfloat16" else empty end),
     (if $core == null then "no core run of \($b.model) in this verdict"
      elif $core.decode_tokens != $b.result.decode_tokens then "core decoded \($core.decode_tokens) tokens, llama.cpp \($b.result.decode_tokens)"
      elif ([$b.result.lengths[]? | select(.status == "ok") | . as $l
             | $core.lengths[]? | select(.name == $l.name and .status == "ok" and .tokens == $l.tokens and .source == $l.source)] | length) == 0
        then "no length measured by both on the same ids"
      else empty end)] as $why
  | . + {comparable: ($why | length == 0), not_comparable_because: $why};
($ref[0] // {}) as $R
| (if $gpu == null then null else ($R.runs[$gpu.name].metrics // null) end) as $held
| [$runs[] | . as $r
   | if ($r.tier | IN("core", "engine", "load", "soak")) and $r.status == "ok" and $r.result != null then
       ($r.result | metrics($r.tier; $r.model)) as $ms
       | ratchet($ms; $held; ($R.tolerance_pct[$r.tier] // 0)) as $rt
       | $r + {metrics: $ms, ratchet: $rt}
       | if $rt.status == "red" then .status = "fail" | .reason = "below the reference" else . end
     elif $r.tier == "baseline" and $r.status == "ok" then $r + {ratchet: {status: "not held", why: "llama.cpp is a comparison, not a reference"}}
     else $r end] as $rs1
| [$rs1[] | if .tier == "baseline" and .status == "ok" then comparable($rs1) else . end] as $rs
| ($rs | map(select(.tier == "build") | del(.tier, .model, .about, .result)) | .[0] // null) as $build
| ($rs | map(select(.status == "fail") | nm)) as $failed
| ($rs | map(select(.status == "skipped") | nm)) as $skipped
| ($rs | map(select(.status == "ok" and ((.ratchet.status // "") | IN("no reference", "partly new", "nothing measured"))) | nm)) as $unref
| {schema: 1,
   status: (if ($failed | length) > 0 or $stopped != "" then "red"
            elif ($skipped | length) > 0 or ($unref | length) > 0 then "partial" else "green" end),
   sha: $sha, short: ($sha[0:7] + (if $dirty == "1" then "+dirty" else "" end)), dirty: ($dirty == "1"),
   tree: $tree, branch: $branch, started: $started, ended: $ended, duration_s: $duration,
   machine: {gpu: $gpu, host: $host, runner: $runner, image: $image, podman_args: $podman_args},
   settings: $settings,
   reference: {file: "ci/bench-baseline.json", recorded_for_this_gpu: ($held != null),
               sha: (if $gpu == null then null else ($R.runs[$gpu.name].sha // null) end)},
   stopped: (if $stopped == "" then null else $stopped end),
   failed: $failed, skipped: $skipped, unreferenced: $unref, log_dir: $log, build: $build,
   tiers: [$steps[] | select(. != "build") as $s | {name: $s, runs: [$rs[] | select(.tier == $s) | del(.tier)]}]}
JQ

report() { jq -r "$REPORT_JQ" "$1"; }

# update VERDICT [--replace]: the verdict's metrics into the reference of its GPU. A metric already
# held is only raised (the better of the two medians), and only when the run holds the reference;
# --replace writes the run's metrics as they are. A dirty tree, a failed run: refused.
update() {
  local v=$1 replace=0 gpu failed red n
  [ "${2:-}" != --replace ] || replace=1
  [ -f "$v" ] || usage "no verdict $v"
  jq -e '.schema == 1 and (.tiers | type) == "array"' "$v" > /dev/null 2>&1 || usage "$v is not a verdict of bench.sh (schema 1)"
  if jq -e '.dirty' "$v" > /dev/null; then
    usage "$v is the verdict of a dirty tree ($(jq -r .short "$v")): a reference names a commit; commit, run again, record that run"
  fi
  gpu=$(jq -r '.machine.gpu.name // empty' "$v")
  [ -n "$gpu" ] || usage "$v saw no GPU: nothing to record"
  failed=$(jq -r '[.tiers[] | .name as $t | .runs[] | select(.status == "fail" and .reason != "below the reference")
    | "\($t)\(if .model then "/" + .model else "" end): \(.reason)"] | join("; ")' "$v")
  if [ -n "$failed" ]; then echo "bench: failed in $v: $failed. A failed run is not a reference" >&2; exit 1; fi
  red=$(jq -r '[.tiers[].runs[].ratchet.lines[]? | select(.state == "red") | .metric] | join(", ")' "$v")
  if [ -n "$red" ] && [ "$replace" = 0 ]; then
    echo "bench: below the reference in $v: $red. Not recorded (--replace writes the run as it is)" >&2
    exit 1
  fi
  n=$(jq '[.tiers[].runs[] | select(.status == "ok" or .reason == "below the reference") | .metrics // {} | keys[]] | length' "$v")
  if [ "$n" = 0 ]; then echo "bench: nothing measured in $v: nothing recorded" >&2; exit 1; fi
  jq --slurpfile v "$v" --argjson replace "$replace" '
    def higher: test("_(tok|per)_s$");
    $v[0] as $v
    | ([$v.tiers[].runs[] | select(.status == "ok" or .reason == "below the reference") | .metrics // {}] | add // {}) as $new
    | (.runs[$v.machine.gpu.name].metrics // {}) as $old
    | ($new | with_entries(.key as $k | .value as $x | $old[$k] as $o
        | .value = (if $replace == 1 or ($o | type) != "number" then $x
                    elif ($k | higher) then ([$o, $x] | max) else ([$o, $x] | min) end))) as $up
    | .runs[$v.machine.gpu.name] = {date: $v.ended, sha: $v.sha, branch: $v.branch,
        driver: $v.machine.gpu.driver, memory: $v.machine.gpu.memory,
        runner_sha256: ($v.machine.runner.sha256 // null), image: $v.machine.image,
        reps: $v.settings.reps, decode_tokens: $v.settings.decode_tokens, metrics: ($old + $up)}' \
    "$REF" > "$REF.tmp"
  mv "$REF.tmp" "$REF"
  echo "bench: $n metrics of $gpu from $(jq -r .short "$v") in ci/bench-baseline.json: commit it with the change it measures"
}

case ${1:-run} in
  report) [ $# = 2 ] && [ -f "$2" ] || usage "usage: bench.sh report VERDICT"; report "$2"; exit 0 ;;
  update) [ $# -ge 2 ] && [ $# -le 3 ] || usage "usage: bench.sh update VERDICT [--replace]"; update "$2" "${3:-}"; exit 0 ;;
  run) ;;
  *) usage "usage: bench.sh [run | report VERDICT | update VERDICT [--replace]]" ;;
esac

# ---- the settings, all checked before anything runs.
steps=${BENCH_STEPS:-build core engine load soak baseline}
seen=" "
for step in $steps; do
  case $step in
    build|core|engine|load|soak|baseline) ;;
    *) usage "unknown step $step (build, core, engine, load, soak, baseline)" ;;
  esac
  case $seen in *" $step "*) usage "step $step given twice" ;; esac
  seen+="$step "
done
whole() { # whole VAR VALUE MIN: VALUE, a whole number not below MIN, or usage
  [[ $2 =~ ^[0-9]+$ ]] && [ "$2" -ge "$3" ] || usage "$1 must be a whole number not below $3, not \"$2\""
  printf '%s' "$2"
}
words() { # words VAR VALUE PATTERN: VALUE's words, each matching PATTERN, one space between
  local w ws=()
  for w in $2; do [[ $w =~ $3 ]] || usage "$1: \"$w\" is not allowed"; ws+=("$w"); done
  [ "${#ws[@]}" -gt 0 ] || usage "$1 is empty"
  printf '%s' "${ws[*]}"
}
reps=$(whole BENCH_REPS "${BENCH_REPS:-5}" 1)
decode=$(whole BENCH_DECODE "${BENCH_DECODE:-256}" 2)
level_s=$(whole BENCH_LOAD_SECS "${BENCH_LOAD_SECS:-30}" 1)
load_tokens=$(whole BENCH_LOAD_TOKENS "${BENCH_LOAD_TOKENS:-256}" 1)
max_queue=$(whole BENCH_MAX_QUEUE "${BENCH_MAX_QUEUE:-8}" 1)
soak_min=$(whole BENCH_SOAK_MIN "${BENCH_SOAK_MIN:-10}" 1)
soak_clients=$(whole BENCH_SOAK_CLIENTS "${BENCH_SOAK_CLIENTS:-4}" 1)
soak_window=$(whole BENCH_SOAK_WINDOW "${BENCH_SOAK_WINDOW:-60}" 1)
runner_mem=$(whole BENCH_RUNNER_MEM "${BENCH_RUNNER_MEM:-16384}" 1)
runner_secs=$(whole BENCH_RUNNER_TIMEOUT "${BENCH_RUNNER_TIMEOUT:-3600}" 1)
soak_s=$((soak_min * 60))
name_re='^[A-Za-z0-9][A-Za-z0-9_.-]*$'
lengths=$(words BENCH_LENGTHS "${BENCH_LENGTHS:-short 1k 4k 8k 16k max}" "$name_re")
moe_lengths=$(words BENCH_MOE_LENGTHS "${BENCH_MOE_LENGTHS:-short 1k 4k}" "$name_re")
engine_prompts=$(words BENCH_ENGINE_PROMPTS "${BENCH_ENGINE_PROMPTS:-short 4k 8k 16k}" "$name_re")
ticks=$(words BENCH_TICKS "${BENCH_TICKS:-production short}" '^(production|short)$')
clients=$(words BENCH_CLIENTS "${BENCH_CLIENTS:-1 2 4 8 16 32}" '^[1-9][0-9]*$')
load_prompt=$(words BENCH_LOAD_PROMPT "${BENCH_LOAD_PROMPT:-short}" "$name_re")
image=${BENCH_IMAGE:-${CORE_GPU_IMAGE:-}}
[ -n "$image" ] || usage "set BENCH_IMAGE (or CORE_GPU_IMAGE) to a CUDA devel image with nvcc and Rust"
vendor=${BENCH_VENDOR:-${CORE_GPU_VENDOR:-}}
[ -n "$vendor" ] || usage "set BENCH_VENDOR (or CORE_GPU_VENDOR) to a cargo vendor directory of Cargo.lock"
build=${BENCH_TARGET:-${CORE_GPU_TARGET:-$ROOT/target/core-gpu}}
out=${BENCH_OUT:-$ROOT/out/bench}
podman_args=${BENCH_PODMAN_ARGS:-${CORE_GPU_PODMAN_ARGS:---device nvidia.com/gpu=all}}
runner_args=${BENCH_RUNNER_ARGS:-${CORE_GPU_RUNNER_ARGS:-}}
prompts=${BENCH_PROMPTS:-${CORE_GPU_PROMPTS:-}}
[ -z "$prompts" ] || [ -d "$prompts" ] || usage "BENCH_PROMPTS $prompts is not a directory"
runner=${BENCH_RUNNER:-${CORE_GPU_RUNNER:-}}
runner_cmd=() runner_sha=
if [ -n "$runner" ]; then
  [ -f "$runner" ] || usage "BENCH_RUNNER $runner is not a file"
  case $runner in /*) ;; *) runner=$PWD/$runner ;; esac
  runner_sha=$(sha256sum < "$runner" | cut -c1-64)
  runner_cmd=("$runner")
  [ -x "$runner" ] || runner_cmd=(bash "$runner")
  # The device flag is for a GPU the host gives to containers: with a runner, none gets it.
  podman_args=${podman_args//--device nvidia.com\/gpu=all/}
fi
llama_server=${BENCH_LLAMA_SERVER:-}

# The models: the slots, then the extra directories, each named by its last component.
names=()
declare -A dir_of var_of
slot() { names+=("$1"); var_of[$1]=$2; dir_of[$1]=$3; }
slot qwen3-0.6b BENCH_QWEN3_0_6B "${BENCH_QWEN3_0_6B:-${CORE_GPU_QWEN3_0_6B:-}}"
slot qwen3-8b BENCH_QWEN3_8B "${BENCH_QWEN3_8B:-${CORE_GPU_QWEN3_8B:-}}"
slot qwen3-14b BENCH_QWEN3_14B "${BENCH_QWEN3_14B:-}"
slot qwen3-32b BENCH_QWEN3_32B "${BENCH_QWEN3_32B:-}"
slot qwen3-30b-a3b BENCH_QWEN3_30B_A3B "${BENCH_QWEN3_30B_A3B:-}"
moe_slots=" qwen3-30b-a3b "
for d in ${BENCH_EXTRA_MODELS:-}; do
  d=${d%/}
  n=${d##*/}
  [[ $n =~ $name_re ]] || usage "BENCH_EXTRA_MODELS: $d: a model is named by its directory's last component, and \"$n\" cannot be"
  [ -z "${var_of[$n]:-}" ] || usage "BENCH_EXTRA_MODELS: two models named $n"
  slot "$n" BENCH_EXTRA_MODELS "$d"
done
pick() { # pick VAR VALUE: the names of VALUE, each a model
  local w
  for w in $(words "$1" "$2" "$name_re"); do [ -n "${var_of[$w]:-}" ] || usage "$1: no model named $w (${names[*]})"; done
  printf '%s' "$2"
}
load_models=$(pick BENCH_LOAD_MODELS "${BENCH_LOAD_MODELS:-qwen3-0.6b}")
soak_models=$(pick BENCH_SOAK_MODELS "${BENCH_SOAK_MODELS:-qwen3-0.6b}")
baseline_models=$(pick BENCH_BASELINE_MODELS "${BENCH_BASELINE_MODELS:-qwen3-0.6b}")
server_only="the server serves Qwen3 dense only (core_backend::core_config)"

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
logs=$out/logs/$id
runs=$logs/runs.jsonl
mkdir -p "$build/target" "$build/cargo-home" "$logs"
# The runner wants one empty results directory per program.
rm -rf "$logs/runner"
rm -f "$logs"/*.out "$logs"/*.err "$logs"/*.json "$logs"/*.log "$logs/card-lost" "$logs/build-failed" "$runs"
mkdir -p "$logs/runner"
: > "$runs"

cat > "$build/cargo-home/config.toml" <<'TOML'
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "/vendor"

[net]
offline = true
TOML
mounts=(-v "$ROOT:/src:ro" -v "$vendor:/vendor:ro" -v "$build/target:/target" -v "$build/cargo-home:/cargo-home")

# on_card NAME SECONDS [RUNNER OPTIONS...] PROGRAM [ARGS...]: one program on the card, once. 199
# is written down: nothing runs after it.
on_card() {
  local name=$1 secs=$2 rc=0
  shift 2
  # shellcheck disable=SC2086
  "${runner_cmd[@]}" -m "$runner_mem" -T "$secs" $runner_args -o "$logs/runner/$name" "$@" < /dev/null || rc=$?
  if [ "$rc" = 199 ]; then
    echo "card lost: the runner exited 199 at $name, nothing was run after it; the host needs a reboot" > "$logs/card-lost"
    echo "bench: CARD LOST at $name (runner exit 199): every remaining run is dropped, nothing is retried" >&2
  fi
  return "$rc"
}

# ---- the GPU: the guest's nvidia-smi through the runner, or a container's with the flags given;
# asked only when a step needs the card.
gpu_line= no_gpu="no GPU"
case " $steps " in
  *" core "*|*" engine "*|*" load "*|*" soak "*|*" baseline "*) wants_gpu=1 ;;
  *) wants_gpu=0 ;;
esac
smi=(--query-gpu=name,driver_version,memory.total --format=csv,noheader)
if [ "$wants_gpu" = 1 ] && [ -n "$runner" ]; then
  echo "bench: the programs go through the runner $runner"
  prc=0
  on_card probe 600 -g /usr/bin/nvidia-smi "${smi[@]}" > "$logs/probe.out" 2> "$logs/probe.err" || prc=$?
  if [ "$prc" = 0 ]; then
    gpu_line=$(head -n 1 "$logs/probe.out")
    [ -n "$gpu_line" ] || no_gpu="no GPU: the runner's nvidia-smi named none"
  else
    no_gpu="no GPU: runner exit $prc: $(grep -v '^bench: ' "$logs/probe.err" | tail -n 1 | tr -d '\t\r' | cut -c1-200)"
  fi
elif [ "$wants_gpu" = 1 ]; then
  # shellcheck disable=SC2086
  gpu_line=$(podman run --rm --network=none --name "x0ne-bench-probe-$$" $podman_args "$image" \
    nvidia-smi "${smi[@]}" 2> /dev/null | head -n 1) || gpu_line=
fi
gpu=null run_args=$podman_args
if [ -n "$gpu_line" ]; then
  gpu=$(jq -nc --arg l "$gpu_line" '$l | split(",") | map(gsub("^\\s+|\\s+$"; "")) | {name: .[0], driver: .[1], memory: .[2]}')
  echo "bench: GPU $gpu_line"
else
  # podman refuses a device it lacks: the containers that need none go without the flag.
  run_args=${podman_args//--device nvidia.com\/gpu=all/}
  [ "$wants_gpu" = 0 ] || [ -s "$logs/card-lost" ] || echo "bench: $no_gpu: the tiers on the card are skipped, not passed"
fi

# podman_run NAME CMD...: CMD in the image, the tree read-only, the build directory's target.
podman_run() {
  local name="x0ne-bench-$1-$$"
  shift
  # shellcheck disable=SC2086
  podman run --rm --network=none --name "$name" $run_args "${mounts[@]}" -w /src \
    -e CARGO_HOME=/cargo-home -e CARGO_TARGET_DIR=/target -e CARGO_INCREMENTAL=0 "$image" "$@"
}
cargo_args=(--release --locked --offline -p x0ne-infer --features cuda)
# x0ne-core's own workspace: its lock, its target directory, the same vendored crates.
core_cargo='cd /src/x0ne-core && export CARGO_TARGET_DIR=/target/core && cargo'

# built KIND NAME CARGO...: the path in the containers (/target/...) of the one executable the
# cargo command given, with --message-format=json, names for the target NAME of kind KIND (lib:
# its test binary); else why not, and 2. Not a file pattern: an older binary stays in deps/.
built() {
  local kind=$1 name=$2 json paths n
  shift 2
  json=$(podman_run bin "$@") || { echo "cargo did not name the $kind $name: run the build step"; return 2; }
  paths=$(jq -rR --arg k "$kind" --arg n "$name" 'fromjson? | select(type == "object")
    | select(.reason == "compiler-artifact" and (.executable | type) == "string"
        and ((.target.name // "") | gsub("-"; "_")) == ($n | gsub("-"; "_")) and ((.target.kind // []) | index($k)) != null)
    | .executable' <<<"$json" | sort -u)
  n=$(grep -c . <<<"$paths" || true)
  [ "$n" = 1 ] || { echo "$n executables for the $kind $name, one expected"; return 2; }
  case $paths in /target/*) ;; *) echo "the $kind $name is at $paths, outside /target"; return 2 ;; esac
  [ -f "$build$paths" ] && [ -x "$build$paths" ] || { echo "no executable file $build$paths"; return 2; }
  printf '%s\n' "$paths"
}
bin_core= why_core= bin_infer= why_infer=
binary() { # binary core|infer: bin, or why and 1. Asked of cargo once per run.
  local got
  case $1 in
    core)
      if [ -z "$bin_core$why_core" ]; then
        if got=$(built example core-bench bash -c "$core_cargo build --release --locked --offline -p x0ne-core-oracle \
          --example core-bench --message-format=json"); then bin_core=$got; else why_core=$got; fi
      fi
      bin=$bin_core why=$why_core ;;
    infer)
      if [ -z "$bin_infer$why_infer" ]; then
        if got=$(built lib x0ne_infer cargo test "${cargo_args[@]}" --lib --no-run --message-format=json); then bin_infer=$got; else why_infer=$got; fi
      fi
      bin=$bin_infer why=$why_infer ;;
  esac
  [ -n "$bin" ]
}

# rec STATUS [REASON]: the run of $tier and $m into the verdict, with $rc, $about, $res (its
# result file) and $more (JSON fields), as the caller has them.
rec() {
  local status=$1 reason=${2:-} more_json=${more:-}
  [ -n "$more_json" ] || more_json='{}'
  jq -nc --arg tier "$tier" --arg model "$m" --arg status "$status" --arg reason "$reason" \
    --argjson exit "${rc:-null}" --argjson seconds "$((SECONDS - t0))" --argjson about "${about:-null}" \
    --argjson more "$more_json" --slurpfile res "${res:-/dev/null}" '
    {tier: $tier, model: (if $model == "" then null else $model end), status: $status,
     reason: (if $reason == "" then null else $reason end), exit: $exit, seconds: $seconds,
     about: $about, result: ($res[0] // null)} + $more' >> "$runs"
  echo "bench: $tier${m:+ $m}: $status${reason:+: $reason} in $((SECONDS - t0)) s"
}

# gpu_run NAME SECONDS PROGRAM ARGS...: PROGRAM (a path of /target) on the card, with the inputs
# of ins (HOST:GUEST, read-only) and the variables of envs; through the runner, or in a container
# given the GPU. Its stdout in NAME.out, its stderr in NAME.err and on the terminal; rc its code.
gpu_run() {
  local name=$1 secs=$2 prog=$3 a=() i
  shift 3
  rc=0
  if [ -n "$runner" ]; then
    for i in "${ins[@]}"; do a+=(-f "$i"); done
    for i in "${envs[@]}"; do a+=(-e "$i"); done
    { on_card "$name" "$secs" "${a[@]}" -w /tmp/x0ne-bench "$build$prog" "$@" 2>&1 > "$logs/$name.out" \
        | tee "$logs/$name.err" >&2; } || rc=$?
  else
    for i in "${ins[@]}"; do a+=(-v "$i:ro"); done
    for i in "${envs[@]}"; do a+=(-e "$i"); done
    # shellcheck disable=SC2086
    { podman run --rm --network=none --name "x0ne-bench-$name-$$" $run_args -v "$build/target:/target:ro" \
        "${a[@]}" -w /tmp "$image" "$prog" "$@" 2>&1 > "$logs/$name.out" < /dev/null \
        | tee "$logs/$name.err" >&2; } || rc=$?
  fi
}

# collect NAME: the result of the program run as NAME, checked, recorded.
collect() {
  local name=$1 n why
  if [ -s "$logs/card-lost" ]; then rec fail "$(cat "$logs/card-lost")"; return; fi
  case $rc in
    0) ;;
    125) if [ -n "$runner" ]; then rec fail "runner exit 125: no result from the card"; else rec fail "exit 125"; fi; return ;;
    *) rec fail "exit $rc"; return ;;
  esac
  n=$(grep -c 'x0ne-bench-result: ' "$logs/$name.out" || true)
  [ "$n" != 0 ] || { rec fail "no x0ne-bench-result line on its output"; return; }
  [ "$n" = 1 ] || { rec fail "$n x0ne-bench-result lines on its output, one expected"; return; }
  sed -n 's/.*x0ne-bench-result: //p' "$logs/$name.out" > "$logs/$name.json"
  jq -e 'type == "object"' "$logs/$name.json" > /dev/null 2>&1 || { rec fail "its x0ne-bench-result line is not a JSON object"; return; }
  res=$logs/$name.json
  jq -e --arg k "$tier" '.schema == 1 and .kind == $k' "$res" > /dev/null || { rec fail "not a $tier result (schema 1, kind $tier)"; return; }
  # Out of memory at every context the program tried: what the card cannot hold is an outcome.
  why=$(jq -r '.refused // empty | tostring' "$res")
  if [ -n "$why" ]; then rec skipped "refused on this card: $why"; return; fi
  why=$(jq -r --arg t "$tier" "$CHECKS_JQ" "$res")
  if [ -n "$why" ]; then rec fail "$why"; else rec ok; fi
}

is_moe() { jq -e '((.model_type // "") | test("moe")) or ((.num_experts // 0) > 0)' "$1/config.json" > /dev/null; }

# one TIER MODEL: the run of MODEL in TIER, from its checks to its result.
one() {
  local tier=$1 m=$2 t0=$SECONDS rc=null about=null res= more= dir var cut lens bin why gvar fvar gguf from root lib
  local ins=() envs=() server_tier=0
  dir=${dir_of[$m]} var=${var_of[$m]}
  case $tier in engine|load|soak) server_tier=1 ;; esac
  if [ -s "$logs/card-lost" ]; then rec skipped "card lost: not run"; return; fi
  if [ -z "$dir" ]; then
    case "$server_tier$moe_slots" in 1*" $m "*) rec skipped "$server_only" ;; *) rec skipped "$var is not set" ;; esac
    return
  fi
  dir=${dir%/}
  [ -f "$dir/config.json" ] || { rec fail "$var: no config.json in $dir"; return; }
  cut=null
  [ ! -f "$dir/CUT.json" ] || cut=$(jq -c '.what // null' "$dir/CUT.json" 2> /dev/null) || cut=null
  about=$(jq -c --argjson cut "$cut" '{model_type: (.model_type // null), layers: (.num_hidden_layers // null),
    experts: (.num_experts // null), truncated: $cut}' "$dir/config.json" 2> /dev/null) \
    || { about=null; rec fail "$var: $dir/config.json is not JSON"; return; }
  if [ "$server_tier" = 1 ]; then
    [ "$(jq -r '.model_type // ""' "$dir/config.json")" = qwen3 ] && ! is_moe "$dir" || { rec skipped "$server_only"; return; }
    [ -f "$dir/tokenizer.json" ] || { rec skipped "no tokenizer.json in $dir: the server needs one"; return; }
    [ -n "$prompts" ] || { rec skipped "BENCH_PROMPTS is not set: the server's tiers send the kit's <name>.chat.json"; return; }
  fi
  gvar=BENCH_GGUF_$(tr 'a-z.-' 'A-Z__' <<<"$m")
  if [ "$tier" = baseline ]; then
    gguf=${!gvar:-}
    if [ -z "$llama_server" ]; then
      rec skipped "BENCH_LLAMA_SERVER is not set: a llama-server built with CUDA for this GPU, and $gvar, a GGUF of the same weights (BF16)"
      return
    fi
    [ -f "$llama_server" ] && [ -x "$llama_server" ] || { rec fail "BENCH_LLAMA_SERVER: no executable file $llama_server"; return; }
    [ -n "$gguf" ] || { rec skipped "$gvar is not set: a GGUF of $dir (BF16), $gvar""_FROM the directory it was made from"; return; }
    [ -f "$gguf" ] || { rec fail "$gvar: no file $gguf"; return; }
  fi
  if [ -n "$(cat "$logs/build-failed" 2> /dev/null)" ]; then rec skipped "the build failed: not run"; return; fi
  [ "$gpu" != null ] || { rec skipped "$no_gpu"; return; }

  ins=("$dir:/data/model")
  [ -z "$prompts" ] || ins+=("$prompts:/data/prompts")
  envs=(X0NE_BENCH_MODEL=/data/model)
  [ -z "$prompts" ] || envs+=(X0NE_BENCH_PROMPTS=/data/prompts)
  envs+=("X0NE_BENCH_REPS=$reps" "X0NE_BENCH_DECODE=$decode")
  lens=$lengths
  if is_moe "$dir"; then lens=$moe_lengths; fi
  local tests=(--exact --include-ignored --test-threads=1 --nocapture)
  case $tier in
    core)
      binary core || { rec fail "$why"; return; }
      local pargs=()
      [ -z "$prompts" ] || pargs=(--prompts /data/prompts)
      envs=()
      gpu_run "core-$m" "$runner_secs" "$bin" /data/model "${pargs[@]}" --lengths "${lens// /,}" --decode "$decode" --reps "$reps" ;;
    engine)
      binary infer || { rec fail "$why"; return; }
      envs+=("X0NE_BENCH_ENGINE_PROMPTS=${engine_prompts// /,}" "X0NE_BENCH_TICKS=${ticks// /,}")
      gpu_run "engine-$m" "$runner_secs" "$bin" bench_tests::gpu::engine "${tests[@]}" ;;
    load)
      binary infer || { rec fail "$why"; return; }
      envs+=("X0NE_BENCH_CLIENTS=${clients// /,}" "X0NE_BENCH_LEVEL_SECS=$level_s" "X0NE_BENCH_LOAD_PROMPT=$load_prompt"
        "X0NE_BENCH_LOAD_TOKENS=$load_tokens" "X0NE_BENCH_MAX_QUEUE=$max_queue")
      gpu_run "load-$m" "$runner_secs" "$bin" bench_tests::gpu::load "${tests[@]}" ;;
    soak)
      binary infer || { rec fail "$why"; return; }
      envs+=("X0NE_BENCH_LOAD_PROMPT=$load_prompt" "X0NE_BENCH_LOAD_TOKENS=$load_tokens" "X0NE_BENCH_MAX_QUEUE=$max_queue"
        "X0NE_BENCH_SOAK_SECS=$soak_s" "X0NE_BENCH_SOAK_CLIENTS=$soak_clients" "X0NE_BENCH_SOAK_WINDOW_SECS=$soak_window")
      gpu_run "soak-$m" "$((soak_s + 1800))" "$bin" bench_tests::gpu::soak "${tests[@]}" ;;
    baseline)
      root=${BENCH_LLAMA_ROOT:-$(dirname "$(dirname "$llama_server")")}
      root=${root%/}
      lib=${BENCH_LLAMA_LIB:-$root/lib}
      [ -d "$lib" ] || lib=$(dirname "$llama_server")
      lib=${lib%/}
      case $llama_server in "$root"/*) ;; *) rec fail "BENCH_LLAMA_SERVER $llama_server is not under BENCH_LLAMA_ROOT $root"; return ;; esac
      case $lib/ in "$root"/*) ;; *) rec fail "BENCH_LLAMA_LIB $lib is not under BENCH_LLAMA_ROOT $root"; return ;; esac
      binary infer || { rec fail "$why"; return; }
      fvar=${gvar}_FROM
      from=${!fvar:-}
      echo "bench: the sha256 of $gguf"
      more=$(jq -nc --arg path "$gguf" --arg sha "$(sha256sum < "$gguf" | cut -c1-64)" --argjson bytes "$(stat -c %s "$gguf")" \
        --arg var "$gvar" --arg from "$from" --arg from_real "$( [ -z "$from" ] || readlink -f "$from" || true)" \
        --arg dir "$(readlink -f "$dir")" --arg dtype "$(jq -r '.torch_dtype // ""' "$dir/config.json")" '
        {gguf: {path: $path, sha256: $sha, bytes: $bytes, var: $var, from: (if $from == "" then null else $from end),
                from_real: (if $from_real == "" then null else $from_real end)},
         model_dir: $dir, weights_dtype: (if $dtype == "" then null else $dtype end)}')
      ins=("$root:/data/llama" "$gguf:/data/model.gguf" "$dir/config.json:/data/model/config.json")
      [ -z "$prompts" ] || ins+=("$prompts:/data/prompts")
      envs+=("X0NE_BENCH_LLAMA_SERVER=/data/llama/${llama_server#"$root"/}" "X0NE_BENCH_LLAMA_LIB=/data/llama${lib#"$root"}"
        "X0NE_BENCH_GGUF=/data/model.gguf" "X0NE_BENCH_LENGTHS=${lens// /,}")
      gpu_run "baseline-$m" "$runner_secs" "$bin" bench_tests::baseline_llama "${tests[@]}" ;;
  esac
  collect "$tier-$m"
}

step_build() {
  local tier=build m= t0=$SECONDS rc=0 about=null res= more=
  if [ -s "$logs/card-lost" ]; then rc=null; rec skipped "card lost: not run"; return; fi
  echo "bench: build: x0ne-infer's lib test binary (--features cuda) and Core's core-bench"
  { podman_run build cargo test "${cargo_args[@]}" --lib --no-run \
      && podman_run build bash -c "$core_cargo build --release --locked --offline -p x0ne-core-oracle --example core-bench"; } 2>&1 \
    | tee "$logs/build.log" || rc=$?
  if [ "$rc" = 0 ]; then rec ok; else echo 1 > "$logs/build-failed"; rec fail "exit $rc"; fi
}

for step in $steps; do
  case $step in
    build) step_build ;;
    core|engine) for model in "${names[@]}"; do one "$step" "$model"; done ;;
    load) for model in $load_models; do one load "$model"; done ;;
    soak) for model in $soak_models; do one soak "$model"; done ;;
    baseline) for model in $baseline_models; do one baseline "$model"; done ;;
  esac
done

# ---- the verdict and its report.
host=$(jq -nc --arg cpu "$(awk -F': *' '/^model name/ { print $2; exit }' /proc/cpuinfo)" --argjson threads "$(nproc)" \
  --argjson mem "$(awk '/^MemTotal:/ { print int($2 / 1024); exit }' /proc/meminfo)" --arg kernel "$(uname -r)" \
  '{cpu: $cpu, threads: $threads, mem_mib: $mem, kernel: $kernel}')
runner_json=null
if [ -n "$runner" ]; then
  runner_json=$(jq -nc --arg path "$runner" --arg sha "$runner_sha" --argjson mem "$runner_mem" --argjson secs "$runner_secs" \
    --arg args "$runner_args" '{path: $path, sha256: $sha, mem_mib: $mem, timeout_s: $secs, args: $args}')
fi
models_json=$(for n in "${names[@]}"; do printf '%s\t%s\n' "$n" "${dir_of[$n]}"; done \
  | jq -Rn '[inputs | split("\t") | {key: .[0], value: (if .[1] == "" then null else .[1] end)}] | from_entries')
settings=$(jq -nc --argjson reps "$reps" --argjson decode "$decode" --argjson soak_s "$soak_s" \
  --arg lengths "$lengths" --arg moe_lengths "$moe_lengths" --arg engine_prompts "$engine_prompts" --arg ticks "$ticks" \
  --arg clients "$clients" --argjson level_s "$level_s" --arg load_prompt "$load_prompt" --argjson load_tokens "$load_tokens" \
  --argjson max_queue "$max_queue" --argjson soak_clients "$soak_clients" --argjson soak_window "$soak_window" \
  --arg prompts "$prompts" --argjson models "$models_json" --arg load_models "$load_models" --arg soak_models "$soak_models" \
  --arg baseline_models "$baseline_models" --arg llama "$llama_server" --arg steps "$steps" '
  def w: [splits(" +")] | map(select(length > 0));
  {steps: ($steps | w), reps: $reps, decode_tokens: $decode, lengths: ($lengths | w), moe_lengths: ($moe_lengths | w),
   engine_prompts: ($engine_prompts | w), ticks: ($ticks | w), clients: ($clients | w | map(tonumber)), level_s: $level_s,
   load_prompt: $load_prompt, load_tokens: $load_tokens, max_queue: $max_queue, soak_s: $soak_s, soak_clients: $soak_clients,
   soak_window_s: $soak_window, prompts: (if $prompts == "" then null else $prompts end), models: $models,
   load_models: ($load_models | w), soak_models: ($soak_models | w), baseline_models: ($baseline_models | w),
   llama_server: (if $llama == "" then null else $llama end)}')
steps_json=$(jq -nc --arg s "$steps" '[$s | splits(" +")] | map(select(length > 0))')
ref_file=$REF
[ -f "$ref_file" ] || ref_file=/dev/null
jq -n --slurpfile runs "$runs" --slurpfile ref "$ref_file" --argjson gpu "$gpu" --argjson host "$host" \
  --argjson runner "$runner_json" --argjson settings "$settings" --argjson steps "$steps_json" \
  --arg sha "$sha" --arg branch "$branch" --arg dirty "$dirty" --arg tree "$tree" --arg image "$image" \
  --arg podman_args "$podman_args" --arg started "$started" --arg ended "$(date -Is)" --argjson duration "$SECONDS" \
  --arg log "$logs" --arg stopped "$(cat "$logs/card-lost" 2> /dev/null || true)" "$VERDICT_JQ" > "$out/$id.json.tmp"
mv "$out/$id.json.tmp" "$out/$id.json"
report "$out/$id.json" > "$out/$id.md.tmp"
mv "$out/$id.md.tmp" "$out/$id.md"
ln -sfn "$id.json" "$out/latest.json"
ln -sfn "$id.md" "$out/latest.md"
status=$(jq -r .status "$out/$id.json")
echo "bench: $status $(jq -r '.short + " in \(.duration_s) s"
  + (if .stopped then " (" + .stopped + ")" else "" end)
  + (if (.failed | length) > 0 then " (failed: " + (.failed | join(", ")) + ")" else "" end)
  + (if (.skipped | length) > 0 then " (skipped: " + (.skipped | join(", ")) + ")" else "" end)
  + (if (.unreferenced | length) > 0 then " (no reference: " + (.unreferenced | join(", ")) + ")" else "" end)' "$out/$id.json"); $out/$id.md"
case $status in
  green) exit 0 ;;
  partial) exit 3 ;;
  *) exit 1 ;;
esac
