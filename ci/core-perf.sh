#!/usr/bin/env bash
# The performance gate of x0ne Core inside x0ne Engine: a run of the core-perf example
# (x0ne-core/oracle/examples/core-perf.rs, step perf of make core-gpu) against the medians held
# in ci/core-perf-baseline.json for the same model on the same GPU.
#
#   core-perf.sh compare RESULT.json [BASELINE.json]
#       exit 0  every metric holds: no rate below, no time above, the baseline by more than
#               its tolerance_pct
#       exit 1  a metric regressed
#       exit 3  the baseline has no entry for this model and GPU: nothing was compared
#   core-perf.sh update RESULT.json [BASELINE.json] [--replace]
#       records the run. An entry that exists is only raised: per metric the better of the two
#       medians, and only when the run holds it (compare exits 0). --replace writes the run as
#       it is: for a new driver, a new GPU of the same name, or a slowdown that was accepted.
#
# A metric named *_tok_s is a rate (higher is better); any other is a time in seconds (lower
# is better). Needs jq. Exit 2: bad usage or a file that is not what it should be.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
die() { echo "core-perf: $*" >&2; exit 2; }
cmd=${1:-}; result=${2:-}
if [ -z "$cmd" ] || [ -z "$result" ]; then die "usage: core-perf.sh compare|update RESULT.json [BASELINE.json] [--replace]"; fi
shift 2
baseline=$ROOT/ci/core-perf-baseline.json replace=0
for a in "$@"; do
  case $a in
    --replace) replace=1 ;;
    *) baseline=$a ;;
  esac
done
command -v jq >/dev/null || die "jq not found"
[ -s "$result" ] || die "no result at $result"
[ -s "$baseline" ] || die "no baseline at $baseline"

# The run's key and its medians, or nothing when the file is not a core-perf result.
JQ_RUN='
  select(.schema == 1 and (.model | type) == "string" and (.engine | type) == "string"
         and (.metrics | type) == "object" and (.metrics | length) > 0
         and ([.metrics[] | .median | type == "number" and . > 0] | all))
  | {key: "\(.model) on \(.engine | split(" (cc")[0])",
     prompt_tokens, decode_tokens, kernel_set,
     metrics: (.metrics | map_values(.median))}'
run=$(jq -c "$JQ_RUN" "$result" 2>/dev/null) || die "$result is not JSON"
[ -n "$run" ] || die "$result is not a core-perf result (schema 1, model, engine, positive medians)"
jq -e '.schema == 1 and (.tolerance_pct | type) == "number" and .tolerance_pct >= 0 and (.runs | type) == "object"' \
  "$baseline" >/dev/null 2>&1 || die "$baseline is not a core-perf baseline (schema 1, tolerance_pct, runs)"

# compare: prints one line per metric, exits 0, 1 or 3.
compare() {
  jq -r --argjson run "$run" '
    .tolerance_pct as $tol
    | .runs[$run.key] as $held
    | if $held == null then "none\t\($run.key)"
      else
        ($run.metrics | to_entries | map(
          .key as $m | .value as $v | $held.metrics[$m] as $b
          | if $b == null then {m: $m, v: $v, b: null, ok: true}
            else (if ($m | endswith("_tok_s")) then $v >= $b * (1 - $tol / 100) else $v <= $b * (1 + $tol / 100) end) as $ok
              | {m: $m, v: $v, b: $b, ok: $ok, pct: (($v - $b) / $b * 100)}
            end)) as $rows
        # A metric the baseline holds and the run lacks was not measured: not held.
        | ([$held.metrics | keys[] | select($run.metrics[.] == null)]) as $gone
        | ($rows[] | if .b == null then "new\t\(.m)\t\(.v)" else "\(if .ok then "held" else "RED" end)\t\(.m)\t\(.v)\t\(.b)\t\(.pct * 100 | round / 100)" end),
          ($gone[] | "RED\t\(.)\tmissing"),
          "verdict\t\(if ($rows | all(.ok)) and ($gone | length) == 0 then 0 else 1 end)\t\($run.key)\t\($tol)"
      end' "$baseline"
}

report=$(compare)
verdict=$(printf '%s\n' "$report" | tail -1)
if [ "${verdict%%$'\t'*}" = none ]; then
  rc=3
else
  rc=$(printf '%s' "$verdict" | cut -f2)
fi
key=$(printf '%s' "$run" | jq -r .key)

case $cmd in
  compare)
    if [ "$rc" = 3 ]; then
      echo "core-perf: no baseline for \"$key\" in $baseline: nothing compared (ci/core-perf.sh update records this run)"
      exit 3
    fi
    printf '%s\n' "$report" | sed '$d' | while IFS=$'\t' read -r state metric value held pct; do
      case $state in
        new)  echo "core-perf: $metric $value (not in the baseline)" ;;
        RED)  if [ "$value" = missing ]; then echo "core-perf: RED   $metric is in the baseline and not in the run"
              else echo "core-perf: RED   $metric $value against $held ($pct %)"; fi ;;
        held) echo "core-perf: held  $metric $value against $held ($pct %)" ;;
      esac
    done
    tol=$(printf '%s' "$verdict" | cut -f4)
    if [ "$rc" = 0 ]; then echo "core-perf: \"$key\" holds its baseline (tolerance $tol %)"
    else echo "core-perf: \"$key\" is below its baseline (tolerance $tol %)"; fi
    exit "$rc"
    ;;
  update)
    if [ "$rc" = 1 ] && [ "$replace" = 0 ]; then
      echo "core-perf: \"$key\" does not hold its baseline: not recorded (--replace writes it as it is)" >&2
      exit 1
    fi
    tmp=$(mktemp "$baseline.XXXXXX")
    trap 'rm -f "$tmp"' EXIT
    jq --argjson run "$run" --arg date "$(date -u -Is)" --argjson replace "$replace" '
      .runs[$run.key] as $held
      | ($run.metrics | to_entries | map(
          .key as $m | .value as $v | ($held.metrics[$m] // null) as $b
          | {key: $m, value: (if $replace == 1 or $b == null then $v
                              elif ($m | endswith("_tok_s")) then ([$v, $b] | max) else ([$v, $b] | min) end)})
         | from_entries) as $metrics
      | .runs[$run.key] = {date: $date, prompt_tokens: $run.prompt_tokens, decode_tokens: $run.decode_tokens,
                           kernel_set: $run.kernel_set, metrics: $metrics}' "$baseline" > "$tmp"
    mv "$tmp" "$baseline"
    trap - EXIT
    echo "core-perf: \"$key\" recorded in $baseline"
    ;;
  *) die "unknown command $cmd (compare, update)" ;;
esac
