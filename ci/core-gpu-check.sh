#!/usr/bin/env bash
# The gate on make core-gpu: whether a commit has a green verdict of scripts/core-gpu.sh, read
# from out/core-gpu/<sha>.json. Nothing calls it: it is for whoever merges a change of
# x0ne-core/ or of the code that serves through it, and for a hook once that is decided.
#
#   core-gpu-check.sh [SHA]       (default: HEAD)
#
# Exit 0 only when the verdict is of this commit, of its clean tree (never <sha>+dirty), on a
# GPU, with status "green" and every required step ok. A step that was skipped ("no GPU",
# "no baseline") or left out of the run is named and refuses the gate, as a failed one does.
# Exit 1 otherwise, 2 on bad usage.
#
# Environment:
#   CORE_GPU_OUT       where the verdicts are (default out/core-gpu)
#   CORE_GPU_REQUIRED  the steps a merge needs (default "build cpu test reference sanitize perf")
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
out=${CORE_GPU_OUT:-$ROOT/out/core-gpu}
required=${CORE_GPU_REQUIRED:-build cpu test reference sanitize perf}
[ $# -le 1 ] || { echo "usage: core-gpu-check.sh [SHA]" >&2; exit 2; }
command -v jq >/dev/null || { echo "core-gpu-check: jq not found" >&2; exit 2; }
sha=$(git -C "$ROOT" rev-parse --verify --quiet "${1:-HEAD}^{commit}") || { echo "core-gpu-check: no commit ${1:-HEAD}" >&2; exit 2; }
file=$out/$sha.json
no() { echo "core-gpu-check: REFUSED ${sha:0:7}: $*"; exit 1; }

if [ ! -s "$file" ]; then
  if [ -s "$out/$sha+dirty.json" ]; then no "the only verdict is of a tree that was not the commit's ($out/$sha+dirty.json)"; fi
  no "no verdict at $file (run make core-gpu on this commit)"
fi
jq -e . "$file" >/dev/null 2>&1 || no "$file is not JSON"
[ "$(jq -r '.sha // ""' "$file")" = "$sha" ] || no "$file is the verdict of another commit"
[ "$(jq -r '.dirty' "$file")" = false ] || no "the verdict is of a tree that was not the commit's"

problems=()
for step in $required; do
  line=$(jq -r --arg s "$step" '[.steps[]? | select(.name == $s)] | last
    | if . == null then "absent" else "\(.status)\(if .reason then ": " + .reason else "" end)" end' "$file")
  [ "$line" = ok ] || problems+=("$step $line")
done
gpu=$(jq -r '.gpu // ""' "$file")
[ -n "$gpu" ] || problems+=("no GPU was seen")
status=$(jq -r '.status // ""' "$file")
if [ "${#problems[@]}" -gt 0 ]; then
  no "status $status; $(IFS=';'; echo "${problems[*]}" | sed 's/;/; /g')"
fi
[ "$status" = green ] || no "status $status"
echo "core-gpu-check: ok ${sha:0:7}: green on $gpu, steps $required ($(jq -r .ended "$file"))"
