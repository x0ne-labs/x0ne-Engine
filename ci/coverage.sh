#!/usr/bin/env bash
# Test coverage of the workspace per crate, held to a committed baseline that only goes up.
# Usage: ci/coverage.sh [run|update|status|check <report> <baseline>|summarize <export> <members.tsv>]
#        (make coverage, coverage-update, coverage-status)
#
# run     the commit of HEAD (CI_WORKTREE=1: the working tree, as make ci) in a throwaway
#         container with no network (ci/Containerfile.coverage: the CI image plus llvm-tools),
#         under the build lock of make ci (CI_LOCK), in its own source and target directories
#         under CI_COVERAGE_CACHE, never those of make ci, without sccache. ci/inside-coverage.sh
#         builds and runs the tests with -C instrument-coverage on the workspace members
#         (RUSTC_WORKSPACE_WRAPPER, not the dependencies) and the feature set of make ci
#         (ALL_FEATURES of ci/inside.sh), then llvm-profdata merge and llvm-cov export. Counted per
#         crate: the files under <member>/src/, but tests.rs and *_tests.rs. Red when a test fails,
#         or a crate of the measured tree's ci/coverage-baseline.json loses more than its
#         tolerance_pt of lines or regions, or a crate is on one side only. Writes
#         out/coverage/<sha>.json (<sha>+dirty.json for a modified working tree): commit, dirty,
#         source, durations, counts per crate and per file, the comparison.
#         Exit 0 green, 1 red, 2 setup or baseline missing or unreadable.
# update  writes ci/coverage-baseline.json in the working tree from the measure of HEAD: the
#         report of HEAD when it is clean and its tests passed, else a new run. Raises each crate
#         to the measure and keeps a value the measure is under within the tolerance. Refuses a
#         modified tree, a failed test, and a drop past the tolerance or a crate gone, unless
#         CI_COVERAGE_ALLOW_DROP=1, which writes the measure as it is. Commit the file after.
# status  the last report, short.
# check   <report> <baseline>: the comparison alone. 0 green, 1 red, 2 unreadable.
# summarize <llvm-cov export> <members.tsv>: counts per crate, JSON on stdout.
#
# Settings: those of ci/ci.sh, sourced (CI_ENV_FILE, CI_CACHE, CI_LOCK, CI_WAIT_MAX, CI_JOBS,
# CI_MEMORY, CI_CGROUP_PARENT, CI_MIN_DISK_GB, CI_DEBUGINFO, CI_WORKTREE, CI_ALL_FEATURES), and
# CI_COVERAGE_CACHE (default: engine-coverage beside CI_CACHE), CI_COVERAGE_ALLOW_DROP.
set -euo pipefail
cmd=${1:-run}
BASELINE_REL=ci/coverage-baseline.json

clog() { printf 'coverage: %s\n' "$*" >&2; }

# summarize <export> <members.tsv>: lines, regions and functions per crate and in total, the
# per file counts, and how many files of the members were left out (tests, build.rs).
summarize() {
  jq -n --slurpfile cov "$1" --rawfile members "$2" '
    def pct(c; n): if n > 0 then (c * 10000 / n | round) / 100 else null end;
    def tally: . as $rows | reduce ("lines", "regions", "functions") as $k ({};
      ([$rows[] | .s[$k].covered] | add // 0) as $c | ([$rows[] | .s[$k].count] | add // 0) as $n
      | .[$k] = {covered: $c, count: $n, percent: pct($c; $n)});
    def norm: if startswith("/") then . else
      reduce split("/")[] as $p ([]; if $p == ".." then .[:-1] elif $p == "." or $p == "" then . else . + [$p] end) | join("/") end;
    ($members | split("\n") | map(select(length > 0) | split("\t") | {name: .[0], dir: .[1]})) as $ms
    | [$cov[0].data[]?.files[]? | (.filename | sub("^/src/"; "")) as $f | {f: $f, n: ($f | norm), s: .summary}] as $all
    | if ($all | length) == 0 then error("the export lists no file") else . end
    | [$all[] | . as $r | first($ms[] | select(.dir as $d | $r.n | startswith($d + "/src/"))) as $c
        | select($r.n | test("(^|/)tests\\.rs$|_tests\\.rs$") | not) | $r + {crate: $c.name}] as $owned
    # 17 #[path] copies on 2026-10-30 (x0ne-sdk 15 of x0ne-infer): each file counts once, as the
    # owning crate builds and tests it; a copy stands in only when the owner has none.
    | [$owned | group_by(.n)[] | ([.[] | select(.f == .n)][0] // .[0])] as $kept
    | {crates: ($kept | group_by(.crate) | map({key: .[0].crate, value: tally}) | from_entries),
       total: ($kept | tally),
       excluded_files: (([$all[] | . as $r | select(any($ms[]; .dir as $d | $r.f | startswith($d + "/")))] | length) - ($kept | length)),
       path_copies: (($owned | length) - ($kept | length)),
       files: ($kept | map({key: .n, value: {lines: [.s.lines.covered, .s.lines.count], regions: [.s.regions.covered, .s.regions.count]}}) | from_entries)}'
}

# baseline_ok <file>: one document, schema 1, a tolerance >= 0, at least one crate with numeric
# lines and regions. Slurped: jq 1.6 -e exits 0 on an empty file.
baseline_ok() {
  jq -e -s 'length == 1 and (.[0] | type == "object" and .schema == 1 and (.tolerance_pt | type == "number" and . >= 0)
    and (.crates | type == "object" and length > 0
      and all(.[]; type == "object" and (.lines | type == "number") and (.regions | type == "number"))))' "$1" > /dev/null 2>&1
}

# compare <report> <baseline>: one entry per crate of either side, as a JSON array. A drop is
# judged in hundredths of a point: a loss of exactly the tolerance passes.
compare() {
  jq --slurpfile b "$2" '
    $b[0] as $b | ($b.tolerance_pt * 100 | round) as $tol | .crates as $m
    | [([($b.crates | keys[]), ($m | keys[])] | unique)[] as $c
      | if $m[$c] == null then {crate: $c, verdict: "red", reason: "absent from the measure"}
        elif $b.crates[$c] == null then {crate: $c, verdict: "red", reason: "not in the baseline (make coverage-update adds it)"}
        else {crate: $c, lines: {now: ($m[$c].lines.percent // 0), base: $b.crates[$c].lines},
                         regions: {now: ($m[$c].regions.percent // 0), base: $b.crates[$c].regions}}
          | .drops = [("lines", "regions") as $k | select(((.[$k].base - .[$k].now) * 100 | round) > $tol) | $k]
          | if (.drops | length) > 0 then .verdict = "red" | .reason = "\(.drops | join(" and ")) down past \($tol / 100) pt"
            else .verdict = "ok" end
        end]' "$1"
}

# render: the comparison as one line per crate, then the verdict line.
render() {
  jq -r '
    def two($x): ($x | tostring | if length < 2 then "0" + . else . end);
    def p2: (. * 100 | round) as $x | "\($x / 100 | floor).\(two($x % 100))";
    def d2: (. * 100 | round) as $x | (if $x < 0 then "-" else "+" end)
      + ((if $x < 0 then 0 - $x else $x end) as $a | "\($a / 100 | floor).\(two($a % 100))");
    def pad: . + (" " * ([20 - length, 1] | max));
    (.[] | if .lines == null then "\(.crate | pad)RED: \(.reason)"
      else "\(.crate | pad)lines \(.lines.now | p2) % (base \(.lines.base | p2), \(.lines.now - .lines.base | d2))  regions \(.regions.now | p2) % (base \(.regions.base | p2), \(.regions.now - .regions.base | d2))  \(if .verdict == "ok" then "ok" else "RED: " + .reason end)" end),
    (map(select(.verdict != "ok")) | if length == 0 then "green: no crate under its baseline past the tolerance"
      else "red: \(map(.crate) | join(", "))" end)'
}

# check <report> <baseline> [<compare out>]: 0 green, 1 red, 2 missing or unreadable.
check() {
  local report=$1 base=$2 out=${3:-}
  [ -e "$base" ] || { echo "no baseline at $base: make coverage-update writes the first one"; return 2; }
  baseline_ok "$base" || { echo "unreadable baseline $base (schema 1, tolerance_pt >= 0, crates with numeric lines and regions)"; return 2; }
  jq -e '.crates | type == "object"' "$report" > /dev/null 2>&1 || { echo "unreadable report $report"; return 2; }
  local cmp; cmp=$(compare "$report" "$base")
  [ -z "$out" ] || printf '%s\n' "$cmp" > "$out"
  render <<<"$cmp"
  jq -e 'all(.[]; .verdict == "ok")' <<<"$cmp" > /dev/null || return 1
}

case $cmd in
  summarize) [ $# = 3 ] || { clog "usage: ci/coverage.sh summarize <export> <members.tsv>"; exit 2; }
             summarize "$2" "$3" || { clog "summarize: no file to count in $2"; exit 2; }; exit 0 ;;
  check)     [ $# = 3 ] || { clog "usage: ci/coverage.sh check <report> <baseline>"; exit 2; }
             rc=0; check "$2" "$3" || rc=$?; exit $rc ;;
  run|update|status) ;;
  *) clog "usage: ci/coverage.sh [run|update|status|check <report> <baseline>|summarize <export> <members.tsv>]"; exit 2 ;;
esac

# The settings and helpers of make ci: env file, cache, lock, staging, disk guard, CI image.
# shellcheck source=ci/ci.sh
source "$(dirname "$0")/ci.sh"
COV_CACHE=${CI_COVERAGE_CACHE:-$(dirname "$CI_CACHE")/engine-coverage}
COV_SRC=$COV_CACHE/src
COV_TARGET=$COV_CACHE/target
COV_OUT=$REPO/out/coverage
COV_IMG=localhost/x0ne-engine-coverage:$(cat "$CI_DIR/Containerfile" "$CI_DIR/Containerfile.coverage" | sha256sum | cut -c1-12)
mkdir -p "$COV_OUT" "$COV_SRC" "$COV_TARGET" "$COV_CACHE/logs" "$COV_CACHE/stage"

cov_image() {
  image
  if podman image exists "$COV_IMG"; then clog "image $COV_IMG present"; return; fi
  clog "building $COV_IMG"
  podman build --pull=never ${CI_CGROUP_PARENT:+--cgroup-parent=$CI_CGROUP_PARENT} --build-arg CI_IMAGE="$IMG" \
    -f "$CI_DIR/Containerfile.coverage" -t "$COV_IMG" "$CI_DIR" >&2
}

# measure: one run; sets REPORT (the report written) and VERDICT (0 green, 1 red, 2 baseline).
measure() {
  local sha branch started t0 t_all=$SECONDS dirty=0 features
  sha=$(git -C "$REPO" rev-parse HEAD); branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
  disk_guard "$COV_CACHE"
  cov_image
  # stage_tree and stage_is of ci/ci.sh, with the stage under CI_COVERAGE_CACHE.
  trap 'rm -rf "${STAGE:-}"' EXIT
  CI_CACHE=$COV_CACHE stage_tree "$sha"
  if [ "${CI_WORKTREE:-0}" = 1 ] && ! stage_is "$sha"; then dirty=1; fi
  local id=$sha short=${sha:0:7}
  [ "$dirty" = 1 ] && { id=$sha+dirty; short=$short+dirty; }
  local key; key=$(sha256sum "$STAGE/Cargo.lock" | cut -c1-16)
  VENDOR=$CI_CACHE/vendor/$key
  [ -s "$VENDOR/config.toml" ] || die "no vendored crates for this Cargo.lock ($VENDOR): make ci-refresh, or make ci, fetches them"
  # The feature set of make ci: ALL_FEATURES of the measured tree's ci/inside.sh.
  features=${CI_ALL_FEATURES:-$(sed -n 's/^ALL_FEATURES=\${CI_ALL_FEATURES:-\(.*\)}$/\1/p' "$STAGE/ci/inside.sh")}
  [ -n "$features" ] || die "no ALL_FEATURES line in ci/inside.sh"
  local rundir=$COV_CACHE/logs/$short-$(date +%Y%m%d-%H%M%S)
  mkdir -p "$rundir"
  local tree; tree=$(tree_id "$STAGE")
  lock_build
  sync_src "$STAGE" "$COV_SRC"; rm -rf "$STAGE"
  started=$(date -u -Is); t0=$SECONDS
  clog "run $short on $branch, tree $tree, image $COV_IMG, features $features, logs $rundir"
  set +e
  # /src as an overlay (:O), thrown away at exit: a child started with env_clear() writes its
  # default_*.profraw in its working directory; on a read-only /src its "LLVM Profile Error" on
  # stderr turned three tests red (golden outputs, 2026-10-30). The copy on the host stays as is.
  $NICE $BUILD_RUN --userns=keep-id --name "engine-coverage-${sha:0:7}-$$" \
    -v "$COV_SRC":/src:O -v "$VENDOR":/vendor:ro -v "$COV_TARGET":/target -v "$rundir":/out \
    -e CARGO_PROFILE_DEV_DEBUG="$CI_DEBUGINFO" -e CARGO_BUILD_JOBS="$CI_JOBS" -e COVERAGE_FEATURES="$features" \
    "$COV_IMG" bash /src/ci/inside-coverage.sh
  local rc=$?
  set -e
  # The baseline of the measured tree, taken before the lock is released.
  local base=$rundir/baseline.json
  if [ -e "$COV_SRC/$BASELINE_REL" ]; then cp "$COV_SRC/$BASELINE_REL" "$base"; fi
  exec 9>&-
  local dur=$((SECONDS - t0))
  touch "$rundir/status.env"
  local failed=() measured=0
  if [ $rc -ne 0 ]; then failed+=("container exit $rc"); fi
  local st; st=$(cat "$rundir/status.env")
  sv() { sed -n "s/^$1=//p" <<<"$st" | tail -n 1; }
  ex() { local v; v=$(sv "$1"); printf '%s' "${v:-none, not reached}"; }
  [ "$(sv build_exit)" = 0 ] || failed+=("build exit $(ex build_exit)")
  [ "$(sv test_exit)" = 0 ] || failed+=("tests exit $(ex test_exit)")
  if [ "$(sv merge_exit)" = 0 ] && [ "$(sv export_exit)" = 0 ] \
    && summarize "$rundir/llvm-cov.json" "$rundir/members.tsv" > "$rundir/summary.json" 2> "$rundir/summarize.log"; then
    measured=1
  else
    failed+=("no measure (merge exit $(ex merge_exit), export exit $(ex export_exit))")
    echo '{"crates": {}, "total": null, "files": {}, "excluded_files": 0, "path_copies": 0}' > "$rundir/summary.json"
  fi
  VERDICT=0
  local check_out="" crc=0
  check_out=$(check "$rundir/summary.json" "$base" "$rundir/compare.json") || crc=$?
  [ -s "$rundir/compare.json" ] || echo '[]' > "$rundir/compare.json"
  printf '%s\n' "$check_out" >&2
  if [ $crc = 2 ] && [ ! -e "$base" ]; then failed+=("no $BASELINE_REL in the measured tree: make coverage-update writes one"); VERDICT=2
  elif [ $crc = 2 ]; then failed+=("unreadable $BASELINE_REL in the measured tree"); VERDICT=2
  elif [ $crc = 1 ]; then
    mapfile -t -O "${#failed[@]}" failed < <(jq -r '.[] | select(.verdict != "ok") | "\(.crate): \(.reason)"' "$rundir/compare.json")
  fi
  [ ${#failed[@]} = 0 ] || [ "$VERDICT" = 2 ] || VERDICT=1
  local status=green; [ "$VERDICT" = 0 ] || status=red
  REPORT=$COV_OUT/$id.json
  jq -n --arg status "$status" --arg commit "$sha" --arg short "$short" --argjson dirty "$([ "$dirty" = 1 ] && echo true || echo false)" \
     --arg source "$([ "${CI_WORKTREE:-0}" = 1 ] && echo worktree || echo commit)" --arg branch "$branch" --arg tree "$tree" \
     --arg started "$started" --arg ended "$(date -u -Is)" --argjson duration "$dur" --argjson total "$((SECONDS - t_all))" \
     --argjson wait "$LOCK_WAIT" --arg image "$COV_IMG" --arg features "$features" --arg log "$rundir" \
     --rawfile st "$rundir/status.env" --slurpfile sum "$rundir/summary.json" --slurpfile cmp "$rundir/compare.json" \
     --argjson measured "$measured" --arg failed "$(printf '%s\n' "${failed[@]}")" \
     --slurpfile base <(if baseline_ok "$base" 2>/dev/null; then cat "$base"; else echo null; fi) '
    ($st | split("\n") | map(select(test("^[a-z_]+=")) | capture("^(?<k>[a-z_]+)=(?<v>.*)$")) | map({(.k): .v}) | add // {}) as $s
    | def num(k): ($s[k] // "" | tonumber? // null);
    {schema: 1, status: $status, commit: $commit, short: $short, dirty: $dirty, source: $source, branch: $branch,
     tree: $tree, started: $started, ended: $ended, duration_s: $duration, total_s: $total, lock_wait_s: $wait,
     image: $image, toolchain: $s.toolchain, llvm: $s.llvm, features: $features, measured: ($measured == 1),
     tests: {exit: num("test_exit")},
     steps: {build_s: num("build_s"), test_s: num("test_s"), merge_s: num("merge_s"), export_s: num("export_s")},
     objects: num("objects"), profraw: num("profraw"), stray_profraw: num("stray_profraw"),
     baseline: (if $base[0] == null then null else {date: $base[0].measured.date, tolerance_pt: $base[0].tolerance_pt} end),
     total: $sum[0].total, crates: $sum[0].crates, excluded_files: $sum[0].excluded_files, path_copies: $sum[0].path_copies,
     compare: $cmp[0], failed: ($failed | split("\n") | map(select(length > 0))),
     log_dir: $log, files: $sum[0].files}' > "$REPORT.tmp"
  mv "$REPORT.tmp" "$REPORT"; ln -sfn "$id.json" "$COV_OUT/latest.json"
  clog "$status $short in ${dur}s (lock wait ${LOCK_WAIT}s), total $(jq -r '.total.lines.percent // "n/a"' "$REPORT") % of lines; $REPORT"
}

update() {
  local sha; sha=$(git -C "$REPO" rev-parse HEAD)
  REPORT=$COV_OUT/$sha.json
  if [ "${CI_WORKTREE:-0}" != 1 ] && [ -s "$REPORT" ] \
    && jq -e --arg c "$sha" '.commit == $c and .dirty == false and .tests.exit == 0 and .measured == true' "$REPORT" > /dev/null 2>&1; then
    clog "using $REPORT (the measure of HEAD)"
  else
    measure
  fi
  jq -e '.dirty == false' "$REPORT" > /dev/null || { clog "refused: the measure is of a modified tree (dirty); commit, then make coverage-update"; exit 1; }
  jq -e '.tests.exit == 0 and .measured == true' "$REPORT" > /dev/null || { clog "refused: a test failed or nothing was measured ($REPORT)"; exit 1; }
  local base=$REPO/$BASELINE_REL allow=${CI_COVERAGE_ALLOW_DROP:-0} prev=null rc=0
  if [ -e "$base" ]; then
    check "$REPORT" "$base" >&2 || rc=$?
    [ $rc != 2 ] || { clog "refused: $base is unreadable; fix or remove it"; exit 2; }
    if [ $rc = 1 ] && [ "$allow" != 1 ]; then clog "refused: the measure is under the baseline past the tolerance; CI_COVERAGE_ALLOW_DROP=1 writes it as it is"; exit 1; fi
    prev=$(cat "$base")
  fi
  jq --argjson prev "$prev" --argjson allow "$([ "$allow" = 1 ] && echo true || echo false)" '
    {schema: 1, tolerance_pt: ($prev.tolerance_pt // 0.2),
     about: "make coverage: red when a crate loses more than tolerance_pt points of lines or regions (percent). make coverage-update raises it.",
     measured: {date: .ended, toolchain: .toolchain, llvm: .llvm, features: .features},
     crates: (.crates | with_entries(.key as $c | .value = {
       lines: (if $allow or $prev == null then .value.lines.percent else ([.value.lines.percent, ($prev.crates[$c].lines // 0)] | max) end),
       regions: (if $allow or $prev == null then .value.regions.percent else ([.value.regions.percent, ($prev.crates[$c].regions // 0)] | max) end)}))}' \
    "$REPORT" > "$base.tmp"
  mv "$base.tmp" "$base"
  # No commit in the file: it is published, the private history is not; the report keeps it.
  clog "wrote $base from $(jq -r .short "$REPORT"); commit it"
}

case $cmd in
  run)    measure; exit "$VERDICT" ;;
  update) update ;;
  status) [ -e "$COV_OUT/latest.json" ] && jq '{status, short, dirty, source, duration_s, lock_wait_s, lines: .total.lines.percent, regions: .total.regions.percent, failed, ended}' "$COV_OUT/latest.json" || die "no coverage report yet" ;;
esac
