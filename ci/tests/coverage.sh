#!/usr/bin/env bash
# Tests of ci/coverage.sh (make coverage, coverage-update) against a fake podman and df:
# no container, no build, no network. The flows run in a throwaway git repository.
# Run by ci/inside.sh (step ci-sh), or by hand: bash ci/tests/coverage.sh
set -euo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)
T=$(mktemp -d)
trap 'rm -rf "${T:?}"' EXIT
FAILS=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s: %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }
COV=$SRC/ci/coverage.sh

# ---- summarize: llvm-cov export -> lines, regions and functions per crate.
# Counted: <member dir>/src/ of each member. Left out: tests/, build.rs, *_tests.rs and
# tests.rs (test code), dependencies, the standard library, files of no member. A file one
# member compiles from another's src/ through #[path] counts once, under its owner.
printf 'a\ta\nb\tb\n' > "$T/members.tsv"
f() { printf '{"filename":"%s","summary":{"lines":{"count":%s,"covered":%s},"regions":{"count":%s,"covered":%s},"functions":{"count":%s,"covered":%s}}}' "$@"; }
{
  printf '{"type":"llvm.coverage.json.export","version":"3.1.0","data":[{"files":['
  f /src/a/src/lib.rs 10 8 20 15 4 3; printf ,
  f /src/a/src/m.rs 10 2 10 2 2 1; printf ,
  f /src/a/tests/t.rs 100 100 100 100 9 9; printf ,
  f /src/a/src/foo_tests.rs 50 50 50 50 5 5; printf ,
  f /src/a/src/inner/tests.rs 50 50 50 50 5 5; printf ,
  f /src/a/build.rs 30 0 30 0 1 0; printf ,
  f b/src/lib.rs 4 1 8 2 1 1; printf ,
  f /src/b/src/../../a/src/m.rs 10 10 10 10 2 2; printf ,
  f /src/a/src/../../b/src/only.rs 6 3 6 3 1 1; printf ,
  f /vendor/crates/x-1.0.0/src/lib.rs 100 0 100 0 9 0; printf ,
  f /rustc/0123/library/core/src/lib.rs 100 0 100 0 9 0; printf ,
  f /src/ab/src/lib.rs 100 0 100 0 9 0
  printf ']}]}\n'
} > "$T/export.json"
rc=0; out=$(bash "$COV" summarize "$T/export.json" "$T/members.tsv" 2>&1) || rc=$?
got=$(jq -c '[.crates.a.lines, .crates.a.regions, .crates.a.functions, .crates.b.lines, .total.lines, .total.regions, .excluded_files, .path_copies, (.crates | keys), (.files | keys)]' <<<"$out" 2>&1) || got="exit $rc: $out"
want='[{"covered":10,"count":20,"percent":50},{"covered":17,"count":30,"percent":56.67},{"covered":4,"count":6,"percent":66.67},{"covered":4,"count":10,"percent":40},{"covered":14,"count":30,"percent":46.67},{"covered":22,"count":44,"percent":50},5,1,["a","b"],["a/src/lib.rs","a/src/m.rs","b/src/lib.rs","b/src/only.rs"]]'
if [ "$rc" = 0 ] && [ "$got" = "$want" ]; then pass "summarize counts <member>/src per crate once, not tests, build.rs, #[path] copies, dependencies or std"
else fail "summarize counts <member>/src per crate once, not tests, build.rs, #[path] copies, dependencies or std" "exit $rc, $got"; fi
printf '{"data":[]}' > "$T/empty-export.json"
rc=0; bash "$COV" summarize "$T/empty-export.json" "$T/members.tsv" > /dev/null 2>&1 || rc=$?
if [ "$rc" = 2 ]; then pass "summarize refuses an export with no file"; else fail "summarize refuses an export with no file" "exit $rc"; fi

# ---- check: a report against a baseline. 0 green, 1 red, 2 unreadable.
# report <file> <a lines> <a regions> [<b lines> <b regions>]: percents of a measure.
report() {
  local o=$1; shift
  jq -n --argjson al "$1" --argjson ar "$2" --arg bl "${3:-}" --arg br "${4:-}" '
    def m(p): {covered: 0, count: 0, percent: p};
    {crates: ({a: {lines: m($al), regions: m($ar), functions: m(0)}}
      + (if $bl == "" then {} else {b: {lines: m($bl | tonumber), regions: m($br | tonumber), functions: m(0)}} end))}' > "$o"
}
# baseline <file> <a lines> <a regions> [<b lines> <b regions>]
baseline() {
  local o=$1; shift
  jq -n --argjson al "$1" --argjson ar "$2" --arg bl "${3:-}" --arg br "${4:-}" '
    {schema: 1, tolerance_pt: 0.2, measured: {date: "2026-10-30"},
     crates: ({a: {lines: $al, regions: $ar}}
       + (if $bl == "" then {} else {b: {lines: ($bl | tonumber), regions: ($br | tonumber)}} end))}' > "$o"
}
check() { rc=0; out=$(bash "$COV" check "$1" "$2" 2>&1) || rc=$?; }
baseline "$T/base.json" 80 70 60 50
report "$T/r.json" 80 70 60 50; check "$T/r.json" "$T/base.json"
if [ "$rc" = 0 ]; then pass "check: green on the baseline"; else fail "check: green on the baseline" "exit $rc, $out"; fi
report "$T/r.json" 79.81 69.81 60.5 50; check "$T/r.json" "$T/base.json"
if [ "$rc" = 0 ]; then pass "check: green within the tolerance (-0.19 pt)"; else fail "check: green within the tolerance (-0.19 pt)" "exit $rc, $out"; fi
report "$T/r.json" 79.7 70 60 50; check "$T/r.json" "$T/base.json"
if [ "$rc" = 1 ] && grep -q 'a .*lines.*-0.30' <<<"$out"; then pass "check: red when a crate loses 0.3 pt of lines"
else fail "check: red when a crate loses 0.3 pt of lines" "exit $rc, $out"; fi
report "$T/r.json" 80 70 60 49.5; check "$T/r.json" "$T/base.json"
if [ "$rc" = 1 ] && grep -q 'b .*regions.*-0.50' <<<"$out"; then pass "check: red when a crate loses 0.5 pt of regions"
else fail "check: red when a crate loses 0.5 pt of regions" "exit $rc, $out"; fi
report "$T/r.json" 80 70; check "$T/r.json" "$T/base.json"
if [ "$rc" = 1 ] && grep -q 'b .*absent from the measure' <<<"$out"; then pass "check: red when a crate of the baseline is absent from the measure"
else fail "check: red when a crate of the baseline is absent from the measure" "exit $rc, $out"; fi
baseline "$T/base-a.json" 80 70; report "$T/r.json" 80 70 60 50; check "$T/r.json" "$T/base-a.json"
if [ "$rc" = 1 ] && grep -q 'b .*not in the baseline' <<<"$out"; then pass "check: red when a crate of the measure is not in the baseline"
else fail "check: red when a crate of the measure is not in the baseline" "exit $rc, $out"; fi
# unreadable <case> <baseline content>: exit 2, said.
unreadable() {
  printf '%s' "$2" > "$T/bad.json"; report "$T/r.json" 80 70 60 50; check "$T/r.json" "$T/bad.json"
  if [ "$rc" = 2 ] && grep -q 'baseline' <<<"$out"; then pass "check: exit 2 on a baseline that is $1"
  else fail "check: exit 2 on a baseline that is $1" "exit $rc, $out"; fi
}
unreadable "not JSON" '{"schema": 1, "crates": {'
unreadable "empty" ''
unreadable "of another schema" '{"schema": 2, "tolerance_pt": 0.2, "crates": {"a": {"lines": 80, "regions": 70}}}'
unreadable "without crates" '{"schema": 1, "tolerance_pt": 0.2, "crates": {}}'
unreadable "a crate with a text percent" '{"schema": 1, "tolerance_pt": 0.2, "crates": {"a": {"lines": "80", "regions": 70}}}'
unreadable "a negative tolerance" '{"schema": 1, "tolerance_pt": -1, "crates": {"a": {"lines": 80, "regions": 70}}}'
rc=0; out=$(bash "$COV" check "$T/r.json" "$T/no-such.json" 2>&1) || rc=$?
if [ "$rc" = 2 ] && grep -q 'no baseline' <<<"$out"; then pass "check: exit 2 without a baseline"; else fail "check: exit 2 without a baseline" "exit $rc, $out"; fi

# ---- the flows: make coverage and make coverage-update in a fixture repository.
REPO=$T/repo BIN=$T/bin DATA=$T/data
CI_CACHE=$DATA/engine-ci COV_CACHE=$DATA/engine-coverage LOCK=$T/locks/build.lock
mkdir -p "$REPO" "$BIN" "$T/home" "$T/locks"
cp -r "$SRC/ci" "$SRC/Cargo.toml" "$SRC/Cargo.lock" "$SRC/.gitignore" "$REPO/"
echo probe > "$REPO/probe.txt"
baseline "$REPO/ci/coverage-baseline.json" 80 70 60 50
git -C "$REPO" init -q -b main
git -C "$REPO" add -A
git -C "$REPO" -c user.name=t -c user.email=t@t commit -qm fixture
SHA=$(git -C "$REPO" rev-parse HEAD)
KEY=$(sha256sum "$REPO/Cargo.lock" | cut -c1-16)
mkdir -p "$CI_CACHE/vendor/$KEY"
echo '[source]' > "$CI_CACHE/vendor/$KEY/config.toml"
WANT_FEATURES=$(sed -n 's/^ALL_FEATURES=\${CI_ALL_FEATURES:-\(.*\)}$/\1/p' "$SRC/ci/inside.sh")
STAMP=$T/stamp; touch "$STAMP"

# df: 500 GB everywhere.
printf '#!/bin/sh\nprintf "Avail Mounted on\\n 500G /\\n"\n' > "$BIN/df"
# podman: one line per call in $FAKE_LOG. The coverage container (inside-coverage.sh) writes
# what the real one would: members.tsv, llvm-cov.json (a: 1000 lines and 2000 regions, b: 400
# and 800, covered FAKE_A_LINES FAKE_A_REGIONS FAKE_B_LINES FAKE_B_REGIONS), status.env with
# test_exit=FAKE_TEST_EXIT. It appends to $FAKE_RUNS what it was given: source tree, target,
# vendor, features, whether the build lock was held, whether probe.txt was edited.
cat > "$BIN/podman" <<'P'
#!/usr/bin/env bash
echo "podman $*" >> "$FAKE_LOG"
[ "${1:-}" = run ] || exit 0
shift
declare -A m=() e=() mo=()
while [ $# -gt 0 ]; do
  case $1 in
    -v) IFS=: read -r h c opt <<<"$2"; m[$c]=$h; mo[$c]=$opt; shift 2 ;;
    -e) e[${2%%=*}]=${2#*=}; shift 2 ;;
    --name|-w) shift 2 ;;
    localhost/*) shift; break ;;
    *) shift ;;
  esac
done
[ "${2:-}" = /src/ci/inside-coverage.sh ] || exit 0
o=${m[/out]}
held=no; flock -n "$FAKE_LOCK" true || held=yes
echo "inside-coverage src=${m[/src]} srcopt=${mo[/src]:-rw} target=${m[/target]} vendor=${m[/vendor]} sccache=${m[/sccache]:-none} held=$held probe=$(cat "${m[/src]}/probe.txt") features=${e[COVERAGE_FEATURES]:-}" >> "$FAKE_RUNS"
printf 'a\ta\nb\tb\n' > "$o/members.tsv"
s() { printf '{"lines":{"count":%s,"covered":%s},"regions":{"count":%s,"covered":%s},"functions":{"count":1,"covered":1}}' "$@"; }
printf '{"data":[{"files":[{"filename":"/src/a/src/lib.rs","summary":%s},{"filename":"/src/b/src/lib.rs","summary":%s}]}]}\n' \
  "$(s 1000 "${FAKE_A_LINES:-800}" 2000 "${FAKE_A_REGIONS:-1400}")" "$(s 400 "${FAKE_B_LINES:-240}" 800 "${FAKE_B_REGIONS:-400}")" > "$o/llvm-cov.json"
printf 'build_exit=0\ntest_exit=%s\nmerge_exit=0\nexport_exit=0\nbuild_s=1\ntest_s=1\nmerge_s=0\nexport_s=0\nobjects=2\nprofraw=3\ntoolchain=rustc 1.98.1\nllvm=LLVM version 22\n' "${FAKE_TEST_EXIT:-0}" > "$o/status.env"
exit 0
P
chmod +x "$BIN"/*

# cov <args...>: ci/coverage.sh in the fixture with a clean environment; sets RC and OUT.
cov() {
  : > "$T/podman.log"; : > "$T/runs.log"
  set +e
  OUT=$(cd "$REPO" && env -i HOME="$T/home" PATH="$BIN:/usr/local/bin:/usr/bin:/bin" \
    CI_ENV_FILE="$T/no-ci.env" CI_CACHE="$CI_CACHE" CI_LOCK="$LOCK" CI_COVERAGE_CACHE="$COV_CACHE" \
    FAKE_LOG="$T/podman.log" FAKE_RUNS="$T/runs.log" FAKE_LOCK="$LOCK" ${EXTRA:-} bash ci/coverage.sh "$@" 2>&1)
  RC=$?
  set -e
}
J=$REPO/out/coverage
jqr() { jq -r "$1" "$2" 2>/dev/null || echo "unreadable $2"; }
base_sum() { sha256sum "$REPO/ci/coverage-baseline.json" | cut -c1-64; }
drop_report() { rm -f "${J:?}/${SHA:?}.json"; }

cov run
r=$J/$SHA.json
if [ "$RC" = 0 ] && [ "$(jqr '[.commit, .dirty, .source, .status, (.duration_s | type)] | @tsv' "$r")" = "$SHA	false	commit	green	number" ]; then
  pass "make coverage: out/coverage/<commit>.json with the commit, dirty false, source commit, the duration"
else fail "make coverage: out/coverage/<commit>.json with the commit, dirty false, source commit, the duration" "exit $RC, $(tail -n 3 <<<"$OUT" | tr '\n' ' ')"; fi
if [ "$(jqr '[.crates.a.lines.percent, .crates.b.regions.percent] | @tsv' "$r")" = "80	50" ]; then pass "make coverage: per crate percents in the report"
else fail "make coverage: per crate percents in the report" "$(jqr ".crates | tojson" "$r")"; fi
run=$(cat "$T/runs.log")
if grep -q "held=yes" <<<"$run"; then pass "make coverage measures under the build lock of make ci (CI_LOCK)"
else fail "make coverage measures under the build lock of make ci (CI_LOCK)" "[$run]"; fi
written=$(find "$CI_CACHE" -type f -newer "$STAMP" ! -path "$CI_CACHE/vendor/*" | head -n 3)
if grep -q "target=$COV_CACHE/target " <<<"$run" && grep -q "src=$COV_CACHE/src " <<<"$run" && grep -q "sccache=none" <<<"$run" && [ -z "$written" ]; then
  pass "make coverage builds in its own source and target, without sccache, and writes no file of make ci's cache"
else fail "make coverage builds in its own source and target, without sccache, and writes no file of make ci's cache" "[$run] [$written]"; fi
# The source is an overlay (:O): writable in the container, where a process started with
# env_clear() writes its default_*.profraw, and thrown away after; the copy on the host stays.
if grep -q "srcopt=O " <<<"$run"; then pass "make coverage mounts the source as a throwaway overlay"
else fail "make coverage mounts the source as a throwaway overlay" "[$run]"; fi
if grep 'inside-coverage.sh' "$T/podman.log" | grep -q -- '--network=none'; then pass "make coverage measures with no network"
else fail "make coverage measures with no network" "$(grep inside-coverage "$T/podman.log" | head -n 1)"; fi
if [ -n "$WANT_FEATURES" ] && grep -q "features=$WANT_FEATURES\$" <<<"$run"; then pass "make coverage measures the feature set of make ci (ALL_FEATURES of ci/inside.sh)"
else fail "make coverage measures the feature set of make ci (ALL_FEATURES of ci/inside.sh)" "want [$WANT_FEATURES], [$run]"; fi

base_before=$(base_sum)
EXTRA="FAKE_A_LINES=797" cov run
if [ "$RC" = 1 ] && [ "$(jqr '.status' "$J/$SHA.json")" = red ] && grep -q 'a .*lines.*-0.30' <<<"$OUT"; then pass "make coverage: red on a baseline that drops 0.3 pt"
else fail "make coverage: red on a baseline that drops 0.3 pt" "exit $RC, $(tail -n 4 <<<"$OUT" | tr '\n' ' ')"; fi
if [ "$(base_sum)" = "$base_before" ]; then pass "make coverage never writes the baseline"
else fail "make coverage never writes the baseline" "changed"; fi

EXTRA="FAKE_TEST_EXIT=101" cov run
if [ "$RC" = 1 ] && [ "$(jqr '[.status, .tests.exit] | @tsv' "$J/$SHA.json")" = "red	101" ]; then pass "make coverage: red when a test fails"
else fail "make coverage: red when a test fails" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
EXTRA="FAKE_TEST_EXIT=101 FAKE_A_LINES=900" cov update
if [ "$RC" = 1 ] && [ "$(base_sum)" = "$base_before" ]; then pass "coverage-update refuses a measure with a failed test"
else fail "coverage-update refuses a measure with a failed test" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi

# A modified tree: make coverage measures the commit, not the edit; CI_WORKTREE=1 measures
# the edit, says dirty, and coverage-update refuses it.
echo edited > "$REPO/probe.txt"
cov run
if [ "$RC" = 0 ] && grep -q 'probe=probe ' "$T/runs.log" && [ "$(jqr '[.dirty, .source] | @tsv' "$J/$SHA.json")" = "false	commit" ]; then
  pass "a modified tree: make coverage measures the commit (dirty false)"
else fail "a modified tree: make coverage measures the commit (dirty false)" "exit $RC, [$(cat "$T/runs.log")]"; fi
EXTRA="CI_WORKTREE=1" cov run
d=$J/$SHA+dirty.json
if [ -s "$d" ] && grep -q 'probe=edited ' "$T/runs.log" && [ "$(jqr '[.dirty, .source, .short] | @tsv' "$d")" = "true	worktree	${SHA:0:7}+dirty" ]; then
  pass "a modified tree with CI_WORKTREE=1: <commit>+dirty.json, dirty true"
else fail "a modified tree with CI_WORKTREE=1: <commit>+dirty.json, dirty true" "exit $RC, [$(cat "$T/runs.log")] $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
EXTRA="CI_WORKTREE=1 FAKE_A_LINES=900" cov update
if [ "$RC" = 1 ] && grep -q 'dirty' <<<"$OUT" && [ "$(base_sum)" = "$base_before" ]; then
  pass "coverage-update refuses a modified tree"
else fail "coverage-update refuses a modified tree" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
git -C "$REPO" checkout -q -- probe.txt

# coverage-update: raises, never lowers, unless CI_COVERAGE_ALLOW_DROP=1.
drop_report
EXTRA="FAKE_A_LINES=812 FAKE_A_REGIONS=1398" cov update
if [ "$RC" = 0 ] && [ "$(jq -c '[.crates.a.lines, .crates.a.regions, .crates.b.lines]' "$REPO/ci/coverage-baseline.json")" = "[81.2,70,60]" ]; then
  pass "coverage-update raises a crate and keeps a value that went down within the tolerance"
else fail "coverage-update raises a crate and keeps a value that went down within the tolerance" "exit $RC, $(jq -c . "$REPO/ci/coverage-baseline.json")"; fi
# The baseline is published with the code: it names no commit of the private history.
m=$(jq -c '[(.measured | has("commit")), (.measured.date | type), (.measured.toolchain | type)]' "$REPO/ci/coverage-baseline.json")
if [ "$m" = '[false,"string","string"]' ] && ! grep -q "${SHA:0:7}" "$REPO/ci/coverage-baseline.json"; then pass "the baseline names the date and toolchain of its measure, no commit"
else fail "the baseline names the date and toolchain of its measure, no commit" "[$m] $(jq -c .measured "$REPO/ci/coverage-baseline.json")"; fi
n=$(grep -c 'inside-coverage' "$T/runs.log" || true)
EXTRA="FAKE_A_LINES=812 FAKE_A_REGIONS=1398" cov update
if [ "$RC" = 0 ] && [ "$(grep -c 'inside-coverage' "$T/runs.log" || true)" = 0 ] && [ "$n" = 1 ]; then
  pass "coverage-update reuses the clean report of HEAD instead of measuring again"
else fail "coverage-update reuses the clean report of HEAD instead of measuring again" "exit $RC, measures: $n then $(grep -c 'inside-coverage' "$T/runs.log" || true)"; fi
drop_report
EXTRA="FAKE_A_LINES=700" cov update
if [ "$RC" = 1 ] && [ "$(jq -c '.crates.a.lines' "$REPO/ci/coverage-baseline.json")" = 81.2 ]; then pass "coverage-update refuses a drop past the tolerance"
else fail "coverage-update refuses a drop past the tolerance" "exit $RC, $(jq -c . "$REPO/ci/coverage-baseline.json")"; fi
EXTRA="FAKE_A_LINES=700 CI_COVERAGE_ALLOW_DROP=1" cov update
if [ "$RC" = 0 ] && [ "$(jq -c '.crates.a.lines' "$REPO/ci/coverage-baseline.json")" = 70 ]; then pass "coverage-update lowers with CI_COVERAGE_ALLOW_DROP=1"
else fail "coverage-update lowers with CI_COVERAGE_ALLOW_DROP=1" "exit $RC, $(jq -c . "$REPO/ci/coverage-baseline.json")"; fi
printf '{"schema": 1, "crates": {' > "$REPO/ci/coverage-baseline.json"
drop_report
cov update
if [ "$RC" = 2 ] && [ "$(cat "$REPO/ci/coverage-baseline.json")" = '{"schema": 1, "crates": {' ]; then pass "coverage-update leaves an unreadable baseline alone (exit 2)"
else fail "coverage-update leaves an unreadable baseline alone (exit 2)" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
rm -f "${REPO:?}/ci/coverage-baseline.json"
cov update
if [ "$RC" = 0 ] && [ "$(jq -c '[.schema, .tolerance_pt, .crates.a.lines, .crates.b.regions]' "$REPO/ci/coverage-baseline.json")" = "[1,0.2,80,50]" ]; then
  pass "coverage-update writes a first baseline from the measure"
else fail "coverage-update writes a first baseline from the measure" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
git -C "$REPO" checkout -q -- ci/coverage-baseline.json

rm -rf "${CI_CACHE:?}/vendor/${KEY:?}"
cov run
if [ "$RC" = 2 ] && grep -q 'make ci-refresh' <<<"$OUT" && ! grep -q 'inside-coverage' "$T/podman.log"; then pass "make coverage stops without the vendored crates of Cargo.lock (no network)"
else fail "make coverage stops without the vendored crates of Cargo.lock (no network)" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi

[ "$FAILS" = 0 ] || { echo "$FAILS failed"; exit 1; }
echo "coverage.sh: all passed"
