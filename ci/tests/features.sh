#!/usr/bin/env bash
# Tests of ci/features.py: a minimal FEATURES.toml, one refusal per rule, and this tree's own
# FEATURES.md up to date. Run by ci/inside.sh (step features), or by hand: bash ci/tests/features.sh
set -euo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
FAILS=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s: %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }
PY=${PYTHON:-python3}
if ! command -v "$PY" >/dev/null; then
  # The CI image and the GitHub runner ship python3: there the skip is a failure.
  if [ -n "${X0NE_REQUIRE_CI_TOOLS:-}" ]; then fail python3 "not found"; exit 1; fi
  echo "skip: python3 not found"; exit 0
fi
SCRIPT=$SRC/ci/features.py

cat > "$T/runs.md" <<'RUNS'
# Runs
## 2026-10-14
## 2026-10-17
RUNS
cat > "$T/base.toml" <<'TOML'
[[feature]]
id = "verified-vm"
level = "hardware"
order = 1
since = "2026-10-12"
hardware_since = "2026-10-14"
title_en = "The VM is verified first"
sentence_en = "A relay checks the VM before anything leaves your machine."
proof = "CHANGELOG 2026-10-12"
limit = "Not end-to-end confidential: the host sees sizes."

[[feature]]
id = "old-client"
level = "retired"
order = 1
since = "2026-09-18"
title_en = "The old client"
sentence_en = "It sent the question before any check."
proof = "CHANGELOG 2026-09-18"
replaced_by = "verified-vm"

[[feature]]
id = "public-log"
level = "planned"
order = 1
title_en = "A public log"
sentence_en = "Planned: a public transparency log."
proof = "README"
TOML

# check <toml> <md>: --check against the fixture runs; RC and OUT are left for the caller.
check() {
  RC=0
  OUT=$("$PY" -I "$SCRIPT" --check --toml "$1" --md "$2" --runs "$T/runs.md" 2>&1) || RC=$?
}

# ---- the minimal list is accepted, written, and then up to date
RC=0; OUT=$("$PY" -I "$SCRIPT" --toml "$T/base.toml" --md "$T/base.md" --runs "$T/runs.md" 2>&1) || RC=$?
if [ "$RC" = 0 ] && [ -s "$T/base.md" ]; then pass "writes FEATURES.md"; else fail "writes FEATURES.md" "exit $RC: $OUT"; fi
check "$T/base.toml" "$T/base.md"
if [ "$RC" = 0 ] && grep -q 'check: ok' <<<"$OUT"; then pass "check ok when up to date"; else fail "check ok when up to date" "exit $RC: $OUT"; fi
if grep -q '^1 measured on hardware since 2026-10-14, 0 tested, 1 planned, 1 replaced.$' "$T/base.md" \
   && grep -q 'Replaced by \*\*The VM is verified first\*\* (`verified-vm`)' "$T/base.md"; then
  pass "summary and successor in FEATURES.md"
else fail "summary and successor in FEATURES.md" "$(grep -n 'measured\|Replaced by' "$T/base.md")"; fi

# ---- one refusal per rule: refused <name> <expected message> <sed expression...>
refused() {
  local name=$1 want=$2; shift 2
  sed "$@" "$T/base.toml" > "$T/case.toml"
  check "$T/case.toml" "$T/base.md"
  if [ "$RC" = 1 ] && grep -qF "REFUSED" <<<"$OUT" && grep -qF "$want" <<<"$OUT"; then pass "$name"
  else fail "$name" "exit $RC, wanted '$want': $OUT"; fi
}
refused "duplicate id" "duplicate id" -e 's/^id = "old-client"/id = "verified-vm"/'
refused "hardware entry without hardware_since" "without a hardware_since" -e '/^hardware_since/d'
refused "hardware_since outside the runs" "is not a run date" -e 's/^hardware_since = "2026-10-14"/hardware_since = "2026-10-15"/'
refused "hardware_since before since" "before since" -e 's/^since = "2026-10-12"/since = "2026-10-16"/'
refused "hardware_since on a tested entry" "hardware_since on a local entry" -e '0,/^level = "hardware"/s//level = "local"/'
refused "jargon in a title" "say it in plain words" -e 's/The VM is verified first/The RTMR0 is verified first/'
refused "title too long" "words (2 to 9)" -e 's/The VM is verified first/The VM is verified first and then once more again/'
refused "sentence too long" "(at most 24)" -e 's/A relay checks the VM before anything leaves your machine./A relay on your machine checks the VM and its proof before anything at all is allowed to leave your machine for the cloud, every single time./'
# The publication gate refuses a task number anywhere in the tree: the fixture's is assembled at run time.
HASH='#'
refused "task number" "carries a task number" -e "s/^proof = \"CHANGELOG 2026-10-12\"/proof = \"CHANGELOG ${HASH}123\"/"
refused "commit hash" "carries a commit hash" -e 's/^proof = "CHANGELOG 2026-10-12"/proof = "commit 0badf00"/'
refused "end-to-end without a negation" "claims end-to-end confidentiality" -e 's/Not end-to-end confidential: the host sees sizes./End-to-end confidential./'
refused "planned entry with a date" "a planned entry has no since" -e 's/^id = "public-log"/id = "public-log"\nsince = "2026-10-01"/'
refused "replaced_by names no entry" "is not an id" -e 's/^replaced_by = "verified-vm"/replaced_by = "nobody"/'
refused "retired entry without replaced_by" "names no replaced_by" -e '/^replaced_by/d'
refused "unknown field" "unknown field foo" -e 's/^id = "public-log"/id = "public-log"\nfoo = "bar"/'
refused "missing sentence" "sentence_en missing" -e '/^sentence_en = "Planned/d'
refused "unknown level" "unknown level" -e 's/^level = "planned"/level = "soon"/'
# Two entries of one level with one order.
{ cat "$T/base.toml"; printf '\n[[feature]]\nid = "second-log"\nlevel = "planned"\norder = 1\ntitle_en = "A second log"\nsentence_en = "Planned too."\nproof = "README"\n'; } > "$T/case.toml"
check "$T/case.toml" "$T/base.md"
if [ "$RC" = 1 ] && grep -qF "share an order" <<<"$OUT"; then pass "two entries share an order"; else fail "two entries share an order" "exit $RC: $OUT"; fi
# A stale FEATURES.md.
cp "$T/base.md" "$T/stale.md"; echo "edited by hand" >> "$T/stale.md"
check "$T/base.toml" "$T/stale.md"
if [ "$RC" = 1 ] && grep -qF "is not what" <<<"$OUT"; then pass "stale FEATURES.md refused"; else fail "stale FEATURES.md refused" "exit $RC: $OUT"; fi
# A missing FEATURES.md.
check "$T/base.toml" "$T/absent.md"
if [ "$RC" = 1 ] && grep -qF "is not what" <<<"$OUT"; then pass "missing FEATURES.md refused"; else fail "missing FEATURES.md refused" "exit $RC: $OUT"; fi
# An unreadable TOML exits 2 (tomllib's error), not a traceback hidden in a success.
echo 'id = ' > "$T/case.toml"
RC=0; OUT=$("$PY" -I "$SCRIPT" --check --toml "$T/case.toml" --md "$T/base.md" --runs "$T/runs.md" 2>&1) || RC=$?
if [ "$RC" != 0 ]; then pass "unreadable TOML refused (exit $RC)"; else fail "unreadable TOML refused" "exit 0: $OUT"; fi

# ---- this tree: FEATURES.md is what the script writes from FEATURES.toml
RC=0; OUT=$(cd "$SRC" && "$PY" -I ci/features.py --check 2>&1) || RC=$?
if [ "$RC" = 0 ]; then pass "this tree's FEATURES.md is up to date"; else fail "this tree's FEATURES.md is up to date" "exit $RC: $OUT"; fi

[ "$FAILS" = 0 ] || { echo "$FAILS failure(s)"; exit 1; }
echo "all features.py tests passed"
