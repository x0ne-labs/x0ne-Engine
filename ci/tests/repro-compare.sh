#!/usr/bin/env bash
# Tests of ci/repro-compare.sh: two build output trees, byte for byte. No
# container, no build. Run by ci/tests/ci-sh.sh, or by hand.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)
CMP=$SRC/ci/repro-compare.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
FAILS=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s: %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }

# tree <dir>: the outputs of one build, and the environment it ran in.
tree() {
  mkdir -p "$1/debs" "$1/cvm"
  printf 'deb one\n' > "$1/debs/x0ne-engine_0.2.0_amd64.deb"
  printf 'deb two\n' > "$1/debs/x0ne-infer_0.2.0_amd64.deb"
  printf 'disk\n' > "$1/cvm/x0ne-cvm-0.2.3.tar.gz"
  printf '{"kernel": "%s"}\n' "$2" > "$1/build-env.json"
}
# run <A> <B>: RC, OUT (stdout and stderr), JSON (the report).
run() {
  RC=0
  OUT=$(bash "$CMP" "$1" "$2" --json "$T/report.json" 2>&1) || RC=$?
  JSON=$(cat "$T/report.json" 2>/dev/null || true)
}

tree "$T/a" 6.12; tree "$T/b" 6.1
run "$T/a" "$T/b"
if [ "$RC" = 0 ] && grep -q '"identical": true' <<<"$JSON" && grep -q '"files": 3' <<<"$JSON"; then
  pass "two builds with the same bytes are identical, whatever their environment"
else fail "two builds with the same bytes are identical, whatever their environment" "exit $RC, $OUT"; fi
if grep -q '"kernel": "6.12"' <<<"$JSON" && grep -q '"kernel": "6.1"' <<<"$JSON"; then
  pass "the report carries both environments"
else fail "the report carries both environments" "$JSON"; fi

# SECURITY: one byte changed in one file of B: not reproducible, the file named.
printf 'deb twO\n' > "$T/b/debs/x0ne-infer_0.2.0_amd64.deb"
run "$T/a" "$T/b"
if [ "$RC" = 1 ] && grep -q 'DIFFER.*debs/x0ne-infer_0.2.0_amd64.deb' <<<"$OUT" \
  && grep -q '"identical": false' <<<"$JSON"; then pass "one changed byte is a difference, and the file is named"
else fail "one changed byte is a difference, and the file is named" "exit $RC, $OUT"; fi
printf 'deb two\n' > "$T/b/debs/x0ne-infer_0.2.0_amd64.deb"

# SECURITY: a file only one build made (missing, or extra): a difference.
rm "$T/b/cvm/x0ne-cvm-0.2.3.tar.gz"
run "$T/a" "$T/b"
if [ "$RC" = 1 ] && grep -q 'ONLY-A.*cvm/x0ne-cvm-0.2.3.tar.gz' <<<"$OUT"; then pass "a file missing from one build is a difference"
else fail "a file missing from one build is a difference" "exit $RC, $OUT"; fi
printf 'disk\n' > "$T/b/cvm/x0ne-cvm-0.2.3.tar.gz"
printf 'x\n' > "$T/b/cvm/extra.bin"
run "$T/a" "$T/b"
if [ "$RC" = 1 ] && grep -q 'ONLY-B.*cvm/extra.bin' <<<"$OUT"; then pass "a file only one build made is a difference"
else fail "a file only one build made is a difference" "exit $RC, $OUT"; fi
rm "$T/b/cvm/extra.bin"

# SECURITY: nothing to compare never passes as reproducible.
mkdir -p "$T/e1" "$T/e2"
run "$T/e1" "$T/e2"
if [ "$RC" = 2 ] && grep -qi 'no file' <<<"$OUT"; then pass "two empty trees are refused, not identical"
else fail "two empty trees are refused, not identical" "exit $RC, $OUT"; fi
run "$T/a" "$T/a"
if [ "$RC" = 2 ]; then pass "a tree compared with itself is refused"
else fail "a tree compared with itself is refused" "exit $RC, $OUT"; fi
run "$T/a" "$T/nowhere"
if [ "$RC" = 2 ]; then pass "a missing tree is refused"
else fail "a missing tree is refused" "exit $RC, $OUT"; fi

# SECURITY: a symbolic link could point both trees at one file: refused.
ln -s "$T/a/debs/x0ne-engine_0.2.0_amd64.deb" "$T/b/debs/link.deb"
run "$T/a" "$T/b"
if [ "$RC" = 2 ] && grep -q 'not a regular file' <<<"$OUT"; then pass "a symbolic link in a tree is refused"
else fail "a symbolic link in a tree is refused" "exit $RC, $OUT"; fi
rm "$T/b/debs/link.deb"

# A name the JSON report could not carry as it is: refused, not escaped.
printf 'q\n' > "$T/a/debs/bad\"name.deb"; printf 'q\n' > "$T/b/debs/bad\"name.deb"
run "$T/a" "$T/b"
if [ "$RC" = 2 ] && grep -q 'file name' <<<"$OUT"; then pass "a file name with a quote is refused"
else fail "a file name with a quote is refused" "exit $RC, $OUT"; fi
rm "$T/a/debs/bad\"name.deb" "$T/b/debs/bad\"name.deb"

# build-env.json differs by design; it is reported, never compared.
printf '{"kernel": "other"}\n' > "$T/b/build-env.json"
run "$T/a" "$T/b"
if [ "$RC" = 0 ]; then pass "build-env.json is not an output"
else fail "build-env.json is not an output" "exit $RC, $OUT"; fi

[ "$FAILS" = 0 ] || { echo "repro-compare: $FAILS failed"; exit 1; }
echo "repro-compare: all passed"
