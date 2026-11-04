#!/usr/bin/env bash
# Replays the "Tag is annotated and its signature verified by GitHub" step of
# .github/workflows/release.yml, read from the file (never a copy), against a
# fake `gh api` that answers the two calls from canned JSON and applies --jq
# with jq, as gh does. Four tags: signed and verified (accepted), annotated but
# unsigned, lightweight, tag object without .verification (each refused with
# exit 1, not a jq error). Run by ci/inside.sh (step ci-sh) or by hand:
# bash ci/tests/release-verify-tag.sh. Needs jq; without it the test says so and skips,
# unless X0NE_REQUIRE_JQ is set (ci/inside.sh does: the CI image ships jq), then it fails.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)
WF=$SRC/.github/workflows/release.yml
STEP='Tag is annotated and its signature verified by GitHub'
if ! command -v jq >/dev/null; then
  if [ -n "${X0NE_REQUIRE_JQ:-}" ]; then echo "FAIL  release-verify-tag: jq not found, X0NE_REQUIRE_JQ is set"; exit 1; fi
  echo "skip  release-verify-tag: jq not found"; exit 0
fi
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
FAILS=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s: %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }

# The step's `run:` block, dedented: from the line after `run: |` to the next step.
awk -v name="$STEP" '
  index($0, "- name: " name) { found = 1; next }
  found && /^ *run: \|/ { inrun = 1; next }
  inrun && /^ *- (name|uses):/ { exit }
  inrun { print }
' "$WF" | sed 's/^          //' > "$T/step.sh"
[ -s "$T/step.sh" ] || { echo "FAIL  step \"$STEP\" not found in $WF"; exit 1; }
grep -q 'gh api' "$T/step.sh" || { echo "FAIL  the extracted step calls no gh api"; exit 1; }

# gh api <path> --jq <filter>: the ref call answers ref.json, the tag call tag.json.
mkdir -p "$T/bin"
cat > "$T/bin/gh" <<'GH'
#!/usr/bin/env bash
set -euo pipefail
[ "${1:-}" = api ] && [ "${3:-}" = --jq ] || { echo "fake gh: unexpected call: $*" >&2; exit 2; }
case $2 in
  */git/ref/tags/*) f=$FAKE_GH_DIR/ref.json ;;
  */git/tags/*)     f=$FAKE_GH_DIR/tag.json ;;
  *) echo "fake gh: unknown path $2" >&2; exit 2 ;;
esac
exec jq -r "$4" "$f"
GH
chmod +x "$T/bin/gh"

# case <name> <ref json> <tag json> <expected exit> <expected output>
case_run() {
  local name=$1 ref=$2 tag=$3 want_rc=$4 want_out=$5 rc=0 out
  mkdir -p "$T/$name"
  printf '%s\n' "$ref" > "$T/$name/ref.json"
  printf '%s\n' "$tag" > "$T/$name/tag.json"
  out=$(cd "$T" && env -i PATH="$T/bin:/usr/bin:/bin" HOME="$T" \
    FAKE_GH_DIR="$T/$name" GH_TOKEN=fake TAG=v9.9.9 GITHUB_REPOSITORY=example/engine \
    bash "$T/step.sh" 2>&1) || rc=$?
  if [ "$rc" != "$want_rc" ]; then
    fail "$name" "exit $rc, expected $want_rc: $out"
  elif ! grep -qF -- "$want_out" <<<"$out"; then
    fail "$name" "output lacks \"$want_out\": $out"
  else
    pass "$name"
  fi
}

ANNOTATED='{"ref":"refs/tags/v9.9.9","object":{"type":"tag","sha":"a1"}}'
case_run signed-verified "$ANNOTATED" \
  '{"sha":"a1","tag":"v9.9.9","verification":{"verified":true,"reason":"valid"}}' \
  0 'signature: true valid'
case_run annotated-unsigned "$ANNOTATED" \
  '{"sha":"a1","tag":"v9.9.9","verification":{"verified":false,"reason":"unsigned"}}' \
  1 '::error::v9.9.9 is not signed by a verified key'
case_run lightweight '{"ref":"refs/tags/v9.9.9","object":{"type":"commit","sha":"c1"}}' \
  '{}' 1 '::error::v9.9.9 is a lightweight tag'
case_run no-verification "$ANNOTATED" '{"sha":"a1","tag":"v9.9.9"}' \
  1 '::error::v9.9.9 is not signed by a verified key'

[ "$FAILS" -eq 0 ] || { echo "$FAILS failure(s)"; exit 1; }
echo "release-verify-tag: 4 ok"
