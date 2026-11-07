#!/usr/bin/env bash
# Replays the "Draft release" step of .github/workflows/release.yml, read from the
# file (never a copy), against a fake gh that keeps the notes file it is given.
# GitHub refuses a release body over 125,000 characters ("body is too long"), and
# the step then fails after the attestation, with no draft: on 2026-11-07 the
# 0.2.0-beta.7 section of this CHANGELOG was 125,589. Cases: this repository's
# CHANGELOG at its own version, a short section (passed whole, pre-release flag),
# a section over the cap (replaced by a link to CHANGELOG.md), no section.
# Run by ci/tests/ci-sh.sh, or by hand: bash ci/tests/release-notes.sh.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)
WF=$SRC/.github/workflows/release.yml
STEP='Draft release'
CAP=125000
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
FAILS=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s: %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }

awk -v name="$STEP" '
  index($0, "- name: " name) { found = 1; next }
  found && /^ *run: \|/ { inrun = 1; next }
  inrun && /^ *- (name|uses):/ { exit }
  inrun { print }
' "$WF" | sed 's/^          //' > "$T/step.sh"
[ -s "$T/step.sh" ] || { echo "FAIL  step \"$STEP\" not found in $WF"; exit 1; }
grep -q 'gh release create' "$T/step.sh" || { echo "FAIL  the extracted step creates no release"; exit 1; }

# TEST fake gh: keeps the notes file and the arguments, reaches nothing.
mkdir -p "$T/bin"
cat > "$T/bin/gh" <<'GH'
#!/usr/bin/env bash
set -euo pipefail
[ "${1:-} ${2:-}" = "release create" ] || { echo "fake gh: unexpected call: $*" >&2; exit 2; }
printf '%s\n' "$*" > "$FAKE_OUT/args"
while [ $# -gt 0 ]; do
  if [ "$1" = --notes-file ]; then cp "$2" "$FAKE_OUT/notes.md"; fi
  shift
done
GH
chmod +x "$T/bin/gh"

# draft <case> <tag> <CHANGELOG file>: runs the step in a fresh workspace.
draft() {
  local w=$T/$1
  mkdir -p "$w/dist" "$w/out"
  cp "$3" "$w/CHANGELOG.md"
  : > "$w/dist/SHA256SUMS"
  (cd "$w" && PATH="$T/bin:$PATH" FAKE_OUT="$w/out" TAG=$2 GITHUB_REPOSITORY=x0ne-test/x0ne-Engine \
    bash -euo pipefail "$T/step.sh") > "$w/log" 2>&1
}
chars() { LC_ALL=C.UTF-8 wc -m < "$1"; }

version=$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$SRC/Cargo.toml" | head -n 1)
if ! draft own "v$version" "$SRC/CHANGELOG.md"; then fail own "the step failed: $(tr '\n' ' ' < "$T/own/log")"
elif [ ! -s "$T/own/out/notes.md" ]; then fail own "no notes given to gh"
elif [ "$(chars "$T/own/out/notes.md")" -gt "$CAP" ]; then fail own "notes of v$version: $(chars "$T/own/out/notes.md") characters, over $CAP"
else pass "own: notes of v$version, $(chars "$T/own/out/notes.md") characters"; fi

printf '# Changelog\n\n## Unreleased\n\n## 1.2.3-beta.1 — 2026-01-02\n\n- One change.\n\n## 1.2.2 — 2026-01-01\n\n- Older.\n' > "$T/short.md"
if ! draft short v1.2.3-beta.1 "$T/short.md"; then fail short "the step failed: $(tr '\n' ' ' < "$T/short/log")"
elif [ "$(cat "$T/short/out/notes.md")" != "$(printf '\n- One change.\n')" ]; then fail short "notes: $(tr '\n' '|' < "$T/short/out/notes.md")"
elif ! grep -q -- '--prerelease' "$T/short/out/args"; then fail short "no --prerelease for a tag with a hyphen"
else pass "short: the section whole, --prerelease"; fi

line=$(printf 'x%.0s' $(seq 1 100))
{ printf '# Changelog\n\n## 2.0.0 — 2026-01-03\n\n'
  for _ in $(seq 1 1300); do printf -- '- %s\n' "$line"; done
  printf '\n## 1.0.0 — 2026-01-01\n\n- Older.\n'; } > "$T/long.md"
if ! draft long v2.0.0 "$T/long.md"; then fail long "the step failed: $(tr '\n' ' ' < "$T/long/log")"
elif [ "$(chars "$T/long/out/notes.md")" -gt "$CAP" ]; then fail long "notes: $(chars "$T/long/out/notes.md") characters, over $CAP"
elif ! grep -q 'https://github.com/x0ne-test/x0ne-Engine/blob/v2.0.0/CHANGELOG.md' "$T/long/out/notes.md"; then
  fail long "no link to the tag's CHANGELOG.md: $(head -c 300 "$T/long/out/notes.md")"
else pass "long: a section over $CAP characters gives a link to CHANGELOG.md"; fi

printf '# Changelog\n\n## 1.0.0 — 2026-01-01\n\n- Older.\n' > "$T/none.md"
if ! draft none v3.0.0 "$T/none.md"; then fail none "the step failed: $(tr '\n' ' ' < "$T/none/log")"
elif [ "$(cat "$T/none/out/notes.md")" != "See CHANGELOG.md." ]; then fail none "notes: $(cat "$T/none/out/notes.md")"
elif grep -q -- '--prerelease' "$T/none/out/args"; then fail none "--prerelease for v3.0.0"
else pass "none: no section, a pointer to CHANGELOG.md"; fi

[ "$FAILS" = 0 ] || { echo "release-notes: $FAILS failed"; exit 1; }
echo "release-notes: 4 cases passed"
