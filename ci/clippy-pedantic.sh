#!/usr/bin/env bash
# Prints the lint flags of the `clippy` job of .github/workflows/ci.yml (its PEDANTIC
# variable), one word per line: ci/inside.sh lints with the same list, kept in one place.
# Usage: ci/clippy-pedantic.sh [ci.yml]. Exit 0 printed; 2 when the file has no single
# PEDANTIC block, or a word that is not part of a `-W|-A|-D clippy::<lint>` pair, or the
# list does not open on `-W clippy::pedantic`. Tested by ci/tests/ci-sh.sh.
set -euo pipefail
f=${1:-.github/workflows/ci.yml}
die() { printf 'clippy-pedantic: %s\n' "$*" >&2; exit 2; }
[ -r "$f" ] || die "cannot read $f"
# The folded scalar `PEDANTIC: >-`: every following line indented deeper than the key.
block=$(awk '
  on && /^[[:space:]]*$/ { next }
  on { if (match($0, /[^ ]/) <= ind) on = 0; else { print; next } }
  /^[[:space:]]*PEDANTIC:[[:space:]]*>-[[:space:]]*$/ { on = 1; ind = match($0, /[^ ]/); n++ }
  END { if (n != 1) exit 3 }' "$f") || die "$f: not exactly one PEDANTIC: >- block"
read -r -a w <<< "$(tr '\n' ' ' <<< "$block")"
[ "${#w[@]}" -ge 2 ] && [ $(( ${#w[@]} % 2 )) = 0 ] || die "$f: ${#w[@]} words, not flag and lint pairs"
[ "${w[0]} ${w[1]}" = "-W clippy::pedantic" ] || die "$f: the list opens on '${w[0]} ${w[1]}', not '-W clippy::pedantic'"
for ((i = 0; i < ${#w[@]}; i += 2)); do
  [[ ${w[i]} =~ ^-[WAD]$ && ${w[i+1]} =~ ^clippy::[a-z0-9_]+$ ]] || die "$f: '${w[i]} ${w[i+1]}' is not a -W|-A|-D clippy::<lint> pair"
done
printf '%s\n' "${w[@]}"
