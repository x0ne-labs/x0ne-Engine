#!/usr/bin/env bash
# The outputs of two builds of one commit, made in two environments, compared
# byte for byte (make repro-compare A=<dir> B=<dir>). Each tree holds what one
# build made: .deb files, binaries, the CVM tar, EFI and reference, as copied
# from its staging and output directories. `build-env.json` at the root of a
# tree describes where it was built (ci/build-env.sh); it is reported, never
# compared.
#
# Usage: ci/repro-compare.sh <tree A> <tree B> [--json <report>]
# Exit: 0 every file of each tree is in the other with the same SHA-256;
#       1 a file differs or is in one tree only (each one printed: DIFFER,
#         ONLY-A, ONLY-B); 2 nothing comparable (missing or empty tree, a tree
#         compared with itself, a symbolic link or other non-regular file, a
#         file name the report cannot carry).
set -euo pipefail
die() { printf 'repro-compare: %s\n' "$*" >&2; exit 2; }

[ $# -ge 2 ] || die "usage: ci/repro-compare.sh <tree A> <tree B> [--json <report>]"
A=$1 B=$2 REPORT=""
shift 2
while [ $# -gt 0 ]; do
  case $1 in
    --json) [ $# -ge 2 ] || die "--json needs a file"; REPORT=$2; shift 2 ;;
    *) die "unknown argument $1" ;;
  esac
done
for d in "$A" "$B"; do [ -d "$d" ] || die "$d: not a directory"; done
A=$(cd "$A" && pwd -P) B=$(cd "$B" && pwd -P)
[ "$A" != "$B" ] || die "$A is compared with itself"
for d in "$A" "$B"; do [[ $d =~ ^[A-Za-z0-9._+~/@:=,-]+$ ]] || die "tree $d: only letters, digits and ._+~/@:=,- are reported"; done

# files <tree>: its regular files, relative and sorted, build-env.json left out.
files() {
  local bad
  bad=$(cd "$1" && find . -mindepth 1 ! -type f ! -type d -print -quit)
  [ -z "$bad" ] || die "$1/${bad#./}: not a regular file"
  (cd "$1" && find . -type f ! -path ./build-env.json -print | sed 's#^\./##' | LC_ALL=C sort)
}
LA=$(files "$A") LB=$(files "$B")
[ -n "$LA$LB" ] || die "no file in $A nor in $B"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  [[ $f =~ ^[A-Za-z0-9._+~/@:=,-]+$ ]] || die "file name $f: only letters, digits and ._+~/@:=,- are reported"
done <<<"$LA"$'\n'"$LB"

sum() { sha256sum < "$1" | cut -c1-64; }
same=0 differ=0 rows=()
while IFS= read -r f; do
  [ -z "$f" ] && continue
  ha="" hb=""
  [ -f "$A/$f" ] && ha=$(sum "$A/$f")
  [ -f "$B/$f" ] && hb=$(sum "$B/$f")
  if [ -z "$hb" ]; then status=only-a; printf 'ONLY-A  %s  %s\n' "$ha" "$f"
  elif [ -z "$ha" ]; then status=only-b; printf 'ONLY-B  %s  %s\n' "$hb" "$f"
  elif [ "$ha" = "$hb" ]; then status=same; printf 'same    %s  %s\n' "$ha" "$f"
  else status=differ; printf 'DIFFER  %s %s  %s\n' "$ha" "$hb" "$f"; fi
  if [ "$status" = same ]; then same=$((same + 1)); else differ=$((differ + 1)); fi
  rows+=("    {\"file\": \"$f\", \"a\": \"$ha\", \"b\": \"$hb\", \"status\": \"$status\"}")
done < <(printf '%s\n%s\n' "$LA" "$LB" | LC_ALL=C sort -u)

identical=false; [ "$differ" = 0 ] && identical=true
printf 'repro-compare: %d same, %d different or in one tree only: %s\n' "$same" "$differ" \
  "$([ "$identical" = true ] && echo identical || echo NOT reproducible)"

if [ -n "$REPORT" ]; then
  env_of() { if [ -s "$1/build-env.json" ]; then cat "$1/build-env.json"; else echo null; fi; }
  {
    printf '{\n  "format": 1,\n  "identical": %s,\n  "files": %d,\n  "differ": %d,\n' "$identical" $((same + differ)) "$differ"
    printf '  "a": {"tree": "%s", "env": %s},\n' "$A" "$(env_of "$A")"
    printf '  "b": {"tree": "%s", "env": %s},\n' "$B" "$(env_of "$B")"
    printf '  "outputs": [\n'
    local_n=${#rows[@]}
    for i in "${!rows[@]}"; do
      if [ "$i" -lt $((local_n - 1)) ]; then printf '%s,\n' "${rows[$i]}"; else printf '%s\n' "${rows[$i]}"; fi
    done
    printf '  ]\n}\n'
  } > "$REPORT.tmp"
  mv "$REPORT.tmp" "$REPORT"
fi
[ "$identical" = true ] || exit 1
