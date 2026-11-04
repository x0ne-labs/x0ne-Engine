#!/usr/bin/env bash
# The check of make deb (ci/ci.sh deb_verify, in the packaging image, no network): each .deb of
# <work dir>/debs ships exactly its binaries, by name and sha256: x0ne-engine, those of
# bin/SHA256SUMS.engine; no other package, no usr/bin/x0ne-infer (the TD's server ships in the
# CVM image only, 2026-11-04), no binary in a subdirectory of usr/bin, and no
# ELF, executable or symlink outside usr/bin. Tested by ci/tests/deb-verify.sh.
#
#   ci/deb-verify.sh <work dir>
#
# Prints "same <deb> <sha256>  <name>" per binary. Exit 0 exact, 1 otherwise.
set -euo pipefail
[ $# = 1 ] || { echo "usage: ci/deb-verify.sh <work dir>" >&2; exit 2; }
w=$(cd "$1" && pwd)
x=$(mktemp -d)
trap 'rm -rf "$x"' EXIT
seen=
for d in "$w"/debs/*.deb; do
  [ -f "$d" ] || { echo "NO .deb in $w/debs"; exit 1; }
  p=$(dpkg-deb -f "$d" Package)
  case $p in
    x0ne-engine) want=$(cat "$w/bin/SHA256SUMS.engine") ;;
    *) echo "UNEXPECTED PACKAGE $p (${d##*/})"; exit 1 ;;
  esac
  [ -n "$want" ] || { echo "NO BINARY LISTED FOR $p"; exit 1; }
  seen="$seen $p"; cd /; rm -rf "$x/root"; dpkg-deb -x "$d" "$x/root"; cd "$x/root"; got=
  while IFS= read -r -d '' f; do
    elf=; [ -f "$f" ] && [ "$(head -c 4 "$f" | od -An -tx1 | tr -d ' \n')" = 7f454c46 ] && elf=1
    case $f in
      ./usr/bin/*/*) echo "IN A SUBDIRECTORY OF usr/bin: $p ${f#.}"; exit 1 ;;
      ./usr/bin/x0ne-infer) echo "THE SERVER IN A .deb: $p ships ${f#.}"; exit 1 ;;
      ./usr/bin/*) if [ -L "$f" ] || [ ! -f "$f" ]; then echo "NOT A REGULAR FILE: $p ${f#.}"; exit 1; fi
                   got="$got$(sha256sum < "$f" | cut -c1-64)  ${f##*/}"$'\n' ;;
      *) if [ -L "$f" ] || [ -n "$elf" ] || { [ -f "$f" ] && [ -x "$f" ]; }; then echo "OUTSIDE usr/bin: $p ${f#.}"; exit 1; fi ;;
    esac
  done < <(find . -mindepth 1 ! -type d -print0)
  if [ "$(printf '%s' "$got" | sort)" != "$(printf '%s\n' "$want" | sort)" ]; then
    echo "NOT THE BINARIES OF SHA256SUMS.engine: $p"; diff <(printf '%s\n' "$want" | sort) <(printf '%s' "$got" | sort) || true; exit 1
  fi
  printf '%s' "$got" | sed "s/^/same ${d##*/} /"
done
[ "$(tr ' ' '\n' <<<"$seen" | sed '/^$/d' | sort | tr '\n' ' ')" = "x0ne-engine " ] \
  || { echo "PACKAGES:$seen, want x0ne-engine"; exit 1; }
