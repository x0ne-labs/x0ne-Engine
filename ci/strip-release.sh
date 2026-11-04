#!/usr/bin/env bash
# The strip of every delivered binary: what ships is the stripped file, its symbols and line
# tables go to a debug file found by build-id. Used by ci/inside-deb.sh and scripts/sbom.sh;
# ci/comment-invariance.sh checks the result.
#
#   ci/strip-release.sh <debug dir> <binary>...
#
# Each <binary> (built by the `release` profile of Cargo.toml: symbols and line tables) is
# rewritten in place:
#   1. objcopy --only-keep-debug --compress-debug-sections: the debug file;
#   2. strip --remove-section=.comment --remove-section=.note: the flags of dh_strip for an
#      executable, so its own pass in the packaging (equivs) finds nothing left to change;
#   3. the GNU build-id note is recomputed over the stripped bytes (its descriptor zeroed,
#      SHA-256, first <descsz> bytes) and written into the binary and into the debug file,
#      which goes to <debug dir>/.build-id/<first 2 hex>/<rest>.debug (gdb, coredumpctl,
#      debuginfod: `set debug-file-directory <debug dir>`).
# No .gnu_debuglink: it carries a CRC-32 of the whole debug file.
# Measured 2026-10-25: the linker hashes the whole unstripped file into the build-id,
# symbol names included, and a comment renames ThinLTO's .llvm.<hash> locals.
# Prints one line per binary: sha256, build-id, size, debug file.
set -euo pipefail
die() { printf 'strip-release: %s\n' "$*" >&2; exit 2; }
[ $# -ge 2 ] || die "usage: ci/strip-release.sh <debug dir> <binary>..."
dbg=$1; shift
mkdir -p "$dbg"
# A refusal after objcopy leaves no <binary>.debug.tmp behind.
tmp=; trap '[ -z "$tmp" ] || rm -f "$tmp"' EXIT

# note_off <file>: file offset of the one .note.gnu.build-id section, which must hold one note
# and nothing else (16-byte header, "GNU\0", the descriptor rounded up to 4).
note_off() {
  local lines n hex size descsz
  lines=$(readelf -S -W "$1" 2>/dev/null | sed -n 's/^ *\[ *[0-9]*\] \.note\.gnu\.build-id  *NOTE  *[0-9a-f]*  *\([0-9a-f]*\)  *\([0-9a-f]*\) .*/\1 \2/p') || true
  n=$(printf '%s' "$lines" | grep -c . || true)
  [ "$n" = 1 ] || die "$1: $n .note.gnu.build-id sections"
  read -r hex size <<<"$lines"
  descsz=$(od -An -tu4 -j $((16#$hex + 4)) -N 4 "$1" | tr -d ' ')
  [ -n "$descsz" ] && [ $((16#$size)) = $((16 + (descsz + 3) / 4 * 4)) ] \
    || die "$1: .note.gnu.build-id is $((16#$size)) bytes, its note says ${descsz:-?}"
  echo $((16#$hex))
}
# note_desc <file> <offset>: descriptor size, after checking the note is GNU's NT_GNU_BUILD_ID.
note_desc() {
  local namesz descsz type name
  read -r namesz descsz type < <(od -An -tu4 -j "$2" -N 12 "$1")
  name=$(od -An -c -j $(($2 + 12)) -N 4 "$1" | tr -d ' ')
  [ "$namesz" = 4 ] && [ "$type" = 3 ] && [ "$name" = 'GNU\0' ] || die "$1: not a GNU build-id note at $2"
  [ "$descsz" -ge 8 ] && [ "$descsz" -le 32 ] || die "$1: build-id of $descsz bytes"
  echo "$descsz"
}
# put_id <file> <hex>: writes the build-id <hex> into the note of <file>.
put_id() {
  local off size
  off=$(note_off "$1"); size=$(note_desc "$1" "$off")
  [ ${#2} = $((size * 2)) ] || die "$1: build-id of $size bytes, $2 given"
  printf "$(printf '%s' "$2" | sed 's/../\\x&/g')" | dd of="$1" bs=1 seek=$((off + 16)) conv=notrunc status=none
}

for bin in "$@"; do
  [ -f "$bin" ] || die "$bin: not a file"
  # The whole listing first: grep -q exits at the match, and readelf killed by SIGPIPE turned a
  # binary with symbols into "no symbol table" under pipefail (x0ne-engine-cloud: 26 and 1 in
  # two series of 2000, 2026-10-25).
  sections=$(readelf -S -W "$bin" 2>/dev/null) || die "$bin: not an ELF file readelf can read"
  grep -q ' \.symtab ' <<<"$sections" || die "$bin: no symbol table, stripped already (build it with --release)"
  # The note is checked before the binary is touched: a refusal leaves it as it was.
  off=$(note_off "$bin"); note_desc "$bin" "$off" > /dev/null
  tmp=$bin.debug.tmp
  objcopy --only-keep-debug --compress-debug-sections "$bin" "$tmp"
  strip --remove-section=.comment --remove-section=.note "$bin"
  off=$(note_off "$bin"); size=$(note_desc "$bin" "$off")
  dd if=/dev/zero of="$bin" bs=1 seek=$((off + 16)) count="$size" conv=notrunc status=none
  id=$(sha256sum < "$bin" | cut -c1-$((size * 2)))
  put_id "$bin" "$id"
  put_id "$tmp" "$id"
  # One build-id, the new one, in both files, and a binary readelf reads without a warning.
  for f in "$bin" "$tmp"; do
    [ "$(readelf -n "$f" 2>/dev/null | sed -n 's/.*Build ID: //p')" = "$id" ] || die "$f: build-id not $id after the rewrite"
  done
  [ -z "$(readelf -a -W "$bin" 2>&1 >/dev/null)" ] || die "$bin: readelf warns after the rewrite"
  mkdir -p "$dbg/.build-id/${id:0:2}"
  mv "$tmp" "$dbg/.build-id/${id:0:2}/${id:2}.debug"
  printf '%s  %s  build-id %s  %s bytes  .build-id/%s/%s.debug\n' \
    "$(sha256sum < "$bin" | cut -c1-64)" "$(basename "$bin")" "$id" "$(stat -c %s "$bin")" "${id:0:2}" "${id:2}"
done
