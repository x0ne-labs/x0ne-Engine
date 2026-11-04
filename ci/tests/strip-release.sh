#!/usr/bin/env bash
# Tests of ci/strip-release.sh on small binaries built here (cc -g): the rewrite of a nominal
# binary, and the refusal of every malformed note or input, the input untouched and no
# .debug.tmp left. No container, no network. Run by ci/tests/ci-sh.sh, or by hand. Needs cc,
# binutils and perl; without them it says so and skips, unless X0NE_REQUIRE_CI_TOOLS is set
# (ci/inside.sh does: the CI image ships them), then it fails.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)
SR=$SRC/ci/strip-release.sh
for t in cc readelf objcopy strip perl; do
  command -v "$t" >/dev/null && continue
  if [ -n "${X0NE_REQUIRE_CI_TOOLS:-}" ]; then echo "FAIL  strip-release: $t not found, X0NE_REQUIRE_CI_TOOLS is set"; exit 1; fi
  echo "skip  strip-release: $t not found"; exit 0
done
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
FAILS=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s: %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }
cd "$T"
cat > hello.c <<'C'
#include <stdio.h>
int main(int c, char **v) { (void)v; printf("hello %d\n", c); return 0; }
C
cc -g -O1 -o base hello.c -Wl,--build-id=sha1

noteoff() { printf '%d' "0x$(readelf -S -W "$1" | sed -n 's/^ *\[ *[0-9]*\] \.note\.gnu\.build-id  *NOTE  *[0-9a-f]*  *\([0-9a-f]*\) .*/\1/p' | head -1)"; }
u32() { perl -e 'open F, "+<", $ARGV[0] or die; binmode F; seek F, $ARGV[1], 0; print F pack("V", $ARGV[2]); close F' "$@"; }
bytes() { perl -e 'open F, "+<", $ARGV[0] or die; binmode F; seek F, $ARGV[1], 0; print F $ARGV[2]; close F' "$@"; }
# run_sr <case> <input> [env...]: strip-release on a copy in <case>/; RC, ERR, D (the case dir).
run_sr() {
  local c=$1 in=$2; shift 2
  D=$T/$c; mkdir -p "$D"; cp "$in" "$D/bin"; cp "$in" "$D/orig"; RC=0
  env "$@" bash "$SR" "$D/debug" "$D/bin" > "$D/out" 2> "$D/err" || RC=$?
  ERR=$(head -c 300 "$D/err" | tr '\n' ' ')
}
# refused <case> [touched]: exit != 0 with a message of strip-release, no .debug.tmp, no debug
# file, and the input as it was (unless "touched": a refusal after the rewrite started).
refused() {
  local why=
  [ "$RC" != 0 ] || why="exit 0"
  case $ERR in "strip-release: "*) ;; *) why="$why, message '$ERR'" ;; esac
  [ ! -e "$D/bin.debug.tmp" ] || why="$why, bin.debug.tmp left"
  [ -z "$(find "$D/debug" -name '*.debug' 2>/dev/null)" ] || why="$why, a debug file written"
  [ "${2:-}" = touched ] || cmp -s "$D/bin" "$D/orig" || why="$why, input rewritten"
  if [ -z "$why" ]; then pass "$1: refused (exit $RC: ${ERR% })"; else fail "$1" "${why#, }"; fi
}
# rewritten <case>: exit 0; build-id = sha256(stripped, descriptor zeroed)[:descsz] in the binary
# and in its debug file under .build-id/; only the descriptor differs from a plain strip; the
# binary runs and readelf reads it without a warning; no .debug.tmp left.
rewritten() {
  local c=$1 off ds id want diff dbg did= run rerr
  if [ "$RC" != 0 ]; then fail "$c" "exit $RC ($ERR)"; return; fi
  off=$(noteoff "$D/bin"); ds=$(od -An -tu4 -j $((off + 4)) -N 4 "$D/bin" | tr -d ' ')
  id=$(readelf -n "$D/bin" | sed -n 's/.*Build ID: //p')
  cp "$D/bin" "$D/z"; dd if=/dev/zero of="$D/z" bs=1 seek=$((off + 16)) count="$ds" conv=notrunc status=none
  want=$(sha256sum < "$D/z" | cut -c1-$((ds * 2)))
  cp "$D/orig" "$D/plain"; strip --remove-section=.comment --remove-section=.note "$D/plain"
  diff=$({ cmp -l "$D/plain" "$D/bin" || true; } | awk -v lo=$((off + 17)) -v hi=$((off + 16 + ds)) '$1 < lo || $1 > hi' | wc -l)
  dbg=$D/debug/.build-id/${id:0:2}/${id:2}.debug
  [ -f "$dbg" ] && did=$(readelf -n "$dbg" 2>/dev/null | sed -n 's/.*Build ID: //p')
  run=$("$D/bin" 2>&1 | head -1)
  rerr=$(readelf -a -W "$D/bin" 2>&1 >/dev/null | head -2 | tr '\n' ' ')
  if [ -n "$id" ] && [ "$id" = "$want" ] && [ "$did" = "$id" ] && [ "$diff" = 0 ] && [ "$run" = "hello 1" ] \
    && [ -z "$rerr" ] && [ ! -e "$D/bin.debug.tmp" ]; then
    pass "$c: rewritten, build-id of $ds bytes = sha256 of the stripped file, same in its debug file"
  else
    fail "$c" "id=$id want=$want debug=$did bytes-off-descriptor=$diff run='$run' readelf='$rerr'"
  fi
}

# ---- nominal
run_sr a0-sha1 base; rewritten a0-sha1
run_sr a1-again base
if cmp -s "$T/a0-sha1/bin" "$T/a1-again/bin" && diff -r -q "$T/a0-sha1/debug" "$T/a1-again/debug" >/dev/null; then
  pass "a1-determinism: two runs, same binary and same debug file"
else fail "a1-determinism" "the two runs differ"; fi
mkdir -p "$T/sp ace"; cp base "$T/sp ace/bi n"
if bash "$SR" "$T/sp ace/dbg dir" "$T/sp ace/bi n" > /dev/null 2>&1 && [ -n "$(find "$T/sp ace/dbg dir" -name '*.debug')" ]; then
  pass "a2-spaces: paths with spaces"
else fail "a2-spaces" "paths with spaces"; fi
cc -g -O1 -o md5 hello.c -Wl,--build-id=md5; run_sr a3-md5 md5; rewritten a3-md5
cc -g -O1 -o uuid hello.c -Wl,--build-id=uuid; run_sr a3-uuid uuid; rewritten a3-uuid
mkdir -p a4; cp base a4/one; cp md5 a4/two
if bash "$SR" "$T/a4/debug" "$T/a4/one" "$T/a4/two" > a4/out 2>&1 && [ "$(wc -l < a4/out)" = 2 ] \
  && [ "$(find a4/debug -name '*.debug' | wc -l)" = 2 ]; then pass "a4-two-binaries: one line and one debug file each"
else fail "a4-two-binaries" "$(tr '\n' ' ' < a4/out)"; fi

# ---- no note, wrong note: refused before the input is touched
cc -g -O1 -o nonote hello.c -Wl,--build-id=none; run_sr b1-no-note nonote; refused b1-no-note
cc -g -O1 -o two hello.c -Wl,--build-id=0x0102; run_sr b2-descsz-2 two; refused b2-descsz-2
cc -g -O1 -o forty hello.c -Wl,--build-id=0x"$(printf 'ab%.0s' $(seq 40))"; run_sr b3-descsz-40 forty; refused b3-descsz-40
# SECURITY: a descriptor size that disagrees with the section (16 + 20 bytes) was rewritten in place.
for n in 32 24 8; do
  cp base "d$n"; u32 "d$n" $(( $(noteoff "d$n") + 4 )) "$n"; run_sr "b4-descsz-patched-to-$n" "d$n"; refused "b4-descsz-patched-to-$n"
done
cp base t1; u32 t1 $(( $(noteoff t1) + 8 )) 1; run_sr b5-type-1 t1; refused b5-type-1
cp base gnx; bytes gnx $(( $(noteoff gnx) + 12 )) GNX; run_sr b6-name-GNX gnx; refused b6-name-GNX
cp base ns8; u32 ns8 "$(noteoff ns8)" 8; run_sr b7-namesz-8 ns8; refused b7-namesz-8
# SECURITY: two notes in one .note.gnu.build-id section; only the first was rewritten.
cc -g -O1 -o nonote2 hello.c -Wl,--build-id=none
note() { printf '\x04\x00\x00\x00\x14\x00\x00\x00\x03\x00\x00\x00GNU\x00'; head -c 20 /dev/zero | tr '\0' "$1"; }
{ note A; note B; } > two.notes
objcopy --add-section .note.gnu.build-id=two.notes --set-section-flags .note.gnu.build-id=alloc,readonly nonote2 twonotes 2>/dev/null
if [ "$(readelf -n twonotes | grep -c 'Build ID')" = 2 ]; then run_sr b8-two-notes-one-section twonotes; refused b8-two-notes-one-section
else fail b8-two-notes-one-section "setup: no input with two notes"; fi
# SECURITY: two sections named .note.gnu.build-id (.note.gnu.build-iX added, renamed in .shstrtab).
objcopy -O binary --only-section=.note.gnu.build-id base base.note
objcopy --add-section .note.gnu.build-iX=base.note --set-section-flags .note.gnu.build-iX=alloc,readonly base twosec 2>/dev/null
perl -e 'local $/; open F, "+<", $ARGV[0] or die; binmode F; my $b = <F>; $b =~ s/\.note\.gnu\.build-iX\0/.note.gnu.build-id\0/ or die; seek F, 0, 0; print F $b; close F' twosec
if [ "$(readelf -S -W twosec | grep -c '\.note\.gnu\.build-id  *NOTE')" = 2 ]; then run_sr b9-two-sections twosec; refused b9-two-sections
else fail b9-two-sections "setup: no input with two sections"; fi

# ---- not an unstripped ELF file
printf 'not an ELF\n' > text; run_sr c1-text text; refused c1-text
: > empty; run_sr c2-empty empty; refused c2-empty
head -c 64 base > hdr; run_sr c3-elf-header-only hdr; refused c3-elf-header-only
head -c $(( $(stat -c %s base) / 2 )) base > half; run_sr c4-truncated-half half; refused c4-truncated-half
cp "$T/a0-sha1/bin" done; run_sr c5-already-processed done; refused c5-already-processed
cp base stripped; strip stripped; run_sr c6-stripped stripped; refused c6-stripped
mkdir -p notafile c7; RC=0; bash "$SR" "$T/c7/debug" "$T/notafile" > /dev/null 2> c7/err || RC=$?
if [ "$RC" != 0 ] && grep -q '^strip-release: .*not a file' c7/err; then pass "c7-directory: refused (exit $RC)"
else fail c7-directory "exit $RC: $(cat c7/err)"; fi
RC=0; bash "$SR" "$T/only-dir" > /dev/null 2>&1 || RC=$?
if [ "$RC" != 0 ]; then pass "c8-no-binary: refused (exit $RC)"; else fail c8-no-binary "exit 0"; fi

# ---- readelf as the script meets it: a pipe closed early, a warning after the rewrite
mkdir -p shim
cat > shim/readelf <<'SHIM'
#!/usr/bin/env bash
# SHIM_MODE=pipe: 4 MiB more after the section list, past a pipe buffer: a reader that stops at
# the first match (grep -q) closes the pipe early, and this readelf dies of SIGPIPE.
# SHIM_MODE=warn: readelf -a warns on stderr.
"$REAL_READELF" "$@" || exit
case "$SHIM_MODE: $* " in
  pipe:*" -S "*) head -c 4194304 /dev/zero | tr '\0' '\n' ;;
  warn:*" -a "*) echo "readelf: Warning: shim" >&2 ;;
esac
SHIM
chmod 0755 shim/readelf
run_sr d1-sigpipe base PATH="$T/shim:$PATH" REAL_READELF="$(command -v readelf)" SHIM_MODE=pipe; rewritten d1-sigpipe
if [ "$RC" = 0 ] && cmp -s "$T/a0-sha1/bin" "$D/bin"; then pass "d1-sigpipe: same bytes as without the shim"
else fail d1-sigpipe "other bytes than a0-sha1"; fi
run_sr d2-refused-after-objcopy base PATH="$T/shim:$PATH" REAL_READELF="$(command -v readelf)" SHIM_MODE=warn
refused d2-refused-after-objcopy touched

[ "$FAILS" = 0 ] || { echo "strip-release: $FAILS failed"; exit 1; }
echo "strip-release: all passed"
