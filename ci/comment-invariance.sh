#!/usr/bin/env bash
# A comment changes no delivered byte (make ci, step comment-invariance). Builds the three
# binaries of the .deb package from a copy of /src with ci/inside-deb.sh (profile release,
# then ci/strip-release.sh), appends one comment line at the end of three source files,
# builds again, and requires the same sha256 for each binary.
# Appended at the end, the comment moves no line: no panic location, no code changes. A
# difference means the packaging lets in bytes the code does not decide (measured
# 2026-10-25: the build-id of a strip done after the link, the CRC of a .gnu_debuglink).
# Inside the CI image (ci/inside.sh): /src read-only, /vendor, /target (profile release in
# /target/release). Work tree: COMMENT_DIR (/tmp/comment-invariance). Exit 0 same, 1 different,
# 2 setup.
set -euo pipefail
# make ci gives its test builds debug info (ci/ci.sh: CARGO_PROFILE_DEV_DEBUG), make deb gives no
# CARGO_PROFILE_* and ci/release-env.sh refuses them all: the builds below run as make deb runs them.
unset "${!CARGO_PROFILE_DEV_@}"
W=${COMMENT_DIR:-/tmp/comment-invariance}
FILES="x0ne-infer/src/server.rs x0ne-engine-cloud/src/main.rs x0ne-confidential/src/lib.rs"
BINS="x0ne-engine x0ne-engine-cloud x0ne-engine-proxy"
export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-0} X0NE_GIT_SHA=${X0NE_GIT_SHA:-comment-invariance} DEB_VERSION=${DEB_VERSION:-0}
rm -rf "$W"; mkdir -p "$W"
cp -r /src "$W/src"
for f in $FILES; do [ -f "$W/src/$f" ] || { echo "comment-invariance: no $f" >&2; exit 2; }; done

# build <a|b>: the deb script of the copy, outputs in $W/<a|b>/{bin,debug}.
build() {
  local t0=$SECONDS
  X0NE_DEB_SRC=$W/src X0NE_DEB_OUT=$W/$1 bash "$W/src/ci/inside-deb.sh" > "$W/$1.log" 2>&1 \
    || { tail -n 30 "$W/$1.log"; echo "comment-invariance: build $1 failed" >&2; exit 2; }
  echo "build $1: $((SECONDS - t0)) s"
}
build a
for f in $FILES; do printf '\n// Comment-invariance probe (ci/comment-invariance.sh): no line moves.\n' >> "$W/src/$f"; done
build b

# sections <file>: one line per section with file contents, name and sha256 of its bytes
# (diagnosis only; read in place, so the two sides can run at once).
sections() {
  local n t off size
  readelf -S -W "$1" | sed -n 's/^ *\[ *[0-9]*\] \(\.[^ ]*\)  *\([A-Z_0-9]*\)  *[0-9a-f]*  *\([0-9a-f]*\)  *\([0-9a-f]*\) .*/\1 \2 \3 \4/p' |
    while read -r n t off size; do
      [ "$t" = NOBITS ] && continue
      printf '%s %s\n' "$n" "$(dd if="$1" bs=64K iflag=skip_bytes,count_bytes skip=$((16#$off)) count=$((16#$size)) status=none | sha256sum | cut -c1-16)"
    done
}
rc=0
for b in $BINS; do
  ha=$(sha256sum < "$W/a/bin/$b" | cut -c1-64) hb=$(sha256sum < "$W/b/bin/$b" | cut -c1-64)
  if [ "$ha" = "$hb" ]; then
    printf 'same    %s  %s  (%s bytes)\n' "$ha" "$b" "$(stat -c %s "$W/a/bin/$b")"
  else
    rc=1
    printf 'DIFFER  %s %s  %s\n' "$ha" "$hb" "$b"
    printf '        build-id %s / %s\n' "$(readelf -n "$W/a/bin/$b" | sed -n 's/.*Build ID: //p')" "$(readelf -n "$W/b/bin/$b" | sed -n 's/.*Build ID: //p')"
    diff <(sections "$W/a/bin/$b") <(sections "$W/b/bin/$b") | sed -n 's/^> /        section differs: /p' || true
  fi
done
if [ "$rc" = 0 ]; then echo "comment-invariance: the $(wc -w <<<"$BINS") delivered binaries are identical after a comment"
else echo "comment-invariance: a comment changed a delivered binary (see ci/strip-release.sh)"; fi
exit "$rc"
