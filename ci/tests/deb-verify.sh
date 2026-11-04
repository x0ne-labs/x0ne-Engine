#!/usr/bin/env bash
# Tests of ci/deb-verify.sh on packages built here (dpkg-deb): the .deb of make deb (the
# client x0ne-engine; the server x0ne-infer ships in the CVM image only), then one change
# each that must be red. No container, no network. Run by ci/tests/ci-sh.sh, or by
# hand. Needs dpkg-deb; without it the test says so and skips, unless X0NE_REQUIRE_CI_TOOLS is
# set (ci/inside.sh does: the CI image ships it), then it fails.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)
DV=$SRC/ci/deb-verify.sh
if ! command -v dpkg-deb >/dev/null; then
  if [ -n "${X0NE_REQUIRE_CI_TOOLS:-}" ]; then echo "FAIL  deb-verify: dpkg-deb not found, X0NE_REQUIRE_CI_TOOLS is set"; exit 1; fi
  echo "skip  deb-verify: dpkg-deb not found"; exit 0
fi
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
FAILS=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s: %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }
umask 022
ELF=$(readlink -f "$(type -P true)")
ENGINE="x0ne-engine x0ne-engine-cloud x0ne-engine-proxy"

# fixture <case>: bin/ (three ELF files and SHA256SUMS.engine, as ci/inside-deb.sh writes them)
# and root/x0ne-engine/.
fixture() {
  local d=$T/$1 b
  mkdir -p "$d/bin" "$d/debs"
  for b in $ENGINE; do { cat "$ELF"; printf '%s' "$b"; } > "$d/bin/$b"; chmod 0755 "$d/bin/$b"; done
  (cd "$d/bin" && sha256sum $ENGINE > SHA256SUMS.engine)
  pkgroot "$d" x0ne-engine $ENGINE
  R=$d/root
}
# pkgroot <dir> <package> <binary>...: a control file, a copyright file, the binaries in usr/bin.
pkgroot() {
  local d=$1 p=$2 r b; shift 2; r=$d/root/$p
  mkdir -p "$r/DEBIAN" "$r/usr/bin" "$r/usr/share/doc/$p"
  printf 'Package: %s\nVersion: 0\nArchitecture: amd64\nMaintainer: t <t@t>\nDescription: t\n' "$p" > "$r/DEBIAN/control"
  echo copyright > "$r/usr/share/doc/$p/copyright"
  for b in "$@"; do cp "$d/bin/$b" "$r/usr/bin/$b"; done
}
# pack <case>: one .deb per root/<package>, then ci/deb-verify.sh; RC, OUT.
pack() {
  local d=$T/$1 r
  for r in "$d"/root/*; do [ -d "$r" ] || continue; dpkg-deb --root-owner-group -Zgzip -b "$r" "$d/debs/${r##*/}_0_amd64.deb" > /dev/null; done
  RC=0; OUT=$(bash "$DV" "$d" 2>&1) || RC=$?
}
# red <case> <message>: exit 1, and the message says why.
red() {
  if [ "$RC" = 1 ] && grep -qF -- "$2" <<<"$OUT"; then pass "$1: red, $2"
  else fail "$1" "exit $RC, $(head -n 3 <<<"$OUT" | tr '\n' ' ')"; fi
}

fixture v0-control; pack v0-control
if [ "$RC" = 0 ] && [ "$(grep -c '^same ' <<<"$OUT")" = 3 ]; then pass "v0-control: green, 3 binaries"
else fail v0-control "exit $RC, $(tr '\n' ' ' <<<"$OUT")"; fi

# SECURITY: a binary SHA256SUMS.engine does not name, or one it names missing.
fixture v1-extra-usr-bin; cp "$ELF" "$R/x0ne-engine/usr/bin/x0ne-extra"; pack v1-extra-usr-bin; red v1-extra-usr-bin "NOT THE BINARIES OF SHA256SUMS.engine: x0ne-engine"
fixture v2-missing-binary; rm "$R/x0ne-engine/usr/bin/x0ne-engine-proxy"; pack v2-missing-binary; red v2-missing-binary "NOT THE BINARIES OF SHA256SUMS.engine: x0ne-engine"
fixture v2c-package-twice; cp -r "$R/x0ne-engine" "$R/x0ne-engine-again"; pack v2c-package-twice; red v2c-package-twice "PACKAGES:"
fixture v2d-other-package; cp -r "$R/x0ne-engine" "$R/x0ne-other"; sed -i 's/^Package: .*/Package: x0ne-other/' "$R/x0ne-other/DEBIAN/control"
pack v2d-other-package; red v2d-other-package "UNEXPECTED PACKAGE x0ne-other"
# SECURITY: other bytes under a name SHA256SUMS.engine gives.
fixture v3-modified-byte; printf '\x90' | dd of="$R/x0ne-engine/usr/bin/x0ne-engine" bs=1 seek=100 conv=notrunc status=none
pack v3-modified-byte; red v3-modified-byte "NOT THE BINARIES OF SHA256SUMS.engine: x0ne-engine"
fixture v4-name-as-regex; mv "$R/x0ne-engine/usr/bin/x0ne-engine" "$R/x0ne-engine/usr/bin/x0ne.engine"; pack v4-name-as-regex; red v4-name-as-regex "NOT THE BINARIES OF SHA256SUMS.engine: x0ne-engine"
fixture v9-swapped-content; cp "$R/x0ne-engine/usr/bin/x0ne-engine" "$R/x0ne-engine/usr/bin/x0ne-engine-cloud"; pack v9-swapped-content; red v9-swapped-content "NOT THE BINARIES OF SHA256SUMS.engine: x0ne-engine"
# The TD's server in a .deb, even listed in SHA256SUMS.engine: the CVM image ships it (2026-11-04).
fixture v13-server-in-client; cp "$ELF" "$R/x0ne-engine/usr/bin/x0ne-infer"
(cd "$T/v13-server-in-client/bin" && cp "$ELF" x0ne-infer && sha256sum x0ne-infer >> SHA256SUMS.engine); pack v13-server-in-client
red v13-server-in-client "THE SERVER IN A .deb: x0ne-engine ships /usr/bin/x0ne-infer"
fixture v13c-server-package; pkgroot "$T/v13c-server-package" x0ne-infer; cp "$ELF" "$R/x0ne-infer/usr/bin/x0ne-infer"
pack v13c-server-package; red v13c-server-package "UNEXPECTED PACKAGE x0ne-infer"
# SECURITY: what usr/bin/* did not see: a dotfile, a subdirectory, a link, files elsewhere.
fixture v7-dotfile; cp "$ELF" "$R/x0ne-engine/usr/bin/.x0ne-hidden"; pack v7-dotfile; red v7-dotfile "NOT THE BINARIES OF SHA256SUMS.engine: x0ne-engine"
fixture v7b-subdirectory; mkdir "$R/x0ne-engine/usr/bin/sub"; mv "$R/x0ne-engine/usr/bin/x0ne-engine" "$R/x0ne-engine/usr/bin/sub/"
pack v7b-subdirectory; red v7b-subdirectory "IN A SUBDIRECTORY OF usr/bin: x0ne-engine /usr/bin/sub/x0ne-engine"
fixture v8-dangling-symlink; ln -s /opt/x0ne/evil "$R/x0ne-engine/usr/bin/x0ne-alias"; pack v8-dangling-symlink; red v8-dangling-symlink "NOT A REGULAR FILE: x0ne-engine /usr/bin/x0ne-alias"
fixture v6-elf-elsewhere; mkdir -p "$R/x0ne-engine/usr/libexec/x0ne"; cp "$ELF" "$R/x0ne-engine/usr/libexec/x0ne/helper"; chmod 0644 "$R/x0ne-engine/usr/libexec/x0ne/helper"
pack v6-elf-elsewhere; red v6-elf-elsewhere "OUTSIDE usr/bin: x0ne-engine /usr/libexec/x0ne/helper"
fixture v6b-script-elsewhere; printf '#!/bin/sh\n' > "$R/x0ne-engine/usr/share/doc/x0ne-engine/run"; chmod 0755 "$R/x0ne-engine/usr/share/doc/x0ne-engine/run"
pack v6b-script-elsewhere; red v6b-script-elsewhere "OUTSIDE usr/bin: x0ne-engine /usr/share/doc/x0ne-engine/run"
fixture v6c-link-elsewhere; mkdir -p "$R/x0ne-engine/usr/lib/x0ne"; ln -s /usr/bin/x0ne-engine "$R/x0ne-engine/usr/lib/x0ne/alias"
pack v6c-link-elsewhere; red v6c-link-elsewhere "OUTSIDE usr/bin: x0ne-engine /usr/lib/x0ne/alias"
# The lists themselves: nothing listed for a package.
fixture v10-empty-engine-list; : > "$T/v10-empty-engine-list/bin/SHA256SUMS.engine"; rm "$R/x0ne-engine/usr/bin/"*
pack v10-empty-engine-list; red v10-empty-engine-list "NO BINARY LISTED FOR x0ne-engine"
fixture v12-no-deb; rm -r "$R"; mkdir "$R"; pack v12-no-deb; red v12-no-deb "NO .deb in"

[ "$FAILS" = 0 ] || { echo "deb-verify: $FAILS failed"; exit 1; }
echo "deb-verify: all passed"
