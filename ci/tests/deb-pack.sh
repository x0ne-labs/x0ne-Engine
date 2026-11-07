#!/usr/bin/env bash
# Tests of ci/deb-pack.sh (the .deb of a GitHub release) on a copy of the ctl and the crate
# version, with three small ELF files: the package ci/deb-verify.sh accepts, the same bytes from
# another directory, umask and file dates, the dates and modes it sets, and each refusal. No
# container, no network. Run by ci/tests/ci-sh.sh, or by hand. Needs dpkg-deb; without it the
# test says so and skips, unless X0NE_REQUIRE_CI_TOOLS is set (ci/inside.sh does), then it fails.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)
if ! command -v dpkg-deb >/dev/null; then
  if [ -n "${X0NE_REQUIRE_CI_TOOLS:-}" ]; then echo "FAIL  deb-pack: dpkg-deb not found, X0NE_REQUIRE_CI_TOOLS is set"; exit 1; fi
  echo "skip  deb-pack: dpkg-deb not found"; exit 0
fi
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
FAILS=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s: %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }
umask 022
ELF=$(readlink -f "$(type -P true)")
ENGINE="x0ne-engine x0ne-engine-cloud x0ne-engine-proxy"
EPOCH=1793369424

# tree <dir>: what deb-pack.sh reads, copied (ctl, Cargo.toml) or small stand-ins (licences).
tree() {
  mkdir -p "$1/ci/deb"
  cp "$SRC/ci/deb-pack.sh" "$1/ci/"
  cp "$SRC/ci/deb/x0ne-engine.ctl" "$1/ci/deb/"
  cp "$SRC/Cargo.toml" "$1/"
  for f in LICENSE LICENSING.md NOTICE THIRD_PARTY_LICENSES.html; do printf 'stand-in %s\n' "$f" > "$1/$f"; done
}
# bins <dir>: the three binaries and x0ne-engine.version, as the release workflow stages them.
bins() {
  local b
  mkdir -p "$1"
  for b in $ENGINE; do { cat "$ELF"; printf '%s' "$b"; } > "$1/$b"; chmod 0755 "$1/$b"; done
  printf 'x0ne-Engine test, commit 0000000\n' > "$1/x0ne-engine.version"
}
# pack <tree> <bin dir> <deb> [epoch]: exit status in $rc, stderr in $T/err.
pack() {
  rc=0
  SOURCE_DATE_EPOCH=${4:-$EPOCH} bash "$1/ci/deb-pack.sh" "$2" "$3" > "$T/out" 2> "$T/err" || rc=$?
}

tree "$T/a"; bins "$T/a/bin"
crate=$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$T/a/Cargo.toml" | head -n 1)
debver=$(printf '%s' "$crate" | sed -E 's/-(alpha|beta|rc)/~\1/')

pack "$T/a" "$T/a/bin" "$T/a/out/x0ne-engine_amd64.deb"
D=$T/a/out/x0ne-engine_amd64.deb
if [ "$rc" != 0 ] || [ ! -s "$D" ]; then
  fail build "exit $rc: $(cat "$T/err")"
  echo "$FAILS failure(s)"; exit 1
fi
pass "build: $(tail -n 1 "$T/out" | cut -c1-80)"

# The package ci/deb-verify.sh accepts: exactly the three binaries of SHA256SUMS.engine.
mkdir -p "$T/v/debs" "$T/v/bin"; cp "$D" "$T/v/debs/"
# shellcheck disable=SC2086 # three names, split on purpose
(cd "$T/a/bin" && sha256sum $ENGINE > "$T/v/bin/SHA256SUMS.engine")
if bash "$SRC/ci/deb-verify.sh" "$T/v" > "$T/verify" 2>&1; then pass "deb-verify accepts it: $(grep -c '^same ' "$T/verify") binaries"
else fail deb-verify "$(cat "$T/verify")"; fi

# Control fields: the ctl's, its version, no source-only field.
ctl=$T/ctl; dpkg-deb -f "$D" > "$ctl"
want_dep=$(sed -n 's/^Depends: //p' "$SRC/ci/deb/x0ne-engine.ctl")
if [ "$(dpkg-deb -f "$D" Package)" = x0ne-engine ] && [ "$(dpkg-deb -f "$D" Version)" = "$debver" ] \
  && [ "$(dpkg-deb -f "$D" Architecture)" = amd64 ] && [ "$(dpkg-deb -f "$D" Depends)" = "$want_dep" ] \
  && grep -q '^Installed-Size: [0-9]' "$ctl" && ! grep -qE '^(Files|Standards-Version):' "$ctl"; then
  pass "control: x0ne-engine $debver amd64, Depends of the ctl, no Files or Standards-Version"
else fail control "$(cat "$ctl")"; fi

# Contents: the Files: list, the licence files, nothing else; modes, owner and dates.
dpkg-deb -c "$D" > "$T/list"
got=$(awk '$1 !~ /^d/ { print $1, $2, $6 }' "$T/list" | sort)
want=$(printf '%s\n' \
  "-rwxr-xr-x root/root ./usr/bin/x0ne-engine" \
  "-rwxr-xr-x root/root ./usr/bin/x0ne-engine-cloud" \
  "-rwxr-xr-x root/root ./usr/bin/x0ne-engine-proxy" \
  "-rw-r--r-- root/root ./usr/share/doc/x0ne-engine/LICENSING.md" \
  "-rw-r--r-- root/root ./usr/share/doc/x0ne-engine/NOTICE" \
  "-rw-r--r-- root/root ./usr/share/doc/x0ne-engine/THIRD_PARTY_LICENSES.html" \
  "-rw-r--r-- root/root ./usr/share/doc/x0ne-engine/copyright" \
  "-rw-r--r-- root/root ./usr/share/x0ne-engine/x0ne-engine.version" | sort)
if [ "$got" = "$want" ]; then pass "contents: 3 binaries 0755, version file and licences 0644, root/root"
else fail contents "$(diff <(printf '%s\n' "$want") <(printf '%s\n' "$got") || true)"; fi
stamp=$(TZ=UTC date -d "@$EPOCH" '+%Y-%m-%d %H:%M')
if [ "$(TZ=UTC dpkg-deb -c "$D" | awk '{ print $4, $5 }' | sort -u)" = "$stamp" ]; then pass "every entry dated SOURCE_DATE_EPOCH ($stamp UTC)"
else fail dates "$(TZ=UTC dpkg-deb -c "$D" | awk '{ print $4, $5, $6 }' | head -n 5)"; fi

# The same bytes from another tree, another umask and other file dates.
tree "$T/b"; bins "$T/b/elsewhere"
touch -d '2001-01-01' "$T/b/elsewhere"/* "$T/b/LICENSE"
(umask 077; pack "$T/b" "$T/b/elsewhere" "$T/b/x.deb"; [ "$rc" = 0 ] || exit 1) || fail repro "second build failed: $(cat "$T/err")"
if cmp -s "$D" "$T/b/x.deb"; then pass "reproducible: same bytes from another directory, umask 077, other dates"
else fail repro "the two builds differ"; fi
pack "$T/a" "$T/a/bin" "$T/a/later.deb" $((EPOCH + 86400))
if [ "$rc" = 0 ] && ! cmp -s "$D" "$T/a/later.deb"; then pass "another SOURCE_DATE_EPOCH, other bytes"
else fail epoch "exit $rc, or same bytes"; fi

# refuse <name> <message> <command...>: deb-pack.sh must exit 2, name the cause, write no .deb.
refuse() {
  local name=$1 msg=$2; shift 2
  rm -f "$T/r.deb"; rc=0
  "$@" > /dev/null 2> "$T/err" || rc=$?
  if [ "$rc" = 2 ] && grep -qF -- "$msg" "$T/err" && [ ! -e "$T/r.deb" ]; then pass "$name: refused, $(head -n 1 "$T/err" | cut -c1-90)"
  else fail "$name" "exit $rc, no \"$msg\" or a .deb was written: $(cat "$T/err")"; fi
}
tree "$T/c"; bins "$T/c/bin"
sed -i 's/^Version: .*/Version: 9.9.9/' "$T/c/ci/deb/x0ne-engine.ctl"
refuse ctl-version "says Version: 9.9.9" env SOURCE_DATE_EPOCH=$EPOCH bash "$T/c/ci/deb-pack.sh" "$T/c/bin" "$T/r.deb"
tree "$T/d"; bins "$T/d/bin"; rm "$T/d/bin/x0ne-engine-proxy"
refuse missing-binary "missing $T/d/bin/x0ne-engine-proxy" env SOURCE_DATE_EPOCH=$EPOCH bash "$T/d/ci/deb-pack.sh" "$T/d/bin" "$T/r.deb"
refuse no-epoch "SOURCE_DATE_EPOCH is not set" env -u SOURCE_DATE_EPOCH bash "$T/a/ci/deb-pack.sh" "$T/a/bin" "$T/r.deb"
refuse not-a-deb "must end in .deb" env SOURCE_DATE_EPOCH=$EPOCH bash "$T/a/ci/deb-pack.sh" "$T/a/bin" "$T/r.tar"
tree "$T/e"; bins "$T/e/bin"
sed -i 's#^ /out/bin/x0ne-engine-proxy /usr/bin/# /tmp/x0ne-engine-proxy /usr/bin/#' "$T/e/ci/deb/x0ne-engine.ctl"
refuse files-outside "Files: /tmp/x0ne-engine-proxy is not under /out/bin/" env SOURCE_DATE_EPOCH=$EPOCH bash "$T/e/ci/deb-pack.sh" "$T/e/bin" "$T/r.deb"

[ "$FAILS" -eq 0 ] || { echo "$FAILS failure(s)"; exit 1; }
echo "deb-pack: all passed"
