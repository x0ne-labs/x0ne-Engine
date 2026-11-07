#!/usr/bin/env bash
# The x0ne-engine .deb of a GitHub release (.github/workflows/release.yml), from the
# stripped binaries the release archive ships: the fields and the Files: list of
# ci/deb/x0ne-engine.ctl, packed by dpkg-deb alone (no equivs, no x0ne OS checkout,
# which make deb needs), every date at SOURCE_DATE_EPOCH: two runs give the same bytes.
#
#   SOURCE_DATE_EPOCH=<commit time> ci/deb-pack.sh <bin dir> <output .deb>
#
# <bin dir> holds what the Files: lines name under /out/bin/ (the three binaries and
# x0ne-engine.version). Version: the ctl's, refused unless it is the crate version in
# Debian form (0.2.0-beta.7 -> 0.2.0~beta.7). /usr/share/doc/x0ne-engine gets the licence
# files the archive ships (copyright = LICENSE). Tested by ci/tests/deb-pack.sh.
set -euo pipefail
die() { echo "deb-pack: $*" >&2; exit 2; }
[ $# = 2 ] || die "usage: SOURCE_DATE_EPOCH=<t> ci/deb-pack.sh <bin dir> <output .deb>"
[ -n "${SOURCE_DATE_EPOCH:-}" ] || die "SOURCE_DATE_EPOCH is not set"
command -v dpkg-deb >/dev/null || die "dpkg-deb is required"
umask 022
SRC=$(cd "$(dirname "$0")/.." && pwd)
CTL=$SRC/ci/deb/x0ne-engine.ctl
bin=$(cd "$1" && pwd) || die "no directory $1"
deb=$2
case $deb in *.deb) ;; *) die "the output must end in .deb: $deb" ;; esac

crate=$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$SRC/Cargo.toml" | head -n 1)
[ -n "$crate" ] || die "no workspace version in Cargo.toml"
want=$(printf '%s' "$crate" | sed -E 's/-(alpha|beta|rc)/~\1/')
ver=$(sed -n 's/^Version: //p' "$CTL")
[ "$ver" = "$want" ] || die "$CTL says Version: $ver, the crate version $crate gives $want"
[ "$(sed -n 's/^Package: //p' "$CTL")" = x0ne-engine ] || die "$CTL is not the x0ne-engine package"

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
pkg=$T/pkg
mkdir -p "$pkg/DEBIAN"

# Files: "<source> <destination dir>/" pairs; the first sits on the Files: line itself.
n=0
while read -r s d; do
  case $s in /out/bin/?*) ;; *) die "Files: $s is not under /out/bin/" ;; esac
  case $d in /*/) ;; *) die "Files: $s goes to $d, not a directory" ;; esac
  f=$bin/${s#/out/bin/}
  if [ ! -f "$f" ] || [ -L "$f" ]; then die "missing $f (Files: $s)"; fi
  mode=0644
  [ "$d" = /usr/bin/ ] && mode=0755
  install -D -m "$mode" "$f" "$pkg$d${s##*/}"
  n=$((n + 1))
done < <(sed -n '/^Files:/,/^[^ ]/p' "$CTL" | sed 's/^Files://' | awk '/^ *\// { print $1, $2 }')
[ "$n" -gt 0 ] || die "no Files: entry in $CTL"

doc=$pkg/usr/share/doc/x0ne-engine
install -D -m 0644 "$SRC/LICENSE" "$doc/copyright"
for f in LICENSING.md NOTICE THIRD_PARTY_LICENSES.html; do install -m 0644 "$SRC/$f" "$doc/$f"; done

# Installed-Size as dpkg-gencontrol counts it: KiB per file rounded up, plus one per directory.
size=$(find "$pkg" -path "$pkg/DEBIAN" -prune -o -type f -printf '%s\n' | awk '{ s += int(($1 + 1023) / 1024) } END { print s + 0 }')
dirs=$(find "$pkg" -path "$pkg/DEBIAN" -prune -o -type d -print | wc -l)
size=$((size + dirs - 1))

# control: the binary fields of the ctl, in this order; Files: and Standards-Version stay out.
awk -v size="$size" '
  /^#/ { next }
  /^[^ ]/ { cur = substr($0, 1, index($0, ":") - 1) }
  { val[cur] = val[cur] $0 "\n" }
  END {
    n = split("Package Version Architecture Maintainer Installed-Size Depends Suggests Section Priority Description", w, " ")
    for (i = 1; i <= n; i++) {
      if (w[i] == "Installed-Size") printf "Installed-Size: %s\n", size
      else if (w[i] in val) printf "%s", val[w[i]]
    }
  }' "$CTL" > "$pkg/DEBIAN/control"
grep -q '^Description: ' "$pkg/DEBIAN/control" || die "no Description in $CTL"
(cd "$pkg" && find . -path ./DEBIAN -prune -o -type f -printf '%P\n' | LC_ALL=C sort \
  | while IFS= read -r f; do md5sum "$f"; done) > "$pkg/DEBIAN/md5sums"
chmod 0644 "$pkg/DEBIAN/control" "$pkg/DEBIAN/md5sums"
find "$pkg" -type d -exec chmod 0755 {} +
find "$pkg" -exec touch -h -d "@$SOURCE_DATE_EPOCH" {} +

mkdir -p "$(dirname "$deb")"
# One xz thread: a threaded xz stream differs from a single-threaded one.
dpkg-deb --root-owner-group -Zxz --threads-max=1 --build "$pkg" "$deb" >/dev/null
echo "deb-pack: $deb: x0ne-engine $ver, $n files from $bin, $(sha256sum < "$deb" | cut -c1-64)"
