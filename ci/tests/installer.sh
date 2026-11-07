#!/bin/sh
# Tests of install.sh against fake releases served on 127.0.0.1: HTTPS with a certificate
# made here, plain HTTP, and an HTTPS redirect to HTTP. Each case runs the installer with
# its own HOME and install directory, and a PATH without the real gh. Cases: install (the
# default version, --version, X0NE_ENGINE_VERSION), the two requests and nothing else,
# checksum mismatch, SHA256SUMS missing, archive not listed, unsupported OS and CPU (a fake
# uname), a binary that does not run (the old install is kept), http and redirects to http,
# no curl, provenance (a fake gh: absent, too old, logged out, verifies, fails), uninstall.
# Without the glibc loader (Alpine) only the refusal is tested.
# Adapted from x0ne-Harness scripts/test-install-https-only.sh and test-install-provenance.sh.
#
#   sh ci/tests/installer.sh            (ci/install-matrix.sh runs it in four distributions)
#
# ENGINE_ARCHIVE=<x0ne-engine-x86_64-unknown-linux-gnu.tar.gz> serves that archive (real
# binaries) as the good release. INSTALL_SH overrides the script under test. Needs curl,
# openssl, python3, tar, gzip. Writes only under a temporary directory.
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
installer=${INSTALL_SH:-$root/install.sh}
tmp=$(mktemp -d)
# shellcheck disable=SC2317,SC2329 # reached through the EXIT trap
cleanup() {
    if [ -f "$tmp/pids" ]; then
        while read -r p; do kill "$p" 2>/dev/null || true; done <"$tmp/pids"
    fi
    rm -rf "$tmp"
}
trap cleanup EXIT
fail=0
n=0
ok() {
    n=$((n + 1))
    echo "ok    $*"
}
ko() {
    echo "FAIL  $*"
    fail=1
}

for tool in curl openssl python3 tar gzip; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "FAIL  $tool is not installed"
        exit 1
    }
done
version=$(sed -n 's/^VERSION="\(.*\)"$/\1/p' "$installer")
[ -n "$version" ] || {
    echo "FAIL  no VERSION= line in $installer"
    exit 1
}
target=x86_64-unknown-linux-gnu
asset="x0ne-engine-$target.tar.gz"
bins="x0ne-engine x0ne-engine-cloud x0ne-engine-proxy"

# The release workflow makes the assets these cases fake, under these names.
wf="$root/.github/workflows/release.yml"
if [ -f "$wf" ]; then
    # shellcheck disable=SC2016 # literal workflow text
    for want in "TARGET: $target" 'name="x0ne-engine-${TARGET}"' 'tar --sort=name' \
        'cp out/x0ne-engine out/x0ne-engine-cloud out/x0ne-engine-proxy' \
        'dist/x0ne-engine_amd64.deb' 'cp install.sh dist/install.sh' 'SHA256SUMS' \
        'actions/attest-build-provenance@'; do
        grep -qF -e "$want" "$wf" || ko "release.yml: no \"$want\""
    done
    [ "$fail" = 1 ] || ok "release.yml makes $asset, x0ne-engine_amd64.deb, install.sh, SHA256SUMS, attested"
fi

# ── Fake releases ────────────────────────────────────────────────────────────
# archive <file> <marker> [<x0ne-engine body>] [flat|noproxy]: three scripts that print
# "<name> <marker>", in x0ne-engine-<target>/ as the release workflow packs them.
archive() {
    d="$tmp/pack/x0ne-engine-$target"
    rm -rf "$tmp/pack"
    mkdir -p "$d"
    for b in $bins; do
        printf '#!/bin/sh\necho "%s %s"\n' "$b" "$2" >"$d/$b"
        chmod 0755 "$d/$b"
    done
    [ -z "${3:-}" ] || printf '#!/bin/sh\n%s\n' "$3" >"$d/x0ne-engine"
    case "${4:-}" in
        flat) tar -czf "$1" -C "$d" . ;;
        noproxy) rm "$d/x0ne-engine-proxy" && tar -czf "$1" -C "$tmp/pack" "x0ne-engine-$target" ;;
        *) tar -czf "$1" -C "$tmp/pack" "x0ne-engine-$target" ;;
    esac
}
# sha <file>: its SHA-256 (python3: Fedora's base image has no cmp, and the
# installer's own sha256sum is what is under test).
sha() {
    python3 -I -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"
}
# sums <dir> <file>...: SHA256SUMS of these files, in that directory.
sums() {
    dir=$1
    shift
    (cd "$dir" && for f in "$@"; do printf '%s  %s\n' "$(sha "$f")" "$f"; done) >"$dir/SHA256SUMS"
}
# release <site> [<tag>]: the directory of that release, GitHub's layout.
release() {
    r="$tmp/www/$1/download/${2:-v$version}"
    mkdir -p "$r"
    echo "$r"
}

r=$(release good)
if [ -n "${ENGINE_ARCHIVE:-}" ]; then
    cp "$ENGINE_ARCHIVE" "$r/$asset"
    real=1
else
    archive "$r/$asset" GOOD
    real=0
fi
printf 'other\n' >"$r/x0ne-engine-other.txt"
sums "$r" "$asset" x0ne-engine-other.txt
r2=$(release good v0.0.1-test)
archive "$r2/$asset" OLDER
sums "$r2" "$asset"
# bad: the good SHA256SUMS, the archive with one byte changed.
r=$(release bad)
cp "$tmp/www/good/download/v$version/$asset" "$tmp/www/good/download/v$version/SHA256SUMS" "$r/"
python3 -I -c '
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
b[len(b) // 2] ^= 1
open(p, "wb").write(b)' "$r/$asset"
r=$(release nosums)
archive "$r/$asset" NOSUMS
r=$(release unlisted)
archive "$r/$asset" UNLISTED
cp "$r/$asset" "$r/renamed.tar.gz"
sums "$r" renamed.tar.gz
rm "$r/renamed.tar.gz"
r=$(release broken)
archive "$r/$asset" BROKEN "echo \"x0ne-engine: version 'GLIBC_2.99' not found\" >&2; exit 1"
sums "$r" "$asset"
r=$(release flat)
archive "$r/$asset" FLAT "" flat
sums "$r" "$asset"
r=$(release noproxy)
archive "$r/$asset" NOPROXY "" noproxy
sums "$r" "$asset"

# ── Servers: HTTPS (files, and /redirect/ to HTTP), plain HTTP ───────────────
openssl req -x509 -newkey rsa:2048 -nodes -days 1 -subj /CN=127.0.0.1 \
    -addext subjectAltName=IP:127.0.0.1 \
    -keyout "$tmp/key.pem" -out "$tmp/cert.pem" >/dev/null 2>&1
python3 -I -c '
import functools, http.server, os, ssl, sys, threading
www, cert, key, portfile, log = sys.argv[1:6]
class Files(http.server.SimpleHTTPRequestHandler):
    scheme = "https"
    def __init__(self, *a, **k):
        super().__init__(*a, directory=www, **k)
    def log_message(self, *a):
        pass
    def do_GET(self):
        with open(log, "a") as f:
            f.write(self.scheme + " " + self.path + "\n")
        if self.path.startswith("/redirect/"):
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:%d/%s" % (plain.server_address[1], self.path[len("/redirect/"):]))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_GET()
class Plain(Files):
    scheme = "http"
tls = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Files)
c = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
c.load_cert_chain(cert, key)
tls.socket = c.wrap_socket(tls.socket, server_side=True)
plain = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Plain)
threading.Thread(target=plain.serve_forever, daemon=True).start()
open(portfile + ".tmp", "w").write("%d %d\n" % (tls.server_address[1], plain.server_address[1]))
os.replace(portfile + ".tmp", portfile)
tls.serve_forever()' "$tmp/www" "$tmp/cert.pem" "$tmp/key.pem" "$tmp/ports" "$tmp/requests" >"$tmp/server.log" 2>&1 &
echo "$!" >>"$tmp/pids"
i=0
while [ ! -s "$tmp/ports" ] && [ "$i" -lt 100 ]; do
    sleep 0.1
    i=$((i + 1))
done
[ -s "$tmp/ports" ] || {
    echo "FAIL  the test server did not start: $(cat "$tmp/server.log")"
    exit 1
}
read -r https_port http_port <"$tmp/ports"
url="https://127.0.0.1:${https_port}"
http_url="http://127.0.0.1:${http_port}"
# Controls: each server answers, so a refusal comes from the installer.
curl -fsS --cacert "$tmp/cert.pem" "$url/good/download/v$version/SHA256SUMS" >/dev/null
curl -fsS "$http_url/good/download/v$version/SHA256SUMS" >/dev/null
curl -fsSL --cacert "$tmp/cert.pem" "$url/redirect/good/download/v$version/SHA256SUMS" >/dev/null

# ── PATHs: without gh (and without curl), a fake gh, a fake uname ────────────
# strip <dir> <name>...: <dir> links every program of PATH but these.
strip() {
    out=$1
    shift
    mkdir "$out"
    IFS=:
    for d in $PATH; do
        for f in "$d"/*; do
            [ -e "$f" ] || continue
            p=${f##*/}
            skip=0
            for s in "$@"; do [ "$p" = "$s" ] && skip=1; done
            [ "$skip" = 1 ] || [ -e "$out/$p" ] || ln -s "$f" "$out/$p"
        done
    done
    unset IFS
}
strip "$tmp/nogh" gh
strip "$tmp/nocurl" gh curl

# The fake gh answers as gh FAKE_GH_VERSION would: FAKE_GH_ATTESTATION is none (before
# 2.49), nosigner (2.49, 2.50) or full; FAKE_GH_AUTH is the exit status of
# `gh auth status`, FAKE_GH_VERIFY that of a verification.
mkdir "$tmp/gh"
cat >"$tmp/gh/gh" <<'GH'
#!/bin/sh
# TEST fake gh: nothing reaches GitHub.
echo "gh $*" >>"$FAKE_GH_LOG"
case "$*" in
    --version) echo "gh version $FAKE_GH_VERSION (fake)" ;;
    "auth status") exit "$FAKE_GH_AUTH" ;;
    attestation*)
        if [ "$FAKE_GH_ATTESTATION" = none ]; then
            echo 'unknown command "attestation" for "gh"' >&2
            exit 1
        fi
        case "$*" in
            "attestation verify --help")
                echo "      --repo string              Repository name in the format <owner>/<repo>"
                [ "$FAKE_GH_ATTESTATION" = nosigner ] ||
                    echo "      --signer-workflow string   Workflow that signed attestation"
                ;;
            *" --signer-workflow "*)
                if [ "$FAKE_GH_ATTESTATION" = nosigner ]; then
                    echo "unknown flag: --signer-workflow" >&2
                    exit 1
                fi
                exit "$FAKE_GH_VERIFY"
                ;;
            *) exit "$FAKE_GH_VERIFY" ;;
        esac
        ;;
    *) exit 1 ;;
esac
GH
chmod 0755 "$tmp/gh/gh"
mkdir "$tmp/uname"
real_uname=$(command -v uname)
cat >"$tmp/uname/uname" <<EOF
#!/bin/sh
# TEST fake uname: the OS and CPU of FAKE_UNAME_S and FAKE_UNAME_M.
case "\$1" in
    -s) echo "\$FAKE_UNAME_S" ;;
    -m) echo "\$FAKE_UNAME_M" ;;
    *) exec "$real_uname" "\$@" ;;
esac
EOF
chmod 0755 "$tmp/uname/uname"

# ── Cases ────────────────────────────────────────────────────────────────────
# try <label> <installed|refused> <line> <releases URL> [VAR=value ...]: runs the installer
# (arguments in $ARGS) and checks the outcome, its exit status and that <line> was printed.
# Refused: the install directory holds nothing new. The PATH is $tmp/nogh unless PATH= is given.
ARGS=
# staged <dir>: true if a staged copy (.<name>.new) was left behind in <dir>.
staged() {
    for f in "$1"/.*.new; do
        [ -e "$f" ] && return 0
    done
    return 1
}
# run <label> <releases URL> [VAR=value ...]: the installer, its exit status in $rc,
# its output in $tmp/<label>.log, installing into $tmp/bin-<label>.
run() {
    label=$1 site=$2
    shift 2
    dir="$tmp/bin-$label"
    rc=0
    # shellcheck disable=SC2086 # ARGS is split on purpose
    env PATH="$tmp/nogh" HOME="$tmp/home-$label" X0NE_ENGINE_RELEASES_URL="$site" \
        X0NE_ENGINE_INSTALL_DIR="$dir" CURL_CA_BUNDLE="$tmp/cert.pem" FAKE_GH_LOG="$tmp/$label.gh" \
        FAKE_GH_VERSION= FAKE_GH_ATTESTATION= FAKE_GH_AUTH=1 FAKE_GH_VERIFY=1 \
        FAKE_UNAME_S=Linux FAKE_UNAME_M=x86_64 "$@" \
        sh "$installer" $ARGS >"$tmp/$label.log" 2>&1 || rc=$?
}
try() {
    label=$1 want=$2 line=$3 site=$4
    shift 4
    run "$label" "$site" "$@"
    got=refused
    [ ! -e "$dir/x0ne-engine" ] || got=installed
    good=1
    [ "$got" = "$want" ] || good=0
    case "$want:$rc" in
        installed:0 | refused:[1-9]*) ;;
        *) good=0 ;;
    esac
    if [ "$want" = installed ]; then
        for b in $bins; do [ -x "$dir/$b" ] || good=0; done
    fi
    staged "$dir" && good=0
    grep -qF -e "$line" "$tmp/$label.log" || good=0
    if [ "$good" = 1 ]; then
        ok "$label: $got, $(grep -F -e "$line" "$tmp/$label.log" | head -n 1 | cut -c1-110)"
    else
        ko "$label: $got (exit $rc), expected $want and \"$line\""
        sed 's/^/      /' "$tmp/$label.log"
    fi
}
# check <label> <condition text> <command...>: one more assertion on a case.
check() {
    label=$1 what=$2
    shift 2
    if "$@"; then ok "$label: $what"; else ko "$label: $what"; fi
}

if [ ! -e /lib64/ld-linux-x86-64.so.2 ]; then
    # No glibc loader (musl): the installer must stop before it downloads anything.
    : >"$tmp/requests"
    try no-glibc refused "no glibc loader at /lib64/ld-linux-x86-64.so.2" "$url/good"
    check no-glibc "no request reached the server" test ! -s "$tmp/requests"
    echo "installer: $n ok (no glibc loader on this system: the refusal only)"
    exit "$fail"
fi

: >"$tmp/requests"
try default installed "Installed $tmp/bin-default/x0ne-engine-proxy" "$url/good"
# The two downloads, and no other request (no telemetry).
want_req=$(printf 'https /good/download/v%s/%s\nhttps /good/download/v%s/SHA256SUMS\n' "$version" "$asset" "$version")
check default "only the archive and SHA256SUMS of v$version were requested" test "$(cat "$tmp/requests")" = "$want_req"
# shellcheck disable=SC2317 # called through check
same_files() {
    rm -rf "$tmp/x"
    mkdir "$tmp/x"
    tar -xzf "$tmp/www/good/download/v$version/$asset" -C "$tmp/x"
    for b in $bins; do [ "$(sha "$tmp/x/x0ne-engine-$target/$b")" = "$(sha "$tmp/bin-default/$b")" ] || return 1; done
}
check default "the installed binaries are the archive's, byte for byte" same_files
if [ "$real" = 1 ]; then
    check default "x0ne-engine --version runs: $("$tmp/bin-default/x0ne-engine" --version 2>&1 | head -n 1)" \
        "$tmp/bin-default/x0ne-engine" --version
fi
check default "the PATH hint is printed" grep -qF "export PATH=\"$tmp/bin-default:" "$tmp/default.log"
check default "provenance said NOT verified (no gh)" grep -qF "Provenance NOT verified: the GitHub CLI (gh) is not installed" "$tmp/default.log"

ARGS="--version 0.0.1-test"
try version-option installed "x0ne-engine OLDER" "$url/good"
ARGS="--version=v0.0.1-test"
try version-equals installed "x0ne-engine OLDER" "$url/good"
ARGS=
try version-env installed "x0ne-engine OLDER" "$url/good" X0NE_ENGINE_VERSION=0.0.1-test
ARGS="--version 9.9.9"
try version-unpublished refused "no release archive for $target (v9.9.9)" "$url/good"
ARGS="--version ../../x"
try version-invalid refused "invalid version: '../../x'" "$url/good"
ARGS="--version"
try version-no-value refused "--version needs a value" "$url/good"
ARGS="--bogus"
try unknown-option refused "sh install.sh --uninstall" "$url/good"
ARGS=

try checksum-mismatch refused "checksum MISMATCH for $asset" "$url/bad"
try sums-missing refused "no SHA256SUMS for this release: refusing to install an unverified archive" "$url/nosums"
try sums-unlisted refused "$asset is not listed in SHA256SUMS" "$url/unlisted"
try layout refused "unexpected archive layout: no x0ne-engine-$target/" "$url/flat"
try binary-missing refused "x0ne-engine-proxy is missing from $asset" "$url/noproxy"

: >"$tmp/requests"
try cpu-aarch64 refused "unsupported CPU: aarch64" "$url/good" PATH="$tmp/uname:$tmp/nogh" FAKE_UNAME_M=aarch64
try cpu-armv7l refused "unsupported CPU: armv7l" "$url/good" PATH="$tmp/uname:$tmp/nogh" FAKE_UNAME_M=armv7l
try os-darwin refused "unsupported OS: Darwin" "$url/good" PATH="$tmp/uname:$tmp/nogh" FAKE_UNAME_S=Darwin FAKE_UNAME_M=arm64
try os-freebsd refused "unsupported OS: FreeBSD" "$url/good" PATH="$tmp/uname:$tmp/nogh" FAKE_UNAME_S=FreeBSD FAKE_UNAME_M=amd64
check unsupported "no request reached the server" test ! -s "$tmp/requests"

# A binary this machine cannot run: refused, and the previous install stays in place.
mkdir -p "$tmp/bin-does-not-run"
for b in $bins; do printf '#!/bin/sh\necho "%s PREVIOUS"\n' "$b" >"$tmp/bin-does-not-run/$b"; done
chmod 0755 "$tmp/bin-does-not-run"/*
run does-not-run "$url/broken"
kept=1
for b in $bins; do grep -q PREVIOUS "$tmp/bin-does-not-run/$b" || kept=0; done
line="does not run on this machine: x0ne-engine: version 'GLIBC_2.99' not found"
if [ "$rc" != 0 ] && [ "$kept" = 1 ] && ! staged "$tmp/bin-does-not-run" &&
    grep -qF -e "$line" "$tmp/does-not-run.log"; then
    ok "does-not-run: refused (exit $rc), the previous binaries untouched, $(grep -F -e "$line" "$tmp/does-not-run.log" | cut -c1-80)"
else
    ko "does-not-run: exit $rc, previous kept: $kept"
    sed 's/^/      /' "$tmp/does-not-run.log"
fi

try http-mirror refused "download failed" "$http_url/good"
try https-redirected-to-http refused "download failed" "$url/redirect/good"
try without-curl refused "curl is required" "$url/good" PATH="$tmp/nocurl"

try no-gh installed "Provenance NOT verified: the GitHub CLI (gh) is not installed" "$url/good"
G="PATH=$tmp/gh:$tmp/nogh"
try gh-2.46 installed "Provenance NOT verified: gh 2.46.0 is too old" "$url/good" "$G" FAKE_GH_VERSION=2.46.0 FAKE_GH_ATTESTATION=none FAKE_GH_AUTH=0 FAKE_GH_VERIFY=0
try gh-2.50 installed "Provenance NOT verified: gh 2.50.0 is too old" "$url/good" "$G" FAKE_GH_VERSION=2.50.0 FAKE_GH_ATTESTATION=nosigner FAKE_GH_AUTH=0 FAKE_GH_VERIFY=0
try gh-2.55 installed "Provenance NOT verified: gh 2.55.0 is too old" "$url/good" "$G" FAKE_GH_VERSION=2.55.0 FAKE_GH_ATTESTATION=full FAKE_GH_AUTH=0 FAKE_GH_VERIFY=1
try gh-dev-without-attestation installed "Provenance NOT verified: gh DEV has no 'gh attestation verify --signer-workflow'" "$url/good" "$G" FAKE_GH_VERSION=DEV FAKE_GH_ATTESTATION=none FAKE_GH_AUTH=0 FAKE_GH_VERIFY=0
try gh-logged-out installed "Provenance NOT verified: gh is not logged in" "$url/good" "$G" FAKE_GH_VERSION=2.103.3 FAKE_GH_ATTESTATION=full FAKE_GH_AUTH=1 FAKE_GH_VERIFY=0
for c in gh-2.46 gh-2.50 gh-2.55 gh-logged-out; do check "$c" "gh was not asked to verify" sh -c "! grep -q '^gh attestation verify .*$asset' '$tmp/$c.gh'"; done
try gh-passes installed "Provenance verified (gh attestation verify)" "$url/good" "$G" FAKE_GH_VERSION=2.103.3 FAKE_GH_ATTESTATION=full FAKE_GH_AUTH=0 FAKE_GH_VERIFY=0
repo=$(sed -n 's/^REPO="\(.*\)"$/\1/p' "$installer")
want_gh="--repo ${repo:-?} --signer-workflow ${repo:-?}/.github/workflows/release.yml"
check gh-passes "gh verified with $want_gh" grep -qF -e "$want_gh" "$tmp/gh-passes.gh"
try gh-fails refused "provenance verification FAILED" "$url/good" "$G" FAKE_GH_VERSION=2.103.3 FAKE_GH_ATTESTATION=full FAKE_GH_AUTH=0 FAKE_GH_VERIFY=1
try gh-2.56-fails refused "provenance verification FAILED" "$url/good" "$G" FAKE_GH_VERSION=2.56.0 FAKE_GH_ATTESTATION=full FAKE_GH_AUTH=0 FAKE_GH_VERIFY=1
try gh-dev-fails refused "provenance verification FAILED" "$url/good" "$G" FAKE_GH_VERSION=DEV FAKE_GH_ATTESTATION=full FAKE_GH_AUTH=0 FAKE_GH_VERIFY=1
try tampered-gh-passes refused "checksum MISMATCH" "$url/bad" "$G" FAKE_GH_VERSION=2.103.3 FAKE_GH_ATTESTATION=full FAKE_GH_AUTH=0 FAKE_GH_VERIFY=0
check tampered-gh-passes "gh was not asked to verify" sh -c "! grep -q '^gh attestation verify' '$tmp/tampered-gh-passes.gh' 2>/dev/null"

# Uninstall: the three binaries go, anything else in the directory stays.
dir="$tmp/bin-default"
printf 'mine\n' >"$dir/other-tool"
rc=0
env PATH="$tmp/nogh" HOME="$tmp/home-default" X0NE_ENGINE_INSTALL_DIR="$dir" sh "$installer" --uninstall >"$tmp/uninstall.log" 2>&1 || rc=$?
gone=1
for b in $bins; do [ ! -e "$dir/$b" ] || gone=0; done
if [ "$rc" = 0 ] && [ "$gone" = 1 ] && [ -f "$dir/other-tool" ] && grep -qF "Removed $dir/x0ne-engine-proxy" "$tmp/uninstall.log"; then
    ok "uninstall: the three binaries removed, other-tool kept"
else
    ko "uninstall: exit $rc"
    sed 's/^/      /' "$tmp/uninstall.log"
fi
env PATH="$tmp/nogh" HOME="$tmp/home-default" X0NE_ENGINE_INSTALL_DIR="$dir" sh "$installer" --uninstall >"$tmp/uninstall2.log" 2>&1 || rc=$?
check uninstall "again: Nothing to remove" grep -qF "Nothing to remove in $dir" "$tmp/uninstall2.log"

if [ "$fail" = 0 ]; then echo "installer: $n ok"; else echo "installer: FAILED"; fi
exit "$fail"
