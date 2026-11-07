#!/bin/sh
# Tests of install.sh against fake releases served on 127.0.0.1: HTTPS with a certificate
# made here, plain HTTP, and an HTTPS redirect to HTTP. Each case runs the installer with
# its own HOME and install directory, and a PATH without the real gh. The releases are
# signed with a TEST key made here, which replaces the release key in a copy of install.sh.
# Cases: install (the default version, --version, X0NE_ENGINE_VERSION), the three requests
# and nothing else; the signature (missing, not a signature, another key, another namespace,
# SHA256SUMS replaced, the signed list of another release, no release line, no ssh-keygen,
# an ssh-keygen without -Y verify, the key overridden from the environment), refused before
# the archive is downloaded; checksum mismatch, SHA256SUMS missing, archive not listed,
# unsupported OS and CPU (a fake uname), a binary that does not run (the old install is
# kept), http and redirects to http, no curl, provenance (a fake gh: absent, too old, logged
# out, verifies, fails), uninstall. Then the README's .deb steps and "Verify a release by
# hand", replayed with sh and bash: each stops at the first check that fails.
# Without the glibc loader (Alpine) only the refusal is tested.
# Adapted from x0ne-Harness scripts/test-install-https-only.sh and test-install-provenance.sh.
#
#   sh ci/tests/installer.sh            (ci/install-matrix.sh runs it in four distributions)
#
# ENGINE_ARCHIVE=<x0ne-engine-x86_64-unknown-linux-gnu.tar.gz> serves that archive (real
# binaries) as the good release. INSTALL_SH overrides the script under test. Needs curl,
# openssl, python3, tar, gzip, ssh-keygen. Writes only under a temporary directory.
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
installer=${INSTALL_SH:-$root/install.sh}
readme=$root/README.md
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

for tool in curl openssl python3 tar gzip ssh-keygen; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "FAIL  $tool is not installed"
        exit 1
    }
done
version=$(sed -n 's/^VERSION="\(.*\)"$/\1/p' "$installer")
repo=$(sed -n 's/^REPO="\(.*\)"$/\1/p' "$installer")
if [ -z "$version" ] || [ -z "$repo" ]; then
    echo "FAIL  no VERSION= or REPO= line in $installer"
    exit 1
fi
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
    # Its "SHA256SUMS" step, replayed in a fake workspace with bash as on the runner: the
    # first line is the one install.sh wants for the tag, and sha256sum -c takes the list.
    if command -v bash >/dev/null 2>&1; then
        awk 'index($0, "- name: SHA256SUMS") { f = 1; next }
            f && /^ *run: \|/ { r = 1; next }
            r && /^ *- (name|uses):/ { exit }
            r { print }' "$wf" | sed 's/^          //' >"$tmp/sums-step.sh"
        mkdir -p "$tmp/ws/dist"
        for f in a.tar.gz b.deb c.cdx.json binaries.sha256; do printf '%s\n' "$f" >"$tmp/ws/dist/$f"; done
        cp "$installer" "$tmp/ws/install.sh"
        rc=0
        (cd "$tmp/ws" && TAG="v$version" bash "$tmp/sums-step.sh") >"$tmp/sums-step.log" 2>&1 || rc=$?
        first=$(awk 'NR == 1' "$tmp/ws/dist/SHA256SUMS" 2>/dev/null || true)
        if [ "$rc" = 0 ] && [ "$first" = "# x0ne-Engine v$version" ] && grep -q '  install.sh$' "$tmp/ws/dist/SHA256SUMS"; then
            ok "release.yml: SHA256SUMS starts with \"$first\", sha256sum -c passes"
        else
            ko "release.yml: the SHA256SUMS step (exit $rc) wrote \"$first\", expected \"# x0ne-Engine v$version\""
            sed 's/^/      /' "$tmp/sums-step.log"
        fi
    fi
fi

# ── Keys: TEST keys, made here, removed with $tmp ────────────────────────────
mkdir "$tmp/keys"
for k in release other; do
    ssh-keygen -q -t ed25519 -N '' -C "TEST $k key, ci/tests/installer.sh" -f "$tmp/keys/$k"
done
allowed=$(sed -n "s/^ALLOWED_SIGNER='\(.*\)'\$/\1/p" "$installer")
real_key=$(printf '%s\n' "$allowed" | sed -n 's|.* ssh-ed25519 \([A-Za-z0-9+/=]*\)$|\1|p')
test_key=$(cut -d ' ' -f 2 "$tmp/keys/release.pub")
other_key=$(cut -d ' ' -f 2 "$tmp/keys/other.pub")
[ -n "$real_key" ] || {
    echo "FAIL  no ALLOWED_SIGNER='... ssh-ed25519 <key>' line in $installer"
    exit 1
}
# The installer under test: install.sh with the TEST key in place of the release key, nothing else.
sed "s|$real_key|$test_key|g" "$installer" >"$tmp/install.sh"
if [ "$(grep -c -F -e "$real_key" "$installer")" = 1 ] && ! grep -q -F -e "$real_key" "$tmp/install.sh" &&
    [ "$(grep -v '^ALLOWED_SIGNER=' "$installer" | cksum)" = "$(grep -v '^ALLOWED_SIGNER=' "$tmp/install.sh" | cksum)" ]; then
    ok "install.sh under test: its one release key line, with a TEST key"
else
    echo "FAIL  install.sh: the release key is not on exactly one ALLOWED_SIGNER line"
    exit 1
fi
under_test=$installer
installer=$tmp/install.sh

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
# deb <dir> <marker>: a stand-in x0ne-engine_amd64.deb (only the fake sudo receives it).
deb() {
    printf 'TEST deb %s\n' "$2" >"$1/x0ne-engine_amd64.deb"
}
# sha <file>: its SHA-256 (python3: Fedora's base image has no cmp, and the
# installer's own sha256sum is what is under test).
sha() {
    python3 -I -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"
}
# sums <dir> <file>...: SHA256SUMS of these files, in that directory, after the line
# "# x0ne-Engine <tag>" the release workflow writes first (<tag>: the directory's name).
sums() {
    dir=$1
    shift
    {
        printf '# x0ne-Engine %s\n' "${dir##*/}"
        (cd "$dir" && for f in "$@"; do printf '%s  %s\n' "$(sha "$f")" "$f"; done)
    } >"$dir/SHA256SUMS"
}
# sign <dir> [<key> [<namespace>]]: <dir>/SHA256SUMS.sig, by default the TEST release key's
# signature in the release namespace, as the maintainer makes it.
sign() {
    ssh-keygen -Y sign -f "$tmp/keys/${2:-release}" -n "${3:-x0ne-engine-release}" "$1/SHA256SUMS" 2>"$tmp/sign.log" || {
        echo "FAIL  ssh-keygen -Y sign: $(cat "$tmp/sign.log")"
        exit 1
    }
}
# flip <file>: one byte changed.
flip() {
    python3 -I -c '
import sys
p = sys.argv[1]
b = bytearray(open(p, "rb").read())
b[len(b) // 2] ^= 1
open(p, "wb").write(b)' "$1"
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
deb "$r" GOOD
sums "$r" "$asset" x0ne-engine-other.txt x0ne-engine_amd64.deb
sign "$r"
goodrel=$r
older=$(release good v0.0.1-test)
archive "$older/$asset" OLDER
deb "$older" OLDER
sums "$older" "$asset" x0ne-engine_amd64.deb
sign "$older"
# bad: the good SHA256SUMS and signature, the archive with one byte changed.
r=$(release bad)
cp "$goodrel/$asset" "$goodrel/SHA256SUMS" "$goodrel/SHA256SUMS.sig" "$r/"
flip "$r/$asset"
r=$(release nosums)
archive "$r/$asset" NOSUMS
r=$(release unlisted)
archive "$r/$asset" UNLISTED
cp "$r/$asset" "$r/renamed.tar.gz"
sums "$r" renamed.tar.gz
rm "$r/renamed.tar.gz"
sign "$r"
r=$(release broken)
archive "$r/$asset" BROKEN "echo \"x0ne-engine: version 'GLIBC_2.99' not found\" >&2; exit 1"
sums "$r" "$asset"
sign "$r"
r=$(release flat)
archive "$r/$asset" FLAT "" flat
sums "$r" "$asset"
sign "$r"
r=$(release noproxy)
archive "$r/$asset" NOPROXY "" noproxy
sums "$r" "$asset"
sign "$r"

# Releases the signature must refuse. Their x0ne-engine creates $tmp/canary if it ever runs;
# canary, signed with the TEST key, shows that it does.
evil="touch '$tmp/canary'; echo pwned"
r=$(release canary)
archive "$r/$asset" CANARY "$evil"
sums "$r" "$asset"
sign "$r"
# nosig: SHA256SUMS without its signature.
r=$(release nosig)
archive "$r/$asset" NOSIG "$evil"
deb "$r" NOSIG
sums "$r" "$asset" x0ne-engine_amd64.deb
# notasig: a .sig that is no signature.
r=$(release notasig)
archive "$r/$asset" NOTASIG "$evil"
sums "$r" "$asset"
printf -- '-----BEGIN SSH SIGNATURE-----\nU1NIU0lHAAAAAQ==\n-----END SSH SIGNATURE-----\n' >"$r/SHA256SUMS.sig"
# forged: who controls the release assets replaces the archive, the .deb and SHA256SUMS,
# and keeps the maintainer's signature.
r=$(release forged)
archive "$r/$asset" FORGED "$evil"
deb "$r" FORGED
sums "$r" "$asset" x0ne-engine_amd64.deb
cp "$goodrel/SHA256SUMS.sig" "$r/"
# otherkey: signed by another key.
r=$(release otherkey)
archive "$r/$asset" OTHERKEY "$evil"
deb "$r" OTHERKEY
sums "$r" "$asset" x0ne-engine_amd64.deb
sign "$r" other
# othernamespace: the release key, the namespace of a signed git commit.
r=$(release othernamespace)
archive "$r/$asset" OTHERNAMESPACE "$evil"
sums "$r" "$asset"
sign "$r" release git
# replay: the signed release v0.0.1-test, served as v$version.
r=$(release replay)
cp "$older/$asset" "$older/x0ne-engine_amd64.deb" "$older/SHA256SUMS" "$older/SHA256SUMS.sig" "$r/"
# noline: signed, but without the line that names the release.
r=$(release noline)
archive "$r/$asset" NOLINE "$evil"
sums "$r" "$asset"
awk 'NR > 1' "$r/SHA256SUMS" >"$r/SHA256SUMS.new"
mv "$r/SHA256SUMS.new" "$r/SHA256SUMS"
sign "$r"
# For the README's .deb steps: the .deb changed, not listed, missing.
r=$(release debtampered)
cp "$goodrel/SHA256SUMS" "$goodrel/SHA256SUMS.sig" "$r/"
deb "$r" TAMPERED
r=$(release debunlisted)
deb "$r" UNLISTED
printf 'other\n' >"$r/x0ne-engine-other.txt"
sums "$r" x0ne-engine-other.txt
sign "$r"
r=$(release debmissing)
cp "$goodrel/SHA256SUMS" "$goodrel/SHA256SUMS.sig" "$r/"

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

# ── PATHs: without gh, ssh-keygen or curl; a fake gh, ssh-keygen, uname, sudo ─
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
strip "$tmp/nossh" gh ssh-keygen

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
# The fake ssh-keygen of OpenSSH 8.0, which has no -Y: its usage, exit 1, whatever it is asked.
mkdir "$tmp/oldssh"
cat >"$tmp/oldssh/ssh-keygen" <<'SSH'
#!/bin/sh
# TEST fake ssh-keygen (OpenSSH 8.0 usage text).
echo "ssh-keygen: invalid option -- 'Y'" >&2
echo "usage: ssh-keygen [-q] [-b bits] [-t dsa | ecdsa | ed25519 | rsa] [-m format]" >&2
echo "       ssh-keygen -Q -f krl_file file ..." >&2
exit 1
SSH
chmod 0755 "$tmp/oldssh/ssh-keygen"
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
# The fake sudo of the README replays: records its arguments and the SHA-256 of the last
# one, runs nothing.
mkdir "$tmp/sudo"
cat >"$tmp/sudo/sudo" <<'SUDO'
#!/bin/sh
# TEST fake sudo: runs nothing.
for last; do :; done
printf '%s sha256=%s\n' "$*" "$(python3 -I -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$last")" >>"$FAKE_SUDO_LOG"
SUDO
chmod 0755 "$tmp/sudo/sudo"

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
# untouched: since $tmp/requests was emptied, the archive was not requested, and no
# x0ne-engine of a refused release ever ran.
# shellcheck disable=SC2317 # called through check
untouched() {
    ! grep -qF -e "/$asset" "$tmp/requests" && [ ! -e "$tmp/canary" ]
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
# The three downloads, the signed list first, and no other request (no telemetry).
want_req=$(printf 'https /good/download/v%s/%s\n' "$version" SHA256SUMS "$version" SHA256SUMS.sig "$version" "$asset")
check default "only SHA256SUMS, SHA256SUMS.sig, then the archive of v$version were requested" test "$(cat "$tmp/requests")" = "$want_req"
# shellcheck disable=SC2317 # called through check
same_files() {
    rm -rf "$tmp/x"
    mkdir "$tmp/x"
    tar -xzf "$goodrel/$asset" -C "$tmp/x"
    for b in $bins; do [ "$(sha "$tmp/x/x0ne-engine-$target/$b")" = "$(sha "$tmp/bin-default/$b")" ] || return 1; done
}
check default "the installed binaries are the archive's, byte for byte" same_files
if [ "$real" = 1 ]; then
    check default "x0ne-engine --version runs: $("$tmp/bin-default/x0ne-engine" --version 2>&1 | head -n 1)" \
        "$tmp/bin-default/x0ne-engine" --version
fi
check default "the PATH hint is printed" grep -qF "export PATH=\"$tmp/bin-default:" "$tmp/default.log"
check default "the signature verified: Good \"x0ne-engine-release\" signature for dev@x0ne.co" \
    grep -qF 'Signature verified: Good "x0ne-engine-release" signature for dev@x0ne.co' "$tmp/default.log"
check default "build provenance said not checked (no gh)" grep -qF "Build provenance not checked: the GitHub CLI (gh) is not installed" "$tmp/default.log"

ARGS="--version 0.0.1-test"
try version-option installed "x0ne-engine OLDER" "$url/good"
ARGS="--version=v0.0.1-test"
try version-equals installed "x0ne-engine OLDER" "$url/good"
ARGS=
try version-env installed "x0ne-engine OLDER" "$url/good" X0NE_ENGINE_VERSION=0.0.1-test
ARGS="--version 9.9.9"
try version-unpublished refused "no SHA256SUMS for v9.9.9: this release is not published" "$url/good"
ARGS="--version ../../x"
try version-invalid refused "invalid version: '../../x'" "$url/good"
ARGS="--version"
try version-no-value refused "--version needs a value" "$url/good"
ARGS="--bogus"
try unknown-option refused "sh install.sh --uninstall" "$url/good"
ARGS=

try checksum-mismatch refused "checksum MISMATCH for $asset" "$url/bad"
try sums-missing refused "no SHA256SUMS for v$version" "$url/nosums"
try sums-unlisted refused "$asset is not listed in SHA256SUMS" "$url/unlisted"
try layout refused "unexpected archive layout: no x0ne-engine-$target/" "$url/flat"
try binary-missing refused "x0ne-engine-proxy is missing from $asset" "$url/noproxy"

# The signature. The canary goes off when its x0ne-engine runs; in every refusal below,
# the archive is not even downloaded.
try canary installed "pwned" "$url/canary"
check canary "its x0ne-engine ran (--version) and made the canary" test -e "$tmp/canary"
rm -f "$tmp/canary"
sigfail="signature verification FAILED: SHA256SUMS of v$version is not signed by the release key"
for c in "sig-missing:nosig:no SHA256SUMS.sig for v$version: refusing a release without the maintainer's signature" \
    "sig-not-a-signature:notasig:$sigfail" \
    "sig-sums-replaced:forged:$sigfail" \
    "sig-other-key:otherkey:$sigfail" \
    "sig-other-namespace:othernamespace:$sigfail" \
    "sig-other-release:replay:the signed SHA256SUMS is not the list of v$version: its first line is '# x0ne-Engine v0.0.1-test'" \
    "sig-no-release-line:noline:the signed SHA256SUMS is not the list of v$version"; do
    label=${c%%:*} rest=${c#*:}
    : >"$tmp/requests"
    try "$label" refused "${rest#*:}" "$url/${rest%%:*}"
    check "$label" "the archive was not downloaded, nothing ran" untouched
done
# The key and the check come from the script only: variables of the same names change nothing.
: >"$tmp/requests"
try sig-key-from-environment refused "$sigfail" "$url/otherkey" \
    ALLOWED_SIGNER="dev@x0ne.co namespaces=\"x0ne-engine-release\" ssh-ed25519 $other_key" \
    SIGNER=dev@x0ne.co NAMESPACE=x0ne-engine-release
check sig-key-from-environment "the archive was not downloaded, nothing ran" untouched
: >"$tmp/requests"
try no-ssh-keygen refused "ssh-keygen is required to verify the release signature: install OpenSSH 8.1 or later" "$url/good" PATH="$tmp/nossh"
try ssh-keygen-8.0 refused "this ssh-keygen cannot verify signatures (no 'ssh-keygen -Y verify')" "$url/good" PATH="$tmp/oldssh:$tmp/nossh"
check ssh-keygen "no request reached the server" test ! -s "$tmp/requests"

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

try http-mirror refused "download of SHA256SUMS failed" "$http_url/good"
try https-redirected-to-http refused "download of SHA256SUMS failed" "$url/redirect/good"
try without-curl refused "curl is required" "$url/good" PATH="$tmp/nocurl"

try no-gh installed "Build provenance not checked: the GitHub CLI (gh) is not installed" "$url/good"
G="PATH=$tmp/gh:$tmp/nogh"
try gh-2.46 installed "Build provenance not checked: gh 2.46.0 is too old" "$url/good" "$G" FAKE_GH_VERSION=2.46.0 FAKE_GH_ATTESTATION=none FAKE_GH_AUTH=0 FAKE_GH_VERIFY=0
try gh-2.50 installed "Build provenance not checked: gh 2.50.0 is too old" "$url/good" "$G" FAKE_GH_VERSION=2.50.0 FAKE_GH_ATTESTATION=nosigner FAKE_GH_AUTH=0 FAKE_GH_VERIFY=0
try gh-2.55 installed "Build provenance not checked: gh 2.55.0 is too old" "$url/good" "$G" FAKE_GH_VERSION=2.55.0 FAKE_GH_ATTESTATION=full FAKE_GH_AUTH=0 FAKE_GH_VERIFY=1
try gh-dev-without-attestation installed "Build provenance not checked: gh DEV has no 'gh attestation verify --signer-workflow'" "$url/good" "$G" FAKE_GH_VERSION=DEV FAKE_GH_ATTESTATION=none FAKE_GH_AUTH=0 FAKE_GH_VERIFY=0
try gh-logged-out installed "Build provenance not checked: gh is not logged in" "$url/good" "$G" FAKE_GH_VERSION=2.103.3 FAKE_GH_ATTESTATION=full FAKE_GH_AUTH=1 FAKE_GH_VERIFY=0
for c in gh-2.46 gh-2.50 gh-2.55 gh-logged-out; do check "$c" "gh was not asked to verify" sh -c "! grep -q '^gh attestation verify .*$asset' '$tmp/$c.gh'"; done
try gh-passes installed "Provenance verified (gh attestation verify)" "$url/good" "$G" FAKE_GH_VERSION=2.103.3 FAKE_GH_ATTESTATION=full FAKE_GH_AUTH=0 FAKE_GH_VERIFY=0
want_gh="--repo $repo --signer-workflow $repo/.github/workflows/release.yml"
check gh-passes "gh verified with $want_gh" grep -qF -e "$want_gh" "$tmp/gh-passes.gh"
try gh-fails refused "provenance verification FAILED" "$url/good" "$G" FAKE_GH_VERSION=2.103.3 FAKE_GH_ATTESTATION=full FAKE_GH_AUTH=0 FAKE_GH_VERIFY=1
try gh-2.56-fails refused "provenance verification FAILED" "$url/good" "$G" FAKE_GH_VERSION=2.56.0 FAKE_GH_ATTESTATION=full FAKE_GH_AUTH=0 FAKE_GH_VERIFY=1
try gh-dev-fails refused "provenance verification FAILED" "$url/good" "$G" FAKE_GH_VERSION=DEV FAKE_GH_ATTESTATION=full FAKE_GH_AUTH=0 FAKE_GH_VERIFY=1
try tampered-gh-passes refused "checksum MISMATCH" "$url/bad" "$G" FAKE_GH_VERSION=2.103.3 FAKE_GH_ATTESTATION=full FAKE_GH_AUTH=0 FAKE_GH_VERIFY=0
check tampered-gh-passes "gh was not asked to verify" sh -c "! grep -q '^gh attestation verify' '$tmp/tampered-gh-passes.gh' 2>/dev/null"
: >"$tmp/requests"
try forged-gh-passes refused "$sigfail" "$url/forged" "$G" FAKE_GH_VERSION=2.103.3 FAKE_GH_ATTESTATION=full FAKE_GH_AUTH=0 FAKE_GH_VERIFY=0
check forged-gh-passes "gh was not asked to verify, the archive not downloaded" \
    sh -c "! grep -q '^gh attestation verify' '$tmp/forged-gh-passes.gh' 2>/dev/null && ! grep -qF '/$asset' '$tmp/requests'"

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

# ── README: the .deb steps and "Verify a release by hand", replayed ─────────
# block <text>: the first ```sh block of README.md that contains <text>.
block() {
    awk -v w="$1" '
        $0 == "```sh" { inb = 1; b = ""; next }
        inb && $0 == "```" { inb = 0; if (index(b, w)) { printf "%s", b; found = 1; exit } next }
        inb { b = b $0 "\n" }
        END { exit !found }' "$readme"
}
releases="https://github.com/$repo/releases"
shells="sh"
if command -v bash >/dev/null 2>&1; then shells="sh bash"; fi
if ! block 'dpkg -i' >"$tmp/deb.block" || ! block 'verified: ' >"$tmp/hand.block"; then
    ko "README.md: no sh block with dpkg -i, or none with \"verified: \""
else
    for b in deb hand; do
        if [ "$(grep -cF -e "echo '$allowed' > allowed_signers" "$tmp/$b.block")" = 1 ] &&
            grep -qF -e "tag=v$version" "$tmp/$b.block" && grep -qF -e "url=$releases/download/" "$tmp/$b.block"; then
            ok "README.md, $b block: install.sh's release key, tag v$version, $releases"
        else
            ko "README.md, $b block: not install.sh's ALLOWED_SIGNER, or not tag=v$version, or not url=$releases/download/"
        fi
    done
    deb_sha=$(sha "$goodrel/x0ne-engine_amd64.deb")
    # replay <label> <deb|hand> <site> <passes|stops> [VAR=value ...]: the block with sh and bash,
    # its URL on <site> of the test server, the TEST key in place of the release key. deb passes:
    # the fake sudo got "dpkg -i" and the good .deb; it stops: exit not 0 and no sudo at all.
    replay() {
        label=$1 kind=$2 site=$3 want=$4
        shift 4
        sed -e "s|$releases|$url/$site|g" -e "s|$real_key|$test_key|g" "$tmp/$kind.block" >"$tmp/$label.sh"
        if grep -q 'github\.com' "$tmp/$label.sh"; then
            ko "$label: the replayed block still names github.com"
            return
        fi
        for shell in $shells; do
            log="$tmp/$label.$shell.log" sudolog="$tmp/$label.$shell.sudo"
            mkdir -p "$tmp/rtmp-$label-$shell"
            rc=0
            env PATH="$tmp/sudo:$tmp/nogh" CURL_CA_BUNDLE="$tmp/cert.pem" TMPDIR="$tmp/rtmp-$label-$shell" \
                FAKE_SUDO_LOG="$sudolog" "$@" "$shell" "$tmp/$label.sh" >"$log" 2>&1 || rc=$?
            passed=0
            if [ "$kind" = deb ]; then
                grep -q "^dpkg -i .*/x0ne-engine_amd64.deb sha256=$deb_sha\$" "$sudolog" 2>/dev/null && passed=1
            else
                grep -q "^verified: .*/$asset\$" "$log" && passed=1
            fi
            if [ "$want:$rc:$passed" = passes:0:1 ]; then
                ok "$label ($shell): passed, $(tail -n 1 "$log" | cut -c1-90)"
            elif [ "$want" = stops ] && [ "$rc" != 0 ] && [ "$passed" = 0 ] && [ ! -s "$sudolog" ]; then
                ok "$label ($shell): stopped (exit $rc), $(grep -v '^ *$' "$log" | tail -n 1 | cut -c1-90)"
            else
                ko "$label ($shell): exit $rc, expected it $want"
                sed 's/^/      /' "$log"
            fi
        done
    }
    replay readme-deb deb good passes
    replay readme-deb-sig-missing deb nosig stops
    replay readme-deb-sums-replaced deb forged stops
    replay readme-deb-other-key deb otherkey stops
    replay readme-deb-other-release deb replay stops
    replay readme-deb-tampered deb debtampered stops
    replay readme-deb-unlisted deb debunlisted stops
    replay readme-deb-missing deb debmissing stops
    replay readme-deb-no-ssh-keygen deb good stops PATH="$tmp/sudo:$tmp/nossh"
    replay readme-by-hand hand good passes
    replay readme-by-hand-sums-replaced hand forged stops
    replay readme-by-hand-other-release hand replay stops
fi

[ "$under_test" = "$root/install.sh" ] || echo "installer: tested $under_test (INSTALL_SH)"
if [ "$fail" = 0 ]; then echo "installer: $n ok"; else echo "installer: FAILED"; fi
exit "$fail"
