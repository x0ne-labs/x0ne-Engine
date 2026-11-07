#!/bin/sh
# x0ne Engine installer: downloads a release of x0ne-Engine, verifies it,
# installs the client binaries. Where no release archive fits (a version or a
# platform without one), the way in is the source build (README.md, "Quick start").
#
#   curl --proto '=https' --tlsv1.2 -fsSL https://raw.githubusercontent.com/x0ne-labs/x0ne-Engine/main/install.sh | sh
#   curl --proto '=https' --tlsv1.2 -fsSL <same URL> | sh -s -- --version 0.2.0-beta.7
#   sh install.sh [--version VERSION] [--uninstall | --help]
#
# Environment:
#   X0NE_ENGINE_VERSION       release to install (0.2.0-beta.7 or v0.2.0-beta.7);
#                             default: VERSION below. --version takes precedence.
#   X0NE_ENGINE_INSTALL_DIR   where the binaries go; default: ~/.local/bin
#   X0NE_ENGINE_RELEASES_URL  releases base URL, for a mirror (https only); default: GitHub
#
# Linux x86_64 with the glibc loader only: the release workflow builds no other
# target. It downloads SHA256SUMS and SHA256SUMS.sig, verifies the signature with
# `ssh-keygen -Y verify` against the release key below, and refuses unless the
# list names the release asked for. Only then it downloads the archive, refuses
# on a missing or mismatching checksum, and, when gh 2.56+ is logged in, also
# verifies the build provenance with `gh attestation verify`. It runs the new
# x0ne-engine once, then moves three binaries into the install directory.
# No sudo, no telemetry: the only requests are the three downloads, plus gh's own
# when it verifies. It never edits a shell startup file.

set -eu

REPO="x0ne-labs/x0ne-Engine"
# The release this file ships with; the release workflow refuses a tag that differs.
# GitHub's releases/latest skips pre-releases, and Engine releases are 0.2.0-beta.N.
VERSION="0.2.0-beta.7"
RELEASES_URL="${X0NE_ENGINE_RELEASES_URL:-https://github.com/${REPO}/releases}"
INSTALL_DIR="${X0NE_ENGINE_INSTALL_DIR:-${HOME}/.local/bin}"
BINS="x0ne-engine x0ne-engine-cloud x0ne-engine-proxy"
LOADER="/lib64/ld-linux-x86-64.so.2"
# The release key, the maintainer's SSH key (SHA256:+8onkVcMPL3D/8vOL/xn5TrHLGWNYOKWbWIMwM+BPtE).
# SHA256SUMS.sig is its signature of SHA256SUMS, namespace x0ne-engine-release.
# Written here only: no variable or option replaces it or skips the check.
SIGNER="dev@x0ne.co"
NAMESPACE="x0ne-engine-release"
ALLOWED_SIGNER='dev@x0ne.co namespaces="x0ne-engine-release" ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICYdHwVN3y1azWxmVmm7sMQOI/BJGN2YI7qpGiQ0+4/Y'

say() { printf '%s\n' "$*"; }
die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<EOF
x0ne Engine installer

  sh install.sh                     install or update x0ne-engine (release ${VERSION})
  sh install.sh --version VERSION   install that release instead
  sh install.sh --uninstall         remove the binaries this script installs
  sh install.sh --help              this help

Environment: X0NE_ENGINE_VERSION (default: ${VERSION}), X0NE_ENGINE_INSTALL_DIR
(default: ~/.local/bin), X0NE_ENGINE_RELEASES_URL (default: GitHub releases).
EOF
}

# ── Platform and tools ───────────────────────────────────────────────────────

detect_target() {
    case "$(uname -s)" in
        Linux) ;;
        *) die "unsupported OS: $(uname -s) (releases are built for Linux x86_64 only; build from source: README.md, Quick start)" ;;
    esac
    case "$(uname -m)" in
        x86_64 | amd64) arch="x86_64" ;;
        *) die "unsupported CPU: $(uname -m) (releases are built for x86_64 only; build from source: README.md, Quick start)" ;;
    esac
    # Dynamic glibc binaries: without this loader (musl, as on Alpine) they cannot start.
    [ -e "$LOADER" ] ||
        die "no glibc loader at ${LOADER}: the release binaries need glibc (musl systems such as Alpine are not supported)"
    TARGET="${arch}-unknown-linux-gnu"
    ASSET="x0ne-engine-${TARGET}.tar.gz"
}

detect_tools() {
    # Not wget: GNU wget 1.25 follows an https to http redirect even with
    # --https-only, which applies to recursive mode only (measured 2026-11-02).
    command -v curl >/dev/null 2>&1 || die "curl is required"
    for tool in tar gzip awk; do
        command -v "$tool" >/dev/null 2>&1 || die "$tool is required"
    done
    if command -v sha256sum >/dev/null 2>&1; then
        SHA256="sha256sum"
    elif command -v shasum >/dev/null 2>&1; then
        SHA256="shasum -a 256"
    else
        die "sha256sum or shasum is required to verify the download"
    fi
    command -v ssh-keygen >/dev/null 2>&1 ||
        die "ssh-keygen is required to verify the release signature: install OpenSSH 8.1 or later (Debian and Ubuntu: openssh-client; Fedora: openssh; Alpine: openssh-keygen), then run this script again"
    # -Y verify came with OpenSSH 8.1; its usage text names it (8.0's does not).
    ssh-keygen -Y 2>&1 | grep -q -e '-Y verify' ||
        die "this ssh-keygen cannot verify signatures (no 'ssh-keygen -Y verify'): install OpenSSH 8.1 or later, then run this script again"
}

# fetch <url> <file>: fails on any HTTP error, and on any URL or redirect
# that is not https. Prints the last HTTP status, 000 when no response came back.
fetch() {
    curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL -w '%{http_code}' "$1" -o "$2"
}

# ── Download and verification ────────────────────────────────────────────────

# The tag goes into a URL: letters, digits, dots, plus and minus only.
resolve_tag() {
    v="${X0NE_ENGINE_VERSION:-$VERSION}"
    case "$v" in
        "" | *[!0-9A-Za-z.+-]*) die "invalid version: '${v}' (expected e.g. ${VERSION})" ;;
    esac
    case "$v" in
        v*) TAG="$v" ;;
        *) TAG="v$v" ;;
    esac
}

release_url() {
    printf '%s/download/%s/%s' "$RELEASES_URL" "$TAG" "$1"
}

# The checksum list and its signature come first: nothing else is fetched,
# read, extracted or run before the signature verifies.
download_sums() {
    say "Downloading $(release_url SHA256SUMS)"
    if ! status=$(fetch "$(release_url SHA256SUMS)" "${TMP}/SHA256SUMS"); then
        if [ "$status" = 404 ]; then
            die "no SHA256SUMS for ${TAG}: this release is not published, or this version does not exist. Nothing was installed. Build from source instead: README.md in https://github.com/${REPO}"
        fi
        die "download of SHA256SUMS failed (HTTP status ${status:-000})"
    fi
    if ! status=$(fetch "$(release_url SHA256SUMS.sig)" "${TMP}/SHA256SUMS.sig"); then
        if [ "$status" = 404 ]; then
            die "no SHA256SUMS.sig for ${TAG}: refusing a release without the maintainer's signature. Nothing was installed. Please report it: https://github.com/${REPO}/security"
        fi
        die "download of SHA256SUMS.sig failed (HTTP status ${status:-000})"
    fi
}

# The first line of the signed list names the release: an older list, signed
# too, served under this tag's URL is refused.
verify_signature() {
    printf '%s\n' "$ALLOWED_SIGNER" >"${TMP}/allowed_signers"
    if ! out=$(ssh-keygen -Y verify -f "${TMP}/allowed_signers" -I "$SIGNER" -n "$NAMESPACE" \
        -s "${TMP}/SHA256SUMS.sig" <"${TMP}/SHA256SUMS" 2>&1); then
        die "signature verification FAILED: SHA256SUMS of ${TAG} is not signed by the release key (${out:-no output}). Nothing was installed."
    fi
    say "Signature verified: ${out}"
    first=$(awk 'NR == 1 { print; exit }' "${TMP}/SHA256SUMS")
    [ "$first" = "# x0ne-Engine ${TAG}" ] ||
        die "the signed SHA256SUMS is not the list of ${TAG}: its first line is '${first}', not '# x0ne-Engine ${TAG}'. Nothing was installed."
}

download_archive() {
    say "Downloading $(release_url "$ASSET")"
    if ! status=$(fetch "$(release_url "$ASSET")" "${TMP}/${ASSET}"); then
        if [ "$status" = 404 ]; then
            die "no release archive for ${TARGET} (${TAG}): this release has no archive for this platform. Build from source instead: README.md in https://github.com/${REPO}"
        fi
        die "download failed (HTTP status ${status:-000})"
    fi
}

verify_checksum() {
    expected=$(awk -v f="$ASSET" '$2 == f || $2 == "*" f { print $1 }' "${TMP}/SHA256SUMS")
    [ -n "$expected" ] || die "${ASSET} is not listed in SHA256SUMS: refusing to install"
    actual=$($SHA256 "${TMP}/${ASSET}" | awk '{ print $1 }')
    [ "$expected" = "$actual" ] ||
        die "checksum MISMATCH for ${ASSET}: expected ${expected}, got ${actual}. Nothing was installed."
    say "Checksum verified (SHA-256)"
}

# gh_unusable: prints why gh cannot verify the provenance, nothing if it can.
# gh has `attestation` from 2.49 and --signer-workflow from 2.51; 2.51 to 2.55
# cannot read the current Sigstore trusted root (measured 2026-11-05).
gh_unusable() {
    if ! command -v gh >/dev/null 2>&1; then
        say "the GitHub CLI (gh) is not installed (gh 2.56 or later verifies it)"
        return 0
    fi
    v=$(gh --version 2>/dev/null | awk 'NR == 1 { print $3 }')
    major=${v%%.*}
    minor=${v#*.}
    minor=${minor%%.*}
    case "${major}:${minor}" in
        *[!0-9:]* | :* | *:) ;; # unreadable (a dev build): the help decides
        *)
            if [ "$major" -lt 2 ] || { [ "$major" -eq 2 ] && [ "$minor" -lt 56 ]; }; then
                say "gh ${v} is too old (gh 2.56 or later verifies it)"
                return 0
            fi
            ;;
    esac
    if ! gh attestation verify --help 2>&1 | grep -q -e '--signer-workflow'; then
        say "gh ${v:-(unknown version)} has no 'gh attestation verify --signer-workflow' (gh 2.56 or later has it)"
        return 0
    fi
    gh auth status >/dev/null 2>&1 || say "gh is not logged in (gh auth login)"
}

# An extra check on top of the signature: the attestation says the release
# workflow built the archive from the tagged source. A gh that cannot verify
# counts as no gh; a gh that verifies and fails stops the install.
verify_provenance() {
    why=$(gh_unusable)
    if [ -n "$why" ]; then
        say "Build provenance not checked: ${why}. The signed SHA256SUMS is the check."
        return 0
    fi
    gh attestation verify "${TMP}/${ASSET}" --repo "$REPO" \
        --signer-workflow "${REPO}/.github/workflows/release.yml" >/dev/null ||
        die "provenance verification FAILED for ${ASSET}. Nothing was installed."
    say "Provenance verified (gh attestation verify)"
}

# ── Install and uninstall ────────────────────────────────────────────────────

# Staged copies sit next to their targets as .<name>.new; removed on any exit.
unstage() {
    for bin in $BINS; do
        rm -f "${INSTALL_DIR:?}/.${bin}.new"
    done
}

install_files() {
    tar -xzf "${TMP}/${ASSET}" -C "$TMP"
    src="${TMP}/x0ne-engine-${TARGET}"
    [ -d "$src" ] || die "unexpected archive layout: no x0ne-engine-${TARGET}/ in ${ASSET}"
    for bin in $BINS; do
        [ -f "${src}/${bin}" ] || die "${bin} is missing from ${ASSET}"
    done
    mkdir -p "$INSTALL_DIR"
    for bin in $BINS; do
        cp "${src}/${bin}" "${INSTALL_DIR}/.${bin}.new"
        chmod 0755 "${INSTALL_DIR}/.${bin}.new"
    done
    # Run once where it will live, before anything is replaced: a binary this
    # machine cannot run (glibc too old, noexec mount) leaves the old install.
    if ! out=$("${INSTALL_DIR}/.x0ne-engine.new" --version 2>&1); then
        unstage
        die "the downloaded x0ne-engine does not run on this machine: ${out:-no output}. Nothing was installed."
    fi
    # Rename, never rewrite: a running binary is replaced, not modified.
    for bin in $BINS; do
        mv -f "${INSTALL_DIR}/.${bin}.new" "${INSTALL_DIR}/${bin}"
        say "Installed ${INSTALL_DIR}/${bin}"
    done
    say "$out"
}

check_path() {
    case ":${PATH}:" in
        *":${INSTALL_DIR}:"*) ;;
        *)
            say ""
            say "${INSTALL_DIR} is not in your PATH. This script edits no shell file;"
            say "add this line to your shell startup file (~/.profile, ~/.bashrc or ~/.zshrc):"
            say "  export PATH=\"${INSTALL_DIR}:\$PATH\""
            ;;
    esac
}

uninstall() {
    removed=0
    for bin in $BINS; do
        if [ -e "${INSTALL_DIR}/${bin}" ]; then
            rm -f "${INSTALL_DIR:?}/${bin}"
            say "Removed ${INSTALL_DIR}/${bin}"
            removed=1
        fi
    done
    [ "$removed" = 1 ] || say "Nothing to remove in ${INSTALL_DIR}"
    say "Kept the model cache (\$X0NE_ENGINE_HOME, default ~/.cache/x0ne-engine) and the"
    say "session logs (~/.local/state/x0ne-engine); remove them by hand if you wish."
}

main() {
    action=install
    while [ $# -gt 0 ]; do
        case "$1" in
            -h | --help)
                usage
                exit 0
                ;;
            --uninstall) action=uninstall ;;
            --version)
                [ $# -ge 2 ] || die "--version needs a value (e.g. ${VERSION})"
                X0NE_ENGINE_VERSION=$2
                shift
                ;;
            --version=*) X0NE_ENGINE_VERSION=${1#--version=} ;;
            *)
                usage >&2
                exit 2
                ;;
        esac
        shift
    done
    if [ "$action" = uninstall ]; then
        uninstall
        exit 0
    fi
    resolve_tag
    detect_target
    detect_tools
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"; unstage' EXIT
    trap 'exit 130' INT TERM
    download_sums
    verify_signature
    download_archive
    verify_checksum
    verify_provenance
    install_files
    check_path
}

main "$@"
