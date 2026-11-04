#!/bin/bash
# One more TEST key, in the shape of x0ne OS scripts/gen-test-key.sh (version of 2026-10-17):
# a single ed25519 primary that may sign (30 days), no subkey; the signature is made by that
# primary, as cvm-sign.sh signs reference.json with it. The archive key never signs this way.
# Run like gen.sh; the secret lives in /tmp of this container and dies with it.
set -euo pipefail
export LANG=C.UTF-8 LC_ALL=C.UTF-8
O=/out; R=/in/reference.json; H=/tmp/p
G() { gpg --homedir "$H" --batch --quiet --no-tty --yes --faked-system-time "$1!" \
  --passphrase '' --pinentry-mode loopback "${@:2}"; }
mkdir -m 700 -p "$H"
G 20261016T000000 --quick-gen-key 'x0ne OS TEST key, primary only, not for release <test@invalid>' ed25519 sign 30d
p=$(gpg --homedir "$H" --batch --with-colons --list-keys | awk -F: '$1=="fpr"{print $10; exit}')
G 20261016T000000 --export "$p" > $O/primary-only-keyring.gpg
G 20261016T010000 --digest-algo SHA512 --detach-sign -o $O/reference.json.primary-only.sig "$R"
V=$(mktemp -d); gpg --homedir "$V" --batch --quiet --import $O/primary-only-keyring.gpg 2>/dev/null
echo "primary-only-keyring + reference.json.primary-only.sig: $(gpg --homedir "$V" --batch --status-fd 1 \
  --verify $O/reference.json.primary-only.sig "$R" 2>/dev/null | awk '{print $2}' \
  | grep -E 'GOODSIG|BADSIG|ERRSIG|VALIDSIG|NO_PUBKEY' | sort -u | tr '\n' ' ')" > $O/gnupg-verdicts-primary-only.txt
echo "primary-only $p" > $O/FINGERPRINTS-primary-only.txt
