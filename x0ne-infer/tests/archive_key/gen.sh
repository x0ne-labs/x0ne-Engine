#!/bin/bash
# Throwaway TEST keys, made the way scripts/archive-key/archive-key.sh makes the archive key
# (x0ne OS of 2026-10-16: primary ed25519 certify-only 5y, signing subkey ed25519 2y), and detached
# signatures made the way scripts/release-sign.sh signs (--detach-sign --digest-algo SHA512
# --local-user SUBKEY!). Secrets live in /tmp of this container and die with it.
set -euo pipefail
export LANG=C.UTF-8 LC_ALL=C.UTF-8
O=/out; R=/in/reference.json
G() { local H=$1 t=$2; shift 2; gpg --homedir "$H" --batch --quiet --no-tty --yes --faked-system-time "$t!" \
  --passphrase '' --pinentry-mode loopback "$@"; }
gen() { # <home> <uid> <time> <subkey expiry>; prints the primary fingerprint
  local H=$1; mkdir -m 700 -p "$H"
  G "$H" "$3" --quick-generate-key "$2" ed25519 cert 5y
  local p; p=$(gpg --homedir "$H" --batch --with-colons --list-secret-keys | awk -F: '$1=="fpr"{print $10; exit}')
  G "$H" "$3" --quick-add-key "$p" ed25519 sign "$4"
  echo "$p"
}
subs() { gpg --homedir "$1" --batch --with-colons --list-keys | awk -F: '$1=="sub"{getline; print $10}'; }
sign() { G "$1" "$2" --digest-algo SHA512 --local-user "$3!" --detach-sign -o "$4" "$R"; }

A=/tmp/a; B=/tmp/b; E=/tmp/e
pa=$(gen $A 'x0ne Engine TEST archive key, not for release <test@invalid>' 20261016T000000 2y)
sa=$(subs $A | head -1)
G $A 20261016T000000 --export "$pa" > $O/test-archive-keyring.gpg
sign $A 20261016T010000 "$sa" $O/reference.json.sig

pb=$(gen $B 'rogue TEST key <rogue@invalid>' 20261016T000000 2y)
sb=$(subs $B | head -1)
G $B 20261016T000000 --export "$pb" > $O/rogue-keyring.gpg
sign $B 20261016T010000 "$sb" $O/reference.json.rogue.sig

pe=$(gen $E 'x0ne Engine TEST key, subkey expires after a day <test@invalid>' 20250101T000000 1d)
se=$(subs $E | head -1)
G $E 20250101T000000 --export "$pe" > $O/expired-subkey-keyring.gpg
sign $E 20250101T010000 "$se" $O/reference.json.expired.sig

# Rotation then revocation of the first subkey (archive-key.sh rotate, revoke-subkey ... compromised).
G $A 20261016T020000 --quick-add-key "$pa" ed25519 sign 2y
sa2=$(subs $A | tail -1)
printf 'key %s\nrevkey\ny\n1\nx0ne OS archive key: subkey compromised\n\ny\nsave\n' "$sa" \
  | G $A 20261016T030000 --command-fd 0 --status-fd 1 --edit-key "$pa" >/dev/null
G $A 20261016T030000 --export "$pa" > $O/revoked-subkey-keyring.gpg
sign $A 20261016T040000 "$sa2" $O/reference.json.rotated.sig

for k in test-archive-keyring rogue-keyring expired-subkey-keyring revoked-subkey-keyring; do
  gpg --homedir "$(mktemp -d)" --batch --list-packets $O/$k.gpg > $O/$k.packets.txt 2>&1 || true
done
for s in reference.json.sig reference.json.rogue.sig reference.json.expired.sig reference.json.rotated.sig; do
  gpg --homedir "$(mktemp -d)" --batch --list-packets $O/$s > $O/$s.packets.txt 2>&1 || true
done
# GnuPG's own verdicts, for the record (real clock: 2026-10-16).
for case in "test-archive-keyring reference.json.sig" "rogue-keyring reference.json.sig" \
            "expired-subkey-keyring reference.json.expired.sig" "revoked-subkey-keyring reference.json.sig" \
            "revoked-subkey-keyring reference.json.rotated.sig"; do
  set -- $case; V=$(mktemp -d); gpg --homedir "$V" --batch --quiet --import $O/$1.gpg 2>/dev/null
  echo "$1 + $2: $(gpg --homedir "$V" --batch --status-fd 1 --verify $O/$2 $R 2>/dev/null | awk '{print $2}' | grep -E 'GOODSIG|BADSIG|ERRSIG|EXPKEYSIG|REVKEYSIG|VALIDSIG|NO_PUBKEY|KEYEXPIRED|KEYREVOKED' | sort -u | tr '\n' ' ')"
done > $O/gnupg-verdicts.txt
{ echo "primary $pa"; echo "subkey  $sa (revoked in revoked-subkey-keyring)"; echo "subkey2 $sa2 (rotated in)";
  echo "rogue   $pb $sb"; echo "expired $pe $se"; gpg --version | head -2; } > $O/FINGERPRINTS.txt
cp $R $O/reference.json
ls -l $O
