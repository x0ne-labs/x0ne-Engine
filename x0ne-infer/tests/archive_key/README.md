# TEST archive keys and signatures, made by GnuPG

Public keys and detached signatures that `src/openpgp.rs` and
`src/reference.rs` check against GnuPG's own output. **TEST keys, not for
release**: every user ID says so, and a release pin refuses a TEST user ID
(`archive_key::pinned`). No secret key exists any more: they lived in a
container's `/tmp` and died with it.

Made on 2026-10-16 by [`gen.sh`](gen.sh), with GnuPG 2.4.7 (libgcrypt 1.11.0,
Debian 13), times faked with `--faked-system-time`. The keys follow x0ne-OS
`scripts/archive-key/archive-key.sh` as of 2026-10-16 (primary ed25519,
certify-only, 5 years; signing subkey ed25519, 2 years), and the signatures
`scripts/release-sign.sh` (`--detach-sign --digest-algo SHA512 --local-user
SUBKEY!`, binary).

```sh
podman run --rm --network=none -v "$PWD/in:/in:ro" -v "$PWD/out:/out" \
  <debian 13 image with gnupg> bash /path/to/gen.sh
```

`in/reference.json` is `reference.json` here. Keys are random at each run:
a new run gives new fingerprints, and the tests pin these.

| File | What |
|---|---|
| `test-archive-keyring.gpg` | primary `2F957D30FAE5CD616C23FD44AAC12AC5E4D7FDCB`, signing subkey `43B2C8EB81620E1C19492F2C86D2CD081726C748`, made 2026-10-16T00:00Z |
| `reference.json.sig` | `reference.json` signed by that subkey at 01:00Z |
| `rogue-keyring.gpg`, `reference.json.rogue.sig` | a second TEST primary and subkey, signing the same file |
| `expired-subkey-keyring.gpg`, `reference.json.expired.sig` | made 2025-01-01, subkey valid one day, signed at 01:00Z that day |
| `revoked-subkey-keyring.gpg` | the first key after a rotation (new subkey `E0486A12674E4FBF617FD3A5B1D55A0047986463` at 02:00Z) and the revocation of the first subkey, reason "compromised", at 03:00Z |
| `reference.json.rotated.sig` | signed by the new subkey at 04:00Z |
| `primary-only-keyring.gpg`, `reference.json.primary-only.sig` | made 2026-10-17 by [`gen-primary-only.sh`](gen-primary-only.sh), times faked to 2026-10-16T00:00Z and 01:00Z: primary `8023EEA58F1ECFCB4F9210365086A8DF9D53A766` that may sign, no subkey, as x0ne OS `scripts/gen-test-key.sh` (version of 2026-10-17) makes the TEST key of `make cvm-sign`; the signature is that primary's |
| `gnupg-verdicts.txt` | `gpg --verify` status for each pair, for the record |

GnuPG 2.4.7 made every self-signature and binding with SHA-512 (digest
algorithm 10), measured in `--list-packets` on 2026-10-16.
