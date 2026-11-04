# Security policy

x0ne Engine is a security project: the `x0ne-infer` server refuses to serve
without cryptographic proof of the execution environment. Reports that show
a way to reach its socket, its key, or sealed weights without that proof
are the highest-value contribution this project can receive.

## Supported versions

Security fixes go to 0.2.0-beta.7 only, the version this repository is
published at: no earlier version was published.

## Reporting

Please report vulnerabilities privately first: open a GitHub security
advisory (Security → Advisories → Report a vulnerability) on this
repository. You should hear back within 72 hours.

Please do NOT open public issues for exploitable findings before a fix
ships.

## Known advisories

`cargo audit` is the first command many readers run, so here is what it said
and why. Last recorded run: 2026-10-31 (cargo-audit 0.22.2, 1395 advisories,
416 crates; RustSec database at its commit 2295466 of 2026-10-30, the copy
the local CI judges against, no fetch that day): no vulnerability, one
warning (`paste`, below). `cargo deny check` (cargo-deny 0.20.2, same
database, offline against the vendored crates as `make ci` runs it; policy in
`deny.toml`: advisories, licenses, bans, sources) passed the same day:
advisories, bans, licenses and sources ok, 24 duplicate-version warnings;
offline, it could not read the crates.io index, so that run says nothing
about yanked crates (383 `index-failure` warnings). On 2026-10-23 the
database at its commit 6e86877 reported RUSTSEC-2026-0385 (`ctutils` 0.4.2,
non-constant-time mask generation); the lock moved to `ctutils` 0.4.5 that
day. On 2026-11-03 `cargo deny check`, online this time (crates.io index read,
database at its commit 6c5fedf of that day), refused two things:
`crypto-bigint` 0.7.6, yanked, and RUSTSEC-2026-0429 (`zip`, below). The lock
moved to `crypto-bigint` 0.7.10 and `ctutils` 0.4.8, which 0.7.10 requires;
the advisory is ignored, with the three other `zip` ones; the same online
check then passed. `cargo deny` runs in CI (the `Security` workflow, on
pushes and pull requests to `main` and weekly, and the local `make ci`); no
CI job runs `cargo audit`. Run them yourself rather than trust this date.

Yanked crates are refused (`deny.toml`, `yanked = "deny"`): `yoke-derive`
0.8.3, yanked on crates.io on 2026-09-15, sat in `Cargo.lock` until
2026-10-05, when the lock moved to 0.8.4; `crypto-bigint` 0.7.6, found
yanked on 2026-11-03, left it that day for 0.7.10.

**RUSTSEC-2023-0071 — `rsa` 0.9.x, Marvin timing sidechannel (medium, no
fixed upgrade available).** Ignored in `.cargo/audit.toml`, and this is the
reason. Marvin recovers a *private* key through timing during RSA
operations that use one. This crate never holds an RSA private key. `rsa`
appears in four source files, and all four only verify, with public keys
(measured by grep of the workspace, 2026-11-03): `nras.rs` builds an
`RsaPublicKey` from the pinned NVIDIA intermediate 004 certificate and
verifies the NRAS leaf's PKCS#1 v1.5 signature under it; `nras_ocsp.rs`
verifies the OCSP responses of that chain under their signer's public key
(PKCS#1 v1.5, SHA-256 or SHA-384); `gce_endorsement.rs` builds one from the
pinned GCE root, verifies the endorsement signer's certificate under it,
then the endorsement under the signer's key (RSASSA-PSS, SHA-256); `snp.rs`
verifies AMD's SEV-SNP certificate chain (RSASSA-PSS, SHA-384).
`RsaPrivateKey` appears in three of them, `nras.rs`, `nras_ocsp.rs` and
`gce_endorsement.rs`, each time inside a `#[cfg(all(test, not(x0ne_sdk)))]`
module, minting the synthetic chains the tests verify against; it never
ships.

What would make this live again: any code path that decrypts with RSA, signs
with RSA, or accepts an RSA key from a caller. If you are adding one, remove
the ignore rather than reasoning about whether it still applies.

**RUSTSEC-2024-0436 — `paste`, unmaintained.** A proc-macro pulled in
transitively by `gemm`, which candle uses for matrix multiplication. Since
2026-11-04 candle is the reference of the tests only, on the CPU: the
feature `candle-reference`, which the tests' dev-dependency turns on and no
feature of a build reaches, `cuda` included
(`x0ne-infer/tests/served_build.rs`). The build of the image,
`cargo tree -e normal -p x0ne-infer --features "cuda nvml"`, lists no
`candle-*` crate (277 packages, none of them candle, measured on
2026-11-04). candle stays in `Cargo.lock`
for the tests. Not ignored by `cargo audit` (`.cargo/audit.toml`), which
reports unmaintained crates as warnings, not failures: there the warning
stays visible. `cargo deny`, the check CI runs, ignores it (`deny.toml`), so
CI shows nothing for it.

**RUSTSEC-2026-0367, RUSTSEC-2026-0368, RUSTSEC-2026-0369 — `zip` 8.6.0,
extraction (setuid bits kept, symlinks out of the destination, permissions
applied through symlinks; fixed in 9.0.0 according to RustSec, in 9.0.1
according to GHSA-4684-h2q9-2vcp, the GitHub advisory behind
RUSTSEC-2026-0369).**
Ignored in `.cargo/audit.toml` and `deny.toml`, and this is the reason. All
three are in the code that writes an archive's entries to disk:
`ZipArchive::extract`, `ZipArchive::extract_unwrapped_root_dir` (both
through `extract_internal`, `read.rs` lines 298 and 360 of the vendored
8.6.0) and `ZipStreamReader::extract`. In that version, every
`set_permissions`, `symlink` and directory creation of the reading side is
reached from these two paths only. `zip` reaches this workspace through
`candle-core` 0.11.0 only, which requires `zip` ^8.6 and does not make it
optional; 0.11.0 is candle's latest release (crates.io, 2026-10-21).
candle-core opens zip archives in `npy.rs` (npz files) and `pickle.rs`
(PyTorch checkpoints) with `ZipArchive::new`, `by_index` and `by_name`,
writes npz files with `ZipWriter`, and never extracts. No crate of this
workspace calls `zip`, `candle_core::npy`, `candle_core::pickle`, nor
`VarBuilder::from_npz` or `from_pth`, and the catalog refuses pickle
checkpoints (`registry/catalog.rs`): only safetensors weights are loaded.
Measured on 2026-10-17 by grep of the three crates and of the vendored
candle-core, candle-nn and candle-transformers 0.11.0, and by `nm` on every
binary of the workspace (debug, default and all features): the only `zip`
symbols are those of `zip::result::ZipError`, which candle's `Error` wraps;
no `ZipArchive`, no `ZipStreamReader`, no candle npy or pickle function.
Measured again on 2026-10-21, by the same grep and by `nm` on the debug and
release binaries that `make ci` builds: same result. The test
`the_zip_ignores_hold_only_while_nothing_opens_an_archive`
(`x0ne-infer/tests/security_doc.rs`) fails if `zip` gains a dependent other
than candle-core 0.11.0, or if a source of the workspace names `zip`, an
archive type, or candle's npz and PyTorch loaders. Since 2026-11-04
candle-core is in the tests only (above).

What would make this live again: a call to `ZipArchive::extract`,
`ZipArchive::extract_unwrapped_root_dir` or `ZipStreamReader::extract`, to
candle's npy or pickle loaders, or a candle-core release that calls them.
Re-examine by 2026-11-17, or when a candle-core release accepts `zip` 9.0.1
or later: upgrade then, and remove the three ignores.

**RUSTSEC-2026-0429 — `zip` 8.6.0, `ZipArchive::extract` aborts the process
on a symlink entry with a large declared size (denial of service, fixed in
9.0.0).** Ignored in `.cargo/audit.toml` and `deny.toml`, for the reason of
the three above. The allocation it describes, a symlink target's buffer sized
from the size the archive declares, is in `extract_internal` (`read.rs` line
417 of the vendored 8.6.0) and in `ZipStreamReader::extract`
(`read/stream.rs` line 72): the same paths, which nothing in this workspace
calls (measured by grep of the vendored crate, 2026-11-03), and the same test
fails if a source of the workspace starts to name them.

What would make this live again: what would make the three above live, a
call to either extraction path. Re-examine by 2026-11-17 with them, and
remove the four ignores together.

## Scope notes for researchers

- The limitations listed in `docs/limitations.md` are the boundary of today's
  guarantees. Each verdict's `verifier` string carries the verifier's own
  subset of them, not all of them (the v0 key transport, for
  example, is not in it). A finding inside a stated
  limitation is still worth reporting (it helps prioritize), but it is a
  known gap, not a bypass.
- The `x0ne-infer` binary has no development bypass: no feature, variable
  or flag makes it serve without a verified enclave (outside a TEE it refuses
  before it binds, whatever the features: exit 87 in a release build whose
  measurements file has nothing to pin, as this version's, else 64). A
  report that `x0ne-infer`, built with any features, can be made to serve
  unattested is critical.
- Security-relevant lines in the code are marked `SECURITY:`. Security tests
  are written to state, in their doc-comment, what a failure would mean;
  that convention has not been audited across every test.
