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
and why. Last recorded run: 2026-11-17 (cargo-audit 0.22.2, 1509 advisories,
380 crates; RustSec database at its commit 02e2517 of 2026-11-15, the copy
the local CI judges against, no fetch during the run), on the lock with
`rsa` 0.10.0 and with nothing ignored: no vulnerability, one warning
(`paste`, below). Before the move to `rsa` 0.10.0 the lock held 391 crates
and the same run gave the same result with RUSTSEC-2023-0071 ignored (below).
A first run that day reported three,
RUSTSEC-2026-0523 and RUSTSEC-2026-0524 (`quinn-proto` 0.11.18) and
RUSTSEC-2026-0538 (`quinn-udp` 0.5.15), entered in the database on
2026-11-13: the lock moved to `quinn-proto` 0.11.19 and `quinn-udp` 0.5.16
that day. `quinn` is in `Cargo.lock` and in no build: it is an optional
dependency of `reqwest` 0.12.28 that no feature of this workspace enables
(`cargo tree --workspace --all-features --target all -e all -i quinn-proto`
prints nothing, and nothing for `quinn-udp`, measured the same day), which is
why `cargo deny check`, which reads the resolved dependency graph, had not
reported them. `cargo deny check` (cargo-deny 0.20.2, same database, offline
against the vendored crates as `make ci` runs it; policy in `deny.toml`:
advisories, licenses, bans, sources) passed the same day: advisories, bans,
licenses and sources ok, 24 duplicate-version warnings; offline, it could not
read the crates.io index, so that run says nothing about yanked crates (375
`index-failure` warnings). On 2026-10-23 the
database at its commit 6e86877 reported RUSTSEC-2026-0385 (`ctutils` 0.4.2,
non-constant-time mask generation); the lock moved to `ctutils` 0.4.5 that
day. On 2026-11-03 `cargo deny check`, online this time (crates.io index read,
database at its commit 6c5fedf of that day), refused two things:
`crypto-bigint` 0.7.6, yanked, and RUSTSEC-2026-0429 (`zip` 8.6.0). The lock
moved to `crypto-bigint` 0.7.10 and `ctutils` 0.4.8, which 0.7.10 requires;
the advisory was ignored, with the three other `zip` ones (RUSTSEC-2026-0367,
0368 and 0369), all four in `zip`'s extraction, which nothing called; the
same online check then passed. On 2026-11-14 the lock moved to candle
0.11.1, which requires `zip` ^9.1.3, and to `zip` 9.5.1, where the four are
fixed: their ignores were removed. `cargo deny check`, offline against the
vendored crates and the database fetched that day, then passed: advisories,
bans, licenses and sources ok. `cargo deny` runs in CI (the `Security` workflow, on
pushes and pull requests to `main` and weekly, and the local `make ci`); no
CI job runs `cargo audit`. Run them yourself rather than trust this date.

Yanked crates are refused (`deny.toml`, `yanked = "deny"`): `yoke-derive`
0.8.3, yanked on crates.io on 2026-09-15, sat in `Cargo.lock` until
2026-10-05, when the lock moved to 0.8.4; `crypto-bigint` 0.7.6, found
yanked on 2026-11-03, left it that day for 0.7.10.

**RUSTSEC-2023-0071 — `rsa`, Marvin timing sidechannel: fixed, no longer
ignored.** RustSec gives it as patched in `rsa` 0.10.0 and later. The locks
moved from `rsa` 0.9.10 to 0.10.0 on 2026-11-17, and the ignore left
`.cargo/audit.toml` and `deny.toml` that day: `cargo audit` finds no
vulnerability without it. Until that day the advisory was ignored, for this
reason: Marvin recovers a *private* key through timing during RSA
operations that use one, and this crate never holds an RSA private key.
That is still so. `rsa` appears in four source files, and all four only
verify, with public keys (measured by grep of the workspace, 2026-11-17):
`nras.rs` builds a public key from the pinned NVIDIA intermediate 004
certificate and verifies the NRAS leaf's PKCS#1 v1.5 signature under it;
`nras_ocsp.rs` verifies the OCSP responses of that chain under their
signer's public key (PKCS#1 v1.5, SHA-256 or SHA-384); `gce_endorsement.rs`
builds one from the pinned GCE root, verifies the endorsement signer's
certificate under it, then the endorsement under the signer's key
(RSASSA-PSS, SHA-256); `snp.rs` verifies AMD's SEV-SNP certificate chain
(RSASSA-PSS, SHA-384). A private key appears in three of them, `nras.rs`,
`nras_ocsp.rs` and `gce_endorsement.rs`, each time inside a
`#[cfg(all(test, not(x0ne_sdk)))]` module, minting the synthetic chains the
tests verify against; it never ships. `rsa` 0.10 makes PKCS#1 v1.5 an opt-in
feature (`pkcs1v15`); it is on, for the verification of the NRAS leaf and of
the OCSP responses, and the encryption it also compiles is called nowhere.

**RUSTSEC-2024-0436 — `paste`, unmaintained.** A proc-macro: it runs in
the compiler while the crates that use it are built, and its own code is
not linked into a binary. Two crates pull it in. `tokenizers`, a normal
dependency of `x0ne-infer`: every build of `x0ne-infer` runs it, the served
ones included. The image
(`cargo tree -e normal -i paste -p x0ne-infer --features "cuda nvml"`) and
the .deb (`-p x0ne-infer -p x0ne-engine-cloud`, no feature) both reach it
through `tokenizers` 0.23.2 (measured on 2026-11-14). And `gemm`, which
candle uses for matrix multiplication. Since
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

What would make this live again: an advisory that reports a vulnerability in
`paste`, which is another id and is not ignored, or `paste` becoming code a
binary links. If you are adding either, remove the ignore rather than
reasoning about whether it still applies.

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
