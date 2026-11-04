# Vendored trust material — provenance and re-verification

Every binary artifact in this directory is **measured, not trusted on faith**:
a unit test digests each file through the same code path the verifier uses
and compares the result to the pinned constant in the source. If either the
file or the constant changes, a test fails — both are review events, never
silent. This file records where each artifact came from and how to re-fetch
and re-check it yourself.

## `intel_sgx_root_ca.der` (659 bytes)

The Intel SGX Provisioning Certification Root CA — the anchor of the DCAP
chain (`src/pcs.rs`, `INTEL_SGX_ROOT_SPKI_SHA256_HEX`).

- Source: `https://certificates.trustedservices.intel.com/Intel_SGX_Provisioning_Certification_RootCA.cer`
  (Intel's documented distribution point), fetched 2026-08-20.
- Re-fetch and compare:

  ```sh
  curl -fsSL -o /tmp/intel_root.cer \
    https://certificates.trustedservices.intel.com/Intel_SGX_Provisioning_Certification_RootCA.cer
  cmp /tmp/intel_root.cer x0ne-infer/tests/intel_sgx_root_ca.der && echo identical
  ```

- The anchoring test: `pcs::tests` digests this DER's SPKI with the exact
  x509-parser path production uses and asserts equality with the constant.

## `nvidia_nras_gpu_intermediate_004.der` and `nvidia_nras_gpu_intermediate_005.der`

The two NVIDIA intermediate CAs that sign NRAS's GPU attestation JWT leaves
(`src/nras.rs`, `NVIDIA_NRAS_GPU_INTERMEDIATE_PINS`): "NVIDIA Attestation
Service GPU Intermediate 004" (RSA-2048, signs leaves sha256WithRSAEncryption,
`NVIDIA_NRAS_GPU_INTERMEDIATE_004_SPKI_SHA256_HEX`) and "… Intermediate 005"
(P-384, signs leaves ecdsa-with-SHA384,
`NVIDIA_NRAS_GPU_INTERMEDIATE_005_SPKI_SHA256_HEX`). Both are issued by
"NVIDIA Attestation Service CA 001" (authority key ID `BE:0A:6F:ED:…:04:BF`).

- Source: the `x5c[1]` entries of NVIDIA's **production JWKS**,
  `https://nras.attestation.nvidia.com/.well-known/jwks.json`: 004 on
  2026-08-20; 005 on 2026-10-18 at 10:03:39 UTC, when 72 of the 200 keys
  chained to 005 and 128 to 004. Certificate SHA-256: 004
  `2df8907c…e27c5175`, 005 `38e3daf3…31a0d343`.
- Stated limitation (carried inside every verdict this crate emits): NVIDIA
  publishes no root certificate for this chain, so the pins are the
  intermediates. CA 001 alone is not pinned, on purpose: it would accept any
  intermediate NVIDIA certifies under it, including one nobody here measured.
- Re-fetch and compare (the JWKS rotates leaves; the intermediates should be
  stable, and a new one is exactly what the pins exist to catch):

  ```sh
  curl -fsS https://nras.attestation.nvidia.com/.well-known/jwks.json \
    | python3 -c 'import json,sys; print("\n".join(sorted({k["x5c"][1] for k in json.load(sys.stdin)["keys"]})))' \
    | while read -r c; do echo "$c" | base64 -d | sha256sum; done
  sha256sum x0ne-infer/tests/nvidia_nras_gpu_intermediate_00[45].der
  ```

- Each is pinned twice: the SHA-256 of its whole DER
  (`NVIDIA_NRAS_GPU_INTERMEDIATE_00[45]_CERT_SHA256_HEX`, which the decoded
  `x5c[1]` must match byte for byte) and of its SPKI. CA 001's signature on it is not
  checked, so the whole-certificate pin is what makes its validity window
  NVIDIA's. On 2026-10-18 at 10:03:39 UTC, all 200 `x5c[1]` of the JWKS were
  byte-identical to one of these two files.
- The anchoring test: `nras::tests::the_production_pins_match_nvidias_published_intermediates`
  digests both certificates and both SPKIs through the verifier's own code
  path, checks each key type, and that the production pin set holds these
  two and nothing else.

## `nvidia_nras_gpu_leaf_under_004_2026-10-18.der` and `nvidia_nras_gpu_leaf_under_005_2026-10-18.der`

Two NRAS signing leaves ("NVIDIA Attestation Service GPU GH100", P-384),
one under each intermediate, taken from the same JWKS GET of 2026-10-18
10:03:39 UTC: kid `nv-eat-kid-prod-20261018085814195-7ee2ab06-…` (under 004,
valid 08:53:05 to 20:53:35 UTC that day) and
`nv-eat-kid-prod-20261018072240741-79256c57-…` (under 005, valid 2026-10-18
07:12:12 to 2026-10-20 07:12:42 UTC). `openssl verify -partial_chain`
accepted each under its own intermediate. They expire; the test judges them at
the time of the GET.

- The test: `nras::tests::nvidias_real_leaves_verify_under_their_own_production_pin_only`.
  NVIDIA's own ECDSA-SHA384 encoding under 005, not only rcgen's.

## `pcs_live/` (quote.dat, tcb_info.json, tcb_issuer_chain.pem, qe_identity.json, qe_issuer_chain.pem)

A frozen snapshot used by the live-TCB tests (`tests/tcb_live.rs`) so the
tcbInfo/QE-identity parsing and signature checks run offline and
deterministically, on real bytes:

- `quote.dat` — a genuine TDX v4 quote (the public sample of
  https://github.com/MoeMahhouk/tdx-quote-parser; its chain verifies to the
  real pinned Intel root). That repository states no license (GitHub reports
  none, read 2026-11-03); the file is a quote a TDX machine produced, kept
  byte for byte as test data. The GCE quote of 2026-10-18
  (`run_fixtures/`) does not replace it: `tcb_policy_live.rs` needs a
  platform below every level of Intel's latest recovery, and the collateral
  below is for this quote's FMSPC.
- The collateral — Intel PCS **production** answers for that quote's FMSPC
  (`https://api.trustedservices.intel.com/tdx/certification/v4/…`), fetched
  2026-08-20; validity window 2026-08-20 → 2026-09-19, and the tests use a
  fixed `NOW` inside it so they do not rot with the wall clock. When the
  window passes, re-vendor and move `NOW` (said in `tcb_live.rs`).
- The sample platform is genuinely STALE against today's tcbInfo — that is
  the point: the refusal path runs on real signatures. The accept path is
  proven under an rcgen chain, because no fresh UpToDate quote can be
  vendored without hardware in hand.

These are TEST FIXTURES, not anchors: production fetches live collateral at
startup and refuses stale or unverifiable answers (fail-closed). Nothing in
the runtime path reads these files.

## `pcs_live/pck_crl_platform_2026-10-18.der` (3354 bytes)

Intel's PCK CRL for the "Intel SGX PCK Platform CA", DER, the issuer of the
PCK in `run_fixtures/gce-a3-tdx-2026-10-18.quote`.

- Source: `GET https://api.trustedservices.intel.com/sgx/certification/v4/pckcrl?ca=platform&encoding=der`
  (Intel PCS v4, "Get Revocation List V4"), answered with `Date: Sun, 18 Oct
  2026 19:08:58 GMT`, `Content-Type: application/pkix-crl`. SHA-256
  `193fb81ebe38c4e6aaad3d4a4a6c9afcc1203aeb4c792843eb5416de517b0f09`.
- Content as read by `openssl crl -inform der -text` that day: thisUpdate
  2026-10-18 18:34:22 UTC, nextUpdate 2026-11-17 18:34:22 UTC, 57 revoked
  serials, all with reason "Key Compromise"; not the serial of the 18/10 PCK
  (`862C5FFF…E5694BC`). `openssl crl -CAfile <PCK Platform CA + root of that
  quote>` said `verify OK`.
- The processor CA's CRL of the same GET (1003 bytes, 13 serials) was read,
  not vendored. The root CA CRL read that day is the one vendored on
  2026-10-23 below, byte for byte (same window, same number).
- Re-fetch and compare (a new CRL is expected every time: Intel re-signs it):

  ```sh
  curl -fsS -o /tmp/pckcrl.der \
    'https://api.trustedservices.intel.com/sgx/certification/v4/pckcrl?ca=platform&encoding=der'
  openssl crl -inform der -in /tmp/pckcrl.der -noout -lastupdate -nextupdate
  ```

## `pcs_live/2026-10-23/root_ca_crl.der` (294 bytes)

Intel's root CA CRL: what the Intel SGX Root CA has revoked among the
certificates it issued (PCK Platform and Processor CAs, TCB Signing). Used
by `tests/root_crl_live.rs` with the 18/10 quote and the TCB signing chain
above; `tcb::verify_root_crl` reads the live one in the TD and in the relay.

- Source: `https://certificates.trustedservices.intel.com/IntelSGXRootCA.der`,
  the CRL distribution point named in the root certificate itself (the test
  reads it out of the certificate), fetched 2026-10-23 at 06:11:57 UTC
  (`Last-Modified: Wed, 30 Sep 2026 07:01:36 GMT`). SHA-256
  `c114be5d5ca7aab2d37817eca55aa8a2353853809904ff3695b31c26bcb7465b`.
  PCS v4 `GET /sgx/certification/v4/rootcacrl` answered 404 that day.
- Content as read by `openssl crl -inform der -text`: version 2, CRL number
  1, thisUpdate 2026-02-26 13:04:00 UTC, nextUpdate 2027-02-26 13:04:00 UTC,
  authority key identifier `22650CD6…2706AC` (the root's serial), no
  revoked certificate. `openssl crl -CAfile <root of tcb_issuer_chain.pem>`
  said `verify OK`.
- Re-fetch and compare (identical until Intel re-issues it, expected before
  2027-02-26):

  ```sh
  curl -fsS -o /tmp/rootcrl.der https://certificates.trustedservices.intel.com/IntelSGXRootCA.der
  cmp /tmp/rootcrl.der x0ne-infer/tests/pcs_live/2026-10-23/root_ca_crl.der && echo identical
  ```

- The tests: `tests/pck_crl_live.rs`, judged at the time of the GET and at
  both ends of the window.

## `pcs_live/2026-10-22/` (Intel PCS of 2026-10-22, the TCB floor)

Intel PCS production answers fetched on 2026-10-22 between 10:55:38 and
10:55:40 UTC (the `Date` headers), read by `tests/tcb_policy_live.rs` at a
fixed `NOW` of 10:55:40. The issuer chains of the tcbInfo and QE identity
answers were byte for byte `tcb_issuer_chain.pem` and `qe_issuer_chain.pem`
of 2026-08-20, so they are not vendored twice.

| File | GET (`https://api.trustedservices.intel.com` + ) | Evaluation | issueDate | SHA-256 |
|---|---|---|---|---|
| `tcb_info_00806f050000_early.json` | `/tdx/certification/v4/tcb?fmspc=00806f050000&update=early` | 22 | 2026-10-22T10:32:14Z | `79df600afa5af823a09ada61fc8ce690bbd28cbb04ad119977d77df5b3fe10f6` |
| `tcb_info_00806f050000_standard.json` | `/tdx/certification/v4/tcb?fmspc=00806f050000` | 20 | 2026-10-22T10:07:27Z | `d22db49a200eb726b948284bae59b367d75ba56ce8f8a1b6dc60724f4eebcd07` |
| `tcb_info_90c06f000000_early.json` | `/tdx/certification/v4/tcb?fmspc=90c06f000000&update=early` | 22 | 2026-10-22T10:23:35Z | `2fcfedefa457265d482e0d9ba2a3ce5efd6223a4820fbca210e8a91bc6400c81` |
| `qe_identity_early.json` | `/tdx/certification/v4/qe/identity?update=early` | 22 | 2026-10-22T10:36:16Z | `519276965b877acafaedee6a583e131a062e4953665db17a812369256c104429` |
| `qe_identity_standard.json` | `/tdx/certification/v4/qe/identity` | 20 | 2026-10-22T09:48:13Z | `e444e64af3f04dcb74f2aa5bf46df13ef12b71650343c011dc93ecdcfd803539` |
| `pck_crl_platform.der` | `/sgx/certification/v4/pckcrl?ca=platform&encoding=der` | - | thisUpdate 2026-10-22T10:37:17Z | `331983f4ca14550abdc648b99281d5d533a7d2ec9e5fcadb190567e5bbcf5743` |

Every piece is valid 30 days from its issue date. FMSPC `00806f050000` is
the platform of `run_fixtures/gce-a3-tdx-2026-10-18.quote` (GCE
a3-highgpu-1g), `90c06f000000` that of the public sample `quote.dat`.

Read the same day, not vendored: `GET
/tdx/certification/v4/tcbevaluationdatanumbers` (with and without
`update=early`, same answer) listed evaluation 22 with
`tcbRecoveryEventDate` 2026-08-11, 21 with 2026-02-10, 20 with 2025-08-12,
19 with 2025-05-13. Intel's PCS API documentation (read 2026-10-22) says
`standard` is the default for `update` and is "commonly 12 months after
public disclosure of the items in scope". In the evaluation 22 tcbInfo for
`00806f050000`, the TDX_01 level at ISVSVN 11 is OutOfDate with
INTEL-SA-01314, -01397, -01419 and -01436; in the `standard` one the same
ISVSVN is UpToDate.

Re-fetch and compare (a new issue date is expected every time):

```sh
curl -fsS -D - -o /tmp/tcb.json \
  'https://api.trustedservices.intel.com/tdx/certification/v4/tcb?fmspc=00806f050000&update=early'
jq '.tcbInfo | {tcbEvaluationDataNumber, issueDate, nextUpdate}' /tmp/tcb.json
```

## `pcs_live/2026-10-28/` (Intel PCS of 2026-10-28, the hosts a strict TD refused)

The two tcbInfo answers for FMSPC `00806f050000` fetched during the GCE run of
2026-10-28, when a TD of that day (strict, the only policy then) refused its
a3-highgpu host in us-central1-a and in us-east5-a (exit 76, "TCB status
OutOfDate for FMSPC 00806f050000 ... level dated 2026-02-11"). Read by
`src/tcb.rs` (`the_gce_hosts_of_2026_10_28_on_intels_two_answers_of_that_day`)
at a fixed `NOW` of 2026-10-28T00:56:58Z, the `Date` header of the standard
answer. Its `TCB-Info-Issuer-Chain` header was byte for byte
`tcb_issuer_chain.pem`, so it is not vendored twice. The hosts' quotes and
that night's QE identity were not kept.

| File | GET (`https://api.trustedservices.intel.com` + ) | Evaluation | issueDate | SHA-256 |
|---|---|---|---|---|
| `tcb_info_00806f050000_early.json` | `/tdx/certification/v4/tcb?fmspc=00806f050000&update=early` | 22 | 2026-10-28T00:35:59Z | `ed7dc5d62bf2b5fc788a71174813695eb5ba5fecb061b5f30b080433e04c2751` |
| `tcb_info_00806f050000_standard.json` | `/tdx/certification/v4/tcb?fmspc=00806f050000` | 20 | 2026-10-28T00:00:55Z | `b1f61e393395262e4343df4b76ef24143c9562869f8d5b0b36d51177e7a5e986` |

Under evaluation 22 the level of 2026-02-11 (PCESVN 11, SGX components
9, 9, 2, 2, 4, 1, 0, 6, TDX components 5, 0, 9) is OutOfDate under
INTEL-DOC-10003, INTEL-SA-01439 and INTEL-SA-01442; under evaluation 20 the
newest level, of 2025-08-13, asks the same SVNs and is UpToDate. A
`transparent` verdict accepts the second and names the first.

## `nras_live/gpu0_2026-10-17.claims.json` and `nras_live/gpu0_2026-10-18.claims.json`

The claims of the per-GPU NRAS token (`document.gpu[0].nras_token`) of the
attestation document the relay saved on each cloud run: the JWT's payload
segment, base64url-decoded, bytes as NRAS signed them. Read by
`nras_policy_live.rs`. The JWT itself is not vendored (gitleaks' `jwt`
rule).

| File | Source (the run's attestation document, kept outside this repository) | SHA-256 of the token | SHA-256 of the file |
|---|---|---|---|
| `gpu0_2026-10-17.claims.json` | run of 2026-10-17 | `8b92c6781d6375e82fd3f987218aea935b43e247d7d644f6b23b8b93aae0e547` | `4e84c095d80ea6a8e8c505ea13cff2e6f77f7e52d562aa4eee85393eb62f6a0e` |
| `gpu0_2026-10-18.claims.json` | run of 2026-10-18 | `4fe99747853384bb34c3dd0d5464a453e91e2482ad75dfd5f596d8904508c773` | `ae3fb7857a8927dddb8c49009bc1515e0ea5d1c27e020cb1445b42923c574a5e` |

The 2026-10-17 token digest is the `submods` digest of GPU-0 in that run's
overall token. Both carry the three certificate chains `valid`, OCSP `good`, OCSP
nonce matched and response valid, `secboot` true, `dbgstat` "disabled", and
no `x-nvidia-mismatch-opaque-data-records`. No signature is checked here,
only the claims are vendored; the relay checked the tokens on the run
(verdict VERIFIED both days).

## Rules

1. Never update a vendored file and its pinned constant in the same commit
   without saying WHY in the commit message — the pair changing together is
   what an attacker's commit would look like.
2. A new artifact lands with: its source URL, the fetch date, the anchoring
   test, and an entry here.

## `gce_tcb_root_1.der` (1570 bytes)

Google's root for GCE confidential-VM firmware endorsements, `GCE-cc-tcb-root`
serial 1 (`src/gce_endorsement.rs`). The verifier pins this DER; the signer
certificate inside every endorsement must chain to it (RSASSA-PSS, SHA-256).

- Source: `https://pki.goog/cloud_integrity/GCE-cc-tcb-root_1.crt`, fetched
  2026-10-12. SHA-256 of the file:
  `e876bc6978bf4f3da445f98a0a82363c8c0bae5a1fc033c6df65846a6cb0f18c`.
- Re-fetch and compare:

  ```sh
  curl -fsSL -o /tmp/gce_root.crt https://pki.goog/cloud_integrity/GCE-cc-tcb-root_1.crt
  cmp /tmp/gce_root.crt x0ne-infer/tests/gce_tcb_root_1.der && echo identical
  ```

- The anchoring test: `tests/cloud_gce_endorsement.rs` verifies the vendored
  endorsement below against this root at a fixed clock, and refuses it once
  one signature byte is changed.

## `gce_endorsement_tdx_85aefc5d.binarypb` (5308 bytes)

One real, public GCE launch endorsement (`VMLaunchEndorsement`, protobuf from
`google/gce-tcb-verifier`): the TDX firmware whose MRTD is
`85aefc5d443857c9df7fdcb2a52819268aaa5f094a40601fad6d525d18651c005fe23f4efb9c10d02e3fa27edf147b7b`.
Signed by `GCE-uefi-signer` serial 33 (valid 2026-08-04 to 2031-08-04) under
the root above; the TDX section carries `svn 2` and that single MRTD.

- Source:
  `https://storage.googleapis.com/gce_tcb_integrity/ovmf_x64_csm/tdx/85aefc5d443857c9df7fdcb2a52819268aaa5f094a40601fad6d525d18651c005fe23f4efb9c10d02e3fa27edf147b7b.binarypb`
  (public bucket, no credentials), object created 2026-10-05, fetched
  2026-10-12. SHA-256 of the file:
  `addd7155dbfe2d68192e8467630a4a753e90a31501f0d8c753cfee444bbd94cb`.
- Re-fetch and compare:

  ```sh
  curl -fsSL -o /tmp/e.binarypb \
    https://storage.googleapis.com/gce_tcb_integrity/ovmf_x64_csm/tdx/85aefc5d443857c9df7fdcb2a52819268aaa5f094a40601fad6d525d18651c005fe23f4efb9c10d02e3fa27edf147b7b.binarypb
  cmp /tmp/e.binarypb x0ne-infer/tests/gce_endorsement_tdx_85aefc5d.binarypb && echo identical
  ```

- Used by the simulated end-to-end tests as the endorsement a fake bucket
  serves; the fake TD's synthetic quote carries this MRTD.

## `tlog/go/` — a transparency log made by Go (test data, not trust material)

A log of 600 entries, its checkpoints at sizes 301 and 600 (signed by a log
key, cosigned by a witness key, both derived from public strings), the tiles
of both sizes in the C2SP `tlog-tiles` layout, an inclusion proof of
`reference-300.json` (`tlog-proof@v1`) and the consistency proof from 301
to 600. Made on 2026-10-22 by `tlog/go-src/main.go` with Go 1.25.14,
golang.org/x/mod v0.36.0 (`sumdb/note`, `sumdb/tlog`),
github.com/transparency-dev/formats v0.1.1 and merkle v0.0.2 (`go.sum`).
`tests/tlog_interop.rs` reads them with `tlog.rs` and `transparency.rs`.
The cosignatures carry the time they were made, so a rerun gives other
cosignature bytes; everything else is the same.

```sh
cd x0ne-infer/tests/tlog/go-src
go run . make ../go       # writes the files above
go run . verify <dir>     # checks a log made by transparency::test_log
```

## `tlog/tessera/` — a transparency log made by Tessera (test data, not trust material)

A log of 300 entries in this crate's entry format
(`x0ne-engine/image-reference/v1`), written by Tessera v1.0.4
(github.com/transparency-dev/tessera, the tile-based log library Rekor v2 is
built on) with its POSIX storage driver. Its checkpoints at sizes 100 and
300 are signed by the log key through Tessera and cosigned (cosignature/v1)
with github.com/transparency-dev/formats at the version Tessera pins
(v0.0.0-20251017110053-404c0d5b696c); both keys are derived from public
strings. Kept: the two checkpoints, the four tiles the proofs need
(`log/tile/...`, C2SP `tlog-tiles` layout), the inclusion proof of
`r-042.json` at size 100 (`tlog-proof@v1`, built with Tessera's
`client.ProofBuilder`) and the consistency proof from 100 to 300. Made on
2026-10-22 by `tlog/tessera-src/main.go` with Go 1.25.14, offline once the
modules were fetched; `tests/tlog_interop.rs` reads them with `tlog.rs` and
`transparency.rs`. No network log, no witness server: a local directory.

```sh
cd x0ne-infer/tests/tlog/tessera-src
go run . make <dir>   # <dir>/store is the Tessera log; copy the files above
```

## `gce_endorsement_tdx_c1ee9c16.binarypb` (5308 bytes)

Google's launch endorsement of the TDX firmware the GCE `a3-highgpu-1g` boots
of 2026-10-14, -17 and -18 ran: MRTD
`c1ee9c16e3afc506cfe042c5b846a368528f3b37618eafb27469bc114cf914e9222c91618470e7f2b28ac360968270a5`
(`ram_gib` 0), `svn 2`, firmware binary SHA-384
`53c558fe06714203d7428484c46ee7c0b9df748a9789e173421c28d097115b30efb8476304617532488ab40c3122db88`,
timestamp 2026-05-21. Signed by `GCE-uefi-signer` serial 30 (valid
2026-05-05 to 2031-05-05) under `GCE-cc-tcb-root` serial 1.

- Source:
  `https://storage.googleapis.com/gce_tcb_integrity/ovmf_x64_csm/tdx/c1ee9c16e3afc506cfe042c5b846a368528f3b37618eafb27469bc114cf914e9222c91618470e7f2b28ac360968270a5.binarypb`
  (public bucket, no credentials), object last modified 2026-06-17, fetched
  2026-10-22. SHA-256 of the file:
  `9bb40f84dd373261baba826c05a3fa2f3f88471f713bdebb2f537764d1c96471`.
- The anchoring test: `gce_endorsement::tests::the_endorsement_of_the_real_boots_names_its_binary`
  verifies it under the pinned root and pins the MRTD, the binary's SHA-384
  and the timestamp.
- The binary itself (2 MiB, Google's) is not vendored. Fetch it by that
  SHA-384 and run `tests/gce_firmware_live.rs` (`--ignored`): it must
  measure the MRTD above, and its configuration volume
  (`4e1221cf6f7815beec39a994bff99142ed305210e2735b1134fc8ef363923cbdcca91fa71e2e25ee8e689fb7d2c075bf`)
  must be the RTMR0 event of the three logs in `ccel/`. Passed on
  2026-10-22.

## The synthetic TDVF image (16384 bytes, not a file)

Not a firmware: a 16 KiB image in Google's TDVF layout (a CFV page, three
BFV pages marked EXTEND_MR, a TD HOB and temporary memory), built by
`tdvf::tests::synthetic()` and written byte for byte the same by
`tdvf/tdvf_reference.py make`. `tdvf.rs` tests parse and measure it, and our
own test Google (`gce_endorsement::tests::fake_google`) endorses it for the
simulated end-to-end tests. SHA-256
`05ae51dcd06ca931c0ac74f19b63fce60aa64e12e488499bed699ab7e3125ef3`.

- Write it with the independent implementation, then measure it:

  ```sh
  python3 x0ne-infer/tests/tdvf/tdvf_reference.py make /tmp/s.fd
  python3 x0ne-infer/tests/tdvf/tdvf_reference.py measure /tmp/s.fd
  ```

- The anchoring test: `tdvf::tests::the_synthetic_image_measures_as_the_independent_implementation_says`
  pins the SHA-384 the script prints for its image, and its MRTD and CFV
  digest: the image built in Rust is the script's.

## `snp/amd_{milan,genoa,turin}_{ark,ask}.der` (AMD SEV-SNP roots)

AMD's root (ARK) and signing (ASK) keys for the three SEV-SNP product lines
the verifier knows (`src/snp.rs`, `SnpAnchors::production()`). Each ARK is
self-signed and signs its ASK with RSASSA-PSS SHA-384, salt 48; each ASK signs
the chip's VCEK. The verifier pins the SHA-256 of each certificate's SPKI
(`AMD_ARK_*_SPKI_SHA256_HEX`, `AMD_ASK_*_SPKI_SHA256_HEX`) and compiles the
DER in.

- Source: AMD's Key Distribution Service,
  `https://kdsintf.amd.com/vcek/v1/{Milan,Genoa,Turin}/cert_chain` (a PEM
  file, ASK first, then ARK), fetched 2026-10-22. Byte-equal to the
  `src/certs/snp/builtin/{milan,genoa,turin}/{ark,ask}.pem` of
  `virtee/sev` at commit `a966d06d9bf0e68afe2733e7bfbf278fd7fd65b5`.
- SHA-256 of the files: ARK Milan `69d063b4…9a540bcd`, Genoa
  `4c6598d1…103db2f1`, Turin `1f084161…73dd3f6a`; ASK Milan
  `67d303bd…8c0bac1b`, Genoa `5464738c…6e828521`, Turin `5b77ef5f…ae3b3185`.
- Re-fetch and compare:

  ```sh
  for p in Milan Genoa Turin; do
    l=$(echo "$p" | tr A-Z a-z)
    curl -fsSL -o "/tmp/$p.pem" "https://kdsintf.amd.com/vcek/v1/$p/cert_chain"
    csplit -s -z -f "/tmp/$p." "/tmp/$p.pem" '/-----BEGIN CERTIFICATE-----/' '{*}'
    openssl x509 -in "/tmp/$p.00" -outform der | cmp - "x0ne-infer/tests/snp/amd_${l}_ask.der" && echo "$p ASK identical"
    openssl x509 -in "/tmp/$p.01" -outform der | cmp - "x0ne-infer/tests/snp/amd_${l}_ark.der" && echo "$p ARK identical"
  done
  ```

- The anchoring test: `snp::tests::the_production_pins_match_amds_kds_certificates`
  digests each SPKI with the production code path, compares it to its pin,
  and verifies each ARK→ASK link at a fixed clock.
- Milan is pinned like the others and refused by default (`SnpPolicy::allow_milan`):
  the chain is AMD's either way; what is in doubt is the chip (README,
  THREAT_MODEL "AMD SEV-SNP").

## `snp/amd_{milan,genoa,turin}_asvk.der` (AMD SEV-SNP VLEK signing keys)

AMD's ASVK for each product: the key, signed by the product's ARK, that
signs the VLEKs AMD issues to cloud providers. The verifier pins the SHA-256
of each SPKI (`AMD_ASVK_*_SPKI_SHA256_HEX`) and compiles the DER in.

- Source: AMD KDS, `https://kdsintf.amd.com/vlek/v1/{Milan,Genoa,Turin}/cert_chain`
  (PEM, ASVK first, then the ARK), fetched 2026-10-28 02:51 UTC
  (`Last-Modified: Thu, 20 Nov 2025 07:44:46 GMT` for Genoa). The ARK in each
  file is byte-equal to `amd_<product>_ark.der`.
- Serials: Milan `01:01:01`, Genoa `02:01:01`, Turin `03:01:01`; windows
  2022-11-16 (Milan), 2022-11-18 (Genoa), 2023-05-15 (Turin), 25 years each.
- SHA-256 of the files: Milan `c5e081f5…439082f5`, Genoa
  `197e6107…51457372`, Turin `104e10a8…f8df1a3c`; the test
  `the_production_pins_match_amds_kds_certificates` checks each prefix and
  each pin, and verifies each ARK→ASVK link.
- Re-fetch and compare as the ARK/ASK above, with `vlek` for `vcek` in the URL
  and `amd_${l}_asvk.der` for the first certificate.

## `snp/amd_{milan,genoa,turin}_crl_2026-10-27.der` (AMD SEV-SNP CRLs)

AMD's certificate revocation list for each product, as KDS served it: issued
and signed by the product's ARK (RSASSA-PSS SHA-384, salt 48), CRL v2, with a
CRL number and an authority key identifier, neither critical.

- Source: `https://kdsintf.amd.com/vcek/v1/{Milan,Genoa,Turin}/crl` and
  `https://kdsintf.amd.com/vlek/v1/{Milan,Genoa,Turin}/crl`, fetched
  2026-10-28 02:50-02:51 UTC. The `vcek` and `vlek` files are byte-equal per
  product. `Last-Modified: Tue, 27 Oct 2026 06:22:47 GMT` (Genoa).
- thisUpdate 2026-10-27, nextUpdate 2026-12-09 01:00 UTC. Genoa's lists
  serial `02:00:01` (revoked 2022-10-31): an earlier Genoa ASK, not the pinned
  one (`02:00:02`).
- SHA-256: Milan `c06e8684…`, Genoa `d803150f…`, Turin `949c93ca…` (checked
  by `the_production_pins_match_amds_kds_certificates`).
- The tests: `snp::tests::amds_crls_verify_under_the_pinned_arks_and_name_the_revoked_genoa_ask`,
  and every `production(…)` verdict on a real report reads them at
  2026-10-28 00:00 UTC.
- Re-fetch: `curl -fsSL -o crl.der https://kdsintf.amd.com/vcek/v1/Genoa/crl;
  openssl crl -inform der -in crl.der -noout -text`. A newer CRL replaces the
  file under a new date and the test clock moves with it.

## `snp/virtee_*`, `snp/trustee_*` (real SEV-SNP reports and VCEKs)

Public test data of two open-source verifiers, both Apache-2.0, copied
unchanged except where an extraction is stated.

| File | Upstream path | Commit |
|---|---|---|
| `virtee_milan_report.bin` (1184 bytes) | `tests/certs_data/report_milan.hex`, hex-decoded | `virtee/sev` `a966d06d9bf0e68afe2733e7bfbf278fd7fd65b5` |
| `virtee_milan_vcek.der` | `tests/certs_data/vcek_milan.der` | same |
| `virtee_turin_vcek.der` | `tests/certs_data/vcek_turin.der` | same |
| `trustee_azure_genoa_report.bin` (1184 bytes) | `deps/verifier/test_data/az-snp-vtpm/evidence-genoa.json`, bytes `0x20..0x4C0` of `report` (the SNP report inside Azure's HCL report) | `confidential-containers/trustee` `7001db235568b4b989c035b6a53015fbd3b33b7f` |
| `trustee_azure_genoa_vcek.der` | `deps/verifier/test_data/az-snp-vtpm/vcek-genoa.pem`, as DER | same |
| `trustee_aws_vlek_report.bin` | `deps/verifier/test_data/snp/test-vlek-report.bin` | same |
| `trustee_aws_vlek.der` | `deps/verifier/test_data/snp/test-vlek.der` | same |

- What they exercise: the Azure Genoa report verifies under the pinned Genoa
  root and AMD's Genoa CRL, is refused by default for SMT (its guest policy
  accepts it), then under the default minimum TCB (its SNP SPL is `0x17`,
  below AMD-SB-3020's `0x1B`), and is accepted when SMT and the AMD-SB-3019
  levels are given explicitly; the Milan report is refused by default and
  verifies once Milan is allowed with its own minimum; the Turin VCEK chains
  to the pinned Turin root; the AWS VLEK (issued by `SEV-VLEK-Milan`, cspId
  `CN=cc-us-east-2.amazonaws.com`, valid 2024-12-10 to 2025-12-10) chains to
  the pinned Milan ASVK and signs its report, which is refused until the
  policy names AWS, then for its VMPL 1.
- Re-extract the Azure report (the other files compare with `cmp`):

  ```sh
  python3 -c 'import json,sys; sys.stdout.buffer.write(bytes(json.load(open(sys.argv[1]))["report"])[0x20:0x4C0])' \
    evidence-genoa.json | cmp - x0ne-infer/tests/snp/trustee_azure_genoa_report.bin && echo identical
  ```

- The tests: `snp::tests::a_real_genoa_report_verifies_under_amds_roots_and_crl_and_is_held_to_the_default_minimum`,
  `a_real_milan_report_is_refused_by_default_and_verifies_only_when_allowed`,
  `a_real_turin_vcek_chains_to_the_pinned_turin_root`,
  `a_real_aws_vlek_chains_to_amds_asvk_and_its_vmpl1_report_is_refused`.


## `snp/scaleway_siena_report_2026-10-28.bin` and `snp/scaleway_siena_vcek_2026-10-28.der`

The guest report of x0ne's Scaleway Elastic Metal run of 2026-10-28 (run
`1028-110059` of the SEV-SNP kit, `e2e-cloud/scaleway-snp/`, not published)
and the VCEK AMD KDS served for it during that run. No secret: a report and
a public certificate. The guest's `REPORT_DATA` is the kit's nonce.

| File | Run file | Bytes | SHA-256 |
|---|---|---|---|
| `scaleway_siena_report_2026-10-28.bin` | `verify/report.bin` | 1184 | `700b6a628df8245811660759928e7d1dbf072726d60287411c8777c464b83fa9` |
| `scaleway_siena_vcek_2026-10-28.der` | `verify/certs/vcek.der` (byte for byte `verify/kds/vcek.der`) | 1347 | `4059c8d324747608658f888488eafdeb586b46db7726f7f8fcbbd9e9eab98079` |

- What it is: report version 3, CPUID family 0x19 model 0xA0 stepping 2
  (Bergamo or Siena), guest `POLICY` 0x30000 (SMT allowed), `PLATFORM_INFO`
  0x4 (alias check not complete), reported, current, committed and launch
  TCB all boot loader 0x09, TEE 0x00, SNP 0x17, microcode 0x15.
- The test: `snp::tests::the_scaleway_host_of_2026_10_28_under_both_tcb_policies`
  (AMD's Genoa chain and CRL of 2026-10-27; refused by the default policy
  for SMT under both TCB modes; with SMT allowed, below AMD-SB-3020 under
  both).

## `receipts/receipt_before_tcb_disclosure.*` (test data, not trust material)

A receipt signed by `receipt::build_signed` on 2026-10-28, before the
`tcb` field existed, with a test key, and the exact bytes it signed.
`receipt::tests::a_receipt_from_before_the_tcb_disclosure_still_verifies`
parses it with this build and checks the signature over those bytes.

## `snp/test_*.der` (TEST chains, not AMD)

Written by `x0ne-infer/scripts/make-snp-fixtures.py` on 2026-10-22, re-run
on 2026-10-28 (`cryptography` 43.0.0, image `localhost/x0ne-vm-tools:1`):
two TEST chains shaped like AMD's (Genoa and Turin), VCEKs at and below the
default minimum TCB, a VCEK signed by an ASK of the same name under another
ARK; since 2026-10-28 also a Genoa ASVK (serial 3) and a VLEK (scalar
`0x7C`, cspId `CN=test-csp.x0ne.invalid`), and TEST CRLs signed by the TEST
ARKs (window 2026-01-27 to 2026-03-10): a good one listing serial `0x0F`,
one listing the ASK (2), one the ASVK (3), an expired one, one signed by the
rogue ARK of the same name, one with an unknown critical extension, and an
empty Turin one. The VCEK private keys are fixed public scalars written in the
script and in `src/snp.rs`'s tests, so the tests can sign reports AMD never
would (debug, migration agent, low TCB, another session). The ARK and ASK
keys are random and discarded: a re-run writes other files, and nothing
outside `#[cfg(test)]` can reach them (`tests/cloud_source_sweep.rs`,
`snp_anchors_are_built_by_production_alone`).
