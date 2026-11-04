# Licensing

x0ne Engine is under two licenses. x0ne holds the copyright of it, except
the third parties' work [NOTICE](NOTICE) names: `x0ne-infer/src/qwen3.rs`
is a modified copy of a file of candle-transformers (Hugging Face), and
`x0ne-infer/src/nvml.rs` copies struct layouts and constants of
nvidia-ml-py (NVIDIA, BSD-3-Clause).

- The server, and everything that is not listed below as Apache-2.0, is under
  the GNU Affero General Public License, version 3 only (AGPL-3.0-only), with
  an additional permission for NVIDIA's libraries. Both are in
  [LICENSE](LICENSE).
- What an application builds in is under the Apache License, Version 2.0
  (Apache-2.0): the crates `x0ne-sdk`, `x0ne-confidential` and `x0ne-ohttp`,
  each with its own `LICENSE`, and the files of x0ne-infer that `x0ne-sdk`
  compiles.

`x0ne-infer/tests/licensing.rs` holds this file to the manifests and to the
license files: the crate table below, each crate's `license` field, the
`LICENSE` files and the list of shared files say the same thing, and no
Apache-2.0 crate depends on an AGPL-3.0-only crate.

## The crates

| Crate | Directory | License | What it is |
|---|---|---|---|
| `x0ne-infer` | `x0ne-infer` | AGPL-3.0-only | the inference server, the `x0ne-engine` command and its relay |
| `x0ne-engine-cloud` | `x0ne-engine-cloud` | AGPL-3.0-only | the cloud lifecycle behind `x0ne-engine cloud` |
| `x0ne-core-kernels` | `x0ne-core/kernels` | AGPL-3.0-only | x0ne Core's CUDA kernels |
| `x0ne-core-runtime` | `x0ne-core/runtime` | AGPL-3.0-only | x0ne Core's runtime, the backend `core` of x0ne-infer |
| `x0ne-sdk` | `x0ne-sdk` | Apache-2.0 | the caller's side: checks the proof, seals requests |
| `x0ne-confidential` | `x0ne-confidential` | Apache-2.0 | the attestation vocabulary the crates share |
| `x0ne-ohttp` | `x0ne-ohttp` | Apache-2.0 | a prototype of Oblivious HTTP that no x0ne binary uses |

## Files of x0ne-infer compiled into x0ne-sdk

`x0ne-sdk` compiles these files of x0ne-infer as they are (`#[path]`, see
`x0ne-sdk/src/lib.rs`), so that the client and the server check a proof
with the same code. They are under Apache-2.0, as `x0ne-sdk` is: each opens
with `// SPDX-License-Identifier: Apache-2.0`.

- `x0ne-infer/src/archive_key.rs`
- `x0ne-infer/src/ccel.rs`
- `x0ne-infer/src/cloud_config_types.rs`
- `x0ne-infer/src/context.rs`
- `x0ne-infer/src/decoding.rs`
- `x0ne-infer/src/gce_endorsement.rs`
- `x0ne-infer/src/hex.rs`
- `x0ne-infer/src/json_schema.rs`
- `x0ne-infer/src/nras.rs`
- `x0ne-infer/src/nras_ocsp.rs`
- `x0ne-infer/src/openpgp.rs`
- `x0ne-infer/src/pcs.rs`
- `x0ne-infer/src/reference.rs`
- `x0ne-infer/src/tcb.rs`
- `x0ne-infer/src/tcb_disclosure.rs`
- `x0ne-infer/src/tdvf.rs`
- `x0ne-infer/src/tlog.rs`
- `x0ne-infer/src/transparency.rs`

The certificates these files embed (`x0ne-infer/tests/*.der`) are their
issuers' published material ([NOTICE](NOTICE)). The x0ne test data their
tests include, `x0ne-infer/tests/reference/x0ne-cvm-0.2*.json` and the TEST
keyring `x0ne-infer/tests/archive_key/test-archive-keyring.gpg`, are under
Apache-2.0 too; third parties' data they include keeps its own terms
([x0ne-infer/tests/VENDORED.md](x0ne-infer/tests/VENDORED.md)).

## Everything else

The scripts, `ci/`, `docs/`, the packaging and the tests' data are under
AGPL-3.0-only with the permission of [LICENSE](LICENSE), like the server.

One exception: the code samples of `README.md` and `docs/` (the fenced
blocks that call the API or use the SDK) may also be used under Apache-2.0,
so that an application can copy them.

Third parties' material keeps its own license: [NOTICE](NOTICE),
[THIRD_PARTY_LICENSES.html](THIRD_PARTY_LICENSES.html) for the Rust crates,
and [x0ne-infer/tests/VENDORED.md](x0ne-infer/tests/VENDORED.md) for the
vendored test data.

## NVIDIA's libraries

With the `cuda` feature the server is built with NVIDIA's CUDA libraries;
with `nvml` it loads `libnvidia-ml.so.1`, a library of the NVIDIA driver, at
run time. They are not free software. The end of
[LICENSE](LICENSE) is x0ne's additional permission, under section 7 of the
AGPL, to convey a work that combines the server with them. The libraries
themselves stay under NVIDIA's terms.

## Commercial license

x0ne also licenses the server under commercial terms, for whoever the AGPL
does not fit: a company that modifies the server and lets others use it over
a network without offering them the source of its changes (section 13 of the
AGPL), or that ships the server inside a product under other terms. Write to
contact@x0ne.co.

`x0ne-sdk`, `x0ne-confidential` and `x0ne-ohttp` need no commercial license:
they are under Apache-2.0.

## Contributions

External pull requests are accepted once the Contributor License Agreement
is signed. It is not published here yet: until it is, no external pull
request is merged. Issues are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).
