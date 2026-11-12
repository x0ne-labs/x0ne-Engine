# Fuzzing the parsers

Two layers. The first exists and runs in `make ci`. The second is a proposal:
it needs a dependency this workspace does not have, which is a decision for
the maintainer.

## What runs today: the hostile-bytes suites

Seeded, deterministic mutation of the real fixtures, as ordinary `cargo test`,
with no dependency beyond the workspace's.

| Suite | Formats |
|---|---|
| `x0ne-infer/src/hostile_bytes_tests.rs` | TD quote layout, PCK chain, Intel collateral and CRLs, SNP report and evidence, CC event log, TDVF metadata, OVMF footer table and SEV metadata, Google's launch endorsement and its signer certificate, OpenPGP keys and signatures, OCSP responses and a delegated responder's certificate, NVIDIA's NRAS signing leaves, the bounds of an OVMF descriptor, sealed blobs, HPKE messages, the SNP certificate table |
| `x0ne-infer/src/hostile_text_tests.rs` | signed notes, checkpoints, log proofs, Merkle proofs, NVIDIA's JWT, JWKS and claims, published JSON documents (references, measurements, model configurations, receipts), JSON schemas, chat requests, relay control lines, SSE streams |
| `x0ne-sdk/tests/hostile_bytes.rs` | the whole verdict of `attest::verify_quote` on a real quote, event log and collateral; OHTTP key configurations, sealed requests and chunked answers; binary HTTP |
| `x0ne-ohttp/tests/hostile_bytes.rs` | `application/ohttp-keys`, pinned key configurations, node sets, inner messages |
| `x0ne-engine-cloud/tests/hostile_bytes.rs` | the DNS answer of `myaddr`, the serial console, relay event lines, the provider's price list, durations and keys of the CLI |

The harness is `x0ne-sdk/tests/support/hostile.rs` (bytes and text) and
`hostile_json.rs` (JSON values). Each case is held to:

- no panic (caught, reported with the mutation that caused it), no hang (a
  watchdog aborts the process after 120 s on one case and reports cases over
  20 s), no single allocation beyond `max(64 x input, 1 MiB)` (a counting
  global allocator, per thread);
- a truncation is refused, for every prefix length where the format allows
  the statement, or what a cut input still yields is stated in the test;
- a signed object whose signed bytes changed does not verify; a mutated input
  that does verify gives the verdict of the intact one, field for field;
- what parses and has a serializer reads back the same.

A failing campaign lists one example per kind of finding, then panics.

### Running deeper

```sh
X0NE_FUZZ_ITERS=100 cargo test --workspace hostile -- --nocapture
X0NE_FUZZ_SEED=12345 X0NE_FUZZ_ITERS=100 cargo test -p x0ne-infer --lib hostile
```

`X0NE_FUZZ_ITERS` multiplies every sampled count; from 10 the sampled
exhaustive passes (one bit per signed byte, one length in N) become complete.
`X0NE_FUZZ_SEED` changes the sequence; the default is fixed, so `make ci`
runs the same cases every time. Each campaign prints its counts on standard
error. `make ci` does not forward these variables: a deep run uses the test
binaries directly, in a container without network.

### Limits

Mutation without coverage feedback reaches the first refusal of a parser
quickly and its deep branches slowly. A checksum or a signature in front of a
parser stops nearly every mutant there: the suites call the inner parsers
directly where they are reachable, and say where they are not. Nothing here
measures which lines the mutants reached; `make coverage` does, per crate.

## Proposal: coverage-guided targets

`cargo fuzz` (libFuzzer, through the `libfuzzer-sys` crate) would drive the
same entry points with coverage feedback. Not done here because it adds:

- a crate to resolve and vendor (`libfuzzer-sys`, and `arbitrary` for
  structured inputs), in a `fuzz/Cargo.toml` outside the workspace so that
  the workspace's lock file and `cargo deny` scope do not change;
- a nightly toolchain for `-Z sanitizer` and `-C passes=sancov-module`, next
  to the pinned stable one of `rust-toolchain.toml`;
- a corpus directory per target, to keep or to regenerate.

### Layout

```
fuzz/
  Cargo.toml            # [workspace] of its own, publish = false
  fuzz_targets/
    tdx_quote.rs
    pcs_chain.rs
    ...
  corpus/<target>/      # seeded from x0ne-infer/tests/
```

### Targets and entry points

Each target is one function call on the fuzzer's bytes. The seeds are the
fixtures the hostile-bytes suites already use.

| Target | Entry point | Seeds |
|---|---|---|
| `tdx_quote` | `x0ne_confidential::tdx_quote::parse` | `tests/pcs_live/quote.dat`, `tests/run_fixtures/*.quote` |
| `pcs_chain` | `x0ne_infer::pcs::verify_tdx_chain` | the same |
| `tcb_collateral` | `x0ne_infer::tcb::verify_live` (input split in four) | `tests/pcs_live/2026-10-22/*.json`, `*_issuer_chain.pem` |
| `crl` | `x0ne_infer::tcb::verify_pck_crl`, `verify_root_crl` | `tests/pcs_live/**/*.der` |
| `snp_report` | `x0ne_confidential::snp_report::parse`, `x0ne_infer::snp::verify_snp_report` | `tests/snp/` |
| `ccel` | `x0ne_infer::ccel::parse_with_end`, then `replay` and `encode` | `tests/ccel/*.bin` |
| `tdvf` | `x0ne_infer::tdvf::Image::parse`, then `mrtd` | a real TDVF image, and the synthetic one of `tdvf.rs` |
| `ovmf` | `x0ne_infer::snp_launch::launch_digest` | a real OVMF build |
| `gce_endorsement` | `x0ne_infer::gce_endorsement::verify` | `tests/gce_endorsement_tdx_*.binarypb` |
| `openpgp_cert` | `x0ne_infer::openpgp::Cert::parse` | `tests/archive_key/*.gpg` |
| `openpgp_sig` | `Cert::verify_detached` under the test keyring | `tests/archive_key/*.sig` |
| `ocsp` | `x0ne_infer::nras_ocsp::verify_response` | responses of the test responder, written to the corpus by a test |
| `nras_token` | `x0ne_infer::nras::verify_nras_token` | a token of the test issuer |
| `tlog_note` | `x0ne_infer::tlog::parse_note`, `parse_checkpoint`, `parse_tlog_proof` | `tests/tlog/go/`, `tests/tlog/tessera/` |
| `transparency` | `x0ne_infer::transparency::check_offline` | the same |
| `reference` | `x0ne_infer::reference::parse`, `snp_reference::parse`, `measurements::parse` | `tests/reference/`, `tests/measurements/` |
| `model_config` | `x0ne_infer::arch::DecoderConfig::parse`, `model_spec::parse_qwen3_config` | `tests/arch/real/` |
| `json_schema` | `JsonSchema::compile`, then `accepts` on the rest of the input | schemas of the unit tests |
| `chat_request` | `x0ne_infer::openai::request::parse_chat_request` | requests of the API tests |
| `sse` | `x0ne_infer::sse::SseReader::push`, the input cut at a fuzzed offset | a recorded stream |
| `ohttp_request` | `x0ne_sdk::ohttp::decapsulate_request`, `decapsulate_chunked_request` | the vectors of `tests/chunked_ohttp.rs` |
| `ohttp_key_config` | `x0ne_sdk::ohttp::KeyConfig::parse` | the same |
| `bhttp` | `x0ne_sdk::bhttp::Request::decode`, `Response::decode`, `ResponseReader` | messages written by `encode` |
| `verify_quote` | `x0ne_sdk::attest::verify_quote` (structured input: quote, log, collateral) | the run of 2026-10-18 |
| `dns_answer` | `x0ne_engine_cloud::myaddr::parse_answer` | an answer built as in `myaddr.rs` |
| `serial_console` | `x0ne_engine_cloud::td_console::Watch::feed` | `x0ne-engine-cloud/tests/serial/` |
| `relay_event` | `x0ne_engine_cloud::proxy_link::parse_event`, `x0ne_infer::relay::parse_cmd` | lines of the unit tests |

A target is a few lines:

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = x0ne_infer::ccel::parse_with_end(data);
});
```

The oracles of the hostile-bytes suites carry over as assertions inside a
target: the round trip of a parsed log, the equality of a stream read in one
piece and in two, the verdict of a mutated signed object against the intact
one. `pcs::tests`, `nras::tests` and `nras_ocsp::tests` build their signed
fixtures inside the crate's test modules; a target outside the crate needs
them written to the corpus once, by a test or an example.

### What a run would need

- `cargo +nightly fuzz run <target> -- -max_total_time=<s> -rss_limit_mb=4096`
  in the CI image, without network, the vendored crates of `fuzz/Cargo.lock`
  mounted;
- a decision on where corpora live (in the tree, or in the CI cache) and on
  how long each target runs, per commit or nightly;
- a crash found becomes a fixed-input test in the crate it concerns, as for
  the hostile-bytes suites, before its fix.
