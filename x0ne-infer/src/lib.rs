//! x0ne-infer — x0ne Engine's confidential-native inference server.
//!
//! The value is the boundary, not the matmul: **no proof, no socket** — the
//! listener can only be created from an [`gate::Attested`] value, and that
//! value can only be produced by [`gate::Gate::attest`] from a fresh,
//! cryptographic [`x0ne_confidential::model::Verdict::Verified`]. The model
//! key is released under the same rule (`keys`), and every answer carries a
//! receipt naming the environment it ran in (`receipt`).
//!
//! **What is proven, and what is not.** The gate, the key path, the evidence
//! source (`configfs-tsm`), the cryptographic verifiers (full Intel DCAP
//! chain to a pinned root, live Intel PCS TCB and QE identity, NVIDIA NRAS
//! under a pinned intermediate), the weight envelope, the x0ne Core backend
//! and the HTTP surface all exist and are tested, and the composite ran on real
//! hardware (GCP a3-highgpu-1g, TDX + H100 in CC mode): the first run recorded
//! here is `RUN_2026-10-14_beta2.md` (2026-10-14); earlier runs are not recorded.
//!
//! Which IMAGE is running is established only if somebody pins it:
//! `mrtd` and `rtmr0..3` are in every verdict and every signed receipt, the
//! server enforces `--expect-measurement`, and — the half an operator cannot
//! switch off — a client enforces its own against a quote it verified itself
//! ([`client`]). With nothing pinned, a verified quote proves genuine
//! confidential hardware answering this session's challenge, not that the
//! binary inside it is the one you audited. Every verdict carries its own
//! limitations verbatim (`verifier` string); read them rather than this
//! paragraph, which can drift.

/// The request table of the OpenAI surface, played against the four routers.
#[cfg(test)]
mod api_contract_tests;
pub mod arch;
/// The x0ne archive key the image references are signed by, pinned.
pub mod archive_key;
/// Unknown command-line arguments named without their value, the lifecycle's own file.
#[path = "../../x0ne-engine-cloud/src/argv.rs"]
pub mod argv;
/// The TD side of the attested channel: evidence, verdict, attestation document.
pub mod attest_v2;
pub mod backend;
pub mod backends;
pub mod binding;
/// candle on the CPU, the reference of the tests (`candle-reference`).
#[cfg(feature = "candle-reference")]
pub mod candle_backend;
/// candle's reader over checked weight files (`shards::Routing`).
#[cfg(feature = "candle-reference")]
pub mod candle_shards;
/// TCG2 event log (CCEL) parse and RTMR replay.
pub mod ccel;
/// The session key brought by the client over the attested channel.
pub mod channel_key;
pub mod client;
/// The cloud config, bytes hashed into the quote.
pub mod cloud_config;
/// Its types, compiled into x0ne-sdk too; re-exported by `cloud_config`.
mod cloud_config_types;
/// The simulated end to end and the refusal table.
#[cfg(test)]
mod cloud_e2e_tests;
/// The real-hardware run's measuring client; tests and `--features probe`.
#[cfg(any(test, feature = "probe"))]
pub mod cloud_probe;
/// `/v2/*` and the TLS server of cloud mode.
pub mod cloud_serve;
pub mod conformance;
/// The token mask of a `json_schema` answer, over a model's vocabulary.
pub mod constrain;
/// The context a turn of a conversation reused, as its receipt signs it.
pub mod context;
/// The one conversation a model keeps between its turns.
pub mod conversation;
/// x0ne Core as the backend `core`; Core itself is linked with `cuda`.
pub mod core_backend;
/// Token ids to streamed text, stop sequences held back: every backend's.
pub mod decode_stream;
/// The decoder behind the OLMo families (candle), for the tests.
#[cfg(feature = "candle-reference")]
pub mod decoder;
/// What a request asks of the decoding, and the check of a receipt against it.
pub mod decoding;
/// The TD's end of the caller-sealed path: `/v2/e2e/*`, its OHTTP gateway.
pub mod e2e_gateway;
/// The x0ne-sdk prototype against the simulated TD: attestation checked by
/// the caller, the question sealed to the TD's OHTTP key, the receipt checked.
#[cfg(test)]
mod e2e_sdk_tests;
/// The TD's chunked answer to a caller-sealed stream, at the session's cadence.
pub mod e2e_stream;
/// The TD's `/v2/e2e/*` routes through a recording carrier, on one host.
#[cfg(test)]
mod e2e_td_tests;
pub mod evidence;
/// Model and config files: regular files only, capped.
pub mod fsread;
pub mod gate;
/// Google launch endorsements: the MRTD signed by Google.
pub mod gce_endorsement;
/// The TD process: not dumpable, no core file; keys out of swap and dumps.
pub mod harden;
/// Hex decoding of untrusted input, re-exported by `cloud_config`.
mod hex;
/// HPKE (RFC 9180), base mode, one message: the model key's envelope to a TD.
pub mod hpke;
/// HTTP bodies read against a cap as they arrive.
pub mod http_body;
/// JSON constrained by a schema: the subset, its bounds, its digest, its automaton.
pub mod json_schema;
pub mod keys;
/// I6: the loopback OpenAI endpoint the relay serves.
pub mod local_api;
/// `x0ne-engine-proxy chat`: a terminal client of that endpoint.
pub mod local_chat;
/// Reference measurements published per version, pinned by default.
pub mod measurements;
/// Ministral 3 (text), on the Qwen3 fork's in-place KV cache (candle).
#[cfg(feature = "candle-reference")]
pub mod ministral3;
/// The served-model digest (v2): a sorted `sha256sum` manifest, hashed.
pub mod model_digest;
/// The model owner's key, released by the owner's relay to one attested TD.
pub mod model_key;
/// A model's metadata read without candle: artifacts, Qwen3 config, spec, served digest.
pub mod model_spec;
pub mod nras;
pub mod nras_client;
pub mod nras_http;
pub mod nras_ocsp;
#[cfg(feature = "nvml")]
pub mod nvml;
/// What the engine knows about a model's openness, by a digest of its files.
pub mod oblivious_tok;
/// The OpenAI-compatible contract the chat surfaces share.
pub mod openai;
pub mod openness;
/// Detached OpenPGP signatures (RFC 9580 v4, Ed25519 only), the archive key profile.
pub mod openpgp;
/// An answer's length rounded up to one of few sizes (Padme), for the host.
pub mod padding;
pub mod pcs;
/// Chat prompts: the template's control markers apart from the messages' text.
pub mod prompt;
/// Qwen3 on candle, forked for an in-place KV cache: the reference of the tests.
#[cfg(feature = "candle-reference")]
pub mod qwen3;
pub mod receipt;
/// The image reference the client pins.
pub mod reference;
pub mod registry;
/// The relay, driven over stdin/stdout by the lifecycle.
pub mod relay;
/// The client's verdict on a cloud TD, on the client's machine.
pub mod remote_verify;
/// The fixtures of a real boot, from its run's attestation document; tests
/// and `--features probe` (`examples/run_fixtures.rs`).
#[cfg(any(test, feature = "probe"))]
pub mod run_fixtures;
/// Token draws on the host, without candle: candle's draws for a seed.
pub mod sampler;
pub mod seal;
pub mod server;
/// The key lease on two clocks, judged when a computation starts.
pub mod serving_lease;
pub mod shards;
/// AMD SEV-SNP: the VCEK or VLEK chain to a pinned AMD root, AMD's CRL, the
/// report signature, the TCB floor, the guest policy. An SEV-SNP guest produces
/// its evidence through configfs-tsm (`evidence`); the relay judges it under an
/// SEV-SNP reference.
pub mod snp;
/// The client's verdict on an SEV-SNP guest, from the document to each connection.
#[cfg(test)]
mod snp_client_tests;
/// The SEV-SNP launch measurement of a QEMU guest, predicted offline.
pub mod snp_launch;
/// The written SEV-SNP policy a client pins (`x0ne-engine/snp-reference/v1`).
pub mod snp_reference;
/// The strict reader of server-sent events: the relay's, on a TD stream.
pub mod sse;
pub mod stepped;
/// A streamed answer's body: fixed-size frames at a fixed cadence.
pub mod stream;
pub mod tcb;
/// What a verdict says about the platform's patch level (2026-10-28).
pub mod tcb_disclosure;
/// The platform's standing under Intel's live verdict, re-read while serving.
pub mod tcb_standing;
/// Google's TDX firmware binary: its MRTD and the CFV it extends into RTMR0.
pub mod tdvf;
/// Terminal colors, the lifecycle's own file (`x0ne-engine-cloud/src/color.rs`):
/// the relay and its chat follow `NO_COLOR`, `CLICOLOR_FORCE`, `CLICOLOR` and `TERM` as it does.
#[path = "../../x0ne-engine-cloud/src/color.rs"]
pub mod term_color;
/// Transparency-log formats and proofs (C2SP checkpoints, notes, tiles, tlog-proof).
pub mod tlog;
/// Running that log: keys, appends, witness requests, cosignatures, proofs (`x0ne-engine-tlog`).
pub mod tlog_ops;
/// TLS 1.3 for the attested channel, both ends pinned by SPKI.
pub mod tls;
/// The log of image references this build pins, and its check.
pub mod transparency;
pub mod verify;
