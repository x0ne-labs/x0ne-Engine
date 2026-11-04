//! x0ne SDK, prototype: the question is encrypted on the caller's side, to a
//! key the caller checked was made inside an attested TD.
//!
//! The path, in order (`e2e.rs`):
//!
//! 1. the caller draws a nonce; the TD answers a [`attest::KeyDocument`]: its
//!    Oblivious HTTP key configuration (RFC 9458 §3), the Ed25519 key that
//!    signs its receipts, the session config it serves under and its boot
//!    binding ([`e2e::Boot`]), and a TDX quote whose `report_data` is
//!    [`e2e::key_binding`] over them and the caller's nonce;
//! 2. [`attest::verify_key`] checks, here, as the relay does: the quote's
//!    chain to the pinned Intel root (`pcs.rs`, DEBUG refused), the binding,
//!    the live TCB status, the PCK CRL and the root CA CRL from collateral
//!    anyone may relay since Intel signs it (`tcb.rs`), the MRTD against
//!    Google's signed launch endorsement (`gce_endorsement.rs`), and the TD's
//!    event log replayed into RTMR0..3 against an image reference signed by
//!    x0ne's archive key and shown in x0ne's transparency log, admitted as
//!    the relay admits it (`reference.rs`, `openpgp.rs`, `transparency.rs`);
//!    pins on MRTD and RTMR1..3, optional; the session config the quote
//!    bound (the model, its end, the relay's TLS key) is returned to the
//!    caller, who reads it before sealing;
//!    for a TD that attests GPUs (`attest=cpu+gpu`), the quote also binds
//!    each GPU's report and certificate chain, and NVIDIA's NRAS verdict on
//!    them, carried in the key document, is checked as the relay checks it
//!    (`nras.rs`); the Intel TCB verdict is the relay's, `transparent` by
//!    default, `strict` on request, disclosed in the same
//!    `x0ne-infer/tcb-disclosure/v1` statement (`tcb_disclosure.rs`);
//! 3. [`e2e::Client`] seals a Binary HTTP request (RFC 9292), padded to a
//!    size bucket, as an Encapsulated Request (RFC 9458, HPKE RFC 9180 base
//!    mode, DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, AES-128-GCM), and opens
//!    the Encapsulated Response, which only the holder of that request's
//!    context can read.
//!
//! Whoever carries the bytes (the relay, a gateway, x0ne) sees sizes in
//! buckets and timing, not the text. What this does not establish is in
//! [`LIMITATIONS`], returned with every verified key.
//!
//! `pcs.rs`, `tcb.rs`, `tcb_disclosure.rs`, `gce_endorsement.rs`, `ccel.rs`,
//! `reference.rs`, `openpgp.rs`, `archive_key.rs`, `tlog.rs`,
//! `transparency.rs`, `tdvf.rs`, `nras.rs` and `cloud_config_types.rs` are
//! x0ne-infer's files,
//! compiled here as they are (`#[path]`):
//! one verifier, two builds. Nothing in this crate opens a
//! socket, reads a clock or keeps a file: the caller passes `now` and the
//! collateral, and keeps the log's last verified checkpoint
//! ([`attest::LogMemory`]) between calls, so
//! the same code runs in a browser (wasm32-unknown-unknown, [`abi`]).
//! `decoding.rs` and `json_schema.rs` are x0ne-infer's too: a sealed answer's
//! decoding is checked by the automaton that held it (`check_answer_for`).
//!
//! Those files of x0ne-infer are under Apache-2.0, as this crate is: each
//! opens with its SPDX line, and `LICENSING.md` lists them.

#![deny(unsafe_code)]

pub mod abi;
pub mod attest;
pub mod bhttp;
pub mod e2e;
pub mod hpke;
pub mod ohttp;
pub mod receipt;
pub mod sealed_stream;

/// x0ne's archive key, pinned: the signer of every image reference
/// (x0ne-infer's file).
#[path = "../../x0ne-infer/src/archive_key.rs"]
#[allow(dead_code)]
mod archive_key;
/// The CC event log: parse, replay into RTMR0..3 (x0ne-infer's file).
#[path = "../../x0ne-infer/src/ccel.rs"]
#[allow(dead_code)]
mod ccel;
/// The checked hex decoder `tcb.rs` calls as `crate::cloud_config::hex_bytes`
/// (x0ne-infer's `hex.rs`, the file `cloud_config` re-exports).
#[path = "../../x0ne-infer/src/hex.rs"]
#[allow(dead_code)]
mod cloud_config;
/// The TD's session config (`x0ne-engine/cloud-config/v1`), its types only:
/// a key document carries its bytes (x0ne-infer's file).
#[path = "../../x0ne-infer/src/cloud_config_types.rs"]
#[allow(dead_code)]
mod cloud_config_types;
/// Google's signed launch endorsement of the TD firmware (x0ne-infer's file).
#[path = "../../x0ne-infer/src/gce_endorsement.rs"]
#[allow(dead_code)]
mod gce_endorsement;
/// NVIDIA's NRAS token: signature under the pinned intermediates, claims,
/// per-GPU verdicts and identity (x0ne-infer's file).
#[path = "../../x0ne-infer/src/nras.rs"]
#[allow(dead_code)]
mod nras;
/// The NRAS signing chain's revocation, judged from the OCSP responses a key
/// document staples (x0ne-infer's file).
#[path = "../../x0ne-infer/src/nras_ocsp.rs"]
#[allow(dead_code)]
mod nras_ocsp;
pub use nras_ocsp::{NrasOcspMode, NrasOcspPolicy};
/// OpenPGP detached signatures under the archive key (x0ne-infer's file).
#[path = "../../x0ne-infer/src/openpgp.rs"]
#[allow(dead_code)]
mod openpgp;
/// The DCAP chain of a TD Quote v4 to a pinned root (x0ne-infer's file).
#[path = "../../x0ne-infer/src/pcs.rs"]
#[allow(dead_code)]
mod pcs;
/// The image reference and its event-log policy (x0ne-infer's file). Its
/// loading from disk is the relay's and is not compiled here; its admission
/// from bytes (`admit_offline`) runs the relay's signature and log steps.
#[path = "../../x0ne-infer/src/reference.rs"]
#[allow(dead_code, unused_imports)]
mod reference;
/// Intel PCS collateral: TCB info, QE identity, PCK CRL (x0ne-infer's file).
#[path = "../../x0ne-infer/src/tcb.rs"]
#[allow(dead_code)]
mod tcb;
/// What a verdict says of Intel's TCB, versioned (x0ne-infer's file).
#[path = "../../x0ne-infer/src/tcb_disclosure.rs"]
#[allow(dead_code)]
mod tcb_disclosure;
/// Google's firmware layout, named by `reference.rs` (x0ne-infer's file).
#[path = "../../x0ne-infer/src/tdvf.rs"]
#[allow(dead_code)]
mod tdvf;
/// Signed notes, checkpoints and inclusion proofs, C2SP (x0ne-infer's file).
#[path = "../../x0ne-infer/src/tlog.rs"]
#[allow(dead_code)]
mod tlog;
/// x0ne's transparency log, pinned, its check on a proof alone and on the
/// caller's memory of the log (x0ne-infer's file; its file and network
/// readers are not compiled here).
#[path = "../../x0ne-infer/src/transparency.rs"]
#[allow(dead_code, unused_imports)]
mod transparency;

/// The context a conversation's turn reused, checked against its receipt
/// (x0ne-infer's file).
#[path = "../../x0ne-infer/src/context.rs"]
pub mod context;
/// What a request asks of the decoding, checked against a receipt
/// (x0ne-infer's file).
#[path = "../../x0ne-infer/src/decoding.rs"]
pub mod decoding;
/// JSON constrained by a schema: its digest and automaton (x0ne-infer's file).
#[path = "../../x0ne-infer/src/json_schema.rs"]
pub mod json_schema;

pub use nras::{NrasIntermediatePin, NVIDIA_NRAS_GPU_INTERMEDIATE_PINS};
pub use pcs::{TrustAnchor, INTEL_SGX_ROOT_CA_SPKI_SHA256_HEX};
pub use tcb::TcbMode;
pub use tcb_disclosure::{TcbDisclosure, SCHEMA as TCB_DISCLOSURE_SCHEMA};

/// A pin over a bench's NVIDIA intermediate (its whole DER), for
/// [`attest::Policy::nras_pins`]: only in a build with `test-anchor`.
#[cfg(feature = "test-anchor")]
pub fn nras_test_pin(intermediate_der: &[u8]) -> Result<NrasIntermediatePin, String> {
    nras::throwaway_pin(intermediate_der)
        .ok_or_else(|| "not a certificate with an RSA or a P-384 key".into())
}

/// Exactly `2 * N` hex digits, either case: the one hex decoder of this crate.
/// Bytes are checked before slicing: a multibyte character at the right byte
/// length panicked before 2026-10-24, a trap of the wasm instance.
pub(crate) fn unhex<const N: usize>(s: &str) -> Result<[u8; N], String> {
    if s.len() != 2 * N || !s.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("{} bytes, expected {} hex digits", s.len(), 2 * N));
    }
    let mut out = [0u8; N];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// What a verified key does not establish. Returned with every
/// [`attest::VerifiedKey`]; a caller that shows a verdict shows these.
pub const LIMITATIONS: &[&str] = &[
    "prototype: the TD's e2e routes ran on Intel TDX hardware with one H100 (Google Cloud a3-highgpu-1g) on 2026-10-28, 2026-10-30, 2026-10-31 and 2026-11-01, one sealed question each time, asked whole and, on 2026-11-01, streamed as well, under the TEST archive key and TEST transparency log; every other test ran on synthetic TDX evidence",
    "the clock is the caller's: collateral windows, certificate validity, the reference's signature and its log checkpoint are judged against it, and it is not checked against the Date Intel's PCS answers with, as the relay checks its own",
    "the image reference is the caller's: whoever supplies it chooses the image accepted among those x0ne signed (its registers, its version, the name shown as image), so it must come from the integrator, never from the carrier; it is admitted only signed by the archive key this build pins and shown in the log this build pins under a fresh quorum of its witnesses, by the relay's own steps, and held to the caller's memory of the log: the last checkpoint verified is returned to keep and must be handed back, a checkpoint of that size must show the same root, a smaller one is refused, a larger one is admitted only with a consistency proof the caller supplies; no checkpoint is fetched here, so a fork is seen only once this caller has met both heads, and a memory lost or emptied starts over; a build made before the signing ceremony pins no key and admits no reference",
    "the TCB policy is the relay's: transparent by default, the platform accepted on Intel's standard collateral and Intel's early view of it named in the disclosure, not judged; strict accepts on early alone and holds every receipt to it; the disclosure (x0ne-infer/tcb-disclosure/v1) says which",
    "the collateral is relayed: its carrier cannot forge it, but can relay genuine collateral older than Intel's newest, TCB info, QE identity and PCK CRL up to a day old (max_collateral_age_secs shortens that), Intel's root CA CRL until it expires (2027-02-26 for the CRL of 2026-02-26); a TCB recovery or a revocation published since is not seen until it is relayed, where the relay fetches from Intel itself",
    "the model is the one whose digest the caller names: the relay checks the digest against its catalogue, nothing here checks where the caller's digest came from",
    "the session config the quote binds names the model by its catalogue name (session_config.model), shown to the caller before sealing; nothing here maps that name to the digest the caller pins",
    "Google's firmware: the binary Google signed is relayed in the collateral and measured here (its SHA-384 is the one the endorsement names, its MRTD the quote's), but its source is not published and its code is not audited; RTMR0 is pinned to the reference's host events (TD HOB, Secure Boot variables, ACPI tables, boot variables) as one boot showed them, not audited; Google's signing key can endorse any binary",
    "the carrier, the relay included, sees the size bucket of each question and answer, their timing, which TD they go to (and so the model it serves), the key document, and the caller's address; of a streamed answer, its count of chunks (a power of two at the session's cadence, so its duration within a factor of two) and a reader that stops reading",
    "the TD's OHTTP key is per boot and the same for every caller of that TD; nothing here shows two callers got the same key",
    "physical attacks on the host and side channels are out of scope",
];

/// Added to [`LIMITATIONS`] for a key verified under `attest=cpu`, which only
/// the simulated TDs of the tests are (`test-anchor`, `attest::Attest::Cpu`).
#[cfg(feature = "test-anchor")]
pub const CPU_ONLY_LIMITATIONS: &[&str] = &[
    "attest=cpu: no GPU evidence is checked; a key document that carries some is refused, and a GPU the TD did not attest is not seen here",
];

/// Added to [`LIMITATIONS`] for a key verified under `attest=cpu+gpu`.
pub const GPU_LIMITATIONS: &[&str] = &[
    "prototype: the GPU path ran on one H100 in CC mode with NVIDIA's NRAS on 2026-10-28, 2026-10-30, 2026-10-31 and 2026-11-01, under the TEST archive key and TEST transparency log; every other test ran on a simulated NVIDIA and NRAS",
    "the H100's report names no TD: it is tied to this one by the quote, which binds each report and certificate chain the TD read, and by the nonce each report answers, derived from this caller's nonce (checked by NRAS, and by the TD's measured code reading the reports it binds, not by this SDK); a GPU the host attaches elsewhere is not excluded by NVIDIA's evidence itself",
    "this SDK checks neither the bound certificate chains nor the bound reports: the GPU's identity rests on the leaf serial alone, each token's ueid matched to the serial a bound chain names",
    "NVIDIA's NRAS is trusted: it verifies the reports, the GPUs' certificate chains and their revocation, and the driver and VBIOS measurements against NVIDIA's RIMs; this SDK checks NRAS's signed verdict, not the reports",
    "NRAS's token carries no digest of the report it judged: it says a report of the GPU named by ueid answered this document's nonce, not that it is the very report the quote binds",
    "anchored at NVIDIA's GPU intermediate CAs 004 and 005, each pinned as its whole certificate, not at their root CA 001 (not published); revocation of the NRAS signing chain is judged from the OCSP responses the key document staples, under the policy the verdict names (a verified revoked refuses either way; the intermediate's own status is unverifiable while CA 001 is unpublished; NVIDIA's CRL is not read), and no TD staples them yet; the x5c validity windows are checked; the signing leaf's key usage, extended key usage and subject are not checked, NVIDIA publishing no profile for them",
    "GPU identity: each token's ueid must be the leaf serial of a certificate chain the quote bound, a derivation taken from NVIDIA's nvtrust SDK and matched by production NRAS tokens on 2026-10-14, -17 and -18",
    "no NRAS claim names the GPU's confidential-computing mode: the TD reads no report unless NVML says CC is on, in production, devtools off, a check not yet measured on an H100; this SDK sees only NRAS's verdict (overall result, measres, dbgstat)",
    "up to 8 GPUs, each attested alone: the protected PCIe (PPCIe) and NVSwitch attestation of several GPUs is not done, and nothing here covers the traffic between GPUs",
    "NRAS's answer and its JWKS come in the key document, fetched by the TD for that document's nonce (from a TD before 2026-10-28, relayed by the carrier): neither can forge them, either can withhold them (refused); NRAS is called once per key document, at whatever rate it serves (not measured)",
];
