//! The caller's check of a TD's OHTTP key, before anything is sealed to it.
//!
//! In order, each refusal naming its step and the relay's exit code for it
//! ([`Refusal::code`]). The order is the relay's (`remote_verify.rs`): the
//! platform is judged before the image, so the same evidence meets the same
//! first refusal from both verifiers.
//!
//! 0. `reference`, once, before any key ([`admit_reference`]): the image
//!    reference is signed by x0ne's archive key pinned in this build (99
//!    otherwise) and shown in x0ne's transparency log pinned in this build,
//!    under a fresh quorum of its witnesses (100 otherwise), by the relay's
//!    own steps (`reference::admit_signed`); then parsed (89), and refused
//!    if it reports RTMR0 instead of pinning it (104: `rtmr0.policy` must
//!    be `gce-endorsed`, as the relay requires); then its checkpoint is held
//!    to the caller's memory of the log ([`LogMemory`], the last checkpoint
//!    verified, kept by the caller): a fork (`log-fork`), a smaller
//!    checkpoint (`log-rollback`), a larger one without a consistency proof
//!    from the remembered size or with a false one (`log-consistency`), or
//!    a memory that is not a checkpoint of the pinned log (`log-memory`),
//!    each refused with the relay's log code, 100;
//! 1. `schema`: the document's schema is [`KEY_SCHEMA`];
//! 2. `key`: the key configuration is exactly the suite this SDK speaks,
//!    the receipt key is an Ed25519 point, the session config is one
//!    `x0ne-engine/cloud-config/v1` object of at most 64 KiB, the boot nonce
//!    and the TLS key digest are 32 bytes of hex, the GPU evidence (if any) is
//!    1 to [`crate::e2e::MAX_GPUS`] entries of strict base64 under the caps;
//! 3. `quote`: the DCAP chain of the TDX quote to the pinned root, every
//!    certificate in its window at `now`, TDATTRIBUTES.DEBUG clear (`pcs.rs`);
//! 4. `binding`: `report_data` = [`crate::e2e::key_binding`] over this key
//!    configuration, THIS caller's nonce, the receipt key and the TD's boot
//!    binding (its boot nonce, its TLS key, SHA-256 of the session config's
//!    bytes): a quote for another key, another caller, another moment,
//!    another TD or another session config is refused. A document
//!    with GPU evidence is bound by [`crate::e2e::key_binding_gpu`], over
//!    that evidence too: evidence added, removed or swapped is refused;
//! 5. `tcb`: Intel's signed TCB info and QE identity, relayed by anyone,
//!    judged as the relay judges them (`tcb::judge_live`): under
//!    `transparent` (the default, [`Policy::production`]) the platform, the
//!    TDX module and the QE are `UpToDate` or `SWHardeningNeeded` on Intel's
//!    `standard` collateral, and Intel's `early` view of them is named, not
//!    judged; under `strict` they must be so on the `early` collateral. The
//!    verdict is disclosed as the relay discloses it
//!    ([`VerifiedKey::tcb`], `x0ne-infer/tcb-disclosure/v1`);
//! 6. `revocation`: the PCK is not on Intel's signed PCK CRL
//!    (`tcb::verify_pck_crl`), nor the PCK CA or a collateral signer on the
//!    root CA's CRL (`tcb::verify_root_crl`);
//! 7. `measurement`: the MRTD is in Google's launch endorsement, signed
//!    under the Google root the reference names, for the caller's machine
//!    type (`gce_endorsement.rs`, `reference::check_mrtd`); the firmware
//!    binary that endorsement names, relayed in the collateral, is the one
//!    Google signed and measures the quoted MRTD (`tdvf::verify_endorsed`);
//!    the TD's event log replays into RTMR0..3, every RTMR1 and RTMR2 event
//!    on the reference's closed lists, RTMR3 zero, RTMR0 pinned to the
//!    reference's host events with that binary's configuration volume in
//!    its slot (`ccel.rs`, `reference::check_image_with`; the reference
//!    was admitted `gce-endorsed` at step 0, 104 otherwise); then the
//!    caller's pins, if any;
//! 8. `gpu`: the policy's `attest` against the document (no GPU evidence
//!    under `attest=cpu+gpu`, the one value a caller asks for: refused);
//!    then NVIDIA's signed NRAS verdict, which the TD fetched for this
//!    document and carries in it (or, from a TD before 2026-10-28, which
//!    the carrier relays in the collateral), checked as the relay checks it (`nras.rs`, x0ne-infer's file): ES384 under a
//!    pinned NVIDIA intermediate, issuer, expiry, the nonce
//!    [`crate::e2e::gpu_nonce`] of THIS document, the overall verdict, per
//!    GPU `measres`, NVIDIA's required claims, `secboot`, `dbgstat`
//!    disabled, the three certificate chains valid and not revoked, and
//!    `submods` naming exactly these per-GPU tokens; each token's `ueid` is
//!    the leaf serial of a certificate chain the quote bound, one per GPU;
//! 9. `session`: the session config the quote bound asks for the policy's
//!    `attest`, and ends more than [`SESSION_EXPIRY_GUARD_SECS`] after `now`,
//!    as the relay takes no new request then. The verified key names that
//!    config ([`VerifiedKey::session_config`]): the model the TD serves, when
//!    it ends, the relay's TLS key; the caller reads them before sealing.
//!
//! Steps 3 to 7 are [`verify_quote`], for a `report_data` the caller computed;
//! [`verify_key`] runs it with the key binding of step 4.
//!
//! The collateral is signed by Intel or by Google under roots pinned here, so
//! a carrier may fetch it: it cannot forge it, and withheld it is refused.
//! The firmware binary is named by Google's signed endorsement (its SHA-384)
//! and measured here: a carrier can withhold it (refused), not swap it.
//! It can relay genuine collateral older than Intel's newest: TCB info, QE
//! identity and PCK CRL up to the relay's day old
//! ([`Policy::max_collateral_age_secs`], which a caller may shorten), and
//! the root CA's CRL, whose age is not judged, until its nextUpdate (a year
//! on the CRL of 2026-02-26). A recovery or a revocation Intel published
//! since reaches this check when the carrier relays it; the relay fetches
//! from Intel itself. Under `transparent`, a carrier that withholds Intel's
//! `early` collateral leaves its view unnamed, said so in the disclosure;
//! `strict` refuses. The reference is admitted only signed and logged, as
//! the relay admits it; the log is checked on the proof and the caller's
//! memory of the log (no file, no socket): a checkpoint without a fresh
//! quorum is refused here where the relay reads the log's newer one, and
//! the memory is the caller's to keep ([`LogMemory`]), so a log that shows
//! two callers two histories is seen by a caller who met both heads, not
//! by one who met one alone.

use crate::e2e::{gpu_nonce, key_binding, key_binding_gpu, Boot, MAX_GPUS};
use crate::nras::NrasIntermediatePin;
use crate::ohttp::KeyConfig;
use crate::pcs::{verify_tdx_chain, TrustAnchor};
use crate::tcb::{self, TcbMode};
use crate::tcb_disclosure::TcbDisclosure;
use base64::Engine as _;
use sha2::Digest as _;

/// The relay's cap on the TD's event log (`ccel::MAX_LOG_BYTES`).
const MAX_LOG_BYTES: usize = crate::ccel::MAX_LOG_BYTES as usize;
use serde::{Deserialize, Serialize};

/// An image reference (`x0ne-engine/reference/v2`): x0ne-infer's
/// `reference.rs`, the relay's own type, read from an [`AdmittedReference`].
pub use crate::reference::Reference;

/// v2 since 2026-10-30: the document carries the TD's session config and
/// boot binding, and its quote binds them. A v1 document is refused at
/// `schema`; an SDK before v2 refuses a v2 document by its fields' names.
pub const KEY_SCHEMA: &str = "x0ne-e2e/key/v2";

/// The TD's session config, as the relay writes it and the quote binds it
/// (x0ne-infer's `cloud_config_types.rs`).
pub use crate::cloud_config_types::{
    CloudConfig, E2eStream, SealedSource, SealedWeights, SCHEMA as SESSION_CONFIG_SCHEMA,
};

/// New requests are refused this long before the session's
/// `expires_at_unix`, by [`verify_key`] (`session`) and by
/// [`crate::e2e::Client::seal_chat`]: the relay's guard
/// (`local_api::EXPIRY_GUARD_SECS`).
pub const SESSION_EXPIRY_GUARD_SECS: u64 = 60;

/// The relay's cap on one piece of collateral (`remote_verify.rs`), applied
/// here to each collateral field and to the quote, before decoding.
pub const MAX_COLLATERAL_BYTES: usize = 4 * 1024 * 1024;

/// The cap on one GPU's report or certificate chain, decoded. NVML's
/// buffers hold 8 KiB of report, 4 KiB of CEC report, 5 KiB of chain.
pub const MAX_GPU_EVIDENCE_BYTES: usize = 64 * 1024;

/// Where a carrier POSTs [`nras_request`]: NVIDIA's NRAS, the relay's URL
/// (`nras_client::NRAS_URL`).
pub const NRAS_ATTEST_URL: &str = "https://nras.attestation.nvidia.com/v4/attest/gpu";
/// Where a carrier fetches [`Collateral::nras_jwks`] (`nras_client::NRAS_JWKS_URL`).
pub const NRAS_JWKS_URL: &str = "https://nras.attestation.nvidia.com/.well-known/jwks.json";
/// The architecture [`nras_request`] names: H100, the GPU of a3-highgpu.
pub const NRAS_ARCH: &str = "HOPPER";

/// What the TD answers to `GET /v2/e2e/key?nonce=<hex>`. Untrusted until
/// [`verify_key`] says otherwise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyDocument {
    pub schema: String,
    /// The OHTTP key configuration (RFC 9458 §3), base64.
    pub key_config_b64: String,
    /// The Ed25519 key the TD signs receipts with, hex.
    pub session_pubkey_hex: String,
    /// The TD's session config, its exact bytes, base64: the model it
    /// serves, `expires_at_unix`, and `client_spki_sha256`, the relay's TLS
    /// key, the one client its TLS accepts. The quote binds their SHA-256
    /// ([`crate::e2e::Boot`]). Since v2.
    #[serde(default)]
    pub session_config_b64: String,
    /// The nonce the TD drew at boot, hex: its boot binding's, bound here too.
    #[serde(default)]
    pub boot_nonce_hex: String,
    /// SHA-256 of the TD's TLS SubjectPublicKeyInfo, hex: the key the relay's
    /// channel pins, bound here too.
    #[serde(default)]
    pub tls_spki_sha256_hex: String,
    /// TD Quote v4, base64.
    pub quote_b64: String,
    /// The TD's CC event log (the data of its ACPI CCEL table), base64:
    /// replayed into RTMR0..3 and judged against the caller's reference.
    pub ccel_b64: String,
    /// A TD that attests GPUs: one entry per GPU, as NRAS takes them, read
    /// over NVML for [`crate::e2e::gpu_nonce`]. The quote binds every byte
    /// ([`crate::e2e::key_binding_gpu`]). Absent for a CPU-attested TD.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_evidence_list: Option<Vec<GpuEvidence>>,
    /// NRAS's answer to [`nras_request`] for this document, fetched by the
    /// TD, verbatim in shape: `[["JWT", <overall>], {"GPU-0": <token>}]`.
    /// Signed by NVIDIA, so whoever fetched it cannot forge it. With
    /// `gpu_evidence_list` only; absent from a TD before 2026-10-28, whose
    /// carrier relays it in [`Collateral::nras_answer`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nras_answer: Option<String>,
    /// NRAS's JWKS, as the TD fetched it with [`Self::nras_answer`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nras_jwks: Option<String>,
    /// OCSP responses about the JWKS signing chain, DER base64, as the TD
    /// fetched them: judged under [`Policy::nras_ocsp`]. Absent from every
    /// TD so far: no TD fetches them yet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nras_ocsp: Option<Vec<String>>,
}

/// One GPU's evidence, NRAS's `evidence_list` entry (`nras_client.rs`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuEvidence {
    /// The attestation report (SPDM measurements, signed by the GPU), base64.
    pub evidence: String,
    /// The GPU's attestation certificate chain as NVML returned it, base64.
    pub certificate: String,
}

/// Intel PCS collateral, as PCS serves it: anyone may fetch and relay it.
/// Both of Intel's answers: `update=early` (evaluation 22 on 2026-10-28),
/// what `strict` accepts on and `transparent` names; `update=standard`
/// (evaluation 20 that day), what `transparent` accepts on.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Collateral {
    /// `GET /tdx/certification/v4/tcb?fmspc=<fmspc>&update=early`: the body, verbatim.
    pub tcb_info: String,
    /// Its `TCB-Info-Issuer-Chain` header, percent-decoded (PEM).
    pub tcb_info_chain_pem: String,
    /// `GET /tdx/certification/v4/qe/identity?update=early`: the body, verbatim.
    pub qe_identity: String,
    /// Its `SGX-Enclave-Identity-Issuer-Chain` header, percent-decoded.
    pub qe_identity_chain_pem: String,
    /// `GET /tdx/certification/v4/tcb?fmspc=<fmspc>&update=standard`: the
    /// body, verbatim. Required under `transparent`; unread under `strict`.
    #[serde(default)]
    pub tcb_info_standard: String,
    /// Its `TCB-Info-Issuer-Chain` header, percent-decoded (PEM).
    #[serde(default)]
    pub tcb_info_standard_chain_pem: String,
    /// `GET /tdx/certification/v4/qe/identity?update=standard`: the body, verbatim.
    #[serde(default)]
    pub qe_identity_standard: String,
    /// Its `SGX-Enclave-Identity-Issuer-Chain` header, percent-decoded.
    #[serde(default)]
    pub qe_identity_standard_chain_pem: String,
    /// `GET /sgx/certification/v4/pckcrl?ca=<ca>&encoding=der`, base64.
    pub pck_crl_der_b64: String,
    /// `GET https://certificates.trustedservices.intel.com/IntelSGXRootCA.der`
    /// (the CRL distribution point of Intel's root), base64.
    pub root_ca_crl_der_b64: String,
    /// `GET https://storage.googleapis.com/gce_tcb_integrity/ovmf_x64_csm/tdx/<MRTD hex>.binarypb`,
    /// base64: Google's launch endorsement of the quote's firmware, signed
    /// under Google's root, so a carrier may fetch it too.
    pub gce_endorsement_b64: String,
    /// `GET https://storage.googleapis.com/gce_tcb_integrity/ovmf_x64_csm/<UEFI SHA-384>.fd`
    /// ([`gce_firmware_url`] of that endorsement), base64: the firmware
    /// binary the endorsement names, 2 MiB, at most [`MAX_FIRMWARE_BYTES`].
    /// Its SHA-384 is the one Google signed and it must measure the quoted
    /// MRTD, so a carrier may fetch it; RTMR0 is pinned to its configuration
    /// volume. Required: empty is refused at `measurement`.
    #[serde(default)]
    pub gce_firmware_b64: String,
    /// NRAS's answer to [`nras_request`] POSTed to [`NRAS_ATTEST_URL`], the
    /// body verbatim: `[["JWT", <overall>], {"GPU-0": <token>, ...}]`. Signed
    /// by NVIDIA under pinned intermediates, so a carrier may fetch it.
    /// Read only for a key document that carries no NRAS answer of its own
    /// (a TD before 2026-10-28). Empty for a CPU-attested TD.
    #[serde(default)]
    pub nras_answer: String,
    /// `GET` [`NRAS_JWKS_URL`]: the body verbatim. Empty for a CPU-attested TD.
    #[serde(default)]
    pub nras_jwks: String,
}

/// What the caller requires the TD to attest. A served TD attests its GPUs:
/// x0ne-infer refuses a session config that attests the CPU alone (its exit
/// 90, 2026-11-04), so `cpu+gpu` is the one value a caller asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attest {
    /// The TD alone, a key document with GPU evidence refused: the simulated
    /// TDs of the tests only (`test-anchor`), which serve on the CPU.
    #[cfg(feature = "test-anchor")]
    Cpu,
    /// The TD and every GPU it carries evidence for, at least one.
    CpuGpu,
}

/// [`Attest::parse`] on `"cpu"`: a refusal that says why, not an unknown value.
pub const ATTEST_CPU_REFUSED: &str = "\"cpu\" is refused: a TD serves only on its attested GPU \
     (x0ne-infer refuses a CPU-only session config), ask for \"cpu+gpu\"";

impl Attest {
    /// `"cpu+gpu"`, the relay's session spelling. `"cpu"` is refused by name
    /// ([`ATTEST_CPU_REFUSED`]); a `test-anchor` build keeps it for the
    /// simulated TDs, as x0ne-infer's tests keep the CPU reference.
    pub fn parse(s: &str) -> Result<Self, String> {
        Self::parse_in(s, cfg!(feature = "test-anchor"))
    }

    /// [`Self::parse`], the simulated TDs' `"cpu"` allowed or not: `parse`
    /// passes whether this build has `test-anchor`; a test passes `false` to
    /// see the refusal of a build without it.
    #[doc(hidden)]
    #[cfg_attr(not(feature = "test-anchor"), allow(unused_variables))]
    pub fn parse_in(s: &str, simulated_cpu: bool) -> Result<Self, String> {
        match s {
            "cpu+gpu" => Ok(Self::CpuGpu),
            #[cfg(feature = "test-anchor")]
            "cpu" if simulated_cpu => Ok(Self::Cpu),
            "cpu" => Err(ATTEST_CPU_REFUSED.into()),
            other => Err(format!("{other:?}, expected \"cpu+gpu\"")),
        }
    }
}

/// What the caller requires: the roots, the image reference the TD's
/// registers are judged against, and pins the caller may add on top.
pub struct Policy {
    /// Intel's root: the quote's chain, the collateral, both CRLs.
    pub anchor: TrustAnchor,
    /// Google's root, DER: the signer of every launch endorsement.
    pub gce_root_der: Vec<u8>,
    /// The image the TD must run: signed and logged ([`admit_reference`]).
    pub reference: AdmittedReference,
    /// The GCE machine type the caller expects (`a3-highgpu-1g`): the
    /// reference names the endorsement's `ram_gib` and RTMR0 per machine type.
    pub machine_type: String,
    /// Optional: MRTD must also equal this.
    pub mrtd_pin: Option<[u8; 48]>,
    /// Optional: RTMR1, RTMR2, RTMR3 must also equal these.
    pub rtmr_pins: [Option<[u8; 48]>; 3],
    /// Optional: RTMR0 must also equal this, on top of the reference's pin
    /// to the endorsed firmware and its host events.
    pub rtmr0_pin: Option<[u8; 48]>,
    /// `test-anchor` builds only, never set by default: Google's firmware
    /// binary as measured elsewhere (`tdvf::verify_endorsed` on the real
    /// 2 MiB binary, `x0ne-infer/tests/gce_firmware_live.rs`), for an
    /// offline replay that carries no binary. The endorsement must still
    /// name that SHA-384 and list the quoted MRTD.
    #[cfg(feature = "test-anchor")]
    pub firmware_measured_elsewhere: Option<FirmwareMeasuredElsewhere>,
    /// TCB info and QE identity issued, PCK CRL updated, at most this long
    /// before `now`: the relay's day by default. Shorter is the caller's
    /// choice; longer is refused at `tcb` (`tcb::TcbPolicy::tightened`).
    pub max_collateral_age_secs: u64,
    /// The TD alone, or the TD and its GPUs: refused at `gpu` either way it differs.
    pub attest: Attest,
    /// Which of Intel's collateral the platform is accepted on: `transparent`
    /// (the relay's default) or `strict` ([`tcb::TcbMode`]).
    pub tcb_mode: TcbMode,
    /// The NVIDIA intermediates NRAS tokens must chain to: NVIDIA's 004 and
    /// 005 (`nras::NVIDIA_NRAS_GPU_INTERMEDIATE_PINS`). A pin is built in
    /// `nras.rs` alone; a bench's only with the `test-anchor` feature.
    pub nras_pins: Vec<NrasIntermediatePin>,
    /// How the NRAS signing chain's revocation is judged from the OCSP
    /// responses the key document staples: `transparent` by default, as the
    /// relay's, a verified revoked refusing either way; `required`
    /// ([`Self::with_required_nras_ocsp`]) refuses anything short of a
    /// verified, current `good` for the signing leaf.
    pub nras_ocsp: crate::nras_ocsp::NrasOcspPolicy,
}

impl Policy {
    /// Intel's root pinned by its SPKI digest, Google's `GCE-cc-tcb-root`,
    /// `reference` for `machine_type`, the `transparent` TCB policy as the
    /// relay's default, no pin.
    pub fn production(reference: AdmittedReference, machine_type: &str) -> Self {
        Self {
            anchor: TrustAnchor::intel_production(),
            gce_root_der: crate::gce_endorsement::ROOT_1_DER.to_vec(),
            reference,
            machine_type: machine_type.into(),
            mrtd_pin: None,
            rtmr_pins: [None; 3],
            rtmr0_pin: None,
            #[cfg(feature = "test-anchor")]
            firmware_measured_elsewhere: None,
            max_collateral_age_secs: tcb::TcbPolicy::MAX_COLLATERAL_AGE_SECS,
            attest: Attest::CpuGpu,
            tcb_mode: tcb::TcbPolicy::production().mode,
            nras_pins: crate::nras::NVIDIA_NRAS_GPU_INTERMEDIATE_PINS.to_vec(),
            nras_ocsp: crate::nras_ocsp::NrasOcspPolicy::production(),
        }
    }

    /// The same policy under `strict`: the platform accepted on Intel's
    /// `early` collateral only, and every answer's receipt held to it
    /// (`tcb_disclosure::strict_receipt_check`, in `e2e::Pending`).
    #[must_use]
    pub fn with_strict_tcb(mut self) -> Self {
        self.tcb_mode = TcbMode::Strict;
        self
    }

    /// The same policy with the NRAS signing chain's OCSP `required`: the
    /// key document must staple a verified, current `good` for the signing
    /// leaf, or the key is refused at `revocation` (75).
    #[must_use]
    pub fn with_required_nras_ocsp(mut self) -> Self {
        self.nras_ocsp.mode = crate::nras_ocsp::NrasOcspMode::Required;
        self
    }

    /// The same policy, collateral at most `secs` old: at most the relay's
    /// day, a longer one is an error.
    pub fn with_max_collateral_age(mut self, secs: u64) -> Result<Self, String> {
        tcb::TcbPolicy::tightened(None, Some(secs))?;
        self.max_collateral_age_secs = secs;
        Ok(self)
    }

    /// The same policy, MRTD and RTMR1..3 also pinned to these values.
    #[must_use]
    pub fn pinned(mut self, mrtd: [u8; 48], rtmr: [[u8; 48]; 3]) -> Self {
        self.mrtd_pin = Some(mrtd);
        self.rtmr_pins = rtmr.map(Some);
        self
    }

    /// The same policy, RTMR0 also pinned to this value.
    #[must_use]
    pub fn pinned_rtmr0(mut self, rtmr0: [u8; 48]) -> Self {
        self.rtmr0_pin = Some(rtmr0);
        self
    }

    /// The same policy on a firmware binary measured elsewhere
    /// ([`Policy::firmware_measured_elsewhere`]). `test-anchor` builds only.
    #[cfg(feature = "test-anchor")]
    #[must_use]
    pub fn with_firmware_measured_elsewhere(mut self, firmware: FirmwareMeasuredElsewhere) -> Self {
        self.firmware_measured_elsewhere = Some(firmware);
        self
    }
}

/// Google's firmware binary as `tdvf::verify_endorsed` measured it on the
/// binary itself, for a `test-anchor` replay that carries none
/// ([`Policy::firmware_measured_elsewhere`]).
#[cfg(feature = "test-anchor")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirmwareMeasuredElsewhere {
    /// SHA-384 of the binary, hex: the endorsement must name it.
    pub sha384_hex: String,
    /// SHA-384 of its configuration volume, the firmware event of RTMR0.
    pub cfv_sha384: [u8; 48],
}

/// The largest firmware binary read from [`Collateral::gce_firmware_b64`]:
/// the relay's cap on its own fetch (`tdvf::MAX_FIRMWARE_BYTES`).
pub const MAX_FIRMWARE_BYTES: usize = crate::tdvf::MAX_FIRMWARE_BYTES;

/// An image reference file as x0ne publishes it: the file, its detached
/// OpenPGP signature (`<file>.sig`, binary) and its log proof
/// (`<file>.tlog-proof`, C2SP tlog-proof text).
#[derive(Debug, Clone, Copy)]
pub struct SignedReference<'a> {
    pub json: &'a [u8],
    pub signature: &'a [u8],
    pub tlog_proof: &'a str,
}

/// A reference [`admit_reference`] admitted: what the relay's `reference::load`
/// establishes, on bytes. Only the functions of this module build one.
#[derive(Debug, Clone)]
pub struct AdmittedReference {
    reference: Reference,
    sha256_hex: String,
    signer: crate::openpgp::Signer,
    logged: crate::transparency::Logged,
    /// The size of the log the caller remembered before this admission
    /// (None the first time), and the size kept after it.
    memory_before: Option<u64>,
    memory_kept: u64,
}

impl AdmittedReference {
    pub fn reference(&self) -> &Reference {
        &self.reference
    }

    /// SHA-256 of the reference file's bytes, hex.
    pub fn sha256_hex(&self) -> &str {
        &self.sha256_hex
    }

    /// Who signed it and where the log shows it, as the relay's verdict
    /// says it: the primary key's fingerprint, the signing subkey, the
    /// signature's time, TEST when it is the TEST key; the log's origin,
    /// the entry's index, the checkpoint's size and root, the witnesses.
    pub fn provenance(&self) -> serde_json::Value {
        serde_json::json!({
            "sha256_hex": self.sha256_hex,
            "signed_by": {
                "primary_fingerprint": self.signer.primary_fingerprint,
                "subkey_fingerprint": self.signer.subkey_fingerprint,
                "signed_at_unix": self.signer.signed_at_unix,
                "test_key": self.signer.test_key,
            },
            "log": {
                "origin": self.logged.origin,
                "index": self.logged.index,
                "size": self.logged.size,
                "root_hex": self.logged.root_hex,
                "cosigned_by": self.logged.cosigned_by,
                // The caller's memory of the log: the size it remembered
                // before (null the first time), the size it keeps now.
                "memory": { "before": self.memory_before, "kept": self.memory_kept },
            },
        })
    }
}

/// The caller's memory of the pinned log: the last checkpoint it verified,
/// as the signed note of that checkpoint. Held by the caller between calls
/// and between runs, where a witness keeps its latest checkpoint
/// (tlog-witness): this crate reads and writes no file, so the caller keeps
/// [`LogMemory::note`] and hands it back with [`LogMemory::from_note`]. A
/// checkpoint of the same size must show the same root, a smaller one is
/// refused, a larger one needs a consistency proof ([`admit_reference`]).
/// A memory lost or emptied starts over: the next checkpoint is kept as
/// the first.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogMemory {
    note: Option<String>,
}

impl LogMemory {
    /// Nothing remembered: a caller's first admission.
    pub fn new() -> Self {
        Self::default()
    }

    /// The note kept from an earlier admission. Verified under the pinned
    /// log at the next admission, not here: a note that is not a checkpoint
    /// signed by that log refuses every reference until the caller clears it.
    pub fn from_note(note: impl Into<String>) -> Self {
        Self {
            note: Some(note.into()),
        }
    }

    /// What to keep: the signed checkpoint note, or None when nothing was
    /// admitted yet.
    pub fn note(&self) -> Option<&str> {
        self.note.as_deref()
    }

    /// The remembered checkpoint's size and root (hex), read from the note
    /// without verifying it: for display, and for the `old` size of the
    /// consistency proof the caller asks the log for.
    pub fn checkpoint(&self) -> Option<(u64, String)> {
        let n = crate::tlog::parse_note(self.note.as_deref()?).ok()?;
        let cp = crate::tlog::parse_checkpoint(&n.text).ok()?;
        Some((cp.size, crate::tlog::hex(&cp.root)))
    }
}

/// The admission of an image reference, by the relay's own steps on bytes
/// (`reference::admit_offline`): signed by the archive key this build pins
/// (99), shown in the log this build pins under a fresh quorum (100),
/// parsed as schema v2 with the image version floor (89); then held to the
/// caller's memory of the log (`transparency::remember`, 100 too, the step
/// named): the proof's checkpoint must be the one remembered (same size,
/// same root: a fork otherwise, `log-fork`), or a larger one shown to
/// extend it by `consistency`, a tlog-witness `add-checkpoint` body from
/// the remembered size (`log-consistency` when missing, false or from
/// another size); a smaller one is refused (`log-rollback`). On success
/// `memory` holds the checkpoint to keep. A build made before the signing
/// ceremony pins no key and no log: it refuses every reference, with 99,
/// as the relay does.
pub fn admit_reference(
    signed: &SignedReference<'_>,
    memory: &mut LogMemory,
    consistency: Option<&str>,
    now: u64,
) -> Result<AdmittedReference, Refusal> {
    admit_under(
        signed,
        memory,
        consistency,
        &crate::archive_key::Anchor::production(),
        &crate::transparency::Anchor::production(),
        now,
    )
}

/// [`admit_reference`] under the TEST archive key and the TEST log, whose
/// secrets anyone can derive: a bench's references. `test-anchor` builds only.
#[cfg(feature = "test-anchor")]
pub fn admit_reference_under_test_keys(
    signed: &SignedReference<'_>,
    memory: &mut LogMemory,
    consistency: Option<&str>,
    now: u64,
) -> Result<AdmittedReference, Refusal> {
    admit_under(
        signed,
        memory,
        consistency,
        &crate::archive_key::Anchor::test(),
        &crate::transparency::Anchor::test(),
        now,
    )
}

/// The TEST log in memory, for a bench that grows, forks and proves it:
/// `test-anchor` builds only.
#[cfg(feature = "test-anchor")]
pub use crate::transparency::test_log::MemoryLog as TestLog;

/// A bench's signature and log proof for `json`: the TEST key's detached
/// signature made at `now`, and a TEST log that holds it alone, cosigned at
/// `now`. `test-anchor` builds only.
#[cfg(feature = "test-anchor")]
pub fn sign_reference_under_test_keys(json: &[u8], now: u64) -> (Vec<u8>, String) {
    let at = u32::try_from(now).unwrap_or(u32::MAX);
    (
        crate::archive_key::test_key::sign(json, at),
        crate::transparency::test_log::proof_of_one(json, now),
    )
}

fn admit_under(
    signed: &SignedReference<'_>,
    memory: &mut LogMemory,
    consistency: Option<&str>,
    key: &crate::archive_key::Anchor,
    log: &crate::transparency::Anchor,
    now: u64,
) -> Result<AdmittedReference, Refusal> {
    let a = crate::reference::admit_offline(
        signed.json,
        signed.signature,
        signed.tlog_proof,
        key,
        log,
        now,
    )
    .map_err(|e| Refusal {
        step: "reference",
        code: e.exit_code(),
        reason: e.to_string(),
    })?;
    // The proof passed on its own; now against what this caller remembers of
    // the log. The policy is the one admit_offline just used: a build with no
    // log was refused there.
    let policy = log.policy().map_err(|e| Refusal {
        step: "log-memory",
        code: crate::transparency::EXIT_NOT_LOGGED,
        reason: e,
    })?;
    let kept = crate::transparency::remember(policy, memory.note(), signed.tlog_proof, consistency)
        .map_err(|e| Refusal {
            step: e.step(),
            code: crate::transparency::EXIT_NOT_LOGGED,
            reason: e.to_string(),
        })?;
    memory.note = Some(kept.note);
    Ok(AdmittedReference {
        reference: a.reference,
        sha256_hex: a.sha256_hex,
        signer: a.signed_by,
        logged: a.logged,
        memory_before: kept.before,
        memory_kept: kept.size,
    })
}

/// `field` as text, at most `cap` bytes.
fn capped<'a>(
    step: &'static str,
    field: &str,
    text: &'a str,
    cap: usize,
) -> Result<&'a str, Refusal> {
    if text.len() > cap {
        return Err(refuse(
            step,
            format!("{field}: {} bytes, cap {cap}", text.len()),
        ));
    }
    Ok(text)
}

/// `field` decoded from base64, refused unread when its length could hold
/// more than `cap` bytes.
fn b64_capped(field: &str, text: &str, cap: usize) -> Result<Vec<u8>, String> {
    let most = cap.div_ceil(3).saturating_mul(4);
    if text.len() > most {
        return Err(format!(
            "{field}: {} bytes of base64, over the cap of {cap} bytes",
            text.len()
        ));
    }
    b64(text).map_err(|e| format!("{field}: {e}"))
}

/// Why a key was refused: the step, what it found, and the relay's exit
/// code for the same refusal (`bin/proxy.rs`): 65 evidence, 75 GPU, 76
/// Intel's TCB, 88 schema, key, binding or session, 89 an unusable reference or a Google
/// root other than the reference's, 99 signature, 100 log, 104 a reference
/// that reports RTMR0 instead of pinning it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Refusal {
    pub step: &'static str,
    pub code: i32,
    pub reason: String,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.step, self.reason)
    }
}

fn refuse(step: &'static str, reason: impl Into<String>) -> Refusal {
    let code = match step {
        "schema" | "key" | "binding" | "session" => 88,
        "tcb" => 76,
        "gpu" => 75,
        _ => 65,
    };
    refuse_with(step, code, reason)
}

fn refuse_with(step: &'static str, code: i32, reason: impl Into<String>) -> Refusal {
    Refusal {
        step,
        code,
        reason: reason.into(),
    }
}

/// `revocation`, as the relay codes it: a listed PCK or CA is evidence (65),
/// a CRL that cannot be used is Intel's collateral (76).
fn revocation(e: tcb::TcbRejected) -> Refusal {
    match e {
        tcb::TcbRejected::PckRevoked { .. } | tcb::TcbRejected::CaRevoked { .. } => {
            refuse_with("revocation", 65, e.to_string())
        }
        _ => refuse_with("revocation", 76, e.to_string()),
    }
}

/// A key [`verify_key`] accepted. Only that function builds one.
#[derive(Debug, Clone)]
pub struct VerifiedKey {
    key_config: KeyConfig,
    session_pubkey: [u8; 32],
    pub tcb_platform: String,
    pub tcb_qe: String,
    /// The TCB verdict as the relay discloses it to its caller and in every
    /// receipt (`x0ne-infer/tcb-disclosure/v1`): the policy, the collateral
    /// accepted on, Intel's latest view or why it is missing, and one line.
    pub tcb: TcbDisclosure,
    pub mrtd_hex: String,
    /// Pinned: the reference's host events, the endorsed firmware's
    /// configuration volume in its slot ([`VerifiedQuote::rtmr0_hex`]).
    pub rtmr0_hex: String,
    /// SHA-384 of the firmware binary measured here, the one Google signed.
    pub firmware_sha384_hex: String,
    pub pck_sha256_hex: String,
    /// The reference's image, `<name> <version>`.
    pub image: String,
    /// The admitted reference's digest, signer and log entry
    /// ([`AdmittedReference::provenance`]).
    pub reference: serde_json::Value,
    /// What the image check reported without refusing: empty since RTMR0 is
    /// pinned (2026-10-31); kept for the callers that show it.
    pub warnings: Vec<String>,
    /// The attested GPUs, by the `ueid` NRAS names (the leaf serial of each
    /// certificate chain the quote bound). Empty under `attest=cpu`.
    pub gpu_ueids: Vec<String>,
    /// The NRAS signing chain's revocation, by certificate, from the OCSP
    /// responses the document stapled under [`Policy::nras_ocsp`]
    /// (`nras_ocsp::OcspStanding`, one line each). Empty under `attest=cpu`.
    pub nras_revocation: Vec<String>,
    /// The session config the quote bound, as its bytes say it: `model`,
    /// the catalog name the TD serves (the receipt's digest is still the
    /// caller's to pin); `expires_at_unix`; `client_spki_sha256`, the
    /// relay's TLS key. Read before anything is sealed.
    pub session_config: CloudConfig,
    /// SHA-256 of those bytes, hex: the `config_sha256` of the boot binding.
    pub session_config_sha256_hex: String,
    /// The TD's boot nonce, hex, as the quote bound it.
    pub boot_nonce_hex: String,
    /// SHA-256 of the TD's TLS key, hex, as the quote bound it.
    pub tls_spki_sha256_hex: String,
    pub limitations: Vec<String>,
}

impl VerifiedKey {
    pub fn key_config(&self) -> &KeyConfig {
        &self.key_config
    }

    /// The Ed25519 key the TD signs receipts with, as the quote bound it.
    pub fn session_pubkey(&self) -> &[u8; 32] {
        &self.session_pubkey
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn b64(s: &str) -> Result<Vec<u8>, String> {
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| e.to_string())
}

/// The caller's verdict on `doc`, for the nonce it drew, at `now` (Unix
/// seconds, the caller's clock).
pub fn verify_key(
    doc: &KeyDocument,
    collateral: &Collateral,
    policy: &Policy,
    client_nonce: &[u8; 32],
    now: u64,
) -> Result<VerifiedKey, Refusal> {
    if doc.schema != KEY_SCHEMA {
        return Err(refuse(
            "schema",
            format!("{:?}, expected {KEY_SCHEMA:?}", doc.schema),
        ));
    }
    let config_bytes =
        b64(&doc.key_config_b64).map_err(|e| refuse("key", format!("key_config_b64: {e}")))?;
    let key_config = KeyConfig::parse(&config_bytes).map_err(|e| refuse("key", e.to_string()))?;
    let session_pubkey = crate::unhex::<32>(&doc.session_pubkey_hex)
        .map_err(|e| refuse("key", format!("session_pubkey_hex: {e}")))?;
    ed25519_dalek::VerifyingKey::from_bytes(&session_pubkey)
        .map_err(|e| refuse("key", format!("session_pubkey_hex: {e}")))?;
    let (session_config, boot) = session_and_boot(doc).map_err(|e| refuse("key", e))?;
    let gpus = doc
        .gpu_evidence_list
        .as_deref()
        .map(gpu_evidence_bytes)
        .transpose()
        .map_err(|e| refuse("key", e))?;
    if gpus.is_none() && (doc.nras_answer.is_some() || doc.nras_jwks.is_some()) {
        return Err(refuse(
            "key",
            "an NRAS verdict without GPU evidence to judge",
        ));
    }

    let quote = b64_capped("quote_b64", &doc.quote_b64, MAX_COLLATERAL_BYTES)
        .map_err(|e| refuse("quote", e))?;
    let expected = match &gpus {
        None => key_binding(&config_bytes, client_nonce, &session_pubkey, &boot),
        Some(gpus) => {
            let views: Vec<(&[u8], &[u8])> = gpus
                .iter()
                .map(|(r, c)| (r.as_slice(), c.as_slice()))
                .collect();
            key_binding_gpu(&config_bytes, client_nonce, &session_pubkey, &boot, &views)
        }
    };
    let ccel = b64_capped("ccel_b64", &doc.ccel_b64, MAX_LOG_BYTES);
    let checked = checked_quote(
        &quote,
        ccel.as_deref(),
        &expected,
        if gpus.is_some() {
            "report_data is not SHA-512 over this key configuration, this caller's nonce, this receipt key, this boot binding (boot nonce, TLS key, session config) and this GPU evidence"
        } else {
            "report_data is not SHA-512 over this key configuration, this caller's nonce, this receipt key and this boot binding (boot nonce, TLS key, session config)"
        },
        collateral,
        policy,
        now,
    )?;
    let nonce = hex(&gpu_nonce(&config_bytes, client_nonce, &session_pubkey));
    let (gpu_ueids, nras_revocation) = checked_gpus(doc, &nonce, collateral, policy, now)?;
    checked_session(&session_config, policy, now)?;
    let mut limitations: Vec<String> = crate::LIMITATIONS.iter().map(|s| s.to_string()).collect();
    let per_attest = match policy.attest {
        #[cfg(feature = "test-anchor")]
        Attest::Cpu => crate::CPU_ONLY_LIMITATIONS,
        Attest::CpuGpu => crate::GPU_LIMITATIONS,
    };
    limitations.extend(per_attest.iter().map(|s| s.to_string()));
    if matches!(policy.attest, Attest::CpuGpu) {
        limitations.push(format!(
            "NRAS signing chain revocation, policy {}: {}",
            policy.nras_ocsp.mode.as_str(),
            nras_revocation.join(", ")
        ));
    }
    Ok(VerifiedKey {
        key_config,
        session_pubkey,
        tcb_platform: checked.tcb_platform,
        tcb_qe: checked.tcb_qe,
        tcb: checked.tcb,
        mrtd_hex: checked.mrtd_hex,
        rtmr0_hex: checked.rtmr0_hex,
        firmware_sha384_hex: checked.firmware_sha384_hex,
        pck_sha256_hex: checked.pck_sha256_hex,
        image: checked.image,
        reference: policy.reference.provenance(),
        warnings: checked.warnings,
        gpu_ueids,
        nras_revocation,
        session_config,
        session_config_sha256_hex: hex(&boot.config_sha256),
        boot_nonce_hex: hex(&boot.nonce),
        tls_spki_sha256_hex: hex(&boot.tls_spki_sha256),
        limitations,
    })
}

/// `key`, the session config and the boot binding: the config's bytes,
/// strict base64 of at most the TD's cap (`cloud_config::MAX_BYTES`), one
/// `x0ne-engine/cloud-config/v1` object, no field this SDK does not know;
/// the boot nonce and the TLS key digest, 32 bytes each. The [`Boot`] the
/// binding hashes takes SHA-256 of the bytes as they came, never of a
/// re-serialization.
fn session_and_boot(doc: &KeyDocument) -> Result<(CloudConfig, Boot), String> {
    let cap = usize::try_from(crate::cloud_config_types::MAX_BYTES).unwrap_or(usize::MAX);
    let bytes = b64_capped("session_config_b64", &doc.session_config_b64, cap)?;
    if bytes.len() > cap {
        return Err(format!(
            "session_config_b64: {} bytes, cap {cap}",
            bytes.len()
        ));
    }
    let config: CloudConfig = serde_json::from_slice(&bytes)
        .map_err(|e| format!("session_config_b64: not a session config: {e}"))?;
    if config.schema != SESSION_CONFIG_SCHEMA {
        return Err(format!(
            "session_config_b64: schema {:?}, expected {SESSION_CONFIG_SCHEMA:?}",
            config.schema
        ));
    }
    let boot = Boot {
        nonce: crate::unhex::<32>(&doc.boot_nonce_hex)
            .map_err(|e| format!("boot_nonce_hex: {e}"))?,
        tls_spki_sha256: crate::unhex::<32>(&doc.tls_spki_sha256_hex)
            .map_err(|e| format!("tls_spki_sha256_hex: {e}"))?,
        config_sha256: sha2::Sha256::digest(&bytes).into(),
    };
    Ok((config, boot))
}

/// `session`: the session config the quote bound asks for what the policy
/// attests, and is not about to end: the relay takes no new request within
/// [`SESSION_EXPIRY_GUARD_SECS`] of `expires_at_unix`.
fn checked_session(config: &CloudConfig, policy: &Policy, now: u64) -> Result<(), Refusal> {
    let wanted = match policy.attest {
        #[cfg(feature = "test-anchor")]
        Attest::Cpu => "cpu",
        Attest::CpuGpu => "cpu+gpu",
    };
    if config.attest != wanted {
        return Err(refuse(
            "session",
            format!(
                "the TD's session config asks for attest={}; the policy is attest={wanted}",
                config.attest
            ),
        ));
    }
    let ends = config.expires_at_unix;
    if now.saturating_add(SESSION_EXPIRY_GUARD_SECS) >= ends {
        return Err(refuse(
            "session",
            format!(
                "the TD's session expires at {ends}, within {SESSION_EXPIRY_GUARD_SECS}s of now ({now}): no new request, as the relay"
            ),
        ));
    }
    Ok(())
}

/// `key`, the GPU evidence: 1 to [`MAX_GPUS`] entries, each field strict
/// base64 of at most [`MAX_GPU_EVIDENCE_BYTES`]; the decoded (report, chain)
/// pairs [`key_binding_gpu`] hashes.
fn gpu_evidence_bytes(list: &[GpuEvidence]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, String> {
    if list.is_empty() || list.len() > MAX_GPUS {
        return Err(format!(
            "gpu_evidence_list: {} entries, expected 1 to {MAX_GPUS}",
            list.len()
        ));
    }
    list.iter()
        .enumerate()
        .map(|(i, g)| {
            let field = |name: &str| format!("gpu_evidence_list[{i}].{name}");
            Ok((
                b64_capped(&field("evidence"), &g.evidence, MAX_GPU_EVIDENCE_BYTES)?,
                b64_capped(
                    &field("certificate"),
                    &g.certificate,
                    MAX_GPU_EVIDENCE_BYTES,
                )?,
            ))
        })
        .collect()
}

/// `gpu`: the policy's [`Attest`] against the document, both ways; then
/// NRAS's verdict on the GPUs whose evidence the quote bound, by the
/// relay's own check (`nras::verify_nras_attestation`), the local serials
/// read from those bound chains. Returns the attested `ueid`s.
fn checked_gpus(
    doc: &KeyDocument,
    nonce_hex: &str,
    collateral: &Collateral,
    policy: &Policy,
    now: u64,
) -> Result<(Vec<String>, Vec<String>), Refusal> {
    let gpus = match (policy.attest, doc.gpu_evidence_list.as_deref()) {
        #[cfg(feature = "test-anchor")]
        (Attest::Cpu, None) => return Ok((Vec::new(), Vec::new())),
        #[cfg(feature = "test-anchor")]
        (Attest::Cpu, Some(g)) => {
            return Err(refuse(
                "gpu",
                format!(
                    "the policy is attest=cpu; this key document carries evidence for {} GPU(s)",
                    g.len()
                ),
            ))
        }
        (Attest::CpuGpu, None) => {
            return Err(refuse(
                "gpu",
                "the policy is attest=cpu+gpu; this key document carries no GPU evidence",
            ))
        }
        (Attest::CpuGpu, Some(g)) => g,
    };
    // The document's own verdict, fetched by the TD; a TD before 2026-10-28
    // carries none, and its carrier relays it. Either is NVIDIA's signature,
    // checked the same way below.
    let (answer, jwks, from) = match (&doc.nras_answer, &doc.nras_jwks) {
        (Some(a), Some(j)) => (a.as_str(), j.as_str(), "key document"),
        (None, None) => (
            collateral.nras_answer.as_str(),
            collateral.nras_jwks.as_str(),
            "collateral",
        ),
        _ => {
            return Err(refuse(
                "gpu",
                "the key document carries an NRAS answer without its JWKS, or the reverse",
            ))
        }
    };
    let cap = MAX_COLLATERAL_BYTES;
    let answer = capped("gpu", "nras_answer", answer, cap)?;
    let jwks = capped("gpu", "nras_jwks", jwks, cap)?;
    if answer.is_empty() || jwks.is_empty() {
        return Err(refuse(
            "gpu",
            format!("no NRAS answer or no NRAS JWKS in the {from}"),
        ));
    }
    let answer: serde_json::Value =
        serde_json::from_str(answer).map_err(|e| refuse("gpu", format!("nras_answer: {e}")))?;
    let (overall, per_gpu) =
        split_nras_answer(&answer).map_err(|e| refuse("gpu", format!("nras_answer: {e}")))?;
    let jwks: serde_json::Value =
        serde_json::from_str(jwks).map_err(|e| refuse("gpu", format!("nras_jwks: {e}")))?;
    let serials = gpus
        .iter()
        .map(|g| crate::nras::gpu_leaf_serial_decimal(&g.certificate))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| refuse("gpu", e.to_string()))?;
    // The OCSP responses the TD stapled about the signing chain, judged under
    // the caller's policy; a document without any is judged as such.
    let ocsp = match &doc.nras_ocsp {
        Some(list) => list
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let s = capped("gpu", "nras_ocsp", s, cap)?;
                base64::Engine::decode(&base64::engine::general_purpose::STANDARD, s)
                    .map_err(|e| refuse("gpu", format!("nras_ocsp[{i}]: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };
    let verified = crate::nras::verify_nras_attestation(
        &overall,
        &per_gpu,
        &serials,
        &jwks,
        nonce_hex,
        now,
        &policy.nras_pins,
        &ocsp,
        &policy.nras_ocsp,
    )
    .map_err(|e| match &e {
        crate::nras::NrasRejected::SigningChainRevoked(_)
        | crate::nras::NrasRejected::SigningChainRevocationUnverified(_) => {
            refuse_with("revocation", 75, format!("NRAS: {e}"))
        }
        _ => refuse("gpu", format!("NRAS: {e}")),
    })?;
    Ok((
        verified.gpu_ueids,
        verified
            .revocation
            .iter()
            .map(ToString::to_string)
            .collect(),
    ))
}

/// NRAS's answer split as the relay splits it
/// (`nras_client::split_attest_response`): the overall token, then each
/// per-GPU token by its name.
fn split_nras_answer(body: &serde_json::Value) -> Result<(String, Vec<(String, String)>), String> {
    let [first, second] = body
        .as_array()
        .map(Vec::as_slice)
        .ok_or("not the unified-EAT array")?
    else {
        return Err("not two elements".into());
    };
    let overall = match first.as_array().map(Vec::as_slice) {
        Some([kind, token]) if kind == "JWT" => token.as_str(),
        _ => None,
    }
    .ok_or("first element is not [\"JWT\", <token>]")?
    .to_string();
    let per_gpu = second
        .as_object()
        .ok_or("second element is not the per-GPU map")?
        .iter()
        .map(|(gpu, v)| {
            v.as_str()
                .map(|t| (gpu.clone(), t.to_string()))
                .ok_or_else(|| format!("{gpu}: token is not a string"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if per_gpu.is_empty() {
        return Err("zero GPUs".into());
    }
    Ok((overall, per_gpu))
}

/// The body a carrier POSTs to [`NRAS_ATTEST_URL`] for `doc`, the key
/// document that answers `client_nonce`: NVIDIA's v4 request, as the relay
/// builds it (`nras_client::build_attest_request`), the document's GPU
/// evidence verbatim, nonce [`crate::e2e::gpu_nonce`]. NRAS's answer goes,
/// verbatim, in [`Collateral::nras_answer`]. Nothing here is checked: the
/// verdict is [`verify_key`]'s.
pub fn nras_request(
    doc: &KeyDocument,
    client_nonce: &[u8; 32],
) -> Result<serde_json::Value, String> {
    let list = doc
        .gpu_evidence_list
        .as_ref()
        .ok_or("this key document carries no GPU evidence")?;
    let config = b64(&doc.key_config_b64).map_err(|e| format!("key_config_b64: {e}"))?;
    KeyConfig::parse(&config).map_err(|e| format!("key_config_b64: {e}"))?;
    let pubkey = crate::unhex::<32>(&doc.session_pubkey_hex)
        .map_err(|e| format!("session_pubkey_hex: {e}"))?;
    Ok(serde_json::json!({
        "nonce": hex(&gpu_nonce(&config, client_nonce, &pubkey)),
        "evidence_list": list,
        "arch": NRAS_ARCH,
        "claims_version": "3.0",
    }))
}

/// What [`verify_quote`] established about a TD Quote v4.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedQuote {
    /// Intel's TCB status of the platform, then of the QE, on the collateral
    /// the acceptance stands on.
    pub tcb_platform: String,
    pub tcb_qe: String,
    /// The verdict as the relay discloses it.
    pub tcb: TcbDisclosure,
    pub mrtd_hex: String,
    /// Pinned: every RTMR0 event is the reference's for the machine type,
    /// the firmware slot being the configuration volume of the binary
    /// Google endorsed, measured here (`reference::check_image_with`).
    pub rtmr0_hex: String,
    /// SHA-384 of that binary, the one Google signed.
    pub firmware_sha384_hex: String,
    /// SHA-256 of the PCK certificate (DER).
    pub pck_sha256_hex: String,
    /// The reference's image, `<name> <version>`.
    pub image: String,
    /// What the image check reported without refusing: empty since RTMR0 is
    /// pinned (2026-10-31); kept for the callers that show it.
    pub warnings: Vec<String>,
}

/// Steps `quote`, `binding`, `tcb`, `revocation` and `measurement` of
/// [`verify_key`], in that order, on a TD Quote v4 and the TD's event log
/// `ccel`, the quote's `report_data` equal to `expected_report_data`, which
/// the caller computed from what it asked for. [`verify_key`] passes [`crate::e2e::key_binding`]; a quote that binds
/// anything else is refused at `binding`.
pub fn verify_quote(
    quote: &[u8],
    ccel: &[u8],
    expected_report_data: &[u8; 64],
    collateral: &Collateral,
    policy: &Policy,
    now: u64,
) -> Result<VerifiedQuote, Refusal> {
    checked_quote(
        quote,
        Ok(ccel),
        expected_report_data,
        "report_data is not the one this caller expects",
        collateral,
        policy,
        now,
    )
}

fn checked_quote(
    quote: &[u8],
    ccel: Result<&[u8], &String>,
    expected_report_data: &[u8; 64],
    binding_refusal: &str,
    collateral: &Collateral,
    policy: &Policy,
    now: u64,
) -> Result<VerifiedQuote, Refusal> {
    if quote.len() > MAX_COLLATERAL_BYTES {
        return Err(refuse(
            "quote",
            format!("{} bytes, cap {MAX_COLLATERAL_BYTES}", quote.len()),
        ));
    }
    let chain =
        verify_tdx_chain(quote, &policy.anchor, now).map_err(|e| refuse("quote", e.to_string()))?;
    if chain.report_data != *expected_report_data {
        return Err(refuse("binding", binding_refusal));
    }

    let claims = x0ne_confidential::tdx_quote::parse(quote)
        .map_err(|e| refuse("quote", format!("{e:?}")))?;

    // The relay's policy (2026-10-28): its mode, TCB evaluation 22 on
    // `early`, 20 on `standard`, collateral at most a day old. The caller may
    // shorten the age, not lengthen it, and ask for `strict`.
    let tcb_policy = tcb::TcbPolicy::tightened(None, Some(policy.max_collateral_age_secs))
        .map_err(|e| refuse("tcb", e))?
        .with_mode(policy.tcb_mode);
    let cap = MAX_COLLATERAL_BYTES;
    let early = relayed(
        "early",
        [
            ("tcb_info", &collateral.tcb_info),
            ("tcb_info_chain_pem", &collateral.tcb_info_chain_pem),
            ("qe_identity", &collateral.qe_identity),
            ("qe_identity_chain_pem", &collateral.qe_identity_chain_pem),
        ],
    )?;
    let standard = relayed(
        "standard",
        [
            ("tcb_info_standard", &collateral.tcb_info_standard),
            (
                "tcb_info_standard_chain_pem",
                &collateral.tcb_info_standard_chain_pem,
            ),
            ("qe_identity_standard", &collateral.qe_identity_standard),
            (
                "qe_identity_standard_chain_pem",
                &collateral.qe_identity_standard_chain_pem,
            ),
        ],
    )?;
    // The relay's step 4 (`remote_verify::verify_document`): accepted on the
    // collateral of the policy's update; under `transparent`, `early` too,
    // rated and named, or why it is missing.
    let missing = |update: &str| {
        refuse(
            "tcb",
            format!(
                "the {} TCB policy accepts on Intel's update={update} collateral: none relayed",
                tcb_policy.mode.as_str()
            ),
        )
    };
    let (accept, latest) = match tcb_policy.mode {
        TcbMode::Strict => (early.ok_or_else(|| missing("early"))?, None),
        TcbMode::Transparent => (
            standard.ok_or_else(|| missing("standard"))?,
            Some(early.ok_or_else(|| "Intel's early collateral was not relayed".to_string())),
        ),
    };
    let judgement = tcb::judge_live(quote, &accept, latest, &policy.anchor, now, &tcb_policy)
        .map_err(|e| refuse("tcb", format!("live Intel TCB: {e}")))?;
    let (platform, qe) = judgement.statuses();
    let disclosure = TcbDisclosure::intel(
        &judgement,
        tcb_policy.min_tcb_evaluation_data_number,
        tcb_policy.max_collateral_age_secs,
        now,
    );
    // Which CRL covers the PCK, named by its issuer, as the relay asks it
    // before fetching: an issuer other than Intel's PCK CAs has none.
    tcb::pck_crl_ca(quote).map_err(|e| refuse("revocation", format!("platform identity: {e}")))?;
    let crl = b64_capped("pck_crl_der_b64", &collateral.pck_crl_der_b64, cap)
        .map_err(|e| refuse_with("revocation", 76, e))?;
    tcb::verify_pck_crl(quote, &crl, &policy.anchor, now, &tcb_policy).map_err(revocation)?;
    // What the root issued (the PCK CA, the collateral signers) against the
    // root's own CRL, as the relay reads it: every chain relayed, the one
    // accepted on and `early` when it came. Unusable or listing: refused.
    let root_crl = b64_capped("root_ca_crl_der_b64", &collateral.root_ca_crl_der_b64, cap)
        .map_err(|e| refuse_with("revocation", 76, e))?;
    let mut signer_chains: Vec<&str> = vec![accept.tcb_chain_pem, accept.qe_chain_pem];
    if let (TcbMode::Transparent, Some(e)) = (tcb_policy.mode, &early) {
        signer_chains.extend([e.tcb_chain_pem, e.qe_chain_pem]);
    }
    tcb::verify_root_crl(quote, &signer_chains, &root_crl, &policy.anchor, now)
        .map_err(revocation)?;

    // The registers, on a platform Intel's collateral accepted: the relay's
    // order, so the same evidence meets the same first refusal. The caller's
    // pins come last, on top of the reference.
    let endorsement = endorsed_mrtd(&claims.mr_td, collateral, policy, now)?;
    let firmware = endorsed_firmware(&claims.mr_td, collateral, &endorsement, policy)?;
    let image = checked_image(&claims, ccel, &firmware, policy)?;
    if let Some(pin) = policy.mrtd_pin.filter(|pin| *pin != claims.mr_td) {
        return Err(refuse(
            "measurement",
            format!(
                "MRTD {} is not the pinned {}",
                hex(&claims.mr_td),
                hex(&pin)
            ),
        ));
    }
    for (i, pin) in policy.rtmr_pins.iter().enumerate() {
        let got = &claims.rtmr[i + 1];
        if let Some(want) = pin.as_ref().filter(|want| *want != got) {
            return Err(refuse(
                "measurement",
                format!("RTMR{} {} is not the pinned {}", i + 1, hex(got), hex(want)),
            ));
        }
    }
    if let Some(want) = policy.rtmr0_pin.filter(|want| *want != claims.rtmr[0]) {
        return Err(refuse(
            "measurement",
            format!(
                "RTMR0 {} is not the pinned {}",
                hex(&claims.rtmr[0]),
                hex(&want)
            ),
        ));
    }
    debug_assert!(image.rtmr0_pinned);

    Ok(VerifiedQuote {
        tcb_platform: platform.as_str().to_string(),
        tcb_qe: qe.as_str().to_string(),
        tcb: disclosure,
        mrtd_hex: hex(&claims.mr_td),
        rtmr0_hex: image.rtmr0_hex,
        firmware_sha384_hex: firmware.sha384_hex,
        pck_sha256_hex: chain.pck_sha256_hex,
        image: format!(
            "{} {}",
            policy.reference.reference.image.name, policy.reference.reference.image.version
        ),
        warnings: image.warnings,
    })
}

/// One of Intel's two answers, its four fields under the cap: `None` when
/// none was relayed (all four empty), refused when only some were.
fn relayed<'a>(
    update: &str,
    fields: [(&str, &'a String); 4],
) -> Result<Option<tcb::Collateral<'a>>, Refusal> {
    if fields.iter().all(|(_, v)| v.is_empty()) {
        return Ok(None);
    }
    for (name, v) in &fields {
        capped("tcb", name, v, MAX_COLLATERAL_BYTES)?;
        if v.is_empty() {
            return Err(refuse(
                "tcb",
                format!("Intel's update={update} collateral relayed without {name}"),
            ));
        }
    }
    let [t, tc, q, qc] = fields.map(|(_, v)| v.as_str());
    Ok(Some(tcb::Collateral {
        tcb_body: t,
        tcb_chain_pem: tc,
        qe_body: q,
        qe_chain_pem: qc,
    }))
}

/// SHA-256 of a certificate's SubjectPublicKeyInfo, hex: what a reference's
/// `mrtd.root_spki_sha256` names (the relay's `tls::spki_sha256_of_cert`).
fn spki_sha256_hex(der: &[u8]) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use x509_parser::prelude::FromDer;
    let (_, cert) = x509_parser::certificate::X509Certificate::from_der(der)
        .map_err(|e| format!("certificate: {e}"))?;
    Ok(hex(&Sha256::digest(cert.public_key().raw)))
}

/// `measurement`, the MRTD: Google's launch endorsement of exactly this
/// firmware, signed under the Google root the reference names, for the
/// caller's machine type (the relay's step 5, `remote_verify.rs`).
fn endorsed_mrtd(
    mrtd: &[u8; 48],
    collateral: &Collateral,
    policy: &Policy,
    now: u64,
) -> Result<crate::gce_endorsement::Endorsement, Refusal> {
    let reference = &policy.reference.reference;
    let root_spki = spki_sha256_hex(&policy.gce_root_der)
        .map_err(|e| refuse_with("measurement", 89, format!("Google root: {e}")))?;
    if reference.mrtd.root_spki_sha256 != root_spki {
        return Err(refuse_with(
            "measurement",
            89,
            format!(
                "reference {} {} names Google root {} but this policy pins {root_spki}",
                reference.image.name, reference.image.version, reference.mrtd.root_spki_sha256
            ),
        ));
    }
    let blob = b64_capped(
        "gce_endorsement_b64",
        &collateral.gce_endorsement_b64,
        MAX_COLLATERAL_BYTES,
    )
    .map_err(|e| refuse("measurement", e))?;
    if blob.is_empty() {
        return Err(refuse(
            "measurement",
            format!(
                "MRTD {} has no Google launch endorsement in the collateral",
                hex(mrtd)
            ),
        ));
    }
    let endorsement = crate::gce_endorsement::verify(&blob, &policy.gce_root_der, now)
        .map_err(|e| refuse("measurement", format!("Google endorsement: {e}")))?;
    crate::reference::check_mrtd(reference, &policy.machine_type, &endorsement, mrtd)
        .map_err(|e| refuse("measurement", format!("MRTD: {e}")))?;
    Ok(endorsement)
}

/// `measurement`, the firmware: the binary `endorsement` names, relayed in
/// the collateral, is the one Google signed (its SHA-384) and measures the
/// quoted MRTD, as the relay checks the binary it fetches
/// (`remote_verify::endorsed_firmware`, `tdvf::verify_endorsed`).
fn endorsed_firmware(
    mrtd: &[u8; 48],
    collateral: &Collateral,
    endorsement: &crate::gce_endorsement::Endorsement,
    policy: &Policy,
) -> Result<crate::tdvf::EndorsedFirmware, Refusal> {
    #[cfg(feature = "test-anchor")]
    if let Some(m) = &policy.firmware_measured_elsewhere {
        // Named, test-anchor only: the measurement stands in for the binary,
        // the endorsement is still held to it and to the quoted MRTD.
        if endorsement.uefi_sha384_hex != m.sha384_hex {
            return Err(refuse(
                "measurement",
                format!(
                    "firmware: the endorsement names binary {}, the one measured elsewhere is {}",
                    endorsement.uefi_sha384_hex, m.sha384_hex
                ),
            ));
        }
        if endorsement.tdx_measurement(mrtd).is_none() {
            return Err(refuse(
                "measurement",
                format!(
                    "firmware: the endorsement does not list the quoted MRTD {}",
                    hex(mrtd)
                ),
            ));
        }
        return Ok(crate::tdvf::EndorsedFirmware {
            sha384_hex: m.sha384_hex.clone(),
            mrtd: *mrtd,
            cfv_sha384: m.cfv_sha384,
            tdx_svn: endorsement.tdx_svn,
        });
    }
    #[cfg(not(feature = "test-anchor"))]
    let _ = policy;
    let binary = b64_capped(
        "gce_firmware_b64",
        &collateral.gce_firmware_b64,
        MAX_FIRMWARE_BYTES,
    )
    .map_err(|e| refuse("measurement", e))?;
    if binary.is_empty() {
        return Err(refuse(
            "measurement",
            format!(
                "firmware: the binary Google's endorsement names ({}) was not relayed \
                 (gce_firmware_b64 empty): RTMR0 cannot be pinned",
                endorsement.uefi_sha384_hex
            ),
        ));
    }
    crate::tdvf::verify_endorsed(&binary, endorsement, mrtd)
        .map_err(|e| refuse("measurement", format!("firmware: {e}")))
}

/// `measurement`, RTMR0..3: the TD's log replayed into the quote's registers,
/// each RTMR1 and RTMR2 event on the reference's closed list, RTMR3 zero,
/// RTMR0 pinned to the reference's host events and `firmware`'s
/// configuration volume (the relay's step 6, `reference::check_image_with`).
fn checked_image(
    claims: &x0ne_confidential::tdx_quote::TdxQuoteClaims,
    ccel: Result<&[u8], &String>,
    firmware: &crate::tdvf::EndorsedFirmware,
    policy: &Policy,
) -> Result<crate::reference::ImageReport, Refusal> {
    let log = ccel.map_err(|e| refuse("measurement", e.as_str()))?;
    if log.len() > MAX_LOG_BYTES {
        return Err(refuse(
            "measurement",
            format!("CCEL: {} bytes, cap {MAX_LOG_BYTES}", log.len()),
        ));
    }
    let events =
        crate::ccel::parse(log).map_err(|e| refuse("measurement", format!("CCEL: {e}")))?;
    crate::reference::check_image_with(
        &policy.reference.reference,
        &policy.machine_type,
        claims,
        &events,
        firmware,
    )
    .map_err(|e| refuse("measurement", format!("image: {e}")))
}

/// Which PCK CRL the quote's PCK falls under (`platform` or `processor`):
/// what a carrier fetches for [`Collateral::pck_crl_der_b64`].
pub fn pck_crl_ca(quote: &[u8]) -> Result<&'static str, String> {
    tcb::pck_crl_ca(quote).map_err(|e| e.to_string())
}

/// Where a carrier fetches [`Collateral::root_ca_crl_der_b64`]: the relay's
/// own GET (`tcb::INTEL_SGX_ROOT_CA_CRL_URL`).
pub const ROOT_CA_CRL_URL: &str = tcb::INTEL_SGX_ROOT_CA_CRL_URL;

/// Where a carrier fetches [`Collateral::gce_endorsement_b64`] for the
/// quote's MRTD: the relay's own GET (`gce_endorsement::url`). A 404 is
/// relayed as an empty string, refused at `measurement` as the relay does.
pub fn gce_endorsement_url(quote: &[u8]) -> Result<String, String> {
    let claims = x0ne_confidential::tdx_quote::parse(quote).map_err(|e| format!("{e:?}"))?;
    Ok(crate::gce_endorsement::url(
        crate::gce_endorsement::BUCKET_BASE_URL,
        &claims.mr_td,
    ))
}

/// Where a carrier fetches [`Collateral::gce_firmware_b64`]: the binary the
/// endorsement blob ([`Collateral::gce_endorsement_b64`], decoded) names by
/// its SHA-384, in Google's bucket, where the relay fetches its own
/// (`tdvf::fetch`). The blob is read, not verified: the caller verifies it.
pub fn gce_firmware_url(endorsement_blob: &[u8]) -> Result<String, String> {
    let digest = crate::gce_endorsement::uefi_sha384_hex_unverified(endorsement_blob)?;
    Ok(format!(
        "{}/{}",
        crate::gce_endorsement::BUCKET_BASE_URL,
        crate::tdvf::object_name(&digest)?
    ))
}

/// The FMSPC of the quote's platform, hex: what a carrier fetches TCB info for.
pub fn fmspc_hex(quote: &[u8]) -> Result<String, String> {
    tcb::pck_tcb(quote)
        .map(|p| p.fmspc_hex)
        .map_err(|e| e.to_string())
}
