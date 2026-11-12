//! The TD side of the attested channel: collect, verify, publish the
//! attestation document.
//!
//! Same pipeline as `main.rs` runs for `--listen`, behind a trait so the
//! whole cloud path — relay against server — is exercised on synthetic
//! evidence signed by test anchors. Production anchors come from
//! [`TdAnchors::production`](crate::attest_v2::TdAnchors::production) and nowhere else (`tests/cloud_source_sweep.rs`).

use crate::binding::{BindingV2, SessionIdentity};
use crate::cloud_config::{hex, Loaded};
use crate::evidence::{Collected, EvidenceSource};
use crate::gate::{Attested, Gate};
use crate::pcs::TrustAnchor;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use x0ne_confidential::model::AttestationEvidence;
use x0ne_confidential::policy::VerificationPolicy;

pub const SCHEMA: &str = "x0ne-infer/attestation/v2";
/// `GET /v2/attestation/session`: one quote per TLS connection (`BindingV3`)
/// and, with the GPU product, that connection's GPU evidence
/// (`binding::session_gpu_nonce`).
pub const SESSION_SCHEMA: &str = "x0ne-infer/session-attestation/v4";

/// The quote of THIS connection. The TD derives the exporter itself; nothing
/// in the request feeds `report_data` nor the GPU nonce.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionDoc {
    pub schema: String,
    /// The TDX quote; empty on an SEV-SNP guest, whose evidence is `snp`.
    pub quote_b64: String,
    /// SEV-SNP: this connection's report and its VCEK or VLEK.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snp: Option<SnpDoc>,
    /// The GPU product: NRAS tokens answering the nonce derived from this
    /// connection's exporter and this quote. Absent for the cpu product.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<SessionGpuDoc>,
}

/// The GPU half of one connection's attestation, as the document's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionGpuDoc {
    pub gpu: Vec<GpuDoc>,
    pub nras: NrasDoc,
}

impl SessionGpuDoc {
    pub fn of(attested: &GpuAttested) -> Self {
        let (gpu, nras) = gpu_docs(attested);
        Self { gpu, nras }
    }
}

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// What the NVIDIA half produced, verbatim: what was POSTed and what came back.
#[derive(Debug, Clone)]
pub struct GpuAttested {
    pub overall_token: String,
    /// `("GPU-0", token)`, as NRAS names them.
    pub per_gpu: Vec<(String, String)>,
    pub jwks: serde_json::Value,
    /// The evidence POSTed, per GPU; the certificate chain is what names a GPU.
    pub evidence: Vec<crate::nras_client::GpuEvidence>,
    pub local_serials: Vec<String>,
    /// OCSP responses about the JWKS signing chain, DER, for the relay and
    /// the SDK to judge (`nras_ocsp`). Empty: this TD fetches none yet.
    pub ocsp: Vec<Vec<u8>>,
}

/// NVIDIA's NRAS verdict on one e2e key document's GPU evidence: NRAS's
/// answer, `[["JWT", <overall>], {"GPU-0": <token>, ...}]`, and its JWKS.
#[derive(Debug, Clone)]
pub struct NrasVerdict {
    pub answer: serde_json::Value,
    pub jwks: serde_json::Value,
}

/// NRAS's unified-EAT answer from its parts, as NRAS sends it.
pub fn nras_answer(overall_token: &str, per_gpu: &[(String, String)]) -> serde_json::Value {
    let gpus: serde_json::Map<String, serde_json::Value> = per_gpu
        .iter()
        .map(|(gpu, token)| (gpu.clone(), serde_json::Value::String(token.clone())))
        .collect();
    serde_json::json!([["JWT", overall_token], gpus])
}

/// NVML's confidential-computing state of the system
/// (`nvmlSystemGetConfComputeState`, `nvmlConfComputeSystemState_t`:
/// <https://docs.nvidia.com/deploy/nvml-api/api/structnvmlConfComputeSystemState__t.html>).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CcSystemState {
    pub environment: u32,
    pub cc_feature: u32,
    pub dev_tools_mode: u32,
}

impl CcSystemState {
    /// `NVML_CC_SYSTEM_ENVIRONMENT_PROD` (2), `NVML_CC_SYSTEM_FEATURE_ENABLED`
    /// (1), `NVML_CC_SYSTEM_DEVTOOLS_MODE_OFF` (0): nvml.h's values, as
    /// nvidia-ml-py 12.570.86 copies them.
    pub const PRODUCTION: Self = Self {
        environment: 2,
        cc_feature: 1,
        dev_tools_mode: 0,
    };
}

/// The e2e key document's GPU evidence is read only from GPUs in
/// confidential computing, in its production environment, devtools off;
/// any other value refuses, code 74. Not yet measured on an H100.
pub fn check_cc_state(state: &CcSystemState) -> Result<(), GpuError> {
    // SECURITY: devtools mode still gives reports, judged by NRAS alone,
    // and no NRAS claim names the mode (review of 2026-10-27, finding 3).
    if *state == CcSystemState::PRODUCTION {
        return Ok(());
    }
    Err(GpuError {
        code: 74,
        message: format!(
            "NVML CC state: environment {}, CC feature {}, devtools {}; \
             production CC with devtools off is required",
            state.environment, state.cc_feature, state.dev_tools_mode
        ),
    })
}

/// What NVML says of this machine's GPUs: [`judged_gpu_evidence`] reads it
/// from the driver (`nvml::NvmlLib`), the tests from a simulation.
pub trait LocalGpus {
    /// `nvmlSystemGetConfComputeState`.
    fn cc_system_state(&self) -> anyhow::Result<CcSystemState>;
    /// `nvmlDeviceGetArchitecture` of each GPU, in index order.
    fn architectures(&self) -> anyhow::Result<Vec<u32>>;
    /// Each GPU's report for `nonce` and its certificate chain.
    fn read_evidence(
        &self,
        nonce: &[u8; 32],
    ) -> anyhow::Result<Vec<crate::nras_client::GpuEvidence>>;
}

/// The architectures this build sends to NRAS as `arch`.
pub const NRAS_ARCHS: [&str; 2] = ["HOPPER", "BLACKWELL"];

/// NRAS's `arch` for NVML's `nvmlDeviceArchitecture_t`: nvml.h of CUDA 13.3
/// gives NVML_DEVICE_ARCH_HOPPER 9 and NVML_DEVICE_ARCH_BLACKWELL 10. Any
/// other value has no NRAS name here.
pub fn nras_arch(nvml_arch: u32) -> Option<&'static str> {
    match nvml_arch {
        9 => Some(NRAS_ARCHS[0]),
        10 => Some(NRAS_ARCHS[1]),
        _ => None,
    }
}

/// Every GPU NVML lists is of the architecture `--gpu-arch` declares, the
/// one the NRAS request names; another, an unknown one or none refuses (74).
/// A claim that names no architecture is not printed: argv may hold what
/// was meant for another flag.
pub fn check_gpu_arch(declared: &str, archs: &[u32]) -> Result<(), GpuError> {
    let fail = |message: String| GpuError { code: 74, message };
    let claim = if NRAS_ARCHS.contains(&declared) {
        format!("--gpu-arch {declared}")
    } else {
        format!(
            "--gpu-arch names none of {} (case matters)",
            NRAS_ARCHS.join(", ")
        )
    };
    if archs.is_empty() {
        return Err(fail(format!("NVML names no GPU architecture; {claim}")));
    }
    for (i, &a) in archs.iter().enumerate() {
        match nras_arch(a) {
            Some(name) if name == declared => {}
            Some(name) => {
                return Err(fail(format!(
                    "GPU {i} is of the {name} architecture by NVML; {claim}"
                )))
            }
            None => {
                return Err(fail(format!(
                    "GPU {i}: NVML architecture {a}, none of {}; {claim}",
                    NRAS_ARCHS.join(", ")
                )))
            }
        }
    }
    Ok(())
}

/// The GPUs' evidence, read only once NVML says confidential computing is
/// on, in production, devtools off, and every GPU is of the declared
/// architecture. The one path to GPU evidence: `run`, and the cloud TD's
/// session and key document. Each refusal is 74, before any report is read.
pub fn judged_gpu_evidence(
    gpus: &dyn LocalGpus,
    nonce: &[u8; 32],
    declared_arch: &str,
) -> Result<Vec<crate::nras_client::GpuEvidence>, GpuError> {
    let fail = |message: String| GpuError { code: 74, message };
    let state = gpus
        .cc_system_state()
        .map_err(|e| fail(format!("NVML CC state unreadable: {e:#}")))?;
    check_cc_state(&state)?;
    let archs = gpus
        .architectures()
        .map_err(|e| fail(format!("NVML GPU architecture unreadable: {e:#}")))?;
    check_gpu_arch(declared_arch, &archs)?;
    let evidence = gpus
        .read_evidence(nonce)
        .map_err(|e| fail(format!("GPU evidence collection failed: {e:#}")))?;
    if evidence.len() != archs.len() {
        return Err(fail(format!(
            "NVML gave evidence for {} GPU(s) after naming {} GPUs",
            evidence.len(),
            archs.len()
        )));
    }
    Ok(evidence)
}

#[derive(Debug)]
pub struct GpuError {
    /// 74 collection (the CC state included), 75 NRAS, 78 malformed request,
    /// 83 local certificate, 103 NRAS refused the key.
    pub code: i32,
    pub message: String,
}

/// x0ne-infer's exit when NRAS refuses the key of `--nras-key-file` (401 or
/// 403): the image passes the same key at every start, 10 s apart, so the
/// refusal is final. 75 called NVIDIA in a loop (review of 2026-10-30).
pub const EXIT_NRAS_KEY_REFUSED: i32 = 103;

/// A failed NRAS call: [`EXIT_NRAS_KEY_REFUSED`] when NRAS refused the key it
/// was sent; 75 otherwise (no answer, 5xx, 429, a 401 or 403 without a key),
/// which may pass on a later call.
fn nras_failed(e: &anyhow::Error) -> GpuError {
    if e.downcast_ref::<crate::nras_http::KeyRefused>().is_some() {
        return GpuError {
            code: EXIT_NRAS_KEY_REFUSED,
            message: format!("NRAS refused the key of --nras-key-file: {e:#}"),
        };
    }
    GpuError {
        code: 75,
        message: format!("NRAS attestation failed: {e:#}"),
    }
}

/// Where the TD's evidence comes from. Production: configfs-tsm, the ACPI
/// CCEL table and NVML + NRAS. Tests: synthetic, signed by test anchors.
pub trait TdEvidence: Send + Sync {
    /// The CPU evidence for `report_data`: a TDX quote, or on an SEV-SNP
    /// guest a report with its certificate (`evidence::ConfigfsTsm`).
    fn tdx(&self, report_data: &[u8; 64], now_unix: u64) -> anyhow::Result<Collected>;
    fn ccel(&self) -> anyhow::Result<Vec<u8>>;
    /// AMD's CRL for the SEV-SNP report's product. A source without one
    /// refuses: an SNP guest does not serve on an unchecked chain.
    fn snp_crl(&self) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("this evidence source has no AMD CRL")
    }
    fn gpu<'a>(&'a self, gpu_nonce: &'a [u8; 32]) -> BoxFuture<'a, Result<GpuAttested, GpuError>>;
    /// Each GPU's report for `gpu_nonce` and its certificate chain, read
    /// over NVML, NRAS not called: the e2e key document's evidence, whose
    /// NRAS verdict the caller's carrier fetches. Blocking. A source without
    /// GPU evidence refuses.
    fn gpu_evidence(
        &self,
        gpu_nonce: &[u8; 32],
    ) -> Result<Vec<crate::nras_client::GpuEvidence>, GpuError> {
        let _ = gpu_nonce;
        Err(GpuError {
            code: 74,
            message: "this evidence source collects no GPU evidence".into(),
        })
    }
    /// NVIDIA's NRAS verdict on `evidence` (what [`Self::gpu_evidence`] read
    /// for `gpu_nonce`), with NRAS's JWKS: the e2e key document carries
    /// both, as a session's attestation document does. Not checked here:
    /// the caller checks it. A source without NRAS refuses.
    fn nras<'a>(
        &'a self,
        gpu_nonce: &'a [u8; 32],
        evidence: &'a [crate::nras_client::GpuEvidence],
    ) -> BoxFuture<'a, Result<NrasVerdict, GpuError>> {
        let _ = (gpu_nonce, evidence);
        Box::pin(async {
            Err(GpuError {
                code: 75,
                message: "this evidence source calls no NRAS".into(),
            })
        })
    }
}

/// A TD source seen as an [`EvidenceSource`]: its TDX quote only.
pub struct TdQuotes(pub std::sync::Arc<dyn TdEvidence>);

impl EvidenceSource for TdQuotes {
    fn collect(&self, nonce: &[u8; 64], now_unix: u64) -> anyhow::Result<Collected> {
        self.0.tdx(nonce, now_unix)
    }
}

/// The production source.
pub struct SystemEvidence {
    pub nras_url: String,
    pub nras_jwks_url: String,
    /// `--nras-key-file`: a Bearer on the attest call; the JWKS stays anonymous.
    pub nras_key: Option<crate::nras_http::NrasKey>,
    pub gpu_arch: String,
}

impl TdEvidence for SystemEvidence {
    fn tdx(&self, report_data: &[u8; 64], now_unix: u64) -> anyhow::Result<Collected> {
        crate::evidence::ConfigfsTsm::system().collect(report_data, now_unix)
    }

    fn ccel(&self) -> anyhow::Result<Vec<u8>> {
        let log = crate::ccel::read_system()
            .map_err(|e| anyhow::anyhow!("{}: {e}", crate::ccel::CCEL_DATA_PATH))?;
        // Parsed once here so an unreadable log refuses before it is served.
        crate::ccel::parse(&log).map_err(|e| anyhow::anyhow!("CCEL: {e}"))?;
        Ok(log)
    }

    fn snp_crl(&self) -> anyhow::Result<Vec<u8>> {
        crate::evidence::read_snp_crl(std::path::Path::new(crate::evidence::SNP_CRL_FILE))
    }

    fn gpu<'a>(&'a self, gpu_nonce: &'a [u8; 32]) -> BoxFuture<'a, Result<GpuAttested, GpuError>> {
        Box::pin(async move {
            #[cfg(not(feature = "nvml"))]
            {
                let _ = gpu_nonce;
                Err(GpuError {
                    code: 74,
                    message: "this build has no NVML support (--features nvml): no GPU evidence"
                        .into(),
                })
            }
            #[cfg(feature = "nvml")]
            {
                let fail = |code, message: String| GpuError { code, message };
                let lib = crate::nvml::NvmlLib::open()
                    .map_err(|e| fail(74, format!("GPU evidence collection failed: {e:#}")))?;
                let evidence = judged_gpu_evidence(&lib, gpu_nonce, &self.gpu_arch)?;
                let request = crate::nras_client::build_attest_request(
                    &hex(gpu_nonce),
                    evidence,
                    &self.gpu_arch,
                )
                .map_err(|e| fail(78, format!("NRAS request refused before sending: {e:#}")))?;
                let local_serials = request
                    .evidence_list
                    .iter()
                    .map(|e| crate::nras::gpu_leaf_serial_decimal(&e.certificate))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| fail(83, format!("local GPU certificate unreadable: {e}")))?;
                // Together: neither waits for the other's answer, and both
                // were always awaited; the attest call's failure comes first.
                let (attested, jwks) = tokio::join!(
                    crate::nras_http::attest_gpu(&self.nras_url, &request, self.nras_key.as_ref()),
                    crate::nras_http::fetch_jwks(&self.nras_jwks_url),
                );
                match (attested, jwks) {
                    (Ok((overall_token, per_gpu)), Ok(jwks)) => Ok(GpuAttested {
                        overall_token,
                        per_gpu,
                        jwks,
                        evidence: request.evidence_list,
                        local_serials,
                        ocsp: Vec::new(),
                    }),
                    (Err(e), _) | (_, Err(e)) => Err(nras_failed(&e)),
                }
            }
        })
    }

    fn gpu_evidence(
        &self,
        gpu_nonce: &[u8; 32],
    ) -> Result<Vec<crate::nras_client::GpuEvidence>, GpuError> {
        #[cfg(not(feature = "nvml"))]
        {
            let _ = gpu_nonce;
            Err(GpuError {
                code: 74,
                message: "this build has no NVML support (--features nvml): no GPU evidence".into(),
            })
        }
        #[cfg(feature = "nvml")]
        {
            let fail = |code, message: String| GpuError { code, message };
            let lib = crate::nvml::NvmlLib::open()
                .map_err(|e| fail(74, format!("GPU evidence collection failed: {e:#}")))?;
            let evidence = judged_gpu_evidence(&lib, gpu_nonce, &self.gpu_arch)?;
            if evidence.is_empty() {
                return Err(fail(74, "NVML returned no GPU evidence".into()));
            }
            // The serial the caller binds each token's ueid to: unreadable, not served.
            for e in &evidence {
                crate::nras::gpu_leaf_serial_decimal(&e.certificate)
                    .map_err(|e| fail(83, format!("local GPU certificate unreadable: {e}")))?;
            }
            Ok(evidence)
        }
    }

    fn nras<'a>(
        &'a self,
        gpu_nonce: &'a [u8; 32],
        evidence: &'a [crate::nras_client::GpuEvidence],
    ) -> BoxFuture<'a, Result<NrasVerdict, GpuError>> {
        Box::pin(async move {
            let fail = |code, message: String| GpuError { code, message };
            let request = crate::nras_client::build_attest_request(
                &hex(gpu_nonce),
                evidence.to_vec(),
                &self.gpu_arch,
            )
            .map_err(|e| fail(78, format!("NRAS request refused before sending: {e:#}")))?;
            // Together: neither waits for the other's answer, and both
            // were always awaited; the attest call's failure comes first.
            let (attested, jwks) = tokio::join!(
                crate::nras_http::attest_gpu(&self.nras_url, &request, self.nras_key.as_ref()),
                crate::nras_http::fetch_jwks(&self.nras_jwks_url),
            );
            match (attested, jwks) {
                (Ok((overall, per_gpu)), Ok(jwks)) => Ok(NrasVerdict {
                    answer: nras_answer(&overall, &per_gpu),
                    jwks,
                }),
                (Err(e), _) | (_, Err(e)) => Err(nras_failed(&e)),
            }
        })
    }
}

/// The roots the TD verifies itself against before it serves. The client
/// re-verifies with its own copies.
pub struct TdAnchors {
    pub tdx: TrustAnchor,
    pub nras_intermediate_pins: Vec<crate::nras::NrasIntermediatePin>,
    /// The minimum TCB evaluation and the maximum age of Intel's collateral.
    pub tcb_policy: crate::tcb::TcbPolicy,
    /// Where the pinned root publishes its CRL (its CRL distribution point).
    pub root_crl_url: String,
    /// The AMD roots an SEV-SNP guest verifies its own report against.
    pub snp: crate::snp::SnpAnchors,
    /// Its TCB minimum and Milan stance: the defaults (`snp::SnpPolicy`).
    pub snp_policy: crate::snp::SnpPolicy,
}

impl TdAnchors {
    pub fn production() -> Self {
        Self {
            tdx: TrustAnchor::intel_production(),
            nras_intermediate_pins: crate::nras::NVIDIA_NRAS_GPU_INTERMEDIATE_PINS.to_vec(),
            tcb_policy: crate::tcb::TcbPolicy::production(),
            root_crl_url: crate::tcb::INTEL_SGX_ROOT_CA_CRL_URL.into(),
            snp: crate::snp::SnpAnchors::production(),
            snp_policy: crate::snp::SnpPolicy {
                tcb_mode: crate::tcb::TcbPolicy::production().mode,
                ..crate::snp::SnpPolicy::default()
            },
        }
    }
}

impl Clone for TdAnchors {
    fn clone(&self) -> Self {
        Self {
            tdx: match &self.tdx {
                TrustAnchor::SpkiSha256Hex(s) => TrustAnchor::SpkiSha256Hex(s.clone()),
                TrustAnchor::RootDer(d) => TrustAnchor::RootDer(d.clone()),
            },
            nras_intermediate_pins: self.nras_intermediate_pins.clone(),
            tcb_policy: self.tcb_policy,
            root_crl_url: self.root_crl_url.clone(),
            snp: self.snp.clone(),
            snp_policy: self.snp_policy.clone(),
        }
    }
}

/// Why the TD does not serve: an exit code from the `main.rs` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub code: i32,
    pub message: String,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (exit {})", self.message, self.code)
    }
}

/// One successful run of the pipeline: the proof the gate issued and every
/// piece of evidence it was issued on, verbatim.
#[derive(Debug, Clone)]
pub struct Attestation {
    pub proof: Attested,
    pub evidence: AttestationEvidence,
    pub ccel: Vec<u8>,
    pub gpu: Option<GpuAttested>,
    /// Live Intel PCS statuses (platform, QE) at the time of the verdict.
    pub live_tcb: (String, String),
    /// The platform's patch level as the verdict saw it: Intel's latest
    /// collateral or AMD's bulletins ([`crate::tcb_disclosure`]).
    pub tcb_disclosure: crate::tcb_disclosure::TcbDisclosure,
}

impl Attestation {
    /// The CPU evidence's payload: the TDX quote, or the SEV-SNP report
    /// followed by its certificate.
    pub fn quote(&self) -> &[u8] {
        &self.evidence.payload
    }

    pub fn is_snp(&self) -> bool {
        self.evidence.format == x0ne_confidential::model::EvidenceFormat::SevSnpReport
    }
}

/// What the live TCB pair says on an SEV-SNP guest: AMD has no live
/// service like Intel PCS; the report's TCB is held to `SnpPolicy`.
pub const SNP_LIVE_TCB: &str = "not applicable (AMD SEV-SNP)";

/// Collect for `binding`, verify to `anchors` with live PCS collateral from
/// `pcs_base_url`, pass the gate. Clock read at each decision, as `main.rs`.
pub async fn attest_td(
    source: &dyn TdEvidence,
    binding: &BindingV2,
    gpu_required: bool,
    pcs_base_url: &str,
    anchors: &TdAnchors,
    now: &(dyn Fn() -> u64 + Sync),
) -> Result<Attestation, Failure> {
    let fail = |code, message: String| Failure { code, message };
    let report_data = binding.tdx_report_data();
    let collected = source
        .tdx(&report_data, now())
        .map_err(|e| fail(64, format!("no evidence, not serving: {e:#}")))?;
    if collected.evidence.format == x0ne_confidential::model::EvidenceFormat::SevSnpReport {
        return attest_snp(source, binding, collected, gpu_required, anchors, now).await;
    }
    let ccel = source
        .ccel()
        .map_err(|e| fail(64, format!("no event log, not serving: {e:#}")))?;
    let gpu_nonce = binding.gpu_nonce();
    let gpu = if gpu_required {
        Some(
            source
                .gpu(&gpu_nonce)
                .await
                .map_err(|e| fail(e.code, e.message))?,
        )
    } else {
        None
    };
    let policy = VerificationPolicy {
        workload_name: "x0ne-infer".into(),
        accepted_cpu: vec![x0ne_confidential::model::TeeTechnology::IntelTdx],
        require_gpu_cc_on: gpu_required,
        // The client pins the image; the TD has no reference to pin against.
        expected_measurements: vec![],
    };
    let quote = &collected.evidence.payload;
    // Intel's collateral under the policy's mode (`tcb_standing`): fetch,
    // signatures, statuses, the PCK CRL and the root CA CRL.
    let judgement = crate::tcb_standing::judge_platform_live(
        quote,
        pcs_base_url,
        &anchors.root_crl_url,
        &anchors.tdx,
        &anchors.tcb_policy,
        now(),
    )
    .await
    .map_err(|e| match e {
        crate::tcb_standing::LiveFailure::Refused(
            r @ (crate::tcb::TcbRejected::PckRevoked { .. }
            | crate::tcb::TcbRejected::CaRevoked { .. }),
        ) => fail(65, format!("not serving: {r}")),
        crate::tcb_standing::LiveFailure::Fetch(m) => fail(
            76,
            format!("live TCB collateral unreachable, not serving: {m}"),
        ),
        e => fail(76, format!("live TCB verification failed: {e}")),
    })?;
    let live = judgement.statuses();
    let tcb_disclosure = crate::tcb_disclosure::TcbDisclosure::intel(
        &judgement,
        anchors.tcb_policy.min_tcb_evaluation_data_number,
        anchors.tcb_policy.max_collateral_age_secs,
        now(),
    );
    let now_unix = now();
    let verdict = match &gpu {
        Some(g) => {
            let composite = crate::verify::CompositeEvidence {
                tdx: &collected.evidence,
                nras_token: &g.overall_token,
                nras_jwks: &g.jwks,
                nonce_hex: &hex(&gpu_nonce),
                expected_tdx_report_data: &report_data,
                nras_per_gpu: &g.per_gpu,
                local_gpu_serials: &g.local_serials,
                nras_ocsp_responses: &g.ocsp,
                nras_ocsp: &crate::nras_ocsp::NrasOcspPolicy::production(),
            };
            crate::verify::cryptographic_with_anchors(
                &composite,
                &policy,
                now_unix,
                &anchors.tdx,
                &anchors.nras_intermediate_pins,
                Some((live.0.as_str(), live.1.as_str())),
            )
        }
        None => crate::verify::cryptographic_tdx_after_live_tcb_with_anchor(
            &collected.evidence,
            &policy,
            now_unix,
            &anchors.tdx,
            &report_data,
            &live.0,
            &live.1,
        ),
    };
    let proof = Gate::attest(
        collected.environment,
        &collected.evidence,
        verdict,
        now_unix,
    )
    .map_err(|refused| {
        fail(
            refused.exit_code(),
            format!("gate refused, not serving: {refused}"),
        )
    })?;
    Ok(Attestation {
        proof,
        evidence: collected.evidence,
        ccel,
        gpu,
        live_tcb: (live.0.as_str().to_string(), live.1.as_str().to_string()),
        tcb_disclosure,
    })
}

/// [`attest_td`] on an SEV-SNP guest: no event log (the launch measurement
/// is the report's), no Intel PCS; the report verified to a pinned AMD root
/// and AMD's CRL (none: no service), and with the GPU product the NRAS token
/// for the same binding.
async fn attest_snp(
    source: &dyn TdEvidence,
    binding: &BindingV2,
    collected: Collected,
    gpu_required: bool,
    anchors: &TdAnchors,
    now: &(dyn Fn() -> u64 + Sync),
) -> Result<Attestation, Failure> {
    let fail = |code, message: String| Failure { code, message };
    let report_data = binding.snp_report_data();
    let gpu_nonce = binding.gpu_nonce();
    let crl = source
        .snp_crl()
        .map_err(|e| fail(76, format!("AMD CRL unreadable, not serving: {e:#}")))?;
    let gpu = if gpu_required {
        Some(
            source
                .gpu(&gpu_nonce)
                .await
                .map_err(|e| fail(e.code, e.message))?,
        )
    } else {
        None
    };
    let policy = VerificationPolicy {
        workload_name: "x0ne-infer".into(),
        accepted_cpu: vec![x0ne_confidential::model::TeeTechnology::AmdSevSnp],
        require_gpu_cc_on: gpu_required,
        // The client pins the launch measurement; the guest has none to pin.
        expected_measurements: vec![],
    };
    let now_unix = now();
    let verdict = match &gpu {
        Some(g) => crate::verify::cryptographic_snp_with_gpu(
            &crate::verify::SnpCompositeEvidence {
                snp: &collected.evidence,
                expected_report_data: &report_data,
                nras_token: &g.overall_token,
                nras_jwks: &g.jwks,
                nonce_hex: &hex(&gpu_nonce),
                nras_per_gpu: &g.per_gpu,
                local_gpu_serials: &g.local_serials,
                nras_ocsp_responses: &g.ocsp,
                nras_ocsp: &crate::nras_ocsp::NrasOcspPolicy::production(),
            },
            &policy,
            &anchors.snp_policy,
            &anchors.snp,
            &crl,
            &anchors.nras_intermediate_pins,
            now_unix,
        ),
        None => crate::verify::cryptographic_snp(
            &collected.evidence,
            &policy,
            &anchors.snp_policy,
            &anchors.snp,
            &crl,
            Some(&report_data),
            now_unix,
        ),
    };
    let proof = Gate::attest(
        collected.environment,
        &collected.evidence,
        verdict,
        now_unix,
    )
    .map_err(|refused| {
        fail(
            refused.exit_code(),
            format!("gate refused, not serving: {refused}"),
        )
    })?;
    // The verdict above passed: the same report, read again for the
    // bulletins it names (imposed or not) under the same policy.
    let verified = crate::snp::verify_snp_report(
        &collected.evidence.payload,
        &crl,
        &anchors.snp,
        &anchors.snp_policy,
        now_unix,
    )
    .map_err(|e| fail(65, format!("not serving: SEV-SNP report: {e}")))?;
    Ok(Attestation {
        proof,
        evidence: collected.evidence,
        ccel: Vec::new(),
        gpu,
        live_tcb: (SNP_LIVE_TCB.into(), SNP_LIVE_TCB.into()),
        tcb_disclosure: crate::tcb_disclosure::TcbDisclosure::amd(&verified, now_unix),
    })
}

// ── The document (I2) ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BindingDoc {
    pub context: String,
    pub session_nonce_hex: String,
    pub session_pubkey_hex: String,
    pub tls_spki_sha256_hex: String,
    pub config_sha256_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TdxDoc {
    pub quote_b64: String,
    pub ccel_b64: String,
}

/// SEV-SNP evidence as `EvidenceFormat::SevSnpReport` carries it: the
/// 0x4A0-byte report, then the DER of the VCEK or VLEK that signed it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnpDoc {
    pub evidence_b64: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuDoc {
    pub index: usize,
    /// The key NRAS used for this GPU's token (`GPU-0`).
    pub nras_name: String,
    /// The attestation certificate chain as POSTed: its leaf serial is the
    /// `ueid` the token must name.
    pub certificate_b64: String,
    pub nras_token: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NrasDoc {
    pub overall_token: String,
    pub jwks: serde_json::Value,
    /// OCSP responses about the JWKS signing chain, DER base64, as the TD
    /// fetched them: judged by the relay under its `nras_ocsp` policy.
    /// Empty while no TD fetches them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ocsp_b64: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileDoc {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDoc {
    /// The catalog name the config asked for.
    pub name: String,
    /// What every receipt will carry: the backend's digest of the loaded model.
    pub digest_hex: String,
    /// The files served, each with its SHA-256: what the client compares to
    /// its own copy of the catalog.
    pub files: Vec<FileDoc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerDoc {
    pub version: String,
    pub attested_at_unix: u64,
    /// Informative: the client re-verifies everything.
    pub verdict: serde_json::Value,
    pub live_tcb: (String, String),
    /// Informative, as `verdict`: what the TD's own verdict said about the
    /// platform's patch level. Absent from a TD before 2026-10-28.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tcb: Option<crate::tcb_disclosure::TcbDisclosure>,
}

/// `GET /v2/attestation`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    pub schema: String,
    pub binding: BindingDoc,
    pub config_b64: String,
    /// Empty quote and log on an SEV-SNP guest: a TDX-only client refuses it.
    pub tdx: TdxDoc,
    /// SEV-SNP: the boot report and its certificate. Absent on TDX.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snp: Option<SnpDoc>,
    pub gpu: Vec<GpuDoc>,
    pub nras: NrasDoc,
    pub model: ModelDoc,
    pub server: ServerDoc,
}

/// What the TD says about the model it pulled: name, files and the digest
/// the backend will report (`cloud_serve` refuses to serve if it differs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelReport {
    pub name: String,
    pub digest_hex: String,
    pub files: Vec<(String, String)>,
}

pub fn b64(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub fn unb64(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| format!("base64: {e}"))
}

impl Document {
    pub fn build(
        attestation: &Attestation,
        identity: &SessionIdentity,
        binding: &BindingV2,
        config: &Loaded,
        model: &ModelReport,
        version: &str,
    ) -> Self {
        let (gpu, nras) = match &attestation.gpu {
            Some(g) => gpu_docs(g),
            None => (
                Vec::new(),
                NrasDoc {
                    overall_token: String::new(),
                    jwks: serde_json::json!({"keys": []}),
                    ocsp_b64: Vec::new(),
                },
            ),
        };
        Self {
            schema: SCHEMA.into(),
            binding: BindingDoc {
                context: String::from_utf8_lossy(crate::binding::CONTEXT_V2).into_owned(),
                session_nonce_hex: hex(&binding.session_nonce),
                session_pubkey_hex: hex(identity.public_key().as_bytes()),
                tls_spki_sha256_hex: hex(&binding.tls_spki_sha256),
                config_sha256_hex: hex(&binding.config_sha256),
            },
            config_b64: b64(&config.bytes),
            tdx: if attestation.is_snp() {
                TdxDoc {
                    quote_b64: String::new(),
                    ccel_b64: String::new(),
                }
            } else {
                TdxDoc {
                    quote_b64: b64(attestation.quote()),
                    ccel_b64: b64(&attestation.ccel),
                }
            },
            snp: attestation.is_snp().then(|| SnpDoc {
                evidence_b64: b64(attestation.quote()),
            }),
            gpu,
            nras,
            model: ModelDoc {
                name: model.name.clone(),
                digest_hex: model.digest_hex.clone(),
                files: model
                    .files
                    .iter()
                    .map(|(path, sha256)| FileDoc {
                        path: path.clone(),
                        sha256: sha256.clone(),
                    })
                    .collect(),
            },
            server: ServerDoc {
                version: version.into(),
                attested_at_unix: attestation.proof.attested_at_unix(),
                verdict: serde_json::to_value(attestation.proof.verdict()).unwrap_or_default(),
                live_tcb: attestation.live_tcb.clone(),
                tcb: Some(attestation.tcb_disclosure.clone()),
            },
        }
    }
}

/// What the NVIDIA half produced, as the documents publish it: one entry per
/// GPU the evidence was sent for, with its certificate chain and its token.
fn gpu_docs(g: &GpuAttested) -> (Vec<GpuDoc>, NrasDoc) {
    (
        g.evidence
            .iter()
            .enumerate()
            .map(|(index, ev)| GpuDoc {
                index,
                nras_name: g
                    .per_gpu
                    .get(index)
                    .map(|(n, _)| n.clone())
                    .unwrap_or_default(),
                certificate_b64: ev.certificate.clone(),
                nras_token: g
                    .per_gpu
                    .get(index)
                    .map(|(_, t)| t.clone())
                    .unwrap_or_default(),
            })
            .collect(),
        NrasDoc {
            overall_token: g.overall_token.clone(),
            jwks: g.jwks.clone(),
            ocsp_b64: g
                .ocsp
                .iter()
                .map(|r| base64::Engine::encode(&base64::engine::general_purpose::STANDARD, r))
                .collect(),
        },
    )
}

/// The catalog's view of a model as the document must list it, for the
/// client's comparison: sorted by path.
pub fn catalog_files(entry: &crate::registry::Entry) -> Vec<(String, String)> {
    let mut files: Vec<(String, String)> = entry
        .files
        .iter()
        .map(|f| (f.path.clone(), f.sha256.clone()))
        .collect();
    files.sort();
    files
}

// ── Synthetic evidence, test anchors ─────────────────────────────────────────

/// Our own Intel, NVIDIA and TD: a complete chain each, so the production
/// code runs unchanged on both sides and every refusal is a real one.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;
    use crate::ccel::Event;
    use p256::ecdsa::signature::Signer as _;
    use p256::ecdsa::{Signature, SigningKey};
    use rcgen::{
        date_time_ymd, BasicConstraints, CertificateParams, CertificateRevocationListParams,
        CustomExtension, DistinguishedName, DnType, IsCa, Issuer, KeyIdMethod, KeyPair,
        KeyUsagePurpose, RevocationReason, RevokedCertParams, SerialNumber, PKCS_ECDSA_P256_SHA256,
    };
    use sha2::{Digest, Sha256};
    use std::sync::Arc;
    use x0ne_confidential::model::{EvidenceFormat, ExecutionEnvironment, TeeTechnology};

    pub(crate) const SIGNED_LEN: usize = 48 + 584;
    pub(crate) const FMSPC_HEX: &str = "00906ed50000";
    pub(crate) const PCESVN: u16 = 13;
    pub(crate) const QE_MRSIGNER: [u8; 32] = [0x33; 32];
    pub(crate) const QE_ISVPRODID: u16 = 2;
    pub(crate) const QE_ISVSVN: u16 = 1;
    /// The level date of the synthetic TCB levels.
    const TCB_DATE: &str = "2026-08-12T00:00:00Z";
    /// The TCB evaluation Intel's `standard` feed served on 2026-10-22, below
    /// the production minimum; `early` served the minimum itself.
    pub(crate) const STANDARD_EVALUATION: u64 = 20;
    /// The advisoryIDs Intel's `early` tcbInfo for FMSPC 00806f050000 lists
    /// against its level of 2026-02-11, in Intel's order (read 2026-10-28).
    pub(crate) const ADVISORIES_2026_10_28: [&str; 3] =
        ["INTEL-DOC-10003", "INTEL-SA-01439", "INTEL-SA-01442"];

    /// What Intel serves at `now`, as (issued, next update) in unix seconds:
    /// issued within the hour, next update 30 days on (measured 2026-10-22).
    pub(crate) const fn fresh(now: u64) -> (u64, u64) {
        (now.saturating_sub(3600), now + 30 * 86400)
    }

    /// 1 January of `from` to 1 January of `to`, in unix seconds.
    pub(crate) fn years(from: i32, to: i32) -> (u64, u64) {
        let jan1 = |y| date_time_ymd(y, 1, 1).unix_timestamp() as u64;
        (jan1(from), jan1(to))
    }

    /// `YYYY-MM-DDThh:mm:ssZ`, the PCS date format.
    pub(crate) fn pcs_date(unix: u64) -> String {
        let (y, m, d, hh, mm, ss) = crate::remote_verify::civil_from_unix(unix);
        format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
    }

    fn tlv(tag: u8, content: &[u8]) -> Vec<u8> {
        let mut out = vec![tag];
        let n = content.len();
        if n < 0x80 {
            out.push(n as u8);
        } else if n < 0x100 {
            out.extend_from_slice(&[0x81, n as u8]);
        } else {
            out.extend_from_slice(&[0x82, (n >> 8) as u8, n as u8]);
        }
        out.extend_from_slice(content);
        out
    }

    fn oid_pair(oid: &[u8], value: Vec<u8>) -> Vec<u8> {
        let mut inner = tlv(0x06, oid);
        inner.extend_from_slice(&value);
        tlv(0x30, &inner)
    }

    /// The SGX extension as `tcb::pck_tcb` reads it: FMSPC and the TCB
    /// sequence with CPUSVN and PCESVN.
    fn sgx_extension(fmspc: [u8; 6], cpusvn: &[u8; 16], pcesvn: u16) -> Vec<u8> {
        const OID_TCB: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf8, 0x4d, 0x01, 0x0d, 0x01, 0x02];
        const OID_PCESVN: &[u8] = &[
            0x2a, 0x86, 0x48, 0x86, 0xf8, 0x4d, 0x01, 0x0d, 0x01, 0x02, 0x11,
        ];
        const OID_CPUSVN: &[u8] = &[
            0x2a, 0x86, 0x48, 0x86, 0xf8, 0x4d, 0x01, 0x0d, 0x01, 0x02, 0x12,
        ];
        const OID_FMSPC: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf8, 0x4d, 0x01, 0x0d, 0x01, 0x04];
        let mut tcb = oid_pair(OID_PCESVN, tlv(0x02, &pcesvn.to_be_bytes()));
        tcb.extend_from_slice(&oid_pair(OID_CPUSVN, tlv(0x04, cpusvn)));
        let mut seq = oid_pair(OID_TCB, tlv(0x30, &tcb));
        seq.extend_from_slice(&oid_pair(OID_FMSPC, tlv(0x04, &fmspc)));
        tlv(0x30, &seq)
    }

    fn p256_signing_key(kp: &KeyPair) -> SigningKey {
        p256::SecretKey::from_sec1_pem(&kp.serialize_pem())
            .map(SigningKey::from)
            .or_else(|_| {
                p256::pkcs8::DecodePrivateKey::from_pkcs8_pem(&kp.serialize_pem())
                    .map(|sk: p256::SecretKey| SigningKey::from(sk))
            })
            .expect("rcgen key readable")
    }

    fn wide_window(params: &mut CertificateParams) {
        params.not_before = date_time_ymd(1970, 1, 1);
        params.not_after = date_time_ymd(4096, 1, 1);
    }

    /// Our Intel: one root, a PCK Platform CA under it, a PCK leaf with the
    /// SGX extension, a TCB signing leaf, and the collateral they vouch for.
    pub(crate) struct FakeIntel {
        pub(crate) root_der: Vec<u8>,
        root_pem: String,
        root_params: CertificateParams,
        root_key_der: Vec<u8>,
        pck_ca_pem: String,
        pck_ca_params: CertificateParams,
        pck_ca_key_der: Vec<u8>,
        pck_pem: String,
        pck_key: SigningKey,
        /// The PCK's own parameters and key (DER): what a holder of its
        /// private key signs with (TEE.fail, `THREAT_MODEL.md`).
        pck_params: CertificateParams,
        pck_key_der: Vec<u8>,
        pck_serial: Vec<u8>,
        tcb_signing_pem: String,
        tcb_signing_key: SigningKey,
    }

    /// The serials of what the root issues, fixed so a root CRL can name them.
    const PCK_CA_SERIAL: [u8; 16] = [0x5c; 16];
    const TCB_SIGNING_SERIAL: [u8; 16] = [0x7c; 16];

    impl FakeIntel {
        pub(crate) fn new() -> Self {
            Self::two_platforms().0
        }

        /// Two platforms under one root: a PCK each, the same collateral.
        pub(crate) fn two_platforms() -> (Self, Self) {
            let root_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
            let root_key_der = root_kp.serialize_der();
            let mut root_params = CertificateParams::new(vec![]).unwrap();
            root_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
            root_params
                .distinguished_name
                .push(DnType::CommonName, "Test SGX Root CA");
            wide_window(&mut root_params);
            let root_cert = root_params.clone().self_signed(&root_kp).unwrap();
            let root_issuer = Issuer::new(root_params.clone(), root_kp);

            // Intel's CN: `tcb::pck_crl_ca` names the CRL by the PCK's issuer.
            let ca_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
            let mut ca_params = CertificateParams::new(vec![]).unwrap();
            ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
            ca_params
                .distinguished_name
                .push(DnType::CommonName, "Intel SGX PCK Platform CA");
            ca_params.serial_number = Some(SerialNumber::from_slice(&PCK_CA_SERIAL));
            wide_window(&mut ca_params);
            let ca_cert = ca_params.clone().signed_by(&ca_kp, &root_issuer).unwrap();
            let ca_key_der = ca_kp.serialize_der();
            let ca_issuer = Issuer::new(ca_params.clone(), ca_kp);

            let pck = |serial: [u8; 16]| {
                let pck_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
                let mut pck_params = CertificateParams::new(vec![]).unwrap();
                pck_params
                    .distinguished_name
                    .push(DnType::CommonName, "Test PCK Certificate");
                pck_params.serial_number = Some(SerialNumber::from_slice(&serial));
                wide_window(&mut pck_params);
                let fmspc = crate::cloud_config::unhex::<6>(FMSPC_HEX).unwrap();
                pck_params
                    .custom_extensions
                    .push(CustomExtension::from_oid_content(
                        &[1, 2, 840, 113741, 1, 13, 1],
                        sgx_extension(fmspc, &[0u8; 16], PCESVN),
                    ));
                let pck_cert = pck_params.clone().signed_by(&pck_kp, &ca_issuer).unwrap();
                (
                    pck_cert.pem(),
                    p256_signing_key(&pck_kp),
                    serial.to_vec(),
                    pck_params,
                    pck_kp.serialize_der(),
                )
            };
            let (pck_a, pck_b) = (pck([0x1a; 16]), pck([0x1b; 16]));

            let tcb_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
            let mut tcb_params = CertificateParams::new(vec![]).unwrap();
            tcb_params
                .distinguished_name
                .push(DnType::CommonName, "Intel SGX TCB Signing");
            tcb_params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
            tcb_params.serial_number = Some(SerialNumber::from_slice(&TCB_SIGNING_SERIAL));
            wide_window(&mut tcb_params);
            let tcb_cert = tcb_params.signed_by(&tcb_kp, &root_issuer).unwrap();

            type Pck = (String, SigningKey, Vec<u8>, CertificateParams, Vec<u8>);
            let platform = |(pck_pem, pck_key, pck_serial, pck_params, pck_key_der): Pck| Self {
                root_der: root_cert.der().to_vec(),
                root_pem: root_cert.pem(),
                root_params: root_params.clone(),
                root_key_der: root_key_der.clone(),
                pck_ca_pem: ca_cert.pem(),
                pck_ca_params: ca_params.clone(),
                pck_ca_key_der: ca_key_der.clone(),
                pck_pem,
                pck_key,
                pck_params,
                pck_key_der,
                pck_serial,
                tcb_signing_pem: tcb_cert.pem(),
                tcb_signing_key: p256_signing_key(&tcb_kp),
            };
            (platform(pck_a), platform(pck_b))
        }

        pub(crate) fn anchor(&self) -> TrustAnchor {
            TrustAnchor::RootDer(self.root_der.clone())
        }

        /// The issuer chain PCS puts in its response header.
        pub(crate) fn collateral_chain_pem(&self) -> String {
            format!("{}{}", self.tcb_signing_pem, self.root_pem)
        }

        /// The serial number of this platform's PCK certificate.
        pub(crate) fn pck_serial(&self) -> Vec<u8> {
            self.pck_serial.clone()
        }

        /// The PCK CA's CRL (DER) listing `revoked`, valid over `window`
        /// (thisUpdate, nextUpdate, unix seconds: [`fresh`] or [`years`]).
        pub(crate) fn pck_crl(&self, revoked: &[Vec<u8>], window: (u64, u64)) -> Vec<u8> {
            self.crl_signed_by(&self.pck_ca_key_der, revoked, window)
        }

        /// The same CRL signed by `key_der`, a key that is not the PCK CA's.
        pub(crate) fn crl_signed_by(
            &self,
            key_der: &[u8],
            revoked: &[Vec<u8>],
            window: (u64, u64),
        ) -> Vec<u8> {
            self.crl_with(key_der, revoked, window, |_| {})
        }

        pub(crate) fn pck_ca_key_der(&self) -> &[u8] {
            &self.pck_ca_key_der
        }

        pub(crate) fn root_key_der(&self) -> &[u8] {
            &self.root_key_der
        }

        /// The serial of the PCK CA that issued this platform's PCK.
        pub(crate) fn pck_ca_serial(&self) -> Vec<u8> {
            PCK_CA_SERIAL.to_vec()
        }

        /// The serial of the TCB signing certificate behind the collateral.
        pub(crate) fn tcb_signing_serial(&self) -> Vec<u8> {
            TCB_SIGNING_SERIAL.to_vec()
        }

        /// The PCK CA's CRL with `edit` applied to its parameters before signing.
        pub(crate) fn crl_with(
            &self,
            key_der: &[u8],
            revoked: &[Vec<u8>],
            window: (u64, u64),
            edit: impl FnOnce(&mut CertificateRevocationListParams),
        ) -> Vec<u8> {
            crl_under(&self.pck_ca_params, key_der, revoked, window, edit)
        }

        /// The root CA's CRL (DER) listing `revoked`, valid over `window`:
        /// what the root's CRL distribution point serves.
        pub(crate) fn root_crl(&self, revoked: &[Vec<u8>], window: (u64, u64)) -> Vec<u8> {
            self.root_crl_with(&self.root_key_der, revoked, window, |_| {})
        }

        /// The root CA's CRL signed by `key_der`, `edit` applied before signing.
        pub(crate) fn root_crl_with(
            &self,
            key_der: &[u8],
            revoked: &[Vec<u8>],
            window: (u64, u64),
            edit: impl FnOnce(&mut CertificateRevocationListParams),
        ) -> Vec<u8> {
            crl_under(&self.root_params, key_der, revoked, window, edit)
        }

        /// An empty root CRL valid over every test clock: what the fake PCS
        /// serves unless a test says otherwise.
        pub(crate) fn root_crl_wide(&self) -> Vec<u8> {
            self.root_crl(&[], years(2020, 2040))
        }

        /// The chain PCS sends with the PCK CRL: PCK CA, root.
        pub(crate) fn pck_crl_chain_pem(&self) -> String {
            format!("{}{}", self.pck_ca_pem, self.root_pem)
        }
    }

    /// A CRL of the CA described by `issuer_params`, signed by `key_der`,
    /// listing `revoked` over `window` (thisUpdate, nextUpdate, unix seconds).
    fn crl_under(
        issuer_params: &CertificateParams,
        key_der: &[u8],
        revoked: &[Vec<u8>],
        window: (u64, u64),
        edit: impl FnOnce(&mut CertificateRevocationListParams),
    ) -> Vec<u8> {
        let issuer = Issuer::new(issuer_params.clone(), KeyPair::try_from(key_der).unwrap());
        let at = |unix: u64| date_time_ymd(1970, 1, 1) + std::time::Duration::from_secs(unix);
        let mut params = CertificateRevocationListParams {
            this_update: at(window.0),
            next_update: at(window.1),
            crl_number: SerialNumber::from(1u64),
            issuing_distribution_point: None,
            revoked_certs: revoked
                .iter()
                .map(|s| RevokedCertParams {
                    serial_number: SerialNumber::from_slice(s),
                    revocation_time: at(window.0),
                    reason_code: Some(RevocationReason::KeyCompromise),
                    invalidity_date: None,
                })
                .collect(),
            key_identifier_method: KeyIdMethod::Sha256,
        };
        edit(&mut params);
        params.signed_by(&issuer).unwrap().der().to_vec()
    }

    impl FakeIntel {
        fn sign_collateral(&self, key: &str, value: &str) -> String {
            let sig: Signature = self.tcb_signing_key.sign(value.as_bytes());
            format!(
                "{{\"{key}\":{value},\"signature\":\"{}\"}}",
                hex(&sig.to_bytes())
            )
        }

        /// The tcbInfo Intel's `early` feed serves, issued at `issued`: the
        /// production minimum evaluation, this platform UpToDate.
        pub(crate) fn tcb_info(&self, issued: u64) -> String {
            self.tcb_info_with(
                issued,
                crate::tcb::TcbPolicy::MIN_TCB_EVALUATION_DATA_NUMBER,
                "UpToDate",
            )
        }

        /// A tcbInfo of evaluation `evaluation` whose one level rates this
        /// platform `status`, valid 30 days from `issued`.
        pub(crate) fn tcb_info_with(&self, issued: u64, evaluation: u64, status: &str) -> String {
            let comps = (0..16).map(|_| "{\"svn\":0}").collect::<Vec<_>>().join(",");
            let (issue, next) = (pcs_date(issued), pcs_date(issued + 30 * 86400));
            let value = format!(
                "{{\"id\":\"TDX\",\"version\":3,\"issueDate\":\"{issue}\",\"nextUpdate\":\"{next}\",\
                 \"fmspc\":\"{FMSPC_HEX}\",\"pceId\":\"0000\",\"tcbType\":0,\"tcbEvaluationDataNumber\":{evaluation},\
                 \"tdxModule\":{{\"mrsigner\":\"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000\",\"attributes\":\"0000000000000000\",\"attributesMask\":\"FFFFFFFFFFFFFFFF\"}},\
                 \"tcbLevels\":[{{\"tcb\":{{\"sgxtcbcomponents\":[{comps}],\"pcesvn\":{PCESVN},\
                 \"tdxtcbcomponents\":[{comps}]}},\"tcbDate\":\"{TCB_DATE}\",\"tcbStatus\":\"{status}\"}}]}}"
            );
            self.sign_collateral("tcbInfo", &value)
        }

        /// A tcbInfo of evaluation `evaluation` with `levels`, newest first,
        /// each as (SVN of SGX component 0, tcbDate day, status, advisoryIDs).
        /// This platform's SVNs are all 0: the first level asking 0 is its own.
        pub(crate) fn tcb_info_levels(
            &self,
            issued: u64,
            evaluation: u64,
            levels: &[(u64, &str, &str, &[&str])],
        ) -> String {
            let zeros = (0..16).map(|_| "{\"svn\":0}").collect::<Vec<_>>().join(",");
            let rest = (1..16).map(|_| "{\"svn\":0}").collect::<Vec<_>>().join(",");
            let levels = levels
                .iter()
                .map(|(svn, day, status, advisories)| {
                    let ids = if advisories.is_empty() {
                        String::new()
                    } else {
                        format!(
                            ",\"advisoryIDs\":[{}]",
                            advisories
                                .iter()
                                .map(|a| format!("\"{a}\""))
                                .collect::<Vec<_>>()
                                .join(",")
                        )
                    };
                    format!(
                        "{{\"tcb\":{{\"sgxtcbcomponents\":[{{\"svn\":{svn}}},{rest}],\"pcesvn\":{PCESVN},\
                         \"tdxtcbcomponents\":[{zeros}]}},\"tcbDate\":\"{day}T00:00:00Z\",\"tcbStatus\":\"{status}\"{ids}}}"
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            let (issue, next) = (pcs_date(issued), pcs_date(issued + 30 * 86400));
            let value = format!(
                "{{\"id\":\"TDX\",\"version\":3,\"issueDate\":\"{issue}\",\"nextUpdate\":\"{next}\",\
                 \"fmspc\":\"{FMSPC_HEX}\",\"pceId\":\"0000\",\"tcbType\":0,\"tcbEvaluationDataNumber\":{evaluation},\
                 \"tdxModule\":{{\"mrsigner\":\"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000\",\"attributes\":\"0000000000000000\",\"attributesMask\":\"FFFFFFFFFFFFFFFF\"}},\
                 \"tcbLevels\":[{levels}]}}"
            );
            self.sign_collateral("tcbInfo", &value)
        }

        /// Intel's two answers for this platform as PCS gave them for the GCE
        /// H100 hosts on 2026-10-28 (`tests/pcs_live/2026-10-28/`), on this
        /// fake's SVNs: (`standard` evaluation 20, its level of 2025-08-13
        /// UpToDate; `early` evaluation 22, its level of 2026-02-11 OutOfDate
        /// with Intel's three advisories, a newer level UpToDate).
        pub(crate) fn tcb_info_of_2026_10_28(&self, issued: u64) -> (String, String) {
            (
                self.tcb_info_levels(
                    issued,
                    STANDARD_EVALUATION,
                    &[(0, "2025-08-13", "UpToDate", &[])],
                ),
                self.tcb_info_levels(
                    issued,
                    crate::tcb::TcbPolicy::MIN_TCB_EVALUATION_DATA_NUMBER,
                    &[
                        (1, "2026-08-12", "UpToDate", &[]),
                        (0, "2026-02-11", "OutOfDate", &ADVISORIES_2026_10_28),
                    ],
                ),
            )
        }

        /// A tcbInfo of the production evaluation, this platform UpToDate,
        /// with a `TDX_01` module identity whose only level asks ISVSVN
        /// `isvsvn`.
        pub(crate) fn tcb_info_module(&self, issued: u64, isvsvn: u64) -> String {
            let comps = (0..16).map(|_| "{\"svn\":0}").collect::<Vec<_>>().join(",");
            let (issue, next) = (pcs_date(issued), pcs_date(issued + 30 * 86400));
            let zero48 = "0".repeat(96);
            let value = format!(
                "{{\"id\":\"TDX\",\"version\":3,\"issueDate\":\"{issue}\",\"nextUpdate\":\"{next}\",\
                 \"fmspc\":\"{FMSPC_HEX}\",\"pceId\":\"0000\",\"tcbType\":0,\"tcbEvaluationDataNumber\":{},\
                 \"tdxModule\":{{\"mrsigner\":\"{zero48}\",\"attributes\":\"0000000000000000\",\"attributesMask\":\"FFFFFFFFFFFFFFFF\"}},\
                 \"tdxModuleIdentities\":[{{\"id\":\"TDX_01\",\"mrsigner\":\"{zero48}\",\"attributes\":\"0000000000000000\",\"attributesMask\":\"FFFFFFFFFFFFFFFF\",\
                 \"tcbLevels\":[{{\"tcb\":{{\"isvsvn\":{isvsvn}}},\"tcbDate\":\"{TCB_DATE}\",\"tcbStatus\":\"UpToDate\"}}]}}],\
                 \"tcbLevels\":[{{\"tcb\":{{\"sgxtcbcomponents\":[{comps}],\"pcesvn\":{PCESVN},\
                 \"tdxtcbcomponents\":[{comps}]}},\"tcbDate\":\"{TCB_DATE}\",\"tcbStatus\":\"UpToDate\"}}]}}",
                crate::tcb::TcbPolicy::MIN_TCB_EVALUATION_DATA_NUMBER
            );
            self.sign_collateral("tcbInfo", &value)
        }

        /// The QE identity Intel's `early` feed serves, issued at `issued`.
        pub(crate) fn qe_identity(&self, issued: u64) -> String {
            self.qe_identity_with(
                issued,
                crate::tcb::TcbPolicy::MIN_TCB_EVALUATION_DATA_NUMBER,
            )
        }

        /// The QE identity of evaluation `evaluation`, valid 30 days from `issued`.
        pub(crate) fn qe_identity_with(&self, issued: u64, evaluation: u64) -> String {
            let (issue, next) = (pcs_date(issued), pcs_date(issued + 30 * 86400));
            let value = format!(
                "{{\"id\":\"TD_QE\",\"version\":2,\"issueDate\":\"{issue}\",\"nextUpdate\":\"{next}\",\
                 \"tcbEvaluationDataNumber\":{evaluation},\"miscselect\":\"00000000\",\"miscselectMask\":\"FFFFFFFF\",\
                 \"attributes\":\"00000000000000000000000000000000\",\"attributesMask\":\"FBFFFFFFFFFFFFFF0000000000000000\",\
                 \"mrsigner\":\"{}\",\"isvprodid\":{QE_ISVPRODID},\
                 \"tcbLevels\":[{{\"tcb\":{{\"isvsvn\":{QE_ISVSVN}}},\"tcbDate\":\"{TCB_DATE}\",\"tcbStatus\":\"UpToDate\"}}]}}",
                hex(&QE_MRSIGNER)
            );
            self.sign_collateral("enclaveIdentity", &value)
        }

        /// A TD Quote v4 over these registers, signed like Intel signs one.
        pub(crate) fn quote(
            &self,
            report_data: &[u8; 64],
            mrtd: &[u8; 48],
            rtmr: &[[u8; 48]; 4],
            td_attributes: [u8; 8],
        ) -> Vec<u8> {
            let chain_pem = format!("{}{}{}", self.pck_pem, self.pck_ca_pem, self.root_pem);
            Self::quote_signed(
                report_data,
                mrtd,
                rtmr,
                td_attributes,
                &self.pck_key,
                &chain_pem,
            )
        }

        /// The attack of a holder of this PCK's private key (TEE.fail): a
        /// leaf of its own making, issued by the PCK, carrying this platform's
        /// SGX extension, signs the QE report; the chain is that leaf, the
        /// genuine PCK, the PCK CA and the root. Every signature verifies.
        pub(crate) fn quote_under_extracted_pck(
            &self,
            report_data: &[u8; 64],
            mrtd: &[u8; 48],
            rtmr: &[[u8; 48]; 4],
            td_attributes: [u8; 8],
        ) -> Vec<u8> {
            let pck_issuer = Issuer::new(
                self.pck_params.clone(),
                KeyPair::try_from(self.pck_key_der.as_slice()).unwrap(),
            );
            let leaf_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
            let mut leaf = CertificateParams::new(vec![]).unwrap();
            leaf.distinguished_name
                .push(DnType::CommonName, "Intel SGX PCK Certificate");
            wide_window(&mut leaf);
            let fmspc = crate::cloud_config::unhex::<6>(FMSPC_HEX).unwrap();
            leaf.custom_extensions
                .push(CustomExtension::from_oid_content(
                    &[1, 2, 840, 113741, 1, 13, 1],
                    sgx_extension(fmspc, &[0u8; 16], PCESVN),
                ));
            let leaf_cert = leaf.signed_by(&leaf_kp, &pck_issuer).unwrap();
            let chain_pem = format!(
                "{}{}{}{}",
                leaf_cert.pem(),
                self.pck_pem,
                self.pck_ca_pem,
                self.root_pem
            );
            Self::quote_signed(
                report_data,
                mrtd,
                rtmr,
                td_attributes,
                &p256_signing_key(&leaf_kp),
                &chain_pem,
            )
        }

        /// A tcbInfo and a QE identity issued at `issued`, signed by a new
        /// "Intel SGX TCB Signing" end entity (digitalSignature) that the
        /// holder of `issuer` (its parameters, its key) issues; its PEM.
        fn collateral_signed_under(
            &self,
            issued: u64,
            issuer: &CertificateParams,
            issuer_key_der: &[u8],
        ) -> (String, String, String) {
            let issuer = Issuer::new(issuer.clone(), KeyPair::try_from(issuer_key_der).unwrap());
            let kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
            let mut params = CertificateParams::new(vec![]).unwrap();
            params
                .distinguished_name
                .push(DnType::CommonName, "Intel SGX TCB Signing");
            params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
            wide_window(&mut params);
            let cert = params.signed_by(&kp, &issuer).unwrap();
            let key = p256_signing_key(&kp);
            let resign = |name: &str, body: String| {
                let prefix = format!("{{\"{name}\":");
                let value = &body[prefix.len()..body.rfind(",\"signature\"").unwrap()];
                let sig: Signature = key.sign(value.as_bytes());
                format!(
                    "{prefix}{value},\"signature\":\"{}\"}}",
                    hex(&sig.to_bytes())
                )
            };
            (
                resign("tcbInfo", self.tcb_info(issued)),
                resign("enclaveIdentity", self.qe_identity(issued)),
                cert.pem(),
            )
        }

        /// A TCB signer issued by this PCK's key,
        /// chained to the PCK, the PCK CA and the root; every signature
        /// verifies. (tcbInfo, QE identity, chain.)
        pub(crate) fn collateral_under_extracted_pck(
            &self,
            issued: u64,
        ) -> (String, String, String) {
            let (tcb, qe, signer) =
                self.collateral_signed_under(issued, &self.pck_params, &self.pck_key_der);
            let chain = format!(
                "{signer}{}{}{}",
                self.pck_pem, self.pck_ca_pem, self.root_pem
            );
            (tcb, qe, chain)
        }

        /// A TCB signer the root's key signs under an issuer name that is
        /// not the root's subject: two certificates, every signature
        /// verifies, the names do not chain. (tcbInfo, QE identity, chain.)
        pub(crate) fn collateral_under_another_issuer_name(
            &self,
            issued: u64,
        ) -> (String, String, String) {
            let mut misnamed = self.root_params.clone();
            misnamed.distinguished_name = DistinguishedName::new();
            misnamed
                .distinguished_name
                .push(DnType::CommonName, "Not The Root");
            let (tcb, qe, signer) =
                self.collateral_signed_under(issued, &misnamed, &self.root_key_der);
            (tcb, qe, format!("{signer}{}", self.root_pem))
        }

        /// The root's name and key in a self-signed certificate that is no
        /// CA, and collateral under it. (tcbInfo, QE identity, chain, DER.)
        pub(crate) fn collateral_under_a_root_that_is_no_ca(
            &self,
            issued: u64,
        ) -> (String, String, String, Vec<u8>) {
            let mut no_ca = self.root_params.clone();
            no_ca.is_ca = IsCa::ExplicitNoCa;
            let kp = KeyPair::try_from(self.root_key_der.as_slice()).unwrap();
            let root = no_ca.clone().self_signed(&kp).unwrap();
            let (tcb, qe, signer) =
                self.collateral_signed_under(issued, &no_ca, &self.root_key_der);
            (
                tcb,
                qe,
                format!("{signer}{}", root.pem()),
                root.der().to_vec(),
            )
        }

        /// A quote whose PCK a CA of this root other than Intel's PCK CAs
        /// issued ("Test Other CA", cA true), in a chain of Intel's shape,
        /// and that CA's CRL listing nothing over `window`. (Quote, CRL.)
        pub(crate) fn quote_under_another_ca(
            &self,
            report_data: &[u8; 64],
            mrtd: &[u8; 48],
            rtmr: &[[u8; 48]; 4],
            window: (u64, u64),
        ) -> (Vec<u8>, Vec<u8>) {
            let root = Issuer::new(
                self.root_params.clone(),
                KeyPair::try_from(self.root_key_der.as_slice()).unwrap(),
            );
            let ca_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
            let ca_key_der = ca_kp.serialize_der();
            let mut ca = CertificateParams::new(vec![]).unwrap();
            ca.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
            ca.distinguished_name
                .push(DnType::CommonName, "Test Other CA");
            wide_window(&mut ca);
            let ca_cert = ca.clone().signed_by(&ca_kp, &root).unwrap();
            let ca_issuer = Issuer::new(ca.clone(), ca_kp);
            let pck_kp = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256).unwrap();
            let mut pck = CertificateParams::new(vec![]).unwrap();
            pck.distinguished_name
                .push(DnType::CommonName, "Test PCK Certificate");
            wide_window(&mut pck);
            let fmspc = crate::cloud_config::unhex::<6>(FMSPC_HEX).unwrap();
            pck.custom_extensions
                .push(CustomExtension::from_oid_content(
                    &[1, 2, 840, 113741, 1, 13, 1],
                    sgx_extension(fmspc, &[0u8; 16], PCESVN),
                ));
            let pck_cert = pck.signed_by(&pck_kp, &ca_issuer).unwrap();
            let chain = format!("{}{}{}", pck_cert.pem(), ca_cert.pem(), self.root_pem);
            let quote = Self::quote_signed(
                report_data,
                mrtd,
                rtmr,
                [0; 8],
                &p256_signing_key(&pck_kp),
                &chain,
            );
            (quote, crl_under(&ca, &ca_key_der, &[], window, |_| {}))
        }

        /// A CRL issued and signed by this PCK, as its key holder signs one.
        pub(crate) fn crl_signed_by_the_pck(&self, window: (u64, u64)) -> Vec<u8> {
            crl_under(&self.pck_params, &self.pck_key_der, &[], window, |_| {})
        }

        /// A TD Quote v4 whose QE report `qe_signer` signs, `chain_pem` in
        /// its certification data.
        fn quote_signed(
            report_data: &[u8; 64],
            mrtd: &[u8; 48],
            rtmr: &[[u8; 48]; 4],
            td_attributes: [u8; 8],
            qe_signer: &SigningKey,
            chain_pem: &str,
        ) -> Vec<u8> {
            let att_sk = SigningKey::from_slice(&[0x42u8; 32]).unwrap();
            let att_vk_xy = att_sk.verifying_key().to_sec1_bytes()[1..].to_vec();
            let qe_auth: &[u8] = b"synthetic-qe-auth";
            let mut qe_report = vec![0u8; 384];
            qe_report[128..160].copy_from_slice(&QE_MRSIGNER);
            qe_report[256..258].copy_from_slice(&QE_ISVPRODID.to_le_bytes());
            qe_report[258..260].copy_from_slice(&QE_ISVSVN.to_le_bytes());
            let mut h = Sha256::new();
            h.update(&att_vk_xy);
            h.update(qe_auth);
            qe_report[320..352].copy_from_slice(&h.finalize());
            let qe_sig: Signature = qe_signer.sign(&qe_report);

            let mut quote = crate::pcs::tests::td_quote_header_and_body();
            quote[48 + 120..48 + 128].copy_from_slice(&td_attributes);
            quote[48 + 136..48 + 184].copy_from_slice(mrtd);
            for (i, r) in rtmr.iter().enumerate() {
                quote[48 + 328 + i * 48..48 + 328 + (i + 1) * 48].copy_from_slice(r);
            }
            quote[SIGNED_LEN - 64..].copy_from_slice(report_data);
            let quote_sig: Signature = att_sk.sign(&quote[..SIGNED_LEN]);

            let mut sig_data = Vec::new();
            sig_data.extend_from_slice(&quote_sig.to_bytes());
            sig_data.extend_from_slice(&att_vk_xy);
            sig_data.extend_from_slice(&6u16.to_le_bytes());
            sig_data.extend_from_slice(&0u32.to_le_bytes());
            sig_data.extend_from_slice(&qe_report);
            sig_data.extend_from_slice(&qe_sig.to_bytes());
            sig_data.extend_from_slice(&(qe_auth.len() as u16).to_le_bytes());
            sig_data.extend_from_slice(qe_auth);
            sig_data.extend_from_slice(&5u16.to_le_bytes());
            sig_data.extend_from_slice(&(chain_pem.len() as u32).to_le_bytes());
            sig_data.extend_from_slice(chain_pem.as_bytes());
            quote.extend_from_slice(&(sig_data.len() as u32).to_le_bytes());
            quote.extend_from_slice(&sig_data);
            quote
        }
    }

    /// Our NVIDIA: the `nras` fixture's signing leaf and JWKS, minting
    /// tokens for any nonce and clock.
    pub(crate) struct FakeNvidia {
        fx: crate::nras::tests::Fixture,
    }

    impl FakeNvidia {
        pub(crate) fn new() -> Self {
            Self {
                fx: crate::nras::tests::fixture_for_composite(),
            }
        }

        /// NRAS under a P-384 intermediate, as under 005.
        pub(crate) fn p384() -> Self {
            Self {
                fx: crate::nras::tests::fixture_for_composite_p384(),
            }
        }

        pub(crate) fn intermediate_pins(&self) -> Vec<crate::nras::NrasIntermediatePin> {
            self.fx.pins.clone()
        }

        /// Tokens answering `nonce_hex` at `now`; `signer` replaces the
        /// genuine leaf key when a test wants a token nobody pinned.
        pub(crate) fn attest(
            &self,
            nonce_hex: &str,
            now: u64,
            signer: Option<&p384::ecdsa::SigningKey>,
        ) -> GpuAttested {
            use crate::nras::tests::{GPU_SERIAL_BYTES, GPU_SERIAL_DECIMAL};
            self.attest_gpu(
                nonce_hex,
                now,
                signer,
                (&GPU_SERIAL_BYTES, GPU_SERIAL_DECIMAL),
            )
        }

        /// The same for the GPU whose attestation certificate has `serial`
        /// (its bytes, and its decimal as NRAS writes it in `ueid`).
        pub(crate) fn attest_gpu(
            &self,
            nonce_hex: &str,
            now: u64,
            signer: Option<&p384::ecdsa::SigningKey>,
            serial: (&[u8], &str),
        ) -> GpuAttested {
            use crate::nras::tests::{gpu_chain_b64, gpu_claims, submods_of};
            let (serial_bytes, serial_decimal) = serial;
            let exp = now + 3600;
            let mut overall = serde_json::json!({
                "iss": "https://nras.attestation.nvidia.com",
                "exp": exp,
                "iat": now.saturating_sub(60),
                "eat_nonce": nonce_hex,
                "x-nvidia-overall-att-result": true,
                "measres": "success",
                "hwmodel": "GH100 A01 GSP BROM",
            });
            let sign = |claims: &serde_json::Value| -> String {
                match signer {
                    None => (self.fx.sign)(claims),
                    Some(sk) => {
                        use base64::Engine as _;
                        use p384::ecdsa::signature::Signer as _;
                        let b64 =
                            |b: &[u8]| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b);
                        let header =
                            serde_json::json!({"alg": "ES384", "kid": "nv-eat-kid-test-1"});
                        let signed = format!(
                            "{}.{}",
                            b64(&serde_json::to_vec(&header).unwrap()),
                            b64(&serde_json::to_vec(claims).unwrap())
                        );
                        let sig: p384::ecdsa::Signature = sk.sign(signed.as_bytes());
                        format!("{signed}.{}", b64(&sig.to_bytes()))
                    }
                }
            };
            let per_gpu = vec![(
                "GPU-0".to_string(),
                sign(&gpu_claims(nonce_hex, exp, serial_decimal)),
            )];
            overall["submods"] = submods_of(&per_gpu);
            GpuAttested {
                overall_token: sign(&overall),
                per_gpu,
                jwks: self.fx.jwks.clone(),
                evidence: vec![crate::nras_client::GpuEvidence {
                    evidence: b64(b"synthetic-spdm-report"),
                    certificate: gpu_chain_b64(serial_bytes),
                }],
                local_serials: vec![serial_decimal.to_string()],
                ocsp: Vec::new(),
            }
        }
    }

    /// What a simulated NRAS answer gets wrong ([`FakeNvidia::answer`]).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum NrasLie {
        None,
        /// Every token names another nonce than the request's.
        OtherNonce,
        /// Signed by a key no pinned intermediate certified, under the genuine kid.
        Unpinned,
        /// `x-nvidia-overall-att-result` false.
        OverallFalse,
        /// GPU-0's `measres` is not "success".
        MeasresFail,
        /// GPU-0 reports `dbgstat` "enabled".
        DebugEnabled,
        /// GPU-0's token names another GPU (`ueid`): the answer for a
        /// carrier's own H100, asked for the same nonce.
        OtherUeid,
        /// The per-GPU tokens are not the ones the overall token's `submods` names.
        ForeignSubmods,
        /// Expired an hour ago.
        Expired,
        /// Every token valid only from an hour on (`nbf`).
        NotYetValid,
    }

    /// The report a simulated GPU signs for `nonce_hex`: it echoes the nonce,
    /// as an SPDM measurement report echoes the requester's.
    pub(crate) fn synthetic_report(nonce_hex: &str) -> Vec<u8> {
        format!("synthetic-spdm-report:{nonce_hex}").into_bytes()
    }

    /// A GPU attestation certificate serial other than the TD's
    /// ([`crate::nras::tests::GPU_SERIAL_BYTES`]): 2^64 + 1.
    pub(crate) const OTHER_GPU_SERIAL_BYTES: [u8; 9] = [1, 0, 0, 0, 0, 0, 0, 0, 1];

    impl FakeNvidia {
        /// The JWKS NRAS publishes: the signing leaf and its intermediate in `x5c`.
        pub(crate) fn jwks(&self) -> serde_json::Value {
            self.fx.jwks.clone()
        }

        /// The DER of this NVIDIA's intermediate (`x5c[1]`): what a caller pins.
        pub(crate) fn intermediate_der(&self) -> Vec<u8> {
            use base64::Engine as _;
            let x5c1 = self.fx.jwks["keys"][0]["x5c"][1].as_str().unwrap();
            base64::engine::general_purpose::STANDARD
                .decode(x5c1)
                .unwrap()
        }

        /// An OCSP response about this NVIDIA's signing leaf, signed by its
        /// intermediate, DER base64: what a TD would staple.
        pub(crate) fn ocsp_b64(&self, status: crate::nras_ocsp::tests::Status, now: u64) -> String {
            use base64::Engine as _;
            let r = crate::nras_ocsp::tests::Response {
                status,
                ..crate::nras_ocsp::tests::Response::good_at(now)
            };
            let der = crate::nras_ocsp::tests::respond_with(
                &r,
                &self.fx.leaf_der,
                &self.fx.inter_der,
                &self.fx.inter_signer,
            );
            base64::engine::general_purpose::STANDARD.encode(der)
        }

        /// NRAS's answer to `request` (the v4 body: nonce, evidence_list) at
        /// `now`, `[["JWT", overall], {"GPU-i": token}]`, with `lie` told. As
        /// NVIDIA's: each report must echo the request's nonce, else that GPU's
        /// nonce-match is false and the verdict too; `ueid` is the serial of
        /// the leaf of the chain sent with the report.
        pub(crate) fn answer(
            &self,
            request: &serde_json::Value,
            now: u64,
            lie: NrasLie,
        ) -> serde_json::Value {
            use crate::nras::tests::{gpu_claims, submods_of};
            use base64::Engine as _;
            let std = base64::engine::general_purpose::STANDARD;
            let asked = request["nonce"].as_str().unwrap_or_default().to_string();
            let nonce = if lie == NrasLie::OtherNonce {
                "ee".repeat(32)
            } else {
                asked.clone()
            };
            let exp = if lie == NrasLie::Expired {
                now - 3600
            } else {
                now + 3600
            };
            let other_key = p384::ecdsa::SigningKey::from_slice(&[0x42; 48]).unwrap();
            let sign = |claims: &serde_json::Value| -> String {
                if lie != NrasLie::Unpinned {
                    return (self.fx.sign)(claims);
                }
                use p384::ecdsa::signature::Signer as _;
                let b64 = |b: &[u8]| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b);
                let kid = self.fx.jwks["keys"][0]["kid"].clone();
                let header = serde_json::json!({"alg": "ES384", "kid": kid});
                let signed = format!(
                    "{}.{}",
                    b64(&serde_json::to_vec(&header).unwrap()),
                    b64(&serde_json::to_vec(claims).unwrap())
                );
                let sig: p384::ecdsa::Signature = other_key.sign(signed.as_bytes());
                format!("{signed}.{}", b64(&sig.to_bytes()))
            };
            let mut all_match = true;
            let mut per_gpu_claims = Vec::new();
            for (i, e) in request["evidence_list"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .enumerate()
            {
                let report = std
                    .decode(e["evidence"].as_str().unwrap_or_default())
                    .unwrap_or_default();
                let matches = report == synthetic_report(&asked);
                all_match &= matches;
                let mut ueid = crate::nras::gpu_leaf_serial_decimal(
                    e["certificate"].as_str().unwrap_or_default(),
                )
                .unwrap_or_default();
                if i == 0 && lie == NrasLie::OtherUeid {
                    ueid = "18446744073709551617".into();
                }
                let mut claims = gpu_claims(&nonce, exp, &ueid);
                if lie == NrasLie::NotYetValid {
                    claims["nbf"] = (now + 3600).into();
                }
                claims["x-nvidia-gpu-attestation-report-nonce-match"] = matches.into();
                if !matches {
                    claims["measres"] = "fail".into();
                }
                if i == 0 && lie == NrasLie::MeasresFail {
                    claims["measres"] = "fail".into();
                }
                if i == 0 && lie == NrasLie::DebugEnabled {
                    claims["dbgstat"] = "enabled".into();
                }
                per_gpu_claims.push((format!("GPU-{i}"), claims));
            }
            let mint = |shift: u64| -> Vec<(String, String)> {
                per_gpu_claims
                    .iter()
                    .map(|(name, c)| {
                        let mut c = c.clone();
                        c["iat"] = (exp - 3600 - shift).into();
                        (name.clone(), sign(&c))
                    })
                    .collect()
            };
            let per_gpu = mint(0);
            let named = if lie == NrasLie::ForeignSubmods {
                mint(1)
            } else {
                per_gpu.clone()
            };
            // NVIDIA's overall token carries `nbf` (Hopper single-GPU example).
            let nbf = if lie == NrasLie::NotYetValid {
                now + 3600
            } else {
                exp - 3600
            };
            let overall = serde_json::json!({
                "iss": "https://nras.attestation.nvidia.com",
                "exp": exp,
                "iat": exp - 3600,
                "nbf": nbf,
                "eat_nonce": nonce,
                "x-nvidia-overall-att-result": all_match && lie != NrasLie::OverallFalse,
                "submods": submods_of(&named),
            });
            let map: serde_json::Map<String, serde_json::Value> = per_gpu
                .into_iter()
                .map(|(n, t)| (n, serde_json::Value::String(t)))
                .collect();
            serde_json::json!([["JWT", sign(&overall)], map])
        }
    }

    /// A fake NRAS on the loopback: `POST /v4/attest/gpu` answers
    /// [`FakeNvidia::answer`] at the wall clock (every answer counted in
    /// the returned counter), `GET /.well-known/jwks.json` its JWKS.
    pub(crate) async fn fake_nras(
        nvidia: Arc<FakeNvidia>,
        lie: NrasLie,
    ) -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        use axum::routing::{get, post};
        let asked = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (n1, n2, count) = (nvidia.clone(), nvidia, asked.clone());
        let app = axum::Router::new()
            .route(
                "/v4/attest/gpu",
                post(move |axum::Json(req): axum::Json<serde_json::Value>| {
                    let n = n1.clone();
                    count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    async move {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs();
                        axum::Json(n.answer(&req, now, lie))
                    }
                }),
            )
            .route(
                "/.well-known/jwks.json",
                get(move || {
                    let jwks = n2.jwks();
                    async move { axum::Json(jwks) }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), asked)
    }

    /// The TD under test: honest by default, each field a lie a test can tell.
    pub(crate) struct FakeTd {
        pub(crate) intel: Arc<FakeIntel>,
        pub(crate) nvidia: Arc<FakeNvidia>,
        pub(crate) now: u64,
        pub(crate) mrtd: [u8; 48],
        pub(crate) events: Vec<Event>,
        /// Registers written into the quote; `None` = the log's replay.
        pub(crate) rtmr_in_quote: Option<[[u8; 48]; 4]>,
        pub(crate) td_attributes: [u8; 8],
        /// Quote answers this instead of the binding (a replayed quote).
        pub(crate) report_data_override: Option<[u8; 64]>,
        /// Tokens answer this nonce instead of the binding's.
        pub(crate) gpu_nonce_override: Option<String>,
        /// Tokens signed by this key instead of the pinned leaf's.
        pub(crate) other_nras_signer: Option<p384::ecdsa::SigningKey>,
        /// The GPU's attestation certificate chain, base64 of its PEM, as NVML gives it.
        pub(crate) gpu_chain_b64: String,
        /// NVML gives no report: the GPU is outside CC mode.
        pub(crate) gpu_cc_off: bool,
        /// How long NVML takes to read the GPUs' evidence.
        pub(crate) gpu_delay: std::time::Duration,
        /// What NVML says of the system's confidential computing.
        pub(crate) gpu_cc_state: CcSystemState,
        /// What the e2e key document's NRAS answer gets wrong.
        pub(crate) nras_lie: NrasLie,
        /// NRAS does not answer the TD.
        pub(crate) nras_down: bool,
    }

    impl FakeTd {
        pub(crate) fn honest(intel: Arc<FakeIntel>, nvidia: Arc<FakeNvidia>, now: u64) -> Self {
            Self {
                intel,
                nvidia,
                now,
                // The synthetic firmware our Google endorses (`bucket_with_endorsed`):
                // its MRTD, and a GCE-shaped RTMR0 whose configuration volume is its.
                mrtd: crate::tdvf::tests::synthetic_mrtd(),
                events: crate::reference::tests::gce_log(crate::tdvf::tests::synthetic_cfv()),
                rtmr_in_quote: None,
                td_attributes: [0; 8],
                report_data_override: None,
                gpu_nonce_override: None,
                other_nras_signer: None,
                gpu_chain_b64: crate::nras::tests::gpu_chain_b64(
                    &crate::nras::tests::GPU_SERIAL_BYTES,
                ),
                gpu_cc_off: false,
                gpu_delay: std::time::Duration::ZERO,
                gpu_cc_state: CcSystemState::PRODUCTION,
                nras_lie: NrasLie::None,
                nras_down: false,
            }
        }

        pub(crate) fn rtmr(&self) -> [[u8; 48]; 4] {
            self.rtmr_in_quote
                .unwrap_or_else(|| crate::ccel::replay(&self.events).expect("replayable log"))
        }
    }

    impl TdEvidence for FakeTd {
        fn tdx(&self, report_data: &[u8; 64], now_unix: u64) -> anyhow::Result<Collected> {
            let rd = self.report_data_override.unwrap_or(*report_data);
            Ok(Collected {
                environment: ExecutionEnvironment::ConfidentialVm {
                    cpu: TeeTechnology::IntelTdx,
                },
                evidence: AttestationEvidence {
                    format: EvidenceFormat::TdxQuote,
                    payload: self
                        .intel
                        .quote(&rd, &self.mrtd, &self.rtmr(), self.td_attributes),
                    collected_at_unix: now_unix,
                },
            })
        }

        fn ccel(&self) -> anyhow::Result<Vec<u8>> {
            Ok(crate::ccel::encode(&self.events))
        }

        fn gpu<'a>(
            &'a self,
            gpu_nonce: &'a [u8; 32],
        ) -> BoxFuture<'a, Result<GpuAttested, GpuError>> {
            Box::pin(async move {
                let nonce = self
                    .gpu_nonce_override
                    .clone()
                    .unwrap_or_else(|| hex(gpu_nonce));
                Ok(self
                    .nvidia
                    .attest(&nonce, self.now, self.other_nras_signer.as_ref()))
            })
        }

        fn gpu_evidence(
            &self,
            gpu_nonce: &[u8; 32],
        ) -> Result<Vec<crate::nras_client::GpuEvidence>, GpuError> {
            std::thread::sleep(self.gpu_delay);
            if self.gpu_cc_off {
                return Err(GpuError {
                    code: 74,
                    message: "GPU evidence collection failed: NVML: Not Supported".into(),
                });
            }
            check_cc_state(&self.gpu_cc_state)?;
            Ok(vec![crate::nras_client::GpuEvidence {
                evidence: b64(&synthetic_report(&hex(gpu_nonce))),
                certificate: self.gpu_chain_b64.clone(),
            }])
        }

        fn nras<'a>(
            &'a self,
            gpu_nonce: &'a [u8; 32],
            evidence: &'a [crate::nras_client::GpuEvidence],
        ) -> BoxFuture<'a, Result<NrasVerdict, GpuError>> {
            Box::pin(async move {
                if self.nras_down {
                    return Err(GpuError {
                        code: 75,
                        message: "NRAS attestation failed: connection refused".into(),
                    });
                }
                let request = serde_json::json!({
                    "nonce": hex(gpu_nonce),
                    "evidence_list": evidence,
                    "arch": "HOPPER",
                    "claims_version": "3.0",
                });
                Ok(NrasVerdict {
                    answer: self.nvidia.answer(&request, self.now, self.nras_lie),
                    jwks: self.nvidia.jwks(),
                })
            })
        }
    }

    /// Strict percent-encoding of a header value, as PCS sends its chains.
    pub(crate) fn percent_encode(s: &str) -> String {
        let mut out = String::new();
        for b in s.bytes() {
            if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
                out.push(b as char);
            } else {
                out.push_str(&format!("%{b:02X}"));
            }
        }
        out
    }

    /// A fake Intel PCS on the loopback. `date` is the `Date` header it
    /// sends (the clock check); `None` sends none. Like Intel's, it answers
    /// `update=early` with the production minimum evaluation and any other
    /// query with the `standard` one ([`STANDARD_EVALUATION`]); everything it
    /// serves was issued an hour before `date` ([`fresh`]).
    pub(crate) struct FakePcs {
        pub(crate) base_url: String,
        /// Where this PCS serves the root CA's CRL (Intel's is on another
        /// host, the root certificate's CRL distribution point).
        pub(crate) root_crl_url: String,
    }

    pub(crate) async fn fake_pcs(intel: &FakeIntel, date_unix: Option<u64>) -> FakePcs {
        let crl = intel.pck_crl(&[], fresh(date_unix.unwrap_or(3600)));
        fake_pcs_with_crl(intel, date_unix, Some(crl)).await
    }

    /// `crl`: the DER PCS serves for `pckcrl?ca=platform&encoding=der`; `None`
    /// serves no CRL at all (404).
    pub(crate) async fn fake_pcs_with_crl(
        intel: &FakeIntel,
        date_unix: Option<u64>,
        crl: Option<Vec<u8>>,
    ) -> FakePcs {
        let issued = fresh(date_unix.unwrap_or(3600)).0;
        let tcb_info = (
            intel.tcb_info_with(issued, STANDARD_EVALUATION, "UpToDate"),
            intel.tcb_info(issued),
        );
        let qe_identity = (
            intel.qe_identity_with(issued, STANDARD_EVALUATION),
            intel.qe_identity(issued),
        );
        fake_pcs_serving(intel, date_unix, crl, tcb_info, qe_identity).await
    }

    /// A fake PCS serving these bodies: tcbInfo and QE identity as
    /// (`standard`, `early`), the answers to a query without and with
    /// `update=early`.
    pub(crate) async fn fake_pcs_serving(
        intel: &FakeIntel,
        date_unix: Option<u64>,
        crl: Option<Vec<u8>>,
        tcb_info: (String, String),
        qe_identity: (String, String),
    ) -> FakePcs {
        let root_crl = Some(intel.root_crl_wide());
        fake_pcs_complete(intel, date_unix, crl, root_crl, tcb_info, qe_identity).await
    }

    /// [`fake_pcs_serving`] with the root CA's CRL chosen too: `None` serves
    /// none (404).
    pub(crate) async fn fake_pcs_complete(
        intel: &FakeIntel,
        date_unix: Option<u64>,
        crl: Option<Vec<u8>>,
        root_crl: Option<Vec<u8>>,
        tcb_info: (String, String),
        qe_identity: (String, String),
    ) -> FakePcs {
        use axum::http::HeaderValue;
        use axum::routing::get;
        let chain = percent_encode(&intel.collateral_chain_pem());
        let crl_chain = percent_encode(&intel.pck_crl_chain_pem());
        let date = Arc::new(std::sync::Mutex::new(date_unix));
        let respond = |header: &'static str,
                       bodies: (String, String),
                       date: Arc<std::sync::Mutex<Option<u64>>>,
                       chain: String| {
            move |axum::extract::RawQuery(q): axum::extract::RawQuery| async move {
                let early = q
                    .as_deref()
                    .is_some_and(|q| q.split('&').any(|kv| kv == "update=early"));
                let body = if early { bodies.1 } else { bodies.0 };
                // An empty body stands for a feed that does not answer.
                if body.is_empty() {
                    let mut r = axum::response::Response::new(axum::body::Body::empty());
                    *r.status_mut() = axum::http::StatusCode::SERVICE_UNAVAILABLE;
                    return r;
                }
                let mut resp = axum::response::Response::new(axum::body::Body::from(body));
                resp.headers_mut()
                    .insert(header, HeaderValue::from_str(&chain).unwrap());
                if let Some(d) = *date.lock().unwrap() {
                    resp.headers_mut().insert(
                        axum::http::header::DATE,
                        HeaderValue::from_str(&crate::remote_verify::http_date(d)).unwrap(),
                    );
                }
                resp
            }
        };
        let app = axum::Router::new()
            .route(
                "/tdx/certification/v4/tcb",
                get(respond(
                    "TCB-Info-Issuer-Chain",
                    tcb_info,
                    date.clone(),
                    chain.clone(),
                )),
            )
            .route(
                "/tdx/certification/v4/qe/identity",
                get(respond(
                    "SGX-Enclave-Identity-Issuer-Chain",
                    qe_identity,
                    date.clone(),
                    chain.clone(),
                )),
            );
        let app = match crl {
            None => app,
            Some(crl) => {
                let date = date.clone();
                app.route(
                    "/sgx/certification/v4/pckcrl",
                    get(move |axum::extract::RawQuery(q): axum::extract::RawQuery| {
                        let (crl, date, crl_chain) = (crl.clone(), date.clone(), crl_chain.clone());
                        async move {
                            if q.as_deref() != Some("ca=platform&encoding=der") {
                                let mut r =
                                    axum::response::Response::new(axum::body::Body::empty());
                                *r.status_mut() = axum::http::StatusCode::BAD_REQUEST;
                                return r;
                            }
                            let mut resp =
                                axum::response::Response::new(axum::body::Body::from(crl));
                            let h = resp.headers_mut();
                            h.insert(
                                axum::http::header::CONTENT_TYPE,
                                HeaderValue::from_static("application/pkix-crl"),
                            );
                            h.insert(
                                "SGX-PCK-CRL-Issuer-Chain",
                                HeaderValue::from_str(&crl_chain).unwrap(),
                            );
                            if let Some(d) = *date.lock().unwrap() {
                                h.insert(
                                    axum::http::header::DATE,
                                    HeaderValue::from_str(&crate::remote_verify::http_date(d))
                                        .unwrap(),
                                );
                            }
                            resp
                        }
                    }),
                )
            }
        };
        let app = match root_crl {
            None => app,
            Some(root_crl) => app.route(
                "/rootcacrl.der",
                get(move || {
                    let root_crl = root_crl.clone();
                    async move {
                        let mut resp =
                            axum::response::Response::new(axum::body::Body::from(root_crl));
                        resp.headers_mut().insert(
                            axum::http::header::CONTENT_TYPE,
                            HeaderValue::from_static("application/pkix-crl"),
                        );
                        resp
                    }
                }),
            ),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        FakePcs {
            base_url: format!("http://{addr}"),
            root_crl_url: format!("http://{addr}/rootcacrl.der"),
        }
    }

    /// A fake Google bucket: objects by name, 404 otherwise.
    pub(crate) async fn fake_bucket(objects: Vec<(String, Vec<u8>)>) -> String {
        use axum::extract::Path;
        let objects = Arc::new(objects);
        let app = axum::Router::new().route(
            "/{*path}",
            axum::routing::get(move |Path(path): Path<String>| {
                let objects = objects.clone();
                async move {
                    match objects.iter().find(|(name, _)| *name == path) {
                        Some((_, bytes)) => (axum::http::StatusCode::OK, bytes.clone()),
                        None => (axum::http::StatusCode::NOT_FOUND, Vec::new()),
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    /// The synthetic firmware's objects in a bucket: our Google's endorsement
    /// of the synthetic binary for each of `mrtds`, under that MRTD's name,
    /// and `firmware` under the synthetic binary's SHA-384 (the binary
    /// itself, or a tampered copy a test serves in its place).
    pub(crate) fn endorsed_objects(firmware: &[u8], mrtds: &[[u8; 48]]) -> Vec<(String, Vec<u8>)> {
        use sha2::Digest as _;
        let google = crate::gce_endorsement::tests::fake_google::fake_google();
        let synthetic = crate::tdvf::tests::synthetic();
        let mut objects: Vec<(String, Vec<u8>)> = mrtds
            .iter()
            .map(|m| {
                (
                    crate::gce_endorsement::object_name(m),
                    google.endorse_binary(synthetic, mrtds),
                )
            })
            .collect();
        objects.push((
            crate::tdvf::object_name(&crate::ccel::hex(&sha2::Sha384::digest(synthetic))).unwrap(),
            firmware.to_vec(),
        ));
        objects
    }

    /// The bucket of the honest TD's world (`FakeTd::honest`): our Google's
    /// endorsement of the synthetic firmware for its MRTD, and the binary.
    pub(crate) async fn bucket_with_endorsed() -> String {
        fake_bucket(endorsed_objects(
            crate::tdvf::tests::synthetic(),
            &[crate::tdvf::tests::synthetic_mrtd()],
        ))
        .await
    }

    /// An SEV-SNP guest on synthetic evidence: the TEST AMD chain, a TEST
    /// Genoa report over what the pipeline writes, `edit` applied first.
    pub(crate) struct FakeSnp {
        pub(crate) nvidia: Arc<FakeNvidia>,
        pub(crate) edit: fn(&mut crate::snp::tests::Spec),
        pub(crate) report_data_override: Option<[u8; 64]>,
        /// The TEST Genoa CRL by default; `None`: the image left none.
        pub(crate) crl: Option<Vec<u8>>,
    }

    impl TdEvidence for FakeSnp {
        fn tdx(&self, report_data: &[u8; 64], now_unix: u64) -> anyhow::Result<Collected> {
            use crate::snp::tests::{sign, Spec, CURRENT, VCEK_CURRENT};
            let mut spec = Spec::genoa();
            spec.report_data = self.report_data_override.unwrap_or(*report_data);
            (self.edit)(&mut spec);
            Ok(Collected {
                environment: ExecutionEnvironment::ConfidentialVm {
                    cpu: TeeTechnology::AmdSevSnp,
                },
                evidence: AttestationEvidence {
                    format: EvidenceFormat::SevSnpReport,
                    payload: crate::snp::evidence_payload(
                        &sign(spec.bytes(), CURRENT),
                        VCEK_CURRENT,
                    ),
                    collected_at_unix: now_unix,
                },
            })
        }

        /// An SNP guest has no CCEL: the pipeline must not ask.
        fn ccel(&self) -> anyhow::Result<Vec<u8>> {
            anyhow::bail!("no CCEL on an SEV-SNP guest")
        }

        fn snp_crl(&self) -> anyhow::Result<Vec<u8>> {
            self.crl
                .clone()
                .ok_or_else(|| anyhow::anyhow!("AMD CRL /run/x0ne/snp-crl.der: absent"))
        }

        fn gpu<'a>(
            &'a self,
            gpu_nonce: &'a [u8; 32],
        ) -> BoxFuture<'a, Result<GpuAttested, GpuError>> {
            Box::pin(async move {
                Ok(self
                    .nvidia
                    .attest(&hex(gpu_nonce), crate::snp::tests::NOW, None))
            })
        }
    }

    pub(crate) fn snp_guest(edit: fn(&mut crate::snp::tests::Spec)) -> FakeSnp {
        FakeSnp {
            nvidia: Arc::new(FakeNvidia::new()),
            edit,
            report_data_override: None,
            crl: Some(crate::snp::tests::TEST_GENOA_CRL.to_vec()),
        }
    }

    /// Production anchors but the TEST AMD chain and the fake NVIDIA pins.
    pub(crate) fn snp_anchors(nvidia: &FakeNvidia) -> TdAnchors {
        let mut a = TdAnchors::production();
        a.snp = crate::snp::tests::test_anchors();
        a.nras_intermediate_pins = nvidia.intermediate_pins();
        a
    }
}

#[cfg(test)]
mod tests {
    /// SECURITY: the GPU evidence of an e2e key document comes from GPUs in
    /// production confidential computing with devtools off, and from no
    /// other state: CC off, devtools on, the simulated environment, an
    /// unavailable one, or a value nvml.h does not define (review of
    /// 2026-10-27, finding 3).
    #[test]
    fn only_production_cc_with_devtools_off_gives_e2e_gpu_evidence() {
        use super::{check_cc_state, CcSystemState};
        let ok = CcSystemState::PRODUCTION;
        check_cc_state(&ok).expect("production CC, devtools off");
        for (state, why) in [
            (
                CcSystemState {
                    cc_feature: 0,
                    ..ok
                },
                "CC feature disabled",
            ),
            (
                CcSystemState {
                    cc_feature: 2,
                    ..ok
                },
                "CC feature undefined",
            ),
            (
                CcSystemState {
                    dev_tools_mode: 1,
                    ..ok
                },
                "devtools on",
            ),
            (
                CcSystemState {
                    dev_tools_mode: 7,
                    ..ok
                },
                "devtools undefined",
            ),
            (
                CcSystemState {
                    environment: 0,
                    ..ok
                },
                "environment unavailable",
            ),
            (
                CcSystemState {
                    environment: 1,
                    ..ok
                },
                "environment simulated",
            ),
            (
                CcSystemState {
                    environment: 3,
                    ..ok
                },
                "environment undefined",
            ),
        ] {
            let e = check_cc_state(&state).expect_err(why);
            assert_eq!(e.code, 74, "{why}");
            assert!(e.message.contains("NVML CC state"), "{why}: {}", e.message);
        }
    }

    /// NVML as the tests simulate it; the H100's own answers are read on
    /// the next run on one, never here. `None` is a call that fails.
    struct FakeNvml {
        state: Option<super::CcSystemState>,
        archs: Option<Vec<u32>>,
        /// Evidence per GPU it gives; `None`: as many as `archs`.
        evidence_count: Option<usize>,
        reads: std::sync::atomic::AtomicUsize,
    }

    impl FakeNvml {
        fn h100() -> Self {
            Self {
                state: Some(super::CcSystemState::PRODUCTION),
                archs: Some(vec![9]),
                evidence_count: None,
                reads: std::sync::atomic::AtomicUsize::new(0),
            }
        }
        fn reads(&self) -> usize {
            self.reads.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl super::LocalGpus for FakeNvml {
        fn cc_system_state(&self) -> anyhow::Result<super::CcSystemState> {
            self.state
                .ok_or_else(|| anyhow::anyhow!("NVML error 3 (NotSupported)"))
        }
        fn architectures(&self) -> anyhow::Result<Vec<u32>> {
            self.archs
                .clone()
                .ok_or_else(|| anyhow::anyhow!("NVML error 3 (NotSupported)"))
        }
        fn read_evidence(
            &self,
            _: &[u8; 32],
        ) -> anyhow::Result<Vec<crate::nras_client::GpuEvidence>> {
            self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let n = self
                .evidence_count
                .unwrap_or_else(|| self.archs.as_ref().map_or(1, Vec::len));
            Ok((0..n)
                .map(|i| crate::nras_client::GpuEvidence {
                    evidence: format!("report-{i}"),
                    certificate: format!("chain-{i}"),
                })
                .collect())
        }
    }

    /// An H100 (NVML architecture 9) in production CC with devtools off,
    /// declared HOPPER: its evidence is read, once.
    #[test]
    fn a_hopper_in_production_cc_gives_its_evidence() {
        let gpus = FakeNvml::h100();
        let ev = super::judged_gpu_evidence(&gpus, &[7; 32], "HOPPER").expect("served");
        assert_eq!(ev.len(), 1);
        assert_eq!(gpus.reads(), 1);
    }

    /// SECURITY: a GPU outside production CC, or devtools on, gives no
    /// evidence on any path, `run` included: refused 74 before any report
    /// is read (until 2026-10-30 only the e2e key document judged it).
    #[test]
    fn no_evidence_is_read_from_a_gpu_outside_production_cc() {
        let ok = super::CcSystemState::PRODUCTION;
        for (state, why) in [
            (
                Some(super::CcSystemState {
                    dev_tools_mode: 1,
                    ..ok
                }),
                "devtools on",
            ),
            (
                Some(super::CcSystemState {
                    cc_feature: 0,
                    ..ok
                }),
                "CC off",
            ),
            (
                Some(super::CcSystemState {
                    environment: 1,
                    ..ok
                }),
                "simulated",
            ),
            (None, "state unreadable"),
        ] {
            let gpus = FakeNvml {
                state,
                ..FakeNvml::h100()
            };
            let e = super::judged_gpu_evidence(&gpus, &[7; 32], "HOPPER").expect_err(why);
            assert_eq!(e.code, 74, "{why}");
            assert!(e.message.contains("NVML CC state"), "{why}: {}", e.message);
            assert_eq!(gpus.reads(), 0, "{why}: evidence read before the refusal");
        }
    }

    /// SECURITY: the architecture NRAS is told is the one NVML reads, not
    /// the one `--gpu-arch` claims. Another, an unknown one, none, or one
    /// GPU of eight that differs: refused 74 before any report is read.
    #[test]
    fn an_architecture_nvml_does_not_confirm_is_refused() {
        for (archs, declared, why) in [
            (Some(vec![10]), "HOPPER", "a Blackwell declared HOPPER"),
            (Some(vec![9]), "BLACKWELL", "a Hopper declared BLACKWELL"),
            (Some(vec![9, 7]), "HOPPER", "an Ampere beside the Hopper"),
            (Some(vec![8]), "HOPPER", "an Ada"),
            (
                Some(vec![0xffff_ffff]),
                "HOPPER",
                "NVML_DEVICE_ARCH_UNKNOWN",
            ),
            (Some(vec![]), "HOPPER", "no GPU"),
            (Some(vec![9]), "hopper", "the claim spelt as NRAS does not"),
            (None, "HOPPER", "architecture unreadable"),
        ] {
            let gpus = FakeNvml {
                archs,
                ..FakeNvml::h100()
            };
            let e = super::judged_gpu_evidence(&gpus, &[7; 32], declared).expect_err(why);
            assert_eq!(e.code, 74, "{why}");
            assert!(e.message.contains("architecture"), "{why}: {}", e.message);
            assert_eq!(gpus.reads(), 0, "{why}: evidence read before the refusal");
        }
        let gpus = FakeNvml {
            archs: Some(vec![10, 10]),
            ..FakeNvml::h100()
        };
        let ev = super::judged_gpu_evidence(&gpus, &[7; 32], "BLACKWELL").expect("two B200");
        assert_eq!(ev.len(), 2);
    }

    /// A value `--gpu-arch` was given that names no architecture is not
    /// printed back: argv may hold what was meant for another flag.
    #[test]
    fn an_unknown_gpu_arch_claim_is_not_printed() {
        let e = super::judged_gpu_evidence(&FakeNvml::h100(), &[7; 32], "5ec75ec7")
            .expect_err("no such architecture");
        assert!(!e.message.contains("5ec75ec7"), "{}", e.message);
        assert!(e.message.contains("HOPPER"), "{}", e.message);
    }

    /// nvml.h of CUDA 13.3: NVML_DEVICE_ARCH_HOPPER 9, _BLACKWELL 10; the
    /// others have no NRAS name here.
    #[test]
    fn nvml_architectures_are_named_as_nras_names_them() {
        assert_eq!(super::nras_arch(9), Some("HOPPER"));
        assert_eq!(super::nras_arch(10), Some("BLACKWELL"));
        for other in [2, 7, 8, 13, 0xffff_ffff] {
            assert_eq!(super::nras_arch(other), None, "{other}");
        }
    }

    /// Evidence for more or fewer GPUs than NVML named is a library that
    /// changed its mind between two calls: refused, not sent.
    #[test]
    fn evidence_for_another_count_of_gpus_is_refused() {
        let gpus = FakeNvml {
            archs: Some(vec![9, 9]),
            evidence_count: Some(1),
            ..FakeNvml::h100()
        };
        let e = super::judged_gpu_evidence(&gpus, &[7; 32], "HOPPER").expect_err("1 of 2");
        assert_eq!(e.code, 74);
        assert!(e.message.contains("2 GPUs"), "{}", e.message);
    }

    use super::fixtures::*;
    use super::*;
    use rcgen::{KeyPair, PKCS_ECDSA_P256_SHA256};
    use std::sync::Arc;
    use x0ne_confidential::model::{ExecutionEnvironment, TeeTechnology};

    const NOW: u64 = crate::gce_endorsement::tests::NOW;

    fn binding() -> (SessionIdentity, BindingV2) {
        let id = SessionIdentity::generate([1u8; 32], [2u8; 32]);
        let b = BindingV2::of(&id, [3u8; 32], [4u8; 32]);
        (id, b)
    }

    /// The synthetic Intel passes the production verifiers: chain, live
    /// collateral, QE identity. Without this the refusal tests would refuse
    /// for the wrong reason.
    #[tokio::test]
    async fn the_fake_intel_passes_the_production_verifiers() {
        let intel = FakeIntel::new();
        let (_, b) = binding();
        let quote = intel.quote(&b.tdx_report_data(), &[7; 48], &[[0; 48]; 4], [0; 8]);
        let chain = crate::pcs::verify_tdx_chain(&quote, &intel.anchor(), NOW).expect("chain");
        assert_eq!(chain.report_data, b.tdx_report_data());
        let pck = crate::tcb::pck_tcb(&quote).expect("extension");
        assert_eq!((pck.fmspc_hex.as_str(), pck.pcesvn), (FMSPC_HEX, PCESVN));
        let (platform, qe) = crate::tcb::verify_live(
            &quote,
            &intel.tcb_info(fresh(NOW).0),
            &intel.collateral_chain_pem(),
            &intel.qe_identity(fresh(NOW).0),
            &intel.collateral_chain_pem(),
            &intel.anchor(),
            NOW,
            &crate::tcb::TcbPolicy::production(),
        )
        .expect("live");
        assert!(platform.accepted() && qe.accepted());
    }

    /// The TD's anchors on our Intel and NVIDIA, the root CRL where `pcs` serves it.
    fn td_anchors(intel: &FakeIntel, nvidia: &FakeNvidia, pcs: &FakePcs) -> TdAnchors {
        TdAnchors {
            tdx: intel.anchor(),
            nras_intermediate_pins: nvidia.intermediate_pins(),
            tcb_policy: crate::tcb::TcbPolicy::production(),
            root_crl_url: pcs.root_crl_url.clone(),
            snp: crate::snp::SnpAnchors::production(),
            snp_policy: crate::snp::SnpPolicy::default(),
        }
    }

    /// On SEV-SNP the pipeline reads no event log and asks no Intel PCS (the
    /// URL is unreachable): the report verified to the AMD root opens the
    /// gate, with the GPU when the product asks for it. The document carries
    /// the evidence under `snp` and leaves the TDX quote empty.
    #[tokio::test]
    async fn attest_td_on_an_snp_guest_passes_the_gate_for_both_products() {
        let guest = snp_guest(|_| {});
        let anchors = snp_anchors(&guest.nvidia);
        let (id, b) = binding();
        let now = || crate::snp::tests::NOW;
        for gpu in [true, false] {
            let a = attest_td(&guest, &b, gpu, "http://127.0.0.1:9", &anchors, &now)
                .await
                .unwrap_or_else(|e| panic!("gpu={gpu}: {e}"));
            assert!(a.is_snp());
            assert_eq!(a.gpu.is_some(), gpu);
            assert!(a.ccel.is_empty());
            assert_eq!(a.live_tcb.0, SNP_LIVE_TCB);
            let want = if gpu {
                ExecutionEnvironment::ConfidentialVmWithGpu {
                    cpu: TeeTechnology::AmdSevSnp,
                    gpu_mode: x0ne_confidential::model::GpuCcMode::On,
                }
            } else {
                ExecutionEnvironment::ConfidentialVm {
                    cpu: TeeTechnology::AmdSevSnp,
                }
            };
            assert_eq!(a.proof.environment(), &want);
            let config = crate::cloud_config::CloudConfig::new(
                "qwen3:0.6b",
                [5; 32],
                [6; 32],
                crate::snp::tests::NOW + 600,
                if gpu { "cpu+gpu" } else { "cpu" },
                443,
                64,
            )
            .to_loaded()
            .unwrap();
            let model = ModelReport {
                name: "qwen3:0.6b".into(),
                digest_hex: "00".repeat(32),
                files: vec![],
            };
            let doc = Document::build(&a, &id, &b, &config, &model, "test");
            assert!(doc.tdx.quote_b64.is_empty() && doc.tdx.ccel_b64.is_empty());
            let snp = doc.snp.as_ref().expect("the SNP evidence is published");
            let payload = unb64(&snp.evidence_b64).unwrap();
            assert_eq!(payload, a.evidence.payload);
            let (report, _) = crate::snp::split_evidence(&payload).unwrap();
            assert_eq!(
                x0ne_confidential::snp_report::parse(report)
                    .unwrap()
                    .report_data,
                b.snp_report_data()
            );
            assert_eq!(doc.gpu.len(), usize::from(gpu));
            let json = serde_json::to_value(&doc).unwrap();
            assert!(json["snp"]["evidence_b64"].is_string());
            let back: Document = serde_json::from_value(json).unwrap();
            assert_eq!(back, doc);
        }
    }

    /// A TDX document keeps its shape: no `snp` key at all.
    #[test]
    fn a_tdx_document_has_no_snp_key() {
        let session = SessionDoc {
            schema: SESSION_SCHEMA.into(),
            quote_b64: "AA==".into(),
            snp: None,
            gpu: None,
        };
        let json = serde_json::to_value(&session).unwrap();
        assert!(json.get("snp").is_none(), "{json}");
        let old: SessionDoc = serde_json::from_value(
            serde_json::json!({"schema": SESSION_SCHEMA, "quote_b64": "AA=="}),
        )
        .unwrap();
        assert_eq!(old, session);
    }

    /// SECURITY: the SNP guest refuses to serve when launched with DEBUG or a
    /// migration agent allowed, at a VMPL other than 0, when its report
    /// answers another binding, when its GPU token answers another nonce,
    /// and when it has no evidence at all; each refusal names its cause.
    #[tokio::test]
    async fn the_snp_guest_refuses_its_own_bad_evidence() {
        use x0ne_confidential::snp_report::{POLICY_DEBUG, POLICY_MIGRATE_MA};
        let (_, b) = binding();
        let now = || crate::snp::tests::NOW;
        let cases: [(fn(&mut crate::snp::tests::Spec), &str); 3] = [
            (|s| s.policy |= POLICY_DEBUG, "DEBUG"),
            (|s| s.policy |= POLICY_MIGRATE_MA, "MIGRATE_MA"),
            (|s| s.vmpl = 1, "VMPL"),
        ];
        for (edit, name) in cases {
            let guest = snp_guest(edit);
            let anchors = snp_anchors(&guest.nvidia);
            for gpu in [true, false] {
                let e = attest_td(&guest, &b, gpu, "http://127.0.0.1:9", &anchors, &now)
                    .await
                    .expect_err(name);
                assert_eq!(e.code, 65, "{name}, gpu={gpu}: {e}");
                assert!(e.message.contains(name), "{name}, gpu={gpu}: {e}");
            }
        }
        let mut guest = snp_guest(|_| {});
        guest.report_data_override = Some([9; 64]);
        let anchors = snp_anchors(&guest.nvidia);
        for gpu in [true, false] {
            let e = attest_td(&guest, &b, gpu, "http://127.0.0.1:9", &anchors, &now)
                .await
                .unwrap_err();
            assert_eq!(e.code, 65, "{e}");
            assert!(e.message.contains("replay"), "{e}");
        }
        // The production AMD roots do not chain the TEST VCEK.
        let guest = snp_guest(|_| {});
        let mut production = snp_anchors(&guest.nvidia);
        production.snp = crate::snp::SnpAnchors::production();
        let e = attest_td(&guest, &b, false, "http://127.0.0.1:9", &production, &now)
            .await
            .unwrap_err();
        assert_eq!(e.code, 65, "{e}");
        // A GPU token for another nonce.
        struct OtherNonce(FakeSnp);
        impl TdEvidence for OtherNonce {
            fn tdx(&self, rd: &[u8; 64], now: u64) -> anyhow::Result<Collected> {
                self.0.tdx(rd, now)
            }
            fn ccel(&self) -> anyhow::Result<Vec<u8>> {
                self.0.ccel()
            }
            fn snp_crl(&self) -> anyhow::Result<Vec<u8>> {
                self.0.snp_crl()
            }
            fn gpu<'a>(&'a self, _: &'a [u8; 32]) -> BoxFuture<'a, Result<GpuAttested, GpuError>> {
                self.0.gpu(&[0xEE; 32])
            }
        }
        let guest = OtherNonce(snp_guest(|_| {}));
        let anchors = snp_anchors(&guest.0.nvidia);
        let e = attest_td(&guest, &b, true, "http://127.0.0.1:9", &anchors, &now)
            .await
            .unwrap_err();
        assert_eq!(e.code, 65, "{e}");
        assert!(e.message.contains("GPU attestation"), "{e}");
    }

    /// SECURITY: no AMD CRL left by the image, or bytes that are not AMD's
    /// CRL: the SNP guest does not serve, with or without the GPU.
    #[tokio::test]
    async fn the_snp_guest_refuses_without_a_valid_amd_crl() {
        let (_, b) = binding();
        let now = || crate::snp::tests::NOW;
        for gpu in [true, false] {
            let mut guest = snp_guest(|_| {});
            guest.crl = None;
            let anchors = snp_anchors(&guest.nvidia);
            let e = attest_td(&guest, &b, gpu, "http://127.0.0.1:9", &anchors, &now)
                .await
                .unwrap_err();
            assert_eq!(e.code, 76, "gpu={gpu}: {e}");
            assert!(e.message.contains("AMD CRL"), "gpu={gpu}: {e}");
            guest.crl = Some(b"not a CRL".to_vec());
            let e = attest_td(&guest, &b, gpu, "http://127.0.0.1:9", &anchors, &now)
                .await
                .unwrap_err();
            assert_eq!(e.code, 65, "gpu={gpu}: {e}");
        }
    }

    /// The image's CRL file: read when present, refused when absent, empty
    /// or past the bound.
    #[test]
    fn the_amd_crl_file_is_read_only_when_present_and_bounded() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("snp-crl.der");
        assert!(crate::evidence::read_snp_crl(&p).is_err());
        std::fs::write(&p, b"").unwrap();
        assert!(crate::evidence::read_snp_crl(&p).is_err());
        std::fs::write(&p, crate::snp::tests::TEST_GENOA_CRL).unwrap();
        assert_eq!(
            crate::evidence::read_snp_crl(&p).unwrap(),
            crate::snp::tests::TEST_GENOA_CRL
        );
        std::fs::write(&p, vec![0x30; 64 * 1024 + 1]).unwrap();
        assert!(crate::evidence::read_snp_crl(&p).is_err());
    }

    /// The whole TD pipeline on synthetic evidence, both products.
    #[tokio::test]
    async fn attest_td_passes_the_gate_for_both_products() {
        let intel = Arc::new(FakeIntel::new());
        let nvidia = Arc::new(FakeNvidia::new());
        let pcs = fake_pcs(&intel, Some(NOW)).await;
        let anchors = td_anchors(&intel, &nvidia, &pcs);
        let td = FakeTd::honest(intel.clone(), nvidia.clone(), NOW);
        let (_, b) = binding();
        for gpu in [true, false] {
            let a = attest_td(&td, &b, gpu, &pcs.base_url, &anchors, &|| NOW)
                .await
                .unwrap_or_else(|e| panic!("gpu={gpu}: {e}"));
            assert_eq!(a.gpu.is_some(), gpu);
            assert_eq!(a.live_tcb, ("UpToDate".to_string(), "UpToDate".to_string()));
            let want = if gpu {
                "ConfidentialVmWithGpu"
            } else {
                "ConfidentialVm"
            };
            assert!(format!("{:?}", a.proof.environment()).starts_with(want));
        }
    }

    /// SECURITY: the TD's own gate refuses a quote that does not answer its
    /// binding, a token for another nonce, and an unreachable PCS.
    #[tokio::test]
    async fn the_td_refuses_its_own_bad_evidence() {
        let intel = Arc::new(FakeIntel::new());
        let nvidia = Arc::new(FakeNvidia::new());
        let pcs = fake_pcs(&intel, Some(NOW)).await;
        let anchors = td_anchors(&intel, &nvidia, &pcs);
        let (_, b) = binding();
        let mut td = FakeTd::honest(intel.clone(), nvidia.clone(), NOW);
        td.report_data_override = Some([9; 64]);
        let e = attest_td(&td, &b, true, &pcs.base_url, &anchors, &|| NOW)
            .await
            .unwrap_err();
        assert_eq!(e.code, 65, "{e}");
        let mut td = FakeTd::honest(intel.clone(), nvidia.clone(), NOW);
        td.gpu_nonce_override = Some("ab".repeat(32));
        let e = attest_td(&td, &b, true, &pcs.base_url, &anchors, &|| NOW)
            .await
            .unwrap_err();
        assert_eq!(e.code, 65, "{e}");
        let td = FakeTd::honest(intel.clone(), nvidia.clone(), NOW);
        let e = attest_td(&td, &b, true, "http://127.0.0.1:9", &anchors, &|| NOW)
            .await
            .unwrap_err();
        assert_eq!(e.code, 76, "{e}");
    }

    /// SECURITY: a host that launches the genuine image as a migratable or a
    /// profiled TD gets no service: the TD's own gate refuses, by name, before
    /// any route opens.
    #[tokio::test]
    async fn the_td_refuses_to_serve_when_launched_migratable_or_profiled() {
        let intel = Arc::new(FakeIntel::new());
        let nvidia = Arc::new(FakeNvidia::new());
        let pcs = fake_pcs(&intel, Some(NOW)).await;
        let anchors = td_anchors(&intel, &nvidia, &pcs);
        let (_, b) = binding();
        for (attributes, name) in [
            ([0, 0, 0, 0x30, 0, 0, 0, 0], "MIGRATABLE"),
            ([0x20, 0, 0, 0x10, 0, 0, 0, 0], "PERF_PROF"),
        ] {
            let mut td = FakeTd::honest(intel.clone(), nvidia.clone(), NOW);
            td.td_attributes = attributes;
            for gpu in [true, false] {
                let e = attest_td(&td, &b, gpu, &pcs.base_url, &anchors, &|| NOW)
                    .await
                    .expect_err(name);
                assert_eq!(e.code, 65, "{name}, gpu={gpu}: {e}");
                assert!(e.message.contains(name), "{name}, gpu={gpu}: {e}");
            }
        }
    }

    /// SECURITY: an extracted PCK key (TEE.fail) signs quotes that pass every other check.
    /// A PCK on Intel's CRL is refused (65); a CRL missing, stale, not yet valid or not
    /// signed by the PCK CA is unusable collateral (76).
    #[tokio::test]
    async fn the_td_refuses_a_pck_on_intels_crl_and_an_unusable_crl() {
        let intel = Arc::new(FakeIntel::new());
        let nvidia = Arc::new(FakeNvidia::new());
        let td = FakeTd::honest(intel.clone(), nvidia.clone(), NOW);
        let (_, b) = binding();
        let other = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256)
            .unwrap()
            .serialize_der();
        let cases: Vec<(&str, Option<Vec<u8>>, i32, &str)> = vec![
            (
                "listed",
                Some(intel.pck_crl(&[vec![0x77; 16], intel.pck_serial()], fresh(NOW))),
                65,
                "revoked",
            ),
            ("missing", None, 76, "CRL"),
            (
                "stale",
                Some(intel.pck_crl(&[], years(2020, 2021))),
                76,
                "CRL",
            ),
            (
                "not yet valid",
                Some(intel.pck_crl(&[], years(2039, 2040))),
                76,
                "CRL",
            ),
            (
                "another signer",
                Some(intel.crl_signed_by(&other, &[], fresh(NOW))),
                76,
                "CRL",
            ),
        ];
        for (name, crl, code, says) in cases {
            let pcs = fake_pcs_with_crl(&intel, Some(NOW), crl).await;
            let anchors = td_anchors(&intel, &nvidia, &pcs);
            for gpu in [true, false] {
                let e = attest_td(&td, &b, gpu, &pcs.base_url, &anchors, &|| NOW)
                    .await
                    .err()
                    .unwrap_or_else(|| panic!("{name}, gpu={gpu}: attested"));
                assert_eq!(e.code, code, "{name}, gpu={gpu}: {e}");
                assert!(e.message.contains(says), "{name}, gpu={gpu}: {e}");
            }
        }
        // Control: the other platform's PCK on the CRL does not refuse this one.
        let (_, twin) = FakeIntel::two_platforms();
        assert_ne!(twin.pck_serial(), intel.pck_serial());
        let crl = intel.pck_crl(&[vec![0x1b; 16]], fresh(NOW));
        let pcs = fake_pcs_with_crl(&intel, Some(NOW), Some(crl)).await;
        attest_td(
            &td,
            &b,
            true,
            &pcs.base_url,
            &td_anchors(&intel, &nvidia, &pcs),
            &|| NOW,
        )
        .await
        .expect("a CRL listing another PCK");
    }

    /// SECURITY: the TD reads Intel's root CA CRL before it serves. Its own
    /// PCK CA or Intel's TCB signer listed: not serving, a revocation (65).
    /// A root CRL that is missing, expired, or not signed by the root: not
    /// serving, unusable (76). A root CRL that lists a leaf's serial, or
    /// another CA's, does not refuse.
    #[tokio::test]
    async fn the_td_refuses_to_serve_on_a_root_crl_listing_its_pck_ca_or_unusable() {
        let intel = Arc::new(FakeIntel::new());
        let nvidia = Arc::new(FakeNvidia::new());
        let td = FakeTd::honest(intel.clone(), nvidia.clone(), NOW);
        let (_, b) = binding();
        let year = (NOW - 200 * 86_400, NOW + 165 * 86_400);
        let other = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256)
            .unwrap()
            .serialize_der();
        let cases: Vec<(&str, Option<Vec<u8>>, i32, &str)> = vec![
            (
                "PCK CA listed",
                Some(intel.root_crl(&[vec![0x77; 16], intel.pck_ca_serial()], year)),
                65,
                "revoked",
            ),
            (
                "TCB signer listed",
                Some(intel.root_crl(&[intel.tcb_signing_serial()], year)),
                65,
                "revoked",
            ),
            ("missing", None, 76, "root CA CRL"),
            (
                "expired",
                Some(intel.root_crl(&[], (NOW - 400 * 86_400, NOW - 86_400))),
                76,
                "root CA CRL",
            ),
            (
                "signed by the PCK CA",
                Some(intel.root_crl_with(intel.pck_ca_key_der(), &[], year, |_| {})),
                76,
                "root CA CRL",
            ),
            (
                "another signer",
                Some(intel.root_crl_with(&other, &[], year, |_| {})),
                76,
                "root CA CRL",
            ),
        ];
        /// Fresh collateral and an empty PCK CRL, the root CRL as given.
        async fn serving(intel: &FakeIntel, root_crl: Option<Vec<u8>>) -> FakePcs {
            let issued = fresh(NOW).0;
            fake_pcs_complete(
                intel,
                Some(NOW),
                Some(intel.pck_crl(&[], fresh(NOW))),
                root_crl,
                (intel.tcb_info(issued), intel.tcb_info(issued)),
                (intel.qe_identity(issued), intel.qe_identity(issued)),
            )
            .await
        }
        for (name, root_crl, code, says) in cases {
            let pcs = serving(&intel, root_crl).await;
            let anchors = td_anchors(&intel, &nvidia, &pcs);
            for gpu in [true, false] {
                let e = attest_td(&td, &b, gpu, &pcs.base_url, &anchors, &|| NOW)
                    .await
                    .err()
                    .unwrap_or_else(|| panic!("{name}, gpu={gpu}: attested"));
                assert_eq!(e.code, code, "{name}, gpu={gpu}: {e}");
                assert!(e.message.contains(says), "{name}, gpu={gpu}: {e}");
            }
        }
        // Controls: a leaf's serial, or a serial of nothing here, on the root CRL.
        for listed in [intel.pck_serial(), vec![0x42; 16]] {
            let pcs = serving(&intel, Some(intel.root_crl(&[listed], year))).await;
            attest_td(
                &td,
                &b,
                true,
                &pcs.base_url,
                &td_anchors(&intel, &nvidia, &pcs),
                &|| NOW,
            )
            .await
            .expect("a root CRL listing nothing the root issued here");
        }
    }

    /// SECURITY (2026-10-28): the TD judges itself under its anchors' TCB policy
    /// before it serves. Transparent (production): accepted on Intel's
    /// `standard` answer, Intel's `early` answer named in the disclosure the
    /// document and every receipt carry; an `early` answer that cannot be
    /// read is said so, never "up to date". Strict: refused (76) on anything
    /// `early` does not accept at evaluation 22. Both refuse a `standard`
    /// answer that rates the platform OutOfDate or is two days old.
    #[tokio::test]
    async fn the_td_judges_itself_under_its_tcb_policy() {
        let intel = Arc::new(FakeIntel::new());
        let nvidia = Arc::new(FakeNvidia::new());
        let td = FakeTd::honest(intel.clone(), nvidia.clone(), NOW);
        let (_, b) = binding();
        let issued = fresh(NOW).0;
        let standard = intel.tcb_info_with(issued, STANDARD_EVALUATION, "UpToDate");
        let qe = (
            intel.qe_identity_with(issued, STANDARD_EVALUATION),
            intel.qe_identity(issued),
        );
        let of_28_10 = intel.tcb_info_of_2026_10_28(issued);
        // (case, tcbInfo (standard, early), transparent: Ok(line) or Err(says), strict: Err(says) or Ok)
        type Want = Result<&'static str, &'static str>;
        let cases: [(&str, (String, String), Want, Want); 7] = [
            (
                "control",
                (standard.clone(), intel.tcb_info(issued)),
                Ok("Platform TCB 2026-08-12, UpToDate under Intel's TCB evaluation 22; Intel lists no advisory against it (transparent TCB policy)"),
                Ok("Platform TCB 2026-08-12, UpToDate under Intel's TCB evaluation 22; Intel lists no advisory against it (strict TCB policy)"),
            ),
            (
                "the GCE hosts of 2026-10-28",
                of_28_10.clone(),
                Ok("Platform TCB 2026-02-11, OutOfDate under Intel's TCB evaluation 22; Intel advisories not fixed: INTEL-DOC-10003, INTEL-SA-01439, INTEL-SA-01442 (transparent TCB policy)"),
                Err("TCB status OutOfDate for FMSPC"),
            ),
            (
                "standard rates it OutOfDate",
                (
                    intel.tcb_info_with(issued, STANDARD_EVALUATION, "OutOfDate"),
                    intel.tcb_info_with(issued, 22, "OutOfDate"),
                ),
                Err("TCB status OutOfDate for FMSPC"),
                Err("TCB status OutOfDate for FMSPC"),
            ),
            (
                "standard issued two days ago",
                (
                    intel.tcb_info_with(NOW - 2 * 86_400, STANDARD_EVALUATION, "UpToDate"),
                    intel.tcb_info(NOW - 2 * 86_400),
                ),
                Err("tcbInfo was issued"),
                Err("tcbInfo was issued"),
            ),
            (
                "early does not answer",
                (standard.clone(), String::new()),
                Ok("Platform TCB 2026-08-12, UpToDate under Intel's standard TCB evaluation 20; Intel's latest TCB evaluation could not be read (early tcbInfo not fetched: "),
                Err("unreachable"),
            ),
            (
                "early of the standard evaluation only",
                (standard.clone(), standard.clone()),
                Ok("Intel's latest TCB evaluation could not be read (tcbInfo is Intel's TCB evaluation 20; this verifier requires at least 22"),
                Err("evaluation 20"),
            ),
            (
                "early issued two days ago",
                (standard.clone(), intel.tcb_info(NOW - 2 * 86_400)),
                Ok("Intel's latest TCB evaluation could not be read (tcbInfo was issued"),
                Err("tcbInfo was issued"),
            ),
        ];
        for (name, tcb, transparent, strict) in cases {
            let crl = Some(intel.pck_crl(&[], fresh(NOW)));
            let pcs = fake_pcs_serving(&intel, Some(NOW), crl, tcb, qe.clone()).await;
            for (policy, want) in [
                (crate::tcb::TcbPolicy::production(), transparent),
                (crate::tcb::TcbPolicy::strict(), strict),
            ] {
                let mut anchors = td_anchors(&intel, &nvidia, &pcs);
                anchors.tcb_policy = policy;
                let mode = policy.mode.as_str();
                let got = attest_td(&td, &b, false, &pcs.base_url, &anchors, &|| NOW).await;
                match (got, want) {
                    (Ok(a), Ok(line)) => {
                        let d = &a.tcb_disclosure;
                        assert!(
                            d.line.starts_with(line) || d.line.contains(line),
                            "{name}, {mode}: {}",
                            d.line
                        );
                        assert_eq!(d.policy, mode, "{name}");
                        assert!(
                            !d.line.to_lowercase().contains("up to date")
                                || d.line
                                    .contains("nothing here says the platform is up to date"),
                            "{name}, {mode}: {}",
                            d.line
                        );
                    }
                    (Err(e), Err(says)) => {
                        assert_eq!(e.code, 76, "{name}, {mode}: {e}");
                        assert!(e.message.contains(says), "{name}, {mode}: {e}");
                    }
                    (Ok(a), Err(_)) => {
                        panic!("{name}, {mode}: attested: {}", a.tcb_disclosure.line)
                    }
                    (Err(e), Ok(_)) => panic!("{name}, {mode}: refused: {e}"),
                }
            }
        }
        // The 28/10 view, field by field.
        let crl = Some(intel.pck_crl(&[], fresh(NOW)));
        let pcs = fake_pcs_serving(&intel, Some(NOW), crl, of_28_10, qe).await;
        let a = attest_td(
            &td,
            &b,
            false,
            &pcs.base_url,
            &td_anchors(&intel, &nvidia, &pcs),
            &|| NOW,
        )
        .await
        .expect("transparent");
        assert_eq!(a.live_tcb, ("UpToDate".to_string(), "UpToDate".to_string()));
        let i = a.tcb_disclosure.intel.as_ref().unwrap();
        assert_eq!(
            (i.accepted_under.as_str(), i.accepted_evaluation),
            ("standard", 20)
        );
        assert_eq!(i.platform_tcb_date, "2025-08-13T00:00:00Z");
        let l = i.latest.as_ref().unwrap();
        assert_eq!((l.platform.as_str(), l.tcb_evaluation), ("OutOfDate", 22));
        assert_eq!(l.platform_tcb_date, "2026-02-11T00:00:00Z");
        assert_eq!(l.advisories, ADVISORIES_2026_10_28);
        assert_eq!(l.judged_at_unix, NOW);
    }

    #[test]
    fn the_document_round_trips_and_names_the_catalog_files() {
        let catalog = crate::registry::Catalog::embedded();
        let entry = catalog.get("qwen3:0.6b").unwrap();
        let files = catalog_files(entry);
        assert!(files.windows(2).all(|w| w[0] <= w[1]), "sorted");
        assert!(files.iter().any(|(p, _)| p == "model.safetensors"));
    }

    /// NRAS answering `status` to every attest call, with its own words, and
    /// a JWKS of one key.
    async fn nras_answering(status: u16) -> String {
        use axum::routing::{get, post};
        let app = axum::Router::new()
            .route(
                "/v4/attest/gpu",
                post(move || async move {
                    (
                        axum::http::StatusCode::from_u16(status).unwrap(),
                        "NRAS's own words",
                    )
                }),
            )
            .route(
                "/.well-known/jwks.json",
                get(|| async { axum::Json(serde_json::json!({"keys": [{"kty": "EC"}]})) }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{addr}")
    }

    /// NRAS refusing the key of `--nras-key-file` (401, 403) has its own
    /// final exit: the image passes the same key at every start, 10 s apart.
    /// Under 75, a refused key called NVIDIA in a loop until the VM died.
    /// Without a key, a 5xx, a 429 or no answer stay 75: they may pass.
    #[tokio::test]
    async fn nras_refusing_the_given_key_is_its_own_final_exit() {
        let key = crate::nras_http::NrasKey::parse(b"nvapi-CANARY0123456789").unwrap();
        let evidence = vec![crate::nras_client::GpuEvidence {
            evidence: "ZXY=".into(),
            certificate: "Y2M=".into(),
        }];
        let closed = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            format!("http://{}", l.local_addr().unwrap())
        };
        let mut cases = Vec::new();
        for (status, keyed, code) in [
            (401, true, 103),
            (403, true, 103),
            (401, false, 75),
            (403, false, 75),
            (503, true, 75),
            (429, true, 75),
            (500, true, 75),
        ] {
            cases.push((nras_answering(status).await, Some(status), keyed, code));
        }
        cases.push((closed, None, true, 75));
        for (base, status, keyed, code) in cases {
            let source = SystemEvidence {
                nras_url: format!("{base}/v4/attest/gpu"),
                nras_jwks_url: format!("{base}/.well-known/jwks.json"),
                nras_key: keyed.then(|| key.clone()),
                gpu_arch: "HOPPER".into(),
            };
            let e = match source.nras(&[7; 32], &evidence).await {
                Ok(_) => panic!("{status:?}: served"),
                Err(e) => e,
            };
            assert_eq!(e.code, code, "{status:?}, key {keyed}: {}", e.message);
            if let Some(s) = status {
                assert!(e.message.contains(&s.to_string()), "{}", e.message);
            }
            assert!(!e.message.contains("CANARY"), "{}", e.message);
        }
    }

    /// The attest call and the JWKS wait for nothing from each other: NRAS
    /// sees both in flight at once, one round trip for a connection's GPU
    /// proof, not two. When both fail, the failure is the attest call's,
    /// also when it comes back last.
    #[tokio::test]
    async fn the_nras_call_and_its_jwks_are_asked_for_together() {
        use axum::routing::{get, post};
        use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
        use std::time::Duration;
        struct Seen {
            in_flight: AtomicUsize,
            peak: AtomicUsize,
        }
        // An NRAS waiting `attest` then answering `status`, its JWKS
        // waiting `jwks` then answering `jwks_status`.
        async fn nras(
            seen: Arc<Seen>,
            (attest, status): (Duration, u16),
            (jwks, jwks_status): (Duration, u16),
        ) -> String {
            async fn held(seen: &Seen, wait: Duration) {
                let now = seen.in_flight.fetch_add(1, SeqCst) + 1;
                seen.peak.fetch_max(now, SeqCst);
                tokio::time::sleep(wait).await;
                seen.in_flight.fetch_sub(1, SeqCst);
            }
            let (a, j) = (seen.clone(), seen);
            let app = axum::Router::new()
                .route(
                    "/v4/attest/gpu",
                    post(move || async move {
                        held(&a, attest).await;
                        (
                            axum::http::StatusCode::from_u16(status).unwrap(),
                            axum::Json(serde_json::json!([["JWT", "o.o.o"], {"GPU-0": "g.g.g"}])),
                        )
                    }),
                )
                .route(
                    "/.well-known/jwks.json",
                    get(move || async move {
                        held(&j, jwks).await;
                        (
                            axum::http::StatusCode::from_u16(jwks_status).unwrap(),
                            axum::Json(serde_json::json!({"keys": [{"kty": "EC"}]})),
                        )
                    }),
                );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });
            format!("http://{addr}")
        }
        let source = |base: &str| SystemEvidence {
            nras_url: format!("{base}/v4/attest/gpu"),
            nras_jwks_url: format!("{base}/.well-known/jwks.json"),
            nras_key: None,
            gpu_arch: "HOPPER".into(),
        };
        let evidence = vec![crate::nras_client::GpuEvidence {
            evidence: "ZXY=".into(),
            certificate: "Y2M=".into(),
        }];
        let ms = Duration::from_millis;
        let seen = Arc::new(Seen {
            in_flight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        });
        let base = nras(seen.clone(), (ms(50), 200), (ms(50), 200)).await;
        let started = std::time::Instant::now();
        let verdict = source(&base).nras(&[7; 32], &evidence).await.unwrap();
        println!(
            "NRAS, one connection's GPU proof: POST and GET at 50 ms each took {} ms, {} in flight at most",
            started.elapsed().as_millis(),
            seen.peak.load(SeqCst)
        );
        assert_eq!(
            verdict.answer,
            nras_answer("o.o.o", &[("GPU-0".into(), "g.g.g".into())])
        );
        assert_eq!(verdict.jwks["keys"][0]["kty"], "EC");
        // Counted on a longer wait: the two calls would have to arrive
        // 300 ms apart to be counted apart.
        let seen = Arc::new(Seen {
            in_flight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        });
        let base = nras(seen.clone(), (ms(300), 200), (ms(300), 200)).await;
        source(&base).nras(&[7; 32], &evidence).await.unwrap();
        assert_eq!(seen.peak.load(SeqCst), 2, "calls in flight at once");

        for (attest_wait, jwks_wait) in [(0, 0), (60, 0), (0, 60)] {
            let seen = Arc::new(Seen {
                in_flight: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
            });
            let base = nras(seen, (ms(attest_wait), 500), (ms(jwks_wait), 503)).await;
            let Err(e) = source(&base).nras(&[7; 32], &evidence).await else {
                panic!("served");
            };
            assert_eq!(e.code, 75, "{}", e.message);
            assert!(e.message.contains("NRAS answered 500"), "{}", e.message);
        }
        let base = nras(
            Arc::new(Seen {
                in_flight: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
            }),
            (ms(0), 200),
            (ms(0), 503),
        )
        .await;
        let Err(e) = source(&base).nras(&[7; 32], &evidence).await else {
            panic!("served");
        };
        assert!(
            e.message.contains("JWKS endpoint answered 503"),
            "{}",
            e.message
        );
    }
}
