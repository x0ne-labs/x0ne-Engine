//! The verifier chain.
//!
//! Two halves, both required before the gate opens:
//!
//! - **TDX (CPU)** — implemented: [`crate::pcs::verify_tdx_chain`] verifies
//!   the quote's signature chain to the pinned Intel root, and
//!   [`crate::tcb`] checks the platform's TCB status and the QE identity
//!   against LIVE Intel PCS collateral. The quote's `report_data` is
//!   compared to this session's challenge — a genuine chain answering
//!   someone else's challenge is a replay, and refused.
//! - **GPU (NVIDIA NRAS)** — implemented: [`crate::nras::verify_nras_token`]
//!   verifies the ES384 EAT under a pinned NVIDIA intermediate, with our
//!   nonce echoed and `overall == true`. Without the `nvml` feature the
//!   token is empty and this half refuses, so a CPU-only build fails closed
//!   rather than serving on half a proof.
//!
//! The structural verifier (`x0ne_confidential::policy::StructuralVerifier`)
//! stays available for `/attestation` diagnostics; it never opens the gate.

use x0ne_confidential::model::{
    AttestationEvidence, EvidenceFormat, EvidenceVerifier, Measurement, Verdict, VerifierStrength,
    WorkloadIdentity,
};
use x0ne_confidential::policy::{StructuralVerifier, VerificationPolicy};

use crate::pcs::{self, TrustAnchor};

/// The structural verifier, as it exists: for `/attestation` diagnostics.
pub fn structural(evidence: &AttestationEvidence, policy: &VerificationPolicy) -> Verdict {
    StructuralVerifier.verify(evidence, policy)
}

/// The cryptographic verifier over a raw TDX quote.
///
/// Verifies the DCAP chain (quote → attestation key → QE report → PCK chain →
/// pinned Intel root). Returns `Verified { strength: Cryptographic }` ONLY
/// for the CPU half — and since the composite below still refuses on the GPU
/// half, the server cannot serve yet. The verdict's verifier string carries
/// the limitations, so a reader of a receipt sees them.
pub fn cryptographic_tdx(
    evidence: &AttestationEvidence,
    policy: &VerificationPolicy,
    anchor: &TrustAnchor,
    // `expected_report_data`: the challenge this caller wrote into the quote
    // request. `None` verifies the CHAIN only, for tools inspecting a quote
    // they did not request; the serving path always passes it, and passing
    // `None` there would be the replay hole it exists to close.
    expected_report_data: Option<&[u8; 64]>,
    // `now_unix`: the instant every certificate's validity window is tested
    // against, all the way down the PCK chain.
    now_unix: u64,
) -> Verdict {
    // SECURITY: a TDX-only verdict can only ever describe
    // `ConfidentialVm` — no GPU, so `GpuCcMode::On` is not expressible here.
    // Returning `Verified` to a caller whose policy demands the GPU tells it
    // its policy was met by a verdict that cannot carry the property.
    //
    // `require_gpu_cc_on` was written `true` by `main.rs` and read in
    // exactly one place, the STRUCTURAL verifier
    // (`x0ne_confidential::policy`, the GpuCcMode branch); the cryptographic
    // path never looked at it. The eighteenth dead scaffold. The rule below
    // is that verifier's rule, restated at the gate that lacked it.
    //
    // The gate lives in the PUBLIC entry point only: the composite path
    // reaches the chain through `tdx_chain_verdict` below, and it is the one
    // that goes on to verify the GPU. Gating the shared step would refuse
    // precisely the callers that satisfy the policy.
    if policy.require_gpu_cc_on {
        return Verdict::Rejected {
            reason: "policy requires GPU confidential mode ON, and this verdict covers the \
                     TDX CPU only — verify the GPU too (`verify::cryptographic`)"
                .into(),
            verifier: "x0ne-infer::verify::cryptographic_tdx".into(),
        };
    }
    tdx_chain_verdict(
        evidence,
        policy,
        anchor,
        expected_report_data,
        now_unix,
        None,
    )
}

/// The CPU-only product's production entry point, mirroring
/// `cryptographic_after_live_tcb` for a machine with no GPU.
///
/// A TDX guest without an H100 is a real and much cheaper offer — 0.24 $/h
/// against 6.55 $ (a3-highgpu-1g spot, 2026-10-14) — and it is genuinely confidential: the RAM is encrypted
/// and the operator reads nothing. It is a DIFFERENT guarantee, not a
/// degraded one, and this is a separate entry point rather than a fallback
/// for exactly that reason: a fallback would let a GPU attestation failure
/// silently become a CPU-only service, which is the very substitution
/// the receipt's device field exists to make visible.
pub fn cryptographic_tdx_after_live_tcb(
    evidence: &AttestationEvidence,
    policy: &VerificationPolicy,
    now_unix: u64,
    expected_report_data: &[u8; 64],
    platform: &crate::tcb::TcbStatus,
    qe: &crate::tcb::TcbStatus,
) -> Verdict {
    cryptographic_tdx_after_live_tcb_with_anchor(
        evidence,
        policy,
        now_unix,
        &TrustAnchor::intel_production(),
        expected_report_data,
        platform,
        qe,
    )
}

/// Same path, anchor injected — so a test can drive the WHOLE CPU-only
/// production path on a fixture. Production has exactly one caller, above,
/// with the pin. Mirrors `cryptographic` / `cryptographic_with_anchors`.
pub fn cryptographic_tdx_after_live_tcb_with_anchor(
    evidence: &AttestationEvidence,
    policy: &VerificationPolicy,
    now_unix: u64,
    anchor: &TrustAnchor,
    expected_report_data: &[u8; 64],
    platform: &crate::tcb::TcbStatus,
    qe: &crate::tcb::TcbStatus,
) -> Verdict {
    if policy.require_gpu_cc_on {
        return Verdict::Rejected {
            reason: "policy requires GPU confidential mode ON, and this verdict covers the \
                     TDX CPU only — verify the GPU too (`verify::cryptographic`)"
                .into(),
            verifier: "x0ne-infer::verify::cryptographic_tdx_after_live_tcb".into(),
        };
    }
    tdx_chain_verdict(
        evidence,
        policy,
        anchor,
        Some(expected_report_data),
        now_unix,
        Some((platform.as_str(), qe.as_str())),
    )
}

/// The TDX half, with no opinion on the GPU policy — the step the composite
/// path shares. Not public: a caller outside this module wanting a TDX-only
/// verdict wants `cryptographic_tdx`, gate included.
/// The registers a TD Quote v4 carries, under the names this crate gives
/// them in a verdict and accepts in `--expect-measurement`.
///
/// What each one is a measurement OF — said here because pinning the wrong
/// one buys nothing: `mrtd` is the initial TD image, in practice the virtual
/// firmware the CLOUD provides (on GCP, Google's); `rtmr0` the firmware's
/// configuration; `rtmr1` and `rtmr2` what the firmware and bootloader
/// measured on the way up — kernel, command line, initrd; `rtmr3` is left to
/// the guest (dstack's convention extends it with the application). An image
/// YOU built is named by `rtmr1`/`rtmr2`, not by `mrtd`.
pub const TDX_REGISTERS: [&str; 5] = ["mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3"];

const TDX_MEASUREMENT_ALGORITHM: &str = "sha384";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `mrtd` and `rtmr0..3` of a quote whose signature the caller has ALREADY
/// verified: they sit inside the signed body, so they are the hardware's
/// statement, not the collector's.
pub fn tdx_measurements(quote: &[u8]) -> Result<Vec<Measurement>, String> {
    let claims =
        x0ne_confidential::tdx_quote::parse(quote).map_err(|e| format!("TD report: {e:?}"))?;
    let registers =
        std::iter::once(claims.mr_td.as_slice()).chain(claims.rtmr.iter().map(|r| r.as_slice()));
    Ok(TDX_REGISTERS
        .iter()
        .zip(registers)
        .map(|(name, value)| Measurement {
            register: (*name).into(),
            algorithm: TDX_MEASUREMENT_ALGORITHM.into(),
            value_hex: hex(value),
        })
        .collect())
}

/// `mrtd=<96 hex>` → an expectation. Refuses anything it would not be able
/// to enforce: an unknown register would never be found and a short value
/// would never match, and both would read as "the image changed" on the day
/// they fire, instead of as the typo they are today.
pub fn parse_expected_measurement(
    spec: &str,
) -> Result<x0ne_confidential::policy::ExpectedMeasurement, String> {
    let (register, value) = spec
        .split_once('=')
        .ok_or_else(|| "expected <register>=<hex>; the value given is not printed".to_string())?;
    if !TDX_REGISTERS.contains(&register) {
        return Err(format!(
            "not a TDX register before `=`: use one of {TDX_REGISTERS:?}; the value given is not printed"
        ));
    }
    if value.len() != 96 || !value.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "{register}: a SHA-384 measurement is 96 hex characters, got {}",
            value.len()
        ));
    }
    Ok(x0ne_confidential::policy::ExpectedMeasurement {
        register: register.into(),
        algorithm: TDX_MEASUREMENT_ALGORITHM.into(),
        value_hex: value.to_lowercase(),
    })
}

/// SECURITY (2026-09-18): `expected_measurements` was written `vec![]`
/// by `main`, documented, and read by the STRUCTURAL verifier alone — the
/// cryptographic path, the only one that opens the gate, never looked. Same
/// shape as `require_gpu_cc_on` and `accepted_cpu` had. Without
/// it a verdict says "genuine TDX hardware answered our challenge" and
/// nothing about WHAT is running there: whoever controls the guest image
/// controls the enclave's contents, attested and all.
///
/// Exact match on every expectation, the structural verifier's rule restated
/// here; an expectation naming a register the quote does not carry refuses.
fn unmet_expectation(measured: &[Measurement], policy: &VerificationPolicy) -> Option<String> {
    unmet_among(measured, &policy.expected_measurements)
}

/// The same rule for a CLIENT, which holds expectations and no policy.
pub fn unmet_among(
    measured: &[Measurement],
    expected: &[x0ne_confidential::policy::ExpectedMeasurement],
) -> Option<String> {
    expected.iter().find_map(|want| {
        let got = measured
            .iter()
            .find(|m| m.register == want.register && m.algorithm == want.algorithm);
        match got {
            None => Some(format!(
                "expected measurement '{}' ({}) is not in the quote",
                want.register, want.algorithm
            )),
            Some(m) if !m.value_hex.eq_ignore_ascii_case(&want.value_hex) => Some(format!(
                "measurement '{}' mismatch: pinned {}, the hardware reports {} — \
                 genuine TDX, but not the image this policy names",
                want.register, want.value_hex, m.value_hex
            )),
            Some(_) => None,
        }
    })
}

/// What the verdict must say about pinning, so that "verified" is never read
/// as more than it proves.
fn pinning_note(policy: &VerificationPolicy) -> String {
    let pinned: Vec<&str> = policy
        .expected_measurements
        .iter()
        .map(|m| m.register.as_str())
        .collect();
    if pinned.is_empty() {
        "no launch measurement pinned by this server: this proves genuine hardware, not a \
         particular image — mrtd and rtmr0-3 are in this verdict for the CLIENT to compare"
            .into()
    } else {
        format!("pinned and matched: {}", pinned.join(", "))
    }
}

fn tdx_chain_verdict(
    evidence: &AttestationEvidence,
    policy: &VerificationPolicy,
    anchor: &TrustAnchor,
    expected_report_data: Option<&[u8; 64]>,
    now_unix: u64,
    // The measured Intel PCS statuses, when the live half has run. `None`
    // keeps `pcs::LIMITATIONS` verbatim — including "TCB status and
    // revocation not checked against live Intel PCS", which would be a lie
    // once it HAS been checked. The composite path passes `None` here and
    // builds its own verifier string from both halves; this argument exists
    // for the CPU-only product, which has no second half to say it for it.
    live_tcb: Option<(&str, &str)>,
) -> Verdict {
    // SECURITY: every verdict this function can produce says the CPU
    // is Intel TDX — that is what a DCAP quote proves and the only thing it
    // proves. A policy that does not list TDX among the technologies it
    // accepts is therefore not met, whichever path called us.
    //
    // Until 2026-08-22 nobody looked. `accepted_cpu` was read by the
    // STRUCTURAL verifier alone (`x0ne_confidential::policy`), exactly like
    // `require_gpu_cc_on` used to be. The tell was in our own tests: the
    // policy they drove this path with said `accepted_cpu: vec![]` — accept
    // NO cpu technology at all — and the verdict came back `Verified`.
    //
    // Unlike the GPU rule, this one belongs on the shared step: the composite
    // path produces `ConfidentialVmWithGpu { cpu: IntelTdx, .. }`, so its CPU
    // claim is this one.
    if !policy
        .accepted_cpu
        .contains(&x0ne_confidential::model::TeeTechnology::IntelTdx)
    {
        return Verdict::Rejected {
            reason: format!(
                "policy accepts {:?} and this evidence proves Intel TDX",
                policy.accepted_cpu
            ),
            verifier: "x0ne-infer::verify::cryptographic_tdx".into(),
        };
    }
    if evidence.format != EvidenceFormat::TdxQuote {
        return Verdict::Rejected {
            reason: format!(
                "cryptographic TDX verification needs a raw TDX quote, got {:?}",
                evidence.format
            ),
            verifier: "x0ne-infer::verify::cryptographic_tdx".into(),
        };
    }
    // SECURITY: this entry point can only ever produce a TDX-only
    // environment — `ConfidentialVm`, no GPU. A caller whose policy says
    // `require_gpu_cc_on` and who receives `Verified` from here has been
    // told its policy was met when the verdict cannot express the property
    // at all. The field was written `true` by `main.rs` and read by exactly
    // one place, the STRUCTURAL verifier (`x0ne_confidential::policy`,
    // GpuCcMode check); the cryptographic path never looked at it. The
    // eighteenth dead scaffold, and the rule is the structural verifier's,
    // restated at the one gate that lacked it.
    match pcs::verify_tdx_chain(&evidence.payload, anchor, now_unix) {
        Ok(v) if expected_report_data.is_some_and(|want| &v.report_data != want) => {
            // SECURITY: the chain is genuine and the quote is NOT ours.
            //
            // Without this comparison a quote captured on ANY current TDX
            // machine — this repository vendors one for tests — verifies to
            // the pinned Intel root and opens the gate, because the freshness
            // window compares `now` to a timestamp our own collector wrote.
            // The nonce is what ties a genuine platform to THIS process and
            // THIS session key; a verified chain without it proves a real
            // enclave exists somewhere, which is not what anyone is buying.
            //
            // Found by audit 2026-08-21: the check existed in
            // `binding::verify_binding` and in `x0ne_confidential::tdx_quote`,
            // was tested in both, and had no caller here — the seventeenth
            // dead scaffold in this repository, and the one carrying the
            // central claim.
            Verdict::Rejected {
                reason: "the TDX quote's report_data is not this session's challenge — \
                         a genuine platform, but not this one (replay)"
                    .into(),
                verifier: "x0ne-infer::verify::cryptographic_tdx".into(),
            }
        }
        Ok(v) => {
            let mut measurements = vec![Measurement {
                register: "pck_sha256".into(),
                algorithm: "sha256".into(),
                value_hex: v.pck_sha256_hex,
            }];
            match tdx_measurements(&evidence.payload) {
                Ok(m) => measurements.extend(m),
                Err(reason) => {
                    return Verdict::Rejected {
                        reason,
                        verifier: "x0ne-infer::verify::cryptographic_tdx".into(),
                    }
                }
            }
            if let Some(reason) = unmet_expectation(&measurements, policy) {
                return Verdict::Rejected {
                    reason,
                    verifier: "x0ne-infer::verify::cryptographic_tdx".into(),
                };
            }
            Verdict::Verified {
                identity: WorkloadIdentity {
                    name: policy.workload_name.clone(),
                    environment: x0ne_confidential::model::ExecutionEnvironment::ConfidentialVm {
                        cpu: x0ne_confidential::model::TeeTechnology::IntelTdx,
                    },
                    // In every receipt, under its signature: what lets a
                    // CLIENT pin the image without trusting this server's
                    // flags — an operator who controls the launch command
                    // simply does not pass `--expect-measurement`.
                    measurements,
                },
                verifier: match live_tcb {
                    Some((platform, qe)) => format!(
                        "x0ne-infer::pcs + x0ne-infer::tcb (live Intel PCS as of {now_unix}: \
                         platform {platform}, QE {qe}) (limitations: {}; {})",
                        crate::tcb::LIVE_LIMITATIONS.join("; "),
                        pinning_note(policy)
                    ),
                    None => format!(
                        "x0ne-infer::pcs (limitations: {}; {})",
                        v.limitations.join("; "),
                        pinning_note(policy)
                    ),
                },
                strength: VerifierStrength::Cryptographic,
            }
        }
        Err(rejected) => Verdict::Rejected {
            reason: rejected.to_string(),
            verifier: "x0ne-infer::pcs".into(),
        },
    }
}

/// The cryptographic verifier over an AMD SEV-SNP report, the twin of
/// [`cryptographic_tdx`]: the report and its VCEK
/// ([`crate::snp::evidence_payload`]) verified to a pinned AMD root under
/// `snp_policy`, `REPORT_DATA` compared to the caller's challenge, every
/// expected measurement enforced. A separate entry point, never a fallback:
/// a TDX failure does not become an SNP verdict, nor the reverse, and the
/// verdict names the technology it proves.
///
/// No TD produces SNP evidence yet, and the gate is not wired to this: it
/// is the verifier a client will hold for a host of its own.
pub fn cryptographic_snp(
    evidence: &AttestationEvidence,
    policy: &VerificationPolicy,
    snp_policy: &crate::snp::SnpPolicy,
    anchors: &crate::snp::SnpAnchors,
    // AMD's CRL for the report's product (`snp::kds_crl_url`), fetched by
    // the caller: no verdict without one.
    crl_der: &[u8],
    // `None` verifies the chain only, as for TDX; a serving or client path
    // always passes the challenge.
    expected_report_data: Option<&[u8; 64]>,
    now_unix: u64,
) -> Verdict {
    let rejected = |reason: String| Verdict::Rejected {
        reason,
        verifier: SNP_VERIFIER.into(),
    };
    // The TDX path's gates, restated: a CPU-only verdict cannot carry the GPU
    // property, and a policy that does not list SEV-SNP is not met by it.
    if policy.require_gpu_cc_on {
        return rejected(
            "policy requires GPU confidential mode ON, and an SEV-SNP report covers the CPU only"
                .into(),
        );
    }
    if !policy
        .accepted_cpu
        .contains(&x0ne_confidential::model::TeeTechnology::AmdSevSnp)
    {
        return rejected(format!(
            "policy accepts {:?} and this evidence proves AMD SEV-SNP",
            policy.accepted_cpu
        ));
    }
    if evidence.format != EvidenceFormat::SevSnpReport {
        return rejected(format!(
            "cryptographic SEV-SNP verification needs an SEV-SNP report, got {:?}",
            evidence.format
        ));
    }
    let v = match crate::snp::verify_snp_report(
        &evidence.payload,
        crl_der,
        anchors,
        snp_policy,
        now_unix,
    ) {
        Ok(v) => v,
        Err(e) => {
            return Verdict::Rejected {
                reason: e.to_string(),
                verifier: "x0ne-infer::snp".into(),
            }
        }
    };
    // SECURITY: as for TDX, a genuine report bound to another challenge
    // proves a guest somewhere, not this session.
    if expected_report_data.is_some_and(|want| v.report_data() != want) {
        return rejected(
            "the SEV-SNP report's REPORT_DATA is not this session's challenge — a genuine \
             guest, but not this one (replay)"
                .into(),
        );
    }
    let measurements = crate::snp::snp_measurements(&v);
    if let Some(reason) = crate::snp::unmet_snp(&measurements, &policy.expected_measurements) {
        return rejected(reason);
    }
    let pinned: Vec<&str> = policy
        .expected_measurements
        .iter()
        .map(|m| m.register.as_str())
        .collect();
    let pinning = if pinned.is_empty() {
        "no launch measurement pinned by this policy: this proves genuine hardware, not a \
         particular image — snp_measurement is in this verdict for the CLIENT to compare"
            .to_string()
    } else {
        format!("pinned and matched: {}", pinned.join(", "))
    };
    let t = v.reported_tcb;
    Verdict::Verified {
        identity: WorkloadIdentity {
            name: policy.workload_name.clone(),
            environment: x0ne_confidential::model::ExecutionEnvironment::ConfidentialVm {
                cpu: x0ne_confidential::model::TeeTechnology::AmdSevSnp,
            },
            measurements,
        },
        verifier: format!(
            "x0ne-infer::snp (AMD {} chain, report v{}, TCB SNP {:#04x} microcode {:#04x}, \
             minimum {}) (limitations: {}; {})",
            v.product.name(),
            v.report.version,
            t.snp,
            t.microcode,
            v.minimum_source,
            v.limitations.join("; "),
            pinning
        ),
        strength: VerifierStrength::Cryptographic,
    }
}

const SNP_VERIFIER: &str = "x0ne-infer::verify::cryptographic_snp";

/// The SEV-SNP composite: the report proves the CPU enclave, the NRAS token
/// the GPU. `REPORT_DATA` and the NRAS nonce come from one binding
/// (`BindingV2`), so both answer the same start-up.
pub struct SnpCompositeEvidence<'a> {
    pub snp: &'a AttestationEvidence,
    /// What the guest wrote as `REPORT_DATA`.
    pub expected_report_data: &'a [u8; 64],
    pub nras_token: &'a str,
    pub nras_jwks: &'a serde_json::Value,
    pub nonce_hex: &'a str,
    pub nras_per_gpu: &'a [(String, String)],
    pub local_gpu_serials: &'a [String],
    /// The OCSP responses stapled about the NRAS signing chain (DER) and the
    /// policy they are judged under (`nras_ocsp`): the verdict names both.
    pub nras_ocsp_responses: &'a [Vec<u8>],
    pub nras_ocsp: &'a crate::nras_ocsp::NrasOcspPolicy,
}

/// CPU **and** GPU on SEV-SNP, one rejection short-circuits: the report to a
/// pinned AMD root and AMD's CRL under `snp_policy`, then the NRAS token under
/// a pinned NVIDIA intermediate with this start-up's nonce. `Verified` names
/// `ConfidentialVmWithGpu { cpu: AmdSevSnp }`.
pub fn cryptographic_snp_with_gpu(
    ev: &SnpCompositeEvidence<'_>,
    policy: &VerificationPolicy,
    snp_policy: &crate::snp::SnpPolicy,
    anchors: &crate::snp::SnpAnchors,
    crl_der: &[u8],
    nras_pins: &[crate::nras::NrasIntermediatePin],
    now_unix: u64,
) -> Verdict {
    let cpu_policy = VerificationPolicy {
        require_gpu_cc_on: false,
        ..policy.clone()
    };
    let cpu = cryptographic_snp(
        ev.snp,
        &cpu_policy,
        snp_policy,
        anchors,
        crl_der,
        Some(ev.expected_report_data),
        now_unix,
    );
    let (identity, cpu_verifier) = match cpu {
        Verdict::Verified {
            identity, verifier, ..
        } => (identity, verifier),
        rejected @ Verdict::Rejected { .. } => return rejected,
    };
    match crate::nras::verify_nras_attestation(
        ev.nras_token,
        ev.nras_per_gpu,
        ev.local_gpu_serials,
        ev.nras_jwks,
        ev.nonce_hex,
        now_unix,
        nras_pins,
        ev.nras_ocsp_responses,
        ev.nras_ocsp,
    ) {
        Ok(gpu) => Verdict::Verified {
            identity: WorkloadIdentity {
                environment:
                    x0ne_confidential::model::ExecutionEnvironment::ConfidentialVmWithGpu {
                        cpu: x0ne_confidential::model::TeeTechnology::AmdSevSnp,
                        gpu_mode: x0ne_confidential::model::GpuCcMode::On,
                    },
                ..identity
            },
            // The report's own "CPU only" limitation is lifted by the GPU half.
            verifier: format!(
                "{} + x0ne-infer::nras ({} GPU(s) bound by ueid; {}; limitations: {})",
                cpu_verifier.replace(&format!("{}; ", crate::snp::CPU_ONLY), ""),
                gpu.gpu_ueids.len(),
                revocation_note(ev.nras_ocsp, &gpu),
                gpu.limitations.join("; ")
            ),
            strength: VerifierStrength::Cryptographic,
        },
        Err(rejected) => Verdict::Rejected {
            reason: format!("GPU attestation: {rejected}"),
            verifier: "x0ne-infer::nras".into(),
        },
    }
}

/// Everything the composite verdict needs: both pieces of evidence and the
/// nonce they must echo. The TDX quote proves the CPU enclave; the NRAS
/// token proves the GPU; the SAME nonce ties both to this start-up.
pub struct CompositeEvidence<'a> {
    pub tdx: &'a AttestationEvidence,
    pub nras_token: &'a str,
    pub nras_jwks: &'a serde_json::Value,
    pub nonce_hex: &'a str,
    /// The `report_data` the caller wrote into the quote request — the CPU
    /// half of the challenge, compared against what came back. Its GPU
    /// counterpart is `nonce_hex`, already checked by `nras`.
    pub expected_tdx_report_data: &'a [u8; 64],
    /// The per-GPU tokens NRAS answered with — verified like the
    /// overall one, not dropped.
    pub nras_per_gpu: &'a [(String, String)],
    /// Serial (decimal) of the attestation certificate read from each LOCAL
    /// GPU: what the tokens' `ueid` must name for the attested GPU
    /// to be ours. `nras::gpu_leaf_serial_decimal` over what we POSTed.
    pub local_gpu_serials: &'a [String],
    /// The OCSP responses stapled about the NRAS signing chain (DER) and the
    /// policy they are judged under (`nras_ocsp`): the verdict names both.
    pub nras_ocsp_responses: &'a [Vec<u8>],
    pub nras_ocsp: &'a crate::nras_ocsp::NrasOcspPolicy,
}

/// The composite verdict the gate consumes: CPU **and** GPU, one rejection
/// short-circuits. `Verified` only when the TDX chain verifies to the pinned
/// Intel root AND the NRAS token verifies under a pinned NVIDIA
/// intermediate with `overall == true` and our nonce echoed.
pub fn cryptographic(
    ev: &CompositeEvidence<'_>,
    policy: &VerificationPolicy,
    now_unix: u64,
) -> Verdict {
    cryptographic_with_anchors(
        ev,
        policy,
        now_unix,
        &TrustAnchor::intel_production(),
        &crate::nras::NVIDIA_NRAS_GPU_INTERMEDIATE_PINS,
        None,
    )
}

/// Production path once the LIVE half of the CPU proof has passed
/// (`tcb::verify_live`): the verdict's verifier string names the two
/// statuses instead of the "not checked against live Intel PCS" limitation
/// they lift.
pub fn cryptographic_after_live_tcb(
    ev: &CompositeEvidence<'_>,
    policy: &VerificationPolicy,
    now_unix: u64,
    platform: &crate::tcb::TcbStatus,
    qe: &crate::tcb::TcbStatus,
) -> Verdict {
    cryptographic_with_anchors(
        ev,
        policy,
        now_unix,
        &TrustAnchor::intel_production(),
        &crate::nras::NVIDIA_NRAS_GPU_INTERMEDIATE_PINS,
        Some((platform.as_str(), qe.as_str())),
    )
}

/// Same composition, anchors injected — so a test can drive the WHOLE
/// production path (both halves, the environment upgrade, the verdict) on
/// fixtures. Production has exactly one caller, above, with the pins.
pub fn cryptographic_with_anchors(
    ev: &CompositeEvidence<'_>,
    policy: &VerificationPolicy,
    now_unix: u64,
    tdx_anchor: &TrustAnchor,
    nras_pins: &[crate::nras::NrasIntermediatePin],
    live_tcb: Option<(&str, &str)>,
) -> Verdict {
    let tdx = tdx_chain_verdict(
        ev.tdx,
        policy,
        tdx_anchor,
        Some(ev.expected_tdx_report_data),
        now_unix,
        // `None`: this path discards the CPU verdict's verifier string and
        // writes its own from BOTH halves, below.
        None,
    );
    let identity = match tdx {
        Verdict::Verified { identity, .. } => identity,
        rejected @ Verdict::Rejected { .. } => return rejected,
    };
    match crate::nras::verify_nras_attestation(
        ev.nras_token,
        ev.nras_per_gpu,
        ev.local_gpu_serials,
        ev.nras_jwks,
        ev.nonce_hex,
        now_unix,
        nras_pins,
        ev.nras_ocsp_responses,
        ev.nras_ocsp,
    ) {
        Ok(gpu) => Verdict::Verified {
            identity: WorkloadIdentity {
                environment: x0ne_confidential::model::ExecutionEnvironment::ConfidentialVmWithGpu {
                    cpu: x0ne_confidential::model::TeeTechnology::IntelTdx,
                    gpu_mode: x0ne_confidential::model::GpuCcMode::On,
                },
                ..identity
            },
            // The verifier string is the verdict's honesty: with the live
            // check done, the lifted limitations are REPLACED by the measured
            // statuses and by what still remains (`tcb::LIVE_LIMITATIONS`).
            verifier: match live_tcb {
                Some((platform, qe)) => format!(
                    "x0ne-infer::pcs + x0ne-infer::tcb (live Intel PCS as of {now_unix}: platform {platform}, QE {qe}) \
                     + x0ne-infer::nras ({} GPU(s) bound by ueid; {}; limitations: {}; {}; {})",
                    gpu.gpu_ueids.len(),
                    revocation_note(ev.nras_ocsp, &gpu),
                    pinning_note(policy),
                    crate::tcb::LIVE_LIMITATIONS.join("; "),
                    gpu.limitations.join("; ")
                ),
                None => format!(
                    "x0ne-infer::pcs + x0ne-infer::nras ({} GPU(s) bound by ueid; {}; limitations: {}; {}; {})",
                    gpu.gpu_ueids.len(),
                    revocation_note(ev.nras_ocsp, &gpu),
                    pinning_note(policy),
                    crate::pcs::LIMITATIONS.join("; "),
                    gpu.limitations.join("; ")
                ),
            },
            strength: VerifierStrength::Cryptographic,
        },
        Err(rejected) => Verdict::Rejected {
            reason: format!("GPU attestation: {rejected}"),
            verifier: "x0ne-infer::nras".into(),
        },
    }
}

/// What a verdict says of the NRAS signing chain's revocation: the policy,
/// then each certificate's standing from the stapled OCSP responses.
pub fn revocation_note(
    policy: &crate::nras_ocsp::NrasOcspPolicy,
    gpu: &crate::nras::VerifiedGpu,
) -> String {
    format!(
        "NRAS signing chain revocation, policy {}: {}",
        policy.mode.as_str(),
        gpu.revocation
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::tests::NOW;

    /// The policy a caller of the TDX-ONLY entry point can hold: a verdict
    /// about the CPU cannot satisfy `require_gpu_cc_on`, and it now
    /// says so instead of returning `Verified` (see the test below).
    fn tdx_only_policy() -> VerificationPolicy {
        VerificationPolicy {
            require_gpu_cc_on: false,
            ..policy()
        }
    }

    fn policy() -> VerificationPolicy {
        VerificationPolicy {
            workload_name: "x0ne-infer".into(),
            // Was `vec![]` — accept no CPU technology — and the cryptographic
            // path verified anyway. That was the whole bug in one line.
            accepted_cpu: vec![x0ne_confidential::model::TeeTechnology::IntelTdx],
            require_gpu_cc_on: true,
            expected_measurements: vec![],
        }
    }

    /// SECURITY: the eighteenth dead scaffold, closed. A policy that
    /// demands GPU confidential mode is refused by the TDX-only entry point
    /// — the very evidence that opens it cannot speak about a GPU. A failure
    /// here means `require_gpu_cc_on` is once again written by `main.rs`,
    /// documented, and read by nobody on the cryptographic path: a client
    /// asking for the GPU would be told yes by a CPU proof.
    #[test]
    fn a_policy_demanding_the_gpu_is_refused_by_the_cpu_only_entry_point() {
        let fx = crate::pcs::tests::fixture_for_composite();
        let ev = AttestationEvidence {
            format: EvidenceFormat::TdxQuote,
            payload: fx.quote.clone(),
            collected_at_unix: 1_000,
        };
        let anchor = TrustAnchor::RootDer(fx.root_der);
        // Same evidence, same anchor, same challenge: only the policy differs.
        assert!(
            cryptographic_tdx(&ev, &tdx_only_policy(), &anchor, Some(&[0x5a; 64]), NOW)
                .is_verified(),
            "the chain itself must verify, or this test proves nothing"
        );
        let v = cryptographic_tdx(&ev, &policy(), &anchor, Some(&[0x5a; 64]), NOW);
        match v {
            Verdict::Rejected { reason, .. } => assert!(
                reason.contains("GPU confidential mode"),
                "the refusal must name the policy it could not meet: {reason}"
            ),
            other => panic!("a CPU-only verdict must not satisfy require_gpu_cc_on: {other:?}"),
        }
    }

    /// SECURITY: the server-side path refuses a debug TD.
    #[test]
    fn a_debug_td_is_refused_by_the_server_side_verifier() {
        let fx = crate::pcs::tests::fixture_debug_td([0x5a; 64]);
        let ev = AttestationEvidence {
            format: EvidenceFormat::TdxQuote,
            payload: fx.quote.clone(),
            collected_at_unix: 1_000,
        };
        let anchor = TrustAnchor::RootDer(fx.root_der);
        match cryptographic_tdx(&ev, &tdx_only_policy(), &anchor, Some(&[0x5a; 64]), NOW) {
            Verdict::Rejected { reason, .. } => assert!(reason.contains("DEBUG"), "{reason}"),
            other => panic!("a debug TD must not verify: {other:?}"),
        }
    }

    /// SECURITY: a policy that does not accept Intel TDX is refused
    /// by the path that can only ever prove Intel TDX. A failure here means
    /// `accepted_cpu` is decorative again — an operator writing
    /// `accepted_cpu = ["AmdSevSnp"]`, meaning "I do not trust Intel", would
    /// be served by an Intel machine and told their policy was met.
    ///
    /// The assertion above the interesting one is not decoration either: it
    /// proves the chain itself verifies, so the refusal below is the policy
    /// and not a broken fixture.
    #[test]
    fn a_policy_that_does_not_accept_tdx_is_refused_by_the_tdx_path() {
        let fx = crate::pcs::tests::fixture_for_composite();
        let ev = AttestationEvidence {
            format: EvidenceFormat::TdxQuote,
            payload: fx.quote.clone(),
            collected_at_unix: 1_000,
        };
        let anchor = TrustAnchor::RootDer(fx.root_der);
        assert!(
            cryptographic_tdx(&ev, &tdx_only_policy(), &anchor, Some(&[0x5a; 64]), NOW)
                .is_verified(),
            "the chain must verify under a policy that accepts TDX"
        );
        let sev_only = VerificationPolicy {
            accepted_cpu: vec![x0ne_confidential::model::TeeTechnology::AmdSevSnp],
            ..tdx_only_policy()
        };
        match cryptographic_tdx(&ev, &sev_only, &anchor, Some(&[0x5a; 64]), NOW) {
            Verdict::Rejected { reason, .. } => assert!(
                reason.contains("Intel TDX"),
                "the refusal must name what the evidence proves: {reason}"
            ),
            other => panic!("a policy refusing Intel must not be met by Intel: {other:?}"),
        }
        // And the empty list — the shape our own tests carried for months —
        // accepts nothing, so it refuses too.
        let nothing = VerificationPolicy {
            accepted_cpu: vec![],
            ..tdx_only_policy()
        };
        assert!(
            !cryptographic_tdx(&ev, &nothing, &anchor, Some(&[0x5a; 64]), NOW).is_verified(),
            "an empty accepted_cpu accepts nothing"
        );
    }

    /// SECURITY: the CPU-only product's verdict can never claim a
    /// GPU. This is the whole safety argument for offering it: an operator
    /// running `--attest cpu` sells a different guarantee, and a receipt that
    /// said `ConfidentialVmWithGpu { gpu_mode: On }` without a single NRAS
    /// byte having been fetched would be the very substitution the receipt exists to
    /// prevent, done by us rather than by an operator.
    ///
    /// It also checks the verifier string swapped its limitations: with the
    /// live half run, "TCB status and revocation not checked against live
    /// Intel PCS" is no longer true and must not be shipped in a receipt.
    #[test]
    fn the_cpu_only_verdict_names_the_live_statuses_and_never_claims_a_gpu() {
        use crate::tcb::TcbStatus;
        let fx = crate::pcs::tests::fixture_for_composite();
        let ev = AttestationEvidence {
            format: EvidenceFormat::TdxQuote,
            payload: fx.quote.clone(),
            collected_at_unix: 1_000,
        };
        let v = cryptographic_tdx_after_live_tcb_with_anchor(
            &ev,
            &tdx_only_policy(),
            NOW,
            &TrustAnchor::RootDer(fx.root_der),
            &[0x5a; 64],
            &TcbStatus::UpToDate,
            &TcbStatus::UpToDate,
        );
        let (identity, verifier) = match v {
            Verdict::Verified {
                identity, verifier, ..
            } => (identity, verifier),
            other => panic!("the CPU-only path must verify on a good chain: {other:?}"),
        };
        assert!(
            matches!(
                identity.environment,
                x0ne_confidential::model::ExecutionEnvironment::ConfidentialVm { .. }
            ),
            "a CPU-only proof must not describe a GPU: {:?}",
            identity.environment
        );
        assert!(
            verifier.contains("live Intel PCS") && verifier.contains("UpToDate"),
            "the measured statuses belong in the verifier string: {verifier}"
        );
        assert!(
            !verifier.contains("not checked against live Intel PCS"),
            "the lifted limitation must not survive the check that lifted it: {verifier}"
        );
    }

    /// The composite refuses when either half is bad — here a garbage quote
    /// with no NRAS token at all: rejected on the TDX arm, the GPU arm never
    /// reached. A stub that returned Verified would open the gate on nothing.
    #[test]
    fn the_composite_rejects_on_a_bad_tdx_half_before_reading_the_gpu_half() {
        let ev = AttestationEvidence {
            format: EvidenceFormat::TdxQuote,
            payload: vec![0u8; 64],
            collected_at_unix: 0,
        };
        let composite = CompositeEvidence {
            tdx: &ev,
            nras_token: "not-a-jwt",
            nras_jwks: &serde_json::json!({"keys": []}),
            nonce_hex: "00",
            expected_tdx_report_data: &[0u8; 64],
            nras_per_gpu: &[],
            local_gpu_serials: &[],
            nras_ocsp_responses: &[],
            nras_ocsp: &crate::nras_ocsp::NrasOcspPolicy::production(),
        };
        let v = cryptographic(&composite, &policy(), 1_000);
        assert!(matches!(v, Verdict::Rejected { .. }));
    }

    /// The whole composite, end to end with both fixtures: a pinned TDX
    /// chain AND a pinned NRAS token, same nonce → Verified{Cryptographic},
    /// and THAT verdict opens the gate. This is the serving condition of the
    /// spec, proven in one test — with test pins; production differs only by
    /// the two constants.
    #[test]
    fn both_halves_verified_yield_the_verdict_that_serves() {
        let tdx_fx = crate::pcs::tests::fixture_for_composite();
        let nras_fx = crate::nras::tests::fixture_for_composite();
        let tdx_ev = AttestationEvidence {
            format: EvidenceFormat::TdxQuote,
            payload: tdx_fx.quote.clone(),
            collected_at_unix: 1_000,
        };
        // Composite with per-test anchors: same code as production
        // `cryptographic`, anchors injected.
        let tdx = cryptographic_tdx(
            &tdx_ev,
            &tdx_only_policy(),
            &TrustAnchor::RootDer(tdx_fx.root_der),
            Some(&[0x5a; 64]),
            NOW,
        );
        let identity = match tdx {
            Verdict::Verified { identity, .. } => identity,
            other => panic!("TDX half must verify: {other:?}"),
        };
        let gpu = crate::nras::verify_nras_token(
            &nras_fx.token,
            &nras_fx.jwks,
            &nras_fx.nonce_hex,
            1_500,
            &nras_fx.pins,
        )
        .expect("GPU half must verify");
        assert!(gpu.overall);
        let verdict = Verdict::Verified {
            identity,
            verifier: "test-composite".into(),
            strength: VerifierStrength::Cryptographic,
        };
        let proof = crate::gate::Gate::attest(
            x0ne_confidential::model::ExecutionEnvironment::ConfidentialVmWithGpu {
                cpu: x0ne_confidential::model::TeeTechnology::IntelTdx,
                gpu_mode: x0ne_confidential::model::GpuCcMode::On,
            },
            &tdx_ev,
            verdict,
            1_000,
        )
        .expect("both halves open the gate");
        assert!(proof.environment().claims_confidentiality());
    }

    /// SECURITY: a failure means a pinned image is not enforced on
    /// the path that opens the gate — genuine TDX hardware running ANY guest
    /// image verifies, which is what "no measurement pinned" meant.
    #[test]
    fn a_pinned_measurement_is_enforced_on_the_cryptographic_path() {
        let fx = crate::pcs::tests::fixture_for_composite();
        let ev = AttestationEvidence {
            format: EvidenceFormat::TdxQuote,
            payload: fx.quote.clone(),
            collected_at_unix: 1_000,
        };
        let anchor = TrustAnchor::RootDer(fx.root_der.clone());
        let pinned = |spec: &str| VerificationPolicy {
            expected_measurements: vec![parse_expected_measurement(spec).expect("well-formed")],
            ..tdx_only_policy()
        };
        let verdict =
            |p: &VerificationPolicy| cryptographic_tdx(&ev, p, &anchor, Some(&[0x5a; 64]), NOW);

        // The fixture's rtmr1 is 0xb2 × 48 and its mrtd 0xa1 × 48.
        let ours = format!("rtmr1={}", "b2".repeat(48));
        let Verdict::Verified {
            identity, verifier, ..
        } = verdict(&pinned(&ours))
        else {
            panic!("the image the policy names must verify");
        };
        assert!(verifier.contains("pinned and matched: rtmr1"), "{verifier}");
        // Every register is in the verdict — that is what a CLIENT pins.
        let reg = |name: &str| {
            identity
                .measurements
                .iter()
                .find(|m| m.register == name)
                .map(|m| m.value_hex.clone())
        };
        assert_eq!(reg("mrtd"), Some("a1".repeat(48)));
        assert_eq!(reg("rtmr1"), Some("b2".repeat(48)));
        assert_eq!(reg("rtmr3"), Some("00".repeat(48)));

        // Another image on the same genuine hardware: refused.
        let theirs = format!("rtmr1={}", "c3".repeat(48));
        let Verdict::Rejected { reason, .. } = verdict(&pinned(&theirs)) else {
            panic!("a different image must be refused");
        };
        assert!(
            reason.contains("rtmr1") && reason.contains("mismatch"),
            "{reason}"
        );

        // The right VALUE under the wrong REGISTER is still a mismatch.
        let misplaced = format!("mrtd={}", "b2".repeat(48));
        assert!(matches!(
            verdict(&pinned(&misplaced)),
            Verdict::Rejected { .. }
        ));

        // And with nothing pinned the verdict says so, in so many words.
        let Verdict::Verified { verifier, .. } = verdict(&tdx_only_policy()) else {
            panic!("unpinned still verifies");
        };
        assert!(
            verifier.contains("no launch measurement pinned"),
            "{verifier}"
        );
    }

    #[test]
    fn an_expectation_that_could_never_match_is_refused_when_it_is_typed() {
        assert!(parse_expected_measurement(&format!("mrtd={}", "AB".repeat(48))).is_ok());
        for bad in [
            "mrtd".to_string(),
            format!("rtmr4={}", "ab".repeat(48)),
            format!("MRTD={}", "ab".repeat(48)),
            format!("mrtd={}", "ab".repeat(32)),
            format!("mrtd={}", "zz".repeat(48)),
        ] {
            assert!(parse_expected_measurement(&bad).is_err(), "{bad}");
        }
    }

    /// The TDX half refuses evidence that is not a raw quote — parsed claims
    /// cannot carry a signature chain.
    /// A genuine quote that is not OURS is refused — the anti-replay.
    ///
    /// SECURITY: this is the property the whole product rests on. The
    /// fixture's chain is real and verifies to its pinned root; only the
    /// `report_data` differs from the challenge we made. A failure here means
    /// a quote captured on any current TDX machine — including the one this
    /// repository vendors under `tests/pcs_live/` — opens the gate, and the
    /// verdict then attests a machine nobody in this process is running on.
    ///
    /// The check was missing until an audit on 2026-08-21, while
    /// `binding::verify_binding` and `x0ne_confidential::tdx_quote::verify`
    /// both implemented it, tested, and uncalled.
    #[test]
    fn a_genuine_quote_answering_someone_elses_challenge_is_refused() {
        let fx = crate::pcs::tests::fixture_for_composite();
        let ev = AttestationEvidence {
            format: EvidenceFormat::TdxQuote,
            payload: fx.quote.clone(),
            collected_at_unix: 1_000,
        };
        let anchor = TrustAnchor::RootDer(fx.root_der.clone());

        // The fixture's quote carries [0x5a; 64]. Ask for that: accepted.
        assert!(
            matches!(
                cryptographic_tdx(&ev, &tdx_only_policy(), &anchor, Some(&[0x5a; 64]), NOW),
                Verdict::Verified { .. }
            ),
            "our own challenge must verify"
        );

        // Ask for anything else: the SAME genuine quote is now a replay.
        let theirs = [0x11_u8; 64];
        let v = cryptographic_tdx(&ev, &tdx_only_policy(), &anchor, Some(&theirs), NOW);
        let Verdict::Rejected { reason, .. } = v else {
            panic!("a quote answering another challenge must be refused");
        };
        assert!(reason.contains("replay"), "{reason}");

        // And the composite refuses it too — the path the gate actually uses.
        let nras_fx = crate::nras::tests::fixture_for_composite();
        let composite = CompositeEvidence {
            tdx: &ev,
            nras_token: &nras_fx.token,
            nras_jwks: &nras_fx.jwks,
            nonce_hex: &nras_fx.nonce_hex,
            expected_tdx_report_data: &theirs,
            nras_per_gpu: &nras_fx.per_gpu,
            local_gpu_serials: &nras_fx.gpu_serials,
            nras_ocsp_responses: &[],
            nras_ocsp: &crate::nras_ocsp::NrasOcspPolicy::production(),
        };
        assert!(matches!(
            cryptographic_with_anchors(
                &composite,
                &policy(),
                1_000,
                &TrustAnchor::RootDer(fx.root_der),
                &nras_fx.pins,
                None,
            ),
            Verdict::Rejected { .. }
        ));
    }

    /// The proof publishes the environment the VERDICT established, never
    /// the collector's self-report. Mirrors main.rs exactly: configfs-tsm
    /// can only claim `ConfidentialVm` (it cannot see a GPU), the composite
    /// upgrades to `ConfidentialVmWithGpu` — and /attestation plus every
    /// signed receipt read the proof's environment. A failure here means
    /// the published environment is whatever the unverified caller said
    /// (once, the verified identity was discarded and the
    /// self-report served — dead scaffolding in the security payload).
    #[test]
    fn the_proof_publishes_the_verdicts_environment_not_the_collectors() {
        use x0ne_confidential::model::{ExecutionEnvironment, GpuCcMode, TeeTechnology};
        let tdx_fx = crate::pcs::tests::fixture_for_composite();
        let nras_fx = crate::nras::tests::fixture_for_composite();
        let tdx_ev = AttestationEvidence {
            format: EvidenceFormat::TdxQuote,
            payload: tdx_fx.quote.clone(),
            collected_at_unix: 1_000,
        };
        let composite = CompositeEvidence {
            tdx: &tdx_ev,
            nras_token: &nras_fx.token,
            nras_jwks: &nras_fx.jwks,
            expected_tdx_report_data: &[0x5a; 64],
            nonce_hex: &nras_fx.nonce_hex,
            nras_per_gpu: &nras_fx.per_gpu,
            local_gpu_serials: &nras_fx.gpu_serials,
            nras_ocsp_responses: &[],
            nras_ocsp: &crate::nras_ocsp::NrasOcspPolicy::production(),
        };
        // The REAL composition, fixtures only in the anchors.
        let verdict = cryptographic_with_anchors(
            &composite,
            &policy(),
            1_000,
            &TrustAnchor::RootDer(tdx_fx.root_der),
            &nras_fx.pins,
            Some(("UpToDate", "UpToDate")),
        );
        assert!(matches!(verdict, Verdict::Verified { .. }), "{verdict:?}");
        let proof = crate::gate::Gate::attest(
            // What main.rs actually passes: the collector's claim, no GPU.
            ExecutionEnvironment::ConfidentialVm {
                cpu: TeeTechnology::IntelTdx,
            },
            &tdx_ev,
            verdict,
            1_000,
        )
        .expect("a verified composite opens the gate");
        assert_eq!(
            proof.environment(),
            &ExecutionEnvironment::ConfidentialVmWithGpu {
                cpu: TeeTechnology::IntelTdx,
                gpu_mode: GpuCcMode::On,
            },
            "the proof must carry the verified environment, not the self-report"
        );
        // And the verdict SAYS the live check ran: the lifted limitation is
        // gone, the measured statuses are named.
        let Verdict::Verified { verifier, .. } = proof.verdict() else {
            unreachable!()
        };
        assert!(
            verifier.contains("live Intel PCS as of ") && verifier.contains(": platform UpToDate"),
            "{verifier}"
        );
        assert!(
            !verifier.contains("not checked against live Intel PCS"),
            "the lifted limitation must not survive in the verifier: {verifier}"
        );
    }

    #[test]
    fn the_tdx_half_requires_a_raw_quote() {
        let ev = AttestationEvidence {
            format: EvidenceFormat::StructuredClaims,
            payload: b"{}".to_vec(),
            collected_at_unix: 0,
        };
        let v = cryptographic_tdx(&ev, &policy(), &TrustAnchor::intel_production(), None, NOW);
        assert!(matches!(v, Verdict::Rejected { .. }));
    }

    /// `--expect-measurement` quoted a malformed spec back: a key typed there
    /// reached the terminal. The refusal says what is wrong, not the value.
    #[test]
    fn a_malformed_measurement_spec_is_not_printed() {
        let key = "5ec7".repeat(16);
        for spec in [key.clone(), format!("{key}=00"), format!("X{key}={key}")] {
            let e = parse_expected_measurement(&spec).unwrap_err();
            assert!(!e.contains("5ec75ec7"), "{spec}: {e}");
        }
    }

    // ── SEV-SNP ─────────────────────────────────────────────────────────

    fn snp_policy() -> VerificationPolicy {
        VerificationPolicy {
            workload_name: "x0ne-infer".into(),
            accepted_cpu: vec![x0ne_confidential::model::TeeTechnology::AmdSevSnp],
            require_gpu_cc_on: false,
            expected_measurements: vec![],
        }
    }

    fn snp_evidence(payload: Vec<u8>) -> AttestationEvidence {
        AttestationEvidence {
            format: EvidenceFormat::SevSnpReport,
            payload,
            collected_at_unix: NOW,
        }
    }

    fn snp_verdict(policy: &VerificationPolicy, payload: Vec<u8>, challenge: &[u8; 64]) -> Verdict {
        cryptographic_snp(
            &snp_evidence(payload),
            policy,
            &crate::snp::SnpPolicy::default(),
            &crate::snp::tests::test_anchors(),
            crate::snp::tests::TEST_GENOA_CRL,
            Some(challenge),
            NOW,
        )
    }

    #[test]
    fn an_snp_report_yields_a_cryptographic_verdict_naming_amd_sev_snp() {
        let v = snp_verdict(
            &snp_policy(),
            crate::snp::tests::genoa_evidence(|_| {}),
            &[0x64; 64],
        );
        let Verdict::Verified {
            identity,
            verifier,
            strength,
        } = v
        else {
            panic!("{v:?}")
        };
        assert_eq!(strength, VerifierStrength::Cryptographic);
        assert_eq!(
            identity.environment,
            x0ne_confidential::model::ExecutionEnvironment::ConfidentialVm {
                cpu: x0ne_confidential::model::TeeTechnology::AmdSevSnp
            }
        );
        let names: Vec<_> = identity
            .measurements
            .iter()
            .map(|m| m.register.as_str())
            .collect();
        assert_eq!(names[..2], ["vcek_sha256", "snp_measurement"]);
        for said in [
            "Genoa",
            "AMD's CRL for the product",
            "no launch measurement pinned",
        ] {
            assert!(verifier.contains(said), "{said}: {verifier}");
        }
    }

    /// SECURITY: a genuine report bound to someone else's session.
    #[test]
    fn an_snp_report_answering_another_session_is_refused() {
        let v = snp_verdict(
            &snp_policy(),
            crate::snp::tests::genoa_evidence(|_| {}),
            &[0x65; 64],
        );
        let Verdict::Rejected { reason, .. } = v else {
            panic!("{v:?}")
        };
        assert!(reason.contains("REPORT_DATA"), "{reason}");
    }

    #[test]
    fn a_pinned_snp_measurement_is_enforced() {
        let pin = |hex: &str| VerificationPolicy {
            expected_measurements: vec![crate::snp::parse_expected_snp_measurement(&format!(
                "snp_measurement={}",
                hex.repeat(48)
            ))
            .unwrap()],
            ..snp_policy()
        };
        let evidence = || crate::snp::tests::genoa_evidence(|_| {});
        let v = snp_verdict(&pin("4e"), evidence(), &[0x64; 64]);
        assert!(
            matches!(&v, Verdict::Rejected { reason, .. } if reason.contains("not the image")),
            "{v:?}"
        );
        let v = snp_verdict(&pin("4d"), evidence(), &[0x64; 64]);
        assert!(
            matches!(&v, Verdict::Verified { verifier, .. } if verifier.contains("pinned and matched: snp_measurement")),
            "{v:?}"
        );
    }

    /// The gates of the TDX path, restated on the SNP path: the policy must
    /// accept SEV-SNP, a CPU-only verdict cannot meet a GPU demand, the
    /// evidence must be an SNP report — and the TDX path refuses one.
    #[test]
    fn the_snp_path_has_the_tdx_paths_gates_and_neither_accepts_the_other() {
        let payload = || crate::snp::tests::genoa_evidence(|_| {});
        // Refused, and for the reason named: a refuse-everything verifier
        // would pass a bare `is Rejected` check.
        let refused_for = |v: Verdict, why: &str| {
            matches!(&v, Verdict::Rejected { reason, .. } if reason.contains(why))
                || panic!("expected a refusal naming {why:?}, got {v:?}")
        };
        refused_for(
            snp_verdict(&tdx_only_policy(), payload(), &[0x64; 64]),
            "proves AMD SEV-SNP",
        );
        let gpu = VerificationPolicy {
            require_gpu_cc_on: true,
            ..snp_policy()
        };
        refused_for(snp_verdict(&gpu, payload(), &[0x64; 64]), "GPU");
        let quote = AttestationEvidence {
            format: EvidenceFormat::TdxQuote,
            payload: payload(),
            collected_at_unix: NOW,
        };
        refused_for(
            cryptographic_snp(
                &quote,
                &snp_policy(),
                &crate::snp::SnpPolicy::default(),
                &crate::snp::tests::test_anchors(),
                crate::snp::tests::TEST_GENOA_CRL,
                Some(&[0x64; 64]),
                NOW,
            ),
            "SEV-SNP report",
        );
        let tdx_policy = VerificationPolicy {
            accepted_cpu: vec![
                x0ne_confidential::model::TeeTechnology::IntelTdx,
                x0ne_confidential::model::TeeTechnology::AmdSevSnp,
            ],
            ..snp_policy()
        };
        refused_for(
            cryptographic_tdx(
                &snp_evidence(payload()),
                &tdx_policy,
                &TrustAnchor::intel_production(),
                Some(&[0x64; 64]),
                NOW,
            ),
            "raw TDX quote",
        );
    }

    // ── SEV-SNP with the GPU ────────────────────────────────────────────

    /// An NRAS fixture valid at the SNP fixtures' time, for `nonce_hex`.
    fn nras_at_snp_time(nonce_hex: &str) -> crate::nras::tests::Fixture {
        crate::nras::tests::build_fixture_under(
            crate::nras::tests::Inter::Rsa,
            true,
            nonce_hex,
            crate::snp::tests::NOW + 3_600,
            (1970, 4096),
        )
    }

    fn snp_gpu_verdict(
        policy: &VerificationPolicy,
        payload: Vec<u8>,
        challenge: &[u8; 64],
        nras: &crate::nras::tests::Fixture,
        nonce_hex: &str,
    ) -> Verdict {
        let snp = AttestationEvidence {
            format: EvidenceFormat::SevSnpReport,
            payload,
            collected_at_unix: crate::snp::tests::NOW,
        };
        cryptographic_snp_with_gpu(
            &SnpCompositeEvidence {
                snp: &snp,
                expected_report_data: challenge,
                nras_token: &nras.token,
                nras_jwks: &nras.jwks,
                nonce_hex,
                nras_per_gpu: &nras.per_gpu,
                local_gpu_serials: &nras.gpu_serials,
                nras_ocsp_responses: &[],
                nras_ocsp: &crate::nras_ocsp::NrasOcspPolicy::production(),
            },
            policy,
            &crate::snp::SnpPolicy::default(),
            &crate::snp::tests::test_anchors(),
            crate::snp::tests::TEST_GENOA_CRL,
            &nras.pins,
            crate::snp::tests::NOW,
        )
    }

    fn snp_gpu_policy() -> VerificationPolicy {
        VerificationPolicy {
            require_gpu_cc_on: true,
            ..snp_policy()
        }
    }

    /// Report and token both genuine and both this start-up's: the verdict
    /// names SEV-SNP with the GPU on, and drops the report's "CPU only".
    #[test]
    fn an_snp_report_with_an_nras_token_verifies_as_cpu_and_gpu() {
        let nonce = "ab".repeat(32);
        let nras = nras_at_snp_time(&nonce);
        let v = snp_gpu_verdict(
            &snp_gpu_policy(),
            crate::snp::tests::genoa_evidence(|_| {}),
            &[0x64; 64],
            &nras,
            &nonce,
        );
        let Verdict::Verified {
            identity,
            verifier,
            strength,
        } = v
        else {
            panic!("{v:?}")
        };
        assert_eq!(strength, VerifierStrength::Cryptographic);
        assert_eq!(
            identity.environment,
            x0ne_confidential::model::ExecutionEnvironment::ConfidentialVmWithGpu {
                cpu: x0ne_confidential::model::TeeTechnology::AmdSevSnp,
                gpu_mode: x0ne_confidential::model::GpuCcMode::On,
            }
        );
        assert!(identity
            .measurements
            .iter()
            .any(|m| m.register == "snp_measurement"));
        assert!(
            verifier.contains("x0ne-infer::snp") && verifier.contains("x0ne-infer::nras"),
            "{verifier}"
        );
        assert!(!verifier.contains("CPU only"), "{verifier}");
    }

    /// Each half refuses on its own: another challenge in REPORT_DATA, a
    /// debug guest, a token for another nonce, a policy without SEV-SNP.
    #[test]
    fn the_snp_composite_refuses_when_either_half_does() {
        let nonce = "ab".repeat(32);
        let nras = nras_at_snp_time(&nonce);
        let ok = || crate::snp::tests::genoa_evidence(|_| {});
        let reason = |v: Verdict| match v {
            Verdict::Rejected { reason, .. } => reason,
            other => panic!("must refuse: {other:?}"),
        };
        let r = reason(snp_gpu_verdict(
            &snp_gpu_policy(),
            ok(),
            &[0x11; 64],
            &nras,
            &nonce,
        ));
        assert!(r.contains("replay"), "{r}");
        let debug = crate::snp::tests::genoa_evidence(|s| {
            s.policy |= x0ne_confidential::snp_report::POLICY_DEBUG
        });
        let r = reason(snp_gpu_verdict(
            &snp_gpu_policy(),
            debug,
            &[0x64; 64],
            &nras,
            &nonce,
        ));
        assert!(r.contains("DEBUG"), "{r}");
        let other = "cd".repeat(32);
        let r = reason(snp_gpu_verdict(
            &snp_gpu_policy(),
            ok(),
            &[0x64; 64],
            &nras,
            &other,
        ));
        assert!(r.contains("GPU attestation"), "{r}");
        let tdx_only = VerificationPolicy {
            accepted_cpu: vec![x0ne_confidential::model::TeeTechnology::IntelTdx],
            ..snp_gpu_policy()
        };
        let r = reason(snp_gpu_verdict(&tdx_only, ok(), &[0x64; 64], &nras, &nonce));
        assert!(r.contains("AMD SEV-SNP"), "{r}");
    }
}
