//! Verification policy and the structural verifier.
//!
//! A [`VerificationPolicy`] states what an environment must prove before a
//! workload identity is established: which TEE technologies are acceptable,
//! whether the GPU must be in confidential mode, and which exact measurements
//! are expected. The [`StructuralVerifier`] checks *structured claims*
//! against that policy — it validates shape and expectation, not
//! cryptography, and every verdict it produces says so by name, so nothing
//! downstream can mistake it for a hardware root of trust.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::model::{
    AttestationEvidence, EvidenceFormat, EvidenceVerifier, ExecutionEnvironment, GpuCcMode,
    Measurement, TeeTechnology, Verdict, VerifierStrength, WorkloadIdentity,
};

/// The name every structural verdict carries. Downstream policy uses it to
/// refuse structural verdicts where cryptographic ones are demanded.
pub const STRUCTURAL_VERIFIER_NAME: &str = "structural";

/// One expected measurement: the register that must be present and the exact
/// value it must carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpectedMeasurement {
    pub register: String,
    pub algorithm: String,
    pub value_hex: String,
}

/// What evidence must establish for a workload to be identified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationPolicy {
    /// The workload name the identity will carry once established.
    pub workload_name: String,
    /// TEE technologies this policy accepts. Empty means the policy accepts
    /// no technology — which fails everything, and deliberately so: an empty
    /// allowlist is a closed door, not an open one.
    pub accepted_cpu: Vec<TeeTechnology>,
    /// Whether the GPU must be in full confidential mode. When true,
    /// [`GpuCcMode::DevTools`] and [`GpuCcMode::Off`] both fail — DevTools is
    /// a diagnostics posture, not a production one.
    pub require_gpu_cc_on: bool,
    /// Every listed measurement must be present in the claims with exactly
    /// this value. Claims may carry more measurements than the policy names;
    /// they may not miss or mismatch any the policy names.
    pub expected_measurements: Vec<ExpectedMeasurement>,
}

/// The claims document the structural verifier reads: JSON of the shape
///
/// ```json
/// {
///   "environment": { "cpu": "amd-sev-snp", "gpu_mode": "on" },
///   "measurements": [
///     { "register": "launch", "algorithm": "sha384", "value_hex": "ab…" }
///   ]
/// }
/// ```
///
/// Produced today by tests and future provisioning tooling; produced later by
/// adapters that translate real attestation results into claims *after*
/// cryptographic verification elsewhere.
pub struct StructuralVerifier;

impl StructuralVerifier {
    fn reject(reason: impl Into<String>) -> Verdict {
        Verdict::Rejected {
            reason: reason.into(),
            verifier: STRUCTURAL_VERIFIER_NAME.into(),
        }
    }

    fn parse_cpu(s: &str) -> TeeTechnology {
        match s {
            "amd-sev-snp" => TeeTechnology::AmdSevSnp,
            "intel-tdx" => TeeTechnology::IntelTdx,
            other => TeeTechnology::Other(other.to_string()),
        }
    }

    fn parse_gpu(s: &str) -> Option<GpuCcMode> {
        match s {
            "on" => Some(GpuCcMode::On),
            "devtools" => Some(GpuCcMode::DevTools),
            "off" => Some(GpuCcMode::Off),
            _ => None,
        }
    }
}

impl EvidenceVerifier for StructuralVerifier {
    fn verify(&self, evidence: &AttestationEvidence, policy: &VerificationPolicy) -> Verdict {
        // Format first: this verifier reads structured claims and nothing
        // else. A hardware blob is not something it can check, and saying
        // "no" is the only honest answer.
        if evidence.format != EvidenceFormat::StructuredClaims {
            return Self::reject(format!(
                "structural verifier cannot read {:?} evidence",
                evidence.format
            ));
        }
        let claims: Value = match serde_json::from_slice(&evidence.payload) {
            Ok(v) => v,
            Err(e) => return Self::reject(format!("unreadable claims: {e}")),
        };

        // Environment.
        let cpu = match claims.pointer("/environment/cpu").and_then(|v| v.as_str()) {
            Some(s) => Self::parse_cpu(s),
            None => return Self::reject("claims carry no environment.cpu"),
        };
        if !policy.accepted_cpu.contains(&cpu) {
            return Self::reject(format!("cpu technology {cpu:?} is not accepted by policy"));
        }
        let gpu_mode = claims
            .pointer("/environment/gpu_mode")
            .and_then(|v| v.as_str())
            .map(|s| Self::parse_gpu(s).ok_or_else(|| format!("unknown gpu_mode claim '{s}'")))
            .transpose();
        let gpu_mode = match gpu_mode {
            Ok(m) => m,
            Err(e) => return Self::reject(e),
        };
        if policy.require_gpu_cc_on && gpu_mode != Some(GpuCcMode::On) {
            return Self::reject(format!(
                "policy requires GPU confidential mode ON; claims say {gpu_mode:?}"
            ));
        }

        // Measurements: exact-match every expectation.
        let claimed: Vec<Measurement> = claims
            .get("measurements")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        Some(Measurement {
                            register: m.get("register")?.as_str()?.to_string(),
                            algorithm: m.get("algorithm")?.as_str()?.to_string(),
                            value_hex: m.get("value_hex")?.as_str()?.to_lowercase(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        for expected in &policy.expected_measurements {
            let hit = claimed
                .iter()
                .find(|m| m.register == expected.register && m.algorithm == expected.algorithm);
            match hit {
                None => {
                    return Self::reject(format!(
                        "expected measurement '{}' ({}) is absent from the claims",
                        expected.register, expected.algorithm
                    ))
                }
                Some(m) if m.value_hex != expected.value_hex.to_lowercase() => {
                    return Self::reject(format!(
                        "measurement '{}' mismatch: expected {}, claimed {}",
                        expected.register, expected.value_hex, m.value_hex
                    ))
                }
                Some(_) => {}
            }
        }

        let environment = match gpu_mode {
            Some(mode) => ExecutionEnvironment::ConfidentialVmWithGpu {
                cpu,
                gpu_mode: mode,
            },
            None => ExecutionEnvironment::ConfidentialVm { cpu },
        };
        Verdict::Verified {
            identity: WorkloadIdentity {
                name: policy.workload_name.clone(),
                environment,
                measurements: claimed,
            },
            verifier: STRUCTURAL_VERIFIER_NAME.into(),
            // Declared truthfully: this checker read claims, not proofs. The
            // release path refuses anything below Cryptographic, so this
            // verdict can gate nothing that stakes weights or data.
            strength: VerifierStrength::Structural,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(json: &str) -> AttestationEvidence {
        AttestationEvidence {
            format: EvidenceFormat::StructuredClaims,
            payload: json.as_bytes().to_vec(),
            collected_at_unix: 1_000,
        }
    }

    fn policy() -> VerificationPolicy {
        VerificationPolicy {
            workload_name: "confidential-inference".into(),
            accepted_cpu: vec![TeeTechnology::AmdSevSnp],
            require_gpu_cc_on: true,
            expected_measurements: vec![ExpectedMeasurement {
                register: "launch".into(),
                algorithm: "sha384".into(),
                value_hex: "abcd".into(),
            }],
        }
    }

    const GOOD: &str = r#"{
        "environment": { "cpu": "amd-sev-snp", "gpu_mode": "on" },
        "measurements": [
            { "register": "launch", "algorithm": "sha384", "value_hex": "ABCD" }
        ]
    }"#;

    #[test]
    fn matching_claims_establish_the_identity() {
        let verdict = StructuralVerifier.verify(&evidence(GOOD), &policy());
        match verdict {
            Verdict::Verified {
                identity,
                verifier,
                strength,
            } => {
                assert_eq!(verifier, STRUCTURAL_VERIFIER_NAME);
                assert_eq!(strength, VerifierStrength::Structural);
                assert_eq!(identity.name, "confidential-inference");
                assert!(matches!(
                    identity.environment,
                    ExecutionEnvironment::ConfidentialVmWithGpu {
                        cpu: TeeTechnology::AmdSevSnp,
                        gpu_mode: GpuCcMode::On
                    }
                ));
            }
            Verdict::Rejected { reason, .. } => panic!("rejected: {reason}"),
        }
    }

    #[test]
    fn a_measurement_mismatch_rejects() {
        let bad = GOOD.replace("ABCD", "beef");
        let verdict = StructuralVerifier.verify(&evidence(&bad), &policy());
        assert!(!verdict.is_verified());
    }

    #[test]
    fn a_missing_measurement_rejects() {
        let bad = r#"{ "environment": { "cpu": "amd-sev-snp", "gpu_mode": "on" },
                       "measurements": [] }"#;
        assert!(!StructuralVerifier
            .verify(&evidence(bad), &policy())
            .is_verified());
    }

    /// DevTools is a diagnostics posture, not a production one: where policy
    /// demands GPU confidentiality, it fails exactly like Off.
    #[test]
    fn devtools_mode_does_not_satisfy_gpu_cc_on() {
        let bad = GOOD.replace("\"on\"", "\"devtools\"");
        let verdict = StructuralVerifier.verify(&evidence(&bad), &policy());
        match verdict {
            Verdict::Rejected { reason, .. } => assert!(reason.contains("DevTools")),
            Verdict::Verified { .. } => panic!("devtools satisfied gpu-cc-on"),
        }
    }

    #[test]
    fn an_unaccepted_cpu_technology_rejects() {
        let bad = GOOD.replace("amd-sev-snp", "intel-tdx");
        assert!(!StructuralVerifier
            .verify(&evidence(&bad), &policy())
            .is_verified());
    }

    /// An empty allowlist is a closed door: nothing passes it.
    #[test]
    fn an_empty_cpu_allowlist_rejects_everything() {
        let mut p = policy();
        p.accepted_cpu.clear();
        assert!(!StructuralVerifier.verify(&evidence(GOOD), &p).is_verified());
    }

    /// Ambiguity fails closed: unreadable claims, unknown modes, wrong
    /// evidence formats.
    #[test]
    fn ambiguity_rejects() {
        let p = policy();
        assert!(!StructuralVerifier
            .verify(&evidence("not json"), &p)
            .is_verified());
        let unknown_mode = GOOD.replace("\"on\"", "\"secret-mode\"");
        assert!(!StructuralVerifier
            .verify(&evidence(&unknown_mode), &p)
            .is_verified());
        let hardware_blob = AttestationEvidence {
            format: EvidenceFormat::SevSnpReport,
            payload: vec![0u8; 32],
            collected_at_unix: 1_000,
        };
        assert!(!StructuralVerifier.verify(&hardware_blob, &p).is_verified());
    }
}
