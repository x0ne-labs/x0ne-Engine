//! The confidential-runtime model: x0ne Engine's vocabulary for protected
//! execution — environments, measurements, workload identity, attestation
//! evidence, verdicts, key release and receipts. The trust model and its
//! limits are in the repository's `x0ne-infer/THREAT_MODEL.md` and
//! `docs/limitations.md`.
//!
//! The types are provider-neutral by construction: there is no `if GCP`, no
//! `if H100`; a provider adapter maps its concrete world onto them.
//!
//! ## What this module deliberately is not
//!
//! It performs **no cryptographic verification**. Attestation evidence is
//! verified against vendor roots and attestation services (Intel PCS,
//! NVIDIA's GPU attestation, AMD KDS) by `x0ne-infer` (`verify.rs`, `pcs`,
//! `tcb`, `nras`, `snp`), whose verdicts say [`VerifierStrength::Cryptographic`].
//! The [`policy::StructuralVerifier`] included here checks evidence *claims*
//! against a policy's expectations; `x0ne-infer` keeps it for diagnostics,
//! and it never opens the gate. Nothing here manufactures a hardware
//! guarantee, and nothing downstream may treat a structural verdict as one —
//! the verdict says which verifier produced it.
//!
//! ## How it composes
//!
//! - a *capability over time* is a [`crate::lease::CapabilityLease`];
//!   an attestation-bound capability is a lease whose issuance required a
//!   [`Verdict::Verified`] — see [`KeyReleaseDecision`];
//! - execution facts become [`ExecutionReceipt`]s, assembled from semantic
//!   facts, never scraped from presentation;
//! - ambiguity fails closed.

use serde::{Deserialize, Serialize};

use crate::policy;

// ─────────────────────────────────────────────────────────────────────────────
// Execution environments
// ─────────────────────────────────────────────────────────────────────────────

/// CPU trusted-execution technology, named for reasoning and policy — not a
/// guarantee of anything by itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TeeTechnology {
    /// AMD SEV-SNP confidential VMs.
    AmdSevSnp,
    /// Intel TDX trust domains.
    IntelTdx,
    /// A technology this build does not model yet. Policy treats unknown
    /// technologies as unverifiable — fail closed.
    Other(String),
}

/// The GPU's confidential-computing posture, as reported by the environment.
///
/// Descriptive identifiers for the modes NVIDIA's confidential-computing
/// stack exposes; whether a given mode is *actually* in force is exactly what
/// attestation evidence exists to prove, so nothing may branch on this field
/// as if it were proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuCcMode {
    /// Full confidential-computing mode.
    On,
    /// Vendor development/diagnostics mode — NOT a production posture;
    /// policy must reject it wherever protected data or weights are staked.
    DevTools,
    /// Confidential computing off.
    Off,
}

/// Where a workload executes, in the terms policy cares about.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionEnvironment {
    /// Ordinary infrastructure: no hardware confidentiality claim at all.
    /// `x0ne-infer`'s gate never opens here (`Refused::NotConfidential`), and
    /// saying so explicitly is the point — absence of a claim is not a weak
    /// claim.
    Standard,
    /// A confidential VM: CPU/memory protected by a TEE.
    ConfidentialVm { cpu: TeeTechnology },
    /// A confidential VM with an attached confidential-mode GPU — the
    /// NVIDIA-target shape. The CPU TEE and the GPU posture are separate
    /// facts with separate evidence; policy checks both.
    ConfidentialVmWithGpu {
        cpu: TeeTechnology,
        gpu_mode: GpuCcMode,
    },
}

impl ExecutionEnvironment {
    /// Whether this environment even claims hardware confidentiality.
    pub fn claims_confidentiality(&self) -> bool {
        !matches!(self, ExecutionEnvironment::Standard)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Measurements and identity
// ─────────────────────────────────────────────────────────────────────────────

/// One measured value from the environment's measurement chain — a launch
/// digest, a firmware hash, an image digest, a GPU attestation claim.
///
/// `register` names *what* was measured in the platform's own vocabulary
/// (e.g. a launch-measurement register, an RTMR index, an image digest slot);
/// `value_hex` is the digest as lowercase hex. Comparison is exact-match on
/// both — there is no fuzzy matching in a measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Measurement {
    pub register: String,
    pub algorithm: String,
    pub value_hex: String,
}

/// What a workload *is*, for the purpose of trust decisions: a name for
/// humans, and the measurements that pin it for machines.
///
/// Identity here is established by verification, not asserted by the
/// workload: a `WorkloadIdentity` is what a [`Verdict::Verified`] hands back,
/// and constructing one by hand is constructing a claim nobody checked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadIdentity {
    pub name: String,
    pub environment: ExecutionEnvironment,
    pub measurements: Vec<Measurement>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Evidence
// ─────────────────────────────────────────────────────────────────────────────

/// The format of a piece of attestation evidence. Opaque payloads; the
/// format tag tells a verifier which external machinery can check it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceFormat {
    /// An AMD SEV-SNP attestation report.
    SevSnpReport,
    /// An Intel TDX quote.
    TdxQuote,
    /// GPU attestation evidence from the NVIDIA stack.
    NvidiaGpuEvidence,
    /// Structured claims already parsed into JSON — the form the structural
    /// verifier can inspect, and the form composite attestation results
    /// arrive in from external verification services.
    StructuredClaims,
    Other(String),
}

/// A piece of evidence the environment produced about itself.
///
/// The payload is opaque bytes at this layer. For [`EvidenceFormat::StructuredClaims`]
/// it is a JSON document of measurement claims; for hardware formats it is
/// whatever blob the platform emitted, destined for an external verifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationEvidence {
    pub format: EvidenceFormat,
    pub payload: Vec<u8>,
    /// Unix seconds when the evidence was collected. Freshness policy is the
    /// verifier's concern; the timestamp is carried so it can have one.
    pub collected_at_unix: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Verification
// ─────────────────────────────────────────────────────────────────────────────

/// How strong a verifier's word is.
///
/// Declared by the verifier implementation (trusted code in this process),
/// carried on the verdict, and consumed as a **typed allowlist**: release
/// policy demands `Cryptographic` and anything else refuses. The earlier
/// shape — comparing the verifier's *name* against the one known structural
/// checker — was a denylist, and a denylist fails open the day a second
/// non-cryptographic verifier appears.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerifierStrength {
    /// Checked claims for shape and expectation. No cryptography.
    Structural,
    /// Verified evidence against hardware vendor roots / an attestation
    /// service. The only strength a key release may stand on.
    Cryptographic,
}

/// The outcome of verifying evidence against a policy.
///
/// `Verified` carries the identity the evidence established, which verifier
/// said so, and how strong that word is — a structural verdict and a
/// cryptographic one must never be confusable downstream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    Verified {
        identity: WorkloadIdentity,
        verifier: String,
        strength: VerifierStrength,
    },
    Rejected {
        reason: String,
        verifier: String,
    },
}

impl Verdict {
    pub fn is_verified(&self) -> bool {
        matches!(self, Verdict::Verified { .. })
    }
}

/// The boundary where verification machinery plugs in.
///
/// Implementations range from the in-process structural checker
/// ([`policy::StructuralVerifier`]) to future adapters that call vendor
/// attestation services and key-broker verifiers. The contract every
/// implementation owes: **ambiguity is rejection.** Evidence that cannot be
/// read, claims that are missing, technologies the policy does not know —
/// all `Rejected`, never a permissive default.
pub trait EvidenceVerifier {
    fn verify(
        &self,
        evidence: &AttestationEvidence,
        policy: &policy::VerificationPolicy,
    ) -> Verdict;
}

// ─────────────────────────────────────────────────────────────────────────────
// Protected artifacts and key release
// ─────────────────────────────────────────────────────────────────────────────

/// What kind of thing a protected artifact is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactKind {
    /// Proprietary model weights — the model provider's asset.
    ModelWeights,
    /// Customer data — the customer's asset.
    Dataset,
    /// Configuration/prompts/adapters that ride with a deployment.
    Config,
}

/// An encrypted artifact that may live in untrusted storage.
///
/// The artifact carries a *reference* to its key, never material: plaintext
/// exists only inside an environment whose evidence satisfied the key's
/// release policy. This is the type that makes "the encrypted model can sit
/// in the customer's bucket" a statement about types rather than discipline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtectedArtifact {
    pub id: String,
    pub kind: ArtifactKind,
    /// Where the ciphertext lives (a URL, a path, a bucket key). Untrusted.
    pub storage_ref: String,
    /// The encryption scheme, named for audit (e.g. "aes-256-gcm").
    pub encryption: String,
    /// The key's identity at the key-broker. An opaque name — key material
    /// has no representation in this codebase, deliberately.
    pub key_id: String,
}

/// A request that a key be released to a workload, justified by evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyReleaseRequest {
    pub key_id: String,
    pub evidence: AttestationEvidence,
}

/// How old evidence may be and still mint a release: five minutes. Evidence
/// is a statement about a moment; a release request replaying last week's
/// report must re-attest, not reuse. Clock skew into the future is bounded
/// by the same margin.
pub const MAX_EVIDENCE_AGE_SECS: u64 = 300;

/// The decision on a key-release request.
///
/// Release is attestation-bound *by construction*: the only way to build
/// `Released` is [`KeyReleaseDecision::grant`], which demands a fresh
/// [`Verdict::Verified`] of [`VerifierStrength::Cryptographic`]. The decision
/// carries a [`crate::lease::CapabilityLease`] — release is an
/// authority over time, with a window, a budget and
/// revocation, not a permanent fact.
#[derive(Debug, Clone)]
pub enum KeyReleaseDecision {
    Released {
        key_id: String,
        /// The identity the key was released to.
        to: WorkloadIdentity,
        /// The bounded authority under which the release stands. Its scope
        /// binds the key to the *measured* identity, not merely its name.
        lease: crate::lease::CapabilityLease,
    },
    Refused {
        key_id: String,
        reason: String,
    },
}

impl KeyReleaseDecision {
    /// Grant a release on a verdict.
    ///
    /// Refuses, in order: stale or future-dated evidence (replay
    /// protection), a `Rejected` verdict, and any verified verdict whose
    /// strength is not [`VerifierStrength::Cryptographic`] — a typed
    /// allowlist, so a new verifier is untrusted until it declares the one
    /// accepted strength truthfully.
    pub fn grant(request: &KeyReleaseRequest, verdict: &Verdict, now_unix: u64) -> Self {
        let refuse = |reason: String| KeyReleaseDecision::Refused {
            key_id: request.key_id.clone(),
            reason,
        };
        // Freshness first: a perfect verdict about stale evidence is a
        // perfect verdict about the past.
        let age_ok = request.evidence.collected_at_unix
            <= now_unix.saturating_add(MAX_EVIDENCE_AGE_SECS)
            && now_unix.saturating_sub(request.evidence.collected_at_unix) <= MAX_EVIDENCE_AGE_SECS;
        if !age_ok {
            return refuse(format!(
                "evidence collected at {} is outside the freshness window ({}s) at {}",
                request.evidence.collected_at_unix, MAX_EVIDENCE_AGE_SECS, now_unix
            ));
        }
        match verdict {
            Verdict::Verified {
                identity,
                verifier,
                strength,
            } => {
                if *strength != VerifierStrength::Cryptographic {
                    return refuse(format!(
                        "key release requires a cryptographic verdict; '{verifier}' \
                         supplied a {strength:?} one"
                    ));
                }
                // The lease's scope carries the measured identity, so the
                // receipt trail binds key → measurements, not key → a name
                // any workload could claim.
                let measured = identity
                    .measurements
                    .iter()
                    .map(|m| format!("{}:{}", m.register, m.value_hex))
                    .collect::<Vec<_>>()
                    .join(",");
                KeyReleaseDecision::Released {
                    key_id: request.key_id.clone(),
                    to: identity.clone(),
                    lease: crate::lease::CapabilityLease::new(
                        format!(
                            "key '{}' → workload '{}' [{}]",
                            request.key_id, identity.name, measured
                        ),
                        crate::lease::LeaseProvenance::Verification,
                        now_unix,
                        None,
                        None,
                    ),
                }
            }
            Verdict::Rejected { reason, .. } => refuse(reason.clone()),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Receipts
// ─────────────────────────────────────────────────────────────────────────────

/// The durable record of a protected execution: what ran, where, under which
/// verdict, and what it did — assembled from semantic execution facts.
///
/// A receipt is evidence *about* an execution, not proof of hardware claims:
/// its strength is exactly the strength of the verdict it embeds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionReceipt {
    pub workload: String,
    pub environment: ExecutionEnvironment,
    /// The verdict the execution ran under, if any verification happened.
    pub verdict: Option<Verdict>,
    pub started_at_unix: u64,
    pub finished_at_unix: u64,
    /// Semantic action facts: (action type, success) per executed action.
    pub actions: Vec<(String, bool)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_standard_environment_claims_nothing() {
        assert!(!ExecutionEnvironment::Standard.claims_confidentiality());
        assert!(ExecutionEnvironment::ConfidentialVm {
            cpu: TeeTechnology::AmdSevSnp
        }
        .claims_confidentiality());
    }

    #[test]
    fn key_release_requires_a_verified_verdict() {
        let request = KeyReleaseRequest {
            key_id: "model-key-1".into(),
            evidence: AttestationEvidence {
                format: EvidenceFormat::StructuredClaims,
                payload: b"{}".to_vec(),
                collected_at_unix: 1_000,
            },
        };
        let rejected = Verdict::Rejected {
            reason: "measurement mismatch".into(),
            verifier: "test".into(),
        };
        match KeyReleaseDecision::grant(&request, &rejected, 1_000) {
            KeyReleaseDecision::Refused { reason, .. } => {
                assert!(reason.contains("mismatch"));
            }
            KeyReleaseDecision::Released { .. } => panic!("released on a rejected verdict"),
        }
    }

    /// The confusability guard: where policy demands cryptographic
    /// verification, a structural verdict — however positive — refuses the
    /// release. A checker of claims must never impersonate a checker of
    /// proofs.
    #[test]
    fn a_structural_verdict_cannot_satisfy_a_cryptographic_requirement() {
        let request = KeyReleaseRequest {
            key_id: "model-key-1".into(),
            evidence: AttestationEvidence {
                format: EvidenceFormat::StructuredClaims,
                payload: b"{}".to_vec(),
                collected_at_unix: 1_000,
            },
        };
        let structural = Verdict::Verified {
            identity: WorkloadIdentity {
                name: "w".into(),
                environment: ExecutionEnvironment::ConfidentialVm {
                    cpu: TeeTechnology::AmdSevSnp,
                },
                measurements: vec![],
            },
            verifier: policy::STRUCTURAL_VERIFIER_NAME.into(),
            strength: VerifierStrength::Structural,
        };
        match KeyReleaseDecision::grant(&request, &structural, 1_000) {
            KeyReleaseDecision::Refused { reason, .. } => {
                assert!(reason.contains("cryptographic"));
            }
            KeyReleaseDecision::Released { .. } => {
                panic!("a structural verdict satisfied a cryptographic requirement")
            }
        }
    }

    /// Replay protection: a cryptographically perfect verdict about stale
    /// evidence is a perfect verdict about the past, and the past does not
    /// hold keys. Future-dated evidence beyond the skew margin fails too.
    #[test]
    fn stale_or_future_evidence_cannot_mint_a_release() {
        let verdict = Verdict::Verified {
            identity: WorkloadIdentity {
                name: "w".into(),
                environment: ExecutionEnvironment::ConfidentialVm {
                    cpu: TeeTechnology::AmdSevSnp,
                },
                measurements: vec![],
            },
            verifier: "external-kbs".into(),
            strength: VerifierStrength::Cryptographic,
        };
        let stale = KeyReleaseRequest {
            key_id: "k".into(),
            evidence: AttestationEvidence {
                format: EvidenceFormat::StructuredClaims,
                payload: b"{}".to_vec(),
                collected_at_unix: 1_000,
            },
        };
        let now = 1_000 + MAX_EVIDENCE_AGE_SECS + 1;
        assert!(matches!(
            KeyReleaseDecision::grant(&stale, &verdict, now),
            KeyReleaseDecision::Refused { .. }
        ));
        let future = KeyReleaseRequest {
            key_id: "k".into(),
            evidence: AttestationEvidence {
                format: EvidenceFormat::StructuredClaims,
                payload: b"{}".to_vec(),
                collected_at_unix: 10_000,
            },
        };
        assert!(matches!(
            KeyReleaseDecision::grant(&future, &verdict, 1_000),
            KeyReleaseDecision::Refused { .. }
        ));
    }

    /// And a release, when granted, is a bounded lease — never permanent.
    #[test]
    fn a_granted_release_is_a_bounded_lease() {
        let request = KeyReleaseRequest {
            key_id: "model-key-1".into(),
            evidence: AttestationEvidence {
                format: EvidenceFormat::StructuredClaims,
                payload: b"{}".to_vec(),
                collected_at_unix: 4_900,
            },
        };
        let verdict = Verdict::Verified {
            identity: WorkloadIdentity {
                name: "inference".into(),
                environment: ExecutionEnvironment::ConfidentialVmWithGpu {
                    cpu: TeeTechnology::AmdSevSnp,
                    gpu_mode: GpuCcMode::On,
                },
                measurements: vec![],
            },
            verifier: "external-kbs".into(),
            strength: VerifierStrength::Cryptographic,
        };
        match KeyReleaseDecision::grant(&request, &verdict, 5_000) {
            KeyReleaseDecision::Released { lease, .. } => {
                assert_eq!(
                    lease.provenance,
                    crate::lease::LeaseProvenance::Verification,
                    "a key release must not claim a human or a model created it"
                );
                assert!(lease.is_live(5_001));
                assert!(
                    !lease.is_live(5_000 + crate::lease::MAX_VALIDITY_SECS + 1),
                    "a key release outlived the lease ceiling"
                );
            }
            KeyReleaseDecision::Refused { reason, .. } => panic!("refused: {reason}"),
        }
    }
}
