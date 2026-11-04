//! The gate: `Unattested` → [`Attested`], fail-closed.
//!
//! [`Attested`] is a proof object. Its only constructor is private to this
//! module and is reached only through [`Gate::attest`], which demands a fresh
//! [`Verdict::Verified`] of [`VerifierStrength::Cryptographic`]. Everything
//! that must not happen outside a trusted environment — binding the socket,
//! releasing the key — takes an `Attested` by value or reference, so the type
//! system, not a runtime flag, is what forbids it.

use x0ne_confidential::model::{
    AttestationEvidence, ExecutionEnvironment, Verdict, VerifierStrength, WorkloadIdentity,
    MAX_EVIDENCE_AGE_SECS,
};

/// Why the gate refused. Each reason is a distinct exit code of the binary.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Refused {
    #[error("evidence rejected by the verifier: {0}")]
    Rejected(String),
    #[error("verifier strength is {0:?}; serving requires Cryptographic")]
    NotCryptographic(VerifierStrength),
    #[error(
        "evidence collected at {collected_at} is outside the freshness window at {now} ({max}s)"
    )]
    Stale {
        collected_at: u64,
        now: u64,
        max: u64,
    },
    #[error("environment claims no confidentiality: {0:?}")]
    NotConfidential(ExecutionEnvironment),
}

impl Refused {
    /// Process exit code: 64 + the reason. Non-zero, distinct, documented.
    pub fn exit_code(&self) -> i32 {
        match self {
            Refused::Rejected(_) => 65,
            Refused::NotCryptographic(_) => 66,
            Refused::Stale { .. } => 67,
            Refused::NotConfidential(_) => 68,
        }
    }
}

/// The proof that this process runs where it may serve.
///
/// Not `Clone` by accident of design? It IS `Clone`: a proof copied is the
/// same proof. What it is not is *constructible* outside [`Gate::attest`] —
/// the field is private and there is no `Default`, no `new`, no `From`.
#[derive(Debug, Clone)]
pub struct Attested {
    identity: WorkloadIdentity,
    environment: ExecutionEnvironment,
    verdict: Verdict,
    attested_at_unix: u64,
}

impl Attested {
    pub fn identity(&self) -> &WorkloadIdentity {
        &self.identity
    }
    pub fn environment(&self) -> &ExecutionEnvironment {
        &self.environment
    }
    pub fn verdict(&self) -> &Verdict {
        &self.verdict
    }
    pub fn attested_at_unix(&self) -> u64 {
        self.attested_at_unix
    }
}

/// The gate itself: stateless, the rules live here and nowhere else.
pub struct Gate;

impl Gate {
    /// Turn evidence + verdict into a proof, or refuse.
    ///
    /// Order of refusals, each a test: not confidential → stale → rejected →
    /// not cryptographic. The order matters only for the message; every one of
    /// them is a refusal.
    pub fn attest(
        environment: ExecutionEnvironment,
        evidence: &AttestationEvidence,
        verdict: Verdict,
        now_unix: u64,
    ) -> Result<Attested, Refused> {
        if !environment.claims_confidentiality() {
            return Err(Refused::NotConfidential(environment));
        }
        let age_ok = evidence.collected_at_unix <= now_unix.saturating_add(MAX_EVIDENCE_AGE_SECS)
            && now_unix.saturating_sub(evidence.collected_at_unix) <= MAX_EVIDENCE_AGE_SECS;
        if !age_ok {
            return Err(Refused::Stale {
                collected_at: evidence.collected_at_unix,
                now: now_unix,
                max: MAX_EVIDENCE_AGE_SECS,
            });
        }
        match &verdict {
            Verdict::Rejected { reason, .. } => Err(Refused::Rejected(reason.clone())),
            Verdict::Verified {
                identity, strength, ..
            } => {
                if *strength != VerifierStrength::Cryptographic {
                    return Err(Refused::NotCryptographic(*strength));
                }
                Ok(Attested {
                    identity: identity.clone(),
                    // SECURITY: the environment the proof publishes is the
                    // VERIFIED one, from the verdict's identity — never the
                    // caller's, which is the collector's self-report that
                    // nothing checked. Before this, main fed the
                    // collector's `ConfidentialVm` here and every receipt
                    // published it while the verdict inside the same signed
                    // document said `ConfidentialVmWithGpu` — the caller's
                    // value only gates the NotConfidential pre-check above.
                    environment: identity.environment.clone(),
                    verdict: verdict.clone(),
                    attested_at_unix: now_unix,
                })
            }
        }
    }
}

/// Bind the listener. Takes the proof by reference: **there is no other
/// `bind` in the library or in `x0ne-infer`**, and this one cannot be called
/// without an
/// [`Attested`] — which is the whole security argument of the gate.
///
/// Returns the bound address. The HTTP surface (`server` module, next step)
/// is built on the listener this returns, never on its own.
pub async fn bind(proof: &Attested, addr: &str) -> std::io::Result<tokio::net::TcpListener> {
    tracing::info!(
        workload = %proof.identity().name,
        environment = ?proof.environment(),
        "binding: attested"
    );
    tokio::net::TcpListener::bind(addr).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use x0ne_confidential::model::{EvidenceFormat, GpuCcMode, TeeTechnology};

    fn evidence(at: u64) -> AttestationEvidence {
        AttestationEvidence {
            format: EvidenceFormat::StructuredClaims,
            payload: b"{}".to_vec(),
            collected_at_unix: at,
        }
    }
    fn identity() -> WorkloadIdentity {
        WorkloadIdentity {
            name: "x0ne-infer".into(),
            environment: cc_env(),
            measurements: vec![],
        }
    }
    fn cc_env() -> ExecutionEnvironment {
        ExecutionEnvironment::ConfidentialVmWithGpu {
            cpu: TeeTechnology::IntelTdx,
            gpu_mode: GpuCcMode::On,
        }
    }
    fn verified(strength: VerifierStrength) -> Verdict {
        Verdict::Verified {
            identity: identity(),
            verifier: "test".into(),
            strength,
        }
    }

    /// A structural verdict — a shape check with no signature behind it — does NOT
    /// open the gate. A failure would mean the engine serves on an unsigned
    /// quote, which is the difference between "attested" and "looks right".
    #[test]
    fn a_structural_verdict_does_not_open_the_gate() {
        let r = Gate::attest(
            cc_env(),
            &evidence(1_000),
            verified(VerifierStrength::Structural),
            1_000,
        );
        assert_eq!(
            r.unwrap_err(),
            Refused::NotCryptographic(VerifierStrength::Structural)
        );
    }

    /// A rejected verdict refuses with the verifier's reason.
    #[test]
    fn a_rejected_verdict_refuses_with_its_reason() {
        let v = Verdict::Rejected {
            reason: "measurement mismatch on RTMR0".into(),
            verifier: "test".into(),
        };
        let r = Gate::attest(cc_env(), &evidence(1_000), v, 1_000);
        assert!(matches!(r, Err(Refused::Rejected(ref s)) if s.contains("RTMR0")));
    }

    /// Fresh, cryptographic, confidential: the gate opens — and the proof
    /// carries the identity and the environment for the receipts.
    #[test]
    fn a_fresh_cryptographic_verdict_opens_the_gate() {
        let proof = Gate::attest(
            cc_env(),
            &evidence(1_000),
            verified(VerifierStrength::Cryptographic),
            1_010,
        )
        .expect("opens");
        assert_eq!(proof.identity().name, "x0ne-infer");
        assert!(proof.environment().claims_confidentiality());
        assert_eq!(proof.attested_at_unix(), 1_010);
    }

    /// Evidence older than the window, or from the future, is refused even
    /// with a perfect verdict — a perfect verdict about the past.
    #[test]
    fn stale_or_future_evidence_is_refused() {
        let old = Gate::attest(
            cc_env(),
            &evidence(1_000),
            verified(VerifierStrength::Cryptographic),
            1_000 + MAX_EVIDENCE_AGE_SECS + 1,
        );
        assert!(matches!(old, Err(Refused::Stale { .. })));
        let future = Gate::attest(
            cc_env(),
            &evidence(10_000),
            verified(VerifierStrength::Cryptographic),
            1_000,
        );
        assert!(matches!(future, Err(Refused::Stale { .. })));
    }

    /// A standard environment never opens the gate, verdict or not: absence
    /// of a confidentiality claim is not a weak claim.
    #[test]
    fn a_standard_environment_is_refused_before_the_verdict_is_read() {
        let r = Gate::attest(
            ExecutionEnvironment::Standard,
            &evidence(1_000),
            verified(VerifierStrength::Cryptographic),
            1_000,
        );
        assert!(matches!(r, Err(Refused::NotConfidential(_))));
    }

    /// Every refusal has a distinct, non-zero exit code.
    #[test]
    fn refusals_exit_non_zero_and_distinct() {
        let codes = [
            Refused::Rejected("x".into()).exit_code(),
            Refused::NotCryptographic(VerifierStrength::Structural).exit_code(),
            Refused::Stale {
                collected_at: 0,
                now: 0,
                max: 0,
            }
            .exit_code(),
            Refused::NotConfidential(ExecutionEnvironment::Standard).exit_code(),
        ];
        assert!(codes.iter().all(|c| *c != 0));
        let mut sorted = codes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), codes.len(), "distinct");
    }

    /// The proof is the only way to bind: with it, a socket; without it, the
    /// function cannot even be called — this test is the documentation of
    /// that fact, and it binds for real on a loopback ephemeral port.
    #[tokio::test]
    async fn binding_requires_the_proof_and_then_binds() {
        let proof = Gate::attest(
            cc_env(),
            &evidence(1_000),
            verified(VerifierStrength::Cryptographic),
            1_000,
        )
        .unwrap();
        let listener = bind(&proof, "127.0.0.1:0").await.expect("binds");
        assert!(listener.local_addr().unwrap().port() > 0);
    }
}
