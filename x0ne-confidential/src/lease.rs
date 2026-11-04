//! Bounded authority across time.
//!
//! "Allow this now" is not "allow this every hour forever": a capability
//! that keeps firing without a bound is permanent trust created as a side
//! effect of convenience.
//!
//! A [`CapabilityLease`] is the bound. It answers one question — *may this
//! deferred thing fire again, now?* — from four facts it carries:
//!
//! - a validity window (`expires_at_unix`),
//! - a firing budget (`max_fires`),
//! - a revocation bit,
//! - auditable provenance (who created it, for what scope).
//!
//! What a lease is **not**: authority to execute. A live lease only lets the
//! deferred action reach the check that runs *at fire time*, under the
//! policy of that moment: a revoked policy wins over a live lease, and an
//! expired lease stops a fire before anything is proposed.
//!
//! A key release is a lease whose scope is a workload identity and whose
//! issuance is a verified attestation verdict
//! ([`crate::model::KeyReleaseDecision`]).

/// Who brought this lease into existence. Auditable provenance, carried for
/// the receipt trail rather than consulted for decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseProvenance {
    /// The model scheduled it (e.g. a recurring `CronSchedule` action).
    Model,
    /// A human typed it at their own keyboard.
    Human,
    /// Issued by system machinery on the strength of a verification verdict
    /// (e.g. an attestation-bound key release). Neither a human keystroke nor
    /// a model proposal — the receipt trail must not claim either.
    Verification,
}

impl LeaseProvenance {
    pub fn as_str(self) -> &'static str {
        match self {
            LeaseProvenance::Model => "model",
            LeaseProvenance::Human => "human",
            LeaseProvenance::Verification => "verification",
        }
    }
}

/// Why a lease refused to fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseDenied {
    /// The validity window is behind us.
    Expired,
    /// The firing budget is spent.
    Exhausted,
    /// Someone revoked it.
    Revoked,
}

impl LeaseDenied {
    pub fn as_str(self) -> &'static str {
        match self {
            LeaseDenied::Expired => "expired",
            LeaseDenied::Exhausted => "exhausted",
            LeaseDenied::Revoked => "revoked",
        }
    }
}

/// The ceiling on how many times any single deferred grant may fire,
/// whatever it asks for. At the tightest plausible cadence the ticker
/// supports (one fire per second is clamped upward elsewhere), 720 fires is
/// hours of standing behaviour — enough for a monitoring loop, far short of
/// forever.
pub const MAX_FIRES_CEILING: u32 = 720;

/// The ceiling on any single deferred grant's validity window: 24 hours.
/// Standing behaviour that should outlive a day should be re-established by
/// whoever wanted it, not inherited from a decision nobody remembers.
pub const MAX_VALIDITY_SECS: u64 = 24 * 60 * 60;

/// A bounded, revocable permission for deferred work to keep becoming
/// actions.
#[derive(Debug, Clone)]
pub struct CapabilityLease {
    /// What this lease is for, in words a receipt can carry
    /// (e.g. `"cron 'disk' → RunCommand df"`).
    pub scope: String,
    pub provenance: LeaseProvenance,
    pub issued_at_unix: u64,
    /// End of the validity window. Always set — [`CapabilityLease::new`]
    /// clamps to [`MAX_VALIDITY_SECS`], so "no expiry" cannot be expressed.
    pub expires_at_unix: u64,
    /// Firing budget. Always set — clamped to [`MAX_FIRES_CEILING`].
    pub max_fires: u32,
    fires: u32,
    revoked: bool,
}

impl CapabilityLease {
    /// Issue a lease, clamping every requested bound to the ceilings. A
    /// request for more than the ceiling gets the ceiling, silently — the
    /// caller asked for standing behaviour and receives the longest the
    /// policy allows, which is the useful interpretation of the request.
    pub fn new(
        scope: impl Into<String>,
        provenance: LeaseProvenance,
        now_unix: u64,
        requested_validity_secs: Option<u64>,
        requested_max_fires: Option<u32>,
    ) -> Self {
        let validity = requested_validity_secs
            .unwrap_or(MAX_VALIDITY_SECS)
            .min(MAX_VALIDITY_SECS);
        let max_fires = requested_max_fires
            .unwrap_or(MAX_FIRES_CEILING)
            .min(MAX_FIRES_CEILING)
            .max(1);
        Self {
            scope: scope.into(),
            provenance,
            issued_at_unix: now_unix,
            expires_at_unix: now_unix.saturating_add(validity),
            max_fires,
            fires: 0,
            revoked: false,
        }
    }

    /// A single-shot lease: one fire, ordinary window.
    pub fn single_fire(
        scope: impl Into<String>,
        provenance: LeaseProvenance,
        now_unix: u64,
    ) -> Self {
        Self::new(scope, provenance, now_unix, None, Some(1))
    }

    /// May the deferred thing fire now? Consumes one firing on success.
    ///
    /// Checked in the order a receipt should report: revocation is a
    /// decision, expiry is a boundary, exhaustion is a budget.
    pub fn try_fire(&mut self, now_unix: u64) -> Result<u32, LeaseDenied> {
        if self.revoked {
            return Err(LeaseDenied::Revoked);
        }
        if now_unix >= self.expires_at_unix {
            return Err(LeaseDenied::Expired);
        }
        if self.fires >= self.max_fires {
            return Err(LeaseDenied::Exhausted);
        }
        self.fires += 1;
        Ok(self.fires)
    }

    /// Whether the lease could still fire, without consuming anything.
    pub fn is_live(&self, now_unix: u64) -> bool {
        !self.revoked && now_unix < self.expires_at_unix && self.fires < self.max_fires
    }

    /// Withdraw the lease. Nothing un-revokes it.
    pub fn revoke(&mut self) {
        self.revoked = true;
    }

    /// How many times it has fired, for receipts.
    pub fn fires(&self) -> u32 {
        self.fires
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lease(validity: Option<u64>, fires: Option<u32>) -> CapabilityLease {
        CapabilityLease::new("test scope", LeaseProvenance::Model, 1_000, validity, fires)
    }

    #[test]
    fn a_lease_fires_within_its_bounds_and_counts() {
        let mut l = lease(Some(100), Some(2));
        assert_eq!(l.try_fire(1_010), Ok(1));
        assert_eq!(l.try_fire(1_020), Ok(2));
        assert_eq!(l.try_fire(1_030), Err(LeaseDenied::Exhausted));
        assert_eq!(l.fires(), 2, "a refused fire must not count");
    }

    #[test]
    fn expiry_wins_over_remaining_budget() {
        let mut l = lease(Some(50), Some(10));
        assert_eq!(l.try_fire(1_010), Ok(1));
        assert_eq!(l.try_fire(1_050), Err(LeaseDenied::Expired));
        assert!(!l.is_live(1_050));
    }

    #[test]
    fn revocation_wins_over_everything() {
        let mut l = lease(Some(1_000_000), Some(100));
        l.revoke();
        assert_eq!(l.try_fire(1_001), Err(LeaseDenied::Revoked));
        assert!(!l.is_live(1_001));
    }

    /// "No bounds requested" is not "no bounds": the ceilings apply. This is
    /// the invariant that stops permanent autonomous trust from being created
    /// by omission.
    #[test]
    fn unbounded_requests_receive_the_ceilings() {
        let l = lease(None, None);
        assert_eq!(l.expires_at_unix, 1_000 + MAX_VALIDITY_SECS);
        assert_eq!(l.max_fires, MAX_FIRES_CEILING);
    }

    /// Nor can a request exceed them.
    #[test]
    fn oversized_requests_are_clamped() {
        let l = lease(Some(MAX_VALIDITY_SECS * 10), Some(MAX_FIRES_CEILING * 10));
        assert_eq!(l.expires_at_unix, 1_000 + MAX_VALIDITY_SECS);
        assert_eq!(l.max_fires, MAX_FIRES_CEILING);
    }

    #[test]
    fn a_single_fire_lease_fires_exactly_once() {
        let mut l = CapabilityLease::single_fire("one shot", LeaseProvenance::Human, 1_000);
        assert_eq!(l.try_fire(1_001), Ok(1));
        assert_eq!(l.try_fire(1_002), Err(LeaseDenied::Exhausted));
    }
}
