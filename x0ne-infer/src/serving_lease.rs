//! The key-release lease as the server enforces it: on the wall clock AND a
//! monotonic one, re-checked when a computation starts, not before its wait.
//! A wall clock moved back by the host extended a SystemTime-only lease.

use std::time::{Duration, Instant};

use x0ne_confidential::lease::CapabilityLease;

/// Wall-clock steps back larger than this, since the release, are refused.
pub const MAX_CLOCK_REGRESSION_SECS: u64 = 60;

/// Why the server will not compute: the message is the 503 body.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct LeaseExpired(pub String);

pub struct ServingLease {
    lease: CapabilityLease,
    /// The monotonic instant of the key release, paired with `issued_at_unix`.
    anchor: Instant,
    /// One computation at a time; the lease is judged once this is held.
    compute: std::sync::Mutex<()>,
    /// The platform's standing under Intel's live verdict, re-read while
    /// serving (`tcb_standing`): lost or stale, nothing computes either.
    standing: Option<std::sync::Arc<crate::tcb_standing::PlatformStanding>>,
    /// One answer at a time, until its time is a size (`crate::padding`).
    turns: crate::padding::Turns,
}

fn wall_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl ServingLease {
    /// `anchor` is the `Instant` of the release that minted `lease`.
    pub fn new(lease: CapabilityLease, anchor: Instant) -> Self {
        Self {
            lease,
            anchor,
            compute: std::sync::Mutex::new(()),
            standing: None,
            turns: crate::padding::Turns::default(),
        }
    }

    /// The same lease, judged with the platform's standing too: a
    /// computation needs the key's window AND a live verdict on the
    /// platform ([`crate::tcb_standing::PlatformStanding::check`]).
    #[must_use]
    pub fn on(mut self, standing: std::sync::Arc<crate::tcb_standing::PlatformStanding>) -> Self {
        self.standing = Some(standing);
        self
    }

    /// The model's next turn: taken before a chat computes, held past it
    /// until the answer's time is a size, so that the next one starts then.
    pub async fn take_turn(&self) -> crate::padding::Turn {
        self.turns.take().await
    }

    pub fn lease(&self) -> &CapabilityLease {
        &self.lease
    }

    /// What the platform's last verdict said about its patch level, for the
    /// receipt of the answer being computed; `None` without a standing.
    pub fn tcb_disclosure(&self) -> Option<crate::tcb_disclosure::TcbDisclosure> {
        self.standing.as_ref().and_then(|s| s.disclosure())
    }

    /// Live now, by both clocks.
    pub fn check(&self) -> Result<(), LeaseExpired> {
        self.check_at(wall_unix(), Instant::now())
    }

    /// Live at `wall` (unix seconds) and `mono`: the stricter clock wins.
    pub fn check_at(&self, wall: u64, mono: Instant) -> Result<(), LeaseExpired> {
        let l = &self.lease;
        let window = l.expires_at_unix.saturating_sub(l.issued_at_unix);
        let elapsed = mono.saturating_duration_since(self.anchor);
        let restart = "restart to re-attest; nothing is served on a stale proof";
        if elapsed >= Duration::from_secs(window) {
            return Err(LeaseExpired(format!(
                "the attestation this server was admitted on expired: {}s of its {window}s \
                 have run on the monotonic clock; {restart}",
                elapsed.as_secs()
            )));
        }
        if !l.is_live(wall) {
            return Err(LeaseExpired(format!(
                "the attestation this server was admitted on expired at {} (unix); {restart}",
                l.expires_at_unix
            )));
        }
        let expected = l.issued_at_unix.saturating_add(elapsed.as_secs());
        if wall.saturating_add(MAX_CLOCK_REGRESSION_SECS) < expected {
            return Err(LeaseExpired(format!(
                "the wall clock went back {}s since the key release; the lease is treated \
                 as expired; {restart}",
                expected - wall
            )));
        }
        // The key's window is open; the platform must still stand under
        // Intel's live verdict (`tcb_standing`), judged on the same clock.
        if let Some(standing) = &self.standing {
            standing.check(wall).map_err(|e| LeaseExpired(e.0))?;
        }
        Ok(())
    }

    /// Run `f` once no other computation runs and the lease is still live.
    /// Blocking: call from `spawn_blocking`.
    pub fn run_live<R>(&self, f: impl FnOnce() -> R) -> Result<R, LeaseExpired> {
        let _one = self.compute.lock().unwrap_or_else(|p| p.into_inner());
        self.check()?;
        Ok(f())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use x0ne_confidential::lease::LeaseProvenance;

    const T0: u64 = 1_800_000_000;

    fn lease(validity: u64) -> (ServingLease, Instant) {
        let anchor = Instant::now();
        let l = CapabilityLease::new("t", LeaseProvenance::Verification, T0, Some(validity), None);
        (ServingLease::new(l, anchor), anchor)
    }

    #[test]
    fn live_within_the_window_on_both_clocks() {
        let (s, a) = lease(3600);
        s.check_at(T0 + 10, a + Duration::from_secs(10)).unwrap();
    }

    /// A host that moves the guest clock back kept a SystemTime lease
    /// alive. The monotonic clock ends it anyway.
    #[test]
    fn a_wall_clock_held_back_does_not_extend_the_lease() {
        let (s, a) = lease(3600);
        let err = s
            .check_at(T0 + 100, a + Duration::from_secs(3600))
            .unwrap_err();
        assert!(err.0.contains("monotonic"), "{err}");
    }

    #[test]
    fn a_wall_clock_moved_back_past_the_tolerance_is_refused() {
        let (s, a) = lease(3600);
        let mono = a + Duration::from_secs(1000);
        s.check_at(T0 + 1000 - MAX_CLOCK_REGRESSION_SECS, mono)
            .unwrap();
        let err = s
            .check_at(T0 + 1000 - MAX_CLOCK_REGRESSION_SECS - 1, mono)
            .unwrap_err();
        assert!(err.0.contains("went back"), "{err}");
    }

    #[test]
    fn the_wall_clock_alone_still_ends_the_lease() {
        let (s, a) = lease(3600);
        let err = s
            .check_at(T0 + 3600, a + Duration::from_secs(5))
            .unwrap_err();
        assert!(err.0.contains("expired at"), "{err}");
    }

    /// SECURITY: the platform's standing bounds the computation like the
    /// key's window does. Lost after start-up (Intel's verdict turned), or
    /// not renewed within the collateral age limit, the lease refuses and
    /// nothing runs; a standing renewed in time computes.
    #[test]
    fn a_lease_on_a_lost_or_stale_standing_refuses_to_compute() {
        use crate::tcb::TcbPolicy;
        use crate::tcb_standing::{LiveFailure, PlatformStanding};
        // The production floor with an hour of collateral age: the key's
        // window (a day at most) stays open while the standing goes stale.
        let policy = TcbPolicy::tightened(None, Some(3600)).unwrap();
        let standing = std::sync::Arc::new(PlatformStanding::verified_at(T0, policy));
        let anchor = Instant::now();
        let l = CapabilityLease::new("t", LeaseProvenance::Verification, T0, Some(86_400), None);
        let s = ServingLease::new(l, anchor).on(standing.clone());
        s.check_at(T0 + 10, anchor + Duration::from_secs(10))
            .unwrap();
        // Stale: an hour and a second without a renewal, the key's window still open.
        let later = policy.max_collateral_age_secs + 1;
        let err = s
            .check_at(T0 + later, anchor + Duration::from_secs(later))
            .unwrap_err();
        assert!(err.0.contains("not renewed"), "{err}");
        standing.apply(T0 + later, Ok(()));
        s.check_at(T0 + later, anchor + Duration::from_secs(later))
            .unwrap();
        // Lost: Intel's verdict against the platform.
        standing.apply(
            T0 + later + 1,
            Err(LiveFailure::Refused(
                crate::tcb::TcbRejected::BelowEveryLevel,
            )),
        );
        let err = s
            .check_at(T0 + later + 2, anchor + Duration::from_secs(later + 2))
            .unwrap_err();
        assert!(
            err.0.contains("lost its standing") && err.0.contains("re-attest"),
            "{err}"
        );
        // Nothing renews it: a verdict in the platform's favour changes nothing.
        standing.apply(T0 + later + 3, Ok(()));
        assert!(s
            .check_at(T0 + later + 3, anchor + Duration::from_secs(later + 3))
            .is_err());
    }

    #[test]
    fn run_live_refuses_without_running_once_expired() {
        let l = CapabilityLease::new("t", LeaseProvenance::Verification, 1_000, Some(1), None);
        let s = ServingLease::new(l, Instant::now());
        let mut ran = false;
        assert!(s.run_live(|| ran = true).is_err());
        assert!(!ran);
    }
}
