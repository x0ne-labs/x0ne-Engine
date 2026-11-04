//! The platform's standing while the process serves: Intel's verdict
//! re-read at an interval, instead of a start-up measurement asserted for
//! the process lifetime.
//!
//! Until 2026-10-23 the live collateral (tcbInfo, QE identity, PCK CRL, root
//! CA CRL) was fetched once, before the first answer, and
//! `tcb::LIVE_LIMITATIONS` said so: a platform Intel rated OutOfDate after
//! boot kept serving until restart. Now
//! [`PlatformStanding`](crate::tcb_standing::PlatformStanding) holds the time
//! of the last verdict in the platform's favour,
//! [`revalidate`](crate::tcb_standing::revalidate) re-reads the collateral
//! every [`REVALIDATE_EVERY_SECS`](crate::tcb_standing::REVALIDATE_EVERY_SECS)
//! against the quote the process was admitted on, under the same
//! [`TcbPolicy`](crate::tcb::TcbPolicy), and
//! `serving_lease::ServingLease` refuses to compute on a standing that is
//! lost or stale:
//!
//! - a verdict against the platform (a status Intel does not accept, no
//!   level at all, a TDX module version Intel publishes no identity or no
//!   level for, a revoked PCK or PCK CA, collateral under an evaluation
//!   below the floor, a QE identity that does not match): the standing is
//!   LOST, for good. The next question is a 503 naming the reason and the
//!   model does not run. Only a restart, a new attestation, serves again;
//! - collateral that could not be fetched or read (PCS or the CRL host
//!   unreachable, a body that does not parse or verify, collateral issued
//!   past the age limit): the last verdict stands until it is older than the
//!   policy's collateral age limit (`max_collateral_age_secs`, 24 hours in
//!   production), then the answers stop too, with the same 503. PCS saying
//!   nothing for a day is a day without a verdict, and nothing is served
//!   on no verdict.
//!
//! What a verdict in the platform's favour means is unchanged: collateral
//! issued at most `max_collateral_age_secs` before it. So while PCS
//! answers, the statuses behind any answer are at worst
//! `REVALIDATE_EVERY_SECS` plus that age behind Intel.
//!
//! Since 2026-10-28 the verdict follows the policy's
//! [`TcbMode`](crate::tcb::TcbMode) ([`judge_platform_live`](crate::tcb_standing::judge_platform_live)): under
//! `Transparent`, the production mode, a verdict in the platform's favour
//! stands on Intel's `standard` collateral, and Intel's `early` collateral
//! is read too, for the disclosure the standing carries into every receipt
//! ([`crate::tcb_disclosure`]): a platform `early` rates OutOfDate keeps
//! serving, and every answer names it.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::attest_v2::BoxFuture;
use crate::nras_http::{fetch_pcs_collateral, fetch_pcs_der};
use crate::pcs::TrustAnchor;
use crate::tcb::{self, LiveJudgement, PcsUpdate, TcbPolicy, TcbRejected, TcbStatus};
use crate::tcb_disclosure::TcbDisclosure;

/// How often the live collateral is re-read while serving. Intel publishes
/// a TCB recovery on its advisory day and PCS carries it in the `early`
/// tcbInfo the same day: an hour bounds how long a platform Intel has just
/// rated OutOfDate keeps answering, at four PCS GETs an hour per process.
pub const REVALIDATE_EVERY_SECS: u64 = 3600;

/// Why the live check did not end in the platform's favour.
#[derive(Debug, thiserror::Error)]
pub enum LiveFailure {
    /// Fetched and read: Intel's collateral refuses the platform, or could
    /// not be verified ([`LiveFailure::is_a_verdict`] tells which).
    #[error("{0}")]
    Refused(#[from] TcbRejected),
    /// Not fetched: the host did not answer, or answered too much or too little.
    #[error("not fetched: {0}")]
    Fetch(String),
}

impl LiveFailure {
    /// True when Intel's collateral, read and verified, speaks against the
    /// platform; false when the collateral could not be fetched, read,
    /// verified or was too old to count, which says nothing about the
    /// platform. A variant not listed here is a verdict: new refusals fail
    /// closed.
    pub fn is_a_verdict(&self) -> bool {
        match self {
            Self::Fetch(_) => false,
            Self::Refused(e) => !matches!(
                e,
                TcbRejected::Collateral(..)
                    | TcbRejected::PckCrl(_)
                    | TcbRejected::RootCrl(_)
                    | TcbRejected::CollateralTooOld { .. }
            ),
        }
    }
}

/// What applying one live outcome did to the standing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Applied {
    /// A verdict in the platform's favour: the standing is dated now.
    Renewed,
    /// No verdict either way: the standing keeps its date, and the reason.
    Unchanged(String),
    /// A verdict against the platform, now or earlier: the standing is lost.
    Lost(String),
}

/// Why the server will not compute: the message is the 503 body.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct StandingLost(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    Standing { verified_at: u64 },
    Lost { at: u64, reason: String },
}

/// The last verdict in the platform's favour, and the policy that bounds
/// its age. Shared between the serving path, which reads it before every
/// computation, and [`revalidate`], which renews or loses it.
pub struct PlatformStanding {
    state: Mutex<State>,
    policy: TcbPolicy,
    /// What the last verdict said about the platform's patch level: the
    /// start-up one, then each renewal's. Read into every receipt.
    disclosure: Mutex<Option<TcbDisclosure>>,
}

const RESTART: &str = "restart to re-attest; nothing is served on a stale proof";

impl PlatformStanding {
    /// A standing established at `now_unix` (the start-up verdict).
    pub fn verified_at(now_unix: u64, policy: TcbPolicy) -> Self {
        Self {
            state: Mutex::new(State::Standing {
                verified_at: now_unix,
            }),
            policy,
            disclosure: Mutex::new(None),
        }
    }

    /// The same standing, carrying the start-up verdict's disclosure.
    #[must_use]
    pub fn disclosing(self, disclosure: Option<TcbDisclosure>) -> Self {
        *self.disclosure.lock().unwrap_or_else(|e| e.into_inner()) = disclosure;
        self
    }

    /// The disclosure of the last verdict in the platform's favour.
    pub fn disclosure(&self) -> Option<TcbDisclosure> {
        self.disclosure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// A renewal's disclosure: `j` at `now_unix`; a view of Intel's latest
    /// collateral this check could not read is kept from the last one, with
    /// the failure named ([`TcbDisclosure::renewed`]).
    pub fn disclose(&self, j: &LiveJudgement, now_unix: u64) {
        let mut d = self.disclosure.lock().unwrap_or_else(|e| e.into_inner());
        *d = Some(TcbDisclosure::renewed(
            d.as_ref(),
            j,
            self.policy.min_tcb_evaluation_data_number,
            self.policy.max_collateral_age_secs,
            now_unix,
        ));
    }

    /// When the standing was last renewed; `None` once lost.
    pub fn verified_at_unix(&self) -> Option<u64> {
        match &*self.state.lock().unwrap_or_else(|e| e.into_inner()) {
            State::Standing { verified_at } => Some(*verified_at),
            State::Lost { .. } => None,
        }
    }

    /// The verdict that lost the standing, if any.
    pub fn lost_reason(&self) -> Option<String> {
        match &*self.state.lock().unwrap_or_else(|e| e.into_inner()) {
            State::Standing { .. } => None,
            State::Lost { reason, .. } => Some(reason.clone()),
        }
    }

    /// Live at `now_unix`: not lost, and renewed within the collateral age
    /// limit. A wall clock behind the last renewal reads as age zero; the
    /// lease's monotonic clock is what catches a clock held back.
    pub fn check(&self, now_unix: u64) -> Result<(), StandingLost> {
        match &*self.state.lock().unwrap_or_else(|e| e.into_inner()) {
            State::Lost { at, reason } => Err(StandingLost(format!(
                "the platform this server was admitted on lost its standing at {at} (unix): \
                 {reason}; {RESTART}"
            ))),
            State::Standing { verified_at } => {
                let age = now_unix.saturating_sub(*verified_at);
                let max = self.policy.max_collateral_age_secs;
                if age > max {
                    return Err(StandingLost(format!(
                        "the live verdict this server stands on is {age} s old, past the {max} s \
                         collateral age limit, and was not renewed; {RESTART}"
                    )));
                }
                Ok(())
            }
        }
    }

    /// One live outcome at `now_unix`: renewed on a verdict in the
    /// platform's favour, lost on one against it, unchanged otherwise. A
    /// lost standing stays lost whatever comes after.
    pub fn apply(&self, now_unix: u64, outcome: Result<(), LiveFailure>) -> Applied {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let State::Lost { reason, .. } = &*state {
            return Applied::Lost(reason.clone());
        }
        match outcome {
            Ok(()) => {
                *state = State::Standing {
                    verified_at: now_unix,
                };
                Applied::Renewed
            }
            Err(f) if f.is_a_verdict() => {
                let reason = f.to_string();
                *state = State::Lost {
                    at: now_unix,
                    reason: reason.clone(),
                };
                Applied::Lost(reason)
            }
            Err(f) => Applied::Unchanged(f.to_string()),
        }
    }
}

/// The live half of the CPU proof, as the start-up paths run it, in one
/// call: tcbInfo and QE identity under `policy` (the `standard` answer and
/// the `early` one under `Transparent`, the `early` one under `Strict`), the
/// PCK CRL, the root CA CRL, all against `quote` and `anchor` at
/// `now_unix`. Under `Transparent`, an `early` answer that cannot be
/// fetched or read is named in the judgement, not refused.
pub async fn judge_platform_live(
    quote: &[u8],
    pcs_base_url: &str,
    root_crl_url: &str,
    anchor: &TrustAnchor,
    policy: &TcbPolicy,
    now_unix: u64,
) -> Result<LiveJudgement, LiveFailure> {
    let fetch = |what: &str, e: anyhow::Error| LiveFailure::Fetch(format!("{what}: {e:#}"));
    let fmspc = tcb::pck_tcb(quote)?.fmspc_hex;
    let update = policy.acceptance_update();
    let (tcb_body, tcb_chain) =
        fetch_pcs_collateral(&tcb::pcs_tcb_url_for(pcs_base_url, &fmspc, update))
            .await
            .map_err(|e| fetch("tcbInfo", e))?;
    let (qe_body, qe_chain) =
        fetch_pcs_collateral(&tcb::pcs_qe_identity_url_for(pcs_base_url, update))
            .await
            .map_err(|e| fetch("QE identity", e))?;
    let early = match update {
        PcsUpdate::Early => None,
        PcsUpdate::Standard => Some(fetch_early(pcs_base_url, &fmspc).await),
    };
    let accept = tcb::Collateral {
        tcb_body: &tcb_body,
        tcb_chain_pem: &tcb_chain,
        qe_body: &qe_body,
        qe_chain_pem: &qe_chain,
    };
    let latest = early.as_ref().map(|r| {
        r.as_ref()
            .map(|(tb, tc, qb, qc)| tcb::Collateral {
                tcb_body: tb,
                tcb_chain_pem: tc,
                qe_body: qb,
                qe_chain_pem: qc,
            })
            .map_err(Clone::clone)
    });
    let judgement = tcb::judge_live(quote, &accept, latest, anchor, now_unix, policy)?;
    let ca = tcb::pck_crl_ca(quote)?;
    let crl = fetch_pcs_der(&tcb::pcs_pck_crl_url(pcs_base_url, ca))
        .await
        .map_err(|e| fetch("PCK CRL", e))?;
    tcb::verify_pck_crl(quote, &crl, anchor, now_unix, policy)?;
    let root_crl = fetch_pcs_der(root_crl_url)
        .await
        .map_err(|e| fetch("root CA CRL", e))?;
    let mut chains: Vec<&str> = vec![&tcb_chain, &qe_chain];
    if let Some(Ok((_, tc, _, qc))) = &early {
        chains.extend([tc.as_str(), qc.as_str()]);
    }
    tcb::verify_root_crl(quote, &chains, &root_crl, anchor, now_unix)?;
    Ok(judgement)
}

/// Intel's `early` tcbInfo and QE identity with their chains, or why not.
async fn fetch_early(
    pcs_base_url: &str,
    fmspc_hex: &str,
) -> Result<(String, String, String, String), String> {
    let (tb, tc) = fetch_pcs_collateral(&tcb::pcs_tcb_url(pcs_base_url, fmspc_hex))
        .await
        .map_err(|e| format!("early tcbInfo not fetched: {e:#}"))?;
    let (qb, qc) = fetch_pcs_collateral(&tcb::pcs_qe_identity_url(pcs_base_url))
        .await
        .map_err(|e| format!("early QE identity not fetched: {e:#}"))?;
    Ok((tb, tc, qb, qc))
}

/// [`judge_platform_live`], the two statuses the acceptance stands on.
pub async fn verify_platform_live(
    quote: &[u8],
    pcs_base_url: &str,
    root_crl_url: &str,
    anchor: &TrustAnchor,
    policy: &TcbPolicy,
    now_unix: u64,
) -> Result<(TcbStatus, TcbStatus), LiveFailure> {
    judge_platform_live(quote, pcs_base_url, root_crl_url, anchor, policy, now_unix)
        .await
        .map(|j| j.statuses())
}

/// One live check at the instant it is given (unix seconds), run by
/// [`revalidate`] at each tick: the collateral is judged fresh against that
/// instant, and the standing is dated by it.
pub type LiveCheck =
    Box<dyn Fn(u64) -> BoxFuture<'static, Result<LiveJudgement, LiveFailure>> + Send + Sync>;

/// Every `every`, read `clock()`, run `check` at that instant and apply its
/// outcome to `standing`, dated by the same instant: a slow fetch does not
/// date a verdict later than the time its collateral was judged against.
/// Log each outcome; return once the standing is lost. Spawn it next to the
/// server: it holds nothing the server waits on.
pub async fn revalidate(
    standing: Arc<PlatformStanding>,
    every: Duration,
    clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    check: LiveCheck,
) {
    loop {
        tokio::time::sleep(every).await;
        let now = clock();
        let outcome = check(now).await;
        let judgement = outcome.as_ref().ok().cloned();
        match standing.apply(now, outcome.map(|_| ())) {
            Applied::Renewed => {
                let said = judgement.map(|j| {
                    standing.disclose(&j, now);
                    let (p, q) = j.statuses();
                    format!(
                        "TCB {} / QE {} ({} collateral)",
                        p.as_str(),
                        q.as_str(),
                        j.accepted_under.as_str()
                    )
                });
                eprintln!(
                    "x0ne-infer: platform standing renewed against live Intel PCS: {}{}",
                    said.unwrap_or_default(),
                    standing
                        .disclosure()
                        .map(|d| format!("; {}", d.line))
                        .unwrap_or_default()
                )
            }
            Applied::Unchanged(why) => {
                eprintln!("x0ne-infer: platform standing not renewed, last verdict stands: {why}")
            }
            Applied::Lost(reason) => {
                eprintln!("x0ne-infer: platform standing lost, not answering: {reason}");
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attest_v2::fixtures::{
        fake_pcs, fake_pcs_serving, fresh, FakeIntel, STANDARD_EVALUATION,
    };

    const NOW: u64 = crate::gce_endorsement::tests::NOW;
    const PROD: TcbPolicy = TcbPolicy::production();
    const DAY: u64 = 86_400;

    fn quote(intel: &FakeIntel) -> Vec<u8> {
        intel.quote(&[0x5a; 64], &[7; 48], &[[0; 48]; 4], [0; 8])
    }

    /// A verdict in the platform's favour, as a check returns it.
    fn judged() -> LiveJudgement {
        let level = |status| tcb::Rating {
            status,
            tcb_date: "2026-08-12T00:00:00Z".into(),
            advisories: Vec::new(),
        };
        let rated = tcb::RatedTcb {
            tcb_evaluation: 22,
            qe_evaluation: 22,
            issued_unix: NOW,
            platform: level(TcbStatus::UpToDate),
            tdx_module: tcb::TdxModule {
                id: "tdxModule".into(),
                status: None,
                advisories: Vec::new(),
            },
            qe: level(TcbStatus::UpToDate),
        };
        LiveJudgement {
            mode: tcb::TcbMode::Strict,
            accepted_under: PcsUpdate::Early,
            latest: Ok(rated.clone()),
            accepted: rated,
        }
    }

    fn out_of_date() -> LiveFailure {
        LiveFailure::Refused(TcbRejected::StatusRefused {
            status: "OutOfDate".into(),
            fmspc: "00906ed50000".into(),
            level_date: "2026-08-11T00:00:00Z".into(),
        })
    }

    /// SECURITY: a verdict against the platform after start-up ends the
    /// answers, and nothing renews a lost standing.
    #[test]
    fn a_verdict_against_the_platform_loses_the_standing_for_good() {
        let s = PlatformStanding::verified_at(NOW, PROD);
        s.check(NOW + 3600).expect("standing");
        let applied = s.apply(NOW + 3600, Err(out_of_date()));
        assert!(
            matches!(&applied, Applied::Lost(r) if r.contains("OutOfDate")),
            "{applied:?}"
        );
        let e = s.check(NOW + 3601).unwrap_err();
        assert!(
            e.0.contains("OutOfDate") && e.0.contains("re-attest"),
            "{e}"
        );
        assert_eq!(s.verified_at_unix(), None);
        assert!(s.lost_reason().unwrap().contains("OutOfDate"));
        assert!(matches!(s.apply(NOW + 7200, Ok(())), Applied::Lost(_)));
        assert!(s.check(NOW + 7201).is_err());
    }

    /// A standing not renewed within the collateral age limit refuses; a
    /// renewal inside it dates the standing again.
    #[test]
    fn a_standing_older_than_the_collateral_age_limit_refuses_until_renewed() {
        let s = PlatformStanding::verified_at(NOW, PROD);
        let max = PROD.max_collateral_age_secs;
        s.check(NOW + max).expect("at the limit");
        let e = s.check(NOW + max + 1).unwrap_err();
        assert!(
            e.0.contains("not renewed") && e.0.contains("re-attest"),
            "{e}"
        );
        assert_eq!(s.apply(NOW + max + 1, Ok(())), Applied::Renewed);
        s.check(NOW + 2 * max).expect("renewed");
        assert_eq!(s.verified_at_unix(), Some(NOW + max + 1));
        // A wall clock behind the renewal reads as age zero here.
        s.check(NOW).expect("clock behind");
    }

    /// Collateral not fetched, not readable or too old to count says
    /// nothing about the platform: the standing keeps its date and ages out.
    #[test]
    fn collateral_not_fetched_or_not_readable_leaves_the_standing_to_age_out() {
        let s = PlatformStanding::verified_at(NOW, PROD);
        for f in [
            LiveFailure::Fetch("tcbInfo: connection refused".into()),
            LiveFailure::Refused(TcbRejected::Collateral("tcbInfo", "signature".into())),
            LiveFailure::Refused(TcbRejected::PckCrl("not a DER CRL".into())),
            LiveFailure::Refused(TcbRejected::RootCrl("no nextUpdate".into())),
            LiveFailure::Refused(TcbRejected::CollateralTooOld {
                what: "tcbInfo",
                age_secs: 90_000,
                max_secs: 86_400,
            }),
        ] {
            assert!(!f.is_a_verdict(), "{f}");
            let applied = s.apply(NOW + DAY - 1, Err(f));
            assert!(matches!(applied, Applied::Unchanged(_)), "{applied:?}");
        }
        assert_eq!(s.verified_at_unix(), Some(NOW));
        s.check(NOW + DAY).expect("inside the limit");
        assert!(s.check(NOW + DAY + 1).is_err());
    }

    /// Every other refusal is a verdict: a new variant fails closed.
    #[test]
    fn every_refusal_of_the_platform_itself_is_a_verdict() {
        for e in [
            out_of_date(),
            LiveFailure::Refused(TcbRejected::BelowEveryLevel),
            LiveFailure::Refused(TcbRejected::ModuleUnknown {
                id: "TDX_03".into(),
                version: 3,
            }),
            LiveFailure::Refused(TcbRejected::ModuleBelowEveryLevel {
                id: "TDX_01".into(),
                isvsvn: 2,
            }),
            LiveFailure::Refused(TcbRejected::PckRevoked {
                serial: "01".into(),
            }),
            LiveFailure::Refused(TcbRejected::CaRevoked {
                subject: "CN=Intel SGX PCK Platform CA".into(),
                serial: "02".into(),
            }),
            LiveFailure::Refused(TcbRejected::EvaluationBelowMinimum {
                what: "tcbInfo",
                have: 20,
                min: 22,
            }),
            LiveFailure::Refused(TcbRejected::Qe("MRSIGNER".into())),
            LiveFailure::Refused(TcbRejected::Quote("short".into())),
            LiveFailure::Refused(TcbRejected::PckExtension("missing".into())),
        ] {
            assert!(e.is_a_verdict(), "{e}");
        }
    }

    /// SECURITY: the live check on a fake PCS: fresh collateral renews; an
    /// `early` tcbInfo that now rates the platform OutOfDate is a verdict
    /// under the strict policy and a disclosure under the transparent one
    /// (its `standard` answer still UpToDate); a PCS that does not answer is
    /// not a verdict.
    #[tokio::test]
    async fn the_live_check_reads_intels_verdict_of_the_day() {
        let intel = FakeIntel::new();
        let q = quote(&intel);
        let anchor = intel.anchor();
        let strict = TcbPolicy::strict();
        let pcs = fake_pcs(&intel, Some(NOW)).await;
        for policy in [PROD, strict] {
            let (p, qe) =
                verify_platform_live(&q, &pcs.base_url, &pcs.root_crl_url, &anchor, &policy, NOW)
                    .await
                    .expect("fresh collateral");
            assert!(p.accepted() && qe.accepted());
        }

        let issued = fresh(NOW).0;
        let out_of_date =
            intel.tcb_info_with(issued, PROD.min_tcb_evaluation_data_number, "OutOfDate");
        let qe_bodies = (
            intel.qe_identity_with(issued, STANDARD_EVALUATION),
            intel.qe_identity(issued),
        );
        let stale = fake_pcs_serving(
            &intel,
            Some(NOW),
            Some(intel.pck_crl(&[], fresh(NOW))),
            (
                intel.tcb_info_with(issued, STANDARD_EVALUATION, "UpToDate"),
                out_of_date,
            ),
            qe_bodies,
        )
        .await;
        let e = verify_platform_live(
            &q,
            &stale.base_url,
            &stale.root_crl_url,
            &anchor,
            &strict,
            NOW,
        )
        .await
        .unwrap_err();
        assert!(e.is_a_verdict(), "{e}");
        assert!(e.to_string().contains("OutOfDate"), "{e}");
        let j = judge_platform_live(
            &q,
            &stale.base_url,
            &stale.root_crl_url,
            &anchor,
            &PROD,
            NOW,
        )
        .await
        .expect("transparent: accepted on the standard answer");
        assert_eq!(j.accepted_under, PcsUpdate::Standard);
        assert_eq!(j.accepted.tcb_evaluation, STANDARD_EVALUATION);
        assert!(j.accepted.accepted());
        let latest = j.latest.as_ref().expect("early read");
        assert_eq!(latest.platform.status, TcbStatus::OutOfDate);
        assert_eq!(latest.tcb_evaluation, PROD.min_tcb_evaluation_data_number);

        for policy in [PROD, strict] {
            let e = verify_platform_live(
                &q,
                "http://127.0.0.1:1",
                "http://127.0.0.1:1/root.der",
                &anchor,
                &policy,
                NOW,
            )
            .await
            .unwrap_err();
            assert!(
                matches!(&e, LiveFailure::Fetch(m) if m.starts_with("tcbInfo")),
                "{e}"
            );
            assert!(!e.is_a_verdict());
        }
    }

    /// A renewal whose `early` answer could not be read keeps the last view
    /// of it, dated, with the failure named: never a fresh "up to date".
    #[test]
    fn a_renewal_without_intels_latest_keeps_the_last_view_and_says_so() {
        let s = PlatformStanding::verified_at(NOW, PROD);
        let mut j = judged();
        j.mode = tcb::TcbMode::Transparent;
        j.accepted_under = PcsUpdate::Standard;
        s.disclose(&j, NOW);
        let first = s.disclosure().unwrap();
        assert_eq!(
            first
                .intel
                .as_ref()
                .unwrap()
                .latest
                .as_ref()
                .unwrap()
                .judged_at_unix,
            NOW
        );
        j.latest = Err("early tcbInfo not fetched: timed out".into());
        s.disclose(&j, NOW + 3600);
        let d = s.disclosure().unwrap();
        let intel = d.intel.as_ref().unwrap();
        assert_eq!(intel.latest.as_ref().unwrap().judged_at_unix, NOW);
        assert_eq!(
            intel.latest_unavailable.as_deref(),
            Some("early tcbInfo not fetched: timed out")
        );
        assert!(d.line.contains("not renewed since"), "{}", d.line);
        assert!(d.line.contains("timed out"), "{}", d.line);
        assert_eq!(d.as_of_unix, NOW + 3600);
    }

    /// The loop: a renewal, then a verdict against the platform; it stops
    /// once the standing is lost and applies nothing after.
    #[tokio::test]
    async fn the_loop_renews_then_stops_on_a_verdict_against_the_platform() {
        use std::sync::atomic::{AtomicU64, Ordering::SeqCst};
        let standing = Arc::new(PlatformStanding::verified_at(NOW, PROD));
        let calls = Arc::new(AtomicU64::new(0));
        let clock = Arc::new(AtomicU64::new(NOW));
        let check: LiveCheck = {
            let calls = calls.clone();
            Box::new(move |_now: u64| {
                let n = calls.fetch_add(1, SeqCst);
                Box::pin(async move {
                    match n {
                        0 => Ok(judged()),
                        1 => Err(LiveFailure::Fetch("tcbInfo: timed out".into())),
                        _ => Err(out_of_date()),
                    }
                })
            })
        };
        let c = clock.clone();
        let tick = Arc::new(move || c.fetch_add(1000, SeqCst) + 1000);
        let run = tokio::spawn(revalidate(
            standing.clone(),
            Duration::from_millis(5),
            tick,
            check,
        ));
        tokio::time::timeout(Duration::from_secs(10), run)
            .await
            .expect("the loop returns once the standing is lost")
            .unwrap();
        assert_eq!(calls.load(SeqCst), 3);
        assert!(standing.lost_reason().unwrap().contains("OutOfDate"));
        assert!(standing.check(NOW + 3000).is_err());
    }

    /// SECURITY: a TDX module Intel publishes no accepted level for, or no
    /// identity at all, is a verdict against the platform, as
    /// `BelowEveryLevel` is for its TCB: the standing is lost within the
    /// hour, not left to age out for a day.
    #[test]
    fn a_tdx_module_intel_lists_no_level_or_identity_for_is_a_verdict() {
        let intel = FakeIntel::new();
        let mut q = quote(&intel);
        q[48] = 10; // TEE_TCB_SVN[0]: module ISVSVN 10
        q[49] = 1; // TEE_TCB_SVN[1]: module identity TDX_01
        let issued = fresh(NOW).0;
        let chain = intel.collateral_chain_pem();
        let qe = intel.qe_identity(issued);
        let anchor = intel.anchor();
        let live =
            |body: &str| tcb::verify_live(&q, body, &chain, &qe, &chain, &anchor, NOW, &PROD);
        live(intel.tcb_info_module(issued, 10).as_str())
            .expect("control: a level asking ISVSVN 10");
        let e = live(intel.tcb_info_module(issued, 15).as_str()).unwrap_err();
        assert!(
            matches!(&e, TcbRejected::ModuleBelowEveryLevel { id, isvsvn: 10 } if id == "TDX_01"),
            "{e}"
        );
        assert!(LiveFailure::Refused(e).is_a_verdict());
        let e = live(intel.tcb_info(issued).as_str()).unwrap_err();
        assert!(
            matches!(&e, TcbRejected::ModuleUnknown { id, version: 1 } if id == "TDX_01"),
            "{e}"
        );
        let s = PlatformStanding::verified_at(NOW, PROD);
        assert!(matches!(
            s.apply(NOW + 3600, Err(LiveFailure::Refused(e))),
            Applied::Lost(r) if r.contains("no identity")
        ));
    }

    /// The standing is dated at the instant the collateral was judged
    /// against, which the check is given, not when a slow fetch returns.
    #[tokio::test]
    async fn the_standing_is_dated_at_the_instant_the_collateral_was_judged() {
        use std::sync::atomic::{AtomicU64, Ordering::SeqCst};
        let standing = Arc::new(PlatformStanding::verified_at(NOW - 3600, PROD));
        let clock = Arc::new(AtomicU64::new(NOW));
        let judged_at = Arc::new(AtomicU64::new(0));
        let calls = Arc::new(AtomicU64::new(0));
        let check: LiveCheck = {
            let (clock, judged_at, calls) = (clock.clone(), judged_at.clone(), calls.clone());
            Box::new(move |now: u64| {
                let n = calls.fetch_add(1, SeqCst);
                if n == 0 {
                    judged_at.store(now, SeqCst);
                    // The fetch takes two days of wall clock.
                    clock.fetch_add(2 * DAY, SeqCst);
                }
                Box::pin(async move {
                    if n == 0 {
                        Ok(judged())
                    } else {
                        Err(LiveFailure::Fetch("tcbInfo: timed out".into()))
                    }
                })
            })
        };
        let c = clock.clone();
        let run = tokio::spawn(revalidate(
            standing.clone(),
            Duration::from_millis(5),
            Arc::new(move || c.load(SeqCst)),
            check,
        ));
        while calls.load(SeqCst) < 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        run.abort();
        assert_eq!(judged_at.load(SeqCst), NOW);
        assert_eq!(standing.verified_at_unix(), Some(NOW));
    }
}
