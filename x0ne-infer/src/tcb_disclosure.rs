// SPDX-License-Identifier: Apache-2.0
//! What a verdict says about the platform's patch level, beside the verdict.
//!
//! Decision of 2026-10-28: a verdict accepts a TDX
//! platform on Intel's `standard` collateral, as the market does, and names
//! what Intel's latest (`early`) collateral rates not fixed: the platform's
//! TCB date and the advisories listed against its level. On SEV-SNP, a
//! bulletin is imposed once a year old and named before. The client may ask
//! for `strict`, which refuses instead ([`crate::tcb::TcbMode`]).
//!
//! A [`TcbDisclosure`](crate::tcb_disclosure::TcbDisclosure) is that statement, versioned
//! ([`SCHEMA`](crate::tcb_disclosure::SCHEMA)): in the
//! attestation document, in the relay's verdict and `/x0ne/v1/status`, in
//! every signed receipt. It discloses; it protects against nothing it names.
//! A disclosure that could not read Intel's latest collateral says so and
//! never says "up to date".

use serde::{Deserialize, Serialize};

#[cfg(not(x0ne_sdk))]
use crate::snp::{BulletinStanding, VerifiedSnp};
use crate::tcb::{LiveJudgement, RatedTcb, TcbMode};

/// The version of the disclosure's JSON form. A reader refuses another.
pub const SCHEMA: &str = "x0ne-infer/tcb-disclosure/v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TcbDisclosure {
    pub schema: String,
    /// The TCB policy the verdict applied: `transparent` or `strict`.
    pub policy: String,
    /// When the verdict was reached (Unix seconds).
    pub as_of_unix: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intel: Option<IntelTcb>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub amd: Option<AmdTcb>,
    /// The sentence a client prints after the verdict.
    pub line: String,
}

/// Intel TDX: what the acceptance stands on, and Intel's latest view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntelTcb {
    /// The PCS answer the acceptance stands on: `standard` or `early`.
    pub accepted_under: String,
    pub accepted_evaluation: u64,
    pub platform: String,
    pub platform_tcb_date: String,
    pub qe: String,
    /// The policy's floor on the latest collateral, and how long a view of
    /// it counts after it was judged.
    pub latest_floor_evaluation: u64,
    pub latest_age_limit_secs: u64,
    /// Intel's latest collateral as last read; `None` if never read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest: Option<IntelLatest>,
    /// Why the last attempt to read it failed, if it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest_unavailable: Option<String>,
}

/// Intel's latest (`early`) tcbInfo and QE identity, rated for the platform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntelLatest {
    /// When this view was judged (Unix seconds).
    pub judged_at_unix: u64,
    pub tcb_evaluation: u64,
    pub qe_evaluation: u64,
    pub issued_unix: u64,
    pub platform: String,
    pub platform_tcb_date: String,
    /// `TDX_<nn>`, or `tdxModule` for a version-0 module, rated through
    /// the platform levels (no status of its own).
    pub tdx_module: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tdx_module_status: Option<String>,
    pub qe: String,
    /// The advisories Intel lists against the platform's level, the TDX
    /// module's and the QE's, in that order, each once.
    pub advisories: Vec<String>,
}

/// AMD SEV-SNP: the report's lowest TCB and each bulletin of its CPU line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AmdTcb {
    pub product: String,
    pub cpu_line: String,
    /// The component-wise lowest of the report's four TCBs.
    pub reported: String,
    pub bulletins: Vec<AmdBulletin>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AmdBulletin {
    pub id: String,
    pub published: String,
    /// Imposed by the verdict (a report below it is refused), or only named.
    pub imposed: bool,
    /// What the bulletin asks of this CPU line.
    pub expected: String,
    /// What the report misses of it; empty when met.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub not_met: Vec<String>,
}

/// `2026-02-11T00:00:00Z` → `2026-02-11`.
fn day(tcb_date: &str) -> &str {
    tcb_date.get(..10).unwrap_or(tcb_date)
}

fn latest_of(r: &RatedTcb, judged_at_unix: u64) -> IntelLatest {
    IntelLatest {
        judged_at_unix,
        tcb_evaluation: r.tcb_evaluation,
        qe_evaluation: r.qe_evaluation,
        issued_unix: r.issued_unix,
        platform: r.platform.status.as_str().to_string(),
        platform_tcb_date: r.platform.tcb_date.clone(),
        tdx_module: r.tdx_module.id.clone(),
        tdx_module_status: r.tdx_module.status.as_ref().map(|s| s.as_str().to_string()),
        qe: r.qe.status.as_str().to_string(),
        advisories: r.advisories(),
    }
}

/// [`crate::tcb::TcbStatus::accepted`], on the status as Intel spells it.
fn accepted_status(s: &str) -> bool {
    matches!(s, "UpToDate" | "SWHardeningNeeded")
}

impl IntelLatest {
    /// Platform, module and QE all statuses a verdict accepts.
    pub fn accepted(&self) -> bool {
        accepted_status(&self.platform)
            && accepted_status(&self.qe)
            && self
                .tdx_module_status
                .as_deref()
                .is_none_or(accepted_status)
    }
}

impl TcbDisclosure {
    /// A TDX verdict's disclosure, at `now_unix`.
    pub fn intel(j: &LiveJudgement, floor: u64, age_limit_secs: u64, now_unix: u64) -> Self {
        let intel = IntelTcb {
            accepted_under: j.accepted_under.as_str().to_string(),
            accepted_evaluation: j.accepted.tcb_evaluation,
            platform: j.accepted.platform.status.as_str().to_string(),
            platform_tcb_date: j.accepted.platform.tcb_date.clone(),
            qe: j.accepted.qe.status.as_str().to_string(),
            latest_floor_evaluation: floor,
            latest_age_limit_secs: age_limit_secs,
            latest: j.latest.as_ref().ok().map(|r| latest_of(r, now_unix)),
            latest_unavailable: j.latest.as_ref().err().cloned(),
        };
        Self::of_intel(j.mode, intel, now_unix)
    }

    fn of_intel(mode: TcbMode, intel: IntelTcb, now_unix: u64) -> Self {
        let line = intel_line(mode, &intel);
        Self {
            schema: SCHEMA.into(),
            policy: mode.as_str().into(),
            as_of_unix: now_unix,
            intel: Some(intel),
            amd: None,
            line,
        }
    }

    /// A re-check's disclosure: `j` at `now_unix`, keeping `previous`'s
    /// latest view when this check could not read Intel's latest collateral.
    /// The view keeps the time it was judged; the failure is named.
    pub fn renewed(
        previous: Option<&TcbDisclosure>,
        j: &LiveJudgement,
        floor: u64,
        age_limit_secs: u64,
        now_unix: u64,
    ) -> Self {
        let mut next = Self::intel(j, floor, age_limit_secs, now_unix);
        let kept = previous
            .and_then(|p| p.intel.as_ref())
            .and_then(|i| i.latest.clone());
        if let (Some(intel), Some(kept)) = (next.intel.as_mut(), kept) {
            if intel.latest.is_none() {
                intel.latest = Some(kept);
                next = Self::of_intel(j.mode, intel.clone(), now_unix);
            }
        }
        next
    }

    /// An SEV-SNP verdict's disclosure, at `now_unix`.
    #[cfg(not(x0ne_sdk))]
    pub fn amd(v: &VerifiedSnp, now_unix: u64) -> Self {
        let lowest = crate::snp::lowest_tcb(&v.report, v.product);
        let reported = format!(
            "boot loader {:#04x}, TEE {:#04x}, SNP {:#04x}, microcode {:#04x}, FMC {:#04x}",
            lowest.boot_loader, lowest.tee, lowest.snp, lowest.microcode, lowest.fmc
        );
        let amd = AmdTcb {
            product: v.product.name().to_string(),
            cpu_line: v
                .cpu_line
                .clone()
                .unwrap_or_else(|| "explicit minimum".into()),
            reported,
            bulletins: v.bulletins.iter().map(bulletin_of).collect(),
        };
        let line = amd_line(v.tcb_mode, &amd, &v.minimum_source);
        Self {
            schema: SCHEMA.into(),
            policy: v.tcb_mode.as_str().into(),
            as_of_unix: now_unix,
            intel: None,
            amd: Some(amd),
            line,
        }
    }
}

#[cfg(not(x0ne_sdk))]
fn expected_text(m: crate::snp::SnpTcbMinimum) -> String {
    let mut parts = Vec::new();
    for (name, v) in [
        ("boot loader", m.boot_loader),
        ("TEE", m.tee),
        ("SNP", m.snp),
        ("microcode", m.microcode),
        ("FMC", m.fmc),
    ] {
        if v != 0 {
            parts.push(format!("{name} {v:#04x}"));
        }
    }
    if m.require_alias_check {
        parts.push("alias check complete".into());
    }
    parts.join(", ")
}

#[cfg(not(x0ne_sdk))]
fn bulletin_of(b: &BulletinStanding) -> AmdBulletin {
    let mut expected = expected_text(b.minimum);
    if let Some(note) = b.note {
        expected.push_str(&format!(" ({note})"));
    }
    AmdBulletin {
        id: b.bulletin.id.into(),
        published: b.bulletin.published.into(),
        imposed: b.imposed,
        expected,
        not_met: b.not_met.clone(),
    }
}

fn advisories_text(status_ok: bool, ids: &[String]) -> String {
    if ids.is_empty() {
        "Intel lists no advisory against it".into()
    } else if status_ok {
        format!(
            "Intel advisories listed against it (software hardening): {}",
            ids.join(", ")
        )
    } else {
        format!("Intel advisories not fixed: {}", ids.join(", "))
    }
}

/// The sentence after VERIFIED. Intel's statuses are named as Intel writes
/// them; nothing here says the platform carries every fix unless Intel's
/// latest collateral lists no advisory against it, and then it says only that.
fn intel_line(mode: TcbMode, i: &IntelTcb) -> String {
    let policy = format!("{} TCB policy", mode.as_str());
    match &i.latest {
        Some(l) => {
            let mut out = format!(
                "Platform TCB {}, {} under Intel's TCB evaluation {}; {}",
                day(&l.platform_tcb_date),
                l.platform,
                l.tcb_evaluation,
                advisories_text(l.accepted(), &l.advisories)
            );
            if let Some(why) = &i.latest_unavailable {
                out.push_str(&format!(
                    " (as read at {}; not renewed since: {why})",
                    l.judged_at_unix
                ));
            }
            format!("{out} ({policy})")
        }
        None => format!(
            "Platform TCB {}, {} under Intel's {} TCB evaluation {}; Intel's latest TCB \
             evaluation could not be read ({}): no advisory is named and nothing here says \
             the platform is up to date ({policy})",
            day(&i.platform_tcb_date),
            i.platform,
            i.accepted_under,
            i.accepted_evaluation,
            i.latest_unavailable
                .as_deref()
                .unwrap_or("no reason recorded")
        ),
    }
}

#[cfg(not(x0ne_sdk))]
fn amd_line(mode: TcbMode, a: &AmdTcb, source: &str) -> String {
    let policy = format!("{} TCB policy", mode.as_str());
    if a.bulletins.is_empty() {
        return format!(
            "AMD {} ({}), lowest reported TCB {}; held to {source} ({policy})",
            a.product, a.cpu_line, a.reported
        );
    }
    let imposed: Vec<&str> = a
        .bulletins
        .iter()
        .filter(|b| b.imposed)
        .map(|b| b.id.as_str())
        .collect();
    let mut out = format!(
        "AMD {} ({}), lowest reported TCB {}; bulletins imposed: {}",
        a.product,
        a.cpu_line,
        a.reported,
        if imposed.is_empty() {
            "none".to_string()
        } else {
            imposed.join(", ")
        }
    );
    let named: Vec<String> = a
        .bulletins
        .iter()
        .filter(|b| !b.imposed)
        .map(|b| {
            if b.not_met.is_empty() {
                format!("{} met", b.id)
            } else {
                format!("{} not met: {}", b.id, b.not_met.join(", "))
            }
        })
        .collect();
    if !named.is_empty() {
        out.push_str(&format!(
            "; named, not imposed (under a year old): {}",
            named.join("; ")
        ));
    }
    format!("{out} ({policy})")
}

/// How every refusal of [`strict_receipt_check`] begins.
pub const STRICT_REFUSAL: &str = "strict TCB policy: ";

/// A strict relay's check of one receipt: the TD's latest view of Intel's
/// collateral, in the receipt, must be present, accepted, at the relay's
/// floor and judged within its age limit. Before 2026-10-28 the TD itself
/// stopped answering on such a view; a TD now serves under `transparent`,
/// and a strict client stops instead.
pub fn strict_receipt_check(
    relay: &TcbDisclosure,
    receipt: Option<&TcbDisclosure>,
    now_unix: u64,
) -> Result<(), String> {
    let Some(want) = relay.intel.as_ref() else {
        return Ok(());
    };
    let refuse = |why: String| Err(format!("{STRICT_REFUSAL}{why}"));
    let Some(d) = receipt else {
        return refuse("the receipt carries no TCB disclosure".into());
    };
    if d.schema != SCHEMA {
        return refuse(format!("the receipt's TCB disclosure is {:?}", d.schema));
    }
    let Some(latest) = d.intel.as_ref().and_then(|i| i.latest.as_ref()) else {
        return refuse("the receipt names no view of Intel's latest TCB collateral".into());
    };
    if !latest.accepted() {
        return refuse(format!(
            "Intel's latest collateral rates the platform {}, its TDX module {} {}, its QE {}",
            latest.platform,
            latest.tdx_module,
            latest.tdx_module_status.as_deref().unwrap_or("-"),
            latest.qe
        ));
    }
    let floor = want.latest_floor_evaluation;
    if latest.tcb_evaluation < floor || latest.qe_evaluation < floor {
        return refuse(format!(
            "the receipt's view is Intel's TCB evaluation {}/{}, below {floor}",
            latest.tcb_evaluation, latest.qe_evaluation
        ));
    }
    let age = now_unix.saturating_sub(latest.judged_at_unix);
    if age > want.latest_age_limit_secs {
        return refuse(format!(
            "the receipt's view of Intel's latest collateral is {age} s old, past {} s",
            want.latest_age_limit_secs
        ));
    }
    Ok(())
}

#[cfg(all(test, not(x0ne_sdk)))]
mod tests {
    use super::*;

    /// The GCE hosts of 2026-10-28 (FMSPC 00806f050000), as a transparent
    /// TD discloses them: accepted on `standard`, named on `early`.
    fn host_2026_10_28(latest: bool) -> IntelTcb {
        IntelTcb {
            accepted_under: "standard".into(),
            accepted_evaluation: 20,
            platform: "UpToDate".into(),
            platform_tcb_date: "2025-08-13T00:00:00Z".into(),
            qe: "UpToDate".into(),
            latest_floor_evaluation: 22,
            latest_age_limit_secs: 86_400,
            latest: latest.then(|| IntelLatest {
                judged_at_unix: 1_000,
                tcb_evaluation: 22,
                qe_evaluation: 22,
                issued_unix: 900,
                platform: "OutOfDate".into(),
                platform_tcb_date: "2026-02-11T00:00:00Z".into(),
                tdx_module: "TDX_03".into(),
                tdx_module_status: Some("UpToDate".into()),
                qe: "UpToDate".into(),
                advisories: ["INTEL-DOC-10003", "INTEL-SA-01439", "INTEL-SA-01442"]
                    .map(String::from)
                    .to_vec(),
            }),
            latest_unavailable: (!latest).then(|| "early tcbInfo not fetched: HTTP 503".into()),
        }
    }

    #[test]
    fn the_line_names_the_advisories_or_says_it_could_not_read_them() {
        let named = TcbDisclosure::of_intel(TcbMode::Transparent, host_2026_10_28(true), 1_000);
        assert_eq!(
            named.line,
            "Platform TCB 2026-02-11, OutOfDate under Intel's TCB evaluation 22; Intel advisories \
             not fixed: INTEL-DOC-10003, INTEL-SA-01439, INTEL-SA-01442 (transparent TCB policy)"
        );
        let unread = TcbDisclosure::of_intel(TcbMode::Transparent, host_2026_10_28(false), 1_000);
        assert_eq!(
            unread.line,
            "Platform TCB 2025-08-13, UpToDate under Intel's standard TCB evaluation 20; Intel's \
             latest TCB evaluation could not be read (early tcbInfo not fetched: HTTP 503): no \
             advisory is named and nothing here says the platform is up to date (transparent TCB \
             policy)"
        );
        // A view kept from an earlier check says when it was read.
        let mut kept = host_2026_10_28(true);
        kept.latest_unavailable = Some("early tcbInfo not fetched: HTTP 503".into());
        let kept = TcbDisclosure::of_intel(TcbMode::Transparent, kept, 5_000);
        assert!(
            kept.line
                .contains("(as read at 1000; not renewed since: early tcbInfo not fetched"),
            "{}",
            kept.line
        );
        // SWHardeningNeeded is accepted; its advisories are not called "not fixed".
        let mut hardening = host_2026_10_28(true);
        hardening.latest.as_mut().unwrap().platform = "SWHardeningNeeded".into();
        let line = TcbDisclosure::of_intel(TcbMode::Strict, hardening, 1_000).line;
        assert!(line
            .contains("Intel advisories listed against it (software hardening): INTEL-DOC-10003"));
        assert!(line.ends_with("(strict TCB policy)"));
        // No line says "up to date" in lower case, whatever it reports.
        for d in [&named, &unread, &kept] {
            assert!(
                !d.line.replace("is up to date", "").contains("up to date"),
                "{}",
                d.line
            );
        }
    }

    /// SECURITY (2026-10-28): a strict relay refuses a receipt whose TD served on
    /// a view it would not accept, or on no view, or on a stale one.
    #[test]
    fn a_strict_relay_refuses_each_receipt_its_policy_would_not_accept() {
        let strict = TcbDisclosure::of_intel(TcbMode::Strict, host_2026_10_28(true), 1_000);
        let receipt = |intel: IntelTcb| TcbDisclosure::of_intel(TcbMode::Transparent, intel, 1_000);
        let refused =
            |r: Option<&TcbDisclosure>, now| strict_receipt_check(&strict, r, now).unwrap_err();
        assert!(refused(None, 1_000).starts_with(STRICT_REFUSAL));
        assert_eq!(
            refused(Some(&receipt(host_2026_10_28(true))), 1_000),
            "strict TCB policy: Intel's latest collateral rates the platform OutOfDate, its TDX \
             module TDX_03 UpToDate, its QE UpToDate"
        );
        assert!(refused(Some(&receipt(host_2026_10_28(false))), 1_000).contains("names no view"));
        let mut fixed = host_2026_10_28(true);
        let latest = fixed.latest.as_mut().unwrap();
        latest.platform = "UpToDate".into();
        latest.advisories.clear();
        assert_eq!(
            strict_receipt_check(&strict, Some(&receipt(fixed.clone())), 1_000),
            Ok(())
        );
        assert!(refused(Some(&receipt(fixed.clone())), 1_000 + 86_401).contains("86401 s old"));
        let mut low = fixed.clone();
        low.latest.as_mut().unwrap().qe_evaluation = 21;
        assert!(refused(Some(&receipt(low)), 1_000).contains("evaluation 22/21, below 22"));
        let mut other = receipt(fixed);
        other.schema = "x0ne-infer/tcb-disclosure/v2".into();
        assert!(refused(Some(&other), 1_000).contains("tcb-disclosure/v2"));
        // A relay with no Intel view (SEV-SNP) has nothing to hold a receipt to.
        let snp = TcbDisclosure {
            intel: None,
            ..strict.clone()
        };
        assert_eq!(strict_receipt_check(&snp, None, 1_000), Ok(()));
    }

    /// The JSON form is closed: an unknown field is refused, so a reader
    /// never drops what it does not understand.
    #[test]
    fn the_disclosure_round_trips_and_refuses_unknown_fields() {
        let d = TcbDisclosure::of_intel(TcbMode::Transparent, host_2026_10_28(true), 1_000);
        let json = serde_json::to_value(&d).unwrap();
        assert_eq!(json["schema"], SCHEMA);
        assert_eq!(json["intel"]["latest"]["advisories"][1], "INTEL-SA-01439");
        assert!(json.get("amd").is_none());
        assert_eq!(
            serde_json::from_value::<TcbDisclosure>(json.clone()).unwrap(),
            d
        );
        let mut extra = json;
        extra["intel"]["patched"] = true.into();
        assert!(serde_json::from_value::<TcbDisclosure>(extra).is_err());
    }
}
