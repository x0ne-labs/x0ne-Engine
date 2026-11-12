// SPDX-License-Identifier: Apache-2.0
//! The LIVE half of the CPU proof — Intel PCS TCB status and QE identity.
//!
//! `pcs.rs` proves the quote's certificate chain reaches Intel's pinned
//! root. What it cannot prove alone is that the platform is CURRENT: a CPU
//! with a revoked PCK or a hole-ridden microcode carries a perfectly valid
//! chain. This module closes that (a design decision): the verdict is
//! measured against what Intel publishes TODAY —
//!
//! 1. `tcbInfo` (TDX, v3) for the platform's FMSPC — fetched from PCS,
//!    signed by Intel's TCB-Signing certificate, whose chain ends on the
//!    SAME pinned root as the PCK chain;
//! 2. the platform's TCB level, matched by CPUSVN components ‖ PCESVN
//!    (from the PCK certificate's SGX extension) and by the quote's
//!    `tee_tcb_svn` against `tdxtcbcomponents`;
//! 3. the QE identity (TD_QE) — MRSIGNER, ISVPRODID, MISCSELECT and
//!    ATTRIBUTES under their masks, ISVSVN against the published levels.
//!
//! Statuses accepted: [`TcbStatus::accepted`]. Everything else — `OutOfDate`,
//! `Revoked`, configuration-needed variants, unknown strings — refuses.
//! The vendored sample quote in `tests/pcs_live/` is genuinely stale against
//! today's tcbInfo (its CPUSVN is below every level Intel publishes for its
//! FMSPC), and the tests assert that refusal on real data.
//!
//! 4. the TDX module ([`tdx_module_status`]): MRSIGNERSEAM and
//!    SEAMATTRIBUTES against the identity `TDX_<TEE_TCB_SVN[1]>`, its
//!    ISVSVN (`TEE_TCB_SVN[0]`) against that identity's levels.
//! 5. the verifier's own floor ([`TcbPolicy`]): tcbInfo and QE identity of
//!    at least a pinned TCB evaluation (asked with `update=early`), and every
//!    piece of collateral, PCK CRL included, issued recently enough.
//! 6. the policy's mode ([`TcbMode`], since 2026-10-28): `Transparent`
//!    accepts on Intel's `standard` collateral and names what the `early`
//!    one rates not fixed ([`judge_live`]); `Strict` accepts only what the
//!    `early` collateral accepts, as every verdict did before.

use crate::pcs::{
    intel_shape, p256_key_from_cert, parse_sig_section, root_matches, verified_pck_chain,
    verify_cert_signed_by, TrustAnchor,
};
use p256::ecdsa::signature::Verifier as _;
use p256::ecdsa::Signature;
use x509_parser::pem::Pem;
use x509_parser::prelude::{FromDer, X509Certificate};
use x509_parser::revocation_list::CertificateRevocationList;

/// What the live path DOES establish about the collateral, stated so a
/// reader need not infer it: the signer is Intel's TCB signing certificate
/// by subject, is an end entity, is marked for signing, and every
/// certificate of its chain is inside its own validity window.
pub const LIVE_LIMITATIONS: &[&str] = &[
    // The PCK leaf is checked against Intel's PCK CRL (`verify_pck_crl`, fetched
    // with the collateral); the PCK CA and the TCB signing certificate against
    // Intel's root CA CRL (`verify_root_crl`), issued for a year.
    "PCK revocation: the PCK certificate is checked against Intel's PCK CRL fetched with this collateral; the PCK CA and the TCB signing certificate against Intel's root CA CRL, judged by its own window (issued for a year)",
    // The statuses were a STARTUP measurement
    // asserted for the process lifetime — a platform Intel revoked after
    // boot kept serving until restart. Since 2026-10-23 `tcb_standing`
    // re-reads the collateral every hour and a verdict against the platform
    // stops the answers. The verifier string self-dates (`as of <unix>`), so
    // a three-week-old receipt says when it was true.
    "collateral fetched at process start, then re-read every hour against the same quote; a verdict against the platform stops the answers, as does one not renewed within the collateral age limit; the statuses in a receipt are as of the last verdict, not re-verified per request",
];

/// The verifier's own floor under Intel's verdict, and how old that verdict
/// may be. Intel's status is relative to the collateral that carries it: PCS
/// serves by default the `standard` tcbInfo, updated after a TCB recovery
/// "commonly 12 months after public disclosure" (Intel PCS API, `update`).
/// Measured on 2026-10-22: `standard` was TCB evaluation 20 (recovery of
/// 2025-08-12) and rated TDX module `TDX_01` ISVSVN 11 UpToDate; `early` was
/// evaluation 22 (recovery of 2026-08-11) and rated the same module OutOfDate
/// under INTEL-SA-01314, -01397, -01419 and -01436. The floor is set here, in
/// the verifier's source, as Meta's fix for TOB-WAPI-8 (Trail of Bits review
/// of WhatsApp Private Processing) sets minimum SEV-SNP patch levels in the
/// client's source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TcbPolicy {
    /// Which collateral the acceptance stands on ([`TcbMode`]).
    pub mode: TcbMode,
    /// Intel's latest (`early`) tcbInfo and QE identity must carry at least
    /// this `tcbEvaluationDataNumber`: to accept (`Strict`), to be named in
    /// the verdict (`Transparent`).
    pub min_tcb_evaluation_data_number: u64,
    /// tcbInfo and QE identity `issueDate`, PCK CRL `thisUpdate`: at most
    /// this many seconds before `now`, whatever their `nextUpdate` says.
    pub max_collateral_age_secs: u64,
}

impl TcbPolicy {
    /// TCB recovery of 2026-08-11 (Intel PCS `tcbevaluationdatanumbers`,
    /// read 2026-10-22). Google's bulletin GCP-2026-053 of the same day
    /// names six Intel advisories against TDX firmware (INTEL-SA-01404,
    /// -01419, -01428, -01436, -01439, -01442). In this evaluation's
    /// tcbInfo for the GCE a3-highgpu FMSPC `00806f050000`, TDX_01 below
    /// ISVSVN 15 is OutOfDate under SA-01419 and SA-01436 (TDX module 1.5.28
    /// and earlier) and the platform levels name SA-01439 and SA-01442;
    /// evaluation 20 rated TDX_01 ISVSVN 11 and up UpToDate.
    pub const MIN_TCB_EVALUATION_DATA_NUMBER: u64 = 22;
    /// One day. Measured on 2026-10-22: Intel's tcbInfo, QE identity and PCK
    /// CRL were issued 18 to 68 minutes before they were fetched; their
    /// `nextUpdate` is 30 days later. A cache that serves month-old collateral
    /// hides a revocation or a recovery for that month; this bounds it to a day.
    pub const MAX_COLLATERAL_AGE_SECS: u64 = 24 * 3600;

    /// The floor on Intel's `standard` tcbInfo and QE identity when a
    /// `Transparent` verdict accepts on them. Measured on 2026-10-28 for
    /// FMSPC 00806f050000: `standard` was evaluation 20, issued that day.
    /// Intel moves `standard` up about 12 months after `early`.
    pub const MIN_STANDARD_TCB_EVALUATION_DATA_NUMBER: u64 = 20;

    /// The values above, `Transparent`: what every production path verifies
    /// with unless the client asks for `Strict` (decision of 2026-10-28).
    pub const fn production() -> Self {
        Self {
            mode: TcbMode::Transparent,
            min_tcb_evaluation_data_number: Self::MIN_TCB_EVALUATION_DATA_NUMBER,
            max_collateral_age_secs: Self::MAX_COLLATERAL_AGE_SECS,
        }
    }

    /// The production values under `Strict`: the policy of every verdict
    /// before 2026-10-28.
    pub const fn strict() -> Self {
        Self {
            mode: TcbMode::Strict,
            ..Self::production()
        }
    }

    /// The same floor and age under `mode`. Both modes are at least as
    /// strict as [`TcbPolicy::production`], so this never loosens a policy
    /// built from it, except `Strict` back to `Transparent`: callers that
    /// must not loosen compare with [`TcbPolicy::at_least`].
    #[must_use]
    pub const fn with_mode(self, mode: TcbMode) -> Self {
        Self { mode, ..self }
    }

    /// True when `self` judges at least as strictly as `other`: a mode at
    /// least as strict, a floor at least as high, an age limit at most as long.
    pub fn at_least(&self, other: &Self) -> bool {
        self.mode >= other.mode
            && self.min_tcb_evaluation_data_number >= other.min_tcb_evaluation_data_number
            && self.max_collateral_age_secs <= other.max_collateral_age_secs
    }

    /// Which PCS answer the acceptance stands on.
    pub fn acceptance_update(&self) -> PcsUpdate {
        match self.mode {
            TcbMode::Strict => PcsUpdate::Early,
            TcbMode::Transparent => PcsUpdate::Standard,
        }
    }

    /// The policy `standard` collateral is accepted under: its own floor,
    /// the same age limit.
    fn on_standard(&self) -> Self {
        Self {
            min_tcb_evaluation_data_number: Self::MIN_STANDARD_TCB_EVALUATION_DATA_NUMBER,
            ..*self
        }
    }

    /// The production policy made stricter: a higher minimum evaluation (a
    /// recovery newer than this build), a shorter maximum age. A looser value
    /// is refused; loosening takes a new build, where it is reviewed. A
    /// minimum given is a floor to refuse on: the mode is then `Strict`, as
    /// before 2026-10-28; otherwise the production one
    /// ([`TcbPolicy::with_mode`] sets another).
    pub fn tightened(
        min_tcb_evaluation_data_number: Option<u64>,
        max_collateral_age_secs: Option<u64>,
    ) -> Result<Self, String> {
        let base = Self::production();
        let min = min_tcb_evaluation_data_number.unwrap_or(base.min_tcb_evaluation_data_number);
        if min < base.min_tcb_evaluation_data_number {
            return Err(format!(
                "minimum TCB evaluation {min} is below this build's {}: it can be raised, not lowered",
                base.min_tcb_evaluation_data_number
            ));
        }
        let max = max_collateral_age_secs.unwrap_or(base.max_collateral_age_secs);
        if max > base.max_collateral_age_secs {
            return Err(format!(
                "maximum collateral age {max} s is above this build's {} s: it can be shortened, not lengthened",
                base.max_collateral_age_secs
            ));
        }
        Ok(Self {
            mode: match min_tcb_evaluation_data_number {
                Some(_) => TcbMode::Strict,
                None => base.mode,
            },
            min_tcb_evaluation_data_number: min,
            max_collateral_age_secs: max,
        })
    }

    /// The floor on one piece of collateral: its `tcbEvaluationDataNumber`
    /// (tcbInfo, QE identity), then its age at `now` from `issued` (its
    /// issueDate, a CRL's thisUpdate). The caller has checked the window,
    /// so `issued <= now`.
    fn judge(
        &self,
        what: &'static str,
        evaluation: Option<u64>,
        issued: u64,
        now: u64,
    ) -> Result<(), TcbRejected> {
        if let Some(have) = evaluation {
            if have < self.min_tcb_evaluation_data_number {
                return Err(TcbRejected::EvaluationBelowMinimum {
                    what,
                    have,
                    min: self.min_tcb_evaluation_data_number,
                });
            }
        }
        let age_secs = now.saturating_sub(issued);
        if age_secs > self.max_collateral_age_secs {
            return Err(TcbRejected::CollateralTooOld {
                what,
                age_secs,
                max_secs: self.max_collateral_age_secs,
            });
        }
        Ok(())
    }
}

/// Which of Intel's two TCB answers a verdict accepts on (decision of
/// 2026-10-28). Ordered: `Strict` is stricter.
/// `Default` is `Strict`, the fail-closed value of a policy written
/// without a mode; `TcbPolicy::production` names `Transparent`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TcbMode {
    /// Accept on Intel's `standard` collateral, the channel the market
    /// serves on, with the accepted statuses; read the `early` collateral
    /// too and name, in the verdict and every receipt, the platform's TCB
    /// date and the advisories Intel lists against its level there. This
    /// discloses; it does not protect against the advisories it names.
    Transparent,
    /// Accept only on the `early` collateral at the policy's floor: a
    /// platform Intel's latest recovery rates OutOfDate is refused (76).
    #[default]
    Strict,
}

impl TcbMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Transparent => "transparent",
            Self::Strict => "strict",
        }
    }
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "transparent" => Ok(Self::Transparent),
            "strict" => Ok(Self::Strict),
            other => Err(format!(
                "TCB policy {other:?}: \"transparent\" or \"strict\""
            )),
        }
    }
}

/// Intel PCS v4 `update`: `early`, the latest TCB recovery, "commonly the
/// day of public disclosure"; `standard`, "commonly 12 months after".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcsUpdate {
    Early,
    Standard,
}

impl PcsUpdate {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Early => "early",
            Self::Standard => "standard",
        }
    }
}

/// What Intel may say about a TCB level, and what we do with each answer.
///
/// SECURITY: the accepted set is a POLICY, written here once. `UpToDate` is
/// obvious; `SWHardeningNeeded` is accepted because it means "microcode
/// current, OS-level mitigations advised" — the fleet state of most public
/// clouds most of the time, and refusing it would make the engine unusable
/// everywhere while protecting against nothing the enclave can fix. Every
/// other status refuses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TcbStatus {
    UpToDate,
    SwHardeningNeeded,
    ConfigurationNeeded,
    ConfigurationAndSwHardeningNeeded,
    OutOfDate,
    OutOfDateConfigurationNeeded,
    Revoked,
    Other(String),
}

impl TcbStatus {
    fn parse(s: &str) -> Self {
        match s {
            "UpToDate" => Self::UpToDate,
            "SWHardeningNeeded" => Self::SwHardeningNeeded,
            "ConfigurationNeeded" => Self::ConfigurationNeeded,
            "ConfigurationAndSWHardeningNeeded" => Self::ConfigurationAndSwHardeningNeeded,
            "OutOfDate" => Self::OutOfDate,
            "OutOfDateConfigurationNeeded" => Self::OutOfDateConfigurationNeeded,
            "Revoked" => Self::Revoked,
            other => Self::Other(other.to_string()),
        }
    }
    pub fn accepted(&self) -> bool {
        matches!(self, Self::UpToDate | Self::SwHardeningNeeded)
    }
    pub fn as_str(&self) -> &str {
        match self {
            Self::UpToDate => "UpToDate",
            Self::SwHardeningNeeded => "SWHardeningNeeded",
            Self::ConfigurationNeeded => "ConfigurationNeeded",
            Self::ConfigurationAndSwHardeningNeeded => "ConfigurationAndSWHardeningNeeded",
            Self::OutOfDate => "OutOfDate",
            Self::OutOfDateConfigurationNeeded => "OutOfDateConfigurationNeeded",
            Self::Revoked => "Revoked",
            Self::Other(s) => s,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TcbRejected {
    #[error("quote: {0}")]
    Quote(String),
    #[error("PCK SGX extension: {0}")]
    PckExtension(String),
    #[error("signed collateral ({0}): {1}")]
    Collateral(&'static str, String),
    #[error("TCB status {status} for FMSPC {fmspc} is not accepted (level dated {level_date})")]
    StatusRefused {
        status: String,
        fmspc: String,
        level_date: String,
    },
    #[error("no TCB level matches the platform's SVNs (below every published level)")]
    BelowEveryLevel,
    #[error("QE identity: {0}")]
    Qe(String),
    #[error("PCK certificate {serial} is revoked: Intel's PCK CRL lists it")]
    PckRevoked { serial: String },
    #[error("PCK CRL: {0}")]
    PckCrl(String),
    #[error("{subject} (serial {serial}) is revoked: Intel's root CA CRL lists it")]
    CaRevoked { subject: String, serial: String },
    #[error("root CA CRL: {0}")]
    RootCrl(String),
    /// Intel's tcbInfo publishes no identity for the TDX module version the
    /// quote names: a verdict on the platform, not an unusable collateral.
    #[error("TDX module version {version} ({id}) has no identity in tcbInfo")]
    ModuleUnknown { id: String, version: u8 },
    /// Intel's tcbInfo publishes no level at or below the module's ISVSVN.
    #[error("TDX module {id} ISVSVN {isvsvn} is below every published level")]
    ModuleBelowEveryLevel { id: String, isvsvn: u8 },
    #[error(
        "{what} is Intel's TCB evaluation {have}; this verifier requires at least {min} \
         (the TCB recovery its policy names)"
    )]
    EvaluationBelowMinimum {
        what: &'static str,
        have: u64,
        min: u64,
    },
    #[error(
        "{what} was issued {age_secs} s before now; this verifier accepts at most {max_secs} s"
    )]
    CollateralTooOld {
        what: &'static str,
        age_secs: u64,
        max_secs: u64,
    },
}

/// The platform identity the PCK certificate carries, read from Intel's SGX
/// extension (OID 1.2.840.113741.1.13.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PckTcb {
    pub fmspc_hex: String,
    /// The 16 CPUSVN component bytes — identical to the per-component SVNs
    /// the extension lists individually (they are the same values, stated by
    /// the DCAP spec; matching uses these bytes directly).
    pub cpusvn: [u8; 16],
    pub pcesvn: u16,
}

// ── Minimal DER walking, for exactly one extension ──────────────────────────

/// One TLV: (tag, content). Long-form lengths up to 4 bytes; anything
/// stranger refuses — this walks Intel-issued certificates, not arbitrary
/// input, and a shape outside their profile is a reason to stop, not parse.
fn der_tlv(bytes: &[u8]) -> Result<(u8, &[u8], &[u8]), String> {
    let (&tag, rest) = bytes.split_first().ok_or("empty TLV")?;
    let (&l0, rest) = rest.split_first().ok_or("missing length")?;
    let (len, rest) = if l0 < 0x80 {
        (l0 as usize, rest)
    } else {
        let n = (l0 & 0x7f) as usize;
        if n == 0 || n > 4 || rest.len() < n {
            return Err(format!("unsupported length form {l0:#x}"));
        }
        let mut len = 0usize;
        for &b in &rest[..n] {
            len = (len << 8) | b as usize;
        }
        (len, &rest[n..])
    };
    if rest.len() < len {
        return Err(format!("length {len} exceeds remaining {}", rest.len()));
    }
    Ok((tag, &rest[..len], &rest[len..]))
}

/// Walk a SEQUENCE OF SEQUENCE { OID, value } and hand each (oid, value TLV
/// bytes) to `f`.
fn walk_oid_pairs(
    seq_content: &[u8],
    mut f: impl FnMut(&[u8], &[u8]) -> Result<(), String>,
) -> Result<(), String> {
    let mut rest = seq_content;
    while !rest.is_empty() {
        let (tag, inner, next) = der_tlv(rest)?;
        if tag != 0x30 {
            return Err(format!("expected SEQUENCE in pair list, got tag {tag:#x}"));
        }
        let (otag, oid, after_oid) = der_tlv(inner)?;
        if otag != 0x06 {
            return Err(format!("expected OID, got tag {otag:#x}"));
        }
        f(oid, after_oid)?;
        rest = next;
    }
    Ok(())
}

const OID_SGX_EXT: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf8, 0x4d, 0x01, 0x0d, 0x01];
const OID_TCB: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf8, 0x4d, 0x01, 0x0d, 0x01, 0x02];
const OID_PCESVN: &[u8] = &[
    0x2a, 0x86, 0x48, 0x86, 0xf8, 0x4d, 0x01, 0x0d, 0x01, 0x02, 0x11,
];
const OID_CPUSVN: &[u8] = &[
    0x2a, 0x86, 0x48, 0x86, 0xf8, 0x4d, 0x01, 0x0d, 0x01, 0x02, 0x12,
];
const OID_FMSPC: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf8, 0x4d, 0x01, 0x0d, 0x01, 0x04];

/// Read FMSPC, CPUSVN and PCESVN from the quote's own PCK leaf certificate.
pub fn pck_tcb(quote: &[u8]) -> Result<PckTcb, TcbRejected> {
    let sig = parse_sig_section(quote).map_err(|e| TcbRejected::Quote(e.to_string()))?;
    let leaf_pem = Pem::iter_from_buffer(sig.pck_chain_pem)
        .next()
        .ok_or_else(|| TcbRejected::PckExtension("no PEM in chain".into()))?
        .map_err(|e| TcbRejected::PckExtension(format!("PEM: {e}")))?;
    let (_, leaf) = X509Certificate::from_der(&leaf_pem.contents)
        .map_err(|e| TcbRejected::PckExtension(format!("X.509: {e}")))?;
    let ext = leaf
        .extensions()
        .iter()
        .find(|e| e.oid.as_bytes() == OID_SGX_EXT)
        .ok_or_else(|| TcbRejected::PckExtension("SGX extension absent from PCK leaf".into()))?;
    // Extension value: SEQUENCE OF SEQUENCE { OID, value }.
    let (tag, seq, _) = der_tlv(ext.value).map_err(TcbRejected::PckExtension)?;
    if tag != 0x30 {
        return Err(TcbRejected::PckExtension(format!(
            "extension is not a SEQUENCE (tag {tag:#x})"
        )));
    }
    let mut fmspc: Option<[u8; 6]> = None;
    let mut cpusvn: Option<[u8; 16]> = None;
    let mut pcesvn: Option<u16> = None;
    walk_oid_pairs(seq, |oid, value| {
        if oid == OID_FMSPC {
            let (t, v, _) = der_tlv(value)?;
            if t != 0x04 || v.len() != 6 {
                return Err(format!("FMSPC: tag {t:#x} len {}", v.len()));
            }
            fmspc = Some(v.try_into().unwrap());
        } else if oid == OID_TCB {
            let (t, tcb_seq, _) = der_tlv(value)?;
            if t != 0x30 {
                return Err(format!("TCB: not a SEQUENCE (tag {t:#x})"));
            }
            walk_oid_pairs(tcb_seq, |oid, value| {
                if oid == OID_PCESVN {
                    let (t, v, _) = der_tlv(value)?;
                    if t != 0x02 || v.is_empty() || v.len() > 3 {
                        return Err(format!("PCESVN: tag {t:#x} len {}", v.len()));
                    }
                    let mut n: u32 = 0;
                    for &b in v {
                        n = (n << 8) | b as u32;
                    }
                    pcesvn = Some(n as u16);
                } else if oid == OID_CPUSVN {
                    let (t, v, _) = der_tlv(value)?;
                    if t != 0x04 || v.len() != 16 {
                        return Err(format!("CPUSVN: tag {t:#x} len {}", v.len()));
                    }
                    cpusvn = Some(v.try_into().unwrap());
                }
                Ok(())
            })?;
        }
        Ok(())
    })
    .map_err(TcbRejected::PckExtension)?;
    match (fmspc, cpusvn, pcesvn) {
        (Some(f), Some(c), Some(p)) => Ok(PckTcb {
            fmspc_hex: f.iter().map(|b| format!("{b:02x}")).collect(),
            cpusvn: c,
            pcesvn: p,
        }),
        _ => Err(TcbRejected::PckExtension(
            "FMSPC, CPUSVN or PCESVN missing from the SGX extension".into(),
        )),
    }
}

/// The quote's QE report body (384 bytes) — exposed so external verifiers
/// (and the accept-path test) can read the identity fields the QE identity
/// check matches against.
pub fn qe_report(quote: &[u8]) -> Result<[u8; 384], TcbRejected> {
    let sig = parse_sig_section(quote).map_err(|e| TcbRejected::Quote(e.to_string()))?;
    sig.qe_report
        .try_into()
        .map_err(|_| TcbRejected::Quote("QE report is not 384 bytes".into()))
}

/// The quote body's `tee_tcb_svn` — the TDX module/TD stack SVNs.
pub fn tee_tcb_svn(quote: &[u8]) -> Result<[u8; 16], TcbRejected> {
    quote
        .get(48..64)
        .and_then(|s| <[u8; 16]>::try_from(s).ok())
        .ok_or_else(|| TcbRejected::Quote("too short for tee_tcb_svn".into()))
}

/// What the quote says about the TDX module, judged against tcbInfo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdxModule {
    /// `TDX_<version>` from `TEE_TCB_SVN[1]`; `tdxModule` when that byte is 0.
    pub id: String,
    /// The module's own level; `None` for a version-0 module, which Intel's
    /// algorithm judges through the platform levels alone.
    pub status: Option<TcbStatus>,
    /// The `advisoryIDs` Intel lists against that level, in its order.
    pub advisories: Vec<String>,
}

/// The TDX module against Intel's tcbInfo — "Get TDX TCB Info", steps 5
/// and 6 of the PCS API documentation (read 2026-10-13): `TEE_TCB_SVN[1]`
/// names the identity `TDX_<version>`, `TEE_TCB_SVN[0]` is its ISVSVN, the
/// first level at or below it gives the status. MRSIGNERSEAM and
/// SEAMATTRIBUTES (TD report body [64..112] and [112..120]) must equal the
/// identity's `mrsigner` and masked `attributes` (`tdxModule` for version
/// 0). No identity, or below every level: refused.
/// The id is formatted as two uppercase hex digits: only `TDX_01` and
/// `TDX_03` were observed in Intel's collateral (vendored 2026-08-20).
pub fn tdx_module_status(
    tcb_info: &serde_json::Value,
    quote: &[u8],
) -> Result<TdxModule, TcbRejected> {
    let err = |m: String| TcbRejected::Collateral("tcbInfo", m);
    let tee = tee_tcb_svn(quote)?;
    let body = quote
        .get(48..48 + 584)
        .ok_or_else(|| TcbRejected::Quote("too short for the TD report body".into()))?;
    let mrsignerseam = &body[64..112];
    let seam_attributes = &body[112..120];
    let (id, golden) = if tee[1] == 0 {
        ("tdxModule".to_string(), &tcb_info["tdxModule"])
    } else {
        let id = format!("TDX_{:02X}", tee[1]);
        let identity = tcb_info["tdxModuleIdentities"]
            .as_array()
            .and_then(|a| a.iter().find(|m| m["id"].as_str() == Some(id.as_str())))
            .ok_or_else(|| TcbRejected::ModuleUnknown {
                id: id.clone(),
                version: tee[1],
            })?;
        (id, identity)
    };
    if !golden.is_object() {
        return Err(err(format!("{id}: no TDX module identity in tcbInfo")));
    }
    let field = |name: &str, len: usize| -> Result<Vec<u8>, TcbRejected> {
        let v = crate::cloud_config::hex_bytes(golden[name].as_str().unwrap_or_default())
            .map_err(|e| err(format!("{id}.{name} is not hex: {e}")))?;
        if v.len() != len {
            return Err(err(format!(
                "{id}.{name}: {} bytes, expected {len}",
                v.len()
            )));
        }
        Ok(v)
    };
    let mrsigner = field("mrsigner", 48)?;
    if mrsigner != mrsignerseam {
        return Err(err(format!(
            "MRSIGNERSEAM {} is not the module signer Intel publishes for {id}",
            mrsignerseam
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        )));
    }
    let attributes = field("attributes", 8)?;
    let mask = field("attributesMask", 8)?;
    let masked_ok = seam_attributes
        .iter()
        .zip(&mask)
        .map(|(h, m)| h & m)
        .eq(attributes.iter().zip(&mask).map(|(w, m)| w & m));
    if !masked_ok {
        return Err(err(format!(
            "SEAMATTRIBUTES do not match the attributes Intel publishes for {id}"
        )));
    }
    if tee[1] == 0 {
        return Ok(TdxModule {
            id,
            status: None,
            advisories: Vec::new(),
        });
    }
    let levels = golden["tcbLevels"]
        .as_array()
        .ok_or_else(|| err(format!("{id}: no tcbLevels")))?;
    for level in levels {
        let need = level["tcb"]["isvsvn"]
            .as_u64()
            .ok_or_else(|| err(format!("{id}: a level without isvsvn")))?;
        if u64::from(tee[0]) >= need {
            let status = TcbStatus::parse(level["tcbStatus"].as_str().unwrap_or_default());
            return Ok(TdxModule {
                id,
                status: Some(status),
                advisories: advisory_ids(level),
            });
        }
    }
    Err(TcbRejected::ModuleBelowEveryLevel { id, isvsvn: tee[0] })
}

// ── Signed collateral (tcbInfo / QE identity) ───────────────────────────────

/// "Intel SGX TCB Signing", then the root: the chain PCS sends with a tcbInfo
/// and a QE identity.
const COLLATERAL_CHAIN_LEN: usize = 2;

/// Verify one piece of PCS collateral: the issuer chain ends on the pinned
/// root, the leaf signs the RAW BYTES of `body`'s `key` value (ECDSA P-256 /
/// SHA-256), the validity window contains `now`, and `policy` accepts its
/// TCB evaluation and its age. Returns the parsed value.
fn verify_collateral(
    what: &'static str,
    body: &str,
    key: &str,
    issuer_chain_pem: &str,
    anchor: &TrustAnchor,
    now_unix: u64,
    policy: &TcbPolicy,
) -> Result<serde_json::Value, TcbRejected> {
    let err = |m: String| TcbRejected::Collateral(what, m);
    let mut certs: Vec<X509Certificate<'_>> = Vec::new();
    let ders: Vec<Vec<u8>> = Pem::iter_from_buffer(issuer_chain_pem.as_bytes())
        .map(|p| p.map(|p| p.contents))
        .collect::<Result<_, _>>()
        .map_err(|e| err(format!("issuer chain PEM: {e}")))?;
    // Intel's chain is the TCB Signing certificate and the root (PCS headers,
    // `tests/pcs_live/*_issuer_chain.pem`). A longer one is a signer issued
    // under a PCK whose key a host extracted.
    if ders.len() != COLLATERAL_CHAIN_LEN {
        return Err(err(format!(
            "issuer chain has {} certificate(s); Intel's is the TCB Signing certificate and the root",
            ders.len()
        )));
    }
    for der in &ders {
        certs.push(
            X509Certificate::from_der(der)
                .map(|(_, c)| c)
                .map_err(|e| err(format!("issuer chain X.509: {e}")))?,
        );
    }
    let root = certs.last().expect("len checked");
    if !root_matches(root, anchor) {
        return Err(err(
            "issuer chain root does not match the pinned anchor".into()
        ));
    }
    // Each certificate signed by the next; the root self-signed.
    for i in 0..certs.len() {
        let issuer =
            p256_key_from_cert(certs.get(i + 1).unwrap_or(root)).map_err(|e| err(e.to_string()))?;
        verify_cert_signed_by(&certs[i], &issuer, what, now_unix)
            .map_err(|e| err(e.to_string()))?;
    }

    // SECURITY: WHO signed the verdict, not merely that someone in
    // Intel's PKI did. Until 2026-08-22 the only requirement was "chains to
    // the pinned Intel root" — and Intel's root issues a great many
    // certificates. A PCK leaf, held by whoever runs that platform, chains
    // to exactly this root; without the three checks below it could sign a
    // tcbInfo saying every platform is UpToDate, and this verifier would
    // have agreed. Intel's own QVL requires the same identity, which is why
    // the constant is a literal and not a policy knob.
    //
    // Three properties, because one alone is not the identity:
    //   - subject CN is the TCB signing certificate's,
    //   - it is an end entity (a CA certificate signing leaf data is a
    //     confusion of roles, and the path length would be nobody's job),
    //   - its key is usable for signatures at all.
    const TCB_SIGNING_CN: &str = "Intel SGX TCB Signing";
    let leaf = &certs[0];
    let cn_matches = leaf
        .subject()
        .iter_common_name()
        .filter_map(|a| a.as_str().ok())
        .any(|cn| cn == TCB_SIGNING_CN);
    if !cn_matches {
        return Err(err(format!(
            "collateral signer is not \"{TCB_SIGNING_CN}\": subject is {}",
            leaf.subject()
        )));
    }
    match leaf.basic_constraints() {
        Ok(Some(bc)) if bc.value.ca => {
            return Err(err(
                "collateral signer is a CA certificate, not an end entity".into(),
            ))
        }
        Ok(_) => {}
        Err(e) => return Err(err(format!("collateral signer basicConstraints: {e}"))),
    }
    match leaf.key_usage() {
        Ok(Some(ku)) if ku.value.digital_signature() => {}
        Ok(Some(_)) => {
            return Err(err(
                "collateral signer's keyUsage does not allow digitalSignature".into(),
            ))
        }
        Ok(None) => return Err(err("collateral signer carries no keyUsage extension".into())),
        Err(e) => return Err(err(format!("collateral signer keyUsage: {e}"))),
    }

    intel_shape(&certs, &["the collateral signer", "the root"]).map_err(err)?;

    let leaf_key = p256_key_from_cert(&certs[0]).map_err(|e| err(e.to_string()))?;

    // The signature covers the raw JSON bytes of the value — extracted from
    // the body VERBATIM, never re-serialized (serde would reorder keys and
    // change the bytes).
    let raw = raw_json_value(body, key).ok_or_else(|| err(format!("no \"{key}\" value")))?;
    let parsed: serde_json::Value =
        serde_json::from_str(body).map_err(|e| err(format!("body is not JSON: {e}")))?;
    let sig_hex = parsed["signature"]
        .as_str()
        .ok_or_else(|| err("no signature field".into()))?;
    let sig_bytes = crate::cloud_config::hex_bytes(sig_hex)
        .map_err(|e| err(format!("signature is not hex: {e}")))?;
    if sig_bytes.len() != 64 {
        return Err(err(format!(
            "signature is {} bytes, expected 64",
            sig_bytes.len()
        )));
    }
    let sig = Signature::from_slice(&sig_bytes).map_err(|e| err(format!("signature: {e}")))?;
    leaf_key
        .verify(raw.as_bytes(), &sig)
        .map_err(|_| err("signature does not verify over the collateral bytes".into()))?;

    // SECURITY (critical): the verdict reads EXACTLY the
    // bytes the signature covered — `raw`, reparsed — never `parsed[key]`.
    // The two could differ: `raw_json_value` finds the FIRST textual
    // occurrence, serde_json resolves duplicate keys LAST-wins, so a body
    // carrying Intel's genuine object plus a forged duplicate would verify
    // the genuine bytes and serve the forged status (Revoked → UpToDate).
    // A duplicate key is itself a refusal: Intel's real answers have one.
    let occurrences = body.matches(&format!("\"{key}\":")).count();
    if occurrences != 1 {
        return Err(err(format!(
            "\"{key}\" appears {occurrences} times in the collateral body — \
             a duplicate is indistinguishable from a substitution, refused"
        )));
    }
    let value: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| err(format!("{key} value is not JSON: {e}")))?;
    let issue = date_unix(value["issueDate"].as_str().unwrap_or_default())
        .ok_or_else(|| err("issueDate unreadable".into()))?;
    let next = date_unix(value["nextUpdate"].as_str().unwrap_or_default())
        .ok_or_else(|| err("nextUpdate unreadable".into()))?;
    if now_unix < issue || now_unix > next {
        return Err(err(format!(
            "validity window [{issue}, {next}] does not contain now={now_unix}"
        )));
    }
    // SECURITY (TOB-WAPI-8): Intel's genuine `standard` answer, or a cached
    // one, verifies as well as the latest: the signature says who, not when
    // nor under which recovery. The floor is the verifier's own.
    let evaluation = value["tcbEvaluationDataNumber"]
        .as_u64()
        .ok_or_else(|| err("tcbEvaluationDataNumber unreadable".into()))?;
    policy.judge(what, Some(evaluation), issue, now_unix)?;
    Ok(value)
}

/// The raw byte-slice of `"key": <value>` in `body` — balanced braces,
/// string- and escape-aware. PCS values are objects; anything else refuses.
fn raw_json_value<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{key}\":");
    let start = body.find(&pat)? + pat.len();
    let bytes = body.as_bytes();
    let mut i = start;
    while i < bytes.len() && (bytes[i] as char).is_whitespace() {
        i += 1;
    }
    if bytes.get(i) != Some(&b'{') {
        return None;
    }
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    for (j, &b) in bytes[i..].iter().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&body[i..i + j + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

/// The value of one field of a date: unsigned decimal digits, one to
/// `max_len` of them. `str::parse` alone reads a sign (`+1`, `-1`). The
/// reader of every date field of this file and of
/// `remote_verify::parse_http_date`.
pub(crate) fn date_field(field: &str, max_len: usize) -> Option<i64> {
    let digits =
        !field.is_empty() && field.len() <= max_len && field.bytes().all(|b| b.is_ascii_digit());
    if digits {
        field.parse().ok()
    } else {
        None
    }
}

/// `YYYY-MM-DDThh:mm:ssZ` → unix seconds. Days-from-civil (Howard Hinnant's
/// algorithm); no dependency for one fixed format. Each field is digits
/// alone ([`date_field`]), within its range: month 1 to 12, day 1 to 31, hour
/// 0 to 23, minute 0 to 59, second 0 to 60 (a leap second).
fn date_unix(s: &str) -> Option<u64> {
    let b = s.as_bytes();
    // ASCII only: the fields below are cut by byte offsets.
    if b.len() != 20
        || !s.is_ascii()
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[19] != b'Z'
    {
        return None;
    }
    let n = |r: std::ops::Range<usize>| date_field(&s[r.clone()], r.len());
    let (y, m, d) = (n(0..4)?, n(5..7)?, n(8..10)?);
    let (hh, mm, ss) = (n(11..13)?, n(14..16)?, n(17..19)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || hh > 23 || mm > 59 || ss > 60 {
        return None;
    }
    let y_adj = if m <= 2 { y - 1 } else { y };
    let era = y_adj.div_euclid(400);
    let yoe = y_adj - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    u64::try_from(days * 86400 + hh * 3600 + mm * 60 + ss).ok()
}

// ── Matching ────────────────────────────────────────────────────────────────

fn comps(level: &serde_json::Value, key: &str) -> Option<Vec<u8>> {
    let arr = level["tcb"][key].as_array()?;
    if arr.len() != 16 {
        return None;
    }
    arr.iter()
        .map(|c| c["svn"].as_u64().and_then(|v| u8::try_from(v).ok()))
        .collect()
}

/// One TCB level as Intel rates it: its status, its `tcbDate` and the
/// `advisoryIDs` it lists, in Intel's order (none for an UpToDate level).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rating {
    pub status: TcbStatus,
    pub tcb_date: String,
    pub advisories: Vec<String>,
}

/// A level's `advisoryIDs`, strings only.
fn advisory_ids(level: &serde_json::Value) -> Vec<String> {
    level["advisoryIDs"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// The platform's TCB status: first published level (they are ordered
/// newest-first) whose SGX components, PCESVN AND TDX components are all
/// ≤ the platform's. Below every level is its own refusal.
pub fn tdx_tcb_status(
    tcb_info: &serde_json::Value,
    pck: &PckTcb,
    tee_svn: &[u8; 16],
) -> Result<(TcbStatus, String), TcbRejected> {
    tdx_tcb_level(tcb_info, pck, tee_svn).map(|r| (r.status, r.tcb_date))
}

/// [`tdx_tcb_status`], with the advisories Intel lists against the level.
pub fn tdx_tcb_level(
    tcb_info: &serde_json::Value,
    pck: &PckTcb,
    tee_svn: &[u8; 16],
) -> Result<Rating, TcbRejected> {
    let err = |m: String| TcbRejected::Collateral("tcbInfo", m);
    // Same rule as the QE identity below: refuse schema versions this
    // verifier does not implement, by name.
    if tcb_info["version"].as_u64() != Some(3) {
        return Err(err(format!(
            "tcbInfo version {:?}, this verifier implements 3",
            tcb_info["version"]
        )));
    }
    if tcb_info["id"].as_str() != Some("TDX") {
        return Err(err(format!("id {:?}, expected \"TDX\"", tcb_info["id"])));
    }
    let fmspc = tcb_info["fmspc"]
        .as_str()
        .unwrap_or_default()
        .to_lowercase();
    if fmspc != pck.fmspc_hex {
        return Err(err(format!(
            "tcbInfo is for FMSPC {fmspc}, the platform is {}",
            pck.fmspc_hex
        )));
    }
    let levels = tcb_info["tcbLevels"]
        .as_array()
        .ok_or_else(|| err("no tcbLevels".into()))?;
    for level in levels {
        let sgx = comps(level, "sgxtcbcomponents").ok_or_else(|| err("sgxtcbcomponents".into()))?;
        let tdx = comps(level, "tdxtcbcomponents").ok_or_else(|| err("tdxtcbcomponents".into()))?;
        let pce = level["tcb"]["pcesvn"]
            .as_u64()
            .ok_or_else(|| err("pcesvn".into()))?;
        let sgx_ok = pck.cpusvn.iter().zip(&sgx).all(|(have, need)| have >= need);
        let tdx_ok = tee_svn.iter().zip(&tdx).all(|(have, need)| have >= need);
        if sgx_ok && tdx_ok && u64::from(pck.pcesvn) >= pce {
            return Ok(Rating {
                status: TcbStatus::parse(level["tcbStatus"].as_str().unwrap_or_default()),
                tcb_date: level["tcbDate"].as_str().unwrap_or("?").to_string(),
                advisories: advisory_ids(level),
            });
        }
    }
    Err(TcbRejected::BelowEveryLevel)
}

/// The QE's status against the published TD_QE identity, schema 2: MRSIGNER,
/// ISVPRODID, MISCSELECT/ATTRIBUTES under their masks, then the ISVSVN
/// level. Field offsets are the SGX report body layout (384 bytes).
pub fn qe_status(identity: &serde_json::Value, qe_report: &[u8]) -> Result<TcbStatus, TcbRejected> {
    qe_level(identity, qe_report).map(|r| r.status)
}

/// [`qe_status`], with the level's date and advisories.
pub fn qe_level(identity: &serde_json::Value, qe_report: &[u8]) -> Result<Rating, TcbRejected> {
    let err = |m: String| TcbRejected::Qe(m);
    if qe_report.len() != 384 {
        return Err(err(format!(
            "QE report is {} bytes, expected 384",
            qe_report.len()
        )));
    }
    if identity["id"].as_str() != Some("TD_QE") {
        return Err(err(format!(
            "identity id {:?}, expected \"TD_QE\"",
            identity["id"]
        )));
    }
    // SECURITY: unchecked until 2026-11-05, while tcbInfo refused all but 3. Intel's QVL
    // takes TD_QE at 2 and refuses 1; the 3 identities vendored from Intel are 2.
    if identity["version"].as_u64() != Some(2) {
        return Err(err(format!(
            "identity version {:?}, this verifier implements 2",
            identity["version"]
        )));
    }
    let field = |name: &str| -> Result<Vec<u8>, TcbRejected> {
        crate::cloud_config::hex_bytes(identity[name].as_str().unwrap_or_default())
            .map_err(|e| err(format!("{name} is not hex: {e}")))
    };
    let masked_eq =
        |have: &[u8], want: &[u8], mask: &[u8], what: &str| -> Result<(), TcbRejected> {
            if have.len() != want.len() || want.len() != mask.len() {
                return Err(err(format!("{what}: length mismatch")));
            }
            let ok = have
                .iter()
                .zip(mask)
                .map(|(h, m)| h & m)
                .eq(want.iter().zip(mask).map(|(w, m)| w & m));
            if ok {
                Ok(())
            } else {
                Err(err(format!("{what} does not match the published identity")))
            }
        };
    masked_eq(
        // sgx_report_body_t: cpu_svn [0..16), misc_select u32 at 16 —
        // the old [8..12] read four CPUSVN bytes, and the check
        // passed by accident (both sides zero on most platforms) while the
        // real MISCSELECT was never compared. The test encoded the same
        // wrong index — fixed together, pinned on the vendored quote.
        &qe_report[16..20],
        &field("miscselect")?,
        &field("miscselectMask")?,
        "MISCSELECT",
    )?;
    masked_eq(
        &qe_report[48..64],
        &field("attributes")?,
        &field("attributesMask")?,
        "ATTRIBUTES",
    )?;
    let mrsigner = field("mrsigner")?;
    if qe_report[128..160] != mrsigner[..] {
        return Err(err("MRSIGNER does not match the published identity".into()));
    }
    let isvprodid = u16::from_le_bytes([qe_report[256], qe_report[257]]);
    if u64::from(isvprodid) != identity["isvprodid"].as_u64().unwrap_or(u64::MAX) {
        return Err(err(format!("ISVPRODID {isvprodid} does not match")));
    }
    let isvsvn = u16::from_le_bytes([qe_report[258], qe_report[259]]);
    let levels = identity["tcbLevels"]
        .as_array()
        .ok_or_else(|| err("no tcbLevels".into()))?;
    for level in levels {
        let need = level["tcb"]["isvsvn"].as_u64().unwrap_or(u64::MAX);
        if u64::from(isvsvn) >= need {
            return Ok(Rating {
                status: TcbStatus::parse(level["tcbStatus"].as_str().unwrap_or_default()),
                tcb_date: level["tcbDate"].as_str().unwrap_or("?").to_string(),
                advisories: advisory_ids(level),
            });
        }
    }
    Err(err(format!(
        "QE ISVSVN {isvsvn} is below every published level"
    )))
}

/// The tcbInfo and QE identity one verdict reads, as PCS sent them: each
/// body with the issuer chain of its response header.
#[derive(Debug, Clone, Copy)]
pub struct Collateral<'a> {
    pub tcb_body: &'a str,
    pub tcb_chain_pem: &'a str,
    pub qe_body: &'a str,
    pub qe_chain_pem: &'a str,
}

/// Intel's collateral for one platform, verified and rated level by level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RatedTcb {
    /// The tcbInfo's and the QE identity's `tcbEvaluationDataNumber`.
    pub tcb_evaluation: u64,
    pub qe_evaluation: u64,
    /// The older of the two `issueDate`s.
    pub issued_unix: u64,
    pub platform: Rating,
    pub tdx_module: TdxModule,
    pub qe: Rating,
}

impl RatedTcb {
    /// Every advisory Intel lists against the platform's level, then the TDX
    /// module's, then the QE's, each once, in that order.
    pub fn advisories(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for id in self
            .platform
            .advisories
            .iter()
            .chain(&self.tdx_module.advisories)
            .chain(&self.qe.advisories)
        {
            if !out.contains(id) {
                out.push(id.clone());
            }
        }
        out
    }

    /// Platform, TDX module (when it has a level of its own) and QE all
    /// [`TcbStatus::accepted`].
    pub fn accepted(&self) -> bool {
        self.platform.status.accepted()
            && self
                .tdx_module
                .status
                .as_ref()
                .is_none_or(TcbStatus::accepted)
            && self.qe.status.accepted()
    }
}

/// Verify both pieces of collateral under `policy` and rate the platform,
/// the TDX module and the QE. With `refuse`, a status Intel does not accept
/// is refused where it is read, platform first: the order of the checks is
/// the order of the refusals.
fn rate(
    quote: &[u8],
    c: &Collateral<'_>,
    anchor: &TrustAnchor,
    now_unix: u64,
    policy: &TcbPolicy,
    refuse: bool,
) -> Result<RatedTcb, TcbRejected> {
    let pck = pck_tcb(quote)?;
    let tee = tee_tcb_svn(quote)?;
    let tcb_info = verify_collateral(
        "tcbInfo",
        c.tcb_body,
        "tcbInfo",
        c.tcb_chain_pem,
        anchor,
        now_unix,
        policy,
    )?;
    let platform = tdx_tcb_level(&tcb_info, &pck, &tee)?;
    if refuse && !platform.status.accepted() {
        return Err(TcbRejected::StatusRefused {
            status: platform.status.as_str().to_string(),
            fmspc: pck.fmspc_hex,
            level_date: platform.tcb_date,
        });
    }
    let module = tdx_module_status(&tcb_info, quote)?;
    if let Some(status) = &module.status {
        if refuse && !status.accepted() {
            return Err(TcbRejected::StatusRefused {
                status: format!("TDX module {} {}", module.id, status.as_str()),
                fmspc: pck.fmspc_hex,
                level_date: "-".into(),
            });
        }
    }
    let identity = verify_collateral(
        "qeIdentity",
        c.qe_body,
        "enclaveIdentity",
        c.qe_chain_pem,
        anchor,
        now_unix,
        policy,
    )?;
    let sig = parse_sig_section(quote).map_err(|e| TcbRejected::Quote(e.to_string()))?;
    let qe = qe_level(&identity, sig.qe_report)?;
    if refuse && !qe.status.accepted() {
        return Err(TcbRejected::StatusRefused {
            status: format!("QE {}", qe.status.as_str()),
            fmspc: pck.fmspc_hex,
            level_date: "-".into(),
        });
    }
    // verify_collateral read both numbers and dates: unreadable is refused there.
    let number = |v: &serde_json::Value| v["tcbEvaluationDataNumber"].as_u64().unwrap_or(0);
    let issued = |v: &serde_json::Value| date_unix(v["issueDate"].as_str().unwrap_or_default());
    Ok(RatedTcb {
        tcb_evaluation: number(&tcb_info),
        qe_evaluation: number(&identity),
        issued_unix: issued(&tcb_info)
            .unwrap_or(0)
            .min(issued(&identity).unwrap_or(0)),
        platform,
        tdx_module: module,
        qe,
    })
}

/// Everything, end to end: parse the platform identity out of the quote,
/// verify both pieces of collateral against the pinned root, match, and
/// accept only [`TcbStatus::accepted`] on BOTH. Returns the two statuses
/// for the verdict's verifier string. One channel, whatever `policy.mode`
/// says: [`judge_live`] applies the mode.
#[allow(clippy::too_many_arguments)]
pub fn verify_live(
    quote: &[u8],
    tcb_body: &str,
    tcb_chain_pem: &str,
    qe_body: &str,
    qe_chain_pem: &str,
    anchor: &TrustAnchor,
    now_unix: u64,
    policy: &TcbPolicy,
) -> Result<(TcbStatus, TcbStatus), TcbRejected> {
    let c = Collateral {
        tcb_body,
        tcb_chain_pem,
        qe_body,
        qe_chain_pem,
    };
    let rated = rate(quote, &c, anchor, now_unix, policy, true)?;
    Ok((rated.platform.status, rated.qe.status))
}

/// Intel's latest (`early`) collateral for this platform, verified at the
/// policy's floor and age, rated without refusing on any status: what a
/// `Transparent` verdict names.
pub fn rate_latest(
    quote: &[u8],
    c: &Collateral<'_>,
    anchor: &TrustAnchor,
    now_unix: u64,
    policy: &TcbPolicy,
) -> Result<RatedTcb, TcbRejected> {
    rate(quote, c, anchor, now_unix, policy, false)
}

/// One live verdict on the platform, under a [`TcbMode`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveJudgement {
    pub mode: TcbMode,
    /// The collateral the acceptance stands on, and what it says.
    pub accepted_under: PcsUpdate,
    pub accepted: RatedTcb,
    /// Intel's latest (`early`) collateral, rated; or why it could not be
    /// fetched, verified or rated. Never read as "up to date" when absent.
    pub latest: Result<RatedTcb, String>,
}

impl LiveJudgement {
    /// The platform and QE statuses the acceptance stands on.
    pub fn statuses(&self) -> (TcbStatus, TcbStatus) {
        (
            self.accepted.platform.status.clone(),
            self.accepted.qe.status.clone(),
        )
    }
}

/// The live verdict under `policy.mode`. `accept` is the collateral of
/// [`TcbPolicy::acceptance_update`]. `Strict`: it is the `early` collateral,
/// accepted at the policy's floor, and `latest` is not read. `Transparent`:
/// it is the `standard` collateral, accepted at
/// [`TcbPolicy::MIN_STANDARD_TCB_EVALUATION_DATA_NUMBER`] with the same
/// statuses and age; `latest` is the `early` collateral as fetched, or why
/// it was not, rated at the policy's floor without refusing anything.
pub fn judge_live(
    quote: &[u8],
    accept: &Collateral<'_>,
    latest: Option<Result<Collateral<'_>, String>>,
    anchor: &TrustAnchor,
    now_unix: u64,
    policy: &TcbPolicy,
) -> Result<LiveJudgement, TcbRejected> {
    match policy.mode {
        TcbMode::Strict => {
            let accepted = rate(quote, accept, anchor, now_unix, policy, true)?;
            Ok(LiveJudgement {
                mode: policy.mode,
                accepted_under: PcsUpdate::Early,
                latest: Ok(accepted.clone()),
                accepted,
            })
        }
        TcbMode::Transparent => {
            let on_standard = policy.on_standard();
            let accepted = rate(quote, accept, anchor, now_unix, &on_standard, true)?;
            let latest = match latest {
                Some(Ok(c)) => {
                    rate_latest(quote, &c, anchor, now_unix, policy).map_err(|e| e.to_string())
                }
                Some(Err(e)) => Err(e),
                None => Err("Intel's early collateral was not fetched".into()),
            };
            Ok(LiveJudgement {
                mode: policy.mode,
                accepted_under: PcsUpdate::Standard,
                accepted,
                latest,
            })
        }
    }
}

// ── PCK revocation ──────────────────────────────────────────────────────────

/// Which PCK CRL covers this quote's PCK: `platform` or `processor`, named by
/// the leaf's issuer (Intel PCS v4, `GET /sgx/certification/v4/pckcrl?ca=`).
/// Any other issuer refuses: no CRL is known for it.
pub fn pck_crl_ca(quote: &[u8]) -> Result<&'static str, TcbRejected> {
    let sig = parse_sig_section(quote).map_err(|e| TcbRejected::Quote(e.to_string()))?;
    let leaf_pem = Pem::iter_from_buffer(sig.pck_chain_pem)
        .next()
        .ok_or_else(|| TcbRejected::PckCrl("no PEM in the PCK chain".into()))?
        .map_err(|e| TcbRejected::PckCrl(format!("PEM: {e}")))?;
    let (_, leaf) = X509Certificate::from_der(&leaf_pem.contents)
        .map_err(|e| TcbRejected::PckCrl(format!("PCK X.509: {e}")))?;
    let cn = leaf
        .issuer()
        .iter_common_name()
        .filter_map(|a| a.as_str().ok())
        .collect::<Vec<_>>();
    match cn.as_slice() {
        ["Intel SGX PCK Platform CA"] => Ok("platform"),
        ["Intel SGX PCK Processor CA"] => Ok("processor"),
        _ => Err(TcbRejected::PckCrl(format!(
            "the PCK's issuer {} is neither Intel SGX PCK Platform CA nor Processor CA",
            leaf.issuer()
        ))),
    }
}

/// The quote's PCK against Intel's PCK CRL (DER): issued and signed by the PCK's issuer (its
/// chain verified to the pinned root), `now` in [thisUpdate, nextUpdate], issued at most
/// `policy.max_collateral_age_secs` before `now`, no critical extension.
/// A listed serial is [`TcbRejected::PckRevoked`]; any other failure is an unusable CRL.
pub fn verify_pck_crl(
    quote: &[u8],
    crl_der: &[u8],
    anchor: &TrustAnchor,
    now_unix: u64,
    policy: &TcbPolicy,
) -> Result<(), TcbRejected> {
    let err = |m: String| TcbRejected::PckCrl(m);
    let sig = parse_sig_section(quote).map_err(|e| TcbRejected::Quote(e.to_string()))?;
    let chain = verified_pck_chain(sig.pck_chain_pem, anchor, now_unix)
        .map_err(|e| err(format!("PCK chain: {e}")))?;
    let (_, leaf) =
        X509Certificate::from_der(&chain[0]).map_err(|e| err(format!("PCK X.509: {e}")))?;
    let (_, ca) =
        X509Certificate::from_der(&chain[1]).map_err(|e| err(format!("PCK CA X.509: {e}")))?;
    let (rest, crl) = CertificateRevocationList::from_der(crl_der)
        .map_err(|e| err(format!("not a DER CRL: {e}")))?;
    if !rest.is_empty() {
        return Err(err(format!("{} bytes after the CRL", rest.len())));
    }
    if crl.issuer().as_raw() != ca.subject().as_raw()
        || leaf.issuer().as_raw() != ca.subject().as_raw()
    {
        return Err(err(format!(
            "issued by {}, the PCK's issuer is {}",
            crl.issuer(),
            leaf.issuer()
        )));
    }
    // RFC 5280, section 5.1.1.2: the algorithm written after the signed part
    // MUST be the one inside it. The outer one is covered by no signature.
    if !crate::pcs::signature_algorithm_repeated(crl.as_raw()) {
        return Err(err(
            "the signed and the outer signature algorithms differ".into()
        ));
    }
    let sig = Signature::from_der(crl.signature_value.data.as_ref())
        .map_err(|e| err(format!("signature not DER ECDSA: {e}")))?;
    p256_key_from_cert(&ca)
        .map_err(|e| err(e.to_string()))?
        .verify(crl.tbs_cert_list.as_ref(), &sig)
        .map_err(|_| err(format!("signature does not verify under {}", ca.subject())))?;
    let this_update = crl.last_update().timestamp();
    let next_update = crl
        .next_update()
        .ok_or_else(|| err("no nextUpdate".into()))?
        .timestamp();
    let now = now_unix as i64;
    if now < this_update || now > next_update {
        return Err(err(format!(
            "validity window [{this_update}, {next_update}] does not contain now={now}"
        )));
    }
    // Inside its window, a CRL issued more than the policy's age ago is what a
    // stale cache serves: a PCK Intel revoked since is not on it.
    // A thisUpdate before 1970 counts as issued at 0: too old, not of age 0.
    policy.judge(
        "PCK CRL",
        None,
        u64::try_from(this_update).unwrap_or(0),
        now_unix,
    )?;
    if let Some(x) = crl.extensions().iter().find(|x| x.critical) {
        return Err(err(format!("critical extension {} not understood", x.oid)));
    }
    let serial = &leaf.tbs_certificate.serial;
    for r in crl.iter_revoked_certificates() {
        if let Some(x) = r.extensions().iter().find(|x| x.critical) {
            return Err(err(format!(
                "critical entry extension {} not understood",
                x.oid
            )));
        }
        if r.serial() == serial {
            return Err(TcbRejected::PckRevoked {
                serial: format!("{serial:x}"),
            });
        }
    }
    Ok(())
}

// ── Root CA revocation ──────────────────────────────────────────────────────

/// Where the Intel SGX Root CA publishes its CRL: the CRL distribution point
/// of the root certificate itself (`tests/root_crl_live.rs` reads it from
/// the certificate). Measured on 2026-10-23: 294 bytes, CRL number 1,
/// thisUpdate 2026-02-26, nextUpdate 2027-02-26, no revoked certificate;
/// PCS v4 `/sgx/certification/v4/rootcacrl` answered 404.
pub const INTEL_SGX_ROOT_CA_CRL_URL: &str =
    "https://certificates.trustedservices.intel.com/IntelSGXRootCA.der";

/// What the root issued, against the root's own CRL (DER): the PCK CA that
/// issued the quote's PCK (second certificate of its chain, verified to the
/// pinned root), and the signer at the head of each collateral chain in
/// `signer_chains_pem` (Intel's TCB signing certificate; verified to be the
/// root's, inside its window). The CRL must be issued and signed by that
/// root, contain `now` in [thisUpdate, nextUpdate], and carry no critical
/// extension. Its age is not judged: Intel issues it for a year, so the
/// collateral age limit does not apply. A listed serial is
/// [`TcbRejected::CaRevoked`]; any other failure is an unusable CRL.
pub fn verify_root_crl(
    quote: &[u8],
    signer_chains_pem: &[&str],
    crl_der: &[u8],
    anchor: &TrustAnchor,
    now_unix: u64,
) -> Result<(), TcbRejected> {
    let err = |m: String| TcbRejected::RootCrl(m);
    let sig = parse_sig_section(quote).map_err(|e| TcbRejected::Quote(e.to_string()))?;
    let chain = verified_pck_chain(sig.pck_chain_pem, anchor, now_unix)
        .map_err(|e| err(format!("PCK chain: {e}")))?;
    let (_, root) = X509Certificate::from_der(chain.last().expect("at least two"))
        .map_err(|e| err(format!("root X.509: {e}")))?;
    let (_, pck_ca) =
        X509Certificate::from_der(&chain[1]).map_err(|e| err(format!("PCK CA X.509: {e}")))?;
    let root_key = p256_key_from_cert(&root).map_err(|e| err(e.to_string()))?;
    let mut issued = vec![(
        pck_ca.subject().to_string(),
        pck_ca.tbs_certificate.serial.clone(),
    )];
    let mut signers = Vec::new();
    for pem in signer_chains_pem {
        let first = Pem::iter_from_buffer(pem.as_bytes())
            .next()
            .ok_or_else(|| err("a collateral chain has no certificate".into()))?
            .map_err(|e| err(format!("collateral chain PEM: {e}")))?;
        signers.push(first.contents);
    }
    for der in &signers {
        let (_, signer) =
            X509Certificate::from_der(der).map_err(|e| err(format!("signer X.509: {e}")))?;
        verify_cert_signed_by(&signer, &root_key, "collateral signer", now_unix)
            .map_err(|e| err(format!("{e}: not issued by the root")))?;
        issued.push((
            signer.subject().to_string(),
            signer.tbs_certificate.serial.clone(),
        ));
    }
    let (rest, crl) = CertificateRevocationList::from_der(crl_der)
        .map_err(|e| err(format!("not a DER CRL: {e}")))?;
    if !rest.is_empty() {
        return Err(err(format!("{} bytes after the CRL", rest.len())));
    }
    if crl.issuer().as_raw() != root.subject().as_raw() {
        return Err(err(format!(
            "issued by {}, the root is {}",
            crl.issuer(),
            root.subject()
        )));
    }
    // RFC 5280, section 5.1.1.2: the algorithm written after the signed part
    // MUST be the one inside it. The outer one is covered by no signature.
    if !crate::pcs::signature_algorithm_repeated(crl.as_raw()) {
        return Err(err(
            "the signed and the outer signature algorithms differ".into()
        ));
    }
    let sig = Signature::from_der(crl.signature_value.data.as_ref())
        .map_err(|e| err(format!("signature not DER ECDSA: {e}")))?;
    root_key
        .verify(crl.tbs_cert_list.as_ref(), &sig)
        .map_err(|_| {
            err(format!(
                "signature does not verify under {}",
                root.subject()
            ))
        })?;
    let this_update = crl.last_update().timestamp();
    let next_update = crl
        .next_update()
        .ok_or_else(|| err("no nextUpdate".into()))?
        .timestamp();
    let now = now_unix as i64;
    if now < this_update || now > next_update {
        return Err(err(format!(
            "validity window [{this_update}, {next_update}] does not contain now={now}"
        )));
    }
    if let Some(x) = crl.extensions().iter().find(|x| x.critical) {
        return Err(err(format!("critical extension {} not understood", x.oid)));
    }
    for r in crl.iter_revoked_certificates() {
        if let Some(x) = r.extensions().iter().find(|x| x.critical) {
            return Err(err(format!(
                "critical entry extension {} not understood",
                x.oid
            )));
        }
        if let Some((subject, serial)) = issued.iter().find(|(_, s)| r.serial() == s) {
            return Err(TcbRejected::CaRevoked {
                subject: subject.clone(),
                serial: format!("{serial:x}"),
            });
        }
    }
    Ok(())
}

/// Production PCS endpoints. Injectable everywhere they are used.
pub fn pcs_pck_crl_url(base: &str, ca: &str) -> String {
    format!("{base}/sgx/certification/v4/pckcrl?ca={ca}&encoding=der")
}
/// `update=early`: Intel's latest TCB recovery, "commonly the day of public
/// disclosure"; without it PCS answers `standard`, "commonly 12 months after"
/// (Intel PCS API v4), which [`TcbPolicy`] refuses below its floor.
pub fn pcs_tcb_url(base: &str, fmspc_hex: &str) -> String {
    format!("{base}/tdx/certification/v4/tcb?fmspc={fmspc_hex}&update=early")
}
pub fn pcs_qe_identity_url(base: &str) -> String {
    format!("{base}/tdx/certification/v4/qe/identity?update=early")
}
/// The tcbInfo URL of one PCS answer: [`pcs_tcb_url`] for `early`,
/// `update=standard` (what PCS answers without `update`) for `standard`.
pub fn pcs_tcb_url_for(base: &str, fmspc_hex: &str, update: PcsUpdate) -> String {
    match update {
        PcsUpdate::Early => pcs_tcb_url(base, fmspc_hex),
        PcsUpdate::Standard => {
            format!("{base}/tdx/certification/v4/tcb?fmspc={fmspc_hex}&update=standard")
        }
    }
}
/// The QE identity URL of one PCS answer, as [`pcs_tcb_url_for`].
pub fn pcs_qe_identity_url_for(base: &str, update: PcsUpdate) -> String {
    match update {
        PcsUpdate::Early => pcs_qe_identity_url(base),
        PcsUpdate::Standard => format!("{base}/tdx/certification/v4/qe/identity?update=standard"),
    }
}
pub const PCS_BASE_URL: &str = "https://api.trustedservices.intel.com";

// Not when compiled into x0ne-sdk (its build.rs sets `x0ne_sdk`): these tests
// need this crate's fixtures and dev-dependencies.
#[cfg(all(test, not(x0ne_sdk)))]
mod tests {
    use super::*;
    use crate::attest_v2::fixtures::{fresh, years, FakeIntel, STANDARD_EVALUATION};

    /// 2026-09-09, inside every fixture certificate.
    const NOW: u64 = 1_789_000_000;
    const PROD: TcbPolicy = TcbPolicy::production();
    const DAY: u64 = 86_400;

    fn quote(intel: &FakeIntel) -> Vec<u8> {
        intel.quote(&[0x5a; 64], &[7; 48], &[[0; 48]; 4], [0; 8])
    }

    /// `verify_live` on synthetic collateral: `tcb` and `qe` bodies, at `now`.
    fn live(
        intel: &FakeIntel,
        tcb: &str,
        qe: &str,
        now: u64,
        policy: &TcbPolicy,
    ) -> Result<(TcbStatus, TcbStatus), TcbRejected> {
        let chain = intel.collateral_chain_pem();
        verify_live(
            &quote(intel),
            tcb,
            &chain,
            qe,
            &chain,
            &intel.anchor(),
            now,
            policy,
        )
    }

    /// Intel PCS v4 (`Get Revocation List V4`): `ca` is processor or
    /// platform, `encoding` der; the TDX API has no CRL route of its own.
    #[test]
    fn the_pck_crl_url_is_intels_v4_endpoint() {
        assert_eq!(
            pcs_pck_crl_url(PCS_BASE_URL, "platform"),
            "https://api.trustedservices.intel.com/sgx/certification/v4/pckcrl?ca=platform&encoding=der"
        );
    }

    /// Intel PCS v4 `update`: without it PCS answers `standard`, the TCB
    /// recovery "commonly 12 months after public disclosure". Both pieces of
    /// collateral the TCB status rests on ask for `early`, the latest one.
    #[test]
    fn the_tcb_and_qe_identity_urls_ask_for_the_latest_tcb_recovery() {
        assert_eq!(
            pcs_tcb_url(PCS_BASE_URL, "00806f050000"),
            "https://api.trustedservices.intel.com/tdx/certification/v4/tcb?fmspc=00806f050000&update=early"
        );
        assert_eq!(
            pcs_qe_identity_url(PCS_BASE_URL),
            "https://api.trustedservices.intel.com/tdx/certification/v4/qe/identity?update=early"
        );
    }

    /// The documented defaults, pinned: the recovery of 2026-08-11 and a day.
    #[test]
    fn the_production_policy_is_evaluation_22_and_one_day() {
        assert_eq!(
            TcbPolicy::production(),
            TcbPolicy {
                mode: TcbMode::Transparent,
                min_tcb_evaluation_data_number: 22,
                max_collateral_age_secs: 24 * 3600,
            }
        );
        assert_eq!(
            TcbPolicy::strict(),
            TcbPolicy::production().with_mode(TcbMode::Strict)
        );
    }

    /// SECURITY: configuration may raise the floor, never lower it.
    #[test]
    fn the_policy_can_be_tightened_and_not_loosened() {
        assert_eq!(TcbPolicy::tightened(None, None), Ok(PROD));
        let strict = TcbPolicy::tightened(Some(23), Some(3600)).unwrap();
        assert_eq!(
            strict.mode,
            TcbMode::Strict,
            "a floor given refuses below it"
        );
        assert_eq!(
            TcbPolicy::tightened(None, Some(3600)).unwrap().mode,
            TcbMode::Transparent
        );
        assert_eq!(strict.min_tcb_evaluation_data_number, 23);
        assert_eq!(strict.max_collateral_age_secs, 3600);
        let e = TcbPolicy::tightened(Some(21), None).unwrap_err();
        assert!(e.contains("21") && e.contains("22"), "{e}");
        let e = TcbPolicy::tightened(None, Some(2 * DAY)).unwrap_err();
        assert!(e.contains(&(2 * DAY).to_string()), "{e}");
    }

    /// Control: fresh collateral of the minimum evaluation passes.
    #[test]
    fn fresh_collateral_of_the_minimum_evaluation_passes() {
        let intel = FakeIntel::new();
        let issued = fresh(NOW).0;
        let (platform, qe) = live(
            &intel,
            &intel.tcb_info(issued),
            &intel.qe_identity(issued),
            NOW,
            &PROD,
        )
        .expect("fresh, minimum evaluation");
        assert_eq!((platform, qe), (TcbStatus::UpToDate, TcbStatus::UpToDate));
    }

    /// SECURITY (TOB-WAPI-8): Intel's `standard` tcbInfo rates UpToDate what
    /// the recovery of 2026-08-11 rates OutOfDate; below the floor, the
    /// tcbInfo and the QE identity are each refused, whatever they say.
    #[test]
    fn collateral_below_the_minimum_evaluation_is_refused() {
        let intel = FakeIntel::new();
        let issued = fresh(NOW).0;
        let old_tcb = intel.tcb_info_with(issued, STANDARD_EVALUATION, "UpToDate");
        let old_qe = intel.qe_identity_with(issued, STANDARD_EVALUATION);
        let e = live(&intel, &old_tcb, &intel.qe_identity(issued), NOW, &PROD).unwrap_err();
        assert!(
            matches!(
                e,
                TcbRejected::EvaluationBelowMinimum {
                    what: "tcbInfo",
                    have: 20,
                    min: 22
                }
            ),
            "{e}"
        );
        let e = live(&intel, &intel.tcb_info(issued), &old_qe, NOW, &PROD).unwrap_err();
        assert!(
            matches!(
                e,
                TcbRejected::EvaluationBelowMinimum {
                    what: "qeIdentity",
                    have: 20,
                    min: 22
                }
            ),
            "{e}"
        );
        // One below the floor is below the floor; a stricter build refuses 22.
        let e = live(
            &intel,
            &intel.tcb_info_with(issued, 21, "UpToDate"),
            &intel.qe_identity(issued),
            NOW,
            &PROD,
        )
        .unwrap_err();
        assert!(
            matches!(e, TcbRejected::EvaluationBelowMinimum { .. }),
            "{e}"
        );
        let strict = TcbPolicy::tightened(Some(23), None).unwrap();
        let e = live(
            &intel,
            &intel.tcb_info(issued),
            &intel.qe_identity(issued),
            NOW,
            &strict,
        )
        .unwrap_err();
        assert!(
            matches!(
                e,
                TcbRejected::EvaluationBelowMinimum {
                    have: 22,
                    min: 23,
                    ..
                }
            ),
            "{e}"
        );
    }

    /// SECURITY: an OutOfDate platform is refused even under the latest
    /// recovery; the floor adds to the status, it does not replace it.
    #[test]
    fn an_out_of_date_platform_is_refused_at_the_minimum_evaluation() {
        let intel = FakeIntel::new();
        let issued = fresh(NOW).0;
        for status in ["OutOfDate", "Revoked", "OutOfDateConfigurationNeeded"] {
            let tcb = intel.tcb_info_with(issued, PROD.min_tcb_evaluation_data_number, status);
            let e = live(&intel, &tcb, &intel.qe_identity(issued), NOW, &PROD).unwrap_err();
            assert!(
                matches!(&e, TcbRejected::StatusRefused { status: s, .. } if s == status),
                "{status}: {e}"
            );
        }
    }

    /// SECURITY: collateral still inside its 30-day window but issued more
    /// than a day ago is refused, for the tcbInfo and the QE identity each.
    #[test]
    fn collateral_older_than_the_maximum_age_is_refused_inside_its_window() {
        let intel = FakeIntel::new();
        let fresh_at = fresh(NOW).0;
        let old_at = NOW - DAY - 1;
        let e = live(
            &intel,
            &intel.tcb_info(old_at),
            &intel.qe_identity(fresh_at),
            NOW,
            &PROD,
        )
        .unwrap_err();
        assert!(
            matches!(
                e,
                TcbRejected::CollateralTooOld {
                    what: "tcbInfo",
                    age_secs,
                    max_secs: DAY
                } if age_secs == DAY + 1
            ),
            "{e}"
        );
        let e = live(
            &intel,
            &intel.tcb_info(fresh_at),
            &intel.qe_identity(old_at),
            NOW,
            &PROD,
        )
        .unwrap_err();
        assert!(
            matches!(
                e,
                TcbRejected::CollateralTooOld {
                    what: "qeIdentity",
                    ..
                }
            ),
            "{e}"
        );
        // Exactly a day old is still accepted.
        let edge = NOW - DAY;
        live(
            &intel,
            &intel.tcb_info(edge),
            &intel.qe_identity(edge),
            NOW,
            &PROD,
        )
        .expect("a day old");
    }

    /// SECURITY: collateral past its own `nextUpdate` is refused by the
    /// window, before any policy.
    #[test]
    fn expired_collateral_is_refused_by_its_window() {
        let intel = FakeIntel::new();
        let issued = NOW - 31 * DAY;
        let loose = TcbPolicy {
            max_collateral_age_secs: u64::MAX,
            ..PROD
        };
        let e = live(
            &intel,
            &intel.tcb_info(issued),
            &intel.qe_identity(fresh(NOW).0),
            NOW,
            &loose,
        )
        .unwrap_err();
        assert!(e.to_string().contains("validity window"), "{e}");
    }

    #[test]
    fn the_crl_is_named_by_the_pcks_issuer_and_an_unknown_issuer_refuses() {
        let intel = FakeIntel::new();
        assert_eq!(pck_crl_ca(&quote(&intel)).unwrap(), "platform");
        // The pcs fixture's intermediate is not an Intel PCK CA.
        let other = crate::pcs::tests::fixture_for_composite();
        let e = pck_crl_ca(&other.quote).unwrap_err().to_string();
        assert!(e.contains("neither Intel SGX PCK Platform CA"), "{e}");
    }

    #[test]
    fn a_listed_pck_is_revoked_and_a_prefix_of_its_serial_is_not() {
        let intel = FakeIntel::new();
        let q = quote(&intel);
        let serial = intel.pck_serial();
        let listed = intel.pck_crl(std::slice::from_ref(&serial), fresh(NOW));
        let e = verify_pck_crl(&q, &listed, &intel.anchor(), NOW, &PROD).unwrap_err();
        assert!(matches!(e, TcbRejected::PckRevoked { .. }), "{e}");
        let prefix = intel.pck_crl(&[serial[..15].to_vec(), vec![0x01]], fresh(NOW));
        verify_pck_crl(&q, &prefix, &intel.anchor(), NOW, &PROD).expect("another serial");
        let empty = intel.pck_crl(&[], fresh(NOW));
        verify_pck_crl(&q, &empty, &intel.anchor(), NOW, &PROD).expect("an empty CRL");
    }

    /// SECURITY: a CRL past its `nextUpdate` (expired) or before its
    /// `thisUpdate` is refused; so is one inside its window but issued more
    /// than a day ago, which a stale cache would serve.
    #[test]
    fn an_expired_or_day_old_crl_is_refused() {
        let intel = FakeIntel::new();
        let q = quote(&intel);
        let expired = intel.pck_crl(&[], (NOW - 31 * DAY, NOW - DAY));
        let e = verify_pck_crl(&q, &expired, &intel.anchor(), NOW, &PROD).unwrap_err();
        assert!(
            matches!(&e, TcbRejected::PckCrl(m) if m.contains("validity window")),
            "{e}"
        );
        let early = intel.pck_crl(&[], years(2039, 2040));
        let e = verify_pck_crl(&q, &early, &intel.anchor(), NOW, &PROD).unwrap_err();
        assert!(e.to_string().contains("validity window"), "{e}");
        let day_old = intel.pck_crl(&[], (NOW - DAY - 1, NOW + 29 * DAY));
        let e = verify_pck_crl(&q, &day_old, &intel.anchor(), NOW, &PROD).unwrap_err();
        assert!(
            matches!(
                e,
                TcbRejected::CollateralTooOld {
                    what: "PCK CRL",
                    max_secs: DAY,
                    ..
                }
            ),
            "{e}"
        );
        let edge = intel.pck_crl(&[], (NOW - DAY, NOW + 29 * DAY));
        verify_pck_crl(&q, &edge, &intel.anchor(), NOW, &PROD).expect("a day old");
    }

    /// A critical extension this verifier does not read (here an issuing
    /// distribution point: a partial CRL) makes the list unusable.
    #[test]
    fn a_crl_with_a_critical_extension_is_refused() {
        let intel = FakeIntel::new();
        let partial = intel.crl_with(intel.pck_ca_key_der(), &[], fresh(NOW), |p| {
            p.issuing_distribution_point = Some(rcgen::CrlIssuingDistributionPoint {
                distribution_point: rcgen::CrlDistributionPoint {
                    uris: vec!["https://example.invalid/part1.crl".into()],
                },
                scope: Some(rcgen::CrlScope::UserCertsOnly),
            });
        });
        let e = verify_pck_crl(&quote(&intel), &partial, &intel.anchor(), NOW, &PROD).unwrap_err();
        assert!(
            matches!(&e, TcbRejected::PckCrl(m) if m.contains("critical")),
            "{e}"
        );
    }

    #[test]
    fn a_crl_with_trailing_bytes_or_under_another_root_is_refused() {
        let intel = FakeIntel::new();
        let q = quote(&intel);
        let mut crl = intel.pck_crl(&[], fresh(NOW));
        verify_pck_crl(&q, &crl, &intel.anchor(), NOW, &PROD).expect("control");
        let e = verify_pck_crl(&q, &crl, &TrustAnchor::intel_production(), NOW, &PROD).unwrap_err();
        assert!(e.to_string().contains("PCK chain"), "{e}");
        crl.push(0);
        let e = verify_pck_crl(&q, &crl, &intel.anchor(), NOW, &PROD).unwrap_err();
        assert!(e.to_string().contains("after the CRL"), "{e}");
    }

    /// A date whose 20 bytes hold a multi-byte character where digits belong
    /// is refused: slicing it by byte offsets panicked.
    #[test]
    fn a_date_with_a_multi_byte_character_is_refused_not_a_panic() {
        for s in [
            "2026-10-22T1\u{e9}00:00Z",
            "202\u{e9}-0-22T10:00:00Z",
            "2026-10-22T10:0\u{e9}:0Z",
        ] {
            assert_eq!(s.len(), 20, "{s}");
            assert_eq!(date_unix(s), None, "{s}");
        }
        assert_eq!(date_unix("1970-01-02T00:00:00Z"), Some(86_400));
    }

    /// `date_unix` on each date: all refused.
    fn refused(dates: &[&str]) {
        for s in dates {
            assert_eq!(s.len(), 20, "{s}");
            assert_eq!(date_unix(s), None, "{s}");
        }
        // The same shape with every field in range reads.
        assert_eq!(date_unix("2026-10-22T10:00:00Z"), Some(1_792_663_200));
    }

    /// A year with a sign is refused: `str::parse` read `+202` as 202.
    #[test]
    fn a_date_whose_year_has_a_sign_is_refused() {
        refused(&["+202-10-22T10:00:00Z", "-202-10-22T10:00:00Z"]);
    }

    /// A month with a sign, 0 or past 12 is refused.
    #[test]
    fn a_date_whose_month_is_signed_or_out_of_range_is_refused() {
        refused(&[
            "2026-+1-22T10:00:00Z",
            "2026--1-22T10:00:00Z",
            "2026-00-22T10:00:00Z",
            "2026-13-22T10:00:00Z",
        ]);
        assert_eq!(date_unix("2026-12-22T10:00:00Z"), Some(1_797_933_600));
    }

    /// A day with a sign, 0 or past 31 is refused.
    #[test]
    fn a_date_whose_day_is_signed_or_out_of_range_is_refused() {
        refused(&[
            "2026-10-+1T10:00:00Z",
            "2026-10--1T10:00:00Z",
            "2026-10-00T10:00:00Z",
            "2026-10-32T10:00:00Z",
        ]);
        assert_eq!(date_unix("2026-10-31T10:00:00Z"), Some(1_793_440_800));
    }

    /// An hour with a sign or past 23 is refused.
    #[test]
    fn a_date_whose_hour_is_signed_or_out_of_range_is_refused() {
        refused(&[
            "2026-10-22T+1:00:00Z",
            "2026-10-22T-1:00:00Z",
            "2026-10-22T24:00:00Z",
        ]);
        assert_eq!(date_unix("2026-10-22T23:00:00Z"), Some(1_792_710_000));
    }

    /// A minute with a sign or past 59 is refused.
    #[test]
    fn a_date_whose_minute_is_signed_or_out_of_range_is_refused() {
        refused(&[
            "2026-10-22T10:+1:00Z",
            "2026-10-22T10:-1:00Z",
            "2026-10-22T10:60:00Z",
        ]);
        assert_eq!(date_unix("2026-10-22T10:59:00Z"), Some(1_792_666_740));
    }

    /// A second with a sign or past 60 is refused; 60 is a leap second.
    #[test]
    fn a_date_whose_second_is_signed_or_out_of_range_is_refused() {
        refused(&[
            "2026-10-22T10:00:+1Z",
            "2026-10-22T10:00:-1Z",
            "2026-10-22T10:00:61Z",
        ]);
        assert_eq!(date_unix("2026-10-22T10:00:60Z"), Some(1_792_663_260));
    }

    /// The field reader takes digits alone, one to its length.
    #[test]
    fn a_date_field_is_digits_alone() {
        assert_eq!(date_field("07", 2), Some(7));
        assert_eq!(date_field("2026", 4), Some(2026));
        for (f, n) in [
            ("", 2),
            ("+7", 2),
            ("-7", 2),
            ("123", 2),
            (" 7", 2),
            ("7 ", 2),
            ("٣", 2),
        ] {
            assert_eq!(date_field(f, n), None, "{f:?}");
        }
    }

    /// SECURITY: Intel's identity JSON is parsed before its signature can
    /// vouch for it. A hex field with a multibyte character, an odd length or
    /// a non-digit is refused with a message, never a panic (before
    /// 2026-10-23 "€0" panicked the str slicing of the decoder).
    #[test]
    fn a_malformed_hex_field_of_the_qe_identity_is_refused_not_a_panic() {
        let report = [0u8; 384];
        for bad in ["€0", "€", "0", "0g", "+f"] {
            let identity = serde_json::json!({"id": "TD_QE", "version": 2, "miscselect": bad});
            let e = qe_status(&identity, &report).unwrap_err();
            assert!(
                e.to_string().contains("miscselect is not hex"),
                "{bad:?}: {e}"
            );
        }
        // Intel publishes uppercase hex: accepted, and the next field is reached.
        let identity = serde_json::json!({
            "id": "TD_QE", "version": 2, "miscselect": "0000000A", "miscselectMask": "€€"
        });
        let e = qe_status(&identity, &report).unwrap_err();
        assert!(e.to_string().contains("miscselectMask is not hex"), "{e}");
    }

    // ── root CA CRL ─────────────────────────────────────────────────────────

    /// `verify_root_crl` on this platform's quote and both collateral chains.
    fn root(intel: &FakeIntel, q: &[u8], crl: &[u8]) -> Result<(), TcbRejected> {
        let chain = intel.collateral_chain_pem();
        verify_root_crl(q, &[&chain, &chain], crl, &intel.anchor(), NOW)
    }

    /// 2026 to 2027, around `NOW`: a year, as Intel issues it.
    const YEAR: (u64, u64) = (NOW - 180 * DAY, NOW + 185 * DAY);

    /// SECURITY: the root CA CRL revokes what the root issued: the PCK CA
    /// behind the quote's PCK and the TCB signing certificate behind the
    /// collateral. Either listed is a revocation; an empty list, or a leaf's
    /// serial on it (the PCK CRL's business), passes.
    #[test]
    fn a_root_crl_listing_the_pck_ca_or_the_tcb_signer_revokes_and_an_empty_one_passes() {
        let intel = FakeIntel::new();
        let q = quote(&intel);
        root(&intel, &q, &intel.root_crl(&[], YEAR)).expect("empty, inside its window");
        for (what, serial) in [
            ("PCK CA", intel.pck_ca_serial()),
            ("TCB signer", intel.tcb_signing_serial()),
        ] {
            let listed = intel.root_crl(&[vec![0x77; 16], serial], YEAR);
            let e = root(&intel, &q, &listed).unwrap_err();
            assert!(matches!(e, TcbRejected::CaRevoked { .. }), "{what}: {e}");
            assert!(e.to_string().contains("revoked"), "{what}: {e}");
        }
        let leaf = intel.root_crl(&[intel.pck_serial()], YEAR);
        root(&intel, &q, &leaf).expect("a leaf's serial names nothing the root issued");
    }

    /// The root CRL is judged by its own window, not by the collateral age
    /// limit: Intel issues it for a year (2026-02-26 to 2027-02-26, read on
    /// 2026-10-23), so a months-old one inside its window is the current one.
    #[test]
    fn a_root_crl_outside_its_window_is_refused_and_a_months_old_one_inside_it_passes() {
        let intel = FakeIntel::new();
        let q = quote(&intel);
        for (name, window) in [
            ("expired", (NOW - 400 * DAY, NOW - DAY)),
            ("not yet valid", (NOW + DAY, NOW + 366 * DAY)),
        ] {
            let e = root(&intel, &q, &intel.root_crl(&[], window)).unwrap_err();
            assert!(
                matches!(&e, TcbRejected::RootCrl(m) if m.contains("validity window")),
                "{name}: {e}"
            );
        }
        root(
            &intel,
            &q,
            &intel.root_crl(&[], (NOW - 240 * DAY, NOW + 125 * DAY)),
        )
        .expect("eight months old, inside its year");
    }

    /// SECURITY: a root CRL signed by the PCK CA's key or another key, with
    /// trailing bytes, a critical extension, not a CRL, or under another
    /// root is unusable, not a revocation.
    #[test]
    fn a_root_crl_not_signed_by_the_root_or_malformed_is_refused() {
        let intel = FakeIntel::new();
        let q = quote(&intel);
        let by_pck_ca = intel.root_crl_with(intel.pck_ca_key_der(), &[], YEAR, |_| {});
        let e = root(&intel, &q, &by_pck_ca).unwrap_err();
        assert!(
            matches!(&e, TcbRejected::RootCrl(m) if m.contains("signature")),
            "{e}"
        );
        let other = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)
            .unwrap()
            .serialize_der();
        let e = root(&intel, &q, &intel.root_crl_with(&other, &[], YEAR, |_| {})).unwrap_err();
        assert!(matches!(e, TcbRejected::RootCrl(_)), "{e}");
        let mut trailing = intel.root_crl(&[], YEAR);
        trailing.push(0);
        let e = root(&intel, &q, &trailing).unwrap_err();
        assert!(e.to_string().contains("after the CRL"), "{e}");
        let partial = intel.root_crl_with(intel.root_key_der(), &[], YEAR, |p| {
            p.issuing_distribution_point = Some(rcgen::CrlIssuingDistributionPoint {
                distribution_point: rcgen::CrlDistributionPoint {
                    uris: vec!["https://example.invalid/part1.crl".into()],
                },
                scope: Some(rcgen::CrlScope::CaCertsOnly),
            });
        });
        let e = root(&intel, &q, &partial).unwrap_err();
        assert!(
            matches!(&e, TcbRejected::RootCrl(m) if m.contains("critical")),
            "{e}"
        );
        let e = root(&intel, &q, b"not a CRL").unwrap_err();
        assert!(matches!(e, TcbRejected::RootCrl(_)), "{e}");
        let chain = intel.collateral_chain_pem();
        let e = verify_root_crl(
            &q,
            &[&chain],
            &intel.root_crl(&[], YEAR),
            &TrustAnchor::intel_production(),
            NOW,
        )
        .unwrap_err();
        assert!(e.to_string().contains("PCK chain"), "{e}");
    }

    /// SECURITY: a TCB signer issued by a PCK whose
    /// key was extracted signs an UpToDate tcbInfo; every signature verifies.
    /// Until then `verify_live` took it (UpToDate, UpToDate); refused alone now.
    #[test]
    fn collateral_signed_under_an_extracted_pck_is_refused() {
        let intel = FakeIntel::new();
        let (tcb, qe, chain) = intel.collateral_under_extracted_pck(NOW - 3600);
        let e = verify_live(
            &quote(&intel),
            &tcb,
            &chain,
            &qe,
            &chain,
            &intel.anchor(),
            NOW,
            &PROD,
        )
        .unwrap_err();
        assert!(
            e.to_string().contains("issuer chain has 4 certificate(s)"),
            "{e}"
        );
        // The same bodies under Intel's chain: only the chain is refused.
        assert!(live(
            &intel,
            &intel.tcb_info(NOW - 3600),
            &intel.qe_identity(NOW - 3600),
            NOW,
            &PROD
        )
        .is_ok());
    }

    /// SECURITY (C1): beyond its signatures, the collateral chain has
    /// Intel's shape, as the PCK chain (`pcs::intel_shape`): the root a
    /// CA, the signer naming the root as its issuer.
    #[test]
    fn a_collateral_chain_of_another_shape_than_intels_is_refused() {
        let intel = FakeIntel::new();
        let (tcb, qe, chain) = intel.collateral_under_another_issuer_name(NOW - 3600);
        let e = verify_live(
            &quote(&intel),
            &tcb,
            &chain,
            &qe,
            &chain,
            &intel.anchor(),
            NOW,
            &PROD,
        )
        .unwrap_err();
        assert!(e.to_string().contains("chain link 0: its issuer"), "{e}");
        let (tcb, qe, chain, root) = intel.collateral_under_a_root_that_is_no_ca(NOW - 3600);
        let e = verify_live(
            &quote(&intel),
            &tcb,
            &chain,
            &qe,
            &chain,
            &TrustAnchor::RootDer(root),
            NOW,
            &PROD,
        )
        .unwrap_err();
        assert!(e.to_string().contains("the root is not a CA"), "{e}");
    }

    /// A collateral signer the root did not issue is refused before any
    /// list is read: the root's CRL cannot speak for it.
    #[test]
    fn a_collateral_signer_not_issued_by_the_root_is_refused() {
        let intel = FakeIntel::new();
        let other = FakeIntel::new();
        let q = quote(&intel);
        let crl = intel.root_crl(&[], YEAR);
        let foreign = other.collateral_chain_pem();
        let e = verify_root_crl(&q, &[&foreign], &crl, &intel.anchor(), NOW).unwrap_err();
        assert!(
            matches!(&e, TcbRejected::RootCrl(m) if m.contains("signer")),
            "{e}"
        );
        let e = verify_root_crl(&q, &[""], &crl, &intel.anchor(), NOW).unwrap_err();
        assert!(matches!(e, TcbRejected::RootCrl(_)), "{e}");
    }

    // Intel PCS production answers of 2026-10-28 for FMSPC 00806f050000, the
    // GCE a3-highgpu hosts a strict TD refused that night (exit 76, "TCB
    // status OutOfDate ... level dated 2026-02-11"). tests/VENDORED.md.
    const EARLY_2026_10_28: &str =
        include_str!("../tests/pcs_live/2026-10-28/tcb_info_00806f050000_early.json");
    const STANDARD_2026_10_28: &str =
        include_str!("../tests/pcs_live/2026-10-28/tcb_info_00806f050000_standard.json");
    /// 2026-10-28T00:56:58Z, the `Date` of the standard answer.
    const NOW_2026_10_28: u64 = 1_793_149_018;

    fn level_of(body: &str, date: &str) -> serde_json::Value {
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        v["tcbInfo"]["tcbLevels"]
            .as_array()
            .unwrap()
            .iter()
            .find(|l| l["tcbDate"] == date)
            .cloned()
            .unwrap_or_else(|| panic!("no level of {date}"))
    }

    /// 2026-10-28, on Intel's signed bytes: a platform at the SVNs of the early
    /// level of 2026-02-11 is accepted on `standard` (evaluation 20, its
    /// newest level, 2025-08-13, UpToDate) and named on `early`
    /// (evaluation 22, OutOfDate, three advisories). Its quote was not
    /// kept: the TDX module and the QE are not judged here.
    #[test]
    fn the_gce_hosts_of_2026_10_28_on_intels_two_answers_of_that_day() {
        let anchor = TrustAnchor::intel_production();
        let chain = include_str!("../tests/pcs_live/tcb_issuer_chain.pem");
        let collateral = |body, policy: &TcbPolicy| {
            verify_collateral(
                "tcbInfo",
                body,
                "tcbInfo",
                chain,
                &anchor,
                NOW_2026_10_28,
                policy,
            )
        };
        let strict = TcbPolicy::strict();
        let early =
            collateral(EARLY_2026_10_28, &strict).expect("Intel's signature, evaluation 22");
        let e = collateral(STANDARD_2026_10_28, &strict).unwrap_err();
        assert!(
            matches!(
                e,
                TcbRejected::EvaluationBelowMinimum {
                    have: 20,
                    min: 22,
                    ..
                }
            ),
            "{e}"
        );
        let standard = collateral(STANDARD_2026_10_28, &TcbPolicy::production().on_standard())
            .expect("Intel's signature, evaluation 20");
        let named = level_of(EARLY_2026_10_28, "2026-02-11T00:00:00Z");
        let svns = |key: &str| -> [u8; 16] { comps(&named, key).unwrap().try_into().unwrap() };
        let pck = PckTcb {
            fmspc_hex: "00806f050000".into(),
            cpusvn: svns("sgxtcbcomponents"),
            pcesvn: named["tcb"]["pcesvn"].as_u64().unwrap() as u16,
        };
        let tee = svns("tdxtcbcomponents");
        let latest = tdx_tcb_level(&early, &pck, &tee).unwrap();
        assert_eq!(
            (latest.status.clone(), latest.tcb_date.as_str()),
            (TcbStatus::OutOfDate, "2026-02-11T00:00:00Z")
        );
        assert_eq!(
            latest.advisories,
            ["INTEL-DOC-10003", "INTEL-SA-01439", "INTEL-SA-01442"]
        );
        assert!(!latest.status.accepted());
        let accepted = tdx_tcb_level(&standard, &pck, &tee).unwrap();
        assert_eq!(
            (accepted.status.clone(), accepted.tcb_date.as_str()),
            (TcbStatus::UpToDate, "2025-08-13T00:00:00Z")
        );
        assert!(accepted.status.accepted() && accepted.advisories.is_empty());
        // Every platform at that early level: the newest standard level asks
        // exactly its SVNs.
        assert_eq!(
            level_of(STANDARD_2026_10_28, "2025-08-13T00:00:00Z")["tcb"],
            named["tcb"]
        );
    }
}
