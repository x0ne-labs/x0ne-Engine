//! The client's verdict on a cloud TD, computed on the client's machine from
//! the attestation document and its own anchors, in order:
//! clock, config, binding, quote and live Intel collateral, Google firmware
//! endorsement, event log against the image reference, GPU, model.
//!
//! Nothing but `GET /v2/attestation` has been sent when this runs, and
//! nothing but `GET /v2/attestation/session` is sent unless it returns `Ok`
//! and that connection's session quote (and, with the GPU product, its GPU
//! evidence) passes [`SessionCheck`](crate::remote_verify::SessionCheck).
//!
//! Under an SEV-SNP reference ([`Pinned::Snp`](crate::remote_verify::Pinned::Snp)) the order is clock and AMD
//! CRL, config, binding, SEV-SNP report against the reference's policy and
//! predicted launch digests, GPU, model
//! ([`verify_snp_document`](crate::remote_verify::verify_snp_document)); a session
//! pins one TEE and never falls back to the other.

use crate::attest_v2::{unb64, Document};
use crate::binding::BindingV2;
use crate::cloud_config::{hex, unhex32, Loaded};
use crate::pcs::TrustAnchor;
use crate::tls::{ServerPin, TlsIdentity};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use x0ne_confidential::model::{AttestationEvidence, EvidenceFormat, Verdict, VerifierStrength};
use x0ne_confidential::policy::VerificationPolicy;

/// The clock may differ from Intel's by this much; more is a wrong clock.
pub const CLOCK_SKEW_SECS: u64 = 300;
const PCS_TIMEOUT_SECS: u64 = 20;
const MAX_COLLATERAL_BYTES: usize = 4 * 1024 * 1024;

/// The firmware line of [`LIMITATIONS`]: what pinning RTMR0 to the endorsed
/// binary (`rtmr0.policy` `gce-endorsed`, the only policy admitted) leaves.
pub const FIRMWARE_LIMITATION_GCE_ENDORSED: &str = "Google's firmware: the binary Google signed is the one this TD measured (MRTD recomputed here), but its source is not published and its code is not audited; RTMR0 is pinned to the reference's host events (TD HOB, Secure Boot variables, ACPI tables, boot variables) as one boot showed them, not audited; Google's signing key can endorse any binary";

/// What the client does not establish, said in every verdict.
pub const LIMITATIONS: &[&str] = &[
    "this machine: software in the same session can read the local key and the questions",
    FIRMWARE_LIMITATION_GCE_ENDORSED,
    "revocations: the TD's PCK certificate is checked against Intel's PCK CRL, and its PCK CA and Intel's TCB signing certificate against Intel's root CA CRL, both fetched live at each verdict (the root CA CRL is issued for a year and judged by its own window); the NRAS signing chain's revocation is judged from the OCSP responses the document staples, under the policy the verdict names, a verified revoked refusing under every policy; no TD staples them yet, and the verdict says so",
    "the GPU verdict is NVIDIA's NRAS service; each connection's GPU evidence answers a nonce derived from that connection's TLS exporter and TDX quote, so the TD holding this TLS session asked for it, but CPU and GPU are bound by the TD's software, not by hardware (Hopper's GPU report names no VM): the link falls with the CPU TEE (TEE.fail)",
    "physical attacks on memory and side channels are out of scope",
    "a compromise of the Intel, NVIDIA or Google roots defeats this verdict",
    "availability: Google may stop a spot VM; it sees timing, sizes, IP addresses and the bill",
    "the image reference (the installed one or your --reference) is checked against the x0ne archive key pinned in this build: whoever holds its signing subkey can sign any image, and a revocation reaches you only with a new build",
    "the prompt commitment in a receipt is not opened by the relay (the chat template is applied in the TD)",
    "a streamed answer's text is passed on before its receipt is verified: only text followed by a final chunk with receipt_verified is covered, and text kept above an error is unverified",
    "the model's own behaviour",
];

/// The client's anchors. Production values only through [`Anchors::production`].
pub struct Anchors {
    pub tdx: TrustAnchor,
    pub nras_intermediate_pins: Vec<crate::nras::NrasIntermediatePin>,
    pub gce_root_der: Vec<u8>,
    pub pcs_base_url: String,
    /// Where the pinned Intel root publishes its CRL (its CRL distribution point).
    pub root_crl_url: String,
    pub gce_bucket_base_url: String,
    /// The minimum TCB evaluation and the maximum age of Intel's collateral.
    pub tcb_policy: crate::tcb::TcbPolicy,
    /// How the NRAS signing chain's revocation is judged from the OCSP
    /// responses the document staples (`nras.ocsp_b64`): `transparent` in
    /// production, named in every verdict; a verified revoked refuses either way.
    pub nras_ocsp: crate::nras_ocsp::NrasOcspPolicy,
    /// The pinned AMD roots and the KDS that serves their CRLs (SEV-SNP).
    pub snp: crate::snp::SnpAnchors,
}

impl Anchors {
    pub fn production() -> Self {
        Self {
            tdx: TrustAnchor::intel_production(),
            nras_intermediate_pins: crate::nras::NVIDIA_NRAS_GPU_INTERMEDIATE_PINS.to_vec(),
            gce_root_der: crate::gce_endorsement::ROOT_1_DER.to_vec(),
            pcs_base_url: crate::tcb::PCS_BASE_URL.into(),
            root_crl_url: crate::tcb::INTEL_SGX_ROOT_CA_CRL_URL.into(),
            gce_bucket_base_url: crate::gce_endorsement::BUCKET_BASE_URL.into(),
            tcb_policy: crate::tcb::TcbPolicy::production(),
            nras_ocsp: crate::nras_ocsp::NrasOcspPolicy::production(),
            snp: crate::snp::SnpAnchors::production(),
        }
    }

    /// SHA-256 of the Google root's SPKI: what a reference names in `mrtd.root_spki_sha256`.
    pub fn gce_root_spki_sha256_hex(&self) -> Result<String, String> {
        crate::tls::spki_sha256_of_cert(&self.gce_root_der).map(|d| hex(&d))
    }
}

/// What this session expects the TD to prove.
pub struct Expectations<'a> {
    pub config: &'a Loaded,
    /// The SPKI of the server this connection shook hands with.
    pub tls_spki_sha256: [u8; 32],
    pub machine_type: &'a str,
    pub reference: &'a crate::reference::Loaded,
    pub catalog_entry: &'a crate::registry::Entry,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Refusal {
    /// 65 measurement or verdict, 66 not cryptographic, 75 NRAS, 76 Intel
    /// or Google unreachable, 77 model, 87 clock, 88 channel not bound,
    /// 89 reference unusable, 104 reference reporting RTMR0 instead of
    /// pinning it (`reference::EXIT_RTMR0_UNPINNED`).
    pub code: i32,
    pub reason: String,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (exit {})", self.reason, self.code)
    }
}

/// What was established, for the status page and the audit file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub verdict: serde_json::Value,
    pub environment: String,
    pub session_pubkey_hex: String,
    pub model_name: String,
    pub model_digest_hex: String,
    pub tcb_platform: String,
    pub tcb_qe: String,
    pub endorsement_signer: String,
    pub endorsement_timestamp_unix: u64,
    pub firmware_svn: u32,
    pub mrtd_hex: String,
    pub image_name: String,
    pub image_version: String,
    pub rtmr0_hex: String,
    pub rtmr0_known: Option<bool>,
    /// RTMR0 pinned (reference policy `gce-endorsed`), not only reported.
    #[serde(default)]
    pub rtmr0_pinned: bool,
    /// SHA-384 of the firmware binary this client fetched, checked against
    /// Google's endorsement and measured (empty in an SEV-SNP verdict).
    #[serde(default)]
    pub firmware_sha384_hex: String,
    pub sections: Vec<String>,
    pub gpu_count: usize,
    pub reference_path: String,
    pub reference_sha256_hex: String,
    pub reference_explicit: bool,
    /// The archive subkey that signed the reference, and when.
    #[serde(default)]
    pub reference_signer_subkey: String,
    #[serde(default)]
    pub reference_signed_at_unix: u64,
    pub summary: Vec<String>,
    pub limitations: Vec<String>,
    /// SEV-SNP sessions only; the TDX fields above then say "not
    /// applicable" or stay empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snp: Option<SnpView>,
    /// What this verdict says about the platform's patch level: Intel's
    /// latest collateral or AMD's bulletins (`tcb_disclosure`, 2026-10-28).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tcb: Option<crate::tcb_disclosure::TcbDisclosure>,
}

fn refuse(code: i32, reason: impl Into<String>) -> Refusal {
    Refusal {
        code,
        reason: reason.into(),
    }
}

// ── HTTP dates ───────────────────────────────────────────────────────────────

const DAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `(year, month, day, hour, minute, second)` in UTC.
pub fn civil_from_unix(unix: u64) -> (i64, i64, i64, u64, u64, u64) {
    let days = (unix / 86400) as i64;
    let secs = unix % 86400;
    let (y, m, d) = civil_from_days(days);
    (y, m, d, secs / 3600, secs % 3600 / 60, secs % 60)
}

/// `2026-10-12T21-09Z`: a file-name-safe UTC stamp to the minute.
pub fn file_stamp(unix: u64) -> String {
    let (y, m, d, hh, mm, _) = civil_from_unix(unix);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}-{mm:02}Z")
}

/// RFC 7231 IMF-fixdate, `Sun, 06 Nov 1994 08:49:37 GMT`.
pub fn http_date(unix: u64) -> String {
    let days = (unix / 86400) as i64;
    let secs = unix % 86400;
    let (y, m, d) = civil_from_days(days);
    format!(
        "{}, {:02} {} {:04} {:02}:{:02}:{:02} GMT",
        DAYS[(days.rem_euclid(7)) as usize],
        d,
        MONTHS[(m - 1) as usize],
        y,
        secs / 3600,
        secs % 3600 / 60,
        secs % 60
    )
}

pub fn parse_http_date(s: &str) -> Option<u64> {
    let (_, rest) = s.trim().split_once(", ")?;
    let mut it = rest.split(' ');
    let d: i64 = it.next()?.parse().ok()?;
    let month = it.next()?;
    let m = MONTHS.iter().position(|x| *x == month)? as i64 + 1;
    let y: i64 = it.next()?.parse().ok()?;
    let mut t = it.next()?.split(':');
    let hh: i64 = t.next()?.parse().ok()?;
    let mm: i64 = t.next()?.parse().ok()?;
    let ss: i64 = t.next()?.parse().ok()?;
    if it.next()? != "GMT"
        || it.next().is_some()
        || !(1..=31).contains(&d)
        || hh > 23
        || mm > 59
        || ss > 60
    {
        return None;
    }
    u64::try_from(days_from_civil(y, m, d) * 86400 + hh * 3600 + mm * 60 + ss).ok()
}

// ── Intel PCS, fetched by the client ─────────────────────────────────────────

pub struct Collateral {
    pub body: String,
    pub chain_pem: String,
    pub date_unix: Option<u64>,
}

fn percent_decode(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' => {
                out.push(u8::from_str_radix(s.get(i + 1..i + 3)?, 16).ok()?);
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

pub async fn fetch_pcs(http: &reqwest::Client, url: &str) -> Result<Collateral, String> {
    let resp = http
        .get(url)
        .timeout(std::time::Duration::from_secs(PCS_TIMEOUT_SECS))
        .send()
        .await
        .map_err(|e| format!("GET {url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("PCS answered {} for {url}", resp.status()));
    }
    let date_unix = resp
        .headers()
        .get(reqwest::header::DATE)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_http_date);
    let chain_encoded = ["TCB-Info-Issuer-Chain", "SGX-Enclave-Identity-Issuer-Chain"]
        .iter()
        .find_map(|h| resp.headers().get(*h))
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .ok_or("PCS answer carries no issuer-chain header")?;
    let chain_pem = percent_decode(&chain_encoded)
        .ok_or("issuer-chain header is not valid percent-encoding")?;
    let body = crate::http_body::read_capped(resp, MAX_COLLATERAL_BYTES)
        .await
        .map_err(|e| format!("PCS body: {e}"))?;
    Ok(Collateral {
        body: String::from_utf8(body).map_err(|_| "PCS body is not UTF-8")?,
        chain_pem,
        date_unix,
    })
}

/// Intel's `early` tcbInfo and QE identity for a transparent verdict, or
/// why they could not be fetched: named in the verdict, never a refusal.
/// Where [`sdk_collateral`] fetches: the relay's own [`Anchors`] URLs.
#[derive(Debug, Clone)]
pub struct CollateralSources {
    pub pcs_base_url: String,
    pub root_crl_url: String,
    pub gce_bucket_base_url: String,
}

/// The collateral an SDK verifies a key document of this session's TD with
/// (`x0ne_sdk::attest::Collateral`), fetched as the relay fetches its own:
/// Intel's tcbInfo and QE identity, both `standard` and `early`, the PCK
/// CRL, the root CA's CRL, Google's launch endorsement of the quote's MRTD
/// and the firmware binary it names. All signed by Intel or Google, or
/// named by what Google signed: the SDK checks them, not this relay. An
/// answer Intel did not give is left empty, for the SDK's policy to judge;
/// the CRLs, the endorsement and the binary are required.
pub async fn sdk_collateral(
    http: &reqwest::Client,
    sources: &CollateralSources,
    quote: &[u8],
) -> Result<x0ne_sdk::attest::Collateral, String> {
    use crate::tcb::PcsUpdate::{Early, Standard};
    use base64::Engine as _;
    let anchors = sources;
    let base = anchors.pcs_base_url.as_str();
    let fmspc = crate::tcb::pck_tcb(quote)
        .map_err(|e| format!("platform identity: {e}"))?
        .fmspc_hex;
    let fmspc = fmspc.as_str();
    let ca = crate::tcb::pck_crl_ca(quote).map_err(|e| format!("platform identity: {e}"))?;
    let mrtd = x0ne_confidential::tdx_quote::parse(quote)
        .map_err(|e| format!("TDX quote: {e:?}"))?
        .mr_td;
    let pair = |update| async move {
        let t = fetch_pcs(http, &crate::tcb::pcs_tcb_url_for(base, fmspc, update)).await;
        let q = fetch_pcs(http, &crate::tcb::pcs_qe_identity_url_for(base, update)).await;
        match (t, q) {
            (Ok(t), Ok(q)) => (t.body, t.chain_pem, q.body, q.chain_pem),
            _ => Default::default(),
        }
    };
    let (tcb_info, tcb_info_chain_pem, qe_identity, qe_identity_chain_pem) = pair(Early).await;
    let (
        tcb_info_standard,
        tcb_info_standard_chain_pem,
        qe_identity_standard,
        qe_identity_standard_chain_pem,
    ) = pair(Standard).await;
    let b64 = |b: &[u8]| base64::engine::general_purpose::STANDARD.encode(b);
    let pck_crl = fetch_pcs_der(http, &crate::tcb::pcs_pck_crl_url(base, ca))
        .await
        .map_err(|e| format!("PCK CRL: {e}"))?;
    let root_crl = fetch_pcs_der(http, &anchors.root_crl_url)
        .await
        .map_err(|e| format!("root CA CRL: {e}"))?;
    let endorsement = crate::gce_endorsement::fetch(http, &anchors.gce_bucket_base_url, &mrtd)
        .await
        .map_err(|e| format!("Google endorsement: {e}"))?
        .unwrap_or_default();
    // The binary the endorsement names, which the SDK measures itself; an
    // endorsement that names none, or a bucket that lacks it, is left empty
    // for the SDK to refuse, as it refuses the endorsement.
    let firmware = match crate::gce_endorsement::uefi_sha384_hex_unverified(&endorsement) {
        Ok(digest) if !digest.is_empty() => {
            crate::tdvf::fetch(http, &anchors.gce_bucket_base_url, &digest)
                .await
                .map_err(|e| format!("Google firmware binary: {e}"))?
                .unwrap_or_default()
        }
        _ => Vec::new(),
    };
    Ok(x0ne_sdk::attest::Collateral {
        tcb_info,
        tcb_info_chain_pem,
        qe_identity,
        qe_identity_chain_pem,
        tcb_info_standard,
        tcb_info_standard_chain_pem,
        qe_identity_standard,
        qe_identity_standard_chain_pem,
        pck_crl_der_b64: b64(&pck_crl),
        root_ca_crl_der_b64: b64(&root_crl),
        gce_endorsement_b64: b64(&endorsement),
        gce_firmware_b64: b64(&firmware),
        nras_answer: String::new(),
        nras_jwks: String::new(),
    })
}

async fn fetch_early(
    http: &reqwest::Client,
    base: &str,
    fmspc_hex: &str,
) -> Result<(Collateral, Collateral), String> {
    let t = fetch_pcs(http, &crate::tcb::pcs_tcb_url(base, fmspc_hex))
        .await
        .map_err(|e| format!("early tcbInfo not fetched: {e}"))?;
    let q = fetch_pcs(http, &crate::tcb::pcs_qe_identity_url(base))
        .await
        .map_err(|e| format!("early QE identity not fetched: {e}"))?;
    Ok((t, q))
}

/// GET one binary PCS answer (the PCK CRL, DER): status and size checked.
pub async fn fetch_pcs_der(http: &reqwest::Client, url: &str) -> Result<Vec<u8>, String> {
    let resp = http
        .get(url)
        .timeout(std::time::Duration::from_secs(PCS_TIMEOUT_SECS))
        .send()
        .await
        .map_err(|e| format!("GET {url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("PCS answered {} for {url}", resp.status()));
    }
    crate::http_body::read_capped(resp, MAX_COLLATERAL_BYTES)
        .await
        .map_err(|e| format!("PCS body: {e}"))
}

// ── The verdict ──────────────────────────────────────────────────────────────

/// The firmware binary `endorsement` names, from Google's bucket: its
/// SHA-384 is the signed one and the MRTD it measures is the quote's (65).
/// A bucket that cannot be read refuses with 76, a binary it lacks with 65.
async fn endorsed_firmware(
    http: &reqwest::Client,
    anchors: &Anchors,
    endorsement: &crate::gce_endorsement::Endorsement,
    quote_mrtd: &[u8; 48],
) -> Result<crate::tdvf::EndorsedFirmware, Refusal> {
    let digest = &endorsement.uefi_sha384_hex;
    crate::tdvf::object_name(digest).map_err(|e| {
        refuse(
            65,
            format!("firmware: the Google endorsement names no firmware binary: {e}"),
        )
    })?;
    let binary = crate::tdvf::fetch(http, &anchors.gce_bucket_base_url, digest)
        .await
        .map_err(|e| {
            refuse(
                76,
                format!("Google bucket unreachable: firmware binary: {e}"),
            )
        })?
        .ok_or_else(|| {
            refuse(
                65,
                format!("firmware: Google's bucket has no firmware binary {digest}"),
            )
        })?;
    crate::tdvf::verify_endorsed(&binary, endorsement, quote_mrtd)
        .map_err(|e| refuse(65, format!("firmware: {e}")))
}

pub async fn verify_document(
    doc: &Document,
    exp: &Expectations<'_>,
    anchors: &Anchors,
    http: &reqwest::Client,
    now: u64,
) -> Result<Report, Refusal> {
    if doc.schema != crate::attest_v2::SCHEMA {
        return Err(refuse(
            88,
            format!(
                "attestation schema {:?}, expected {:?}",
                doc.schema,
                crate::attest_v2::SCHEMA
            ),
        ));
    }
    if doc.snp.is_some() {
        return Err(refuse(
            88,
            "the TD sent AMD SEV-SNP evidence and this session pins an Intel TDX image: refused, no fallback",
        ));
    }
    // 1. The clock, against Intel's, on the QE identity the acceptance
    // stands on: `standard` under the transparent policy, `early` under strict.
    let update = anchors.tcb_policy.acceptance_update();
    let qe = fetch_pcs(
        http,
        &crate::tcb::pcs_qe_identity_url_for(&anchors.pcs_base_url, update),
    )
    .await
    .map_err(|e| refuse(76, format!("Intel PCS unreachable: {e}")))?;
    let intel_now = qe
        .date_unix
        .ok_or_else(|| refuse(87, "clock cannot be checked: Intel PCS sent no Date header"))?;
    if now.abs_diff(intel_now) > CLOCK_SKEW_SECS {
        return Err(refuse(
            87,
            format!("clock is wrong: this machine says {now}, Intel PCS says {intel_now} (more than {CLOCK_SKEW_SECS} s apart)"),
        ));
    }
    // 2-3. The config and the binding.
    let binding = check_config_and_binding(doc, exp.config, &exp.tls_spki_sha256)?;
    let report_data = binding.tdx_report_data();
    // 4. The quote: DCAP to the pinned Intel root, then live collateral.
    let quote = unb64(&doc.tdx.quote_b64).map_err(|e| refuse(65, format!("quote_b64: {e}")))?;
    let chain = crate::pcs::verify_tdx_chain(&quote, &anchors.tdx, now)
        .map_err(|e| refuse(65, format!("TDX quote: {e}")))?;
    if !binding.matches(&chain.report_data) {
        return Err(refuse(
            88,
            "the quote does not bind this channel: report_data differs from SHA-512 over the session key, this connection's TLS key and this session's config",
        ));
    }
    let claims = x0ne_confidential::tdx_quote::parse(&quote)
        .map_err(|e| refuse(65, format!("TDX quote: {e:?}")))?;
    let fmspc = crate::tcb::pck_tcb(&quote)
        .map_err(|e| refuse(65, format!("platform identity: {e}")))?
        .fmspc_hex;
    let tcb = fetch_pcs(
        http,
        &crate::tcb::pcs_tcb_url_for(&anchors.pcs_base_url, &fmspc, update),
    )
    .await
    .map_err(|e| refuse(76, format!("Intel PCS unreachable: {e}")))?;
    // Transparent: Intel's latest (`early`) answer too, named in the verdict;
    // one that cannot be fetched is said so, not refused.
    let early = match update {
        crate::tcb::PcsUpdate::Early => None,
        crate::tcb::PcsUpdate::Standard => {
            Some(fetch_early(http, &anchors.pcs_base_url, &fmspc).await)
        }
    };
    let latest = early.as_ref().map(|r| {
        r.as_ref()
            .map(|(t, q)| crate::tcb::Collateral {
                tcb_body: &t.body,
                tcb_chain_pem: &t.chain_pem,
                qe_body: &q.body,
                qe_chain_pem: &q.chain_pem,
            })
            .map_err(Clone::clone)
    });
    let judgement = crate::tcb::judge_live(
        &quote,
        &crate::tcb::Collateral {
            tcb_body: &tcb.body,
            tcb_chain_pem: &tcb.chain_pem,
            qe_body: &qe.body,
            qe_chain_pem: &qe.chain_pem,
        },
        latest,
        &anchors.tdx,
        now,
        &anchors.tcb_policy,
    )
    .map_err(|e| refuse(76, format!("live Intel TCB: {e}")))?;
    let (platform, qe_status) = judgement.statuses();
    let disclosure = crate::tcb_disclosure::TcbDisclosure::intel(
        &judgement,
        anchors.tcb_policy.min_tcb_evaluation_data_number,
        anchors.tcb_policy.max_collateral_age_secs,
        now,
    );
    // The PCK itself: a key extracted from one machine signs genuine-looking
    // quotes (TEE.fail, 2025); Intel revokes it on the PCK CRL.
    let ca = crate::tcb::pck_crl_ca(&quote)
        .map_err(|e| refuse(65, format!("platform identity: {e}")))?;
    let crl = fetch_pcs_der(
        http,
        &crate::tcb::pcs_pck_crl_url(&anchors.pcs_base_url, ca),
    )
    .await
    .map_err(|e| refuse(76, format!("Intel PCS unreachable: PCK CRL: {e}")))?;
    crate::tcb::verify_pck_crl(&quote, &crl, &anchors.tdx, now, &anchors.tcb_policy).map_err(
        |e| match e {
            crate::tcb::TcbRejected::PckRevoked { .. } => refuse(65, format!("TDX quote: {e}")),
            _ => refuse(76, format!("live Intel {e}")),
        },
    )?;
    // What the root issued, the PCK CA and the TCB signer: Intel's root CA CRL.
    let root_crl = fetch_pcs_der(http, &anchors.root_crl_url)
        .await
        .map_err(|e| refuse(76, format!("Intel root CA CRL unreachable: {e}")))?;
    let mut signer_chains: Vec<&str> = vec![&tcb.chain_pem, &qe.chain_pem];
    if let Some(Ok((t, q))) = &early {
        signer_chains.extend([t.chain_pem.as_str(), q.chain_pem.as_str()]);
    }
    crate::tcb::verify_root_crl(&quote, &signer_chains, &root_crl, &anchors.tdx, now).map_err(
        |e| match e {
            crate::tcb::TcbRejected::CaRevoked { .. } => refuse(65, format!("TDX quote: {e}")),
            _ => refuse(76, format!("live Intel {e}")),
        },
    )?;
    // 5. MRTD: Google's launch endorsement for exactly this firmware.
    let reference = &exp.reference.reference;
    let root_spki = anchors
        .gce_root_spki_sha256_hex()
        .map_err(|e| refuse(89, e))?;
    if reference.mrtd.root_spki_sha256 != root_spki {
        return Err(refuse(
            89,
            format!(
                "reference {} names Google root {} but this verifier pins {}",
                exp.reference.path.display(),
                reference.mrtd.root_spki_sha256,
                root_spki
            ),
        ));
    }
    let blob = crate::gce_endorsement::fetch(http, &anchors.gce_bucket_base_url, &claims.mr_td)
        .await
        .map_err(|e| refuse(76, format!("Google endorsement bucket unreachable: {e}")))?
        .ok_or_else(|| {
            refuse(
                65,
                format!(
                    "MRTD {} has no Google launch endorsement",
                    hex(&claims.mr_td)
                ),
            )
        })?;
    let endorsement = crate::gce_endorsement::verify(&blob, &anchors.gce_root_der, now)
        .map_err(|e| refuse(65, format!("Google endorsement: {e}")))?;
    crate::reference::check_mrtd(reference, exp.machine_type, &endorsement, &claims.mr_td)
        .map_err(|e| refuse(65, format!("MRTD: {e}")))?;
    // The binary the endorsement names: Google's signature over its SHA-384,
    // and the MRTD computed here from it. Every admitted reference pins RTMR0
    // to it (`gce-endorsed`, `reference::load`); `check_image_with` refuses
    // one that does not.
    let firmware = endorsed_firmware(http, anchors, &endorsement, &claims.mr_td).await?;
    // 6. The event log against the image reference.
    let log = unb64(&doc.tdx.ccel_b64).map_err(|e| refuse(65, format!("ccel_b64: {e}")))?;
    let events = crate::ccel::parse(&log).map_err(|e| refuse(65, format!("CCEL: {e}")))?;
    let image = crate::reference::check_image_with(
        reference,
        exp.machine_type,
        &claims,
        &events,
        &firmware,
    )
    .map_err(|e| refuse(65, format!("image: {e}")))?;
    // 7. The GPU, under a pinned NVIDIA intermediate (004 or 005), for the v2 nonce.
    let gpu_required = exp.config.config.gpu_required();
    let evidence = AttestationEvidence {
        format: EvidenceFormat::TdxQuote,
        payload: quote.clone(),
        collected_at_unix: now,
    };
    let policy = VerificationPolicy {
        workload_name: "x0ne-infer".into(),
        accepted_cpu: vec![x0ne_confidential::model::TeeTechnology::IntelTdx],
        require_gpu_cc_on: gpu_required,
        expected_measurements: vec![],
    };
    let mut per_gpu = Vec::new();
    let mut serials = Vec::new();
    let mut ocsp = Vec::new();
    if gpu_required {
        (per_gpu, serials, ocsp, _) = doc_gpu(doc, &binding, anchors, now)?;
    }
    let verdict = if gpu_required {
        crate::verify::cryptographic_with_anchors(
            &crate::verify::CompositeEvidence {
                tdx: &evidence,
                nras_token: &doc.nras.overall_token,
                nras_jwks: &doc.nras.jwks,
                nonce_hex: &binding.gpu_nonce_hex(),
                expected_tdx_report_data: &report_data,
                nras_per_gpu: &per_gpu,
                local_gpu_serials: &serials,
                nras_ocsp_responses: &ocsp,
                nras_ocsp: &anchors.nras_ocsp,
            },
            &policy,
            now,
            &anchors.tdx,
            &anchors.nras_intermediate_pins,
            Some((platform.as_str(), qe_status.as_str())),
        )
    } else {
        crate::verify::cryptographic_tdx_after_live_tcb_with_anchor(
            &evidence,
            &policy,
            now,
            &anchors.tdx,
            &report_data,
            &platform,
            &qe_status,
        )
    };
    let (identity, verifier) = match &verdict {
        Verdict::Rejected { reason, .. } => {
            return Err(refuse(65, format!("verdict rejected: {reason}")))
        }
        Verdict::Verified { strength, .. } if *strength != VerifierStrength::Cryptographic => {
            return Err(refuse(
                66,
                format!("verdict is {strength:?}, not cryptographic"),
            ))
        }
        Verdict::Verified {
            identity, verifier, ..
        } => (identity.clone(), verifier.clone()),
    };
    // 8. The model: the name asked for, the catalog's files.
    check_model(doc, exp.config, exp.catalog_entry)?;

    let mut limitations: Vec<String> = LIMITATIONS.iter().map(|s| (*s).to_string()).collect();
    limitations.extend(image.warnings.iter().cloned());
    limitations.push(format!("verdict verifier: {verifier}"));
    if exp.reference.explicit {
        limitations.push(format!(
            "image reference given explicitly: {} (sha256 {})",
            exp.reference.path.display(),
            exp.reference.sha256_hex
        ));
    }
    // `check_image_with` pinned it or refused: no other note.
    debug_assert!(image.rtmr0_pinned);
    let rtmr0_note =
        "; RTMR0 pinned (the endorsed firmware's configuration volume, the reference's host events)";
    let accepted_under = match judgement.mode {
        crate::tcb::TcbMode::Strict => format!(
            "Intel PCS, fetched now: TCB evaluation {} or later, collateral at most {} s old",
            anchors.tcb_policy.min_tcb_evaluation_data_number,
            anchors.tcb_policy.max_collateral_age_secs
        ),
        crate::tcb::TcbMode::Transparent => format!(
            "Intel PCS standard collateral, fetched now: TCB evaluation {}, collateral at most {} s old; transparent TCB policy, Intel's latest collateral on the TCB line",
            judgement.accepted.tcb_evaluation,
            anchors.tcb_policy.max_collateral_age_secs
        ),
    };
    let mut summary = vec![
        format!(
            "CPU   Intel TDX quote genuine, TCB {} / QE {}, TDX module identity matched, PCK not on Intel's CRL, its CA and Intel's TCB signer not on Intel's root CA CRL ({accepted_under})",
            platform.as_str(),
            qe_status.as_str(),
        ),
        format!(
            "Firm  Google-endorsed firmware binary {}... measures the quoted MRTD {}... (svn {}, signed {})",
            &firmware.sha384_hex[..8],
            &hex(&claims.mr_td)[..8],
            endorsement.tdx_svn,
            endorsement.signer_subject
        ),
        format!(
            "Image {} {} (reference signed {} by archive subkey with key ID {}{}; logged at index {} of {}, size {}): RTMR1, RTMR2 match the reference; RTMR3 empty{rtmr0_note}",
            reference.image.name,
            reference.image.version,
            crate::openpgp::utc_date(exp.reference.signed_by().signed_at_unix),
            exp.reference.signed_by().subkey_key_id(),
            if exp.reference.signed_by().test_key {
                "; TEST key, not for release"
            } else {
                ""
            },
            exp.reference.logged().index,
            exp.reference.logged().origin,
            exp.reference.logged().size,
        ),
        if gpu_required {
            format!("GPU   {} GPU(s), confidential computing on (NRAS tokens, nonce bound to this session, then to each connection's TLS exporter and quote)", doc.gpu.len())
        } else {
            "GPU   not attested: the cpu product (TDX only)".to_string()
        },
        "Link  TLS key and your session config are bound to this quote".to_string(),
        format!(
            "Model {}, digest {}... matches the catalog",
            doc.model.name,
            &doc.model.digest_hex[..8]
        ),
    ];
    summary.push(format!("TCB   {}", disclosure.line));
    Ok(Report {
        verdict: serde_json::to_value(&verdict).unwrap_or_default(),
        environment: format!("{:?}", identity.environment),
        session_pubkey_hex: doc.binding.session_pubkey_hex.clone(),
        model_name: doc.model.name.clone(),
        model_digest_hex: doc.model.digest_hex.clone(),
        tcb_platform: platform.as_str().to_string(),
        tcb_qe: qe_status.as_str().to_string(),
        endorsement_signer: endorsement.signer_subject,
        endorsement_timestamp_unix: endorsement.timestamp_unix,
        firmware_svn: endorsement.tdx_svn,
        mrtd_hex: hex(&claims.mr_td),
        image_name: reference.image.name.clone(),
        image_version: reference.image.version.clone(),
        rtmr0_hex: image.rtmr0_hex,
        rtmr0_known: image.rtmr0_known,
        rtmr0_pinned: image.rtmr0_pinned,
        firmware_sha384_hex: firmware.sha384_hex,
        sections: image.sections,
        gpu_count: doc.gpu.len(),
        reference_path: exp.reference.path.display().to_string(),
        reference_sha256_hex: exp.reference.sha256_hex.clone(),
        reference_explicit: exp.reference.explicit,
        reference_signer_subkey: exp.reference.signed_by().subkey_fingerprint.clone(),
        reference_signed_at_unix: exp.reference.signed_by().signed_at_unix,
        summary,
        limitations,
        snp: None,
        tcb: Some(disclosure),
    })
}

/// Steps 2 and 3, the same for every TEE: the config bytes this session
/// sent, then the binding of the session key, this connection's TLS key and
/// the config hash (88).
fn check_config_and_binding(
    doc: &Document,
    config: &Loaded,
    tls_spki_sha256: &[u8; 32],
) -> Result<BindingV2, Refusal> {
    // 2. The config: the exact bytes this session sent.
    let config_bytes =
        unb64(&doc.config_b64).map_err(|e| refuse(88, format!("config_b64: {e}")))?;
    if config_bytes != config.bytes {
        return Err(refuse(
            88,
            "the TD's config is not the one this session sent (bytes differ)",
        ));
    }
    // 3. The binding: session key, TLS key of THIS connection, config hash.
    let session_nonce = unhex32(&doc.binding.session_nonce_hex)
        .map_err(|e| refuse(88, format!("session_nonce_hex: {e}")))?;
    let session_pubkey = unhex32(&doc.binding.session_pubkey_hex)
        .map_err(|e| refuse(88, format!("session_pubkey_hex: {e}")))?;
    ed25519_dalek::VerifyingKey::from_bytes(&session_pubkey)
        .map_err(|e| refuse(88, format!("session_pubkey_hex: {e}")))?;
    if doc.binding.tls_spki_sha256_hex != hex(tls_spki_sha256) {
        return Err(refuse(
            88,
            format!(
                "the TD attested TLS key {} but this connection sees {}",
                doc.binding.tls_spki_sha256_hex,
                hex(tls_spki_sha256)
            ),
        ));
    }
    if doc.binding.config_sha256_hex != hex(&config.sha256) {
        return Err(refuse(
            88,
            "the TD attested another config hash than this session's",
        ));
    }
    Ok(BindingV2 {
        session_nonce,
        session_pubkey,
        tls_spki_sha256: *tls_spki_sha256,
        config_sha256: config.sha256,
    })
}

/// The document's GPUs, under a pinned NVIDIA intermediate, for the v2
/// nonce (75): the same for every TEE.
fn doc_gpu(
    doc: &Document,
    binding: &BindingV2,
    anchors: &Anchors,
    now: u64,
) -> Result<
    (
        Vec<(String, String)>,
        Vec<String>,
        Vec<Vec<u8>>,
        crate::nras::VerifiedGpu,
    ),
    Refusal,
> {
    if doc.gpu.is_empty() {
        return Err(refuse(
            75,
            "the config asked for cpu+gpu and the document attests no GPU",
        ));
    }
    let mut per_gpu = Vec::new();
    let mut serials = Vec::new();
    for g in &doc.gpu {
        per_gpu.push((g.nras_name.clone(), g.nras_token.clone()));
        serials.push(
            crate::nras::gpu_leaf_serial_decimal(&g.certificate_b64)
                .map_err(|e| refuse(75, format!("GPU {}: {e}", g.index)))?,
        );
    }
    let ocsp = stapled_ocsp(&doc.nras.ocsp_b64)?;
    let verified = crate::nras::verify_nras_attestation(
        &doc.nras.overall_token,
        &per_gpu,
        &serials,
        &doc.nras.jwks,
        &binding.gpu_nonce_hex(),
        now,
        &anchors.nras_intermediate_pins,
        &ocsp,
        &anchors.nras_ocsp,
    )
    .map_err(|e| refuse(75, format!("NRAS: {e}")))?;
    Ok((per_gpu, serials, ocsp, verified))
}

/// The OCSP responses a document staples about the NRAS signing chain
/// (`nras.ocsp_b64`), DER; one that is not base64 refuses (75).
fn stapled_ocsp(b64: &[String]) -> Result<Vec<Vec<u8>>, Refusal> {
    b64.iter()
        .enumerate()
        .map(|(i, s)| unb64(s).map_err(|e| refuse(75, format!("nras.ocsp_b64[{i}]: {e}"))))
        .collect()
}

/// Step 8, the same for every TEE: the model the config asked for, the
/// catalog's files (or the sealed file the config names), the digest every
/// receipt will carry (77).
fn check_model(
    doc: &Document,
    config: &Loaded,
    entry: &crate::registry::Entry,
) -> Result<(), Refusal> {
    if doc.model.name != config.config.model {
        return Err(refuse(
            77,
            format!(
                "the TD serves model {:?}, this session asked for {:?}",
                doc.model.name, config.config.model
            ),
        ));
    }
    let mut have: Vec<(String, String)> = doc
        .model
        .files
        .iter()
        .map(|f| (f.path.clone(), f.sha256.clone()))
        .collect();
    have.sort();
    // Sealed weights: the base model's files but the weights, the sealed
    // file the config names, and the digest the config promises (the
    // owner's, from the policy); the catalog pins neither.
    let sealed = config.config.sealed.as_ref();
    let want = match sealed {
        Some(s) => crate::model_key::sealed_files(entry, &s.sealed_sha256),
        None => crate::attest_v2::catalog_files(entry),
    };
    if have != want {
        return Err(refuse(
            77,
            match sealed {
                Some(_) => format!(
                    "the TD's files for {} are not the base model's with the sealed file the config names",
                    doc.model.name
                ),
                None => format!("the TD's files for {} are not the catalog's", doc.model.name),
            },
        ));
    }
    if unhex32(&doc.model.digest_hex).is_err() {
        return Err(refuse(
            77,
            format!("model digest {:?} is not 64 hex", doc.model.digest_hex),
        ));
    }
    // The digest every receipt will carry, against the catalog's pin: the
    // file list above says what was pulled, this says what is served.
    let pinned = match sealed {
        Some(s) => Some(s.digest_hex.as_str()),
        None => entry.digest_sha256.as_deref(),
    };
    let Some(pinned) = pinned else {
        return Err(refuse(
            77,
            format!(
                "the catalog pins no served-model digest for {}: not servable in cloud mode",
                doc.model.name
            ),
        ));
    };
    if doc.model.digest_hex != pinned {
        return Err(refuse(
            77,
            format!(
                "the TD's model digest {} is not the catalog's served-model digest {} for {}",
                doc.model.digest_hex, pinned, doc.model.name
            ),
        ));
    }
    Ok(())
}

// ── Per-connection binding (v3) ──────────────────────────────────────────────

/// Offset of `report_data` in a TDX v4 quote: header (48) + body up to it (520).
const QUOTE_REPORT_DATA_AT: usize = 568;

fn clone_anchor(a: &TrustAnchor) -> TrustAnchor {
    match a {
        TrustAnchor::SpkiSha256Hex(s) => TrustAnchor::SpkiSha256Hex(s.clone()),
        TrustAnchor::RootDer(d) => TrustAnchor::RootDer(d.clone()),
    }
}

/// What every connection's session quote must show, fixed by a verified
/// document: the same TD (header and body up to `report_data` byte for
/// byte), the same platform (PCK), and `BindingV3` over THIS connection's
/// exporter. With the GPU product, also that connection's GPU evidence: NRAS
/// tokens for `session_gpu_nonce(exporter, the session quote's TD report)`,
/// naming the GPUs the document attested.
struct TdxSessionCheck {
    binding: BindingV2,
    doc_quote_prefix: Vec<u8>,
    doc_pck_sha256_hex: String,
    anchor: TrustAnchor,
    gpu: Option<SessionGpuCheck>,
    clock: Arc<dyn Fn() -> u64 + Send + Sync>,
}

/// The document's GPUs (their certificate serials) and the NRAS pins.
struct SessionGpuCheck {
    serials: Vec<String>,
    pins: Vec<crate::nras::NrasIntermediatePin>,
    ocsp: crate::nras_ocsp::NrasOcspPolicy,
}

impl SessionGpuCheck {
    /// This connection's GPU evidence, `signed` being the signed report of
    /// its verified session evidence (a TD report, or an SNP report's
    /// first 0x2A0 bytes).
    fn verify(
        &self,
        gpu: Option<&crate::attest_v2::SessionGpuDoc>,
        exporter: &[u8; 32],
        signed: &[u8],
        now: u64,
    ) -> Result<(), Refusal> {
        let gpu = gpu.ok_or_else(|| {
            refuse(
                75,
                "the config asked for cpu+gpu and this connection's session attests no GPU",
            )
        })?;
        let nonce = crate::binding::session_gpu_nonce(exporter, signed);
        let mut per_gpu = Vec::with_capacity(gpu.gpu.len());
        let mut serials = Vec::with_capacity(gpu.gpu.len());
        for g in &gpu.gpu {
            per_gpu.push((g.nras_name.clone(), g.nras_token.clone()));
            serials.push(
                crate::nras::gpu_leaf_serial_decimal(&g.certificate_b64).map_err(|e| {
                    refuse(
                        75,
                        format!("this connection's GPU evidence: GPU {}: {e}", g.index),
                    )
                })?,
            );
        }
        let ocsp = stapled_ocsp(&gpu.nras.ocsp_b64)?;
        crate::nras::verify_nras_attestation(
            &gpu.nras.overall_token,
            &per_gpu,
            &serials,
            &gpu.nras.jwks,
            &hex(&nonce),
            now,
            &self.pins,
            &ocsp,
            &self.ocsp,
        )
        .map_err(|e| refuse(75, format!("this connection's GPU evidence: NRAS: {e}")))?;
        let (mut now_seen, mut at_boot) = (serials, self.serials.clone());
        now_seen.sort();
        at_boot.sort();
        if now_seen != at_boot {
            return Err(refuse(
                75,
                format!(
                    "the GPUs attested on this connection are not the document's: certificate serials {now_seen:?}, document {at_boot:?}"
                ),
            ));
        }
        Ok(())
    }
}

impl TdxSessionCheck {
    /// `doc` must already have passed [`verify_document`] with `exp`.
    pub fn new(
        doc: &Document,
        exp: &Expectations<'_>,
        anchors: &Anchors,
        clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    ) -> Result<Self, Refusal> {
        let quote = unb64(&doc.tdx.quote_b64).map_err(|e| refuse(65, format!("quote_b64: {e}")))?;
        let chain = crate::pcs::verify_tdx_chain(&quote, &anchors.tdx, clock())
            .map_err(|e| refuse(65, format!("TDX quote: {e}")))?;
        let prefix = quote
            .get(..QUOTE_REPORT_DATA_AT)
            .ok_or_else(|| refuse(65, "TDX quote too short"))?
            .to_vec();
        let session_pubkey = unhex32(&doc.binding.session_pubkey_hex)
            .map_err(|e| refuse(88, format!("session_pubkey_hex: {e}")))?;
        let session_nonce = unhex32(&doc.binding.session_nonce_hex)
            .map_err(|e| refuse(88, format!("session_nonce_hex: {e}")))?;
        let gpu = if exp.config.config.gpu_required() {
            let serials = doc
                .gpu
                .iter()
                .map(|g| {
                    crate::nras::gpu_leaf_serial_decimal(&g.certificate_b64)
                        .map_err(|e| refuse(75, format!("GPU {}: {e}", g.index)))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if serials.is_empty() {
                return Err(refuse(
                    75,
                    "the config asked for cpu+gpu and the document attests no GPU",
                ));
            }
            Some(SessionGpuCheck {
                serials,
                pins: anchors.nras_intermediate_pins.clone(),
                ocsp: anchors.nras_ocsp.clone(),
            })
        } else {
            None
        };
        Ok(Self {
            binding: BindingV2 {
                session_nonce,
                session_pubkey,
                tls_spki_sha256: exp.tls_spki_sha256,
                config_sha256: exp.config.sha256,
            },
            doc_quote_prefix: prefix,
            doc_pck_sha256_hex: chain.pck_sha256_hex,
            anchor: clone_anchor(&anchors.tdx),
            gpu,
            clock,
        })
    }

    /// The session quote of the connection that exported `exporter`.
    pub fn verify(
        &self,
        session: &crate::attest_v2::SessionDoc,
        exporter: &[u8; 32],
    ) -> Result<(), Refusal> {
        if session.schema != crate::attest_v2::SESSION_SCHEMA {
            return Err(refuse(
                88,
                format!(
                    "session attestation schema {:?}, expected {:?}",
                    session.schema,
                    crate::attest_v2::SESSION_SCHEMA
                ),
            ));
        }
        if session.snp.is_some() {
            return Err(refuse(
                88,
                "the session attestation carries AMD SEV-SNP evidence and this session pins Intel TDX: refused, no fallback",
            ));
        }
        let quote =
            unb64(&session.quote_b64).map_err(|e| refuse(65, format!("session quote_b64: {e}")))?;
        let chain = crate::pcs::verify_tdx_chain(&quote, &self.anchor, (self.clock)())
            .map_err(|e| refuse(65, format!("session TDX quote: {e}")))?;
        if chain.pck_sha256_hex != self.doc_pck_sha256_hex {
            return Err(refuse(
                88,
                "the session quote comes from another platform than the document's",
            ));
        }
        if quote.get(..QUOTE_REPORT_DATA_AT) != Some(&self.doc_quote_prefix[..]) {
            return Err(refuse(
                88,
                "the session quote reports another TD or TCB than the document's",
            ));
        }
        let v3 = crate::binding::BindingV3::of(&self.binding, *exporter);
        if !v3.matches(&chain.report_data) {
            return Err(refuse(
                88,
                "the session quote does not bind this connection's TLS exporter: the TLS session is not the TD's (relay or replay)",
            ));
        }
        if let Some(gpu) = &self.gpu {
            let report = crate::pcs::signed_td_report(&quote)
                .ok_or_else(|| refuse(65, "session TDX quote too short"))?;
            gpu.verify(session.gpu.as_ref(), exporter, report, (self.clock)())?;
        }
        Ok(())
    }

    /// The recipient quote for `release_nonce`: the same TD and platform as
    /// the document's, over THIS release, THIS session and the X25519 key
    /// it names. What it returns is what the model key may be wrapped to.
    pub fn verify_recipient(
        &self,
        doc: &crate::model_key::RecipientDoc,
        release_nonce: &[u8; 32],
    ) -> Result<crate::model_key::RecipientBinding, Refusal> {
        if doc.schema != crate::model_key::RECIPIENT_SCHEMA {
            return Err(refuse(
                88,
                format!(
                    "recipient schema {:?}, expected {:?}",
                    doc.schema,
                    crate::model_key::RECIPIENT_SCHEMA
                ),
            ));
        }
        if doc.suite != crate::hpke::SUITE {
            return Err(refuse(88, format!("recipient suite {:?}", doc.suite)));
        }
        if doc.release_nonce_hex != hex(release_nonce) {
            return Err(refuse(
                88,
                "the recipient quote answers another release nonce: a replay",
            ));
        }
        let kem_pk = unhex32(&doc.kem_pk_hex)
            .map_err(|e| refuse(88, format!("recipient kem_pk_hex: {e}")))?;
        let quote =
            unb64(&doc.quote_b64).map_err(|e| refuse(65, format!("recipient quote_b64: {e}")))?;
        let chain = crate::pcs::verify_tdx_chain(&quote, &self.anchor, (self.clock)())
            .map_err(|e| refuse(65, format!("recipient TDX quote: {e}")))?;
        if chain.pck_sha256_hex != self.doc_pck_sha256_hex {
            return Err(refuse(
                88,
                "the recipient quote comes from another platform than the document's",
            ));
        }
        if quote.get(..QUOTE_REPORT_DATA_AT) != Some(&self.doc_quote_prefix[..]) {
            return Err(refuse(
                88,
                "the recipient quote reports another TD or TCB than the document's",
            ));
        }
        let binding = crate::model_key::RecipientBinding::of(&self.binding, *release_nonce, kem_pk);
        if !binding.matches(&chain.report_data) {
            return Err(refuse(
                88,
                "the recipient quote does not bind this release, this session and the key it names: replayed or substituted",
            ));
        }
        Ok(binding)
    }
}

/// What every connection's attestation must show once the document passed:
/// the TDX check ([`verify_document`]) or the SEV-SNP one
/// ([`verify_snp_document`]), by the TEE the session pins.
pub struct SessionCheck(SessionKind);

enum SessionKind {
    Tdx(TdxSessionCheck),
    Snp(SnpSessionCheck),
}

impl SessionCheck {
    /// `doc` must already have passed [`verify_document`] with `exp`.
    pub fn new(
        doc: &Document,
        exp: &Expectations<'_>,
        anchors: &Anchors,
        clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    ) -> Result<Self, Refusal> {
        TdxSessionCheck::new(doc, exp, anchors, clock).map(|c| Self(SessionKind::Tdx(c)))
    }

    /// `doc` must already have passed [`verify_snp_document`] with `exp`,
    /// which returned `crl`.
    pub fn snp(
        doc: &Document,
        exp: &SnpExpectations<'_>,
        anchors: &Anchors,
        crl: &SnpCrl,
        clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    ) -> Result<Self, Refusal> {
        let payload = snp_payload(
            &doc.tdx.quote_b64,
            &doc.tdx.ccel_b64,
            doc.snp.as_ref(),
            "document",
        )?;
        let binding = check_config_and_binding(doc, exp.config, &exp.tls_spki_sha256)?;
        let gpu = if exp.config.config.gpu_required() {
            let serials = doc
                .gpu
                .iter()
                .map(|g| {
                    crate::nras::gpu_leaf_serial_decimal(&g.certificate_b64)
                        .map_err(|e| refuse(75, format!("GPU {}: {e}", g.index)))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if serials.is_empty() {
                return Err(refuse(
                    75,
                    "the config asked for cpu+gpu and the document attests no GPU",
                ));
            }
            Some(SessionGpuCheck {
                serials,
                pins: anchors.nras_intermediate_pins.clone(),
                ocsp: anchors.nras_ocsp.clone(),
            })
        } else {
            None
        };
        SnpSessionCheck::new(
            &payload,
            &crl.der,
            binding,
            &exp.reference.reference,
            &anchors.snp,
            anchors.tcb_policy.mode,
            gpu,
            clock,
        )
        .map(|c| Self(SessionKind::Snp(c)))
    }

    /// The session attestation of the connection that exported `exporter`.
    pub fn verify(
        &self,
        session: &crate::attest_v2::SessionDoc,
        exporter: &[u8; 32],
    ) -> Result<(), Refusal> {
        match &self.0 {
            SessionKind::Tdx(c) => c.verify(session, exporter),
            SessionKind::Snp(c) => c.verify(session, exporter),
        }
    }

    /// The recipient quote for `release_nonce` (TDX). An SEV-SNP guest
    /// gets no model key: the release is written for a TDX quote.
    pub fn verify_recipient(
        &self,
        doc: &crate::model_key::RecipientDoc,
        release_nonce: &[u8; 32],
    ) -> Result<crate::model_key::RecipientBinding, Refusal> {
        match &self.0 {
            SessionKind::Tdx(c) => c.verify_recipient(doc, release_nonce),
            SessionKind::Snp(_) => Err(refuse(
                88,
                "the model key is released to Intel TDX guests only: this session's guest is AMD SEV-SNP",
            )),
        }
    }
}

// ── AMD SEV-SNP ──────────────────────────────────────────────────────────────

/// The image reference a session pins, by TEE. The relay holds one and
/// verifies only that TEE's evidence: a TDX reference refuses an SEV-SNP
/// document and an SEV-SNP reference a TDX one (88), no fallback.
pub enum Pinned {
    Tdx(Box<crate::reference::Loaded>),
    Snp(Box<crate::snp_reference::Loaded>),
}

impl Pinned {
    pub fn provider(&self) -> &str {
        match self {
            Self::Tdx(r) => &r.reference.provider,
            Self::Snp(r) => &r.reference.provider,
        }
    }

    pub fn tee(&self) -> &'static str {
        match self {
            Self::Tdx(_) => "intel-tdx",
            Self::Snp(_) => crate::snp_reference::TEE,
        }
    }

    /// A reference file, by its schema: signature and log first
    /// (`reference::load_signed`), then the TDX or the SEV-SNP parser. An
    /// unknown schema falls to the TDX parser, which names it and refuses.
    pub fn load(
        path: &std::path::Path,
        explicit: bool,
        key: &crate::archive_key::Anchor,
        log: &crate::transparency::Anchor,
        state_dir: &std::path::Path,
        now_unix: u64,
    ) -> Result<Self, crate::reference::LoadError> {
        #[derive(Deserialize)]
        struct Head {
            schema: Option<serde_json::Value>,
        }
        let s = crate::reference::load_signed(path, explicit, key, log, state_dir, now_unix)?;
        let snp = serde_json::from_slice::<Head>(&s.bytes)
            .ok()
            .and_then(|h| h.schema)
            .is_some_and(|v| v.as_str() == Some(crate::snp_reference::SCHEMA));
        Ok(if snp {
            Self::Snp(Box::new(crate::snp_reference::from_signed(
                s, path, explicit,
            )?))
        } else {
            Self::Tdx(Box::new(crate::reference::from_signed(s, path, explicit)?))
        })
    }
}

/// What an SEV-SNP session expects the guest to prove: as [`Expectations`],
/// the SEV-SNP reference in place of the TDX one, no machine type (the
/// launch digest names the vCPU count and type).
pub struct SnpExpectations<'a> {
    pub config: &'a Loaded,
    pub tls_spki_sha256: [u8; 32],
    pub reference: &'a crate::snp_reference::Loaded,
    pub catalog_entry: &'a crate::registry::Entry,
}

/// What an SEV-SNP verdict established beyond the TDX fields of [`Report`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnpView {
    pub product: String,
    /// "VCEK" or "VLEK".
    pub signing_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub csp_id: Option<String>,
    pub key_sha256_hex: String,
    pub report_version: u32,
    pub measurement_hex: String,
    pub host_data_hex: String,
    /// boot loader, TEE, SNP, microcode, FMC.
    pub reported_tcb: [u8; 5],
    pub minimum_tcb: [u8; 5],
    pub minimum_source: String,
    pub crl_url: String,
    pub crl_this_update_unix: i64,
    pub crl_next_update_unix: i64,
}

/// AMD's CRL the verdict was given under, kept for the per-connection
/// checks of the same session ([`SessionCheck::snp`]).
pub struct SnpCrl {
    der: Vec<u8>,
}

/// What every SEV-SNP verdict says it did not establish, beside the
/// verifier's own lines (`snp::LIMITATIONS`).
pub const SNP_LIMITATIONS: &[&str] = &[
    "this machine: software in the same session can read the local key and the questions",
    "launch digests: the reference lists digests computed outside this verifier from the firmware, kernel, initrd, command line and vCPUs (sev-snp-measure, x0ne-engine snp-measure); its signer vouches for those inputs, nothing here rebuilds them",
    "no event log: an SEV-SNP guest has no RTMR; what it loads after launch is covered only by the launch digest of the code that loads it",
    "TCB: AMD publishes no live status like Intel PCS; the report's TCB is held to the reference's floor and to the bulletins' default compiled into this build",
    "a compromise of the AMD or NVIDIA roots defeats this verdict",
    "availability: whoever runs the host may stop the guest; it sees timing, sizes, IP addresses",
    "the SEV-SNP reference (your --reference) is checked against the x0ne archive key pinned in this build: whoever holds its signing subkey can sign any launch digest, and a revocation reaches you only with a new build",
    "the prompt commitment in a receipt is not opened by the relay (the chat template is applied in the guest)",
    "a streamed answer's text is passed on before its receipt is verified: only text followed by a final chunk with receipt_verified is covered, and text kept above an error is unverified",
    "the model's own behaviour",
];

const SNP_GPU_LIMITATION: &str = "the GPU verdict is NVIDIA's NRAS service; each connection's GPU evidence answers a nonce derived from that connection's TLS exporter and the signed bytes of its SEV-SNP report, so the guest holding this TLS session asked for it, but CPU and GPU are bound by the guest's software, not by hardware (Hopper's GPU report names no VM)";

const SNP_TCB_QE: &str = "not applicable (AMD SEV-SNP)";

/// The SEV-SNP evidence of a document or a session: present under `snp`,
/// with no TDX evidence beside it (88), base64 (65).
fn snp_payload(
    tdx_quote_b64: &str,
    extra_tdx: &str,
    snp: Option<&crate::attest_v2::SnpDoc>,
    what: &str,
) -> Result<Vec<u8>, Refusal> {
    if !tdx_quote_b64.is_empty() || !extra_tdx.is_empty() {
        return Err(refuse(
            88,
            format!("the {what} carries Intel TDX evidence and this session pins an AMD SEV-SNP image: refused, no fallback"),
        ));
    }
    let snp = snp.ok_or_else(|| refuse(88, format!("the {what} carries no SEV-SNP evidence")))?;
    unb64(&snp.evidence_b64).map_err(|e| refuse(65, format!("{what} SEV-SNP evidence_b64: {e}")))
}

/// GET AMD's CRL (DER) from KDS, with the server's clock.
async fn fetch_kds_der(
    http: &reqwest::Client,
    url: &str,
) -> Result<(Vec<u8>, Option<u64>), String> {
    let resp = http
        .get(url)
        .timeout(std::time::Duration::from_secs(PCS_TIMEOUT_SECS))
        .send()
        .await
        .map_err(|e| format!("GET {url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("KDS answered {} for {url}", resp.status()));
    }
    let date_unix = resp
        .headers()
        .get(reqwest::header::DATE)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_http_date);
    let body = crate::http_body::read_capped(resp, MAX_COLLATERAL_BYTES)
        .await
        .map_err(|e| format!("KDS body: {e}"))?;
    Ok((body, date_unix))
}

/// The client's check of one SEV-SNP evidence payload (report ‖ VCEK or
/// VLEK) for the session `binding` names, under the written `reference`:
/// the report verified to a pinned AMD root under the reference's policy
/// and AMD's CRL (65), `REPORT_DATA` equal to the same v2 binding a TDX
/// quote carries — session key, this connection's TLS key, the config hash
/// (88) — and the launch digest one the reference predicts (65). Returns
/// what the verdict says about the launch. `mode` is the relay's TCB policy
/// mode: which of AMD's bulletins the default minimum imposes.
pub fn verify_snp_evidence(
    payload: &[u8],
    crl_der: &[u8],
    binding: &BindingV2,
    reference: &crate::snp_reference::SnpReference,
    anchors: &crate::snp::SnpAnchors,
    mode: crate::tcb::TcbMode,
    now: u64,
) -> Result<(crate::snp::VerifiedSnp, Vec<String>), Refusal> {
    let v =
        crate::snp::verify_snp_report(payload, crl_der, anchors, &reference.policy_with(mode), now)
            .map_err(|e| refuse(65, format!("SEV-SNP report: {e}")))?;
    if !binding.matches(v.report_data()) {
        return Err(refuse(
            88,
            "the SEV-SNP report does not bind this session: REPORT_DATA is not SHA-512 over its key, this connection's TLS key and the config (another session or a relay)",
        ));
    }
    let launch = reference
        .check_launch(&v)
        .map_err(|e| refuse(65, format!("SEV-SNP launch: {e}")))?;
    Ok((v, launch))
}

fn tcb5(t: x0ne_confidential::snp_report::TcbParts) -> [u8; 5] {
    [t.boot_loader, t.tee, t.snp, t.microcode, t.fmc]
}

/// The client's verdict on an SEV-SNP guest, in the order of
/// [`verify_document`]: clock (against AMD KDS, which serves the CRL),
/// config, binding, the report under the written reference, GPU, model.
/// Every step refuses; none falls back to TDX or to "not verified".
pub async fn verify_snp_document(
    doc: &Document,
    exp: &SnpExpectations<'_>,
    anchors: &Anchors,
    http: &reqwest::Client,
    now: u64,
) -> Result<(Report, SnpCrl), Refusal> {
    if doc.schema != crate::attest_v2::SCHEMA {
        return Err(refuse(
            88,
            format!(
                "attestation schema {:?}, expected {:?}",
                doc.schema,
                crate::attest_v2::SCHEMA
            ),
        ));
    }
    let payload = snp_payload(
        &doc.tdx.quote_b64,
        &doc.tdx.ccel_b64,
        doc.snp.as_ref(),
        "document",
    )?;
    // 1. The clock, against AMD's, on the CRL of the product the report
    // claims: read unverified here only to name the URL, verified below.
    let (product, key) = crate::snp::evidence_product(&payload)
        .map_err(|e| refuse(65, format!("SEV-SNP evidence: {e}")))?;
    let crl_url = crate::snp::kds_crl_url(&anchors.snp.kds_base_url, product, key);
    let (crl, kds_now) = fetch_kds_der(http, &crl_url)
        .await
        .map_err(|e| refuse(76, format!("AMD KDS unreachable: {e}")))?;
    let kds_now = kds_now
        .ok_or_else(|| refuse(87, "clock cannot be checked: AMD KDS sent no Date header"))?;
    if now.abs_diff(kds_now) > CLOCK_SKEW_SECS {
        return Err(refuse(
            87,
            format!("clock is wrong: this machine says {now}, AMD KDS says {kds_now} (more than {CLOCK_SKEW_SECS} s apart)"),
        ));
    }
    // 2-3. The config and the binding, as for TDX.
    let binding = check_config_and_binding(doc, exp.config, &exp.tls_spki_sha256)?;
    // 4. The report: AMD's chain and CRL, the written policy, REPORT_DATA,
    // the predicted launch digest.
    let reference = &exp.reference.reference;
    let mode = anchors.tcb_policy.mode;
    let (v, launch) =
        verify_snp_evidence(&payload, &crl, &binding, reference, &anchors.snp, mode, now)?;
    // 5. The GPU, as for TDX: NRAS tokens for the v2 nonce.
    let gpu_required = exp.config.config.gpu_required();
    let gpu = if gpu_required {
        Some(doc_gpu(doc, &binding, anchors, now)?)
    } else {
        None
    };
    // The verdict the gate issues, the launch digest pinned to the one matched.
    let mut expected = vec![crate::snp::parse_expected_snp_measurement(&format!(
        "snp_measurement={}",
        hex(&v.report.measurement)
    ))
    .map_err(|e| refuse(65, e))?];
    if reference.launch.host_data.is_some() {
        expected.push(
            crate::snp::parse_expected_snp_measurement(&format!(
                "snp_host_data={}",
                hex(&v.report.host_data)
            ))
            .map_err(|e| refuse(65, e))?,
        );
    }
    let policy = VerificationPolicy {
        workload_name: "x0ne-infer".into(),
        accepted_cpu: vec![x0ne_confidential::model::TeeTechnology::AmdSevSnp],
        require_gpu_cc_on: false,
        expected_measurements: expected,
    };
    let evidence = AttestationEvidence {
        format: EvidenceFormat::SevSnpReport,
        payload: payload.clone(),
        collected_at_unix: now,
    };
    let cpu = crate::verify::cryptographic_snp(
        &evidence,
        &policy,
        &reference.policy_with(mode),
        &anchors.snp,
        &crl,
        Some(&binding.snp_report_data()),
        now,
    );
    let verdict = match (cpu, &gpu) {
        (
            Verdict::Verified {
                identity,
                verifier,
                strength,
            },
            Some((_, _, _, g)),
        ) => Verdict::Verified {
            identity: x0ne_confidential::model::WorkloadIdentity {
                environment:
                    x0ne_confidential::model::ExecutionEnvironment::ConfidentialVmWithGpu {
                        cpu: x0ne_confidential::model::TeeTechnology::AmdSevSnp,
                        gpu_mode: x0ne_confidential::model::GpuCcMode::On,
                    },
                ..identity
            },
            verifier: format!(
                "{} + x0ne-infer::nras ({} GPU(s) bound by ueid; {}; limitations: {})",
                verifier.replace(&format!("{}; ", crate::snp::CPU_ONLY), ""),
                g.gpu_ueids.len(),
                crate::verify::revocation_note(&anchors.nras_ocsp, g),
                g.limitations.join("; ")
            ),
            strength,
        },
        (other, _) => other,
    };
    let (identity, verifier) = match &verdict {
        Verdict::Rejected { reason, .. } => {
            return Err(refuse(65, format!("verdict rejected: {reason}")))
        }
        Verdict::Verified { strength, .. } if *strength != VerifierStrength::Cryptographic => {
            return Err(refuse(
                66,
                format!("verdict is {strength:?}, not cryptographic"),
            ))
        }
        Verdict::Verified {
            identity, verifier, ..
        } => (identity.clone(), verifier.clone()),
    };
    // 6. The model, as for TDX.
    check_model(doc, exp.config, exp.catalog_entry)?;

    let mut limitations: Vec<String> = SNP_LIMITATIONS.iter().map(|s| s.to_string()).collect();
    for l in &v.limitations {
        // The CPU-only line stands for the cpu product; with the GPU the
        // NRAS line below replaces it.
        if gpu.is_some() && l == crate::snp::CPU_ONLY {
            continue;
        }
        limitations.push(l.clone());
    }
    if gpu.is_some() {
        limitations.push(SNP_GPU_LIMITATION.into());
    }
    limitations.extend(launch.iter().filter(|l| l.contains("not judged")).cloned());
    limitations.push(format!("verdict verifier: {verifier}"));
    if exp.reference.explicit {
        limitations.push(format!(
            "SEV-SNP reference given explicitly: {} (sha256 {})",
            exp.reference.path.display(),
            exp.reference.sha256_hex
        ));
    }
    let t = v.reported_tcb;
    let m = v.minimum;
    let summary = vec![
        format!(
            "CPU   AMD SEV-SNP report genuine ({} {} chained to the pinned ARK, its {} not on AMD's CRL fetched now from KDS), TCB boot loader {:#04x} TEE {:#04x} SNP {:#04x} microcode {:#04x} at or above {} ; VMPL 0, debug off, no migration agent, SMT {}",
            v.product.name(),
            v.signing_key.name(),
            v.signing_key.intermediate(),
            t.boot_loader,
            t.tee,
            t.snp,
            t.microcode,
            v.minimum_source,
            match reference.guest_policy.smt {
                crate::snp::SnpSmt::Forbidden => "off",
                crate::snp::SnpSmt::Allowed => "allowed by the reference",
            }
        ),
        format!(
            "Image {} {} (reference signed {} by archive subkey with key ID {}{}; logged at index {} of {}, size {}): {}",
            reference.image.name,
            reference.image.version,
            crate::openpgp::utc_date(exp.reference.signed_by().signed_at_unix),
            exp.reference.signed_by().subkey_key_id(),
            if exp.reference.signed_by().test_key {
                "; TEST key, not for release"
            } else {
                ""
            },
            exp.reference.logged().index,
            exp.reference.logged().origin,
            exp.reference.logged().size,
            launch.join("; "),
        ),
        if gpu_required {
            format!("GPU   {} GPU(s), confidential computing on (NRAS tokens, nonce bound to this session, then to each connection's TLS exporter and SEV-SNP report)", doc.gpu.len())
        } else {
            "GPU   not attested: the cpu product (SEV-SNP only)".to_string()
        },
        "Link  TLS key and your session config are bound to this report (REPORT_DATA)".to_string(),
        format!(
            "Model {}, digest {}... matches the catalog",
            doc.model.name,
            &doc.model.digest_hex[..8]
        ),
        format!(
            "TCB   {}",
            crate::tcb_disclosure::TcbDisclosure::amd(&v, now).line
        ),
    ];
    let snp = SnpView {
        product: v.product.name().into(),
        signing_key: v.signing_key.name().into(),
        csp_id: v.csp_id.clone(),
        key_sha256_hex: v.key_sha256_hex.clone(),
        report_version: v.report.version,
        measurement_hex: hex(&v.report.measurement),
        host_data_hex: hex(&v.report.host_data),
        reported_tcb: tcb5(t),
        minimum_tcb: [m.boot_loader, m.tee, m.snp, m.microcode, m.fmc],
        minimum_source: v.minimum_source.clone(),
        crl_url,
        crl_this_update_unix: v.crl_window.0,
        crl_next_update_unix: v.crl_window.1,
    };
    let report = Report {
        verdict: serde_json::to_value(&verdict).unwrap_or_default(),
        environment: format!("{:?}", identity.environment),
        session_pubkey_hex: doc.binding.session_pubkey_hex.clone(),
        model_name: doc.model.name.clone(),
        model_digest_hex: doc.model.digest_hex.clone(),
        tcb_platform: format!(
            "SEV-SNP boot loader {:#04x} TEE {:#04x} SNP {:#04x} microcode {:#04x} FMC {:#04x}",
            t.boot_loader, t.tee, t.snp, t.microcode, t.fmc
        ),
        tcb_qe: SNP_TCB_QE.into(),
        endorsement_signer: format!("AMD {} {}", v.product.name(), v.signing_key.name()),
        endorsement_timestamp_unix: 0,
        firmware_svn: 0,
        mrtd_hex: String::new(),
        image_name: reference.image.name.clone(),
        image_version: reference.image.version.clone(),
        rtmr0_hex: String::new(),
        rtmr0_known: None,
        rtmr0_pinned: false,
        firmware_sha384_hex: String::new(),
        sections: Vec::new(),
        gpu_count: if gpu_required { doc.gpu.len() } else { 0 },
        reference_path: exp.reference.path.display().to_string(),
        reference_sha256_hex: exp.reference.sha256_hex.clone(),
        reference_explicit: exp.reference.explicit,
        reference_signer_subkey: exp.reference.signed_by().subkey_fingerprint.clone(),
        reference_signed_at_unix: exp.reference.signed_by().signed_at_unix,
        summary,
        limitations,
        snp: Some(snp),
        tcb: Some(crate::tcb_disclosure::TcbDisclosure::amd(&v, now)),
    };
    Ok((report, SnpCrl { der: crl }))
}

/// What every connection's SEV-SNP report must show once the document
/// passed [`verify_snp_document`]: the same guest on the same chip and TCB
/// (every signed byte but `REPORT_DATA`, and the same VCEK or VLEK), and
/// `BindingV3` over THIS connection's exporter; with the GPU product, that
/// connection's NRAS tokens for `session_gpu_nonce(exporter, the report's
/// signed bytes)`.
pub struct SnpSessionCheck {
    binding: BindingV2,
    doc_body: Vec<u8>,
    doc_key_sha256_hex: String,
    crl_der: Vec<u8>,
    anchors: crate::snp::SnpAnchors,
    policy: crate::snp::SnpPolicy,
    gpu: Option<SessionGpuCheck>,
    clock: Arc<dyn Fn() -> u64 + Send + Sync>,
}

impl SnpSessionCheck {
    /// `doc_payload` is verified again here, under `reference`: a check
    /// built on an unverified report would compare against the attacker's.
    fn new(
        doc_payload: &[u8],
        crl_der: &[u8],
        binding: BindingV2,
        reference: &crate::snp_reference::SnpReference,
        anchors: &crate::snp::SnpAnchors,
        mode: crate::tcb::TcbMode,
        gpu: Option<SessionGpuCheck>,
        clock: Arc<dyn Fn() -> u64 + Send + Sync>,
    ) -> Result<Self, Refusal> {
        let (v, _) = verify_snp_evidence(
            doc_payload,
            crl_der,
            &binding,
            reference,
            anchors,
            mode,
            clock(),
        )?;
        let (report, _) =
            crate::snp::split_evidence(doc_payload).map_err(|e| refuse(65, e.to_string()))?;
        let doc_body = crate::snp::signed_body_without_report_data(report)
            .ok_or_else(|| refuse(65, "SEV-SNP report of the wrong length"))?;
        Ok(Self {
            binding,
            doc_body,
            doc_key_sha256_hex: v.key_sha256_hex,
            crl_der: crl_der.to_vec(),
            anchors: anchors.clone(),
            policy: reference.policy_with(mode),
            gpu,
            clock,
        })
    }

    /// The report of the connection that exported `exporter`.
    fn verify_report(&self, session_payload: &[u8], exporter: &[u8; 32]) -> Result<(), Refusal> {
        let v = crate::snp::verify_snp_report(
            session_payload,
            &self.crl_der,
            &self.anchors,
            &self.policy,
            (self.clock)(),
        )
        .map_err(|e| refuse(65, format!("session SEV-SNP report: {e}")))?;
        if v.key_sha256_hex != self.doc_key_sha256_hex {
            return Err(refuse(
                88,
                "the session report comes from another chip or TCB than the document's (another VCEK or VLEK)",
            ));
        }
        let (report, _) =
            crate::snp::split_evidence(session_payload).map_err(|e| refuse(65, e.to_string()))?;
        if crate::snp::signed_body_without_report_data(report).as_deref()
            != Some(&self.doc_body[..])
        {
            return Err(refuse(
                88,
                "the session report describes another guest than the document's (launch digest, report ID, policy or TCB)",
            ));
        }
        let v3 = crate::binding::BindingV3::of(&self.binding, *exporter);
        if v3.snp_report_data() != *v.report_data() {
            return Err(refuse(
                88,
                "the session report does not bind this connection's TLS exporter: the TLS session is not the guest's (relay or replay)",
            ));
        }
        Ok(())
    }

    fn verify(
        &self,
        session: &crate::attest_v2::SessionDoc,
        exporter: &[u8; 32],
    ) -> Result<(), Refusal> {
        if session.schema != crate::attest_v2::SESSION_SCHEMA {
            return Err(refuse(
                88,
                format!(
                    "session attestation schema {:?}, expected {:?}",
                    session.schema,
                    crate::attest_v2::SESSION_SCHEMA
                ),
            ));
        }
        let payload = snp_payload(
            &session.quote_b64,
            "",
            session.snp.as_ref(),
            "session attestation",
        )?;
        self.verify_report(&payload, exporter)?;
        if let Some(gpu) = &self.gpu {
            let signed = payload
                .get(..x0ne_confidential::snp_report::SIGNED_LEN)
                .ok_or_else(|| refuse(65, "session SEV-SNP report too short"))?;
            gpu.verify(session.gpu.as_ref(), exporter, signed, (self.clock)())?;
        }
        Ok(())
    }
}

// ── The TD's client ──────────────────────────────────────────────────────────

/// `GET /v2/ready`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReadyStatus {
    pub ready: bool,
    pub state: String,
    #[serde(default)]
    pub detail: Option<String>,
    #[serde(default)]
    pub code: Option<i32>,
    #[serde(default)]
    pub gpu_ready_state: Option<u32>,
    #[serde(default)]
    pub model_digest_hex: Option<String>,
    /// Longest prompt plus completion the TD accepts; absent before 0.2.0-beta.6.
    #[serde(default)]
    pub context_tokens: Option<u64>,
    /// Tokens per prefill pass at the full context.
    #[serde(default)]
    pub prefill_chunk: Option<u64>,
    /// The TD streams chat answers; absent, false, from a TD that does not.
    #[serde(default)]
    pub streaming: bool,
    /// `(prompt tokens, chunk)` rows, the chunk being sized per prompt; absent in older files.
    #[serde(default)]
    pub prefill_chunks: Option<Vec<(u64, u64)>>,
}

const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
/// One chat, headers and body, whole or streamed.
pub const CHAT_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(15);
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// One TLS connection to the TD and its exporter.
struct Conn {
    send: hyper::client::conn::http1::SendRequest<axum::body::Body>,
    exporter: [u8; 32],
    /// The task driving the connection: aborted, the TCP connection closes.
    task: tokio::task::AbortHandle,
}

/// A connection dropped is closed, never left to hyper: a stream's TD then
/// sees its reader leave and stops generating.
impl Drop for Conn {
    fn drop(&mut self) {
        self.task.abort();
    }
}

type Pool = Arc<std::sync::Mutex<Vec<Conn>>>;

/// The body of a streamed answer, read a frame at a time. Behind a trait so
/// the relay's reading of it is tested with a scripted source.
pub trait FrameSource: Send + 'static {
    /// The next bytes; `Ok(None)` at the clean end, `Err` if the body broke.
    /// Cancel-safe: nothing is lost when the future is dropped unfinished.
    fn next_bytes(
        &mut self,
    ) -> impl std::future::Future<Output = Result<Option<hyper::body::Bytes>, BodyBroke>> + Send;
    /// The body ended clean, whole and right or right after its error event:
    /// its connection may carry another request.
    fn keep(self);
}

/// A streamed body ended without its clean end, or past its size bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BodyBroke;

/// A streamed answer from the TD. Dropped without [`FrameSource::keep`], its
/// connection is closed: the TD sees the reader leave and stops generating.
pub struct TdStream {
    body: std::pin::Pin<Box<hyper::body::Incoming>>,
    conn: Conn,
    pool: Pool,
    read: usize,
}

impl FrameSource for TdStream {
    async fn next_bytes(&mut self) -> Result<Option<hyper::body::Bytes>, BodyBroke> {
        use hyper::body::Body as _;
        loop {
            let frame = std::future::poll_fn(|cx| self.body.as_mut().poll_frame(cx)).await;
            let Some(frame) = frame else {
                return Ok(None);
            };
            // Trailers are not part of this protocol: a broken body too.
            let data = frame
                .map_err(|_| BodyBroke)?
                .into_data()
                .map_err(|_| BodyBroke)?;
            self.read += data.len();
            if self.read > MAX_BODY_BYTES {
                return Err(BodyBroke);
            }
            if !data.is_empty() {
                return Ok(Some(data));
            }
        }
    }

    fn keep(self) {
        if !self.conn.send.is_closed() {
            self.pool
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(self.conn);
        }
    }
}

/// How the TD took a streamed chat.
pub enum ChatStream {
    /// Not 2xx: the body read whole, as [`TdClient::chat`] gives it.
    Status(TdAnswer),
    /// 2xx, and whether its content type is `text/event-stream`.
    Open { event_stream: bool, body: TdStream },
}

/// How the TD took an SDK's chunked sealed request.
pub enum SealedStream {
    /// Not 2xx: the TD's refusal, read whole, with its `Retry-After`.
    Refused {
        status: axum::http::StatusCode,
        retry_after: Option<u32>,
        body: hyper::body::Bytes,
    },
    /// 2xx, and whether its content type is `message/ohttp-chunked-res`.
    Open { chunked: bool, body: TdStream },
}

/// A TD answer read whole.
#[derive(Debug)]
pub struct TdAnswer {
    pub status: axum::http::StatusCode,
    /// The body as JSON, or the start of its text as `{"error": ...}`.
    pub body: serde_json::Value,
    /// `Retry-After` in delay-seconds, when the TD sent one; 120 at most.
    pub retry_after: Option<u32>,
}

/// The longest `Retry-After` taken from the TD, and so passed on to a client.
const MAX_RETRY_AFTER_SECS: u32 = 120;

/// `Retry-After` as delay-seconds (RFC 9110: digits only), capped at
/// [`MAX_RETRY_AFTER_SECS`]; an HTTP date or anything else is dropped.
fn retry_after_secs(headers: &axum::http::HeaderMap) -> Option<u32> {
    let v = headers.get(axum::http::header::RETRY_AFTER)?.as_bytes();
    if v.is_empty() || v.len() > 9 || !v.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let secs: u32 = std::str::from_utf8(v).ok()?.parse().ok()?;
    Some(secs.min(MAX_RETRY_AFTER_SECS))
}

/// A body as JSON, or the start of its text as an error.
fn json_or_error(bytes: &[u8]) -> serde_json::Value {
    serde_json::from_slice(bytes).unwrap_or_else(
        |_| serde_json::json!({ "error": String::from_utf8_lossy(&bytes[..bytes.len().min(500)]) }),
    )
}

/// Why a request got no answer from the TD.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TdError {
    /// The TD's quote limits turned a new connection away (429, or 503
    /// `busy`): it answered over the pinned channel and is at its limit.
    /// Nothing else was sent on that connection.
    Busy(String),
    /// No answer, or one the session cannot go on with.
    Failed(String),
}

impl std::fmt::Display for TdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy(why) | Self::Failed(why) => f.write_str(why),
        }
    }
}

impl From<String> for TdError {
    fn from(why: String) -> Self {
        Self::Failed(why)
    }
}

impl From<TdError> for String {
    fn from(e: TdError) -> Self {
        e.to_string()
    }
}

/// Pinned, mutually authenticated HTTP/1.1 connections to one TD. No request
/// but the two attestation GETs and an SDK's e2e key GET ([`Self::e2e_key`],
/// whose answer this relay neither reads nor vouches for) goes over a
/// connection whose session quote has not passed [`SessionCheck`]; every new
/// connection is checked first.
pub struct TdClient {
    connector: tokio_rustls::TlsConnector,
    endpoint: String,
    host: String,
    client_nonce: [u8; 32],
    pin: Arc<ServerPin>,
    /// The connection the document came on, not yet checked.
    pending: tokio::sync::Mutex<Option<Conn>>,
    /// Checked connections, free.
    idle: Pool,
    check: std::sync::OnceLock<SessionCheck>,
    /// The first check a later connection failed: the session must stop.
    refused: std::sync::Mutex<Option<Refusal>>,
}

impl TdClient {
    /// `endpoint` is `host:port`. TLS 1.3, our certificate offered, the
    /// server pinned on first use (`tls::client_config`); `client_nonce` is
    /// the config's, the exporter context.
    pub fn new(
        identity: &TlsIdentity,
        endpoint: &str,
        client_nonce: [u8; 32],
    ) -> Result<Self, String> {
        let pin = ServerPin::new();
        let tls =
            crate::tls::client_config(identity, pin.clone()).map_err(|e| format!("TLS: {e}"))?;
        if endpoint.is_empty() || endpoint.contains('/') {
            return Err(format!("endpoint {endpoint:?} is not host:port"));
        }
        let host = endpoint
            .rsplit_once(':')
            .map(|(h, _)| h)
            .ok_or_else(|| format!("endpoint {endpoint:?} is not host:port"))?;
        let host = host
            .trim_start_matches('[')
            .trim_end_matches(']')
            .to_string();
        rustls::pki_types::ServerName::try_from(host.clone())
            .map_err(|e| format!("endpoint {endpoint:?}: {e}"))?;
        Ok(Self {
            connector: tokio_rustls::TlsConnector::from(Arc::new(tls)),
            endpoint: endpoint.to_string(),
            host,
            client_nonce,
            pin,
            pending: tokio::sync::Mutex::new(None),
            idle: Pool::default(),
            check: std::sync::OnceLock::new(),
            refused: std::sync::Mutex::new(None),
        })
    }

    pub fn pinned_spki(&self) -> Option<[u8; 32]> {
        self.pin.pinned()
    }

    /// Server certificates checked: one per connection, none resumed.
    pub fn server_checks(&self) -> u64 {
        self.pin.checks()
    }

    /// Why a later connection was refused, once one was.
    pub fn refusal(&self) -> Option<Refusal> {
        self.refused
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    async fn open(&self) -> Result<Conn, String> {
        let tcp = tokio::time::timeout(
            CONNECT_TIMEOUT,
            tokio::net::TcpStream::connect(&self.endpoint),
        )
        .await
        .map_err(|_| format!("connect {}: timed out", self.endpoint))?
        .map_err(|e| format!("connect {}: {e}", self.endpoint))?;
        let _ = tcp.set_nodelay(true);
        let name = rustls::pki_types::ServerName::try_from(self.host.clone())
            .map_err(|e| format!("server name: {e}"))?;
        let tls = tokio::time::timeout(CONNECT_TIMEOUT, self.connector.connect(name, tcp))
            .await
            .map_err(|_| "TLS handshake: timed out".to_string())?
            .map_err(|e| format!("TLS handshake: {e}"))?;
        let exporter = crate::tls::exporter(tls.get_ref().1, &self.client_nonce)
            .map_err(|e| format!("TLS exporter: {e}"))?;
        let (send, conn) = hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(tls))
            .await
            .map_err(|e| format!("HTTP: {e}"))?;
        let task = tokio::spawn(async move {
            let _ = conn.await;
        })
        .abort_handle();
        Ok(Conn {
            send,
            exporter,
            task,
        })
    }

    async fn exchange(
        &self,
        conn: &mut Conn,
        method: axum::http::Method,
        path: &str,
        body: Option<Vec<u8>>,
        timeout: std::time::Duration,
    ) -> Result<TdAnswer, String> {
        let body = body.map(|b| ("application/json", b));
        let (status, retry_after, bytes) =
            self.exchange_raw(conn, method, path, body, timeout).await?;
        Ok(TdAnswer {
            status,
            body: json_or_error(&bytes),
            retry_after,
        })
    }

    /// One request, its body of this content type, the answer's bytes as they came.
    async fn exchange_raw(
        &self,
        conn: &mut Conn,
        method: axum::http::Method,
        path: &str,
        body: Option<(&'static str, Vec<u8>)>,
        timeout: std::time::Duration,
    ) -> Result<(axum::http::StatusCode, Option<u32>, hyper::body::Bytes), String> {
        let what = format!("{method} {path}");
        let mut req = axum::http::Request::builder()
            .method(method)
            .uri(path)
            .header(axum::http::header::HOST, &self.endpoint);
        let body = match body {
            Some((content_type, b)) => {
                req = req.header(axum::http::header::CONTENT_TYPE, content_type);
                b
            }
            None => Vec::new(),
        };
        let req = req
            .body(axum::body::Body::from(body))
            .map_err(|e| format!("{what}: {e}"))?;
        let exchange = async {
            conn.send
                .ready()
                .await
                .map_err(|e| format!("{what}: {e}"))?;
            let resp = conn
                .send
                .send_request(req)
                .await
                .map_err(|e| format!("{what}: {e}"))?;
            let status = resp.status();
            let retry_after = retry_after_secs(resp.headers());
            let bytes =
                axum::body::to_bytes(axum::body::Body::new(resp.into_body()), MAX_BODY_BYTES)
                    .await
                    .map_err(|e| format!("{what}: {e}"))?;
            Ok::<_, String>((status, retry_after, bytes))
        };
        tokio::time::timeout(timeout, exchange)
            .await
            .map_err(|_| format!("{what}: timed out"))?
    }

    /// `GET /v2/e2e/key?nonce=<nonce>` for an SDK, on a new connection to
    /// the pinned TD: the TD quotes once per connection, and a checked one
    /// took its session quote. The answer is the TD's status and bytes as
    /// they came: the SDK verifies the document, this relay does not.
    pub async fn e2e_key(
        &self,
        nonce: &[u8; 32],
    ) -> Result<(axum::http::StatusCode, hyper::body::Bytes), TdError> {
        if let Some(r) = self.refusal() {
            return Err(TdError::Failed(format!("session stopped: {r}")));
        }
        let path = format!("/v2/e2e/key?nonce={}", hex(nonce));
        let mut conn = self.open().await.map_err(TdError::Failed)?;
        let (status, _, bytes) = self
            .exchange_raw(
                &mut conn,
                axum::http::Method::GET,
                &path,
                None,
                REQUEST_TIMEOUT,
            )
            .await
            .map_err(TdError::Failed)?;
        Ok((status, bytes))
    }

    /// `POST /v2/e2e/chat` for an SDK: its sealed request, carried as it
    /// came, on a checked connection; the sealed answer as it came, with the
    /// TD's `Retry-After`. Nothing here can open either.
    pub async fn e2e_chat(
        &self,
        sealed: Vec<u8>,
    ) -> Result<(axum::http::StatusCode, Option<u32>, hyper::body::Bytes), TdError> {
        let mut conn = self.checked().await?;
        let out = self
            .exchange_raw(
                &mut conn,
                axum::http::Method::POST,
                "/v2/e2e/chat",
                Some((x0ne_sdk::ohttp::MEDIA_TYPE_REQUEST, sealed)),
                CHAT_TIMEOUT,
            )
            .await;
        if out.is_ok() {
            self.idle
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(conn);
        }
        out.map_err(TdError::Failed)
    }

    /// `POST /v2/e2e/chat` for an SDK's chunked request
    /// (`message/ohttp-chunked-req`), carried as it came, on a checked
    /// connection; the TD's head by `deadline`. A 2xx hands the body over
    /// unread, frame by frame: its connection returns to the pool only
    /// through [`FrameSource::keep`]. Nothing here can open either side.
    pub async fn e2e_chat_stream(
        &self,
        sealed: Vec<u8>,
        deadline: tokio::time::Instant,
    ) -> Result<SealedStream, TdError> {
        let what = "POST /v2/e2e/chat";
        let mut conn = self.checked().await?;
        let req = axum::http::Request::builder()
            .method(axum::http::Method::POST)
            .uri("/v2/e2e/chat")
            .header(axum::http::header::HOST, &self.endpoint)
            .header(
                axum::http::header::CONTENT_TYPE,
                x0ne_sdk::ohttp::MEDIA_TYPE_CHUNKED_REQUEST,
            )
            .body(axum::body::Body::from(sealed))
            .map_err(|e| format!("{what}: {e}"))?;
        let sent = async {
            conn.send
                .ready()
                .await
                .map_err(|e| format!("{what}: {e}"))?;
            conn.send
                .send_request(req)
                .await
                .map_err(|e| format!("{what}: {e}"))
        };
        let resp = tokio::time::timeout_at(deadline, sent)
            .await
            .map_err(|_| format!("{what}: timed out"))??;
        let status = resp.status();
        if !status.is_success() {
            let retry_after = retry_after_secs(resp.headers());
            let read =
                axum::body::to_bytes(axum::body::Body::new(resp.into_body()), MAX_BODY_BYTES);
            let body = tokio::time::timeout_at(deadline, read)
                .await
                .map_err(|_| format!("{what}: timed out"))?
                .map_err(|e| format!("{what}: {e}"))?;
            self.idle
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(conn);
            return Ok(SealedStream::Refused {
                status,
                retry_after,
                body,
            });
        }
        let chunked = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v == x0ne_sdk::ohttp::MEDIA_TYPE_CHUNKED_RESPONSE);
        Ok(SealedStream::Open {
            chunked,
            body: TdStream {
                body: Box::pin(resp.into_body()),
                conn,
                pool: self.idle.clone(),
                read: 0,
            },
        })
    }

    async fn session_doc(&self, conn: &mut Conn) -> Result<crate::attest_v2::SessionDoc, TdError> {
        let TdAnswer {
            status,
            body: value,
            ..
        } = self
            .exchange(
                conn,
                axum::http::Method::GET,
                "/v2/attestation/session",
                None,
                REQUEST_TIMEOUT,
            )
            .await?;
        if !status.is_success() {
            let why = format!("/v2/attestation/session answered {status}: {value}");
            // The TD's quote limits (`cloud_serve::refused`); a quote that
            // failed in the TD (503 "session quote unavailable") is no limit.
            let limited = status == axum::http::StatusCode::TOO_MANY_REQUESTS
                || (status == axum::http::StatusCode::SERVICE_UNAVAILABLE
                    && value["error"] == "busy");
            return Err(if limited {
                TdError::Busy(why)
            } else {
                TdError::Failed(why)
            });
        }
        serde_json::from_value(value)
            .map_err(|e| TdError::Failed(format!("/v2/attestation/session: {e}")))
    }

    /// `GET /v2/attestation` on a new connection, kept for [`Self::establish`].
    pub async fn attestation(&self) -> Result<Document, String> {
        let mut conn = self
            .open()
            .await
            .map_err(|e| format!("GET /v2/attestation: {e}"))?;
        let TdAnswer {
            status,
            body: value,
            ..
        } = self
            .exchange(
                &mut conn,
                axum::http::Method::GET,
                "/v2/attestation",
                None,
                REQUEST_TIMEOUT,
            )
            .await?;
        if !status.is_success() {
            return Err(format!("/v2/attestation answered {status}: {value}"));
        }
        let doc = serde_json::from_value(value)
            .map_err(|e| format!("/v2/attestation is not an attestation document: {e}"))?;
        *self.pending.lock().await = Some(conn);
        Ok(doc)
    }

    /// The verdict passed: check the session quote of the connection the
    /// document came on, then admit it. Nothing else is sent before.
    pub async fn establish(&self, check: SessionCheck) -> Result<(), Refusal> {
        let mut conn = self
            .pending
            .lock()
            .await
            .take()
            .ok_or_else(|| refuse(88, "no connection carried the document"))?;
        let session = self
            .session_doc(&mut conn)
            .await
            .map_err(|e| refuse(88, format!("session attestation: {e}")))?;
        check.verify(&session, &conn.exporter)?;
        self.check
            .set(check)
            .map_err(|_| refuse(88, "the session was already established"))?;
        self.idle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(conn);
        Ok(())
    }

    /// A new connection whose session quote passed the check, or the refusal.
    pub async fn connect_verified(&self) -> Result<(), String> {
        let conn = self.verified_conn_new().await?;
        self.idle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(conn);
        Ok(())
    }

    async fn verified_conn_new(&self) -> Result<Conn, TdError> {
        let check = self.check.get().ok_or_else(|| {
            TdError::Failed("the session is not established: nothing is sent".into())
        })?;
        let mut conn = self.open().await?;
        let session = self.session_doc(&mut conn).await?;
        if let Err(r) = check.verify(&session, &conn.exporter) {
            let mut refused = self.refused.lock().unwrap_or_else(|e| e.into_inner());
            refused.get_or_insert(r.clone());
            return Err(TdError::Failed(format!("new connection refused: {r}")));
        }
        Ok(conn)
    }

    /// A checked connection: a free one still open, else a new one.
    async fn checked(&self) -> Result<Conn, TdError> {
        if let Some(r) = self.refusal() {
            return Err(TdError::Failed(format!("session stopped: {r}")));
        }
        loop {
            let free = self.idle.lock().unwrap_or_else(|e| e.into_inner()).pop();
            match free {
                Some(c) if !c.send.is_closed() => return Ok(c),
                Some(_) => {}
                None => return self.verified_conn_new().await,
            }
        }
    }

    async fn request(
        &self,
        method: axum::http::Method,
        path: &str,
        body: Option<Vec<u8>>,
        timeout: std::time::Duration,
    ) -> Result<TdAnswer, TdError> {
        let mut conn = self.checked().await?;
        let out = self.exchange(&mut conn, method, path, body, timeout).await;
        if out.is_ok() {
            self.idle
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(conn);
        }
        out.map_err(TdError::Failed)
    }

    pub async fn post_key(&self, key: &[u8]) -> Result<(), String> {
        let body = zeroize::Zeroizing::new(crate::channel_key::KeyRequest::to_json(key));
        let TdAnswer {
            status,
            body: value,
            ..
        } = self
            .request(
                axum::http::Method::POST,
                "/v2/key",
                Some(body.to_string().into_bytes()),
                REQUEST_TIMEOUT,
            )
            .await?;
        if status != axum::http::StatusCode::ACCEPTED {
            return Err(format!(
                "/v2/key answered {status}: {}",
                value["error"].as_str().unwrap_or("")
            ));
        }
        Ok(())
    }

    /// The TD's recipient key for one release, checked against the session:
    /// the key the owner's model key may be wrapped to, or the refusal.
    pub async fn model_key_recipient(
        &self,
        release_nonce: &[u8; 32],
    ) -> Result<crate::model_key::RecipientBinding, Refusal> {
        let check = self
            .check
            .get()
            .ok_or_else(|| refuse(88, "the session is not established: nothing is sent"))?;
        let path = format!("/v2/model-key/recipient?nonce={}", hex(release_nonce));
        let TdAnswer {
            status,
            body: value,
            ..
        } = self
            .request(axum::http::Method::GET, &path, None, REQUEST_TIMEOUT)
            .await
            .map_err(|e| refuse(88, format!("recipient quote: {e}")))?;
        if !status.is_success() {
            return Err(refuse(
                70,
                format!(
                    "the TD gave no recipient quote ({status}): {}",
                    value["error"].as_str().unwrap_or("")
                ),
            ));
        }
        let doc: crate::model_key::RecipientDoc = serde_json::from_value(value)
            .map_err(|e| refuse(88, format!("recipient document: {e}")))?;
        check.verify_recipient(&doc, release_nonce)
    }

    /// `POST /v2/model-key`: the owner's key, wrapped.
    pub async fn post_model_key(&self, wrapped: &crate::model_key::Wrapped) -> Result<(), String> {
        let body = serde_json::to_vec(wrapped).map_err(|e| e.to_string())?;
        let TdAnswer {
            status,
            body: value,
            ..
        } = self
            .request(
                axum::http::Method::POST,
                "/v2/model-key",
                Some(body),
                REQUEST_TIMEOUT,
            )
            .await?;
        if status != axum::http::StatusCode::ACCEPTED {
            return Err(format!(
                "/v2/model-key answered {status}: {}",
                value["error"].as_str().unwrap_or("")
            ));
        }
        Ok(())
    }

    pub async fn ready(&self) -> Result<ReadyStatus, String> {
        let TdAnswer {
            status,
            body: value,
            ..
        } = self
            .request(axum::http::Method::GET, "/v2/ready", None, REQUEST_TIMEOUT)
            .await?;
        if !status.is_success() {
            return Err(format!("/v2/ready answered {status}"));
        }
        serde_json::from_value(value).map_err(|e| format!("/v2/ready: {e}"))
    }

    pub async fn chat(&self, body: &serde_json::Value) -> Result<TdAnswer, TdError> {
        let bytes = serde_json::to_vec(body).map_err(|e| format!("chat body: {e}"))?;
        self.request(
            axum::http::Method::POST,
            "/v1/chat/completions",
            Some(bytes),
            CHAT_TIMEOUT,
        )
        .await
    }

    /// `body` with `stream: true`, on a checked connection; the headers by
    /// `deadline`. A 2xx hands the body over unread: its connection returns
    /// to the pool only through [`FrameSource::keep`].
    pub async fn chat_stream(
        &self,
        body: &serde_json::Value,
        deadline: tokio::time::Instant,
    ) -> Result<ChatStream, TdError> {
        let what = "POST /v1/chat/completions";
        let bytes = serde_json::to_vec(body).map_err(|e| format!("chat body: {e}"))?;
        let mut conn = self.checked().await?;
        let req = axum::http::Request::builder()
            .method(axum::http::Method::POST)
            .uri("/v1/chat/completions")
            .header(axum::http::header::HOST, &self.endpoint)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(bytes))
            .map_err(|e| format!("{what}: {e}"))?;
        let sent = async {
            conn.send
                .ready()
                .await
                .map_err(|e| format!("{what}: {e}"))?;
            conn.send
                .send_request(req)
                .await
                .map_err(|e| format!("{what}: {e}"))
        };
        let resp = tokio::time::timeout_at(deadline, sent)
            .await
            .map_err(|_| format!("{what}: timed out"))??;
        let status = resp.status();
        if !status.is_success() {
            let retry_after = retry_after_secs(resp.headers());
            let read =
                axum::body::to_bytes(axum::body::Body::new(resp.into_body()), MAX_BODY_BYTES);
            let bytes = tokio::time::timeout_at(deadline, read)
                .await
                .map_err(|_| format!("{what}: timed out"))?
                .map_err(|e| format!("{what}: {e}"))?;
            self.idle
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(conn);
            return Ok(ChatStream::Status(TdAnswer {
                status,
                body: json_or_error(&bytes),
                retry_after,
            }));
        }
        let event_stream = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("text/event-stream"));
        Ok(ChatStream::Open {
            event_stream,
            body: TdStream {
                body: Box::pin(resp.into_body()),
                conn,
                pool: self.idle.clone(),
                read: 0,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every verdict says that a stream's text comes before its receipt (D1).
    #[test]
    fn the_limitations_say_a_streamed_text_precedes_its_receipt() {
        let line = LIMITATIONS
            .iter()
            .find(|l| l.contains("passed on before its receipt is verified"))
            .expect("a limitation on streamed text");
        assert!(line.contains("receipt_verified") && line.contains("unverified"));
    }

    #[test]
    fn http_dates_round_trip_and_match_the_rfc_example() {
        assert_eq!(
            parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT"),
            Some(784_111_777)
        );
        assert_eq!(http_date(784_111_777), "Sun, 06 Nov 1994 08:49:37 GMT");
        for t in [0u64, 951_782_400, 1_789_000_000, 4_102_444_800] {
            assert_eq!(parse_http_date(&http_date(t)), Some(t), "{t}");
        }
        assert_eq!(http_date(0), "Thu, 01 Jan 1970 00:00:00 GMT");
        assert_eq!(file_stamp(1_760_303_100), "2025-10-12T21-05Z");
        assert_eq!(parse_http_date("Sun, 06 Nov 1994 08:49:37 PST"), None);
        assert_eq!(parse_http_date("garbage"), None);
    }

    #[test]
    fn retry_after_is_read_as_delay_seconds_only() {
        let of = |v: &str| {
            let mut h = axum::http::HeaderMap::new();
            h.insert(axum::http::header::RETRY_AFTER, v.parse().unwrap());
            retry_after_secs(&h)
        };
        assert_eq!(of("5"), Some(5));
        assert_eq!(of("0"), Some(0));
        assert_eq!(of("120"), Some(120));
        assert_eq!(of("121"), Some(120));
        assert_eq!(of("999999999"), Some(120));
        for bad in [
            "",
            "+5",
            "-1",
            "5.0",
            " 5",
            "1e3",
            "1234567890",
            "Sun, 06 Nov 1994 08:49:37 GMT",
        ] {
            assert_eq!(of(bad), None, "{bad:?}");
        }
        assert_eq!(retry_after_secs(&axum::http::HeaderMap::new()), None);
    }

    #[test]
    fn percent_decoding_is_strict() {
        assert_eq!(percent_decode("a%20b+c").as_deref(), Some("a b c"));
        assert_eq!(percent_decode("a%2"), None);
        assert_eq!(percent_decode("%zz"), None);
    }

    #[test]
    fn production_anchors_are_the_pinned_constants() {
        let a = Anchors::production();
        assert!(a
            .pcs_base_url
            .starts_with("https://api.trustedservices.intel.com"));
        assert!(a
            .gce_bucket_base_url
            .starts_with("https://storage.googleapis.com/"));
        assert_eq!(a.gce_root_der, crate::gce_endorsement::ROOT_1_DER);
        assert_eq!(a.gce_root_spki_sha256_hex().unwrap().len(), 64);
        assert_eq!(a.tcb_policy, crate::tcb::TcbPolicy::production());
        assert_eq!(a.root_crl_url, crate::tcb::INTEL_SGX_ROOT_CA_CRL_URL);
        assert_eq!(
            crate::attest_v2::TdAnchors::production().root_crl_url,
            crate::tcb::INTEL_SGX_ROOT_CA_CRL_URL
        );
        assert_eq!(
            crate::attest_v2::TdAnchors::production().tcb_policy,
            crate::tcb::TcbPolicy::production()
        );
    }

    // ── AMD SEV-SNP ─────────────────────────────────────────────────────

    use crate::snp::tests::{genoa_evidence, test_anchors, NOW as SNP_NOW, TEST_GENOA_CRL};

    fn snp_binding() -> BindingV2 {
        BindingV2 {
            session_nonce: [0x01; 32],
            session_pubkey: [0x02; 32],
            tls_spki_sha256: [0x03; 32],
            config_sha256: [0x04; 32],
        }
    }

    /// The TEST reference: Genoa or Turin, 0x4D… among its two predicted
    /// launch digests, the TEST guest's TCB as its floor, SMT forbidden.
    fn snp_ref() -> crate::snp_reference::SnpReference {
        crate::snp_reference::tests::reference()
    }

    /// A Genoa report bound to `report_data`, launch digest 0x4D…
    fn bound(report_data: [u8; 64]) -> Vec<u8> {
        genoa_evidence(|s| s.report_data = report_data)
    }

    fn check_with(
        payload: &[u8],
        reference: &crate::snp_reference::SnpReference,
        anchors: &crate::snp::SnpAnchors,
    ) -> Result<crate::snp::VerifiedSnp, Refusal> {
        verify_snp_evidence(
            payload,
            TEST_GENOA_CRL,
            &snp_binding(),
            reference,
            anchors,
            crate::tcb::TcbMode::Strict,
            SNP_NOW,
        )
        .map(|(v, _)| v)
    }

    fn check(payload: &[u8]) -> Result<crate::snp::VerifiedSnp, Refusal> {
        check_with(payload, &snp_ref(), &test_anchors())
    }

    #[test]
    fn an_snp_report_bound_to_this_session_with_a_predicted_launch_digest_verifies() {
        let v = check(&bound(snp_binding().snp_report_data())).expect("verifies");
        assert_eq!(v.product, crate::snp::SnpProduct::Genoa);
        let (_, launch) = verify_snp_evidence(
            &bound(snp_binding().snp_report_data()),
            TEST_GENOA_CRL,
            &snp_binding(),
            &snp_ref(),
            &test_anchors(),
            crate::tcb::TcbMode::Strict,
            SNP_NOW,
        )
        .unwrap();
        assert!(launch.iter().any(|l| l.contains("predicts")), "{launch:?}");
    }

    /// Required refusal: REPORT_DATA not bound to this session. The same
    /// binding as TDX: another session, TLS key or config is another value.
    #[test]
    fn an_snp_report_of_another_session_tls_key_or_config_is_refused() {
        for i in 0..4 {
            let mut other = snp_binding();
            match i {
                0 => other.session_nonce[0] ^= 1,
                1 => other.session_pubkey[0] ^= 1,
                2 => other.tls_spki_sha256[0] ^= 1,
                _ => other.config_sha256[0] ^= 1,
            }
            let e = check(&bound(other.snp_report_data())).unwrap_err();
            assert_eq!(e.code, 88, "field {i}: {e}");
            assert!(e.reason.contains("REPORT_DATA"), "{e}");
        }
    }

    /// Required refusal: a launch digest the reference does not predict is
    /// genuine hardware running another image.
    #[test]
    fn an_snp_report_whose_launch_digest_is_not_predicted_is_refused() {
        let p = genoa_evidence(|s| {
            s.report_data = snp_binding().snp_report_data();
            s.measurement = [0x4E; 48];
        });
        let e = check(&p).unwrap_err();
        assert_eq!(e.code, 65, "{e}");
        assert!(e.reason.contains("not this image"), "{e}");
    }

    /// Required refusal: DEBUG, whatever the reference allows (no field of
    /// it can allow DEBUG; SMT allowed and Milan listed change nothing).
    #[test]
    fn a_debug_guest_is_refused_whatever_the_reference_says() {
        let p = genoa_evidence(|s| {
            s.report_data = snp_binding().snp_report_data();
            s.policy |= x0ne_confidential::snp_report::POLICY_DEBUG;
        });
        let mut lenient = snp_ref();
        lenient.guest_policy.smt = crate::snp::SnpSmt::Allowed;
        lenient.products.push("Milan".into());
        for r in [snp_ref(), lenient] {
            let e = check_with(&p, &r, &test_anchors()).unwrap_err();
            assert_eq!(e.code, 65);
            assert!(e.reason.contains("DEBUG"), "{e}");
        }
    }

    /// Required refusal: a TCB under the reference's written floor.
    #[test]
    fn a_tcb_under_the_written_floor_is_refused() {
        let p = bound(snp_binding().snp_report_data());
        for edit in [
            (|r: &mut crate::snp_reference::SnpReference| r.tcb_minimum.snp = 0x1C)
                as fn(&mut crate::snp_reference::SnpReference),
            |r| r.tcb_minimum.microcode = 0x59,
            |r| r.tcb_minimum.boot_loader = 0x0B,
            |r| r.tcb_minimum.tee = 1,
        ] {
            let mut r = snp_ref();
            edit(&mut r);
            let e = check_with(&p, &r, &test_anchors()).unwrap_err();
            assert_eq!(e.code, 65, "{e}");
            assert!(e.reason.contains("TCB"), "{e}");
        }
    }

    /// Required refusal: a chain to another root than the pinned one. The
    /// TEST chain under the production anchors (AMD's pinned ARKs).
    #[test]
    fn a_report_chained_to_another_root_is_refused() {
        let p = bound(snp_binding().snp_report_data());
        let e = check_with(&p, &snp_ref(), &crate::snp::SnpAnchors::production()).unwrap_err();
        assert_eq!(e.code, 65, "{e}");
        assert!(e.reason.contains("SEV-SNP report"), "{e}");
    }

    fn session_check() -> SnpSessionCheck {
        SnpSessionCheck::new(
            &bound(snp_binding().snp_report_data()),
            TEST_GENOA_CRL,
            snp_binding(),
            &snp_ref(),
            &test_anchors(),
            crate::tcb::TcbMode::Strict,
            None,
            Arc::new(|| SNP_NOW),
        )
        .expect("the document's report verifies")
    }

    fn v3(exporter: [u8; 32]) -> [u8; 64] {
        crate::binding::BindingV3::of(&snp_binding(), exporter).snp_report_data()
    }

    fn snp_session(payload: &[u8]) -> crate::attest_v2::SessionDoc {
        crate::attest_v2::SessionDoc {
            schema: crate::attest_v2::SESSION_SCHEMA.into(),
            quote_b64: String::new(),
            snp: Some(crate::attest_v2::SnpDoc {
                evidence_b64: crate::attest_v2::b64(payload),
            }),
            gpu: None,
        }
    }

    /// SECURITY: each connection's report binds its own TLS exporter; a
    /// party relaying the session runs two handshakes, two exporters.
    #[test]
    fn a_session_report_binds_this_connections_exporter_and_nothing_else() {
        let c = session_check();
        c.verify(&snp_session(&bound(v3([0x33; 32]))), &[0x33; 32])
            .expect("same guest, this connection");
        let e = c
            .verify(&snp_session(&bound(v3([0x34; 32]))), &[0x33; 32])
            .unwrap_err();
        assert_eq!(e.code, 88, "{e}");
        // The document's v2 report replayed on a connection.
        let e = c
            .verify(
                &snp_session(&bound(snp_binding().snp_report_data())),
                &[0x33; 32],
            )
            .unwrap_err();
        assert_eq!(e.code, 88, "{e}");
    }

    /// No fallback: a TDX quote in an SEV-SNP session, no SNP evidence, or
    /// another schema is refused before anything is parsed.
    #[test]
    fn a_session_attestation_that_is_not_snp_is_refused() {
        let c = session_check();
        let good = snp_session(&bound(v3([0x33; 32])));
        let mut tdx = good.clone();
        tdx.quote_b64 = "AAAA".into();
        let e = c.verify(&tdx, &[0x33; 32]).unwrap_err();
        assert_eq!(e.code, 88, "{e}");
        assert!(e.reason.contains("no fallback"), "{e}");
        let mut none = good.clone();
        none.snp = None;
        let e = c.verify(&none, &[0x33; 32]).unwrap_err();
        assert_eq!(e.code, 88, "{e}");
        let mut schema = good.clone();
        schema.schema = "x0ne-infer/session-attestation/v3".into();
        assert_eq!(c.verify(&schema, &[0x33; 32]).unwrap_err().code, 88);
        let mut garbled = good;
        garbled.snp = Some(crate::attest_v2::SnpDoc {
            evidence_b64: "not base64!".into(),
        });
        assert_eq!(c.verify(&garbled, &[0x33; 32]).unwrap_err().code, 65);
    }

    /// Azure's paravisor stays out: a genuine Azure report (AMD's chain,
    /// AMD's CRL of 2026-10-27) through the client path is refused under
    /// any written reference, by SMT, then by the bulletins' minimum (the
    /// floor only raises it), and its REPORT_DATA binds no session of ours.
    #[test]
    fn a_genuine_azure_paravisor_report_is_refused_by_the_client() {
        use crate::snp::tests::{AMD_GENOA_CRL, AZURE_GENOA_REPORT, AZURE_GENOA_VCEK, NOW_REAL};
        let p = crate::snp::evidence_payload(AZURE_GENOA_REPORT, AZURE_GENOA_VCEK);
        let run = |r: &crate::snp_reference::SnpReference| {
            verify_snp_evidence(
                &p,
                AMD_GENOA_CRL,
                &snp_binding(),
                r,
                &crate::snp::SnpAnchors::production(),
                crate::tcb::TcbMode::Strict,
                NOW_REAL,
            )
            .map(|_| ())
            .unwrap_err()
        };
        let e = run(&snp_ref());
        assert_eq!(e.code, 65);
        assert!(e.reason.contains("SMT"), "{e}");
        let mut lenient = snp_ref();
        lenient.guest_policy.smt = crate::snp::SnpSmt::Allowed;
        lenient.tcb_minimum = crate::snp_reference::TcbMinimum {
            boot_loader: 0,
            tee: 0,
            snp: 0,
            microcode: 0,
            fmc: 0,
        };
        let e = run(&lenient);
        assert_eq!(e.code, 65);
        assert!(e.reason.contains("TCB"), "{e}");
        let (report, _) = crate::snp::split_evidence(&p).unwrap();
        let parsed = x0ne_confidential::snp_report::parse(report).unwrap();
        assert!(!snp_binding().matches(&parsed.report_data));
    }

    /// The model key is not released to an SEV-SNP guest.
    #[test]
    fn an_snp_session_releases_no_model_key() {
        let c = SessionCheck(SessionKind::Snp(session_check()));
        let doc = crate::model_key::RecipientDoc {
            schema: crate::model_key::RECIPIENT_SCHEMA.into(),
            suite: crate::hpke::SUITE.into(),
            release_nonce_hex: hex(&[7; 32]),
            kem_pk_hex: hex(&[8; 32]),
            quote_b64: String::new(),
        };
        let e = c.verify_recipient(&doc, &[7; 32]).unwrap_err();
        assert_eq!(e.code, 88, "{e}");
        assert!(e.reason.contains("TDX guests only"), "{e}");
    }

    /// Another guest (another launch digest, another report ID) or another
    /// chip, even genuine and bound to this connection, is not the document's.
    #[test]
    fn a_session_report_from_another_guest_or_chip_is_refused() {
        let c = session_check();
        for edit in [
            (|s: &mut crate::snp::tests::Spec| s.measurement = [0x4C; 48])
                as fn(&mut crate::snp::tests::Spec),
            |s| s.report_id = [0x1E; 32],
            |s| s.host_data = [0x01; 32],
        ] {
            let p = genoa_evidence(|s| {
                edit(s);
                s.report_data = v3([0x33; 32]);
            });
            let e = c.verify(&snp_session(&p), &[0x33; 32]).unwrap_err();
            assert_eq!(e.code, 88, "{e}");
        }
        // Another TCB of the same chip: another VCEK.
        let mut spec = crate::snp::tests::Spec::genoa();
        spec.report_data = v3([0x33; 32]);
        for t in [
            &mut spec.reported_tcb,
            &mut spec.current_tcb,
            &mut spec.committed_tcb,
            &mut spec.launch_tcb,
        ] {
            *t = crate::snp::tests::GENOA_TCB_OLD;
        }
        let p = crate::snp::evidence_payload(
            &crate::snp::tests::sign(spec.bytes(), crate::snp::tests::OLD),
            crate::snp::tests::VCEK_OLD,
        );
        let e = c.verify(&snp_session(&p), &[0x33; 32]).unwrap_err();
        // Refused by the written floor before the comparison: still refused.
        assert!(matches!(e.code, 65 | 88), "{e}");
    }
}
