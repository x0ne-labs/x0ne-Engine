// SPDX-License-Identifier: Apache-2.0
//! Revocation of the NRAS signing chain by OCSP (RFC 6960), judged on
//! responses handed to the verifier, never fetched by it: the TD staples
//! them next to the JWKS it fetched (RFC 6066's model), and the relay and
//! the SDK judge them offline, as they judge the tokens.
//!
//! The chain is the JWKS `x5c`: the ES384 signing leaf (`x5c[0]`, valid 12 h
//! to 2 days) under a pinned intermediate (`x5c[1]`, 004 or 005), itself
//! under "NVIDIA Attestation Service CA 001". MEASURED 2026-10-31 on the
//! vendored certificates (`tests/nvidia_nras_gpu_*.der`, `openssl x509
//! -ext authorityInfoAccess,crlDistributionPoints`): the intermediates name
//! the responder `http://ocsp.ndis.nvidia.com` and a CRL; the leaf under 004
//! names neither; the leaf under 005 names a CRL and a `caIssuers` URL at
//! `prod.vault.nvidia.com`, no responder. So what a responder says about a
//! leaf is verified here under the pinned intermediate, the only issuer whose
//! key is at hand; what one says about an intermediate is signed by CA 001,
//! which NVIDIA does not publish: it is judged only when a caller pins CA 001
//! (`NrasOcspPolicy::intermediate_issuer_der`), and said unverifiable
//! otherwise, the intermediate staying pinned whole (its revocation reaches a
//! client with a release, as the archive key's).
//!
//! What a response must show before its status counts (RFC 6960 §3.2):
//! `successful`, `id-pkix-ocsp-basic`, one `SingleResponse` whose `CertID`
//! names the judged certificate under its issuer (SHA-1, SHA-256 or SHA-384
//! of the issuer's name and key, the serial); signed by the issuer's key or
//! by a responder certificate the response carries, issued by that issuer,
//! marked `id-kp-OCSPSigning`, inside its window (§4.2.2.2); `responderID`
//! the signer's (§4.2.2.3); `thisUpdate` not after now, `nextUpdate` present
//! (RFC 5019 §2.2.4, §4: absent, the response is refused) and not before
//! now, both within [`crate::nras::CLOCK_SKEW_SECS`]; issued at most
//! [`NrasOcspPolicy::max_age_secs`] ago, the day Intel's collateral gets
//! (`tcb::TcbPolicy::MAX_COLLATERAL_AGE_SECS`); the nonce echoed when the
//! fetcher sent one (RFC 8954), ignored when it did not (RFC 5019 §4); no
//! critical extension this module does not read.
//!
//! Two modes, as the TCB policy's. `Required`, the default of a policy written
//! without a mode: every certificate whose issuer is at hand must be `good`;
//! `revoked`, `unknown`, a stale, unsigned or missing response refuses.
//! `Transparent`, what production names until the TD staples responses: a
//! verified `revoked` still refuses; anything else is admitted and said, in
//! the verdict, by certificate. NVIDIA's own verifiers refuse `unknown` and
//! anything but `good` (nvtrust `cc_admin_utils.py`
//! `ocsp_certificate_chain_validation`; NVAT's default policy), and read no
//! `thisUpdate`/`nextUpdate`; both are read here.

use crate::nras::CLOCK_SKEW_SECS;
use rsa::pkcs1v15::{Signature as RsaSignature, VerifyingKey as RsaVerifyingKey};
use rsa::pkcs8::DecodePublicKey;
use rsa::signature::Verifier as _;
use rsa::RsaPublicKey;
use sha2::Digest;
use x509_parser::asn1_rs::{oid, Any, Class, FromDer, GeneralizedTime, Oid, Tag};
use x509_parser::oid_registry::{
    OID_EC_P256, OID_HASH_SHA1, OID_KEY_TYPE_EC_PUBLIC_KEY, OID_NIST_EC_P384, OID_NIST_HASH_SHA256,
    OID_NIST_HASH_SHA384, OID_PKCS1_RSAENCRYPTION, OID_PKCS1_SHA256WITHRSA,
    OID_PKCS1_SHA384WITHRSA, OID_SIG_ECDSA_WITH_SHA256, OID_SIG_ECDSA_WITH_SHA384,
};
use x509_parser::prelude::X509Certificate;

/// Which of two readings a verdict takes of the chain's revocation
/// (decision of 2026-10-31). Ordered: `Required` is stricter. `Default` is
/// `Required`, the fail-closed value of a policy written without a mode;
/// [`NrasOcspPolicy::production`] names `Transparent`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NrasOcspMode {
    /// A verified `revoked` refuses; `unknown`, a stale, unsigned, missing
    /// or unverifiable response is admitted and named in the verdict, by
    /// certificate. What production verifies with until the TD staples
    /// responses: before that, no response exists to require.
    Transparent,
    /// Every certificate whose issuer is at hand must show a verified,
    /// current `good`; anything else refuses, by name.
    #[default]
    Required,
}

impl NrasOcspMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Transparent => "transparent",
            Self::Required => "required",
        }
    }
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "transparent" => Ok(Self::Transparent),
            "required" => Ok(Self::Required),
            other => Err(format!(
                "NRAS OCSP policy {other:?}: \"transparent\" or \"required\""
            )),
        }
    }
}

/// How the NRAS signing chain's revocation is judged: the mode, the age a
/// response may have, the nonce the fetcher sent (none for a pre-produced
/// response, RFC 5019), and CA 001 when a caller pins it for the
/// intermediate's own status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NrasOcspPolicy {
    pub mode: NrasOcspMode,
    /// A response's `thisUpdate` at most this long before now, inside its
    /// own window: a cache that serves last week's `good` hides this week's
    /// revocation for that long. One day, as Intel's collateral.
    pub max_age_secs: u64,
    /// The nonce the fetcher put in its request (`id-pkix-ocsp-nonce`): the
    /// response must echo it. `None`: a pre-produced response, the nonce
    /// extension ignored (RFC 5019 §4).
    pub nonce: Option<Vec<u8>>,
    /// The DER certificate of the intermediates' issuer, CA 001, when
    /// pinned: the intermediate's own status is then verified under it.
    /// `None` in production (NVIDIA publishes it nowhere, searched
    /// 2026-10-31): the intermediate's status is said unverifiable.
    pub intermediate_issuer_der: Option<Vec<u8>>,
}

impl NrasOcspPolicy {
    /// One day: `tcb::TcbPolicy::MAX_COLLATERAL_AGE_SECS`, held equal by a test.
    pub const MAX_AGE_SECS: u64 = 24 * 3600;

    /// `Required`, one day, no nonce, no CA 001: the fail-closed policy.
    pub const fn required() -> Self {
        Self {
            mode: NrasOcspMode::Required,
            max_age_secs: Self::MAX_AGE_SECS,
            nonce: None,
            intermediate_issuer_der: None,
        }
    }

    /// `Transparent`, otherwise [`Self::required`]: what the relay, the TD's
    /// own gate and the SDK verify with (decision of 2026-10-31) until the
    /// TD staples the responses, which no TD does yet. Named in every
    /// verdict, with the standing of each certificate.
    pub const fn production() -> Self {
        Self {
            mode: NrasOcspMode::Transparent,
            max_age_secs: Self::MAX_AGE_SECS,
            nonce: None,
            intermediate_issuer_der: None,
        }
    }

    #[must_use]
    pub fn with_mode(self, mode: NrasOcspMode) -> Self {
        Self { mode, ..self }
    }

    /// True when `self` judges at least as strictly as `other`: a mode at
    /// least as strict, an age limit at most as long, a nonce required when
    /// `other` requires one, CA 001 pinned when `other` pins it.
    pub fn at_least(&self, other: &Self) -> bool {
        self.mode >= other.mode
            && self.max_age_secs <= other.max_age_secs
            && (other.nonce.is_none() || self.nonce.is_some())
            && (other.intermediate_issuer_der.is_none() || self.intermediate_issuer_der.is_some())
    }

    /// One line for a `config` event or a verdict.
    pub fn describe(&self) -> String {
        format!(
            "nras-ocsp {} (response age at most {} s; nonce {}; intermediate issuer CA 001 {})",
            self.mode.as_str(),
            self.max_age_secs,
            if self.nonce.is_some() {
                "required"
            } else {
                "not sent"
            },
            if self.intermediate_issuer_der.is_some() {
                "pinned"
            } else {
                "not pinned"
            }
        )
    }
}

impl Default for NrasOcspPolicy {
    fn default() -> Self {
        Self::required()
    }
}

/// What the responses said about one certificate, once verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OcspStatus {
    Good {
        this_update: u64,
        next_update: u64,
    },
    Revoked {
        at: u64,
        reason: Option<u64>,
    },
    Unknown,
    /// Responses were given and none verified for this certificate: the
    /// last reason. Under `Required` a refusal, under `Transparent` a line.
    Unverified(String),
    NoResponse,
    /// The issuer's key is not at hand (CA 001 not pinned): nothing a
    /// responder says about this certificate can be verified.
    IssuerNotPublished,
}

/// One certificate of the chain and what was established about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcspStanding {
    /// `x5c leaf` or `x5c intermediate`.
    pub what: &'static str,
    pub serial_hex: String,
    pub status: OcspStatus,
}

impl std::fmt::Display for OcspStanding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (serial {}): ", self.what, self.serial_hex)?;
        match &self.status {
            OcspStatus::Good {
                this_update,
                next_update,
            } => write!(
                f,
                "good (thisUpdate {this_update}, nextUpdate {next_update})"
            ),
            OcspStatus::Revoked { at, reason } => match reason {
                Some(r) => write!(f, "REVOKED at {at}, reason {r}"),
                None => write!(f, "REVOKED at {at}"),
            },
            OcspStatus::Unknown => write!(f, "unknown to the responder"),
            OcspStatus::Unverified(why) => write!(f, "no response verified ({why})"),
            OcspStatus::NoResponse => write!(f, "no response stapled"),
            OcspStatus::IssuerNotPublished => write!(
                f,
                "not judged: its issuer CA 001 is not published, the certificate is pinned whole"
            ),
        }
    }
}

/// Why the chain's revocation refused the token.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OcspRefused {
    #[error("{what} (serial {serial_hex}) is REVOKED since {at}{}: a verified OCSP response says so", reason.map(|r| format!(", reason {r}")).unwrap_or_default())]
    Revoked {
        what: &'static str,
        serial_hex: String,
        at: u64,
        reason: Option<u64>,
    },
    /// `Required` alone: a status other than a verified `good`.
    #[error("{what} (serial {serial_hex}): {standing}; this verifier requires a verified good OCSP response")]
    NotGood {
        what: &'static str,
        serial_hex: String,
        standing: String,
    },
}

#[rustfmt::skip]
const OID_OCSP_BASIC: Oid<'static> = oid!(1.3.6.1.5.5.7.48.1.1);
#[rustfmt::skip]
const OID_OCSP_NONCE: Oid<'static> = oid!(1.3.6.1.5.5.7.48.1.2);

/// `OCSPResponseStatus` (RFC 6960 §4.2.1), for a refusal's text.
fn response_status_name(n: u64) -> &'static str {
    match n {
        0 => "successful",
        1 => "malformedRequest",
        2 => "internalError",
        3 => "tryLater",
        5 => "sigRequired",
        6 => "unauthorized",
        _ => "unknown status",
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ── DER reading, on asn1-rs's `Any` (what x509-parser is built on) ──────────

/// The next element of `input`: its header and content, the whole TLV
/// (what a signature covers), and what follows.
fn next<'a>(input: &'a [u8], what: &str) -> Result<(Any<'a>, &'a [u8], &'a [u8]), String> {
    let (rest, any) = Any::from_der(input).map_err(|e| format!("{what}: {e}"))?;
    let tlv = &input[..input.len() - rest.len()];
    Ok((any, tlv, rest))
}

/// `input` as one element and nothing after it.
fn only<'a>(input: &'a [u8], what: &str) -> Result<(Any<'a>, &'a [u8]), String> {
    let (any, tlv, rest) = next(input, what)?;
    if !rest.is_empty() {
        return Err(format!("{what}: {} bytes after it", rest.len()));
    }
    Ok((any, tlv))
}

fn expect_tag(any: &Any<'_>, tag: Tag, what: &str) -> Result<(), String> {
    if any.class() != Class::Universal || any.tag() != tag {
        return Err(format!(
            "{what}: {} {:?}, not {tag:?}",
            match any.class() {
                Class::Universal => "universal",
                Class::ContextSpecific => "context-specific",
                Class::Application => "application",
                Class::Private => "private",
            },
            any.tag()
        ));
    }
    Ok(())
}

fn context(any: &Any<'_>) -> Option<u32> {
    (any.class() == Class::ContextSpecific).then_some(any.tag().0)
}

fn sequence<'a>(any: &Any<'a>, what: &str) -> Result<&'a [u8], String> {
    expect_tag(any, Tag::Sequence, what)?;
    Ok(any.data)
}

fn oid_of<'a>(any: &Any<'a>, what: &str) -> Result<Oid<'a>, String> {
    expect_tag(any, Tag::Oid, what)?;
    Ok(Oid::new(std::borrow::Cow::Borrowed(any.data)))
}

fn octets<'a>(any: &Any<'a>, what: &str) -> Result<&'a [u8], String> {
    expect_tag(any, Tag::OctetString, what)?;
    Ok(any.data)
}

fn generalized_time(any: &Any<'_>, what: &str) -> Result<u64, String> {
    expect_tag(any, Tag::GeneralizedTime, what)?;
    let t = GeneralizedTime::from_bytes(any.data).map_err(|e| format!("{what}: {e}"))?;
    let unix = t
        .utc_datetime()
        .map_err(|e| format!("{what}: {e}"))?
        .unix_timestamp();
    u64::try_from(unix).map_err(|_| format!("{what}: before 1970"))
}

/// A small non-negative INTEGER or ENUMERATED (a status, a version, a reason).
fn small_uint(any: &Any<'_>, tag: Tag, what: &str) -> Result<u64, String> {
    expect_tag(any, tag, what)?;
    if any.data.is_empty() || any.data.len() > 8 || any.data[0] & 0x80 != 0 {
        return Err(format!(
            "{what}: {} is not a small non-negative integer",
            hex(any.data)
        ));
    }
    Ok(any
        .data
        .iter()
        .fold(0u64, |acc, b| (acc << 8) | u64::from(*b)))
}

/// `AlgorithmIdentifier`: the OID (parameters unread).
fn algorithm(any: &Any<'_>, what: &str) -> Result<Oid<'static>, String> {
    let body = sequence(any, what)?;
    let (alg, _, _) = next(body, what)?;
    Ok(oid_of(&alg, what)?.to_owned())
}

/// An INTEGER's magnitude bytes, leading zero dropped: a serial as written
/// in a certificate and in a `CertID` compare equal this way.
fn magnitude(int: &[u8]) -> &[u8] {
    match int {
        [0, rest @ ..] if !rest.is_empty() => rest,
        other => other,
    }
}

// ── The response ────────────────────────────────────────────────────────────

struct Parsed<'a> {
    tbs: &'a [u8],
    signature_alg: Oid<'static>,
    signature: &'a [u8],
    certs: Vec<&'a [u8]>,
    responder_by_name: Option<&'a [u8]>,
    responder_by_key: Option<&'a [u8]>,
    single: &'a [u8],
    response_extensions: Option<&'a [u8]>,
}

/// `OCSPResponse` down to the one `SingleResponse`, nothing judged yet.
fn parse(der: &[u8]) -> Result<Parsed<'_>, String> {
    let (outer, _) = only(der, "OCSPResponse")?;
    let body = sequence(&outer, "OCSPResponse")?;
    let (status, _, rest) = next(body, "responseStatus")?;
    let status = small_uint(&status, Tag::Enumerated, "responseStatus")?;
    if status != 0 {
        return Err(format!(
            "the responder answered {} ({status}), not successful",
            response_status_name(status)
        ));
    }
    let (rb, _) = only(rest, "responseBytes")?;
    if context(&rb) != Some(0) {
        return Err("responseBytes: not [0]".into());
    }
    let (rb, _) = only(rb.data, "ResponseBytes")?;
    let rb = sequence(&rb, "ResponseBytes")?;
    let (kind, _, rest) = next(rb, "responseType")?;
    if oid_of(&kind, "responseType")? != OID_OCSP_BASIC {
        return Err(format!(
            "responseType {} is not id-pkix-ocsp-basic",
            oid_of(&kind, "responseType")?
        ));
    }
    let (basic, _) = only(rest, "response")?;
    let basic = octets(&basic, "response")?;

    let (basic, _) = only(basic, "BasicOCSPResponse")?;
    let basic = sequence(&basic, "BasicOCSPResponse")?;
    let (tbs_any, tbs, rest) = next(basic, "tbsResponseData")?;
    let tbs_body = sequence(&tbs_any, "tbsResponseData")?;
    let (alg, _, rest) = next(rest, "signatureAlgorithm")?;
    let signature_alg = algorithm(&alg, "signatureAlgorithm")?;
    let (sig, _, rest) = next(rest, "signature")?;
    expect_tag(&sig, Tag::BitString, "signature")?;
    let signature = match sig.data {
        [0, bits @ ..] => bits,
        _ => return Err("signature: not a whole number of bytes".into()),
    };
    let mut certs = Vec::new();
    if !rest.is_empty() {
        let (c, _) = only(rest, "certs")?;
        if context(&c) != Some(0) {
            return Err("certs: not [0]".into());
        }
        let (c, _) = only(c.data, "certs")?;
        let mut list = sequence(&c, "certs")?;
        while !list.is_empty() {
            let (cert, tlv, after) = next(list, "certs entry")?;
            expect_tag(&cert, Tag::Sequence, "certs entry")?;
            certs.push(tlv);
            list = after;
        }
    }

    // ResponseData: [0] version OPTIONAL, responderID, producedAt, responses, [1] extensions OPTIONAL.
    let (first, _, mut rest) = next(tbs_body, "ResponseData")?;
    let responder = if context(&first) == Some(0) {
        let (v, _) = only(first.data, "version")?;
        if small_uint(&v, Tag::Integer, "version")? != 0 {
            return Err("version is not v1".into());
        }
        let (r, _, after) = next(rest, "responderID")?;
        rest = after;
        r
    } else {
        first
    };
    let (responder_by_name, responder_by_key) = match context(&responder) {
        Some(1) => {
            let (name, tlv) = only(responder.data, "responderID byName")?;
            expect_tag(&name, Tag::Sequence, "responderID byName")?;
            (Some(tlv), None)
        }
        Some(2) => {
            let (key, _) = only(responder.data, "responderID byKey")?;
            (None, Some(octets(&key, "responderID byKey")?))
        }
        _ => return Err("responderID: neither byName [1] nor byKey [2]".into()),
    };
    let (produced, _, rest) = next(rest, "producedAt")?;
    generalized_time(&produced, "producedAt")?;
    let (responses, _, rest) = next(rest, "responses")?;
    let responses = sequence(&responses, "responses")?;
    let (single, single_tlv, after) = next(responses, "SingleResponse")?;
    expect_tag(&single, Tag::Sequence, "SingleResponse")?;
    if !after.is_empty() {
        return Err("more than one SingleResponse: one certificate was asked about".into());
    }
    let _ = single_tlv;
    let response_extensions = if rest.is_empty() {
        None
    } else {
        let (ext, _) = only(rest, "responseExtensions")?;
        if context(&ext) != Some(1) {
            return Err("responseExtensions: not [1]".into());
        }
        let (ext, _) = only(ext.data, "responseExtensions")?;
        Some(sequence(&ext, "responseExtensions")?)
    };
    Ok(Parsed {
        tbs,
        signature_alg,
        signature,
        certs,
        responder_by_name,
        responder_by_key,
        single: single.data,
        response_extensions,
    })
}

struct Single<'a> {
    hash_alg: Oid<'static>,
    issuer_name_hash: &'a [u8],
    issuer_key_hash: &'a [u8],
    serial: &'a [u8],
    status: OcspStatus,
    this_update: u64,
    next_update: Option<u64>,
    extensions: Option<&'a [u8]>,
}

fn parse_single(body: &[u8]) -> Result<Single<'_>, String> {
    let (cert_id, _, rest) = next(body, "certID")?;
    let cert_id = sequence(&cert_id, "certID")?;
    let (alg, _, r) = next(cert_id, "certID hashAlgorithm")?;
    let hash_alg = algorithm(&alg, "certID hashAlgorithm")?;
    let (name_hash, _, r) = next(r, "issuerNameHash")?;
    let (key_hash, _, r) = next(r, "issuerKeyHash")?;
    let (serial, _) = only(r, "serialNumber")?;
    expect_tag(&serial, Tag::Integer, "serialNumber")?;

    let (status, _, rest) = next(rest, "certStatus")?;
    let status = match context(&status) {
        Some(0) => {
            if !status.data.is_empty() {
                return Err("certStatus good carries data".into());
            }
            // Filled below, once the times are read.
            OcspStatus::Good {
                this_update: 0,
                next_update: 0,
            }
        }
        Some(1) => {
            let (at, _, r) = next(status.data, "revocationTime")?;
            let at = generalized_time(&at, "revocationTime")?;
            let reason = if r.is_empty() {
                None
            } else {
                let (reason, _) = only(r, "revocationReason")?;
                if context(&reason) != Some(0) {
                    return Err("revocationReason: not [0]".into());
                }
                let (reason, _) = only(reason.data, "revocationReason")?;
                Some(small_uint(&reason, Tag::Enumerated, "revocationReason")?)
            };
            OcspStatus::Revoked { at, reason }
        }
        Some(2) => OcspStatus::Unknown,
        _ => return Err("certStatus: neither good [0], revoked [1] nor unknown [2]".into()),
    };
    let (this, _, mut rest) = next(rest, "thisUpdate")?;
    let this_update = generalized_time(&this, "thisUpdate")?;
    let mut next_update = None;
    let mut extensions = None;
    if !rest.is_empty() {
        let (n, _, after) = next(rest, "nextUpdate")?;
        if context(&n) == Some(0) {
            let (t, _) = only(n.data, "nextUpdate")?;
            next_update = Some(generalized_time(&t, "nextUpdate")?);
            rest = after;
        }
    }
    if !rest.is_empty() {
        let (ext, _) = only(rest, "singleExtensions")?;
        if context(&ext) != Some(1) {
            return Err("singleExtensions: not [1]".into());
        }
        let (ext, _) = only(ext.data, "singleExtensions")?;
        extensions = Some(sequence(&ext, "singleExtensions")?);
    }
    Ok(Single {
        hash_alg,
        issuer_name_hash: octets(&name_hash, "issuerNameHash")?,
        issuer_key_hash: octets(&key_hash, "issuerKeyHash")?,
        serial: serial.data,
        status,
        this_update,
        next_update,
        extensions,
    })
}

/// Each `Extension { OID, critical DEFAULT FALSE, OCTET STRING }` of a list.
fn extensions<'a>(
    list: &'a [u8],
    what: &str,
) -> Result<Vec<(Oid<'static>, bool, &'a [u8])>, String> {
    let mut out = Vec::new();
    let mut rest = list;
    while !rest.is_empty() {
        let (ext, _, after) = next(rest, what)?;
        rest = after;
        let body = sequence(&ext, what)?;
        let (o, _, r) = next(body, what)?;
        let oid = oid_of(&o, what)?.to_owned();
        let (second, _, r) = next(r, what)?;
        let (critical, value) =
            if second.tag() == Tag::Boolean && second.class() == Class::Universal {
                let (v, _) = only(r, what)?;
                (second.data == [0xff], octets(&v, what)?)
            } else {
                if !r.is_empty() {
                    return Err(format!("{what}: {} bytes after extnValue", r.len()));
                }
                (false, octets(&second, what)?)
            };
        out.push((oid, critical, value));
    }
    Ok(out)
}

fn hash_with(alg: &Oid<'_>, data: &[u8]) -> Result<Vec<u8>, String> {
    if *alg == OID_HASH_SHA1 {
        Ok(sha1::Sha1::digest(data).to_vec())
    } else if *alg == OID_NIST_HASH_SHA256 {
        Ok(sha2::Sha256::digest(data).to_vec())
    } else if *alg == OID_NIST_HASH_SHA384 {
        Ok(sha2::Sha384::digest(data).to_vec())
    } else {
        Err(format!(
            "certID hashAlgorithm {alg} is not SHA-1, SHA-256 or SHA-384"
        ))
    }
}

/// The signer's key type, from its SubjectPublicKeyInfo.
enum KeyType {
    Rsa,
    P256,
    P384,
}

fn key_type(cert: &X509Certificate<'_>) -> Result<KeyType, String> {
    let spki = cert.public_key();
    if spki.algorithm.algorithm == OID_PKCS1_RSAENCRYPTION {
        return Ok(KeyType::Rsa);
    }
    if spki.algorithm.algorithm == OID_KEY_TYPE_EC_PUBLIC_KEY {
        let curve = spki
            .algorithm
            .parameters
            .as_ref()
            .ok_or("EC key without a curve")?;
        let curve = oid_of(curve, "EC curve")?;
        if curve == OID_EC_P256 {
            return Ok(KeyType::P256);
        }
        if curve == OID_NIST_EC_P384 {
            return Ok(KeyType::P384);
        }
        return Err(format!("EC curve {curve} is neither P-256 nor P-384"));
    }
    Err(format!(
        "key type {} is neither RSA nor EC",
        spki.algorithm.algorithm
    ))
}

/// `signature` over `message` under `signer`'s key, by `alg` alone: the
/// algorithm must be the one that key type signs with here (RSA PKCS#1 v1.5
/// SHA-256 or SHA-384; ECDSA SHA-256 on P-256; ECDSA SHA-384 on P-384).
fn verify_signature(
    signer: &X509Certificate<'_>,
    alg: &Oid<'_>,
    message: &[u8],
    signature: &[u8],
    what: &str,
) -> Result<(), String> {
    let spki = signer.public_key();
    let bad = |why: String| format!("{what}: {why}");
    match key_type(signer).map_err(bad)? {
        KeyType::Rsa => {
            let key = RsaPublicKey::from_public_key_der(spki.raw)
                .map_err(|e| bad(format!("RSA key: {e}")))?;
            let sig = RsaSignature::try_from(signature)
                .map_err(|e| bad(format!("RSA signature: {e}")))?;
            let ok = if *alg == OID_PKCS1_SHA256WITHRSA {
                RsaVerifyingKey::<rsa::sha2::Sha256>::new(key)
                    .verify(message, &sig)
                    .is_ok()
            } else if *alg == OID_PKCS1_SHA384WITHRSA {
                RsaVerifyingKey::<rsa::sha2::Sha384>::new(key)
                    .verify(message, &sig)
                    .is_ok()
            } else {
                return Err(bad(format!(
                    "signature algorithm {alg} is not sha256WithRSAEncryption or sha384WithRSAEncryption for an RSA key"
                )));
            };
            ok.then_some(())
                .ok_or_else(|| bad("signature does not verify".into()))
        }
        KeyType::P256 => {
            if *alg != OID_SIG_ECDSA_WITH_SHA256 {
                return Err(bad(format!(
                    "signature algorithm {alg} is not ecdsa-with-SHA256 for a P-256 key"
                )));
            }
            let key = p256::ecdsa::VerifyingKey::from_sec1_bytes(&spki.subject_public_key.data)
                .map_err(|e| bad(format!("P-256 key: {e}")))?;
            let sig = p256::ecdsa::Signature::from_der(signature)
                .map_err(|e| bad(format!("ECDSA signature: {e}")))?;
            p256::ecdsa::signature::Verifier::verify(&key, message, &sig)
                .map_err(|_| bad("signature does not verify".into()))
        }
        KeyType::P384 => {
            if *alg != OID_SIG_ECDSA_WITH_SHA384 {
                return Err(bad(format!(
                    "signature algorithm {alg} is not ecdsa-with-SHA384 for a P-384 key"
                )));
            }
            let key = p384::ecdsa::VerifyingKey::from_sec1_bytes(&spki.subject_public_key.data)
                .map_err(|e| bad(format!("P-384 key: {e}")))?;
            let sig = p384::ecdsa::Signature::from_der(signature)
                .map_err(|e| bad(format!("ECDSA signature: {e}")))?;
            p384::ecdsa::signature::Verifier::verify(&key, message, &sig)
                .map_err(|_| bad("signature does not verify".into()))
        }
    }
}

fn key_sha1(cert: &X509Certificate<'_>) -> Vec<u8> {
    sha1::Sha1::digest(&cert.public_key().subject_public_key.data).to_vec()
}

/// One OCSP response about `subject`, issued by `issuer`, at `now`: its
/// verified status, or why it does not count. `issuer`'s key is trusted by
/// the caller (a pinned intermediate, or CA 001 when pinned).
pub fn verify_response(
    der: &[u8],
    subject: &X509Certificate<'_>,
    issuer: &X509Certificate<'_>,
    now_unix: u64,
    policy: &NrasOcspPolicy,
) -> Result<OcspStatus, String> {
    let p = parse(der)?;

    // ── The signer: the issuer, or a responder it certified for this ────
    let delegated: Option<X509Certificate<'_>>;
    let signer: &X509Certificate<'_> = if p.responder_by_key == Some(key_sha1(issuer).as_slice())
        || p.responder_by_name == Some(issuer.subject().as_raw())
    {
        issuer
    } else {
        let mut found = None;
        for der in &p.certs {
            let (_, c) = X509Certificate::from_der(der)
                .map_err(|e| format!("a certificate in certs: {e}"))?;
            if p.responder_by_key == Some(key_sha1(&c).as_slice())
                || p.responder_by_name == Some(c.subject().as_raw())
            {
                found = Some(c);
                break;
            }
        }
        let c = found
            .ok_or("responderID names neither the issuer nor a certificate the response carries")?;
        // RFC 6960 §4.2.2.2: issued directly by the CA, marked id-kp-OCSPSigning.
        if c.issuer().as_raw() != issuer.subject().as_raw() {
            return Err(format!(
                "responder certificate issued by {}, not by {}",
                c.issuer(),
                issuer.subject()
            ));
        }
        let eku = c
            .extended_key_usage()
            .map_err(|e| format!("responder certificate extendedKeyUsage: {e}"))?
            .ok_or("responder certificate has no extendedKeyUsage: not an OCSP signer")?;
        if !eku.value.ocsp_signing {
            return Err("responder certificate is not marked id-kp-OCSPSigning".into());
        }
        let (nb, na) = (
            c.validity().not_before.timestamp(),
            c.validity().not_after.timestamp(),
        );
        let now = now_unix as i64;
        if now < nb || now > na {
            return Err(format!(
                "responder certificate outside its validity window [{nb}, {na}] at {now}"
            ));
        }
        // RFC 5280, section 4.1.1.2: the algorithm written after the signed
        // part MUST be the one inside it. `certs` is outside what the
        // response's signature covers, and no signature covers the outer one.
        if !crate::pcs::signature_algorithm_repeated(c.as_raw()) {
            return Err(
                "responder certificate: the signed and the outer signature algorithms differ"
                    .into(),
            );
        }
        verify_signature(
            issuer,
            &c.signature_algorithm.algorithm,
            c.tbs_certificate.as_ref(),
            c.signature_value.data.as_ref(),
            "responder certificate",
        )?;
        delegated = Some(c);
        delegated.as_ref().expect("just set")
    };

    // ── The signature, before anything signed is believed ───────────────
    verify_signature(signer, &p.signature_alg, p.tbs, p.signature, "response")?;

    // ── Extensions: the nonce when one was sent; nothing critical unread ─
    let mut nonce_seen = None;
    for (oid, critical, value) in
        extensions(p.response_extensions.unwrap_or(&[]), "responseExtensions")?
    {
        if oid == OID_OCSP_NONCE {
            // RFC 8954: the extnValue holds an OCTET STRING; older responders
            // put the bytes raw. Either reading is compared.
            let inner = Any::from_der(value)
                .ok()
                .filter(|(rest, a)| {
                    rest.is_empty() && a.tag() == Tag::OctetString && a.class() == Class::Universal
                })
                .map(|(_, a)| a.data);
            nonce_seen = Some((value, inner));
        } else if critical {
            return Err(format!("critical response extension {oid} not understood"));
        }
    }
    if let Some(want) = &policy.nonce {
        match nonce_seen {
            Some((raw, inner)) if raw == want.as_slice() || inner == Some(want.as_slice()) => {}
            Some(_) => return Err("the response's nonce is not the one sent".into()),
            None => return Err("the response carries no nonce; one was sent".into()),
        }
    }

    // ── The one SingleResponse: about this certificate, under this issuer ─
    let s = parse_single(p.single)?;
    for (oid, critical, _) in extensions(s.extensions.unwrap_or(&[]), "singleExtensions")? {
        if critical {
            return Err(format!("critical single extension {oid} not understood"));
        }
    }
    let name_hash = hash_with(&s.hash_alg, issuer.subject().as_raw())?;
    let key_hash = hash_with(&s.hash_alg, &issuer.public_key().subject_public_key.data)?;
    if s.issuer_name_hash != name_hash.as_slice() || s.issuer_key_hash != key_hash.as_slice() {
        return Err("certID names another issuer".into());
    }
    if magnitude(s.serial) != magnitude(subject.raw_serial()) {
        return Err(format!(
            "certID is about serial {}, not {}",
            hex(magnitude(s.serial)),
            hex(magnitude(subject.raw_serial()))
        ));
    }

    // ── Times ───────────────────────────────────────────────────────────
    if s.this_update > now_unix.saturating_add(CLOCK_SKEW_SECS) {
        return Err(format!(
            "thisUpdate {} is after now {now_unix}",
            s.this_update
        ));
    }
    let next_update = s.next_update.ok_or(
        "no nextUpdate: a response without one cannot be judged current (RFC 5019 §2.2.4)",
    )?;
    if now_unix > next_update.saturating_add(CLOCK_SKEW_SECS) {
        return Err(format!(
            "nextUpdate {next_update} is before now {now_unix}: the response is stale"
        ));
    }
    let age = now_unix.saturating_sub(s.this_update);
    if age > policy.max_age_secs {
        return Err(format!(
            "issued {age} s before now; this verifier accepts at most {} s",
            policy.max_age_secs
        ));
    }
    Ok(match s.status {
        OcspStatus::Good { .. } => OcspStatus::Good {
            this_update: s.this_update,
            next_update,
        },
        other => other,
    })
}

/// The standing of one certificate from every stapled response: a verified
/// `revoked` wins over a verified `good`, which wins over `unknown`; none
/// verified: the last reason, or that none was given.
fn standing(
    what: &'static str,
    subject: &X509Certificate<'_>,
    issuer: Option<&X509Certificate<'_>>,
    responses: &[Vec<u8>],
    now_unix: u64,
    policy: &NrasOcspPolicy,
) -> OcspStanding {
    let serial_hex = hex(magnitude(subject.raw_serial()));
    let Some(issuer) = issuer else {
        return OcspStanding {
            what,
            serial_hex,
            status: OcspStatus::IssuerNotPublished,
        };
    };
    let mut best: Option<OcspStatus> = None;
    let mut last_err = None;
    for der in responses {
        match verify_response(der, subject, issuer, now_unix, policy) {
            Ok(status) => {
                let rank = |s: &OcspStatus| match s {
                    OcspStatus::Revoked { .. } => 3,
                    OcspStatus::Good { .. } => 2,
                    _ => 1,
                };
                if best.as_ref().is_none_or(|b| rank(&status) > rank(b)) {
                    best = Some(status);
                }
            }
            Err(e) => last_err = Some(e),
        }
    }
    let status = match (best, last_err) {
        (Some(s), _) => s,
        (None, Some(e)) => OcspStatus::Unverified(e),
        (None, None) => OcspStatus::NoResponse,
    };
    OcspStanding {
        what,
        serial_hex,
        status,
    }
}

/// The chain's revocation under `policy`: the leaf under the intermediate,
/// the intermediate under CA 001 when the policy pins it. `Err` refuses the
/// token; `Ok` carries what every verdict says, by certificate.
pub fn judge_chain(
    leaf: &X509Certificate<'_>,
    intermediate: &X509Certificate<'_>,
    responses: &[Vec<u8>],
    now_unix: u64,
    policy: &NrasOcspPolicy,
) -> Result<Vec<OcspStanding>, OcspRefused> {
    let ca = match &policy.intermediate_issuer_der {
        Some(der) => Some(
            X509Certificate::from_der(der)
                .map(|(_, c)| c)
                .map_err(|e| OcspRefused::NotGood {
                    what: "x5c intermediate",
                    serial_hex: hex(magnitude(intermediate.raw_serial())),
                    standing: format!("the pinned CA 001 does not parse: {e}"),
                })?,
        ),
        None => None,
    };
    let standings = vec![
        standing(
            "x5c leaf",
            leaf,
            Some(intermediate),
            responses,
            now_unix,
            policy,
        ),
        standing(
            "x5c intermediate",
            intermediate,
            ca.as_ref(),
            responses,
            now_unix,
            policy,
        ),
    ];
    for s in &standings {
        match &s.status {
            OcspStatus::Revoked { at, reason } => {
                return Err(OcspRefused::Revoked {
                    what: s.what,
                    serial_hex: s.serial_hex.clone(),
                    at: *at,
                    reason: *reason,
                })
            }
            OcspStatus::Good { .. } | OcspStatus::IssuerNotPublished => {}
            _ if policy.mode == NrasOcspMode::Required => {
                return Err(OcspRefused::NotGood {
                    what: s.what,
                    serial_hex: s.serial_hex.clone(),
                    standing: s.to_string(),
                })
            }
            _ => {}
        }
    }
    Ok(standings)
}

// Not when compiled into x0ne-sdk (its build.rs sets `x0ne_sdk`): these tests
// need this crate's dev-dependencies (rcgen).
#[cfg(all(test, not(x0ne_sdk)))]
pub(crate) mod tests {
    use super::*;
    use rcgen::{
        BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    };

    // ── A DER writer of the few shapes an OCSP response takes ───────────

    fn tlv(tag: u8, content: &[u8]) -> Vec<u8> {
        let mut out = vec![tag];
        let n = content.len();
        if n < 0x80 {
            out.push(n as u8);
        } else {
            let bytes = n.to_be_bytes();
            let first = bytes.iter().position(|b| *b != 0).unwrap();
            out.push(0x80 | (bytes.len() - first) as u8);
            out.extend_from_slice(&bytes[first..]);
        }
        out.extend_from_slice(content);
        out
    }
    pub(crate) fn seq(parts: &[&[u8]]) -> Vec<u8> {
        tlv(0x30, &parts.concat())
    }
    /// `[n] EXPLICIT`: constructed, context-specific.
    fn ctx(n: u8, inner: &[u8]) -> Vec<u8> {
        tlv(0xa0 | n, inner)
    }
    /// `[n] IMPLICIT` of a primitive: context-specific, the content raw.
    fn ctx_prim(n: u8, content: &[u8]) -> Vec<u8> {
        tlv(0x80 | n, content)
    }
    pub(crate) fn der_oid(arcs: &[u64]) -> Vec<u8> {
        let mut body = vec![(arcs[0] * 40 + arcs[1]) as u8];
        for &a in &arcs[2..] {
            let mut tmp = Vec::new();
            let mut v = a;
            tmp.push((v & 0x7f) as u8);
            v >>= 7;
            while v > 0 {
                tmp.push(0x80 | (v & 0x7f) as u8);
                v >>= 7;
            }
            tmp.reverse();
            body.extend(tmp);
        }
        tlv(0x06, &body)
    }
    fn der_int(magnitude: &[u8]) -> Vec<u8> {
        let mut m: Vec<u8> = magnitude.iter().copied().skip_while(|b| *b == 0).collect();
        if m.is_empty() {
            m.push(0);
        }
        if m[0] & 0x80 != 0 {
            m.insert(0, 0);
        }
        tlv(0x02, &m)
    }
    fn octets(b: &[u8]) -> Vec<u8> {
        tlv(0x04, b)
    }
    pub(crate) fn bitstring(b: &[u8]) -> Vec<u8> {
        let mut c = vec![0u8];
        c.extend_from_slice(b);
        tlv(0x03, &c)
    }
    fn enumerated(n: u8) -> Vec<u8> {
        tlv(0x0a, &[n])
    }
    pub(crate) fn null() -> Vec<u8> {
        tlv(0x05, &[])
    }
    fn boolean(b: bool) -> Vec<u8> {
        tlv(0x01, &[if b { 0xff } else { 0 }])
    }
    /// `GeneralizedTime` of a unix second, `YYYYMMDDHHMMSSZ`.
    fn gen_time(unix: u64) -> Vec<u8> {
        let days = (unix / 86_400) as i64;
        let secs = unix % 86_400;
        // Howard Hinnant's civil_from_days.
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        let s = format!(
            "{y:04}{m:02}{d:02}{:02}{:02}{:02}Z",
            secs / 3600,
            (secs % 3600) / 60,
            secs % 60
        );
        tlv(0x18, s.as_bytes())
    }

    // ── Keys, certificates, signers ─────────────────────────────────────

    /// A signer of OCSP responses: a PKCS#8 key and the algorithm it signs
    /// with (rcgen signs: DER ECDSA, or RSA PKCS#1 v1.5 SHA-256).
    #[derive(Clone)]
    pub(crate) struct Signer {
        pem: String,
        kind: SignerKind,
    }

    #[derive(Clone, Copy)]
    enum SignerKind {
        P384,
        P256,
        Rsa,
    }

    impl Signer {
        pub(crate) fn of(kp: &KeyPair) -> Self {
            let kind = if std::ptr::eq(kp.algorithm(), &rcgen::PKCS_ECDSA_P384_SHA384) {
                SignerKind::P384
            } else if std::ptr::eq(kp.algorithm(), &rcgen::PKCS_ECDSA_P256_SHA256) {
                SignerKind::P256
            } else {
                SignerKind::Rsa
            };
            Self {
                pem: kp.serialize_pem(),
                kind,
            }
        }
        fn algorithm(&self) -> &'static rcgen::SignatureAlgorithm {
            match self.kind {
                SignerKind::P384 => &rcgen::PKCS_ECDSA_P384_SHA384,
                SignerKind::P256 => &rcgen::PKCS_ECDSA_P256_SHA256,
                SignerKind::Rsa => &rcgen::PKCS_RSA_SHA256,
            }
        }
        fn algorithm_oid(&self) -> Vec<u8> {
            match self.kind {
                SignerKind::P384 => der_oid(&[1, 2, 840, 10045, 4, 3, 3]),
                SignerKind::P256 => der_oid(&[1, 2, 840, 10045, 4, 3, 2]),
                SignerKind::Rsa => der_oid(&[1, 2, 840, 113549, 1, 1, 11]),
            }
        }
        pub(crate) fn algorithm_identifier(&self) -> Vec<u8> {
            match self.kind {
                SignerKind::Rsa => seq(&[&self.algorithm_oid(), &null()]),
                _ => seq(&[&self.algorithm_oid()]),
            }
        }
        pub(crate) fn sign(&self, msg: &[u8]) -> Vec<u8> {
            let kp = KeyPair::from_pkcs8_pem_and_sign_algo(&self.pem, self.algorithm())
                .expect("the key pair it was made from");
            rcgen::SigningKey::sign(&kp, msg).expect("signs")
        }
    }

    /// A CA (an intermediate or a root) that signs certificates and
    /// responses: its DER, its issuer handle, its signer.
    pub(crate) struct Ca {
        pub(crate) der: Vec<u8>,
        pub(crate) issuer: Issuer<'static, KeyPair>,
        pub(crate) signer: Signer,
    }

    fn ca_params(cn: &str) -> CertificateParams {
        let mut params = CertificateParams::new(vec![]).unwrap();
        params
            .distinguished_name
            .push(rcgen::DnType::CommonName, cn);
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        params.not_before = rcgen::date_time_ymd(1970, 1, 1);
        params.not_after = rcgen::date_time_ymd(4096, 1, 1);
        params
    }

    pub(crate) fn p384_kp() -> KeyPair {
        KeyPair::generate_for(&rcgen::PKCS_ECDSA_P384_SHA384).unwrap()
    }

    /// A self-signed CA.
    pub(crate) fn root(cn: &str, kp: KeyPair) -> Ca {
        let params = ca_params(cn);
        let der = params.clone().self_signed(&kp).unwrap().der().to_vec();
        let signer = Signer::of(&kp);
        Ca {
            der,
            issuer: Issuer::new(params, kp),
            signer,
        }
    }

    /// A CA issued by `parent`.
    pub(crate) fn sub_ca(cn: &str, kp: KeyPair, parent: &Ca) -> Ca {
        let params = ca_params(cn);
        let der = params
            .clone()
            .signed_by(&kp, &parent.issuer)
            .unwrap()
            .der()
            .to_vec();
        let signer = Signer::of(&kp);
        Ca {
            der,
            issuer: Issuer::new(params, kp),
            signer,
        }
    }

    /// An end-entity certificate under `ca`, with `serial`, valid 1970 to
    /// 4096, `eku` as asked.
    pub(crate) fn end_entity(
        ca: &Ca,
        kp: &KeyPair,
        serial: &[u8],
        eku: &[ExtendedKeyUsagePurpose],
        window: (i32, i32),
    ) -> Vec<u8> {
        let mut params = CertificateParams::new(vec![]).unwrap();
        params.serial_number = Some(rcgen::SerialNumber::from_slice(serial));
        params.not_before = rcgen::date_time_ymd(window.0, 1, 1);
        params.not_after = rcgen::date_time_ymd(window.1, 1, 1);
        params.extended_key_usages = eku.to_vec();
        params.signed_by(kp, &ca.issuer).unwrap().der().to_vec()
    }

    fn cert(der: &[u8]) -> X509Certificate<'_> {
        X509Certificate::from_der(der).unwrap().1
    }

    /// `cert_der` with the `signatureAlgorithm` after its signed part
    /// replaced by `algorithm`, the DER of an `AlgorithmIdentifier`: bytes
    /// no signature covers.
    pub(crate) fn with_outer_algorithm(cert_der: &[u8], algorithm: &[u8]) -> Vec<u8> {
        let c = cert(cert_der);
        seq(&[
            c.tbs_certificate.as_ref(),
            algorithm,
            &bitstring(&c.signature_value.data),
        ])
    }

    /// `cert_der` signed again by `issuer` after `edit` changed its signed
    /// part; the outer `signatureAlgorithm` is the one `issuer` signs with.
    pub(crate) fn signed_again(
        cert_der: &[u8],
        issuer: &Signer,
        edit: impl FnOnce(&mut Vec<u8>),
    ) -> Vec<u8> {
        let mut tbs = cert(cert_der).tbs_certificate.as_ref().to_vec();
        edit(&mut tbs);
        seq(&[
            &tbs,
            &issuer.algorithm_identifier(),
            &bitstring(&issuer.sign(&tbs)),
        ])
    }

    // ── The synthetic responder ─────────────────────────────────────────

    #[derive(Clone, Copy)]
    pub(crate) enum Status {
        Good,
        Revoked(u64, Option<u8>),
        Unknown,
    }

    /// One deliberate malformation of a response, for the refusal tests:
    /// each names the place the verifier must refuse at.
    #[derive(Clone, Debug)]
    pub(crate) enum Twist {
        /// Bytes after the `OCSPResponse`.
        Trailing,
        /// Another tag than SEQUENCE around the `OCSPResponse`.
        TopTag(u8),
        /// A `responseStatus` that is not a small integer.
        WideStatus,
        /// `responseBytes` under this context tag instead of [0].
        ResponseBytesTag(u8),
        /// Another `responseType` than id-pkix-ocsp-basic.
        ResponseType(&'static [u64]),
        /// The signature BIT STRING with unused bits.
        UnusedBits,
        /// `certs` under this context tag instead of [0].
        CertsTag(u8),
        /// An explicit `version` in `ResponseData`.
        Version(u8),
        /// `responderID` under this context tag.
        ResponderTag(u8),
        /// Two `SingleResponse`s.
        TwoSingles,
        /// `responseExtensions` under this context tag instead of [1].
        ResponseExtTag(u8),
        /// `certStatus` good with content.
        GoodWithData,
        /// `revocationReason` under this context tag instead of [0].
        ReasonTag(u8),
        /// `certStatus` under this context tag.
        CertStatusTag(u8),
        /// A `singleExtensions` entry: `(oid arcs, critical)`.
        SingleExt(Vec<u64>, bool),
        /// `singleExtensions` under this context tag instead of [1].
        SingleExtTag(u8),
        /// A response extension with bytes after its `extnValue`.
        ExtTrailing,
        /// The `signatureAlgorithm` written with this OID, the signature as made.
        SigAlg(&'static [u64]),
        /// The `CertID` hashed from this issuer certificate (DER) instead.
        CertIdIssuer(Vec<u8>),
    }

    /// What one response says and how it is made; `Default` is a good,
    /// fresh response signed by the issuer itself, `CertID` in SHA-1.
    #[derive(Clone)]
    pub(crate) struct Response {
        pub(crate) status: Status,
        pub(crate) this_update: u64,
        pub(crate) next_update: Option<u64>,
        pub(crate) produced_at: u64,
        /// `None`: the issuer signs. `Some`: this responder certificate (DER)
        /// and its signer sign, the certificate carried in `certs`.
        pub(crate) delegated: Option<(Vec<u8>, Signer)>,
        /// A signer other than the one `responderID` names: a forged signature.
        pub(crate) forged_by: Option<Signer>,
        pub(crate) by_name: bool,
        pub(crate) nonce: Option<Vec<u8>>,
        /// An extension nobody reads, critical: `(oid arcs, critical)`.
        pub(crate) extra_extension: Option<(Vec<u64>, bool)>,
        pub(crate) response_status: u8,
        pub(crate) hash: &'static [u64],
        /// Another serial in the `CertID` than the subject's.
        pub(crate) other_serial: Option<Vec<u8>>,
        /// One deliberate malformation, for the refusal tests.
        pub(crate) twist: Option<Twist>,
    }

    pub(crate) const SHA1: &[u64] = &[1, 3, 14, 3, 2, 26];
    pub(crate) const SHA256: &[u64] = &[2, 16, 840, 1, 101, 3, 4, 2, 1];
    pub(crate) const SHA384: &[u64] = &[2, 16, 840, 1, 101, 3, 4, 2, 2];

    impl Response {
        pub(crate) fn good_at(now: u64) -> Self {
            Self {
                status: Status::Good,
                this_update: now.saturating_sub(600),
                next_update: Some(now + 6 * 3600),
                produced_at: now.saturating_sub(600),
                delegated: None,
                forged_by: None,
                by_name: false,
                nonce: None,
                extra_extension: None,
                response_status: 0,
                hash: SHA1,
                other_serial: None,
                twist: None,
            }
        }
    }

    fn hash_bytes(arcs: &[u64], data: &[u8]) -> Vec<u8> {
        if arcs == SHA1 {
            sha1::Sha1::digest(data).to_vec()
        } else if arcs == SHA256 {
            sha2::Sha256::digest(data).to_vec()
        } else {
            sha2::Sha384::digest(data).to_vec()
        }
    }

    /// The DER `OCSPResponse` of `r` about `subject_der` under `issuer`.
    pub(crate) fn respond(r: &Response, subject_der: &[u8], issuer: &Ca) -> Vec<u8> {
        respond_with(r, subject_der, &issuer.der, &issuer.signer)
    }

    /// The same, the issuer given as its DER and the signer of its key.
    pub(crate) fn respond_with(
        r: &Response,
        subject_der: &[u8],
        issuer_der: &[u8],
        issuer_signer: &Signer,
    ) -> Vec<u8> {
        let subject = cert(subject_der);
        let named_issuer_der = match &r.twist {
            Some(Twist::CertIdIssuer(der)) => der.as_slice(),
            _ => issuer_der,
        };
        let issuer_cert = cert(named_issuer_der);
        let (signer_cert_der, signer) = match &r.delegated {
            Some((der, s)) => (der.clone(), s.clone()),
            None => (issuer_der.to_vec(), issuer_signer.clone()),
        };
        let signer_cert = cert(&signer_cert_der);
        let responder_id = match (&r.twist, r.by_name) {
            (Some(Twist::ResponderTag(t)), _) => ctx(*t, &octets(&key_sha1(&signer_cert))),
            (_, true) => ctx(1, signer_cert.subject().as_raw()),
            (_, false) => ctx(2, &octets(&key_sha1(&signer_cert))),
        };
        let serial = r
            .other_serial
            .clone()
            .unwrap_or_else(|| subject.raw_serial().to_vec());
        let cert_id = seq(&[
            &seq(&[&der_oid(r.hash), &null()]),
            &octets(&hash_bytes(r.hash, issuer_cert.subject().as_raw())),
            &octets(&hash_bytes(
                r.hash,
                &issuer_cert.public_key().subject_public_key.data,
            )),
            &der_int(&serial),
        ]);
        let status = match (&r.twist, r.status) {
            (Some(Twist::CertStatusTag(t)), _) => ctx_prim(*t, &[]),
            (Some(Twist::GoodWithData), Status::Good) => ctx_prim(0, &[1]),
            (_, Status::Good) => ctx_prim(0, &[]),
            (twist, Status::Revoked(at, reason)) => {
                let mut body = gen_time(at);
                if let Some(reason) = reason {
                    let tag = match twist {
                        Some(Twist::ReasonTag(t)) => *t,
                        _ => 0,
                    };
                    body.extend(ctx(tag, &enumerated(reason)));
                }
                tlv(0xa1, &body)
            }
            (_, Status::Unknown) => ctx_prim(2, &[]),
        };
        let mut single_parts: Vec<Vec<u8>> = vec![cert_id, status, gen_time(r.this_update)];
        if let Some(n) = r.next_update {
            single_parts.push(ctx(0, &gen_time(n)));
        }
        match &r.twist {
            Some(Twist::SingleExt(arcs, critical)) => single_parts.push(ctx(
                1,
                &seq(&[&seq(&[
                    &der_oid(arcs),
                    &boolean(*critical),
                    &octets(&[1, 2, 3]),
                ])]),
            )),
            Some(Twist::SingleExtTag(t)) => single_parts.push(ctx(
                *t,
                &seq(&[&seq(&[&der_oid(&[1, 2, 3, 4]), &octets(&[1])])]),
            )),
            _ => {}
        }
        let refs: Vec<&[u8]> = single_parts.iter().map(|p| p.as_slice()).collect();
        let single = seq(&refs);
        let singles = match r.twist {
            Some(Twist::TwoSingles) => seq(&[&single, &single]),
            _ => seq(&[&single]),
        };
        let mut exts = Vec::new();
        if let Some(nonce) = &r.nonce {
            exts.push(seq(&[
                &der_oid(&[1, 3, 6, 1, 5, 5, 7, 48, 1, 2]),
                &octets(&octets(nonce)),
            ]));
        }
        if let Some((arcs, critical)) = &r.extra_extension {
            exts.push(seq(&[
                &der_oid(arcs),
                &boolean(*critical),
                &octets(&[1, 2, 3]),
            ]));
        }
        if matches!(r.twist, Some(Twist::ExtTrailing)) {
            exts.push(seq(&[&der_oid(&[1, 2, 3, 4]), &octets(&[1]), &null()]));
        }
        let mut tbs_parts: Vec<Vec<u8>> = Vec::new();
        if let Some(Twist::Version(v)) = r.twist {
            tbs_parts.push(ctx(0, &der_int(&[v])));
        }
        tbs_parts.extend([responder_id, gen_time(r.produced_at), singles]);
        if !exts.is_empty() {
            let list: Vec<&[u8]> = exts.iter().map(|e| e.as_slice()).collect();
            let tag = match r.twist {
                Some(Twist::ResponseExtTag(t)) => t,
                _ => 1,
            };
            tbs_parts.push(ctx(tag, &seq(&list)));
        }
        let refs: Vec<&[u8]> = tbs_parts.iter().map(|p| p.as_slice()).collect();
        let tbs = seq(&refs);
        let signing = r.forged_by.clone().unwrap_or(signer);
        let signature = signing.sign(&tbs);
        let algorithm = match r.twist {
            Some(Twist::SigAlg(arcs)) => seq(&[&der_oid(arcs)]),
            _ => signing.algorithm_identifier(),
        };
        let signature = match r.twist {
            Some(Twist::UnusedBits) => tlv(0x03, &[&[1u8][..], signature.as_slice()].concat()),
            _ => bitstring(&signature),
        };
        let mut basic_parts: Vec<Vec<u8>> = vec![tbs, algorithm, signature];
        if r.delegated.is_some() {
            let tag = match r.twist {
                Some(Twist::CertsTag(t)) => t,
                _ => 0,
            };
            basic_parts.push(ctx(tag, &seq(&[&signer_cert_der])));
        }
        let refs: Vec<&[u8]> = basic_parts.iter().map(|p| p.as_slice()).collect();
        let basic = seq(&refs);
        let response_type = match r.twist {
            Some(Twist::ResponseType(arcs)) => der_oid(arcs),
            _ => der_oid(&[1, 3, 6, 1, 5, 5, 7, 48, 1, 1]),
        };
        let response_bytes = seq(&[&response_type, &octets(&basic)]);
        if r.response_status != 0 {
            return seq(&[&enumerated(r.response_status)]);
        }
        let status = match r.twist {
            Some(Twist::WideStatus) => tlv(0x0a, &[0x80]),
            _ => enumerated(0),
        };
        let rb_tag = match r.twist {
            Some(Twist::ResponseBytesTag(t)) => t,
            _ => 0,
        };
        let body = [status, ctx(rb_tag, &response_bytes)].concat();
        match r.twist {
            Some(Twist::TopTag(t)) => tlv(t, &body),
            Some(Twist::Trailing) => [seq(&[&body]), vec![0]].concat(),
            _ => seq(&[&body]),
        }
    }

    // ── The bench: CA 001, an intermediate, a leaf ──────────────────────

    pub(crate) struct Bench {
        pub(crate) root: Ca,
        pub(crate) inter: Ca,
        pub(crate) leaf_der: Vec<u8>,
    }

    pub(crate) const LEAF_SERIAL: &[u8] = &[0x38, 0xc3, 0x01, 0x87, 0x93, 0xb8, 0x96, 0x3f];
    pub(crate) const NOW: u64 = 1_800_000_000;

    pub(crate) fn bench() -> Bench {
        let root = root("CA 001", p384_kp());
        let inter = sub_ca("GPU Intermediate 005", p384_kp(), &root);
        let leaf_der = end_entity(&inter, &p384_kp(), LEAF_SERIAL, &[], (1970, 4096));
        Bench {
            root,
            inter,
            leaf_der,
        }
    }

    pub(crate) fn rsa_kp() -> KeyPair {
        let mut rng = crate::nras::tests::rsa_test_rng();
        let sk = rsa::RsaPrivateKey::new(&mut rng, 2048).expect("rsa key");
        KeyPair::from_pkcs8_pem_and_sign_algo(
            &crate::nras::tests::rsa_to_pkcs8_pem(&sk),
            &rcgen::PKCS_RSA_SHA256,
        )
        .expect("rsa keypair")
    }

    fn judge(
        b: &Bench,
        responses: &[Vec<u8>],
        policy: &NrasOcspPolicy,
    ) -> Result<Vec<OcspStanding>, OcspRefused> {
        judge_chain(
            &cert(&b.leaf_der),
            &cert(&b.inter.der),
            responses,
            NOW,
            policy,
        )
    }

    fn with_ca(b: &Bench, policy: NrasOcspPolicy) -> NrasOcspPolicy {
        NrasOcspPolicy {
            intermediate_issuer_der: Some(b.root.der.clone()),
            ..policy
        }
    }

    fn leaf_standing(r: &Result<Vec<OcspStanding>, OcspRefused>) -> &OcspStatus {
        &r.as_ref().expect("admitted")[0].status
    }

    // ── Tests ───────────────────────────────────────────────────────────

    #[test]
    fn the_default_policy_is_required_and_production_names_transparent() {
        assert_eq!(NrasOcspPolicy::default().mode, NrasOcspMode::Required);
        assert_eq!(NrasOcspMode::default(), NrasOcspMode::Required);
        assert_eq!(NrasOcspPolicy::production().mode, NrasOcspMode::Transparent);
        assert_eq!(
            NrasOcspPolicy::MAX_AGE_SECS,
            crate::tcb::TcbPolicy::MAX_COLLATERAL_AGE_SECS
        );
        assert!(NrasOcspPolicy::required().at_least(&NrasOcspPolicy::production()));
        assert!(!NrasOcspPolicy::production().at_least(&NrasOcspPolicy::required()));
        assert_eq!(NrasOcspMode::parse("required"), Ok(NrasOcspMode::Required));
        assert!(NrasOcspMode::parse("soft").is_err());
        assert!(NrasOcspPolicy::production()
            .describe()
            .contains("nras-ocsp transparent"));
    }

    #[test]
    fn a_good_response_signed_by_the_issuer_is_accepted_under_both_modes() {
        let b = bench();
        let good = respond(&Response::good_at(NOW), &b.leaf_der, &b.inter);
        for policy in [NrasOcspPolicy::required(), NrasOcspPolicy::production()] {
            let r = judge(&b, std::slice::from_ref(&good), &policy);
            assert!(
                matches!(leaf_standing(&r), OcspStatus::Good { .. }),
                "{policy:?}: {r:?}"
            );
            assert_eq!(r.unwrap()[1].status, OcspStatus::IssuerNotPublished);
        }
    }

    #[test]
    fn a_good_response_signed_by_an_rsa_issuer_is_accepted() {
        let root = root("CA 001", p384_kp());
        let inter = sub_ca("GPU Intermediate 004", rsa_kp(), &root);
        let leaf_der = end_entity(&inter, &p384_kp(), LEAF_SERIAL, &[], (1970, 4096));
        let b = Bench {
            root,
            inter,
            leaf_der,
        };
        let good = respond(&Response::good_at(NOW), &b.leaf_der, &b.inter);
        let r = judge(&b, &[good], &NrasOcspPolicy::required());
        assert!(
            matches!(leaf_standing(&r), OcspStatus::Good { .. }),
            "{r:?}"
        );
    }

    #[test]
    fn a_certid_hashed_with_sha256_or_sha384_is_matched_and_by_name_responders_too() {
        let b = bench();
        for hash in [SHA256, SHA384] {
            for by_name in [false, true] {
                let r = Response {
                    by_name,
                    hash,
                    ..Response::good_at(NOW)
                };
                let der = respond(&r, &b.leaf_der, &b.inter);
                let j = judge(&b, &[der], &NrasOcspPolicy::required());
                assert!(
                    matches!(leaf_standing(&j), OcspStatus::Good { .. }),
                    "{j:?}"
                );
            }
        }
    }

    #[test]
    fn a_delegated_responder_marked_ocsp_signing_is_accepted_and_one_unmarked_is_not() {
        let b = bench();
        let kp = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap();
        let marked = end_entity(
            &b.inter,
            &kp,
            &[7],
            &[ExtendedKeyUsagePurpose::OcspSigning],
            (1970, 4096),
        );
        let r = Response {
            delegated: Some((marked, Signer::of(&kp))),
            ..Response::good_at(NOW)
        };
        let j = judge(
            &b,
            &[respond(&r, &b.leaf_der, &b.inter)],
            &NrasOcspPolicy::required(),
        );
        assert!(
            matches!(leaf_standing(&j), OcspStatus::Good { .. }),
            "{j:?}"
        );

        let unmarked = end_entity(
            &b.inter,
            &kp,
            &[8],
            &[ExtendedKeyUsagePurpose::ServerAuth],
            (1970, 4096),
        );
        let r = Response {
            delegated: Some((unmarked, Signer::of(&kp))),
            ..Response::good_at(NOW)
        };
        let j = judge(
            &b,
            &[respond(&r, &b.leaf_der, &b.inter)],
            &NrasOcspPolicy::required(),
        );
        let e = j.expect_err("an unmarked responder");
        assert!(e.to_string().contains("id-kp-OCSPSigning"), "{e}");

        // Marked, but issued by another CA: not this issuer's responder.
        let other = root("Other CA", p384_kp());
        let elsewhere = end_entity(
            &other,
            &kp,
            &[9],
            &[ExtendedKeyUsagePurpose::OcspSigning],
            (1970, 4096),
        );
        let r = Response {
            delegated: Some((elsewhere, Signer::of(&kp))),
            ..Response::good_at(NOW)
        };
        let j = judge(
            &b,
            &[respond(&r, &b.leaf_der, &b.inter)],
            &NrasOcspPolicy::required(),
        );
        let e = j.expect_err("a responder of another CA");
        assert!(e.to_string().contains("issued by"), "{e}");

        // Marked, issued here, expired.
        let expired = end_entity(
            &b.inter,
            &kp,
            &[10],
            &[ExtendedKeyUsagePurpose::OcspSigning],
            (1970, 1980),
        );
        let r = Response {
            delegated: Some((expired, Signer::of(&kp))),
            ..Response::good_at(NOW)
        };
        let j = judge(
            &b,
            &[respond(&r, &b.leaf_der, &b.inter)],
            &NrasOcspPolicy::required(),
        );
        let e = j.expect_err("an expired responder");
        assert!(e.to_string().contains("validity window"), "{e}");
    }

    /// RFC 5280, section 4.1.1.2: the algorithm written after the signed part
    /// of a certificate MUST be the one inside it. The responder certificate
    /// travels in `certs`, which the response's signature does not cover, and
    /// its outer algorithm is covered by no signature at all.
    #[test]
    fn a_responder_certificate_whose_outer_algorithm_is_not_the_signed_one_is_refused() {
        const ECDSA_SHA384: &[u64] = &[1, 2, 840, 10045, 4, 3, 3];
        let b = bench();
        let kp = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap();
        let marked = end_entity(
            &b.inter,
            &kp,
            &[7],
            &[ExtendedKeyUsagePurpose::OcspSigning],
            (1970, 4096),
        );
        let judged = |responder: Vec<u8>| {
            let r = Response {
                delegated: Some((responder, Signer::of(&kp))),
                ..Response::good_at(NOW)
            };
            judge(
                &b,
                &[respond(&r, &b.leaf_der, &b.inter)],
                &NrasOcspPolicy::required(),
            )
        };
        let j = judged(marked.clone());
        assert!(
            matches!(leaf_standing(&j), OcspStatus::Good { .. }),
            "{j:?}"
        );

        // The same algorithm outside, written with a NULL the signed one has not.
        let with_null = with_outer_algorithm(&marked, &seq(&[&der_oid(ECDSA_SHA384), &null()]));
        assert_ne!(with_null, marked);
        // A signed part that names ecdsa-with-SHA256, signed with SHA-384 by
        // the issuer and saying so outside only.
        let signed = seq(&[&der_oid(ECDSA_SHA384)]);
        let renamed = signed_again(&marked, &b.inter.signer, |tbs| {
            let at = tbs
                .windows(signed.len())
                .position(|w| w == signed)
                .expect("the signed algorithm");
            tbs[at + signed.len() - 1] = 0x02;
        });
        let mut not_refused = Vec::new();
        for (what, responder) in [
            ("an outer algorithm with other parameters", with_null),
            ("a signed algorithm that is not the one verified", renamed),
        ] {
            match judged(responder) {
                Err(e) if e.to_string().contains("algorithms differ") => {}
                other => not_refused.push(format!("{what}: {other:?}")),
            }
        }
        assert!(not_refused.is_empty(), "{not_refused:#?}");
    }

    /// SECURITY: a revoked signing leaf is a leaked or retired key; nothing
    /// admits it, under either mode.
    #[test]
    fn a_revoked_leaf_is_refused_under_both_modes_with_its_time_and_reason() {
        let b = bench();
        let r = Response {
            status: Status::Revoked(NOW - 7200, Some(1)),
            ..Response::good_at(NOW)
        };
        let revoked = respond(&r, &b.leaf_der, &b.inter);
        for policy in [NrasOcspPolicy::required(), NrasOcspPolicy::production()] {
            let e = judge(&b, std::slice::from_ref(&revoked), &policy).expect_err("revoked");
            assert_eq!(
                e,
                OcspRefused::Revoked {
                    what: "x5c leaf",
                    serial_hex: hex(LEAF_SERIAL),
                    at: NOW - 7200,
                    reason: Some(1),
                },
                "{policy:?}"
            );
            assert!(e.to_string().contains("REVOKED"), "{e}");
        }
        // Next to a good one, the revoked answer still wins.
        let good = respond(&Response::good_at(NOW), &b.leaf_der, &b.inter);
        assert!(judge(&b, &[good, revoked], &NrasOcspPolicy::production()).is_err());
    }

    #[test]
    fn unknown_refuses_under_required_and_is_named_under_transparent() {
        let b = bench();
        let r = Response {
            status: Status::Unknown,
            ..Response::good_at(NOW)
        };
        let unknown = respond(&r, &b.leaf_der, &b.inter);
        let e = judge(
            &b,
            std::slice::from_ref(&unknown),
            &NrasOcspPolicy::required(),
        )
        .expect_err("unknown");
        assert!(
            matches!(
                e,
                OcspRefused::NotGood {
                    what: "x5c leaf",
                    ..
                }
            ),
            "{e}"
        );
        assert!(e.to_string().contains("unknown to the responder"), "{e}");
        let j = judge(&b, &[unknown], &NrasOcspPolicy::production());
        assert_eq!(*leaf_standing(&j), OcspStatus::Unknown);
    }

    #[test]
    fn no_response_refuses_under_required_and_is_named_under_transparent() {
        let b = bench();
        let e = judge(&b, &[], &NrasOcspPolicy::required()).expect_err("none");
        assert!(e.to_string().contains("no response stapled"), "{e}");
        let j = judge(&b, &[], &NrasOcspPolicy::production());
        assert_eq!(*leaf_standing(&j), OcspStatus::NoResponse);
        assert!(j.unwrap()[0].to_string().contains("no response stapled"));
    }

    #[test]
    fn a_stale_old_future_or_open_ended_response_does_not_count() {
        let b = bench();
        let cases: [(&str, Response); 4] = [
            (
                "stale",
                Response {
                    this_update: NOW - 7 * 3600,
                    next_update: Some(NOW - 3600),
                    ..Response::good_at(NOW)
                },
            ),
            (
                "issued",
                Response {
                    this_update: NOW - 2 * 86_400,
                    next_update: Some(NOW + 86_400),
                    ..Response::good_at(NOW)
                },
            ),
            (
                "after now",
                Response {
                    this_update: NOW + 3600,
                    next_update: Some(NOW + 7200),
                    ..Response::good_at(NOW)
                },
            ),
            (
                "no nextUpdate",
                Response {
                    next_update: None,
                    ..Response::good_at(NOW)
                },
            ),
        ];
        for (says, r) in cases {
            let der = respond(&r, &b.leaf_der, &b.inter);
            let e =
                judge(&b, std::slice::from_ref(&der), &NrasOcspPolicy::required()).expect_err(says);
            assert!(e.to_string().contains(says), "{says}: {e}");
            let j = judge(&b, &[der], &NrasOcspPolicy::production());
            assert!(
                matches!(leaf_standing(&j), OcspStatus::Unverified(w) if w.contains(says)),
                "{says}: {j:?}"
            );
        }
        // The clock skew: a response whose nextUpdate passed 100 s ago still counts.
        let r = Response {
            next_update: Some(NOW - 100),
            ..Response::good_at(NOW)
        };
        let j = judge(
            &b,
            &[respond(&r, &b.leaf_der, &b.inter)],
            &NrasOcspPolicy::required(),
        );
        assert!(
            matches!(leaf_standing(&j), OcspStatus::Good { .. }),
            "{j:?}"
        );
    }

    /// SECURITY: a response signed by anyone but the issuer or its responder
    /// is not a response. The forged `good` is not believed.
    #[test]
    fn a_response_whose_signature_does_not_verify_is_refused() {
        let b = bench();
        let r = Response {
            forged_by: Some(Signer::of(&p384_kp())),
            ..Response::good_at(NOW)
        };
        let forged = respond(&r, &b.leaf_der, &b.inter);
        let e = judge(
            &b,
            std::slice::from_ref(&forged),
            &NrasOcspPolicy::required(),
        )
        .expect_err("forged");
        assert!(e.to_string().contains("signature does not verify"), "{e}");
        let j = judge(&b, &[forged], &NrasOcspPolicy::production());
        assert!(
            matches!(leaf_standing(&j), OcspStatus::Unverified(_)),
            "{j:?}"
        );
        // A response about a leaf of another CA, under that CA: the certID
        // names another issuer.
        let other = sub_ca("Another intermediate", p384_kp(), &b.root);
        let other_leaf = end_entity(&other, &p384_kp(), LEAF_SERIAL, &[], (1970, 4096));
        let elsewhere = respond(&Response::good_at(NOW), &other_leaf, &other);
        let e = judge(&b, &[elsewhere], &NrasOcspPolicy::required()).expect_err("elsewhere");
        assert!(e.to_string().contains("responderID names neither"), "{e}");
    }

    #[test]
    fn a_response_about_another_certificate_is_not_this_ones() {
        let b = bench();
        let r = Response {
            other_serial: Some(vec![0x11, 0x22]),
            ..Response::good_at(NOW)
        };
        let e = judge(
            &b,
            &[respond(&r, &b.leaf_der, &b.inter)],
            &NrasOcspPolicy::required(),
        )
        .expect_err("another serial");
        assert!(e.to_string().contains("about serial 1122"), "{e}");
    }

    #[test]
    fn a_nonce_sent_must_be_echoed_and_one_not_sent_is_ignored() {
        let b = bench();
        let asked = NrasOcspPolicy {
            nonce: Some(vec![9; 32]),
            ..NrasOcspPolicy::required()
        };
        let echoed = respond(
            &Response {
                nonce: Some(vec![9; 32]),
                ..Response::good_at(NOW)
            },
            &b.leaf_der,
            &b.inter,
        );
        let j = judge(&b, &[echoed], &asked);
        assert!(
            matches!(leaf_standing(&j), OcspStatus::Good { .. }),
            "{j:?}"
        );
        let other = respond(
            &Response {
                nonce: Some(vec![8; 32]),
                ..Response::good_at(NOW)
            },
            &b.leaf_der,
            &b.inter,
        );
        let e = judge(&b, std::slice::from_ref(&other), &asked).expect_err("another nonce");
        assert!(e.to_string().contains("nonce is not the one sent"), "{e}");
        let none = respond(&Response::good_at(NOW), &b.leaf_der, &b.inter);
        let e = judge(&b, std::slice::from_ref(&none), &asked).expect_err("no nonce");
        assert!(e.to_string().contains("carries no nonce"), "{e}");
        // Not asked: a nonce in the response, any nonce, is ignored (RFC 5019 §4).
        let j = judge(&b, &[other], &NrasOcspPolicy::required());
        assert!(
            matches!(leaf_standing(&j), OcspStatus::Good { .. }),
            "{j:?}"
        );
        let j = judge(&b, &[none], &NrasOcspPolicy::required());
        assert!(
            matches!(leaf_standing(&j), OcspStatus::Good { .. }),
            "{j:?}"
        );
    }

    #[test]
    fn a_critical_extension_not_understood_refuses_and_a_non_critical_one_passes() {
        let b = bench();
        let r = Response {
            extra_extension: Some((vec![1, 2, 3, 4], true)),
            ..Response::good_at(NOW)
        };
        let e = judge(
            &b,
            &[respond(&r, &b.leaf_der, &b.inter)],
            &NrasOcspPolicy::required(),
        )
        .expect_err("critical");
        assert!(e.to_string().contains("critical response extension"), "{e}");
        let r = Response {
            extra_extension: Some((vec![1, 2, 3, 4], false)),
            ..Response::good_at(NOW)
        };
        let j = judge(
            &b,
            &[respond(&r, &b.leaf_der, &b.inter)],
            &NrasOcspPolicy::required(),
        );
        assert!(
            matches!(leaf_standing(&j), OcspStatus::Good { .. }),
            "{j:?}"
        );
    }

    #[test]
    fn a_responder_status_other_than_successful_is_refused() {
        let b = bench();
        let r = Response {
            response_status: 3,
            ..Response::good_at(NOW)
        };
        let e = judge(
            &b,
            &[respond(&r, &b.leaf_der, &b.inter)],
            &NrasOcspPolicy::required(),
        )
        .expect_err("tryLater");
        assert!(e.to_string().contains("tryLater"), "{e}");
        let e =
            judge(&b, &[b"not DER".to_vec()], &NrasOcspPolicy::required()).expect_err("garbage");
        assert!(e.to_string().contains("OCSPResponse"), "{e}");
    }

    /// The whole chain: the intermediate is judged under a pinned CA 001,
    /// and said unverifiable when none is pinned.
    #[test]
    fn the_intermediate_is_judged_under_a_pinned_ca_001_and_said_unverifiable_without() {
        let b = bench();
        let leaf_good = respond(&Response::good_at(NOW), &b.leaf_der, &b.inter);
        let inter_good = respond(&Response::good_at(NOW), &b.inter.der, &b.root);
        let policy = with_ca(&b, NrasOcspPolicy::required());
        let j = judge(&b, &[leaf_good.clone(), inter_good], &policy).expect("both good");
        assert!(matches!(j[0].status, OcspStatus::Good { .. }), "{j:?}");
        assert!(matches!(j[1].status, OcspStatus::Good { .. }), "{j:?}");
        assert_eq!(j[1].what, "x5c intermediate");
        // CA 001 pinned, no response about the intermediate: Required refuses it.
        let e = judge(&b, std::slice::from_ref(&leaf_good), &policy)
            .expect_err("intermediate unanswered");
        assert!(
            matches!(
                e,
                OcspRefused::NotGood {
                    what: "x5c intermediate",
                    ..
                }
            ),
            "{e}"
        );
        // A revoked intermediate refuses under both modes.
        let r = Response {
            status: Status::Revoked(NOW - 10, None),
            ..Response::good_at(NOW)
        };
        let inter_revoked = respond(&r, &b.inter.der, &b.root);
        for mode in [NrasOcspMode::Required, NrasOcspMode::Transparent] {
            let e = judge(
                &b,
                &[leaf_good.clone(), inter_revoked.clone()],
                &policy.clone().with_mode(mode),
            )
            .expect_err("revoked intermediate");
            assert!(
                matches!(
                    e,
                    OcspRefused::Revoked {
                        what: "x5c intermediate",
                        ..
                    }
                ),
                "{e}"
            );
        }
        // Not pinned: said, not judged, under both modes.
        let j = judge(
            &b,
            &[leaf_good.clone(), inter_revoked],
            &NrasOcspPolicy::required(),
        )
        .expect("leaf good");
        assert_eq!(j[1].status, OcspStatus::IssuerNotPublished);
        assert!(j[1].to_string().contains("CA 001 is not published"));
    }

    /// The policy and the standings say their names: `required`, a pinned
    /// CA 001, a good status with its window, a revoked one with and without
    /// its reason.
    #[test]
    fn the_policy_and_the_standings_say_their_names() {
        assert_eq!(NrasOcspMode::Required.as_str(), "required");
        let b = bench();
        let d = with_ca(&b, NrasOcspPolicy::required()).describe();
        assert!(
            d.contains("nras-ocsp required") && d.contains("pinned"),
            "{d}"
        );
        let say = |status: OcspStatus| {
            OcspStanding {
                what: "x5c leaf",
                serial_hex: "01".into(),
                status,
            }
            .to_string()
        };
        assert_eq!(
            say(OcspStatus::Good {
                this_update: 1,
                next_update: 2,
            }),
            "x5c leaf (serial 01): good (thisUpdate 1, nextUpdate 2)"
        );
        assert_eq!(
            say(OcspStatus::Revoked {
                at: 3,
                reason: Some(1),
            }),
            "x5c leaf (serial 01): REVOKED at 3, reason 1"
        );
        assert_eq!(
            say(OcspStatus::Revoked {
                at: 3,
                reason: None
            }),
            "x5c leaf (serial 01): REVOKED at 3"
        );
    }

    /// Every responder status other than `successful` is refused by its
    /// name; a status outside the RFC's list is said unknown.
    #[test]
    fn each_responder_error_status_is_refused_by_its_name() {
        let b = bench();
        for (status, name) in [
            (1u8, "malformedRequest"),
            (2, "internalError"),
            (5, "sigRequired"),
            (6, "unauthorized"),
            (9, "unknown status"),
        ] {
            let r = Response {
                response_status: status,
                ..Response::good_at(NOW)
            };
            let e = judge(
                &b,
                &[respond(&r, &b.leaf_der, &b.inter)],
                &NrasOcspPolicy::required(),
            )
            .expect_err("not successful");
            assert!(e.to_string().contains(name), "{status}: {e}");
        }
    }

    /// Next to an unknown answer, a good one is the leaf's standing: the
    /// best verified answer counts, revoked above good above the rest.
    #[test]
    fn a_good_response_outranks_an_unknown_one_next_to_it() {
        let b = bench();
        let unknown = respond(
            &Response {
                status: Status::Unknown,
                ..Response::good_at(NOW)
            },
            &b.leaf_der,
            &b.inter,
        );
        let good = respond(&Response::good_at(NOW), &b.leaf_der, &b.inter);
        let j = judge(&b, &[unknown, good], &NrasOcspPolicy::production());
        assert!(
            matches!(leaf_standing(&j), OcspStatus::Good { .. }),
            "{j:?}"
        );
    }

    /// A pinned CA 001 that does not parse is a verifier misconfigured, not
    /// a responder's answer: refused under both modes, the intermediate
    /// named.
    #[test]
    fn a_pinned_ca_001_that_does_not_parse_refuses_under_both_modes() {
        let b = bench();
        let good = respond(&Response::good_at(NOW), &b.leaf_der, &b.inter);
        let broken = NrasOcspPolicy {
            intermediate_issuer_der: Some(vec![1, 2, 3]),
            ..NrasOcspPolicy::required()
        };
        for policy in [broken.clone(), broken.with_mode(NrasOcspMode::Transparent)] {
            let e = judge(&b, std::slice::from_ref(&good), &policy).expect_err("unparsable CA");
            assert!(
                e.to_string().contains("x5c intermediate")
                    && e.to_string().contains("does not parse"),
                "{e}"
            );
        }
    }

    /// A malformed response is refused with the place named, whatever the
    /// malformation, and a signature algorithm that does not fit the key is
    /// refused before any verification. An explicit v1 version, a single
    /// extension nobody reads and a delegated responder named by name are
    /// not malformations.
    #[test]
    fn a_malformed_response_is_refused_with_the_place_named() {
        let b = bench();
        let kp = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap();
        let marked = end_entity(
            &b.inter,
            &kp,
            &[7],
            &[ExtendedKeyUsagePurpose::OcspSigning],
            (1970, 4096),
        );
        let delegated = || Some((marked.clone(), Signer::of(&kp)));
        let ed = KeyPair::generate_for(&rcgen::PKCS_ED25519).unwrap();
        let ed_responder = end_entity(
            &b.inter,
            &ed,
            &[9],
            &[ExtendedKeyUsagePurpose::OcspSigning],
            (1970, 4096),
        );
        let other_ca = root("Another CA", p384_kp());
        let good = Response::good_at(NOW);
        let twisted = |t: Twist| Response {
            twist: Some(t),
            ..good.clone()
        };
        let cases: Vec<(Response, &str)> = vec![
            (twisted(Twist::Trailing), "bytes after it"),
            (twisted(Twist::TopTag(0xa0)), "context-specific"),
            (twisted(Twist::TopTag(0x60)), "application"),
            (twisted(Twist::TopTag(0xe0)), "private"),
            (twisted(Twist::TopTag(0x04)), "universal"),
            (twisted(Twist::WideStatus), "small non-negative integer"),
            (
                twisted(Twist::ResponseBytesTag(1)),
                "responseBytes: not [0]",
            ),
            (
                twisted(Twist::ResponseType(&[1, 3, 6, 1, 5, 5, 7, 48, 1, 4])),
                "is not id-pkix-ocsp-basic",
            ),
            (twisted(Twist::UnusedBits), "not a whole number of bytes"),
            (
                Response {
                    delegated: delegated(),
                    ..twisted(Twist::CertsTag(1))
                },
                "certs: not [0]",
            ),
            (twisted(Twist::Version(1)), "version is not v1"),
            (
                twisted(Twist::ResponderTag(3)),
                "neither byName [1] nor byKey [2]",
            ),
            (twisted(Twist::TwoSingles), "more than one SingleResponse"),
            (
                Response {
                    extra_extension: Some((vec![1, 2, 3, 4], false)),
                    ..twisted(Twist::ResponseExtTag(2))
                },
                "responseExtensions: not [1]",
            ),
            (twisted(Twist::GoodWithData), "good carries data"),
            (
                Response {
                    status: Status::Revoked(NOW - 3600, Some(1)),
                    ..twisted(Twist::ReasonTag(1))
                },
                "revocationReason: not [0]",
            ),
            (twisted(Twist::CertStatusTag(3)), "certStatus: neither good"),
            (
                twisted(Twist::SingleExt(vec![1, 2, 3, 4], true)),
                "critical single extension",
            ),
            (twisted(Twist::SingleExtTag(2)), "singleExtensions: not [1]"),
            (twisted(Twist::ExtTrailing), "bytes after extnValue"),
            (
                Response {
                    hash: &[2, 16, 840, 1, 101, 3, 4, 2, 3],
                    ..good.clone()
                },
                "is not SHA-1, SHA-256 or SHA-384",
            ),
            (
                twisted(Twist::SigAlg(&[1, 2, 840, 10045, 4, 3, 2])),
                "is not ecdsa-with-SHA384 for a P-384 key",
            ),
            (
                Response {
                    delegated: delegated(),
                    ..twisted(Twist::SigAlg(&[1, 2, 840, 10045, 4, 3, 3]))
                },
                "is not ecdsa-with-SHA256 for a P-256 key",
            ),
            (
                Response {
                    delegated: Some((ed_responder.clone(), b.inter.signer.clone())),
                    ..good.clone()
                },
                "neither RSA nor EC",
            ),
            (
                twisted(Twist::CertIdIssuer(other_ca.der.clone())),
                "names another issuer",
            ),
            (
                Response {
                    other_serial: Some(vec![0x80, 1, 2]),
                    ..good.clone()
                },
                "about serial",
            ),
            (
                Response {
                    other_serial: Some(vec![0]),
                    ..good.clone()
                },
                "about serial",
            ),
        ];
        for (r, says) in &cases {
            let e = judge(
                &b,
                &[respond(r, &b.leaf_der, &b.inter)],
                &NrasOcspPolicy::required(),
            )
            .expect_err(says);
            assert!(e.to_string().contains(says), "{says}: {e}");
        }
        for r in [
            twisted(Twist::Version(0)),
            twisted(Twist::SingleExt(vec![1, 2, 3, 4], false)),
            Response {
                delegated: delegated(),
                by_name: true,
                ..good.clone()
            },
        ] {
            let j = judge(
                &b,
                &[respond(&r, &b.leaf_der, &b.inter)],
                &NrasOcspPolicy::required(),
            );
            assert!(
                matches!(leaf_standing(&j), OcspStatus::Good { .. }),
                "{j:?}"
            );
        }
    }

    /// An RSA issuer's response: a signature algorithm that is not RSA's is
    /// refused by name; sha384WithRSAEncryption is read, and a signature made
    /// with SHA-256 does not verify under it.
    #[test]
    fn an_rsa_responses_signature_algorithm_must_be_rsas() {
        let root = root("CA 001", p384_kp());
        let inter = sub_ca("GPU Intermediate 004", rsa_kp(), &root);
        let leaf_der = end_entity(&inter, &p384_kp(), LEAF_SERIAL, &[], (1970, 4096));
        let b = Bench {
            root,
            inter,
            leaf_der,
        };
        const ECDSA_SHA256: &[u64] = &[1, 2, 840, 10045, 4, 3, 2];
        const SHA384_RSA: &[u64] = &[1, 2, 840, 113549, 1, 1, 12];
        for (arcs, says) in [(ECDSA_SHA256, "for an RSA key"), (SHA384_RSA, "signature")] {
            let r = Response {
                twist: Some(Twist::SigAlg(arcs)),
                ..Response::good_at(NOW)
            };
            let e = judge(
                &b,
                &[respond(&r, &b.leaf_der, &b.inter)],
                &NrasOcspPolicy::required(),
            )
            .expect_err(says);
            assert!(e.to_string().contains(says), "{says}: {e}");
        }
    }
}
