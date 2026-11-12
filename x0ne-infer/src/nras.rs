// SPDX-License-Identifier: Apache-2.0
//! Cryptographic verification of an NVIDIA NRAS GPU attestation token.
//!
//! The flow, as NVIDIA defines it: the guest collects GPU evidence and POSTs
//! it to `https://nras.attestation.nvidia.com/v4/attest/gpu` with a 64-hex
//! `eat_nonce`; NRAS answers with an EAT — a JWT signed **ES384**, whose
//! signing key is published at `/.well-known/jwks.json` with an `x5c`
//! certificate chain (leaf "NVIDIA Attestation Service GPU GH100", ~2-day
//! validity, rotating) issued by "NVIDIA Attestation Service GPU
//! Intermediate 004" (RSA, valid to 2029) or "… Intermediate 005" (P-384,
//! valid 2026-10-08 to 2030; 72 of the 200 keys of the JWKS read on
//! 2026-10-18, whose oldest 005 leaf dates from 2026-10-15), both issued by
//! "NVIDIA Attestation Service CA 001" — which NVIDIA does not publish
//! separately.
//!
//! What this module verifies, offline, given the token and a JWKS document:
//!
//! 1. the JWT signature (ES384, nothing else) under the P-384 key of the
//!    JWKS entry named by the token's `kid`;
//! 2. that key's `x5c` leaf is signed by a **pinned NVIDIA intermediate**,
//!    with the one algorithm its key type allows ([`NrasLeafSignature`]:
//!    RSA-SHA256 under 004, ECDSA-SHA384 under 005) — one arrow beyond what
//!    NVIDIA's own SDK checks (nvtrust `nras_utils.py` verifies only the leaf
//!    over TLS);
//! 3. the claims: issuer, expiry, our nonce echoed, and
//!    `x-nvidia-overall-att-result == true`;
//! 4. every PER-GPU token ([`verify_nras_attestation`]): same
//!    signature chain, our nonce, `measres == "success"` and the claims
//!    NVIDIA's own example relying-party policy requires
//!    ([`REQUIRED_TRUE_PER_GPU`]), NVAT's default policy on its three
//!    certificate chains plus `secboot` and `dbgstat` ([`gpu_policy`]), and
//!    `ueid` equal to the serial of the attestation certificate WE read from
//!    the local GPU — what makes the attested GPU ours rather than merely
//!    genuine;
//! 5. the overall token's `submods`: one entry per per-GPU token, by its
//!    name, holding the SHA-256 of that token, and no other entry — so the
//!    overall verdict and the per-GPU tokens come from one NRAS answer;
//!    and no local certificate serial twice, so one GPU never counts as two.
//!
//! Stated limitation (`LIMITATIONS`): the anchors are the two intermediates,
//! not the root — NVIDIA publishes no root certificate for this chain. The
//! pins are MEASURED (fetched from the production JWKS, vendored under
//! `tests/`), like the Intel pin in `pcs`. CA 001 alone is not pinned: it
//! would accept any key it certifies, under a pin nobody reviewed.
//! Without CA 001, an intermediate's own signature cannot be checked, so each
//! is pinned as its whole DER certificate, not only its key: its validity
//! window is then NVIDIA's, not one chosen by whoever serves the JWKS.

use crate::nras_ocsp::{NrasOcspPolicy, OcspRefused, OcspStanding};
use base64::Engine;
use p384::ecdsa::signature::Verifier as _;
use rsa::pkcs1v15::{Signature as RsaSignature, VerifyingKey as RsaVerifyingKey};
use rsa::pkcs8::DecodePublicKey;
use rsa::signature::Verifier as _;
use rsa::RsaPublicKey;
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use x509_parser::prelude::{FromDer, X509Certificate};

/// SHA-256 of the DER SubjectPublicKeyInfo of "NVIDIA Attestation Service
/// GPU Intermediate 004" (serial 5FE7637A…, valid 2025-12-08 → 2029-12-08).
///
/// MEASURED, not recalled: extracted from the `x5c` of the production JWKS
/// (`https://nras.attestation.nvidia.com/.well-known/jwks.json`) on
/// 2026-08-20 and digested with `openssl x509 -pubkey | openssl pkey -pubin
/// -outform der | sha256sum`. The certificate itself is vendored under
/// `tests/nvidia_nras_gpu_intermediate_004.der` and a test keeps this
/// constant honest through the same code path the verifier uses.
pub const NVIDIA_NRAS_GPU_INTERMEDIATE_004_SPKI_SHA256_HEX: &str =
    "fd32837f954e2c45db073105166dfe6985ae0480bb113fba63b091a75affe896";

// MEASURED 2026-10-18T10:03:39Z: SPKI SHA-256 of x5c[1] "GPU Intermediate 005" (P-384), 72 of
// 200 keys of GET https://nras.attestation.nvidia.com/.well-known/jwks.json. Cert SHA-256
// 38e3daf3…31a0d343, vendored as tests/nvidia_nras_gpu_intermediate_005.der, digested by a test.
pub const NVIDIA_NRAS_GPU_INTERMEDIATE_005_SPKI_SHA256_HEX: &str =
    "ba14d00377699189b6762c22c744a9dbdc62aa9396097ada7150426c6ff39afc";

// MEASURED 2026-10-18: `sha256sum` of the vendored DER of each intermediate. All 200 x5c[1] of the
// production JWKS GET of 2026-10-18T10:03:39Z are byte-identical to one of them (128 to 004, 72 to
// 005). A re-issued intermediate, same key or not, is refused until it is measured and pinned here.
pub const NVIDIA_NRAS_GPU_INTERMEDIATE_004_CERT_SHA256_HEX: &str =
    "2df8907cf4d6c277b855d407ec178a649930a8e73f7294e8fce66e6ee27c5175";
pub const NVIDIA_NRAS_GPU_INTERMEDIATE_005_CERT_SHA256_HEX: &str =
    "38e3daf3cdbd5243bb15dd7993edcdd83a7238cb6745b980f8f3bf0a31a0d343";

/// The one algorithm a pinned intermediate signs NRAS leaves with. Bound to
/// the pin's key type, never read from the certificate being judged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NrasLeafSignature {
    /// RSA key; leaf `sha256WithRSAEncryption` (PKCS#1 v1.5, SHA-256).
    RsaPkcs1Sha256,
    /// P-384 key; leaf `ecdsa-with-SHA384`.
    EcdsaP384Sha384,
}

/// One pinned NRAS intermediate: the SHA-256 of its whole DER certificate and
/// of its SubjectPublicKeyInfo, and the leaf signature its key type allows.
/// The fields are private: outside this module a pin is cloned and read,
/// never built or edited (`tests/cloud_source_sweep.rs` holds it so).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NrasIntermediatePin {
    cert_sha256_hex: Cow<'static, str>,
    spki_sha256_hex: Cow<'static, str>,
    leaf_signature: NrasLeafSignature,
}

impl NrasIntermediatePin {
    pub fn cert_sha256_hex(&self) -> &str {
        &self.cert_sha256_hex
    }
    pub fn spki_sha256_hex(&self) -> &str {
        &self.spki_sha256_hex
    }
    pub fn leaf_signature(&self) -> NrasLeafSignature {
        self.leaf_signature
    }
}

/// What production verifies NRAS under: Intermediates 004 and 005, nothing
/// else. Tests inject their own throwaway chains through the same parameter.
pub const NVIDIA_NRAS_GPU_INTERMEDIATE_PINS: [NrasIntermediatePin; 2] = [
    NrasIntermediatePin {
        cert_sha256_hex: Cow::Borrowed(NVIDIA_NRAS_GPU_INTERMEDIATE_004_CERT_SHA256_HEX),
        spki_sha256_hex: Cow::Borrowed(NVIDIA_NRAS_GPU_INTERMEDIATE_004_SPKI_SHA256_HEX),
        leaf_signature: NrasLeafSignature::RsaPkcs1Sha256,
    },
    NrasIntermediatePin {
        cert_sha256_hex: Cow::Borrowed(NVIDIA_NRAS_GPU_INTERMEDIATE_005_CERT_SHA256_HEX),
        spki_sha256_hex: Cow::Borrowed(NVIDIA_NRAS_GPU_INTERMEDIATE_005_SPKI_SHA256_HEX),
        leaf_signature: NrasLeafSignature::EcdsaP384Sha384,
    },
];

pub const LIMITATIONS: &[&str] = &[
    "anchored at the NVIDIA GPU intermediate CAs 004 and 005, each pinned as its whole certificate, not at their root CA 001 (not published)",
    // 2026-10-31: judged from stapled OCSP responses (`nras_ocsp`); the CRL
    // NVIDIA also publishes is not read, and CA 001 is not published, so the
    // intermediate's own status cannot be verified.
    "revocation of the NRAS signing chain: judged from the OCSP responses stapled with the token, under the policy each verdict names, a verified revoked refusing under every policy; the intermediate's own status is unverifiable while NVIDIA does not publish CA 001, and the CRL NVIDIA also publishes is not read; the x5c validity windows ARE checked",
    "clock rollback is caught only against NVIDIA's signed iat, so only on the GPU path and only beyond 300 s; the CPU-only path still trusts the host clock entirely",
    // Derivation read from nvtrust `cc_admin.py` (ueid = serial of the GPU
    // attestation certificate). Production tokens matched it on 2026-10-14,
    // -17 and -18 (verdicts "bound by ueid"); a mismatch refuses, never admits.
    "GPU identity bound by ueid == local attestation-certificate serial, a derivation taken from NVIDIA's nvtrust SDK; production NRAS tokens matched it on the cloud runs of 2026-10-14, 2026-10-17 and 2026-10-18, and a token whose ueid differs is refused",
];

/// How far behind NVIDIA's `iat` the local clock may read before it counts
/// as moved rather than drifting. Five minutes: what JWT libraries allow by
/// convention, and far below the day-scale rollback that revives collateral.
pub const CLOCK_SKEW_SECS: u64 = 300;

/// Per-GPU claims that must be literally `true`.
///
/// MEASURED, not recalled (2026-09-18): the intersection of the
/// `detached-claims` of NVIDIA's two published example policies, nvtrust
/// `attestation_sdk/tests/policies/remote/{v3,v4}/NVGPURemotePolicyExample.json`.
/// The intersection, because the two versions rename the certificate-chain
/// claims (a boolean in v3, a nested status object in v4) and requiring
/// either spelling would refuse genuine hardware under the other.
pub const REQUIRED_TRUE_PER_GPU: &[&str] = &[
    "x-nvidia-gpu-arch-check",
    "x-nvidia-gpu-attestation-report-parsed",
    "x-nvidia-gpu-attestation-report-nonce-match",
    "x-nvidia-gpu-attestation-report-signature-verified",
    "x-nvidia-gpu-driver-rim-fetched",
    "x-nvidia-gpu-driver-rim-schema-validated",
    "x-nvidia-gpu-driver-rim-signature-verified",
    "x-nvidia-gpu-driver-rim-measurements-available",
    "x-nvidia-gpu-vbios-rim-fetched",
    "x-nvidia-gpu-vbios-rim-schema-validated",
    "x-nvidia-gpu-vbios-rim-signature-verified",
    "x-nvidia-gpu-vbios-rim-measurements-available",
    "x-nvidia-gpu-vbios-index-no-conflict",
];

/// The GPU's attestation-certificate chain and the driver and VBIOS RIM
/// signers' chains, as NRAS reports them per GPU (claims 3.0 and 4.0 alike;
/// read on the production tokens of 2026-10-17 and 2026-10-18).
pub const CERT_CHAINS_PER_GPU: &[&str] = &[
    "x-nvidia-gpu-attestation-report-cert-chain",
    "x-nvidia-gpu-driver-rim-cert-chain",
    "x-nvidia-gpu-vbios-rim-cert-chain",
];

/// NVAT 2026.10.02's default policy (`src/claims_evaluator.cpp:81-165`): each chain
/// `valid`, OCSP `good`, OCSP nonce matched, response valid; no opaque-data mismatch record.
/// Hardening NVAT's default does not read: `secboot` true and `dbgstat` "disabled".
pub fn gpu_policy(claims: &serde_json::Value) -> Result<(), String> {
    for chain in CERT_CHAINS_PER_GPU {
        for (field, want) in [
            ("x-nvidia-cert-status", serde_json::json!("valid")),
            ("x-nvidia-cert-ocsp-status", serde_json::json!("good")),
            ("x-nvidia-cert-ocsp-nonce-matches", serde_json::json!(true)),
            ("x-nvidia-cert-ocsp-response-valid", serde_json::json!(true)),
        ] {
            let got = &claims[*chain][field];
            if *got != want {
                return Err(format!("{chain}: {field} is {got}, not {want}"));
            }
        }
    }
    let opaque = &claims["x-nvidia-mismatch-opaque-data-records"];
    if !opaque.is_null() {
        return Err(format!("x-nvidia-mismatch-opaque-data-records is {opaque}"));
    }
    if claims["secboot"] != true {
        return Err(format!("secboot is {}, not true", claims["secboot"]));
    }
    if claims["dbgstat"] != "disabled" {
        return Err(format!(
            "dbgstat is {}, not \"disabled\"",
            claims["dbgstat"]
        ));
    }
    Ok(())
}

/// Why the token did not verify — one variant per arrow.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NrasRejected {
    #[error("token is not a three-part JWT")]
    NotAJwt,
    #[error("JWT header: {0}")]
    Header(String),
    #[error("no JWKS key matches kid {0:?}")]
    NoMatchingKey(String),
    #[error("JWKS key {0}: {1}")]
    BadJwksKey(String, String),
    #[error("x5c leaf is not signed by the pinned NVIDIA intermediate: {0}")]
    LeafNotPinnedIssuer(String),
    #[error("JWT signature does not verify under the JWKS leaf key")]
    JwtSignature,
    #[error("claims: {0}")]
    Claims(String),
    /// A per-GPU token (`GPU-0`…), or `*` for the set as a whole.
    #[error("per-GPU token {0}: {1}")]
    PerGpu(String, String),
    #[error("GPU identity: {0}")]
    GpuIdentity(String),
    #[error(
        "the local clock ({now_unix}) is behind the time NVIDIA signed this fresh token \
         ({issued_at}) — a rolled-back clock makes stale collateral look current"
    )]
    ClockBehindIssuer { now_unix: u64, issued_at: u64 },
    /// A verified OCSP response says a certificate of the signing chain is
    /// revoked: refused under every `nras_ocsp` policy.
    #[error("NRAS signing chain revoked: {0}")]
    SigningChainRevoked(String),
    /// The `nras_ocsp` policy requires a verified `good` for the chain and
    /// the stapled responses do not establish one.
    #[error("NRAS signing chain revocation not established: {0}")]
    SigningChainRevocationUnverified(String),
}

/// What a verified token established.
#[derive(Debug, Clone)]
pub struct VerifiedGpu {
    /// `x-nvidia-overall-att-result` — true means every submodule passed.
    pub overall: bool,
    /// The decoded claims, for receipts and diagnostics.
    pub claims: serde_json::Value,
    /// `ueid` of every per-GPU token, each matched to a local certificate
    /// serial. Empty only from [`verify_nras_token`] alone, which sees no
    /// per-GPU token; the production path is [`verify_nras_attestation`].
    pub gpu_ueids: Vec<String>,
    /// What the stapled OCSP responses established about the signing chain,
    /// by certificate, under the `nras_ocsp` policy given: what every verdict
    /// names. Empty from [`verify_nras_token`] alone.
    pub revocation: Vec<OcspStanding>,
    pub limitations: &'static [&'static str],
}

fn b64url(seg: &str) -> Result<Vec<u8>, NrasRejected> {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(seg)
        .map_err(|e| NrasRejected::Header(format!("base64url: {e}")))
}

/// Verify one NRAS JWT against a JWKS document and the pinned intermediates.
///
/// `expected_nonce_hex` is the `eat_nonce` we sent; `now_unix` bounds
/// `exp`/`iat`. Pure: no network — the caller fetched the JWKS (over TLS)
/// and the token; this function decides. Production passes
/// [`NVIDIA_NRAS_GPU_INTERMEDIATE_PINS`].
pub fn verify_nras_token(
    token: &str,
    jwks_json: &serde_json::Value,
    expected_nonce_hex: &str,
    now_unix: u64,
    pins: &[NrasIntermediatePin],
) -> Result<VerifiedGpu, NrasRejected> {
    verified_overall(token, jwks_json, expected_nonce_hex, now_unix, pins).map(|(v, _)| v)
}

/// [`verify_nras_token`], and the `kid` the token was verified under.
fn verified_overall(
    token: &str,
    jwks_json: &serde_json::Value,
    expected_nonce_hex: &str,
    now_unix: u64,
    pins: &[NrasIntermediatePin],
) -> Result<(VerifiedGpu, String), NrasRejected> {
    let (claims, kid) = verified_claims(token, jwks_json, now_unix, pins)?;
    check_nonce(&claims, expected_nonce_hex)?;
    let overall = claims["x-nvidia-overall-att-result"] == true
        || claims["x-nvidia-overall-att-result"] == "true";
    if !overall {
        return Err(NrasRejected::Claims(
            "x-nvidia-overall-att-result is not true".into(),
        ));
    }
    Ok((
        VerifiedGpu {
            overall,
            claims,
            gpu_ueids: Vec::new(),
            revocation: Vec::new(),
            limitations: LIMITATIONS,
        },
        kid,
    ))
}

fn check_nonce(claims: &serde_json::Value, expected_nonce_hex: &str) -> Result<(), NrasRejected> {
    // SECURITY: an absent claim read as "" and answered an empty nonce (measured
    // 2026-11-05). In-tree callers pass 64 hex digits; the API held no such floor.
    if expected_nonce_hex.is_empty() {
        return Err(NrasRejected::Claims(
            "no nonce of ours to compare eat_nonce with".into(),
        ));
    }
    let nonce = claims["eat_nonce"]
        .as_str()
        .ok_or_else(|| NrasRejected::Claims("no eat_nonce in the token".into()))?;
    if !nonce.eq_ignore_ascii_case(expected_nonce_hex) {
        return Err(NrasRejected::Claims("eat_nonce does not match ours".into()));
    }
    Ok(())
}

/// Arrows 1 and 2, plus issuer and expiry: the claims of ONE NRAS JWT, once
/// it is known to be NVIDIA's and current. Shared by the overall token and
/// every per-GPU token — they are signed by the same key, and a per-GPU
/// token checked any less than the overall one is a claim nobody signed.
fn verified_claims(
    token: &str,
    jwks_json: &serde_json::Value,
    now_unix: u64,
    pins: &[NrasIntermediatePin],
) -> Result<(serde_json::Value, String), NrasRejected> {
    // ── Split and parse ──────────────────────────────────────────────────
    let mut parts = token.split('.');
    let (h64, p64, s64) = match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(h), Some(p), Some(s), None) => (h, p, s),
        _ => return Err(NrasRejected::NotAJwt),
    };
    let header: serde_json::Value =
        serde_json::from_slice(&b64url(h64)?).map_err(|e| NrasRejected::Header(e.to_string()))?;
    if header["alg"] != "ES384" {
        return Err(NrasRejected::Header(format!(
            "alg {} (only ES384 is accepted)",
            header["alg"]
        )));
    }
    // SECURITY: RFC 7515, 4.1.11: an extension listed and not understood makes the JWS
    // invalid, and none is understood here. 8 real runs carry `alg` and `kid` alone (2026-11-05).
    if header.get("crit").is_some() {
        return Err(NrasRejected::Header(
            "crit is present and this verifier understands no JWS extension".into(),
        ));
    }
    let kid = header["kid"]
        .as_str()
        .ok_or_else(|| NrasRejected::Header("no kid".into()))?;

    // ── The JWKS key for this kid, and its x5c chain ─────────────────────
    let keys = jwks_json["keys"]
        .as_array()
        .ok_or_else(|| NrasRejected::Header("JWKS has no keys".into()))?;
    let key = keys
        .iter()
        .find(|k| k["kid"] == kid)
        .ok_or_else(|| NrasRejected::NoMatchingKey(kid.into()))?;
    let x5c = key["x5c"]
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| NrasRejected::BadJwksKey(kid.into(), "no x5c".into()))?;

    // ── Arrow 2: the leaf is signed by a pinned intermediate ─────────────
    let leaf_key = pinned_leaf_key(kid, x5c, pins, now_unix)?;

    // ── Arrow 1: the JWT under the leaf's ES384 key ──────────────────────
    let sig_bytes = b64url(s64)?;
    let sig =
        p384::ecdsa::Signature::from_slice(&sig_bytes).map_err(|_| NrasRejected::JwtSignature)?;
    let signed = format!("{h64}.{p64}");
    leaf_key
        .verify(signed.as_bytes(), &sig)
        .map_err(|_| NrasRejected::JwtSignature)?;

    // ── Arrow 3: the claims ──────────────────────────────────────────────
    let claims: serde_json::Value =
        serde_json::from_slice(&b64url(p64)?).map_err(|e| NrasRejected::Claims(e.to_string()))?;
    let iss = claims["iss"].as_str().unwrap_or_default();
    if !is_nras_issuer(iss) {
        return Err(NrasRejected::Claims(format!("issuer {iss:?}")));
    }
    if let Some(exp) = claims["exp"].as_u64() {
        if now_unix > exp {
            return Err(NrasRejected::Claims(format!(
                "expired at {exp}, now {now_unix}"
            )));
        }
    } else {
        return Err(NrasRejected::Claims("no exp".into()));
    }
    // SECURITY (2026-09-18): the clock is the host's, and every
    // freshness window in this crate — Intel's collateral above all — is
    // judged against it. Wind it back and a cached, perfectly authentic,
    // long-superseded collateral is "current" again, without forging a byte.
    // This token is the one timestamp here the host cannot choose: it answers
    // OUR nonce, so it was minted just now, and NVIDIA signed when. A clock
    // that reads earlier than `iat` is not slow, it has been moved.
    let iat = claims["iat"]
        .as_u64()
        .ok_or_else(|| NrasRejected::Claims("no iat".into()))?;
    if now_unix.saturating_add(CLOCK_SKEW_SECS) < iat {
        return Err(NrasRejected::ClockBehindIssuer {
            now_unix,
            issued_at: iat,
        });
    }
    // RFC 7519 §4.1.5, optional; NVIDIA's overall token carries it. Unread
    // until 2026-10-27 (review of that day, finding 5); same skew as `iat`.
    if !claims["nbf"].is_null() {
        let nbf = claims["nbf"]
            .as_u64()
            .ok_or_else(|| NrasRejected::Claims(format!("nbf is {}, not a time", claims["nbf"])))?;
        if now_unix.saturating_add(CLOCK_SKEW_SECS) < nbf {
            return Err(NrasRejected::Claims(format!(
                "not valid before nbf {nbf}, now {now_unix}"
            )));
        }
    }
    Ok((claims, kid.to_string()))
}

/// NRAS's origin, as production tokens name it (2026-10-17, 2026-10-18).
const NRAS_ISSUER: &str = "https://nras.attestation.nvidia.com";

/// The issuer is that origin, or a path under it. A prefix test alone let
/// `https://nras.attestation.nvidia.com.example.net` pass as NRAS.
fn is_nras_issuer(iss: &str) -> bool {
    iss.strip_prefix(NRAS_ISSUER)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

/// SHA-256, lowercase hex: the form of every pin in this module.
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// SHA-256 of a certificate's DER SubjectPublicKeyInfo, lowercase hex.
fn spki_sha256_hex(cert: &X509Certificate<'_>) -> String {
    sha256_hex(cert.public_key().raw)
}

/// Arrow 2 for one JWKS entry: `x5c[1]` is a pinned certificate, its key verifies
/// `x5c[0]` with the one algorithm that pin allows, and both validity windows
/// contain `now_unix`. Returns the leaf's P-384 key, the only key a token may
/// be verified with. The leaf's key usage, extended key usage and subject are
/// not checked: NVIDIA publishes no profile for the NRAS signing leaf
/// (searched 2026-10-27), and a guessed one could refuse genuine tokens.
fn pinned_leaf_key(
    kid: &str,
    x5c: &[serde_json::Value],
    pins: &[NrasIntermediatePin],
    now_unix: u64,
) -> Result<p384::ecdsa::VerifyingKey, NrasRejected> {
    let leaf_der = base64::engine::general_purpose::STANDARD
        .decode(x5c[0].as_str().unwrap_or_default())
        .map_err(|e| NrasRejected::BadJwksKey(kid.into(), format!("x5c[0] base64: {e}")))?;
    let (_, leaf) = X509Certificate::from_der(&leaf_der)
        .map_err(|e| NrasRejected::BadJwksKey(kid.into(), format!("x5c[0] X.509: {e}")))?;
    // The intermediate travels in x5c[1]; its bytes and its SPKI must match a
    // pin, and its key must verify the leaf's signature by that pin's algorithm.
    let inter_der = base64::engine::general_purpose::STANDARD
        .decode(x5c.get(1).and_then(|v| v.as_str()).unwrap_or_default())
        .map_err(|e| NrasRejected::LeafNotPinnedIssuer(format!("x5c[1] base64: {e}")))?;
    let (_, inter) = X509Certificate::from_der(&inter_der)
        .map_err(|e| NrasRejected::LeafNotPinnedIssuer(format!("x5c[1] X.509: {e}")))?;
    // SECURITY: the whole DER, not only the key. Nothing checks
    // CA 001's signature on x5c[1]; a certificate made around a pinned key
    // would bring its own validity window (checked below) and pass.
    let (inter_cert_hex, inter_spki_hex) = (sha256_hex(&inter_der), spki_sha256_hex(&inter));
    let pin = pins
        .iter()
        .find(|p| p.cert_sha256_hex == inter_cert_hex && p.spki_sha256_hex == inter_spki_hex)
        .ok_or_else(|| {
            NrasRejected::LeafNotPinnedIssuer(
                if pins.iter().any(|p| p.spki_sha256_hex == inter_spki_hex) {
                    "x5c[1] carries a pinned key but is not the pinned certificate"
                } else {
                    "intermediate SPKI matches no pin"
                }
                .into(),
            )
        })?;
    verify_leaf_signature(&leaf, &inter, pin.leaf_signature)?;

    // SECURITY: both certificates' validity windows. NVIDIA gives
    // the NRAS leaf roughly two days precisely so a leaked signing key stops
    // working; until 2026-08-21 nothing here read `notAfter`. The only
    // freshness check was the JWT's own `exp` claim — signed by this very
    // leaf, so a holder of the leaked key simply wrote a later one. Neither
    // window was in the path, and `LIMITATIONS` did not say so. The
    // intermediate's window is NVIDIA's because its DER is pinned above.
    for (cert, what) in [(&leaf, "x5c leaf"), (&inter, "x5c intermediate")] {
        let (nb, na) = (
            cert.validity().not_before.timestamp(),
            cert.validity().not_after.timestamp(),
        );
        let now = now_unix as i64;
        if now < nb || now > na {
            return Err(NrasRejected::LeafNotPinnedIssuer(format!(
                "{what}: outside its validity window [{nb}, {na}] at {now}"
            )));
        }
    }

    // id-ecPublicKey on secp384r1, read from the whole SPKI: the key the JWT
    // is verified with is P-384 and declared so, whatever signed the leaf.
    <p384::ecdsa::VerifyingKey as p384::pkcs8::DecodePublicKey>::from_public_key_der(
        leaf.public_key().raw,
    )
    .map_err(|e| NrasRejected::BadJwksKey(kid.into(), format!("leaf key not P-384: {e}")))
}

/// The leaf's signature under the pinned intermediate, by `alg` only. The
/// leaf must declare that same algorithm, in its signed TBS field and in its
/// unsigned outer one, written the same way in both; the intermediate's key
/// must be of the type `alg` needs.
fn verify_leaf_signature(
    leaf: &X509Certificate<'_>,
    inter: &X509Certificate<'_>,
    alg: NrasLeafSignature,
) -> Result<(), NrasRejected> {
    use x509_parser::oid_registry::{OID_PKCS1_SHA256WITHRSA, OID_SIG_ECDSA_WITH_SHA384};
    let bad = |why: String| NrasRejected::LeafNotPinnedIssuer(why);
    let expected = match alg {
        NrasLeafSignature::RsaPkcs1Sha256 => OID_PKCS1_SHA256WITHRSA,
        NrasLeafSignature::EcdsaP384Sha384 => OID_SIG_ECDSA_WITH_SHA384,
    };
    for (field, oid) in [
        ("signatureAlgorithm", &leaf.signature_algorithm.algorithm),
        (
            "tbsCertificate.signature",
            &leaf.tbs_certificate.signature.algorithm,
        ),
    ] {
        if *oid != expected {
            return Err(bad(format!(
                "leaf {field} is {oid}, the pinned intermediate signs with {expected}"
            )));
        }
    }
    // RFC 5280, section 4.1.1.2: the two are the same AlgorithmIdentifier,
    // byte for byte. No signature covers the outer one.
    if !crate::pcs::signature_algorithm_repeated(leaf.as_raw()) {
        return Err(bad(
            "leaf: the signed and the outer signature algorithms differ".into(),
        ));
    }
    let tbs = leaf.tbs_certificate.as_ref();
    let sig = leaf.signature_value.data.as_ref();
    // `public_key().raw` is the full SPKI DER — the form pkcs8 decodes.
    let spki = inter.public_key().raw;
    match alg {
        NrasLeafSignature::RsaPkcs1Sha256 => {
            let key = RsaPublicKey::from_public_key_der(spki)
                .map_err(|e| bad(format!("intermediate key is not RSA: {e}")))?;
            let sig =
                RsaSignature::try_from(sig).map_err(|e| bad(format!("leaf signature: {e}")))?;
            // `rsa::sha2`, not `sha2`: rsa 0.9 is on the previous `digest` generation
            // (its 0.10 is still a release candidate, 2026-09-18) and only accepts
            // the hasher it re-exports. Collapses back to one `sha2` when rsa moves.
            RsaVerifyingKey::<rsa::sha2::Sha256>::new(key)
                .verify(tbs, &sig)
                .map_err(|_| bad("leaf signature does not verify".into()))
        }
        NrasLeafSignature::EcdsaP384Sha384 => {
            let key =
                <p384::ecdsa::VerifyingKey as p384::pkcs8::DecodePublicKey>::from_public_key_der(
                    spki,
                )
                .map_err(|e| bad(format!("intermediate key is not P-384: {e}")))?;
            // X.509 carries ECDSA signatures as DER; the P-384 verifier hashes with SHA-384.
            let sig = p384::ecdsa::Signature::from_der(sig)
                .map_err(|e| bad(format!("leaf signature: {e}")))?;
            key.verify(tbs, &sig)
                .map_err(|_| bad("leaf signature does not verify".into()))
        }
    }
}

/// The serial of the LEAF of a GPU attestation certificate chain, in decimal
/// — the form NRAS puts in `ueid`. `chain_b64` is what we POST as
/// `certificate`: base64 of the chain NVML returned, PEM, leaf first (DER is
/// accepted too, a single certificate).
pub fn gpu_leaf_serial_decimal(chain_b64: &str) -> Result<String, NrasRejected> {
    let bad = |why: String| NrasRejected::GpuIdentity(format!("local certificate chain: {why}"));
    let raw = base64::engine::general_purpose::STANDARD
        .decode(chain_b64.trim())
        .map_err(|e| bad(format!("base64: {e}")))?;
    let der = if raw.starts_with(b"-----BEGIN") {
        let (_, pem) =
            x509_parser::pem::parse_x509_pem(&raw).map_err(|e| bad(format!("PEM: {e}")))?;
        pem.contents
    } else {
        raw
    };
    let (_, leaf) = X509Certificate::from_der(&der).map_err(|e| bad(format!("X.509: {e}")))?;
    Ok(leaf.tbs_certificate.serial.to_str_radix(10))
}

/// The production entry point: the overall token AND every per-GPU token.
///
/// SECURITY (2026-09-18). Until this function, `main` bound
/// the per-GPU tokens to `_per_gpu` and dropped them, so two things were
/// never established:
///
/// - **whose GPU it is.** The overall token proves that A genuine GPU in CC
///   mode signed our nonce. TEE.fail says it in as many words: NVIDIA quotes
///   "are not bound to the identity of a specific CVM or even the CPU running
///   it". An operator relays our challenge to an H100 elsewhere and serves
///   from whatever is actually in the machine. `ueid` is the serial of the
///   attested GPU's certificate; `local_gpu_serials` are the serials of the
///   certificates WE read over NVML. They must be the same set.
/// - **what it measured.** `measres` and the RIM checks live per GPU. The
///   overall boolean is NVIDIA's summary of them; we now read the detail we
///   were handed instead of trusting the summary of a detail we threw away.
///
/// One token per GPU we sent evidence for, no more and no fewer: a dropped
/// token is an unattested GPU.
///
/// `ocsp_responses` are the OCSP responses stapled about the JWKS signing
/// chain (DER), judged under `ocsp` ([`crate::nras_ocsp`]): a verified
/// revoked refuses under every policy; what else they establish is
/// [`VerifiedGpu::revocation`], for the verdict to name.
#[allow(clippy::too_many_arguments)]
pub fn verify_nras_attestation(
    overall_token: &str,
    per_gpu: &[(String, String)],
    local_gpu_serials: &[String],
    jwks_json: &serde_json::Value,
    expected_nonce_hex: &str,
    now_unix: u64,
    pins: &[NrasIntermediatePin],
    ocsp_responses: &[Vec<u8>],
    ocsp: &NrasOcspPolicy,
) -> Result<VerifiedGpu, NrasRejected> {
    let (mut verified, overall_kid) =
        verified_overall(overall_token, jwks_json, expected_nonce_hex, now_unix, pins)?;
    let mut kids = vec![overall_kid];
    if per_gpu.is_empty() || per_gpu.len() != local_gpu_serials.len() {
        return Err(NrasRejected::PerGpu(
            "*".into(),
            format!(
                "NRAS answered for {} GPU(s), evidence was sent for {}",
                per_gpu.len(),
                local_gpu_serials.len()
            ),
        ));
    }
    // A serial named twice lets one GPU's token stand for two GPUs: the
    // multiset comparison below holds only over distinct serials.
    let mut distinct = local_gpu_serials.to_vec();
    distinct.sort();
    if let Some([serial, ..]) = distinct.windows(2).find(|w| w[0] == w[1]) {
        return Err(NrasRejected::GpuIdentity(format!(
            "local attestation-certificate serial {} named twice: one GPU is not two",
            serial
        )));
    }
    let mut ueids = Vec::with_capacity(per_gpu.len());
    for (name, token) in per_gpu {
        let reject = |why: String| NrasRejected::PerGpu(name.clone(), why);
        let (claims, kid) =
            verified_claims(token, jwks_json, now_unix, pins).map_err(|e| reject(e.to_string()))?;
        kids.push(kid);
        check_nonce(&claims, expected_nonce_hex).map_err(|e| reject(e.to_string()))?;
        if claims["measres"] != "success" {
            return Err(reject(format!("measres is {}", claims["measres"])));
        }
        if let Some(claim) = REQUIRED_TRUE_PER_GPU.iter().find(|c| claims[**c] != true) {
            return Err(reject(format!("{claim} is {}", claims[*claim])));
        }
        gpu_policy(&claims).map_err(reject)?;
        let ueid = claims["ueid"]
            .as_str()
            .ok_or_else(|| reject(format!("ueid is {}, not a string", claims["ueid"])))?;
        ueids.push(ueid.to_string());
    }
    // Same SET, compared sorted: NRAS names GPUs by index and owes us no
    // ordering; the local serials are distinct (above), so two tokens cannot
    // claim one GPU.
    let (mut theirs, mut ours) = (ueids.clone(), local_gpu_serials.to_vec());
    theirs.sort();
    ours.sort();
    if theirs != ours {
        return Err(NrasRejected::GpuIdentity(format!(
            "the attested GPU is not the local one: token ueid {theirs:?}, \
             local attestation-certificate serial {ours:?}"
        )));
    }
    check_submods(&verified.claims, per_gpu)?;
    verified.revocation =
        signing_chain_revocation(jwks_json, &kids, ocsp_responses, now_unix, ocsp)?;
    verified.gpu_ueids = ueids;
    Ok(verified)
}

/// The signing chain's revocation from the stapled OCSP responses: each
/// distinct `kid`'s x5c, already verified under a pin by `pinned_leaf_key`,
/// judged once under `ocsp`.
fn signing_chain_revocation(
    jwks_json: &serde_json::Value,
    kids: &[String],
    ocsp_responses: &[Vec<u8>],
    now_unix: u64,
    ocsp: &NrasOcspPolicy,
) -> Result<Vec<OcspStanding>, NrasRejected> {
    let mut distinct = kids.to_vec();
    distinct.sort();
    distinct.dedup();
    let mut standings = Vec::new();
    for kid in &distinct {
        let (leaf_der, inter_der) = x5c_chain_der(jwks_json, kid)?;
        let (_, leaf) = X509Certificate::from_der(&leaf_der)
            .map_err(|e| NrasRejected::BadJwksKey(kid.clone(), format!("x5c[0] X.509: {e}")))?;
        let (_, inter) = X509Certificate::from_der(&inter_der)
            .map_err(|e| NrasRejected::LeafNotPinnedIssuer(format!("x5c[1] X.509: {e}")))?;
        let judged = crate::nras_ocsp::judge_chain(&leaf, &inter, ocsp_responses, now_unix, ocsp)
            .map_err(|e| match &e {
            OcspRefused::Revoked { .. } => NrasRejected::SigningChainRevoked(e.to_string()),
            OcspRefused::NotGood { .. } => {
                NrasRejected::SigningChainRevocationUnverified(e.to_string())
            }
        })?;
        standings.extend(judged);
    }
    Ok(standings)
}

/// `x5c[0]` and `x5c[1]` of the JWKS key `kid`, DER.
fn x5c_chain_der(
    jwks_json: &serde_json::Value,
    kid: &str,
) -> Result<(Vec<u8>, Vec<u8>), NrasRejected> {
    let key = jwks_json["keys"]
        .as_array()
        .and_then(|keys| keys.iter().find(|k| k["kid"] == kid))
        .ok_or_else(|| NrasRejected::NoMatchingKey(kid.into()))?;
    let x5c = key["x5c"]
        .as_array()
        .ok_or_else(|| NrasRejected::BadJwksKey(kid.into(), "no x5c".into()))?;
    let der = |i: usize, what: &str| {
        base64::engine::general_purpose::STANDARD
            .decode(x5c.get(i).and_then(|v| v.as_str()).unwrap_or_default())
            .map_err(|e| NrasRejected::BadJwksKey(kid.into(), format!("{what} base64: {e}")))
    };
    Ok((der(0, "x5c[0]")?, der(1, "x5c[1]")?))
}

/// The overall token covers exactly these per-GPU tokens: its `submods`
/// names each one, by its name, with the SHA-256 of the token as sent, and
/// names nothing else. NVIDIA's NVAT checks the digests and their number;
/// the shape is the one NRAS gave on the run of 2026-10-17, read more
/// strictly than NVAT does (`submod_sha256`). Without it, an overall verdict
/// and per-GPU tokens from two NRAS answers could be put together.
fn check_submods(
    overall: &serde_json::Value,
    per_gpu: &[(String, String)],
) -> Result<(), NrasRejected> {
    let set = |why: String| NrasRejected::PerGpu("*".into(), why);
    let submods = overall["submods"].as_object().ok_or_else(|| {
        set(format!(
            "the overall token's submods is {}, not an object naming each per-GPU token",
            overall["submods"]
        ))
    })?;
    let mut names: Vec<&str> = per_gpu.iter().map(|(n, _)| n.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    if names.len() != per_gpu.len() || submods.len() != per_gpu.len() {
        return Err(set(format!(
            "the overall token's submods names {} GPU(s), {} per-GPU token(s) came under {} name(s)",
            submods.len(),
            per_gpu.len(),
            names.len()
        )));
    }
    for (name, token) in per_gpu {
        let reject = |why: String| NrasRejected::PerGpu(name.clone(), why);
        let entry = submods
            .get(name)
            .ok_or_else(|| reject("the overall token's submods does not name it".into()))?;
        let named = submod_sha256(entry).ok_or_else(|| {
            reject(format!(
                "the overall token's submods entry is {entry}, not [\"DIGEST\", [\"SHA-256\" or \"SHA256\", <64 hex>]]"
            ))
        })?;
        let this = sha256_hex(token.as_bytes());
        if !named.eq_ignore_ascii_case(&this) {
            return Err(reject(format!(
                "the overall token's submods names another token (SHA-256 {named}, this one {this})"
            )));
        }
    }
    Ok(())
}

/// The digest of one `submods` entry: `["DIGEST", [<label>, <64 hex>]]`.
/// Label: NRAS's "SHA-256" or "SHA256", NVAT's own spelling. NVAT reads the
/// digest without the label; any other label refuses here.
fn submod_sha256(entry: &serde_json::Value) -> Option<&str> {
    let [kind, digest] = entry.as_array()?.as_slice() else {
        return None;
    };
    let [alg, hex] = digest.as_array()?.as_slice() else {
        return None;
    };
    let hex = hex.as_str()?;
    (kind == "DIGEST"
        && (alg == "SHA-256" || alg == "SHA256")
        && hex.len() == 64
        && hex.bytes().all(|b| b.is_ascii_hexdigit()))
    .then_some(hex)
}

/// A pin over a throwaway intermediate, its whole DER, for a bench: compiled
/// only where `x0ne_sdk_test_anchor` is set, which x0ne-sdk's build script sets
/// for its `test-anchor` feature and nothing else sets. Without it, the two
/// pins above are the only ones a build can hold.
/// `None`: not a certificate, or its key is neither RSA nor P-384.
#[cfg(x0ne_sdk_test_anchor)]
pub fn throwaway_pin(der: &[u8]) -> Option<NrasIntermediatePin> {
    let (_, cert) = X509Certificate::from_der(der).ok()?;
    let spki = cert.public_key().raw;
    let p384 = <p384::ecdsa::VerifyingKey as p384::pkcs8::DecodePublicKey>::from_public_key_der;
    let leaf_signature = if RsaPublicKey::from_public_key_der(spki).is_ok() {
        NrasLeafSignature::RsaPkcs1Sha256
    } else if p384(spki).is_ok() {
        NrasLeafSignature::EcdsaP384Sha384
    } else {
        return None;
    };
    Some(NrasIntermediatePin {
        cert_sha256_hex: Cow::Owned(sha256_hex(der)),
        spki_sha256_hex: Cow::Owned(spki_sha256_hex(&cert)),
        leaf_signature,
    })
}

// Not when compiled into x0ne-sdk (its build.rs sets `x0ne_sdk`): these tests
// need this crate's dev-dependencies (rcgen) and fixtures.
#[cfg(all(test, not(x0ne_sdk)))]
pub(crate) mod tests {
    use super::*;
    use p384::ecdsa::signature::Signer as _;
    use rcgen::{BasicConstraints, CertificateParams, IsCa, Issuer, KeyPair};

    /// Our own "NVIDIA": an RSA (004-like) or P-384 (005-like)
    /// intermediate and an ES384 leaf, a JWKS carrying their x5c, and a
    /// correctly signed EAT — so the whole path is proven both ways, like the
    /// PCS fixture.
    pub(crate) struct Fixture {
        pub(crate) token: String,
        pub(crate) jwks: serde_json::Value,
        /// The one pin this fixture's intermediate verifies under.
        pub(crate) pins: Vec<NrasIntermediatePin>,
        pub(crate) nonce_hex: String,
        /// One per-GPU token, signed like the overall one.
        pub(crate) per_gpu: Vec<(String, String)>,
        /// What `gpu_leaf_serial_decimal` reads from the local chain.
        pub(crate) gpu_serials: Vec<String>,
        /// Signs further tokens under the same leaf, for the refusal tests.
        pub(crate) sign: std::sync::Arc<dyn Fn(&serde_json::Value) -> String + Send + Sync>,
        /// The same, under the JWS header given first.
        pub(crate) sign_with_header: HeaderSigner,
        /// The x5c as DER, and the signer of the intermediate's key: a test
        /// staples OCSP responses about the leaf with them.
        pub(crate) leaf_der: Vec<u8>,
        pub(crate) inter_der: Vec<u8>,
        pub(crate) inter_signer: crate::nras_ocsp::tests::Signer,
    }

    type HeaderSigner =
        std::sync::Arc<dyn Fn(&serde_json::Value, &serde_json::Value) -> String + Send + Sync>;

    /// 2^64: one past `u64`, so a serial read through a machine integer
    /// instead of a big one cannot produce it.
    pub(crate) const GPU_SERIAL_BYTES: [u8; 9] = [1, 0, 0, 0, 0, 0, 0, 0, 0];
    pub(crate) const GPU_SERIAL_DECIMAL: &str = "18446744073709551616";

    /// Per-GPU claims as NVIDIA's sample `detached_claims_remote.json` has
    /// them, every required boolean true.
    pub(crate) fn gpu_claims(nonce_hex: &str, exp: u64, ueid: &str) -> serde_json::Value {
        let mut claims = serde_json::json!({
            "iss": "https://nras.attestation.nvidia.com",
            "exp": exp,
            "iat": exp.saturating_sub(3600),
            "eat_nonce": nonce_hex,
            "measres": "success",
            "ueid": ueid,
            "hwmodel": "GH100 A01 GSP BROM",
        });
        for c in REQUIRED_TRUE_PER_GPU {
            claims[*c] = serde_json::json!(true);
        }
        for chain in CERT_CHAINS_PER_GPU {
            claims[*chain] = serde_json::json!({
                "x-nvidia-cert-status": "valid",
                "x-nvidia-cert-ocsp-status": "good",
                "x-nvidia-cert-ocsp-nonce-matches": true,
                "x-nvidia-cert-ocsp-response-valid": true,
                "x-nvidia-cert-revocation-reason": null,
            });
        }
        claims["secboot"] = serde_json::json!(true);
        claims["dbgstat"] = serde_json::json!("disabled");
        claims
    }

    /// The overall token's `submods` over `per_gpu`, in the shape NRAS gave
    /// on the run of 2026-10-17: `{"GPU-0": ["DIGEST", ["SHA-256", hex]]}`,
    /// hex being the SHA-256 of that GPU's token as sent.
    pub(crate) fn submods_of(per_gpu: &[(String, String)]) -> serde_json::Value {
        per_gpu
            .iter()
            .map(|(name, token)| {
                (
                    name.clone(),
                    serde_json::json!(["DIGEST", ["SHA-256", sha256_hex(token.as_bytes())]]),
                )
            })
            .collect::<serde_json::Map<_, _>>()
            .into()
    }

    /// The local side: a PEM chain as NVML hands it, base64'd as we POST it.
    pub(crate) fn gpu_chain_b64(serial: &[u8]) -> String {
        let kp = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P384_SHA384).unwrap();
        let mut params = CertificateParams::new(vec![]).unwrap();
        params.serial_number = Some(rcgen::SerialNumber::from_slice(serial));
        let cert = params.self_signed(&kp).unwrap();
        b64std(cert.pem().as_bytes())
    }

    /// The composite test in `verify` uses the same fixture.
    pub(crate) fn fixture_for_composite() -> Fixture {
        build_fixture(true, NONCE, 2_000)
    }

    /// The same, under a P-384 intermediate: what NRAS publishes under 005.
    pub(crate) fn fixture_for_composite_p384() -> Fixture {
        build_fixture_under(Inter::P384, true, NONCE, 2_000, (1970, 4096))
    }

    /// The key type of a fixture's intermediate.
    #[derive(Clone, Copy)]
    pub(crate) enum Inter {
        Rsa,
        P384,
    }

    fn b64url_enc(b: &[u8]) -> String {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b)
    }
    fn b64std(b: &[u8]) -> String {
        base64::engine::general_purpose::STANDARD.encode(b)
    }

    fn build_fixture(overall: bool, nonce_hex: &str, exp: u64) -> Fixture {
        build_fixture_with_leaf_window(overall, nonce_hex, exp, (1970, 4096))
    }

    /// `leaf_window`: the (notBefore, notAfter) years of the ES384 leaf. The
    /// default starts at 1970 because these tests speak in small epoch
    /// seconds (1_500, 9_000) about the JWT's own claims; the x5c validity
    /// windows are now checked, and a fixture minted with rcgen's 1975 default would
    /// make every one of them fail for the wrong reason.
    fn build_fixture_with_leaf_window(
        overall: bool,
        nonce_hex: &str,
        exp: u64,
        leaf_window: (i32, i32),
    ) -> Fixture {
        build_fixture_under(Inter::Rsa, overall, nonce_hex, exp, leaf_window)
    }

    /// The intermediate's key pair and the leaf signature it makes.
    fn intermediate_key(kind: Inter) -> (KeyPair, NrasLeafSignature) {
        match kind {
            // RSA-2048 (rcgen: PKCS_RSA_SHA256 needs an external key;
            // generate with the rsa crate and wrap).
            Inter::Rsa => {
                let mut rng = rsa_test_rng();
                let sk = rsa::RsaPrivateKey::new(&mut rng, 2048).expect("rsa key");
                let kp = KeyPair::from_pkcs8_pem_and_sign_algo(
                    &rsa_to_pkcs8_pem(&sk),
                    &rcgen::PKCS_RSA_SHA256,
                )
                .expect("rsa keypair");
                (kp, NrasLeafSignature::RsaPkcs1Sha256)
            }
            Inter::P384 => (
                KeyPair::generate_for(&rcgen::PKCS_ECDSA_P384_SHA384).unwrap(),
                NrasLeafSignature::EcdsaP384Sha384,
            ),
        }
    }

    /// A throwaway intermediate CA of `kind`: its DER, the issuer that signs
    /// leaves under it, and the pin that accepts it.
    fn intermediate(kind: Inter) -> (Vec<u8>, Issuer<'static, KeyPair>, NrasIntermediatePin) {
        let (kp, leaf_signature) = intermediate_key(kind);
        let mut params = CertificateParams::new(vec![]).unwrap();
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        params.not_before = rcgen::date_time_ymd(1970, 1, 1);
        params.not_after = rcgen::date_time_ymd(4096, 1, 1);
        let der = params.clone().self_signed(&kp).unwrap().der().to_vec();
        let pin = NrasIntermediatePin {
            cert_sha256_hex: sha256_hex(&der).into(),
            spki_sha256_hex: spki_hex(&der).into(),
            leaf_signature,
        };
        (der, Issuer::new(params, kp), pin)
    }

    fn spki_hex(der: &[u8]) -> String {
        spki_sha256_hex(&X509Certificate::from_der(der).expect("parses").1)
    }

    pub(crate) fn build_fixture_under(
        kind: Inter,
        overall: bool,
        nonce_hex: &str,
        exp: u64,
        leaf_window: (i32, i32),
    ) -> Fixture {
        let (inter_der, inter_issuer, pin) = intermediate(kind);

        // ES384 leaf, signed by the intermediate.
        let leaf_kp = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P384_SHA384).unwrap();
        let mut leaf_params = CertificateParams::new(vec![]).unwrap();
        leaf_params.not_before = rcgen::date_time_ymd(leaf_window.0, 1, 1);
        leaf_params.not_after = rcgen::date_time_ymd(leaf_window.1, 1, 1);
        let leaf_cert = leaf_params.signed_by(&leaf_kp, &inter_issuer).unwrap();

        // The JWT, signed with the leaf key.
        let kid = "nv-eat-kid-test-1";
        let sign = es384_signer(&leaf_kp, kid);
        let sign_with_header = es384_signer_with_header(&leaf_kp);
        let per_gpu = vec![(
            "GPU-0".to_string(),
            sign(&gpu_claims(nonce_hex, exp, GPU_SERIAL_DECIMAL)),
        )];
        let claims = serde_json::json!({
            "iss": "https://nras.attestation.nvidia.com",
            "exp": exp,
            "iat": exp.saturating_sub(3600),
            "eat_nonce": nonce_hex,
            "x-nvidia-overall-att-result": overall,
            "measres": "success",
            "hwmodel": "GH100 A01 GSP BROM",
            "submods": submods_of(&per_gpu),
        });
        let token = sign(&claims);

        let jwks = serde_json::json!({"keys": [{
            "kty": "EC", "crv": "P-384", "kid": kid,
            "x5c": [b64std(leaf_cert.der()), b64std(&inter_der)],
        }]});
        Fixture {
            token,
            jwks,
            pins: vec![pin],
            nonce_hex: nonce_hex.into(),
            per_gpu,
            gpu_serials: vec![GPU_SERIAL_DECIMAL.to_string()],
            sign,
            sign_with_header,
            leaf_der: leaf_cert.der().to_vec(),
            inter_der,
            inter_signer: crate::nras_ocsp::tests::Signer::of(inter_issuer.key()),
        }
    }

    /// Signs EATs as `kid` (ES384) with an rcgen P-384 leaf key.
    fn es384_signer(
        leaf_kp: &KeyPair,
        kid: &str,
    ) -> std::sync::Arc<dyn Fn(&serde_json::Value) -> String + Send + Sync> {
        let sign = es384_signer_with_header(leaf_kp);
        let header = serde_json::json!({"alg": "ES384", "kid": kid});
        std::sync::Arc::new(move |claims: &serde_json::Value| sign(&header, claims))
    }

    /// Signs EATs under any JWS header (ES384) with an rcgen P-384 leaf key.
    fn es384_signer_with_header(leaf_kp: &KeyPair) -> HeaderSigner {
        let leaf_sk = p384::SecretKey::from_sec1_pem(&leaf_kp.serialize_pem())
            .map(p384::ecdsa::SigningKey::from)
            .or_else(|_| {
                p384::pkcs8::DecodePrivateKey::from_pkcs8_pem(&leaf_kp.serialize_pem())
                    .map(|sk: p384::SecretKey| p384::ecdsa::SigningKey::from(sk))
            })
            .expect("leaf key readable");
        std::sync::Arc::new(
            move |header: &serde_json::Value, claims: &serde_json::Value| {
                let signed = format!(
                    "{}.{}",
                    b64url_enc(serde_json::to_vec(header).unwrap().as_slice()),
                    b64url_enc(serde_json::to_vec(claims).unwrap().as_slice())
                );
                let sig: p384::ecdsa::Signature = leaf_sk.sign(signed.as_bytes());
                format!("{signed}.{}", b64url_enc(&sig.to_bytes()))
            },
        )
    }

    fn attest(
        fx: &Fixture,
        per_gpu: &[(String, String)],
        local: &[String],
    ) -> Result<VerifiedGpu, NrasRejected> {
        verify_nras_attestation(
            &fx.token,
            per_gpu,
            local,
            &fx.jwks,
            &fx.nonce_hex,
            1_500,
            &fx.pins,
            &[],
            &NrasOcspPolicy::production(),
        )
    }

    /// SECURITY: a failure means a host that winds its clock back —
    /// to make superseded Intel collateral read as current — is believed.
    #[test]
    fn a_clock_wound_back_behind_the_tokens_issue_time_is_refused() {
        // exp 100_000 → iat 96_400 in this fixture.
        let fx = build_fixture(true, NONCE, 100_000);
        let at = |now| verify_nras_token(&fx.token, &fx.jwks, NONCE, now, &fx.pins);
        assert!(at(96_400).is_ok(), "on time");
        assert!(
            at(96_400 - CLOCK_SKEW_SECS).is_ok(),
            "drift within the skew"
        );
        assert_eq!(
            at(96_400 - CLOCK_SKEW_SECS - 1).unwrap_err(),
            NrasRejected::ClockBehindIssuer {
                now_unix: 96_400 - CLOCK_SKEW_SECS - 1,
                issued_at: 96_400
            }
        );
        assert_eq!(
            CLOCK_SKEW_SECS, 300,
            "a fixture never derives from a mutable constant"
        );
    }

    /// `nbf` (RFC 7519 §4.1.5), which NVIDIA's overall token carries (Hopper
    /// single-GPU example), is read when present: later than the caller's
    /// clock by more than the skew `iat` allows, refused; within it, or
    /// absent, not. One that is not a time is refused.
    #[test]
    fn a_token_not_yet_valid_is_refused() {
        // exp 100_000 → iat 96_400 in this fixture.
        let fx = build_fixture(true, NONCE, 100_000);
        let p64 = fx.token.split('.').nth(1).unwrap();
        let claims: serde_json::Value = serde_json::from_slice(&b64url(p64).unwrap()).unwrap();
        let with = |nbf: serde_json::Value| {
            let mut c = claims.clone();
            c["nbf"] = nbf;
            (fx.sign)(&c)
        };
        let at = |token: &str| verify_nras_token(token, &fx.jwks, NONCE, 96_400, &fx.pins);
        assert!(at(&fx.token).is_ok(), "no nbf");
        assert!(at(&with(96_400.into())).is_ok(), "valid from now");
        assert!(
            at(&with((96_400 + CLOCK_SKEW_SECS).into())).is_ok(),
            "within the skew"
        );
        for (nbf, why) in [
            (
                serde_json::json!(96_400 + CLOCK_SKEW_SECS + 1),
                "past the skew",
            ),
            (serde_json::json!(u64::MAX), "never"),
            (serde_json::json!("96400"), "a string"),
            (serde_json::json!(-1), "negative"),
        ] {
            match at(&with(nbf)) {
                Err(NrasRejected::Claims(m)) => assert!(m.contains("nbf"), "{why}: {m}"),
                other => panic!("{why}: {other:?}"),
            }
        }
    }

    #[test]
    fn the_local_serial_is_read_in_decimal_and_past_64_bits() {
        assert_eq!(
            gpu_leaf_serial_decimal(&gpu_chain_b64(&GPU_SERIAL_BYTES)).unwrap(),
            GPU_SERIAL_DECIMAL
        );
        assert!(matches!(
            gpu_leaf_serial_decimal("bm90IGEgY2VydA=="),
            Err(NrasRejected::GpuIdentity(_))
        ));
    }

    #[test]
    fn a_gpu_whose_token_names_our_certificate_is_accepted() {
        let fx = build_fixture(true, NONCE, 2_000);
        let v = attest(&fx, &fx.per_gpu, &fx.gpu_serials).expect("verifies");
        assert_eq!(v.gpu_ueids, [GPU_SERIAL_DECIMAL]);
    }

    /// SECURITY: a failure means the relay attack is open again — a
    /// perfectly genuine token, for a genuine GPU in CC mode, that is simply
    /// not the GPU in this machine, opens the server.
    #[test]
    fn a_genuine_token_for_a_gpu_that_is_not_ours_is_refused() {
        let fx = build_fixture(true, NONCE, 2_000);
        let elsewhere = vec!["481858311570236547493498289839486179865867984732".to_string()];
        assert!(matches!(
            attest(&fx, &fx.per_gpu, &elsewhere),
            Err(NrasRejected::GpuIdentity(_))
        ));
    }

    /// SECURITY: a failure means a GPU whose measurements did NOT
    /// match NVIDIA's reference values serves anyway, under a green overall.
    #[test]
    fn a_gpu_whose_measurements_failed_is_refused_whatever_the_overall_says() {
        let fx = build_fixture(true, NONCE, 2_000);
        let mut claims = gpu_claims(NONCE, 2_000, GPU_SERIAL_DECIMAL);
        claims["measres"] = serde_json::json!("fail");
        let bad = vec![("GPU-0".to_string(), (fx.sign)(&claims))];
        assert!(matches!(
            attest(&fx, &bad, &fx.gpu_serials),
            Err(NrasRejected::PerGpu(_, why)) if why.contains("measres")
        ));
        for claim in REQUIRED_TRUE_PER_GPU {
            let mut claims = gpu_claims(NONCE, 2_000, GPU_SERIAL_DECIMAL);
            claims[*claim] = serde_json::json!(false);
            let bad = vec![("GPU-0".to_string(), (fx.sign)(&claims))];
            assert!(
                matches!(attest(&fx, &bad, &fx.gpu_serials), Err(NrasRejected::PerGpu(_, why)) if why.contains(claim)),
                "{claim}"
            );
        }
    }

    /// A token NVIDIA signed for our nonce and GPU, outside NVAT's
    /// default policy or our hardening, is refused; the fixture as is passes.
    #[test]
    fn a_gpu_outside_nvidias_policy_or_our_hardening_is_refused() {
        let fx = build_fixture(true, NONCE, 2_000);
        let token = |claims: &serde_json::Value| vec![("GPU-0".to_string(), (fx.sign)(claims))];
        let ok = gpu_claims(NONCE, 2_000, GPU_SERIAL_DECIMAL);
        assert!(attest(&fx, &token(&ok), &fx.gpu_serials).is_ok());
        let mut cases: Vec<(String, serde_json::Value)> = Vec::new();
        for chain in [
            "x-nvidia-gpu-attestation-report-cert-chain",
            "x-nvidia-gpu-driver-rim-cert-chain",
            "x-nvidia-gpu-vbios-rim-cert-chain",
        ] {
            for (field, bad) in [
                ("x-nvidia-cert-status", serde_json::json!("revoked")),
                ("x-nvidia-cert-ocsp-status", serde_json::json!("revoked")),
                ("x-nvidia-cert-ocsp-nonce-matches", serde_json::json!(false)),
                (
                    "x-nvidia-cert-ocsp-response-valid",
                    serde_json::json!(false),
                ),
            ] {
                let mut c = ok.clone();
                c[chain][field] = bad;
                cases.push((format!("{chain}.{field}"), c));
            }
            let mut c = ok.clone();
            c.as_object_mut().expect("an object").remove(chain);
            cases.push((format!("{chain} absent"), c));
        }
        for (claim, bad) in [
            ("secboot", serde_json::json!(false)),
            ("secboot", serde_json::Value::Null),
            ("dbgstat", serde_json::json!("enabled")),
            ("dbgstat", serde_json::Value::Null),
            (
                "x-nvidia-mismatch-opaque-data-records",
                serde_json::json!([{ "index": 1 }]),
            ),
        ] {
            let mut c = ok.clone();
            c[claim] = bad;
            cases.push((format!("{claim} = {}", c[claim]), c));
        }
        for (what, claims) in cases {
            assert!(
                matches!(
                    attest(&fx, &token(&claims), &fx.gpu_serials),
                    Err(NrasRejected::PerGpu(_, _))
                ),
                "{what}"
            );
        }
    }

    /// SECURITY: a failure means per-GPU claims are read without being
    /// authenticated — anyone on the path writes `measres: success`.
    #[test]
    fn a_per_gpu_token_is_held_to_the_same_signature_and_nonce_as_the_overall() {
        let fx = build_fixture(true, NONCE, 2_000);
        let (name, token) = fx.per_gpu[0].clone();
        let (body, _sig) = token.rsplit_once('.').unwrap();
        let forged = vec![(name.clone(), format!("{body}.{}", b64url_enc(&[0u8; 96])))];
        assert!(matches!(
            attest(&fx, &forged, &fx.gpu_serials),
            Err(NrasRejected::PerGpu(_, _))
        ));
        let replayed = vec![(
            name,
            (fx.sign)(&gpu_claims(&"00".repeat(32), 2_000, GPU_SERIAL_DECIMAL)),
        )];
        assert!(matches!(
            attest(&fx, &replayed, &fx.gpu_serials),
            Err(NrasRejected::PerGpu(_, why)) if why.contains("eat_nonce")
        ));
    }

    /// One token per GPU we sent evidence for: a dropped token is an
    /// unattested GPU, and a duplicated one is two GPUs vouched for by one.
    #[test]
    fn the_tokens_must_cover_exactly_the_gpus_we_sent() {
        let fx = build_fixture(true, NONCE, 2_000);
        let two_local = vec![GPU_SERIAL_DECIMAL.to_string(), "7".to_string()];
        assert!(matches!(
            attest(&fx, &fx.per_gpu, &two_local),
            Err(NrasRejected::PerGpu(star, _)) if star == "*"
        ));
        // Zero of each is not "they match": it is an overall token alone,
        // which is exactly what this function exists to stop accepting.
        assert!(matches!(
            attest(&fx, &[], &[]),
            Err(NrasRejected::PerGpu(star, _)) if star == "*"
        ));
        let twice = vec![
            fx.per_gpu[0].clone(),
            ("GPU-1".into(), fx.per_gpu[0].1.clone()),
        ];
        assert!(matches!(
            attest(&fx, &twice, &two_local),
            Err(NrasRejected::GpuIdentity(_))
        ));
    }

    /// The fixture's overall claims with `submods` replaced (`None`:
    /// removed), signed again under the fixture's leaf.
    fn overall_with_submods(fx: &Fixture, submods: Option<serde_json::Value>) -> String {
        let payload = fx.token.split('.').nth(1).unwrap();
        let mut claims: serde_json::Value =
            serde_json::from_slice(&b64url(payload).unwrap()).unwrap();
        match submods {
            Some(s) => claims["submods"] = s,
            None => {
                claims.as_object_mut().unwrap().remove("submods");
            }
        }
        (fx.sign)(&claims)
    }

    fn attest_under(
        fx: &Fixture,
        overall: &str,
        per_gpu: &[(String, String)],
    ) -> Result<VerifiedGpu, NrasRejected> {
        verify_nras_attestation(
            overall,
            per_gpu,
            &fx.gpu_serials,
            &fx.jwks,
            &fx.nonce_hex,
            1_500,
            &fx.pins,
            &[],
            &NrasOcspPolicy::production(),
        )
    }

    /// SECURITY: the overall token names, in `submods`, the SHA-256 of each
    /// per-GPU token it covers (NVIDIA's NVAT checks it). A genuine per-GPU
    /// token for the same nonce and GPU, taken from another NRAS answer and
    /// put next to this overall token, is refused; so are an overall token
    /// without `submods`, one naming a GPU more or another GPU, and an entry
    /// that is not a SHA-256 digest.
    #[test]
    fn the_overall_token_must_name_each_per_gpu_token_it_covers() {
        let fx = build_fixture(true, NONCE, 2_000);
        let refused = |overall: &str, per_gpu: &[(String, String)], what: &str| {
            let r = attest_under(&fx, overall, per_gpu);
            assert!(
                matches!(&r, Err(NrasRejected::PerGpu(_, why)) if why.contains("submods")),
                "{what}: {r:?}"
            );
        };
        attest_under(&fx, &fx.token, &fx.per_gpu).expect("control: the fixture passes");
        // Another answer of NRAS for the same nonce and GPU, issued later.
        let mut later = gpu_claims(NONCE, 2_000, GPU_SERIAL_DECIMAL);
        later["iat"] = serde_json::json!(1);
        let other = vec![("GPU-0".to_string(), (fx.sign)(&later))];
        attest_under(
            &fx,
            &overall_with_submods(&fx, Some(submods_of(&other))),
            &other,
        )
        .expect("control: the other answer passes under its own overall token");
        refused(&fx.token, &other, "a per-GPU token of another answer");
        refused(&overall_with_submods(&fx, None), &fx.per_gpu, "no submods");
        refused(
            &overall_with_submods(&fx, Some(serde_json::json!([]))),
            &fx.per_gpu,
            "submods not an object",
        );
        let mut more = submods_of(&fx.per_gpu);
        more["GPU-1"] = more["GPU-0"].clone();
        refused(
            &overall_with_submods(&fx, Some(more)),
            &fx.per_gpu,
            "a GPU more",
        );
        let renamed = submods_of(&[("GPU-7".to_string(), fx.per_gpu[0].1.clone())]);
        refused(
            &overall_with_submods(&fx, Some(renamed)),
            &fx.per_gpu,
            "another GPU name",
        );
        let digest = sha256_hex(fx.per_gpu[0].1.as_bytes());
        for (what, entry) in [
            (
                "SHA-384 label",
                serde_json::json!(["DIGEST", ["SHA-384", digest]]),
            ),
            ("not DIGEST", serde_json::json!(["JWT", fx.per_gpu[0].1])),
            (
                "short digest",
                serde_json::json!(["DIGEST", ["SHA-256", &digest[..32]]]),
            ),
            ("bare digest", serde_json::json!(digest)),
        ] {
            refused(
                &overall_with_submods(&fx, Some(serde_json::json!({ "GPU-0": entry }))),
                &fx.per_gpu,
                what,
            );
        }
    }

    /// The `submods` of the overall token NRAS gave on the run of
    /// 2026-10-17, as it came: its GPU-0 entry is that run's per-GPU token
    /// digest (`tests/VENDORED.md`). The check reads NRAS's shape.
    #[test]
    fn the_submods_of_a_real_overall_token_read_as_its_per_gpu_token_digest() {
        const GPU0_2026_10_17: &str =
            "8b92c6781d6375e82fd3f987218aea935b43e247d7d644f6b23b8b93aae0e547";
        let real: serde_json::Value = serde_json::from_str(
            r#"{"GPU-0": ["DIGEST", ["SHA-256", "8b92c6781d6375e82fd3f987218aea935b43e247d7d644f6b23b8b93aae0e547"]]}"#,
        )
        .unwrap();
        assert_eq!(submod_sha256(&real["GPU-0"]), Some(GPU0_2026_10_17));
    }

    /// SECURITY: the issuer is NVIDIA's NRAS origin, compared whole: a host
    /// that merely starts with it, a userinfo or a port is another server.
    /// A path under the origin is still NRAS.
    #[test]
    fn the_issuer_is_nras_origin_not_a_host_that_starts_with_it() {
        let fx = build_fixture(true, NONCE, 2_000);
        let payload = fx.token.split('.').nth(1).unwrap();
        let claims: serde_json::Value = serde_json::from_slice(&b64url(payload).unwrap()).unwrap();
        let with_iss = |iss: &str| {
            let mut c = claims.clone();
            c["iss"] = serde_json::json!(iss);
            verify_nras_token(&(fx.sign)(&c), &fx.jwks, &fx.nonce_hex, 1_500, &fx.pins)
        };
        for iss in [
            "https://nras.attestation.nvidia.com",
            "https://nras.attestation.nvidia.com/v3",
        ] {
            with_iss(iss).unwrap_or_else(|e| panic!("{iss}: {e}"));
        }
        for iss in [
            "https://nras.attestation.nvidia.com.example.net/v3",
            "https://nras.attestation.nvidia.com@example.net",
            "https://nras.attestation.nvidia.com:8443",
            "http://nras.attestation.nvidia.com",
            "",
        ] {
            let r = with_iss(iss);
            assert!(
                matches!(&r, Err(NrasRejected::Claims(why)) if why.contains("issuer")),
                "{iss}: {r:?}"
            );
        }
    }

    /// NRAS labels a `submods` digest "SHA-256"; NVAT's own EAT writes
    /// "SHA256" for the same algorithm. Either names SHA-256, whose value is
    /// recomputed here; any other label is refused.
    #[test]
    fn a_submods_digest_labelled_sha256_without_a_dash_is_read() {
        let fx = build_fixture(true, NONCE, 2_000);
        let digest = sha256_hex(fx.per_gpu[0].1.as_bytes());
        let labelled = |alg: &str| {
            overall_with_submods(
                &fx,
                Some(serde_json::json!({ "GPU-0": ["DIGEST", [alg, &digest]] })),
            )
        };
        attest_under(&fx, &labelled("SHA256"), &fx.per_gpu).expect("NVAT's label");
        attest_under(&fx, &labelled("SHA-256"), &fx.per_gpu).expect("NRAS's label");
        for alg in ["sha256", "SHA-512", "SHA_256", ""] {
            assert!(
                attest_under(&fx, &labelled(alg), &fx.per_gpu).is_err(),
                "{alg}"
            );
        }
    }

    /// SECURITY: one GPU is not two. A local certificate serial named twice
    /// lets one GPU's token stand for both, under two names or as two NRAS
    /// answers for it; both are refused, as is one token under two names.
    #[test]
    fn one_gpu_counted_twice_is_refused() {
        let fx = build_fixture(true, NONCE, 2_000);
        let twice_local = vec![fx.gpu_serials[0].clone(), fx.gpu_serials[0].clone()];
        let token = fx.per_gpu[0].1.clone();
        let mut later = gpu_claims(NONCE, 2_000, GPU_SERIAL_DECIMAL);
        later["iat"] = serde_json::json!(1);
        for (what, per_gpu) in [
            (
                "one token under two names",
                vec![
                    ("GPU-0".to_string(), token.clone()),
                    ("GPU-1".to_string(), token.clone()),
                ],
            ),
            (
                "two answers for one GPU",
                vec![
                    ("GPU-0".to_string(), token),
                    ("GPU-1".to_string(), (fx.sign)(&later)),
                ],
            ),
        ] {
            let overall = overall_with_submods(&fx, Some(submods_of(&per_gpu)));
            let r = verify_nras_attestation(
                &overall,
                &per_gpu,
                &twice_local,
                &fx.jwks,
                &fx.nonce_hex,
                1_500,
                &fx.pins,
                &[],
                &NrasOcspPolicy::production(),
            );
            assert!(
                matches!(&r, Err(NrasRejected::GpuIdentity(why)) if why.contains("twice")),
                "{what}: {:?}",
                r.map(|v| v.gpu_ueids)
            );
        }
    }

    /// Deterministic RNG for RSA keygen: a SHA-256 counter stream. Tests
    /// need reproducibility, not entropy.
    pub(crate) struct CounterRng {
        counter: u64,
        buf: Vec<u8>,
    }
    impl rsa::rand_core::RngCore for CounterRng {
        fn next_u32(&mut self) -> u32 {
            let mut b = [0u8; 4];
            self.fill_bytes(&mut b);
            u32::from_le_bytes(b)
        }
        fn next_u64(&mut self) -> u64 {
            let mut b = [0u8; 8];
            self.fill_bytes(&mut b);
            u64::from_le_bytes(b)
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for d in dest.iter_mut() {
                if self.buf.is_empty() {
                    self.counter += 1;
                    self.buf = Sha256::digest(self.counter.to_le_bytes()).to_vec();
                }
                *d = self.buf.pop().unwrap();
            }
        }
        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rsa::rand_core::Error> {
            self.fill_bytes(dest);
            Ok(())
        }
    }
    impl rsa::rand_core::CryptoRng for CounterRng {}

    pub(crate) fn rsa_test_rng() -> CounterRng {
        CounterRng {
            counter: 0,
            buf: Vec::new(),
        }
    }

    pub(crate) fn rsa_to_pkcs8_pem(sk: &rsa::RsaPrivateKey) -> String {
        use rsa::pkcs8::EncodePrivateKey;
        sk.to_pkcs8_pem(rsa::pkcs8::LineEnding::LF)
            .unwrap()
            .to_string()
    }

    const NONCE: &str = "931d8dd0add203ac3d8b4fbde75e115278eefcdceac5b87671a748f32364dfcb";

    // NVIDIA's own bytes (tests/VENDORED.md), and the time of the production JWKS GET
    // (2026-10-18 10:03:39 UTC) that holds both leaves: every window is open then.
    const JWKS_GET_AT: u64 = 1_792_317_819;
    const I004: &[u8] = include_bytes!("../tests/nvidia_nras_gpu_intermediate_004.der");
    const I005: &[u8] = include_bytes!("../tests/nvidia_nras_gpu_intermediate_005.der");
    const L004: &[u8] = include_bytes!("../tests/nvidia_nras_gpu_leaf_under_004_2026-10-18.der");
    const L005: &[u8] = include_bytes!("../tests/nvidia_nras_gpu_leaf_under_005_2026-10-18.der");

    /// NVIDIA's two real leaves, each with its intermediate and its name.
    pub(crate) const REAL_LEAVES: [(&[u8], &[u8], &str); 2] =
        [(L004, I004, "004"), (L005, I005, "005")];

    /// The key `leaf` gives under `inter` and the production pins, at the
    /// time the real ones were fetched: what `pinned_leaf_key` says.
    pub(crate) fn real_leaf_key(leaf: &[u8], inter: &[u8]) -> Result<Vec<u8>, NrasRejected> {
        pinned_leaf_key(
            "real",
            &x5c(leaf, inter),
            &NVIDIA_NRAS_GPU_INTERMEDIATE_PINS,
            JWKS_GET_AT,
        )
        .map(|k| k.to_sec1_bytes().to_vec())
    }

    /// A JWKS entry's `x5c`: leaf, then intermediate.
    fn x5c(leaf: &[u8], inter: &[u8]) -> Vec<serde_json::Value> {
        vec![
            serde_json::json!(b64std(leaf)),
            serde_json::json!(b64std(inter)),
        ]
    }

    /// A correct token verifies under its pinned intermediate.
    #[test]
    fn a_correct_token_verifies_under_the_pinned_intermediate() {
        let fx = build_fixture(true, NONCE, 2_000);
        let v = verify_nras_token(&fx.token, &fx.jwks, &fx.nonce_hex, 1_500, &fx.pins)
            .expect("verifies");
        assert!(v.overall);
        assert!(!v.limitations.is_empty());
    }

    /// SECURITY: NVIDIA gives the NRAS signing leaf about two days
    /// precisely so a leaked key stops working. This token's own `exp` claim
    /// is generous and its signature verifies perfectly — but the leaf that
    /// signed it retired in 2020. A failure here means the only freshness
    /// check left is `exp`, a claim signed by the very leaf in question:
    /// whoever holds the leaked key simply writes a later one.
    #[test]
    fn a_token_signed_by_a_retired_leaf_is_refused_however_generous_its_exp() {
        // exp in 2096, leaf window closed on 2020-01-01, now is 2026-02-12.
        let fx = build_fixture_with_leaf_window(true, NONCE, 4_000_000_000, (2015, 2020));
        let r = verify_nras_token(&fx.token, &fx.jwks, &fx.nonce_hex, 1_770_000_000, &fx.pins);
        let msg = format!("{:?}", r.expect_err("a retired leaf must be refused"));
        assert!(
            msg.contains("validity window"),
            "the refusal must name the window, got: {msg}"
        );
    }

    /// The same token, pinned to PRODUCTION NVIDIA: refused — a perfect
    /// chain to the wrong issuer is the attack. Both key types.
    #[test]
    fn a_token_from_an_unpinned_issuer_is_refused() {
        for kind in [Inter::Rsa, Inter::P384] {
            let fx = build_fixture_under(kind, true, NONCE, 2_000, (1970, 4096));
            let r = verify_nras_token(
                &fx.token,
                &fx.jwks,
                &fx.nonce_hex,
                1_500,
                &NVIDIA_NRAS_GPU_INTERMEDIATE_PINS,
            );
            assert!(matches!(r, Err(NrasRejected::LeafNotPinnedIssuer(_))));
        }
    }

    /// Tampered payload → signature refused; wrong nonce → claims refused;
    /// expired → claims refused; overall=false → claims refused.
    #[test]
    fn every_broken_arrow_is_its_own_refusal() {
        let fx = build_fixture(true, NONCE, 2_000);
        // tamper the payload segment
        let mut parts: Vec<&str> = fx.token.split('.').collect();
        let mut claims: serde_json::Value =
            serde_json::from_slice(&b64url(parts[1]).unwrap()).unwrap();
        claims["x-nvidia-overall-att-result"] = serde_json::json!(true);
        claims["hwmodel"] = serde_json::json!("FORGED");
        let forged = b64url_enc(serde_json::to_vec(&claims).unwrap().as_slice());
        parts[1] = &forged;
        let tampered = parts.join(".");
        assert!(matches!(
            verify_nras_token(&tampered, &fx.jwks, &fx.nonce_hex, 1_500, &fx.pins),
            Err(NrasRejected::JwtSignature)
        ));
        // wrong nonce
        assert!(matches!(
            verify_nras_token(&fx.token, &fx.jwks, "deadbeef", 1_500, &fx.pins),
            Err(NrasRejected::Claims(_))
        ));
        // expired
        assert!(matches!(
            verify_nras_token(&fx.token, &fx.jwks, &fx.nonce_hex, 9_000, &fx.pins),
            Err(NrasRejected::Claims(_))
        ));
        // overall = false
        let bad = build_fixture(false, NONCE, 2_000);
        assert!(matches!(
            verify_nras_token(&bad.token, &bad.jwks, &bad.nonce_hex, 1_500, &bad.pins),
            Err(NrasRejected::Claims(_))
        ));
    }

    /// SECURITY: a failure means extensions the signer says must be understood
    /// are ignored, on a token RFC 7515, 4.1.11, calls invalid. Listed or not.
    #[test]
    fn a_token_whose_header_has_crit_is_refused_whatever_it_lists() {
        let fx = build_fixture(true, NONCE, 2_000);
        let payload = fx.token.split('.').nth(1).unwrap();
        let claims: serde_json::Value = serde_json::from_slice(&b64url(payload).unwrap()).unwrap();
        let kid = "nv-eat-kid-test-1";
        let plain =
            (fx.sign_with_header)(&serde_json::json!({"alg": "ES384", "kid": kid}), &claims);
        verify_nras_token(&plain, &fx.jwks, NONCE, 1_500, &fx.pins).expect("the control");
        for crit in [
            serde_json::json!(["x-ext"]),
            serde_json::json!([]),
            serde_json::json!("x-ext"),
            serde_json::Value::Null,
        ] {
            let header = serde_json::json!({"alg": "ES384", "kid": kid, "crit": crit, "x-ext": 1});
            let token = (fx.sign_with_header)(&header, &claims);
            let r = verify_nras_token(&token, &fx.jwks, NONCE, 1_500, &fx.pins);
            assert!(
                matches!(&r, Err(NrasRejected::Header(why)) if why.contains("crit")),
                "{crit}: {r:?}"
            );
        }
    }

    /// SECURITY: a failure means a token that answers no nonce is accepted
    /// as fresh by a caller whose nonce came out empty: no anti-replay.
    #[test]
    fn a_token_with_no_nonce_answers_no_caller_not_even_an_empty_nonce() {
        let fx = build_fixture(true, NONCE, 2_000);
        let payload = fx.token.split('.').nth(1).unwrap();
        let claims: serde_json::Value = serde_json::from_slice(&b64url(payload).unwrap()).unwrap();
        let mut absent = claims.clone();
        absent.as_object_mut().unwrap().remove("eat_nonce");
        let mut not_text = claims.clone();
        not_text["eat_nonce"] = serde_json::json!([NONCE]);
        let mut empty = claims;
        empty["eat_nonce"] = serde_json::json!("");
        for (what, claims) in [("absent", absent), ("array", not_text), ("empty", empty)] {
            let token = (fx.sign)(&claims);
            for expected in ["", NONCE] {
                let r = verify_nras_token(&token, &fx.jwks, expected, 1_500, &fx.pins);
                assert!(
                    matches!(&r, Err(NrasRejected::Claims(why)) if why.contains("eat_nonce")),
                    "{what} against {expected:?}: {r:?}"
                );
            }
        }
    }

    // ── Intermediate 005 (P-384) next to 004 (RSA) ────────────────

    /// Case (a) of the 005 switch: NRAS signs under a P-384 intermediate. A failure
    /// means every genuine token refuses the day NRAS switches to 005.
    #[test]
    fn an_es384_token_under_a_pinned_p384_intermediate_is_accepted() {
        let fx = fixture_for_composite_p384();
        assert_eq!(
            fx.pins[0].leaf_signature,
            NrasLeafSignature::EcdsaP384Sha384
        );
        let v = verify_nras_token(&fx.token, &fx.jwks, NONCE, 1_500, &fx.pins).expect("verifies");
        assert!(v.overall);
        let v = attest(&fx, &fx.per_gpu, &fx.gpu_serials).expect("per-GPU tokens verify");
        assert_eq!(v.gpu_ueids, [GPU_SERIAL_DECIMAL]);
    }

    /// (e): the RSA intermediate keeps working, under a pin set shaped like
    /// production (one RSA pin, one P-384 pin), and so does the P-384 one.
    #[test]
    fn rsa_and_p384_intermediates_both_verify_under_one_pin_set() {
        let rsa = build_fixture(true, NONCE, 2_000);
        let ec = fixture_for_composite_p384();
        let pins = [rsa.pins.clone(), ec.pins.clone()].concat();
        for fx in [&rsa, &ec] {
            verify_nras_attestation(
                &fx.token,
                &fx.per_gpu,
                &fx.gpu_serials,
                &fx.jwks,
                NONCE,
                1_500,
                &pins,
                &[],
                &NrasOcspPolicy::production(),
            )
            .expect("verifies");
        }
    }

    /// (b): a chain whose intermediate is no pin's — another P-384 CA, the
    /// production pins, no pin at all — and a pinned intermediate presented
    /// next to a leaf it did not sign.
    #[test]
    fn an_intermediate_that_matches_no_pin_is_refused() {
        let fx = fixture_for_composite_p384();
        let other = fixture_for_composite_p384();
        for pins in [
            other.pins.clone(),
            NVIDIA_NRAS_GPU_INTERMEDIATE_PINS.to_vec(),
            Vec::new(),
        ] {
            assert!(matches!(
                verify_nras_token(&fx.token, &fx.jwks, NONCE, 1_500, &pins),
                Err(NrasRejected::LeafNotPinnedIssuer(_))
            ));
        }
        let mut swapped = fx.jwks.clone();
        swapped["keys"][0]["x5c"][1] = other.jwks["keys"][0]["x5c"][1].clone();
        assert!(matches!(
            verify_nras_token(&fx.token, &swapped, NONCE, 1_500, &other.pins),
            Err(NrasRejected::LeafNotPinnedIssuer(_))
        ));
    }

    fn flip(b: &mut [u8], at: usize) {
        b[at] ^= 0x01;
    }

    /// (c): one byte changed anywhere that is signed — the JWT signature,
    /// its payload, the leaf certificate's signature, the leaf's key inside
    /// its TBS, the intermediate's key — is refused under a P-384 chain.
    #[test]
    fn one_tampered_byte_is_refused_under_a_p384_chain() {
        let fx = fixture_for_composite_p384();
        let verify = |token: &str, jwks: &serde_json::Value| {
            verify_nras_token(token, jwks, NONCE, 1_500, &fx.pins)
        };
        let parts: Vec<&str> = fx.token.split('.').collect();
        for seg in [1usize, 2] {
            let mut raw = b64url(parts[seg]).unwrap();
            let at = raw.len() / 2;
            flip(&mut raw, at);
            let mut p = parts.clone();
            let enc = b64url_enc(&raw);
            p[seg] = &enc;
            assert_eq!(
                verify(&p.join("."), &fx.jwks).unwrap_err(),
                NrasRejected::JwtSignature,
                "segment {seg}"
            );
        }
        let cert = |i: usize| {
            base64::engine::general_purpose::STANDARD
                .decode(fx.jwks["keys"][0]["x5c"][i].as_str().unwrap())
                .unwrap()
        };
        let with_cert = |i: usize, der: &[u8]| {
            let mut j = fx.jwks.clone();
            j["keys"][0]["x5c"][i] = b64std(der).into();
            j
        };
        // Last byte of the leaf DER: inside its signature.
        let mut leaf = cert(0);
        let n = leaf.len() - 1;
        flip(&mut leaf, n);
        assert!(matches!(
            verify(&fx.token, &with_cert(0, &leaf)),
            Err(NrasRejected::LeafNotPinnedIssuer(_))
        ));
        // Last byte of each certificate's public key: inside the signed TBS.
        for i in [0usize, 1] {
            let mut der = cert(i);
            let spki = X509Certificate::from_der(&der)
                .unwrap()
                .1
                .public_key()
                .raw
                .to_vec();
            let at = der.windows(spki.len()).position(|w| w == spki).unwrap() + spki.len() - 1;
            flip(&mut der, at);
            assert!(
                matches!(
                    verify(&fx.token, &with_cert(i, &der)),
                    Err(NrasRejected::LeafNotPinnedIssuer(_))
                ),
                "x5c[{i}]"
            );
        }
        assert!(verify(&fx.token, &fx.jwks).is_ok(), "the untouched token");
    }

    /// The outer `signatureAlgorithm` of a certificate is not signed: whoever
    /// relays the JWKS can rewrite it. Its last OID arc, in place.
    fn relabel_outer_signature_oid(der: &[u8], from: &[u8], last_arc: u8) -> Vec<u8> {
        let at = der
            .windows(from.len())
            .rposition(|w| w == from)
            .expect("outer signatureAlgorithm");
        let mut out = der.to_vec();
        out[at + from.len() - 1] = last_arc;
        out
    }

    /// (d), the JWT: ES384 and nothing else, and ES384 only under a P-384
    /// leaf key. RS256 against the EC leaf, `none`, HS*, other ES/PS sizes,
    /// and an ES384 header whose JWKS leaf holds an RSA key are refused.
    #[test]
    fn jwt_alg_confusion_is_refused_both_ways() {
        let fx = fixture_for_composite_p384();
        let parts: Vec<&str> = fx.token.split('.').collect();
        let kid = "nv-eat-kid-test-1";
        for alg in [
            "RS256", "RS384", "PS384", "HS256", "HS384", "none", "ES256", "ES512", "es384",
        ] {
            let h = b64url_enc(
                &serde_json::to_vec(&serde_json::json!({"alg": alg, "kid": kid})).unwrap(),
            );
            let sig = if alg == "none" { "" } else { parts[2] };
            let token = format!("{h}.{}.{sig}", parts[1]);
            assert!(
                matches!(
                    verify_nras_token(&token, &fx.jwks, NONCE, 1_500, &fx.pins),
                    Err(NrasRejected::Header(_))
                ),
                "{alg}"
            );
        }
        // An RSA leaf key, certified by a pinned intermediate of each type.
        // Another counter than `rsa_test_rng`: a key distinct from the RSA intermediate's.
        let mut rng = CounterRng {
            counter: 1_000_000,
            buf: Vec::new(),
        };
        let leaf_rsa = rsa::RsaPrivateKey::new(&mut rng, 2048).unwrap();
        let leaf_kp = KeyPair::from_pkcs8_pem_and_sign_algo(
            &rsa_to_pkcs8_pem(&leaf_rsa),
            &rcgen::PKCS_RSA_SHA256,
        )
        .unwrap();
        for kind in [Inter::P384, Inter::Rsa] {
            let (inter_der, issuer, pin) = intermediate(kind);
            let mut params = CertificateParams::new(vec![]).unwrap();
            params.not_before = rcgen::date_time_ymd(1970, 1, 1);
            params.not_after = rcgen::date_time_ymd(4096, 1, 1);
            let leaf = params.signed_by(&leaf_kp, &issuer).unwrap();
            let jwks = serde_json::json!({"keys": [{
                "kty": "RSA", "kid": kid,
                "x5c": [b64std(leaf.der()), b64std(&inter_der)],
            }]});
            let h = b64url_enc(
                &serde_json::to_vec(&serde_json::json!({"alg": "ES384", "kid": kid})).unwrap(),
            );
            let token = format!("{h}.{}.{}", parts[1], b64url_enc(&[7u8; 96]));
            let err = verify_nras_token(&token, &jwks, NONCE, 1_500, &[pin]).unwrap_err();
            assert!(
                matches!(&err, NrasRejected::BadJwksKey(_, why) if why.contains("P-384")),
                "{err:?}"
            );
        }
    }

    /// (d), the certificate: a pin's key type fixes the leaf signature. An
    /// ECDSA chain under a pin that says RSA, an RSA chain under a pin that
    /// says P-384, and a leaf that relabels its (unsigned) outer algorithm
    /// are refused, though each signature is genuine.
    #[test]
    fn certificate_alg_confusion_is_refused_both_ways() {
        let ec = fixture_for_composite_p384();
        let rsa = build_fixture(true, NONCE, 2_000);
        let retyped = |fx: &Fixture, t: NrasLeafSignature| {
            vec![NrasIntermediatePin {
                cert_sha256_hex: fx.pins[0].cert_sha256_hex.clone(),
                spki_sha256_hex: fx.pins[0].spki_sha256_hex.clone(),
                leaf_signature: t,
            }]
        };
        for (fx, wrong) in [
            (&ec, NrasLeafSignature::RsaPkcs1Sha256),
            (&rsa, NrasLeafSignature::EcdsaP384Sha384),
        ] {
            assert!(matches!(
                verify_nras_token(&fx.token, &fx.jwks, NONCE, 1_500, &retyped(fx, wrong)),
                Err(NrasRejected::LeafNotPinnedIssuer(_))
            ));
        }
        // ecdsa-with-SHA384 1.2.840.10045.4.3.3 -> ecdsa-with-SHA256 (.2);
        // sha256WithRSAEncryption 1.2.840.113549.1.1.11 -> sha384WithRSA (.12).
        const ECDSA_SHA384: &[u8] = &[0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x03];
        const RSA_SHA256: &[u8] = &[
            0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b,
        ];
        for (fx, oid, arc) in [(&ec, ECDSA_SHA384, 0x02u8), (&rsa, RSA_SHA256, 0x0c)] {
            let leaf = base64::engine::general_purpose::STANDARD
                .decode(fx.jwks["keys"][0]["x5c"][0].as_str().unwrap())
                .unwrap();
            let mut jwks = fx.jwks.clone();
            jwks["keys"][0]["x5c"][0] =
                b64std(&relabel_outer_signature_oid(&leaf, oid, arc)).into();
            assert!(matches!(
                verify_nras_token(&fx.token, &jwks, NONCE, 1_500, &fx.pins),
                Err(NrasRejected::LeafNotPinnedIssuer(_))
            ));
        }
    }

    /// The production pins are 004 (RSA) and 005 (P-384), exactly, each
    /// matching its vendored NVIDIA certificate through the verifier's own
    /// digest, each bound to the key type that certificate carries.
    #[test]
    fn the_production_pins_match_nvidias_published_intermediates() {
        assert_eq!(
            sha256_hex(I004),
            NVIDIA_NRAS_GPU_INTERMEDIATE_004_CERT_SHA256_HEX
        );
        assert_eq!(
            sha256_hex(I005),
            NVIDIA_NRAS_GPU_INTERMEDIATE_005_CERT_SHA256_HEX
        );
        assert_eq!(
            spki_hex(I004),
            NVIDIA_NRAS_GPU_INTERMEDIATE_004_SPKI_SHA256_HEX
        );
        assert_eq!(
            spki_hex(I005),
            NVIDIA_NRAS_GPU_INTERMEDIATE_005_SPKI_SHA256_HEX
        );
        let raw = |der: &[u8]| {
            X509Certificate::from_der(der)
                .unwrap()
                .1
                .public_key()
                .raw
                .to_vec()
        };
        assert!(
            RsaPublicKey::from_public_key_der(&raw(I004)).is_ok(),
            "004 is RSA"
        );
        assert!(
            <p384::ecdsa::VerifyingKey as p384::pkcs8::DecodePublicKey>::from_public_key_der(&raw(
                I005
            ))
            .is_ok(),
            "005 is P-384"
        );
        let pins = NVIDIA_NRAS_GPU_INTERMEDIATE_PINS;
        assert_eq!(pins.len(), 2, "no pin beyond 004 and 005");
        assert_eq!(
            pins[0].cert_sha256_hex,
            NVIDIA_NRAS_GPU_INTERMEDIATE_004_CERT_SHA256_HEX
        );
        assert_eq!(
            pins[0].spki_sha256_hex,
            NVIDIA_NRAS_GPU_INTERMEDIATE_004_SPKI_SHA256_HEX
        );
        assert_eq!(pins[0].leaf_signature, NrasLeafSignature::RsaPkcs1Sha256);
        assert_eq!(
            pins[1].cert_sha256_hex,
            NVIDIA_NRAS_GPU_INTERMEDIATE_005_CERT_SHA256_HEX
        );
        assert_eq!(
            pins[1].spki_sha256_hex,
            NVIDIA_NRAS_GPU_INTERMEDIATE_005_SPKI_SHA256_HEX
        );
        assert_eq!(pins[1].leaf_signature, NrasLeafSignature::EcdsaP384Sha384);
    }

    /// NVIDIA's own bytes: one leaf under each intermediate, from the
    /// production JWKS of 2026-10-18 10:03:39 UTC (both windows hold then).
    /// Each verifies under the production pins and under its own pin alone,
    /// never under the other's, nor presented with the other intermediate.
    #[test]
    fn nvidias_real_leaves_verify_under_their_own_production_pin_only() {
        let all = NVIDIA_NRAS_GPU_INTERMEDIATE_PINS;
        let (p004, p005) = (all[..1].to_vec(), all[1..].to_vec());
        for (leaf, inter, own, other, other_inter, name) in [
            (L004, I004, &p004, &p005, I005, "004"),
            (L005, I005, &p005, &p004, I004, "005"),
        ] {
            pinned_leaf_key("real", &x5c(leaf, inter), &all, JWKS_GET_AT)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            pinned_leaf_key("real", &x5c(leaf, inter), own, JWKS_GET_AT)
                .unwrap_or_else(|e| panic!("{name} alone: {e}"));
            assert!(
                matches!(
                    pinned_leaf_key("real", &x5c(leaf, inter), other, JWKS_GET_AT),
                    Err(NrasRejected::LeafNotPinnedIssuer(_))
                ),
                "{name} under the other pin"
            );
            assert!(
                matches!(
                    pinned_leaf_key("real", &x5c(leaf, other_inter), &all, JWKS_GET_AT),
                    Err(NrasRejected::LeafNotPinnedIssuer(_))
                ),
                "{name} presented with the other intermediate"
            );
        }
    }

    /// RFC 5280, section 4.1.1.2: the leaf's outer `signatureAlgorithm` MUST
    /// be the one inside its signed part, parameters included. NVIDIA's two
    /// real leaves, their outer algorithm written the other way (a NULL
    /// dropped under 004, a NULL added under 005): the same OID, bytes no
    /// signature covers.
    #[test]
    fn a_real_leaf_whose_outer_algorithm_is_written_otherwise_is_refused() {
        use crate::nras_ocsp::tests::{der_oid, null, seq, with_outer_algorithm};
        const SHA256_RSA: &[u64] = &[1, 2, 840, 113549, 1, 1, 11];
        const ECDSA_SHA384: &[u64] = &[1, 2, 840, 10045, 4, 3, 3];
        let all = NVIDIA_NRAS_GPU_INTERMEDIATE_PINS;
        let mut not_refused = Vec::new();
        for (leaf, inter, outer, name) in [
            (L004, I004, seq(&[&der_oid(SHA256_RSA)]), "004"),
            (L005, I005, seq(&[&der_oid(ECDSA_SHA384), &null()]), "005"),
        ] {
            pinned_leaf_key("real", &x5c(leaf, inter), &all, JWKS_GET_AT)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let changed = with_outer_algorithm(leaf, &outer);
            assert_ne!(changed, leaf, "{name}");
            match pinned_leaf_key("real", &x5c(&changed, inter), &all, JWKS_GET_AT) {
                Err(NrasRejected::LeafNotPinnedIssuer(why))
                    if why.contains("algorithms differ") => {}
                other => not_refused.push(format!("{name}: {:?}", other.map(|_| "a key"))),
            }
        }
        assert!(not_refused.is_empty(), "{not_refused:#?}");
    }

    /// SECURITY: under 004 the leaf's RSA signature is verified,
    /// not only the algorithm it names. A failure means a leaf of anyone's
    /// making, naming sha256WithRSAEncryption with 256 bytes of noise for a
    /// signature, passes next to NVIDIA's real 004, and its key signs as NRAS.
    #[test]
    fn a_leaf_naming_rsa_with_a_garbage_signature_is_refused_under_the_real_004() {
        use x509_parser::oid_registry::OID_PKCS1_SHA256WITHRSA;
        let (_, not_nvidia, _) = intermediate(Inter::Rsa);
        let leaf_kp = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P384_SHA384).unwrap();
        let mut params = CertificateParams::new(vec![]).unwrap();
        params.not_before = rcgen::date_time_ymd(1970, 1, 1);
        params.not_after = rcgen::date_time_ymd(4096, 1, 1);
        let mut leaf = params
            .signed_by(&leaf_kp, &not_nvidia)
            .unwrap()
            .der()
            .to_vec();
        let sig_len = {
            let (_, c) = X509Certificate::from_der(&leaf).unwrap();
            assert_eq!(c.signature_algorithm.algorithm, OID_PKCS1_SHA256WITHRSA);
            assert_eq!(
                c.tbs_certificate.signature.algorithm,
                OID_PKCS1_SHA256WITHRSA
            );
            assert!(
                leaf.ends_with(&c.signature_value.data),
                "the signature ends the DER"
            );
            c.signature_value.data.len()
        };
        assert_eq!(sig_len, 256, "RSA-2048, the size of 004's key");
        // 0x42.. is below any 2048-bit modulus: what refuses is the RSA check itself.
        let n = leaf.len();
        leaf[n - 256..].fill(0x42);

        let kid = "not-nvidia";
        let jwks = serde_json::json!({"keys": [{
            "kty": "EC", "crv": "P-384", "kid": kid, "x5c": x5c(&leaf, I004),
        }]});
        let token = es384_signer(&leaf_kp, kid)(&serde_json::json!({
            "iss": "https://nras.attestation.nvidia.com",
            "iat": JWKS_GET_AT,
            "exp": JWKS_GET_AT + 3_600,
            "eat_nonce": NONCE,
            "x-nvidia-overall-att-result": true,
        }));
        let r = verify_nras_token(
            &token,
            &jwks,
            NONCE,
            JWKS_GET_AT,
            &NVIDIA_NRAS_GPU_INTERMEDIATE_PINS,
        );
        assert!(
            matches!(&r, Err(NrasRejected::LeafNotPinnedIssuer(why)) if why == "leaf signature does not verify"),
            "{r:?}"
        );
    }

    /// SECURITY: x5c[1] must BE the pinned certificate, not only
    /// carry its key. Its own signature, by CA 001, is not checked (CA 001 is
    /// not published). A failure means whoever serves the JWKS (on the relay
    /// path, the TD's document) re-wraps 004's or 005's key in a certificate of
    /// their making, and the intermediate's validity window becomes theirs.
    #[test]
    fn a_pinned_key_in_a_certificate_that_is_not_the_pinned_one_is_refused() {
        fn cert(der: &[u8]) -> X509Certificate<'_> {
            X509Certificate::from_der(der).unwrap().1
        }
        let not_after = |der: &[u8]| cert(der).validity().not_after.timestamp();
        let mut not_refused = Vec::new();
        for (kind, inter, leaf, name) in [
            (Inter::Rsa, I004, L004, "004"),
            (Inter::P384, I005, L005, "005"),
        ] {
            // A CA certificate of our own making (1970 to 4096), NVIDIA's SPKI spliced in.
            let (mut forged, _, _) = intermediate(kind);
            let ours = cert(&forged).public_key().raw.to_vec();
            let theirs = cert(inter).public_key().raw.to_vec();
            assert_eq!(
                ours.len(),
                theirs.len(),
                "{name}: one key type, one SPKI size"
            );
            let at = forged.windows(ours.len()).position(|w| w == ours).unwrap();
            forged[at..at + ours.len()].copy_from_slice(&theirs);
            assert_eq!(spki_hex(&forged), spki_hex(inter), "{name}");
            assert!(
                not_after(&forged) > not_after(inter),
                "{name}: the forged window outlives NVIDIA's"
            );
            let r = pinned_leaf_key(
                "forged",
                &x5c(leaf, &forged),
                &NVIDIA_NRAS_GPU_INTERMEDIATE_PINS,
                JWKS_GET_AT,
            );
            if !matches!(&r, Err(NrasRejected::LeafNotPinnedIssuer(why)) if why.contains("not the pinned certificate"))
            {
                not_refused.push(format!("{name}: {r:?}"));
            }
        }
        assert!(not_refused.is_empty(), "{not_refused:#?}");
    }

    fn attest_with(
        fx: &Fixture,
        responses: &[Vec<u8>],
        policy: &NrasOcspPolicy,
    ) -> Result<VerifiedGpu, NrasRejected> {
        verify_nras_attestation(
            &fx.token,
            &fx.per_gpu,
            &fx.gpu_serials,
            &fx.jwks,
            &fx.nonce_hex,
            1_500,
            &fx.pins,
            responses,
            policy,
        )
    }

    /// A good response about the signing leaf, stapled: `required` admits
    /// and the standing travels with the verdict; the intermediate is said
    /// unverifiable (CA 001 not pinned).
    #[test]
    fn a_stapled_good_ocsp_response_is_admitted_under_required_and_named() {
        use crate::nras_ocsp::tests::{respond_with, Response};
        use crate::nras_ocsp::OcspStatus;
        let fx = fixture_for_composite();
        let good = respond_with(
            &Response::good_at(1_500),
            &fx.leaf_der,
            &fx.inter_der,
            &fx.inter_signer,
        );
        let v = attest_with(&fx, &[good], &NrasOcspPolicy::required()).expect("admitted");
        assert_eq!(v.revocation.len(), 2, "{:?}", v.revocation);
        assert_eq!(v.revocation[0].what, "x5c leaf");
        assert!(
            matches!(v.revocation[0].status, OcspStatus::Good { .. }),
            "{:?}",
            v.revocation
        );
        assert_eq!(v.revocation[1].status, OcspStatus::IssuerNotPublished);
    }

    /// SECURITY: a revoked signing leaf is a key NVIDIA no longer stands
    /// behind; its tokens are refused under both policies, by name.
    #[test]
    fn a_revoked_signing_leaf_refuses_the_attestation_under_both_policies() {
        use crate::nras_ocsp::tests::{respond_with, Response, Status};
        let fx = fixture_for_composite();
        let revoked = respond_with(
            &Response {
                status: Status::Revoked(1_000, Some(1)),
                ..Response::good_at(1_500)
            },
            &fx.leaf_der,
            &fx.inter_der,
            &fx.inter_signer,
        );
        for policy in [NrasOcspPolicy::required(), NrasOcspPolicy::production()] {
            let e = attest_with(&fx, std::slice::from_ref(&revoked), &policy).expect_err("revoked");
            assert!(
                matches!(&e, NrasRejected::SigningChainRevoked(why)
                    if why.contains("REVOKED") && why.contains("x5c leaf")),
                "{policy:?}: {e}"
            );
        }
    }

    /// Nothing stapled: `required` refuses by name; `production`
    /// (transparent) admits and says so in the standing every verdict carries.
    #[test]
    fn without_stapled_responses_required_refuses_and_production_names_it() {
        use crate::nras_ocsp::OcspStatus;
        let fx = fixture_for_composite();
        let e = attest_with(&fx, &[], &NrasOcspPolicy::required()).expect_err("nothing stapled");
        assert!(
            matches!(&e, NrasRejected::SigningChainRevocationUnverified(why)
                if why.contains("no response stapled")),
            "{e}"
        );
        let v = attest(&fx, &fx.per_gpu, &fx.gpu_serials).expect("transparent admits");
        assert_eq!(v.revocation[0].status, OcspStatus::NoResponse);
        assert!(v.revocation[0].to_string().contains("no response stapled"));
    }
}
