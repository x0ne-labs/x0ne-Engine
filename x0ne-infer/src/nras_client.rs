//! The NRAS client: build the v4 attest request, split the answer.
//!
//! The wire format is nvtrust's, copied exactly (attest_gpu_remote.py,
//! build_payload): POST JSON to `/v4/attest/gpu` —
//!
//! ```json
//! { "nonce": "<64-hex>",
//!   "evidence_list": [ { "evidence": "<b64 attestation report>",
//!                        "certificate": "<b64 GPU cert chain>" } ],
//!   "arch": "HOPPER",
//!   "claims_version": "3.0" }
//! ```
//!
//! — and the answer carries the EAT: the overall JWT plus one JWT per GPU
//! (`[["JWT", <overall>], {"GPU-0": <token>, …}]` in the SDK's unified EAT
//! shape). This module is PURE: the payload builder and the answer splitter
//! are functions over values, tested here; the HTTP POST itself is a thin
//! `reqwest` call in `main` and is exercised only on real hardware —
//! never simulated.
//!
//! Evidence collection (NVML `nvmlDeviceGetConfComputeGpuAttestationReport` /
//! `...GpuCertificate`) requires a CC GPU and is provided by the `nvml`
//! feature later; the types here are what it must produce.

use serde::{Deserialize, Serialize};

/// One GPU's evidence, as NRAS wants it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuEvidence {
    /// Base64 of the raw attestation report (SPDM measurements), which
    /// echoes our 32-byte GPU nonce.
    pub evidence: String,
    /// Base64 of the GPU attestation certificate chain.
    pub certificate: String,
}

/// The NRAS v4 request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestRequest {
    pub nonce: String,
    pub evidence_list: Vec<GpuEvidence>,
    pub arch: String,
    pub claims_version: String,
}

/// Production endpoint. Injectable everywhere it is used.
pub const NRAS_URL: &str = "https://nras.attestation.nvidia.com/v4/attest/gpu";
pub const NRAS_JWKS_URL: &str = "https://nras.attestation.nvidia.com/.well-known/jwks.json";

/// Build the request. Refuses an empty evidence list and a nonce that is not
/// 64 hex chars — a malformed request must fail HERE, with our words, not at
/// NVIDIA with theirs.
pub fn build_attest_request(
    nonce_hex: &str,
    evidence: Vec<GpuEvidence>,
    arch: &str,
) -> anyhow::Result<AttestRequest> {
    if nonce_hex.len() != 64 || !nonce_hex.chars().all(|c| c.is_ascii_hexdigit()) {
        anyhow::bail!("GPU nonce must be 64 hex chars, got {}", nonce_hex.len());
    }
    if evidence.is_empty() {
        anyhow::bail!("no GPU evidence collected");
    }
    Ok(AttestRequest {
        nonce: nonce_hex.to_lowercase(),
        evidence_list: evidence,
        arch: arch.to_string(),
        claims_version: "3.0".into(),
    })
}

/// Split the NRAS answer into (overall JWT, per-GPU JWTs). Accepts the
/// unified-EAT array shape the SDK documents; anything else is refused with
/// the shape named.
pub fn split_attest_response(
    body: &serde_json::Value,
) -> anyhow::Result<(String, Vec<(String, String)>)> {
    let arr = body
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("NRAS answer is not the unified-EAT array"))?;
    if arr.len() != 2 {
        anyhow::bail!("NRAS answer has {} elements, expected 2", arr.len());
    }
    let overall = arr[0]
        .as_array()
        .and_then(|pair| {
            (pair.len() == 2 && pair[0] == "JWT")
                .then(|| pair[1].as_str())
                .flatten()
        })
        .ok_or_else(|| anyhow::anyhow!("first element is not [\"JWT\", <token>]"))?
        .to_string();
    let per_gpu = arr[1]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("second element is not the per-GPU map"))?
        .iter()
        .map(|(gpu, v)| {
            v.as_str()
                .map(|t| (gpu.clone(), t.to_string()))
                .ok_or_else(|| anyhow::anyhow!("{gpu}: token is not a string"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    if per_gpu.is_empty() {
        anyhow::bail!("NRAS answered with zero GPUs");
    }
    Ok((overall, per_gpu))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The request matches nvtrust's shape byte-for-byte in the fields NRAS
    /// reads; malformed inputs are refused with our words.
    #[test]
    fn the_request_matches_the_reference_shape_and_refuses_malformed_input() {
        let req = build_attest_request(
            &"ab".repeat(32),
            vec![GpuEvidence {
                evidence: "ZXY=".into(),
                certificate: "Y2VydA==".into(),
            }],
            "HOPPER",
        )
        .expect("builds");
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["arch"], "HOPPER");
        assert_eq!(json["claims_version"], "3.0");
        assert_eq!(json["evidence_list"][0]["evidence"], "ZXY=");
        assert!(
            build_attest_request("deadbeef", vec![], "HOPPER").is_err(),
            "short nonce"
        );
        assert!(
            build_attest_request(&"ab".repeat(32), vec![], "HOPPER").is_err(),
            "no evidence"
        );
    }

    /// The unified-EAT answer splits into overall + per-GPU; every other
    /// shape is refused.
    #[test]
    fn the_answer_splits_or_is_refused_with_the_shape_named() {
        let ok = serde_json::json!([["JWT", "eyJhb.overall.sig"], {"GPU-0": "eyJhb.gpu0.sig"}]);
        let (overall, gpus) = split_attest_response(&ok).expect("splits");
        assert_eq!(overall, "eyJhb.overall.sig");
        assert_eq!(
            gpus,
            vec![("GPU-0".to_string(), "eyJhb.gpu0.sig".to_string())]
        );
        for bad in [
            serde_json::json!({"token": "x"}),
            serde_json::json!([["JWS", "x"], {}]),
            serde_json::json!([["JWT", "x"], {}]),
            serde_json::json!([["JWT", "x"], {"GPU-0": 5}]),
        ] {
            assert!(split_attest_response(&bad).is_err(), "{bad}");
        }
    }
}
