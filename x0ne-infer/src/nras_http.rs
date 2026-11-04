//! The two HTTP calls to NRAS — POST the evidence, GET the JWKS.
//!
//! Thin on purpose: everything decidable without a network lives in
//! `nras_client` (payload shape, answer splitting) and `nras` (token
//! verification). This module only moves bytes, with the URL injectable so
//! the tests talk to a local mock and real hardware talks to
//! `https://nras.attestation.nvidia.com`. A non-2xx answer is refused with
//! the status and the first bytes of the body — NRAS's words, attributed to
//! NRAS, never mistaken for a verification verdict of ours.

use crate::nras_client::{split_attest_response, AttestRequest};

const BODY_EXCERPT: usize = 500;

/// A key file is one line; anything bigger is not one.
pub const KEY_FILE_CAP: u64 = 4096;
/// The longest key taken; an NGC key is about 70 characters.
pub const MAX_KEY_BYTES: usize = 1024;
/// The shortest key taken: [`NrasKey::redact`] replaces each copy of the key,
/// and a key of 1 to 3 characters masked ordinary text (review of 2026-10-30).
pub const MIN_KEY_BYTES: usize = 16;

/// NRAS refused the key it was sent (401 or 403): a cause of its own, which
/// the next call with the same key meets again (`attest_v2`, exit 103).
#[derive(Debug)]
pub struct KeyRefused(pub String);

impl std::fmt::Display for KeyRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for KeyRefused {}

/// An NVIDIA API key for NRAS. Its Debug says a key is there, never which.
#[derive(Clone)]
pub struct NrasKey(std::sync::Arc<zeroize::Zeroizing<String>>);

impl std::fmt::Debug for NrasKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NrasKey(not printed)")
    }
}

impl NrasKey {
    /// One token of [`MIN_KEY_BYTES`] to [`MAX_KEY_BYTES`] visible ASCII
    /// characters, one trailing newline allowed.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let token = bytes
            .strip_suffix(b"\r\n")
            .or_else(|| bytes.strip_suffix(b"\n"))
            .unwrap_or(bytes);
        if token.is_empty() {
            return Err("the key is empty".into());
        }
        if token.len() < MIN_KEY_BYTES {
            return Err(format!(
                "the key is shorter than {MIN_KEY_BYTES} characters"
            ));
        }
        if token.len() > MAX_KEY_BYTES {
            return Err(format!("the key is longer than {MAX_KEY_BYTES} bytes"));
        }
        if !token.iter().all(|b| (0x21..=0x7e).contains(b)) {
            return Err(
                "the key is not one token of visible ASCII (no space or control character, \
                 one final newline at most)"
                    .into(),
            );
        }
        // Checked ASCII above: the conversion cannot fail.
        let text = String::from_utf8(token.to_vec()).map_err(|_| "not ASCII".to_string())?;
        Ok(Self(std::sync::Arc::new(zeroize::Zeroizing::new(text))))
    }

    /// `--nras-key-file`: a regular file, not a link, of at most
    /// [`KEY_FILE_CAP`] bytes. Errors name the path, never the content.
    pub fn read(path: &std::path::Path) -> Result<Self, String> {
        let shown = path.display().to_string();
        let named = |m: String| {
            if m.contains(&shown) {
                m
            } else {
                format!("{shown}: {m}")
            }
        };
        let bytes = zeroize::Zeroizing::new(
            crate::fsread::read_capped(path, KEY_FILE_CAP).map_err(|e| named(e.to_string()))?,
        );
        Self::parse(&bytes).map_err(named)
    }

    /// `text` with the key, wherever it appears, replaced.
    pub fn redact(&self, text: &str) -> String {
        text.replace(self.0.as_str(), "(key not printed)")
    }
}

/// Every answer here, read as it arrives. The PCS answers kept under
/// `tests/pcs_live` are 6 KB at most (2026-10-24).
const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;

async fn read_body(resp: reqwest::Response, what: &str) -> anyhow::Result<Vec<u8>> {
    crate::http_body::read_capped(resp, MAX_BODY_BYTES)
        .await
        .map_err(|e| anyhow::anyhow!("{what} body: {e}"))
}

fn client() -> anyhow::Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?)
}

/// POST the attest request; return (overall JWT, per-GPU JWTs).
pub async fn attest_gpu(
    url: &str,
    req: &AttestRequest,
    key: Option<&NrasKey>,
) -> anyhow::Result<(String, Vec<(String, String)>)> {
    let mut post = client()?.post(url).json(req);
    // `bearer_auth` marks the header sensitive: reqwest's Debug hides it.
    if let Some(key) = key {
        post = post.bearer_auth(key.0.as_str());
    }
    let resp = post
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("POST {url}: {e}"))?;
    let status = resp.status();
    let body = String::from_utf8_lossy(&read_body(resp, "NRAS").await?).into_owned();
    if !status.is_success() {
        // Redacted whole, then cut: a cut key would leave its prefix.
        let body = key.map_or(body.clone(), |k| k.redact(&body));
        let said = format!(
            "NRAS answered {status}: {}",
            body.chars().take(BODY_EXCERPT).collect::<String>()
        );
        if key.is_some() && matches!(status.as_u16(), 401 | 403) {
            return Err(anyhow::Error::new(KeyRefused(said)));
        }
        anyhow::bail!(said);
    }
    let value: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| anyhow::anyhow!("NRAS 2xx but not JSON: {e}"))?;
    split_attest_response(&value)
}

/// GET the JWKS the token will be verified against. An empty key set is
/// refused here: `verify_nras_token` would refuse it too, but with a message
/// about the token, when the problem is the key fetch.
pub async fn fetch_jwks(url: &str) -> anyhow::Result<serde_json::Value> {
    let resp = client()?
        .get(url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("GET {url}: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("JWKS endpoint answered {status}");
    }
    let jwks: serde_json::Value = serde_json::from_slice(&read_body(resp, "JWKS").await?)
        .map_err(|e| anyhow::anyhow!("JWKS is not JSON: {e}"))?;
    match jwks.get("keys").and_then(|k| k.as_array()) {
        Some(keys) if !keys.is_empty() => Ok(jwks),
        Some(_) => anyhow::bail!("JWKS has zero keys"),
        None => anyhow::bail!("JWKS has no \"keys\" array"),
    }
}

/// GET one piece of Intel PCS collateral: the JSON body plus the issuer
/// chain that PCS returns URL-encoded in a response header
/// (`TCB-Info-Issuer-Chain` for tcbInfo, `SGX-Enclave-Identity-Issuer-Chain`
/// for the QE identity — both accepted here, by name).
pub async fn fetch_pcs_collateral(url: &str) -> anyhow::Result<(String, String)> {
    let resp = client()?
        .get(url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("GET {url}: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("PCS answered {status} for {url}");
    }
    let chain_encoded = ["TCB-Info-Issuer-Chain", "SGX-Enclave-Identity-Issuer-Chain"]
        .iter()
        .find_map(|h| resp.headers().get(*h))
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("PCS answer carries no issuer-chain header"))?;
    let chain = percent_decode(&chain_encoded)
        .ok_or_else(|| anyhow::anyhow!("issuer-chain header is not valid percent-encoding"))?;
    let body = String::from_utf8(read_body(resp, "PCS").await?)
        .map_err(|_| anyhow::anyhow!("PCS body is not UTF-8"))?;
    Ok((body, chain))
}

/// GET Intel's PCK CRL (DER), at most 4 MiB.
pub async fn fetch_pcs_der(url: &str) -> anyhow::Result<Vec<u8>> {
    let resp = client()?
        .get(url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("GET {url}: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("PCS answered {status} for {url}");
    }
    read_body(resp, "PCS").await
}

/// Strict percent-decoding: a lone or malformed escape is a refusal, not a
/// passthrough — this decodes certificate PEM, where silent garbage would
/// surface later as an inexplicable chain failure.
fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = s.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else if bytes[i] == b'+' {
            out.push(b' ');
            i += 1;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::{get, post};

    /// A local mock speaking NRAS's unified-EAT shape. This tests OUR
    /// transport (serialization, status handling, splitting) — the real
    /// NRAS answer is only ever exercised on real hardware.
    async fn serve(router: axum::Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn a_unified_eat_answer_is_split_into_overall_and_per_gpu() {
        let base = serve(axum::Router::new().route(
            "/v4/attest/gpu",
            post(|body: axum::Json<serde_json::Value>| async move {
                // The mock checks what NRAS would check first: our shape.
                assert_eq!(body["arch"], "HOPPER");
                assert_eq!(body["nonce"].as_str().unwrap().len(), 64);
                axum::Json(serde_json::json!([
                    ["JWT", "overall.jwt.here"],
                    {"GPU-0": "gpu0.jwt.here"}
                ]))
            }),
        ))
        .await;
        let req = crate::nras_client::build_attest_request(
            &"ab".repeat(32),
            vec![crate::nras_client::GpuEvidence {
                evidence: "ZXY=".into(),
                certificate: "Y2M=".into(),
            }],
            "HOPPER",
        )
        .unwrap();
        let (overall, per_gpu) = attest_gpu(&format!("{base}/v4/attest/gpu"), &req, None)
            .await
            .unwrap();
        assert_eq!(overall, "overall.jwt.here");
        assert_eq!(per_gpu, vec![("GPU-0".into(), "gpu0.jwt.here".into())]);
    }

    /// A non-2xx from NRAS is a refusal that carries NRAS's status and words.
    /// A failure here would let a 4xx flow onward as if it were evidence.
    #[tokio::test]
    async fn a_nras_error_status_is_refused_with_its_words() {
        let base = serve(axum::Router::new().route(
            "/v4/attest/gpu",
            post(|| async { (axum::http::StatusCode::UNPROCESSABLE_ENTITY, "bad evidence") }),
        ))
        .await;
        let req = crate::nras_client::build_attest_request(
            &"ab".repeat(32),
            vec![crate::nras_client::GpuEvidence {
                evidence: "eA==".into(),
                certificate: "eA==".into(),
            }],
            "HOPPER",
        )
        .unwrap();
        let err = attest_gpu(&format!("{base}/v4/attest/gpu"), &req, None)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("422"), "{err}");
        assert!(err.contains("bad evidence"), "{err}");
    }

    /// The PCS fetch returns body AND the URL-decoded issuer chain from the
    /// header — either header name; no header is a refusal (a collateral
    /// with no chain can never verify, better refused with the true cause).
    #[tokio::test]
    async fn pcs_collateral_comes_back_with_its_decoded_issuer_chain() {
        let base = serve(
            axum::Router::new()
                .route(
                    "/tcb",
                    get(|| async {
                        (
                            [(
                                "TCB-Info-Issuer-Chain",
                                "-----BEGIN%20CERTIFICATE-----%0Aabc%0A-----END%20CERTIFICATE-----",
                            )],
                            "{\"tcbInfo\":{}}",
                        )
                    }),
                )
                .route("/bare", get(|| async { "{}" })),
        )
        .await;
        let (body, chain) = fetch_pcs_collateral(&format!("{base}/tcb")).await.unwrap();
        assert_eq!(body, "{\"tcbInfo\":{}}");
        assert!(
            chain.starts_with("-----BEGIN CERTIFICATE-----\nabc"),
            "{chain}"
        );
        let err = fetch_pcs_collateral(&format!("{base}/bare"))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("no issuer-chain header"), "{err}");
    }

    /// An empty or absent key set is refused at fetch time, named as a JWKS
    /// problem — not later, misattributed to the token.
    #[tokio::test]
    async fn an_empty_jwks_is_refused_at_fetch_time() {
        let base = serve(
            axum::Router::new()
                .route(
                    "/empty",
                    get(|| async { axum::Json(serde_json::json!({"keys": []})) }),
                )
                .route(
                    "/absent",
                    get(|| async { axum::Json(serde_json::json!({"nope": 1})) }),
                )
                .route(
                    "/good",
                    get(|| async { axum::Json(serde_json::json!({"keys": [{"kty": "EC"}]})) }),
                ),
        )
        .await;
        assert!(fetch_jwks(&format!("{base}/empty"))
            .await
            .unwrap_err()
            .to_string()
            .contains("zero keys"));
        assert!(fetch_jwks(&format!("{base}/absent"))
            .await
            .unwrap_err()
            .to_string()
            .contains("no \"keys\""));
        assert!(fetch_jwks(&format!("{base}/good")).await.is_ok());
    }

    /// Shaped like an NGC key (`nvapi-` and 64 characters); a canary.
    const KEY: &str = "nvapi-CANARY0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL_-xyz";

    fn one_gpu_request() -> AttestRequest {
        crate::nras_client::build_attest_request(
            &"ab".repeat(32),
            vec![crate::nras_client::GpuEvidence {
                evidence: "ZXY=".into(),
                certificate: "Y2M=".into(),
            }],
            "HOPPER",
        )
        .unwrap()
    }

    /// A NRAS that requires a key, as NVIDIA says it will: the Authorization
    /// header of each attest call is recorded; the right Bearer is answered,
    /// anything else is 401 with NRAS's words, which repeat the header.
    async fn nras_requiring(seen: std::sync::Arc<std::sync::Mutex<Vec<Option<String>>>>) -> String {
        serve(axum::Router::new().route(
            "/v4/attest/gpu",
            post(move |headers: axum::http::HeaderMap| async move {
                let auth = headers
                    .get(axum::http::header::AUTHORIZATION)
                    .map(|v| v.to_str().unwrap().to_string());
                seen.lock().unwrap().push(auth.clone());
                if auth.as_deref() == Some(format!("Bearer {KEY}").as_str()) {
                    axum::Json(serde_json::json!([["JWT", "o.j.t"], {"GPU-0": "g.j.t"}]))
                        .into_response()
                } else {
                    let words = format!("invalid API key: {auth:?}; expected {KEY}");
                    (axum::http::StatusCode::UNAUTHORIZED, words).into_response()
                }
            }),
        ))
        .await
    }

    use axum::response::IntoResponse as _;

    /// The key goes as `Authorization: Bearer <key>` on the attest call when
    /// one is given, and no Authorization header goes without one.
    #[tokio::test]
    async fn the_key_is_a_bearer_on_the_attest_call_when_given() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let base = nras_requiring(seen.clone()).await;
        let key = NrasKey::parse(KEY.as_bytes()).unwrap();
        let url = format!("{base}/v4/attest/gpu");
        let (overall, per_gpu) = attest_gpu(&url, &one_gpu_request(), Some(&key))
            .await
            .unwrap();
        assert_eq!(overall, "o.j.t");
        assert_eq!(per_gpu, vec![("GPU-0".into(), "g.j.t".into())]);
        let err = attest_gpu(&url, &one_gpu_request(), None)
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("401"), "{err:#}");
        assert_eq!(
            *seen.lock().unwrap(),
            vec![Some(format!("Bearer {KEY}")), None]
        );
    }

    /// A refusal carries NRAS's status and words, never the key, even when
    /// NRAS repeats it, in any of the error's forms.
    #[tokio::test]
    async fn a_refusal_that_repeats_the_key_is_said_without_it() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let base = nras_requiring(seen).await;
        let wrong = NrasKey::parse(b"nvapi-WRONG0123456789").unwrap();
        let url = format!("{base}/v4/attest/gpu");
        let err = attest_gpu(&url, &one_gpu_request(), Some(&wrong))
            .await
            .unwrap_err();
        for text in [err.to_string(), format!("{err:#}"), format!("{err:?}")] {
            assert!(
                text.contains("401") && text.contains("invalid API key"),
                "{text}"
            );
            assert!(!text.contains("nvapi-WRONG"), "{text}");
            assert!(err.downcast_ref::<KeyRefused>().is_some(), "{text}");
        }
        // The server's own copy of the right key is NRAS's words, not ours.
        let key = NrasKey::parse(KEY.as_bytes()).unwrap();
        assert_eq!(
            key.redact(&format!("a{KEY}b{KEY}")),
            "a(key not printed)b(key not printed)"
        );
    }

    /// A key under 16 characters is refused: `redact` replaces each copy of
    /// the key in NRAS's words, and a key of 1 to 3 characters would mask
    /// ordinary text there. An NGC key is about 70 characters.
    #[test]
    fn a_key_shorter_than_16_characters_is_refused() {
        let (sixteen, fifteen) = (&KEY[..16], &KEY[..15]);
        assert!(NrasKey::parse(sixteen.as_bytes()).is_ok());
        assert!(NrasKey::parse(format!("{sixteen}\n").as_bytes()).is_ok());
        let shorts = [
            "e".to_string(),
            "abc".into(),
            fifteen.into(),
            format!("{fifteen}\n"),
        ];
        for short in shorts {
            let e = NrasKey::parse(short.as_bytes()).unwrap_err();
            assert!(e.contains("16"), "{short:?}: {e}");
            assert!(
                !e.contains("CANARY") && !e.contains("abc"),
                "{short:?}: {e}"
            );
        }
    }

    /// One token of visible ASCII, one trailing newline taken off; anything
    /// else is refused without its content.
    #[test]
    fn a_key_is_one_token_of_visible_ascii_and_never_printed() {
        for good in [KEY.to_string(), format!("{KEY}\n"), format!("{KEY}\r\n")] {
            let k = NrasKey::parse(good.as_bytes()).unwrap();
            assert_eq!(k.redact(KEY), "(key not printed)", "{good:?}");
            assert!(!format!("{k:?}").contains("CANARY"));
        }
        let long = format!("nvapi-CANARY{}", "x".repeat(MAX_KEY_BYTES));
        for bad in [
            String::new(),
            "\n".into(),
            "nvapi-CANARY a".into(),
            "nvapi-CANARY\tb".into(),
            "nvapi-CANARY\u{e9}".into(),
            format!("{KEY}\n\n"),
            format!("\n{KEY}"),
            "nvapi-CANARY\u{1b}[2J".into(),
            long,
        ] {
            let e = NrasKey::parse(bad.as_bytes()).unwrap_err();
            assert!(!e.contains("CANARY"), "{bad:?}: {e}");
        }
        assert!(NrasKey::parse(&[0x6e, 0xff, 0x6b]).is_err());
    }

    /// The file is read whole, through no link; every refusal names the path
    /// and never the content.
    #[test]
    fn a_key_file_is_a_regular_file_and_its_content_is_never_printed() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("nras.key");
        std::fs::write(&file, format!("{KEY}\n")).unwrap();
        let k = NrasKey::read(&file).unwrap();
        assert_eq!(k.redact(KEY), "(key not printed)");
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&file, &link).unwrap();
        let big = dir.path().join("big");
        std::fs::write(&big, format!("{KEY}{}", " ".repeat(KEY_FILE_CAP as usize))).unwrap();
        let spaced = dir.path().join("spaced");
        std::fs::write(&spaced, format!("{KEY} {KEY}")).unwrap();
        for (p, says) in [
            (link, "not a regular file"),
            (dir.path().to_path_buf(), "not a regular file"),
            (dir.path().join("absent"), "No such file"),
            (big, "cap"),
            (spaced, "one token"),
        ] {
            let e = NrasKey::read(&p).unwrap_err();
            assert!(e.contains(says), "{}: {e}", p.display());
            assert!(!e.contains("CANARY"), "{}: {e}", p.display());
        }
    }
}
