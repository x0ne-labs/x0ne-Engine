//! HTTP to the provider's API. The token goes to fixed `https://*.googleapis.com`
//! hosts and nowhere else; no redirects. A `test-endpoints` build (debug,
//! enabled by this crate's tests alone) may point it at one loopback fake.

use std::cell::RefCell;
use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use reqwest::{Method, StatusCode, Url};
use serde_json::Value;
use zeroize::Zeroizing;

use crate::token::TokenSource;
use crate::Failure;

/// The override lets the token travel in cleartext to a loopback port. A
/// release carrying it would ship that path (2026-10-12 attack: a planted
/// X0NE_ENGINE_GCP_ENDPOINT captured a live token locally).
#[cfg(all(feature = "test-endpoints", not(debug_assertions)))]
compile_error!("feature `test-endpoints` is for debug test builds only");

#[cfg(feature = "test-endpoints")]
pub const ENDPOINT_ENV: &str = "X0NE_ENGINE_GCP_ENDPOINT";
/// Largest JSON body read from the API (the billing catalog pages are the big ones).
pub const MAX_BODY: usize = 64 << 20;

#[derive(Debug, Clone)]
pub struct Endpoints {
    pub compute: Url,
    pub oauth2: Url,
    pub billing: Url,
    pub storage: Url,
    /// Cloud Quotas v1: the spot H100 quota when regions.get does not list it.
    pub quotas: Url,
    /// True when every endpoint is the loopback test base.
    pub loopback: bool,
}

/// Parsed, not prefix-matched: `http://127.0.0.1:1@evil.example` is evil.example.
pub fn is_loopback_http(u: &Url) -> bool {
    u.scheme() == "http"
        && u.username().is_empty()
        && u.password().is_none()
        && matches!(u.host_str(), Some("127.0.0.1" | "[::1]"))
        && u.port().is_some()
}

impl Endpoints {
    pub fn google() -> Self {
        let u = |s: &str| Url::parse(s).expect("constant URL");
        Endpoints {
            compute: u("https://compute.googleapis.com/"),
            oauth2: u("https://oauth2.googleapis.com/"),
            billing: u("https://cloudbilling.googleapis.com/"),
            storage: u("https://storage.googleapis.com/"),
            quotas: u("https://cloudquotas.googleapis.com/"),
            loopback: false,
        }
    }

    /// The endpoints this build talks to: Google's, and nothing is read.
    #[cfg(not(feature = "test-endpoints"))]
    pub fn resolve() -> Result<Self, Failure> {
        Ok(Self::google())
    }

    /// Test builds only: `X0NE_ENGINE_GCP_ENDPOINT` names one loopback fake.
    #[cfg(feature = "test-endpoints")]
    pub fn resolve() -> Result<Self, Failure> {
        match std::env::var(ENDPOINT_ENV) {
            Err(std::env::VarError::NotPresent) => Ok(Self::google()),
            Err(_) => Err(Failure::Usage(format!("{ENDPOINT_ENV} is not UTF-8"))),
            Ok(v) => Self::loopback(&v),
        }
    }

    #[cfg(feature = "test-endpoints")]
    pub fn loopback(base: &str) -> Result<Self, Failure> {
        let refused = || {
            Failure::Usage(format!(
                "{ENDPOINT_ENV}={base:?} refused: test overrides go to http://127.0.0.1:<port> only \
                 (the token never leaves for anything but *.googleapis.com)"
            ))
        };
        let mut u = Url::parse(base).map_err(|_| refused())?;
        if !is_loopback_http(&u) || u.query().is_some() || u.fragment().is_some() {
            return Err(refused());
        }
        u.set_path("/");
        Ok(Endpoints {
            compute: u.clone(),
            oauth2: u.clone(),
            billing: u.clone(),
            storage: u.clone(),
            quotas: u,
            loopback: true,
        })
    }

    /// May the token be sent to this URL?
    pub fn allows(&self, u: &Url) -> bool {
        let same = |b: &Url| {
            u.scheme() == b.scheme()
                && u.host_str() == b.host_str()
                && u.port_or_known_default() == b.port_or_known_default()
                && u.username().is_empty()
                && u.password().is_none()
        };
        [
            &self.compute,
            &self.oauth2,
            &self.billing,
            &self.storage,
            &self.quotas,
        ]
        .into_iter()
        .any(same)
    }
}

/// An API answer that was not 2xx, or no answer.
#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: Option<u16>,
    pub reason: String,
    pub message: String,
}

impl ApiError {
    pub fn not_found(&self) -> bool {
        self.status == Some(404)
    }
    pub fn unauthorized(&self) -> bool {
        matches!(self.status, Some(401))
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.status {
            Some(s) if self.reason.is_empty() => write!(f, "HTTP {s}: {}", self.message),
            Some(s) => write!(f, "HTTP {s} {}: {}", self.reason, self.message),
            None => write!(f, "{}", self.message),
        }
    }
}

pub struct Response {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

impl Response {
    pub fn json(&self) -> Result<Value, ApiError> {
        if self.body.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_slice(&self.body).map_err(|e| ApiError {
            status: Some(self.status.as_u16()),
            reason: "invalid-json".into(),
            message: format!("unreadable answer: {e}"),
        })
    }
}

/// Google's error envelope: `{"error":{"code","message","errors":[{"reason"}]}}`.
fn api_error(status: StatusCode, body: &[u8]) -> ApiError {
    let v: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    let e = &v["error"];
    let message = e["message"]
        .as_str()
        .or_else(|| e.as_str())
        .or_else(|| v["error_description"].as_str())
        .unwrap_or_else(|| status.canonical_reason().unwrap_or("error"));
    let reason = e["errors"][0]["reason"]
        .as_str()
        .or_else(|| e["status"].as_str())
        .unwrap_or("");
    ApiError {
        status: Some(status.as_u16()),
        reason: crate::sanitize(reason, 80),
        message: crate::sanitize(message, 400),
    }
}

pub struct Api {
    client: reqwest::Client,
    pub ep: Endpoints,
    token: RefCell<Box<dyn TokenSource>>,
}

pub enum Body {
    None,
    Json(Value),
    Form(Vec<(String, Zeroizing<String>)>),
    Bytes(Vec<u8>, &'static str),
}

impl Api {
    pub fn new(ep: Endpoints, token: Box<dyn TokenSource>) -> Result<Self, Failure> {
        let client = reqwest::Client::builder()
            .user_agent(concat!("x0ne-engine-cloud/", env!("CARGO_PKG_VERSION")))
            .redirect(reqwest::redirect::Policy::none())
            // `loopback` is set by `Endpoints::loopback` alone, which a
            // release build does not contain: HTTPS only, there.
            .https_only(!ep.loopback)
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| Failure::Preflight(format!("http client: {e}")))?;
        Ok(Api {
            client,
            ep,
            token: RefCell::new(token),
        })
    }

    pub fn token_refreshable(&self) -> bool {
        self.token.borrow().refreshable()
    }

    /// The token itself, for the one call that sends it in a body (tokeninfo).
    pub fn raw_token(&self) -> Result<Zeroizing<String>, Failure> {
        self.token.borrow_mut().token()
    }

    pub async fn send(
        &self,
        method: Method,
        url: Url,
        body: Body,
        extra: &[(&'static str, String)],
        bearer: bool,
    ) -> Result<Response, ApiError> {
        if !self.ep.allows(&url) {
            return Err(ApiError {
                status: None,
                reason: "refused".into(),
                message: format!(
                    "{} is not an API endpoint this tool talks to",
                    url.origin().ascii_serialization()
                ),
            });
        }
        let mut rb = self.client.request(method, url);
        if bearer {
            let tok = self.token.borrow_mut().token().map_err(|f| ApiError {
                status: None,
                reason: "token".into(),
                message: f.message().to_string(),
            })?;
            let mut line = Zeroizing::new(String::with_capacity(tok.len() + 7));
            line.push_str("Bearer ");
            line.push_str(&tok);
            let mut hv = HeaderValue::from_str(&line).map_err(|_| ApiError {
                status: None,
                reason: "token".into(),
                message: "token not usable in a header".into(),
            })?;
            hv.set_sensitive(true);
            rb = rb.header(AUTHORIZATION, hv);
        }
        for (k, v) in extra {
            rb = rb.header(*k, v.as_str());
        }
        rb = match body {
            Body::None => rb,
            Body::Json(v) => rb.json(&v),
            Body::Form(pairs) => {
                let mut s = Zeroizing::new(String::new());
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        s.push('&');
                    }
                    s.push_str(&url_encode(k));
                    s.push('=');
                    s.push_str(&url_encode(v));
                }
                rb.header("content-type", "application/x-www-form-urlencoded")
                    .body(s.as_bytes().to_vec())
            }
            Body::Bytes(b, ct) => rb.header("content-type", ct).body(b),
        };
        let mut resp = rb.send().await.map_err(|e| ApiError {
            status: None,
            reason: "network".into(),
            message: crate::sanitize(&without_url(&e), 300),
        })?;
        let status = resp.status();
        let headers = resp.headers().clone();
        let mut body = Vec::new();
        loop {
            match resp.chunk().await {
                Ok(Some(c)) => {
                    if body.len() + c.len() > MAX_BODY {
                        return Err(ApiError {
                            status: Some(status.as_u16()),
                            reason: "too-large".into(),
                            message: format!("answer larger than {MAX_BODY} bytes"),
                        });
                    }
                    body.extend_from_slice(&c);
                }
                Ok(None) => break,
                Err(e) => {
                    return Err(ApiError {
                        status: Some(status.as_u16()),
                        reason: "network".into(),
                        message: crate::sanitize(&without_url(&e), 300),
                    })
                }
            }
        }
        if status.is_success() || status == StatusCode::PERMANENT_REDIRECT {
            Ok(Response {
                status,
                headers,
                body,
            })
        } else {
            Err(api_error(status, &body))
        }
    }

    pub async fn get_json(&self, url: Url) -> Result<Value, ApiError> {
        self.send(Method::GET, url, Body::None, &[], true)
            .await?
            .json()
    }

    pub async fn post_json(&self, url: Url, body: Value) -> Result<Value, ApiError> {
        self.send(Method::POST, url, Body::Json(body), &[], true)
            .await?
            .json()
    }

    pub async fn delete(&self, url: Url) -> Result<Value, ApiError> {
        self.send(Method::DELETE, url, Body::None, &[], true)
            .await?
            .json()
    }
}

/// reqwest errors print the URL; ours carry no secret, but stay short.
fn without_url(e: &reqwest::Error) -> String {
    let mut s = e.to_string();
    if let Some(u) = e.url() {
        s = s.replace(u.as_str(), &u.origin().ascii_serialization());
    }
    if e.is_timeout() {
        s.push_str(" (timeout)");
    }
    s
}

/// application/x-www-form-urlencoded, byte by byte.
fn url_encode(s: &str) -> Zeroizing<String> {
    let mut out = Zeroizing::new(String::with_capacity(s.len() * 3));
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "test-endpoints")]
    #[test]
    fn only_a_loopback_http_base_is_accepted_as_override() {
        assert!(Endpoints::loopback("http://127.0.0.1:8080").is_ok());
        assert!(Endpoints::loopback("http://[::1]:8080/").is_ok());
        for bad in [
            "http://example.com:80",
            "https://evil.example",
            "https://127.0.0.1:443",
            "http://127.0.0.1:1@evil.example",
            "http://127.0.0.1.evil.example:9",
            "http://localhost:1@evil.example:9/",
            // A name needs resolving; the override takes IP literals only.
            "http://localhost:8080",
            "http://127.0.0.1",
            "ftp://127.0.0.1:21",
            "http://127.0.0.1:9/?x=1",
        ] {
            assert!(Endpoints::loopback(bad).is_err(), "{bad} accepted");
        }
    }

    #[test]
    fn the_token_goes_to_google_hosts_only() {
        let g = Endpoints::google();
        let ok = |s: &str| g.allows(&Url::parse(s).unwrap());
        assert!(ok("https://compute.googleapis.com/compute/v1/projects/p"));
        assert!(ok(
            "https://storage.googleapis.com/upload/storage/v1/b/x/o?upload_id=1"
        ));
        assert!(ok("https://cloudquotas.googleapis.com/v1/projects/p/locations/global/services/compute.googleapis.com/quotaInfos/q"));
        assert!(!ok("https://cloudquotas.googleapis.com.evil.example/v1"));
        assert!(!ok("http://compute.googleapis.com/compute/v1"));
        assert!(!ok("https://evil.example/upload"));
        assert!(!ok("https://storage.googleapis.com.evil.example/x"));
        assert!(!ok("https://user@storage.googleapis.com/x"));
        #[cfg(feature = "test-endpoints")]
        {
            let l = Endpoints::loopback("http://127.0.0.1:4000").unwrap();
            assert!(l.allows(&Url::parse("http://127.0.0.1:4000/x").unwrap()));
            assert!(!l.allows(&Url::parse("http://127.0.0.1:4001/x").unwrap()));
        }
    }

    #[test]
    fn google_error_envelopes_are_read_and_sanitized() {
        let e = api_error(
            StatusCode::FORBIDDEN,
            br#"{"error":{"code":403,"message":"Quota \u001b[31mexceeded","errors":[{"reason":"quotaExceeded"}]}}"#,
        );
        assert_eq!(e.status, Some(403));
        assert_eq!(e.reason, "quotaExceeded");
        assert_eq!(e.message, "Quota ?[31mexceeded");
    }
}
