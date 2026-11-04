//! The client: encapsulates to a pinned key configuration and sends through the
//! relay, once.
//!
//! It never retries and never redirects. A retry after a failure, on the same node
//! or another, would let whoever drops requests steer them (RFC 9458 section 6.5
//! forbids automatic retries without a positive signal).

use std::net::IpAddr;
use std::time::SystemTime;

use reqwest::Url;

use crate::config::{ConfigError, KeyPin, PinnedConfig};
use crate::message::{self, InnerRequest, InnerResponse, MessageError};
use crate::{KEYS_MEDIA_TYPE, REQUEST_MEDIA_TYPE, RESPONSE_MEDIA_TYPE};

/// Larger answers are refused before decapsulation.
const MAX_RESPONSE: usize = 64 << 20;

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("relay URL refused: {0}")]
    Url(String),
    #[error("header field not allowed inside the encapsulation: {0}")]
    HeaderRefused(String),
    #[error("key configuration: {0}")]
    Config(ConfigError),
    #[error("encapsulation failed")]
    Encapsulation,
    #[error("transport: {0}")]
    Transport(String),
    #[error("relay or gateway answered {0}")]
    Status(u16),
    #[error("unexpected media type")]
    MediaType,
    /// The answer was not sealed for this request: tampered, swapped or replayed.
    #[error("the answer does not open with this request's keys")]
    Decapsulation,
    #[error("malformed answer")]
    Malformed,
}

/// The state needed to open the answer to one request. Single use.
pub struct PendingResponse(ohttp::ClientResponse);

impl PendingResponse {
    pub fn open(self, enc_response: &[u8]) -> Result<InnerResponse, ClientError> {
        let bhttp = self
            .0
            .decapsulate(enc_response)
            .map_err(|_| ClientError::Decapsulation)?;
        message::decode_response(&bhttp).map_err(|_| ClientError::Malformed)
    }
}

/// Encodes, pads and encapsulates one request with a fresh HPKE context, dated by
/// the system clock.
pub fn encapsulate(
    to: &PinnedConfig,
    req: &InnerRequest,
    authority: &str,
) -> Result<(Vec<u8>, PendingResponse), ClientError> {
    encapsulate_at(to, req, authority, SystemTime::now())
}

/// [`encapsulate`] with the client's clock given: the request carries `now` to the
/// minute (see [`message::date_field`]).
pub fn encapsulate_at(
    to: &PinnedConfig,
    req: &InnerRequest,
    authority: &str,
    now: SystemTime,
) -> Result<(Vec<u8>, PendingResponse), ClientError> {
    let bhttp = message::encode_request(req, authority, now).map_err(|e| match e {
        MessageError::HeaderRefused(h) => ClientError::HeaderRefused(h),
        _ => ClientError::Encapsulation,
    })?;
    let ohttp_req = ohttp::ClientRequest::from_encoded_config(&to.encapsulation_config())
        .map_err(|_| ClientError::Encapsulation)?;
    let (enc, pending) = ohttp_req
        .encapsulate(&bhttp)
        .map_err(|_| ClientError::Encapsulation)?;
    Ok((enc, PendingResponse(pending)))
}

/// `https`, or `http` to a loopback address (tests and a relay on the same host).
pub(crate) fn check_url(url: &str) -> Result<Url, String> {
    let u = Url::parse(url).map_err(|e| format!("{url}: {e}"))?;
    let loopback = u.host_str().is_some_and(|h| {
        let h = h.trim_start_matches('[').trim_end_matches(']');
        h == "localhost" || h.parse::<IpAddr>().is_ok_and(|a| a.is_loopback())
    });
    match u.scheme() {
        "https" if u.host_str().is_some() => Ok(u),
        "http" if loopback => Ok(u),
        _ => Err(format!("{url}: https required outside the loopback")),
    }
}

/// `rel` resolved under `base`'s path, whether or not it ends with a slash.
pub(crate) fn under(base: &Url, rel: &str) -> Result<Url, String> {
    let mut base = base.clone();
    if !base.path().ends_with('/') {
        let p = format!("{}/", base.path());
        base.set_path(&p);
    }
    base.join(rel).map_err(|e| e.to_string())
}

pub(crate) fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

pub struct Client {
    relay: Url,
    authority: String,
    http: reqwest::Client,
}

impl Client {
    /// `relay_url` is the relay's base URL; `authority` the name the gateway
    /// serves, which only the gateway sees.
    pub fn new(relay_url: &str, authority: &str) -> Result<Self, ClientError> {
        let relay = check_url(relay_url).map_err(ClientError::Url)?;
        Ok(Self {
            relay,
            authority: authority.to_string(),
            http: http_client(),
        })
    }

    fn endpoint(&self, rel: &str) -> Result<Url, ClientError> {
        under(&self.relay, rel).map_err(ClientError::Url)
    }

    /// One attempt. Any failure is returned as is; the caller decides, and a
    /// caller that wants non-targeting does not redraw a node after a failure.
    pub async fn send(
        &self,
        to: &PinnedConfig,
        req: &InnerRequest,
    ) -> Result<InnerResponse, ClientError> {
        let (enc, pending) = encapsulate(to, req, &self.authority)?;
        let resp = self
            .http
            .post(self.endpoint("")?)
            .header("content-type", REQUEST_MEDIA_TYPE)
            .body(enc)
            .send()
            .await
            .map_err(|e| ClientError::Transport(e.without_url().to_string()))?;
        let status = resp.status().as_u16();
        if status != 200 {
            return Err(ClientError::Status(status));
        }
        if !has_media_type(&resp, RESPONSE_MEDIA_TYPE) {
            return Err(ClientError::MediaType);
        }
        let body = read_limited(resp, MAX_RESPONSE).await?;
        pending.open(&body)
    }

    /// The gateway's key configuration, through the relay, accepted only if it is
    /// the one pinned.
    pub async fn fetch_config(&self, pin: &KeyPin) -> Result<PinnedConfig, ClientError> {
        let resp = self
            .http
            .get(self.endpoint("keys")?)
            .send()
            .await
            .map_err(|e| ClientError::Transport(e.without_url().to_string()))?;
        let status = resp.status().as_u16();
        if status != 200 {
            return Err(ClientError::Status(status));
        }
        if !has_media_type(&resp, KEYS_MEDIA_TYPE) {
            return Err(ClientError::MediaType);
        }
        let body = read_limited(resp, 1 << 16).await?;
        PinnedConfig::from_keys_media(&body, pin).map_err(ClientError::Config)
    }
}

fn has_media_type(resp: &reqwest::Response, want: &str) -> bool {
    resp.headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case(want))
}

async fn read_limited(mut resp: reqwest::Response, max: usize) -> Result<Vec<u8>, ClientError> {
    let mut out = Vec::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| ClientError::Transport(e.without_url().to_string()))?
    {
        if out.len() + chunk.len() > max {
            return Err(ClientError::Malformed);
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}
