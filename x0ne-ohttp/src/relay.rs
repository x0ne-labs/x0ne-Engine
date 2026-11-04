//! The relay: forwards encapsulated requests to one gateway and the answers back.
//!
//! It sees who is asking but not what; the gateway sees what but not who. It
//! forwards the body and its media type, nothing else: no client header, no
//! `Via` or `Forwarded` (RFC 9458 section 6.2). It does not retry.

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use reqwest::Url;

use crate::client::{check_url, http_client, under};
use crate::{KEYS_MEDIA_TYPE, REQUEST_MEDIA_TYPE, RESPONSE_MEDIA_TYPE};

/// Larger answers from the gateway are dropped (502).
const MAX_UPSTREAM_RESPONSE: usize = 64 << 20;

#[derive(Clone, Debug)]
pub struct RelayConfig {
    /// The one gateway this relay serves.
    pub gateway_url: String,
    /// Largest encapsulated request accepted (413 above).
    pub max_body: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum RelayError {
    #[error("gateway URL refused: {0}")]
    Url(String),
}

struct Relay {
    requests: Url,
    keys: Url,
    http: reqwest::Client,
}

pub fn router(cfg: RelayConfig) -> Result<Router, RelayError> {
    let gateway = check_url(&cfg.gateway_url).map_err(RelayError::Url)?;
    let relay = Arc::new(Relay {
        requests: under(&gateway, "").map_err(RelayError::Url)?,
        keys: under(&gateway, "keys").map_err(RelayError::Url)?,
        http: http_client(),
    });
    Ok(Router::new()
        .route("/", post(forward))
        .route("/keys", get(keys))
        .layer(DefaultBodyLimit::max(cfg.max_body))
        .with_state(relay))
}

fn media_type_is(headers: &HeaderMap, want: &str) -> bool {
    headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case(want))
}

async fn forward(State(relay): State<Arc<Relay>>, headers: HeaderMap, body: Bytes) -> Response {
    if !media_type_is(&headers, REQUEST_MEDIA_TYPE) {
        return StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response();
    }
    let upstream = relay
        .http
        .post(relay.requests.clone())
        .header("content-type", REQUEST_MEDIA_TYPE)
        .body(body)
        .send()
        .await;
    pass_back(upstream, RESPONSE_MEDIA_TYPE).await
}

async fn keys(State(relay): State<Arc<Relay>>) -> Response {
    let upstream = relay.http.get(relay.keys.clone()).send().await;
    pass_back(upstream, KEYS_MEDIA_TYPE).await
}

/// Status and body back, with `media_type` only when the gateway set it.
async fn pass_back(
    upstream: reqwest::Result<reqwest::Response>,
    media_type: &'static str,
) -> Response {
    let Ok(mut resp) = upstream else {
        return StatusCode::BAD_GATEWAY.into_response();
    };
    let Ok(status) = StatusCode::from_u16(resp.status().as_u16()) else {
        return StatusCode::BAD_GATEWAY.into_response();
    };
    let typed = media_type_is(resp.headers(), media_type);
    let mut body = Vec::new();
    loop {
        match resp.chunk().await {
            Ok(Some(chunk)) => {
                if body.len() + chunk.len() > MAX_UPSTREAM_RESPONSE {
                    return StatusCode::BAD_GATEWAY.into_response();
                }
                body.extend_from_slice(&chunk);
            }
            Ok(None) => break,
            Err(_) => return StatusCode::BAD_GATEWAY.into_response(),
        }
    }
    let mut out = Response::new(Body::from(body));
    *out.status_mut() = status;
    if typed {
        out.headers_mut()
            .insert("content-type", HeaderValue::from_static(media_type));
    }
    out
}
