//! The error side of the OpenAI contract: one shape for the refusals of
//! every surface, `{"error": {"message", "type", "param", "code"}}`, as
//! OpenAI's API answers it (`param` and `code` a string or null). A surface
//! that says more puts it beside `error`, never inside it. A TD pads the
//! bytes this writes; it does not change them.
//! No log macro in this file: a message may name a field of a request.

use axum::extract::rejection::JsonRejection;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

/// `error.type` of every error the engine writes: the relay's since it was
/// written, and the one the error event of a stream carries.
pub const TYPE: &str = "x0ne_engine";

/// What a surface says when a request's own task failed: never the panic's
/// payload, which may quote the request.
pub const INTERNAL: &str = "internal error: this request failed; the server is still serving";

/// The most of a name or JSON path of the request an error quotes. Measured
/// 2026-10-27: a field name of 20001 bytes made the TD's padded refusal
/// 20480 bytes, where a 200 is 16384.
pub const MAX_QUOTED_BYTES: usize = 64;

/// `fragment`'s first [`MAX_QUOTED_BYTES`] bytes at most, cut on a character.
pub fn quoted(fragment: &str) -> &str {
    let mut end = fragment.len().min(MAX_QUOTED_BYTES);
    while !fragment.is_char_boundary(end) {
        end -= 1;
    }
    &fragment[..end]
}

/// axum's sentence for a body serde_json refused is `{axum's}: {path}:
/// {serde_json's}` (axum 0.8, serde_path_to_error), the path the request's
/// own keys: cut by [`quoted`]. serde_json's quotes nothing of a `Value`.
fn path_cut(text: &str) -> String {
    let Some((axum, rest)) = text.split_once(": ") else {
        return text.into();
    };
    match rest.rsplit_once(": ") {
        Some((path, serde)) => format!("{axum}: {}: {serde}", quoted(path)),
        None => text.into(),
    }
}

/// One refusal: the status, the sentence, and what OpenAI's shape adds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    /// The request field the refusal is about, when it names one.
    pub param: Option<String>,
    /// A stable name a client may branch on, such as `model_not_found`.
    pub code: Option<String>,
    /// Seconds for `Retry-After`: a full queue says when to come back.
    pub retry_after: Option<u64>,
}

impl ApiError {
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            param: None,
            code: None,
            retry_after: None,
        }
    }

    #[must_use]
    pub fn code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    #[must_use]
    pub fn param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }

    #[must_use]
    pub fn retry_after(mut self, seconds: u64) -> Self {
        self.retry_after = Some(seconds);
        self
    }

    /// `{"error": {"message", "type", "param", "code"}}`.
    #[must_use]
    pub fn body(&self) -> Value {
        json!({
            "error": {
                "message": self.message,
                "type": TYPE,
                "param": self.param,
                "code": self.code,
            }
        })
    }

    /// The error a surface read from another's body: OpenAI's shape, or the
    /// bare `{"error": "<message>"}` the TD answered before this shape.
    /// `None` when `body` holds neither.
    #[must_use]
    pub fn read(status: StatusCode, body: &Value) -> Option<Self> {
        let error = &body["error"];
        if let Some(message) = error.as_str() {
            return Some(Self::new(status, message));
        }
        let message = error["message"].as_str()?;
        let text = |key: &str| error[key].as_str().map(str::to_string);
        Some(Self {
            param: text("param"),
            code: text("code"),
            ..Self::new(status, message)
        })
    }

    /// [`Self::body`] with `key` beside `error`, as JSON with its status.
    #[must_use]
    pub fn response_with(self, key: &str, value: Value) -> Response {
        let mut body = self.body();
        body[key] = value;
        self.respond(body)
    }

    fn respond(self, body: Value) -> Response {
        let mut response = (self.status, Json(body)).into_response();
        if let Some(seconds) = self.retry_after {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(seconds));
        }
        response
    }
}

impl From<(StatusCode, String)> for ApiError {
    fn from((status, message): (StatusCode, String)) -> Self {
        Self::new(status, message)
    }
}

/// What the reader of a request body gets when the body was not whole in
/// time (`crate::server::BODY_TIMEOUT`); a 408 once it reaches [`ApiError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BodyLate(pub std::time::Duration);

impl std::fmt::Display for BodyLate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the request body was not whole {} seconds after its head",
            self.0.as_secs()
        )
    }
}

impl std::error::Error for BodyLate {}

impl BodyLate {
    /// The [`BodyLate`] `e` or one of its sources is, if any: axum wraps a
    /// body's error in its rejection.
    fn within(e: &(dyn std::error::Error + 'static)) -> Option<Self> {
        let mut at = Some(e);
        while let Some(e) = at {
            if let Some(late) = e.downcast_ref::<Self>() {
                return Some(*late);
            }
            at = e.source();
        }
        None
    }
}

impl From<JsonRejection> for ApiError {
    /// Malformed JSON 400, a body that is not JSON's shape 422, no JSON
    /// content type 415: axum's status and sentence, in this shape, its
    /// JSON path cut by [`quoted`]. A body that was not whole in time
    /// ([`BodyLate`]) 408, its own sentence.
    fn from(e: JsonRejection) -> Self {
        if let Some(late) = BodyLate::within(&e) {
            return Self::new(StatusCode::REQUEST_TIMEOUT, late.to_string());
        }
        let text = match e {
            JsonRejection::JsonSyntaxError(_) | JsonRejection::JsonDataError(_) => {
                path_cut(&e.body_text())
            }
            _ => e.body_text(),
        };
        Self::new(e.status(), text)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = self.body();
        self.respond(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fragment of the request is quoted to 64
    /// bytes, cut on a character; axum's JSON path too, with serde_json's
    /// sentence kept whole, whatever the key holds.
    #[test]
    fn a_quoted_fragment_stops_at_64_bytes_on_a_character() {
        assert_eq!(quoted("model"), "model");
        assert_eq!(quoted(&"k".repeat(64)), "k".repeat(64));
        assert_eq!(quoted(&"k".repeat(65)), "k".repeat(64));
        // The 64th byte inside an é: 63 bytes.
        let long = format!("k{}", "é".repeat(10_000));
        assert_eq!(quoted(&long), format!("k{}", "é".repeat(31)));
        for key in [long.clone(), format!("a: b{}", "x".repeat(20_000))] {
            let body = format!(r#"{{"{key}": ["#);
            let rejected = Json::<Value>::from_bytes(body.as_bytes()).unwrap_err();
            let said = ApiError::from(rejected).message;
            let tail = format!(": EOF while parsing a list at line 1 column {}", body.len());
            assert_eq!(
                said,
                format!(
                    "Failed to parse the request body as JSON: {}{tail}",
                    quoted(&key)
                )
            );
        }
        let trailing = Json::<Value>::from_bytes(b"{} x").unwrap_err();
        assert_eq!(
            ApiError::from(trailing).message,
            "Failed to parse the request body as JSON: trailing characters at line 1 column 4"
        );
    }

    async fn parts(r: Response) -> (StatusCode, Option<String>, Value) {
        let status = r.status();
        let retry = r
            .headers()
            .get(header::RETRY_AFTER)
            .map(|v| v.to_str().unwrap().to_string());
        let bytes = axum::body::to_bytes(r.into_body(), 1 << 16).await.unwrap();
        (status, retry, serde_json::from_slice(&bytes).unwrap())
    }

    /// OpenAI's four keys, `param` and `code` null when unknown: a client
    /// that reads `error.code` finds the key, as the spec has it.
    #[tokio::test]
    async fn an_error_has_the_four_keys_of_openai_and_nothing_else() {
        let (status, retry, body) = parts(
            ApiError::new(StatusCode::NOT_FOUND, "no such model")
                .code("model_not_found")
                .param("model")
                .into_response(),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(retry, None);
        assert_eq!(
            body,
            json!({"error": {"message": "no such model", "type": "x0ne_engine",
                "param": "model", "code": "model_not_found"}})
        );
        let (_, _, bare) =
            parts(ApiError::new(StatusCode::BAD_REQUEST, "no").into_response()).await;
        assert_eq!(
            bare,
            json!({"error": {"message": "no", "type": "x0ne_engine", "param": null, "code": null}})
        );
    }

    #[tokio::test]
    async fn a_full_queue_says_when_to_come_back_and_a_surface_adds_beside_error() {
        let (status, retry, body) = parts(
            ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "busy")
                .retry_after(5)
                .response_with("x0ne", json!({"confidential": false})),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(retry.as_deref(), Some("5"));
        assert_eq!(body["x0ne"], json!({"confidential": false}));
        assert_eq!(body["error"]["message"], "busy");
    }

    /// A relay in front of a TD of either age reads its refusal.
    #[test]
    fn an_error_reads_back_from_either_shape() {
        let s = StatusCode::BAD_REQUEST;
        assert_eq!(
            ApiError::read(s, &json!({"error": "n: only 1"})),
            Some(ApiError::new(s, "n: only 1"))
        );
        let new = ApiError::new(s, "nope")
            .code("model_not_found")
            .param("model");
        assert_eq!(ApiError::read(s, &new.body()), Some(new));
        assert_eq!(ApiError::read(s, &json!({"error": {"code": "x"}})), None);
        assert_eq!(ApiError::read(s, &json!({"detail": "x"})), None);
    }
}
