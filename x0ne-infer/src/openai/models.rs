//! `GET /v1/models` and the `model` a request names. A surface serves one
//! model, by one id: the list names it as OpenAI's `Model` objects do (`id`,
//! `object`, `created`, `owned_by`), with what the engine adds under `x0ne`;
//! a request naming another id is refused, and the answer names the id
//! served, never the request's.

use axum::http::StatusCode;
use serde_json::{json, Value};

use crate::openai::error::ApiError;
use crate::openai::request::{stream_model_bytes, MAX_STREAM_MODEL_BYTES};

/// `owned_by` of every model the engine lists.
pub const OWNED_BY: &str = "x0ne-engine";

/// `error.code` of a request naming a model the surface does not serve, as
/// OpenAI's API names it.
pub const NOT_FOUND: &str = "model_not_found";

/// One model a surface serves.
#[derive(Debug, Clone)]
pub struct ModelCard {
    pub id: String,
    /// When the surface began serving it, in Unix seconds.
    pub created: u64,
    /// What the engine says of it beyond OpenAI's fields.
    pub x0ne: Value,
}

/// The body of `GET /v1/models`.
#[must_use]
pub fn list(cards: &[ModelCard]) -> Value {
    let data: Vec<Value> = cards
        .iter()
        .map(|c| {
            json!({
                "id": c.id,
                "object": "model",
                "created": c.created,
                "owned_by": OWNED_BY,
                "x0ne": c.x0ne,
            })
        })
        .collect();
    json!({ "object": "list", "data": data })
}

/// The id a request is answered under: `served`, when `asked` names it or
/// names none (an absent `model` is read as empty). Anything else is a 404
/// that quotes nothing of the request.
pub fn served<'a>(asked: &str, served: &'a str) -> Result<&'a str, ApiError> {
    if asked.is_empty() || asked == served {
        return Ok(served);
    }
    Err(ApiError::new(
        StatusCode::NOT_FOUND,
        "model: not served here; GET /v1/models lists the model this server serves",
    )
    .param("model")
    .code(NOT_FOUND))
}

/// Whether a surface may serve under `id`: every frame of a stream names
/// it, and a frame stays within 768 bytes for an id up to the request's
/// bound, [`MAX_STREAM_MODEL_BYTES`] once JSON-escaped. The cloud TD asks
/// it before anything is collected (exit 96).
pub fn servable(id: &str) -> Result<(), String> {
    let bytes = stream_model_bytes(id);
    if bytes <= MAX_STREAM_MODEL_BYTES {
        return Ok(());
    }
    Err(format!(
        "its name is {bytes} bytes once JSON-escaped, past the {MAX_STREAM_MODEL_BYTES} \
         a stream's frames are sized for; not served"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every frame of a stream names the id served (`stream.rs`), and stays
    /// within 768 bytes for an id up to the request's bound. The cloud TD
    /// does not start under a longer one (exit 96); every catalog name fits.
    #[test]
    fn an_id_past_the_frame_bound_is_not_servable() {
        assert_eq!(servable(&"m".repeat(MAX_STREAM_MODEL_BYTES)), Ok(()));
        assert!(servable(&"m".repeat(MAX_STREAM_MODEL_BYTES + 1)).is_err());
        // Six bytes each once JSON-escaped.
        assert!(servable(&"\u{1}".repeat(MAX_STREAM_MODEL_BYTES / 6 + 1)).is_err());
        for entry in crate::registry::Catalog::embedded().models {
            assert_eq!(servable(&entry.name), Ok(()), "{}", entry.name);
        }
    }

    #[test]
    fn the_list_has_openais_fields_and_the_engines_beside_them() {
        let list = list(&[ModelCard {
            id: "qwen3:0.6b".into(),
            created: 1_761_480_000,
            x0ne: json!({"digest_hex": "ab"}),
        }]);
        assert_eq!(
            list,
            json!({"object": "list", "data": [{"id": "qwen3:0.6b", "object": "model",
                "created": 1_761_480_000_u64, "owned_by": "x0ne-engine",
                "x0ne": {"digest_hex": "ab"}}]})
        );
    }

    /// A client set to any name used to get an answer under that name; it
    /// now learns the name served, and is answered under it.
    #[test]
    fn a_model_is_the_one_served_or_none() {
        assert_eq!(served("", "m:1b").unwrap(), "m:1b");
        assert_eq!(served("m:1b", "m:1b").unwrap(), "m:1b");
        let refused = served("gpt-4o", "m:1b").unwrap_err();
        assert_eq!(refused.status, StatusCode::NOT_FOUND);
        assert_eq!(refused.code.as_deref(), Some(NOT_FOUND));
        assert_eq!(refused.param.as_deref(), Some("model"));
        assert!(!refused.message.contains("gpt-4o"), "{}", refused.message);
    }
}
