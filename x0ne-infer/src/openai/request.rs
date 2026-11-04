//! The request side of the OpenAI contract: the one validator every chat
//! surface runs (the TD and the relay), and the request it reads.
//! No log macro in this file: it holds the text of a prompt.

use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use super::error::quoted;
use crate::backend::{GenerationParams, Sampling};

#[derive(Debug)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// `max_completion_tokens`, else its deprecated alias `max_tokens`, else 256.
    pub max_tokens: usize,
    /// 0 decodes greedily, and so does an absent `temperature` (OpenAI's default is 1).
    pub temperature: f64,
    pub top_p: f64,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    /// OpenAI `response_format: json_schema`, compiled.
    pub json_schema: Option<crate::json_schema::JsonSchema>,
    /// OpenAI `stream`: SSE chunks as the text is final, the receipt on the last.
    pub stream: bool,
    /// `stream_options.include_usage`: a usage chunk before `[DONE]`.
    pub include_usage: bool,
    /// The non-OpenAI part of the request: the conversation, and an adapter
    /// every server refuses by name (`server::admit_chat`).
    pub x0ne: ChatRequestOptions,
}
fn default_max_tokens() -> usize {
    256
}

impl ChatRequest {
    /// The decoding asked for, `max_tokens` cut to the server's `ceiling`.
    /// `fresh_seed` is drawn only when sampling without a `seed`.
    pub fn generation_params(
        &self,
        ceiling: usize,
        fresh_seed: impl FnOnce() -> Result<u64, String>,
    ) -> Result<GenerationParams, String> {
        let sampling = if self.temperature > 0.0 {
            Some(Sampling {
                temperature: self.temperature,
                top_p: self.top_p,
                seed: match self.seed {
                    Some(seed) => seed,
                    None => fresh_seed()?,
                },
            })
        } else {
            None
        };
        Ok(GenerationParams {
            max_tokens: self.max_tokens.min(ceiling),
            sampling,
            stop: self.stop.clone(),
            json_schema: self.json_schema.clone(),
        })
    }

    /// What a receipt for this request must sign about the decoding
    /// ([`crate::decoding::Asked::check_signed`]).
    pub fn asked(&self) -> crate::decoding::Asked {
        crate::decoding::Asked {
            sampling: (self.temperature > 0.0).then_some(crate::decoding::AskedSampling {
                temperature: self.temperature,
                top_p: self.top_p,
                seed: self.seed,
            }),
            json_schema: self.json_schema.clone(),
            conversation: self.x0ne.conversation.clone(),
        }
    }
}

/// What serde reads once the fields are judged; the rest is read by hand.
#[derive(Deserialize)]
struct WireRequest {
    #[serde(default)]
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    x0ne: ChatRequestOptions,
}

#[derive(Debug, Default, Deserialize)]
pub struct ChatRequestOptions {
    /// A per-request adapter, by name: read only to be refused, one attested
    /// model is served.
    #[serde(default)]
    pub adapter: Option<String>,
    /// `conversation` and `end_conversation`, read by hand
    /// ([`crate::context::AskedConversation::from_x0ne`]). The TD server only.
    #[serde(skip)]
    pub conversation: crate::context::AskedConversation,
}

/// The OpenAI request fields both servers parse. Anything else is
/// refused with a 400 that names it: an ignored `logprobs` answers a
/// question the caller did not ask. Both stream; only this server pads.
pub const HONOURED_FIELDS: &[&str] = &[
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "seed",
    "stop",
    "response_format",
    "stream",
    "stream_options",
    "x0ne",
];

/// Fields accepted only at the value this engine has anyway: one choice.
pub const PINNED_FIELDS: &[(&str, &str)] = &[("n", "1")];

/// Tool calling, in the request and in its messages: not implemented, so
/// refused with [`TOOLS_NOT_YET`] rather than answered without the tools.
pub const TOOL_FIELDS: &[&str] = &[
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "functions",
    "function_call",
];
const TOOL_MESSAGE_FIELDS: &[&str] = &["tool_calls", "tool_call_id", "function_call"];
const TOOL_ROLES: &[&str] = &["tool", "function"];
pub const TOOLS_NOT_YET: &str =
    "tool calling is not supported yet by this server (describe the tools in the prompt text)";

/// OpenAI's bound on `stop`.
pub const MAX_STOP_SEQUENCES: usize = 4;
/// Bytes per stop sequence: the stream holds back up to this many minus one,
/// and searches that many again at every step, under the model lock.
pub const MAX_STOP_BYTES: usize = 256;
/// `model` once JSON-escaped, with `stream: true`, and the id a TD serves
/// under (`models::servable`): every frame names the id served, so it sets
/// the frame size, which must stay within 768 bytes.
pub const MAX_STREAM_MODEL_BYTES: usize = 256;

/// The bytes `model` takes in every frame of a stream: its length once
/// JSON-escaped, the quotes left out.
#[must_use]
pub fn stream_model_bytes(model: &str) -> usize {
    serde_json::to_string(model).map_or(usize::MAX, |m| m.len() - 2)
}

/// `stream` and `stream_options.include_usage`: a boolean, `null` is absent.
fn flag(
    fields: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    name: &str,
) -> Result<bool, (StatusCode, String)> {
    match fields.get(key) {
        None | Some(serde_json::Value::Null) => Ok(false),
        Some(serde_json::Value::Bool(b)) => Ok(*b),
        Some(_) => Err(refusal(name, "must be true, false or null")),
    }
}

/// Why `top_p` below 1, or `seed`, is refused with an absent or 0 `temperature`.
const GREEDY_DRAWS_NOTHING: &str = "a temperature absent or 0 decodes greedily here \
                                    (OpenAI's default is 1), and greedy decoding draws nothing";

fn refusal(field: &str, why: &str) -> (StatusCode, String) {
    (
        StatusCode::BAD_REQUEST,
        format!("{field}: {why}; refused rather than ignored"),
    )
}

/// An optional number within `lo..=hi`; `null` is absent.
fn number_in(
    fields: &serde_json::Map<String, serde_json::Value>,
    name: &str,
    lo: f64,
    hi: f64,
) -> Result<Option<f64>, (StatusCode, String)> {
    match fields.get(name) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(v) => match v.as_f64() {
            Some(x) if (lo..=hi).contains(&x) => Ok(Some(x)),
            _ => Err(refusal(
                name,
                &format!("must be a number from {lo} to {hi}"),
            )),
        },
    }
}

/// `stop`: absent, `null`, a string, or up to [`MAX_STOP_SEQUENCES`] strings,
/// none empty nor over [`MAX_STOP_BYTES`]. The values are never quoted back.
fn stop_sequences(value: Option<&serde_json::Value>) -> Result<Vec<String>, (StatusCode, String)> {
    use serde_json::Value;
    let bad = || {
        refusal(
            "stop",
            &format!(
                "must be a non-empty string of at most {MAX_STOP_BYTES} bytes, \
                 or up to {MAX_STOP_SEQUENCES} of them"
            ),
        )
    };
    let list = match value {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| v.as_str().map(str::to_string))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(bad)?,
        Some(_) => return Err(bad()),
    };
    if list.len() > MAX_STOP_SEQUENCES
        || list
            .iter()
            .any(|s| s.is_empty() || s.len() > MAX_STOP_BYTES)
    {
        return Err(bad());
    }
    Ok(list)
}

/// The 422 when serde's sentence is not one [`unquoted`] knows.
const WRONG_SHAPE: &str = "a field of the request has the wrong type";

/// serde's sentence for a field of the wrong shape, without the value it
/// quotes: `invalid type: string "…", expected a sequence` becomes
/// `invalid type: string, expected a sequence`. Its words come from fixed
/// lists, so its length follows no byte of the request; else [`WRONG_SHAPE`].
fn unquoted(e: &serde_json::Error) -> String {
    const FOUND: &[&str] = &[
        "string",
        "integer",
        "floating point",
        "boolean",
        "null",
        "map",
        "sequence",
    ];
    const MISSING: &[&str] = &[
        "missing field `messages`",
        "missing field `role`",
        "missing field `content`",
    ];
    let said = e.to_string();
    if MISSING.contains(&said.as_str()) {
        return said;
    }
    // The last one is serde's: what it expects follows what it found.
    let Some((found, expected)) = said.rsplit_once(", expected ") else {
        return WRONG_SHAPE.into();
    };
    if expected.is_empty()
        || !expected
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b' ')
    {
        return WRONG_SHAPE.into();
    }
    if found.starts_with("invalid length ") {
        return format!("invalid length, expected {expected}");
    }
    for kind in ["invalid type", "invalid value"] {
        let Some(found) = found.strip_prefix(kind).and_then(|f| f.strip_prefix(": ")) else {
            continue;
        };
        return match FOUND
            .iter()
            .find(|f| found == **f || found.strip_prefix(**f).is_some_and(|v| v.starts_with(' ')))
        {
            Some(f) => format!("{kind}: {f}, expected {expected}"),
            None => format!("{kind}, expected {expected}"),
        };
    }
    WRONG_SHAPE.into()
}

/// A chat request, judged against [`HONOURED_FIELDS`], [`PINNED_FIELDS`] and
/// [`TOOL_FIELDS`] before it is read: 400 naming the first unhonoured field or
/// out-of-range value, 422 for a field of the wrong shape.
pub fn parse_chat_request(body: serde_json::Value) -> Result<ChatRequest, (StatusCode, String)> {
    use serde_json::Value;
    let refuse = |field: String, why: &str| Err(refusal(&field, why));
    let Some(fields) = body.as_object() else {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "the body must be a JSON object".into(),
        ));
    };
    for (name, value) in fields {
        if HONOURED_FIELDS.contains(&name.as_str()) {
            continue;
        }
        if TOOL_FIELDS.contains(&name.as_str()) {
            return refuse(name.clone(), TOOLS_NOT_YET);
        }
        let Some((_, only)) = PINNED_FIELDS.iter().find(|(n, _)| n == name) else {
            return refuse(
                quoted(name).to_string(),
                "not supported by this server (supported: model, messages, \
                 max_completion_tokens or max_tokens, temperature, top_p, seed, stop, \
                 response_format, stream, stream_options, and n 1)",
            );
        };
        let honoured = match value {
            Value::Number(n) => n.as_f64().map(|v| v.to_string()).as_deref() == Some(*only),
            Value::Bool(b) => b.to_string() == *only,
            _ => false,
        };
        if !honoured {
            return refuse(
                name.clone(),
                &format!("only {only} is supported (one choice)"),
            );
        }
    }
    if let Some(messages) = fields.get("messages").and_then(Value::as_array) {
        for message in messages.iter().filter_map(Value::as_object) {
            if let Some(name) = message
                .keys()
                .find(|k| TOOL_MESSAGE_FIELDS.contains(&k.as_str()))
            {
                return refuse(format!("messages[].{name}"), TOOLS_NOT_YET);
            }
            if let Some(role) = message
                .get("role")
                .and_then(Value::as_str)
                .filter(|r| TOOL_ROLES.contains(r))
            {
                return refuse(format!("messages[].role {role}"), TOOLS_NOT_YET);
            }
            if let Some(name) = message.keys().find(|k| *k != "role" && *k != "content") {
                return refuse(
                    format!("messages[].{}", quoted(name)),
                    "not supported (a message is role and content)",
                );
            }
        }
    }
    if let Some(options) = fields.get("x0ne").and_then(Value::as_object) {
        if let Some(name) = options
            .keys()
            .find(|k| !["adapter", "conversation", "end_conversation"].contains(&k.as_str()))
        {
            return refuse(
                format!("x0ne.{}", quoted(name)),
                "not supported (x0ne takes adapter, conversation and end_conversation)",
            );
        }
    }
    let conversation = crate::context::AskedConversation::from_x0ne(fields.get("x0ne"))
        .map_err(|(field, why)| refusal(&format!("x0ne.{field}"), &why))?;
    let temperature = number_in(fields, "temperature", 0.0, 2.0)?.unwrap_or(0.0);
    let top_p = number_in(fields, "top_p", 0.0, 1.0)?.unwrap_or(1.0);
    // OpenAI types it int: a negative seed is accepted, its bits kept.
    let seed = match fields.get("seed") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_u64()
                .or_else(|| v.as_i64().map(|s| s as u64))
                .ok_or_else(|| refusal("seed", "must be an integer"))?,
        ),
    };
    // Accepted here, both would be ignored. top_p 1 is no nucleus, the value
    // greedy decoding accepted before sampling existed.
    if temperature <= 0.0 {
        if top_p < 1.0 {
            return refuse(
                "top_p".into(),
                &format!("only 1 without a temperature above 0; {GREEDY_DRAWS_NOTHING}"),
            );
        }
        if seed.is_some() {
            return refuse(
                "seed".into(),
                &format!("needs a temperature above 0; {GREEDY_DRAWS_NOTHING}"),
            );
        }
    }
    let stop = stop_sequences(fields.get("stop"))?;
    // Compiled here, before any lock or device: a schema past a bound costs
    // its parse and nothing else (crate::json_schema).
    let json_schema = crate::decoding::response_format(fields.get("response_format"))
        .map_err(|e| refusal("response_format", &e))?;
    if json_schema.is_some() && !stop.is_empty() {
        return refuse(
            "stop".into(),
            "not with response_format json_schema (a stop sequence would cut the document)",
        );
    }
    let stream = flag(fields, "stream", "stream")?;
    let include_usage = match fields.get("stream_options") {
        None | Some(Value::Null) => false,
        Some(Value::Object(options)) => {
            if !stream {
                return refuse("stream_options".into(), "only with stream: true");
            }
            if let Some(name) = options.keys().find(|k| *k != "include_usage") {
                return refuse(
                    format!("stream_options.{}", quoted(name)),
                    "not supported (stream_options takes include_usage; \
                     the padding of a stream cannot be turned off)",
                );
            }
            flag(options, "include_usage", "stream_options.include_usage")?
        }
        Some(_) => return refuse("stream_options".into(), "must be an object or null"),
    };
    let mut wire: WireRequest = serde_json::from_value(body)
        .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, unquoted(&e)))?;
    wire.x0ne.conversation = conversation;
    let max_tokens = match (wire.max_completion_tokens, wire.max_tokens) {
        (Some(a), Some(b)) if a != b => {
            return refuse(
                "max_completion_tokens".into(),
                "differs from max_tokens, its deprecated alias; send one",
            )
        }
        (Some(n), _) | (None, Some(n)) => n,
        (None, None) => default_max_tokens(),
    };
    if stream && stream_model_bytes(&wire.model) > MAX_STREAM_MODEL_BYTES {
        return refuse(
            "model".into(),
            &format!(
                "at most {MAX_STREAM_MODEL_BYTES} bytes once JSON-escaped with stream: true \
                 (every frame of the stream repeats it)"
            ),
        );
    }
    Ok(ChatRequest {
        model: wire.model,
        messages: wire.messages,
        max_tokens,
        temperature,
        top_p,
        seed,
        stop,
        json_schema,
        stream,
        include_usage,
        x0ne: wire.x0ne,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[cfg(test)]
mod tests {
    use super::{unquoted, WRONG_SHAPE};
    use serde::de::Error as _;

    /// A sentence [`unquoted`] does not know, or whose expectation carries
    /// a quote, is the fixed one: a failure is a value that reaches a 422.
    #[test]
    fn a_sentence_unquoted_does_not_know_is_the_fixed_one() {
        for said in [
            "s3cr3t",
            "invalid type: string \"s3cr3t\", expected \"s3cr3t\"",
            "invalid type: string \"s3cr3t\", expected `s3cr3t`",
            "invalid type: string \"s3cr3t\", expected ",
            "invalid sorts: string \"s3cr3t\", expected a string",
            "missing field `s3cr3t`",
        ] {
            assert_eq!(
                unquoted(&serde_json::Error::custom(said)),
                WRONG_SHAPE,
                "{said}"
            );
        }
        let said = "invalid type: text \"s3cr3t\", expected a string";
        assert_eq!(
            unquoted(&serde_json::Error::custom(said)),
            "invalid type, expected a string"
        );
    }
}
