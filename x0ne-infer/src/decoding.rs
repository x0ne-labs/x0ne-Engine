// SPDX-License-Identifier: Apache-2.0
//! What a request asks of the decoding (sampling, seed, `json_schema`) and the
//! check that a signed answer says it was done, text against schema included.
//! Run by the relay (`local_api.rs`) and by `x0ne-sdk` (`#[path]`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::json_schema::{quoted, Accepted, JsonSchema};

/// The sampling a receipt signs. Absent from a receipt: greedy decoding,
/// which draws nothing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedSampling {
    pub temperature: f64,
    pub top_p: f64,
    /// The seed of the draws: the caller's, or drawn by the TD when absent.
    pub seed: u64,
}

/// The sampling a request asks for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AskedSampling {
    pub temperature: f64,
    pub top_p: f64,
    /// `None`: the TD draws one and signs it.
    pub seed: Option<u64>,
}

/// What a request asks of the decoding. The default: greedy, unconstrained.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Asked {
    /// `None`: greedy (a `temperature` absent or 0).
    pub sampling: Option<AskedSampling>,
    pub json_schema: Option<JsonSchema>,
    /// The conversation it names, whose context the receipt must sign
    /// ([`Self::check_context`]).
    pub conversation: crate::context::AskedConversation,
}

/// OpenAI types `seed` int: a negative one keeps its bits.
pub fn read_seed(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_i64().map(|s| s as u64))
}

const JSON_SCHEMA_FIELDS: &[&str] = &["name", "schema", "strict", "description"];

/// `response_format`: absent, `null` or `text` asks nothing; `json_schema`
/// (`name`, `schema`, `strict`, `description`) asks for `schema`, compiled.
/// Anything else is refused; the reason quotes a name at most.
pub fn response_format(v: Option<&Value>) -> Result<Option<JsonSchema>, String> {
    let rf = match v {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(o)) => o,
        Some(_) => return Err("must be an object with a type".into()),
    };
    let extra = |allowed: &[&str]| {
        rf.keys()
            .find(|k| !allowed.contains(&k.as_str()))
            .map(|k| format!("`{}` is not supported here", quoted(k)))
    };
    match rf.get("type").and_then(Value::as_str) {
        Some("text") => extra(&["type"]).map_or(Ok(None), Err),
        Some("json_schema") => {
            if let Some(e) = extra(&["type", "json_schema"]) {
                return Err(e);
            }
            let Some(Value::Object(js)) = rf.get("json_schema") else {
                return Err("json_schema: an object with a name and a schema".into());
            };
            if let Some(k) = js
                .keys()
                .find(|k| !JSON_SCHEMA_FIELDS.contains(&k.as_str()))
            {
                return Err(format!("json_schema.{} is not supported", quoted(k)));
            }
            match js.get("name") {
                None => {}
                Some(Value::String(n))
                    if n.len() <= 64
                        && n.bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') => {}
                Some(_) => {
                    return Err("json_schema.name: up to 64 of a-z, A-Z, 0-9, _ and -".into())
                }
            }
            if !matches!(js.get("strict"), None | Some(Value::Null | Value::Bool(_))) {
                return Err("json_schema.strict: true, false or null".into());
            }
            if !matches!(js.get("description"), None | Some(Value::String(_))) {
                return Err("json_schema.description: a string".into());
            }
            let Some(schema) = js.get("schema") else {
                return Err(
                    "json_schema.schema: required; the engine enforces the schema \
                            it is given"
                        .into(),
                );
            };
            JsonSchema::compile(schema)
                .map(Some)
                .map_err(|e| format!("json_schema.{e}"))
        }
        Some("json_object") => Err("type json_object is not supported (send type \
                                    json_schema, with a schema)"
            .into()),
        Some(other) => Err(format!(
            "type `{}` is not supported (text or json_schema)",
            quoted(other)
        )),
        None => Err("type: text or json_schema".into()),
    }
}

impl Asked {
    /// From a chat request body, read as the TD reads it: `temperature`
    /// absent, `null` or 0 is greedy, `top_p` defaults to 1.
    pub fn from_body(body: &Value) -> Result<Self, String> {
        let number = |k: &str, default: f64| match body.get(k) {
            None | Some(Value::Null) => Ok(default),
            Some(v) => v.as_f64().ok_or_else(|| format!("{k}: not a number")),
        };
        let temperature = number("temperature", 0.0)?;
        let top_p = number("top_p", 1.0)?;
        let seed = match body.get("seed") {
            None | Some(Value::Null) => None,
            Some(v) => Some(read_seed(v).ok_or("seed: not an integer")?),
        };
        Ok(Self {
            sampling: (temperature > 0.0).then_some(AskedSampling {
                temperature,
                top_p,
                seed,
            }),
            json_schema: response_format(body.get("response_format"))
                .map_err(|e| format!("response_format: {e}"))?,
            conversation: crate::context::AskedConversation::from_x0ne(body.get("x0ne"))
                .map_err(|(field, why)| format!("x0ne.{field}: {why}"))?,
        })
    }

    /// Whether `signed`, a receipt's signed payload, signs the context the
    /// conversation asked implies, and the answer's `x0ne` block hands back
    /// the handle it commits to ([`crate::context::AskedConversation::check_signed`]).
    pub fn check_context(&self, signed: &Value, x0ne: &Value) -> Result<(), String> {
        self.conversation.check_signed(signed, x0ne)
    }

    /// Whether `signed`, a receipt's signed payload, says the decoding asked
    /// was done, and `text` holds to the schema: whole, or begun when the
    /// answer was cut at its token budget (`cut`, `finish_reason: "length"`).
    pub fn check_signed(&self, signed: &Value, text: &str, cut: bool) -> Result<(), String> {
        let sampling = match signed.get("sampling") {
            None => None,
            Some(v) => Some(
                serde_json::from_value::<SignedSampling>(v.clone())
                    .map_err(|e| format!("sampling: {e}"))?,
            ),
        };
        match (&self.sampling, sampling) {
            (None, None) => {}
            (None, Some(_)) => {
                return Err("sampling: the receipt signs draws; the request asked for \
                            greedy decoding"
                    .into())
            }
            (Some(_), None) => {
                return Err("sampling: the request asked to sample; the receipt signs \
                            greedy decoding"
                    .into())
            }
            (Some(a), Some(s)) => {
                if a.temperature != s.temperature || a.top_p != s.top_p {
                    return Err(
                        "sampling: the receipt signs other parameters than the request's".into(),
                    );
                }
                if a.seed.is_some_and(|seed| seed != s.seed) {
                    return Err(
                        "sampling: the receipt signs another seed than the request's".into(),
                    );
                }
            }
        }
        let digest = match signed.get("json_schema_sha256_hex") {
            None => None,
            Some(v) => Some(v.as_str().ok_or("json_schema_sha256_hex: not a string")?),
        };
        match (&self.json_schema, digest) {
            (None, None) => Ok(()),
            (None, Some(_)) => {
                Err("json_schema: the receipt signs a schema the request did not send".into())
            }
            (Some(_), None) => {
                Err("json_schema: the request sent a schema; the receipt signs none".into())
            }
            (Some(s), Some(d)) => {
                if d != s.sha256_hex() {
                    return Err("json_schema: the receipt signs another schema's digest".into());
                }
                match s.accepts(text) {
                    Accepted::Complete => Ok(()),
                    Accepted::Prefix if cut => Ok(()),
                    Accepted::Prefix => {
                        Err("json_schema: the answer stops before its document ends".into())
                    }
                    Accepted::No => Err("json_schema: the answer is not a document of the \
                                         schema"
                        .into()),
                }
            }
        }
    }
}

#[cfg(all(test, not(x0ne_sdk)))]
mod tests {
    use super::*;
    use serde_json::json;

    fn body(extra: Value) -> Value {
        let mut b = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        b.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        b
    }

    fn rf(schema: Value) -> Value {
        json!({"response_format": {"type": "json_schema",
            "json_schema": {"name": "answer", "strict": true, "schema": schema}}})
    }

    /// Every shape of `response_format` is read or refused by name; a
    /// failure means one is ignored, or a schema reaches the model unread.
    #[test]
    fn response_format_is_read_or_refused_by_name() {
        for absent in [
            json!({}),
            json!({"response_format": null}),
            json!({"response_format": {"type": "text"}}),
        ] {
            assert_eq!(Asked::from_body(&body(absent)).unwrap().json_schema, None);
        }
        let s = Asked::from_body(&body(rf(json!({"type": "boolean"}))))
            .unwrap()
            .json_schema
            .unwrap();
        assert_eq!(
            s.sha256_hex(),
            crate::json_schema::schema_sha256_hex(&json!({"type": "boolean"}))
        );
        for (format, word) in [
            (json!({"type": "json_object"}), "json_object"),
            (json!({"type": "xml"}), "xml"),
            (json!({"type": "text", "x": 1}), "`x`"),
            (json!("json"), "object"),
            (json!({"type": "json_schema"}), "json_schema"),
            (
                json!({"type": "json_schema", "json_schema": {"name": "a"}}),
                "json_schema.schema",
            ),
            (
                json!({"type": "json_schema", "json_schema": {"name": "a b", "schema": {}}}),
                "name",
            ),
            (
                json!({"type": "json_schema", "json_schema": {"schema": {"type": "null"}, "x": 1}}),
                "json_schema.x",
            ),
            (
                json!({"type": "json_schema", "json_schema": {"schema": {"type": "string", "format": "date"}}}),
                "format",
            ),
        ] {
            let e = Asked::from_body(&body(json!({"response_format": format}))).unwrap_err();
            assert!(
                e.starts_with("response_format: ") && e.contains(word),
                "{word}: {e}"
            );
        }
    }

    /// The check of a signed payload against the request: each lie a TD's
    /// receipt could tell about the decoding is refused by name.
    #[test]
    fn a_receipt_that_says_another_decoding_is_refused() {
        let schema = json!({"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]});
        let digest = crate::json_schema::schema_sha256_hex(&schema);
        let mut asked_body = rf(schema);
        asked_body["temperature"] = json!(0.7);
        asked_body["seed"] = json!(-1);
        let asked = Asked::from_body(&body(asked_body)).unwrap();
        let signed = |sampling: Value, digest: Value| {
            let mut s = json!({"model_digest_hex": "00"});
            if !sampling.is_null() {
                s["sampling"] = sampling;
            }
            if !digest.is_null() {
                s["json_schema_sha256_hex"] = digest;
            }
            s
        };
        let honest = signed(
            json!({"temperature": 0.7, "top_p": 1.0, "seed": u64::MAX}),
            json!(digest),
        );
        asked
            .check_signed(&honest, r#"{"ok":true}"#, false)
            .unwrap();
        asked.check_signed(&honest, r#"{"ok":tr"#, true).unwrap();
        for (lie, text, cut, word) in [
            (
                signed(
                    json!({"temperature": 0.7, "top_p": 1.0, "seed": 1}),
                    json!(digest),
                ),
                r#"{"ok":true}"#,
                false,
                "another seed",
            ),
            (
                signed(
                    json!({"temperature": 0.8, "top_p": 1.0, "seed": u64::MAX}),
                    json!(digest),
                ),
                r#"{"ok":true}"#,
                false,
                "other parameters",
            ),
            (
                signed(Value::Null, json!(digest)),
                r#"{"ok":true}"#,
                false,
                "greedy",
            ),
            (
                signed(
                    json!({"temperature": 0.7, "top_p": 1.0, "seed": u64::MAX, "x": 1}),
                    json!(digest),
                ),
                r#"{"ok":true}"#,
                false,
                "sampling",
            ),
            (
                signed(
                    json!({"temperature": 0.7, "top_p": 1.0, "seed": u64::MAX}),
                    json!("ab"),
                ),
                r#"{"ok":true}"#,
                false,
                "another schema",
            ),
            (
                signed(
                    json!({"temperature": 0.7, "top_p": 1.0, "seed": u64::MAX}),
                    Value::Null,
                ),
                r#"{"ok":true}"#,
                false,
                "signs none",
            ),
            (honest.clone(), r#"{"ok":1}"#, false, "not a document"),
            (honest.clone(), r#"{"ok":tr"#, false, "stops before"),
        ] {
            let e = asked.check_signed(&lie, text, cut).unwrap_err();
            assert!(e.contains(word), "{word}: {e}");
        }
        // Greedy and unconstrained: a receipt that signs either is refused.
        let plain = Asked::from_body(&body(json!({}))).unwrap();
        plain.check_signed(&json!({}), "anything", false).unwrap();
        assert!(plain
            .check_signed(&honest, r#"{"ok":true}"#, false)
            .is_err());
        assert!(plain
            .check_signed(&signed(Value::Null, json!(digest)), r#"{"ok":true}"#, false)
            .unwrap_err()
            .contains("did not send"));
    }
}
