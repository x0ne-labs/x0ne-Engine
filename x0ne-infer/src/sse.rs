//! A strict reader of the server-sent events x0ne's servers write: one
//! `data: ` line per event, ended by an empty line, and `:` comments.
//! Anything else is refused, never guessed: a second `data:` line, another
//! field, a carriage return, a line over
//! [`MAX_LINE_BYTES`](crate::sse::MAX_LINE_BYTES), JSON that does not parse
//! (lone surrogates included). Reasons are fixed: no byte read.
//!
//! And the writer of the unpadded events ([`Chunks`](crate::sse::Chunks)):
//! what the relay sends its local client. The attested server pads its own
//! (`crate::stream`).
//!
//! No log macro in this file: it handles the text of the answer.

use serde_json::{json, Value};

/// The longest line, comment or event, in bytes, its newline excluded.
pub const MAX_LINE_BYTES: usize = 64 * 1024;

/// Why a stream was refused, by a fixed reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct SseError(pub &'static str);

/// One event: `[DONE]`, or its JSON.
#[derive(Debug, Clone, PartialEq)]
pub enum Data {
    Done,
    Json(Value),
}

/// Fed bytes as they come, in pieces of any size.
#[derive(Debug, Default)]
pub struct SseReader {
    line: Vec<u8>,
    /// A `data:` line read, waiting for the empty line that ends its event.
    data: Option<String>,
    /// A byte was read since the last event ended.
    past_event: bool,
}

impl SseReader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads `bytes`, appending the events they complete to `out`, in order.
    pub fn push(&mut self, bytes: &[u8], out: &mut Vec<Data>) -> Result<(), SseError> {
        let mut rest = bytes;
        while !rest.is_empty() {
            match rest.iter().position(|&b| b == b'\n') {
                Some(at) => {
                    self.extend(&rest[..at])?;
                    self.end_line(out)?;
                    rest = &rest[at + 1..];
                }
                None => {
                    self.extend(rest)?;
                    break;
                }
            }
        }
        Ok(())
    }

    /// Nothing held: no partial line, no event without its empty line.
    pub fn at_boundary(&self) -> bool {
        self.line.is_empty() && self.data.is_none()
    }

    /// A byte, comment or newline included, was read after the last event.
    pub fn read_past_event(&self) -> bool {
        self.past_event
    }

    fn extend(&mut self, part: &[u8]) -> Result<(), SseError> {
        if part.contains(&b'\r') {
            return Err(SseError("a carriage return"));
        }
        if self.line.len() + part.len() > MAX_LINE_BYTES {
            return Err(SseError("a line over 64 KiB"));
        }
        self.past_event |= !part.is_empty();
        self.line.extend_from_slice(part);
        Ok(())
    }

    fn end_line(&mut self, out: &mut Vec<Data>) -> Result<(), SseError> {
        let line = std::mem::take(&mut self.line);
        if let Some(data) = self.data.take() {
            if !line.is_empty() {
                return Err(SseError("an event of more than one line"));
            }
            out.push(parse(&data)?);
            self.past_event = false;
            return Ok(());
        }
        self.past_event = true;
        if line.is_empty() || line[0] == b':' {
            return Ok(());
        }
        let Some(payload) = line.strip_prefix(b"data: ") else {
            return Err(SseError("a field other than data"));
        };
        let text = String::from_utf8(payload.to_vec()).map_err(|_| SseError("invalid UTF-8"))?;
        self.data = Some(text);
        Ok(())
    }
}

/// The event that ends a stream that went well.
pub const DONE: &str = "data: [DONE]\n\n";

/// An error event: `code` and a fixed `message`, never text of the answer.
pub fn error_event(code: &str, message: &str) -> String {
    let event = json!({ "error": { "message": message, "type": "x0ne_engine", "code": code } });
    format!("data: {event}\n\n")
}

/// The `chat.completion.chunk` events of one stream, unpadded. Each repeats
/// `id`, `created` and `model`; with `include_usage`, each but the usage
/// chunk carries `usage: null`, as OpenAI's do.
pub struct Chunks {
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
}

impl Chunks {
    pub fn new(id: &str, created: u64, model: &str, include_usage: bool) -> Self {
        Self {
            id: id.to_string(),
            created,
            model: model.to_string(),
            include_usage,
        }
    }

    pub fn include_usage(&self) -> bool {
        self.include_usage
    }

    /// One chunk: its `choices`, and `extra` as one more top-level field.
    pub fn chunk(&self, choices: Value, extra: Option<(&str, Value)>) -> String {
        let mut c = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": choices,
        });
        if self.include_usage {
            c["usage"] = Value::Null;
        }
        if let Some((key, value)) = extra {
            c[key] = value;
        }
        format!("data: {c}\n\n")
    }

    /// The first chunk: the assistant's role, with `content` (empty but in an
    /// emulated whole answer) and an `x0ne` block if given.
    pub fn role(&self, content: &str, x0ne: Option<Value>) -> String {
        self.chunk(
            json!([{ "index": 0, "delta": { "role": "assistant", "content": content }, "finish_reason": null }]),
            x0ne.map(|v| ("x0ne", v)),
        )
    }

    pub fn content(&self, text: &str) -> String {
        self.chunk(
            json!([{ "index": 0, "delta": { "content": text }, "finish_reason": null }]),
            None,
        )
    }

    /// The final chunk: an empty delta, why the answer ended, the `x0ne` block.
    pub fn finish(&self, reason: &str, x0ne: Value) -> String {
        self.chunk(
            json!([{ "index": 0, "delta": {}, "finish_reason": reason }]),
            Some(("x0ne", x0ne)),
        )
    }

    /// The usage chunk, when asked: no choice, the counts.
    pub fn usage(&self, prompt_tokens: u64, completion_tokens: u64) -> String {
        self.chunk(
            json!([]),
            Some((
                "usage",
                json!({
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }),
            )),
        )
    }
}

fn parse(data: &str) -> Result<Data, SseError> {
    if data == "[DONE]" {
        return Ok(Data::Done);
    }
    // serde_json refuses lone surrogates in \u escapes; its error quotes nothing here.
    serde_json::from_str(data)
        .map(Data::Json)
        .map_err(|_| SseError("an event that is not JSON"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_all(pieces: &[&[u8]]) -> Result<Vec<Data>, SseError> {
        let mut reader = SseReader::new();
        let mut out = Vec::new();
        for p in pieces {
            reader.push(p, &mut out)?;
        }
        Ok(out)
    }

    /// Events split anywhere, comments between them: the same events.
    #[test]
    fn events_split_at_any_byte_read_the_same() {
        let stream = b": pad   \n\ndata: {\"a\":\"\xc3\xa9\"}\n\n:\n\ndata: [DONE]\n\n";
        let whole = read_all(&[stream]).unwrap();
        assert_eq!(
            whole,
            vec![Data::Json(serde_json::json!({"a": "é"})), Data::Done]
        );
        for cut in 0..stream.len() {
            let (a, b) = stream.split_at(cut);
            assert_eq!(read_all(&[a, b]).unwrap(), whole, "cut at {cut}");
        }
        let bytes: Vec<&[u8]> = stream.chunks(1).collect();
        assert_eq!(read_all(&bytes).unwrap(), whole);
    }

    /// Each deviation is refused by its fixed reason, quoting nothing read.
    #[test]
    fn every_deviation_is_refused_by_a_fixed_reason() {
        let long = format!(":{}\n\n", " ".repeat(MAX_LINE_BYTES));
        let cases: [(&[u8], &str); 8] = [
            (long.as_bytes(), "a line over 64 KiB"),
            (b"data: {}\ndata: {}\n\n", "an event of more than one line"),
            (b"data: {}\n: c\n\n", "an event of more than one line"),
            (b"data: {}\r\n\r\n", "a carriage return"),
            (b"event: x\n\n", "a field other than data"),
            (b"data:{}\n\n", "a field other than data"),
            (b"data: \"\\ud800\"\n\n", "an event that is not JSON"),
            (b"data: \xff\n\n", "invalid UTF-8"),
        ];
        for (bytes, reason) in cases {
            let err = read_all(&[bytes]).unwrap_err();
            assert_eq!(
                err,
                SseError(reason),
                "{:?}",
                String::from_utf8_lossy(bytes)
            );
        }
        let exact = format!(":{}\n\n", " ".repeat(MAX_LINE_BYTES - 1));
        assert!(read_all(&[exact.as_bytes()]).unwrap().is_empty());
    }

    /// A failure means the writer's events no longer read back through the
    /// strict reader, or a chunk leaves OpenAI's shape.
    #[test]
    fn the_writer_writes_what_the_reader_reads() {
        for include_usage in [false, true] {
            let c = Chunks::new("chatcmpl-0", 7, "m\"\n", include_usage);
            let x0ne = serde_json::json!({"confidential": false});
            let stream = [
                c.role("", Some(x0ne.clone())),
                c.content("a\u{1}\"\\\n😀"),
                c.finish("stop", x0ne.clone()),
                c.usage(3, 4),
                DONE.to_string(),
                error_event("generation_failed", "generation failed"),
            ]
            .concat();
            let events = read_all(&[stream.as_bytes()]).unwrap();
            assert_eq!(events.len(), 6);
            let Data::Json(role) = &events[0] else {
                panic!()
            };
            assert_eq!(role["model"], "m\"\n");
            assert_eq!(role["object"], "chat.completion.chunk");
            assert_eq!(role["x0ne"], x0ne);
            assert_eq!(role.get("usage"), include_usage.then_some(&Value::Null));
            let Data::Json(text) = &events[1] else {
                panic!()
            };
            assert_eq!(text["choices"][0]["delta"]["content"], "a\u{1}\"\\\n😀");
            let Data::Json(usage) = &events[3] else {
                panic!()
            };
            assert_eq!(usage["usage"]["total_tokens"], 7);
            assert_eq!(usage["choices"], serde_json::json!([]));
            assert_eq!(events[4], Data::Done);
            let Data::Json(error) = &events[5] else {
                panic!()
            };
            assert_eq!(error["error"]["code"], "generation_failed");
        }
    }

    /// Where a body may end, and whether bytes followed the last event.
    #[test]
    fn the_reader_says_whether_it_holds_a_partial_event() {
        let mut r = SseReader::new();
        let mut out = Vec::new();
        r.push(b"data: [DONE]", &mut out).unwrap();
        assert!(!r.at_boundary());
        r.push(b"\n", &mut out).unwrap();
        assert!(!r.at_boundary() && out.is_empty(), "no empty line yet");
        r.push(b"\n", &mut out).unwrap();
        assert!(r.at_boundary() && !r.read_past_event());
        assert_eq!(out, vec![Data::Done]);
        r.push(b":", &mut out).unwrap();
        assert!(!r.at_boundary() && r.read_past_event());
        r.push(b"\n\n", &mut out).unwrap();
        assert!(r.at_boundary() && r.read_past_event());
    }
}
