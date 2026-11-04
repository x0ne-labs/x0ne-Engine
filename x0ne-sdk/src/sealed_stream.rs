//! The streamed answer of the caller-sealed path (docs/sdk-e2e.md, step 6):
//! chunked Oblivious HTTP (draft-ietf-ohai-chunked-ohttp-08, [`crate::ohttp`])
//! at a cadence the session config fixes ([`E2eStream`]), so that what a
//! carrier sees does not follow the answer's tokens:
//!
//! - every chunk but the last carries exactly `chunk_bytes` (L) bytes of
//!   plaintext, the last [`FINAL_CHUNK_BYTES`]; the TD writes one every
//!   `tick_ms` (T), the first T after it read the request, tokens or not;
//! - their number, the last included, is a power of two, [`MIN_CHUNKS`] at
//!   least and [`max_chunks`] at most: success, refusal and failure alike;
//! - the plaintext is one Binary HTTP response of indeterminate length (RFC
//!   9292 §3.2), as the draft requires: a 200 whose content is OpenAI's SSE
//!   chunks, or a refusal whose content is its JSON error. Content that does
//!   not fill a chunk is followed by line feeds, blank lines SSE and JSON
//!   both skip, and only after a whole event; the last chunk ends the
//!   content and the trailers, then zeros (§3.8).
//!
//! [`PendingStream`] checks all of it as the bytes come: a chunk of another
//! size, out of order, replayed, altered, or a stream without its final
//! chunk (CVE-2026-48480: a cut stream read as whole) is refused. The
//! deltas it hands out before [`PendingStream::finish`] are not verified,
//! and their type says so ([`UnverifiedDelta`]): the receipt, in the final
//! chunk, is checked over the whole text there.

use crate::attest::{CloudConfig, E2eStream};
use crate::bhttp::{self, Part};
use crate::e2e::E2eError;
use crate::ohttp::{self, ChunkedResponseReader};
use crate::receipt::{self, Checked};
use serde_json::Value;

/// Chunks of every stream, the final included, at least: 1.6 s at T = 100 ms.
pub const MIN_CHUNKS: u64 = 16;
/// Plaintext of the final chunk: the data every chunked OHTTP receiver takes
/// in one chunk (draft-08 §3). The x0ne block of a real H100 answer was
/// 7322 bytes (2026-10-17).
pub const FINAL_CHUNK_BYTES: usize = ohttp::MAX_CHUNK_DATA;
/// The longest stream: the relay's 15 minutes, as `padding::MAX_TICKS`
/// ticks of 100 ms (x0ne-infer).
pub const MAX_STREAM_MS: u64 = 870_400;
/// The SSE line the reader holds at most: a final event is under 16 KiB.
const MAX_LINE: usize = 1 << 20;
/// Content of one stream, at most.
const MAX_CONTENT: usize = bhttp::MAX_CONTENT_BYTES;

/// The session's cadence: its config's, or [`E2eStream::DEFAULT`].
pub fn cadence(config: &CloudConfig) -> Result<E2eStream, String> {
    let c = config.e2e_stream.unwrap_or(E2eStream::DEFAULT);
    c.check()?;
    Ok(c)
}

/// The most chunks a stream of cadence `c` may take: the largest power of
/// two whose chunks all leave within [`MAX_STREAM_MS`].
pub fn max_chunks(c: &E2eStream) -> u64 {
    let fit = MAX_STREAM_MS / u64::from(c.tick_ms.max(1));
    if fit < MIN_CHUNKS {
        return MIN_CHUNKS;
    }
    1 << fit.ilog2()
}

/// The most bytes a stream of cadence `c` puts on the wire, nonce, lengths
/// and tags included: [`max_chunks`] less one chunks of `chunk_bytes`, then
/// the final one. A carrier that opens nothing still holds a stream to it.
pub fn max_wire_bytes(c: &E2eStream) -> usize {
    let chunk = 2 + c.chunk_bytes as usize + ohttp::CHUNK_TAG;
    let last = 1 + FINAL_CHUNK_BYTES + ohttp::CHUNK_TAG;
    let before_last = usize::try_from(max_chunks(c) - 1).unwrap_or(usize::MAX);
    ohttp::RESPONSE_NONCE_LEN + before_last * chunk + last
}

/// The most chunks a stream asked at `now` (Unix seconds) may take in a
/// session ending at `ends`: the largest power of two whose last chunk
/// leaves within [`MAX_STREAM_MS`] and before `ends` less
/// [`crate::attest::SESSION_EXPIRY_GUARD_SECS`]; `None` below [`MIN_CHUNKS`].
/// Chunk `k` leaves `(k + 1)` ticks after the request; one tick of margin.
/// The TD bounds its streams by it, the caller its reading.
pub fn cap(c: &E2eStream, now: u64, ends: u64) -> Option<u64> {
    let left_ms = ends
        .saturating_sub(crate::attest::SESSION_EXPIRY_GUARD_SECS)
        .saturating_sub(now)
        .saturating_mul(1000);
    let fit = (left_ms / u64::from(c.tick_ms.max(1))).saturating_sub(1);
    let fit = fit.min(max_chunks(c));
    (fit >= MIN_CHUNKS).then(|| 1 << fit.ilog2())
}

/// A count of chunks a stream may end at: a power of two in
/// [`MIN_CHUNKS`]..=`max`.
pub fn may_end_at(chunks: u64, max: u64) -> bool {
    chunks >= MIN_CHUNKS && chunks <= max && chunks.is_power_of_two()
}

/// The text of one delta as the stream brought it, NOT VERIFIED: the
/// stream may still be refused, its receipt checked only by
/// [`PendingStream::finish`]. Only the text of `finish`'s receipt is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnverifiedDelta(String);

impl UnverifiedDelta {
    /// The text, unverified.
    #[must_use]
    pub fn unverified_text(&self) -> &str {
        &self.0
    }

    /// The text, unverified.
    #[must_use]
    pub fn into_unverified_text(self) -> String {
        self.0
    }
}

/// One streamed request in flight: reads its own response and no other.
pub struct PendingStream {
    /// Taken by [`Self::finish`].
    reader: Option<ChunkedResponseReader>,
    chunk_bytes: usize,
    max_chunks: u64,
    response: bhttp::ResponseReader,
    session_pubkey: [u8; 32],
    strict_tcb: Option<crate::tcb_disclosure::TcbDisclosure>,
    /// The decoding the question asked, which the final receipt must sign;
    /// else why it could not be read. The question itself is not kept.
    asked: Result<crate::decoding::Asked, String>,
    head: Option<(u16, bool)>,
    line: Vec<u8>,
    data: Option<String>,
    body: Vec<u8>,
    text: String,
    finish_reason: Option<String>,
    last: Option<Value>,
    usage: Option<Value>,
    error: Option<Value>,
    done: bool,
    broken: bool,
}

/// What a stream came to, decrypted.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamAnswer {
    /// The TD's status: 200 once the stream began, else the refusal's.
    pub status: u16,
    /// Every delta, in order.
    pub text: String,
    /// The deltas the final chunk carried, which [`PendingStream::push`]
    /// never handed out: `text` is what `push` handed out, then these.
    /// Verified with `text`, when `finish` returns a receipt.
    pub final_deltas: Vec<String>,
    pub finish_reason: Option<String>,
    /// The usage chunk, when the request asked for it.
    pub usage: Option<Value>,
    /// The `error` object: a 200's error event, or a refusal's body's.
    pub error: Option<Value>,
    /// A refusal's JSON body, blanks trimmed; empty for a 200.
    pub body: Vec<u8>,
    /// Chunks read, the final included.
    pub chunks: u64,
    /// The next turn's handle, when the question named a conversation
    /// (`x0ne.conversation`, docs/api.md): the final chunk's, which
    /// [`PendingStream::finish`] checked against the receipt's commitment.
    pub conversation: Option<crate::context::Issued>,
}

fn refused(m: impl Into<String>) -> E2eError {
    E2eError::Stream(m.into())
}

impl PendingStream {
    /// The reader of `response` at `cadence`, asked at `now` in a session
    /// ending at `ends` ([`cap`]); refused if no stream fits before it.
    pub(crate) fn new(
        response: ohttp::ChunkedClientResponse,
        cadence: E2eStream,
        (now, ends): (u64, u64),
        session_pubkey: [u8; 32],
        strict_tcb: Option<crate::tcb_disclosure::TcbDisclosure>,
        asked: Result<crate::decoding::Asked, String>,
    ) -> Result<Self, E2eError> {
        let max_chunks = cap(&cadence, now, ends).ok_or_else(|| {
            E2eError::Refused(format!(
                "the TD's session ends at {ends}: a stream asked at {now} could not end {}s before it",
                crate::attest::SESSION_EXPIRY_GUARD_SECS
            ))
        })?;
        Ok(Self {
            reader: Some(response.reader()),
            chunk_bytes: cadence.chunk_bytes as usize,
            max_chunks,
            response: bhttp::ResponseReader::new(MAX_CONTENT),
            session_pubkey,
            strict_tcb,
            asked,
            head: None,
            line: Vec::new(),
            data: None,
            body: Vec::new(),
            text: String::new(),
            finish_reason: None,
            last: None,
            usage: None,
            error: None,
            done: false,
            broken: false,
        })
    }

    /// `bytes` as they came off the transport: every delta they complete,
    /// NOT VERIFIED until [`Self::finish`] checks the receipt.
    /// After an error, every call is refused.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<UnverifiedDelta>, E2eError> {
        if self.broken {
            return Err(refused("the stream was refused already"));
        }
        let read = self.read(bytes);
        self.broken = read.is_err();
        Ok(read?.into_iter().map(UnverifiedDelta).collect())
    }

    fn read(&mut self, bytes: &[u8]) -> Result<Vec<String>, E2eError> {
        let mut deltas = Vec::new();
        let reader = self.reader.as_mut().ok_or_else(|| refused("finished"))?;
        let chunks = reader.push(bytes)?;
        let read = reader.chunks_read();
        if read >= self.max_chunks {
            // The final chunk is one more: it must still land on a size.
            return Err(refused(format!("more than {} chunks", self.max_chunks)));
        }
        for chunk in chunks {
            if chunk.len() != self.chunk_bytes {
                return Err(refused(format!(
                    "a chunk of {} bytes, not the session's {}",
                    chunk.len(),
                    self.chunk_bytes
                )));
            }
            self.plaintext(&chunk, &mut deltas)?;
        }
        Ok(deltas)
    }

    fn plaintext(&mut self, pt: &[u8], deltas: &mut Vec<String>) -> Result<(), E2eError> {
        for part in self.response.push(pt)? {
            match part {
                Part::Head { status, headers } => {
                    let ctype = headers
                        .iter()
                        .find(|(n, _)| n.eq_ignore_ascii_case("content-type"))
                        .map(|(_, v)| v.as_str());
                    let want = if status == 200 {
                        "text/event-stream"
                    } else {
                        "application/json"
                    };
                    if ctype != Some(want) {
                        return Err(refused(format!("a {status} whose content is not {want}")));
                    }
                    self.head = Some((status, status == 200));
                }
                Part::Content(bytes) => match self.head {
                    Some((_, true)) => self.sse(&bytes, deltas)?,
                    _ => self.body.extend_from_slice(&bytes),
                },
            }
        }
        Ok(())
    }

    /// SSE (WHATWG HTML §9.2.6), as the TD writes it: `data:` lines, blank
    /// lines, comments; other fields are skipped, as the standard says.
    fn sse(&mut self, bytes: &[u8], deltas: &mut Vec<String>) -> Result<(), E2eError> {
        for b in bytes {
            if *b != b'\n' {
                if self.line.len() == MAX_LINE {
                    return Err(refused("an SSE line over 1 MiB"));
                }
                self.line.push(*b);
                continue;
            }
            let mut line = std::mem::take(&mut self.line);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line =
                String::from_utf8(line).map_err(|_| refused("an SSE line that is not UTF-8"))?;
            if line.is_empty() {
                if let Some(data) = self.data.take() {
                    self.event(&data, deltas)?;
                }
            } else if let Some(value) = line.strip_prefix("data:") {
                let value = value.strip_prefix(' ').unwrap_or(value);
                match &mut self.data {
                    Some(d) => {
                        d.push('\n');
                        d.push_str(value);
                    }
                    None => self.data = Some(value.to_string()),
                }
            }
        }
        Ok(())
    }

    fn event(&mut self, data: &str, deltas: &mut Vec<String>) -> Result<(), E2eError> {
        if self.done {
            return Err(refused("an event after [DONE]"));
        }
        if data == "[DONE]" {
            if self.error.is_some() || self.last.is_none() {
                return Err(refused("[DONE] without the final chunk"));
            }
            self.done = true;
            return Ok(());
        }
        if self.error.is_some() {
            return Err(refused("an event after the error event"));
        }
        let v: Value =
            serde_json::from_str(data).map_err(|_| refused("an event that is not JSON"))?;
        if let Some(error) = v.get("error") {
            self.error = Some(error.clone());
            return Ok(());
        }
        let choices = v
            .get("choices")
            .and_then(Value::as_array)
            .ok_or_else(|| refused("an event without choices"))?;
        match choices.as_slice() {
            [] => {
                let usage = v.get("usage").ok_or_else(|| refused("an empty chunk"))?;
                if self.last.is_none() || self.usage.is_some() {
                    return Err(refused("a usage chunk out of place"));
                }
                self.usage = Some(usage.clone());
            }
            [choice] => {
                if self.last.is_some() {
                    return Err(refused("a chunk after the final chunk"));
                }
                if let Some(text) = choice.pointer("/delta/content").and_then(Value::as_str) {
                    if !text.is_empty() {
                        self.text.push_str(text);
                        deltas.push(text.to_string());
                    }
                }
                match choice.get("finish_reason") {
                    None | Some(Value::Null) => {}
                    Some(Value::String(reason)) => {
                        self.finish_reason = Some(reason.clone());
                        self.last = Some(v.clone());
                    }
                    Some(_) => return Err(refused("finish_reason is not a string")),
                }
            }
            _ => {
                return Err(refused(format!(
                    "{} choices: one is committed",
                    choices.len()
                )))
            }
        }
        Ok(())
    }

    /// The body ended. Whole only if: its final chunk came, of
    /// [`FINAL_CHUNK_BYTES`], at a count of chunks [`may_end_at`] allows,
    /// and ended the Binary HTTP response, zeros after. Then, for a 200
    /// without an error event: `[DONE]` came, and the receipt of the final
    /// chunk checks over the whole text, as
    /// [`crate::e2e::Pending::open_verified`] checks a whole answer's (the
    /// model the caller names, the attested key, the decoding the question
    /// asked, the TCB under `strict` at `now`). A 200 with an error event, or a refusal, comes back with no
    /// receipt.
    pub fn finish(
        mut self,
        model_digest_hex: &str,
        now: u64,
    ) -> Result<(StreamAnswer, Option<Checked>), E2eError> {
        if self.broken {
            return Err(refused("the stream was refused already"));
        }
        let reader = self.reader.take().ok_or_else(|| refused("finished"))?;
        let chunks = reader.chunks_read() + 1;
        let last = reader.finish()?;
        if last.len() != FINAL_CHUNK_BYTES {
            return Err(refused(format!(
                "a final chunk of {} bytes, not {FINAL_CHUNK_BYTES}",
                last.len()
            )));
        }
        if !may_end_at(chunks, self.max_chunks) {
            return Err(refused(format!(
                "{chunks} chunks: not a power of two in {MIN_CHUNKS}..={}",
                self.max_chunks
            )));
        }
        let mut deltas = Vec::new();
        self.plaintext(&last, &mut deltas)?;
        self.response.finish()?;
        let Some((status, sse)) = self.head else {
            return Err(refused("no response head"));
        };
        if !sse {
            let body = self.body.trim_ascii().to_vec();
            let v: Value =
                serde_json::from_slice(&body).map_err(|_| refused("a refusal that is not JSON"))?;
            return Ok((
                StreamAnswer {
                    status,
                    text: String::new(),
                    final_deltas: Vec::new(),
                    finish_reason: None,
                    usage: None,
                    error: v.get("error").cloned(),
                    body,
                    chunks,
                    conversation: None,
                },
                None,
            ));
        }
        if !self.line.is_empty() || self.data.is_some() {
            return Err(refused("the content ended inside an event"));
        }
        let mut answer = StreamAnswer {
            status,
            text: std::mem::take(&mut self.text),
            final_deltas: deltas,
            finish_reason: self.finish_reason.take(),
            usage: self.usage.take(),
            error: self.error.take(),
            body: Vec::new(),
            chunks,
            conversation: None,
        };
        if answer.error.is_some() {
            return Ok((answer, None));
        }
        let last = match (&self.last, self.done) {
            (Some(last), true) => last,
            _ => {
                return Err(refused(
                    "the stream ended without its final chunk and [DONE]",
                ))
            }
        };
        let checked = receipt::check_text(
            &answer.text,
            last,
            &self.session_pubkey,
            Some(model_digest_hex),
        )
        .map_err(E2eError::Receipt)?;
        // As for a whole answer (`receipt::check_answer_for`): finish_reason
        // is not signed, `length` admits a begun document, never another one.
        let asked = self
            .asked
            .as_ref()
            .map_err(|e| E2eError::Receipt(format!("decoding: the request: {e}")))?;
        let cut = answer.finish_reason.as_deref() == Some("length");
        asked
            .check_signed(&checked.signed, &checked.text, cut)
            .map_err(|e| E2eError::Receipt(format!("decoding: {e}")))?;
        // The context the question's conversation implies, and the handle
        // the final chunk hands back, as for a whole answer
        // (`receipt::check_answer_for`).
        asked
            .check_context(&checked.signed, &last["x0ne"])
            .map_err(|e| E2eError::Receipt(format!("conversation: {e}")))?;
        crate::e2e::strict_check(self.strict_tcb.as_ref(), &checked, now)?;
        answer.conversation = last["x0ne"]
            .get("conversation")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()
            .map_err(|_| E2eError::Receipt("conversation: a malformed handle".into()))?;
        Ok((answer, Some(checked)))
    }
}

#[cfg(feature = "test-anchor")]
impl PendingStream {
    /// A reader of `response` to the question `body` under `session_pubkey`,
    /// asked at `now` in a session ending at `ends`, under `strict` if
    /// given, with no attested key: this crate's tests only (`test-anchor`,
    /// a dev-dependency feature).
    pub fn for_tests(
        response: ohttp::ChunkedClientResponse,
        cadence: E2eStream,
        session_pubkey: [u8; 32],
        body: &[u8],
        now: u64,
        ends: u64,
        strict: Option<crate::tcb_disclosure::TcbDisclosure>,
    ) -> Result<Self, E2eError> {
        Self::new(
            response,
            cadence,
            (now, ends),
            session_pubkey,
            strict,
            crate::e2e::asked_of(body),
        )
    }
}

// ── The TD's side: the plaintext of each chunk ──────────────────────────────

/// Writes the plaintext of a stream's chunks, the TD's side: the response's
/// head first, then one content chunk per chunk, filled with queued bytes,
/// then line feeds; the last one ends the content, then zeros. Bytes enter
/// the queue only as whole events (or a whole JSON body), so a line feed
/// never lands inside one: a chunk with less than its room queued takes it
/// all and pads; one with more takes exactly its room and no padding.
pub struct Framer {
    chunk_bytes: usize,
    head: Option<Vec<u8>>,
    queue: std::collections::VecDeque<u8>,
}

/// Blank-line padding: SSE skips it between events, JSON around a value.
pub const FILL: u8 = b'\n';

/// The data length `n` whose content chunk, length included, is `room` bytes.
fn data_for(room: usize) -> Option<usize> {
    [1usize, 2, 4, 8].into_iter().find_map(|len| {
        let n = room.checked_sub(len)?;
        (n > 0 && bhttp::indeterminate::chunk_len(n) == room).then_some(n)
    })
}

impl Framer {
    /// A response of `status` whose content is `content_type`, plus
    /// `headers`, cut into chunks of the cadence's size.
    pub fn new(
        cadence: E2eStream,
        status: u16,
        content_type: &str,
        headers: &[(&str, &str)],
    ) -> Self {
        let mut fields = vec![("content-type", content_type)];
        fields.extend_from_slice(headers);
        Self {
            chunk_bytes: cadence.chunk_bytes as usize,
            head: Some(bhttp::indeterminate::head(status, &fields)),
            queue: std::collections::VecDeque::new(),
        }
    }

    /// One whole event, or one whole JSON body.
    pub fn push(&mut self, whole: &[u8]) {
        self.queue.extend(whole);
    }

    /// Bytes queued and not yet written.
    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    /// Content bytes the next non-final chunk carries.
    pub fn room(&self) -> usize {
        let head = self.head.as_ref().map_or(0, Vec::len);
        data_for(self.chunk_bytes.saturating_sub(head)).unwrap_or(0)
    }

    /// The next non-final chunk's plaintext: exactly `chunk_bytes` bytes.
    pub fn chunk(&mut self) -> Result<Vec<u8>, E2eError> {
        let mut out = self.head.take().unwrap_or_default();
        let n = data_for(self.chunk_bytes.saturating_sub(out.len()))
            .ok_or_else(|| refused("a chunk too small for its head"))?;
        let take = n.min(self.queue.len());
        let mut data: Vec<u8> = self.queue.drain(..take).collect();
        data.resize(n, FILL);
        bhttp::indeterminate::chunk(&mut out, &data);
        debug_assert_eq!(out.len(), self.chunk_bytes);
        Ok(out)
    }

    /// Whether the queue and `tail` fit the final chunk together.
    pub fn fits_last(&self, tail: usize) -> bool {
        let head = self.head.as_ref().map_or(0, Vec::len);
        let content = self.queue.len() + tail;
        let framed = if content == 0 {
            0
        } else {
            bhttp::indeterminate::chunk_len(content)
        };
        head + framed + bhttp::indeterminate::END.len() <= FINAL_CHUNK_BYTES
    }

    /// The final chunk's plaintext, [`FINAL_CHUNK_BYTES`]: the queue, then
    /// `tail`, then the end of the content and trailers, then zeros.
    pub fn last(mut self, tail: &[u8]) -> Result<Vec<u8>, E2eError> {
        if !self.fits_last(tail.len()) {
            return Err(refused("the final chunk cannot hold what is left"));
        }
        let mut out = self.head.take().unwrap_or_default();
        let mut content: Vec<u8> = self.queue.drain(..).collect();
        content.extend_from_slice(tail);
        if !content.is_empty() {
            bhttp::indeterminate::chunk(&mut out, &content);
        }
        out.extend_from_slice(&bhttp::indeterminate::END);
        out.resize(FINAL_CHUNK_BYTES, 0);
        Ok(out)
    }
}
