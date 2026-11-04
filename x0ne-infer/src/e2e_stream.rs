//! The TD's side of a caller-sealed stream (docs/sdk-e2e.md, step 6): the
//! chunked OHTTP response (draft-ietf-ohai-chunked-ohttp-08) a carrier sees
//! as chunks of one size, one per tick of the session's cadence
//! (`x0ne_sdk::attest::E2eStream`, L bytes every T), the first T after the
//! request, whatever the text, the model's speed or the outcome:
//!
//! - every chunk carries L bytes of plaintext (`x0ne_sdk::sealed_stream::Framer`):
//!   SSE events queued whole, at most one new event of text per tick and
//!   only while the queue is under a chunk's room, then line feeds; a
//!   generation faster than a chunk per tick waits in the queue, unseen;
//! - the last chunk, of 16384 bytes, goes at the first count of chunks that
//!   is a power of two, 16 at least, once the answer is out: the receipt
//!   and `[DONE]`, an error event, or a refusal's JSON error. Refused before
//!   the model (a bad request, a full queue, a session about to end), the
//!   answer is a status and its JSON error at 16 chunks; failed after it
//!   was admitted, an error event in a 200, at the next size;
//! - a stream is [`cap`](crate::e2e_stream::cap) chunks at most: past the 15 minutes the relay
//!   waits, or past the session's end less the relay's 60 s, it ends at its
//!   cap with an error event, its generation cancelled: no receipt is ever
//!   signed for a stream past the session's end.
//!
//! What stays visible to the host: the time the request came, the size
//! bucket of the question, the count of chunks (a power of two, so the
//! answer's duration to within a factor of two, its length too when the
//! text sets the count), the model's turn, which ends at that same count
//! of ticks ([`StreamTurn`](crate::e2e_stream::StreamTurn)), and a reader
//! that leaves.
//!
//! No log macro in this file: it handles the text of the answer.

use std::pin::Pin;
use std::task::{Context, Poll};

use hyper::body::{Body, Bytes, Frame};
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::time::Instant;
use x0ne_sdk::attest::E2eStream;
use x0ne_sdk::ohttp::ChunkedResponseWriter;
use x0ne_sdk::sealed_stream::{self, Framer, MIN_CHUNKS};

use crate::openai::error::ApiError;
use crate::stream::{Broken, Cadence, Event, Finale, Shape, INTERNAL, STALL, STALLED, TOO_LONG};

/// How a caller-sealed stream starts.
pub enum Start {
    /// Refused before the model: a status and its JSON error.
    Refused(ApiError),
    /// Admitted: the generation's events and the stream's shape.
    Running {
        rx: UnboundedReceiver<Event>,
        shape: Shape,
    },
}

/// The session's cadence in the SDK's type: the config's, or the default.
pub fn cadence_of(config: &crate::cloud_config::CloudConfig) -> E2eStream {
    config.e2e_stream.map_or(E2eStream::DEFAULT, |c| E2eStream {
        chunk_bytes: c.chunk_bytes,
        tick_ms: c.tick_ms,
    })
}

/// The most chunks a stream may take before its session's end: the SDK's
/// bound, which the caller reads it under.
pub use x0ne_sdk::sealed_stream::cap;

/// What a sealed stream's turn ends by: its request's instant, its tick,
/// its cap of chunks.
#[derive(Clone, Copy, Debug)]
pub struct StreamTurn {
    pub asked: Instant,
    pub tick: std::time::Duration,
    pub cap: u64,
}

impl StreamTurn {
    /// The turn's end, the generation done at `done`: the first count of
    /// ticks after the request that is a power of two, [`MIN_CHUNKS`] at
    /// least, as a count of chunks; past the cap, `done`. The queue place
    /// goes back then: a host that probes the queue reads the generation's
    /// time to a factor of two, as from the chunks, not to a Padme size.
    pub fn end(&self, done: Instant) -> Instant {
        let tick_ns = self.tick.as_nanos().max(1);
        let ticks = done
            .saturating_duration_since(self.asked)
            .as_nanos()
            .div_ceil(tick_ns);
        let ticks = u64::try_from(ticks).unwrap_or(u64::MAX).max(MIN_CHUNKS);
        match ticks.checked_next_power_of_two() {
            Some(n) if n <= self.cap => {
                self.asked
                    + self
                        .tick
                        .saturating_mul(u32::try_from(n).unwrap_or(u32::MAX))
            }
            _ => done,
        }
    }
}

fn json(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())
}

/// OpenAI's SSE chunks of one stream, unpadded: the chunk pads.
struct Events {
    /// `data: {"id":…,"model":…,"choices":[{"index":0,"delta":`
    head: String,
    /// `data: {"id":…,"model":…,"choices":[],"usage":` when asked.
    usage_head: Option<String>,
}

impl Events {
    fn new(shape: &Shape) -> Self {
        let common = format!(
            "data: {{\"id\":{},\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":{},\"choices\":",
            json(&shape.id),
            shape.created,
            json(&shape.model)
        );
        Self {
            head: format!("{common}[{{\"index\":0,\"delta\":"),
            usage_head: shape
                .include_usage
                .then(|| format!("{common}[],\"usage\":")),
        }
    }

    fn role(&self) -> String {
        format!(
            "{}{{\"role\":\"assistant\",\"content\":\"\"}},\"finish_reason\":null}}]}}\n\n",
            self.head
        )
    }

    fn content(&self, text: &str) -> String {
        format!(
            "{}{{\"content\":{}}},\"finish_reason\":null}}]}}\n\n",
            self.head,
            json(text)
        )
    }

    /// The final chunk with its receipt, the usage chunk if asked, `[DONE]`.
    fn last(&self, finale: &Finale) -> String {
        let mut out = format!(
            "{}{{}},\"finish_reason\":{}}}],\"x0ne\":{}}}\n\n",
            self.head,
            json(finale.finish_reason),
            finale.x0ne_json
        );
        if let Some(usage_head) = &self.usage_head {
            let (p, c) = (finale.prompt_tokens, finale.completion_tokens);
            out.push_str(&format!(
                "{usage_head}{{\"prompt_tokens\":{p},\"completion_tokens\":{c},\"total_tokens\":{}}}}}\n\n",
                u64::from(p) + u64::from(c)
            ));
        }
        out.push_str("data: [DONE]\n\n");
        out
    }
}

/// The error event that ends a 200: `message`, and `status` for a refusal
/// the model made once admitted (a prompt past its context).
fn error_event(message: &str, status: Option<u16>) -> Vec<u8> {
    let status = status.map_or(String::new(), |s| format!(",\"status\":{s}"));
    format!(
        "data: {{\"error\":{{\"message\":{},\"type\":\"x0ne_engine\",\"code\":\"generation_failed\"{status}}}}}\n\n",
        json(message)
    )
    .into_bytes()
}

/// The body of one caller-sealed stream, read by hyper one chunk at a time.
pub struct SealedStream<C: Cadence> {
    cadence: C,
    tick: E2eStream,
    writer: Option<ChunkedResponseWriter>,
    framer: Option<Framer>,
    events: Option<Events>,
    rx: Option<UnboundedReceiver<Event>>,
    pending: String,
    finale: Option<Box<Finale>>,
    /// What ends the stream instead of its receipt: an error event, or
    /// nothing after a refusal's body.
    failure: Option<Vec<u8>>,
    started: bool,
    heard: Option<Instant>,
    /// Chunks built so far, the one being built included.
    built: u64,
    cap: u64,
    prepared: Option<(Bytes, bool)>,
    ended: bool,
    _place: Option<tokio::sync::OwnedSemaphorePermit>,
}

impl<C: Cadence> SealedStream<C> {
    /// `start` answered at cadence `tick`, `cap` chunks at most (a power of
    /// two, [`MIN_CHUNKS`] at least), sealed by `writer`, its chunks timed
    /// by `cadence`.
    pub fn new(
        start: Start,
        tick: E2eStream,
        cap: u64,
        writer: ChunkedResponseWriter,
        cadence: C,
    ) -> Self {
        let mut body = Self {
            cadence,
            tick,
            writer: Some(writer),
            framer: None,
            events: None,
            rx: None,
            pending: String::new(),
            finale: None,
            failure: None,
            started: false,
            heard: None,
            built: 0,
            cap: cap.max(MIN_CHUNKS),
            prepared: None,
            ended: false,
            _place: None,
        };
        match start {
            Start::Refused(e) => {
                let retry = e.retry_after.map(|s| s.to_string());
                let headers: Vec<(&str, &str)> =
                    retry.iter().map(|s| ("retry-after", s.as_str())).collect();
                let mut framer = Framer::new(tick, e.status.as_u16(), "application/json", &headers);
                framer.push(e.body().to_string().as_bytes());
                body.framer = Some(framer);
                body.failure = Some(Vec::new());
            }
            Start::Running { rx, shape } => {
                body.framer = Some(Framer::new(tick, 200, "text/event-stream", &[]));
                body.events = Some(Events::new(&shape));
                body.rx = Some(rx);
            }
        }
        body
    }

    /// The body holds `place`, a place among the open streams, to its end.
    #[must_use]
    pub fn holding(self, place: Option<tokio::sync::OwnedSemaphorePermit>) -> Self {
        Self {
            _place: place,
            ..self
        }
    }

    /// Ends the stream with `event`: the text not queued is dropped, the
    /// generation cancelled (its channel closed). The queue stays: it holds
    /// whole events, some maybe half sent.
    fn fail(&mut self, event: Vec<u8>) {
        if self.failure.is_none() {
            self.failure = Some(event);
        }
        self.pending.clear();
        self.finale = None;
        self.rx = None;
    }

    fn drain(&mut self, now: Instant) {
        loop {
            let Some(rx) = self.rx.as_mut() else {
                return;
            };
            match rx.try_recv() {
                Ok(event) => {
                    self.heard = Some(now);
                    match event {
                        Event::Started => {
                            self.started = true;
                            if let (Some(f), Some(e)) = (self.framer.as_mut(), &self.events) {
                                f.push(e.role().as_bytes());
                            }
                        }
                        Event::Delta(text) => self.pending.push_str(&text),
                        Event::Final(finale) => self.finale = Some(finale),
                        Event::Failed(status, message) => {
                            self.fail(error_event(&message, Some(status.as_u16())));
                        }
                        Event::Error(message) => self.fail(error_event(message, None)),
                    }
                }
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    if self.finale.is_none() {
                        self.fail(error_event(INTERNAL, None));
                    }
                    self.rx = None;
                    return;
                }
            }
        }
    }

    /// What the last chunk would carry now: the error event, a refusal's
    /// nothing, or the receipt once the text is all queued.
    fn tail(&self) -> Option<Vec<u8>> {
        if let Some(event) = &self.failure {
            return Some(event.clone());
        }
        let (finale, events) = (self.finale.as_ref()?, self.events.as_ref()?);
        self.pending
            .is_empty()
            .then(|| events.last(finale).into_bytes())
    }

    /// The next chunk's plaintext, and whether it is the last.
    fn prepare(&mut self, now: Instant) -> Result<(Vec<u8>, bool), Broken> {
        self.drain(now);
        self.built = self.built.saturating_add(1);
        if self.failure.is_none()
            && self.finale.is_none()
            && self.started
            && self
                .heard
                .is_some_and(|h| now.saturating_duration_since(h) >= STALL)
        {
            self.fail(error_event(STALLED, None));
        }
        // One event of text per tick, and only while the queue is short:
        // the rest waits here, where the carrier does not see it.
        let framer = self.framer.as_ref().ok_or(Broken)?;
        if self.events.is_some()
            && self.failure.is_none()
            && !self.pending.is_empty()
            && framer.queued() < framer.room()
        {
            let room = self.tick.chunk_bytes as usize;
            let text = crate::stream::take_fitting(&mut self.pending, room);
            match (&self.events, self.framer.as_mut()) {
                (Some(events), Some(framer)) if !text.is_empty() => {
                    framer.push(events.content(&text).as_bytes());
                }
                // A character wider than a chunk's room: never from a model.
                _ => self.fail(error_event(crate::stream::FRAME_SIZE, None)),
            }
        }
        let may_end = sealed_stream::may_end_at(self.built, self.cap);
        let framer = self.framer.as_ref().ok_or(Broken)?;
        let ready = self.tail().filter(|t| framer.fits_last(t.len()));
        let tail = match ready {
            Some(tail) if may_end => Some(tail),
            _ if self.built >= self.cap => {
                self.fail(error_event(TOO_LONG, None));
                self.tail()
            }
            _ => None,
        };
        let framer = self.framer.take().ok_or(Broken)?;
        match tail {
            Some(tail) => Ok((framer.last(&tail).map_err(|_| Broken)?, true)),
            None => {
                let mut framer = framer;
                let pt = framer.chunk().map_err(|_| Broken)?;
                self.framer = Some(framer);
                Ok((pt, false))
            }
        }
    }

    /// The chunk built and sealed: its bytes on the wire, and whether it is
    /// the last.
    fn sealed(&mut self, now: Instant) -> Result<(Bytes, bool), Broken> {
        let (pt, last) = self.prepare(now)?;
        let bytes = if last {
            self.rx = None;
            self.writer.take().ok_or(Broken)?.last(&pt)
        } else {
            self.writer.as_mut().ok_or(Broken)?.chunk(&pt)
        };
        Ok((Bytes::from(bytes.map_err(|_| Broken)?), last))
    }
}

impl<C: Cadence> Body for SealedStream<C> {
    type Data = Bytes;
    type Error = Broken;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Broken>>> {
        let this = self.get_mut();
        if this.ended {
            return Poll::Ready(None);
        }
        if this.prepared.is_none() {
            let now = std::task::ready!(this.cadence.poll_prepare(cx));
            match this.sealed(now) {
                Ok(chunk) => this.prepared = Some(chunk),
                Err(broken) => {
                    this.ended = true;
                    return Poll::Ready(Some(Err(broken)));
                }
            }
        }
        std::task::ready!(this.cadence.poll_release(cx));
        let Some((chunk, last)) = this.prepared.take() else {
            return Poll::Ready(Some(Err(Broken)));
        };
        this.ended = last;
        Poll::Ready(Some(Ok(Frame::data(chunk))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::{Clock, PREPARE_AHEAD};
    use axum::http::StatusCode;
    use std::time::Duration;
    use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
    use x0ne_sdk::bhttp::{Part, ResponseReader};
    use x0ne_sdk::e2e::{self, Gateway};
    use x0ne_sdk::ohttp::{self, KeyConfig};

    const T: Duration = Duration::from_millis(100);
    const NOW: u64 = 1_800_000_000;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// The TD's writer and the caller's reader of one chunked exchange.
    fn exchange() -> (ChunkedResponseWriter, ohttp::ChunkedClientResponse) {
        let g = Gateway::from_seed(&[7; 32]);
        let config = KeyConfig::parse(&g.key_config()).unwrap();
        let inner = e2e::chat_request(br#"{"stream":true,"messages":[]}"#, NOW).unwrap();
        let (req, client) =
            ohttp::encapsulate_chunked_request(&config, &[&inner[..]], &[1; 32]).unwrap();
        let (_, responder) = g.open_stream(&req).unwrap();
        (responder.writer(&[3; 16]).unwrap(), client)
    }

    fn shape() -> Shape {
        Shape {
            id: "chatcmpl-0123456789abcdef01234567".into(),
            created: NOW,
            model: "qwen3:0.6b".into(),
            include_usage: true,
        }
    }

    /// A finale whose x0ne block is the size of a real H100 one (7322 bytes,
    /// 2026-10-17).
    fn finale(tokens: u32) -> Event {
        Event::Final(Box::new(Finale {
            finish_reason: "stop",
            prompt_tokens: 12,
            completion_tokens: tokens,
            x0ne_json: format!("{{\"receipt\":\"{}\"}}", "r".repeat(7322 - 14)),
        }))
    }

    /// Each chunk as the carrier records it: when it left after the
    /// request, and its size.
    type Trace = Vec<(Duration, usize)>;

    struct Ran {
        trace: Trace,
        status: u16,
        content: String,
        /// The generation's channel closed before its script ended.
        cancelled: bool,
    }

    /// `script` played at its times after the request, `start` given its
    /// channel, the body read to its end as hyper reads it.
    async fn run(
        start: impl FnOnce(UnboundedReceiver<Event>) -> Start,
        script: Vec<(Duration, Event)>,
        hold: Duration,
        cap: u64,
    ) -> Ran {
        run_at(E2eStream::DEFAULT, start, script, hold, cap).await
    }

    /// [`run`] at the cadence `c`.
    async fn run_at(
        c: E2eStream,
        start: impl FnOnce(UnboundedReceiver<Event>) -> Start,
        script: Vec<(Duration, Event)>,
        hold: Duration,
        cap: u64,
    ) -> Ran {
        let asked = Instant::now();
        let (tx, rx) = unbounded_channel();
        let (writer, client) = exchange();
        let body = SealedStream::new(start(rx), c, cap, writer, Clock::every(asked + T, T));
        let player = tokio::spawn(play(tx, script, asked, hold));
        let mut body = Box::pin(body);
        let (mut trace, mut wire) = (Trace::new(), Vec::new());
        while let Some(frame) = std::future::poll_fn(|cx| body.as_mut().poll_frame(cx)).await {
            let data = frame.unwrap().into_data().unwrap();
            trace.push((asked.elapsed(), data.len()));
            wire.push(data);
        }
        drop(body);
        let cancelled = player.await.unwrap();
        let (status, content) = open(client, &wire);
        Ran {
            trace,
            status,
            content,
            cancelled,
        }
    }

    /// Sends `script`, then holds the channel open until `hold`; true if
    /// the body closed it first.
    async fn play(
        tx: UnboundedSender<Event>,
        script: Vec<(Duration, Event)>,
        asked: Instant,
        hold: Duration,
    ) -> bool {
        for (at, event) in script {
            tokio::time::sleep_until(asked + at).await;
            if tx.send(event).is_err() {
                return true;
            }
        }
        tokio::time::sleep_until(asked + hold).await;
        tx.is_closed()
    }

    /// The caller's side: chunks opened, Binary HTTP read, content as text.
    fn open(client: ohttp::ChunkedClientResponse, wire: &[Bytes]) -> (u16, String) {
        let mut reader = client.reader();
        let mut pts = Vec::new();
        for piece in wire {
            pts.extend(reader.push(piece).unwrap().into_iter().map(|p| p.to_vec()));
        }
        pts.push(reader.finish().unwrap().to_vec());
        let mut r = ResponseReader::new(1 << 24);
        let (mut status, mut content) = (0, Vec::new());
        for pt in &pts {
            for part in r.push(pt).unwrap() {
                match part {
                    Part::Head { status: s, .. } => status = s,
                    Part::Content(c) => content.extend(c),
                }
            }
        }
        r.finish().unwrap();
        (status, String::from_utf8(content).unwrap())
    }

    fn running(rx: UnboundedReceiver<Event>) -> Start {
        Start::Running { rx, shape: shape() }
    }

    /// Tokens of `text`, `every` apart from `from`, then the finale.
    fn tokens(text: &str, from: Duration, every: Duration) -> Vec<(Duration, Event)> {
        let mut script = vec![(from, Event::Started)];
        let mut at = from;
        let words: Vec<String> = text.split_inclusive(' ').map(str::to_string).collect();
        let n = words.len() as u32;
        for w in words {
            at += every;
            script.push((at, Event::Delta(w)));
        }
        script.push((at + every, finale(n)));
        script
    }

    /// The trace of every stream of `chunks` chunks: chunk k leaves (k+1)
    /// ticks after the request, written PREPARE_AHEAD after it is built.
    fn expected(chunks: u64) -> Trace {
        let l = E2eStream::DEFAULT.chunk_bytes as usize;
        (0..chunks)
            .map(|k| {
                let at = T * (k as u32 + 1) + PREPARE_AHEAD;
                let size = match k {
                    0 => 16 + 2 + l + 16,
                    k if k + 1 == chunks => 1 + sealed_stream::FINAL_CHUNK_BYTES + 16,
                    _ => 2 + l + 16,
                };
                (at, size)
            })
            .collect()
    }

    /// The text of the content's deltas, in order.
    fn deltas(content: &str) -> String {
        content
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter_map(|d| serde_json::from_str::<serde_json::Value>(d).ok())
            .filter_map(|v| {
                v.pointer("/choices/0/delta/content")
                    .and_then(|c| c.as_str())
                    .map(str::to_string)
            })
            .collect()
    }

    /// SECURITY: two answers of different lengths, subjects and speeds
    /// leave in chunks of the same sizes at the same instants: what the
    /// carrier records does not follow the text. A failure is a size or a
    /// time that depends on the answer.
    #[tokio::test(start_paused = true)]
    async fn two_answers_leave_at_the_same_times_in_the_same_sizes() {
        let short = run(running, tokens("4", ms(150), ms(50)), ms(0), 8192).await;
        let long_text = "The Treaty of Westphalia ended the Thirty Years War in 1648 and \
                         set out the rule of sovereign states over their own lands. "
            .repeat(2);
        let long = run(running, tokens(&long_text, ms(120), ms(20)), ms(0), 8192).await;
        assert_eq!(short.trace, expected(MIN_CHUNKS));
        assert_eq!(long.trace, short.trace);
        assert_eq!((short.status, long.status), (200, 200));
        assert_eq!(deltas(&short.content), "4");
        assert_eq!(deltas(&long.content), long_text);
        assert!(long.content.trim_end().ends_with("data: [DONE]"));
    }

    /// An answer longer than 16 chunks can carry ends at the next power of
    /// two, the same sizes throughout.
    #[tokio::test(start_paused = true)]
    async fn a_longer_answer_ends_at_the_next_power_of_two() {
        let text = "word ".repeat(1200);
        let ran = run(running, tokens(&text, ms(100), ms(5)), ms(0), 8192).await;
        let n = ran.trace.len() as u64;
        assert!(n > MIN_CHUNKS && n.is_power_of_two(), "{n} chunks");
        assert_eq!(ran.trace, expected(n));
        assert_eq!(deltas(&ran.content), text);
    }

    /// SECURITY: a refusal before the model, a failure of the model once
    /// admitted, an error after the first tokens, a generation that dies
    /// without its end: each leaves at the rhythm of a short success, 16
    /// chunks of the same sizes at the same times. A failure is an outcome
    /// the carrier tells apart.
    #[tokio::test(start_paused = true)]
    async fn refusals_errors_and_panics_keep_the_rhythm_of_a_success() {
        let want = expected(MIN_CHUNKS);
        let refused = run(
            |_| Start::Refused(ApiError::new(StatusCode::BAD_REQUEST, "messages is empty")),
            vec![],
            ms(0),
            8192,
        )
        .await;
        assert_eq!(refused.trace, want);
        assert_eq!(refused.status, 400);
        let body: serde_json::Value = serde_json::from_str(refused.content.trim()).unwrap();
        assert_eq!(body["error"]["message"], "messages is empty");

        let busy = run(
            |_| {
                Start::Refused(
                    ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "busy").retry_after(5),
                )
            },
            vec![],
            ms(0),
            8192,
        )
        .await;
        assert_eq!((busy.trace.clone(), busy.status), (want.clone(), 503));

        let too_long = run(
            running,
            vec![(
                ms(300),
                Event::Failed(StatusCode::PAYLOAD_TOO_LARGE, "prompt too long".into()),
            )],
            ms(0),
            8192,
        )
        .await;
        assert_eq!(
            (too_long.trace.clone(), too_long.status),
            (want.clone(), 200)
        );
        assert!(
            too_long.content.contains(r#""status":413"#),
            "{}",
            too_long.content
        );

        let failed = run(
            running,
            vec![
                (ms(100), Event::Started),
                (ms(200), Event::Delta("par".into())),
                (ms(400), Event::Error(crate::stream::GENERATION_FAILED)),
            ],
            ms(0),
            8192,
        )
        .await;
        assert_eq!(failed.trace, want);
        assert_eq!(deltas(&failed.content), "par");
        assert!(failed.content.contains("generation failed"));
        assert!(!failed.content.contains("[DONE]"));

        // A panic drops the sender without a final event.
        let panicked = run(
            running,
            vec![
                (ms(100), Event::Started),
                (ms(150), Event::Delta("a".into())),
            ],
            ms(0),
            8192,
        )
        .await;
        assert_eq!(panicked.trace, want);
        assert!(panicked.content.contains(INTERNAL));
    }

    /// A generation silent for 90 s after it started ends in error, at the
    /// size after; its channel is closed, which cancels it.
    #[tokio::test(start_paused = true)]
    async fn a_stalled_generation_ends_in_error_at_a_size() {
        let ran = run(
            running,
            vec![(ms(100), Event::Started)],
            Duration::from_secs(200),
            8192,
        )
        .await;
        assert_eq!(ran.trace, expected(1024), "90.1 s: 902 ticks, then 1024");
        assert!(ran.content.contains(STALLED));
        assert!(ran.cancelled);
    }

    /// A stream that would run past its cap ends there with an error event,
    /// and its generation is cancelled: nothing is signed for it later.
    #[tokio::test(start_paused = true)]
    async fn a_stream_ends_at_its_cap_and_cancels_its_generation() {
        let mut script = vec![(ms(100), Event::Started)];
        script.extend((1..200).map(|i| (ms(100 + 50 * i), Event::Delta("go ".into()))));
        let ran = run(running, script, ms(20_000), 64).await;
        assert_eq!(ran.trace, expected(64));
        assert!(ran.content.contains(TOO_LONG));
        assert!(ran.cancelled, "the generation was not cancelled");
    }

    /// A request waiting for its turn still gets its first chunk one tick
    /// after it came: the queue in front of the model is not on the wire.
    #[tokio::test(start_paused = true)]
    async fn the_first_chunk_leaves_one_tick_after_the_request_whatever_the_queue() {
        let ran = run(running, tokens("late", ms(5_000), ms(50)), ms(0), 8192).await;
        assert_eq!(ran.trace, expected(64));
        assert_eq!(deltas(&ran.content), "late");
    }

    /// The final chunk at the largest cadence (L = 4096) behind the fullest
    /// queue the TD builds (a chunk's room less one byte, then one event of
    /// L bytes of text), the widest stream (`model` of 256 bytes, counters
    /// at their widest), and the largest real `x0ne` block
    /// (`server::tests::largest_x0ne_block_bytes`: GPU, a strict TCB view,
    /// the decoding's claims): they do not fit. What the stream does then is
    /// the next test's. A failure is a margin that changed sign: the docs
    /// say it does not fit (docs/sdk-e2e.md). No print: no log in this file.
    #[test]
    fn at_l_4096_the_largest_real_receipt_does_not_fit_behind_a_full_queue() {
        let c = E2eStream {
            chunk_bytes: 4096,
            tick_ms: 100,
        };
        let block = crate::server::tests::largest_x0ne_block_bytes();
        let filler = block - r#"{"receipt":{"pad":""}}"#.len();
        let finale = Finale {
            finish_reason: "length",
            prompt_tokens: u32::MAX,
            completion_tokens: u32::MAX,
            x0ne_json: format!(r#"{{"receipt":{{"pad":"{}"}}}}"#, "r".repeat(filler)),
        };
        let margin = |model: &str| {
            let events = Events::new(&Shape {
                id: format!("chatcmpl-{}", "f".repeat(24)),
                created: u64::MAX,
                model: model.into(),
                include_usage: true,
            });
            let mut f = Framer::new(c, 200, "text/event-stream", &[]);
            f.chunk().unwrap();
            let room = f.room();
            let overhead = events.content("").len();
            f.push(events.content(&"a".repeat(room - 1 - overhead)).as_bytes());
            assert_eq!(f.queued(), room - 1);
            f.push(events.content(&"a".repeat(4096)).as_bytes());
            let tail = events.last(&finale).len();
            let used = chunk_len(f.queued() + tail) + END.len();
            let margin = FINAL as i64 - used as i64;
            assert_eq!(f.fits_last(tail), margin >= 0);
            margin
        };
        use x0ne_sdk::bhttp::indeterminate::{chunk_len, END};
        const FINAL: usize = sealed_stream::FINAL_CHUNK_BYTES;
        let (widest, real) = (margin(&"m".repeat(256)), margin("qwen3:0.6b"));
        // Measured 2026-10-30: 10025 B, margins -3236 and -2498 B; on
        // 2026-10-31, a conversation's context and handle counted: 10747 B.
        assert_eq!((block, widest, real), (10_747, -3958, -3220));
    }

    /// A finale whose `x0ne` block is `bytes` long.
    fn finale_of(bytes: usize) -> Event {
        let filler = "r".repeat(bytes - r#"{"receipt":{"pad":""}}"#.len());
        Event::Final(Box::new(Finale {
            finish_reason: "stop",
            prompt_tokens: 12,
            completion_tokens: 4096,
            x0ne_json: format!(r#"{{"receipt":{{"pad":"{filler}"}}}}"#),
        }))
    }

    /// At L = 4096, texts of 16, 31 and 62 events of L bytes, the model
    /// faster than a tick: the largest real receipt (10747 bytes) does not
    /// fit behind the queue left at 16, 32 and 64 chunks, and the stream
    /// ends a size later, its text, receipt and `[DONE]` whole; the H100's
    /// block (7322 bytes) and a small one fit. A failure is a receipt lost,
    /// a stream ended in error, or a size that changed.
    #[tokio::test(start_paused = true)]
    async fn at_l_4096_the_largest_real_receipt_behind_a_full_queue_costs_a_size() {
        let c = E2eStream {
            chunk_bytes: 4096,
            tick_ms: 100,
        };
        let largest = crate::server::tests::largest_x0ne_block_bytes();
        for (k, sizes) in [(16, [32, 16, 16]), (31, [64, 32, 32]), (62, [128, 64, 64])] {
            let text: String = (0..k).map(|_| "a".repeat(4095) + " ").collect();
            for (block, chunks) in [largest, 7322, 64].into_iter().zip(sizes) {
                let script = vec![
                    (ms(50), Event::Started),
                    (ms(60), Event::Delta(text.clone())),
                    (ms(70), finale_of(block)),
                ];
                let ran = run_at(c, running, script, ms(0), 8192).await;
                let at = format!("{k} events, a block of {block} bytes");
                assert_eq!((ran.trace.len(), ran.status), (chunks, 200), "{at}");
                assert_eq!(deltas(&ran.content), text, "{at}");
                assert!(ran.content.trim_end().ends_with("data: [DONE]"), "{at}");
                assert!(ran.content.contains("\"x0ne\":{\"receipt\""), "{at}");
                assert!(!ran.content.contains("\"error\""), "{at}");
            }
        }
    }

    /// A sealed stream's turn ends at the first power of two of ticks after
    /// its request that its generation fits, 16 at least, as its chunks; past
    /// its cap, when the generation ended. The cost, against the Padme turn
    /// of a whole answer begun at the request, is the table (docs/sdk-e2e.md).
    #[test]
    fn a_sealed_streams_turn_ends_at_a_power_of_two_of_ticks_after_its_request() {
        let asked = Instant::now();
        let turn = |cap| StreamTurn {
            asked,
            tick: T,
            cap,
        };
        let end = |done: u64, cap| turn(cap).end(asked + ms(done)).duration_since(asked);
        assert_eq!(end(0, 8192), ms(1600));
        assert_eq!(end(50, 8192), ms(1600));
        assert_eq!(end(1600, 8192), ms(1600));
        assert_eq!(end(1601, 8192), ms(3200));
        assert_eq!(end(6500, 8192), ms(12_800));
        assert_eq!(end(6500, 64), ms(6500), "past its cap: when it ended");
        let table = [
            (50, 1_600, 1_000),
            (1_000, 1_600, 1_000),
            (1_700, 3_200, 1_800),
            (3_300, 6_400, 3_600),
            (6_500, 12_800, 7_200),
            (13_000, 25_600, 14_400),
            (100_000, 102_400, 102_400),
        ];
        for (done, turn, padme) in table {
            let padded = crate::padding::release_after(ms(done), T);
            assert_eq!(
                (end(done, 8192), padded),
                (ms(turn), ms(padme)),
                "{done} ms"
            );
        }
    }

    /// The cap: the largest power of two whose last chunk leaves before the
    /// session's end less 60 s, and within the relay's 15 minutes.
    #[test]
    fn the_cap_follows_the_session_end_and_the_relays_wait() {
        let c = E2eStream::DEFAULT;
        assert_eq!(cap(&c, NOW, NOW + 3600), Some(8192));
        assert_eq!(cap(&c, NOW, NOW + 60 + 100), Some(512), "999 ticks fit");
        assert_eq!(cap(&c, NOW, NOW + 60 + 2), Some(16), "19 ticks fit");
        assert_eq!(cap(&c, NOW, NOW + 60 + 1), None, "9 ticks fit");
        assert_eq!(cap(&c, NOW, NOW), None);
        let slow = E2eStream {
            chunk_bytes: 256,
            tick_ms: 1000,
        };
        assert_eq!(cap(&slow, NOW, NOW + 3600), Some(512));
    }
}
