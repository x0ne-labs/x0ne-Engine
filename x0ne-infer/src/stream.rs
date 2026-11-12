//! The body of a streamed chat answer: OpenAI's SSE chunks, written as one
//! frame of a fixed size every tick, whatever the text. The host of the VM
//! sees frames of one size at one cadence, then a last frame of another
//! fixed size: neither token sizes nor token timing.
//!
//! The number of frames, the last included, is a size of `crate::padding`,
//! [`crate::padding::MIN_TICKS`] at least: the last frame waits for one, on
//! success and on error. The host reads the stream's length to within a size,
//! not to the frame. A stream is [`crate::padding::MAX_TICKS`] frames at most,
//! within the relay's 15 minutes: one whose text is not out by then ends there
//! with an error event, [`crate::stream::TOO_LONG`], and its generation is
//! cancelled.
//!
//! A frame with text pads it in `obfuscation`, a field OpenAI's SDKs ignore;
//! a tick without text is an SSE comment. The last frame carries the final
//! chunk (the receipt), the usage chunk when asked, and `[DONE]`; an error
//! after the 200 is one error event of that same last size, then the body's
//! end, as after a success: the event, never the end, says the stream failed.
//!
//! No log macro in this file: it handles the text of the answer.

use std::future::Future;
use std::ops::ControlFlow;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::http::StatusCode;
use hyper::body::{Body, Bytes, Frame};
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::time::Instant;

use crate::backend::{Cancelled, Completion, TokenSink};

/// One frame per tick.
pub const TICK: Duration = Duration::from_millis(100);
/// A frame is built this long before its tick, and written at the tick.
pub const PREPARE_AHEAD: Duration = Duration::from_millis(10);
/// The last frame of every stream, success or error. The x0ne block of a real
/// H100 answer was 7322 bytes (2026-10-17): with its chunk, over 8 KiB.
pub const FINAL_FRAME_BYTES: usize = 16 * 1024;
/// No event from the generation for this long ends the stream in error.
pub const STALL: Duration = Duration::from_secs(90);
const MIN_FRAME_BYTES: usize = 512;
/// Room for text in a frame, at least: the frame is the empty chunk plus this.
const TEXT_ROOM: usize = 256;

/// Fixed messages of an error event: nothing of the request, nothing generated.
pub const GENERATION_FAILED: &str = "generation failed";
pub const NOT_PREFIX_STABLE: &str = "decoder is not prefix-stable";
pub const STALLED: &str = "generation stalled";
pub const INTERNAL: &str = "internal error";
pub const FRAME_SIZE: &str = "a frame did not fit its size";
pub const RECEIPT_TOO_LARGE: &str = "the receipt did not fit the last frame";
pub const TOO_LONG: &str = "the answer would end past the 15 minutes the relay waits";

/// What the generation side tells the body, in order.
pub enum Event {
    /// Every refusal is past: the 200 may go.
    Started,
    /// Final text, never empty.
    Delta(String),
    /// The signed end.
    Final(Box<Finale>),
    /// A refusal before `Started`: an HTTP status, never a stream.
    Failed(StatusCode, String),
    /// A failure after `Started`, by its fixed message.
    Error(&'static str),
}

/// The end of a stream that went well.
pub struct Finale {
    pub finish_reason: &'static str,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// The `x0ne` block, serialized once: receipt, opening, signed payload.
    pub x0ne_json: String,
}

/// What every chunk of one stream repeats.
pub struct Shape {
    /// `chatcmpl-` and 24 hex.
    pub id: String,
    pub created: u64,
    pub model: String,
    pub include_usage: bool,
}

/// The message of an error event for a failure after `Started`. Fixed: an
/// error's own text may quote decoded output.
pub fn failure_after_start(e: &anyhow::Error) -> &'static str {
    if e.downcast_ref::<crate::backend::NotPrefixStable>()
        .is_some()
    {
        NOT_PREFIX_STABLE
    } else {
        GENERATION_FAILED
    }
}

/// Hands a generation to the attested server's stream body. The reader gone
/// is `alive() == false`.
pub struct ChannelSink {
    tx: UnboundedSender<Event>,
    /// Every delta sent, for the check against the completion ([`outcome`]).
    pub text: String,
    pub started: bool,
}

impl ChannelSink {
    pub fn new(tx: UnboundedSender<Event>) -> Self {
        Self {
            tx,
            text: String::new(),
            started: false,
        }
    }

    fn send(&self, event: Event) -> ControlFlow<()> {
        match self.tx.send(event) {
            Ok(()) => ControlFlow::Continue(()),
            Err(_) => ControlFlow::Break(()),
        }
    }
}

impl TokenSink for ChannelSink {
    fn started(&mut self) -> ControlFlow<()> {
        self.started = true;
        self.send(Event::Started)
    }
    fn delta(&mut self, text: &str) -> ControlFlow<()> {
        self.text.push_str(text);
        self.send(Event::Delta(text.to_string()))
    }
    fn alive(&self) -> bool {
        !self.tx.is_closed()
    }
}

/// How a streamed generation ended: the completion to end the stream with,
/// or the event that ends it instead; `None` after a cancellation, when
/// nobody reads. A completion is kept only past `Started` and equal to the
/// text `streamed`. A failure before `Started` is an HTTP status
/// ([`crate::server::generate_error`]); after it, a fixed message.
pub fn outcome(
    ran: anyhow::Result<Completion>,
    started: bool,
    streamed: &str,
) -> Option<Result<Completion, Event>> {
    match ran {
        Err(e) if e.downcast_ref::<Cancelled>().is_some() => None,
        Err(e) if started => Some(Err(Event::Error(failure_after_start(&e)))),
        Err(e) => {
            let (status, message) = crate::server::generate_error(e);
            Some(Err(Event::Failed(status, message)))
        }
        Ok(completion) if started && completion.text == streamed => Some(Ok(completion)),
        Ok(_) => Some(Err(Event::Error(GENERATION_FAILED))),
    }
}

/// When frames are built and when they are written. Injected: tests tick by hand.
pub trait Cadence: Send + Unpin + 'static {
    /// Ready when the next frame is to be built; gives the time it is built at.
    fn poll_prepare(&mut self, cx: &mut Context<'_>) -> Poll<Instant>;
    /// Ready when the frame built last may be written.
    fn poll_release(&mut self, cx: &mut Context<'_>) -> Poll<()>;
}

/// The production cadence: built every [`TICK`], written [`PREPARE_AHEAD`] later.
pub struct Clock {
    ticks: tokio::time::Interval,
    release: Pin<Box<tokio::time::Sleep>>,
}

impl Clock {
    /// Within a tokio runtime. The first frame is built at once.
    pub fn start() -> Self {
        Self::ticking(crate::padding::Tick::PRODUCTION)
    }

    /// Within a tokio runtime. The first frame is built at once, then one
    /// every `tick`.
    pub fn ticking(tick: crate::padding::Tick) -> Self {
        Self::every(Instant::now(), tick.period())
    }

    /// Frames built at `first`, then every `period`, each written
    /// [`PREPARE_AHEAD`] after it is built. Within a tokio runtime.
    pub fn every(first: Instant, period: Duration) -> Self {
        let mut ticks = tokio::time::interval_at(first, period);
        // Burst, the default, would write late frames back to back.
        ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        Self {
            ticks,
            release: Box::pin(tokio::time::sleep(Duration::ZERO)),
        }
    }
}

impl Cadence for Clock {
    fn poll_prepare(&mut self, cx: &mut Context<'_>) -> Poll<Instant> {
        let tick = std::task::ready!(self.ticks.poll_tick(cx));
        self.release.as_mut().reset(tick + PREPARE_AHEAD);
        Poll::Ready(Instant::now())
    }
    fn poll_release(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        self.release.as_mut().poll(cx)
    }
}

/// A frame missing where one was prepared: the body ends without its end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the stream ended in error")]
pub struct Broken;

/// The SSE body, read by hyper one frame at a time.
pub struct SseBody<C: Cadence> {
    rx: UnboundedReceiver<Event>,
    cadence: C,
    frames: Frames,
    pending: String,
    role_sent: bool,
    finale: Option<Box<Finale>>,
    failure: Option<&'static str>,
    heard: Instant,
    /// The next frame, and whether it is the last.
    prepared: Option<(Bytes, bool)>,
    /// Frames built so far, the one being built included.
    built: u64,
    /// The final frame, built, waiting for a frame number that is a size.
    last: Option<Bytes>,
    ended: bool,
    /// A place among the open streams, given back when hyper drops the body.
    _place: Option<tokio::sync::OwnedSemaphorePermit>,
}

impl<C: Cadence> SseBody<C> {
    /// `rx` after its `Started`; `now` starts the [`STALL`] clock.
    pub fn new(rx: UnboundedReceiver<Event>, shape: Shape, cadence: C, now: Instant) -> Self {
        Self {
            rx,
            cadence,
            frames: Frames::new(&shape),
            pending: String::new(),
            role_sent: false,
            finale: None,
            failure: None,
            heard: now,
            prepared: None,
            built: 0,
            last: None,
            ended: false,
            _place: None,
        }
    }

    /// Holds `place` until the body is dropped: the stream counts until then.
    #[must_use]
    // `_place` is held until the drop, never read.
    #[allow(clippy::used_underscore_binding)]
    pub fn holding(mut self, place: Option<tokio::sync::OwnedSemaphorePermit>) -> Self {
        self._place = place;
        self
    }

    /// Bytes of every frame but the last.
    pub fn frame_size(&self) -> usize {
        self.frames.size
    }

    fn drain(&mut self, now: Instant) {
        loop {
            match self.rx.try_recv() {
                Ok(event) => {
                    self.heard = now;
                    match event {
                        Event::Delta(text) => self.pending.push_str(&text),
                        Event::Final(finale) => self.finale = Some(finale),
                        Event::Error(message) => self.failure = Some(message),
                        Event::Started => {}
                        Event::Failed(..) => self.failure = Some(INTERNAL),
                    }
                }
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    if self.finale.is_none() && self.last.is_none() && self.failure.is_none() {
                        self.failure = Some(INTERNAL);
                    }
                    return;
                }
            }
        }
    }

    /// The next frame, and whether it is the last. A failure drops the text
    /// still pending: nothing follows an error event but the body's end. The
    /// last frame, final or error, goes out only as a frame whose number is a
    /// size; padding comments fill the frames before it.
    fn prepare(&mut self, now: Instant) -> (Bytes, bool) {
        self.drain(now);
        self.built = self.built.saturating_add(1);
        let may_end = crate::padding::is_size(self.built, crate::padding::MIN_TICKS);
        if self.failure.is_none()
            && self.finale.is_none()
            && self.last.is_none()
            && now.saturating_duration_since(self.heard) >= STALL
        {
            self.failure = Some(STALLED);
        }
        // The last frame is built once, when the text is out.
        if self.failure.is_none() && self.role_sent && self.pending.is_empty() {
            if let Some(finale) = self.finale.take() {
                match self.frames.last(&finale) {
                    Some(frame) => self.last = Some(frame),
                    None => self.failure = Some(RECEIPT_TOO_LARGE),
                }
            }
        }
        // The last size the relay waits for: no final by then, an error.
        if self.failure.is_none() && self.last.is_none() && self.built >= crate::padding::MAX_TICKS
        {
            self.failure = Some(TOO_LONG);
        }
        if self.failure.is_none() {
            if may_end {
                if let Some(frame) = self.last.take() {
                    return (frame, true);
                }
            }
            let built = if !self.role_sent {
                self.role_sent = true;
                self.frames.role()
            } else if !self.pending.is_empty() {
                let text = take_fitting(&mut self.pending, self.frames.text_room);
                self.frames.content(&text)
            } else {
                Some(self.frames.comment.clone())
            };
            match built {
                Some(frame) => return (frame, false),
                None => self.failure = Some(FRAME_SIZE),
            }
        }
        self.pending.clear();
        self.finale = None;
        self.last = None;
        match self.failure {
            Some(message) if may_end => (self.frames.error(message), true),
            _ => (self.frames.comment.clone(), false),
        }
    }
}

impl<C: Cadence> Body for SseBody<C> {
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
            this.prepared = Some(this.prepare(now));
        }
        std::task::ready!(this.cadence.poll_release(cx));
        let Some((frame, last)) = this.prepared.take() else {
            return Poll::Ready(Some(Err(Broken)));
        };
        this.ended = last;
        Poll::Ready(Some(Ok(Frame::data(frame))))
    }
}

/// Bytes `c` takes inside a JSON string as serde_json writes it.
fn escaped_len(c: char) -> usize {
    match c {
        '"' | '\\' | '\u{8}' | '\t' | '\n' | '\u{c}' | '\r' => 2,
        '\0'..='\u{1f}' => 6,
        _ => c.len_utf8(),
    }
}

/// The longest run of whole characters of `pending` whose escaped form fits
/// `room` bytes, taken off its front.
pub(crate) fn take_fitting(pending: &mut String, room: usize) -> String {
    let mut used = 0;
    let mut cut = pending.len();
    for (at, c) in pending.char_indices() {
        let n = escaped_len(c);
        if used + n > room {
            cut = at;
            break;
        }
        used += n;
    }
    let rest = pending.split_off(cut);
    std::mem::replace(pending, rest)
}

fn json(s: &str) -> Option<String> {
    serde_json::to_string(s).ok()
}

/// `before`, padding, `after`: exactly `size` bytes, or `None` if they exceed it.
fn padded(before: &str, after: &str, size: usize) -> Option<Bytes> {
    let pad = size.checked_sub(before.len() + after.len())?;
    let mut frame = String::with_capacity(size);
    frame.push_str(before);
    frame.extend(std::iter::repeat_n('_', pad));
    frame.push_str(after);
    (frame.len() == size).then(|| Bytes::from(frame))
}

const CONTENT_OPEN: &str = r#"{"content":"#;
const ROLE_DELTA: &str = r#"{"role":"assistant","content":""}"#;
const OPEN_PAD: &str = r#","finish_reason":null}],"obfuscation":""#;
const CLOSE_PAD: &str = "\"}\n\n";

/// The frames of one stream, built around its [`Shape`].
struct Frames {
    /// `data: {"id":…,"model":…,"choices":[{"index":0,"delta":`
    head: String,
    /// `data: {"id":…,"model":…,"choices":[],"usage":` when asked, else empty.
    usage_head: Option<String>,
    size: usize,
    text_room: usize,
    comment: Bytes,
}

impl Frames {
    fn new(shape: &Shape) -> Self {
        let id = json(&shape.id).unwrap_or_default();
        let model = json(&shape.model).unwrap_or_default();
        let common = format!(
            "data: {{\"id\":{id},\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":{model},\"choices\":",
            shape.created
        );
        let head = format!("{common}[{{\"index\":0,\"delta\":");
        let usage_head = shape
            .include_usage
            .then(|| format!("{common}[],\"usage\":"));
        let skeleton = head.len() + CONTENT_OPEN.len() + 2 + 1 + OPEN_PAD.len() + CLOSE_PAD.len();
        let size = MIN_FRAME_BYTES.max((skeleton + TEXT_ROOM).div_ceil(256) * 256);
        let mut comment = String::with_capacity(size);
        comment.push(':');
        comment.extend(std::iter::repeat_n(' ', size - 3));
        comment.push_str("\n\n");
        Self {
            head,
            usage_head,
            size,
            text_room: size - skeleton,
            comment: Bytes::from(comment),
        }
    }

    fn role(&self) -> Option<Bytes> {
        let before = format!("{}{ROLE_DELTA}{OPEN_PAD}", self.head);
        padded(&before, CLOSE_PAD, self.size)
    }

    fn content(&self, text: &str) -> Option<Bytes> {
        let before = format!("{}{CONTENT_OPEN}{}}}{OPEN_PAD}", self.head, json(text)?);
        padded(&before, CLOSE_PAD, self.size)
    }

    /// Final chunk, usage chunk if asked, `[DONE]`: [`FINAL_FRAME_BYTES`] in all.
    fn last(&self, finale: &Finale) -> Option<Bytes> {
        let before = format!(
            "{}{{}},\"finish_reason\":{}}}],\"x0ne\":{},\"obfuscation\":\"",
            self.head,
            json(finale.finish_reason)?,
            finale.x0ne_json
        );
        let mut after = String::from(CLOSE_PAD);
        if let Some(usage_head) = &self.usage_head {
            let (p, c) = (finale.prompt_tokens, finale.completion_tokens);
            after.push_str(&format!(
                "{usage_head}{{\"prompt_tokens\":{p},\"completion_tokens\":{c},\"total_tokens\":{}}}}}\n\n",
                u64::from(p) + u64::from(c)
            ));
        }
        after.push_str("data: [DONE]\n\n");
        padded(&before, &after, FINAL_FRAME_BYTES)
    }

    /// An SSE comment, then the error event: [`FINAL_FRAME_BYTES`] in all.
    fn error(&self, message: &'static str) -> Bytes {
        let event = format!(
            "data: {{\"error\":{{\"message\":{},\"type\":\"x0ne_engine\",\"code\":\"generation_failed\"}}}}\n\n",
            json(message).unwrap_or_else(|| "\"\"".into())
        );
        let mut frame = String::with_capacity(FINAL_FRAME_BYTES);
        frame.push(':');
        let room = FINAL_FRAME_BYTES.saturating_sub(event.len() + 3);
        frame.extend(std::iter::repeat_n(' ', room));
        frame.push_str("\n\n");
        frame.push_str(&event);
        Bytes::from(frame)
    }
}

/// The last frame a stream of `shape` would end with, for size checks.
pub fn last_frame(shape: &Shape, finale: &Finale) -> Option<Bytes> {
    Frames::new(shape).last(finale)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

    /// Builds when the test sends a time; writes at once.
    pub(crate) struct ByHand(pub(crate) UnboundedReceiver<Instant>);

    impl Cadence for ByHand {
        fn poll_prepare(&mut self, cx: &mut Context<'_>) -> Poll<Instant> {
            match self.0.poll_recv(cx) {
                Poll::Ready(Some(now)) => Poll::Ready(now),
                Poll::Ready(None) | Poll::Pending => Poll::Pending,
            }
        }
        fn poll_release(&mut self, _cx: &mut Context<'_>) -> Poll<()> {
            Poll::Ready(())
        }
    }

    fn completion(text: &str) -> Completion {
        Completion {
            text: text.into(),
            prompt_tokens: 1,
            completion_tokens: 1,
            stopped: true,
            length: false,
        }
    }

    /// A failure means a cancelled generation gets an end, an error's text
    /// reaches the stream, or a completion other than the text streamed ends it.
    #[test]
    fn outcome_ends_a_stream_by_what_the_generation_did() {
        assert!(outcome(Err(Cancelled.into()), true, "a").is_none());
        let failed = outcome(Err(anyhow::anyhow!("device lost: secret")), true, "a");
        assert!(matches!(failed, Some(Err(Event::Error(GENERATION_FAILED)))));
        let prefix = outcome(Err(crate::backend::NotPrefixStable.into()), true, "a");
        assert!(matches!(prefix, Some(Err(Event::Error(NOT_PREFIX_STABLE)))));
        let too_long = crate::backend::PromptTooLong {
            prompt_tokens: 9,
            context_tokens: 8,
        };
        match outcome(Err(too_long.into()), false, "") {
            Some(Err(Event::Failed(status, _))) => {
                assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE)
            }
            _ => panic!("a refusal before the start is a status"),
        }
        assert!(matches!(
            outcome(Ok(completion("ab")), true, "ab"),
            Some(Ok(_))
        ));
        let other = outcome(Ok(completion("ab")), true, "a");
        assert!(matches!(other, Some(Err(Event::Error(GENERATION_FAILED)))));
        let unstarted = outcome(Ok(completion("")), false, "");
        assert!(matches!(
            unstarted,
            Some(Err(Event::Error(GENERATION_FAILED)))
        ));
    }

    pub(crate) fn shape(model: &str, include_usage: bool) -> Shape {
        Shape {
            id: format!("chatcmpl-{}", "ab".repeat(12)),
            created: 1_760_000_000,
            model: model.to_string(),
            include_usage,
        }
    }

    type Polled = Poll<Option<Result<Bytes, Broken>>>;

    fn poll_once(body: &mut SseBody<ByHand>) -> Polled {
        let mut cx = Context::from_waker(std::task::Waker::noop());
        Pin::new(body)
            .poll_frame(&mut cx)
            .map(|f| f.map(|r| r.map(|frame| frame.into_data().unwrap_or_default())))
    }

    struct Harness {
        body: SseBody<ByHand>,
        events: Option<UnboundedSender<Event>>,
        ticks: UnboundedSender<Instant>,
    }

    impl Harness {
        fn new(model: &str, include_usage: bool, t0: Instant) -> Self {
            let (events, rx) = unbounded_channel();
            let (ticks, by_hand) = unbounded_channel();
            Self {
                body: SseBody::new(rx, shape(model, include_usage), ByHand(by_hand), t0),
                events: Some(events),
                ticks,
            }
        }
        fn send(&self, event: Event) {
            self.events.as_ref().unwrap().send(event).unwrap();
        }
        /// The generation side goes away.
        fn hang_up(&mut self) {
            self.events = None;
        }
        /// One tick at `now`, and the one frame it gives.
        fn tick(&mut self, now: Instant) -> Polled {
            self.ticks.send(now).unwrap();
            poll_once(&mut self.body)
        }
    }

    fn data_of(frame: &[u8]) -> Vec<serde_json::Value> {
        let text = std::str::from_utf8(frame).unwrap();
        text.split("\n\n")
            .filter_map(|event| event.strip_prefix("data: "))
            .filter(|d| *d != "[DONE]")
            .map(|d| serde_json::from_str(d).unwrap())
            .collect()
    }

    /// T10.
    /// The size of every frame is a function of the model name only, never
    /// of the text; one frame per tick; the last is [`FINAL_FRAME_BYTES`].
    /// A failure gives the host token sizes back, or splits a character.
    #[test]
    fn every_frame_is_one_serialized_size_whatever_the_text() {
        let nasty = "\"\\\n\u{1}😀é\t\u{1f}abc".repeat(40);
        let longest = "x".repeat(crate::server::MAX_STREAM_MODEL_BYTES);
        for (model, include_usage) in [("m", false), (longest.as_str(), true), ("\"é\\", true)] {
            let t0 = Instant::now();
            let mut h = Harness::new(model, include_usage, t0);
            let size = h.body.frame_size();
            assert!((512..=768).contains(&size), "{model}: {size}");
            assert!(poll_once(&mut h.body).is_pending(), "no tick, no frame");
            for piece in nasty.split_inclusive('é') {
                h.send(Event::Delta(piece.to_string()));
            }
            h.send(Event::Final(Box::new(Finale {
                finish_reason: "stop",
                prompt_tokens: 7,
                completion_tokens: 9,
                x0ne_json: r#"{"receipt":{"k":"v"}}"#.into(),
            })));
            h.hang_up();
            let mut frames = Vec::new();
            let mut now = t0;
            loop {
                now += TICK;
                match h.tick(now) {
                    Poll::Ready(Some(Ok(frame))) => frames.push(frame),
                    Poll::Ready(None) => break,
                    other => panic!("{model}: {other:?}"),
                }
                // Polled again without a tick: no frame, but the end.
                let again = poll_once(&mut h.body);
                if frames.last().unwrap().len() == FINAL_FRAME_BYTES {
                    assert_eq!(again, Poll::Ready(None), "{model}");
                    break;
                }
                assert!(again.is_pending(), "{model}: a frame without its tick");
                assert!(frames.len() < 100, "{model}: no end");
            }
            let (last, rest) = frames.split_last().unwrap();
            assert_eq!(last.len(), FINAL_FRAME_BYTES, "{model}");
            assert!(rest.iter().all(|f| f.len() == size), "{model}");
            assert!(
                rest.len() >= 4,
                "the text took several frames: {}",
                rest.len()
            );

            let chunks: Vec<serde_json::Value> = rest.iter().flat_map(|f| data_of(f)).collect();
            assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
            let text: String = chunks[1..]
                .iter()
                .map(|c| c["choices"][0]["delta"]["content"].as_str().unwrap())
                .collect();
            assert_eq!(text, nasty, "{model}");
            for c in &chunks {
                assert_eq!(c["id"], chunks[0]["id"]);
                assert_eq!(c["model"], model);
                assert_eq!(c["choices"][0]["finish_reason"], serde_json::Value::Null);
                assert!(c.get("x0ne").is_none() && c.get("usage").is_none(), "{c}");
            }

            let end = data_of(last);
            assert_eq!(end[0]["choices"][0]["finish_reason"], "stop");
            assert_eq!(end[0]["x0ne"]["receipt"]["k"], "v");
            assert_eq!(end.len(), 1 + usize::from(include_usage));
            if include_usage {
                assert_eq!(end[1]["choices"], serde_json::json!([]));
                assert_eq!(end[1]["usage"]["total_tokens"], 16);
            }
            assert!(last.ends_with(b"data: [DONE]\n\n"));
            assert_eq!(
                h.tick(now + TICK),
                Poll::Ready(None),
                "nothing after the end"
            );
        }
    }

    /// The x0ne block of a real H100 answer was 7322 bytes (2026-10-17). One
    /// of that size, a model of 256 bytes and the widest counters fit the last
    /// frame. A failure is a real receipt that cannot end its stream.
    #[test]
    fn the_last_frame_holds_an_x0ne_block_the_size_of_a_real_one() {
        const REAL_X0NE_BLOCK_BYTES: usize = 7322;
        let filler = REAL_X0NE_BLOCK_BYTES - r#"{"receipt":{"pad":""}}"#.len();
        let x0ne_json = format!(r#"{{"receipt":{{"pad":"{}"}}}}"#, "r".repeat(filler));
        assert_eq!(x0ne_json.len(), REAL_X0NE_BLOCK_BYTES);
        let widest = Shape {
            id: format!("chatcmpl-{}", "f".repeat(24)),
            created: u64::MAX,
            model: "m".repeat(crate::server::MAX_STREAM_MODEL_BYTES),
            include_usage: true,
        };
        let finale = Finale {
            finish_reason: "length",
            prompt_tokens: u32::MAX,
            completion_tokens: u32::MAX,
            x0ne_json,
        };
        let frame = last_frame(&widest, &finale).expect("a real receipt fits the last frame");
        assert_eq!(frame.len(), FINAL_FRAME_BYTES);
        let end = data_of(&frame);
        assert_eq!(end.len(), 2, "final chunk and usage chunk");
        assert_eq!(
            end[0]["x0ne"]["receipt"]["pad"].as_str().map(str::len),
            Some(filler)
        );
        assert_eq!(end[0]["choices"][0]["finish_reason"], "length");
        assert_eq!(end[1]["usage"]["total_tokens"], 2 * u64::from(u32::MAX));
        assert!(frame.ends_with(b"data: [DONE]\n\n"));
    }

    /// The largest x0ne block the last frame holds at the widest stream, and
    /// one byte more on a stream: an error frame of the last size, nothing of
    /// the receipt, no `[DONE]`, at frame [`crate::padding::MIN_TICKS`] as a
    /// final would be, then the body's end at once. Fixes what the code does.
    #[test]
    fn an_x0ne_block_past_the_last_frame_fails_closed() {
        let finale = |n: usize| {
            let filler = "r".repeat(n - r#"{"receipt":{"pad":""}}"#.len());
            Finale {
                finish_reason: "length",
                prompt_tokens: u32::MAX,
                completion_tokens: u32::MAX,
                x0ne_json: format!(r#"{{"receipt":{{"pad":"{filler}"}}}}"#),
            }
        };
        let room = |shape: &Shape| {
            (64..=FINAL_FRAME_BYTES)
                .rev()
                .find(|&n| last_frame(shape, &finale(n)).is_some())
                .unwrap()
        };
        let model = "m".repeat(crate::server::MAX_STREAM_MODEL_BYTES);
        let widest = Shape {
            id: format!("chatcmpl-{}", "f".repeat(24)),
            created: u64::MAX,
            model: model.clone(),
            include_usage: true,
        };
        // Measured 2026-10-20: twice the 7322-byte block of the H100 run, and more.
        assert_eq!(room(&widest), 15_416);

        let t0 = Instant::now();
        let mut h = Harness::new(&model, true, t0);
        let over = room(&shape(&model, true)) + 1;
        h.send(Event::Delta("a".into()));
        h.send(Event::Final(Box::new(finale(over))));
        h.hang_up();
        let frames = to_the_end(&mut h, t0);
        assert_eq!(frames.len() as u64, crate::padding::MIN_TICKS);
        let mut frames = frames;
        let error = frames.pop().unwrap();
        assert!(frames.iter().all(|f| f.len() == h.body.frame_size()));
        assert_eq!(error.len(), FINAL_FRAME_BYTES);
        let text = std::str::from_utf8(&error).unwrap();
        assert!(!text.contains("rrrr") && !text.contains("[DONE]"), "{text}");
        let events = data_of(&error);
        assert_eq!(events.len(), 1, "{text}");
        assert_eq!(events[0]["error"]["message"], RECEIPT_TOO_LARGE);
        assert_eq!(
            poll_once(&mut h.body),
            Poll::Ready(None),
            "no tick waited for"
        );
    }

    /// T20.
    /// A generation that stops sending ends the stream in error after
    /// [`STALL`]: padding up to a frame number that is a size, a frame of the
    /// last size, then the body's end. A failure is padding with no end,
    /// until the relay gives up, or an error that ends on its own frame.
    #[test]
    fn a_generator_silent_for_90_s_ends_the_stream_in_error() {
        let t0 = Instant::now();
        let mut h = Harness::new("m", false, t0);
        let size = h.body.frame_size();
        let Poll::Ready(Some(Ok(role))) = h.tick(t0) else {
            panic!("the role frame")
        };
        assert_eq!(role.len(), size);
        h.send(Event::Delta("a".into()));
        let heard = t0 + Duration::from_secs(1);
        let Poll::Ready(Some(Ok(text))) = h.tick(heard) else {
            panic!("the text frame")
        };
        assert_eq!(data_of(&text)[0]["choices"][0]["delta"]["content"], "a");
        let Poll::Ready(Some(Ok(comment))) = h.tick(heard + STALL - Duration::from_millis(1))
        else {
            panic!("a padding frame before the stall")
        };
        assert!(comment.starts_with(b":") && comment.len() == size);
        let mut rest = to_the_end(&mut h, heard + STALL - TICK);
        let error = rest.pop().unwrap();
        assert_eq!(3 + rest.len() + 1, crate::padding::MIN_TICKS as usize);
        assert!(rest.iter().all(|f| f.starts_with(b":") && f.len() == size));
        assert_eq!(error.len(), FINAL_FRAME_BYTES);
        let event = &data_of(&error)[0]["error"];
        assert_eq!(event["message"], STALLED);
        assert_eq!(event["code"], "generation_failed");
        assert_eq!(poll_once(&mut h.body), Poll::Ready(None));

        // A generation side gone without an end is the same, at the floor.
        let mut h = Harness::new("m", false, t0);
        let _role = h.tick(t0);
        h.send(Event::Delta("b".into()));
        h.hang_up();
        let mut rest = to_the_end(&mut h, t0);
        let error = rest.pop().unwrap();
        assert_eq!(1 + rest.len() + 1, crate::padding::MIN_TICKS as usize);
        assert!(
            rest.iter().all(|f| f.starts_with(b":")),
            "no text after a failure"
        );
        assert_eq!(error.len(), FINAL_FRAME_BYTES);
        assert_eq!(data_of(&error)[0]["error"]["message"], INTERNAL);
        assert!(
            !String::from_utf8_lossy(&error).contains("\"b\""),
            "pending text dropped"
        );
        assert_eq!(poll_once(&mut h.body), Poll::Ready(None));
    }

    /// Ticks from `from` until the body ends: every frame, the last included.
    fn to_the_end(h: &mut Harness, from: Instant) -> Vec<Bytes> {
        let mut frames = Vec::new();
        let mut now = from;
        loop {
            now += TICK;
            match h.tick(now) {
                Poll::Ready(Some(Ok(frame))) => frames.push(frame),
                Poll::Ready(None) => return frames,
                other => panic!("frame {}: {other:?}", frames.len() + 1),
            }
            if frames.last().is_some_and(|f| f.len() == FINAL_FRAME_BYTES) {
                assert_eq!(poll_once(&mut h.body), Poll::Ready(None), "the end at once");
                return frames;
            }
            assert!(frames.len() < 10_000, "no end");
        }
    }

    /// One stream whose text is `len` ASCII bytes, all there at the first
    /// tick, then its final: the length of each frame, the last included.
    fn trace_of(len: usize) -> Vec<usize> {
        let t0 = Instant::now();
        let mut h = Harness::new("m", true, t0);
        if len > 0 {
            h.send(Event::Delta("x".repeat(len)));
        }
        h.send(Event::Final(Box::new(Finale {
            finish_reason: "stop",
            prompt_tokens: 7,
            completion_tokens: u32::try_from(len).unwrap(),
            x0ne_json: r#"{"receipt":{"k":"v"}}"#.into(),
        })));
        h.hang_up();
        let mut frames = vec![match h.tick(t0) {
            Poll::Ready(Some(Ok(frame))) => frame,
            other => panic!("the role frame: {other:?}"),
        }];
        frames.extend(to_the_end(&mut h, t0));
        frames.iter().map(Bytes::len).collect()
    }

    /// Length padding.
    /// The number of frames of a stream, the last included, is a size of
    /// `crate::padding`, [`crate::padding::MIN_TICKS`] at least, whatever
    /// the length of its text: answers of 0 to 7 frames of text leave the
    /// same trace of sizes, frame for frame, and longer ones one of few. A
    /// failure is a stream whose frame count follows its text again.
    #[test]
    fn the_number_of_frames_is_a_size_whatever_the_length_of_the_text() {
        let room = Frames::new(&shape("m", true)).text_room;
        let floor = crate::padding::MIN_TICKS as usize;
        let short = trace_of(1);
        assert_eq!(short.len(), floor, "{short:?}");
        // Role, text frames, final: up to the floor, one trace.
        for len in [0, 1, 2, room - 1, room, room + 1, 7 * room] {
            assert_eq!(trace_of(len), short, "{len} bytes");
        }
        let mut counts = std::collections::BTreeSet::new();
        for len in (0..=200 * room).step_by(2 * room + 13) {
            let trace = trace_of(len);
            let n = trace.len() as u64;
            assert!(
                crate::padding::is_size(n, crate::padding::MIN_TICKS),
                "{len}: {n}"
            );
            let wanted = 2 + len.div_ceil(room) as u64;
            assert_eq!(
                n,
                crate::padding::size_for(wanted, crate::padding::MIN_TICKS),
                "{len}"
            );
            counts.insert(n);
        }
        // 98 lengths, up to 202 frames: 33 counts, 98 without padding.
        assert_eq!(counts.len(), 33, "{counts:?}");
    }

    /// Length padding, attack: a failure after k frames of text, for every k up to 40,
    /// still ends on a frame number that is a size, with an error frame of
    /// the last size, padding comments and no text after the failure. A
    /// failure is an error that tells its own position, hence the length
    /// generated so far.
    #[test]
    fn an_error_ends_on_a_size_wherever_it_comes() {
        let room = Frames::new(&shape("m", false)).text_room;
        for k in 0..=40usize {
            let t0 = Instant::now();
            let mut h = Harness::new("m", false, t0);
            let size = h.body.frame_size();
            h.send(Event::Delta("y".repeat(k * room)));
            let mut frames = Vec::new();
            for n in 0..=k {
                let Poll::Ready(Some(Ok(frame))) = h.tick(t0 + TICK * n as u32) else {
                    panic!("{k}: frame {n}");
                };
                frames.push(frame);
            }
            h.send(Event::Error(GENERATION_FAILED));
            h.send(Event::Delta("z".repeat(room)));
            frames.extend(to_the_end(&mut h, t0 + TICK * k as u32));
            let n = frames.len() as u64;
            assert!(
                crate::padding::is_size(n, crate::padding::MIN_TICKS),
                "{k}: {n}"
            );
            assert_eq!(
                n,
                crate::padding::size_for(k as u64 + 2, crate::padding::MIN_TICKS)
            );
            let (error, rest) = frames.split_last().unwrap();
            assert_eq!(error.len(), FINAL_FRAME_BYTES);
            assert_eq!(data_of(error)[0]["error"]["message"], GENERATION_FAILED);
            assert!(rest.iter().all(|f| f.len() == size), "{k}");
            assert!(
                rest[k + 1..].iter().all(|f| f.starts_with(b":")),
                "{k}: no text after the failure"
            );
        }
    }

    /// Length padding: the frames between the last text and the final, at most, stay
    /// under the relay's [`crate::local_api::CONTENT_SILENCE`] for every
    /// stream that fits [`crate::remote_verify::CHAT_TIMEOUT`]: 511 frames,
    /// 51.1 s. Rounding to powers of two instead would wait up to 819.1 s,
    /// and the relay would cut the stream. A failure is a rounding whose
    /// padding outlasts the relay's patience.
    #[test]
    fn the_padding_after_the_text_stays_under_the_relays_content_silence() {
        let frames = |d: Duration| (d.as_millis() / TICK.as_millis()) as u64;
        let (longest, silence) = (
            frames(crate::remote_verify::CHAT_TIMEOUT),
            frames(crate::local_api::CONTENT_SILENCE),
        );
        let tail = |round: &dyn Fn(u64) -> u64| (1..=longest).map(|n| round(n) - n).max().unwrap();
        let padme = tail(&|n| crate::padding::size_for(n, crate::padding::MIN_TICKS));
        assert_eq!(padme, 511);
        assert!(
            padme < silence,
            "{padme} frames of padding, {silence} of patience"
        );
        let powers_of_two = tail(&|n: u64| n.next_power_of_two());
        assert_eq!(powers_of_two, 8191);
        assert!(powers_of_two > silence);
    }

    /// Length padding: a stream whose text runs to frame 8800 would be rounded to 9216
    /// frames, 921.6 s, past the 900 s the relay waits for it
    /// ([`crate::remote_verify::CHAT_TIMEOUT`]), which then cuts it and loses
    /// the session. It ends within them; one whose text is out by frame 8600
    /// still ends on its final. A failure is a stream the padding pushes
    /// past the relay's patience.
    #[test]
    fn a_stream_ends_within_the_relays_fifteen_minutes() {
        let within = (crate::remote_verify::CHAT_TIMEOUT.as_millis() / TICK.as_millis()) as usize;
        for text_frames in [8_600usize, 8_800] {
            let t0 = Instant::now();
            let mut h = Harness::new("m", false, t0);
            let mut frames = Vec::new();
            let mut now = t0;
            let ended = |f: &Bytes| f.len() == FINAL_FRAME_BYTES;
            for n in 0..=text_frames {
                h.send(Event::Delta("x".into()));
                let Poll::Ready(Some(Ok(frame))) = h.tick(now) else {
                    panic!("{text_frames}: frame {n}");
                };
                frames.push(frame);
                if frames.last().is_some_and(ended) {
                    break;
                }
                now += TICK;
            }
            if !frames.last().is_some_and(ended) {
                h.send(Event::Final(Box::new(Finale {
                    finish_reason: "stop",
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    x0ne_json: r#"{"receipt":{"k":"v"}}"#.into(),
                })));
                h.hang_up();
                frames.extend(to_the_end(&mut h, now));
            }
            assert!(
                frames.len() <= within,
                "{text_frames} frames of text end at frame {}, past the {within} the relay waits",
                frames.len()
            );
            assert_eq!(frames.len() as u64, crate::padding::MAX_TICKS);
            let last = data_of(frames.last().unwrap());
            let finished = last
                .iter()
                .any(|d| d["choices"][0]["finish_reason"] == "stop");
            assert_eq!(finished, text_frames == 8_600, "{text_frames}: {last:?}");
            if !finished {
                assert_eq!(last[0]["error"]["message"], TOO_LONG);
                assert_eq!(poll_once(&mut h.body), Poll::Ready(None), "the end at once");
            }
        }
    }

    /// Frames hyper takes in one pass, which it flushes as one write: the
    /// body polled until it is pending.
    fn pass(body: &mut SseBody<Clock>) -> usize {
        let mut cx = Context::from_waker(std::task::Waker::noop());
        let mut n = 0;
        while let Poll::Ready(Some(Ok(_))) = Pin::new(&mut *body).poll_frame(&mut cx) {
            n += 1;
        }
        n
    }

    /// The production clock when its task is not run for a while (a loaded
    /// host; the host can stall a TD at will). Stalled after a frame
    /// is built, that frame and the next leave in one pass, so in one TLS
    /// record; stalled after a write, one late frame. Never a third: the
    /// tick after a late one is a whole [`TICK`] later. A failure is a burst
    /// of late frames back to back, or a frame early.
    #[tokio::test(start_paused = true)]
    async fn a_stalled_clock_writes_two_frames_at_once_never_more() {
        let (_events, rx) = unbounded_channel();
        let clock = Clock::start();
        let mut body = SseBody::new(rx, shape("m", false), clock, Instant::now());
        let step = |d: Duration| tokio::time::advance(d);
        assert_eq!(pass(&mut body), 0, "built at once, written later");
        step(PREPARE_AHEAD).await;
        assert_eq!(pass(&mut body), 1);
        for stall in [TICK + TICK / 2, TICK * 10, TICK * 100] {
            step(TICK.saturating_sub(PREPARE_AHEAD)).await;
            assert_eq!(pass(&mut body), 0, "built, not written yet");
            step(stall).await;
            assert_eq!(pass(&mut body), 2, "stalled {stall:?} after a build");
            step(TICK).await;
            assert_eq!(pass(&mut body), 0, "a whole tick after the late one");
            step(PREPARE_AHEAD).await;
            assert_eq!(pass(&mut body), 1);
            step(stall).await;
            assert_eq!(pass(&mut body), 1, "stalled {stall:?} after a write");
            step(PREPARE_AHEAD).await;
            assert_eq!(pass(&mut body), 0, "a whole tick after the late one");
        }
    }
}
