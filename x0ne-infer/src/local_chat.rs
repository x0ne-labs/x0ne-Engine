//! `x0ne-engine-proxy chat`: a terminal chat on a serving relay's local API,
//! the one `x0ne-engine cloud run` starts. A client like any other: the
//! loopback key, the whole conversation at each turn, and under each answer
//! whether the relay verified its receipt (`x0ne.receipt_verified`). To a
//! terminal the answer is streamed: its text shows as it comes, and it is
//! marked verified only once the stream has ended clean.

use std::future::Future;
use std::io::{IsTerminal as _, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tokio::io::{AsyncBufRead, AsyncBufReadExt};
use zeroize::Zeroizing;

use crate::sse::{Data, SseReader};
use crate::term_color::{paint, Style};

pub const DEFAULT_MAX_TOKENS: u32 = 1024;
/// The TD's ceiling: the relay's default `--max-completion-tokens`.
pub const MAX_TOKENS_CAP: u32 = crate::server::DEFAULT_MAX_COMPLETION_TOKENS as u32;
/// One message; a longer line is not sent.
const MAX_LINE: usize = 256 * 1024;
/// One answer, from the request to the verified receipt.
const ANSWER_TIMEOUT: Duration = Duration::from_mins(15);

pub const USAGE: &str = "\
  x0ne-engine-proxy chat --listen <addr> --key-file <file> [--max-tokens N]
                    a chat in this terminal on a serving session's local API;
                    leaving it does not end the session [max tokens: 1024, at most 4096]
";

#[derive(Debug, Clone, PartialEq)]
pub struct Opts {
    pub listen: SocketAddr,
    pub key_file: PathBuf,
    pub max_tokens: u32,
}

/// `--listen <addr> --key-file <file> [--max-tokens N]`, nothing else.
pub fn parse(args: &[String]) -> Result<Opts, String> {
    let (mut listen, mut key_file, mut max_tokens) = (None, None, DEFAULT_MAX_TOKENS);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        // Named before its value is looked for: an unknown one is never printed.
        let flag = match a.as_str() {
            f @ ("--listen" | "--key-file" | "--max-tokens") => f,
            _ => {
                return Err(format!(
                    "chat: unknown argument {} (chat takes --listen, --key-file and --max-tokens)",
                    crate::argv::shown(a)
                ))
            }
        };
        let v = it
            .next()
            .ok_or_else(|| format!("chat: {flag} needs a value"))?;
        match flag {
            "--listen" => {
                let addr = loopback(v)
                    .map_err(|_| crate::argv::refused(flag, "a loopback address and port"))?;
                listen = Some(addr);
            }
            // The key itself typed where its file goes: refused, never read.
            "--key-file" if crate::argv::is_hex_key(v) => {
                return Err(
                    "chat --key-file takes the path of the relay's key file, not the \
                     key: a key on a command line is visible to every local process"
                        .into(),
                )
            }
            "--key-file" => key_file = Some(PathBuf::from(v)),
            _ => {
                max_tokens = v
                    .parse::<u32>()
                    .ok()
                    .filter(|n| (1..=MAX_TOKENS_CAP).contains(n))
                    .ok_or_else(|| {
                        crate::argv::refused(flag, &format!("from 1 to {MAX_TOKENS_CAP}"))
                    })?;
            }
        }
    }
    Ok(Opts {
        listen: listen.ok_or("chat needs --listen")?,
        key_file: key_file.ok_or("chat needs --key-file")?,
        max_tokens,
    })
}

/// Loopback only: the key goes nowhere else.
pub fn loopback(listen: &str) -> Result<SocketAddr, String> {
    listen
        .parse::<SocketAddr>()
        .ok()
        .filter(|a| a.ip().is_loopback() && a.port() != 0)
        .ok_or_else(|| format!("{listen:?} is not a loopback address and port"))
}

/// The relay's key file: 64 hex characters and a newline. A path that holds
/// a key-like run of hex digits is not printed.
pub fn read_key(path: &Path) -> Result<Zeroizing<String>, String> {
    let shown = crate::argv::path_shown(&path.to_string_lossy());
    let len = std::fs::metadata(path)
        .map_err(|e| format!("--key-file {shown}: {e}"))?
        .len();
    if len > 1024 {
        return Err(format!("--key-file {shown}: not a key file"));
    }
    let raw = Zeroizing::new(std::fs::read(path).map_err(|e| format!("--key-file {shown}: {e}"))?);
    let key = std::str::from_utf8(&raw).unwrap_or("").trim();
    if key.len() != 64 || !key.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("--key-file {shown}: not a key file"));
    }
    Ok(Zeroizing::new(key.to_string()))
}

/// Text from the VM for the terminal: control characters other than newline
/// and tab become `?` (no escape sequence reaches the terminal).
pub fn printable(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_control() && c != '\n' && c != '\t' {
                '?'
            } else {
                c
            }
        })
        .collect()
}

/// One line of server text: printable, no newline, at most `max` characters.
fn clip(s: &str, max: usize) -> String {
    printable(s)
        .replace(['\n', '\t'], " ")
        .chars()
        .take(max)
        .collect()
}

/// The first 16 hex digits, for one line.
fn short(hex: &str) -> String {
    clip(hex, 16)
}

/// The line printed under each answer.
pub fn receipt_line(answer: &Value, secs: f64) -> String {
    let x = &answer["x0ne"];
    let mut s = if x["receipt_verified"] == Value::Bool(true) {
        let digest = x["receipt"]["model_digest_hex"]
            .as_str()
            .map(short)
            .unwrap_or_else(|| "?".into());
        format!("[receipt verified by the relay: attested session key, model digest {digest}")
    } else {
        "[receipt NOT verified: the answer carries no x0ne.receipt_verified".to_string()
    };
    if let Some(n) = answer["usage"]["completion_tokens"].as_u64() {
        s.push_str(&format!("; {n} tokens"));
    }
    s.push_str(&format!(" in {secs:.1} s"));
    if answer["choices"][0]["finish_reason"] == "length" {
        s.push_str("; cut at --max-tokens");
    }
    s.push(']');
    s
}

/// The relay's error text, or the body's start.
fn error_text(status: u16, body: &str) -> String {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let msg = v["error"]["message"]
        .as_str()
        .or_else(|| v["error"].as_str())
        .unwrap_or(body);
    format!("HTTP {status}: {}", clip(msg, 500))
}

/// The relay's error codes after which the session is stopping or lost.
const ENDS_THE_SESSION: [&str; 4] = [
    "receipt_verification_failed",
    "stream_protocol",
    "session_stopped",
    "stream_cut",
];

/// The chat's exit code when the relay stopped the session on its answer
/// (all of `ENDS_THE_SESSION` but `stream_cut`): `cloud run` then reads
/// the relay's own refusal. Every other failure of the chat exits 1.
pub const EXIT_SESSION_STOPPED: i32 = 3;

/// How the chat ended without the user: what to print, and the exit code.
#[derive(Debug)]
pub struct ChatEnd {
    pub message: String,
    pub code: i32,
}

impl From<String> for ChatEnd {
    fn from(message: String) -> Self {
        ChatEnd { message, code: 1 }
    }
}

impl From<&str> for ChatEnd {
    fn from(message: &str) -> Self {
        message.to_string().into()
    }
}

impl std::fmt::Display for ChatEnd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// A turn without a verified answer: its red line, and the chat's exit code
/// when the relay ended the session with it, which ends the chat.
#[derive(Debug)]
struct NoAnswer {
    line: String,
    ends: Option<i32>,
}

impl NoAnswer {
    fn turn(why: impl std::fmt::Display) -> Self {
        NoAnswer {
            line: format!("no answer: {why}"),
            ends: None,
        }
    }

    fn over(line: String) -> Self {
        NoAnswer {
            line,
            ends: Some(1),
        }
    }

    /// The relay is stopping the session: its refusal follows.
    fn stopping(line: String) -> Self {
        NoAnswer {
            line,
            ends: Some(EXIT_SESSION_STOPPED),
        }
    }

    fn cut() -> Self {
        Self::over(
            "stream cut before its receipt: the text above is not verified; the session is over"
                .into(),
        )
    }
}

/// A status other than 200, by the relay's code when it has one.
fn refused(status: u16, body: &str) -> NoAnswer {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let line = error_text(status, body);
    match v["error"]["code"].as_str() {
        Some("stream_cut") => NoAnswer::over(format!("no answer: {line}")),
        Some(code) if ENDS_THE_SESSION.contains(&code) => {
            NoAnswer::stopping(format!("no answer: {line}"))
        }
        _ => NoAnswer::turn(line),
    }
}

/// The error event of a stream, after which nothing is verified.
fn stream_error(e: &Value) -> NoAnswer {
    let message = e["message"].as_str().unwrap_or("");
    match e["code"].as_str().unwrap_or("") {
        "generation_failed" => NoAnswer::turn(clip(message, 500)),
        "receipt_verification_failed" => {
            let why = message
                .strip_prefix("receipt verification failed: ")
                .unwrap_or(message);
            let why = why.split("; the text streamed above").next().unwrap_or(why);
            NoAnswer::stopping(format!(
                "receipt FAILED: {}: the text above is NOT verified; the session is stopping",
                clip(why, 500)
            ))
        }
        "stream_cut" => NoAnswer::cut(),
        "stream_protocol" | "session_stopped" => {
            NoAnswer::stopping(format!("stream FAILED: {}", clip(message, 500)))
        }
        _ => NoAnswer::over(format!("stream FAILED: {}", clip(message, 500))),
    }
}

/// What a stream gave next.
enum Piece {
    Text(String),
    /// The clean end after the final chunk and `[DONE]`: what
    /// [`receipt_line`] reads, from the final chunk and the usage chunk.
    End(Value),
}

/// A streamed answer, read as it comes.
struct Streamed {
    resp: reqwest::Response,
    reader: SseReader,
    ready: std::collections::VecDeque<Data>,
    last: Option<Value>,
    usage: Option<Value>,
    done: bool,
}

impl Streamed {
    fn new(resp: reqwest::Response) -> Self {
        Streamed {
            resp,
            reader: SseReader::new(),
            ready: std::collections::VecDeque::new(),
            last: None,
            usage: None,
            done: false,
        }
    }

    /// The next text, or the end. `End` needs the final chunk, `[DONE]` and
    /// the body's clean end; a body that ends otherwise is a cut.
    async fn piece(&mut self) -> Result<Piece, NoAnswer> {
        loop {
            while let Some(data) = self.ready.pop_front() {
                if let Some(text) = self.take(data)? {
                    return Ok(Piece::Text(text));
                }
            }
            match self.resp.chunk().await {
                Ok(Some(bytes)) => {
                    let mut events = Vec::new();
                    self.reader.push(&bytes, &mut events).map_err(|e| {
                        NoAnswer::over(format!("stream FAILED: the local API's stream: {e}"))
                    })?;
                    self.ready.extend(events);
                }
                Ok(None) if self.done && self.reader.at_boundary() => {
                    return Ok(Piece::End(self.answer()))
                }
                Ok(None) | Err(_) => return Err(NoAnswer::cut()),
            }
        }
    }

    /// One event: its text, if it carries some to show.
    fn take(&mut self, data: Data) -> Result<Option<String>, NoAnswer> {
        let protocol = |what: &str| NoAnswer::over(format!("stream FAILED: {what}"));
        let chunk = match data {
            _ if self.done => return Err(protocol("an event after [DONE]")),
            Data::Done if self.last.is_some() => {
                self.done = true;
                return Ok(None);
            }
            Data::Done => return Err(protocol("[DONE] before the final chunk")),
            Data::Json(chunk) => chunk,
        };
        if let Some(e) = chunk.get("error") {
            return Err(stream_error(e));
        }
        if self.last.is_some() {
            let empty = chunk["choices"].as_array().is_some_and(|c| c.is_empty());
            if !empty || self.usage.is_some() {
                return Err(protocol("a chunk after the final that is not its usage"));
            }
            self.usage = Some(chunk["usage"].clone());
            return Ok(None);
        }
        let choice = &chunk["choices"][0];
        if choice["finish_reason"].is_string() {
            self.last = Some(chunk);
            return Ok(None);
        }
        Ok(choice["delta"]["content"]
            .as_str()
            .filter(|t| !t.is_empty())
            .map(str::to_string))
    }

    fn answer(&self) -> Value {
        let last = self.last.as_ref().unwrap_or(&Value::Null);
        json!({
            "x0ne": last["x0ne"],
            "choices": [{ "finish_reason": last["choices"][0]["finish_reason"] }],
            "usage": self.usage.clone().unwrap_or(Value::Null),
        })
    }
}

/// The local API, with the relay's key.
pub struct Api {
    http: reqwest::Client,
    base: String,
    bearer: Zeroizing<String>,
}

impl Api {
    pub fn new(listen: SocketAddr, bearer: Zeroizing<String>) -> Result<Api, String> {
        if !listen.ip().is_loopback() {
            return Err(format!("{listen} is not a loopback address"));
        }
        let http = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_secs(5))
            .timeout(ANSWER_TIMEOUT)
            .build()
            .map_err(|e| format!("HTTP client: {e}"))?;
        Ok(Api {
            http,
            base: format!("http://{listen}"),
            bearer,
        })
    }

    async fn call(&self, req: reqwest::RequestBuilder) -> Result<Value, String> {
        let r = req
            .bearer_auth(self.bearer.as_str())
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = r.status().as_u16();
        let text = r.text().await.map_err(|e| e.to_string())?;
        if status != 200 {
            return Err(error_text(status, &text));
        }
        serde_json::from_str(&text).map_err(|e| format!("not JSON: {e}"))
    }

    async fn status(&self) -> Result<Value, String> {
        self.call(self.http.get(format!("{}/x0ne/v1/status", self.base)))
            .await
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response, NoAnswer> {
        self.http
            .post(format!("{}/v1/chat/completions", self.base))
            .bearer_auth(self.bearer.as_str())
            .json(body)
            .send()
            .await
            .map_err(NoAnswer::turn)
    }

    async fn complete(&self, body: &Value) -> Result<Value, NoAnswer> {
        let r = self.post(body).await?;
        let status = r.status().as_u16();
        let text = r.text().await.map_err(NoAnswer::turn)?;
        if status != 200 {
            return Err(refused(status, &text));
        }
        serde_json::from_str(&text).map_err(|e| NoAnswer::turn(format!("not JSON: {e}")))
    }

    async fn open_stream(&self, body: &Value) -> Result<Streamed, NoAnswer> {
        let r = self.post(body).await?;
        let status = r.status().as_u16();
        if status != 200 {
            let text = r.text().await.map_err(NoAnswer::turn)?;
            return Err(refused(status, &text));
        }
        let content_type = r
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok());
        if content_type != Some("text/event-stream") {
            return Err(NoAnswer::over(
                "stream FAILED: a 200 that is not text/event-stream".into(),
            ));
        }
        Ok(Streamed::new(r))
    }
}

fn note(err: &mut impl Write, s: &str) {
    let _ = err.write_all(s.as_bytes());
    let _ = err.flush();
}

fn show(out: &mut impl Write, s: &str) -> Result<(), String> {
    out.write_all(s.as_bytes())
        .and_then(|()| out.flush())
        .map_err(|e| format!("standard output: {e}"))
}

/// Seconds counted on a terminal while `f` runs, erased when it ends.
async fn counting<F: Future>(f: F, err: &mut impl Write, show_it: bool) -> F::Output {
    if !show_it {
        return f.await;
    }
    tokio::pin!(f);
    let t0 = Instant::now();
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    loop {
        tokio::select! {
            r = &mut f => {
                note(err, "\r\x1b[K");
                return r;
            }
            _ = tick.tick() => note(err, &format!("\r  ... {} s", t0.elapsed().as_secs())),
        }
    }
}

/// A whole answer: shown once the relay has verified it. The text shown,
/// and the answer or why there is none; Err when `out` fails.
async fn whole_turn(
    api: &Api,
    body: &Value,
    out: &mut impl Write,
    err: &mut impl Write,
    counter: bool,
) -> Result<(String, Result<Value, NoAnswer>), String> {
    let answer = counting(api.complete(body), err, counter).await;
    match answer {
        Ok(a) => match a["choices"][0]["message"]["content"].as_str() {
            Some(t) => {
                show(out, &printable(t))?;
                Ok((t.to_string(), Ok(a)))
            }
            None => Ok((
                String::new(),
                Err(NoAnswer::turn("the answer carries no text")),
            )),
        },
        Err(e) => Ok((String::new(), Err(e))),
    }
}

/// A streamed answer: each piece shown as it comes, the counter until the
/// first. The answer only at a verified end, after the text shown.
async fn stream_turn(
    api: &Api,
    body: &Value,
    out: &mut impl Write,
    err: &mut impl Write,
    counter: bool,
) -> Result<(String, Result<Value, NoAnswer>), String> {
    let first = async {
        let mut s = api.open_stream(body).await?;
        let piece = s.piece().await?;
        Ok::<_, NoAnswer>((s, piece))
    };
    let (mut s, mut piece) = match counting(first, err, counter).await {
        Ok(opened) => opened,
        Err(e) => return Ok((String::new(), Err(e))),
    };
    let mut said = String::new();
    loop {
        match piece {
            Piece::Text(t) => {
                show(out, &printable(&t))?;
                said.push_str(&t);
            }
            Piece::End(answer) => return Ok((said, Ok(answer))),
        }
        piece = match s.piece().await {
            Ok(p) => p,
            Err(e) => return Ok((said, Err(e))),
        };
    }
}

/// Lines from `input`, answers on `out`, notes and receipts on `err`. Ends
/// at end of input, `/exit` or `/bye`; Err when `out` or `input` fails, or
/// when the relay ended the session on an answer (a failed receipt, a cut).
/// `color`: `err` is colored (the time left yellow, a verified receipt dim
/// green, a missing receipt or answer red); `out`, the model's text, never is.
/// `stream`: each answer is shown as it comes, marked verified at its end.
#[allow(clippy::fn_params_excessive_bools)]
pub async fn converse(
    api: &Api,
    max_tokens: u32,
    mut input: impl AsyncBufRead + Unpin,
    out: &mut impl Write,
    err: &mut impl Write,
    interactive: bool,
    counter: bool,
    color: bool,
    stream: bool,
) -> Result<(), ChatEnd> {
    let st = api
        .status()
        .await
        .map_err(|e| format!("the local API at {}: {e}", api.base))?;
    let model = st["model"]["name"]
        .as_str()
        .map(|m| clip(m, 64))
        .ok_or("the local API names no model")?;
    let left = st["seconds_left"].as_u64().unwrap_or(0) / 60;
    note(
        err,
        &format!(
            "Chat with {model} ({}). /exit or Ctrl-D to leave.\n",
            paint(color, Style::Warn, &format!("{left} min left"))
        ),
    );
    let mut history: Vec<Value> = Vec::new();
    loop {
        if interactive {
            show(out, ">>> ")?;
        }
        let mut line = Vec::new();
        let n = input
            .read_until(b'\n', &mut line)
            .await
            .map_err(|e| format!("reading your input: {e}"))?;
        if n == 0 {
            if interactive {
                show(out, "\n")?;
            }
            return Ok(());
        }
        if line.len() > MAX_LINE {
            note(
                err,
                &format!(
                    "  {}\n",
                    paint(
                        color,
                        Style::Warn,
                        "(a message is at most 256 KiB: not sent)"
                    )
                ),
            );
            continue;
        }
        let text = String::from_utf8_lossy(&line)
            .trim_end_matches(['\n', '\r'])
            .to_string();
        match text.trim() {
            "" => continue,
            "/exit" | "/bye" => return Ok(()),
            c if c.starts_with('/') => {
                note(
                    err,
                    "  /exit or /bye (or Ctrl-D) leaves; nothing else starts with /\n",
                );
                continue;
            }
            _ => {}
        }
        history.push(json!({ "role": "user", "content": text }));
        let mut body = json!({ "model": model, "messages": history, "max_tokens": max_tokens });
        if stream {
            body["stream"] = json!(true);
            body["stream_options"] = json!({ "include_usage": true });
        }
        let t0 = Instant::now();
        let (said, answer) = if stream {
            stream_turn(api, &body, out, err, counter).await?
        } else {
            whole_turn(api, &body, out, err, counter).await?
        };
        let secs = t0.elapsed().as_secs_f64();
        match answer {
            Ok(a) => {
                if !said.ends_with('\n') {
                    show(out, "\n")?;
                }
                let style = if a["x0ne"]["receipt_verified"] == Value::Bool(true) {
                    Style::Receipt
                } else {
                    Style::Fail
                };
                note(
                    err,
                    &format!("  {}\n", paint(color, style, &receipt_line(&a, secs))),
                );
                history.push(json!({ "role": "assistant", "content": said }));
            }
            Err(no) => {
                if !said.is_empty() && !said.ends_with('\n') {
                    show(out, "\n")?;
                }
                history.pop();
                let line = format!("[{}]", no.line);
                note(err, &format!("  {}\n", paint(color, Style::Fail, &line)));
                if let Some(code) = no.ends {
                    return Err(ChatEnd {
                        message: "the session is over: its last answer was not verified".into(),
                        code,
                    });
                }
            }
        }
    }
}

/// Whether answers to `out` are streamed: only to a terminal, so that a
/// redirected output never holds text the relay had not verified.
pub fn streams_to(out: &impl std::io::IsTerminal) -> bool {
    out.is_terminal()
}

/// `x0ne-engine-proxy chat …` on this process's terminal.
pub async fn run(o: &Opts) -> Result<(), ChatEnd> {
    let api = Api::new(o.listen, read_key(&o.key_file)?)?;
    let interactive = std::io::stdin().is_terminal();
    let counter = std::io::stderr().is_terminal();
    let input = tokio::io::BufReader::new(tokio::io::stdin());
    converse(
        &api,
        o.max_tokens,
        input,
        &mut std::io::stdout(),
        &mut std::io::stderr(),
        interactive,
        counter,
        crate::term_color::stderr_on(),
        streams_to(&std::io::stdout()),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn the_receipt_line_says_whether_the_relay_verified_it() {
        let a = json!({
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 7},
            "x0ne": {"receipt_verified": true, "receipt": {"model_digest_hex": "a".repeat(64)}},
        });
        assert_eq!(
            receipt_line(&a, 1.26),
            format!(
                "[receipt verified by the relay: attested session key, model digest {}; 7 tokens in 1.3 s]",
                "a".repeat(16)
            )
        );
        let b =
            json!({"choices": [{"finish_reason": "length"}], "x0ne": {"receipt_verified": false}});
        assert_eq!(
            receipt_line(&b, 2.0),
            "[receipt NOT verified: the answer carries no x0ne.receipt_verified in 2.0 s; cut at --max-tokens]"
        );
    }

    #[test]
    fn model_text_cannot_drive_the_terminal() {
        assert_eq!(printable("a\x1b[2Jb\n\tc\r"), "a?[2Jb\n\tc?");
        assert_eq!(clip("a\nb\x07c", 3), "a b");
    }

    #[test]
    fn the_key_goes_to_the_loopback_only() {
        assert!(loopback("127.0.0.1:8478").is_ok());
        assert!(loopback("[::1]:8478").is_ok());
        assert!(loopback("203.0.113.7:8478").is_err());
        assert!(loopback("localhost:8478").is_err());
        assert!(loopback("127.0.0.1:0").is_err());
        assert!(Api::new(
            "203.0.113.7:8478".parse().unwrap(),
            Zeroizing::new("k".into())
        )
        .is_err());
    }

    #[test]
    fn chat_takes_three_options_and_needs_two() {
        let p = |s: &str| parse(&s.split_whitespace().map(String::from).collect::<Vec<_>>());
        let o = p("--listen 127.0.0.1:9 --key-file /k --max-tokens 64").unwrap();
        assert_eq!((o.listen.port(), o.max_tokens), (9, 64));
        assert_eq!(
            p("--listen 127.0.0.1:9 --key-file /k").unwrap().max_tokens,
            1024
        );
        assert!(p("--listen 127.0.0.1:9").is_err());
        assert!(p("--key-file /k").is_err());
        assert!(p("--listen 203.0.113.7:9 --key-file /k").is_err());
        assert!(p("--listen 127.0.0.1:9 --key-file /k --max-tokens 4097").is_err());
        assert!(p("--listen 127.0.0.1:9 --key-file /k --yes 1").is_err());
    }

    /// An unknown argument is named, never printed; a known flag is.
    #[test]
    fn an_unknown_argument_is_refused_without_its_value() {
        let key = "5ec7".repeat(16);
        for args in [
            vec![key.clone()],
            vec![format!("--key-hex={key}")],
            vec!["--key-hex".to_string(), key.clone()],
            vec![format!("X0NE_INFER_KEY_HEX={key}")],
            vec![
                "--listen".to_string(),
                "127.0.0.1:9".to_string(),
                key.clone(),
            ],
        ] {
            let e = parse(&args).unwrap_err();
            assert!(e.contains("unknown"), "{args:?}: {e}");
            assert!(!e.contains("5ec75ec7"), "{args:?}: {e}");
        }
        assert_eq!(
            parse(&["--key-file".to_string()]).unwrap_err(),
            "chat: --key-file needs a value"
        );
    }

    #[test]
    fn errors_quote_the_relay_message() {
        assert_eq!(
            error_text(
                503,
                r#"{"error":{"message":"the attested VM is lost","type":"x0ne_engine"}}"#
            ),
            "HTTP 503: the attested VM is lost"
        );
        assert_eq!(error_text(502, "bad\x1b[0m"), "HTTP 502: bad?[0m");
    }

    #[test]
    fn a_key_file_is_64_hex_characters() {
        let d = tempfile::tempdir().unwrap();
        let k = d.path().join("local.key");
        std::fs::write(&k, format!("{}\n", "c".repeat(64))).unwrap();
        assert_eq!(read_key(&k).unwrap().as_str(), "c".repeat(64));
        std::fs::write(&k, "not a key\n").unwrap();
        assert!(read_key(&k).is_err());
        assert!(read_key(&d.path().join("absent")).is_err());
    }

    type Seen = Arc<Mutex<Vec<Value>>>;

    /// The local API as the chat sees it: status, then completions echoing
    /// the last message; no receipt for "no-receipt", an escape for "escape".
    async fn fake_api() -> (SocketAddr, Seen) {
        use axum::extract::State;
        use axum::http::{HeaderMap, StatusCode};
        use axum::routing::{get, post};
        async fn status(h: HeaderMap) -> (StatusCode, axum::Json<Value>) {
            if h.get("authorization").and_then(|v| v.to_str().ok())
                != Some(format!("Bearer {}", "c".repeat(64)).as_str())
            {
                return (
                    StatusCode::UNAUTHORIZED,
                    axum::Json(json!({"error": {"message": "bad key"}})),
                );
            }
            (
                StatusCode::OK,
                axum::Json(json!({"model": {"name": "m:1b"}, "seconds_left": 1740})),
            )
        }
        async fn complete(
            State(seen): State<Seen>,
            axum::Json(b): axum::Json<Value>,
        ) -> axum::Json<Value> {
            seen.lock().unwrap().push(b.clone());
            let msgs = b["messages"].as_array().cloned().unwrap_or_default();
            let users = msgs.iter().filter(|m| m["role"] == "user").count();
            let last = msgs
                .last()
                .and_then(|m| m["content"].as_str())
                .unwrap_or("")
                .to_string();
            let mut text = format!("echo {users}: {last}");
            if last.contains("escape") {
                text.push_str(" \u{1b}[31mred");
            }
            let mut a = json!({
                "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5},
                "x0ne": {"receipt_verified": true, "receipt": {"model_digest_hex": "d".repeat(64)}},
            });
            if last.contains("no-receipt") {
                a.as_object_mut().unwrap().remove("x0ne");
            }
            axum::Json(a)
        }
        let seen: Seen = Arc::default();
        let app = axum::Router::new()
            .route("/x0ne/v1/status", get(status))
            .route("/v1/chat/completions", post(complete))
            .with_state(seen.clone());
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(l, app).await });
        (addr, seen)
    }

    #[tokio::test]
    async fn a_conversation_sends_its_history_and_shows_each_receipt() {
        let (addr, seen) = fake_api().await;
        let api = Api::new(addr, Zeroizing::new("c".repeat(64))).unwrap();
        let input: &[u8] = b"hello\n\n/help\nagain\nno-receipt\nescape\n/exit\nnever sent\n";
        let (mut out, mut err) = (Vec::new(), Vec::new());
        converse(
            &api, 64, input, &mut out, &mut err, true, false, false, false,
        )
        .await
        .unwrap();
        let out = String::from_utf8(out).unwrap();
        let err = String::from_utf8(err).unwrap();
        assert!(
            err.starts_with("Chat with m:1b (29 min left). /exit or Ctrl-D to leave.\n"),
            "{err}"
        );
        assert!(out.contains("echo 1: hello\n"), "{out}");
        assert!(out.contains("echo 2: again\n"), "{out}");
        assert!(out.contains("echo 4: escape ?[31mred\n"), "{out:?}");
        assert!(!out.contains('\u{1b}'), "{out:?}");
        assert!(out.starts_with(">>> "), "{out}");
        let verified = format!(
            "  [receipt verified by the relay: attested session key, model digest {}; 5 tokens in ",
            "d".repeat(16)
        );
        assert_eq!(err.matches(&verified).count(), 3, "{err}");
        assert_eq!(err.matches("  [receipt NOT verified").count(), 1, "{err}");
        assert!(err.contains("nothing else starts with /"), "{err}");
        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 4, "/exit stops reading: {seen:?}");
        assert_eq!(
            seen[1],
            json!({
                "model": "m:1b",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "echo 1: hello"},
                    {"role": "user", "content": "again"},
                ],
                "max_tokens": 64,
            })
        );
    }

    /// Colored, the notes only: time left yellow, a verified receipt dim
    /// green, a missing one red. Stripped, the same text as uncolored.
    #[tokio::test]
    async fn colors_go_to_the_notes_and_never_to_the_answers() {
        let (addr, _) = fake_api().await;
        let api = Api::new(addr, Zeroizing::new("c".repeat(64))).unwrap();
        let input: &[u8] = b"hello\nno-receipt\nescape\n/exit\n";
        let mut runs = Vec::new();
        for color in [false, true] {
            let (mut out, mut err) = (Vec::new(), Vec::new());
            converse(
                &api, 64, input, &mut out, &mut err, true, false, color, false,
            )
            .await
            .unwrap();
            runs.push((
                String::from_utf8(out).unwrap(),
                String::from_utf8(err).unwrap(),
            ));
        }
        let ((plain_out, plain_err), (out, err)) = (&runs[0], &runs[1]);
        assert!(
            err.starts_with("Chat with m:1b (\x1b[33m29 min left\x1b[0m). /exit"),
            "{err:?}"
        );
        let verified = "  \x1b[2;32m[receipt verified by the relay: attested session key";
        assert_eq!(err.matches(verified).count(), 2, "{err:?}");
        assert_eq!(
            err.matches("  \x1b[31m[receipt NOT verified").count(),
            1,
            "{err:?}"
        );
        assert!(
            !plain_err.contains('\x1b') && !out.contains('\x1b'),
            "{out:?}"
        );
        assert_eq!(out, plain_out);
        let timeless = |s: &str| -> Vec<String> {
            s.lines()
                .map(|l| l.split(" tokens in ").next().unwrap_or("").to_string())
                .collect()
        };
        assert_eq!(
            timeless(&crate::term_color::strip(err)),
            timeless(plain_err)
        );
    }

    #[tokio::test]
    async fn a_wrong_key_or_no_api_is_an_error_before_any_question() {
        let (addr, seen) = fake_api().await;
        let api = Api::new(addr, Zeroizing::new("0".repeat(64))).unwrap();
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let e = converse(
            &api,
            64,
            &b"hello\n"[..],
            &mut out,
            &mut err,
            false,
            false,
            false,
            false,
        )
        .await
        .unwrap_err();
        assert!(e.message.contains("HTTP 401: bad key"), "{e}");
        assert_eq!(e.code, 1);
        assert!(seen.lock().unwrap().is_empty());
    }

    // ── Streamed answers ────────────────────────────────────────────────────

    use hyper::body::{Bytes, Frame};

    /// One event as the relay writes it.
    fn event(v: &Value) -> Bytes {
        Bytes::from(format!("data: {v}\n\n"))
    }

    fn role() -> Value {
        json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]})
    }

    fn delta(text: &str) -> Value {
        json!({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]})
    }

    /// The final chunk, the usage chunk and `[DONE]`, in one write, as the relay sends them.
    fn last() -> Bytes {
        let fin = json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "x0ne": {"receipt_verified": true, "receipt": {"model_digest_hex": "d".repeat(64)}},
        });
        let usage = json!({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}});
        Bytes::from(format!("data: {fin}\n\ndata: {usage}\n\ndata: [DONE]\n\n"))
    }

    fn failure(code: &str, message: &str) -> Bytes {
        event(&json!({"error": {"message": message, "type": "x0ne_engine", "code": code}}))
    }

    /// A body the test writes as it goes: `Some` bytes, then a clean end
    /// once the sender drops, or a broken body after `None`.
    struct Scripted {
        rx: tokio::sync::mpsc::UnboundedReceiver<Option<Bytes>>,
        breaking: bool,
    }

    impl hyper::body::Body for Scripted {
        type Data = Bytes;
        type Error = std::io::Error;

        fn poll_frame(
            self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Result<Frame<Bytes>, std::io::Error>>> {
            use std::task::Poll;
            let this = self.get_mut();
            if this.breaking {
                return Poll::Ready(Some(Err(std::io::Error::other("cut"))));
            }
            match std::task::ready!(this.rx.poll_recv(cx)) {
                Some(Some(bytes)) => Poll::Ready(Some(Ok(Frame::data(bytes)))),
                // Pending once: hyper flushes what it holds before the error.
                Some(None) => {
                    this.breaking = true;
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
                None => Poll::Ready(None),
            }
        }
    }

    type Pen = tokio::sync::mpsc::UnboundedSender<Option<Bytes>>;

    /// What the scripted API answers, one per question, in order.
    enum Reply {
        Stream(Scripted),
        Status(u16, Value),
    }

    fn scripted() -> (Pen, Reply) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (
            tx,
            Reply::Stream(Scripted {
                rx,
                breaking: false,
            }),
        )
    }

    /// A stream whose writes are all made already.
    fn written(parts: Vec<Option<Bytes>>) -> Reply {
        let (pen, reply) = scripted();
        for part in parts {
            pen.send(part).unwrap();
        }
        reply
    }

    /// The status of [`fake_api`], and completions answered by `replies`.
    async fn scripted_api(replies: Vec<Reply>) -> (SocketAddr, Seen) {
        use axum::extract::State;
        use axum::http::StatusCode;
        use axum::response::{IntoResponse, Response};
        use axum::routing::{get, post};
        type Replies = Arc<Mutex<std::collections::VecDeque<Reply>>>;
        async fn status() -> axum::Json<Value> {
            axum::Json(json!({"model": {"name": "m:1b"}, "seconds_left": 1740}))
        }
        async fn complete(
            State((seen, replies)): State<(Seen, Replies)>,
            axum::Json(b): axum::Json<Value>,
        ) -> Response {
            seen.lock().unwrap().push(b);
            let reply = replies.lock().unwrap().pop_front();
            match reply {
                Some(Reply::Stream(body)) => (
                    [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                    axum::body::Body::new(body),
                )
                    .into_response(),
                Some(Reply::Status(code, v)) => {
                    (StatusCode::from_u16(code).unwrap(), axum::Json(v)).into_response()
                }
                None => StatusCode::GONE.into_response(),
            }
        }
        let seen: Seen = Arc::default();
        let replies: Replies = Arc::new(Mutex::new(replies.into()));
        let app = axum::Router::new()
            .route("/x0ne/v1/status", get(status))
            .route("/v1/chat/completions", post(complete))
            .with_state((seen.clone(), replies));
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(l, app).await });
        (addr, seen)
    }

    /// What was written where, in order: `o` stdout, `e` stderr, `s` a mark
    /// of the scripted API.
    #[derive(Clone, Default)]
    struct Tape(Arc<Mutex<Vec<(char, String)>>>);

    struct Side(Tape, char);

    impl Write for Side {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            let record = (self.1, String::from_utf8_lossy(b).into_owned());
            self.0 .0.lock().unwrap().push(record);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Tape {
        fn side(&self, side: char) -> Side {
            Side(self.clone(), side)
        }

        fn mark(&self, what: &str) {
            self.0.lock().unwrap().push(('s', what.to_string()));
        }

        /// Everything written on `side`.
        fn of(&self, side: char) -> String {
            let records = self.0.lock().unwrap();
            records
                .iter()
                .filter(|(s, _)| *s == side)
                .map(|(_, t)| t.as_str())
                .collect()
        }

        /// Where the first record on `side` that holds `needle` is.
        fn at(&self, side: char, needle: &str) -> Option<usize> {
            let records = self.0.lock().unwrap();
            records
                .iter()
                .position(|(s, t)| *s == side && t.contains(needle))
        }

        /// Returns once `side` holds `text`; 5 s at most.
        async fn shows(&self, side: char, text: &str) {
            let deadline = Instant::now() + Duration::from_secs(5);
            while !self.of(side).contains(text) {
                assert!(Instant::now() < deadline, "never shown: {text:?}");
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }
    }

    fn key() -> Zeroizing<String> {
        Zeroizing::new("c".repeat(64))
    }

    /// C1.
    /// Each piece is on stdout before the API sends the next, made printable
    /// piece by piece. The receipt line comes once per answer, after the
    /// body's end, not at the final chunk. The streamed text joins the history.
    #[tokio::test]
    async fn a_streamed_answer_is_printed_as_it_arrives_then_marked_verified() {
        let (pen, first) = scripted();
        let second = written(vec![
            Some(event(&role())),
            Some(event(&delta("fine"))),
            Some(last()),
        ]);
        let (addr, seen) = scripted_api(vec![first, second]).await;
        let api = Api::new(addr, key()).unwrap();
        let tape = Tape::default();
        let (mut out, mut err) = (tape.side('o'), tape.side('e'));
        let input: &[u8] = b"hello\nagain\n";
        let chat = converse(
            &api, 64, input, &mut out, &mut err, false, false, false, true,
        );
        let script = async {
            pen.send(Some(event(&role()))).unwrap();
            for piece in ["Bon", "jour \u{1b}[2J", "\u{e9}t\u{e9}"] {
                tape.mark(&format!("sent {piece}"));
                pen.send(Some(event(&delta(piece)))).unwrap();
                tape.shows('o', &printable(piece)).await;
            }
            pen.send(Some(last())).unwrap();
            tokio::time::sleep(Duration::from_millis(300)).await;
            assert!(!tape.of('e').contains("[receipt"), "{:?}", tape.of('e'));
            tape.mark("ended");
            drop(pen);
        };
        let (chat, ()) = tokio::join!(chat, script);
        chat.unwrap();
        assert_eq!(tape.of('o'), "Bonjour ?[2J\u{e9}t\u{e9}\nfine\n");
        let receipt = format!(
            "  [receipt verified by the relay: attested session key, model digest {}; 5 tokens in ",
            "d".repeat(16)
        );
        let notes = tape.of('e');
        assert_eq!(notes.matches(&receipt).count(), 2, "{notes:?}");
        assert!(tape.at('e', &receipt) > tape.at('s', "ended"), "{notes:?}");
        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen[0]["stream"], true);
        assert_eq!(seen[0]["stream_options"], json!({"include_usage": true}));
        assert_eq!(
            seen[1]["messages"][1],
            json!({"role": "assistant", "content": "Bonjour \u{1b}[2J\u{e9}t\u{e9}"})
        );
    }

    /// C2.
    /// A generation that failed after its text: red, out of the history, the
    /// chat goes on. A receipt that failed after its text: red, and the chat
    /// ends in error, nothing more sent.
    #[tokio::test]
    async fn a_receipt_failure_after_streamed_text_is_red_leaves_the_history_and_fails() {
        let failed = written(vec![
            Some(event(&role())),
            Some(event(&delta("partial"))),
            Some(failure(
                "generation_failed",
                "the attested VM reported a failed generation: the text above is not verified",
            )),
            None,
        ]);
        let forged = written(vec![
            Some(event(&role())),
            Some(event(&delta("shown"))),
            Some(failure(
                "receipt_verification_failed",
                "receipt verification failed: the output commitment does not open to the \
                 answer text; the text streamed above is NOT verified and must be discarded; \
                 the session is being stopped",
            )),
            None,
        ]);
        let (addr, seen) = scripted_api(vec![failed, forged]).await;
        let api = Api::new(addr, key()).unwrap();
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let input: &[u8] = b"one\ntwo\nthree\n";
        let e = converse(
            &api, 64, input, &mut out, &mut err, false, false, true, true,
        )
        .await
        .unwrap_err();
        assert!(e.message.contains("the session is over"), "{e}");
        assert_eq!(e.code, EXIT_SESSION_STOPPED);
        let out = String::from_utf8(out).unwrap();
        let err = String::from_utf8(err).unwrap();
        assert_eq!(out, "partial\nshown\n");
        assert!(
            err.contains(
                "  \x1b[31m[no answer: the attested VM reported a failed generation: \
                 the text above is not verified]\x1b[0m\n"
            ),
            "{err:?}"
        );
        assert!(
            err.contains(
                "  \x1b[31m[receipt FAILED: the output commitment does not open to the answer \
                 text: the text above is NOT verified; the session is stopping]\x1b[0m\n"
            ),
            "{err:?}"
        );
        assert!(!err.contains("receipt verified"), "{err:?}");
        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 2, "nothing sent after the failed receipt");
        assert_eq!(
            seen[1]["messages"],
            json!([{"role": "user", "content": "two"}])
        );
    }

    /// A stream that ends without its receipt, however it ends, is a cut:
    /// never marked verified, even with the final chunk and `[DONE]` read.
    #[tokio::test]
    async fn a_stream_that_ends_without_its_receipt_is_a_cut_and_ends_the_chat() {
        let start = || vec![Some(event(&role())), Some(event(&delta("a")))];
        let ends: [Vec<Option<Bytes>>; 4] = [
            vec![None],
            vec![],
            vec![Some(last()), None],
            vec![
                Some(failure(
                    "stream_cut",
                    "the stream from the attested VM ended",
                )),
                None,
            ],
        ];
        for end in ends {
            let parts = start().into_iter().chain(end).collect();
            let (addr, seen) = scripted_api(vec![written(parts)]).await;
            let api = Api::new(addr, key()).unwrap();
            let (mut out, mut err) = (Vec::new(), Vec::new());
            let input: &[u8] = b"one\ntwo\n";
            let e = converse(
                &api, 64, input, &mut out, &mut err, false, false, false, true,
            )
            .await
            .unwrap_err();
            assert!(e.message.contains("the session is over"), "{e}");
            assert_eq!(e.code, 1);
            let err = String::from_utf8(err).unwrap();
            assert_eq!(String::from_utf8(out).unwrap(), "a\n");
            assert!(
                err.contains(
                    "  [stream cut before its receipt: the text above is not verified; \
                     the session is over]\n"
                ),
                "{err:?}"
            );
            assert!(!err.contains("receipt verified"), "{err:?}");
            assert_eq!(seen.lock().unwrap().len(), 1);
        }
    }

    /// C3.
    /// Redirected, stdout is not a terminal: whole answers, verified before
    /// any text is written. A whole answer whose receipt failed, the relay's
    /// coded 502, ends the chat in error; another refusal does not.
    #[tokio::test]
    async fn a_redirected_stdout_asks_for_a_whole_verified_answer() {
        assert!(!streams_to(&tempfile::tempfile().unwrap()));
        let busy = Reply::Status(
            503,
            json!({"error": {"message": "busy", "type": "x0ne_engine"}}),
        );
        let forged = Reply::Status(
            502,
            json!({"error": {
                "message": "receipt verification failed: the receipt is signed by a key the \
                            hardware did not attest; the session is being stopped",
                "type": "x0ne_engine",
                "code": "receipt_verification_failed",
            }}),
        );
        let (addr, seen) = scripted_api(vec![busy, forged]).await;
        let api = Api::new(addr, key()).unwrap();
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let input: &[u8] = b"one\ntwo\nthree\n";
        let e = converse(
            &api, 64, input, &mut out, &mut err, false, false, false, false,
        )
        .await
        .unwrap_err();
        assert!(e.message.contains("the session is over"), "{e}");
        assert_eq!(e.code, EXIT_SESSION_STOPPED);
        assert!(out.is_empty());
        let err = String::from_utf8(err).unwrap();
        assert!(err.contains("  [no answer: HTTP 503: busy]\n"), "{err:?}");
        assert!(
            err.contains(
                "  [no answer: HTTP 502: receipt verification failed: the receipt is signed by \
                 a key the hardware did not attest; the session is being stopped]\n"
            ),
            "{err:?}"
        );
        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 2);
        for body in &seen {
            assert!(body.get("stream").is_none(), "{body}");
            assert!(body.get("stream_options").is_none(), "{body}");
        }
    }

    /// The codes after which the relay stops the session end the chat with
    /// [`EXIT_SESSION_STOPPED`]; a cut, or a code it does not know, with 1.
    #[test]
    fn the_codes_that_stop_the_session_end_the_chat_with_their_own_exit() {
        let event = |code: &str| stream_error(&json!({"code": code, "message": "m"})).ends;
        for &code in ENDS_THE_SESSION.iter().filter(|c| **c != "stream_cut") {
            assert_eq!(event(code), Some(EXIT_SESSION_STOPPED), "{code}");
            let body = json!({"error": {"message": "m", "code": code}}).to_string();
            assert_eq!(
                refused(502, &body).ends,
                Some(EXIT_SESSION_STOPPED),
                "{code}"
            );
        }
        assert_eq!(event("stream_cut"), Some(1));
        assert_eq!(event("not a code"), Some(1));
        assert_eq!(event("generation_failed"), None);
        let cut = json!({"error": {"message": "m", "code": "stream_cut"}}).to_string();
        assert_eq!(refused(502, &cut).ends, Some(1));
        assert_eq!(refused(503, r#"{"error":{"message":"busy"}}"#).ends, None);
    }

    /// `x0ne-engine cloud run` reads the relay's refusal on this exit code:
    /// its crate does not depend on this one and names the number itself.
    #[test]
    fn the_cloud_crate_names_the_same_session_stopped_exit() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../x0ne-engine-cloud/src/chat.rs"
        );
        let cloud = std::fs::read_to_string(path).unwrap();
        let line = format!("pub const CHAT_SESSION_STOPPED: i32 = {EXIT_SESSION_STOPPED};");
        assert!(cloud.contains(&line), "{path}: {line}");
    }
}
