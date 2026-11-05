//! The caller-to-TD path: what binds the TD's OHTTP key to its quote, what
//! a question and an answer look like inside the encryption, and the two ends.
//!
//! - [`key_binding`]: the 64 bytes the TD writes into `report_data`, over
//!   its OHTTP key, the caller's nonce, the receipt key and the TD's [`Boot`]
//!   (its boot nonce, its TLS key, its session config's digest);
//!   [`key_binding_gpu`] for a TD that attests GPUs, over their evidence
//!   too, which answers [`gpu_nonce`].
//! - [`Client`]: built only from a [`VerifiedKey`]; seals a chat request,
//!   opens its response and checks the receipt under the attested key.
//! - [`Gateway`]: the TD's end; holds the private key, opens a request,
//!   refuses anything but `POST /v1/chat/completions` with a JSON object body
//!   and one valid `Date`, and seals the answer. [`ReplayGuard`] answers each
//!   request once, inside a time window.
//!
//! A question that asks to stream is sealed as a chunked OHTTP request
//! ([`Client::seal_chat_stream`], [`Gateway::open_stream`]) and answered by
//! a chunked response at the session's cadence ([`crate::sealed_stream`]);
//! one that does not, as one Encapsulated Request, answered whole. Neither
//! path opens the other's requests.
//!
//! Both directions are padded (RFC 9292 §3.8) to [`padded_len`]: a carrier
//! learns the bucket, not the length.

use crate::attest::VerifiedKey;
use crate::bhttp::{self, BhttpError};
use crate::hpke::{self, PrivateKey};
use crate::ohttp::{self, KeyConfig, OhttpError, RESPONSE_NONCE_LEN};
use crate::receipt::{self, Checked};
use sha2::{Digest, Sha256, Sha512};
use std::collections::HashMap;
use zeroize::Zeroizing;

/// Domain of [`key_binding`]: no other x0ne `report_data` starts with it.
/// v2 since 2026-10-30, when the binding took in the TD's [`Boot`]; a quote
/// over v1 binds no session config and is refused.
pub const BINDING_LABEL: &[u8] = b"x0ne-e2e/v2/ohttp-key";
/// Domain of [`key_binding_gpu`]: neither it nor [`BINDING_LABEL`] starts
/// with the other, nor with any other x0ne `report_data` context.
pub const GPU_BINDING_LABEL: &[u8] = b"x0ne-e2e/v2/ohttp-gpu-key";
/// Domain of [`gpu_nonce`], the nonce the TD's GPUs sign their reports over.
pub const GPU_NONCE_LABEL: &[u8] = b"x0ne-e2e/v1/gpu";
/// GPUs a key document may carry evidence for: an a3-highgpu-8g has 8.
pub const MAX_GPUS: usize = 8;
/// The one route the gateway forwards.
pub const CHAT_PATH: &str = "/v1/chat/completions";
/// Smallest padded message; then powers of two.
pub const MIN_PADDED: usize = 4096;
/// Largest message either way, padded: 4 MiB.
pub const MAX_PADDED: usize = 4 * 1024 * 1024;
/// The largest sealed request on the wire: [`MAX_PADDED`] sealed. The
/// gateway's route reads no more; [`Client::seal_chat`] makes no larger.
pub const MAX_SEALED_REQUEST: usize = MAX_PADDED + ohttp::REQUEST_OVERHEAD;
/// The largest chunked request: the same bucket cut in chunks of 16384
/// bytes (draft-08 §3), each non-final one 20 bytes more (a 4-byte length
/// and its tag), the final one 1 byte more (its zero length).
pub const MAX_SEALED_STREAM_REQUEST: usize =
    MAX_SEALED_REQUEST + 1 + (MAX_PADDED / ohttp::MAX_CHUNK_DATA - 1) * (4 + ohttp::CHUNK_TAG);
/// The gateway's key id in its configuration.
pub const KEY_ID: u8 = 1;
/// The header that carries, inside the encryption, when the caller sealed
/// the question: HTTP `Date` (RFC 9458 §6.5.1), IMF-fixdate (RFC 9110 §5.6.7).
pub const SENT_AT_HEADER: &str = "date";
/// How far from the gateway's clock a `Date` may be, either way, by default.
pub const DEFAULT_WINDOW_SECS: u64 = 120;
/// Requests a [`ReplayGuard`] remembers at most, by default.
pub const DEFAULT_REPLAY_CAPACITY: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum E2eError {
    #[error("{0}")]
    Ohttp(#[from] OhttpError),
    #[error("binary HTTP: {0}")]
    Bhttp(#[from] BhttpError),
    #[error("message of {0} bytes is larger than the largest bucket ({MAX_PADDED})")]
    TooLarge(usize),
    #[error("message of {len} bytes is not padded to its bucket ({bucket})")]
    NotPadded { len: usize, bucket: usize },
    #[error("refused: {0}")]
    Refused(String),
    #[error("replay: {0}")]
    Replay(String),
    #[error("receipt: {0}")]
    Receipt(String),
    #[error("stream: {0}")]
    Stream(String),
}

/// What the TD's boot binding names besides the receipt key (x0ne-infer's
/// `binding::BindingV2`, the `report_data` of its boot quote and, per
/// connection, of the relay's session quotes): the nonce the TD drew at
/// boot, SHA-256 of its TLS key, SHA-256 of its session config's exact bytes
/// (`x0ne-engine/cloud-config/v1`: the model it serves, `expires_at_unix`,
/// the relay's TLS key). Every field is 32 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Boot {
    pub nonce: [u8; 32],
    pub tls_spki_sha256: [u8; 32],
    pub config_sha256: [u8; 32],
}

impl Boot {
    /// boot.nonce ‖ boot.tls_spki_sha256 ‖ boot.config_sha256, fixed width.
    fn feed(&self, h: &mut Sha512) {
        h.update(self.nonce);
        h.update(self.tls_spki_sha256);
        h.update(self.config_sha256);
    }
}

/// SHA-512(label ‖ 0x00 ‖ len(key_config) as u16 ‖ key_config ‖
/// client_nonce ‖ session_public_key ‖ boot.nonce ‖ boot.tls_spki_sha256 ‖
/// boot.config_sha256): the TD's `report_data` for the key document that
/// answers `client_nonce`. The receipt key rides along, so a caller can
/// check receipts with the key the hardware vouched for; the boot binding
/// too, so the caller reads, before sending, the session config this TD
/// serves under, as its quote binds it.
pub fn key_binding(
    key_config: &[u8],
    client_nonce: &[u8; 32],
    session_public_key: &[u8; 32],
    boot: &Boot,
) -> [u8; 64] {
    let len = u16::try_from(key_config.len()).expect("a key configuration is short");
    let mut h = Sha512::new();
    h.update(BINDING_LABEL);
    h.update([0u8]);
    h.update(len.to_be_bytes());
    h.update(key_config);
    h.update(client_nonce);
    h.update(session_public_key);
    boot.feed(&mut h);
    h.finalize().into()
}

/// The 32 bytes the TD's GPUs answer for the key document that answers
/// `client_nonce`: SHA-256(label ‖ 0x00 ‖ len(key_config) as u16 ‖
/// key_config ‖ client_nonce ‖ session_public_key). Anyone can compute it:
/// what ties the GPUs' evidence to the TD is [`key_binding_gpu`], in the quote.
pub fn gpu_nonce(
    key_config: &[u8],
    client_nonce: &[u8; 32],
    session_public_key: &[u8; 32],
) -> [u8; 32] {
    let len = u16::try_from(key_config.len()).expect("a key configuration is short");
    let mut h = Sha256::new();
    h.update(GPU_NONCE_LABEL);
    h.update([0u8]);
    h.update(len.to_be_bytes());
    h.update(key_config);
    h.update(client_nonce);
    h.update(session_public_key);
    h.finalize().into()
}

/// [`key_binding`] for a TD that attests GPUs: SHA-512(label ‖ 0x00 ‖
/// len(key_config) as u16 ‖ key_config ‖ client_nonce ‖ session_public_key ‖
/// boot.nonce ‖ boot.tls_spki_sha256 ‖ boot.config_sha256 ‖
/// count as u64 ‖ for each GPU, in the document's order: len(report) as u64
/// ‖ report ‖ len(certificate chain) as u64 ‖ chain). Each `(report, chain)`
/// is the decoded bytes of one `gpu_evidence_list` entry: the quote binds
/// the very evidence the carrier hands NRAS, and the chain whose leaf
/// serial the NRAS token must name.
pub fn key_binding_gpu(
    key_config: &[u8],
    client_nonce: &[u8; 32],
    session_public_key: &[u8; 32],
    boot: &Boot,
    gpus: &[(&[u8], &[u8])],
) -> [u8; 64] {
    let len = u16::try_from(key_config.len()).expect("a key configuration is short");
    let mut h = Sha512::new();
    h.update(GPU_BINDING_LABEL);
    h.update([0u8]);
    h.update(len.to_be_bytes());
    h.update(key_config);
    h.update(client_nonce);
    h.update(session_public_key);
    boot.feed(&mut h);
    h.update((gpus.len() as u64).to_be_bytes());
    for (report, chain) in gpus {
        h.update((report.len() as u64).to_be_bytes());
        h.update(report);
        h.update((chain.len() as u64).to_be_bytes());
        h.update(chain);
    }
    h.finalize().into()
}

/// The padded length of an `n`-byte message: 4 KiB, then the next power of two.
pub fn padded_len(n: usize) -> Result<usize, E2eError> {
    let bucket = n.max(MIN_PADDED).next_power_of_two();
    if bucket > MAX_PADDED {
        return Err(E2eError::TooLarge(n));
    }
    Ok(bucket)
}

/// A message is exactly its bucket: a shorter or longer one was not padded
/// by this code, and its length says more than the bucket would.
fn check_padded(len: usize) -> Result<(), E2eError> {
    if len != MIN_PADDED && !(len > MIN_PADDED && len.is_power_of_two() && len <= MAX_PADDED) {
        return Err(E2eError::NotPadded {
            len,
            bucket: padded_len(len).unwrap_or(MAX_PADDED),
        });
    }
    Ok(())
}

// ── HTTP dates (RFC 9110 §5.6.7, IMF-fixdate only) ──────────────────────────

const DAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// Days since 1970-01-01 of a proleptic Gregorian date (H. Hinnant's
/// `days_from_civil`).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = i64::from((m + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The inverse of [`days_from_civil`].
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        2 if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// `unix` (seconds) as an IMF-fixdate: `Fri, 15 Jan 2027 08:00:00 GMT`.
pub fn http_date(unix: u64) -> String {
    let days = (unix / 86_400) as i64;
    let secs = unix % 86_400;
    let (y, m, d) = civil_from_days(days);
    format!(
        "{}, {d:02} {} {y:04} {:02}:{:02}:{:02} GMT",
        DAYS[((days + 4) % 7) as usize],
        MONTHS[(m - 1) as usize],
        secs / 3600,
        (secs / 60) % 60,
        secs % 60
    )
}

/// An IMF-fixdate, exactly: 29 bytes, the weekday that date falls on, a real
/// day of a real month, 1970 or later, `GMT`. The obsolete RFC 850 and
/// asctime forms are refused: this SDK writes neither.
pub fn parse_http_date(s: &str) -> Option<u64> {
    let b = s.as_bytes();
    if b.len() != 29 || &b[3..5] != b", " || &b[25..] != b" GMT" {
        return None;
    }
    let num = |r: std::ops::Range<usize>| -> Option<u32> {
        let part = &b[r];
        if part.iter().all(u8::is_ascii_digit) {
            std::str::from_utf8(part).ok()?.parse().ok()
        } else {
            None
        }
    };
    let sep = |i: usize, c: u8| b[i] == c;
    if !(sep(7, b' ') && sep(11, b' ') && sep(16, b' ') && sep(19, b':') && sep(22, b':')) {
        return None;
    }
    let weekday = DAYS.iter().position(|d| d.as_bytes() == &b[..3])?;
    let day = num(5..7)?;
    let month = MONTHS.iter().position(|m| m.as_bytes() == &b[8..11])? as u32 + 1;
    let year = i64::from(num(12..16)?);
    let (hour, min, sec) = (num(17..19)?, num(20..22)?, num(23..25)?);
    if year < 1970
        || day == 0
        || day > days_in_month(year, month)
        || hour > 23
        || min > 59
        || sec > 59
    {
        return None;
    }
    let days = days_from_civil(year, month, day);
    if ((days + 4) % 7) as usize != weekday {
        return None;
    }
    Some(days as u64 * 86_400 + u64::from(hour) * 3600 + u64::from(min) * 60 + u64::from(sec))
}

/// Both ends' rule: one JSON object whose `stream` is `streamed`: `true` on
/// the chunked path, absent or `false` on the whole one. Every string in it
/// is text: a lone surrogate (`\ud800`), which the grammar allows and the
/// TD's chat cannot read, is refused here, by the caller before sending and
/// by the gateway at opening. A refusal leaves the TD in clear: it names no
/// byte of the body.
fn check_chat_body(body: &[u8], streamed: bool) -> Result<(), E2eError> {
    #[derive(serde::Deserialize)]
    struct ChatBody {
        #[serde(default)]
        stream: Option<bool>,
    }
    let object = body.iter().find(|c| !c.is_ascii_whitespace()) == Some(&b'{');
    // `Value` decodes every string, as the TD's chat handler does; the typed
    // parse refuses a `stream` given twice.
    let text = serde_json::from_slice::<serde_json::Value>(body).is_ok();
    let parsed = serde_json::from_slice::<ChatBody>(body)
        .ok()
        .filter(|_| object && text);
    match parsed {
        None => Err(E2eError::Refused(
            "the body is not one JSON object with a boolean stream".into(),
        )),
        Some(ChatBody { stream: Some(true) }) if !streamed => Err(E2eError::Refused(
            "stream: true is sealed as a chunked request (seal_chat_stream)".into(),
        )),
        Some(ChatBody {
            stream: None | Some(false),
        }) if streamed => Err(E2eError::Refused(
            "a chunked request asks to stream: stream must be true".into(),
        )),
        Some(_) => Ok(()),
    }
}

/// What `body` asks of the decoding, read as the TD reads it; else why not
/// (the TD refuses such a body, so no 200 is checked against it).
pub(crate) fn asked_of(body: &[u8]) -> Result<crate::decoding::Asked, String> {
    serde_json::from_slice::<serde_json::Value>(body)
        .map_err(|e| e.to_string())
        .and_then(|v| crate::decoding::Asked::from_body(&v))
}

fn json_headers() -> Vec<(String, String)> {
    vec![("content-type".into(), "application/json".into())]
}

/// The Binary HTTP chat request for `body` (JSON), sealed at `sent_at`
/// (Unix seconds, the caller's clock), padded.
pub fn chat_request(body: &[u8], sent_at: u64) -> Result<Vec<u8>, E2eError> {
    let mut headers = json_headers();
    headers.push((SENT_AT_HEADER.into(), http_date(sent_at)));
    let req = bhttp::Request {
        method: "POST".into(),
        scheme: "https".into(),
        authority: String::new(),
        path: CHAT_PATH.into(),
        headers,
        content: body.to_vec(),
    };
    let unpadded = req.encode(0).len();
    Ok(req.encode(padded_len(unpadded)?))
}

/// The Binary HTTP response for (`status`, JSON `body`), padded.
pub fn chat_response(status: u16, body: &[u8]) -> Result<Vec<u8>, E2eError> {
    let resp = bhttp::Response {
        status,
        headers: json_headers(),
        content: body.to_vec(),
    };
    let unpadded = resp.encode(0).len();
    Ok(resp.encode(padded_len(unpadded)?))
}

// ── The caller's end ─────────────────────────────────────────────────────────

/// Seals to a key [`crate::attest::verify_key`] accepted, and to no other.
pub struct Client {
    key: VerifiedKey,
}

/// One request in flight: opens its own response and no other, and checks
/// its receipt under the key the quote bound.
pub struct Pending {
    response: ohttp::ClientResponse,
    session_pubkey: [u8; 32],
    /// The key's TCB verdict when its policy is `strict`: every receipt is
    /// held to it, as a strict relay holds them.
    strict_tcb: Option<crate::tcb_disclosure::TcbDisclosure>,
    /// The decoding the body asked for, which the receipt must sign; the
    /// reason, for a body the TD will refuse (no 200 is checked against it).
    /// The body itself is not kept: it holds the prompt.
    asked: Result<crate::decoding::Asked, String>,
}

/// What came back, decrypted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub status: u16,
    pub body: Vec<u8>,
}

impl Client {
    pub fn new(key: VerifiedKey) -> Self {
        Self { key }
    }

    pub fn key(&self) -> &VerifiedKey {
        &self.key
    }

    /// `body`: the JSON of a chat completion request. `ikm`: 32 fresh random
    /// bytes (the ephemeral key); never reused. `now`: the caller's clock,
    /// sealed inside as the request's `Date`. Refused within
    /// [`crate::attest::SESSION_EXPIRY_GUARD_SECS`] of the session's end, as
    /// the relay refuses new requests then.
    pub fn seal_chat(
        &self,
        body: &[u8],
        ikm: &[u8; 32],
        now: u64,
    ) -> Result<(Vec<u8>, Pending), E2eError> {
        self.check_session(now)?;
        check_chat_body(body, false)?;
        let asked = asked_of(body);
        let inner = Zeroizing::new(chat_request(body, now)?);
        let (bytes, response) = ohttp::encapsulate_request(self.key.key_config(), &inner, ikm)?;
        Ok((
            bytes,
            Pending {
                response,
                session_pubkey: *self.key.session_pubkey(),
                strict_tcb: self.strict_tcb(),
                asked,
            },
        ))
    }

    /// [`Self::seal_chat`] for a question whose `stream` is `true`: the same
    /// padded request, sealed as a chunked OHTTP request
    /// (`message/ohttp-chunked-req`) in chunks of 16384 bytes, the last one
    /// final (draft-08 §3: what every receiver takes), so its size is the
    /// bucket's; its answer comes as a chunked
    /// response at the cadence of the session config the quote bound
    /// ([`crate::sealed_stream`]).
    pub fn seal_chat_stream(
        &self,
        body: &[u8],
        ikm: &[u8; 32],
        now: u64,
    ) -> Result<(Vec<u8>, crate::sealed_stream::PendingStream), E2eError> {
        self.check_session(now)?;
        let cadence =
            crate::sealed_stream::cadence(&self.key.session_config).map_err(E2eError::Stream)?;
        check_chat_body(body, true)?;
        let asked = asked_of(body);
        let inner = Zeroizing::new(chat_request(body, now)?);
        let chunks: Vec<&[u8]> = inner.chunks(ohttp::MAX_CHUNK_DATA).collect();
        let (bytes, response) =
            ohttp::encapsulate_chunked_request(self.key.key_config(), &chunks, ikm)?;
        let pending = crate::sealed_stream::PendingStream::new(
            response,
            cadence,
            (now, self.key.session_config.expires_at_unix),
            *self.key.session_pubkey(),
            self.strict_tcb(),
            asked,
        )?;
        Ok((bytes, pending))
    }

    /// Refused within [`crate::attest::SESSION_EXPIRY_GUARD_SECS`] of the
    /// session's end, as the relay refuses new requests then.
    fn check_session(&self, now: u64) -> Result<(), E2eError> {
        let ends = self.key.session_config.expires_at_unix;
        if now.saturating_add(crate::attest::SESSION_EXPIRY_GUARD_SECS) >= ends {
            return Err(E2eError::Refused(format!(
                "the TD's session expires at {ends}: no new request within {}s of it, as the relay",
                crate::attest::SESSION_EXPIRY_GUARD_SECS
            )));
        }
        Ok(())
    }

    /// The key's TCB verdict when its policy is `strict`.
    fn strict_tcb(&self) -> Option<crate::tcb_disclosure::TcbDisclosure> {
        (self.key.tcb.policy == crate::tcb::TcbMode::Strict.as_str()).then(|| self.key.tcb.clone())
    }
}

/// Under `strict` (`want`), the receipt's signed TCB disclosure at the key's
/// floor and judged within its age limit at `now`
/// (`tcb_disclosure::strict_receipt_check`, the relay's check).
pub(crate) fn strict_check(
    want: Option<&crate::tcb_disclosure::TcbDisclosure>,
    checked: &Checked,
    now: u64,
) -> Result<(), E2eError> {
    let Some(want) = want else {
        return Ok(());
    };
    let signed = match checked.signed.get("tcb") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => Some(
            serde_json::from_value::<crate::tcb_disclosure::TcbDisclosure>(v.clone())
                .map_err(|e| E2eError::Receipt(format!("tcb: {e}")))?,
        ),
    };
    crate::tcb_disclosure::strict_receipt_check(want, signed.as_ref(), now)
        .map_err(E2eError::Receipt)
}

impl Pending {
    /// The response, decrypted: only the TD that opened this request could
    /// have sealed it. Its receipt is not checked here: see [`Self::open_verified`].
    pub fn open(&self, encapsulated: &[u8]) -> Result<Answer, E2eError> {
        let inner = self.response.decapsulate(encapsulated)?;
        check_padded(inner.len())?;
        let resp = bhttp::Response::decode(&inner)?;
        Ok(Answer {
            status: resp.status,
            body: resp.content,
        })
    }

    /// [`Self::open`], then, for a 200, the receipt: signed by the session
    /// key the quote bound, over the text shown, for the model whose digest
    /// the caller names (the relay checks its catalogue's; required here
    /// since 2026-10-27); under `strict`, its signed TCB disclosure at the
    /// key's floor and judged within its age limit at `now`, the caller's
    /// clock (`tcb_disclosure::strict_receipt_check`, the relay's check). A
    /// 200 without a receipt that checks is refused. Another status is
    /// returned as the TD sent it, with no receipt.
    pub fn open_verified(
        &self,
        encapsulated: &[u8],
        model_digest_hex: &str,
        now: u64,
    ) -> Result<(Answer, Option<Checked>), E2eError> {
        let answer = self.open(encapsulated)?;
        if answer.status != 200 {
            return Ok((answer, None));
        }
        // A body this end could not read as a decoding was still answered:
        // nothing says what its receipt must sign, so it is refused.
        let asked = self
            .asked
            .as_ref()
            .map_err(|e| E2eError::Receipt(format!("decoding: the request: {e}")))?;
        let checked = receipt::check_answer_for(
            &answer.body,
            &self.session_pubkey,
            Some(model_digest_hex),
            asked,
        )
        .map_err(E2eError::Receipt)?;
        strict_check(self.strict_tcb.as_ref(), &checked, now)?;
        Ok((answer, Some(checked)))
    }
}

// ── The TD's end ─────────────────────────────────────────────────────────────

/// The TD's OHTTP gateway: one key per boot, derived from a seed the TD
/// draws (RFC 9180 `DeriveKeyPair`).
pub struct Gateway {
    sk: PrivateKey,
    config: KeyConfig,
}

/// One opened request: writes its one response.
pub struct Responder(ohttp::ServerResponse);

/// An opened chat request: its JSON body and the `Date` it was sealed at.
pub struct Opened {
    pub body: Zeroizing<Vec<u8>>,
    pub sent_at: u64,
}

impl Gateway {
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let (sk, public_key) = hpke::derive_key_pair(seed);
        Self {
            sk,
            config: KeyConfig {
                key_id: KEY_ID,
                public_key,
            },
        }
    }

    /// [`Self::from_seed`], the private scalar moved into the memory `store`
    /// returns (the TD's `memfd_secret` or `mlock`ed memory); the derived
    /// copy is wiped.
    ///
    /// # Errors
    /// `store`'s.
    pub fn from_seed_held_by<S: hpke::SecretScalar + 'static, E>(
        seed: &[u8; 32],
        store: impl FnOnce(&[u8]) -> Result<S, E>,
    ) -> Result<Self, E> {
        let (sk, public_key) = hpke::derive_key_pair(seed);
        Ok(Self {
            sk: sk.held_by(store)?,
            config: KeyConfig {
                key_id: KEY_ID,
                public_key,
            },
        })
    }

    /// The configuration bytes the binding and the key document carry.
    pub fn key_config(&self) -> Vec<u8> {
        self.config.encode()
    }

    /// The chat request, or why not: padded, `POST` [`CHAT_PATH`], a JSON
    /// object that does not ask to stream ([`Client::seal_chat`]'s rule), one
    /// valid `Date`. The caller then runs [`ReplayGuard::admit`] on
    /// [`Responder::replay_id`] and [`Opened::sent_at`] before answering.
    pub fn open(&self, encapsulated: &[u8]) -> Result<(Opened, Responder), E2eError> {
        let (inner, responder) = ohttp::decapsulate_request(&self.sk, KEY_ID, encapsulated)?;
        Ok((chat_opened(&inner, false)?, Responder(responder)))
    }

    /// [`Self::open`] for a chunked request (`message/ohttp-chunked-req`):
    /// its chunks opened in order, whole, to at most [`MAX_PADDED`] bytes,
    /// the same chat request inside, whose `stream` is `true`. Its non-final
    /// chunks carry [`ohttp::MAX_CHUNK_DATA`] bytes, as
    /// [`Client::seal_chat_stream`] seals them: a smaller one is refused
    /// before any chunk is opened, so at most 257 are.
    pub fn open_stream(&self, encapsulated: &[u8]) -> Result<(Opened, StreamResponder), E2eError> {
        let (inner, responder) = ohttp::decapsulate_chunked_request_in(
            &self.sk,
            KEY_ID,
            encapsulated,
            ohttp::MAX_CHUNK_DATA,
            MAX_PADDED,
        )?;
        Ok((chat_opened(&inner, true)?, StreamResponder(responder)))
    }
}

/// The chat request inside an opened request, or why not.
fn chat_opened(inner: &[u8], streamed: bool) -> Result<Opened, E2eError> {
    check_padded(inner.len())?;
    let req = bhttp::Request::decode(inner)?;
    if req.method != "POST" || req.path != CHAT_PATH {
        return Err(E2eError::Refused(format!(
            "{} {}: only POST {CHAT_PATH} is forwarded",
            req.method, req.path
        )));
    }
    if req.header("content-type") != Some("application/json") {
        return Err(E2eError::Refused(
            "content-type is not application/json".into(),
        ));
    }
    check_chat_body(&req.content, streamed)?;
    let mut dates = req
        .headers
        .iter()
        .filter(|(n, _)| n.eq_ignore_ascii_case(SENT_AT_HEADER))
        .map(|(_, v)| v.as_str());
    let sent_at = match (dates.next(), dates.next()) {
        (Some(v), None) => parse_http_date(v)
            .ok_or_else(|| E2eError::Refused(format!("date {v:?} is not an IMF-fixdate")))?,
        (None, _) => return Err(E2eError::Refused("no date: RFC 9458 §6.5.1".into())),
        (Some(_), Some(_)) => return Err(E2eError::Refused("more than one date".into())),
    };
    Ok(Opened {
        body: Zeroizing::new(req.content),
        sent_at,
    })
}

/// One opened chunked request: writes its one chunked response.
pub struct StreamResponder(ohttp::ChunkedServerResponse);

impl StreamResponder {
    /// The request's encapsulated key (RFC 9458 §6.5): unique per request.
    pub fn replay_id(&self) -> [u8; 32] {
        self.0.enc()
    }

    /// The writer of the chunked response under `nonce`, fresh random.
    pub fn writer(
        &self,
        nonce: &[u8; RESPONSE_NONCE_LEN],
    ) -> Result<ohttp::ChunkedResponseWriter, E2eError> {
        Ok(self.0.writer(nonce)?)
    }
}

impl Responder {
    /// The request's encapsulated key (RFC 9458 §6.5): unique per request.
    pub fn replay_id(&self) -> [u8; 32] {
        self.0.enc()
    }

    /// The padded response, sealed. `nonce`: fresh random.
    pub fn seal(
        &self,
        status: u16,
        body: &[u8],
        nonce: &[u8; RESPONSE_NONCE_LEN],
    ) -> Result<Vec<u8>, E2eError> {
        let inner = Zeroizing::new(chat_response(status, body)?);
        Ok(self.0.encapsulate(&inner, nonce)?)
    }
}

/// The gateway's memory of the requests it answered (RFC 9458 §6.5.1): a
/// request is answered once, and only within `window` seconds of the
/// gateway's clock. That clock is the TD's wall clock, which the host sets:
/// the window's old edge follows the highest `now` the guard was given, a
/// high-water mark as in IPsec's anti-replay window (RFC 4303 §3.4.3), so
/// a clock set back reopens nothing. An entry is dropped only once its date
/// is past that edge, where its date alone refuses it for good; a full table
/// refuses new requests rather than forget one still in the window. The
/// memory lives as long as the gateway's key, one boot: a request sealed to
/// an earlier boot's key does not open.
pub struct ReplayGuard {
    window: u64,
    capacity: usize,
    /// The highest `now` [`Self::admit`] was given.
    high: u64,
    seen: HashMap<[u8; 32], u64>,
}

impl Default for ReplayGuard {
    fn default() -> Self {
        Self::new(DEFAULT_WINDOW_SECS, DEFAULT_REPLAY_CAPACITY)
    }
}

impl ReplayGuard {
    pub fn new(window: u64, capacity: usize) -> Self {
        Self {
            window,
            capacity,
            high: 0,
            seen: HashMap::new(),
        }
    }

    /// Requests remembered now.
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    /// Admit the request `id` sealed at `sent_at`, at the gateway's `now`:
    /// no older than the window before the highest `now` seen, no further
    /// ahead of this `now` than the window, never admitted before.
    pub fn admit(&mut self, id: [u8; 32], sent_at: u64, now: u64) -> Result<(), E2eError> {
        self.high = self.high.max(now);
        let high = self.high;
        if sent_at.saturating_add(self.window) < high {
            return Err(E2eError::Replay(format!(
                "sealed {}s before the latest time this gateway saw, outside the {}s window",
                high - sent_at,
                self.window
            )));
        }
        if sent_at > now.saturating_add(self.window) {
            return Err(E2eError::Replay(format!(
                "sealed {}s ahead of this clock, outside the {}s window",
                sent_at - now,
                self.window
            )));
        }
        // SECURITY: dropped only past the high-water edge, which never goes
        // back: a request dropped here is refused by its date from now on.
        let floor = high.saturating_sub(self.window);
        self.seen.retain(|_, t| *t >= floor);
        if self.seen.contains_key(&id) {
            return Err(E2eError::Replay("this request was answered already".into()));
        }
        if self.seen.len() >= self.capacity {
            return Err(E2eError::Replay(format!(
                "{} requests in the window, the most this gateway remembers",
                self.capacity
            )));
        }
        self.seen.insert(id, sent_at);
        Ok(())
    }
}
