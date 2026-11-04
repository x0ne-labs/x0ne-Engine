//! The HTTP messages inside the encapsulation, in Binary HTTP (RFC 9292),
//! known-length form, zero-padded to a size bucket.
//!
//! Inside the encapsulation, only the gateway reads the fields. A field whose value
//! the client chooses freely could name the client to it, so each allowed field has
//! one exact value, or one well-formed anonymous token; the encoder adds the date,
//! to the minute. The body is the caller's and stays free.

use std::io::Cursor;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bhttp::{Message, Mode, StatusCode};

/// Smallest padded size: requests and answers below it look the same on the wire.
pub const MIN_PADDED: usize = 1024;

/// Header fields a caller may put inside the encapsulation, in the order they are
/// encoded, each at most once.
pub const INNER_HEADERS: [&str; 3] = ["content-type", "accept", "authorization"];
/// The one value of `content-type` and `accept`: the target's media type.
pub const JSON: &str = "application/json";
/// Added by the encoder after the caller's fields: the minute the request was made
/// (RFC 9458 section 6.5.1). It bounds the gateway's replay cache in time.
pub const DATE_HEADER: &str = "date";
/// RFC 9577 section 2.2.2, with the one parameter quoted: `PrivateToken token="…"`.
const TOKEN_PREFIX: &str = "PrivateToken token=\"";
/// RFC 9578 token type 0x0002 (Blind RSA 2048): type, nonce, challenge digest,
/// key id (32) and authenticator (256), 354 bytes. 354 is a multiple of 3, so its
/// base64url has no padding and each token one spelling.
pub const TOKEN_TYPE: u16 = 0x0002;
pub const TOKEN_LEN: usize = 354;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InnerRequest {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InnerResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl InnerResponse {
    pub fn status_only(status: u16) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MessageError {
    #[error("malformed binary HTTP message")]
    Malformed,
    #[error("non-zero padding")]
    Padding,
    /// Not allowed, not its exact value, repeated, or out of order.
    #[error("header field not allowed inside the encapsulation: {0}")]
    HeaderRefused(String),
    #[error("message too large")]
    TooLarge,
    #[error("missing or malformed date")]
    Date,
}

/// The size a message of `len` bytes is padded to: the next power of two, at
/// least [`MIN_PADDED`]. Lengths then leak only their bucket.
pub fn padded_len(len: usize) -> usize {
    len.max(MIN_PADDED).next_power_of_two()
}

/// One caller field: an allowed name (any case) with its exact value.
pub fn check_header(name: &str, value: &str) -> Result<(), MessageError> {
    let lower = name.to_ascii_lowercase();
    let allowed = match lower.as_str() {
        "content-type" | "accept" => value == JSON,
        "authorization" => is_private_token(value),
        _ => false,
    };
    if allowed {
        Ok(())
    } else {
        Err(MessageError::HeaderRefused(lower))
    }
}

/// `PrivateToken token="<base64url>"` and nothing else: one parameter, quoted, a
/// token of type [`TOKEN_TYPE`] and [`TOKEN_LEN`] bytes. Its signature is not
/// verified here: until then, its nonce is 32 bytes the client chooses.
pub fn is_private_token(value: &str) -> bool {
    let Some(b64) = value
        .strip_prefix(TOKEN_PREFIX)
        .and_then(|v| v.strip_suffix('"'))
    else {
        return false;
    };
    base64url_decode(b64)
        .is_some_and(|t| t.len() == TOKEN_LEN && t[..2] == TOKEN_TYPE.to_be_bytes())
}

/// Unpadded base64url (RFC 4648 section 5) of a whole number of 3-byte groups.
fn base64url_decode(s: &str) -> Option<Vec<u8>> {
    let (quads, rest) = s.as_bytes().as_chunks::<4>();
    if !rest.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(quads.len() * 3);
    for quad in quads {
        let mut n = 0u32;
        for &c in quad {
            n = (n << 6) | u32::from(sextet(c)?);
        }
        out.extend_from_slice(&n.to_be_bytes()[1..]);
    }
    Some(out)
}

fn sextet(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'-' => Some(62),
        b'_' => Some(63),
        _ => None,
    }
}

/// The `date` value for a request made at `now`: the minute, in IMF-fixdate
/// (RFC 9110 section 5.6.7). Rounding hides the client's clock to the second
/// (RFC 9458 section 6.8); every client in the same minute sends the same value.
pub fn date_field(now: SystemTime) -> String {
    let secs = now
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    httpdate::fmt_http_date(UNIX_EPOCH + Duration::from_secs(secs - secs % 60))
}

/// A `date` value as [`date_field`] writes it, and nothing else: IMF-fixdate on a
/// whole minute.
fn parse_date(value: &str) -> Option<SystemTime> {
    let t = httpdate::parse_http_date(value).ok()?;
    let secs = t.duration_since(UNIX_EPOCH).ok()?.as_secs();
    (secs % 60 == 0 && httpdate::fmt_http_date(t) == value).then_some(t)
}

/// The caller's fields as the encoder writes them: lowercase names in
/// [`INNER_HEADERS`] order, each once, each value exact.
fn ordered(headers: &[(String, String)]) -> Result<Vec<(usize, String, &str)>, MessageError> {
    let mut out = Vec::with_capacity(headers.len());
    for (k, v) in headers {
        check_header(k, v)?;
        let name = k.to_ascii_lowercase();
        let rank = INNER_HEADERS
            .iter()
            .position(|h| *h == name)
            .ok_or_else(|| MessageError::HeaderRefused(name.clone()))?;
        if out.iter().any(|(r, _, _)| *r == rank) {
            return Err(MessageError::HeaderRefused(name));
        }
        out.push((rank, name, v.as_str()));
    }
    out.sort_by_key(|(r, _, _)| *r);
    Ok(out)
}

/// What the gateway accepts as a request's header section: exactly the encoder's
/// form, the caller's fields then one `date`. Returns that date.
pub fn check_fields(fields: &[(String, String)]) -> Result<SystemTime, MessageError> {
    let Some(((name, value), callers)) = fields.split_last() else {
        return Err(MessageError::Date);
    };
    if name != DATE_HEADER {
        return Err(MessageError::Date);
    }
    let mut next = 0;
    for (k, v) in callers {
        let rank = INNER_HEADERS
            .iter()
            .position(|h| h == k)
            .filter(|r| *r >= next)
            .ok_or_else(|| MessageError::HeaderRefused(k.clone()))?;
        check_header(k, v)?;
        next = rank + 1;
    }
    parse_date(value).ok_or(MessageError::Date)
}

/// Encodes a request made at `now`: the caller's fields in [`INNER_HEADERS`]
/// order, then the date (see [`date_field`]).
pub fn encode_request(
    req: &InnerRequest,
    authority: &str,
    now: SystemTime,
) -> Result<Vec<u8>, MessageError> {
    let fields = ordered(&req.headers)?;
    let mut m = Message::request(
        req.method.as_bytes().to_vec(),
        b"https".to_vec(),
        authority.as_bytes().to_vec(),
        req.path.as_bytes().to_vec(),
    );
    for (_, k, v) in fields {
        m.put_header(k.into_bytes(), v.as_bytes().to_vec());
    }
    m.put_header(DATE_HEADER, date_field(now));
    if !req.body.is_empty() {
        m.write_content(&req.body);
    }
    write_padded(&m)
}

/// What the gateway reads: the request plus the scheme and authority it named.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedRequest {
    pub scheme: String,
    pub authority: String,
    pub request: InnerRequest,
}

pub fn decode_request(bytes: &[u8]) -> Result<DecodedRequest, MessageError> {
    let m = read_padded(bytes, FRAMING_REQUEST)?;
    let c = m.control();
    let text = |v: Option<&[u8]>| -> Result<String, MessageError> {
        String::from_utf8(v.ok_or(MessageError::Malformed)?.to_vec())
            .map_err(|_| MessageError::Malformed)
    };
    Ok(DecodedRequest {
        scheme: text(c.scheme())?,
        authority: text(c.authority())?,
        request: InnerRequest {
            method: text(c.method())?,
            path: text(c.path())?,
            headers: fields(&m)?,
            body: m.content().to_vec(),
        },
    })
}

pub fn encode_response(resp: &InnerResponse) -> Result<Vec<u8>, MessageError> {
    let status = StatusCode::try_from(resp.status).map_err(|_| MessageError::Malformed)?;
    let mut m = Message::response(status);
    for (k, v) in &resp.headers {
        m.put_header(k.to_ascii_lowercase().into_bytes(), v.as_bytes().to_vec());
    }
    if !resp.body.is_empty() {
        m.write_content(&resp.body);
    }
    write_padded(&m)
}

pub fn decode_response(bytes: &[u8]) -> Result<InnerResponse, MessageError> {
    let m = read_padded(bytes, FRAMING_RESPONSE)?;
    let status = m.control().status().ok_or(MessageError::Malformed)?;
    Ok(InnerResponse {
        status: status.code(),
        headers: fields(&m)?,
        body: m.content().to_vec(),
    })
}

fn fields(m: &Message) -> Result<Vec<(String, String)>, MessageError> {
    m.header()
        .fields()
        .iter()
        .map(|f| {
            let k = String::from_utf8(f.name().to_vec()).map_err(|_| MessageError::Malformed)?;
            let v = String::from_utf8(f.value().to_vec()).map_err(|_| MessageError::Malformed)?;
            Ok((k, v))
        })
        .collect()
}

fn write_padded(m: &Message) -> Result<Vec<u8>, MessageError> {
    let mut out = Vec::new();
    m.write_bhttp(Mode::KnownLength, &mut out)
        .map_err(|_| MessageError::Malformed)?;
    out.resize(padded_len(out.len()), 0);
    Ok(out)
}

/// Known-length framing indicators (RFC 9292 section 3.3).
const FRAMING_REQUEST: u8 = 0;
const FRAMING_RESPONSE: u8 = 1;

/// Reads one known-length message of the expected kind; everything after it must
/// be zero padding (RFC 9292 section 3.8). Informational responses and trailers
/// are refused.
fn read_padded(bytes: &[u8], framing: u8) -> Result<Message, MessageError> {
    if bytes.first() != Some(&framing) {
        return Err(MessageError::Malformed);
    }
    let mut r = Cursor::new(bytes);
    let m = Message::read_bhttp(&mut r).map_err(|_| MessageError::Malformed)?;
    if !m.informational().is_empty() || !m.trailer().is_empty() {
        return Err(MessageError::Malformed);
    }
    let used = usize::try_from(r.position()).map_err(|_| MessageError::Malformed)?;
    if bytes[used.min(bytes.len())..].iter().any(|&b| b != 0) {
        return Err(MessageError::Padding);
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use super::*;

    /// 2026-10-24 00:00:37 UTC.
    fn now() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_792_800_037)
    }

    /// A syntactically valid RFC 9578 token of type 0x0002: "AAIA" is 00 02 00.
    fn token() -> String {
        format!("PrivateToken token=\"AAIA{}\"", "A".repeat(468))
    }

    fn req(body: &[u8]) -> InnerRequest {
        InnerRequest {
            method: "POST".into(),
            path: "/v1/chat/completions".into(),
            headers: vec![("Content-Type".into(), "application/json".into())],
            body: body.to_vec(),
        }
    }

    fn fields(v: &[(&str, &str)]) -> Vec<(String, String)> {
        v.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn requests_round_trip_with_lowercased_names() {
        let enc = encode_request(&req(b"{}"), "engine.x0ne.internal", now()).unwrap();
        assert_eq!(enc.len(), MIN_PADDED);
        let d = decode_request(&enc).unwrap();
        assert_eq!(d.scheme, "https");
        assert_eq!(d.authority, "engine.x0ne.internal");
        assert_eq!(d.request.headers[0].0, "content-type");
        assert_eq!(d.request.body, b"{}");
    }

    #[test]
    fn padding_buckets_are_powers_of_two() {
        assert_eq!(padded_len(0), 1024);
        assert_eq!(padded_len(1024), 1024);
        assert_eq!(padded_len(1025), 2048);
        assert_eq!(padded_len(5000), 8192);
    }

    #[test]
    fn non_zero_padding_is_refused() {
        let mut enc = encode_request(&req(b"{}"), "a", now()).unwrap();
        let last = enc.len() - 1;
        enc[last] = 1;
        assert_eq!(decode_request(&enc).unwrap_err(), MessageError::Padding);
    }

    #[test]
    fn a_response_read_as_a_request_is_refused() {
        let enc = encode_response(&InnerResponse::status_only(200)).unwrap();
        assert!(decode_request(&enc).is_err());
    }

    #[test]
    fn indeterminate_length_framing_is_refused() {
        let mut enc = encode_request(&req(b"{}"), "a", now()).unwrap();
        enc[0] = 2;
        assert_eq!(decode_request(&enc).unwrap_err(), MessageError::Malformed);
    }

    #[test]
    fn only_allowlisted_headers_and_anonymous_tokens_pass() {
        for (k, v) in [
            ("cookie", "a=b"),
            ("Authorization", "Bearer k"),
            ("user-agent", "x"),
            ("x-forwarded-for", "1.2.3.4"),
        ] {
            let mut r = req(b"");
            r.headers.push((k.into(), v.into()));
            assert_eq!(
                encode_request(&r, "a", now()).unwrap_err(),
                MessageError::HeaderRefused(k.to_ascii_lowercase())
            );
        }
        let mut r = req(b"");
        r.headers.push(("authorization".into(), token()));
        encode_request(&r, "a", now()).unwrap();
    }

    /// The value of an allowed field is the other place a name could ride.
    #[test]
    fn only_exact_values_and_well_formed_tokens_pass() {
        let token = token();
        for (k, v) in [
            ("content-type", JSON),
            ("Content-Type", JSON),
            ("accept", JSON),
            ("authorization", token.as_str()),
        ] {
            check_header(k, v).unwrap_or_else(|e| panic!("{k}: {v}: {e}"));
        }
        let body = "A".repeat(468);
        let type_1 = format!("PrivateToken token=\"AAEA{body}\"");
        let short = format!("PrivateToken token=\"AAIA{}\"", &body[4..]);
        let long = format!("PrivateToken token=\"AAIA{body}AAAA\"");
        let unquoted = format!("PrivateToken token=AAIA{body}");
        let padded = format!("PrivateToken token=\"AAIA{}==\"", &body[2..]);
        let not_url_safe = format!("PrivateToken token=\"AAIA{}+\"", &body[1..]);
        let extra = format!("{token}, user=\"alice\"");
        let lower_scheme = format!("privatetoken token=\"AAIA{body}\"");
        for (k, v) in [
            ("content-type", "application/json; boundary=alice-7731"),
            ("content-type", "Application/JSON"),
            ("content-type", "application/json "),
            ("accept", "application/json; user=alice"),
            ("accept", "*/*"),
            ("authorization", "PrivateToken alice-7731@example.com"),
            ("authorization", "PrivateToken token=\"AAAA\""),
            ("authorization", type_1.as_str()),
            ("authorization", short.as_str()),
            ("authorization", long.as_str()),
            ("authorization", unquoted.as_str()),
            ("authorization", padded.as_str()),
            ("authorization", not_url_safe.as_str()),
            ("authorization", extra.as_str()),
            ("authorization", lower_scheme.as_str()),
        ] {
            assert_eq!(
                check_header(k, v),
                Err(MessageError::HeaderRefused(k.to_string())),
                "{k}: {v}"
            );
        }
    }

    #[test]
    fn the_encoder_orders_the_fields_refuses_a_repeat_and_adds_the_minute_last() {
        let mut r = req(b"");
        r.headers = fields(&[("accept", JSON), ("content-type", JSON)]);
        let d = decode_request(&encode_request(&r, "a", now()).unwrap()).unwrap();
        assert_eq!(
            d.request.headers,
            fields(&[
                ("content-type", JSON),
                ("accept", JSON),
                ("date", "Sat, 24 Oct 2026 00:00:00 GMT")
            ])
        );
        // Sat, 24 Oct 2026 00:00:00 GMT, in Unix seconds.
        let date: u64 = 1_792_800_000;
        assert_eq!(
            check_fields(&d.request.headers),
            Ok(UNIX_EPOCH + Duration::from_secs(date))
        );

        r.headers.push(("Accept".into(), JSON.into()));
        assert_eq!(
            encode_request(&r, "a", now()).unwrap_err(),
            MessageError::HeaderRefused("accept".into())
        );
        let mut r = req(b"");
        r.headers.push(("date".into(), date_field(now())));
        assert_eq!(
            encode_request(&r, "a", now()).unwrap_err(),
            MessageError::HeaderRefused("date".into())
        );
    }

    /// What the gateway accepts is the encoder's form and nothing near it.
    #[test]
    fn the_gateway_check_wants_the_encoders_exact_form() {
        let date = date_field(now());
        let date = date.as_str();
        let token = token();
        let token = token.as_str();
        for good in [
            vec![("date", date)],
            vec![("content-type", JSON), ("date", date)],
            vec![
                ("content-type", JSON),
                ("accept", JSON),
                ("authorization", token),
                ("date", date),
            ],
        ] {
            check_fields(&fields(&good)).unwrap_or_else(|e| panic!("{good:?}: {e}"));
        }
        for bad in [
            vec![],
            vec![("content-type", JSON)],
            vec![("Content-Type", JSON), ("date", date)],
            vec![("accept", JSON), ("content-type", JSON), ("date", date)],
            vec![
                ("content-type", JSON),
                ("content-type", JSON),
                ("date", date),
            ],
            vec![("date", date), ("content-type", JSON)],
            vec![("date", date), ("date", date)],
            vec![("x-user", "alice"), ("date", date)],
            vec![("date", "Sat, 24 Oct 2026 00:00:37 GMT")],
            vec![("date", "Saturday, 24-Oct-26 00:00:00 GMT")],
            vec![("date", "Sat Oct 24 00:00:00 2026")],
            vec![("date", "sat, 24 oct 2026 00:00:00 gmt")],
        ] {
            assert!(check_fields(&fields(&bad)).is_err(), "{bad:?}");
        }
    }
}
