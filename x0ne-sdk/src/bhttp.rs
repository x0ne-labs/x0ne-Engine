//! Binary HTTP (RFC 9292): what Oblivious HTTP carries. Known-length
//! messages whole: encoding writes every section and pads with zero bytes
//! (RFC 9292 §3.8); decoding accepts the truncated forms RFC 9292 allows,
//! refuses indeterminate-length framing, informational responses, trailers
//! and non-zero padding, and bounds every length before it allocates.
//! Indeterminate-length responses (§3.2) piece by piece, for chunked OHTTP:
//! [`indeterminate`] writes them, [`ResponseReader`] reads them as they
//! arrive, under the same refusals.

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BhttpError {
    #[error("truncated inside {0}")]
    Truncated(&'static str),
    #[error("framing indicator {0}: only known-length requests (0) and responses (1)")]
    Framing(u64),
    #[error("{what} is {len} bytes, more than {max}")]
    TooLong {
        what: &'static str,
        len: u64,
        max: usize,
    },
    #[error("{0} is not valid UTF-8 text")]
    Text(&'static str),
    #[error("status {0} is not a final status (200..=599)")]
    Status(u64),
    #[error("trailer fields are not accepted")]
    Trailers,
    #[error("padding holds a non-zero byte at offset {0}")]
    Padding(usize),
    #[error("more than {0} header fields")]
    Fields(usize),
}

/// Bounds of one decoded message. The content bound is the caller's.
pub const MAX_FIELD_BYTES: usize = 8 * 1024;
pub const MAX_FIELDS: usize = 64;
pub const MAX_CONTENT_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Request {
    pub method: String,
    pub scheme: String,
    pub authority: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub content: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub content: Vec<u8>,
}

// ── Variable-length integers (RFC 9000 §16) ─────────────────────────────────

pub fn put_varint(out: &mut Vec<u8>, v: u64) {
    match v {
        0..=0x3f => out.push(v as u8),
        0x40..=0x3fff => out.extend_from_slice(&((v as u16) | 0x4000).to_be_bytes()),
        0x4000..=0x3fff_ffff => out.extend_from_slice(&((v as u32) | 0x8000_0000).to_be_bytes()),
        _ => {
            assert!(v < 1 << 62, "varint out of range");
            out.extend_from_slice(&(v | 0xc000_0000_0000_0000).to_be_bytes())
        }
    }
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn at_end(&self) -> bool {
        self.at == self.b.len()
    }

    fn varint(&mut self, what: &'static str) -> Result<u64, BhttpError> {
        let first = *self.b.get(self.at).ok_or(BhttpError::Truncated(what))?;
        let len = 1usize << (first >> 6);
        let bytes = self
            .b
            .get(self.at..self.at + len)
            .ok_or(BhttpError::Truncated(what))?;
        let mut v = u64::from(first & 0x3f);
        for b in &bytes[1..] {
            v = (v << 8) | u64::from(*b);
        }
        self.at += len;
        Ok(v)
    }

    fn bytes(&mut self, what: &'static str, max: usize) -> Result<&'a [u8], BhttpError> {
        let len = self.varint(what)?;
        if len > max as u64 {
            return Err(BhttpError::TooLong { what, len, max });
        }
        let len = len as usize;
        let out = self
            .b
            .get(self.at..self.at + len)
            .ok_or(BhttpError::Truncated(what))?;
        self.at += len;
        Ok(out)
    }

    fn text(&mut self, what: &'static str) -> Result<String, BhttpError> {
        let b = self.bytes(what, MAX_FIELD_BYTES)?;
        String::from_utf8(b.to_vec()).map_err(|_| BhttpError::Text(what))
    }

    /// A field section; an absent one (end of message) is empty.
    fn fields(&mut self, what: &'static str) -> Result<Vec<(String, String)>, BhttpError> {
        if self.at_end() {
            return Ok(Vec::new());
        }
        let section = self.bytes(what, MAX_FIELDS * 2 * (MAX_FIELD_BYTES + 8))?;
        let mut r = Reader { b: section, at: 0 };
        let mut out = Vec::new();
        while !r.at_end() {
            if out.len() == MAX_FIELDS {
                return Err(BhttpError::Fields(MAX_FIELDS));
            }
            let name = r.text("field name")?;
            let value = r.text("field value")?;
            out.push((name, value));
        }
        Ok(out)
    }

    fn content(&mut self) -> Result<Vec<u8>, BhttpError> {
        if self.at_end() {
            return Ok(Vec::new());
        }
        Ok(self.bytes("content", MAX_CONTENT_BYTES)?.to_vec())
    }

    /// Trailers (none accepted), then padding: zeros to the end.
    fn finish(&mut self) -> Result<(), BhttpError> {
        if !self.at_end() && self.varint("trailer section")? != 0 {
            return Err(BhttpError::Trailers);
        }
        if let Some(i) = self.b[self.at..].iter().position(|b| *b != 0) {
            return Err(BhttpError::Padding(self.at + i));
        }
        Ok(())
    }
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_varint(out, b.len() as u64);
    out.extend_from_slice(b);
}

fn put_fields(out: &mut Vec<u8>, fields: &[(String, String)]) {
    let mut section = Vec::new();
    for (name, value) in fields {
        put_bytes(&mut section, name.as_bytes());
        put_bytes(&mut section, value.as_bytes());
    }
    put_bytes(out, &section);
}

/// Pads `out` with zero bytes to `pad_to`, when it is shorter.
fn pad(mut out: Vec<u8>, pad_to: usize) -> Vec<u8> {
    if out.len() < pad_to {
        out.resize(pad_to, 0);
    }
    out
}

impl Request {
    /// The known-length encoding, then zeros up to `pad_to` bytes.
    pub fn encode(&self, pad_to: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(pad_to.max(64 + self.content.len()));
        put_varint(&mut out, 0);
        put_bytes(&mut out, self.method.as_bytes());
        put_bytes(&mut out, self.scheme.as_bytes());
        put_bytes(&mut out, self.authority.as_bytes());
        put_bytes(&mut out, self.path.as_bytes());
        put_fields(&mut out, &self.headers);
        put_bytes(&mut out, &self.content);
        put_varint(&mut out, 0); // no trailers
        pad(out, pad_to)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, BhttpError> {
        let mut r = Reader { b: bytes, at: 0 };
        let framing = r.varint("framing indicator")?;
        if framing != 0 {
            return Err(BhttpError::Framing(framing));
        }
        let method = r.text("method")?;
        let scheme = r.text("scheme")?;
        let authority = r.text("authority")?;
        let path = r.text("path")?;
        let headers = r.fields("header section")?;
        let content = r.content()?;
        r.finish()?;
        Ok(Self {
            method,
            scheme,
            authority,
            path,
            headers,
            content,
        })
    }

    /// The first value of a header field, by case-insensitive name.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

impl Response {
    pub fn encode(&self, pad_to: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(pad_to.max(64 + self.content.len()));
        put_varint(&mut out, 1);
        put_varint(&mut out, u64::from(self.status));
        put_fields(&mut out, &self.headers);
        put_bytes(&mut out, &self.content);
        put_varint(&mut out, 0);
        pad(out, pad_to)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, BhttpError> {
        let mut r = Reader { b: bytes, at: 0 };
        let framing = r.varint("framing indicator")?;
        if framing != 1 {
            return Err(BhttpError::Framing(framing));
        }
        let status = r.varint("status")?;
        // 1xx would be an informational response: not produced by the TD, refused.
        if !(200..=599).contains(&status) {
            return Err(BhttpError::Status(status));
        }
        let headers = r.fields("header section")?;
        let content = r.content()?;
        r.finish()?;
        Ok(Self {
            status: status as u16,
            headers,
            content,
        })
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

// ── Indeterminate-length responses (RFC 9292 §3.2), piece by piece ─────────

/// The writing side of an indeterminate-length response: its head, its
/// content chunks, its end. Their concatenation, then zeros, is one response.
pub mod indeterminate {
    use super::put_varint;

    /// Framing indicator 3, the status, the header fields, their end.
    pub fn head(status: u16, headers: &[(&str, &str)]) -> Vec<u8> {
        let mut out = Vec::with_capacity(64);
        put_varint(&mut out, 3);
        put_varint(&mut out, u64::from(status));
        for (name, value) in headers {
            super::put_bytes(&mut out, name.as_bytes());
            super::put_bytes(&mut out, value.as_bytes());
        }
        put_varint(&mut out, 0);
        out
    }

    /// One content chunk: its length, then `data`. `data` is never empty: a
    /// zero length ends the content.
    pub fn chunk(out: &mut Vec<u8>, data: &[u8]) {
        assert!(!data.is_empty(), "an empty content chunk ends the content");
        super::put_bytes(out, data);
    }

    /// The bytes [`chunk`] takes for `n` bytes of data.
    pub fn chunk_len(n: usize) -> usize {
        let mut v = Vec::new();
        put_varint(&mut v, n as u64);
        v.len() + n
    }

    /// The end of the content, then an empty trailer section.
    pub const END: [u8; 2] = [0, 0];
}

/// One piece of an indeterminate-length response, as it is read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Part {
    /// The final status and its header fields.
    Head {
        status: u16,
        headers: Vec<(String, String)>,
    },
    /// Content bytes, in order; never empty.
    Content(Vec<u8>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadAt {
    Framing,
    Status,
    FieldName,
    FieldValue,
    ChunkLength,
    /// Bytes of the current content chunk still to come.
    ChunkData(usize),
    Trailers,
    Padding,
}

/// Reads one indeterminate-length response (RFC 9292 §3.2) from bytes as
/// they arrive. Refuses another framing, an informational or invalid
/// status, more than [`MAX_FIELDS`] fields or one over [`MAX_FIELD_BYTES`],
/// content past `max_content`, any trailer, a non-zero byte after the end.
/// Whole only at [`Self::finish`], once the content and trailers ended.
pub struct ResponseReader {
    at: ReadAt,
    buf: Vec<u8>,
    headers: Vec<(String, String)>,
    name: String,
    status: u16,
    content: usize,
    max_content: usize,
}

impl ResponseReader {
    pub fn new(max_content: usize) -> Self {
        Self {
            at: ReadAt::Framing,
            buf: Vec::new(),
            headers: Vec::new(),
            name: String::new(),
            status: 0,
            content: 0,
            max_content,
        }
    }

    /// `bytes`, then every part they complete. An error ends the reading.
    /// Content leaves as it comes; what stays held is at most one field.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<Part>, BhttpError> {
        self.buf.extend_from_slice(bytes);
        let mut parts = Vec::new();
        let mut at = 0;
        loop {
            let rest = &self.buf[at..];
            let mut r = Reader { b: rest, at: 0 };
            match self.at {
                ReadAt::ChunkData(left) => {
                    let take = left.min(rest.len());
                    if take == 0 {
                        break;
                    }
                    parts.push(Part::Content(rest[..take].to_vec()));
                    at += take;
                    self.at = if take == left {
                        ReadAt::ChunkLength
                    } else {
                        ReadAt::ChunkData(left - take)
                    };
                }
                ReadAt::Padding => {
                    if let Some(i) = rest.iter().position(|b| *b != 0) {
                        return Err(BhttpError::Padding(i));
                    }
                    at = self.buf.len();
                    break;
                }
                ReadAt::FieldName if rest.first() == Some(&0) => {
                    at += 1;
                    parts.push(Part::Head {
                        status: self.status,
                        headers: std::mem::take(&mut self.headers),
                    });
                    self.at = ReadAt::ChunkLength;
                }
                ReadAt::FieldName | ReadAt::FieldValue => {
                    let name = self.at == ReadAt::FieldName;
                    if name && self.headers.len() == MAX_FIELDS {
                        return Err(BhttpError::Fields(MAX_FIELDS));
                    }
                    let what = if name { "field name" } else { "field value" };
                    let text = match r.bytes(what, MAX_FIELD_BYTES) {
                        Ok(b) => {
                            String::from_utf8(b.to_vec()).map_err(|_| BhttpError::Text(what))?
                        }
                        Err(BhttpError::Truncated(_)) => break,
                        Err(e) => return Err(e),
                    };
                    at += r.at;
                    if name {
                        self.name = text;
                        self.at = ReadAt::FieldValue;
                    } else {
                        self.headers.push((std::mem::take(&mut self.name), text));
                        self.at = ReadAt::FieldName;
                    }
                }
                ReadAt::Framing | ReadAt::Status | ReadAt::ChunkLength | ReadAt::Trailers => {
                    let Ok(v) = r.varint("") else {
                        break;
                    };
                    at += r.at;
                    self.at = match (self.at, v) {
                        (ReadAt::Framing, 3) => ReadAt::Status,
                        (ReadAt::Framing, f) => return Err(BhttpError::Framing(f)),
                        (ReadAt::Status, s) if (200..=599).contains(&s) => {
                            self.status = s as u16;
                            ReadAt::FieldName
                        }
                        // 1xx, informational: not produced by the TD, refused.
                        (ReadAt::Status, s) => return Err(BhttpError::Status(s)),
                        (ReadAt::ChunkLength, 0) => ReadAt::Trailers,
                        (ReadAt::ChunkLength, n) => {
                            let room = (self.max_content - self.content) as u64;
                            if n > room {
                                return Err(BhttpError::TooLong {
                                    what: "content",
                                    len: (self.content as u64).saturating_add(n),
                                    max: self.max_content,
                                });
                            }
                            self.content += n as usize;
                            ReadAt::ChunkData(n as usize)
                        }
                        (ReadAt::Trailers, 0) => ReadAt::Padding,
                        (ReadAt::Trailers, _) => return Err(BhttpError::Trailers),
                        _ => unreachable!("states read by varint"),
                    };
                }
            }
        }
        self.buf.drain(..at);
        if self.buf.len() > MAX_FIELD_BYTES + 8 {
            return Err(BhttpError::TooLong {
                what: "field",
                len: self.buf.len() as u64,
                max: MAX_FIELD_BYTES + 8,
            });
        }
        Ok(parts)
    }

    /// The response ended: whole only once its content and trailers did.
    pub fn finish(&self) -> Result<(), BhttpError> {
        match self.at {
            ReadAt::Padding => Ok(()),
            ReadAt::Framing | ReadAt::Status => Err(BhttpError::Truncated("control data")),
            ReadAt::FieldName | ReadAt::FieldValue => Err(BhttpError::Truncated("header section")),
            ReadAt::ChunkLength | ReadAt::ChunkData(_) => Err(BhttpError::Truncated("content")),
            ReadAt::Trailers => Err(BhttpError::Truncated("trailer section")),
        }
    }
}
