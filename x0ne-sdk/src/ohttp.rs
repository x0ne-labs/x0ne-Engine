//! Oblivious HTTP message encapsulation (RFC 9458 §3 and §4), on [`hpke`]:
//! the key configuration, the Encapsulated Request and the Encapsulated
//! Response. Standard labels and media types, so any OHTTP relay can carry
//! these bytes and any OHTTP client can be checked against them.
//!
//! One suite only, the one [`hpke`] implements. [`KeyConfig::parse`] refuses
//! a configuration that offers anything else: a client never picks among
//! suites a carrier could have edited.

use crate::hpke::{
    self, kdf, HpkeError, PrivateKey, AEAD_AES_128_GCM, KDF_HKDF_SHA256, KEM_X25519_HKDF_SHA256,
    N_ENC, N_K, N_N, N_PK,
};
use zeroize::Zeroizing;

pub const REQUEST_LABEL: &[u8] = b"message/bhttp request";
pub const RESPONSE_LABEL: &[u8] = b"message/bhttp response";
pub const MEDIA_TYPE_REQUEST: &str = "message/ohttp-req";
pub const MEDIA_TYPE_RESPONSE: &str = "message/ohttp-res";
pub const MEDIA_TYPE_KEYS: &str = "application/ohttp-keys";
/// key_id (1) ‖ kem_id (2) ‖ kdf_id (2) ‖ aead_id (2).
pub const HEADER_LEN: usize = 7;
/// `max(Nn, Nk)` (RFC 9458 §4.4).
pub const RESPONSE_NONCE_LEN: usize = if N_N > N_K { N_N } else { N_K };
/// AES-GCM tag.
const TAG_LEN: usize = 16;
/// What sealing adds to a Binary HTTP request on the wire: the header, the
/// encapsulated key and the AEAD tag (RFC 9458 §4.3).
pub const REQUEST_OVERHEAD: usize = HEADER_LEN + N_ENC + TAG_LEN;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OhttpError {
    #[error("key configuration: {0}")]
    KeyConfig(String),
    #[error("encapsulated request too short: {0} bytes")]
    Short(usize),
    #[error("request names key {0}, this gateway holds another")]
    KeyId(u8),
    #[error(
        "request asks for kem {kem:#06x} kdf {kdf:#06x} aead {aead:#06x}, not this gateway's suite"
    )]
    Suite { kem: u16, kdf: u16, aead: u16 },
    #[error("encapsulated response too short: {0} bytes")]
    ResponseShort(usize),
    #[error("chunked: {0}")]
    Chunked(String),
    #[error("{0}")]
    Hpke(#[from] HpkeError),
}

/// A gateway's key configuration (RFC 9458 §3), with this module's suite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyConfig {
    pub key_id: u8,
    pub public_key: [u8; N_PK],
}

impl KeyConfig {
    /// `key_id ‖ kem_id ‖ public_key ‖ suites length ‖ (kdf_id ‖ aead_id)`.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + 2 + N_PK + 2 + 4);
        out.push(self.key_id);
        out.extend_from_slice(&KEM_X25519_HKDF_SHA256.to_be_bytes());
        out.extend_from_slice(&self.public_key);
        out.extend_from_slice(&4u16.to_be_bytes());
        out.extend_from_slice(&KDF_HKDF_SHA256.to_be_bytes());
        out.extend_from_slice(&AEAD_AES_128_GCM.to_be_bytes());
        out
    }

    /// Exactly [`encode`](Self::encode)'s shape: X25519, one suite,
    /// HKDF-SHA256 with AES-128-GCM, no trailing byte.
    pub fn parse(bytes: &[u8]) -> Result<Self, OhttpError> {
        let (cfg, suites) = Self::parse_with_suites(bytes)?;
        if suites != [(KDF_HKDF_SHA256, AEAD_AES_128_GCM)] {
            return Err(OhttpError::KeyConfig(format!(
                "suites {suites:04x?}, expected exactly [(0001, 0001)]"
            )));
        }
        Ok(cfg)
    }

    /// The key of a configuration that lists our suite among others: for
    /// reading published examples, never for choosing a key to send to.
    pub fn parse_any_suite(bytes: &[u8]) -> Result<Self, OhttpError> {
        let (cfg, suites) = Self::parse_with_suites(bytes)?;
        if !suites.contains(&(KDF_HKDF_SHA256, AEAD_AES_128_GCM)) {
            return Err(OhttpError::KeyConfig(
                "HKDF-SHA256 with AES-128-GCM not offered".into(),
            ));
        }
        Ok(cfg)
    }

    fn parse_with_suites(bytes: &[u8]) -> Result<(Self, Vec<(u16, u16)>), OhttpError> {
        let bad = |m: &str| OhttpError::KeyConfig(m.to_string());
        if bytes.len() < 1 + 2 + N_PK + 2 {
            return Err(bad("too short"));
        }
        let key_id = bytes[0];
        let kem = u16::from_be_bytes([bytes[1], bytes[2]]);
        if kem != KEM_X25519_HKDF_SHA256 {
            return Err(OhttpError::KeyConfig(format!(
                "kem {kem:#06x}, expected X25519"
            )));
        }
        let mut public_key = [0u8; N_PK];
        public_key.copy_from_slice(&bytes[3..3 + N_PK]);
        // Refused here, before any verdict names this key, not only at seal time.
        if hpke::is_low_order(&public_key) {
            return Err(bad(
                "public key has low order: every sender would share one secret (RFC 9180 §7.1.4)",
            ));
        }
        let at = 3 + N_PK;
        let len = usize::from(u16::from_be_bytes([bytes[at], bytes[at + 1]]));
        let rest = &bytes[at + 2..];
        if len == 0 || len % 4 != 0 || rest.len() != len {
            return Err(bad("suites length does not match the bytes"));
        }
        let suites = rest
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| {
                (
                    u16::from_be_bytes([c[0], c[1]]),
                    u16::from_be_bytes([c[2], c[3]]),
                )
            })
            .collect();
        Ok((Self { key_id, public_key }, suites))
    }

    fn header(&self) -> [u8; HEADER_LEN] {
        let mut h = [0u8; HEADER_LEN];
        h[0] = self.key_id;
        h[1..3].copy_from_slice(&KEM_X25519_HKDF_SHA256.to_be_bytes());
        h[3..5].copy_from_slice(&KDF_HKDF_SHA256.to_be_bytes());
        h[5..7].copy_from_slice(&AEAD_AES_128_GCM.to_be_bytes());
        h
    }
}

fn request_info(header: [u8; HEADER_LEN]) -> Vec<u8> {
    let mut info = Vec::with_capacity(REQUEST_LABEL.len() + 1 + HEADER_LEN);
    info.extend_from_slice(REQUEST_LABEL);
    info.push(0);
    info.extend_from_slice(&header);
    info
}

/// What one side keeps to read or write the response: `enc` and the secret
/// exported from the request's context.
struct ResponseSecret {
    enc: [u8; N_ENC],
    secret: Zeroizing<Vec<u8>>,
}

impl ResponseSecret {
    fn of(enc: [u8; N_ENC], ctx: &hpke::Context) -> Result<Self, HpkeError> {
        Self::exported(enc, ctx, RESPONSE_LABEL)
    }

    /// The secret exported under `label`: [`RESPONSE_LABEL`], or
    /// [`CHUNKED_RESPONSE_LABEL`] for a chunked response (draft-08 §6.2).
    fn exported(enc: [u8; N_ENC], ctx: &hpke::Context, label: &[u8]) -> Result<Self, HpkeError> {
        Ok(Self {
            enc,
            secret: ctx.export(label, RESPONSE_NONCE_LEN)?,
        })
    }

    /// `aead_key ‖ aead_nonce` for one response nonce (RFC 9458 §4.4).
    fn keys(&self, response_nonce: &[u8]) -> Result<(Zeroizing<[u8; N_K]>, [u8; N_N]), HpkeError> {
        let mut salt = Vec::with_capacity(N_ENC + RESPONSE_NONCE_LEN);
        salt.extend_from_slice(&self.enc);
        salt.extend_from_slice(response_nonce);
        let prk = kdf::extract(&salt, &self.secret);
        let key_bytes = kdf::expand(&*prk, b"key", N_K)?;
        let nonce_bytes = kdf::expand(&*prk, b"nonce", N_N)?;
        let mut key = Zeroizing::new([0u8; N_K]);
        key.copy_from_slice(&key_bytes);
        let mut nonce = [0u8; N_N];
        nonce.copy_from_slice(&nonce_bytes);
        Ok((key, nonce))
    }
}

/// The client's half after sending: reads the one response to its request.
pub struct ClientResponse(ResponseSecret);

/// The gateway's half after reading a request: writes its one response.
pub struct ServerResponse(ResponseSecret);

/// Seal `request` (a Binary HTTP message) to `config`. `ikm_e`: fresh random
/// bytes, at least 32, never reused: two requests under one `ikm_e` share
/// the AES-GCM keystream and disclose the XOR of their plaintexts to whoever
/// carries them, before a gateway can refuse the second as a replay.
pub fn encapsulate_request(
    config: &KeyConfig,
    request: &[u8],
    ikm_e: &[u8],
) -> Result<(Vec<u8>, ClientResponse), OhttpError> {
    let sk_e = hpke::ephemeral_key(ikm_e)?;
    encapsulate_request_with_ephemeral(config, request, &sk_e)
}

/// [`encapsulate_request`] with the ephemeral key given (RFC 9458 Appendix A).
pub fn encapsulate_request_with_ephemeral(
    config: &KeyConfig,
    request: &[u8],
    sk_e: &PrivateKey,
) -> Result<(Vec<u8>, ClientResponse), OhttpError> {
    let header = config.header();
    let (enc, mut ctx) =
        hpke::setup_base_s_with_ephemeral(&config.public_key, &request_info(header), sk_e)?;
    let ct = ctx.seal(b"", request)?;
    let mut out = Vec::with_capacity(HEADER_LEN + N_ENC + ct.len());
    out.extend_from_slice(&header);
    out.extend_from_slice(&enc);
    out.extend_from_slice(&ct);
    Ok((out, ClientResponse(ResponseSecret::of(enc, &ctx)?)))
}

/// The gateway's side: the request, and what writes its response. Header
/// first: another key id or another suite is refused before any decryption.
pub fn decapsulate_request(
    sk: &PrivateKey,
    key_id: u8,
    encapsulated: &[u8],
) -> Result<(Zeroizing<Vec<u8>>, ServerResponse), OhttpError> {
    if encapsulated.len() < HEADER_LEN + N_ENC + TAG_LEN {
        return Err(OhttpError::Short(encapsulated.len()));
    }
    let header: [u8; HEADER_LEN] = encapsulated[..HEADER_LEN]
        .try_into()
        .expect("length checked");
    check_header(header, key_id)?;
    let enc: [u8; N_ENC] = encapsulated[HEADER_LEN..HEADER_LEN + N_ENC]
        .try_into()
        .expect("length checked");
    let mut ctx = hpke::setup_base_r(&enc, sk, &request_info(header))?;
    let request = Zeroizing::new(ctx.open(b"", &encapsulated[HEADER_LEN + N_ENC..])?);
    Ok((request, ServerResponse(ResponseSecret::of(enc, &ctx)?)))
}

impl ServerResponse {
    /// The request's encapsulated key: unique per request, what a gateway
    /// remembers to refuse a replay (RFC 9458 §6.5).
    pub fn enc(&self) -> [u8; N_ENC] {
        self.0.enc
    }

    /// `response_nonce ‖ Seal(response)`. `response_nonce`: fresh random.
    pub fn encapsulate(
        &self,
        response: &[u8],
        response_nonce: &[u8; RESPONSE_NONCE_LEN],
    ) -> Result<Vec<u8>, OhttpError> {
        let (key, nonce) = self.0.keys(response_nonce)?;
        let ct = hpke::aead_seal(&key, &nonce, b"", response);
        let mut out = Vec::with_capacity(RESPONSE_NONCE_LEN + ct.len());
        out.extend_from_slice(response_nonce);
        out.extend_from_slice(&ct);
        Ok(out)
    }
}

impl ClientResponse {
    pub fn decapsulate(&self, encapsulated: &[u8]) -> Result<Zeroizing<Vec<u8>>, OhttpError> {
        if encapsulated.len() < RESPONSE_NONCE_LEN + TAG_LEN {
            return Err(OhttpError::ResponseShort(encapsulated.len()));
        }
        let (nonce_bytes, ct) = encapsulated.split_at(RESPONSE_NONCE_LEN);
        let (key, nonce) = self.0.keys(nonce_bytes)?;
        Ok(Zeroizing::new(hpke::aead_open(&key, &nonce, b"", ct)?))
    }
}

// ── Chunked OHTTP (draft-ietf-ohai-chunked-ohttp-08) ────────────────────────
//
// The draft's §4 to §6.2, to the letter: a request is its header, its
// encapsulated key, then chunks, each a QUIC varint length (non-zero) and an
// HPKE-sealed chunk, the last a zero length and a chunk sealed with AAD
// "final" up to the end of the body; a response is the response nonce, then
// chunks framed the same way, sealed under the key and nonce of the
// non-chunked response but exported under the chunked label, each nonce
// XORed with the chunk's counter. Lengths are not authenticated (§6); order
// and finality are, by the nonce and the AAD. A non-final chunk with no data
// is a decryption error (§6). Checked against the draft's Appendix A
// (`tests/chunked_ohttp.rs`).

/// HPKE info label of a chunked request (draft-08 §6.1).
pub const CHUNKED_REQUEST_LABEL: &[u8] = b"message/bhttp chunked request";
/// Exporter label of a chunked response (draft-08 §6.2).
pub const CHUNKED_RESPONSE_LABEL: &[u8] = b"message/bhttp chunked response";
pub const MEDIA_TYPE_CHUNKED_REQUEST: &str = "message/ohttp-chunked-req";
pub const MEDIA_TYPE_CHUNKED_RESPONSE: &str = "message/ohttp-chunked-res";
/// The Incremental header field both chunked messages carry (draft-08 §3).
pub const INCREMENTAL: (&str, &str) = ("incremental", "?1");
/// The AAD of the final chunk (draft-08 §6.1, §6.2).
pub const FINAL_AAD: &[u8] = b"final";
/// Data every receiver takes in one chunk, and what this module seals or
/// reads at most (draft-08 §3).
pub const MAX_CHUNK_DATA: usize = 16_384;
/// The AEAD tag each chunk adds.
pub const CHUNK_TAG: usize = TAG_LEN;

/// `aead_nonce XOR encode(Nn, counter)` (draft-08 §6.2).
pub fn chunk_nonce(aead_nonce: &[u8; N_N], counter: u64) -> [u8; N_N] {
    let mut nonce = *aead_nonce;
    for (n, c) in nonce[N_N - 8..].iter_mut().zip(counter.to_be_bytes()) {
        *n ^= c;
    }
    nonce
}

fn chunked_request_info(header: [u8; HEADER_LEN]) -> Vec<u8> {
    let mut info = Vec::with_capacity(CHUNKED_REQUEST_LABEL.len() + 1 + HEADER_LEN);
    info.extend_from_slice(CHUNKED_REQUEST_LABEL);
    info.push(0);
    info.extend_from_slice(&header);
    info
}

/// A QUIC varint (RFC 9000 §16) at the front of `b`: its value and length,
/// `None` while incomplete. Any of its four lengths is taken (draft-08 §6:
/// the choice of encoding is not authenticated).
fn varint(b: &[u8]) -> Option<(u64, usize)> {
    let first = *b.first()?;
    let len = 1usize << (first >> 6);
    let bytes = b.get(..len)?;
    let mut v = u64::from(first & 0x3f);
    for x in &bytes[1..] {
        v = (v << 8) | u64::from(*x);
    }
    Some((v, len))
}

fn chunked(m: &str) -> OhttpError {
    OhttpError::Chunked(m.to_string())
}

/// The client's half of a chunked exchange: reads the chunked response.
pub struct ChunkedClientResponse(ResponseSecret);

/// The gateway's half: writes the chunked response.
pub struct ChunkedServerResponse(ResponseSecret);

/// `chunks` sealed as a chunked request to `config`: every chunk but the
/// last non-final (none empty), the last final, possibly empty. `ikm_e`:
/// fresh random bytes, at least 32.
pub fn encapsulate_chunked_request(
    config: &KeyConfig,
    chunks: &[&[u8]],
    ikm_e: &[u8],
) -> Result<(Vec<u8>, ChunkedClientResponse), OhttpError> {
    let sk_e = hpke::ephemeral_key(ikm_e)?;
    encapsulate_chunked_request_with_ephemeral(config, chunks, &sk_e)
}

/// [`encapsulate_chunked_request`] with the ephemeral key given (draft-08 Appendix A).
pub fn encapsulate_chunked_request_with_ephemeral(
    config: &KeyConfig,
    chunks: &[&[u8]],
    sk_e: &PrivateKey,
) -> Result<(Vec<u8>, ChunkedClientResponse), OhttpError> {
    let Some((last, rest)) = chunks.split_last() else {
        return Err(chunked("no final chunk"));
    };
    if rest.iter().any(|c| c.is_empty()) {
        return Err(chunked("a non-final chunk without data (draft-08 §6)"));
    }
    if chunks.iter().any(|c| c.len() > MAX_CHUNK_DATA) {
        return Err(chunked("a chunk over 16384 bytes of data"));
    }
    let header = config.header();
    let (enc, mut ctx) =
        hpke::setup_base_s_with_ephemeral(&config.public_key, &chunked_request_info(header), sk_e)?;
    let mut out = Vec::with_capacity(
        HEADER_LEN + N_ENC + chunks.iter().map(|c| c.len() + 4 + TAG_LEN).sum::<usize>(),
    );
    out.extend_from_slice(&header);
    out.extend_from_slice(&enc);
    for chunk in rest {
        let sealed = ctx.seal(b"", chunk)?;
        crate::bhttp::put_varint(&mut out, sealed.len() as u64);
        out.extend_from_slice(&sealed);
    }
    out.push(0);
    out.extend_from_slice(&ctx.seal(FINAL_AAD, last)?);
    let secret = ResponseSecret::exported(enc, &ctx, CHUNKED_RESPONSE_LABEL)?;
    Ok((out, ChunkedClientResponse(secret)))
}

/// The gateway's side of a chunked request held whole: its chunks opened in
/// order into one plaintext of at most `max` bytes, and what writes the
/// response. Header first: another key id or suite is refused before any
/// decryption. No final chunk, an empty non-final chunk, a chunk that does
/// not open in its place: refused.
pub fn decapsulate_chunked_request(
    sk: &PrivateKey,
    key_id: u8,
    encapsulated: &[u8],
    max: usize,
) -> Result<(Zeroizing<Vec<u8>>, ChunkedServerResponse), OhttpError> {
    decapsulate_chunked_request_in(sk, key_id, encapsulated, 1, max)
}

/// [`decapsulate_chunked_request`] whose non-final chunks carry `min_data`
/// bytes at least: each chunk's length is checked, and the total against
/// `max`, before it is opened, so at most `max / min_data + 1` chunks are.
pub fn decapsulate_chunked_request_in(
    sk: &PrivateKey,
    key_id: u8,
    encapsulated: &[u8],
    min_data: usize,
    max: usize,
) -> Result<(Zeroizing<Vec<u8>>, ChunkedServerResponse), OhttpError> {
    if encapsulated.len() < HEADER_LEN + N_ENC + 1 + TAG_LEN {
        return Err(OhttpError::Short(encapsulated.len()));
    }
    let header: [u8; HEADER_LEN] = encapsulated[..HEADER_LEN]
        .try_into()
        .expect("length checked");
    check_header(header, key_id)?;
    let enc: [u8; N_ENC] = encapsulated[HEADER_LEN..HEADER_LEN + N_ENC]
        .try_into()
        .expect("length checked");
    let mut ctx = hpke::setup_base_r(&enc, sk, &chunked_request_info(header))?;
    let mut request = Zeroizing::new(Vec::new());
    let mut at = HEADER_LEN + N_ENC;
    loop {
        let (len, n) = varint(&encapsulated[at..]).ok_or_else(|| chunked("no final chunk"))?;
        at += n;
        let opened = if len == 0 {
            ctx.open(FINAL_AAD, &encapsulated[at..])?
        } else {
            let len = usize::try_from(len).map_err(|_| chunked("chunk length"))?;
            let end = at
                .checked_add(len)
                .filter(|e| *e <= encapsulated.len())
                .ok_or_else(|| chunked("a chunk past the end of the request"))?;
            let data = len.saturating_sub(TAG_LEN);
            if data == 0 {
                return Err(chunked("a non-final chunk without data (draft-08 §6)"));
            }
            if data < min_data {
                return Err(OhttpError::Chunked(format!(
                    "a non-final chunk of {data} bytes, under {min_data}"
                )));
            }
            if request.len() + data > max {
                return Err(chunked("the request opens to more than its bound"));
            }
            let pt = ctx.open(b"", &encapsulated[at..end])?;
            at = end;
            pt
        };
        if request.len() + opened.len() > max {
            return Err(chunked("the request opens to more than its bound"));
        }
        request.extend_from_slice(&Zeroizing::new(opened));
        if len == 0 {
            break;
        }
    }
    let secret = ResponseSecret::exported(enc, &ctx, CHUNKED_RESPONSE_LABEL)?;
    Ok((request, ChunkedServerResponse(secret)))
}

fn check_header(header: [u8; HEADER_LEN], key_id: u8) -> Result<(), OhttpError> {
    if header[0] != key_id {
        return Err(OhttpError::KeyId(header[0]));
    }
    let (kem, kdf_id, aead) = (
        u16::from_be_bytes([header[1], header[2]]),
        u16::from_be_bytes([header[3], header[4]]),
        u16::from_be_bytes([header[5], header[6]]),
    );
    if (kem, kdf_id, aead) != (KEM_X25519_HKDF_SHA256, KDF_HKDF_SHA256, AEAD_AES_128_GCM) {
        return Err(OhttpError::Suite {
            kem,
            kdf: kdf_id,
            aead,
        });
    }
    Ok(())
}

impl ChunkedServerResponse {
    /// The request's encapsulated key: unique per request (RFC 9458 §6.5).
    pub fn enc(&self) -> [u8; N_ENC] {
        self.0.enc
    }

    /// The writer of this response under `response_nonce`, fresh random.
    pub fn writer(
        &self,
        response_nonce: &[u8; RESPONSE_NONCE_LEN],
    ) -> Result<ChunkedResponseWriter, OhttpError> {
        let (key, nonce) = self.0.keys(response_nonce)?;
        Ok(ChunkedResponseWriter {
            key,
            nonce,
            counter: 0,
            prefix: Some(*response_nonce),
        })
    }
}

/// Seals a chunked response, chunk by chunk (draft-08 §6.2). The response
/// nonce leads the first piece it returns.
pub struct ChunkedResponseWriter {
    key: Zeroizing<[u8; N_K]>,
    nonce: [u8; N_N],
    counter: u64,
    prefix: Option<[u8; RESPONSE_NONCE_LEN]>,
}

impl ChunkedResponseWriter {
    fn seal(&mut self, aad: &[u8], pt: &[u8], out: &mut Vec<u8>) -> Result<Vec<u8>, OhttpError> {
        if pt.len() > MAX_CHUNK_DATA {
            return Err(chunked("a chunk over 16384 bytes of data"));
        }
        // 256^Nn chunks would wrap the nonce (§6.2); a u64 stops far before.
        let counter = self.counter;
        self.counter = counter
            .checked_add(1)
            .ok_or_else(|| chunked("chunk counter exhausted"))?;
        if let Some(n) = self.prefix.take() {
            out.extend_from_slice(&n);
        }
        Ok(hpke::aead_seal(
            &self.key,
            &chunk_nonce(&self.nonce, counter),
            aad,
            pt,
        ))
    }

    /// A non-final chunk: its length, then `pt` sealed. `pt` is never empty.
    pub fn chunk(&mut self, pt: &[u8]) -> Result<Vec<u8>, OhttpError> {
        if pt.is_empty() {
            return Err(chunked("a non-final chunk without data (draft-08 §6)"));
        }
        let mut out = Vec::with_capacity(RESPONSE_NONCE_LEN + 4 + pt.len() + TAG_LEN);
        let sealed = self.seal(b"", pt, &mut out)?;
        crate::bhttp::put_varint(&mut out, sealed.len() as u64);
        out.extend_from_slice(&sealed);
        Ok(out)
    }

    /// The final chunk: a zero length, then `pt` sealed with AAD "final".
    /// The body ends after it.
    pub fn last(mut self, pt: &[u8]) -> Result<Vec<u8>, OhttpError> {
        let mut out = Vec::with_capacity(RESPONSE_NONCE_LEN + 1 + pt.len() + TAG_LEN);
        let sealed = self.seal(FINAL_AAD, pt, &mut out)?;
        out.push(0);
        out.extend_from_slice(&sealed);
        Ok(out)
    }
}

impl ChunkedClientResponse {
    /// The reader of the response to this request: chunks of at most
    /// [`MAX_CHUNK_DATA`] bytes of data.
    pub fn reader(self) -> ChunkedResponseReader {
        ChunkedResponseReader {
            secret: self.0,
            keys: None,
            buf: Vec::new(),
            counter: 0,
            in_final: false,
        }
    }
}

/// Opens a chunked response as its bytes arrive: each non-final chunk once
/// whole, in order, never empty; the final one only at the end of the body
/// ([`Self::finish`]), which is the only place a response is whole (draft-08
/// §7.1: a stream without its final chunk is never complete).
pub struct ChunkedResponseReader {
    secret: ResponseSecret,
    keys: ResponseKeys,
    /// Bytes received and not yet opened.
    buf: Vec<u8>,
    counter: u64,
    /// The zero length came: the rest of the body is the final chunk.
    in_final: bool,
}

impl ChunkedResponseReader {
    /// `bytes` as they came off the transport: the data of every non-final
    /// chunk they complete, in order. An error ends the response: nothing
    /// after it is read.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<Zeroizing<Vec<u8>>>, OhttpError> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        if self.keys.is_none() {
            if self.buf.len() < RESPONSE_NONCE_LEN {
                return Ok(out);
            }
            self.keys = Some(self.secret.keys(&self.buf[..RESPONSE_NONCE_LEN])?);
            self.buf.drain(..RESPONSE_NONCE_LEN);
        }
        let mut at = 0;
        while !self.in_final {
            let Some((len, n)) = varint(&self.buf[at..]) else {
                break;
            };
            if len == 0 {
                self.in_final = true;
                at += n;
                break;
            }
            if len > (MAX_CHUNK_DATA + TAG_LEN) as u64 {
                return Err(chunked("a chunk longer than this reader takes"));
            }
            let end = at + n + len as usize;
            if self.buf.len() < end {
                break;
            }
            let pt = open_chunk(&self.keys, &mut self.counter, b"", &self.buf[at + n..end])?;
            if pt.is_empty() {
                return Err(chunked("a non-final chunk without data (draft-08 §6)"));
            }
            out.push(pt);
            at = end;
        }
        self.buf.drain(..at);
        if self.in_final && self.buf.len() > MAX_CHUNK_DATA + TAG_LEN {
            return Err(chunked("a final chunk longer than this reader takes"));
        }
        Ok(out)
    }

    /// Non-final chunks opened so far.
    pub fn chunks_read(&self) -> u64 {
        self.counter
    }

    /// The body ended: the final chunk's data, if the zero length came and
    /// what followed it opens with AAD "final" at its place. Otherwise the
    /// response was cut, and is refused.
    pub fn finish(mut self) -> Result<Zeroizing<Vec<u8>>, OhttpError> {
        if !self.in_final {
            return Err(chunked("the response ended without its final chunk"));
        }
        let ct = std::mem::take(&mut self.buf);
        open_chunk(&self.keys, &mut self.counter, FINAL_AAD, &ct)
    }
}

type ResponseKeys = Option<(Zeroizing<[u8; N_K]>, [u8; N_N])>;

/// The chunk at `counter` opened, and the counter moved past it.
fn open_chunk(
    keys: &ResponseKeys,
    counter: &mut u64,
    aad: &[u8],
    ct: &[u8],
) -> Result<Zeroizing<Vec<u8>>, OhttpError> {
    let (key, nonce) = keys.as_ref().ok_or_else(|| chunked("no response nonce"))?;
    let pt = hpke::aead_open(key, &chunk_nonce(nonce, *counter), aad, ct)?;
    *counter += 1;
    Ok(Zeroizing::new(pt))
}
