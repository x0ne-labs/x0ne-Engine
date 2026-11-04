//! Chunked Oblivious HTTP, draft-ietf-ohai-chunked-ohttp-08 (approved by the
//! IESG on 2026-02-27, with the RFC Editor): the example of its Appendix A,
//! byte for byte, then what a carrier can do to a chunked response and to a
//! chunked request. The vectors were read from the draft's text; no other
//! implementation was run.

use x0ne_sdk::hpke::{self, kdf, PrivateKey};
use x0ne_sdk::ohttp::{self, KeyConfig, OhttpError};

fn unhex(s: &str) -> Vec<u8> {
    let s: String = s.split_whitespace().collect();
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn arr<const N: usize>(s: &str) -> [u8; N] {
    unhex(s).try_into().unwrap()
}

// ── Appendix A ───────────────────────────────────────────────────────────────

const SK_R: &str = "1c190d72acdbe4dbc69e680503bb781a932c70a12c8f3754434c67d8640d8698";
const KEY_CONFIG: &str = "010020668eb21aace159803974a4c67f08b4152d29bed10735fd08f98ccdd6fe
                          09570800080001000100010003";
const REQUEST: &str = "00034745540568747470730b6578616d706c652e636f6d012f";
const SK_E: &str = "b26d565f3f875ed480d1abced3d665159650c99174fd0b124ac4bda0c64ae324";
const PK_E: &str = "8811eb457e100811c40a0aa71340a1b81d804bb986f736f2f566a7199761a032";
const INFO: &str = "6d6573736167652f6268747470206368756e6b65642072657175657374000100
                    2000010001";
const ENCAPSULATED_REQUEST: &str = "01002000010001
    8811eb457e100811c40a0aa71340a1b81d804bb986f736f2f566a7199761a032
    1c2ad24942d4d692563012f2980c8fef437a336b9b2fc938ef77a5834f
    1d2e33d8fd25577afe31bd1c79d094f76b6250ae6549b473ecd950501311
    001c6c1395d0ef7c1022297966307b8a7f";
const RESPONSE: &str = "0140c8";
const EXPORTED: &str = "1d4484834ae36102a6ac42a5523454d9";
const SALT: &str = "8811eb457e100811c40a0aa71340a1b81d804bb986f736f2f566a7199761a032
                    bcce7f4cb921309ba5d62edf1769ef09";
const PRK: &str = "3967884b5f7b4bce4a5320a3e3f79fdc97389f7deba1c1e11c5ea62278187786";
const AEAD_KEY: &str = "8209f78f2a1610d80c7125009b00aff0";
const AEAD_NONCE: &str = "fead854635d2d5527d64f546";
const RESPONSE_NONCE: &str = "bcce7f4cb921309ba5d62edf1769ef09";
const ENCAPSULATED_RESPONSE: &str = "bcce7f4cb921309ba5d62edf1769ef09
    1179bf1cc87fa0e2c02de4546945aa3d1e48
    12b348b5bd4c594c16b6170b07b475845d1f32
    00ed9d8a796617a5b27265f4d73247f639";
const CHUNK_NONCES: [&str; 3] = [
    "fead854635d2d5527d64f546",
    "fead854635d2d5527d64f547",
    "fead854635d2d5527d64f544",
];

fn config() -> KeyConfig {
    KeyConfig::parse_any_suite(&unhex(KEY_CONFIG)).unwrap()
}

fn gateway_key() -> PrivateKey {
    PrivateKey::from_bytes(arr(SK_R))
}

/// The request in the example's three chunks: 12 bytes, 13, and an empty final.
fn request_chunks() -> Vec<Vec<u8>> {
    let r = unhex(REQUEST);
    vec![r[..12].to_vec(), r[12..].to_vec(), Vec::new()]
}

fn sealed_request() -> (Vec<u8>, ohttp::ChunkedClientResponse) {
    let chunks = request_chunks();
    let views: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
    ohttp::encapsulate_chunked_request_with_ephemeral(
        &config(),
        &views,
        &PrivateKey::from_bytes(arr(SK_E)),
    )
    .unwrap()
}

/// The response as the example writes it: one byte, two, and an empty final.
fn sealed_response() -> Vec<u8> {
    let (_, server) = ohttp::decapsulate_chunked_request(
        &gateway_key(),
        1,
        &unhex(ENCAPSULATED_REQUEST),
        1 << 20,
    )
    .unwrap();
    let mut w = server.writer(&arr(RESPONSE_NONCE)).unwrap();
    let r = unhex(RESPONSE);
    let mut out = w.chunk(&r[..1]).unwrap();
    out.extend(w.chunk(&r[1..]).unwrap());
    out.extend(w.last(b"").unwrap());
    out
}

/// A failure means the chunked request differs from the draft's by a byte:
/// the info string, the encapsulated key, a chunk's length, its AAD or its
/// sequence number.
#[test]
fn appendix_a_request_is_sealed_byte_for_byte() {
    assert_eq!(config().key_id, 1);
    let mut info = ohttp::CHUNKED_REQUEST_LABEL.to_vec();
    info.extend_from_slice(&unhex("00 01 0020 0001 0001"));
    assert_eq!(info, unhex(INFO));
    let sk_e = PrivateKey::from_bytes(arr(SK_E));
    assert_eq!(sk_e.public_key().to_vec(), unhex(PK_E));
    let (request, _) = sealed_request();
    assert_eq!(hex(&request), hex(&unhex(ENCAPSULATED_REQUEST)));
}

/// The gateway opens the example's request into the Binary HTTP request it
/// carried; a failure means one of its three chunks did not open in turn.
#[test]
fn appendix_a_request_opens_at_the_gateway() {
    let (request, server) = ohttp::decapsulate_chunked_request(
        &gateway_key(),
        1,
        &unhex(ENCAPSULATED_REQUEST),
        1 << 20,
    )
    .unwrap();
    assert_eq!(*request, unhex(REQUEST));
    assert_eq!(server.enc().to_vec(), unhex(PK_E));
}

/// Each step of §6.2 against the example's intermediate values: the secret
/// exported under the chunked label, the salt, the PRK, the key, the base
/// nonce, each chunk's nonce, then the whole Encapsulated Response.
#[test]
fn appendix_a_response_is_sealed_byte_for_byte() {
    let (_, ctx) = hpke::setup_base_s_with_ephemeral(
        &config().public_key,
        &unhex(INFO),
        &PrivateKey::from_bytes(arr(SK_E)),
    )
    .unwrap();
    let secret = ctx.export(ohttp::CHUNKED_RESPONSE_LABEL, 16).unwrap();
    assert_eq!(hex(&secret), EXPORTED);
    let mut salt = unhex(PK_E);
    salt.extend(unhex(RESPONSE_NONCE));
    assert_eq!(salt, unhex(SALT));
    let prk = kdf::extract(&salt, &secret);
    assert_eq!(hex(&*prk), PRK);
    assert_eq!(hex(&kdf::expand(&*prk, b"key", 16).unwrap()), AEAD_KEY);
    let base = kdf::expand(&*prk, b"nonce", 12).unwrap();
    assert_eq!(hex(&base), AEAD_NONCE);
    for (counter, want) in CHUNK_NONCES.iter().enumerate() {
        assert_eq!(
            hex(&ohttp::chunk_nonce(&arr(AEAD_NONCE), counter as u64)),
            *want
        );
    }
    assert_eq!(hex(&sealed_response()), hex(&unhex(ENCAPSULATED_RESPONSE)));
}

/// The client reads the example's response chunk by chunk, whether it comes
/// whole or one byte at a time, and has it all only with the final chunk.
#[test]
fn appendix_a_response_opens_chunk_by_chunk() {
    let wire = unhex(ENCAPSULATED_RESPONSE);
    for step in [wire.len(), 1, 2, 17] {
        let (_, client) = sealed_request();
        let mut reader = client.reader();
        let mut got = Vec::new();
        for piece in wire.chunks(step) {
            for chunk in reader.push(piece).unwrap() {
                got.push(chunk.to_vec());
            }
        }
        assert_eq!(got, vec![vec![0x01], vec![0x40, 0xc8]], "step {step}");
        assert_eq!(reader.chunks_read(), 2);
        let last = reader.finish().unwrap();
        assert!(last.is_empty(), "step {step}");
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ── What a carrier can do to a chunked response ──────────────────────────────

/// The example's response read by a fresh client: the chunks opened, then
/// the final one, or the first refusal.
fn read(wire: &[u8]) -> Result<(Vec<Vec<u8>>, Vec<u8>), OhttpError> {
    let (_, client) = sealed_request();
    let mut reader = client.reader();
    let chunks = reader.push(wire)?.into_iter().map(|c| c.to_vec()).collect();
    let last = reader.finish()?.to_vec();
    Ok((chunks, last))
}

/// The example's response cut into nonce, chunk 1, chunk 2, final.
fn parts() -> [Vec<u8>; 4] {
    let w = unhex(ENCAPSULATED_RESPONSE);
    [
        w[..16].to_vec(),
        w[16..34].to_vec(),
        w[34..53].to_vec(),
        w[53..].to_vec(),
    ]
}

/// SECURITY (CVE-2026-48480: a reader that took a cut stream for a whole
/// one): every truncation of the response, at every byte, is refused; none
/// reads as complete. A failure is a stream without its final chunk taken
/// for a whole answer.
#[test]
fn every_truncation_of_a_response_is_refused() {
    let wire = unhex(ENCAPSULATED_RESPONSE);
    assert!(read(&wire).is_ok());
    for cut in 0..wire.len() {
        assert!(read(&wire[..cut]).is_err(), "cut at {cut} read as whole");
    }
}

/// SECURITY: chunks reordered, one replayed in place of the next, one
/// dropped, a non-final chunk passed off as the final one (and the final as
/// a non-final), one bit flipped anywhere, a byte appended: refused. A
/// failure is a response the gateway did not write, read as its own.
#[test]
fn reordered_replayed_dropped_relabelled_or_altered_chunks_are_refused() {
    let [nonce, c1, c2, fin] = parts();
    let join = |ps: &[&[u8]]| ps.concat();
    let refused = |what: &str, wire: Vec<u8>| assert!(read(&wire).is_err(), "{what}");
    refused("reordered", join(&[&nonce, &c2, &c1, &fin]));
    refused("chunk 1 replayed", join(&[&nonce, &c1, &c1, &fin]));
    refused("chunk 1 dropped", join(&[&nonce, &c2, &fin]));
    refused("chunk 2 dropped", join(&[&nonce, &c1, &fin]));
    // Chunk 2 relabelled final: a zero length before its ciphertext.
    let mut as_final = vec![0u8];
    as_final.extend_from_slice(&c2[1..]);
    refused(
        "non-final passed off as final",
        join(&[&nonce, &c1, &as_final]),
    );
    // The final relabelled non-final: its length before its ciphertext.
    let mut as_chunk = vec![u8::try_from(fin.len() - 1).unwrap()];
    as_chunk.extend_from_slice(&fin[1..]);
    refused(
        "final passed off as non-final",
        join(&[&nonce, &c1, &c2, &as_chunk]),
    );
    let whole = unhex(ENCAPSULATED_RESPONSE);
    for i in 0..whole.len() {
        for bit in [0x01u8, 0x80] {
            let mut w = whole.clone();
            w[i] ^= bit;
            assert!(
                read(&w).is_err(),
                "byte {i} bit {bit:#x}: an altered response opened"
            );
        }
    }
    let mut longer = whole;
    longer.push(0);
    refused("a byte after the final chunk", longer);
}

/// SECURITY (draft §6): a non-final chunk with no data is a decryption
/// error. Sealed under the example's own key and nonce at counter 0, it
/// opens, and is refused for being empty.
#[test]
fn an_empty_non_final_chunk_is_refused() {
    let [nonce, _, c2, fin] = parts();
    let empty = hpke::aead_seal(&arr(AEAD_KEY), &arr(AEAD_NONCE), b"", b"");
    let mut chunk = vec![u8::try_from(empty.len()).unwrap()];
    chunk.extend(empty);
    let wire = [nonce.as_slice(), &chunk, &c2, &fin].concat();
    assert!(read(&wire).is_err());
    // The writer makes none.
    let (_, server) = ohttp::decapsulate_chunked_request(
        &gateway_key(),
        1,
        &unhex(ENCAPSULATED_REQUEST),
        1 << 20,
    )
    .unwrap();
    let mut w = server.writer(&arr(RESPONSE_NONCE)).unwrap();
    assert!(w.chunk(b"").is_err());
}

/// A chunk longer than the reader's bound is refused before its bytes are
/// held: a carrier cannot make the caller buffer an unbounded chunk.
#[test]
fn a_chunk_longer_than_the_bound_is_refused_as_its_length_arrives() {
    let [nonce, ..] = parts();
    let (_, client) = sealed_request();
    let mut reader = client.reader();
    let mut huge = nonce;
    // A 4-byte varint: 2^20.
    huge.extend_from_slice(&[0x80, 0x10, 0x00, 0x00]);
    assert!(reader.push(&huge).is_err());
}

// ── What a carrier can do to a chunked request ───────────────────────────────

/// SECURITY: the gateway opens a chunked request only whole: cut anywhere,
/// one bit flipped, its chunks reordered, a chunk dropped, its final chunk
/// missing, or sealed under the non-chunked label, it does not open.
#[test]
fn a_cut_altered_reordered_or_relabelled_chunked_request_does_not_open() {
    let wire = unhex(ENCAPSULATED_REQUEST);
    let open = |w: &[u8]| ohttp::decapsulate_chunked_request(&gateway_key(), 1, w, 1 << 20);
    assert!(open(&wire).is_ok());
    for cut in 0..wire.len() {
        assert!(open(&wire[..cut]).is_err(), "cut at {cut}");
    }
    for i in 0..wire.len() {
        let mut w = wire.clone();
        w[i] ^= 0x80;
        assert!(open(&w).is_err(), "byte {i}");
    }
    let (head, c1, c2, fin) = (&wire[..39], &wire[39..68], &wire[68..98], &wire[98..]);
    assert!(open(&[head, c2, c1, fin].concat()).is_err(), "reordered");
    assert!(open(&[head, c1, fin].concat()).is_err(), "dropped");
    assert!(open(&[head, c1, c2].concat()).is_err(), "no final chunk");
    // The whole request sealed as one non-chunked Encapsulated Request.
    let (plain, _) = ohttp::encapsulate_request_with_ephemeral(
        &config(),
        &unhex(REQUEST),
        &PrivateKey::from_bytes(arr(SK_E)),
    )
    .unwrap();
    let mut relabelled = plain[..39].to_vec();
    relabelled.push(0);
    relabelled.extend_from_slice(&plain[39..]);
    assert!(
        open(&relabelled).is_err(),
        "sealed under message/bhttp request"
    );
    // And the bound: a request whose chunks open to more than `max` bytes.
    assert!(ohttp::decapsulate_chunked_request(&gateway_key(), 1, &wire, 24).is_err());
}

/// The client seals no empty non-final chunk, and needs a final one.
#[test]
fn the_client_seals_chunks_the_draft_allows_only() {
    let sk = || PrivateKey::from_bytes(arr(SK_E));
    let seal = |chunks: &[&[u8]]| {
        ohttp::encapsulate_chunked_request_with_ephemeral(&config(), chunks, &sk())
    };
    assert!(seal(&[]).is_err(), "no final chunk");
    assert!(seal(&[b"", b"x"]).is_err(), "an empty non-final chunk");
    assert!(seal(&[b"one final chunk"]).is_ok());
}

/// Every label this crate seals or exports under is no prefix of another:
/// the chunked ones are the draft's, distinct from RFC 9458's and x0ne's.
#[test]
fn the_ohttp_and_e2e_labels_are_no_prefix_of_one_another() {
    use x0ne_sdk::e2e::{BINDING_LABEL, GPU_BINDING_LABEL, GPU_NONCE_LABEL};
    let labels = [
        ohttp::REQUEST_LABEL,
        ohttp::RESPONSE_LABEL,
        ohttp::CHUNKED_REQUEST_LABEL,
        ohttp::CHUNKED_RESPONSE_LABEL,
        BINDING_LABEL,
        GPU_BINDING_LABEL,
        GPU_NONCE_LABEL,
    ];
    for (i, x) in labels.iter().enumerate() {
        for y in labels.iter().skip(i + 1) {
            assert!(!x.starts_with(y) && !y.starts_with(x), "{x:?} {y:?}");
        }
    }
}

/// SECURITY (draft §6): the gateway refuses a request whose non-final chunk
/// is empty, sealed under the example's own context, as the client refuses
/// to seal one. A failure: a request the draft calls a decryption error
/// opened at the gateway.
#[test]
fn the_gateway_refuses_an_empty_non_final_request_chunk() {
    let (enc, mut ctx) = hpke::setup_base_s_with_ephemeral(
        &config().public_key,
        &unhex(INFO),
        &PrivateKey::from_bytes(arr(SK_E)),
    )
    .unwrap();
    let mut wire = unhex(ENCAPSULATED_REQUEST)[..7].to_vec();
    wire.extend_from_slice(&enc);
    let empty = ctx.seal(b"", b"").unwrap();
    wire.push(u8::try_from(empty.len()).unwrap());
    wire.extend(empty);
    wire.push(0);
    wire.extend(ctx.seal(b"final", &unhex(REQUEST)).unwrap());
    let Err(e) = ohttp::decapsulate_chunked_request(&gateway_key(), 1, &wire, 1 << 20) else {
        panic!("an empty non-final chunk opened");
    };
    let e = e.to_string();
    assert!(e.contains("a non-final chunk without data"), "{e}");
}

/// The writer seals no chunk of more than 16384 bytes of data, final or
/// not, which every receiver may refuse (draft §3); 16384 it seals. A
/// failure: a TD that writes chunks a conforming client refuses.
#[test]
fn the_writer_seals_no_chunk_over_16384_bytes() {
    let writer = || {
        let (_, server) = ohttp::decapsulate_chunked_request(
            &gateway_key(),
            1,
            &unhex(ENCAPSULATED_REQUEST),
            1 << 20,
        )
        .unwrap();
        server.writer(&arr(RESPONSE_NONCE)).unwrap()
    };
    let over = vec![1u8; ohttp::MAX_CHUNK_DATA + 1];
    let e = writer().chunk(&over).unwrap_err().to_string();
    assert!(e.contains("a chunk over 16384 bytes"), "{e}");
    let e = writer().last(&over).unwrap_err().to_string();
    assert!(e.contains("a chunk over 16384 bytes"), "{e}");
    let mut w = writer();
    assert!(w.chunk(&over[1..]).is_ok());
    assert!(w.last(&over[1..]).is_ok());
}

/// The final chunk is held to the same bound as the others: its bytes past
/// 16384 of data and the tag are refused as they arrive, before the end of
/// the body. A failure: a carrier makes the caller buffer an unbounded
/// final chunk.
#[test]
fn a_final_chunk_longer_than_the_bound_is_refused_as_it_arrives() {
    let [nonce, ..] = parts();
    let (_, client) = sealed_request();
    let mut reader = client.reader();
    let mut wire = nonce;
    wire.push(0);
    wire.extend(vec![7u8; ohttp::MAX_CHUNK_DATA + 16]);
    assert!(reader.push(&wire).is_ok(), "the bound itself is held");
    let e = reader.push(&[7]).unwrap_err().to_string();
    assert!(
        e.contains("a final chunk longer than this reader takes"),
        "{e}"
    );
}
