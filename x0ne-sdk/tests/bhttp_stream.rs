//! The reader of an indeterminate-length Binary HTTP response (RFC 9292
//! §3.2, `bhttp::ResponseReader`): the plaintext of a sealed stream, which
//! arrives cut at chunk boundaries that follow neither its fields nor its
//! content chunks. What it reads, and what it refuses.

use x0ne_sdk::bhttp::indeterminate::{chunk, head, END};
use x0ne_sdk::bhttp::{BhttpError, Part, ResponseReader, MAX_FIELDS, MAX_FIELD_BYTES};

const SSE: (&str, &str) = ("content-type", "text/event-stream");

/// A 200 with two content chunks, its end, then zeros.
fn response() -> Vec<u8> {
    let mut out = head(200, &[SSE]);
    chunk(&mut out, b"hello ");
    chunk(&mut out, b"world");
    out.extend_from_slice(&END);
    out.extend_from_slice(&[0; 5]);
    out
}

/// The head and the content `bytes` read to, pushed `step` bytes at a time.
fn read(bytes: &[u8], step: usize) -> Result<(Option<u16>, Vec<u8>, ResponseReader), BhttpError> {
    let mut r = ResponseReader::new(1 << 20);
    let (mut status, mut content) = (None, Vec::new());
    for piece in bytes.chunks(step.max(1)) {
        for part in r.push(piece)? {
            match part {
                Part::Head { status: s, headers } => {
                    assert_eq!(headers, [(SSE.0.to_string(), SSE.1.to_string())]);
                    status = Some(s);
                }
                Part::Content(c) => {
                    assert!(!c.is_empty());
                    content.extend(c);
                }
            }
        }
    }
    Ok((status, content, r))
}

/// The response pushed whole, byte by byte, or in pieces of 3 bytes gives
/// one head and one content, whatever the cuts: a field or a content chunk
/// split between two chunks of the stream waits for its rest. A failure:
/// a reader that depends on where the stream's chunks cut its plaintext.
#[test]
fn a_response_cut_anywhere_reads_as_the_whole() {
    let wire = response();
    for step in [wire.len(), 1, 3] {
        let (status, content, r) = read(&wire, step).unwrap();
        assert_eq!(status, Some(200), "step {step}");
        assert_eq!(content, b"hello world", "step {step}");
        r.finish().unwrap();
    }
}

/// SECURITY (CVE-2026-48480's class: a cut stream read as whole): a
/// response that stops before its content and trailers ended is refused at
/// `finish`, wherever it stops, naming where. A failure: a cut answer
/// taken for a whole one.
#[test]
fn a_response_that_stops_early_is_refused_at_its_end() {
    let wire = response();
    let end = wire.len() - 5;
    let mut sections = std::collections::BTreeSet::new();
    for cut in 0..end {
        let (_, _, r) = read(&wire[..cut], 1).unwrap();
        match r.finish() {
            Err(BhttpError::Truncated(what)) => {
                sections.insert(what);
            }
            other => panic!("cut at {cut}: {other:?}"),
        }
    }
    assert_eq!(
        sections.into_iter().collect::<Vec<_>>(),
        [
            "content",
            "control data",
            "header section",
            "trailer section"
        ]
    );
    read(&wire[..end], 1).unwrap().2.finish().unwrap();
}

/// The refusal `bytes` meet, pushed whole into a reader of `max` content.
fn refusal(bytes: &[u8], max: usize) -> BhttpError {
    let mut r = ResponseReader::new(max);
    match r.push(bytes) {
        Err(e) => e,
        Ok(parts) => panic!("read: {parts:?}"),
    }
}

/// SECURITY: what a TD may not send in a stream's plaintext is refused as
/// it arrives: another framing, an informational or invalid status, more
/// than 64 fields, a field over 8 KiB or not UTF-8, content past the
/// reader's bound, a trailer field, a non-zero byte after the end. A
/// failure: bytes outside the response, or more than the caller bounds,
/// read as part of it.
#[test]
fn what_a_response_may_not_hold_is_refused() {
    let mut known_length = head(200, &[SSE]);
    known_length[0] = 1;
    assert_eq!(refusal(&known_length, 64), BhttpError::Framing(1));
    assert_eq!(refusal(&head(103, &[SSE]), 64), BhttpError::Status(103));
    assert_eq!(refusal(&head(600, &[SSE]), 64), BhttpError::Status(600));

    let names: Vec<String> = (0..=MAX_FIELDS).map(|i| format!("x-{i}")).collect();
    let many: Vec<(&str, &str)> = names.iter().map(|n| (n.as_str(), "v")).collect();
    assert_eq!(
        refusal(&head(200, &many), 64),
        BhttpError::Fields(MAX_FIELDS)
    );
    let long = "v".repeat(MAX_FIELD_BYTES + 1);
    assert!(matches!(
        refusal(&head(200, &[("x-long", &long)]), 64),
        BhttpError::TooLong {
            what: "field value",
            ..
        }
    ));
    // Framing 3, status 200, a field "a" whose value is the byte 0xff.
    assert_eq!(
        refusal(&[3, 0x40, 0xc8, 1, b'a', 1, 0xff, 0], 64),
        BhttpError::Text("field value")
    );

    let mut past = head(200, &[SSE]);
    chunk(&mut past, &[b'x'; 6]);
    chunk(&mut past, &[b'x'; 5]);
    assert!(matches!(
        refusal(&past, 10),
        BhttpError::TooLong {
            what: "content",
            len: 11,
            max: 10
        }
    ));

    let mut trailer = head(200, &[SSE]);
    chunk(&mut trailer, b"x");
    trailer.extend_from_slice(&[0, 1, b't', 1, b'v', 0]);
    assert_eq!(refusal(&trailer, 64), BhttpError::Trailers);

    let mut tail = response();
    tail.push(9);
    let at = tail.len() - (response().len() - 5) - 1;
    assert_eq!(refusal(&tail, 64), BhttpError::Padding(at));
}
