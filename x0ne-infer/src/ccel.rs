// SPDX-License-Identifier: Apache-2.0
//! The TDX CC event log (CCEL): parse, replay into RTMR0..3, encode.
//!
//! Format: TCG PC Client "crypto-agile" event log. The first entry is the
//! legacy `TCG_PCR_EVENT` carrying the Spec ID header; every later one is a
//! `TCG_PCR_EVENT2`. In a CCEL the `pcrIndex` field holds the CC MR index:
//! 0 = MRTD, 1..4 = RTMR0..3 [S, to confirm on the first real log, lot 4].
//! The kernel exposes the log at [`CCEL_DATA_PATH`](crate::ccel::CCEL_DATA_PATH) (ACPI CCEL table data).
//!
//! The replay rule is the hardware's: `RTMR[i] = SHA-384(RTMR[i] ‖ digest)`,
//! in log order, `EV_NO_ACTION` entries excluded.

use sha2::{Digest, Sha384};

pub const CCEL_DATA_PATH: &str = "/sys/firmware/acpi/tables/data/CCEL";
/// A real log is tens of KiB; the ACPI area may be padded with zeros.
pub const MAX_LOG_BYTES: u64 = 16 * 1024 * 1024;
const MAX_EVENT_DATA: usize = 1024 * 1024;
const MAX_DIGESTS_PER_EVENT: u32 = 8;

pub const EV_NO_ACTION: u32 = 0x0000_0003;
pub const EV_SEPARATOR: u32 = 0x0000_0004;
pub const EV_EVENT_TAG: u32 = 0x0000_0006;
pub const EV_IPL: u32 = 0x0000_000d;
pub const EV_EFI_BOOT_SERVICES_APPLICATION: u32 = 0x8000_0003;
pub const EV_EFI_BOOT_SERVICES_DRIVER: u32 = 0x8000_0004;
pub const EV_EFI_GPT_EVENT: u32 = 0x8000_0006;
pub const EV_EFI_ACTION: u32 = 0x8000_0007;

pub const TPM_ALG_SHA1: u16 = 0x0004;
pub const TPM_ALG_SHA256: u16 = 0x000b;
pub const TPM_ALG_SHA384: u16 = 0x000c;
pub const TPM_ALG_SHA512: u16 = 0x000d;
pub const TPM_ALG_SM3_256: u16 = 0x0012;

const SPEC_ID_SIGNATURE: &[u8; 16] = b"Spec ID Event03\0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    /// CC MR index as logged: 0 MRTD, 1..4 RTMR0..3.
    pub mr_index: u32,
    pub event_type: u32,
    /// `(TPM_ALG_*, digest)`, as logged.
    pub digests: Vec<(u16, Vec<u8>)>,
    pub data: Vec<u8>,
}

impl Event {
    pub fn new_sha384(mr_index: u32, event_type: u32, digest: [u8; 48], data: Vec<u8>) -> Self {
        Self {
            mr_index,
            event_type,
            digests: vec![(TPM_ALG_SHA384, digest.to_vec())],
            data,
        }
    }

    pub fn sha384(&self) -> Option<[u8; 48]> {
        self.digests
            .iter()
            .find(|(alg, _)| *alg == TPM_ALG_SHA384)
            .and_then(|(_, d)| <[u8; 48]>::try_from(d.as_slice()).ok())
    }

    /// `Some(n)` when the event extends RTMR`n`.
    pub fn rtmr(&self) -> Option<usize> {
        match self.mr_index {
            1..=4 => Some(self.mr_index as usize - 1),
            _ => None,
        }
    }

    /// The event data as text: UTF-16LE (systemd-stub) or UTF-8, NULs
    /// stripped. `None` when it is neither.
    pub fn name(&self) -> Option<String> {
        let d = &self.data;
        if d.is_empty() {
            return None;
        }
        if d.len().is_multiple_of(2) && d.chunks(2).all(|c| c[1] == 0) {
            let units: Vec<u16> = d
                .chunks(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .filter(|u| *u != 0)
                .collect();
            if !units.is_empty() {
                return String::from_utf16(&units).ok();
            }
        }
        let trimmed: Vec<u8> = d.iter().copied().filter(|b| *b != 0).collect();
        String::from_utf8(trimmed)
            .ok()
            .filter(|s| !s.is_empty() && s.chars().all(|c| !c.is_control()))
    }
}

fn known_digest_size(alg: u16) -> Option<usize> {
    match alg {
        TPM_ALG_SHA1 => Some(20),
        TPM_ALG_SHA256 | TPM_ALG_SM3_256 => Some(32),
        TPM_ALG_SHA384 => Some(48),
        TPM_ALG_SHA512 => Some(64),
        _ => None,
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], String> {
        let end = self.i.checked_add(n).ok_or("length overflow")?;
        if end > self.b.len() {
            return Err(format!(
                "truncated: {what} needs {n} bytes at offset {}",
                self.i
            ));
        }
        let s = &self.b[self.i..end];
        self.i = end;
        Ok(s)
    }
    fn u16(&mut self, what: &str) -> Result<u16, String> {
        let s = self.take(2, what)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }
    fn u32(&mut self, what: &str) -> Result<u32, String> {
        let s = self.take(4, what)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn remaining(&self) -> usize {
        self.b.len() - self.i
    }
}

/// Parse a CCEL area. Stops at the first all-zero event header (padding).
pub fn parse(log: &[u8]) -> Result<Vec<Event>, String> {
    parse_with_end(log).map(|(events, _)| events)
}

/// `parse`, and the offset where the events end: the padding, or fewer than
/// 12 bytes, starts there.
pub fn parse_with_end(log: &[u8]) -> Result<(Vec<Event>, usize), String> {
    let mut c = Cursor { b: log, i: 0 };
    // Spec ID header, legacy shape.
    let hdr_mr = c.u32("header mr index")?;
    let hdr_type = c.u32("header event type")?;
    let _sha1 = c.take(20, "header digest")?;
    let hdr_len = c.u32("header size")? as usize;
    if hdr_len > MAX_EVENT_DATA {
        return Err(format!("Spec ID header claims {hdr_len} bytes"));
    }
    let hdr = c.take(hdr_len, "header data")?;
    // TDX logs the header at index 1, MR 0 being MRTD (Linux tpm_eventlog.h, 2025-07
    // fix). Index 0 only refused the first GCE TDX log, 2026-10-14 (RUN_2026-10-14_beta2.md).
    if hdr_mr > 1 || hdr_type != EV_NO_ACTION || hdr.len() < 28 || &hdr[..16] != SPEC_ID_SIGNATURE {
        return Err("log does not start with the TCG Spec ID Event03 header".into());
    }
    let mut sizes: Vec<(u16, usize)> = Vec::new();
    {
        let mut h = Cursor { b: hdr, i: 24 };
        let n = h.u32("numberOfAlgorithms")?;
        if n == 0 || n > 16 {
            return Err(format!("Spec ID header lists {n} algorithms"));
        }
        for _ in 0..n {
            let alg = h.u16("algorithmId")?;
            let size = h.u16("digestSize")? as usize;
            if let Some(known) = known_digest_size(alg) {
                if known != size {
                    return Err(format!(
                        "algorithm {alg:#06x} declared with {size} bytes, expected {known}"
                    ));
                }
            }
            sizes.push((alg, size));
        }
    }
    let mut events = Vec::new();
    let mut end = c.i;
    while c.remaining() >= 12 {
        let n = events.len();
        let mr_index = c.u32("mr index")?;
        let event_type = c.u32("event type")?;
        let count = c.u32("digest count")?;
        if mr_index == 0 && event_type == 0 && count == 0 {
            break; // zero padding after the last event
        }
        // GCE TDX pads the area with 0xFF (measured 2026-10-14, tests/ccel/).
        if mr_index == u32::MAX && event_type == u32::MAX && count == u32::MAX {
            break;
        }
        if count > MAX_DIGESTS_PER_EVENT {
            return Err(format!("event {n} declares {count} digests"));
        }
        let mut digests = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let alg = c.u16("algorithm")?;
            let size = sizes
                .iter()
                .find(|(a, _)| *a == alg)
                .map(|(_, s)| *s)
                .or_else(|| known_digest_size(alg))
                .ok_or_else(|| {
                    format!("event {n}: algorithm {alg:#06x} not declared in the header")
                })?;
            digests.push((alg, c.take(size, "digest")?.to_vec()));
        }
        let len = c.u32("event size")? as usize;
        if len > MAX_EVENT_DATA {
            return Err(format!("event {n} claims {len} bytes of data"));
        }
        let data = c.take(len, "event data")?.to_vec();
        events.push(Event {
            mr_index,
            event_type,
            digests,
            data,
        });
        end = c.i;
    }
    Ok((events, end))
}

/// Replay the log into RTMR0..3. Every extending event needs a SHA-384.
pub fn replay(events: &[Event]) -> Result<[[u8; 48]; 4], String> {
    let mut rtmr = [[0u8; 48]; 4];
    for (n, ev) in events.iter().enumerate() {
        if ev.event_type == EV_NO_ACTION {
            continue;
        }
        let Some(i) = ev.rtmr() else {
            return Err(format!(
                "event {n} targets MR index {}, which is not an RTMR",
                ev.mr_index
            ));
        };
        let digest = ev
            .sha384()
            .ok_or_else(|| format!("event {n} carries no SHA-384 digest"))?;
        let mut h = Sha384::new();
        h.update(rtmr[i]);
        h.update(digest);
        rtmr[i] = h.finalize().into();
    }
    Ok(rtmr)
}

/// Encode events as a CCEL area (Spec ID header, SHA-384 declared, then
/// the events). The shape [`parse`] reads; used by the fake TD and by
/// image tooling that replays a swtpm log into the reference.
pub fn encode(events: &[Event]) -> Vec<u8> {
    let mut algs: Vec<(u16, usize)> = vec![(TPM_ALG_SHA384, 48)];
    for ev in events {
        for (alg, d) in &ev.digests {
            if !algs.iter().any(|(a, _)| a == alg) {
                algs.push((*alg, d.len()));
            }
        }
    }
    let mut hdr = Vec::new();
    hdr.extend_from_slice(SPEC_ID_SIGNATURE);
    hdr.extend_from_slice(&0u32.to_le_bytes()); // platformClass
    hdr.extend_from_slice(&[0, 2, 0, 2]); // minor, major, errata, uintnSize
    hdr.extend_from_slice(&(algs.len() as u32).to_le_bytes());
    for (alg, size) in &algs {
        hdr.extend_from_slice(&alg.to_le_bytes());
        hdr.extend_from_slice(&(*size as u16).to_le_bytes());
    }
    hdr.push(0); // vendorInfoSize
    let mut out = Vec::new();
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&EV_NO_ACTION.to_le_bytes());
    out.extend_from_slice(&[0u8; 20]);
    out.extend_from_slice(&(hdr.len() as u32).to_le_bytes());
    out.extend_from_slice(&hdr);
    for ev in events {
        out.extend_from_slice(&ev.mr_index.to_le_bytes());
        out.extend_from_slice(&ev.event_type.to_le_bytes());
        out.extend_from_slice(&(ev.digests.len() as u32).to_le_bytes());
        for (alg, d) in &ev.digests {
            out.extend_from_slice(&alg.to_le_bytes());
            out.extend_from_slice(d);
        }
        out.extend_from_slice(&(ev.data.len() as u32).to_le_bytes());
        out.extend_from_slice(&ev.data);
    }
    out
}

/// The running kernel's CCEL. Absent outside a TD.
// Not in x0ne-sdk (its build.rs sets `x0ne_sdk`): the TD reads it, the caller
// is handed it.
#[cfg(not(x0ne_sdk))]
pub fn read_system() -> std::io::Result<Vec<u8>> {
    crate::fsread::read_capped(std::path::Path::new(CCEL_DATA_PATH), MAX_LOG_BYTES)
}

/// UTF-16LE with a terminating NUL, as systemd-stub logs section names.
pub fn utf16le_nul(s: &str) -> Vec<u8> {
    let mut out: Vec<u8> = s.encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
    out.extend_from_slice(&[0, 0]);
    out
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-384 of event data: what a content-measured event's digest must be.
pub fn sha384(data: &[u8]) -> [u8; 48] {
    Sha384::digest(data).into()
}

// Not when compiled into x0ne-sdk (its build.rs sets `x0ne_sdk`): these tests
// need this crate's fixtures and dev-dependencies.
#[cfg(all(test, not(x0ne_sdk)))]
mod tests {
    use super::*;

    fn sample() -> Vec<Event> {
        vec![
            Event::new_sha384(1, EV_SEPARATOR, [0x01; 48], vec![0, 0, 0, 0]),
            Event::new_sha384(
                2,
                EV_EFI_BOOT_SERVICES_APPLICATION,
                [0x02; 48],
                b"BOOTX64.EFI".to_vec(),
            ),
            Event::new_sha384(3, EV_IPL, [0x03; 48], utf16le_nul(".linux")),
            Event::new_sha384(3, EV_IPL, [0x04; 48], utf16le_nul(".cmdline")),
        ]
    }

    #[test]
    fn encode_then_parse_is_the_identity_and_padding_stops_the_parser() {
        let events = sample();
        let mut bytes = encode(&events);
        assert_eq!(parse(&bytes).unwrap(), events);
        bytes.extend_from_slice(&[0u8; 300]);
        assert_eq!(parse(&bytes).unwrap(), events, "zero padding");
    }

    #[test]
    fn the_events_end_where_the_padding_starts() {
        let events = sample();
        let bytes = encode(&events);
        for tail in [&[][..], &[0u8; 300], &[0xFF; 256], &[0xFF; 11], &[7; 5]] {
            let mut log = bytes.clone();
            log.extend_from_slice(tail);
            assert_eq!(parse_with_end(&log).unwrap(), (events.clone(), bytes.len()));
        }
    }

    /// The replay is the hardware's extend rule, computed by hand here.
    #[test]
    fn replay_matches_the_extend_rule() {
        let rtmr = replay(&sample()).unwrap();
        let ext = |acc: [u8; 48], d: [u8; 48]| -> [u8; 48] {
            let mut h = Sha384::new();
            h.update(acc);
            h.update(d);
            h.finalize().into()
        };
        assert_eq!(rtmr[0], ext([0; 48], [0x01; 48]));
        assert_eq!(rtmr[1], ext([0; 48], [0x02; 48]));
        assert_eq!(rtmr[2], ext(ext([0; 48], [0x03; 48]), [0x04; 48]));
        assert_eq!(rtmr[3], [0; 48]);
    }

    #[test]
    fn no_action_events_do_not_extend_and_mrtd_events_are_refused() {
        let mut events = sample();
        events.push(Event::new_sha384(3, EV_NO_ACTION, [0xff; 48], vec![]));
        assert_eq!(replay(&events).unwrap(), replay(&sample()).unwrap());
        events.push(Event::new_sha384(0, EV_IPL, [0xff; 48], vec![]));
        assert!(replay(&events).unwrap_err().contains("not an RTMR"));
        let no_sha384 = vec![Event {
            mr_index: 2,
            event_type: EV_IPL,
            digests: vec![(TPM_ALG_SHA256, vec![0; 32])],
            data: vec![],
        }];
        assert!(replay(&no_sha384).unwrap_err().contains("SHA-384"));
    }

    /// The CCEL of GCE a3-highgpu-1g TDX attempt 3 of RUN_2026-10-14_beta2.md, a
    /// test image (x0ne-OS keeps attempt 9 with its quote: other GPT, UKI and
    /// .cmdline), read from the serial console, cut 256 bytes into the 0xFF padding.
    #[test]
    fn the_first_real_gce_tdx_log_parses() {
        let log = include_bytes!("../tests/ccel/gce-a3-tdx-2026-10-14.bin");
        let (events, end) = parse_with_end(log).expect("real GCE log");
        assert_eq!(log[end..], [0xFF; 256]);
        assert_eq!(events.len(), 39);
        assert!(replay(&events).is_ok());
        let ipl = events.iter().filter(|e| e.event_type == EV_IPL).count();
        assert_eq!(
            ipl, 12,
            "systemd-stub: one name event and one content event per section"
        );
    }

    #[test]
    fn a_header_at_index_1_is_the_tdx_shape_and_parses() {
        let ev = Event::new_sha384(2, EV_EFI_ACTION, [7u8; 48], b"x".to_vec());
        let mut log = encode(std::slice::from_ref(&ev));
        log[..4].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(parse(&log).expect("index 1 header"), vec![ev]);
        log[..4].copy_from_slice(&2u32.to_le_bytes());
        assert!(parse(&log).unwrap_err().contains("Spec ID"));
    }

    #[test]
    fn truncation_and_a_foreign_header_are_refused() {
        let bytes = encode(&sample());
        for cut in [10, 40, bytes.len() - 5] {
            assert!(parse(&bytes[..cut]).is_err(), "cut at {cut}");
        }
        let mut other = bytes.clone();
        other[8 + 20 + 4] = b'X';
        assert!(parse(&other).unwrap_err().contains("Spec ID"));
    }

    #[test]
    fn names_decode_from_utf16le_and_utf8() {
        assert_eq!(sample()[2].name().as_deref(), Some(".linux"));
        assert_eq!(sample()[1].name().as_deref(), Some("BOOTX64.EFI"));
        assert_eq!(sample()[0].name(), None);
        let utf8_nul = Event::new_sha384(3, EV_IPL, [0; 48], b".initrd\0".to_vec());
        assert_eq!(utf8_nul.name().as_deref(), Some(".initrd"));
    }
}
