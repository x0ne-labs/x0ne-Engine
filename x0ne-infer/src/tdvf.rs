// SPDX-License-Identifier: Apache-2.0
//! Google's TDX virtual firmware, read from the binary Google publishes: the
//! TDVF metadata, the MRTD the binary measures, and the SHA-384 of its
//! configuration firmware volume (CFV), which the firmware extends into RTMR0.
//!
//! The binary is object `ovmf_x64_csm/<SHA-384 of the binary>.fd` of the
//! public bucket `gce_tcb_integrity`, and that SHA-384 is in the golden
//! measurement Google signs (GCP, "Verify Confidential VM firmware", read
//! 2026-10-22). Nothing here trusts the bucket or its TLS: a binary is used
//! only when its SHA-384 is the signed one.
//!
//! Layout and measurement as google/gce-tcb-verifier reads them
//! (`ovmf/fw_guid_table_ops.go`, `ovmf/tdx_data.go`, `tdx/measurement.go`,
//! read 2026-10-22): the GUIDed table that ends 0x20 bytes before the end of
//! the image, its TDX metadata offset block, the `TDVF` descriptor and its
//! sections; MRTD = SHA-384 over a `MEM.PAGE.ADD` buffer for every 4 KiB page
//! of every section, and a `MR.EXTEND` buffer plus the data for every
//! 256-byte chunk of a section marked EXTEND_MR. Google's older "TD HOB bug"
//! measurements (every section extended, RAM banks in the HOB) are not
//! reproduced: a binary whose MRTD computed this way is not the quote's is
//! refused.
//!
//! Measured on 2026-10-22, on the binary 53c558fe… that Google endorses for
//! MRTD c1ee9c16… (the firmware of the GCE `a3-highgpu-1g` boots of
//! 2026-10-14, -17 and -18): the MRTD computed here is their quotes', and the
//! CFV digest is the `EV_EFI_PLATFORM_FIRMWARE_BLOB2` event of RTMR0 in their
//! logs (`tests/ccel/`, `tests/gce_firmware_live.rs`).

use crate::gce_endorsement::Endorsement;
use sha2::{Digest as _, Sha384};

/// Directory of the binaries in Google's bucket.
pub const FIRMWARE_PREFIX: &str = "ovmf_x64_csm";
/// Google's binaries are 2 MiB (GCP docs); the cap only bounds a hostile answer.
pub const MAX_FIRMWARE_BYTES: usize = 8 * 1024 * 1024;
/// 2 MiB from one Google region (us-west1, GCP docs): a minute is ample.
const FETCH_TIMEOUT_SECS: u64 = 60;
/// The section attribute that puts a section's bytes into MRTD.
pub const ATTRIBUTE_EXTEND_MR: u32 = 1;
/// Memory one section may ask to be measured: Google's largest is 1.9 MiB.
const MAX_SECTION_MEMORY: u64 = 64 * 1024 * 1024;
const MAX_SECTIONS: u32 = 64;
const PAGE: u64 = 4096;
const CHUNK: usize = 256;
/// The GUIDed table ends this many bytes before the end of the image.
const TABLE_END_FROM_END: usize = 0x20;
/// A table entry ends with its u16 size and its GUID.
const ENTRY_TRAILER: usize = 2 + 16;
const DESCRIPTOR_HEADER: usize = 16;
const SECTION_BYTES: usize = 32;

/// EFI GUIDs as stored: the first three fields little-endian.
const FOOTER_GUID: [u8; 16] = efi_guid(
    0x96b5_82de,
    0x1fb2,
    0x45f7,
    [0xba, 0xea, 0xa3, 0x66, 0xc5, 0x5a, 0x08, 0x2d],
);
const METADATA_OFFSET_GUID: [u8; 16] = efi_guid(
    0xe47a_6535,
    0x984a,
    0x4798,
    [0x86, 0x5e, 0x46, 0x85, 0xa7, 0xbf, 0x8e, 0xc2],
);
const METADATA_GUID: [u8; 16] = efi_guid(
    0xe9ea_f9f3,
    0x168e,
    0x44d5,
    [0xa8, 0xeb, 0x7f, 0x4d, 0x87, 0x38, 0xf6, 0xae],
);
const TDVF_SIGNATURE: &[u8; 4] = b"TDVF";

const fn efi_guid(a: u32, b: u16, c: u16, d: [u8; 8]) -> [u8; 16] {
    let a = a.to_le_bytes();
    let b = b.to_le_bytes();
    let c = c.to_le_bytes();
    [
        a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], d[0], d[1], d[2], d[3], d[4], d[5], d[6],
        d[7],
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    /// Boot firmware volume: the code.
    Bfv,
    /// Configuration firmware volume: measured into RTMR0 at boot, not MRTD.
    Cfv,
    TdHob,
    TempMem,
}

impl SectionKind {
    fn of(code: u32) -> Result<Self, String> {
        Ok(match code {
            0 => Self::Bfv,
            1 => Self::Cfv,
            2 => Self::TdHob,
            3 => Self::TempMem,
            other => return Err(format!("section type {other} is not one TDVF defines")),
        })
    }

    fn is_volume(self) -> bool {
        matches!(self, Self::Bfv | Self::Cfv)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub kind: SectionKind,
    pub data_offset: u32,
    pub raw_size: u32,
    pub memory_base: u64,
    pub memory_size: u64,
    pub attributes: u32,
}

impl Section {
    fn data<'a>(&self, image: &'a [u8]) -> &'a [u8] {
        let start = self.data_offset as usize;
        &image[start..start + self.raw_size as usize]
    }
}

/// A TDVF image and its validated sections.
pub struct Image<'a> {
    bytes: &'a [u8],
    pub sections: Vec<Section>,
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"))
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

/// The payload of the GUIDed table entry `guid`, or `None`.
fn table_entry<'a>(image: &'a [u8], guid: &[u8; 16]) -> Result<Option<&'a [u8]>, String> {
    if image.len() < TABLE_END_FROM_END + ENTRY_TRAILER {
        return Err(format!(
            "image of {} bytes is too small for a TDVF",
            image.len()
        ));
    }
    let end = image.len() - TABLE_END_FROM_END;
    if image[end - 16..end] != FOOTER_GUID {
        return Err("no GUIDed table footer 0x20 bytes before the end of the image".into());
    }
    let table_size = u16_at(image, end - ENTRY_TRAILER) as usize;
    if table_size < ENTRY_TRAILER || table_size > end {
        return Err(format!(
            "GUIDed table footer gives a table of {table_size} bytes"
        ));
    }
    let table = &image[end - table_size..end - ENTRY_TRAILER];
    let mut n = table.len();
    while n > 0 {
        if n < ENTRY_TRAILER {
            return Err("GUIDed table entry truncated".into());
        }
        let size = u16_at(table, n - ENTRY_TRAILER) as usize;
        if size < ENTRY_TRAILER || size > n {
            return Err(format!("GUIDed table entry of {size} bytes"));
        }
        if table[n - 16..n] == *guid {
            return Ok(Some(&table[n - size..n - ENTRY_TRAILER]));
        }
        n -= size;
    }
    Ok(None)
}

impl<'a> Image<'a> {
    /// The TDVF metadata of `bytes`, refused unless it is a layout the TDX
    /// module can load and this verifier can measure: one CFV, one TD HOB,
    /// at least one BFV, volumes that cover the image exactly, page-aligned
    /// memory that no two sections share, EXTEND_MR on volumes only.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        let block = table_entry(bytes, &METADATA_OFFSET_GUID)?
            .ok_or("the GUIDed table has no TDX metadata offset entry")?;
        if block.len() < 4 {
            return Err("TDX metadata offset entry is shorter than 4 bytes".into());
        }
        let offset = u32_at(block, 0) as usize;
        let len = bytes.len();
        let desc = len
            .checked_sub(offset)
            .filter(|d| *d >= 16 && d.checked_add(DESCRIPTOR_HEADER).is_some_and(|e| e <= len));
        let Some(desc) = desc else {
            return Err(format!(
                "TDX metadata offset {offset:#x} puts the descriptor outside the image"
            ));
        };
        if bytes[desc - 16..desc] != METADATA_GUID {
            return Err("no TDX metadata GUID before the descriptor".into());
        }
        if &bytes[desc..desc + 4] != TDVF_SIGNATURE {
            return Err("TDX metadata descriptor signature is not TDVF".into());
        }
        let length = u32_at(bytes, desc + 4) as usize;
        let version = u32_at(bytes, desc + 8);
        let count = u32_at(bytes, desc + 12);
        if version != 1 {
            return Err(format!(
                "TDX metadata version {version}: this verifier reads 1"
            ));
        }
        if count == 0 || count > MAX_SECTIONS {
            return Err(format!("TDX metadata lists {count} sections"));
        }
        if length != DESCRIPTOR_HEADER + SECTION_BYTES * count as usize {
            return Err(format!(
                "TDX metadata length {length} is not that of {count} sections"
            ));
        }
        if desc + length > len {
            return Err("TDX metadata sections run past the image".into());
        }
        let mut sections = Vec::with_capacity(count as usize);
        for k in 0..count as usize {
            let at = desc + DESCRIPTOR_HEADER + SECTION_BYTES * k;
            sections.push(Section {
                data_offset: u32_at(bytes, at),
                raw_size: u32_at(bytes, at + 4),
                memory_base: u64_at(bytes, at + 8),
                memory_size: u64_at(bytes, at + 16),
                kind: SectionKind::of(u32_at(bytes, at + 24))
                    .map_err(|e| format!("section {k}: {e}"))?,
                attributes: u32_at(bytes, at + 28),
            });
        }
        validate(&sections, len)?;
        Ok(Self { bytes, sections })
    }

    /// The MRTD the TDX module computes when the VMM loads this image:
    /// TDH.MEM.PAGE.ADD of every page of every section, in metadata order,
    /// and TDH.MR.EXTEND of every 256-byte chunk of a section marked so.
    pub fn mrtd(&self) -> [u8; 48] {
        let mut h = Sha384::new();
        for s in &self.sections {
            let data = s.kind.is_volume().then(|| s.data(self.bytes));
            let mut offset = 0u64;
            while offset < s.memory_size {
                let gpa = s.memory_base + offset;
                if offset.is_multiple_of(PAGE) {
                    h.update(tdcall_buffer(b"MEM.PAGE.ADD", gpa));
                }
                if s.attributes & ATTRIBUTE_EXTEND_MR != 0 {
                    // `validate`: only volumes, whose data is their memory.
                    let data = data.expect("EXTEND_MR on a volume");
                    h.update(tdcall_buffer(b"MR.EXTEND", gpa));
                    h.update(&data[offset as usize..offset as usize + CHUNK]);
                }
                offset += CHUNK as u64;
            }
        }
        h.finalize().into()
    }

    /// The one configuration firmware volume.
    pub fn cfv(&self) -> &Section {
        self.sections
            .iter()
            .find(|s| s.kind == SectionKind::Cfv)
            .expect("validate: exactly one CFV")
    }

    /// SHA-384 of the CFV's bytes: the digest of its RTMR0 event.
    pub fn cfv_sha384(&self) -> [u8; 48] {
        Sha384::digest(self.cfv().data(self.bytes)).into()
    }
}

/// The 128-byte buffer the TDX module hashes for a page add or an extend.
fn tdcall_buffer(op: &[u8], gpa: u64) -> [u8; 128] {
    let mut b = [0u8; 128];
    b[..op.len()].copy_from_slice(op);
    b[16..24].copy_from_slice(&gpa.to_le_bytes());
    b
}

fn validate(sections: &[Section], image_len: usize) -> Result<(), String> {
    let count = |k: SectionKind| sections.iter().filter(|s| s.kind == k).count();
    if count(SectionKind::Bfv) == 0 {
        return Err("TDX metadata has no BFV".into());
    }
    if count(SectionKind::Cfv) != 1 {
        return Err(format!(
            "TDX metadata has {} CFV(s), exactly one CFV is measured into RTMR0",
            count(SectionKind::Cfv)
        ));
    }
    if count(SectionKind::TdHob) != 1 {
        return Err(format!(
            "TDX metadata has {} TD HOB(s), exactly one TD HOB is expected",
            count(SectionKind::TdHob)
        ));
    }
    let mut volumes = 0u64;
    for (k, s) in sections.iter().enumerate() {
        if s.memory_size > MAX_SECTION_MEMORY {
            return Err(format!(
                "section {k}: memory size {:#x} exceeds the {MAX_SECTION_MEMORY:#x} this verifier measures",
                s.memory_size
            ));
        }
        if !s.memory_base.is_multiple_of(PAGE)
            || !s.memory_size.is_multiple_of(PAGE)
            || s.memory_size == 0
        {
            return Err(format!(
                "section {k}: memory {:#x}+{:#x} is not page aligned",
                s.memory_base, s.memory_size
            ));
        }
        if s.memory_base.checked_add(s.memory_size).is_none() {
            return Err(format!("section {k}: memory wraps around"));
        }
        if s.attributes & !ATTRIBUTE_EXTEND_MR != 0 {
            return Err(format!(
                "section {k}: unknown attributes {:#x}",
                s.attributes
            ));
        }
        if s.kind.is_volume() {
            if u64::from(s.raw_size) != s.memory_size {
                return Err(format!(
                    "section {k}: volume of {:#x} bytes in a memory size of {:#x}",
                    s.raw_size, s.memory_size
                ));
            }
            // In u64: two u32 from the file add up past a 32-bit usize
            // (wasm32, through x0ne-sdk), where the sum would wrap under
            // the image's length.
            let end = u64::from(s.data_offset) + u64::from(s.raw_size);
            if end > image_len as u64 {
                return Err(format!(
                    "section {k}: volume {:#x}+{:#x} runs past the image",
                    s.data_offset, s.raw_size
                ));
            }
            volumes += u64::from(s.raw_size);
        } else if s.attributes & ATTRIBUTE_EXTEND_MR != 0 {
            return Err(format!(
                "section {k}: EXTEND_MR on a {:?}, which has no bytes in the image",
                s.kind
            ));
        }
    }
    if volumes != image_len as u64 {
        return Err(format!(
            "the volumes ({volumes:#x} bytes) do not add up to the image ({image_len:#x})"
        ));
    }
    for (i, a) in sections.iter().enumerate() {
        for (j, b) in sections.iter().enumerate().skip(i + 1) {
            if a.memory_base < b.memory_base + b.memory_size
                && b.memory_base < a.memory_base + a.memory_size
            {
                return Err(format!("section {j}'s memory overlaps section {i}'s"));
            }
            if a.kind.is_volume()
                && b.kind.is_volume()
                && u64::from(a.data_offset) < u64::from(b.data_offset) + u64::from(b.raw_size)
                && u64::from(b.data_offset) < u64::from(a.data_offset) + u64::from(a.raw_size)
            {
                return Err(format!("volume {j} overlaps volume {i} in the image"));
            }
        }
    }
    Ok(())
}

/// The firmware binary Google endorsed, as this client measured it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndorsedFirmware {
    /// SHA-384 of the binary: the one Google signed.
    pub sha384_hex: String,
    /// The MRTD this binary measures, computed here: the quote's.
    pub mrtd: [u8; 48],
    /// SHA-384 of its CFV, which the firmware extends into RTMR0.
    pub cfv_sha384: [u8; 48],
    pub tdx_svn: u32,
}

/// The binary `endorsement` names: its SHA-384 must be the one Google signed,
/// and the MRTD it measures must be the quote's and one Google lists.
pub fn verify_endorsed(
    firmware: &[u8],
    endorsement: &Endorsement,
    quote_mrtd: &[u8; 48],
) -> Result<EndorsedFirmware, String> {
    let signed = &endorsement.uefi_sha384_hex;
    if signed.is_empty() {
        return Err("the endorsement names no firmware binary (no UEFI digest)".into());
    }
    let sha384_hex = crate::ccel::hex(&Sha384::digest(firmware));
    if sha384_hex != *signed {
        return Err(format!(
            "binary of SHA-384 {sha384_hex} is not the firmware Google endorsed ({signed})"
        ));
    }
    let image = Image::parse(firmware).map_err(|e| format!("endorsed binary {signed}: {e}"))?;
    if endorsement.tdx_measurement(quote_mrtd).is_none() {
        return Err(format!(
            "the endorsement does not list the quoted MRTD {}",
            crate::ccel::hex(quote_mrtd)
        ));
    }
    let mrtd = image.mrtd();
    if mrtd != *quote_mrtd {
        return Err(format!(
            "the endorsed binary measures MRTD {}, the quote's is {}",
            crate::ccel::hex(&mrtd),
            crate::ccel::hex(quote_mrtd)
        ));
    }
    Ok(EndorsedFirmware {
        sha384_hex,
        mrtd,
        cfv_sha384: image.cfv_sha384(),
        tdx_svn: endorsement.tdx_svn,
    })
}

/// `ovmf_x64_csm/<digest>.fd`; the digest must be 96 lowercase hex.
pub fn object_name(sha384_hex: &str) -> Result<String, String> {
    let ok = sha384_hex.len() == 96
        && sha384_hex
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c));
    if !ok {
        return Err(format!(
            "firmware digest {sha384_hex:?} is not a SHA-384 in lowercase hex"
        ));
    }
    Ok(format!("{FIRMWARE_PREFIX}/{sha384_hex}.fd"))
}

/// `Ok(None)` when the bucket has no such binary (404).
// Not in x0ne-sdk (its build.rs sets `x0ne_sdk`): it opens no socket, the
// caller relays these bytes.
#[cfg(not(x0ne_sdk))]
pub async fn fetch(
    http: &reqwest::Client,
    base_url: &str,
    sha384_hex: &str,
) -> Result<Option<Vec<u8>>, String> {
    let url = format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        object_name(sha384_hex)?
    );
    let resp = http
        .get(&url)
        .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_SECS))
        .send()
        .await
        .map_err(|e| format!("GET {url}: {e}"))?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !resp.status().is_success() {
        return Err(format!("GET {url}: {}", resp.status()));
    }
    if resp
        .content_length()
        .is_some_and(|n| n > MAX_FIRMWARE_BYTES as u64)
    {
        return Err(format!(
            "GET {url}: firmware of {} bytes, cap {MAX_FIRMWARE_BYTES}",
            resp.content_length().unwrap_or(0)
        ));
    }
    let mut resp = resp;
    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(|e| format!("GET {url}: {e}"))? {
        if body.len() + chunk.len() > MAX_FIRMWARE_BYTES {
            return Err(format!(
                "GET {url}: firmware over the cap of {MAX_FIRMWARE_BYTES} bytes"
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Some(body))
}

// Not when compiled into x0ne-sdk (its build.rs sets `x0ne_sdk`): these tests
// need this crate's fixtures and dev-dependencies.
#[cfg(all(test, not(x0ne_sdk)))]
pub(crate) mod tests {
    use super::*;
    use crate::ccel::hex;

    /// 16 KiB in Google's TDVF layout, the bytes `tests/tdvf/tdvf_reference.py
    /// make` writes: a CFV page, three BFV pages marked EXTEND_MR, a TD HOB and
    /// a temporary memory section. Built here: the tree holds no image file.
    pub(crate) fn synthetic() -> &'static [u8] {
        static IMAGE: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
        IMAGE.get_or_init(|| {
            let n = 0x4000;
            let mut b: Vec<u8> = (0..n).map(|i| (i * 7 + 3) as u8).collect();
            let cfv = b"x0ne synthetic CFV for tests; not a firmware. ";
            b[..cfv.len()].copy_from_slice(cfv);
            // Data offset, raw size, memory base, memory size, kind, attributes.
            let sections: [(u32, u32, u64, u64, u32, u32); 4] = [
                (0x1000, 0x3000, 0xFFFF_D000, 0x3000, 0, ATTRIBUTE_EXTEND_MR),
                (0, 0x1000, 0xFFFF_C000, 0x1000, 1, 0),
                (0, 0, 0x80_9000, 0x2000, 2, 0),
                (0, 0, 0x80_0000, 0x6000, 3, 0),
            ];
            let d = DESCRIPTOR_AT;
            b[d - 16..d].copy_from_slice(&METADATA_GUID);
            b[d..d + 4].copy_from_slice(TDVF_SIGNATURE);
            let length = DESCRIPTOR_HEADER + SECTION_BYTES * sections.len();
            put_u32(&mut b, d + 4, length as u32);
            put_u32(&mut b, d + 8, 1);
            put_u32(&mut b, d + 12, sections.len() as u32);
            for (k, &(offset, raw, base, size, kind, attributes)) in sections.iter().enumerate() {
                let at = section_at(k);
                put_u32(&mut b, at, offset);
                put_u32(&mut b, at + 4, raw);
                put_u64(&mut b, at + 8, base);
                put_u64(&mut b, at + 16, size);
                put_u32(&mut b, at + 24, kind);
                put_u32(&mut b, at + 28, attributes);
            }
            // The GUIDed table: the metadata offset entry, then the footer.
            let entry = 4 + ENTRY_TRAILER;
            let at = OFFSET_BLOCK_AT;
            put_u32(&mut b, at, (n - d) as u32);
            b[at + 4..at + 6].copy_from_slice(&(entry as u16).to_le_bytes());
            b[at + 6..at + entry].copy_from_slice(&METADATA_OFFSET_GUID);
            let end = n - TABLE_END_FROM_END;
            assert_eq!(at + entry, end - ENTRY_TRAILER);
            b[end - 18..end - 16].copy_from_slice(&((ENTRY_TRAILER + entry) as u16).to_le_bytes());
            b[end - 16..end].copy_from_slice(&FOOTER_GUID);
            b
        })
    }
    const SYNTHETIC_SHA384: &str = "020d482a2f439524cec711da8343612056d6caff81e3fd294ddb104089a6e090110babcbfe64c3a8d707958a52283075";
    /// `tdvf_reference.py measure`, the implementation that gives Google's
    /// MRTD c1ee9c16… and RTMR0's CFV digest 4e1221cf… from Google's binary.
    pub(crate) const SYNTHETIC_MRTD: &str = "af705509c4d9fc09fd61469f2e60f58b111cea5ff6c45f01578e6cf959fe30c0d159b492c2e19167f55e0f30f8107b2e";
    pub(crate) const SYNTHETIC_CFV: &str = "4ca3cca6c6125fd8647942fd77325adfe3c87e29b14a4f2318057fd28c7313ee7e26eadfce9d6931bd7b8e04347e5cb2";
    /// Where the descriptor and its sections lie in the synthetic image.
    const DESCRIPTOR_AT: usize = 0x2F10;
    const OFFSET_BLOCK_AT: usize = 0x3FB8;

    pub(crate) fn unhex48(s: &str) -> [u8; 48] {
        crate::cloud_config::unhex::<48>(s).unwrap()
    }

    pub(crate) fn synthetic_mrtd() -> [u8; 48] {
        unhex48(SYNTHETIC_MRTD)
    }

    pub(crate) fn synthetic_cfv() -> [u8; 48] {
        unhex48(SYNTHETIC_CFV)
    }

    /// An endorsement as `gce_endorsement::verify` returns it, for this binary.
    pub(crate) fn endorsement_for(firmware: &[u8], mrtds: &[[u8; 48]]) -> Endorsement {
        use sha2::Digest as _;
        Endorsement {
            timestamp_unix: 1_789_000_000,
            cl_spec: 1,
            commit_hex: String::new(),
            uefi_sha384_hex: hex(&sha2::Sha384::digest(firmware)),
            tdx_svn: 2,
            tdx_measurements: mrtds
                .iter()
                .map(|m| crate::gce_endorsement::TdxMeasurement {
                    ram_gib: 0,
                    early_accept: false,
                    mrtd: *m,
                })
                .collect(),
            signer_subject: "CN=test".into(),
        }
    }

    fn section_at(k: usize) -> usize {
        DESCRIPTOR_AT + 16 + 32 * k
    }

    fn put_u32(b: &mut [u8], at: usize, v: u32) {
        b[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn put_u64(b: &mut [u8], at: usize, v: u64) {
        b[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }

    /// The image is the one the generator writes, and this parser measures it
    /// as the independent implementation does.
    #[test]
    fn the_synthetic_image_measures_as_the_independent_implementation_says() {
        use sha2::Digest as _;
        assert_eq!(hex(&sha2::Sha384::digest(synthetic())), SYNTHETIC_SHA384);
        let image = Image::parse(synthetic()).expect("a valid TDVF image");
        assert_eq!(hex(&image.mrtd()), SYNTHETIC_MRTD);
        assert_eq!(hex(&image.cfv_sha384()), SYNTHETIC_CFV);
        let kinds: Vec<SectionKind> = image.sections.iter().map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            [
                SectionKind::Bfv,
                SectionKind::Cfv,
                SectionKind::TdHob,
                SectionKind::TempMem
            ]
        );
        let cfv = image.cfv();
        assert_eq!(
            (
                cfv.data_offset,
                cfv.raw_size,
                cfv.memory_base,
                cfv.attributes
            ),
            (0, 0x1000, 0xFFFF_C000, 0)
        );
    }

    /// SECURITY: a layout the TDVF reader would not accept is refused, each
    /// for its own reason, before anything is measured.
    #[test]
    fn a_broken_layout_is_refused() {
        type Edit = Box<dyn Fn(&mut Vec<u8>)>;
        let cases: Vec<(&str, Edit, &str)> = vec![
            ("too small", Box::new(|b| b.truncate(0x20)), "too small"),
            (
                "no footer",
                Box::new(|b| {
                    let at = b.len() - 0x21;
                    b[at] ^= 1
                }),
                "footer",
            ),
            (
                "offset past the image",
                Box::new(|b| put_u32(b, OFFSET_BLOCK_AT, 0x10_0000)),
                "outside the image",
            ),
            (
                "offset inside the descriptor",
                Box::new(|b| put_u32(b, OFFSET_BLOCK_AT, 8)),
                "outside the image",
            ),
            (
                "metadata GUID",
                Box::new(|b| b[DESCRIPTOR_AT - 1] ^= 1),
                "metadata GUID",
            ),
            (
                "signature",
                Box::new(|b| b[DESCRIPTOR_AT + 3] = b'X'),
                "signature",
            ),
            (
                "version",
                Box::new(|b| put_u32(b, DESCRIPTOR_AT + 8, 2)),
                "version",
            ),
            (
                "length",
                Box::new(|b| put_u32(b, DESCRIPTOR_AT + 4, 16 + 32 * 5)),
                "length",
            ),
            (
                // 0xFFFFF000 + 0x3000 wraps to 0x2000 in 32 bits, under the
                // image's length: a 32-bit usize must not take it for an end.
                "a volume whose offset and size wrap in 32 bits",
                Box::new(|b| put_u32(b, section_at(0), 0xFFFF_F000)),
                "runs past the image",
            ),
            (
                // The CFV at 0xFFFFE000 + 0x1000 ends within 32 bits; its
                // overlap with the BFV is then compared in 64 bits.
                "a volume at the top of 32 bits",
                Box::new(|b| put_u32(b, section_at(1), 0xFFFF_E000)),
                "runs past the image",
            ),
            (
                "unknown section type",
                Box::new(|b| put_u32(b, section_at(3) + 24, 7)),
                "section type 7",
            ),
            (
                "no CFV",
                Box::new(|b| put_u32(b, section_at(1) + 24, 0)),
                "exactly one CFV",
            ),
            (
                "no TD HOB",
                Box::new(|b| put_u32(b, section_at(2) + 24, 3)),
                "exactly one TD HOB",
            ),
            (
                "CFV memory size",
                Box::new(|b| put_u64(b, section_at(1) + 16, 0x2000)),
                "memory size",
            ),
            (
                "volumes do not cover the image",
                Box::new(|b| {
                    put_u32(b, section_at(0) + 4, 0x2000);
                    put_u64(b, section_at(0) + 16, 0x2000)
                }),
                "do not add up",
            ),
            (
                "volume past the image",
                Box::new(|b| put_u32(b, section_at(0), 0x2000)),
                "past the image",
            ),
            (
                "overlap",
                Box::new(|b| put_u64(b, section_at(3) + 8, 0x809000)),
                "overlaps",
            ),
            (
                "unaligned",
                Box::new(|b| put_u64(b, section_at(3) + 8, 0x800010)),
                "aligned",
            ),
            (
                "EXTEND_MR on the TD HOB",
                Box::new(|b| put_u32(b, section_at(2) + 28, ATTRIBUTE_EXTEND_MR)),
                "EXTEND_MR",
            ),
            (
                "memory too large to measure",
                Box::new(|b| put_u64(b, section_at(3) + 16, 1 << 40)),
                "memory",
            ),
        ];
        for (name, edit, want) in cases {
            let mut b = synthetic().to_vec();
            edit(&mut b);
            let err = Image::parse(&b)
                .err()
                .unwrap_or_else(|| panic!("{name}: accepted"));
            assert!(err.contains(want), "{name}: {err}");
        }
    }

    #[test]
    fn the_endorsed_binary_gives_its_mrtd_and_cfv() {
        let mrtd = synthetic_mrtd();
        let e = endorsement_for(synthetic(), &[mrtd]);
        let fw = verify_endorsed(synthetic(), &e, &mrtd).expect("endorsed");
        assert_eq!(fw.sha384_hex, SYNTHETIC_SHA384);
        assert_eq!(
            (fw.mrtd, fw.cfv_sha384, fw.tdx_svn),
            (mrtd, synthetic_cfv(), 2)
        );
    }

    /// SECURITY: firmware not endorsed. One byte changed in the binary, in its
    /// code or in its CFV, and its SHA-384 is not the one Google signed.
    #[test]
    fn a_binary_other_than_the_signed_one_is_refused() {
        let mrtd = synthetic_mrtd();
        let e = endorsement_for(synthetic(), &[mrtd]);
        for at in [0x10, 0x2000] {
            let mut other = synthetic().to_vec();
            other[at] ^= 1;
            let err = verify_endorsed(&other, &e, &mrtd).unwrap_err();
            assert!(
                err.contains("not the firmware Google endorsed"),
                "{at:#x}: {err}"
            );
        }
        let mut none = e.clone();
        none.uefi_sha384_hex = String::new();
        let err = verify_endorsed(synthetic(), &none, &mrtd).unwrap_err();
        assert!(err.contains("names no firmware binary"), "{err}");
    }

    /// SECURITY: Google's MRTD is not taken on its word. A quote whose MRTD
    /// Google lists, next to a binary that measures another one, is refused;
    /// so is an endorsement that does not list what its own binary measures.
    #[test]
    fn an_mrtd_the_binary_does_not_measure_is_refused() {
        let mrtd = synthetic_mrtd();
        let listed = [0x66u8; 48];
        let e = endorsement_for(synthetic(), &[listed, mrtd]);
        let err = verify_endorsed(synthetic(), &e, &listed).unwrap_err();
        assert!(err.contains("the endorsed binary measures"), "{err}");
        let e = endorsement_for(synthetic(), &[listed]);
        let err = verify_endorsed(synthetic(), &e, &mrtd).unwrap_err();
        assert!(err.contains("does not list"), "{err}");
    }

    #[test]
    fn object_names_take_a_sha384_only() {
        assert_eq!(
            object_name(SYNTHETIC_SHA384).unwrap(),
            format!("ovmf_x64_csm/{SYNTHETIC_SHA384}.fd")
        );
        for bad in [
            "",
            "../x",
            &SYNTHETIC_SHA384.to_uppercase(),
            &SYNTHETIC_SHA384[..94],
        ] {
            assert!(object_name(bad).is_err(), "{bad:?}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fetch_reads_the_binary_by_its_digest_and_caps_it() {
        let big = "ab".repeat(48);
        let bucket = crate::attest_v2::fixtures::fake_bucket(vec![
            (object_name(SYNTHETIC_SHA384).unwrap(), synthetic().to_vec()),
            (object_name(&big).unwrap(), vec![0; MAX_FIRMWARE_BYTES + 1]),
        ])
        .await;
        let http = reqwest::Client::new();
        let got = fetch(&http, &bucket, SYNTHETIC_SHA384).await.unwrap();
        assert_eq!(got.as_deref(), Some(synthetic()));
        let missing = "cd".repeat(48);
        assert_eq!(fetch(&http, &bucket, &missing).await.unwrap(), None);
        let err = fetch(&http, &bucket, &big).await.unwrap_err();
        assert!(err.contains("cap"), "{err}");
        assert!(fetch(&http, &bucket, "../etc").await.is_err());
    }

    /// SECURITY: the CFV is not in MRTD. A binary changed there measures the
    /// quoted MRTD, and only the signed digest refuses it.
    #[test]
    fn a_change_outside_mrtd_is_still_refused_by_the_digest() {
        let mrtd = synthetic_mrtd();
        let e = endorsement_for(synthetic(), &[mrtd]);
        let mut other = synthetic().to_vec();
        other[0x10] ^= 1; // inside the CFV (data offset 0, not EXTEND_MR)
        let img = Image::parse(&other).unwrap();
        assert_eq!(img.mrtd(), mrtd, "the CFV is outside MRTD");
        assert_ne!(img.cfv_sha384(), synthetic_cfv());
        let err = verify_endorsed(&other, &e, &mrtd).unwrap_err();
        assert!(err.contains("not the firmware Google endorsed"), "{err}");
    }

    /// EXTEND_MR on the CFV puts its bytes into MRTD: the layout parses and
    /// measures another MRTD.
    #[test]
    fn extend_mr_on_the_cfv_changes_the_mrtd() {
        let mut b = synthetic().to_vec();
        put_u32(&mut b, section_at(1) + 28, ATTRIBUTE_EXTEND_MR);
        let img = Image::parse(&b).expect("a volume may be EXTEND_MR");
        assert_ne!(img.mrtd(), synthetic_mrtd());
    }

    /// Two section entries swapped: no byte of a volume moves, the pages are
    /// added in another order, the MRTD differs.
    #[test]
    fn the_metadata_order_is_the_measurement_order() {
        let mut b = synthetic().to_vec();
        let (a, c) = (section_at(2), section_at(3));
        let mut tmp = [0u8; 32];
        tmp.copy_from_slice(&b[a..a + 32]);
        b.copy_within(c..c + 32, a);
        b[c..c + 32].copy_from_slice(&tmp);
        let img = Image::parse(&b).unwrap();
        assert_ne!(img.mrtd(), synthetic_mrtd());
    }
}
