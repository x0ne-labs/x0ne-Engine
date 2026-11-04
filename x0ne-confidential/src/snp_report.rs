//! Structural parsing of an AMD SEV-SNP attestation report.
//!
//! Scope, stated plainly: this reads the fields of `ATTESTATION_REPORT` and
//! refuses a length or a version it does not implement. It verifies NO
//! signature and judges NO policy: that is `x0ne-infer::snp`, which checks
//! the VCEK chain to a pinned AMD root, the report signature, and the guest
//! policy. A parsed report is the collector's claim until then.
//!
//! Layout: AMD "SEV Secure Nested Paging Firmware ABI Specification" (56860),
//! rev. 1.58 (May 2025), Table 23, `ATTESTATION_REPORT`, 0x4A0 bytes,
//! little-endian. The signature (Table 141, ECDSA P-384 with SHA-384) covers
//! bytes 0x000..0x2A0; R and S are 72-byte little-endian fields at 0x2A0 and
//! 0x2E8.
//!
//! Versions read: 2 and 3, of which real reports are vendored in
//! `x0ne-infer/tests/snp/` (v3 adds CPUID family, model and stepping at
//! 0x188..0x18B), and 5, from Table 23 of rev. 1.58 (it adds the launch and
//! current mitigation vectors at 0x1F8 and 0x200) with no real v5 report in
//! hand. Version 4 is refused: neither the specification in hand nor a real
//! report gives its layout, and a parser that names a version it does not
//! implement is the one `tdx_quote` refuses to be.

/// The whole report, signature included.
pub const REPORT_LEN: usize = 0x4A0;
/// The bytes the signature covers: `report[..SIGNED_LEN]`.
pub const SIGNED_LEN: usize = 0x2A0;
/// `REPORT_DATA`, the 64 bytes the guest chose when it asked for the report.
pub const REPORT_DATA_AT: usize = 0x50;
const MEASUREMENT_AT: usize = 0x90;
const HOST_DATA_AT: usize = 0xC0;
const ID_KEY_DIGEST_AT: usize = 0xE0;
const AUTHOR_KEY_DIGEST_AT: usize = 0x110;
const REPORT_ID_AT: usize = 0x140;
const REPORT_ID_MA_AT: usize = 0x160;
const REPORTED_TCB_AT: usize = 0x180;
const CPUID_AT: usize = 0x188;
const CHIP_ID_AT: usize = 0x1A0;
const COMMITTED_TCB_AT: usize = 0x1E0;
const CURRENT_VERSION_AT: usize = 0x1E8;
const COMMITTED_VERSION_AT: usize = 0x1EC;
const LAUNCH_TCB_AT: usize = 0x1F0;
const LAUNCH_MIT_VECTOR_AT: usize = 0x1F8;
const CURRENT_MIT_VECTOR_AT: usize = 0x200;
const SIGNATURE_R_AT: usize = 0x2A0;
const SIGNATURE_S_AT: usize = 0x2E8;
/// Each signature component is a 72-byte field (Table 141); P-384 uses 48.
pub const SIGNATURE_COMPONENT_LEN: usize = 72;

/// `SIGNATURE_ALGO` 1: ECDSA P-384 with SHA-384 (Table 139).
pub const SIGNATURE_ALGO_ECDSA_P384_SHA384: u32 = 1;
/// `SIGNING_KEY` values (`KEY_INFO` bits 4:2).
pub const SIGNING_KEY_VCEK: u32 = 0;
pub const SIGNING_KEY_VLEK: u32 = 1;
pub const SIGNING_KEY_NONE: u32 = 7;
/// `VMPL` of a report the HOST requested (SNP_HV_REPORT_REQ, rev. 1.56+):
/// its `REPORT_DATA` is the host's, not the guest's.
pub const VMPL_HOST_REQUESTED: u32 = 0xFFFF_FFFF;

/// Guest policy bits (Table 9).
pub const POLICY_SMT: u64 = 1 << 16;
pub const POLICY_RESERVED_MUST_BE_ONE: u64 = 1 << 17;
pub const POLICY_MIGRATE_MA: u64 = 1 << 18;
pub const POLICY_DEBUG: u64 = 1 << 19;
pub const POLICY_SINGLE_SOCKET: u64 = 1 << 20;
pub const POLICY_CXL_ALLOW: u64 = 1 << 21;
pub const POLICY_MEM_AES_256_XTS: u64 = 1 << 22;
pub const POLICY_RAPL_DIS: u64 = 1 << 23;
pub const POLICY_CIPHERTEXT_HIDING_DRAM: u64 = 1 << 24;
pub const POLICY_PAGE_SWAP_DISABLE: u64 = 1 << 25;

/// `PLATFORM_INFO` bit 5: alias detection completed since reset and found
/// no aliasing address (the BadRAM mitigation, AMD-SB-3015).
pub const PLATFORM_INFO_ALIAS_CHECK_COMPLETE: u64 = 1 << 5;

/// How the eight bytes of a `TCB_VERSION` are laid out — per product
/// (56860 rev. 1.58, Tables 3 and 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcbLayout {
    /// Milan and Genoa: byte 0 boot loader, 1 TEE, 2..5 reserved, 6 SNP,
    /// 7 microcode.
    MilanGenoa,
    /// Turin: byte 0 FMC, 1 boot loader, 2 TEE, 3 SNP, 4..6 reserved,
    /// 7 microcode.
    Turin,
}

/// The security patch levels of one `TCB_VERSION`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TcbParts {
    /// Turin only; 0 on Milan and Genoa, which have no FMC component.
    pub fmc: u8,
    pub boot_loader: u8,
    pub tee: u8,
    pub snp: u8,
    pub microcode: u8,
}

impl TcbParts {
    pub fn decode(raw: [u8; 8], layout: TcbLayout) -> Self {
        match layout {
            TcbLayout::MilanGenoa => Self {
                fmc: 0,
                boot_loader: raw[0],
                tee: raw[1],
                snp: raw[6],
                microcode: raw[7],
            },
            TcbLayout::Turin => Self {
                fmc: raw[0],
                boot_loader: raw[1],
                tee: raw[2],
                snp: raw[3],
                microcode: raw[7],
            },
        }
    }
}

/// CPUID family, model and stepping as the firmware reports them (v3+):
/// family is the extended family (0x19 Milan/Genoa, 0x1A Turin), model the
/// full model byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cpuid {
    pub family: u8,
    pub model: u8,
    pub stepping: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnpReport {
    pub version: u32,
    pub guest_svn: u32,
    pub policy: u64,
    pub family_id: [u8; 16],
    pub image_id: [u8; 16],
    pub vmpl: u32,
    pub signature_algo: u32,
    pub current_tcb: [u8; 8],
    pub platform_info: u64,
    /// `AUTHOR_KEY_EN` (bit 0), `MASK_CHIP_KEY` (bit 1), `SIGNING_KEY` (bits 4:2).
    pub key_info: u32,
    pub report_data: [u8; 64],
    pub measurement: [u8; 48],
    pub host_data: [u8; 32],
    pub id_key_digest: [u8; 48],
    pub author_key_digest: [u8; 48],
    pub report_id: [u8; 32],
    pub report_id_ma: [u8; 32],
    pub reported_tcb: [u8; 8],
    /// v3 and later; `None` on v2, where the bytes are reserved.
    pub cpuid: Option<Cpuid>,
    pub chip_id: [u8; 64],
    pub committed_tcb: [u8; 8],
    /// Firmware (build, minor, major).
    pub current_version: (u8, u8, u8),
    pub committed_version: (u8, u8, u8),
    pub launch_tcb: [u8; 8],
    /// v5 and later; `None` before.
    pub launch_mit_vector: Option<u64>,
    pub current_mit_vector: Option<u64>,
    pub signature_r: [u8; SIGNATURE_COMPONENT_LEN],
    pub signature_s: [u8; SIGNATURE_COMPONENT_LEN],
}

impl SnpReport {
    /// `POLICY.DEBUG`: the host may read and write the guest's memory
    /// through the firmware's debug commands.
    pub fn is_debug(&self) -> bool {
        self.policy & POLICY_DEBUG != 0
    }
    /// `POLICY.MIGRATE_MA`: a migration agent may be associated.
    pub fn allows_migration_agent(&self) -> bool {
        self.policy & POLICY_MIGRATE_MA != 0
    }
    /// `REPORT_ID_MA` is all 0xFF when no migration agent is bound.
    pub fn has_migration_agent(&self) -> bool {
        self.report_id_ma != [0xFF; 32]
    }
    pub fn signing_key(&self) -> u32 {
        (self.key_info >> 2) & 0b111
    }
    pub fn mask_chip_key(&self) -> bool {
        self.key_info & 0b10 != 0
    }
    pub fn author_key_en(&self) -> bool {
        self.key_info & 1 != 0
    }
    pub fn alias_check_complete(&self) -> bool {
        self.platform_info & PLATFORM_INFO_ALIAS_CHECK_COMPLETE != 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnpReportRejected {
    WrongLength {
        len: usize,
        need: usize,
    },
    UnsupportedVersion {
        version: u32,
    },
    /// A byte Table 23 reserves for this version is not zero: a field this
    /// parser does not know.
    ReservedSet {
        at: usize,
    },
}

impl std::fmt::Display for SnpReportRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongLength { len, need } => {
                write!(f, "an SEV-SNP report is {need} bytes, got {len}")
            }
            Self::UnsupportedVersion { version } => write!(
                f,
                "SEV-SNP report version {version} is not read here (2, 3 and 5 are)"
            ),
            Self::ReservedSet { at } => write!(
                f,
                "byte {at:#x} is reserved and set: a field this parser does not know"
            ),
        }
    }
}

fn arr<const N: usize>(b: &[u8], at: usize) -> [u8; N] {
    let mut out = [0u8; N];
    out.copy_from_slice(&b[at..at + N]);
    out
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(arr(b, at))
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(arr(b, at))
}

/// The byte ranges Table 23 (ABI 56860 rev. 1.58) reserves in a report of
/// `version`: CPUID is reserved before v3, the mitigation vectors before
/// v5. The signature's unused bytes are judged with the signature.
fn reserved(version: u32) -> [std::ops::Range<usize>; 6] {
    let cpuid_from = if version >= 3 { 0x18B } else { 0x188 };
    let mit_end = if version >= 5 { 0x208 } else { 0x1F8 };
    [
        0x4C..0x50,
        cpuid_from..0x1A0,
        0x1EB..0x1EC,
        0x1EF..0x1F0,
        mit_end..SIGNED_LEN,
        0x330..REPORT_LEN,
    ]
}

/// Parse exactly one report: [`REPORT_LEN`] bytes, version 2, 3 or 5, every
/// reserved byte zero.
pub fn parse(report: &[u8]) -> Result<SnpReport, SnpReportRejected> {
    if report.len() != REPORT_LEN {
        return Err(SnpReportRejected::WrongLength {
            len: report.len(),
            need: REPORT_LEN,
        });
    }
    let b = report;
    let version = u32_at(b, 0x00);
    if !matches!(version, 2 | 3 | 5) {
        return Err(SnpReportRejected::UnsupportedVersion { version });
    }
    for range in reserved(version) {
        if let Some(i) = b[range.clone()].iter().position(|x| *x != 0) {
            return Err(SnpReportRejected::ReservedSet {
                at: range.start + i,
            });
        }
    }
    let cpuid = (version >= 3).then(|| Cpuid {
        family: b[CPUID_AT],
        model: b[CPUID_AT + 1],
        stepping: b[CPUID_AT + 2],
    });
    let (launch_mit_vector, current_mit_vector) = if version >= 5 {
        (
            Some(u64_at(b, LAUNCH_MIT_VECTOR_AT)),
            Some(u64_at(b, CURRENT_MIT_VECTOR_AT)),
        )
    } else {
        (None, None)
    };
    let version3 = |at: usize| (b[at], b[at + 1], b[at + 2]);
    Ok(SnpReport {
        version,
        guest_svn: u32_at(b, 0x04),
        policy: u64_at(b, 0x08),
        family_id: arr(b, 0x10),
        image_id: arr(b, 0x20),
        vmpl: u32_at(b, 0x30),
        signature_algo: u32_at(b, 0x34),
        current_tcb: arr(b, 0x38),
        platform_info: u64_at(b, 0x40),
        key_info: u32_at(b, 0x48),
        report_data: arr(b, REPORT_DATA_AT),
        measurement: arr(b, MEASUREMENT_AT),
        host_data: arr(b, HOST_DATA_AT),
        id_key_digest: arr(b, ID_KEY_DIGEST_AT),
        author_key_digest: arr(b, AUTHOR_KEY_DIGEST_AT),
        report_id: arr(b, REPORT_ID_AT),
        report_id_ma: arr(b, REPORT_ID_MA_AT),
        reported_tcb: arr(b, REPORTED_TCB_AT),
        cpuid,
        chip_id: arr(b, CHIP_ID_AT),
        committed_tcb: arr(b, COMMITTED_TCB_AT),
        current_version: version3(CURRENT_VERSION_AT),
        committed_version: version3(COMMITTED_VERSION_AT),
        launch_tcb: arr(b, LAUNCH_TCB_AT),
        launch_mit_vector,
        current_mit_vector,
        signature_r: arr(b, SIGNATURE_R_AT),
        signature_s: arr(b, SIGNATURE_S_AT),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A report with a sentinel in every field, so a shifted offset shows.
    fn synthetic(version: u32) -> Vec<u8> {
        let mut r = vec![0u8; REPORT_LEN];
        r[0..4].copy_from_slice(&version.to_le_bytes());
        r[0x04..0x08].copy_from_slice(&7u32.to_le_bytes());
        r[0x08..0x10].copy_from_slice(&0x0003_001Fu64.to_le_bytes());
        r[0x10..0x20].fill(0xF1);
        r[0x20..0x30].fill(0xF2);
        r[0x30..0x34].copy_from_slice(&0u32.to_le_bytes());
        r[0x34..0x38].copy_from_slice(&1u32.to_le_bytes());
        r[0x38..0x40].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        r[0x40..0x48].copy_from_slice(&0x24u64.to_le_bytes());
        r[0x48..0x4C].copy_from_slice(&0b1_0011u32.to_le_bytes());
        r[0x50..0x90].fill(0xA0);
        r[0x90..0xC0].fill(0xA1);
        r[0xC0..0xE0].fill(0xA2);
        r[0xE0..0x110].fill(0xA3);
        r[0x110..0x140].fill(0xA4);
        r[0x140..0x160].fill(0xA5);
        r[0x160..0x180].fill(0xFF);
        r[0x180..0x188].copy_from_slice(&[0x0A, 0, 0, 0, 0, 0, 0x1B, 0x58]);
        if version >= 3 {
            r[0x188..0x18B].copy_from_slice(&[0x19, 0x11, 0x01]);
        }
        r[0x1A0..0x1E0].fill(0xA6);
        r[0x1E0..0x1E8].copy_from_slice(&[9, 9, 9, 9, 9, 9, 9, 9]);
        r[0x1E8..0x1EB].copy_from_slice(&[0x28, 0x37, 0x01]);
        r[0x1EC..0x1EF].copy_from_slice(&[0x27, 0x37, 0x01]);
        r[0x1F0..0x1F8].copy_from_slice(&[8, 8, 8, 8, 8, 8, 8, 8]);
        if version >= 5 {
            r[0x1F8..0x200].copy_from_slice(&0x36u64.to_le_bytes());
            r[0x200..0x208].copy_from_slice(&0x16u64.to_le_bytes());
        }
        r[0x2A0..0x2E8].fill(0xB1);
        r[0x2E8..0x330].fill(0xB2);
        r
    }

    #[test]
    fn every_field_is_read_at_its_table_23_offset() {
        let r = parse(&synthetic(5)).unwrap();
        assert_eq!((r.version, r.guest_svn, r.policy), (5, 7, 0x0003_001F));
        assert_eq!((r.family_id, r.image_id), ([0xF1; 16], [0xF2; 16]));
        assert_eq!((r.vmpl, r.signature_algo), (0, 1));
        assert_eq!(r.current_tcb, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(r.platform_info, 0x24);
        assert!(r.alias_check_complete());
        assert_eq!(
            (r.signing_key(), r.mask_chip_key(), r.author_key_en()),
            (4, true, true)
        );
        assert_eq!(r.report_data, [0xA0; 64]);
        assert_eq!(r.measurement, [0xA1; 48]);
        assert_eq!(r.host_data, [0xA2; 32]);
        assert_eq!(r.id_key_digest, [0xA3; 48]);
        assert_eq!(r.author_key_digest, [0xA4; 48]);
        assert_eq!(r.report_id, [0xA5; 32]);
        assert!(!r.has_migration_agent());
        assert_eq!(r.reported_tcb, [0x0A, 0, 0, 0, 0, 0, 0x1B, 0x58]);
        assert_eq!(
            r.cpuid,
            Some(Cpuid {
                family: 0x19,
                model: 0x11,
                stepping: 1
            })
        );
        assert_eq!(r.chip_id, [0xA6; 64]);
        assert_eq!(r.committed_tcb, [9; 8]);
        assert_eq!(r.current_version, (0x28, 0x37, 0x01));
        assert_eq!(r.committed_version, (0x27, 0x37, 0x01));
        assert_eq!(r.launch_tcb, [8; 8]);
        assert_eq!(
            (r.launch_mit_vector, r.current_mit_vector),
            (Some(0x36), Some(0x16))
        );
        assert_eq!((r.signature_r, r.signature_s), ([0xB1; 72], [0xB2; 72]));
    }

    /// v2 has no CPUID and v3 no mitigation vectors: the bytes there are
    /// reserved, and reading them as fields would invent values.
    #[test]
    fn fields_a_version_does_not_define_are_not_read() {
        let v2 = parse(&synthetic(2)).unwrap();
        assert_eq!((v2.cpuid, v2.launch_mit_vector), (None, None));
        let v3 = parse(&synthetic(3)).unwrap();
        assert!(v3.cpuid.is_some());
        assert_eq!((v3.launch_mit_vector, v3.current_mit_vector), (None, None));
    }

    /// A reserved byte set is a field this parser would silently drop:
    /// refused, per version (CPUID before v3, mitigation vectors before v5).
    #[test]
    fn a_reserved_byte_set_is_refused() {
        for (version, at) in [
            (3, 0x4C),
            (3, 0x4F),
            (2, 0x188),
            (3, 0x18B),
            (5, 0x19F),
            (3, 0x1EB),
            (3, 0x1EF),
            (3, 0x1F8),
            (5, 0x208),
            (5, 0x29F),
            (3, 0x330),
            (2, 0x49F),
        ] {
            let mut b = synthetic(version);
            b[at] = 0x01;
            assert_eq!(
                parse(&b),
                Err(SnpReportRejected::ReservedSet { at }),
                "v{version} {at:#x}"
            );
        }
        // The same bytes are fields in a later version.
        let mut b = synthetic(5);
        b[0x1F8] = 0x01;
        assert!(parse(&b).is_ok());
    }

    #[test]
    fn a_version_whose_layout_is_not_implemented_is_refused() {
        for version in [0, 1, 4, 6, 0xFFFF_FFFF] {
            assert_eq!(
                parse(&synthetic(version)),
                Err(SnpReportRejected::UnsupportedVersion { version }),
                "version {version}"
            );
        }
    }

    #[test]
    fn a_report_of_another_length_is_refused_not_truncated() {
        for len in [0, REPORT_LEN - 1, REPORT_LEN + 1] {
            let mut r = synthetic(3);
            r.resize(len, 0);
            assert_eq!(
                parse(&r),
                Err(SnpReportRejected::WrongLength {
                    len,
                    need: REPORT_LEN
                })
            );
        }
    }

    #[test]
    fn the_tcb_layout_is_the_products() {
        let raw = [0x0A, 0x01, 0x02, 0x1B, 0, 0, 0x17, 0x54];
        assert_eq!(
            TcbParts::decode(raw, TcbLayout::MilanGenoa),
            TcbParts {
                fmc: 0,
                boot_loader: 0x0A,
                tee: 0x01,
                snp: 0x17,
                microcode: 0x54
            }
        );
        assert_eq!(
            TcbParts::decode(raw, TcbLayout::Turin),
            TcbParts {
                fmc: 0x0A,
                boot_loader: 0x01,
                tee: 0x02,
                snp: 0x1B,
                microcode: 0x54
            }
        );
    }

    #[test]
    fn debug_migration_agent_and_policy_bits_are_read() {
        let mut b = synthetic(3);
        let policy = POLICY_RESERVED_MUST_BE_ONE | POLICY_DEBUG | POLICY_MIGRATE_MA;
        b[0x08..0x10].copy_from_slice(&policy.to_le_bytes());
        b[0x160] = 0;
        let r = parse(&b).unwrap();
        assert!(r.is_debug() && r.allows_migration_agent() && r.has_migration_agent());
    }
}
