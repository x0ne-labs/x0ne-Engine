//! The SEV-SNP launch measurement of a QEMU guest, predicted offline from
//! its launch inputs: what `MEASUREMENT` in the guest's report must be.
//!
//! The AMD firmware extends a SHA-384 launch digest once per page given to
//! `SNP_LAUNCH_UPDATE`, over a `PAGE_INFO` (AMD SEV-SNP Firmware ABI,
//! publication 56860, `SNP_LAUNCH_UPDATE`): the digest so far, the page's
//! contents digest, its type and its guest physical address. For a QEMU
//! guest booted on an OVMF `AmdSevX64` build with `-kernel`:
//!
//! 1. every 4 KiB page of the firmware, NORMAL, from 4 GiB minus its size;
//! 2. the firmware's SEV metadata sections, in table order: pre-validated
//!    memory as ZERO pages, the secrets page, the CPUID page, and the SEV
//!    hashes table page (SHA-256 of the kernel, of the initrd, of the command
//!    line with its NUL) as a NORMAL page;
//! 3. one VMSA page per vCPU: the reset state QEMU gives the boot CPU and the
//!    others (their RIP from the firmware's SEV-ES reset block), RDX holding
//!    the vCPU model's CPUID signature, SEV_FEATURES the guest features.
//!
//! Same algorithm and constants as virtee/sev-snp-measure 0.0.13
//! (Apache-2.0), rewritten here without a dependency; the tests hold it to
//! answers that tool computed. QEMU only: the host that launches the guest
//! is ours, and other VMMs (EC2, GCE) build other VMSAs.
//!
//! What it does not prove: that a host launched this guest with these inputs.
//! Only a report whose `MEASUREMENT` equals the prediction says that, and the
//! prediction is only as good as the inputs it was given (the firmware bytes,
//! the kernel bytes, the vCPU count and model, the guest features).

use sha2::{Digest, Sha256, Sha384};

pub const PAGE: usize = 4096;
const FOUR_GIB: u64 = 1 << 32;
/// A VMSA page is recorded at GPA -1, page aligned, bits above 51 clear.
const VMSA_GPA: u64 = 0xFFFF_FFFF_F000;

// PAGE_INFO.PAGE_TYPE (56860, table 67).
const PAGE_TYPE_NORMAL: u8 = 1;
const PAGE_TYPE_VMSA: u8 = 2;
const PAGE_TYPE_ZERO: u8 = 3;
const PAGE_TYPE_SECRETS: u8 = 5;
const PAGE_TYPE_CPUID: u8 = 6;

// OVMF SEV metadata section types (OvmfPkg/ResetVector/X64/OvmfSevMetadata.asm).
const SECTION_SNP_SEC_MEM: u32 = 1;
const SECTION_SNP_SECRETS: u32 = 2;
const SECTION_CPUID: u32 = 3;
const SECTION_SVSM_CAA: u32 = 4;
const SECTION_SNP_KERNEL_HASHES: u32 = 0x10;

// GUIDs of the OVMF footer table (OvmfPkg/ResetVector/Ia16/ResetVectorVtf0.asm).
const OVMF_TABLE_FOOTER: &str = "96b582de-1fb2-45f7-baea-a366c55a082d";
const SEV_HASH_TABLE_RV: &str = "7255371f-3a3b-4b04-927b-1da6efa8d454";
const SEV_ES_RESET_BLOCK: &str = "00f771de-1a7e-4fcb-890e-68c77e2fb44e";
const OVMF_SEV_METADATA: &str = "dc886566-984a-4798-a75e-5585a7bf67cc";

// The SEV hashes table (QEMU target/i386/sev.c, OvmfPkg BlobVerifierLibSevHashes).
const HASHES_HEADER: &str = "9438d606-4f22-4cc9-b479-a793d411fd21";
const HASHES_KERNEL: &str = "4de79437-abd2-427f-b835-d5b172d2045b";
const HASHES_INITRD: &str = "44baf731-3a2f-4bd7-9af1-41e29169781d";
const HASHES_CMDLINE: &str = "97d02dd8-bd20-4c94-aa78-e7714d36ab2a";

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LaunchError {
    #[error("firmware: {0}")]
    Firmware(String),
    #[error(
        "a kernel is given and this firmware has no SNP_KERNEL_HASHES section: -kernel would \
         boot unmeasured (not an AmdSev build)"
    )]
    KernelUnmeasured,
    #[error("an initrd or a command line without a kernel: QEMU measures them only with -kernel")]
    NoKernel,
    #[error("{0} vCPUs: between 1 and 512")]
    Vcpus(u32),
}

/// Everything QEMU feeds into the launch digest of one guest.
#[derive(Debug, Clone, Copy)]
pub struct Launch<'a> {
    /// The firmware passed to `-bios`, exactly.
    pub ovmf: &'a [u8],
    /// `-kernel` (a bzImage or a UKI), exactly as passed.
    pub kernel: Option<&'a [u8]>,
    /// `-initrd`; absent hashes the empty string.
    pub initrd: Option<&'a [u8]>,
    /// `-append`; absent or empty hashes a single NUL.
    pub append: Option<&'a str>,
    /// `-smp`.
    pub vcpus: u32,
    /// CPUID Fn0000_0001_EAX of the `-cpu` model ([`vcpu_signature`]).
    pub vcpu_sig: u32,
    /// The VMSA's SEV_FEATURES: 0x1 (SNPActive) for QEMU's defaults.
    pub guest_features: u64,
}

/// CPUID Fn0000_0001_EAX from family, model, stepping (AMD publication 25481).
pub const fn cpu_signature(family: u32, model: u32, stepping: u32) -> u32 {
    let (fam_lo, fam_hi) = if family > 0xF {
        (0xF, (family - 0xF) & 0xFF)
    } else {
        (family, 0)
    };
    (fam_hi << 20)
        | ((model >> 4 & 0xF) << 16)
        | (fam_lo << 8)
        | ((model & 0xF) << 4)
        | (stepping & 0xF)
}

/// What RDX holds at reset for a QEMU `-cpu` model (QEMU `builtin_x86_defs`,
/// as sev-snp-measure 0.0.13 lists them). `host` is not one: the
/// measurement must not depend on the chip it runs on.
pub fn vcpu_signature(model: &str) -> Option<u32> {
    Some(match model {
        "EPYC" | "EPYC-v1" | "EPYC-v2" | "EPYC-IBPB" | "EPYC-v3" | "EPYC-v4" => {
            cpu_signature(23, 1, 2)
        }
        "EPYC-Rome" | "EPYC-Rome-v1" | "EPYC-Rome-v2" | "EPYC-Rome-v3" => cpu_signature(23, 49, 0),
        "EPYC-Milan" | "EPYC-Milan-v1" | "EPYC-Milan-v2" => cpu_signature(25, 1, 1),
        "EPYC-Genoa" | "EPYC-Genoa-v1" => cpu_signature(25, 17, 0),
        "EPYC-Turin" => cpu_signature(26, 0, 0),
        _ => return None,
    })
}

/// A GUID in its in-memory (mixed-endian) form: the first three fields
/// little-endian, the last two as written.
fn guid_le(s: &str) -> [u8; 16] {
    let hex: Vec<u8> = s
        .bytes()
        .filter(|b| *b != b'-')
        .collect::<Vec<u8>>()
        .chunks(2)
        .map(|c| u8::from_str_radix(std::str::from_utf8(c).expect("ascii"), 16).expect("hex"))
        .collect();
    assert_eq!(hex.len(), 16, "{s}");
    let mut g = [0u8; 16];
    g[0..4].copy_from_slice(&[hex[3], hex[2], hex[1], hex[0]]);
    g[4..6].copy_from_slice(&[hex[5], hex[4]]);
    g[6..8].copy_from_slice(&[hex[7], hex[6]]);
    g[8..16].copy_from_slice(&hex[8..16]);
    g
}

fn le16(d: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(d.get(at..at + 2)?.try_into().ok()?))
}

fn le32(d: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(at..at + 4)?.try_into().ok()?))
}

/// One OVMF SEV metadata section: where, how big, what.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Section {
    pub gpa: u32,
    pub size: u32,
    pub kind: u32,
}

/// An OVMF binary, its footer GUID table and its SEV metadata.
pub struct Ovmf<'a> {
    data: &'a [u8],
    table: Vec<([u8; 16], &'a [u8])>,
    sections: Vec<Section>,
}

impl<'a> Ovmf<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, LaunchError> {
        let bad = |m: &str| LaunchError::Firmware(m.to_string());
        if data.is_empty() || !data.len().is_multiple_of(PAGE) || data.len() as u64 > FOUR_GIB {
            return Err(LaunchError::Firmware(format!(
                "size {} is not a non-zero multiple of 4096 below 4 GiB",
                data.len()
            )));
        }
        // The footer entry (u16 size, GUID) ends 32 bytes before the end; the
        // table grows downwards from it, each entry `data, u16 size, GUID`.
        let foot = data
            .len()
            .checked_sub(32 + 18)
            .ok_or_else(|| bad("too small for a footer table"))?;
        let mut table = Vec::new();
        if data[foot + 2..foot + 18] == guid_le(OVMF_TABLE_FOOTER) {
            let size = usize::from(le16(data, foot).ok_or_else(|| bad("footer"))?);
            let start = foot
                .checked_sub(size.checked_sub(18).ok_or_else(|| bad("footer size"))?)
                .ok_or_else(|| bad("footer table larger than the firmware"))?;
            let mut body = &data[start..foot];
            while body.len() >= 18 {
                let n = body.len();
                let esize = usize::from(le16(body, n - 18).expect("18 bytes"));
                if esize < 18 || esize > n {
                    return Err(bad("invalid footer table entry size"));
                }
                let mut guid = [0u8; 16];
                guid.copy_from_slice(&body[n - 16..]);
                table.push((guid, &body[n - esize..n - 18]));
                body = &body[..n - esize];
            }
        }
        let mut ovmf = Self {
            data,
            table,
            sections: Vec::new(),
        };
        if let Some(entry) = ovmf.entry(OVMF_SEV_METADATA) {
            let back = le32(entry, 0).ok_or_else(|| bad("SEV metadata entry"))? as usize;
            let start = data
                .len()
                .checked_sub(back)
                .ok_or_else(|| bad("SEV metadata offset beyond the firmware"))?;
            let (sig, version, n) = (
                data.get(start..start + 4)
                    .ok_or_else(|| bad("SEV metadata"))?,
                le32(data, start + 8).ok_or_else(|| bad("SEV metadata"))?,
                le32(data, start + 12).ok_or_else(|| bad("SEV metadata"))?,
            );
            if sig != b"ASEV" || version != 1 {
                return Err(bad("SEV metadata: not \"ASEV\" version 1"));
            }
            for i in 0..n as usize {
                let at = start + 16 + 12 * i;
                let field = |k: usize| le32(data, at + 4 * k).ok_or_else(|| bad("SEV metadata"));
                ovmf.sections.push(Section {
                    gpa: field(0)?,
                    size: field(1)?,
                    kind: field(2)?,
                });
            }
        }
        Ok(ovmf)
    }

    fn entry(&self, guid: &str) -> Option<&'a [u8]> {
        let g = guid_le(guid);
        self.table.iter().find(|(k, _)| *k == g).map(|(_, v)| *v)
    }

    /// Where the firmware is loaded: its last byte at 4 GiB - 1.
    pub fn gpa(&self) -> u64 {
        FOUR_GIB - self.data.len() as u64
    }

    pub fn sections(&self) -> &[Section] {
        &self.sections
    }

    fn has(&self, kind: u32) -> bool {
        self.sections.iter().any(|s| s.kind == kind)
    }

    /// The APs' first instruction, from the SEV-ES reset block.
    pub fn sev_es_reset_eip(&self) -> Result<u32, LaunchError> {
        self.entry(SEV_ES_RESET_BLOCK)
            .and_then(|e| le32(e, 0))
            .ok_or_else(|| {
                LaunchError::Firmware("no SEV-ES reset block: not an SEV-capable OVMF".into())
            })
    }

    fn hashes_table_gpa(&self) -> Result<u32, LaunchError> {
        self.entry(SEV_HASH_TABLE_RV)
            .and_then(|e| le32(e, 0))
            .ok_or_else(|| {
                LaunchError::Firmware(
                    "no SEV hashes table: not an AmdSev build, -kernel is not measured".into(),
                )
            })
    }
}

/// The table QEMU writes at the firmware's hashes GPA
/// (`sev_add_kernel_loader_hashes`): the header GUID and its length, then the
/// command line, initrd and kernel entries (GUID, u16 length 50, SHA-256),
/// padded to 16 bytes.
pub fn sev_hashes_table(kernel: &[u8], initrd: Option<&[u8]>, append: Option<&str>) -> Vec<u8> {
    let mut cmdline = append.unwrap_or("").as_bytes().to_vec();
    cmdline.push(0);
    let entry = |guid: &str, data: &[u8]| {
        let mut e = guid_le(guid).to_vec();
        e.extend_from_slice(&50u16.to_le_bytes());
        e.extend_from_slice(&Sha256::digest(data));
        e
    };
    let mut body = entry(HASHES_CMDLINE, &cmdline);
    body.extend(entry(HASHES_INITRD, initrd.unwrap_or(b"")));
    body.extend(entry(HASHES_KERNEL, kernel));
    let mut table = guid_le(HASHES_HEADER).to_vec();
    table.extend_from_slice(&((18 + body.len()) as u16).to_le_bytes());
    table.extend(body);
    table.resize(table.len().div_ceil(16) * 16, 0);
    table
}

/// The launch digest, one `PAGE_INFO` at a time.
struct Digest384([u8; 48]);

impl Digest384 {
    fn page(&mut self, page_type: u8, gpa: u64, contents: &[u8; 48]) {
        // PAGE_INFO (0x70): DIGEST_CUR, CONTENTS, LENGTH, PAGE_TYPE, IMI_PAGE,
        // VMPL3..VMPL1 permissions, reserved, GPA.
        let mut info = [0u8; 0x70];
        info[..48].copy_from_slice(&self.0);
        info[48..96].copy_from_slice(contents);
        info[96..98].copy_from_slice(&0x70u16.to_le_bytes());
        info[98] = page_type;
        info[104..112].copy_from_slice(&gpa.to_le_bytes());
        self.0 = Sha384::digest(info).into();
    }

    fn normal(&mut self, gpa: u64, data: &[u8]) {
        debug_assert_eq!(data.len() % PAGE, 0);
        for (i, p) in data.chunks(PAGE).enumerate() {
            self.page(
                PAGE_TYPE_NORMAL,
                gpa + (i * PAGE) as u64,
                &Sha384::digest(p).into(),
            );
        }
    }

    fn zero(&mut self, gpa: u64, size: u64) {
        for off in (0..size).step_by(PAGE) {
            self.page(PAGE_TYPE_ZERO, gpa + off, &[0; 48]);
        }
    }
}

/// The reset state QEMU and KVM give a vCPU of an SNP guest, as a VMSA page
/// (AMD APM volume 2, table B-4; `struct sev_es_save_area`).
pub fn vmsa_page(eip: u32, guest_features: u64, vcpu_sig: u32) -> [u8; PAGE] {
    let mut p = [0u8; PAGE];
    let mut put = |at: usize, bytes: &[u8]| p[at..at + bytes.len()].copy_from_slice(bytes);
    // Segments: selector u16, attributes u16, limit u32, base u64.
    let seg = |selector: u16, attrib: u16, base: u64| {
        let mut s = [0u8; 16];
        s[0..2].copy_from_slice(&selector.to_le_bytes());
        s[2..4].copy_from_slice(&attrib.to_le_bytes());
        s[4..8].copy_from_slice(&0xFFFFu32.to_le_bytes());
        s[8..16].copy_from_slice(&base.to_le_bytes());
        s
    };
    put(0x00, &seg(0, 0x93, 0)); // es
    put(0x10, &seg(0xF000, 0x9B, u64::from(eip & 0xFFFF_0000))); // cs
    put(0x20, &seg(0, 0x93, 0)); // ss
    put(0x30, &seg(0, 0x93, 0)); // ds
    put(0x40, &seg(0, 0x93, 0)); // fs
    put(0x50, &seg(0, 0x93, 0)); // gs
    put(0x60, &seg(0, 0, 0)); // gdtr
    put(0x70, &seg(0, 0x82, 0)); // ldtr
    put(0x80, &seg(0, 0, 0)); // idtr
    put(0x90, &seg(0, 0x8B, 0)); // tr
    put(0xD0, &0x1000u64.to_le_bytes()); // efer: SVME
    put(0x148, &0x40u64.to_le_bytes()); // cr4: MCE
    put(0x158, &0x10u64.to_le_bytes()); // cr0: ET
    put(0x160, &0x400u64.to_le_bytes()); // dr7
    put(0x168, &0xFFFF_0FF0u64.to_le_bytes()); // dr6
    put(0x170, &0x2u64.to_le_bytes()); // rflags
    put(0x178, &u64::from(eip & 0xFFFF).to_le_bytes()); // rip
    put(0x268, &0x0007_0406_0007_0406u64.to_le_bytes()); // g_pat
    put(0x310, &u64::from(vcpu_sig).to_le_bytes()); // rdx
    put(0x3B0, &guest_features.to_le_bytes()); // sev_features
    put(0x3E8, &0x1u64.to_le_bytes()); // xcr0
    put(0x408, &0x1F80u32.to_le_bytes()); // mxcsr
    put(0x410, &0x37Fu16.to_le_bytes()); // x87_fcw
    p
}

/// The predicted `MEASUREMENT` of the guest `l` launches.
pub fn launch_digest(l: &Launch) -> Result<[u8; 48], LaunchError> {
    if l.vcpus == 0 || l.vcpus > 512 {
        return Err(LaunchError::Vcpus(l.vcpus));
    }
    if l.kernel.is_none() && (l.initrd.is_some() || l.append.is_some()) {
        return Err(LaunchError::NoKernel);
    }
    let ovmf = Ovmf::parse(l.ovmf)?;
    if l.kernel.is_some() && !ovmf.has(SECTION_SNP_KERNEL_HASHES) {
        return Err(LaunchError::KernelUnmeasured);
    }
    let mut ld = Digest384([0; 48]);
    ld.normal(ovmf.gpa(), l.ovmf);
    for s in ovmf.sections() {
        let (gpa, size) = (u64::from(s.gpa), u64::from(s.size));
        match s.kind {
            SECTION_SNP_SEC_MEM | SECTION_SVSM_CAA => ld.zero(gpa, size),
            SECTION_SNP_SECRETS => ld.page(PAGE_TYPE_SECRETS, gpa, &[0; 48]),
            SECTION_CPUID => ld.page(PAGE_TYPE_CPUID, gpa, &[0; 48]),
            SECTION_SNP_KERNEL_HASHES => match l.kernel {
                None => ld.zero(gpa, size),
                Some(kernel) => {
                    if size != PAGE as u64 {
                        return Err(LaunchError::Firmware(format!(
                            "kernel hashes section of {size} bytes, one page expected"
                        )));
                    }
                    let off = (ovmf.hashes_table_gpa()? & 0xFFF) as usize;
                    let table = sev_hashes_table(kernel, l.initrd, l.append);
                    if off + table.len() > PAGE {
                        return Err(LaunchError::Firmware(
                            "the SEV hashes table crosses its page".into(),
                        ));
                    }
                    let mut page = [0u8; PAGE];
                    page[off..off + table.len()].copy_from_slice(&table);
                    ld.normal(gpa, &page);
                }
            },
            other => {
                return Err(LaunchError::Firmware(format!(
                    "unknown SEV metadata section type {other:#x}"
                )))
            }
        }
    }
    let bsp = vmsa_page(0xFFFF_FFF0, l.guest_features, l.vcpu_sig);
    let ap = vmsa_page(ovmf.sev_es_reset_eip()?, l.guest_features, l.vcpu_sig);
    for i in 0..l.vcpus {
        let vmsa = if i == 0 { &bsp } else { &ap };
        ld.page(PAGE_TYPE_VMSA, VMSA_GPA, &Sha384::digest(vmsa).into());
    }
    Ok(ld.0)
}

/// Lowercase hex, the form `snp_measurement=<96 hex>` takes.
pub fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// `x0ne-engine snp-measure`: the arguments, as QEMU's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasureArgs {
    pub ovmf: std::path::PathBuf,
    pub kernel: Option<std::path::PathBuf>,
    pub initrd: Option<std::path::PathBuf>,
    pub append: Option<String>,
    pub vcpus: u32,
    pub vcpu_type: String,
    pub guest_features: u64,
}

pub const MEASURE_USAGE: &str = "\
x0ne-engine snp-measure --ovmf <file> [--kernel <file>] [--initrd <file>]
    [--append <cmdline>] --vcpus <n> [--vcpu-type <model>] [--guest-features <hex>]

The SEV-SNP launch measurement of a QEMU guest booted on this firmware with
these -kernel, -initrd, -append, -smp and -cpu, computed offline. Prints
snp_measurement=<96 hex>: what the guest's report must carry. Defaults:
--vcpu-type EPYC-Genoa, --guest-features 0x1. QEMU only.
";

/// Parse `snp-measure`'s arguments. Each flag once; nothing positional.
pub fn parse_measure_args(args: &[String]) -> Result<MeasureArgs, String> {
    let mut seen = std::collections::BTreeMap::new();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let known = [
            "--ovmf",
            "--kernel",
            "--initrd",
            "--append",
            "--vcpus",
            "--vcpu-type",
            "--guest-features",
        ];
        if !known.contains(&flag.as_str()) {
            return Err(format!("unknown argument {}", crate::argv::shown(flag)));
        }
        let value = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value"))?
            .clone();
        if seen.insert(flag.clone(), value).is_some() {
            return Err(format!("{flag} given twice"));
        }
    }
    let ovmf = seen
        .remove("--ovmf")
        .ok_or("--ovmf is required: the firmware passed to -bios")?;
    let vcpus = seen
        .remove("--vcpus")
        .ok_or("--vcpus is required: QEMU's -smp")?;
    let vcpus: u32 = vcpus
        .parse()
        .map_err(|_| format!("--vcpus {vcpus}: a number"))?;
    let guest_features = match seen.remove("--guest-features") {
        None => 0x1,
        Some(f) => u64::from_str_radix(f.trim_start_matches("0x"), 16)
            .map_err(|_| format!("--guest-features {f}: hexadecimal"))?,
    };
    let vcpu_type = seen
        .remove("--vcpu-type")
        .unwrap_or_else(|| "EPYC-Genoa".into());
    if vcpu_signature(&vcpu_type).is_none() {
        return Err(format!(
            "--vcpu-type {vcpu_type}: a named EPYC model (EPYC, EPYC-v1..v4, EPYC-IBPB, \
             EPYC-Rome[-v1..v3], EPYC-Milan[-v1, -v2], EPYC-Genoa[-v1], EPYC-Turin)"
        ));
    }
    Ok(MeasureArgs {
        ovmf: ovmf.into(),
        kernel: seen.remove("--kernel").map(Into::into),
        initrd: seen.remove("--initrd").map(Into::into),
        append: seen.remove("--append"),
        vcpus,
        vcpu_type,
        guest_features,
    })
}

/// Read the files `a` names and predict: `snp_measurement=<96 hex>`.
pub fn measure_files(a: &MeasureArgs) -> Result<String, String> {
    // A UKI is tens of MiB; nothing QEMU boots this way comes near 4 GiB.
    let read = |p: &std::path::Path| {
        crate::fsread::read_capped(p, 4 << 30).map_err(|e| format!("{}: {e}", p.display()))
    };
    let ovmf = read(&a.ovmf)?;
    let kernel = a.kernel.as_deref().map(read).transpose()?;
    let initrd = a.initrd.as_deref().map(read).transpose()?;
    let d = launch_digest(&Launch {
        ovmf: &ovmf,
        kernel: kernel.as_deref(),
        initrd: initrd.as_deref(),
        append: a.append.as_deref(),
        vcpus: a.vcpus,
        vcpu_sig: vcpu_signature(&a.vcpu_type).ok_or("unknown --vcpu-type")?,
        guest_features: a.guest_features,
    })
    .map_err(|e| e.to_string())?;
    Ok(format!("snp_measurement={}", hex(&d)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic byte stream: SHA-256(label ‖ counter), concatenated.
    fn stream(label: &str, len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len + 32);
        let mut i = 0u64;
        while out.len() < len {
            let mut h = Sha256::new();
            h.update(label.as_bytes());
            h.update(i.to_le_bytes());
            out.extend_from_slice(&h.finalize());
            i += 1;
        }
        out.truncate(len);
        out
    }

    fn footer_entry(data: &[u8], guid: &str) -> Vec<u8> {
        let mut e = data.to_vec();
        e.extend_from_slice(&((data.len() + 18) as u16).to_le_bytes());
        e.extend_from_slice(&guid_le(guid));
        e
    }

    const FW_SIZE: usize = 0x40000;
    const HASHES_PAGE: u32 = 0x00FF_E000;

    /// A 256 KiB firmware with the footer table and SEV metadata the
    /// measurement reads, and an entry it must skip. `kernel_hashes: false`
    /// is a generic (non-AmdSev) build. Its bytes are what
    /// `tests/snp/launch-fixture/` holds, and what the known answers below
    /// were computed on.
    pub(crate) fn fixture_ovmf(kernel_hashes: bool) -> Vec<u8> {
        let mut fw = stream("x0ne snp_launch fixture firmware", FW_SIZE);
        let mut sections = vec![
            (0x0080_0000u32, 0x6000u32, SECTION_SNP_SEC_MEM),
            (0x0080_B000, 0x1000, SECTION_SNP_SECRETS),
            (0x0080_C000, 0x1000, SECTION_CPUID),
            (0x0081_0000, 0x2000, SECTION_SNP_SEC_MEM),
        ];
        if kernel_hashes {
            sections.push((HASHES_PAGE, 0x1000, SECTION_SNP_KERNEL_HASHES));
        }
        let mut meta = b"ASEV".to_vec();
        meta.extend_from_slice(&((16 + 12 * sections.len()) as u32).to_le_bytes());
        meta.extend_from_slice(&1u32.to_le_bytes());
        meta.extend_from_slice(&(sections.len() as u32).to_le_bytes());
        for (gpa, size, kind) in &sections {
            for v in [gpa, size, kind] {
                meta.extend_from_slice(&v.to_le_bytes());
            }
        }
        let meta_off = FW_SIZE - 0x2000;
        fw[meta_off..meta_off + meta.len()].copy_from_slice(&meta);
        // An entry real firmware carries and the measurement ignores.
        let mut sev_info = 0x0080_6000u32.to_le_bytes().to_vec();
        sev_info.extend_from_slice(&0x1000u32.to_le_bytes());
        let mut body = footer_entry(&sev_info, "00f771de-1a7e-4fcb-890e-68c77e2fb44f");
        if kernel_hashes {
            // The table sits 0x400 into its page, as in real AmdSev builds.
            let mut t = (HASHES_PAGE + 0x400).to_le_bytes().to_vec();
            t.extend_from_slice(&0x400u32.to_le_bytes());
            body.extend(footer_entry(&t, SEV_HASH_TABLE_RV));
        }
        body.extend(footer_entry(
            &0xFFFF_FD40u32.to_le_bytes(),
            SEV_ES_RESET_BLOCK,
        ));
        body.extend(footer_entry(
            &((FW_SIZE - meta_off) as u32).to_le_bytes(),
            OVMF_SEV_METADATA,
        ));
        body.extend_from_slice(&((body.len() + 18) as u16).to_le_bytes());
        body.extend_from_slice(&guid_le(OVMF_TABLE_FOOTER));
        let start = FW_SIZE - 32 - body.len();
        fw[start..FW_SIZE - 32].copy_from_slice(&body);
        fw[FW_SIZE - 32..].fill(0);
        fw
    }

    pub(crate) fn fixture_kernel() -> Vec<u8> {
        stream("x0ne snp_launch fixture kernel", 0x20000 + 0x123)
    }

    pub(crate) fn fixture_initrd() -> Vec<u8> {
        stream("x0ne snp_launch fixture initrd", 0x8000 + 0x45)
    }

    struct Case {
        name: &'static str,
        hashes_fw: bool,
        vcpus: u32,
        cpu: &'static str,
        features: u64,
        kernel: bool,
        initrd: bool,
        append: Option<&'static str>,
        expect: &'static str,
    }

    /// Computed on 2026-10-28 by virtee/sev-snp-measure 0.0.13 (container
    /// `localhost/x0ne-ssm-ref:0.0.13`, no network), `--mode snp --vmm-type
    /// QEMU`, on the files `fixture_files_for_the_reference_tool` writes.
    const KNOWN: &[Case] = &[
        Case {
            name: "genoa_1vcpu_kernel_initrd_append",
            hashes_fw: true,
            vcpus: 1,
            cpu: "EPYC-Genoa",
            features: 0x1,
            kernel: true,
            initrd: true,
            append: Some("console=ttyS0 roothash=4bf766c7d52318e17d75598b4753de1a05795c2af03ab799659c70c1a653fdf3"),
            expect: "a99e2682962ebd058202f3aec6468b1e1a27a1878dfd263fb1e0c4d15fad62668fde5f07671b0dedd94557ec78286245",
        },
        Case {
            name: "genoa_4vcpu_kernel_only",
            hashes_fw: true,
            vcpus: 4,
            cpu: "EPYC-Genoa",
            features: 0x1,
            kernel: true,
            initrd: false,
            append: None,
            expect: "750798d83eb4d37458ee68fdf5b95634153dbf5289496b79ef45ffa9d684161bb3e8b48facdba629d2b3e366bbfa0f0b",
        },
        Case {
            name: "milan_2vcpu_kernel_append",
            hashes_fw: true,
            vcpus: 2,
            cpu: "EPYC-Milan",
            features: 0x1,
            kernel: true,
            initrd: false,
            append: Some("quiet"),
            expect: "c30c75fceea848acff07dd1b52a6b8c100ea0637b3b39b1932aa9ab232730dbf793bf3a572b8d374c4c5370518a4e973",
        },
        Case {
            name: "genoa_16vcpu_no_kernel",
            hashes_fw: true,
            vcpus: 16,
            cpu: "EPYC-Genoa",
            features: 0x1,
            kernel: false,
            initrd: false,
            append: None,
            expect: "96ec97ebcb68e2dcb815c4b8534095be3645b536c060501527f9924b76cb5a66db6de14151b809a601e855eb5261e46f",
        },
        Case {
            name: "epycv4_8vcpu_features_0x21_kernel_initrd",
            hashes_fw: true,
            vcpus: 8,
            cpu: "EPYC-v4",
            features: 0x21,
            kernel: true,
            initrd: true,
            append: None,
            expect: "b8191d364078e822e90b5944c9174e6abf1a3c9de8d162c2228fffe83b40ab0a4aab611010d7bb177ead61a77f5d7c9c",
        },
        Case {
            name: "turin_1vcpu_kernel",
            hashes_fw: true,
            vcpus: 1,
            cpu: "EPYC-Turin",
            features: 0x1,
            kernel: true,
            initrd: false,
            append: None,
            expect: "bf157c97d345d9eb22474277f6840ba89d9a7f720355adf99a7e076e83d68b46f128d399c8fdfd6f6bccd47bb9799856",
        },
        Case {
            name: "plain_firmware_genoa_2vcpu",
            hashes_fw: false,
            vcpus: 2,
            cpu: "EPYC-Genoa",
            features: 0x1,
            kernel: false,
            initrd: false,
            append: None,
            expect: "e07651781122ded26acda4b3fc49f3e17f3f9695156c9054e94e5bfe3913bf4973a96b55b68c2c70db5eb63bdc00faea",
        },
    ];

    fn run(c: &Case) -> Result<[u8; 48], LaunchError> {
        let (fw, k, i) = (
            fixture_ovmf(c.hashes_fw),
            fixture_kernel(),
            fixture_initrd(),
        );
        launch_digest(&Launch {
            ovmf: &fw,
            kernel: c.kernel.then_some(&k[..]),
            initrd: c.initrd.then_some(&i[..]),
            append: c.append,
            vcpus: c.vcpus,
            vcpu_sig: vcpu_signature(c.cpu).expect("known model"),
            guest_features: c.features,
        })
    }

    fn sha256_hex(d: &[u8]) -> String {
        hex(&Sha256::digest(d))
    }

    #[test]
    fn the_prediction_is_the_reference_tools_answer() {
        // The bytes the reference tool read: a generator change is a new
        // fixture, and its answers must be computed again.
        assert_eq!(
            sha256_hex(&fixture_ovmf(true)),
            "833cfcf267aa165663a3b8cee119f6bbc22b826cfb40f94b020dce42720da4c4"
        );
        assert_eq!(
            sha256_hex(&fixture_ovmf(false)),
            "08867049a51e48bf8c6f83145c9f84d5f0efc6fb55bc385e253c4c74ca5fd3eb"
        );
        assert_eq!(
            sha256_hex(&fixture_kernel()),
            "e4a1fd73b7d149051c3bcd370741ea89e9a40f8e0d460227fd7be68645e11ea8"
        );
        assert_eq!(
            sha256_hex(&fixture_initrd()),
            "2df6d2b2c81491ef7d4743ae3d3943bcdc716034c17a26759f483983ab96f708"
        );
        for c in KNOWN {
            assert_eq!(hex(&run(c).expect(c.name)), c.expect, "{}", c.name);
        }
    }

    /// SECURITY: a kernel on a firmware without the hashes section boots
    /// unmeasured; there is no measurement to predict, so none is given
    /// (the reference tool refuses it too: "Kernel specified but OVMF
    /// metadata doesn't include SNP_KERNEL_HASHES section").
    #[test]
    fn a_kernel_on_a_firmware_that_does_not_measure_it_is_refused() {
        let (fw, k) = (fixture_ovmf(false), fixture_kernel());
        let l = Launch {
            ovmf: &fw,
            kernel: Some(&k),
            initrd: None,
            append: None,
            vcpus: 2,
            vcpu_sig: vcpu_signature("EPYC-Genoa").unwrap(),
            guest_features: 1,
        };
        assert_eq!(launch_digest(&l), Err(LaunchError::KernelUnmeasured));
    }

    #[test]
    fn no_vcpu_an_initrd_without_a_kernel_or_a_firmware_that_is_not_one_is_refused() {
        let (fw, i) = (fixture_ovmf(true), fixture_initrd());
        let base = Launch {
            ovmf: &fw,
            kernel: None,
            initrd: None,
            append: None,
            vcpus: 1,
            vcpu_sig: vcpu_signature("EPYC-Genoa").unwrap(),
            guest_features: 1,
        };
        assert_eq!(
            launch_digest(&Launch { vcpus: 0, ..base }),
            Err(LaunchError::Vcpus(0))
        );
        assert_eq!(
            launch_digest(&Launch {
                initrd: Some(&i),
                ..base
            }),
            Err(LaunchError::NoKernel)
        );
        assert_eq!(
            launch_digest(&Launch {
                append: Some("quiet"),
                ..base
            }),
            Err(LaunchError::NoKernel)
        );
        let odd = vec![0u8; PAGE + 1];
        assert!(matches!(
            launch_digest(&Launch { ovmf: &odd, ..base }),
            Err(LaunchError::Firmware(_))
        ));
        // A firmware with no SEV-ES reset block cannot start the APs.
        let mut no_reset = fw.clone();
        let g = guid_le(SEV_ES_RESET_BLOCK);
        let at = no_reset.windows(16).position(|w| w == g).unwrap();
        no_reset[at] ^= 1;
        assert!(matches!(
            launch_digest(&Launch {
                ovmf: &no_reset,
                ..base
            }),
            Err(LaunchError::Firmware(m)) if m.contains("reset block")
        ));
        assert_eq!(vcpu_signature("host"), None);
    }

    /// Every input moves the digest: one flipped firmware byte, the kernel,
    /// the initrd, the command line, the vCPU count, model and features.
    #[test]
    fn every_launch_input_is_in_the_measurement() {
        let (fw, k, i) = (fixture_ovmf(true), fixture_kernel(), fixture_initrd());
        let base = Launch {
            ovmf: &fw,
            kernel: Some(&k),
            initrd: Some(&i),
            append: Some("console=ttyS0"),
            vcpus: 4,
            vcpu_sig: vcpu_signature("EPYC-Genoa").unwrap(),
            guest_features: 1,
        };
        let d0 = launch_digest(&base).unwrap();
        assert_eq!(launch_digest(&base).unwrap(), d0, "deterministic");
        let mut fw2 = fw.clone();
        fw2[0x1234] ^= 1;
        let mut k2 = k.clone();
        k2[7] ^= 1;
        let mut i2 = i.clone();
        i2[7] ^= 1;
        let variants = [
            Launch { ovmf: &fw2, ..base },
            Launch {
                kernel: Some(&k2),
                ..base
            },
            Launch {
                initrd: Some(&i2),
                ..base
            },
            Launch {
                initrd: None,
                ..base
            },
            Launch {
                append: Some("console=ttyS1"),
                ..base
            },
            Launch {
                append: None,
                ..base
            },
            Launch { vcpus: 5, ..base },
            Launch {
                vcpu_sig: vcpu_signature("EPYC-Milan").unwrap(),
                ..base
            },
            Launch {
                guest_features: 0x21,
                ..base
            },
        ];
        for (n, v) in variants.iter().enumerate() {
            assert_ne!(launch_digest(v).unwrap(), d0, "variant {n}");
        }
    }

    /// The two answers the reference tool gave on 2026-10-23 for our AmdSev
    /// firmware and the x0ne-cvm 0.2.3 UKI (x0ne-OS
    /// `tests/fixtures/snp/known-answers.json`), and those it gave for
    /// Debian 13's two OVMF builds. The files are not in this repository
    /// (4 MiB and 54 MiB): `X0NE_SNP_REAL_DIR` holds `OVMF.amdsev.fd`,
    /// `x0ne-cvm-0.2.3.efi`, `OVMF_CODE_4M.fd` and `OVMF.fd`.
    #[test]
    #[ignore = "needs the firmware and UKI files: X0NE_SNP_REAL_DIR"]
    fn real_firmware_answers_of_the_reference_tool() {
        let dir = std::path::PathBuf::from(std::env::var("X0NE_SNP_REAL_DIR").expect("dir"));
        let read = |n: &str| std::fs::read(dir.join(n)).unwrap_or_else(|e| panic!("{n}: {e}"));
        let (amdsev, uki) = (read("OVMF.amdsev.fd"), read("x0ne-cvm-0.2.3.efi"));
        assert_eq!(
            sha256_hex(&amdsev),
            "673a3c284f76ce6b30ea19d94f345d92c39a9bd69b32c4f99bf1478b635cc025"
        );
        assert_eq!(
            sha256_hex(&uki),
            "0d047b869ba04283a3bd3b22cafd998c48d062b800a77aa8ee394b63ca70dc4f"
        );
        let genoa = vcpu_signature("EPYC-Genoa").unwrap();
        let milan = vcpu_signature("EPYC-Milan").unwrap();
        let epyc = vcpu_signature("EPYC").unwrap();
        let digest = |ovmf: &[u8], kernel: Option<&[u8]>, vcpus, vcpu_sig, guest_features| {
            hex(&launch_digest(&Launch {
                ovmf,
                kernel,
                initrd: None,
                append: None,
                vcpus,
                vcpu_sig,
                guest_features,
            })
            .unwrap())
        };
        for (vcpus, want) in [
            (1, "16e1d2eb784607b9c2d6e34ada752821472a1f95417898c3470bb2c471c7b822d46240a4218834ae0563c41581993547"),
            (4, "c937e9ee327a71537590496d2b1c0f4160ef1ffca59372284883cd4eb14b4bda2d2144a9b07af04f74af665c509a47f9"),
        ] {
            assert_eq!(digest(&amdsev, Some(&uki), vcpus, genoa, 1), want, "AmdSev + UKI, {vcpus} vCPU");
        }
        for (file, sha, cases) in [
            (
                "OVMF_CODE_4M.fd",
                "624e06de18b4fa535e90db7160d00d3d07d206422b89999bf1e27d920264e4e0",
                [
                    (1, genoa, 1, "1592c29fb0066d3b723d201d8f74bd8dd8a3f53617ea437e8cb1b85e1e8d629827a1e44c2703d9f24b9687ed7e5d4ece"),
                    (1, milan, 1, "b36682b48bf692d5e4cc99718952e730a070cf604a31dc9b8bc17abc8c5ced34d6df5fd79defdf88bb6057913deb6aa1"),
                    (4, genoa, 1, "224d4b2183939f43b1e0ace3edb7ec959bf9a06787e24d3b5b634a6e1b4a2a80507db6ed33e478f13965c9c79cd55c55"),
                    (4, milan, 1, "dfead3f3e2e6567ca202058330a8600dd2f80660fda9e9f83c6d58f7f088bce3c0d3f609da12015c6f51b6c40a8a0643"),
                    (1, epyc, 0x21, "c26fbf1b3786c03808f35702c475e65dd58128ef85a4ea3a077d685a7210d9173cdf629e64d74067ef8715c5c66c0694"),
                ],
            ),
            (
                "OVMF.fd",
                "14954040cc754ade0c625ac3470ea46d4c14318b5b1396de4ef77090203d76db",
                [
                    (1, genoa, 1, "e16765c33b15fe94ef66a1fb04ff6a25828e11b95ea48c534194f829812c57d16530418db5d54b7ed0573eb9a1ddb3a2"),
                    (1, milan, 1, "1066da73495d1df04d21222f8277ea5e18be8cc33e99ad44e13060292f9320ddf67af6cdd10ccfc22222bb9d6ae3f96c"),
                    (4, genoa, 1, "2ace5a7d2e782e2ac29a6b6f9a13280594bb018ba5131144c4c6f51a8c2c94e3a7c20f92effcb1808bd4252e88d44a61"),
                    (4, milan, 1, "23a7cfea107ad340deb77960c711065bea0527cb551ec5eb3ec8234c730947fe05f172939695b2079efb5c3f841e733a"),
                    (1, epyc, 0x21, "b248ef14a8c3f47ffd3e4d5d0ead2921a72556f625ed24bb9b8d0d3be2bd8dd1230bf774018ea810ad7032da4889ce7f"),
                ],
            ),
        ] {
            let fw = read(file);
            assert_eq!(sha256_hex(&fw), sha, "{file}");
            for (vcpus, sig, features, want) in cases {
                assert_eq!(digest(&fw, None, vcpus, sig, features), want, "{file} {vcpus} {sig:#x} {features:#x}");
            }
        }
    }

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    /// The command line reads the files QEMU would and prints the same
    /// digest as the library, which is the reference tool's.
    #[test]
    fn snp_measure_reads_the_files_and_prints_the_reference_answer() {
        let dir = tempfile::tempdir().unwrap();
        let p = |n: &str| dir.path().join(n).to_string_lossy().into_owned();
        std::fs::write(p("ovmf.fd"), fixture_ovmf(true)).unwrap();
        std::fs::write(p("kernel"), fixture_kernel()).unwrap();
        let a = parse_measure_args(&args(&[
            "--ovmf",
            &p("ovmf.fd"),
            "--kernel",
            &p("kernel"),
            "--vcpus",
            "4",
        ]))
        .unwrap();
        assert_eq!(a.vcpu_type, "EPYC-Genoa");
        assert_eq!(a.guest_features, 1);
        assert_eq!(
            measure_files(&a).unwrap(),
            format!("snp_measurement={}", KNOWN[1].expect),
            "genoa_4vcpu_kernel_only"
        );
        let a = parse_measure_args(&args(&[
            "--vcpus",
            "2",
            "--vcpu-type",
            "EPYC-Milan",
            "--append",
            "quiet",
            "--kernel",
            &p("kernel"),
            "--ovmf",
            &p("ovmf.fd"),
        ]))
        .unwrap();
        assert_eq!(
            measure_files(&a).unwrap(),
            format!("snp_measurement={}", KNOWN[2].expect)
        );
        let missing = MeasureArgs {
            kernel: Some(dir.path().join("absent")),
            ..a
        };
        assert!(measure_files(&missing).unwrap_err().contains("absent"));
    }

    #[test]
    fn snp_measure_refuses_what_qemu_would_not_be_given() {
        for (a, why) in [
            (vec!["--vcpus", "1"], "--ovmf is required"),
            (vec!["--ovmf", "f"], "--vcpus is required"),
            (vec!["--ovmf", "f", "--vcpus", "x"], "a number"),
            (
                vec!["--ovmf", "f", "--vcpus", "1", "--vcpu-type", "host"],
                "named EPYC",
            ),
            (
                vec!["--ovmf", "f", "--vcpus", "1", "--guest-features", "zz"],
                "hexadecimal",
            ),
            (vec!["--ovmf", "f", "--ovmf", "g", "--vcpus", "1"], "twice"),
            (vec!["--ovmf"], "needs a value"),
            (vec!["--smp", "4"], "unknown argument"),
        ] {
            let e = parse_measure_args(&args(&a)).unwrap_err();
            assert!(e.contains(why), "{a:?}: {e}");
        }
        let a = parse_measure_args(&args(&[
            "--ovmf",
            "f",
            "--vcpus",
            "1",
            "--guest-features",
            "0x21",
        ]))
        .unwrap();
        assert_eq!(a.guest_features, 0x21);
    }

    /// Writes the fixtures for the reference tool: `X0NE_SNP_FIXTURE_DIR=<dir>
    /// cargo test -p x0ne-infer snp_launch -- --ignored fixture_files`.
    #[test]
    #[ignore = "writes files for the reference tool"]
    fn fixture_files_for_the_reference_tool() {
        let dir = std::path::PathBuf::from(std::env::var("X0NE_SNP_FIXTURE_DIR").expect("dir"));
        std::fs::write(dir.join("ovmf-hashes.fd"), fixture_ovmf(true)).unwrap();
        std::fs::write(dir.join("ovmf-plain.fd"), fixture_ovmf(false)).unwrap();
        std::fs::write(dir.join("kernel.bin"), fixture_kernel()).unwrap();
        std::fs::write(dir.join("initrd.img"), fixture_initrd()).unwrap();
        for c in KNOWN {
            println!("{} {}", c.name, hex(&run(c).expect(c.name)));
        }
    }
}
