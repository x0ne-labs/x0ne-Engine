//! Cubins built from `cuda/` by build.rs, one per architecture, embedded in the binary.
//! No PTX is shipped and nothing is compiled at run time.

/// One architecture's cubin. `arch` is the nvcc `--gpu-architecture` value.
#[derive(Debug, Clone, Copy)]
pub struct Cubin {
    pub arch: &'static str,
    pub major: u32,
    pub minor: u32,
    pub bytes: &'static [u8],
}

macro_rules! cubin {
    ($arch:literal, $major:literal, $minor:literal) => {
        Cubin {
            arch: $arch,
            major: $major,
            minor: $minor,
            bytes: include_bytes!(concat!(env!("OUT_DIR"), "/x0ne_kernels.", $arch, ".cubin")),
        }
    };
}

/// Same order as `ARCHS` in build.rs; the set hash depends on it.
pub const CUBINS: &[Cubin] = &[
    cubin!("sm_90a", 9, 0),
    cubin!("sm_100a", 10, 0),
    cubin!("sm_120a", 12, 0),
];

/// sha256 over (arch, length as u64 LE, bytes) of every cubin, in `CUBINS` order.
/// Meant for the receipt field "kernel set hash".
pub const KERNEL_SET_SHA256: &str = env!("X0NE_CORE_KERNEL_SET_SHA256");

/// The `nvcc --version` release line of the compiler that produced `CUBINS`.
pub const NVCC_RELEASE: &str = env!("X0NE_CORE_NVCC_RELEASE");

/// Every kernel symbol the cubins must export.
pub const KERNEL_NAMES: &[&str] = &[
    "x0ne_probe_iota_u32",
    "x0ne_embedding_bf16",
    "x0ne_add_rmsnorm_bf16",
    "x0ne_qk_norm_rope_kv_write_bf16",
    "x0ne_attention_decode_bf16",
    "x0ne_attention_decode_split_bf16",
    "x0ne_attention_decode_split_g1_bf16",
    "x0ne_attention_decode_split_g2_bf16",
    "x0ne_attention_decode_split_g4_bf16",
    "x0ne_attention_decode_split_g8_bf16",
    "x0ne_attention_merge_bf16",
    "x0ne_attention_prefill_bf16",
    "x0ne_swiglu_bf16",
    "x0ne_argmax_f32",
    "x0ne_advance_u32",
    "x0ne_moe_route_f32",
    "x0ne_moe_gemv_bf16",
    "x0ne_moe_gemv_f32",
    "x0ne_moe_combine_bf16",
];

/// C type of a kernel parameter in `cuda/x0ne_kernels.h`: a device pointer, `uint32_t` or
/// `float`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Param {
    Ptr,
    U32,
    F32,
}

impl Param {
    /// Bytes the parameter takes in a launch's parameter buffer.
    pub const fn size(self) -> usize {
        match self {
            Param::Ptr => 8,
            Param::U32 | Param::F32 => 4,
        }
    }
}

use Param::{Ptr, F32, U32};

/// Parameters of the grouped split-K kernels: q, k, v, state, counter, out, pos, then q_heads,
/// kv_heads, max_ctx, splits, scale.
const SPLIT_GROUP: &[Param] = &[Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, U32, U32, U32, U32, F32];

/// Parameters of the MoE expert GEMVs: w, ids, x, y, then rows, cols, experts, x_group, x_stride.
const MOE_GEMV: &[Param] = &[Ptr, Ptr, Ptr, Ptr, U32, U32, U32, U32, U32];

/// The parameters of every kernel of `KERNEL_NAMES`, in that order, as `cuda/x0ne_kernels.h`
/// declares them. The runtime compares them with the cubin when it loads it, and every
/// launch's arguments with them.
pub const KERNEL_PARAMS: &[(&str, &[Param])] = &[
    ("x0ne_probe_iota_u32", &[Ptr, U32]),
    ("x0ne_embedding_bf16", &[Ptr, Ptr, Ptr, U32, U32]),
    ("x0ne_add_rmsnorm_bf16", &[Ptr, Ptr, Ptr, Ptr, U32, F32]),
    (
        "x0ne_qk_norm_rope_kv_write_bf16",
        &[Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, U32, U32, U32, F32],
    ),
    (
        "x0ne_attention_decode_bf16",
        &[Ptr, Ptr, Ptr, Ptr, Ptr, U32, U32, U32, F32],
    ),
    (
        "x0ne_attention_decode_split_bf16",
        &[Ptr, Ptr, Ptr, Ptr, Ptr, U32, U32, U32, U32, F32],
    ),
    ("x0ne_attention_decode_split_g1_bf16", SPLIT_GROUP),
    ("x0ne_attention_decode_split_g2_bf16", SPLIT_GROUP),
    ("x0ne_attention_decode_split_g4_bf16", SPLIT_GROUP),
    ("x0ne_attention_decode_split_g8_bf16", SPLIT_GROUP),
    ("x0ne_attention_merge_bf16", &[Ptr, Ptr, Ptr, U32, U32]),
    (
        "x0ne_attention_prefill_bf16",
        &[Ptr, Ptr, Ptr, Ptr, Ptr, U32, U32, U32, U32, F32],
    ),
    ("x0ne_swiglu_bf16", &[Ptr, Ptr, U32]),
    ("x0ne_argmax_f32", &[Ptr, U32, Ptr]),
    ("x0ne_advance_u32", &[Ptr]),
    ("x0ne_moe_route_f32", &[Ptr, U32, U32, Ptr, Ptr]),
    ("x0ne_moe_gemv_bf16", MOE_GEMV),
    ("x0ne_moe_gemv_f32", MOE_GEMV),
    ("x0ne_moe_combine_bf16", &[Ptr, Ptr, Ptr, U32, U32]),
];

/// The declared parameters of kernel `name`.
pub fn params_of(name: &str) -> Option<&'static [Param]> {
    KERNEL_PARAMS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|&(_, p)| p)
}

/// Head size every attention and RoPE kernel is compiled for (`X0NE_HEAD_DIM`).
pub const HEAD_DIM: usize = 128;

/// Keys per tile of the split-K decode attention (`X0NE_SPLIT_TILE`).
pub const SPLIT_TILE: usize = 128;

/// f32 words per split-K partial state: m, l, two unused, o[HEAD_DIM] (`X0NE_SPLIT_STATE`).
pub const SPLIT_STATE: usize = HEAD_DIM + 4;

/// Threads per block of the split-K kernel and of its merge.
pub const SPLIT_BLOCK: u32 = 128;

/// Largest expert count of the MoE routing kernel (`X0NE_MOE_MAX_EXPERTS`); top-k is at most 32.
pub const MOE_MAX_EXPERTS: usize = 1024;

/// Expert index the routing kernel writes for a row with a non-finite router logit
/// (`X0NE_MOE_NO_EXPERT`): the expert kernels then write NaN.
pub const MOE_NO_EXPERT: u32 = u32::MAX;

/// The cubin for an exact compute capability. `a` targets do not run on any other
/// major.minor, so there is no fallback: an unknown GPU is refused.
pub fn cubin_for(major: u32, minor: u32) -> Option<&'static Cubin> {
    CUBINS.iter().find(|c| c.major == major && c.minor == minor)
}

/// Facts read from a cubin's ELF header and section table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CubinInfo {
    pub e_flags: u32,
    pub sections: Vec<String>,
}

/// Checks that `bytes` is a CUDA ELF (EM_CUDA) holding SASS for every kernel in
/// `KERNEL_NAMES`, with no PTX section and no PTX text.
pub fn check_cubin(bytes: &[u8]) -> Result<CubinInfo, String> {
    const EM_CUDA: u16 = 190;
    if bytes.len() < 64 || &bytes[..4] != b"\x7fELF" {
        return Err("not an ELF file".into());
    }
    if bytes[4] != 2 || bytes[5] != 1 {
        return Err("not a 64-bit little-endian ELF".into());
    }
    let u16_at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    if u16_at(18) != EM_CUDA {
        return Err(format!("e_machine {} is not EM_CUDA", u16_at(18)));
    }
    let e_flags = u32_at(48);
    let shoff = usize::try_from(u64_at(40)).map_err(|_| "e_shoff overflows")?;
    let shentsize = usize::from(u16_at(58));
    let shnum = usize::from(u16_at(60));
    let shstrndx = usize::from(u16_at(62));
    if shentsize != 64 || shstrndx >= shnum {
        return Err("unexpected section table layout".into());
    }
    let table_end = shnum.checked_mul(64).and_then(|n| n.checked_add(shoff));
    if table_end.is_none_or(|end| end > bytes.len()) {
        return Err("section table out of bounds".into());
    }
    let header = |i: usize| shoff + i * 64;
    let strtab_off =
        usize::try_from(u64_at(header(shstrndx) + 24)).map_err(|_| "offset overflows")?;
    let strtab_len =
        usize::try_from(u64_at(header(shstrndx) + 32)).map_err(|_| "size overflows")?;
    let strtab = bytes
        .get(strtab_off..strtab_off.saturating_add(strtab_len))
        .ok_or("section name table out of bounds")?;
    let mut sections = Vec::with_capacity(shnum);
    for i in 0..shnum {
        let name_off = u32_at(header(i)) as usize;
        let rest = strtab.get(name_off..).ok_or("section name out of bounds")?;
        let end = rest
            .iter()
            .position(|&b| b == 0)
            .ok_or("unterminated section name")?;
        sections.push(String::from_utf8_lossy(&rest[..end]).into_owned());
    }
    if let Some(s) = sections
        .iter()
        .find(|s| s.to_ascii_lowercase().contains("ptx"))
    {
        return Err(format!("PTX section present: {s}"));
    }
    let marker = b"Generated by NVIDIA NVVM Compiler";
    if bytes.windows(marker.len()).any(|w| w == marker) {
        return Err("PTX text present".into());
    }
    for name in KERNEL_NAMES {
        let text = format!(".text.{name}");
        if !sections.contains(&text) {
            return Err(format!("no SASS section {text}"));
        }
    }
    Ok(CubinInfo { e_flags, sections })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn every_cubin_is_sass_only_and_exports_every_kernel() {
        for c in CUBINS {
            let info = check_cubin(c.bytes).unwrap_or_else(|e| panic!("{}: {e}", c.arch));
            assert!(!info.sections.is_empty(), "{}", c.arch);
        }
    }

    #[test]
    fn kernel_set_hash_matches_the_embedded_bytes() {
        let mut h = Sha256::new();
        for c in CUBINS {
            h.update(c.arch.as_bytes());
            h.update((c.bytes.len() as u64).to_le_bytes());
            h.update(c.bytes);
        }
        assert_eq!(format!("{:x}", h.finalize()), KERNEL_SET_SHA256);
    }

    #[test]
    fn cubins_differ_per_architecture() {
        assert_ne!(CUBINS[0].bytes, CUBINS[1].bytes);
        assert_ne!(CUBINS[1].bytes, CUBINS[2].bytes);
        assert_ne!(CUBINS[0].bytes, CUBINS[2].bytes);
    }

    #[test]
    fn compute_capability_maps_exactly_or_is_refused() {
        assert_eq!(cubin_for(9, 0).unwrap().arch, "sm_90a");
        assert_eq!(cubin_for(10, 0).unwrap().arch, "sm_100a");
        assert_eq!(cubin_for(12, 0).unwrap().arch, "sm_120a");
        for (major, minor) in [(8, 9), (9, 1), (10, 3), (12, 1), (11, 0)] {
            assert!(
                cubin_for(major, minor).is_none(),
                "{major}.{minor} must be refused"
            );
        }
    }

    #[test]
    fn check_cubin_rejects_non_cuda_and_ptx() {
        assert!(
            check_cubin(b"not an elf at all, padded to sixty-four bytes .................")
                .is_err()
        );
        let mut host_elf = CUBINS[0].bytes.to_vec();
        host_elf[18] = 62; // EM_X86_64
        assert!(check_cubin(&host_elf).unwrap_err().contains("EM_CUDA"));
        let mut with_ptx = CUBINS[0].bytes.to_vec();
        with_ptx.extend_from_slice(b"//\n// Generated by NVIDIA NVVM Compiler\n.version 9.0\n");
        assert!(check_cubin(&with_ptx).unwrap_err().contains("PTX"));
    }

    /// (name, parameters) of every `extern "C" __global__` declaration of the header.
    fn header_kernels() -> Vec<(String, Vec<Param>)> {
        let text = include_str!("../cuda/x0ne_kernels.h");
        let decls: Vec<&str> = text
            .split("extern \"C\" __global__ void ")
            .skip(1)
            .collect();
        decls
            .iter()
            .map(|d| {
                let open = d.find('(').expect("(");
                let close = d.find(')').expect(")");
                let params = d[open + 1..close]
                    .split(',')
                    .map(|p| {
                        let p = p.split_whitespace().collect::<Vec<_>>().join(" ");
                        if p.contains('*') {
                            Param::Ptr
                        } else if p.starts_with("uint32_t ") {
                            Param::U32
                        } else if p.starts_with("float ") {
                            Param::F32
                        } else {
                            panic!("unknown parameter type: {p}")
                        }
                    })
                    .collect();
                (d[..open].trim().to_string(), params)
            })
            .collect()
    }

    #[test]
    fn declared_parameters_are_the_headers() {
        let header = header_kernels();
        let names: Vec<&str> = header.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names, KERNEL_NAMES,
            "the header declares the kernels of KERNEL_NAMES, in order"
        );
        let declared: Vec<(&str, &[Param])> = KERNEL_PARAMS.to_vec();
        let read: Vec<(&str, &[Param])> = header
            .iter()
            .map(|(n, p)| (n.as_str(), p.as_slice()))
            .collect();
        assert_eq!(declared, read);
        for name in KERNEL_NAMES {
            assert!(params_of(name).is_some(), "{name}");
        }
        assert_eq!(params_of("x0ne_absent"), None);
    }

    #[test]
    fn param_sizes_are_the_c_sizes() {
        assert_eq!(
            (Param::Ptr.size(), Param::U32.size(), Param::F32.size()),
            (8, 4, 4)
        );
    }

    #[test]
    fn nvcc_release_is_recorded() {
        assert!(NVCC_RELEASE.contains("release"), "{NVCC_RELEASE}");
    }
}
