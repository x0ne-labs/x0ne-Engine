//! GPU evidence collection over NVML — the `nvml` feature.
//!
//! Loads `libnvidia-ml.so.1` at runtime (`libloading`, no link-time
//! dependency: the binary must still run on a CPU-only CVM where the gate
//! refuses cleanly instead of failing to start). Every symbol is resolved
//! eagerly in [`NvmlLib::open`] so a driver too old for confidential
//! computing fails THERE, named, before any attestation is attempted.
//!
//! Struct layouts and constants are copied from nvidia-ml-py 13.610.43
//! (pynvml.py:2868-2918, extracted 2026-08-20) and the symbols were checked
//! against a real driver (`nm -D libnvidia-ml.so.610.57.04`, same day).
//! The compile-time size asserts below hold the layout; the
//! `the_local_driver_resolves_the_cc_symbols` test holds the symbol names.
//!
//! HARDWARE STATUS: symbol resolution and the error path are measured on a
//! consumer GPU (RTX 5090 — CC calls answer NotSupported); the success path
//! needs an H100/H200 in CC mode and runs on that hardware, never simulated.
//! `nvmlDeviceGetArchitecture` (2026-10-30) is pinned to nvml.h of CUDA 13.3
//! and not yet called on a driver: the next H100 run reads it.

use crate::nras_client::GpuEvidence;
use base64::Engine as _;

pub const NVML_GPU_CERT_CHAIN_SIZE: usize = 0x1000;
pub const NVML_GPU_ATTESTATION_CERT_CHAIN_SIZE: usize = 0x1400;
pub const NVML_CC_GPU_CEC_NONCE_SIZE: usize = 0x20;
pub const NVML_CC_GPU_ATTESTATION_REPORT_SIZE: usize = 0x2000;
pub const NVML_CC_GPU_CEC_ATTESTATION_REPORT_SIZE: usize = 0x1000;

/// `nvmlConfComputeGpuAttestationReport_t`. The nonce is INPUT (echoed into
/// the SPDM report — that echo is what `nras.rs` checks); the rest is output.
#[repr(C)]
pub struct AttestationReport {
    pub is_cec_attestation_report_present: u32,
    pub attestation_report_size: u32,
    pub cec_attestation_report_size: u32,
    pub nonce: [u8; NVML_CC_GPU_CEC_NONCE_SIZE],
    pub attestation_report: [u8; NVML_CC_GPU_ATTESTATION_REPORT_SIZE],
    pub cec_attestation_report: [u8; NVML_CC_GPU_CEC_ATTESTATION_REPORT_SIZE],
}

/// `nvmlConfComputeGpuCertificate_t`. NRAS wants `attestation_cert_chain`
/// (the device identity chain the report is signed under), not `cert_chain`.
#[repr(C)]
pub struct GpuCertificate {
    pub cert_chain_size: u32,
    pub attestation_cert_chain_size: u32,
    pub cert_chain: [u8; NVML_GPU_CERT_CHAIN_SIZE],
    pub attestation_cert_chain: [u8; NVML_GPU_ATTESTATION_CERT_CHAIN_SIZE],
}

// SECURITY: the driver writes through these pointers with ITS idea of the
// layout. A size mismatch is memory corruption, not a wrong answer — so the
// layout is pinned at compile time against the sizes computed from the
// pynvml field list.
const _: () = assert!(std::mem::size_of::<AttestationReport>() == 12 + 0x20 + 0x2000 + 0x1000);
const _: () = assert!(std::mem::size_of::<GpuCertificate>() == 8 + 0x1000 + 0x1400);

type NvmlDevice = *mut std::ffi::c_void;
type FnInit = unsafe extern "C" fn() -> u32;
type FnShutdown = unsafe extern "C" fn() -> u32;
type FnDeviceCount = unsafe extern "C" fn(*mut u32) -> u32;
type FnDeviceByIndex = unsafe extern "C" fn(u32, *mut NvmlDevice) -> u32;
type FnAttestationReport = unsafe extern "C" fn(NvmlDevice, *mut AttestationReport) -> u32;
type FnCertificate = unsafe extern "C" fn(NvmlDevice, *mut GpuCertificate) -> u32;
type FnGpusReadyState = unsafe extern "C" fn(*mut u32) -> u32;
type FnConfComputeState = unsafe extern "C" fn(*mut ConfComputeSystemState) -> u32;
/// `nvmlDeviceGetArchitecture(nvmlDevice_t, nvmlDeviceArchitecture_t*)`,
/// the second an `unsigned int` (nvml.h of CUDA 13.3).
type FnDeviceArchitecture = unsafe extern "C" fn(NvmlDevice, *mut u32) -> u32;

/// `nvmlConfComputeSystemState_t`: three `unsigned int`, in this order
/// (<https://docs.nvidia.com/deploy/nvml-api/api/structnvmlConfComputeSystemState__t.html>;
/// nvidia-ml-py 12.570.86, `c_nvmlConfComputeSystemState_t`).
#[repr(C)]
pub struct ConfComputeSystemState {
    pub environment: u32,
    pub cc_feature: u32,
    pub dev_tools_mode: u32,
}
const _: () = assert!(std::mem::size_of::<ConfComputeSystemState>() == 12);

/// The handful of NVML return codes we can name; the rest stay numeric.
fn nvml_error(code: u32) -> String {
    let name = match code {
        1 => " (Uninitialized)",
        2 => " (InvalidArgument)",
        3 => " (NotSupported — not a CC GPU, or CC mode off)",
        6 => " (NotFound)",
        999 => " (Unknown)",
        _ => "",
    };
    format!("NVML error {code}{name}")
}

/// The library, with every needed symbol already resolved.
pub struct NvmlLib {
    // Order matters: `Library` must outlive the raw fn pointers, and drop
    // order is declaration order — the pointers are Copy and die with the
    // struct, the library unloads last.
    init: FnInit,
    shutdown: FnShutdown,
    device_count: FnDeviceCount,
    device_by_index: FnDeviceByIndex,
    attestation_report: FnAttestationReport,
    certificate: FnCertificate,
    lib: libloading::Library,
}

impl NvmlLib {
    /// dlopen `libnvidia-ml.so.1` and resolve all six symbols, eagerly.
    pub fn open() -> anyhow::Result<Self> {
        // SAFETY: loading the vendor's own management library by its
        // canonical soname; no init routine runs beyond the loader's.
        let lib = unsafe { libloading::Library::new("libnvidia-ml.so.1") }
            .map_err(|e| anyhow::anyhow!("libnvidia-ml.so.1: {e}"))?;
        macro_rules! sym {
            ($name:literal, $ty:ty) => {
                // SAFETY: the signature is pinned above against pynvml's
                // ctypes declaration for the same symbol.
                *unsafe { lib.get::<$ty>($name) }.map_err(|e| {
                    anyhow::anyhow!("symbol {}: {e}", String::from_utf8_lossy($name))
                })?
            };
        }
        Ok(Self {
            init: sym!(b"nvmlInit_v2", FnInit),
            shutdown: sym!(b"nvmlShutdown", FnShutdown),
            device_count: sym!(b"nvmlDeviceGetCount_v2", FnDeviceCount),
            device_by_index: sym!(b"nvmlDeviceGetHandleByIndex_v2", FnDeviceByIndex),
            attestation_report: sym!(
                b"nvmlDeviceGetConfComputeGpuAttestationReport",
                FnAttestationReport
            ),
            certificate: sym!(b"nvmlDeviceGetConfComputeGpuCertificate", FnCertificate),
            lib,
        })
    }

    /// Collect (report, certificate) from every GPU, the same 32-byte nonce
    /// in each report. Any failure on any GPU refuses the whole collection:
    /// a machine with one attestable GPU out of eight is not attested.
    /// Private: callers go through `attest_v2::judged_gpu_evidence`, which
    /// judges the CC state and the architecture first.
    fn collect_gpu_evidence(&self, nonce: &[u8; 32]) -> anyhow::Result<Vec<GpuEvidence>> {
        // SAFETY: init/shutdown bracket every NVML call below; a failed init
        // returns before any device call.
        let rc = unsafe { (self.init)() };
        if rc != 0 {
            anyhow::bail!("nvmlInit_v2: {}", nvml_error(rc));
        }
        let result = self.collect_inner(nonce);
        // SAFETY: paired with the successful init above.
        unsafe { (self.shutdown)() };
        result
    }

    /// `nvmlSystemGetConfComputeGpusReadyState`: 1 when the CC GPUs accept
    /// client requests (`cloud_serve` refuses to load before that, exit 79).
    /// Resolved here, not in `open()`: a driver without it still attests.
    pub fn gpus_ready_state(&self) -> anyhow::Result<u32> {
        // SAFETY: the signature is pynvml's for the same symbol
        // (`nvmlSystemGetConfComputeGpusReadyState(c_uint*)`), one out-param.
        let f = unsafe {
            self.lib
                .get::<FnGpusReadyState>(b"nvmlSystemGetConfComputeGpusReadyState")
        }
        .map_err(|e| anyhow::anyhow!("symbol nvmlSystemGetConfComputeGpusReadyState: {e}"))?;
        // SAFETY: init/shutdown bracket the call, as in `collect_gpu_evidence`.
        let rc = unsafe { (self.init)() };
        if rc != 0 {
            anyhow::bail!("nvmlInit_v2: {}", nvml_error(rc));
        }
        let mut state: u32 = u32::MAX;
        let rc = unsafe { f(&mut state) };
        unsafe { (self.shutdown)() };
        if rc != 0 {
            anyhow::bail!("nvmlSystemGetConfComputeGpusReadyState: {}", nvml_error(rc));
        }
        Ok(state)
    }

    /// `nvmlSystemGetConfComputeState`: the system's CC environment, feature
    /// and devtools mode. Resolved here, not in `open()`, as the ready state:
    /// a driver without it gives no e2e GPU evidence (the caller refuses).
    pub fn cc_system_state(&self) -> anyhow::Result<crate::attest_v2::CcSystemState> {
        // SAFETY: the signature is nvml.h's and pynvml's for the same symbol
        // (`nvmlSystemGetConfComputeState(nvmlConfComputeSystemState_t*)`).
        let f = unsafe {
            self.lib
                .get::<FnConfComputeState>(b"nvmlSystemGetConfComputeState")
        }
        .map_err(|e| anyhow::anyhow!("symbol nvmlSystemGetConfComputeState: {e}"))?;
        // SAFETY: init/shutdown bracket the call, as in `collect_gpu_evidence`.
        let rc = unsafe { (self.init)() };
        if rc != 0 {
            anyhow::bail!("nvmlInit_v2: {}", nvml_error(rc));
        }
        // Out of range on purpose: a call that writes nothing is refused.
        let mut state = ConfComputeSystemState {
            environment: u32::MAX,
            cc_feature: u32::MAX,
            dev_tools_mode: u32::MAX,
        };
        // SAFETY: valid out-pointer, layout pinned by the const assert.
        let rc = unsafe { f(&mut state) };
        unsafe { (self.shutdown)() };
        if rc != 0 {
            anyhow::bail!("nvmlSystemGetConfComputeState: {}", nvml_error(rc));
        }
        Ok(crate::attest_v2::CcSystemState {
            environment: state.environment,
            cc_feature: state.cc_feature,
            dev_tools_mode: state.dev_tools_mode,
        })
    }

    /// `nvmlDeviceGetArchitecture` of every GPU, in index order. Resolved
    /// here, not in `open()`, as the CC state.
    pub fn architectures(&self) -> anyhow::Result<Vec<u32>> {
        // SAFETY: the signature is nvml.h's for the same symbol.
        let f = unsafe {
            self.lib
                .get::<FnDeviceArchitecture>(b"nvmlDeviceGetArchitecture")
        }
        .map_err(|e| anyhow::anyhow!("symbol nvmlDeviceGetArchitecture: {e}"))?;
        // SAFETY: init/shutdown bracket the calls, as in `collect_gpu_evidence`.
        let rc = unsafe { (self.init)() };
        if rc != 0 {
            anyhow::bail!("nvmlInit_v2: {}", nvml_error(rc));
        }
        let result = self.architectures_inner(*f);
        unsafe { (self.shutdown)() };
        result
    }

    fn architectures_inner(&self, f: FnDeviceArchitecture) -> anyhow::Result<Vec<u32>> {
        let count = self.device_count_bounded()?;
        let mut out = Vec::with_capacity(count as usize);
        for idx in 0..count {
            let mut device: NvmlDevice = std::ptr::null_mut();
            // SAFETY: valid out-pointer, index < count.
            let rc = unsafe { (self.device_by_index)(idx, &mut device) };
            if rc != 0 {
                anyhow::bail!("GPU {idx}: handle: {}", nvml_error(rc));
            }
            // Out of range on purpose: a call that writes nothing is refused.
            let mut arch: u32 = u32::MAX - 1;
            // SAFETY: device from a successful handle call, valid out-pointer.
            let rc = unsafe { f(device, &mut arch) };
            if rc != 0 {
                anyhow::bail!("GPU {idx}: architecture: {}", nvml_error(rc));
            }
            out.push(arch);
        }
        Ok(out)
    }

    /// `nvmlDeviceGetCount_v2`, bounded by refusal.
    fn device_count_bounded(&self) -> anyhow::Result<u32> {
        let mut count: u32 = 0;
        // SAFETY: valid out-pointer, library initialized.
        let rc = unsafe { (self.device_count)(&mut count) };
        if rc != 0 {
            anyhow::bail!("nvmlDeviceGetCount_v2: {}", nvml_error(rc));
        }
        // SECURITY: `count` sizes an allocation and comes from the driver —
        // the same discipline as `package_evidence`: bound by refusal, never
        // trusted raw. NVML tops out at 64 devices; a larger count is a
        // lying or mismatched library, and feeding it to `with_capacity`
        // would abort the process instead of refusing with a name.
        const MAX_GPUS: u32 = 64;
        if count == 0 || count > MAX_GPUS {
            anyhow::bail!("NVML reports {count} GPUs (expected 1..={MAX_GPUS})");
        }
        Ok(count)
    }

    fn collect_inner(&self, nonce: &[u8; 32]) -> anyhow::Result<Vec<GpuEvidence>> {
        let count = self.device_count_bounded()?;
        let mut out = Vec::with_capacity(count as usize);
        for idx in 0..count {
            let mut device: NvmlDevice = std::ptr::null_mut();
            // SAFETY: valid out-pointer, index < count.
            let rc = unsafe { (self.device_by_index)(idx, &mut device) };
            if rc != 0 {
                anyhow::bail!("GPU {idx}: handle: {}", nvml_error(rc));
            }
            // SAFETY: zeroed is a valid value — all fields are integers and
            // byte arrays. Boxed: the report is ~12 KiB.
            let mut report: Box<AttestationReport> = unsafe { Box::new(std::mem::zeroed()) };
            report.nonce = *nonce;
            // SAFETY: device from a successful handle call, struct layout
            // pinned by the const asserts.
            let rc = unsafe { (self.attestation_report)(device, &mut *report) };
            if rc != 0 {
                anyhow::bail!("GPU {idx}: attestation report: {}", nvml_error(rc));
            }
            // SAFETY: same as above.
            let mut cert: Box<GpuCertificate> = unsafe { Box::new(std::mem::zeroed()) };
            let rc = unsafe { (self.certificate)(device, &mut *cert) };
            if rc != 0 {
                anyhow::bail!("GPU {idx}: certificate: {}", nvml_error(rc));
            }
            out.push(package_evidence(
                &report.attestation_report,
                report.attestation_report_size,
                &cert.attestation_cert_chain,
                cert.attestation_cert_chain_size,
            )?);
        }
        Ok(out)
    }
}

impl crate::attest_v2::LocalGpus for NvmlLib {
    fn cc_system_state(&self) -> anyhow::Result<crate::attest_v2::CcSystemState> {
        NvmlLib::cc_system_state(self)
    }
    fn architectures(&self) -> anyhow::Result<Vec<u32>> {
        NvmlLib::architectures(self)
    }
    fn read_evidence(&self, nonce: &[u8; 32]) -> anyhow::Result<Vec<GpuEvidence>> {
        self.collect_gpu_evidence(nonce)
    }
}

/// Trim the fixed buffers to their declared sizes and base64 them, the shape
/// NRAS wants. Pure, and the only place the driver's declared sizes are
/// trusted — bounded here, by refusal, never by clamping.
///
/// SECURITY: a size larger than the buffer is a driver lying about its own
/// ABI; serving a truncated report instead of refusing would hand NRAS
/// something that fails verification *sometimes* instead of never.
pub fn package_evidence(
    report: &[u8],
    report_size: u32,
    chain: &[u8],
    chain_size: u32,
) -> anyhow::Result<GpuEvidence> {
    let b64 = base64::engine::general_purpose::STANDARD;
    let report = report.get(..report_size as usize).ok_or_else(|| {
        anyhow::anyhow!("report size {report_size} exceeds buffer {}", report.len())
    })?;
    let chain = chain.get(..chain_size as usize).ok_or_else(|| {
        anyhow::anyhow!(
            "cert chain size {chain_size} exceeds buffer {}",
            chain.len()
        )
    })?;
    if report.is_empty() || chain.is_empty() {
        anyhow::bail!(
            "empty GPU evidence (report {} bytes, chain {} bytes)",
            report.len(),
            chain.len()
        );
    }
    Ok(GpuEvidence {
        evidence: b64.encode(report),
        certificate: b64.encode(chain),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A driver whose declared sizes exceed the fixed ABI buffers is refused,
    /// not clamped. A failure here would mean evidence silently truncated
    /// before it reaches NRAS.
    #[test]
    fn a_lying_driver_size_is_refused_not_clamped() {
        let report = [1u8; 16];
        let chain = [2u8; 16];
        assert!(package_evidence(&report, 17, &chain, 8).is_err());
        assert!(package_evidence(&report, 8, &chain, 17).is_err());
        assert!(package_evidence(&report, 0, &chain, 8).is_err());
        assert!(package_evidence(&report, 8, &chain, 0).is_err());
    }

    /// The happy path base64s exactly the declared prefix of each buffer.
    #[test]
    fn evidence_is_the_declared_prefix_base64() {
        let mut report = [0u8; 16];
        report[..4].copy_from_slice(b"SPDM");
        let mut chain = [0u8; 16];
        chain[..3].copy_from_slice(b"PEM");
        let ev = package_evidence(&report, 4, &chain, 3).unwrap();
        let b64 = base64::engine::general_purpose::STANDARD;
        assert_eq!(b64.decode(&ev.evidence).unwrap(), b"SPDM");
        assert_eq!(b64.decode(&ev.certificate).unwrap(), b"PEM");
    }

    /// The six symbols this module needs exist in the installed driver.
    /// Skips (with a note) on a machine without libnvidia-ml — the CVM
    /// integration runs on real hardware. A failure on a machine WITH the library means the
    /// driver is too old for confidential computing.
    #[test]
    fn the_local_driver_resolves_the_cc_symbols() {
        match NvmlLib::open() {
            Ok(_) => {}
            Err(e) if e.to_string().contains("libnvidia-ml.so.1") => {
                eprintln!("skipped: no NVIDIA driver on this machine ({e})");
            }
            Err(e) => panic!("driver present but a CC symbol is missing: {e}"),
        }
    }
}
