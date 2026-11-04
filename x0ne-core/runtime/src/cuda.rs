//! CUDA driver API and cuBLAS, opened with dlopen at run time. The build links
//! neither, so a host without a GPU builds and runs the CPU tests, and gets an
//! error (not a loader abort) when it asks for a device.

use crate::error::{Error, Result};
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::sync::OnceLock;

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

const RTLD_NOW: c_int = 2;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUdeviceptr = u64;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;
pub type CUgraphNode = *mut c_void;
pub type CUevent = *mut c_void;
pub type CublasHandle = *mut c_void;
pub type CublasStatus = c_int;

pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
pub const CU_STREAM_NON_BLOCKING: c_uint = 1;
/// Capture mode GLOBAL: any unsafe API call from any thread during the capture fails it.
pub const CU_STREAM_CAPTURE_MODE_GLOBAL: c_uint = 0;
pub const CU_EVENT_DISABLE_TIMING: c_uint = 2;

pub const CUBLAS_OP_N: c_int = 0;
pub const CUBLAS_OP_T: c_int = 1;
pub const CUDA_R_32F: c_int = 0;
pub const CUDA_R_16BF: c_int = 14;
pub const CUBLAS_COMPUTE_32F: c_int = 68;
pub const CUBLAS_GEMM_DEFAULT: c_int = -1;
pub const CUBLAS_DEFAULT_MATH: c_int = 0;

fn last_dl_error() -> String {
    // SAFETY: dlerror returns a thread-local C string or null.
    let p = unsafe { dlerror() };
    if p.is_null() {
        "unknown dlerror".into()
    } else {
        // SAFETY: non-null; the string stays valid until the next dl call on this thread.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

fn open(name: &str) -> std::result::Result<*mut c_void, String> {
    let c = CString::new(name).expect("library name");
    // SAFETY: dlopen with a valid C string; the handle is never closed (process lifetime).
    let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
    if h.is_null() {
        Err(format!("dlopen {name}: {}", last_dl_error()))
    } else {
        Ok(h)
    }
}

/// Symbol `name` of library `h`, or dlerror's reason (`resolve` adds the symbol's name).
fn sym(h: *mut c_void, name: &str) -> std::result::Result<*mut c_void, String> {
    let c = CString::new(name).expect("symbol name");
    // SAFETY: valid handle from `open` and a valid C string.
    let p = unsafe { dlsym(h, c.as_ptr()) };
    if p.is_null() {
        Err(last_dl_error())
    } else {
        Ok(p)
    }
}

// Every symbol is required: a library that lacks one is refused at load.
macro_rules! api {
    ($ty:ident { $( $field:ident = $sym:literal : fn($($arg:ty),*) -> $ret:ty; )* }) => {
        #[allow(non_snake_case)]
        pub struct $ty {
            $( pub $field: unsafe extern "C" fn($($arg),*) -> $ret, )*
        }
        impl $ty {
            fn load(h: *mut c_void) -> std::result::Result<Self, String> {
                Self::resolve(|name| sym(h, name))
            }

            /// Every symbol through `get`; the first one missing is an error naming it.
            fn resolve(
                get: impl Fn(&str) -> std::result::Result<*mut c_void, String>,
            ) -> std::result::Result<Self, String> {
                let need = |name: &str| get(name).map_err(|e| format!("dlsym {name}: {e}"));
                // SAFETY: each symbol is cast to the signature of the CUDA 13 headers.
                Ok($ty { $( $field: unsafe {
                    std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) -> $ret>(need($sym)?)
                }, )* })
            }
        }
    };
}

api!(Driver {
    init = "cuInit": fn(c_uint) -> CUresult;
    driver_version = "cuDriverGetVersion": fn(*mut c_int) -> CUresult;
    device_get = "cuDeviceGet": fn(*mut CUdevice, c_int) -> CUresult;
    device_attribute = "cuDeviceGetAttribute": fn(*mut c_int, c_int, CUdevice) -> CUresult;
    device_name = "cuDeviceGetName": fn(*mut c_char, c_int, CUdevice) -> CUresult;
    primary_ctx_retain = "cuDevicePrimaryCtxRetain": fn(*mut CUcontext, CUdevice) -> CUresult;
    primary_ctx_release = "cuDevicePrimaryCtxRelease_v2": fn(CUdevice) -> CUresult;
    ctx_set_current = "cuCtxSetCurrent": fn(CUcontext) -> CUresult;
    module_load_data = "cuModuleLoadData": fn(*mut CUmodule, *const c_void) -> CUresult;
    module_unload = "cuModuleUnload": fn(CUmodule) -> CUresult;
    module_get_function = "cuModuleGetFunction": fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
    func_get_param_count = "cuFuncGetParamCount": fn(CUfunction, *mut usize) -> CUresult;
    func_get_param_info = "cuFuncGetParamInfo": fn(CUfunction, usize, *mut usize, *mut usize) -> CUresult;
    func_set_attribute = "cuFuncSetAttribute": fn(CUfunction, c_int, c_int) -> CUresult;
    occupancy_max_blocks = "cuOccupancyMaxActiveBlocksPerMultiprocessor": fn(*mut c_int, CUfunction, c_int, usize) -> CUresult;
    mem_alloc = "cuMemAlloc_v2": fn(*mut CUdeviceptr, usize) -> CUresult;
    mem_free = "cuMemFree_v2": fn(CUdeviceptr) -> CUresult;
    mem_get_info = "cuMemGetInfo_v2": fn(*mut usize, *mut usize) -> CUresult;
    memcpy_htod_async = "cuMemcpyHtoDAsync_v2": fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;
    memcpy_dtoh_async = "cuMemcpyDtoHAsync_v2": fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult;
    memset_d8_async = "cuMemsetD8Async": fn(CUdeviceptr, u8, usize, CUstream) -> CUresult;
    memset_d32_async = "cuMemsetD32Async": fn(CUdeviceptr, c_uint, usize, CUstream) -> CUresult;
    stream_create = "cuStreamCreate": fn(*mut CUstream, c_uint) -> CUresult;
    stream_destroy = "cuStreamDestroy_v2": fn(CUstream) -> CUresult;
    stream_synchronize = "cuStreamSynchronize": fn(CUstream) -> CUresult;
    launch_kernel = "cuLaunchKernel": fn(CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, *mut *mut c_void, *mut *mut c_void) -> CUresult;
    error_name = "cuGetErrorName": fn(CUresult, *mut *const c_char) -> CUresult;
    stream_begin_capture = "cuStreamBeginCapture_v2": fn(CUstream, c_uint) -> CUresult;
    stream_end_capture = "cuStreamEndCapture": fn(CUstream, *mut CUgraph) -> CUresult;
    graph_instantiate = "cuGraphInstantiateWithFlags": fn(*mut CUgraphExec, CUgraph, std::ffi::c_ulonglong) -> CUresult;
    graph_launch = "cuGraphLaunch": fn(CUgraphExec, CUstream) -> CUresult;
    graph_exec_destroy = "cuGraphExecDestroy": fn(CUgraphExec) -> CUresult;
    graph_destroy = "cuGraphDestroy": fn(CUgraph) -> CUresult;
    graph_get_nodes = "cuGraphGetNodes": fn(CUgraph, *mut CUgraphNode, *mut usize) -> CUresult;
    event_create = "cuEventCreate": fn(*mut CUevent, c_uint) -> CUresult;
    event_record = "cuEventRecord": fn(CUevent, CUstream) -> CUresult;
    event_synchronize = "cuEventSynchronize": fn(CUevent) -> CUresult;
    event_destroy = "cuEventDestroy_v2": fn(CUevent) -> CUresult;
    mem_host_alloc = "cuMemHostAlloc": fn(*mut *mut c_void, usize, c_uint) -> CUresult;
    mem_free_host = "cuMemFreeHost": fn(*mut c_void) -> CUresult;
});

api!(Blas {
    create = "cublasCreate_v2": fn(*mut CublasHandle) -> CublasStatus;
    destroy = "cublasDestroy_v2": fn(CublasHandle) -> CublasStatus;
    version = "cublasGetVersion_v2": fn(CublasHandle, *mut c_int) -> CublasStatus;
    set_stream = "cublasSetStream_v2": fn(CublasHandle, CUstream) -> CublasStatus;
    set_workspace = "cublasSetWorkspace_v2": fn(CublasHandle, *mut c_void, usize) -> CublasStatus;
    set_math_mode = "cublasSetMathMode": fn(CublasHandle, c_int) -> CublasStatus;
    gemm_ex = "cublasGemmEx": fn(CublasHandle, c_int, c_int, c_int, c_int, c_int, *const c_void, *const c_void, c_int, c_int, *const c_void, c_int, c_int, *const c_void, *mut c_void, c_int, c_int, c_int, c_int) -> CublasStatus;
});

/// libcuda comes from the driver (CDI in a container); libcublas from the CUDA 13 runtime.
pub const LIBCUDA: &str = "libcuda.so.1";
pub const LIBCUBLAS: &str = "libcublas.so.13";

static DRIVER: OnceLock<std::result::Result<Driver, String>> = OnceLock::new();
static BLAS: OnceLock<std::result::Result<Blas, String>> = OnceLock::new();

pub fn driver() -> Result<&'static Driver> {
    DRIVER
        .get_or_init(|| open(LIBCUDA).and_then(Driver::load))
        .as_ref()
        .map_err(|e| Error::Gpu(e.clone()))
}

pub fn blas() -> Result<&'static Blas> {
    BLAS.get_or_init(|| open(LIBCUBLAS).and_then(Blas::load))
        .as_ref()
        .map_err(|e| Error::Gpu(e.clone()))
}

/// Turns a driver status into a Result, with the driver's error name.
pub fn check(r: CUresult, what: &str) -> Result<()> {
    if r == 0 {
        return Ok(());
    }
    let mut name: *const c_char = std::ptr::null();
    // SAFETY: cuGetErrorName writes a pointer to a static string into the live local `name`.
    let named = driver()
        .ok()
        .map(|d| unsafe { (d.error_name)(r, &mut name) } == 0 && !name.is_null())
        == Some(true);
    let text = if named {
        // SAFETY: `name` is non-null and points to a static C string (checked above).
        unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned()
    } else {
        "?".into()
    };
    Err(Error::Gpu(format!("{what}: CUresult {r} ({text})")))
}

pub fn check_blas(r: CublasStatus, what: &str) -> Result<()> {
    if r == 0 {
        Ok(())
    } else {
        Err(Error::Gpu(format!("{what}: cublasStatus {r}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A library that is not there is an error carrying dlopen's reason, not a loader abort.
    #[test]
    fn a_missing_library_is_an_error() {
        let e = open("libx0ne-core-absent.so.1").unwrap_err();
        assert!(e.starts_with("dlopen libx0ne-core-absent.so.1: "), "{e}");
    }

    /// A libcuda without cuFuncGetParamCount is refused at load, with an error naming it: the
    /// parameter count of each kernel is checked against the cubin with no fallback.
    #[test]
    fn a_driver_without_cu_func_get_param_count_is_an_error() {
        // Never called: `resolve` only stores the addresses.
        let fake = std::ptr::NonNull::<c_void>::dangling().as_ptr();
        assert!(Driver::resolve(|_| Ok(fake)).is_ok());
        let e = Driver::resolve(|name| match name {
            "cuFuncGetParamCount" => Err("undefined symbol".into()),
            _ => Ok(fake),
        })
        .err()
        .expect("a driver without cuFuncGetParamCount was loaded");
        assert_eq!(e, "dlsym cuFuncGetParamCount: undefined symbol");
    }

    /// Without libcuda (a host or container with no NVIDIA driver, as in the CI), `driver()`
    /// and `Gpu::new()` return an error. With libcuda there is nothing to check here.
    #[test]
    fn driver_is_an_error_without_libcuda() {
        if open(LIBCUDA).is_ok() {
            eprintln!("{LIBCUDA} is present: nothing to check");
            return;
        }
        eprintln!("{LIBCUDA} is absent");
        let e = driver().err().expect("driver() without libcuda");
        assert!(e.to_string().contains("dlopen libcuda.so.1"), "{e}");
        assert!(matches!(crate::gpu::Gpu::new(), Err(Error::Gpu(_))));
    }
}
