//! One CUDA device: its primary context, one stream, the embedded cubin for its compute
//! capability, and one cuBLAS handle. Device memory comes only from `allocate`, which
//! `Arena::freeze` calls once; everything after that works inside the frozen arena.
//!
//! The calls that take a device address or a host pointer nothing checks (a launch, a GEMM,
//! a memset, an asynchronous copy, the cuBLAS workspace, the replay of a captured graph) are
//! `unsafe`: safe code cannot make them. Each of these does not compile:
//!
//! ```compile_fail
//! fn f(g: &x0ne_core_runtime::gpu::Gpu) -> x0ne_core_runtime::Result<()> {
//!     x0ne_core_runtime::launch!(g, g.f.advance, (1, 1), 32; 0x1000u64)
//! }
//! ```
//!
//! ```compile_fail
//! use x0ne_core_runtime::gpu::{Gpu, Out};
//! fn f(g: &Gpu) -> x0ne_core_runtime::Result<()> {
//!     g.gemm(0x1000, 4096, 4096, 0x2000, 512, 0x3000, Out::F32)
//! }
//! ```
//!
//! ```compile_fail
//! fn f(g: &x0ne_core_runtime::gpu::Gpu) -> x0ne_core_runtime::Result<()> {
//!     g.memset_u32(0x1000, 0, 1 << 30)
//! }
//! ```
//!
//! ```compile_fail
//! fn f(g: &x0ne_core_runtime::gpu::Gpu) -> x0ne_core_runtime::Result<()> {
//!     let mut word = 0u32;
//!     g.copy_to_host_async((&mut word as *mut u32).cast(), 0x1000, 64)
//! }
//! ```
//!
//! ```compile_fail
//! fn f(g: &x0ne_core_runtime::gpu::Gpu) -> x0ne_core_runtime::Result<()> {
//!     let word = 7u32;
//!     g.copy_to_device_async(0x1000, (&word as *const u32).cast(), 64)
//! }
//! ```
//!
//! ```compile_fail
//! fn f(g: &x0ne_core_runtime::gpu::Gpu) -> x0ne_core_runtime::Result<()> {
//!     g.set_cublas_workspace(0x1000, 32 << 20)
//! }
//! ```
//!
//! ```compile_fail
//! use x0ne_core_runtime::gpu::{Gpu, Graph};
//! fn f(g: &Gpu, graph: &Graph) -> x0ne_core_runtime::Result<()> {
//!     g.launch_graph(graph)
//! }
//! ```
//!
//! The same calls compile inside `unsafe`, where the caller states why the addresses hold:
//!
//! ```no_run
//! use x0ne_core_runtime::gpu::{Gpu, Graph, Out};
//! fn f(g: &Gpu, graph: &Graph) -> x0ne_core_runtime::Result<()> {
//!     let mut word = 0u32;
//!     // SAFETY: none, this only shows that the calls compile in an unsafe block.
//!     unsafe {
//!         x0ne_core_runtime::launch!(g, g.f.advance, (1, 1), 32; 0x1000u64)?;
//!         g.gemm(0x1000, 4096, 4096, 0x2000, 512, 0x3000, Out::F32)?;
//!         g.memset_u32(0x1000, 0, 1 << 30)?;
//!         g.copy_to_host_async((&mut word as *mut u32).cast(), 0x1000, 64)?;
//!         g.copy_to_device_async(0x1000, (&word as *const u32).cast(), 64)?;
//!         g.set_cublas_workspace(0x1000, 32 << 20)?;
//!         g.launch_graph(graph)
//!     }
//! }
//! ```

use crate::arena::{Arena, Backing, DeviceAddressed, Frozen, Slice};
use crate::cuda::{
    self, check, check_blas, CUcontext, CUdevice, CUdeviceptr, CUevent, CUfunction, CUgraph,
    CUgraphExec, CUmodule, CUstream,
};
use crate::error::{Error, Result};
use std::ffi::{c_int, c_void, CString};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Context and stream, shared by the `Gpu` and every `DeviceBacking` it allocated.
pub struct Core {
    drv: &'static cuda::Driver,
    dev: CUdevice,
    ctx: CUcontext,
    stream: CUstream,
    tally: Tally,
}

/// Driver calls made through this context (see `Counts`).
#[derive(Default)]
struct Tally {
    kernel_launches: AtomicU64,
    graph_launches: AtomicU64,
    gemms: AtomicU64,
    htod: AtomicU64,
    dtoh: AtomicU64,
    memsets: AtomicU64,
    syncs: AtomicU64,
}

fn bump(c: &AtomicU64) {
    c.fetch_add(1, Ordering::Relaxed);
}

// SAFETY: a CUDA context and stream may be used from any thread once made current;
// `bind` makes the context current before every driver call that needs it.
unsafe impl Send for Core {}
// SAFETY: shared, a Core only reads its handles and bumps atomic counters; every driver call
// goes through `bind` first, and the driver API may be called from several threads.
unsafe impl Sync for Core {}

impl Core {
    pub fn bind(&self) -> Result<()> {
        // SAFETY: the primary context retained in `Gpu::new`, released when the last Core goes.
        check(
            unsafe { (self.drv.ctx_set_current)(self.ctx) },
            "cuCtxSetCurrent",
        )
    }

    pub fn sync(&self) -> Result<()> {
        self.bind()?;
        bump(&self.tally.syncs);
        // SAFETY: the stream of this context, current after `bind`.
        check(
            unsafe { (self.drv.stream_synchronize)(self.stream) },
            "cuStreamSynchronize",
        )
    }

    fn counts(&self) -> Counts {
        let t = &self.tally;
        let g = |c: &AtomicU64| c.load(Ordering::Relaxed);
        Counts {
            kernel_launches: g(&t.kernel_launches),
            graph_launches: g(&t.graph_launches),
            gemms: g(&t.gemms),
            htod: g(&t.htod),
            dtoh: g(&t.dtoh),
            memsets: g(&t.memsets),
            syncs: g(&t.syncs),
        }
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        // SAFETY: the last reference to this context, so no Gpu, arena, graph, event or host
        // words remain; the stream is drained, destroyed, then the context released.
        // Errors cannot be returned from drop; the context is released either way.
        unsafe {
            (self.drv.ctx_set_current)(self.ctx);
            (self.drv.stream_synchronize)(self.stream);
            (self.drv.stream_destroy)(self.stream);
            (self.drv.primary_ctx_release)(self.dev);
        }
    }
}

/// Device memory behind a frozen arena: one cuMemAlloc, freed on drop.
pub struct DeviceBacking {
    core: Arc<Core>,
    ptr: CUdeviceptr,
    len: u64,
}

impl DeviceAddressed for DeviceBacking {
    fn base_addr(&self) -> u64 {
        self.ptr
    }
}

fn range_ok(total: u64, offset: u64, len: u64) -> Result<()> {
    match offset.checked_add(len) {
        Some(end) if end <= total => Ok(()),
        _ => Err(Error::Arena(format!(
            "device range {offset}+{len} outside {total} bytes"
        ))),
    }
}

impl Backing for DeviceBacking {
    fn len(&self) -> u64 {
        self.len
    }

    fn write(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        range_ok(self.len, offset, data.len() as u64)?;
        let c = &self.core;
        c.bind()?;
        bump(&c.tally.htod);
        check(
            // SAFETY: `range_ok` keeps the device range inside this allocation; `data` stays
            // borrowed until the synchronization below.
            unsafe {
                (c.drv.memcpy_htod_async)(
                    self.ptr + offset,
                    data.as_ptr().cast(),
                    data.len(),
                    c.stream,
                )
            },
            "cuMemcpyHtoDAsync",
        )?;
        c.sync()
    }

    fn read(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        range_ok(self.len, offset, out.len() as u64)?;
        let c = &self.core;
        c.bind()?;
        bump(&c.tally.dtoh);
        check(
            // SAFETY: `range_ok` keeps the device range inside this allocation; `out` stays
            // borrowed until the synchronization below.
            unsafe {
                (c.drv.memcpy_dtoh_async)(
                    out.as_mut_ptr().cast(),
                    self.ptr + offset,
                    out.len(),
                    c.stream,
                )
            },
            "cuMemcpyDtoHAsync",
        )?;
        c.sync()
    }

    fn zero(&mut self, offset: u64, len: u64) -> Result<()> {
        range_ok(self.len, offset, len)?;
        let c = &self.core;
        c.bind()?;
        bump(&c.tally.memsets);
        check(
            // SAFETY: `range_ok` keeps the range inside this allocation.
            unsafe { (c.drv.memset_d8_async)(self.ptr + offset, 0, len as usize, c.stream) },
            "cuMemsetD8Async",
        )?;
        c.sync()
    }
}

impl Drop for DeviceBacking {
    fn drop(&mut self) {
        let c = &self.core;
        // SAFETY: the stream is drained, so no queued work uses the allocation; freed once,
        // here. The Arc keeps the context alive.
        unsafe {
            (c.drv.ctx_set_current)(c.ctx);
            (c.drv.stream_synchronize)(c.stream);
            (c.drv.mem_free)(self.ptr);
        }
    }
}

/// How the split-K decode attention is launched for one geometry (`Gpu::split_plan`).
#[derive(Clone, Copy)]
pub struct SplitPlan {
    pub kernel: Kernel,
    /// Blocks along x: kv heads when a block serves a GQA group, else q heads.
    pub grid_x: u32,
    /// q heads served by one block (1 for the per-q-head kernel).
    pub group: u32,
    /// The kernel is a grouped one (`x0ne_attention_decode_split_g*`): it takes a counter and
    /// an output and can merge in its last block (option 7).
    pub grouped: bool,
    /// Blocks of the kernel one SM holds at once.
    pub per_sm: u32,
    /// P, blocks along y: the slices of the context.
    pub splits: u32,
}

pub use x0ne_core_kernels::Param;

/// One kernel argument, by the C type of its parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Arg {
    Ptr(u64),
    U32(u32),
    F32(f32),
}

impl Arg {
    pub fn param(&self) -> Param {
        match self {
            Arg::Ptr(_) => Param::Ptr,
            Arg::U32(_) => Param::U32,
            Arg::F32(_) => Param::F32,
        }
    }
}

/// A device address.
impl From<u64> for Arg {
    fn from(v: u64) -> Self {
        Arg::Ptr(v)
    }
}

impl From<u32> for Arg {
    fn from(v: u32) -> Self {
        Arg::U32(v)
    }
}

impl From<f32> for Arg {
    fn from(v: f32) -> Self {
        Arg::F32(v)
    }
}

/// `args` against the declared parameters of kernel `name`: as many, each of the same C type.
pub fn check_args(name: &str, params: &[Param], args: &[Arg]) -> Result<()> {
    if args.len() != params.len() {
        return Err(Error::Gpu(format!(
            "{name}: {} arguments for {} parameters",
            args.len(),
            params.len()
        )));
    }
    match args.iter().zip(params).position(|(a, p)| a.param() != *p) {
        Some(i) => Err(Error::Gpu(format!(
            "{name}: argument {i} is {:?}, the kernel takes {:?}",
            args[i].param(),
            params[i]
        ))),
        None => Ok(()),
    }
}

/// Most parameters a launch passes (the kernels take 12 at most).
const MAX_ARGS: usize = 16;

/// Compares the parameters of `f` in the loaded cubin with `params`: the same count
/// (cuFuncGetParamCount), and each of the size of its declared C type (cuFuncGetParamInfo,
/// which knows sizes, not types).
fn check_cubin_params(
    drv: &cuda::Driver,
    f: CUfunction,
    name: &str,
    params: &[Param],
) -> Result<()> {
    let mut n = 0usize;
    check(
        // SAFETY: `f` is a function of a loaded module; `n` is a live local.
        unsafe { (drv.func_get_param_count)(f, &mut n) },
        &format!("cuFuncGetParamCount {name}"),
    )?;
    if n != params.len() {
        return Err(Error::Gpu(format!(
            "{name}: {n} parameters in the cubin, {} declared",
            params.len()
        )));
    }
    let (mut offset, mut size) = (0usize, 0usize);
    for (i, p) in params.iter().enumerate() {
        check(
            // SAFETY: `f` is a function of a loaded module; both out-pointers are live locals.
            unsafe { (drv.func_get_param_info)(f, i, &mut offset, &mut size) },
            &format!("cuFuncGetParamInfo {name} {i}"),
        )?;
        if size != p.size() {
            return Err(Error::Gpu(format!(
                "{name}: parameter {i} is {size} bytes in the cubin, {} declared ({p:?})",
                p.size()
            )));
        }
    }
    Ok(())
}

/// A kernel of `x0ne_core_kernels::KERNEL_NAMES`, resolved in the loaded cubin, with its
/// declared parameters (checked against the cubin when it was resolved) and the id of the
/// `Gpu` whose module holds it (`Gpu::launch` refuses it on any other).
#[derive(Clone, Copy)]
pub struct Kernel {
    f: CUfunction,
    name: &'static str,
    params: &'static [Param],
    gpu: u64,
}

impl Kernel {
    pub fn name(&self) -> &'static str {
        self.name
    }
    pub fn params(&self) -> &'static [Param] {
        self.params
    }
}

pub struct Funcs {
    pub embedding: Kernel,
    pub add_rmsnorm: Kernel,
    pub qk_norm_rope_kv_write: Kernel,
    pub attention_decode: Kernel,
    pub attention_decode_split: Kernel,
    /// The split-K kernel with one block per (kv head, slice), for GQA groups of 1, 2, 4, 8.
    pub attention_decode_split_group: [(u32, Kernel); 4],
    pub attention_merge: Kernel,
    pub attention_prefill: Kernel,
    pub swiglu: Kernel,
    pub argmax: Kernel,
    pub advance: Kernel,
    pub moe_route: Kernel,
    pub moe_gemv_bf16: Kernel,
    pub moe_gemv_f32: Kernel,
    pub moe_combine: Kernel,
}

/// A captured launch sequence, instantiated once, replayed with `Gpu::launch_graph`.
pub struct Graph {
    core: Arc<Core>,
    graph: CUgraph,
    exec: CUgraphExec,
    nodes: usize,
}

// SAFETY: see `Core`; graph, event and host handles are used under `bind`, one thread at a time.
unsafe impl Send for Graph {}
// SAFETY: as for Graph: an event handle, used under `bind`; Event is not Sync.
unsafe impl Send for Event {}
// SAFETY: as for Graph: page-locked words and their events, used under `bind`; not Sync.
unsafe impl Send for HostWords {}

impl Graph {
    /// Nodes of the captured graph (kernels, memsets; cuBLAS may add more than one per GEMM).
    pub fn nodes(&self) -> usize {
        self.nodes
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        let c = &self.core;
        // SAFETY: the stream is drained, so no replay is queued; destroyed once, here.
        unsafe {
            (c.drv.ctx_set_current)(c.ctx);
            (c.drv.stream_synchronize)(c.stream);
            (c.drv.graph_exec_destroy)(self.exec);
            (c.drv.graph_destroy)(self.graph);
        }
    }
}

/// A stream event: recorded after work on the stream, waited for on the host.
pub struct Event {
    core: Arc<Core>,
    ev: CUevent,
}

impl Event {
    pub fn record(&self) -> Result<()> {
        let c = &self.core;
        c.bind()?;
        // SAFETY: an event and the stream of this context, current after `bind`.
        check(
            unsafe { (c.drv.event_record)(self.ev, c.stream) },
            "cuEventRecord",
        )
    }
    /// Blocks until the recorded work is done. Not counted as a stream synchronization.
    pub fn wait(&self) -> Result<()> {
        let c = &self.core;
        c.bind()?;
        // SAFETY: an event of this context; the wait touches no Rust memory.
        check(
            unsafe { (c.drv.event_synchronize)(self.ev) },
            "cuEventSynchronize",
        )
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: destroyed once, here; the driver releases a recorded event once it completes.
        unsafe {
            (self.core.drv.ctx_set_current)(self.core.ctx);
            (self.core.drv.event_destroy)(self.ev);
        }
    }
}

/// Page-locked host words, the landing zone of asynchronous 4-byte copies (one per token),
/// each with the event recorded behind the last copy into it.
///
/// A copy into a word is `fetch`: 4 bytes, from a range of a frozen arena that `Arena::span`
/// checks. A word is read with `take`, which first waits for the event of the last copy into it;
/// no read skips that wait, so this does not compile:
///
/// ```compile_fail
/// fn read(w: &x0ne_core_runtime::gpu::HostWords) -> x0ne_core_runtime::Result<u32> {
///     w.get(0)
/// }
/// ```
///
/// ```no_run
/// fn read(w: &x0ne_core_runtime::gpu::HostWords) -> x0ne_core_runtime::Result<u32> {
///     w.take(0)
/// }
/// ```
pub struct HostWords {
    core: Arc<Core>,
    ptr: *mut u32,
    len: usize,
    /// Event `i` is recorded behind each copy into word `i`.
    events: Vec<Event>,
}

impl HostWords {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn word(&self, i: usize) -> Result<*mut u32> {
        if i >= self.len {
            return Err(Error::Arena(format!("host word {i} outside {}", self.len)));
        }
        // SAFETY: i < len: inside the allocation of len words.
        Ok(unsafe { self.ptr.add(i) })
    }

    /// Copies the 4 bytes at byte `at` of slice `src` of `arena` into word `i`, on the stream,
    /// and records the word's event behind the copy. Counted as one device-to-host copy.
    pub fn fetch(
        &self,
        arena: &Arena<Frozen<DeviceBacking>>,
        src: Slice,
        at: u64,
        i: usize,
    ) -> Result<()> {
        let dst = self.word(i)?;
        if !Arc::ptr_eq(&arena.backing().core, &self.core) {
            return Err(Error::Gpu("host words and arena of two contexts".into()));
        }
        let src = arena.span(src, at, 4)?;
        let c = &self.core;
        c.bind()?;
        bump(&c.tally.dtoh);
        check(
            // SAFETY: `dst` is one page-locked word of this allocation, freed only after a stream
            // synchronization (drop); `src..src + 4` lies in a slice of a frozen arena of this
            // context. The word is read only by `take`, after the event recorded below.
            unsafe { (c.drv.memcpy_dtoh_async)(dst.cast(), src, 4, c.stream) },
            "cuMemcpyDtoHAsync",
        )?;
        self.events[i].record()
    }

    /// Word `i` once the last copy into it is done: waits for the word's event, then reads it.
    pub fn take(&self, i: usize) -> Result<u32> {
        let w = self.word(i)?;
        self.events[i].wait()?;
        // SAFETY: a word of this live page-locked allocation, aligned; the copies into it are
        // done (event waited). Volatile: the device wrote it outside the compiler's view.
        Ok(unsafe { std::ptr::read_volatile(w) })
    }

    /// Writes word `i` once the last copy into or from it is done (waits for the word's event).
    pub fn set(&self, i: usize, v: u32) -> Result<()> {
        let w = self.word(i)?;
        self.events[i].wait()?;
        // SAFETY: a word of this live page-locked allocation, aligned; no copy into or from it
        // is in flight (event waited). Volatile: a later copy reads it outside the compiler's view.
        unsafe { std::ptr::write_volatile(w, v) };
        Ok(())
    }

    /// Copies word `i` to the 4 bytes at byte `at` of slice `dst` of `arena`, on the stream,
    /// and records the word's event behind the copy. Counted as one host-to-device copy.
    pub fn send(
        &self,
        i: usize,
        arena: &Arena<Frozen<DeviceBacking>>,
        dst: Slice,
        at: u64,
    ) -> Result<()> {
        let src = self.word(i)?;
        if !Arc::ptr_eq(&arena.backing().core, &self.core) {
            return Err(Error::Gpu("host words and arena of two contexts".into()));
        }
        let dst = arena.span(dst, at, 4)?;
        let c = &self.core;
        c.bind()?;
        bump(&c.tally.htod);
        check(
            // SAFETY: `src` is one page-locked word of this allocation, freed only after a stream
            // synchronization (drop), written only by `set` after the event recorded below;
            // `dst..dst + 4` lies in a slice of a frozen arena of this context.
            unsafe { (c.drv.memcpy_htod_async)(dst, src.cast(), 4, c.stream) },
            "cuMemcpyHtoDAsync",
        )?;
        self.events[i].record()
    }

    /// Zeroes every word, each once its copies are done.
    pub fn zero(&self) -> Result<()> {
        (0..self.len).try_for_each(|i| self.set(i, 0))
    }

    /// True when every word reads back as 0 once its copies are done; false if a wait fails.
    pub fn is_zero(&self) -> bool {
        (0..self.len).all(|i| self.take(i).is_ok_and(|v| v == 0))
    }
}

impl Drop for HostWords {
    fn drop(&mut self) {
        // SAFETY: the stream is drained, so no copy into the words is pending; freed once, here.
        unsafe {
            (self.core.drv.ctx_set_current)(self.core.ctx);
            (self.core.drv.stream_synchronize)(self.core.stream);
            (self.core.drv.mem_free_host)(self.ptr.cast());
        }
    }
}

/// Page-locked host f32 values, the landing zone of one logits row per decode step, with the
/// event recorded behind the last copy into them.
///
/// A copy is `fetch`, from a range of a frozen arena that `Arena::span` checks; it takes
/// `&mut self`, so no slice of the values is borrowed while it runs. The values are read with
/// `values`, which first waits for the event; there is no read that skips the wait:
///
/// ```compile_fail
/// fn read(h: &x0ne_core_runtime::gpu::HostFloats) -> &[f32] {
///     h.as_slice()
/// }
/// ```
///
/// ```no_run
/// fn read(h: &x0ne_core_runtime::gpu::HostFloats) -> x0ne_core_runtime::Result<&[f32]> {
///     h.values()
/// }
/// ```
pub struct HostFloats {
    core: Arc<Core>,
    ptr: *mut f32,
    len: usize,
    /// Recorded behind each copy into the values.
    event: Event,
}

// SAFETY: as for Graph: page-locked values and their event, used under `bind`; not Sync.
unsafe impl Send for HostFloats {}

impl HostFloats {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copies `n` f32 values from byte `at` of slice `src` of `arena` into the first `n` values,
    /// on the stream, and records the event behind the copy. Counted as one device-to-host copy.
    pub fn fetch(
        &mut self,
        arena: &Arena<Frozen<DeviceBacking>>,
        src: Slice,
        at: u64,
        n: usize,
    ) -> Result<()> {
        if n > self.len {
            return Err(Error::Arena(format!(
                "{n} values into {} host floats",
                self.len
            )));
        }
        if !Arc::ptr_eq(&arena.backing().core, &self.core) {
            return Err(Error::Gpu("host floats and arena of two contexts".into()));
        }
        let src = arena.span(src, at, n as u64 * 4)?;
        let c = &self.core;
        c.bind()?;
        bump(&c.tally.dtoh);
        check(
            // SAFETY: the first `n` values of this page-locked allocation, freed only after a
            // stream synchronization (drop); `&mut self`: no slice of them is borrowed, and
            // `values` waits for the event recorded below. The source is a checked arena span.
            unsafe { (c.drv.memcpy_dtoh_async)(self.ptr.cast(), src, n * 4, c.stream) },
            "cuMemcpyDtoHAsync",
        )?;
        self.event.record()
    }

    /// The values once the last copy into them is done: waits for the event first.
    pub fn values(&self) -> Result<&[f32]> {
        self.event.wait()?;
        // SAFETY: len values of this live page-locked allocation, initialized (zeroed at
        // allocation); the last copy is done and the next one (`fetch`) needs `&mut self`.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr, self.len) })
    }

    /// Zeroes every value once the last copy is done.
    pub fn zero(&mut self) -> Result<()> {
        self.event.wait()?;
        // SAFETY: len values of this live allocation; no copy is in flight (event waited) and
        // `&mut self` excludes any borrowed slice.
        unsafe { std::ptr::write_bytes(self.ptr, 0, self.len) };
        Ok(())
    }

    /// True when every value reads back as +0.0 bits; false if the wait fails.
    pub fn is_zero(&self) -> bool {
        self.values()
            .is_ok_and(|v| v.iter().all(|x| x.to_bits() == 0))
    }
}

impl Drop for HostFloats {
    fn drop(&mut self) {
        // SAFETY: the stream is drained, so no copy into the values is pending; freed once, here.
        unsafe {
            (self.core.drv.ctx_set_current)(self.core.ctx);
            (self.core.drv.stream_synchronize)(self.core.stream);
            (self.core.drv.mem_free_host)(self.ptr.cast());
        }
    }
}

/// Driver calls made through this module since the context was opened. Host-side counts
/// of this code's own calls: what the E2 gate "one launch and one 4-byte copy per token"
/// is measured on inside the tests (nsys is the external witness).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Counts {
    pub kernel_launches: u64,
    pub graph_launches: u64,
    pub gemms: u64,
    pub htod: u64,
    pub dtoh: u64,
    pub memsets: u64,
    pub syncs: u64,
}

/// Output element type of a GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Out {
    Bf16,
    F32,
}

pub struct Gpu {
    /// Unique in the process, never reused: a Kernel carries it.
    id: u64,
    core: Arc<Core>,
    module: CUmodule,
    blas: cuda::CublasHandle,
    pub f: Funcs,
    /// Compute capability (major, minor) and the cubin chosen for it.
    pub cc: (u32, u32),
    pub arch: &'static str,
    pub sms: u32,
    pub name: String,
    pub driver_version: i32,
    pub cublas_version: i32,
}

// SAFETY: see `Core`; the module and cuBLAS handle are used under `bind`, one thread at a time
// (Engine and Kernels take `&mut self` or are not shared across threads).
unsafe impl Send for Gpu {}
// SAFETY: a CUfunction is an immutable handle into the module `Gpu` owns, launched under `bind`.
unsafe impl Send for Kernel {}

/// Ids of `Gpu::new`, from 1.
static NEXT_GPU: AtomicU64 = AtomicU64::new(1);

/// Function `name` of a loaded module.
fn function(drv: &cuda::Driver, module: CUmodule, name: &str) -> Result<CUfunction> {
    let c = CString::new(name).map_err(|_| Error::Gpu(format!("kernel name {name:?}")))?;
    let mut f: CUfunction = std::ptr::null_mut();
    check(
        // SAFETY: `module` is loaded, `c` is a C string and `f` a live local.
        unsafe { (drv.module_get_function)(&mut f, module, c.as_ptr()) },
        name,
    )?;
    Ok(f)
}

fn attr(drv: &cuda::Driver, a: c_int, dev: CUdevice) -> Result<i32> {
    let mut v = 0;
    check(
        // SAFETY: `v` is a live local; `dev` comes from cuDeviceGet.
        unsafe { (drv.device_attribute)(&mut v, a, dev) },
        "cuDeviceGetAttribute",
    )?;
    Ok(v)
}

impl Gpu {
    /// GPU 0, refused when the embedded cubins hold no SASS for its compute capability.
    pub fn new() -> Result<Gpu> {
        let drv = cuda::driver()?;
        // SAFETY: flags 0, the only value documented; no Rust memory involved.
        check(unsafe { (drv.init)(0) }, "cuInit")?;
        let mut dev: CUdevice = 0;
        // SAFETY: `dev` is a live local.
        check(unsafe { (drv.device_get)(&mut dev, 0) }, "cuDeviceGet")?;
        let major = attr(drv, cuda::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)? as u32;
        let minor = attr(drv, cuda::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)? as u32;
        let sms = attr(drv, cuda::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)? as u32;
        let cubin = x0ne_core_kernels::cubin_for(major, minor).ok_or_else(|| {
            Error::Gpu(format!("no cubin for compute capability {major}.{minor}"))
        })?;
        let mut name_buf = [0u8; 256];
        check(
            // SAFETY: the driver writes at most 256 bytes, the length of `name_buf`.
            unsafe { (drv.device_name)(name_buf.as_mut_ptr().cast(), 256, dev) },
            "cuDeviceGetName",
        )?;
        let name = String::from_utf8_lossy(
            &name_buf[..name_buf.iter().position(|&b| b == 0).unwrap_or(0)],
        )
        .into_owned();
        let mut driver_version = 0;
        check(
            // SAFETY: `driver_version` is a live local.
            unsafe { (drv.driver_version)(&mut driver_version) },
            "cuDriverGetVersion",
        )?;

        let mut ctx: CUcontext = std::ptr::null_mut();
        check(
            // SAFETY: `ctx` is a live local; `dev` comes from cuDeviceGet.
            unsafe { (drv.primary_ctx_retain)(&mut ctx, dev) },
            "cuDevicePrimaryCtxRetain",
        )?;
        // SAFETY: the context retained just above.
        check(unsafe { (drv.ctx_set_current)(ctx) }, "cuCtxSetCurrent")?;
        let mut stream: CUstream = std::ptr::null_mut();
        if let Err(e) = check(
            // SAFETY: `stream` is a live local; the context is current.
            unsafe { (drv.stream_create)(&mut stream, cuda::CU_STREAM_NON_BLOCKING) },
            "cuStreamCreate",
        ) {
            // SAFETY: gives back the retain made above, which nothing uses yet.
            unsafe { (drv.primary_ctx_release)(dev) };
            return Err(e);
        }
        let core = Arc::new(Core {
            drv,
            dev,
            ctx,
            stream,
            tally: Tally::default(),
        });

        let mut module: CUmodule = std::ptr::null_mut();
        check(
            // SAFETY: `cubin.bytes` is an embedded cubin, read during the call; `module` is
            // a live local.
            unsafe { (drv.module_load_data)(&mut module, cubin.bytes.as_ptr().cast()) },
            "cuModuleLoadData",
        )?;
        let id = NEXT_GPU.fetch_add(1, Ordering::Relaxed);
        let get = |name: &'static str| -> Result<Kernel> {
            let params = x0ne_core_kernels::params_of(name)
                .ok_or_else(|| Error::Gpu(format!("{name}: no declared parameters")))?;
            let f = function(drv, module, name)?;
            check_cubin_params(drv, f, name, params)?;
            Ok(Kernel {
                f,
                name,
                params,
                gpu: id,
            })
        };
        let f = Funcs {
            embedding: get("x0ne_embedding_bf16")?,
            add_rmsnorm: get("x0ne_add_rmsnorm_bf16")?,
            qk_norm_rope_kv_write: get("x0ne_qk_norm_rope_kv_write_bf16")?,
            attention_decode: get("x0ne_attention_decode_bf16")?,
            attention_decode_split: get("x0ne_attention_decode_split_bf16")?,
            attention_decode_split_group: [
                (1, get("x0ne_attention_decode_split_g1_bf16")?),
                (2, get("x0ne_attention_decode_split_g2_bf16")?),
                (4, get("x0ne_attention_decode_split_g4_bf16")?),
                (8, get("x0ne_attention_decode_split_g8_bf16")?),
            ],
            attention_merge: get("x0ne_attention_merge_bf16")?,
            attention_prefill: get("x0ne_attention_prefill_bf16")?,
            swiglu: get("x0ne_swiglu_bf16")?,
            argmax: get("x0ne_argmax_f32")?,
            advance: get("x0ne_advance_u32")?,
            moe_route: get("x0ne_moe_route_f32")?,
            moe_gemv_bf16: get("x0ne_moe_gemv_bf16")?,
            moe_gemv_f32: get("x0ne_moe_gemv_f32")?,
            moe_combine: get("x0ne_moe_combine_bf16")?,
        };

        let b = cuda::blas()?;
        let mut blas: cuda::CublasHandle = std::ptr::null_mut();
        // SAFETY: `blas` is a live local; the context is current.
        check_blas(unsafe { (b.create)(&mut blas) }, "cublasCreate")?;
        let mut cublas_version = 0;
        check_blas(
            // SAFETY: the handle just created; `cublas_version` is a live local.
            unsafe { (b.version)(blas, &mut cublas_version) },
            "cublasGetVersion",
        )?;
        // SAFETY: a live handle and the stream of the current context.
        check_blas(unsafe { (b.set_stream)(blas, stream) }, "cublasSetStream")?;
        check_blas(
            // SAFETY: a live handle.
            unsafe { (b.set_math_mode)(blas, cuda::CUBLAS_DEFAULT_MATH) },
            "cublasSetMathMode",
        )?;
        Ok(Gpu {
            id,
            core,
            module,
            blas,
            f,
            cc: (major, minor),
            arch: cubin.arch,
            sms,
            name,
            driver_version,
            cublas_version,
        })
    }

    /// Compares kernel `name` of the loaded cubin with `params`: the check `new` makes for
    /// every kernel against `x0ne_core_kernels::KERNEL_PARAMS`.
    pub fn check_signature(&self, name: &str, params: &[Param]) -> Result<()> {
        self.core.bind()?;
        let f = function(self.core.drv, self.module, name)?;
        check_cubin_params(self.core.drv, f, name, params)
    }

    /// Blocks of `block` threads of `k` that one SM holds at once (no dynamic shared memory).
    /// `k` is looked up by name in this Gpu's module: a Kernel of an unloaded module is refused.
    pub fn blocks_per_sm(&self, k: Kernel, block: u32) -> Result<u32> {
        let c = &self.core;
        c.bind()?;
        let f = function(c.drv, self.module, k.name)?;
        let mut n: c_int = 0;
        check(
            // SAFETY: `f` is a function of this Gpu's loaded module; `n` is a live local.
            unsafe { (c.drv.occupancy_max_blocks)(&mut n, f, block as c_int, 0) },
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        )?;
        Ok(n.max(0) as u32)
    }

    /// The split-K decode attention for a geometry. A block serves the q heads of one kv head
    /// (grid kv_heads x P) when the GQA group size has a compiled kernel (1, 2, 4, 8), else one
    /// q head (grid q_heads x P). P is one wave of that kernel over the GPU: SMs times blocks
    /// per SM (driver occupancy) over the blocks along x, at least 1. Fixed per GPU and
    /// geometry, so the result of a decode step depends on ctx only.
    pub fn split_plan(&self, q_heads: usize, kv_heads: usize) -> Result<SplitPlan> {
        if kv_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
            return Err(Error::Gpu(format!(
                "{q_heads} q heads for {kv_heads} kv heads"
            )));
        }
        let u =
            |v: usize| u32::try_from(v).map_err(|_| Error::Gpu("head count exceeds u32".into()));
        let group = u(q_heads / kv_heads)?;
        let (kernel, grid_x, group, grouped) = match self
            .f
            .attention_decode_split_group
            .iter()
            .find(|(g, _)| *g == group)
        {
            Some(&(g, k)) => (k, u(kv_heads)?, g, true),
            None => (self.f.attention_decode_split, u(q_heads)?, 1, false),
        };
        let per_sm = self.blocks_per_sm(kernel, x0ne_core_kernels::SPLIT_BLOCK)?;
        Ok(SplitPlan {
            kernel,
            grid_x,
            group,
            grouped,
            per_sm,
            splits: (self.sms * per_sm / grid_x).max(1),
        })
    }

    /// The single device allocation of an arena (`Arena::freeze`).
    pub fn allocate(&self, len: u64) -> Result<DeviceBacking> {
        let c = &self.core;
        c.bind()?;
        let mut ptr: CUdeviceptr = 0;
        check(
            // SAFETY: `ptr` is a live local; the allocation is freed by `DeviceBacking::drop`.
            unsafe { (c.drv.mem_alloc)(&mut ptr, len as usize) },
            &format!("cuMemAlloc of {len} bytes"),
        )?;
        Ok(DeviceBacking {
            core: self.core.clone(),
            ptr,
            len,
        })
    }

    /// cuBLAS works in `len` bytes at `ptr` (inside the arena) and allocates nothing itself.
    /// cublasSetStream resets the workspace, so the stream was set first, in `new`.
    ///
    /// # Safety
    ///
    /// `ptr..ptr + len` is device memory of this context that stays allocated while this
    /// `Gpu` can run a GEMM, and nothing else writes it while a GEMM runs.
    pub unsafe fn set_cublas_workspace(&self, ptr: u64, len: u64) -> Result<()> {
        self.core.bind()?;
        let b = cuda::blas()?;
        // SAFETY: `self.blas` is this Gpu's live handle; the range is the caller's contract.
        check_blas(
            unsafe { (b.set_workspace)(self.blas, ptr as *mut c_void, len as usize) },
            "cublasSetWorkspace",
        )
    }

    /// Launches `k` on the stream with `args`, one per parameter. Refused before the driver
    /// sees them: a kernel of another `Gpu` (one on the same device shares this context, one
    /// dropped has unloaded its module), and arguments that differ from the kernel's declared
    /// parameters in number or C type (`check_args`).
    ///
    /// # Safety
    ///
    /// Every pointer argument is a device address of this context, valid for every byte the
    /// kernel reads or writes at this grid with these sizes (see its definition in `cuda/`),
    /// until the launch is done on the stream.
    pub unsafe fn launch(
        &self,
        k: Kernel,
        grid: (u32, u32),
        block: u32,
        args: &[Arg],
    ) -> Result<()> {
        if k.gpu != self.id {
            return Err(Error::Gpu(format!(
                "{}: a kernel of Gpu {}, launched by Gpu {}",
                k.name, k.gpu, self.id
            )));
        }
        check_args(k.name, k.params, args)?;
        if args.len() > MAX_ARGS {
            return Err(Error::Gpu(format!(
                "{}: {} arguments, at most {MAX_ARGS}",
                k.name,
                args.len()
            )));
        }
        // cuLaunchKernel reads one pointer per parameter; each points into `args`, which
        // outlives the call.
        let mut ptrs = [std::ptr::null_mut::<c_void>(); MAX_ARGS];
        for (p, a) in ptrs.iter_mut().zip(args) {
            *p = match a {
                Arg::Ptr(v) => std::ptr::from_ref(v).cast_mut().cast(),
                Arg::U32(v) => std::ptr::from_ref(v).cast_mut().cast(),
                Arg::F32(v) => std::ptr::from_ref(v).cast_mut().cast(),
            };
        }
        let c = &self.core;
        c.bind()?;
        bump(&c.tally.kernel_launches);
        check(
            // SAFETY: a function of this Gpu's loaded module (checked above), on this context's
            // stream; one pointer per parameter, each to a value of its declared C type
            // (checked above); the device addresses are the caller's contract.
            unsafe {
                (c.drv.launch_kernel)(
                    k.f,
                    grid.0,
                    grid.1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    c.stream,
                    ptrs.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    /// `Y[n][rows] = X[n][cols] W^T` with `W [rows][cols]`, all row-major, bf16 in, f32
    /// accumulation.
    ///
    /// # Safety
    ///
    /// `w`, `x` and `y` are device addresses of this context holding `rows * cols` bf16,
    /// `n * cols` bf16 and `n * rows` values of `out`; `y` overlaps neither `w` nor `x`; all
    /// three stay allocated until the GEMM is done on the stream. The workspace was set
    /// (`set_cublas_workspace`).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm(
        &self,
        w: u64,
        rows: usize,
        cols: usize,
        x: u64,
        n: usize,
        y: u64,
        out: Out,
    ) -> Result<()> {
        let i = |v: usize| {
            i32::try_from(v).map_err(|_| Error::Gpu(format!("gemm dimension {v} exceeds i32")))
        };
        let (m, nn, k) = (i(rows)?, i(n)?, i(cols)?);
        let (alpha, beta) = (1f32, 0f32);
        let ctype = match out {
            Out::Bf16 => cuda::CUDA_R_16BF,
            Out::F32 => cuda::CUDA_R_32F,
        };
        self.core.bind()?;
        bump(&self.core.tally.gemms);
        let b = cuda::blas()?;
        // Column-major view: C (m x n, ldc m) = A^T (A = W as cols x rows, lda cols) * B (cols x n).
        check_blas(
            // SAFETY: alpha and beta live across the call (host pointer mode); the three
            // matrices are the caller's contract.
            unsafe {
                (b.gemm_ex)(
                    self.blas,
                    cuda::CUBLAS_OP_T,
                    cuda::CUBLAS_OP_N,
                    m,
                    nn,
                    k,
                    (&alpha as *const f32).cast(),
                    w as *const c_void,
                    cuda::CUDA_R_16BF,
                    k,
                    x as *const c_void,
                    cuda::CUDA_R_16BF,
                    k,
                    (&beta as *const f32).cast(),
                    y as *mut c_void,
                    ctype,
                    m,
                    cuda::CUBLAS_COMPUTE_32F,
                    cuda::CUBLAS_GEMM_DEFAULT,
                )
            },
            "cublasGemmEx",
        )
    }

    /// Fills `count` u32 words at `ptr` with `value`, on the stream.
    ///
    /// # Safety
    ///
    /// `ptr..ptr + 4 * count` is device memory of this context, allocated until the memset is
    /// done on the stream.
    pub unsafe fn memset_u32(&self, ptr: u64, value: u32, count: usize) -> Result<()> {
        let c = &self.core;
        c.bind()?;
        bump(&c.tally.memsets);
        // SAFETY: the range is the caller's contract; the stream is this context's.
        check(
            unsafe { (c.drv.memset_d32_async)(ptr, value, count, c.stream) },
            "cuMemsetD32Async",
        )
    }

    /// Captures every launch `f` issues on the stream into one graph and instantiates it.
    /// Mode GLOBAL: an allocation, a synchronization or a host copy inside `f` fails the capture.
    pub fn capture(&self, f: impl FnOnce() -> Result<()>) -> Result<Graph> {
        let c = &self.core;
        c.bind()?;
        check(
            // SAFETY: the stream of this context, current after `bind`.
            unsafe { (c.drv.stream_begin_capture)(c.stream, cuda::CU_STREAM_CAPTURE_MODE_GLOBAL) },
            "cuStreamBeginCapture",
        )?;
        let body = f();
        let mut graph: CUgraph = std::ptr::null_mut();
        // SAFETY: ends the capture begun above; `graph` is a live local.
        let ended = check(
            unsafe { (c.drv.stream_end_capture)(c.stream, &mut graph) },
            "cuStreamEndCapture",
        );
        if let Err(e) = body.and(ended) {
            if !graph.is_null() {
                // SAFETY: the graph of this capture, destroyed once, here.
                unsafe { (c.drv.graph_destroy)(graph) };
            }
            return Err(e);
        }
        let mut nodes = 0usize;
        // SAFETY: a null array asks for the count only, written into the live local `nodes`.
        let counted = check(
            unsafe { (c.drv.graph_get_nodes)(graph, std::ptr::null_mut(), &mut nodes) },
            "cuGraphGetNodes",
        );
        let mut exec: CUgraphExec = std::ptr::null_mut();
        let made = counted.and_then(|_| {
            check(
                // SAFETY: `exec` is a live local; `graph` comes from the capture.
                unsafe { (c.drv.graph_instantiate)(&mut exec, graph, 0) },
                "cuGraphInstantiate",
            )
        });
        if let Err(e) = made {
            // SAFETY: the graph of this capture, destroyed once, here; no exec was made.
            unsafe { (c.drv.graph_destroy)(graph) };
            return Err(e);
        }
        Ok(Graph {
            core: self.core.clone(),
            graph,
            exec,
            nodes,
        })
    }

    /// Replays a captured graph on the stream. A graph of another `Gpu` is refused.
    ///
    /// # Safety
    ///
    /// Every device range the captured launches use is still allocated, as their contract
    /// required at capture, until the replay is done on the stream.
    pub unsafe fn launch_graph(&self, g: &Graph) -> Result<()> {
        if !Arc::ptr_eq(&g.core, &self.core) {
            return Err(Error::Gpu("graph captured by another Gpu".into()));
        }
        let c = &self.core;
        c.bind()?;
        bump(&c.tally.graph_launches);
        // SAFETY: a graph instantiated on this context, replayed on its stream; the ranges
        // its launches use are the caller's contract.
        check(
            unsafe { (c.drv.graph_launch)(g.exec, c.stream) },
            "cuGraphLaunch",
        )
    }

    pub fn event(&self) -> Result<Event> {
        let c = &self.core;
        c.bind()?;
        let mut ev: CUevent = std::ptr::null_mut();
        check(
            // SAFETY: `ev` is a live local; destroyed by `Event::drop`.
            unsafe { (c.drv.event_create)(&mut ev, cuda::CU_EVENT_DISABLE_TIMING) },
            "cuEventCreate",
        )?;
        Ok(Event {
            core: self.core.clone(),
            ev,
        })
    }

    /// `n` page-locked host words, zeroed, each with its event.
    pub fn host_words(&self, n: usize) -> Result<HostWords> {
        let events = (0..n).map(|_| self.event()).collect::<Result<Vec<_>>>()?;
        let c = &self.core;
        c.bind()?;
        let mut p: *mut c_void = std::ptr::null_mut();
        check(
            // SAFETY: `p` is a live local; the allocation is freed by `HostWords::drop`.
            unsafe { (c.drv.mem_host_alloc)(&mut p, n.max(1) * 4, 0) },
            "cuMemHostAlloc",
        )?;
        let ptr: *mut u32 = p.cast();
        // SAFETY: n.max(1) fresh words, written before any copy can target them.
        unsafe { std::ptr::write_bytes(ptr, 0, n.max(1)) };
        Ok(HostWords {
            core: self.core.clone(),
            ptr,
            len: n,
            events,
        })
    }

    /// `len` bytes from device address `src` to page-locked host memory, on the stream,
    /// without waiting. Counted as one device-to-host copy.
    ///
    /// # Safety
    ///
    /// `dst` is page-locked host memory valid for `len` bytes that nothing reads or writes
    /// until an event recorded after this copy was waited for, and stays allocated until then;
    /// `src..src + len` is device memory of this context.
    pub unsafe fn copy_to_host_async(&self, dst: *mut c_void, src: u64, len: usize) -> Result<()> {
        let c = &self.core;
        c.bind()?;
        bump(&c.tally.dtoh);
        // SAFETY: both ranges are the caller's contract; the stream is this context's.
        check(
            unsafe { (c.drv.memcpy_dtoh_async)(dst, src, len, c.stream) },
            "cuMemcpyDtoHAsync",
        )
    }

    /// `n` page-locked host f32 values, zeroed, with their event.
    pub fn host_floats(&self, n: usize) -> Result<HostFloats> {
        let event = self.event()?;
        let c = &self.core;
        c.bind()?;
        let mut p: *mut c_void = std::ptr::null_mut();
        check(
            // SAFETY: `p` is a live local; the allocation is freed by `HostFloats::drop`.
            unsafe { (c.drv.mem_host_alloc)(&mut p, n.max(1) * 4, 0) },
            "cuMemHostAlloc",
        )?;
        let ptr: *mut f32 = p.cast();
        // SAFETY: n.max(1) fresh values, written before any copy can target them.
        unsafe { std::ptr::write_bytes(ptr, 0, n.max(1)) };
        Ok(HostFloats {
            core: self.core.clone(),
            ptr,
            len: n,
            event,
        })
    }

    /// `len` bytes from page-locked host memory `src` to device address `dst`, on the stream,
    /// without waiting. Counted as one host-to-device copy. `HostWords::send` is the checked
    /// form for one word.
    ///
    /// # Safety
    ///
    /// `src` is page-locked host memory valid for `len` bytes that nothing writes until an
    /// event recorded after this copy was waited for, and stays allocated until then;
    /// `dst..dst + len` is device memory of this context.
    pub unsafe fn copy_to_device_async(
        &self,
        dst: u64,
        src: *const c_void,
        len: usize,
    ) -> Result<()> {
        let c = &self.core;
        c.bind()?;
        bump(&c.tally.htod);
        check(
            // SAFETY: both ranges are the caller's contract; the stream is this context's.
            unsafe { (c.drv.memcpy_htod_async)(dst, src, len, c.stream) },
            "cuMemcpyHtoDAsync",
        )
    }

    pub fn sync(&self) -> Result<()> {
        self.core.sync()
    }

    /// Driver calls made through this context so far.
    pub fn counts(&self) -> Counts {
        self.core.counts()
    }

    /// (free, total) device bytes, as cuMemGetInfo reports them for the whole device.
    pub fn mem_info(&self) -> Result<(u64, u64)> {
        let c = &self.core;
        c.bind()?;
        let (mut free, mut total) = (0usize, 0usize);
        check(
            // SAFETY: two live locals.
            unsafe { (c.drv.mem_get_info)(&mut free, &mut total) },
            "cuMemGetInfo",
        )?;
        Ok((free as u64, total as u64))
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        let _ = self.core.sync();
        if let Ok(b) = cuda::blas() {
            // SAFETY: this Gpu's handle, destroyed once, here, after the stream is drained.
            unsafe { (b.destroy)(self.blas) };
        }
        // SAFETY: unloaded once, here, after the stream is drained. `launch` refuses a Kernel
        // of another Gpu; `blocks_per_sm` looks functions up by name.
        unsafe { (self.core.drv.module_unload)(self.module) };
    }
}

/// `Gpu::launch` with the arguments as Rust values: u64 for a device address, u32, f32 (any
/// other type does not compile). It calls `Gpu::launch`, so it is used inside `unsafe` under
/// the same contract.
#[macro_export]
macro_rules! launch {
    ($gpu:expr, $k:expr, $grid:expr, $block:expr; $($arg:expr),* $(,)?) => {
        $gpu.launch($k, $grid, $block, &[$($crate::gpu::Arg::from($arg)),*])
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_must_match_the_parameters() {
        let advance = [Param::Ptr];
        let argmax = [Param::Ptr, Param::U32, Param::Ptr];
        assert!(check_args("x0ne_advance_u32", &advance, &[Arg::Ptr(0x1000)]).is_ok());
        assert!(check_args(
            "x0ne_argmax_f32",
            &argmax,
            &[Arg::Ptr(0x1000), Arg::U32(8), Arg::Ptr(0x2000)]
        )
        .is_ok());
        // One argument short, one too many.
        assert!(check_args("x0ne_argmax_f32", &argmax, &[Arg::Ptr(0x1000), Arg::U32(8)]).is_err());
        assert!(check_args(
            "x0ne_advance_u32",
            &advance,
            &[Arg::Ptr(0x1000), Arg::U32(1)]
        )
        .is_err());
        // A 4-byte value where the kernel reads an 8-byte pointer, and the reverse.
        assert!(check_args("x0ne_advance_u32", &advance, &[Arg::U32(0x1000)]).is_err());
        assert!(check_args(
            "x0ne_argmax_f32",
            &argmax,
            &[Arg::Ptr(0x1000), Arg::Ptr(8), Arg::Ptr(0x2000)]
        )
        .is_err());
        // Same size, other type: an f32 where the kernel reads a u32.
        let e = check_args(
            "x0ne_argmax_f32",
            &argmax,
            &[Arg::Ptr(0x1000), Arg::F32(8.0), Arg::Ptr(0x2000)],
        )
        .unwrap_err();
        assert!(e.to_string().contains("x0ne_argmax_f32"), "{e}");
    }

    #[test]
    fn arguments_take_the_c_type_of_their_rust_type() {
        assert_eq!(Arg::from(0x1000u64), Arg::Ptr(0x1000));
        assert_eq!(Arg::from(7u32), Arg::U32(7));
        assert_eq!(Arg::from(0.5f32), Arg::F32(0.5));
        assert_eq!(
            [
                Arg::Ptr(0).param(),
                Arg::U32(0).param(),
                Arg::F32(0.0).param()
            ],
            [Param::Ptr, Param::U32, Param::F32]
        );
    }
}
