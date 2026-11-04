//! The kernel surface the tests in tests/e1_kernels.rs call: each entry point plans an
//! arena for its inputs and outputs, freezes it (the one allocation), uploads, runs one
//! kernel on GPU 0, and reads the result back.

use crate::arena::{Arena, Class, Frozen, Slice};
use crate::error::{format_err, Error, Result};
pub use crate::gpu::SplitPlan;
use crate::gpu::{DeviceBacking, Gpu, Out};
use crate::launch;
use half::bf16;

pub use x0ne_core_kernels::{
    HEAD_DIM, MOE_MAX_EXPERTS, MOE_NO_EXPERT, SPLIT_BLOCK, SPLIT_STATE, SPLIT_TILE,
};

/// Attention geometry shared by the attention and RoPE entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Heads {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl Heads {
    pub fn q_dim(&self) -> usize {
        self.q_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
    /// The kernels are compiled for head_dim 128 and need q_heads to be a multiple of kv_heads.
    pub fn check(&self) -> Result<()> {
        if self.head_dim != x0ne_core_kernels::HEAD_DIM {
            return format_err(format!(
                "head_dim {} (kernels are built for {})",
                self.head_dim,
                x0ne_core_kernels::HEAD_DIM
            ));
        }
        if self.kv_heads == 0 || !self.q_heads.is_multiple_of(self.kv_heads) {
            return format_err(format!(
                "{} q heads for {} kv heads",
                self.q_heads, self.kv_heads
            ));
        }
        Ok(())
    }
}

/// The decode attention kernel. `PerHead` is E2's, one block per q head over the whole
/// context; it stays in the cubin as the reference of the tests. `SplitK` is E3's: tiles of
/// 128 keys at absolute positions, P blocks per q head (P fixed per GPU and geometry, from
/// the occupancy and the SM count), the partial states merged in the fixed order 0..P-1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecodeAttention {
    PerHead,
    SplitK,
}

/// How the split-K states are merged: by `x0ne_attention_merge_bf16` after the split kernel
/// (`Kernel`), or by the last block of each kv head inside it (`LastBlock`, option 7, grouped
/// kernels only, one u32 counter per kv head that the kernel leaves at 0). Same bits either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SplitMerge {
    Kernel,
    LastBlock,
}

/// The split-K launches of one attention: the split kernel, then the merge kernel unless the
/// merge is `LastBlock`. Device addresses; `counter` holds one zeroed u32 per kv head.
///
/// # Safety
///
/// Device addresses of `g`'s context, allocated until the launches are done on the stream:
/// `q` and `out` hold `qh * HEAD_DIM` bf16, `k` and `v` `max_ctx * kvh * HEAD_DIM` bf16,
/// `state` `split_state_bytes(qh, plan.splits)` bytes, `counter` `kvh` zeroed u32 (unused
/// unless the merge is `LastBlock`), `pos` one u32 below `max_ctx`. `plan` is `g`'s plan for
/// (`qh`, `kvh`).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch_split(
    g: &Gpu,
    plan: &SplitPlan,
    merge: SplitMerge,
    q: u64,
    k: u64,
    v: u64,
    state: u64,
    counter: u64,
    out: u64,
    pos: u64,
    qh: u32,
    kvh: u32,
    max_ctx: u32,
    scale: f32,
) -> Result<()> {
    let p = plan.splits;
    match (plan.grouped, merge) {
        (true, SplitMerge::LastBlock) => {
            // SAFETY: the grouped kernel's arguments, sized by the caller's contract.
            return unsafe {
                launch!(g, plan.kernel, (plan.grid_x, p), SPLIT_BLOCK; q, k, v, state, counter, out, pos, qh, kvh, max_ctx, p, scale)
            };
        }
        (true, SplitMerge::Kernel) => {
            // SAFETY: as above; null counter and output: the kernel leaves the merge to the merge kernel.
            unsafe {
                launch!(g, plan.kernel, (plan.grid_x, p), SPLIT_BLOCK; q, k, v, state, 0u64, 0u64, pos, qh, kvh, max_ctx, p, scale)
            }?
        }
        // SAFETY: the per-q-head kernel's arguments, sized by the caller's contract.
        (false, SplitMerge::Kernel) => unsafe {
            launch!(g, plan.kernel, (plan.grid_x, p), SPLIT_BLOCK; q, k, v, state, pos, qh, kvh, max_ctx, p, scale)
        }?,
        (false, SplitMerge::LastBlock) => {
            return Err(Error::Gpu(
                "the last-block merge needs a grouped split-K kernel".into(),
            ))
        }
    }
    // SAFETY: the states written above, merged into `out`; sizes by the caller's contract.
    unsafe { launch!(g, g.f.attention_merge, (qh, 1), SPLIT_BLOCK; state, out, pos, max_ctx, p) }
}

/// Bytes of the split-K partial states: q_heads x P states of `SPLIT_STATE` f32 words.
pub fn split_state_bytes(q_heads: usize, p: u32) -> u64 {
    (q_heads * p as usize * x0ne_core_kernels::SPLIT_STATE * 4) as u64
}

/// Where the KV write lands: layout `[layer][k|v][position][kv_heads * head_dim]`, bf16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheAt {
    pub layers: usize,
    pub max_ctx: usize,
    pub layer: usize,
    pub pos: usize,
}

/// What `qk_norm_rope_kv_write` returns: the rotated q, the k and v rows read back
/// from the cache at (layer, pos), and whether every other cache byte kept its sentinel.
#[derive(Debug, Clone, PartialEq)]
pub struct QkvOut {
    pub q: Vec<bf16>,
    pub k_row: Vec<bf16>,
    pub v_row: Vec<bf16>,
    pub rest_untouched: bool,
}

/// Plain values whose bytes are their little-endian device representation: every bit pattern
/// is a value and there is no padding. Only bf16, f32, u32 and u8 qualify; a type with
/// invalid bit patterns must not be filled from raw bytes in safe code, so this does not
/// compile:
///
/// ```compile_fail
/// use x0ne_core_runtime::ops::{as_bytes_mut, Pod};
/// #[derive(Clone, Copy, Default)]
/// struct Flag(bool);
/// impl Pod for Flag {}
/// let mut v = [Flag(false)];
/// as_bytes_mut(&mut v)[0] = 2; // not a bool
/// ```
///
/// The same bytes written into a u8 compile and run:
///
/// ```
/// use x0ne_core_runtime::ops::as_bytes_mut;
/// let mut v = [0u8];
/// as_bytes_mut(&mut v)[0] = 2;
/// assert_eq!(v, [2]);
/// ```
pub trait Pod: sealed::Sealed + Copy + Default {}
impl Pod for bf16 {}
impl Pod for f32 {}
impl Pod for u32 {}
impl Pod for u8 {}

/// Seals `Pod`: the supertrait cannot be named outside this crate, so no other crate can
/// implement `Pod`.
mod sealed {
    pub trait Sealed {}
    impl Sealed for half::bf16 {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
    impl Sealed for u8 {}
}

pub fn as_bytes<T: Pod>(v: &[T]) -> &[u8] {
    // SAFETY: Pod is sealed to bf16, f32, u32 and u8: no padding, no invalid bit pattern; the
    // byte view covers the same allocation and borrows it. x86-64 is little-endian.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

pub fn as_bytes_mut<T: Pod>(v: &mut [T]) -> &mut [u8] {
    // SAFETY: as above; every byte pattern is a valid bf16, f32, u32 or u8, and the mutable
    // borrow of `v` is held by the byte view.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast(), std::mem::size_of_val(v)) }
}

pub fn read_vec<T: Pod>(
    a: &Arena<Frozen<DeviceBacking>>,
    s: Slice,
    at: u64,
    n: usize,
) -> Result<Vec<T>> {
    let mut v = vec![T::default(); n];
    a.read(s, at, as_bytes_mut(&mut v))?;
    Ok(v)
}

/// 1 / theta^(2i / d) in f32, the formula of transformers and of the oracle.
pub fn inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
        .collect()
}

/// Softmax scale of Qwen3 attention, head_dim^-1/2.
pub fn attn_scale(head_dim: usize) -> f32 {
    (1.0 / (head_dim as f64).sqrt()) as f32
}

/// cuBLAS workspace given to every handle (cuBLAS documents 32 MiB for Hopper and later).
pub const CUBLAS_WORKSPACE: u64 = 32 << 20;

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Format(format!("{what} {v} exceeds u32")))
}

/// GPU 0 with the cubin for its compute capability.
pub struct Kernels {
    gpu: Gpu,
}

impl Kernels {
    pub fn new() -> Result<Self> {
        Ok(Kernels { gpu: Gpu::new()? })
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    fn freeze(&self, a: Arena<crate::arena::Building>) -> Result<Arena<Frozen<DeviceBacking>>> {
        a.freeze(|len| self.gpu.allocate(len))
    }

    /// Row `token` of a [vocab, hidden] table; the token id is read from device memory.
    pub fn embedding(&self, table: &[bf16], hidden: usize, token: u32) -> Result<Vec<bf16>> {
        if hidden == 0 || !hidden.is_multiple_of(8) || !table.len().is_multiple_of(hidden) {
            return format_err(format!(
                "embedding: hidden {hidden}, table of {}",
                table.len()
            ));
        }
        let vocab = table.len() / hidden;
        let mut a = Arena::new();
        let t = a.alloc("table", (table.len() * 2) as u64, Class::Weights)?;
        let tok = a.alloc("token", 4, Class::ClientData)?;
        let out = a.alloc("out", (hidden * 2) as u64, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.write(t, 0, as_bytes(table))?;
        f.write(tok, 0, &token.to_le_bytes())?;
        let g = &self.gpu;
        // SAFETY: slices of `f`, sized above: a [vocab][hidden] table, one token, one row out.
        unsafe {
            launch!(g, g.f.embedding, (1, 1), 128; f.ptr(t)?, f.ptr(tok)?, f.ptr(out)?, u32_of(hidden, "hidden")?, u32_of(vocab, "vocab")?)
        }?;
        g.sync()?;
        read_vec(&f, out, 0, hidden)
    }

    /// residual += x (f32), then out = rmsnorm(residual) * w in bf16. Returns (residual, out).
    pub fn add_rmsnorm(
        &self,
        residual: &[f32],
        x: &[bf16],
        w: &[bf16],
        eps: f32,
    ) -> Result<(Vec<f32>, Vec<bf16>)> {
        let h = residual.len();
        if h == 0 || x.len() != h || w.len() != h {
            return format_err("add_rmsnorm: lengths differ");
        }
        let mut a = Arena::new();
        let r = a.alloc("residual", (h * 4) as u64, Class::ClientData)?;
        let xs = a.alloc("x", (h * 2) as u64, Class::ClientData)?;
        let ws = a.alloc("w", (h * 2) as u64, Class::Weights)?;
        let out = a.alloc("out", (h * 2) as u64, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.write(r, 0, as_bytes(residual))?;
        f.write(xs, 0, as_bytes(x))?;
        f.write(ws, 0, as_bytes(w))?;
        let g = &self.gpu;
        // SAFETY: slices of `f`, sized above for one row of `h` values.
        unsafe {
            launch!(g, g.f.add_rmsnorm, (1, 1), 256; f.ptr(r)?, f.ptr(xs)?, f.ptr(ws)?, f.ptr(out)?, u32_of(h, "hidden")?, eps)
        }?;
        g.sync()?;
        Ok((read_vec(&f, r, 0, h)?, read_vec(&f, out, 0, h)?))
    }

    /// Per-head RMSNorm of q and k, RoPE (rotate-half, as transformers), then k and v
    /// written into the cache at `at`. `qkv` is one row of the packed [q; k; v] output.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_kv_write(
        &self,
        qkv: &[bf16],
        q_norm: &[bf16],
        k_norm: &[bf16],
        heads: Heads,
        theta: f32,
        eps: f32,
        at: CacheAt,
    ) -> Result<QkvOut> {
        heads.check()?;
        let (hd, qd, kvd) = (heads.head_dim, heads.q_dim(), heads.kv_dim());
        if qkv.len() != qd + 2 * kvd || q_norm.len() != hd || k_norm.len() != hd {
            return format_err("qk_norm_rope_kv_write: lengths");
        }
        if at.layer >= at.layers || at.pos >= at.max_ctx {
            return format_err(format!("qk_norm_rope_kv_write: {at:?} out of range"));
        }
        const SENTINEL: u8 = 0xA5;
        let plane = at.max_ctx * kvd * 2; // bytes of one K or V plane of one layer
        let cache_bytes = at.layers * 2 * plane;
        let mut a = Arena::new();
        let x = a.alloc("qkv", (qkv.len() * 2) as u64, Class::ClientData)?;
        let qn = a.alloc("q_norm", (hd * 2) as u64, Class::Weights)?;
        let kn = a.alloc("k_norm", (hd * 2) as u64, Class::Weights)?;
        let inv = a.alloc("inv_freq", (hd / 2 * 4) as u64, Class::Weights)?;
        let q = a.alloc("q", (qd * 2) as u64, Class::ClientData)?;
        let cache = a.alloc("cache", cache_bytes as u64, Class::ClientData)?;
        let pos = a.alloc("pos", 4, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.write(x, 0, as_bytes(qkv))?;
        f.write(qn, 0, as_bytes(q_norm))?;
        f.write(kn, 0, as_bytes(k_norm))?;
        f.write(inv, 0, as_bytes(&inv_freq(hd, theta)))?;
        f.write(cache, 0, &vec![SENTINEL; cache_bytes])?;
        f.write(pos, 0, &u32_of(at.pos, "pos")?.to_le_bytes())?;
        let k_off = (at.layer * 2 * plane) as u64;
        let v_off = k_off + plane as u64;
        let g = &self.gpu;
        // SAFETY: slices of `f`, sized above for these heads; the K and V planes of layer
        // `at.layer` hold `max_ctx` rows and `at.pos < max_ctx`, checked above.
        unsafe {
            launch!(g, g.f.qk_norm_rope_kv_write, (u32_of(heads.q_heads + heads.kv_heads, "heads")?, 1), 64;
                f.ptr(x)?, f.ptr(qn)?, f.ptr(kn)?, f.ptr(inv)?, f.ptr(q)?, f.addr(cache, k_off)?, f.addr(cache, v_off)?,
                f.ptr(pos)?, u32_of(heads.q_heads, "q_heads")?, u32_of(heads.kv_heads, "kv_heads")?,
                u32_of(at.max_ctx, "max_ctx")?, eps)
        }?;
        g.sync()?;
        let q_out: Vec<bf16> = read_vec(&f, q, 0, qd)?;
        let mut all = vec![0u8; cache_bytes];
        f.read(cache, 0, &mut all)?;
        let row = |off: usize| -> Vec<bf16> {
            all[off..off + kvd * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| bf16::from_bits(u16::from_le_bytes(b)))
                .collect()
        };
        let (k_at, v_at) = (
            k_off as usize + at.pos * kvd * 2,
            v_off as usize + at.pos * kvd * 2,
        );
        let k_row = row(k_at);
        let v_row = row(v_at);
        let rest_untouched = all.iter().enumerate().all(|(i, &b)| {
            (k_at..k_at + kvd * 2).contains(&i)
                || (v_at..v_at + kvd * 2).contains(&i)
                || b == SENTINEL
        });
        Ok(QkvOut {
            q: q_out,
            k_row,
            v_row,
            rest_untouched,
        })
    }

    fn gemv(&self, w: &[bf16], rows: usize, cols: usize, x: &[bf16], out: Out) -> Result<Vec<u8>> {
        if rows == 0 || cols == 0 || w.len() != rows * cols || x.len() != cols {
            return format_err(format!(
                "gemv: W {} for {rows}x{cols}, x {}",
                w.len(),
                x.len()
            ));
        }
        let ob = match out {
            Out::Bf16 => 2,
            Out::F32 => 4,
        };
        let mut a = Arena::new();
        let ws = a.alloc("w", (w.len() * 2) as u64, Class::Weights)?;
        let xs = a.alloc("x", (cols * 2) as u64, Class::ClientData)?;
        let ys = a.alloc("y", (rows * ob) as u64, Class::ClientData)?;
        let work = a.alloc("cublas_workspace", CUBLAS_WORKSPACE, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.write(ws, 0, as_bytes(w))?;
        f.write(xs, 0, as_bytes(x))?;
        let g = &self.gpu;
        // SAFETY: `work` is a CUBLAS_WORKSPACE-byte slice of `f`, which outlives the GEMM below
        // (synchronized before return); W, x and y are slices of `f` sized above, y apart.
        unsafe {
            g.set_cublas_workspace(f.ptr(work)?, CUBLAS_WORKSPACE)?;
            g.gemm(f.ptr(ws)?, rows, cols, f.ptr(xs)?, 1, f.ptr(ys)?, out)?;
        }
        g.sync()?;
        let mut y = vec![0u8; rows * ob];
        f.read(ys, 0, &mut y)?;
        Ok(y)
    }

    /// y = W x for W [rows, cols] row-major, f32 accumulation, f32 out (logits).
    pub fn gemv_f32_out(
        &self,
        w: &[bf16],
        rows: usize,
        cols: usize,
        x: &[bf16],
    ) -> Result<Vec<f32>> {
        let y = self.gemv(w, rows, cols, x, Out::F32)?;
        Ok(y.as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect())
    }

    /// y = W x for W [rows, cols] row-major, f32 accumulation, bf16 out (activations).
    pub fn gemv_bf16_out(
        &self,
        w: &[bf16],
        rows: usize,
        cols: usize,
        x: &[bf16],
    ) -> Result<Vec<bf16>> {
        let y = self.gemv(w, rows, cols, x, Out::Bf16)?;
        Ok(y.as_chunks::<2>()
            .0
            .iter()
            .map(|&b| bf16::from_bits(u16::from_le_bytes(b)))
            .collect())
    }

    /// One query position against `ctx` cached positions (layout `[pos][kv_heads][head_dim]`),
    /// GQA, scale 1/sqrt(head_dim), softmax in f32. Returns [q_heads * head_dim].
    pub fn attention_decode(
        &self,
        q: &[bf16],
        k: &[bf16],
        v: &[bf16],
        ctx: usize,
        h: Heads,
    ) -> Result<Vec<bf16>> {
        h.check()?;
        let (qd, kvd) = (h.q_dim(), h.kv_dim());
        if ctx == 0 || q.len() != qd || k.len() < ctx * kvd || v.len() < ctx * kvd {
            return format_err("attention_decode: lengths");
        }
        let mut a = Arena::new();
        let qs = a.alloc("q", (qd * 2) as u64, Class::ClientData)?;
        let ks = a.alloc("k", (ctx * kvd * 2) as u64, Class::ClientData)?;
        let vs = a.alloc("v", (ctx * kvd * 2) as u64, Class::ClientData)?;
        let out = a.alloc("out", (qd * 2) as u64, Class::ClientData)?;
        let pos = a.alloc("pos", 4, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.write(qs, 0, as_bytes(q))?;
        f.write(ks, 0, as_bytes(&k[..ctx * kvd]))?;
        f.write(vs, 0, as_bytes(&v[..ctx * kvd]))?;
        f.write(pos, 0, &u32_of(ctx - 1, "pos")?.to_le_bytes())?;
        let g = &self.gpu;
        // SAFETY: slices of `f`, sized above for `ctx` positions of these heads; pos = ctx - 1.
        unsafe {
            launch!(g, g.f.attention_decode, (u32_of(h.q_heads, "q_heads")?, 1), 256;
                f.ptr(qs)?, f.ptr(ks)?, f.ptr(vs)?, f.ptr(out)?, f.ptr(pos)?,
                u32_of(h.q_heads, "q_heads")?, u32_of(h.kv_heads, "kv_heads")?, u32_of(ctx, "ctx")?, attn_scale(h.head_dim))
        }?;
        g.sync()?;
        read_vec(&f, out, 0, qd)
    }

    /// P, the number of split-K slices for this GPU and geometry. The output of the split-K
    /// kernel depends on the inputs, ctx and P only.
    pub fn split_count(&self, h: Heads) -> Result<u32> {
        Ok(self.split_plan(h)?.splits)
    }

    /// The split-K launch for this geometry (`Gpu::split_plan`).
    pub fn split_plan(&self, h: Heads) -> Result<SplitPlan> {
        h.check()?;
        self.gpu.split_plan(h.q_heads, h.kv_heads)
    }

    /// `attention_decode` computed by the split-K kernel (E3) instead of E2's, with the merge
    /// the engine uses by default.
    pub fn attention_decode_split(
        &self,
        q: &[bf16],
        k: &[bf16],
        v: &[bf16],
        ctx: usize,
        h: Heads,
    ) -> Result<Vec<bf16>> {
        self.attention_decode_split_merge(q, k, v, ctx, h, crate::engine::DEFAULT_SPLIT_MERGE)
    }

    /// `attention_decode_split` with the merge given.
    pub fn attention_decode_split_merge(
        &self,
        q: &[bf16],
        k: &[bf16],
        v: &[bf16],
        ctx: usize,
        h: Heads,
        merge: SplitMerge,
    ) -> Result<Vec<bf16>> {
        h.check()?;
        let (qd, kvd) = (h.q_dim(), h.kv_dim());
        if ctx == 0 || q.len() != qd || k.len() < ctx * kvd || v.len() < ctx * kvd {
            return format_err("attention_decode_split: lengths");
        }
        let plan = self.split_plan(h)?;
        let mut a = Arena::new();
        let qs = a.alloc("q", (qd * 2) as u64, Class::ClientData)?;
        let ks = a.alloc("k", (ctx * kvd * 2) as u64, Class::ClientData)?;
        let vs = a.alloc("v", (ctx * kvd * 2) as u64, Class::ClientData)?;
        let out = a.alloc("out", (qd * 2) as u64, Class::ClientData)?;
        let pos = a.alloc("pos", 4, Class::ClientData)?;
        let state = a.alloc(
            "split_state",
            split_state_bytes(h.q_heads, plan.splits),
            Class::ClientData,
        )?;
        let counter = a.alloc("split_counter", (h.kv_heads * 4) as u64, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.scrub_client_data()?;
        f.write(qs, 0, as_bytes(q))?;
        f.write(ks, 0, as_bytes(&k[..ctx * kvd]))?;
        f.write(vs, 0, as_bytes(&v[..ctx * kvd]))?;
        f.write(pos, 0, &u32_of(ctx - 1, "pos")?.to_le_bytes())?;
        let g = &self.gpu;
        let (qh, kvh, max_ctx) = (
            u32_of(h.q_heads, "q_heads")?,
            u32_of(h.kv_heads, "kv_heads")?,
            u32_of(ctx, "ctx")?,
        );
        // SAFETY: slices of `f`, sized above for `ctx` positions of these heads and the plan's
        // P; the counters were zeroed by the scrub; pos = ctx - 1.
        unsafe {
            launch_split(
                g,
                &plan,
                merge,
                f.ptr(qs)?,
                f.ptr(ks)?,
                f.ptr(vs)?,
                f.ptr(state)?,
                f.ptr(counter)?,
                f.ptr(out)?,
                f.ptr(pos)?,
                qh,
                kvh,
                max_ctx,
                attn_scale(h.head_dim),
            )
        }?;
        g.sync()?;
        let counts: Vec<u32> = read_vec(&f, counter, 0, h.kv_heads)?;
        if counts.iter().any(|&c| c != 0) {
            return Err(Error::Gpu(format!(
                "split-K counters not reset: {counts:?}"
            )));
        }
        read_vec(&f, out, 0, qd)
    }

    /// P3: the decode attention alone over `layers` KV caches of `ctx` positions (as in one
    /// decode step of the model), `rounds` times, wall clock after a warm-up round. The cache
    /// holds a constant (the time does not depend on the values). `splits` overrides P for a
    /// sweep (`None`: the plan's). Returns the seconds of the `rounds`, the KV bytes they read
    /// (each K and V row once per kv head and layer) and the P used.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_timed(
        &self,
        kind: DecodeAttention,
        merge: SplitMerge,
        h: Heads,
        ctx: usize,
        layers: usize,
        rounds: usize,
        splits: Option<u32>,
    ) -> Result<(f64, u64, u32)> {
        h.check()?;
        let (qd, kvd) = (h.q_dim(), h.kv_dim());
        if ctx == 0 || layers == 0 || rounds == 0 || splits == Some(0) {
            return format_err(
                "attention_decode_timed: ctx, layers, rounds and splits must be positive",
            );
        }
        let mut plan = self.split_plan(h)?;
        plan.splits = splits.unwrap_or(plan.splits);
        let p = plan.splits;
        let plane = (ctx * kvd * 2) as u64;
        let mut a = Arena::new();
        let qs = a.alloc("q", (qd * 2) as u64, Class::ClientData)?;
        let kv = a.alloc("kv", layers as u64 * 2 * plane, Class::ClientData)?;
        let out = a.alloc("out", (qd * 2) as u64, Class::ClientData)?;
        let pos = a.alloc("pos", 4, Class::ClientData)?;
        let state = a.alloc(
            "split_state",
            split_state_bytes(h.q_heads, p),
            Class::ClientData,
        )?;
        let counter = a.alloc("split_counter", (h.kv_heads * 4) as u64, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.scrub_client_data()?;
        let g = &self.gpu;
        // bf16 0x3C00 = 2^-7, two per word.
        // SAFETY: `qs` holds qd bf16 (qd / 2 words), `kv` 2 * layers planes (plane / 4 words each).
        unsafe {
            g.memset_u32(f.ptr(qs)?, 0x3C00_3C00, qd / 2)?;
            g.memset_u32(
                f.ptr(kv)?,
                0x3C00_3C00,
                (layers as u64 * 2 * plane / 4) as usize,
            )?;
        }
        f.write(pos, 0, &u32_of(ctx - 1, "pos")?.to_le_bytes())?;
        let (qh, kvh, max_ctx, scale) = (
            u32_of(h.q_heads, "q_heads")?,
            u32_of(h.kv_heads, "kv_heads")?,
            u32_of(ctx, "ctx")?,
            attn_scale(h.head_dim),
        );
        let round = || -> Result<()> {
            for l in 0..layers {
                let (kc, vc) = (
                    f.addr(kv, 2 * l as u64 * plane)?,
                    f.addr(kv, (2 * l as u64 + 1) * plane)?,
                );
                match kind {
                    // SAFETY: slices of `f`, sized above: layer l's K and V planes of `ctx` rows.
                    DecodeAttention::PerHead => unsafe {
                        launch!(g, g.f.attention_decode, (qh, 1), 256; f.ptr(qs)?, kc, vc, f.ptr(out)?, f.ptr(pos)?, qh, kvh, max_ctx, scale)
                    }?,
                    // SAFETY: as above, with the states for P = `p` (the plan's or the override)
                    // and the counters zeroed by the scrub.
                    DecodeAttention::SplitK => unsafe {
                        launch_split(
                            g,
                            &plan,
                            merge,
                            f.ptr(qs)?,
                            kc,
                            vc,
                            f.ptr(state)?,
                            f.ptr(counter)?,
                            f.ptr(out)?,
                            f.ptr(pos)?,
                            qh,
                            kvh,
                            max_ctx,
                            scale,
                        )
                    }?,
                }
            }
            Ok(())
        };
        round()?;
        g.sync()?;
        let t0 = std::time::Instant::now();
        for _ in 0..rounds {
            round()?;
        }
        g.sync()?;
        let secs = t0.elapsed().as_secs_f64();
        Ok((secs, rounds as u64 * layers as u64 * 2 * plane, p))
    }

    /// Causal attention over `t` positions, computed whole on the GPU; returns the
    /// requested rows only ([q_heads * head_dim] each).
    pub fn attention_prefill_rows(
        &self,
        q: &[bf16],
        k: &[bf16],
        v: &[bf16],
        t: usize,
        h: Heads,
        rows: &[usize],
    ) -> Result<Vec<Vec<bf16>>> {
        h.check()?;
        let (qd, kvd) = (h.q_dim(), h.kv_dim());
        if t == 0
            || q.len() != t * qd
            || k.len() != t * kvd
            || v.len() != t * kvd
            || rows.iter().any(|&r| r >= t)
        {
            return format_err("attention_prefill_rows: lengths");
        }
        let mut a = Arena::new();
        let qs = a.alloc("q", (t * qd * 2) as u64, Class::ClientData)?;
        let ks = a.alloc("k", (t * kvd * 2) as u64, Class::ClientData)?;
        let vs = a.alloc("v", (t * kvd * 2) as u64, Class::ClientData)?;
        let out = a.alloc("out", (t * qd * 2) as u64, Class::ClientData)?;
        let pos = a.alloc("pos0", 4, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.write(qs, 0, as_bytes(q))?;
        f.write(ks, 0, as_bytes(k))?;
        f.write(vs, 0, as_bytes(v))?;
        f.write(pos, 0, &0u32.to_le_bytes())?;
        let g = &self.gpu;
        // SAFETY: slices of `f`, sized above for `t` rows and positions of these heads.
        unsafe {
            launch!(g, g.f.attention_prefill, (u32_of(h.q_heads, "q_heads")?, u32_of(t.div_ceil(8), "row blocks")?), 256;
                f.ptr(qs)?, f.ptr(ks)?, f.ptr(vs)?, f.ptr(out)?, f.ptr(pos)?, u32_of(t, "rows")?,
                u32_of(h.q_heads, "q_heads")?, u32_of(h.kv_heads, "kv_heads")?, u32_of(t, "max_ctx")?, attn_scale(h.head_dim))
        }?;
        g.sync()?;
        rows.iter()
            .map(|&r| read_vec(&f, out, (r * qd * 2) as u64, qd))
            .collect()
    }

    /// Causal attention for positions pos0..t of a t-position cache, in one launch at pos0:
    /// how the engine runs its second and later prefill chunks. `q` holds those t - pos0
    /// rows; `rows` index into them. Returns [q_heads * head_dim] per requested row.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_at(
        &self,
        q: &[bf16],
        k: &[bf16],
        v: &[bf16],
        t: usize,
        pos0: usize,
        h: Heads,
        rows: &[usize],
    ) -> Result<Vec<Vec<bf16>>> {
        h.check()?;
        let (qd, kvd) = (h.q_dim(), h.kv_dim());
        let m = t.saturating_sub(pos0);
        if m == 0
            || q.len() != m * qd
            || k.len() != t * kvd
            || v.len() != t * kvd
            || rows.iter().any(|&r| r >= m)
        {
            return format_err("attention_prefill_at: lengths");
        }
        let mut a = Arena::new();
        let qs = a.alloc("q", (m * qd * 2) as u64, Class::ClientData)?;
        let ks = a.alloc("k", (t * kvd * 2) as u64, Class::ClientData)?;
        let vs = a.alloc("v", (t * kvd * 2) as u64, Class::ClientData)?;
        let out = a.alloc("out", (m * qd * 2) as u64, Class::ClientData)?;
        let pos = a.alloc("pos0", 4, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.write(qs, 0, as_bytes(q))?;
        f.write(ks, 0, as_bytes(k))?;
        f.write(vs, 0, as_bytes(v))?;
        f.write(pos, 0, &u32_of(pos0, "pos0")?.to_le_bytes())?;
        let g = &self.gpu;
        // SAFETY: slices of `f`, sized above: `m` query and output rows, `t` cache positions.
        unsafe {
            launch!(g, g.f.attention_prefill, (u32_of(h.q_heads, "q_heads")?, u32_of(m.div_ceil(8), "row blocks")?), 256;
                f.ptr(qs)?, f.ptr(ks)?, f.ptr(vs)?, f.ptr(out)?, f.ptr(pos)?, u32_of(m, "rows")?,
                u32_of(h.q_heads, "q_heads")?, u32_of(h.kv_heads, "kv_heads")?, u32_of(t, "max_ctx")?, attn_scale(h.head_dim))
        }?;
        g.sync()?;
        rows.iter()
            .map(|&r| read_vec(&f, out, (r * qd * 2) as u64, qd))
            .collect()
    }

    /// silu(gate) * up on one row of the packed [gate; up] output.
    pub fn swiglu(&self, gate_up: &[bf16]) -> Result<Vec<bf16>> {
        if gate_up.is_empty() || !gate_up.len().is_multiple_of(2) {
            return format_err("swiglu: odd length");
        }
        let inter = gate_up.len() / 2;
        let mut a = Arena::new();
        let gu = a.alloc("gate_up", (gate_up.len() * 2) as u64, Class::ClientData)?;
        let out = a.alloc("out", (inter * 2) as u64, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.write(gu, 0, as_bytes(gate_up))?;
        let g = &self.gpu;
        // SAFETY: slices of `f`, sized above: 2 * inter values in, inter out.
        unsafe {
            launch!(g, g.f.swiglu, (u32_of(inter.div_ceil(256), "blocks")?, 1), 256; f.ptr(gu)?, f.ptr(out)?, u32_of(inter, "inter")?)
        }?;
        g.sync()?;
        read_vec(&f, out, 0, inter)
    }

    /// GM1 routing of `logits.len() / experts` rows of f32 router logits: the `top_k` chosen
    /// experts of each row (larger logit first, then the smaller index) and their renormalised
    /// weights, both `[rows][top_k]`. A row with a non-finite logit gets `MOE_NO_EXPERT` and NaN.
    pub fn moe_route(
        &self,
        logits: &[f32],
        experts: usize,
        top_k: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        if experts == 0
            || experts > MOE_MAX_EXPERTS
            || top_k == 0
            || top_k > 32
            || top_k > experts
            || logits.is_empty()
            || !logits.len().is_multiple_of(experts)
        {
            return format_err(format!(
                "moe_route: {} logits, {experts} experts, top {top_k}",
                logits.len()
            ));
        }
        let rows = logits.len() / experts;
        let mut a = Arena::new();
        let l = a.alloc("logits", (logits.len() * 4) as u64, Class::ClientData)?;
        let ids = a.alloc("ids", (rows * top_k * 4) as u64, Class::ClientData)?;
        let w = a.alloc("weights", (rows * top_k * 4) as u64, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.write(l, 0, as_bytes(logits))?;
        let g = &self.gpu;
        // SAFETY: slices of `f`, sized above: rows x experts logits, rows x top_k ids and weights;
        // experts <= MOE_MAX_EXPERTS and top_k <= 32 <= the 128 threads (checked above).
        unsafe {
            launch!(g, g.f.moe_route, (u32_of(rows, "rows")?, 1), 128;
                f.ptr(l)?, u32_of(experts, "experts")?, u32_of(top_k, "top_k")?, f.ptr(ids)?, f.ptr(w)?)
        }?;
        g.sync()?;
        Ok((
            read_vec(&f, ids, 0, rows * top_k)?,
            read_vec(&f, w, 0, rows * top_k)?,
        ))
    }

    /// GM1 expert GEMVs: for each slot s of `ids` (written to device memory, read there by the
    /// kernel), `y[s] = W[ids[s]] x[s / x_group]`, W `[experts][rows][cols]` bf16 and x
    /// `[slots / x_group][cols]` bf16. f32 accumulation; y in bf16 (returned as f32) or f32.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gemv(
        &self,
        w: &[bf16],
        experts: usize,
        rows: usize,
        cols: usize,
        ids: &[u32],
        x: &[bf16],
        x_group: usize,
        f32_out: bool,
    ) -> Result<Vec<f32>> {
        let slots = ids.len();
        if experts == 0
            || rows == 0
            || cols == 0
            || !cols.is_multiple_of(8)
            || w.len() != experts * rows * cols
            || slots == 0
            || x_group == 0
            || !slots.is_multiple_of(x_group)
            || x.len() != slots / x_group * cols
        {
            return format_err(format!(
                "moe_gemv: W {} for {experts}x{rows}x{cols}, {slots} slots, x {} in groups of {x_group}",
                w.len(),
                x.len()
            ));
        }
        let ob = if f32_out { 4 } else { 2 };
        let mut a = Arena::new();
        let ws = a.alloc("w", (w.len() * 2) as u64, Class::Weights)?;
        let is = a.alloc("ids", (slots * 4) as u64, Class::ClientData)?;
        let xs = a.alloc("x", (x.len() * 2) as u64, Class::ClientData)?;
        let ys = a.alloc("y", (slots * rows * ob) as u64, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.write(ws, 0, as_bytes(w))?;
        f.write(is, 0, as_bytes(ids))?;
        f.write(xs, 0, as_bytes(x))?;
        let g = &self.gpu;
        let k = if f32_out {
            g.f.moe_gemv_f32
        } else {
            g.f.moe_gemv_bf16
        };
        // SAFETY: slices of `f`, sized above: W [experts][rows][cols], slots ids, slots / x_group
        // input rows of cols (a multiple of 8, 16-byte loads from 256-byte aligned slices),
        // slots x rows outputs. An id >= experts makes the kernel write NaN, not read.
        unsafe {
            launch!(g, k, (u32_of(rows.div_ceil(8), "blocks")?, u32_of(slots, "slots")?), 256;
                f.ptr(ws)?, f.ptr(is)?, f.ptr(xs)?, f.ptr(ys)?, u32_of(rows, "rows")?, u32_of(cols, "cols")?,
                u32_of(experts, "experts")?, u32_of(x_group, "x_group")?, u32_of(cols, "x_stride")?)
        }?;
        g.sync()?;
        if f32_out {
            read_vec(&f, ys, 0, slots * rows)
        } else {
            let y: Vec<bf16> = read_vec(&f, ys, 0, slots * rows)?;
            Ok(y.iter().map(|v| v.to_f32()).collect())
        }
    }

    /// GM1 combination: `out[r] = bf16(sum_s weights[r][s] y[r][s])` for y `[rows][top_k][hidden]`
    /// f32 and weights `[rows][top_k]` f32; f32 accumulation in slot order.
    pub fn moe_combine(
        &self,
        y: &[f32],
        weights: &[f32],
        hidden: usize,
        top_k: usize,
    ) -> Result<Vec<bf16>> {
        if hidden == 0
            || top_k == 0
            || weights.is_empty()
            || !weights.len().is_multiple_of(top_k)
            || y.len() != weights.len() * hidden
        {
            return format_err(format!(
                "moe_combine: y {}, weights {}, hidden {hidden}, top {top_k}",
                y.len(),
                weights.len()
            ));
        }
        let rows = weights.len() / top_k;
        let mut a = Arena::new();
        let ys = a.alloc("y", (y.len() * 4) as u64, Class::ClientData)?;
        let ws = a.alloc("weights", (weights.len() * 4) as u64, Class::ClientData)?;
        let out = a.alloc("out", (rows * hidden * 2) as u64, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.write(ys, 0, as_bytes(y))?;
        f.write(ws, 0, as_bytes(weights))?;
        let g = &self.gpu;
        // SAFETY: slices of `f`, sized above: rows x top_k x hidden f32, rows x top_k weights,
        // rows x hidden bf16 out.
        unsafe {
            launch!(g, g.f.moe_combine, (u32_of(hidden.div_ceil(256), "blocks")?, u32_of(rows, "rows")?), 256;
                f.ptr(ys)?, f.ptr(ws)?, f.ptr(out)?, u32_of(hidden, "hidden")?, u32_of(top_k, "top_k")?)
        }?;
        g.sync()?;
        read_vec(&f, out, 0, rows * hidden)
    }

    /// silu(gate) * up on `rows` packed [gate; up] rows, as the MoE block launches it.
    pub fn swiglu_rows(&self, gate_up: &[bf16], rows: usize) -> Result<Vec<bf16>> {
        if rows == 0 || gate_up.is_empty() || !gate_up.len().is_multiple_of(2 * rows) {
            return format_err("swiglu_rows: length is not rows x 2 x inter");
        }
        let inter = gate_up.len() / rows / 2;
        let mut a = Arena::new();
        let gu = a.alloc("gate_up", (gate_up.len() * 2) as u64, Class::ClientData)?;
        let out = a.alloc("out", (rows * inter * 2) as u64, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.write(gu, 0, as_bytes(gate_up))?;
        let g = &self.gpu;
        // SAFETY: slices of `f`, sized above: rows x 2 x inter values in, rows x inter out.
        unsafe {
            launch!(g, g.f.swiglu, (u32_of(inter.div_ceil(256), "blocks")?, u32_of(rows, "rows")?), 256; f.ptr(gu)?, f.ptr(out)?, u32_of(inter, "inter")?)
        }?;
        g.sync()?;
        read_vec(&f, out, 0, rows * inter)
    }

    /// `value + 1`, computed by the position-advance kernel on one device word.
    pub fn advance(&self, value: u32) -> Result<u32> {
        let mut a = Arena::new();
        let pos = a.alloc("pos", 4, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.write(pos, 0, &value.to_le_bytes())?;
        let g = &self.gpu;
        // SAFETY: one u32 slice of `f`.
        unsafe { launch!(g, g.f.advance, (1, 1), 32; f.ptr(pos)?) }?;
        g.sync()?;
        let v: Vec<u32> = read_vec(&f, pos, 0, 1)?;
        Ok(v[0])
    }

    /// Index of the largest logit; ties go to the lowest index (torch.argmax). The
    /// result is written to device memory, then read back here.
    pub fn argmax(&self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() {
            return format_err("argmax: empty");
        }
        let mut a = Arena::new();
        let l = a.alloc("logits", (logits.len() * 4) as u64, Class::ClientData)?;
        let out = a.alloc("out", 4, Class::ClientData)?;
        let mut f = self.freeze(a)?;
        f.write(l, 0, as_bytes(logits))?;
        let g = &self.gpu;
        // SAFETY: slices of `f`, sized above: one row of logits.len() f32, one u32 out.
        unsafe {
            launch!(g, g.f.argmax, (1, 1), 1024; f.ptr(l)?, u32_of(logits.len(), "vocab")?, f.ptr(out)?)
        }?;
        g.sync()?;
        let v: Vec<u32> = read_vec(&f, out, 0, 1)?;
        if v[0] as usize >= logits.len() {
            return Err(Error::Gpu("argmax: no finite maximum (NaN row)".into()));
        }
        Ok(v[0])
    }
}
