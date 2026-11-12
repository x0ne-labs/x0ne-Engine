//! CPU reference of x0ne Core's kernels and of its Qwen3 dense forward, written in x0ne Engine
//! (x0ne Core keeps its own oracle with its own tests; `x0ne-core/PROVENANCE.md`).
//!
//! Plain loops, one value at a time, f64 inside. Every function takes and returns f32: a value
//! the device holds in bf16 is given here as the f32 it converts to, and a result the device
//! rounds to bf16 is returned unrounded (`bf16::round` rounds it). Nothing here shares code with
//! the runtime or with a bf16 crate: a test that compares the two compares two implementations.
//!
//! The functions follow the kernels' definitions in `kernels/cuda/*.cu`, their order of
//! accumulation aside: where a kernel sums in f32 in a fixed order, the reference sums in f64, and
//! the tests state the tolerance between the two.
#![forbid(unsafe_code)]

pub mod bf16;
pub mod model;

/// Attention geometry: q head `h` reads kv head `h / (q_heads / kv_heads)`.
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
    fn kv_head_of(&self, q_head: usize) -> usize {
        q_head / (self.q_heads / self.kv_heads)
    }
}

/// Row `token` of a `[vocab][hidden]` table; a token outside the table gives a zero row, as the
/// kernel does.
pub fn embedding(table: &[f32], hidden: usize, token: u32) -> Vec<f32> {
    assert!(hidden > 0 && table.len().is_multiple_of(hidden));
    let t = token as usize;
    if t >= table.len() / hidden {
        return vec![0.0; hidden];
    }
    table[t * hidden..(t + 1) * hidden].to_vec()
}

/// `residual += x` in f32 (one IEEE addition per value, as the kernel), then
/// `residual / rms(residual) * w`, with `rms = sqrt(mean(residual^2) + eps)`.
pub fn add_rmsnorm(residual: &mut [f32], x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let h = residual.len();
    assert!(h > 0 && x.len() == h && w.len() == h);
    for (r, &x) in residual.iter_mut().zip(x) {
        *r += x;
    }
    let scale = rms_scale(residual, eps);
    residual
        .iter()
        .zip(w)
        .map(|(&r, &w)| (r as f64 * scale * w as f64) as f32)
        .collect()
}

fn rms_scale(x: &[f32], eps: f32) -> f64 {
    let ss: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
    1.0 / (ss / x.len() as f64 + eps as f64).sqrt()
}

/// `1 / theta^(2i / d)` for i in 0..d/2, in f64. The device receives these values in f32 from
/// the host (`ops::inv_freq`): a test compares the two, and `qk_norm_rope` takes the f32 values
/// the device was given.
pub fn inv_freq(head_dim: usize, theta: f64) -> Vec<f64> {
    (0..head_dim / 2)
        .map(|i| 1.0 / theta.powf((2 * i) as f64 / head_dim as f64))
        .collect()
}

/// One row of the packed `[q; k; v]` projection at position `pos`: RMSNorm of each q and k head
/// (weights `q_norm`, `k_norm`), then rotate-half RoPE of each: the pair `(i, i + d/2)` turns by
/// the angle `pos * inv_freq[i]`, the f32 product the kernel takes the sine and cosine of.
/// Returns (q, k, v); v is copied.
pub fn qk_norm_rope(
    qkv: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    inv_freq: &[f32],
    h: Heads,
    pos: u32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (hd, qd, kvd) = (h.head_dim, h.q_dim(), h.kv_dim());
    let half = hd / 2;
    assert!(hd.is_multiple_of(2) && qkv.len() == qd + 2 * kvd);
    assert!(q_norm.len() == hd && k_norm.len() == hd && inv_freq.len() == half);
    let head = |x: &[f32], w: &[f32]| -> Vec<f32> {
        let scale = rms_scale(x, eps);
        let n: Vec<f64> = x
            .iter()
            .zip(w)
            .map(|(&x, &w)| x as f64 * scale * w as f64)
            .collect();
        let mut out = vec![0f32; hd];
        for i in 0..half {
            let angle = (pos as f32 * inv_freq[i]) as f64;
            let (s, c) = angle.sin_cos();
            out[i] = (n[i] * c - n[i + half] * s) as f32;
            out[i + half] = (n[i + half] * c + n[i] * s) as f32;
        }
        out
    };
    let q = qkv[..qd]
        .chunks_exact(hd)
        .flat_map(|x| head(x, q_norm))
        .collect();
    let k = qkv[qd..qd + kvd]
        .chunks_exact(hd)
        .flat_map(|x| head(x, k_norm))
        .collect();
    (q, k, qkv[qd + kvd..].to_vec())
}

/// `y = W x` for W `[rows][cols]` row-major.
pub fn gemv(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    assert!(w.len() == rows * cols && x.len() == cols);
    w.chunks_exact(cols)
        .map(|r| {
            r.iter()
                .zip(x)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

/// `sum_j |W[i][j] x[j]|` per row: the magnitude an f32 accumulation's error is a share of.
pub fn gemv_magnitude(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f64> {
    assert!(w.len() == rows * cols && x.len() == cols);
    w.chunks_exact(cols)
        .map(|r| {
            r.iter()
                .zip(x)
                .map(|(&a, &b)| (a as f64 * b as f64).abs())
                .sum()
        })
        .collect()
}

/// Softmax attention of one query of `head_dim` values over `keys` rows of kv head `kh`:
/// scores `q . k_j / sqrt(head_dim)`, weights their softmax, result `sum_j p_j v_j`.
fn attend(q: &[f32], k: &[f32], v: &[f32], keys: usize, kh: usize, h: Heads) -> Vec<f32> {
    let (hd, kvd) = (h.head_dim, h.kv_dim());
    let scale = 1.0 / (hd as f64).sqrt();
    let row = |j: usize| j * kvd + kh * hd..j * kvd + (kh + 1) * hd;
    let scores: Vec<f64> = (0..keys)
        .map(|j| {
            let kr = &k[row(j)];
            q.iter()
                .zip(kr)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum::<f64>()
                * scale
        })
        .collect();
    let m = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let p: Vec<f64> = scores.iter().map(|&s| (s - m).exp()).collect();
    let l: f64 = p.iter().sum();
    let mut out = vec![0f64; hd];
    for (j, &pj) in p.iter().enumerate() {
        for (o, &x) in out.iter_mut().zip(&v[row(j)]) {
            *o += pj * x as f64;
        }
    }
    out.iter().map(|&o| (o / l) as f32).collect()
}

/// One query (`[q_heads][head_dim]`) over the first `ctx` positions of a cache laid out
/// `[position][kv_heads][head_dim]`. Returns `[q_heads][head_dim]`.
pub fn attention_decode(q: &[f32], k: &[f32], v: &[f32], ctx: usize, h: Heads) -> Vec<f32> {
    assert!(ctx > 0 && q.len() == h.q_dim());
    assert!(k.len() >= ctx * h.kv_dim() && v.len() >= ctx * h.kv_dim());
    q.chunks_exact(h.head_dim)
        .enumerate()
        .flat_map(|(head, qh)| attend(qh, k, v, ctx, h.kv_head_of(head), h))
        .collect()
}

/// Causal attention: row `r` of `q` (`[rows][q_heads][head_dim]`) sits at position `pos0 + r`
/// and reads keys `0..=pos0 + r` of the cache. Returns the rows asked for.
pub fn attention_rows(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    pos0: usize,
    h: Heads,
    rows: &[usize],
) -> Vec<Vec<f32>> {
    let qd = h.q_dim();
    rows.iter()
        .map(|&r| attention_decode(&q[r * qd..(r + 1) * qd], k, v, pos0 + r + 1, h))
        .collect()
}

/// `silu(gate) * up` on a packed `[gate; up]` row, with the kernel's formula
/// `g / (1 + e^-g)`: `g = -inf` gives NaN there and here.
pub fn swiglu(gate_up: &[f32]) -> Vec<f32> {
    assert!(gate_up.len().is_multiple_of(2));
    let inter = gate_up.len() / 2;
    (0..inter)
        .map(|i| {
            let (g, u) = (gate_up[i] as f64, gate_up[inter + i] as f64);
            (g / (1.0 + (-g).exp()) * u) as f32
        })
        .collect()
}

/// First index of the largest value (torch.argmax on a row without NaN). `None` when the row is
/// empty or holds a NaN: the kernel writes `0xFFFFFFFF` and the host refuses the token.
pub fn argmax(x: &[f32]) -> Option<u32> {
    let mut best: Option<usize> = None;
    for (i, v) in x.iter().enumerate() {
        if v.is_nan() {
            return None;
        }
        match best {
            Some(b) if x[b] >= *v => {}
            _ => best = Some(i),
        }
    }
    best.map(|i| i as u32)
}

/// `log(sum(exp(x)))`.
pub fn logsumexp(x: &[f32]) -> f64 {
    let m = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b as f64));
    if !m.is_finite() {
        return m;
    }
    m + x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
}

/// KL(p || q) in nats of two logits rows over one vocabulary.
pub fn kl(p: &[f32], q: &[f32]) -> f64 {
    assert_eq!(p.len(), q.len());
    let (lp, lq) = (logsumexp(p), logsumexp(q));
    p.iter()
        .zip(q)
        .map(|(&a, &b)| {
            let (a, b) = (a as f64 - lp, b as f64 - lq);
            a.exp() * (a - b)
        })
        .sum()
}

/// Expert index of a row whose routing is undefined.
pub const MOE_NO_EXPERT: u32 = u32::MAX;

/// Routing of `[rows][experts]` router logits: per row the `top_k` experts, larger logit first
/// then the smaller index, and `exp(l_e - l_max)` renormalised over the chosen. A row with a
/// non-finite logit gets `MOE_NO_EXPERT` and NaN.
pub fn moe_route(logits: &[f32], experts: usize, top_k: usize) -> (Vec<u32>, Vec<f32>) {
    assert!(experts > 0 && top_k > 0 && top_k <= experts);
    assert!(logits.len().is_multiple_of(experts));
    let (mut ids, mut weights) = (Vec::new(), Vec::new());
    for row in logits.chunks_exact(experts) {
        if row.iter().any(|v| !v.is_finite()) {
            ids.extend(std::iter::repeat_n(MOE_NO_EXPERT, top_k));
            weights.extend(std::iter::repeat_n(f32::NAN, top_k));
            continue;
        }
        let mut order: Vec<usize> = (0..experts).collect();
        // Stable: equal logits (-0.0 equals 0.0) keep the smaller index first.
        order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).expect("finite"));
        let chosen = &order[..top_k];
        let m = row[chosen[0]] as f64;
        let e: Vec<f64> = chosen.iter().map(|&i| (row[i] as f64 - m).exp()).collect();
        let sum: f64 = e.iter().sum();
        ids.extend(chosen.iter().map(|&i| i as u32));
        weights.extend(e.iter().map(|&x| (x / sum) as f32));
    }
    (ids, weights)
}

/// Expert GEMVs: `y[s] = W[ids[s]] x[s / x_group]` for W `[experts][rows][cols]` and x
/// `[slots / x_group][cols]`; an id outside the experts gives a NaN row. Returns
/// `[slots][rows]`.
pub fn moe_gemv(
    w: &[f32],
    experts: usize,
    rows: usize,
    cols: usize,
    ids: &[u32],
    x: &[f32],
    x_group: usize,
) -> Vec<f32> {
    assert!(w.len() == experts * rows * cols && x_group > 0);
    let mut y = Vec::with_capacity(ids.len() * rows);
    for (s, &e) in ids.iter().enumerate() {
        let e = e as usize;
        if e >= experts {
            y.extend(std::iter::repeat_n(f32::NAN, rows));
            continue;
        }
        let xr = &x[s / x_group * cols..(s / x_group + 1) * cols];
        y.extend(gemv(
            &w[e * rows * cols..(e + 1) * rows * cols],
            rows,
            cols,
            xr,
        ));
    }
    y
}

/// `out[r] = sum_s weights[r][s] y[r][s]` for y `[rows][top_k][hidden]`.
pub fn moe_combine(y: &[f32], weights: &[f32], hidden: usize, top_k: usize) -> Vec<f32> {
    assert!(weights.len().is_multiple_of(top_k) && y.len() == weights.len() * hidden);
    let mut out = Vec::with_capacity(weights.len() / top_k * hidden);
    for (r, wr) in weights.chunks_exact(top_k).enumerate() {
        for i in 0..hidden {
            let acc: f64 = wr
                .iter()
                .enumerate()
                .map(|(s, &w)| w as f64 * y[(r * top_k + s) * hidden + i] as f64)
                .sum();
            out.push(acc as f32);
        }
    }
    out
}
