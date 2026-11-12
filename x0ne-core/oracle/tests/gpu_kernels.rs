//! Every kernel against the CPU reference, on random shapes and seeds and on the shapes at the
//! edges: one value, one key, the tile and chunk boundaries, values that are not numbers.
//!
//! These tests need the GPU and are ignored without it: `scripts/core-gpu.sh` runs them
//! (`--include-ignored`, one thread). The tolerances are x0ne Core's (`refs/thresholds.json`
//! there), restated in `common`: a bf16 result within 2^-7 of the reference, an f32 accumulation
//! within 1e-5 of the sum of its terms' magnitudes, an attention within 2^-7 of the largest V, a
//! rotation within 1e-3 of the largest value. What a kernel copies is compared bit for bit.

mod common;

use common::{
    assert_close, from_half, heads, to_half, Rng, ATTENTION_ATOL_OF_VMAX, BF16_ATOL, BF16_RTOL,
    F32_RTOL, ROPE_ATOL_OF_XMAX,
};
use x0ne_core_oracle::{self as oracle, bf16, Heads};
use x0ne_core_runtime::ops::{CacheAt, Kernels, SplitMerge};
use x0ne_core_runtime::Error;

fn kernels() -> Kernels {
    Kernels::new().expect("a GPU the kernels are built for")
}

fn max_abs(v: &[f32]) -> f64 {
    v.iter()
        .filter(|x| x.is_finite())
        .fold(0f64, |m, &x| m.max(x.abs() as f64))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Values of bf16 with a few that are not ordinary: zeros of both signs, denormals, the
/// largest finite.
fn bf16s_with_edges(rng: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    let edges = [
        0.0,
        -0.0,
        bf16::from_bits(1),
        bf16::from_bits(0x8001),
        bf16::from_bits(0x007F),
        // Rounded as every other value: the kernels are handed bf16, the reference the same.
        bf16::round(scale),
        bf16::round(-scale),
    ];
    (0..n)
        .map(|_| {
            if rng.chance(16) {
                *rng.pick(&edges)
            } else {
                bf16::round(rng.unit() * scale)
            }
        })
        .collect()
}

#[test]
#[ignore = "needs the GPU"]
fn embedding_copies_the_row_bit_for_bit_and_zeroes_a_token_outside() {
    let k = kernels();
    let mut rng = Rng::new(1);
    for hidden in [8, 16, 64, 1024, 4096] {
        for vocab in [1, 2, 7, 300] {
            // Every bit pattern: NaN, infinities and denormals are copied like the rest.
            let raw: Vec<half::bf16> = (0..vocab * hidden)
                .map(|_| half::bf16::from_bits(rng.next_u64() as u16))
                .collect();
            let table = from_half(&raw);
            for token in [
                0,
                vocab - 1,
                rng.below(vocab),
                vocab,
                vocab + 1,
                u32::MAX as usize,
            ] {
                let got = from_half(&k.embedding(&raw, hidden, token as u32).expect("embedding"));
                let want = oracle::embedding(&table, hidden, token as u32);
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "hidden {hidden}, vocab {vocab}, token {token}"
                );
            }
        }
    }
    let table = vec![half::bf16::ZERO; 24];
    for hidden in [0, 4, 12, 16] {
        assert!(
            matches!(k.embedding(&table, hidden, 0), Err(Error::Format(_))),
            "hidden {hidden}"
        );
    }
}

#[test]
#[ignore = "needs the GPU"]
fn add_rmsnorm_is_the_reference_on_random_rows() {
    let k = kernels();
    for seed in 0..60 {
        let mut rng = Rng::new(100 + seed);
        let h = *rng.pick(&[1usize, 2, 7, 8, 64, 255, 256, 257, 1024, 4096, 5120]);
        let scale = *rng.pick(&[1e-3f32, 1.0, 30.0, 1e4]);
        let residual = rng.f32s(h, scale);
        let x = bf16s_with_edges(&mut rng, h, scale);
        let w = rng.bf16s(h, 2.0);
        let eps = *rng.pick(&[1e-6f32, 1e-5]);
        let (r, out) = k
            .add_rmsnorm(&residual, &to_half(&x), &to_half(&w), eps)
            .expect("add_rmsnorm");
        let mut want_r = residual.clone();
        let want = oracle::add_rmsnorm(&mut want_r, &x, &w, eps);
        assert_eq!(
            bits(&r),
            bits(&want_r),
            "seed {seed}: the residual is one f32 addition per value"
        );
        assert_close(
            &format!("add_rmsnorm seed {seed} h {h}"),
            &from_half(&out),
            &want,
            BF16_ATOL,
            BF16_RTOL + F32_RTOL,
        );
    }
    assert!(k.add_rmsnorm(&[], &[], &[], 1e-6).is_err());
    assert!(k
        .add_rmsnorm(
            &[0.0; 8],
            &[half::bf16::ZERO; 7],
            &[half::bf16::ZERO; 8],
            1e-6
        )
        .is_err());
}

/// Rows that are not ordinary: all zero (the epsilon alone under the root), denormals only, a
/// NaN or an infinity (the whole row is then NaN, here as in the reference).
#[test]
#[ignore = "needs the GPU"]
fn add_rmsnorm_on_zero_denormal_and_non_finite_rows() {
    let k = kernels();
    let h = 64;
    let w = vec![1.0f32; h];
    let run = |residual: &[f32], x: &[f32]| {
        let (r, out) = k
            .add_rmsnorm(residual, &to_half(x), &to_half(&w), 1e-6)
            .expect("add_rmsnorm");
        let mut want_r = residual.to_vec();
        let want = oracle::add_rmsnorm(&mut want_r, x, &w, 1e-6);
        // A NaN is a NaN: its payload is the device's own (0x7FFFFFFF on the card measured,
        // 0x7FC00000 on the host), every other value is the same bits.
        let one_nan = |v: &[f32]| -> Vec<u32> {
            v.iter()
                .map(|x| {
                    if x.is_nan() {
                        f32::NAN.to_bits()
                    } else {
                        x.to_bits()
                    }
                })
                .collect()
        };
        assert_eq!(one_nan(&r), one_nan(&want_r));
        (from_half(&out), want)
    };
    let (got, want) = run(&vec![0.0; h], &vec![0.0; h]);
    assert_eq!(got, want, "a zero row stays zero");
    let tiny = vec![bf16::from_bits(1); h];
    let (got, want) = run(&vec![0.0; h], &tiny);
    assert_close("denormal row", &got, &want, 1e-30, BF16_RTOL + F32_RTOL);
    for special in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut x = vec![1.0f32; h];
        x[17] = special;
        let (got, want) = run(&vec![0.5; h], &x);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.is_nan(),
                w.is_nan(),
                "{special}: value {i}: {g} against {w}"
            );
            if !w.is_nan() {
                assert!(
                    (g - w).abs() <= 1e-6,
                    "{special}: value {i}: {g} against {w}"
                );
            }
        }
    }
}

#[test]
#[ignore = "needs the GPU"]
fn qk_norm_rope_is_the_reference_and_writes_one_cache_row() {
    let k = kernels();
    for seed in 0..40 {
        let mut rng = Rng::new(200 + seed);
        let kv = *rng.pick(&[1usize, 2, 4, 8]);
        let h = Heads {
            q_heads: kv * *rng.pick(&[1usize, 2, 4]),
            kv_heads: kv,
            head_dim: 128,
        };
        let (qd, kvd) = (h.q_dim(), h.kv_dim());
        let layers = rng.range(1, 3);
        let max_ctx = *rng.pick(&[1usize, 2, 64, 513]);
        let inside = rng.below(max_ctx);
        let at = CacheAt {
            layers,
            max_ctx,
            layer: rng.below(layers),
            pos: *rng.pick(&[0, max_ctx - 1, inside]),
        };
        let theta = *rng.pick(&[1e4f32, 1e6]);
        let qkv = bf16s_with_edges(&mut rng, qd + 2 * kvd, 8.0);
        let (qn, kn) = (rng.bf16s(128, 2.0), rng.bf16s(128, 2.0));
        let out = k
            .qk_norm_rope_kv_write(
                &to_half(&qkv),
                &to_half(&qn),
                &to_half(&kn),
                heads(h),
                theta,
                1e-6,
                at,
            )
            .expect("qk_norm_rope_kv_write");
        let inv = x0ne_core_runtime::ops::inv_freq(128, theta);
        let (q, kr, v) = oracle::qk_norm_rope(&qkv, &qn, &kn, &inv, h, at.pos as u32, 1e-6);
        let what = format!("seed {seed} {h:?} {at:?}");
        assert!(
            out.rest_untouched,
            "{what}: a cache byte outside the row was written"
        );
        assert_eq!(
            bits(&from_half(&out.v_row)),
            bits(&v),
            "{what}: v is copied"
        );
        for (name, got, want) in [("q", &out.q, &q), ("k", &out.k_row, &kr)] {
            let atol = ROPE_ATOL_OF_XMAX * max_abs(want) + BF16_ATOL;
            assert_close(
                &format!("{what} {name}"),
                &from_half(got),
                want,
                atol,
                BF16_RTOL,
            );
        }
    }
    // Positions far into a long context: the angle is an f32 product, here as on the device.
    let h = Heads {
        q_heads: 2,
        kv_heads: 1,
        head_dim: 128,
    };
    let mut rng = Rng::new(299);
    let qkv = rng.bf16s(h.q_dim() + 2 * h.kv_dim(), 4.0);
    let ones = vec![1.0f32; 128];
    for pos in [4095usize, 32_767, 131_071] {
        let at = CacheAt {
            layers: 1,
            max_ctx: pos + 1,
            layer: 0,
            pos,
        };
        let out = k
            .qk_norm_rope_kv_write(
                &to_half(&qkv),
                &to_half(&ones),
                &to_half(&ones),
                heads(h),
                1e6,
                1e-6,
                at,
            )
            .expect("rope");
        let inv = x0ne_core_runtime::ops::inv_freq(128, 1e6);
        let (q, _, _) = oracle::qk_norm_rope(&qkv, &ones, &ones, &inv, h, pos as u32, 1e-6);
        assert_close(
            &format!("rope at {pos}"),
            &from_half(&out.q),
            &q,
            ROPE_ATOL_OF_XMAX * max_abs(&q),
            BF16_RTOL,
        );
        assert!(out.rest_untouched);
    }
    // Refused before any launch: a position or a layer outside the cache, a head the kernels
    // are not built for.
    let x = to_half(&qkv);
    let n = to_half(&ones);
    let ok = CacheAt {
        layers: 1,
        max_ctx: 4,
        layer: 0,
        pos: 3,
    };
    for bad in [
        CacheAt { pos: 4, ..ok },
        CacheAt { layer: 1, ..ok },
        CacheAt {
            max_ctx: 0,
            pos: 0,
            ..ok
        },
    ] {
        assert!(
            matches!(
                k.qk_norm_rope_kv_write(&x, &n, &n, heads(h), 1e6, 1e-6, bad),
                Err(Error::Format(_))
            ),
            "{bad:?}"
        );
    }
    assert!(k
        .qk_norm_rope_kv_write(&x[1..], &n, &n, heads(h), 1e6, 1e-6, ok)
        .is_err());
    let h64 = x0ne_core_runtime::ops::Heads {
        q_heads: 2,
        kv_heads: 1,
        head_dim: 64,
    };
    assert!(k
        .qk_norm_rope_kv_write(&x, &n, &n, h64, 1e6, 1e-6, ok)
        .is_err());
}

/// `|got - want| <= rtol * magnitude + bf16 rounding`, row by row.
fn assert_gemv(what: &str, got: &[f32], want: &[f32], magnitude: &[f64], rounded: bool) {
    let mut worst = 0f64;
    for (i, ((&g, &w), &m)) in got.iter().zip(want).zip(magnitude).enumerate() {
        let bound = F32_RTOL * m
            + if rounded {
                BF16_RTOL * (w.abs() as f64) + BF16_ATOL
            } else {
                1e-30
            };
        let r = (g as f64 - w as f64).abs() / bound;
        worst = worst.max(r);
        assert!(r <= 1.0, "{what}: row {i}: got {g} want {w} (ratio {r:.3})");
    }
    eprintln!("MEASURE {what}: worst ratio {worst:.4}");
}

#[test]
#[ignore = "needs the GPU"]
fn gemv_is_the_reference_on_random_shapes() {
    let k = kernels();
    for seed in 0..50 {
        let mut rng = Rng::new(300 + seed);
        let rows = *rng.pick(&[1usize, 2, 3, 8, 127, 1024, 2049]);
        let cols = *rng.pick(&[1usize, 2, 8, 9, 64, 1000, 1024, 4096]);
        let scale = *rng.pick(&[1e-2f32, 1.0, 50.0]);
        let (w, x) = (
            bf16s_with_edges(&mut rng, rows * cols, scale),
            bf16s_with_edges(&mut rng, cols, scale),
        );
        let want = oracle::gemv(&w, rows, cols, &x);
        let magnitude = oracle::gemv_magnitude(&w, rows, cols, &x);
        let what = format!("gemv seed {seed} {rows}x{cols}");
        let f = k
            .gemv_f32_out(&to_half(&w), rows, cols, &to_half(&x))
            .expect("gemv f32");
        assert_gemv(&format!("{what} f32"), &f, &want, &magnitude, false);
        let b = k
            .gemv_bf16_out(&to_half(&w), rows, cols, &to_half(&x))
            .expect("gemv bf16");
        assert_gemv(
            &format!("{what} bf16"),
            &from_half(&b),
            &want,
            &magnitude,
            true,
        );
        // The bf16 output is the f32 output, rounded: one accumulation, two stores.
        eprintln!(
            "MEASURE {what}: bf16 out is the rounded f32 out: {}",
            bits(&from_half(&b)) == bits(&bf16::round_all(&f))
        );
    }
    let w = vec![half::bf16::ONE; 16];
    for (rows, cols, x) in [(0, 8, 8), (2, 0, 0), (2, 8, 7), (3, 8, 8), (2, 8, 9)] {
        assert!(
            matches!(
                k.gemv_f32_out(&w, rows, cols, &vec![half::bf16::ONE; x]),
                Err(Error::Format(_))
            ),
            "{rows}x{cols}, x {x}"
        );
    }
}

/// A NaN or an infinity in x reaches every row it has a weight in, and no other.
#[test]
#[ignore = "needs the GPU"]
fn gemv_carries_non_finite_values_to_the_rows_that_read_them() {
    let k = kernels();
    let (rows, cols) = (16, 64);
    let mut rng = Rng::new(399);
    let mut w = rng.bf16s(rows * cols, 1.0);
    // Rows 0..8 have a zero weight in column 5: they do not read it.
    for r in 0..8 {
        w[r * cols + 5] = 0.0;
    }
    let mut x = rng.bf16s(cols, 1.0);
    x[5] = f32::INFINITY;
    let got = k
        .gemv_f32_out(&to_half(&w), rows, cols, &to_half(&x))
        .expect("gemv");
    let want = oracle::gemv(&w, rows, cols, &x);
    for (r, (g, w)) in got.iter().zip(&want).enumerate() {
        // 0 x inf is NaN in IEEE: both sides agree on which rows are finite.
        assert_eq!(g.is_finite(), w.is_finite(), "row {r}: {g} against {w}");
        assert_eq!(g.is_nan(), w.is_nan(), "row {r}: {g} against {w}");
    }
}

fn attention_inputs(
    rng: &mut Rng,
    ctx: usize,
    h: Heads,
    scale: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        rng.bf16s(ctx * h.q_dim(), scale),
        rng.bf16s(ctx * h.kv_dim(), scale),
        bf16s_with_edges(rng, ctx * h.kv_dim(), 4.0),
    )
}

const GEOMETRIES: [(usize, usize); 6] = [(1, 1), (2, 1), (4, 2), (16, 8), (32, 8), (40, 8)];

/// The decode attention, per-head and split-K with each merge, against the reference, at the
/// context lengths around the split tile (128 keys) and the prefill chunk (512).
#[test]
#[ignore = "needs the GPU"]
fn decode_attention_is_the_reference_around_every_boundary() {
    let k = kernels();
    let mut rng = Rng::new(400);
    for (qh, kvh) in GEOMETRIES {
        let h = Heads {
            q_heads: qh,
            kv_heads: kvh,
            head_dim: 128,
        };
        let plan = k.split_plan(heads(h)).expect("split plan");
        eprintln!(
            "MEASURE {qh}/{kvh} heads: P {} grouped {}",
            plan.splits, plan.grouped
        );
        for ctx in [
            1usize, 2, 3, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1000, 2048, 4097,
        ] {
            // Sharp and flat distributions: scores of a few units, and of a few hundredths.
            for scale in [0.5f32, 0.02] {
                let (q, kk, v) = attention_inputs(&mut rng, ctx, h, scale * 128f32.sqrt());
                let q = &q[..h.q_dim()];
                let want = oracle::attention_decode(q, &kk, &v, ctx, h);
                let atol = ATTENTION_ATOL_OF_VMAX * max_abs(&v);
                let what = format!("attention {qh}/{kvh} ctx {ctx} scale {scale}");
                let (qb, kb, vb) = (to_half(q), to_half(&kk), to_half(&v));
                let per_head = from_half(
                    &k.attention_decode(&qb, &kb, &vb, ctx, heads(h))
                        .expect("per head"),
                );
                assert_close(&format!("{what} per-head"), &per_head, &want, atol, 0.0);
                let split = from_half(
                    &k.attention_decode_split(&qb, &kb, &vb, ctx, heads(h))
                        .expect("split"),
                );
                assert_close(&format!("{what} split-K"), &split, &want, atol, 0.0);
                if plan.grouped {
                    let last = k
                        .attention_decode_split_merge(
                            &qb,
                            &kb,
                            &vb,
                            ctx,
                            heads(h),
                            SplitMerge::LastBlock,
                        )
                        .expect("last-block merge");
                    assert_eq!(
                        bits(&from_half(&last)),
                        bits(&split),
                        "{what}: the two merges give the same bits"
                    );
                } else {
                    assert!(k
                        .attention_decode_split_merge(
                            &qb,
                            &kb,
                            &vb,
                            ctx,
                            heads(h),
                            SplitMerge::LastBlock
                        )
                        .is_err());
                }
                // Twice the same launch: the same bits.
                let again = from_half(
                    &k.attention_decode_split(&qb, &kb, &vb, ctx, heads(h))
                        .expect("split"),
                );
                assert_eq!(
                    bits(&again),
                    bits(&split),
                    "{what}: split-K is not deterministic"
                );
            }
        }
    }
}

/// One key: the output is its V row, exactly. One key far above the others: its V row within
/// the rounding. Equal keys: the mean of the V rows.
#[test]
#[ignore = "needs the GPU"]
fn decode_attention_on_one_key_a_dominant_key_and_equal_keys() {
    let k = kernels();
    let h = Heads {
        q_heads: 4,
        kv_heads: 2,
        head_dim: 128,
    };
    let (qd, kvd) = (h.q_dim(), h.kv_dim());
    let mut rng = Rng::new(450);
    let q = rng.bf16s(qd, 4.0);
    let v = rng.bf16s(300 * kvd, 4.0);
    let kk = rng.bf16s(300 * kvd, 4.0);
    for split in [false, true] {
        let run = |kk: &[f32], v: &[f32], ctx: usize| {
            let (qb, kb, vb) = (to_half(&q), to_half(kk), to_half(v));
            let out = if split {
                k.attention_decode_split(&qb, &kb, &vb, ctx, heads(h))
            } else {
                k.attention_decode(&qb, &kb, &vb, ctx, heads(h))
            };
            from_half(&out.expect("attention"))
        };
        let one = run(&kk, &v, 1);
        for head in 0..h.q_heads {
            let kh = head / 2;
            assert_eq!(
                bits(&one[head * 128..(head + 1) * 128]),
                bits(&v[kh * 128..(kh + 1) * 128]),
                "split {split}: one key"
            );
        }
        // Key 137 is q itself, scaled up: its score is far above the others'.
        let mut dominant = kk.clone();
        for kh in 0..h.kv_heads {
            for i in 0..128 {
                dominant[137 * kvd + kh * 128 + i] = bf16::round(q[kh * 2 * 128 + i] * 16.0);
            }
        }
        let got = run(&dominant, &v, 300);
        let want = oracle::attention_decode(&q, &dominant, &v, 300, h);
        assert_close(
            &format!("dominant key, split {split}"),
            &got,
            &want,
            ATTENTION_ATOL_OF_VMAX * 4.0,
            0.0,
        );
        // Every key the same: uniform weights.
        let same: Vec<f32> = (0..300).flat_map(|_| kk[..kvd].to_vec()).collect();
        let got = run(&same, &v, 300);
        let want = oracle::attention_decode(&q, &same, &v, 300, h);
        assert_close(
            &format!("equal keys, split {split}"),
            &got,
            &want,
            ATTENTION_ATOL_OF_VMAX * 4.0,
            0.0,
        );
    }
    let (qb, kb, vb) = (to_half(&q), to_half(&kk), to_half(&v));
    assert!(
        matches!(
            k.attention_decode(&qb, &kb, &vb, 0, heads(h)),
            Err(Error::Format(_))
        ),
        "no key"
    );
    assert!(
        k.attention_decode(&qb, &kb, &vb, 301, heads(h)).is_err(),
        "more keys than the cache holds"
    );
    assert!(k.attention_decode(&qb[1..], &kb, &vb, 4, heads(h)).is_err());
    assert!(k
        .attention_decode_split(&qb, &kb, &vb, 0, heads(h))
        .is_err());
    assert!(k
        .attention_decode_split(&qb, &kb, &vb, 301, heads(h))
        .is_err());
}

/// The causal prefill attention against the reference, whole and from a chunk boundary, and
/// against the decode kernel on the same rows.
#[test]
#[ignore = "needs the GPU"]
fn prefill_attention_is_the_reference_whole_and_from_a_chunk_boundary() {
    let k = kernels();
    let mut rng = Rng::new(500);
    for (qh, kvh) in [(2, 1), (4, 2), (16, 8)] {
        let h = Heads {
            q_heads: qh,
            kv_heads: kvh,
            head_dim: 128,
        };
        let qd = h.q_dim();
        for t in [1usize, 2, 127, 128, 129, 511, 512, 513, 1025] {
            let (q, kk, v) = attention_inputs(&mut rng, t, h, 0.3 * 128f32.sqrt());
            let (qb, kb, vb) = (to_half(&q), to_half(&kk), to_half(&v));
            let atol = ATTENTION_ATOL_OF_VMAX * max_abs(&v);
            let mut rows: Vec<usize> = vec![0, t - 1, t / 2, rng.below(t), rng.below(t)];
            rows.dedup();
            let want = oracle::attention_rows(&q, &kk, &v, 0, h, &rows);
            let got = k
                .attention_prefill_rows(&qb, &kb, &vb, t, heads(h), &rows)
                .expect("prefill");
            assert_eq!(got.len(), rows.len());
            for ((r, g), w) in rows.iter().zip(&got).zip(&want) {
                assert_close(
                    &format!("prefill {qh}/{kvh} t {t} row {r}"),
                    &from_half(g),
                    w,
                    atol,
                    0.0,
                );
                let decode = k
                    .attention_decode(&qb[r * qd..(r + 1) * qd], &kb, &vb, r + 1, heads(h))
                    .expect("decode");
                assert_close(
                    &format!("prefill against decode, t {t} row {r}"),
                    &from_half(g),
                    &from_half(&decode),
                    2.0 * atol,
                    0.0,
                );
            }
            // The same rows computed by a launch that starts at pos0: a later chunk.
            for pos0 in [0, 1, t / 2, t - 1].into_iter().filter(|&p| p < t) {
                let tail: Vec<usize> = (0..t - pos0).step_by(((t - pos0) / 5).max(1)).collect();
                let got = k
                    .attention_prefill_at(&qb[pos0 * qd..], &kb, &vb, t, pos0, heads(h), &tail)
                    .expect("prefill at");
                let want = oracle::attention_rows(&q[pos0 * qd..], &kk, &v, pos0, h, &tail);
                for ((r, g), w) in tail.iter().zip(&got).zip(&want) {
                    assert_close(
                        &format!("prefill at {pos0}, t {t} row {r}"),
                        &from_half(g),
                        w,
                        atol,
                        0.0,
                    );
                }
            }
        }
    }
    let h = Heads {
        q_heads: 2,
        kv_heads: 1,
        head_dim: 128,
    };
    let z = vec![half::bf16::ZERO; 4 * 256];
    assert!(
        matches!(
            k.attention_prefill_rows(&z, &z[..4 * 128], &z[..4 * 128], 4, heads(h), &[4]),
            Err(Error::Format(_))
        ),
        "a row past t"
    );
    assert!(k
        .attention_prefill_rows(&z, &z[..4 * 128], &z[..4 * 128], 0, heads(h), &[])
        .is_err());
    assert!(k
        .attention_prefill_rows(&z[..256], &z[..4 * 128], &z[..4 * 128], 4, heads(h), &[0])
        .is_err());
    assert!(
        k.attention_prefill_at(
            &z[..256],
            &z[..4 * 128],
            &z[..4 * 128],
            4,
            4,
            heads(h),
            &[0]
        )
        .is_err(),
        "pos0 = t"
    );
}

#[test]
#[ignore = "needs the GPU"]
fn swiglu_is_the_reference_non_finite_values_included() {
    let k = kernels();
    for seed in 0..40 {
        let mut rng = Rng::new(600 + seed);
        let inter = *rng.pick(&[1usize, 2, 7, 8, 255, 256, 3072, 12288]);
        let scale = *rng.pick(&[1.0f32, 8.0, 40.0, 200.0]);
        let gate_up = bf16s_with_edges(&mut rng, 2 * inter, scale);
        let got = from_half(&k.swiglu(&to_half(&gate_up)).expect("swiglu"));
        assert_close(
            &format!("swiglu seed {seed} inter {inter}"),
            &got,
            &oracle::swiglu(&gate_up),
            BF16_ATOL,
            BF16_RTOL,
        );
        let rows = *rng.pick(&[1usize, 2, 5]);
        let many = rng.bf16s(rows * 2 * inter.min(256), 8.0);
        let got = from_half(&k.swiglu_rows(&to_half(&many), rows).expect("swiglu rows"));
        let want: Vec<f32> = many
            .chunks_exact(many.len() / rows)
            .flat_map(oracle::swiglu)
            .collect();
        assert_close(
            &format!("swiglu rows seed {seed}"),
            &got,
            &want,
            BF16_ATOL,
            BF16_RTOL,
        );
    }
    // gate: every special against up = 2; then up: every special against gate = 1.
    let s = [
        0.0f32,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        200.0,
        -200.0,
        bf16::from_bits(1),
        bf16::from_bits(0x7F7F),
        bf16::from_bits(0xFF7F),
    ];
    for (gate, up) in [
        (s.to_vec(), vec![2.0; s.len()]),
        (vec![1.0; s.len()], s.to_vec()),
    ] {
        let gate_up: Vec<f32> = gate.iter().chain(&up).copied().collect();
        let got = from_half(&k.swiglu(&to_half(&gate_up)).expect("swiglu"));
        let want = oracle::swiglu(&gate_up);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let w = bf16::round(*w);
            assert_eq!(
                g.is_nan(),
                w.is_nan(),
                "gate {} up {}: {g} against {w}",
                gate[i],
                up[i]
            );
            if !w.is_nan() {
                assert!(
                    *g == w || (g - w).abs() <= BF16_RTOL as f32 * w.abs() + 1e-30,
                    "gate {} up {}: {g} against {w}",
                    gate[i],
                    up[i]
                );
            }
        }
    }
    assert!(k.swiglu(&[]).is_err() && k.swiglu(&[half::bf16::ONE; 3]).is_err());
    assert!(
        k.swiglu_rows(&[half::bf16::ONE; 8], 0).is_err()
            && k.swiglu_rows(&[half::bf16::ONE; 8], 3).is_err()
    );
}

/// The device's argmax is the first largest value, for every length around its block size and
/// with ties, zeros of both signs and infinities; a row with a NaN has none.
#[test]
#[ignore = "needs the GPU"]
fn argmax_is_the_first_largest_and_a_nan_row_has_none() {
    let k = kernels();
    let mut rng = Rng::new(700);
    for round in 0..300 {
        let n = *rng.pick(&[
            1usize, 2, 3, 31, 32, 33, 1023, 1024, 1025, 2047, 4096, 151_936,
        ]);
        let mut row: Vec<f32> = if rng.chance(2) {
            (0..n).map(|_| rng.below(4) as f32 - 2.0).collect()
        } else {
            rng.f32s(n, 50.0)
        };
        for _ in 0..rng.below(4) {
            let at = rng.below(n);
            row[at] = *rng.pick(&[
                0.0,
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::MAX,
                f32::MIN,
            ]);
        }
        let want = oracle::argmax(&row).expect("no NaN");
        assert_eq!(
            k.argmax(&row).expect("argmax"),
            want,
            "round {round}: {n} values"
        );
        // The largest value moved to the two ends.
        for at in [0, n - 1] {
            let mut moved = row.clone();
            moved[at] = f32::INFINITY;
            assert_eq!(
                k.argmax(&moved).expect("argmax"),
                oracle::argmax(&moved).unwrap(),
                "round {round}"
            );
        }
        let mut nan = row.clone();
        nan[rng.below(n)] = f32::NAN;
        assert!(
            matches!(k.argmax(&nan), Err(Error::Gpu(_))),
            "round {round}: a NaN row has an argmax"
        );
    }
    assert_eq!(k.argmax(&[f32::NEG_INFINITY; 5]).expect("all -inf"), 0);
    assert_eq!(k.argmax(&[-0.0, 0.0]).expect("zeros"), 0);
    assert!(matches!(k.argmax(&[]), Err(Error::Format(_))));
}

#[test]
#[ignore = "needs the GPU"]
fn advance_adds_one() {
    let k = kernels();
    for v in [0u32, 1, 511, 512, 131_071, u32::MAX - 1] {
        assert_eq!(k.advance(v).expect("advance"), v + 1);
    }
    eprintln!("MEASURE advance(u32::MAX) = {:?}", k.advance(u32::MAX));
}

#[test]
#[ignore = "needs the GPU"]
fn moe_kernels_are_the_reference() {
    let k = kernels();
    for seed in 0..40 {
        let mut rng = Rng::new(800 + seed);
        let experts = *rng.pick(&[1usize, 2, 8, 128, 1024]);
        let top_k = rng.range(1, experts.min(32));
        let rows = rng.range(1, 9);
        let mut logits: Vec<f32> = if rng.chance(3) {
            (0..rows * experts).map(|_| rng.below(3) as f32).collect()
        } else {
            rng.f32s(rows * experts, 6.0)
        };
        if rng.chance(3) {
            logits[rng.below(rows * experts)] =
                *rng.pick(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
        }
        let (ids, w) = k.moe_route(&logits, experts, top_k).expect("route");
        let (want_ids, want_w) = oracle::moe_route(&logits, experts, top_k);
        assert_eq!(ids, want_ids, "seed {seed}: {experts} experts, top {top_k}");
        assert_close(&format!("moe weights seed {seed}"), &w, &want_w, 1e-6, 1e-5);

        // The expert GEMVs on the routing above, undefined rows included.
        let (r, cols) = (*rng.pick(&[1usize, 8, 24]), 8 * rng.range(1, 8));
        let small = experts.min(8);
        let ids: Vec<u32> = ids
            .iter()
            .map(|&i| {
                if i == oracle::MOE_NO_EXPERT || (i as usize) < small {
                    i
                } else {
                    i % small as u32
                }
            })
            .collect();
        let wt = rng.bf16s(small * r * cols, 1.0);
        let x = rng.bf16s(rows * cols, 2.0);
        let want = oracle::moe_gemv(&wt, small, r, cols, &ids, &x, top_k);
        for f32_out in [true, false] {
            let got = k
                .moe_gemv(
                    &to_half(&wt),
                    small,
                    r,
                    cols,
                    &ids,
                    &to_half(&x),
                    top_k,
                    f32_out,
                )
                .expect("moe gemv");
            assert_close(
                &format!("moe gemv seed {seed} f32 {f32_out}"),
                &got,
                &want,
                1e-4,
                if f32_out { 1e-4 } else { BF16_RTOL },
            );
        }
        let hidden = 8 * rng.range(1, 16);
        let y = rng.f32s(rows * top_k * hidden, 3.0);
        let wts = rng.f32s(rows * top_k, 1.0);
        let got = from_half(&k.moe_combine(&y, &wts, hidden, top_k).expect("combine"));
        assert_close(
            &format!("moe combine seed {seed}"),
            &got,
            &oracle::moe_combine(&y, &wts, hidden, top_k),
            1e-4,
            BF16_RTOL,
        );
    }
    for (experts, top_k, n) in [
        (0, 1, 8),
        (1025, 1, 1025),
        (8, 0, 8),
        (8, 9, 8),
        (64, 33, 64),
        (8, 2, 7),
        (8, 2, 0),
    ] {
        assert!(
            matches!(
                k.moe_route(&vec![0.0; n], experts, top_k),
                Err(Error::Format(_))
            ),
            "{experts} experts, top {top_k}, {n} logits"
        );
    }
}
