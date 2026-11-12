//! The reference against itself and against the host side of the runtime: bf16 by hand against
//! the `half` crate over every f32, the runtime's host metrics and constants against the
//! reference, properties any correct forward has (causality, a cache equal to a recompute), the
//! launch argument check, and a model of the greedy loop's ring. No GPU.

mod common;

use common::{model_tensors, reference, tiny, Rng};
use x0ne_core_oracle::model::Cache;
use x0ne_core_oracle::{self as oracle, bf16, Heads};
use x0ne_core_runtime::gpu::{check_args, Arg, Param};
use x0ne_core_runtime::{engine, metrics, ops, Error};

/// Rounding to bf16 by hand is `half`'s, for every one of the 2^32 f32 bit patterns: the
/// packing of every value the runtime sends to the device or reads from it.
#[test]
fn bf16_rounding_is_halfs_for_every_f32() {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()) as u64;
    let per = (1u64 << 32).div_ceil(threads);
    std::thread::scope(|s| {
        for t in 0..threads {
            s.spawn(move || {
                for bits in t * per..((t + 1) * per).min(1 << 32) {
                    let x = f32::from_bits(bits as u32);
                    let (ours, theirs) = (bf16::to_bits(x), half::bf16::from_f32(x).to_bits());
                    assert_eq!(ours, theirs, "f32 {bits:#010x}");
                }
            });
        }
    });
}

/// Unpacking is exact, packing a bf16 value gives it back, and rounding is to the nearest with
/// ties to even: the properties, apart from the comparison with `half`.
#[test]
fn bf16_packing_round_trips_and_rounds_to_nearest_even() {
    for bits in 0..=u16::MAX {
        let x = bf16::from_bits(bits);
        assert_eq!(x.to_bits(), (bits as u32) << 16);
        if x.is_nan() {
            assert!(bf16::from_bits(bf16::to_bits(x)).is_nan());
            continue;
        }
        assert_eq!(
            bf16::to_bits(x),
            bits,
            "a bf16 value does not pack to itself"
        );
        assert_eq!(
            bf16::from_bytes(&bf16::bytes(&[x]))[0].to_bits(),
            x.to_bits()
        );
        if !x.is_finite() || bits & 0x7FFF == 0x7F7F {
            continue;
        }
        // Between x and the next value away from zero: below the middle, the middle, above.
        let (lo, hi) = (x.to_bits(), bf16::from_bits(bits + 1).to_bits());
        let mid = lo + 0x8000;
        assert_eq!(bf16::to_bits(f32::from_bits(mid - 1)), bits);
        assert_eq!(bf16::to_bits(f32::from_bits(mid + 1)), bits + 1);
        let even = if bits & 1 == 0 { bits } else { bits + 1 };
        assert_eq!(
            bf16::to_bits(f32::from_bits(mid)),
            even,
            "tie at {bits:#06x}"
        );
        assert_eq!(hi - lo, 0x1_0000);
        assert_eq!(bf16::ulp(x), (f32::from_bits(hi) as f64 - x as f64).abs());
    }
    // Past the largest bf16: infinity, with the sign.
    assert_eq!(bf16::round(f32::MAX), f32::INFINITY);
    assert_eq!(bf16::round(f32::MIN), f32::NEG_INFINITY);
    assert_eq!(bf16::round(-0.0).to_bits(), (-0.0f32).to_bits());
    // The smallest f32 denormals round to zero or to the smallest bf16 denormal.
    assert_eq!(bf16::round(f32::from_bits(1)), 0.0);
    assert_eq!(bf16::round(f32::from_bits(0x8001)).to_bits(), 0x1_0000);
}

fn rows(rng: &mut Rng) -> Vec<Vec<f32>> {
    let specials = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
        1e-45,
    ];
    let mut out = vec![
        vec![],
        vec![0.0],
        vec![-0.0, 0.0],
        vec![0.0, -0.0],
        vec![f32::NAN],
        vec![1.0; 7],
    ];
    for _ in 0..4000 {
        let n = rng.range(1, 40);
        let mut row = if rng.chance(3) {
            // Few distinct values: ties.
            (0..n).map(|_| rng.below(3) as f32 - 1.0).collect()
        } else {
            rng.f32s(n, 30.0)
        };
        if rng.chance(4) {
            for _ in 0..rng.range(1, 3) {
                let at = rng.below(n);
                row[at] = *rng.pick(&specials);
            }
        }
        out.push(row);
    }
    out
}

/// The host's argmax, log-sum-exp and KL against the reference's, ties, zeros of both signs,
/// infinities and NaN included.
#[test]
fn host_metrics_are_the_references() {
    let mut rng = Rng::new(3);
    for row in rows(&mut rng) {
        assert_eq!(metrics::argmax(&row), oracle::argmax(&row), "{row:?}");
        if let Some(i) = oracle::argmax(&row) {
            let i = i as usize;
            assert!(row.iter().all(|&v| v <= row[i]), "not a maximum: {row:?}");
            assert!(
                row[..i].iter().all(|&v| v < row[i]),
                "not the first maximum: {row:?}"
            );
        } else {
            assert!(row.is_empty() || row.iter().any(|v| v.is_nan()));
        }
        if !row.is_empty() && row.iter().all(|v| v.is_finite()) {
            let (got, want) = (metrics::logsumexp(&row), oracle::logsumexp(&row));
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "{row:?}: {got} against {want}"
            );
            let other: Vec<f32> = row.iter().map(|&v| v * 0.5 + rng.unit()).collect();
            let (got, want) = (metrics::kl(&row, &other), oracle::kl(&row, &other));
            assert!((got - want).abs() <= 1e-9, "KL {got} against {want}");
            assert!(want >= -1e-12, "a negative KL");
            assert!(oracle::kl(&row, &row).abs() < 1e-12);
        }
    }
}

/// The constants the host gives the device, against their definitions in f64.
#[test]
fn host_constants_are_their_definitions() {
    for head_dim in [2, 64, 128, 256] {
        for theta in [1e4f32, 1e6, 5e5] {
            let (got, want) = (
                ops::inv_freq(head_dim, theta),
                oracle::inv_freq(head_dim, theta as f64),
            );
            assert_eq!(got.len(), head_dim / 2);
            assert_eq!(got[0], 1.0);
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g as f64 - w).abs() <= 16.0 * f32::EPSILON as f64 * w,
                    "inv_freq[{i}] of {head_dim}: {g} against {w}"
                );
            }
            assert!(
                got.windows(2).all(|w| w[1] < w[0]),
                "the frequencies decrease"
            );
        }
        let s = ops::attn_scale(head_dim) as f64;
        assert!((s * s * head_dim as f64 - 1.0).abs() < 1e-6);
    }
    assert_eq!(ops::HEAD_DIM, 128);
    assert_eq!(ops::SPLIT_STATE, ops::HEAD_DIM + 4);
    assert_eq!(ops::MOE_NO_EXPERT, oracle::MOE_NO_EXPERT);
    for (q, p) in [(1, 1), (16, 1), (32, 7), (64, 64)] {
        assert_eq!(
            ops::split_state_bytes(q, p),
            (q * p as usize * 132 * 4) as u64
        );
    }
    assert_eq!(engine::GREEDY_AHEAD_MAX, 15);
    assert!((1..=engine::GREEDY_AHEAD_MAX).contains(&engine::GREEDY_AHEAD));
    let h = ops::Heads {
        q_heads: 4,
        kv_heads: 2,
        head_dim: 128,
    };
    assert!(h.check().is_ok());
    for bad in [
        ops::Heads { head_dim: 64, ..h },
        ops::Heads { kv_heads: 0, ..h },
        ops::Heads { kv_heads: 3, ..h },
        ops::Heads {
            q_heads: 2,
            kv_heads: 4,
            ..h
        },
    ] {
        assert!(matches!(bad.check(), Err(Error::Format(_))), "{bad:?}");
    }
}

/// The byte views the runtime uploads and reads back with (`as_bytes`, `as_bytes_mut`: two of
/// its `unsafe` blocks): the little-endian bytes of the values, for each type they are sealed
/// to, empty slices included.
#[test]
fn byte_views_are_the_little_endian_bytes_of_their_values() {
    let mut rng = Rng::new(5);
    for n in [0, 1, 2, 3, 17, 1000] {
        let f: Vec<f32> = (0..n)
            .map(|_| f32::from_bits(rng.next_u64() as u32))
            .collect();
        let want: Vec<u8> = f.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        assert_eq!(ops::as_bytes(&f), &want[..]);
        let u: Vec<u32> = (0..n).map(|_| rng.next_u64() as u32).collect();
        let want: Vec<u8> = u.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(ops::as_bytes(&u), &want[..]);
        let h: Vec<half::bf16> = (0..n)
            .map(|_| half::bf16::from_bits(rng.next_u64() as u16))
            .collect();
        let want: Vec<u8> = h.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        assert_eq!(ops::as_bytes(&h), &want[..]);
        let b = rng.bytes(n);
        assert_eq!(ops::as_bytes(&b), &b[..]);

        // Written through the mutable view, every bit pattern is a value of the type.
        let bytes = rng.bytes(n * 4);
        let mut back = vec![0f32; n];
        ops::as_bytes_mut(&mut back).copy_from_slice(&bytes);
        for (v, &c) in back.iter().zip(bytes.as_chunks::<4>().0) {
            assert_eq!(v.to_bits(), u32::from_le_bytes(c));
        }
        let mut back = vec![half::bf16::ZERO; n];
        ops::as_bytes_mut(&mut back).copy_from_slice(&bytes[..n * 2]);
        for (v, &c) in back.iter().zip(bytes.as_chunks::<2>().0) {
            assert_eq!(v.to_bits(), u16::from_le_bytes(c));
        }
        assert_eq!(ops::as_bytes_mut(&mut back).len(), n * 2);
    }
}

/// A launch's arguments against the kernel's declared parameters (`Gpu::launch` calls this
/// before it hands the driver a pointer to them): accepted exactly when each has its
/// parameter's C type, for every kernel of the set.
#[test]
fn launch_arguments_are_accepted_only_with_the_kernels_types() {
    let arg = |p: Param| match p {
        Param::Ptr => Arg::Ptr(0x1000),
        Param::U32 => Arg::U32(7),
        Param::F32 => Arg::F32(0.5),
    };
    let other = |p: Param| match p {
        Param::Ptr => [Arg::U32(7), Arg::F32(0.5)],
        Param::U32 => [Arg::Ptr(0x1000), Arg::F32(0.5)],
        Param::F32 => [Arg::Ptr(0x1000), Arg::U32(7)],
    };
    let table = x0ne_core_kernels::KERNEL_PARAMS;
    assert_eq!(table.len(), x0ne_core_kernels::KERNEL_NAMES.len());
    for (name, params) in table {
        assert!(x0ne_core_kernels::KERNEL_NAMES.contains(name));
        assert_eq!(x0ne_core_kernels::params_of(name), Some(*params));
        assert!(
            params.len() <= 12,
            "{name} takes {} parameters",
            params.len()
        );
        let good: Vec<Arg> = params.iter().map(|&p| arg(p)).collect();
        check_args(name, params, &good).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            matches!(
                check_args(name, params, &good[..good.len() - 1]),
                Err(Error::Gpu(_))
            ),
            "{name}: one fewer"
        );
        let mut more = good.clone();
        more.push(Arg::U32(0));
        assert!(check_args(name, params, &more).is_err(), "{name}: one more");
        assert!(check_args(name, params, &[]).is_err() || params.is_empty());
        for i in 0..params.len() {
            for wrong in other(params[i]) {
                let mut bad = good.clone();
                bad[i] = wrong;
                let e = check_args(name, params, &bad).expect_err("a wrong type");
                assert!(
                    e.to_string().contains(&format!("argument {i}")),
                    "{name}: {e}"
                );
            }
        }
    }
    assert_eq!(x0ne_core_kernels::params_of("x0ne_no_such_kernel"), None);
    assert_eq!(
        (Param::Ptr.size(), Param::U32.size(), Param::F32.size()),
        (8, 4, 4)
    );
    assert_eq!(
        (Arg::from(1u64), Arg::from(1u32), Arg::from(1f32)),
        (Arg::Ptr(1), Arg::U32(1), Arg::F32(1.0))
    );
}

fn heads() -> Heads {
    Heads {
        q_heads: 4,
        kv_heads: 2,
        head_dim: 128,
    }
}

/// Properties of the reference's kernels that hold whatever the values.
#[test]
fn the_reference_kernels_have_the_properties_of_their_definitions() {
    let mut rng = Rng::new(7);
    let h = heads();
    let (qd, kvd, hd) = (h.q_dim(), h.kv_dim(), h.head_dim);
    for round in 0..40 {
        // RMSNorm: the residual is the f32 sum; the output's mean square is that of the weights
        // when they are all w; scaling the input scales nothing.
        let n = 8 * rng.range(1, 32);
        let (r0, x) = (rng.f32s(n, 3.0), rng.bf16s(n, 3.0));
        let mut r = r0.clone();
        let out = oracle::add_rmsnorm(&mut r, &x, &vec![2.0; n], 1e-6);
        assert!(r
            .iter()
            .zip(r0.iter().zip(&x))
            .all(|(&s, (&a, &b))| s == a + b));
        let ms: f64 = out.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / n as f64;
        assert!((ms - 4.0).abs() < 1e-3, "round {round}: mean square {ms}");
        let mut scaled: Vec<f32> = r.iter().map(|&v| v * 4.0).collect();
        let out4 = oracle::add_rmsnorm(&mut scaled, &vec![0.0; n], &vec![2.0; n], 1e-6);
        assert!(out
            .iter()
            .zip(&out4)
            .all(|(&a, &b)| (a - b).abs() <= 1e-5 * a.abs().max(1e-3)));

        // RoPE: position 0 is the norm alone; a rotation keeps each pair's length; v is copied;
        // the score of q at a against k at b depends on a - b only.
        let qkv = rng.bf16s(qd + 2 * kvd, 2.0);
        let (qn, kn) = (rng.bf16s(hd, 1.0), rng.bf16s(hd, 1.0));
        let inv = ops::inv_freq(hd, 1e6);
        let at = |pos: u32| oracle::qk_norm_rope(&qkv, &qn, &kn, &inv, h, pos, 1e-6);
        let (q0, k0, v0) = at(0);
        assert_eq!(v0, qkv[qd + kvd..]);
        let pos = rng.below(100_000) as u32;
        let (q, k, v) = at(pos);
        assert_eq!(v, v0);
        for (x0, x) in [(&q0, &q), (&k0, &k)] {
            for head in 0..x.len() / hd {
                for i in 0..hd / 2 {
                    let (a, b) = (head * hd + i, head * hd + i + hd / 2);
                    let (l0, l) = (x0[a].hypot(x0[b]), x[a].hypot(x[b]));
                    assert!(
                        (l0 - l).abs() <= 1e-5 * l0.max(1e-3),
                        "round {round}: pair {i} changed length"
                    );
                }
            }
        }
        let score = |a: u32, b: u32| -> f64 {
            let (q, _, _) = at(a);
            let (_, k, _) = at(b);
            q[..hd]
                .iter()
                .zip(&k[..hd])
                .map(|(&x, &y)| x as f64 * y as f64)
                .sum()
        };
        let (a, b, shift) = (
            rng.below(500) as u32,
            rng.below(500) as u32,
            rng.below(500) as u32,
        );
        let (s0, s1) = (score(a, b), score(a + shift, b + shift));
        assert!(
            (s0 - s1).abs() < 2e-2,
            "round {round}: score {s0} at ({a}, {b}), {s1} shifted by {shift}"
        );

        // Attention: one key gives its V row; every output lies between the smallest and the
        // largest V of its kv head; the causal rows are the decode rows.
        let ctx = rng.range(1, 40);
        let (q, k, v) = (
            rng.bf16s(ctx * qd, 4.0),
            rng.bf16s(ctx * kvd, 4.0),
            rng.bf16s(ctx * kvd, 4.0),
        );
        let one = oracle::attention_decode(&q[..qd], &k, &v, 1, h);
        for head in 0..h.q_heads {
            let kh = head / (h.q_heads / h.kv_heads);
            assert_eq!(one[head * hd..(head + 1) * hd], v[kh * hd..(kh + 1) * hd]);
        }
        let all: Vec<usize> = (0..ctx).collect();
        let causal = oracle::attention_rows(&q, &k, &v, 0, h, &all);
        for (r, row) in causal.iter().enumerate() {
            assert_eq!(
                row,
                &oracle::attention_decode(&q[r * qd..(r + 1) * qd], &k, &v, r + 1, h)
            );
            for head in 0..h.q_heads {
                let kh = head / (h.q_heads / h.kv_heads);
                for i in 0..hd {
                    let col = (0..=r).map(|j| v[j * kvd + kh * hd + i]);
                    let (lo, hi) =
                        col.fold((f32::MAX, f32::MIN), |(lo, hi), x| (lo.min(x), hi.max(x)));
                    let o = row[head * hd + i];
                    assert!(
                        o >= lo - 1e-5 && o <= hi + 1e-5,
                        "round {round}: {o} outside [{lo}, {hi}]"
                    );
                }
            }
        }
        // A chunk that starts at pos0 reads the same keys.
        let pos0 = rng.below(ctx);
        let tail: Vec<usize> = (0..ctx - pos0).collect();
        assert_eq!(
            oracle::attention_rows(&q[pos0 * qd..], &k, &v, pos0, h, &tail),
            causal[pos0..]
        );
    }
    // SwiGLU at known points, and where the kernel's formula has no value.
    let g = [
        0.0f32,
        1.0,
        -1.0,
        20.0,
        -20.0,
        100.0,
        -100.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let mut gate_up = g.to_vec();
    gate_up.extend(std::iter::repeat_n(2.0, g.len()));
    let out = oracle::swiglu(&gate_up);
    assert_eq!(out[0], 0.0);
    assert!((out[1] as f64 - 2.0 / (1.0 + (-1f64).exp())).abs() < 1e-6);
    assert!((out[2] as f64 + 2.0 / (1.0 + 1f64.exp())).abs() < 1e-6);
    assert!((out[3] - 40.0).abs() < 1e-5 && out[4].abs() < 1e-6 && out[5] == 200.0);
    assert!(out[6].abs() < 1e-40 && out[7] == f32::INFINITY);
    assert!(out[8].is_nan() && out[9].is_nan());
}

/// MoE routing: the chosen are the largest, in order, the weights are a distribution, and a
/// row without a finite routing has none.
#[test]
fn the_reference_routing_chooses_the_largest_and_normalises() {
    let mut rng = Rng::new(9);
    for _ in 0..2000 {
        let experts = rng.range(1, 24);
        let top_k = rng.range(1, experts);
        let mut row: Vec<f32> = if rng.chance(3) {
            (0..experts).map(|_| rng.below(3) as f32).collect()
        } else {
            rng.f32s(experts, 8.0)
        };
        let poisoned = rng.chance(8);
        if poisoned {
            row[rng.below(experts)] = *rng.pick(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
        }
        let (ids, w) = oracle::moe_route(&row, experts, top_k);
        assert_eq!((ids.len(), w.len()), (top_k, top_k));
        if poisoned {
            assert!(
                ids.iter().all(|&i| i == oracle::MOE_NO_EXPERT) && w.iter().all(|x| x.is_nan())
            );
            continue;
        }
        let mut seen = std::collections::BTreeSet::new();
        for pair in ids.windows(2) {
            let (a, b) = (pair[0] as usize, pair[1] as usize);
            assert!(
                row[a] > row[b] || (row[a] == row[b] && a < b),
                "{row:?}: {ids:?}"
            );
        }
        assert!(ids.iter().all(|&i| seen.insert(i)));
        let least = row[*ids.last().unwrap() as usize];
        assert!((0..experts)
            .filter(|i| !seen.contains(&(*i as u32)))
            .all(|i| row[i] <= least));
        assert!((w.iter().map(|&x| x as f64).sum::<f64>() - 1.0).abs() < 1e-6);
        assert!(w.windows(2).all(|p| p[0] >= p[1]) && w.iter().all(|&x| x > 0.0 || top_k > 1));
    }
    // An expert outside the table gives a NaN row; the others are their GEMV.
    let (w, x) = (rng.bf16s(3 * 4 * 8, 1.0), rng.bf16s(2 * 8, 1.0));
    let y = oracle::moe_gemv(&w, 3, 4, 8, &[2, 3, oracle::MOE_NO_EXPERT, 0], &x, 2);
    assert_eq!(y[..4], oracle::gemv(&w[2 * 32..], 4, 8, &x[..8])[..]);
    assert!(y[4..12].iter().all(|v| v.is_nan()));
    assert_eq!(y[12..], oracle::gemv(&w[..32], 4, 8, &x[8..])[..]);
    let c = oracle::moe_combine(&[1.0, 2.0, 3.0, 4.0], &[0.25, 0.75], 2, 2);
    assert_eq!(c, [2.5, 3.5]);
}

/// The reference forward: a token's logits depend on the tokens before it only; a cache
/// truncated and refilled is the cache; a cache is a recompute. Bit for bit: one code path.
#[test]
fn the_reference_forward_is_causal_and_its_cache_is_a_recompute() {
    for (seed, tied) in [(1u64, false), (2, true)] {
        let mut rng = Rng::new(seed);
        let c = tiny(2, 4, 2, tied);
        let tensors = model_tensors(&c, &mut rng);
        let m = reference(&c, &tensors);
        let ids = rng.tokens(12, c.vocab_size);
        let full = m.logits(&ids);
        assert!(full
            .iter()
            .all(|row| row.len() == c.vocab_size && row.iter().all(|v| v.is_finite())));
        // The logits move with the prompt: the model is not degenerate.
        assert_ne!(full[3], full[4]);
        assert!(full
            .iter()
            .any(|r| oracle::argmax(r) != oracle::argmax(&full[0])));

        // Causality: another future, the same past.
        let mut other = ids.clone();
        for t in &mut other[7..] {
            *t = (*t + 1) % c.vocab_size as u32;
        }
        let changed = m.logits(&other);
        assert_eq!(full[..7], changed[..7]);
        assert_ne!(full[7], changed[7]);

        // A cache carried forward is a recompute from nothing.
        let mut cache = Cache::default();
        let split = rng.range(1, 10);
        let a = m.extend(&mut cache, &ids[..split]);
        assert_eq!(a, full[split - 1]);
        let b = m.extend(&mut cache, &ids[split..]);
        assert_eq!(b, full[11]);
        assert_eq!(cache.len(), 12);

        // Truncated to a prefix and refilled with another suffix, then with the first again.
        let snapshot = cache.clone();
        cache.truncate(7);
        assert_eq!(m.extend(&mut cache, &other[7..]), changed[11]);
        cache.truncate(7);
        assert_eq!(m.extend(&mut cache, &ids[7..]), full[11]);
        assert_eq!(cache, snapshot);
        cache.truncate(0);
        assert!(cache.is_empty());
        assert_eq!(m.extend(&mut cache, &ids[..1]), full[0]);

        // Greedy by the reference: deterministic, and each token is the argmax of its logits.
        let greedy = |n: usize| {
            let mut cache = Cache::default();
            let mut logits = m.extend(&mut cache, &ids[..4]);
            let mut out = Vec::new();
            for _ in 0..n {
                let t = oracle::argmax(&logits).expect("finite logits");
                out.push(t);
                logits = m.step(&mut cache, t);
            }
            out
        };
        let g = greedy(8);
        assert_eq!(g, greedy(8));
        assert_eq!(g[..5], greedy(5)[..]);
    }
}

/// A model of `Greedy::fill` and `Greedy::next_token` (engine.rs): the same counters and the
/// same two conditions, over every `ahead` the engine accepts. It checks the arithmetic the
/// ring relies on, not the engine's code, which needs a device: a ring word is never rewritten
/// before the caller took it, no step is launched past the tokens asked, and at most `ahead`
/// steps are in flight. The engine's loop itself is in `gpu_engine.rs`.
#[test]
fn the_greedy_ring_never_overwrites_a_token_not_yet_taken() {
    const RING: usize = engine::GREEDY_AHEAD_MAX + 1;
    for ahead in 1..=engine::GREEDY_AHEAD_MAX {
        for max_tokens in 1..=70usize {
            // Ring word -> the step whose token it holds, not yet taken.
            let mut ring: [Option<usize>; RING] = [None; RING];
            let (mut taken, mut launched) = (0usize, 0usize);
            let fill = |taken: usize, launched: &mut usize, ring: &mut [Option<usize>; RING]| {
                while *launched + 1 < max_tokens && *launched - taken.saturating_sub(1) < ahead {
                    let j = *launched + 1;
                    assert!(
                        ring[j % RING].is_none(),
                        "ahead {ahead}: step {j} overwrites step {:?}",
                        ring[j % RING]
                    );
                    ring[j % RING] = Some(j);
                    *launched = j;
                }
            };
            fill(taken, &mut launched, &mut ring);
            while taken < max_tokens {
                if taken > 0 {
                    assert_eq!(
                        ring[taken % RING].take(),
                        Some(taken),
                        "ahead {ahead}: token {taken} is not in its word"
                    );
                }
                taken += 1;
                fill(taken, &mut launched, &mut ring);
                assert!(launched < max_tokens, "a step past the tokens asked");
                assert!(
                    launched + 1 - taken <= ahead,
                    "ahead {ahead}: {} steps in flight",
                    launched + 1 - taken
                );
            }
            assert_eq!(launched, max_tokens - 1, "every token asked was launched");
            assert!(ring.iter().all(Option::is_none));
        }
    }
}

/// Without a device the kernel surface reports it; it does not panic. (With one, it opens.)
#[test]
fn the_kernel_surface_reports_a_missing_device() {
    match ops::Kernels::new() {
        Ok(_) => eprintln!("MEASURE a GPU is present: Kernels::new() opened it"),
        Err(Error::Gpu(m)) => eprintln!("MEASURE no GPU: {m}"),
        Err(e) => panic!("{e}"),
    }
    let (set, nvcc) = x0ne_core_runtime::kernel_set();
    eprintln!("MEASURE kernel set {set}, nvcc {nvcc}");
}

/// The reference on what `gpu_engine.rs` asks of it, so that a GPU run does not find out: the
/// wider model's logits are finite and spread, a long sequence stays finite up to the model's
/// last position, and a value that is not finite in one embedding row reaches that position
/// and those after it, none before.
#[test]
fn the_reference_serves_the_models_of_the_gpu_tests() {
    let wide = x0ne_core_oracle::model::Config {
        hidden_size: 128,
        intermediate_size: 256,
        ..tiny(3, 16, 8, true)
    };
    let mut rng = Rng::new(3);
    let m = reference(&wide, &model_tensors(&wide, &mut rng));
    let rows = m.logits(&rng.tokens(40, wide.vocab_size));
    assert!(rows.iter().flatten().all(|v| v.is_finite()));
    let spread = rows
        .iter()
        .map(|r| r.iter().fold(0f32, |a, v| a.max(v.abs())))
        .fold(f32::MAX, f32::min);
    eprintln!("MEASURE wide model: the smallest row maximum is {spread}");
    assert!(spread > 0.1, "logits too flat to compare");

    let small = tiny(1, 2, 1, false);
    let m = reference(&small, &model_tensors(&small, &mut Rng::new(13)));
    let ids = Rng::new(24).tokens(small.max_position_embeddings, small.vocab_size);
    let t = std::time::Instant::now();
    let rows = m.logits(&ids);
    eprintln!(
        "MEASURE reference over {} positions: {:.2} s",
        ids.len(),
        t.elapsed().as_secs_f64()
    );
    assert!(rows.iter().flatten().all(|v| v.is_finite()));

    for special in [f32::INFINITY, f32::NAN] {
        let c = tiny(2, 4, 2, false);
        let mut tensors = model_tensors(&c, &mut Rng::new(16));
        let embed = tensors
            .iter_mut()
            .find(|t| t.name == "model.embed_tokens.weight")
            .unwrap();
        let mut values = bf16::from_bytes(&embed.data);
        values[7 * c.hidden_size + 3] = special;
        embed.data = bf16::bytes(&values);
        let m = reference(&c, &tensors);
        let mut ids: Vec<u32> = Rng::new(26)
            .tokens(30, c.vocab_size)
            .into_iter()
            .map(|t| if t == 7 { 8 } else { t })
            .collect();
        ids[12] = 7;
        let rows = m.logits(&ids);
        for (p, row) in rows.iter().enumerate() {
            if p < 12 {
                assert!(
                    row.iter().all(|v| v.is_finite()),
                    "{special}: position {p} before the token"
                );
            } else {
                assert!(
                    row.iter().all(|v| v.is_nan()),
                    "{special}: position {p} from the token on"
                );
            }
        }
    }
}
