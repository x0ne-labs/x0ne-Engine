//! Token draws on the host, without candle.
//!
//! For one seed, [`Sampler`] draws the tokens candle 0.11's `LogitsProcessor`
//! draws: the same `StdRng` and `WeightedIndex` of rand 0.9, the temperature
//! and the softmax in f32 in candle's order of operations (affine
//! `v * (1 / t) + 0`, sequential max, `exp`, candle's sum, division). The
//! tests prove it against candle, kept as the reference. A seed signed in a
//! receipt therefore gives the tokens it gave before.
//!
//! What is faster without changing a draw: candle sorts the whole vocabulary
//! (151,936 entries for Qwen3) for each `top_p` token. Here only the nucleus
//! is sorted, found from the mass of each bucket of values, under the same
//! total order (probability descending, then index ascending: candle's stable
//! sort), and the draw runs over the nucleus alone. Weights outside it are 0 for candle:
//! they add nothing to its cumulative sums, so the index drawn is the same.
//!
//! [`Sampler`]: crate::sampler::Sampler

use std::cmp::Ordering;

use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::backend::Sampling;

/// How a row becomes a token.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Draw {
    ArgMax,
    All { temperature: f64 },
    TopP { p: f64, temperature: f64 },
}

/// One request's draws: no state outlives it.
pub struct Sampler {
    rng: StdRng,
    draw: Draw,
    /// The probabilities of the last row, reused across tokens.
    prs: Vec<f32>,
    /// Vocabulary indices, reused by the nucleus selection.
    order: Vec<u32>,
}

impl Sampler {
    /// Engine's reading of OpenAI's parameters: `temperature` under 1e-7 or
    /// `top_p` 0 or below is argmax (the nucleus of mass 0 is the top token),
    /// `top_p` 1 or above is the whole distribution.
    pub fn new(s: Sampling) -> Self {
        let draw = if s.temperature < 1e-7 || s.top_p <= 0.0 {
            Draw::ArgMax
        } else if s.top_p >= 1.0 {
            Draw::All {
                temperature: s.temperature,
            }
        } else {
            Draw::TopP {
                p: s.top_p,
                temperature: s.temperature,
            }
        };
        Self {
            rng: StdRng::seed_from_u64(s.seed),
            draw,
            prs: Vec::new(),
            order: Vec::new(),
        }
    }

    /// Whether every draw is the row's argmax.
    pub fn is_greedy(&self) -> bool {
        self.draw == Draw::ArgMax
    }

    /// The token drawn from `row`, the f32 logits of one position. A row that
    /// is empty or holds a NaN draws none.
    pub fn sample(&mut self, row: &[f32]) -> anyhow::Result<u32> {
        if row.is_empty() || row.iter().any(|v| v.is_nan()) {
            anyhow::bail!("the logits hold a NaN: no token");
        }
        match self.draw {
            Draw::ArgMax => crate::stepped::argmax(row)
                .ok_or_else(|| anyhow::anyhow!("the logits hold a NaN: no token")),
            Draw::All { temperature } => {
                softmax_into(row, temperature, &mut self.prs);
                let d =
                    WeightedIndex::new(&self.prs).map_err(|e| anyhow::anyhow!("sampling: {e}"))?;
                to_token(d.sample(&mut self.rng))
            }
            Draw::TopP { p, temperature } => {
                softmax_into(row, temperature, &mut self.prs);
                let kept = nucleus(&self.prs, p as f32, &mut self.order);
                // Index order: candle's cumulative sums, without its zeros.
                kept.sort_unstable();
                let d = WeightedIndex::new(kept.iter().map(|&i| self.prs[i as usize]))
                    .map_err(|e| anyhow::anyhow!("sampling: {e}"))?;
                Ok(kept[d.sample(&mut self.rng)])
            }
        }
    }
}

fn to_token(i: usize) -> anyhow::Result<u32> {
    u32::try_from(i).map_err(|_| anyhow::anyhow!("sampling: index {i} past u32"))
}

/// candle 0.11: `(&logits / t)` is `affine(1 / t, 0)`, then `softmax_last_dim`.
fn softmax_into(row: &[f32], temperature: f64, prs: &mut Vec<f32>) {
    let mul = (1.0 / temperature) as f32;
    let add = 0.0f32;
    prs.clear();
    prs.extend(row.iter().map(|&v| v * mul + add));
    let mut max = prs[0];
    for &v in &prs[1..] {
        max = max.max(v);
    }
    for d in prs.iter_mut() {
        *d = (*d - max).exp();
    }
    let sum = candle_sum(prs);
    for d in prs.iter_mut() {
        *d /= sum;
    }
}

/// candle-core 0.11 `cpu::vec_sum`, x86 without AVX2 (the default target):
/// one accumulator, in index order.
#[cfg(not(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
)))]
fn candle_sum(xs: &[f32]) -> f32 {
    let mut b = 0f32;
    for &x in xs {
        b += x;
    }
    b
}

/// candle-core 0.11 `cpu::vec_sum` with AVX2 (`cpu/avx.rs`): 4 registers of
/// 8 lanes over blocks of 32, reduced by its fixed tree, then the remainder.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
))]
fn candle_sum(xs: &[f32]) -> f32 {
    let np = xs.len() & !31;
    let mut acc = [0f32; 32];
    for block in xs[..np].chunks_exact(32) {
        for (a, &x) in acc.iter_mut().zip(block) {
            *a += x;
        }
    }
    let mut r = [0f32; 8];
    for (l, v) in r.iter_mut().enumerate() {
        *v = (acc[l] + acc[8 + l]) + (acc[16 + l] + acc[24 + l]);
    }
    let t0 = [r[0] + r[4], r[1] + r[5], r[2] + r[6], r[3] + r[7]];
    let mut b = (t0[0] + t0[1]) + (t0[2] + t0[3]);
    for &x in &xs[np..] {
        b += x;
    }
    b
}

/// candle's sort order for `top_p`: probability descending (`total_cmp`),
/// then index ascending (its sort is stable).
fn before(prs: &[f32], i: u32, j: u32) -> Ordering {
    prs[j as usize].total_cmp(&prs[i as usize]).then(i.cmp(&j))
}

/// Buckets of the nucleus search: the top 12 bits of a non-negative f32 below
/// its sign (exponent and 4 mantissa bits), in the order of the values.
const BUCKET_SHIFT: u32 = 19;
const BUCKETS: usize = 1 << (31 - BUCKET_SHIFT);

/// The indices candle keeps for `top_p` (each one whose predecessors in the
/// sorted order sum to less than `top_p`, in f32 and in that order), in
/// sorted order. The mass of each bucket of values, summed once, says from
/// which bucket down the nucleus closes; only those candidates are sorted,
/// and the exact f32 sum decides. A nucleus the sum leaves open takes the
/// next bucket.
fn nucleus<'o>(prs: &[f32], top_p: f32, order: &'o mut Vec<u32>) -> &'o mut [u32] {
    let bucket = |p: f32| (p.to_bits() >> BUCKET_SHIFT) as usize;
    let mut mass = vec![0f64; BUCKETS];
    for &p in prs {
        // A negative value's bits are past the table; its draw fails, as candle's.
        if let Some(m) = mass.get_mut(bucket(p)) {
            *m += f64::from(p);
        }
    }
    // A NaN (an infinite logit) never lets candle's sum reach top_p: it keeps
    // every index, and the draw fails on the NaN weight.
    if mass.iter().any(|m| m.is_nan()) {
        order.clear();
        order.extend(0..prs.len() as u32);
        order.sort_unstable_by(|&i, &j| before(prs, i, j));
        let n = order.len();
        return &mut order[..n];
    }
    let mut from = BUCKETS;
    let mut seen = 0f64;
    while from > 0 && seen < f64::from(top_p) {
        from -= 1;
        seen += mass[from];
    }
    loop {
        order.clear();
        order.extend(
            prs.iter()
                .enumerate()
                .filter(|(_, &p)| from == 0 || bucket(p) >= from)
                .map(|(i, _)| i as u32),
        );
        order.sort_unstable_by(|&i, &j| before(prs, i, j));
        let mut cumsum = 0f32;
        for (k, &i) in order.iter().enumerate() {
            if cumsum >= top_p {
                return &mut order[..k];
            }
            cumsum += prs[i as usize];
        }
        // Closed on the last candidate (candle stops at the next one), or nothing left.
        if cumsum >= top_p || from == 0 {
            let n = order.len();
            return &mut order[..n];
        }
        from -= 1;
        while from > 0 && mass[from] == 0.0 {
            from -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// candle's draw: the processor Engine used before this module
    /// (`candle_backend::logits_processor`), on a CPU tensor.
    fn candle_draws(s: Sampling, rows: &[Vec<f32>]) -> Vec<u32> {
        let mut p = crate::candle_backend::logits_processor(s);
        rows.iter()
            .map(|r| {
                p.sample(&Tensor::from_slice(r, r.len(), &Device::Cpu).unwrap())
                    .unwrap()
            })
            .collect()
    }

    fn our_draws(s: Sampling, rows: &[Vec<f32>]) -> Vec<u32> {
        let mut p = Sampler::new(s);
        rows.iter().map(|r| p.sample(r).unwrap()).collect()
    }

    /// A deterministic row: `n` logits from a xorshift, spread by `scale`,
    /// with `ties` copies of the top value at the first indices.
    fn row(n: usize, seed: u64, scale: f32, ties: usize) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut r: Vec<f32> = (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
            })
            .collect();
        let top = r.iter().copied().fold(f32::NEG_INFINITY, f32::max) + 1.0;
        for v in r.iter_mut().take(ties) {
            *v = top;
        }
        r
    }

    /// Bit for bit candle's draws: every mapping of the parameters, peaked and
    /// flat rows, ties, the Qwen3 vocabulary size; 40 tokens per seed on the
    /// small rows, 4 on the large ones.
    #[test]
    fn draws_are_candles() {
        let mut checked = 0;
        for &(n, scale, ties) in &[
            (151_936usize, 30.0f32, 0usize),
            (151_936, 8.0, 0),
            (151_936, 2.0, 3),
            (1_000, 1.0, 0),
            (7, 0.0, 7),
            (33, 4.0, 2),
        ] {
            // Fewer tokens on the large rows: candle's full sort is slow in a debug build.
            let tokens = if n > 10_000 { 4 } else { 40 };
            let rows: Vec<Vec<f32>> = (0..tokens)
                .map(|k| row(n, k as u64 + 1, scale, ties))
                .collect();
            for &(temperature, top_p) in &[
                (0.0, 1.0),
                (1e-8, 0.5),
                (0.7, 0.0),
                (0.7, 1.0),
                (1.0, 1.5),
                (0.7, 0.9),
                (1.0, 0.95),
                (1.3, 0.5),
                (0.2, 0.05),
                (2.0, 0.999),
            ] {
                for seed in [0u64, 1, 299_792_458] {
                    let s = Sampling {
                        temperature,
                        top_p,
                        seed,
                    };
                    assert_eq!(
                        our_draws(s, &rows),
                        candle_draws(s, &rows),
                        "n {n} scale {scale} ties {ties} t {temperature} top_p {top_p} seed {seed}"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 180);
    }

    /// The nucleus is candle's set: every index candle keeps, none it zeroes,
    /// across the growth of the selection (a flat row needs several rounds).
    #[test]
    fn nucleus_is_candles_set() {
        for &(n, scale, top_p) in &[
            (151_936usize, 30.0f32, 0.9f32),
            (151_936, 0.5, 0.9),
            (151_936, 0.0, 0.3),
            (100, 3.0, 0.999_999),
            (64, 1.0, 0.5),
            (65, 1.0, 0.5),
        ] {
            let mut prs = Vec::new();
            softmax_into(&row(n, 7, scale, 0), 1.0, &mut prs);
            // candle's sample_topp, minus the draw.
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));
            let mut cumsum = 0f32;
            let mut want = Vec::new();
            for &i in &idx {
                if cumsum >= top_p {
                    break;
                }
                cumsum += prs[i];
                want.push(i as u32);
            }
            let mut order = Vec::new();
            let got = nucleus(&prs, top_p, &mut order).to_vec();
            assert_eq!(got, want, "n {n} scale {scale} top_p {top_p}");
        }
    }

    /// The softmax is candle's, value for value.
    #[test]
    fn softmax_is_candles() {
        for (n, scale, t) in [
            (151_936usize, 20.0f32, 0.7f64),
            (1_000, 3.0, 1.3),
            (5, 0.0, 1.0),
        ] {
            let r = row(n, 3, scale, 0);
            let mut ours = Vec::new();
            softmax_into(&r, t, &mut ours);
            let logits = (Tensor::from_slice(&r, n, &Device::Cpu).unwrap() / t).unwrap();
            let theirs: Vec<f32> = candle_nn::ops::softmax_last_dim(&logits)
                .unwrap()
                .to_vec1()
                .unwrap();
            let same = ours
                .iter()
                .zip(&theirs)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "n {n} t {t}");
        }
    }

    #[test]
    fn greedy_and_refusals() {
        let greedy = |t: f64, p: f64| {
            Sampler::new(Sampling {
                temperature: t,
                top_p: p,
                seed: 0,
            })
            .is_greedy()
        };
        assert!(greedy(0.0, 0.9) && greedy(0.5, 0.0) && greedy(0.5, -1.0));
        assert!(!greedy(0.5, 0.9) && !greedy(0.5, 1.0));
        let mut s = Sampler::new(Sampling {
            temperature: 0.7,
            top_p: 0.9,
            seed: 0,
        });
        assert!(s.sample(&[]).is_err());
        assert!(s.sample(&[0.0, f32::NAN]).is_err());
        assert_eq!(s.sample(&[f32::NEG_INFINITY, 1.0]).unwrap(), 1);
        // An infinite logit: a NaN probability, refused by candle's draw and this one.
        let inf = [0.5, f32::INFINITY, 1.0];
        assert!(s.sample(&inf).is_err());
        let mut p = crate::candle_backend::logits_processor(Sampling {
            temperature: 0.7,
            top_p: 0.9,
            seed: 0,
        });
        assert!(p
            .sample(&Tensor::from_slice(&inf, 3, &Device::Cpu).unwrap())
            .is_err());
    }

    /// Release build (`--ignored --nocapture`): time per draw, candle and here,
    /// on the Qwen3 vocabulary. Same tokens checked on the way.
    #[test]
    #[ignore = "timing: cargo test --release -p x0ne-infer sampler::tests::speed -- --ignored --nocapture"]
    fn speed() {
        let rows: Vec<Vec<f32>> = (0..200).map(|k| row(151_936, k + 11, 24.0, 0)).collect();
        for &(temperature, top_p) in &[(0.7, 0.9), (1.0, 0.95), (0.7, 1.0)] {
            let s = Sampling {
                temperature,
                top_p,
                seed: 42,
            };
            let t0 = std::time::Instant::now();
            let theirs = candle_draws(s, &rows);
            let candle = t0.elapsed().as_secs_f64() / rows.len() as f64;
            let t1 = std::time::Instant::now();
            let ours = our_draws(s, &rows);
            let here = t1.elapsed().as_secs_f64() / rows.len() as f64;
            assert_eq!(ours, theirs);
            println!(
                "MEASURE sampler t {temperature} top_p {top_p}: candle {:.1} us, here {:.1} us per token ({:.1}x)",
                candle * 1e6,
                here * 1e6,
                candle / here
            );
        }
    }
}
