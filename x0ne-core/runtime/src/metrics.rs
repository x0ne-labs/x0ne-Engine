//! Comparisons for the model-level gates, computed in f64.

/// log(sum(exp(x))), stable.
pub fn logsumexp(x: &[f32]) -> f64 {
    let m = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    m + x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
}

/// First index of the maximum (torch.argmax). `None` on an empty slice or any NaN.
pub fn argmax(x: &[f32]) -> Option<u32> {
    if x.iter().any(|v| v.is_nan()) {
        return None;
    }
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in x.iter().enumerate() {
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((i, v));
        }
    }
    best.map(|(i, _)| i as u32)
}

/// KL(p_ref || p_test) in nats, both given as logits over the same vocabulary.
pub fn kl(reference: &[f32], test: &[f32]) -> f64 {
    assert_eq!(reference.len(), test.len(), "vocabulary sizes differ");
    let (lr, lt) = (logsumexp(reference), logsumexp(test));
    reference
        .iter()
        .zip(test)
        .map(|(&r, &t)| {
            let lp = r as f64 - lr;
            let lq = t as f64 - lt;
            lp.exp() * (lp - lq)
        })
        .sum()
}

/// exp(-mean(log p(next))) over finite entries.
pub fn perplexity(next_logprobs: &[f32]) -> f64 {
    let v: Vec<f64> = next_logprobs
        .iter()
        .filter(|x| x.is_finite())
        .map(|&x| x as f64)
        .collect();
    (-v.iter().sum::<f64>() / v.len() as f64).exp()
}

/// Share of equal entries.
pub fn agreement(a: &[u32], b: &[u32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len() as f64
}

/// Nearest-rank quantile, q in [0, 1].
pub fn quantile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let i = ((q * s.len() as f64).ceil() as usize).clamp(1, s.len()) - 1;
    s[i]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_is_zero_for_equal_and_shift_invariant() {
        let a = [1.0f32, 2.0, 3.0, -1.0];
        let b = [11.0f32, 12.0, 13.0, 9.0];
        assert!(kl(&a, &a).abs() < 1e-12);
        assert!(kl(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn kl_matches_a_hand_computed_case() {
        // p = (0.5, 0.5), q = (0.25, 0.75): KL = 0.5 ln 2 + 0.5 ln(2/3).
        let p = [0.0f32, 0.0];
        let q = [0.0f32, 3f32.ln()];
        let want = 0.5 * 2f64.ln() + 0.5 * (2.0f64 / 3.0).ln();
        assert!((kl(&p, &q) - want).abs() < 1e-7);
    }

    #[test]
    fn argmax_takes_the_first_maximum_and_refuses_nan() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), Some(1));
        assert_eq!(argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), Some(0));
        assert_eq!(argmax(&[1.0, f32::NAN]), None);
        assert_eq!(argmax(&[]), None);
    }

    #[test]
    fn perplexity_of_uniform_is_vocab() {
        let lp = vec![-(4f32.ln()); 10];
        assert!((perplexity(&lp) - 4.0).abs() < 1e-5);
    }

    #[test]
    fn quantile_nearest_rank() {
        let v = [5.0, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(quantile(&v, 0.5), 3.0);
        assert_eq!(quantile(&v, 1.0), 5.0);
        assert_eq!(quantile(&v, 0.0), 1.0);
    }
}
