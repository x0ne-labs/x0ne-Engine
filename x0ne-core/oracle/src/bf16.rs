//! bfloat16 by hand: the upper 16 bits of an f32. No dependency: the tests compare this with
//! the `half` crate the runtime uses, over every bit pattern.

/// The f32 a bf16 bit pattern stands for. Exact: every bf16 is an f32.
pub fn from_bits(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// `x` rounded to the nearest bf16, ties to the even mantissa (`__float2bfloat16_rn`). A value
/// past the largest bf16 rounds to infinity. A NaN gives a quiet NaN of the same sign.
pub fn to_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let (hi, lo) = (b >> 16, b & 0xFFFF);
    if x.is_nan() {
        return hi as u16 | 0x0040;
    }
    // The carry of a mantissa of all ones goes into the exponent: the next binade, or infinity.
    let up = lo > 0x8000 || (lo == 0x8000 && hi & 1 == 1);
    (hi + up as u32) as u16
}

/// `x` rounded to the nearest bf16, as an f32.
pub fn round(x: f32) -> f32 {
    from_bits(to_bits(x))
}

/// `round` over a slice.
pub fn round_all(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| round(v)).collect()
}

/// The little-endian bytes of the bf16 values of `x`, which must be bf16 values already.
pub fn bytes(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|&v| to_bits(v).to_le_bytes()).collect()
}

/// The values of little-endian bf16 bytes; a trailing odd byte is dropped.
pub fn from_bytes(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|&p| from_bits(u16::from_le_bytes(p)))
        .collect()
}

/// The gap between `x` and the next bf16 away from zero: 2^-7 of its binade (8 significant
/// bits), the unit the kernel tolerances are stated in.
pub fn ulp(x: f32) -> f64 {
    let bits = to_bits(x.abs());
    if bits >= 0x7F80 {
        return f64::INFINITY;
    }
    from_bits(bits + 1) as f64 - from_bits(bits) as f64
}
