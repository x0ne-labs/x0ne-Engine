#![allow(dead_code)]
//! Shared by the tests: deterministic data, checkpoints built in memory, comparisons.

use x0ne_core_oracle::model::{Config, Model};
use x0ne_core_oracle::{bf16, Heads};
use x0ne_core_runtime::config::Qwen3Config;

/// splitmix64: the same sequence on every machine, no dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in 0..n.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// Uniform in lo..=hi.
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }
    pub fn chance(&mut self, one_in: usize) -> bool {
        self.below(one_in) == 0
    }
    pub fn pick<'a, T>(&mut self, of: &'a [T]) -> &'a T {
        &of[self.below(of.len())]
    }
    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    pub fn f32s(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }
    /// Values of bf16, as f32.
    pub fn bf16s(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| bf16::round(self.unit() * scale)).collect()
    }
    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next_u64() as u8).collect()
    }
    pub fn tokens(&mut self, n: usize, vocab: usize) -> Vec<u32> {
        (0..n).map(|_| self.below(vocab) as u32).collect()
    }
}

/// The `half` values of f32 values that are bf16 already.
pub fn to_half(v: &[f32]) -> Vec<half::bf16> {
    v.iter()
        .map(|&x| half::bf16::from_bits(bf16::to_bits(x)))
        .collect()
}

pub fn from_half(v: &[half::bf16]) -> Vec<f32> {
    v.iter().map(|x| bf16::from_bits(x.to_bits())).collect()
}

/// Largest |got - want| / (atol + rtol |want|) and where; a value that is not finite on one
/// side only is infinitely far, and two NaN or two equal infinities agree.
pub fn worst_ratio(got: &[f32], want: &[f32], atol: f64, rtol: f64) -> (f64, usize) {
    assert_eq!(got.len(), want.len(), "length");
    let mut worst = (0f64, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let r = if g.is_nan() && w.is_nan() || g == w {
            0.0
        } else if g.is_finite() && w.is_finite() {
            (g as f64 - w as f64).abs() / (atol + rtol * (w as f64).abs())
        } else {
            f64::INFINITY
        };
        if r > worst.0 {
            worst = (r, i);
        }
    }
    worst
}

/// `|got - want| <= atol + rtol |want|` for every value; prints the worst ratio.
pub fn assert_close(what: &str, got: &[f32], want: &[f32], atol: f64, rtol: f64) {
    let (r, i) = worst_ratio(got, want, atol, rtol);
    eprintln!("MEASURE {what}: worst ratio {r:.4} (atol {atol:e}, rtol {rtol:e})");
    assert!(
        r <= 1.0,
        "{what}: index {i}: got {} want {} (ratio {r:.3}, atol {atol:e}, rtol {rtol:e})",
        got[i],
        want[i]
    );
}

/// One bf16 ulp, relative: the kernels' outputs are rounded to 8 significant bits.
pub const BF16_RTOL: f64 = 1.0 / 128.0;
/// Below this an absolute difference is not counted.
pub const BF16_ATOL: f64 = 1e-6;
/// An f32 accumulation against an f64 one, as a share of the sum of the terms' magnitudes.
pub const F32_RTOL: f64 = 1e-5;
/// Attention: bf16 rounding of a convex combination of V rows, as a share of max |v|.
pub const ATTENTION_ATOL_OF_VMAX: f64 = 1.0 / 128.0;
/// RoPE: a rotated value against the head's largest, f32 sine and cosine.
pub const ROPE_ATOL_OF_XMAX: f64 = 1e-3;

/// A tensor of a checkpoint under construction: name, dtype as safetensors spells it, shape,
/// bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

/// A safetensors header for `tensors`, laid out one after the other from offset 0.
pub fn header_of(tensors: &[Tensor]) -> String {
    let mut at = 0usize;
    let entries: Vec<String> = tensors
        .iter()
        .map(|t| {
            let e = format!(
                "{}:{{\"dtype\":\"{}\",\"shape\":{:?},\"data_offsets\":[{},{}]}}",
                serde_json::Value::String(t.name.clone()),
                t.dtype,
                t.shape,
                at,
                at + t.data.len()
            );
            at += t.data.len();
            e
        })
        .collect();
    format!("{{{}}}", entries.join(","))
}

/// A safetensors file from a header text and the data that follows it; the header is not
/// padded, so a test controls every byte.
pub fn raw_file(header: &[u8], data: &[u8]) -> Vec<u8> {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header);
    out.extend_from_slice(data);
    out
}

/// The safetensors file of `tensors`.
pub fn file_of(tensors: &[Tensor]) -> Vec<u8> {
    let data: Vec<u8> = tensors.iter().flat_map(|t| t.data.clone()).collect();
    raw_file(header_of(tensors).as_bytes(), &data)
}

/// A random checkpoint of `shapes`, BF16, values in [-scale, scale).
pub fn random_tensors(shapes: &[(String, Vec<usize>)], rng: &mut Rng, scale: f32) -> Vec<Tensor> {
    shapes
        .iter()
        .map(|(name, shape)| Tensor {
            name: name.clone(),
            dtype: "BF16".into(),
            shape: shape.clone(),
            data: bf16::bytes(&rng.bf16s(shape.iter().product(), scale)),
        })
        .collect()
}

pub fn get<'a>(tensors: &'a [Tensor], name: &str) -> &'a Tensor {
    tensors
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("no tensor {name}"))
}

/// The runtime's reading of the reference's config: its strict parser on the JSON.
pub fn runtime_config(c: &Config) -> Qwen3Config {
    Qwen3Config::from_json(&c.to_json()).unwrap_or_else(|e| panic!("{}: {e}", c.to_json()))
}

/// A small model the kernels accept: head_dim 128, hidden a multiple of 8.
pub fn tiny(layers: usize, q_heads: usize, kv_heads: usize, tied: bool) -> Config {
    Config {
        hidden_size: 64,
        intermediate_size: 96,
        num_hidden_layers: layers,
        num_attention_heads: q_heads,
        num_key_value_heads: kv_heads,
        head_dim: 128,
        vocab_size: 384,
        max_position_embeddings: 2048,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        tie_word_embeddings: tied,
    }
}

/// Random weights for `c` that keep the activations of a few layers in range: projections
/// around 1 / sqrt(fan-in), norm weights around 1, a wide embedding.
pub fn model_tensors(c: &Config, rng: &mut Rng) -> Vec<Tensor> {
    c.tensors()
        .iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let values: Vec<f32> = if shape.len() == 1 {
                (0..n)
                    .map(|_| bf16::round(1.0 + 0.25 * rng.unit()))
                    .collect()
            } else if name.contains("embed_tokens") || name.contains("lm_head") {
                rng.bf16s(n, 1.0)
            } else {
                rng.bf16s(n, 2.0 / (shape[1] as f32).sqrt())
            };
            Tensor {
                name: name.clone(),
                dtype: "BF16".into(),
                shape: shape.clone(),
                data: bf16::bytes(&values),
            }
        })
        .collect()
}

/// The reference model over `tensors`, with the inverse frequencies the host gives the device.
pub fn reference(c: &Config, tensors: &[Tensor]) -> Model {
    let inv = x0ne_core_runtime::ops::inv_freq(c.head_dim, c.rope_theta);
    Model::new(c.clone(), inv, |name| {
        bf16::from_bytes(&get(tensors, name).data)
    })
}

pub fn heads(h: Heads) -> x0ne_core_runtime::ops::Heads {
    x0ne_core_runtime::ops::Heads {
        q_heads: h.q_heads,
        kv_heads: h.kv_heads,
        head_dim: h.head_dim,
    }
}

/// A fresh directory under the system's temporary one, removed when dropped.
pub struct TempDir(pub std::path::PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let p = std::env::temp_dir().join(format!(
            "x0ne-core-oracle-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&p).expect("temporary directory");
        TempDir(p)
    }
    pub fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
