//! Decode bandwidth ceiling. At batch 1, each generated token reads every weight
//! once and the whole KV cache once, so tok/s <= bandwidth / bytes per token.
//! Compute, launches, and caches are ignored: this is a ceiling, not a model.
//! A Qwen3-MoE token reads its active bytes: attention, router and the k chosen experts.

use crate::config::{Qwen3Config, Qwen3MoeConfig};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gpu {
    pub key: &'static str,
    pub name: &'static str,
    /// Decimal GB/s (1e9 bytes per second).
    pub bandwidth_gb_s: f64,
    pub source: &'static str,
}

pub const GPUS: &[Gpu] = &[
    Gpu {
        key: "rtx5090",
        name: "RTX 5090",
        bandwidth_gb_s: 1792.0,
        source: "NVIDIA spec sheet: 28 Gbps GDDR7, 512-bit bus",
    },
    Gpu {
        key: "rtx5090-measured", // gitleaks:allow (a GPU name, no secret)
        name: "RTX 5090, measured read",
        bandwidth_gb_s: 1698.86,
        source: "measured on one RTX 5090, 2026-10-20: median read bandwidth of a device-memory read kernel",
    },
    Gpu { key: "h100-sxm", name: "H100 SXM 80GB", bandwidth_gb_s: 3352.0, source: "NVIDIA H100 datasheet: 3.35 TB/s" },
    Gpu { key: "h100-pcie", name: "H100 PCIe 80GB", bandwidth_gb_s: 2039.0, source: "NVIDIA H100 datasheet: 2 TB/s" },
];

pub fn gpu(key: &str) -> Option<&'static Gpu> {
    GPUS.iter().find(|g| g.key == key)
}

/// Bytes one decode step moves, bf16 weights and bf16 KV cache. `kv_read` is
/// fractional because a measured decode is placed at its mean context.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenBytes {
    /// All layer weights and norms, final norm, LM head, one embedding row.
    pub weights: f64,
    /// K and V of every cached position the step attends, its own included.
    pub kv_read: f64,
}

impl TokenBytes {
    pub fn total(&self) -> f64 {
        self.weights + self.kv_read
    }
}

/// Weight bytes read per decoded token (the embedding is read for one row only).
pub fn weight_bytes_per_token(c: &Qwen3Config) -> u64 {
    let h = c.hidden_size as u64;
    let per_layer = c.linear_params_per_layer() + c.norm_params_per_layer();
    let params = c.num_hidden_layers as u64 * per_layer + h + c.vocab_size as u64 * h + h;
    params * 2
}

/// Active weight bytes read per decoded token by a Qwen3-MoE model: per layer the attention,
/// the router and `num_experts_per_tok` experts, then the head, final norm and one embedding row.
pub fn moe_weight_bytes_per_token(c: &Qwen3MoeConfig) -> u64 {
    let (b, m) = (&c.base, &c.moe);
    let h = b.hidden_size as u64;
    let attn = h * b.qkv_rows() as u64 + b.q_dim() as u64 * h;
    let router = m.num_experts as u64 * h;
    let experts = m.num_experts_per_tok as u64 * 3 * h * m.moe_intermediate_size as u64;
    let per_layer = attn + router + experts + b.norm_params_per_layer();
    let params = b.num_hidden_layers as u64 * per_layer + h + b.vocab_size as u64 * h + h;
    params * 2
}

/// Bytes for one decode step attending `ctx` positions.
pub fn token_bytes(c: &Qwen3Config, ctx: f64) -> TokenBytes {
    TokenBytes {
        weights: weight_bytes_per_token(c) as f64,
        kv_read: c.kv_bytes_per_position() as f64 * ctx,
    }
}

/// `token_bytes` of a dense model, or the active bytes of a MoE one.
pub fn token_bytes_of(c: &Qwen3Config, moe: Option<&Qwen3MoeConfig>, ctx: f64) -> TokenBytes {
    match moe {
        None => token_bytes(c, ctx),
        Some(m) => TokenBytes {
            weights: moe_weight_bytes_per_token(m) as f64,
            kv_read: m.base.kv_bytes_per_position() as f64 * ctx,
        },
    }
}

/// Mean attended context over a decode measured as (n - 1) / (t_n - t_1):
/// steps j = 2..=n attend prompt + j - 1 positions, whose mean is prompt + n / 2.
pub fn mean_decode_ctx(prompt_tokens: u64, generated: u64) -> f64 {
    prompt_tokens as f64 + generated as f64 / 2.0
}

pub fn ceiling_tok_s(c: &Qwen3Config, bandwidth_gb_s: f64, ctx: f64) -> f64 {
    bandwidth_gb_s * 1e9 / token_bytes(c, ctx).total()
}

/// One measurement placed against its ceiling.
#[derive(Debug, Clone, PartialEq)]
pub struct Placed {
    pub label: String,
    pub prompt_tokens: u64,
    pub generated: u64,
    pub mean_ctx: f64,
    pub gb_per_token: f64,
    pub ceiling_tok_s: f64,
    pub floor_ms: f64,
    pub measured_tok_s: f64,
    pub measured_ms: f64,
    pub percent_of_ceiling: f64,
    /// Measured step time minus the bandwidth floor.
    pub remainder_ms: f64,
}

pub fn place(
    c: &Qwen3Config,
    bandwidth_gb_s: f64,
    label: &str,
    prompt: u64,
    generated: u64,
    tok_s: f64,
) -> Placed {
    place_of(c, None, bandwidth_gb_s, label, prompt, generated, tok_s)
}

/// `place` against the active bytes when `moe` is `Some`.
pub fn place_of(
    c: &Qwen3Config,
    moe: Option<&Qwen3MoeConfig>,
    bandwidth_gb_s: f64,
    label: &str,
    prompt: u64,
    generated: u64,
    tok_s: f64,
) -> Placed {
    let mean_ctx = mean_decode_ctx(prompt, generated);
    let bytes = token_bytes_of(c, moe, mean_ctx).total();
    let ceiling = bandwidth_gb_s * 1e9 / bytes;
    Placed {
        label: label.to_owned(),
        prompt_tokens: prompt,
        generated,
        mean_ctx,
        gb_per_token: bytes / 1e9,
        ceiling_tok_s: ceiling,
        floor_ms: 1e3 / ceiling,
        measured_tok_s: tok_s,
        measured_ms: 1e3 / tok_s,
        percent_of_ceiling: 100.0 * tok_s / ceiling,
        remainder_ms: 1e3 / tok_s - 1e3 / ceiling,
    }
}

/// Reads `label<TAB>prompt_tokens<TAB>generated<TAB>tok_s` lines; `#` starts a comment.
pub fn parse_tsv(text: &str) -> Result<Vec<(String, u64, u64, f64)>, String> {
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 4 {
            return Err(format!(
                "line {}: expected 4 tab-separated fields, got {}",
                i + 1,
                f.len()
            ));
        }
        let num = |s: &str| {
            s.trim()
                .parse::<u64>()
                .map_err(|e| format!("line {}: {e}", i + 1))
        };
        let tok_s: f64 = f[3]
            .trim()
            .parse()
            .map_err(|e| format!("line {}: {e}", i + 1))?;
        if !(tok_s.is_finite() && tok_s > 0.0) {
            return Err(format!("line {}: tok/s must be positive", i + 1));
        }
        rows.push((f[0].trim().to_owned(), num(f[1])?, num(f[2])?, tok_s));
    }
    Ok(rows)
}

pub fn render(gpu_label: &str, bandwidth_gb_s: f64, rows: &[Placed]) -> String {
    let mut s = format!("# ceiling: {gpu_label}, {bandwidth_gb_s} GB/s; bf16 weights (active bytes for a MoE) and KV; mean ctx = prompt + generated/2\n");
    s.push_str("label\tprompt\tgen\tmean_ctx\tGB/token\tceiling_tok_s\tfloor_ms\tmeasured_tok_s\tmeasured_ms\t%ceiling\tremainder_ms\n");
    for p in rows {
        s.push_str(&format!(
            "{}\t{}\t{}\t{:.0}\t{:.4}\t{:.2}\t{:.3}\t{:.2}\t{:.3}\t{:.2}\t{:.3}\n",
            p.label,
            p.prompt_tokens,
            p.generated,
            p.mean_ctx,
            p.gb_per_token,
            p.ceiling_tok_s,
            p.floor_ms,
            p.measured_tok_s,
            p.measured_ms,
            p.percent_of_ceiling,
            p.remainder_ms
        ));
    }
    s
}
