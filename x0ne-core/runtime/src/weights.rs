//! Qwen3 dense and Qwen3-MoE weights. The checkpoint's tensor set is checked whole before any
//! byte is copied; then the weights are written into the arena with [q; k; v] and
//! [gate; up] concatenated by rows (row-major [out, in], so a byte concatenation). MoE experts
//! are packed by expert index: `[e][gate; up][hidden]` and `[e][hidden][moe_intermediate]`.

use crate::arena::{Arena, Backing, Building, Class, Frozen, Slice};
use crate::config::{Qwen3Config, Qwen3MoeConfig};
use crate::error::{format_err, Error, Result};
use crate::st::TensorSource;
use safetensors::Dtype;

pub const EMBED: &str = "model.embed_tokens.weight";
pub const FINAL_NORM: &str = "model.norm.weight";
pub const LM_HEAD: &str = "lm_head.weight";

pub fn layer_name(i: usize, leaf: &str) -> String {
    format!("model.layers.{i}.{leaf}.weight")
}

/// Leaf names of one layer, in the order the arena lays them out.
pub const LAYER_LEAVES: [&str; 11] = [
    "input_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "self_attn.o_proj",
    "post_attention_layernorm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

fn leaf_shape(c: &Qwen3Config, leaf: &str) -> Vec<usize> {
    let (h, i) = (c.hidden_size, c.intermediate_size);
    match leaf {
        "input_layernorm" | "post_attention_layernorm" => vec![h],
        "self_attn.q_proj" => vec![c.q_dim(), h],
        "self_attn.k_proj" | "self_attn.v_proj" => vec![c.kv_dim(), h],
        "self_attn.q_norm" | "self_attn.k_norm" => vec![c.head_dim],
        "self_attn.o_proj" => vec![h, c.q_dim()],
        "mlp.gate_proj" | "mlp.up_proj" => vec![i, h],
        "mlp.down_proj" => vec![h, i],
        _ => unreachable!("unknown leaf {leaf}"),
    }
}

/// Every tensor the checkpoint must hold, with its shape. A tied checkpoint may
/// also carry `lm_head.weight`; `check` then requires it to equal the embedding.
pub fn expected(c: &Qwen3Config) -> Vec<(String, Vec<usize>)> {
    expected_each(c).collect()
}

type Expected = (String, Vec<usize>);

/// The tensors after the layers: the final norm, and the LM head unless tied.
fn tail(c: &Qwen3Config) -> impl Iterator<Item = Expected> {
    let head =
        (!c.tie_word_embeddings).then(|| (LM_HEAD.to_owned(), vec![c.vocab_size, c.hidden_size]));
    std::iter::once((FINAL_NORM.to_owned(), vec![c.hidden_size])).chain(head)
}

/// `expected`, one tensor at a time: nothing is built for a layer that is not reached.
fn expected_each(c: &Qwen3Config) -> impl Iterator<Item = Expected> + '_ {
    let layers = (0..c.num_hidden_layers).flat_map(move |l| {
        LAYER_LEAVES
            .iter()
            .map(move |leaf| (layer_name(l, leaf), leaf_shape(c, leaf)))
    });
    std::iter::once((EMBED.to_owned(), vec![c.vocab_size, c.hidden_size]))
        .chain(layers)
        .chain(tail(c))
}

/// A decimal index below `end`, written as `layer_name` writes it (no sign, no leading zero).
fn index_below(text: &str, end: usize) -> bool {
    text.parse::<usize>()
        .is_ok_and(|i| i < end && i.to_string() == text)
}

/// The leaf of `model.layers.<l>.<leaf>.weight` when `l` is a layer of the config.
fn layer_leaf<'a>(c: &Qwen3Config, name: &'a str) -> Option<&'a str> {
    let rest = name
        .strip_prefix("model.layers.")?
        .strip_suffix(".weight")?;
    let (l, leaf) = rest.split_once('.')?;
    index_below(l, c.num_hidden_layers).then_some(leaf)
}

/// True for the tensors outside the layers; a tied checkpoint may carry `lm_head.weight` too.
fn is_outer(name: &str) -> bool {
    [EMBED, FINAL_NORM, LM_HEAD].contains(&name)
}

/// True when `expected` lists `name` (or it is a tied checkpoint's `lm_head.weight`), without
/// listing: the config states how many layers there are, the checkpoint how many are looked at.
fn is_expected(c: &Qwen3Config, name: &str) -> bool {
    is_outer(name) || layer_leaf(c, name).is_some_and(|leaf| LAYER_LEAVES.contains(&leaf))
}

/// As `is_expected`, for the tensor set of `moe_expected`.
fn is_moe_expected(c: &Qwen3MoeConfig, name: &str) -> bool {
    let expert = |leaf: &str| {
        leaf.strip_prefix("mlp.experts.")
            .and_then(|rest| rest.split_once('.'))
            .is_some_and(|(e, proj)| {
                index_below(e, c.moe.num_experts)
                    && ["gate_proj", "up_proj", "down_proj"].contains(&proj)
            })
    };
    is_outer(name)
        || layer_leaf(&c.base, name).is_some_and(|leaf| {
            leaf == "mlp.gate"
                || expert(leaf)
                || (!leaf.starts_with("mlp.") && LAYER_LEAVES.contains(&leaf))
        })
}

/// Refuses a missing tensor, an extra tensor, a dtype other than BF16, a wrong shape,
/// or a tied checkpoint whose `lm_head.weight` differs from the embedding. The expected
/// tensors are looked up one at a time and the first one missing ends the check: its cost
/// follows the checkpoint, not the number of layers the config states.
pub fn check(c: &Qwen3Config, src: &impl TensorSource) -> Result<()> {
    c.check_sizes()?;
    check_set(
        expected_each(c),
        |name| is_expected(c, name),
        c.tie_word_embeddings,
        src,
    )
}

fn check_set(
    want: impl Iterator<Item = Expected>,
    allowed: impl Fn(&str) -> bool,
    tied: bool,
    src: &impl TensorSource,
) -> Result<()> {
    for name in src.names() {
        if !allowed(&name) {
            return format_err(format!("{name}: unexpected tensor"));
        }
    }
    for (name, shape) in want {
        let t = src.tensor(&name)?;
        if t.dtype != Dtype::BF16 {
            return format_err(format!("{name}: dtype {:?}, expected BF16", t.dtype));
        }
        if t.shape != shape {
            return format_err(format!("{name}: shape {:?}, expected {shape:?}", t.shape));
        }
    }
    if tied && src.names().iter().any(|n| n == LM_HEAD) {
        let (head, embed) = (src.tensor(LM_HEAD)?, src.tensor(EMBED)?);
        if head.dtype != Dtype::BF16 || head.shape != embed.shape || head.data != embed.data {
            return format_err("tied config, but lm_head.weight differs from the embedding");
        }
    }
    Ok(())
}

/// Every tensor of a Qwen3-MoE checkpoint, with its shape: the attention and norms of Qwen3,
/// then per layer the router `mlp.gate` and the experts `mlp.experts.<e>.{gate,up,down}_proj`.
pub fn moe_expected(c: &Qwen3MoeConfig) -> Vec<(String, Vec<usize>)> {
    moe_expected_each(c).collect()
}

/// `moe_expected`, one tensor at a time.
fn moe_expected_each(c: &Qwen3MoeConfig) -> impl Iterator<Item = Expected> + '_ {
    let b = &c.base;
    let (h, i, n) = (
        b.hidden_size,
        c.moe.moe_intermediate_size,
        c.moe.num_experts,
    );
    let layers = (0..b.num_hidden_layers).flat_map(move |l| {
        let attention = LAYER_LEAVES
            .iter()
            .filter(|leaf| !leaf.starts_with("mlp."))
            .map(move |leaf| (layer_name(l, leaf), leaf_shape(b, leaf)));
        let router = std::iter::once((layer_name(l, "mlp.gate"), vec![n, h]));
        let experts = (0..n).flat_map(move |e| {
            let x = move |leaf: &str| layer_name(l, &format!("mlp.experts.{e}.{leaf}"));
            [
                (x("gate_proj"), vec![i, h]),
                (x("up_proj"), vec![i, h]),
                (x("down_proj"), vec![h, i]),
            ]
        });
        attention.chain(router).chain(experts)
    });
    std::iter::once((EMBED.to_owned(), vec![b.vocab_size, h]))
        .chain(layers)
        .chain(tail(b))
}

/// As `check`, for a Qwen3-MoE checkpoint: the exact tensor set of `c`, all BF16.
pub fn check_moe(c: &Qwen3MoeConfig, src: &impl TensorSource) -> Result<()> {
    c.check_sizes()?;
    check_set(
        moe_expected_each(c),
        |name| is_moe_expected(c, name),
        c.base.tie_word_embeddings,
        src,
    )
}

#[derive(Debug, Clone)]
pub struct LayerSlices {
    pub input_norm: Slice,
    /// [q_dim + 2 kv_dim, hidden]: q rows, then k rows, then v rows.
    pub qkv: Slice,
    pub q_norm: Slice,
    pub k_norm: Slice,
    pub o: Slice,
    pub post_norm: Slice,
    /// Dense: [2 intermediate, hidden], gate rows then up rows. MoE: the same per expert,
    /// [num_experts][2 moe_intermediate, hidden].
    pub gate_up: Slice,
    /// Dense: [hidden, intermediate]. MoE: [num_experts][hidden, moe_intermediate].
    pub down: Slice,
    /// MoE only: the router `mlp.gate`, [num_experts, hidden].
    pub router: Option<Slice>,
}

#[derive(Debug, Clone)]
pub struct WeightSlices {
    pub embed: Slice,
    pub layers: Vec<LayerSlices>,
    pub final_norm: Slice,
    /// `None` when tied: the LM head reads `embed`.
    pub lm_head: Option<Slice>,
}

/// Bytes of a bf16 tensor of `shape`. A product past u64 is refused, never wrapped.
fn bf16_bytes(shape: &[usize]) -> Result<u64> {
    shape
        .iter()
        .try_fold(2u64, |bytes, &dim| {
            bytes.checked_mul(u64::try_from(dim).ok()?)
        })
        .ok_or_else(|| {
            Error::Format(format!(
                "a bf16 tensor of shape {shape:?} overflows u64 bytes"
            ))
        })
}

/// Bytes of several tensors packed in one region.
fn packed_bytes(shapes: &[Vec<usize>]) -> Result<u64> {
    shapes.iter().try_fold(0u64, |sum, shape| {
        sum.checked_add(bf16_bytes(shape)?)
            .ok_or_else(|| Error::Format("packed tensors overflow u64 bytes".into()))
    })
}

/// Plans every weight region in a building arena. A config whose sizes overflow is refused.
pub fn plan(c: &Qwen3Config, a: &mut Arena<Building>) -> Result<WeightSlices> {
    c.check_sizes()?;
    let gate_up = packed_bytes(&[leaf_shape(c, "mlp.gate_proj"), leaf_shape(c, "mlp.up_proj")])?;
    let down = bf16_bytes(&leaf_shape(c, "mlp.down_proj"))?;
    plan_with(c, a, |a, l| {
        Ok((
            a.alloc(&format!("l{l}.gate_up"), gate_up, Class::Weights)?,
            a.alloc(&format!("l{l}.down"), down, Class::Weights)?,
            None,
        ))
    })
}

/// As `plan`, for Qwen3-MoE: per layer the router, then every expert's [gate; up] packed by
/// expert index, then every expert's down.
pub fn plan_moe(c: &Qwen3MoeConfig, a: &mut Arena<Building>) -> Result<WeightSlices> {
    let (h, i, n) = (
        c.base.hidden_size,
        c.moe.moe_intermediate_size,
        c.moe.num_experts,
    );
    c.check_sizes()?;
    let router = bf16_bytes(&[n, h])?;
    let gate_up = bf16_bytes(&[n, 2, i, h])?;
    let down = bf16_bytes(&[n, h, i])?;
    plan_with(&c.base, a, |a, l| {
        let w = Class::Weights;
        let router = a.alloc(&format!("l{l}.router"), router, w)?;
        Ok((
            a.alloc(&format!("l{l}.experts_gate_up"), gate_up, w)?,
            a.alloc(&format!("l{l}.experts_down"), down, w)?,
            Some(router),
        ))
    })
}

type FfnPlan = (Slice, Slice, Option<Slice>);

fn plan_with(
    c: &Qwen3Config,
    a: &mut Arena<Building>,
    mut ffn: impl FnMut(&mut Arena<Building>, usize) -> Result<FfnPlan>,
) -> Result<WeightSlices> {
    let s = |leaf: &str| bf16_bytes(&leaf_shape(c, leaf));
    let w = Class::Weights;
    let table = bf16_bytes(&[c.vocab_size, c.hidden_size])?;
    // The sizes of a layer are those of every layer: computed, and refused, once.
    let qkv_bytes = packed_bytes(&[
        leaf_shape(c, "self_attn.q_proj"),
        leaf_shape(c, "self_attn.k_proj"),
        leaf_shape(c, "self_attn.v_proj"),
    ])?;
    let (norm, head_norm, o_bytes) = (
        s("input_layernorm")?,
        s("self_attn.q_norm")?,
        s("self_attn.o_proj")?,
    );
    let embed = a.alloc("embed", table, w)?;
    let mut layers = Vec::with_capacity(c.num_hidden_layers);
    for l in 0..c.num_hidden_layers {
        let input_norm = a.alloc(&format!("l{l}.input_norm"), norm, w)?;
        let qkv = a.alloc(&format!("l{l}.qkv"), qkv_bytes, w)?;
        let q_norm = a.alloc(&format!("l{l}.q_norm"), head_norm, w)?;
        let k_norm = a.alloc(&format!("l{l}.k_norm"), head_norm, w)?;
        let o = a.alloc(&format!("l{l}.o"), o_bytes, w)?;
        let post_norm = a.alloc(&format!("l{l}.post_norm"), norm, w)?;
        let (gate_up, down, router) = ffn(a, l)?;
        layers.push(LayerSlices {
            input_norm,
            qkv,
            q_norm,
            k_norm,
            o,
            post_norm,
            gate_up,
            down,
            router,
        });
    }
    let final_norm = a.alloc("final_norm", norm, w)?;
    let lm_head = if c.tie_word_embeddings {
        None
    } else {
        Some(a.alloc("lm_head", table, w)?)
    };
    Ok(WeightSlices {
        embed,
        layers,
        final_norm,
        lm_head,
    })
}

/// Copies the checked checkpoint into the frozen arena. Returns the bytes written.
pub fn upload<B: Backing>(
    c: &Qwen3Config,
    src: &impl TensorSource,
    ws: &WeightSlices,
    a: &mut Arena<Frozen<B>>,
) -> Result<u64> {
    check(c, src)?;
    upload_with(src, ws, a, |l| {
        let n = |leaf: &str| layer_name(l, leaf);
        (
            vec![n("mlp.gate_proj"), n("mlp.up_proj")],
            vec![n("mlp.down_proj")],
            Vec::new(),
        )
    })
}

/// As `upload`, for a Qwen3-MoE checkpoint planned by `plan_moe`.
pub fn upload_moe<B: Backing>(
    c: &Qwen3MoeConfig,
    src: &impl TensorSource,
    ws: &WeightSlices,
    a: &mut Arena<Frozen<B>>,
) -> Result<u64> {
    check_moe(c, src)?;
    let n = c.moe.num_experts;
    upload_with(src, ws, a, |l| {
        let x = |e: usize, leaf: &str| layer_name(l, &format!("mlp.experts.{e}.{leaf}"));
        (
            (0..n)
                .flat_map(|e| [x(e, "gate_proj"), x(e, "up_proj")])
                .collect(),
            (0..n).map(|e| x(e, "down_proj")).collect(),
            vec![layer_name(l, "mlp.gate")],
        )
    })
}

/// Tensor names of one layer's feed-forward: [gate; up] slice, down slice, router slice.
type FfnNames = (Vec<String>, Vec<String>, Vec<String>);

fn upload_with<B: Backing>(
    src: &impl TensorSource,
    ws: &WeightSlices,
    a: &mut Arena<Frozen<B>>,
    ffn: impl Fn(usize) -> FfnNames,
) -> Result<u64> {
    let mut written = 0u64;
    let mut put = |a: &mut Arena<Frozen<B>>, slice: Slice, names: &[String]| -> Result<()> {
        let mut at = 0u64;
        for n in names {
            let t = src.tensor(n)?;
            a.write(slice, at, t.data)?;
            at += t.data.len() as u64;
        }
        if at != slice.len() {
            return format_err(format!(
                "{names:?}: wrote {at} bytes into a {}-byte slice",
                slice.len()
            ));
        }
        written += at;
        Ok(())
    };
    put(a, ws.embed, &[EMBED.to_owned()])?;
    for (l, ls) in ws.layers.iter().enumerate() {
        let n = |leaf: &str| layer_name(l, leaf);
        put(a, ls.input_norm, &[n("input_layernorm")])?;
        put(
            a,
            ls.qkv,
            &[
                n("self_attn.q_proj"),
                n("self_attn.k_proj"),
                n("self_attn.v_proj"),
            ],
        )?;
        put(a, ls.q_norm, &[n("self_attn.q_norm")])?;
        put(a, ls.k_norm, &[n("self_attn.k_norm")])?;
        put(a, ls.o, &[n("self_attn.o_proj")])?;
        put(a, ls.post_norm, &[n("post_attention_layernorm")])?;
        let (gate_up, down, router) = ffn(l);
        put(a, ls.gate_up, &gate_up)?;
        put(a, ls.down, &down)?;
        match (ls.router, router.is_empty()) {
            (Some(slice), false) => put(a, slice, &router)?,
            (None, true) => {}
            _ => return format_err(format!("layer {l}: router slice and tensor disagree")),
        }
    }
    put(a, ws.final_norm, &[FINAL_NORM.to_owned()])?;
    if let Some(head) = ws.lm_head {
        put(a, head, &[LM_HEAD.to_owned()])?;
    }
    Ok(written)
}

/// Resident weight bytes, embedding and (untied) LM head included.
pub fn resident_bytes(c: &Qwen3Config) -> u64 {
    total_bytes(expected_each(c))
}

/// The bytes of a tensor set. A parsed config cannot overflow them; one built by hand can, and
/// panics here instead of wrapping, as the sizes of `config.rs` do.
fn total_bytes(mut tensors: impl Iterator<Item = Expected>) -> u64 {
    tensors
        .try_fold(0u64, |sum, (_, shape)| {
            sum.checked_add(bf16_bytes(&shape).ok()?)
        })
        .expect("a config's weight bytes overflow; from_json refuses such a config")
}

/// As `resident_bytes`, for Qwen3-MoE: every expert of every layer is resident.
pub fn resident_bytes_moe(c: &Qwen3MoeConfig) -> u64 {
    total_bytes(moe_expected_each(c))
}

/// Decodes bf16 bytes (little-endian) to f32. Exact: every bf16 is an f32.
pub fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| half::bf16::from_bits(u16::from_le_bytes(b)).to_f32())
        .collect()
}
