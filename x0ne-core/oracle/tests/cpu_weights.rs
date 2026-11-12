//! The weights of a checkpoint: the tensor set the runtime asks for against the reference's own
//! listing, every way a checkpoint can differ from it, and the bytes an upload leaves in the
//! arena against a byte model. No GPU: the arena is on host memory.

mod common;

use common::{file_of, get, model_tensors, runtime_config, tiny, Rng, Tensor};
use x0ne_core_oracle::bf16;
use x0ne_core_oracle::model::Config;
use x0ne_core_runtime::arena::{Arena, Class, HostBacking, Slice};
use x0ne_core_runtime::config::{MoeConfig, Qwen3Config, Qwen3MoeConfig};
use x0ne_core_runtime::st::SliceSource;
use x0ne_core_runtime::{weights, Error};

fn random_config(rng: &mut Rng) -> Config {
    let kv = rng.range(1, 3);
    Config {
        hidden_size: 8 * rng.range(1, 6),
        intermediate_size: 8 * rng.range(1, 6),
        num_hidden_layers: rng.range(1, 4),
        num_attention_heads: kv * rng.range(1, 3),
        num_key_value_heads: kv,
        head_dim: 2 * rng.range(1, 8),
        vocab_size: rng.range(1, 40),
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        tie_word_embeddings: rng.chance(2),
    }
}

fn check(c: &Config, tensors: &[Tensor]) -> Result<(), Error> {
    let file = file_of(tensors);
    let src = SliceSource::new(None, &[("model.safetensors", &file)])?;
    weights::check(&runtime_config(c), &src)
}

/// The runtime's list of tensors and shapes is the reference's, written apart from it.
#[test]
fn the_expected_tensor_set_is_the_reference_listing() {
    for seed in 0..200 {
        let c = random_config(&mut Rng::new(seed));
        let rc = runtime_config(&c);
        assert_eq!(weights::expected(&rc), c.tensors(), "seed {seed}: {c:?}");
        let bytes: usize = c
            .tensors()
            .iter()
            .map(|(_, s)| s.iter().product::<usize>() * 2)
            .sum();
        assert_eq!(weights::resident_bytes(&rc), bytes as u64);
        assert_eq!(
            (rc.q_dim(), rc.kv_dim()),
            (c.heads().q_dim(), c.heads().kv_dim())
        );
        assert_eq!(rc.qkv_rows(), c.heads().q_dim() + 2 * c.heads().kv_dim());
    }
}

/// Every way a checkpoint differs from its config is refused: a tensor missing, one too many,
/// another dtype over the same bytes, another shape over the same bytes.
#[test]
fn a_checkpoint_that_differs_from_its_config_is_refused() {
    for seed in 0..40 {
        let mut rng = Rng::new(100 + seed);
        let c = random_config(&mut rng);
        let good = model_tensors(&c, &mut rng);
        check(&c, &good).unwrap_or_else(|e| panic!("seed {seed}: the checkpoint itself: {e}"));
        for i in 0..good.len() {
            let name = &good[i].name;
            let mut t = good.clone();
            t.remove(i);
            assert!(check(&c, &t).is_err(), "seed {seed}: without {name}");

            for dtype in ["F16", "I16", "U16"] {
                let mut t = good.clone();
                t[i].dtype = dtype.into();
                assert!(
                    matches!(check(&c, &t), Err(Error::Format(_))),
                    "seed {seed}: {name} as {dtype}"
                );
            }

            let shape = &good[i].shape;
            let n: usize = shape.iter().product();
            let mut others = vec![
                vec![n],
                vec![1, n],
                vec![n, 1],
                shape.iter().rev().copied().collect(),
            ];
            if shape.len() == 2 {
                others.push(vec![shape[0], shape[1], 1]);
                others.push(vec![1, shape[0], shape[1]]);
            } else {
                others.push(vec![]);
            }
            for other in others
                .into_iter()
                .filter(|o| o != shape && o.iter().product::<usize>() == n)
            {
                let mut t = good.clone();
                t[i].shape = other.clone();
                assert!(check(&c, &t).is_err(), "seed {seed}: {name} as {other:?}");
            }
        }
        for extra in [
            "model.layers.0.self_attn.q_proj.bias",
            "extra",
            "model.layers.99.mlp.up_proj.weight",
            "",
        ] {
            let mut t = good.clone();
            t.push(Tensor {
                name: extra.into(),
                dtype: "BF16".into(),
                shape: vec![2],
                data: vec![0; 4],
            });
            assert!(
                matches!(check(&c, &t), Err(Error::Format(_))),
                "seed {seed}: with {extra:?}"
            );
        }
        // The layer count is part of the shape of a checkpoint.
        let mut more = c.clone();
        more.num_hidden_layers += 1;
        assert!(check(&more, &good).is_err());
        if c.num_hidden_layers > 1 {
            let mut fewer = c.clone();
            fewer.num_hidden_layers -= 1;
            assert!(check(&fewer, &good).is_err());
        }
    }
}

/// A tied checkpoint may carry its LM head only when it is the embedding, byte for byte; an
/// untied one must carry it.
#[test]
fn a_tied_checkpoint_carries_no_other_lm_head_than_its_embedding() {
    for seed in 0..40 {
        let mut rng = Rng::new(300 + seed);
        let mut c = random_config(&mut rng);
        c.tie_word_embeddings = true;
        let good = model_tensors(&c, &mut rng);
        check(&c, &good).expect("tied, no LM head");
        let embed = get(&good, weights::EMBED).clone();
        let head = |data: Vec<u8>, shape: Vec<usize>, dtype: &str| {
            let mut t = good.clone();
            t.push(Tensor {
                name: weights::LM_HEAD.into(),
                dtype: dtype.into(),
                shape,
                data,
            });
            check(&c, &t)
        };
        head(embed.data.clone(), embed.shape.clone(), "BF16")
            .expect("tied, the LM head is the embedding");
        for bit in [0, embed.data.len() * 8 - 1, rng.below(embed.data.len() * 8)] {
            let mut d = embed.data.clone();
            d[bit / 8] ^= 1 << (bit % 8);
            assert!(
                head(d, embed.shape.clone(), "BF16").is_err(),
                "seed {seed}: bit {bit} differs"
            );
        }
        assert!(head(embed.data.clone(), embed.shape.clone(), "F16").is_err());
        if embed.shape[0] != embed.shape[1] {
            assert!(head(
                embed.data.clone(),
                vec![embed.shape[1], embed.shape[0]],
                "BF16"
            )
            .is_err());
        }
        // The same tensors under an untied config: with the head they are a checkpoint, without
        // it they are not.
        let mut untied = c.clone();
        untied.tie_word_embeddings = false;
        assert!(
            check(&untied, &good).is_err(),
            "seed {seed}: untied without an LM head"
        );
        let mut with = good.clone();
        with.push(Tensor {
            name: weights::LM_HEAD.into(),
            ..embed.clone()
        });
        check(&untied, &with).expect("untied with an LM head");
    }
}

/// The concatenation of the bytes of `names`.
fn cat(tensors: &[Tensor], names: &[String]) -> Vec<u8> {
    names
        .iter()
        .flat_map(|n| get(tensors, n).data.clone())
        .collect()
}

/// After an upload on host memory: every slice holds its tensors' bytes in the documented
/// order ([q; k; v], [gate; up]), the bytes written are the resident bytes, every weight byte
/// was written, and nothing outside the weight regions was.
#[test]
fn an_upload_leaves_exactly_the_checkpoint_in_the_arena() {
    for seed in 0..60 {
        let mut rng = Rng::new(500 + seed);
        let c = random_config(&mut rng);
        let rc = runtime_config(&c);
        // Random bytes, every bit pattern: an upload copies, it does not interpret.
        let tensors: Vec<Tensor> = c
            .tensors()
            .into_iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                Tensor {
                    name,
                    dtype: "BF16".into(),
                    shape,
                    data: rng.bytes(n * 2),
                }
            })
            .collect();
        let file = file_of(&tensors);
        let src = SliceSource::new(None, &[("m", &file)]).expect("source");

        let mut a = Arena::new();
        let before = a
            .alloc("client before", rng.range(1, 600) as u64, Class::ClientData)
            .unwrap();
        let ws = weights::plan(&rc, &mut a).expect("plan");
        let after = a
            .alloc("client after", rng.range(1, 600) as u64, Class::ClientData)
            .unwrap();
        assert_eq!(ws.layers.len(), c.num_hidden_layers);
        assert_eq!(ws.lm_head.is_none(), c.tie_word_embeddings);
        let planned: u64 = a
            .regions()
            .iter()
            .filter(|r| r.class == Class::Weights)
            .map(|r| r.len)
            .sum();
        assert_eq!(
            planned,
            weights::resident_bytes(&rc),
            "seed {seed}: bytes planned"
        );

        let mut f = a.freeze(HostBacking::allocate).expect("freeze");
        let written = weights::upload(&rc, &src, &ws, &mut f).expect("upload");
        assert_eq!(
            written,
            weights::resident_bytes(&rc),
            "seed {seed}: bytes written"
        );

        let read = |s: Slice| {
            let mut out = vec![0u8; s.len() as usize];
            f.read(s, 0, &mut out).expect("read");
            out
        };
        let one = |n: &str| cat(&tensors, &[n.to_owned()]);
        assert_eq!(read(ws.embed), one(weights::EMBED));
        assert_eq!(read(ws.final_norm), one(weights::FINAL_NORM));
        if let Some(h) = ws.lm_head {
            assert_eq!(read(h), one(weights::LM_HEAD));
        }
        for (l, s) in ws.layers.iter().enumerate() {
            let n = |leaf: &str| format!("model.layers.{l}.{leaf}.weight");
            assert_eq!(read(s.input_norm), one(&n("input_layernorm")));
            assert_eq!(
                read(s.qkv),
                cat(
                    &tensors,
                    &[
                        n("self_attn.q_proj"),
                        n("self_attn.k_proj"),
                        n("self_attn.v_proj")
                    ]
                ),
                "seed {seed}: layer {l}: [q; k; v]"
            );
            assert_eq!(read(s.q_norm), one(&n("self_attn.q_norm")));
            assert_eq!(read(s.k_norm), one(&n("self_attn.k_norm")));
            assert_eq!(read(s.o), one(&n("self_attn.o_proj")));
            assert_eq!(read(s.post_norm), one(&n("post_attention_layernorm")));
            assert_eq!(
                read(s.gate_up),
                cat(&tensors, &[n("mlp.gate_proj"), n("mlp.up_proj")]),
                "seed {seed}: layer {l}: [gate; up]"
            );
            assert_eq!(read(s.down), one(&n("mlp.down_proj")));
            assert!(s.router.is_none());
        }
        // The client regions and the padding still hold the poison of a new host backing.
        assert!(
            read(before).iter().chain(&read(after)).all(|&b| b == 0xA5),
            "seed {seed}: client bytes written"
        );
        let mem = f.backing().bytes();
        let mut inside = vec![false; mem.len()];
        for r in f.regions() {
            inside[r.offset as usize..(r.offset + r.len) as usize].fill(true);
        }
        assert!(
            mem.iter().zip(&inside).all(|(&b, &i)| i || b == 0xA5),
            "seed {seed}: padding written"
        );
        // A scrub leaves the weights.
        f.scrub_client_data().expect("scrub");
        let mut out = vec![0u8; ws.embed.len() as usize];
        f.read(ws.embed, 0, &mut out).unwrap();
        assert_eq!(out, one(weights::EMBED));
    }
}

/// An upload checks the checkpoint first: a refused one writes nothing. Slices planned for
/// another config are refused, not overrun.
#[test]
fn a_refused_upload_writes_nothing() {
    let mut rng = Rng::new(900);
    let c = tiny(2, 4, 2, false);
    let rc = runtime_config(&c);
    let good = model_tensors(&c, &mut rng);
    let mut bad = good.clone();
    bad.pop();
    let file = file_of(&bad);
    let src = SliceSource::new(None, &[("m", &file)]).unwrap();
    let mut a = Arena::new();
    let ws = weights::plan(&rc, &mut a).unwrap();
    let mut f = a.freeze(HostBacking::allocate).unwrap();
    assert!(matches!(
        weights::upload(&rc, &src, &ws, &mut f),
        Err(Error::Format(_))
    ));
    assert!(
        f.backing().bytes().iter().all(|&b| b == 0xA5),
        "a refused checkpoint was partly written"
    );

    // The plan of a smaller model, given the checkpoint of this one.
    let small = tiny(1, 2, 2, false);
    let file = file_of(&good);
    let src = SliceSource::new(None, &[("m", &file)]).unwrap();
    let mut a = Arena::new();
    let ws_small = weights::plan(&runtime_config(&small), &mut a).unwrap();
    let mut f = a.freeze(HostBacking::allocate).unwrap();
    assert!(
        weights::upload(&rc, &src, &ws_small, &mut f).is_err(),
        "slices of another config"
    );
    // The plan of another arena.
    let mut a = Arena::new();
    weights::plan(&rc, &mut a).unwrap();
    let mut f = a.freeze(HostBacking::allocate).unwrap();
    assert!(matches!(
        weights::upload(&rc, &src, &ws, &mut f),
        Err(Error::Arena(_))
    ));
}

fn moe_config(rng: &mut Rng) -> Qwen3MoeConfig {
    let base = runtime_config(&random_config(rng));
    let experts = rng.range(1, 5);
    Qwen3MoeConfig {
        base,
        moe: MoeConfig {
            num_experts: experts,
            num_experts_per_tok: rng.range(1, experts),
            moe_intermediate_size: 8 * rng.range(1, 3),
        },
    }
}

/// Qwen3-MoE: the tensor set written out here, and the experts packed by index in the arena.
#[test]
fn a_moe_checkpoint_is_checked_and_packed_by_expert() {
    for seed in 0..40 {
        let mut rng = Rng::new(700 + seed);
        let c = moe_config(&mut rng);
        let (b, m) = (&c.base, &c.moe);
        let (h, i, n) = (b.hidden_size, m.moe_intermediate_size, m.num_experts);
        // The listing, leaf by leaf: Qwen3's attention and norms, a router, n experts.
        let mut want = vec![(weights::EMBED.to_owned(), vec![b.vocab_size, h])];
        for l in 0..b.num_hidden_layers {
            let p = format!("model.layers.{l}");
            want.push((format!("{p}.input_layernorm.weight"), vec![h]));
            want.push((format!("{p}.self_attn.q_proj.weight"), vec![b.q_dim(), h]));
            want.push((format!("{p}.self_attn.k_proj.weight"), vec![b.kv_dim(), h]));
            want.push((format!("{p}.self_attn.v_proj.weight"), vec![b.kv_dim(), h]));
            want.push((format!("{p}.self_attn.q_norm.weight"), vec![b.head_dim]));
            want.push((format!("{p}.self_attn.k_norm.weight"), vec![b.head_dim]));
            want.push((format!("{p}.self_attn.o_proj.weight"), vec![h, b.q_dim()]));
            want.push((format!("{p}.post_attention_layernorm.weight"), vec![h]));
            want.push((format!("{p}.mlp.gate.weight"), vec![n, h]));
            for e in 0..n {
                want.push((format!("{p}.mlp.experts.{e}.gate_proj.weight"), vec![i, h]));
                want.push((format!("{p}.mlp.experts.{e}.up_proj.weight"), vec![i, h]));
                want.push((format!("{p}.mlp.experts.{e}.down_proj.weight"), vec![h, i]));
            }
        }
        want.push((weights::FINAL_NORM.to_owned(), vec![h]));
        if !b.tie_word_embeddings {
            want.push((weights::LM_HEAD.to_owned(), vec![b.vocab_size, h]));
        }
        assert_eq!(weights::moe_expected(&c), want, "seed {seed}");

        let tensors: Vec<Tensor> = want
            .iter()
            .map(|(name, shape)| Tensor {
                name: name.clone(),
                dtype: "BF16".into(),
                shape: shape.clone(),
                data: rng.bytes(shape.iter().product::<usize>() * 2),
            })
            .collect();
        let file = file_of(&tensors);
        let src = SliceSource::new(None, &[("m", &file)]).unwrap();
        weights::check_moe(&c, &src).expect("the checkpoint itself");
        // A dense config does not accept it, and it does not accept one expert fewer.
        assert!(weights::check(b, &src).is_err());
        let drop_at = rng.below(tensors.len());
        let mut fewer = tensors.clone();
        fewer.remove(drop_at);
        let short = file_of(&fewer);
        assert!(
            weights::check_moe(&c, &SliceSource::new(None, &[("m", &short)]).unwrap()).is_err()
        );

        let mut a = Arena::new();
        let ws = weights::plan_moe(&c, &mut a).expect("plan");
        let mut f = a.freeze(HostBacking::allocate).unwrap();
        let written = weights::upload_moe(&c, &src, &ws, &mut f).expect("upload");
        assert_eq!(written, weights::resident_bytes_moe(&c));
        assert_eq!(
            written as usize,
            tensors.iter().map(|t| t.data.len()).sum::<usize>()
        );
        for (l, s) in ws.layers.iter().enumerate() {
            let x =
                |e: usize, leaf: &str| format!("model.layers.{l}.mlp.experts.{e}.{leaf}.weight");
            let read = |s: Slice| {
                let mut out = vec![0u8; s.len() as usize];
                f.read(s, 0, &mut out).unwrap();
                out
            };
            let gate_up: Vec<String> = (0..n)
                .flat_map(|e| [x(e, "gate_proj"), x(e, "up_proj")])
                .collect();
            let down: Vec<String> = (0..n).map(|e| x(e, "down_proj")).collect();
            assert_eq!(
                read(s.gate_up),
                cat(&tensors, &gate_up),
                "seed {seed}: layer {l}: experts' [gate; up]"
            );
            assert_eq!(
                read(s.down),
                cat(&tensors, &down),
                "seed {seed}: layer {l}: experts' down"
            );
            let router = s.router.expect("a MoE layer has a router");
            assert_eq!(
                read(router),
                cat(&tensors, &[format!("model.layers.{l}.mlp.gate.weight")])
            );
        }
    }
}

/// `bf16_to_f32` over every bit pattern against the reference's decoding, bit for bit
/// (a signaling NaN comes out quiet).
#[test]
fn bf16_bytes_decode_to_the_same_f32_for_every_pattern() {
    let bytes: Vec<u8> = (0..=u16::MAX).flat_map(|b| b.to_le_bytes()).collect();
    let (got, want) = (weights::bf16_to_f32(&bytes), bf16::from_bytes(&bytes));
    assert_eq!(got.len(), 1 << 16);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(w.to_bits(), (i as u32) << 16);
        if w.is_nan() {
            // `half` quiets a signaling NaN; the payload and the sign are kept.
            assert_eq!(g.to_bits(), w.to_bits() | 0x0040_0000, "pattern {i:#06x}");
        } else {
            assert_eq!(g.to_bits(), w.to_bits(), "pattern {i:#06x}");
        }
    }
    // A trailing odd byte is not a value.
    assert_eq!(weights::bf16_to_f32(&bytes[..5]).len(), 2);
    assert!(weights::bf16_to_f32(&[]).is_empty());
}

fn huge() -> Qwen3Config {
    Qwen3Config {
        hidden_size: 1 << 40,
        intermediate_size: 1 << 40,
        num_hidden_layers: 1,
        num_attention_heads: 1 << 40,
        num_key_value_heads: 1 << 40,
        head_dim: 128,
        vocab_size: u32::MAX as usize,
        max_position_embeddings: 1 << 20,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        tie_word_embeddings: false,
    }
}

/// Sizes whose products pass u64: a plan is refused, it neither panics nor wraps into a small
/// arena. `weights::plan` multiplied in usize without a check (`bf16_bytes`); fixed in x0ne
/// Core and copied on 2026-11-10.
#[test]
fn a_plan_whose_sizes_overflow_is_refused() {
    let r = std::panic::catch_unwind(|| {
        let mut a = Arena::new();
        weights::plan(&huge(), &mut a).map(|_| a.planned_bytes())
    });
    match r {
        Ok(Err(_)) => {}
        Ok(Ok(bytes)) => panic!("a plan of 2^80 parameters was made in {bytes} bytes"),
        Err(_) => panic!("weights::plan panicked instead of refusing"),
    }
}
