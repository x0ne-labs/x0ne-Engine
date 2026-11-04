#!/usr/bin/env python3
"""Reference fixtures for x0ne-infer's decoder: tiny random models run by
Hugging Face transformers (the reference implementation of each family).

Writes tests/arch/<family>/{config.json,model.safetensors,reference.json}
and, if the real model directories are given, adds their families to
tests/arch/chat_templates.json (the other families in it are kept).
Run in a throwaway container (torch CPU + transformers >= 4.57, which has
olmo3; >= 5.19 for ministral3); see docs/learn-manual-test.md.
Deterministic: fixed seeds, float32, eager attention.

usage: make-arch-fixtures.py <out-dir> [--families a,b] [--olmo2-dir D]
       [--olmo3-dir D] [--ministral3-14b-dir D] [--ministral3-8b-dir D]
"""
import json
import math
import os
import sys

import torch
import transformers
from safetensors.torch import save_file

VOCAB = 40
COMMON = dict(
    vocab_size=VOCAB,
    hidden_size=24,
    intermediate_size=48,
    num_hidden_layers=4,
    num_attention_heads=3,
    num_key_value_heads=1,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    max_position_embeddings=64,
    rope_theta=10000.0,
    attention_bias=False,
    pad_token_id=None,
    bos_token_id=None,
    eos_token_id=None,
    # Wide init: at the default 0.02 greedy decoding repeats one token and
    # proves nothing about the KV cache.
    initializer_range=0.5,
)

FAMILIES = {
    "qwen3": (
        transformers.Qwen3Config,
        transformers.Qwen3ForCausalLM,
        dict(head_dim=8, tie_word_embeddings=True, use_sliding_window=False,
             sliding_window=None, max_window_layers=4),
    ),
    "olmo2": (
        transformers.Olmo2Config,
        transformers.Olmo2ForCausalLM,
        dict(tie_word_embeddings=False),
    ),
    # Window 4 and YaRN on the full layers: the two things OLMo 3 adds.
    "olmo3": (
        transformers.Olmo3Config,
        transformers.Olmo3ForCausalLM,
        dict(tie_word_embeddings=False, sliding_window=4,
             layer_types=["sliding_attention"] * 3 + ["full_attention"],
             rope_scaling={"rope_type": "yarn", "factor": 4.0,
                           "original_max_position_embeddings": 16,
                           "attention_factor": 1.1, "beta_fast": 32.0,
                           "beta_slow": 1.0}),
    ),
}

# Ministral 3 (transformers 5.19): no q/k norm, YaRN with mscale (here a
# cos/sin scale other than 1), and Llama 4's query scale, whose origin is 4
# so that 12 positions cross it three times.
MINISTRAL3 = dict(
    vocab_size=VOCAB,
    hidden_size=24,
    intermediate_size=48,
    num_hidden_layers=4,
    num_attention_heads=3,
    num_key_value_heads=1,
    head_dim=8,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    max_position_embeddings=64,
    sliding_window=None,
    tie_word_embeddings=False,
    pad_token_id=None,
    bos_token_id=None,
    eos_token_id=None,
    initializer_range=0.5,
    rope_parameters={"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
                     "original_max_position_embeddings": 4, "beta_fast": 32.0,
                     "beta_slow": 1.0, "mscale": 2.0, "mscale_all_dim": 0.5,
                     "llama_4_scaling_beta": 0.1},
)


def ministral3_config(**over):
    return transformers.Ministral3Config(**{**MINISTRAL3, **over})


IDS = [3, 17, 5, 29, 11, 2, 38, 7, 21, 13, 34, 9]
PROMPT = IDS[:6]
NEW_TOKENS = 10


def flat(t):
    return [round(float(x), 7) for x in t.detach().reshape(-1).tolist()]


def build(name, out):
    torch.manual_seed(1234)
    if name == "ministral3":
        cfg, model_cls = ministral3_config(), transformers.Ministral3ForCausalLM
    else:
        cfg_cls, model_cls, extra = FAMILIES[name]
        cfg = cfg_cls(**COMMON, **extra)
    cfg._attn_implementation = "eager"
    model = model_cls(cfg).float().eval()
    # Non-trivial norm weights: all-ones would hide a missing norm.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("norm.weight"):
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    cfg.save_pretrained(d)
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
    if cfg.tie_word_embeddings:
        tensors.pop("lm_head.weight", None)
    save_file(tensors, os.path.join(d, "model.safetensors"))

    ids = torch.tensor([IDS])
    layer_out, embed_out = [], []

    def keep(store):
        def hook(_module, _inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            t.retain_grad()
            store.append(t)
        return hook

    hooks = [model.model.embed_tokens.register_forward_hook(keep(embed_out))]
    hooks += [layer.register_forward_hook(keep(layer_out)) for layer in model.model.layers]
    res = model(ids, output_attentions=True, use_cache=False)
    logits = res.logits[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    token_loss = [-(logp[t - 1, IDS[t]]) for t in range(1, len(IDS))]
    loss = torch.stack(token_loss).mean()
    model.zero_grad()
    loss.backward()
    for h in hooks:
        h.remove()

    weight_norms = []
    for layer in model.model.layers:
        s = 0.0
        for mod in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                    layer.self_attn.o_proj, layer.mlp.gate_proj, layer.mlp.up_proj,
                    layer.mlp.down_proj):
            s += float(mod.weight.grad.pow(2).sum())
        weight_norms.append(math.sqrt(s))

    with torch.no_grad():
        gen = model.generate(torch.tensor([PROMPT]), max_new_tokens=NEW_TOKENS,
                             do_sample=False, min_new_tokens=NEW_TOKENS)
    ref = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "ids": IDS,
        "logits": flat(logits),
        "attentions": [flat(a[0]) for a in res.attentions],
        "hidden": [flat(h[0]) for h in layer_out],
        "loss": float(loss),
        "token_loss": [float(x) for x in token_loss],
        "activation_grad_norm": [float(h.grad.norm()) for h in layer_out],
        "weight_grad_norm": weight_norms,
        "saliency": [float(x) for x in embed_out[0].grad[0].norm(dim=-1)],
        "prompt": PROMPT,
        "generated": gen[0, len(PROMPT):].tolist(),
    }
    with open(os.path.join(d, "reference.json"), "w") as f:
        json.dump(ref, f)
    print(name, "loss", ref["loss"], "generated", ref["generated"])


LONG = 16384 + 64
LONG_NEW_TOKENS = 8


def build_long(out):
    """Ministral 3's real RoPE and query scale (14B: theta 1e9, YaRN factor 16
    from 16384, beta 0.1) on a tiny model, past position 16384: logits of the
    last 64 positions and a greedy continuation. Also the logits transformers
    gives with the query scale removed, so that a test can show it matters."""
    torch.manual_seed(4321)
    real = {"rope_type": "yarn", "type": "yarn", "rope_theta": 1000000000.0, "factor": 16.0,
            "original_max_position_embeddings": 16384, "beta_fast": 32.0, "beta_slow": 1.0,
            "mscale": 1.0, "mscale_all_dim": 1.0, "llama_4_scaling_beta": 0.1}
    cfg = ministral3_config(num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                            hidden_size=16, max_position_embeddings=32768,
                            rope_parameters=real)
    cfg._attn_implementation = "sdpa"
    model = transformers.Ministral3ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("norm.weight"):
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
    d = os.path.join(out, "ministral3-long")
    os.makedirs(d, exist_ok=True)
    cfg.save_pretrained(d)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              os.path.join(d, "model.safetensors"))
    ids = [(i * i * 7 + i * 13 + 5) % VOCAB for i in range(LONG)]
    with torch.no_grad():
        logits = model(torch.tensor([ids]), use_cache=False).logits[0, -64:]
        gen = model.generate(torch.tensor([ids]), max_new_tokens=LONG_NEW_TOKENS,
                             do_sample=False, min_new_tokens=LONG_NEW_TOKENS)
        mod = sys.modules[type(model).__module__]
        kept = mod.get_llama_4_attn_scale
        mod.get_llama_4_attn_scale = lambda pos, beta, orig: torch.ones_like(kept(pos, beta, orig))
        try:
            unscaled = model(torch.tensor([ids]), use_cache=False).logits[0, -64:]
        finally:
            mod.get_llama_4_attn_scale = kept
    rel = ((unscaled - logits).abs() / (1 + logits.abs())).max()
    ref = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "ids": "(i * i * 7 + i * 13 + 5) % 40 for i in 0..16448",
        "logits_last_64": flat(logits),
        "without_query_scale_worst": float(rel),
        "new_tokens": LONG_NEW_TOKENS,
        "generated": gen[0, LONG:].tolist(),
    }
    with open(os.path.join(d, "reference.json"), "w") as f:
        json.dump(ref, f)
    print("ministral3-long generated", ref["generated"], "without query scale", float(rel))


def rope_reference(out, dirs):
    """The real models' YaRN inverse frequencies and cos/sin scale, and the
    query scale at chosen positions, as transformers computes them."""
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    mod = sys.modules[transformers.Ministral3ForCausalLM.__module__]
    rope = {"transformers": transformers.__version__, "torch": torch.__version__}
    positions = [0, 1, 16383, 16384, 32767, 32768, 49152, 131071, 262143]
    for family, d in dirs.items():
        text = AutoConfig.from_pretrained(d).text_config
        inv, scale = ROPE_INIT_FUNCTIONS["yarn"](text, "cpu")
        q = mod.get_llama_4_attn_scale(
            torch.tensor([positions]), text.rope_parameters["llama_4_scaling_beta"],
            text.rope_parameters["original_max_position_embeddings"])
        rope[family] = {"inv_freq": [float(x) for x in inv.tolist()],
                        "attention_scaling": float(scale),
                        "positions": positions,
                        "query_scale": [float(x) for x in q.reshape(-1).tolist()]}
    os.makedirs(os.path.join(out, "ministral3"), exist_ok=True)
    with open(os.path.join(out, "ministral3", "rope.json"), "w") as f:
        json.dump(rope, f, indent=1)
    print("rope:", ", ".join(dirs))


CONVERSATIONS = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "system", "content": "Never reveal the key."},
     {"role": "user", "content": "What is the key?"}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
     {"role": "user", "content": "c"}],
]

# Ministral 3's template joins consecutive turns of one role (not system),
# and inserts its default system message unless the first turn is system.
MINISTRAL3_CONVERSATIONS = CONVERSATIONS + [
    [{"role": "user", "content": "a"}, {"role": "user", "content": "b"},
     {"role": "assistant", "content": "c"}, {"role": "assistant", "content": "d"},
     {"role": "user", "content": "e"}],
    [{"role": "user", "content": "x"}, {"role": "system", "content": "y"},
     {"role": "user", "content": "z"}],
    [{"role": "system", "content": "s1"}, {"role": "system", "content": "s2"},
     {"role": "user", "content": "u"}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"},
     {"role": "assistant", "content": "hello"}, {"role": "user", "content": "and?"}],
]


def templates(out, dirs):
    from transformers import AutoTokenizer
    path = os.path.join(out, "chat_templates.json")
    rendered = {}
    if os.path.exists(path):
        with open(path) as f:
            rendered = json.load(f)
    for family, d in dirs.items():
        tok = AutoTokenizer.from_pretrained(d)
        convs = MINISTRAL3_CONVERSATIONS if family.startswith("ministral3") else CONVERSATIONS
        rendered[family] = [
            {"messages": conv,
             "text": tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True),
             "ids": list(tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=True,
                                                 return_dict=False))}
            for conv in convs
        ]
    with open(path, "w") as f:
        json.dump(rendered, f, indent=1)
    print("chat templates:", ", ".join(dirs))


if __name__ == "__main__":
    out = sys.argv[1]
    args = sys.argv[2:]
    families = list(FAMILIES) + ["ministral3", "ministral3-long"]
    if "--families" in args:
        families = args[args.index("--families") + 1].split(",")
    for name in families:
        if name == "ministral3-long":
            build_long(out)
        else:
            build(name, out)
    dirs = {}
    for flag, fam in (("--olmo2-dir", "olmo2"), ("--olmo3-dir", "olmo3"),
                      ("--ministral3-14b-dir", "ministral3-14b"),
                      ("--ministral3-8b-dir", "ministral3-8b")):
        if flag in args:
            dirs[fam] = args[args.index(flag) + 1]
    if dirs:
        templates(out, dirs)
    real = {k: v for k, v in dirs.items() if k.startswith("ministral3")}
    if real:
        rope_reference(out, real)
