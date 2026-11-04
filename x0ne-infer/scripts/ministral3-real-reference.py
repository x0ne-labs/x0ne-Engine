#!/usr/bin/env python3
"""Reference logits of a real Ministral 3 release, computed by transformers on
the CPU (--dtype bf16, f16 or f32), for x0ne-infer's ignored test
ministral3::tests::real_weights_match_transformers. --text-only loads the
text model alone (Ministral3ForCausalLM, the `language_model.` prefix
mapped away), as the engine does.

Prompts: three chat conversations rendered by the release's template (the
first without a system message, so with the default one), one raw text, and
optionally one raw text of --long tokens (past 16384 positions, where Llama
4's query scale acts). For each: the ids, the logits of the last 64
positions, and --greedy tokens of greedy decoding (no stop on </s>).

Writes one safetensors file; its metadata holds the conversations and the
timings. Never run this and the engine on a 14B together: each held about
30 GB on 2026-10-28 (BF16, then F16).

usage: ministral3-real-reference.py <model dir> <template dir> <out.safetensors>
       [--dtype bf16|f16|f32] [--text-only] [--then-f16 <out2>] [--long N]
       [--greedy N] [--threads N] [--extra <prompts.jsonl> <out.jsonl> [--extra-new N]]

--then-f16 casts the loaded model to F16 and writes a second reference:
one load gives two dtypes. --extra renders each conversation of a JSONL
file (id, messages) with the template and its default system message, and
writes greedy tokens up to </s> or N (default 64).
"""
import gc
import json
import os
import resource
import sys
import time

import torch
import transformers
from safetensors.torch import save_file

KEEP = 64
CONVERSATIONS = [
    [{"role": "user", "content": "Name the largest planet of the solar system."}],
    [
        {"role": "system", "content": "Answer in one short sentence."},
        {"role": "user", "content": "What is 17 + 25?"},
    ],
    [
        {"role": "system", "content": "Responde en español, en una frase."},
        {"role": "user", "content": "¿Cuál es la capital de Italia?"},
    ],
]
RAW = "The quick brown fox jumps over the lazy dog. The"


DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


def arg(name, default, kind=int):
    a = sys.argv
    return kind(a[a.index(name) + 1]) if name in a else default


def long_text(tok, n):
    lines, ids = [], []
    i = 0
    while len(ids) < n:
        lines.extend(f"Item {j}: the value of item {j} is {(j * 7919) % 1000}." for j in range(i, i + 500))
        i += 500
        ids = tok("\n".join(lines)).input_ids
    return ids[:n]


def chunked(model, ids, greedy):
    """A long prompt through the KV cache, 512 positions at a time (sdpa would
    otherwise build a 16k x 16k mask per head), then greedy steps."""
    x = torch.tensor([ids])
    cache = transformers.DynamicCache(config=model.config)
    end = len(ids) - KEEP
    for s in range(0, end, 512):
        model(input_ids=x[:, s:min(s + 512, end)], past_key_values=cache, use_cache=True)
    logits = model(input_ids=x[:, end:], past_key_values=cache, use_cache=True).logits[0]
    gen = [int(logits[-1].argmax())]
    for _ in range(greedy - 1):
        out = model(input_ids=torch.tensor([[gen[-1]]]), past_key_values=cache, use_cache=True)
        gen.append(int(out.logits[0, -1].argmax()))
    return logits.float(), torch.tensor(gen)


def run(model, tok, prompts, greedy, dtype, meta):
    tensors, meta = {}, dict(meta, dtype=dtype)
    with torch.inference_mode():
        for i, ids in enumerate(prompts):
            x = torch.tensor([ids])
            t0 = time.time()
            if len(ids) > 4096:
                logits, gen = chunked(model, ids, greedy)
                meta[f"p{i}.prefill_s"] = f"{time.time() - t0:.2f} (with the greedy steps)"
                meta[f"p{i}.generate_s"] = "in prefill_s"
                gen = torch.cat([x[0], gen]).unsqueeze(0)
            else:
                logits = model(input_ids=x, logits_to_keep=KEEP).logits[0].float()
                meta[f"p{i}.prefill_s"] = f"{time.time() - t0:.2f}"
                t0 = time.time()
                gen = model.generate(x, max_new_tokens=greedy, do_sample=False, eos_token_id=None,
                                     pad_token_id=11)
                meta[f"p{i}.generate_s"] = f"{time.time() - t0:.2f}"
            tensors[f"p{i}.ids"] = x[0]
            tensors[f"p{i}.logits"] = logits.contiguous()
            tensors[f"p{i}.greedy"] = gen[0, len(ids):].contiguous()
            print(f"{dtype} p{i}: {len(ids)} ids, prefill {meta[f'p{i}.prefill_s']} s, "
                  f"{greedy} tokens in {meta[f'p{i}.generate_s']} s: "
                  f"{tok.decode(gen[0, len(ids):])!r}", flush=True)
    meta["maxrss_kib"] = str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return tensors, meta


def extra(model, tok, src, out, new, dtype):
    """Each conversation of `src` (JSONL: id, messages) rendered by the
    release's template, default system message included, then greedy until
    </s> or `new` tokens."""
    eos = tok.convert_tokens_to_ids("</s>")
    with open(src) as f, open(out, "w") as w, torch.inference_mode():
        for line in f:
            case = json.loads(line)
            ids = list(tok.apply_chat_template(case["messages"], tokenize=True,
                                               add_generation_prompt=True, return_dict=False))
            t0 = time.time()
            gen = model.generate(torch.tensor([ids]), max_new_tokens=new, do_sample=False,
                                 eos_token_id=eos, pad_token_id=11)[0, len(ids):].tolist()
            row = {"id": case["id"], "prompt_ids": ids, "generated_ids": gen,
                   "text": tok.decode(gen, skip_special_tokens=True), "dtype": dtype,
                   "stopped_on_eos": bool(gen) and gen[-1] == eos,
                   "seconds": round(time.time() - t0, 1)}
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            w.flush()
            print(f"extra {case['id']}: {len(ids)} ids, {len(gen)} tokens in {row['seconds']} s: "
                  f"{row['text']!r}", flush=True)


def main():
    model_dir, tmpl_dir, out = sys.argv[1:4]
    long_n, greedy = arg("--long", 0), arg("--greedy", 32)
    torch.set_num_threads(arg("--threads", 12))
    tok = transformers.AutoTokenizer.from_pretrained(tmpl_dir)
    prompts = [
        list(tok.apply_chat_template(c, tokenize=True, add_generation_prompt=True, return_dict=False))
        for c in CONVERSATIONS
    ]
    prompts.append(tok(RAW).input_ids)
    if long_n:
        prompts.append(long_text(tok, long_n))
    t0 = time.time()
    dtype = arg("--dtype", "bf16", str)
    if "--text-only" in sys.argv:
        config = transformers.AutoConfig.from_pretrained(model_dir).text_config
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir, config=config, dtype=DTYPES[dtype], attn_implementation="sdpa",
            key_mapping={r"^language_model\.": ""},
        ).eval()
    else:
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=DTYPES[dtype], attn_implementation="sdpa"
        ).eval()
    meta = {"load_s": f"{time.time() - t0:.1f}", "transformers": transformers.__version__,
            "torch": torch.__version__, "threads": str(torch.get_num_threads()),
            "conversations": json.dumps(CONVERSATIONS, ensure_ascii=False), "raw": RAW,
            "class": type(model).__name__}
    tensors, m = run(model, tok, prompts, greedy, dtype, meta)
    save_file(tensors, out, metadata=m)
    print("maxrss", m["maxrss_kib"], "KiB", flush=True)
    if "--extra" in sys.argv:
        a = sys.argv.index("--extra")
        extra(model, tok, sys.argv[a + 1], sys.argv[a + 2], arg("--extra-new", 64), dtype)
    if "--then-f16" in sys.argv:
        # The bf16 or f32 weights cast once: the same rounding as candle's F16 load.
        # One parameter at a time, the old one freed: a 14B fits in 31 GB.
        with torch.no_grad():
            for module in model.modules():
                for name, param in list(module.named_parameters(recurse=False)):
                    module._parameters[name] = torch.nn.Parameter(
                        param.data.to(torch.float16), requires_grad=False)
                    del param
                for name, buf in list(module.named_buffers(recurse=False)):
                    if buf is not None and buf.is_floating_point() and name != "inv_freq":
                        module._buffers[name] = buf.to(torch.float16)
            gc.collect()
        model.config.dtype = torch.float16
        tensors, m = run(model, tok, prompts, greedy, "f16", meta)
        save_file(tensors, sys.argv[sys.argv.index("--then-f16") + 1], metadata=m)
        print("maxrss", m["maxrss_kib"], "KiB", flush=True)


if __name__ == "__main__":
    main()
