#!/usr/bin/env python3
"""x0ne Core alone on an H100, Intel TDX and the GPU in CC mode, 2026-11-04: the summary of x0ne-core-itl's JSON lines.

Standard library only. Per model and prompt length, warm-ups excluded:
  n             measured runs (the bench ran one warm-up and 5 runs per length);
  first token   times[0], from the call to the first token back on the host, prefill included;
  prefill       prompt tokens / mean first token;
  decode        (generated - 1) / (times[-1] - times[0]), per run;
  ITL           every gap times[i + 1] - times[i] of every measured run, pooled: p50, p90, p99 (linear interpolation);
  last token    times[-1], from the call to the last token back on the host.
Means come with the half-width of a 95 % interval over runs (Student's t, n - 1 degrees of freedom).
summary.json keeps the values unrounded (the same bytes under Python 3.11, 3.12 and 3.13, checked 2026-11-04);
summary.txt rounds them for reading.
The rows are refused unless every measured run has its generated count of times, non-decreasing, and its label
"in<prompt tokens>".

Usage: core-summary.py OUT_DIR core-<model>-ccon.jsonl...   writes OUT_DIR/summary.json and OUT_DIR/summary.txt
"""
import hashlib
import json
import math
import os
import re
import statistics as st
import sys

# Two-sided 97.5 % quantiles of Student's t by degrees of freedom.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
# The bench's tags and the Engine catalog's names (x0ne-infer/catalog/models.toml), smallest first.
MODELS = {"q06": "qwen3:0.6b", "q8b": "qwen3:8b", "q14b": "qwen3:14b", "q32b": "qwen3:32b"}


def die(msg):
    sys.exit(f"core-summary.py: {msg}")


def quantile(v, p):
    v = sorted(v)
    k = (len(v) - 1) * p
    f = math.floor(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def mean_ci(v):
    m = st.mean(v)
    if len(v) < 2:
        return m, float("nan")
    return m, T975.get(len(v) - 1, 1.96) * st.stdev(v) / math.sqrt(len(v))


def stat(v):
    m, c = mean_ci(v)
    return {"mean": m, "ci95": c, "min": min(v), "max": max(v)}


def read(path):
    load, runs, warm = None, {}, {}
    with open(path, encoding="utf-8") as fh:
        for no, line in enumerate(fh, 1):
            d = json.loads(line)
            if d.get("engine") != "x0ne-core" or d.get("tool") != "x0ne-core-itl":
                die(f"{path}:{no}: not a line of x0ne-core-itl")
            if d.get("event") == "load":
                if load is not None:
                    die(f"{path}:{no}: a second load line")
                load = d
                continue
            t, n, L = d["times"], d["gen"], d["prompt_len"]
            if d["label"] != f"in{L}" or len(t) != n or n < 2 or any(b < a for a, b in zip(t, t[1:])) or t[-1] <= t[0]:
                die(f"{path}:{no}: a run without {n} non-decreasing times or with a label other than in{L}")
            if d["warmup"]:
                warm[L] = warm.get(L, 0) + 1
            else:
                runs.setdefault(L, []).append(d)
    if load is None:
        die(f"{path}: no load line")
    return load, runs, warm


def main():
    if len(sys.argv) < 3:
        die(__doc__.strip().splitlines()[-1].strip())
    out_dir, files = sys.argv[1], sys.argv[2:]
    models, rows = [], []
    for path in files:
        load, runs, warm = read(path)
        tag = load["tag"]
        if tag not in MODELS:
            die(f"{path}: unknown tag {tag}")
        desc = load["describe"]
        arena = re.search(r"arena (\d+) bytes", desc)
        nodes = re.search(r"decode graph (\d+) nodes", desc)
        models.append({
            "tag": tag, "model": MODELS[tag], "file": os.path.basename(path),
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "load_s": load["load_s"], "decode_mode": load["decode"], "describe": desc,
            "arena_bytes": int(arena[1]) if arena else None, "decode_graph_nodes": int(nodes[1]) if nodes else None,
        })
        for L in sorted(runs):
            rs = runs[L]
            gen = {r["gen"] for r in rs}
            if len(gen) != 1:
                die(f"{path}: in{L} mixes generated counts {sorted(gen)}")
            ttft = [r["times"][0] for r in rs]
            dec = [(len(r["times"]) - 1) / (r["times"][-1] - r["times"][0]) for r in rs]
            e2e = [r["times"][-1] for r in rs]
            itl = [1000 * (b - a) for r in rs for a, b in zip(r["times"], r["times"][1:])]
            rows.append({
                "tag": tag, "model": MODELS[tag], "prompt_tokens": L, "generated": gen.pop(), "n": len(rs),
                "warmups_excluded": warm.get(L, 0),
                "first_token_s": stat(ttft),
                "prefill_tok_s": L / st.mean(ttft),
                "decode_tok_s": stat(dec),
                "itl_ms": {"gaps": len(itl), "p50": quantile(itl, .5), "p90": quantile(itl, .9),
                           "p99": quantile(itl, .99), "min": min(itl), "max": max(itl)},
                "last_token_s": stat(e2e),
            })
    order = list(MODELS)
    models.sort(key=lambda m: order.index(m["tag"]))
    rows.sort(key=lambda r: (order.index(r["tag"]), r["prompt_tokens"]))
    summary = {
        "what": "x0ne Core alone, x0ne-core-itl in the process, one H100 80GB HBM3, Intel TDX guest, GPU in CC mode, "
                "2026-11-04; greedy, one request at a time, CUDA graph decode",
        "method": {
            "first_token_s": "times[0]: from the call to the first token back on the host, prefill included",
            "prefill_tok_s": "prompt tokens / mean first_token_s",
            "decode_tok_s": "(generated - 1) / (times[-1] - times[0]) per run",
            "itl_ms": "every gap between consecutive token times, all measured runs pooled; linear interpolation",
            "last_token_s": "times[-1]: from the call to the last token back on the host",
            "interval": "mean and the half-width of a 95 % interval over runs, Student's t with n - 1 degrees of freedom",
            "excluded": "the warm-up run of each prompt length (\"warmup\": true)",
        },
        "models": models,
        "rows": rows,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
        fh.write("\n")
    lines = [
        "x0ne Core alone (x0ne-core-itl), H100 80GB HBM3, Intel TDX guest, GPU in CC mode, 2026-11-04.",
        "Greedy, one request at a time, CUDA graph decode, warm-ups excluded.",
        "Mean +- half-width of a 95 % interval over the n runs (Student's t); ITL percentiles over every gap of the n runs.",
        "",
    ]
    hdr = f"{'prompt':>6} {'gen':>4} {'n':>2}  {'first token ms':>16}  {'prefill tok/s':>13}  {'decode tok/s':>16}  " \
          f"{'ITL p50':>8} {'p90':>8} {'p99 ms':>8}  {'last token s':>16}"
    for m in models:
        lines.append(f"{m['model']} ({m['tag']}, {m['file']}): loaded in {m['load_s']:.1f} s, arena "
                     f"{m['arena_bytes']:,} bytes, decode graph {m['decode_graph_nodes']} nodes")
        lines.append(hdr)
        for r in (r for r in rows if r["tag"] == m["tag"]):
            ft, d, e, i = r["first_token_s"], r["decode_tok_s"], r["last_token_s"], r["itl_ms"]
            lines.append(
                f"{r['prompt_tokens']:>6} {r['generated']:>4} {r['n']:>2}  "
                f"{1000 * ft['mean']:>8.2f} +- {1000 * ft['ci95']:<5.2f}  {r['prefill_tok_s']:>13,.0f}  "
                f"{d['mean']:>8.2f} +- {d['ci95']:<5.2f}  {i['p50']:>8.3f} {i['p90']:>8.3f} {i['p99']:>8.3f}  "
                f"{e['mean']:>8.3f} +- {e['ci95']:<5.3f}")
        lines.append("")
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
