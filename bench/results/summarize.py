#!/usr/bin/env python3
"""Recompute every figure of docs/benchmarks.md from the rows in bench/results/.

Usage: python3 bench/results/summarize.py [results dir]   (standard library only)

Each <date>-h100-tdx-cc/ directory holds times.log ("<name>: <seconds> s,
HTTP <code>", curl's time_total through the relay) and responses.jsonl
(token counts, finish reason, receipt check). decode = (completion tokens -
1) / (t(N) - t(1)), on the medians of each request kind, except in PADDED
directories.
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
# The TD of these runs padded a whole answer to a Padme number of 100 ms ticks, ten at least:
# every t(N) - t(1) of their medians is a multiple of 100 ms within 0.012 s (2026-11-03).
# Their times give no decode rate, so none is printed.
PADDED = {"2026-10-28-h100-tdx-cc", "2026-10-31-h100-tdx-cc"}


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def cloud(directory):
    times = {}
    for line in (directory / "times.log").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([a-z0-9]+?)(?:-(\d+))?: ([\d.]+) s, HTTP (\d+)$", line)
        if m:
            times[(m.group(1), m.group(2) or "1")] = (float(m.group(3)), m.group(4))
    resp = {r["name"]: r for r in rows(directory / "responses.jsonl")}
    kinds = {}
    for (kind, i), (t, code) in times.items():
        r = resp.get(kind if kind == "warm" else f"{kind}-{i}")
        if r is None:
            continue
        kinds.setdefault(kind, []).append({"i": i, "t": t, "http": code, **r})
    print(f"\n## {directory.name}\n")
    model = sorted({r["model"] for rs in kinds.values() for r in rs})
    print(f"model {model}")
    print("\n| request | n | median s | min s | max s | prompt tokens | completion tokens | finish | receipts verified | device, environment |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    med = {}
    # The fixed names first, then a prompt ladder (k<N>p-i / k<N>-i, 2026-10-31) by prompt length.
    named = ["warm", "c1", "c256", "long4kp", "long4k"]
    ladder = sorted((k for k in kinds if k not in named),
                    key=lambda k: (st.median(r["prompt_tokens"] for r in kinds[k]), k.endswith("p"), k))
    for kind in named + ladder:
        rs = kinds.get(kind)
        if not rs:
            continue
        ts = [r["t"] for r in rs]
        med[kind] = {"t": st.median(ts), "c": st.median([r["completion_tokens"] for r in rs]),
                     "p": st.median([r["prompt_tokens"] for r in rs]), "rows": rs}
        env = sorted({(r["compute_device"], json.dumps(r["environment"], sort_keys=True)) for r in rs})
        print(f"| {kind} | {len(rs)} | {st.median(ts):.3f} | {min(ts):.3f} | {max(ts):.3f} "
              f"| {sorted({r['prompt_tokens'] for r in rs})} | {sorted({r['completion_tokens'] for r in rs})} "
              f"| {sorted({r['finish_reason'] for r in rs})} | {sum(1 for r in rs if r['receipt_verified'])}/{len(rs)} "
              f"| {env} |")
    print()

    def pairwise(a, b):
        by = {r["i"]: r for r in med[a]["rows"]}
        return [(r["completion_tokens"] - 1) / (r["t"] - by[r["i"]]["t"]) for r in med[b]["rows"] if r["i"] in by]

    padded = directory.name in PADDED
    if padded:
        print("decode: not derived, whole answers padded to 100 ms ticks in the TD (docs/benchmarks.md, Method)")
    if "c1" in med and "c256" in med and not padded:
        d = (med["c256"]["c"] - 1) / (med["c256"]["t"] - med["c1"]["t"])
        pw = pairwise("c1", "c256")
        print(f"decode, short prompt: {d:.2f} tok/s on the medians; pairwise [{min(pw):.2f}, {max(pw):.2f}] (n={len(pw)})")
        print(f"first token, relay and round trip included (median c1): {med['c1']['t']:.3f} s")
    if "long4kp" in med and "long4k" in med and not padded:
        d = (med["long4k"]["c"] - 1) / (med["long4k"]["t"] - med["long4kp"]["t"])
        pw = pairwise("long4kp", "long4k")
        print(f"decode at {med['long4k']['p']:.0f} prompt tokens: {d:.2f} tok/s on the medians, over "
              f"{med['long4k']['c'] - 1:.0f} tokens; pairwise [{min(pw):.2f}, {max(pw):.2f}] (n={len(pw)})")
        print(f"first token at {med['long4kp']['p']:.0f} prompt tokens, relay and round trip included "
              f"(median long4kp): {med['long4kp']['t']:.3f} s")
    elif "long4k" in med and not padded:
        print(f"decode at {med['long4k']['p']:.0f} prompt tokens: not isolated (no max_tokens 1 request at that length)")
    if "c1" in med and "long4kp" in med and not padded:
        dt = med["long4kp"]["t"] - med["c1"]["t"]
        print(f"prefill difference: {med['long4kp']['p'] - med['c1']['p']:.0f} more prompt tokens took {dt:.3f} s more")
    for one in (k for k in ladder if k.endswith("p") and k[:-1] in med and not padded):
        many = one[:-1]
        d = (med[many]["c"] - 1) / (med[many]["t"] - med[one]["t"])
        pw = pairwise(one, many)
        print(f"decode at {med[many]['p']:.0f} prompt tokens: {d:.2f} tok/s on the medians, over "
              f"{med[many]['c'] - 1:.0f} tokens ({1000 / d:.2f} ms per token); pairwise [{min(pw):.2f}, {max(pw):.2f}] (n={len(pw)}); "
              f"first token, relay and round trip included (median {one}): {med[one]['t']:.3f} s")
    k = directory / "kv-cache.jsonl"
    if k.is_file():
        kv = rows(k)
        print("\n| second turn | request | n | prompt tokens | first content, median s | min s | max s | turn | reused tokens | kv cache shared | receipts verified |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for level in sorted({r["level"] for r in kv}):
            for req in ("c1", "c2", "n1", "n2"):
                rs = [r for r in kv if r["level"] == level and r["request"] == req]
                if not rs:
                    continue
                ts = [r["t_first_content"] for r in rs]
                print(f"| {level} | {req} | {len(rs)} | {sorted({r['prompt_tokens'] for r in rs})} | {st.median(ts):.3f} "
                      f"| {min(ts):.3f} | {max(ts):.3f} | {sorted({r['turn'] for r in rs}, key=str)} "
                      f"| {sorted({r['reused_tokens'] for r in rs}, key=str)} "
                      f"| {sorted({r['kv_cache_shared_between_requests'] for r in rs})} "
                      f"| {sum(1 for r in rs if r['receipt_verified'])}/{len(rs)} |")
    s = directory / "streams.jsonl"
    if s.is_file():
        print("\n| stream | prompt tokens | completion tokens | first content s | end s | finish | receipt verified |")
        print("|---|---|---|---|---|---|---|")
        for r in rows(s):
            u = r.get("usage") or {}
            print(f"| {r['name']} | {u.get('prompt_tokens')} | {u.get('completion_tokens')} | {r['first_content']:.3f} "
                  f"| {r['end']:.3f} | {r['finish_reason']} | {r['receipt_verified']} |")


def main():
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        if d.name.endswith("-h100-tdx-cc"):
            cloud(d)


if __name__ == "__main__":
    main()
