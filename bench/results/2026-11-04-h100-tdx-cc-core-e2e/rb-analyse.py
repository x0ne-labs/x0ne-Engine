#!/usr/bin/env python3
"""Reads rb-bench.py's streams.jsonl, counts the tokens of each delta with the model's own
tokenizer.json (the catalog's file, sha256 checked by the caller), and computes, per stream and per group:
  decode rate, span:  (tokens - tokens of the first content frame) / (t last content frame - t first content frame)
  decode rate, fit:   least-squares slope of the cumulative tokens against the frame times (content frames)
  first token:        t of the first content frame, from the request's send (client clock, relay and network in)
  frame gaps, tokens per frame, the padded tail (t final chunk - t last content frame), and for c > 1 the queue.
The token count of a delta is the difference of the counts of two prefixes of the answer (encoded without special
tokens), so the per-delta counts sum to the count of the whole text; that sum is set against the signed usage.
Times are the client's (time.monotonic in the client container): each includes the TD's 100 ms tick, the relay and
the network. Nothing is extrapolated; a stream with an error or no usage is listed and left out of the groups.
Output: <out>/frames.jsonl (per stream: the frames, time, tokens, text), <out>/summary.json, <out>/summary.txt;
last line ANALYSE-OK or ANALYSE-KO <why>.
Usage: rb-analyse.py <streams.jsonl> <tokenizer.json> <out dir>   (needs the tokenizers package on sys.path)"""
import json, os, statistics, sys

sys.path.insert(0, os.environ.get("RB_DEPS", "/deps"))
from tokenizers import Tokenizer  # noqa: E402


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def fit_slope(ts, ys):
    n = len(ts)
    if n < 3:
        return None
    mt, my = sum(ts) / n, sum(ys) / n
    sxx = sum((t - mt) ** 2 for t in ts)
    if sxx <= 0:
        return None
    return sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / sxx


def padme(n):
    if n < 2:
        return n
    e = n.bit_length() - 1
    s = e.bit_length()
    mask = (1 << (e - s)) - 1
    return (n + mask) & ~mask


def r3(x):
    return None if x is None else round(x, 3)


def analyse(rec, tok):
    ev = rec.get("events") or []
    cont = [e for e in ev if e.get("kind") == "content" and e.get("content")]
    role = next((e for e in ev if e.get("kind") == "role"), None)
    final = next((e for e in ev if e.get("kind") == "final"), None)
    u = rec.get("usage") or {}
    fin = rec.get("final") or {}
    out = {k: rec.get(k) for k in ("name", "level", "rep", "c", "batch", "idx", "max_tokens", "t_start_utc",
                                    "t_send_since_barrier", "status", "error")}
    out.update(prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
               finish_reason=rec.get("finish_reason"), receipt_verified=fin.get("receipt_verified"),
               kv_cache_shared_between_requests=fin.get("kv_cache_shared_between_requests"),
               requests_batched_across_clients=fin.get("requests_batched_across_clients"),
               model_digest_hex=fin.get("model_digest_hex") or fin.get("receipt_model_digest_hex"),
               t_headers=r3(rec.get("t_headers")), t_role=r3(role["t"]) if role else None,
               t_final=r3(final["t"]) if final else None, t_eof=r3(rec.get("t_eof")),
               kinds={k: sum(1 for e in ev if e.get("kind") == k) for k in sorted({e.get("kind") for e in ev})})
    frames, prefix, before = [], "", 0
    if role and role.get("content"):
        prefix = role["content"]
        before = len(tok.encode(prefix, add_special_tokens=False).ids)
    for e in cont:
        prefix += e["content"]
        now = len(tok.encode(prefix, add_special_tokens=False).ids)
        frames.append({"t": e["t"], "tokens": now - before, "chars": len(e["content"]), "text": e["content"]})
        before = now
    out["frames"] = frames
    out["content_frames"] = len(frames)
    out["tokens_counted"] = sum(f["tokens"] for f in frames)
    if not frames:
        return out
    t = [f["t"] for f in frames]
    k = [f["tokens"] for f in frames]
    out["t_first_content"] = r3(t[0])
    out["t_last_content"] = r3(t[-1])
    out["tail_padded_s"] = r3(final["t"] - t[-1]) if final else None
    if role and final:
        ticks = round((final["t"] - role["t"]) / 0.1) + 1
        out["ticks_role_to_final"] = ticks
        out["ticks_is_padme_size"] = ticks >= 10 and padme(ticks) == ticks
    gaps = [(b - a) * 1000 for a, b in zip(t, t[1:])]
    if gaps:
        out["gap_ms"] = {"median": r3(q(gaps, 0.5)), "p10": r3(q(gaps, 0.1)), "p90": r3(q(gaps, 0.9)),
                         "min": r3(min(gaps)), "max": r3(max(gaps)), "over_150": sum(1 for g in gaps if g > 150)}
    out["tokens_per_frame"] = {"first": k[0], "median": q(k[1:], 0.5) if len(k) > 1 else None,
                               "min": min(k[1:]) if len(k) > 1 else None, "max": max(k[1:]) if len(k) > 1 else None}
    span = t[-1] - t[0]
    if span > 0:
        out["decode_tok_s_span"] = r3((out["tokens_counted"] - k[0]) / span)
        if out["completion_tokens"]:
            out["decode_tok_s_span_usage"] = r3((out["completion_tokens"] - k[0]) / span)
    cum, run = [], 0
    for x in k:
        run += x
        cum.append(run)
    s = fit_slope(t[1:], cum[1:])
    out["decode_tok_s_fit"] = r3(s)
    if out["completion_tokens"]:
        out["tokens_counted_minus_usage"] = out["tokens_counted"] - out["completion_tokens"]
    return out


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "median": r3(statistics.median(xs)), "min": r3(min(xs)), "max": r3(max(xs))}


def main():
    src, tokf, od = sys.argv[1:4]
    tok = Tokenizer.from_file(tokf)
    os.makedirs(od, exist_ok=True)
    recs = [json.loads(line) for line in open(src) if line.strip()]
    res = [analyse(r, tok) for r in recs]
    with open(os.path.join(od, "frames.jsonl"), "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    good = lambda r: r.get("status") == 200 and not r.get("error") and r.get("completion_tokens") and r.get("frames")
    groups = {}
    for r in res:
        if r["name"] == "warm" or not good(r):
            continue
        key = f"{r['level']} c=1" if r.get("c") == 1 else f"{r['level']} c={r['c']}"
        groups.setdefault(key, []).append(r)
    summ = {"streams": [{k: v for k, v in r.items() if k != "frames"} for r in res], "groups": {}}
    for key, rs in groups.items():
        g = {"n": len(rs), "prompt_tokens": sorted({r["prompt_tokens"] for r in rs}),
             "completion_tokens": sorted({r["completion_tokens"] for r in rs}),
             "finish_reason": sorted({str(r["finish_reason"]) for r in rs}),
             "decode_tok_s_span": stats([r.get("decode_tok_s_span") for r in rs]),
             "decode_tok_s_span_usage": stats([r.get("decode_tok_s_span_usage") for r in rs]),
             "decode_tok_s_fit": stats([r.get("decode_tok_s_fit") for r in rs]),
             "t_first_content_s": stats([r.get("t_first_content") for r in rs]),
             "t_final_s": stats([r.get("t_final") for r in rs]),
             "tail_padded_s": stats([r.get("tail_padded_s") for r in rs]),
             "gap_median_ms": stats([(r.get("gap_ms") or {}).get("median") for r in rs]),
             "tokens_per_frame_median": stats([(r.get("tokens_per_frame") or {}).get("median") for r in rs]),
             "tokens_counted_minus_usage": sorted({r.get("tokens_counted_minus_usage") for r in rs}, key=str),
             "receipt_verified": sorted({str(r.get("receipt_verified")) for r in rs}),
             "kv_cache_shared_between_requests": sorted({str(r.get("kv_cache_shared_between_requests")) for r in rs})}
        if rs[0].get("c", 1) > 1:
            by_batch = {}
            for r in rs:
                by_batch.setdefault(r.get("batch"), []).append(r)
            bl = []
            for b, members in by_batch.items():
                ends = [(m.get("t_send_since_barrier") or 0) + (m.get("t_final") or 0) for m in members]
                firsts = sorted((m.get("t_send_since_barrier") or 0) + (m.get("t_first_content") or 0) for m in members)
                toks = sum(m["completion_tokens"] for m in members)
                bl.append({"batch": b, "streams": len(members), "first_content_since_barrier_s": [r3(x) for x in firsts],
                           "last_final_since_barrier_s": r3(max(ends)), "tokens": toks,
                           "aggregate_tok_s": r3(toks / max(ends)) if max(ends) > 0 else None})
            g["batches"] = bl
        summ["groups"][key] = g
    json.dump(summ, open(os.path.join(od, "summary.json"), "w"), indent=1, ensure_ascii=False)
    lines = []
    for r in res:
        lines.append(f"{r['name']:<10} HTTP {r.get('status')} prompt {r.get('prompt_tokens')} out {r.get('completion_tokens')} "
                     f"({r.get('finish_reason')}) counted {r.get('tokens_counted')} frames {r.get('content_frames')} "
                     f"first {r.get('t_first_content')} s last {r.get('t_last_content')} s final {r.get('t_final')} s "
                     f"span {r.get('decode_tok_s_span')} fit {r.get('decode_tok_s_fit')} tok/s "
                     f"gap med {(r.get('gap_ms') or {}).get('median')} ms"
                     + (f" ERROR {r['error']}" if r.get("error") else ""))
    lines.append("")
    for key, g in summ["groups"].items():
        lines.append(f"{key}: n={g['n']} prompt {g['prompt_tokens']} out {g['completion_tokens']} "
                     f"decode span {g['decode_tok_s_span']} fit {g['decode_tok_s_fit']} "
                     f"first {g['t_first_content_s']} tail {g['tail_padded_s']} gap {g['gap_median_ms']}")
        for b in g.get("batches", []):
            lines.append(f"   batch {b['batch']}: first contents {b['first_content_since_barrier_s']} s, "
                         f"last final {b['last_final_since_barrier_s']} s, {b['tokens']} tokens, "
                         f"aggregate {b['aggregate_tok_s']} tok/s")
    open(os.path.join(od, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    c1 = [r for k, rs in groups.items() if k.endswith("c=1") for r in rs]
    if not c1:
        print("ANALYSE-KO no c=1 stream with usage and frames")
        sys.exit(1)
    print("ANALYSE-OK")


if __name__ == "__main__":
    main()
