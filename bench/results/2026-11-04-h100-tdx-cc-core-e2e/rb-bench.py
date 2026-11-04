#!/usr/bin/env python3
"""The bench client of the H100 TDX runs, against the relay's local API (127.0.0.1:8478) once `up`
serves. Standard library only (the client container's Python 3.13). Every SSE event of every stream is kept with the
time it was read (time.monotonic, seconds since the request was sent) and its delta text, so the decode rate is
computed afterwards from the frames themselves (rb-analyse.py counts the tokens of each delta with the model's
tokenizer). Nothing here interprets the timing: it records.

Plan, in order, each step gated on the time budget (--budget-s, from this script's start) and on the session's
seconds_left (GET /x0ne/v1/status, kept whole at the start and the end):
  warm      one stream, 16 tokens, not counted
  then --reps rounds (7), each one stream per level of --levels in order, max_tokens 1024:
  p535      the essay instruction alone (58 prompt tokens with the chat template of qwen3:8b, measured 2026-11-04)
  p1k       the same after a list of 480 words (548 prompt tokens)
  p4k       the same after a list of 3500 words (3,568 prompt tokens)
  p8k       the same after a list of 7400 words (7,468 prompt tokens)
  p16k      the same after a list of 16000 words (16,068 prompt tokens)
  The level names are not token counts: they come from an earlier run with another model's template.
  c2, c4    2 then 4 streams of p535 sent together (one barrier), only if --batches names them (none by default)
  c2b, c4b  a second batch of each, only if the budget leaves room
Output: <out>/streams.jsonl (one line per stream, appended and flushed as each ends), <out>/plan.log (one line per
step, SKIP lines included), <out>/status-start.json, <out>/status-end.json; last line on stdout: RB-FIN.
The bearer is read from a 0600 header file ("Authorization: Bearer ..."), never from argv; it is written nowhere.
Usage: rb-bench.py --base http://127.0.0.1:8478 --auth <header file> --model <name> --out <dir>
                   [--budget-s 900] [--min-left-s 150] [--levels p535,p1k,p4k,p8k,p16k] [--reps 7] [--max-tokens 1024]
                   [--batches c2,c4,c2b,c4b]"""
import argparse, base64, hashlib, http.client, json, os, sys, threading, time, urllib.parse
from datetime import datetime, timezone

INSTR = ("Write a very long and detailed essay, of at least 3000 words, on the history of computing, "
         "with one section per decade from the 1940s to the 2020s.")
LIST_WORDS = {"p535": 0, "p1k": 480, "p4k": 3500, "p8k": 7400, "p16k": 16000}
# Seconds a step is expected to take (2026-10-31: first token 0.21 s at 538 prompt tokens, 1.12 s at 4.5k, 8.85 s at
# 16.5k; decode supposed 35 to 60 tokens/s), with room for the padded end. A step starts only if this fits.
EST = {"warm": 5, "p535": 30, "p1k": 32, "p4k": 38, "p8k": 45, "p16k": 55, "c2": 60, "c4": 120, "c2b": 60, "c4b": 120}


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def prompt(level):
    k = LIST_WORDS[level]
    if k == 0:
        return INSTR
    return "Here is a list. " + " ".join(["apple"] * k) + ". Ignore the list. " + INSTR


class Api:
    def __init__(self, base, auth_file):
        u = urllib.parse.urlsplit(base)
        self.host, self.port = u.hostname, u.port or 80
        line = open(auth_file).read().strip()
        k, _, v = line.partition(":")
        if k.strip().lower() != "authorization" or not v.strip():
            raise SystemExit("rb-bench: the header file does not hold an Authorization line")
        self.auth = v.strip()

    def get_json(self, path, timeout=60):
        c = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            c.request("GET", path, headers={"Authorization": self.auth})
            r = c.getresponse()
            body = r.read()
            return r.status, json.loads(body) if body else None
        except (OSError, ValueError, http.client.HTTPException) as e:
            return None, {"error": f"{type(e).__name__}: {e}"}
        finally:
            c.close()

    def seconds_left(self):
        s, j = self.get_json("/x0ne/v1/status")
        try:
            return int(j.get("seconds_left")) if s == 200 else None
        except (AttributeError, TypeError, ValueError):
            return None


def signed_subset(x0ne):
    """What the final chunk says about the answer, kept small: the receipt is verified by the relay already
    (receipt_verified); the signed payload is decoded for the fields that say how it was served."""
    out = {"receipt_verified": x0ne.get("receipt_verified")}
    p = x0ne.get("signed_payload_b64")
    if isinstance(p, str):
        try:
            sp = json.loads(base64.b64decode(p))
            out["signed_keys"] = sorted(sp.keys()) if isinstance(sp, dict) else None
            for k in ("model_digest_hex", "compute_device", "kv_cache_shared_between_requests",
                      "requests_batched_across_clients", "prompt_tokens", "completion_tokens", "finish_reason"):
                if isinstance(sp, dict) and k in sp:
                    out[k] = sp[k]
            iso = sp.get("isolation") if isinstance(sp, dict) else None
            if isinstance(iso, dict):
                out["isolation"] = iso
        except (ValueError, TypeError):
            out["signed_payload_undecodable"] = True
    r = x0ne.get("receipt")
    if isinstance(r, dict):
        for k in ("model_digest_hex", "prompt_tokens", "completion_tokens", "compute_device"):
            if k in r:
                out.setdefault("receipt_" + k, r[k])
    return out


def one_stream(api, body, rec, t_barrier=None):
    """Sends one streamed chat completion and records every SSE event with the time it was read. rec is filled in
    place; nothing raises out of here."""
    data = json.dumps(body).encode()
    rec.update(status=None, content_type=None, t_headers=None, events=[], final=None, usage=None, error=None,
               bytes_in=0, reads=0)
    c = http.client.HTTPConnection(api.host, api.port, timeout=900)
    try:
        rec["t_start_utc"] = now_utc()
        t0 = time.monotonic()
        if t_barrier is not None:
            rec["t_send_since_barrier"] = t0 - t_barrier
        c.request("POST", "/v1/chat/completions", body=data,
                  headers={"Authorization": api.auth, "Content-Type": "application/json"})
        r = c.getresponse()
        rec["t_headers"] = time.monotonic() - t0
        rec["status"] = r.status
        rec["content_type"] = r.getheader("Content-Type")
        if r.status != 200 or not (rec["content_type"] or "").startswith("text/event-stream"):
            raw = r.read(65536)
            rec["error"] = f"HTTP {r.status}: {raw[:400].decode('utf-8', 'replace')}"
            rec["retry_after"] = r.getheader("Retry-After")
            return
        buf = b""
        while True:
            chunk = r.read1(65536)
            t = time.monotonic() - t0
            if not chunk:
                rec["t_eof"] = t
                break
            rec["bytes_in"] += len(chunk)
            rec["reads"] += 1
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                rec["events"].append(parse_event(ev, t, rec))
        if buf.strip():
            rec["events"].append({"t": rec.get("t_eof"), "kind": "trailing", "n": len(buf)})
    except (OSError, http.client.HTTPException, ValueError) as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        c.close()


def parse_event(ev, t, rec):
    s = ev.decode("utf-8", "replace")
    out = {"t": round(t, 6), "n": len(ev) + 2}
    if s.startswith(":"):
        out["kind"] = "comment"
        return out
    lines = [ln[5:].lstrip() for ln in s.split("\n") if ln.startswith("data:")]
    if not lines:
        out["kind"] = "other"
        return out
    d = "\n".join(lines)
    if d == "[DONE]":
        out["kind"] = "done"
        return out
    try:
        j = json.loads(d)
    except ValueError:
        out["kind"] = "unparsed"
        return out
    if "error" in j:
        out["kind"] = "error"
        out["error"] = j["error"]
        rec["error"] = f"error event: {json.dumps(j['error'])[:300]}"
        return out
    ch = j.get("choices") or []
    if j.get("usage"):
        rec["usage"] = j["usage"]
    if isinstance(j.get("x0ne"), dict):
        rec["final"] = signed_subset(j["x0ne"])
    if not ch:
        out["kind"] = "usage" if j.get("usage") else "empty"
        return out
    c0 = ch[0]
    delta = c0.get("delta") or {}
    if c0.get("finish_reason"):
        out["kind"] = "final"
        out["finish_reason"] = c0["finish_reason"]
        rec["finish_reason"] = c0["finish_reason"]
    elif "role" in delta:
        out["kind"] = "role"
        out["content"] = delta.get("content") or ""
    else:
        out["kind"] = "content"
        out["content"] = delta.get("content") or ""
    return out


def summary(rec):
    cont = [e for e in rec.get("events", []) if e.get("kind") == "content" and e.get("content")]
    fin = next((e["t"] for e in rec.get("events", []) if e.get("kind") == "final"), None)
    u = rec.get("usage") or {}
    s = {"name": rec["name"], "status": rec.get("status"), "prompt_tokens": u.get("prompt_tokens"),
         "completion_tokens": u.get("completion_tokens"), "finish_reason": rec.get("finish_reason"),
         "content_frames": len(cont), "t_headers": rec.get("t_headers"),
         "t_first_content": cont[0]["t"] if cont else None, "t_last_content": cont[-1]["t"] if cont else None,
         "t_final": fin, "receipt_verified": (rec.get("final") or {}).get("receipt_verified"), "error": rec.get("error")}
    if cont and len(cont) > 1 and u.get("completion_tokens"):
        span = cont[-1]["t"] - cont[0]["t"]
        # Rough: all tokens over the content span; rb-analyse.py takes the first frame's tokens out.
        s["rough_tok_s"] = round(u["completion_tokens"] / span, 2) if span > 0 else None
    return s


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--base", required=True)
    a.add_argument("--auth", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--budget-s", type=int, default=900)
    a.add_argument("--min-left-s", type=int, default=150)
    a.add_argument("--levels", default="p535,p1k,p4k,p8k,p16k")
    a.add_argument("--reps", type=int, default=7)
    a.add_argument("--max-tokens", type=int, default=1024)
    a.add_argument("--batches", default="")
    o = a.parse_args()
    os.makedirs(o.out, exist_ok=True)
    api = Api(o.base, o.auth)
    t_begin = time.monotonic()
    plan_log = open(os.path.join(o.out, "plan.log"), "a", buffering=1)
    out = open(os.path.join(o.out, "streams.jsonl"), "a", buffering=1)

    def log(msg):
        line = f"{now_utc()} {msg}"
        print(line, flush=True)
        plan_log.write(line + "\n")

    s, st = api.get_json("/x0ne/v1/status")
    json.dump({"http": s, "status": st}, open(os.path.join(o.out, "status-start.json"), "w"))
    log(f"start: status HTTP {s}, seconds_left {(st or {}).get('seconds_left')}, model {o.model}, "
        f"budget {o.budget_s} s, max_tokens {o.max_tokens}")

    def body(level, n):
        return {"model": o.model, "messages": [{"role": "user", "content": prompt(level)}], "max_tokens": n,
                "stream": True, "stream_options": {"include_usage": True}}

    def fits(step):
        el = time.monotonic() - t_begin
        if el + EST[step] > o.budget_s:
            return f"budget: {el:.0f} s used of {o.budget_s}, {EST[step]} s wanted"
        left = api.seconds_left()
        if left is not None and left < EST[step] + o.min_left_s:
            return f"session: {left} s left, {EST[step] + o.min_left_s} s wanted"
        return None

    def run_batch(name, level, c, n, rep=1, batch=None):
        recs = [{"name": f"{name}-{i + 1}" if c > 1 else name, "level": level, "rep": rep, "c": c,
                 "batch": batch, "idx": i, "max_tokens": n, "prompt_chars": len(prompt(level)),
                 "prompt_sha256": hashlib.sha256(prompt(level).encode()).hexdigest()} for i in range(c)]
        b = body(level, n)
        if c == 1:
            one_stream(api, b, recs[0])
        else:
            bar = threading.Barrier(c + 1)
            t_bar = [None]

            def go(r):
                bar.wait()
                one_stream(api, b, r, t_bar[0])
            th = [threading.Thread(target=go, args=(r,)) for r in recs]
            for t in th:
                t.start()
            t_bar[0] = time.monotonic()
            bar.wait()
            for t in th:
                t.join()
        for r in recs:
            r["summary"] = summary(r)
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            log("stream " + json.dumps(r["summary"]))
        return recs

    steps = [("warm", "warm", "p535", 1, 16, 1, None)]
    # 2026-11-04: one round over every level, then the next round: a cut keeps every level, with fewer reps.
    for i in range(o.reps):
        for lv in o.levels.split(","):
            steps.append((lv, f"{lv}-r{i + 1}", lv, 1, o.max_tokens, i + 1, None))
    for bn in [x for x in o.batches.split(",") if x]:
        c = int(bn[1])
        steps.append((bn, bn, "p535", c, o.max_tokens, 1, bn))
    for step, name, level, c, n, rep, batch in steps:
        why = fits(step)
        if why:
            log(f"SKIP {name}: {why}")
            continue
        log(f"step {name}: level {level}, c={c}, max_tokens {n}")
        recs = run_batch(name, level, c, n, rep, batch)
        # No HTTP status at all: the relay is gone (VM lost, session ended); the next steps would only fail.
        if all(r.get("status") is None for r in recs):
            log(f"STOP after {name}: the local API does not answer ({str(recs[0].get('error'))[:160]})")
            break
    s, st = api.get_json("/x0ne/v1/status")
    json.dump({"http": s, "status": st}, open(os.path.join(o.out, "status-end.json"), "w"))
    log(f"end: status HTTP {s}, seconds_left {(st or {}).get('seconds_left')}, "
        f"{time.monotonic() - t_begin:.0f} s of measures")
    print("RB-FIN", flush=True)


if __name__ == "__main__":
    main()
