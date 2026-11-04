#!/usr/bin/env python3
"""The duration of each step of `x0ne-engine cloud up`, from timestamp.py's jsonl. Rebuilds the lines as up
wrote them, each with the time its first byte and its newline were read (seconds since up's output began), and
writes the lines from "[1/5]" to "[5/5]" with addresses, accounts and the project withheld,
then the steps:
  1 creation        "[1/5]" written to the end of its line ("done (N s)": up's own count is kept too)
  2 boot wait       "[2/5]" to the end of its line
  3 proof           "[3/5]" to "[4/5]" (TD reached, attestation document received, verdict), with its inner lines dated
  4 key and model   "[4/5]" to "[5/5]" (key released, model fetched and loaded in the TD, first serving)
  total             up's first byte to "[5/5]"
Output: JSON on stdout; nothing that the masks do not cover is copied but these lines.
Usage: MASK_PROJECT=<p> rb-up-steps.py <up-times.jsonl>"""
import json, os, re, sys

proj = os.environ.get("MASK_PROJECT", "")
RULES = [
    re.compile(r"ya29\.[A-Za-z0-9._\-]*"), re.compile(r"Bearer\s+\S*"),
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)+"),
    re.compile(r"(?<![\d.])(?!0\.|127\.0\.0\.1(?![\d]))(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d]|\.\d)"),
    re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{1,4}:){3,7}[0-9A-Fa-f]{1,4}(?![0-9A-Fa-f:])"),
]


def mask(s):
    if proj:
        s = s.replace(proj, "<project withheld>")
    for rx in RULES:
        s = rx.sub("<withheld>", s)
    return s


lines, cur, cur_t0 = [], "", None
for raw in open(sys.argv[1]):
    r = json.loads(raw)
    if "text" not in r:
        continue
    for part in re.split(r"(\n)", r["text"]):
        if part == "\n":
            lines.append({"t_first": cur_t0 if cur_t0 is not None else r["t"], "t_end": r["t"], "text": cur})
            cur, cur_t0 = "", None
        elif part:
            if cur_t0 is None:
                cur_t0 = r["t"]
            cur += part
if cur:
    lines.append({"t_first": cur_t0, "t_end": None, "text": cur})


def find(prefix):
    return next((ln for ln in lines if ln["text"].lstrip().startswith(prefix)), None)


s = {k: find(f"[{k}/5]") for k in range(1, 6)}
out = {"lines": [], "steps": {}}
inside = False
for ln in lines:
    if ln["text"].startswith("[1/5]"):
        inside = True
    if inside:
        out["lines"].append({"t_first": ln["t_first"], "t_end": ln["t_end"], "text": mask(ln["text"])[:400]})
    if ln["text"].startswith("[5/5]"):
        inside = False


def d(a, b):
    return None if a is None or b is None else round(b - a, 3)


if s[1]:
    m = re.search(r"done \((\d+) s\)", s[1]["text"])
    out["steps"]["1 creation"] = {"s": d(s[1]["t_first"], s[1]["t_end"]), "up_says_s": int(m.group(1)) if m else None}
if s[2]:
    out["steps"]["2 boot wait"] = {"s": d(s[2]["t_first"], s[2]["t_end"])}
if s[3] and s[4]:
    out["steps"]["3 proof"] = {"s": d(s[3]["t_first"], s[4]["t_first"])}
if s[4] and s[5]:
    out["steps"]["4 key and model"] = {"s": d(s[4]["t_first"], s[5]["t_first"])}
if s[5]:
    out["steps"]["total to [5/5]"] = {"s": s[5]["t_first"]}
json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
print()
