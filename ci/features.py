#!/usr/bin/env python3
"""FEATURES.toml, the feature list of this repository: checks, and FEATURES.md.

  python3 ci/features.py           write FEATURES.md from FEATURES.toml
  python3 ci/features.py --check   validate FEATURES.toml; FEATURES.md must be what it would write

Standard library only (tomllib, Python 3.11 or later). Exit 0 green, 1 refused, 2 unreadable.
The site reads the same file: titles and sentences are written for a reader, not for us. The
repository is in English; the site keeps the French of each entry, keyed by id.
Technical names (register values, protocol names, exit codes) belong in proof, figure and limit.

Levels: hardware (exercised on a trusted execution environment, in a run docs/hardware-runs.md
dates), local (tests, CI or a machine without a TEE), prototype (code present, off the served
path), planned (announced, not built), retired (replaced; replaced_by names the successor).
since: date of the first proof. hardware_since: the first hardware run that exercised it.
"""
import argparse
import datetime
import re
import sys
import tomllib

PRODUCT = "x0ne Engine"
TOML = "FEATURES.toml"
MD = "FEATURES.md"
SCRIPT = "ci/features.py"
# A hardware entry's hardware_since must be a date one of these files names.
RUN_FILES = ["docs/hardware-runs.md"]
HEADNOTE = (
    "Not end-to-end confidential: the host still sees timing, sizes and the model name; the "
    "recorded runs pinned a TEST archive key and a TEST transparency log through a relay built "
    "for tests; no release is published and no external audit has been done. The README's "
    "[What is left](README.md#what-is-left) is the reference."
)

LEVELS = ("hardware", "local", "prototype", "planned", "retired")
GROUPS = [
    ("Done, measured on hardware", ("hardware",)),
    ("Done, tested", ("local", "prototype")),
    ("Planned", ("planned",)),
    ("Replaced", ("retired",)),
]
FIELDS = {
    "id": str, "title_en": str, "sentence_en": str,
    "level": str, "since": str, "hardware_since": str, "version": str, "proof": str,
    "figure": str, "limit": str, "replaced_by": str, "order": int, "public": bool,
}
REQUIRED = ("id", "title_en", "sentence_en", "level", "proof", "order")
TITLE_WORDS = (2, 9)
SENTENCE_WORDS = 24
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
# Words a reader of the site should not meet in a title or a sentence.
JARGON = re.compile(
    r"\b(RTMR[0-9]?|MRTD|HPKE|CCEL|TCB|PCS|NRAS|OHTTP|UKI|SVN|DCAP|FMSPC|OCSP|CRL|PCK|JWKS|NVML|"
    r"ueid|memfd(_secret)?|Padme|KV|logprob|GLiClass|tlog|C2SP|dm-verity|SMBIOS|fw_cfg|nonce|"
    r"quote[sd]?|mTLS|SSE|0-RTT|Ed25519|AES-256-GCM|POC)\b|proof of concept",
    re.I,
)
# A claim of end-to-end confidentiality is refused unless the same text negates it.
E2E = re.compile(r"end-to-end confidential|confidentiel(?:le)? de bout en bout", re.I)
NEG = re.compile(r"\b(not|no|never|without|n't|yet|pas|non|jamais|ni|sans|encore)\b", re.I)
# No task number and no short commit hash: the list is read outside this repository.
TASK_REF = re.compile(r"#\d+")
SHORT_HASH = re.compile(r"(?<![0-9a-zA-Z])[0-9a-f]{7}(?![0-9a-zA-Z])|\b[0-9a-f]{40}\b")


def words(s):
    return [w for w in re.split(r"[ \t\n]+", s.strip()) if re.search(r"[^\W_]", w)]


def iso(s):
    try:
        return datetime.date.fromisoformat(s)
    except ValueError:
        return None


def load(path):
    with open(path, "rb") as fh:
        doc = tomllib.load(fh)
    feats = doc.get("feature")
    if not isinstance(feats, list) or not feats:
        sys.exit(f"{path}: no [[feature]] table")
    return feats


def run_dates(paths):
    found = set()
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                found.update(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", fh.read()))
        except OSError as e:
            sys.exit(f"{p}: {e.strerror}")
    return found


def check(feats, runs):
    errs, ids = [], {}
    for f in feats:
        i = f.get("id", "?")
        for k in f:
            if k not in FIELDS:
                errs.append(f"{i}: unknown field {k}")
        for k, t in FIELDS.items():
            if k in f and not isinstance(f[k], t):
                errs.append(f"{i}: {k} is not {t.__name__}")
        for k in REQUIRED:
            if not f.get(k) and f.get(k) != 0:
                errs.append(f"{i}: {k} missing")
        if not isinstance(i, str) or not ID_RE.match(i):
            errs.append(f"{i}: id is not lower-case letters, digits and dashes")
        if i in ids:
            errs.append(f"{i}: duplicate id")
        ids[i] = f
        lvl = f.get("level")
        if lvl not in LEVELS:
            errs.append(f"{i}: unknown level {lvl!r}")
            continue
        since, hw = f.get("since", ""), f.get("hardware_since", "")
        if lvl == "planned":
            if since or hw:
                errs.append(f"{i}: a planned entry has no since or hardware_since")
        else:
            if not since or not iso(since):
                errs.append(f"{i}: since is not a date")
        if lvl == "hardware":
            if not hw or not iso(hw):
                errs.append(f"{i}: hardware entry without a hardware_since date")
            else:
                if hw not in runs:
                    errs.append(f"{i}: hardware_since {hw} is not a run date of {', '.join(RUN_FILES)}")
                if iso(since) and iso(hw) < iso(since):
                    errs.append(f"{i}: hardware_since {hw} before since {since}")
        elif hw:
            errs.append(f"{i}: hardware_since on a {lvl} entry")
        if lvl == "retired":
            if not f.get("replaced_by"):
                errs.append(f"{i}: retired entry names no replaced_by")
        elif f.get("replaced_by"):
            errs.append(f"{i}: replaced_by on a {lvl} entry")
        for k in ("title_en",):
            n = len(words(f.get(k, "")))
            if not TITLE_WORDS[0] <= n <= TITLE_WORDS[1]:
                errs.append(f"{i}: {k} has {n} words ({TITLE_WORDS[0]} to {TITLE_WORDS[1]})")
        for k in ("sentence_en",):
            n = len(words(f.get(k, "")))
            if n > SENTENCE_WORDS:
                errs.append(f"{i}: {k} has {n} words (at most {SENTENCE_WORDS})")
        for k in ("title_en", "sentence_en"):
            m = JARGON.search(f.get(k, ""))
            if m:
                errs.append(f"{i}: {k} says {m.group(0)!r}: say it in plain words, or keep it in proof or limit")
        for k, v in f.items():
            if not isinstance(v, str):
                continue
            if E2E.search(v) and not NEG.search(v):
                errs.append(f"{i}: {k} claims end-to-end confidentiality")
            if TASK_REF.search(v):
                errs.append(f"{i}: {k} carries a task number")
            if SHORT_HASH.search(v):
                errs.append(f"{i}: {k} carries a commit hash")
    for f in feats:
        r = f.get("replaced_by")
        if r:
            if r not in ids:
                errs.append(f"{f['id']}: replaced_by {r!r} is not an id")
            elif ids[r].get("level") == "retired":
                errs.append(f"{f['id']}: replaced_by {r!r} is itself retired")
    orders = [(f.get("level"), f.get("order")) for f in feats]
    if len(set(orders)) != len(orders):
        errs.append("two entries of one level share an order")
    return errs


def md_cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def feature_cell(f, ids):
    s = f"**{md_cell(f['title_en'])}** — {md_cell(f['sentence_en'])}"
    if f["level"] == "prototype":
        s += " (prototype)"
    if f.get("figure"):
        s += f" _{md_cell(f['figure'])}_"
    if f["level"] == "retired":
        r = f["replaced_by"]
        s += f" Replaced by **{md_cell(ids[r]['title_en'])}** (`{r}`)."
    return s


def features_md(feats, runs_note):
    pub = [f for f in feats if f.get("public", True)]
    ids = {f["id"]: f for f in feats}
    hw = [f for f in pub if f["level"] == "hardware"]
    first_hw = min((f["hardware_since"] for f in hw), default="")
    n = {g: sum(1 for f in pub if f["level"] in lv) for g, lv in GROUPS}
    out = [
        f"# {PRODUCT}: features", "",
        f"Generated from `{TOML}` by `{SCRIPT}`; edit the TOML, run the script, commit both "
        f"(`--check` refuses a stale {MD}). One line per feature, public entries only, each group "
        "in the order of the TOML's `order`.", "",
        "`since` is the date of the first proof. `hardware run` is the first run on real hardware "
        f"that exercised the feature, dated in {runs_note}. A tested entry ran in tests, in CI or "
        "on a machine without a trusted execution environment; a prototype is code off the served path.", "",
        HEADNOTE, "",
        f"{n['Done, measured on hardware']} measured on hardware"
        + (f" since {first_hw}" if first_hw else "")
        + f", {n['Done, tested']} tested, {n['Planned']} planned, {n['Replaced']} replaced.", "",
    ]
    for g, lv in GROUPS:
        rows = sorted((f for f in pub if f["level"] in lv), key=lambda f: (LEVELS.index(f["level"]), f["order"]))
        if not rows:
            continue
        out += [f"## {g} ({len(rows)})", ""]
        if g == "Done, measured on hardware":
            out += ["| Feature | Hardware run | Since | Proof | Limit |", "|---|---|---|---|---|"]
            for f in rows:
                out.append(f"| {feature_cell(f, ids)} | {f['hardware_since']} | {f['since']} | "
                           f"{md_cell(f['proof'])} | {md_cell(f.get('limit') or '—')} |")
        elif g == "Planned":
            out += ["| Feature | Where it is said | Limit |", "|---|---|---|"]
            for f in rows:
                out.append(f"| {feature_cell(f, ids)} | {md_cell(f['proof'])} | {md_cell(f.get('limit') or '—')} |")
        else:
            out += ["| Feature | Since | Proof | Limit |", "|---|---|---|---|"]
            for f in rows:
                out.append(f"| {feature_cell(f, ids)} | {f['since']} | {md_cell(f['proof'])} | "
                           f"{md_cell(f.get('limit') or '—')} |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="validate, and require FEATURES.md up to date")
    ap.add_argument("--toml", default=TOML)
    ap.add_argument("--md", default=MD)
    ap.add_argument("--runs", nargs="*", default=RUN_FILES, help="files whose dates hardware_since may take")
    a = ap.parse_args()
    feats = load(a.toml)
    runs = run_dates(a.runs)
    errs = check(feats, runs)
    pub = sum(1 for f in feats if f.get("public", True))
    by = ", ".join(f"{lv} {sum(1 for f in feats if f.get('level') == lv)}" for lv in LEVELS)
    print(f"{a.toml}: {len(feats)} features, {pub} public; {by}")
    if errs:
        print("\n".join("REFUSED " + e for e in errs))
        sys.exit(1)
    text = features_md(feats, " and ".join(f"`{p}`" for p in a.runs))
    if a.check:
        try:
            with open(a.md, encoding="utf-8") as fh:
                current = fh.read()
        except OSError:
            current = None
        if current != text:
            print(f"REFUSED {a.md} is not what {SCRIPT} writes from {a.toml}: run python3 {SCRIPT}")
            sys.exit(1)
        print(f"check: ok ({a.md} up to date)")
        return
    with open(a.md + ".tmp", "w", encoding="utf-8") as fh:
        fh.write(text)
    import os
    os.replace(a.md + ".tmp", a.md)
    print(f"wrote {a.md}")


if __name__ == "__main__":
    main()
