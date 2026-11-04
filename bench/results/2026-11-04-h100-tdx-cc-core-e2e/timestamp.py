#!/usr/bin/env python3
"""Passes stdin to stdout unchanged, and writes to <out jsonl> one line per read: the time it came
(seconds since this process started, time.monotonic, and UTC) and its text. Run between `up` and tee, it dates each
step line of `up` as it is written, a partial line included ("[1/5] Creating ... " before its "done (39 s)").
The jsonl holds what `up` printed (project, account, addresses): it stays in the run directory (0700), never in the
repository; rb-up-steps.py extracts the step labels and times from it.
Usage: timestamp.py <out jsonl>"""
import json, os, sys, time
from datetime import datetime, timezone

t0 = time.monotonic()
with open(sys.argv[1], "a", buffering=1) as log:
    log.write(json.dumps({"t": 0.0, "utc": datetime.now(timezone.utc).isoformat(), "start": True}) + "\n")
    while True:
        b = os.read(0, 65536)
        t = time.monotonic() - t0
        if not b:
            log.write(json.dumps({"t": round(t, 3), "utc": datetime.now(timezone.utc).isoformat(), "eof": True}) + "\n")
            break
        os.write(1, b)
        log.write(json.dumps({"t": round(t, 3), "utc": datetime.now(timezone.utc).isoformat(),
                              "text": b.decode("utf-8", "replace")}) + "\n")
