# 2026-11-04, H100 TDX, `qwen3:8b` on x0ne Core, streamed through the relay

**VERIFIED with the TEST archive key and the TEST transparency log; not a
release measurement.**

The decode rate of this Engine on x0ne Core, `qwen3:8b`, read frame by frame
on the client's clock, on the product path: `x0ne-engine cloud up`, its
relay, and the TD's padded stream (one frame per 100 ms tick, a tick without
text sent as an SSE comment that the relay drops, the last frame on a Padme
frame count). None of it changes what the README says under
[What is left](../../../README.md#what-is-left): x0ne Engine is not
end-to-end confidential yet.

GCP `a3-highgpu-1g` spot, us-east5-a, Intel TDX and one H100 in CC mode,
transparent TCB policy, the VM from 15:20 to 15:31 UTC. Image x0ne-cvm 0.2.9
(disk tar `1ebd1348…`, its `x0ne-infer` `0254fd6b…`), not published: a build
of this tree does not give it ([CHANGELOG](../../../CHANGELOG.md), 0.2.0-beta.7,
"x0ne Core is the served engine"). The `x0ne-engine` front and the relay
`x0ne-engine-proxy` were built from earlier sources of the same day, with
the TEST archive key and the TEST transparency log pinned. `up` was the
`x0ne-engine-cloud` built from the sources of 2026-10-31, before `up` refused
a relay that pins the TEST keys (2026-11-03): the `up` of this tree refuses
this relay with 95 ([cloud.md](../../../docs/cloud.md)). Verdict VERIFIED, see
[hardware-runs.md](../../../docs/hardware-runs.md). `summarize.py` does not
read this directory; `rb-analyse.py` does.

## Method

`rb-bench.py`, standard library only, in a container on the operator's
machine, against the relay's local API (`127.0.0.1:8478`):

```sh
rb-bench.py --base http://127.0.0.1:8478 --auth <header file> --model qwen3:8b --out <dir> --budget-s 1080 --min-left-s 150
```

Streamed chat completions, `max_tokens` 1024, `stream_options.include_usage`,
no sampling parameter sent. The prompt is the instruction alone (58 tokens
with Qwen3's chat template, `usage.prompt_tokens`), or the instruction after a
list of 480, 3,500, 7,400 or 16,000 "apple" (548, 3,568, 7,468 and 16,068
tokens): `prompt()` in `rb-bench.py`, its SHA-256 in each row. The files name
these groups `p535`, `p1k`, `p4k`, `p8k` and `p16k`: names kept from an
earlier run with another model's template, not token counts. One warm-up
stream of 16 tokens, then seven rounds of one stream per prompt, from the
shortest to the longest, one at a time: 36 streams from 15:23:48 to
15:30:03 UTC. No streams were sent together.

Each SSE event is timed when it is read (`time.monotonic`, from the send).
`rb-analyse.py` counts the tokens of each frame as the difference of the
prefix counts with the model's `tokenizer.json` (the catalog's file for
`qwen3:8b`, SHA-256 `aeb13307…`); the sum equals `usage.completion_tokens` in
all 36 streams. Decode, two ways: span = (tokens after the first content
frame) / (time from the first to the last content frame); fit = least-squares
slope of the cumulative tokens against time.

## Results

c = 1, n = 7 each; brackets are min and max. Every answer had 1,024 tokens,
finish `length`.

| Prompt tokens | Decode, span | Decode, fit | First content | Last content to final frame | Role to final, frames |
|---|---|---|---|---|---|
| 58 | 136.87 tok/s [136.73, 137.01] | 137.29 [137.21, 137.30] | 0.202 s [0.200, 0.208] | 0.402 s [0.395, 0.417] | 80 |
| 548 | 135.49 tok/s [135.47, 135.76] | 136.27 [136.26, 136.31] | 0.208 s [0.204, 0.209] | 0.305 s [0.301, 0.311] | 80 |
| 3,568 | 128.98 tok/s [128.74, 129.00] | 128.96 [128.94, 129.00] | 0.705 s [0.699, 0.708] | 0.304 s [0.302, 0.309] | 88 |
| 7,468 | 122.65 tok/s [121.30, 122.83] | 122.51 [122.30, 122.56] | 2.098 s [2.093, 2.108] | 0.201 s [0.104, 0.213] | 104 |
| 16,068 | 116.17 tok/s [115.92, 116.70] | 116.58 [116.46, 116.61] | 8.042 s [8.025, 8.155] | 1.008 s [1.000, 1.012] | 176 |

- Frames with text came 100 ms apart: per stream, the median gap was 99.39
  to 100.19 ms; every gap of the 35 long streams lay between 79.5 and
  120.8 ms. A frame carried a median of 14 tokens after 58 and 548 prompt
  tokens, 13 after 3,568, 12 after 7,468 and 16,068.
- Every final chunk said `receipt_verified` true, 36 of 36 with the warm-up:
  the relay verified each receipt. Its signed fields gave `compute_device`
  `cuda:0`, the catalog's model digest `e1776f7c…`, and the four `isolation`
  flags false (`kv_cache_shared_between_requests`,
  `prefix_cache_shared_between_sessions`, `requests_batched_across_clients`,
  `prompt_text_logged`).
- x0ne Core alone, its own bench in the process on another VM in CC mode,
  decoded at 138.62 tok/s after 20 prompt tokens over 256 generated
  (2026-11-03, n = 3, [benchmarks.md](../../../docs/benchmarks.md)); 136.87
  here after 58 over 1,024, on the client's clock, through the relay. The
  prompts, the generated length, the clock and the VM differ; this page does
  not split the causes.
- The steps of `up` (`up-steps.json`, seconds on the client's clock):
  creation 39.0 (`up` says 38), boot wait 0.7, from `[3/5]` to `[4/5]`
  (waiting for the TD, its attestation document, the checks) 64.7, from
  `[4/5]` to `[5/5]` (key released, model loaded) 11.0; `[5/5]` 131.6 s
  after the client began reading `up`'s output. The `[3/5]` line is that of
  the sources of 2026-10-31 word for word: "Nothing has been sent yet" is
  wrong there, the VM's metadata already holds the session config; the line
  changed on 2026-11-03, after that build.

## What the padding leaves measurable

- Decode: yes. Frames leave on the tick and carry what was generated since
  the last one, so 7.4 to 8.8 s of text per stream are read to within one
  frame; span and fit agree within 0.84 %.
- First token: to the tick only. The first content frame leaves on the tick
  that follows the first token.
- End of the generation: no. The final frame waits for a Padme frame count;
  here 0.104 to 1.012 s after the last text.
- The role-to-final frame count, rounded from the client's times: 80, 80,
  88, 104 and 176 for the c = 1 streams, Padme sizes; 10, the floor, for the
  16-token warm-up. These counts come from the client's clock; they do not
  establish the TD's own frame numbering.

## Not measured

The cost of the confidential mode on this path (no run of this Engine with
confidential computing off), streams sent together, whole answers (padded),
sealed streams, other models, sampling parameters, prompts beyond 16,068
tokens, answers beyond 1,024 tokens, a release build (the relay pinned the
TEST keys). One VM, one host.

## Files

- `streams.jsonl`: one stream per line as `rb-bench.py` wrote it: every SSE
  event with its time, size and text, the usage, and the final chunk's
  fields (receipt verified, token counts, model digest, isolation flags, the
  names of the signed fields); no receipt, signature or session key.
- `plan.log`: the bench's own log, one line per step and stream.
- `summary.json`, `summary.txt`: `rb-analyse.py`'s output.
- `up-steps.json`: the lines of `up` from `[1/5]` to `[5/5]`, addresses
  withheld, with their times (`rb-up-steps.py` over `timestamp.py`'s record).
  The VM's name is replaced by `x0ne-engine-<id>`, after the scripts ran
  (2026-11-04).
- `rb-bench.py`, `rb-analyse.py`, `rb-up-steps.py`, `timestamp.py`: the
  scripts as run. For publication their docstrings and comments were
  edited, and two file names and the variable `MASK_PROJECT` were put in
  English; the code is otherwise unchanged.

The attestation document, quote, reference, receipts, status files and serial
console are not in this repository; the run's record is kept outside it.

```sh
# tokenizers 0.21.1 on sys.path (RB_DEPS), tokenizer.json from the catalog entry of qwen3:8b
RB_DEPS=<dir> python3 rb-analyse.py streams.jsonl tokenizer.json <out dir>
```

It writes `summary.json` and `summary.txt` byte for byte as kept here
(checked on 2026-11-04), and `frames.jsonl`, each frame's tokens.

Estimated cost 1.07 USD for the VM (5.66 USD/h on Cloud Billing's catalog),
at most 0.02 USD more for the image, its transit bucket and the egress:
1.09 USD.
