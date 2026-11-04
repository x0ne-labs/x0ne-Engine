# Benchmark rows

The raw rows behind [docs/benchmarks.md](../../docs/benchmarks.md), one
directory per date and machine. Dates are UTC, the date of the first row.

```sh
python3 bench/results/summarize.py      # standard library only; prints every table
```

Nothing here is a full run record. Server logs, recordings, attestation
documents, receipts and quotes stayed with the runs: they name machines,
accounts and registers that are not part of a speed figure. What was kept is
what the figures are computed from.

## Row formats

**H100, Intel TDX, GPU in CC mode** (`*-h100-tdx-cc/`). `times.log`: one line
per request, `<name>: <seconds> s, HTTP <code>`, written by `curl -w
'%{time_total}'` on the operator's machine, through the relay and the attested
channel. `responses.jsonl`: per response, the token counts, the finish
reason, whether the relay verified its receipt, and from that receipt the
device that ran the arithmetic and its environment; plus the SHA-256 of the
answer's text. Names: `c1-N` and `c256-N`, the short prompt with `max_tokens`
1 and 256; `long4kp-N` and `long4k-N`, the long prompt with `max_tokens` 1
and 129; `warm`, one request before them. On 2026-10-17, `times.log` was
read back from the terminal recording of the run (`max_tokens=<n> run <i>:
<seconds> s`), the only place curl printed it. On 2026-10-31 the long prompt
gave way to a ladder: `k<N>p-i` and `k<N>-i`, a prompt of about N more tokens
than the short one, with `max_tokens` 1 and 129 (1,048, 2,048, 4,048 and
8,048 prompt tokens). `streams.jsonl` (2026-10-22, 2026-10-31): per streamed
request, the token counts, the seconds to the first content event and to the
end, the finish reason and whether the relay verified the receipt. That
directory also has `kv-cache.jsonl`: per streamed request of the session KV
cache measure, the level (about N tokens in the first turn), the run, the
request (`c1` and `c2`, the two turns of a conversation; `n1` and `n2`, the
same two prompts without one), the token counts, the seconds to the headers,
to the first content event and to the end (client wall clock through the
relay), whether a handle came back, and from the signed receipt the turn,
the tokens reused and the `kv_cache_shared_between_requests` flag.

**x0ne Core alone** (`2026-11-03-h100-tdx-cc-core/`). The output of x0ne Core's bench, not of
this Engine: one line per prompt in the `.txt` files (prompt tokens, tokens
generated, the median decode rate and each run's, the first token), one
line per run in the `.err` files, the medians in the `.tsv` files; on
2026-11-03, x0ne Core's GPU tests too, their reports and `MEASURE` lines.
Its README has the rest. `summarize.py` does not read it.

**x0ne Core alone, per token** (`2026-11-04-h100-tdx-cc-core-bench/`). The
output of `x0ne-core-itl`, a bench program on x0ne Core's library, not of
this Engine: one JSON line per run with the time of each of its 256 tokens,
the prompts as token ids, and `core-summary.py` with its output. Its README
has the rest. `summarize.py` does not read it.

**This Engine on x0ne Core, streamed** (`2026-11-04-h100-tdx-cc-core-e2e/`).
A TEST measurement: VERIFIED with the TEST archive key and the TEST
transparency log; not a release measurement. `rb-bench.py`'s streams, every
SSE event timed on the client through the relay, and `rb-analyse.py`'s
summary; no receipt, signature or session key. Its README has the rest.
`summarize.py` does not read it.

## Directories

| Directory | Machine | What ran |
|---|---|---|
| `2026-10-17-h100-tdx-cc/` | GCP `a3-highgpu-1g`, us-central1-a | `qwen3:8b`, TD at 0.2.0-beta.5, 09:39 to 09:45 |
| `2026-10-18-h100-tdx-cc/` | GCP `a3-highgpu-1g`, us-central1-a | `qwen3:32b`, TD at 0.2.0-beta.7, 17:56 to 17:57 |
| `2026-10-22-h100-tdx-cc/` | GCP `a3-highgpu-1g`, us-central1-a | `qwen3:8b`, TD from 0.2.0-beta.7 sources of that day, 00:46 to 00:48 |
| `2026-10-28-h100-tdx-cc/` | GCP `a3-highgpu-1g`, us-east5-a | `ministral3:14b`, TD from 0.2.0-beta.7 sources of that day, 15:22 |
| `2026-10-31-h100-tdx-cc/` | GCP `a3-highgpu-1g`, us-east5-a | `ministral3:14b`, TD from 0.2.0-beta.7 sources of that day, image x0ne-cvm 0.2.8, RTMR0 pinned: one stream and the session KV cache 07:41 to 07:44, the prompt ladder 07:44 to 07:45 |
| `2026-11-03-h100-tdx-cc-core/` | GCP `a3-highgpu-1g`, us-east5-a, TDX, GPU in CC mode | x0ne Core alone, its bench and GPU tests, `qwen3:8b`, 23:30 to 23:40 |
| `2026-11-04-h100-tdx-cc-core-bench/` | GCP `a3-highgpu-1g`, us-east5-a, TDX, GPU in CC mode | x0ne Core alone, its per-token bench, `qwen3:0.6b`, `qwen3:8b`, `qwen3:14b` and `qwen3:32b`, 16:34 to 16:48 |
| `2026-11-04-h100-tdx-cc-core-e2e/` | GCP `a3-highgpu-1g`, us-east5-a, TDX, GPU in CC mode | this Engine on x0ne Core, image x0ne-cvm 0.2.9, `qwen3:8b` streamed through the relay, a TEST measurement, 15:23 to 15:30 |
| `prompts/` | | the frozen prompts, as text and as Qwen3 token ids |

### The H100 VMs

GCP `a3-highgpu-1g`, spot: one NVIDIA H100 80GB HBM3, an Intel Xeon of family
6 model 0x8f (serial console), 26 vCPU (Google's machine-type page), Intel TDX
guest with the GPU in confidential computing mode. Each run was a new VM, so
possibly another host. The image was a test build of x0ne-cvm
([hardware-runs.md](../../docs/hardware-runs.md) gives each one); the serial
console of the 2026-10-17 and 2026-10-22 runs shows NVIDIA's open kernel
module 595.91.07 loading (that of 2026-10-18 was lost, that of 2026-10-28 not
kept with these rows). The relay ran on the operator's machine, in a
container: built `--release` on 2026-10-17, `--release` with debug
assertions and `test-archive-key` on the other three days. Every response kept
here has a receipt the relay verified, naming `cuda:0` in a
`ConfidentialVmWithGpu` environment (`IntelTdx`, GPU mode `On`).

- `2026-10-17-h100-tdx-cc/`: client 0.2.0-beta.5, TD from the 0.2.0-beta.5
  sources of 2026-10-17, release relay. No `long4kp` request.
- `2026-10-18-h100-tdx-cc/`: client and TD at the 0.2.0-beta.7 release
  sources, image x0ne-cvm 0.2.3.
- `2026-10-22-h100-tdx-cc/`: client, relay and TD from the 0.2.0-beta.7
  sources of 2026-10-22 (the decode change included). `streams.jsonl`: eight
  streamed requests through the relay, summarised by the client (seconds to
  the first event and to the first content, gaps between content events,
  end): `s256-N`, the short prompt; `prefill-16000` and `prefill-32000`,
  prompts of 16,021 and 32,021 tokens with `max_tokens` 8; `conc-N`, three
  streams of 128 tokens sent at the same instant.
- `2026-10-28-h100-tdx-cc/`: client, relay and TD from the 0.2.0-beta.7
  sources of 2026-10-28, image x0ne-cvm 0.2.5.2, `ministral3:14b`. Ministral
  3's chat template makes the short prompt 535 tokens and the long one 4,436.
  This TD, and that of 2026-10-31, padded the time of a whole answer to a
  Padme number of 100 ms ticks, ten at least: `summarize.py` derives no
  decode rate from these two directories
  ([benchmarks.md](../../docs/benchmarks.md#method)).

## prompts/

`short`: "Explain how a CPU cache works." `4k`, `8k`, `16k`: "Here is a list
of records.", a blank line, then 180, 374 and 762 lines `Record <i:05d>: the
code of the <colour> <object> is <code>.`, a blank line, then the same
question. The lines come from a linear congruential generator, so they do
not depend on any library: from x = 12345, for each record x = (1103515245 x
+ 12345) mod 2^31, code = 1000 + x mod 9000, colour = COLOURS[(x >> 8) mod 8]
of red, blue, green, black, white, orange, purple, yellow, object =
OBJECTS[(x >> 16) mod 8] of warehouse, locker, garage, cabinet, vault,
office, archive, storage room. The record count is the largest whose
templated prompt stays within 4096, 8192 and 16384 tokens minus 256. The
template is Qwen3's with thinking off: `<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`. With Qwen3's chat template (Qwen3-0.6B and
Qwen3-8B share one tokenizer): 20, 3,827, 7,926 and 16,120 tokens.
`<name>.user.txt` is the user message; `<name>.ids.json` the token ids,
template applied, that x0ne Core's bench read on 2026-11-03. `SHA256SUMS`
lists both.
