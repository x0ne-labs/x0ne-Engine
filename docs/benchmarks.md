# Benchmarks

Every speed figure this repository keeps the rows for, all measured in CC
mode: the machine, the model, the build, the prompt, the number of requests,
the date, and the file it is computed from.
`python3 bench/results/summarize.py` recomputes every number on this page
from [`bench/results/`](../bench/results/README.md),
except those of x0ne Core alone and of this Engine on x0ne Core, which
their sections read from their own files.
Dates are UTC.

## Read this first

- **Every row ran in CC mode**: an NVIDIA H100 in an Intel TDX guest, the
  GPU in confidential computing mode. This page publishes no figure measured
  with confidential computing off.
- **Every row from 2026-10-17 to 2026-10-31 ran the engine before x0ne
  Core**, which this version replaces ([CHANGELOG](../CHANGELOG.md),
  2026-11-04). They are dated measurements of that engine, not figures of
  x0ne Core. The H100 figures of x0ne Core alone are in
  [their own section](#x0ne-core-alone-h100-in-cc-mode).
- **The rows of the engine before x0ne Core ran through the relay and the
  attested channel**, each response with a receipt the relay verified. They
  say how fast that Engine answered there. **The rows of x0ne Core alone**
  (2026-11-03 and 2026-11-04) come from its own benches, without this Engine: the speed of its
  arithmetic on the GPU, not that of this Engine's path (see
  [What was not measured](#what-was-not-measured)).
- **The H100 rows of this Engine on x0ne Core are a TEST measurement**
  (2026-11-04): VERIFIED with the TEST archive key and the TEST transparency
  log; not a release measurement. Read on the client's clock through the
  relay, they do not compare directly with x0ne Core's own bench.
- None of these figures changes what the README says under
  [What is left](../README.md#what-is-left) about the confidentiality of the
  Engine today, nor what [cloud.md](cloud.md) says about who holds the text
  where.
- One request at a time, BF16 weights, greedy decoding (temperature 0 or
  absent, [api.md](api.md)). No figure here is a throughput under load.
- `make bench` ([The suite](#the-suite-make-bench)) measures a local GPU
  without confidential computing. None of its numbers is on this page.

## Method

- **Decode rate** = (N - 1) / (t(N) - t(1)), where t(1) is the time to a whole
  answer of `max_tokens` 1 and t(N) the time to a whole answer of
  `max_tokens` N on the same prompt, both on the client's wall clock. The
  prompt, the HTTP round trip and the first token cancel out of the
  difference; what is left is N - 1 decode steps.
- **A padded answer gives no decode rate.** The TD pads the time of a whole
  answer: it leaves once a Padme number of 100 ms ticks has passed since its
  turn began, ten at least, so never within one second (the SECURITY entry
  on an answer's length in the [CHANGELOG](../CHANGELOG.md)). With this
  padding, t(1) and t(N) of a whole answer no longer measure decode: their
  difference is a difference of padded sizes. The H100 runs of 2026-10-28 and
  2026-10-31 ran such a TD: no whole answer took less than 1.197 s, and
  every t(N) - t(1) of their medians is a multiple of 100 ms within
  0.012 s. No decode rate is derived from them. The runs of 2026-10-17,
  2026-10-18 and 2026-10-22 ran without it: their one-token answers took
  0.140 to 0.175 s (medians), under the padding's floor of one second.
- **A row counts only if** the answer has exactly N tokens, finish `length`,
  and the prompt its expected token count: the Engine has no `ignore_eos`,
  so an answer that stops early is a different measure. Where a model stopped
  before N (the long prompt on the H100 on 2026-10-18), the rate is over
  the tokens it did generate, and the tables say how many.
- **First token** below is t(1): the whole one-token answer, not a streamed
  first byte, unless the column says "streamed".
- **n** is the number of pairs; the range is the lowest and highest pair, or
  `[min, max]` over rows.
- Through this Engine the client ran on the operator's machine: t(1)
  includes the relay and the round trip to the cloud region.

## x0ne Core alone, H100, in CC mode

Not this Engine: x0ne Core's own bench, `x0ne-core-bench`, in the process,
with no relay, no attested channel, no receipt and no padding. This Engine
on x0ne Core, streamed through the relay, has
[its own section](#this-engine-on-x0ne-core-h100-streamed-through-the-relay),
a TEST measurement.

GCP `a3-highgpu-1g` spot in us-east5-a, on 2026-11-03: an Intel TDX guest
with the GPU in CC mode, a bench image (not x0ne-cvm, without NVIDIA
attestation), the bench binary SHA-256 `c91a22ee…`, an H100 80GB HBM3 with
driver 595.91.07, `qwen3:8b` in BF16, and the prompts of
[`prompts/`](../bench/results/prompts/): `short`, the first 31 tokens of
`4k`, `4k`, the first 7,926 tokens of `16k`, and `16k`. Greedy, 256 tokens
generated, one warm-up of 8 tokens then three runs per prompt, one CUDA
graph per token. Decode = 255 / (t(256) - t(1)) on the host's clock in the
process, the prefill excluded; first token = t(1), prefill included. The
figure is the median of the three runs, the brackets their min and max.

| Prompt tokens | Decode | First token |
|---|---|---|
| 20 | 138.62 tok/s [138.60, 138.63] | 0.008 s |
| 31 | 138.56 tok/s [138.55, 138.57] | 0.008 s |
| 3,827 | 129.30 tok/s [129.30, 129.30] | 0.513 s |
| 7,926 | 122.22 tok/s [122.22, 122.26] | 1.954 s |
| 16,120 | 116.43 tok/s [116.43, 116.45] | 7.716 s |

One VM, three runs per prompt. The rows are in
[`2026-11-03-h100-tdx-cc-core`](../bench/results/2026-11-03-h100-tdx-cc-core/),
with the same bench without the CUDA graph and x0ne Core's GPU tests on
that VM. `summarize.py` does not read it.

## x0ne Core alone, H100, in CC mode, four models

On 2026-11-04, another VM of the same type, an Intel TDX guest with the GPU
in CC mode and the same bench image: x0ne Core's per-token bench,
`x0ne-core-itl`, in the process. `qwen3:0.6b`, `qwen3:8b`, `qwen3:14b` and
`qwen3:32b` in BF16, prompts of random token ids, greedy, 256 tokens
generated, one warm-up then five runs per length, one CUDA graph per token.
Decode = 255 / (t(256) - t(1)) on the host's clock in the process, the
prefill excluded; the figure is the mean of the five runs.

| Prompt tokens | `qwen3:0.6b` | `qwen3:8b` | `qwen3:14b` | `qwen3:32b` |
|---|---|---|---|---|
| 128 | 729.62 tok/s | 138.21 tok/s | 81.40 tok/s | 38.37 tok/s |
| 1,024 | 690.90 tok/s | 136.34 tok/s | 80.22 tok/s | 38.20 tok/s |
| 4,096 | 552.02 tok/s | 128.82 tok/s | 75.74 tok/s | 37.00 tok/s |
| 8,192 | 442.66 tok/s | 122.51 tok/s | 72.50 tok/s | 36.40 tok/s |
| 16,384 | 385.28 tok/s | 116.58 tok/s | 65.57 tok/s | 35.15 tok/s |

The first token, the prefill rate, the gaps between tokens and the 95 %
intervals are in
[`2026-11-04-h100-tdx-cc-core-bench`](../bench/results/2026-11-04-h100-tdx-cc-core-bench/README.md),
with the rows and the script that computes them. Its tool, prompts and
statistic are not those of the section above. `summarize.py` does not read
it.

## This Engine on x0ne Core, H100, streamed through the relay

**A TEST measurement: VERIFIED with the TEST archive key and the TEST
transparency log; not a release measurement.** On 2026-11-04, GCP
`a3-highgpu-1g` spot in us-east5-a, Intel TDX and the H100 in CC mode, the
image x0ne-cvm 0.2.9 (not published), the relay built from the sources of
that day with the TEST pins and `up` from those of 2026-10-31
([hardware-runs.md](hardware-runs.md)).
`qwen3:8b`, 36 streamed answers of 1,024 tokens (but a 16-token warm-up), one
at a time, every receipt verified by the relay. Each answer was timed frame
by frame on the client's clock, through the relay: decode = the tokens after
the first content frame over the time to the last one (a least-squares fit
gives the same within 0.84 %); first content = the first frame with text,
from the send. The TD sends one frame per 100 ms tick and holds the last one
until a Padme frame count: the first token is known to the tick only, the
end of the generation not at all. Medians of n = 7, brackets min and max.

| Prompt tokens | Decode | First content |
|---|---|---|
| 58 | 136.87 tok/s [136.73, 137.01] | 0.202 s [0.200, 0.208] |
| 548 | 135.49 tok/s [135.47, 135.76] | 0.208 s [0.204, 0.209] |
| 3,568 | 128.98 tok/s [128.74, 129.00] | 0.705 s [0.699, 0.708] |
| 7,468 | 122.65 tok/s [121.30, 122.83] | 2.098 s [2.093, 2.108] |
| 16,068 | 116.17 tok/s [115.92, 116.70] | 8.042 s [8.025, 8.155] |

Not the method of the section above, nor its clock: x0ne Core alone counts
255 tokens in its process, the prefill excluded; here the tokens after the
first frame, read on the client through the relay, on other prompts. This
page does not split the difference (138.62 tok/s after 20 prompt tokens
there, 136.87 after 58 here). Rows and method:
[`2026-11-04-h100-tdx-cc-core-e2e`](../bench/results/2026-11-04-h100-tdx-cc-core-e2e/README.md).

## H100, Intel TDX, GPU in CC mode, before x0ne Core

GCP `a3-highgpu-1g`, spot: one NVIDIA H100 80GB HBM3, 26 vCPU of a Xeon
Sapphire Rapids (family 6, model 0x8f), an Intel TDX guest booted from a test
build of x0ne-cvm, the GPU in confidential computing mode. Each run is a new
VM, so possibly another host. [hardware-runs.md](hardware-runs.md) records the
attestation side of each run; this table is its speed side.

| Date | Zone | Model | Engine in the TD | n | Decode, 20-token prompt, 256 tokens | Decode after the long prompt | First token, 20-token prompt | First token, long prompt | Rows |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-17 | us-central1-a | `qwen3:8b` | 0.2.0-beta.5 sources | 3 | 42.51 tok/s [42.48, 42.56] | not isolated | 0.140 s | not measured | [`2026-10-17-h100-tdx-cc`](../bench/results/2026-10-17-h100-tdx-cc/) |
| 2026-10-18 | us-central1-a | `qwen3:32b` | 0.2.0-beta.7 release sources | 3 | 18.66 tok/s [18.60, 18.66] | 17.55 tok/s over 91 tokens [17.49, 17.57] | 0.175 s | 2.326 s | [`2026-10-18-h100-tdx-cc`](../bench/results/2026-10-18-h100-tdx-cc/) |
| 2026-10-22 | us-central1-a | `qwen3:8b` | 0.2.0-beta.7 sources of that day, decode change included | 10 | 71.17 tok/s [70.54, 74.96] | 66.96 tok/s over 128 tokens [64.28, 68.85] | 0.145 s | 0.908 s | [`2026-10-22-h100-tdx-cc`](../bench/results/2026-10-22-h100-tdx-cc/) |
| 2026-10-28 | us-east5-a | `ministral3:14b` | 0.2.0-beta.7 sources of that day | 3 | not derived: padded times ([Method](#method)) | not derived: padded times | 1.203 s, padded | 1.296 s, padded | [`2026-10-28-h100-tdx-cc`](../bench/results/2026-10-28-h100-tdx-cc/) |
| 2026-10-31 | us-east5-a | `ministral3:14b` | 0.2.0-beta.7 sources of that day, image x0ne-cvm 0.2.8 | 3 | not derived: padded times ([Method](#method)) | not run: a prompt ladder instead, below | 1.200 s, padded | not run | [`2026-10-31-h100-tdx-cc`](../bench/results/2026-10-31-h100-tdx-cc/) |

Decode on the medians of t(1) and t(N); the brackets, pair by pair. The
short prompt is "Explain how a CPU cache works." (20 tokens with Qwen3's
template, 535 with Ministral 3's). The long prompt is `"Here is a list. " +
" ".join(["apple"] * 3900) + " How many items?"`: 3,921 tokens with Qwen3,
4,436 with Ministral 3, asked for 129 tokens.

- 2026-10-17: no one-token request on the long prompt, so its decode rate
  cannot be separated from its prefill (three answers of 129 tokens in 3.929
  to 3.961 s).
- 2026-10-18: `qwen3:32b` stopped on its own at 92 tokens on the long prompt.
  The 3,901 extra prompt tokens took 2.151 s more than the short prompt's one
  token.
- 2026-10-22: the extra 3,901 prompt tokens took 0.763 s more. The client's
  machine carried other work (load average 21 to 36), which the wider range
  may reflect; not established.
- 2026-10-28: `ministral3:14b` stopped on its own at 30 tokens on the long
  prompt. Its one-token answer took 1.20 s on the 535-token prompt and 1.30 s
  on the 4,436-token one: both are padded times ([Method](#method)), so
  neither a decode nor a prefill rate is derived from them.
- 2026-10-31: no long prompt; a ladder of prompts of 1,048, 2,048, 4,048
  and 8,048 tokens instead, each asked for 1 and 129 tokens, n = 3; every
  answer of 129 tokens, finish `length`, every receipt verified by the relay
  (31 of 31 requests). The one-token answer took 1.203, 1.214, 1.302 and
  2.810 s (medians): padded times, as are those of the 129-token answers, so
  no decode rate is derived from them ([Method](#method)).
- `ministral3:14b` is no longer served ([CHANGELOG](../CHANGELOG.md),
  2026-11-04); its rows stay as dated measurements.
- The Engine in the TD is not the same from one run to the next. The 42.51
  and 71.17 tok/s of `qwen3:8b` come from different sources (the 2026-10-22
  ones include the decode change of 2026-10-21, CHANGELOG) on different VMs;
  this page does not split the difference between those two causes.

Streamed through the relay on 2026-10-22 (`qwen3:8b`, `streams.jsonl`, one
request each unless said):

| Request | Prompt tokens | Completion tokens | First content | End |
|---|---|---|---|---|
| short prompt, 256 tokens, n = 3 | 20 | 256 | 0.219, 0.232, 0.312 s | 3.843 to 3.973 s |
| a 16,021-token prompt | 16,021 | 8 | 7.774 s | 8.195 s |
| a 32,021-token prompt | 32,021 | 8 | 27.618 s | 28.148 s |
| three streams sent at once | 20 each | 128 each | 0.222, 2.042, 4.158 s | 2.142, 4.385, 6.093 s |

The three streams sent at once were not decoded together: their first
contents came 1.8 and 2.1 s apart, about the time one 128-token answer takes.

Streamed through the relay on 2026-10-31 (`ministral3:14b`, `streams.jsonl`
of that day, n = 1): a 538-token prompt, its first content at 0.211 s, its
end at 2.312 s, 23 frames, finish `stop`, the receipt verified on the final
chunk; the same question answered whole took 2.216 s and gave the same text
(the run's record, kept outside this repository).
The session KV cache measured on that run, a second turn's first content at
0.321, 0.421 and 0.472 s continued against 1.114, 3.028 and 8.872 s without
a conversation (n = 3): [session-kv-cache.md](session-kv-cache.md#measurements),
rows in `2026-10-31-h100-tdx-cc/kv-cache.jsonl`.
Every stream ended with a receipt the relay verified.

The same protocol on this version, which serves with x0ne Core and no other
engine, has not run on hardware (a billed GCP VM; [cloud.md](cloud.md)); the
run of 2026-11-04 streamed instead
([its section](#this-engine-on-x0ne-core-h100-streamed-through-the-relay)):

```sh
x0ne-engine cloud up --zone us-central1-a --model qwen3:8b --reference x0ne-cvm-<version>.json
K=$(cat "$XDG_RUNTIME_DIR/x0ne-engine/local.key")
body() { jq -nc --arg m qwen3:8b --arg c "$1" --argjson n "$2" '{model:$m,messages:[{role:"user",content:$c}],max_tokens:$n}'; }
req() { curl -s -o "$1.json" -w "$1: %{time_total} s, HTTP %{http_code}\n" -H "Authorization: Bearer $K" \
  -H 'Content-Type: application/json' http://127.0.0.1:8478/v1/chat/completions -d "$2" >> times.log; }
for i in 1 2 3; do req c1-$i "$(body 'Explain how a CPU cache works.' 1)"; req c256-$i "$(body 'Explain how a CPU cache works.' 256)"; done
long=$(python3 -I -c 'print("Here is a list. " + " ".join(["apple"] * 3900) + " How many items?")')
for i in 1 2 3; do req long4kp-$i "$(body "$long" 1)"; req long4k-$i "$(body "$long" 129)"; done
```

`times.log` is then in the format of the rows; `responses.jsonl` keeps, from
each answer, `usage`, `finish_reason` and the receipt fields named in
[bench/results/README.md](../bench/results/README.md).

## What was not measured

- **Speed with confidential computing off.** This page publishes runs in CC
  mode only, and gives no cost of confidential computing.
- **A release build of this Engine on x0ne Core on the H100.** A release
  relay waits for the signing ceremony ([hardware-runs.md](hardware-runs.md));
  the run of 2026-11-04 pinned the TEST keys.
- **Other engines** on the same machines.
- **Concurrency and throughput.** Every figure is one request at a time. The
  three streams sent at once on 2026-10-22 were not decoded together.
  No batching, no queue under load, no tokens per second across requests.
- **The same model across runs on the H100.** Through the Engine,
  `qwen3:32b` ran once, n = 3, on one VM; `qwen3:8b` on three (2026-10-17,
  2026-10-22, 2026-11-04) and `ministral3:14b` on two, with different Engine
  sources.
- **The decode rate of `ministral3:14b` on the H100**, a model no longer
  served. Its two runs
  (2026-10-28 and 2026-10-31) answered through a TD that pads the time of a
  whole answer, and their times do not give it ([Method](#method)).
- **Long prompts on the H100** beyond one 16,021-token and one 32,021-token
  stream (n = 1 each) and the 16,590-token turns of the session KV cache
  measure (n = 3); whole answers beyond 8,048 prompt tokens;
  for x0ne Core alone, beyond 16,384 tokens; through this Engine
  on x0ne Core, beyond 16,068 tokens.
- **Quantized weights** (everything here is BF16), and any model other than
  Qwen3-0.6B, Qwen3-8B, Qwen3-14B, Qwen3-32B and Ministral 3 14B.
- **Start-up**: model load, VM boot, attestation, key release. A trace
  only: x0ne Core's benches, by their own count, loaded `qwen3:8b` in 3.6 s
  and `qwen3:32b` in 12.3 s.
- **Energy, memory footprint, GPU utilisation.**
- **Sampling** other than greedy, and the quality of the answers: the rows
  only check that a build gives the same text to the same prompt.
- **A published release.** Every Engine here was built from sources at the
  dates given, with versions 0.2.0-beta.5 to 0.2.0-beta.7; none is a tag.

## The suite: make bench

`make bench` (`scripts/bench.sh`) measures this Engine on the local NVIDIA
GPU, tier by tier, and writes a verdict, `out/bench/<sha>.json`, and its
report, `out/bench/<sha>.md` (`<sha>+dirty` for a tree that is not its
commit's; `latest.json` and `latest.md` point at the last). Its GPU runs
without confidential computing, so no figure it measures goes on this page:
they are the numbers of one machine and one run, said with its GPU, driver,
host, runner, image and settings in the verdict.

The tiers (`BENCH_STEPS`, default all of them, in this order):

- **build**: x0ne-infer's lib test binary (release, `--features cuda`) and
  x0ne Core's `core-bench`, in a container with nvcc. Nothing on the card.
- **core**: x0ne Core alone (`x0ne-core/oracle/examples/core-bench.rs`), per
  model: for each length (`short`, `1k`, `4k`, `8k`, `16k`, and `max`, the
  largest context the card holds; `short`, `1k`, `4k` for a mixture of
  experts) the time to the first token and the prefill and decode rates of
  `Engine::greedy_timed`; a cold load and a second one; the arena and the
  device memory taken. A context the card refuses for memory is tried
  smaller, and written down. The ids are the prompt kit's `<name>.ids.json`
  (the chat template applied; `1k` is the first 1024 of `4k`), or drawn from
  a fixed seed for `max` and a length without a file.
- **engine**: the kit's chat prompts (`<name>.chat.json`) through the server
  of x0ne-infer's unit tests on x0ne Core (`bench_tests::gpu::engine`),
  streamed and whole, at the production tick (100 ms) and at the tests' short
  tick (5 ms). The report gives the stream's time to its first token minus
  core's for the same ids: HTTP, the chat template and the tokenizer,
  admission, and the wait for the next tick.
- **load**: 1, 2, 4, 8, 16 and 32 clients, 30 s each, each sending again as
  soon as its answer ends, against one server that lets 8 wait and answers
  503 with `Retry-After` past them (`bench_tests::gpu::load`): time to the
  first token and response time at p50, p95 and p99 (nearest rank), tokens
  per second over all answers, the share of 503 and their `Retry-After`,
  Jain's index of the answers per client, and FIFO inversions (pairs of
  requests that waited together and whose first tokens came in the other
  order than their sending).
- **soak**: 4 clients for 10 minutes (`BENCH_SOAK_MIN`, 60 possible):
  throughput per 60 s window, its drift (the last window against the
  first), the least-squares slope through every window per hour, the
  process's resident memory and the device memory at each window's end, and
  errors.
- **baseline**: llama.cpp's `llama-server` on core's ids
  (`bench_tests::baseline_llama`, `BENCH_LLAMA_SERVER`,
  `BENCH_GGUF_<MODEL>`). A comparison, never a reference: the report sets
  its rates beside core's only when the GGUF was made from the same model
  directory (`BENCH_GGUF_<MODEL>_FROM`), is BF16, core decoded as many
  tokens, and a length has the same ids in both. Otherwise it says why not.

Each number of core and engine is the median of `BENCH_REPS` (5)
repetitions after a first pass reported apart, with the smallest, the
largest, the mean and the standard deviation; 256 tokens decoded
(`BENCH_DECODE`), greedy. Every program prints one line
`x0ne-bench-result: <JSON>` (schema 1, `kind` its tier); `scripts/bench.sh`
keeps it whole in its run of the verdict.

**The verdict.** A run (a tier and a model) is `ok`, `fail` or `skipped`,
with its reason. A model not given is skipped with the variable it needs; a
model the card cannot hold is skipped, "refused on this card"; a model the
server cannot serve (a mixture of experts, no `tokenizer.json`) is skipped
for the server's tiers. The status is `red` (a run failed, or the card was
lost; exit 1), `partial` (a run skipped, or measured without a reference;
exit 3, never green) or `green` (exit 0); 2 is a usage error. The repetitions
of core giving other tokens, an error of the server, or a 503 without
`Retry-After` fail their run whatever the numbers.

**The reference.** `ci/bench-baseline.json` holds, per GPU name, the metrics
of a recorded run. A rate (a metric named `*_tok_s` or `*_per_s`) lower, or a
time higher, than its reference by more than its tier's `tolerance_pct`
fails the run, reason "below the reference". `scripts/bench.sh update
VERDICT` records a clean run whose metrics hold, keeping the better of the
old and the new value of each; `--replace` writes the run as it is. The
tolerances (5 % for core, 10 % for engine, 15 % for load and soak) are
placeholders until the first run on the card: to be set from the spread it
shows.

**Not measured by the suite:**

- TDX: no trust domain, no quote, no private memory. The server is the unit
  tests' one, with their proof; its client is in the same process, over
  loopback. The relay, the attested channel and the network between a
  client and the server are not in any of its numbers.
- Confidential computing on the GPU.
- The decode rate of a stream is from its first content to its last, and a
  tick quantizes it: each frame carries what came during its tick.
- The resident memory of the soak is that of the test process: the server,
  the host side of the model and the clients together. The device memory is
  the whole device's, as `cuMemGetInfo` gives it, the reading's own CUDA
  context included.
- Sampling other than greedy, quantized weights, and the quality of the
  answers.
