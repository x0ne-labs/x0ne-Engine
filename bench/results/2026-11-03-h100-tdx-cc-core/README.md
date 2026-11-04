# 2026-11-03, H100 in a TDX VM with the GPU in CC mode, x0ne Core alone

x0ne Core's own bench, `x0ne-core-bench`, and five of its GPU test files, on
a GCP `a3-highgpu-1g` in an Intel TDX guest with the GPU in confidential
computing mode. Public weights only, no client data. Not this Engine: no
relay, no attested channel, no receipt, no model key. A bench image, not
x0ne-cvm: its TD quote was checked for the run's nonce, the TDX type and the
DEBUG bit only, and the GPU was set ready in CC mode without NVIDIA
attestation. These are the rows of x0ne Core's H100 figures.

GCP `a3-highgpu-1g` spot, us-east5-a, the run from 23:25 to 23:42 UTC, the
VM 14.0 minutes, the measures from 23:30 to 23:40: one NVIDIA H100 80GB HBM3
(81,559 MiB, driver 595.91.07, CUDA 13.2), 26 vCPU (family 6, model 0x8f;
the guest's CPUID names it `Intel TDX`), Linux 6.12 (Debian 13).
`systemd-detect-virt --cvm` said `tdx`; `nvidia-smi conf-compute -f` said
`CC status: ON` before the measures (`m1-conf-compute.txt`) and after them
(`end-conf-compute.txt`). The bench image `x0ne-cvm-bench` 0.2, without
NVIDIA attestation; `x0ne-core-bench` SHA-256 `c91a22ee…`; x0ne Core's sources of that day, not the copy in
[`x0ne-core/`](../../../x0ne-core/PROVENANCE.md)
([hardware-runs.md](../../../docs/hardware-runs.md) says how they differ).
The figures are in
[benchmarks.md](../../../docs/benchmarks.md#x0ne-core-alone-h100-in-cc-mode).
`summarize.py` does not read this directory.

## Method

The model: `qwen3:8b`, Qwen/Qwen3-8B at the catalog's revision, BF16,
downloaded in the VM, 16,381,561,725 bytes in 63 s, each file checked
against the size and SHA-256 the run pinned before any load
(`download.tsv`).

The CC probe (`m1-ccprobe.json`), before the model: a pinned host
allocation, launches, 4-byte and 256 MiB copies, pageable and pinned, a
BF16 GEMM and an HBM read, five repetitions each, in CC mode.

The bench, one process per call, `--gpu h100-sxm`.
`c1-start`: 20 prompt tokens, 16 generated, one run. `c2-graph`: the
prompts of [`prompts/`](../prompts/) as token ids, the files
`prompts/SHA256SUMS` lists: `s20` (`short`, 20 tokens), `s31` (the first
31 tokens of `4k`), `4k` (3,827), `8k` (the first 7,926 tokens of `16k`,
not the `8k` prompt) and `16k` (16,120); greedy, 256 tokens generated,
three runs each, one CUDA graph replayed per token. `c2-eager`: `s20` and
`4k` the same way, without the graph (`--decode eager`). Before the runs
of a prompt, one warm-up of 8 tokens; the bench stops if a run's tokens
differ from the first run's, or their first 8 from the warm-up's.
Decode = 255 / (t(256) - t(1)), on the host's monotonic clock in the
process, each token timed when it reaches the host: the prefill is not in
it. First token = t(1) from the call, prefill included. The figure is the
median of the three runs. `ceiling` and `%` in the `.txt` files are the
bench's bound from the bytes read per token over the H100 SXM's datasheet
bandwidth (3.35 TB/s): computed, not measured.

The tests, `cargo test` binaries of x0ne Core built for the run, on
`qwen3:8b`: `c3-api` (`api_model_8b`, the session route), `c4-graph`
(`e2_graph_8b`, the CUDA graph against the step-by-step run), `c4-fresh`
(`e2_fresh_8b`), `c5-model` (`e1_model_8b`, prefill and decode against
float32 reference logits) and `c6-long` (`e3_model_8b`, the split-K decode
attention against the per-head one).

## Results

| Prompt | Prompt tokens | Decode, median [min, max] | First token |
|---|---|---|---|
| `s20` | 20 | 138.62 tok/s [138.60, 138.63] | 0.008 s |
| `s31` | 31 | 138.56 tok/s [138.55, 138.57] | 0.008 s |
| `4k` | 3,827 | 129.30 tok/s [129.30, 129.30] | 0.513 s |
| `8k` | 7,926 | 122.22 tok/s [122.22, 122.26] | 1.954 s |
| `16k` | 16,120 | 116.43 tok/s [116.43, 116.45] | 7.716 s |

- Without the graph (`c2-eager`): 108.55 tok/s [108.54, 108.61] at 20
  tokens and 107.46 [107.40, 107.52] at 3,827; the graph's rates are 1.28
  and 1.20 times these.
- `c2-graph` loaded the model in 3.6 s, with an arena of 19,210,968,096
  bytes and a decode graph of 439 nodes. `c1-start`: 138.96 tok/s over 16
  tokens, loaded in 3.6 s.
- The CC probe: `pinned_host` `granted`; over 256 MiB, pinned copies at
  7.05 GB/s to the GPU and 7.23 GB/s from it, pageable ones at 7.59 and
  6.27; a BF16 GEMM at 800.9 TFLOPS and an HBM read at 3,093 GB/s
  (medians of five).
- The tests: 33 passed, 0 failed (`core.tsv`, the `.txt` files). From their
  `MEASURE` lines: the session route's greedy tokens equal the engine's on
  the five prompts (256 of 256), and its logits after 0, 1 and 16 steps
  are identical (0 of 151,936 values differ at 31 and 3,827 tokens);
  against the float32 reference logits of `short` and `4k` (84 and 86
  rows), a mean KL of 9.56e-5 and 4.34e-5, top-1 agreement 1.000 and
  0.994; the graph's decode
  logits equal the step-by-step ones (84 rows, 0 differ); the free device
  memory unchanged over 1,000 requests and over 1,000 sessions; a request
  after 18 others gives the output of a fresh process (the same SHA-256).
  The backtrace in `c3-api.err` is the test `x3_scrub_on_every_path`
  panicking inside a session on purpose; the test passed.
- Split-K against per-head attention (`c6-long`): the same greedy tokens
  at 31, 3,827 and 7,926 tokens; at 20 and 16,120 the two part at token
  171 and 208, where the per-head route's top-2 margin was 0.014 and
  0.032. All five tests of the file passed.
- `xid.txt` is empty: `dmesg` had no Xid, NVRM error or "fallen off" line
  at the end of the run.

## Not measured

This Engine on x0ne Core: its relay, its attested channel, its receipts and
the padding of an answer's time did not run, and the cost of confidential
computing on that path is not measured. NVIDIA attestation of the GPU.
Other models, sampling, concurrency, other prompt lengths, MDWE under CC.

## Files

- `m1-conf-compute.txt`, `end-conf-compute.txt`: `nvidia-smi conf-compute
  -f` before and after the measures.
- `m1-ccprobe.json`: the CC probe. Its `gpu.uuid` field is replaced by
  `withheld`, so its line is not in `SHA256SUMS`.
- `download.tsv`: per file of the model, its check, its size and its
  download time.
- `c1-start.*`, `c2-graph.*`, `c2-eager.*`: the bench's output (`.txt`),
  its load line and one line per run (`.err`), its medians (`.tsv`: label,
  prompt tokens, tokens generated, tok/s).
- `c3-api.*`, `c4-graph.*`, `c4-fresh.*`, `c5-model.*`, `c6-long.*`: each
  test binary's report (`.txt`) and its `MEASURE` lines (`.err`).
- `core.tsv`: one line per bench or test call: the time (UTC), the exit
  code, the seconds, the test result, and `nvidia-smi`'s temperature, SM
  and memory clocks, power, memory used and throttle reasons before
  (`gpu0`) and after (`gpu1`) it.
- `xid.txt`: the lines of `dmesg` naming an Xid, an NVRM error or a GPU
  fallen off the bus, at the end of the run.
- `phases.log`: one line per step of the run in the VM, its exit code and
  seconds (`download-status` 51: the download was still running).
- `SHA256SUMS`: the lines of the run's own list for these files.
  `phases.log` is not in it: the run wrote its last line after the list,
  whose sum is that of the file without that line.

Not kept: the VM's manifest, which names the GPU's serial number and UUID,
the GPU monitor's log, whose name is the GPU's UUID, the VM's `dmesg` and
journal, the TD quote, and the run's log outside the VM, which holds the
quote's check.

Estimated cost 1.32 USD for the VM (14.0 minutes at 5.6482 USD/h on Cloud
Billing's catalog, confidential computing included, read at the run), at
most 0.02 USD more for the bench image's import and its transit bucket.
