# 2026-11-04, H100 in a TDX VM with the GPU in CC mode, x0ne Core alone, four models

x0ne Core's per-token bench, `x0ne-core-itl`, in the process, on a GCP
`a3-highgpu-1g` in an Intel TDX guest with the GPU in confidential computing
mode: four Qwen3 models from 0.6B to 32B, five prompt lengths from 128 to
16,384 tokens, 256 tokens generated. Public weights only, no client data.
Not this Engine: no relay, no attested channel, no receipt, no model key, no
padding of the answer's time. The product path ran the same day on another
VM, `qwen3:8b` only:
[`2026-11-04-h100-tdx-cc-core-e2e/`](../2026-11-04-h100-tdx-cc-core-e2e/README.md).
None of it changes what the README says under
[What is left](../../../README.md#what-is-left) about the confidentiality of
the Engine today. `summarize.py` does not read this directory.

GCP `a3-highgpu-1g` spot, us-east5-a, the VM from 16:27 to 16:52 UTC, the
measures of x0ne Core from 16:34 to 16:48: one NVIDIA H100 80GB HBM3
(81,559 MiB, VBIOS 96.00.D9.00.01, driver 595.91.07, CUDA 13.2), 26 vCPU
(13 cores, 2 threads each; the guest's CPUID names it `Intel TDX`, family 6,
model 0x8f), 229 GiB of memory, Linux 6.12.111 (Debian 13).
`systemd-detect-virt --cvm` said `tdx`; the kernel logged "Memory Encryption
Features active: Intel TDX" and DMA through SWIOTLB bounce buffers (1,024 MB);
`nvidia-smi conf-compute -f` said `CC status: ON`, `-grs` said `ready`, and
`nvidia-smi -q` counted 80,609 MiB of CC-protected memory, all before the
measures. The run was interrupted (exit code 130) at 16:51 UTC, during the
VM's other measurements and after the last phase of x0ne Core (16:47:41): the
end-of-run records (`conf-compute -f` after the measures, the kernel log, the
Xid check) were not taken.

A bench image, not x0ne-cvm: the build of
[`2026-11-03-h100-tdx-cc-core/`](../2026-11-03-h100-tdx-cc-core/), the same
disk SHA-256 pinned by both runs, its root hash read on this VM's serial
console. How it differs from the product image, from its build files in
x0ne OS and from this run:

- It is not signed and has no reference: its own description reads
  "benchmark VM: not attested, not a product image". The TD quote was checked
  by the run for its nonce, the TDX type and the DEBUG bit only, not against
  measurements.
- The GPU was set ready in CC mode by the image's own unit,
  `nvidia-smi conf-compute -srs 1`, without NVIDIA attestation. Nothing in the
  run asked for or checked the GPU's attestation report.
- It leaves out of the product image (x0ne OS
  `cvm-bench/product-files.txt`) `x0ne-infer` and its unit, the product's
  GPU-ready, TSM report, CCEL, cloud config, status and boot report units,
  the models mount, the product's nftables rules, and the reference and
  event-log scripts.
- It adds sshd for root, with one key from the instance metadata and only
  from the sources listed there; the VM's local NVMe disks as scratch, where
  the weights were written; the CUDA runtime libraries; `nvidia-smi dmon`
  from boot. x0ne Core and its bench were copied in over ssh.
- Its kernel, NVIDIA driver and Debian snapshot are those of the product
  image built from the same x0ne OS sources (2026-10-17). It was not compared
  with x0ne-cvm 0.2.9, the image of the product run.

## Method

The models: Qwen/Qwen3-0.6B, -8B, -14B and -32B at the revisions of the
catalog (`qwen3:0.6b`, `qwen3:8b`, `qwen3:14b`, `qwen3:32b` in
[`models.toml`](../../../x0ne-infer/catalog/models.toml)), BF16 as published
(`torch_dtype` of each pinned `config.json`, read on 2026-11-04), downloaded
in the VM, 112,945,987,308 bytes in 235 s. Each of the 42 files
was checked against the size and SHA-256 the catalog pins before any load
(`download.tsv`).

The tool: `x0ne-core-itl` ([`x0ne-core-itl.rs`](x0ne-core-itl.rs)), a bench
program on x0ne Core's library, written for this bench and not part of
x0ne Core. The library is x0ne Core's sources of 2026-10-31, those of the
2026-11-03 bench, not the copy in
[`x0ne-core/`](../../../x0ne-core/PROVENANCE.md), built with CUDA 13.2
(nvcc 13.2.78) and rustc 1.98.1, with cubins for `sm_90a` (the H100's),
`sm_100a` and `sm_120a` and no PTX. One process per model, in the
order 8B, 32B, 0.6B, 14B, with `X0NE_CORE_THREADS=8`. Greedy decoding, one
request at a time, one CUDA graph replayed per token, a context of the
longest prompt plus 256 tokens.

The prompts: [`prompts/`](prompts/), one file per length L (128, 1,024,
4,096, 8,192, 16,384), each six lists of L random token ids in
[0, 151643), seed 20261104 (`PROMPTS.txt`; the generator is not kept), no two
with the same first id (checked on 2026-11-04). Random ids, not text. The
first prompt of each file is the warm-up, the five others are measured, one
run each.

The clock: x0ne Core's `greedy_timed` returns, for each of the 256 tokens,
the time from the call at which it was back on the host, on the process's
monotonic clock. From these times, per run:

- first token = t(1), the prefill included;
- decode = 255 / (t(256) - t(1)), the prefill excluded;
- last token = t(256), from the call;
- the gaps t(i + 1) - t(i), 255 per run, the five runs pooled (1,275 gaps):
  p50, p90 and p99, by linear interpolation;
- prefill = prompt tokens / mean first token: it includes the first output
  step and the call.

Each figure is the mean of the five runs, ± the half-width of a 95 %
interval over them (Student's t with 4 degrees of freedom). `core-summary.py`
computes them from the JSON lines. To the digits printed, they equal those
the bench's own statistics script gave after the run (checked on
2026-11-04).

## Results

n = 5 for each row, the warm-up excluded; every run generated 256 tokens.

`qwen3:0.6b` (`core-q06-ccon.jsonl`), loaded in 1.1 s, an arena of 3,467,809,824 bytes, a decode graph of 315 nodes:

| Prompt tokens | Decode | First token | Prefill | Gap p50 / p90 / p99 | Last token |
|---|---|---|---|---|---|
| 128 | 729.62 ± 0.47 tok/s | 3.52 ± 0.03 ms | 36,318 tok/s | 1.374 / 1.387 / 1.393 ms | 0.353 ± 0.000 s |
| 1,024 | 690.90 ± 0.09 tok/s | 17.40 ± 0.01 ms | 58,851 tok/s | 1.449 / 1.459 / 1.465 ms | 0.386 ± 0.000 s |
| 4,096 | 552.02 ± 0.56 tok/s | 209.23 ± 1.94 ms | 19,576 tok/s | 1.811 / 1.819 / 1.826 ms | 0.671 ± 0.002 s |
| 8,192 | 442.66 ± 0.54 tok/s | 798.65 ± 3.34 ms | 10,257 tok/s | 2.260 / 2.267 / 2.274 ms | 1.375 ± 0.004 s |
| 16,384 | 385.28 ± 0.15 tok/s | 3,147.70 ± 0.31 ms | 5,205 tok/s | 2.595 / 2.601 / 2.608 ms | 3.810 ± 0.000 s |

`qwen3:8b` (`core-q8b-ccon.jsonl`), loaded in 3.6 s, an arena of 19,249,896,480 bytes, a decode graph of 439 nodes:

| Prompt tokens | Decode | First token | Prefill | Gap p50 / p90 / p99 | Last token |
|---|---|---|---|---|---|
| 128 | 138.21 ± 0.02 tok/s | 8.90 ± 0.01 ms | 14,378 tok/s | 7.243 / 7.263 / 7.272 ms | 1.854 ± 0.000 s |
| 1,024 | 136.34 ± 0.03 tok/s | 55.04 ± 0.81 ms | 18,604 tok/s | 7.338 / 7.350 / 7.369 ms | 1.925 ± 0.001 s |
| 4,096 | 128.82 ± 0.01 tok/s | 567.74 ± 4.52 ms | 7,215 tok/s | 7.755 / 7.798 / 7.812 ms | 2.547 ± 0.004 s |
| 8,192 | 122.51 ± 0.00 tok/s | 2,064.09 ± 4.23 ms | 3,969 tok/s | 8.158 / 8.167 / 8.194 ms | 4.146 ± 0.004 s |
| 16,384 | 116.58 ± 0.04 tok/s | 7,940.30 ± 3.62 ms | 2,063 tok/s | 8.572 / 8.583 / 8.627 ms | 10.128 ± 0.004 s |

`qwen3:14b` (`core-q14b-ccon.jsonl`), loaded in 5.9 s, an arena of 32,700,607,776 bytes, a decode graph of 447 nodes:

| Prompt tokens | Decode | First token | Prefill | Gap p50 / p90 / p99 | Last token |
|---|---|---|---|---|---|
| 128 | 81.40 ± 0.01 tok/s | 14.73 ± 0.02 ms | 8,691 tok/s | 12.286 / 12.306 / 12.316 ms | 3.148 ± 0.000 s |
| 1,024 | 80.22 ± 0.02 tok/s | 88.09 ± 1.10 ms | 11,624 tok/s | 12.460 / 12.496 / 12.516 ms | 3.267 ± 0.001 s |
| 4,096 | 75.74 ± 0.00 tok/s | 827.46 ± 0.33 ms | 4,950 tok/s | 13.084 / 13.381 / 13.396 ms | 4.194 ± 0.000 s |
| 8,192 | 72.50 ± 0.01 tok/s | 2,931.65 ± 0.82 ms | 2,794 tok/s | 13.790 / 13.799 / 13.812 ms | 6.449 ± 0.001 s |
| 16,384 | 65.57 ± 0.01 tok/s | 11,076.25 ± 4.34 ms | 1,479 tok/s | 15.233 / 15.310 / 15.325 ms | 14.965 ± 0.004 s |

`qwen3:32b` (`core-q32b-ccon.jsonl`), loaded in 12.3 s, an arena of 70,359,039,520 bytes, a decode graph of 647 nodes:

| Prompt tokens | Decode | First token | Prefill | Gap p50 / p90 / p99 | Last token |
|---|---|---|---|---|---|
| 128 | 38.37 ± 0.00 tok/s | 31.70 ± 0.06 ms | 4,038 tok/s | 26.068 / 26.095 / 26.110 ms | 6.677 ± 0.000 s |
| 1,024 | 38.20 ± 0.01 tok/s | 210.49 ± 0.84 ms | 4,865 tok/s | 26.179 / 26.206 / 26.226 ms | 6.887 ± 0.002 s |
| 4,096 | 37.00 ± 0.00 tok/s | 2,032.02 ± 0.98 ms | 2,016 tok/s | 27.120 / 27.157 / 27.172 ms | 8.924 ± 0.001 s |
| 8,192 | 36.40 ± 0.00 tok/s | 7,274.90 ± 2.24 ms | 1,126 tok/s | 27.468 / 27.484 / 27.499 ms | 14.280 ± 0.002 s |
| 16,384 | 35.15 ± 0.00 tok/s | 27,386.71 ± 4.27 ms | 598 tok/s | 28.449 / 28.476 / 28.501 ms | 34.642 ± 0.004 s |

- From 128 to 16,384 prompt tokens, decode fell by 47.2 % on the 0.6B,
  15.6 % on the 8B, 19.4 % on the 14B and 8.4 % on the 32B.
- The runs of a row agree closely: the 95 % half-width of decode is at most
  0.56 tok/s on the 0.6B and 0.04 tok/s on the other three; that of the first
  token at most 4.52 ms.
- What the gaps measure: the host launches up to 15 steps ahead of the
  tokens it reads (a ring of 16 in x0ne Core) and times a token when it
  reads it. In each of the 100 measured runs, the gap after the first token
  spans 14.8 to 15.1 times the median of the 240 gaps that follow it, and
  the last 14 gaps are under half that median, down to 0.001 ms; none of
  the 240 exceeds 1.23 times it. Over those 240 alone, p50, p90 and p99 move
  by 0.54 % at most. Decode and the last token use only the first and
  the last time.
- `nvidia-smi` after each model's process: SM clock 1,980 MHz, memory clock
  2,619 MHz, no active throttle reason, 44 to 45 °C, 242 to 324 W
  (`core-runs.tsv`).

## Not measured

This Engine on x0ne Core: its relay, its attested channel, its receipts and
the padding of an answer's time did not run here. NVIDIA attestation of the
GPU, and the TD's measurements against a reference. The generated tokens:
they are not recorded, and no test of x0ne Core's outputs ran on this VM.
Text prompts, sampling, requests sent together, prompts beyond 16,384
tokens, answers beyond 256 tokens, the decode without the CUDA graph, other
GPUs, quantized weights. One VM, one host, one day; five runs per row.

## Files

- `core-q06-ccon.jsonl`, `core-q8b-ccon.jsonl`, `core-q14b-ccon.jsonl`,
  `core-q32b-ccon.jsonl`: `x0ne-core-itl`'s output as it wrote it in the VM,
  unchanged (`ccon`: CC mode on). One load line (load time, x0ne Core's
  description of the GPU, the CUDA driver and cuBLAS versions, the arena and
  the decode graph), then one line per run: the model's tag, the prompt's
  label and index, the warm-up flag, the prompt's length, the tokens
  generated, and the 256 times in seconds.
- `prompts/`: the five prompt files and `PROMPTS.txt`, their SHA-256.
- `download.tsv`: per file of the four models, its check, its size and its
  download time.
- `core-runs.tsv`: one line per model's process: its end time (UTC), the
  mode, the exit code, the seconds, and `nvidia-smi`'s temperature, SM and
  memory clocks, power, memory used and throttle reasons before (`gpu0`) and
  after (`gpu1`) it.
- `x0ne-core-itl.rs`: the tool's source as built; only its first comment
  line was edited for publication (it named a commit of x0ne Core).
- `core-summary.py`, standard library only, and its output `summary.json`
  (every figure, unrounded) and `summary.txt` (the same, rounded).
- `SHA256SUMS`: the SHA-256 of every other file here.

```sh
python3 -I core-summary.py <out dir> core-q06-ccon.jsonl core-q8b-ccon.jsonl core-q14b-ccon.jsonl core-q32b-ccon.jsonl
```

It writes `summary.json` and `summary.txt` byte for byte as kept here, under
Python 3.11, 3.12 and 3.13 (checked on 2026-11-04).

Not kept: the VM's manifest, which names the GPU's serial number and UUID and
the VM's host name (what it says is above), the TD quote and its check, the
serial console, the GPU monitor's log, the tool's error output (one line per
run, the figures of the JSON lines), the run's log outside the VM and the
VM's other files.

Estimated cost 2.37 USD for the VM (25.1 minutes at 5.6482 USD/h on Cloud
Billing's catalog, confidential computing included, read at the run), at
most 0.02 USD more for the bench image's import and its transit bucket:
2.39 USD. The VM also served other measurements, not published here; the
cost is that of the whole VM.
