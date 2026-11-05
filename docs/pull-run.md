# `x0ne-engine`: pull, list, rm

`x0ne-engine pull` fetches an open model from the x0ne catalog, verifies every
byte and installs it in a local cache. In cloud mode the TD runs the same
catalog and the same checks before it serves a model (`registry::pull_served`,
[cloud.md](cloud.md)).

```sh
x0ne-engine pull qwen3:0.6b         # download, verify, install; prints the model directory
x0ne-engine verify qwen3:0.6b       # re-hash the installed files
```

## Commands

| command | what it does |
|---|---|
| `pull <name>` | download, verify (size + sha256 of each file), install; prints the model directory |
| `list` | installed models, with `ok (sizes; verify re-hashes)`, `DAMAGED` (a file missing, not regular, or of another size), `outdated` (the catalog moved) or `not in this catalog`; sizes only, no hashing |
| `catalog` | what this build can pull |
| `verify <name>` | re-hash an installed model against its record |
| `rm <name>` | remove an installed model and its partial downloads |
| `path <name>` | print an installed model's directory |

Names are `family:tag` (`qwen3:0.6b`). There is no implicit `latest`: `qwen3`
alone is refused with the list of tags.

Build it with:

```sh
cargo build --release -p x0ne-infer --bin x0ne-engine
```

## The catalog

`x0ne-infer/catalog/models.toml`, embedded in the binary. `--catalog <file>` or
`X0NE_ENGINE_CATALOG` replaces it; the replacement goes through the same checks.

```toml
schema = 1

[[model]]
name = "qwen3:0.6b"
repo = "Qwen/Qwen3-0.6B"
revision = "c1899de289a04d12100db370d81485cdf75e47ca"   # a commit, never a branch
architecture = "qwen3"                                   # qwen3, the one architecture served
license = "Apache-2.0"
openness = "open-weights"        # optional; tested against src/openness.rs
measured = "2026-10-10"          # optional; when the digests were taken
files = [
  { path = "config.json", size = 726, sha256 = "660db3b7…" },
  …
]
```

Refused at parse time, with the entry's name and the reason:

- a revision that is not 40 lowercase hex (a branch or tag can move);
- a file outside the allow-list: `config.json`, `tokenizer.json`,
  `model.safetensors`, `model-N-of-M.safetensors` with
  `model.safetensors.index.json`, `LICENSE`, `NOTICE`. Pickle checkpoints
  (`.bin`, `.pt`, `.pth`, `.pkl`, `.ckpt`) and Python are named as such in the
  error. Anything with a `/` is refused, so no path leaves the model directory;
- a sha256 that is not 64 lowercase hex, a size of 0 or above 256 GiB;
- a missing `config.json` or `tokenizer.json`, no weights, or both a single
  file and shards;
- an architecture the engine does not serve, a malformed name or repository,
  duplicate names or paths, unknown keys, a schema other than 1.

Entries (added 2026-10-10, 2026-10-16 for the three large Qwen3). Since
2026-11-04 the catalog serves Qwen3 only; `olmo2:1b`, `ministral3:14b` and
`ministral3:8b` left it ([CHANGELOG](../CHANGELOG.md)):

| name | source (revision) | size | how the digests were measured |
|---|---|---|---|
| `qwen3:0.6b` | Qwen/Qwen3-0.6B (`c1899de2`) | 1.4 GiB | downloaded at the revision; weights and tokenizer equal the Hub's LFS sha256; weights equal `openness.rs` (2026-09-25) |
| `qwen3:1.7b` | Qwen/Qwen3-1.7B (`70d244cc`) | 3.8 GiB, 2 shards | small files downloaded; shards = Hub LFS sha256, then confirmed by a full `pull` |
| `qwen3:8b` | Qwen/Qwen3-8B (`b968826d`) | 15.3 GiB, 5 shards | Hub API tree at the commit: shards and tokenizer = LFS sha256; `config.json`, index and `LICENSE` downloaded, matched to their git blob id; then all 9 files downloaded at the commit and hashed: equal |
| `qwen3:14b` | Qwen/Qwen3-14B (`40c06982`) | 27.5 GiB, 8 shards | same method as `qwen3:8b` |
| `qwen3:32b` | Qwen/Qwen3-32B (`9216db57`) | 61.0 GiB, 17 shards | same method as `qwen3:8b` |

Each entry also pins its served-model digest (`digest_sha256`, version 2,
[docs/cloud.md](cloud.md#the-served-model-digest)). It follows from the
sha256s listed, and the parser refuses any other value.

Adding an entry: download each file at a pinned commit, `sha256sum` it, compare
with the Hub's LFS digest where there is one, compute the served-model digest
(the `sha256sum` manifest in docs/cloud.md), then `cargo test -p x0ne-infer
registry` and a real `pull`. Another family needs an architecture in x0ne
Core first; the catalog refuses it until then.

### Qwen3 8b, 14b and 32b on one H100 80 GB

Computed on 2026-10-16. Cloud mode serves an entry only on a pinned
served-model digest; since 0.2.0-beta.6 all three pin one.
An unpinned entry is refused by the client (exit 2) and the relay (96) before
any VM exists (`Entry::cloud_refusal`), as the TD does after boot. In the TD,
`registry::pull_served` checks every shard, maps the shards the catalog
lists, checks the index against their headers (72), and refuses with 77 a
served digest other than the pin. The context the engine serves, on x0ne
Core the model's whole `max_position_embeddings`, is reported at `/v2/ready`
(`context_tokens`) and printed by `up` at `[5/5]`.

The catalog carries no context or memory limit. The context is
`max_position_embeddings` of `config.json`, 40,960 for all three, and it is
pinned with the file's sha256. x0ne Core runs bf16 on the GPU and has no CPU
path.

- Weights: `total_size` of `model.safetensors.index.json`, bf16.
- KV cache per token: 2 (K and V) x layers x KV heads x head_dim x 2 bytes.
  The inputs come from `config.json` at the pinned commit, checked by the test
  `the_large_qwen3_configs_are_the_catalogs_and_parse`.
- x0ne Core sizes one arena when the model loads, for the weights, the KV
  cache and its working buffers, and allocates nothing after
  (`x0ne-core/runtime/src/arena.rs`). The arena is sized for the model's
  whole context: a model whose arena does not fit the GPU memory left is not
  loaded (`x0ne-infer/src/core_backend.rs`, read in the code).

| | `qwen3:8b` | `qwen3:14b` | `qwen3:32b` |
|---|---|---|---|
| layers, KV heads, head_dim | 36, 8, 128 | 40, 8, 128 | 64, 8, 128 |
| weights, bf16 | 15.26 GiB | 27.51 GiB | 61.02 GiB |
| KV cache per token | 144 KiB | 160 KiB | 256 KiB |
| KV cache at 40,960 tokens | 5.63 GiB | 6.25 GiB | 10.00 GiB |

`nvidia-smi` showed 81,559 MiB (79.65 GiB) for the H100 80GB HBM3 of our
VM in CC mode on 2026-11-03. On x0ne Core a model loads with its whole context, 40,960
tokens, or not at all. Through this Engine on an H100, `qwen3:8b` loaded
with its 40,960 tokens on 2026-11-04, a TEST measurement
([hardware runs](hardware-runs.md)); whether `qwen3:14b` and `qwen3:32b`
fit is not measured.

On the host side of a TD (`a3-highgpu-1g`, 234 GiB), the CVM image mounts the
model store as a tmpfs of half the memory (117 GiB). Until 0.2.0-beta.6,
`pull` kept the shards and their merge side by side (2 x 61.0 GiB for
`qwen3:32b`), which did not fit. Since then the shards stay as published,
moved from `partial/` by rename, and the engine maps them read-only: 61.0 GiB
of the tmpfs. x0ne Core copies each tensor from the mapping to the GPU with
one `cuMemcpyHtoDAsync`, with no copy in the process heap
(`x0ne-core/runtime/src/weights.rs`, `upload`, and `gpu.rs`; read in the
code, not measured). That is not "no copy in host memory": with the GPU in CC mode,
every host-to-GPU copy goes through bounce buffers in memory the guest shares
with the host, where the guest's driver encrypts and signs the data and the
GPU decrypts it (NVIDIA, 2023, "Confidential Compute on NVIDIA Hopper H100",
WP-11459-001, pp. 11 and 27,
<https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/HCC-Whitepaper-v1.0.pdf>).
The host sees every weight byte pass, encrypted. The same paper gives about
4 GB/s for that path (p. 29): about 16 s for 61.0 GiB, computed. Not measured
on a TD yet.

## What `pull` guarantees

In this order:

1. each file is downloaded to `partial/<sha256>.part`, up to 4 files at once
   and at most 16 requests in flight (`registry::Transfer`). A file larger
   than 64 MiB is fetched in byte ranges of 64 MiB, several at once, written in
   place; the ranges written are listed in `partial/<sha256>.ranges`, so the
   next `pull` asks only for the missing ones; a range listed past the end of
   the `.part` (cut short, or gone) counts as missing. A range is asked again, up to 3
   more times, after a transport error, a 5xx, a 429, a short body or a
   Content-Range other than the one asked; it never counts as written before
   all its bytes are. After a redirect (the Hub sends weights to a CDN) the
   ranges go to the redirect target directly, one resolve per file; a failed
   range goes through the catalog URL again. A source that answers the first
   range with 200 is read in one stream from byte 0, the ranges list
   discarded. A file of 64 MiB or less is one request: an existing partial is
   resumed with `Range: bytes=N-`, and a server that answers 200 instead of
   206 restarts the file from zero. After a failure no new request starts;
   those in flight finish and are kept;
2. the server may not send more than the catalog's size (the download is cut
   and deleted); a declared size other than the catalog's, or a body that ends
   short, is refused with exit 65 and the partial deleted (a cut connection
   stays exit 69 and resumes);
3. the whole file is hashed from disk, resumed prefix included; a mismatch
   deletes the partial and fails the pull with exit 65;
4. only when every file has passed are they moved to `staging/`;
5. `config.json` is refused if it carries `auto_map` (code shipped with the
   model, what `trust_remote_code` would run) or names another architecture
   than the catalog;
6. `x0ne-model.json` is written (the catalog's files, shards as published),
   then the staging directory is renamed into `models/<family>/<tag>`. A copy
   pulled before 0.2.0-beta.6, its shards merged into one file, is replaced.

A directory under `models/` therefore exists only once verified. A model
already installed at the same revision and digests is re-hashed and, if intact,
not downloaded again; a damaged copy or a catalog that moved is replaced, by
rename. Two pulls of the same name are refused (`flock` on
`locks/<family>-<tag>.lock`, removed on release). Installed files are 0640 and
their directory 0750; partial downloads are created 0640.

Transport: `https://huggingface.co` by default, `X0NE_ENGINE_HF_ENDPOINT` to
change it. Plain HTTP is refused, except to the loopback (tests, a local
mirror); redirects (the Hub sends weights to a CDN) are followed only to https,
at most 10, and from a public endpoint never to a literal loopback, link-local,
private or unspecified address nor a metadata name (names are not resolved; an
endpoint you set on such a host keeps its redirects). TLS roots are the ones compiled into `reqwest` (`webpki-roots`).
Integrity does not rest on the transport: a mirror that serves other bytes is
refused by step 3.

What it does not do: sign the catalog (the catalog is trusted as part of the
binary, or as the file you pass), authenticate to the Hub (gated models answer
401, reported as such), or protect the cache against a local user who can write
to it after `verify`.

## The cache

```text
$X0NE_ENGINE_HOME                       (else $XDG_CACHE_HOME/x0ne-engine,
                                         else ~/.cache/x0ne-engine)
├── models/<family>/<tag>/              verified, ready to serve
│   ├── config.json, tokenizer.json, model.safetensors or the index and
│   │   its shards, [LICENSE]
│   └── x0ne-model.json
├── partial/<sha256>.part               resumable downloads
├── partial/<sha256>.ranges             byte ranges of a ranged .part already written
├── staging/<family>-<tag>/             assembled under the lock
└── locks/
```

A relative `XDG_CACHE_HOME` or `X0NE_ENGINE_HOME` is ignored, as the XDG Base
Directory specification asks.

## Exit codes

`0` ok · `1` not installed, not in the catalog · `2` usage · `65` integrity
(sha256, oversize, invalid catalog, remote code, architecture) · `69` network
(run `pull` again: it resumes) · `74` local I/O.

## Measured on 2026-10-10 (CPU, containers, no GPU)

- `pull qwen3:0.6b` interrupted after 6 s with 95 MB of weights on disk; the
  next `pull` resumed at 6 % through the Hub's CDN (one 206 answer) and
  installed after verification: "1.3 GiB downloaded".
- `pull qwen3:1.7b`: two shards verified and merged, 177 s including the
  download; `verify` then matched the 4 served files.

Unit tests: `cargo test -p x0ne-infer registry` (28 tests: catalog validation,
cache paths, lock, and `pull` against a local server playing the Hub —
corruption, oversize, resume, ignored range, poisoned partial, remote code,
wrong architecture, 404, tampering after install, revision change, `rm`,
shards). Disabling the sha256 comparison and the `auto_map` check turns 3 of
them red.
