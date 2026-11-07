# Changelog

Dates are when the change landed. 0.2.0-beta.1 is the first version number;
the first non-beta tag will freeze the receipt protocol below.

## Unreleased

## 0.2.0-beta.7 — 2026-11-07

This version's number was set on 2026-10-18, by the entries at the end of
this section; the others landed until 2026-11-07.

### Added

- `install.sh` installs the client binaries of a GitHub release
  (2026-11-07), adapted from x0ne-Harness's installer: Linux x86_64 with
  glibc only, over https only, `~/.local/bin`, no sudo, no other request.
  It verifies `SHA256SUMS.sig`, the maintainer's SSH signature of
  `SHA256SUMS`, with `ssh-keygen -Y verify` (OpenSSH 8.1+) against the key
  written in the script, before it downloads the archive (2026-11-07); it
  refuses a missing or invalid signature, a missing or older `ssh-keygen`,
  and a signed list whose first line names another release. No variable or
  option changes the key or skips the check. Then it refuses a missing or
  mismatching checksum, and runs `gh attestation verify` when gh 2.56+ is
  logged in, refused on failure. It installs the release on its `VERSION`
  line (GitHub's `releases/latest` skips pre-releases), takes `--version`,
  and runs the new `x0ne-engine` before it replaces anything.
  `make install-check` tests it against fake releases signed with a test
  key, in Debian 13, Ubuntu 24.04, Fedora 43 and Alpine 3 containers
  (`ci/tests/installer.sh`, `ci/install-matrix.sh`), and so does the
  `ci-scripts` job; the same test replays the README's `.deb` steps, which
  stop at the first failed check, before `dpkg`.
- The release workflow attaches `x0ne-engine_amd64.deb` and `install.sh`,
  both in `SHA256SUMS` and attested (2026-11-07); `SHA256SUMS` starts with
  `# x0ne-Engine <tag>`, and the maintainer signs it and uploads
  `SHA256SUMS.sig` to the draft before publishing it (`docs/BRANCHING.md`,
  2026-11-07). The `.deb` ships the
  archive's three binaries (`ci/deb-pack.sh`: the fields of
  `ci/deb/x0ne-engine.ctl`, dpkg-deb, every date at `SOURCE_DATE_EPOCH`;
  `ci/deb-verify.sh` checks it); the two builds must give the same `.deb`.
  Before the draft, a new job installs the archive with `install.sh` in
  Debian 13, Ubuntu 24.04 and Fedora 43 (refused on Alpine 3), and the
  `.deb` with `dpkg` in Debian 13 and Ubuntu 24.04. The tag must equal
  `install.sh`'s `VERSION`.

### Security

- Receipt signatures are checked with `verify_strict` (2026-11-05). `verify`
  accepted a small-order key: with the identity point as the key, the
  signature (R = identity, s = 0) verified over every body, and
  `receipt::verify_signed` passed such a forged receipt. The relay and the
  client compare the receipt's key with the attested one next, and refused it
  already.
- An NRAS token without `eat_nonce` is refused by name, and so is an empty
  expected nonce (2026-11-05). An absent claim read as an empty nonce and
  matched an empty expected one. Every caller in this tree passes 64 hex
  digits.
- A TDX quote whose QE report carries non-zero bytes in
  `report_data[32..64]` is refused (2026-11-05). No check read them; Intel's
  TD QE writes zeros there. Producing such a report takes the PCK key or a QE
  other than Intel's: no attack is known.
- A QE identity whose schema version is not 2 is refused by name
  (2026-11-05), as a tcbInfo of any version but 3 already was. Such a
  document needs Intel's signature: this refuses an unknown value, not a
  forgery.
- An NRAS token whose JWS header lists `crit` is refused (RFC 7515, 4.1.11;
  2026-11-05): this verifier understands no extension and read past it. The
  header is under NVIDIA's signature.
- A catalog pull and the TD's fetch of a sealed source refuse a redirect from a
  public endpoint to a literal loopback, link-local, private or unspecified
  address or a metadata name, before any connection, and say so (2026-11-05).
  Until then only the scheme was checked: a 302 to `https://0.0.0.0:<port>/`
  reached a service on the machine's own loopback. Names are not resolved, as
  for the source itself.
- The TD closes a connection whose next request head is not whole within 61
  minutes of the handshake or of the previous answer's end, slow or idle, and
  counts it (`head_timeouts`, 2026-11-05). There was no limit: hyper drops its
  30 s default when no timer is set. An idle relay connection is closed too;
  the relay opens a new checked one.
- The relay compares the bearer token of its local endpoint with `subtle`
  (2026-11-05). The hand-written comparison compiled without a branch on the
  data (rustc 1.98.1, opt-level 3), but nothing guaranteed it.
- x0ne-sdk for wasm32 with `test-anchor` is refused by its build script
  unless `X0NE_SDK_BENCH=1` (2026-11-04). Such a `.wasm` takes the roots a
  policy names (Intel's, Google's, NVIDIA's), the TEST archive key and the
  TEST log, and nothing stopped a page from loading one.
  `scripts/sdk-wasm.sh` and `scripts/e2e-local.sh` set it and keep the
  bench's `.wasm` under `out/`; no wasm module is in the sources
  (`x0ne-sdk/tests/test_anchor_guard.rs`). Native builds are unchanged:
  x0ne-infer's tests and examples, release ones included, link x0ne-sdk
  with the feature.
- `x0ne-engine-proxy` exits 73 when it cannot make itself non-dumpable, and
  drops its core limit (`harden::process`, as `x0ne-infer` does;
  2026-11-04). It ignored what `prctl` returned until then.
- `x0ne-infer seal` refuses a file over 2^36 - 32 bytes, AES-GCM's limit
  for one message, before reading it, and `seal::seal` returns an error past
  that limit (2026-11-04). It panicked there, under the 128 GiB weights cap.
- `x0ne-infer` built without debug assertions also exits 73 first thing in
  `main` when `LD_LIBRARY_PATH`, `CUDA_INJECTION64_PATH` or
  `NVTX_INJECTION64_PATH` is in its environment, whatever the value, naming
  it (2026-11-04). x0ne Core and NVML open `libcuda.so.1`,
  `libcublas.so.13` and `libnvidia-ml.so.1` by name after `main`: a
  `libnvidia-ml.so.1` put on `LD_LIBRARY_PATH` ran its constructor in a
  release build before this change. The CUDA driver loads the library
  `CUDA_INJECTION64_PATH` names, NVTX the one `NVTX_INJECTION64_PATH` names,
  and cuBLAS and cuBLASLt carry NVTX. A debug build takes the three: cargo
  sets `LD_LIBRARY_PATH` for every test. The Containerfile no longer sets
  `LD_LIBRARY_PATH`: its image finds cuBLAS through
  `/etc/ld.so.conf.d/x0ne-infer.conf` and the loader's cache.
- In local mode `--expected-model-digest` is compared on the plaintext
  weights read, before a backend is opened or parses them (exit 77,
  2026-11-04); x0ne Core loaded them first until then. Sealed weights are
  pinned on the model loaded, as before.
- `--device cuda:N` with N other than 0 exits 81 when the flags are read
  (2026-11-04): x0ne Core runs on GPU 0 only, and refused it with 79 after
  the attestation. The help says `--device <cuda|cuda:0>`.
- `ci/release-refusal.sh` runs 22 cases on each binary instead of one
  (2026-11-04), all decided before any evidence: no argument (87), the CPU
  asked to serve in six forms (90), each variable `harden.rs` refuses (73),
  `--device cuda:1` and `cpu:0` (81), `--attest gpu` and `CPU` (82), the
  lab server's flags (2), a session key or a pin in cloud mode (95).

### Removed

- `bench/h100/` (2026-11-04): the scripts of an H100 bench campaign, not
  executed (its README said so).
- The lab server (2026-11-04): `x0ne-infer-lab` with its
  `not-confidential`, `learn` and `decide` features, the `/learn` routes and
  the lab's `/v1/decide`, `x0ne-engine run`, `x0ne-infer-ssh-chat`, the
  per-request LoRA adapters, the lab's container image and its package.
  `x0ne-engine run` is refused by name and points to `x0ne-engine cloud run`.
- The `x0ne-infer` package (2026-11-04): the TD's server ships in the CVM
  image only, built with x0ne Core (`cuda nvml`); the `.deb` keeps the
  client, `x0ne-engine`, `x0ne-engine-cloud` and `x0ne-engine-proxy`.
  The release archive, `x0ne-engine-<target>`, carries the same three
  binaries, and the release workflow builds no image of the server.
- The CPU-only mode (2026-11-04): `--attest cpu`, a TDX guest without a GPU.
  It ran on the engine before x0ne Core; x0ne Core runs on the GPU only, and
  `--attest cpu` is refused, by the server (exit 90), by
  `x0ne-engine-proxy` and by x0ne-sdk (`attest=cpu`).
- Verified decisions (2026-11-04): `x0ne-infer --decide` (feature
  `confidential-decide`), its GLiClass encoder over ModernBERT, the decision
  receipt and `x0ne-decide-verify`. They ran on the candle library.
- Conversations kept between turns, from the served build (2026-11-04):
  x0ne Core keeps none, so a request with `x0ne.conversation` gets a 400
  naming it, on every model, and each turn is computed from token 0. The
  code stays with the engine before x0ne Core, which only the tests build
  (`docs/session-kv-cache.md`).
- Models other than Qwen3 (2026-11-04): `olmo2:1b`, `ministral3:8b` and
  `ministral3:14b` leave the catalog, which keeps `qwen3:0.6b`,
  `qwen3:1.7b`, `qwen3:8b`, `qwen3:14b` and `qwen3:32b`: x0ne Core runs
  Qwen3 dense in BF16 only. Their figures in `docs/benchmarks.md` and
  `docs/hardware-runs.md` stay as dated measurements of the engine before
  x0ne Core.

The entries below that name them, or that say a model runs on candle, are
history: candle has been the CPU reference of the tests since 2026-11-04.

### Changes, newest first

- The draft release's notes are a link to the tag's `CHANGELOG.md` when the
  version's section is over 125,000 bytes (2026-11-07). GitHub refuses a
  longer release body, and the release job then failed after the
  attestation, with no draft: this section was 125,589 characters on
  2026-11-07. `ci/tests/release-notes.sh` replays the step with a fake `gh`
  (in `ci-sh`): this CHANGELOG, a short section, one over the cap, none.
- A `memfd_secret` key is tested against `process_vm_readv(2)` as well as
  `/proc/self/mem` (2026-11-07): the read fails with EFAULT, while a heap
  buffer, the control, reads back. Measured on 6.12.115; with the key on an
  ordinary memfd named `secretmem`, the read returned the key's bytes and the
  test failed. The `mlock` fallback still has to carry `lo`, `dd` and `wf`.
- The `memfd_secret` tests no longer pass on the `mlock` fallback where the
  kernel offers `memfd_secret` (2026-11-06). They ask the kernel for a
  descriptor themselves: where it gives one, a key in `mlock` memory fails,
  as does a mapping other than `/secretmem (deleted)` (an ordinary memfd
  named `secretmem` maps as `/memfd:secretmem (deleted)`), and the
  `/proc/self/mem` test no longer returns early. A `SecretBytes` that never
  called `memfd_secret` passed both before. The `Unevictable` measure reads
  the test's cgroup v2 `memory.stat` where it has one, else `/proc/meminfo`,
  and holds a lock file in the temporary directory: copies started 1 s
  apart counted each other's pages, in two containers through
  `/proc/meminfo` and in one, as `make ci`'s three test profiles share it.
- `SecretBytes` writes its `memfd_secret` pages when it is made, by
  `read(2)` from `/dev/zero` (2026-11-06). A kernel with Linux commit
  97d34aa65c29 counts them against the user's `RLIMIT_MEMLOCK` at that first
  write: past it, the kernel's copy fails with EFAULT, where a store from the
  process got SIGBUS. Construction then falls back to `mlock`, and fails if
  that fails too, naming both and the limit: with `RLIMIT_MEMLOCK` 0, cloud
  mode exits 97 at its TLS identity, where it died of SIGBUS on 6.12.115
  (`tests/process_hardening.rs`). `MADV_POPULATE_WRITE` does not fault these
  pages in (EINVAL, measured on 2026-11-05).
- The relay makes each GET to Intel (PCS, root CA CRL) up to three times:
  again 1 s then 2 s after a transport error, a 5xx or a 429, never after
  another answer or a document that does not parse (2026-11-06). Each
  GET has 20 s, so one document takes 63 s at most. Three failures refuse
  with 76 as one did, and the reason now carries the transport error's
  causes, where reqwest's summary said "error sending request" alone. On
  2026-11-06 one transport error on the root CA CRL refused a paid H100 VM
  with 76; the server answered two minutes later. Google's bucket is still
  read once.
- `x0ne-engine cloud up` and `run` exit 102 when the model owner's key
  policy refuses the TD or the session, as the relay does (2026-11-06). They
  kept the relay's codes 64-87 only: on 2026-11-06, on an H100 TDX VM the
  policy did not name, `up` printed `REFUSED (102)`, sent nothing to the VM,
  deleted it and exited 95. Any other relay code is still 95. The help and
  docs/cloud.md list 102.
- The `memfd_secret` test checks that the pages stay out of swap, not that
  they are `mlock`ed (2026-11-05). Linux commit 97d34aa65c29 (stable
  6.12.113, CVE-2026-93243) no longer marks a `memfd_secret` mapping
  `VM_LOCKED`; its pages stay unevictable. The test writes up to 4 MiB of
  `memfd_secret` memory and as much ordinary memory, and requires
  `Unevictable` in `/proc/meminfo` to grow by three quarters of the first
  at least and by less than a quarter of the second. It passes on 6.12.111
  and 6.12.115, and fails when the region is an ordinary memfd. The same
  commit counts these pages against the user's `RLIMIT_MEMLOCK` when they are
  first written: past it, `SecretBytes` got SIGBUS at the first write,
  where its construction failed with an error before (the TD exited 97);
  since 2026-11-06 construction fails there again (above).
  The docs no longer call this memory locked.
- x0ne Core is the served engine (2026-11-04). With the `cuda` feature, the
  attested server runs the model on x0ne Core, written for this project in
  Rust and CUDA: its kernels compiled ahead of time, one arena reserved at
  load, one CUDA graph replayed per generated token (`x0ne-core/`, copied
  from x0ne Core as `x0ne-core/PROVENANCE.md` says;
  `x0ne-infer/src/core_backend.rs`). It replaces the engine built on the
  candle library (Hugging Face, 0.11.0), which stays in the tests only, as
  the reference x0ne Core is checked against:
  `cargo tree -e normal -p x0ne-infer --features "cuda nvml"` lists no
  `candle-*` crate (277 packages, none of them candle, measured on
  2026-11-04). A seed signed in a receipt draws the tokens it drew before
  (`x0ne-infer/src/sampler.rs`, tested against that reference). The context
  is the model's whole `max_position_embeddings`, reserved in the arena at
  load: a model whose arena does not fit the GPU is not loaded, where the
  engine before x0ne Core cut the context to the memory left
  (`x0ne-infer/src/core_backend.rs`, read in the code). The receipt's format
  does not change: it names no engine; the image's measurements name the
  binary, and x0ne Core with it (x0ne-cvm 0.2.9, built on 2026-11-04: its
  `x0ne-infer`, SHA-256 `0254fd6b…`, in the dm-verity tree whose root hash
  `c708283b…` is on the kernel command line; RTMR1 `59a3b771…` and RTMR2
  `7c34a34b…` as its build predicts them). That image is not published. It
  was built from x0ne OS sources that are not published either, and from a
  pre-release commit of this Engine whose `x0ne-infer` predates the Security
  entries above and the x0ne Core copy's current fingerprint; its
  `x0ne-infer --version` names that commit, so a build of this tree does not
  give these fingerprints. On
  2026-11-04 `x0ne-engine cloud up` refused the run's relay, a test build,
  before the token (exit 95, no VM created). Later that day, with the `up`
  of an earlier build (sources of 2026-10-31), the image booted in a TD on
  an H100 in CC mode: verdict VERIFIED under the transparent TCB policy,
  RTMR1 and RTMR2 as predicted; the SDK sealed one question and verified the
  answer's receipt (its record is kept outside this repository); 36 answers streamed through the relay, every receipt
  verified by it (`bench/results/2026-11-04-h100-tdx-cc-core-e2e/`). A TEST
  measurement, with the TEST archive key and the TEST transparency log, not
  a release measurement: a release relay waits for the signing ceremony. x0ne
  Core alone, its own bench on a bench image without NVIDIA attestation,
  decoded `qwen3:8b` on an H100 at 138.62 tokens a second after a 20-token
  prompt in a TDX VM in CC mode (2026-11-03; `docs/benchmarks.md`,
  `docs/hardware-runs.md`, `bench/results/2026-11-03-h100-tdx-cc-core/`);
  on 2026-11-04, its per-token bench in CC mode decoded `qwen3:0.6b`,
  `qwen3:8b`, `qwen3:14b` and `qwen3:32b`, from 729.62 to 35.15 tokens a
  second (`bench/results/2026-11-04-h100-tdx-cc-core-bench/`).
- Corrected (2026-11-04): the padding of an answer's length (2026-10-24,
  below) hides it from the network, not from the VM's host. The computation
  stops when the generation ends, and x0ne Core copies one token from the
  GPU and, after the first, replays one CUDA graph per generated token (read
  in the code): the VM's CPU time and the GPU's work follow the length. A
  host that reads them would count the tokens: deduced, not measured in CC
  mode (THREAT_MODEL.md, row "Host reading an answer's length from the
  computation").
- THREAT_MODEL (2026-11-04): a known attack, the choice of experts in a
  mixture-of-experts model read from the TD's memory accesses (Nuriyev and
  Kulp; SparSEEty; TDXRay). Not exposed by this version: no model of the
  catalog is a mixture of experts.

- Tests (2026-11-03): the two tests that read the TLS records a simulated
  TD writes for a stream failed on a loaded host (one CPU shared with 24
  busy loops: the one-size test in 10 runs of 20, the two-answers test in
  1 of 20). When the TD's task is not run for a tick or more, a frame
  already built leaves whole with the next one, in one record of two
  frames (1055 bytes for 512-byte frames), or with the last frame; the
  number of frames does not change. The tests now split those two records
  only, and any other size still fails them. The one-size test also checks
  that the frames, the last included, count to a Padme size, and a test of
  the stream's clock that a stall never writes more than two frames at once.

- Corrected (2026-11-03): the decode rates of `ministral3:14b` on the H100,
  runs of 2026-10-28 and 2026-10-31, are withdrawn from `docs/benchmarks.md`,
  `docs/hardware-runs.md` and FEATURES, with the ms per token and the fit of
  the second run. They were (N - 1) / (t(N) - t(1)) on whole answers whose
  time the TD pads to a Padme number of 100 ms ticks, ten at least (the
  SECURITY entry on an answer's length, below): in those two runs no whole
  answer took less than 1.197 s, and every t(N) - t(1) of the medians is a
  multiple of 100 ms within 0.012 s, a ratio of padded sizes rather than a
  decode rate. `docs/benchmarks.md` says so under Method, and `summarize.py`
  derives no decode rate from those two directories. The rates of `qwen3:8b`
  and `qwen3:32b` (2026-10-17, 2026-10-18, 2026-10-22) were measured without
  the padding and stand.

- Before publication, after three exposure audits (2026-11-03):
  `x0ne-engine-proxy pins` says `"test": true` for the TEST archive key and
  log of a debug build, and `cloud up` refuses that relay before the token
  (95), except against a test build's loopback fake. `cloud up` and the relay
  say that no question and no session key have been sent before the verdict;
  they said nothing had been, while the VM's metadata already held the session
  config and, with `--nras-key-file`, the NRAS key. The lock leaves the yanked
  crypto-bigint 0.7.6 for 0.7.10, and RUSTSEC-2026-0429 (zip) is ignored with
  the other zip extract advisories ([SECURITY.md](SECURITY.md)). FEATURES now
  gives the TCB policy's transparent default, the sampling a receipt signs and
  1,370 tests (`cargo test -p x0ne-infer`, 32 ignored); the README the device
  a receipt names (`cuda:N` or `cpu`), the binaries' sizes and the disk a
  first build takes. The CI's `test`, `msrv` and `stable` jobs build with line
  tables only, after freeing the runner's disk.

- The relay rebuilds a whole answer from what it verified (2026-11-02), as
  it rebuilds a stream: one choice, its text under the receipt; the receipt,
  its opening and the next turn's handle from their types; `usage` from the
  receipt's token counts. Two choices or none, a `usage` that differs from the
  receipt's, or an `id`, `created`, `model`, role or `finish_reason` out of
  shape is refused as a receipt that does not verify: a 502
  `receipt_verification_failed`, and the session stops (65). The TD's JSON
  went out as written, marked `receipt_verified`: a field added inside the
  receipt, outside the signed bytes, a second choice and the TD's own `usage`
  reached the client with it. The emulated stream (`stream_emulated`) is
  written from the rebuilt answer.
- The README and `docs/assets/how-it-works.svg` say what goes to the VM before
  the verdict (2026-11-02): the relay asks it for its proof, and sends no
  question and no session key. They said nothing was sent.
- The server is under AGPL-3.0-only (2026-11-02): `x0ne-infer`,
  `x0ne-engine-cloud`, and `x0ne-infer-core` with x0ne Core's two crates. The
  end of `LICENSE` adds a permission (AGPL section 7) to combine them with
  NVIDIA's CUDA and driver libraries. `x0ne-sdk`, `x0ne-confidential` and
  `x0ne-ohttp` stay under Apache-2.0, each with its own `LICENSE`, and so do
  the 18 files of x0ne-infer that `x0ne-sdk` compiles (an SPDX line opens
  each). `LICENSING.md` maps them; `x0ne-infer/tests/licensing.rs` holds it
  to the manifests and the license files. External pull requests wait for
  the Contributor License Agreement (`CONTRIBUTING.md`).
  `THIRD_PARTY_LICENSES.html` no longer lists x0ne's own crates and follows
  the current `Cargo.lock`.
- The C ABI keeps each thread's result apart (2026-11-02): `x0ne_result_ptr`
  points at the calling thread's last result, valid until that thread's
  next `x0ne_call`. One result was kept for the instance: two threads of a
  native host calling in turn could read each other's answer, or a buffer
  the other's call had freed. The handles stay the instance's, shared by
  its threads (`x0ne-sdk/tests/abi_ffi.rs`).
- No panic crosses the C ABI (2026-11-02): every export catches it;
  `x0ne_call` returns `{"ok":false,"code":101,"step":"panic"}` and drops
  every key, pending request and stream of the instance, whose state the
  panic may have left half changed (handle numbers are not reused). A panic
  aborted the native host before. In wasm32 a panic is still a trap:
  `js/x0ne-sdk.mjs` refuses every call after one. `OP_TEST_PANIC`, under
  `test-anchor` only, injects one for the tests.
- The C ABI sets apart what a receipt does not sign (2026-11-02; breaks
  callers of ops 3 and 8): they return `status`, `text` and `receipt` (op 8
  also `final_deltas`, `chunks`, `conversation`) and, under `unverified`,
  the answer's `body` (op 3) or the stream's `finish_reason` and `usage`
  (op 8), which sat beside the signed text. An answer with no checked
  receipt (the TD's refusal, a stream ended by an error event) is now an
  error, its status and body or error under `unverified`; op 8 returned it
  as a result, with `receipt.checked: false` beside unsigned
  `final_deltas`, and `chatStream` did not throw. `chat`, `chatStream` and
  `smoke.mjs` follow.
- The relay's key file: a fallback held to the caller's own directories,
  and the documentation of it (2026-11-02). Under `$XDG_RUNTIME_DIR`, else
  `/run/user/<uid>`, else `<temp dir>/x0ne-engine-<uid>` (macOS has no
  `XDG_RUNTIME_DIR`); each directory is created 0700, or must be a
  directory of the caller's uid, not a symbolic link, that no other account
  can write (the key's own: 0700); otherwise the relay stops with 98 before
  writing the key. The `/tmp` fallback had its mode corrected, its owner
  unchecked, and a symbolic link in its place was followed;
  `docs/attested-channel.md` named `$XDG_RUNTIME_DIR` alone. A relative
  `$XDG_RUNTIME_DIR` is ignored, as the XDG spec asks.

- The README rewritten for the publication (2026-11-01): a header with the
  tagline, badges and links, the 2026-10-18 run's output as the client printed
  it (project id and VM address withheld, lines over 100 characters cut),
  "Highlights", "How it works" with `docs/assets/how-it-works.svg`, "Status"
  (the sections "What is protected, and measured", "What is left" and "What no
  TEE protects", unchanged), "The crates", "Contributing" and "Security". The
  opening paragraph that counted the crates is gone: `doc_claims_follow_code`
  now holds the table "The crates" to `Cargo.toml` (every member, no
  stranger, the excluded `x0ne-infer-core` said to be a workspace of its own);
  the Status box's one new negation of the confidentiality claim is listed in
  `x0ne-infer/tests/public_claims/allowed.tsv`.
- Before the publication (2026-11-01): `CODE_OF_CONDUCT.md` names
  contact@x0ne.co for reports; `.github/dependabot.yml` gains a `docker`
  entry for the base images pinned by digest (Dependabot reads literal
  `FROM` lines; an image named by an `ARG` or in a workflow stays pinned by
  hand); `x0ne_sdk::LIMITATIONS`, `GPU_LIMITATIONS`, their copy in
  `docs/sdk-e2e.md` and `docs/cloud.md` date the TD's e2e routes and the GPU
  path to 2026-11-01 as well, where `docs/cloud.md` still said "run once".
- The documents after the H100 run of 2026-11-01, the first sealed stream on
  hardware: `docs/hardware-runs.md` records it (the relay's
  `sealed_streams` carried 1 and cut none); `docs/sdk-e2e.md`,
  `docs/limitations.md`, `docs/README.md` and the README no longer say that
  no sealed stream has crossed the relay on hardware; `FEATURES.toml` moves
  the sealed stream to the hardware level, dated 2026-11-01, and its planned
  entry keeps the browser tab.
- The NRAS signing chain's revocation is judged (2026-10-31): the relay,
  the TD's own gate and the SDK read the OCSP responses a document staples
  about the JWKS `x5c` (`nras_ocsp.rs`, RFC 6960: `successful`,
  `id-pkix-ocsp-basic`, one `SingleResponse` whose `CertID` names the
  certificate under its issuer, signed by the issuer or by a responder it
  issued and marked `id-kp-OCSPSigning`, `thisUpdate` not after now,
  `nextUpdate` present and not before now as RFC 5019 asks, issued at most
  a day ago as Intel's collateral, the nonce echoed when one was sent, no
  critical extension unread), under a policy every verdict names
  (`NrasOcspPolicy`): `required`, the default of a policy written without
  one, refuses anything short of a verified, current `good`; `transparent`,
  what production verifies with until the TD staples responses, refuses a
  verified `revoked` and names the rest, by certificate. Revoked:
  `NrasRejected::SigningChainRevoked`, 75; in the SDK the step is
  `revocation` (`Policy::nras_ocsp`, `with_required_nras_ocsp`, `nras_ocsp`
  in the ABI's policy, `nras_revocation` in its result). The document's
  `nras.ocsp_b64` and the key document's `nras_ocsp` carry the responses.
  The intermediate's own status is said unverifiable: NVIDIA publishes no
  CA 001. Tested on an in-process responder: good, revoked, unknown, stale,
  future, open-ended, forged, wrong serial, delegated responder with and
  without the mark, nonce, critical extension, responder error, under both
  policies. No TD fetches a response yet, and the verdict says so. Until
  then the chain's revocation was not checked at all.
- The SDK remembers the transparency log (2026-10-31): `attest::LogMemory`,
  the signed note of the last checkpoint it verified, returned with every
  admission for the caller to keep and handed back before the next
  (`admit_reference` takes it; `tlog_checkpoint` in and out of the ABI's op
  1, kept in `localStorage` by `js/x0ne-sdk.mjs`). The reference's
  checkpoint is held to it as a witness holds a submission (C2SP
  tlog-witness, `transparency::remember`): the same size must show the same
  root (`log-fork`), a smaller one is refused (`log-rollback`), a larger
  one is admitted only with a consistency proof from the remembered size,
  the tlog-witness `add-checkpoint` body (`tlog::parse_add_checkpoint`,
  `tlog_consistency` in the ABI; missing, truncated, altered, from another
  size, to another checkpoint or from another history: `log-consistency`),
  and a memory that is not a checkpoint of the pinned log refuses every
  reference (`log-memory`); each with the relay's log code, 100, the step
  named (`Refusal::step`, `step` in the ABI's result). Until then the SDK
  checked the proof it was given and kept nothing between calls. The SDK
  still fetches no checkpoint and no tile: the integrator supplies the body
  (`x0ne-engine-tlog witness-request`, or from the tiles). Tested on the TEST
  log grown in memory (`transparency::test_log::MemoryLog`), natively and
  through the C ABI; the `x0ne-engine-tlog cosign` command parses the body
  with the same function.
- `x0ne-engine cloud status --limits` shows each limitation the relay
  reported whole, up to 2000 characters, control characters still replaced
  (2026-10-31): the relay's `verdict verifier` line of that day's H100 run,
  1513 characters, was cut at 500 after its first clauses.
- The documents after the H100 run of 2026-10-31. `x0ne_sdk::LIMITATIONS`
  and `GPU_LIMITATIONS` (and their copy in `docs/sdk-e2e.md`) date the TD's
  e2e routes and the GPU path to 2026-10-28, 2026-10-30 and 2026-10-31,
  where they said "ran once". `docs/session-kv-cache.md` adds the measure on
  that run (`ministral3:14b`, n = 3): a second turn's first content at 0.321,
  0.421 and 0.472 s continued, against 1.114, 3.028 and 8.872 s without a
  conversation, for turns of 4,590, 8,590 and 16,590 tokens.
  `docs/benchmarks.md` adds the run's row (first token 1.200 s on the short
  prompt) and its prompt ladder up to 8,048 tokens, from the
  rows in `bench/results/2026-10-31-h100-tdx-cc/`, which `summarize.py` now
  reads (ladder, `kv-cache.jsonl`). `docs/limitations.md` lists the seven
  H100 runs of CUDA inference in cloud mode. SECURITY.md records the `cargo
  audit` and `cargo deny check` of 2026-10-31.
- The relay carries the caller-sealed stream (2026-10-31): `POST
  /v2/e2e/chat` on its local endpoint takes `message/ohttp-chunked-req`
  and passes the TD's `message/ohttp-chunked-res` on frame by frame, as the
  TD writes it, opening nothing; it bounds a stream by its cadence's bytes,
  30 s between frames and 15 min in all, breaks the body on a cut and
  counts each by its code in `/x0ne/v1/status` (`sealed_streams`); a
  request over its largest gets 413 `sealed_request_too_large`. The wasm C
  ABI streams (ops 6, 7, 8) and `js/x0ne-sdk.mjs` has `chatStream`.
  Synthetic evidence only: no sealed stream has crossed a relay on hardware.
- `FEATURES.toml`, the feature list, and `FEATURES.md` written from it by
  `ci/features.py` (2026-10-31): one line per feature with its level (measured
  on hardware, dated by the first run of `docs/hardware-runs.md` that exercised
  it; tested; prototype; planned; replaced), its proof and its limit. `make ci`
  runs the step `features`: the file's rules (plain words in titles and
  sentences, a run date for every hardware entry, no task number or commit
  hash) and a stale `FEATURES.md` refused. The x0ne.co site reads the same file.
- `x0ne-engine cloud up` shows the relay's verdict lines whole, up to 1000
  characters, control characters still replaced (2026-10-31): the Image
  line of that day's H100 run, 321 characters, was cut at 300 before its
  RTMR0 clause, and a check reading it stopped the run. The run, the first
  with RTMR0 pinned on a TD, is in `docs/hardware-runs.md`.
- A conversation's KV cache kept between its turns, on request
  (2026-10-30, `docs/session-kv-cache.md`). `"x0ne": {"conversation":
  "new"}` opens one and the answer carries `x0ne.conversation.handle`, 32
  bytes the TD drew; the next turn sends that handle with its whole history
  and the TD computes only the tokens past the prefix it kept, compared id
  by id. A handle serves one turn, on the route that issued it. One
  conversation is resident at a time, zeroed before any other request runs,
  after 300 s without a turn, after `"end_conversation": true` and after a
  turn that fails. Such a turn signs `kv_cache_shared_between_requests:
  true` and a `context` block (turn, exact tokens reused, a commitment to
  them and one to the handles); the relay and x0ne-sdk refuse a receipt
  that hides it or signs one the request did not ask for. The host learns
  from timing which turns continued a conversation: accepted, and written
  in `docs/limitations.md` and THREAT_MODEL.md. OLMo 2 and 3,
  x0ne Core and the lab server refuse a conversation with a 400. Clearing a
  Qwen3 or Ministral 3 cache now writes zeros over it before dropping it,
  on every request, and growing it zeroes the old buffer. A turn that
  outgrows the cache kept grows it to its reservation, which the served
  context bounds, not to twice its size. Measured on the CPU, Qwen3-0.6B,
  n = 3: a second turn of 4 102, 8 198 and 16 390 tokens answers its first
  token in 1.85, 4.46 and 12.3 s continued, 79.4, 356.8 and 1 722 s from
  token 0 before this change (medians, other jobs on the machine).
- A conversation asked on the caller-sealed stream (2026-10-31). The TD
  served such a turn as a request without a conversation, and the SDK's
  `PendingStream::finish` did not check the context block, so the ask was
  dropped unseen; a whole answer's check did refuse it. The sealed stream
  now serves the conversation with its signed `context`, as the clear
  stream does, and `finish` holds the receipt to the conversation asked:
  no context without one, with one the flag, the context and the handle
  the final chunk hands back, which `StreamAnswer::conversation` carries
  once checked. A model that keeps no conversation refuses one inside the
  stream, 400 by name. In the backend, a panic in the middle of a turn
  zeroes the cache at once rather than when the next request takes the
  poisoned lock, and that request clears the poison: before, every later
  lock took the recovery branch and no conversation continued until a
  restart. The largest real `x0ne` block, which sizes a sealed stream's
  final chunk, now counts a conversation's context and handle: 10747
  bytes, 722 more. The docs restrict the zeroing to Qwen3 and Ministral 3 (OLMo's
  cache is dropped, not zeroed), say that CUDA zeroing is candle's
  `const_set` not yet run on a GPU, and that a handle presented twice
  opens a new conversation that evicts the first.
- RTMR0 is pinned, never only reported (2026-10-31, labs audit of
  2026-10-30). The relay and the SDK admit an image reference only with
  `rtmr0.policy` `gce-endorsed`: a `report` reference is refused when
  loaded, before anything is created, with the new code 104
  (`reference::EXIT_RTMR0_UNPINNED`; `x0ne-engine cloud up` shows it as
  95 with the relay's reason). Every verdict fetches the firmware binary
  Google's endorsement names and pins RTMR0 to the reference's events with
  that binary's configuration volume; `reference::check_image_with` takes
  the measured binary, and the lab's replay of recorded runs is the named
  `check_image_rtmr0_reported`. The SDK's collateral gains
  `gce_firmware_b64`, the binary the carrier fetches from
  `attest::gce_firmware_url` (the relay's `/v2/e2e/collateral` carries it;
  a relay from before serves none, and the SDK refuses its collateral at
  `measurement`); the SDK measures it (`tdvf::verify_endorsed`) and refuses
  it withheld or swapped; `Policy::pinned_rtmr0` and the ABI's `rtmr0_hex`
  pin RTMR0 on top; `VerifiedQuote`/`VerifiedKey` carry
  `firmware_sha384_hex`; `warnings` is empty; a `test-anchor` build takes
  `gce_root_der_b64` in the ABI and `Policy::with_firmware_measured_elsewhere`
  for a replay without the binary, as it takes Intel's test root; a release
  build refuses both. The owner's key policy may
  name `rtmr0` per image (`"gce-endorsed"` or a value). The published
  x0ne-cvm 0.2.x references are `report`: none is admitted until a
  `gce-endorsed` one is published (0.2.7's). Measured on 2026-10-31 on
  Google's binary `53c558fe…` and the boot of 2026-10-18: it passes with
  RTMR0 pinned, as it did on 2026-10-17's log.

- The caller-sealed path streams (2026-10-30, docs/sdk-e2e.md step 6).
  `Client::seal_chat_stream` seals a chat request whose `stream` is `true`
  as chunked Oblivious HTTP (draft-ietf-ohai-chunked-ohttp-08, tested
  against its appendix A), and `PendingStream` opens the answer chunk by
  chunk; the TD answers it on `POST /v2/e2e/chat` under
  `message/ohttp-chunked-req`. Every chunk carries the same plaintext
  length, one every tick, the first a tick after the request, and their
  count is a power of two from 16, refusals and failures included: the
  carrier sees the count, so the answer's duration to a factor of two and a
  bound on its length, its length to a factor of two as well when the text,
  not the model, sets the count; not the text, its pace or its outcome. The
  model's turn, and the queue place it holds, end at the same power of two
  of ticks after the request: two short streams asked together, the second
  ends at 32 chunks instead of 16 (measured 2026-10-30). The cadence is the
  session config's new optional `e2e_stream` (`chunk_bytes` 128 to 4096,
  `tick_ms` 20 to 1000; 256 and 100 when absent, a config without it keeps
  its bytes and digest). A stream that could end past the session's end less
  60 s is cut at the last power of two before it, or refused without running
  the model. The SDK refuses a chunk of another size, out of order, replayed
  or altered, and a stream without its final chunk; the receipt of the final
  chunk is checked over the whole text and against the decoding the question
  asked, as a whole answer's, and the TCB disclosure under `strict`; the
  deltas the final chunk carries come back from `finish`
  (`StreamAnswer::final_deltas`). The SDK bounds a stream's count of chunks
  by its session's end as the TD does (`sealed_stream::cap`), and the TD
  refuses a request chunk of less than 16 KiB but the last before opening
  any. The deltas `PendingStream::push` hands out before `finish` are
  `UnverifiedDelta`s: not verified, and named so. A whole request asking to
  stream is still refused, and a chunked one that does not. Not done: the
  relay does not carry chunked requests, the browser ABI has no streaming
  call. API: `ohttp` gains the chunked request and response; `bhttp` an
  indeterminate-length response reader; `e2e::Gateway::open_stream`,
  `MAX_SEALED_STREAM_REQUEST`, `E2eError::Stream`;
  `CloudConfig::e2e_stream`; `sealed_stream`.

- `up` ends its wait when systemd gave up on x0ne-infer: a status block with
  `ActiveState=failed SubState=failed` after an exit code other than 0, as
  an image whose unit sets `RestartPreventExitStatus=` prints it, ends the
  wait whatever the code and `NRestarts`; systemd's start limit prints the
  same fields. `td_console` counted restarts only, so an exit 103 that
  systemd did not restart left `up` waiting to its boot deadline, then exit
  93. `td_console::Refusal` gains `gave_up` and `reason()`: for a code the
  list does not name, the restarts and the last exit.

- `up` reads a status block of x0ne-cvm 0.2.7 (2026-10-30). Since
  2026-10-27 x0ne-OS prints systemd's fields only on the serial console, no
  `infer_tail=`; `td_console` ended the wait only on a tail, so a final exit
  of x0ne-infer (2, 65, 95, 101…) left `up` waiting to its boot deadline,
  about 12 min on an H100, then exit 93. A block now ends the wait at its
  `end` line on a final code alone, and `up` prints what the code means;
  an older image's tail still gives x0ne-infer's line, and 76 and 77 still
  need its words. `td_console::Refusal::line` becomes an `Option`, and
  `td_console::why` gives a code's meaning.

- x0ne-infer exits 103 when NRAS refuses the key of `--nras-key-file` (401
  or 403 to a call that carried it, 2026-10-30). It exited 75, which may
  pass: under `Restart=on-failure` the TD called NVIDIA with the refused key
  every 10 s, and `up` waited to its deadline. 103 is final in `td_console`.
  Without a key, a 401 or 403 stays 75, as do NRAS's 5xx, 429 and network
  errors. `nras_http::KeyRefused` is the error `attest_gpu` returns for it;
  `attest_v2::EXIT_NRAS_KEY_REFUSED` is 103.

- An NRAS key is 16 characters or more, in `up`'s parser and in
  x0ne-infer's (2026-10-30). The refusal NRAS sends back is printed with
  each copy of the key replaced, and a key of 1 to 3 characters masked
  ordinary text in it.

- The `x0ne-engine` package is the client alone (2026-10-30): `x0ne-engine`,
  `x0ne-engine-cloud` and `x0ne-engine-proxy`. It also shipped `x0ne-infer`,
  the server of the TD, which has no use on a client machine (16.6 MB of
  the 38.0 MB of binaries it installed, measured on 2026-10-30), and so
  conflicted with the `x0ne-infer` package; the two now install side by
  side; the `.deb` goes from 9.9 MB to 5.3 MB. `x0ne-infer` stays its own
  package, built by `make deb` as before.
  `ci/deb-verify.sh` turns red on `/usr/bin/x0ne-infer` in `x0ne-engine`,
  `make deb-check` requires its absence and checks the `--version` of
  `x0ne-infer` from its own package, and `tests/cloud_source_sweep.rs`
  refuses it in `ci/deb/x0ne-engine.ctl`.

- `make deb` builds outside the caches of `make ci` and `make run-fixtures`
  (2026-10-30). The release build compiled in `$CI_CACHE/target`, which
  `make run-fixtures` also writes, with the sccache `make ci` fills: a
  `build.rs`, a test or an example could leave there what the `.deb` then
  shipped. Each release build now compiles in a new, empty directory under
  `DEB_TARGET` (default `$CI_CACHE/deb-target`), removed after, without
  sccache; `DEB_SCCACHE=1` shares it again and `deb.json` records it.
  `make deb-repro` makes both of its builds that way; `$CI_CACHE/target-repro`
  is no longer used. Measured on one commit: 102 s instead of 79 s, and the
  same three `.deb` byte for byte.

- An NVIDIA API key for NRAS (2026-10-30). NVIDIA says API keys "will be
  required in the future for all production deployments". `x0ne-infer
  --nras-key-file <file>` reads one before any evidence (exit 101 on a
  refused file) and sends it as `Authorization: Bearer` on the attest call;
  the JWKS fetch stays anonymous, and a refusal NRAS sends back is printed
  with the key replaced. `x0ne-engine cloud up` and `run` take
  `--nras-key-file <file>` (mode 0600 or stricter, read with the command
  line) and hand the key to the VM in its own metadata item,
  `x0ne-nras-key`, never in `x0ne-config`, whose bytes are hashed into the
  quote and published. Neither prints the key, its file's path in a
  refusal, nor a Google error that quotes it. The CVM image reads the item
  since x0ne-cvm 0.2.7; an older image calls NRAS without the key
  (docs/cloud.md, "The NRAS key"). `nras_http::attest_gpu` takes the key as
  a third argument; `attest_v2::SystemEvidence` gains `nras_key`.

- `x0ne-engine cloud up` and `run` open the VM's tcp:443 to this machine's
  public IPv4 address alone (2026-10-30). The firewall rule admitted
  `0.0.0.0/0`: anyone who found the VM's address could open connections,
  which mutual TLS refused, each holding one of the TD's handshake slots
  (10 s, 256 in flight, unchanged). The address is asked of Google's name
  server (`ns1.google.com`, TXT `o-o.myaddr.l.google.com`) before the quote
  and printed on a `Firewall:` line; no public address found is exit 91,
  with nothing created and no fallback to `0.0.0.0/0`. `--allow-from <CIDR>`
  (IPv4, repeatable) names the sources instead, for a NAT whose outgoing
  address moves or a network that blocks DNS; `--allow-from 0.0.0.0/0`
  restores the old rule, printed in yellow. A session whose client address
  changes loses its connections to the VM (docs/cloud.md, "The firewall
  rule").

- `make coverage`: test coverage per crate, held to a committed baseline
  that only goes up (2026-10-30). The tests are built and run with the
  workspace members under `-C instrument-coverage` (cargo's
  `RUSTC_WORKSPACE_WRAPPER`) and the features of `make ci`, in a no-network
  container that adds to the CI image the `llvm-tools` of the pinned
  toolchain (`llvm-profdata`, `llvm-cov`), outside the caches of `make ci`
  and under its build lock; it is not a step of `make ci`. Red when a test
  fails or a crate loses more than 0.2 point of lines or regions against
  `ci/coverage-baseline.json`; `make coverage-update` raises the baseline.
  A file two crates compile through `#[path]` counts once, under the crate
  that holds it. First baseline: 91.25 % of lines and 90.71 % of regions
  over the five crates, from 89.01 % of lines (x0ne-engine-cloud) to
  96.62 % (x0ne-confidential). One run took 146 s on an empty target
  (build 73 s, tests 71 s) and 74 s warm, lock wait apart; runs of one code
  differed by 0.03 point at most. Doctests are not counted (nightly only).
  See `docs/ci-local.md`.

- The caller-sealed path's key document binds the TD's session config
  (2026-10-30). Its quote's `report_data` named the OHTTP key, the caller's
  nonce and the receipt key only: before sealing, the caller saw neither
  the model the TD was configured for, nor when its session ends, nor the
  relay's TLS key. The binding now also takes the TD's boot binding (its
  boot nonce, SHA-256 of its TLS key, SHA-256 of its session config's exact
  bytes), the key document carries those bytes, and `attest::verify_key`
  returns the config (`VerifiedKey::session_config`, also in the ABI's
  verify summary) with the boot nonce and the TLS key's digest. A new last
  step, `session` (88), refuses a config whose `attest` is not the
  policy's, or that ends within 60 s of `now`; `Client::seal_chat` refuses
  at the same edge, as the relay refuses new requests then. Breaking, with
  no compatibility kept: labels `x0ne-e2e/v2/ohttp-key` and
  `x0ne-e2e/v2/ohttp-gpu-key`, schema `x0ne-e2e/key/v2`; a v1 document is
  refused at `schema`, and an SDK before this refuses a v2 document by its
  field `session_config_b64`. API changes: `e2e::key_binding` and
  `key_binding_gpu` take an `e2e::Boot`; `KeyDocument` gains
  `session_config_b64`, `boot_nonce_hex`, `tls_spki_sha256_hex`;
  `e2e_gateway::E2e::from_seed` takes the boot binding and the config's
  bytes, and its `report_data`, `gpu_nonce` and `document` no longer take
  the receipt key. The model's catalogue name is shown, not mapped to the
  digest the caller pins (a new line of `x0ne_sdk::LIMITATIONS`).

- `docs/benchmarks.md` and `bench/results/` (2026-10-30): every speed
  figure whose rows were kept, with its machine, build, prompt, n, spread,
  date and raw data, recomputed by `bench/results/summarize.py`. H100 in a
  TDX guest with the GPU in CC mode, through the relay: `qwen3:8b`,
  `qwen3:32b` and `ministral3:14b`, 2026-10-17 to 2026-10-28. The page
  lists what was not measured, the cost of confidential computing first.
  `docs/related-work.md` points to it.

- The caller-sealed path's replay window no longer goes back with the TD's
  clock (2026-10-30). `e2e::ReplayGuard` judged a request's window on the
  TD's wall clock, which the host sets: once that clock passed a request's
  window its entry was dropped, and with the clock set back the same sealed
  request was answered a second time (still sealed to its caller; one more
  size and timing sample for the host). The window's old edge now follows
  the highest time the guard has seen, and an entry is dropped only past
  it. A clock pushed forward makes the TD refuse honest requests until real
  time catches up, a denial the host could already cause.

- The caller-sealed path through the relay (2026-10-28, prototype). The
  relay's local endpoint carries `GET /v2/e2e/key`, `POST /v2/e2e/chat`
  and `GET /v2/e2e/collateral` for a TDX session, behind its Bearer: it
  forwards the key document and the sealed messages as they came, serves
  Intel's and Google's collateral from its own fetches (both of Intel's
  updates), and holds no key that opens a sealed message. A GPU session's
  key document carries NVIDIA's verdict (`nras_answer`, `nras_jwks`),
  fetched by the TD for that document's nonce after the quote; NRAS down,
  503 and no document. x0ne-sdk now judges a key document as the relay
  judges its session: Intel's collateral under the relay's TCB policy
  (`transparent` by default, `strict` on request, the same disclosure
  returned with the verified key, every receipt held to it under
  `strict`), and the image reference admitted by the relay's own function
  (`reference::admit_offline`): signed by the archive key the build pins
  (99), shown in the log it pins under a fresh witness quorum (100).
  Before the archive key is pinned, a build admits no reference. API changes:
  `attest::Policy` takes an `AdmittedReference` (`attest::admit_reference`)
  and a `tcb_mode`; `Collateral` gains the `standard` TCB info and QE
  identity; `KeyDocument` gains `nras_answer` and `nras_jwks`;
  `Pending::open_verified` takes `now`. The ABI requires
  `reference_sig_b64`, `reference_tlog_proof` and, in op 3, `now`, takes
  `tcb` and, in a `test-anchor` build, `reference_keys: "test"`, and returns
  a refusal's relay exit code as `code`. A caller from before reads a CPU
  key document as before and refuses a GPU document and the new collateral
  by the unknown field's name; an ABI caller from before is refused by the
  fields it lacks. `js/x0ne-sdk.mjs`: `connect` takes `{ headers }` for
  the relay's Bearer. Tested on synthetic evidence only. x0ne-sdk now
  depends on `sha1` (OpenPGP fingerprints), already in the workspace lock:
  no new package.

- `x0ne-engine cloud image delete <x0ne-cvm-…>` (2026-10-30): the
  image `image import` made is deleted without gcloud, after "yes" or with
  `--yes`. Only an `x0ne-cvm-*` name carrying the label
  `x0ne-engine=image`, never one a VM boots from (a disk made from it and
  attached to a VM: exit 91, the VM named); a missing image is said, exit 0;
  a deletion Google does not confirm is exit 94 with the gcloud command.
  `image import` now names this command where it named gcloud. Tested
  against the fake Google Cloud only.

- NVML judges the GPUs before their evidence is read (2026-10-30).
  `x0ne-infer` reads each GPU's architecture (`nvmlDeviceGetArchitecture`)
  and refuses, exit 74, when it is not the one `--gpu-arch` names (HOPPER by
  default, BLACKWELL), instead of sending NRAS the claim unchecked. The CC
  state (`nvmlSystemGetConfComputeState`: on, production, devtools off),
  judged until now for the e2e key document only, is judged on every path:
  `run`, the cloud TD's session and its key document. One function,
  `attest_v2::judged_gpu_evidence`, is the only way to the reports. Tested
  against a simulated NVML; a real H100's answers to the two calls are read
  on the next run on one.

- Three wrong texts of the CLI (2026-10-30). `x0ne-infer --help` no longer
  prints a price for `--attest cpu` ("about 0.24 $/h against 3.72 $": an
  estimate of 2026-08 against a rate the run of 2026-10-14 had measured at
  6.55 $/h); the price is the quote's, read at run time. `x0ne-engine cloud
  --help` says `up` and `run` serve a catalog model with a pinned
  served-model digest, sharded or not, where it said "in one weight file".
  A tokeninfo request that gets no answer (offline, DNS, TLS) is said as a
  network failure, no longer as "Google refused the token" (exit 91 as
  before).

- TCB policy, transparent by default (2026-10-28). The TD and the relay
  accept a TDX platform and its QE that Intel's `standard` tcbInfo and QE
  identity (evaluation 20 or later) rate UpToDate or SWHardeningNeeded, and
  also read Intel's latest (`update=early`, evaluation 22 or later) without
  refusing on it. What the latest says is named: the platform's TCB date,
  its status, the advisory IDs Intel lists against its level, its TDX
  module and its QE, in the attestation document (`tcb`), the relay's
  verdict (`tcb` and the last summary line), every signed receipt (optional
  field `tcb`, schema `x0ne-infer/tcb-disclosure/v1`; a receipt without it
  verifies as before), `/x0ne/v1/status` (`tcb`, `tcb_last_receipt`) and the
  client's VERIFIED line. When the latest cannot be read, the line says so
  and names no advisory. This mode discloses; it does not refuse a platform
  that lacks the fixes it names. `--tcb-policy strict` (relay, `x0ne-engine
  cloud up` and `run`) refuses as every verdict did since 2026-10-22, and
  ends a session whose receipt names a view it would refuse (76);
  `--min-tcb-evaluation` implies it. The relay's `config` event names the
  mode; the lifecycle refuses a relay looser than asked (95). On SEV-SNP, a
  bulletin is imposed once a year old and named before under
  `transparent`, every bulletin imposed under `strict`; the bulletins'
  dates are constants of `snp.rs`. Measured on Intel's answers of
  2026-10-28: the GCE a3-highgpu hosts of FMSPC `00806f050000` are UpToDate
  under evaluation 20 and OutOfDate under 22 (INTEL-DOC-10003,
  INTEL-SA-01439, INTEL-SA-01442). Each x0ne block of a stream grows by
  about 2.3 KB. No new dependency.

- `make ci` lints as the GitHub `clippy` job does: `-D warnings`, then the
  `PEDANTIC` list of `.github/workflows/ci.yml` (`clippy::pedantic` minus 34
  allowed lints), read by `ci/clippy-pedantic.sh`, one list for both CIs. A
  list the script cannot read fails the step; `ci/tests/ci-sh.sh` tests it.
  On the tree of 2026-10-29 the list found 89 remarks of 26 lints that only
  GitHub would have reported. 77 are fixed in the code; 12 are allowed where
  they stand, each with its reason: the forced inlining of the oblivious
  tokenizer's branch-free helpers, serde's `skip_serializing_if` signature, a
  result fallible only with `cuda`, the four switches of `converse`, a permit
  held unread, the scripted model's panic, an async test helper its callers
  await. Visible changes: `OwnerKey`'s `Debug` ends with `..` (the policy
  digest was already left out), and the probe's HTTP reads go through a heap
  buffer instead of a 16 KiB one held in each future.

- Dependencies build optimized in the dev and test profiles
  (`[profile.dev.package."*"] opt-level = 2`); the workspace crates stay at
  opt-level 0 with `debug_assertions`. `cargo test --workspace --all-targets`
  built from cold in 78 s instead of 59 s (one build each), then ran in 71 s
  instead of 126 to 152 s, the 987 tests of the x0ne-infer lib in 35 s
  instead of 88 to 110 s (3 runs each; CI image, 24 jobs, no sccache,
  2026-10-30). The release profile is
  unchanged, and `tests/cloud_source_sweep.rs` holds the new table to the
  opt-level alone.

- AMD SEV-SNP verdicts in the relay, client side. A `--reference`
  with schema `x0ne-engine/snp-reference/v1` (signed by the archive key and
  shown in the transparency log, like the TDX reference) pins products, a
  TCB floor raised over the default of AMD's bulletins, SMT, the cloud
  providers whose VLEK is accepted and the predicted launch digests. The
  relay then verifies the document and each connection's report: the
  VCEK or VLEK chain to the pinned ARK, ASK or ASVK, AMD's CRL fetched from
  KDS (Date within 300 s), the floor, DEBUG, migration and VMPL refused,
  SMT refused unless written, `MEASUREMENT` among the predictions,
  `REPORT_DATA` bound as on TDX, and the GPUs through NRAS on a nonce over
  the report's signed bytes. A session pins one TEE: TDX evidence under an
  SEV-SNP reference, or SEV-SNP evidence under a TDX one, is refused (88),
  with no fallback. Sealed weights (`--model-key`, 89) and the SDK's
  caller-sealed path stay TDX only. A genuine Azure paravisor report is
  refused by the client path (test). No SEV-SNP machine has run this code.
  Documents gain an optional `snp` field, omitted for TDX. No new
  dependency.

- `response_format: json_schema`, and the decoding signed in the receipt
  (2026-10-30). Both servers and the sealed path read `response_format`:
  `text`, or `json_schema` with a schema of the subset `json_schema.rs`
  compiles (`type`, `enum`, `const`, `properties`, `required`,
  `additionalProperties: false`, `items`, `minItems`, `maxItems`), each
  token masked to keep the answer a document of it; any other keyword, a
  schema past a bound (16 KiB, 10 levels, 512 subschemas, 64 properties,
  256 enum values), `json_object`, `stop` beside a schema, and a tokenizer
  whose decoder is not `ByteLevel` or `Fuse` get a 400 naming it before
  the model runs. The answer takes at most one space between two tokens
  (Outlines' default); numbers are bounded (18 digits before the point, 2
  of exponent). Receipts gain two optional signed fields: `sampling`
  (`temperature`, `top_p`, `seed`) when the answer samples,
  `json_schema_sha256_hex` under a schema; a greedy answer without a
  schema signs the same bytes as before. A seed the server draws is now
  below 2^53. The TD checks the text against the schema before it signs;
  the relay and x0ne-sdk hold every receipt to the request sent (sampling,
  seed, schema digest, text) and stop (65) or refuse otherwise.
  Compatibility: a relay from before refuses a receipt that carries either
  field (it re-serializes the receipt without them, and the bytes differ
  from the signed ones); an SDK from before checks the signature over the
  signed bytes and accepts such a receipt without reading the two fields.
  A relay or SDK from now refuses a sampled answer from a TD from before,
  whose receipt signs no `sampling`. Mask cost on Qwen3-0.6B (release
  build, Ryzen 9 7900X): the vocabulary's table 79 to 104 ms once per
  loaded model, 144 µs per token on average, 343 µs at worst. Tested on
  the tiny qwen3 fixture, synthetic evidence, and `x0ne-infer-lab` on
  Qwen3-0.6B on the CPU; no TD run.

- AMD SEV-SNP guest: x0ne-infer serves from an SEV-SNP guest. configfs-tsm's
  `provider` names the evidence: `sev_guest` gives a report (0x4A0 bytes)
  followed by the VCEK or VLEK that signed it, taken from the auxblob's
  certificate table, else from `/run/x0ne/snp-vcek.der`; an unknown or
  missing provider, a report of another length, one that does not parse or
  names no signing key, a malformed table: no evidence, exit 64. The guest
  verifies its own report to a pinned AMD root and AMD's CRL before it
  binds a socket; the CRL is read from `/run/x0ne/snp-crl.der`, and without
  it the guest does not serve (exit 76) (`verify::cryptographic_snp`, and
  with the GPU product
  `cryptographic_snp_with_gpu`: the report and the NRAS token for the same
  binding); DEBUG, a migration agent, VMPL other than 0, another binding or
  another GPU nonce: exit 65. `REPORT_DATA` is the TDX binding
  (`BindingV2::snp_report_data`, `BindingV3` per connection). The
  attestation document and each session document carry it as
  `snp.evidence_b64` with an empty TDX quote; the session GPU nonce hashes
  the report's signed 0x2A0 bytes. No event log, no Intel PCS: the standing
  is not renewed and lapses after 24 h. Sealed weights (exit 96) and
  `/v2/e2e/key` (501) stay TDX only. The relay reads `snp` under an SEV-SNP
  reference (above); no SNP guest has run this code. The README's statement
  stands.

- `x0ne-engine snp-measure` and `snp_launch`: the SEV-SNP launch measurement
  of a QEMU guest (OVMF with its SEV metadata, kernel hashes table, one VMSA
  per vCPU), predicted offline; equal to virtee/sev-snp-measure 0.0.13 on
  seven generated cases and on our AmdSev firmware with the x0ne-cvm UKI.

- Loading a model logs which default system message a chat without one
  gets (release digest, length, sha256), or that none is added.

- SECURITY: config.json sizes are bounded at parse, in every family:
  layers, heads, key-value heads, `head_dim` (even), hidden, intermediate,
  vocabulary and positions, heads x `head_dim` and positions x `head_dim`.
  u64::MAX layers panicked, 2^40 layers aborted the process on an
  allocation, a 2^62 `head_dim` panicked at load. A RoPE base or norm
  epsilon at or below zero, a YaRN factor below 1 and a negative Ministral 3
  query-scale beta are refused; they were served as numbers.

- SECURITY: a weight stored as anything but BF16, F16 or F32 (an integer,
  FP8) is refused when read, in every family, served and lab paths alike.
  It was cast to a float in silence: quantized storage without its scales,
  served as another model. The lab no longer loads Ministral 3's vision
  tower, which it dropped after loading. The pinned releases measured are
  all BF16 (qwen3:1.7b not measured: no copy at hand).

- SECURITY: a tokenizer.json in which a marker the family's chat template
  writes is not a special added token is refused at load, in every family.
  As a plain word, a message could type it and write a turn of another role.
  Before, Ministral 3's `</s>` alone was checked, by name, which a plain word
  satisfies. The tokenizers the catalog pins all pass.

- Ministral 3 Instruct 2512 (Mistral AI, Apache-2.0), text model, BF16:
  catalog entries `ministral3:14b` and `ministral3:8b` with their served
  digests pinned. `arch.rs` reads `ministral3` alone or inside its
  vision-language envelope (`Mistral3ForConditionalGeneration`,
  `text_config`, tensors under `language_model.`); the vision tower's tensors
  are accepted and never read, any other tensor is refused at load. YaRN with
  `mscale` as transformers' `get_mscale`, Llama 4's query scale; a
  `quantization_config`, a sliding window, an unknown RoPE type or key is
  refused. `ministral3.rs` runs on the Qwen3 fork's in-place KV cache and
  matches transformers 5.19 on tiny random models: logits within 7.0e-6,
  attention 5.5e-6, greedy tokens, and 16448 positions with the 14B's real
  RoPE, where leaving the query scale out moves the logits by 0.18. With
  its real weights, pulled and checked, on the CPU in F16 against
  transformers in F16: same ids, same top token at the last 120 positions,
  same 32 greedy tokens on four prompts, logits within 0.26 (8B) and 0.60
  (14B). The servers' own dtypes (F32 on the CPU,
  BF16 on the GPU) are not part of this comparison. The chat template is the release's
  `chat_template.jinja`: text and token ids equal transformers', on the
  published tokenizer.json. Its default system message is given to the two
  pinned digests only and keeps `{today}` and `{yesterday}` as written: a
  prompt never depends on the host's clock. The template writes `<s>`; the
  post-processor adds none (`Prompt::mark_specials_written`). `[INST]`,
  `[/INST]`, `[SYSTEM_PROMPT]`, `[TOOL_CALLS]`, `<s>` or `</s>` typed in a
  message stays text. `</s>` ends an answer; a tokenizer without it is
  refused at load. Token ids compared with mistral_common 1.13.0, Mistral's
  reference tokenizer, on 307 conversations: 279 equal, 4 differ only by the
  spaces it cuts at the end of an assistant turn (the published template
  keeps them), 24 it refuses (a system turn after an assistant turn). The
  README's statement stands.

- SECURITY: a transparency log pin that names one witness key under two
  names, or the log's own key as a witness, is refused like a missing pin
  (C2SP `tlog-policy`, `tlog-cosignature`): a cosignature does not sign its
  cosigner's name, so that key counted once per name toward the quorum, and
  the log could cosign its own checkpoints. No log is pinned yet
  (`transparency::RELEASE` is `None`); the check holds the first pin.

- The caller-sealed path attests the GPU. On a `cpu+gpu` session
  the TD's key document carries each H100's report, read for a nonce derived
  from the caller's, and its certificate chain, and its quote binds them
  (`x0ne-e2e/v1/ohttp-gpu-key`); the 404 of GPU sessions is gone. x0ne-sdk
  takes `attest` (`cpu` or `cpu+gpu`, required in the ABI), builds NVIDIA's
  NRAS request (op 5), which the carrier forwards blind (`/v2/e2e/nras`),
  and checks NRAS's answer at a new last step, `gpu`, with the relay's own
  code (`nras.rs`): signature under NVIDIA's pinned intermediates, nonce,
  verdicts, debug, per-GPU tokens and `submods`, expiry and `nbf` (now read
  by the relay too), and each `ueid` against a chain the quote bound. The
  TD reads NVML outside the quote lock, and only from GPUs NVML says are in
  production CC with devtools off. Tested on a simulated NVIDIA, NVML and
  NRAS only, never on an H100; the README's statement stands.

- The TD's e2e key document carries the TD's event log (`ccel_b64`), the
  bytes its attestation document carries for the relay
  (`CloudState::ccel`): x0ne-sdk replays it since its alignment on the
  relay, and refused every key document without it at `measurement`.
  Tested on synthetic TDX evidence only (`e2e_td_tests`); the README's
  statement stands.
- An error quotes at most 64 bytes of a field name or JSON path of the
  request, cut on a character (`openai::error::quoted`): the validator's
  unknown fields, `messages[].`, `x0ne.` and `stream_options.` names, and
  the path in axum's JSON syntax error. A name of 20001 bytes made the TD's
  padded refusal 20480 bytes, where a 200 is 16384; 16384 now, on the TD
  served directly and the cloud TD (2026-10-27).

- Every refusal of `/v1/chat/completions` on the TD that names part of the
  request leaves as a whole answer's refusal: its status and message, padded
  to 16 KiB with a head of a 200's length, a second after the request. That
  is a stream's refusal before its turn (the validator's, a model not served
  here, empty messages, an adapter, a lease expired) and a body axum's JSON
  extractor rejects (400 naming where the JSON broke, 413, 415). They left
  at once, unpadded: a stream naming an unknown field of 4 or 400 bytes got
  a 400 of 264 or 660 bytes, a body whose JSON broke after a name of 4 or
  400 bytes a 400 of 162 or 559 bytes, each within 1 ms; 16 KiB at 1.00 s
  for all of them now, on the TD served directly and the cloud TD
  (2026-10-27). A full queue, a full set of open streams and a foreign
  `Host` still get a short status at once, their sentences set by the
  server.

- A stream refused once its turn began, by the backend (a prompt past the
  context, a lease expired at the model, a failure before the first token)
  or by a panic before the first token, leaves as a whole answer's
  refusal: its status and message, padded to 16 KiB with a head of a
  200's length, a size of ticks after the turn began. It left at once,
  unpadded: a 413 of 122 or 130 bytes for prompts of 9 or 123456789
  tokens, a panic's 500 when it happened; 16 KiB at 1.00 s for all three
  now (2026-10-27, found in an internal review).

- A field of the wrong type is a 422 that no longer quotes its value.
  serde's sentence did (`invalid type: string "…", expected a sequence`),
  and a stream's refusal left at once, unpadded: 122 and 716 bytes for
  values of 6 and 600 bytes, 111 for both now (2026-10-27, found in an
  internal review). The sentence keeps the kind found and the type expected
  (`openai/request.rs`); one it does not know is `a field of the request
  has the wrong type`. The TD, the relay and the lab share this validator.

- A whole answer's panic on `/v1/chat/completions`, the model's or the
  receipt's salt draw after it (no `/dev/urandom`), is contained as on the
  sealed path: a 500 padded to 16 KiB, a head of a 200's length, at the
  turn's end. It unwound to a 134-byte 500 that left when the generation
  ended: 0.30 s and 0.60 s for generations of 300 and 600 ms, 1.00 s for
  both now (2026-10-27, found in an internal review).

- The cloud TD serves the caller-sealed path of `x0ne-sdk`
  (`docs/sdk-e2e.md`, step 1): `GET /v2/e2e/key?nonce=` returns a key
  document whose TDX quote binds an OHTTP key drawn at boot and the session's
  receipt key to the caller's nonce, one quote per document under the session
  quote's limits; `POST /v2/e2e/chat` opens a `message/ohttp-req`, admits it
  once, answers it through the attested chat handler (queue place, padding,
  turn cadence, signed receipt) and seals the answer. A request that does not
  open, is replayed, stale or of another type gets one 400 after one second;
  the gateway (`x0ne-infer/src/e2e_gateway.rs`) has no log macro. Both routes
  wait for the model; a `cpu+gpu` session answers 404, its document carrying
  no GPU evidence yet. The simulated bench now runs the same gateway. Tested
  on synthetic TDX evidence only (`e2e_td_tests`): through a carrier that
  records every byte it moves, the question and the answer appear in none of
  them, and the caller refuses a flipped quote, a swapped key, a wrong pinned
  measurement and an altered request or answer. `make e2e-local` runs it
  with the wasm32 build in Node and Qwen3-0.6B on CPU (measured 2026-10-26:
  the sealed answer equals what the model generated). Not on TDX hardware;
  the README's statement stands.

- The sealed path, after an internal security review (2026-10-27): a panic of the
  model's code on `/v2/e2e/chat` is answered as any failure, a padded 500
  sealed inside a 200 at the turn's cadence, where it was a 500 in clear at
  once; `GET /v2/e2e/key` draws on a quote bucket of its own, so key
  documents no longer close `/v1` to new sessions; the chat route reads the
  largest request the SDK seals (`x0ne_sdk::e2e::MAX_SEALED_REQUEST`, 4 MiB
  and 55 bytes) and no byte more, where axum's implicit 2 MiB refused
  questions the SDK had sealed; the OHTTP private key is held in
  `memfd_secret` (else `mlock`) as the model key's recipient key, exit 97
  without it; a lone surrogate (`\ud800`) in the question is refused by the
  caller before sending and by the gateway at opening, where the TD opened
  it and then refused it. `x0ne_sdk::hpke::PrivateKey` no longer exposes its
  bytes (`to_bytes` removed; `held_by` moves it into caller memory). Tests
  added: refusals bounded from above, requests past the AEAD that are no
  chat request, a request dated ahead. Left, in `docs/sdk-e2e.md`: the
  replay window's clock, the binding's `config_sha256`, NRAS.

- SECURITY: the issuer chain of a tcbInfo and of a QE identity must be
  Intel's: exactly the TCB Signing certificate and the root, the root marked
  CA by basicConstraints, the signer naming it as its issuer (`tcb.rs`,
  sharing `pcs::intel_shape` with the PCK chain; read by the TD, the relay
  and the SDK). Any chain of two or more certificates whose signatures
  verified was taken, so whoever holds one PCK's private key could issue a
  "TCB Signing" end entity under it and sign an UpToDate tcbInfo:
  `verify_live` accepted it (measured on the bench, 2026-10-27), and only
  the root CA CRL check, which requires each collateral signer to be issued
  by the root, refused it. Found in an internal review of the SDK
  (2026-10-27).

- SECURITY: a TDX quote's PCK chain must have Intel's shape: exactly the
  PCK, its CA and the root, each naming the next as its issuer, the CA and
  the root marked CA by basicConstraints and the PCK not (`pcs.rs`, read by
  the TD, the relay, the SSH client and the SDK). Any chain of two or more
  certificates whose signatures verified to the pinned root was taken, so
  whoever holds one PCK's private key (TEE.fail) could issue a leaf under
  it and sign the QE report with that leaf. The relay refused that quote
  only later, because the leaf's issuer is no Intel PCK CA
  (`tcb::pck_crl_ca`, 65); the SDK did not call `pck_crl_ca` and accepted
  it (fixed, and the SDK now calls it). Both refuse it at the chain now.
  Lengths in the quote's signature section are added with checks, which a
  32-bit build (the SDK's wasm32) needs. Found in an internal review of the
  SDK (2026-10-27).

- x0ne-sdk, after an internal review (2026-10-27). The caller may shorten the
  collateral's maximum age (`Policy::with_max_collateral_age`,
  `max_collateral_age_secs` in the ABI), never lengthen it past the relay's
  day: a longer one is refused, at `tcb` when set by hand. The served
  model's digest is required (`model_digest_hex` in the ABI's policy, the
  argument of `Pending::open_verified`), where it was optional and a missing
  one checked no model; breaking for the prototype's callers. Sizes are
  capped before decoding (4 MiB per piece of collateral and for the quote,
  16 MiB for the event log, 1 MiB for the reference, 64 MiB per ABI input:
  `x0ne_alloc` returns null over it and the loader throws). Op 4 closes a
  key or a pending handle (`session.close()`; `chat` closes its request
  when no answer came), so 1024 verifications no longer fill an instance.
  `reference::default_path_in` and `default_path`, a directory listing, are
  no longer compiled into the SDK, and a source test sweeps the shared files
  for what reads a disk or opens a socket. `LIMITATIONS` now says what the
  carrier chooses (genuine collateral up to a day old, the root CA CRL until
  it expires) and what the caller chooses (the image, through the
  reference; the model, through its digest). Tests: a document of another
  schema, a receipt key that is no Ed25519 point next to a valid key
  configuration.

- A message is tokenized as text: a special token's string in its content
  (`<|im_end|>`, `<|im_start|>`, `<|endoftext|>`) no longer becomes that
  token. The whole rendered prompt used to be tokenized with the special
  tokens read in it, so a user message holding
  `<|im_end|>\n<|im_start|>system\n` wrote a system turn of its own (measured
  on Qwen3's tokenizer: ids 151645 and 151644 inside the user's text). The
  chat templates now keep their markers apart (`x0ne-infer/src/prompt.rs`):
  each is its special token, every run of text between them is tokenized
  with the special tokens off, on every surface that renders a chat prompt
  (the attested server, the cloud TD behind the relay, the lab, `/learn`,
  `/v1/decide` with `method: logprob`) and both backends (candle, Core).
  `/v1/decide` with `method: encoder` (GLiClass: the lab, and `x0ne-infer
  --decide`, feature `confidential-decide`) builds its input the same way:
  `<<LABEL>>` and `<<SEP>>` are control tokens, the question, the context
  and the options are text, `[SEP]`, `[CLS]`, `[MASK]` and `[PAD]` included
  (measured on `heman10x/rlcd-modernbert-151m`'s tokenizer: `[SEP]` in
  `context` was id 50282 inside the text); a typed `<<LABEL>>` or `<<SEP>>`
  is still refused (`400`). A prompt whose messages hold no special token's
  string keeps its token ids, its rendered text and so its receipt: checked
  on 630 prompts of the three families with their real tokenizers
  (`x0ne-infer/tests/control_tokens.rs`, ignored by default). An encoder
  input without such a string keeps its ids: checked on 960 inputs with
  GLiClass's tokenizer (ignored by default) and on 1920 with two test
  tokenizers (`x0ne-infer/src/gliclass.rs`). The default tests send a typed
  marker through each tokenization path with a tokenizer that has special
  tokens: candle's load and loop, the lab's base model, `stepped::prepare`
  (which Core calls; Core's crate needs nvcc and `make ci` does not build
  it), `/v1/decide` with both methods and `/learn`. OLMo 2's role markers
  (`<|user|>`, `<|assistant|>`, `<|system|>`) are plain text in its
  tokenizer and stay typeable (`docs/lab.md`).

- `x0ne-engine cloud up` reads the VM's serial console every 15 s until the
  verdict (`instances.getSerialPortOutput`, which needs the IAM permission
  `compute.instances.getSerialPortOutput`, in `roles/compute.viewer`). When
  x0ne-OS's status block shows x0ne-infer exiting on a final refusal after
  at least two restarts, `up` prints the line x0ne-infer gave (cleaned, at
  most 300 characters), saves the console, deletes the VM and exits with the
  new code 97. It used to wait for the boot deadline: 39 minutes after
  creation for `qwen3:32b`, billed. The final refusals are a list with their
  reasons (`x0ne-engine-cloud/src/td_console.rs`): 76 only for Intel's
  rating of the host, not for its PCS unreachable; 75 (NRAS unreachable) and
  96 (pull failed) keep waiting. The console is unverified: it can end the
  wait and nothing else, and a console that cannot be read leaves the wait
  as it was. Found by the GCE run of 2026-10-25, whose second VM the TD
  refused at every exit its console shows (exit 76, 25 restarts in 5 min).

- `measurements/0.2.0-beta.7.json` names the RTMR1 and RTMR2 of the
  2026-10-25 GCE run of x0ne-cvm 0.2.4 (disk tar `03d53a41…`, built from
  the 0.2.0-beta.7 sources of that day, verdict VERIFIED), equal to the
  prediction of that image's build; it named those of x0ne-cvm 0.2.3
  (2026-10-18). Same TEST
  provenance: a release build refuses it (exit 87). That run's event log,
  TD report and reference are kept as fixtures
  (`x0ne-infer/tests/{ccel,reference}`). A second VM of that day was
  refused by the TD itself, its host's TCB level being OutOfDate under
  evaluation 22 (exit 76, `docs/hardware-runs.md`).
- `x0ne-infer-lab` (both binaries, `--backend candle|core`) answers
  `"stream": true`; it used to refuse it with a 400. OpenAI's chunks: the
  role chunk with the 200, one chunk per piece of text as the model gives it,
  the final chunk with `finish_reason` and the `x0ne` block
  (`"confidential": false`, also on the role chunk), the usage chunk with
  `stream_options.include_usage`, then `[DONE]`. Refusals before the first
  token stay the JSON statuses of a whole answer; a failure after the 200 is
  one error event with a fixed message, then the end of the body; a caller
  that hangs up stops the generation at its next token. Unlike the attested
  server, the lab pads nothing and writes each chunk at once: the size and
  timing of its chunks follow the tokens. It is not confidential, so it has
  nothing to hide them for (docs/lab.md). The relay's writer of unpadded
  chunks moved to `sse.rs` and serves both; the generation's end is decided
  by one function for the attested server and the lab (`stream::outcome`).
- The binaries that ship are stripped by the build, and a comment no longer
  changes them. The five binaries of the `.deb` packages (`ci/inside-deb.sh`,
  `ci/inside-deb-lab.sh`) and the four of the release archive
  (`scripts/sbom.sh`) build with the Cargo profile `release`, which now keeps
  line tables, then `ci/strip-release.sh` moves the symbols and line tables
  to a debug file, strips the binary with the flags of `dh_strip`, and
  recomputes the GNU build-id over the stripped bytes, in the binary and in
  its debug file, `debug/.build-id/<2 hex>/<38 hex>.debug`. `make deb` stages
  the debug files next to the `.deb`, a release attaches them as
  `x0ne-infer-<target>-debug.tar.gz`, and `gdb -iex 'set
  debug-file-directory <dir>'` reads a core dump with them
  (`docs/ci-local.md`, "Stripped binaries"). Until now `SHA256SUMS.engine`
  and `.engine-lab` named the unstripped files, while the `.deb` shipped the
  output of `dh_strip` (equivs): other bytes. They now name the stripped
  files, and `make deb` checks that each `.deb` ships exactly its binaries,
  by name and sha256 (`ci/deb-verify.sh`): none missing or added, no other
  package, nothing executable outside `usr/bin`. `ci/strip-release.sh`
  refuses, before it touches the file, a binary whose build-id is not one
  GNU note alone in its section. The step `ci-sh` of `make ci` plays both
  scripts on inputs it builds (`ci/tests/strip-release.sh`,
  `ci/tests/deb-verify.sh`), and the step `test-release` tests the profile
  that ships (`--release`). Measured on 2026-10-25 with a comment
  appended at the end of three source files: the loaded sections of the five
  binaries stayed the same, their files did not. ThinLTO renamed 923 local
  symbols of `x0ne-infer` (`.llvm.<hash>`), the linker's build-id covers
  them, and the `.gnu_debuglink` that `dh_strip` adds carries a CRC-32 of
  them. With `ci/strip-release.sh` the five are byte for byte the same. A
  new step of `make ci`, `comment-invariance` (`ci/comment-invariance.sh`),
  builds them twice around such a comment and is red when one differs. A
  comment that adds a line before code still changes a binary: panic
  locations and log call sites carry line numbers.
- One recipe for what ships: `[profile.release]` of `Cargo.toml`, the profile
  `make ci` tests. A plain `cargo build --release` now keeps symbols and line
  tables (`x0ne-infer` 142 MB, 15.7 MB once stripped by
  `ci/strip-release.sh`). `ci/release-env.sh`, run first by `ci/inside-deb.sh`,
  `ci/inside-deb-lab.sh` and `scripts/sbom.sh`, refuses with exit 2 an
  environment that sets `CARGO_PROFILE_*`, `RUSTFLAGS` or a
  `CARGO_*RUSTFLAGS` variable, which would build other bytes than those
  tested. `ci/release-refusal.sh` requires a shipped `x0ne-infer`, run with
  no argument, to refuse the built-in TEST measurements as a release build
  (exit 87): step `release-refusal` of `make ci`, each `.deb` in `make deb`,
  the archive in the release workflow. A build with `debug_assertions` fails
  it (measured on 2026-10-25). `docs/ci-local.md` says which pipeline
  reproduces which binary: each embeds the path of its crates registry.
- `x0ne-engine-tlog` runs the transparency log of image references: keys,
  appends in the tlog-tiles layout, the tlog-witness request, cosignatures,
  `.tlog-proof` files, and the relay's check against a pin file (docs/cloud.md,
  "Running the log"). `scripts/pin-transparency-log.sh` pins a log after the
  archive key. `scripts/pin-archive-key.sh` adds an Unreleased entry instead
  of rewriting a released one, and refuses a REHEARSAL key unless
  `ALLOW_REHEARSAL_KEY=1`.
- `x0ne-infer/scripts/gcp-cvm.sh`: `run` and `console` pin the registers
  `X0NE_INFER_EXPECT` names (`--expect-measurement`) and refuse with 2,
  before ssh, when it is empty or malformed. On the Ubuntu VM of this recipe
  a release `x0ne-infer` with nothing pinned exits 87, and `run` used to
  wait two minutes and report 65. `run` now exits with the server's own code
  as soon as it stops. A new phase, `measure`, prints the VM's RTMR1 and
  RTMR2 from a fresh quote: trust on first use, for this development recipe
  only; `x0ne-engine cloud up` keeps pinning a signed release reference.
  `X0NE_INFER_PORT` sets the port; the source tarball is a `mktemp` file
  under `TMPDIR`, removed once copied. `tests/gcp_cvm_recipe.rs` plays every
  phase against a fake `gcloud` (measured on 2026-10-24).
- `make ci` builds the client as docs/cloud.md says (`x0ne-engine`,
  `x0ne-engine-proxy`, `x0ne-engine-cloud`, release) and runs
  `x0ne-engine cloud --help`: step `build-cloud`, after `test-release` in
  its lane; the GitHub workflow runs the same two lines. No CI built it
  before. docs/ci-local.md names the four lanes.
- `x0ne-engine-proxy pins` prints the archive key and the transparency log
  its build pins, or the refusal a session of that build gets, 99 or 100.
  `x0ne-engine cloud up` asks it first and ends with 95 there, before the
  token, the preflight, the quote and "yes"; it used to learn it after
  "yes" (nothing was created either way). The status lines of the README,
  docs/cloud.md and docs/attested-channel.md name the missing transparency
  log as well as the missing key, and `scripts/pin-archive-key.sh` keeps the
  log there when it pins the key.
- THREAT_MODEL.md lists the published attacks on this platform, read on
  2026-10-22: TEE.fail, DDRop, TDXRay, TDXdown and StumbleStepping, Battering
  RAM, WireTap, BadRAM, Heracles, RMPocalypse, StackWarp, EntrySign,
  BadFuse on EPYC Milan, Google's review of TDX Module 1.5, INTEL-SA-01436,
  GCP-2026-053, the GPU-CC analysis of arXiv 2507.02770, the H100's
  unencrypted HBM, Spill the Beans, NVBleed and GPUHammer. Each row gives its
  source, what it breaks, whether this engine is exposed, and a status; none
  is countered here. `tests/known_attacks.rs` holds the table to that form.
  The platforms this version does not run on, GCP's `g4-standard-48` (AMD
  SEV, not SEV-SNP) and Azure, are named with the reason. Three rows cover
  serving features this engine does not have, with the change that would
  expose it: a shared KV or prefix cache (PromptPeek, Gu et al.,
  CVE-2026-105752), speculative decoding (Wei et al.) and a mixture of
  experts in a shared batch (Yona et al.). The binding v3 row cites what an
  IETF draft on attested TLS says of binders inside the handshake and of
  attestation after it, and that no one outside this project has analysed
  binding v3; the availability item names the dependency on NVIDIA's NRAS
  service.
- THREAT_MODEL.md lists sixteen promises this project does not make, with
  the reason for each. `tests/public_claims.rs` reads every Markdown file,
  this section of the changelog, CITATION.cff, the package descriptions and
  the served pages, and fails on a fixed list of phrases for each promise
  (other wordings are not caught) unless the occurrence is listed with the
  reason it is not that promise (`tests/public_claims/allowed.tsv`).
- docs/cloud.md says what "not end-to-end confidential" covers, place by
  place: where the text is in plaintext and who can read it there.
- README: three crates, not two. CUDA inference in the TD ran on the H100 on
  2026-10-17 and 2026-10-18, every receipt kept from those runs naming
  `cuda:0` (the responses are kept outside this repository); the README said
  no run recorded it. `qwen3:32b`, 17 shards, was served on 2026-10-18; the
  README said no sharded model had been. The H100 timings are this engine
  alone, not a comparison with vLLM. AMD SEV-SNP is not built in this
  version, and work on it has started; the README said it was not planned.

- `make ci`: the verdict carries the age of the RustSec database the `deny`
  step judged against (`advisory_db_age_s`). A fetch that fails, in a
  container without network or on a bad day, used to leave the old copy
  without a word and `deny` then said "advisories ok" against it; the run
  still goes on, and now says so with that age. `make ci-refresh`, which
  fetches whatever the age, says a failed fetch too.

- The serving loader (`shards::map_dir`) is held to safetensors by a test: a
  directory of pickle checkpoints (`pytorch_model.bin`, `.pth`, `.pkl`) is
  refused naming `model.safetensors`, not them, and a pickle beside the
  weights is never opened. `pull` refused such files already; this covers a
  directory that did not come through it.

- SECURITY: the TDX verifiers judge every TD attribute bit, not DEBUG alone
  (Intel TDX Module ABI 348551-008, table 3.23). A quote whose TDATTRIBUTES
  carry a debug or profiling bit (HGS_PLUS_PROF, PERF_PROF, PMT_PROF, any of
  15:0), MIGRATABLE, or a reserved bit Intel says must be 0 (26:23, 55:32,
  61:56) is refused, each bit by name: by the TD's own gate before it serves
  (65), by the relay before any key (65), by the server and the SSH client,
  signed or structural. SEPT_VE_DISABLE, the SEC bits Intel marks positive
  (ICSSD, SERVTD_EXT, 22:18, LASS, PKS, bit 31) and the two defined OTHER
  bits (TPA, PERFMON) are accepted. GCE set SEPT_VE_DISABLE alone on
  2026-10-17 and 10-18. A reserved bit a later TDX module defines is refused
  until a release accepts it.
  Before, a host could launch the image profiled or migratable and the TD,
  the relay and every client verified it.

- SECURITY: a signed image reference is refused (89) when its
  `image.version` is below `reference::MIN_IMAGE_VERSION` (0.1), or is not a
  dotted version. Raising the floor in a release retires every older image:
  a reference for an image withdrawn for a flaw, signed before the
  withdrawal, can no longer be handed back to a newer relay. Nothing is
  retired yet; 0.2 and 0.2.3 are the images measured on GCE.

- The files that hold the text of an answer (`stream.rs`, `sse.rs`,
  `local_api.rs`) say so in their header and carry no log macro, `tracing`,
  standard output macro or raw `io::stdout()`/`io::stderr()` handle;
  `tests/no_log_on_answer_paths.rs` holds each file to its header and the
  list of files to those paths.

- SECURITY: `x0ne-infer` exits 73 first thing in `main` when `LD_PRELOAD`,
  `LD_AUDIT` or `GLIBC_TUNABLES` is in its environment, whatever the value,
  naming the variable: through the first two the dynamic loader brings a
  shared object the image never measured into the process; the third sets
  glibc's behaviour from outside the image and is parsed by the loader
  before `main` (CVE-2023-4911). The binary is dynamically linked (glibc,
  libstdc++), and its unit sets no environment file, so this holds a
  launcher's mistake, not an attacker who already writes the image. An
  object already preloaded has run its constructors by then, and the
  tunables are in effect: the refusal stops the process from serving, it
  does not undo either.
  `LD_LIBRARY_PATH` was not refused: cargo sets it for every test (a
  release build refuses it since 2026-11-04, above).

- SECURITY: the relay takes an image reference only with a C2SP
  `tlog-proof@v1` next to it (`<reference>.tlog-proof`), showing the
  reference's own bytes in the transparency log pinned in the build: a
  checkpoint signed by the log, cosigned by a quorum of pinned witnesses
  within a maximum age (else the log's current checkpoint, read from its
  tiles and shown consistent), and consistent with the last checkpoint that
  machine verified. Otherwise it exits 100 before parsing the file; `up`
  ends with 95. A fork keeps both signed heads as evidence. This tree pins no
  log (`transparency::RELEASE` is `None`), so a release relay refuses every
  reference; debug test builds pin a TEST log. The formats are checked
  against Go's (golang.org/x/mod `sumdb`, transparency-dev `formats` and
  `merkle`) on a 600-entry log made by Go, and against a 300-entry log
  written by Tessera v1.0.4, the log library of Rekor v2. A pinned log is
  read at its `https://` location (`HttpTiles`: one GET per file, no
  redirect, 20 s, capped bodies, plain `http://` to the loopback alone);
  a local instance serving the TEST log and the Tessera log refreshes a
  stale proof, and an unreachable, hanging, redirecting, incomplete or
  forked log is refused (`tests/tlog_http.rs`).
- `make repro-compare A=<dir> B=<dir>` (`ci/repro-compare.sh`): the outputs
  of two builds of one commit, from two environments, byte for byte, with a
  JSON report carrying each environment (`ci/build-env.sh`). Exit 1 on a
  difference or a file in one tree only, 2 when there is nothing to compare
  (empty tree, the same tree twice, a symbolic link). Measured on
  2026-10-22 between the host and a VM on it with another kernel, OS
  release, user, locale and time zone, with the build images loaded from the
  host and again with them rebuilt in the VM: the three packages of one
  commit and the x0ne-cvm 0.2.3 image were identical; same machine and
  operator, so not yet an independent rebuild.
- SECURITY: the TCB policy options are honoured or refused, never taken and
  ignored. `x0ne-engine cloud quote`, `down`, `sweep` and `image import`,
  which start no relay, refuse `--min-tcb-evaluation` and
  `--max-collateral-age` (exit 2); they used to accept them without effect.
  The relay's `config` event names the policy its verdicts will apply
  (`tcb_policy`), and `up` and `run` refuse a relay that names none, or one
  looser than the options or the build's floor (95, nothing created). Tested
  on the relay binary (its config with and without the flags) and on a fake
  relay that takes the flags and keeps the floor.
- SECURITY: a TDX module version Intel's tcbInfo publishes no identity for,
  or whose ISVSVN is below every level Intel publishes for it, is a verdict
  against the platform (`TcbRejected::ModuleUnknown`,
  `ModuleBelowEveryLevel`), as a platform TCB below every level already was.
  While serving it loses the standing at the next hourly check; it used to
  read as unusable collateral, which leaves the standing to age out over 24
  hours. At start-up it was and stays a refusal (76). The live check is also
  given the instant the standing is dated by, read before the fetch: a slow
  fetch no longer dates a renewal later than the time its collateral was
  judged against.
- SECURITY: the live verdict is re-read while serving (`tcb_standing`): every
  hour the TD, in cloud mode and outside it, fetches the tcbInfo, the QE
  identity, the PCK CRL and the root CA CRL again and judges them against
  the quote it was admitted on, under the same `TcbPolicy`. A verdict against
  the platform (a status Intel does not accept, no matching level, a revoked
  PCK or PCK CA, an evaluation below the floor, a QE identity that does not
  match) loses the platform's standing for good: the next question is a 503
  naming it, before the model runs, until a restart re-attests. Collateral
  not fetched or not readable leaves the last verdict standing for at most
  the policy's collateral age limit (24 hours), then the answers stop too.
  Until now the collateral was fetched once at start-up and
  `tcb::LIVE_LIMITATIONS` said so. Tested on a fake PCS that rates the
  platform OutOfDate after boot (TD and relay: 503, the model does not run,
  a later UpToDate restores nothing) and on one that stops answering (the
  answers go on until the age limit).
- SECURITY: Intel's root CA CRL is read (`tcb::verify_root_crl`), by the TD
  before it serves and by the relay at each verdict, from the root
  certificate's own CRL distribution point
  (`certificates.trustedservices.intel.com/IntelSGXRootCA.der`;
  `x0ne-infer --root-crl-url` overrides it). The PCK CA that issued the TD's
  PCK, or Intel's TCB signing certificate, listed on it is a revocation (65);
  a root CRL missing, outside its window, not signed by the pinned root or
  carrying a critical extension is unusable (76). Intel issues it for a year
  (number 1, 2026-02-26 to 2027-02-26, empty, read 2026-10-23), so it is
  judged by its own window and not by the 24-hour collateral age limit. The
  verdict's CPU line and the limitations say so. Tested on synthetic CRLs
  (PCK CA listed, TCB signer listed, a leaf's serial listed, expired, signed
  by the PCK CA, missing: TD and relay, no key sent) and on Intel's CRL of
  2026-10-23 with the GCE quote of 2026-10-18.
- SECURITY: a TCB floor under Intel's verdict (`tcb::TcbPolicy`), in the TD
  before it serves and in the relay at each verdict. The tcbInfo and the QE
  identity are fetched with `update=early` and refused below TCB evaluation
  22, Intel's TCB recovery of 2026-08-11 (INTEL-SA-01436, TDX module 1.5.28
  and earlier); the tcbInfo, the QE identity and the PCK CRL are refused when
  issued more than 24 hours before the verdict, even inside their 30-day
  window (exit 76). Until now PCS's default `standard` answer was taken: on
  2026-10-22 it was evaluation 20 and rated TDX module ISVSVN 11 UpToDate,
  which evaluation 22 rates OutOfDate. A stricter floor can be built with
  `TcbPolicy::tightened`; a looser one cannot. A host Intel does not rate
  UpToDate or SWHardeningNeeded under evaluation 22 is now refused. Tested on
  synthetic collateral and on Intel's collateral of 2026-10-22 with the GCE
  quote of 2026-10-18, which passes. The relay takes
  `--min-tcb-evaluation N` and `--max-collateral-age secs`, stricter than the
  build or a usage error (2), and its verdict's CPU line names the floor it
  applied; `x0ne-engine cloud up` and `run` take the same options and pass
  them to it, a looser value refused before any quote (exit 2).
- SECURITY: an answer's length is padded for the VM's host, without an
  option to turn it off. A whole answer's JSON body ends in spaces up to a
  Padme size (PURBs, Nikitin et al., PETS 2019) of at least 16 KiB, and the
  answer leaves the TD once a Padme number of 100 ms ticks has passed since
  its turn began, at least 10, so never within one second. An error or a
  refusal of a whole answer leaves the same way, `{"error": ...}` padded,
  its head of one length whatever the status (an `x0ne-pad` header): it
  used to leave as a short text body, a refusal at once. A chat, whole or
  streamed, runs in the model's turn, which ends at that same size of ticks
  after it began, counted when the generation ends, and holds the answer's
  queue place until then: an answer asked while another runs starts then,
  and a full queue's 503s stop then, not when the model frees up, which
  told how long the other ran to the token. A stream's number of frames,
  the last included, is rounded up the same way, at least 10 frames, the
  last frame waiting for it after the text has passed on; an error frame
  too. Both stop at 8704 ticks (870.4 s), the last size within the 15
  minutes a relay waits: a stream whose text is not out by then ends in
  error there, and a whole answer that ran longer leaves when done, its
  time not rounded. Padme adds at most 11.6 % (15 on 129) and leaves 300
  sizes up to 2^20. Measured on the loopback between the relay and a
  simulated TD (2026-10-23): whole answers of 1 byte to 8 KiB of content,
  seven record traces before, leave as one trace of 16581 bytes at 1.00 s
  (2026-10-24); streams of 1 to 16 steps, three traces before, as one of nine
  frames and the final pair at 0.92 s; a 9 s stream ends 0.5 s later and
  costs 4 % more bytes. The host still reads the length to within a size,
  the frame cadence, when a stream starts and ends, and the request's size.
  Whole answers used to leave as soon as done, at their own size; streams
  ended on their last token. With `--max-queue 0`, a request sent as a
  stream ends may get a 503 until that stream's turn ends.
- SECURITY: the overall NRAS token must name each per-GPU token it covers.
  Its `submods` claim holds, for each GPU, `["DIGEST", ["SHA-256", hex]]`
  over that GPU's token as sent; the verifier now requires one entry per
  per-GPU token, under its name, with that digest, and no other entry.
  NVIDIA's NVAT checks the digests and their number but not the label; the
  verifier reads NRAS's `SHA-256` and NVAT's own spelling `SHA256` and
  refuses any other. A per-GPU token from another NRAS answer for the same
  nonce and GPU, an overall token without `submods`, naming a GPU more or
  another GPU, or holding anything but a SHA-256 digest, is refused (75).
  The shape and the digest were read from the production token of the
  2026-10-17 run. Applies to the boot document and to each connection.

- SECURITY: a local GPU certificate serial named twice is refused (75): the
  sorted comparison of token `ueid`s with local serials let one GPU's token
  stand for two GPUs when a serial repeated. The NRAS issuer is now compared
  as the origin `https://nras.attestation.nvidia.com`, or a path under it;
  a prefix test let a longer host such as
  `https://nras.attestation.nvidia.com.example.net` pass.

- SECURITY: with the GPU product (`--attest cpu+gpu`), every TLS connection
  to the TD now carries GPU evidence of its own. `GET
  /v2/attestation/session` (schema `x0ne-infer/session-attestation/v4`)
  returns, after the connection's quote, NRAS tokens for the nonce
  `SHA-256("x0ne-infer/v4/gpu" ‖ exporter ‖ SHA-512(signed TD report))`:
  the connection's TLS exporter and the header and body of that same quote.
  The relay verifies them like the document's (pinned NVIDIA intermediates,
  per-GPU policy, `ueid`) and requires the GPUs the document attested; a
  connection whose tokens answer another nonce (an earlier connection, a TD
  on another machine, the boot document) or name another GPU is refused
  (75) and stops the session, and a session quote without its GPU section
  is refused. Until now the GPU tokens were taken once per boot, for a
  nonce anyone could compute from the document. The binding is made by the
  TD's software: Hopper's GPU report names no VM, and the link falls with
  the CPU TEE (TEE.fail). The session schema v3 is refused, so a relay and
  a TD of different versions do not agree. Each new connection now costs one
  NVML collection and one NRAS round trip (not measured on hardware).

- `x0ne-sdk`, a prototype, not served by the TD binary: the caller checks a
  TD's Oblivious HTTP key itself (TDX quote to the pinned Intel root, live TCB
  and PCK CRL from relayed collateral, MRTD and RTMR1..3 pins, `report_data`
  binding the key, this caller's nonce and the receipt key), then seals its
  chat request to that key (RFC 9458 over RFC 9180, Binary HTTP padded to
  4 KiB..4 MiB buckets) and checks the answer's receipt under the key the quote
  bound. The verifier is x0ne-infer's own `pcs.rs` and `tcb.rs`. Builds for
  `wasm32-unknown-unknown` with a C ABI and no imports; `make sdk-wasm` runs it
  in Node against the simulated TD. `docs/sdk-e2e.md` says what it does not
  do; the product claim does not move.
- SECURITY: an image reference may pin RTMR0 to Google's endorsed firmware,
  `rtmr0.policy: "gce-endorsed"`. The client then fetches the firmware
  binary Google's signed endorsement names from Google's bucket, checks its
  SHA-384 is the signed one, computes the MRTD the binary measures (TDVF
  metadata, as google/gce-tcb-verifier does) and requires the quote's; every
  RTMR0 event must then be the reference's for the machine type, the
  firmware's configuration volume being the binary's (`"firmware:cfv"`).
  A binary other than the signed one, an MRTD the binary does not measure,
  a changed ACPI table, TD HOB, Secure Boot or boot variable, an event
  missing, added or moved refuse (65); a bucket that cannot be read refuses
  with 76. Google's binary `53c558fe…` measures MRTD `c1ee9c16…`, the
  quoted MRTD of the GCE boots of 2026-10-17 and -18, and its configuration
  volume is the RTMR0 event of their logs (measured 2026-10-22). `report`
  is unchanged and is still the policy of every published reference; a
  client without this change refuses a `gce-endorsed` reference (unknown
  field `events_sha384`). `gce_firmware` (example)
  prints a binary's MRTD and the RTMR0 list of a boot. README "RTMR0 and
  Google's firmware" says what this removes from the trust in Google and
  what remains.
- SECURITY: an image reference may set `mrtd.min_svn`, a floor on the TDX
  SVN of the Google endorsement that names the quoted MRTD: an older
  firmware Google once endorsed is refused (65), under `report` and
  `gce-endorsed` alike. Absent or 0, no floor; published references carry
  none.
- Cloud mode serves a model owner's sealed weights, the key staying on the
  owner's machine (`docs/model-key.md`). `x0ne-engine-proxy --model-key
  <file> --key-policy <file>`: after the verdict and the owner's policy
  (images by MRTD/RTMR1/RTMR2, TCB, GPU, model, sealed file, expiry; exit
  102 if not), the relay draws a nonce, the TD quotes an X25519 key born at
  its boot over that nonce and the session binding (`GET
  /v2/model-key/recipient`), and the key goes to it wrapped by HPKE (RFC
  9180, X25519, HKDF-SHA256, AES-256-GCM) with a grant naming the sealed
  file, the served digest and the expiry (`POST /v2/model-key`). The TD
  (`--sealed-weights`) checks the sealed file against the config before any
  evidence, opens only a nonce it issued, once, drops its key pair after one
  key or three failures, re-takes the release decision, unseals, enforces the
  served digest and stops serving once its own clock passes the grant's
  expiry. That clock is the host's: a TD clock held back serves as much
  longer, within 24 h of running time from the release, and the relay never
  compares it with its own (a test keeps this behaviour). Refused, each by
  a test against the simulated TD: an image the policy does not name, an
  expired policy, a replayed recipient quote, a recipient key the quote does
  not name, a wrapped key posted to another TD, a release nonce posted twice,
  a tampered sealed file, other weights under the same key. `x0ne-engine
  cloud up|run --model-key <file> --key-policy <file>` passes both to the
  relay (both or neither, files only, a key typed in their place refused
  unprinted); the policy's `sealed_source` (URL and size, https or loopback
  http, no userinfo) travels in the session config, so in the quote, and the
  TD fetches the file after its attestation with the catalog's transfer code,
  pinned by the config's size and SHA-256, before its TLS server opens (77
  for other bytes or a file that is not an envelope, nothing kept; 96 a source
  it may not use, or a failed transfer, whose partial of the source's bytes
  stays for the next attempt). An https source on a literal loopback,
  link-local, private or unspecified address, or on `localhost` or a cloud
  metadata name, is refused; names are not resolved. No message prints a
  source URL's userinfo. `up` refuses a
  relay config that names sealed weights without a model key, none with one,
  or no source, before anything is created. `x0ne-infer seal` prints the
  sealed size the policy needs. Never run on TDX hardware; the sealed file is
  hosted by the owner, not uploaded by `x0ne-engine`; TDX only.
- AMD SEV-SNP verifier (`x0ne-infer::snp`, `verify::cryptographic_snp`,
  `remote_verify::verify_snp_evidence`, `SnpSessionCheck`), not wired to any
  host, guest image or relay. A report (versions 2, 3 and 5) must chain
  through its VCEK to an ARK and ASK pinned in the code (Milan, Genoa,
  Turin, as AMD's KDS serves them), describe that VCEK's chip and TCB, come
  from VMPL 0 with no DEBUG and no migration agent, and carry every TCB at
  or above a minimum (default per CPU line from AMD-SB-3020 and
  AMD-SB-3027; the BadRAM alias check on Milan and Genoa). Milan is refused
  unless allowed. `REPORT_DATA` is the TDX binding; the client requires a
  pinned launch digest (89). No new dependency. The TDX path is unchanged.
  README and THREAT_MODEL say what it proves and what it does not.
- `x0ne-engine cloud up` and `run` with a token they cannot renew (`--auth
  paste`, `stdin`) end the session 5 min before the token expires, so that
  the VM is deleted while the token is still valid. `--max-run` is cut to
  fit: a token with 59 min left and the default 60 min give 54 min, and the
  quote, `--max-cost` and the VM's `maxRunDuration` count 54. The cut, the
  token's expiry and the reason are printed before the quote, at `[5/5]`, at
  the end and in `status`; the token is not. A token that leaves less than the
  boot timeout plus 5 min of serving is refused before anything else is read
  (exit 91), and again after "yes" when the wait at the prompt used it up;
  so is a token whose expiry Google's tokeninfo does not give. They used to
  start, with a warning that Ctrl-C could not delete the VM once the token
  had expired. 5 min is about four times a `down` measured once: about 69 s
  to delete the VM and 4 s its firewall rule. `--auth gcloud`, which renews
  the token, is not cut. With any token, `serving` is no longer waited for
  past the end of the session (exit 93).
- SECURITY: `x0ne-infer` overwrites the value of `X0NE_INFER_KEY_HEX` with
  zeros in the environment block set at exec, before it removes the
  variable: `/proc/<pid>/environ`, which the same user and root can read,
  no longer shows the key for the life of the process. It still does from
  exec to the first step of `main`, and the launcher's environment keeps it.
- OpenAI API: `temperature` (0 to 2) and `top_p` (0 to 1) sample, through
  candle's `LogitsProcessor`, one per request; `seed` repeats a draw, and the
  server draws one from `/dev/urandom` when absent. 0 or no `temperature` stays
  greedy, and there a `top_p` below 1 or a `seed` is a 400 that names it rather
  than a parameter ignored. `max_completion_tokens` is read as `max_tokens` (two
  different values: 400). `stop` (a string or up to 4, each at most 256 bytes)
  ends the generation loop and the answer before the first one met,
  `finish_reason: "stop"`; the receipt commits to the text returned. All four
  were a 400 before. Tool calling (`tools`,
  `tool_choice`, `functions`, `function_call`, `parallel_tool_calls`, a `tool`
  message, `tool_calls`) is a 400 saying it is not supported yet; the relay
  applies the server's rule instead of its own. Both servers, and the relay,
  which forwards the fields.
- Streaming: `stream: true` on `x0ne-infer` (TD and plaintext) answers
  OpenAI's `chat.completion.chunk` events as the model generates: the role,
  the content deltas, a final chunk with `finish_reason`, the usage chunk
  (`choices: []`) when `stream_options.include_usage` is true, then
  `data: [DONE]`. The `x0ne` block (receipt, opening, signed payload) rides
  the final chunk only; the receipt commits to the concatenated deltas, and
  nothing is signed for a stream its client left. A stop sequence split
  across tokens is never streamed, and no delta splits a UTF-8 character.
  Refusals before the first token (400, 413, 503) stay HTTP statuses. The
  text is shown before the receipt: a failure after the 200 is one error
  event with a fixed message (`"code": "generation_failed"`), then the
  body's end, as after a success; never a final chunk nor `[DONE]`.
  Backends stream through `LoadedModel::generate_streaming` and a
  `TokenSink`.
- Stream padding, not optional: one frame every 100 ms, all of one size per
  stream (512 to 768 bytes, set by the echoed `model`) whatever the text, an
  SSE comment when there is none, and a last frame of 16 KiB, success or
  error. Text is padded in `obfuscation`; `stream_options.include_obfuscation`
  is a 400 that names it. With `stream: true`, `model` is at most 256 bytes
  once JSON-escaped (400). No event from the generation for 90 s ends the
  stream in error.
- Tests: the TLS records a simulated TD writes to the relay, read off the
  TCP link between them: past the HTTP header, a record per frame, all of
  one size, for short, mixed and burst streams; the last frame a full
  record and a short one with an `x0ne` block of 7322 bytes; an error the
  same two records, the chunked end included, then its connection still
  open 1.5 s after the body's end and taken by the next stream, no new one
  opened. Two `#[ignore]`
  measurements: `measure_stream_records_on_the_wire` (sizes, and the gaps
  before frames with and without text, the TD on its own runtime) and
  `measure_stop_under_load_of_concurrent_and_slow_streams`.
- A stream holds its `--max-queue` place until its generation ends, not
  until its header; a client that hangs up stops the generation at its next
  step and frees the place. A body still written to a slow reader holds one
  of `--max-queue` + 2 open stream places until it ends: one stream more is a
  503 with `Retry-After` before anything runs; whole answers are not counted.
- `stream` takes a bool or `null`, `stream_options` an object or `null`;
  `stream_options` without `stream: true` is a 400. `x0ne-infer-lab` refuses
  `stream: true` with a 400 rather than ignoring it. `/v2/ready` says
  `"streaming": true`.
- Relay streaming: the local endpoint streams `stream: true` from a TD whose
  `/v2/ready` says `"streaming": true`, and emulates it from a whole answer
  otherwise (`x0ne.stream_emulated: true`). Every TD event is checked against
  a whitelist and rebuilt, never copied; the padding stops at the relay. The
  text passes on as it comes. The final chunk, the usage chunk (always asked
  of the TD, passed on when the client asked) and `[DONE]` go out together,
  once the receipt verifies over the streamed text, the usage equals the
  receipt's and the TD's body ended clean after `[DONE]`. `x0ne`, with
  `receipt_verified` and `session`, rides the final chunk only; with
  `include_usage` the other chunks carry `usage: null`. Otherwise one error
  event (`"type": "x0ne_engine"`, a fixed message) takes their place, then
  the body ends as a whole one does, and its `code` says what follows:
  `receipt_verification_failed` and `stream_protocol` stop the session
  (exit 65, after the event); `stream_cut` loses it (a cut body, 30 s
  without a byte read, 120 s without an event read, 15 min in all, a local
  client that leaves a stream without a byte read for over 1 s);
  `generation_failed` keeps it, and the TD's connection too when the TD's
  body ends within 1 s of the event: the relay reads that end, waiting 1 s
  at most, before it passes the event on; `session_stopped` ends
  every stream of a session that stops, and new requests are a 503; a whole
  answer that comes back after the stop is a 503 `session_stopped` too. A
  local client that reads nothing for 30 s is dropped and its TD connection
  closed; stopping the endpoint waits at most 5 s for open connections.
- Changed for clients of the relay's streams: `x0ne` now rides the final
  chunk only, where the emulation, the relay's only stream before, repeats
  it on every chunk (it still does for a TD without streaming). Read it from
  the chunk with `finish_reason`.
- The relay's 502 for a whole answer whose receipt fails carries
  `"code": "receipt_verification_failed"`, the code of the stream's event;
  a 200 that is not `text/event-stream` to a stream, `"stream_protocol"`.
- `x0ne-engine-proxy chat`, and so `x0ne-engine cloud run` and `chat`,
  stream when standard output is a terminal: the text shows as it comes,
  the receipt line only after the final chunk, `[DONE]` and the clean end
  of the body. Redirected, it asks for whole answers, verified before they
  are written. A failed receipt, a stream cut before its receipt or a
  session stopping prints a red line under the text saying it is not
  verified, and ends the chat. A cut exits 1 (`run`: 96). A failed receipt,
  a broken protocol or a stopping session exits 3, and `run` then waits up
  to 10 s for the relay's refusal and exits with its code, 65 (96 without
  it). A whole answer's coded 502 ends it too.
- `finish_reason` is `"length"` when the context window cut the answer below
  the `max_tokens` asked for; it read `"stop"`, as if the model had ended it.
  Whole and streamed answers, and `x0ne-infer-lab`.
- Relay: when the TD's quote limits turn a new connection away (429, or 503
  `busy`), the local request is a 503 with `Retry-After: 1` and the session
  goes on. It was a 502 that lost the session: eight local requests at once
  against a TD at `--max-queue 2` lost it (2026-10-22). The relay opens one
  connection per local request in flight when none is free, so a burst can
  meet the quote gate, whose entry below says the honest relay never does. A
  failed quote (503 `session quote unavailable`) still loses the session.
- Relay: the TD's 503 for a full queue keeps the TD's `Retry-After` (5 s),
  capped at 120 s. The relay dropped it, and openai-python fell back to its
  own backoff, about 0.5 s then 1 s. Every other 5xx of the relay (a failed
  generation, a lost or stopped session) carries `x-should-retry: false`,
  which openai-python reads before the status: it sent a whole answer whose
  generation failed twice more, three generations for one request
  (openai-python 3.36.0 and 1.109.1, 2026-10-22). A 503 with `Retry-After`
  carries no such header and is retried after the delay it gives.
- CI: the defaults name no host's layout. `CI_CACHE` defaults to
  `${XDG_CACHE_HOME:-~/.cache}/x0ne-engine-ci`, with the staging, the sccache
  disk cache and the build lock under it; containers join a cgroup only when
  `CI_CGROUP_PARENT` names one; `make deb` needs `X0NE_OS`, and the
  post-merge hook covers `main` unless `x0ne.ci.branches` says otherwise. A
  host's own settings go in `~/.config/x0ne-engine/ci.env` (`KEY=value`,
  read, not run; the environment wins). No desktop notification any more:
  `CI_NOTIFY_CMD`, when set, is run after each verdict.
- CI: `make ci` waits for the build lock alone, at most `CI_WAIT_MAX` s,
  and records the wait (`lock_wait_s`); a `cargo` that holds no lock no
  longer stalls it. The lock is `CI_LOCK` (default `$CI_CACHE/build.lock`),
  the same from any clone.
- CI: every run copies its tree into `$CI_CACHE/v2/src`, rewriting only the
  files whose content differs, and builds it in `$CI_CACHE/v2/target`. An
  unchanged tree rebuilds nothing, and two worktrees at one commit get one
  verdict: on the shared target, a tree older than the last build ran that
  build's binaries. `CI_WORKTREE=1` writes `<sha>+dirty.json` when
  the tree under test is not the commit's, changes git status does not show
  included (a file marked `--skip-worktree`). Dev and test profiles build with `line-tables-only`
  debug info (`CI_DEBUGINFO`); `v2/target` is emptied past
  `CI_TARGET_MAX_GB` (40).
- SECURITY: the TD's PCK certificate is checked against Intel's PCK CRL
  (PCS v4 `pckcrl`, for the CA that issued the PCK), fetched live by the TD
  at start-up and by the client at each verdict. A listed PCK refuses (65);
  a CRL that is missing, outside its thisUpdate to nextUpdate window, not
  signed by the PCK's issuer or carrying a critical extension refuses (76);
  `x0ne-infer` outside cloud mode exits 76 for both, with the rest of the
  live collateral. Intel's root CA CRL is not read. Intel's platform CRL of 2026-10-18 is
  vendored (`x0ne-infer/tests/pcs_live/`): it lists 57 PCKs, not the one of
  that day's run.
- SECURITY: no command line prints an argument it does not know: a `--flag`
  is named without its `=value`, anything else is not printed. Nor a value it
  refuses: the flag is named, the value is not (a model name, `--auth`,
  `--project`, `--listen`, `--max-tokens`, `--device`...), and a path that
  holds a run of 32 or more hex digits is not printed either. `x0ne-infer`
  refuses with exit 2, before anything is collected, an unknown argument, a
  `--flag=value`, a flag without its value, a malformed or zero
  `--max-completion-tokens` or `--max-handshakes`, a malformed `--max-queue`
  and an invalid `--allow-host`. It used to print "ignored", or nothing, and
  start: `--expected-model-digest=<hex>` served with no pin on the weights,
  `--max-queue 8o` with the default. `--help` is read anywhere on its line.
  `chat --key-file` followed by 64 hex digits, the key itself, is refused
  (relay and `x0ne-engine cloud chat`).
- SECURITY: each per-GPU NRAS token must report its three certificate
  chains (attestation report, driver RIM, VBIOS RIM) `valid` with OCSP
  `good`, OCSP nonce matched and response valid, and no
  `x-nvidia-mismatch-opaque-data-records`, as NVAT's default policy; and,
  beyond it, `secboot` true and `dbgstat` "disabled". The per-GPU
  tokens of the 2026-10-17 and 2026-10-18 runs pass
  (`x0ne-infer/tests/nras_live/`).
- The `ueid` limitation names the cloud runs where production NRAS tokens
  matched it (2026-10-14, 2026-10-17, 2026-10-18) instead of saying it was
  never exercised.
- Documentation brought back to the code in README.md, SECURITY.md and
  THREAT_MODEL.md. The TD's key lease runs 24 h from the release at most; the
  TD does not read the session expiry, the relay does. Beside `memfd_secret`,
  the released key has ordinary copies (the TD's parsed array, the loader's
  `ReleasedKey`), and the receipt signing key is ordinary heap. PCS statuses
  `UpToDate` and `SWHardeningNeeded` are both accepted. Google's pinned root
  is not checked against its validity window. Of revocations, Intel's root
  CA CRL and the NRAS chain's own OCSP are not read. CI runs `cargo deny`,
  not `cargo audit`, and `deny.toml` ignores `paste`. Cloud mode's attested
  channel and key release, and the release workflow, are no longer called
  unbuilt; the 0.2.0-beta.7 measurements file names its registers.
  `tests/doc_claims_follow_code.rs` holds each statement to the code fact.
- `pull` downloads up to 4 files at once, and a file over 64 MiB in 64 MiB
  byte ranges at once, at most 16 requests in flight in all. Ranges
  written are listed in `partial/<sha256>.ranges`: an interrupted pull asks
  only for the missing ones, and a range listed past the end of the `.part`
  counts as missing. A range is retried up to 3 times; after a
  redirect, ranges go to the redirect target, one resolve per file. A source
  that ignores ranges is read in one stream. Each file is still hashed whole
  against the catalog before anything is installed. `x0ne-engine pull`
  progress is kept per file.
- 2026-10-21, decode speed: Qwen3 runs on `x0ne-infer/src/qwen3.rs`, a
  modified copy of candle-transformers 0.11.0's model (NOTICE). The KV cache
  is reserved once per request and written in place, grouped-query attention
  reads each KV head once, and the server fuses q/k/v and gate/up into one
  matmul each (the lab keeps them split). A prompt is prefilled in passes
  sized for its own length instead of the full context: never below the chunk
  at the full context, at most the larger of that chunk and 1024 tokens.
  `/v2/ready` adds `prefill_chunks`, the chunk by prompt length, fixed at
  load; the relay passes `prefill_chunk` and `prefill_chunks` in its
  `serving` event and in `/x0ne/v1/status`. Greedy answers are not
  byte-identical to the previous version's: on prompts of 3 827, 7 926 and
  16 120 tokens they part after 39 to 55 words.

- x0ne-sdk: `attest::verify_quote` runs the steps of `verify_key` from
  `quote` to `revocation` for a `report_data` the caller computed;
  `verify_key` calls it with the key binding, its refusals unchanged.
  `x0ne-sdk/tests/real_quote_2026_10_18.rs` replays the whole quote of the
  GCE run of 2026-10-18 through it, offline: accepted (UpToDate, the MRTD,
  RTMR0 and PCK digest the relay's verdict named) on Intel's collateral of
  2026-10-22 for FMSPC `00806f050000` at a clock of that day; refused at
  `binding` by `verify_key`, its `report_data` being the relay's binding
  v2; refused at the step named for one bit of any signature or of the
  signed TD report, other pins, another `report_data`, collateral of
  another platform, stale, of evaluation 20 or cut, a stale or foreign PCK
  CRL, and a cut PCK chain. The TCB info and QE identity of the run's day
  were not kept: the collateral is four days later. `Collateral`'s doc
  names the GETs with `update=early`, as the relay makes them: PCS's
  default answers evaluation 20, which the floor of 22 refuses.

- x0ne-sdk verifies a TD as the relay does, with the relay's files compiled
  in by `#[path]` (`gce_endorsement.rs`, `ccel.rs`, `reference.rs`,
  `tdvf.rs`, as `pcs.rs` and `tcb.rs` already were). The MRTD must be in
  Google's launch endorsement, signed under the Google root the reference
  names, for the caller's machine type, instead of equal to a pin; RTMR0..3
  are the replay of the TD's event log (`KeyDocument.ccel_b64`), every
  RTMR1 and RTMR2 event on the image reference's closed lists, RTMR3 zero,
  instead of RTMR1..3 equal to pins; Intel's root CA CRL is read after the
  PCK CRL (`revocation`); the platform (`tcb`, `revocation`) is judged
  before the image (`measurement`), the relay's order. `Policy` takes an
  image reference (`attest::parse_reference`) and a machine type, pins
  optional on top; `Collateral` carries the root CA CRL and Google's
  endorsement, `attest::ROOT_CA_CRL_URL` and `attest::gce_endorsement_url`
  name the relay's GETs. Breaking for the prototype's callers: the key
  document, the collateral and the C ABI's policy have new required fields.
  The quote of 2026-10-18 now passes with its own log, Google's endorsement
  of its MRTD and the run's reference, without pins (image x0ne-cvm 0.2.3);
  each alteration of the endorsement, of the log, of the reference's root
  or machine type, and of the root CA CRL is refused at its step, and the
  bench refuses a root CA CRL withheld, expired, foreign, altered or listing
  the PCK CA or the TCB signer. The relay's own verification is unchanged:
  those files gain `#[cfg(not(x0ne_sdk))]` on their tests, their fetches and
  the reference's loading from disk, nothing else. Left as it was: the
  caller's clock, the NRAS token, the reference's archive-key signature and
  transparency log entry, RTMR0 pinned to the endorsed firmware (refused).

On 2026-10-18, when this version's number was set:

- Version 0.2.0-beta.7 (`Cargo.toml`, `Cargo.lock`, `CITATION.cff`,
  `ci/deb`). Builds with and without the NRAS P-384 intermediate (below) both named themselves
  0.2.0-beta.6; this version tells them apart.
- `measurements/0.2.0-beta.7.json` names the RTMR1 and RTMR2 of the
  2026-10-18 GCE run of x0ne-cvm 0.2.3 (disk tar `9537364d…`, verdict
  VERIFIED), equal to what the x0ne OS predictor (`cvm/predict_measurements.py`)
  computed for that image. That run's reference was signed by the
  TEST archive key: the file says `"provenance": {"reference_signed_by":
  "test-archive-key", "release": false}` and stays TEST until the signing
  ceremony: a release build of this version refuses it (exit 87) unless
  pinned otherwise. Its log, TD report, reference, quote and signature are kept as
  fixtures (`x0ne-infer/tests/{ccel,reference,run_fixtures}`).
- SECURITY: `make run-fixtures` writes nothing unless the relay's report
  names this reference's SHA-256 (another one was a note), the quote
  verifies again to the pinned Intel root, and the release archive key
  signed the reference; `RUN_TEST_KEY=1` (`--test-key`) accepts a TEST
  signature, says so in the first note and writes `"release": false`.
- SECURITY: a measurements file may carry `provenance`; a release build
  refuses one with `"release": false` (exit 87), a debug or lab build warns
  and goes on.
- SECURITY: NRAS tokens are verified under two pinned intermediates, 004
  (RSA) and 005 (P-384), on the client and in the TD. In NVIDIA's
  production JWKS as read on 2026-10-18 at 10:03:39 UTC, 72 of the 200
  keys chained to 005 (104 of the 200 had already expired); the oldest 005
  leaf in that one snapshot dates from 2026-10-15 23:21 UTC, and 005 is
  valid from 2026-10-08. When NRAS began signing under 005 was not
  observed. The leaf was verified as RSA only, so a token signed under 005
  was refused. The pin fixes the algorithm of the leaf's signature:
  sha256WithRSAEncryption under 004, ecdsa-with-SHA384 under 005; a leaf
  declaring another, or a pin of the wrong key type, is refused. The JWT
  stays ES384 only, under a P-384 leaf key. Their root,
  CA 001, is not pinned, so nothing checks its signature on an
  intermediate: each is pinned by the SHA-256 of its whole certificate as
  well as of its SPKI, and `x5c[1]` must be that certificate byte for byte.
  A certificate made around a pinned key no longer brings its own validity
  window; one NVIDIA re-issues is refused until it is pinned. `Anchors`
  and `TdAnchors` carry `nras_intermediate_pins`;
  `NVIDIA_NRAS_GPU_INTERMEDIATE_SPKI_SHA256_HEX` is now
  `NVIDIA_NRAS_GPU_INTERMEDIATE_004_SPKI_SHA256_HEX`.

## 0.2.0-beta.6 — 2026-10-18

- Version 0.2.0-beta.6 (`Cargo.toml`, `Cargo.lock`, `CITATION.cff`,
  `ci/deb`). `measurements/0.2.0-beta.6.json` names no register: no boot
  of its image has been measured yet.
- Sharded models are served as published. `pull` keeps the shards
  and their index; nothing is merged. The engine maps each weight file
  read-only (one open, one `mmap`) and reads each tensor from the shard the
  index names (`shards::ShardedSlices`), instead of copying the weights into
  the process. An aligned tensor goes from the mapping to the GPU (candle
  0.11 `convert_slice`); in CC mode that copy still crosses memory shared
  with the host, encrypted and signed, through the driver's bounce buffers
  (NVIDIA, 2023, WP-11459-001, p. 27; docs/pull-run.md). In a TD, `qwen3:32b` (61.0 GiB, 17 shards) takes 61.0 GiB
  of the model store, a tmpfs of 117 GiB; with the merge, shards and
  merged file took 122 GiB (computed, not measured on a TD). A copy pulled
  earlier, shards merged, is replaced on the next `pull`.
- SECURITY: the shard set is checked, not trusted. The weight files come
  from the catalog entry, never from a directory listing. The index and
  the shard headers must agree exactly; a tensor in two shards, a shard the
  index does not name, one it names and is missing, and a key twice among
  a header's tensors, in one tensor's fields or in `weight_map` are refused
  (72 in cloud mode). Keys inside `__metadata__` and the index's
  `metadata` are not checked: no tensor is read through them. candle's
  `MmapedSafetensors::multi` returns the last file's copy of a duplicated
  tensor; it is not used.
- The served-model digest is version 2, one definition for every
  generative model: sha256 of `x0ne-model-digest/2\n` and the `sha256sum`
  lines of `config.json`, `tokenizer.json` and the weights (the file, or the
  index and every shard), sorted by name (`model_digest.rs`, docs/cloud.md).
  Anyone can recompute it with coreutils, it no longer depends on a merge,
  and the catalog parser refuses a pin that does not follow from the
  sha256s it lists. The engine recomputes it on the files it loads. Pins
  changed:

  | model | v1 | v2 |
  |---|---|---|
  | `qwen3:0.6b` | `a21fba48…` | `b430888f…` |
  | `olmo2:1b` | `06fe6839…` | `084ebfa7…` |
  | `qwen3:8b` | `d750e7af…` | `e1776f7c…` |
  | `qwen3:1.7b` | none | `24d5d12d…` |
  | `qwen3:14b` | none | `a59e04e0…` |
  | `qwen3:32b` | none | `745f026c…` |

  `--expected-model-digest` and every record that names a v1 value must
  take the v2 one. Sealed models (`decide`, GLiClass) keep their own pin.
- The lab looks a base's openness up by its served digest (v2), no longer
  by the sha256 of `model.safetensors`, or of the index for shards: two
  shard sets under one index had one key. `/learn/v1/model` gives
  `openness.served_digest`; the decision encoder keeps its file's sha256.
- Cloud mode serves all six catalog models. `qwen3:32b` needs a
  `--max-run` of 45 min or more (boot timeout 39.3 min plus 5).
- `/v2/ready` reports `context_tokens` and `prefill_chunk`: the model's
  context, cut to what fits in the GPU memory left once the weights are
  loaded, and the prefill chunk the TD uses. The
  relay passes the context in its `serving` event, `up` prints it at
  `[5/5]` and `/x0ne/v1/status` carries it as `model.context_tokens`.
- `x0ne-infer --model-dir` and `x0ne-infer-lab` load a directory with
  `model.safetensors` or an index and its shards; both is refused (71).
  `x0ne-infer merge-shards` remains, for tools that take one file.
- `memmap2` 0.9 becomes a direct dependency of `x0ne-infer` (it was in
  the lock through candle).
- `quote`, `up`, `run` and `image import` name the image of the reference
  the relay will take: `--reference`, else the newest installed
  `x0ne-cvm-<version>.json`, no longer `DEFAULT_IMAGE` whatever that
  version. The relay's `config` event names the image and disk of
  the reference it verified; `up` refuses another, or none, with 91 before
  anything is created. A drift used to fail at verification, on a VM
  already paid for. `DEFAULT_IMAGE` stays `x0ne-cvm-0-2-1`, used only when
  no reference is given or installed.
- `make run-fixtures RUN_WRITE=1` no longer replaces a file of the tree
  silently: the measurements of a run, or the fixtures of a `RUN_DATE`
  already taken, are replaced only with `RUN_OVERWRITE=1`; otherwise it
  names them and writes nothing. A measurements file naming no image
  and no register, as `0.2.0-beta.6.json` is today, is filled.

## 0.2.0-beta.5 — 2026-10-17

- SECURITY: `x0ne-engine-proxy` checks the image reference's signature
  before it parses the file: `<reference>.sig`, a detached OpenPGP
  signature by a signing subkey of the x0ne archive key pinned in the build
  (`archive_key.rs`, `openpgp.rs`; Ed25519 v4 only, no OpenPGP library).
  Missing, made by another key or subkey, over other bytes, by an expired or
  revoked subkey: exit 99, before a config exists. Before this,
  docs/cloud.md and `cloud status --limits` said the references were
  signed, and nothing checked it.
- A signature by the primary key alone is refused (99), as the TEST key
  of x0ne-OS `make cvm-sign` signs today: references are signed by a
  signing subkey bound to the primary.
- The archive key is written into the source after the signing ceremony;
  this source tree pins no archive key (`RELEASE` is `None`): its release
  build refuses every reference with 99, before it looks for one, and says
  it was built before the signing ceremony. Debug test builds pin a TEST
  key through the `test-archive-key` feature; without debug assertions the
  feature pins nothing, so `cargo test --release -p x0ne-infer` and the
  release `cloud_probe` keep `RELEASE`, and `make ci` runs the release
  tests (`test-release`). A verdict on a reference signed by a key whose user ID
  says TEST says "TEST key, not for release"; it names the subkey by its
  long key ID.
- `sha1` 0.11.0 (RustCrypto) joins the lock for v4 fingerprints, and the
  `x0ne-infer` entry lists `x0ne-infer` itself (the dev-dependency that
  enables `test-archive-key` in its tests); no other entry changed.
- The image reference is schema `x0ne-engine/reference/v2` (x0ne-OS
  `cvm/reference.py`). RTMR1 must hold the image disk's GPT event exactly
  once, equal to `rtmr1.gpt_event_sha384` and before the first boot
  application (EDK2 `DxeTpm2MeasureBootHandler`; the GCE logs of 2026-10-14
  and 2026-10-17 put it there), then the boot applications of
  `rtmr1.boot_applications_sha384` (UKI, then kernel PE) once each, in that
  order. The v1 policy accepted any correctly hashed GPT, a log without the
  UKI when the kernel PE was there, and the UKI twice (adversarial check,
  2026-10-17). The real log of 2026-10-17 passes the v2 reference of its
  image; thirteen changes to its RTMR1 are refused (`reference.rs`), among
  them the GPT after the UKI or the kernel PE, and the pinned digest under
  another event type or over other data. A v1
  reference is refused by name: by the proxy at start (89), by `image
  import` and `up` (2).
- `measurements/0.2.0-beta.5.json` names the RTMR1 and RTMR2 of the
  2026-10-17 run's quote (x0ne-cvm 0.2, disk tar `48180d78…`), unsigned:
  `x0ne-infer` outside cloud mode, `x0ne-infer-ssh-chat` and
  `x0ne-decide-verify` of this version pin them by default.
- Preflight read the spot H100 quota from `regions.get` only. That list had
  no `PREEMPTIBLE_NVIDIA_H100_GPUS` in us-east5, europe-west4 nor us-central1
  on a real project (2026-10-17): every zone was refused (91). Without the
  metric, the limit is now read from the Cloud Quotas API
  (`PREEMPTIBLE-NVIDIA-H100-GPUS-per-project-region`: the region's entry, else
  the default one; none or 0 is 91), and the usage is the H100 GPUs of the
  project's VMs in the region. A 400, 403 or 404 from Cloud Quotas is a
  warning: Google refuses the create if there is no quota, and nothing is
  billed. A zone where `PREEMPTIBLE_CPUS` or `PREEMPTIBLE_LOCAL_SSD_GB` cannot
  hold the machine (a3-highgpu-1g: 26 vCPU, 750 GiB) is skipped; 91 if none
  is left.
- Cloud mode serves a sharded model when the catalog pins its served-model
  digest. The TD checks every shard against the catalog, merges them, and
  serves only if sha256(weights ‖ config.json ‖ tokenizer.json) is the pin
  (`registry::pull_served`): 77 otherwise, 96 for a model with no pin. The
  served digest of `qwen3:8b` is pinned (two pulls from a local mirror,
  2026-10-16); `qwen3:14b` and `qwen3:32b` are refused before any VM, "no
  served digest pinned yet". `qwen3:8b` needs a `--max-run` of 24 min or more
  (boot timeout 18.8 min plus 5).
- The quote adds Google's Confidential Computing surcharge for A3 spot,
  0.4391592 USD/h, read on its pricing page on 2026-10-16, printed with the
  page and the date; `--max-cost` is compared to the total. No Cloud Billing
  SKU is read for it. The fallback table is dated 2026-10-16 and covers
  every region `up` may pick: us-central1 6.60, europe-west4 6.26, us-east5
  3.16 USD/h.
- Ctrl-\ (SIGQUIT) in the chat of `cloud run`, or while `up` serves, killed
  the lifecycle without deleting the VM. It now stops the session like
  Ctrl-C: the VM is deleted.
- `x0ne-engine cloud chat` on a session whose `up` process is gone sent the
  session's API key to whatever listened on its port. Without `--listen`
  and `--key-file`, it now refuses (96) a session no live `up` holds.
- `chat` took a lock held by another process (a second `up` waiting for
  "yes", or a `down`) as proof that the session's `up` lived, and sent the
  key and the question to the dead session's port. It now also requires the
  session's control socket to be served by the pid its state names
  (SO_PEERCRED) and to answer the state's session id, else 96. A second `up`
  serves its socket before it writes its own state (up to 30 s, the relay's
  config); with the dead pid handed out to it again, the pid alone passed.
- `status`, `sweep` and `down` took the same lock as proof that the
  session's `up` lived. With a dead `up` and the lock held by another
  process: `status` said "alive", `sweep` left the dead session's VM out of
  its list, and `down` stopped the second `up` (exit 0) while the dead
  session's VM ran on to `--max-run`. All three now ask the control socket
  for the session id under the lock (not the pid, which `chat` still
  compares): no listener or another session's id is a dead `up`, and
  `down` then refuses (91), naming the VM and `sweep --project`. An `up`
  that holds the lock and does not answer within 2 s (Ctrl-Z, SIGSTOP,
  busy, or older than beta.5) is not taken for dead: `status` says it does
  not answer, `sweep` leaves its VM, `down` refuses (91) and says so.
- SIGUSR1, SIGUSR2, SIGALRM, SIGVTALRM, SIGPROF, SIGIO, SIGPWR, SIGSTKFLT,
  SIGXCPU, SIGXFSZ and the real-time signals ended a serving `up` without
  deleting the VM, left to run until `--max-run`. Each now stops the session
  like Ctrl-C. SIGKILL, faults, SIGTRAP, SIGSYS, SIGABRT and signal 33
  still end it: glibc keeps 33 (SIGSETXID) for itself and its `sigaction`
  refuses it. SIGSTOP freezes it. `--max-run` stays the backstop.
- `image import` caught Ctrl-C, SIGTERM and SIGHUP only: Ctrl-\ (SIGQUIT)
  and the signals above ended it with its transit bucket left in the
  project. It now catches the same signals as `up` and deletes the bucket.
- `cargo deny` and `cargo audit` ignore RUSTSEC-2026-0367, 0368 and 0369
  (`zip` 8.6.0, via `candle-core` 0.11.0 which requires ^8.6): all three are
  in `zip`'s `extract` and `extract_unwrapped_root_dir`, which nothing in
  the graph calls (SECURITY.md).
  Re-examine by 2026-11-17.
- With `--reference`, `image import` and `up` refuse a READY image whose
  recorded disk sha256 is not `image.disk_tar_sha256` (exit 91). beta.4 and
  beta.5 both build `x0ne-cvm-0-2`.
- The TD generates two tokens from a fixed prompt before `ready`, so the
  first question does not pay the GPU's first-run compilation. A failed
  warm-up is a load failure (72).
- `x0ne-engine cloud`, the relay and its chat color their output on a
  terminal: the steps bold, what the proof established green, what it did
  not and the costs yellow, failures red. `NO_COLOR` turns colors off,
  `CLICOLOR_FORCE=1` forces them; into a file or a pipe the output is
  unchanged, byte for byte. The relay's environment now keeps `TERM`,
  `NO_COLOR`, `CLICOLOR_FORCE` and `CLICOLOR` (docs/cloud.md, "Colors").
- On a real `cloud run` (2026-10-17), each line of the relay showed twice:
  its own, `x0ne-engine-proxy: …` on the standard error it inherits, then the
  lifecycle's, indented. `up` and `run` now start it with `--quiet`, a new
  relay flag: progress, verdict and refusals go to the events only, and the
  lifecycle prints them once. `REFUSED` now carries the relay's code, which
  only its own line showed. Started alone, the relay prints as before.

## 0.2.0-beta.4 — 2026-10-16

- `x0ne-engine cloud run <family:tag>`: `up` for one model, then a chat in
  the terminal. The chat is the relay's binary (`x0ne-engine-proxy chat`);
  each answer says whether the relay verified its receipt. `/exit`, Ctrl-D
  and Ctrl-C end the session and delete the VM. `x0ne-engine cloud chat`
  opens the same chat on a serving session. Tested against a fake Compute
  Engine and the real relay with a fake TD; not yet run on Google Cloud.
- `up` and `run` refuse, before the quote, a model that is not in the
  embedded catalog, is sharded, or has no served-model digest (exit 2).
  The catalog is read by one TOML parser, for this check and for the boot
  timeout below.
- A terminal hangup while `up` served skipped the VM deletion (a write to
  the closed terminal panicked). The lifecycle ignores those write errors;
  the VM is deleted.
- The relay's verdict lines are printed at step [3/5]; they were read from
  a field the relay does not send.
- Catalog: `qwen3:8b`, `qwen3:14b` and `qwen3:32b`, pinned to a Hugging Face
  commit with the sha256 of every file. Cloud mode does not serve them yet
  (sharded, no served-model digest). Sizing for one H100 80 GB in
  [`docs/pull-run.md`](docs/pull-run.md).
- `x0ne-engine cloud up` with a model cloud mode does not serve (sharded,
  or no served-model digest): the relay refuses it with 96 before its
  config, so no VM is created. Before, the TD exited 96 after boot.
- `make ci` extracts the commit with files dated at the run: on the shared
  target, a commit older than the previous build was not recompiled.
- Cloud mode runs on the GPU when the config attests one, without
  `--device`: the CPU is only ever `--device cpu`, typed. `--device cuda`
  with a config that attests the CPU only exits 95.
- On a GPU, the context and the prefill chunk are bounded by the memory left
  once the weights are resident (`cuMemGetInfo`), with a reserve of 2 % of the
  card: a longer prompt is a 413, not an out-of-memory mid-generation. Under
  1024 tokens of context the load is refused (exit 72). Computed for
  `qwen3:32b` on an H100 80 GB: about 37,900 tokens, prefill chunk 67.
- `x0ne-engine cloud up`: the boot timeout grows with the model's catalog
  bytes (12 min plus the bytes at 40 MB/s), and a `--max-run` under the boot
  timeout plus 5 min is refused before anything is created (exit 2).
- `measurements/0.2.0-beta.4.json` names no register: no image of this
  version has been measured. Cloud mode does not read this file:
  `x0ne-engine-proxy` pins the image reference.

## 0.2.0-beta.3 — 2026-10-16

- SECURITY: rustls 0.23.46. `cargo deny` reported RUSTSEC-2026-0346 to
  -0349 on 0.23.45, the version beta.2 locked.
- SECURITY: on the TD, TLS handshakes are bounded in time and in number;
  `x0ne-infer` (v1 and `--decide`) and `x0ne-infer-lab` answer 421 to a
  `Host` they do not serve (DNS rebinding). Details below.
- Cloud mode collects the TDX evidence before it pulls the model: off a TD
  it exits 64 with no outbound connection. `--catalog-endpoint <url>`
  stands in for the catalog in tests.
- The first run on GCP TDX + H100 (2026-10-14, beta.2) refused its own
  event log; the CCEL parser now accepts the Spec ID header at index 1 and
  stops at the 0xFF padding after the last event.
  `measurements/0.2.0-beta.2.json` records the RTMR1/RTMR2 of that run's
  test image.
- `x0ne-engine-lab`: the lab server in its own package (below).
- `measurements/0.2.0-beta.3.json` names no register: no image of this
  version has been measured. A release `x0ne-infer` outside cloud mode
  serves only under `--expect-measurement` or `--measurements` (exit 87).
  Cloud mode does not read this file: `x0ne-engine-proxy` pins the image
  reference.

### 2026-10-16 — packaging checks

- `x0ne-engine-cloud --version` and `x0ne-engine-proxy --version` print the
  git commit when `X0NE_GIT_SHA` is set at build time, like `x0ne-engine`
  and `x0ne-infer`; `make deb-check` requires the commit from each of the
  four binaries, not once in the whole output.
- `make deb-repro`: the second build runs without sccache, in its own
  emptied target, so identical `.deb` means two independent compilations
  (`DEB_REPRO_SCCACHE=1` shares the cache again); `deb.json` records the
  sccache use of each build.

### 2026-10-16 — rustls 0.23.46

- SECURITY: `Cargo.lock` moves rustls from 0.23.45 to 0.23.46, which fixes
  RUSTSEC-2026-0346, -0347, -0348 and -0349. No other entry of the lock
  file changed.

### 2026-10-16 — bounded TLS handshakes on the TD; `Host` checked on the plain-HTTP servers

- Cloud mode: a TLS handshake has 10 s to finish, and at most
  `--max-handshakes` (default 256) are in flight. One more drops the oldest
  handshake of the source address (IPv6: its /64) holding the most; drops
  and timeouts are counted and logged at the 1st, 2nd, 4th… Before, a
  socket that never sent its ClientHello was held forever: 600 of them
  against a TD limited to 512 descriptors kept the relay out (accept failed
  with EMFILE); now the relay is served and every silent socket is closed
  within 15 s (`cloud_e2e_tests`). The GCP firewall rule is unchanged
  (`0.0.0.0/0` on 443). The TD only runs this code once the x0ne OS image
  is rebuilt and its measurements republished.
- `x0ne-infer` (v1 and `--decide`) and `x0ne-infer-lab` answer 421 to a
  request whose `Host`, or absolute URI, names anything but `localhost`,
  `127.0.0.1`, `[::1]` (any port), an `--allow-host <host[:port]>` value
  (repeatable) and, for the lab, `x0ne-infer-lab:8477`. A page whose name
  is rebound to the machine is no longer answered (DNS rebinding; Ollama
  CVE-2024-28224). A caller that names the server otherwise needs
  `--allow-host`. There is still no token on these servers.

### 2026-10-14 — the lab in its own package

- `make deb` also builds `x0ne-engine-lab`: `/usr/bin/x0ne-infer-lab` alone,
  `Depends: x0ne-engine`, synopsis `NOT CONFIDENTIAL`. No package depends on,
  recommends or suggests it; it is installed by name (edge channel of a
  local repository: no apt repository is published). The
  `x0ne-engine` package is unchanged (no lab binary, no relation to the lab
  package: `tests/lab_package.rs`, `tests/cloud_source_sweep.rs`).
  `x0ne-engine run` without the binary exits 1 with `apt install
  x0ne-engine-lab` and the warning. `make deb-check` installs both and checks
  the contents of each archive.

## 0.2.0-beta.2 — 2026-10-13

- The attested channel, end to end in simulation: `x0ne-engine cloud`
  drives `x0ne-engine-proxy`, which verifies the TD on the user's machine
  (DCAP, Google's launch endorsement, closed CCEL policy, TDX module, model
  digest, published measurements pinned by default) before any question
  leaves it. Each TLS connection carries its own TDX quote bound to the
  session's exporter, answers a client nonce, has no resumption or 0-RTT
  and only the two TLS 1.3 AES-GCM suites; the TD process is closed to its
  peers. The release ships no SSH path. Nothing has run on TDX hardware.
- `make deb`: a reproducible `x0ne-engine` package (same SHA-256 from two
  builds in two containers) built offline from the commit, holding the
  attested path only: `x0ne-infer`, `x0ne-engine`, `x0ne-engine-cloud`,
  `x0ne-engine-proxy`. No lab binary: `x0ne-engine run` needs
  `x0ne-infer-lab` built from source (`--features not-confidential`). The
  package replaces the `x0ne-infer` package, which ships the same server.
- `measurements/0.2.0-beta.2.json`, like beta.1's, names no register.

### 2026-10-13 — reproducible .deb, attested path only

- `make deb`, `make deb-repro`, `make deb-check` (`docs/ci-local.md`):
  release build of the commit in the CI container, no network, packaged
  with x0ne-OS `scripts/build.sh`; the `x0ne-engine` control file is
  `ci/deb/x0ne-engine.ctl`. Debian version
  `0.2.0~beta.2+git<commit time>.<sha7>`, above `0.2.0~beta1`.
- Tests: `cloud_source_sweep.rs` reads `ci/inside-deb.sh` and the control
  file: a lab or SSH binary, or a missing attested-path binary, is red.
- `x0ne-engine --version` and `x0ne-infer --version` print the git commit
  when `X0NE_GIT_SHA` is set at build time.

### 2026-10-13 — no SSH in the release

- SECURITY: the release archive, its SBOM build and the .deb packaging
  ship the attested path (`x0ne-infer`, `x0ne-engine`, `x0ne-engine-cloud`,
  `x0ne-engine-proxy`) and nothing that reaches SSH or the cloud CLI. The
  SSH demonstration client, renamed `x0ne-infer-ssh-chat`
  (`src/bin/ssh_chat.rs`), needs `--features not-confidential` like the
  lab server, so no release build has it (`tests/lab_option.rs`).
- Tests: every packaging list is resolved against the `[[bin]]` tables and
  each shipped source is read; a lab binary, an SSH or `gcloud` call, or a
  missing attested-path binary turns the test red. On the shipped client
  side the relay binary is the only source that carries a question.

### 2026-10-13 — downgrade, diversion, and a TD process closed to its peers

- SECURITY: the attested channel offers the two TLS 1.3 AES-GCM suites and
  nothing else (`tls::CIPHER_SUITES`). rustls 0.23.45's ring provider also
  offered ChaCha20 and six TLS 1.2 suites in our ClientHello (measured on
  the wire); a ChaCha20-only peer now gets no session at either end.
- Tests (downgrade): a raw TLS 1.2 ServerHello to our client and a raw
  TLS 1.2 ClientHello to our TD (protocol_version alert); documents and
  session quotes under other schema versions (88); a TD that serves the
  document but no session quote gets no key, no fallback to the v2 binding.
- Tests (diversion): a session quote from another platform under the same
  Intel root, or from a TD whose RTMR3 differs (88); a later connection
  steered to a twin TD booted with the same config dies at the handshake,
  before any of its routes. These checks existed; each removed alone turns
  its test red.
- SECURITY: `x0ne-infer` makes itself non-dumpable and sets `RLIMIT_CORE`
  to 0 before anything else (`harden::process`, exit 73 if it cannot).
  Measured on the real binary: `/proc/<pid>` owned by root,
  `/proc/<pid>/mem` EACCES and `PTRACE_ATTACH` EPERM for a process of the
  same user, core limit 0; an ordinary child of the same test is open.
- SECURITY: the TD's session key and both ends' TLS private key DER live
  in `harden::SecretBytes`: `memfd_secret` memory where the kernel has it
  (not readable even through `/proc/self/mem`, tested), else `mlock`ed
  pages with `MADV_DONTDUMP` and `MADV_WIPEONFORK`; zeroed before unmap
  (instrumented drop test). Not on the heap on Linux: construction fails
  instead. The copies ring and rustls make while parsing the TLS key are
  not covered.
- Source test: no shipped source on the attested path runs SSH; the
  demonstration client `x0ne-infer-chat` is the one exception left.
- Test fixture: `FakeIntel::two_platforms`, two PCKs under one root.

### 2026-10-13 — one full handshake per connection; bounded quotes

- SECURITY: TLS resumption and 0-RTT are off at both ends of the attested
  channel. rustls 0.23.45 defaults to a server session cache with two
  TLS 1.3 tickets per handshake and a resuming client; a resumed handshake
  calls neither the client- nor the server-certificate verifier. Now the TD
  keeps no session (`NoServerSessionStorage`), sends no ticket
  (`send_tls13_tickets = 0`), takes no early data (`max_early_data_size = 0`),
  and drops a resumed connection if one ever appears; the relay keeps no
  ticket (`Resumption::disabled()`) and sends no early data.
  One connection = one full handshake = one session quote.
- Tests: our client against a resuming TD, a resuming client asking 0-RTT
  against our TD, a genuine ticket from a resuming twin of the same TD key;
  each sees full handshakes only, no ticket, no early data, one certificate
  check per connection. A control pair with rustls' defaults does resume
  and accept 0-RTT, so the probes see it. Red before (the second connection
  resumed; the TD sent 2 tickets); each side's lines removed alone turn its
  own test red.
- SECURITY (denial of service): `GET /v2/attestation/session` passes a
  quote gate. One quote in flight (also against the document refresh),
  `--max-queue` requests may wait up to 10 s, the next get 503 with
  `Retry-After: 1`; one session quote per connection (the second is 429:
  same exporter, nothing new to prove); a global bucket of `--max-queue + 2`
  quotes refilled at one per second (429 with `Retry-After`). Default
  `--max-queue` 8: 8 waiting, a burst of 10. Sized so the honest relay
  never meets a limit: it needs one quote per new connection and at most
  `--max-queue + 1` new connections at once, plus the document's.
- Errors of a failed quote or re-attestation answer a fixed message; the
  cause stays in the TD's log.
- Tests: a burst of 24 session requests runs one quote at a time, sheds the
  excess with 503 at once, keeps `/v2/ready` answering, then quotes a new
  connection at the nominal cost; a second quote on one connection is 429;
  past the bucket, 429 with the wait; a failed quote names no path. Red
  with the gate removed (before, the hardware lock already serialized
  quotes, but every request queued behind it without bound). The source
  sweep now pins the five TLS settings, the `report_data` line of the
  session route, and the gate before the quote.
- Test hygiene: `a_gcloud_in_path_is_never_run` no longer needs a machine
  without gcloud: the test hides `/usr/bin/gcloud` and
  `/usr/lib/google-cloud-sdk/bin/gcloud` from the binary alone, in its own
  user and mount namespaces (no privilege, the machine untouched). Without
  the hiding, the binary runs the installed gcloud and the test fails.

### 2026-10-13 — one quote per TLS connection (binding v3)

- SECURITY: the v2 binding names the TD's static TLS key; whoever extracts
  it can relay the session and receive the session key. The TD now serves
  `GET /v2/attestation/session`: a TDX quote over `SHA-512("x0ne-infer/v3"
  ‖ exporter ‖ session key ‖ SHA-256(TLS SPKI) ‖ SHA-256(config))`, where
  the exporter is the RFC 8446 exporter of the connection the request came
  on (label `EXPORTER-x0ne-infer-attest/v3`, context the config's client
  nonce). Nothing in the request feeds `report_data`.
- The relay checks it before `POST /v2/key`, on the connection the document
  came on, and on every new connection before its first request
  (`remote_verify::SessionCheck`): same quote header and TD body as the
  document's, same PCK, v3 binding over its own exporter; else 88. A later
  connection that fails stops the session. `TdClient` now drives its own
  HTTP/1.1 connections over tokio-rustls (hyper `client` feature, no new
  crate) instead of reqwest's pool, so no request can leave on an unchecked
  connection.
- Tests: a man in the middle holding the TD's TLS key, a session quote over
  the v2 binding, a quote of another connection, a quote replayed onto a new
  connection; each refused with no key and no question at the TD, and the
  exporter equal at both ends of one connection only. Red before the
  relay's check, green after; removing the comparison turns all four red.
- Cost, synthetic (2026-10-13, release, loopback): check 0.45 ms, new
  checked connection 1.4 to 2.1 ms against 0.31 ms. The real TDX quote is
  not measured.
- Merged with the published measurements: `--measurements` is refused in
  cloud mode like `--expect-measurement` (the client pins the image).

### 2026-10-13 — attack of 2026-10-12 on the attested channel: three fixes

- CCEL policy (`reference.rs`): RTMR1 and RTMR2 are judged as a closed
  list on digests. Every extending event is the reference UKI or a
  reference section, or benign content (separator, EFI action, GPT in
  RTMR1) whose digest is the SHA-384 of its own data. `event_type`, which
  the quote does not cover, decides nothing any more.
- The TD's model digest is compared to the catalog's new `digest_sha256`
  pin (`models.toml`, measured 2026-10-13); an entry without a pin is not
  servable in cloud mode.
- The TDX module is verified (`tcb::tdx_module_status`): MRSIGNERSEAM,
  SEAMATTRIBUTES and the module ISVSVN against Intel's `tdxModule` /
  `tdxModuleIdentities`, per the PCS "Get TDX TCB Info" algorithm; the
  corresponding limitation is gone from `LIVE_LIMITATIONS`.

### 2026-10-12 — the attested channel

- New binary `x0ne-engine-proxy`, the relay the coming `x0ne-engine cloud`
  drives over stdin/stdout: it verifies the TD's attestation document on the
  user's machine — DCAP chain and live Intel PCS, Google's signed launch
  endorsement for the MRTD under the pinned `GCE-cc-tcb-root_1`, CCEL replay
  against an image reference (RTMR1, RTMR2; RTMR3 zero), NRAS per GPU, the
  v2 binding of the session key, the TLS key and the session config into
  `report_data`, the catalog model — then releases the session key over
  mutual TLS 1.3 and serves `/v1/chat/completions` on the loopback with
  verified receipts (`remote_verify.rs`, `relay.rs`, `local_api.rs`).
- `x0ne-infer --cloud-config <file>`: TLS 1.3 with the client pinned by
  SPKI, `/v2/attestation`, `/v2/key`, `/v2/ready`; the model is pulled from
  the catalog into `--model-store`; the environment key, `--decide` and the
  operator pins are refused (95). The v1 routes and `fn main` are unchanged.
- Exit codes added: 87 clock wrong, 88 channel not bound, 89 reference
  unusable, 95 cloud config, 96 pull, 97 TLS, 98 cloud server or local
  endpoint; the relay's control-stream and usage errors are 2.
- Tests: a simulated end to end (the real relay against a fake TD built from
  the real server code; 18 refusals before the key and one after it, a
  model whose digest drifts from the document, none with a question), mTLS
  at the route and nothing without a client certificate, the vendored Google
  root and endorsement,
  the relay as a process, a source sweep (no flag or variable sets an anchor
  or skips a check). Nothing has run on TDX hardware or a GCP VM.
- Dependencies: no new crate; rustls, tokio-rustls, hyper and hyper-util gain
  features (feature-level `Cargo.lock` diff only).

### 2026-10-13 — published measurements, pinned by default

- SECURITY: `measurements/<version>.json` (format 1: `version`, `algorithm`
  sha384, `image`, `registers`, `signature` reserved for the release key) is
  embedded in `x0ne-infer`, `x0ne-infer-chat` and `x0ne-decide-verify` and
  pinned by default; `--expect-measurement` overrides it, `--measurements
  <file>` replaces it. A release build with nothing to pin refuses to serve
  or to verify, exit 87, before any evidence; a debug or lab build warns. A
  file of another version, another format, with an unknown register or a
  signature this version cannot check is refused in every build.
  `x0ne-infer measurements <attestation.json>` prints the file for an image
  that answered. No image has been measured yet: the file of 0.2.0-beta.1
  names no register, so a release serves only under an explicit pin. The
  relay's refusal range in `x0ne-engine-cloud` extends to 87.

### 2026-10-13 — client challenge

- SECURITY: the attestation answers a nonce the CLIENT draws. Until now the
  only nonce was the server's, drawn at boot: one quote verified for every
  client, forever, and a recorded `/attestation` replayed. `GET
  /attestation?nonce=<64 hex>` now returns, under `challenge`, a TDX quote
  taken on request over `SHA-512(context ‖ nonce ‖ session key)`;
  `client::verify_attestation_document` takes a `ClientChallenge` and
  refuses a document that does not answer it, or whose two quotes report
  different images. `x0ne-infer-chat` and `x0ne-decide-verify ask` draw the
  nonce (`/dev/urandom`, 32 bytes); offline `x0ne-decide-verify` takes
  `--nonce`. The NRAS token still answers the boot nonce (stated in
  `client::LIMITATIONS`).

## 0.2.0-beta.1 — 2026-10-13

- First numbered version, a beta: the CLI, the receipt protocol and the
  cloud session commands may still change before 0.2.0. Built as the
  `x0ne-engine` (client: `x0ne-engine`, `x0ne-engine-cloud`,
  `x0ne-infer-lab`) and `x0ne-infer` (server) packages of x0ne OS 0.2.

### 2026-10-11 — security fixes

- Model and config files are read through `fsread::read_capped`: regular
  files only (a link to `/dev/zero` or a FIFO is refused, it used to read
  until killed or block forever), capped per file (config 1 MiB, tokenizer
  64 MiB, weights 128 GiB).
- `x0ne-engine pull` re-hashes an installed model before saying it is up to
  date, and replaces a damaged copy; a source file shorter than the catalog
  is exit 65 with the partial deleted (was exit 69, retried forever); lock
  files are removed on release; installed files are 0640, directories 0750.
  `list` checks sizes and says `DAMAGED`.
- `x0ne-infer` takes `X0NE_INFER_KEY_HEX` out of its environment before any
  thread starts (children no longer inherit it), holds the hex in zeroized
  memory and drops it right after the key release. `/proc/<pid>/environ`
  still shows the exec-time value (measured, tested): a known limit until the
  key travels by another channel.
- The key lease is judged on the monotonic clock as well as the wall clock
  (a wall clock moved back no longer extends it; back by more than 60 s ends
  it), and re-checked once a request holds the model, after the queue.
  `_verdict_of` is gone.

### 2026-10-10 — `x0ne-engine`: pull, run, list, rm from a pinned catalog

- New binary `x0ne-engine` (always built; serves nothing itself): `pull`,
  `run`, `list`, `catalog`, `verify`, `rm`, `path`. `run` execs
  `x0ne-infer-lab` (`--features not-confidential`) on `127.0.0.1:8477` by
  default; `x0ne-infer` is unchanged.
- Catalog `x0ne-infer/catalog/models.toml` (schema 1), embedded: `qwen3:0.6b`,
  `qwen3:1.7b` (sharded), `olmo2:1b`, each pinned to a commit with the size and
  sha256 of every file, measured on 2026-10-10. Validation refuses moving
  revisions, files outside an allow-list (pickle, Python, paths), malformed
  digests, unserved architectures.
- `src/registry`: download with HTTP range resume, https only (loopback
  excepted), size cap, sha256 from disk before install, `auto_map` and
  architecture checks on `config.json`, shard merge, atomic install by rename,
  per-name and per-file `flock` (two names can list the same file), XDG cache
  (`X0NE_ENGINE_HOME` overrides). The loopback exception parses the URL: a
  `http://127.0.0.1:x@host` endpoint is refused.
- `toml` moves from dev-dependencies to dependencies.
- 30 tests, including `pull` against a local server that corrupts, oversends,
  ignores ranges and 404s. Contract: `docs/pull-run.md`.

### 2026-09-28 — `confidential-decide`: attested decisions with receipts (prototype)

- Feature `confidential-decide` (implies nothing, implied by nothing, off by
  default): `x0ne-infer --decide` serves `POST /v1/decide` (encoder only) and
  `/attestation`, no chat, after the usual gate and key release. The GLiClass
  encoder runs on the CPU; `--attest cpu` is the intended product. Sealed
  weights open under the released key. Without the feature `--decide` exits 85
  (it used to be ignored as an unknown argument); `--device` other than `cpu`
  also 85; `--expected-model-digest` is mandatory with `--decide` (86, judged
  before any evidence); a digest mismatch exits 77. The digest covers weights,
  config, tokenizer and calibrator (`gliclass::digest_of`, equal to
  `sha256sum` of the four files concatenated). `GliClass::load_pinned`: no
  openness-table fallback.
- **New receipt protocol `x0ne-decision-receipt/v1`** (`decision_receipt.rs`),
  signed under its own domain `x0ne-infer/decision-receipt/v1\0`: model digest,
  verdict (PCK and registers), quote sha256, commitments to the input and the
  options under a CLIENT-drawn salt (`x0ne.salt_hex`, required), option count,
  outcome by index, timestamps; no text. Chat receipts (`receipt/v1`) are
  unchanged.
- `x0ne-decide-verify` (bin, same feature): the client's replay — quote to the
  pinned Intel root, key binding, signature over the given bytes, pins, quote
  hash, verdict registers, model digest, commitments against the client's own
  request, displayed outcome = signed outcome. `ask` mode checks the quote
  before sending.
- Tests on fixtures only (synthetic quote through the real TDX verifier and
  gate): end to end on a socket, every receipt leaf altered, re-signed
  misstatements, another request, cross-domain signature, pins, lease, refusals.
  Binary scan: a default release `x0ne-infer` holds none of the route, contract
  or receipt-domain strings; with the feature it holds them. Real OpenJev
  through the confidential path on a local workstation's CPU: 116 ms median over HTTP.
  Never run on TDX hardware: `docs/decide-manual-test.md` § 5.
- `engine-check`/CI: a `confidential-decide` test and clippy step.

### 2026-09-28 — `decide` feature: decision models (prototype)

- Feature `decide` (implies `not-confidential`, independent of `learn`):
  `POST /v1/decide` in `x0ne-infer-lab`. Context, question, 2 to 24 options,
  kind `choice` / `score` / `noul`; answers a distribution, the chosen option
  (always the caller's string, or `null` when abstaining), a confidence and
  `abstain` under `abstain_below`. Contract and bounds in `docs/decide-api.md`.
  Absent from `x0ne-infer` (binary scanned for the route and its contract
  marker) and 404 in a lab without `decide`, tested.
- `method: logprob` (`src/decide.rs`): the served generative model scores each
  option by the mean log-probability of its tokens after a fixed chat prompt;
  one prompt pass, then each option on a clone of the prompt's KV cache
  (`Decoder::forward_all`, `KvCache: Clone`, held to a whole-text pass).
- `method: encoder` (`src/gliclass.rs`): GLiClass uni-encoder over candle's
  `modernbert`, ported from gliclass 0.1.20 and Verdict's formatting and
  per-k temperature calibration. `--decision-model-dir`, weights refused unless
  their sha256 is pinned (`--decision-model-sha256`) or listed in
  `openness.rs` (OpenJev, `open-weights`). Unimplemented GLiClass options are
  refused at load; a caller string carrying a label marker is refused (400).
  Tiny random fixture from gliclass + transformers 5.17 in
  `tests/arch/gliclass/` (`scripts/make-gliclass-fixture.py`).
- Measured on CPU (`docs/decide-manual-test.md`): encoder within 5.6e-6 of
  Verdict's PyTorch engine on OpenJev over 28 cases; `logprob` 18/23 on
  OLMo-2 1B, 14/23 on Qwen3-0.6B; no answer outside the list.
- CI: the `lab` job also tests and lints `--features decide`.

### 2026-09-26 — open model families, `learn` feature

- The model family is read from `config.json` (`model_type`, else
  `architectures[0]`); an unknown, missing or contradictory one is refused at
  load with a sentence (`src/arch.rs`). Before, any config was parsed as Qwen3.
- OLMo 2 and OLMo 3 (Ai2, weights, code and data published) are served by a new
  decoder written in elementary tensor operations (`src/decoder.rs`): OLMo's
  post-norm and whole-projection QK-norm, OLMo 3's sliding window (3 layers in
  4) and YaRN on the full-attention layers. Held to transformers 5.17 on tiny
  random models (`tests/arch/`, generated by `scripts/make-arch-fixtures.py`):
  logits, attention, KV-cache generation past the window, gradients. OLMo 2 1B
  Instruct ran end to end on CPU. OLMo 3 has not run with real weights (7B is
  the smallest). Not implemented and therefore refused: `clip_qkv`, RoPE
  scalings other than YaRN, YaRN with `mscale`.
- Qwen3 still runs on candle's `qwen3`, through the same calls: Qwen3-0.6B gave
  byte-identical answers, same digest, before and after (6 prompts, CPU).
- Each family has its own chat template and stop tokens; the OLMo templates
  match transformers' rendering of the published ones, ids included.
  `LoadedModel::chat_prompt` (default: the Qwen3 template) is what both servers
  now call. `CandleQwen3Backend`/`Qwen3Artifacts` are renamed
  `CandleBackend`/`ModelArtifacts`; the old names remain as aliases.
- `src/openness.rs`: `fully-open` / `open-weights` / `unknown`, looked up by the
  sha256 of the weight file. The lab prints it at start-up.
- Feature `learn` (implies `not-confidential`): `/learn` page and five
  introspection routes in `x0ne-infer-lab` — tokens, embeddings with a stable 2D
  projection, next-token distribution, attention per head, one backward pass
  (loss, per-layer gradient norms, input saliency). Contract and bounds in
  `docs/learn-api.md`. Same admission as completions, nothing logged, base model
  only. Absent from `x0ne-infer` and from a lab without `learn` (404), tested.
- CI: the `lab` job also tests and lints `--features learn`. `Containerfile`:
  `--build-arg LAB_FEATURES="cuda learn"` (default unchanged).

### 2026-09-25 — lab container image, stricter lab flags, SIGTERM

- `Containerfile`: `x0ne-infer-lab` built with `cuda not-confidential` for
  compute capability 12.0 on CUDA 13.3.1 (cudarc 0.19.9 refuses 13.4),
  distroless runtime, uid 65532, no shell, ~0.8 GB; base images pinned by
  digest. A `confidential` target is defined and has never been built.
  `docs/container-manual-test.md` is the GPU test procedure.
- CI: a `lab` job (tests and clippy with `not-confidential`) and an `image` job
  that builds the image and runs `x0ne-infer/scripts/image-smoke.sh` without a
  GPU, through NVIDIA's stub libcuda in a CI-only target.
- `x0ne-infer-lab` refuses a malformed or missing flag value with exit 2. It
  used to keep the default: `--max-completion-tokens 70O` served up to 4096.
  `--help` and `--version` answer with exit 0 (it exited 2).
- The CUDA device is opened before the weights are read: no GPU is refused
  (exit 79) before 1.5 GB is read, with a line on passing one through CDI.
- SIGTERM and Ctrl-C stop the lab server gracefully. As PID 1 in a container
  it ignored SIGTERM: `podman stop` waited 10 s and killed it.
- `x0ne-infer/scripts/lab-replay.py`: replay prompts against one or more lab
  servers and compare two runs byte for byte.

### 2026-09-23 — RTMR registers read one slot early

- `tdx_quote::parse` read RTMR0 at body offset 280, which is MROWNERCONFIG;
  RTMR0..3 are at 328/376/424/472 (Intel TD Quote v4). Until this fix,
  `rtmr0` carried MROWNERCONFIG, `rtmrN` carried the real RTMR(N-1), and the
  real RTMR3 was never read. Every verdict, signed receipt and
  `--expect-measurement rtmrN=…` pin (server and `x0ne-infer-chat`) since
  the 2026-09-18 image-pinning entry used the shifted labels: a pin written
  against a pre-fix output now compares a different register and refuses.
  `mrtd` and `report_data` were at the right offsets.
- New test `tests/tdx_layout_live.rs` checks every parsed field against the
  vendored real quote at offsets written independently of the parser; the
  synthetic fixtures (`pcs.rs`, `tdx_quote.rs`) now place RTMRs at the real
  offsets — the `pcs.rs` one encoded the same shift, so no test could see it.

### 2026-09-23 — debug TDs refused, lab server, comments

- A TDX quote whose TDATTRIBUTES DEBUG bit is set is refused by
  `pcs::verify_tdx_chain`, hence by the server verifier and by the client,
  and by the structural verifier. Other attribute bits are not judged.
- `x0ne-infer-lab`: the adapter identity covers `adapter_config.json` (a
  changed config re-merges, `model` changes, `x0ne.adapter.config_sha256`);
  a panicking merge no longer blocks adapters until restart; a client that
  hangs up keeps its admission slot until its generation ends; malformed
  bodies get the JSON error; one generation at a time across base and
  adapters, not one per model.
- Comments brought back to the code: `main.rs` exit codes 65-68, `keys.rs`
  (AES-256-GCM is implemented in `seal.rs`; no KEK wrap), `server.rs` (the
  lab binary is another route to the model), `console.html` (the signature
  key is the one the receipt names).

### 2026-09-23 — documentation brought back to the code

- `SECURITY.md` said no feature lets a build serve unattested; the
  `not-confidential` feature builds `x0ne-infer-lab`, which does. Now said.
- The README credited the 2026-08-20 record with a live Intel PCS check it
  did not have, and linked it at the wrong path.
- Newly stated limitations: the TD attributes (debug bit included) are not
  checked; the v0 key is in the process environment from launch; the
  browser console's verdict and environment are the server's word.
- `x0ne-infer-chat` says it is a demonstration: the question is sent before
  any check and staged on the VM's disk. Its "billed" figure is the turn's
  wall-clock time, and now says so.
- `THREAT_MODEL.md`: contradictory rows merged (GPU `ueid`, image pinning),
  model-digest exit code corrected to 77, invariants renumbered.

### 2026-09-19 — a lab binary, opt-in

- `finish_reason` is `"length"` when the token ceiling cut the answer. Both
  servers wrote `"stop"` unconditionally, against what the docs promised.
- The `cuda` feature compiles again: it called a candle method that no longer
  exists, in a `cfg` arm no machine without the CUDA toolkit ever built. First
  run on an RTX 5090 (compute capability 12.0, CUDA 13.4): byte-identical
  output on Qwen3-0.6B across repeated requests (record not in this
  repository).
- `x0ne-infer-lab`, built only with `--features not-confidential`: serves
  outside any enclave, for tests and classrooms without TEE hardware. No
  attestation, no signed receipt, sealed weights refused, NOT CONFIDENTIAL at
  start-up and in every completion. `x0ne-infer` itself is unchanged and still
  exits 64 outside an enclave, with or without the feature. Never released.

### 2026-09-18 — receipt protocol v1 (BREAKING), GPU identity, clock

- **Breaking: receipts signed before this date do not verify after it.**
  The session key now signs `"x0ne-infer/receipt/v1\0" ‖ payload` instead of
  the bare payload, and the two commitments are
  `SHA-256(label ‖ salt ‖ text)` with distinct labels for prompt and output.
  Before, both commitments were the same function of (salt, text), so an
  opening for one opened the other. `console.html` and `x0ne-infer-chat`
  follow; a test reads the page and requires the same literals.
- A receipt that cannot be serialized is an error and the answer is withheld.
  It used to sign the empty message.
- **Per-GPU NRAS tokens are verified** instead of dropped: same pinned chain
  and nonce as the overall token, `measres == "success"`, and the claims
  common to NVIDIA's two published example policies.
- **The attested GPU must be the local one**: each token's `ueid` must equal
  the serial of the attestation certificate read from the local GPU. The
  derivation is taken from NVIDIA's nvtrust SDK and has not yet met a
  production token; a wrong derivation refuses, it does not admit. New exit
  code 83: local GPU certificate unreadable.
- **A host clock reading more than 300 s earlier than the NRAS token's signed
  `iat` is refused** — a rolled-back clock revives superseded collateral.
  GPU path only; `--attest cpu` still trusts the host clock.
- **Image pinning.** `mrtd` and `rtmr0..3`, read from the verified quote, are
  in every verdict and signed receipt. `--expect-measurement <reg>=<hex>` on
  the server (exit 84 if malformed) and on `x0ne-infer-chat`. The policy field
  existed before and was never read on the cryptographic path.
- **`x0ne-infer-chat` verifies the TDX quote itself** — chain to the pinned
  Intel root, quote bound to the session key, receipt signed by that key. It
  used to print the server's own verdict.
- Dependencies: `axum` 0.8, `thiserror` 2, `sha2` 0.11, `aes-gcm` 0.11,
  `rustls` 0.23.45 (RUSTSEC-2026-0285). The sealed-weights format is held by a
  known-answer vector computed outside this crate: weights sealed by an older
  build still open. `reqwest` stays on 0.12 on purpose — 0.13 swaps the
  crypto provider and moves TLS trust roots to the system store.
- A missing `X0NE_INFER_KEY_HEX` is a loud refusal, never a default.

### 2026-08-22

- Certificate validity windows checked on all three chains (PCK, Intel
  collateral, NRAS `x5c`).
- The Intel collateral signer must be the TCB Signing end-entity certificate.
- The key lease is consulted before each generation: nothing is served on an
  expired attestation.
- `accepted_cpu` and `require_gpu_cc_on` are read by the cryptographic path.
- `--attest <cpu+gpu|cpu>`: a TDX-only product, chosen by flag, never by
  fallback.
- CI uses `--features x0ne-infer/nvml`, not `--all-features`.

### 2026-08-21

- The TDX `report_data` is compared on the path that opens the gate: a
  genuine quote answering someone else's challenge is refused.
- TCB status and QE identity verified against live Intel PCS, fail-closed.
- CUDA inference measured on an H100 in CC mode (record not in this
  repository).

### 2026-08-20

- First end-to-end run on real hardware (TDX + H100 CC); record not in this
  repository.
