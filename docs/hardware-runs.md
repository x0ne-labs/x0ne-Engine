# Runs on real hardware

## Proven on hardware, versus merely implemented

These are not the same claim, and the difference is usually where products
lie. The runs up to 2026-11-01 served their model on the engine before x0ne
Core, which this version replaces, and `ministral3:14b`, served from
2026-10-28 to 2026-11-01, is no longer in the catalog
([CHANGELOG](../CHANGELOG.md), 2026-11-04). Recorded in this repository,
except what an item says is kept outside it:

- 2026-10-14, GCP `a3-highgpu-1g` spot with TDX and an H100 in CC mode,
  0.2.0-beta.2, the attested channel. On the operator's machine the relay
  verified the TD's document in 1.2 s (TDX quote, event log replayed to the
  image reference, NRAS per-GPU token, model digest; Intel PCS, Google's
  launch endorsement and NRAS fetched live); the key was released and the
  model ready 2.4 s later; the first chat through the relay took 1.2 s. On
  the real TD a session quote replayed onto another connection and a foreign
  client certificate were refused, and the RTMR1/RTMR2 recorded in one pass
  matched when pinned in the next (same test image). Inference ran on the
  CPU (`--device cpu`); the
  burst limit on session quotes was not reached at the rate tried. Nine VMs,
  eight of them failed attempts, 54.7 billed minutes, about 6.02 $. Record,
  with the test image's deviations from x0ne OS:
  [`x0ne-infer/RUN_2026-10-14_beta2.md`](../x0ne-infer/RUN_2026-10-14_beta2.md).
- 2026-10-17, GCP `a3-highgpu-1g` spot in us-central1-a, TDX and an H100 in
  CC mode, 0.2.0-beta.5: `x0ne-engine cloud image import` of
  x0ne-cvm 0.2 (disk tar `48180d78…`, its `x0ne-infer` built from pre-release
  0.2.0-beta.5 sources), then `x0ne-engine cloud run qwen3:8b`: VM created, verdict
  VERIFIED on the operator's machine
  (RTMR1 and RTMR2 matched the reference, RTMR3 empty; Intel TDX and one GPU
  in CC mode; model digest `d750e7af…`), key released, chat and OpenAI API
  through the relay with receipts verified, VM deleted from the chat,
  estimated cost 1.38 USD. From the run's responses, kept outside this
  repository: 256 tokens took 6.13 to 6.14 s (three requests, one after the
  other), and each of the ten responses kept carries a receipt the relay
  verified that names `cuda:0` as the processor that ran the arithmetic, in
  an environment of Intel TDX with the GPU in CC mode: the TD's inference
  ran on the H100. The model was served before shards were
  served as published (0.2.0-beta.6).
  That client predates the reference signature check: it took
  the v1 reference unsigned (`x0ne-infer/tests/reference/x0ne-cvm-0.2-v1.json`).
  Kept here: its event log and TD report (`x0ne-infer/tests/ccel/*2026-10-17*`)
  and its RTMR1/RTMR2 (`measurements/0.2.0-beta.5.json`); the recording and
  the responses are not in this repository.
- 2026-10-18, GCP `a3-highgpu-1g` spot in us-central1-a, TDX and an H100 in
  CC mode, client and TD at 0.2.0-beta.7, image x0ne-cvm 0.2.3
  (disk tar `9537364d…`), `qwen3:32b`: verdict VERIFIED (RTMR1 and RTMR2
  matched the reference, RTMR3 empty; one GPU in CC mode; model digest
  `745f026c…`), 17 shards served as published, estimated cost 1.66 USD.
  From the run's responses, kept outside this repository: 256 tokens took
  13.84 to 13.91 s (three requests, one after the other), and each of the
  thirteen responses kept carries a receipt the relay verified that names
  `cuda:0`, in an environment of Intel TDX with the GPU in CC mode. The
  relay was built `--release`
  with debug assertions and `test-archive-key`: it pinned the TEST archive
  key, which signed that reference. RTMR1 and RTMR2
  equal what x0ne OS (not public yet) `cvm/predict_measurements.py` computed for
  that image. Kept here: its event log and TD report
  (`x0ne-infer/tests/ccel/*2026-10-18*`), its reference
  (`x0ne-infer/tests/reference/x0ne-cvm-0.2.3.json`), its whole quote and
  the reference's signature (`x0ne-infer/tests/run_fixtures/`) and its
  RTMR1/RTMR2 (named by `measurements/0.2.0-beta.7.json` until 2026-10-25,
  provenance TEST); the recording and the responses are not in this
  repository.
- 2026-10-22, GCP `a3-highgpu-1g` spot in us-central1-a, TDX and an H100 in
  CC mode, a rehearsal of the release run: client, relay and TD built from
  the 0.2.0-beta.7 sources of that day, image x0ne-cvm labelled 0.2.3 but
  rebuilt from those sources (disk tar `0146b079…`, not the 0.2.3 of
  2026-10-18), `qwen3:8b`: verdict VERIFIED (TCB UpToDate on Intel PCS
  collateral fetched live; RTMR1 and RTMR2 matched the reference, RTMR3
  empty; one GPU in CC mode; model digest `e1776f7c…`), estimated cost
  0.66 USD for a VM that lived 339 s from creation to deletion. The relay
  was built `--release` with debug assertions and `test-archive-key`: it
  pinned the TEST archive key, which signed that reference, and predates
  the transparency-log check. RTMR1 and RTMR2 equal what x0ne OS (not public yet)
  `cvm/predict_measurements.py` computed for that image. From the run's
  responses, kept outside this repository: 49 requests, 41 non-streamed and
  8 streamed, each with a receipt the relay verified that names `cuda:0`,
  in an environment of Intel TDX with the GPU in CC mode; 256 tokens took
  3.70 to 3.75 s (ten non-streamed requests); streamed prompts of 16,021
  and 32,021 tokens gave their first content after 7.77 and 27.62 s; three
  streams sent at once gave theirs after 0.22, 2.04 and 4.16 s. Its event
  log, TD report, quote and reference are not in this repository, nor are
  its RTMR1/RTMR2 in `measurements/`; its record, attestation document
  (quote and event log), responses and serial console are kept outside it.
- 2026-10-25, GCP `a3-highgpu-1g` spot in us-central1-a, TDX and an H100 in
  CC mode, client and TD built from the 0.2.0-beta.7 sources of that day
  that brought the one release recipe (`docs/ci-local.md`), image
  x0ne-cvm 0.2.4
  (disk tar `03d53a41…`), `qwen3:0.6b`: verdict VERIFIED (TCB UpToDate on
  Intel PCS collateral of evaluation 22 or later, fetched live; RTMR1 and
  RTMR2 matched the reference, RTMR3 empty; one GPU in CC mode; model digest
  `b430888f…`), estimated cost 0.37 USD. The relay was built `--release` with debug
  assertions and `test-archive-key`: it pinned the TEST archive key, which
  signed that reference, and the TEST transparency log, which held the
  reference at index 0 of a log of size 1. RTMR1 and RTMR2 equal what x0ne
  OS `cvm/predict_measurements.py` computed for that image. From the run's
  responses, kept outside this repository: two requests (2 and 32 tokens,
  1.24 and 1.22 s), each with a receipt the relay verified that names
  `cuda:0`, in an environment of Intel TDX with the GPU in CC mode. Kept
  here: its event log and TD report (`x0ne-infer/tests/ccel/*2026-10-25*`),
  its reference (`x0ne-infer/tests/reference/x0ne-cvm-0.2.4.json`) and its
  RTMR1/RTMR2 (`measurements/0.2.0-beta.7.json`, provenance TEST).
- 2026-10-25, a second VM of the same type, zone, image, client and TD, for
  `qwen3:32b`: the TD's own live TCB check refused its host ("TCB status
  OutOfDate for FMSPC 00806f050000 is not accepted (level dated
  2026-02-11T00:00:00Z)"), and `x0ne-infer` exited 76 at each of its 20
  starts. It sent no attestation document, so no key was released and no
  model was served. The client was still waiting for the TD when the
  operator deleted the VM, about seven minutes after its creation; estimated
  cost 0.82 USD. Which component of that host's TCB was below the level was
  not read. Its serial console is kept outside this repository.
- 2026-10-25, a third VM of the same type, zone and image for `qwen3:32b`,
  the client built from later sources of the same day (`x0ne-engine cloud up`
  reads the serial console until the verdict; the relay built as above), the
  TD unchanged: the TD refused its host on the same rating, FMSPC
  `00806f050000` OutOfDate (level dated 2026-02-11), and exited 76 at each
  start. No attestation document, no key released, no model served. `up`
  read the status block at 61 s of uptime (exit 76 after three restarts),
  printed x0ne-infer's line, deleted the VM and exited 97. On Google's
  operations: created at 18:06:34 UTC, RUNNING 34 s later, its deletion
  asked 133 s after creation and done 201 s after it; on the operator's
  machine the refusal was printed 95 s after RUNNING. Estimated cost
  0.40 USD. Its serial console is kept outside this repository.
- 2026-10-25, a fourth VM, five minutes later, same type, zone, image,
  client, TD and model: the same refusal (FMSPC `00806f050000`
  OutOfDate, exit 76 at each start), nothing sent or served, and the same
  end: `up` read the block at 61 s of uptime and exited 97. Created at
  18:11:48 UTC, RUNNING 34 s later, its deletion asked 132 s after creation
  and done 199 s after it; refusal printed 95 s after RUNNING. Estimated
  cost 0.40 USD. Its serial console is kept outside this repository.
- 2026-10-28, GCP `a3-highgpu-1g` spot in us-east5-a, TDX and an H100 in
  CC mode, under the transparent TCB policy (15:16 to 15:24 UTC): client,
  relay and TD built from the 0.2.0-beta.7 sources of that day, image
  x0ne-cvm 0.2.5.2 (disk tar `ad36866e…`), `ministral3:14b`: verdict
  VERIFIED under `transparent` (platform and QE UpToDate on Intel's standard
  collateral of evaluation 20, fetched live; the VERIFIED line named the
  platform's level of 2026-02-11 OutOfDate under Intel's evaluation 22, and
  INTEL-DOC-10003, INTEL-SA-01439 and INTEL-SA-01442 as not fixed; RTMR1 and
  RTMR2 matched the reference, RTMR3 empty; one GPU in CC mode; model
  digest `ba565c1c…`). The same attestation document, judged again on the
  operator's machine by the relay's verification code under `strict` (a
  tool outside this repository), was refused with 76: "TCB status
  OutOfDate for FMSPC 00806f050000 is not accepted (level dated
  2026-02-11T00:00:00Z)". From the run's responses, kept outside this
  repository: six prompts at temperature 0, five outputs identical to a
  reference recorded before the run and one differing from its 176th
  character; 19 responses, each with a receipt the relay verified and the
  TCB disclosure (`x0ne-infer/tcb-disclosure/v1`) naming the same
  advisories; a one-token answer in 1.20 s, relay and round trip included.
  The TD padded the time of each whole answer to a Padme number of 100 ms
  ticks, ten at least, so these times give no decode rate
  ([benchmarks.md](benchmarks.md#method)). Estimated cost 0.50 USD. The
  relay was built `--release` with
  debug assertions and `test-archive-key`: it pinned the TEST archive key,
  which signed that reference, and the TEST transparency log, which held it
  at index 0 of a log of size 1. RTMR1 and RTMR2 equal what x0ne OS (not public yet)
  `cvm/predict_measurements.py` computed for that image. RTMR0 was reported,
  not pinned; the OCSP of NVIDIA's chain was not checked. Its event log,
  quote and reference are not in this repository, nor are its RTMR1/RTMR2
  in `measurements/`; its record is kept outside it.
- 2026-10-28, a second VM of the same type and zone, for the caller-sealed
  path (20:13 to 20:21 UTC): client, relay, TD and x0ne-sdk built from later
  sources of the same day, image x0ne-cvm 0.2.5.3 (disk tar `55ff982a…`),
  `ministral3:14b`: the relay's verdict VERIFIED under `transparent`, on the
  same TCB line as above (RTMR1 and RTMR2 matched the reference and equal
  what x0ne OS (not public yet) `cvm/predict_measurements.py` computed for that image, RTMR3
  empty; one GPU in CC mode; model digest `ba565c1c…`). Then the
  application's side, on the operator's machine: x0ne-sdk built for wasm32
  with `test-anchor`, run in Node, fetched the TD's key document and the
  collateral through the relay (`GET /v2/e2e/key`, `GET /v2/e2e/collateral`)
  and verified the document itself under `transparent`: the TDX quote and
  its binding to the SDK's nonce, Intel's collateral, the event log against
  the image reference, admitted under the TEST archive key and TEST log at
  the caller's request, and NVIDIA's verdict on one GPU, which the document
  carries; the TCB line it returned was the relay's. It sealed one question
  (4,151 bytes sent, `POST /v2/e2e/chat`), opened the answer (32,800 bytes)
  and verified its receipt (the catalog's model digest, computed on
  `cuda:0`). A sealed request with one bit flipped after sealing got the
  TD's 400 through the relay. Under `strict`, the SDK refused the same TD
  at `tcb` with 76, as a strict relay does. A random word written in the
  question only was in no byte the SDK sent to the relay or received from
  it, in no file the run wrote on the operator's machine (the relay's logs
  included; the SDK's output and the run's log, which print it on purpose,
  left out) and not in the relay's state: that measures what the relay
  held, not what the host could read. Estimated cost 0.43 USD. One run and
  one sealed question; the relay was built as above, pinning the TEST
  archive key and the TEST log; RTMR0 was reported, not pinned; the OCSP of
  NVIDIA's chain was not checked. Its event log, quote, key document and
  reference are not in this repository; its record is kept outside it.
- 2026-10-30, GCP `a3-highgpu-1g` spot in us-east5-a, TDX and an H100 in
  CC mode, under the transparent TCB policy (18:22 to 18:28 UTC): client,
  relay, TD and x0ne-sdk built from the 0.2.0-beta.7 sources of that day,
  image x0ne-cvm 0.2.7 (disk tar `24708df3…`), `ministral3:14b`: verdict
  VERIFIED under `transparent` (platform and QE UpToDate on Intel's standard
  collateral of evaluation 20, fetched live; the VERIFIED line named the
  platform's level of 2026-02-11 OutOfDate under Intel's evaluation 22, and
  INTEL-DOC-10003, INTEL-SA-01439 and INTEL-SA-01442 as not fixed; RTMR1 and
  RTMR2 matched the reference and equal what x0ne OS (not public yet)
  `cvm/predict_measurements.py` computed for that image, RTMR3 empty; one
  GPU in CC mode; model digest `ba565c1c…`); the session served 187 s after
  `up` started. Then the application's side, on the operator's machine:
  x0ne-sdk built for wasm32 with `test-anchor`, run in Node, fetched the
  TD's key document through the relay, of schema `x0ne-e2e/key/v2`: it
  carried the GPU's evidence, bound by the TD's quote, and NVIDIA's verdict
  on it, which the TD had fetched from NRAS. The SDK verified that document
  under `transparent`, NVIDIA's verdict on one GPU included, and named the
  relay's TCB line; it sealed one question (4,151 bytes sent), opened the
  answer (32,800 bytes) and verified its receipt (the catalog's model
  digest, computed on `cuda:0`). A sealed request with one bit flipped
  after sealing got the TD's 400 through the relay; under `strict` the SDK
  refused the same TD at `tcb` with 76. A random word written in the
  question only was in no byte the SDK exchanged with the relay, in no file
  the run wrote on the operator's machine and not in the relay's state. Five
  more key documents, each fetched fresh, were v2 with one GPU's evidence
  and NVIDIA's verdict, its overall result true; each took 1.28 to 1.75 s at
  the client (median 1.31 s), the TD's evidence, quote, NRAS call and the
  relay included. The run gave the TD no NRAS key: its boot report printed
  `nras_key=absent`, and NRAS answered without one. Through the relay, each
  of a series of 31 requests carried a receipt the relay verified; a
  `response_format` of type `json_schema` gave a
  document of that schema, and its receipt signed the schema's canonical
  SHA-256; a schema using `pattern` was refused with 400 naming it, and the
  session went on serving; a greedy request's receipt signed no sampling,
  `seed` 42 was signed as asked, and a request without a seed got one the
  TD drew, signed. Estimated cost 0.61 USD. The relay was built `--release`
  with debug assertions and `test-archive-key`: it pinned the TEST archive
  key, which signed that reference, and the TEST transparency log, which
  held it at index 0 of a log of size 1. RTMR0 was reported, not pinned;
  the OCSP of NVIDIA's chain was not checked. One sealed question, answered
  whole: no sealed answer was streamed. Its event log, quote, key documents
  and reference are not in this repository; its record and serial console
  are kept outside it.
- 2026-10-30, a second VM of the same type, zone, image and client, for
  `qwen3:0.6b`, the relay under `strict` (18:28 to 18:32 UTC): the relay
  refused the TD's attestation document with 76, "TCB status OutOfDate for
  FMSPC 00806f050000 is not accepted (level dated 2026-02-11T00:00:00Z)".
  No question and no session key were sent to the VM, no model was served;
  `up` deleted the VM and exited 76, 190 s after it started. Estimated cost
  0.29 USD; 0.92 USD for the day, the image's import, its transit bucket
  and the egress (at most 0.02 USD) included.
- 2026-10-31, GCP `a3-highgpu-1g` spot in us-east5-a, TDX and an H100 in
  CC mode, under the transparent TCB policy, the first run with RTMR0
  pinned (07:18 to 07:26 UTC): client, relay, TD and x0ne-sdk built from
  the 0.2.0-beta.7 sources of that day, image x0ne-cvm 0.2.8 built by x0ne
  OS from them (disk tar `0b057316…`), its reference `gce-endorsed`, signed
  that day by the TEST archive key and held at index 0 of the TEST log of
  size 1, `ministral3:14b`: verdict VERIFIED under `transparent` (platform
  and QE UpToDate on Intel's standard collateral of evaluation 20, fetched
  live; the VERIFIED line named the platform's level of 2026-02-11
  OutOfDate under Intel's evaluation 22, and INTEL-DOC-10003,
  INTEL-SA-01439 and INTEL-SA-01442 as not fixed; Google's endorsed
  firmware binary `53c558fe…`, fetched, measures the quoted MRTD
  `c1ee9c16…`, svn 2; RTMR1 and RTMR2 matched the reference, RTMR3 empty,
  RTMR0 pinned to the reference's host events with that binary's
  configuration volume; one GPU in CC mode; model digest `ba565c1c…`), the
  key released and the session served 186 s after `up` started. Then a
  check of the run's kit, which read `up`'s Image line for its RTMR0
  clause, found the line cut at 300 characters by `up` (the relay wrote
  321) and stopped the run; the VM was deleted, estimated cost 0.40 USD.
  `up` shows the verdict's lines whole since this run, up to 1000
  characters, control characters still replaced.
- 2026-10-31, a second VM of the same type, zone, image, client and TD
  (07:33 to 07:47 UTC): the same verdict VERIFIED under `transparent`, on
  the same TCB line, with RTMR0 pinned (RTMR1 and RTMR2 equal what x0ne OS
  (not public yet) predicted for that image before the run; model digest `ba565c1c…`); the
  session served 176 s after `up` started. The TD's serial console
  reported `efi_options=var:absent manager_env:set
  default_standard_output:journal`, `Secure boot disabled` and
  `nft_metadata=2`. Then the application's side, on the operator's machine:
  x0ne-sdk built for wasm32 with `test-anchor`, run in Node, fetched the
  TD's key document (schema `x0ne-e2e/key/v2`, 895,480 bytes, 1.43 s) and
  the collateral (2,838,218 bytes, Google's firmware binary included,
  1.12 s) through the relay and verified the document under `transparent`,
  NVIDIA's verdict on one GPU included, with the relay's TCB line; its
  verdict carried `rtmr0_hex` `10944cd1…`, the value the boot of 2026-10-18
  quoted, and `firmware_sha384_hex` `53c558fe…`: the SDK pinned RTMR0 too.
  It sealed one question (4,151 bytes sent, 1.20 s), opened the answer
  (32,800 bytes) and verified its receipt (the catalog's model digest,
  computed on `cuda:0`). A sealed request with one bit flipped after
  sealing got the TD's 400 through the relay; under `strict` the SDK
  refused the same TD at `tcb` with 76. A random word written in the
  question only was in no byte the SDK exchanged with the relay, in no
  file the run wrote on the operator's machine and not in the relay's
  state. A request of content type `message/ohttp-chunked-req` on
  `/v2/e2e/chat` got 415 from the relay, which carries no sealed stream:
  no sealed answer was streamed on TDX. Through the relay, in the clear: a
  streamed answer to a prompt of 538 tokens came in 23 frames, its first
  token at 0.211 s and its end at 2.312 s, its receipt verified on the
  final chunk and its text that of the same question answered whole
  (greedy). A conversation's KV cache kept between turns, against the same
  two turns without a conversation, n = 3, medians of the first token of
  the second turn: 0.321 s against 1.114 s for prompts of 4,548 then 4,590
  tokens, 0.421 s against 3.028 s for 8,548 then 8,590, 0.472 s against
  8.872 s for 16,548 then 16,590; each continued turn's signed `context`
  said `turn` 2 and `reused_tokens` 4,579, 8,579 and 16,579, and its
  receipt `kv_cache_shared_between_requests` true, false without a
  conversation; every receipt verified. Whole answers on prompts of 535 to
  8,048 tokens, 31 requests, each receipt verified by the relay; the TD
  padded their times to a Padme number of 100 ms ticks, ten at least, so
  they give no decode rate ([benchmarks.md](benchmarks.md#method)).
  Estimated cost 0.99 USD. The relay was built `--release`
  with debug assertions and `test-archive-key`: it pinned the TEST archive
  key, which signed that reference, and the TEST transparency log, which
  held it at index 0 of a log of size 1. The OCSP of NVIDIA's chain was
  not checked. One sealed question, answered whole. Its event log, quote,
  key document, reference, responses and serial console are not in this
  repository; its record is kept outside it.
- 2026-10-31, a third VM of the same type, zone, image and client, for
  `qwen3:0.6b`, the relay under `strict` (07:47 to 07:51 UTC): the relay
  refused the TD's attestation document with 76, "live Intel TCB: TCB
  status OutOfDate for FMSPC 00806f050000 is not accepted (level dated
  2026-02-11T00:00:00Z)". No question and no session key were sent to the
  VM, no model was served; `up` deleted the VM and exited 76, 189 s after it
  started. Its serial console reported the same three lines as the
  second's. Estimated cost 0.29 USD; 1.72 USD at most for the day, the
  image's two imports, their transit buckets and the egress (at most
  0.02 USD each) included.
- 2026-11-01, GCP `a3-highgpu-1g` spot in us-east5-a, TDX and an H100 in
  CC mode, under the transparent TCB policy, the first sealed stream on
  hardware (00:14 to 00:20 UTC): client, relay and x0ne-sdk built from the
  0.2.0-beta.7 sources of the evening of 2026-10-31, the tree that carries
  the relay's sealed stream; the TD as on 2026-10-31, image x0ne-cvm 0.2.8
  imported again (disk tar `0b057316…`, 0.86 GiB), its reference
  `gce-endorsed` signed by the TEST archive key and held at index 0 of the
  TEST log of size 1, `ministral3:14b`: verdict VERIFIED under
  `transparent` on the same TCB line as 2026-10-31 (platform and QE
  UpToDate on Intel's standard collateral of evaluation 20, fetched live;
  the platform's level of 2026-02-11 OutOfDate under Intel's evaluation 22,
  INTEL-DOC-10003, INTEL-SA-01439 and INTEL-SA-01442 named as not fixed;
  Google's endorsed firmware binary `53c558fe…` measures the quoted MRTD
  `c1ee9c16…`, svn 2; RTMR1 and RTMR2 matched the reference and what x0ne
  OS predicted for the image, RTMR3 empty, RTMR0 pinned; one GPU in CC
  mode; model digest `ba565c1c…`), the key released and the session served
  185 s after `up` started; the TD's serial console reported the same three
  lines as on 2026-10-31. Then the application's side, on the operator's
  machine: x0ne-sdk built for wasm32 with `test-anchor`, run in Node,
  fetched the TD's key document (888,352 bytes, 1.44 s) and the collateral
  (2,838,218 bytes, 1.36 s) through the relay and verified the document
  under `transparent`, RTMR0 pinned (`rtmr0_hex` `10944cd1…`,
  `firmware_sha384_hex` `53c558fe…`), NVIDIA's verdict on one GPU included;
  it sealed one question (4,151 bytes, 1.20 s), opened the answer (32,800
  bytes) and verified its receipt. Then the same question as a sealed
  stream, the first on hardware: `POST /v2/e2e/chat` as
  `message/ohttp-chunked-req` (4,152 bytes) through the relay, which passed
  the TD's `message/ohttp-chunked-res` on frame by frame and opened none of
  it (its `sealed_streams` went from 0 carried to 1, none cut), opened by
  the SDK frame by frame, its receipt verified; the deltas equal the signed
  text, and the text is that of the same question
  answered whole (greedy). A sealed request with one bit flipped after
  sealing got the TD's 400 through the relay; under `strict` the SDK
  refused the same TD at `tcb` with 76. A random word written in the
  question only was in no byte the SDK exchanged with the relay, whole or
  streamed, in no file the run wrote on the operator's machine and not in
  the relay's state. Through the relay, in the clear: a streamed answer to
  a prompt of 538 tokens came in 23 frames, its first token at 0.250 s and
  its end at 2.418 s, its receipt verified on the final chunk and its text
  that of the same question answered whole (2.203 s). Estimated cost
  0.60 USD for the VM, 0.62 USD at most with the image's import, its
  transit bucket and the egress. The relay was built `--release` with debug
  assertions and `test-archive-key`: it pinned the TEST archive key and the
  TEST transparency log. The OCSP of NVIDIA's chain was not checked. One
  sealed question, answered whole and then streamed, from Node and not from
  a browser tab. Its event log, quote, key document, reference, responses
  and serial console are not in this repository; its record is kept
  outside it.
- 2026-11-03, GCP `a3-highgpu-1g` spot in us-east5-a, TDX and an H100 in
  CC mode, x0ne Core alone as of that day (the copy in `x0ne-core/` adds
  greedy decoding on the device), not the Engine: no relay, no receipt, no
  model key (23:25 to 23:42 UTC). A bench image, not
  x0ne-cvm: its TD quote was checked for the run's nonce, the TDX type and
  the DEBUG bit only, and the GPU was set ready in CC mode without NVIDIA
  attestation. `cuMemHostAlloc` was granted in CC mode; `qwen3:8b` (BF16)
  loaded in 3.6 s. Timed in the process, 256 tokens generated, three runs
  each, on the prompts of [`bench/results/prompts/`](../bench/results/prompts/):
  decode at 138.62 tok/s after a 20-token prompt, 129.30 after 3,827, 122.22
  after 7,926 and 116.43 after 16,120, with one CUDA graph per token; 108.55
  and 107.46 tok/s at 20 and 3,827 tokens without the graph. 33 tests of
  x0ne Core passed there: its logits against frozen float32 logits of
  Hugging Face transformers, and the graph against the step-by-step run,
  byte for byte. 14 minutes of VM, estimated 1.32 USD, at most 0.02 USD
  more for the bench image's import and its transit bucket. Its rows:
  [`bench/results/2026-11-03-h100-tdx-cc-core/`](../bench/results/2026-11-03-h100-tdx-cc-core/);
  its quote, manifest, `dmesg`, journal and log are kept outside this
  repository.
- 2026-11-04, GCP, us-east5-a, the first attempt of the Engine on x0ne Core,
  which did not run (03:20 to 03:24 UTC). The image x0ne-cvm 0.2.9 (disk
  tar `1ebd1348…`, 0.79 GiB, matching its reference; not published, and a
  build of this tree does not give it: [CHANGELOG](../CHANGELOG.md),
  0.2.0-beta.7, "x0ne Core is the served engine") was imported through a
  transit bucket, READY at 03:23:29. Then `x0ne-engine cloud up` refused the
  run's relay before asking for the token, with 95: "the relay pins the TEST
  archive key and the TEST transparency log, whose private keys are derived
  from public strings: anyone can sign a reference it would accept. Use a
  release build of x0ne-engine-proxy; no token was asked, nothing was
  created" ([cloud.md](cloud.md)). No VM was created, the image was deleted
  at 03:23:41, and the account was left with no instance, disk, address,
  image, firewall rule, bucket or snapshot. A release relay built from
  these sources refuses every reference before the signing ceremony (99,
  [cloud.md](cloud.md)): a release build of this Engine on x0ne Core cannot
  run on hardware before it.
  Estimated cost at most 0.02 USD, the image's import and its transit
  bucket.
- 2026-11-04, the same zone, GCP `a3-highgpu-1g` spot with confidential
  computing off (no TDX: `systemd-detect-virt --cvm` said `none`; the GPU's
  CC status OFF), not a product mode (the VM from 03:26 to 03:31 UTC). The
  product's `x0ne-infer`, the build in x0ne-cvm 0.2.9 (SHA-256 `0254fd6b…`,
  the unpublished image above), started with the image unit's sandbox and
  `--attest cpu+gpu`, exited 64 before serving ("no evidence, not serving:
  configfs-tsm: cannot create report session […]; not a confidential guest,
  or not root") and never listened on port 443. x0ne Core's bench of
  2026-11-03, under the `[Service]` sandbox of the image's
  `x0ne-infer.service` with `MemoryDenyWriteExecute=yes`, generated 64 tokens
  and exited 0. No speed figure of this VM is published: the figures of this
  repository are of runs in CC mode. 5.1 minutes of VM, estimated 0.45 USD,
  at most 0.02 USD more for the bench image's import. Its records are kept
  outside this repository.
- 2026-11-04, the same zone, GCP `a3-highgpu-1g` spot, TDX and an H100 in CC
  mode, under the transparent TCB policy, this Engine on x0ne Core (the VM from
  15:20 to 15:31 UTC): a TEST measurement, VERIFIED with the TEST archive key
  and the TEST transparency log, not a release measurement. The image x0ne-cvm
  0.2.9 above, imported again (disk tar `1ebd1348…`), its reference signed by
  the TEST archive key and held at index 0 of the TEST log of size 1; the
  `x0ne-engine` front and the relay built from earlier sources of the same day
  with those TEST pins; `up` built from the sources of 2026-10-31, before `up`
  refused a relay that pins the TEST keys (2026-11-03): the `up` of this tree
  refuses this relay with 95. `qwen3:8b`: verdict VERIFIED under `transparent`
  (platform and QE UpToDate on Intel's standard collateral of evaluation 20,
  fetched live; the platform's level of 2026-02-11 OutOfDate under evaluation
  22, INTEL-DOC-10003, INTEL-SA-01439 and INTEL-SA-01442 named as not fixed;
  Google's endorsed firmware binary `53c558fe…` measures the quoted MRTD
  `c1ee9c16…`, svn 2; RTMR1 and RTMR2 matched the reference and what x0ne OS
  predicted for the image, RTMR3 empty, RTMR0 pinned; one GPU in CC mode; model
  digest `e1776f7c…`), the session served 132 s after `up` started (creation 39
  s, from `[3/5]` to `[4/5]` 65 s, key and model 11 s), `qwen3:8b` loaded on
  x0ne Core with its whole context, 40,960 tokens (`/x0ne/v1/status`). The SDK,
  in Node through the relay, verified the key document itself under
  `transparent`, sealed one question, opened the answer and verified its
  receipt; a sealed request with one bit flipped got the TD's 400; under
  `strict` the SDK refused the same TD at `tcb` with 76; a random word written
  in the question only was in no byte exchanged with the relay, in no file of
  the run and not in the relay's state. Then 36 answers streamed through the
  relay, 1,024 tokens each but a 16-token warm-up, every receipt verified by the
  relay: decode 136.87 tok/s after a prompt of 58 tokens, 128.98 after 3,568 and
  116.17 after 16,068 (medians, n = 7, frame by frame on the client's clock),
  the first content at 0.202, 0.705 and 8.042 s
  ([benchmarks.md](benchmarks.md#this-engine-on-x0ne-core-h100-streamed-through-the-relay)).
  Estimated cost 1.07 USD for the VM, 1.09 USD at most with the image's import,
  its transit bucket and the egress. No OCSP response about NVIDIA's chain was
  stapled. Its rows:
  [`bench/results/2026-11-04-h100-tdx-cc-core-e2e/`](../bench/results/2026-11-04-h100-tdx-cc-core-e2e/);
  its event log, quote, reference, receipts and serial console are kept outside
  this repository.
- 2026-11-04, the same zone, GCP `a3-highgpu-1g` spot, TDX and an H100 in CC
  mode, x0ne Core alone on the bench image of 2026-11-03, not the Engine: no
  relay, no receipt, no model key (the VM from 16:27 to 16:52 UTC). The TD
  quote was checked for the run's nonce, the TDX type and the DEBUG bit
  only, and the GPU was set ready in CC mode without NVIDIA attestation.
  x0ne Core's per-token bench, `x0ne-core-itl`, ran `qwen3:0.6b`,
  `qwen3:8b`, `qwen3:14b` and `qwen3:32b` (BF16, each file checked against
  the catalog's size and SHA-256 before any load) on prompts of 128 to
  16,384 random token ids, 256 tokens generated, five runs per length:
  `qwen3:8b` decoded at 138.21 tok/s after 128 prompt tokens and 116.58
  after 16,384, `qwen3:32b` at 38.37 and 35.15
  ([benchmarks.md](benchmarks.md#x0ne-core-alone-h100-in-cc-mode-four-models)).
  The run was interrupted after the last bench (exit 130 at 16:51): the
  records after the measures (`conf-compute`, the kernel log, the Xid
  check) were not taken. Estimated cost 2.37 USD for the VM, 2.39 USD at
  most with the image's import and its transit bucket; the VM served other
  measurements too, not published. Its rows:
  [`bench/results/2026-11-04-h100-tdx-cc-core-bench/`](../bench/results/2026-11-04-h100-tdx-cc-core-bench/);
  its quote, manifest, serial console and log are not in this repository.
- The DCAP and collateral verifiers against a genuine TD Quote v4 and
  in x0ne-sdk's build too, offline, on the quote of the 2026-10-18 run
  (`x0ne-sdk/tests/real_quote_2026_10_18.rs`). That quote binds the relay's
  channel, not an OHTTP key: the SDK's key check stops at `binding`.

The speed side of the H100 runs in CC mode of 2026-10-17, 2026-10-18,
2026-10-22, 2026-10-28 and 2026-10-31, x0ne Core alone on 2026-11-03 and
2026-11-04, and this Engine on x0ne Core on 2026-11-04, with their rows:
[benchmarks.md](benchmarks.md).

Reported, with no record in this repository: a run on 2026-08-21 with the
live Intel PCS check and CUDA inference on the H100; a turn of the
wake-ask-verify-sleep client, its time and cost not kept here.

No hardware run recorded for: a release build of this Engine on x0ne Core
(2026-11-04: `up` refused the run's test relay before any VM; the run after
it pinned the TEST keys), what landed on 2026-09-18 on
the server path
without the relay (per-GPU token checks and `ueid` binding, measurement
pinning, receipt protocol v1, client-side quote verification — fixtures
only; the 2026-10-14 run exercised the cloud path's checks, in the TD and
in the relay), sealed weights, CUDA inference outside cloud mode (the runs
of 2026-10-17, 2026-10-18, 2026-10-22, 2026-10-25 and 2026-10-28 ran it in
cloud mode),
more than one client per session (a session admits the one client key its
config pins) and more than three concurrent requests within a session,
more than one real GPU, and MIG
partitioning in CC mode (forum reports say it is refused outright; not
checked here), a model larger than `qwen3:0.6b` served in a TD by the
Engine of 2026-10-25 (the three VMs of that day meant to serve `qwen3:32b`
were refused by their host's TCB level), the caller-sealed path past one
question per run (asked whole, then streamed, on 2026-11-01) or under keys
other than the TEST ones, a sealed stream opened in a browser tab (the SDK
ran in Node), and the relay's
transparency-log
check against a log other than the TEST one (no public log or witness is
pinned).
Where a [limitation](limitations.md) says "not yet", it means not yet,
not "probably fine".
