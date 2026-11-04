# x0ne-Engine documentation

## The crates

Five Rust crates. **`x0ne-infer`** is a confidential-native inference server
that **refuses by construction to serve outside a cryptographically verified
trusted execution environment**; **`x0ne-confidential`** is the attestation
vocabulary it is built on (evidence, verdict, policy, key release, receipts);
**`x0ne-engine-cloud`** quotes, creates, watches and deletes the one
confidential GPU VM of a cloud session ([`docs/cloud.md`](cloud.md));
**`x0ne-ohttp`** is a prototype of an Oblivious HTTP layer that no x0ne binary
uses ([`x0ne-ohttp/README.md`](../x0ne-ohttp/README.md)); **`x0ne-sdk`** is a
prototype of the caller's side, which checks a TD's attested key and seals a
question to it, on the TD's `/v2/e2e/*` routes, tested on synthetic TDX
evidence and run on an H100 TDX VM on 2026-10-28, 2026-10-30, 2026-10-31 and
2026-11-01, under the TEST archive key and log ([`docs/sdk-e2e.md`](sdk-e2e.md)).
Written for operators, and for students and researchers who want to read how
an attested server is made ([related-work.md](related-work.md), `CITATION.cff`).

## The documents

How it works and what it proves:

- [verification.md](verification.md): what a verified verdict proves, what is
  actually verified, receipts, sealed weights, the CPU and GPU halves.
- [limitations.md](limitations.md): the honesty section (stated limitations)
  and what is out of scope.
- [hardware-runs.md](hardware-runs.md): proven on hardware, versus merely
  implemented.
- [benchmarks.md](benchmarks.md): every speed figure kept, all in CC mode,
  with its machine, build, date and raw rows; what was not measured.
- [session-kv-cache.md](session-kv-cache.md): a conversation's KV cache
  kept between its turns, on request: scope, lifetime, zeroing, what the
  receipt signs, what timing still shows, and the CPU measurements.
- [pinning.md](pinning.md): pinning an image, RTMR0 and Google's firmware.
- [../x0ne-infer/THREAT_MODEL.md](../x0ne-infer/THREAT_MODEL.md): the threat
  model and the known attacks.

The cloud path:

- [attested-channel.md](attested-channel.md): the attested channel between
  your machine and the TD, and every check the relay makes. Does not start
  from this source tree yet: no archive key is pinned, no image published.
- [cloud.md](cloud.md): `x0ne-engine cloud`, one confidential GPU VM per
  session, and how to build it.
- [model-key.md](model-key.md): the model key, released after the proof to
  the attested TD only.
- [sdk-e2e.md](sdk-e2e.md): the caller's side, prototype.

Running and using it:

- [running.md](running.md): building and running `x0ne-infer`.
- [api.md](api.md): what the API honours and refuses, the token ceiling,
  Open WebUI.
- [models.md](models.md): open models, the families the engine implements
  and what it refuses.
- [pull-run.md](pull-run.md): `x0ne-engine` pull, list, rm.
- [learn-manual-test.md](learn-manual-test.md): the model families checked
  against transformers, by hand.

Contributing:

- [testing.md](testing.md): the tests and their fixtures.
- [ci-local.md](ci-local.md): the local CI, `make ci`.
- [BRANCHING.md](BRANCHING.md): branches, commits and releases.
- [related-work.md](related-work.md): related work, and why not vLLM.
