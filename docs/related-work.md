# Related work

## Related work

Products that serve LLMs from confidential hardware and let the client verify
it: [Tinfoil](https://docs.tinfoil.sh/verification/verification-in-tinfoil)
(vLLM in a CVM, reproducible image, measurements in a Sigstore log),
[Privatemode](https://docs.privatemode.ai/architecture/overview/) by Edgeless
Systems (attested channel to vLLM in a CVM), and the building blocks
[dstack](https://github.com/Dstack-TEE/dstack) (TDX guest OS and KMS,
externally audited) and
[Trustee](https://github.com/confidential-containers/trustee) (key broker,
Rust). They attest the *image*; this project additionally signs, per request,
what the generation loop did not do (no prefix cache, no shared KV cache, no
cross-client batching). An IETF individual draft,
[Attested Inference Receipt](https://datatracker.ietf.org/doc/draft-tsyrulnikov-rats-attested-inference-receipt/),
defines a receipt of the same intent in COSE/CWT/EAT; this project's JSON
receipt (`receipt/v1`) predates it and is not aligned with it.

The side channels the negative attestations answer: prefix-cache timing
([Early Bird, arXiv 2409.20002](https://arxiv.org/abs/2409.20002)), PROMPTPEEK
and InputSnatch on shared radix caches, CacheSolidarity (2026), SpliceLeak
([2606.21842](https://www.alphaxiv.org/abs/2606.21842)) and
[KVGov (arXiv 2608.09225)](https://arxiv.org/abs/2608.09225), which proposes
a per-principal salt in the cache key rather than no cache. On the hardware
side: [Blueprint, Bootstrap, and Bridge (arXiv 2507.02770)](https://arxiv.org/abs/2507.02770)
on NVIDIA GPU confidential computing, and the memory interposer and side
channel attacks on TDX and SEV-SNP (TEE.fail, DDRop, TDXRay and others) in
`x0ne-infer/THREAT_MODEL.md`, "Known attacks as of 2026-10-22".

## Why not vLLM?

The obvious question, and the answer costs us something, so here it is
plainly.

vLLM is a far better inference engine than this one and will stay that way.
It has paged attention, continuous batching, quantisation, speculative
decoding and a wide model zoo. This engine has none of those. One request
at a time. This repository publishes no measured comparison with vLLM or
llama.cpp. On architecture alone, expect this to be **several times
slower** on any workload with concurrency, and expect that to remain true
for a long time. Publishing a measured number is on the list, and until it
exists we will not quote one. The timings under [Proven on
hardware](hardware-runs.md) are this engine alone, one request at a time, on the H100 of each
run, in CC mode; no other engine ran on those machines, so they are not a
comparison with vLLM.

What we get for that price is the thing vLLM cannot give us:

- **The negative attestations ([receipts](verification.md#receipts)).** They are properties of the generation
  loop. Bolting them onto an engine whose headline feature is a shared
  prefix cache means removing the feature.
- **A signed receipt per request.** Salted commitments over the exact text,
  under a key the hardware attested, built by the process that generated
  the tokens right after generating them; it is not a middleware.
- **Refusal as the default.** No proof, no socket — enforced by a type that
  only the attestation sequence can construct, not by a check somebody can
  forget to call.

Chasing vLLM's feature list would mean losing on their ground with their
weapons. The trust layer is the part that does not get copied in a week.
