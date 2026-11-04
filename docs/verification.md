# What a verdict proves

No proof → no socket. No proof → the key is not read. Sealed weights open
only with that key.

**What that proves, precisely** — and the precision is the product: a
verified verdict means *genuine confidential hardware answered THIS
session's challenge*. It means *this exact image is running* only through a
pin: the measurements published for the release (`measurements/<version>.json`,
built into server and clients, pinned by default) or the client's own
`--expect-measurement`. A release build with nothing to pin refuses to serve
or to verify (exit 87). The file of 0.2.0-beta.7 names the RTMR1 and RTMR2
that x0ne-cvm 0.2.4 (disk tar `03d53a41…`) quoted on GCE on 2026-10-25. That
run's reference was signed by the TEST archive key, so the file's
`provenance` says `"release": false` and a release build refuses it (exit
87); it stays TEST provenance until the signing ceremony, before which no
release archive key is pinned. The file of 0.2.0-beta.6 names no register.
The file of 0.2.0-beta.5 names the RTMR1
and RTMR2 that x0ne-cvm 0.2 (disk tar `48180d78…`) quoted on GCE on
2026-10-17, and is not signed (the beta.2 file holds the registers of the test image of the
2026-10-14 run). Every verdict carries its verifier's
limitations verbatim; `/attestation?nonce=<64 hex>` publishes the raw TDX quote and the
overall NRAS token (not the per-GPU tokens) so you can re-verify the CPU
half yourself — plus a second quote, taken when you ask, over the nonce
YOU sent, so the key is shown in hardware now and not in a document recorded
once. A client refuses a document that does not answer its nonce.

```text
collect evidence (TDX quote, GPU SPDM report)
  → verify (Intel DCAP chain to the pinned root
            + live Intel PCS TCB status & QE identity
            + NVIDIA NRAS tokens under the pinned intermediates 004, 005)
  → Attested proof (a type only this sequence constructs)
  → key release decision (same verdict, 300 s freshness), then the key is read
  → unseal weights if sealed (AES-256-GCM; public weights load in the clear)
  → bind socket → serve, with signed receipts
```

In v0 the key is an environment variable (`X0NE_INFER_KEY_HEX`) set by
whoever starts the process: it is in the process environment from launch,
before any proof. The server takes it out of its environment before any
thread starts (no child inherits it), zeroizes its copies and overwrites
the value in the block set at exec, which `/proc/<pid>/environ` shows
(measured 2026-10-22). Between exec and that first step, and in the
environment of whatever started it, the value is still there. What waits
for the proof is the server READING it. See [the honesty section](limitations.md).

## What is actually verified

Both halves of the machine, against **live** authorities, at startup:

| Half | Evidence | Verified against |
|---|---|---|
| CPU (Intel TDX) | TD Quote v4 via `configfs-tsm` | Full DCAP chain to the **pinned** Intel SGX Root CA; TCB status and QE identity against **live Intel PCS**, under a TCB policy (`tcb::TcbPolicy`, since 2026-10-28 one of two modes). `transparent`, the default: a level Intel's `standard` tcbInfo (evaluation 20 or later) lists as out of date or revoked refuses; Intel's latest tcbInfo and QE identity (`update=early`, evaluation 22 or later, Intel's TCB recovery of 2026-08-11) are also read, and the platform's TCB date, its status there and the advisories Intel lists against it are named in the verdict, the document, every receipt and the VERIFIED line, not refused: a platform served under `transparent` may lack the fixes those advisories name. `strict`, the client's choice (`--tcb-policy strict`): a level the latest tcbInfo, evaluation 22 or later, does not rate UpToDate or SWHardeningNeeded refuses. Under both, the tcbInfo, the QE identity and the PCK CRL must have been issued at most 24 hours before the verdict; the PCK certificate is checked against Intel's live PCK CRL, and its PCK CA and Intel's TCB signing certificate against Intel's root CA CRL, judged by its own window: Intel issues it for a year (see [limitations](limitations.md), "TCB policy"). A quote whose TD attributes carry a debug or profiling bit (Intel's TUD and TUP groups, bits 15:0), MIGRATABLE, or a reserved bit Intel says must be 0 is refused, each by name |
| CPU (AMD SEV-SNP), client side only, since 2026-10-28 | Attestation report (0x4A0 bytes) and the VCEK or VLEK that signed it, as an SEV-SNP guest would serve them; this version's TD does not collect them | AMD ARK, ASK and ASVK **pinned** by SPKI digest; **AMD's CRL** for the product fetched from KDS at each verdict, KDS's clock within 300 s; the written SEV-SNP reference (`--reference`, signed and logged like the TDX one): products, a TCB floor raised over the default of AMD's bulletins (under `transparent`, a bulletin less than a year old is named, not imposed; under `strict`, all are imposed), SMT, the cloud providers whose VLEK is accepted, the predicted launch digests. DEBUG, a migration agent and VMPL other than 0 refuse whatever the reference says. Not verified: how the launch digests were predicted, and what the guest loads after launch (no event log). See [limitations](limitations.md), "AMD SEV-SNP" |
| GPU (NVIDIA H100 CC) | SPDM attestation report via NVML (build with `--features nvml`), read only once NVML says CC is on, in production, devtools off, and every GPU is of the architecture `--gpu-arch` names (74 otherwise) | **Production NRAS** (`/v4/attest/gpu`); the overall and per-GPU ES384 tokens verified under the pinned NVIDIA intermediates (004 RSA, 005 P-384), nonce echoed, each `ueid` matched to the local GPU's certificate serial; the signing chain's revocation judged from the OCSP responses the document staples (RFC 6960; `nextUpdate` required, a day old at most, the nonce when one was sent), under a policy the verdict names: a verified revoked refuses under both, `required` refuses anything short of a verified good, `transparent` (production) names the rest; no TD staples a response yet |

Every certificate in the Intel and NVIDIA chains is checked against its
own validity window (on Google's endorsement path the signer's is, the
pinned GCE root's is not), and the collateral signer is required to *be* Intel's TCB signing
certificate — the right subject, an end entity, a key marked for signing —
not merely something Intel's root once issued. Intel's root issues a great
many certificates, and a platform's own PCK key does not live at Intel.

The session's ed25519 public key is folded into the hardware nonces
(`report_data` = SHA-512 over context ‖ nonce ‖ pubkey), so the attestation
covers the very key that signs every receipt.

Runs on real hardware are listed in [hardware-runs.md](hardware-runs.md). The
first one recorded in this repository (GCP `a3-highgpu-1g`, TDX + H100 CC,
2026-10-14) is
[`x0ne-infer/RUN_2026-10-14_beta2.md`](../x0ne-infer/RUN_2026-10-14_beta2.md).

## Receipts

Every completion returns a receipt signed by the attested session key:

- salted commitments over the prompt and the output (dictionary-resistant;
  the salt is returned to the caller, who can verify or discard it),
- the model digest (over the **plaintext** weights + config + tokenizer),
- the execution environment and the full verdict, verifier string included,
- prompt/completion token counts and timestamps,
- **which processor did the work** (`cuda:0`, or `cpu` from a test build).
  The environment says what the machine proved; this says which of its
  chips ran the arithmetic. x0ne Core runs on GPU 0 and signs `cuda:0`
  (`core_backend.rs`, `compute_device`),
- **four negative attestations**: this engine did not share a KV cache
  between requests, did not share a prefix cache between sessions, did not
  batch your tokens with another client's, and logged no fragment of your
  text. Each is signed, so it is a statement you can hold us to later. They
  are a constant written in the code (`Isolation::NONE_OF_IT`), not a
  measurement taken per request; the KV-cache and logging claims are guarded
  by tests that read this crate's own source, not its dependencies,
- **on a turn of a conversation** (`x0ne.conversation`, since 2026-10-30),
  the first of those flags is `true`: the turn's keys and values stay in
  the TD for its conversation's next turn. Beside it, a signed `context`
  block: the turn's number in the conversation, the exact number of prompt
  tokens whose keys and values an earlier turn computed, a commitment to
  those token ids and one to the handle sent and the handle received. Read
  it as "context reused in this session". The relay and x0ne-sdk hold it to
  the request: a receipt that hides a reuse, or signs one the request did
  not ask for, is refused ([session-kv-cache.md](session-kv-cache.md)).

The server's state holds the model, the proof and the session identity, no
request text. That is how the code is written, not something a client can
check from outside: what the client checks is the image's measurements,
and those have no release provenance yet ([Pinning an image](pinning.md)). What
surrounds the server (a reverse proxy) is not covered.

### Why the negative attestations are the interesting half

They are the claims a provider cannot honestly make without owning its
generation loop. vLLM ships shared prefix caching as a *feature*, and the
2026 literature on cross-tenant cache timing attacks targets exactly that
surface. We avoid it by construction — there is no prefix cache in this
engine at all — and until 2026-08-22 we simply never said so.

They are also claims we try to hold to the code rather than to our word. Writing
the first one falsified it: the KV cache was cleared only on the normal exit
path, so a generation that failed mid-way left one caller's keys and values
in place for the next. The test that now guards it fails if the clear moves
back behind a fallible expression.

## Weight sealing

A client model can be sealed before it reaches an operator:

```sh
# client side, before the weights ever reach an operator
X0NE_INFER_KEY_HEX=<32-byte-hex> x0ne-infer seal model.safetensors model.sealed
```

The server reads the key only after the gate has opened and the release has
been re-decided against the same attestation verdict, then unseals. Sealing
protects the file at rest; outside cloud mode it protects nothing against
whoever sets `X0NE_INFER_KEY_HEX` on the server, because that person holds the
key. Public open-weights models load unsealed; the server tells the two apart
by the file's header.

In cloud mode the model owner keeps the key on their own machine and the
relay releases it only to the TD it verified ([docs/model-key.md](model-key.md)):
`x0ne-engine-proxy --model-key <file> --key-policy <file>`. After the verdict
and the owner's policy (images, TCB, GPU, expiry; exit 102 if not), the TD
quotes an X25519 key born at its boot together with a fresh nonce, and the
key goes to it wrapped by HPKE (RFC 9180, X25519, HKDF-SHA256, AES-256-GCM).
Tested against the simulated TD only: a replayed recipient quote, a key the
quote does not name, a wrapped key posted to another TD, a tampered sealed
file and other weights under the same key are each refused. `x0ne-engine
cloud up|run --model-key <file> --key-policy <file>` passes both to the relay;
the policy's `sealed_source` names the URL (https, hosted by the owner) and
the size of the sealed file, and the TD fetches it after its attestation,
refusing other bytes than the config pins before its TLS server opens. The
host sees that URL, the size and the ciphertext. No hardware run with sealed
weights is recorded in this repository.

## The CPU and GPU halves

`--attest cpu+gpu` (default) requires both halves: the TDX quote AND the
NVIDIA GPU attestation. `--attest cpu`, a CPU-only mode on a TDX guest
without a GPU, is refused in this version, by name (exit 90): x0ne Core runs on the GPU only,
and that mode ran on the engine before it ([CHANGELOG](../CHANGELOG.md),
2026-11-04).

What keeps a GPU verdict honest:

- a GPU attestation failure is never a fallback: it exits 74, 75, 78 or 83
  before verification, or 65 when a token fails verification, and serves
  nothing;
- a verdict on CPU evidence alone cannot describe a GPU. The receipt's
  environment reads `ConfidentialVm`, and a test refuses any path that would
  make a CPU-only proof claim `gpu_mode: On`;
- `compute_device` in the receipt says which processor ran the arithmetic,
  so "TDX enclave, CPU inference" and "TDX enclave, GPU inference" are two
  readable answers rather than one ambiguous one;
- an unknown `--attest` value is refused with exit 82. `--attest gpu` is a
  plausible thing to type, and guessing what it meant is how a cheaper
  guarantee gets served to someone who asked for the expensive one.
