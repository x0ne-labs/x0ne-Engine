# The model key: released after the proof, to the attested TD only

A model owner can serve private weights on a rented confidential VM. The
weights reach the VM sealed. The key that opens them leaves the owner's
machine only after the relay has verified the TD there, and only wrapped to an
X25519 public key that a fresh TDX quote of that same TD names. No x0ne service
holds the key or sees it in the clear: there is no x0ne key server in this
path.

> **Status: built and tested against the simulated TD (`cloud_e2e_tests.rs`),
> never run on TDX hardware.** No hardware run with sealed weights is recorded
> in this repository. The sealed file is hosted by the owner at a URL the
> policy names, and fetched by the TD after its attestation (see
> [Limits](#limits)). This document does not make the cloud session
> end-to-end confidential: [docs/cloud.md](cloud.md#what-it-does-not-protect)
> lists what it does not protect, and all of it still applies.

## Who holds what

| party | holds | never holds |
|---|---|---|
| model owner, on their machine (`x0ne-engine cloud up\|run --model-key --key-policy`, which passes both to `x0ne-engine-proxy`) | the 32-byte model key (file mode 0600), the release policy, the sealed file at the URL the policy names | the TD's X25519 private key |
| the TD (`x0ne-infer --cloud-config …`, the sealed file fetched from the policy's `sealed_source`, or given as `--sealed-weights`) | the sealed file; an X25519 key pair born at boot, in locked memory; after the release, the model key and the plaintext weights in TDX-protected memory | the model key before the release |
| the host, the cloud, x0ne | the sealed file, its URL and size, the public config and tokenizer of the base model, the HPKE ciphertext | the model key, the plaintext weights |

The relay and the model owner are the same machine. The owner who serves
their own model runs the relay with their key. Serving a third party's private
model on that third party's behalf, with a key the third party keeps, would
need the relay or a key service on the third party's side; that is not built.

## The protocol (`x0ne-infer/src/model_key.rs`)

```text
relay (the owner's machine)                              TD
  verdict on the document (remote_verify, steps 1-8)
  the owner's policy: image, TCB, GPU, expiry, model ── 102 if not
  release nonce N, 32 random bytes
  GET /v2/model-key/recipient?nonce=N ──────────────────▶ quote over SHA-512(CONTEXT ‖ N
                                                           ‖ kem_pk ‖ session key ‖ TLS key ‖ config)
  ◀──────────────────────────────────── {kem_pk, quote}
  same TD and platform as the document's quote; report_data
  binds N and kem_pk ────────────────────────────────── 88 if not
  HPKE to kem_pk: info = the same fields, aad = the grant
  POST /v2/model-key {N, enc, grant, ct} ───────────────▶ N issued here, used once; HPKE open
                                                           with kem_sk; the grant names this
                                                           config's sealed file and is live;
                                                           keys::release (same verdict, 300 s);
                                                           unseal; served digest = the grant's (77)
```

Both requests travel on the attested TLS channel of the session (client
certificate pinned by the config, each connection's session quote checked
before its first request, [attested-channel.md](attested-channel.md)). The HPKE layer does not depend
on that channel for secrecy: a wrapped key that leaks from it still opens only
with the TD's X25519 private key. It gives no authenticity: HPKE's base mode
lets anyone who knows `kem_pk` wrap a key to it. Who may post one is the
channel's business (the client certificate the config pins: the owner's
relay), and a key that is not the owner's opens neither the sealed file the
config pins by its SHA-256 (72) nor, through another file, the promised
served digest (77).

**The recipient binding.** `report_data` of the recipient quote is

```text
SHA-512("x0ne-infer/model-key/v1" ‖ N ‖ kem_pk ‖ session_pubkey ‖ SHA-256(TLS SPKI) ‖ SHA-256(config))
```

every field 32 bytes. The relay recomputes it from the binding it already
verified on the document and from the nonce it just drew. The quote must also
come from the same platform (same PCK certificate) and carry the same TD
report up to `report_data` as the document's quote: same MRTD, RTMRs and TCB.
The public key is named in the quote itself, as Trustee's KBS protocol binds
its `tee-pubkey` [1].

**HPKE** (RFC 9180 [2], base mode, single shot):
DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, AES-256-GCM, the suite of Azure's
transparent KMS [3] and of Tinfoil's EHBP [4]. `info` is
`"x0ne-infer/model-key/v1/hpke\0"` followed by the five binding fields, so a
ciphertext made for one release, one session or one key does not open under
another. `aad` is the grant's exact bytes. The implementation (`hpke.rs`) is
checked against RFC 9180 A.1.1 and against the CFRG test vector for this
exact suite (draft-irtf-cfrg-hpke at commit `5f503c5`); an all-zero shared
secret, which a low-order public key produces, is refused at both ends.

**The grant** (`x0ne-infer/model-key-grant/v1`) says what the key is released
for: the base model, the sealed file's SHA-256, the served-model digest, the
policy's `not_after_unix` and the policy's SHA-256. The TD refuses a grant for
another sealed file, model or digest, or past its expiry, and stops serving
when its own wall clock passes that expiry: the serving lease is the shorter
of the usual 24 hours and the grant's `not_after_unix` (`bound_lease`).

**Whose clock.** The TD judges the grant's end on its own wall clock, and the
host provides that clock. The relay checks the policy's expiry on the owner's
machine before it releases the key, and never compares the TD's clock with
its own. A host that holds the TD's clock D seconds behind therefore gets D
more seconds of service past `not_after_unix`, up to 24 hours of the TD's
running time after the release, which the lease also counts on the monotonic
clock (a paused guest pauses that clock too). A test keeps this behaviour as
built (`model_key.rs`, `a_td_clock_held_back_serves_past_the_grant_end_by_as_much`).
No trusted time source is built.

**The TD's key pair** is drawn from the kernel at boot, kept in
`harden::SecretBytes` (out of the direct map where `memfd_secret` exists,
else locked and out of dumps), and dropped as soon as one key opened. Three
failed openings drop it too. A release nonce opens one attempt: the first
wrapped key that names it uses it up, whether it opens or not, and a retry
asks for a new recipient quote. A wrapped key recorded on the way opens nowhere
afterwards: a later boot has another key pair, and the quote of this boot named
this one.

**Before the evidence**: a TD given a sealed config and a `--sealed-weights`
file checks that file against the config's `sealed_sha256` first (77 if it is
another file), before it collects any evidence or contacts anything. A config
whose `sealed.source` names a URL and a size, and no file, is not refused (95
only when it has neither): the TD fetches the file after its attestation, as
it pulls the base model's public files, with the catalog's transfer code
(`registry::fetch_file`: ranges, resumes, the hash of the whole), pinned by the
config's size and SHA-256, and checks it the same way before its TLS server
opens, so before any key can arrive (77 for other bytes or for a file that is
not an envelope, nothing kept; 96 a source it may not use, see below, or a
transfer that failed). The source is never contacted before the evidence
(`tests/model_key_cli.rs`). After the release it unseals in memory (exit 72 if
the key does not open the file), computes the served-model digest over the
base model's `config.json` and `tokenizer.json` and the plaintext
`model.safetensors`, and refuses to serve a digest other than the config's
(77).

**On the TD's disk.** The verified sealed file stays under
`<model-store>/sealed/partial/<sha256>.part` for a restart. A transfer cut
short or a range the source keeps refusing (96) leaves that partial, and for a
ranged one its `.ranges` list, for the next attempt to resume. They hold only
bytes the source sent, at their offsets (zeros in the holes of a ranged
partial): the sealed file's bytes when the source serves the file the config
pins. The key and the unsealed weights never reach a file: the TD unseals in
memory, after the release, a file that is whole and hashed. Bytes that are not
the pinned file fail the hash of the whole (77) and are deleted, and so is a
pinned file that is not an envelope. A source that serves other bytes and is
cut before the end leaves them until the next attempt refuses and deletes
them; whoever holds the URL, and the host reads it in the config, can
download the same bytes.

## The owner's release policy

`x0ne-engine/model-key-policy/v1`, read by the relay before anything is
created (an expired or unreadable policy is exit 102, and no VM):

```json
{
  "schema": "x0ne-engine/model-key-policy/v1",
  "model": "qwen3:8b",
  "sealed_sha256": "<sha256 of model.sealed>",
  "weights_sha256": "<sha256 of the plaintext model.safetensors>",
  "images": [{ "mrtd": "<96 hex>", "rtmr1": "<96 hex>", "rtmr2": "<96 hex>", "rtmr0": "gce-endorsed" }],
  "tcb": ["UpToDate"],
  "require_gpu": true,
  "not_after_unix": 1798761600,
  "sealed_source": { "url": "https://weights.example/qwen3-8b/model.sealed", "size": 16388263269 }
}
```

`x0ne-infer seal` prints the two hashes and the sealed size. `sealed_source`
is where the TD fetches the sealed file: an https URL the owner hosts (a
bucket, a signed URL whose token sits in the query, never in userinfo), or
plain http towards the loopback on a bench. An https URL whose host is a
literal loopback, link-local, private (RFC 1918, ULA `fc00::/7`) or
unspecified address, in any spelling the URL parser accepts, or the name
`localhost`, `metadata.google.internal`, `metadata` or `instance-data`, is
refused when the policy and the config are read, and again by the TD before
any request (`registry::check_source_url`). Each image may also name
`rtmr0`: `"gce-endorsed"` requires the verdict to have pinned RTMR0 to
Google's endorsed firmware and the reference's host events (every verdict
does since 2026-10-31), 96 hex requires that exact value; absent, RTMR0 is
the reference's business alone. DNS resolution is not checked: a
name that resolves to such an address is not refused. No message prints the
URL's userinfo. `sealed_source` is copied into the
session config, so it is in every quote, and `x0ne-engine cloud` refuses a relay
config that has a model key and no source, before anything is created. Without
it the file must be on the VM already (`--sealed-weights`). The policy is judged after the
verdict, on the registers of the verified quote: an image it does not name, a
TCB status it does not list, no attested GPU when it requires one, another
model or a past expiry is refused with exit 102 before any request about the
key. The verdict's own checks (Intel chain, PCS status, Google's MRTD
endorsement, CCEL replay, the signed image reference) still run first: the
policy can only narrow them.

Unknown fields are refused, and so are an empty image list, a TCB value the
verifier never accepts and a `not_after_unix` of 0.

## Revocation

- **An image**: remove it from the policy. The next release refuses it.
- **Everything, at a date**: `not_after_unix`. No release after it by the
  relay's clock, and a TD that received the key stops serving once its own
  clock passes it. That clock comes from the host: held back, it serves as
  much longer, up to 24 hours of running time after the release (see
  "Whose clock" above).
- **A key**: re-seal under a new key and destroy the old one. Copies of the
  old sealed file then open nowhere.
- **A TD that already holds the key** keeps it until its lease ends (24 hours
  of its running time at most, or the grant's expiry by its own clock if
  sooner) or until the VM is deleted. There
  is no remote erase: a TD that would honour one is the same TD that the
  attestation already trusts with the key.

## What the host sees

The sealed file, its URL and its size (the plaintext size plus 37 bytes), the
base model's name and its public `config.json` and `tokenizer.json` (pulled
from the catalog), the timing and size of the two requests, then the same
traffic as any session. It does not see the key, the plaintext weights or the
HPKE plaintext. The URL is the owner's to choose: a name that says nothing
about the model gives the host only the size. A memory read of the TD by physical means (TEE.fail and DDR5
interposers, 2025) is out of this model, as in docs/cloud.md.

## Tests

Simulated TD, real relay, real TLS, quotes under a test root
(`cloud_e2e_tests.rs`):

- sealed weights released after the verdict and served under the policy's
  digest;
- an image the policy does not name: no request about the key (102);
- an expired policy: refused before anything is created (102);
- a replayed recipient quote, answering an earlier nonce: refused (88);
- a recipient key the quote does not name: the TD gets nothing (88);
- a wrapped key for one TD, taken off the wire and posted to another booted
  with the same config: refused because that TD never issued its nonce (403);
  once it has issued the same nonce, the key still does not open (403), and
  after three failures that TD stops and takes no fourth try (409);
- a release nonce used up: the relay's honest wrapped key, recorded on the
  way and posted after a corrupted copy failed under the same nonce, is
  refused (403) and nothing is loaded;
- a tampered sealed file: the binary refuses it before the evidence (77,
  `tests/model_key_cli.rs`), and the TD's loader refuses it after the release
  (72) if it got that far;
- a sealed file fetched from the policy's source: the bytes the config pins
  come back; other bytes under the URL are 77 and nothing is kept, nor is a
  pinned file that is not an envelope; a transfer cut short or a range refused
  keeps a partial of the source's sealed bytes only, never the key nor
  plaintext, and the next attempt completes it; a source the TD may not use
  (not https, a non-public literal address, a metadata name) is 96 before any
  request; no message carries a URL's userinfo (`registry/tests.rs`); off a
  TD, the source is not contacted before the evidence (`tests/model_key_cli.rs`);
- a TD whose clock is held back serves past the grant's end by as much,
  within 24 hours of running time from the release: the behaviour as built
  (`model_key.rs`);
- `x0ne-engine cloud`: `--model-key` and `--key-policy` go together, name
  files, are for `up` and `run` only, and the key typed in their place is
  refused unprinted; a relay config that names sealed weights without a model
  key, none with one, or no source, is refused before anything is created
  (`x0ne-engine-cloud/src/cli.rs`, `proxy_link.rs`);
- plaintext weights other than the policy names: not served (77);
- the two key routes do not cross (a public session takes no model key, a
  sealed one no random key) and recipient quotes are bounded (4 per boot).

Units: `hpke.rs` (RFC and CFRG vectors, wrong key, info, aad, enc or
ciphertext, low-order points), `model_key.rs` (binding, grant, policy
refusals, the source's URL and size, lease bound, the TD's clock as built,
file selection),
`cloud_config.rs` (the source travels in the config and is judged; a config
without one is byte for byte what it was), `registry` (`fetch_file` once only,
altered bytes and another size refused), `tests/model_key_cli.rs` (the binary's
`seal` output and its checks before the evidence), `tests/cloud_source_sweep.rs`
(the two routes and their binding stay as written).

## Compared with the field

Read on 2026-10-22 from each project's own pages; the quotes are theirs. What
is called "same" or "not here" is a reading of those pages, not a test of their
systems.

| | who holds the weights key | released on what proof | wrapped how | what x0ne does |
|---|---|---|---|---|
| Azure transparent KMS [3] | the KMS, a CCF service with a governed "key release policy"; "This version of KMS can only be used for testing and not for production" | attestation sent with a `wrappingKey` in the request; the claims it checks are not on the page | HPKE `DHKEM_X25519_HKDF_SHA256_HKDF_SHA256_AES_256_GCM`, receipts that "allow third parties to verify the authenticity of the generated keys" | same HPKE suite; no key service and no ledger: the owner's own relay decides, and the receipt of the decision is the relay's verdict log. A third party cannot audit the release, and nobody else holds the key |
| Apple PCC [5] | Apple; "all code and model assets use the same integrity protection" as the system volume. Request keys are per node, in the Secure Enclave: they "cannot be duplicated or extracted" | the device "will not send data to any PCC nodes if it cannot validate their certificates", and only to keys whose release is in the transparency log | requests encrypted to "the public keys of the PCC nodes that it has first confirmed are valid" | the TD's X25519 key plays the node key's part: born in the TD, named by a quote the owner verifies first. PCC protects the user's request this way; here it is the owner's weights. No transparency log of releases yet |
| Anthropic, trusted VMs [6] | "stored encrypted, decrypted at the loader, and never released from there"; "A keyserver can check this proof and only release decryption keys when the recipient has proven itself secure" | TPM-measured boot; "The decision of whether an environment is 'trusted' ultimately rests on the keyserver"; "just a sketch of our research" | not on the page | same shape, with the TDX quote in the TPM's place and the owner's relay as the keyserver. Built and tested, against the simulated TD only |
| Tinfoil [4][7] | nobody: "Model weights are loaded from disk after boot, so the CPU does not measure them"; integrity by a dm-verity root hash pinned in the enclave's config. Encrypted volumes with a "customer-held key" exist for containers, not for weights | the SDK checks the quote, the Sigstore bundle and "confirms the server's public key matches the one in the attestation document" | HPKE to the enclave's attested key for request bodies (EHBP); the TLS key is attested too | x0ne's weights are confidential, not only integrity-checked: the host sees a sealed file. Tinfoil's attested-key idea is applied to the key itself, not to the requests |
| Privatemode [8] | the service: a "Secret Service" keeps prompt keys and a "disk mounter" mounts "disks containing stored model weights as read-only devices"; whether weights are encrypted at rest, and under whose key, is not on the page | the Contrast Coordinator attests the workers; the client proxy verifies "before any inference communication begins" | not named on the page ("modern encryption schemes") | x0ne has no coordinator: one TD, one owner, one relay. The policy's images, TCB and expiry are the owner's, not a coordinator's manifest |
| Chutes [9] | the operator: a LUKS disk that "unlocks only after successful attestation" | attestation of the VM | disk key, release path not on the page | the same gate, but the key never belongs to the operator: the owner keeps it, and a copy of the VM's disk is a sealed file without its key |

What none of those pages describe, and this design does: a key the service
operator never holds, released by its owner's own verifier, with a replayed
or diverted recipient quote, a wrapped key posted to another VM and a
substituted file each refused by a test. What they have and this does not:
a transparency log of releases (Apple, Azure's receipts), a third party who
can audit a release, hardware runs, and any operation at scale.

## Limits

- Never run on TDX hardware or with a real GPU, and no cloud run with sealed
  weights: the `x0ne-engine cloud` path with a model key is tested at the
  command line and the config check only, the TD's fetch against a local
  server.
- **The owner hosts the sealed file.** `x0ne-engine` uploads nothing: the
  policy's `sealed_source` must be reachable from the VM (a bucket, a signed
  URL), and a download of its whole size is paid for on the VM's side.
- One file, one safetensors: sharded private models are not handled. The
  sealed file and the plaintext are both in memory while the weights open
  (at least twice the model's size).
- **Time comes from the host.** The grant's end is judged on the TD's wall
  clock, which the host sets, and nothing compares it with the owner's clock
  (see "Whose clock").
- TDX only. With an SEV-SNP reference the relay refuses `--model-key` before
  anything is sent (89), and its check of a recipient on an SEV-SNP session
  refuses (88). The binding is 64 bytes, the size of an SNP report's
  `REPORT_DATA`, so the same hash can go there unchanged; the relay's check of
  the recipient report (same chip, same launch measurement as the document's
  report) is not written.
- The policy pins images by their registers; who built the image, and what it
  does with the key once it holds it, is what those registers stand for. The
  image reference is signed by x0ne's archive key: an owner who does not
  trust x0ne must pin an image they rebuilt themselves.

## References

1. Trustee, KBS attestation protocol (RCAR), `tee-pubkey` in the evidence.
   <https://github.com/confidential-containers/trustee/blob/main/kbs/docs/kbs_attestation_protocol.md>
2. RFC 9180, Hybrid Public Key Encryption, 2022. <https://www.rfc-editor.org/rfc/rfc9180>
3. Microsoft, Azure transparent KMS. <https://github.com/microsoft/azure-transparent-kms>
4. Tinfoil, Encrypted HTTP Body Protocol. <https://docs.tinfoil.sh/resources/ehbp>
5. Apple, Private Cloud Compute, 2024-06-10. <https://security.apple.com/blog/private-cloud-compute/>
6. Anthropic, Confidential inference via trusted virtual machines, 2025-06-18. <https://www.anthropic.com/research/confidential-inference-trusted-vms>
7. Tinfoil, Attestation architecture. <https://docs.tinfoil.sh/verification/attestation-architecture>
8. Privatemode, Architecture overview. <https://docs.privatemode.ai/architecture/overview>
9. Chutes, Private AI inference, 2026-05-07. <https://chutes.ai/news/private-ai-inference-verifiable-confidential-llm-serving>
