# The attested channel to a cloud GPU

> **Status: does not start from this source tree.** The relay built here
> pins no archive key, so it refuses every image reference and exits 99
> (measured on 2026-10-24); with the key it would refuse them with 100, as
> it pins no transparency log. No `x0ne-cvm` image is published, and
> [`x0ne-engine cloud`](cloud.md#building) needs one built with
> [x0ne OS](https://x0ne.co/x0ne-os/). The runs on real hardware used a
> relay from before the reference signature check (2026-10-14, 2026-10-17)
> or one that pinned the TEST archive key (2026-10-18).

## How a session is proved

`x0ne-engine cloud` ([`docs/cloud.md`](cloud.md)) rents a
confidential GPU VM, starts `x0ne-infer --cloud-config` inside it and runs
`x0ne-engine-proxy` on your machine; it is tested against a fake Compute
Engine and ran once against Google Cloud, on 2026-10-17 ([hardware runs](hardware-runs.md);
the 2026-10-14 run used `scripts/e2e-cloud.sh` and `gcloud`). One model and a chat in the terminal,
with the same quote, "yes" and deletion at the end:

```sh
x0ne-engine cloud run qwen3:0.6b     # served in cloud mode
x0ne-engine cloud run qwen3:8b --max-run 30m   # sharded, served on its pinned digest
x0ne-engine cloud run qwen3:32b --max-run 45m  # 17 shards, 61.0 GiB
```

A cloud session serves a catalog model whose served-model digest the catalog
pins: `qwen3:0.6b`, `qwen3:1.7b`, `qwen3:8b`, `qwen3:14b` and `qwen3:32b`
in this version, Qwen3 only ([docs/models.md](models.md)). Any other is
refused before the quote (exit 2).
A sharded model is checked file by file against the catalog, its shards are
mapped as published (no merge), the index must agree with them (exit 72), and
it is served only if the
[served-model digest](cloud.md#the-served-model-digest) of its files is
the pinned one (exit 77 otherwise). Its download and checks lengthen the boot
timeout: `qwen3:8b` asks for at least 24 min of `--max-run`, `qwen3:32b` 45
min. A token `up` cannot renew (`--auth paste`, the default, or `stdin`)
ends the session 5 min before it expires, so that the VM is deleted while
the token is still valid: a token with 59 min left cuts the default 60 min to
54, `up` says so before the quote, and a token too short for the model's boot
is refused (exit 91). At `[5/5]`, `up` prints the context the TD computed for
its GPU. Of the sharded models, `qwen3:32b` was served on an H100 on
2026-10-18 by a TD at 0.2.0-beta.7, its 17 shards as published, and
`qwen3:8b` on 2026-10-17 by a TD at 0.2.0-beta.5, before shards were served
as published ([hardware runs](hardware-runs.md)). No recorded run served `qwen3:1.7b`
or `qwen3:14b` on an H100.

The attested channel is the two ends and the proof between them. Its tests
run against a fake TD on the loopback (`src/cloud_e2e_tests.rs`): the real
relay, the real server code, synthetic quotes and tokens under test roots,
and one real Google endorsement (`tests/VENDORED.md`). **It has run three
times on real hardware**, each time on one GCP TDX + H100 spot VM ([hardware runs](hardware-runs.md)): 0.2.0-beta.2 on 2026-10-14
([`x0ne-infer/RUN_2026-10-14_beta2.md`](../x0ne-infer/RUN_2026-10-14_beta2.md)),
and 0.2.0-beta.5 on 2026-10-17, with a TD `x0ne-infer` built
from pre-release 0.2.0-beta.5 sources, which carry the TD side of 0.2.0-beta.3 (bounded TLS
handshakes, attestation before the model pull); no test on that VM aimed at
those two; and 0.2.0-beta.7 on 2026-10-18, whose relay checked the
reference's signature under the TEST archive key. No relay that pins the
release archive key has run, on real hardware or anywhere.

When it starts, before it writes a config or contacts anything, the relay
loads the image reference (`reference.rs`) and checks its signature by
the pinned x0ne archive key before it parses the file (`bin/proxy.rs`,
`reference::load`; 99, and a build made before the signing ceremony pins no
key and refuses every reference, see
[docs/cloud.md](cloud.md#the-image-reference)).

Then, before anything but `GET /v2/attestation` and `GET
/v2/attestation/session` reaches the TD, the relay checks on YOUR machine,
in this order (`remote_verify::verify_document`):

1. the TD's config bytes are the ones this session generated — your client
   key's SPKI, a fresh nonce, the expiry, the model, the product (exit 88);
2. the TDX quote chains to the pinned Intel root, and its TD attributes
   carry no debug, profiling or migration bit, nor a reserved bit that must be 0,
   and Intel PCS says, now, that the platform and the QE are `UpToDate` or
   `SWHardeningNeeded` (`tcb::TcbStatus::accepted`), issued at most 24 hours
   ago (`tcb::TcbPolicy`, 76): under the `transparent` TCB policy, the
   default, in Intel's `standard` tcbInfo and QE identity (evaluation 20 or
   later), with Intel's latest ones, of TCB evaluation 22 or later
   (`update=early`), read and named in the verdict (the platform's TCB
   date, its status, the advisories Intel lists against it), not refused
   on; under `strict` (`--tcb-policy strict`), in the latest ones; the
   PCS `Date` header must agree with your clock within 300 s (87, 76); the
   PCK certificate is not on Intel's PCK CRL, fetched now, signed by the
   PCK's issuer, inside its thisUpdate to nextUpdate window and issued at most
   24 hours ago (65 if listed, 76 if the CRL is missing or unusable);
3. `report_data` is SHA-512 over the session key, the TLS key THIS connection
   shook hands with and the config hash (`binding.rs`, v2): a quote from
   another TD, another connection or another config is refused (88);
4. `MRTD` is named by a launch endorsement Google signed for that firmware,
   fetched from Google's public bucket and verified under the pinned
   `GCE-cc-tcb-root_1`, with the machine type's RAM size (65); the firmware
   binary that endorsement names is fetched from the same bucket, its
   SHA-384 must be the signed one and the MRTD it measures the quote's (65;
   76 if the bucket cannot be read);
5. the CCEL event log replays to the quote's RTMR0..3; RTMR1 holds the
   image disk's GPT once, then the reference's boot applications (the UKI,
   then the kernel it loads) once each, in that order, and RTMR2 the kernel,
   initrd and command line of the signed image reference loaded at start-up
   (`reference.rs`, schema `x0ne-engine/reference/v2`: the proxy refuses
   a v1 one at start, and one naming an image below `MIN_IMAGE_VERSION`,
   89); RTMR3 is zero; every RTMR0 event is the reference's
   (`rtmr0.events_sha384`), the firmware's configuration volume the endorsed
   binary's (65). The reference must pin RTMR0: the proxy refuses one whose
   `rtmr0.policy` is `report` at start (104), since 2026-10-31; until then
   RTMR0 was replayed and reported under `report`;
6. with the GPU product, each GPU's NRAS token answers this session's nonce,
   is signed under one of NVIDIA's pinned intermediates, names the GPU the TD read
   over NVML and says confidential computing is on; its three certificate
   chains are `valid` with OCSP `good`, `secboot` is true and `dbgstat`
   disabled; the overall token's `submods` names each GPU token by its
   SHA-256, and nothing else (75);
7. the model is the one you asked for, with the catalog's files (77);
8. the session quote of THIS TLS connection (`GET /v2/attestation/session`,
   binding v3): same TD and platform as the document's quote, and
   `report_data` = SHA-512 over the connection's TLS exporter (RFC 8446
   §7.5, label `EXPORTER-x0ne-infer-attest/v3`, context = the config's
   nonce), the session key, the TLS key and the config hash. A party that
   holds the TD's TLS private key and relays the session runs two
   handshakes, so two exporters: refused (88). With the GPU product, the
   same answer carries that connection's GPU evidence (schema
   `x0ne-infer/session-attestation/v4`): NRAS tokens checked as in 6, for
   the nonce SHA-256 over `x0ne-infer/v4/gpu`, the connection's exporter
   and SHA-512 of the quote's header and TD report body
   (`binding::session_gpu_nonce`), naming the GPUs the document attested;
   tokens taken for another connection, for a TD on another machine or for
   the boot document are refused (75). Every later connection is quoted and
   checked the same way before its first request; a failed check stops the
   session.

Only then does `POST /v2/key` release the session key (for a model owner's
sealed weights: the owner's policy, a recipient quote over a fresh nonce and
`POST /v2/model-key`, [docs/model-key.md](model-key.md)); the TD loads the
model, and `/v1/chat/completions` opens on `127.0.0.1:8478` (Bearer in
`x0ne-engine/local.key`, mode 0600, under `$XDG_RUNTIME_DIR`, else
`/run/user/<uid>`, else `<temp dir>/x0ne-engine-<uid>`, as on macOS; each
directory is created 0700, or must be a directory of the caller's uid, not
a symbolic link, that no other account can write, the key's own 0700:
otherwise the relay stops with 98 before writing the key; `Host` limited
to the loopback). The relay verifies every answer's receipt against the verdict's
session key and model digest (`x0ne.receipt_verified`), and answers `tools`
with a 400. A whole answer is returned once its receipt verifies. A streamed
one (`stream: true`) passes its text on as it comes, before its receipt
exists: the final chunk, which carries `x0ne` with `receipt_verified: true`,
and `[DONE]` come only once the receipt verifies over the text streamed.
Otherwise an error event (`receipt_verification_failed`, `stream_cut`…)
takes their place, then the body ends as a whole one does: the event, not
the end, says the text above it is not verified. A client that keeps that
text keeps unverified text, and one that ignores the event sees a normal
end (curl exits 0). Open WebUI 0.12.3 was measured on streams that end in
`generation_failed` (relay and simulated TD on the loopback, 2026-10-22).
When the relay still ended such a body in error, Open WebUI showed the
partial text followed by its own `TransferEncodingError`, not the event's
message, and stored the answer as empty text with that error. With the body
ended as a whole one is, it shows the event's message under the partial
text and keeps both in the conversation. For text verified before it is
shown, ask with `stream: false`. `x0ne-engine-proxy chat` streams only to a
terminal; with its output redirected it asks for whole answers.

The relay's errors say whether a retry can help. A 503 with `Retry-After`
(the TD's full queue, its quote limits) can be retried after that delay.
Every other 5xx (a failed generation, a lost or stopped session) carries
`x-should-retry: false`. openai-python (3.36.0 and 1.109.1, read in its
source) obeys that header before the status; without it, it sent a whole
answer whose generation failed twice more, three generations for one
request.

The TD side, `x0ne-infer --cloud-config <file>`, serves TLS 1.3 only with the
client certificate pinned by the config's SPKI — nothing without it, the
attestation route included — and refuses `X0NE_INFER_KEY_HEX` and the
operator pins (95): in cloud mode the pins are the client's. The v1
routes and `fn main` are untouched.

Refusals, each tested with the fake TD receiving neither a question nor a
key: MRTD without endorsement, false endorsement signature, RTMR1, RTMR2 or
RTMR3 drift, CCEL replay ≠ quote, `report_data` not bound, TLS key ≠ this
connection, config or nonce ≠, a relay holding the TD's TLS key, a session
quote without the exporter or from another connection, GPU token for another nonce or key, no GPU
attested, a connection's GPU tokens from an earlier connection, from a TD on another machine or
from the boot document, or naming another GPU than the document's, a GPU token the overall token's `submods` does not name, model files or name ≠, Intel PCS or Google unreachable, clock off;
after the key, a model whose digest drifts from the document is refused by
the TD itself (77) before any question. With sealed weights: an image, TCB,
GPU or date the owner's policy does not accept (102), a recipient quote that
answers an earlier nonce or names another key (88), a wrapped key posted to
another TD, a tampered sealed file (77 before the evidence, 72 after the
release) and other weights under the owner's key (77).

Downgrade and diversion, tested the same way: the channel is TLS 1.3 with
the two AES-GCM suites only (`tls::CIPHER_SUITES`), so a TLS 1.2 ServerHello
or ClientHello and a peer that only speaks ChaCha20 get no session; a
document or a session quote under any other schema version is refused (88);
a TD that serves the document but no session quote gets no key, with no
fallback to the v2 binding; a session quote from the same image on another
platform (another PCK under the same Intel root) or from a TD whose
registers differ is refused (88); a later connection steered to a second
TD booted with the same config dies at the handshake, the server key being
pinned, before any route of that TD runs.

The TD process is not dumpable and has no core file from the first line of
`main` (`harden::process`): processes of the same user can neither ptrace it
nor open `/proc/<pid>/mem`, measured from outside on the real binary
(`tests/process_hardening.rs`). The key the relay releases (`POST /v2/key`) and the TLS private key's
DER sit in `memfd_secret` memory where the kernel offers it (out of the
kernel's direct map: even `/proc/self/mem` cannot read it), else in
`mlock`ed pages kept out of core dumps and forks, and are zeroed before the
pages are returned (`harden::SecretBytes`). Other copies are ordinary
memory: the relay's request body (`remote_verify.rs`, `post_key`), the TD's
`axum` body and the 32-byte array it parses the key into, none zeroed; the
heap copy the model loader gets (`ReleasedKey`, zeroized when dropped once
the weights are loaded); the copies ring and rustls make when they parse the
TLS key; and the per-boot signing key that signs receipts, ordinary heap.
Nothing a process does stands against root or the kernel inside the guest.

One connection is one full TLS handshake: session resumption and 0-RTT are
off at both ends (a resumption would skip the certificate checks), so every
connection is quoted. The TD takes one quote at a time; behind it,
`--max-queue` session requests wait at most 10 s and the next get 503;
each connection gets one session quote (then 429), and a global bucket
allows `--max-queue + 2` quotes at once, refilled at one per second (429
with `Retry-After`). A failed quote answers a fixed message, the cause
stays in the TD's log. The relay opens one connection per local request in
flight when none is free, so a burst of local requests can meet these
limits: the request the TD turned away is a 503 with `Retry-After: 1`, and
the session goes on. A failed quote still loses the session.

Before mutual TLS has identified anyone, the TD bounds the handshakes: each
has 10 s to finish, and at most `--max-handshakes` (256) are in flight. One
more drops the oldest handshake of the source address (an IPv6 /64) that
holds the most, so silent sockets from a few addresses neither run the TD
out of descriptors nor push out the relay's handshake. The GCP firewall
rule still admits any address on port 443.

Cost of the per-connection quote, measured on 2026-10-13 on synthetic
evidence (release build, loopback, `session_quote_cost`): the client's check
≈ 0.45 ms, a new checked connection 1.4 to 2.1 ms against 0.31 ms for a new
connection and one GET. The fake TD's quote is a software signature
(≈ 0.11 ms). On the real TD (2026-10-14), one session quote took a median
206 ms seen from the client (min 168, max 308, n=10), one network round trip
included (TCP connect alone 140 to 263 ms), and a new checked connection a
median 698 ms (n=5); the in-TD quote generation time is not isolated, and it
is the term that matters. The relay keeps its connections open, so the cost
is paid per connection, not per request.

No flag and no variable sets an
anchor or skips a check (`tests/cloud_source_sweep.rs`).

What it does not prove is printed with every verdict
(`remote_verify::LIMITATIONS`): software in your session can read the local
key; Google's firmware is the binary Google signed, measured here, its
source unpublished, and RTMR0 is pinned to host values seen on one boot,
not audited; the NRAS
signing chain's revocation is judged from the OCSP responses the document staples, none stapled yet, the
verdict saying so (the PCK CRL and Intel's root CA CRL are read live); CPU and GPU are bound by a nonce
derived from each connection's TLS exporter and quote, made by the TD's software, not by hardware, so the
link falls with the CPU TEE (TEE.fail); whoever
holds the archive signing subkey can sign any image reference, and a
revocation reaches you only with a new build; the relay does not open the
prompt commitment; the model behaves as the model behaves.

The path is not end-to-end confidential: [docs/cloud.md](cloud.md#what-not-end-to-end-confidential-covers)
names each place where the text is in plaintext and who can read it there,
and `x0ne-infer/THREAT_MODEL.md` lists the published attacks on this
platform ("Known attacks as of 2026-10-22") and the promises this project
does not make.
