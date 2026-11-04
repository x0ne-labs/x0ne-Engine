# Limitations and scope

## Honesty section — stated limitations

The verdict's `verifier` string carries the verifier's own limitations,
verbatim, so a client reads them from the receipt rather than from
documentation that can drift. It does not carry all of the list below. Today:

- The TD attributes are judged by the groups of Intel's TDX Module ABI
  (348551-008, table 3.23), server and client side: a debug or profiling
  bit (15:0), MIGRATABLE, or a reserved bit Intel says must be 0 (26:23,
  55:32, 61:56) refuses; SEPT_VE_DISABLE, the SEC bits Intel marks positive
  (22:18 among them, which Intel lets a verifier accept) and the two defined
  OTHER bits, TPA and PERFMON, are accepted. A TDX module that defines one of
  the reserved bits is refused until a release accepts it. XFAM, the CPU
  features the host grants the TD, is neither parsed nor judged.
- Collateral is fetched at process start, then re-read every hour against the
  same quote (`tcb_standing`, since 2026-10-23): a verdict against the
  platform (a status Intel does not accept, no matching level, a TDX module
  version with no identity or no level in the tcbInfo, a revoked PCK or PCK
  CA, an evaluation below the floor) stops the answers for good, a 503
  naming it before the model runs, until a restart re-attests; collateral not
  fetched or not readable leaves the last verdict standing for at most the
  collateral age limit (24 hours), then the answers stop too. The statuses a
  receipt carries are as of the last verdict, not re-verified per request.
- PCK revocation: since 2026-10-18 the PCK certificate is checked against
  Intel's PCK CRL (PCS v4 `pckcrl`, fetched with the collateral, by the TD at
  start-up and by the client at each verdict). A listed PCK refuses (65); a
  CRL that is missing, outside its thisUpdate to nextUpdate window, or not
  signed by the PCK's own issuer refuses (76). `x0ne-infer` outside cloud
  mode exits 76 for both, with the rest of the live collateral. Since
  2026-10-23 the root CA CRL is read too (`tcb::verify_root_crl`, from the
  root certificate's own CRL distribution point,
  `certificates.trustedservices.intel.com/IntelSGXRootCA.der`; `--root-crl-url`
  in `x0ne-infer`): the PCK CA that issued the PCK and Intel's TCB signing
  certificate listed refuse (65); a root CRL missing, outside its window, not
  the root's or carrying a critical extension refuses (76). Measured on
  2026-10-23: CRL number 1, issued 2026-02-26 until 2027-02-26, no revoked
  certificate; PCS v4 `/rootcacrl` answered 404. Issued for a year, it is
  judged by its own window and not by the 24-hour age limit, so a cache can
  serve the root CRL of the day before a revocation for the rest of its year;
  Intel's root has revoked nothing since 2018 (`tests/root_crl_live.rs`).
  Intel's model here is the
  CRL, not OCSP: measured on 2026-08-22, the Intel SGX TCB Signing
  certificate carries a CRL distribution point and no OCSP responder at all.
  TCB recovery is carried by the live tcbInfo levels.
- TCB policy (`tcb::TcbPolicy`: a floor since 2026-10-22, two modes since
  2026-10-28). Intel PCS publishes two tcbInfo answers for a platform:
  `standard`, its default, which follows a TCB recovery "commonly 12 months
  after public disclosure" (Intel PCS API), and `early` (`update=early`),
  Intel's latest. Measured on 2026-10-28 for FMSPC `00806f050000`, the GCE
  a3-highgpu hosts: `standard` was TCB evaluation 20 and rated the platform
  level of 2025-08-13 UpToDate; `early` was evaluation 22 and rated the
  level of 2026-02-11 OutOfDate under INTEL-DOC-10003, INTEL-SA-01439 and
  INTEL-SA-01442; the same SVNs match both levels (`tcb.rs`, on Intel's
  signed answers of that day).
  - `transparent`, the default of the TD and of the relay. The platform and
    the QE must be UpToDate or SWHardeningNeeded under Intel's `standard`
    tcbInfo and QE identity (evaluation 20 or later); any other status
    `standard` gives (OutOfDate, Revoked, a configuration-needed one)
    refuses (76), and so does any collateral issued more than 24 hours
    before the verdict. The TD and the relay also
    fetch `early` and rate the platform, its TDX module and its QE under it
    (evaluation 22 or later, the same age limit), without refusing on what
    it says. What it says (the platform's TCB date, its status under
    evaluation 22, the advisory IDs Intel lists against its level, its
    module and its QE) is written in the attestation document, the relay's
    verdict (its `tcb` field and the last line of its summary), every signed
    receipt (`tcb`, schema `x0ne-infer/tcb-disclosure/v1`), `/x0ne/v1/status`
    and the client's VERIFIED line: "VERIFIED. Platform TCB 2026-02-11,
    OutOfDate under Intel's TCB evaluation 22; Intel advisories not fixed:
    INTEL-DOC-10003, INTEL-SA-01439, INTEL-SA-01442 (transparent TCB
    policy)." This mode guarantees transparency, not protection: a platform
    Intel's latest collateral rates OutOfDate is served, and the advisories
    it names are not fixed on that platform. When `early` cannot be read, is
    below evaluation 22 or is older than 24 hours, the line says so, names
    no advisory, and does not say the platform is up to date. The TD
    re-reads both every hour; a re-check that cannot read `early` keeps the
    last view and says when it was read.
  - `strict`, the client's choice (`x0ne-engine-proxy --tcb-policy strict`,
    `x0ne-engine cloud up` or `run --tcb-policy strict`;
    `--min-tcb-evaluation N` implies it, and with `--tcb-policy transparent`
    is a usage error, 2). The relay judges the TD on `early`, evaluation 22
    or later, as every verdict did from 2026-10-22 to 2026-10-28: a
    platform, module or QE that is not UpToDate or SWHardeningNeeded there
    is refused (76) and receives nothing. The TD itself serves under
    `transparent`, so a strict relay also reads each receipt's disclosure
    and ends the session (76) when the TD's latest view is missing, not
    accepted, below its floor or older than its age limit. The GCE hosts of
    2026-10-28 above are refused under `strict` and served, named, under
    `transparent`.
  - Neither mode is looser than the build. The relay refuses
    `--min-tcb-evaluation` below 22 and `--max-collateral-age` above 86400
    as a usage error (2), and names the policy it applies (mode, floor, age)
    in its `config` event and on the verdict's CPU line. `x0ne-engine cloud
    up` and `run` take the same options, refuse a looser value before any
    quote, and refuse a relay whose config names a looser policy than asked
    or none (95, nothing created); a config without a mode is a relay from
    before 2026-10-28, which is strict. The other `cloud` commands, which
    start no relay, refuse the options (2). A looser floor takes a new
    build. The SDK's caller-sealed path (`docs/sdk-e2e.md`) applies the same
    two modes with the same function (`tcb::judge_live`), on both answers
    the caller fetched, `transparent` by default and `strict` on the
    caller's request, and returns the same disclosure with the verified key;
    under `strict` it holds every receipt to Intel's latest view, as the
    relay does. Its floor and age limit are this build's: a caller can
    shorten the age, neither lower the floor nor lengthen the age (since
    2026-10-28; until then the SDK judged `early` alone, as `strict`).
  - Neither mode sees a vulnerability Intel has not yet put in its
    collateral, or a recovery newer than the day's `early` answer.
    A PCS cache in front of the TD (a PCCS behind `x0ne-infer --pcs-url`)
    that keeps collateral longer than 24 hours is refused (76) at every
    verdict: availability, not a bypass. The GCE a3-highgpu-1g host of
    2026-10-18 is UpToDate under evaluation 22
  (`tests/tcb_policy_live.rs`). On 2026-10-25, four VMs of that type in
  us-central1-a, created between 15:20 and 18:12 UTC, landed once on a host
  rated UpToDate and three times on a host of FMSPC `00806f050000` rated
  OutOfDate (level dated 2026-02-11): the TD refused those three (76) and
  served nothing, under the policy of that day, which was `strict`'s. Four
  VMs on one day: no share of GCE hosts follows from them. The relay sees that refusal only as a TD that does not answer,
  until its boot deadline (39 minutes after
  creation for `qwen3:32b`). `x0ne-engine cloud up` also reads the VM's
  serial console while it waits: a status block that shows a final refusal
  of x0ne-infer after two restarts ends the wait, and the VM is deleted
  (exit 97; the list of final refusals and why is in `docs/cloud.md`, step
  8). On the consoles of 2026-10-25, that block is the one at 61 s of
  uptime. On the two real `up` of that day that reached it (the client that
  reads the console, `docs/hardware-runs.md`), it ended the wait 95 s after
  the VM was RUNNING, and the VM
  was gone about 200 s after its creation (estimated 0.40 USD each, against
  0.82 USD for the VM deleted by hand seven minutes after its creation).
  The console is unverified text: it can end the wait, nothing else. A
  console that cannot be read leaves the wait to the boot deadline, and the
  VM is billed meanwhile.
- NRAS chain anchored at NVIDIA's intermediate CAs 004 and 005 (NVIDIA
  publishes no root for them). Their own signature by CA 001 cannot be
  checked, so each is pinned as its whole certificate, validity window
  included; one NVIDIA re-issues is refused until it is pinned. Revocation
  of that chain: NVIDIA publishes both a CRL and an OCSP responder. Since
  2026-10-31 the relay, the TD's own gate and the SDK judge OCSP responses
  stapled with the token (`nras_ocsp.rs`, RFC 6960: signed by the issuer or
  by a responder it issued and marked for it, `thisUpdate` not after now,
  `nextUpdate` required as RFC 5019 asks and not before now, issued at most
  a day ago as Intel's collateral, the nonce echoed when one was sent),
  under a policy every verdict names: `required`, the default of a policy
  written without one, refuses anything short of a verified `good`;
  `transparent`, what production verifies with, refuses a verified
  `revoked` and names the rest. No TD staples a response yet, so every
  verdict says "no response stapled"; the intermediate's own status cannot
  be verified without CA 001, which NVIDIA does not publish; the CRL is not
  read. Measured 2026-10-31 on the vendored certificates: the intermediates
  name the responder `ocsp.ndis.nvidia.com`, the leaf under 004 names none,
  the leaf under 005 names NVIDIA's Vault PKI without a responder URL.
- The attestation this process serves on has a lifetime: the key-release
  lease, 24 hours. Past it the server refuses new requests and says so —
  restart to re-attest. Nothing is served on a stale proof, but nothing
  re-attests by itself either.
- The key broker is v0: the key is `X0NE_INFER_KEY_HEX`, set by whoever
  starts the process. It sits in the process environment from launch, before
  any proof. The server removes it from its environment at start-up,
  zeroizes the hex string and the parsed 32 bytes, and overwrites the value
  with zeros in the exec-time block the kernel shows as `/proc/<pid>/environ`
  (measured 2026-10-22, `keys.rs` test). Until that first step of `main`,
  and in the environment of the launcher (a shell, a unit file), the value
  is readable. The release *decision* (proof,
  freshness, lease) runs before the server reads it, but whoever sets the
  variable holds the key. In cloud mode no such variable exists: the session
  key comes from the relay, and a model owner's key comes from the owner's
  relay wrapped to the attested TD ([docs/model-key.md](model-key.md),
  simulated TD only, never run on hardware).
  Not in the verifier string.
- **The published measurements (`measurements/*.json`) are measured once
  and not signed** — see
  [Pinning an image](pinning.md). The published file is the default pin and a release
  refuses without one (exit 87). `measurements/0.2.0-beta.7.json` names the
  RTMR1 and RTMR2 of one GCE boot of x0ne-cvm 0.2.4 (2026-10-25), equal to
  what the same predictor computed for that image, with TEST provenance: a
  release build refuses it (exit 87). `measurements/0.2.0-beta.6.json` names
  no register; `measurements/0.2.0-beta.5.json` names the
  RTMR1 and RTMR2 of one GCE boot of x0ne-cvm 0.2 (2026-10-17), equal to what
  the x0ne OS measurement predictor computes for that image. Their `signature` is null, so a client
  trusts them as far as it trusts the binary it runs. A debug build
  whose file names no register serves anyway and says in its `verifier`
  string that nothing is pinned. An operator who
  controls the guest image controls what runs inside the enclave until a
  client pins it.
- The browser console (`GET /`) verifies receipts only — WebCrypto has
  no DCAP verifier. Its "verdict" pill and its environment label are the
  server's own statement, and the session key it checks signatures against
  is the one the server published, not one the page verified against a
  quote. The page itself is served by the server under test, so it cannot
  protect you from that server.
- The GPU half is bound to this machine by comparing each per-GPU token's
  `ueid` to the serial of the attestation certificate read from the local
  GPU over NVML. That derivation is taken from NVIDIA's nvtrust SDK
  (`cc_admin.py`). It has met production NRAS tokens in cloud mode on
  2026-10-14, 2026-10-17 and 2026-10-18: the TD (`attest_v2`) and the relay
  both apply it and refuse on a mismatch, and each time the TD served and the
  relay's verdict was VERIFIED. If
  NRAS derives `ueid` otherwise, the server refuses genuine hardware and prints
  both values — it fails closed, not open. The serial is read through the
  guest's NVIDIA driver, so this binds the token to what the driver reports,
  inside a guest the TDX half attests but whose image is not yet pinned (see
  above).
- Time comes from the host: every freshness window (collateral, token
  expiry, gate staleness) rests on a clock the operator can move. On the GPU
  path a rollback of more than five minutes is now refused, because the NRAS
  token answers our nonce and carries the time NVIDIA signed it — the one
  timestamp here the host does not choose. The 24 h key lease, since 2026-10-11, also
  runs on the guest's monotonic clock and ends if the wall clock goes back
  more than 60 s; a host that pauses the guest pauses that clock too. A model
  owner's grant ends on the guest's wall clock as well: held back, the TD
  serves past the grant's `not_after_unix` by as much, within that 24 h
  ([docs/model-key.md](model-key.md)).
- Outside cloud mode the transport is not attested. That server speaks
  plain HTTP; the scripts here reach it over ssh to the guest's `sshd`, which
  runs in a guest image that is not pinned. Cloud mode has an attested
  channel ([attested-channel.md](attested-channel.md)).
- The relay verifies each receipt's origin and output (signature, attested
  session key, model digest, output commitment), not its correspondence with
  the request it answers: no prompt commitment is recomputed from the sent
  request and no per-request nonce is checked (THREAT_MODEL, the row on a
  TD answering with another model or key).
- The PCK and collateral certificate chains are verified to the pinned root
  by signatures and validity dates. Since 2026-10-27 the PCK chain must be
  exactly the PCK, its CA and the root, each naming the next as issuer, the
  CA and the root marked CA by basicConstraints and the PCK not; each
  collateral chain exactly the TCB Signing certificate and the root, the same
  checks applied. The keyUsage of the PCK's CA is not checked, so this is not
  a full RFC 5280 path validation.
- CUDA inference is built with `--features cuda` (`--device cuda`), and
  since 2026-11-04 it is the only way the server serves: `--device cpu` and
  `--attest cpu` exit 90. In cloud mode, the engine before x0ne Core ran on
  an H100 in CC mode on 2026-10-17, 2026-10-18, 2026-10-22, 2026-10-25,
  2026-10-28, 2026-10-30, 2026-10-31 and 2026-11-01
  ([hardware runs](hardware-runs.md)); the responses are not kept in this
  repository, the speed rows of five of those runs are
  ([benchmarks](benchmarks.md)). x0ne Core alone, its own bench and not
  this Engine, ran on an H100 in CC mode in a TDX VM on 2026-11-03, from a
  bench image without NVIDIA attestation; its rows are in
  [`bench/results/2026-11-03-h100-tdx-cc-core/`](../bench/results/2026-11-03-h100-tdx-cc-core/).
  This Engine on x0ne Core ran there on 2026-11-04, a TEST measurement; its
  rows are in
  [`bench/results/2026-11-04-h100-tdx-cc-core-e2e/`](../bench/results/2026-11-04-h100-tdx-cc-core-e2e/).
  A run on an H100 in CC mode on 2026-08-21 is reported but not recorded
  here.
- The cloud VM's tcp:443 admits one source by default: this machine's public
  IPv4 address, as Google's name server answers it, as a /32 (since
  2026-10-30; before, `0.0.0.0/0`), or the ranges `--allow-from` names.
  Machines behind the same NAT share that address and reach the port, where
  mutual TLS refuses them. A machine whose address changes during the session
  loses the session: the local API answers 502, then 503, and the VM runs,
  billed, until `--max-run`, Ctrl-C or `down`
  ([cloud.md](cloud.md#the-firewall-rule)).
- NRAS is called without a key by default. `--nras-key-file` (cloud `up` and
  `run`) hands an NVIDIA API key to the VM in the metadata item
  `x0ne-nras-key`, and `x0ne-infer --nras-key-file` sends it as
  `Authorization: Bearer`, but the CVM image of x0ne-OS ignores that item
  for now: with it, NRAS is still called without the key. The key is readable
  by Google and by anyone with access to the project while the VM lives
  ([cloud.md](cloud.md#the-nras-key)). Since 2026-10-30.
- A streamed answer's text (`stream: true`) is passed on before its receipt
  is verified: only the final chunk and `[DONE]` wait for the receipt, and an
  error event in their place says the text above it is not verified. For
  text verified before it is shown, ask with `stream: false`
  ([attested-channel.md](attested-channel.md)).
- A caller-sealed stream ([sdk-e2e.md](sdk-e2e.md#streaming), since
  2026-10-30) hides its text from the carrier and the host, not its shape:
  when it was asked, its question's size bucket, its count of chunks (a
  power of two from 16, so the answer's duration to a factor of two, and
  its length to a factor of two when the text sets the count), a reader
  that leaves. The model's turn, and the queue place a host can probe
  (503 when full), end at that same count of ticks after the request, not
  a Padme size after the generation: the price is the model held up to
  twice a generation's time. Of two short streams asked together, the
  second's generation started at 1.65 s instead of 1.05 s and its answer
  took 32 chunks (3.2 s) instead of 16 (1.6 s), measured on 2026-10-30 in
  the lab with a scripted model at T = 100 ms. The deltas a caller shows
  before `finish` are not verified. Through the relay (since 2026-10-31;
  one stream on hardware on 2026-11-01, [hardware-runs.md](hardware-runs.md))
  the host also reads the frames' count, sizes and timing on the relay's
  loopback, the same as on the TD's TLS.

### Conversations kept between turns

Since 2026-10-30 a request may ask the TD to keep its conversation's keys
and values for the next turn (`x0ne.conversation`, [api.md](api.md),
[session-kv-cache.md](session-kv-cache.md)). What that leaves visible, and
accepted as such:

- **The host learns which turns continued a conversation.** A continued
  turn computes its new tokens only, a turn from token 0 its whole history;
  the padding to a Padme number of 100 ms ticks rounds that difference, it
  does not hide it: on the CPU with Qwen3-0.6B, a second turn of 16 390
  tokens took 9.9 to 16.0 s continued and 1 716 to 1 952 s from token 0
  ([measurements](session-kv-cache.md#measurements)). From that the host also learns, to
  within a size, when a conversation was evicted. Holding a continued turn
  to the time of a cold one would cancel what the cache saves; the Engine
  does not.
- **Callers of one TD learn of each other's activity.** One conversation is
  resident at a time: a request of another caller evicts it, and the next
  turn comes back as `turn: 1`, computed from token 0. That tells its caller
  that another request ran in between, not what it asked. No request can
  have the TD compare its tokens with another conversation's: the lookup is
  by handle, never by content.
- **The holder of a live handle can test a guess** about that
  conversation's history, once per turn, from the exact `reused_tokens`
  count. The handle is a bearer secret: on the relay route the relay sees
  it, on the sealed route only the caller does. A handle serves one turn and
  is bound to the route that issued it.
- **What zeroing covers**: every byte x0ne Core derives from a request in
  its GPU memory (KV cache, activations, logits, scratch, cuBLAS's
  workspace) and its host buffers, at the end of every request
  (`x0ne-core/runtime/src/arena.rs`, `Class::ClientData`;
  `x0ne-core/runtime/src/engine.rs`: a session is zeroed when it finishes
  and when it is dropped, on an error or a panic too). x0ne Core keeps
  nothing of a request for the next one: no conversation, no cache. A
  failed zeroing poisons the engine, and no request is computed until a
  new zeroing succeeds. Not the copies of the text the tokenizer and the
  HTTP stack make. A process that ends or is killed zeroes nothing; its
  memory goes back to the TD's kernel.
- **The receipt's flag on such a turn is a constant of that path**, as the
  other three are: `kv_cache_shared_between_requests: true` is signed on
  every turn of a conversation, and the `context` block says how many
  tokens were reused, not a measurement that no other request read them.
- **Conversations on x0ne Core: none in this version.** Its backend keeps
  the default of `generate_turn` (`x0ne-infer/src/backend.rs`): a request
  that asks for a conversation gets a 400 naming `x0ne.conversation`, rather
  than an answer without its context, and every turn is computed from token
  0. What this section describes ran on the engine before x0ne Core, which
  only the tests build now ([session-kv-cache.md](session-kv-cache.md)).

## AMD SEV-SNP

Since 2026-10-28 the relay verifies an AMD SEV-SNP guest when its
`--reference` is an SEV-SNP reference (`x0ne-engine/snp-reference/v1`,
signed by the x0ne archive key and shown in the transparency log, as the
TDX reference is). A session pins one TEE: a TDX reference refuses SEV-SNP
evidence and an SEV-SNP reference refuses TDX evidence (exit 88), with no
fallback either way. The threat model, "AMD SEV-SNP", gives the detail.

Verified, at each verdict:

- the report's signature under a VCEK chained to the AMD ARK and ASK pinned
  in this build, or under a VLEK chained to the pinned ASVK when the
  reference names that cloud provider's cspId (none by default);
- AMD's CRL for the product, fetched from AMD's KDS and signed by the pinned
  ARK, read inside its window; KDS's clock within 300 s of this machine's;
- every TCB of the report (reported, current, committed, launch), component
  by component, against the default compiled from AMD's bulletins, raised by
  the reference's `tcb_minimum`, never lowered. AMD publishes no second
  channel; each bulletin's date is a constant of `snp.rs` (AMD-SB-3015,
  BadRAM, 2024-12-10, a day not re-read on AMD's page; AMD-SB-3020,
  2025-10-13; AMD-SB-3027, 2026-01-15). Under `transparent`, the relay's
  default, a bulletin is imposed once it is a year old and only named
  before; under `strict`, every bulletin is imposed, as before 2026-10-28.
  On 2026-10-28, `transparent` imposes AMD-SB-3015 and AMD-SB-3020 and
  names AMD-SB-3027 until 2027-01-15. Where AMD-SB-3027's own microcode
  floor is not in hand, AMD-SB-3020's stands for both, imposed. The `tcb`
  disclosure names, for each bulletin, what it asks of the CPU line,
  whether it is imposed and what the report misses of it; the verdict's
  last summary line names the imposed ones and what the report misses of
  the others;
- DEBUG, a migration agent and a VMPL other than 0 refused, whatever the
  reference says; SMT refused unless the reference writes `allowed`; any
  policy or platform bit, or reserved byte, this verifier does not know,
  refused;
- `MEASUREMENT` among the reference's predicted launch digests, and
  `HOST_DATA` equal to the reference's when it pins one;
- `REPORT_DATA` bound to the session as on TDX, and each connection's
  report bound to that connection's TLS exporter, the rest of it identical
  to the document's;
- with `--attest cpu+gpu`, the GPUs through NVIDIA's NRAS as on TDX, each
  connection's nonce derived from its exporter and the signed bytes of its
  SEV-SNP report.

Not verified:

- how the predicted launch digests were computed: outside this verifier,
  from the firmware, kernel, initrd, command line and vCPUs; the
  reference's signer vouches for them;
- what the guest loads after launch beyond what its launch digest covers:
  an SEV-SNP guest has no event log;
- a single chip key's revocation: AMD's CRL lists intermediates only;
- the mitigation vectors of a v5 report, and the guest-policy bits for
  ciphertext hiding, page swap, single socket, AES-256-XTS, RAPL and CXL;
- physical attacks on memory, as on TDX.

Measured on the guest report of a Scaleway Elastic Metal host of 2026-10-28
(CPUID model 0xA0 stepping 2, Bergamo or Siena; SNP 0x17, microcode 0x15,
`PLATFORM_INFO` 0x4, guest `POLICY` 0x30000; `snp.rs`, on the report and
AMD's VCEK and CRL): AMD's chain and signature verify; it is refused under
both modes by the default policy, its guest accepting SMT; with SMT allowed,
it is refused under both, below AMD-SB-3020 (SNP 0x17 < 0x1b, microcode
0x15 < 0x1b), and the alias check of AMD-SB-3015 is not complete.

Not built: the guest side. The TD of this version collects Intel TDX
evidence only, so no SEV-SNP machine has run this code. The GCP lifecycle
refuses an SEV-SNP session (its config event names no GCP image). Sealed
weights (`--model-key`, refused with an SEV-SNP reference) and the SDK's
caller-sealed path are TDX only. Azure's paravisor stays refused.

## Out of scope

Not built in this version, and promised nowhere in this repository: an
AMD SEV-SNP guest (the client side is above), more than one GPU, NVSwitch
and Blackwell attestation, an
attested transport outside cloud mode (the plain HTTP server, see the
threat model), a client-held key broker outside cloud mode (v0 reads an
environment variable), a guest image rebuilt by a third party and
measurements with release provenance. The last two are open in the threat
model. The others are not planned here.
