# x0ne-sdk: the caller checks the TD and seals to it — prototype

Status: prototype, 2026-10-22. Since 2026-10-26 the TD binary serves
`/v2/e2e/key` and `/v2/e2e/chat` ("The TD's routes" below). Since 2026-10-28
the relay carries them on its local endpoint without opening them ("Through
the relay"), the TD's key document carries NVIDIA's verdict on its GPUs, and
the SDK judges the key document as the relay judges its session: Intel's
collateral under the relay's TCB policy, the image reference admitted by the
relay's signature and log steps. Tested on synthetic TDX evidence, and run
through the relay on an H100 in a TD on 2026-10-28, 2026-10-30, 2026-10-31,
2026-11-01 and 2026-11-04 (on x0ne Core), under the TEST
archive key and log ("Through the relay", [hardware-runs.md](hardware-runs.md)).
Since 2026-10-30 a streamed
answer goes sealed too, as chunked OHTTP at a cadence the session config
fixes ("Streaming"); since 2026-10-31 the relay carries it frame by frame
and the browser ABI streams it ("Through the relay"); on 2026-11-01 one
sealed stream crossed the relay on hardware, from the SDK run in Node, not yet from a browser
tab. Since 2026-10-31 the SDK keeps the log's last verified checkpoint, as
the caller's memory, and holds every new one to it ("The log's memory"),
on the TEST log. Nothing here
changes what the README says: the product is not end-to-end confidential
today. This document says what the prototype does, what its tests attack,
and what is left before the claim can move.

## Why

Today the attested channel is the local relay's (`x0ne-engine cloud`): mutual
TLS 1.3 from the relay to the TD, a TDX quote bound to the TLS exporter
(binding v3), eight checks before any question or key is sent. A browser,
a Python script or a phone has no such relay and cannot read a TLS
exporter. Whoever terminates their TLS reads their text.

The 2026 pattern (Apple PCC, Brave with NEAR AI, Tinfoil's EHBP, Microsoft's
attested OHTTP, OpenPCC) is for the caller to encrypt to a public key the
attested VM holds, in HPKE (RFC 9180) framed as Oblivious HTTP (RFC 9458),
so that any carrier moves bytes it cannot read. Where offers differ is who
checks the attestation: Brave's browser "relies on the gateway for this
check"; Apple's device checks itself. For x0ne's claim to hold against x0ne,
the caller must check.

## What the prototype does

```
caller                                 carrier(s)                          TD
  nonce ───── GET /v2/e2e/key?nonce= ───────────────────────────────────▶
                                    (GPU session: GPU reports for gpu_nonce,
                                     quote, then NRAS asked by the TD)
  ◀── { key_config, session_pubkey, session_config, boot_nonce, ─────────
        tls_spki_sha256, quote(report_data = binding), ccel,
        GPU session: gpu_evidence_list, nras_answer, nras_jwks }
  ◀── GET /v2/e2e/collateral: Intel TCB info and QE identity (update=early
      and update=standard), PCK CRL, root CA CRL; Google's launch
      endorsement of the quote's MRTD and the firmware binary it names;
      NRAS's JWKS
  verify_key: reference admitted (archive key signature, log proof), then
      schema, key, quote, binding, tcb, revocation, measurement, gpu,
      session; the caller reads the session config (model, expires_at,
      the relay's key) the quote bound
  seal: BHTTP(POST /v1/chat/completions, Date, JSON) padded → OHTTP request
  ───── POST /v2/e2e/chat  message/ohttp-req ─────────────────────────────▶
                                                     open, replay guard, chat
  ◀──── message/ohttp-res ──────────────────────────────────────────────────
  open: padded BHTTP response; receipt checked under session_pubkey
```

- **Binding.** The TD's `report_data` is `SHA-512("x0ne-e2e/v2/ohttp-key" ‖
  0x00 ‖ len16(key_config) ‖ key_config ‖ client_nonce ‖ session_pubkey ‖
  boot_nonce ‖ tls_spki_sha256 ‖ config_sha256)`. The last three, with
  `session_pubkey`, are the TD's boot binding (`binding::BindingV2`, which
  its boot quote and the relay's session quotes bind): the nonce it drew at
  boot, SHA-256 of its TLS key, SHA-256 of the exact bytes of its session
  config (`x0ne-engine/cloud-config/v1`, the relay's). The key document
  carries those bytes (`session_config_b64`), the boot nonce and the TLS
  key's digest; the SDK hashes the bytes as they came, never a parse of
  them re-serialized. A key next to a TD's quote that does not bind it is
  refused (`binding`); so is a quote for another caller's nonce, a receipt
  key the quote did not bind, and, since 2026-10-30, a session config, a
  boot nonce or a TLS key other than the ones the quote bound (another
  relay key, another model, a later end, the same config in other bytes:
  tested). The label shares no prefix with the relay's `report_data`
  contexts (tested).
- **Version 2** (2026-10-30), a clean break: the labels went from
  `x0ne-e2e/v1/` to `x0ne-e2e/v2/` and the document's schema to
  `x0ne-e2e/key/v2`. A v1 document is refused at `schema` (88), whatever it
  carries: its quote binds no session config. An SDK before v2 refuses a v2
  document by the name of its first new field (`unknown field
  session_config_b64`, tested; it is `deny_unknown_fields`). A TD and the
  SDK that verifies it go together; `gpu_nonce` keeps its v1 label, as the
  GPUs answer it before the quote.
- **GPU binding** (since 2026-10-27). On a session that attests a
  GPU, the TD first asks each GPU for a report on `gpu_nonce =
  SHA-256("x0ne-e2e/v1/gpu" ‖ 0x00 ‖ len16(key_config) ‖ key_config ‖
  client_nonce ‖ session_pubkey)`, then quotes `SHA-512("x0ne-e2e/v2/ohttp-gpu-key"
  ‖ 0x00 ‖ len16(key_config) ‖ key_config ‖ client_nonce ‖ session_pubkey ‖
  boot_nonce ‖ tls_spki_sha256 ‖ config_sha256 ‖
  u64(count) ‖ per GPU: u64(len) ‖ report ‖ u64(len) ‖ certificate chain)`,
  the decoded bytes of the document's `gpu_evidence_list`. The quote binds
  the evidence: a report, a chain or a GPU dropped, added or swapped after
  the quote is refused at `binding`. The client nonce is public, so a GPU
  nonce alone would let whoever holds another H100 in CC mode answer it;
  the quote is what ties the evidence to this TD. No label is a prefix of
  another, nor of the relay's contexts (tested).
- **Verification** (`x0ne-sdk/src/attest.rs`), in the relay's order
  (`remote_verify.rs`), each refusal naming its step: `schema`; `key` (one
  suite exactly, an Ed25519 point, one `x0ne-engine/cloud-config/v1` object
  of at most 64 KiB, the TD's cap, with no field the SDK does not know, a
  boot nonce and a TLS key digest of 32 bytes); `quote` (DCAP chain to the pinned root,
  validity at the caller's `now`, DEBUG clear); `binding`; `tcb` (Intel's
  signed TCB info and QE identity under the relay's TCB policy, below);
  `revocation` (the PCK CRL, then the root CA's CRL: the PCK CA and the
  collateral signers not listed; after `gpu`, the NRAS signing chain's
  stapled OCSP responses, a verified revoked refused under both policies,
  75); `measurement` (the MRTD in Google's launch
  endorsement, signed under the Google root the reference names, for the
  caller's machine type; the firmware binary that endorsement names,
  relayed in the collateral, the one Google signed and measuring the quoted
  MRTD; the TD's event log replayed into RTMR0..3, every RTMR1 and RTMR2
  event on the reference's closed lists, RTMR3 zero, every RTMR0 event the
  reference's with that binary's configuration volume in its slot; then the
  caller's pins on MRTD and RTMR0..3, optional); `gpu` (below); `session`
  (below). The reference must pin RTMR0 (`rtmr0.policy` `gce-endorsed`):
  one that reports it is refused when admitted (`reference`, 104, since
  2026-10-31).
  `verify_quote` runs the steps from `quote` on, for a `report_data` the
  caller computed; `verify_key` calls it with the key binding. The policy
  is an admitted image reference and a machine type (both below). The
  verifier is x0ne-infer's own `pcs.rs`, `tcb.rs`, `tcb_disclosure.rs`,
  `gce_endorsement.rs`, `ccel.rs`, `reference.rs`, `openpgp.rs`,
  `archive_key.rs`, `tlog.rs`, `transparency.rs`, `tdvf.rs` and `nras.rs`,
  compiled into the SDK by `#[path]`: one verifier, two builds. The session
  config's types are `cloud_config_types.rs`, compiled in the same way.
  Their test modules need x0ne-infer's fixtures, and their fetches and the
  reference's loading from disk open sockets and files: those are left out
  under the `x0ne_sdk` cfg that `x0ne-sdk/build.rs` sets, and nowhere else.
  Both CRLs and Google's endorsement come with the relayed collateral, from
  the URLs the relay reads (`attest::ROOT_CA_CRL_URL`,
  `attest::gce_endorsement_url`); a 404 for the endorsement is relayed empty
  and refused, as the relay refuses it. The PCK chain must be Intel's three
  certificates (PCK, its CA, root) with their basicConstraints and names
  chained, and the PCK's issuer one of Intel's PCK CAs (`tcb::pck_crl_ca`):
  until 2026-10-27 the SDK took a leaf issued under a PCK whose key was
  extracted. Each collateral chain must be Intel's
  two (TCB Signing, root), checked the same way: until 2026-10-27 a
  collateral signer issued under such a PCK was refused only by the root
  CA CRL check. Each field is capped before decoding:
  4 MiB for each piece of collateral (the relay's cap) and for the quote,
  16 MiB for the event log and 1 MiB for the reference (the relay's caps on
  those files), 64 MiB for one input of the ABI.
- **`gpu`** (`attest::checked_gpus`), after `measurement`. The policy names
  `attest`, `cpu+gpu` (required in the ABI, `Policy::production`'s): a
  document without GPU evidence is refused. `"cpu"` is refused by name since
  2026-11-04 (`ATTEST_CPU_REFUSED`: a TD serves only on its attested GPU);
  `Attest::Cpu`, under which a document with GPU evidence is refused, is left
  to the simulated TDs of `test-anchor` builds. Then NRAS's
  answer and its JWKS, carried in the key document (or relayed, for a TD
  before 2026-10-28), go through the relay's own check
  (`nras::verify_nras_attestation`, `x0ne-infer/src/nras.rs` compiled in by
  `#[path]`): each token ES384 under a leaf whose issuer is one of NVIDIA's
  pinned intermediates, issuer, `exp`, `iat` and `nbf` (since 2026-10-27)
  against the caller's clock,
  `eat_nonce` equal to the document's `gpu_nonce`, the overall verdict true,
  per GPU `measres` success, `dbgstat` disabled, secure boot and the
  certificate and OCSP claims, the overall token's `submods` naming each
  per-GPU token by its digest, one token per bound GPU, and each `ueid` one
  of the serials of the leaf certificates the quote bound, read locally from
  the chains (`nras::gpu_leaf_serial_decimal`). The verified key names them
  (`gpu_ueids`). Since 2026-10-31 the signing chain's own revocation is
  judged there too, from the OCSP responses the key document staples
  (`nras_ocsp`; `Policy::nras_ocsp`, `nras_ocsp` in the ABI): `transparent`
  by default, as the relay's, where a verified revoked refuses
  (`revocation`, 75) and the rest is named in the verified key
  (`nras_revocation`, one line per certificate, repeated in `limitations`);
  `required` (`with_required_nras_ocsp`, `"nras_ocsp":"required"`) refuses
  anything short of a verified, current `good` for the signing leaf. No TD
  staples a response yet. Caps: 1 to 8 GPUs, 64 KiB for each report and chain, at
  `key`; 4 MiB for the answer and the JWKS, at `gpu`. A bench's NVIDIA
  intermediate is accepted only by a `test-anchor` build
  (`nras_intermediate_der_b64`); the cfg it needs, `x0ne_sdk_test_anchor`, is
  set by `x0ne-sdk/build.rs` for that feature and by nothing else (tested).
- **`session`** (`attest::checked_session`, since 2026-10-30), last. The
  session config the quote bound asks for the policy's `attest`, and ends
  more than 60 s after the caller's `now` (`SESSION_EXPIRY_GUARD_SECS`, the
  relay's `local_api::EXPIRY_GUARD_SECS`, tested equal): the relay takes no
  new request from that edge on, and the caller stops at the same one.
  Refused with 88. `Client::seal_chat` refuses at that edge too, on the
  clock it is given. The verified key returns the config
  (`session_config`: `model`, the catalogue name the TD serves;
  `expires_at_unix`; `client_spki_sha256`, the relay's TLS key, the one
  client certificate the TD's TLS accepts), its digest, the boot nonce and
  the TLS key's digest, before anything is sealed; so does the ABI's
  summary of `OP_VERIFY_KEY`. Past `attest` and the end, the SDK pins none
  of them: a caller that expects a relay's key or a model compares these
  fields, and nothing here
  maps the model's name to the digest the caller pins (`LIMITATIONS`).
- **Who calls NRAS: the TD** (since 2026-10-28). Once the key document's
  quote is taken, the TD sends the GPU evidence that quote bound to NRAS for
  the document's `gpu_nonce` (`TdEvidence::nras`, the call the TD makes for
  the relay's session quote), outside the quote's lock and queue place, and
  the key document carries NVIDIA's answer and JWKS (`nras_answer`,
  `nras_jwks`). The caller judges them under its own pins, as it judged a
  relayed answer: whoever carries the document cannot forge the verdict. A
  token of its own making fails the signature or the pin, one for another
  nonce fails `eat_nonce`, one for another GPU fails `ueid`, an old one
  fails `exp`, a partial or recombined answer fails `submods` or the GPU
  count; a verdict without GPU evidence is refused at `key`. NRAS down, the
  TD serves no document (503, the detail in its log). A key document with
  GPU evidence and no verdict, from a TD before 2026-10-28, still takes
  NVIDIA's answer from the collateral: the SDK builds the request
  (`attest::nras_request`, ABI op 5), a carrier forwards it to NRAS
  unchanged (`/v2/e2e/nras` on the bench's carrier; the relay serves no such
  route). A document that carries a verdict is judged on it alone.
- **What the carrier and the caller choose.** The carrier relays the
  collateral: it cannot forge it, but it can relay genuine collateral older
  than Intel's newest. TCB info, QE identity and PCK CRL up to the relay's
  day (`Policy::max_collateral_age_secs`, `max_collateral_age_secs` in the
  ABI, shortens it for one caller; a longer one is refused), Intel's root
  CA CRL, whose age no verifier judges, until its nextUpdate (2027-02-26 for
  the CRL of 2026-02-26). A TCB recovery or a revocation Intel published
  since is not seen until a carrier relays it; the relay fetches from Intel
  itself, so it sees it at once. The served model is checked against the
  digest the caller names (`model_digest_hex`, required in the ABI and in
  `Pending::open_verified` since 2026-10-27), where the relay checks its
  catalogue's.
- **The image reference, admitted as the relay admits it** (since
  2026-10-28). The caller hands the reference's bytes, its detached
  signature and its transparency log proof (`attest::SignedReference`;
  `reference_json`, `reference_sig_b64`, `reference_tlog_proof` in the ABI,
  all three required). `attest::admit_reference` runs
  `reference::admit_offline`, the function the relay's
  `reference::load_signed` runs on the file it reads (`admit_signed`, one
  body for both, tested): signed by the archive key this build pins (99,
  the relay's exit code, returned as `code`), shown in the log this build
  pins under a fresh quorum of its witnesses (100), then parsed as schema v2
  with the image version floor (89). The verified key names what was
  admitted (`reference`: the file's digest, the signer's fingerprints and
  signing time, the log's origin, index, size and root, the memory's sizes
  before and after). Differences from the
  relay, each said in `LIMITATIONS`: the signature time and the checkpoint's
  age are judged on the caller's clock; no newer checkpoint is fetched, and
  the log's history is held to the caller's memory of it, not to a state
  directory (next bullet). A build made before the signing
  ceremony pins no key and no log: it refuses every reference, with 99. A
  `test-anchor` build pins the TEST key and log instead
  (`reference_keys: "test"` in the ABI; `"production"` and anything else are
  refused, tested). Whoever supplies the reference still chooses which image
  x0ne signed the SDK accepts (the digests of RTMR1 and RTMR2, the SVN floor,
  the name shown as `image`): it must come from the integrator's build or a
  source it trusts, never from the carrier.
- **The log's memory** (since 2026-10-31). The relay compares each
  checkpoint with the last one it verified, kept under its state directory,
  and reads the log's tiles to compare two sizes (`docs/cloud.md`, "The
  transparency log"). The SDK reads no file and no log, so the memory is the
  caller's: `attest::LogMemory`, the signed note of the last checkpoint
  verified, returned after every admission (`LogMemory::note`;
  `verified.tlog_checkpoint` in the ABI) and handed back before the next
  (`LogMemory::from_note`; `tlog_checkpoint` in op 1). `admit_reference`
  takes it, and `transparency::remember` holds the proof's checkpoint to it
  as a witness holds a submission (C2SP tlog-witness, `add-checkpoint`):
  the first checkpoint a caller meets is kept; the same size must show the
  same root, else the log forked (`log-fork`); a smaller size is refused
  (`log-rollback`: the log went back, or the proof is stale; a proof against
  the log's current checkpoint is needed, `x0ne-engine-tlog prove`); a
  larger one is admitted only with a consistency proof from the remembered
  size, the tlog-witness `add-checkpoint` body (`old <size>`, the RFC 6962
  consistency hashes, a blank line, the checkpoint; `x0ne-engine-tlog
  witness-request <log-dir> <size>` prints it, or any client computes it
  from the log's tiles), verified with RFC 9162 §2.1.4.2 against both
  roots; missing, truncated, altered, from another size, to another
  checkpoint or from another history, refused (`log-consistency`). A
  memory that is not a checkpoint signed by the pinned log refuses every
  reference (`log-memory`) until the caller clears it. Each refusal carries
  the relay's log code, 100, and names its step (`Refusal::step`, `step`
  in the ABI's result). The memory moves only on an admission, to the
  larger checkpoint. Tested on the TEST log grown in memory
  (`x0ne-sdk/tests/tlog_consistency.rs`, `tests/abi.rs`, and natively
  through the C ABI against the simulated TD in `e2e_sdk_tests.rs`): the
  first sight, the same checkpoint, the growth with and without its proof,
  the rollback, the fork at one size from either side, each false proof,
  the round trip of the memory as text and a tampered one. What it does
  not do: fetch a checkpoint or a tile (the integrator supplies the body
  when the log has grown; `js/x0ne-sdk.mjs` asks `options.consistency` for
  it and keeps the note in `localStorage`, or in the store `options.tlog`
  names); keep the memory for a caller that drops it (a memory emptied
  starts over: the next checkpoint is kept as the first, so whoever writes
  the caller's storage can reset it, or put an older genuine note back; a
  forged one is refused); see a fork a caller never met both heads of;
  accept an older checkpoint shown consistent, as the relay does from the
  tiles (one direction keeps `old` the remembered size, as a witness's
  409 rule). Rekor's clients keep no such state and leave the split view to
  a monitor; Go's checksum database client and a tlog witness keep their
  latest tree as this does.
- **The TCB policy is the relay's** (since 2026-10-28;
  `docs/limitations.md`, "Intel's TCB"). The collateral carries Intel's TCB
  info and QE identity twice, `update=early` (`tcb_info`, `qe_identity`)
  and `update=standard` (`tcb_info_standard`, `qe_identity_standard`), with
  their chains. `transparent`, the default (`Policy::production`, `tcb`
  absent in the ABI): the platform is accepted on `standard` (evaluation
  20 or higher), and Intel's `early` view of it is read, rated and named,
  or why it could not be, never judged; `strict` (`with_strict_tcb`,
  `"tcb":"strict"`): accepted on `early` alone (evaluation 22), refused with
  76 otherwise, and every receipt opened under that key must name Intel's
  latest view (`Pending::open_verified`), as the relay holds its receipts.
  `tcb::judge_live` decides both, one function for both builds. The verified
  key returns the relay's disclosure (`tcb`, `x0ne-infer/tcb-disclosure/v1`),
  which the application can show: which update it was accepted under, the
  platform's rating there and on `early`, and the advisories Intel lists.
  The floor and the day are this build's: a caller can shorten the day,
  neither lower the floor nor lengthen the day (tested).
- **A real quote** (`x0ne-sdk/tests/real_quote_2026_10_18.rs`, offline): the
  whole quote of the GCE run of 2026-10-18. Its `report_data` binds the
  relay's channel (binding v2), not an OHTTP key, so `verify_key` refuses it
  at `binding`. `verify_quote`, given that binding, the run's event log, the
  run's reference (x0ne-cvm 0.2.3) and no pin, accepts it to the end on
  Intel's collateral of 2026-10-22 for its FMSPC (evaluation 22), Intel's
  root CA CRL of 2026-10-23 and Google's endorsement of its MRTD, at a clock
  of 2026-10-22: the TCB info and QE identity of the run's day were not
  kept. Refused, each at its step: one bit of any signature, of the QE
  report or of the signed TD report (`quote`); another `report_data`
  (`binding`); collateral of another platform, out of its window, of
  evaluation 20, older than a day, altered or with a cut chain (`tcb`); the
  PCK CRL of 2026-10-18 four days on, the root CA's CRL in its place, one bit
  flipped; the root CA CRL withheld, the PCK CA's in its place, one bit
  flipped (`revocation`); Google's endorsement withheld, of another
  firmware, one bit of its signature or of what Google signed, under
  Intel's root, a reference naming another Google root, another machine
  type, an SVN floor above it; one event of the log changed (an EFI action's
  text, the GPT's type or one of its bytes), a digest changed, the last
  event cut, an event appended, the log withheld; the endorsed firmware
  binary withheld or another in its place; pins other than the quote's
  (`measurement`); a reference reporting RTMR0 (`reference`, 104); a
  cut PCK chain or a chain other than Intel's three certificates
  (`quote`). DEBUG, MIGRATABLE and register changes are
  reached past the signature by re-signing the real TD report under a test
  root, where register changes now stop at `tcb`, as in the relay. Since
  2026-10-28 the run's reference is admitted on its own signature by the
  archive key that signed it on 2026-10-18 (the TEST key, the run's), with
  a TEST log proof; a byte of it changed is refused with 99. Under
  `transparent` the quote is accepted on Intel's standard collateral of
  2026-10-22 (evaluation 20) and named on early (22); the other tests run
  under `strict`. Not covered by this test: a key document of a real TD and
  the NRAS token (verified once, live, on 2026-10-28: "Through the relay"),
  a revoked PCK, a stale root CA CRL (on the bench only).
- **Sealing** (`hpke.rs`, `ohttp.rs`, `bhttp.rs`, `e2e.rs`): HPKE base mode,
  DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, AES-128-GCM, the suite of RFC 9180
  A.1 and of RFC 9458's example; standard OHTTP labels and media types; Binary
  HTTP known-length messages padded to 4 KiB, then powers of two up to 4 MiB,
  both ways. The caller brings every random byte (`crypto.getRandomValues`)
  and the clock: the crate never draws and never reads a clock.
- **Gateway** (`Gateway::open`, the TD's end): `POST /v1/chat/completions`,
  `application/json`, one JSON object whose `stream` is absent or `false`
  (the rule `Client::seal_chat` applies, held again for a request sealed
  without `Client`), one IMF-fixdate `Date`. A refusal leaves the TD in
  clear and names no byte of the body. `Gateway::open_stream` opens a
  chunked request (`message/ohttp-chunked-req`) to the same rules, except
  that its `stream` must be `true` ("Streaming").
- **Replay** (`ReplayGuard`): one answer per encapsulated key, within 120 s of
  the `Date` sealed inside the request (RFC 9458 §6.5.1); a full table refuses
  rather than forgets. Since 2026-10-30 the window's old edge follows the
  highest time the TD's clock has shown, which never goes back (the
  high-water mark of IPsec's anti-replay window, RFC 4303 §3.4.3): an entry
  is dropped only past that edge, where the request's date refuses it for
  good, so a host that sets the TD's clock back reopens nothing. A request
  dated ahead is judged on the clock as it reads. Pushed forward, the clock
  makes the TD refuse honest requests until real time catches up: a denial
  of service, which the host has anyway. The memory lives as long as the
  key, one boot: a request sealed to an earlier boot's key does not open.
- **Receipts** (`receipt.rs`): the answer's signed payload must be signed by
  the session key the quote bound, name that key, commit to the text shown,
  and name the model digest the caller gave. Every field returned is read
  from the signed bytes, never from the receipt object next to them. Since
  2026-10-30 it must also sign the decoding the sealed body asked
  (`check_answer_for`, `decoding.rs`): the sampling and its seed when the
  body samples, the digest of its `json_schema` and a text that is a
  document of it, or its start under `finish_reason: "length"`
  ([api.md](api.md#response_format-json-held-to-a-schema)).
- **Browser** (`abi.rs`, `js/x0ne-sdk.mjs`): a C ABI, JSON in and out, no
  wasm-bindgen; `wasm32-unknown-unknown`, a module with no imports. A hex field
  of the input or of the key document holding a multibyte character is an
  error naming the field, not a panic (a panic is a trap of the instance). The
  `test-anchor` feature lets a build accept the bench's root; a build without
  it refuses `root_der_b64` and takes Intel's root alone (tested on the
  pinned wasm build, natively by a source test). An instance holds 1024
  handles at once; op 4 closes a key, a pending request or a stream
  (`session.close()`, and `chat` and `chatStream` close theirs when no
  answer came). Since 2026-10-31 the ABI streams: op 6 seals a body whose
  `stream` is `true` as a chunked request and returns a stream handle, op 7
  takes one frame of the response as it came off the wire and returns the
  deltas it completed, NOT VERIFIED, op 8 takes the handle at the end of
  the body and returns the text once the final chunk's receipt checked over
  all of it, with `final_deltas` and the count of chunks; a frame altered,
  dropped, reordered or replayed is refused at op 7, and the stream stays
  refused at op 8. `session.chatStream(body, onDelta)` drives the three
  (`js/x0ne-sdk.mjs`); `smoke.mjs` runs it under `X0NE_SMOKE_STREAM=1`.
  Since 2026-11-02 what a receipt does not sign is apart: ops 3 and 8
  return `status`, `text` and `receipt` (op 8: `final_deltas`, `chunks`,
  `conversation`) and, under `unverified`, the answer's `body` (op 3) or
  the stream's `finish_reason` and `usage` (op 8). An answer with no
  checked receipt, the TD's refusal or a stream ended by an error event, is
  an error whose `unverified` holds the TD's status and its body or error;
  `chat` and `chatStream` throw it, and the deltas already shown are to be
  discarded. A native host may call from several threads: each reads its
  own result at `x0ne_result_ptr`, valid until its own next `x0ne_call`;
  the handles are the instance's, shared (`x0ne-sdk/tests/abi_ffi.rs`). No
  export unwinds into the host: a panic is the result `code: 101`,
  `step: "panic"`, and drops every handle of the instance, whose state it
  may have left half changed
  (`e2e_sdk_tests::a_panic_in_the_c_abi_drops_every_handle`); in wasm32 a
  panic is a trap, after which the loader refuses every call.
- **What is compiled in.** Of the relay's files, what reads a disk or opens a
  socket is left out under `x0ne_sdk`; a source test sweeps them
  (`x0ne-sdk/tests/shared_sources.rs`).

## The TD's routes

`x0ne-infer --cloud-config` serves them behind the mutual TLS of the other
routes (`cloud_serve.rs`); the gateway is `e2e_gateway.rs`, which the bench
(`e2e_sdk_tests.rs`) runs too, its lies told around it. Choices the sections
above left open, each the stricter of two and each reversible:

- **Ready first.** Both routes answer 503 until the model is loaded.
- **GPU sessions** (since 2026-10-27). A `cpu+gpu` session's key document
  carries each GPU's report and certificate chain (`gpu_evidence_list`), read
  through NVML for the document's `gpu_nonce` before the quote. NVML is read
  outside the lock that session quotes share, as on the relay's session
  path: a slow NVML held a session quote 2.8 s behind one key document until
  2026-10-27 (review of that day, finding 1). Documents may interleave; each
  hashes its own evidence into its quote. Before reading the reports, the TD
  reads NVML's CC state (`nvmlSystemGetConfComputeState`) and refuses unless
  confidential computing is on, in its production environment, with devtools
  off: a GPU in devtools mode still gives reports (review of 2026-10-27,
  finding 3). Checked on a simulated NVML only, not yet measured on an
  H100. A GPU outside CC mode, or refused so, gets the caller a 503, the
  class in the TD's log, nothing in the body, and no quote is taken. A chain
  whose leaf serial cannot be read is refused by the TD itself. Once the
  quote is taken, the TD asks NRAS for its verdict on the evidence the
  quote bound, outside the quote's lock and queue place, and the document
  carries it ("Who calls NRAS" above); NRAS down, the caller gets 503 "e2e
  key GPU verdict unavailable", the detail in the TD's log. Each key
  document of a GPU session costs one NRAS call, made by the TD for an
  unauthenticated caller of its route: the bucket below bounds them as it
  bounds quotes.
- **One quote per key document**, under the session quote's limits but not
  its bucket: one quote per TLS connection, a count that a session quote and
  a key document share; a bucket of the same size as the session's
  (`--max-queue + 2`, one more per second) for key documents alone, so a flood
  of them does not close `/v1` to new sessions; one quote in flight.
  `report_data` is the binding above: the caller's nonce enters it hashed,
  never as chosen bytes.
- **The event log** in the key document is the attestation document's
  (`CloudState::ccel`: the ACPI CCEL table, read by `TdEvidence::ccel` when
  the TD attests), the bytes the relay replays. The caller replays it into
  the new quote's RTMRs and judges it against its reference; a register
  extended after the log was read would not replay, and the document would
  be refused at `measurement`.
- **The session config** in the key document is the file the TD booted
  under (`--cloud-config`), its exact bytes, and the boot binding the one
  its boot quote binds; the gateway checks the bytes against that binding's
  `config_sha256` when it is made (`E2e::from_seed`), and the TD exits 97
  otherwise (tested).
- **The key** is derived per boot from 32 bytes of `/dev/urandom`, read into
  memory wiped on drop. Its private scalar is held as the model key's
  recipient key is: `memfd_secret`, else `mlock`, out of core dumps, wiped
  on drop (`harden::SecretBytes`); if neither works the TD exits 97, and past
  `RLIMIT_MEMLOCK` a kernel with Linux commit 97d34aa65c29 kills it with
  SIGBUS at the first write instead. Deriving it and each X25519 operation
  pass the scalar by value, which leaves stack copies that are not zeroed.
  The receipt key is the session
  key the boot binding names.
- **The chat** opens the request (`Gateway::open`), admits it once
  (`ReplayGuard`), and hands its JSON to the attested chat handler
  (`server::chat_as`): the queue place, the padding to a size, the turn
  cadence and the signed receipt of `/v1/chat/completions`. The answer is
  sealed whole under 16 fresh bytes; the outer response is 200
  `message/ohttp-res` whatever the inner status. A panic of the model's code
  is such a status: a 500 padded to the same size as any whole answer,
  sealed, at the turn's cadence (`server::chat_contained_as`), never a 500 in
  clear when it happens. The panic report still writes one line to the TD's
  stderr at that moment, its location and no payload: a host that reads the
  console learns that, and when, a panic happened.
- **Text only.** Every string of the question's JSON is text: a lone
  surrogate (`\ud800`), which the JSON grammar allows and the chat handler
  cannot read, is refused by the caller before anything is sent, and by the
  gateway as it opens; both run the same check (`check_chat_body`).
- **One refusal.** A request that does not open, is not a chat request, was
  answered already, falls outside the 120 s window either way, does not come
  as `message/ohttp-req` or is larger than the largest the SDK seals
  (`e2e::MAX_SEALED_REQUEST`: the 4 MiB bucket and 55 bytes of OHTTP, the
  route's body limit) gets 400 `{"error":"sealed request refused"}`, one
  second after it arrived, as a whole answer refused before its turn. The
  class is counted, not said; `e2e_gateway.rs` has no log macro
  (`tests/no_log_on_answer_paths.rs`).
- **Collateral is relayed**, not served by the TD: the carrier fetches
  Intel's TCB info and QE identity from PCS for the platform of the TD's
  quote, with and without `update=early`, the PCK CRL, Intel's root CA CRL,
  and Google's endorsement of the quote's MRTD
  (`attest::gce_endorsement_url`), as the relay does, and NRAS's JWKS
  (`attest::NRAS_JWKS_URL`). `/v2/e2e/collateral` in the diagram is the
  carrier's: the relay serves it from its own fetches
  (`remote_verify::sdk_collateral`).

`x0ne-infer/src/e2e_td_tests.rs` runs the TD's server code as `run_cloud`
assembles it (`cloud_serve::serve_tls`, mutual TLS, the chat handler) on the
synthetic Intel of `attest_v2::fixtures`: the binary has no simulated
evidence source, by design. A carrier holds the client's TLS key, relays the
collateral from a fake PCS and Google's vendored endorsement from a mirror
of its bucket, and records every byte it moves, on both wires and in clear
as it holds them. The caller is x0ne-sdk, under the image reference of
`reference::tests` and the TD's pins. The tests show:

- the caller gets the model's answer, and its receipt checks under the key
  the quote bound, for the pinned model digest; under another key or another
  pin it does not;
- neither the question nor the answer is in any recorded byte, while the
  same recorder finds both in the traffic of the plain
  `/v1/chat/completions` route;
- refusals at the step named: a bit of the quote flipped (`quote`), the
  carrier's key or another receipt key in the document (`binding`), a pinned
  MRTD or RTMR that is not the TD's, a TD of another image, the key
  document's event log withheld or one event longer (`measurement`), no
  collateral (`tcb`); the event log is the one the attestation document
  carries for the relay;
- a bit flipped in the sealed request, the request replayed, sent under
  another type, sealed an hour ago or dated an hour ahead: the same 400,
  after the floor and less than 500 ms past it, and the model does not run;
  a request answered, the TD's clock moved 130 s on (an honest request
  answered there drops the first one's entry), then set back inside the
  first request's window and the request replayed: the same 400, the model
  not run again (answered twice until 2026-10-30);
  a bit flipped in the answer, or another request's answer: the caller does
  not open it;
- every truncation of a sealed request, its high bit flipped in each byte in
  turn, and noise: refused, no panic; fourteen plaintexts sealed to the TD's
  key that are no chat request (not Binary HTTP, another framing, padding
  not zero, unpadded, another method, path or content-type, `stream`, not
  JSON, an array, no date, two dates, a date that is not an IMF-fixdate):
  the same 400 at the same cadence, the model not run;
- a lone surrogate: refused by the caller before sending, and at opening;
- a panic of the model, in the chat template or the generation: a 200 of the
  size of an honest answer, after the floor, its inner 500 padded; the TD
  answers the next question;
- the largest request the SDK seals is read and answered, one byte more is
  refused unopened; the same for the largest chunked request;
- one quote per connection, shared with the session quote; key documents
  that empty their bucket leave a session quote and a `/v1` chat served; 503
  before the model is loaded;
- a GPU session, on a simulated NVIDIA and NRAS (`attest_v2::fixtures`):
  sealed and answered through the carrier, NVIDIA's verdict in the key
  document and no NRAS call by the carrier, its GPU named; a verdict for
  another nonce or naming another GPU refused at `gpu` (75); NRAS down, 503
  and no document; refused at `gpu` under `attest=cpu` (`test-anchor`), and a
  CPU session under `attest=cpu+gpu`; the GPU evidence and its verdict dropped by the
  carrier, refused at `binding`; a GPU outside CC mode, or whose NVML says CC off, devtools on
  or a simulated environment, 503 with nothing named and no quote; a session
  quote is not held behind a key document's slow NVML.
  The bench (`e2e_sdk_tests`) refuses, each at its step: evidence dropped,
  added or swapped after the quote (`binding`); an NRAS answer for another
  nonce or replayed from another document, under an unpinned key or
  intermediate, judged failing (overall, `measres`), in debug, for another
  GPU, its per-GPU tokens from another answer, expired, withheld or
  malformed (`gpu`); fields over their caps (`key`, `gpu`). The verdict the
  key document carries is the one judged: for another nonce, another GPU or
  failing overall, refused at `gpu` (75), without its JWKS too; a verdict
  without GPU evidence, at `key`.
- the reference and the TCB policy (`e2e_sdk_tests`, "The relay's TCB policy
  and the reference's admission, caller side"; `x0ne-sdk/tests/abi.rs`):
  a reference unsigned, signed over other bytes or with one bit flipped,
  refused with 99; its log proof another file's, empty, or under a
  checkpoint eleven years old, with 100; the SDK runs the relay's
  `admit_signed` (a source test counts one body and its two callers); a
  key document from another image, refused at `measurement` (65). On
  Intel's collateral of 2026-10-28, the host OutOfDate on `early` (three
  advisories named): accepted under `transparent` on `standard`
  (evaluation 20) with `early`'s rating and advisories named, refused under
  `strict` with 76; `early` withheld: named "not relayed" under
  `transparent`, refused under `strict`; under `strict`, an answer whose
  receipt carries no TCB disclosure is not opened. A caller from before
  2026-10-28 (the SDK's earlier `KeyDocument` and `Collateral`, unknown
  fields refused): reads a CPU key document as before, and refuses a GPU
  document (`nras_answer`) and the new collateral (`tcb_info_standard`) by
  the field's name; an earlier ABI caller is refused by the fields it lacks
  (`reference_sig_b64`, `reference_tlog_proof`, `now` in op 3).

The model of `make ci` is scripted, under the catalog's digest for
`qwen3:0.6b`. `make e2e-local` adds the wasm32 build in Node through the same
carrier and, given `X0NE_E2E_QWEN3_DIR`, Qwen3-0.6B's weights on CPU: the
caller's text must be what the model generated for the question, and what a
second load of the same files generates from the same prompt.

## Streaming

Step 6, since 2026-10-30. `Client::seal_chat_stream` seals a chat request
whose `stream` is `true` as a chunked request, and `PendingStream` reads its
answer chunk by chunk (`x0ne-sdk/src/sealed_stream.rs`); the TD answers it
on the same route (`e2e_gateway::answer_stream`, `e2e_stream.rs`).

- **Format.** Chunked Oblivious HTTP, draft-ietf-ohai-chunked-ohttp-08, read
  to the letter (`ohttp.rs`): `message/ohttp-chunked-req` and
  `message/ohttp-chunked-res`, the labels `message/bhttp chunked request`
  and `message/bhttp chunked response`, the response nonce of 16 bytes, the
  AEAD nonce XORed with the chunk counter, `final` as the last chunk's AAD,
  `Incremental: ?1`. Tested byte for byte against the request and the
  response of the draft's appendix A, copied from the IETF text
  (`x0ne-sdk/tests/chunked_ohttp.rs`). Not tested against another
  implementation. The request is the padded Binary HTTP request of the whole
  path, cut in chunks of 16 KiB, the largest a receiver must take (§3). The
  TD refuses a non-final chunk of less, read from its length before any
  chunk is opened, so it opens 257 chunks at most: a question of 4 MiB in
  chunks of one byte took 67.9 ms to open, against 9.4 ms in chunks of 16
  KiB (an audit, 2026-10-30). The response's plaintext is one Binary HTTP response of
  indeterminate length (RFC 9292 §3.2), as the draft requires: a 200 whose
  content is OpenAI's SSE chunks, or a refusal whose content is its JSON
  error. One answer per encapsulated key, whole or streamed: both paths
  share the replay memory.
- **Cadence.** The session config fixes it, so the quote binds it and the
  caller reads it before sealing: `e2e_stream: {chunk_bytes, tick_ms}`, L
  from 128 to 4096 bytes, T from 20 to 1000 ms, 256 bytes every 100 ms when
  absent. Every chunk but the last carries exactly L bytes of plaintext, the
  last 16384; the first leaves T after the TD read the request, then one
  every T, text or not. The TD admits the question within that first T: the
  largest (4 MiB, chunked) took 7.5 to 12.2 ms, median 8.9 ms, against the
  shortest T of 20 ms (release build, scripted model's template, a loaded
  host, seven runs, 2026-10-30,
  `e2e_td_tests::measure_admission_of_the_largest_question_against_a_20_ms_tick`,
  ignored); an admission past T would show in the first chunk's time. The
  SSE events wait in a queue inside the TD: at most one new event of text
  per tick, only while the queue is under a chunk, then line feeds, which
  SSE and JSON both skip, after whole events only. Text that comes in more
  often than that waits in the queue, text that comes in less often leaves
  line feeds; neither shows. The count of chunks, the last included, is a
  power of two, 16 at least: the last chunk goes at the first such count
  once the answer is out, with the receipt and `[DONE]`.
- **Refusals and failures.** Every outcome is a stream of that shape. A
  request refused before the model (a field the validator refuses, a model
  not served, empty messages, a full set of open streams, a lease expired, a
  panic of the chat template) is a stream of 16 chunks whose plaintext is
  the status and its JSON error. A failure once admitted (a prompt past the
  context, a panic of the generation, an error, 90 s without a token) is an
  error event in a 200, at the next power of two. A request that does not
  open, is replayed or is no chat request gets the whole path's one 400 one
  second after it came, as the whole path: the TD holds no key to seal
  anything to its sender. A refused stream holds its queue place 16 ticks
  after its request, as the shortest stream's turn, and its open stream
  place to its last chunk, as an answer: a host that probes either with
  requests of its own (a full queue or a full set of open streams answers
  503) does not tell a refusal from an answer by it.
- **The session's end.** A stream is 8192 chunks at most at T = 100 ms
  (819.2 s, the largest power of two within the relay's 15 minutes), and
  fewer when the session ends sooner: its last chunk must leave a tick
  before the session's end less the relay's 60 s. A stream asked when even
  16 chunks would not fit is refused, 503, inside 16 chunks, the model not
  run. A stream that reaches its cap ends there with an error event, its
  generation cancelled. No receipt is signed for a stream past the
  session's end less 60 s.
- **The caller's checks** (`PendingStream`): each chunk opens under the
  next counter, so a chunk altered, swapped, replayed or from another
  stream is refused; a chunk of another size, a count of chunks that is no
  power of two within the cadence's bounds, and a stream without its final
  chunk are refused (the last, a cut stream read as whole, is
  CVE-2026-48480's class). A stream of more chunks than its session
  allows (`sealed_stream::cap`, the TD's bound: its last chunk a tick
  before the session's end less 60 s) is refused at the chunk past it, and
  one asked too late to fit 16 chunks is refused before it is sent. Once
  refused, the stream stays refused. The
  deltas handed out before `finish` are not verified, and their type says
  so: `push` returns `UnverifiedDelta`, whose text is read by
  `unverified_text` alone. A caller that shows them shows text the stream
  may still be refused for. `finish` checks the
  receipt of the final chunk over the whole text, as for a whole answer,
  the decoding the question asked included (sampling, seed, the schema's
  digest, the text a document of the schema, a start under `length`), the
  context its conversation implies when the question named one
  (`x0ne.conversation`, api.md: the signed `context` block, the KV-cache
  flag, and the handle the final chunk hands back, which must open the
  signed commitment and comes back in `StreamAnswer::conversation`), and
  the TCB disclosure under `strict`, and returns the refusal or the error a
  stream carried. The final chunk may carry the end of the text as well:
  `finish` returns those deltas in `StreamAnswer::final_deltas`, verified
  with the rest; `text` is what `push` handed out, then them. The TD checks a
  streamed text against its schema again at the end: one outside it ends
  in an error event, unsigned, its deltas already sent.
- **What the carrier still sees** of a stream: when it was asked, the
  size bucket of the question, the count of chunks, a reader that stops
  reading, and that it is a stream. The count tells the answer's duration
  to within a factor of two, and bounds its length (under L bytes a
  chunk); when the text, not the model, sets the count (a model that
  gives more than one event of L bytes a tick), it tells the length to
  within a factor of two as well. Not the text, nor its pace, nor success from
  failure. The host sees one thing more, the model's turn: a stream's
  turn, and the queue place it holds (a full queue answers 503), end at
  the first power of two of ticks after the request, 16 at least, that
  the generation fits (`e2e_stream::StreamTurn`), as the chunks do, not a
  Padme size after the generation (11.6 % at most), which told its time
  finer than the count. The price is the model held up to twice a
  generation's time: of two short streams asked together, the second's
  generation started at 1.65 s instead of 1.05 s, and its answer took 32
  chunks (3.2 s) instead of 16 (1.6 s), measured on 2026-10-30
  (`e2e_td_tests::a_sealed_streams_turn_ends_at_a_power_of_two_of_ticks`,
  scripted model, T = 100 ms); a generation of 1.7 s holds the model
  3.2 s instead of 1.8 s, one of 6.5 s, 12.8 s instead of 7.2 s. Past the
  stream's cap the turn ends with the generation.
- **The final chunk's room.** It holds the queue left and the receipt; at
  L = 4096, the fullest queue and the largest real `x0ne` block (10747
  bytes: an H100's, a strict TCB view, the decoding's claims, a
  conversation's context and handle) do not fit, by 3958 bytes. The stream then ends at the next power of two, whole: a
  text of 16 events of L bytes ends at 32 chunks with that block, at 16
  with the H100's 7322 bytes (`e2e_stream::tests`, 2026-10-30).
- **Cost**, measured on 2026-10-30 through the recording carrier on
  loopback, debug build, scripted model, L = 256 and T = 100 ms
  (`e2e_td_tests::the_cost_of_a_sealed_stream_is_measured_against_the_whole_answer`):
  the request is one byte longer than the whole one (4152 against 4151
  bytes); the shortest stream is 20527 bytes (a first chunk of 290 bytes
  with the nonce, 14 of 274, a last of 16401), against 32800 for the whole
  answer to the same question; its first text arrived 253 ms after the
  request was sent, its receipt at 1.65 s, against 1.06 s for the whole
  answer. Past 16 chunks, 274 bytes a chunk, 2740 bytes a second, the count
  rounded up to the next power of two.

Tested (`x0ne-sdk/tests/chunked_ohttp.rs`,
`x0ne-sdk/tests/sealed_stream.rs`, `e2e_stream::tests` on a paused clock,
`e2e_td_tests` through the recording carrier): neither the question nor the
answer is in any byte the carrier moved or held; two answers of two
subjects, one 23 times longer, leave chunks of the same sizes, each chunk
within 6 ms of the other's (one run, debug build, on a loaded host) and none
before its tick, their heads a tick after the request; refusals before the
model, a panic of the template and a panic of the generation leave the same
16 chunks at the same rhythm; a receipt that signs another seed, greedy
decoding, no schema, or a text that is no document of the schema (a start
without `length`) is refused by the caller; a stream sampled with a seed and
held to a schema by the qwen3 fixture on CPU checks, its receipt signing
both, and a backend blind to the schema ends in an error event, unsigned; a
stream asked a minute and a second before the session's end is refused
without running the model; a stream withheld its last chunk, cut inside a
chunk, a bit flipped, two chunks swapped or one replayed: refused by the
caller; a chunked request sent again: the one 400, the model not run again;
a chunked request sent as a whole one and the converse: the one 400,
unopened; a refused stream and an answered one hold the TD's queue and open
stream places for the same time, sampled at 0.5 s, 1.3 s and after the end;
of two short streams asked together, the second's generation waits for the
first's 16th tick; the end of a text the final chunk carries reaches the
caller through `finish`; at L = 4096, the largest real receipt behind a full
queue ends the stream a size later, whole; a request in chunks under 16 KiB
is refused unopened; a stream past its session's count of chunks is refused
by the caller, and under `strict` a receipt without its TCB disclosure at
`finish`. The plaintext's reader (`x0ne-sdk/tests/bhttp_stream.rs`) reads a
response cut at every byte as the whole, refuses one that stops early naming
where, and refuses another framing, an informational status, more than 64
fields, a field over 8 KiB, content past its bound, a trailer field and a
non-zero byte after the end; the stream's reader refuses an SSE line over 1
MiB, a head whose content type does not match its status, and an event out
of its place.

Since 2026-10-31 the relay carries a chunked request and its stream
("Through the relay", below), and the browser ABI streams ("Browser",
above): `e2e_td_tests::the_c_abi_streams_through_the_carrier_and_refuses_a_bent_frame`
drives ops 6, 7 and 8 against the recording carrier (the deltas handed out
unverified equal the text the receipt signs; a bit flipped in the fifth
chunk is refused at op 7 and at op 8; a body not asking to stream is
refused before anything is sealed; a handle never issued serves nothing;
op 4 frees a stream whose end never came), and `wasm_in_node_through_the_carrier`
runs `smoke.mjs` streamed when the wasm32 build is at hand. Not done: a
test against another implementation of the draft.

## Through the relay

Since 2026-10-28 the relay (`x0ne-engine cloud`) carries the sealed path for
a TDX session on its local endpoint (`local_api.rs`): `GET /v2/e2e/key`,
`POST /v2/e2e/chat` and `GET /v2/e2e/collateral`, behind the endpoint's
bearer, loopback only. It forwards the key document and the sealed messages
over its attested TLS connection to the TD, as they came, and serves the
collateral from its own fetches of Intel's PCS, Intel's root CA CRL and
Google's bucket. It verifies none of it for the caller: x0ne-sdk does, for
its own nonce.

```
application + x0ne-sdk ──HTTP, loopback, bearer──▶ relay ──mutual TLS 1.3──▶ TD
   question in clear       sealed (HPKE, OHTTP)    sealed          sealed     opened
```

What each hop holds in clear, segment by segment: the application and the
SDK hold the question and the answer; between the SDK and the relay, and
between the relay and the TD inside its TLS, only the sealed request and
the sealed answer; the TD opens them in its attested memory. The relay
holds no key that opens either: the OHTTP key's private half is the TD's,
derived per boot inside it. What the relay sees: the key document (public:
the quote, the event log, the GPU evidence and verdict), the size bucket of
each sealed message (4 KiB, then powers of two), when each is sent and
answered, the outer status, which TD they go to and so the model it serves,
and the caller's requests on its endpoint. What it can do: withhold, delay,
replay or alter. A replay or an altered request gets the TD's uniform 400;
an altered answer is not opened by the SDK; a key document it altered or
replaced is refused at its step; collateral it withholds or ages is
refused or named, as above.

`cloud_e2e_tests::the_relay_carries_the_sealed_path_and_reads_none_of_it`
brings up the real relay against the TD's server code on synthetic
evidence, records every `/v2/e2e/` exchange on both sides of the relay
(the caller's HTTP, and the HTTP the relay put in its TLS channel to the
TD), and shows: the SDK verifies the key under the lab's reference, signed
and logged under the TEST keys, with the relay's TCB verdict (same policy,
same update, same ratings); the sealed request is the size of its bucket; a
gateway key the relay could hold opens nothing; the answer opens, its
receipt checks for the catalog's digest; a replay and a request with one
bit flipped get 400, an answer with one bit flipped is refused by the SDK,
another content type gets 415 before the TD, no bearer gets 401; under
`strict` the receipt names Intel's latest view and opens; neither the
question nor the answer is in any recorded byte.
`the_relay_carries_a_gpu_key_document_with_its_nras_verdict` does the same
on a GPU session: the document the relay forwards carries NVIDIA's verdict,
the SDK names the GPU, and refuses it at `gpu` under `attest=cpu` (a
`test-anchor` policy).

Since 2026-10-31 the same endpoint carries the sealed stream ("Streaming"):
a `message/ohttp-chunked-req` goes to the TD as it came, and the TD's
`message/ohttp-chunked-res` comes back to the caller frame by frame, as the
TD writes it, through `relay_sealed_stream` (`local_api.rs`): the relay
reads a frame, passes it on, and holds nothing back; it opens no chunk and
parses none. The relay bounds what it carries, and names every cut in
`/x0ne/v1/status` under `sealed_streams` (`carried`, `cut` by code,
`last_cut`): over the bytes the session's cadence allows
(`sealed_stream::max_wire_bytes`, 2 260 751 bytes at the default cadence)
the stream is cut and the session stopped (65, `sealed_stream_too_large`);
30 s without a frame (`sealed_stream_silent`), the body broken
(`sealed_stream_td_broke`) or still open 15 min after the request
(`sealed_stream_too_long`) cut it and lose the session, as a plain stream;
a client that leaves (`sealed_stream_client_gone`) closes the TD's
connection, so the TD stops generating. A cut breaks the client's body
short of its end, so the SDK refuses the stream. A request over the
largest of its kind is refused with 413 and `sealed_request_too_large`
before the TD; the TD's own refusal (400) is carried whole. What the relay
still sees of a stream, on top of the whole path's: the count of frames,
their sizes (the cadence's, the same for every stream) and their timing,
so the stream's duration to a factor of two.

`cloud_e2e_tests::the_relay_carries_a_sealed_stream_chunk_by_chunk_and_reads_none_of_it`
brings up the real relay against the TD's server code on synthetic
evidence, with the TD's answers recorded frame by frame as the TD writes
them, and shows: the answer opens and its receipt checks; the caller read
the first frame before the TD had written the last, and the frames span
over a second (16 chunks at 100 ms); the chunks' sizes are the cadence's;
a replayed request gets the TD's 400 whole; a frame with one bit flipped
is refused by the SDK at `push` and at `finish`, the last frame dropped at
`finish`; a chunked request one byte over its largest, and a whole one
over its own, get 413 with `sealed_request_too_large` and never reach the
TD; the status counts four streams carried and no cut; neither the
question nor the answer is in any recorded byte, on either side of the
relay. `a_conversation_crosses_the_relays_sealed_stream_and_the_sdk_checks_its_context`
runs two streamed turns of a conversation through it on the qwen3 fixture
(turn 1, then turn 2 reusing its keys and values), the handles checked by
the SDK against the receipts and in no byte the relay moved.
`local_api::tests::streams::sealed` drives `relay_sealed_stream` on a
scripted TD body and a paused clock: three frames pass as they came and
the connection is kept; over the cadence's bound the stream is cut, the
session stopped with 65 and the code in its reason; 31 s of silence and a
stream open past 15 min are cut and lose the session; a client that leaves
closes the TD's connection and loses nothing; each cut counted under its
code. One sealed stream crossed the relay on hardware on 2026-11-01, from
the SDK run in Node ([hardware-runs.md](hardware-runs.md)): the relay's
`sealed_streams` counted it carried and none cut, and the text equals the
whole answer's.

On real hardware, first on 2026-10-28 (the runs of 2026-10-30, with a v2 key
document, 2026-10-31, with RTMR0 pinned, 2026-11-01, with the answer
streamed sealed, and 2026-11-04, on x0ne Core, repeated it:
[hardware-runs.md](hardware-runs.md)):
a relay built with debug assertions and `test-archive-key`, a TD on Google
Cloud `a3-highgpu-1g` (Intel TDX, one H100 in CC mode) serving
`ministral3:14b`, and this SDK built for wasm32 with `test-anchor`, run in
Node on the operator's machine with the relay's bearer. The SDK fetched
the key document and the collateral through the relay and verified the
document under `transparent`, the image reference admitted under the TEST
archive key and log at its caller's request, NVIDIA's verdict on one GPU
included, with the relay's TCB line; it sealed one question, opened the
answer and checked its receipt (the catalog's digest, `cuda:0`). A sealed
request with one bit flipped after sealing got the TD's 400; under
`strict` the SDK refused that host at `tcb` (76). A random word written in
the question only was in no byte the SDK sent to the relay or received from
it, in no file the run wrote on the operator's machine (the relay's logs
included) and not in the relay's state. That measures what the relay held;
what the host could read rests on TDX and on the limits below.

A caller of the relay that sends the plain `/v1/chat/completions` keeps the
relay's path: the relay reads its text, as `docs/cloud.md` says.

## What it does not do

`x0ne_sdk::LIMITATIONS`, returned with every verified key:

- prototype: the TD's e2e routes ran on Intel TDX hardware with one H100
  (Google Cloud a3-highgpu-1g) on 2026-10-28, 2026-10-30, 2026-10-31 and
  2026-11-01, one sealed question each time, asked whole and, on 2026-11-01,
  streamed as well, under the TEST archive key and TEST transparency log;
  every other test ran on synthetic TDX evidence;
- the clock is the caller's: collateral windows, certificates, the
  reference's signature and its log checkpoint are judged against it, not
  against the `Date` of Intel's answers as the relay judges its own;
- the image reference is the caller's: whoever supplies it chooses which of
  the images x0ne signed is accepted, so it must come from the integrator,
  never from the carrier; it is admitted only signed by the archive key
  this build pins and shown in the log this build pins, by the relay's own
  steps, and held to the caller's memory of the log (the last checkpoint
  verified, returned to keep and handed back: same size, same root; a
  smaller one refused; a larger one only with a consistency proof the
  caller supplies); no checkpoint is fetched, so a fork is seen only once
  this caller met both heads, and a memory lost starts over; a build made
  before the signing ceremony admits none;
- the TCB policy is the relay's: `transparent` by default, accepted on
  Intel's standard collateral, `early` named in the disclosure and not
  judged; `strict` accepts on `early` alone and holds every receipt to it;
- the collateral is relayed: genuine collateral up to a day old (shorter on
  the caller's request), Intel's root CA CRL until it expires; a recovery or
  a revocation published since is not seen until relayed;
- the model is the one whose digest the caller names;
- the session config the quote binds names the model by its catalogue name,
  shown to the caller before sealing; nothing here maps that name to the
  digest the caller pins;
- Google's firmware binary is relayed in the collateral and measured by the
  SDK (its SHA-384 the one Google signed, its MRTD the quote's), its source
  unpublished; RTMR0 is pinned to the reference's host events as one boot
  showed them, not audited;
- the carrier, the relay included, sees the size bucket of each question
  and answer, their timing, which TD they go to (and so the model it
  serves), the key document and the caller's address; of a streamed
  answer, its count of chunks (its duration within a factor of two) and a
  reader that stops reading;
- the TD's OHTTP key is per boot and the same for every caller of that TD;
  nothing here shows two callers got the same key;
- physical attacks and side channels are out of scope.

A caller asks for `attest=cpu+gpu`, the one value a build without
`test-anchor` takes; under it (`GPU_LIMITATIONS`):

- prototype: the GPU path ran on one H100 in CC mode with NVIDIA's NRAS on
  2026-10-28, 2026-10-30, 2026-10-31, 2026-11-01 and 2026-11-04, under the TEST archive key
  and TEST transparency log; every other test ran on a simulated NVIDIA and
  NRAS;
- the H100's report names no TD (TEE.fail: NVIDIA reports "are not bound to
  the identity of a specific CVM"): it is tied to this one by the quote and
  by the nonce, not by NVIDIA's evidence; the nonce is checked by NRAS, and
  the reports bound being the ones read by the TD's measured code, not by
  the SDK;
- the SDK checks neither the bound certificate chains nor the bound reports:
  the GPU's identity rests on the leaf serial alone (review of 2026-10-27,
  finding 2; checking the chain to NVIDIA's device root and the report's
  signature and nonce in the SDK is left for later);
- NVIDIA's NRAS is trusted: it judges the reports, the chains, their
  revocation and the measurements against NVIDIA's RIMs; the SDK checks
  NRAS's signed verdict, not the reports;
- NRAS's token carries no digest of the report it judged;
- anchored at NVIDIA's intermediates 004 and 005, pinned whole, not at their
  root; revocation of the NRAS signing chain is judged from the OCSP
  responses the key document staples, none stapled yet (`gpu` above); the
  signing leaf's key usage, extended key usage and subject are not checked:
  NVIDIA publishes no profile for them (searched 2026-10-27);
- the GPU's identity is the leaf serial of a bound chain, NVIDIA's own
  derivation, matched by production tokens of 2026-10-14, -17 and -18;
- no NRAS claim names the confidential-computing mode: the TD reads no
  report unless NVML says CC is on, in production, devtools off, a check not
  yet measured on an H100; the SDK sees only NRAS's verdict;
- up to 8 GPUs, each attested alone: the protected PCIe (PPCIe) and NVSwitch
  attestation of several GPUs is not done, and nothing here covers the
  traffic between GPUs;
- NRAS's answer and its JWKS come in the key document, fetched by the TD
  for that document's nonce (relayed by the carrier for a TD before
  2026-10-28): neither can forge them, either can withhold them (refused);
  one NRAS call per key document, at whatever rate NRAS serves (not
  measured).

The first line of `LIMITATIONS` and of `GPU_LIMITATIONS` names the first run
on hardware, that of 2026-10-28 ("Through the relay"); the runs of
2026-10-30, 2026-10-31, 2026-11-01 and 2026-11-04 are in [hardware-runs.md](hardware-runs.md). A
build made before these lines changed, the one of that run included, still
says that neither path ran on hardware.

And, not in the list because it is the caller's, not the key's: a browser
checks nothing about the code a page handed it. The wasm module and the
loader can be replaced by whoever serves them. Signed distribution (npm,
crates.io, Sigstore provenance) and an extension are the answers, not this
crate.

## Running

```sh
cargo test -p x0ne-sdk                                   # vectors, gateway, receipts, ABI, real quote, log memory
cargo test -p x0ne-infer --lib -- e2e_sdk_tests          # against the simulated TD
cargo test -p x0ne-infer --lib -- e2e_td_tests           # the TD's routes, through a carrier
cargo test -p x0ne-infer --lib -- cloud_e2e_tests::the_relay_carries  # through the relay
make sdk-wasm                                            # wasm32 builds + Node run, no network
make e2e-local                                           # + wasm32 through the carrier, Qwen3 on CPU
```

`make sdk-wasm` builds `ci/Containerfile.wasm` once (the wasm32 standard
library; Node 22 from a local image by digest), then under the shared build
lock and with no network builds `x0ne-sdk` for wasm32 twice (with and without
`test-anchor`) and runs the `wasm_*` tests, which drive
`x0ne-sdk/js/smoke.mjs` in Node against the bench. Result in
`out/sdk-wasm/latest.json`.

`make e2e-local` (`scripts/e2e-local.sh`), in the same image, with no network
and its own target directory, builds the wasm32 SDK with `test-anchor`, runs
`e2e_td_tests`, then `e2e_td_tests::wasm` in Node and, when
`X0NE_E2E_QWEN3_DIR` names Qwen3-0.6B's `config.json`, `tokenizer.json` and
`model.safetensors`, `e2e_td_tests::qwen3` in a release build. Result in
`out/e2e-local/latest.json`, which names the model that ran.

## What is left, in order

1. The real-hardware rehearsal: the TD's routes on a TDX VM, carried by
   the relay, and a key document of theirs verified by the SDK under both
   TCB policies. Done on 2026-10-28 ("Through the relay"), then on
   2026-10-30 and 2026-10-31: accepted under `transparent`, refused under
   `strict` (76) on a host Intel's latest collateral rates OutOfDate, one
   question answered each time, under the TEST archive key and log. The
   SDK's acceptance under `strict` was not exercised on hardware.
2. The GPU path on an H100 in CC mode: the NVML report for `gpu_nonce`,
   NRAS's answer for it fetched by the TD, the time NVML and NRAS add to a
   key document, and the rate NRAS serves at one call per document. Done in
   code and on the simulated bench since 2026-10-27 (the verdict in the
   document since 2026-10-28). On 2026-10-28 the SDK accepted two key
   documents of a TD on an H100, each carrying NRAS's verdict on its GPU;
   the time and the rate were not measured. The NRAS signing chain's OCSP
   is judged from stapled responses since 2026-10-31, on synthetic
   responders; not done: the TD fetching and stapling them, and the
   production responder's behaviour (the leaf under 004 names no responder,
   the one under 005 names NVIDIA's Vault PKI without a responder URL).
3. The replay window on a clock that does not go back. Done on 2026-10-30
   ("Replay" above): the window is judged on the highest time the TD's
   clock has shown. Until then, an entry was dropped once the TD's clock
   passed its window, and with the clock set back into that window the
   request was answered again, a second size and timing sample for the
   host, still sealed to the caller.
4. The key document's binding and the TD's session config. Done on
   2026-10-30 ("Binding", "Version 2" and "`session`" above): the quote
   binds the boot binding, the session config's digest among it; the SDK
   returns the configured model, `expires_at` and the relay's key before
   anything is sealed, and stops a minute before the end, as the relay.
   Until then the binding named neither, and the caller saw none of them;
   the receipt named the model afterwards. Not done: the model's catalogue
   name is not mapped to the digest the caller pins.
5. One quote per key document: N callers cost N quotes, about one new
   caller per second per TD once the bucket is empty. Several callers'
   nonces in one quote (a Merkle root in `report_data`, each caller handed
   its path) is not done.
6. Chunked OHTTP for streaming, with the server's padded cadence. Done on
   2026-10-30 between the SDK and the TD ("Streaming"), on 2026-10-31
   through the relay and from the browser ABI ("Through the relay",
   "Browser"), on synthetic evidence; on 2026-11-01 on hardware, through a
   real relay and TD ([hardware-runs.md](hardware-runs.md)). Not done: a
   sealed stream opened in a browser tab (the SDK ran in Node), a test
   against another implementation of the draft.
7. A TDX module threshold, relayed and checked in the SDK as the relay
   checks it. The endorsed firmware binary (RTMR0 `gce-endorsed`) is
   relayed and checked since 2026-10-31.
   The reference's signature and log proof are checked since 2026-10-28;
   since 2026-10-31 the SDK keeps the last checkpoint it verified, as the
   caller's memory, and holds every new one to it ("The log's memory"),
   on the TEST log; it still fetches no checkpoint and no tile, so a grown
   log needs the consistency body from the integrator, and a fork is seen
   only by a caller who met both heads.
   Until x0ne's archive key and log are pinned (the signing ceremony) and an
   image reference is published under them, a release build of the SDK
   admits no reference: the path runs only under the TEST keys.
8. A third-party relay and anonymous tokens; a log of TD keys.
9. A hybrid post-quantum KEM, not implemented: the SDK seals with
   DHKEM(X25519, HKDF-SHA256) alone until HPKE's identifiers for one are stable.
10. THREAT_MODEL: the serving page, the caller's clock. The carrier
    of the sealed path has its row since 2026-10-27.
11. Signed distribution; the external audit reading `hpke.rs` (the RFC
    vectors stand for it until then).
