# Pinning an image

## Pinning an image

A TDX quote carries five SHA-384 registers, signed by the hardware. They are
in every verdict and every receipt under these names:

| Register | What it measures (TDVF convention) |
|---|---|
| `mrtd` | the initial TD image — the virtual firmware, which the **cloud** provides |
| `rtmr0` | firmware configuration |
| `rtmr1` | what the firmware loaded: bootloader, kernel |
| `rtmr2` | kernel command line, initrd |
| `rtmr3` | left to the guest; dstack extends it with the application |

An image *you* built is named by `rtmr1`/`rtmr2`, not by `mrtd`. The mapping
above is Intel's convention. On GCE `a3-highgpu-1g` it was measured on the
event logs of 2026-10-14 and 2026-10-17 (`x0ne-infer/tests/ccel/`, read by
`ccel.rs` and `reference.rs`): RTMR0 holds the TD HOB the host hands to the
firmware, the firmware's configuration volume, the Secure Boot variables,
the host's ACPI tables and the boot variables (16 events, the same on the
boots of 2026-10-14, -17 and -18; see [RTMR0 and Google's
firmware](#rtmr0-and-googles-firmware)); RTMR1, in this order, the EFI action "Calling EFI Application from
Boot Option", a separator, the disk's GPT, the UKI, the kernel PE it loads,
then the actions "Exit Boot Services Invocation" and "Exit Boot Services
Returned with Success" (the same order on three boots: 2026-10-14 attempts 3
and 9, the latter kept in x0ne-OS, and 2026-10-17); RTMR2 the UKI's sections and the kernel's command-line
and initrd tags; RTMR3 nothing. **Other clouds and machine types have not
been measured** — compare two boots of the same image, and one boot of a
changed image, before you rely on a register there.

**The published file is the default pin.** `measurements/<version>.json` at
the repository root is embedded in `x0ne-infer` at build time, for the
version it is built as:

```json
{
  "format": 1,
  "version": "0.2.0-beta.7",
  "algorithm": "sha384",
  "image": { "name": "…", "built": "YYYY-MM-DD", "source": "…" },
  "provenance": { "reference_signed_by": "…", "release": true },
  "registers": { "rtmr1": "<96 hex>", "rtmr2": "<96 hex>" },
  "signature": null
}
```

`registers` is what gets pinned. `provenance`, written by `make
run-fixtures`, names the archive key that signed the reference the run was
checked against (`test-archive-key`, or the release signing subkey's
fingerprint): a release build refuses `"release": false` (exit 87), a debug
build warns and goes on; a file without it is read as before. `signature` is reserved for a detached
signature by the release key (planned, not built): a file carrying one is
refused by this version rather than taken as signed. A file of another
version, another format, an unknown register or a malformed value is refused
(exit 87) in every build. With `registers` empty, a **release** build refuses
to serve and a release client refuses to verify (exit 87, the message says
how to pin); a debug build warns and goes on, and
the verdict says that nothing is pinned.

Both sides pin it, and either can override:

```bash
# Server: the published file, or another file for this version, or explicit
# registers (overrides the file). Protects an honest operator.
x0ne-infer …
x0ne-infer --measurements /path/to/0.2.0-beta.7.json …
x0ne-infer --expect-measurement rtmr1=<96 hex> --expect-measurement rtmr2=<96 hex> …

# Client: `x0ne-engine cloud` pins the image by its signed reference instead
# (docs/cloud.md). This is the pin an operator cannot switch off: the quote
# is verified on your machine, to the Intel root you pin.
```

**Regenerating the file after an image build**, without a cloud run of your
own if someone already booted it: fetch `/attestation?nonce=<64 hex>` from
the booted image, verify it (no command of this version verifies a
standalone `attestation.json`: `x0ne-decide-verify`, which did, left with the
verified decisions), then

```bash
x0ne-infer measurements attestation.json --image "x0ne OS 0.2 TDX guest, 2026-10-13" \
  > measurements/$(x0ne-infer --version | cut -d' ' -f2).json
```

prints the file for the binary's version with `rtmr1` and `rtmr2` read from
the quote (name other registers after the path to pin them too), and reads
it back before printing. Commit it with the release that built the image.

After a run of `x0ne-engine cloud`, the same file and the test fixtures of
that boot come from the attestation document the relay serves (save
`GET /x0ne/v1/attestation` during the run) and the image's v2 reference:

```sh
make run-fixtures RUN_ATTESTATION=attestation.json RUN_REFERENCE=x0ne-cvm-<V>.json \
  RUN_DATE=<YYYY-MM-DD> RUN_IMAGE="<image name>"   # RUN_WRITE=1 copies into the tree
```

It writes `measurements/<version>.json`, the log cut 256 bytes into its 0xFF
padding, the TD report and the reference under `x0ne-infer/tests/`, only if
the relay's report says VERIFIED and names this reference's SHA-256, the
quote verifies again to the pinned Intel root (offline, at this machine's
clock), the log replays to the quote, the reference passes that log, and
the release archive key signed the reference (`<reference>.sig`); then it
compares them with the tree. `RUN_TEST_KEY=1` accepts a reference the
release key did not sign: the first note says TEST and the measurements file
gets `"release": false`. On the document of 2026-10-17 and the v2 reference
of its image it gave back the four committed files byte for byte (measured
2026-10-17, before these checks). `RUN_WRITE=1`
replaces no file of the tree, the measurements of a run or the fixtures of
a `RUN_DATE` already taken, without `RUN_OVERWRITE=1`: it refuses and writes
nothing. A measurements file naming no image and no register is filled.
The registers change with every kernel, initrd or command-line change, so
the file is per image version — a mismatch after an update is the mechanism
working. The 2026-10-14 run measured its test image
(`measurements/0.2.0-beta.2.json`, RTMR1 and RTMR2, matched by the pinned
probe on that VM). `measurements/0.2.0-beta.5.json` holds the RTMR1 and RTMR2
of the 2026-10-17 run's quote, x0ne-cvm 0.2 (disk tar sha256 `48180d78…`);
they equal what x0ne OS (not public yet) `cvm/predict_measurements.py`
computes for that image from the 2026-10-14 log of attempt 9 (kept in
x0ne-OS; measured 2026-10-17). The prediction shipped with the image on the
day of the run had another RTMR1 (`2bf03587…`). `measurements/0.2.0-beta.7.json`
holds those of the 2026-10-25 run's quote, x0ne-cvm 0.2.4 (disk tar
`03d53a41…`), written by `make run-fixtures RUN_TEST_KEY=1 RUN_WRITE=1
RUN_OVERWRITE=1`; they equal the prediction shipped with that image's build
(compared 2026-10-25). Until then it held those of the 2026-10-18 run's
quote, x0ne-cvm 0.2.3 (disk tar `9537364d…`), equal to the prediction
shipped with that build (`measurements-predicted.json`, same predictor;
compared 2026-10-18). Neither file is signed. Image reproducibility was
measured on one machine, in two environments. On 2026-10-17, two x0ne-OS builds of 0.2
(`make test-cvm`, the second with the first's package cache and the same
`x0ne-infer` .deb) gave the same dm-verity root hash, UKI and disk tar. On
2026-10-22, a VM on the same host (Debian 12 and Linux 6.1 instead of
Debian 13 and 6.12, another CPU model, user ID, locale and time zone, empty
caches, every crate fetched again) rebuilt the three `x0ne-engine` packages
of one commit, and the x0ne-cvm 0.2.3 image with its `x0ne-infer` package,
disk tar (`9537364d…`), UKI, dm-verity root hash, reference and predicted
measurements: identical to the host's builds byte for byte (`make
repro-compare`, `docs/ci-local.md`), both with the build images loaded from
the host and with the build images rebuilt in the VM from their
Containerfiles. Both environments ran on one physical machine, for one
operator, from the same network sources (crates.io, a dated
snapshot.debian.org, NVIDIA's repository with pinned SHA-256); the CI
image's own apt packages are not pinned to a snapshot. A rebuild by a third
party, on another machine, has not been tested; until it is, the values are
ours to take.

**x0ne-cvm 0.2.4 (built and measured 2026-10-25).** x0ne OS
built it from this repository's 0.2.0-beta.7 sources of 2026-10-25, whose
binaries the one release recipe builds and strips (`docs/ci-local.md`):
disk tar `03d53a41…`,
dm-verity root hash `50bc36b2…`. Its build predicts RTMR1 `2d014fbe…` and
RTMR2 `37c3c7c5…`, the same from the logs of 2026-10-14 and 2026-10-17
(`x0ne-cvm-0.2.4.measurements-predicted.json`, the predictor of 0.2.3). One
GCE boot on 2026-10-25 quoted those two values (`docs/hardware-runs.md`),
and `measurements/0.2.0-beta.7.json` names them since, with TEST provenance.
The built-in file cannot name the image that carries it: it is compiled
into `x0ne-infer`, the image's dm-verity tree holds that binary, and the
tree's root hash is on the kernel command line that RTMR2 measures
(`.cmdline` and the LoadOptions tag). The `x0ne-infer` of 0.2.4 is the one
built from the commit its reference names (`image.source.engine`), and it
carries the file of 0.2.3 (measured 2026-10-25); `make run-fixtures
RUN_OVERWRITE=1` wrote the file that names 0.2.4 in a later commit. Built
before that commit, `x0ne-infer` outside cloud mode pins 0.2.4 only with `--measurements
x0ne-cvm-0.2.4.measurements-predicted.json`, a file of this version with no
`provenance`; `x0ne-engine cloud` pins the image by its reference instead.

**Transparency log of image references (built, no log runs yet).** Since
2026-10-22 the relay takes an image reference only with a C2SP
`tlog-proof@v1` next to it, showing the reference's own bytes in the
transparency log its build pins, in a checkpoint signed by that log and
cosigned by a quorum of pinned witnesses, and consistent with the last
checkpoint that machine verified; otherwise it exits 100 (`docs/cloud.md`,
"The transparency log"). x0ne-sdk, which reads no file, holds the proof's
checkpoint to the caller's memory of the log instead, the last checkpoint
it verified, kept by the caller (`docs/sdk-e2e.md`, "The log's memory"). The
log makes a reference signed for one user
alone visible to everyone; it does not make it good. This tree pins no log:
x0ne runs no public log and no witness yet, and a release relay refuses
every reference until one is pinned. It was tested against a local TEST log
and a local instance serving it over HTTP, plus a log written by Tessera
(Rekor v2's log library). On the GCE run of 2026-10-25, a relay built
`--release` with debug assertions and `test-archive-key` checked the 0.2.4
reference against the TEST log on disk next to it (index 0 of a log of size
1). No public log has been read.

### RTMR0 and Google's firmware

On GCE the cloud provides the virtual firmware. Google publishes the binary;
neither google/gce-tcb-verifier nor GCP's "Verify Confidential VM firmware"
page (both read 2026-10-22) points to its source or to a way to rebuild it.
What the client checks of the firmware, and from what (`gce_endorsement.rs`,
`tdvf.rs`, `reference.rs`):

- **MRTD, always.** The quote's MRTD must be listed in a launch endorsement
  Google signed under the pinned `GCE-cc-tcb-root_1`.
- **The binary, under `rtmr0.policy: "gce-endorsed"`.** The endorsement also
  signs the SHA-384 of the firmware binary, which Google publishes in the
  same bucket (`ovmf_x64_csm/<sha384>.fd`). The client fetches it, checks
  that its SHA-384 is the signed one, reads its TDVF metadata and computes
  the MRTD the TDX module measures when a host loads it, as
  google/gce-tcb-verifier does (`tdx/measurement.go`): it must be the
  quote's. Measured on 2026-10-22: Google's binary `53c558fe…`, which its
  endorsement of MRTD `c1ee9c16…` names, measures `c1ee9c16…`, the MRTD the
  GCE boots of 2026-10-17 and -18 quoted, in `tdvf.rs` and in an independent
  Python implementation (`x0ne-infer/tests/tdvf/tdvf_reference.py`,
  `x0ne-infer/tests/gce_firmware_live.rs`).
- **RTMR0, under `gce-endorsed`.** Every event that extends RTMR0, in log
  order, must be the one `rtmr0.events_sha384` lists for the machine type.
  The entry `"firmware:cfv"` stands for the SHA-384 of the binary's
  configuration firmware volume (CFV, the UEFI variable store). The host
  loads the CFV outside MRTD (no `EXTEND_MR` on it in Google's binary), and
  the firmware measures it into RTMR0: `4e1221cf…` for `53c558fe…`, the
  second RTMR0 event of the three GCE logs. The other entries are the
  host's, as one boot showed them: the TD HOB (the memory map the host hands
  the firmware), the Secure Boot variables (SecureBoot, PK, KEK, db, dbx), a
  separator, three ACPI tables and five boot variables. The three GCE boots
  of 2026-10-14, -17 and -18 logged the same 16 events and quoted the same
  RTMR0. An event changed, missing, added or moved refuses (65), and so does
  a binary absent from the bucket; a bucket that cannot be read refuses with
  76. `gce-endorsed` is the only policy the relay and the SDK admit since
  2026-10-31: a `report` reference is refused when loaded (104), before
  anything is created. Until then it was replayed and reported under
  `report`, and no binary fetched; the lab's replay of a recorded run
  (`run_fixtures.rs`, `reference::check_image_rtmr0_reported`) is the one
  path that still reads a `report` reference, and it verifies nothing live.

What `gce-endorsed` removes from the trust in Google:

- Google's word on the MRTD: the client computes the MRTD from the binary
  whose SHA-384 Google signed, so an endorsement that lists an MRTD its own
  binary does not measure is refused;
- the host, which is Google's, on what it feeds the firmware outside MRTD,
  between the boot the reference was taken from and yours: another
  configuration volume, ACPI table, memory map, Secure Boot variable or boot
  order changes RTMR0, and the client refuses before the key leaves. Trail
  of Bits found the ACPI tables of WhatsApp's private inference "weren't
  included in the attestation measurement" (TOB-WAPI-17, blog post of
  2026-04-07); on GCE they are in RTMR0, and `gce-endorsed` is what makes
  the client compare them;
- an older firmware Google once endorsed, when the reference sets
  `mrtd.min_svn`: the endorsement's TDX SVN must reach it (65), under
  `report` and `gce-endorsed` alike. Absent or 0, there is no floor; the
  firmware of the boots measured has svn 2.

What remains:

- the firmware's code. The client checks that the binary is the one Google
  signed and the one the TD runs, not what it does; its source is not
  published, and a flaw or a backdoor in it is not detected;
- Google's signing keys: whoever holds them can endorse any binary, and
  the binary then passes as Google's;
- the SVN is Google's number: `mrtd.min_svn` refuses what Google itself
  superseded, not a flaw Google has not numbered;
- the host's values are pinned as one boot showed them, not audited: a
  hostile ACPI table already present at that boot is pinned with the rest;
  the TD HOB is pinned by its digest, not rebuilt from the machine type's
  memory;
- Secure Boot was off on the boots measured (the SecureBoot variable is 0):
  RTMR0 pins that state, it does not turn Secure Boot on;
- availability: the binaries are in one Google bucket, "only hosted in one
  region, us-west1" and "a best-effort backup established for transparency
  purposes" (GCP docs, 2026-10-20); when it cannot be read, a
  `gce-endorsed` session does not start (76);
- one machine type measured, `a3-highgpu-1g`; another needs its own list;
- Google's older measurements that put the RAM size into the MRTD
  (`ram_gib` other than 0) are not reproduced: such a firmware is refused
  under `gce-endorsed` (65);
- the reference decides which events are pinned. The client's own
  reference sets them, never the TD.

**No published reference uses `gce-endorsed` yet**: the x0ne-cvm 0.2.x
references published so far are `report`, which the relay and the SDK
refuse since 2026-10-31 (104). The first `gce-endorsed` reference, x0ne-cvm
0.2.8's, is signed by the TEST key and not published; under it the relay
and the SDK pinned RTMR0 on a GCE TD on 2026-10-31
(`docs/hardware-runs.md`). The fixture
`x0ne-infer/tests/run_fixtures/x0ne-cvm-0.2.3-rtmr0-pinned.json`
is the 0.2.3 reference with RTMR0 pinned by the procedure below from the
boot of 2026-10-18 and Google's binary `53c558fe…` (unsigned: the SDK's
tests sign it with the TEST key; `run_fixtures.rs` checks it differs from
the published 0.2.3 in `rtmr0` alone). A firmware update that changes only the binary
needs no new reference, since the CFV comes from the newly endorsed binary;
GCE upgrades the firmware on reset (GCP docs). An update that changes a host
event (another ACPI table, another memory map) is refused until a reference
lists it. To write or renew the list:

1. boot the image once on the machine type and keep the attestation
   document; its `tdx.ccel_b64`, base64-decoded, is the event log;
2. fetch the binary its MRTD's endorsement names
   (`https://storage.googleapis.com/gce_tcb_integrity/ovmf_x64_csm/<sha384>.fd`);
3. run

   ```sh
   cargo run -p x0ne-infer --example gce_firmware -- <binary.fd> <ccel.bin>
   ```

   It prints the binary's SHA-384, its TDVF sections, the MRTD and CFV it
   measures, and the `rtmr0.events_sha384` list, which fails if the CFV is
   not in the log exactly once;
4. check that the printed MRTD is the quote's and the SHA-384 the
   endorsement's, compare the list with the previous one and say why each
   entry changed, raise `mrtd.min_svn` to the endorsement's svn if Google
   superseded the previous firmware, then sign the reference with the
   archive key.
