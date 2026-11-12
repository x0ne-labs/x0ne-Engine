# `x0ne-engine cloud`: one confidential GPU VM per session

`x0ne-engine cloud up` rents one Google Cloud VM for one session:
`a3-highgpu-1g` spot, which is one H100 80GB with Intel TDX. The attested relay
checks the VM's hardware proof **on your machine**. Only after that does it serve
an OpenAI-compatible API on `127.0.0.1`. Ctrl-C ends the session and deletes the
VM. `x0ne-engine cloud run <model>` is the same session with a chat in the
terminal (see [`run`](#run-a-chat-on-the-session)).

**What is protected, and measured.** The relay releases the session key and
sends a question only after a VERIFIED verdict, and only over mutual TLS 1.3
bound to the VM's TDX quote; the section after
[What it does not protect](#what-it-does-not-protect) says, segment by segment,
where the text is in clear. Verdict VERIFIED on
Google Cloud, TDX and one H100 in CC mode, on 2026-10-18 (`qwen3:32b` served on
the H100) and 2026-10-25 (`qwen3:0.6b`), with a relay built for testing
([hardware runs](hardware-runs.md)). This client ran against Google Cloud on
2026-10-17, `image import` then `run qwen3:8b` of 0.2.0-beta.5,
`a3-highgpu-1g` spot in us-central1-a, before the reference signature check;
and on 2026-10-25, `up` on two VMs whose TD refused its host's TCB level, each
VM deleted by `up`. Everything below is tested against a fake Compute Engine
on the loopback and fake relays (`x0ne-engine-cloud/tests/`). The points
marked *unverified* have not been checked against the real service. The
costs printed are estimates: those runs read no bill.

**What no TEE protects.** Physical attacks on memory, side channels, a
compromise of Intel's, NVIDIA's or Google's roots, availability and what the
traffic shows: [What it does not protect](#what-it-does-not-protect) lists
them, with this path's own gaps.

> **Status: a session does not start from this source tree.** Its relay
> pins no archive key, so it refuses every image reference and exits 99
> (measured on 2026-10-24), and it pins no transparency log, which a relay
> with the key refuses with 100. `up` asks the relay first and stops there,
> before the token. No `x0ne-cvm` image is published to import:
> [x0ne OS](https://x0ne.co/x0ne-os/) builds one. What follows describes
> the code and its tests.

This is **not end-to-end confidential**: see
[what that covers](#what-not-end-to-end-confidential-covers), segment by
segment, and [What it does not protect](#what-it-does-not-protect). The path
meant to lift it is the caller sealing its question to the TD; what that path
still needs, in order, is in [docs/sdk-e2e.md](sdk-e2e.md#what-is-left-in-order).

## Two processes, two secrets

| process | holds | never sees |
|---|---|---|
| `x0ne-engine-cloud` (this crate): the lifecycle | the cloud access token | questions, answers, the session key |
| `x0ne-engine-proxy` (`x0ne-infer`): the attested relay | the client key, your questions | the cloud token |
| `x0ne-engine-proxy chat`, started by `run` or `chat` | the local API key, your questions and the answers | the cloud token, the session key |

The lifecycle quotes, creates, watches and deletes the VM. The relay verifies
the proof, releases the session key to the verified VM, and serves the local
API. They talk over the relay's stdin and stdout in JSON lines (protocol 1, see
[The relay protocol, lifecycle side](#the-relay-protocol-lifecycle-side)).

`x0ne-engine cloud …` runs `x0ne-engine-cloud` from the directory of
`x0ne-engine`, or else from `/usr/libexec/x0ne-engine/`. The relay is looked up
the same way. **`$PATH` is never searched** for either binary. The file found
must be executable and must not be writable by group or others.

## Building

No package is published. The three client binaries build from this tree into
one directory, where `x0ne-engine` finds the other two:

```sh
cargo build --locked --release -p x0ne-infer --bin x0ne-engine --bin x0ne-engine-proxy -p x0ne-engine-cloud --bin x0ne-engine-cloud
target/release/x0ne-engine cloud --help
```

`make ci` runs these two lines offline (step `build-cloud`), as the GitHub
workflow does. The release profile keeps symbols and line tables:
`x0ne-engine`, `x0ne-engine-proxy` and `x0ne-engine-cloud` weigh 133, 142 and
37 MB as built, 5.9, 10.3 and 5.6 MB once `ci/strip-release.sh <debug dir>
<binary>...` has moved them to debug files, as the packages do (measured on
2026-11-03, [ci-local.md](ci-local.md#built-from-source)). `make deb` packages them locally, given an x0ne OS checkout
(`X0NE_OS`, not public yet); nothing is published ([ci-local.md](ci-local.md)).

## Usage

```
x0ne-engine cloud quote                quote only; creates nothing
x0ne-engine cloud up                   quote, "yes", create the VM, check its hardware
                                       proof here, then serve an OpenAI API on the
                                       loopback; Ctrl-C stops and deletes the VM
x0ne-engine cloud run <family:tag>     the same for one model, then a chat in this
                                       terminal; /exit, Ctrl-D or Ctrl-C ends the
                                       session and deletes the VM
x0ne-engine cloud chat                 chat with the session serving on this machine
                                       (another terminal); leaving does not end it
x0ne-engine cloud status [--limits]    the local session (no token needed)
x0ne-engine cloud down                 stop the session and delete its VM (another
                                       terminal, or after a crash)
x0ne-engine cloud sweep                list x0ne-engine leftovers in the project,
                                       delete them after "yes"
x0ne-engine cloud image import <disk.tar.gz>
                                       once per project: import the CVM image
x0ne-engine cloud image delete <x0ne-cvm-…>
                                       delete an image `image import` made, after
                                       "yes"; refused while a VM boots from it
x0ne-engine cloud setup gcp            how to install gcloud on x0ne OS
x0ne-engine cloud login gcp            gcloud auth login --no-launch-browser
```

| option | default |
|---|---|
| `--provider gcp\|azure` | `gcp`; `azure` is refused, see [Azure](#azure) |
| `--project <id>` | `$GOOGLE_CLOUD_PROJECT`; with `--auth gcloud`, gcloud's default project |
| `--zone <zone>` | the first of `europe-west4-c`, `us-central1-a`, `us-east5-a` with free H100 spot quota; with `--auth gcloud`, gcloud's default zone if it is one of these |
| `--region <region>` | `image import` only; `us-central1` |
| `--model <family:tag>` | `up` only; `qwen3:0.6b`. `run` takes the model as its argument. Both refuse a model that is not in this build's catalog or has no served-model digest pinned there (exit 2, nothing created) |
| `--max-run <30m\|2h>` | `60m`; from 10 min to 8 h. The VM deletes itself after this. `up` and `run` refuse a value under the boot timeout plus 5 min (exit 2). With `--auth paste` or `stdin`, the session ends 5 min before the token expires, and a token that leaves too little is refused (exit 91): see [Expiry](#the-token) |
| `--auth paste\|stdin\|gcloud` | `paste` |
| `--listen <ip:port>` | `127.0.0.1:8478`; loopback addresses only |
| `--reference <file>` | the image reference values you chose, signed by the x0ne archive key in `<file>.sig` (see [The image reference](#the-image-reference)); its `image.gcp_image` picks the image. Without it the relay takes the newest installed reference, and `quote`, `up`, `run` and `image import` name the image that reference names; with no reference installed, `x0ne-cvm-0-2-1` (`DEFAULT_IMAGE`), which the relay then refuses (89) |
| `--yes` | ask nothing; `up --yes` and `run --yes` also need `--max-cost` |
| `--max-cost <USD>` | refuse any quote above this |
| `--max-tokens <N>` | `run` and `chat`: tokens per answer; `1024`, at most `4096` (the TD's ceiling) |
| `--key-file <file>` | `chat` only: the local API key; default: the one the serving session names |
| `--allow-from <CIDR>` | `up` and `run`: who may open a connection to the VM's tcp:443, an IPv4 address (`a.b.c.d`, a /32) or network (`a.b.c.d/n`, host bits zero), repeatable; default: this machine's public IPv4 address, /32, see [The firewall rule](#the-firewall-rule). An IPv6 or malformed value is refused without being printed (exit 2); any other command refuses the option (exit 2) |
| `--nras-key-file <file>` | `up` and `run`: an NVIDIA API key for NRAS, read when the command line is: a regular file you own, with no access for group or others (mode 0600 or 0400), one token of visible ASCII. Handed to the VM in its own metadata item; see [The NRAS key](#the-nras-key). A refused file is exit 2, saying neither its path nor its content; a key typed in place of the path is refused unprinted; any other command refuses the option (exit 2). Default: none, NRAS is called without a key |
| `--tcb-policy <mode>` | `up` and `run`: `transparent` (default) accepts a platform on Intel's `standard` collateral and names, on the VERIFIED line and in each receipt, its TCB date and the advisories Intel's latest collateral lists against it, without refusing; `strict` refuses a platform Intel's latest collateral does not rate UpToDate or SWHardeningNeeded (exit 76), and ends the session on a receipt that names one. Any other command refuses the option (exit 2). See [limitations](limitations.md), "TCB policy" |
| `--min-tcb-evaluation <N>` | `up` and `run`: implies `--tcb-policy strict` (with `transparent`, exit 2); the oldest Intel TCB evaluation (TCB recovery) the relay accepts in the TD's latest tcbInfo and QE identity; `22`, Intel's recovery of 2026-08-11. A lower value is refused (exit 2, nothing created): the floor is raised, never lowered. Any other command, which starts no relay, refuses the option (exit 2) |
| `--max-collateral-age <s>` | `up` and `run`: how long before the verdict Intel may have issued the tcbInfo, the QE identity and the PCK CRL; `86400`. A longer value is refused (exit 2, nothing created). Any other command refuses the option (exit 2) |

A first session needs the image and its signed reference. No package installs
a reference (`ci/deb/*.ctl` here and x0ne-OS `packages/*.ctl`, read on
2026-10-17; x0ne OS is not public yet). They come together from the `cvm/` directory of an x0ne OS release tree,
signed by the archive key's subkey (x0ne-OS `docs/archive-key.md`, not public yet):
`x0ne-cvm-<V>.tar.gz`, the reference `x0ne-cvm-<V>.json` and its
`x0ne-cvm-<V>.json.sig`. A reference you build yourself (x0ne-OS `make cvm`, not public yet)
is not signed by that key, and the relay refuses it (99). No such release
tree is published yet (Status, above): the `image import`, `up` and `run`
lines below need files that cannot be obtained today, and they are given as
what a first session will take. Keep the `.sig` next to the reference, then,
for 0.2.1:

```sh
x0ne-engine cloud image import x0ne-cvm-0.2.1.tar.gz --reference x0ne-cvm-0.2.1.json   # once per project
x0ne-engine cloud quote
x0ne-engine cloud up --reference x0ne-cvm-0.2.1.json            # then use http://127.0.0.1:8478/v1
x0ne-engine cloud run qwen3:0.6b --reference x0ne-cvm-0.2.1.json  # or: the same, with a chat here
```

To drop `--reference`, install those two files, owned by root and writable
by root alone, in `/usr/share/x0ne-engine/reference/`. The relay takes the
newest `x0ne-cvm-<version>.json` there, and `up` starts the image it names
(since 0.2.0-beta.6; before, `x0ne-cvm-0-2-1` whatever that version was).

## What `up` does

Before anything is created:

1. It finds the relay (see above) and asks for its pins
   (`x0ne-engine-proxy pins`): a relay built without an archive key (99) or
   a transparency log (100) refuses every session, and a relay that pins the
   TEST ones (a debug build, see [The image reference](#the-image-reference)) would
   accept a reference anyone can sign, so `up` exits 95 there, before the
   token. Any other answer, or none within 10 s, lets `up` go on:
   the relay checks its pins again when the session starts. Then it takes
   the session lock. There is one session per machine.
2. It checks the token with Google's tokeninfo endpoint, then prints who it
   belongs to and how long it has left. A token it cannot renew (`--auth
   paste`, `stdin`) cuts `--max-run` to end 5 min before its expiry, said on
   the next line, or is refused there (exit 91), before anything else is read
   (see [The token](#the-token)). A token Google rejects stops there, exit 91
   with "Google refused the token"; no answer from tokeninfo (offline, DNS,
   TLS) is exit 91 too, said as a network failure, not as a verdict on the
   token.
3. Preflight:
   - free spot H100 quota in the candidate zones: the
     `PREEMPTIBLE_NVIDIA_H100_GPUS` metric of `regions.get`; when that list
     does not carry it (none of us-east5, europe-west4, us-central1 did on a
     real project, 2026-10-17), the Cloud Quotas value of
     `PREEMPTIBLE-NVIDIA-H100-GPUS-per-project-region` for the region, else its
     default entry, with the H100 GPUs of the project's VMs in the region as
     usage. A Cloud Quotas answer of 400, 403 or 404 is a warning, not a refusal:
     Google refuses the create if there is no quota, and nothing is billed;
   - in the zone picked, `PREEMPTIBLE_CPUS` and, for a machine type with local
     SSD (a3-highgpu-1g: 26 vCPU, 750 GiB), `PREEMPTIBLE_LOCAL_SSD_GB` cover the
     machine when `regions.get` lists them; a zone short of either is skipped;
   - no `x0ne-engine` VM already running in the project;
   - the image exists and is `READY`; when a reference names it (given or
     installed), the disk sha256 `image import` recorded in its description
     equals `image.disk_tar_sha256`, or exit 91 before the quote;
   - the firewall rule's sources, printed on a `Firewall:` line: the
     `--allow-from` ranges, else this machine's public IPv4 address as a /32,
     asked of Google's name server; none found is exit 91 before the quote
     (see [The firewall rule](#the-firewall-rule)).
4. It prints the quote (see [The cost guard](#the-cost-guard)), then asks
   `Type "yes" to create the VM:` on the terminal. **Anything but `yes` creates
   nothing:**
   - another answer, or no terminal to ask on: exit 90;
   - Ctrl-C at the prompt: exit 130;
   - a token `up` cannot renew is read again after "yes": a wait at the
     prompt that leaves it too little is exit 91.

After "yes":

5. It starts the relay. The relay first checks the image reference's signature
   ([The image reference](#the-image-reference)), then emits the VM
   configuration and the image its reference names. The lifecycle checks
   the configuration's SHA-256, model and expiry, and that the image and its
   `disk_tar_sha256` are the ones checked at step 3, before it uses the
   configuration; another image, or none named, is exit 91 and nothing is
   created.
6. `[1/5]` It creates a firewall rule (tcp:443 from the sources of step 3 to
   this VM's network tag only), then the VM, which is:
   - spot, confidential type TDX;
   - deleted automatically at the end of `--max-run` (`maxRunDuration` and
     `instanceTerminationAction: DELETE`);
   - given the relay's configuration byte for byte as metadata `x0ne-config`;
   - with `--nras-key-file`, given the key as metadata `x0ne-nras-key`, a
     separate item ([The NRAS key](#the-nras-key)).

   The exact request bodies are in `x0ne-engine-cloud/tests/golden/`.
7. `[2/5]` It waits for `RUNNING` and an external IP address, for at most
   the boot timeout after the creation: 12 minutes, plus the model's catalog
   bytes at 40 MB/s (the TD downloads and hashes it before it answers). That
   is 12.6 min for `qwen3:0.6b`, 18.8 min for 16.4 GB (`qwen3:8b`, so
   `--max-run` of at least 24 min) and 39.3 min for 65.5 GB (`qwen3:32b`,
   `--max-run` of at least 45 min); the rate is an assumption, not measured
   on a TD. Before it answers, the TD hashes each weight file once, on the
   mapping it serves: that pass is the check against the catalog and the
   file's line of the served digest (two passes until 0.2.0-beta.7). The
   small files (`config.json`, `tokenizer.json`, the index) are still hashed
   by `pull` and again for the served digest. The rate does not count the
   passes separately.
8. `[3/5]` It sends `connect` to the relay. **The relay fetches the proof and
   checks it on this machine. No question and no session key are sent to the
   VM before its verdict.**

   Until the verdict, `up` also reads the VM's serial console every 15 s
   (`instances.getSerialPortOutput`, port 1, each read from where the last
   one ended). x0ne-OS writes a status block there every 30 s when it
   changed: x0ne-infer's last exit code and how many times systemd restarted
   it, systemd's fields only since x0ne-cvm 0.2.7 (images 0.2.4 to 0.2.6
   also copied the end of x0ne-infer's journal, its line included). A TD
   that refuses to serve exits, and systemd starts it again 10 s later; the
   relay only sees a TD that does not answer. When a block shows a final
   refusal after at least two restarts, `up` prints the exit code and what
   it means, or the line x0ne-infer printed before that exit when an older
   image's block carries it (control and invisible characters removed, cut
   at 300 characters), saves the console (see [Files](#files)), deletes the
   VM and exits 97. The final refusals are a list in
   `x0ne-engine-cloud/src/td_console.rs`, each code with its reason:
   - 2, 65, 66, 68, 80, 81, 82, 85, 86, 95, 101 (the NRAS key file refused;
     a panic of x0ne-infer also ends with 101) and 103 (NRAS refused the
     key): the same image, configuration and machine give the same refusal
     at every start;
   - 76 only when the line is Intel's rating of the host: a TCB status not
     accepted, SVNs or a TDX module below every published level, a module
     tcbInfo has no identity for, an evaluation under the floor. 76 also
     stands for Intel's PCS or a CRL unreachable, or collateral too old,
     which may pass;
   - 77 only when the line is a served digest other than the catalog's pin.

   A block of x0ne-cvm 0.2.7 carries no line, so 76 and 77 wait there. Every
   other code waits as before, NRAS unreachable (75) and a failed pull (96)
   among them, unless systemd gave up on x0ne-infer: `ActiveState=failed
   SubState=failed` after an exit code other than 0 ends the wait whatever
   the code and the restarts, and `up` says systemd stopped restarting it.
   An image whose unit sets `RestartPreventExitStatus=` prints that after the
   codes it lists; systemd's start limit prints the same fields (systemd
   257.13, `Result=exit-code`, 2026-10-30), which the unit's 10 s between
   starts cannot reach under the default of 5 starts in 10 s. For a code the
   list does not name, `up` prints the restarts and the last exit. The
   console is unverified text: the VM writes it, and the host can rewrite
   it. It can end the wait and nothing else; it never
   accepts a VM, shortens a check or gives a value to the relay. A forged
   refusal ends the session, which the host can already do by stopping the
   VM. Three failed reads in a row stop the reading, with one note, and the
   wait goes on as before. The read needs the IAM permission
   `compute.instances.getSerialPortOutput`, which `roles/compute.viewer`,
   `roles/compute.instanceAdmin.v1` and `roles/compute.admin` include
   (Google's IAM roles reference for Compute Engine, read on 2026-10-25); the
   boot timeout used it before, to save the console.
9. `[4/5]` The relay releases the session key to the verified VM.
10. `[5/5]` The API is served on `--listen`. Under that line, `Context: N
    tokens, prompt and answer together.` gives the TD's bound: the model's
    context, cut to what fits in the GPU memory left once the weights are
    loaded (`/v2/ready`; absent from a TD before 0.2.0-beta.6). A prompt of N tokens or more is refused (413) and a
    completion stops at N; `/x0ne/v1/status` carries it as
    `model.context_tokens`. It serves until one of these: Ctrl-C,
    Ctrl-\ (SIGQUIT), SIGTERM, SIGHUP or another signal that would end the
    process (SIGUSR1, SIGALRM, SIGPWR, the real-time signals...),
    `x0ne-engine cloud down`, the end of the session (`--max-run`, or 5 min
    before a token `up` cannot renew expires), the relay ending, or the
    VM disappearing (spot preemption, checked every 30 s; less often while
    Google does not answer, the pause doubling up to 8 min, and `up` says
    when it cannot see a preemption and when it can again). `serving` is not
    waited for past the end of the session either (exit 93). These do not stop
    the session: SIGKILL; SIGSTOP, which freezes `up`; a fault (SIGSEGV,
    SIGBUS, SIGILL, SIGFPE); SIGTRAP, SIGSYS and SIGABRT; and signal 33, which
    glibc keeps for itself (SIGSETXID: its `sigaction` refuses it). Those that
    end `up` leave the VM to `--max-run` (see below).
11. It stops the relay: `stop`, end of input, 10 s, then SIGKILL. It deletes the
    VM and the firewall rule, confirms that both are gone, and prints the
    estimated cost. A delete that fails in passing (no answer, 408, 429,
    5xx) is asked again, up to four times; each resource is read back, and
    one still there is deleted again. A panic of `up` itself after the
    creation started goes through the same deletion before it ends `up`
    (exit 101).

## The firewall rule

Since 2026-10-30 the rule admits tcp:443 to the session's VM from this
machine's public IPv4 address alone, as a /32. Before, it admitted
`0.0.0.0/0`: anyone who found the address could open connections, which
mutual TLS refused, each at the cost of a handshake slot of the TD
(THREAT_MODEL, the row "Anyone on the Internet who finds the TD's address").

How the address is found. After the preflight and before the quote, `up`
sends one DNS query over UDP to Google's name server `ns1.google.com`, on
its IPv4 address: the TXT record of `o-o.myaddr.l.google.com`, which that
server answers with the address the query came from (`src/myaddr.rs`). A
recursive resolver would answer with its own address, so the query goes to
that server directly. Google sees this machine's address and the session's
times anyway, through the relay's connections to the VM; no other party is
asked. Google does not document that name: on 2026-10-30 it answered the
same address as an HTTPS echo service, over UDP and over TCP. The answer
must be a public unicast IPv4 address; a private, loopback, link-local,
carrier-grade NAT (`100.64.0.0/10`), multicast or reserved one is refused.
No answer within 3 s, twice, or a refused one, is exit 91 before the quote:
nothing is created, and nothing falls back to `0.0.0.0/0`.

Nothing authenticates that answer: it is plain DNS over UDP. Someone who
cannot see the query must forge the server's address, a random 16-bit id and
an ephemeral source port within 3 s. Someone on the path answers as the server
does, with any public address: the `Firewall:` line prints it, the rule admits
it, and the relay's own connections then come from an address the rule does
not name. The rule narrows who can reach the TD's handshake; mutual TLS and
the attestation do not depend on it. `--allow-from` asks nobody.

`--allow-from <CIDR>`, repeatable, names the sources instead, and nothing is
asked. Give the range your network's outgoing address moves in (a
carrier-grade NAT with a pool, several uplinks, a VPN you turn on and off),
or your address when the network blocks DNS to the Internet.
`--allow-from 0.0.0.0/0` is accepted and printed in yellow: the rule is then
what it was before 2026-10-30.

When this machine's address changes during the session (another network, a
VPN turned on or off, a new address from the provider), Google's firewall
drops the relay's new connections to the VM. The relay opens a connection
per exchange with the TD; the first one from the new address times out after
10 s (`remote_verify.rs`, `CONNECT_TIMEOUT`), the local API answers it 502
`attested VM unreachable`, marks the session lost, and answers every later
request 503 `the attested VM is lost`, even once the address is back. `up`
neither watches the address nor changes the rule: the VM keeps running, and
billing, until `--max-run`, Ctrl-C or `down`. Stop the session and start
another from the new address, or with `--allow-from` covering both.

## The NRAS key

The TD's GPU verdict is a token from NVIDIA's Remote Attestation Service
(NRAS). Today the TD calls it with no credential. NVIDIA's deployment guide
says "API key authentication will be required in the future for all
production deployments" and calls the keys "currently an optional step"
(<https://docs.nvidia.com/attestation/poc-to-production/latest/deployment_guide.html>,
"Step 1: Set Up NGC API Keys", page updated 2026-10-06). NRAS v4.0.6
(2025-08-05) takes the key as `Authorization: Bearer <key>`
(<https://docs.nvidia.com/attestation/cloud-services/latest/nras/nras_releases.html>).

`--nras-key-file <file>` gives that key:

- `up` reads the file with its command line, before any call. The key goes
  in the VM's metadata item `x0ne-nras-key`, never in `x0ne-config`: the
  config's bytes are hashed into the TD's quote and published in its
  attestation document, its key documents and the relay's refusal files.
- In the VM, `x0ne-infer --nras-key-file <file>` reads it before any evidence
  (exit 101 on a refused file) and sends it as `Authorization: Bearer` on the
  attest call; the JWKS fetch stays anonymous. A refusal NRAS sends back has
  the key replaced by `(key not printed)` before it is printed. NRAS
  answering 401 or 403 to a call that carried the key exits 103: the image
  passes the same key at every start, so `up` ends its wait on it (see
  [What `up` does](#what-up-does)). Without a key, a 401 or 403 stays 75,
  as do NRAS's 5xx and a network error.
- Both read the same key: one token of 16 to 1024 characters of visible
  ASCII, one final newline allowed. A shorter key is refused: the redaction
  above replaces each copy of the key in NRAS's words, and a key of a few
  characters would mask ordinary text.
- **The image passes it since x0ne-cvm 0.2.7.** Its `x0ne-cloud-config`
  reads the metadata item `x0ne-nras-key`, writes it to `/run/x0ne/nras-key`
  (tmpfs, mode 0640, group `x0ne-infer`) and starts x0ne-infer with
  `--nras-key-file` on that file; its boot report prints `nras_key=` with
  the file's mode and owner, or `absent`, on the serial console, never the
  key. An older image reads `x0ne-config` only: with it the key reaches the
  VM's metadata and NRAS is still called without it. `up` says so on an
  `NRAS:` line.
- Neither `up`, the relay nor x0ne-infer prints the key or writes it to a
  file: the tests look for it in the output, the relay's argv and logs, every
  file of the session, and the error Google returns when it quotes a value
  it refuses (`tests/fake_gcp.rs`, `x0ne-infer/tests/argv_not_echoed.rs`).

The key is not protected from Google: instance metadata is stored by the
cloud provider and readable by anyone with `compute.instances.get` on the
project, while the VM lives. It is a credential to NVIDIA's service, not a
key to your data; use one made for this, and revoke it at NVIDIA if the
project's access is shared more widely than you meant.

## `run`: a chat on the session

`x0ne-engine cloud run <family:tag>` is `up` with that model, followed by a
chat in the terminal at step `[5/5]`.

- **Before the quote**, the model is checked against the catalog compiled
  into this build (the relay's `x0ne-infer/catalog/models.toml`). A model that
  is absent, or present without a pinned served-model digest ("no served
  digest pinned yet"), is refused (exit 2): nothing is asked, nothing is
  created. The message lists the models the catalog pins with a served
  digest: `qwen3:0.6b`, `qwen3:1.7b`, `qwen3:8b`, `qwen3:14b` and `qwen3:32b`
  in this version, Qwen3 only ([docs/models.md](models.md)). On hardware,
  this Engine on x0ne Core has served `qwen3:8b` only, in one TEST run on
  2026-11-04 ([hardware-runs.md](hardware-runs.md)).
- **In the TD**, `x0ne-infer --cloud-config` pulls the model and checks every
  file against the catalog, shards included. It maps, read-only, the weight
  files the catalog lists (never a directory listing): `model.safetensors`, or
  the index and every shard as published (`qwen3:32b`: 17 shards, no merge).
  The index and the shard headers must agree exactly, a tensor in two shards
  is refused, and the index must name exactly the catalog's shards (72). The
  [served digest](#the-served-model-digest) of those files must be the
  catalog's pin, or the TD refuses before loading (77; 96 for a model without
  a pin or a failed pull). The engine recomputes it on what it loads. The
  relay checks the same digest in the TD's config document before the key is
  released.
- **Before `ready`**, the TD generates two tokens from a fixed prompt and
  drops them, so the first question does not carry the GPU's first-run
  compilation. A failed warm-up is a load failure (72).
- `--yes` needs `--max-cost`, as for `up`.
- **At `[5/5]`**, one line sums up the relay's verdict: CPU (Intel TDX) and
  the number of GPUs verified, the model, its digest and the session key
  (first 16 hex digits). The full verdict is printed above it, at `[3/5]`.
- **The chat is a third process**, `x0ne-engine-proxy chat`: the relay's
  binary, so no question or answer passes through the lifecycle, which holds
  the cloud token. It starts with the relay's environment (no token, no
  override) and a parent-death signal.
- Each line you type is sent with the whole conversation so far to
  `/v1/chat/completions` on the local API, with the relay's key and
  `max_tokens` = `--max-tokens`. Under each answer:
  `[receipt verified by the relay: attested session key, model digest …]`
  when the answer carries `x0ne.receipt_verified: true`, otherwise
  `[receipt NOT verified …]`. Control characters in an answer are printed
  as `?`.
- **To a terminal, the answer is streamed** (`stream: true`): its text shows
  as it comes, and the receipt line follows only once the stream has ended
  clean after its verified final chunk. With standard output redirected, the
  chat asks for whole answers, so a file only ever holds text the relay
  verified first. When a stream fails after its text, a red line under it
  says the text above is not verified: `[receipt FAILED: …; the session is
  stopping]`, or `[stream cut before its receipt: …; the session is over]`.
  Either, like a whole answer whose receipt failed, ends the chat with an
  error; in `run`, that ends the session and the VM is deleted. A failed
  generation is a red `[no answer: …]` line, and the chat goes on without
  that question or its partial answer in the conversation.
- **A connection to the local API has 61 minutes to send each request**:
  one that sends no whole request head within 61 minutes of its accept, or
  of the end of the answer before, is closed, idle or slow (since
  2026-11-10; it stayed open for as long as its peer did). An answer is not
  timed by this, a stream included. It bounds how long a connection is
  held, not how many connections a client opens.
- **The local API stays open** to other clients for the whole session:
  `curl`, or `x0ne-engine cloud chat` from another terminal, which finds the
  address and the key in the session state and becomes the relay's chat
  (exec). Leaving that second chat does not end the session. It exits 0 when
  you leave, 3 when the relay stopped the session on an answer (a failed
  receipt, a broken protocol, a stopping session), 1 on any other failure, a
  cut included. It refuses (96)
  a session whose control socket is not served by the `up` the state names:
  the pid the kernel recorded at its `listen` (SO_PEERCRED) and the session
  id it answers must both be the state's.
- With `--auth stdin`, standard input carries the token, so the chat reads
  the terminal (`/dev/tty`). With no terminal, `run --auth stdin` is refused
  before anything is created (exit 2).

`/exit`, `/bye`, Ctrl-D and Ctrl-C end the session: the chat is stopped
first, then the relay, then the VM is deleted as below.

## The served-model digest

One value names a generative model, whatever its layout: the sha256 of a
`sha256sum` manifest of the files it is served from, sorted by name, after a
first line `x0ne-model-digest/2`. The files are `config.json`,
`tokenizer.json` and the weights: `model.safetensors`, or
`model.safetensors.index.json` and every shard. `LICENSE` and `NOTICE` are
left out; the engine never reads them. In a directory holding exactly those
files, with coreutils:

```sh
{ printf 'x0ne-model-digest/2\n'
  sha256sum config.json tokenizer.json model*.safetensors* | LC_ALL=C sort -k2
} | sha256sum
```

The catalog's pin is that value computed from the sha256s it lists, and its
parser refuses a pin that is not (`x0ne-infer/src/model_digest.rs`). A copy
pulled before 0.2.0-beta.6, its shards merged into one file, is replaced on
the next `pull`. Version 1 (sha256 of `model.safetensors` ‖ `config.json` ‖
`tokenizer.json`) is gone; the values it pinned are in the CHANGELOG.

## How a session ends, and what is left

Every ending `up` lives through deletes the VM. An `up` that is killed
cannot: the VM then stays until `--max-run`, or until `down`.

| what happens | exit | what is left |
|---|---|---|
| Ctrl-C, Ctrl-\ (SIGQUIT), SIGTERM, SIGHUP, another catchable ending signal, or `down` while serving | 0 | nothing |
| `up` killed: SIGKILL, signal 33, SIGABRT, SIGTRAP, SIGSYS, a fault | none (killed; the shell says 128 + n) | **the VM**, until `--max-run`: `x0ne-engine cloud down` deletes it now |
| `/exit`, `/bye`, Ctrl-D or Ctrl-C in `run`'s chat | 0 | nothing |
| `run`'s chat cannot start, read its input or write its output | 96 | nothing |
| `run`'s chat ends on an answer whose receipt failed, or on a stream that broke the protocol or that the session's stop ended | the relay's refusal, 65; 96 when none comes within 10 s of the chat's end | nothing |
| `run`'s chat ends on a stream cut before its receipt | 96 | nothing |
| the run reaches `--max-run`, or 5 min before a token `up` cannot renew expires | 0 | nothing |
| a token `up` cannot renew leaves less than the boot timeout plus 5 min, before the quote or after "yes", or tokeninfo gives no expiry | 91 | nothing |
| Ctrl-C before serving | 130 | nothing; during the creation, the deletion waits for the creation call to return |
| the relay refuses the proof | its code, 64–87 | nothing; its evidence file, if it wrote one |
| the model owner's key policy (`--key-policy`) refuses the verified TD, or the session before anything is created | 102 | nothing; after the verdict, its evidence file, if it wrote one |
| x0ne-infer in the VM exits on a final refusal, read on the serial console before the verdict | 97 | the VM's serial console, saved (see [Files](#files)) |
| the relay was built without an archive key or a transparency log, or pins the TEST ones (`pins`, before the token) | 95 | nothing |
| the relay crashes or breaks the protocol | 95 | nothing |
| spot preemption | 93 | nothing |
| no boot within the boot timeout of the creation, or no verdict 30 s after that | 93 | the VM's serial console, saved (see [Files](#files)) |
| no `serving` within the boot timeout of the verdict, or before the end of the session | 93 | nothing |
| the API refuses the creation (e.g. no spot capacity) | 92 | nothing; an insert that failed is still checked absent three times |
| **the deletion cannot be confirmed** | **94** | **possibly the VM**: the message gives the `gcloud` commands to run |

Rules that hold on every path:

- **The deletion is not interrupted.** A Ctrl-C during the deletion is
  ignored, with a note.
- **A closed terminal does not skip it.** After a hangup (window closed, ssh
  dropped), writes to the terminal fail; they are ignored and the deletion
  goes on (`a_terminal_hangup_*` tests).
- **"Deleted" is checked, not assumed.** After its DELETE, the VM must answer
  404. The firewall rule's delete operation must finish without error. The
  rule is deleted even when the VM's deletion fails.
- **If `up` itself is killed** (SIGKILL, signal 33, SIGABRT, SIGTRAP, SIGSYS,
  a fault), the relay gets SIGTERM (a parent-death signal). Whatever happens
  to this machine, even a power loss, the VM still deletes itself at the end
  of `--max-run`.
  `x0ne-engine cloud status` shows that the `up` process is gone.
  `x0ne-engine cloud down` deletes the VM by name, which needs a token.
  `chat` takes the `up` as alive only when it holds the lock and its control
  socket answers with the state's pid and session id. `status`, `down` and
  `sweep` ask the socket for the session id only, under the lock: no
  listener, or another session's id, is a dead `up`. While another process
  holds the lock (a second `up` being prepared, or a `down`), `down` then
  refuses (91), names the VM, when it deletes itself, and the `sweep
  --project` that deletes it now; it no longer stops that other process in
  its place.
- **An `up` that does not answer is not dead.** Stopped by Ctrl-Z or
  SIGSTOP, busy for more than 2 s, or older than 0.2.0-beta.5 (no session id
  command), it still holds the lock: `status` says it does not answer,
  `sweep` leaves its VM, `down` refuses (91) and says so; `chat` refuses (96).
- **One pid namespace for `chat`.** The pid in the state is the one `up`
  sees. Run from another pid namespace that shares `$XDG_RUNTIME_DIR` (a
  container with its own, such as distrobox `--unshare-process`), the kernel
  reports another pid for the control socket, or 0: `chat` refuses (96) a
  live session. Run it where `up` runs. Toolbox and distrobox share the
  host's pid namespace by default.

## The token

| `--auth` | where the token comes from |
|---|---|
| `paste` (default) | typed or pasted on the terminal with echo off; `gcloud auth print-access-token` prints one on any machine you trust |
| `stdin` | standard input, which must be a pipe (`gcloud auth print-access-token \| x0ne-engine cloud …`), read to the end and trimmed |
| `gcloud` | `gcloud auth print-access-token`, run from `/usr/bin/gcloud` or `/usr/lib/google-cloud-sdk/bin/gcloud` only, never from `$PATH`; asked again every 5 minutes, so a session can outlive one token |

- **It never goes into the environment, argv, a file or a log.** It is held in
  memory that is overwritten when freed (`zeroize`), and is at most 4096 bytes.
- **It is sent to Google's API hosts only:**
  - `https://oauth2.googleapis.com` (tokeninfo, in a POST body);
  - `https://compute.googleapis.com`, `https://cloudbilling.googleapis.com`,
    `https://storage.googleapis.com` and `https://cloudquotas.googleapis.com`
    (`Authorization` header).

  No redirect is followed. An upload URL handed back by the server must be on
  one of these hosts, and gets no token.
- **The process is hardened before any token exists:** `PR_SET_DUMPABLE 0`, no
  core file, umask 077. Another process of the same user cannot read its memory
  or `/proc/<pid>/environ`. A process with `CAP_SYS_PTRACE` (root) can.
- **Expiry.** Access tokens have a limited lifetime. `up` asks Google's
  tokeninfo endpoint how long yours has left, and prints it. `--auth gcloud`
  renews the token. A token given by `paste` or `stdin` is not renewed: `up`
  ends the session 5 min before it expires, so that it deletes the VM while
  the token is still valid, and `--max-run` is cut to fit. 5 min because one
  `down` took about 69 s to delete the VM and 4 s to delete its firewall
  rule: a single measurement, so the margin is about four times it. It is
  also more than the 3 min before expiry at which Google's own Java client
  stops using a token (`OAuth2Credentials.DEFAULT_EXPIRATION_MARGIN` in
  google-auth-library-java, read on 2026-10-22). The cut is said before the
  quote:

  ```text
  --max-run cut to 54 min (asked: 60 min): the token expires at 10:42 UTC and up cannot renew it, so the session ends by 10:37 UTC, 5 min before, to delete the VM with it.
  ```

  The quote, `--max-cost` and the VM's own `maxRunDuration` count the
  session as cut; `[5/5]`, the end of the session (`Session ended: its time
  ran out, 5 min before the token expires (10:42 UTC).`) and `status` (its
  `note`) say why. A token that leaves less than the boot timeout plus 5 min
  of serving is refused before anything else is read, with what it would
  need (exit 91, nothing created); so is a token whose expiry tokeninfo does
  not give. The token is never printed.

## The cost guard

- **Price.** It is the hourly price of `a3-highgpu-1g` spot in the zone's
  region, read with your token from the Cloud Billing catalog (Compute Engine,
  service `6F81-5844-456A`): one H100 GPU, 26 vCPU and 234 GiB, as the API
  reports the machine type. The SKU descriptions are matched loosely
  (*unverified*). It adds 0.01 USD/h, estimated, for the external IP and the
  boot disk.
- **Confidential surcharge.** Google bills Confidential Computing on A3 per VM
  hour on top of the VM: 0.4391592 USD/h spot, read on
  <https://cloud.google.com/confidential-computing/confidential-vm/pricing>
  on 2026-10-16 (0.512173102 USD/h with DWS flex-start, which `up` does not
  use). No Cloud Billing SKU for it is read: the value is embedded, and the
  quote prints it on its own line with the page and the date. A machine type
  without an embedded surcharge is refused before the quote.
- **Fallback.** If the catalog cannot be read, the VM price comes from an
  embedded table dated 2026-10-16 (spot prices read 20:14 UTC): us-central1
  6.60 USD/h, europe-west4 6.26 USD/h, us-east5 3.16 USD/h, one row per
  region `up` may pick. The quote says so, with the date.
- **Maximum cost** = (VM + confidential surcharge + 0.01) per hour × `--max-run`,
  rounded up to the cent; `--max-cost` is compared to that total. The
  ceiling is held by time: the VM deletes itself at the end of its run,
  whatever happens on your machine.
- **`--max-cost <USD>`** refuses (exit 90) any quote above it. `up --yes`
  without `--max-cost` is refused before any request (exit 2).
- **Not included:** image storage (about 0.05 USD per GiB per month, an
  *unverified* estimate), network egress and taxes. Spot prices move; the bill
  is Google's.

## `sweep`

`sweep` lists, then deletes after "yes", the project's resources that carry
**both** marks below:

| resource | marks |
|---|---|
| VMs, disks, IP addresses | name starts with `x0ne-engine-` and label `x0ne-engine` |
| firewall rules (they have no labels) | name starts with `x0ne-engine-`, description starts with `x0ne-engine session `, and the target tag equals the name |
| buckets | name starts with `x0ne-engine-` and label `x0ne-engine` |

- **Order:** VMs, rules, addresses, disks, buckets. A bucket is emptied before
  it is deleted.
- **Left alone:** a resource with only one of the marks, and the session this
  machine is running: its `up` holds the lock and its control socket answers
  the state's session id, or does not answer within 2 s (a stopped or busy
  `up`). A session whose `up` is gone (no listener, or another session's id)
  is listed, even while another process holds the lock.
- **Without `--yes` and without a terminal**, it lists and deletes nothing
  (exit 2).

## `image import`

Once per project, it imports the confidential VM image: a `.tar.gz` holding
`disk.raw`, the format Compute Engine imports. The steps:

1. If the image already exists and is `READY`, it stops there. With a
   reference (`--reference`, or the installed one), the disk sha256 recorded
   in the image's description must equal `image.disk_tar_sha256` first;
   otherwise exit 91, with the delete command: another build under the same
   name.
2. It hashes the file. With a reference, the SHA-256 must equal
   `image.disk_tar_sha256`, or nothing is uploaded (exit 91). The import does
   not check the reference's signature; the relay does, at `up`.
3. It prints the estimate and asks "yes".
4. It creates a private transit bucket `x0ne-engine-import-<random>`, with
   public access prevention enforced and label `x0ne-engine=import`.
5. It uploads the file (resumable), then creates the image (`TDX_CAPABLE`,
   `GVNIC`, `UEFI_COMPATIBLE`).
6. It deletes the object and the bucket. It also does so on failure, and on
   Ctrl-C or any signal that stops `up` cleanly (exit 130); if that deletion
   fails, the exit code is 94, with the command to run.

## `image delete`

`x0ne-engine cloud image delete <x0ne-cvm-…>` deletes one image of the
project, so the cycle `image import`, `up` or `run`, `down`, `sweep` needs no
gcloud to end. The image's storage is billed until then (see
[The cost guard](#the-cost-guard)). The steps:

1. The name must be one `x0ne-cvm-*` resource name; any other, or none, is
   exit 2 before the token is read.
2. A missing image is said and left at that (exit 0).
3. An image without the label `x0ne-engine=image`, which `image import`
   writes, is not deleted (exit 91): it was not made here.
4. It lists the project's disks (`disks.aggregatedList`): an image that is the
   `sourceImage` of a disk attached to a VM (`users`) is not deleted (exit 91),
   the VMs named. A disk made from it and attached to nothing does not hold it.
5. It prints the image, its status and the disk sha256 `image import`
   recorded, and asks "yes" (`--yes` skips the question; without a terminal
   and without `--yes`, exit 2, nothing deleted).
6. It deletes the image and reads it again: it must be gone. Otherwise exit
   94, with the gcloud command to run.

Tested against the fake Google Cloud only (`tests/fake_gcp.rs`); see
[Unverified points](#unverified-points).

## Colors

Colors change no character of the text: with them removed, the output is
the same bytes as without them.

- `NO_COLOR`, set and not empty, turns them off everywhere.
- `CLICOLOR_FORCE`, set, not empty and not `0`, turns them on, even into a
  file or a pipe.
- `CLICOLOR=0` turns them off.
- Otherwise standard output and standard error decide each on its own:
  colors on a terminal whose `TERM` is set and is not `dumb`, never into a
  file or a pipe.

The relay and its chat receive these four variables and decide the same
way, on their standard error. The protocol on the relay's standard output
and the model's answers are never colored.

| what | how |
|---|---|
| the steps `[1/5]` to `[5/5]`, the quote's and `status`'s headings | bold |
| `VERIFIED` | bold green |
| a verdict line (`CPU`, `Firm`, `Image`, `GPU`, `Link`, `Model`) | the label bold; `genuine`, `match`, `matched`, `matches`, `on`, `bound`, `verified`, `endorsed`, `UpToDate`, `known` green; a clause that says `not` or `no` yellow |
| prices, the deletion time, spot, time left, token expiry, cost so far, estimated cost, other warnings | yellow; the maximum cost bold yellow |
| `REFUSED`, a failure, an error before exit | red; `NOT CONFIRMED` and `CRITICAL` bold red |
| the chat's receipt | dim green when the relay verified it, red otherwise |

## The image reference

The relay pins the image by its reference: the newest
`/usr/share/x0ne-engine/reference/x0ne-cvm-<version>.json`, which must belong
to root and be writable by nobody else, or your `--reference`. `<version>` is
digits and dots only: `x0ne-cvm-<V>.manifest.json` and
`x0ne-cvm-<V>.measurements-predicted.json`, published next to the reference,
are never taken for it (before 2026-10-17 the prediction was). Next to it,
`<file>.sig` holds a detached OpenPGP signature over the file's bytes, binary,
as `gpg --detach-sign` writes it. The relay checks it against the x0ne archive
key pinned in its build before it parses the file:

- the signature is version 4, Ed25519, SHA-256, SHA-384 or SHA-512, made by
  a signing subkey bound to the pinned primary key (with its back-signature),
  never by the primary itself;
- the subkey and the primary are neither expired nor revoked in the pinned
  keyring, and the signature was made while the subkey was valid, at most
  5 minutes ahead of this machine's clock;
- `<file>.sig` is a regular file, not a symbolic link, of at most 4096 bytes.
  The lifecycle resolves a symbolic `--reference` first, so the `.sig` sits
  next to the file it points to.

Anything else, a missing `.sig` included, makes the relay refuse with exit 99
before it writes a config; `up` then ends with 95, the relay's reason in the
message, and nothing was created. No flag and no variable skips the check
(`x0ne-infer/tests/cloud_source_sweep.rs`). The verdict names the subkey by
its long key ID, the last 16 hex digits of its fingerprint as `gpg
--keyid-format long` prints them, and the signing date.

The pinned key is the x0ne OS archive key (x0ne-OS `docs/archive-key.md`, not public yet),
made at the signing ceremony; its public keyring and primary fingerprint are
written into the source (`x0ne-infer/src/archive_key.rs`, `RELEASE`). A
release build made before that pins no key: its relay refuses every
reference with 99, before it looks for one, and says it was "built before the
signing ceremony". **This source tree pins no archive key**: `RELEASE` is
`None`. A key revoked after a release keeps verifying in that release until
the next one replaces the pin. After the ceremony, `scripts/pin-archive-key.sh
<public/x0ne-archive-keyring.gpg> <primary fingerprint>` writes the pin: the
keyring goes to `x0ne-infer/archive-key/x0ne-archive-keyring.gpg`, and the
texts that say no key is pinned name the fingerprint; then `make ci`. It
refuses a user ID that says TEST, and one that says REHEARSAL unless
`ALLOW_REHEARSAL_KEY=1` (a rehearsal, in a throwaway clone).

### The transparency log

A signature says who made a reference, not that everyone was shown the same
one. Next to the reference, `<file>.tlog-proof` holds a C2SP
[`tlog-proof@v1`](https://c2sp.org/tlog-proof): a leaf index, an RFC 6962
inclusion proof and a signed checkpoint ([`tlog-checkpoint`](https://c2sp.org/tlog-checkpoint),
[`signed-note`](https://c2sp.org/signed-note), witness
[`tlog-cosignature`](https://c2sp.org/tlog-cosignature)). After the
signature and before it parses the file, the relay (`transparency.rs`):

- computes the logged entry from the bytes it read,
  `x0ne-engine/image-reference/v1\nsha256 <hex>\n`, and checks the proof
  leads from its RFC 6962 leaf hash to the checkpoint's root: a valid proof
  of another reference is refused;
- requires the checkpoint to name the pinned origin and carry the pinned
  log key's signature; signatures of unknown keys are ignored, a known key
  whose signature fails refuses the note;
- requires cosignatures by a quorum of the pinned witnesses, dated within
  the pinned maximum age of this machine's clock and at most 300 s ahead of
  it; otherwise it reads the log's current checkpoint from the log's tiles
  ([`tlog-tiles`](https://c2sp.org/tlog-tiles)), which must meet the same
  rules, shows it consistent with the proof's checkpoint and the reference
  included in it, or refuses;
- compares the checkpoint with the last one it verified for that log,
  `$XDG_STATE_HOME/x0ne-engine/tlog/<16 hex>.checkpoint`: same size, same
  root; other sizes, a consistency proof from the tiles. Two signed
  checkpoints that are not a prefix of each other are a fork: refused, and
  both are written next to it as `fork-<unix time>.txt`. A consistency proof
  read from the tiles that fails is refused and kept the same way: it shows
  a fork, or tiles that do not match the log's own signed checkpoints. The
  larger one is kept.

Any failure is exit 100 before a config is written; `up` ends with 95 and the
relay's reason. A release build pins the log as it pins the archive key
(`transparency::RELEASE`: origin, log key, witness keys, quorum, maximum age,
HTTPS tile location). A pin that names one witness key under two names, or
the log's key as a witness, is refused as a missing pin is: a cosignature
does not sign its cosigner's name. **This source tree pins no log**:
`RELEASE` is `None` and a release relay refuses every reference with 100, as
it refuses them with 99 while no archive key is pinned. x0ne runs no public
log and no witness yet. A pinned log is read at its pinned `https://` location
(`transparency::HttpTiles`): one GET per file, no redirect followed, 20 s
for the whole request, the checkpoint capped at 64 KiB and a tile at 8 KiB;
any other status, a longer body or a transport failure refuses. Plain
`http://` is spoken to the loopback alone, for a local instance of the log
in tests (`tests/tlog_http.rs` serves the TEST log and the log Tessera
wrote, and the relay's check refreshes a stale proof from it; a log that is
unreachable, redirects, hangs, serves another log's checkpoint, no tile or
a tile past the cap, or a forked head, is refused).

When the log cannot be read, nothing falls back to accepting: a reference
whose proof file alone does not meet every rule above is refused with 100,
as a missing proof is. Only a proof cosigned within the maximum age whose
checkpoint is the first this machine sees for that log, or the one it last
verified, needs no read of the log. So a log or
witnesses down for longer than the maximum age stop new sessions with fresh
proofs until they are back: the choice is to fail closed, and the maximum
age sets how long an outage is absorbed.

What it does not do: it does not say the image is good, only that its
reference was published in the same list as every other one. With fewer
honest witnesses than the quorum, a log can show one user a view of its own;
the state of one machine catches a fork only once that machine has seen both
heads. Debug test builds pin a TEST log whose keys are derived from public
strings (`transparency::test_log`, with the TEST archive key); its files are
read from `tlog/` next to the reference, and its freshness window is ten years
so that committed fixtures stay valid.

**Running the log.** `x0ne-engine-tlog` (in `x0ne-infer`, in no release
archive) runs it on the operator's machine and opens no connection.
`keygen log|witness` makes a key: the secret goes to a new file of mode
0600, never replaced, and the verifier key and its fingerprint are printed
for paper. `append` logs references, writes the entry bundles and tiles
(`tlog-tiles`), then the checkpoint signed by the log key; a log whose
entries do not give its checkpoint's root is not used. `witness-request`
prints the [`tlog-witness`](https://c2sp.org/tlog-witness) `add-checkpoint`
body, for any HTTP client to send; `add-cosignatures` adds the answers, each
verified under a witness key named on its command line. The same request
sent again with `old` the checkpoint's own size brings each witness's newer
cosignature, which replaces its older one: that refresh keeps the served
checkpoint within the maximum age, and needs no log key. `prove` writes
`<file>.tlog-proof`, and `verify` runs the relay's check of it against a pin
file. `cosign` is a witness for rehearsals; a witness run by the log's own
operator is not independent of it. `scripts/pin-transparency-log.sh
<log.pin>` pins the log, after the archive key, and prints each key's
fingerprint to compare with the paper.

`cargo test` builds pin a TEST key instead, derived from public strings
(feature `test-archive-key`, enabled by a dev-dependency of `x0ne-infer` on
itself). Only a debug build pins it: `cargo test --release -p x0ne-infer` and
a release `cloud_probe` carry the feature too and keep `RELEASE` (`make ci`, step
`test-release`, with `--release`, the profile that ships). When the key
that signed the reference has a user ID that says TEST, the verdict's image
line says "TEST key, not for release". `x0ne-engine-proxy pins` says
`"test": true` for such a build, and `x0ne-engine cloud up` refuses its relay
(exit 95, before the token).

## Files

| path | mode | content |
|---|---|---|
| `$XDG_RUNTIME_DIR/x0ne-engine/` | 0700 | per-user tmpfs; `quote`, `up`, `run`, `status`, `down` and `sweep` refuse to run without it (exit 91) |
| `…/cloud-session.json` | 0600 | the session state, with no secret: VM name, zone, project, phase, times, price, verdict |
| `…/cloud-session.lock` | 0600 | one session per machine |
| `…/cloud.sock` | 0600 | control socket: `stop` for `down`, and the session id of the `up` serving it, which `chat`, `status`, `down` and `sweep` compare with the state's (`chat` the pid too); the peer's uid must be ours |
| `$XDG_STATE_HOME/x0ne-engine/serial-<vm>-<time>.txt` | 0600 | the VM's serial console after a boot timeout or a final refusal of x0ne-infer (97); text from the VM, kept for review and not printed, save the one cleaned line of that refusal |

`$XDG_STATE_HOME` defaults to `~/.local/state`. The lifecycle writes nothing
else. The relay writes its own files, such as the local API key and evidence.

## The relay protocol, lifecycle side

These are JSON lines, one object per line, each carrying `"v":1`.

**The relay's argv:**

```
x0ne-engine-proxy --protocol 1 --provider gcp --machine-type a3-highgpu-1g \
  --model <model> --expires-at-unix <N> --listen 127.0.0.1:8478 --quiet [--reference <file>]
```

**Before that, `x0ne-engine-proxy pins`:** standard input and error closed,
the same environment. The relay prints one line and exits:
`{"v":1,"event":"pins","archive_key":"<primary fingerprint>","transparency_log":"<origin>","test":<bool>}`
and 0, or the `refused` event a session would get, 99 or 100. On 99 or 100
`up` prints `x0ne-engine cloud: the relay refused (N): <reason>; no token was
asked, nothing was created` and exits 95. `test` is true when the key's user
ID says TEST or the log's origin is under `.invalid`, as the TEST anchors of a
debug build are; `up` then exits 95 too, before the token, unless a test
build points it at its loopback fake (`X0NE_ENGINE_GCP_ENDPOINT`), where
nothing real is created.

**`--quiet`:** the relay sends its progress, its verdict and its refusals as
events only, and the lifecycle prints them, once. Its other messages (a
control-stream or usage error) still reach the terminal: it inherits standard
error. Started alone, without `--quiet`, the relay prints them itself.

**The relay's environment** is rebuilt from scratch. It keeps only `HOME`,
`USER`, `LOGNAME`, `LANG`, `LC_ALL`, `TZ`, `XDG_RUNTIME_DIR`, `XDG_STATE_HOME`,
`XDG_CACHE_HOME` and `XDG_CONFIG_HOME`, and `TERM`, `NO_COLOR`,
`CLICOLOR_FORCE` and `CLICOLOR`, which decide the colors of the relay and its
chat. There is no `PATH`, no token and no test override.

**Placement:** the relay runs in its own process group, with a parent-death
signal (SIGTERM).

| direction | event | the lifecycle's rule |
|---|---|---|
| relay → | `config{config_b64, config_sha256, image{gcp_image, disk_tar_sha256}, tcb_policy{mode, min_tcb_evaluation, max_collateral_age_secs}}` | within 30 s, before anything is created; hash, model and expiry checked; `image` must be the one `up` checked (91 otherwise, and without it: a relay before 0.2.0-beta.6); `tcb_policy`, the policy the relay's verdict will apply, must be at least as strict as `--tcb-policy`, `--min-tcb-evaluation` and `--max-collateral-age`, or the build's floor without them (95 otherwise, and without it); a `tcb_policy` without `mode` is a relay from before 2026-10-28, which is strict; a `refused` instead is printed as `x0ne-engine cloud: the relay refused (N): <reason>; nothing was created`, with N and the exit code of `refused` below |
| → relay | `connect{endpoint "IP:443", boot_deadline_unix}` | sent once the VM is `RUNNING` |
| relay → | `progress{stage, detail}` | printed; stage `releasing` is step `[4/5]` |
| relay → | `verified{verdict, limitations}` | printed and saved in the session state |
| relay → | `refused{code, reason, evidence_file}` | `up` waits up to 5 s for the relay to exit, and kills it after; N is its exit code or, without one (killed, or ended by a signal), the event's `code`, 95 when neither 64–87 nor 102; printed as `REFUSED (N): <reason>`; `up` exits with the relay's exit code when it is 64–87 or 102, otherwise with the event's `code` (64–87 or 102; anything else becomes 95) |
| relay → | `serving{listen, key_file}` | accepted only after `verified` |
| → relay | `stop` | then end of input; the relay answers `stopped` and exits |

An end of output, a line longer than 1 MiB, or an event out of order is a
relay failure. The VM is deleted.

## Exit codes

| code | meaning |
|---|---|
| 0 | ok (including a session stopped while serving) |
| 2 | usage |
| 64–87 | the relay's refusal, with x0ne-infer's codes (65 measurement, 75 NRAS, 76 Intel PCS) |
| 90 | quote refused or not confirmed |
| 91 | preflight: token, quota, image, a session already running, local setup |
| 92 | provider API |
| 93 | VM lost: spot preemption, or boot or verdict timeout |
| 94 | **deletion not confirmed**: run the commands printed |
| 95 | the relay failed, or refused before its config: image reference unusable (its 89), not signed by the pinned archive key (its 99), not shown in the pinned transparency log (its 100), or reporting RTMR0 instead of pinning it (its 104: `rtmr0.policy` must be `gce-endorsed`); a relay built without an archive key or a log is refused before the token |
| 96 | `run`'s chat failed; the VM is deleted |
| 97 | x0ne-infer in the VM refused to serve, a refusal its restarts will not change, read on the serial console before the verdict; the VM is deleted |
| 102 | the model owner's key policy (`--model-key`, `--key-policy`) refused the TD or the session, or its files are unusable: the relay's 102; what was created is deleted |
| 130 | interrupted before serving; what was created is deleted |

## Test overrides, loopback only, test builds only

These exist in a `test-endpoints` build alone: a debug build with that
feature, which `cargo test -p x0ne-engine-cloud` enables for this crate (a
dev-dependency on itself). The released binary reads neither variable's
endpoint and is HTTPS-only; the feature refuses to compile outside debug
builds (`tests/source_sweep.rs` keeps it so).

| variable | effect |
|---|---|
| `X0NE_ENGINE_GCP_ENDPOINT=http://127.0.0.1:<port>` | every API call (OAuth, Compute, Billing, Storage) goes to one local fake; anything but plain http to the IP literal `127.0.0.1` or `[::1]`, with an explicit port, is refused (exit 2) |
| `X0NE_ENGINE_CLOUD_TEST_POLL_MS`, `X0NE_ENGINE_CLOUD_TEST_BOOT_TIMEOUT_MS` | shorter timings; refused (exit 2) unless the endpoint above is set to the loopback |
| `X0NE_ENGINE_CLOUD_TEST_OP_TIMEOUT_MS`, `X0NE_ENGINE_CLOUD_TEST_HTTP_TIMEOUT_MS` | the upper bound of one provider operation and of one API request (5 min each otherwise), so that a test of an operation that never ends, or of an answer that never comes, lasts seconds; refused (exit 2) unless the endpoint above is the loopback |
| `X0NE_ENGINE_CLOUD_TEST_PANIC_AT=creating\|booting\|verifying\|serving` | `up` panics at that step, as a bug would: the test that its VM is deleted all the same; refused (exit 2) unless the endpoint above is the loopback |
| `X0NE_ENGINE_CLOUD_TEST_REFERENCE_DIR` | where `up` looks for the installed reference instead of `/usr/share/x0ne-engine/reference`; refused (exit 2) unless the endpoint above is the loopback; the relay never sees it |
| `X0NE_ENGINE_CLOUD_TEST_MYADDR_SERVER=127.0.0.1:<port>` | the name server asked for this machine's public address, instead of `ns1.google.com:53`; a loopback `ip:port` only, refused (exit 2) unless the endpoint above is the loopback |

None of these reach the relay. `cargo test -p x0ne-engine-cloud --locked` needs
no network:

- unit tests;
- `tests/fake_gcp.rs`: a fake OAuth tokeninfo, Compute Engine, Cloud Billing
  and Cloud Storage on `127.0.0.1`, with a fake of Google's name server on
  the same port number in UDP, and a pty for the prompts; its serial
  port serves byte offsets, a console that grows at each read
  (`tests/serial/`, two status blocks of the VM refused on 2026-10-25), or
  an error;
- `tests/fake_proxy.rs`: fake relays that refuse, crash, hang or lie.
- `tests/chaos.rs`: the same fake, failing on purpose. One fault (an HTTP
  error, a reset, a lost, stalled, cut or unreadable answer, an operation
  that fails or never ends) at each request of `up`, `down`, `sweep`,
  `image import` and `image delete`; every ending signal at every phase of
  `up`; `up` killed, then `down` or `sweep`; two commands at once. After
  each case: nothing the command was in charge of is left, or the exit code
  is 94 and standard error names it with the command that removes it.
  `X0NE_ENGINE_CLOUD_CHAOS=full` runs every case instead of one in four
  (`CI_STEPS=chaos make ci`).

## What it does not protect

`x0ne-engine cloud status --limits` prints this list, followed by the relay's
own:

- this machine: malware in your session can read the local key and your
  questions;
- Google's firmware: the binary Google signed is fetched and measured, its
  source is not published; RTMR0 (the host's TD HOB, Secure Boot variables,
  ACPI tables and boot variables) is pinned to the reference's events as one
  boot showed them, not audited;
- revocation: the TD's PCK is checked against Intel's PCK CRL, its PCK CA and
  Intel's TCB signing certificate against Intel's root CA CRL, both fetched
  live; the OCSP responses a document staples about NVIDIA's signing chain
  are judged under the policy the verdict names, a verified revoked refused
  either way; no TD staples them yet;
- the GPU verdict comes from NVIDIA's NRAS service; each connection's GPU
  evidence answers a nonce derived from that connection's TLS exporter and
  TDX quote, but CPU and GPU are bound by the TD's software, not by the
  hardware (Hopper): the link falls with the CPU TEE (TEE.fail);
- physical attacks on memory (TEE.fail, DDR5 interposer, 2025-10) and side
  channels;
- a compromise of Intel's, NVIDIA's or Google's roots;
- availability: Google can stop the VM (spot); it sees times, sizes, IP
  addresses and the bill;
- the image reference: the relay takes one only with a signature by the x0ne
  archive key pinned in its build, and a relay built before the signing
  ceremony pins none and refuses them all; whoever holds the signing subkey
  can sign any image, and a revocation reaches you only with a new release;
- the behaviour of the model itself.

## What "not end-to-end confidential" covers

A question passes through six places before its answer comes back. The
attested channel covers the content of one of them: the TLS connection from
the relay to the TD. Its two ends, the relay on your machine and the TD,
hold the plaintext, and so does everything before the relay. End-to-end
would mean encrypted from the application you type into to the inside of
the TD, with that application checking the proof. Here the relay checks it,
and the application speaks plain HTTP to the relay.

| Segment | In plaintext there | Who can read it there | Source |
|---|---|---|---|
| the application you type into (`x0ne-engine cloud chat`, Open WebUI, a script) and its machine | the question, the answer, and the history if the application keeps one | you, the machine's administrator, software running as you | `remote_verify::LIMITATIONS`, first line |
| the local API, `http://127.0.0.1:8478/v1` | requests and responses, plain HTTP on the loopback | a process holding the Bearer (`$XDG_RUNTIME_DIR/x0ne-engine/local.key`, mode 0600), which software running as you can read; root | THREAT_MODEL, row "Local software on the user's machine": not a boundary against the same user's processes |
| the relay, `x0ne-engine-proxy` | the question, the answer, the client key | root and the kernel of your machine (the relay is not dumpable, which keeps `ptrace` and `/proc/<pid>/mem` from other processes of your user) | THREAT_MODEL, same row |
| the network from your machine to the TD | not the text: TLS 1.3, mutual, each connection's exporter quoted by the TD (binding v3). In clear: the TD's address, when connections open, each request's size, each whole answer's size, a stream's start and end, so its duration, and its frame size class, which follows the model name | anyone on the path, the cloud provider | THREAT_MODEL, rows "Holder of the TD's TLS private key" and "Host reading a stream's record sizes and timing"; [api.md](api.md), "What the API honours, and what it refuses" |
| the cloud project | the session's configuration, the model's name included (instance metadata `x0ne-config`), the NRAS key given by `--nras-key-file` (instance metadata `x0ne-nras-key`), the model download the TD makes from Hugging Face, the bill | the cloud provider, anyone with access to the project | [What `up` does](#what-up-does), steps 6 and 7 |
| inside the TD: its memory, then the H100's | the question, the answer, the session key, the model | the code of the image the signed reference names (signed by the x0ne archive key; no release key is pinned yet); the host through side channels (TDXRay reads the tokenizer's accesses); the operator with physical access to the host (TEE.fail, DDRop forge the proof itself) | THREAT_MODEL, "Explicitly OUT" and "Known attacks as of 2026-10-22" |

### The sealed path through the relay (prototype)

Since 2026-10-28 the relay also carries the path meant to lift the claim
([sdk-e2e.md](sdk-e2e.md#through-the-relay)): an application linking
x0ne-sdk fetches the TD's key document through the relay
(`GET /v2/e2e/key`, `GET /v2/e2e/collateral`, same Bearer), verifies it
itself (Intel's quote and collateral under the relay's TCB policy, the image
reference by its signature and log proof, the GPU verdict), seals its
question to the TD's key and posts it (`POST /v2/e2e/chat`,
`message/ohttp-req`). On that path, segment by segment:

| Segment | In plaintext there | Who can read it there |
|---|---|---|
| the application and x0ne-sdk, in its process | the question, the answer | you, the machine's administrator, software running as you |
| the local API and the relay | the sealed request and answer (HPKE to the TD's per-boot key), the key document, which is public; not the text | nobody: the relay holds no key that opens them. It sees each message's size bucket, its timing, the outer status, which TD |
| the network from your machine to the TD | the sealed bytes inside the relay's TLS | as above, and as in the table before |
| inside the TD | the question, the answer | as in the table before |

It is a prototype: tested on synthetic evidence, through the real relay
(`cloud_e2e_tests::the_relay_carries_*`), and run on TDX hardware with an
H100 on 2026-10-28, 2026-10-30, 2026-10-31, 2026-11-01 (with the answer
also streamed sealed through the relay) and 2026-11-04 (on x0ne Core), under the TEST archive key and log
([hardware-runs.md](hardware-runs.md)); until the
archive key and the log are pinned (the status at the top of this page), a
release build admits no image reference, here as in the relay;
`x0ne-engine cloud chat` and the plain `/v1` routes do not use it.
The claim above stands until [what is left](sdk-e2e.md#what-is-left-in-order)
is done.

x0ne runs no server on this path. The lifecycle, the relay and the TD reach
Google Cloud, Intel PCS, NVIDIA NRAS, Google's endorsement bucket and, from
the TD, Hugging Face: the hosts the code names, none of them x0ne's. x0ne's
part is the software and the archive key that signs image references;
whoever holds its signing subkey can sign any image
(`remote_verify::LIMITATIONS`).

Outside cloud mode, `x0ne-infer --model-dir` speaks plain
HTTP with no attested channel at all (THREAT_MODEL, "Explicitly OUT", item "Transport").

## Azure

`--provider azure` is refused before any token is asked. Azure's confidential
H100 VMs (`Standard_NCC40ads_H100_v5`) attest through an AMD SEV-SNP report
made by Microsoft's paravisor (`SnpViaParavisor`). That is a different
guarantee from the Intel TDX quote that this client verifies
(`TdxQuoteDirect`), and from the SEV-SNP report a guest binds to the session
itself, which the relay verifies under an SEV-SNP reference
([limitations](limitations.md), "AMD SEV-SNP"): the paravisor's report binds
no session of ours, and a genuine Azure report is refused by the client path
(test).

## Unverified points

They are to be settled by the first real run:

- tokeninfo accepting the token in a POST form body;
- the quota metric name `PREEMPTIBLE_NVIDIA_H100_GPUS` (absent from
  `regions.get` on 2026-10-17; the Cloud Quotas path is the one a real
  project takes);
- the SKU descriptions matched in the Billing catalog;
- `enableVtpm: false` accepted with TDX;
- the boot disk type;
- `maxRunDuration` counted from `RUNNING`;
- the image storage price;
- the disk fields `image delete` reads to find a VM booting from the image
  (`sourceImage`, `users`), and the image's labels as `images.get` returns
  them;
- the firewall rule's partial network URL;
- the firewall rule's /32: that the address Google's name server answers is
  the one the relay's connections to the VM come from, on a network with
  several exits (equal on one machine, 2026-10-30);
- the serial console read while waiting: its parser was run on the consoles
  of four real VMs (three whose TD came up on port 443, one the TD refused), not by `up` on a
  real VM; neither the time it saves nor the read's cost in API quota is
  measured.
