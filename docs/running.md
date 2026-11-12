# Building and running the server

## Running

Building needs the toolchain of `rust-toolchain.toml` (1.98.1; rustup fetches
it) and a C++ compiler: `tokenizers` compiles `esaxx-rs` from C++. In the
`rust:1.98.1-slim-bookworm` image a fresh clone stops at `esaxx-rs`
(`failed to find tool "c++"`) until `apt install g++` (measured 2026-10-16).

The server requires a TDX guest with `configfs-tsm` (kernel ≥ 6.7). The
default `--attest cpu+gpu` also requires an H100/H200 in confidential-
computing mode and a build with `--features nvml`, and `cuda` for x0ne Core,
which runs on the GPU only; a build without `nvml` has no GPU evidence and
refuses. `--attest cpu`, the CPU-only mode, is refused in this version,
as is `--device cpu` (exit 90; [CHANGELOG](../CHANGELOG.md), 2026-11-04). Before any GPU report is read, NVML must say
confidential computing is on, in its production environment, with devtools
off (`nvmlSystemGetConfComputeState`), and every GPU must be of the
architecture `--gpu-arch` names (`nvmlDeviceGetArchitecture`; `HOPPER`, the
default, or `BLACKWELL`); otherwise exit 74 and nothing goes to NRAS. The
cloud TD judges its GPUs the same way. Tested against a simulated NVML; what
a real H100 answers to these two calls is read on the next run on one.

```sh
cargo build --release -p x0ne-infer --features nvml
X0NE_INFER_KEY_HEX=<hex> target/release/x0ne-infer \
  --model-dir /path/to/model --listen 127.0.0.1:8477
```

Every failure before the socket is bound exits with a distinct code and a
named reason; nothing binds, nothing decrypts. The server speaks plain HTTP
and serves one generation at a time: one runs, `--max-queue` (8) may wait,
and the next gets 503 with `Retry-After`. A request has 30 s from its queue
place to send its whole body: past it, it gets 408 and its place is free. A
connection has 61 minutes to send each request head, counted from its accept
or from the end of the answer before: it is closed otherwise, idle or slow,
and an answer is not timed by this. The number of connections is not
bounded. A panic while serving a request is
a 500 for that request; the next one is served. A request whose `Host` is not
`localhost`, `127.0.0.1` or `[::1]` (any port), or a name given with
`--allow-host <host[:port]>`, gets 421: a web page whose name an attacker
rebinds to this machine is not answered (DNS rebinding, as in Ollama's
CVE-2024-28224).

**The `x0ne-infer` binary has no development bypass.** No environment
variable or flag makes it serve outside a verified enclave, and no build
feature changes that: on a machine with no TEE it refuses before it binds
anything. A release build exits 87 first when its measurements file names
no register or has TEST provenance (`"release": false`, as this version's
file has; [Pinning an image](pinning.md)); pinned (with `--expect-measurement`, or
`--measurements` of a release run), or in a debug build, it exits 64 (no
evidence). Contributors work against the library tests, which need no hardware
(`cargo test -p x0ne-infer`); `--help` and `--version` answer anywhere.

No apt repository is published: `make deb` builds the client package
locally ([ci-local.md](ci-local.md)), and `dpkg -i` installs it:
`x0ne-engine`, with `x0ne-engine`, `x0ne-engine-cloud` and
`x0ne-engine-proxy`. This server has no package since 2026-11-04: it serves
only inside an attested enclave, so it ships in the confidential VM's image
only, built with x0ne Core (`cuda nvml`) by the x0ne OS image build
([ci-local.md](ci-local.md#which-recipe-reproduces-which-binary)).

`x0ne-infer/scripts/gcp-cvm.sh` does not serve with this version as it
stands. It is the recipe of the attested run of 2026-08-20 (its header):
it provisioned a confidential machine on GCP and ran the sequence, one phase
at a time, about ten minutes of billed VM. It predates x0ne Core and has not
been run with it: by default it builds without `cuda` and starts the server
with `--device cpu`, which this version refuses (exit 90). No other recipe
in this repository brings up a confidential machine outside
`x0ne-engine cloud`. It downloaded a public model in the clear and generated
a random key on the VM, so it did not exercise sealed weights. Its phases,
kept as the record of that run:

```sh
x0ne-infer/scripts/gcp-cvm.sh create   # then wait-ssh, driver, reboot,
                                       # checks, build, model, measure
export X0NE_INFER_EXPECT="rtmr1=<96 hex> rtmr2=<96 hex>"   # what measure printed
x0ne-infer/scripts/gcp-cvm.sh run      # then fetch
x0ne-infer/scripts/gcp-cvm.sh stop     # the phase that stops the meter
x0ne-infer/scripts/gcp-cvm.sh delete
```

The VM is Ubuntu, not an `x0ne-cvm` image, so the measurements a release
`x0ne-infer` pins by default are not its own; with none given it exits 87.
`run` and `console` therefore refuse to start (exit 2, before ssh) until
`X0NE_INFER_EXPECT` names the registers to pin, and pass them as
`--expect-measurement`. `measure` reads them from a fresh quote of the VM
(configfs-tsm, then `x0ne-infer measurements`) and prints the `export` line.
**That is trust on first use, for this development recipe only**: the values
are what the VM reports, not a reference you checked, and `x0ne-engine cloud
up` never pins this way (it pins a signed release reference). When the server
stops before its attestation is served, `run` exits with the server's own
exit code and prints `infer.log`; it no longer waits two minutes to report 65.
`X0NE_INFER_PORT` sets the server's port on the VM's loopback (default 8477).
`tests/gcp_cvm_recipe.rs` plays every phase against a fake `gcloud`, up to
what only a real VM does.
