//! `x0ne-infer` — the startup sequence, and nothing else.
//!
//! ```text
//! generate session identity (its pubkey goes into the nonces)
//! → collect evidence → verify (cryptographic) → Attested
//! → release key (same verdict) → load model (x0ne Core) → bind → serve
//! ```
//!
//! Any failure before `bind`: a distinct non-zero exit, nothing bound,
//! nothing decrypted. The HTTP surface is `server::serve`, built on the
//! listener `gate::bind` returns — there is no other.
//!
//! Subcommand: `x0ne-infer seal <in> <out>` — client-side weight sealing
//! (AES-256-GCM under X0NE_INFER_KEY_HEX); the server unseals after the
//! gate has released the key.
//!
//! Arguments: see [`USAGE`], which `--help` prints — one text, so the help
//! and the parser cannot drift apart.
//!
//! Exit codes are distinct on purpose, so a supervisor can tell WHY nothing
//! was served: 2 command line refused (an unknown argument, a missing or
//! malformed value), before anything is collected; 64 no evidence; gate
//! refusals (`gate::Refused::exit_code`): 65 rejected verdict, 66 verdict
//! not cryptographic, 67 stale evidence, 68 environment not confidential;
//! 69 bind, 70 key refused, 71 model files unreadable, 72 model load, 73
//! server error, 74 GPU evidence, 75 NRAS, 76 live Intel PCS/TCB, 77 model
//! digest mismatch, 78 malformed NRAS request (refused before it left the
//! machine), 79 the requested device would not initialise (on a CC GPU: arm
//! it), 80 CUDA asked of a build without the feature (rebuild), 81 unknown
//! device name (typo) or a GPU other than the first, the only one Core opens,
//! 82 unknown --attest value, 83 local GPU certificate
//! unreadable (its serial is what binds the token), 84 malformed
//! --expect-measurement, 87 no reference measurement
//! to pin (release build) or a measurements file refused, 101 the NRAS key
//! file (`--nras-key-file`) refused, before anything is collected, 103 NRAS
//! refused that key (an exit of its own: 75 may pass, this does not), 90 the
//! CPU asked to serve (`--device cpu`, `--attest cpu`, a cloud config that
//! attests the CPU only): the server runs on the attested GPU, through
//! Core, the CPU is refused and nothing falls back to it (a test build,
//! `CPU_REFERENCE`, excepted). Cloud mode
//! (`--cloud-config`, `cloud_serve`): 95 cloud config invalid or a flag it
//! excludes (a sealed config without `--sealed-weights`, or the reverse), 96
//! model unavailable (the catalog refuses it in cloud mode, its source is
//! invalid, or the pull failed), 97 key or TLS setup (the TLS identity and
//! server config, the model key recipient), 98 cloud server error; 79 also
//! stands for a GPU not yet accepting client requests (arm it); 71 and 77 also
//! stand for a sealed file unreadable or other than the config names,
//! before any evidence; the relay on the client's machine adds 87 clock
//! wrong, 88 attested channel not bound, 89 image reference unusable, 102
//! the model owner's key policy refuses (`src/bin/proxy.rs`). The `seal`
//! subcommand never serves and uses 2 for every usage/IO error.

use x0ne_confidential::policy::VerificationPolicy;
use x0ne_infer::binding::SessionIdentity;
use x0ne_infer::evidence::{ConfigfsTsm, EvidenceSource};
use x0ne_infer::gate::Gate;

fn urandom32() -> [u8; 32] {
    let mut b = [0u8; 32];
    urandom_into(&mut b);
    b
}

/// `buf` filled from `/dev/urandom` in place: a secret read into memory its
/// owner wipes leaves no copy in a returned value.
fn urandom_into(buf: &mut [u8]) {
    use std::io::Read;
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(buf))
        .expect("urandom");
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The whole command-line surface, in one place — printed by `--help` and
/// referenced by the module doc, so help text and parser cannot drift.
pub const USAGE: &str = "\
x0ne-infer — a confidential-native inference server: no proof, no socket.

USAGE:
  x0ne-infer [OPTIONS]                serve, once the enclave is attested
  x0ne-infer seal <in> <out>          seal weights under X0NE_INFER_KEY_HEX
  x0ne-infer merge-shards <dir> <out> merge shards into one file (offline tools;
                                      serving maps the shards as published)
  x0ne-infer measurements <attestation.json> [--image <name>] [<reg>…]
                                      print measurements/<version>.json for the
                                      image that answered (default rtmr1 rtmr2)

OPTIONS:
  --model-dir <dir>              directory with config.json, tokenizer.json
                                 and model.safetensors (sealed or plaintext)
                                 or model.safetensors.index.json and its
                                 plaintext shards; both layouts is refused
                                 [default: .]
  --listen <addr>                [default: 127.0.0.1:8477]
  --key-id <id>                  key the broker is asked for [default: model-key-0]
  --expected-model-digest <hex>  refuse to serve any other weights
  --expect-measurement <reg>=<hex>
                                 refuse unless the TDX quote reports exactly
                                 this SHA-384 for <reg>: mrtd, rtmr0..rtmr3.
                                 Repeatable; overrides the published file. An
                                 image you built is named by rtmr1/rtmr2
                                 (kernel, command line, initrd); mrtd is the
                                 cloud's virtual firmware
  --measurements <file>          measurements file to pin instead of the one
                                 built in (measurements/<version>.json). With
                                 neither flag the built-in file is pinned; if
                                 it names no register a release build exits
                                 87, a debug build warns and serves
                                 with every register in each signed receipt
  --device <cuda|cuda:0>         where inference runs [default: cuda]: the
                                 attested GPU 0, through x0ne Core. A build
                                 without --features cuda refuses it (80),
                                 another GPU is refused (81). `cpu` is
                                 refused (90): the CPU serves nothing, and
                                 nothing falls back to it
  --attest <cpu+gpu>             what must be proven before serving: the
                                 TDX quote and the GPU's evidence [default
                                 and only value]. `cpu`, an enclave with no
                                 GPU attestation, is refused (90); a GPU
                                 attestation failure refuses too, it never
                                 becomes a CPU-only proof
  --max-completion-tokens <n>    hard ceiling on ONE completion [default:
                                 4096]. A request asking for more is clamped,
                                 not refused, and the answer says
                                 finish_reason length. Without a ceiling, one
                                 POST can hold an H100 for as long as it likes
  --max-queue <n>                requests that may wait behind the running one
                                 [default: 8]; past that, 503 with Retry-After
  --allow-host <host[:port]>     also answer requests whose Host is this name
                                 (any port when none is given). Repeatable.
                                 Only localhost, 127.0.0.1 and [::1] are
                                 answered otherwise; any other Host gets 421
                                 (DNS rebinding). An invalid value is refused
  --gpu-arch <arch>              NRAS arch claim [default: HOPPER]: HOPPER
                                 or BLACKWELL. Every GPU's architecture, read
                                 by NVML, must be this one, and NVML must say
                                 CC is on, in production, devtools off;
                                 otherwise exit 74, nothing sent to NRAS
  --nras-url <url>               NVIDIA attestation service (v4 attest/gpu)
  --nras-jwks-url <url>          its JWKS
  --nras-key-file <file>         an NVIDIA API key, sent to NRAS as
                                 `Authorization: Bearer` on the attest call
                                 [default: none, the call is anonymous].
                                 Read before anything else; never printed
  --pcs-url <url>                Intel PCS base URL (TCB + QE identity)
  --root-crl-url <url>           Intel SGX Root CA CRL (the root's own
                                 CRL distribution point)
  --cloud-config <file>          CLOUD MODE: the config the relay wrote;
                                 model, client SPKI, nonce, expiry and attest
                                 come from it; the key arrives over the
                                 attested channel (POST /v2/key), never from
                                 the environment
  --sealed-weights <file>        cloud mode, with a config that names sealed
                                 weights: the model owner's X0NESEAL file,
                                 checked against the config before any
                                 evidence; its key arrives wrapped to this
                                 boot's attested X25519 key (POST
                                 /v2/model-key, docs/model-key.md)
  --tls-listen <addr>            cloud mode: the TLS 1.3 listener
                                 [default: 0.0.0.0:<listen_port> of the config]
  --max-handshakes <n>           cloud mode: TLS handshakes in flight [default:
                                 256]. One more drops the oldest handshake of
                                 the source address holding the most; each has
                                 10 s to finish
  --model-store <dir>            cloud mode: where the catalog model is pulled
                                 [default: /run/x0ne/models]
  --catalog-endpoint <url>       cloud mode: where it is pulled from, AFTER the
                                 attestation (https; plain http on the loopback)
                                 [default: https://huggingface.co]
  -h, --help                     print this and exit
  -V, --version                  print the version and exit

ENVIRONMENT:
  X0NE_INFER_KEY_HEX   32-byte hex data key (v0 stand-in for the client's
                       KMS; the release DECISION path is the real one).
                       Removed from the environment at start-up and
                       overwritten in the block /proc/<pid>/environ shows

The server refuses to bind unless the environment is cryptographically
attested (Intel TDX quote verified to the pinned Intel root and checked
against live Intel PCS, and NVIDIA GPU evidence verified by NRAS). Exit codes say why: 2 command line refused (an unknown
argument, a missing or malformed value; nothing is collected, and the value
is not printed), 64 no evidence, 65-68 gate refusals,
69 bind, 70 key, 71 model files, 72 load, 73 server, 74 GPU evidence,
75 NRAS, 76 TCB, 77 digest, 78 malformed NRAS request, 79 device init,
80 CUDA not built in, 81 unknown device, 82 unknown --attest value,
83 local GPU certificate unreadable, 84 malformed --expect-measurement,
87 no reference measurement to pin, or a measurements file refused,
90 the CPU asked to serve (--device cpu, --attest cpu),
101 NRAS key file refused (--nras-key-file),
103 NRAS refused that key (HTTP 401 or 403).

CLOUD MODE (`--cloud-config`): mutual TLS 1.3 terminated here, both keys
pinned by SPKI, the TLS key and the config hashed into the quote (binding
v2); GET /v2/attestation serves the document, POST /v2/key brings the
session key once (for sealed weights, GET /v2/model-key/recipient then
POST /v2/model-key bring the owner's key), GET /v2/ready says when the
model is loaded. Exit codes: 95 cloud config invalid or a flag it excludes,
96 model unavailable (refused in cloud mode, a bad catalog source, or the
pull failed), 97 key or TLS setup, 98 cloud server error; 71, 77 and 79 as
above. The client's own refusals live in x0ne-engine-proxy.
";

fn main() {
    // No code the image did not measure: LD_PRELOAD, LD_AUDIT and GLIBC_TUNABLES refuse first, and in
    // a release build the variables that choose a library's file (harden::RELEASE_LOADER_VARIABLES).
    if let Err(e) = x0ne_infer::harden::no_loader_variables() {
        eprintln!("x0ne-infer: process hardening: {e}");
        std::process::exit(73);
    }
    // Peers of the same user cannot ptrace it or read its memory; no core file.
    if let Err(e) = x0ne_infer::harden::process() {
        eprintln!("x0ne-infer: process hardening: {e}");
        std::process::exit(73);
    }
    // stderr is the journal: no panic payload in it (it may quote a prompt).
    x0ne_infer::server::withhold_panic_payloads();
    // SAFETY: still one thread: `main` is not async, the panic hook starts
    // none, and the runtime is built below. keys.rs tests this order.
    let key_hex = unsafe { x0ne_infer::keys::take_env_key(x0ne_infer::keys::KEY_ENV) };
    let broker = x0ne_infer::keys::HeldKeyBroker(key_hex);
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("x0ne-infer: tokio runtime: {e}");
            std::process::exit(73);
        }
    };
    runtime.block_on(async_main(broker));
}

async fn async_main(broker: x0ne_infer::keys::HeldKeyBroker) {
    // Before anything: `--help`/`--version` must answer on ANY machine,
    // including one with no TEE. They print and exit — they never reach the
    // gate, and they never pretend the gate would have opened.
    // `--help` anywhere: the rest of the line is not read.
    if std::env::args().skip(1).any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return;
    }
    if let Some("-V" | "--version") = std::env::args().nth(1).as_deref() {
        println!("x0ne-infer {}{}", env!("CARGO_PKG_VERSION"), git_suffix());
        return;
    }
    // `x0ne-infer seal <in> <out>` — the CLIENT-side tool: seal weights
    // under the data key before they ever reach an operator. No gate here
    // on purpose: sealing is what the client does OUTSIDE the enclave, and
    // it serves nothing and opens no socket.
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("seal") {
        std::process::exit(run_seal(args.get(2), args.get(3), &broker));
    }
    // `merge-shards`: same kind of command — offline, serves nothing. A model
    // above a gigabyte ships in several files and this engine loads ONE.
    if args.get(1).map(String::as_str) == Some("merge-shards") {
        std::process::exit(run_merge_shards(args.get(2), args.get(3)));
    }
    // `measurements`: offline too. Reads a verified `/attestation` document
    // and prints the file a release publishes (`measurements` module).
    if args.get(1).map(String::as_str) == Some("measurements") {
        std::process::exit(run_measurements(&args[2..]));
    }
    let code = run(broker).await;
    std::process::exit(code);
}

fn run_measurements(args: &[String]) -> i32 {
    let mut path = None;
    let mut image = None;
    let mut registers: Vec<&str> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--image" => image = it.next().map(String::as_str),
            r if path.is_some() => registers.push(r),
            p => path = Some(p),
        }
    }
    let Some(path) = path else {
        eprintln!("usage: x0ne-infer measurements <attestation.json> [--image <name>] [<reg>…]");
        return 2;
    };
    if registers.is_empty() {
        registers = vec!["rtmr1", "rtmr2"];
    }
    let doc =
        x0ne_infer::fsread::read_capped(std::path::Path::new(path), x0ne_infer::fsread::CONFIG_CAP)
            .map_err(|e| e.to_string())
            .and_then(|s| {
                serde_json::from_slice::<serde_json::Value>(&s).map_err(|e| e.to_string())
            });
    let rendered =
        doc.and_then(|doc| x0ne_infer::measurements::render(&doc, &registers, image, None));
    match rendered {
        Ok(json) => {
            println!("{json}");
            0
        }
        Err(e) => {
            eprintln!(
                "x0ne-infer measurements: {}: {e}",
                x0ne_infer::argv::path_shown(path)
            );
            2
        }
    }
}

fn run_merge_shards(model_dir: Option<&String>, output: Option<&String>) -> i32 {
    let (Some(model_dir), Some(output)) = (model_dir, output) else {
        eprintln!("usage: x0ne-infer merge-shards <model-dir> <out/model.safetensors>");
        return 2;
    };
    match x0ne_infer::shards::merge(model_dir.as_ref(), output.as_ref()) {
        Ok(r) => {
            eprintln!(
                "x0ne-infer merge-shards: {} tensors from {} shards into {output}",
                r.tensors, r.shards
            );
            0
        }
        Err(e) => {
            eprintln!("x0ne-infer merge-shards: {e:#}");
            2
        }
    }
}

fn run_seal(
    input: Option<&String>,
    output: Option<&String>,
    broker: &x0ne_infer::keys::HeldKeyBroker,
) -> i32 {
    let (Some(input), Some(output)) = (input, output) else {
        eprintln!(
            "usage: x0ne-infer seal <plain.safetensors> <sealed.out>  (key in X0NE_INFER_KEY_HEX)"
        );
        return 2;
    };
    let key = match x0ne_infer::keys::KeyBroker::unwrap_key(broker, "seal") {
        Ok(k) => k,
        Err(e) => {
            eprintln!("x0ne-infer seal: {e:#}");
            return 2;
        }
    };
    let key: [u8; 32] = match key.as_slice().try_into() {
        Ok(k) => k,
        Err(_) => {
            eprintln!("x0ne-infer seal: key must be 32 bytes");
            return 2;
        }
    };
    // AES-GCM's own limit, under the weights cap: refused before the file is read.
    let plain = match x0ne_infer::fsread::read_capped(
        std::path::Path::new(input),
        x0ne_infer::seal::MAX_PLAINTEXT.min(x0ne_infer::fsread::WEIGHTS_CAP),
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "x0ne-infer seal: {}: {e}",
                x0ne_infer::argv::path_shown(input)
            );
            return 2;
        }
    };
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&urandom32()[..12]);
    let sealed = match x0ne_infer::seal::seal(&plain, &key, &nonce) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("x0ne-infer seal: {e:#}");
            return 2;
        }
    };
    if let Err(e) = std::fs::write(output, &sealed) {
        eprintln!(
            "x0ne-infer seal: {}: {e}",
            x0ne_infer::argv::path_shown(output)
        );
        return 2;
    }
    eprintln!(
        "sealed {} bytes -> {} ({} bytes)",
        plain.len(),
        output,
        sealed.len()
    );
    // What a model key policy names (`model_key.rs`): neither is secret.
    use x0ne_infer::cloud_config::{hex, sha256};
    println!("weights_sha256 {}", hex(&sha256(&plain)));
    println!("sealed_sha256 {}", hex(&sha256(&sealed)));
    println!("sealed_size {}", sealed.len());
    0
}

async fn run(broker: x0ne_infer::keys::HeldKeyBroker) -> i32 {
    // Arguments FIRST, and what can be judged without hardware is judged
    // here: a typo in `--device` must cost a millisecond, not a full
    // attestation (the `validate_cap` lesson, cc_cmd).
    let args = match Args::parse_env() {
        Ok(args) => args,
        Err(msg) => {
            eprintln!("x0ne-infer: {msg} (see --help)");
            return 2;
        }
    };
    if let Err((code, msg)) = validate_attest(&args.attest) {
        eprintln!("x0ne-infer: {msg}");
        return code;
    }
    let gpu_required = args.attest != "cpu";
    if args.device_explicit {
        if let Err((code, msg)) = validate_device(&args.device) {
            eprintln!("x0ne-infer: {msg}");
            return code;
        }
    }
    // Read before any evidence, in both modes: a bad file costs a millisecond.
    let nras_key = match args
        .nras_key_file
        .as_deref()
        .map(x0ne_infer::nras_http::NrasKey::read)
        .transpose()
    {
        Ok(k) => k,
        Err(e) => {
            eprintln!("x0ne-infer: --nras-key-file {e}");
            return 101;
        }
    };
    // Cloud mode is a different start-up (`run_cloud`): the config decides,
    // the key comes over the attested channel. Judged after the flags above.
    if let Some(path) = args.cloud_config.clone() {
        let outcome = run_cloud(&args, &path, &broker, nras_key).await;
        drop(broker);
        return match outcome {
            Ok(()) => 0,
            Err((code, msg)) => {
                eprintln!("x0ne-infer: {msg}");
                code
            }
        };
    }
    // Parsed before anything is collected, like `--device` and `--attest`: a
    // typo in a 96-character digest must cost a millisecond and say "typo",
    // not cost an attestation round trip and say "the image changed".
    let expected_measurements = match args
        .expect_measurements
        .iter()
        .map(|spec| x0ne_infer::verify::parse_expected_measurement(spec))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(m) => m,
        Err(e) => {
            eprintln!("x0ne-infer: --expect-measurement {e}");
            return 84;
        }
    };
    // The published file is the default pin; a release with none refuses
    // here, before any evidence, and says how to pin.
    let expected_measurements = match x0ne_infer::measurements::resolve(
        &expected_measurements,
        args.measurements.as_deref(),
    ) {
        Ok(resolved) => {
            if let Some(warning) = resolved.warning {
                eprintln!("x0ne-infer: WARNING: {warning}");
            } else {
                eprintln!(
                    "x0ne-infer: pinning {} from {}",
                    resolved
                        .pins
                        .iter()
                        .map(|m| m.register.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                    resolved.origin
                );
            }
            resolved.pins
        }
        Err(e) => {
            eprintln!("x0ne-infer: {e}");
            return x0ne_infer::measurements::EXIT_UNPINNED;
        }
    };
    // The default device after the pin: a build that cannot serve it (no
    // `cuda`) says so before any evidence (80), and a release build run with
    // no argument meets the measurement refusal first (87, release-refusal.sh).
    if !args.device_explicit {
        if let Err((code, msg)) = validate_device(&args.device) {
            eprintln!("x0ne-infer: {msg}");
            return code;
        }
    }
    let policy = VerificationPolicy {
        workload_name: "x0ne-infer".into(),
        accepted_cpu: vec![x0ne_confidential::model::TeeTechnology::IntelTdx],
        // Follows the flag, and the TDX-only refusal makes the pair enforceable: with
        // `--attest cpu` the CPU-only verdict would REFUSE a policy that
        // still demanded the GPU, so the two cannot drift apart into a
        // proof that says less than the policy asked for.
        require_gpu_cc_on: gpu_required,
        // The published file, or what `--expect-measurement` named, enforced
        // by the cryptographic verdict. Empty only outside a release, and the
        // verdict SAYS so.
        expected_measurements,
    };
    // The serving identity is generated FIRST: its public key is folded into
    // the nonces the hardware echoes (`binding`), so the attestation covers
    // the key that will sign every receipt.
    let identity = SessionIdentity::generate(urandom32(), urandom32());
    let nonce = identity.tdx_report_data();
    let now = now_unix();

    let collected = match ConfigfsTsm::system().collect(&nonce, now) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("x0ne-infer: no evidence, not serving: {e:#}");
            return 64;
        }
    };
    // The GPU half: NVML evidence for the same session (the GPU nonce folds
    // the same public key), POSTed to NRAS, verified against its JWKS. The
    // collection exists only under the `nvml` feature; without it this arm
    // stays empty and the gate refuses — fail-closed, never pretended.

    let nonce_hex: String = identity
        .gpu_nonce()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    #[cfg(feature = "nvml")]
    let (nras_token, nras_jwks, nras_per_gpu, local_gpu_serials) = if !gpu_required {
        // Not collected, not fetched, not attempted. The CPU-only product
        // must not touch NVML at all: a machine with no GPU driver would
        // fail here, and failing to collect evidence nobody asked for would
        // refuse a deployment that is perfectly valid.
        (
            String::new(),
            serde_json::json!({"keys": []}),
            Vec::<(String, String)>::new(),
            Vec::<String>::new(),
        )
    } else {
        // NVML's CC state and architecture are judged before any report is
        // read: every refusal of `judged_gpu_evidence` is 74.
        let evidence = match x0ne_infer::nvml::NvmlLib::open()
            .map_err(|e| format!("GPU evidence collection failed: {e:#}"))
            .and_then(|lib| {
                x0ne_infer::attest_v2::judged_gpu_evidence(
                    &lib,
                    &identity.gpu_nonce(),
                    &args.gpu_arch,
                )
                .map_err(|e| e.message)
            }) {
            Ok(ev) => ev,
            Err(e) => {
                eprintln!("x0ne-infer: {e}, not serving");
                return 74;
            }
        };
        let request = match x0ne_infer::nras_client::build_attest_request(
            &nonce_hex,
            evidence,
            &args.gpu_arch,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("x0ne-infer: NRAS request refused before sending: {e:#}");
                return 78;
            }
        };
        // SECURITY: the identity of the GPUs in THIS machine, read
        // from the certificates we collected ourselves — before NRAS says
        // anything. The tokens' `ueid` must name these, or the attested GPU
        // is somebody else's.
        let serials: Result<Vec<String>, _> = request
            .evidence_list
            .iter()
            .map(|e| x0ne_infer::nras::gpu_leaf_serial_decimal(&e.certificate))
            .collect();
        let serials = match serials {
            Ok(s) => s,
            Err(e) => {
                eprintln!("x0ne-infer: local GPU certificate unreadable, not serving: {e}");
                return 83;
            }
        };
        let attested =
            x0ne_infer::nras_http::attest_gpu(&args.nras_url, &request, nras_key.as_ref()).await;
        let jwks = x0ne_infer::nras_http::fetch_jwks(&args.nras_jwks_url).await;
        match (attested, jwks) {
            (Ok((overall, per_gpu)), Ok(jwks)) => (overall, jwks, per_gpu, serials),
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("x0ne-infer: NRAS attestation failed, not serving: {e:#}");
                return 75;
            }
        }
    };
    #[cfg(not(feature = "nvml"))]
    let _ = &nras_key;
    #[cfg(not(feature = "nvml"))]
    let (nras_token, nras_jwks, nras_per_gpu, local_gpu_serials) = (
        String::new(),
        serde_json::json!({"keys": []}),
        Vec::<(String, String)>::new(),
        Vec::<String>::new(),
    );
    let composite = x0ne_infer::verify::CompositeEvidence {
        tdx: &collected.evidence,
        nras_token: &nras_token,
        nras_jwks: &nras_jwks,
        nonce_hex: &nonce_hex,
        // The CPU half of the challenge — the same value written into the
        // quote request at the top of this function. Comparing it on the way
        // back is what makes the quote OURS rather than merely genuine.
        expected_tdx_report_data: &nonce,
        nras_per_gpu: &nras_per_gpu,
        local_gpu_serials: &local_gpu_serials,
        nras_ocsp_responses: &[],
        nras_ocsp: &x0ne_infer::nras_ocsp::NrasOcspPolicy::production(),
    };
    // ── The LIVE half of the CPU proof — fail-closed ─────────────
    // The chain to the pinned root proves the quote is Intel's; only Intel's
    // CURRENT collateral proves the platform is not stale or revoked. Any
    // failure here — fetch, signature, window, status — refuses service.
    // Since 2026-10-28 the production policy is `Transparent`: accepted on
    // Intel's `standard` collateral, Intel's `early` view named in every
    // receipt (`tcb_disclosure`).
    let tcb_policy = x0ne_infer::tcb::TcbPolicy::production();
    let judgement = match x0ne_infer::tcb_standing::judge_platform_live(
        &collected.evidence.payload,
        &args.pcs_url,
        &args.root_crl_url,
        &x0ne_infer::pcs::TrustAnchor::intel_production(),
        &tcb_policy,
        now_unix(),
    )
    .await
    {
        Ok(j) => j,
        Err(e) => {
            eprintln!("x0ne-infer: live TCB verification failed, not serving: {e:#}");
            return 76;
        }
    };
    let live = judgement.statuses();
    let disclosure = x0ne_infer::tcb_disclosure::TcbDisclosure::intel(
        &judgement,
        tcb_policy.min_tcb_evaluation_data_number,
        tcb_policy.max_collateral_age_secs,
        now_unix(),
    );
    eprintln!("x0ne-infer: {}", disclosure.line);
    // The verdict above is re-read while serving (`tcb_standing`): every
    // hour, the same collateral against the same quote; a platform Intel
    // rates OutOfDate after boot stops answering, a PCS that stops answering
    // leaves the verdict standing for the collateral age limit.
    let standing = std::sync::Arc::new(
        x0ne_infer::tcb_standing::PlatformStanding::verified_at(now_unix(), tcb_policy)
            .disclosing(Some(disclosure)),
    );
    {
        let quote = collected.evidence.payload.clone();
        let (pcs_url, root_crl_url) = (args.pcs_url.clone(), args.root_crl_url.clone());
        let check: x0ne_infer::tcb_standing::LiveCheck = Box::new(move |now: u64| {
            let (quote, pcs_url, root_crl_url) =
                (quote.clone(), pcs_url.clone(), root_crl_url.clone());
            Box::pin(async move {
                x0ne_infer::tcb_standing::judge_platform_live(
                    &quote,
                    &pcs_url,
                    &root_crl_url,
                    &x0ne_infer::pcs::TrustAnchor::intel_production(),
                    &tcb_policy,
                    now,
                )
                .await
            })
        });
        tokio::spawn(x0ne_infer::tcb_standing::revalidate(
            standing.clone(),
            std::time::Duration::from_secs(x0ne_infer::tcb_standing::REVALIDATE_EVERY_SECS),
            std::sync::Arc::new(now_unix),
            check,
        ));
    }
    // The clock is re-read at each decision: two 30 s HTTP round trips sit
    // between collection and this verdict, and a frozen `now` made the
    // gate's staleness window and the key lease vacuous by construction
    // (`now - collected_at` was always 0).
    let now = now_unix();
    // Two entry points, chosen by what the operator ASKED for. Never by
    // what succeeded: a GPU attestation failure returns 74/75/78 above and
    // never arrives here, so no run can slide from the GPU product to the
    // CPU one without the operator having typed `--attest cpu`.
    let verdict = if gpu_required {
        x0ne_infer::verify::cryptographic_after_live_tcb(&composite, &policy, now, &live.0, &live.1)
    } else {
        x0ne_infer::verify::cryptographic_tdx_after_live_tcb(
            &collected.evidence,
            &policy,
            now,
            &nonce,
            &live.0,
            &live.1,
        )
    };
    let proof = match Gate::attest(collected.environment, &collected.evidence, verdict, now) {
        Ok(p) => p,
        Err(refused) => {
            eprintln!("x0ne-infer: gate refused, not serving: {refused}");
            return refused.exit_code();
        }
    };
    // ── Key release (same verdict), model load, then serve ──────────────
    let key = x0ne_infer::keys::release(
        &proof,
        &collected.evidence,
        &args.key_id,
        &broker,
        now_unix(),
    );
    // The hex form is zeroized here; only the parsed key lives on.
    drop(broker);
    let key = match key {
        Ok(k) => k,
        Err(e) => {
            eprintln!("x0ne-infer: key refused, not serving: {e:#}");
            return 70;
        }
    };
    let read = |f: &str| x0ne_infer::fsread::read_model_file(&args.model_dir, f);
    let artifacts = match (
        read("config.json"),
        read("tokenizer.json"),
        x0ne_infer::shards::map_dir(&args.model_dir),
    ) {
        (Ok(c), Ok(t), Ok(w)) => (
            x0ne_infer::model_spec::ModelArtifacts {
                config_json: c,
                tokenizer_json: t,
            },
            w,
        ),
        (c, t, w) => {
            for (name, r) in [("config.json", c.err()), ("tokenizer.json", t.err())] {
                if let Some(e) = r {
                    eprintln!("x0ne-infer: {}: {e}", args.model_dir.join(name).display());
                }
            }
            if let Err(e) = w {
                eprintln!("x0ne-infer: weights: {e:#}");
            }
            return 71;
        }
    };
    let (artifacts, weights) = artifacts;
    // The operator's pin on the plaintext weights read, before a backend is opened or parses
    // them (77): until 2026-11-04 it was compared after Core had loaded them. A sealed file is
    // opened by the backend, under the released key, and pinned on the model loaded below.
    let pinned = x0ne_infer::backend::enforce_expected_digest_before_load(
        args.expected_model_digest.as_deref(),
        &artifacts,
        &weights.as_files(),
    )
    .map_err(|e| (77, format!("{e:#}")));
    // The device the model actually runs on. `--device cuda` on a build
    // without the feature is a REFUSAL, never a silent downgrade to CPU: an
    // operator who asked for the confidential GPU and got the CPU would
    // believe the weights were in the protected region when they were in
    // host-visible-but-TDX-encrypted RAM — a different guarantee, and not
    // the one they asked for.
    // The name was validated at parse time (80/81); what remains is the one
    // thing only the hardware can answer — whether it initialises.
    let backend = match pinned.and_then(|()| open_backend(artifacts, &args.device)) {
        Ok(b) => b,
        Err((code, msg)) => {
            eprintln!("x0ne-infer: {msg}");
            return code;
        }
    };
    // One file goes through load(), which opens a sealed envelope; shards are plaintext.
    let loaded = match weights.files.as_slice() {
        [(_, one)] if weights.index_json.is_none() => backend.load(one, &key),
        _ => backend.load_files(&weights.as_files(), &key),
    };
    let model = match loaded {
        Ok(m) => m,
        Err(e) => {
            eprintln!("x0ne-infer: model load failed: {e:#}");
            return 72;
        }
    };
    // The operator's pinned digest, enforced BEFORE anything binds: a
    // mismatch is a refusal with its own exit code, and the message names
    // both digests (THREAT_MODEL "model substitution").
    if let Err(e) = x0ne_infer::backend::enforce_expected_digest(
        args.expected_model_digest.as_deref(),
        &model.model_digest_hex(),
    ) {
        eprintln!("x0ne-infer: {e:#}");
        return 77;
    }
    // The key is zeroized and the raw weight bytes and backend metadata are
    // released HERE: the loaded model is the only copy that serves, and the
    // plaintext byte buffer must not outlive the load window the threat
    // model accounts for.
    // The window survives the key material. The bytes are zeroized
    // the moment the model holds them; the LEASE is what still says how long
    // this process may answer on the attestation that released them.
    let lease =
        x0ne_infer::serving_lease::ServingLease::new(key.lease().clone(), key.released_at())
            .on(standing);
    drop(key);
    drop(weights);
    drop(backend);
    let listener = match x0ne_infer::gate::bind(&proof, &args.listen).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("x0ne-infer: bind failed: {e}");
            return 69;
        }
    };
    eprintln!(
        "x0ne-infer: attested, serving on {} (model digest {})",
        listener
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or_default(),
        model.model_digest_hex()
    );
    let state = x0ne_infer::server::ServerState {
        // The evidence the gate was opened on, verbatim, so a client can
        // re-run the verification instead of trusting our verdict.
        raw_evidence: x0ne_infer::server::RawEvidence {
            tdx_quote: collected.evidence.payload.clone(),
            nras_token: nras_token.clone(),
        },
        proof,
        identity,
        model,
        // The window the key was released under bounds what this
        // process may still answer. `keys.rs` always said so; nothing read it.
        lease,
        // A client cannot buy an unbounded amount of H100 with one POST.
        max_completion_tokens: args.max_completion_tokens,
        salt_source: Box::new(urandom32),
        // Clients' challenges are answered by the same kernel interface the
        // gate was opened on, one quote per challenge.
        challenge_source: std::sync::Arc::new(ConfigfsTsm::system()),
    };
    let serving = x0ne_infer::server::Serving {
        admission: x0ne_infer::server::Admission::new(args.max_queue),
        allowed_hosts: args.allowed_hosts.clone(),
    };
    if let Err(e) = x0ne_infer::server::serve_with(listener, state, serving).await {
        eprintln!("x0ne-infer: server error: {e}");
        return 73;
    }
    0
}

/// Cloud mode: the cloud config names the model, the client and the product;
/// the session key arrives over the attested channel after the client's
/// own verdict (`cloud_serve`, `attest_v2`). Codes as `run`, plus 95-98.
/// What cloud mode loads once the key is released.
enum Weights {
    Public(x0ne_infer::registry::Served),
    Sealed(Vec<u8>),
}

/// How the sealed file reaches a TD whose config names one.
#[derive(Debug, PartialEq, Eq)]
enum SealedPlan {
    /// `--sealed-weights`: read and checked before anything is collected.
    Given(Vec<u8>),
    /// The config's source: fetched after the attestation, as the catalog
    /// model is, and checked then.
    Fetch(x0ne_infer::cloud_config::SealedSource),
}

/// The sealed file a sealed config names, read and checked before anything
/// is collected, or where to fetch it later; `None` for public weights. 95 a
/// flag the config excludes or needs, 71 unreadable, 77 not the file the
/// config names.
fn sealed_intake(
    config: &x0ne_infer::cloud_config::CloudConfig,
    file: Option<&std::path::Path>,
) -> Result<Option<SealedPlan>, (i32, String)> {
    let path =
        match (&config.sealed, file) {
            (None, None) => return Ok(None),
            (None, Some(_)) => {
                return Err((
                    95,
                    "--sealed-weights but the config names no sealed weights".into(),
                ))
            }
            (Some(s), None) => return match &s.source {
                Some(src) => Ok(Some(SealedPlan::Fetch(src.clone()))),
                None => Err((
                    95,
                    "the config names sealed weights and no source: pass --sealed-weights <file>"
                        .into(),
                )),
            },
            (Some(_), Some(p)) => p,
        };
    let sealed = config.sealed.as_ref().expect("matched above");
    let bytes = x0ne_infer::fsread::read_capped(path, x0ne_infer::fsread::cap_for("model.sealed"))
        .map_err(|e| (71, format!("--sealed-weights {}: {e}", path.display())))?;
    x0ne_infer::model_key::check_sealed_file(&bytes, sealed)
        .map_err(|e| (77, format!("--sealed-weights {}: {e}", path.display())))?;
    Ok(Some(SealedPlan::Given(bytes)))
}

async fn run_cloud(
    args: &Args,
    path: &std::path::Path,
    broker: &x0ne_infer::keys::HeldKeyBroker,
    nras_key: Option<x0ne_infer::nras_http::NrasKey>,
) -> Result<(), (i32, String)> {
    use x0ne_infer::{attest_v2, cloud_config, cloud_serve, registry, tls};
    if broker.0.is_some() {
        return Err((
            95,
            "cloud mode takes the session key from the attested channel: unset X0NE_INFER_KEY_HEX"
                .into(),
        ));
    }
    if !args.expect_measurements.is_empty() || args.measurements.is_some() {
        return Err((
            95,
            "the client pins the image in cloud mode (its reference): drop --expect-measurement and --measurements"
                .into(),
        ));
    }
    if args.expected_model_digest.is_some() {
        return Err((
            95,
            "the catalog pins the model in cloud mode: drop --expected-model-digest".into(),
        ));
    }
    let loaded = cloud_config::load(path)
        .map_err(|e| (95, format!("--cloud-config {}: {e}", path.display())))?;
    if !loaded.config.gpu_required() {
        cpu_serving(CPU_REFERENCE, "a cloud config that attests the CPU only")?;
    }
    if args.attest_explicit && args.attest != loaded.config.attest {
        return Err((
            95,
            format!(
                "--attest {} but the config asks for {}: refusing rather than picking one",
                args.attest, loaded.config.attest
            ),
        ));
    }
    let wants_gpu = loaded.config.gpu_required();
    if wants_gpu && !cfg!(feature = "nvml") {
        return Err((74, "the config asks for cpu+gpu and this build has no `nvml` feature: nothing to attest the GPU with".into()));
    }
    let device = cloud_device(wants_gpu, &args.device, args.device_explicit)?;
    validate_device(&device)?;
    if wants_gpu && device == "cpu" {
        eprintln!("x0ne-infer: GPU attested, serving on the CPU as --device cpu asks");
    }
    // Pure catalog checks (95/96) before anything is collected.
    let catalog = registry::Catalog::embedded();
    let entry = catalog.get(&loaded.config.model).map_err(|e| (95, e))?;
    // Stream frames name the id served; their 768 bytes hold one up to MAX_STREAM_MODEL_BYTES.
    x0ne_infer::openai::models::servable(&entry.name)
        .map_err(|e| (96, format!("{}: {e}", entry.name)))?;
    // The owner's sealed weights: the file the config names, read and
    // checked now (95, 71, 77); the key comes after the owner's own verdict.
    let sealed = sealed_intake(&loaded.config, args.sealed_weights.as_deref())?;
    if let Some(why) = entry.cloud_refusal().filter(|_| sealed.is_none()) {
        return Err((96, why));
    }
    let recipient = match &sealed {
        Some(_) => Some(
            x0ne_infer::model_key::Recipient::generate(&urandom32())
                .map_err(|e| (97, format!("model key recipient: {e}")))?,
        ),
        None => None,
    };
    let store = registry::Store::new(args.model_store.clone());
    let source = registry::Source::new(&args.catalog_endpoint).map_err(|e| (96, e))?;
    // Attestation BEFORE the pull: off a TD it exits 64 with no outbound
    // connection (an earlier build pulled first and hung without network).
    // Two keys born here: ed25519 for receipts, P-256 for TLS; both in the quote.
    let tls_identity = tls::TlsIdentity::generate("x0ne-infer")
        .map_err(|e| (97, format!("TLS identity: {e:#}")))?;
    let identity = SessionIdentity::generate(urandom32(), urandom32());
    let binding =
        x0ne_infer::binding::BindingV2::of(&identity, tls_identity.spki_sha256(), loaded.sha256);
    // The OHTTP key of `/v2/e2e/*`, drawn per boot: its seed read into memory
    // wiped on drop, its scalar in locked memory, as the recipient's. Its key
    // documents name this boot's binding and carry the config's bytes.
    let e2e = {
        let mut seed = zeroize::Zeroizing::new([0u8; 32]);
        urandom_into(&mut seed[..]);
        x0ne_infer::e2e_gateway::E2e::from_seed(&seed, binding, &loaded.bytes)
            .map_err(|e| (97, format!("e2e gateway key: {e}")))?
    };
    let evidence_source = std::sync::Arc::new(attest_v2::SystemEvidence {
        nras_url: args.nras_url.clone(),
        nras_jwks_url: args.nras_jwks_url.clone(),
        nras_key,
        gpu_arch: args.gpu_arch.clone(),
    });
    let mut anchors = attest_v2::TdAnchors::production();
    anchors.root_crl_url = args.root_crl_url.clone();
    let pcs_url = args.pcs_url.clone();
    let attestation = attest_v2::attest_td(
        &*evidence_source,
        &binding,
        wants_gpu,
        &pcs_url,
        &anchors,
        &now_unix,
    )
    .await
    .map_err(|f| (f.code, f.message))?;
    // The owner's relay releases a model key against a TDX quote only.
    if attestation.is_snp() && loaded.config.is_sealed() {
        return Err((
            96,
            "sealed weights need a TDX guest: the model key is released against a TDX quote, \
             and this guest is AMD SEV-SNP"
                .into(),
        ));
    }
    // The model: pulled from the catalog into the store (`pull` verifies
    // every byte against the catalog's digests), then read like `run` does.
    eprintln!(
        "x0ne-infer: attested, pulling {} into {}",
        entry.name,
        store.root.display()
    );
    // Every file against the catalog, the catalog's weight files mapped, then
    // the served digest against the catalog's pin (77): nothing is bound before.
    // Sealed weights: the base model's other files only, in a store of their
    // own; the digest is the one the config promises, enforced after unsealing.
    let (model, artifacts, weights) = match (sealed, &loaded.config.sealed) {
        (Some(plan), Some(cfg)) => {
            let base = x0ne_infer::model_key::base_entry(entry);
            let base_store = registry::Store::new(args.model_store.join("sealed-base"));
            let pulled = registry::pull(&base_store, &source, &base, &mut registry::Quiet)
                .await
                .map_err(|e| (96, format!("pull {} (base files): {e}", base.name)))?;
            let dir = pulled.installed.dir;
            let read = |f: &str| {
                x0ne_infer::fsread::read_model_file(&dir, f)
                    .map_err(|e| (71, format!("{}: {e}", dir.join(f).display())))
            };
            // The sealed file itself: given on the command line and checked
            // before the evidence, or fetched now from the config's source
            // and checked against the same pin. Either way it is verified
            // before the TLS server opens, so before any key can arrive.
            let bytes = match plan {
                SealedPlan::Given(bytes) => bytes,
                SealedPlan::Fetch(src) => {
                    eprintln!(
                        "x0ne-infer: fetching the sealed weights ({} bytes, sha256 {})",
                        src.size, cfg.sealed_sha256
                    );
                    x0ne_infer::model_key::fetch_sealed(&args.model_store, &src, cfg, &entry.name)
                        .await?
                }
            };
            eprintln!(
                "x0ne-infer: {} base files pulled; sealed weights {} wait for the owner's key",
                entry.name, cfg.sealed_sha256
            );
            let model = attest_v2::ModelReport {
                name: entry.name.clone(),
                digest_hex: cfg.digest_hex.clone(),
                files: x0ne_infer::model_key::sealed_files(entry, &cfg.sealed_sha256),
            };
            let artifacts = x0ne_infer::model_spec::ModelArtifacts {
                config_json: read("config.json")?,
                tokenizer_json: read("tokenizer.json")?,
            };
            (model, artifacts, Weights::Sealed(bytes))
        }
        _ => {
            let served =
                registry::pull_served(&store, &source, entry, &mut registry::Quiet).await?;
            eprintln!(
                "x0ne-infer: {} served digest {} is the catalog's pin",
                entry.name, served.digest_hex
            );
            let model = attest_v2::ModelReport {
                name: entry.name.clone(),
                digest_hex: served.digest_hex.clone(),
                files: attest_v2::catalog_files(entry),
            };
            let artifacts = x0ne_infer::model_spec::ModelArtifacts {
                config_json: served.config_json.clone(),
                tokenizer_json: served.tokenizer_json.clone(),
            };
            (model, artifacts, Weights::Public(served))
        }
    };
    let backend = open_backend(artifacts, &device)?;
    // Sealed: one envelope, opened under the released key (the plaintext is
    // held in memory beside it while the model loads).
    let loader: cloud_serve::Loader = Box::new(move |key| match &weights {
        Weights::Public(served) => backend.load_files(&served.weight_files(), key),
        Weights::Sealed(bytes) => backend.load(bytes, key),
    });
    let gpu_ready: cloud_serve::GpuReady = {
        #[cfg(feature = "nvml")]
        {
            Box::new(|| {
                x0ne_infer::nvml::NvmlLib::open()
                    .ok()
                    .and_then(|l| l.gpus_ready_state().ok())
            })
        }
        #[cfg(not(feature = "nvml"))]
        {
            Box::new(|| None)
        }
    };
    let evidence_source_for_state: std::sync::Arc<dyn attest_v2::TdEvidence> =
        evidence_source.clone();
    let refresh: cloud_serve::Refresh = {
        let anchors = anchors.clone();
        Box::new(move || {
            let source = evidence_source.clone();
            let anchors = anchors.clone();
            let pcs_url = pcs_url.clone();
            Box::pin(async move {
                attest_v2::attest_td(&*source, &binding, wants_gpu, &pcs_url, &anchors, &now_unix)
                    .await
            })
        })
    };
    let tls_listen = args
        .tls_listen
        .clone()
        .unwrap_or_else(|| format!("0.0.0.0:{}", loaded.config.listen_port));
    let listener = x0ne_infer::gate::bind(&attestation.proof, &tls_listen)
        .await
        .map_err(|e| (69, format!("bind failed: {e}")))?;
    let tls_config = tls::server_config(&tls_identity, loaded.config.client_spki())
        .map_err(|e| (97, format!("TLS server config: {e:#}")))?;
    eprintln!(
        "x0ne-infer: attested, cloud mode: TLS on {} for client SPKI {} (model {}, digest {})",
        listener
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or_default(),
        loaded.config.client_spki_sha256,
        model.name,
        model.digest_hex
    );
    let max_completion_tokens = loaded.config.max_completion_tokens;
    let snp = attestation.is_snp();
    let standing = std::sync::Arc::new(
        x0ne_infer::tcb_standing::PlatformStanding::verified_at(now_unix(), anchors.tcb_policy)
            .disclosing(Some(attestation.tcb_disclosure.clone())),
    );
    eprintln!("x0ne-infer: {}", attestation.tcb_disclosure.line);
    let state = cloud_serve::CloudState::new(cloud_serve::CloudParts {
        identity,
        binding,
        config: loaded,
        model,
        version: env!("CARGO_PKG_VERSION").into(),
        max_completion_tokens,
        attestation,
        evidence: evidence_source_for_state,
        refresh,
        loader,
        gpu_ready,
        serving: x0ne_infer::server::Serving {
            admission: x0ne_infer::server::Admission::new(args.max_queue),
            // Not read behind mutual TLS; the relay names the TD by its IP.
            allowed_hosts: x0ne_infer::server::AllowedHosts::loopback(),
        },
        salt_source: std::sync::Arc::new(urandom32),
        clock: std::sync::Arc::new(now_unix),
        quote_limits: cloud_serve::QuoteLimits::for_queue(args.max_queue),
        handshake_limits: cloud_serve::HandshakeLimits {
            max_in_flight: args.max_handshakes,
            ..cloud_serve::HandshakeLimits::default()
        },
        standing: standing.clone(),
        recipient,
        e2e,
    });
    // The start-up verdict is re-read every hour while serving
    // (`tcb_standing`), against the quote the document carries. SEV-SNP has
    // no Intel PCS to ask: its standing is not renewed and lapses after
    // `max_collateral_age_secs` (24 h), when computations stop until a restart.
    if snp {
        eprintln!(
            "x0ne-infer: AMD SEV-SNP guest: no live TCB service, the standing lapses after {} s",
            anchors.tcb_policy.max_collateral_age_secs
        );
    } else {
        let st = state.clone();
        let (pcs_url, root_crl_url) = (args.pcs_url.clone(), args.root_crl_url.clone());
        let policy = anchors.tcb_policy;
        let check: x0ne_infer::tcb_standing::LiveCheck = Box::new(move |now: u64| {
            let (st, pcs_url, root_crl_url) = (st.clone(), pcs_url.clone(), root_crl_url.clone());
            Box::pin(async move {
                x0ne_infer::tcb_standing::judge_platform_live(
                    &st.quote(),
                    &pcs_url,
                    &root_crl_url,
                    &x0ne_infer::pcs::TrustAnchor::intel_production(),
                    &policy,
                    now,
                )
                .await
            })
        });
        tokio::spawn(x0ne_infer::tcb_standing::revalidate(
            standing,
            std::time::Duration::from_secs(x0ne_infer::tcb_standing::REVALIDATE_EVERY_SECS),
            std::sync::Arc::new(now_unix),
            check,
        ));
    }
    cloud_serve::serve_tls(listener, tls_config, state)
        .await
        .map_err(|e| (98, format!("cloud server error: {e}")))
}

/// Cloud mode: where inference runs. The config that attests a GPU is
/// served on it; the CPU only when `--device cpu` was typed, never by
/// default. A GPU the config does not attest never holds the model (95).
fn cloud_device(
    gpu_attested: bool,
    device: &str,
    device_explicit: bool,
) -> Result<String, (i32, String)> {
    if !device_explicit {
        return Ok(if gpu_attested { "cuda" } else { "cpu" }.to_string());
    }
    if device != "cpu" && !gpu_attested {
        return Err((
            95,
            format!(
                "--device {device} but the config attests the CPU only: \
                 the model would sit on a GPU that is not attested"
            ),
        ));
    }
    Ok(device.to_string())
}

/// The backend that serves on `device`, by the registry: x0ne Core (`core`)
/// on the GPU; on the CPU, candle, which only the tests link
/// (`candle-reference`). 80: none in this build (rebuild); 79: the device
/// exists in this build but would not come up, on a CC GPU almost always the
/// ready state (arm it, do not rebuild).
fn open_backend(
    artifacts: x0ne_infer::model_spec::ModelArtifacts,
    device: &str,
) -> Result<Box<dyn x0ne_infer::backend::InferenceBackend + Send + Sync>, (i32, String)> {
    let name = if device == "cpu" {
        x0ne_infer::backends::CANDLE
    } else {
        x0ne_infer::backends::DEFAULT
    };
    let registry = x0ne_infer::backends::Registry::default();
    let Some(entry) = registry.get(name) else {
        return Err((
            80,
            format!("--device {device}: no backend in this build serves it ({name} is not linked)"),
        ));
    };
    entry
        .open(artifacts, device)
        .map_err(|e| (79, format!("{e:#}")))
}

/// Whether this build may serve on the CPU: a test build only, which links
/// candle as the CPU reference (`candle-reference`, turned on by the
/// dev-dependency, never by `cargo build`). Every other build refuses it (90).
const CPU_REFERENCE: bool = cfg!(feature = "candle-reference");

/// The device of a server started without `--device`: the attested GPU,
/// through x0ne Core; the CPU in a test build, which has no GPU.
const DEFAULT_DEVICE: &str = if CPU_REFERENCE { "cpu" } else { "cuda" };

/// 90: the CPU asked to serve (`what`). x0ne-infer serves on the attested
/// GPU through x0ne Core; but in a test build (`cpu_reference`) the CPU is
/// refused, and nothing falls back to it. Pure.
fn cpu_serving(cpu_reference: bool, what: &str) -> Result<(), (i32, String)> {
    if cpu_reference {
        return Ok(());
    }
    Err((
        90,
        format!(
            "{what}: refused. x0ne-infer serves only on the attested GPU, through \
             x0ne Core (--attest cpu+gpu, --device cuda): the CPU serves nothing, \
             and nothing falls back to it"
        ),
    ))
}

/// `cpu+gpu`, and nothing else (`cpu` in a test build). Pure.
///
/// An unknown value is a REFUSAL, not a default: `--attest gpu`, `--attest
/// none`, `--attest cpu-only` are all plausible things to type. `cpu` is
/// refused by name (90), never treated as a weaker proof to fall back on.
fn validate_attest(spec: &str) -> Result<(), (i32, String)> {
    validate_attest_in(spec, CPU_REFERENCE)
}

fn validate_attest_in(spec: &str, cpu_reference: bool) -> Result<(), (i32, String)> {
    match spec {
        "cpu+gpu" => Ok(()),
        "cpu" => cpu_serving(cpu_reference, "--attest cpu"),
        _ => Err((
            82,
            x0ne_infer::argv::refused("unknown --attest value", "cpu+gpu"),
        )),
    }
}

/// Is this a device name this BUILD can honour? Pure — no hardware touched.
///
/// Called at parse time for a typed device, after the pin for the default,
/// before evidence collection in both cases, for the same reason
/// `validate_cap` exists on the cc path: an operator who typed `--device gpu`
/// should learn it in a millisecond, not after a full attestation. What
/// cannot be decided here is whether the device will INITIALISE — that needs
/// the hardware and stays after the gate (exit 79).
fn validate_device(spec: &str) -> Result<(), (i32, String)> {
    validate_device_in(spec, CPU_REFERENCE)
}

/// `validate_device` with the build's `CPU_REFERENCE` as an argument: `cpu`
/// is served by a test build only, refused (90) by every other.
fn validate_device_in(spec: &str, cpu_reference: bool) -> Result<(), (i32, String)> {
    if spec == "cpu" {
        return cpu_serving(cpu_reference, "--device cpu");
    }
    if spec == "cuda" || spec.starts_with("cuda:") {
        if let Some(ordinal) = spec.strip_prefix("cuda:") {
            if ordinal.parse::<usize>().is_err() {
                return Err((
                    81,
                    x0ne_infer::argv::refused("--device", "a number after `cuda:`"),
                ));
            }
        }
        // Core opens GPU 0 alone (core_backend::check_device): until 2026-11-04 another
        // ordinal passed here and was refused with 79, after the attestation.
        if x0ne_infer::core_backend::check_device(spec).is_err() {
            return Err((
                81,
                x0ne_infer::argv::refused(
                    "--device",
                    "cuda or cuda:0, x0ne Core runs on GPU 0 only",
                ),
            ));
        }
        if cfg!(feature = "cuda") {
            return Ok(());
        }
        return Err((
            80,
            format!(
                "--device {spec} asked for, but this binary was built without the \
                 `cuda` feature — rebuild with `--features cuda` (needs the CUDA \
                 toolkit). Refusing rather than falling back to the CPU: an operator \
                 who asked for the confidential GPU and silently got the CPU would \
                 believe the weights were in the protected region."
            ),
        ));
    }
    Err((
        81,
        x0ne_infer::argv::refused("unknown --device", "cuda or cuda:0"),
    ))
}

/// v0 arguments — parsed by hand, three of them; clap when they grow.
struct Args {
    model_dir: std::path::PathBuf,
    listen: String,
    key_id: String,
    /// NRAS arch claim; HOPPER for H100/H200 (BLACKWELL when a run targets one).
    /// Read only under the `nvml` feature — the fields still parse without it
    /// so the CLI surface is identical in both builds.
    #[cfg_attr(not(feature = "nvml"), allow(dead_code))]
    gpu_arch: String,
    /// `cpu+gpu` (default; `cpu` in a test build). Parsed at start-up by `validate_attest`,
    /// like `--device`: a typo must cost a millisecond, not a full attestation
    /// round trip.
    attest: String,
    /// Hard ceiling on one completion. A request asking for more is
    /// clamped, not refused — `max_tokens` means "at most" everywhere.
    max_completion_tokens: usize,
    #[cfg_attr(not(feature = "nvml"), allow(dead_code))]
    nras_url: String,
    #[cfg_attr(not(feature = "nvml"), allow(dead_code))]
    nras_jwks_url: String,
    pcs_url: String,
    root_crl_url: String,
    /// Refuse to serve any weights whose digest is not exactly this.
    expected_model_digest: Option<String>,
    /// `cuda` or `cuda:<ordinal>` (`cpu` in a test build): `DEFAULT_DEVICE` when not typed.
    device: String,
    /// `--expect-measurement <register>=<hex>`, repeatable.
    expect_measurements: Vec<String>,
    /// `--measurements <file>`: instead of the built-in file.
    measurements: Option<std::path::PathBuf>,
    /// Requests that may wait behind the running one; past it, 503.
    max_queue: usize,
    /// `Host` values the plain-HTTP server answers (`--allow-host`).
    allowed_hosts: x0ne_infer::server::AllowedHosts,
    /// `--max-handshakes <n>`: cloud mode's TLS handshakes in flight.
    max_handshakes: usize,
    /// `--cloud-config <file>`: cloud mode (`run_cloud`).
    cloud_config: Option<std::path::PathBuf>,
    /// `--nras-key-file <file>`: NRAS's API key, read once in `run`.
    nras_key_file: Option<std::path::PathBuf>,
    /// `--tls-listen <addr>`: cloud mode's TLS listener.
    tls_listen: Option<String>,
    /// `--model-store <dir>`: where cloud mode pulls the catalog model.
    model_store: std::path::PathBuf,
    /// `--sealed-weights <file>`: the model owner's sealed weights, for a
    /// config that names them; the key comes after proof (`model_key.rs`).
    sealed_weights: Option<std::path::PathBuf>,
    /// `--catalog-endpoint <url>`: where cloud mode pulls from (https, or
    /// plain http on the loopback: tests, a local mirror).
    catalog_endpoint: String,
    /// `--attest` was typed: cloud mode refuses a value the config contradicts.
    attest_explicit: bool,
    /// `--device` was typed: in cloud mode the CPU is only ever this choice.
    device_explicit: bool,
}

impl Args {
    fn parse_env() -> Result<Self, String> {
        Self::parse(std::env::args().skip(1))
    }

    /// Every argument known and every value read, else exit 2 before anything is
    /// collected; the flag is named, the value never printed. Until 2026-10-22
    /// `--expected-model-digest=<hex>` served unpinned, `--max-queue 8o` with 8.
    fn parse(argv: impl IntoIterator<Item = String>) -> Result<Self, String> {
        use x0ne_infer::argv::{refused, shown};
        let mut model_dir = std::path::PathBuf::from(".");
        let mut listen = "127.0.0.1:8477".to_string();
        let mut key_id = "model-key-0".to_string();
        let mut gpu_arch = "HOPPER".to_string();
        let mut attest = "cpu+gpu".to_string();
        let mut max_completion_tokens = x0ne_infer::server::DEFAULT_MAX_COMPLETION_TOKENS;
        let mut nras_url = x0ne_infer::nras_client::NRAS_URL.to_string();
        let mut nras_jwks_url = x0ne_infer::nras_client::NRAS_JWKS_URL.to_string();
        let mut pcs_url = x0ne_infer::tcb::PCS_BASE_URL.to_string();
        let mut root_crl_url = x0ne_infer::tcb::INTEL_SGX_ROOT_CA_CRL_URL.to_string();
        let mut expected_model_digest: Option<String> = None;
        let mut device = DEFAULT_DEVICE.to_string();
        let mut expect_measurements = Vec::new();
        let mut measurements = None;
        let mut max_queue = x0ne_infer::server::DEFAULT_MAX_QUEUE;
        let mut allowed_hosts = x0ne_infer::server::AllowedHosts::loopback();
        let mut max_handshakes = x0ne_infer::cloud_serve::DEFAULT_MAX_HANDSHAKES;
        let mut cloud_config: Option<std::path::PathBuf> = None;
        let mut nras_key_file: Option<std::path::PathBuf> = None;
        let mut tls_listen: Option<String> = None;
        let mut model_store = std::path::PathBuf::from("/run/x0ne/models");
        let mut sealed_weights: Option<std::path::PathBuf> = None;
        let mut catalog_endpoint = x0ne_infer::registry::DEFAULT_ENDPOINT.to_string();
        let mut attest_explicit = false;
        let mut device_explicit = false;
        let mut it = argv.into_iter();
        while let Some(a) = it.next() {
            let flag = match a.as_str() {
                "-V" | "--version" => {
                    return Err(format!("{a} is read only as the first argument"));
                }
                f @ ("--model-dir"
                | "--listen"
                | "--key-id"
                | "--gpu-arch"
                | "--attest"
                | "--max-completion-tokens"
                | "--nras-url"
                | "--nras-jwks-url"
                | "--nras-key-file"
                | "--pcs-url"
                | "--root-crl-url"
                | "--expected-model-digest"
                | "--device"
                | "--expect-measurement"
                | "--measurements"
                | "--max-queue"
                | "--allow-host"
                | "--max-handshakes"
                | "--cloud-config"
                | "--tls-listen"
                | "--model-store"
                | "--sealed-weights"
                | "--catalog-endpoint") => f,
                other => return Err(format!("unknown argument {}", shown(other))),
            };
            let value = it
                .next()
                .filter(|v| !v.is_empty())
                .ok_or_else(|| format!("{flag} needs a value"))?;
            let number = |min: usize| {
                value
                    .parse::<usize>()
                    .ok()
                    .filter(|n| *n >= min)
                    .ok_or_else(|| refused(flag, &format!("an integer >= {min}")))
            };
            match flag {
                "--model-dir" => model_dir = value.into(),
                "--listen" => listen = value,
                "--key-id" => key_id = value,
                "--gpu-arch" => gpu_arch = value,
                "--attest" => {
                    attest = value;
                    attest_explicit = true;
                }
                // 0 would be a server that answers nothing.
                "--max-completion-tokens" => max_completion_tokens = number(1)?,
                "--nras-url" => nras_url = value,
                "--nras-jwks-url" => nras_jwks_url = value,
                "--nras-key-file" => nras_key_file = Some(value.into()),
                "--pcs-url" => pcs_url = value,
                "--root-crl-url" => root_crl_url = value,
                "--expected-model-digest" => expected_model_digest = Some(value),
                "--device" => {
                    device = value;
                    device_explicit = true;
                }
                "--expect-measurement" => expect_measurements.push(value),
                "--measurements" => measurements = Some(value.into()),
                "--max-queue" => max_queue = number(0)?,
                "--allow-host" => {
                    let entry = value
                        .parse()
                        .map_err(|_| refused(flag, "<host> or <host>:<port>"))?;
                    allowed_hosts = allowed_hosts.with(entry);
                }
                // 0 would serve nobody.
                "--max-handshakes" => max_handshakes = number(1)?,
                "--cloud-config" => cloud_config = Some(value.into()),
                "--tls-listen" => tls_listen = Some(value),
                "--model-store" => model_store = value.into(),
                "--sealed-weights" => sealed_weights = Some(value.into()),
                _ => catalog_endpoint = value,
            }
        }
        Ok(Self {
            model_dir,
            listen,
            key_id,
            gpu_arch,
            attest,
            max_completion_tokens,
            nras_url,
            nras_jwks_url,
            pcs_url,
            root_crl_url,
            expected_model_digest,
            device,
            expect_measurements,
            measurements,
            max_queue,
            allowed_hosts,
            max_handshakes,
            cloud_config,
            nras_key_file,
            tls_listen,
            model_store,
            sealed_weights,
            catalog_endpoint,
            attest_explicit,
            device_explicit,
        })
    }
}

/// `" (git <sha>)"` when the package build set `X0NE_GIT_SHA` (ci/inside-deb.sh), else empty.
fn git_suffix() -> String {
    option_env!("X0NE_GIT_SHA")
        .map(|s| format!(" (git {s})"))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    /// A device name is judged before ANY hardware is touched, and asking
    /// for CUDA of a build that has none is a refusal, never a downgrade.
    ///
    /// SECURITY: the silent-fallback case is the dangerous one — an operator
    /// who asked for the confidential GPU and got the CPU would believe the
    /// weights were in the GPU's protected region when they were not. A
    /// failure here means either that fallback came back, or that a typo now
    /// costs a full attestation to discover.
    #[test]
    fn a_device_name_is_judged_before_any_hardware_is_touched() {
        assert!(super::CPU_REFERENCE, "the tests link candle");
        assert!(super::validate_device("cpu").is_ok());
        let (code, msg) = super::validate_device("gpu").unwrap_err();
        assert_eq!(code, 81, "{msg}");
        let (code, msg) = super::validate_device("cuda:x").unwrap_err();
        assert_eq!(code, 81, "{msg}");
        // x0ne Core runs on GPU 0 only: another ordinal is refused here, in
        // every build, not with 79 after the attestation.
        for other in ["cuda:1", "cuda:7", "cuda:00"] {
            let (code, msg) = super::validate_device(other).unwrap_err();
            assert_eq!(code, 81, "{other}: {msg}");
            assert!(msg.contains("cuda:0"), "{other}: {msg}");
        }
        let cuda = super::validate_device("cuda");
        if cfg!(feature = "cuda") {
            assert!(cuda.is_ok(), "a cuda build must accept --device cuda");
            assert!(super::validate_device("cuda:0").is_ok());
        } else {
            let (code, msg) = cuda.unwrap_err();
            assert_eq!(code, 80, "{msg}");
            assert!(msg.contains("--features cuda"), "the fix is named: {msg}");
            assert!(
                msg.contains("Refusing rather than falling back"),
                "the refusal must say it is NOT downgrading: {msg}"
            );
        }
    }

    /// SECURITY: in local mode the operator's `--expected-model-digest` is
    /// compared on the plaintext weights read from disk before a backend is
    /// opened or reads them: weights other than the pinned ones never reach
    /// safetensors, Core's arena or the GPU (77). The comparison on the model
    /// loaded stays, for sealed weights, before bind.
    #[test]
    fn the_model_pin_is_compared_before_the_backend_reads_the_weights() {
        let src = include_str!("main.rs");
        let body = &src[..src.find("#[cfg(test)]").unwrap_or(src.len())];
        let at = |what: &str| {
            body.find(what)
                .unwrap_or_else(|| panic!("main.rs: no {what}"))
        };
        let pin = at("enforce_expected_digest_before_load(");
        let open = at("open_backend(artifacts");
        let load = at("backend.load(");
        let after = at("enforce_expected_digest(\n");
        assert!(
            pin < open && open < load && load < after,
            "{pin} {open} {load} {after}"
        );
        assert!(
            body[pin..open].contains("(77,"),
            "the pin's refusal is the 77 of a digest mismatch"
        );
    }

    /// A build without the candle reference (every build but the tests')
    /// refuses the CPU by name, with 90: `--device cpu`, `--attest cpu`, a
    /// cloud config that attests the CPU only. Never a fallback, never 81 or
    /// 82, which say "typo". Its default device is the GPU.
    #[test]
    fn a_served_build_refuses_the_cpu_with_90() {
        use super::{cpu_serving, validate_attest_in, validate_device_in};
        for (what, refusal) in [
            ("--device cpu", validate_device_in("cpu", false)),
            ("--attest cpu", validate_attest_in("cpu", false)),
            ("a config", cpu_serving(false, "a config")),
        ] {
            let (code, msg) = refusal.expect_err(what);
            assert_eq!(code, 90, "{what}: {msg}");
            assert!(msg.starts_with(what), "{msg}");
            assert!(msg.contains("nothing falls back"), "{msg}");
        }
        assert!(validate_device_in("cpu", true).is_ok());
        assert!(validate_attest_in("cpu", true).is_ok());
        assert!(validate_attest_in("cpu+gpu", false).is_ok());
        assert_eq!(validate_attest_in("gpu", false).unwrap_err().0, 82);
        assert_eq!(validate_device_in("gpu", false).unwrap_err().0, 81);
        assert_eq!(super::DEFAULT_DEVICE, "cpu", "a test build");
        assert!(super::USAGE.contains("[default: cuda]"));
    }

    /// Cloud mode: the GPU product runs on the GPU unless `--device cpu` was
    /// typed, and a GPU the config does not attest never holds the model.
    #[test]
    fn cloud_mode_never_runs_on_the_cpu_unless_asked() {
        use super::cloud_device;
        assert_eq!(
            cloud_device(true, "cpu", false).unwrap(),
            "cuda",
            "default with a GPU attested"
        );
        assert_eq!(
            cloud_device(true, "cpu", true).unwrap(),
            "cpu",
            "--device cpu typed"
        );
        assert_eq!(cloud_device(true, "cuda:0", true).unwrap(), "cuda:0");
        assert_eq!(
            cloud_device(false, "cpu", false).unwrap(),
            "cpu",
            "TDX-only product"
        );
        let (code, msg) = cloud_device(false, "cuda", true).unwrap_err();
        assert_eq!(code, 95, "{msg}");
        assert!(msg.contains("not attested"), "{msg}");
    }

    /// Every failure exit is DISTINCT, and every one of them is documented.
    ///
    /// SECURITY/OPS: the module doc promises a supervisor can tell WHY
    /// nothing was served from the exit code alone. Two paths sharing a code
    /// break that promise silently — which is exactly what happened on
    /// 2026-08-21, when the model-digest refusal was given 76, already taken
    /// by the live-TCB refusal. A failure here means either a new collision
    /// or a code the header forgot to list.
    #[test]
    fn every_refusal_exit_code_is_distinct_and_documented() {
        let src = include_str!("main.rs");
        let body = &src[..src.find("#[cfg(test)]").unwrap_or(src.len())];
        // The SERVING path only. `run_seal` is a separate command that never
        // binds anything; it uses 2 for every usage/IO error, the shell
        // convention, and says so in the header.
        let mut codes: Vec<u32> = body
            .lines()
            .filter_map(|l| l.trim().strip_prefix("return "))
            .filter_map(|r| r.trim_end_matches(';').parse::<u32>().ok())
            .filter(|c| *c >= 64)
            .collect();
        let found = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(
            found,
            codes.len(),
            "two paths share an exit code: {codes:?}"
        );
        // The header lists them; gate refusals (65-68) live in `gate.rs` and
        // arrive through `refused.exit_code()`, so they are not `return N`.
        let header = &src[..src.find("use ").unwrap_or(0)];
        for code in &codes {
            assert!(
                header.contains(&code.to_string()),
                "exit {code} is returned but not documented in the module header"
            );
        }
    }

    /// The header names each gate refusal next to the code `gate.rs`
    /// gives it. It listed 65-68 in reverse order until 2026-09-23.
    #[test]
    fn the_header_names_each_gate_refusal_with_its_code() {
        use x0ne_confidential::model::{ExecutionEnvironment, VerifierStrength};
        use x0ne_infer::gate::Refused;
        let src = include_str!("main.rs");
        let header: String = src[..src.find("use ").unwrap_or(0)]
            .lines()
            .map(|l| l.trim_start_matches("//!").trim())
            .collect::<Vec<_>>()
            .join(" ");
        let stale = Refused::Stale {
            collected_at: 0,
            now: 0,
            max: 0,
        };
        for (code, label) in [
            (
                Refused::Rejected("x".into()).exit_code(),
                "rejected verdict",
            ),
            (
                Refused::NotCryptographic(VerifierStrength::Structural).exit_code(),
                "verdict not cryptographic",
            ),
            (stale.exit_code(), "stale evidence"),
            (
                Refused::NotConfidential(ExecutionEnvironment::Standard).exit_code(),
                "environment not confidential",
            ),
            (82, "unknown --attest value"),
        ] {
            let want = format!("{code} {label}");
            assert!(header.contains(&want), "header must say {want:?}");
        }
    }

    /// `--attest` takes two values and refuses everything else with its own
    /// exit code.
    ///
    /// An unknown value must not become a default. `--attest gpu`,
    /// `--attest none` and `--attest cpu-only` are all plausible typings; a
    /// silent "assume the safe one" either refuses a legitimate CPU-only
    /// deployment or serves a CPU-only proof to someone who asked for the
    /// GPU. Refusing costs one line of output.
    #[test]
    fn an_unknown_attest_value_is_refused_rather_than_defaulted() {
        assert!(super::validate_attest("cpu+gpu").is_ok());
        assert!(super::validate_attest("cpu").is_ok(), "a test build");
        for bad in ["gpu", "none", "cpu-only", "CPU", "", "cpu+GPU"] {
            let (code, msg) = super::validate_attest(bad).expect_err("{bad:?} must be refused");
            assert_eq!(code, 82, "{bad:?}");
            assert!(
                msg.contains("cpu+gpu"),
                "the message must say what IS valid: {msg}"
            );
        }
    }

    /// SECURITY: the CPU-only verdict is reached by INTENT, never by
    /// failure.
    ///
    /// The danger of shipping a cheaper, weaker-attested product is that a
    /// GPU attestation failure quietly becomes it — the operator believes
    /// they are selling GPU confidential computing and are not. This test
    /// reads the serving path and requires that the only thing selecting
    /// between the two verdict functions is `gpu_required`, which comes from
    /// the flag and from nothing else.
    ///
    /// A failure means someone wired the CPU-only verdict into an error arm.
    #[test]
    fn the_cpu_only_verdict_is_selected_by_the_flag_and_never_by_a_failure() {
        let src = include_str!("main.rs");
        let body = &src[..src.find("#[cfg(test)]").unwrap_or(src.len())];
        let calls: Vec<&str> = body
            .lines()
            .map(str::trim)
            .filter(|l| l.contains("cryptographic_tdx_after_live_tcb"))
            .collect();
        assert_eq!(
            calls.len(),
            1,
            "the CPU-only verdict must have exactly one call site: {calls:?}"
        );
        // `gpu_required` is assigned once, from the flag, and never reassigned.
        let assignments: Vec<&str> = body
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("let gpu_required") || l.starts_with("gpu_required ="))
            .collect();
        assert_eq!(
            assignments,
            vec!["let gpu_required = args.attest != \"cpu\";"],
            "gpu_required must come from the flag alone, once"
        );
        // And the branch that reaches the CPU-only verdict is that variable's
        // `else`, not a `match` on an error.
        let idx = body
            .find("let verdict = if gpu_required {")
            .expect("the verdict must branch on gpu_required");
        let after = &body[idx..];
        assert!(
            after
                .find("cryptographic_tdx_after_live_tcb")
                .unwrap_or(usize::MAX)
                > after.find("} else {").unwrap_or(0),
            "the CPU-only verdict must sit in the else arm of the flag"
        );
    }

    /// The command lines that start x0ne-infer still parse now that it
    /// refuses what it cannot read: the cloud image's unit (2026-10-22), the
    /// GCP test script and the SSH demonstration client.
    #[test]
    fn the_command_lines_that_start_it_still_parse() {
        for line in [
            "--cloud-config /run/x0ne/config.json --tls-listen 0.0.0.0:443 --device cuda \
             --attest cpu+gpu --model-store /run/x0ne/models",
            "--model-dir ~/model --device cpu --listen 127.0.0.1:8477",
            "--max-queue 0 --max-completion-tokens 1 --max-handshakes 1 --allow-host a.example:8 \
             --expect-measurement rtmr1=00 --measurements m.json --expected-model-digest ab",
        ] {
            let args = super::Args::parse(line.split_whitespace().map(String::from));
            assert!(args.is_ok(), "{line}: {:?}", args.err());
        }
        let a = super::Args::parse(
            "--nras-key-file /run/x0ne/nras-key"
                .split_whitespace()
                .map(String::from),
        )
        .unwrap();
        assert_eq!(
            a.nras_key_file.as_deref(),
            Some(std::path::Path::new("/run/x0ne/nras-key"))
        );
        assert!(super::USAGE.contains("--nras-key-file <file>"));
        let a = super::Args::parse(
            "--cloud-config c --device cuda --attest cpu+gpu"
                .split_whitespace()
                .map(String::from),
        )
        .unwrap();
        assert!(a.device_explicit && a.attest_explicit && a.cloud_config.is_some());
        let a = super::Args::parse(
            "--cloud-config c --sealed-weights /run/x0ne/model.sealed"
                .split_whitespace()
                .map(String::from),
        )
        .unwrap();
        assert_eq!(
            a.sealed_weights.as_deref(),
            Some(std::path::Path::new("/run/x0ne/model.sealed"))
        );
    }

    /// The sealed file and the config agree, or nothing is attested: a
    /// sealed config without the file, the file with a public config, or a
    /// file that is not the one the config names (77) stop the TD first.
    #[test]
    fn the_sealed_file_and_the_config_agree_before_anything_is_attested() {
        use x0ne_infer::cloud_config::{hex, sha256, CloudConfig, SealedWeights};
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.sealed");
        let sealed = x0ne_infer::seal::seal(b"weights", &[1; 32], &[2; 12]).unwrap();
        std::fs::write(&file, &sealed).unwrap();
        let public = CloudConfig::new("qwen3:0.6b", [1; 32], [2; 32], 9, "cpu", 443, 8);
        let with = |sha: [u8; 32]| {
            public.clone().with_sealed(SealedWeights {
                sealed_sha256: hex(&sha),
                digest_hex: "44".repeat(32),
                source: None,
            })
        };
        assert!(super::sealed_intake(&public, None).unwrap().is_none());
        assert_eq!(
            super::sealed_intake(&public, Some(&file)).unwrap_err().0,
            95
        );
        assert_eq!(
            super::sealed_intake(&with(sha256(&sealed)), None)
                .unwrap_err()
                .0,
            95
        );
        assert_eq!(
            super::sealed_intake(&with(sha256(&sealed)), Some(&file)).unwrap(),
            Some(super::SealedPlan::Given(sealed.clone()))
        );
        // A source in the config: the file is fetched after the attestation;
        // a file on the command line still wins, checked now.
        let src = x0ne_infer::cloud_config::SealedSource {
            url: "https://w.example/model.sealed".into(),
            size: sealed.len() as u64,
        };
        let mut sourced = with(sha256(&sealed));
        sourced.sealed.as_mut().unwrap().source = Some(src.clone());
        assert_eq!(
            super::sealed_intake(&sourced, None).unwrap(),
            Some(super::SealedPlan::Fetch(src))
        );
        assert_eq!(
            super::sealed_intake(&sourced, Some(&file)).unwrap(),
            Some(super::SealedPlan::Given(sealed.clone()))
        );
        let (code, why) = super::sealed_intake(&with([9; 32]), Some(&file)).unwrap_err();
        assert_eq!(code, 77, "{why}");
        assert_eq!(
            super::sealed_intake(&with(sha256(&sealed)), Some(&dir.path().join("none")))
                .unwrap_err()
                .0,
            71
        );
    }

    /// SECURITY: a skipped argument kept its default and the server started:
    /// `--expected-model-digest=<hex>` served with no pin. A failure here
    /// means a form that is not read is accepted again.
    #[test]
    fn what_it_cannot_read_is_refused_and_not_printed() {
        let key = "5ec7".repeat(16);
        for line in [
            format!("--expected-model-digest={key}"),
            format!("--expected-model-digset {key}"),
            key.clone(),
            format!("--max-queue {key}"),
            "--max-queue -1".into(),
            "--max-completion-tokens 0".into(),
            "--max-handshakes 0".into(),
            format!("--allow-host a/{key}"),
            "--listen".into(),
            "--model-dir x --version".into(),
        ] {
            let e = super::Args::parse(line.split_whitespace().map(String::from))
                .err()
                .unwrap_or_else(|| panic!("{line} accepted"));
            assert!(!e.contains("5ec75ec7"), "{line}: {e}");
        }
        let e = super::Args::parse(["--listen".to_string(), String::new()]).err();
        assert_eq!(e.as_deref(), Some("--listen needs a value"));
    }

    /// SECURITY: `run`, the self-hosted TD, reads its GPUs' evidence only
    /// through `judged_gpu_evidence`: CC state and architecture by NVML
    /// first. Until 2026-10-30 it read the reports unjudged.
    #[test]
    fn run_reads_gpu_evidence_only_once_nvml_judged_the_gpus() {
        let src = include_str!("main.rs");
        let body = &src[..src.find("#[cfg(test)]").unwrap_or(src.len())];
        let run = &body[body.find("async fn run(").expect("fn run")..];
        let run = &run[..run.find("\n}\n").expect("end of fn run")];
        assert!(
            run.contains("judged_gpu_evidence("),
            "run must judge the GPUs"
        );
        assert!(
            !body.contains("collect_gpu_evidence(") && !body.contains("read_evidence("),
            "GPU evidence read outside judged_gpu_evidence"
        );
    }

    /// The help prints no price. Until 2026-10-30 `--attest` read "about
    /// 0.24 $/h against 3.72 $": an estimate of 2026-08 against a rate the
    /// run of 2026-10-14 had measured at 6.55 $/h. A price belongs to the
    /// quote, read at run time, not to a text that outlives it.
    #[test]
    fn the_help_prints_no_price() {
        for line in super::USAGE.lines() {
            assert!(!line.contains('$'), "a price in --help: {line}");
            assert!(!line.contains("USD"), "a price in --help: {line}");
        }
    }
}
