//! Whole-system scenarios: every kind of test at once, on one session.
//!
//! The scenarios above this module each hold one property on a relay and a
//! fake TD that share a process, a runtime and a deterministic random source.
//! Here the parts run together, and in the first two scenarios apart: the
//! platform and its collateral ("the world"), the TD, and the relay are three
//! child processes of this test binary, each with its own keys and its own
//! OS randomness; the test is the fourth, the local client. What crosses
//! between them crosses on sockets and pipes, as in a deployment, so a
//! process can be killed, restarted, and its journal read for what it must
//! never hold.
//!
//! What stays simulated, and why: the TDX quote (signed by a test Intel
//! root, taken from the world over a socket as a TD takes it from its
//! platform), Intel PCS, Google's endorsement bucket, and the model (the echo
//! backend). The server code (`cloud_serve`, `tls`), the relay (`relay::run`)
//! and the SDK are the shipped ones. The shipped binaries themselves
//! (`x0ne-infer`, `x0ne-engine-proxy`) have no seam for a test trust anchor:
//! they are not what runs here.

use super::*;
use crate::attest_v2::TdEvidence;
use std::io::{BufRead as _, Write as _};
use std::path::PathBuf;
use std::sync::Mutex;

/// The role a child process of this binary plays, and what it plays it with.
const ROLE: &str = "X0NE_SYSTEM_ROLE";
const SPEC: &str = "X0NE_SYSTEM_SPEC";

/// The test every child is started through: with [`ROLE`] set it plays the
/// role and exits, and runs nothing of the scenario.
const ENTRY: &str = "cloud_e2e_tests::system::\
a_session_across_four_processes_answers_whole_streamed_and_sealed_and_journals_no_secret";

/// What a child prints once it is up, before its JSON.
const WORLD_UP: &str = "SYSTEM-WORLD ";
const TD_UP: &str = "SYSTEM-TD ";
const TD_STATS: &str = "SYSTEM-STATS ";

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap()
}

/// 32 bytes of the OS: a child shares no random sequence with another, as
/// `test_random` would give two processes.
fn urandom32() -> [u8; 32] {
    use std::io::Read as _;
    let mut b = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .expect("urandom");
    b
}

/// Every expectation of a scenario that did not hold, read to the end
/// before the scenario fails: one run names them all.
#[derive(Default)]
struct Notes(Vec<String>);

impl Notes {
    fn check(&mut self, held: bool, what: impl FnOnce() -> String) {
        if !held {
            self.0.push(what());
        }
    }

    fn done(self) {
        assert!(self.0.is_empty(), "\n{}", self.0.join("\n"));
    }
}

// ── A process of the system ──────────────────────────────────────────────────

/// One child process: its control stream, its output line by line, and its
/// whole journal (stdout and stderr) for the scan.
struct Proc {
    role: &'static str,
    child: std::process::Child,
    stdin: Option<std::process::ChildStdin>,
    lines: std::sync::mpsc::Receiver<String>,
    out: Arc<Mutex<String>>,
    err: Arc<Mutex<String>>,
    readers: Vec<std::thread::JoinHandle<()>>,
}

impl Proc {
    fn spawn(role: &'static str, spec: &Value) -> Self {
        use std::process::Stdio;
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", ENTRY, "--nocapture", "--test-threads=1"])
            .env(ROLE, role)
            .env(SPEC, spec.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let (tx, lines) = std::sync::mpsc::channel();
        let out = Arc::new(Mutex::new(String::new()));
        let err = Arc::new(Mutex::new(String::new()));
        let stdout = child.stdout.take().unwrap();
        let journal = out.clone();
        let read_out = std::thread::spawn(move || {
            for line in std::io::BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                {
                    let mut all = journal.lock().unwrap();
                    all.push_str(&line);
                    all.push('\n');
                }
                let _ = tx.send(line);
            }
        });
        let mut stderr = child.stderr.take().unwrap();
        let journal = err.clone();
        let read_err = std::thread::spawn(move || {
            use std::io::Read as _;
            let mut all = Vec::new();
            let _ = stderr.read_to_end(&mut all);
            journal
                .lock()
                .unwrap()
                .push_str(&String::from_utf8_lossy(&all));
        });
        Self {
            role,
            stdin: child.stdin.take(),
            child,
            lines,
            out,
            err,
            readers: vec![read_out, read_err],
        }
    }

    fn line(&self) -> String {
        match self.lines.recv_timeout(Duration::from_secs(60)) {
            Ok(line) => line,
            Err(e) => panic!(
                "{}: no line within 60 s ({e}); stderr:\n{}",
                self.role,
                self.err.lock().unwrap()
            ),
        }
    }

    /// What follows `marker` in the first line holding it. Not a prefix:
    /// libtest prints `test <name> ... ` on the same line first.
    fn after(&self, marker: &str) -> Value {
        loop {
            let line = self.line();
            if let Some((_, rest)) = line.split_once(marker) {
                return serde_json::from_str(rest).unwrap_or_else(|e| panic!("{line}: {e}"));
            }
        }
    }

    /// The relay's next event.
    fn event(&self) -> Value {
        loop {
            let line = self.line();
            let Some(at) = line.find('{') else { continue };
            if let Ok(v) = serde_json::from_str::<Value>(&line[at..]) {
                if v["v"] == 1 && v["event"].is_string() {
                    return v;
                }
            }
        }
    }

    /// The relay's next event that is not a progress line.
    fn outcome(&self) -> Value {
        loop {
            let v = self.event();
            if v["event"] != "progress" {
                return v;
            }
        }
    }

    fn say(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("control stream open");
        writeln!(stdin, "{line}").unwrap();
        stdin.flush().unwrap();
    }

    /// Closes the control stream: the end of input, as a parent that died.
    fn hang_up(&mut self) {
        self.stdin = None;
    }

    /// SIGKILL: no handler runs, no connection is closed in order.
    fn kill(&mut self) {
        self.child.kill().unwrap();
    }

    /// Waits for the process to end, on the end of its output; its exit
    /// code, -1 for a signal. Its journal is whole afterwards.
    fn ended(&mut self) -> i32 {
        use std::sync::mpsc::RecvTimeoutError;
        loop {
            match self.lines.recv_timeout(Duration::from_secs(60)) {
                Ok(_) => {}
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {
                    panic!("{}: still running after 60 s", self.role)
                }
            }
        }
        let status = self.child.wait().unwrap();
        for reader in self.readers.drain(..) {
            reader.join().unwrap();
        }
        status.code().unwrap_or(-1)
    }

    /// Stdout and stderr, as far as they were read.
    fn journal(&self) -> Vec<(String, String)> {
        vec![
            (
                format!("{} stdout", self.role),
                self.out.lock().unwrap().clone(),
            ),
            (
                format!("{} stderr", self.role),
                self.err.lock().unwrap().clone(),
            ),
        ]
    }
}

impl Drop for Proc {
    /// A scenario that fails leaves no process behind it.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ── The roles ────────────────────────────────────────────────────────────────

/// Plays `role` to its end and exits with its code.
fn play(role: &str) -> ! {
    let spec: Value = serde_json::from_str(&std::env::var(SPEC).unwrap()).unwrap();
    let rt = runtime();
    let code = match role {
        "world" => rt.block_on(Box::pin(world())),
        "td" => rt.block_on(Box::pin(td(&spec))),
        "relay" => rt.block_on(Box::pin(relay_process(&spec))),
        other => panic!("no such role: {other}"),
    };
    std::process::exit(code)
}

/// Blocks until the parent closes this process's input, handing each line
/// to `on_line`.
fn until_input_ends(mut on_line: impl FnMut(&str)) {
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        on_line(line.trim());
    }
}

/// The world: Intel's test root and the platform that quotes under it, PCS,
/// the root CRL, Google's bucket. A TD asks it for a quote over its report
/// data, one line of hex each way, as a guest asks its platform.
async fn world() -> i32 {
    let lab = Lab::new().await;
    let platform = honest_td(&lab);
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let quotes = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(conn) = conn else { continue };
            let mut asked = String::new();
            if std::io::BufReader::new(&conn)
                .read_line(&mut asked)
                .is_err()
            {
                continue;
            }
            let Ok(report_data) = cloud_config::hex_array::<64>(asked.trim()) else {
                continue;
            };
            let quote = platform.tdx(&report_data, now()).unwrap().evidence.payload;
            let _ = writeln!(&conn, "{}", hex(&quote));
        }
    });
    println!(
        "{WORLD_UP}{}",
        json!({
            "quotes": quotes,
            "root_der_b64": crate::attest_v2::b64(&lab.intel.root_der),
            "pcs_base_url": lab.pcs.base_url,
            "root_crl_url": lab.pcs.root_crl_url,
            "gce_bucket_base_url": lab.bucket,
        })
    );
    until_input_ends(|_| {});
    0
}

fn text_of(v: &Value, key: &str) -> String {
    v[key]
        .as_str()
        .unwrap_or_else(|| panic!("no {key} in {v}"))
        .to_string()
}

fn world_root(world: &Value) -> Vec<u8> {
    unb64(&text_of(world, "root_der_b64")).unwrap()
}

/// A TD's evidence in a process of its own: the quote comes from the world,
/// the event log is the one of the firmware the world's Google endorsed.
struct PlatformTd {
    quotes: String,
}

impl TdEvidence for PlatformTd {
    fn tdx(
        &self,
        report_data: &[u8; 64],
        now_unix: u64,
    ) -> anyhow::Result<crate::evidence::Collected> {
        use x0ne_confidential::model::{
            AttestationEvidence, EvidenceFormat, ExecutionEnvironment, TeeTechnology,
        };
        let conn = std::net::TcpStream::connect(&self.quotes)?;
        conn.set_read_timeout(Some(Duration::from_secs(30)))?;
        writeln!(&conn, "{}", hex(report_data))?;
        let mut answer = String::new();
        std::io::BufReader::new(&conn).read_line(&mut answer)?;
        let payload = cloud_config::hex_bytes(answer.trim()).map_err(anyhow::Error::msg)?;
        Ok(crate::evidence::Collected {
            environment: ExecutionEnvironment::ConfidentialVm {
                cpu: TeeTechnology::IntelTdx,
            },
            evidence: AttestationEvidence {
                format: EvidenceFormat::TdxQuote,
                payload,
                collected_at_unix: now_unix,
            },
        })
    }

    fn ccel(&self) -> anyhow::Result<Vec<u8>> {
        Ok(crate::ccel::encode(&crate::reference::tests::gce_log(
            crate::tdvf::tests::synthetic_cfv(),
        )))
    }

    fn gpu<'a>(
        &'a self,
        _gpu_nonce: &'a [u8; 32],
    ) -> crate::attest_v2::BoxFuture<
        'a,
        Result<crate::attest_v2::GpuAttested, crate::attest_v2::GpuError>,
    > {
        Box::pin(async {
            Err(crate::attest_v2::GpuError {
                code: 74,
                message: "this platform has no GPU".into(),
            })
        })
    }
}

/// The TD: the server's own code on the world's platform, with keys drawn
/// here. `model` is `echo`, or `held`: one word, then a wait nothing ends,
/// so that the process dies in the middle of a stream. The session key it
/// is released is written to `oracle`, for the parent to know what to look
/// for in the journals; `stats` on its input prints its counters.
async fn td(spec: &Value) -> i32 {
    let lab = Lab::new().await;
    let world = &spec["world"];
    let config = cloud_config::parse(&unb64(&text_of(spec, "config_b64")).unwrap()).unwrap();
    let mut anchors = lab.td_anchors();
    anchors.tdx = crate::pcs::TrustAnchor::RootDer(world_root(world));
    anchors.root_crl_url = text_of(world, "root_crl_url");
    let pcs = text_of(world, "pcs_base_url");
    let source = Arc::new(PlatformTd {
        quotes: text_of(world, "quotes"),
    });
    let identity = SessionIdentity::generate(urandom32(), urandom32());
    let session_pubkey_hex = hex(&identity.public_key().to_bytes());
    let tls = TlsIdentity::generate("system-td").unwrap();
    let binding = BindingV2::of(&identity, tls.spki_sha256(), config.sha256);
    let attestation = attest_td(&*source, &binding, false, &pcs, &anchors, &|| now())
        .await
        .unwrap_or_else(|e| panic!("the TD attests on the world's platform: {e}"));
    let refresh: cloud_serve::Refresh = {
        let (source, anchors, pcs) = (source.clone(), anchors.clone(), pcs.clone());
        Box::new(move || {
            let (source, anchors, pcs) = (source.clone(), anchors.clone(), pcs.clone());
            Box::pin(async move {
                attest_td(&*source, &binding, false, &pcs, &anchors, &|| now()).await
            })
        })
    };
    let oracle = PathBuf::from(text_of(spec, "oracle"));
    let held = spec["model"] == "held";
    let served = Arc::new(Mutex::new(model_report().digest_hex));
    let loader: cloud_serve::Loader = Box::new(move |key: &ReleasedKey| {
        key.with_material(|m| std::fs::write(&oracle, hex(m)))?;
        let inner: Box<dyn LoadedModel> = if held {
            Box::new(Words {
                hold: Some(Latch::default()),
                ..Words::default()
            })
        } else {
            crate::backend::EchoBackend.load(b"weights", key)?
        };
        Ok(Box::new(FixedDigest {
            inner,
            digest: served.clone(),
            log: Arc::default(),
            asked: Arc::default(),
            prompts: Arc::default(),
            fails: false,
        }) as Box<dyn LoadedModel>)
    });
    let standing = Arc::new(
        crate::tcb_standing::PlatformStanding::verified_at(
            now(),
            crate::tcb::TcbPolicy::production(),
        )
        .disclosing(Some(attestation.tcb_disclosure.clone())),
    );
    let e2e = crate::e2e_gateway::E2e::from_seed(&urandom32(), binding, &config.bytes).unwrap();
    let client_spki = config.config.client_spki();
    let state = CloudState::new(CloudParts {
        identity,
        binding,
        config,
        model: model_report(),
        version: "test".into(),
        max_completion_tokens: 64,
        attestation,
        evidence: source,
        refresh,
        loader,
        gpu_ready: Box::new(|| Some(cloud_serve::NVML_CC_ACCEPTING_CLIENT_REQUESTS_TRUE)),
        serving: crate::server::Serving::default(),
        salt_source: Arc::new(urandom32),
        clock: Arc::new(now),
        quote_limits: cloud_serve::QuoteLimits::default(),
        handshake_limits: cloud_serve::HandshakeLimits::default(),
        standing,
        recipient: None,
        e2e,
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = listener.local_addr().unwrap().to_string();
    let tls_config = crate::tls::server_config(&tls, client_spki).unwrap();
    let app = cloud_serve::router(state.clone());
    let serving = state.clone();
    tokio::spawn(async move {
        let _ = cloud_serve::serve_tls_app(listener, tls_config, serving, app).await;
    });
    println!(
        "{TD_UP}{}",
        json!({ "endpoint": endpoint, "session_pubkey_hex": session_pubkey_hex })
    );
    until_input_ends(|line| {
        if line == "stats" {
            let s = &state.stats;
            println!(
                "{TD_STATS}{}",
                json!({
                    "phase": state.phase_name(),
                    "handshakes": s.handshakes.load(SeqCst),
                    "session_quotes": s.session_quotes.load(SeqCst),
                    "key_posts": s.key_posts.load(SeqCst),
                    "chat_posts": s.chat_posts.load(SeqCst),
                    "e2e_chat_posts": s.e2e_chat_posts.load(SeqCst),
                })
            );
        }
    });
    0
}

/// The relay: `relay::run` on this process's own input and output, as
/// `x0ne-engine-proxy` runs it, trusting the world's Intel root (its own
/// lab's instead under `own_root`: a relay that does not know this
/// platform's maker), fetching from the world's PCS and bucket.
async fn relay_process(spec: &Value) -> i32 {
    let lab = Lab::new().await;
    let world = &spec["world"];
    let mut env = lab.env();
    if spec["own_root"] != true {
        env.anchors.tdx = crate::pcs::TrustAnchor::RootDer(world_root(world));
    }
    env.anchors.pcs_base_url = text_of(world, "pcs_base_url");
    env.anchors.root_crl_url = text_of(world, "root_crl_url");
    env.anchors.gce_bucket_base_url = text_of(world, "gce_bucket_base_url");
    env.runtime_dir = PathBuf::from(text_of(spec, "runtime_dir"));
    env.state_dir = PathBuf::from(text_of(spec, "state_dir"));
    env.random = Arc::new(urandom32);
    relay::run(
        lab.args("cpu"),
        env,
        BufReader::new(tokio::io::stdin()),
        &mut tokio::io::stdout(),
    )
    .await
}

// ── The system, seen from the test ───────────────────────────────────────────

fn spawn_relay(world: &Value, dir: &Path, own_root: bool) -> Proc {
    Proc::spawn(
        "relay",
        &json!({
            "world": world,
            "runtime_dir": dir.join("run"),
            "state_dir": dir.join("state"),
            "own_root": own_root,
        }),
    )
}

fn spawn_td(world: &Value, config_b64: &str, oracle: &Path, model: &str) -> (Proc, Value) {
    std::fs::create_dir_all(oracle.parent().unwrap()).unwrap();
    let td = Proc::spawn(
        "td",
        &json!({ "world": world, "config_b64": config_b64, "oracle": oracle, "model": model }),
    );
    let info = td.after(TD_UP);
    (td, info)
}

fn connect(endpoint: &str) -> String {
    json!({ "v": 1, "cmd": "connect", "endpoint": endpoint, "boot_deadline_unix": now() + 600 })
        .to_string()
}

/// The session config a relay process announced, as it wrote it.
fn config_of(relay: &Proc) -> String {
    let ev = relay.event();
    assert_eq!(ev["event"], "config", "{ev}");
    text_of(&ev, "config_b64")
}

/// World, relay and TD, each a process, the session verified and serving.
struct System {
    dir: tempfile::TempDir,
    world: Proc,
    world_info: Value,
    relay: Proc,
    config_b64: String,
    td: Proc,
    td_info: Value,
    /// Where the relay's TD connections go: the test stands in between.
    route: Arc<Mutex<String>>,
    verified: Value,
    listen: String,
    bearer: String,
    key_file: String,
}

impl System {
    async fn up(model: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let world = Proc::spawn("world", &json!({}));
        let world_info = world.after(WORLD_UP);
        let mut relay = spawn_relay(&world_info, dir.path(), false);
        let config_b64 = config_of(&relay);
        let (td, td_info) = spawn_td(&world_info, &config_b64, &Self::oracle(&dir, 1), model);
        let (endpoint, route) = switchable_forwarder(text_of(&td_info, "endpoint")).await;
        relay.say(&connect(&endpoint));
        let verified = relay.outcome();
        assert_eq!(verified["event"], "verified", "{verified}");
        assert_eq!(
            verified["verdict"]["session_pubkey_hex"], td_info["session_pubkey_hex"],
            "the key the relay verified is the TD process's"
        );
        assert_eq!(verified["verdict"]["model_digest_hex"], digest());
        let serving = relay.outcome();
        assert_eq!(serving["event"], "serving", "{serving}");
        let listen = text_of(&serving, "listen");
        let key_file = text_of(&serving, "key_file");
        let bearer = std::fs::read_to_string(&key_file)
            .unwrap()
            .trim()
            .to_string();
        assert_eq!(bearer.len(), 64);
        Self {
            dir,
            world,
            world_info,
            relay,
            config_b64,
            td,
            td_info,
            route,
            verified,
            listen,
            bearer,
            key_file,
        }
    }

    /// Where the `n`th TD of this system writes the key it was released:
    /// outside what the scan reads.
    fn oracle(dir: &tempfile::TempDir, n: usize) -> PathBuf {
        dir.path().join("oracle").join(format!("session-key-{n}"))
    }

    fn session_key(&self) -> [u8; 32] {
        cloud_config::hex_array(&text_of(&self.td_info, "session_pubkey_hex")).unwrap()
    }

    async fn post(&self, body: &Value) -> reqwest::Response {
        http()
            .post(format!("http://{}/v1/chat/completions", self.listen))
            .bearer_auth(&self.bearer)
            .json(body)
            .send()
            .await
            .unwrap()
    }

    async fn stream(&self, body: &Value) -> Events {
        let resp = self.post(body).await;
        assert_eq!(resp.status(), 200);
        Events {
            resp,
            reader: SseReader::new(),
            ready: std::collections::VecDeque::new(),
        }
    }

    async fn status(&self) -> Value {
        http()
            .get(format!("http://{}/x0ne/v1/status", self.listen))
            .bearer_auth(&self.bearer)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }

    /// `stop`: the relay says `stopped` and its process exits 0.
    fn stop(&mut self) {
        self.relay
            .say(&json!({ "v": 1, "cmd": "stop" }).to_string());
        let ev = self.relay.outcome();
        assert_eq!(ev["event"], "stopped", "{ev}");
        assert_eq!(self.relay.ended(), 0, "a stopped relay exits 0");
    }
}

fn question(text: &str, max_tokens: usize) -> Value {
    json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": text }],
        "max_tokens": max_tokens,
    })
}

fn files_under(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files_under(&path, found);
        } else {
            found.push(path);
        }
    }
}

/// Every place among `places` (its name, its text) that holds one of
/// `secrets` (its name, the forms it could be written in).
fn leaks(places: &[(String, String)], secrets: &[(&str, Vec<String>)]) -> Vec<String> {
    let mut found = Vec::new();
    for (place, text) in places {
        for (secret, forms) in secrets {
            if forms.iter().any(|f| text.contains(f.as_str())) {
                found.push(format!("{secret} in {place}"));
            }
        }
    }
    found
}

/// A secret of 32 bytes in the forms a journal could carry it.
fn forms_of(bytes: &[u8]) -> Vec<String> {
    vec![
        hex(bytes),
        hex(bytes).to_uppercase(),
        crate::attest_v2::b64(bytes),
    ]
}

// ── Scenario 1: the golden path, across processes ────────────────────────────

/// The whole path on four processes: the relay process verifies the TD
/// process on the world's collateral and releases it a session key; the
/// client is answered whole, streamed, sealed whole and sealed streamed, and
/// checks each receipt with the SDK under the key the quote bound. Then the
/// invariants of a session that went well: no session key, no bearer, no
/// question and no answer in any process's stdout or stderr, nor in a file
/// the relay left, the key file apart (0600, the bearer's one place). And
/// the ends a caller branches on: `stop` exits 0, the end of input 2, a
/// refusal its own code, the event and the process agreeing.
///
/// Control of the scan: it finds the bearer in the key file and the public
/// session key in the relay's events.
#[test]
fn a_session_across_four_processes_answers_whole_streamed_and_sealed_and_journals_no_secret() {
    if let Some(role) = std::env::var_os(ROLE) {
        play(role.to_str().unwrap());
    }
    runtime().block_on(Box::pin(golden_path()));
}

async fn golden_path() {
    const WHOLE: &str = "tangerine-whole-7f3a";
    const STREAMED: &str = "tangerine-streamed-51c9";
    const SEALED: &str = "tangerine-sealed-e02d";
    const SEALED_STREAM: &str = "tangerine-sealed-stream-9b44";
    let mut notes = Notes::default();
    let lab = Lab::new().await;
    let mut sys = System::up("echo").await;
    let key = sys.session_key();

    // The document the relay verified, as the local client reads it.
    let doc: Value = http()
        .get(format!("http://{}/x0ne/v1/attestation", sys.listen))
        .bearer_auth(&sys.bearer)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(doc["document"]["tdx"]["quote_b64"].is_string(), "{doc}");

    // Whole: the TD's answer, its receipt checked here by the SDK.
    let resp = sys.post(&question(&format!("say {WHOLE}"), 16)).await;
    assert_eq!(resp.status(), 200);
    let bytes = resp.bytes().await.unwrap();
    let answer: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(answer["x0ne"]["receipt_verified"], true, "{answer}");
    assert_eq!(answer["x0ne"]["session"], sys.td_info["session_pubkey_hex"]);
    let checked = x0ne_sdk::receipt::check_answer(&bytes, &key, Some(&digest()))
        .expect("the receipt of a whole answer checks under the attested key");
    assert!(checked.text.contains(WHOLE), "{}", checked.text);

    // Streamed: every delta, then the receipt over their text.
    let mut body = question(&format!("say {STREAMED}"), 16);
    body["stream"] = json!(true);
    let (events, end) = sys.stream(&body).await.rest().await;
    assert_eq!(end, BodyEnd::Clean);
    assert_eq!(events.last(), Some(&Data::Done), "{events:?}");
    let text = texts(&events).concat();
    assert!(text.contains(STREAMED), "{text}");
    let last = final_of(&events).expect("a final chunk");
    assert_eq!(last["x0ne"]["receipt_verified"], true, "{last}");
    x0ne_sdk::receipt::check_text(&text, last, &key, Some(&digest()))
        .expect("the receipt of a streamed answer checks under the attested key");

    // Sealed: the SDK verifies the TD's key document itself, on collateral
    // the relay fetched from the world, under the world's root.
    let caller = SealedCaller {
        base: format!("http://{}", sys.listen),
        bearer: sys.bearer.clone(),
        seen: Arc::default(),
    };
    let mut policy = sdk_policy(&lab);
    // Control: under another maker's root, the same document is refused.
    let refused = caller
        .verified(&policy)
        .await
        .err()
        .map(|r| (r.step, r.code));
    notes.check(refused == Some(("quote", 65)), || {
        format!("a key document under an unknown root: {refused:?}")
    });
    policy.anchor = x0ne_sdk::TrustAnchor::RootDer(world_root(&sys.world_info));
    let client = caller.verified(&policy).await.unwrap_or_else(|r| {
        panic!(
            "the SDK refuses the TD: {} {}: {}",
            r.step, r.code, r.reason
        )
    });
    assert_eq!(client.key().session_pubkey(), &key);
    let (request, pending) = sealed_question(&client, &format!("say {SEALED}"));
    let (status, ctype, response) = caller
        .chat(request, x0ne_sdk::ohttp::MEDIA_TYPE_REQUEST)
        .await;
    assert_eq!(
        (status, ctype.as_str()),
        (200, x0ne_sdk::ohttp::MEDIA_TYPE_RESPONSE)
    );
    let (_, checked) = pending
        .open_verified(&response, &digest(), now())
        .expect("the sealed answer opens and its receipt checks");
    let sealed_text = checked.expect("a receipt").text;
    assert!(sealed_text.contains(SEALED), "{sealed_text}");
    let body = json!({
        "stream": true,
        "messages": [{ "role": "user", "content": format!("say {SEALED_STREAM}") }],
        "max_tokens": 8,
    });
    let (_, answer, checked) = stream_through_relay(&caller, &client, &body).await;
    assert_eq!(answer.status, 200, "{answer:?}");
    let stream_text = checked.expect("a receipt").text;
    assert!(stream_text.contains(SEALED_STREAM), "{stream_text}");

    // What each process counted: four questions, one key.
    sys.td.say("stats");
    let stats = sys.td.after(TD_STATS);
    assert_eq!(stats["key_posts"], 1, "{stats}");
    assert_eq!(stats["chat_posts"], 2, "{stats}");
    assert_eq!(stats["e2e_chat_posts"], 2, "{stats}");

    sys.stop();
    sys.td.hang_up();
    assert_eq!(sys.td.ended(), 0);
    sys.world.hang_up();
    assert_eq!(sys.world.ended(), 0);

    // The scan, over every journal and every file the relay left.
    let released = std::fs::read_to_string(System::oracle(&sys.dir, 1)).unwrap();
    let released = cloud_config::hex_bytes(released.trim()).unwrap();
    assert_eq!(released.len(), 32);
    let mut places = [&sys.world, &sys.relay, &sys.td]
        .iter()
        .flat_map(|p| p.journal())
        .collect::<Vec<_>>();
    let mut files = Vec::new();
    files_under(&sys.dir.path().join("run"), &mut files);
    files_under(&sys.dir.path().join("state"), &mut files);
    let key_file = PathBuf::from(&sys.key_file);
    for file in files.iter().filter(|f| **f != key_file) {
        let text = String::from_utf8_lossy(&std::fs::read(file).unwrap()).into_owned();
        places.push((file.display().to_string(), text));
    }
    let secrets = [
        ("the session key", forms_of(&released)),
        ("the bearer", vec![sys.bearer.clone()]),
        ("a question", vec!["tangerine-".to_string()]),
    ];
    let found = leaks(&places, &secrets);
    assert!(found.is_empty(), "{found:?}");
    // Control: the scan reads what it is given.
    let relay_out = sys.relay.journal().remove(0);
    let public = [(
        "the public session key",
        vec![text_of(&sys.td_info, "session_pubkey_hex")],
    )];
    assert_eq!(leaks(&[relay_out], &public).len(), 1);
    let held = std::fs::read_to_string(&key_file).unwrap_or_default();
    notes.check(!key_file.exists() || held.trim() == sys.bearer, || {
        "the key file holds something else than the bearer".into()
    });
    assert!(
        sys.verified["limitations"].as_array().unwrap().len() >= 8,
        "the verdict names its limits"
    );

    // The ends of a relay process. The end of its input: 2.
    let mut orphan = spawn_relay(&sys.world_info, &sys.dir.path().join("orphan"), false);
    let _ = config_of(&orphan);
    orphan.hang_up();
    notes.check(orphan.ended() == 2, || {
        "a relay whose input ended does not exit 2".into()
    });
    notes.done();
}

/// Every refusal of a relay process has one code, said twice: in its
/// `refused` event and as the exit status of the process. Here, a relay
/// that trusts another maker's root than the platform's: 65, the evidence
/// kept in a file of its state directory, no key released. Control: the
/// same TD process would serve a relay that knows the root (scenario 1).
#[test]
fn a_relay_process_that_refuses_exits_with_the_code_of_its_refusal() {
    runtime().block_on(Box::pin(async {
        let dir = tempfile::tempdir().unwrap();
        let mut world = Proc::spawn("world", &json!({}));
        let world_info = world.after(WORLD_UP);
        let mut relay = spawn_relay(&world_info, dir.path(), true);
        let config_b64 = config_of(&relay);
        let oracle = dir.path().join("oracle").join("session-key");
        let (mut td, td_info) = spawn_td(&world_info, &config_b64, &oracle, "echo");
        relay.say(&connect(&text_of(&td_info, "endpoint")));
        let ev = relay.outcome();
        assert_eq!(ev["event"], "refused", "{ev}");
        assert_eq!(ev["code"], 65, "{ev}");
        let evidence = PathBuf::from(text_of(&ev, "evidence_file"));
        assert!(evidence.starts_with(dir.path().join("state")), "{ev}");
        let saved: Value = serde_json::from_slice(&std::fs::read(&evidence).unwrap()).unwrap();
        assert_eq!(saved["refusal"]["code"], 65, "{saved}");
        assert_eq!(relay.ended(), 65, "the process exits with the event's code");
        td.say("stats");
        let stats = td.after(TD_STATS);
        assert_eq!(stats["key_posts"], 0, "{stats}");
        assert_eq!(stats["chat_posts"], 0, "{stats}");
        assert_eq!(stats["phase"], "attested", "{stats}");
        assert!(!oracle.exists(), "no key reached the TD's loader");
        td.hang_up();
        assert_eq!(td.ended(), 0);
        world.hang_up();
        assert_eq!(world.ended(), 0);
    }));
}

// ── Scenario 3: lifecycle and chaos ──────────────────────────────────────────

/// The TD process is killed (SIGKILL) after the first word of a stream. The
/// client is told so in the stream itself (`stream_cut`), the session is
/// `lost`, and the next question is a 503 that says not to retry. Then the
/// TD restarts on the same platform with the same session config, at the
/// address the relay knew: a genuine TD, new keys. The relay does not take
/// it for the one it verified: no connection, no key and no question reach
/// it, and the status still names the key of the TD that died.
///
/// Control: before the kill, the same stream gave its first word.
#[test]
fn a_td_process_killed_mid_stream_loses_the_session_and_its_restart_is_given_nothing() {
    runtime().block_on(Box::pin(async {
        let mut notes = Notes::default();
        let mut sys = System::up("held").await;
        let mut body = question("count for me", 8);
        body["stream"] = json!(true);
        let mut stream = sys.stream(&body).await;
        assert_eq!(stream.text().await, word(0));
        sys.td.kill();
        assert_eq!(sys.td.ended(), -1, "killed by a signal");
        let (events, _) = stream.rest().await;
        let error = error_of(&events).unwrap_or_else(|| panic!("no error event: {events:?}"));
        assert_eq!(error["code"], "stream_cut", "{error}");
        assert_eq!(error["type"], crate::openai::error::TYPE, "{error}");
        assert!(final_of(&events).is_none(), "no receipt for a cut stream");
        assert_eq!(sys.status().await["state"], "lost");

        let refused = sys.post(&question("anyone there", 8)).await;
        assert_eq!(refused.status(), 503);
        assert_eq!(refused.headers()["x-should-retry"], "false");
        let said: Value = refused.json().await.unwrap();
        assert!(
            said["error"]["message"].as_str().unwrap().contains("lost"),
            "{said}"
        );

        // The restart, where the relay's connections now arrive.
        let (mut again, again_info) = spawn_td(
            &sys.world_info,
            &sys.config_b64,
            &System::oracle(&sys.dir, 2),
            "echo",
        );
        assert_ne!(
            again_info["session_pubkey_hex"], sys.td_info["session_pubkey_hex"],
            "a restarted TD draws new keys"
        );
        *sys.route.lock().unwrap() = text_of(&again_info, "endpoint");
        let refused = sys.post(&question("and now", 8)).await;
        assert_eq!(refused.status(), 503);
        let status = sys.status().await;
        assert_eq!(status["state"], "lost", "{status}");
        assert_eq!(
            status["session_public_key_hex"], sys.td_info["session_pubkey_hex"],
            "the session is the dead TD's, not the restarted one's"
        );
        again.say("stats");
        let stats = again.after(TD_STATS);
        assert_eq!(stats["key_posts"], 0, "{stats}");
        assert_eq!(stats["chat_posts"], 0, "{stats}");
        notes.check(stats["handshakes"] == 0, || {
            format!("the restarted TD was reached: {stats}")
        });
        assert!(!System::oracle(&sys.dir, 2).exists(), "no key released");

        sys.stop();
        again.hang_up();
        assert_eq!(again.ended(), 0);
        notes.done();
    }));
}

/// The unique word of client `c` of session `s`: no marker is part of another.
fn marker(s: usize, c: usize) -> String {
    format!("client-{s:02}-{c:02}-end")
}

/// What a client of the crowd did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Whole,
    Streamed,
    /// Streamed, and gone after the first bytes.
    HangsUp,
}

/// One client: its question holds its marker, its answer must hold that
/// marker and no other client's, under the session key of its own TD.
async fn one_client(
    listen: String,
    bearer: String,
    session: String,
    mine: String,
    all: Arc<Vec<String>>,
    kind: Kind,
) -> Result<(), String> {
    let mut body = question(&format!("answer {mine}"), 16);
    if kind != Kind::Whole {
        body["stream"] = json!(true);
    }
    let mut resp = http()
        .post(format!("http://{listen}/v1/chat/completions"))
        .bearer_auth(&bearer)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("{mine}: {e}"))?;
    if resp.status() != 200 {
        return Err(format!("{mine}: status {}", resp.status()));
    }
    let (text, x0ne) = match kind {
        Kind::HangsUp => {
            let _ = resp.chunk().await;
            return Ok(());
        }
        Kind::Whole => {
            let answer: Value = resp.json().await.map_err(|e| format!("{mine}: {e}"))?;
            let text = answer["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            (text, answer["x0ne"].clone())
        }
        Kind::Streamed => {
            let raw = resp.text().await.map_err(|e| format!("{mine}: {e}"))?;
            let chunks: Vec<Value> = raw
                .split("\n\n")
                .filter_map(|e| e.strip_prefix("data: "))
                .filter_map(|d| serde_json::from_str(d).ok())
                .collect();
            let text: String = chunks
                .iter()
                .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
                .collect();
            let x0ne = chunks
                .iter()
                .find(|c| c.get("x0ne").is_some())
                .map_or(Value::Null, |c| c["x0ne"].clone());
            if !raw.ends_with("data: [DONE]\n\n") {
                return Err(format!("{mine}: the stream has no [DONE]: {raw}"));
            }
            (text, x0ne)
        }
    };
    if !text.contains(&mine) {
        return Err(format!("{mine}: its answer is {text:?}"));
    }
    if let Some(other) = all
        .iter()
        .find(|m| **m != mine && text.contains(m.as_str()))
    {
        return Err(format!("{mine}: its answer holds {other}: {text:?}"));
    }
    if x0ne["receipt_verified"] != true || x0ne["session"] != session.as_str() {
        return Err(format!("{mine}: receipt {x0ne}"));
    }
    Ok(())
}

/// Fifty-six clients at once on seven sessions, eight a session: whole
/// answers, streams, and one client a session that hangs up in the middle
/// of its stream. Each client that stayed reads its own marker and no other
/// client's, in an answer whose receipt the relay verified under the key of
/// its own TD; every TD was asked eight questions, every session still
/// serves, and stops clean.
///
/// Seven sessions, not one: a TD answers one question at a time and a turn
/// lasts a second at least (`padding`), so fifty-six clients on one TD are
/// a queue of a minute, not a crowd.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn fifty_six_clients_on_seven_sessions_each_read_their_own_answer_and_no_other() {
    const SESSIONS: usize = 7;
    const CLIENTS: usize = 8;
    let lab = Lab::new().await.short_tick();
    let mut ups = Vec::new();
    for s in 0..SESSIONS {
        let mut env = lab.env();
        env.runtime_dir = lab.tmp.path().join(format!("run-{s}"));
        env.state_dir = lab.tmp.path().join(format!("state-{s}"));
        let opts = TdOpts {
            max_queue: Some(CLIENTS),
            quote_limits: Some(cloud_serve::QuoteLimits::for_queue(CLIENTS)),
            ..TdOpts::default()
        };
        ups.push(bring_up_on(&lab, "cpu", env, opts, false).await);
    }
    let all: Arc<Vec<String>> = Arc::new(
        (0..SESSIONS)
            .flat_map(|s| (0..CLIENTS).map(move |c| marker(s, c)))
            .collect(),
    );
    let mut clients = Vec::new();
    for (s, up) in ups.iter().enumerate() {
        for c in 0..CLIENTS {
            let kind = match c {
                0 => Kind::HangsUp,
                c if c % 2 == 1 => Kind::Streamed,
                _ => Kind::Whole,
            };
            clients.push(tokio::spawn(one_client(
                up.listen.clone(),
                up.bearer.clone(),
                up.td.identity_pubkey_hex.clone(),
                marker(s, c),
                all.clone(),
                kind,
            )));
        }
    }
    let mut failed = Vec::new();
    for client in clients {
        let outcome = tokio::time::timeout(Duration::from_secs(60), client)
            .await
            .expect("every client is answered within 60 s")
            .unwrap();
        if let Err(why) = outcome {
            failed.push(why);
        }
    }
    assert!(failed.is_empty(), "\n{}", failed.join("\n"));
    for up in ups {
        assert_eq!(
            up.td.state.stats.chat_posts.load(SeqCst),
            CLIENTS as u64,
            "every question reached its own TD"
        );
        assert_eq!(relay_state(&up).await, "serving");
        stop(up).await;
    }
}

/// Eight clients at once on a TD whose queue holds two behind the one
/// generating: each is either answered, with its own marker, or told to
/// come back: a 503 in the one error shape, `busy` in its words, a
/// `Retry-After` in seconds and no `x-should-retry: false`. Some of each.
/// The session is not lost for it: it still serves, and answers the next.
///
/// Control: the same question alone, afterwards, is a 200.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn eight_clients_at_once_on_a_queue_of_two_are_answered_or_told_when_to_come_back() {
    const CLIENTS: usize = 8;
    let lab = Lab::new().await;
    let opts = TdOpts {
        max_queue: Some(2),
        // The chat queue is what fills: the quotes of eight connections pass.
        quote_limits: Some(cloud_serve::QuoteLimits::for_queue(2 * CLIENTS)),
        ..TdOpts::default()
    };
    let up = bring_up_on(&lab, "cpu", lab.env(), opts, false).await;
    let mut asked = Vec::new();
    for c in 0..CLIENTS {
        let (listen, bearer, mine) = (up.listen.clone(), up.bearer.clone(), marker(0, c));
        asked.push(tokio::spawn(async move {
            let resp = http()
                .post(format!("http://{listen}/v1/chat/completions"))
                .bearer_auth(&bearer)
                .json(&question(&format!("answer {mine}"), 16))
                .send()
                .await
                .unwrap();
            let (status, headers) = (resp.status().as_u16(), resp.headers().clone());
            let body: Value = resp.json().await.unwrap();
            (mine, status, headers, body)
        }));
    }
    let (mut answered, mut refused) = (0, 0);
    for client in asked {
        let (mine, status, headers, body) = tokio::time::timeout(Duration::from_secs(60), client)
            .await
            .expect("an answer or a refusal within 60 s")
            .unwrap();
        match status {
            200 => {
                let text = body["choices"][0]["message"]["content"].as_str().unwrap();
                assert!(text.contains(&mine), "{mine}: {text}");
                assert_eq!(body["x0ne"]["receipt_verified"], true, "{body}");
                answered += 1;
            }
            503 => {
                let error = &body["error"];
                assert_eq!(error["type"], crate::openai::error::TYPE, "{body}");
                assert!(
                    error["message"].as_str().unwrap().contains("busy"),
                    "{body}"
                );
                let wait: u64 = headers["retry-after"].to_str().unwrap().parse().unwrap();
                assert!((1..=60).contains(&wait), "Retry-After: {wait}");
                assert!(!headers.contains_key("x-should-retry"), "{headers:?}");
                refused += 1;
            }
            other => panic!("{mine}: status {other}: {body}"),
        }
    }
    assert_eq!(answered + refused, CLIENTS);
    assert!(answered >= 1, "nobody was answered");
    assert!(refused >= 1, "eight at once passed a queue of two");
    assert_eq!(relay_state(&up).await, "serving");
    let (status, answer) = ask_whole(&up, 2).await;
    assert_eq!(status, 200, "{answer}");
    stop(up).await;
}

/// `body` POSTed as JSON to the relay's `path` on a connection of its own,
/// and the answer read whole, its body as JSON. The client writes
/// `cap` and 256 KiB more, waits for the answer's first byte and 200 ms,
/// then writes the rest 64 KiB at a time; a write the relay refuses ends
/// the body, and the answer is read all the same.
///
/// Past a cap the relay answers and closes with the body unread (RFC 9110
/// §15.5.14 lets it), and the rest of the body meets the peer's reset. The
/// answer is already in the client's socket, and Linux keeps it there:
/// urllib3 reads it after the failed write (its #1524), and so does this.
/// hyper's client, reqwest's, returns `BodyWrite` and drops it instead: at
/// random under load (3 red `make ci` runs on 2026-11-10, broken pipe), and
/// every time when held so (measured on the relay, 2026-11-10).
async fn post_past_the_cap(
    up: &Up,
    path: &str,
    body: Vec<u8>,
    cap: usize,
) -> axum::http::Response<Value> {
    let (listen, bearer, path) = (up.listen.clone(), up.bearer.clone(), path.to_string());
    let (written, read, got) = tokio::task::spawn_blocking(move || {
        use std::io::Read as _;
        let mut tcp = std::net::TcpStream::connect(&listen).unwrap();
        let bound = Some(Duration::from_secs(60));
        tcp.set_read_timeout(bound).unwrap();
        tcp.set_write_timeout(bound).unwrap();
        write!(
            tcp,
            "POST {path} HTTP/1.1\r\nHost: {listen}\r\nAuthorization: Bearer {bearer}\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .unwrap();
        let (first, rest) = body.split_at((cap + (256 << 10)).min(body.len()));
        let written = tcp.write_all(first).and_then(|()| {
            let _ = tcp.peek(&mut [0u8]);
            std::thread::sleep(Duration::from_millis(200));
            rest.chunks(64 << 10).try_for_each(|c| tcp.write_all(c))
        });
        let mut got = Vec::new();
        let read = tcp.read_to_end(&mut got);
        (written, read, got)
    })
    .await
    .unwrap();
    let text = String::from_utf8_lossy(&got);
    let Some((head, json)) = text.split_once("\r\n\r\n") else {
        panic!("no answer (written: {written:?}, read: {read:?}): {text:?}");
    };
    let status: u16 = head
        .split(' ')
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("{head}"));
    let said: Value = serde_json::from_str(json).unwrap_or_else(|e| panic!("{e}: {text}"));
    axum::http::Response::builder()
        .status(status)
        .body(said)
        .unwrap()
}

/// Bodies past their cap are refused at the cap, in the error shape, and
/// nothing of them reaches the TD: a chat request of 3 MiB, a sealed request
/// and a sealed stream request one byte too long. Sixty-four connections
/// that send half a request line and then nothing hold nobody up: the next
/// question is answered while they are open.
///
/// Control: that next question is a 200 with a verified receipt.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn oversized_bodies_are_refused_at_their_cap_and_silent_connections_hold_nobody_up() {
    use tokio::io::AsyncWriteExt as _;
    let lab = Lab::new().await;
    let up = bring_up(&lab, "cpu").await;
    let sealed = format!("http://{}/v2/e2e/chat", up.listen);

    // axum's default cap, the chat route's: the relay stops reading at it.
    let huge = serde_json::to_vec(&question(&"a".repeat(3 << 20), 2)).unwrap();
    let resp = post_past_the_cap(&up, "/v1/chat/completions", huge, 2 << 20).await;
    assert_eq!(resp.status(), 413);
    let said: Value = resp.into_body();
    assert_eq!(said["error"]["type"], crate::openai::error::TYPE, "{said}");

    for (media, largest) in [
        (
            x0ne_sdk::ohttp::MEDIA_TYPE_REQUEST,
            x0ne_sdk::e2e::MAX_SEALED_REQUEST,
        ),
        (
            x0ne_sdk::ohttp::MEDIA_TYPE_CHUNKED_REQUEST,
            x0ne_sdk::e2e::MAX_SEALED_STREAM_REQUEST,
        ),
    ] {
        let resp = http()
            .post(&sealed)
            .bearer_auth(&up.bearer)
            .header("content-type", media)
            .body(vec![0u8; largest + 1])
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 413, "{media}");
        let said: Value = resp.json().await.unwrap();
        assert_eq!(
            said["error"]["code"],
            crate::local_api::SEALED_REQUEST_TOO_LARGE,
            "{said}"
        );
    }
    let stats = &up.td.state.stats;
    assert_eq!(stats.chat_posts.load(SeqCst), 0);
    assert_eq!(stats.e2e_chat_posts.load(SeqCst), 0);

    let mut silent = Vec::new();
    for _ in 0..64 {
        let mut conn = tokio::net::TcpStream::connect(&up.listen).await.unwrap();
        conn.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: ")
            .await
            .unwrap();
        silent.push(conn);
    }
    let (status, answer) = ask_whole(&up, 2).await;
    assert_eq!(status, 200, "{answer}");
    assert_eq!(answer["x0ne"]["receipt_verified"], true, "{answer}");
    assert_eq!(relay_state(&up).await, "serving");
    drop(silent);
    stop(up).await;
}

// ── Scenario 2: the adversarial middle ───────────────────────────────────────

/// What stands between the relay and the TD lies about who answers, five
/// ways, and the relay refuses each by name, exit 88, before the key and
/// before a question: a TLS key of the middle's own under the TD's genuine
/// document; the TD's stolen TLS key on a connection the middle terminates;
/// a quote taken for another channel; the document of an earlier session;
/// and the genuine document of another session's TD, booted for another
/// relay and steered here.
///
/// Control: with nothing in the middle, the same relay verifies the same
/// TD and is answered.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_middle_that_swaps_the_key_or_serves_another_sessions_document_is_refused_by_name() {
    let lab = Lab::new().await;
    let up = bring_up(&lab, "cpu").await;
    let (status, answer) = ask_whole(&up, 2).await;
    assert_eq!(status, 200, "{answer}");
    stop(up).await;

    for (attack, name, reason) in [
        (
            "a TLS key of its own under the TD's document",
            "TLS key attested is not this connection's",
            "attested TLS key",
        ),
        (
            "a quote taken for another channel",
            "report_data not bound to this channel",
            "does not bind this channel",
        ),
        (
            "the document of an earlier session",
            "client nonce differs",
            "config is not the one this session sent",
        ),
    ] {
        let case = cases()
            .into_iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("no case named {name:?}"));
        let mut s = (case.build)(&lab);
        s.attest = "cpu";
        let (event, exit, td) = refusal(&lab, s).await;
        assert_nothing_sent(attack, &event, exit, &td, reason);
    }

    let mut s = Scenario::honest(&lab);
    s.attest = "cpu";
    s.mitm = true;
    let (event, exit, td) = refusal(&lab, s).await;
    assert_nothing_sent("the TD's stolen TLS key", &event, exit, &td, "exporter");

    // Two sessions. The TD of the first, genuine and attested for the first
    // relay's config, is steered to the second relay and lets it in.
    let mut first = start_relay(lab.args("cpu"), lab.env());
    let first_config = relay_config(&mut first).await;
    let mut second = start_relay(lab.args("cpu"), lab.env());
    let second_config = relay_config(&mut second).await;
    let td = start_td(
        &lab,
        first_config,
        honest_td(&lab),
        model_report(),
        None,
        Box::new(|_, _| {}),
        None,
        TdOpts {
            accept_client_spki: Some(second_config.config.client_spki()),
            ..TdOpts::default()
        },
    )
    .await;
    second
        .send(json!({ "v": 1, "cmd": "connect", "endpoint": td.endpoint, "boot_deadline_unix": now() + 600 }))
        .await;
    let event = second.next_outcome().await;
    let exit = second.exit().await;
    assert_nothing_sent(
        "another session's TD",
        &event,
        exit,
        &td,
        "config is not the one this session sent",
    );
    first.cmd = None;
    assert_eq!(first.exit().await, 2, "the end of input");
}

/// The key document's schema, one version down: what a carrier writes to
/// have an older protocol accepted.
fn one_version_down(schema: &str) -> String {
    let digits = schema
        .chars()
        .rev()
        .take_while(char::is_ascii_digit)
        .count();
    let (name, version) = schema.split_at(schema.len() - digits);
    let version: u32 = version.parse().expect("a schema ends on its version");
    format!("{name}{}", version - 1)
}

/// The carrier of the sealed path (the relay, or anything between it and
/// the caller) alters what it carries, and the caller's SDK or the TD
/// refuses each: a key document replayed for another nonce, or rewritten to
/// the schema one version down; a sealed answer with one byte flipped; a
/// sealed request sent twice; a sealed stream with two chunks swapped, and
/// one cut before its last chunk.
///
/// Control: the same key document verifies for its own nonce, and the same
/// answer opens before its byte is flipped.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_carrier_that_replays_downgrades_flips_reorders_or_cuts_the_sealed_path_is_refused() {
    use x0ne_sdk::attest::{verify_key, Collateral, KeyDocument};
    use x0ne_sdk::e2e::E2eError;
    let mut notes = Notes::default();
    let lab = Lab::new().await;
    let up = bring_up(&lab, "cpu").await;
    let caller = SealedCaller::of(&up, Arc::default());
    let policy = sdk_policy(&lab);

    let nonce = test_random();
    let (status, doc) = caller
        .get(&format!("/v2/e2e/key?nonce={}", hex(&nonce)))
        .await;
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&doc));
    let doc: KeyDocument = serde_json::from_slice(&doc).unwrap();
    let (status, collateral) = caller.get("/v2/e2e/collateral").await;
    assert_eq!(status, 200);
    let collateral: Collateral = serde_json::from_slice(&collateral).unwrap();
    let client = verify_key(&doc, &collateral, &policy, &nonce, now())
        .map(x0ne_sdk::e2e::Client::new)
        .unwrap_or_else(|r| panic!("control: {} {}: {}", r.step, r.code, r.reason));

    let replayed = verify_key(&doc, &collateral, &policy, &test_random(), now())
        .err()
        .map(|r| (r.step, r.code));
    notes.check(replayed == Some(("binding", 88)), || {
        format!("a key document replayed for another nonce: {replayed:?}")
    });
    let mut older = doc.clone();
    older.schema = one_version_down(&doc.schema);
    let downgraded = verify_key(&older, &collateral, &policy, &nonce, now())
        .err()
        .map(|r| (r.step, r.code));
    notes.check(downgraded == Some(("schema", 88)), || {
        format!("a key document one schema down: {downgraded:?}")
    });

    // Whole: one byte of the answer, then the request a second time.
    let (request, pending) = sealed_question(&client, "which way to the harbour");
    let (status, _, response) = caller
        .chat(request.clone(), x0ne_sdk::ohttp::MEDIA_TYPE_REQUEST)
        .await;
    assert_eq!(status, 200);
    let mut flipped = response.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 1;
    let err = pending
        .open_verified(&flipped, &digest(), now())
        .expect_err("a flipped byte opens nothing");
    notes.check(matches!(err, E2eError::Ohttp(_)), || {
        format!("a flipped byte: {err:?}")
    });
    pending
        .open_verified(&response, &digest(), now())
        .expect("control: the answer as the TD sealed it");
    // The TD's refusal of a sealed request says nothing of why, by design.
    let (status, _, _) = caller
        .chat(request, x0ne_sdk::ohttp::MEDIA_TYPE_REQUEST)
        .await;
    assert_eq!(status, 400, "a sealed request sent twice");
    assert_eq!(
        up.td.state.stats.e2e_chat_posts.load(SeqCst),
        2,
        "the replay reached the TD and was refused there"
    );

    // Streamed: two chunks swapped, then the last one withheld.
    let body = serde_json::to_vec(&json!({
        "stream": true,
        "messages": [{ "role": "user", "content": "one two three four" }],
        "max_tokens": 4,
    }))
    .unwrap();
    for attack in ["two chunks swapped", "the last chunk withheld"] {
        let (request, mut pending) = client
            .seal_chat_stream(&body, &test_random(), now())
            .unwrap();
        let back = caller.chat_stream(request).await;
        assert_eq!(back.status, 200, "{attack}");
        let mut chunks = back.chunks();
        assert!(chunks.len() >= 3, "{attack}: {} chunks", chunks.len());
        if attack == "two chunks swapped" {
            chunks.swap(1, 2);
        } else {
            chunks.pop();
        }
        let pushed = chunks.iter().try_for_each(|c| pending.push(c).map(|_| ()));
        let err = pushed
            .and_then(|()| pending.finish(&digest(), now()).map(|_| ()))
            .expect_err(attack);
        notes.check(
            matches!(err, E2eError::Ohttp(_) | E2eError::Stream(_)),
            || format!("{attack}: {err:?}"),
        );
    }
    assert_eq!(relay_state(&up).await, "serving");
    stop(up).await;
    notes.done();
}

// ── Scenario 5: two verifiers, one verdict ───────────────────────────────────

/// Evidence a test changes while the TD serves: honest at boot, so that the
/// session comes up, then `lie` for every quote taken afterwards.
struct Swapped {
    honest: FakeTd,
    lie: Mutex<Option<FakeTd>>,
}

impl TdEvidence for Swapped {
    fn tdx(
        &self,
        report_data: &[u8; 64],
        now_unix: u64,
    ) -> anyhow::Result<crate::evidence::Collected> {
        match &*self.lie.lock().unwrap() {
            Some(lie) => lie.tdx(report_data, now_unix),
            None => self.honest.tdx(report_data, now_unix),
        }
    }

    fn ccel(&self) -> anyhow::Result<Vec<u8>> {
        self.honest.ccel()
    }

    fn gpu<'a>(
        &'a self,
        gpu_nonce: &'a [u8; 32],
    ) -> crate::attest_v2::BoxFuture<
        'a,
        Result<crate::attest_v2::GpuAttested, crate::attest_v2::GpuError>,
    > {
        self.honest.gpu(gpu_nonce)
    }
}

/// The TD's key document for `nonce`, asked of its routes on a connection
/// that has taken no quote yet.
async fn key_document(td: &Td, nonce: &[u8; 32]) -> (u16, Vec<u8>) {
    use hyper::service::Service as _;
    let svc = hyper_util::service::TowerToHyperService::new(cloud_serve::router(td.state.clone()));
    let mut req = axum::http::Request::builder()
        .method("GET")
        .uri(format!("/v2/e2e/key?nonce={}", hex(nonce)))
        .body(axum::body::Body::empty())
        .unwrap();
    req.extensions_mut()
        .insert(cloud_serve::ConnectionQuotes::default());
    let resp = svc.call(req).await.unwrap();
    let status = resp.status().as_u16();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, bytes.to_vec())
}

/// The same lie of a platform, judged by the two verifiers that ship: the
/// relay, on the TD's attestation document, and the SDK, on its key
/// document. Each has its own document and its own code path; the SDK
/// documents its refusal code as the relay's exit code for the same
/// refusal. For every lie of the corpus both refuse, with one code.
///
/// Control: the honest platform is accepted by both (the relay serves, the
/// SDK verifies the key document).
///
/// Not judged here: `client::verify_attestation_document`, the verifier of
/// the local server's v1 document, which binds another report data and
/// carries no event log; no document is common to it and these two.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_relay_and_the_sdk_give_one_verdict_on_the_same_lie_of_a_platform() {
    let lab = Lab::new().await;
    let source = Arc::new(Swapped {
        honest: honest_td(&lab),
        lie: Mutex::new(None),
    });
    let evidence = source.clone();
    let opts = TdOpts {
        session_source: Some(Box::new(move |_| evidence as Arc<dyn TdEvidence>)),
        quote_limits: Some(cloud_serve::QuoteLimits::for_queue(16)),
        ..TdOpts::default()
    };
    let up = bring_up_on(&lab, "cpu", lab.env(), opts, false).await;
    let policy = sdk_policy(&lab);
    let sources = crate::remote_verify::CollateralSources {
        pcs_base_url: lab.pcs.base_url.clone(),
        root_crl_url: lab.pcs.root_crl_url.clone(),
        gce_bucket_base_url: lab.bucket.clone(),
    };
    let sdk_verdict = |lie: Option<FakeTd>| {
        let (source, up, policy, sources) = (&source, &up, &policy, &sources);
        async move {
            *source.lie.lock().unwrap() = lie;
            let nonce = test_random();
            let (status, doc) = key_document(&up.td, &nonce).await;
            assert_eq!(status, 200, "{}", String::from_utf8_lossy(&doc));
            let doc: x0ne_sdk::attest::KeyDocument = serde_json::from_slice(&doc).unwrap();
            let quote = unb64(&doc.quote_b64).unwrap();
            let collateral = crate::remote_verify::sdk_collateral(&http(), sources, &quote)
                .await
                .expect("the collateral of a quote under the lab's root");
            x0ne_sdk::attest::verify_key(&doc, &collateral, policy, &nonce, now())
                .map(|_| ())
                .map_err(|r| (r.code, format!("{}: {}", r.step, r.reason)))
        }
    };
    sdk_verdict(None)
        .await
        .expect("control: the SDK accepts the honest platform the relay accepted");

    let mut drifted = honest_td(&lab);
    let mut rtmr = drifted.rtmr();
    rtmr[0][0] ^= 1;
    drifted.rtmr_in_quote = Some(rtmr);
    let mut other_loader = honest_td(&lab);
    other_loader.events[2] = Event::new_sha384(
        2,
        EV_EFI_BOOT_SERVICES_APPLICATION,
        [0xee; 48],
        b"\\EFI\\BOOT\\BOOTX64.EFI".to_vec(),
    );
    let mut migratable = honest_td(&lab);
    migratable.td_attributes = [0, 0, 0, 0x30, 0, 0, 0, 0];
    let mut unbound = honest_td(&lab);
    unbound.report_data_override = Some([9; 64]);
    let corpus = [
        ("CCEL replay differs from the quote", drifted),
        ("RTMR1: another EFI application", other_loader),
        ("migratable TD", migratable),
        ("report_data not bound to this channel", unbound),
    ];
    let mut disagreements = Vec::new();
    for (name, lie) in corpus {
        let (sdk_code, sdk_reason) = sdk_verdict(Some(lie))
            .await
            .expect_err("the SDK accepts a lie");
        let case = cases()
            .into_iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("no case named {name:?}"));
        let mut s = (case.build)(&lab);
        s.attest = "cpu";
        let (event, relay_code, _) = refusal(&lab, s).await;
        assert_eq!(event["event"], "refused", "{name}: {event}");
        if relay_code != sdk_code {
            disagreements.push(format!(
                "{name}: the relay exits {relay_code} ({}), the SDK says {sdk_code} ({sdk_reason})",
                event["reason"]
            ));
        }
    }
    assert!(disagreements.is_empty(), "\n{}", disagreements.join("\n"));
    *source.lie.lock().unwrap() = None;
    stop(up).await;
}
