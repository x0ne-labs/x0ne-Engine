//! Cloud mode attests BEFORE it pulls the model. Off a TD, with the
//! catalog pointed at a loopback server that counts connections, the binary
//! must exit 64 within seconds and the counter must stay at zero: no outbound
//! connection of any kind precedes the evidence step. A 0.2.0-beta.2 build pulled first and,
//! without network, never reached 64 (measured on the qemu CVM, 2026-10-14).

use std::io::Read;
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use x0ne_infer::cloud_config::CloudConfig;

const BIN: &str = env!("CARGO_BIN_EXE_x0ne-infer");

#[path = "support/evidence_host.rs"]
mod evidence_host;

/// A TCP acceptor that counts every connection and answers nothing useful.
fn counting_catalog() -> (String, Arc<AtomicUsize>, TcpListener) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = hits.clone();
    let acceptor = listener.try_clone().unwrap();
    std::thread::spawn(move || {
        for stream in acceptor.incoming() {
            counter.fetch_add(1, Ordering::SeqCst);
            if let Ok(mut s) = stream {
                let mut buf = [0u8; 1024];
                let _ = s.set_read_timeout(Some(Duration::from_millis(200)));
                let _ = s.read(&mut buf);
            }
        }
    });
    (endpoint, hits, listener)
}

fn on_a_td() -> bool {
    evidence_host::may_collect_evidence()
}

#[test]
fn off_a_td_cloud_mode_exits_64_before_any_catalog_connection() {
    if on_a_td() {
        eprintln!("skipped: configfs-tsm is writable here, the 64 path does not apply");
        return;
    }
    let (endpoint, hits, _keep) = counting_catalog();
    let dir = tempfile::tempdir().unwrap();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let config = CloudConfig::new(
        "qwen3:0.6b",
        [0xab; 32],
        [0xcd; 32],
        now + 3600,
        "cpu",
        443,
        64,
    )
    .to_loaded()
    .unwrap();
    let config_path = dir.path().join("config.json");
    std::fs::write(&config_path, &config.bytes).unwrap();
    let store = dir.path().join("models");

    let started = Instant::now();
    let out = Command::new(BIN)
        // cargo sets LD_LIBRARY_PATH for tests; a release x0ne-infer refuses it (73).
        .env_remove("LD_LIBRARY_PATH")
        .args([
            "--cloud-config",
            config_path.to_str().unwrap(),
            "--attest",
            "cpu",
            "--device",
            "cpu",
            "--model-store",
            store.to_str().unwrap(),
            "--catalog-endpoint",
            &endpoint,
            "--tls-listen",
            "127.0.0.1:0",
        ])
        .env_remove("X0NE_INFER_KEY_HEX")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    let elapsed = started.elapsed();
    let err = String::from_utf8_lossy(&out.stderr);

    assert_eq!(out.status.code(), Some(64), "stderr: {err}");
    assert!(err.contains("no evidence, not serving"), "stderr: {err}");
    assert!(!err.contains("pulling"), "the pull was started: {err}");
    assert!(
        elapsed < Duration::from_secs(10),
        "64 took {elapsed:?}: something waited on the network first"
    );
    assert_eq!(
        hits.load(Ordering::SeqCst),
        0,
        "the catalog was contacted before the attestation"
    );
    assert!(
        !store.exists(),
        "the model store was created before the attestation"
    );

    // Control: the counter does count. One connection from here, one hit.
    std::net::TcpStream::connect(endpoint.trim_start_matches("http://")).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while hits.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "the counting catalog does not count"
    );
}

/// The catalog endpoint stays https off the loopback: a test flag must not
/// become a plaintext mirror in production.
#[test]
fn a_plain_http_catalog_off_the_loopback_is_refused_before_anything() {
    if on_a_td() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let config = CloudConfig::new(
        "qwen3:0.6b",
        [0xab; 32],
        [0xcd; 32],
        now + 3600,
        "cpu",
        443,
        64,
    )
    .to_loaded()
    .unwrap();
    let config_path = dir.path().join("config.json");
    std::fs::write(&config_path, &config.bytes).unwrap();
    let out = Command::new(BIN)
        .env_remove("LD_LIBRARY_PATH")
        .args([
            "--cloud-config",
            config_path.to_str().unwrap(),
            "--attest",
            "cpu",
            "--model-store",
            dir.path().join("models").to_str().unwrap(),
            "--catalog-endpoint",
            "http://mirror.example:8080",
        ])
        .env_remove("X0NE_INFER_KEY_HEX")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(96), "stderr: {err}");
    assert!(err.contains("https only"), "stderr: {err}");
}
