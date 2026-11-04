//! The model owner's sealed weights at the binary's edges: `x0ne-infer seal`
//! prints what a key policy names, and a TD given a sealed config checks
//! the sealed file before it collects any evidence or opens any connection.

use std::io::Read;
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use x0ne_infer::cloud_config::{hex, sha256, CloudConfig, SealedSource, SealedWeights};

const BIN: &str = env!("CARGO_BIN_EXE_x0ne-infer");

fn on_a_td() -> bool {
    std::path::Path::new("/sys/kernel/config/tsm/report").exists()
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// `seal` prints the two hashes a policy names; the sealed one is the file's.
#[test]
fn seal_prints_the_hashes_a_policy_names() {
    let dir = tempfile::tempdir().unwrap();
    let (plain, out) = (
        dir.path().join("w.safetensors"),
        dir.path().join("w.sealed"),
    );
    std::fs::write(&plain, b"private weights").unwrap();
    let run = Command::new(BIN)
        // cargo sets LD_LIBRARY_PATH for tests; a release x0ne-infer refuses it (73).
        .env_remove("LD_LIBRARY_PATH")
        .args(["seal", plain.to_str().unwrap(), out.to_str().unwrap()])
        .env("X0NE_INFER_KEY_HEX", "5a".repeat(32))
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert_eq!(
        run.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let sealed = std::fs::read(&out).unwrap();
    assert!(x0ne_infer::seal::is_sealed(&sealed));
    assert!(
        stdout.contains(&format!(
            "weights_sha256 {}",
            hex(&sha256(b"private weights"))
        )),
        "{stdout}"
    );
    assert!(
        stdout.contains(&format!("sealed_sha256 {}", hex(&sha256(&sealed)))),
        "{stdout}"
    );
    assert!(
        stdout.contains(&format!("sealed_size {}\n", sealed.len())),
        "{stdout}"
    );
}

/// A file over AES-GCM's 2^36 - 32 bytes is refused by its size, before it
/// is read or encrypted (2026-11-04: a seal of it panicked in `.expect`).
/// The file is sparse: no 64 GiB on disk, none read.
#[test]
fn seal_refuses_a_file_aes_gcm_cannot_hold_before_reading_it() {
    let dir = tempfile::tempdir().unwrap();
    let (plain, out) = (
        dir.path().join("big.safetensors"),
        dir.path().join("big.sealed"),
    );
    std::fs::File::create(&plain)
        .unwrap()
        .set_len(x0ne_infer::seal::MAX_PLAINTEXT + 1)
        .unwrap();
    let started = std::time::Instant::now();
    let run = Command::new(BIN)
        .env_remove("LD_LIBRARY_PATH")
        .args(["seal", plain.to_str().unwrap(), out.to_str().unwrap()])
        .env("X0NE_INFER_KEY_HEX", "5a".repeat(32))
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert_eq!(run.status.code(), Some(2), "{stderr}");
    assert!(
        stderr.contains(&format!(
            "larger than the {}-byte cap",
            x0ne_infer::seal::MAX_PLAINTEXT
        )),
        "{stderr}"
    );
    assert!(!out.exists(), "a sealed file was written");
    assert!(
        started.elapsed() < Duration::from_secs(30),
        "the file was read"
    );
}

/// A TCP acceptor that counts every connection and answers nothing useful.
fn counting_source() -> (u16, Arc<AtomicUsize>, TcpListener) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
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
    (port, hits, listener)
}

/// A sealed config that names where the file is, and no file on the
/// command line, is not refused: the TD fetches the file after its
/// attestation, as it pulls the catalog model. Off a TD that is 64, and the
/// source is never contacted before.
#[test]
fn a_sealed_source_is_fetched_after_the_evidence_not_before() {
    if on_a_td() {
        eprintln!("skipped: this host exposes configfs-tsm, the 64 path does not apply");
        return;
    }
    let (port, hits, _keep) = counting_source();
    let dir = tempfile::tempdir().unwrap();
    let sealed = x0ne_infer::seal::seal(b"private weights", &[0x5a; 32], &[7; 12]).unwrap();
    let loaded = CloudConfig::new(
        "qwen3:0.6b",
        [0xab; 32],
        [0xcd; 32],
        now() + 3600,
        "cpu",
        443,
        64,
    )
    .with_sealed(SealedWeights {
        sealed_sha256: hex(&sha256(&sealed)),
        digest_hex: "44".repeat(32),
        source: Some(SealedSource {
            url: format!("http://127.0.0.1:{port}/model.sealed"),
            size: sealed.len() as u64,
        }),
    })
    .to_loaded()
    .unwrap();
    let config = dir.path().join("config.json");
    std::fs::write(&config, &loaded.bytes).unwrap();
    let store = dir.path().join("models");
    let out = Command::new(BIN)
        .env_remove("LD_LIBRARY_PATH")
        .args([
            "--cloud-config",
            config.to_str().unwrap(),
            "--attest",
            "cpu",
            "--device",
            "cpu",
            "--model-store",
            store.to_str().unwrap(),
            "--catalog-endpoint",
            "http://127.0.0.1:9",
            "--tls-listen",
            "127.0.0.1:0",
        ])
        .env_remove("X0NE_INFER_KEY_HEX")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(64), "{err}");
    assert!(!err.contains("--sealed-weights"), "{err}");
    assert!(!err.contains("fetch"), "{err}");
    assert_eq!(
        hits.load(Ordering::SeqCst),
        0,
        "the source was contacted before the attestation"
    );
    assert!(!store.exists(), "nothing fetched before the attestation");
}

/// A sealed config and its file must agree before anything else happens:
/// 95 without the file, 77 for another file, and with the right one the
/// binary goes on to the evidence step (64 off a TD), never to the catalog.
#[test]
fn a_sealed_config_checks_its_file_before_any_evidence() {
    if on_a_td() {
        eprintln!("skipped: this host exposes configfs-tsm, the 64 path does not apply");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let sealed = x0ne_infer::seal::seal(b"private weights", &[0x5a; 32], &[7; 12]).unwrap();
    let file = dir.path().join("model.sealed");
    std::fs::write(&file, &sealed).unwrap();
    let config = |sha: [u8; 32]| {
        let loaded = CloudConfig::new(
            "qwen3:0.6b",
            [0xab; 32],
            [0xcd; 32],
            now() + 3600,
            "cpu",
            443,
            64,
        )
        .with_sealed(SealedWeights {
            sealed_sha256: hex(&sha),
            digest_hex: "44".repeat(32),
            source: None,
        })
        .to_loaded()
        .unwrap();
        let path = dir.path().join(format!("config-{}.json", hex(&sha[..4])));
        std::fs::write(&path, &loaded.bytes).unwrap();
        path
    };
    let store = dir.path().join("models");
    let run = |config: &std::path::Path, file: Option<&std::path::Path>| {
        let mut args = vec![
            "--cloud-config".to_string(),
            config.display().to_string(),
            "--attest".into(),
            "cpu".into(),
            "--device".into(),
            "cpu".into(),
            "--model-store".into(),
            store.display().to_string(),
            // Nobody listens: a connection attempt would show as a failed pull.
            "--catalog-endpoint".into(),
            "http://127.0.0.1:9".into(),
            "--tls-listen".into(),
            "127.0.0.1:0".into(),
        ];
        if let Some(f) = file {
            args.extend(["--sealed-weights".into(), f.display().to_string()]);
        }
        let out = Command::new(BIN)
            .env_remove("LD_LIBRARY_PATH")
            .args(&args)
            .env_remove("X0NE_INFER_KEY_HEX")
            .stdin(Stdio::null())
            .output()
            .unwrap();
        (
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).to_string(),
        )
    };
    let right = config(sha256(&sealed));
    let (code, err) = run(&right, None);
    assert_eq!(code, Some(95), "{err}");
    assert!(err.contains("--sealed-weights"), "{err}");
    let (code, err) = run(&config([9; 32]), Some(&file));
    assert_eq!(code, Some(77), "{err}");
    assert!(err.contains("the config names"), "{err}");
    let (code, err) = run(&right, Some(&file));
    assert_eq!(code, Some(64), "{err}");
    assert!(!err.contains("pull"), "{err}");
    assert!(!store.exists(), "nothing pulled before the attestation");
}
