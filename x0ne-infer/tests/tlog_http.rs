//! The pinned log read over HTTP: a local tlog-tiles instance serves the
//! TEST log's directory (and the log Tessera wrote, `tests/tlog/tessera`)
//! on the loopback, and the relay's check refreshes a stale proof from it.
//! Every failure of the reader refuses: a log that cannot be reached, one
//! that serves another origin, a missing or oversized tile, a redirect, a
//! server that hangs. Plain `http://` is taken on the loopback alone; a pin
//! is `https://` (`transparency::pinned`).

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use x0ne_infer::tlog::{NoteKey, TileSource};
use x0ne_infer::transparency::test_log::{
    cosign, log_reference, policy, secret, LocalLog, ORIGIN, WITNESS,
};
use x0ne_infer::transparency::{self, check, entry, DirTiles, HttpTiles, Policy, Tiles};

const DAY: u64 = 24 * 3600;
const T1: u64 = 1_792_627_200; // 2026-10-22T00:00:00Z
const T2: u64 = T1 + 2 * DAY;
const T3: u64 = T2 + 2 * DAY;

/// What the local instance does with a request, besides serving `dir`.
#[derive(Clone, Copy)]
enum Quirk {
    None,
    /// `/checkpoint` answers 302 to `/real/checkpoint`, which serves it.
    RedirectCheckpoint,
    /// Every tile is 1 MiB of zeros.
    HugeTiles,
    /// `/checkpoint` answers after five seconds.
    SlowCheckpoint,
}

struct Instance {
    base: String,
    _runtime: tokio::runtime::Runtime,
}

/// A tlog-tiles instance on the loopback serving `dir`.
fn serve(dir: &Path, quirk: Quirk) -> Instance {
    use axum::body::Body;
    use axum::http::{header, Request, Response, StatusCode};
    let dir: Arc<PathBuf> = Arc::new(dir.to_path_buf());
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    let app = axum::Router::new().fallback(move |req: Request<Body>| {
        let dir = dir.clone();
        async move {
            let path = req.uri().path().trim_start_matches('/').to_string();
            let rel = match quirk {
                Quirk::RedirectCheckpoint if path == "checkpoint" => {
                    return Response::builder()
                        .status(StatusCode::FOUND)
                        .header(header::LOCATION, "/real/checkpoint")
                        .body(Body::empty())
                        .unwrap();
                }
                Quirk::RedirectCheckpoint => path.trim_start_matches("real/").to_string(),
                Quirk::HugeTiles if path.starts_with("tile/") => {
                    return Response::new(Body::from(vec![0u8; 1 << 20]));
                }
                Quirk::SlowCheckpoint if path == "checkpoint" => {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    path
                }
                _ => path,
            };
            match std::fs::read(dir.join(&rel)) {
                Ok(b) => Response::new(Body::from(b)),
                Err(_) => Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(Body::empty())
                    .unwrap(),
            }
        }
    });
    runtime.spawn(async move {
        let l = tokio::net::TcpListener::from_std(listener).unwrap();
        axum::serve(l, app).await.unwrap();
    });
    Instance {
        base: format!("http://127.0.0.1:{port}/"),
        _runtime: runtime,
    }
}

/// The TEST log with a one-day window, read at `base`.
fn http_policy(base: &str) -> Policy {
    Policy {
        max_age_s: DAY,
        tiles: Tiles::Https(base.to_string()),
        ..policy()
    }
}

/// `n` references logged in `dir/tlog`, the i-th cosigned at `times[i]`.
fn logged(dir: &Path, times: &[u64]) -> Vec<PathBuf> {
    times
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let p = dir.join(format!("x0ne-cvm-0.{i}.0.json"));
            std::fs::write(&p, format!("{{\"reference\": {i}}}")).unwrap();
            log_reference(&p, *t).unwrap();
            p
        })
        .collect()
}

fn checked(
    p: &Path,
    policy: &Policy,
    state: &Path,
    now: u64,
) -> Result<transparency::Logged, String> {
    let bytes = std::fs::read(p).unwrap();
    let proof = std::fs::read_to_string(transparency::proof_path(p)).unwrap();
    let source = policy.source(p);
    check(&bytes, &proof, policy, source.as_ref(), state, now)
}

#[test]
fn a_stale_proof_is_refreshed_from_the_log_served_over_http() {
    let dir = tempfile::tempdir().unwrap();
    let refs = logged(dir.path(), &[T1, T2]);
    let instance = serve(&dir.path().join("tlog"), Quirk::None);
    let policy = http_policy(&instance.base);
    let state = dir.path().join("state");
    // The first proof's cosignature is two days old at T2: the log's
    // current checkpoint (size 2, cosigned at T2) is read over HTTP, with
    // the tiles of the consistency proof from 1 to 2.
    let l = checked(&refs[0], &policy, &state, T2 + 60).unwrap();
    assert_eq!((l.index, l.size, l.refreshed), (0, 2, true), "{l:?}");
    assert_eq!(l.cosigned_by, [(WITNESS.to_string(), T2)]);
    // The second proof is fresh: nothing is read.
    let l = checked(&refs[1], &policy, &state, T2 + 60).unwrap();
    assert_eq!((l.index, l.size, l.refreshed), (1, 2, false), "{l:?}");
}

#[test]
fn an_unreachable_log_refuses_a_stale_proof() {
    let dir = tempfile::tempdir().unwrap();
    let refs = logged(dir.path(), &[T1, T2]);
    // A port nothing listens on.
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let policy = http_policy(&format!("http://127.0.0.1:{port}/"));
    let err = checked(&refs[0], &policy, &dir.path().join("state"), T2 + 60).unwrap_err();
    assert!(err.contains("checkpoint stale"), "{err}");
    assert!(err.contains("current checkpoint is unreadable"), "{err}");
    // A fresh proof needs no log.
    checked(&refs[1], &policy, &dir.path().join("state"), T2 + 60).unwrap();
}

#[test]
fn a_log_serving_another_origin_a_missing_tile_or_a_huge_tile_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let refs = logged(dir.path(), &[T1, T2]);
    let tlog = dir.path().join("tlog");

    // Another log under the same key material, cosigned fresh: its
    // signature names its own origin, which is not the pinned key's name.
    let other = tempfile::tempdir().unwrap();
    let mut log = LocalLog::open_as(other.path(), "other.invalid/log", secret("log")).unwrap();
    for p in &refs {
        log.append(&entry(&std::fs::read(p).unwrap())).unwrap();
    }
    let note = log.publish().unwrap();
    let note = cosign(
        other.path(),
        &DirTiles(other.path().to_path_buf()),
        &note,
        &log.log,
        T2,
    )
    .unwrap();
    std::fs::write(other.path().join("checkpoint"), note).unwrap();
    let instance = serve(other.path(), Quirk::None);
    let err = checked(
        &refs[0],
        &http_policy(&instance.base),
        &dir.path().join("state"),
        T2 + 60,
    )
    .unwrap_err();
    assert!(err.contains("not signed by the pinned log"), "{err}");

    // The log's checkpoint without its tiles.
    let bare = tempfile::tempdir().unwrap();
    std::fs::copy(tlog.join("checkpoint"), bare.path().join("checkpoint")).unwrap();
    let instance = serve(bare.path(), Quirk::None);
    let err = checked(
        &refs[0],
        &http_policy(&instance.base),
        &dir.path().join("state"),
        T2 + 60,
    )
    .unwrap_err();
    assert!(err.contains("404"), "{err}");

    // Tiles beyond the cap.
    let instance = serve(&tlog, Quirk::HugeTiles);
    let err = checked(
        &refs[0],
        &http_policy(&instance.base),
        &dir.path().join("state"),
        T2 + 60,
    )
    .unwrap_err();
    assert!(err.contains("cap"), "{err}");
}

#[test]
fn a_redirect_from_the_log_is_not_followed() {
    let dir = tempfile::tempdir().unwrap();
    let refs = logged(dir.path(), &[T1, T2]);
    let instance = serve(&dir.path().join("tlog"), Quirk::RedirectCheckpoint);
    let policy = http_policy(&instance.base);
    let err = checked(&refs[0], &policy, &dir.path().join("state"), T2 + 60).unwrap_err();
    assert!(err.contains("302"), "{err}");
    // The reader itself: the redirect is an error, not a second request.
    let err = HttpTiles::new(&instance.base)
        .unwrap()
        .checkpoint()
        .unwrap_err();
    assert!(err.contains("302"), "{err}");
}

#[test]
fn a_log_that_hangs_times_out() {
    let dir = tempfile::tempdir().unwrap();
    let _ = logged(dir.path(), &[T1]);
    let instance = serve(&dir.path().join("tlog"), Quirk::SlowCheckpoint);
    let reader = HttpTiles::with_timeout(&instance.base, Duration::from_secs(1)).unwrap();
    let start = Instant::now();
    let err = reader.checkpoint().unwrap_err();
    assert!(
        start.elapsed() < Duration::from_secs(4),
        "{:?}",
        start.elapsed()
    );
    assert!(
        err.contains("timed out") || err.contains("timeout"),
        "{err}"
    );
    // A tile of the same instance is served at once.
    assert_eq!(reader.read("tile/0/000.p/1").unwrap().len(), 32);
}

#[test]
fn the_reader_takes_https_or_the_loopback_alone_and_tile_paths_alone() {
    assert!(HttpTiles::new("https://log.example/").is_ok());
    assert!(HttpTiles::new("https://log.example/prefix").is_ok());
    assert!(HttpTiles::new("http://127.0.0.1:1/").is_ok());
    assert!(HttpTiles::new("http://localhost:1/").is_ok());
    for bad in [
        "http://log.example/",
        "http://172.16.0.1/",
        "http://user@127.0.0.1/",
        "ftp://log.example/",
        "log.example",
        "",
    ] {
        let err = HttpTiles::new(bad).unwrap_err();
        assert!(err.contains("https"), "{bad}: {err}");
    }
    let r = HttpTiles::new("http://127.0.0.1:1/").unwrap();
    for bad in [
        "../x",
        "tile/../../x",
        "checkpoint",
        "/tile/0/000",
        "tile/0/A",
    ] {
        let err = r.read(bad).unwrap_err();
        assert!(err.contains("not a tile path"), "{bad}: {err}");
    }
}

#[test]
fn a_fork_served_over_http_is_refused_and_both_heads_are_kept() {
    let dir = tempfile::tempdir().unwrap();
    let refs = logged(dir.path(), &[T1, T2]);
    let state = dir.path().join("state");
    let honest = serve(&dir.path().join("tlog"), Quirk::None);
    // This machine verified the honest head of size 2.
    let l = checked(&refs[1], &http_policy(&honest.base), &state, T2 + 60).unwrap();
    assert_eq!((l.size, l.refreshed), (2, false));

    // Another view of the log, for one user: the same first leaf, another
    // second one, under the same origin and key, cosigned fresh at T3 by a
    // witness that never saw the honest head.
    let forked = tempfile::tempdir().unwrap();
    let mut log = LocalLog::open_as(forked.path(), ORIGIN, secret("log")).unwrap();
    log.append(&entry(&std::fs::read(&refs[0]).unwrap()))
        .unwrap();
    log.append(&entry(b"{\"reference\": \"patched\"}")).unwrap();
    let note = log.publish().unwrap();
    let note = cosign(
        forked.path(),
        &DirTiles(forked.path().to_path_buf()),
        &note,
        &log.log,
        T3,
    )
    .unwrap();
    std::fs::write(forked.path().join("checkpoint"), note).unwrap();
    let served = serve(forked.path(), Quirk::None);
    // The first reference's proof (size 1, cosigned at T1) is stale at T3:
    // the forked head is read, consistent with size 1, and refused against
    // the honest head of the same size this machine kept.
    let err = checked(&refs[0], &http_policy(&served.base), &state, T3 + 60).unwrap_err();
    assert!(err.contains("the log forked"), "{err}");
    assert!(err.contains("two signed checkpoints of size 2"), "{err}");
    let kept: Vec<_> = std::fs::read_dir(state.join("tlog"))
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("fork-"))
        .collect();
    assert_eq!(kept.len(), 1, "{kept:?}");
    let evidence = std::fs::read_to_string(kept[0].path()).unwrap();
    assert!(evidence.contains("# last verified") && evidence.contains("# presented"));
    assert_eq!(
        evidence.matches(&format!("— {ORIGIN} ")).count(),
        2,
        "{evidence}"
    );
}

/// The log Tessera v1.0.4 wrote (the log library of Rekor v2), served
/// locally: the proof of entry 42 holds the size-100 checkpoint cosigned at
/// 1792673709; the log's head is 300, cosigned 30 s later. With a 30 s
/// window, 40 s after the proof, the head is read over HTTP and the
/// consistency from 100 to 300 comes from its tiles.
#[test]
fn the_tessera_log_served_over_http_refreshes_a_stale_proof() {
    let tessera = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/tlog/tessera");
    let vkeys = std::fs::read_to_string(tessera.join("vkeys")).unwrap();
    let key = |role: &str| {
        NoteKey::parse(
            vkeys
                .lines()
                .find_map(|l| l.strip_prefix(&format!("{role} ")))
                .unwrap(),
        )
        .unwrap()
    };
    let instance = serve(&tessera.join("log"), Quirk::None);
    let policy = Policy {
        origin: "interop.invalid/tessera-log".into(),
        log: key("log"),
        witnesses: vec![key("witness")],
        quorum: 1,
        max_age_s: 30,
        tiles: Tiles::Https(instance.base.clone()),
    };
    let bytes = std::fs::read(tessera.join("r-042.json")).unwrap();
    let proof = std::fs::read_to_string(tessera.join("r-042.json.tlog-proof")).unwrap();
    let state = tempfile::tempdir().unwrap();
    let source = policy.source(&tessera.join("r-042.json"));
    let now = 1_792_673_709 + 40;
    let l = check(&bytes, &proof, &policy, source.as_ref(), state.path(), now).unwrap();
    assert_eq!((l.index, l.size, l.refreshed), (42, 300, true), "{l:?}");
    assert_eq!(l.cosigned_by[0].1, 1_792_673_739);
    // Once the head's cosignature is out of the window too, nothing newer exists.
    let err = check(
        &bytes,
        &proof,
        &policy,
        source.as_ref(),
        state.path(),
        now + 31,
    )
    .unwrap_err();
    assert!(err.contains("has 0 fresh cosignatures"), "{err}");
}
