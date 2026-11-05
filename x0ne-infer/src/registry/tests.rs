//! `pull` against a local HTTP server that plays the Hub: resolve URLs,
//! byte ranges, and the ways a source can lie.

use super::*;
use axum::extract::{Path as UrlPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

const REV: &str = "c1899de289a04d12100db370d81485cdf75e47ca";

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Honest,
    /// Answers 200 with the whole file even when asked for a range.
    IgnoreRange,
    /// Serves `model.safetensors` with one byte flipped.
    Corrupt,
    /// Sends the file twice over.
    Oversize,
    /// Holds `model.safetensors` for 600 ms before answering.
    Slow,
    /// Serves the first half of `model.safetensors`, honestly framed.
    Short,
    /// Answers a bounded range with its Content-Range but half its bytes.
    TruncateRange,
    /// Like `TruncateRange`, the first time each bounded range is asked only.
    TruncateOnce,
    /// 503 to a bounded range starting at this offset.
    Unavailable(usize),
}

#[derive(Default)]
struct Seen {
    ranges: Vec<String>,
    gets: usize,
    /// GETs that reached a file's bytes, by file.
    by_file: HashMap<String, usize>,
    /// Hits on `/resolve/`, by file.
    resolves: HashMap<String, usize>,
    in_flight: usize,
    max_in_flight: usize,
}

struct Hub {
    files: HashMap<String, Vec<u8>>,
    mode: Mode,
    seen: Mutex<Seen>,
    /// Held before each answer, so that concurrent requests overlap.
    delay: std::time::Duration,
    /// `/resolve/` answers 302 to `/cdn/`, as the Hub does for LFS files.
    redirect: bool,
}

/// `bytes=a-` or `bytes=a-b`.
fn parse_range(r: &str) -> (usize, Option<usize>) {
    let (a, b) = r.trim_start_matches("bytes=").split_once('-').unwrap();
    (
        a.parse().unwrap(),
        (!b.is_empty()).then(|| b.parse().unwrap()),
    )
}

async fn resolve(
    State(hub): State<Arc<Hub>>,
    UrlPath((o, n, rev, file)): UrlPath<(String, String, String, String)>,
    headers: HeaderMap,
) -> Response {
    *hub.seen
        .lock()
        .unwrap()
        .resolves
        .entry(file.clone())
        .or_default() += 1;
    if hub.redirect {
        let to = format!("/cdn/{o}/{n}/{rev}/{file}");
        return (StatusCode::FOUND, [("location", to)]).into_response();
    }
    answer(hub, rev, file, headers).await
}

async fn cdn(
    State(hub): State<Arc<Hub>>,
    UrlPath((_o, _n, rev, file)): UrlPath<(String, String, String, String)>,
    headers: HeaderMap,
) -> Response {
    answer(hub, rev, file, headers).await
}

async fn answer(hub: Arc<Hub>, rev: String, file: String, headers: HeaderMap) -> Response {
    if hub.mode == Mode::Slow && file == "model.safetensors" {
        hub.seen.lock().unwrap().gets += 1;
        tokio::time::sleep(std::time::Duration::from_millis(600)).await;
        hub.seen.lock().unwrap().gets -= 1;
    }
    {
        let mut seen = hub.seen.lock().unwrap();
        seen.in_flight += 1;
        seen.max_in_flight = seen.max_in_flight.max(seen.in_flight);
    }
    if !hub.delay.is_zero() {
        tokio::time::sleep(hub.delay).await;
    }
    let response = serve_bytes(&hub, &rev, &file, &headers);
    hub.seen.lock().unwrap().in_flight -= 1;
    response
}

fn serve_bytes(hub: &Hub, rev: &str, file: &str, headers: &HeaderMap) -> Response {
    let mut seen = hub.seen.lock().unwrap();
    seen.gets += 1;
    let Some(body) = hub.files.get(file).filter(|_| rev == REV) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    *seen.by_file.entry(file.to_string()).or_default() += 1;
    let mut body = body.clone();
    match hub.mode {
        Mode::Corrupt if file == "model.safetensors" => body[3] ^= 1,
        Mode::Oversize if file == "model.safetensors" => body = body.repeat(2),
        Mode::Short if file == "model.safetensors" => body.truncate(body.len() / 2),
        _ => {}
    }
    let range = headers
        .get("range")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let asked_before = range.as_ref().is_some_and(|r| seen.ranges.contains(r));
    if let Some(r) = &range {
        seen.ranges.push(r.clone());
    }
    match (range, hub.mode) {
        (Some(r), m) if m != Mode::IgnoreRange => {
            let (from, to) = parse_range(&r);
            let total = body.len();
            if from >= total {
                return StatusCode::RANGE_NOT_SATISFIABLE.into_response();
            }
            if to.is_some() && m == Mode::Unavailable(from) {
                return StatusCode::SERVICE_UNAVAILABLE.into_response();
            }
            let last = to.unwrap_or(total - 1).min(total - 1);
            let mut part = body[from..=last].to_vec();
            let cut = match m {
                Mode::TruncateRange => to.is_some(),
                Mode::TruncateOnce => to.is_some() && !asked_before,
                _ => false,
            };
            if cut {
                part.truncate(part.len() / 2);
            }
            (
                StatusCode::PARTIAL_CONTENT,
                [("content-range", format!("bytes {from}-{last}/{total}"))],
                part,
            )
                .into_response()
        }
        _ => (StatusCode::OK, body).into_response(),
    }
}

struct Opts {
    delay: std::time::Duration,
    redirect: bool,
    transfer: Option<Transfer>,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            delay: std::time::Duration::ZERO,
            redirect: false,
            transfer: None,
        }
    }
}

async fn serve(hub: Arc<Hub>) -> String {
    let app = axum::Router::new()
        .route("/{o}/{n}/resolve/{rev}/{file}", axum::routing::get(resolve))
        .route("/cdn/{o}/{n}/{rev}/{file}", axum::routing::get(cdn))
        .with_state(hub);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await });
    format!("http://{addr}")
}

fn model_files() -> HashMap<String, Vec<u8>> {
    let mut m = HashMap::new();
    m.insert(
        "config.json".into(),
        br#"{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}"#.to_vec(),
    );
    m.insert("tokenizer.json".into(), b"{\"tok\":1}".to_vec());
    m.insert(
        "model.safetensors".into(),
        (0..200_000u32).flat_map(|i| i.to_le_bytes()).collect(),
    );
    m
}

fn entry_for(files: &HashMap<String, Vec<u8>>, name: &str) -> Entry {
    let mut paths: Vec<&String> = files.keys().collect();
    paths.sort();
    let specs: String = paths
        .iter()
        .map(|p| {
            format!(
                "{{ path = \"{p}\", size = {}, sha256 = \"{}\" }},",
                files[*p].len(),
                sha256_hex(&files[*p])
            )
        })
        .collect();
    let toml = format!(
        "schema = 1\n[[model]]\nname = \"{name}\"\nrepo = \"o/m\"\nrevision = \"{REV}\"\n\
         architecture = \"qwen3\"\nlicense = \"Apache-2.0\"\nfiles = [{specs}]\n"
    );
    Catalog::parse(&toml).unwrap().models.remove(0)
}

struct Fixture {
    _dir: tempfile::TempDir,
    store: Store,
    source: Source,
    hub: Arc<Hub>,
}

async fn fixture(files: HashMap<String, Vec<u8>>, mode: Mode) -> Fixture {
    fixture_with(files, mode, Opts::default()).await
}

async fn fixture_with(files: HashMap<String, Vec<u8>>, mode: Mode, opts: Opts) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let (hub, source) = hub_with(files, mode, &opts).await;
    Fixture {
        store: Store::new(dir.path().join("cache")),
        _dir: dir,
        source,
        hub,
    }
}

/// Another server over the same files, for a second pull into the same store.
async fn hub_with(files: HashMap<String, Vec<u8>>, mode: Mode, opts: &Opts) -> (Arc<Hub>, Source) {
    let hub = Arc::new(Hub {
        files,
        mode,
        seen: Mutex::default(),
        delay: opts.delay,
        redirect: opts.redirect,
    });
    let mut source = Source::new(&serve(hub.clone()).await).unwrap();
    if let Some(t) = opts.transfer {
        source = source.with_transfer(t);
    }
    (hub, source)
}

#[tokio::test]
async fn pull_installs_verified_files_and_a_record() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let fx = fixture(files.clone(), Mode::Honest).await;
    let p = pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    assert!(!p.already);
    assert_eq!(p.installed.dir, fx.store.model_dir("m", "1"));
    for (name, bytes) in &files {
        assert_eq!(&std::fs::read(p.installed.dir.join(name)).unwrap(), bytes);
    }
    let i = fx.store.installed("m", "1").unwrap().unwrap();
    assert!(i.record.matches(&e));
    fx.store.verify(&i).unwrap();
    // Nothing left behind but the model and the lock.
    assert_eq!(
        std::fs::read_dir(fx.store.root.join("partial"))
            .unwrap()
            .count(),
        0
    );
    assert_eq!(std::fs::read_dir(fx.store.staging()).unwrap().count(), 0);

    let gets = fx.hub.seen.lock().unwrap().gets;
    let again = pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    assert!(again.already);
    assert_eq!(
        fx.hub.seen.lock().unwrap().gets,
        gets,
        "no network when up to date"
    );
}

/// One file by its URL (a model owner's sealed weights): the same transfer,
/// the same hash of the whole, left verified in `partial/`, and found there
/// complete by a later call without a request.
#[tokio::test]
async fn fetch_file_brings_one_url_to_a_verified_partial_and_once_only() {
    let files = model_files();
    let bytes = files["model.safetensors"].clone();
    let fx = fixture(files, Mode::Honest).await;
    let url = format!("{}/o/n/resolve/{REV}/model.safetensors", fx.source.endpoint);
    let spec = FileSpec {
        path: "model.sealed".into(),
        size: bytes.len() as u64,
        sha256: sha256_hex(&bytes),
    };
    let path = fetch_file(&fx.store, &fx.source, "owner:1", &url, &spec, &mut Quiet)
        .await
        .unwrap();
    assert_eq!(path, fx.store.partial(&spec.sha256));
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    let gets = fx.hub.seen.lock().unwrap().gets;
    let again = fetch_file(&fx.store, &fx.source, "owner:1", &url, &spec, &mut Quiet)
        .await
        .unwrap();
    assert_eq!(again, path);
    assert_eq!(
        fx.hub.seen.lock().unwrap().gets,
        gets,
        "a complete file is not fetched again"
    );
}

/// SECURITY (altered sealed file): other bytes under the URL, or another
/// size than the one named, are refused and nothing is kept on disk.
#[tokio::test]
async fn fetch_file_refuses_other_bytes_and_another_size_and_keeps_nothing() {
    let files = model_files();
    let bytes = files["model.safetensors"].clone();
    let spec = FileSpec {
        path: "model.sealed".into(),
        size: bytes.len() as u64,
        sha256: sha256_hex(&bytes),
    };
    let fx = fixture(files.clone(), Mode::Corrupt).await;
    let url = format!("{}/o/n/resolve/{REV}/model.safetensors", fx.source.endpoint);
    let err = fetch_file(&fx.store, &fx.source, "owner:1", &url, &spec, &mut Quiet)
        .await
        .unwrap_err();
    assert!(matches!(err, PullError::Integrity(_)), "{err}");
    assert!(err.to_string().contains("does not match"), "{err}");
    assert!(
        !fx.store.partial(&spec.sha256).exists(),
        "bad bytes are deleted"
    );

    let fx = fixture(files, Mode::Honest).await;
    let url = format!("{}/o/n/resolve/{REV}/model.safetensors", fx.source.endpoint);
    let longer = FileSpec {
        size: bytes.len() as u64 + 1,
        ..spec.clone()
    };
    let err = fetch_file(&fx.store, &fx.source, "owner:1", &url, &longer, &mut Quiet)
        .await
        .unwrap_err();
    assert!(matches!(err, PullError::Integrity(_)), "{err}");
    assert!(err.to_string().contains("declares"), "{err}");
    assert!(!fx.store.partial(&spec.sha256).exists());
}

/// The TD's fetch of the owner's sealed file from the config's source
/// (`model_key::fetch_sealed`): the bytes the config pins come back, under
/// the model store. SECURITY (altered sealed file): other bytes under the
/// URL are 77 and nothing is kept; a source that is not https nor loopback
/// http is 96 before any request.
#[tokio::test]
async fn the_sealed_file_is_fetched_against_the_configs_pin_or_refused() {
    use crate::cloud_config::{SealedSource, SealedWeights};
    let sealed = crate::seal::seal(b"private weights", &[0x5a; 32], &[7; 12]).unwrap();
    let mut files = model_files();
    files.insert("model.safetensors".into(), sealed.clone());
    let cfg = SealedWeights {
        sealed_sha256: sha256_hex(&sealed),
        digest_hex: "44".repeat(32),
        source: None,
    };
    let src = |fx: &Fixture| SealedSource {
        url: format!("{}/o/n/resolve/{REV}/model.safetensors", fx.source.endpoint),
        size: sealed.len() as u64,
    };
    let kept = |root: &std::path::Path| -> Vec<String> {
        std::fs::read_dir(root.join("sealed").join("partial"))
            .map(|d| {
                d.flatten()
                    .map(|e| e.file_name().to_string_lossy().into())
                    .collect()
            })
            .unwrap_or_default()
    };

    let fx = fixture(files.clone(), Mode::Honest).await;
    let root = tempfile::tempdir().unwrap();
    let got = crate::model_key::fetch_sealed(root.path(), &src(&fx), &cfg, "owner:1")
        .await
        .unwrap();
    assert_eq!(got, sealed);
    assert_eq!(kept(root.path()), [format!("{}.part", cfg.sealed_sha256)]);

    let fx = fixture(files, Mode::Corrupt).await;
    let root = tempfile::tempdir().unwrap();
    let (code, why) = crate::model_key::fetch_sealed(root.path(), &src(&fx), &cfg, "owner:1")
        .await
        .unwrap_err();
    assert_eq!(code, 77, "{why}");
    assert!(why.contains("does not match"), "{why}");
    assert!(kept(root.path()).is_empty(), "{:?}", kept(root.path()));

    let elsewhere = SealedSource {
        url: "ftp://w.example/model.sealed".into(),
        size: 5,
    };
    let (code, why) = crate::model_key::fetch_sealed(root.path(), &elsewhere, &cfg, "owner:1")
        .await
        .unwrap_err();
    assert_eq!(code, 96, "{why}");
    assert!(why.contains("https only"), "{why}");
}

#[tokio::test]
async fn a_corrupted_file_is_refused_and_nothing_is_installed() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let fx = fixture(files, Mode::Corrupt).await;
    let err = pull(&fx.store, &fx.source, &e, &mut Quiet)
        .await
        .unwrap_err();
    assert!(matches!(err, PullError::Integrity(_)), "{err}");
    assert!(
        err.to_string().contains("does not match the catalog"),
        "{err}"
    );
    assert!(!fx.store.model_dir("m", "1").exists());
    let w = e
        .files
        .iter()
        .find(|f| f.path == "model.safetensors")
        .unwrap();
    assert!(
        !fx.store.partial(&w.sha256).exists(),
        "bad bytes are deleted"
    );
    assert_eq!(std::fs::read_dir(fx.store.staging()).unwrap().count(), 0);
}

#[tokio::test]
async fn a_server_sending_more_than_the_catalog_size_is_cut_off() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let fx = fixture(files, Mode::Oversize).await;
    let err = pull(&fx.store, &fx.source, &e, &mut Quiet)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("more than the catalog"), "{err}");
    assert!(!fx.store.model_dir("m", "1").exists());
}

#[tokio::test]
async fn an_interrupted_download_resumes_with_a_range() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let fx = fixture(files.clone(), Mode::Honest).await;
    let w = e
        .files
        .iter()
        .find(|f| f.path == "model.safetensors")
        .unwrap();
    std::fs::create_dir_all(fx.store.root.join("partial")).unwrap();
    std::fs::write(
        fx.store.partial(&w.sha256),
        &files["model.safetensors"][..1000],
    )
    .unwrap();
    pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    assert_eq!(
        fx.hub.seen.lock().unwrap().ranges,
        vec!["bytes=1000-".to_string()]
    );
    fx.store
        .verify(&fx.store.installed("m", "1").unwrap().unwrap())
        .unwrap();
}

#[tokio::test]
async fn a_server_ignoring_the_range_restarts_the_file() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let fx = fixture(files.clone(), Mode::IgnoreRange).await;
    let w = e
        .files
        .iter()
        .find(|f| f.path == "model.safetensors")
        .unwrap();
    std::fs::create_dir_all(fx.store.root.join("partial")).unwrap();
    std::fs::write(
        fx.store.partial(&w.sha256),
        &files["model.safetensors"][..1000],
    )
    .unwrap();
    pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    let installed = fx.store.model_dir("m", "1").join("model.safetensors");
    assert_eq!(
        std::fs::read(installed).unwrap(),
        files["model.safetensors"]
    );
}

#[tokio::test]
async fn a_poisoned_partial_is_refused_not_trusted() {
    // Bytes left in partial/ by someone else: the hash covers them too.
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let fx = fixture(files.clone(), Mode::Honest).await;
    let w = e
        .files
        .iter()
        .find(|f| f.path == "model.safetensors")
        .unwrap();
    std::fs::create_dir_all(fx.store.root.join("partial")).unwrap();
    std::fs::write(fx.store.partial(&w.sha256), vec![0xAA; 1000]).unwrap();
    let err = pull(&fx.store, &fx.source, &e, &mut Quiet)
        .await
        .unwrap_err();
    assert!(matches!(err, PullError::Integrity(_)), "{err}");
    assert!(!fx.store.partial(&w.sha256).exists());
    // The next attempt starts clean and succeeds.
    pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
}

#[tokio::test]
async fn remote_code_in_config_is_refused() {
    let mut files = model_files();
    files.insert(
        "config.json".into(),
        br#"{"model_type":"qwen3","auto_map":{"AutoModel":"modeling.Evil"}}"#.to_vec(),
    );
    let e = entry_for(&files, "m:1");
    let fx = fixture(files, Mode::Honest).await;
    let err = pull(&fx.store, &fx.source, &e, &mut Quiet)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("auto_map"), "{err}");
    assert!(!fx.store.model_dir("m", "1").exists());
}

#[tokio::test]
async fn a_config_naming_another_architecture_is_refused() {
    let mut files = model_files();
    files.insert("config.json".into(), br#"{"model_type":"olmo2"}"#.to_vec());
    let e = entry_for(&files, "m:1");
    let fx = fixture(files, Mode::Honest).await;
    let err = pull(&fx.store, &fx.source, &e, &mut Quiet)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("the catalog says qwen3"), "{err}");
}

#[tokio::test]
async fn a_missing_revision_is_a_network_error_with_a_hint() {
    let files = model_files();
    let mut e = entry_for(&files, "m:1");
    e.revision = "0".repeat(40);
    let fx = fixture(files, Mode::Honest).await;
    let err = pull(&fx.store, &fx.source, &e, &mut Quiet)
        .await
        .unwrap_err();
    assert!(matches!(err, PullError::Network(_)), "{err}");
    assert!(err.to_string().contains("404"), "{err}");
}

#[tokio::test]
async fn tampering_after_install_is_caught_by_verify() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let fx = fixture(files, Mode::Honest).await;
    let p = pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    let w = p.installed.dir.join("model.safetensors");
    let mut bytes = std::fs::read(&w).unwrap();
    bytes[10] ^= 1;
    std::fs::write(&w, bytes).unwrap();
    assert!(fx
        .store
        .verify(&p.installed)
        .unwrap_err()
        .contains("sha256"));
}

#[tokio::test]
async fn a_new_catalog_revision_replaces_the_installed_copy() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let fx = fixture(files.clone(), Mode::Honest).await;
    pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    // Same bytes, but the record says otherwise: simulate an old pull.
    let rec = fx.store.model_dir("m", "1").join(store::RECORD);
    let mut r: Record = serde_json::from_slice(&std::fs::read(&rec).unwrap()).unwrap();
    r.revision = "1".repeat(40);
    std::fs::write(&rec, serde_json::to_vec(&r).unwrap()).unwrap();
    let p = pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    assert!(!p.already);
    assert_eq!(
        fx.store
            .installed("m", "1")
            .unwrap()
            .unwrap()
            .record
            .revision,
        REV
    );
    assert_eq!(
        std::fs::read_dir(fx.store.root.join("trash"))
            .unwrap()
            .count(),
        0
    );
}

#[tokio::test]
async fn rm_removes_the_model_and_its_partials() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let fx = fixture(files, Mode::Honest).await;
    pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    std::fs::write(fx.store.partial(&e.files[0].sha256), b"x").unwrap();
    assert!(fx.store.remove("m", "1", Some(&e)).unwrap());
    assert!(!fx.store.model_dir("m", "1").exists());
    assert!(!fx.store.partial(&e.files[0].sha256).exists());
    assert!(!fx.store.remove("m", "1", Some(&e)).unwrap());
    assert!(fx.store.list().is_empty());
}

/// Two shards and their index, plus config and tokenizer; the bytes of tensor `b`.
fn sharded_files() -> (HashMap<String, Vec<u8>>, Vec<u8>) {
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;
    let a: Vec<u8> = [1f32, 2.0].iter().flat_map(|x| x.to_le_bytes()).collect();
    let b: Vec<u8> = [3f32].iter().flat_map(|x| x.to_le_bytes()).collect();
    let s1 = safetensors::serialize(
        [("a", TensorView::new(Dtype::F32, vec![2], &a).unwrap())],
        None,
    )
    .unwrap();
    let s2 = safetensors::serialize(
        [("b", TensorView::new(Dtype::F32, vec![1], &b).unwrap())],
        None,
    )
    .unwrap();
    let mut files = model_files();
    files.remove("model.safetensors");
    files.insert("model-00001-of-00002.safetensors".into(), s1);
    files.insert("model-00002-of-00002.safetensors".into(), s2);
    files.insert(
        crate::shards::INDEX_FILE.into(),
        br#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#
            .to_vec(),
    );
    (files, b)
}

#[tokio::test]
async fn shards_are_verified_and_installed_as_published() {
    let (files, b) = sharded_files();
    let e = entry_for(&files, "m:2");
    assert!(e.is_sharded());
    let fx = fixture(files.clone(), Mode::Honest).await;
    let p = pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    let dir = &p.installed.dir;
    for (name, bytes) in &files {
        assert_eq!(&std::fs::read(dir.join(name)).unwrap(), bytes, "{name}");
    }
    assert!(!dir.join("model.safetensors").exists(), "no merge");
    let i = fx.store.installed("m", "2").unwrap().unwrap();
    assert!(i.record.matches(&e) && !i.record.is_merged());
    fx.store.verify(&i).unwrap();
    let mapped = crate::shards::map_dir(dir).unwrap();
    let st = safetensors::SafeTensors::deserialize(&mapped.files[1].1).unwrap();
    assert_eq!(st.tensor("b").unwrap().data(), &b[..]);
}

/// An install from before 0.2.0-beta.6 holds a merged model.safetensors that
/// no v2 pin describes. A failure means it would be kept, then refused at 77.
#[tokio::test]
async fn a_merged_install_from_before_beta6_is_replaced_by_the_shards() {
    let (files, _) = sharded_files();
    let e = entry_for(&files, "m:2");
    let fx = fixture(files.clone(), Mode::Honest).await;
    let dir = pull(&fx.store, &fx.source, &e, &mut Quiet)
        .await
        .unwrap()
        .installed
        .dir;
    // What a beta.5 pull left: the merged file, the shards in `downloaded`.
    crate::shards::merge(&dir, &dir.join("model.safetensors")).unwrap();
    let merged = std::fs::read(dir.join("model.safetensors")).unwrap();
    for f in e
        .weight_files()
        .map(|f| f.path.clone())
        .chain([crate::shards::INDEX_FILE.into()])
    {
        std::fs::remove_file(dir.join(f)).unwrap();
    }
    let rec = dir.join(store::RECORD);
    let mut r: Record = serde_json::from_slice(&std::fs::read(&rec).unwrap()).unwrap();
    r.downloaded = r.files.clone();
    r.files.retain(|f| !f.path.starts_with("model"));
    r.files.push(FileSpec {
        path: "model.safetensors".into(),
        size: merged.len() as u64,
        sha256: sha256_hex(&merged),
    });
    std::fs::write(&rec, serde_json::to_vec(&r).unwrap()).unwrap();
    let old = fx.store.installed("m", "2").unwrap().unwrap();
    assert!(old.record.is_merged() && !old.record.matches(&e));
    fx.store.verify(&old).unwrap();

    let again = pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    assert!(!again.already);
    assert!(!dir.join("model.safetensors").exists());
    for (name, bytes) in &files {
        assert_eq!(&std::fs::read(dir.join(name)).unwrap(), bytes, "{name}");
    }
    assert!(!fx
        .store
        .installed("m", "2")
        .unwrap()
        .unwrap()
        .record
        .is_merged());
}

/// The pin as docs/cloud.md tells an operator to compute it with coreutils:
/// `{ printf 'x0ne-model-digest/2\n'; sha256sum <files> | LC_ALL=C sort -k2; } | sha256sum`.
fn coreutils_pin(dir: &std::path::Path, e: &Entry) -> String {
    let mut names: Vec<&str> = e.served_files().map(|f| f.path.as_str()).collect();
    names.sort_unstable();
    let mut manifest = String::from("x0ne-model-digest/2\n");
    for n in names {
        manifest += &format!(
            "{}  {n}\n",
            sha256_hex(&std::fs::read(dir.join(n)).unwrap())
        );
    }
    sha256_hex(manifest.as_bytes())
}

/// Cloud mode's intake: each file against its catalog sha256, the catalog's shards
/// mapped, the served digest against the pin, before a byte is served.
/// 96: no pin, or a file the catalog does not pin; 77: not the pinned model.
#[tokio::test]
async fn cloud_intake_serves_a_sharded_model_only_on_its_pinned_digest() {
    let (files, _) = sharded_files();
    let fx = fixture(files.clone(), Mode::Honest).await;
    let mut e = entry_for(&files, "m:2");

    let (code, why) = pull_served(&fx.store, &fx.source, &e, &mut Quiet)
        .await
        .err()
        .expect("served without a pinned digest");
    assert_eq!(code, 96, "{why}");
    assert!(why.contains("no served digest pinned yet"), "{why}");
    assert!(
        fx.store.list().is_empty(),
        "nothing pulled before the pin check"
    );

    // A substituted shard: the source's bytes are not the catalog's.
    let mut bad_shard = e.clone();
    bad_shard.digest_sha256 = Some("0".repeat(64));
    bad_shard
        .files
        .iter_mut()
        .find(|f| f.path.starts_with("model-00002"))
        .unwrap()
        .sha256 = "1".repeat(64);
    let (code, why) = pull_served(&fx.store, &fx.source, &bad_shard, &mut Quiet)
        .await
        .err()
        .expect("served a shard the catalog does not pin");
    assert_eq!(code, 96, "{why}");
    assert!(why.contains("does not match the catalog"), "{why}");

    let wrong = "2".repeat(64);
    e.digest_sha256 = Some(wrong.clone());
    let (code, why) = pull_served(&fx.store, &fx.source, &e, &mut Quiet)
        .await
        .err()
        .expect("served a model whose digest is not the pinned one");
    assert_eq!(code, 77, "{why}");
    assert!(why.contains(&wrong), "{why}");

    let dir = fx.store.installed("m", "2").unwrap().unwrap().dir;
    let pin = coreutils_pin(&dir, &e);
    assert_eq!(e.catalog_digest().unwrap(), pin);
    e.digest_sha256 = Some(pin.clone());
    // Files nobody pinned, dropped next to the install: never listed, never read.
    std::fs::write(dir.join("model.safetensors"), b"planted").unwrap();
    std::fs::write(dir.join("model-00003-of-00002.safetensors"), b"planted").unwrap();
    // The catalog's order does not change what is served.
    e.files.reverse();
    let served = pull_served(&fx.store, &fx.source, &e, &mut Quiet)
        .await
        .unwrap();
    assert_eq!(served.digest_hex, pin);
    assert_eq!(served.config_json, files["config.json"]);
    assert_eq!(served.tokenizer_json, files["tokenizer.json"]);
    let w = served.weight_files();
    assert_eq!(w.index_json, Some(&files[crate::shards::INDEX_FILE][..]));
    let mut names: Vec<&str> = w.files.iter().map(|(n, _)| *n).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors"
        ]
    );
    for (n, bytes) in &w.files {
        assert_eq!(*bytes, &files[*n][..], "{n}");
    }
}

/// The publisher's own files disagree: the catalog pins them faithfully, so
/// `pull` succeeds; the intake refuses (72) before any digest is trusted.
#[tokio::test]
async fn cloud_intake_refuses_an_index_that_disagrees_with_the_shards() {
    const S1: &str = "model-00001-of-00002.safetensors";
    const S2: &str = "model-00002-of-00002.safetensors";
    type Edit = fn(&mut HashMap<String, Vec<u8>>);
    let cases: [(&str, Edit); 4] = [
        ("modified index", |f| {
            f.insert(
                crate::shards::INDEX_FILE.into(),
                format!(r#"{{"weight_map":{{"a":"{S2}","b":"{S1}"}}}}"#).into_bytes(),
            );
        }),
        ("missing shard", |f| {
            f.remove(S2);
        }),
        ("index naming a third shard", |f| {
            f.insert(
                crate::shards::INDEX_FILE.into(),
                format!(r#"{{"weight_map":{{"a":"{S1}","b":"{S2}","c":"model-00003-of-00003.safetensors"}}}}"#)
                    .into_bytes(),
            );
        }),
        ("extra shard the index does not name", |f| {
            use safetensors::tensor::TensorView;
            let c = [0u8; 4];
            let view = TensorView::new(safetensors::Dtype::F32, vec![1], &c).unwrap();
            let bytes = safetensors::serialize([("c", view)], None).unwrap();
            f.insert("model-00003-of-00003.safetensors".into(), bytes);
        }),
    ];
    for (case, edit) in cases {
        let (mut files, _) = sharded_files();
        edit(&mut files);
        let mut e = entry_for(&files, "m:2");
        e.digest_sha256 = Some(e.catalog_digest().unwrap());
        let fx = fixture(files, Mode::Honest).await;
        let (code, why) = pull_served(&fx.store, &fx.source, &e, &mut Quiet)
            .await
            .err()
            .unwrap_or_else(|| panic!("{case}: served"));
        assert_eq!(code, 72, "{case}: {why}");
    }
}

#[test]
fn plain_http_is_refused_except_to_the_loopback() {
    assert!(Source::new("http://huggingface.co").is_err());
    assert!(Source::new("ftp://x").is_err());
    assert!(Source::new("http://127.0.0.1:9").is_ok());
    assert!(Source::new(DEFAULT_ENDPOINT).is_ok());
    assert!(Source::new("http://127.0.0.1.evil.example:9").is_err());
}

#[test]
fn a_loopback_prefix_before_userinfo_is_not_the_loopback() {
    for url in [
        "http://127.0.0.1:x@evil.example",
        "http://localhost:1@evil.example:9/",
        "http://[::1]:x@evil.example",
    ] {
        assert!(Source::new(url).is_err(), "{url} accepted as loopback");
    }
}

/// qwen3:0.6b and qwen3:1.7b list the same tokenizer.json: two names, one
/// partial file. The second pull must not write into the first one's partial.
#[tokio::test]
async fn two_names_sharing_a_file_never_download_it_at_once() {
    let files = model_files();
    let (a, b) = (entry_for(&files, "m:1"), entry_for(&files, "m:2"));
    let fx = fixture(files, Mode::Slow).await;
    let (mut q1, mut q2) = (Quiet, Quiet);
    let first = pull(&fx.store, &fx.source, &a, &mut q1);
    let second = async {
        // config.json done, model.safetensors in flight.
        while fx.hub.seen.lock().unwrap().gets < 2 {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        pull(&fx.store, &fx.source, &b, &mut q2).await
    };
    let (first, second) = tokio::join!(first, second);
    let err = second.expect_err("second pull ran alongside the first");
    assert!(err.to_string().contains("another x0ne-engine"), "{err}");
    first.unwrap();
}

#[tokio::test]
async fn a_failed_pull_keeps_verified_partials_and_no_staging() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let mut served = files.clone();
    served.remove("model.safetensors");
    let fx = fixture(served, Mode::Honest).await;
    let err = pull(&fx.store, &fx.source, &e, &mut Quiet)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("404"), "{err}");
    // config.json, in flight when the weights failed, is finished and kept.
    let sha = |p: &str| &e.files.iter().find(|f| f.path == p).unwrap().sha256;
    assert!(fx.store.partial(sha("config.json")).exists());
    assert!(!fx.store.partial(sha("model.safetensors")).exists());
    assert_eq!(std::fs::read_dir(fx.store.staging()).unwrap().count(), 0);
    assert!(!fx.store.model_dir("m", "1").exists());
}

/// A damaged install answered "up to date". A pull re-hashes what is
/// installed and replaces it when the bytes moved.
#[tokio::test]
async fn pull_over_a_damaged_install_reverifies_and_replaces_it() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let fx = fixture(files.clone(), Mode::Honest).await;
    let p = pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    let w = p.installed.dir.join("model.safetensors");
    let mut bytes = std::fs::read(&w).unwrap();
    bytes[10] ^= 1;
    std::fs::write(&w, bytes).unwrap();
    let again = pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    assert!(!again.already, "a damaged install was reported up to date");
    fx.store.verify(&again.installed).unwrap();
    assert_eq!(std::fs::read(&w).unwrap(), files["model.safetensors"]);
}

/// A source whose file is shorter than the catalog is a wrong file
/// (65), not a network blip to resume (69) forever; nothing is left behind.
#[tokio::test]
async fn a_source_shorter_than_the_catalog_is_an_integrity_error() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let fx = fixture(files, Mode::Short).await;
    let w = e
        .files
        .iter()
        .find(|f| f.path == "model.safetensors")
        .unwrap();
    for attempt in 0..2 {
        let err = pull(&fx.store, &fx.source, &e, &mut Quiet)
            .await
            .unwrap_err();
        assert!(
            matches!(err, PullError::Integrity(_)),
            "attempt {attempt}: {err}"
        );
        assert!(!fx.store.partial(&w.sha256).exists(), "attempt {attempt}");
    }
    assert!(!fx.store.model_dir("m", "1").exists());
}

/// Lock files piled up under locks/ after every pull.
#[tokio::test]
async fn locks_are_removed_once_a_pull_ends() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let fx = fixture(files, Mode::Honest).await;
    pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    let left: Vec<_> = std::fs::read_dir(fx.store.root.join("locks"))
        .unwrap()
        .flatten()
        .map(|d| d.file_name())
        .collect();
    assert!(left.is_empty(), "locks left behind: {left:?}");
}

/// Installed files were 0644, readable by every local account.
#[cfg(unix)]
#[tokio::test]
async fn installed_files_are_not_world_readable() {
    use std::os::unix::fs::PermissionsExt;
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let fx = fixture(files, Mode::Honest).await;
    let p = pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    let mut seen = 0;
    for d in std::fs::read_dir(&p.installed.dir).unwrap().flatten() {
        let mode = d.metadata().unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o640, "{:?} is {mode:o}", d.file_name());
        seen += 1;
    }
    assert_eq!(seen, 4, "three model files and the record");
    let dir_mode = std::fs::metadata(&p.installed.dir)
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(dir_mode & 0o007, 0, "model directory is {dir_mode:o}");
}

/// `list` cannot re-hash gigabytes; it must still not say "ok" over a
/// truncated file.
#[tokio::test]
async fn a_truncated_install_fails_the_quick_size_check() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let fx = fixture(files, Mode::Honest).await;
    let p = pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    fx.store.check_sizes(&p.installed).unwrap();
    let w = p.installed.dir.join("model.safetensors");
    std::fs::OpenOptions::new()
        .write(true)
        .open(&w)
        .unwrap()
        .set_len(10)
        .unwrap();
    assert!(fx.store.check_sizes(&p.installed).is_err());
}

// ---- Parallel transfer: ranges of one file and files at once, under
// bounds; every file still hashed whole against the catalog at the end.

/// model.safetensors (800 000 bytes) in eight ranges of 100 000.
fn ranged(files: usize, connections: usize, retries: u32) -> Transfer {
    Transfer {
        files,
        connections,
        chunk: 100_000,
        retries,
        retry_delay: std::time::Duration::ZERO,
    }
}

fn weights_sha(e: &Entry) -> String {
    e.files
        .iter()
        .find(|f| f.path == "model.safetensors")
        .unwrap()
        .sha256
        .clone()
}

/// The range state kept next to a partial download.
fn ranges_path(store: &Store, sha: &str) -> std::path::PathBuf {
    store.root.join("partial").join(format!("{sha}.ranges"))
}

fn installed_weights(fx: &Fixture) -> Vec<u8> {
    std::fs::read(fx.store.model_dir("m", "1").join("model.safetensors")).unwrap()
}

#[tokio::test]
async fn a_large_file_is_fetched_in_parallel_byte_ranges() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let opts = Opts {
        delay: std::time::Duration::from_millis(100),
        transfer: Some(ranged(2, 4, 0)),
        ..Opts::default()
    };
    let fx = fixture_with(files.clone(), Mode::Honest, opts).await;
    pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    assert_eq!(installed_weights(&fx), files["model.safetensors"]);
    let seen = fx.hub.seen.lock().unwrap();
    let mut got: Vec<(usize, usize)> = seen
        .ranges
        .iter()
        .map(|r| parse_range(r))
        .map(|(a, b)| (a, b.expect("a bounded range") + 1))
        .collect();
    got.sort_unstable();
    let want: Vec<(usize, usize)> = (0..8).map(|k| (k * 100_000, (k + 1) * 100_000)).collect();
    assert_eq!(got, want, "every byte asked once, in ranges of the chunk");
    assert!(
        (2..=4).contains(&seen.max_in_flight),
        "{} requests at once, bound 4",
        seen.max_in_flight
    );
}

/// Eight small files, three at a time: never one after the other, never more.
#[tokio::test]
async fn files_are_fetched_at_once_within_the_file_bound() {
    let mut files = model_files();
    files.remove("model.safetensors");
    for i in 1..=3 {
        files.insert(
            format!("model-{i:05}-of-00003.safetensors"),
            vec![i as u8; 1000],
        );
    }
    files.insert(crate::shards::INDEX_FILE.into(), b"{}".to_vec());
    files.insert("LICENSE".into(), b"Apache-2.0".to_vec());
    files.insert("NOTICE".into(), b"notice".to_vec());
    let e = entry_for(&files, "m:1");
    assert_eq!(e.files.len(), 8);
    let opts = Opts {
        delay: std::time::Duration::from_millis(200),
        transfer: Some(ranged(3, 8, 0)),
        ..Opts::default()
    };
    let fx = fixture_with(files.clone(), Mode::Honest, opts).await;
    let p = pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    for (name, bytes) in &files {
        assert_eq!(&std::fs::read(p.installed.dir.join(name)).unwrap(), bytes);
    }
    let most = fx.hub.seen.lock().unwrap().max_in_flight;
    assert!((2..=3).contains(&most), "{most} requests at once, bound 3");
}

#[tokio::test]
async fn a_source_ignoring_ranges_is_read_in_one_stream() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let opts = Opts {
        transfer: Some(ranged(2, 4, 0)),
        ..Opts::default()
    };
    let fx = fixture_with(files.clone(), Mode::IgnoreRange, opts).await;
    pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    assert_eq!(installed_weights(&fx), files["model.safetensors"]);
    assert_eq!(
        fx.hub.seen.lock().unwrap().by_file["model.safetensors"],
        1,
        "the whole file once, not once per range"
    );
    assert!(!ranges_path(&fx.store, &weights_sha(&e)).exists());
}

/// A range that ends early is asked again; its bytes are never counted as done.
#[tokio::test]
async fn a_truncated_range_is_asked_again_and_never_kept() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let opts = || Opts {
        transfer: Some(ranged(2, 4, 2)),
        ..Opts::default()
    };
    let once = fixture_with(files.clone(), Mode::TruncateOnce, opts()).await;
    pull(&once.store, &once.source, &e, &mut Quiet)
        .await
        .unwrap();
    assert_eq!(installed_weights(&once), files["model.safetensors"]);
    let ranges = once.hub.seen.lock().unwrap().ranges.clone();
    let mut unique = ranges.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 8, "{ranges:?}");
    assert_eq!(
        ranges.len(),
        2 * unique.len(),
        "each cut range asked twice: {ranges:?}"
    );

    let always = fixture_with(files.clone(), Mode::TruncateRange, opts()).await;
    let err = pull(&always.store, &always.source, &e, &mut Quiet)
        .await
        .unwrap_err();
    assert!(matches!(err, PullError::Network(_)), "{err}");
    assert!(err.to_string().contains("attempts"), "{err}");
    assert!(!always.store.model_dir("m", "1").exists());
    // Had a cut range been kept as done, this pull would fail its sha256.
    let (_, honest) = hub_with(files.clone(), Mode::Honest, &opts()).await;
    pull(&always.store, &honest, &e, &mut Quiet).await.unwrap();
    assert_eq!(installed_weights(&always), files["model.safetensors"]);
}

#[tokio::test]
async fn a_digest_mismatch_after_ranges_deletes_the_file_and_its_ranges() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let opts = Opts {
        transfer: Some(ranged(2, 4, 0)),
        ..Opts::default()
    };
    let fx = fixture_with(files, Mode::Corrupt, opts).await;
    let err = pull(&fx.store, &fx.source, &e, &mut Quiet)
        .await
        .unwrap_err();
    assert!(matches!(err, PullError::Integrity(_)), "{err}");
    assert!(
        err.to_string().contains("does not match the catalog"),
        "{err}"
    );
    let sha = weights_sha(&e);
    assert!(!fx.store.partial(&sha).exists(), "bad bytes are deleted");
    assert!(!ranges_path(&fx.store, &sha).exists());
    assert_eq!(std::fs::read_dir(fx.store.staging()).unwrap().count(), 0);
    assert!(!fx.store.model_dir("m", "1").exists());
}

#[tokio::test]
async fn a_ranged_source_declaring_another_size_is_refused() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let opts = Opts {
        transfer: Some(ranged(2, 4, 0)),
        ..Opts::default()
    };
    let fx = fixture_with(files, Mode::Short, opts).await;
    let err = pull(&fx.store, &fx.source, &e, &mut Quiet)
        .await
        .unwrap_err();
    assert!(matches!(err, PullError::Integrity(_)), "{err}");
    assert!(err.to_string().contains("declares 400000 bytes"), "{err}");
    let sha = weights_sha(&e);
    assert!(!fx.store.partial(&sha).exists());
    assert!(!ranges_path(&fx.store, &sha).exists());
}

/// One connection: ranges go in order, the last one fails, the next pull
/// asks for that range only.
#[tokio::test]
async fn an_interrupted_ranged_download_fetches_only_the_missing_ranges() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let opts = || Opts {
        transfer: Some(ranged(2, 1, 1)),
        ..Opts::default()
    };
    let fx = fixture_with(files.clone(), Mode::Unavailable(700_000), opts()).await;
    let err = pull(&fx.store, &fx.source, &e, &mut Quiet)
        .await
        .unwrap_err();
    assert!(matches!(err, PullError::Network(_)), "{err}");
    assert!(err.to_string().contains("503"), "{err}");
    let sha = weights_sha(&e);
    assert!(fx.store.partial(&sha).exists(), "written ranges are kept");
    assert!(ranges_path(&fx.store, &sha).exists());
    assert!(!fx.store.model_dir("m", "1").exists());

    let (hub, honest) = hub_with(files.clone(), Mode::Honest, &opts()).await;
    pull(&fx.store, &honest, &e, &mut Quiet).await.unwrap();
    assert_eq!(installed_weights(&fx), files["model.safetensors"]);
    assert_eq!(
        hub.seen.lock().unwrap().ranges,
        vec!["bytes=700000-799999".to_string()]
    );
    assert!(!ranges_path(&fx.store, &sha).exists());
}

/// A ranges file naming bytes past the end of its partial (cut short, or gone):
/// those bytes are asked again. Until 2026-10-20 the list was trusted, the
/// missing bytes read as zeros and the end hash deleted the whole download.
#[tokio::test]
async fn ranges_past_the_end_of_their_partial_are_asked_again() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let sha = weights_sha(&e);
    let opts = || Opts {
        transfer: Some(ranged(2, 1, 1)),
        ..Opts::default()
    };
    for kept in [Some(250_000u64), None] {
        let fx = fixture_with(files.clone(), Mode::Unavailable(700_000), opts()).await;
        pull(&fx.store, &fx.source, &e, &mut Quiet)
            .await
            .unwrap_err();
        assert!(ranges_path(&fx.store, &sha).exists());
        let part = fx.store.partial(&sha);
        match kept {
            Some(len) => std::fs::OpenOptions::new()
                .write(true)
                .open(&part)
                .and_then(|f| f.set_len(len))
                .unwrap(),
            None => std::fs::remove_file(&part).unwrap(),
        }
        let (hub, honest) = hub_with(files.clone(), Mode::Honest, &opts()).await;
        if let Err(err) = pull(&fx.store, &honest, &e, &mut Quiet).await {
            panic!("partial kept {kept:?}: {err}");
        }
        assert_eq!(installed_weights(&fx), files["model.safetensors"]);
        let mut asked: Vec<(usize, usize)> = hub
            .seen
            .lock()
            .unwrap()
            .ranges
            .iter()
            .map(|r| parse_range(r))
            .map(|(a, b)| (a, b.expect("a bounded range") + 1))
            .collect();
        asked.sort_unstable();
        let from = kept.unwrap_or(0) as usize;
        let want: Vec<(usize, usize)> = (from..800_000)
            .step_by(100_000)
            .map(|a| (a, (a + 100_000).min(800_000)))
            .collect();
        assert_eq!(asked, want, "partial kept {kept:?}");
        assert!(!ranges_path(&fx.store, &sha).exists());
    }
}

/// Ranges written, then a source that ignores ranges: the file restarts from
/// byte 0 in one stream, holes and all discarded.
#[tokio::test]
async fn a_ranged_partial_restarts_whole_when_the_source_ignores_ranges() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let opts = || Opts {
        transfer: Some(ranged(2, 1, 0)),
        ..Opts::default()
    };
    let fx = fixture_with(files.clone(), Mode::Unavailable(300_000), opts()).await;
    pull(&fx.store, &fx.source, &e, &mut Quiet)
        .await
        .unwrap_err();
    assert!(ranges_path(&fx.store, &weights_sha(&e)).exists());
    let (_, plain) = hub_with(files.clone(), Mode::IgnoreRange, &opts()).await;
    pull(&fx.store, &plain, &e, &mut Quiet).await.unwrap();
    assert_eq!(installed_weights(&fx), files["model.safetensors"]);
}

/// The Hub answers /resolve/ with a redirect to its CDN. Each range goes to
/// the CDN directly: one resolve per file, not one per range.
#[tokio::test]
async fn ranges_go_to_the_redirect_target_once_resolved() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let opts = Opts {
        redirect: true,
        transfer: Some(ranged(2, 4, 0)),
        ..Opts::default()
    };
    let fx = fixture_with(files.clone(), Mode::Honest, opts).await;
    pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    assert_eq!(installed_weights(&fx), files["model.safetensors"]);
    let seen = fx.hub.seen.lock().unwrap();
    assert_eq!(seen.resolves["model.safetensors"], 1);
    assert_eq!(seen.by_file["model.safetensors"], 8);
}

#[tokio::test]
async fn rm_removes_the_range_state_too() {
    let files = model_files();
    let e = entry_for(&files, "m:1");
    let fx = fixture(files, Mode::Honest).await;
    pull(&fx.store, &fx.source, &e, &mut Quiet).await.unwrap();
    let sha = weights_sha(&e);
    std::fs::write(fx.store.partial(&sha), b"x").unwrap();
    std::fs::write(ranges_path(&fx.store, &sha), b"x").unwrap();
    assert!(fx.store.remove("m", "1", Some(&e)).unwrap());
    assert!(!fx.store.partial(&sha).exists());
    assert!(!ranges_path(&fx.store, &sha).exists());
}

/// Every file under `root`, relative to it, sorted.
fn files_under(root: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut dirs = vec![root.to_path_buf()];
    while let Some(d) = dirs.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                dirs.push(p);
            } else {
                out.push(p.strip_prefix(root).unwrap().display().to_string());
            }
        }
    }
    out.sort();
    out
}

/// The owner's sealed file, as `model_key::fetch_sealed` is given it.
fn sealed_case(sealed: &[u8]) -> (HashMap<String, Vec<u8>>, crate::cloud_config::SealedWeights) {
    let mut files = model_files();
    files.insert("model.safetensors".into(), sealed.to_vec());
    let cfg = crate::cloud_config::SealedWeights {
        sealed_sha256: sha256_hex(sealed),
        digest_hex: "44".repeat(32),
        source: None,
    };
    (files, cfg)
}

/// SECURITY (altered sealed file): other bytes, the file twice over or half
/// of it give 77 and leave no file, partial or whole, under the store. A
/// source the TD may not use is 96 before the store is touched.
#[tokio::test]
async fn an_altered_sealed_file_leaves_nothing_on_disk_at_any_failure_point() {
    use crate::cloud_config::SealedSource;
    let sealed = crate::seal::seal(&[0x5a; 4096], &[0x5a; 32], &[7; 12]).unwrap();
    let (files, cfg) = sealed_case(&sealed);
    for (name, mode, says) in [
        ("other bytes", Mode::Corrupt, "does not match"),
        ("twice over", Mode::Oversize, ""),
        ("half, honestly framed", Mode::Short, ""),
    ] {
        let fx = fixture(files.clone(), mode).await;
        let root = tempfile::tempdir().unwrap();
        let src = SealedSource {
            url: format!("{}/o/n/resolve/{REV}/model.safetensors", fx.source.endpoint),
            size: sealed.len() as u64,
        };
        let (code, why) = crate::model_key::fetch_sealed(root.path(), &src, &cfg, "owner:1")
            .await
            .unwrap_err();
        assert_eq!(code, 77, "{name}: {why}");
        assert!(why.contains(says), "{name}: {why}");
        assert_eq!(files_under(root.path()), Vec::<String>::new(), "{name}");
    }
    let root = tempfile::tempdir().unwrap();
    for url in [
        "ftp://w.example/model.sealed",
        "http://w.example/model.sealed",
    ] {
        let src = SealedSource {
            url: url.into(),
            size: sealed.len() as u64,
        };
        let (code, why) = crate::model_key::fetch_sealed(root.path(), &src, &cfg, "owner:1")
            .await
            .unwrap_err();
        assert_eq!(code, 96, "{url}: {why}");
    }
    assert!(
        std::fs::read_dir(root.path()).unwrap().next().is_none(),
        "a refused source creates nothing"
    );
}

/// SECURITY (altered sealed file, ranged): one range carries other bytes.
/// The hash of the whole refuses it and neither the partial nor its ranges
/// list is kept.
#[tokio::test]
async fn an_altered_sealed_file_fetched_in_ranges_leaves_nothing_on_disk() {
    let sealed = crate::seal::seal(&vec![0x5a; 300_000], &[0x5a; 32], &[7; 12]).unwrap();
    let (files, cfg) = sealed_case(&sealed);
    let fx = fixture_with(
        files,
        Mode::Corrupt,
        Opts {
            transfer: Some(ranged(2, 1, 1)),
            ..Opts::default()
        },
    )
    .await;
    let url = format!("{}/o/n/resolve/{REV}/model.safetensors", fx.source.endpoint);
    let spec = FileSpec {
        path: crate::model_key::SEALED_FILE.into(),
        size: sealed.len() as u64,
        sha256: cfg.sealed_sha256.clone(),
    };
    let err = fetch_file(&fx.store, &fx.source, "owner:1", &url, &spec, &mut Quiet)
        .await
        .unwrap_err();
    assert!(matches!(err, PullError::Integrity(_)), "{err}");
    assert!(
        fx.hub.seen.lock().unwrap().ranges.len() > 1,
        "fetched in ranges"
    );
    assert_eq!(files_under(&fx.store.root), Vec::<String>::new());
}

/// The config pins a file that is not an X0NESEAL envelope (the owner
/// hosted the plaintext): the TD refuses it as sealed weights (77) and
/// deletes it, though its hash is the one pinned: no plaintext stays on disk.
#[tokio::test]
async fn a_pinned_file_that_is_not_an_envelope_is_refused() {
    let plain = vec![0x11u8; 4096];
    let (files, cfg) = sealed_case(&plain);
    let fx = fixture(files, Mode::Honest).await;
    let root = tempfile::tempdir().unwrap();
    let src = crate::cloud_config::SealedSource {
        url: format!("{}/o/n/resolve/{REV}/model.safetensors", fx.source.endpoint),
        size: plain.len() as u64,
    };
    let (code, why) = crate::model_key::fetch_sealed(root.path(), &src, &cfg, "owner:1")
        .await
        .unwrap_err();
    assert_eq!(code, 77, "{why}");
    assert!(why.contains("not an X0NESEAL envelope"), "{why}");
    assert_eq!(files_under(root.path()), Vec::<String>::new());
}

/// SECURITY (short sealed file): a chunked body that ends cleanly at half the
/// size gives 77 and leaves no partial under the store.
#[tokio::test]
async fn a_sealed_file_ended_early_by_a_clean_chunked_body_leaves_nothing() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let sealed = crate::seal::seal(&[0x5a; 4096], &[0x5a; 32], &[7; 12]).unwrap();
    let (_files, cfg) = sealed_case(&sealed);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let half = sealed[..sealed.len() / 2].to_vec();
    tokio::spawn(async move {
        while let Ok((mut s, _)) = listener.accept().await {
            let mut buf = [0u8; 4096];
            let _ = s.read(&mut buf).await;
            let head = "HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n";
            let _ = s.write_all(head.as_bytes()).await;
            let _ = s
                .write_all(format!("{:x}\r\n", half.len()).as_bytes())
                .await;
            let _ = s.write_all(&half).await;
            let _ = s.write_all(b"\r\n0\r\n\r\n").await;
            let _ = s.flush().await;
        }
    });
    let root = tempfile::tempdir().unwrap();
    let src = crate::cloud_config::SealedSource {
        url: format!("http://{addr}/model.sealed"),
        size: sealed.len() as u64,
    };
    let (code, why) = crate::model_key::fetch_sealed(root.path(), &src, &cfg, "owner:1")
        .await
        .unwrap_err();
    assert_eq!(code, 77, "{why}");
    assert!(why.contains("ends at"), "{why}");
    assert_eq!(files_under(root.path()), Vec::<String>::new());
}

/// Plaintext and key that no byte of the sealed file can contain by chance.
fn private_case() -> (Vec<u8>, [u8; 32], Vec<u8>) {
    let plain: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8 | 0x80).collect();
    let key: [u8; 32] = std::array::from_fn(|i| 0xA5 ^ i as u8);
    let sealed = crate::seal::seal(&plain, &key, &[9; 12]).unwrap();
    (plain, key, sealed)
}

/// What a kept partial may hold: the source's bytes at their offsets (zero in
/// a hole of a ranged one), never the key nor a window of the plaintext.
fn assert_only_sealed_bytes(dir: &std::path::Path, sealed: &[u8], key: &[u8], plain: &[u8]) {
    let needles = [
        ("the key", key.to_vec()),
        ("the plaintext's start", plain[..64].to_vec()),
        ("the plaintext's middle", plain[150_000..150_064].to_vec()),
    ];
    for f in files_under(dir) {
        let bytes = std::fs::read(dir.join(&f)).unwrap();
        for (what, n) in &needles {
            assert!(
                !bytes.windows(n.len()).any(|w| w == &n[..]),
                "{what} in {f}"
            );
        }
        if std::path::Path::new(&f)
            .extension()
            .is_some_and(|e| e == "part")
        {
            assert!(bytes.len() <= sealed.len(), "{f}");
            let written = bytes.iter().zip(sealed).filter(|(b, _)| **b != 0).count();
            assert!(written > 0, "{f}: nothing written");
            for (i, b) in bytes.iter().enumerate() {
                assert!(
                    *b == 0 || *b == sealed[i],
                    "{f}: byte {i} is not the source's"
                );
            }
        } else {
            assert!(
                f.ends_with(".ranges"),
                "{f}: neither a partial nor its ranges"
            );
            let text = String::from_utf8(bytes).unwrap();
            assert!(text
                .lines()
                .skip(1)
                .all(|l| l.split(' ').all(|n| n.parse::<u64>().is_ok())));
        }
    }
}

/// SECURITY (what a resume keeps): a sealed file cut mid-stream (96) or
/// refused on one range (503) keeps its partial for the next attempt. It
/// holds the source's sealed bytes only, never the key nor plaintext; the
/// next attempt completes it and only the verified sealed file stays.
#[tokio::test]
async fn a_cut_or_refused_transfer_keeps_only_sealed_bytes_for_the_resume() {
    use crate::cloud_config::SealedSource;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let (plain, key, sealed) = private_case();
    let (files, cfg) = sealed_case(&sealed);
    // Half the body under the whole Content-Length, then the connection closes.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (half, total) = (sealed[..sealed.len() / 2].to_vec(), sealed.len());
    tokio::spawn(async move {
        while let Ok((mut s, _)) = listener.accept().await {
            let mut buf = [0u8; 4096];
            let _ = s.read(&mut buf).await;
            let head = format!("HTTP/1.1 200 OK\r\ncontent-length: {total}\r\n\r\n");
            let _ = s.write_all(head.as_bytes()).await;
            let _ = s.write_all(&half).await;
            let _ = s.flush().await;
        }
    });
    let root = tempfile::tempdir().unwrap();
    let cut = SealedSource {
        url: format!("http://{addr}/model.sealed"),
        size: total as u64,
    };
    let (code, why) = crate::model_key::fetch_sealed(root.path(), &cut, &cfg, "owner:1")
        .await
        .unwrap_err();
    assert_eq!(code, 96, "{why}");
    let part = format!("sealed/partial/{}.part", cfg.sealed_sha256);
    assert_eq!(files_under(root.path()), std::slice::from_ref(&part));
    assert_only_sealed_bytes(root.path(), &sealed, &key, &plain);
    let fx = fixture(files.clone(), Mode::Honest).await;
    let honest = SealedSource {
        url: format!("{}/o/n/resolve/{REV}/model.safetensors", fx.source.endpoint),
        size: total as u64,
    };
    let got = crate::model_key::fetch_sealed(root.path(), &honest, &cfg, "owner:1")
        .await
        .unwrap();
    assert_eq!(got, sealed);
    assert_eq!(files_under(root.path()), [part]);
    assert_only_sealed_bytes(root.path(), &sealed, &key, &plain);

    // In ranges: the third range is refused twice (503), the others written.
    let opts = Opts {
        transfer: Some(ranged(2, 1, 1)),
        ..Opts::default()
    };
    let fx = fixture_with(files.clone(), Mode::Unavailable(200_000), opts).await;
    let url = format!("{}/o/n/resolve/{REV}/model.safetensors", fx.source.endpoint);
    let spec = FileSpec {
        path: crate::model_key::SEALED_FILE.into(),
        size: total as u64,
        sha256: cfg.sealed_sha256.clone(),
    };
    let err = fetch_file(&fx.store, &fx.source, "owner:1", &url, &spec, &mut Quiet)
        .await
        .unwrap_err();
    assert!(matches!(err, PullError::Network(_)), "{err}");
    let (p, r) = (
        format!("partial/{}.part", spec.sha256),
        format!("partial/{}.ranges", spec.sha256),
    );
    assert_eq!(files_under(&fx.store.root), [p.clone(), r]);
    assert_only_sealed_bytes(&fx.store.root, &sealed, &key, &plain);
    let opts = Opts {
        transfer: Some(ranged(2, 1, 1)),
        ..Opts::default()
    };
    let (_hub, source) = hub_with(files, Mode::Honest, &opts).await;
    let url = format!("{}/o/n/resolve/{REV}/model.safetensors", source.endpoint);
    fetch_file(&fx.store, &source, "owner:1", &url, &spec, &mut Quiet)
        .await
        .unwrap();
    assert_eq!(files_under(&fx.store.root), std::slice::from_ref(&p));
    assert_eq!(std::fs::read(fx.store.root.join(&p)).unwrap(), sealed);
}

/// SECURITY (sealed source): https to a literal loopback, link-local,
/// private or unspecified address, in any spelling the URL parser accepts,
/// or to a metadata or loopback name, is refused; the TD makes no request
/// and creates nothing. Names are not resolved: a public name passes.
#[tokio::test]
async fn a_source_on_a_non_public_address_or_a_metadata_name_is_refused() {
    for (url, why) in [
        ("https://169.254.169.254/computeMetadata/v1/", "link-local"),
        ("https://2852039166/m", "link-local"),
        ("https://0xa9.0xfe.0xa9.0xfe/m", "link-local"),
        ("https://[fe80::1]/m", "link-local"),
        ("https://[::ffff:169.254.169.254]/m", "link-local"),
        (
            "https://metadata.google.internal/computeMetadata/v1/",
            "metadata",
        ),
        ("https://METADATA.Google.Internal./m", "metadata"),
        ("https://metadata/m", "metadata"),
        ("https://instance-data/latest/meta-data/", "metadata"),
        ("https://172.16.0.1/m", "RFC 1918"),
        ("https://172.31.255.255/m", "RFC 1918"),
        ("https://[::ffff:172.16.0.1]/m", "RFC 1918"),
        ("https://[fd00::1]/m", "fc00::/7"),
        ("https://[fc00::1]/m", "fc00::/7"),
        ("https://127.0.0.1/m", "loopback"),
        ("https://127.1.2.3:8443/m", "loopback"),
        ("https://[::1]/m", "loopback"),
        ("https://localhost/m", "loopback"),
        ("https://a.localhost/m", "loopback"),
        ("https://0.0.0.0/m", "unspecified"),
        ("https://0.1.2.3/m", "unspecified"),
        ("https://[::]/m", "unspecified"),
    ] {
        let err = check_source_url(url).expect_err(url);
        assert!(err.contains(why), "{url}: {err}");
        assert!(err.contains("not resolved"), "{url}: {err}");
    }
    for url in [
        "https://w.example/m/model.sealed",
        "https://storage.googleapis.com/b/model.sealed?X-Goog-Signature=ab",
        "https://8.8.8.8/m",
        "https://172.32.0.1/m",
        "https://11.0.0.1/m",
        "https://169.255.0.1/m",
        "https://[2001:4860::1]/m",
        "https://metadata.example/m",
        "http://127.0.0.1:1/model.sealed",
        "http://localhost:1/model.sealed",
        "http://[::1]:1/model.sealed",
    ] {
        check_source_url(url).unwrap_or_else(|e| panic!("{url}: {e}"));
    }
    let root = tempfile::tempdir().unwrap();
    let (_, cfg) = sealed_case(b"x");
    for url in [
        "https://169.254.169.254/model.sealed",
        "https://metadata.google.internal/model.sealed",
        "https://172.20.0.9/model.sealed",
    ] {
        let src = crate::cloud_config::SealedSource {
            url: url.into(),
            size: 5,
        };
        let (code, why) = crate::model_key::fetch_sealed(root.path(), &src, &cfg, "owner:1")
            .await
            .unwrap_err();
        assert_eq!(code, 96, "{url}: {why}");
        assert!(why.contains("refused"), "{url}: {why}");
    }
    assert_eq!(files_under(root.path()), Vec::<String>::new());
}

/// SECURITY (credentials in a source URL): userinfo may be a password. No
/// message names it: the judgement of the URL, the source, a refused or
/// failed fetch, parsed or not.
#[tokio::test]
async fn no_source_message_carries_userinfo() {
    let bad = [
        "https://user:secret@w.example/model.sealed",
        "https://:secret@w.example/model.sealed",
        "https://secret@w.example/model.sealed",
        "http://127.0.0.1:secret@w.example/model.sealed",
        "https://user:secret@w.exa mple/model.sealed",
        "https://user:secret@/model.sealed",
        "ftp://user:secret@w.example/model.sealed",
    ];
    for url in bad {
        let err = check_source_url(url).expect_err(url);
        assert!(!err.contains("secret"), "{url}: {err}");
        if let Err(e) = Source::new(url) {
            assert!(!e.contains("secret"), "{url}: {e}");
        }
    }
    let root = tempfile::tempdir().unwrap();
    let (_, cfg) = sealed_case(b"x");
    for url in bad {
        let src = crate::cloud_config::SealedSource {
            url: url.into(),
            size: 5,
        };
        let (code, why) = crate::model_key::fetch_sealed(root.path(), &src, &cfg, "owner:1")
            .await
            .unwrap_err();
        assert_eq!(code, 96, "{url}: {why}");
        assert!(!why.contains("secret"), "{url}: {why}");
    }
    // The transfer's own messages, given such a URL directly: 404, cut.
    let fx = fixture(model_files(), Mode::Honest).await;
    let at = fx.source.endpoint.trim_start_matches("http://").to_string();
    let spec = FileSpec {
        path: "model.sealed".into(),
        size: 5,
        sha256: "00".repeat(32),
    };
    let url = format!("http://u:secret@{at}/o/n/resolve/{REV}/absent");
    let err = fetch_file(&fx.store, &fx.source, "owner:1", &url, &spec, &mut Quiet)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("404"), "{err}");
    assert!(!err.to_string().contains("secret"), "{err}");
    let url = "http://u:secret@127.0.0.1:9/model.sealed";
    let err = fetch_file(&fx.store, &fx.source, "owner:1", url, &spec, &mut Quiet)
        .await
        .unwrap_err();
    assert!(!err.to_string().contains("secret"), "{err}");
    for url in bad {
        assert!(!shown_url(url).contains("secret"), "{url}");
    }
    assert_eq!(
        shown_url("https://user:secret@w.example/m?sig=1"),
        "https://w.example/m?sig=1"
    );
    assert_eq!(
        shown_url("https://w.example/m?sig=1"),
        "https://w.example/m?sig=1"
    );
}

/// SECURITY (SSRF): the source itself is judged by `check_source_url`; a
/// source that answers 302 must not lead the TD where that check would not
/// let it go. Linux routes a connect to 0.0.0.0 to the loopback, so following
/// this one reaches a service on the TD's own loopback. A failure is a
/// connection on that service, or a refusal that does not say why.
#[tokio::test]
async fn a_redirect_to_a_non_public_address_is_refused_before_any_connection() {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    let service = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = service.local_addr().unwrap().port();
    let reached = Arc::new(AtomicUsize::new(0));
    let count = reached.clone();
    tokio::spawn(async move {
        while service.accept().await.is_ok() {
            count.fetch_add(1, SeqCst);
        }
    });
    let to = format!("https://0.0.0.0:{port}/latest/meta-data?sig=secret");
    let app = axum::Router::new().fallback(move || {
        let to = to.clone();
        async move { (StatusCode::FOUND, [("location", to)]).into_response() }
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await });
    let dir = tempfile::tempdir().unwrap();
    let store = Store::new(dir.path().join("cache"));
    let source = Source::new(&endpoint).unwrap();
    let spec = FileSpec {
        path: "model.sealed".into(),
        size: 5,
        sha256: "00".repeat(32),
    };
    let url = format!("{endpoint}/model.sealed");
    let err = fetch_file(&store, &source, "owner:1", &url, &spec, &mut Quiet)
        .await
        .unwrap_err()
        .to_string();
    assert_eq!(reached.load(SeqCst), 0, "the local service was reached");
    assert!(err.contains("refused"), "{err}");
    assert!(err.contains("unspecified"), "{err}");
    assert!(!err.contains("secret"), "{err}");
}

/// SECURITY (SSRF): where a redirect may lead, by what the endpoint is. From
/// a public endpoint (the Hub, a TD's sealed source): https to a host
/// `check_source_url` would pass. From the loopback (tests, a bench): the
/// loopback too. From a non-public https endpoint the operator named (a
/// mirror, which a TD's sealed source cannot be): https anywhere. A failure
/// is a redirect that reaches a non-public host from a public source.
#[test]
fn a_redirect_is_judged_as_the_source_itself() {
    let url = |s: &str| reqwest::Url::parse(s).unwrap();
    assert_eq!(Reach::of(DEFAULT_ENDPOINT), Reach::Public);
    assert_eq!(Reach::of("https://w.example/m.sealed"), Reach::Public);
    assert_eq!(Reach::of("https://8.8.8.8/m"), Reach::Public);
    assert_eq!(Reach::of("http://127.0.0.1:9"), Reach::Loopback);
    assert_eq!(Reach::of("https://172.16.0.5"), Reach::Private);
    assert_eq!(Reach::of("https://mirror.localhost"), Reach::Private);

    let public = [
        "https://cdn-lfs.huggingface.co/x?sig=1",
        "https://8.8.8.8/m",
        "https://172.32.0.1/m",
        "https://[2001:4860::1]/m",
    ];
    for reach in [Reach::Public, Reach::Loopback, Reach::Private] {
        for to in public {
            assert_eq!(redirect_refusal(reach, &url(to)), None, "{reach:?} {to}");
        }
    }
    for (to, why) in [
        ("https://169.254.169.254/computeMetadata/v1/", "link-local"),
        ("https://2852039166/m", "link-local"),
        ("https://[::ffff:169.254.169.254]/m", "link-local"),
        ("https://[fe80::1]/m", "link-local"),
        ("https://metadata.google.internal/m", "metadata"),
        ("https://METADATA.Google.Internal./m", "metadata"),
        ("https://instance-data/latest/meta-data/", "metadata"),
        ("https://172.16.0.1/m", "RFC 1918"),
        ("https://172.31.1.1/m", "RFC 1918"),
        ("https://[fd00::1]/m", "fc00::/7"),
        ("https://0.0.0.0:443/m", "unspecified"),
        ("https://[::]/m", "unspecified"),
    ] {
        for reach in [Reach::Public, Reach::Loopback] {
            let e = redirect_refusal(reach, &url(to)).unwrap_or_else(|| panic!("{reach:?} {to}"));
            assert!(e.contains("refused") && e.contains(why), "{to}: {e}");
        }
        assert_eq!(redirect_refusal(Reach::Private, &url(to)), None, "{to}");
    }
    for to in [
        "https://127.0.0.1/m",
        "https://[::1]:8443/m",
        "https://localhost/m",
    ] {
        let e = redirect_refusal(Reach::Public, &url(to)).expect(to);
        assert!(e.contains("loopback"), "{to}: {e}");
        assert_eq!(redirect_refusal(Reach::Loopback, &url(to)), None, "{to}");
        assert_eq!(redirect_refusal(Reach::Private, &url(to)), None, "{to}");
    }
    // Plain http from a loopback endpoint to the loopback, and nothing else.
    for to in [
        "http://127.0.0.1:1/m",
        "http://localhost:1/m",
        "http://[::1]:1/m",
    ] {
        assert_eq!(redirect_refusal(Reach::Loopback, &url(to)), None, "{to}");
        for reach in [Reach::Public, Reach::Private] {
            assert!(
                redirect_refusal(reach, &url(to)).is_some(),
                "{reach:?} {to}"
            );
        }
    }
    for reach in [Reach::Public, Reach::Loopback, Reach::Private] {
        for to in [
            "http://w.example/m",
            "http://172.16.0.1/m",
            "ftp://w.example/m",
        ] {
            let e = redirect_refusal(reach, &url(to)).unwrap_or_else(|| panic!("{reach:?} {to}"));
            assert!(e.contains("non-https"), "{to}: {e}");
        }
    }
    // A refusal names the host, never the target's path, query or userinfo:
    // a CDN's URL may carry a signature.
    for to in [
        "https://172.16.0.1/p?sig=secret",
        "http://u:secret@w.example/secret?secret",
    ] {
        let e = redirect_refusal(Reach::Public, &url(to)).expect(to);
        assert!(!e.contains("secret"), "{to}: {e}");
    }
}
