//! `tlog.rs` and `transparency.rs` against Go's implementations of the same
//! C2SP formats: a log of 600 entries, checkpoints at 301 and 600 signed by a
//! log key and cosigned by a witness, the tiles of both sizes, an inclusion
//! proof and a consistency proof, all made by `tests/tlog/go-src/main.go`
//! with golang.org/x/mod v0.36.0 (`sumdb/note`, `sumdb/tlog`) and
//! transparency-dev/formats v0.1.1 (`note`, cosignature/v1). The same
//! program verifies a log made by `transparency::test_log` (run by hand, not
//! here: no Go in `make ci`).
//!
//! And a log written by Tessera v1.0.4, the tile-based log library Rekor v2
//! is built on (POSIX storage driver), holding entries in this crate's
//! format: 300 references, checkpoints at 100 and 300, made by
//! `tests/tlog/tessera-src/main.go`.

use std::path::{Path, PathBuf};

use x0ne_infer::tlog::{self, NoteKey, TileSource};
use x0ne_infer::transparency::{self, DirTiles, Policy, Tiles};

const ORIGIN: &str = "interop.invalid/go-log";
const TESSERA_ORIGIN: &str = "interop.invalid/tessera-log";

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/tlog/go")
}

fn read(name: &str) -> String {
    std::fs::read_to_string(dir().join(name)).unwrap()
}

fn tessera() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/tlog/tessera")
}

fn tessera_read(name: &str) -> String {
    std::fs::read_to_string(tessera().join(name)).unwrap()
}

fn keys() -> (NoteKey, NoteKey) {
    keys_in(&dir())
}

fn keys_in(dir: &Path) -> (NoteKey, NoteKey) {
    let vkeys = std::fs::read_to_string(dir.join("vkeys")).unwrap();
    let key = |role: &str| {
        let line = vkeys
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{role} ")))
            .unwrap();
        NoteKey::parse(line).unwrap()
    };
    (key("log"), key("witness"))
}

fn policy() -> Policy {
    policy_of(ORIGIN, &dir())
}

fn policy_of(origin: &str, dir: &Path) -> Policy {
    let (log, witness) = keys_in(dir);
    Policy {
        origin: origin.into(),
        log,
        witnesses: vec![witness],
        quorum: 1,
        // The cosignatures carry the time they were made (2026-10-22).
        max_age_s: 100 * 365 * 24 * 3600,
        tiles: Tiles::BesideReference,
    }
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn hashes(text: &str) -> Vec<tlog::Hash> {
    text.lines()
        .map(|l| tlog::b64_canonical(l).unwrap().try_into().unwrap())
        .collect()
}

#[test]
fn go_vkeys_parse_with_their_key_ids() {
    let (log, witness) = keys();
    assert_eq!(log.kind, tlog::KeyKind::Ed25519);
    assert_eq!(witness.kind, tlog::KeyKind::CosignatureV1);
    assert_eq!(log.vkey(), read("vkeys").lines().next().unwrap()[4..]);
}

/// Both Go checkpoints open: the log's Ed25519 note signature and the
/// witness's cosignature/v1, with the sizes and roots Go wrote.
#[test]
fn go_checkpoints_open_with_go_keys() {
    let (log, witness) = keys();
    for (name, size) in [("checkpoint-301", 301), ("tlog/checkpoint", 600)] {
        let note = tlog::parse_note(&read(name)).unwrap();
        let verified = tlog::open_note(&note, &[log.clone(), witness.clone()]).unwrap();
        assert_eq!(verified.len(), 2, "{name}");
        assert!(verified
            .iter()
            .any(|v| v.key == witness && v.time.is_some()));
        let cp = tlog::parse_checkpoint(&note.text).unwrap();
        assert_eq!((cp.origin.as_str(), cp.size), (ORIGIN, size));
    }
}

/// Go's tiles give Go's roots, Go's inclusion proof and Go's consistency
/// proof, hash for hash.
#[test]
fn go_tiles_give_go_roots_and_go_proofs() {
    let src = DirTiles(dir().join("tlog"));
    let root = |name: &str| {
        let note = tlog::parse_note(&read(name)).unwrap();
        tlog::parse_checkpoint(&note.text).unwrap().root
    };
    let (r301, r600) = (root("checkpoint-301"), root("tlog/checkpoint"));
    let mut t = tlog::TileHashes::new(&src, 600);
    let consistency = tlog::consistency_proof(&mut t, 301, 600).unwrap();
    assert_eq!(consistency, hashes(&read("consistency-301-600")));
    tlog::verify_consistency(301, 600, &consistency, &r301, &r600).unwrap();
    let proof = tlog::parse_tlog_proof(&read("reference-300.json.tlog-proof")).unwrap();
    let mut t = tlog::TileHashes::new(&src, 301);
    assert_eq!(
        tlog::inclusion_proof(&mut t, 300, 301).unwrap(),
        proof.hashes
    );
    assert_eq!(src.checkpoint().unwrap(), read("tlog/checkpoint"));
}

/// The relay's check on a reference Go logged: its proof at 301, then with
/// this machine having seen 600, consistency read from Go's tiles.
#[test]
fn the_relay_check_passes_a_reference_go_logged() {
    let reference = dir().join("reference-300.json");
    let bytes = std::fs::read(&reference).unwrap();
    let proof = read("reference-300.json.tlog-proof");
    let src = DirTiles(dir().join("tlog"));
    let state = tempfile::tempdir().unwrap();
    let l = transparency::check(&bytes, &proof, &policy(), &src, state.path(), now()).unwrap();
    assert_eq!((l.index, l.size, l.refreshed), (300, 301, false), "{l:?}");
    let state = tempfile::tempdir().unwrap();
    let file = transparency::state_file(state.path(), ORIGIN);
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, read("tlog/checkpoint")).unwrap();
    transparency::check(&bytes, &proof, &policy(), &src, state.path(), now()).unwrap();
}

/// SECURITY: one bit of one Go tile changed, the larger head this machine
/// saw is no longer shown consistent: refused (a fork, or tiles that do not
/// match the signed heads). Another reference with Go's proof: refused.
#[test]
fn a_changed_go_tile_or_another_reference_is_refused() {
    let copy = tempfile::tempdir().unwrap();
    for f in [
        "tile/0/000",
        "tile/0/001",
        "tile/0/002.p/88",
        "tile/1/000.p/2",
    ] {
        let to = copy.path().join(f);
        std::fs::create_dir_all(to.parent().unwrap()).unwrap();
        std::fs::copy(dir().join("tlog").join(f), &to).unwrap();
    }
    let mut tile = std::fs::read(copy.path().join("tile/0/001")).unwrap();
    tile[32 * 200] ^= 1;
    std::fs::write(copy.path().join("tile/0/001"), tile).unwrap();
    let bytes = std::fs::read(dir().join("reference-300.json")).unwrap();
    let proof = read("reference-300.json.tlog-proof");
    let state = tempfile::tempdir().unwrap();
    let file = transparency::state_file(state.path(), ORIGIN);
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, read("tlog/checkpoint")).unwrap();
    let err = transparency::check(
        &bytes,
        &proof,
        &policy(),
        &DirTiles(copy.path().to_path_buf()),
        state.path(),
        now(),
    )
    .unwrap_err();
    assert!(
        err.contains("its tiles do not match its signed checkpoints"),
        "{err}"
    );
    let state = tempfile::tempdir().unwrap();
    let err = transparency::check(
        b"{\"interop\": \"another\"}\n",
        &proof,
        &policy(),
        &DirTiles(dir().join("tlog")),
        state.path(),
        now(),
    )
    .unwrap_err();
    assert!(err.contains("not shown in the log"), "{err}");
}

/// Tessera's checkpoints open with its log key and the witness's
/// cosignature/v1; its tiles give its roots, its inclusion proof and its
/// consistency proof, hash for hash.
#[test]
fn tessera_checkpoints_tiles_and_proofs_match() {
    let (log, witness) = keys_in(&tessera());
    let head = |name: &str| {
        let note = tlog::parse_note(&tessera_read(name)).unwrap();
        let verified = tlog::open_note(&note, &[log.clone(), witness.clone()]).unwrap();
        assert_eq!(verified.len(), 2, "{name}");
        let cp = tlog::parse_checkpoint(&note.text).unwrap();
        assert_eq!(cp.origin, TESSERA_ORIGIN);
        (cp.size, cp.root)
    };
    let ((s100, r100), (s300, r300)) = (head("checkpoint-100"), head("log/checkpoint"));
    assert_eq!((s100, s300), (100, 300));
    let src = DirTiles(tessera().join("log"));
    let mut t = tlog::TileHashes::new(&src, 300);
    let consistency = tlog::consistency_proof(&mut t, 100, 300).unwrap();
    assert_eq!(consistency, hashes(&tessera_read("consistency-100-300")));
    tlog::verify_consistency(100, 300, &consistency, &r100, &r300).unwrap();
    let proof = tlog::parse_tlog_proof(&tessera_read("r-042.json.tlog-proof")).unwrap();
    let mut t = tlog::TileHashes::new(&src, 100);
    assert_eq!(
        tlog::inclusion_proof(&mut t, 42, 100).unwrap(),
        proof.hashes
    );
}

/// The relay's check on a reference Tessera logged: Tessera hashed the same
/// leaf for the same entry. Then with this machine having seen 300,
/// consistency read from Tessera's tiles.
#[test]
fn the_relay_check_passes_a_reference_tessera_logged() {
    let bytes = std::fs::read(tessera().join("r-042.json")).unwrap();
    let proof = tessera_read("r-042.json.tlog-proof");
    let policy = policy_of(TESSERA_ORIGIN, &tessera());
    let src = DirTiles(tessera().join("log"));
    let state = tempfile::tempdir().unwrap();
    let l = transparency::check(&bytes, &proof, &policy, &src, state.path(), now()).unwrap();
    assert_eq!((l.index, l.size, l.refreshed), (42, 100, false), "{l:?}");
    let state = tempfile::tempdir().unwrap();
    let file = transparency::state_file(state.path(), TESSERA_ORIGIN);
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, tessera_read("log/checkpoint")).unwrap();
    transparency::check(&bytes, &proof, &policy, &src, state.path(), now()).unwrap();
}

/// SECURITY: one bit of Tessera's last tile changed, the head of 300 this
/// machine saw is no longer shown consistent: refused (a fork, or tiles that
/// do not match the signed heads). Entry 43, logged, with the proof of 42:
/// refused.
#[test]
fn a_changed_tessera_tile_or_another_reference_is_refused() {
    let copy = tempfile::tempdir().unwrap();
    for f in [
        "tile/0/000",
        "tile/0/000.p/100",
        "tile/0/001.p/44",
        "tile/1/000.p/1",
    ] {
        let to = copy.path().join(f);
        std::fs::create_dir_all(to.parent().unwrap()).unwrap();
        std::fs::copy(tessera().join("log").join(f), &to).unwrap();
    }
    let mut tile = std::fs::read(copy.path().join("tile/0/001.p/44")).unwrap();
    tile[32 * 20] ^= 1;
    std::fs::write(copy.path().join("tile/0/001.p/44"), tile).unwrap();
    let bytes = std::fs::read(tessera().join("r-042.json")).unwrap();
    let proof = tessera_read("r-042.json.tlog-proof");
    let policy = policy_of(TESSERA_ORIGIN, &tessera());
    let state = tempfile::tempdir().unwrap();
    let file = transparency::state_file(state.path(), TESSERA_ORIGIN);
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, tessera_read("log/checkpoint")).unwrap();
    let src = DirTiles(copy.path().to_path_buf());
    let err = transparency::check(&bytes, &proof, &policy, &src, state.path(), now()).unwrap_err();
    assert!(
        err.contains("its tiles do not match its signed checkpoints"),
        "{err}"
    );
    let state = tempfile::tempdir().unwrap();
    let src = DirTiles(tessera().join("log"));
    let err = transparency::check(
        b"{\"tessera\": 43}\n",
        &proof,
        &policy,
        &src,
        state.path(),
        now(),
    )
    .unwrap_err();
    assert!(err.contains("not shown in the log"), "{err}");
}

/// The other direction, run by hand: a TEST log of 300 references made by
/// `transparency::test_log`, in `<CARGO_TARGET_TMPDIR>/rust-log`, for
/// `go run . verify <dir>` (`tests/tlog/go-src`), which checks every proof,
/// checkpoint and cosignature, and every tile, with the Go libraries.
/// `cargo test -p x0ne-infer --test tlog_interop -- --ignored`.
#[test]
#[ignore = "writes a log for the Go verifier; run by hand"]
fn write_a_test_log_for_the_go_verifier() {
    use transparency::test_log;
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("rust-log");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();
    for i in 0..300u64 {
        let p = out.join(format!("r-{i:03}.json"));
        std::fs::write(&p, format!("{{\"rust\": {i}}}\n")).unwrap();
        assert_eq!(test_log::log_reference(&p, 1_792_627_200 + i).unwrap(), i);
    }
    std::fs::write(
        out.join("vkeys"),
        format!(
            "log {}\nwitness {}\n",
            test_log::log_key().vkey(),
            test_log::witness_key().vkey()
        ),
    )
    .unwrap();
    println!("{}", out.display());
}
