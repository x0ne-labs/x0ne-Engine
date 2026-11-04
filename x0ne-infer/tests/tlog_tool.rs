//! `x0ne-engine-tlog`, the tool that runs the log of image references, end
//! to end on keys it makes here (never a real key): a log key and a witness
//! key, two references appended, cosigned through the witness protocol's
//! request, proven, and accepted by the check the relay runs
//! (`transparency::check`) against a pin of that log. Each refusal leaves
//! the log as it was.

use sha2::{Digest, Sha256};
use std::io::Write as _;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use x0ne_infer::tlog::{KeyKind, NoteKey};
use x0ne_infer::transparency::{self, pinned, DirTiles, Pin};

const BIN: &str = env!("CARGO_BIN_EXE_x0ne-engine-tlog");
const ORIGIN: &str = "rehearsal.x0ne.invalid/engine/tlog";
const WITNESS: &str = "rehearsal.x0ne.invalid/witness";

struct Out {
    code: i32,
    stdout: String,
    stderr: String,
}

fn run(args: &[&str], stdin: &str) -> Out {
    let mut child = Command::new(BIN)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the tool runs");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    Out {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn ok(args: &[&str], stdin: &str) -> String {
    let o = run(args, stdin);
    assert_eq!(o.code, 0, "{args:?}: {}{}", o.stdout, o.stderr);
    o.stdout
}

fn refused(args: &[&str], stdin: &str, why: &str) {
    let o = run(args, stdin);
    assert_eq!(
        o.code, 1,
        "{args:?} should refuse: {}{}",
        o.stdout, o.stderr
    );
    assert!(
        o.stderr.contains(why),
        "{args:?}: {:?} lacks {why:?}",
        o.stderr
    );
}

fn p(path: &Path) -> &str {
    path.to_str().unwrap()
}

/// The value after `<label> ` on a line of the tool's output.
fn field(out: &str, label: &str) -> String {
    out.lines()
        .find_map(|l| l.strip_prefix(&format!("{label} ")))
        .unwrap_or_else(|| panic!("no {label} line in {out:?}"))
        .to_string()
}

/// SHA-256 of the key line, 40 hex digits in groups of four: the paper form.
fn fingerprint(vkey: &str) -> String {
    let h: String = Sha256::digest(vkey.as_bytes())
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect();
    h[..40]
        .as_bytes()
        .chunks(4)
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Every file under `dir`, with its bytes: what a refusal must not change.
fn snapshot(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd {
            let path = e.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push((path.clone(), std::fs::read(&path).unwrap()));
            }
        }
    }
    out.sort();
    out
}

struct Keys {
    log_secret: PathBuf,
    log_vkey: String,
    witness_secret: PathBuf,
    witness_vkey: String,
}

fn keys(dir: &Path) -> Keys {
    let log_secret = dir.join("log-REHEARSAL.key");
    let witness_secret = dir.join("witness-REHEARSAL.key");
    let log = ok(&["keygen", "log", ORIGIN, p(&log_secret)], "");
    let witness = ok(&["keygen", "witness", WITNESS, p(&witness_secret)], "");
    Keys {
        log_secret,
        log_vkey: field(&log, "vkey"),
        witness_secret,
        witness_vkey: field(&witness, "vkey"),
    }
}

/// One witness round: the add-checkpoint request from the log's last
/// cosigned size, the witness's cosignature, added to the checkpoint.
fn cosign_round(k: &Keys, log: &Path, state: &Path, old: u64) {
    let request = ok(&["witness-request", p(log), &old.to_string()], "");
    let line = ok(
        &["cosign", p(&k.witness_secret), &k.log_vkey, p(state)],
        &request,
    );
    ok(&["add-cosignatures", p(log), &k.witness_vkey], &line);
}

fn pin_text(k: &Keys) -> String {
    format!(
        "# the rehearsal log\norigin {ORIGIN}\nlog {}\nwitness {}\nquorum 1\nmax-age 86400\n\
         tiles https://rehearsal.x0ne.invalid/engine/tlog/\n",
        k.log_vkey, k.witness_vkey
    )
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

/// The check the relay runs, with the log's files read from `log`.
fn relay_check(k: &Keys, log: &Path, reference: &Path, state: &Path) -> Result<u64, String> {
    let pin = Pin {
        origin: ORIGIN,
        log_key: leak(&k.log_vkey),
        witnesses: Box::leak(vec![leak(&k.witness_vkey)].into_boxed_slice()),
        quorum: 1,
        max_age_s: 86400,
        tiles: "https://rehearsal.x0ne.invalid/engine/tlog/",
    };
    let policy = pinned(&pin)?;
    let bytes = std::fs::read(reference).unwrap();
    let proof = std::fs::read_to_string(transparency::proof_path(reference))
        .map_err(|e| format!("no proof: {e}"))?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    transparency::check(&bytes, &proof, &policy, &DirTiles(log.into()), state, now).map(|l| l.index)
}

#[test]
fn keygen_writes_a_private_secret_and_prints_the_key_for_paper() {
    let t = tempfile::tempdir().unwrap();
    let secret = t.path().join("log-REHEARSAL.key");
    let out = ok(&["keygen", "log", ORIGIN, p(&secret)], "");
    let vkey = field(&out, "vkey");
    let key = NoteKey::parse(&vkey).unwrap();
    assert_eq!((key.name.as_str(), key.kind), (ORIGIN, KeyKind::Ed25519));
    assert_eq!(field(&out, "fingerprint"), fingerprint(&vkey));
    let mode = std::fs::metadata(&secret).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o600, "the secret is readable by others");
    let text = std::fs::read_to_string(&secret).unwrap();
    assert!(
        text.starts_with(&format!("PRIVATE+KEY+{ORIGIN}+{:08x}+", key.id)),
        "not the signed-note secret key form: {text:?}"
    );
    assert!(!out.contains(text.trim()), "the secret is printed");
    // Never over an existing file: the first key stays.
    refused(&["keygen", "log", ORIGIN, p(&secret)], "", "already exists");
    assert_eq!(std::fs::read_to_string(&secret).unwrap(), text);
    // A witness key is a cosignature/v1 key; two keys are never the same.
    let w = ok(
        &["keygen", "witness", WITNESS, p(&t.path().join("w.key"))],
        "",
    );
    assert_eq!(
        NoteKey::parse(&field(&w, "vkey")).unwrap().kind,
        KeyKind::CosignatureV1
    );
    let other = ok(
        &["keygen", "log", ORIGIN, p(&t.path().join("other.key"))],
        "",
    );
    assert_ne!(field(&other, "vkey"), vkey);
    assert_eq!(
        ok(&["fingerprint", &vkey], "").trim(),
        fingerprint(&vkey),
        "fingerprint of a key read from elsewhere"
    );
    refused(
        &["keygen", "log", "a name", p(&t.path().join("x.key"))],
        "",
        "name",
    );
    refused(
        &["keygen", "root", ORIGIN, p(&t.path().join("y.key"))],
        "",
        "log or witness",
    );
}

#[test]
fn a_reference_logged_cosigned_and_proven_passes_the_relays_check() {
    let t = tempfile::tempdir().unwrap();
    let k = keys(t.path());
    let log = t.path().join("log");
    let wstate = t.path().join("witness-state");
    let relay_state = t.path().join("relay-state");
    let r1 = t.path().join("x0ne-cvm-0.2.4.json");
    let r2 = t.path().join("x0ne-cvm-0.2.5.json");
    std::fs::write(&r1, b"{\"schema\":\"x0ne-engine/reference/v2\",\"v\":1}\n").unwrap();
    std::fs::write(&r2, b"{\"schema\":\"x0ne-engine/reference/v2\",\"v\":2}\n").unwrap();

    let out = ok(&["append", p(&log), p(&k.log_secret), p(&r1)], "");
    assert!(out.contains("index 0"), "{out}");
    // The log's files are the tlog-tiles layout: one hash, one entry bundle.
    let entry = transparency::entry(&std::fs::read(&r1).unwrap());
    let bundle = std::fs::read(log.join("tile/entries/000.p/1")).unwrap();
    let mut want = (entry.len() as u16).to_be_bytes().to_vec();
    want.extend_from_slice(&entry);
    assert_eq!(bundle, want, "entry bundle");
    assert_eq!(std::fs::read(log.join("tile/0/000.p/1")).unwrap().len(), 32);
    // Signed by the log alone, the proof is not written: no witness yet.
    refused(&["prove", p(&log), p(&r1)], "", "no cosignature");

    cosign_round(&k, &log, &wstate, 0);
    ok(&["prove", p(&log), p(&r1)], "");
    assert_eq!(relay_check(&k, &log, &r1, &relay_state), Ok(0));
    ok(
        &[
            "verify",
            p(&write(t.path(), "log.pin", &pin_text(&k))),
            p(&r1),
            p(&log),
        ],
        "",
    );

    // A second reference: the witness checks the log grew from what it cosigned.
    let out = ok(&["append", p(&log), p(&k.log_secret), p(&r2)], "");
    assert!(out.contains("index 1"), "{out}");
    cosign_round(&k, &log, &wstate, 1);
    ok(&["prove", p(&log), p(&r2)], "");
    assert_eq!(relay_check(&k, &log, &r2, &relay_state), Ok(1));
    // The first proof still holds against the state the relay kept.
    assert_eq!(relay_check(&k, &log, &r1, &relay_state), Ok(0));
}

fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path
}

#[test]
fn the_tool_refuses_and_leaves_the_log_as_it_was() {
    let t = tempfile::tempdir().unwrap();
    let k = keys(t.path());
    let log = t.path().join("log");
    let wstate = t.path().join("witness-state");
    let r1 = write(t.path(), "r1.json", "{\"v\":1}\n");
    let r2 = write(t.path(), "r2.json", "{\"v\":2}\n");
    ok(&["append", p(&log), p(&k.log_secret), p(&r1)], "");
    cosign_round(&k, &log, &wstate, 0);
    let before = snapshot(&log);
    let wbefore = std::fs::read(&wstate).unwrap();

    // A secret others can read is not used.
    std::fs::set_permissions(&k.log_secret, std::fs::Permissions::from_mode(0o644)).unwrap();
    refused(
        &["append", p(&log), p(&k.log_secret), p(&r2)],
        "",
        "readable",
    );
    std::fs::set_permissions(&k.log_secret, std::fs::Permissions::from_mode(0o600)).unwrap();
    // A witness key does not sign the log; another log key does not sign this one.
    refused(
        &["append", p(&log), p(&k.witness_secret), p(&r2)],
        "",
        "not a log key",
    );
    let other = t.path().join("other-REHEARSAL.key");
    ok(
        &["keygen", "log", "rehearsal.x0ne.invalid/other", p(&other)],
        "",
    );
    refused(&["append", p(&log), p(&other), p(&r2)], "", "origin");
    // The same reference twice.
    refused(
        &["append", p(&log), p(&k.log_secret), p(&r1)],
        "",
        "already in the log",
    );
    refused(
        &["append", p(&log), p(&k.log_secret), p(&r2), p(&r2)],
        "",
        "twice",
    );
    // Not in the log: no proof.
    refused(&["prove", p(&log), p(&r2)], "", "not in the log");
    assert_eq!(snapshot(&log), before, "a refusal changed the log");

    // The witness: a request from a size it has not cosigned (409 in tlog-witness).
    let request = ok(&["witness-request", p(&log), "0"], "");
    refused(
        &["cosign", p(&k.witness_secret), &k.log_vkey, p(&wstate)],
        &request,
        "last cosigned size is 1",
    );
    // A checkpoint the pinned log did not sign.
    let fresh = t.path().join("fresh-state");
    let other_vkey = NoteKey::parse(&k.witness_vkey).unwrap();
    let wrong_log = NoteKey::new(ORIGIN, KeyKind::Ed25519, other_vkey.public_key())
        .unwrap()
        .vkey();
    refused(
        &["cosign", p(&k.witness_secret), &wrong_log, p(&fresh)],
        &ok(&["witness-request", p(&log), "0"], ""),
        "log",
    );
    assert!(
        !fresh.exists(),
        "a refused cosignature left a witness state"
    );
    // A consistency proof that does not lead to the checkpoint.
    ok(&["append", p(&log), p(&k.log_secret), p(&r2)], "");
    let before = snapshot(&log);
    let request = ok(&["witness-request", p(&log), "1"], "");
    let tampered = tamper_proof_line(&request);
    refused(
        &["cosign", p(&k.witness_secret), &k.log_vkey, p(&wstate)],
        &tampered,
        "consistency",
    );
    assert_eq!(std::fs::read(&wstate).unwrap(), wbefore);
    // A cosignature line that does not verify under the key it names.
    let line = ok(
        &["cosign", p(&k.witness_secret), &k.log_vkey, p(&wstate)],
        &request,
    );
    let forged = forge(&line);
    refused(
        &["add-cosignatures", p(&log), &k.witness_vkey],
        &forged,
        "does not verify",
    );
    // A line of a witness not named on the command line.
    let w2 = t.path().join("w2-REHEARSAL.key");
    let w2v = field(
        &ok(
            &["keygen", "witness", "rehearsal.x0ne.invalid/w2", p(&w2)],
            "",
        ),
        "vkey",
    );
    refused(&["add-cosignatures", p(&log), &w2v], &line, "not named");
    assert_eq!(snapshot(&log), before, "a refusal changed the log");

    // Entries that do not give the checkpoint's root: the log is not used.
    ok(&["add-cosignatures", p(&log), &k.witness_vkey], &line);
    let bundle = log.join("tile/entries/000.p/2");
    let mut b = std::fs::read(&bundle).unwrap();
    let last = b.len() - 2;
    b[last] ^= 1;
    std::fs::write(&bundle, &b).unwrap();
    let r3 = write(t.path(), "r3.json", "{\"v\":3}\n");
    refused(&["append", p(&log), p(&k.log_secret), p(&r3)], "", "root");
    refused(&["prove", p(&log), p(&r2)], "", "root");
}

/// The add-checkpoint request with the first proof hash changed.
fn tamper_proof_line(request: &str) -> String {
    let mut lines: Vec<String> = request.split('\n').map(String::from).collect();
    assert!(lines[0].starts_with("old "), "{request}");
    assert!(!lines[1].is_empty(), "no consistency proof in {request:?}");
    let mut raw = base64_decode(&lines[1]);
    raw[0] ^= 1;
    lines[1] = base64_encode(&raw);
    lines.join("\n")
}

/// A cosignature line with one byte of its signature changed.
fn forge(line: &str) -> String {
    let (head, sig) = line.trim_end().rsplit_once(' ').unwrap();
    let mut raw = base64_decode(sig);
    let last = raw.len() - 1;
    raw[last] ^= 1;
    format!("{head} {}\n", base64_encode(&raw))
}

fn base64_decode(s: &str) -> Vec<u8> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.decode(s).unwrap()
}

fn base64_encode(b: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(b)
}

#[test]
fn verify_refuses_a_reference_the_pinned_log_does_not_show() {
    let t = tempfile::tempdir().unwrap();
    let k = keys(t.path());
    let log = t.path().join("log");
    let r1 = write(t.path(), "r1.json", "{\"v\":1}\n");
    ok(&["append", p(&log), p(&k.log_secret), p(&r1)], "");
    cosign_round(&k, &log, &t.path().join("ws"), 0);
    ok(&["prove", p(&log), p(&r1)], "");
    let pin = write(t.path(), "log.pin", &pin_text(&k));
    ok(&["verify", p(&pin), p(&r1), p(&log)], "");
    // The same bytes changed: the proof leads from another entry.
    let proof = std::fs::read(transparency::proof_path(&r1)).unwrap();
    let r1b = write(t.path(), "r1b.json", "{\"v\":9}\n");
    std::fs::write(transparency::proof_path(&r1b), proof).unwrap();
    refused(
        &["verify", p(&pin), p(&r1b), p(&log)],
        "",
        "not shown in the log",
    );
    // A pin of another witness: the cosignature does not count.
    let other = t.path().join("w9-REHEARSAL.key");
    let w9 = field(&ok(&["keygen", "witness", WITNESS, p(&other)], ""), "vkey");
    let pin9 = write(
        t.path(),
        "log9.pin",
        &pin_text(&k).replace(&k.witness_vkey, &w9),
    );
    refused(&["verify", p(&pin9), p(&r1), p(&log)], "", "cosign");
    // A pin the relay would not take.
    let bad = write(
        t.path(),
        "bad.pin",
        &pin_text(&k).replace("quorum 1", "quorum 2"),
    );
    refused(&["verify", p(&bad), p(&r1), p(&log)], "", "quorum");
}

/// The refresh a log runs before its cosignatures pass the pinned maximum
/// age: the same checkpoint sent again (`old` = its size, no proof), and
/// each witness's newer cosignature replaces its older one. No log key.
#[test]
fn a_newer_cosignature_replaces_the_older_one_of_the_same_witness() {
    use x0ne_infer::tlog_ops::{cosign, Log, Secret};
    let t = tempfile::tempdir().unwrap();
    let k = keys(t.path());
    let log_dir = t.path().join("log");
    let r1 = write(t.path(), "r1.json", "{\"v\":1}\n");
    ok(&["append", p(&log_dir), p(&k.log_secret), p(&r1)], "");
    let witness = Secret::read(&k.witness_secret).unwrap();
    let log_key = NoteKey::parse(&k.log_vkey).unwrap();
    let wkey = NoteKey::parse(&k.witness_vkey).unwrap();
    let state = t.path().join("ws");
    let mut log = Log::open(&log_dir).unwrap();
    let request = log.witness_request(0).unwrap();
    let old = cosign(&witness, &log_key, &state, &request, 1_000).unwrap();
    assert_eq!(
        log.add_cosignatures(std::slice::from_ref(&wkey), &old),
        Ok(1)
    );
    let request = log.witness_request(1).unwrap();
    assert!(request.starts_with("old 1\n\n"), "{request}");
    let new = cosign(&witness, &log_key, &state, &request, 2_000).unwrap();
    assert_eq!(
        log.add_cosignatures(std::slice::from_ref(&wkey), &new),
        Ok(1)
    );
    // An older one never replaces a newer one.
    assert_eq!(
        log.add_cosignatures(std::slice::from_ref(&wkey), &old),
        Ok(0)
    );
    let note = std::fs::read_to_string(log_dir.join("checkpoint")).unwrap();
    let n = x0ne_infer::tlog::parse_note(&note).unwrap();
    let times: Vec<Option<u64>> = x0ne_infer::tlog::open_note(&n, &[log_key, wkey])
        .unwrap()
        .into_iter()
        .map(|v| v.time)
        .collect();
    assert_eq!(times, vec![None, Some(2_000)], "{note}");
}
