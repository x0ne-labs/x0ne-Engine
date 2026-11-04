//! The caller's memory of the transparency log (`attest::LogMemory`): the
//! SDK keeps the last checkpoint it verified and holds every new one to it,
//! as a witness does (C2SP tlog-witness), on the TEST log grown in memory.
//! A log that went back, forked, or grew without a consistency proof is
//! refused with the relay's log code, 100, at a step named for it.

use x0ne_sdk::attest::{self, AdmittedReference, LogMemory, Refusal, SignedReference, TestLog};

/// The reference of the 2026-10-18 run with RTMR0 pinned, as `abi.rs` uses it.
const REFERENCE: &[u8] =
    include_bytes!("../../x0ne-infer/tests/run_fixtures/x0ne-cvm-0.2.3-rtmr0-pinned.json");
const NOW: u64 = 1_800_000_000;

/// A TEST log of `others` entries before [`REFERENCE`] and one after it: the
/// reference is at index `others`.
fn log_with(others: &[&[u8]]) -> TestLog {
    let mut log = TestLog::new();
    for o in others {
        log.append(o);
    }
    log.append(REFERENCE);
    log.append(b"an entry after the reference");
    log
}

/// The reference, signed by the TEST archive key and proven in `log`,
/// admitted against `memory` with `consistency`.
fn admit(
    log: &TestLog,
    index: u64,
    memory: &mut LogMemory,
    consistency: Option<&str>,
) -> Result<AdmittedReference, Refusal> {
    let (signature, _) = attest::sign_reference_under_test_keys(REFERENCE, NOW);
    let proof = log.proof(index, NOW);
    attest::admit_reference_under_test_keys(
        &SignedReference {
            json: REFERENCE,
            signature: &signature,
            tlog_proof: &proof,
        },
        memory,
        consistency,
        NOW,
    )
}

fn memory_of(a: &AdmittedReference) -> (Option<u64>, u64) {
    let m = &a.provenance()["log"]["memory"];
    (m["before"].as_u64(), m["kept"].as_u64().unwrap())
}

fn refused(r: Result<AdmittedReference, Refusal>, step: &str) -> Refusal {
    let e = r
        .err()
        .unwrap_or_else(|| panic!("admitted where {step} was expected"));
    assert_eq!((e.step, e.code), (step, 100), "{e}");
    e
}

/// The first checkpoint a caller meets is kept; the same one again needs no
/// proof and changes nothing.
#[test]
fn the_first_checkpoint_is_kept_and_the_same_one_needs_no_proof() {
    let log = log_with(&[b"entry 0"]);
    let mut memory = LogMemory::new();
    assert_eq!(memory.note(), None);
    assert_eq!(memory.checkpoint(), None);
    let a = admit(&log, 1, &mut memory, None).expect("the first sight");
    assert_eq!(memory_of(&a), (None, 3));
    let note = memory.note().expect("a note to keep").to_string();
    assert!(
        note.starts_with("test.invalid/x0ne-engine/tlog\n3\n"),
        "{note}"
    );
    assert_eq!(memory.checkpoint().map(|(s, _)| s), Some(3));
    let a = admit(&log, 1, &mut memory, None).expect("the same checkpoint");
    assert_eq!(memory_of(&a), (Some(3), 3));
    assert_eq!(memory.note(), Some(note.as_str()));
    // An empty consistency body from the remembered size is accepted too.
    admit(&log, 1, &mut memory, Some(&log.consistency(3, NOW))).expect("old 3, no hash");
}

/// SECURITY: once the log has grown, the new checkpoint is admitted only
/// with a consistency proof from the remembered size (tlog-witness
/// `add-checkpoint`); without one, refused at `log-consistency` with 100,
/// and the memory does not move.
#[test]
fn a_grown_log_is_admitted_on_a_consistency_proof_from_the_remembered_size() {
    let mut log = log_with(&[b"entry 0"]);
    let mut memory = LogMemory::new();
    admit(&log, 1, &mut memory, None).expect("size 3");
    let at_three = memory.note().unwrap().to_string();
    log.append(b"entry 3");
    log.append(b"entry 4");
    assert_eq!(log.size(), 5);
    let e = refused(admit(&log, 1, &mut memory, None), "log-consistency");
    assert!(
        e.reason.contains(
            "the log grew from 3 to 5: a consistency proof from 3 is required, none was given"
        ),
        "{e}"
    );
    assert_eq!(
        memory.note(),
        Some(at_three.as_str()),
        "the memory moved on a refusal"
    );
    let body = log.consistency(3, NOW);
    assert!(body.starts_with("old 3\n"), "{body}");
    let a = admit(&log, 1, &mut memory, Some(&body)).expect("3 shown a prefix of 5");
    assert_eq!(memory_of(&a), (Some(3), 5));
    assert_eq!(memory.checkpoint().map(|(s, _)| s), Some(5));
    assert_ne!(memory.note(), Some(at_three.as_str()));
}

/// SECURITY: a proof whose checkpoint is smaller than the one remembered is
/// a log that went back, or a stale proof: refused at `log-rollback`, the
/// memory kept. The caller needs a proof against the log's current head.
#[test]
fn an_older_checkpoint_than_the_remembered_one_is_refused_as_a_rollback() {
    let small = log_with(&[b"entry 0"]);
    let mut large = small.clone();
    large.append(b"entry 3");
    large.append(b"entry 4");
    let mut memory = LogMemory::new();
    admit(&large, 1, &mut memory, None).expect("size 5 first");
    let at_five = memory.note().unwrap().to_string();
    let e = refused(admit(&small, 1, &mut memory, None), "log-rollback");
    assert!(
        e.reason.contains("size 3") && e.reason.contains("size 5"),
        "{e}"
    );
    // With a consistency body, still: the body does not make the log newer.
    let e = refused(
        admit(&small, 1, &mut memory, Some(&large.consistency(3, NOW))),
        "log-rollback",
    );
    assert!(e.reason.contains("went back"), "{e}");
    assert_eq!(memory.note(), Some(at_five.as_str()));
}

/// SECURITY: two checkpoints of one size signed by the log, with two roots,
/// are a fork (a split view): refused at `log-fork` from either side, both
/// roots named, the memory kept.
#[test]
fn two_signed_checkpoints_of_one_size_with_two_roots_are_a_fork() {
    let a = log_with(&[b"entry 0"]);
    let b = log_with(&[b"another entry 0"]);
    assert_eq!(a.size(), b.size());
    let mut memory = LogMemory::new();
    admit(&a, 1, &mut memory, None).expect("history A");
    let seen = memory.note().unwrap().to_string();
    let (_, root_a) = memory.checkpoint().unwrap();
    let e = refused(admit(&b, 1, &mut memory, None), "log-fork");
    assert!(
        e.reason.contains("two signed checkpoints of size 3") && e.reason.contains(&root_a),
        "{e}"
    );
    assert_eq!(memory.note(), Some(seen.as_str()));
    let mut memory_b = LogMemory::new();
    admit(&b, 1, &mut memory_b, None).expect("history B alone");
    refused(admit(&a, 1, &mut memory_b, None), "log-fork");
}

/// SECURITY: a consistency proof truncated, altered, from another size, to
/// another checkpoint, or from another history (a fork at a larger size) is
/// refused at `log-consistency`; a proof given with no memory must be from
/// size 0 and empty. The memory never moves on a refusal.
#[test]
fn a_consistency_proof_truncated_altered_misdirected_or_from_another_history_is_refused() {
    let mut a = log_with(&[b"entry 0"]);
    let mut memory = LogMemory::new();
    admit(&a, 1, &mut memory, None).expect("A at 3");
    let at_three = memory.note().unwrap().to_string();
    a.append(b"entry 3");
    a.append(b"entry 4");
    let good = a.consistency(3, NOW);
    let (head, checkpoint) = good.split_once("\n\n").unwrap();
    let mut lines: Vec<&str> = head.lines().collect();
    assert!(
        lines.len() >= 3,
        "a proof from 3 to 5 has at least two hashes: {good}"
    );
    let truncated = format!("{}\n\n{checkpoint}", lines[..lines.len() - 1].join("\n"));
    let zero = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
    let altered = {
        let kept = lines[1];
        lines[1] = zero;
        let s = format!("{}\n\n{checkpoint}", lines.join("\n"));
        lines[1] = kept;
        s
    };
    let from_two = good.replacen("old 3\n", "old 2\n", 1);
    let mut six = a.clone();
    six.append(b"entry 5");
    let to_six = six.consistency(3, NOW);
    let mut b = log_with(&[b"another entry 0"]);
    b.append(b"entry 3");
    b.append(b"entry 4");
    let from_b = b.consistency(3, NOW);
    let empty_from_three = format!("old 3\n\n{checkpoint}");
    // Without the blank line, the parser meets the checkpoint's own blank
    // line and reads its text as proof lines: not base64, refused.
    let no_blank_line = good.replacen("\n\n", "\n", 1);
    for (what, body, says) in [
        ("truncated", truncated.as_str(), "shorter than the trees"),
        ("altered", altered.as_str(), "is not a prefix of"),
        (
            "from size 2",
            from_two.as_str(),
            "a proof from size 2, the last checkpoint verified here has size 3",
        ),
        ("to size 6", to_six.as_str(), "a proof to size 6"),
        ("from history B", from_b.as_str(), "a proof to size 5 root"),
        (
            "empty",
            empty_from_three.as_str(),
            "empty consistency proof",
        ),
        (
            "no blank line",
            no_blank_line.as_str(),
            "proof line \"test.invalid/x0ne-engine/tlog\"",
        ),
    ] {
        let e = refused(admit(&a, 1, &mut memory, Some(body)), "log-consistency");
        assert!(e.reason.contains(says), "{what}: {e}");
        assert_eq!(
            memory.note(),
            Some(at_three.as_str()),
            "{what}: the memory moved"
        );
    }
    // History B's own proof, with its own consistency body: the body leads to
    // B's checkpoint, which the reference's proof names too, and the hashes
    // do not extend A's root. A fork at a larger size.
    let e = refused(admit(&b, 1, &mut memory, Some(&from_b)), "log-consistency");
    assert!(e.reason.contains("is not a prefix of"), "{e}");
    assert_eq!(memory.note(), Some(at_three.as_str()));
    // The control, then from no memory: a body given must start at size 0.
    admit(&a, 1, &mut memory, Some(&good)).expect("the control");
    let mut fresh = LogMemory::new();
    let e = refused(admit(&a, 1, &mut fresh, Some(&good)), "log-consistency");
    assert!(
        e.reason
            .contains("a proof from size 3, the last checkpoint verified here has size 0"),
        "{e}"
    );
    assert_eq!(fresh.note(), None);
    admit(&a, 1, &mut fresh, Some(&a.consistency(0, NOW))).expect("old 0, no hash");
}

/// The memory is its signed note: kept as text, given back, verified under
/// the pinned log at the next admission. A note altered, or not a
/// checkpoint of the pinned log, refuses every reference at `log-memory`
/// until the caller clears it; `checkpoint` reads it without verifying.
#[test]
fn the_memory_round_trips_as_its_signed_note_and_a_tampered_note_refuses() {
    let log = log_with(&[b"entry 0"]);
    let mut memory = LogMemory::new();
    admit(&log, 1, &mut memory, None).expect("size 3");
    let note = memory.note().unwrap().to_string();
    let mut back = LogMemory::from_note(note.clone());
    assert_eq!(back, memory);
    assert_eq!(back.checkpoint(), memory.checkpoint());
    admit(&log, 1, &mut back, None).expect("the same memory, from its text");
    // The log's signature line, one character changed.
    let sig_line = note
        .lines()
        .find(|l| l.starts_with("\u{2014} test.invalid/x0ne-engine/tlog "))
        .expect("the log's signature line");
    let mut chars: Vec<char> = sig_line.chars().collect();
    let last = chars.len() - 2;
    chars[last] = if chars[last] == 'A' { 'B' } else { 'A' };
    let tampered: String = chars.into_iter().collect();
    let mut bad = LogMemory::from_note(note.replace(sig_line, &tampered));
    assert_eq!(
        bad.checkpoint().map(|(s, _)| s),
        Some(3),
        "readable without its signature"
    );
    let e = refused(admit(&log, 1, &mut bad, None), "log-memory");
    assert!(e.reason.contains("the remembered checkpoint"), "{e}");
    // Another origin under the same text: not the pinned log's.
    let mut other = LogMemory::from_note(note.replacen(
        "test.invalid/x0ne-engine/tlog\n",
        "other.invalid/log\n",
        1,
    ));
    refused(admit(&log, 1, &mut other, None), "log-memory");
    // Not a note at all.
    let mut garbage = LogMemory::from_note("not a checkpoint");
    assert_eq!(garbage.checkpoint(), None);
    refused(admit(&log, 1, &mut garbage, None), "log-memory");
    // A note past the cap, refused unread.
    let mut huge = LogMemory::from_note("x".repeat(64 * 1024 + 1));
    let e = refused(admit(&log, 1, &mut huge, None), "log-memory");
    assert!(e.reason.contains("cap 65536"), "{e}");
}

/// `admit_reference`, under the production anchors. A debug build pins the
/// TEST key and the TEST log there (`Anchor::production`): the reference is
/// admitted and held to the memory as under the TEST anchors. A release
/// build made before the signing ceremony pins nothing: every reference is
/// refused with 99 before the log is read, and the memory stays empty.
#[test]
fn the_production_anchors_hold_the_memory_in_debug_and_pin_nothing_before_the_ceremony() {
    let log = log_with(&[b"entry 0"]);
    let (signature, _) = attest::sign_reference_under_test_keys(REFERENCE, NOW);
    let proof = log.proof(1, NOW);
    let signed = SignedReference {
        json: REFERENCE,
        signature: &signature,
        tlog_proof: &proof,
    };
    let mut memory = LogMemory::new();
    let r = attest::admit_reference(&signed, &mut memory, None, NOW);
    if cfg!(debug_assertions) {
        let a = r.expect("the TEST anchors of a debug build");
        assert_eq!(memory_of(&a), (None, 3));
        assert_eq!(memory.checkpoint().map(|(s, _)| s), Some(3));
        attest::admit_reference(&signed, &mut memory, None, NOW).expect("the same checkpoint");
        let fork = log_with(&[b"another entry 0"]);
        let fork_proof = fork.proof(1, NOW);
        let forked = SignedReference {
            json: REFERENCE,
            signature: &signature,
            tlog_proof: &fork_proof,
        };
        refused(
            attest::admit_reference(&forked, &mut memory, None, NOW),
            "log-fork",
        );
    } else {
        let e = r
            .err()
            .unwrap_or_else(|| panic!("admitted: a release build pins no key"));
        assert_eq!((e.step, e.code), ("reference", 99), "{e}");
        assert!(e.reason.contains("pins no x0ne archive key"), "{e}");
        assert_eq!(memory, LogMemory::new());
    }
}
