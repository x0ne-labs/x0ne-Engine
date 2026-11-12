//! Hostile bytes against the textual formats a third party hands this crate:
//! signed notes, checkpoints and log proofs, Merkle proofs, NVIDIA's JWT and
//! its claims, the JSON documents a deployment publishes (references,
//! measurements, model configurations, receipts), the bodies of the OpenAI
//! API (chat requests, JSON schemas), the relay's control lines, an SSE
//! stream. The binary formats are in `hostile_bytes_tests.rs`, which says
//! what the statements are; the harness is
//! `x0ne-sdk/tests/support/hostile.rs`.

use crate::hostile::{cases, prefixes, Campaign};
use crate::hostile_json::{json_mutant, nested, DEEPEST};
use crate::tlog::{self, Hash, NoteKey};
use crate::transparency::{self, Policy, Tiles};
use serde_json::{json, Value};

const GO_VKEYS: &str = include_str!("../tests/tlog/go/vkeys");
const GO_REFERENCE: &[u8] = include_bytes!("../tests/tlog/go/reference-300.json");
const GO_PROOF: &str = include_str!("../tests/tlog/go/reference-300.json.tlog-proof");
const GO_CHECKPOINT: &str = include_str!("../tests/tlog/go/checkpoint-301");
const GO_ORIGIN: &str = "interop.invalid/go-log";
/// 2026-10-23, the day after the Go fixtures were cosigned.
const GO_NOW: u64 = 1_792_750_000;

fn go_key(role: &str) -> NoteKey {
    let line = GO_VKEYS
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{role} ")))
        .expect("a verifier key of that role");
    NoteKey::parse(line).expect("Go's verifier key parses")
}

fn go_policy() -> Policy {
    Policy {
        origin: GO_ORIGIN.into(),
        log: go_key("log"),
        witnesses: vec![go_key("witness")],
        quorum: 1,
        max_age_s: 100 * 365 * 24 * 3600,
        tiles: Tiles::BesideReference,
    }
}

#[test]
fn a_log_proof_cut_or_changed_is_refused_or_shows_the_same_entry() {
    let policy = go_policy();
    let check = |reference: &[u8], proof: &str| {
        transparency::check_offline(reference, proof, &policy, GO_NOW)
            .map(|l| (l.origin, l.index, l.size, l.root_hex, l.cosigned_by))
    };
    let control = check(GO_REFERENCE, GO_PROOF).expect("Go's proof shows Go's entry");

    let mut c = Campaign::new("transparency proof", 0x7109);
    for p in prefixes(GO_PROOF.as_bytes()) {
        let Ok(p) = std::str::from_utf8(p) else {
            continue;
        };
        let what = format!("the first {} bytes of the proof", p.len());
        let r = c.run(&what, p.len(), || check(GO_REFERENCE, p));
        c.hold(matches!(r, Some(Err(_))), "a cut proof is believed", &what);
    }
    for p in prefixes(GO_REFERENCE) {
        let what = format!("the first {} bytes of the logged file", p.len());
        let r = c.run(&what, p.len(), || check(p, GO_PROOF));
        c.hold(
            matches!(r, Some(Err(_))),
            "a cut file is shown by the proof of the whole one",
            &what,
        );
    }
    // Every character of the proof replaced: the index, a hash of the path,
    // the checkpoint or a signature changed.
    for (i, ch) in GO_PROOF.char_indices() {
        if !ch.is_ascii() {
            continue;
        }
        let other = if ch == 'A' { 'B' } else { 'A' };
        let mut m = GO_PROOF.to_string();
        m.replace_range(i..i + 1, &other.to_string());
        let what = format!("byte {i} of the proof ({ch:?}) replaced by {other:?}");
        if let Some(Ok(v)) = c.run(&what, m.len(), || check(GO_REFERENCE, &m)) {
            c.hold(v == control, "a changed proof shows another entry", &what);
            c.accepted_unchanged();
        }
    }
    for _ in 0..cases(1500) {
        let (m, what) = c.text_mutant(GO_PROOF, GO_CHECKPOINT);
        if let Some(Ok(v)) = c.run(&what, m.len(), || check(GO_REFERENCE, &m)) {
            c.hold(v == control, "a mutated proof shows another entry", &what);
            c.accepted_unchanged();
        }
        let (m, what) = c.mutant(GO_REFERENCE, GO_PROOF.as_bytes());
        let what = format!("logged file: {what}");
        let r = c.run(&what, m.len(), || check(&m, GO_PROOF));
        c.hold(
            !matches!(r, Some(Ok(_))) || m == GO_REFERENCE,
            "a changed file is shown by the proof of the intact one",
            &what,
        );
    }
    c.finish();
}

#[test]
fn notes_checkpoints_and_proofs_cut_or_changed_never_panic_and_read_back_the_same() {
    let keys = [go_key("log"), go_key("witness")];
    let note = tlog::parse_note(GO_CHECKPOINT).expect("Go's checkpoint is a note");
    assert_eq!(tlog::open_note(&note, &keys).map(|v| v.len()), Ok(2));

    let mut c = Campaign::new("signed notes", 0x207e);
    for p in prefixes(GO_CHECKPOINT.as_bytes()) {
        let Ok(p) = std::str::from_utf8(p) else {
            continue;
        };
        let what = format!("the first {} bytes of a cosigned checkpoint", p.len());
        let r = c.run(&what, p.len(), || {
            tlog::parse_note(p).and_then(|n| tlog::open_note(&n, &keys))
        });
        // A note cut after a whole signature line is a note with fewer
        // signatures: never as many as the whole one.
        c.hold(
            !matches!(&r, Some(Ok(v)) if v.len() >= 2),
            "a cut note keeps all its signatures",
            &what,
        );
    }
    for _ in 0..cases(4000) {
        let (m, what) = c.text_mutant(GO_CHECKPOINT, GO_PROOF);
        let opened = c.run(&what, m.len(), || {
            let n = tlog::parse_note(&m)?;
            let who = tlog::open_note(&n, &keys)?;
            Ok::<_, String>((n, who))
        });
        if let Some(Ok((n, who))) = opened {
            // A signature that verifies covers the text: it is the intact one.
            c.hold(
                who.is_empty() || n.text == note.text,
                "a signature verifies on a changed text",
                &what,
            );
            if let Some(Ok(cp)) = c.run(&what, m.len(), || tlog::parse_checkpoint(&n.text)) {
                let again = tlog::parse_checkpoint(&cp.text());
                c.hold(
                    again.as_ref() == Ok(&cp),
                    "a checkpoint does not read back the same",
                    &what,
                );
            }
        }
        let (m, what) = c.text_mutant(GO_PROOF, GO_CHECKPOINT);
        if let Some(Ok(p)) = c.run(&what, m.len(), || tlog::parse_tlog_proof(&m)) {
            if p.extra.is_none() {
                let text = tlog::format_tlog_proof(p.index, &p.hashes, &p.checkpoint);
                c.hold(
                    tlog::parse_tlog_proof(&text).as_ref() == Ok(&p),
                    "a proof does not read back the same",
                    &what,
                );
            }
        }
        // The witness protocol's request: `old <size>`, a consistency path,
        // a checkpoint.
        let request = tlog::format_add_checkpoint(300, &[[7u8; 32], [9u8; 32]], GO_CHECKPOINT);
        let (m, what) = c.text_mutant(&request, GO_PROOF);
        if let Some(Ok(a)) = c.run(&what, m.len(), || tlog::parse_add_checkpoint(&m)) {
            let text = tlog::format_add_checkpoint(a.old, &a.hashes, &a.checkpoint);
            c.hold(
                tlog::parse_add_checkpoint(&text).as_ref() == Ok(&a),
                "a witness request does not read back the same",
                &what,
            );
        }
        let vkey = keys[c.rng.below(2)].vkey();
        let (m, what) = c.text_mutant(&vkey, GO_VKEYS);
        if let Some(Ok(k)) = c.run(&what, m.len(), || NoteKey::parse(&m)) {
            c.hold(
                NoteKey::parse(&k.vkey()).as_ref() == Ok(&k),
                "a verifier key does not read back the same",
                &what,
            );
            c.hold(k.vkey() == m, "two spellings of one verifier key", &what);
        }
        let (m, what) = c.text_mutant("18446744073709551615", "007");
        c.run(&what, m.len(), || {
            let _ = tlog::decimal(&m);
            let _ = tlog::b64_canonical(&m);
        });
    }
    c.finish();
}

#[test]
fn a_merkle_proof_changed_in_any_way_is_refused_for_every_small_tree() {
    let leaves: Vec<Hash> = (0u64..40)
        .map(|i| tlog::leaf_hash(&i.to_le_bytes()))
        .collect();
    let mut c = Campaign::new("Merkle proofs", 0x3e41);
    for size in 1..=leaves.len() as u64 {
        let tree = &leaves[..size as usize];
        let root = tlog::mth_of(tree);
        for index in 0..size {
            let leaf = tree[index as usize];
            let proof = tlog::inclusion_proof(&mut tlog::Leaves(tree), index, size)
                .expect("a proof of a leaf of the tree");
            let what = format!("leaf {index} of {size}");
            c.hold(
                tlog::verify_inclusion(&leaf, index, size, &proof, &root).is_ok(),
                "an inclusion proof this crate built is refused",
                &what,
            );
            let refused = |c: &mut Campaign, what: String, leaf: &Hash, i: u64, p: &[Hash]| {
                let r = c.run(&what, 32 * p.len(), || {
                    tlog::verify_inclusion(leaf, i, size, p, &root)
                });
                c.hold(
                    matches!(r, Some(Err(_))),
                    "a changed inclusion proof verifies",
                    &what,
                );
            };
            for other in 0..size {
                if other != index {
                    let what = format!("{what}, presented as leaf {other}");
                    refused(&mut c, what, &leaf, other, &proof);
                }
            }
            for k in 0..proof.len() {
                let mut p = proof.clone();
                p[k][(k * 7) % 32] ^= 1 << (k % 8);
                refused(
                    &mut c,
                    format!("{what}, hash {k} changed"),
                    &leaf,
                    index,
                    &p,
                );
                let mut p = proof.clone();
                p.remove(k);
                refused(
                    &mut c,
                    format!("{what}, hash {k} removed"),
                    &leaf,
                    index,
                    &p,
                );
                let mut p = proof.clone();
                p.insert(k, proof[k]);
                refused(
                    &mut c,
                    format!("{what}, hash {k} repeated"),
                    &leaf,
                    index,
                    &p,
                );
            }
            let mut other_leaf = leaf;
            other_leaf[31] ^= 0x80;
            refused(
                &mut c,
                format!("{what}, another leaf"),
                &other_leaf,
                index,
                &proof,
            );
        }
        for old in 1..size {
            let old_root = tlog::mth_of(&tree[..old as usize]);
            let proof = tlog::consistency_proof(&mut tlog::Leaves(tree), old, size)
                .expect("a consistency proof between two sizes of the tree");
            let what = format!("consistency from {old} to {size}");
            c.hold(
                tlog::verify_consistency(old, size, &proof, &old_root, &root).is_ok(),
                "a consistency proof this crate built is refused",
                &what,
            );
            for k in 0..proof.len() {
                let mut p = proof.clone();
                p[k][k % 32] ^= 0x10;
                let what = format!("{what}, hash {k} changed");
                let r = c.run(&what, 32 * p.len(), || {
                    tlog::verify_consistency(old, size, &p, &old_root, &root)
                });
                c.hold(
                    matches!(r, Some(Err(_))),
                    "a changed consistency proof verifies",
                    &what,
                );
            }
            let mut forked = old_root;
            forked[0] ^= 1;
            let r = c.run(&what, 32 * proof.len(), || {
                tlog::verify_consistency(old, size, &proof, &forked, &root)
            });
            c.hold(
                matches!(r, Some(Err(_))),
                "a fork is consistent with the tree",
                &what,
            );
        }
    }
    // Sizes and indexes nobody built a tree for.
    for _ in 0..cases(20_000) {
        let size =
            *c.rng.pick(&[0, 1, 2, 3, 1 << 32, u64::MAX - 1, u64::MAX]) ^ c.rng.below(4) as u64;
        let index = c.rng.next_u64() >> c.rng.below(64);
        let n = c.rng.below(70);
        let proof: Vec<Hash> = (0..n).map(|_| [c.rng.byte(); 32]).collect();
        let (a, b) = ([c.rng.byte(); 32], [c.rng.byte(); 32]);
        let what = format!("index {index}, sizes {index} and {size}, {n} arbitrary hashes");
        let r = c.run(&what, 32 * n, || {
            (
                tlog::verify_inclusion(&a, index, size, &proof, &b).is_ok(),
                tlog::verify_consistency(index, size, &proof, &a, &b).is_ok(),
            )
        });
        if let Some((inclusion, consistency)) = r {
            c.hold(!inclusion, "arbitrary hashes prove an inclusion", &what);
            c.hold(
                !consistency || (index == size && a == b) || (index == 0 && n == 0),
                "arbitrary hashes prove a consistency",
                &what,
            );
        }
    }
    c.finish();
}

#[test]
fn nvidias_token_cut_or_changed_is_refused_or_verifies_to_the_same_claims() {
    use crate::nras;
    let f = nras::tests::fixture_for_composite_p384();
    const NOW: u64 = 1_500;
    let verify = |token: &str| {
        nras::verify_nras_token(token, &f.jwks, &f.nonce_hex, NOW, &f.pins)
            .map(|v| format!("{v:?}"))
    };
    let control = verify(&f.token).expect("the fixture's token verifies");
    let signed = f.token.rfind('.').expect("a JWS of three parts");

    let mut c = Campaign::new("NRAS token", 0x2a5);
    for n in crate::hostile::lengths(f.token.len(), 37, 8) {
        let what = format!("the first {n} bytes of the token");
        let r = c.run(&what, n, || verify(&f.token[..n]));
        c.hold(matches!(r, Some(Err(_))), "a cut token verifies", &what);
    }
    // One character of the signed part in a hundred (every one from
    // X0NE_FUZZ_ITERS=10): a signature is checked for each.
    let step = if crate::hostile::depth() >= 10 { 1 } else { 97 };
    for i in (0..signed).step_by(step) {
        let ch = f.token.as_bytes()[i];
        if ch == b'.' {
            continue;
        }
        let other = if ch == b'A' { "B" } else { "A" };
        let mut m = f.token.clone();
        m.replace_range(i..i + 1, other);
        let what = format!("byte {i} of the signed part replaced by {other}");
        let r = c.run(&what, m.len(), || verify(&m));
        c.hold(
            matches!(r, Some(Err(_))),
            "a token with a changed signed byte verifies",
            &what,
        );
    }
    for _ in 0..cases(250) {
        let (m, what) = c.text_mutant(&f.token, &f.token[signed..]);
        if let Some(Ok(v)) = c.run(&what, m.len(), || verify(&m)) {
            c.hold(
                v == control,
                "a mutated token verifies to other claims",
                &what,
            );
            c.accepted_unchanged();
        }
        let (jwks, what) = json_mutant(&mut c.rng, &f.jwks);
        let what = format!("JWKS {what}");
        if let Some(Ok(v)) = c.run(&what, f.token.len(), || {
            nras::verify_nras_token(&f.token, &jwks, &f.nonce_hex, NOW, &f.pins)
                .map(|v| format!("{v:?}"))
        }) {
            c.hold(
                v == control,
                "a mutated JWKS verifies to other claims",
                &what,
            );
            c.accepted_unchanged();
        }
    }
    // What NVIDIA would have to sign: the verifier reads claims of any shape.
    let claims: Value = {
        use base64::Engine;
        let payload = f.token.split('.').nth(1).expect("a payload");
        let raw = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(payload)
            .expect("base64url");
        serde_json::from_slice(&raw).expect("JSON claims")
    };
    for _ in 0..cases(60) {
        let (m, what) = json_mutant(&mut c.rng, &claims);
        let token = (f.sign)(&m);
        let what = format!("signed claims with {what}");
        c.run(&what, token.len(), || {
            let _ = verify(&token);
        });
    }
    for live in [
        include_str!("../tests/nras_live/gpu0_2026-10-17.claims.json"),
        include_str!("../tests/nras_live/gpu0_2026-10-18.claims.json"),
    ] {
        let live: Value = serde_json::from_str(live).expect("NVIDIA's claims are JSON");
        for _ in 0..cases(1500) {
            let (m, what) = json_mutant(&mut c.rng, &live);
            let size = m.to_string().len();
            c.run(&what, size, || {
                let _ = nras::gpu_policy(&m);
                let _ = crate::nras_client::split_attest_response(&m);
                let _ = crate::nras_client::split_attest_response(&json!([["JWT", "x"], m]));
            });
        }
    }
    for _ in 0..cases(2000) {
        use base64::Engine;
        let chain = base64::engine::general_purpose::STANDARD.encode(&f.leaf_der);
        let (m, what) = c.text_mutant(&chain, &f.token);
        c.run(&what, m.len(), || {
            let _ = nras::gpu_leaf_serial_decimal(&m);
        });
        let (der, what) = c.mutant(&f.leaf_der, &f.inter_der);
        let b64 = base64::engine::general_purpose::STANDARD.encode(&der);
        c.run(&what, b64.len(), || {
            let _ = nras::gpu_leaf_serial_decimal(&b64);
        });
    }
    c.finish();
}

/// Every JSON document vendored under `tests/` that a parser of this crate
/// reads, by the name it is reported under.
const JSON_FIXTURES: &[(&str, &str)] = &[
    (
        "reference 0.2.3",
        include_str!("../tests/reference/x0ne-cvm-0.2.3.json"),
    ),
    (
        "reference 0.2.4",
        include_str!("../tests/reference/x0ne-cvm-0.2.4.json"),
    ),
    (
        "reference 0.2 v1",
        include_str!("../tests/reference/x0ne-cvm-0.2-v1.json"),
    ),
    (
        "reference of the run",
        include_str!("../tests/run_fixtures/x0ne-cvm-0.2.3-rtmr0-pinned.json"),
    ),
    (
        "measurements",
        include_str!("../tests/measurements/fixture.json"),
    ),
    (
        "olmo2 config",
        include_str!("../tests/arch/real/olmo2-1b-instruct.json"),
    ),
    (
        "olmo3 config",
        include_str!("../tests/arch/real/olmo3-7b-instruct.json"),
    ),
    (
        "ministral3 config",
        include_str!("../tests/arch/real/ministral3-8b-instruct-2512.json"),
    ),
    (
        "qwen3 config",
        include_str!("../tests/arch/real/qwen3-8b.json"),
    ),
    (
        "qwen3 config with rope_parameters",
        include_str!("../tests/qwen3_config_rope_parameters.json"),
    ),
    (
        "receipt",
        include_str!("../tests/receipts/receipt_before_tcb_disclosure.receipt.json"),
    ),
    (
        "Go's logged reference",
        include_str!("../tests/tlog/go/reference-300.json"),
    ),
];

/// Every reader of a JSON document, on bytes of any shape.
fn read_as_every_document(bytes: &[u8]) {
    let _ = crate::reference::parse(bytes);
    let _ = crate::snp_reference::parse(bytes);
    let _ = crate::arch::DecoderConfig::parse(bytes);
    let _ = crate::model_spec::parse_qwen3_config(bytes);
    let _ = crate::model_key::Wrapped::parse(bytes);
    let _ = crate::model_key::Grant::parse(bytes);
    let _ = crate::model_key::ReleasePolicy::parse(bytes);
    let _ = crate::channel_key::KeyRequest::parse(bytes);
    let _ = crate::shards::Index::parse(bytes);
    let _ = crate::nras_http::NrasKey::parse(bytes);
    if let Ok(loaded) = crate::cloud_config::parse(bytes) {
        let _ = format!("{loaded:?}");
    }
    if let Ok(text) = std::str::from_utf8(bytes) {
        let _ = crate::measurements::parse(text, "0.2.3", "a test");
        let _ = crate::tlog_ops::PinFile::parse(text);
        let _ = crate::tlog_ops::Secret::parse(text);
        let _ = crate::registry::catalog::Catalog::parse(text);
        if let Ok(receipt) = serde_json::from_str::<crate::receipt::InferenceReceipt>(text) {
            let _ = crate::receipt::verify_signed(&receipt);
            let _ = crate::receipt::verify_signed_payload(bytes, &receipt);
        }
    }
}

#[test]
fn the_published_json_documents_cut_or_changed_never_panic_their_readers() {
    let mut c = Campaign::new("JSON documents", 0x150);
    for (i, (name, text)) in JSON_FIXTURES.iter().enumerate() {
        let donor = JSON_FIXTURES[(i + 1) % JSON_FIXTURES.len()].1;
        let value: Value = serde_json::from_str(text).expect("a vendored fixture is JSON");
        let step = 1 + text.len() / 400;
        for n in crate::hostile::lengths(text.len(), step, 16) {
            let what = format!("the first {n} bytes of the {name}");
            c.run(&what, n, || read_as_every_document(&text.as_bytes()[..n]));
        }
        for _ in 0..cases(350) {
            let (m, what) = c.text_mutant(text, donor);
            let what = format!("{name}: {what}");
            c.run(&what, m.len(), || read_as_every_document(m.as_bytes()));
            let (m, what) = json_mutant(&mut c.rng, &value);
            let what = format!("{name}: {what}");
            let m = m.to_string();
            c.run(&what, m.len(), || read_as_every_document(m.as_bytes()));
            let (m, what) = c.mutant(text.as_bytes(), donor.as_bytes());
            let what = format!("{name}: {what}");
            c.run(&what, m.len(), || read_as_every_document(&m));
        }
    }
    c.finish();
}

#[test]
fn a_reference_cut_anywhere_is_refused_and_a_receipt_changed_anywhere_does_not_verify() {
    use crate::receipt::{self, InferenceReceipt};
    let mut c = Campaign::new("reference and receipt", 0x4ec);
    for (name, text) in &JSON_FIXTURES[..5] {
        let whole = text.trim_end().len();
        for p in prefixes(&text.as_bytes()[..whole]) {
            let what = format!("the first {} bytes of the {name}", p.len());
            let r = c.run(&what, p.len(), || {
                (
                    crate::reference::parse(p).is_ok(),
                    crate::snp_reference::parse(p).is_ok(),
                    std::str::from_utf8(p)
                        .is_ok_and(|t| crate::measurements::parse(t, "0.2.3", "a test").is_ok()),
                )
            });
            c.hold(
                r == Some((false, false, false)),
                "a cut JSON document is read",
                &what,
            );
        }
    }

    const PAYLOAD: &[u8] =
        include_bytes!("../tests/receipts/receipt_before_tcb_disclosure.payload.json");
    const RECEIPT: &str =
        include_str!("../tests/receipts/receipt_before_tcb_disclosure.receipt.json");
    let intact: InferenceReceipt = serde_json::from_str(RECEIPT).expect("a receipt");
    assert!(receipt::verify_signed(&intact) && receipt::verify_signed_payload(PAYLOAD, &intact));
    let intact_value = serde_json::to_value(&intact).expect("a receipt is JSON");
    let value: Value = serde_json::from_str(RECEIPT).expect("JSON");
    for _ in 0..cases(1500) {
        let (m, what) = json_mutant(&mut c.rng, &value);
        let size = m.to_string().len();
        let Ok(r) = serde_json::from_value::<InferenceReceipt>(m) else {
            continue;
        };
        let same = serde_json::to_value(&r).ok().as_ref() == Some(&intact_value);
        let what = format!("receipt with {what}");
        let verdict = c.run(&what, size, || {
            (
                receipt::verify_signed(&r),
                receipt::verify_signed_payload(PAYLOAD, &r),
            )
        });
        if let Some((signed, payload)) = verdict {
            c.hold(!signed || same, "a changed receipt verifies", &what);
            c.hold(
                !payload || same,
                "a changed receipt verifies against the payload of the intact one",
                &what,
            );
            if same {
                c.accepted_unchanged();
            }
        }
        let (m, what) = c.mutant(PAYLOAD, RECEIPT.as_bytes());
        let what = format!("receipt payload: {what}");
        let r = c.run(&what, m.len(), || {
            receipt::verify_signed_payload(&m, &intact)
        });
        c.hold(
            r != Some(true) || m == PAYLOAD,
            "a changed payload verifies under the intact receipt",
            &what,
        );
    }
    c.finish();
}

/// Schemas of the shapes `response_format: json_schema` is given.
fn schemas() -> Vec<Value> {
    vec![
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "kind": {"enum": ["a", "b", 3, null, true]},
                "at": {"type": ["number", "null"]}
            },
            "required": ["name", "age", "tags", "kind", "at"],
            "additionalProperties": false
        }),
        json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "number"}, "ok": {"type": "boolean"}},
                "required": ["x", "ok"],
                "additionalProperties": false
            }
        }),
        json!({"anyOf": [{"type": "string"}, {"type": "integer"}, {"const": "x"}]}),
        json!({
            "type": "object",
            "properties": {"next": {"$ref": "#"}, "v": {"type": "null"}},
            "required": ["next", "v"],
            "additionalProperties": false
        }),
    ]
}

#[test]
fn a_json_schema_of_any_shape_never_panics_the_compiler_and_what_it_accepts_is_json() {
    use crate::json_schema::{Accepted, JsonSchema};
    const DOCUMENTS: &[&str] = &[
        r#"{"name":"aé\"b","age":-12,"tags":["x","y"],"kind":3,"at":1.5e-3}"#,
        r#"[{"x":0.25,"ok":true},{"x":-1e9,"ok":false}]"#,
        r#""a string""#,
        r#"{"next":{"next":"x","v":null},"v":null}"#,
        "12345678901234567890",
    ];
    let mut c = Campaign::new("JSON schema", 0x5c4e);
    let seeds = schemas();
    JsonSchema::compile(&seeds[0]).expect("a schema of a supported shape compiles");
    for depth in [1, 2, 16, 64, DEEPEST] {
        let mut s = json!({"type": "string"});
        for _ in 0..depth / 2 {
            s = json!({"type": "array", "items": s});
        }
        let what = format!("a schema of {depth} nested arrays");
        c.run(&what, 0, || {
            let _ = JsonSchema::compile(&s);
            let _ = JsonSchema::compile(&nested(depth));
        });
    }
    for _ in 0..cases(2500) {
        let seed = c.rng.pick(&seeds).clone();
        let (schema, what) = json_mutant(&mut c.rng, &seed);
        let what = format!("schema {schema} ({what})");
        let Some(Ok(compiled)) = c.run(&what, schema.to_string().len(), || {
            JsonSchema::compile(&schema)
        }) else {
            continue;
        };
        for _ in 0..4 {
            let doc = *c.rng.pick(DOCUMENTS);
            let (text, how) = c.text_mutant(doc, DOCUMENTS[0]);
            let what = format!("{what}, on {text:?} ({how})");
            let r = c.run(&what, text.len(), || compiled.accepts(&text));
            // What a schema validates whole is a JSON document: the
            // constraint never lets a model write what no parser reads.
            if r == Some(Accepted::Complete) {
                c.hold(
                    serde_json::from_str::<Value>(&text).is_ok(),
                    "a schema accepts a text that is not JSON",
                    &what,
                );
            }
            // A matcher fed byte by byte says what one fed at once says.
            let piecewise = c.run(&what, text.len(), || {
                let mut m = compiled.matcher();
                let fed = text.bytes().all(|b| m.feed(b));
                (fed, fed && m.complete())
            });
            let at_once = match r {
                Some(Accepted::Complete) => Some((true, true)),
                Some(Accepted::Prefix) => Some((true, false)),
                Some(Accepted::No) => Some((false, false)),
                None => None,
            };
            c.hold(
                piecewise == at_once,
                "a matcher fed byte by byte disagrees with one fed at once",
                &what,
            );
        }
    }
    c.finish();
}

#[test]
fn a_chat_request_of_any_shape_never_panics_its_reader() {
    let mut seeds = vec![
        json!({
            "model": "qwen3:8b",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Say hi."}
            ],
            "max_tokens": 16,
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42,
            "stream": true,
            "stream_options": {"include_usage": true},
            "n": 1,
            "stop": ["\n\n", "END"]
        }),
        json!({
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
            "response_format": {"type": "json_object"}
        }),
    ];
    for s in schemas() {
        seeds.push(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": true, "schema": s}
            },
            "x0ne": {"conversation": "0".repeat(64)}
        }));
    }
    let mut c = Campaign::new("chat request", 0xc4a7);
    for _ in 0..cases(6000) {
        let seed = c.rng.pick(&seeds).clone();
        let (body, what) = json_mutant(&mut c.rng, &seed);
        let size = body.to_string().len();
        c.run(&what, size, || {
            let _ = crate::decoding::Asked::from_body(&body);
            let _ = crate::decoding::response_format(body.get("response_format"));
            let _ = crate::openai::request::parse_chat_request(body.clone());
        });
    }
    c.finish();
}

#[test]
fn the_small_text_formats_cut_or_changed_never_panic_and_parse_to_one_meaning() {
    use crate::relay::{parse_cmd, Cmd};
    use crate::sse::SseReader;
    let mut c = Campaign::new("small text formats", 0x7e47);

    // The relay's control lines, written by the cloud CLI.
    const CONNECT: &str =
        r#"{"v":1,"cmd":"connect","endpoint":"10.0.0.2:8443","boot_deadline_unix":1792666540}"#;
    assert!(matches!(parse_cmd(CONNECT), Cmd::Connect { .. }));
    for p in prefixes(CONNECT.as_bytes()) {
        let p = std::str::from_utf8(p).expect("ASCII");
        let what = format!("the first {} bytes of a connect line", p.len());
        let r = c.run(&what, p.len(), || parse_cmd(p));
        c.hold(
            matches!(r, Some(Cmd::Bad(_))),
            "a cut control line is obeyed",
            &what,
        );
    }
    for _ in 0..cases(6000) {
        let (m, what) = c.text_mutant(CONNECT, r#"{"v":1,"cmd":"stop"}"#);
        c.run(&what, m.len(), || parse_cmd(&m));

        let handle = "0123456789abcdef".repeat(4);
        let (m, what) = c.text_mutant(&handle, CONNECT);
        if let Some(Some(h)) = c.run(&what, m.len(), || crate::context::Handle::parse(&m)) {
            c.hold(
                h.hex() == m,
                "two spellings of one conversation handle",
                &what,
            );
        }
        let (m, what) = c.text_mutant(
            "Thu, 22 Oct 2026 07:35:40 GMT",
            "Sun, 06 Nov 1994 08:49:37 GMT",
        );
        c.run(&what, m.len(), || crate::remote_verify::parse_http_date(&m));
        let rtmr = format!("rtmr2={}", "ab".repeat(48));
        let (m, what) = c.text_mutant(&rtmr, "snp_measurement=00");
        c.run(&what, m.len(), || {
            let _ = crate::verify::parse_expected_measurement(&m);
            let _ = crate::snp::parse_expected_snp_measurement(&m);
        });
    }

    // An SSE stream: the events read do not depend on where the network cut
    // the bytes.
    const STREAM: &str =
        "data: {\"id\":\"a\",\"choices\":[{\"delta\":{\"content\":\"h\u{e9}\"}}]}\n\n\
        : keep-alive\n\n\
        data: {\"id\":\"a\",\"choices\":[]}\n\n\
        data: [DONE]\n\n";
    let read = |pieces: &[&[u8]]| {
        let mut reader = SseReader::new();
        let mut out = Vec::new();
        for p in pieces {
            reader.push(p, &mut out)?;
        }
        Ok::<_, crate::sse::SseError>((out, reader.at_boundary()))
    };
    let whole = read(&[STREAM.as_bytes()]).expect("the stream reads");
    assert_eq!(whole.0.len(), 3);
    for at in 0..=STREAM.len() {
        let (a, b) = STREAM.as_bytes().split_at(at);
        let what = format!("the stream handed in two pieces, cut at byte {at}");
        let r = c.run(&what, STREAM.len(), || read(&[a, b]));
        c.hold(
            r.as_ref() == Some(&Ok(whole.clone())),
            "an SSE stream reads otherwise when cut elsewhere",
            &what,
        );
    }
    for _ in 0..cases(4000) {
        let (m, what) = c.mutant(STREAM.as_bytes(), CONNECT.as_bytes());
        let at_once = c.run(&what, m.len(), || read(&[&m]));
        let cut = c.rng.below(m.len() + 1);
        let (a, b) = m.split_at(cut);
        let in_two = c.run(&what, m.len(), || read(&[a, b]));
        if let (Some(Ok(x)), Some(Ok(y))) = (&at_once, &in_two) {
            c.hold(
                x == y,
                "a mutated SSE stream reads otherwise when cut elsewhere",
                &format!("{what}, cut at byte {cut}"),
            );
        } else {
            c.hold(
                at_once.as_ref().map(Result::is_ok) == in_two.as_ref().map(Result::is_ok),
                "an SSE stream is refused or not by where it is cut",
                &format!("{what}, cut at byte {cut}"),
            );
        }
    }
    c.finish();
}
