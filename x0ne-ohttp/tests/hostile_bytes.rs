//! Hostile bytes against what this prototype reads from a relay, a gateway
//! or a directory it does not trust: the `application/ohttp-keys` media
//! type, a pinned key configuration, a node set, the binary HTTP messages
//! inside a sealed request or answer.
//!
//! The statements, the mutations and the bounds are those of
//! `x0ne-sdk/tests/support/hostile.rs`.

#[path = "../../x0ne-sdk/tests/support/hostile.rs"]
mod hostile;

use std::time::{Duration, SystemTime};

use hostile::{bit_flips, cases, prefixes, Campaign};
use x0ne_ohttp::config::{join_keys_media, split_keys_media, KeyPin, PinnedConfig};
use x0ne_ohttp::gateway::{GatewayKey, Kem};
use x0ne_ohttp::message::{self, InnerRequest, InnerResponse};
use x0ne_ohttp::nodes::{NodePolicy, NodeSet, SetDigest};

#[test]
fn a_key_configuration_cut_or_changed_does_not_match_its_pin() {
    let keys = [
        GatewayKey::generate(1, Kem::X25519).expect("a key"),
        GatewayKey::generate(2, Kem::XWing).expect("a key"),
    ];
    let mut c = Campaign::new("OHTTP key configurations", 0x0c0f);
    for (key, other) in [(&keys[0], &keys[1]), (&keys[1], &keys[0])] {
        let config = key.encoded_config();
        let pin = KeyPin::of(config);
        assert!(PinnedConfig::from_config(config, &pin).is_ok());
        let media = key.keys_media();
        assert!(PinnedConfig::from_keys_media(&media, &pin).is_ok());

        for p in prefixes(config) {
            let what = format!("the first {} bytes of a key configuration", p.len());
            let r = c.run(&what, p.len(), || {
                (
                    PinnedConfig::from_config(p, &pin).is_ok(),
                    // Pinned to itself: what the bytes alone are read as.
                    PinnedConfig::from_config(p, &KeyPin::of(p)).is_ok(),
                )
            });
            c.hold(
                r == Some((false, false)),
                "a cut key configuration is read",
                &what,
            );
        }
        for p in prefixes(&media) {
            let what = format!("the first {} bytes of a keys media body", p.len());
            let r = c.run(&what, p.len(), || {
                PinnedConfig::from_keys_media(p, &pin).is_ok()
            });
            c.hold(r == Some(false), "a cut keys media body is read", &what);
        }
        // The pin is a digest of the configuration: no changed one matches.
        for (at, bit, m) in bit_flips(config, 0..config.len()) {
            let what = format!("bit {bit} of key configuration byte {at} flipped");
            let r = c.run(&what, m.len(), || {
                PinnedConfig::from_config(&m, &pin).is_ok()
            });
            c.hold(
                r == Some(false),
                "a changed key configuration matches its pin",
                &what,
            );
        }
        for _ in 0..cases(4000) {
            let (m, what) = c.mutant(config, other.encoded_config());
            let parsed = c.run(&what, m.len(), || {
                (
                    PinnedConfig::from_config(&m, &pin).is_ok(),
                    PinnedConfig::from_config(&m, &KeyPin::of(&m)),
                )
            });
            if let Some((pinned, alone)) = parsed {
                c.hold(
                    !pinned || m == config,
                    "a mutated key configuration matches its pin",
                    &what,
                );
                if let Ok(k) = alone {
                    c.hold(
                        k.encoded() == m,
                        "a configuration is kept as other bytes",
                        &what,
                    );
                }
            }
            let (m, what) = c.mutant(&media, &other.keys_media());
            let what = format!("keys media: {what}");
            let split = c.run(&what, m.len(), || {
                let _ = PinnedConfig::from_keys_media(&m, &pin);
                split_keys_media(&m).map(|v| v.iter().map(|s| s.to_vec()).collect::<Vec<_>>())
            });
            // What splits joins back to the same bytes.
            if let Some(Ok(parts)) = split {
                c.hold(
                    join_keys_media(&parts).as_deref() == Ok(&m[..]),
                    "a keys media body does not read back the same",
                    &what,
                );
            }
        }
        let hex = pin.to_hex();
        assert_eq!(KeyPin::from_hex(&hex), Ok(pin));
        for _ in 0..cases(2000) {
            let (m, what) = c.text_mutant(&hex, "0123456789ABCDEFabcdef");
            if let Some(Ok(k)) = c.run(&what, m.len(), || KeyPin::from_hex(&m)) {
                c.hold(
                    k.to_hex().eq_ignore_ascii_case(&m),
                    "two spellings of one pin",
                    &what,
                );
            }
        }
    }
    c.finish();
}

#[test]
fn a_node_set_cut_or_changed_does_not_match_its_digest() {
    let keys: Vec<GatewayKey> = (1..=4)
        .map(|id| GatewayKey::generate(id, Kem::X25519).expect("a key"))
        .collect();
    let configs: Vec<&[u8]> = keys.iter().map(GatewayKey::encoded_config).collect();
    let set = NodeSet::encode(7, &configs);
    let digest = SetDigest::of(&set);
    let policy = NodePolicy { min_nodes: 2 };
    assert!(NodeSet::verify(&set, &digest, 7, &policy).is_ok());

    let mut c = Campaign::new("OHTTP node set", 0x40de);
    for p in prefixes(&set) {
        let what = format!("the first {} bytes of a node set", p.len());
        let r = c.run(&what, p.len(), || {
            (
                NodeSet::verify(p, &digest, 7, &policy).is_ok(),
                // Under its own digest: what the bytes alone are read as. A
                // set cut after a whole configuration is a smaller set, which
                // only the digest refuses.
                NodeSet::verify(p, &SetDigest::of(p), 7, &policy).map(|s| s.len()),
            )
        });
        if let Some((pinned, alone)) = r {
            c.hold(!pinned, "a cut node set matches its digest", &what);
            c.hold(
                !alone.is_ok_and(|n| n >= configs.len()),
                "a cut node set keeps all its nodes",
                &what,
            );
        }
    }
    for _ in 0..cases(6000) {
        let (m, what) = c.mutant(&set, configs[0]);
        let epoch = *c.rng.pick(&[0, 6, 7, 8, u64::MAX]);
        let r = c.run(&what, m.len(), || {
            (
                NodeSet::verify(&m, &digest, 7, &policy).is_ok(),
                NodeSet::verify(&m, &SetDigest::of(&m), epoch, &policy),
            )
        });
        if let Some((pinned, alone)) = r {
            c.hold(
                !pinned || m == set,
                "a mutated node set matches its digest",
                &what,
            );
            if let Ok(s) = alone {
                let ids: std::collections::BTreeSet<u8> =
                    s.nodes().iter().map(PinnedConfig::key_id).collect();
                c.hold(
                    s.len() >= 2 && ids.len() == s.len(),
                    "a node set under the policy's minimum or with a repeated key ID",
                    &what,
                );
                c.hold(
                    s.epoch() == epoch || Some(s.epoch()) == epoch.checked_add(1),
                    "a node set of another epoch",
                    &what,
                );
            }
        }
    }
    c.finish();
}

#[test]
fn an_inner_message_cut_or_changed_never_panics_and_reads_back_the_same() {
    let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_792_666_540);
    let request = InnerRequest {
        method: "POST".into(),
        path: "/v1/chat/completions".into(),
        headers: vec![("content-type".into(), "application/json".into())],
        body: br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#.to_vec(),
    };
    let response = InnerResponse {
        status: 200,
        headers: vec![("content-type".into(), "application/json".into())],
        body: br#"{"choices":[]}"#.to_vec(),
    };
    let req = message::encode_request(&request, "engine.x0ne.internal", now).expect("encodes");
    let resp = message::encode_response(&response).expect("encodes");
    assert!(message::decode_request(&req).is_ok());
    assert_eq!(message::decode_response(&resp), Ok(response.clone()));

    let mut c = Campaign::new("OHTTP inner messages", 0x1e55);
    // RFC 9292, section 3.8: a message cut at the end of a section is one
    // whose later sections are empty, and the `bhttp` crate reads it so. A
    // message reaches this reader out of an AEAD, which is what refuses a
    // cut; what a cut never does here is read as the whole message.
    let whole_request = message::decode_request(&req).expect("decodes");
    let unpadded = req.len() - req.iter().rev().take_while(|b| **b == 0).count();
    for p in prefixes(&req[..unpadded]) {
        let what = format!("the first {} bytes of an inner request", p.len());
        let r = c.run(&what, p.len(), || message::decode_request(p));
        c.hold(
            r != Some(Ok(whole_request.clone())),
            "a cut inner request reads as the whole one",
            &what,
        );
    }
    let unpadded = resp.len() - resp.iter().rev().take_while(|b| **b == 0).count();
    for p in prefixes(&resp[..unpadded]) {
        let what = format!("the first {} bytes of an inner response", p.len());
        let r = c.run(&what, p.len(), || message::decode_response(p));
        c.hold(
            r != Some(Ok(response.clone())),
            "a cut inner response reads as the whole one",
            &what,
        );
    }
    for _ in 0..cases(6000) {
        let (m, what) = c.mutant(&req, &resp);
        c.run(&what, m.len(), || {
            if let Ok(d) = message::decode_request(&m) {
                let _ = message::check_fields(&d.request.headers);
                for (k, v) in &d.request.headers {
                    let _ = message::check_header(k, v);
                    let _ = message::is_private_token(v);
                }
            }
        });
        let (m, what) = c.mutant(&resp, &req);
        if let Some(Ok(r)) = c.run(&what, m.len(), || message::decode_response(&m)) {
            // Encoding writes field names in lower case: what was encoded
            // once reads back the same from then on.
            let once = message::encode_response(&r).and_then(|b| message::decode_response(&b));
            if let Ok(once) = once {
                let twice =
                    message::encode_response(&once).and_then(|b| message::decode_response(&b));
                c.hold(
                    twice.as_ref() == Ok(&once),
                    "an inner response does not read back the same",
                    &what,
                );
            }
        }
    }
    c.finish();
}
