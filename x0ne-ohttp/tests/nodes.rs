//! The node set: every client draws its TD itself, uniformly, from one published
//! list. The operator does not route a request to a TD; the client's draw does.

use x0ne_ohttp::gateway::{GatewayKey, Kem};
use x0ne_ohttp::nodes::{uniform_index, NodePolicy, NodeSet, NodeSetError, SetDigest};

fn configs(n: u8) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| {
            GatewayKey::generate(i, Kem::X25519)
                .unwrap()
                .encoded_config()
                .to_vec()
        })
        .collect()
}

fn policy(min_nodes: usize) -> NodePolicy {
    NodePolicy { min_nodes }
}

/// Draws of the uniformity test, and its limit: with 7 degrees of freedom,
/// P(chi2 > 50) < 2e-8 for a uniform draw.
const DRAWS: usize = 400_000;
const CHI2_LIMIT: f64 = 50.0;

#[test]
fn the_published_set_is_accepted_when_its_digest_matches() {
    let c = configs(4);
    let encoded = NodeSet::encode(42, &c);
    let set = NodeSet::verify(&encoded, &SetDigest::of(&encoded), 42, &policy(4)).unwrap();
    assert_eq!(set.epoch(), 42);
    assert_eq!(set.len(), 4);
}

#[test]
fn a_set_made_for_one_user_is_refused() {
    let published = NodeSet::encode(42, &configs(4));
    // Same epoch, one node swapped for one the attacker holds.
    let mut c = configs(4);
    c[2] = GatewayKey::generate(2, Kem::X25519)
        .unwrap()
        .encoded_config()
        .to_vec();
    let targeted = NodeSet::encode(42, &c);
    let err = NodeSet::verify(&targeted, &SetDigest::of(&published), 42, &policy(4)).unwrap_err();
    assert_eq!(err, NodeSetError::Mismatch);
}

#[test]
fn a_set_too_small_to_hide_anyone_is_refused() {
    let encoded = NodeSet::encode(1, &configs(1));
    let err = NodeSet::verify(&encoded, &SetDigest::of(&encoded), 1, &policy(4)).unwrap_err();
    assert_eq!(err, NodeSetError::TooFew { got: 1, min: 4 });
}

#[test]
fn one_node_listed_twice_is_refused() {
    let mut c = configs(4);
    c.push(c[0].clone());
    let encoded = NodeSet::encode(1, &c);
    let err = NodeSet::verify(&encoded, &SetDigest::of(&encoded), 1, &policy(4)).unwrap_err();
    assert!(
        matches!(
            err,
            NodeSetError::DuplicateKeyId(0) | NodeSetError::DuplicateKey
        ),
        "{err:?}"
    );
}

#[test]
fn one_key_behind_two_key_ids_is_refused() {
    let mut c = configs(4);
    let mut alias = c[1].clone();
    alias[0] = 9;
    c.push(alias);
    let encoded = NodeSet::encode(1, &c);
    let err = NodeSet::verify(&encoded, &SetDigest::of(&encoded), 1, &policy(4)).unwrap_err();
    assert_eq!(err, NodeSetError::DuplicateKey);
}

#[test]
fn a_malformed_set_is_refused_whole() {
    let mut encoded = NodeSet::encode(1, &configs(4));
    encoded.push(0);
    let err = NodeSet::verify(&encoded, &SetDigest::of(&encoded), 1, &policy(4)).unwrap_err();
    assert_eq!(err, NodeSetError::Malformed);
    let short = NodeSet::encode(1, &configs(4));
    let short = &short[..short.len() - 3];
    let err = NodeSet::verify(short, &SetDigest::of(short), 1, &policy(4)).unwrap_err();
    assert_eq!(err, NodeSetError::Malformed);
}

#[test]
fn the_draw_is_uniform_over_the_nodes() {
    let encoded = NodeSet::encode(7, &configs(8));
    let set = NodeSet::verify(&encoded, &SetDigest::of(&encoded), 7, &policy(8)).unwrap();
    let mut counts = [0usize; 256];
    for _ in 0..DRAWS {
        counts[set.choose().unwrap().key_id() as usize] += 1;
    }
    let expected = DRAWS as f64 / 8.0;
    let chi2: f64 = counts[..8]
        .iter()
        .map(|&o| (o as f64 - expected).powi(2) / expected)
        .sum();
    assert!(counts[8..].iter().all(|&c| c == 0));
    assert!(
        chi2 < CHI2_LIMIT,
        "chi2 = {chi2}, counts = {:?}",
        &counts[..8]
    );
}

/// What the test above can see. One node drawn 5 %
/// more often than its share moves the statistic by its non-centrality; the test
/// must then fail far more often than not, so that is held at twice the limit.
#[test]
fn the_uniformity_test_sees_one_node_favoured_by_five_percent() {
    let expected = DRAWS as f64 / 8.0;
    let favoured = expected * 1.05;
    let others = (DRAWS as f64 - favoured) / 7.0;
    let shift =
        (favoured - expected).powi(2) / expected + 7.0 * (others - expected).powi(2) / expected;
    assert!(
        shift >= 2.0 * CHI2_LIMIT,
        "a 5 % bias moves chi2 by {shift:.1} over {DRAWS} draws, limit {CHI2_LIMIT}"
    );
}

#[test]
fn the_index_draw_rejects_the_biased_tail() {
    // n = 3: 2^32 mod 3 = 1, so u32::MAX is the one value that must be redrawn.
    let mut values = vec![u32::MAX, u32::MAX, 5].into_iter();
    let i = uniform_index(3, || Ok::<u32, ()>(values.next().unwrap())).unwrap();
    assert_eq!(i, 5 % 3);
    assert!(values.next().is_none(), "the tail was not redrawn");

    // n = 6: values from 2^32 - 4 up are in the tail.
    let mut values = vec![u32::MAX - 3, 11].into_iter();
    assert_eq!(
        uniform_index(6, || Ok::<u32, ()>(values.next().unwrap())).unwrap(),
        11 % 6
    );
    let mut values = vec![u32::MAX - 4].into_iter();
    assert_eq!(
        uniform_index(6, || Ok::<u32, ()>(values.next().unwrap())).unwrap(),
        ((u32::MAX - 4) % 6) as usize
    );
}

#[test]
fn the_chosen_node_is_a_pinned_configuration_of_the_set() {
    let c = configs(4);
    let encoded = NodeSet::encode(3, &c);
    let set = NodeSet::verify(&encoded, &SetDigest::of(&encoded), 3, &policy(4)).unwrap();
    for _ in 0..32 {
        let node = set.choose().unwrap();
        assert!(c.iter().any(|x| x.as_slice() == node.encoded()));
    }
}

/// A set the log once published, served again to one user: the client
/// accepts the current epoch and the next, nothing older or later.
#[test]
fn a_set_of_a_retired_or_later_epoch_is_refused() {
    let c = configs(4);
    for (epoch, accepted) in [(0, false), (4, false), (5, true), (6, true), (7, false)] {
        let encoded = NodeSet::encode(epoch, &c);
        let r = NodeSet::verify(&encoded, &SetDigest::of(&encoded), 5, &policy(4));
        match (accepted, r) {
            (true, Ok(set)) => assert_eq!(set.epoch(), epoch),
            (false, Err(e)) => assert_eq!(
                e,
                NodeSetError::Epoch {
                    got: epoch,
                    current: 5
                }
            ),
            (_, r) => panic!("epoch {epoch} with current 5: {r:?}"),
        }
    }
    let last = NodeSet::encode(u64::MAX, &c);
    NodeSet::verify(&last, &SetDigest::of(&last), u64::MAX, &policy(4)).expect("no overflow");
}
