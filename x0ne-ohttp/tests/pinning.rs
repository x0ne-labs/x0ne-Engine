//! Key consistency: a client encapsulates only to the configuration it was told to
//! expect. Serving one user a key of their own is how an operator would single them
//! out (Trail of Bits, TOB-WAPI-3, on Meta's Private Processing).

use x0ne_ohttp::config::{ConfigError, KeyPin, PinnedConfig};
use x0ne_ohttp::gateway::{GatewayKey, Kem};

fn keys_media(configs: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for c in configs {
        out.extend_from_slice(&(c.len() as u16).to_be_bytes());
        out.extend_from_slice(c);
    }
    out
}

#[test]
fn the_pinned_configuration_is_accepted() {
    let key = GatewayKey::generate(3, Kem::X25519).unwrap();
    let pin = KeyPin::of(key.encoded_config());
    let p = PinnedConfig::from_keys_media(&key.keys_media(), &pin).expect("pinned");
    assert_eq!(p.key_id(), 3);
    assert_eq!(p.encoded(), key.encoded_config());
}

#[test]
fn a_key_served_to_one_user_only_is_refused() {
    let everyone = GatewayKey::generate(3, Kem::X25519).unwrap();
    let targeted = GatewayKey::generate(3, Kem::X25519).unwrap();
    let pin = KeyPin::of(everyone.encoded_config());
    let err = PinnedConfig::from_keys_media(&targeted.keys_media(), &pin).unwrap_err();
    assert_eq!(err, ConfigError::Mismatch);
}

#[test]
fn the_same_key_under_another_key_id_is_refused() {
    let key = GatewayKey::generate(3, Kem::X25519).unwrap();
    let pin = KeyPin::of(key.encoded_config());
    let mut relabelled = key.encoded_config().to_vec();
    relabelled[0] = 4;
    let err = PinnedConfig::from_config(&relabelled, &pin).unwrap_err();
    assert_eq!(err, ConfigError::Mismatch);
}

#[test]
fn a_collection_with_more_than_one_configuration_is_refused() {
    let a = GatewayKey::generate(1, Kem::X25519).unwrap();
    let b = GatewayKey::generate(2, Kem::X25519).unwrap();
    let pin = KeyPin::of(a.encoded_config());
    let both = keys_media(&[a.encoded_config(), b.encoded_config()]);
    assert_eq!(
        PinnedConfig::from_keys_media(&both, &pin).unwrap_err(),
        ConfigError::NotExactlyOne(2)
    );
}

#[test]
fn badly_encoded_collections_are_discarded_whole() {
    let key = GatewayKey::generate(1, Kem::X25519).unwrap();
    let pin = KeyPin::of(key.encoded_config());
    let good = key.keys_media();

    let mut trailing = good.clone();
    trailing.push(0);
    assert_eq!(
        PinnedConfig::from_keys_media(&trailing, &pin).unwrap_err(),
        ConfigError::Malformed
    );

    let truncated = &good[..good.len() - 1];
    assert_eq!(
        PinnedConfig::from_keys_media(truncated, &pin).unwrap_err(),
        ConfigError::Malformed
    );

    assert_eq!(
        PinnedConfig::from_keys_media(&[], &pin).unwrap_err(),
        ConfigError::NotExactlyOne(0)
    );
}

#[test]
fn a_configuration_this_client_cannot_use_is_refused_even_if_pinned() {
    // KEM 0x0011 (P-384) is not offered by this client.
    let mut config = vec![1u8, 0x00, 0x11];
    config.extend_from_slice(&[4u8; 97]);
    config.extend_from_slice(&[0, 4, 0, 1, 0, 1]);
    let pin = KeyPin::of(&config);
    assert_eq!(
        PinnedConfig::from_config(&config, &pin).unwrap_err(),
        ConfigError::Unsupported
    );
}

#[test]
fn pins_round_trip_through_hex_and_reject_garbage() {
    let key = GatewayKey::generate(1, Kem::XWing).unwrap();
    let pin = KeyPin::of(key.encoded_config());
    let hex = pin.to_hex();
    assert_eq!(hex.len(), 64);
    assert_eq!(KeyPin::from_hex(&hex).unwrap(), pin);
    assert!(KeyPin::from_hex(&hex[..63]).is_err());
    assert!(KeyPin::from_hex(&format!("{}zz", &hex[..62])).is_err());
}

/// The configuration's suite list with `suites` in place of its own.
fn with_suites(config: &[u8], suites: &[[u8; 4]]) -> Vec<u8> {
    let n = config.len();
    assert_eq!(
        &config[n - 6..],
        &[0, 4, 0, 1, 0, 1],
        "one suite, AES-128-GCM"
    );
    let mut out = config[..n - 6].to_vec();
    out.extend_from_slice(&((suites.len() * 4) as u16).to_be_bytes());
    for s in suites {
        out.extend_from_slice(s);
    }
    out
}

const AES_128_GCM: [u8; 4] = [0, 1, 0, 1];
const CHACHA20_POLY1305: [u8; 4] = [0, 1, 0, 3];

/// One suite for every client: a configuration that
/// does not offer HKDF-SHA256 with AES-128-GCM is refused.
#[test]
fn a_configuration_without_aes_128_gcm_is_refused() {
    let key = GatewayKey::generate(1, Kem::X25519).unwrap();
    let cfg = with_suites(key.encoded_config(), &[CHACHA20_POLY1305]);
    assert_eq!(
        PinnedConfig::from_config(&cfg, &KeyPin::of(&cfg)).unwrap_err(),
        ConfigError::Unsupported
    );
}

/// The `ohttp` crate encapsulates with the first suite listed; the order of a
/// configuration must not decide what every client sends in the clear.
#[test]
fn the_client_encapsulates_with_aes_128_gcm_whatever_the_order_of_the_suites() {
    let key = GatewayKey::generate(1, Kem::X25519).unwrap();
    let cfg = with_suites(key.encoded_config(), &[CHACHA20_POLY1305, AES_128_GCM]);
    let pinned = PinnedConfig::from_config(&cfg, &KeyPin::of(&cfg)).expect("AES-128-GCM offered");
    let req = x0ne_ohttp::message::InnerRequest {
        method: "POST".into(),
        path: "/v1/chat/completions".into(),
        headers: Vec::new(),
        body: b"{}".to_vec(),
    };
    let (enc, _) = x0ne_ohttp::client::encapsulate(&pinned, &req, "a").unwrap();
    assert_eq!(
        &enc[1..7],
        &[0x00, 0x20, 0x00, 0x01, 0x00, 0x01],
        "KEM, KDF, AEAD"
    );
}
