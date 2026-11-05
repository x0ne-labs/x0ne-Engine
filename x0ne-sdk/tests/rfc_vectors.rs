//! The published vectors, byte for byte: RFC 9180 Appendix A.1 (base mode,
//! DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, AES-128-GCM) and RFC 9458
//! Appendix A (one Oblivious HTTP request and response, Binary HTTP inside).
//! The HPKE and OHTTP code is ours, built on curve25519-dalek, hmac, sha2 and
//! aes-gcm already in the lock: these vectors are what stands for it.

use x0ne_sdk::{bhttp, hpke, ohttp};

fn unhex(s: &str) -> Vec<u8> {
    let s: String = s.split_whitespace().collect();
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn arr32(s: &str) -> [u8; 32] {
    unhex(s).try_into().unwrap()
}

// ── RFC 9180, A.1.1 ─────────────────────────────────────────────────────────

const INFO: &str = "4f6465206f6e2061204772656369616e2055726e";
const IKM_E: &str = "7268600d403fce431561aef583ee1613527cff655c1343f29812e66706df3234";
const PK_EM: &str = "37fda3567bdbd628e88668c3c8d7e97d1d1253b6d4ea6d44c150f741f1bf4431";
const SK_EM: &str = "52c4a758a802cd8b936eceea314432798d5baf2d7e9235dc084ab1b9cfa2f736";
const IKM_R: &str = "6db9df30aa07dd42ee5e8181afdb977e538f5e1fec8a06223f33f7013e525037";
const PK_RM: &str = "3948cfe0ad1ddb695d780e59077195da6c56506b027329794ab02bca80815c4d";
const SK_RM: &str = "4612c550263fc8ad58375df3f557aac531d26850903e55a9f23f21d8534e8ac8";
const PT: &str = "4265617574792069732074727574682c20747275746820626561757479";

/// (sequence number, aad, ct) of A.1.1.1.
const ENCRYPTIONS: &[(u64, &str, &str)] = &[
    (
        0,
        "436f756e742d30",
        "f938558b5d72f1a23810b4be2ab4f84331acc02fc97babc53a52ae8218a355a96d8770ac83d07bea87e13c512a",
    ),
    (
        1,
        "436f756e742d31",
        "af2d7e9ac9ae7e270f46ba1f975be53c09f8d875bdc8535458c2494e8a6eab251c03d0c22a56b8ca42c2063b84",
    ),
    (
        2,
        "436f756e742d32",
        "498dfcabd92e8acedc281e85af1cb4e3e31c7dc394a1ca20e173cb72516491588d96a19ad4a683518973dcc180",
    ),
    (
        4,
        "436f756e742d34",
        "583bd32bc67a5994bb8ceaca813d369bca7b2a42408cddef5e22f880b631215a09fc0012bc69fccaa251c0246d",
    ),
    (
        255,
        "436f756e742d323535",
        "7175db9717964058640a3a11fb9007941a5d1757fda1a6935c805c21af32505bf106deefec4a49ac38d71c9e0a",
    ),
    (
        256,
        "436f756e742d323536",
        "957f9800542b0b8891badb026d79cc54597cb2d225b54c00c5238c25d05c30e3fbeda97d2e0e1aba483a2df9f2",
    ),
];

/// (exporter_context, exported_value), L = 32, of A.1.1.2.
const EXPORTS: &[(&str, &str)] = &[
    (
        "",
        "3853fe2b4035195a573ffc53856e77058e15d9ea064de3e59f4961d0095250ee",
    ),
    (
        "00",
        "2e8f0b54673c7029649d4eb9d5e33bf1872cf76d623ff164ac185da9e88c21a5",
    ),
    (
        "54657374436f6e74657874",
        "e9e43065102c3836401bed8c3c3c75ae46be1639869391d62c61f1ec7af54931",
    ),
];

#[test]
fn derive_key_pair_gives_the_rfc_keys() {
    // No public API reads a private key back: the derived keys are checked
    // by the public keys of the stated ones, and by opening the vector.
    let (sk_e, pk_e) = hpke::derive_key_pair(&unhex(IKM_E));
    assert_eq!(pk_e, arr32(PK_EM));
    assert_eq!(sk_e.public_key(), arr32(PK_EM));
    assert_eq!(
        hpke::PrivateKey::from_bytes(arr32(SK_EM)).public_key(),
        arr32(PK_EM)
    );
    let (sk_r, pk_r) = hpke::derive_key_pair(&unhex(IKM_R));
    assert_eq!(pk_r, arr32(PK_RM));
    assert_eq!(sk_r.public_key(), arr32(PK_RM));
    let (enc, mut sender) = hpke::setup_base_s(&pk_r, &unhex(INFO), &unhex(IKM_E)).unwrap();
    let ct = sender.seal(b"aad", b"vector").unwrap();
    let mut receiver = hpke::setup_base_r(&enc, &sk_r, &unhex(INFO)).unwrap();
    assert_eq!(receiver.open(b"aad", &ct).unwrap(), b"vector");
    assert_eq!(
        hpke::PrivateKey::from_bytes(arr32(SK_RM)).public_key(),
        arr32(PK_RM)
    );
}

#[test]
fn sender_and_receiver_match_every_encryption_and_export_of_a_1_1() {
    let (enc, mut sender) = hpke::setup_base_s(&arr32(PK_RM), &unhex(INFO), &unhex(IKM_E)).unwrap();
    assert_eq!(enc, arr32(PK_EM), "enc is the ephemeral public key");
    let sk_r = hpke::PrivateKey::from_bytes(arr32(SK_RM));
    let mut receiver = hpke::setup_base_r(&enc, &sk_r, &unhex(INFO)).unwrap();
    // The vectors skip sequence numbers 3 and 5..254: both ends step over
    // them with empty messages, the receiver opening what the sender sealed.
    let mut done = 0u64;
    for (seq, aad, ct) in ENCRYPTIONS {
        while done < *seq {
            let filler = sender.seal(b"", b"").unwrap();
            receiver.open(b"", &filler).unwrap();
            done += 1;
        }
        let got = sender.seal(&unhex(aad), &unhex(PT)).unwrap();
        assert_eq!(got, unhex(ct), "seal, sequence number {seq}");
        assert_eq!(
            receiver.open(&unhex(aad), &got).unwrap(),
            unhex(PT),
            "open, sequence number {seq}"
        );
        done += 1;
    }
    for (context, value) in EXPORTS {
        assert_eq!(*sender.export(&unhex(context), 32).unwrap(), unhex(value));
        assert_eq!(*receiver.export(&unhex(context), 32).unwrap(), unhex(value));
    }
}

/// A message replayed to the same receiver context does not open: the
/// nonce moved on.
#[test]
fn a_message_opened_once_does_not_open_twice_in_one_context() {
    let (enc, mut sender) = hpke::setup_base_s(&arr32(PK_RM), &unhex(INFO), &unhex(IKM_E)).unwrap();
    let sk_r = hpke::PrivateKey::from_bytes(arr32(SK_RM));
    let mut receiver = hpke::setup_base_r(&enc, &sk_r, &unhex(INFO)).unwrap();
    let ct = sender.seal(b"", b"once").unwrap();
    assert_eq!(receiver.open(b"", &ct).unwrap(), b"once");
    assert_eq!(receiver.open(b"", &ct), Err(hpke::HpkeError::Open));
}

#[test]
fn a_flipped_byte_or_another_aad_does_not_open() {
    let (enc, mut sender) = hpke::setup_base_s(&arr32(PK_RM), &unhex(INFO), &unhex(IKM_E)).unwrap();
    let ct = sender.seal(b"aad", b"secret").unwrap();
    let sk_r = hpke::PrivateKey::from_bytes(arr32(SK_RM));
    for i in 0..ct.len() {
        let mut bad = ct.clone();
        bad[i] ^= 0x01;
        let mut r = hpke::setup_base_r(&enc, &sk_r, &unhex(INFO)).unwrap();
        assert_eq!(r.open(b"aad", &bad), Err(hpke::HpkeError::Open), "byte {i}");
    }
    let mut r = hpke::setup_base_r(&enc, &sk_r, &unhex(INFO)).unwrap();
    assert_eq!(r.open(b"other", &ct), Err(hpke::HpkeError::Open));
    let mut other_info = hpke::setup_base_r(&enc, &sk_r, b"other info").unwrap();
    assert_eq!(other_info.open(b"aad", &ct), Err(hpke::HpkeError::Open));
}

/// RFC 9180 §7.1.4: an all-zero X25519 output aborts. A low-order point as
/// `enc` (or as a recipient key) would make every sender share the same
/// secret with anyone.
#[test]
fn a_low_order_point_is_refused_on_both_sides() {
    let sk = hpke::PrivateKey::from_bytes(arr32(SK_RM));
    let mut one = [0u8; 32];
    one[0] = 1; // u = 1, order 4
    for low in [[0u8; 32], one] {
        assert_eq!(
            hpke::setup_base_r(&low, &sk, b"").err(),
            Some(hpke::HpkeError::ZeroSharedSecret)
        );
        assert_eq!(
            hpke::setup_base_s(&low, b"", &unhex(IKM_E)).err(),
            Some(hpke::HpkeError::ZeroSharedSecret)
        );
    }
}

// ── RFC 9458, Appendix A ────────────────────────────────────────────────────

const GATEWAY_SK: &str = "3c168975674b2fa8e465970b79c8dcf09f1c741626480bd4c6162fc5b6a98e1a";
const KEY_CONFIG: &str =
    "01002031e1f05a740102115220e9af918f738674aec95f54db6e04eb705aae8e79815500080001000100010003";
const BHTTP_REQUEST: &str = "00034745540568747470730b6578616d706c652e636f6d012f";
const EPHEMERAL_SK: &str = "bc51d5e930bda26589890ac7032f70ad12e4ecb37abb1b65b1256c9c48999c73";
const ENCAPSULATED_REQUEST: &str =
    "010020000100014b28f881333e7c164ffc499ad9796f877f4e1051ee6d31bad1\
    9dec96c208b4726374e469135906992e1268c594d2a10c695d858c40a026e796\
    5e7d86b83dd440b2c0185204b4d63525";
const BHTTP_RESPONSE: &str = "0140c8";
const RESPONSE_NONCE: &str = "c789e7151fcba46158ca84b04464910d";
const ENCAPSULATED_RESPONSE: &str =
    "c789e7151fcba46158ca84b04464910d86f9013e404feea014e7be4a441f234f857fbd";

/// The example's key configuration lists two suites (AES-128-GCM and
/// ChaCha20-Poly1305); ours lists one. Parsing it keeps the key and the id.
#[test]
fn the_rfc_9458_key_configuration_parses_to_its_key() {
    let cfg = ohttp::KeyConfig::parse_any_suite(&unhex(KEY_CONFIG)).unwrap();
    assert_eq!(cfg.key_id, 1);
    assert_eq!(
        cfg.public_key,
        hpke::PrivateKey::from_bytes(arr32(GATEWAY_SK)).public_key()
    );
    // Strict parsing: exactly our one suite. The example's two are refused.
    assert!(ohttp::KeyConfig::parse(&unhex(KEY_CONFIG)).is_err());
    let ours = ohttp::KeyConfig {
        key_id: 1,
        public_key: cfg.public_key,
    };
    assert_eq!(ohttp::KeyConfig::parse(&ours.encode()).unwrap(), ours);
}

#[test]
fn the_rfc_9458_exchange_byte_for_byte() {
    let cfg = ohttp::KeyConfig::parse_any_suite(&unhex(KEY_CONFIG)).unwrap();
    let (request, client) = ohttp::encapsulate_request_with_ephemeral(
        &cfg,
        &unhex(BHTTP_REQUEST),
        &hpke::PrivateKey::from_bytes(arr32(EPHEMERAL_SK)),
    )
    .unwrap();
    assert_eq!(request, unhex(ENCAPSULATED_REQUEST));

    let gateway_sk = hpke::PrivateKey::from_bytes(arr32(GATEWAY_SK));
    let (inner, server) = ohttp::decapsulate_request(&gateway_sk, 1, &request).unwrap();
    assert_eq!(*inner, unhex(BHTTP_REQUEST));
    let nonce: [u8; 16] = unhex(RESPONSE_NONCE).try_into().unwrap();
    let response = server.encapsulate(&unhex(BHTTP_RESPONSE), &nonce).unwrap();
    assert_eq!(response, unhex(ENCAPSULATED_RESPONSE));
    assert_eq!(
        *client.decapsulate(&response).unwrap(),
        unhex(BHTTP_RESPONSE)
    );
}

#[test]
fn the_rfc_9458_binary_http_messages_decode() {
    let req = bhttp::Request::decode(&unhex(BHTTP_REQUEST)).unwrap();
    assert_eq!(req.method, "GET");
    assert_eq!(req.scheme, "https");
    assert_eq!(req.authority, "example.com");
    assert_eq!(req.path, "/");
    assert!(req.headers.is_empty() && req.content.is_empty());
    let resp = bhttp::Response::decode(&unhex(BHTTP_RESPONSE)).unwrap();
    assert_eq!(resp.status, 200);
    assert!(resp.headers.is_empty() && resp.content.is_empty());
}

#[test]
fn a_request_for_another_key_id_or_suite_is_refused_before_any_decryption() {
    let cfg = ohttp::KeyConfig::parse_any_suite(&unhex(KEY_CONFIG)).unwrap();
    let gateway_sk = hpke::PrivateKey::from_bytes(arr32(GATEWAY_SK));
    let request = unhex(ENCAPSULATED_REQUEST);
    assert!(matches!(
        ohttp::decapsulate_request(&gateway_sk, 2, &request),
        Err(ohttp::OhttpError::KeyId(1))
    ));
    let mut other_aead = request.clone();
    other_aead[6] = 3; // ChaCha20-Poly1305
    assert!(matches!(
        ohttp::decapsulate_request(&gateway_sk, 1, &other_aead),
        Err(ohttp::OhttpError::Suite { .. })
    ));
    let mut tampered = request.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 1;
    assert!(matches!(
        ohttp::decapsulate_request(&gateway_sk, 1, &tampered),
        Err(ohttp::OhttpError::Hpke(hpke::HpkeError::Open))
    ));
    let _ = cfg;
}

#[test]
fn a_response_flipped_anywhere_or_cut_short_does_not_open() {
    let cfg = ohttp::KeyConfig::parse_any_suite(&unhex(KEY_CONFIG)).unwrap();
    let (_, client) = ohttp::encapsulate_request_with_ephemeral(
        &cfg,
        &unhex(BHTTP_REQUEST),
        &hpke::PrivateKey::from_bytes(arr32(EPHEMERAL_SK)),
    )
    .unwrap();
    let response = unhex(ENCAPSULATED_RESPONSE);
    for i in 0..response.len() {
        let mut bad = response.clone();
        bad[i] ^= 0x80;
        assert!(client.decapsulate(&bad).is_err(), "byte {i}");
    }
    for n in 0..response.len() {
        assert!(client.decapsulate(&response[..n]).is_err(), "cut at {n}");
    }
}

/// SECURITY: the caller brings the ephemeral key's input keying material.
/// RFC 9180 §7.1.3 wants at least `Nsk` bytes of it; fewer bytes would be a
/// weaker key than the suite promises, and the SDK must not seal with one.
#[test]
fn an_ephemeral_input_shorter_than_the_private_key_is_refused() {
    let pk = arr32(PK_RM);
    for short in [0usize, 1, 16, 31] {
        assert!(
            hpke::setup_base_s(&pk, b"", &vec![7u8; short]).is_err(),
            "{short} bytes of ikm"
        );
    }
    assert!(hpke::setup_base_s(&pk, b"", &[7u8; 32]).is_ok());
    let cfg = ohttp::KeyConfig {
        key_id: 1,
        public_key: pk,
    };
    assert!(ohttp::encapsulate_request(&cfg, b"\x00", &[7u8; 31]).is_err());
    assert!(ohttp::encapsulate_request(&cfg, b"\x00", &[7u8; 32]).is_ok());
}

struct Held(Vec<u8>);

impl hpke::SecretScalar for Held {
    fn expose(&self) -> &[u8] {
        &self.0
    }
}

/// A key moved into memory the caller holds (the TD's `memfd_secret` or
/// `mlock`ed memory) is the
/// same key: its public key, what it opens. Memory that gives back another
/// length fails the operation, without a panic.
#[test]
fn a_key_held_by_the_caller_is_the_same_key() {
    let (own, pk) = hpke::derive_key_pair(&unhex(IKM_R));
    let held = hpke::derive_key_pair(&unhex(IKM_R))
        .0
        .held_by(|s| Ok::<_, ()>(Held(s.to_vec())))
        .unwrap();
    assert_eq!(held.public_key(), pk);
    let (enc, mut sender) = hpke::setup_base_s(&pk, &unhex(INFO), &unhex(IKM_E)).unwrap();
    let ct = sender.seal(b"aad", b"question").unwrap();
    for key in [&own, &held] {
        let mut receiver = hpke::setup_base_r(&enc, key, &unhex(INFO)).unwrap();
        assert_eq!(receiver.open(b"aad", &ct).unwrap(), b"question");
    }
    let short = hpke::derive_key_pair(&unhex(IKM_R))
        .0
        .held_by(|s| Ok::<_, ()>(Held(s[..31].to_vec())))
        .unwrap();
    assert!(matches!(
        hpke::setup_base_r(&enc, &short, &unhex(INFO)),
        Err(hpke::HpkeError::KeyLength(31))
    ));
}
