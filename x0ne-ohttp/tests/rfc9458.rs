//! Interoperability with RFC 9458 itself. An independent implementation of the
//! RFC's pseudocode (sections 4.3 and 4.4), written here on the `hpke`, `hkdf` and
//! `aes-gcm` crates, first reproduces the RFC's Appendix A byte for byte; then it
//! plays the other side against this crate's client and gateway.

use std::time::{Duration, UNIX_EPOCH};

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes128Gcm, Key, Nonce};
use hpke::aead::AesGcm128;
use hpke::kdf::HkdfSha256;
use hpke::kem::X25519HkdfSha256;
use hpke::{Deserializable, Kem as _, OpModeR, OpModeS, Serializable};
use x0ne_ohttp::client::encapsulate_at;
use x0ne_ohttp::config::{KeyPin, PinnedConfig};
use x0ne_ohttp::gateway::{Gateway, GatewayKey, GatewayPolicy, Kem};
use x0ne_ohttp::message::InnerRequest;

mod common;

type Pk = <X25519HkdfSha256 as hpke::Kem>::PublicKey;
type Sk = <X25519HkdfSha256 as hpke::Kem>::PrivateKey;
type Enc = <X25519HkdfSha256 as hpke::Kem>::EncappedKey;

const SK_R: &str = "3c168975674b2fa8e465970b79c8dcf09f1c741626480bd4c6162fc5b6a98e1a";
const KEY_CONFIG: &str =
    "01002031e1f05a740102115220e9af918f738674aec95f54db6e04eb705aae8e79815500080001000100010003";
const REQUEST: &str = "00034745540568747470730b6578616d706c652e636f6d012f";
const ENC_REQUEST: &str = "010020000100014b28f881333e7c164ffc499ad9796f877f4e1051ee6d31bad19dec96c208b4726374e469135906992e1268c594d2a10c695d858c40a026e7965e7d86b83dd440b2c0185204b4d63525";
const RESPONSE: &str = "0140c8";
const SECRET: &str = "62d87a6ba569ee81014c2641f52bea36";
const RESPONSE_NONCE: &str = "c789e7151fcba46158ca84b04464910d";
const ENC_RESPONSE: &str = "c789e7151fcba46158ca84b04464910d86f9013e404feea014e7be4a441f234f857fbd";

/// The client's clock in these tests, 2026-10-24 00:00:37 UTC, and the date field it
/// sends: the minute, in IMF-fixdate (RFC 9110 section 5.6.7).
const NOW: u64 = 1_792_800_037;
const DATE: &[u8] = b"Sat, 24 Oct 2026 00:00:00 GMT";

/// A known-length field section holding `date` alone (RFC 9292 section 3.6):
/// section length, then name length, name, value length, value; all below 64,
/// so each length is one byte.
fn date_section() -> Vec<u8> {
    let mut field = vec![4u8];
    field.extend_from_slice(b"date");
    field.push(DATE.len() as u8);
    field.extend_from_slice(DATE);
    let mut out = vec![field.len() as u8];
    out.extend_from_slice(&field);
    out
}

fn h(s: &str) -> Vec<u8> {
    hex::decode(s).expect("hex")
}

/// RFC 9458 section 4.3: `"message/bhttp request" || 0 || hdr`.
fn info(hdr: &[u8]) -> Vec<u8> {
    let mut i = b"message/bhttp request".to_vec();
    i.push(0);
    i.extend_from_slice(hdr);
    i
}

/// RFC 9458 section 4.4, gateway side, with the nonce given.
fn seal_response(secret: &[u8], enc: &[u8], nonce: &[u8], response: &[u8]) -> Vec<u8> {
    let (key, iv) = response_keys(secret, enc, nonce);
    let ct = Aes128Gcm::new(&Key::<Aes128Gcm>::from(key))
        .encrypt(&Nonce::from(iv), response)
        .expect("seal");
    let mut out = nonce.to_vec();
    out.extend_from_slice(&ct);
    out
}

fn open_response(secret: &[u8], enc: &[u8], enc_response: &[u8]) -> Vec<u8> {
    let (nonce, ct) = enc_response.split_at(16);
    let (key, iv) = response_keys(secret, enc, nonce);
    Aes128Gcm::new(&Key::<Aes128Gcm>::from(key))
        .decrypt(&Nonce::from(iv), ct)
        .expect("open")
}

fn response_keys(secret: &[u8], enc: &[u8], nonce: &[u8]) -> ([u8; 16], [u8; 12]) {
    let mut salt = enc.to_vec();
    salt.extend_from_slice(nonce);
    let (_, hk) = hkdf::Hkdf::<sha2::Sha256>::extract(Some(&salt), secret);
    let mut key = [0u8; 16];
    let mut iv = [0u8; 12];
    hk.expand(b"key", &mut key).expect("key");
    hk.expand(b"nonce", &mut iv).expect("nonce");
    (key, iv)
}

/// Independent gateway: opens `enc_request` with `sk`, returns (request, secret, enc).
fn rfc_open(sk: &Sk, enc_request: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let hdr = &enc_request[..7];
    assert_eq!(&hdr[1..7], &[0x00, 0x20, 0x00, 0x01, 0x00, 0x01]);
    let enc = &enc_request[7..39];
    let mut ctx = hpke::setup_receiver::<AesGcm128, HkdfSha256, X25519HkdfSha256>(
        &OpModeR::Base,
        sk,
        &Enc::from_bytes(enc).expect("enc"),
        &info(hdr),
    )
    .expect("setup receiver");
    let request = ctx.open(&enc_request[39..], b"").expect("open request");
    let mut secret = [0u8; 16];
    ctx.export(b"message/bhttp response", &mut secret)
        .expect("export");
    (request, secret.to_vec(), enc.to_vec())
}

#[test]
fn the_independent_implementation_reproduces_appendix_a() {
    let sk = Sk::from_bytes(&h(SK_R)).expect("sk");
    let pk = X25519HkdfSha256::sk_to_pk(&sk);
    assert_eq!(&h(KEY_CONFIG)[3..35], pk.to_bytes().as_slice());

    let (request, secret, enc) = rfc_open(&sk, &h(ENC_REQUEST));
    assert_eq!(request, h(REQUEST));
    assert_eq!(secret, h(SECRET));
    let sealed = seal_response(&secret, &enc, &h(RESPONSE_NONCE), &h(RESPONSE));
    assert_eq!(sealed, h(ENC_RESPONSE));
}

#[test]
fn the_rfc_key_configuration_pins_and_parses() {
    let config = h(KEY_CONFIG);
    let p = PinnedConfig::from_config(&config, &KeyPin::of(&config)).expect("RFC config");
    assert_eq!(p.key_id(), 1);
}

#[test]
fn our_client_speaks_rfc_9458_to_an_independent_gateway() {
    let config = h(KEY_CONFIG);
    let pinned = PinnedConfig::from_config(&config, &KeyPin::of(&config)).unwrap();
    let get = InnerRequest {
        method: "GET".into(),
        path: "/".into(),
        headers: Vec::new(),
        body: Vec::new(),
    };
    let at = UNIX_EPOCH + Duration::from_secs(NOW);
    let (enc_request, pending) =
        encapsulate_at(&pinned, &get, "example.com", at).expect("encapsulate");

    let sk = Sk::from_bytes(&h(SK_R)).unwrap();
    let (request, secret, enc) = rfc_open(&sk, &enc_request);
    // Our encoding is the RFC's request, then a header section holding the date,
    // then empty content and trailer and zero padding (RFC 9292 section 3.8).
    let mut expected = h(REQUEST);
    expected.extend_from_slice(&date_section());
    assert_eq!(&request[..expected.len()], expected.as_slice());
    assert!(request[expected.len()..].iter().all(|&b| b == 0));

    let enc_response = seal_response(&secret, &enc, &[0x5a; 16], &h(RESPONSE));
    let response = pending.open(&enc_response).expect("our client opens it");
    assert_eq!(response.status, 200);
    assert!(response.body.is_empty());
}

/// The independent client: seals `request` to the gateway's key, returns what the
/// gateway answered, opened (Binary HTTP).
async fn independent_exchange(gateway: &Gateway, config: &[u8], request: &[u8]) -> Vec<u8> {
    let pk = Pk::from_bytes(&config[3..35]).unwrap();
    let hdr = [config[0], 0x00, 0x20, 0x00, 0x01, 0x00, 0x01];
    let (enc, mut ctx) = hpke::setup_sender::<AesGcm128, HkdfSha256, X25519HkdfSha256>(
        &OpModeS::Base,
        &pk,
        &info(&hdr),
    )
    .expect("setup sender");
    let ct = ctx.seal(request, b"").expect("seal");
    let mut enc_request = hdr.to_vec();
    enc_request.extend_from_slice(&enc.to_bytes());
    enc_request.extend_from_slice(&ct);

    let enc_response = gateway.handle(&enc_request).await.expect("opened");
    let mut secret = [0u8; 16];
    ctx.export(b"message/bhttp response", &mut secret).unwrap();
    open_response(&secret, &enc.to_bytes(), &enc_response)
}

#[tokio::test]
async fn an_independent_client_is_served_by_our_gateway() {
    let key = GatewayKey::generate(5, Kem::X25519).unwrap();
    let config = key.encoded_config().to_vec();
    let log = std::sync::Arc::new(common::TargetLog::default());
    let gateway = Gateway::with_clock(
        key,
        GatewayPolicy {
            authority: "example.com".into(),
            paths: vec!["/".into()],
            max_body: 1 << 16,
            replay_capacity: 16,
            replay_window: common::WINDOW,
        },
        common::echo_target(log.clone()),
        std::sync::Arc::new(|| UNIX_EPOCH + Duration::from_secs(NOW)),
    );

    // The RFC's own request carries no date: this gateway bounds its replay cache by
    // the date (RFC 9458 section 6.5.1), so it refuses it, inside the encapsulation.
    let bhttp = independent_exchange(&gateway, &config, &h(REQUEST)).await;
    // Known-length response (framing 1), status 400 (0x41 0x90).
    assert_eq!(&bhttp[..3], &[0x01, 0x41, 0x90]);
    assert_eq!(log.calls(), 0);

    // The RFC's request with a header section holding the date, unpadded, the
    // content and trailer sections truncated.
    let mut request = h(REQUEST);
    request.extend_from_slice(&date_section());
    let bhttp = independent_exchange(&gateway, &config, &request).await;
    // Known-length response (framing 1), status 200 (0x40 0xc8), then
    // a header section, the content "echo:", and padding.
    assert_eq!(&bhttp[..3], &[0x01, 0x40, 0xc8]);
    assert_eq!(log.calls(), 1);
    let seen = log.requests.lock().unwrap()[0].clone();
    assert_eq!((seen.method.as_str(), seen.path.as_str()), ("GET", "/"));
}
