//! What a CLIENT checks, with nothing taken on the server's word.
//!
//! `/attestation` publishes a verdict AND the evidence behind it. Until
//! 2026-09-18 `x0ne-infer-chat` (later `x0ne-infer-ssh-chat`, removed on
//! 2026-11-04) printed the former and never opened the latter: its
//! "verified" line covered the receipt's signature under the receipt's OWN
//! key, and nothing tied that key to any hardware. The server's
//! verdict was believed — by the one program whose purpose is not to.
//!
//! This is also the only place where pinning an image means something. The
//! server's `--expect-measurement` protects an honest operator from a
//! tampered image; a dishonest operator simply does not pass the flag. The
//! client holds the expectation, reads the registers out of a quote IT
//! verified, and refuses by itself.
//!
//! Checked here, offline: the DCAP chain to the Intel root the client pins;
//! that the boot quote answers the PUBLISHED session (nonce, key) — so it is
//! this server's quote, not a genuine one from elsewhere; that a SECOND quote
//! answers the challenge THIS client drew ([`ClientChallenge`]) under the
//! same key — so the key is in hardware now, not in a document recorded
//! once; that both quotes report the same image; that the receipt is signed
//! by that same key; and the expected measurements.
//!
//! Until 2026-10-13 the only nonce was the server's: a quote taken once at
//! boot verified for every client, forever. The client now sends its own
//! and refuses a document that does not answer it.
//!
//! NOT checked here, and said in [`LIMITATIONS`] rather than implied: the
//! live TCB status (needs Intel PCS, a network call this function does not
//! make) and the whole GPU half (the per-GPU tokens and the local certificate
//! serials are not published by `/attestation` yet; the NRAS token answers
//! the boot nonce, not the client's challenge).

use base64::Engine as _;
use x0ne_confidential::model::Measurement;
use x0ne_confidential::policy::ExpectedMeasurement;

use crate::pcs::TrustAnchor;

pub const LIMITATIONS: &[&str] = &[
    "client side: TCB status and QE identity not re-checked against live Intel PCS",
    "client side: the GPU half is the server's verdict, not re-verified here",
    "client side: the challenge covers the TDX quote; the NRAS token answers the boot nonce",
];

/// The challenge a client issues BEFORE anything else: 32 bytes it drew
/// itself, sent as `GET /attestation?nonce=<64 hex>`. The quote that comes
/// back must carry `SHA-512(context ‖ nonce ‖ session key)` in its
/// `report_data`, or it is a replay: a genuine quote taken at some other
/// time, for some other client, or at boot with a nonce the server chose.
///
/// 32 bytes: TDX `REPORTDATA` is 64 bytes and receives a SHA-512 over the
/// challenge (Intel TDX Module ABI, `TDG.MR.REPORT`), and NRAS takes a
/// 32-byte nonce (`nras_client`): 256 bits is the full strength of either
/// binding, and a guess has 2^-256 odds. Drawn from `/dev/urandom`, the
/// kernel CSPRNG (`getrandom(2)` semantics since Linux 5.6), the same
/// source as the server's own session nonce and the decision salt.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ClientChallenge([u8; 32]);

impl ClientChallenge {
    /// A fresh challenge from the kernel CSPRNG.
    pub fn fresh() -> Result<Self, String> {
        use std::io::Read as _;
        let mut b = [0u8; 32];
        std::fs::File::open("/dev/urandom")
            .and_then(|mut f| f.read_exact(&mut b))
            .map_err(|e| format!("/dev/urandom: {e}"))?;
        Ok(Self(b))
    }
    /// A challenge the caller drew (tests, or a nonce recorded at ask time).
    pub const fn from_bytes(nonce: [u8; 32]) -> Self {
        Self(nonce)
    }
    /// The 64 hex characters the client sent, read back.
    pub fn from_hex(hex: &str) -> Result<Self, String> {
        if hex.len() != 64 || !hex.is_ascii() {
            return Err("challenge nonce: expected 64 hex characters".into());
        }
        let mut b = [0u8; 32];
        for (i, byte) in b.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
                .map_err(|_| "challenge nonce: not hex".to_string())?;
        }
        Ok(Self(b))
    }
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
    pub fn hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
    /// The query string of the request that carries it.
    pub fn query(&self) -> String {
        format!("nonce={}", self.hex())
    }
}

impl std::fmt::Debug for ClientChallenge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ClientChallenge({})", self.hex())
    }
}

/// What the client established by itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientVerified {
    /// `mrtd`, `rtmr0..3`, read from the quote this function verified.
    pub measurements: Vec<Measurement>,
    /// Registers the caller pinned, all matched. Empty means nothing pinned:
    /// genuine hardware, unknown image.
    pub pinned: Vec<String>,
}

fn unhex<const N: usize>(doc: &serde_json::Value, field: &str) -> Result<[u8; N], String> {
    let s = doc[field]
        .as_str()
        .ok_or_else(|| format!("attestation has no {field}"))?;
    if s.len() != N * 2 || !s.is_ascii() {
        return Err(format!("{field}: expected {} hex characters", N * 2));
    }
    let mut out = [0u8; N];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .map_err(|_| format!("{field}: not hex"))?;
    }
    Ok(out)
}

fn quote_b64(doc: &serde_json::Value, path: &[&str], what: &str) -> Result<Vec<u8>, String> {
    let mut v = doc;
    for p in path {
        v = &v[*p];
    }
    base64::engine::general_purpose::STANDARD
        .decode(
            v.as_str()
                .ok_or_else(|| format!("attestation publishes no {what}"))?,
        )
        .map_err(|e| format!("{what}: {e}"))
}

/// Verify an `/attestation?nonce=…` document against the key that signed a
/// receipt and against `challenge`, the nonce THIS client sent.
///
/// `anchor` is the CLIENT's pin — `TrustAnchor::intel_production()` outside
/// tests. `receipt_key_hex` is `InferenceReceipt::session_public_key_hex`.
/// Nothing is sent to the server before this returns `Ok`.
pub fn verify_attestation_document(
    doc: &serde_json::Value,
    receipt_key_hex: &str,
    challenge: &ClientChallenge,
    expected: &[ExpectedMeasurement],
    anchor: &TrustAnchor,
    now_unix: u64,
) -> Result<ClientVerified, String> {
    let quote = quote_b64(doc, &["evidence", "tdx_quote_b64"], "TDX quote")?;
    let chain = crate::pcs::verify_tdx_chain(&quote, anchor, now_unix)
        .map_err(|e| format!("TDX quote: {e}"))?;

    // SECURITY: the quote must answer THIS session. Without it any genuine
    // quote — this repository vendors one — would vouch for any key.
    let nonce = unhex::<32>(doc, "session_nonce_hex")?;
    let key_bytes = unhex::<32>(doc, "session_public_key_hex")?;
    let key = ed25519_dalek::VerifyingKey::from_bytes(&key_bytes)
        .map_err(|e| format!("session_public_key_hex: {e}"))?;
    if crate::binding::expected_tdx_report_data(&nonce, &key) != chain.report_data {
        return Err(
            "the quote does not answer the published session key — genuine hardware, \
             but it vouches for some other key"
                .into(),
        );
    }
    // SECURITY: and the receipt must be signed by that attested key, not by
    // one the server merely printed next to it.
    if !receipt_key_hex.eq_ignore_ascii_case(doc["session_public_key_hex"].as_str().unwrap_or("")) {
        return Err("the receipt is signed by a key the hardware did not attest".into());
    }

    // SECURITY: and the quote must answer THIS client's challenge. The boot
    // quote above proves the key was in hardware once; only a quote over a
    // nonce the client drew proves it is there now, for this client. The
    // echoed `challenge.nonce_hex` is not read: the quote is what counts.
    let fresh = quote_b64(
        doc,
        &["challenge", "tdx_quote_b64"],
        "quote answering the challenge",
    )
    .map_err(|e| format!("{e} — the document was not taken for this client"))?;
    let fresh_chain = crate::pcs::verify_tdx_chain(&fresh, anchor, now_unix)
        .map_err(|e| format!("challenge quote: {e}"))?;
    if crate::binding::expected_tdx_report_data(challenge.as_bytes(), &key)
        != fresh_chain.report_data
    {
        return Err(
            "the quote does not answer this client's challenge — genuine hardware, but a \
             document taken for some other nonce: a replay"
                .into(),
        );
    }
    // Both quotes must report the same image: the receipts name the boot
    // quote, the pins are checked on the fresh one.
    let measurements = crate::verify::tdx_measurements(&fresh)?;
    if measurements != crate::verify::tdx_measurements(&quote)? {
        return Err(
            "the quote answering the challenge reports another image than the session quote".into(),
        );
    }
    if let Some(unmet) = crate::verify::unmet_among(&measurements, expected) {
        return Err(unmet);
    }
    Ok(ClientVerified {
        measurements,
        pinned: expected.iter().map(|e| e.register.clone()).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding::SessionIdentity;
    use crate::pcs::tests::{fixture_with_report_data, NOW};

    fn hex(b: &[u8]) -> String {
        b.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The challenge this client drew.
    fn challenge() -> ClientChallenge {
        ClientChallenge::from_bytes([0xc1; 32])
    }

    /// A server's `/attestation?nonce=…`: the boot quote that answers `id`,
    /// and under `challenge` a second quote that answers `nonce`.
    fn published_for(
        id: &SessionIdentity,
        nonce: &ClientChallenge,
    ) -> (serde_json::Value, TrustAnchor) {
        let (mut doc, anchor) = published(id);
        let fx = fixture_with_report_data(id.tdx_report_data_for(nonce.as_bytes()));
        doc["challenge"] = serde_json::json!({
            "nonce_hex": nonce.hex(),
            "tdx_quote_b64": base64::engine::general_purpose::STANDARD.encode(&fx.quote),
        });
        (doc, anchor)
    }

    /// A server's `/attestation` with NO challenge answered: the boot quote
    /// alone, over the nonce the SERVER chose.
    fn published(id: &SessionIdentity) -> (serde_json::Value, TrustAnchor) {
        let fx = fixture_with_report_data(id.tdx_report_data());
        let doc = serde_json::json!({
            "session_public_key_hex": hex(id.public_key().as_bytes()),
            "session_nonce_hex": hex(id.session_nonce()),
            "evidence": {
                "tdx_quote_b64": base64::engine::general_purpose::STANDARD.encode(&fx.quote),
            },
        });
        (doc, TrustAnchor::RootDer(fx.root_der))
    }

    /// SECURITY: a failure means a quote the client never challenged — the
    /// boot quote, genuine, over a nonce the server drew — passes as fresh.
    /// That is the replay: the same document answers every client forever.
    #[test]
    fn a_quote_answering_only_the_servers_own_nonce_is_refused_as_a_replay() {
        let id = SessionIdentity::generate([1u8; 32], [2u8; 32]);
        let (doc, anchor) = published(&id);
        let key = hex(id.public_key().as_bytes());
        let err =
            verify_attestation_document(&doc, &key, &challenge(), &[], &anchor, NOW).unwrap_err();
        assert!(err.contains("challenge"), "{err}");
    }

    /// SECURITY: a failure means a quote that answered ANOTHER client's
    /// challenge is accepted by this one — a recorded answer, replayed.
    #[test]
    fn a_quote_answering_another_clients_challenge_is_refused() {
        let id = SessionIdentity::generate([1u8; 32], [2u8; 32]);
        let other = ClientChallenge::from_bytes([0xd2; 32]);
        let (doc, anchor) = published_for(&id, &other);
        let key = hex(id.public_key().as_bytes());
        let err =
            verify_attestation_document(&doc, &key, &challenge(), &[], &anchor, NOW).unwrap_err();
        assert!(err.contains("challenge"), "{err}");
        // And the echoed nonce is not believed: the quote is what counts.
        let (mut doc, anchor) = published_for(&id, &other);
        doc["challenge"]["nonce_hex"] = challenge().hex().into();
        let err =
            verify_attestation_document(&doc, &key, &challenge(), &[], &anchor, NOW).unwrap_err();
        assert!(err.contains("challenge"), "{err}");
    }

    /// The quote that answers THIS client's challenge, under the attested
    /// key, verifies; the registers come from that fresh quote.
    #[test]
    fn the_quote_that_answers_this_clients_challenge_is_accepted() {
        let id = SessionIdentity::generate([1u8; 32], [2u8; 32]);
        let (doc, anchor) = published_for(&id, &challenge());
        let key = hex(id.public_key().as_bytes());
        let v = verify_attestation_document(&doc, &key, &challenge(), &[], &anchor, NOW)
            .expect("verifies");
        assert_eq!(v.measurements[0].register, "mrtd");
        assert!(v.pinned.is_empty());
    }

    /// Two fresh challenges never coincide, and a challenge round-trips hex.
    #[test]
    fn a_fresh_challenge_is_random_and_round_trips() {
        let a = ClientChallenge::fresh().unwrap();
        let b = ClientChallenge::fresh().unwrap();
        assert_ne!(a, b);
        assert_eq!(ClientChallenge::from_hex(&a.hex()).unwrap(), a);
        assert_eq!(a.query().len(), "nonce=".len() + 64);
        assert!(ClientChallenge::from_hex("zz").is_err());
    }

    #[test]
    fn a_client_reads_the_registers_out_of_a_quote_it_verified_itself() {
        let id = SessionIdentity::generate([1u8; 32], [2u8; 32]);
        let (doc, anchor) = published_for(&id, &challenge());
        let key = hex(id.public_key().as_bytes());
        let pin = crate::verify::parse_expected_measurement(&format!("rtmr1={}", "b2".repeat(48)))
            .unwrap();
        let v = verify_attestation_document(&doc, &key, &challenge(), &[pin], &anchor, NOW)
            .expect("verifies");
        assert_eq!(v.pinned, ["rtmr1"]);
        assert_eq!(v.measurements[0].register, "mrtd");
        assert_eq!(v.measurements[0].value_hex, "a1".repeat(48));
    }

    /// SECURITY: a failure means the client accepts an image it was told to
    /// refuse — the one check an operator cannot switch off from their side.
    #[test]
    fn a_client_refuses_genuine_hardware_running_another_image() {
        let id = SessionIdentity::generate([1u8; 32], [2u8; 32]);
        let (doc, anchor) = published_for(&id, &challenge());
        let key = hex(id.public_key().as_bytes());
        let pin = crate::verify::parse_expected_measurement(&format!("rtmr1={}", "c3".repeat(48)))
            .unwrap();
        let err = verify_attestation_document(&doc, &key, &challenge(), &[pin], &anchor, NOW)
            .unwrap_err();
        assert!(err.contains("rtmr1"), "{err}");
    }

    /// SECURITY: the client refuses a debug TD by itself.
    #[test]
    fn a_client_refuses_a_debug_td() {
        let id = SessionIdentity::generate([1u8; 32], [2u8; 32]);
        let (mut doc, _) = published_for(&id, &challenge());
        let fx = crate::pcs::tests::fixture_debug_td(id.tdx_report_data());
        doc["evidence"]["tdx_quote_b64"] = base64::engine::general_purpose::STANDARD
            .encode(&fx.quote)
            .into();
        let key = hex(id.public_key().as_bytes());
        let err = verify_attestation_document(
            &doc,
            &key,
            &challenge(),
            &[],
            &TrustAnchor::RootDer(fx.root_der),
            NOW,
        )
        .unwrap_err();
        assert!(err.contains("DEBUG"), "{err}");
    }

    /// SECURITY: a failure means a server can publish a genuine quote taken
    /// anywhere next to a key of its own, and the client calls it attested.
    #[test]
    fn a_genuine_quote_that_vouches_for_another_key_is_refused() {
        let attested = SessionIdentity::generate([1u8; 32], [2u8; 32]);
        let impostor = SessionIdentity::generate([3u8; 32], [4u8; 32]);
        let (mut doc, anchor) = published_for(&attested, &challenge());
        doc["session_public_key_hex"] = hex(impostor.public_key().as_bytes()).into();
        let key = hex(impostor.public_key().as_bytes());
        let err =
            verify_attestation_document(&doc, &key, &challenge(), &[], &anchor, NOW).unwrap_err();
        assert!(err.contains("does not answer"), "{err}");
    }

    /// SECURITY: a failure means the attested key and the key that signs
    /// receipts are allowed to be two different keys.
    #[test]
    fn a_receipt_signed_by_an_unattested_key_is_refused() {
        let id = SessionIdentity::generate([1u8; 32], [2u8; 32]);
        let (doc, anchor) = published_for(&id, &challenge());
        let other = hex(SessionIdentity::generate([5u8; 32], [6u8; 32])
            .public_key()
            .as_bytes());
        let err =
            verify_attestation_document(&doc, &other, &challenge(), &[], &anchor, NOW).unwrap_err();
        assert!(err.contains("did not attest"), "{err}");
    }

    #[test]
    fn a_quote_under_another_root_is_refused_by_the_clients_own_pin() {
        let id = SessionIdentity::generate([1u8; 32], [2u8; 32]);
        let (doc, _theirs) = published_for(&id, &challenge());
        let key = hex(id.public_key().as_bytes());
        assert!(verify_attestation_document(
            &doc,
            &key,
            &challenge(),
            &[],
            &TrustAnchor::intel_production(),
            NOW
        )
        .is_err());
    }
}
