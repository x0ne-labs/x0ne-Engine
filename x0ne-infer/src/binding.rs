//! Binding the serving identity into the attestation — invariant 5 of the
//! threat model.
//!
//! At boot, before any evidence is collected, the server generates an
//! ed25519 keypair inside the enclave. The PUBLIC key is folded into the
//! nonces the hardware echoes:
//!
//! ```text
//! tdx report_data (64 B) = SHA-512("x0ne-infer/v1" ‖ session_nonce ‖ pubkey)
//! gpu nonce       (32 B) = SHA-256("x0ne-infer/v1" ‖ session_nonce ‖ pubkey)
//! ```
//!
//! A client who verified the TDX quote and the NRAS token can therefore
//! verify OFFLINE that a receipt signed by this key was produced by the
//! attested session: recompute both digests from (session_nonce, pubkey) —
//! published in `/attestation` — and compare with what the hardware signed.
//! Replay is covered twice: the session_nonce is fresh per boot, and the
//! evidence freshness window (300 s) bounds the quote itself. A CLIENT does
//! not rely on either: it sends its own nonce (`client::ClientChallenge`)
//! and gets a quote over `SHA-512(context ‖ client_nonce ‖ pubkey)` —
//! `tdx_report_data_for` — so the key is shown live in hardware, to it.
//!
//! The SIGNING key never leaves the process and has no serialization; it is
//! zeroized on drop (`ed25519_dalek::SigningKey` implements ZeroizeOnDrop).

use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256, Sha512};

const CONTEXT: &[u8] = b"x0ne-infer/v1";

/// The per-boot serving identity.
#[derive(Clone)]
pub struct SessionIdentity {
    signing: SigningKey,
    session_nonce: [u8; 32],
}

impl SessionIdentity {
    /// Generate. `entropy` must be 32 fresh random bytes (the caller owns
    /// randomness so tests are deterministic and the production source is
    /// explicit — /dev/urandom in main).
    pub fn generate(key_seed: [u8; 32], session_nonce: [u8; 32]) -> Self {
        Self {
            signing: SigningKey::from_bytes(&key_seed),
            session_nonce,
        }
    }

    pub fn public_key(&self) -> VerifyingKey {
        self.signing.verifying_key()
    }
    pub fn session_nonce(&self) -> &[u8; 32] {
        &self.session_nonce
    }

    /// What goes into the TDX inblob.
    pub fn tdx_report_data(&self) -> [u8; 64] {
        let mut h = Sha512::new();
        h.update(CONTEXT);
        h.update(self.session_nonce);
        h.update(self.public_key().as_bytes());
        h.finalize().into()
    }

    /// What goes into the TDX inblob for a CLIENT's challenge: the same
    /// binding, the client's nonce in place of the boot one. The quote
    /// answers that client and no other.
    pub fn tdx_report_data_for(&self, client_nonce: &[u8; 32]) -> [u8; 64] {
        expected_tdx_report_data(client_nonce, &self.public_key())
    }

    /// What goes to NRAS as `nonce` (hex of this).
    pub fn gpu_nonce(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(CONTEXT);
        h.update(self.session_nonce);
        h.update(self.public_key().as_bytes());
        h.finalize().into()
    }

    /// Sign a receipt body (its canonical JSON bytes), under the receipt
    /// domain. There is deliberately no bare `sign`: see [`RECEIPT_DOMAIN`].
    pub fn sign_receipt(&self, body: &[u8]) -> ed25519_dalek::Signature {
        self.signing.sign(&receipt_message(body))
    }
}

/// What the TD report must carry for (`session_nonce`, `public_key`) — the
/// client's half of [`SessionIdentity::tdx_report_data`], computed from what
/// the server PUBLISHED rather than from a secret.
pub fn expected_tdx_report_data(session_nonce: &[u8; 32], public_key: &VerifyingKey) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(CONTEXT);
    h.update(session_nonce);
    h.update(public_key.as_bytes());
    h.finalize().into()
}

/// The client-side check, here so it ships with the server and the SDK can
/// copy it: given the published (session_nonce, pubkey) and the values the
/// hardware echoed, confirm the binding.
pub fn verify_binding(
    session_nonce: &[u8; 32],
    public_key: &VerifyingKey,
    tdx_report_data: &[u8; 64],
    gpu_nonce: &[u8; 32],
) -> bool {
    let mut h = Sha512::new();
    h.update(CONTEXT);
    h.update(session_nonce);
    h.update(public_key.as_bytes());
    let expect_td: [u8; 64] = h.finalize().into();
    let mut h = Sha256::new();
    h.update(CONTEXT);
    h.update(session_nonce);
    h.update(public_key.as_bytes());
    let expect_gpu: [u8; 32] = h.finalize().into();
    // Not secret material — plain comparison is fine and legible.
    &expect_td == tdx_report_data && &expect_gpu == gpu_nonce
}

/// What the session key actually signs: `RECEIPT_DOMAIN ‖ body`.
///
/// SECURITY (2026-09-18): the key used to sign the bare body. It signs
/// one kind of thing today, and that is exactly when a domain is cheap: the
/// day this hardware-attested key signs a second kind of message (a lease, a
/// training manifest), a signature over one must not verify as the other.
/// With no bare `sign` on [`SessionIdentity`], a new use has to declare its
/// own domain to compile. NUL-terminated so no domain prefixes another.
/// PROTOCOL: `console.html` spells the same bytes — a test holds them together.
pub const RECEIPT_DOMAIN: &[u8] = b"x0ne-infer/receipt/v1\0";

fn receipt_message(body: &[u8]) -> Vec<u8> {
    let mut m = Vec::with_capacity(RECEIPT_DOMAIN.len() + body.len());
    m.extend_from_slice(RECEIPT_DOMAIN);
    m.extend_from_slice(body);
    m
}

/// Verify a signed receipt against the published public key.
pub fn verify_receipt_signature(
    public_key: &VerifyingKey,
    body: &[u8],
    signature: &ed25519_dalek::Signature,
) -> bool {
    // SECURITY: `verify` takes a small-order key: the identity point signs every
    // body (measured 2026-11-05). x0ne-sdk, tlog and openpgp are already strict.
    public_key
        .verify_strict(&receipt_message(body), signature)
        .is_ok()
}

/// v2, the cloud channel: the quote binds the session key AND the TLS key
/// AND the client's config, each a fixed-length field. v1 stays for `--listen`.
pub const CONTEXT_V2: &[u8] = b"x0ne-infer/v2";
pub const GPU_CONTEXT_V2: &[u8] = b"x0ne-infer/v2/gpu";

/// What `report_data` covers in v2. Every field is 32 bytes, so there is
/// no framing to get wrong on either side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingV2 {
    pub session_nonce: [u8; 32],
    pub session_pubkey: [u8; 32],
    pub tls_spki_sha256: [u8; 32],
    pub config_sha256: [u8; 32],
}

impl BindingV2 {
    pub fn of(
        identity: &SessionIdentity,
        tls_spki_sha256: [u8; 32],
        config_sha256: [u8; 32],
    ) -> Self {
        Self {
            session_nonce: *identity.session_nonce(),
            session_pubkey: identity.public_key().to_bytes(),
            tls_spki_sha256,
            config_sha256,
        }
    }

    fn feed<D: Digest>(&self, mut h: D, context: &[u8]) -> D {
        h.update(context);
        h.update(self.session_nonce);
        h.update(self.session_pubkey);
        h.update(self.tls_spki_sha256);
        h.update(self.config_sha256);
        h
    }

    /// `SHA-512("x0ne-infer/v2" ‖ nonce ‖ pk ‖ SHA-256(SPKI) ‖ SHA-256(config))`.
    pub fn tdx_report_data(&self) -> [u8; 64] {
        self.feed(Sha512::new(), CONTEXT_V2).finalize().into()
    }

    /// The SEV-SNP `REPORT_DATA` of this binding: the same 64 bytes. The
    /// binding is the session's, not the TEE's; what tells a TDX quote from
    /// an SNP report is the root its signature chains to, Intel's or AMD's.
    pub fn snp_report_data(&self) -> [u8; 64] {
        self.tdx_report_data()
    }

    /// `SHA-256("x0ne-infer/v2/gpu" ‖ the same fields)`.
    pub fn gpu_nonce(&self) -> [u8; 32] {
        self.feed(Sha256::new(), GPU_CONTEXT_V2).finalize().into()
    }

    pub fn gpu_nonce_hex(&self) -> String {
        self.gpu_nonce()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// Does `report_data` (read from a quote) bind exactly these fields?
    pub fn matches(&self, report_data: &[u8; 64]) -> bool {
        &self.tdx_report_data() == report_data
    }
}

/// v3, per connection: the quote binds the TLS exporter of ONE connection
/// (RFC 8446 §7.5, after both `Finished`), the session key, the TLS key and
/// the config. A party holding the TD's TLS key runs its own handshakes, so
/// its exporters differ from the client's: it cannot relay a v3 quote.
pub const CONTEXT_V3: &[u8] = b"x0ne-infer/v3";
/// The exporter label; its context is the client's config nonce.
pub const EXPORTER_LABEL_V3: &[u8] = b"EXPORTER-x0ne-infer-attest/v3";
pub const EXPORTER_LEN: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingV3 {
    pub exporter: [u8; 32],
    pub session_pubkey: [u8; 32],
    pub tls_spki_sha256: [u8; 32],
    pub config_sha256: [u8; 32],
}

impl BindingV3 {
    /// The v2 binding of the session, on the connection that exported `exporter`.
    pub fn of(session: &BindingV2, exporter: [u8; 32]) -> Self {
        Self {
            exporter,
            session_pubkey: session.session_pubkey,
            tls_spki_sha256: session.tls_spki_sha256,
            config_sha256: session.config_sha256,
        }
    }

    /// `SHA-512("x0ne-infer/v3" ‖ exporter ‖ pk ‖ SHA-256(SPKI) ‖ SHA-256(config))`.
    pub fn tdx_report_data(&self) -> [u8; 64] {
        let mut h = Sha512::new();
        h.update(CONTEXT_V3);
        h.update(self.exporter);
        h.update(self.session_pubkey);
        h.update(self.tls_spki_sha256);
        h.update(self.config_sha256);
        h.finalize().into()
    }

    /// The SEV-SNP `REPORT_DATA` of this connection: the same 64 bytes as
    /// [`BindingV3::tdx_report_data`].
    pub fn snp_report_data(&self) -> [u8; 64] {
        self.tdx_report_data()
    }

    pub fn matches(&self, report_data: &[u8; 64]) -> bool {
        &self.tdx_report_data() == report_data
    }
}

/// v4, the GPU of one connection: the nonce NRAS echoes for the GPU evidence
/// taken on a connection.
pub const GPU_CONTEXT_V4: &[u8] = b"x0ne-infer/v4/gpu";

/// `SHA-256("x0ne-infer/v4/gpu" ‖ exporter ‖ SHA-512(td_report))`, where
/// `td_report` is the signed part of that connection's session quote (header
/// and TD report body, `report_data` included: `pcs::signed_td_report`).
///
/// The nonce exists only once the quote does, and only a party that knows the
/// connection's exporter can compute it: the TD and the client. GPU evidence
/// taken for another connection, another TD or the boot document answers
/// another nonce. Hopper's GPU report names no CVM: this binds the evidence
/// to the connection through the TD's software, not through the hardware.
pub fn session_gpu_nonce(exporter: &[u8; 32], td_report: &[u8]) -> [u8; 32] {
    let report: [u8; 64] = Sha512::digest(td_report).into();
    let mut h = Sha256::new();
    h.update(GPU_CONTEXT_V4);
    h.update(exporter);
    h.update(report);
    h.finalize().into()
}

#[cfg(test)]
mod tests_v4_gpu {
    use super::*;

    fn report() -> Vec<u8> {
        (0..632u32).map(|i| (i % 251) as u8).collect()
    }

    /// The derivation, spelled out: the client computes it from the bytes
    /// the quoting enclave signed, nothing published.
    #[test]
    fn the_session_gpu_nonce_is_sha256_over_context_exporter_and_report_digest() {
        let exporter = [0x33; 32];
        let mut h = Sha256::new();
        h.update(b"x0ne-infer/v4/gpu");
        h.update(exporter);
        h.update(Sha512::digest(report()));
        let want: [u8; 32] = h.finalize().into();
        assert_eq!(session_gpu_nonce(&exporter, &report()), want);
    }

    /// SECURITY: another connection (exporter) or another TD report (any
    /// byte: header, MRTD, RTMRs, `report_data`) gives another nonce, so a
    /// GPU token taken for either does not answer this connection's.
    #[test]
    fn the_session_gpu_nonce_binds_the_exporter_and_every_byte_of_the_td_report() {
        let exporter = [0x33; 32];
        let n = session_gpu_nonce(&exporter, &report());
        for i in [0, 31] {
            let mut other = exporter;
            other[i] ^= 1;
            assert_ne!(session_gpu_nonce(&other, &report()), n, "exporter byte {i}");
        }
        for at in [0, 48 + 136, 48 + 328, 568, 631] {
            let mut r = report();
            r[at] ^= 1;
            assert_ne!(session_gpu_nonce(&exporter, &r), n, "report byte {at}");
        }
        let mut longer = report();
        longer.push(0);
        assert_ne!(session_gpu_nonce(&exporter, &longer), n, "report length");
    }
}

#[cfg(test)]
mod tests_v3 {
    use super::*;

    fn v2() -> BindingV2 {
        let id = SessionIdentity::generate([7u8; 32], [9u8; 32]);
        BindingV2::of(&id, [0x11; 32], [0x22; 32])
    }

    /// v3 never equals v2, and every field, the exporter first, is bound.
    #[test]
    fn v3_differs_from_v2_and_binds_every_field() {
        let b = BindingV3::of(&v2(), [0x33; 32]);
        let rd = b.tdx_report_data();
        assert_ne!(rd, v2().tdx_report_data());
        assert!(b.matches(&rd));
        for i in 0..4 {
            let mut other = b;
            match i {
                0 => other.exporter[0] ^= 1,
                1 => other.session_pubkey[0] ^= 1,
                2 => other.tls_spki_sha256[0] ^= 1,
                _ => other.config_sha256[0] ^= 1,
            }
            assert!(!other.matches(&rd), "field {i} substituted");
        }
    }

    /// One binding, two TEEs: an SNP report's REPORT_DATA is the bytes a
    /// TDX quote's report_data is, for v2 and for v3.
    #[test]
    fn the_snp_report_data_is_the_tdx_report_data() {
        let b = BindingV3::of(&v2(), [0x33; 32]);
        assert_eq!(b.snp_report_data(), b.tdx_report_data());
        assert_eq!(v2().snp_report_data(), v2().tdx_report_data());
    }
}

#[cfg(test)]
mod tests_v2 {
    use super::*;

    fn binding() -> BindingV2 {
        let id = SessionIdentity::generate([7u8; 32], [9u8; 32]);
        BindingV2::of(&id, [0x11; 32], [0x22; 32])
    }

    /// The v1 values are unchanged by the v2 addition: a `--listen` server
    /// and its clients keep agreeing.
    #[test]
    fn v1_report_data_is_unchanged() {
        let id = SessionIdentity::generate([7u8; 32], [9u8; 32]);
        let mut h = Sha512::new();
        h.update(b"x0ne-infer/v1");
        h.update([9u8; 32]);
        h.update(id.public_key().as_bytes());
        let want: [u8; 64] = h.finalize().into();
        assert_eq!(id.tdx_report_data(), want);
    }

    /// v2 and v1 never collide for the same identity, and the GPU nonce is
    /// not a truncation of the report data.
    #[test]
    fn v2_differs_from_v1_and_its_two_values_differ() {
        let id = SessionIdentity::generate([7u8; 32], [9u8; 32]);
        let b = BindingV2::of(&id, [0x11; 32], [0x22; 32]);
        assert_ne!(b.tdx_report_data(), id.tdx_report_data());
        assert_ne!(b.gpu_nonce(), id.gpu_nonce());
        assert_ne!(&b.tdx_report_data()[..32], &b.gpu_nonce()[..]);
    }

    /// SECURITY: every field substitution breaks the binding — a host that
    /// swaps the TLS key or the config cannot keep the quote.
    #[test]
    fn every_field_substitution_breaks_the_binding() {
        let b = binding();
        let rd = b.tdx_report_data();
        assert!(b.matches(&rd));
        for i in 0..4 {
            let mut other = b;
            match i {
                0 => other.session_nonce[0] ^= 1,
                1 => other.session_pubkey[0] ^= 1,
                2 => other.tls_spki_sha256[0] ^= 1,
                _ => other.config_sha256[0] ^= 1,
            }
            assert!(!other.matches(&rd), "field {i} substituted");
            assert_ne!(other.gpu_nonce(), b.gpu_nonce(), "field {i}, gpu nonce");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> SessionIdentity {
        SessionIdentity::generate([7u8; 32], [9u8; 32])
    }

    /// The binding verifies with the right tuple and fails on ANY substituted
    /// element — the exact swap a malicious operator would try (their own
    /// key under our attestation, or our key under an old attestation).
    #[test]
    fn the_binding_holds_and_every_substitution_breaks_it() {
        let id = identity();
        let td = id.tdx_report_data();
        let gpu = id.gpu_nonce();
        assert!(verify_binding(
            id.session_nonce(),
            &id.public_key(),
            &td,
            &gpu
        ));
        let other = SessionIdentity::generate([8u8; 32], [9u8; 32]);
        assert!(
            !verify_binding(id.session_nonce(), &other.public_key(), &td, &gpu),
            "swapped key"
        );
        assert!(
            !verify_binding(&[1u8; 32], &id.public_key(), &td, &gpu),
            "swapped nonce"
        );
        let mut bad_td = td;
        bad_td[0] ^= 1;
        assert!(
            !verify_binding(id.session_nonce(), &id.public_key(), &bad_td, &gpu),
            "wrong report_data"
        );
    }

    /// A signed receipt verifies under the session key and under no other.
    #[test]
    fn a_receipt_signature_binds_to_the_session_key() {
        let id = identity();
        let body = br#"{"model_digest_hex":"abcd"}"#;
        let sig = id.sign_receipt(body);
        assert!(verify_receipt_signature(&id.public_key(), body, &sig));
        let other = SessionIdentity::generate([8u8; 32], [9u8; 32]);
        assert!(!verify_receipt_signature(&other.public_key(), body, &sig));
        assert!(!verify_receipt_signature(
            &id.public_key(),
            b"tampered",
            &sig
        ));
    }

    /// SECURITY: a failure means the attested key's signature over some
    /// OTHER kind of message — same bytes, no receipt domain — is accepted as
    /// a receipt. That is the cross-protocol confusion the domain exists for.
    #[test]
    fn a_signature_over_the_bare_body_is_not_a_receipt_signature() {
        let id = identity();
        let body = br#"{"model_digest_hex":"abcd"}"#;
        let bare = id.signing.sign(body);
        assert!(!verify_receipt_signature(&id.public_key(), body, &bare));
        // And the domain is what was agreed with the browser verifier.
        assert_eq!(RECEIPT_DOMAIN, b"x0ne-infer/receipt/v1\0");
    }

    /// SECURITY: a failure means a small-order key verifies one signature
    /// over every body: the identity point, then (R = identity, s = 0).
    #[test]
    fn a_small_order_key_verifies_no_receipt_signature_whatever_the_body() {
        let mut identity_point = [0u8; 32];
        identity_point[0] = 1;
        let key = VerifyingKey::from_bytes(&identity_point).unwrap();
        let mut forged = [0u8; 64];
        forged[0] = 1;
        let forged = ed25519_dalek::Signature::from_bytes(&forged);
        for body in [&b"one body"[..], &b"another body"[..]] {
            assert!(!verify_receipt_signature(&key, body, &forged));
        }
    }
}
