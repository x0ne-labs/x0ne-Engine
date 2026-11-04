//! The attested channel: TLS 1.3, both ends pinned by SPKI.
//!
//! The TD generates a P-256 key at boot and folds `SHA-256(SPKI)` into the
//! quote (`binding::BindingV2`). The client pins the first server key it
//! shakes hands with and later compares it to the attested one; the TD
//! accepts exactly one client key, the one named in the config it was booted
//! with (`cloud_config`). No CA on either side: the quote is the CA.

use crate::harden::SecretBytes;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{
    verify_tls12_signature, verify_tls13_signature, CryptoProvider, WebPkiSupportedAlgorithms,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::DistinguishedName;
use rustls::{CertificateError, DigitallySignedStruct, Error, SignatureScheme};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use x509_parser::prelude::FromDer;
use zeroize::Zeroizing;

/// The v3 exporter of one finished TLS 1.3 connection (RFC 8446 §7.5),
/// `context` = the client's config nonce. Same value at both ends of one
/// connection; any other connection, relayed or not, gives another.
pub fn exporter<D>(
    conn: &rustls::ConnectionCommon<D>,
    client_nonce: &[u8; 32],
) -> Result<[u8; 32], Error> {
    conn.export_keying_material(
        [0u8; crate::binding::EXPORTER_LEN],
        crate::binding::EXPORTER_LABEL_V3,
        Some(client_nonce),
    )
}

/// HTTP/1.1 only on the attested channel: hyper serves http1 (`cloud_serve`).
pub const ALPN_HTTP1: &[u8] = b"http/1.1";

/// One ephemeral P-256 key with a self-signed certificate around it.
pub struct TlsIdentity {
    cert: CertificateDer<'static>,
    key_pkcs8: SecretBytes,
    spki_sha256: [u8; 32],
}

impl TlsIdentity {
    pub fn generate(common_name: &str) -> anyhow::Result<Self> {
        let key = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)?;
        let mut params = rcgen::CertificateParams::new(vec![common_name.to_string()])?;
        params
            .distinguished_name
            .push(rcgen::DnType::CommonName, common_name);
        let cert = params.self_signed(&key)?;
        let spki_sha256 = spki_sha256_of_cert(cert.der()).map_err(|e| anyhow::anyhow!(e))?;
        let der = Zeroizing::new(key.serialize_der());
        Ok(Self {
            cert: cert.der().clone(),
            key_pkcs8: SecretBytes::from_slice(&der)?,
            spki_sha256,
        })
    }

    pub fn spki_sha256(&self) -> [u8; 32] {
        self.spki_sha256
    }
    pub fn cert_der(&self) -> &CertificateDer<'static> {
        &self.cert
    }
    fn private_key(&self) -> PrivateKeyDer<'static> {
        PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(self.key_pkcs8.expose().to_vec()))
    }
}

/// `SHA-256(SubjectPublicKeyInfo)` of a DER certificate — the pin on both sides.
pub fn spki_sha256_of_cert(der: &[u8]) -> Result<[u8; 32], String> {
    let (_, cert) = x509_parser::certificate::X509Certificate::from_der(der)
        .map_err(|e| format!("certificate: {e}"))?;
    Ok(Sha256::digest(cert.public_key().raw).into())
}

/// The two AES-GCM suites of TLS 1.3, nothing else: NIST SP 800-52r2
/// §3.3.1.2 approves GCM and CCM, not ChaCha20; no TLS 1.2 suite is offered
/// either. Both ends are ours, so a peer cannot pick another.
pub const CIPHER_SUITES: [rustls::SupportedCipherSuite; 2] = [
    rustls::crypto::ring::cipher_suite::TLS13_AES_256_GCM_SHA384,
    rustls::crypto::ring::cipher_suite::TLS13_AES_128_GCM_SHA256,
];

pub fn provider() -> Arc<CryptoProvider> {
    Arc::new(CryptoProvider {
        cipher_suites: CIPHER_SUITES.to_vec(),
        ..rustls::crypto::ring::default_provider()
    })
}

/// The TD side: TLS 1.3 only, client certificate mandatory, exactly one
/// client SPKI accepted — on every route, the attestation document included.
pub fn server_config(
    identity: &TlsIdentity,
    client_spki_sha256: [u8; 32],
) -> anyhow::Result<Arc<rustls::ServerConfig>> {
    let provider = provider();
    let verifier = Arc::new(PinnedClient {
        expected: client_spki_sha256,
        algs: provider.signature_verification_algorithms,
        hints: Vec::new(),
    });
    let mut config = rustls::ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .with_client_cert_verifier(verifier)
        .with_single_cert(vec![identity.cert.clone()], identity.private_key())?;
    config.alpn_protocols = vec![ALPN_HTTP1.to_vec()];
    // SECURITY: a resumed handshake skips the client-certificate check and
    // 0-RTT data is replayable; rustls 0.23 defaults keep a session cache
    // and send 2 tickets. One connection = one full handshake = one quote.
    config.session_storage = Arc::new(rustls::server::NoServerSessionStorage {});
    config.send_tls13_tickets = 0;
    config.max_early_data_size = 0;
    Ok(Arc::new(config))
}

/// What the client saw: the server SPKI of the first handshake, which every
/// later handshake on the same config must repeat. Compared to the quote
/// by the verifier (`remote_verify`), not here.
#[derive(Debug, Default)]
pub struct ServerPin {
    seen: Mutex<Option<[u8; 32]>>,
    checks: AtomicU64,
}

impl ServerPin {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
    pub fn pinned(&self) -> Option<[u8; 32]> {
        *self.seen.lock().unwrap_or_else(|e| e.into_inner())
    }
    /// Server certificates checked so far: one per full handshake.
    pub fn checks(&self) -> u64 {
        self.checks.load(Ordering::SeqCst)
    }
}

/// The client side: TLS 1.3 only, our certificate offered, the server pinned
/// on first use. The pin is NOT trust: nothing useful goes over the channel
/// until the verifier has matched it against the quote.
pub fn client_config(
    identity: &TlsIdentity,
    pin: Arc<ServerPin>,
) -> anyhow::Result<rustls::ClientConfig> {
    let provider = provider();
    let verifier = Arc::new(PinOnFirstUse {
        pin,
        algs: provider.signature_verification_algorithms,
    });
    let mut config = rustls::ClientConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .dangerous()
        .with_custom_certificate_verifier(verifier)
        .with_client_auth_cert(vec![identity.cert.clone()], identity.private_key())?;
    config.alpn_protocols = vec![ALPN_HTTP1.to_vec()];
    // SECURITY: rustls 0.23 resumes by default, and a resumption does not
    // call the server-certificate verifier. No tickets kept, no 0-RTT.
    config.resumption = rustls::client::Resumption::disabled();
    config.enable_early_data = false;
    Ok(config)
}

struct PinnedClient {
    expected: [u8; 32],
    algs: WebPkiSupportedAlgorithms,
    hints: Vec<DistinguishedName>,
}

impl std::fmt::Debug for PinnedClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PinnedClient({})",
            crate::cloud_config::hex(&self.expected)
        )
    }
}

fn mismatch() -> Error {
    Error::InvalidCertificate(CertificateError::ApplicationVerificationFailure)
}

impl ClientCertVerifier for PinnedClient {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &self.hints
    }

    fn verify_client_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _now: UnixTime,
    ) -> Result<ClientCertVerified, Error> {
        let spki = spki_sha256_of_cert(end_entity)
            .map_err(|_| Error::InvalidCertificate(CertificateError::BadEncoding))?;
        if spki != self.expected {
            return Err(mismatch());
        }
        Ok(ClientCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        verify_tls12_signature(message, cert, dss, &self.algs)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        verify_tls13_signature(message, cert, dss, &self.algs)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algs.supported_schemes()
    }
}

struct PinOnFirstUse {
    pin: Arc<ServerPin>,
    algs: WebPkiSupportedAlgorithms,
}

impl std::fmt::Debug for PinOnFirstUse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PinOnFirstUse({:?})",
            self.pin.pinned().map(|p| crate::cloud_config::hex(&p))
        )
    }
}

impl ServerCertVerifier for PinOnFirstUse {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, Error> {
        self.pin.checks.fetch_add(1, Ordering::SeqCst);
        let spki = spki_sha256_of_cert(end_entity)
            .map_err(|_| Error::InvalidCertificate(CertificateError::BadEncoding))?;
        let mut seen = self.pin.seen.lock().unwrap_or_else(|e| e.into_inner());
        match *seen {
            None => {
                *seen = Some(spki);
                Ok(ServerCertVerified::assertion())
            }
            Some(pinned) if pinned == spki => Ok(ServerCertVerified::assertion()),
            // SECURITY: a second handshake with another key is a different
            // server, whatever it attests later.
            Some(_) => Err(mismatch()),
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        verify_tls12_signature(message, cert, dss, &self.algs)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        verify_tls13_signature(message, cert, dss, &self.algs)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algs.supported_schemes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn echo_server(
        config: Arc<rustls::ServerConfig>,
    ) -> (
        std::net::SocketAddr,
        tokio::task::JoinHandle<Vec<Result<(), String>>>,
    ) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let acceptor = tokio_rustls::TlsAcceptor::from(config);
        let handle = tokio::spawn(async move {
            let mut outcomes = Vec::new();
            for _ in 0..3 {
                let Ok((tcp, _)) = listener.accept().await else {
                    break;
                };
                match acceptor.accept(tcp).await {
                    Ok(mut tls) => {
                        let mut buf = [0u8; 4];
                        let r = tls.read_exact(&mut buf).await;
                        if r.is_ok() {
                            let _ = tls.write_all(&buf).await;
                        }
                        outcomes.push(Ok(()));
                    }
                    Err(e) => outcomes.push(Err(e.to_string())),
                }
            }
            outcomes
        });
        (addr, handle)
    }

    async fn connect(
        addr: std::net::SocketAddr,
        config: rustls::ClientConfig,
    ) -> Result<(), String> {
        let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
        let tcp = tokio::net::TcpStream::connect(addr)
            .await
            .map_err(|e| e.to_string())?;
        let name = ServerName::try_from("x0ne-infer").unwrap();
        let mut tls = connector
            .connect(name, tcp)
            .await
            .map_err(|e| e.to_string())?;
        tls.write_all(b"ping").await.map_err(|e| e.to_string())?;
        let mut buf = [0u8; 4];
        tls.read_exact(&mut buf).await.map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Both ends of one connection derive the same exporter; a second
    /// connection with the same keys and nonce derives another; another
    /// nonce (exporter context) another again.
    #[tokio::test]
    async fn the_exporter_is_shared_by_one_connection_and_by_no_other() {
        let server = TlsIdentity::generate("x0ne-infer").unwrap();
        let client = TlsIdentity::generate("x0ne-engine-proxy").unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let acceptor =
            tokio_rustls::TlsAcceptor::from(server_config(&server, client.spki_sha256()).unwrap());
        let nonce = [5u8; 32];
        let srv = tokio::spawn(async move {
            let mut out = Vec::new();
            for _ in 0..2 {
                let (tcp, _) = listener.accept().await.unwrap();
                let mut tls = acceptor.accept(tcp).await.unwrap();
                let mut buf = [0u8; 4];
                tls.read_exact(&mut buf).await.unwrap();
                out.push(exporter(tls.get_ref().1, &nonce).unwrap());
                tls.write_all(&buf).await.unwrap();
            }
            out
        });
        let connector = tokio_rustls::TlsConnector::from(Arc::new(
            client_config(&client, ServerPin::new()).unwrap(),
        ));
        let mut seen = Vec::new();
        for _ in 0..2 {
            let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
            let name = ServerName::try_from("x0ne-infer").unwrap();
            let mut tls = connector.connect(name, tcp).await.unwrap();
            tls.write_all(b"ping").await.unwrap();
            let mut buf = [0u8; 4];
            tls.read_exact(&mut buf).await.unwrap();
            let conn = tls.get_ref().1;
            seen.push(exporter(conn, &nonce).unwrap());
            assert_ne!(exporter(conn, &[6u8; 32]).unwrap(), seen[seen.len() - 1]);
        }
        let server_side = srv.await.unwrap();
        assert_eq!(
            server_side, seen,
            "one connection, one exporter at both ends"
        );
        assert_ne!(seen[0], seen[1], "two connections, two exporters");
    }

    /// The honest pair: the TD pins the client's SPKI, the client pins the
    /// server's on first use and sees exactly the server's SPKI.
    #[tokio::test]
    async fn the_pinned_pair_shakes_hands_and_the_client_reads_the_server_spki() {
        let server = TlsIdentity::generate("x0ne-infer").unwrap();
        let client = TlsIdentity::generate("x0ne-engine-proxy").unwrap();
        let (addr, _h) = echo_server(server_config(&server, client.spki_sha256()).unwrap()).await;
        let pin = ServerPin::new();
        connect(addr, client_config(&client, pin.clone()).unwrap())
            .await
            .expect("handshake");
        assert_eq!(pin.pinned(), Some(server.spki_sha256()));
    }

    /// SECURITY: a client whose key is not the one in the config is refused
    /// at the handshake — it never reaches a route, attestation included.
    #[tokio::test]
    async fn a_client_with_another_key_is_refused_at_the_handshake() {
        let server = TlsIdentity::generate("x0ne-infer").unwrap();
        let client = TlsIdentity::generate("x0ne-engine-proxy").unwrap();
        let other = TlsIdentity::generate("x0ne-engine-proxy").unwrap();
        let (addr, h) = echo_server(server_config(&server, client.spki_sha256()).unwrap()).await;
        let err = connect(addr, client_config(&other, ServerPin::new()).unwrap())
            .await
            .expect_err("refused");
        assert!(!err.is_empty());
        // The server side logged a handshake failure, not a served request.
        drop(h);
    }

    /// SECURITY: without any client certificate there is no session at all.
    #[tokio::test]
    async fn no_client_certificate_means_no_session() {
        let server = TlsIdentity::generate("x0ne-infer").unwrap();
        let client = TlsIdentity::generate("x0ne-engine-proxy").unwrap();
        let (addr, _h) = echo_server(server_config(&server, client.spki_sha256()).unwrap()).await;
        let provider = provider();
        let no_cert = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .unwrap()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(PinOnFirstUse {
                pin: ServerPin::new(),
                algs: provider.signature_verification_algorithms,
            }))
            .with_no_client_auth();
        connect(addr, no_cert).await.expect_err("refused");
    }

    /// SECURITY: once pinned, a server presenting another key is refused,
    /// so a connection pool cannot drift to a second machine.
    #[tokio::test]
    async fn a_second_server_key_under_the_same_pin_is_refused() {
        let server_a = TlsIdentity::generate("a").unwrap();
        let server_b = TlsIdentity::generate("b").unwrap();
        let client = TlsIdentity::generate("c").unwrap();
        let (addr_a, _ha) =
            echo_server(server_config(&server_a, client.spki_sha256()).unwrap()).await;
        let (addr_b, _hb) =
            echo_server(server_config(&server_b, client.spki_sha256()).unwrap()).await;
        let pin = ServerPin::new();
        connect(addr_a, client_config(&client, pin.clone()).unwrap())
            .await
            .unwrap();
        connect(addr_b, client_config(&client, pin.clone()).unwrap())
            .await
            .expect_err("another key under the same pin");
        assert_eq!(pin.pinned(), Some(server_a.spki_sha256()));
    }

    #[test]
    fn the_spki_pin_is_the_certificates_public_key_not_the_certificate() {
        let id = TlsIdentity::generate("x").unwrap();
        let cert_hash: [u8; 32] = Sha256::digest(id.cert_der().as_ref()).into();
        assert_ne!(id.spki_sha256(), cert_hash);
        assert_eq!(
            spki_sha256_of_cert(id.cert_der()).unwrap(),
            id.spki_sha256()
        );
        assert!(spki_sha256_of_cert(b"not a certificate").is_err());
        #[cfg(target_os = "linux")]
        assert_ne!(id.key_pkcs8.backing(), crate::harden::Backing::Heap);
    }

    // ── Resumption and 0-RTT: one connection, one full handshake ────────────

    use rustls::client::ClientSessionStore;
    use rustls::HandshakeKind;

    /// A client session store that counts the TLS 1.3 tickets it was given.
    #[derive(Debug)]
    struct CountingStore {
        inner: rustls::client::ClientSessionMemoryCache,
        tickets: AtomicU64,
    }

    impl CountingStore {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                inner: rustls::client::ClientSessionMemoryCache::new(32),
                tickets: AtomicU64::new(0),
            })
        }
        fn tickets(&self) -> u64 {
            self.tickets.load(Ordering::SeqCst)
        }
    }

    impl ClientSessionStore for CountingStore {
        fn set_kx_hint(&self, n: ServerName<'static>, g: rustls::NamedGroup) {
            self.inner.set_kx_hint(n, g);
        }
        fn kx_hint(&self, n: &ServerName<'_>) -> Option<rustls::NamedGroup> {
            self.inner.kx_hint(n)
        }
        fn set_tls12_session(
            &self,
            n: ServerName<'static>,
            v: rustls::client::Tls12ClientSessionValue,
        ) {
            self.inner.set_tls12_session(n, v);
        }
        fn tls12_session(
            &self,
            n: &ServerName<'_>,
        ) -> Option<rustls::client::Tls12ClientSessionValue> {
            self.inner.tls12_session(n)
        }
        fn remove_tls12_session(&self, n: &ServerName<'static>) {
            self.inner.remove_tls12_session(n);
        }
        fn insert_tls13_ticket(
            &self,
            n: ServerName<'static>,
            v: rustls::client::Tls13ClientSessionValue,
        ) {
            self.tickets.fetch_add(1, Ordering::SeqCst);
            self.inner.insert_tls13_ticket(n, v);
        }
        fn take_tls13_ticket(
            &self,
            n: &ServerName<'static>,
        ) -> Option<rustls::client::Tls13ClientSessionValue> {
            self.inner.take_tls13_ticket(n)
        }
    }

    /// rustls' defaults around our verifier: a session cache, two tickets
    /// per handshake, and here 0-RTT on. What a careless TD would run.
    fn resuming_server(
        identity: &TlsIdentity,
        client_spki_sha256: [u8; 32],
    ) -> Arc<rustls::ServerConfig> {
        let provider = provider();
        let verifier = Arc::new(PinnedClient {
            expected: client_spki_sha256,
            algs: provider.signature_verification_algorithms,
            hints: Vec::new(),
        });
        let mut config = rustls::ServerConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&rustls::version::TLS13])
            .unwrap()
            .with_client_cert_verifier(verifier)
            .with_single_cert(vec![identity.cert.clone()], identity.private_key())
            .unwrap();
        config.max_early_data_size = 16 * 1024;
        Arc::new(config)
    }

    /// rustls' defaults around our pin, plus 0-RTT asked: a careless client.
    fn resuming_client(
        identity: &TlsIdentity,
        pin: Arc<ServerPin>,
        store: Arc<CountingStore>,
    ) -> rustls::ClientConfig {
        let provider = provider();
        let mut config = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .unwrap()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(PinOnFirstUse {
                pin,
                algs: provider.signature_verification_algorithms,
            }))
            .with_client_auth_cert(vec![identity.cert.clone()], identity.private_key())
            .unwrap();
        config.resumption = rustls::client::Resumption::store(store);
        config.enable_early_data = true;
        config
    }

    #[derive(Debug)]
    struct Seen {
        kind: Option<HandshakeKind>,
        early_bytes: usize,
        client_cert: bool,
    }

    /// Accepts `n` connections; for each: how the handshake went, how many
    /// 0-RTT bytes were accepted, then the rest of "ping" and a "pong".
    async fn probe_server(
        config: Arc<rustls::ServerConfig>,
        n: usize,
    ) -> (std::net::SocketAddr, tokio::task::JoinHandle<Vec<Seen>>) {
        use std::io::Read as _;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let acceptor = tokio_rustls::TlsAcceptor::from(config);
        let handle = tokio::spawn(async move {
            let mut seen = Vec::new();
            for _ in 0..n {
                let (tcp, _) = listener.accept().await.unwrap();
                let mut tls = acceptor.accept(tcp).await.unwrap();
                let mut early = Vec::new();
                if let Some(mut ed) = tls.get_mut().1.early_data() {
                    ed.read_to_end(&mut early).unwrap();
                }
                let conn = tls.get_ref().1;
                seen.push(Seen {
                    kind: conn.handshake_kind(),
                    early_bytes: early.len(),
                    client_cert: conn.peer_certificates().is_some(),
                });
                let mut rest = vec![0u8; 4usize.saturating_sub(early.len())];
                tls.read_exact(&mut rest).await.unwrap();
                tls.write_all(b"pong").await.unwrap();
                tls.flush().await.unwrap();
            }
            seen
        });
        (addr, handle)
    }

    /// One connection: "ping" (as 0-RTT when the connector may), "pong" back.
    async fn probe(
        connector: &tokio_rustls::TlsConnector,
        addr: std::net::SocketAddr,
    ) -> (Option<HandshakeKind>, bool) {
        let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
        let name = ServerName::try_from("x0ne-infer").unwrap();
        let mut tls = connector.connect(name, tcp).await.unwrap();
        tls.write_all(b"ping").await.unwrap();
        tls.flush().await.unwrap();
        let mut buf = [0u8; 4];
        tls.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"pong");
        let conn = tls.get_ref().1;
        (conn.handshake_kind(), conn.is_early_data_accepted())
    }

    /// Control: the probes do see resumption and 0-RTT when both ends allow
    /// them, so the refusals below are not an artefact of the harness.
    #[tokio::test]
    async fn control_a_careless_pair_resumes_and_sends_early_data() {
        let server = TlsIdentity::generate("x0ne-infer").unwrap();
        let client = TlsIdentity::generate("x0ne-engine-proxy").unwrap();
        let (addr, srv) = probe_server(resuming_server(&server, client.spki_sha256()), 2).await;
        let pin = ServerPin::new();
        let store = CountingStore::new();
        let connector = tokio_rustls::TlsConnector::from(Arc::new(resuming_client(
            &client,
            pin.clone(),
            store.clone(),
        )))
        .early_data(true);
        assert_eq!(probe(&connector, addr).await.0, Some(HandshakeKind::Full));
        let (kind, early) = probe(&connector, addr).await;
        assert_eq!(kind, Some(HandshakeKind::Resumed));
        assert!(early, "0-RTT accepted");
        assert!(store.tickets() >= 1);
        assert_eq!(pin.checks(), 1, "a resumption skips the certificate check");
        let seen = srv.await.unwrap();
        assert_eq!(seen[1].kind, Some(HandshakeKind::Resumed), "{seen:?}");
        assert_eq!(seen[1].early_bytes, 4, "{seen:?}");
    }

    /// SECURITY (client side): against a TD that would resume, our client
    /// shakes hands in full every time and checks the certificate again.
    #[tokio::test]
    async fn our_client_never_resumes_and_rechecks_the_certificate() {
        let server = TlsIdentity::generate("x0ne-infer").unwrap();
        let client = TlsIdentity::generate("x0ne-engine-proxy").unwrap();
        let (addr, srv) = probe_server(resuming_server(&server, client.spki_sha256()), 3).await;
        let pin = ServerPin::new();
        let config = client_config(&client, pin.clone()).unwrap();
        assert!(!config.enable_early_data);
        let connector = tokio_rustls::TlsConnector::from(Arc::new(config)).early_data(true);
        for _ in 0..3 {
            assert_eq!(
                probe(&connector, addr).await,
                (Some(HandshakeKind::Full), false)
            );
        }
        assert_eq!(pin.checks(), 3, "one certificate check per connection");
        for s in srv.await.unwrap() {
            assert_eq!(s.kind, Some(HandshakeKind::Full), "{s:?}");
            assert_eq!(s.early_bytes, 0, "{s:?}");
            assert!(s.client_cert, "{s:?}");
        }
    }

    /// SECURITY (TD side): a client that asks for tickets and 0-RTT gets
    /// neither; every connection is a full handshake with its certificate.
    #[tokio::test]
    async fn our_server_issues_no_ticket_and_takes_no_early_data() {
        let server = TlsIdentity::generate("x0ne-infer").unwrap();
        let client = TlsIdentity::generate("x0ne-engine-proxy").unwrap();
        let config = server_config(&server, client.spki_sha256()).unwrap();
        assert_eq!(config.send_tls13_tickets, 0);
        assert_eq!(config.max_early_data_size, 0);
        assert!(!config.ticketer.enabled());
        let (addr, srv) = probe_server(config, 3).await;
        let pin = ServerPin::new();
        let store = CountingStore::new();
        let connector = tokio_rustls::TlsConnector::from(Arc::new(resuming_client(
            &client,
            pin.clone(),
            store.clone(),
        )))
        .early_data(true);
        for _ in 0..3 {
            assert_eq!(
                probe(&connector, addr).await,
                (Some(HandshakeKind::Full), false)
            );
        }
        assert_eq!(store.tickets(), 0, "the TD sent a ticket");
        assert_eq!(pin.checks(), 3);
        for s in srv.await.unwrap() {
            assert_eq!(s.kind, Some(HandshakeKind::Full), "{s:?}");
            assert_eq!(s.early_bytes, 0, "{s:?}");
            assert!(s.client_cert, "{s:?}");
        }
    }

    /// SECURITY (TD side): a genuine ticket for the same TD key, issued by a
    /// resuming twin, is not honoured: full handshake, no 0-RTT.
    #[tokio::test]
    async fn a_ticket_from_elsewhere_is_refused() {
        let server = TlsIdentity::generate("x0ne-infer").unwrap();
        let client = TlsIdentity::generate("x0ne-engine-proxy").unwrap();
        let (twin, _t) = probe_server(resuming_server(&server, client.spki_sha256()), 1).await;
        let (ours, srv) =
            probe_server(server_config(&server, client.spki_sha256()).unwrap(), 1).await;
        let store = CountingStore::new();
        let connector = tokio_rustls::TlsConnector::from(Arc::new(resuming_client(
            &client,
            ServerPin::new(),
            store.clone(),
        )))
        .early_data(true);
        probe(&connector, twin).await;
        assert!(store.tickets() >= 1, "the twin issued a ticket");
        assert_eq!(
            probe(&connector, ours).await,
            (Some(HandshakeKind::Full), false)
        );
        let seen = srv.await.unwrap();
        assert_eq!(seen[0].kind, Some(HandshakeKind::Full), "{seen:?}");
        assert_eq!(seen[0].early_bytes, 0, "{seen:?}");
    }

    // ── Downgrade: protocol version and cipher suite ────────────────────────

    const AES_128_GCM: u16 = 0x1301;
    const AES_256_GCM: u16 = 0x1302;

    fn u24(n: usize) -> [u8; 3] {
        [(n >> 16) as u8, (n >> 8) as u8, n as u8]
    }

    fn record(content_type: u8, version: [u8; 2], handshake_type: u8, body: &[u8]) -> Vec<u8> {
        let mut hs = vec![handshake_type];
        hs.extend(u24(body.len()));
        hs.extend(body);
        let mut rec = vec![content_type, version[0], version[1]];
        rec.extend((hs.len() as u16).to_be_bytes());
        rec.extend(hs);
        rec
    }

    /// What a ClientHello offers: cipher suites, supported_versions, session id.
    #[derive(Debug)]
    struct Offer {
        suites: Vec<u16>,
        versions: Vec<u16>,
        session_id: Vec<u8>,
    }

    fn parse_client_hello(rec: &[u8]) -> Offer {
        assert_eq!(rec[0], 0x16, "a handshake record");
        let hs = &rec[5..];
        assert_eq!(hs[0], 0x01, "a ClientHello");
        let mut at = 4 + 2 + 32;
        let sid_len = hs[at] as usize;
        let session_id = hs[at + 1..at + 1 + sid_len].to_vec();
        at += 1 + sid_len;
        let cs_len = u16::from_be_bytes([hs[at], hs[at + 1]]) as usize;
        let suites = hs[at + 2..at + 2 + cs_len]
            .chunks(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect();
        at += 2 + cs_len;
        at += 1 + hs[at] as usize;
        let ext_end = at + 2 + u16::from_be_bytes([hs[at], hs[at + 1]]) as usize;
        at += 2;
        let mut versions = Vec::new();
        while at < ext_end {
            let ty = u16::from_be_bytes([hs[at], hs[at + 1]]);
            let len = u16::from_be_bytes([hs[at + 2], hs[at + 3]]) as usize;
            let data = &hs[at + 4..at + 4 + len];
            if ty == 0x002b {
                versions = data[1..]
                    .chunks(2)
                    .map(|c| u16::from_be_bytes([c[0], c[1]]))
                    .collect();
            }
            at += 4 + len;
        }
        Offer {
            suites,
            versions,
            session_id,
        }
    }

    async fn read_record(tcp: &mut tokio::net::TcpStream) -> Vec<u8> {
        let mut head = [0u8; 5];
        tcp.read_exact(&mut head).await.unwrap();
        let mut rec = head.to_vec();
        rec.resize(5 + u16::from_be_bytes([head[3], head[4]]) as usize, 0);
        tcp.read_exact(&mut rec[5..]).await.unwrap();
        rec
    }

    /// The provider both ends build on: TLS 1.3 AES-GCM, nothing else.
    #[test]
    fn the_provider_offers_the_two_tls13_aes_gcm_suites_only() {
        let suites: Vec<u16> = provider()
            .cipher_suites
            .iter()
            .map(|s| u16::from(s.suite()))
            .collect();
        assert_eq!(suites, vec![AES_256_GCM, AES_128_GCM]);
    }

    /// SECURITY (downgrade): a server that answers a TLS 1.2 ServerHello gets
    /// no session, and our ClientHello offered TLS 1.3 and AES-GCM alone.
    #[tokio::test]
    async fn our_client_offers_tls13_aes_gcm_only_and_refuses_a_tls12_server() {
        let client = TlsIdentity::generate("x0ne-engine-proxy").unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut tcp, _) = listener.accept().await.unwrap();
            let offer = parse_client_hello(&read_record(&mut tcp).await);
            let mut sh = vec![0x03, 0x03];
            sh.extend([0x22; 32]);
            sh.push(offer.session_id.len() as u8);
            sh.extend(&offer.session_id);
            sh.extend([0xc0, 0x2b, 0x00]);
            tcp.write_all(&record(0x16, [0x03, 0x03], 0x02, &sh))
                .await
                .unwrap();
            let mut reply = Vec::new();
            let _ = tcp.read_to_end(&mut reply).await;
            (offer, reply)
        });
        let err = connect(addr, client_config(&client, ServerPin::new()).unwrap())
            .await
            .unwrap_err();
        let (offer, reply) = server.await.unwrap();
        assert_eq!(offer.versions, vec![0x0304], "{offer:?}");
        assert_eq!(offer.suites, vec![AES_256_GCM, AES_128_GCM], "{offer:?}");
        assert!(
            err.contains("ServerTlsVersionIsDisabledByOurConfig"),
            "{err}"
        );
        assert_eq!(
            reply.first(),
            Some(&0x15),
            "an alert, then nothing: {reply:?}"
        );
    }

    /// SECURITY (downgrade): a TLS 1.2 ClientHello, whatever its suites, gets
    /// a protocol_version alert and no ServerHello.
    #[tokio::test]
    async fn our_server_answers_a_tls12_client_hello_with_an_alert() {
        let server = TlsIdentity::generate("x0ne-infer").unwrap();
        let client = TlsIdentity::generate("x0ne-engine-proxy").unwrap();
        let (addr, srv) = echo_server(server_config(&server, client.spki_sha256()).unwrap()).await;
        let mut ch = vec![0x03, 0x03];
        ch.extend([0x11; 32]);
        ch.push(0);
        ch.extend([0x00, 0x06, 0xc0, 0x2b, 0xc0, 0x2c, 0xcc, 0xa9, 0x01, 0x00]);
        let ext: &[u8] = &[
            0x00, 0x0a, 0x00, 0x04, 0x00, 0x02, 0x00, 0x17, // supported_groups: P-256
            0x00, 0x0b, 0x00, 0x02, 0x01, 0x00, // ec_point_formats
            0x00, 0x0d, 0x00, 0x04, 0x00, 0x02, 0x04, 0x03, // ecdsa_secp256r1_sha256
        ];
        ch.extend((ext.len() as u16).to_be_bytes());
        ch.extend(ext);
        let mut tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
        tcp.write_all(&record(0x16, [0x03, 0x01], 0x01, &ch))
            .await
            .unwrap();
        let reply = read_record(&mut tcp).await;
        // Alert record, fatal (2), protocol_version (70).
        assert_eq!(reply, vec![0x15, 0x03, 0x03, 0x00, 0x02, 0x02, 70]);
        drop(tcp);
        srv.abort();
    }

    fn chacha_only() -> Arc<CryptoProvider> {
        Arc::new(CryptoProvider {
            cipher_suites: vec![rustls::crypto::ring::cipher_suite::TLS13_CHACHA20_POLY1305_SHA256],
            ..rustls::crypto::ring::default_provider()
        })
    }

    /// SECURITY (downgrade): a TD that only speaks a suite off the list is
    /// refused by our client, whatever key it holds.
    #[tokio::test]
    async fn our_client_refuses_a_server_with_a_suite_off_the_list() {
        let server = TlsIdentity::generate("x0ne-infer").unwrap();
        let client = TlsIdentity::generate("x0ne-engine-proxy").unwrap();
        let provider = chacha_only();
        let verifier = Arc::new(PinnedClient {
            expected: client.spki_sha256(),
            algs: provider.signature_verification_algorithms,
            hints: Vec::new(),
        });
        let config = rustls::ServerConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&rustls::version::TLS13])
            .unwrap()
            .with_client_cert_verifier(verifier)
            .with_single_cert(vec![server.cert.clone()], server.private_key())
            .unwrap();
        let (addr, srv) = echo_server(Arc::new(config)).await;
        let err = connect(addr, client_config(&client, ServerPin::new()).unwrap())
            .await
            .unwrap_err();
        assert!(err.contains("alert"), "{err}");
        srv.abort();
    }

    /// SECURITY (downgrade): a client that only offers a suite off the list
    /// gets no session from our TD, even with the pinned client key.
    #[tokio::test]
    async fn our_server_refuses_a_client_with_a_suite_off_the_list() {
        let server = TlsIdentity::generate("x0ne-infer").unwrap();
        let client = TlsIdentity::generate("x0ne-engine-proxy").unwrap();
        let acceptor =
            tokio_rustls::TlsAcceptor::from(server_config(&server, client.spki_sha256()).unwrap());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let srv = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            acceptor
                .accept(tcp)
                .await
                .map(|_| ())
                .map_err(|e| e.to_string())
        });
        let provider = chacha_only();
        let config = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .unwrap()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(PinOnFirstUse {
                pin: ServerPin::new(),
                algs: provider.signature_verification_algorithms,
            }))
            .with_client_auth_cert(vec![client.cert.clone()], client.private_key())
            .unwrap();
        assert!(connect(addr, config).await.is_err());
        let err = srv.await.unwrap().unwrap_err();
        assert!(err.contains("NoCipherSuitesInCommon"), "{err}");
    }
}
