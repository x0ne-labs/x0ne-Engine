//! The measuring client of the real-hardware run (`examples/cloud_probe.rs`,
//! `scripts/e2e-cloud.sh probe`). Every case here is first played against the
//! fake TD (`cloud_e2e_tests.rs`), so a VM minute only confirms what the bench
//! already showed. Not a product path: built for tests and `--features probe`.

use crate::cloud_serve::{QuoteLimits, SERVER_TIMING};
use crate::remote_verify::TdClient;
use crate::tls;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub type Stream = tokio_rustls::client::TlsStream<tokio::net::TcpStream>;

const IO_TIMEOUT: Duration = Duration::from_secs(20);

#[must_use]
pub fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// One TLS connection as the relay opens it, nothing sent yet.
pub struct Raw {
    pub stream: Stream,
    pub exporter: [u8; 32],
    pub tcp_ms: f64,
    pub tls_ms: f64,
    pub resumed: bool,
}

/// Opens one connection with `cfg` (the relay's identity or another).
///
/// # Errors
/// TCP or TLS failure, as text.
pub async fn open(
    cfg: Arc<rustls::ClientConfig>,
    endpoint: &str,
    host: &str,
    nonce: &[u8; 32],
) -> Result<Raw, String> {
    let t0 = Instant::now();
    let tcp = tokio::time::timeout(IO_TIMEOUT, tokio::net::TcpStream::connect(endpoint))
        .await
        .map_err(|_| "tcp timeout".to_string())?
        .map_err(|e| format!("tcp: {e}"))?;
    let _ = tcp.set_nodelay(true);
    let tcp_ms = ms(t0.elapsed());
    let t1 = Instant::now();
    let name =
        rustls::pki_types::ServerName::try_from(host.to_string()).map_err(|e| e.to_string())?;
    let mut stream = tokio::time::timeout(
        IO_TIMEOUT,
        tokio_rustls::TlsConnector::from(cfg).connect(name, tcp),
    )
    .await
    .map_err(|_| "tls timeout".to_string())?
    .map_err(|e| format!("TLS handshake: {e}"))?;
    // A TLS 1.3 server that refuses our certificate says so after the
    // handshake completes on our side: one read surfaces the alert.
    let _ = stream.get_mut().1.process_new_packets();
    let tls_ms = ms(t1.elapsed());
    let conn = stream.get_ref().1;
    let resumed = matches!(conn.handshake_kind(), Some(rustls::HandshakeKind::Resumed));
    let exporter = tls::exporter(conn, nonce).map_err(|e| format!("exporter: {e}"))?;
    Ok(Raw {
        stream,
        exporter,
        tcp_ms,
        tls_ms,
        resumed,
    })
}

/// One HTTP answer.
#[derive(Debug, Clone)]
pub struct Answer {
    pub status: u16,
    pub retry_after: Option<String>,
    pub server_timing: Option<String>,
    pub body: Vec<u8>,
    pub ms: f64,
}

impl Answer {
    /// The in-TD quote time from `Server-Timing: quote;dur=<ms>`.
    #[must_use]
    pub fn quote_ms(&self) -> Option<f64> {
        self.server_timing.as_deref().and_then(quote_ms)
    }
}

/// `quote;dur=12.5` → 12.5.
#[must_use]
pub fn quote_ms(server_timing: &str) -> Option<f64> {
    server_timing.split(',').find_map(|m| {
        let mut parts = m.trim().split(';');
        (parts.next()? == "quote").then_some(())?;
        parts.find_map(|p| p.trim().strip_prefix("dur=")?.parse().ok())
    })
}

fn header(head: &str, name: &str) -> Option<String> {
    head.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        k.trim()
            .eq_ignore_ascii_case(name)
            .then(|| v.trim().to_string())
    })
}

/// One HTTP/1.1 request on an open stream.
///
/// # Errors
/// I/O failure or a closed stream, as text.
pub async fn request(
    raw: &mut Raw,
    method: &str,
    path: &str,
    host: &str,
    body: Option<&[u8]>,
) -> Result<Answer, String> {
    let t = Instant::now();
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {host}\r\n");
    if let Some(b) = body {
        req.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            b.len()
        ));
    }
    req.push_str("\r\n");
    raw.stream
        .write_all(req.as_bytes())
        .await
        .map_err(|e| format!("write: {e}"))?;
    if let Some(b) = body {
        raw.stream
            .write_all(b)
            .await
            .map_err(|e| format!("write: {e}"))?;
    }
    raw.stream
        .flush()
        .await
        .map_err(|e| format!("flush: {e}"))?;
    let mut buf = Vec::new();
    let mut chunk = vec![0u8; 16384];
    let head_end = loop {
        let n = tokio::time::timeout(Duration::from_secs(120), raw.stream.read(&mut chunk))
            .await
            .map_err(|_| "read timeout".to_string())?
            .map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            return Err(format!("closed after {} bytes", buf.len()));
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let len: usize = header(&head, "content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    while buf.len() < head_end + len {
        let n = raw
            .stream
            .read(&mut chunk)
            .await
            .map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    Ok(Answer {
        status,
        retry_after: header(&head, "retry-after"),
        server_timing: header(&head, SERVER_TIMING),
        body: buf[head_end..].to_vec(),
        ms: ms(t.elapsed()),
    })
}

pub const SESSION: &str = "/v2/attestation/session";

/// C3, attack: `n` connections opened first (no request, so the TLS
/// handshakes do not space the requests out), then one session quote asked
/// on each at the same instant.
pub struct Burst {
    pub answers: Vec<Result<Answer, String>>,
    /// From the first request sent to the last answer.
    pub window: Duration,
}

/// Counts of a burst, by answer.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Tally {
    pub quoted: usize,
    /// 429 "too many quotes": the global bucket.
    pub rate: usize,
    /// 429 "one session quote per connection".
    pub per_connection: usize,
    /// 503: the queue behind the quote in flight is full.
    pub busy: usize,
    pub other: usize,
    pub errors: usize,
}

impl Burst {
    #[must_use]
    pub fn tally(&self) -> Tally {
        let mut t = Tally::default();
        for a in &self.answers {
            match a {
                Ok(a) if a.status == 200 => t.quoted += 1,
                Ok(a) if a.status == 429 && a.body.windows(9).any(|w| w == b"too many ") => {
                    t.rate += 1;
                }
                Ok(a) if a.status == 429 => t.per_connection += 1,
                Ok(a) if a.status == 503 => t.busy += 1,
                Ok(_) => t.other += 1,
                Err(_) => t.errors += 1,
            }
        }
        t
    }

    /// Quotes the bucket can grant in this window, at most: full bucket
    /// plus what refilled, plus one for the refill caught at the edge.
    #[must_use]
    pub fn bound(&self, limits: &QuoteLimits) -> usize {
        let every = limits.refill_every.as_secs_f64();
        let refilled = if every > 0.0 {
            (self.window.as_secs_f64() / every).ceil()
        } else {
            f64::INFINITY
        };
        // Bounded by the window, seconds: no truncation that matters.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let refilled = refilled.min(1e6) as usize;
        usize::try_from(limits.burst).unwrap_or(usize::MAX) + refilled + 1
    }

    /// C3 holds: never more quotes than the bound, and past it the TD said
    /// 429 "too many quotes". The text says what was seen either way.
    ///
    /// # Errors
    /// The breach, in words.
    pub fn verdict(&self, limits: &QuoteLimits) -> Result<String, String> {
        let t = self.tally();
        let n = self.answers.len();
        let bound = self.bound(limits);
        let seen = format!(
            "{n} at once: {} quoted, {} 429 rate, {} 503 busy, {} other, {} errors in {:.0} ms (bound {bound})",
            t.quoted,
            t.rate,
            t.busy,
            t.other,
            t.errors,
            ms(self.window)
        );
        if t.quoted > bound {
            return Err(format!("NOT BOUNDED: {seen}"));
        }
        if n > bound && t.rate == 0 {
            return Err(format!("NO 429 past the bound: {seen}"));
        }
        if t.errors > 0 || t.other > 0 || t.per_connection > 0 {
            return Err(format!("unexpected answers: {seen}"));
        }
        Ok(seen)
    }
}

/// See [`Burst`].
pub async fn quote_burst(
    cfg: Arc<rustls::ClientConfig>,
    endpoint: &str,
    host: &str,
    nonce: [u8; 32],
    n: usize,
) -> Burst {
    let mut opening = Vec::with_capacity(n);
    for _ in 0..n {
        let (cfg, ep, h) = (cfg.clone(), endpoint.to_string(), host.to_string());
        opening.push(tokio::spawn(
            async move { open(cfg, &ep, &h, &nonce).await },
        ));
    }
    let mut answers = Vec::with_capacity(n);
    let mut raws = Vec::with_capacity(n);
    for o in opening {
        match o.await.unwrap_or_else(|e| Err(format!("task: {e}"))) {
            Ok(r) => raws.push(r),
            Err(e) => answers.push(Err(format!("open: {e}"))),
        }
    }
    let go = Arc::new(tokio::sync::Barrier::new(raws.len() + 1));
    let mut asking = Vec::with_capacity(raws.len());
    for mut raw in raws {
        let (go, ep) = (go.clone(), endpoint.to_string());
        asking.push(tokio::spawn(async move {
            go.wait().await;
            request(&mut raw, "GET", SESSION, &ep, None).await
        }));
    }
    go.wait().await;
    let t = Instant::now();
    for a in asking {
        answers.push(a.await.unwrap_or_else(|e| Err(format!("task: {e}"))));
    }
    Burst {
        answers,
        window: t.elapsed(),
    }
}

/// C3, honest relay: one new connection and one session quote every
/// `every`. Every answer should be 200 once the bucket holds a token.
pub async fn quote_nominal(
    cfg: Arc<rustls::ClientConfig>,
    endpoint: &str,
    host: &str,
    nonce: [u8; 32],
    n: usize,
    every: Duration,
) -> Vec<Result<Answer, String>> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if i > 0 {
            tokio::time::sleep(every).await;
        }
        out.push(match open(cfg.clone(), endpoint, host, &nonce).await {
            Ok(mut raw) => request(&mut raw, "GET", SESSION, endpoint, None).await,
            Err(e) => Err(format!("open: {e}")),
        });
    }
    out
}

/// The quote time inside the TD, one new connection per sample, `every`
/// apart (under the bucket's refill): the `Server-Timing` of each 200, and
/// the round trip seen here.
pub async fn quote_times(
    cfg: Arc<rustls::ClientConfig>,
    endpoint: &str,
    host: &str,
    nonce: [u8; 32],
    n: usize,
    every: Duration,
) -> (Vec<f64>, Vec<f64>) {
    let mut inside = Vec::new();
    let mut round_trip = Vec::new();
    for a in quote_nominal(cfg, endpoint, host, nonce, n, every)
        .await
        .into_iter()
        .flatten()
    {
        if a.status == 200 {
            round_trip.push(a.ms);
            if let Some(q) = a.quote_ms() {
                inside.push(q);
            }
        }
    }
    (inside, round_trip)
}

/// Accepts any server certificate: the direct cases test what the TD
/// answers to a client that is not the relay, not who the server is.
#[derive(Debug)]
struct AnyServer(rustls::crypto::WebPkiSupportedAlgorithms);

impl rustls::client::danger::ServerCertVerifier for AnyServer {
    fn verify_server_cert(
        &self,
        _: &rustls::pki_types::CertificateDer<'_>,
        _: &[rustls::pki_types::CertificateDer<'_>],
        _: &rustls::pki_types::ServerName<'_>,
        _: &[u8],
        _: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        _: &[u8],
        _: &rustls::pki_types::CertificateDer<'_>,
        _: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Err(rustls::Error::General("TLS 1.2 is not offered".into()))
    }
    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.0)
    }
    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0.supported_schemes()
    }
}

/// What the TD answered to clients that are not the relay. Each field is
/// `refused: ...` or `ANSWERED ...`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Direct {
    /// Plain HTTP on the TLS port.
    pub plain_http: String,
    /// TLS 1.3 with no client certificate.
    pub anonymous_tls: String,
    /// TLS 1.3 with a certificate the config does not name.
    pub foreign_certificate: String,
}

impl Direct {
    /// No case got an HTTP answer.
    #[must_use]
    pub fn all_refused(&self) -> bool {
        [
            &self.plain_http,
            &self.anonymous_tls,
            &self.foreign_certificate,
        ]
        .iter()
        .all(|s| s.starts_with("refused"))
    }
}

const ASK: &[u8] = b"GET /v2/attestation/session HTTP/1.1\r\nHost: td\r\nConnection: close\r\n\r\n";

/// Reads until close or timeout; an HTTP status line back is an answer.
async fn verdict_of<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin>(mut s: S) -> String {
    if let Err(e) = s.write_all(ASK).await {
        return format!("refused: write: {e}");
    }
    let mut reply = Vec::new();
    let read = tokio::time::timeout(IO_TIMEOUT, s.read_to_end(&mut reply)).await;
    if reply.starts_with(b"HTTP/") {
        let line = reply.split(|b| *b == b'\r').next().unwrap_or_default();
        return format!("ANSWERED {}", String::from_utf8_lossy(line));
    }
    match read {
        Err(_) => format!(
            "refused: no answer in {} s, {} bytes",
            IO_TIMEOUT.as_secs(),
            reply.len()
        ),
        Ok(Err(e)) => format!("refused: {e}"),
        Ok(Ok(_)) => format!("refused: closed, {} bytes and no HTTP", reply.len()),
    }
}

/// The three ways to reach the TD without the relay's key.
pub async fn direct(endpoint: &str, host: &str) -> Direct {
    let plain_http = match tokio::net::TcpStream::connect(endpoint).await {
        Ok(tcp) => verdict_of(tcp).await,
        Err(e) => format!("refused: tcp: {e}"),
    };
    let provider = rustls::crypto::ring::default_provider();
    let algs = provider.signature_verification_algorithms;
    let anonymous = rustls::ClientConfig::builder_with_provider(Arc::new(provider))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map(|b| {
            let mut c = b
                .dangerous()
                .with_custom_certificate_verifier(Arc::new(AnyServer(algs)))
                .with_no_client_auth();
            c.alpn_protocols = vec![tls::ALPN_HTTP1.to_vec()];
            c
        });
    let anonymous_tls = match anonymous {
        Ok(cfg) => tls_case(Arc::new(cfg), endpoint, host).await,
        Err(e) => format!("refused: config: {e}"),
    };
    let foreign = tls::TlsIdentity::generate("x0ne-engine-proxy")
        .and_then(|id| tls::client_config(&id, tls::ServerPin::new()));
    let foreign_certificate = match foreign {
        Ok(cfg) => tls_case(Arc::new(cfg), endpoint, host).await,
        Err(e) => format!("refused: config: {e}"),
    };
    Direct {
        plain_http,
        anonymous_tls,
        foreign_certificate,
    }
}

async fn tls_case(cfg: Arc<rustls::ClientConfig>, endpoint: &str, host: &str) -> String {
    let tcp = match tokio::net::TcpStream::connect(endpoint).await {
        Ok(t) => t,
        Err(e) => return format!("refused: tcp: {e}"),
    };
    let Ok(name) = rustls::pki_types::ServerName::try_from(host.to_string()) else {
        return "refused: bad host".into();
    };
    match tokio::time::timeout(
        IO_TIMEOUT,
        tokio_rustls::TlsConnector::from(cfg).connect(name, tcp),
    )
    .await
    {
        Err(_) => "refused: handshake timeout".into(),
        Ok(Err(e)) => format!("refused: handshake: {e}"),
        Ok(Ok(s)) => verdict_of(s).await,
    }
}

/// Completions measured on the product path (`TdClient`, checked
/// connections), `parallel` at once.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Throughput {
    pub parallel: usize,
    pub statuses: Vec<u16>,
    pub completion_tokens: u64,
    /// Wall time from the first request to the last answer.
    pub secs: f64,
    /// Each request's own time.
    pub each_secs: Vec<f64>,
}

impl Throughput {
    #[must_use]
    pub fn tokens_per_s(&self) -> f64 {
        if self.secs > 0.0 {
            // Token counts: far below 2^52.
            #[allow(clippy::cast_precision_loss)]
            let t = self.completion_tokens as f64;
            t / self.secs
        } else {
            0.0
        }
    }

    /// Dollars per million completion tokens at this rate and this VM price.
    #[must_use]
    pub fn usd_per_million_tokens(&self, usd_per_hour: f64) -> Option<f64> {
        let rate = self.tokens_per_s();
        (rate > 0.0).then(|| usd_per_hour / 3600.0 / rate * 1e6)
    }
}

/// `parallel` completions of `body` at once through the same checked session.
pub async fn chats(client: Arc<TdClient>, body: serde_json::Value, parallel: usize) -> Throughput {
    let t = Instant::now();
    let mut running = Vec::with_capacity(parallel);
    for _ in 0..parallel {
        let (c, b) = (client.clone(), body.clone());
        running.push(tokio::spawn(async move {
            let t = Instant::now();
            let r = c.chat(&b).await;
            (r, t.elapsed().as_secs_f64())
        }));
    }
    let mut statuses = Vec::new();
    let mut each_secs = Vec::new();
    let mut completion_tokens = 0;
    for h in running {
        match h.await {
            Ok((Ok(a), took)) => {
                statuses.push(a.status.as_u16());
                each_secs.push(took);
                if a.status.is_success() {
                    completion_tokens += a.body["usage"]["completion_tokens"].as_u64().unwrap_or(0);
                }
            }
            Ok((Err(_), took)) => {
                statuses.push(0);
                each_secs.push(took);
            }
            Err(_) => statuses.push(0),
        }
    }
    Throughput {
        parallel,
        statuses,
        completion_tokens,
        secs: t.elapsed().as_secs_f64(),
        each_secs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_timing_is_read_back() {
        assert_eq!(quote_ms("quote;dur=12.5"), Some(12.5));
        assert_eq!(quote_ms("lock;dur=3, quote;desc=\"x\";dur=7"), Some(7.0));
        assert_eq!(quote_ms("other;dur=1"), None);
    }

    #[test]
    fn cost_per_token_is_price_over_rate() {
        let t = Throughput {
            parallel: 1,
            statuses: vec![200],
            completion_tokens: 360,
            secs: 1.0,
            each_secs: vec![1.0],
        };
        // 3.6 $/h = 0.001 $/s; 360 tokens/s: 1e6 tokens in 2777.8 s.
        let usd = t.usd_per_million_tokens(3.6).unwrap();
        assert!((usd - 2.7778).abs() < 1e-3, "{usd}");
        let none = Throughput {
            completion_tokens: 0,
            ..t
        };
        assert_eq!(none.usd_per_million_tokens(3.6), None);
    }

    #[test]
    fn the_bound_counts_the_bucket_and_its_refill() {
        let limits = QuoteLimits::default();
        let b = Burst {
            answers: vec![],
            window: Duration::from_millis(1500),
        };
        // burst 10 + 2 refilled + 1 at the edge.
        assert_eq!(b.bound(&limits), 13);
    }
}
