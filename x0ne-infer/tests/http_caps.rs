//! Every attestation fetch caps the body as it arrives. A server on the
//! loopback answers 200 with no length, sends more than any cap, then holds
//! the connection open: a client that buffers the whole body first never
//! returns, one that counts the chunks refuses at once.

use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// More than every cap of these fetches (the largest is 4 MiB).
const SENT: usize = 5 << 20;

/// A base URL whose every request gets `SENT` bytes and no end.
async fn endless() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
            tokio::spawn(async move {
                let mut req = Vec::new();
                let mut buf = [0u8; 4096];
                while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                    match sock.read(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => req.extend_from_slice(&buf[..n]),
                    }
                }
                let head = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                            TCB-Info-Issuer-Chain: x\r\nConnection: close\r\n\r\n";
                if sock.write_all(head.as_bytes()).await.is_err() {
                    return;
                }
                let _ = sock.write_all(&vec![b' '; SENT]).await;
                tokio::time::sleep(Duration::from_secs(3600)).await;
            });
        }
    });
    format!("http://{addr}")
}

/// `fut` ends within five seconds with an error that names the cap.
async fn refused_at_the_cap<T, E: std::fmt::Display>(
    what: &str,
    fut: impl std::future::Future<Output = Result<T, E>>,
) {
    let out = tokio::time::timeout(Duration::from_secs(5), fut)
        .await
        .unwrap_or_else(|_| panic!("{what}: still reading a body past its cap after 5 s"));
    let Err(err) = out else {
        panic!("{what}: a body larger than the cap was accepted");
    };
    let err = err.to_string();
    assert!(err.contains("cap"), "{what}: {err}");
}

fn client() -> reqwest::Client {
    reqwest::Client::builder().build().unwrap()
}

#[tokio::test]
async fn the_pcs_collateral_is_capped_as_it_arrives() {
    let base = endless().await;
    let url = format!("{base}/tcb");
    refused_at_the_cap(
        "remote_verify::fetch_pcs",
        x0ne_infer::remote_verify::fetch_pcs(&client(), &url),
    )
    .await;
    refused_at_the_cap(
        "remote_verify::fetch_pcs_der",
        x0ne_infer::remote_verify::fetch_pcs_der(&client(), &url),
    )
    .await;
}

#[tokio::test]
async fn the_google_endorsement_is_capped_as_it_arrives() {
    let base = endless().await;
    refused_at_the_cap(
        "gce_endorsement::fetch",
        x0ne_infer::gce_endorsement::fetch(&client(), &base, &[0u8; 48]),
    )
    .await;
}

#[tokio::test]
async fn the_nras_answers_are_capped_as_they_arrive() {
    let base = endless().await;
    let url = format!("{base}/nras");
    let req = x0ne_infer::nras_client::AttestRequest {
        nonce: "0".repeat(64),
        evidence_list: Vec::new(),
        arch: "HOPPER".into(),
        claims_version: "3.0".into(),
    };
    use x0ne_infer::nras_http as nras;
    refused_at_the_cap("nras_http::attest_gpu", nras::attest_gpu(&url, &req, None)).await;
    refused_at_the_cap("nras_http::fetch_jwks", nras::fetch_jwks(&url)).await;
    refused_at_the_cap(
        "nras_http::fetch_pcs_collateral",
        nras::fetch_pcs_collateral(&url),
    )
    .await;
    refused_at_the_cap("nras_http::fetch_pcs_der", nras::fetch_pcs_der(&url)).await;
}
