//! The TD's end of the caller-sealed path (docs/sdk-e2e.md): an OHTTP key
//! born at boot, its key document under a quote over that key, the caller's
//! nonce and this boot's binding (the receipt key, the boot nonce, the TLS
//! key, the session config, whose bytes the document carries), over the
//! GPUs' evidence too on a GPU-attested session, and the sealed chat, which
//! opens a request, answers it through the attested chat handler
//! (`server::chat_as`: padding, turn cadence, receipts) and seals the answer;
//! a chunked request (`message/ohttp-chunked-req`, draft-ietf-ohai-chunked-
//! ohttp-08) is answered as a chunked response at the session's cadence
//! (`crate::e2e_stream`, docs/sdk-e2e.md step 6).
//!
//! No log macro in this file: it holds the question and the answer in clear.
//! Refusals are counted by class (`Counts`), never described: every request
//! refused before its answer gets the same status, body and delay.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use x0ne_sdk::attest::{GpuEvidence, KeyDocument, KEY_SCHEMA, MAX_GPU_EVIDENCE_BYTES};
use x0ne_sdk::e2e::{self, Gateway, Opened, ReplayGuard, Responder, StreamResponder};
use x0ne_sdk::ohttp::{MEDIA_TYPE_CHUNKED_RESPONSE, MEDIA_TYPE_RESPONSE, RESPONSE_NONCE_LEN};

/// The body of every sealed request refused before its answer.
pub const REFUSED: &str = "sealed request refused";
/// The body of an answer that ran but could not be sealed: never its text.
pub const UNSEALED: &str = "sealed answer unavailable";

/// Counts by class, for tests and the operator; nothing of a request.
#[derive(Default)]
pub struct Counts {
    /// Requests opened and admitted: they reached the chat handler.
    pub opened: AtomicU64,
    /// Not a sealed request to this key, or not a chat request once opened.
    pub refused_open: AtomicU64,
    /// Opened, then refused by the replay guard (answered already, or out of window).
    pub refused_replay: AtomicU64,
    /// Answers sealed and sent.
    pub sealed: AtomicU64,
    /// Answers that ran and could not be sealed: [`UNSEALED`] went instead.
    pub unsealed: AtomicU64,
    /// Chunked requests opened and admitted, counted in `opened` too: their
    /// answer went as a chunked response.
    pub streams: AtomicU64,
}

/// The gateway of one boot: its key, the boot binding its documents name,
/// its replay memory, its counts.
pub struct E2e {
    gateway: Gateway,
    /// The receipt key, the boot nonce, the TLS key and the session config's
    /// digest, as the boot quote binds them.
    boot: crate::binding::BindingV2,
    /// The session config's exact bytes, whose SHA-256 `boot` names.
    session_config: Vec<u8>,
    replay: Mutex<ReplayGuard>,
    backing: crate::harden::Backing,
    pub counts: Counts,
}

/// The gateway key's scalar in `harden::SecretBytes`, as the model key's recipient
/// key (`model_key::Recipient`): `memfd_secret`, else `mlock`.
struct Locked(crate::harden::SecretBytes);

impl x0ne_sdk::hpke::SecretScalar for Locked {
    fn expose(&self) -> &[u8] {
        self.0.expose()
    }
}

/// Why a sealed request got no answer; the caller is told none of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    Open,
    Replay,
}

impl E2e {
    /// The gateway whose key `seed` derives (RFC 9180 `DeriveKeyPair`), its
    /// private scalar in `harden::SecretBytes`, out of swap and core dumps,
    /// wiped on drop. Its
    /// key documents name `boot`, this boot's binding, and carry
    /// `session_config`, the exact bytes whose SHA-256 `boot` names.
    ///
    /// # Errors
    /// `session_config` is not the config `boot` names; neither
    /// `memfd_secret` nor `mlock` works (Linux).
    pub fn from_seed(
        seed: &[u8; 32],
        boot: crate::binding::BindingV2,
        session_config: &[u8],
    ) -> std::io::Result<Self> {
        if crate::cloud_config::sha256(session_config) != boot.config_sha256 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "the session config is not the one the boot binding names",
            ));
        }
        let mut backing = crate::harden::Backing::Heap;
        let gateway = Gateway::from_seed_held_by(seed, |scalar| {
            let held = crate::harden::SecretBytes::from_slice(scalar)?;
            backing = held.backing();
            Ok::<_, std::io::Error>(Locked(held))
        })?;
        Ok(Self {
            gateway,
            boot,
            session_config: session_config.to_vec(),
            replay: Mutex::new(ReplayGuard::default()),
            backing,
            counts: Counts::default(),
        })
    }

    /// The boot binding's fields the SDK hashes besides the receipt key.
    fn sdk_boot(&self) -> e2e::Boot {
        e2e::Boot {
            nonce: self.boot.session_nonce,
            tls_spki_sha256: self.boot.tls_spki_sha256,
            config_sha256: self.boot.config_sha256,
        }
    }

    /// Where the private scalar lives: never [`crate::harden::Backing::Heap`] on Linux.
    #[must_use]
    pub fn key_backing(&self) -> crate::harden::Backing {
        self.backing
    }

    /// The key configuration the document carries and the binding hashes.
    #[must_use]
    pub fn key_config(&self) -> Vec<u8> {
        self.gateway.key_config()
    }

    /// The quote's `report_data` for the caller's `nonce`: this key, the
    /// receipt key and this boot's binding under the e2e label
    /// (`e2e::key_binding`), never chosen bytes.
    #[must_use]
    pub fn report_data(&self, nonce: &[u8; 32]) -> [u8; 64] {
        e2e::key_binding(
            &self.key_config(),
            nonce,
            &self.boot.session_pubkey,
            &self.sdk_boot(),
        )
    }

    /// The nonce this TD's GPUs answer for the key document of the caller's
    /// `nonce` (`e2e::gpu_nonce`): their reports are read for it, then bound.
    #[must_use]
    pub fn gpu_nonce(&self, nonce: &[u8; 32]) -> [u8; 32] {
        e2e::gpu_nonce(&self.key_config(), nonce, &self.boot.session_pubkey)
    }

    /// The quote's `report_data` on a GPU-attested session: this key, the
    /// receipt key, this boot's binding, `nonce` and each GPU's report and
    /// certificate chain, as decoded bytes (`e2e::key_binding_gpu`).
    ///
    /// # Errors
    /// Evidence the caller's SDK refuses: none, more than `e2e::MAX_GPUS`,
    /// not base64, or a field over `attest::MAX_GPU_EVIDENCE_BYTES`.
    pub fn report_data_gpu(
        &self,
        nonce: &[u8; 32],
        gpus: &[crate::nras_client::GpuEvidence],
    ) -> Result<[u8; 64], String> {
        if gpus.is_empty() || gpus.len() > e2e::MAX_GPUS {
            return Err(format!("{} GPU evidence entries", gpus.len()));
        }
        let decode = |s: &str| {
            base64::engine::general_purpose::STANDARD
                .decode(s)
                .map_err(|e| e.to_string())
                .and_then(|b| {
                    (b.len() <= MAX_GPU_EVIDENCE_BYTES)
                        .then_some(b)
                        .ok_or_else(|| "GPU evidence over the cap".to_string())
                })
        };
        let decoded = gpus
            .iter()
            .map(|g| Ok((decode(&g.evidence)?, decode(&g.certificate)?)))
            .collect::<Result<Vec<_>, String>>()?;
        let views: Vec<(&[u8], &[u8])> = decoded
            .iter()
            .map(|(r, c)| (r.as_slice(), c.as_slice()))
            .collect();
        Ok(e2e::key_binding_gpu(
            &self.key_config(),
            nonce,
            &self.boot.session_pubkey,
            &self.sdk_boot(),
            &views,
        ))
    }

    /// [`Self::document`] of a quote over [`Self::report_data_gpu`], with the
    /// GPUs' evidence that quote bound.
    #[must_use]
    pub fn document_gpu(
        &self,
        quote: &[u8],
        ccel: &[u8],
        gpus: Vec<crate::nras_client::GpuEvidence>,
    ) -> KeyDocument {
        KeyDocument {
            gpu_evidence_list: Some(
                gpus.into_iter()
                    .map(|g| GpuEvidence {
                        evidence: g.evidence,
                        certificate: g.certificate,
                    })
                    .collect(),
            ),
            ..self.document(quote, ccel)
        }
    }

    /// The key document of one quote taken over [`Self::report_data`], with
    /// this boot's binding and session config, which the quote bound, and the
    /// TD's event log (`ccel`), which the caller replays into the quote's
    /// RTMRs: the log the attestation document carries for the relay.
    #[must_use]
    pub fn document(&self, quote: &[u8], ccel: &[u8]) -> KeyDocument {
        let b64 = |b: &[u8]| base64::engine::general_purpose::STANDARD.encode(b);
        let hex = crate::cloud_config::hex;
        KeyDocument {
            schema: KEY_SCHEMA.into(),
            key_config_b64: b64(&self.key_config()),
            session_pubkey_hex: hex(&self.boot.session_pubkey),
            session_config_b64: b64(&self.session_config),
            boot_nonce_hex: hex(&self.boot.session_nonce),
            tls_spki_sha256_hex: hex(&self.boot.tls_spki_sha256),
            quote_b64: b64(quote),
            ccel_b64: b64(ccel),
            gpu_evidence_list: None,
            nras_answer: None,
            nras_jwks: None,
            nras_ocsp: None,
        }
    }

    /// [`Self::document_gpu`] with NVIDIA's verdict on that evidence, as
    /// [`crate::attest_v2::TdEvidence::nras`] fetched it: the caller checks
    /// it under its own pins, as the relay checks a session's.
    #[must_use]
    pub fn with_nras(doc: KeyDocument, verdict: crate::attest_v2::NrasVerdict) -> KeyDocument {
        KeyDocument {
            nras_answer: Some(verdict.answer.to_string()),
            nras_jwks: Some(verdict.jwks.to_string()),
            ..doc
        }
    }

    /// Opens `sealed` and admits it once, at the gateway's `now`.
    ///
    /// # Errors
    /// [`Refusal::Open`] or [`Refusal::Replay`], counted.
    pub fn admit(&self, sealed: &[u8], now: u64) -> Result<(Opened, Responder), Refusal> {
        let opened = self.gateway.open(sealed);
        self.admitted(opened, Responder::replay_id, now)
    }

    /// [`Self::admit`] for a chunked request. One replay memory for both: an
    /// encapsulated key is answered once, whole or streamed.
    ///
    /// # Errors
    /// [`Refusal::Open`] or [`Refusal::Replay`], counted.
    pub fn admit_stream(
        &self,
        sealed: &[u8],
        now: u64,
    ) -> Result<(Opened, StreamResponder), Refusal> {
        let opened = self.gateway.open_stream(sealed);
        self.admitted(opened, StreamResponder::replay_id, now)
    }

    fn admitted<R>(
        &self,
        opened: Result<(Opened, R), e2e::E2eError>,
        replay_id: fn(&R) -> [u8; 32],
        now: u64,
    ) -> Result<(Opened, R), Refusal> {
        let Ok((opened, responder)) = opened else {
            self.counts.refused_open.fetch_add(1, Ordering::SeqCst);
            return Err(Refusal::Open);
        };
        let admitted = self
            .replay
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .admit(replay_id(&responder), opened.sent_at, now);
        if admitted.is_err() {
            self.counts.refused_replay.fetch_add(1, Ordering::SeqCst);
            return Err(Refusal::Replay);
        }
        self.counts.opened.fetch_add(1, Ordering::SeqCst);
        Ok((opened, responder))
    }
}

/// A sealed request refused before its answer: 400, [`REFUSED`], one second
/// after `asked` as a whole answer refused before its turn (`server::chat_as`).
pub async fn refused(asked: tokio::time::Instant) -> Response {
    at_floor(asked).await;
    (
        StatusCode::BAD_REQUEST,
        axum::Json(serde_json::json!({ "error": REFUSED })),
    )
        .into_response()
}

/// Sleeps to the floor of every refusal: one second after `asked`.
async fn at_floor(asked: tokio::time::Instant) {
    let floor = crate::stream::TICK
        .saturating_mul(u32::try_from(crate::padding::MIN_TICKS).unwrap_or(u32::MAX));
    tokio::time::sleep_until(asked + floor).await;
}

/// The padded JSON error sealed in place of an answer that cannot be.
fn unsealed_error() -> Vec<u8> {
    crate::padding::pad_json(
        serde_json::json!({ "error": { "message": UNSEALED, "type": "server_error" } })
            .to_string()
            .into_bytes(),
    )
}

/// `answer` sealed for `responder`: 200, `message/ohttp-res`. An answer the
/// SDK cannot pad (past 4 MiB) is sealed as a padded 500; if even that
/// fails, a clear 500 [`UNSEALED`], never the answer.
fn seal(
    counts: &Counts,
    responder: &Responder,
    status: u16,
    answer: &[u8],
    nonce: &[u8; RESPONSE_NONCE_LEN],
) -> Response {
    let sealed = responder
        .seal(status, answer, nonce)
        .or_else(|_| responder.seal(500, &unsealed_error(), nonce));
    match sealed {
        Ok(bytes) => {
            counts.sealed.fetch_add(1, Ordering::SeqCst);
            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, MEDIA_TYPE_RESPONSE)],
                bytes,
            )
                .into_response()
        }
        Err(_) => {
            counts.unsealed.fetch_add(1, Ordering::SeqCst);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({ "error": UNSEALED })),
            )
                .into_response()
        }
    }
}

/// `POST /v2/e2e/chat` once the TD serves: `sealed` opened and admitted,
/// answered by `server::chat_contained_as` under `served` with the queue
/// place `slot`, the answer sealed under `nonce`. Refused before the answer:
/// [`refused`]. A panic is never answered in clear: the chat handler turns
/// the backend's into a padded error at its cadence, and one anywhere else
/// in the answer is sealed as [`UNSEALED`]'s padded 500, a size of ticks
/// after `asked`.
pub async fn answer(
    e2e: &E2e,
    server: Arc<crate::server::ServerState>,
    slot: Option<axum::Extension<crate::server::AdmissionSlot>>,
    sealed: &[u8],
    served: &str,
    nonce: [u8; RESPONSE_NONCE_LEN],
    now: u64,
    asked: tokio::time::Instant,
) -> Response {
    let Ok((opened, responder)) = e2e.admit(sealed, now) else {
        return refused(asked).await;
    };
    // `Gateway::open` parsed it as one JSON object of text already.
    let Ok(body) = serde_json::from_slice::<serde_json::Value>(&opened.body) else {
        e2e.counts.refused_open.fetch_add(1, Ordering::SeqCst);
        return refused(asked).await;
    };
    drop(opened);
    let served = served.to_string();
    let answered = tokio::spawn(async move {
        let response = crate::server::chat_contained_as(server, slot, body, &served).await;
        let status = response.status().as_u16();
        let body = axum::body::to_bytes(response.into_body(), e2e::MAX_PADDED).await;
        (status, body)
    })
    .await;
    match answered {
        Ok((status, Ok(answer))) => seal(&e2e.counts, &responder, status, &answer, &nonce),
        Ok((_, Err(_))) => seal(&e2e.counts, &responder, 500, &unsealed_error(), &nonce),
        // The payload is dropped unread: it may quote the question.
        Err(_) => {
            let after = crate::padding::release_after(asked.elapsed(), crate::stream::TICK);
            tokio::time::sleep_until(asked + after).await;
            seal(&e2e.counts, &responder, 500, &unsealed_error(), &nonce)
        }
    }
}

/// The refusal of a stream asked too close to the session's end: 503 at
/// [`x0ne_sdk::sealed_stream::MIN_CHUNKS`] chunks, before the model.
pub const SESSION_ENDS: &str =
    "the session ends before a stream asked now could: ask the next session";

/// What a chunked request is answered under: the session's cadence, and its
/// end (Unix seconds) that bounds the stream (`e2e_stream::cap`).
#[derive(Clone, Copy)]
pub struct StreamTerms {
    pub cadence: x0ne_sdk::attest::E2eStream,
    pub ends: u64,
}

/// `POST /v2/e2e/chat` for a chunked request: `sealed` opened and admitted,
/// started by `server::sealed_stream_start` under `served`, answered as a
/// chunked response under `nonce` at `terms.cadence`, its first chunk and
/// the head with it one tick after `asked`. Refused before it opens:
/// [`refused`], as a whole request. Every outcome after that is a stream of
/// the cadence's shape: a stream that could end past `terms.ends` less the
/// relay's guard is refused [`SESSION_ENDS`] inside one, the model not run;
/// a panic of the start is a sealed 500.
#[allow(clippy::too_many_arguments)]
pub async fn answer_stream(
    e2e: &E2e,
    server: Arc<crate::server::ServerState>,
    slot: Option<axum::Extension<crate::server::AdmissionSlot>>,
    sealed: &[u8],
    served: &str,
    nonce: [u8; RESPONSE_NONCE_LEN],
    now: u64,
    asked: tokio::time::Instant,
    terms: StreamTerms,
) -> Response {
    use crate::e2e_stream::{cap, SealedStream, Start};
    use crate::openai::error::ApiError;
    let Ok((opened, responder)) = e2e.admit_stream(sealed, now) else {
        return refused(asked).await;
    };
    let Ok(body) = serde_json::from_slice::<serde_json::Value>(&opened.body) else {
        e2e.counts.refused_open.fetch_add(1, Ordering::SeqCst);
        return refused(asked).await;
    };
    drop(opened);
    let Ok(writer) = responder.writer(&nonce) else {
        e2e.counts.unsealed.fetch_add(1, Ordering::SeqCst);
        at_floor(asked).await;
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({ "error": UNSEALED })),
        )
            .into_response();
    };
    let slot = slot.map(|axum::Extension(s)| s).unwrap_or_default();
    let shortest = x0ne_sdk::sealed_stream::MIN_CHUNKS;
    let tick = std::time::Duration::from_millis(u64::from(terms.cadence.tick_ms));
    let (start, chunks, place) = match (slot.open_stream(), cap(&terms.cadence, now, terms.ends)) {
        (Err(full), _) => (Start::Refused(full.into()), shortest, None),
        (Ok(place), None) => (
            Start::Refused(ApiError::new(StatusCode::SERVICE_UNAVAILABLE, SESSION_ENDS)),
            shortest,
            place,
        ),
        (Ok(place), Some(chunks)) => {
            let start = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let ends = crate::e2e_stream::StreamTurn {
                    asked,
                    tick,
                    cap: chunks,
                };
                crate::server::sealed_stream_start(server, slot.clone(), body, served, ends)
            }))
            .unwrap_or_else(|_| {
                Start::Refused(ApiError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    crate::stream::INTERNAL,
                ))
            });
            (start, chunks, place)
        }
    };
    // A refusal holds the queue place to its 16th tick, as an answer's
    // shortest turn (`StreamTurn::end`), and its open stream place to the
    // body's end, as an answer: a host that probes either tells no refusal
    // from an answer.
    if matches!(start, Start::Refused(_)) {
        let held = slot.take();
        let ends = crate::e2e_stream::StreamTurn {
            asked,
            tick,
            cap: shortest,
        }
        .end(asked);
        tokio::spawn(async move {
            tokio::time::sleep_until(ends).await;
            drop(held);
        });
    }
    e2e.counts.streams.fetch_add(1, Ordering::SeqCst);
    let clock = crate::stream::Clock::every(asked + tick, tick);
    let body = SealedStream::new(start, terms.cadence, chunks, writer, clock).holding(place);
    // The head leaves with the first chunk, not before: its time says nothing.
    tokio::time::sleep_until(asked + tick).await;
    (
        StatusCode::OK,
        [
            (
                axum::http::header::CONTENT_TYPE,
                MEDIA_TYPE_CHUNKED_RESPONSE,
            ),
            (
                axum::http::HeaderName::from_static(x0ne_sdk::ohttp::INCREMENTAL.0),
                x0ne_sdk::ohttp::INCREMENTAL.1,
            ),
        ],
        axum::body::Body::new(body),
    )
        .into_response()
}
