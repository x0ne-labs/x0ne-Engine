//! The replay cache. Anyone holding the public key can
//! encapsulate; if that alone filled the cache, a full cache would stop the node
//! for everyone, and a client who retries elsewhere would be steered to the nodes
//! still up. The cache takes only requests that passed every other check, and
//! forgets each one when its date leaves the window (RFC 9458 section 6.5.1).

mod common;

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use common::*;
use x0ne_ohttp::client::encapsulate_at;
use x0ne_ohttp::config::{KeyPin, PinnedConfig};
use x0ne_ohttp::gateway::{Gateway, GatewayKey, GatewayPolicy, Kem};
use x0ne_ohttp::message::{date_field, InnerRequest};

struct Node {
    gateway: Arc<Gateway>,
    config: PinnedConfig,
    log: Arc<TargetLog>,
    clock: Arc<Mutex<SystemTime>>,
}

impl Node {
    fn new(replay_capacity: usize) -> Self {
        let key = GatewayKey::generate(1, Kem::X25519).unwrap();
        let encoded = key.encoded_config().to_vec();
        let log = Arc::new(TargetLog::default());
        let clock = Arc::new(Mutex::new(SystemTime::now()));
        let c = clock.clone();
        let gateway = Gateway::with_clock(
            key,
            GatewayPolicy {
                authority: AUTHORITY.into(),
                paths: vec![PATH.into()],
                max_body: 1 << 16,
                replay_capacity,
                replay_window: WINDOW,
            },
            echo_target(log.clone()),
            Arc::new(move || *c.lock().unwrap()),
        );
        let config = PinnedConfig::from_config(&encoded, &KeyPin::of(&encoded)).unwrap();
        Self {
            gateway,
            config,
            log,
            clock,
        }
    }

    fn now(&self) -> SystemTime {
        *self.clock.lock().unwrap()
    }

    fn advance(&self, by: Duration) {
        *self.clock.lock().unwrap() += by;
    }

    /// Encapsulates `req` with the client's clock at `at`, returns the status.
    async fn send_at(&self, req: &InnerRequest, at: SystemTime) -> u16 {
        let (enc, pending) = encapsulate_at(&self.config, req, AUTHORITY, at).unwrap();
        self.status(&enc, pending).await
    }

    async fn status(&self, enc: &[u8], pending: x0ne_ohttp::client::PendingResponse) -> u16 {
        let enc_response = self.gateway.handle(enc).await.expect("opened");
        pending.open(&enc_response).expect("answer").status
    }
}

/// The verifier's attack: valid encapsulations of requests the gateway refuses.
#[tokio::test]
async fn requests_the_gateway_refuses_do_not_fill_the_replay_cache() {
    let node = Node::new(16);
    let mut nowhere = chat("junk");
    nowhere.path = "/nowhere".into();
    let mut bad_header = chat("junk");
    bad_header.headers[0].1 = "application/json; user=attacker".into();
    for _ in 0..16 {
        assert_eq!(node.send_at(&nowhere, node.now()).await, 404);
        let date = date_field(node.now());
        let (enc, pending) = hostile_request(
            node.config.encoded(),
            &[("content-type", "text/plain"), ("date", date.as_str())],
        );
        let enc_response = node.gateway.handle(&enc).await.expect("opened");
        assert_eq!(status_of(pending, &enc_response), 400);
    }
    assert_eq!(node.send_at(&chat("honest"), node.now()).await, 200);
    assert_eq!(node.log.calls(), 1);
}

/// Valid requests can still fill it; the node then refuses until they age out of
/// the window, not until a key rotation that nothing performs.
#[tokio::test]
async fn a_full_replay_cache_empties_as_its_requests_leave_the_window() {
    let node = Node::new(4);
    for _ in 0..4 {
        assert_eq!(node.send_at(&chat("flood"), node.now()).await, 200);
    }
    assert_eq!(node.send_at(&chat("honest"), node.now()).await, 503);
    node.advance(WINDOW + Duration::from_secs(61));
    assert_eq!(node.send_at(&chat("honest"), node.now()).await, 200);
    assert_eq!(node.log.calls(), 5);
}

#[tokio::test]
async fn a_request_dated_outside_the_window_is_refused() {
    let node = Node::new(16);
    let late = node.now() - WINDOW - Duration::from_secs(60);
    let early = node.now() + WINDOW + Duration::from_secs(60);
    assert_eq!(node.send_at(&chat("x"), late).await, 400);
    assert_eq!(node.send_at(&chat("x"), early).await, 400);
    assert_eq!(node.log.calls(), 0);
    // A client clock a minute off is inside the window.
    let skewed = node.now() - Duration::from_secs(60);
    assert_eq!(node.send_at(&chat("x"), skewed).await, 200);
}

/// Forgetting a request does not let it in again: by then its date is outside.
/// The replayed bytes answer under the first request's keys, so only the target's
/// count shows whether a replay ran.
#[tokio::test]
async fn a_replay_is_refused_inside_the_window_and_after_it() {
    let node = Node::new(16);
    let (enc, pending) =
        encapsulate_at(&node.config, &chat("once"), AUTHORITY, node.now()).unwrap();
    assert_eq!(node.status(&enc, pending).await, 200);
    node.gateway.handle(&enc).await.expect("opened");
    node.advance(WINDOW + Duration::from_secs(61));
    node.gateway.handle(&enc).await.expect("opened");
    assert_eq!(node.log.calls(), 1, "a replay reached the target");
}
