//! End to end on the loopback: client -> relay -> spy -> gateway -> target.
//! Each guarantee has a test that attacks it.

mod common;

use common::*;
use x0ne_ohttp::client::{Client, ClientError};
use x0ne_ohttp::config::{KeyPin, PinnedConfig};
use x0ne_ohttp::gateway::{Gateway, GatewayKey, Kem};
use x0ne_ohttp::message::{date_field, InnerRequest, JSON};

const SENTINEL: &str = "zq-sentinel-7f3a91-the-question-nobody-may-read";
const ANSWER_SENTINEL: &[u8] = b"echo:zq-sentinel-7f3a91";

fn pinned(t: &Topology) -> PinnedConfig {
    PinnedConfig::from_config(&t.config, &KeyPin::of(&t.config)).expect("pinned config")
}

fn client(t: &Topology) -> Client {
    Client::new(&t.relay_url, AUTHORITY).expect("client")
}

#[tokio::test]
async fn a_request_crosses_relay_and_gateway_and_comes_back() {
    let t = topology().await;
    let resp = client(&t)
        .send(&pinned(&t), &chat(SENTINEL))
        .await
        .expect("round trip");
    assert_eq!(resp.status, 200);
    assert_eq!(resp.body, format!("echo:{SENTINEL}").into_bytes());
    assert_eq!(t.target.calls(), 1);
    let seen = t.target.requests.lock().unwrap()[0].clone();
    assert_eq!(seen.path, PATH);
    assert_eq!(seen.body, SENTINEL.as_bytes());
}

#[tokio::test]
async fn the_post_quantum_hybrid_kem_round_trips_too() {
    let t = topology_with(9, Kem::XWing).await;
    let resp = client(&t)
        .send(&pinned(&t), &chat(SENTINEL))
        .await
        .expect("x-wing round trip");
    assert_eq!(resp.body, format!("echo:{SENTINEL}").into_bytes());
}

#[tokio::test]
async fn the_relay_sees_neither_the_question_nor_the_answer() {
    let t = topology().await;
    client(&t)
        .send(&pinned(&t), &chat(SENTINEL))
        .await
        .expect("round trip");
    let seen = t.spy.all_bytes();
    assert!(!seen.is_empty(), "the spy saw the exchange");
    assert!(!contains(&seen, SENTINEL.as_bytes()), "question in clear");
    assert!(!contains(&seen, ANSWER_SENTINEL), "answer in clear");
    assert!(!contains(&seen, PATH.as_bytes()), "target path in clear");
    assert!(!contains(&seen, AUTHORITY.as_bytes()), "authority in clear");
}

#[tokio::test]
async fn identifying_headers_never_reach_the_gateway() {
    let t = topology().await;
    let config = pinned(&t);
    let req = x0ne_ohttp::client::encapsulate(&config, &chat("hello"), AUTHORITY)
        .expect("encapsulate")
        .0;
    // A client that leaks: the relay must drop all of this.
    let resp = reqwest::Client::new()
        .post(format!("{}/", t.relay_url))
        .header("content-type", "message/ohttp-req")
        .header("cookie", "session=alice-7731")
        .header("user-agent", "alice-laptop/1.0")
        .header("authorization", "Bearer alice-api-key")
        .header("x-forwarded-for", "203.0.113.77")
        .header("forwarded", "for=203.0.113.77")
        .header("via", "1.1 alice-proxy")
        .header("x-client-id", "alice")
        .body(req)
        .send()
        .await
        .expect("post to relay");
    assert_eq!(resp.status().as_u16(), 200);
    let seen = t.spy.requests.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    let allowed = ["host", "content-type", "content-length", "accept"];
    for (name, value) in &seen[0].headers {
        assert!(
            allowed.contains(&name.as_str()),
            "the relay forwarded or added {name}: {value}"
        );
        for leak in ["alice", "203.0.113.77"] {
            assert!(!value.contains(leak), "{name}: {value}");
        }
    }
}

#[tokio::test]
async fn a_byte_flipped_by_the_relay_in_the_request_is_refused_before_the_target() {
    let t = topology().await;
    t.spy.set_mode(Mode::FlipRequestByte);
    let err = client(&t)
        .send(&pinned(&t), &chat(SENTINEL))
        .await
        .expect_err("tampered request");
    assert!(
        matches!(err, ClientError::Status(s) if (400..500).contains(&s)),
        "{err:?}"
    );
    assert_eq!(t.target.calls(), 0);
}

#[tokio::test]
async fn a_byte_flipped_by_the_relay_in_the_response_is_refused_by_the_client() {
    let t = topology().await;
    t.spy.set_mode(Mode::FlipResponseByte);
    let err = client(&t)
        .send(&pinned(&t), &chat(SENTINEL))
        .await
        .expect_err("tampered response");
    assert!(matches!(err, ClientError::Decapsulation), "{err:?}");
}

#[tokio::test]
async fn the_answer_to_another_request_is_refused() {
    let t = topology().await;
    let c = client(&t);
    let config = pinned(&t);
    c.send(&config, &chat("first")).await.expect("first");
    t.spy.set_mode(Mode::ReplayPreviousResponse);
    let err = c
        .send(&config, &chat("second"))
        .await
        .expect_err("swapped response");
    assert!(matches!(err, ClientError::Decapsulation), "{err:?}");
}

#[tokio::test]
async fn a_request_replayed_by_the_relay_reaches_the_target_once() {
    let t = topology().await;
    t.spy.set_mode(Mode::DuplicateRequest);
    let resp = client(&t)
        .send(&pinned(&t), &chat(SENTINEL))
        .await
        .expect("first copy answered");
    assert_eq!(resp.status, 200);
    assert_eq!(t.target.calls(), 1, "the replay ran the request again");
}

#[tokio::test]
async fn a_failing_relay_gets_one_attempt_and_an_error() {
    let t = topology().await;
    t.spy.set_mode(Mode::Fail);
    let err = client(&t)
        .send(&pinned(&t), &chat(SENTINEL))
        .await
        .expect_err("relay failure");
    assert!(matches!(err, ClientError::Status(502)), "{err:?}");
    assert_eq!(
        t.spy.requests.lock().unwrap().len(),
        1,
        "the client retried"
    );
}

#[tokio::test]
async fn the_relay_refuses_anything_but_an_encapsulated_request() {
    let t = topology().await;
    let resp = reqwest::Client::new()
        .post(format!("{}/", t.relay_url))
        .header("content-type", "application/json")
        .body(format!("{{\"prompt\":\"{SENTINEL}\"}}"))
        .send()
        .await
        .expect("post");
    assert_eq!(resp.status().as_u16(), 415);
    assert!(t.spy.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn the_gateway_serves_only_its_own_target() {
    let t = topology().await;
    let c = client(&t);
    let config = pinned(&t);
    let mut other_path = chat("x");
    other_path.path = "/admin/dump".into();
    let resp = c
        .send(&config, &other_path)
        .await
        .expect("encapsulated refusal");
    assert_eq!(resp.status, 404);

    let other_authority = Client::new(&t.relay_url, "metadata.google.internal").expect("client");
    let resp = other_authority
        .send(&config, &chat("x"))
        .await
        .expect("encapsulated refusal");
    assert_eq!(resp.status, 403);
    assert_eq!(t.target.calls(), 0);
}

#[tokio::test]
async fn every_request_gets_a_fresh_hpke_context() {
    let t = topology().await;
    let config = pinned(&t);
    let a = x0ne_ohttp::client::encapsulate(&config, &chat("same"), AUTHORITY)
        .unwrap()
        .0;
    let b = x0ne_ohttp::client::encapsulate(&config, &chat("same"), AUTHORITY)
        .unwrap()
        .0;
    assert_eq!(a.len(), b.len());
    assert_eq!(&a[..7], &b[..7], "same header: key id and suite");
    assert_ne!(&a[7..39], &b[7..39], "the encapsulated key was reused");
    assert_ne!(a, b);
}

#[tokio::test]
async fn sizes_on_the_wire_hide_the_length_within_a_bucket() {
    let t = topology().await;
    let config = pinned(&t);
    let short = x0ne_ohttp::client::encapsulate(&config, &chat("hi"), AUTHORITY)
        .unwrap()
        .0;
    let long_text = "x".repeat(700);
    let long = x0ne_ohttp::client::encapsulate(&config, &chat(&long_text), AUTHORITY)
        .unwrap()
        .0;
    assert_eq!(short.len(), long.len(), "request length leaks the question");

    let c = client(&t);
    c.send(&config, &chat("hi")).await.unwrap();
    c.send(&config, &chat(&long_text)).await.unwrap();
    let responses = t.spy.responses.lock().unwrap().clone();
    assert_eq!(responses.len(), 2);
    assert_eq!(
        responses[0].len(),
        responses[1].len(),
        "response length leaks the answer"
    );
}

#[tokio::test]
async fn the_client_puts_no_identity_inside_the_encapsulation() {
    let t = topology().await;
    let c = client(&t);
    let config = pinned(&t);
    for (name, value) in [
        ("cookie", "session=alice"),
        ("authorization", "Bearer alice-api-key"),
        ("user-agent", "alice-laptop"),
        ("x-forwarded-for", "203.0.113.77"),
    ] {
        let mut req: InnerRequest = chat("x");
        req.headers.push((name.into(), value.into()));
        let err = c.send(&config, &req).await.expect_err(name);
        assert!(
            matches!(err, ClientError::HeaderRefused(ref h) if h == name),
            "{err:?}"
        );
    }
    // An anonymous token is the one credential allowed.
    let mut req = chat("x");
    req.headers.push(("authorization".into(), private_token()));
    c.send(&config, &req).await.expect("private token accepted");
    assert_eq!(t.target.calls(), 1);
    assert!(
        t.spy.requests.lock().unwrap().len() == 1,
        "refused requests were sent"
    );
}

#[tokio::test]
async fn the_client_refuses_cleartext_to_a_remote_relay() {
    for url in [
        "http://relay.example.net",
        "http://172.16.0.8:8080",
        "ftp://127.0.0.1",
    ] {
        assert!(
            matches!(Client::new(url, AUTHORITY), Err(ClientError::Url(_))),
            "{url} accepted"
        );
    }
    Client::new("https://relay.example.net", AUTHORITY).expect("https relay");
    Client::new("http://127.0.0.1:9", AUTHORITY).expect("loopback relay");
}

#[tokio::test]
async fn the_relay_refuses_cleartext_to_a_remote_gateway() {
    use x0ne_ohttp::relay::{router, RelayConfig};
    for url in ["http://gateway.example.net", "http://172.16.1.2"] {
        let r = router(RelayConfig {
            gateway_url: url.to_string(),
            max_body: 1024,
        });
        assert!(r.is_err(), "{url} accepted");
    }
}

#[tokio::test]
async fn the_key_configuration_fetched_through_the_relay_must_match_the_pin() {
    let t = topology().await;
    let c = client(&t);
    let ok = c
        .fetch_config(&KeyPin::of(&t.config))
        .await
        .expect("pinned key served");
    assert_eq!(ok.key_id(), t.key_id);

    let other = topology().await;
    let err = c
        .fetch_config(&KeyPin::of(&other.config))
        .await
        .expect_err("a key for someone else");
    assert!(matches!(err, ClientError::Config(_)), "{err:?}");
}

/// The identity rides in the value of an allowed field.
#[tokio::test]
async fn an_authorization_value_that_is_not_a_token_never_leaves_the_client() {
    let t = topology().await;
    let c = client(&t);
    let with_param = format!("{}, user=\"alice\"", private_token());
    for value in [
        "PrivateToken alice-7731@example.com",
        "PrivateToken token=\"alice-7731\"",
        with_param.as_str(),
    ] {
        let mut req = chat("x");
        req.headers.push(("authorization".into(), value.into()));
        let err = c.send(&pinned(&t), &req).await.expect_err(value);
        assert!(
            matches!(err, ClientError::HeaderRefused(ref h) if h == "authorization"),
            "{value}: {err:?}"
        );
    }
    assert_eq!(t.target.calls(), 0, "the identity reached the target");
    assert!(
        t.spy.requests.lock().unwrap().is_empty(),
        "refused requests were sent"
    );
}

#[tokio::test]
async fn free_form_accept_and_content_type_values_never_leave_the_client() {
    let t = topology().await;
    let c = client(&t);
    let mut req = chat("x");
    req.headers
        .push(("accept".into(), "application/json; user=alice".into()));
    req.headers[0].1 = "application/json; boundary=alice-7731".into();
    let r = c.send(&pinned(&t), &req).await;
    assert!(r.is_err(), "free-form header values were accepted");
    assert_eq!(t.target.calls(), 0, "the identity reached the target");
}

/// A client that skips this crate's checks: the gateway applies them again, and
/// nothing reaches the target but the encoder's exact form.
#[tokio::test]
async fn the_gateway_refuses_any_field_a_hostile_client_could_name_itself_with() {
    let key = GatewayKey::generate(1, Kem::X25519).unwrap();
    let config = key.encoded_config().to_vec();
    let log = std::sync::Arc::new(TargetLog::default());
    let gateway = Gateway::new(key, policy(), echo_target(log.clone()));
    let date = date_field(std::time::SystemTime::now());
    // Inside the window, but not on a minute: a client's own clock to the second.
    let to_the_second = httpdate::fmt_http_date(
        httpdate::parse_http_date(&date).unwrap() + std::time::Duration::from_secs(7),
    );
    let token = private_token();
    let (date, to_the_second, token) = (date.as_str(), to_the_second.as_str(), token.as_str());
    let cases: Vec<Vec<(&str, &str)>> = vec![
        vec![
            ("content-type", "application/json; user=alice"),
            ("date", date),
        ],
        vec![("accept", "application/json; user=alice"), ("date", date)],
        vec![
            ("authorization", "PrivateToken alice-7731@example.com"),
            ("date", date),
        ],
        vec![("x-user", "alice"), ("date", date)],
        vec![("Content-Type", JSON), ("date", date)],
        vec![
            ("content-type", JSON),
            ("content-type", JSON),
            ("date", date),
        ],
        vec![("accept", JSON), ("content-type", JSON), ("date", date)],
        vec![("date", date), ("content-type", JSON)],
        vec![("content-type", JSON)],
        vec![("content-type", JSON), ("date", to_the_second)],
    ];
    for fields in &cases {
        let (enc, pending) = hostile_request(&config, fields);
        let enc_response = gateway.handle(&enc).await.expect("opened");
        assert_eq!(status_of(pending, &enc_response), 400, "{fields:?}");
    }
    assert_eq!(log.calls(), 0, "a hostile field reached the target");

    let (enc, pending) = hostile_request(
        &config,
        &[
            ("content-type", JSON),
            ("accept", JSON),
            ("authorization", token),
            ("date", date),
        ],
    );
    let enc_response = gateway.handle(&enc).await.expect("opened");
    assert_eq!(
        status_of(pending, &enc_response),
        200,
        "the exact form is served"
    );
    assert_eq!(log.calls(), 1);
    let names: Vec<String> = log.requests.lock().unwrap()[0]
        .headers
        .iter()
        .map(|(k, _)| k.clone())
        .collect();
    assert_eq!(
        names,
        ["content-type", "accept", "authorization"],
        "the target saw the date"
    );
}

/// What the padding hides and what it does not: the
/// size on the wire is the bucket, a power of two from 1 KiB; within a bucket the
/// length does not show, across buckets it does.
#[test]
fn sizes_on_the_wire_follow_only_the_bucket_of_the_question() {
    let key = GatewayKey::generate(1, Kem::X25519).unwrap();
    let cfg = key.encoded_config().to_vec();
    let config = PinnedConfig::from_config(&cfg, &KeyPin::of(&cfg)).unwrap();
    // Key id and suite (7), encapsulated key (32), AEAD tag (16).
    let overhead = 7 + 32 + 16;
    let wire = |n: usize| {
        x0ne_ohttp::client::encapsulate(&config, &chat(&"x".repeat(n)), AUTHORITY)
            .unwrap()
            .0
            .len()
            - overhead
    };
    let mut buckets: Vec<usize> = (0..=5000).step_by(25).map(wire).collect();
    assert!(buckets.iter().all(|b| b.is_power_of_two() && *b >= 1024));
    buckets.dedup();
    assert_eq!(buckets, [1024, 2048, 4096, 8192]);
}
