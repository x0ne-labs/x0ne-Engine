//! The collateral of one verdict, fetched abreast: what the fake Intel and
//! the fake Google see (how many GETs, how many at once), and that the
//! refusal is the one a serial run gives when several answers fail.
//!
//! The fakes are the lab's, each behind a loopback proxy that waits before
//! it passes a GET on, counts the GETs in flight and breaks the ones a test
//! names. The milliseconds printed are those of this proxy, not Intel's.
//!
//! The proxy answers a GET one delay after the latest GET arrived, its own
//! or a later one: GETs sent together are in flight together even when the
//! machine is slow to deliver one of them, and a GET sent alone waits one
//! delay, as on a plain wire.

use super::*;
use std::sync::Mutex;
use std::time::Instant;

/// One GET the proxies answer differently: every needle is in its
/// `path?query`.
#[derive(Clone)]
struct Rule {
    all: &'static [&'static str],
    /// Answered with this status and no body, the fake not asked.
    status: Option<u16>,
    /// Waited before the answer, in place of the wire's delay.
    delay: Option<Duration>,
}

fn broken(all: &'static [&'static str], status: u16) -> Rule {
    Rule {
        all,
        status: Some(status),
        delay: None,
    }
}

/// What the proxies of one scene share: the delay of a GET, the rules, and
/// what was seen.
struct Wire {
    delay: Mutex<Duration>,
    rules: Mutex<Vec<Rule>>,
    in_flight: AtomicUsize,
    peak: AtomicUsize,
    last_arrival: Mutex<Instant>,
    /// `path?query`, when it arrived and when it was answered.
    seen: Mutex<Vec<(String, Instant, Instant)>>,
}

impl Wire {
    fn new(delay: Duration) -> Arc<Self> {
        Arc::new(Self {
            delay: Mutex::new(delay),
            rules: Mutex::new(Vec::new()),
            in_flight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            last_arrival: Mutex::new(Instant::now()),
            seen: Mutex::new(Vec::new()),
        })
    }

    /// Forget what was seen, and wait `delay` before each answer from now on.
    fn slow_to(&self, delay: Duration) {
        *self.delay.lock().unwrap() = delay;
        self.reset(Vec::new());
    }

    /// Forget what was seen, and answer by `rules` from now on.
    fn reset(&self, rules: Vec<Rule>) {
        *self.rules.lock().unwrap() = rules;
        self.peak.store(0, SeqCst);
        self.seen.lock().unwrap().clear();
    }

    fn peak(&self) -> usize {
        self.peak.load(SeqCst)
    }

    /// The GETs seen, in the order they arrived.
    fn seen(&self) -> Vec<(String, Instant, Instant)> {
        let mut seen = self.seen.lock().unwrap().clone();
        seen.sort_by_key(|(_, arrived, _)| *arrived);
        seen
    }

    fn count(&self, needle: &str) -> usize {
        self.seen()
            .iter()
            .filter(|(what, _, _)| what.contains(needle))
            .count()
    }
}

/// A proxy on the loopback in front of `upstream`; returns its base URL.
async fn slowed(upstream: &str, wire: Arc<Wire>) -> String {
    let upstream = upstream.trim_end_matches('/').to_string();
    let client = http();
    let app = axum::Router::new().fallback(move |req: axum::extract::Request| {
        let (upstream, wire, client) = (upstream.clone(), wire.clone(), client.clone());
        async move {
            let what = req
                .uri()
                .path_and_query()
                .map(|p| p.as_str().to_string())
                .unwrap_or_default();
            let arrived = Instant::now();
            *wire.last_arrival.lock().unwrap() = arrived;
            let now = wire.in_flight.fetch_add(1, SeqCst) + 1;
            wire.peak.fetch_max(now, SeqCst);
            let rule = wire
                .rules
                .lock()
                .unwrap()
                .iter()
                .find(|r| r.all.iter().all(|n| what.contains(n)))
                .cloned();
            match rule.as_ref().and_then(|r| r.delay) {
                Some(delay) => tokio::time::sleep(delay).await,
                None => loop {
                    let due = *wire.last_arrival.lock().unwrap() + *wire.delay.lock().unwrap();
                    if Instant::now() >= due {
                        break;
                    }
                    tokio::time::sleep_until(due.into()).await;
                },
            }
            let mut out = axum::response::Response::new(axum::body::Body::empty());
            match rule.and_then(|r| r.status) {
                Some(status) => {
                    *out.status_mut() = axum::http::StatusCode::from_u16(status).unwrap();
                }
                None => {
                    let got = client
                        .get(format!("{upstream}{what}"))
                        .send()
                        .await
                        .unwrap();
                    *out.status_mut() = got.status();
                    for (name, value) in got.headers() {
                        if !["content-length", "transfer-encoding", "connection"]
                            .contains(&name.as_str())
                        {
                            out.headers_mut().insert(name.clone(), value.clone());
                        }
                    }
                    *out.body_mut() = axum::body::Body::from(got.bytes().await.unwrap());
                }
            }
            wire.in_flight.fetch_sub(1, SeqCst);
            wire.seen
                .lock()
                .unwrap()
                .push((what, arrived, Instant::now()));
            out
        }
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

/// An honest session's document, and the lab's Intel and Google behind one
/// wire.
struct Scene {
    lab: Lab,
    _td: Td,
    doc: crate::attest_v2::Document,
    config: Loaded,
    spki: [u8; 32],
    wire: Arc<Wire>,
    pcs: String,
    bucket: String,
}

impl Scene {
    async fn new(delay: Duration) -> Self {
        let lab = Lab::new().await;
        let relay_tls = TlsIdentity::generate("relay").unwrap();
        let (td, client, config) = honest_session(&lab, &relay_tls, TdOpts::default()).await;
        let doc = client.attestation().await.unwrap();
        let wire = Wire::new(delay);
        let pcs = slowed(&lab.pcs.base_url, wire.clone()).await;
        let bucket = slowed(&lab.bucket, wire.clone()).await;
        Self {
            spki: client.pinned_spki().unwrap(),
            lab,
            _td: td,
            doc,
            config,
            wire,
            pcs,
            bucket,
        }
    }

    fn root_crl_url(&self) -> String {
        format!("{}/rootcacrl.der", self.pcs)
    }

    fn quote(&self) -> Vec<u8> {
        unb64(&self.doc.tdx.quote_b64).unwrap()
    }

    /// The relay's verdict on the document, everything fetched through the wire.
    async fn verdict(&self) -> Result<crate::remote_verify::Report, crate::remote_verify::Refusal> {
        let reference = self.lab.reference();
        let catalog = crate::registry::Catalog::embedded();
        let exp = crate::remote_verify::Expectations {
            config: &self.config,
            tls_spki_sha256: self.spki,
            machine_type: MACHINE,
            reference: &reference,
            catalog_entry: catalog.get(MODEL).unwrap(),
        };
        let mut anchors = self.lab.anchors();
        anchors.pcs_base_url = self.pcs.clone();
        anchors.root_crl_url = self.root_crl_url();
        anchors.gce_bucket_base_url = self.bucket.clone();
        crate::remote_verify::verify_document(&self.doc, &exp, &anchors, &http(), now()).await
    }

    /// The TD's own live check of its quote, through the wire.
    async fn td_judgement(&self) -> Result<(), String> {
        crate::tcb_standing::judge_platform_live(
            &self.quote(),
            &self.pcs,
            &self.root_crl_url(),
            &self.lab.intel.anchor(),
            &crate::tcb::TcbPolicy::production(),
            now(),
        )
        .await
        .map(|_| ())
        .map_err(|e| e.to_string())
    }

    /// The collateral the relay hands an SDK, through the wire.
    async fn sdk_collateral(&self) -> Result<x0ne_sdk::attest::Collateral, String> {
        let sources = crate::remote_verify::CollateralSources {
            pcs_base_url: self.pcs.clone(),
            root_crl_url: self.root_crl_url(),
            gce_bucket_base_url: self.bucket.clone(),
        };
        crate::remote_verify::sdk_collateral(&http(), &sources, &self.quote()).await
    }
}

const QE_STANDARD: &[&str] = &["/qe/identity", "update=standard"];
const TCB_STANDARD: &[&str] = &["/tcb?", "update=standard"];
const QE_EARLY: &[&str] = &["/qe/identity", "update=early"];
const TCB_EARLY: &[&str] = &["/tcb?", "update=early"];
const PCK_CRL: &[&str] = &["/pckcrl"];
const ROOT_CRL: &[&str] = &["/rootcacrl.der"];
const ENDORSEMENT: &[&str] = &[".binarypb"];
const FIRMWARE: &[&str] = &[".fd"];

/// The delay of the measured runs: 50 ms a GET.
const MEASURED: Duration = Duration::from_millis(50);
/// The delay of the runs that only compare refusals.
const SHORT: Duration = Duration::from_millis(5);
/// The delay of the runs that count the GETs in flight at once: GETs sent
/// together would have to arrive this far apart to be counted apart.
const STEADY: Duration = Duration::from_millis(250);

/// The relay's verdict asks Intel and Google eight things. One GET first
/// (Intel's clock, before anything is judged on a date), then the six that
/// wait for no other answer together, then the firmware binary the verified
/// endorsement names: three round trips, not eight.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_relay_fetches_a_verdicts_collateral_abreast() {
    let s = Scene::new(MEASURED).await;
    let started = Instant::now();
    s.verdict().await.expect("the honest document passes");
    let took = started.elapsed();
    println!(
        "collateral, relay verdict: 8 GETs at {} ms each took {} ms, {} in flight at most",
        MEASURED.as_millis(),
        took.as_millis(),
        s.wire.peak()
    );
    assert_eq!(s.wire.seen().len(), 8);
    s.wire.slow_to(STEADY);
    s.verdict().await.expect("the honest document passes");
    let seen = s.wire.seen();
    assert_eq!(seen.len(), 8, "{seen:?}");
    for one in [
        QE_STANDARD,
        TCB_STANDARD,
        QE_EARLY,
        TCB_EARLY,
        PCK_CRL,
        ROOT_CRL,
        ENDORSEMENT,
        FIRMWARE,
    ] {
        let n = seen
            .iter()
            .filter(|(what, _, _)| one.iter().all(|n| what.contains(n)))
            .count();
        assert_eq!(n, 1, "{one:?} in {seen:?}");
    }
    // The clock's answer is read before anything else is asked.
    let (first, _, first_answered) = &seen[0];
    assert!(QE_STANDARD.iter().all(|n| first.contains(n)), "{seen:?}");
    assert!(seen[1..]
        .iter()
        .all(|(_, arrived, _)| arrived >= first_answered));
    // The firmware binary is asked for once everything else is answered.
    let (last, last_arrived, _) = &seen[7];
    assert!(last.contains(".fd"), "{seen:?}");
    assert!(seen[..7]
        .iter()
        .all(|(_, _, answered)| answered <= last_arrived));
    assert_eq!(s.wire.peak(), 6, "GETs in flight at once: {seen:?}");
}

/// When several answers fail at once, the refusal is the one of the first
/// of them in the order the checks run: the same code and the same words as
/// when that one fails alone, also when it is the last to come back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_relays_refusal_is_the_serial_one_when_several_answers_fail() {
    let s = Scene::new(SHORT).await;
    let late = Duration::from_millis(60);
    for bucket_status in [404, 500] {
        // In the order the checks read them.
        let order: [(Rule, i32, &str); 6] = [
            (
                broken(QE_STANDARD, 404),
                76,
                "Intel PCS unreachable: PCS answered 404",
            ),
            (
                broken(TCB_STANDARD, 404),
                76,
                "Intel PCS unreachable: PCS answered 404",
            ),
            (broken(PCK_CRL, 404), 76, "Intel PCS unreachable: PCK CRL: "),
            (broken(ROOT_CRL, 404), 76, "Intel root CA CRL unreachable: "),
            match bucket_status {
                404 => (
                    broken(ENDORSEMENT, 404),
                    65,
                    "has no Google launch endorsement",
                ),
                _ => (
                    broken(ENDORSEMENT, 500),
                    76,
                    "Google endorsement bucket unreachable: ",
                ),
            },
            match bucket_status {
                404 => (
                    broken(FIRMWARE, 404),
                    65,
                    "Google's bucket has no firmware binary",
                ),
                _ => (
                    broken(FIRMWARE, 500),
                    76,
                    "Google bucket unreachable: firmware binary: ",
                ),
            },
        ];
        for k in 0..order.len() {
            let (rule, code, says) = &order[k];
            s.wire.reset(vec![rule.clone()]);
            let alone = s.verdict().await.expect_err("one answer fails");
            assert_eq!(alone.code, *code, "{k}: {alone}");
            assert!(alone.reason.contains(says), "{k}: {alone}");

            let rest = || order[k + 1..].iter().map(|(r, _, _)| r.clone());
            s.wire
                .reset(std::iter::once(rule.clone()).chain(rest()).collect());
            let together = s.verdict().await.expect_err("several answers fail");
            assert_eq!(
                (together.code, &together.reason),
                (alone.code, &alone.reason),
                "{k}, all at once"
            );

            let mut last_back = rule.clone();
            last_back.delay = Some(late);
            let at_once = rest().map(|mut r| {
                r.delay = Some(Duration::ZERO);
                r
            });
            s.wire
                .reset(std::iter::once(last_back).chain(at_once).collect());
            let together = s.verdict().await.expect_err("several answers fail");
            assert_eq!(
                (together.code, &together.reason),
                (alone.code, &alone.reason),
                "{k}, the first in order answered last"
            );
        }
    }
}

/// Intel's `early` answer is named, never a refusal: with both `early`
/// answers failing the verdict names the tcbInfo, as when it fails alone.
/// And a refusal does not wait for an answer the serial run never asked for.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_early_answer_not_fetched_is_named_and_a_refusal_does_not_wait() {
    let s = Scene::new(SHORT).await;
    let named = |r: crate::remote_verify::Report| {
        let tcb = serde_json::to_value(r.tcb).unwrap();
        (
            tcb["intel"]["latest_unavailable"].clone(),
            r.tcb_platform,
            r.tcb_qe,
        )
    };
    s.wire.reset(vec![broken(TCB_EARLY, 404)]);
    let alone = named(s.verdict().await.expect("early is not a refusal"));
    assert!(
        alone.0.to_string().contains("early tcbInfo not fetched"),
        "{alone:?}"
    );
    s.wire
        .reset(vec![broken(TCB_EARLY, 404), broken(QE_EARLY, 404)]);
    let both = named(s.verdict().await.expect("early is not a refusal"));
    assert_eq!(both, alone);

    let mut slow_root_crl = broken(ROOT_CRL, 404);
    slow_root_crl.delay = Some(Duration::from_secs(4));
    s.wire.reset(vec![broken(TCB_STANDARD, 404), slow_root_crl]);
    let started = Instant::now();
    let r = s.verdict().await.unwrap_err();
    assert!(r.reason.contains("/tcb?"), "{r}");
    assert!(started.elapsed() < Duration::from_secs(2), "{r}");
}

/// The TD's own live check (start-up, then hourly) asks Intel six things,
/// none of which waits for another: one round trip, not six. Its failure is
/// the serial one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_td_fetches_its_live_collateral_abreast() {
    let s = Scene::new(MEASURED).await;
    let started = Instant::now();
    s.td_judgement().await.expect("the honest quote passes");
    println!(
        "collateral, TD live check: 6 GETs at {} ms each took {} ms, {} in flight at most",
        MEASURED.as_millis(),
        started.elapsed().as_millis(),
        s.wire.peak()
    );
    assert_eq!(s.wire.seen().len(), 6);
    s.wire.slow_to(STEADY);
    s.td_judgement().await.expect("the honest quote passes");
    let seen = s.wire.seen();
    assert_eq!(seen.len(), 6, "{seen:?}");
    assert_eq!(s.wire.peak(), 6, "GETs in flight at once: {seen:?}");

    s.wire.slow_to(SHORT);
    let order = [
        (broken(TCB_STANDARD, 404), "not fetched: tcbInfo: "),
        (broken(QE_STANDARD, 404), "not fetched: QE identity: "),
        (broken(PCK_CRL, 404), "not fetched: PCK CRL: "),
        (broken(ROOT_CRL, 404), "not fetched: root CA CRL: "),
    ];
    for k in 0..order.len() {
        let (rule, says) = &order[k];
        s.wire.reset(vec![rule.clone()]);
        let alone = s.td_judgement().await.expect_err("one answer fails");
        assert!(alone.starts_with(says), "{k}: {alone}");
        let mut last_back = rule.clone();
        last_back.delay = Some(Duration::from_millis(60));
        let rest = order[k + 1..].iter().map(|(r, _)| {
            let mut r = r.clone();
            r.delay = Some(Duration::ZERO);
            r
        });
        s.wire
            .reset(std::iter::once(last_back).chain(rest).collect());
        let together = s.td_judgement().await.expect_err("several answers fail");
        assert_eq!(together, alone, "{k}");
    }
    // `early` not fetched is named in the judgement, not a failure.
    s.wire
        .reset(vec![broken(TCB_EARLY, 404), broken(QE_EARLY, 404)]);
    s.td_judgement().await.expect("early is not a refusal");
}

/// The collateral the relay hands an SDK is the same eight answers: seven
/// together, then the firmware binary the endorsement names. Its error is
/// the serial one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_relay_fetches_an_sdks_collateral_abreast() {
    let s = Scene::new(MEASURED).await;
    let started = Instant::now();
    let c = s
        .sdk_collateral()
        .await
        .expect("the honest quote's collateral");
    println!(
        "collateral, for an SDK: 8 GETs at {} ms each took {} ms, {} in flight at most",
        MEASURED.as_millis(),
        started.elapsed().as_millis(),
        s.wire.peak()
    );
    for (name, field) in [
        ("tcb_info", &c.tcb_info),
        ("qe_identity", &c.qe_identity),
        ("tcb_info_standard", &c.tcb_info_standard),
        ("qe_identity_standard", &c.qe_identity_standard),
        ("pck_crl", &c.pck_crl_der_b64),
        ("root_ca_crl", &c.root_ca_crl_der_b64),
        ("gce_endorsement", &c.gce_endorsement_b64),
        ("gce_firmware", &c.gce_firmware_b64),
    ] {
        assert!(!field.is_empty(), "{name} is empty");
    }
    assert_eq!(s.wire.seen().len(), 8);
    s.wire.slow_to(STEADY);
    s.sdk_collateral()
        .await
        .expect("the honest quote's collateral");
    let seen = s.wire.seen();
    assert_eq!(seen.len(), 8, "{seen:?}");
    assert_eq!(s.wire.count(".fd"), 1);
    assert_eq!(s.wire.peak(), 7, "GETs in flight at once: {seen:?}");

    s.wire.slow_to(SHORT);
    let order = [
        (broken(PCK_CRL, 404), "PCK CRL: "),
        (broken(ROOT_CRL, 404), "root CA CRL: "),
        (broken(ENDORSEMENT, 500), "Google endorsement: "),
    ];
    for k in 0..order.len() {
        let (rule, says) = &order[k];
        s.wire.reset(vec![rule.clone()]);
        let Err(alone) = s.sdk_collateral().await else {
            panic!("{k}: one answer fails");
        };
        assert!(alone.starts_with(says), "{k}: {alone}");
        let mut last_back = rule.clone();
        last_back.delay = Some(Duration::from_millis(60));
        let rest = order[k + 1..].iter().map(|(r, _)| {
            let mut r = r.clone();
            r.delay = Some(Duration::ZERO);
            r
        });
        s.wire
            .reset(std::iter::once(last_back).chain(rest).collect());
        let Err(together) = s.sdk_collateral().await else {
            panic!("{k}: several answers fail");
        };
        assert_eq!(together, alone, "{k}");
    }
    s.wire.reset(vec![broken(FIRMWARE, 500)]);
    let Err(e) = s.sdk_collateral().await else {
        panic!("no firmware binary");
    };
    assert!(e.starts_with("Google firmware binary: "), "{e}");
    // An answer Intel did not give is left empty, for the SDK to judge.
    s.wire
        .reset(vec![broken(TCB_EARLY, 404), broken(TCB_STANDARD, 404)]);
    let c = s.sdk_collateral().await.expect("left empty, not an error");
    assert!(c.tcb_info.is_empty() && c.qe_identity.is_empty());
    assert!(c.tcb_info_standard.is_empty() && c.qe_identity_standard.is_empty());
}
