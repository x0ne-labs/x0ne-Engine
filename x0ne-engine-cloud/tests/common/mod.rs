//! Shared by the integration tests: a fake Google Cloud on the loopback, fake
//! relays (bash scripts speaking I4), a pty, one directory per test. Nothing
//! here needs the network; the binary refuses any non-loopback override.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::ffi::CStr;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::Router;
use serde_json::{json, Value};

pub const PROJECT: &str = "x0ne-test-project";
/// Fake; shaped like a Google access token so the client accepts it.
pub const TOKEN: &str = "ya29.FAKE-test-token_0123456789abcdefghijklmnopqrstuvwxyz";
pub const VM_IP: &str = "203.0.113.7";
/// This machine's public address, as the fake name server answers it.
pub const CLIENT_IP: &str = "198.51.100.23";
pub const WAIT: Duration = Duration::from_secs(60);
pub const BIN: &str = env!("CARGO_BIN_EXE_x0ne-engine-cloud");

static FORK_LOCK: Mutex<()> = Mutex::new(());

/// Held while an executable is written and while a child is spawned: a child
/// forked mid-write would keep the script open for writing (ETXTBSY).
pub fn fork_lock() -> MutexGuard<'static, ()> {
    FORK_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn contains(hay: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && hay.windows(needle.len()).any(|w| w == needle)
}

// ── the fake cloud ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Req {
    pub method: String,
    pub path: String,
    pub query: String,
    pub body: Vec<u8>,
    pub auth: Option<String>,
}

impl Req {
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
    pub fn is(&self, method: &str, path_suffix: &str) -> bool {
        self.method == method && self.path.ends_with(path_suffix)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Boot {
    /// RUNNING with an external IP after this many reads.
    After(usize),
    /// Stays STAGING: the boot timeout fires.
    Never,
    /// RUNNING after `after` reads, gone `then` reads later (spot preemption).
    Preempted { after: usize, then: usize },
}

#[derive(Debug, Clone)]
pub struct Vm {
    pub zone: String,
    pub status: String,
    pub labels: Value,
    pub gets: usize,
}

/// What the Cloud Quotas API answers for the spot H100 QuotaInfo.
#[derive(Debug, Clone)]
pub enum QuotaInfo {
    /// HTTP status of an error envelope (400 bad name, 403 API disabled, 404 unknown id).
    Error(u16),
    /// (region dimension or None for the default entry, details.value as Google
    /// writes an int64: a JSON string, absent when 0).
    Dims(Vec<(Option<&'static str>, Option<&'static str>)>),
}

pub struct Cloud {
    pub requests: Vec<Req>,
    /// PREEMPTIBLE_NVIDIA_H100_GPUS of regions.get, listed only when `legacy_h100`.
    pub quotas: HashMap<String, (f64, f64)>,
    pub legacy_h100: bool,
    /// Other metrics of regions.get by region: (metric, limit, usage).
    pub region_quotas: HashMap<String, Vec<(&'static str, f64, f64)>>,
    pub quota_info: QuotaInfo,
    /// VMs that are not ours, (zone, instance), in the aggregated list.
    pub foreign_vms: Vec<(String, Value)>,
    pub images: BTreeMap<String, String>,
    /// An image's description, as `image import` wrote it.
    pub image_descriptions: BTreeMap<String, String>,
    /// An image's labels, as `image import` wrote them.
    pub image_labels: BTreeMap<String, Value>,
    /// `images.delete` answers 403.
    pub fail_image_delete: bool,
    /// None: the Billing API answers 403 and the client uses its table.
    pub catalog: Option<Vec<Value>>,
    pub vms: BTreeMap<String, Vm>,
    pub disks: Vec<(String, Value)>,
    /// (region or "global", address)
    pub addresses: Vec<(String, Value)>,
    pub firewalls: BTreeMap<String, Value>,
    pub buckets: BTreeMap<String, Value>,
    pub objects: BTreeMap<String, BTreeMap<String, u64>>,
    pub boot: Boot,
    /// The VM insert operation stays RUNNING while set.
    pub hold_vm_insert: bool,
    /// The image insert operation of `image import` stays RUNNING while set.
    pub hold_image_insert: bool,
    /// (code, message) of a failed VM insert operation.
    pub insert_error: Option<(String, String)>,
    pub fail_vm_delete: bool,
    pub serial: String,
    /// Appended to `serial`, one piece per read of the serial port.
    pub serial_more: Vec<String>,
    /// The serial port answers this HTTP status, with an error body.
    pub serial_error: Option<u16>,
    pub ops: HashMap<String, (String, Value)>,
    pub next_op: u64,
    pub uploads: HashMap<String, (String, String, u64, u64)>,
    /// TOKEN's lifetime from `token_start`: tokeninfo gives what is left, and
    /// past it every call is refused (401, tokeninfo 400). None: tokeninfo
    /// gives no `expires_in` and the token never expires.
    pub token_life: Option<Duration>,
    pub token_start: Instant,
    /// Calls refused because TOKEN had expired.
    pub expired_uses: usize,
    /// What the fake name server answers `myaddr` with; None: no answer.
    pub public_addr: Option<String>,
    /// Queries the fake name server received.
    pub dns_queries: usize,
}

pub fn sku(desc: &str, usage: &str, region: &str, unit: &str, units: &str, nanos: i64) -> Value {
    json!({
        "description": desc,
        "category": {"usageType": usage, "resourceFamily": "Compute"},
        "serviceRegions": [region],
        "pricingInfo": [{"pricingExpression": {"usageUnit": unit,
            "tieredRates": [{"startUsageAmount": 0,
              "unitPrice": {"currencyCode": "USD", "units": units, "nanos": nanos}}]}}]
    })
}

/// GPU 5.57 + 26 vCPU x 0.01 + 234 GiB x 0.001 = 6.064 USD/h in us-central1.
pub fn catalog() -> Vec<Value> {
    vec![
        sku(
            "Spot Preemptible Nvidia H100 80GB GPU running in Americas",
            "Preemptible",
            "us-central1",
            "h",
            "5",
            570_000_000,
        ),
        sku(
            "Nvidia H100 80GB GPU running in Americas",
            "OnDemand",
            "us-central1",
            "h",
            "11",
            60_000_000,
        ),
        sku(
            "Spot Preemptible A3 Instance Core running in Americas",
            "Preemptible",
            "us-central1",
            "h",
            "0",
            10_000_000,
        ),
        sku(
            "Spot Preemptible A3 Instance Ram running in Americas",
            "Preemptible",
            "us-central1",
            "GiBy.h",
            "0",
            1_000_000,
        ),
        sku(
            "Spot Preemptible Nvidia H100 80GB GPU running in Netherlands",
            "Preemptible",
            "europe-west4",
            "h",
            "7",
            0,
        ),
    ]
}

impl Default for Cloud {
    fn default() -> Self {
        let mut quotas = HashMap::new();
        quotas.insert("europe-west4".to_string(), (0.0, 0.0));
        quotas.insert("us-central1".to_string(), (1.0, 0.0));
        quotas.insert("us-east5".to_string(), (0.0, 0.0));
        // As read on a real project on 2026-10-17: spot vCPU and local SSD quota in us-central1 only.
        let mut region_quotas = HashMap::new();
        for (rg, cpus, ssd) in [
            ("europe-west4", 0.0, None),
            ("us-central1", 26.0, Some(750.0)),
            ("us-east5", 0.0, None),
        ] {
            let mut v = vec![("PREEMPTIBLE_CPUS", cpus, 0.0)];
            if let Some(gb) = ssd {
                v.push(("PREEMPTIBLE_LOCAL_SSD_GB", gb, 0.0));
            }
            region_quotas.insert(rg.to_string(), v);
        }
        let mut images = BTreeMap::new();
        // The image of a session without --reference, and the one the test references name.
        images.insert(
            x0ne_engine_cloud::DEFAULT_IMAGE.to_string(),
            "READY".to_string(),
        );
        images.insert("x0ne-cvm-0-1-0".to_string(), "READY".to_string());
        // Both made by `image import`, which labels what it creates.
        let image_labels = images
            .keys()
            .map(|n| (n.clone(), json!({"x0ne-engine": "image"})))
            .collect();
        Cloud {
            requests: Vec::new(),
            quotas,
            legacy_h100: true,
            region_quotas,
            quota_info: QuotaInfo::Dims(vec![(Some("us-central1"), Some("1")), (None, None)]),
            foreign_vms: Vec::new(),
            images,
            image_descriptions: BTreeMap::new(),
            image_labels,
            fail_image_delete: false,
            catalog: Some(catalog()),
            vms: BTreeMap::new(),
            disks: Vec::new(),
            addresses: Vec::new(),
            firewalls: BTreeMap::new(),
            buckets: BTreeMap::new(),
            objects: BTreeMap::new(),
            boot: Boot::After(2),
            hold_vm_insert: false,
            hold_image_insert: false,
            insert_error: None,
            fail_vm_delete: false,
            serial:
                "SeaBIOS\nTDX guest boot\n\x1b[2J\x1b]0;evil title\x07kernel panic: SERIAL-MARKER\n"
                    .into(),
            serial_more: Vec::new(),
            serial_error: None,
            ops: HashMap::new(),
            next_op: 0,
            uploads: HashMap::new(),
            token_life: Some(Duration::from_secs(3599)),
            token_start: Instant::now(),
            expired_uses: 0,
            public_addr: Some(CLIENT_IP.into()),
            dns_queries: 0,
        }
    }
}

/// Google's name server as `myaddr` asks it: the question echoed with one TXT
/// holding `Cloud::public_addr`, or no answer at all when it is None.
async fn fake_name_server(sock: tokio::net::UdpSocket, cloud: Arc<Mutex<Cloud>>) {
    let mut buf = [0u8; 512];
    while let Ok((n, from)) = sock.recv_from(&mut buf).await {
        let q = &buf[..n];
        let addr = {
            let mut c = cloud.lock().unwrap_or_else(|e| e.into_inner());
            c.dns_queries += 1;
            c.public_addr.clone()
        };
        let Some(text) = addr.filter(|_| n > 12) else {
            continue;
        };
        let mut r = q[..2].to_vec();
        r.extend_from_slice(&[0x84, 0x00, 0, 1, 0, 1, 0, 0, 0, 0]);
        r.extend_from_slice(&q[12..]);
        r.extend_from_slice(&[0xc0, 0x0c, 0, 16, 0, 1, 0, 0, 0, 0]);
        r.extend_from_slice(&(text.len() as u16 + 1).to_be_bytes());
        r.push(text.len() as u8);
        r.extend_from_slice(text.as_bytes());
        let _ = sock.send_to(&r, from).await;
    }
}

pub struct FakeGcp {
    pub cloud: Arc<Mutex<Cloud>>,
    pub base: String,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
}

impl FakeGcp {
    pub fn start(cloud: Cloud) -> FakeGcp {
        let cloud = Arc::new(Mutex::new(cloud));
        let (addr_tx, addr_rx) = std::sync::mpsc::channel();
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let state = cloud.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                // The fake name server of `myaddr` takes the same port number, in UDP.
                let (l, udp) = loop {
                    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                    if let Ok(u) = tokio::net::UdpSocket::bind(l.local_addr().unwrap()).await {
                        break (l, u);
                    }
                };
                addr_tx.send(l.local_addr().unwrap()).unwrap();
                tokio::spawn(fake_name_server(udp, state.clone()));
                let app = Router::new().fallback(handle).with_state(state);
                let _ = axum::serve(l, app)
                    .with_graceful_shutdown(async {
                        let _ = stop_rx.await;
                    })
                    .await;
            });
        });
        let addr = addr_rx.recv().unwrap();
        FakeGcp {
            cloud,
            base: format!("http://{addr}"),
            stop: Some(stop_tx),
        }
    }

    pub fn with<R>(&self, f: impl FnOnce(&mut Cloud) -> R) -> R {
        f(&mut self.cloud.lock().unwrap_or_else(|e| e.into_inner()))
    }

    pub fn requests(&self) -> Vec<Req> {
        self.with(|c| c.requests.clone())
    }

    pub fn find(&self, method: &str, path_suffix: &str) -> Vec<Req> {
        self.requests()
            .into_iter()
            .filter(|r| r.is(method, path_suffix))
            .collect()
    }

    pub fn count(&self, method: &str, path_suffix: &str) -> usize {
        self.find(method, path_suffix).len()
    }

    /// Requests whose path contains `part`, by method.
    pub fn count_containing(&self, method: &str, part: &str) -> usize {
        self.requests()
            .iter()
            .filter(|r| r.method == method && r.path.contains(part))
            .count()
    }

    pub fn wait_request(&self, method: &str, path_suffix: &str) {
        let until = Instant::now() + WAIT;
        while self.count(method, path_suffix) == 0 {
            assert!(
                Instant::now() < until,
                "no {method} …{path_suffix} within {WAIT:?}"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    pub fn deletes(&self) -> Vec<String> {
        self.requests()
            .into_iter()
            .filter(|r| r.method == "DELETE")
            .map(|r| r.path)
            .collect()
    }
}

impl Drop for FakeGcp {
    fn drop(&mut self) {
        if let Some(s) = self.stop.take() {
            let _ = s.send(());
        }
    }
}

fn reply(status: u16, v: Value) -> Response {
    (
        StatusCode::from_u16(status).unwrap(),
        [("content-type", "application/json")],
        v.to_string(),
    )
        .into_response()
}

fn empty(status: u16) -> Response {
    StatusCode::from_u16(status).unwrap().into_response()
}

fn not_found() -> Response {
    reply(
        404,
        json!({"error": {"code": 404, "message": "The resource was not found",
                         "errors": [{"reason": "notFound"}]}}),
    )
}

fn query_param(q: &str, key: &str) -> Option<String> {
    q.split('&')
        .filter_map(|kv| kv.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v.to_string())
}

async fn handle(
    State(c): State<Arc<Mutex<Cloud>>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let mut c = c.lock().unwrap_or_else(|e| e.into_inner());
    let r = Req {
        method: method.to_string(),
        path: uri.path().to_string(),
        query: uri.query().unwrap_or("").to_string(),
        body: body.to_vec(),
        auth: headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(String::from),
    };
    c.requests.push(r.clone());
    let host = headers
        .get("host")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    route(&mut c, &r, &headers, &host)
}

fn route(c: &mut Cloud, r: &Req, headers: &HeaderMap, host: &str) -> Response {
    let segs: Vec<&str> = r.path.trim_start_matches('/').split('/').collect();
    let left = c.token_life.map(|l| l.checked_sub(c.token_start.elapsed()));
    if matches!(left, Some(None)) && (r.auth.is_some() || segs == ["tokeninfo"]) {
        c.expired_uses += 1;
    }
    if r.method == "POST" && segs == ["tokeninfo"] {
        return if r.body == format!("access_token={TOKEN}").as_bytes() && left != Some(None) {
            let mut v = json!({"azp": "32555940559.apps.googleusercontent.com",
                   "scope": "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/userinfo.email",
                   "email": "tester@example.com", "email_verified": "true"});
            if let Some(Some(l)) = left {
                // A string, as Google writes it.
                v["expires_in"] = json!(l.as_secs().to_string());
            }
            reply(200, v)
        } else {
            reply(
                400,
                json!({"error": "invalid_token", "error_description": "Invalid Value"}),
            )
        };
    }
    if segs.first() == Some(&"upload-session") && r.method == "PUT" {
        return upload_chunk(c, r, headers, segs.get(1).copied().unwrap_or(""));
    }
    if r.auth.as_deref() != Some(format!("Bearer {TOKEN}").as_str()) || left == Some(None) {
        return reply(
            401,
            json!({"error": {"code": 401, "message": "Request had invalid authentication credentials.",
                             "status": "UNAUTHENTICATED"}}),
        );
    }
    match segs.as_slice() {
        ["compute", "v1", "projects", p, rest @ ..] if *p == PROJECT => compute(c, r, rest),
        ["v1", "services", "6F81-5844-456A", "skus"] => billing(c, r),
        ["v1", "projects", p, "locations", "global", "services", "compute.googleapis.com", "quotaInfos", id]
            if *p == PROJECT && r.method == "GET" =>
        {
            quota_info(c, id)
        }
        ["storage", "v1", rest @ ..] => storage(c, r, rest),
        ["upload", "storage", "v1", "b", bucket, "o"] if r.method == "POST" => {
            upload_start(c, r, headers, bucket, host)
        }
        _ => not_found(),
    }
}

/// QuotaInfo as cloudquotas.googleapis.com v1 shapes it (2026-10-17 docs and a gcloud read).
fn quota_info(c: &Cloud, id: &str) -> Response {
    if id != "PREEMPTIBLE-NVIDIA-H100-GPUS-per-project-region" {
        return not_found();
    }
    match &c.quota_info {
        QuotaInfo::Error(403) => reply(
            403,
            json!({"error": {"code": 403, "status": "PERMISSION_DENIED",
                "message": "Cloud Quotas API has not been used in project 123 before or it is disabled.",
                "errors": [{"reason": "SERVICE_DISABLED"}]}}),
        ),
        QuotaInfo::Error(400) => reply(
            400,
            json!({"error": {"code": 400, "status": "INVALID_ARGUMENT",
                "message": "Request contains an invalid argument."}}),
        ),
        QuotaInfo::Error(_) => not_found(),
        QuotaInfo::Dims(dims) => {
            let infos: Vec<Value> = dims
                .iter()
                .map(|(rg, value)| {
                    let details = match value {
                        Some(v) => json!({ "value": v }),
                        None => json!({}),
                    };
                    match rg {
                        Some(rg) => json!({"dimensions": {"region": rg}, "details": details,
                                           "applicableLocations": [rg]}),
                        None => json!({"details": details,
                                       "applicableLocations": ["europe-west4", "us-east5"]}),
                    }
                })
                .collect();
            reply(
                200,
                json!({"name": format!("projects/123/locations/global/services/compute.googleapis.com/quotaInfos/{id}"),
                       "quotaId": id, "metric": "compute.googleapis.com/preemptible_nvidia_h100_gpus",
                       "service": "compute.googleapis.com", "isPrecise": true, "containerType": "PROJECT",
                       "dimensions": ["region"], "dimensionsInfos": infos}),
            )
        }
    }
}

fn new_op(
    c: &mut Cloud,
    kind: &str,
    scope: Option<(&str, &str)>,
    error: Option<Value>,
) -> Response {
    c.next_op += 1;
    let name = format!("operation-{}-{kind}", c.next_op);
    let mut v = json!({"kind": "compute#operation", "name": name, "status": "RUNNING"});
    match scope {
        Some(("zone", z)) => {
            v["zone"] = json!(format!(
                "https://www.googleapis.com/compute/v1/projects/{PROJECT}/zones/{z}"
            ))
        }
        Some(("region", rg)) => {
            v["region"] = json!(format!(
                "https://www.googleapis.com/compute/v1/projects/{PROJECT}/regions/{rg}"
            ))
        }
        _ => {}
    }
    let mut done = v.clone();
    done["status"] = json!("DONE");
    if let Some(e) = error {
        done["error"] = e;
        done["httpErrorStatusCode"] = json!(503);
    }
    c.ops.insert(name, (kind.to_string(), done));
    reply(200, v)
}

fn get_op(c: &Cloud, name: &str) -> Response {
    match c.ops.get(name) {
        Some((kind, _))
            if (kind == "vm-insert" && c.hold_vm_insert)
                || (kind == "image-insert" && c.hold_image_insert) =>
        {
            let mut v = c.ops[name].1.clone();
            v["status"] = json!("RUNNING");
            v.as_object_mut().unwrap().remove("error");
            reply(200, v)
        }
        Some((_, done)) => reply(200, done.clone()),
        None => not_found(),
    }
}

fn grouped(items: impl Iterator<Item = (String, Value)>, scope: &str, key: &str) -> Value {
    let mut by: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for (loc, v) in items {
        by.entry(format!("{scope}/{loc}")).or_default().push(v);
    }
    let mut items = serde_json::Map::new();
    for (k, v) in by {
        items.insert(k, json!({ key: v }));
    }
    // A scope with nothing in it, as Google returns them.
    items.insert(
        format!("{scope}/asia-east1-a"),
        json!({"warning": {"code": "NO_RESULTS_ON_PAGE", "message": "There are no results"}}),
    );
    json!({ "items": items })
}

fn compute(c: &mut Cloud, r: &Req, rest: &[&str]) -> Response {
    let m = r.method.as_str();
    match (m, rest) {
        ("GET", ["regions", rg]) => {
            let mut quotas = vec![json!({"metric": "CPUS", "limit": 2400.0, "usage": 0.0})];
            for (m, limit, usage) in c.region_quotas.get(*rg).into_iter().flatten() {
                quotas.push(json!({"metric": m, "limit": limit, "usage": usage}));
            }
            if c.legacy_h100 {
                let (limit, usage) = c.quotas.get(*rg).copied().unwrap_or((0.0, 0.0));
                quotas.push(
                    json!({"metric": "PREEMPTIBLE_NVIDIA_H100_GPUS", "limit": limit, "usage": usage}),
                );
            }
            reply(200, json!({"name": rg, "quotas": quotas}))
        }
        ("GET", ["zones", _z, "machineTypes", mt]) if *mt == "a3-highgpu-1g" => reply(
            200,
            json!({"name": mt, "guestCpus": 26, "memoryMb": 239616,
                   "accelerators": [{"guestAcceleratorType": "nvidia-h100-80gb", "guestAcceleratorCount": 1}]}),
        ),
        ("GET", ["global", "networks", "default"]) => reply(200, json!({"name": "default"})),
        ("GET", ["global", "images", name]) => match c.images.get(*name) {
            Some(s) => {
                let mut v = json!({"name": name, "status": s});
                if let Some(d) = c.image_descriptions.get(*name) {
                    v["description"] = json!(d);
                }
                if let Some(l) = c.image_labels.get(*name) {
                    v["labels"] = l.clone();
                }
                reply(200, v)
            }
            None => not_found(),
        },
        ("DELETE", ["global", "images", name]) => {
            if c.fail_image_delete {
                return reply(
                    403,
                    json!({"error": {"code": 403, "message": "Required 'compute.images.delete' permission",
                                     "errors": [{"reason": "forbidden"}]}}),
                );
            }
            if c.images.remove(*name).is_none() {
                return not_found();
            }
            c.image_descriptions.remove(*name);
            c.image_labels.remove(*name);
            new_op(c, "image-delete", None, None)
        }
        ("POST", ["global", "images"]) => {
            let body = r.json();
            let name = body["name"].as_str().unwrap_or("").to_string();
            if let Some(d) = body["description"].as_str() {
                c.image_descriptions.insert(name.clone(), d.to_string());
            }
            if body["labels"].is_object() {
                c.image_labels.insert(name.clone(), body["labels"].clone());
            }
            c.images.insert(name, "READY".into());
            new_op(c, "image-insert", None, None)
        }
        ("GET", ["zones" | "regions", _, "operations", op] | ["global", "operations", op]) => {
            get_op(c, op)
        }
        ("GET", ["aggregated", "instances"]) => {
            let vms: Vec<(String, Value)> = c
                .vms
                .iter()
                .map(|(n, v)| {
                    (
                        v.zone.clone(),
                        json!({"name": n, "status": v.status, "labels": v.labels}),
                    )
                })
                .chain(c.foreign_vms.iter().cloned())
                .collect();
            reply(200, grouped(vms.into_iter(), "zones", "instances"))
        }
        ("GET", ["aggregated", "disks"]) => {
            reply(200, grouped(c.disks.clone().into_iter(), "zones", "disks"))
        }
        ("GET", ["aggregated", "addresses"]) => {
            let regional = c.addresses.iter().filter(|(l, _)| l != "global").cloned();
            reply(200, grouped(regional, "regions", "addresses"))
        }
        ("GET", ["global", "addresses"]) => {
            let g: Vec<Value> = c
                .addresses
                .iter()
                .filter(|(l, _)| l == "global")
                .map(|(_, v)| v.clone())
                .collect();
            reply(200, json!({ "items": g }))
        }
        ("GET", ["global", "firewalls"]) => reply(
            200,
            json!({ "items": c.firewalls.values().cloned().collect::<Vec<_>>() }),
        ),
        ("POST", ["global", "firewalls"]) => {
            let body = r.json();
            let name = body["name"].as_str().unwrap_or("").to_string();
            if c.firewalls.contains_key(&name) {
                return reply(
                    409,
                    json!({"error": {"code": 409, "message": "already exists",
                                                    "errors": [{"reason": "alreadyExists"}]}}),
                );
            }
            c.firewalls.insert(name, body);
            new_op(c, "fw-insert", None, None)
        }
        ("DELETE", ["global", "firewalls", name]) => {
            if c.firewalls.remove(*name).is_none() {
                return not_found();
            }
            new_op(c, "fw-delete", None, None)
        }
        ("POST", ["zones", z, "instances"]) => {
            let body = r.json();
            let name = body["name"].as_str().unwrap_or("").to_string();
            let zone = z.to_string();
            if let Some((code, msg)) = c.insert_error.clone() {
                // Google quotes a value it refuses: `{metadata:<key>}` is that value.
                let mut msg = msg;
                for item in body["metadata"]["items"].as_array().into_iter().flatten() {
                    let k = item["key"].as_str().unwrap_or("");
                    let v = item["value"].as_str().unwrap_or("");
                    msg = msg.replace(&format!("{{metadata:{k}}}"), v);
                }
                let err = json!({"errors": [{"code": code, "message": msg}]});
                return new_op(c, "vm-insert", Some(("zone", &zone)), Some(err));
            }
            c.vms.insert(
                name,
                Vm {
                    zone: zone.clone(),
                    status: "PROVISIONING".into(),
                    labels: body["labels"].clone(),
                    gets: 0,
                },
            );
            new_op(c, "vm-insert", Some(("zone", &zone)), None)
        }
        ("GET", ["zones", _z, "instances", name]) => {
            let boot = c.boot;
            let Some(vm) = c.vms.get_mut(*name) else {
                return not_found();
            };
            vm.gets += 1;
            let running = match boot {
                Boot::Never => false,
                Boot::After(n) => vm.gets > n,
                Boot::Preempted { after, then } => {
                    if vm.gets > after + then {
                        c.vms.remove(*name);
                        return not_found();
                    }
                    vm.gets > after
                }
            };
            if vm.status == "PROVISIONING" || vm.status == "STAGING" || vm.status == "RUNNING" {
                vm.status = if running { "RUNNING" } else { "STAGING" }.into();
            }
            let ip = if vm.status == "RUNNING" {
                json!(VM_IP)
            } else {
                Value::Null
            };
            reply(
                200,
                json!({"name": name, "status": vm.status, "labels": vm.labels,
                       "networkInterfaces": [{"accessConfigs": [{"type": "ONE_TO_ONE_NAT", "natIP": ip}]}]}),
            )
        }
        ("DELETE", ["zones", z, "instances", name]) => {
            if c.fail_vm_delete {
                return reply(
                    500,
                    json!({"error": {"code": 500, "message": "Internal error \u{1b}[31m",
                                                    "errors": [{"reason": "backendError"}]}}),
                );
            }
            if c.vms.remove(*name).is_none() {
                return not_found();
            }
            let zone = z.to_string();
            new_op(c, "vm-delete", Some(("zone", &zone)), None)
        }
        ("GET", ["zones", _z, "instances", _name, "serialPort"]) => {
            if let Some(code) = c.serial_error {
                return reply(
                    code,
                    json!({"error": {"code": code, "message": "Required 'compute.instances.getSerialPortOutput' permission \u{1b}[31m",
                                     "errors": [{"reason": "forbidden"}]}}),
                );
            }
            if !c.serial_more.is_empty() {
                let piece = c.serial_more.remove(0);
                c.serial.push_str(&piece);
            }
            // Byte offsets, int64 as strings, as compute v1 answers them.
            let Some(start) = query_param(&r.query, "start").and_then(|s| s.parse::<usize>().ok())
            else {
                return reply(400, json!({"error": {"code": 400, "message": "start"}}));
            };
            let mut from = start.min(c.serial.len());
            while !c.serial.is_char_boundary(from) {
                from -= 1;
            }
            reply(
                200,
                json!({"contents": &c.serial[from..], "start": from.to_string(),
                       "next": c.serial.len().max(start).to_string()}),
            )
        }
        ("DELETE", ["zones", z, "disks", name]) => {
            let before = c.disks.len();
            c.disks
                .retain(|(dz, d)| !(dz == z && d["name"].as_str() == Some(*name)));
            if c.disks.len() == before {
                return not_found();
            }
            let zone = z.to_string();
            new_op(c, "disk-delete", Some(("zone", &zone)), None)
        }
        ("DELETE", ["regions", rg, "addresses", name]) => {
            let before = c.addresses.len();
            c.addresses
                .retain(|(l, a)| !(l == rg && a["name"].as_str() == Some(*name)));
            if c.addresses.len() == before {
                return not_found();
            }
            let region = rg.to_string();
            new_op(c, "address-delete", Some(("region", &region)), None)
        }
        ("DELETE", ["global", "addresses", name]) => {
            let before = c.addresses.len();
            c.addresses
                .retain(|(l, a)| !(l == "global" && a["name"].as_str() == Some(*name)));
            if c.addresses.len() == before {
                return not_found();
            }
            new_op(c, "address-delete", None, None)
        }
        _ => not_found(),
    }
}

fn billing(c: &Cloud, r: &Req) -> Response {
    let Some(skus) = &c.catalog else {
        return reply(
            403,
            json!({"error": {"code": 403, "message": "Cloud Billing API has not been used in this project",
                             "status": "PERMISSION_DENIED"}}),
        );
    };
    let half = skus.len() / 2;
    if query_param(&r.query, "pageToken").as_deref() == Some("page-2") {
        reply(200, json!({"skus": skus[half..]}))
    } else {
        reply(
            200,
            json!({"skus": skus[..half], "nextPageToken": "page-2"}),
        )
    }
}

fn storage(c: &mut Cloud, r: &Req, rest: &[&str]) -> Response {
    match (r.method.as_str(), rest) {
        ("GET", ["b"]) => {
            let prefix = query_param(&r.query, "prefix").unwrap_or_default();
            let items: Vec<Value> = c
                .buckets
                .iter()
                .filter(|(n, _)| n.starts_with(&prefix))
                .map(|(_, b)| b.clone())
                .collect();
            reply(200, json!({ "items": items }))
        }
        ("POST", ["b"]) => {
            let body = r.json();
            let name = body["name"].as_str().unwrap_or("").to_string();
            c.objects.entry(name.clone()).or_default();
            c.buckets.insert(name, body.clone());
            reply(200, body)
        }
        ("DELETE", ["b", bucket]) => {
            if c.objects.get(*bucket).is_some_and(|o| !o.is_empty()) {
                return reply(
                    409,
                    json!({"error": {"code": 409, "message": "The bucket you tried to delete is not empty."}}),
                );
            }
            c.objects.remove(*bucket);
            if c.buckets.remove(*bucket).is_none() {
                return not_found();
            }
            empty(204)
        }
        ("GET", ["b", bucket, "o"]) => match c.objects.get(*bucket) {
            Some(o) => reply(
                200,
                json!({"items": o.keys().map(|n| json!({"name": n, "generation": "1"})).collect::<Vec<_>>()}),
            ),
            None => not_found(),
        },
        ("DELETE", ["b", bucket, "o", object]) => {
            match c.objects.get_mut(*bucket).and_then(|o| o.remove(*object)) {
                Some(_) => empty(204),
                None => not_found(),
            }
        }
        _ => not_found(),
    }
}

fn upload_start(c: &mut Cloud, r: &Req, headers: &HeaderMap, bucket: &str, host: &str) -> Response {
    if !c.buckets.contains_key(bucket) {
        return not_found();
    }
    let object = query_param(&r.query, "name").unwrap_or_default();
    let total: u64 = headers
        .get("x-upload-content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let id = format!("u{}", c.uploads.len() + 1);
    c.uploads
        .insert(id.clone(), (bucket.to_string(), object, total, 0));
    (
        StatusCode::OK,
        [("location", format!("http://{host}/upload-session/{id}"))],
        "",
    )
        .into_response()
}

fn upload_chunk(c: &mut Cloud, r: &Req, headers: &HeaderMap, id: &str) -> Response {
    let Some((bucket, object, total, received)) = c.uploads.get(id).cloned() else {
        return not_found();
    };
    let range = headers
        .get("content-range")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let ok_range = format!(
        "bytes {}-{}/{}",
        received,
        received + r.body.len() as u64 - 1,
        total
    );
    if range != ok_range {
        return reply(
            400,
            json!({"error": {"code": 400, "message": format!("bad range {range}")}}),
        );
    }
    let got = received + r.body.len() as u64;
    c.uploads
        .insert(id.to_string(), (bucket.clone(), object.clone(), total, got));
    if got == total {
        c.objects
            .entry(bucket)
            .or_default()
            .insert(object.clone(), total);
        reply(200, json!({"name": object, "size": total.to_string()}))
    } else {
        (
            StatusCode::PERMANENT_REDIRECT,
            [("range", format!("bytes=0-{}", got - 1))],
            "",
        )
            .into_response()
    }
}

// ── fake relays ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Relay {
    /// config, then on connect: progress, verified, serving; `stop` → stopped.
    Ok,
    /// As `Ok`, with the real relay's verdict fields and a key file written
    /// where the real one is (`$XDG_RUNTIME_DIR/x0ne-engine/local.key`).
    Serve,
    /// As `Serve`, with a summary shaped like `remote_verify.rs` writes it
    /// (CPU, Firm, Image, GPU, Link, Model; values fake): the color golden.
    Proof,
    /// On connect: refused with this code, exit with it.
    Refuse(u8),
    /// On connect: exits with this code, no event.
    CrashAfterConnect(u8),
    /// Serving, then exits with this code 300 ms later.
    CrashWhileServing(u8),
    /// On connect: one progress event, then nothing (waits for `stop`).
    Hang,
    /// config_sha256 does not match config_b64.
    BadConfig,
    /// Exits with this code before any event.
    ExitBeforeConfig(u8),
    /// As `Ok`, but waits 1.2 s before its config: `up` reads the clock on
    /// both sides of that wait, so a second boundary always falls between.
    SlowConfig,
    /// As `Ok`, but its config names this image and disk sha256, whatever
    /// the reference.
    Image(&'static str, Option<&'static str>),
    /// As `Ok`, with no `image` in its config: a relay before 0.2.0-beta.6.
    NoImage,
    /// As `Serve`; once its chat has exited on `stop-the-session`, refused
    /// with this code a second later, and exits with it.
    RefuseAfterChat(u8),
    /// On connect: progress, verified this many seconds later, never serving
    /// (waits for `stop`).
    LateVerdict(u8),
    /// As `Ok`, but its config names the build's TCB policy (transparent,
    /// floor 22, one day) whatever its `--tcb-policy`, `--min-tcb-evaluation`
    /// and `--max-collateral-age`: a relay that takes the options and
    /// ignores them.
    IgnoresTcbOptions,
    /// A build that pins no archive key (99) or no transparency log (100):
    /// `pins` and a session are both refused with this code, before any config.
    NoPins(u8),
    /// A debug test build: its `pins` names the TEST key and the TEST log and
    /// says `"test": true`; a session would go on.
    TestPins,
}

impl Relay {
    fn mode(self) -> (&'static str, u8) {
        match self {
            Relay::Ok => ("ok", 0),
            Relay::Serve => ("serve", 0),
            Relay::Proof => ("proof", 0),
            Relay::Refuse(c) => ("refuse", c),
            Relay::CrashAfterConnect(c) => ("crash-after-connect", c),
            Relay::CrashWhileServing(c) => ("crash-while-serving", c),
            Relay::Hang => ("hang", 0),
            Relay::BadConfig => ("bad-config", 0),
            Relay::ExitBeforeConfig(c) => ("exit-before-config", c),
            Relay::SlowConfig => ("slow-config", 0),
            Relay::Image(..) => ("image", 0),
            Relay::NoImage => ("no-image", 0),
            Relay::RefuseAfterChat(c) => ("refuse-after-chat", c),
            Relay::LateVerdict(s) => ("late-verdict", s),
            Relay::IgnoresTcbOptions => ("ignores-tcb-options", 0),
            Relay::NoPins(c) => ("no-pins", c),
            Relay::TestPins => ("test-pins", 0),
        }
    }

    /// Shell that overrides `img` and `dsha`, the image the config names.
    fn forced_image(self) -> String {
        match self {
            Relay::Image(name, sha) => format!(
                "img='{name}'; dsha='{}'",
                sha.map_or("null".into(), |s| format!("\"{s}\""))
            ),
            _ => String::new(),
        }
    }
}

/// Writes what it saw to `<log>.argv`, `.env`, `.config`, `.connect`, `.cmds`;
/// `pins` writes `<log>.pins-argv` and `.pins-env` only.
/// Prints on stderr what the real relay prints there (`relay.rs`), unless `--quiet`.
/// Its config names the image of `--reference` as the real relay does, or
/// `DEFAULT_IMAGE` and no disk sha256 without one, and the TCB policy of
/// `--tcb-policy`, `--min-tcb-evaluation` (strict when given alone) and
/// `--max-collateral-age`, or the build's (transparent, its floor).
pub fn relay_script(log: &Path, relay: Relay) -> String {
    let (mode, code) = relay.mode();
    let forced_image = relay.forced_image();
    let default_image = x0ne_engine_cloud::DEFAULT_IMAGE;
    let stopped = x0ne_engine_cloud::chat::CHAT_SESSION_STOPPED;
    let (floor, max_age) = (
        x0ne_engine_cloud::TCB_EVALUATION_FLOOR,
        x0ne_engine_cloud::MAX_COLLATERAL_AGE_SECS,
    );
    format!(
        r#"#!/bin/bash
# Fake x0ne-engine-proxy for tests: speaks I4 on stdin/stdout.
set -u
LOG='{log}'
MODE='{mode}'
CODE={code}
no_pins() {{
  printf '{{"v":1,"event":"refused","code":%s,"reason":"image reference: this build pins no x0ne archive key (fake)","evidence_file":null}}\n' "$CODE"
  exit "$CODE"
}}
# `pins`: what this build pins, as `up` asks it before the token.
if [ "${{1:-}}" = pins ]; then
  printf '%s\n' "$@" > "$LOG.pins-argv"
  env > "$LOG.pins-env"
  [ "$MODE" = no-pins ] && no_pins
  if [ "$MODE" = test-pins ]; then
    printf '{{"v":1,"event":"pins","archive_key":"FAKE","transparency_log":"test.invalid/x0ne-engine/tlog","test":true}}\n'
    exit 0
  fi
  printf '{{"v":1,"event":"pins","archive_key":"FAKE","transparency_log":"fake.invalid/log"}}\n'
  exit 0
fi
# `chat`: the relay's terminal chat, as `run` and `chat` start it.
if [ "${{1:-}}" = chat ]; then
  printf '%s\n' "$@" > "$LOG.chat-argv"
  env > "$LOG.chat-env"
  printf 'Chat with fake (29 min left). /exit or Ctrl-D to leave.\n' >&2
  n=0
  while :; do
    [ -t 0 ] && printf '>>> '
    IFS= read -r line || {{ [ -t 0 ] && echo; exit 0; }}
    case "$line" in
      /exit|/bye) exit 0;;
      '') continue;;
      stop-the-session)
        : > "$LOG.chat-stopped"
        printf '  [receipt FAILED (fake): the session is stopping]\n' >&2
        exit {stopped};;
    esac
    n=$((n+1))
    printf 'echo %s: %s\n' "$n" "$line"
    printf '  [receipt verified by the relay (fake)]\n' >&2
  done
fi
printf '%s\n' "$@" > "$LOG.argv"
env > "$LOG.env"
QUIET=0
for a in "$@"; do [ "$a" = --quiet ] && QUIET=1; done
say() {{ [ "$QUIET" = 1 ] || printf 'x0ne-engine-proxy: %s\n' "$1" >&2; }}
model=""; exp=""; listen=""; ref=""; tcbmin={floor}; tcbage={max_age}; tcbmode=""
while [ $# -gt 0 ]; do
  case "$1" in
    --tcb-policy) [ "$MODE" = ignores-tcb-options ] || tcbmode="$2"; shift 2;;
    --min-tcb-evaluation) [ "$MODE" = ignores-tcb-options ] || {{ tcbmin="$2"; [ -n "$tcbmode" ] || tcbmode=strict; }}; shift 2;;
    --max-collateral-age) [ "$MODE" = ignores-tcb-options ] || tcbage="$2"; shift 2;;
    --reference) ref="$2"; shift 2;;
    --model) model="$2"; shift 2;;
    --expires-at-unix) exp="$2"; shift 2;;
    --listen) listen="$2"; shift 2;;
    *) shift;;
  esac
done
[ "$MODE" = exit-before-config ] && exit "$CODE"
[ "$MODE" = no-pins ] && no_pins
[ "$MODE" = slow-config ] && sleep 1.2
a=$(printf 'a%.0s' $(seq 64)); b=$(printf 'b%.0s' $(seq 64))
cfg=$(printf '{{"schema":"x0ne-engine/cloud-config/v1","model":"%s","client_spki_sha256":"%s","client_nonce":"%s","expires_at_unix":%s,"attest":"cpu+gpu","listen_port":443,"max_completion_tokens":4096}}' "$model" "$a" "$b" "$exp")
printf '%s' "$cfg" > "$LOG.config"
b64=$(printf '%s' "$cfg" | base64 -w0)
sha=$(printf '%s' "$cfg" | sha256sum | cut -d' ' -f1)
[ "$MODE" = bad-config ] && sha=$(printf '0%.0s' $(seq 64))
img='{default_image}'; dsha=null
if [ -n "$ref" ]; then
  img=$(sed -n 's/.*"gcp_image" *: *"\([^"]*\)".*/\1/p' "$ref" | head -n1)
  d=$(sed -n 's/.*"disk_tar_sha256" *: *"\([^"]*\)".*/\1/p' "$ref" | head -n1)
  [ -n "$d" ] && dsha="\"$d\""
fi
{forced_image}
imgjson=$(printf ',"image":{{"gcp_image":"%s","disk_tar_sha256":%s}}' "$img" "$dsha")
[ "$MODE" = no-image ] && imgjson=""
tcbjson=$(printf ',"tcb_policy":{{"mode":"%s","min_tcb_evaluation":%s,"max_collateral_age_secs":%s}}' "${{tcbmode:-transparent}}" "$tcbmin" "$tcbage")
printf '{{"v":1,"event":"config","config_b64":"%s","config_sha256":"%s"%s%s}}\n' "$b64" "$sha" "$imgjson" "$tcbjson"
IFS= read -r line || exit 0
printf '%s\n' "$line" > "$LOG.connect"
case "$MODE" in
  refuse)
    say 'verifying: checking the TDX quote'
    printf '{{"v":1,"event":"progress","stage":"verifying","detail":"checking the TDX quote"}}\n'
    say "REFUSED ($CODE): RTMR2 does not match the reference (fake)"
    printf '{{"v":1,"event":"refused","code":%s,"reason":"RTMR2 does not match the reference (fake)","evidence_file":"/fake/evidence.json"}}\n' "$CODE"
    exit "$CODE";;
  crash-after-connect)
    exit "$CODE";;
  hang)
    say 'connecting: waiting for the VM (fake)'
    printf '{{"v":1,"event":"progress","stage":"connecting","detail":"waiting for the VM (fake)"}}\n';;
  late-verdict)
    printf '{{"v":1,"event":"progress","stage":"verifying","detail":"checking the TDX quote (fake, slow)"}}\n'
    sleep "$CODE"
    printf '{{"v":1,"event":"verified","verdict":{{"lines":["CPU   fake verdict","GPU   fake verdict"]}},"limitations":["fake relay: nothing was verified"]}}\n';;
  *)
    say 'verifying: checking the TDX quote (fake)'
    printf '{{"v":1,"event":"progress","stage":"verifying","detail":"checking the TDX quote (fake)"}}\n'
    key=/fake/local.key
    if [ "$MODE" = serve ] || [ "$MODE" = proof ] || [ "$MODE" = refuse-after-chat ]; then
      d=$(printf 'd%.0s' $(seq 64)); k=$(printf 'e%.0s' $(seq 64))
      if [ "$MODE" = proof ]; then
        say '  CPU   Intel TDX quote genuine, TCB UpToDate / QE UpToDate, TDX module identity matched (Intel PCS, fetched now)'
        say '  Firm  Google-endorsed firmware (MRTD 0f0f0f0f..., svn 2, signed fake signer)'
        say '  Image x0ne-cvm 0.1.0: RTMR1, RTMR2 match the reference; RTMR3 empty; RTMR0 not among the known values (reported)'
        say '  GPU   1 GPU(s), confidential computing on (NRAS token, nonce bound to this session)'
        say '  Link  TLS key and your session config are bound to this quote'
        say "  Model $model, digest dddddddd... matches the catalog"
        printf '{{"v":1,"event":"verified","verdict":{{"summary":["CPU   Intel TDX quote genuine, TCB UpToDate / QE UpToDate, TDX module identity matched (Intel PCS, fetched now)","Firm  Google-endorsed firmware (MRTD 0f0f0f0f..., svn 2, signed fake signer)","Image x0ne-cvm 0.1.0: RTMR1, RTMR2 match the reference; RTMR3 empty; RTMR0 not among the known values (reported)","GPU   1 GPU(s), confidential computing on (NRAS token, nonce bound to this session)","Link  TLS key and your session config are bound to this quote","Model %s, digest dddddddd... matches the catalog"],"gpu_count":1,"model_name":"%s","model_digest_hex":"%s","session_pubkey_hex":"%s"}},"limitations":["fake relay: nothing was verified"]}}\n' "$model" "$model" "$d" "$k"
      else
        say '  CPU   fake TDX verdict'
        say '  GPU   1 GPU(s), fake NRAS verdict'
        printf '{{"v":1,"event":"verified","verdict":{{"summary":["CPU   fake TDX verdict","GPU   1 GPU(s), fake NRAS verdict"],"gpu_count":1,"model_name":"%s","model_digest_hex":"%s","session_pubkey_hex":"%s"}},"limitations":["fake relay: nothing was verified"]}}\n' "$model" "$d" "$k"
      fi
      key="$XDG_RUNTIME_DIR/x0ne-engine/local.key"
      (umask 077; mkdir -p "${{key%/*}}"; printf 'c%.0s' $(seq 64) > "$key"; echo >> "$key")
    else
      say '  CPU   fake verdict'
      say '  GPU   fake verdict'
      printf '{{"v":1,"event":"verified","verdict":{{"lines":["CPU   fake verdict","GPU   fake verdict"]}},"limitations":["fake relay: nothing was verified"]}}\n'
    fi
    say 'releasing: model loaded (fake)'
    printf '{{"v":1,"event":"progress","stage":"releasing","detail":"model loaded (fake)"}}\n'
    printf '{{"v":1,"event":"serving","listen":"%s","key_file":"%s","context_tokens":40960}}\n' "$listen" "$key"
    if [ "$MODE" = crash-while-serving ]; then sleep 0.3; exit "$CODE"; fi
    if [ "$MODE" = refuse-after-chat ]; then
      for i in $(seq 600); do [ -e "$LOG.chat-stopped" ] && break; sleep 0.05; done
      sleep 1
      say "REFUSED ($CODE): receipt verification failed (fake)"
      printf '{{"v":1,"event":"refused","code":%s,"reason":"receipt verification failed (fake)","evidence_file":null}}\n' "$CODE"
      exit "$CODE"
    fi;;
esac
while IFS= read -r line; do
  printf '%s\n' "$line" >> "$LOG.cmds"
  case "$line" in
    *'"cmd":"stop"'*) printf '{{"v":1,"event":"stopped"}}\n'; exit 0;;
  esac
done
exit 0
"#,
        log = log.display()
    )
}

pub fn write_exe(path: &Path, body: &str, mode: u32) {
    let _g = fork_lock();
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(mode)
        .open(path)
        .unwrap();
    f.write_all(body.as_bytes()).unwrap();
    f.sync_all().unwrap();
    drop(f);
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
}

// ── one directory per test ──────────────────────────────────────────────────

pub struct Env {
    pub dir: tempfile::TempDir,
    /// The binary under test, linked next to the fake relay.
    pub bin: PathBuf,
    pub runtime: PathBuf,
    pub state: PathBuf,
    pub homedir: PathBuf,
    /// First in $PATH: decoys that record it if they ever run.
    pub decoys: PathBuf,
    /// The installed references `up` reads, empty unless a test writes one:
    /// `X0NE_ENGINE_CLOUD_TEST_REFERENCE_DIR`, so the host's never count.
    pub references: PathBuf,
}

impl Env {
    pub fn new(relay: Option<Relay>) -> Env {
        let dir = tempfile::Builder::new()
            .prefix("cloud-")
            .tempdir_in(env!("CARGO_TARGET_TMPDIR"))
            .unwrap();
        let d = dir.path();
        let mk = |p: &Path| {
            fs::create_dir_all(p).unwrap();
            fs::set_permissions(p, fs::Permissions::from_mode(0o700)).unwrap();
        };
        let (bindir, runtime, state, home, decoys, references) = (
            d.join("bin"),
            d.join("run"),
            d.join("state"),
            d.join("home"),
            d.join("decoys"),
            d.join("reference"),
        );
        for p in [&bindir, &runtime, &state, &home, &decoys, &references] {
            mk(p);
        }
        let bin = bindir.join("x0ne-engine-cloud");
        {
            let _g = fork_lock();
            if fs::hard_link(BIN, &bin).is_err() {
                fs::copy(BIN, &bin).unwrap();
            }
        }
        if let Some(r) = relay {
            write_exe(
                &bindir.join("x0ne-engine-proxy"),
                &relay_script(&d.join("relay"), r),
                0o755,
            );
        }
        for name in ["gcloud", "x0ne-engine-proxy", "x0ne-engine-cloud"] {
            let marker = d.join(format!("decoy-ran-{name}"));
            write_exe(
                &decoys.join(name),
                &format!("#!/bin/sh\ntouch '{}'\nexit 0\n", marker.display()),
                0o755,
            );
        }
        Env {
            dir,
            bin,
            runtime,
            state,
            homedir: home,
            decoys,
            references,
        }
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    pub fn relay_log(&self, what: &str) -> String {
        fs::read_to_string(self.path(&format!("relay.{what}"))).unwrap_or_default()
    }

    pub fn decoys_ran(&self) -> Vec<String> {
        ["gcloud", "x0ne-engine-proxy", "x0ne-engine-cloud"]
            .iter()
            .filter(|n| self.path(&format!("decoy-ran-{n}")).exists())
            .map(|n| n.to_string())
            .collect()
    }

    /// The binary with a clean environment pointing at the fake cloud.
    pub fn command(&self, base: &str, args: &[&str]) -> Command {
        let mut c = Command::new(&self.bin);
        c.args(args)
            .env_clear()
            .env("PATH", format!("{}:/usr/bin:/bin", self.decoys.display()))
            .env("HOME", &self.homedir)
            .env("XDG_RUNTIME_DIR", &self.runtime)
            .env("XDG_STATE_HOME", &self.state)
            .env("GOOGLE_CLOUD_PROJECT", PROJECT)
            .env("X0NE_ENGINE_GCP_ENDPOINT", base)
            .env(
                "X0NE_ENGINE_CLOUD_TEST_MYADDR_SERVER",
                base.trim_start_matches("http://"),
            )
            .env("X0NE_ENGINE_CLOUD_TEST_POLL_MS", "40")
            .env("X0NE_ENGINE_CLOUD_TEST_REFERENCE_DIR", &self.references);
        c
    }

    pub fn session_state(&self) -> Option<Value> {
        let p = self.runtime.join("x0ne-engine/cloud-session.json");
        fs::read(p)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
    }

    /// Every file under the test directory whose bytes contain `needle`.
    pub fn files_containing(&self, needle: &[u8]) -> Vec<PathBuf> {
        fn walk(p: &Path, needle: &[u8], out: &mut Vec<PathBuf>) {
            let Ok(rd) = fs::read_dir(p) else { return };
            for e in rd.flatten() {
                let path = e.path();
                let Ok(ft) = e.file_type() else { continue };
                if ft.is_dir() {
                    walk(&path, needle, out);
                } else if ft.is_file() {
                    if let Ok(b) = fs::read(&path) {
                        if contains(&b, needle) {
                            out.push(path);
                        }
                    }
                }
            }
        }
        let mut out = Vec::new();
        walk(self.dir.path(), needle, &mut out);
        out
    }
}

// ── hiding the machine's own files from the binary ──────────────────────────

/// Runs `cmd` in its own user and mount namespaces with `/dev/null` bound
/// over each of `files`: the binary sees no regular file there, and the
/// machine is untouched. Same uid and gid inside; no privilege needed.
pub fn hide_files(cmd: &mut Command, files: &[&Path]) {
    fn put(path: &CStr, data: &[u8]) -> std::io::Result<()> {
        // SAFETY: open, write and close on a NUL-terminated path.
        unsafe {
            let fd = libc::open(path.as_ptr(), libc::O_WRONLY | libc::O_CLOEXEC);
            if fd < 0 {
                return Err(std::io::Error::last_os_error());
            }
            let n = libc::write(fd, data.as_ptr().cast(), data.len());
            libc::close(fd);
            if usize::try_from(n).ok() != Some(data.len()) {
                return Err(std::io::Error::last_os_error());
            }
        }
        Ok(())
    }
    // SAFETY: getuid and getgid cannot fail.
    let (uid, gid) = unsafe { (libc::getuid(), libc::getgid()) };
    let uid_map = format!("{uid} {uid} 1");
    let gid_map = format!("{gid} {gid} 1");
    let targets: Vec<std::ffi::CString> = files
        .iter()
        .map(|p| std::ffi::CString::new(p.as_os_str().as_encoded_bytes()).unwrap())
        .collect();
    // SAFETY: unshare, open/write/close and mount only, in the child before exec.
    unsafe {
        cmd.pre_exec(move || {
            let fail = || Err(std::io::Error::last_os_error());
            if libc::unshare(libc::CLONE_NEWUSER | libc::CLONE_NEWNS) != 0 {
                return fail();
            }
            put(c"/proc/self/setgroups", b"deny")?;
            put(c"/proc/self/uid_map", uid_map.as_bytes())?;
            put(c"/proc/self/gid_map", gid_map.as_bytes())?;
            let none = std::ptr::null();
            if libc::mount(
                none,
                c"/".as_ptr(),
                none,
                libc::MS_REC | libc::MS_PRIVATE,
                std::ptr::null(),
            ) != 0
            {
                return fail();
            }
            for t in &targets {
                if libc::mount(
                    c"/dev/null".as_ptr(),
                    t.as_ptr(),
                    none,
                    libc::MS_BIND,
                    std::ptr::null(),
                ) != 0
                {
                    return fail();
                }
            }
            Ok(())
        });
    }
}

/// Holds the session lock as another process would (a `down`, or an `up` for
/// another project at its "yes"; the fake serves one project) until dropped.
pub fn hold_lock(env: &Env) -> File {
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(env.runtime.join("x0ne-engine/cloud-session.lock"))
        .unwrap();
    // SAFETY: flock on a descriptor we own.
    let r = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(r, 0, "the session lock is held");
    f
}

/// `first`, serving, is killed (state "serving", its pid) and its VM goes at
/// --max-run. A second `up` past "yes", whose relay waits 2 s before its
/// config, then serves its control socket while the state is still the dead one's.
pub fn second_up_in_its_window(gcp: &FakeGcp, env: &Env, first: Run) -> (Value, Run, Pty) {
    first.signal(libc::SIGKILL);
    let _ = first.finish();
    let dead = env.session_state().unwrap();
    assert_eq!(dead["phase"], "serving");
    gcp.with(|c| {
        c.vms.clear();
        c.firewalls.clear();
    });
    let bindir = env.bin.parent().unwrap();
    let relay = bindir.join("x0ne-engine-proxy");
    fs::rename(&relay, bindir.join("x0ne-engine-proxy.orig")).unwrap();
    write_exe(
        &relay,
        "#!/bin/bash\n[ \"${1:-}\" = chat ] || sleep 2\nexec \"$(dirname \"$0\")/x0ne-engine-proxy.orig\" \"$@\"\n",
        0o755,
    );
    let pty = Pty::open();
    let mut b = env.command(&gcp.base, &["up", "--auth", "stdin", "--max-cost", "10"]);
    let b = spawn(env, &mut b, Some(TOKEN), "up-b", Some(&pty));
    pty.wait_screen("Type \"yes\"");
    pty.type_in(b"yes\n");
    let sock = env.runtime.join("x0ne-engine/cloud.sock");
    let until = Instant::now() + WAIT;
    while std::os::unix::net::UnixStream::connect(&sock).is_err() {
        assert!(
            Instant::now() < until,
            "the second up never served its socket"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(env.session_state().unwrap(), dead);
    (dead, b, pty)
}

// ── running the binary ──────────────────────────────────────────────────────

pub struct Run {
    pub child: Child,
    pub out: PathBuf,
    pub err: PathBuf,
}

pub struct Done {
    pub code: i32,
    pub out: String,
    pub err: String,
}

/// Starts `cmd` in a new session: with `pty` as its controlling terminal,
/// or with none (no /dev/tty). `token`, if any, is written to its stdin.
pub fn spawn(
    env: &Env,
    cmd: &mut Command,
    token: Option<&str>,
    name: &str,
    pty: Option<&Pty>,
) -> Run {
    let out = env.path(&format!("{name}.out"));
    let err = env.path(&format!("{name}.err"));
    cmd.stdout(File::create(&out).unwrap())
        .stderr(File::create(&err).unwrap());
    let child = start(cmd, token, pty);
    Run { child, out, err }
}

/// As [`spawn`], with standard output and error on the terminal too: what
/// it prints is read with [`Pty::screen`], and fails once the terminal hangs up.
pub fn spawn_on_tty(
    env: &Env,
    cmd: &mut Command,
    token: Option<&str>,
    name: &str,
    pty: &Pty,
) -> Run {
    cmd.stdout(pty.slave.try_clone().unwrap())
        .stderr(pty.slave.try_clone().unwrap());
    let child = start(cmd, token, Some(pty));
    let none = env.path(&format!("{name}.on-tty"));
    Run {
        child,
        out: none.clone(),
        err: none,
    }
}

/// As [`spawn`], with standard output or standard error (or both) on the
/// terminal and the other in a file: what goes to the terminal is read with
/// [`Pty::screen`], the file with [`Run::stdout`] or [`Run::stderr`].
pub fn spawn_split(
    env: &Env,
    cmd: &mut Command,
    token: Option<&str>,
    name: &str,
    pty: &Pty,
    out_on_tty: bool,
    err_on_tty: bool,
) -> Run {
    let out = env.path(&format!("{name}.out"));
    let err = env.path(&format!("{name}.err"));
    let tty = || Stdio::from(pty.slave.try_clone().unwrap());
    cmd.stdout(if out_on_tty {
        tty()
    } else {
        File::create(&out).unwrap().into()
    })
    .stderr(if err_on_tty {
        tty()
    } else {
        File::create(&err).unwrap().into()
    });
    let child = start(cmd, token, Some(pty));
    Run { child, out, err }
}

/// As a shell runs it: standard input, output and error on `pty`, which is
/// its controlling terminal. A token or an answer is typed with [`Pty::type_in`].
pub fn spawn_interactive(env: &Env, cmd: &mut Command, name: &str, pty: &Pty) -> Run {
    let tty = || Stdio::from(pty.slave.try_clone().unwrap());
    cmd.stdin(tty()).stdout(tty()).stderr(tty());
    let child = launch(cmd, None, Some(pty));
    let none = env.path(&format!("{name}.on-tty"));
    Run {
        child,
        out: none.clone(),
        err: none,
    }
}

fn start(cmd: &mut Command, token: Option<&str>, pty: Option<&Pty>) -> Child {
    cmd.stdin(if token.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    launch(cmd, token, pty)
}

/// A new session, `pty` as its controlling terminal if any, `token` on its stdin.
fn launch(cmd: &mut Command, token: Option<&str>, pty: Option<&Pty>) -> Child {
    let tty_fd = pty.map(|p| p.slave.as_raw_fd());
    // SAFETY: setsid and ioctl are async-signal-safe.
    unsafe {
        cmd.pre_exec(move || {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if let Some(fd) = tty_fd {
                if libc::ioctl(fd, libc::TIOCSCTTY as _, 0) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    let mut child = {
        let _g = fork_lock();
        cmd.spawn().unwrap()
    };
    if let Some(t) = token {
        let mut s = child.stdin.take().unwrap();
        let _ = s.write_all(t.as_bytes());
    }
    child
}

impl Run {
    pub fn pid(&self) -> i32 {
        self.child.id() as i32
    }
    pub fn stderr(&self) -> String {
        fs::read_to_string(&self.err).unwrap_or_default()
    }
    pub fn stdout(&self) -> String {
        fs::read_to_string(&self.out).unwrap_or_default()
    }
    pub fn wait_stderr(&mut self, needle: &str) {
        let until = Instant::now() + WAIT;
        loop {
            if self.stderr().contains(needle) {
                return;
            }
            if let Ok(Some(st)) = self.child.try_wait() {
                panic!("exited ({st}) before {needle:?}:\n{}", self.stderr());
            }
            assert!(
                Instant::now() < until,
                "no {needle:?} within {WAIT:?}:\n{}",
                self.stderr()
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    pub fn signal(&self, sig: i32) {
        // SAFETY: plain kill(2) on our own child.
        unsafe {
            libc::kill(self.pid(), sig);
        }
    }
    pub fn finish(mut self) -> Done {
        let until = Instant::now() + WAIT;
        let status = loop {
            if let Some(st) = self.child.try_wait().unwrap() {
                break st;
            }
            if Instant::now() > until {
                let _ = self.child.kill();
                panic!("still running after {WAIT:?}:\n{}", self.stderr());
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let code = status
            .code()
            .unwrap_or_else(|| 128 + status.signal().unwrap_or(0));
        Done {
            code,
            out: self.stdout(),
            err: self.stderr(),
        }
    }
}

// ── a pseudo-terminal ───────────────────────────────────────────────────────

pub struct Pty {
    master: File,
    slave: File,
    seen: Mutex<Vec<u8>>,
}

impl Pty {
    pub fn open() -> Pty {
        // SAFETY: standard posix_openpt sequence; every return value is checked.
        unsafe {
            let m = libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY | libc::O_CLOEXEC);
            assert!(m >= 0, "posix_openpt: {}", std::io::Error::last_os_error());
            assert_eq!(libc::grantpt(m), 0);
            assert_eq!(libc::unlockpt(m), 0);
            let mut buf = [0 as libc::c_char; 128];
            assert_eq!(libc::ptsname_r(m, buf.as_mut_ptr(), buf.len()), 0);
            let name = CStr::from_ptr(buf.as_ptr()).to_str().unwrap().to_string();
            let flags = libc::fcntl(m, libc::F_GETFL);
            libc::fcntl(m, libc::F_SETFL, flags | libc::O_NONBLOCK);
            let master = File::from_raw_fd(m);
            let slave = OpenOptions::new()
                .read(true)
                .write(true)
                .custom_flags(libc::O_NOCTTY | libc::O_CLOEXEC)
                .open(&name)
                .unwrap();
            Pty {
                master,
                slave,
                seen: Mutex::new(Vec::new()),
            }
        }
    }

    fn pump(&self) {
        let mut buf = [0u8; 4096];
        let mut seen = self.seen.lock().unwrap();
        loop {
            match (&self.master).read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => seen.extend_from_slice(&buf[..n]),
            }
        }
    }

    /// Everything the program wrote to the terminal so far, echo included.
    pub fn screen(&self) -> Vec<u8> {
        self.pump();
        self.seen.lock().unwrap().clone()
    }

    pub fn wait_screen(&self, needle: &str) {
        let until = Instant::now() + WAIT;
        while !contains(&self.screen(), needle.as_bytes()) {
            assert!(
                Instant::now() < until,
                "{needle:?} not on the terminal: {:?}",
                String::from_utf8_lossy(&self.screen())
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    pub fn echo_on(&self) -> bool {
        // SAFETY: tcgetattr fills a plain struct.
        unsafe {
            let mut t: libc::termios = std::mem::zeroed();
            assert_eq!(libc::tcgetattr(self.slave.as_raw_fd(), &mut t), 0);
            t.c_lflag & libc::ECHO != 0
        }
    }

    pub fn wait_echo_off(&self) {
        let until = Instant::now() + WAIT;
        while self.echo_on() {
            assert!(Instant::now() < until, "echo never turned off");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Typed on the keyboard.
    pub fn type_in(&self, bytes: &[u8]) {
        (&self.master).write_all(bytes).unwrap();
    }

    /// The terminal goes away (window closed, ssh dropped): the session
    /// leader gets SIGHUP, and writes to the terminal fail with EIO.
    pub fn hang_up(self) {
        drop(self);
    }
}

/// Golden comparison with the session id and the config replaced by markers.
pub fn normalize(body: &Value, session: &str) -> Value {
    let mut b = body.clone();
    // get_mut, not indexing: IndexMut would add the missing keys as null.
    if let Some(items) = b
        .get_mut("metadata")
        .and_then(|m| m.get_mut("items"))
        .and_then(Value::as_array_mut)
    {
        for it in items {
            if it["key"] == "x0ne-config" {
                it["value"] = json!("<CONFIG>");
            }
        }
    }
    let text = b
        .to_string()
        .replace(&format!("x0ne-engine-{session}"), "x0ne-engine-<SESSION>")
        .replace(&format!("\"{session}\""), "\"<SESSION>\"")
        .replace(&format!("session {session}:"), "session <SESSION>:");
    serde_json::from_str(&text).unwrap()
}

pub fn golden(name: &str) -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name);
    serde_json::from_slice(&fs::read(&p).unwrap()).unwrap()
}

// ── assertions shared by the test files ─────────────────────────────────────

/// `up` with the token on stdin and no question: the scripted path.
pub fn up_args<'a>(extra: &[&'a str]) -> Vec<&'a str> {
    let mut v = vec!["up", "--auth", "stdin", "--yes", "--max-cost", "10"];
    v.extend_from_slice(extra);
    v
}

/// The name of the VM `up` asked for.
pub fn vm_name(gcp: &FakeGcp) -> String {
    let ins = gcp.find("POST", "/zones/us-central1-a/instances");
    assert_eq!(ins.len(), 1, "one instances.insert expected");
    ins[0].json()["name"].as_str().unwrap().to_string()
}

/// DELETE was sent for the VM and its rule, and the fake holds neither.
pub fn assert_cleaned(gcp: &FakeGcp, vm: &str) {
    let d = gcp.deletes();
    assert!(
        d.iter()
            .any(|p| p.ends_with(&format!("/zones/us-central1-a/instances/{vm}"))),
        "no DELETE of the VM {vm}: {d:?}"
    );
    assert!(
        d.iter()
            .any(|p| p.ends_with(&format!("/global/firewalls/{vm}"))),
        "no DELETE of the rule {vm}: {d:?}"
    );
    gcp.with(|c| {
        assert!(!c.vms.contains_key(vm), "VM still in the fake cloud");
        assert!(
            !c.firewalls.contains_key(vm),
            "rule still in the fake cloud"
        );
    });
}

pub fn no_creation(gcp: &FakeGcp) {
    assert_eq!(
        gcp.count_containing("POST", "/instances"),
        0,
        "an instance was inserted"
    );
    assert_eq!(
        gcp.count_containing("POST", "/firewalls"),
        0,
        "a rule was inserted"
    );
}
