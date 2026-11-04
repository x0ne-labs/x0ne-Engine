//! Google Cloud in direct REST (Compute Engine v1, Cloud Storage, Cloud Billing,
//! OAuth2 tokeninfo) with a bearer token, so a pasted token is enough. Not yet
//! run against Google: the first real run settles the points marked "unverified".

use std::io::{Read, Seek, SeekFrom};
use std::net::IpAddr;
use std::time::{Duration, Instant};

use reqwest::{Method, Url};
use serde_json::{json, Value};

use super::{
    BoxFut, Guarantee, Identity, ImageRef, ImportRequest, Instance, Leftover, LeftoverKind,
    Preflight, Provider, SerialChunk, SessionRequest, Shape, VmState,
};
use crate::http::{Api, ApiError, Body};
use crate::quote::{self, PriceSource, Quote};
use crate::{Failure, Knobs, LABEL_KEY, NAME_PREFIX};

/// Regional quota of spot H100 GPUs (Compute API metric name). regions.get did not
/// list it in us-east5, europe-west4 nor us-central1 on 2026-10-17: Cloud Quotas did.
pub const H100_SPOT_QUOTA: &str = "PREEMPTIBLE_NVIDIA_H100_GPUS";
/// The same quota in Cloud Quotas v1 (`gcloud beta quotas info list`, 2026-10-17).
pub const H100_SPOT_QUOTA_ID: &str = "PREEMPTIBLE-NVIDIA-H100-GPUS-per-project-region";
pub const SPOT_CPUS_QUOTA: &str = "PREEMPTIBLE_CPUS";
pub const SPOT_LOCAL_SSD_QUOTA: &str = "PREEMPTIBLE_LOCAL_SSD_GB";
/// Local SSD attached with the machine type, GiB (accelerator-optimized machines page, 2026-10-17).
const BUNDLED_LOCAL_SSD_GB: [(&str, f64); 1] = [("a3-highgpu-1g", 750.0)];
pub const GPU_TYPE: &str = "nvidia-h100-80gb";
/// Firewall rules carry no labels: this description prefix is their mark.
pub const FIREWALL_MARK: &str = "x0ne-engine session ";
pub const IMPORT_BUCKET_PREFIX: &str = "x0ne-engine-import-";
const SCOPES_OK: [&str; 2] = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/compute",
];
const SERIAL_CAP: usize = 1 << 20;
const UPLOAD_CHUNK: usize = 32 << 20;

pub struct Gcp {
    api: Api,
    project: String,
    knobs: Knobs,
}

/// Labels on the VM and its boot disk; `sweep` keys on `x0ne-engine`.
pub fn session_labels(session_id: &str) -> Value {
    json!({ LABEL_KEY: "session", "x0ne-session": session_id })
}

/// The `instances.insert` body (tests/golden/instances_insert.json).
/// `config` is the relay's cloud config, put into metadata byte for byte.
pub fn instance_body(
    project: &str,
    zone: &str,
    name: &str,
    session_id: &str,
    machine_type: &str,
    image: &str,
    config: &str,
    nras_key: Option<&str>,
    max_run_secs: u64,
) -> Value {
    // The key has its own item: `x0ne-config` is hashed into the quote and
    // published in the attestation document.
    let mut items = vec![json!({"key": "x0ne-config", "value": config})];
    if let Some(k) = nras_key {
        items.push(json!({"key": "x0ne-nras-key", "value": k}));
    }
    items.push(json!({"key": "block-project-ssh-keys", "value": "true"}));
    json!({
        "name": name,
        "machineType": format!("zones/{zone}/machineTypes/{machine_type}"),
        "confidentialInstanceConfig": {"confidentialInstanceType": "TDX"},
        "scheduling": {
            "provisioningModel": "SPOT",
            "instanceTerminationAction": "DELETE",
            "maxRunDuration": {"seconds": max_run_secs},
            "onHostMaintenance": "TERMINATE",
            "automaticRestart": false
        },
        "guestAccelerators": [{
            "acceleratorType": format!("zones/{zone}/acceleratorTypes/{GPU_TYPE}"),
            "acceleratorCount": 1
        }],
        "disks": [{
            "boot": true,
            "autoDelete": true,
            "initializeParams": {
                "sourceImage": format!("projects/{project}/global/images/{image}"),
                "diskName": name,
                "labels": session_labels(session_id)
            }
        }],
        "networkInterfaces": [{
            "network": "global/networks/default",
            "nicType": "GVNIC",
            "accessConfigs": [{"type": "ONE_TO_ONE_NAT", "name": "External NAT"}]
        }],
        "shieldedInstanceConfig": {
            "enableSecureBoot": false,
            "enableVtpm": false,
            "enableIntegrityMonitoring": false
        },
        "metadata": {"items": items},
        "labels": session_labels(session_id),
        "tags": {"items": [name]},
        "serviceAccounts": []
    })
}

/// tcp:443 from `sources` (this machine's /32 unless `--allow-from`) to this
/// session's tag only; past the rule, access is held by mutual TLS.
pub fn firewall_body(name: &str, session_id: &str, sources: &[String]) -> Value {
    json!({
        "name": name,
        "description": format!("{FIREWALL_MARK}{session_id}: tcp:443 to this session's VM only"),
        "network": "global/networks/default",
        "direction": "INGRESS",
        "priority": 1000,
        "sourceRanges": sources,
        "targetTags": [name],
        "allowed": [{"IPProtocol": "tcp", "ports": ["443"]}]
    })
}

const IMAGE_MARK: &str = "x0ne CVM image; disk.tar.gz sha256 ";

pub fn image_body(image: &str, bucket: &str, object: &str, sha256_hex: &str) -> Value {
    json!({
        "name": image,
        "description": format!("{IMAGE_MARK}{sha256_hex}"),
        "rawDisk": {"source": format!("https://storage.googleapis.com/{bucket}/{object}")},
        "guestOsFeatures": [{"type": "TDX_CAPABLE"}, {"type": "GVNIC"}, {"type": "UEFI_COMPATIBLE"}],
        "architecture": "X86_64",
        "labels": {LABEL_KEY: "image"}
    })
}

/// The label `image_body` writes: an image without it was not made here.
pub fn image_ours(image: &Value) -> bool {
    image["labels"][LABEL_KEY].as_str() == Some("image")
}

/// A disk whose source is image `name` of `project` (`sourceImage`, a URL
/// ending in `projects/<project>/global/images/<name>`).
pub fn disk_made_from(disk: &Value, project: &str, name: &str) -> bool {
    disk["sourceImage"]
        .as_str()
        .is_some_and(|s| s.ends_with(&format!("projects/{project}/global/images/{name}")))
}

/// The sha256 `image_body` wrote, if the description is exactly that.
pub fn image_disk_sha256(image: &Value) -> Option<String> {
    image["description"]
        .as_str()?
        .strip_prefix(IMAGE_MARK)
        .filter(|h| crate::image_import::is_sha256_hex(h))
        .map(str::to_string)
}

pub fn bucket_body(bucket: &str, region: &str) -> Value {
    json!({
        "name": bucket,
        "location": region.to_ascii_uppercase(),
        "storageClass": "STANDARD",
        "labels": {LABEL_KEY: "import"},
        "iamConfiguration": {
            "uniformBucketLevelAccess": {"enabled": true},
            "publicAccessPrevention": "enforced"
        },
        // Safety net if our own deletion fails: Google drops the object after a day.
        "lifecycle": {"rule": [{"action": {"type": "Delete"}, "condition": {"age": 1}}]}
    })
}

/// Name prefix AND the label: a resource missing either is never ours.
pub fn labelled_ours(name: &str, labels: &Value) -> bool {
    name.starts_with(NAME_PREFIX) && labels.get(LABEL_KEY).is_some()
}

/// Name prefix AND our description mark AND targeting its own tag only.
pub fn firewall_ours(fw: &Value) -> bool {
    let name = fw["name"].as_str().unwrap_or("");
    name.starts_with(NAME_PREFIX)
        && fw["description"]
            .as_str()
            .is_some_and(|d| d.starts_with(FIREWALL_MARK))
        && fw["targetTags"]
            .as_array()
            .is_some_and(|t| t.len() == 1 && t[0].as_str() == Some(name))
}

/// (limit, usage) of one metric in a `regions.get` answer.
pub fn region_quota(region: &Value, metric: &str) -> Option<(f64, f64)> {
    region["quotas"]
        .as_array()?
        .iter()
        .find(|q| q["metric"].as_str() == Some(metric))
        .map(|q| {
            (
                q["limit"].as_f64().unwrap_or(0.0),
                q["usage"].as_f64().unwrap_or(0.0),
            )
        })
}

/// The limit a Cloud Quotas `QuotaInfo` enforces in `region`, and the entry it came
/// from: the region's dimension, else the one without dimensions. `details.value` is an
/// int64, a JSON string, omitted when 0 (proto3); negative is unlimited.
pub fn quota_info_limit(info: &Value, region: &str) -> (f64, &'static str) {
    let infos = info["dimensionsInfos"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let default = |d: &&Value| {
        d["dimensions"].is_null() || d["dimensions"].as_object().is_some_and(|m| m.is_empty())
    };
    let (entry, what) = match infos
        .iter()
        .find(|d| d["dimensions"]["region"].as_str() == Some(region))
    {
        Some(d) => (Some(d), "region entry"),
        None => match infos.iter().find(default) {
            Some(d) => (Some(d), "default entry"),
            None => (None, "no entry"),
        },
    };
    let value = entry.and_then(|d| match &d["details"]["value"] {
        Value::String(s) => s.parse::<i64>().ok(),
        Value::Number(n) => n.as_i64(),
        _ => None,
    });
    match value {
        Some(v) if v < 0 => (f64::INFINITY, what),
        Some(v) => (v as f64, what),
        None => (0.0, what),
    }
}

/// H100 GPUs one instance holds: its `guestAccelerators`, else its A3 machine type
/// (a3-highgpu, a3-megagpu, a3-edgegpu carry H100; a3-ultragpu is H200).
pub fn h100_gpus_of(instance: &Value) -> u64 {
    let listed: u64 = instance["guestAccelerators"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|a| {
            a["acceleratorType"]
                .as_str()
                .is_some_and(|t| last_segment(t).starts_with("nvidia-h100"))
        })
        .map(|a| a["acceleratorCount"].as_u64().unwrap_or(1))
        .sum();
    if listed > 0 {
        return listed;
    }
    let mt = instance["machineType"]
        .as_str()
        .map(last_segment)
        .unwrap_or("");
    ["a3-highgpu-", "a3-megagpu-", "a3-edgegpu-"]
        .iter()
        .find_map(|f| mt.strip_prefix(f)?.strip_suffix('g')?.parse().ok())
        .unwrap_or(0)
}

fn bundled_local_ssd_gb(machine_type: &str) -> Option<f64> {
    BUNDLED_LOCAL_SSD_GB
        .iter()
        .find(|(m, _)| *m == machine_type)
        .map(|(_, gb)| *gb)
}

/// The spot H100 quota of one region, as preflight could read it.
#[derive(Debug, Clone, PartialEq)]
pub enum H100Quota {
    /// `regions.get` listed the metric.
    Compute { limit: f64, usage: f64 },
    /// Cloud Quotas gave the limit; usage is counted from the project's VMs.
    CloudQuotas {
        limit: f64,
        entry: &'static str,
        usage: f64,
    },
    /// Cloud Quotas answered 400, 403 or 404: Google still enforces the quota at create.
    Unreadable(String),
}

fn num(v: f64) -> String {
    if v.is_infinite() {
        "unlimited".into()
    } else {
        v.to_string()
    }
}

impl H100Quota {
    /// None: not known here.
    pub fn free(&self) -> Option<f64> {
        match self {
            H100Quota::Compute { limit, usage } | H100Quota::CloudQuotas { limit, usage, .. } => {
                Some(limit - usage)
            }
            H100Quota::Unreadable(_) => None,
        }
    }

    /// `<limit> (<where>)…` for the preflight line and the refusal.
    pub fn text(&self, region: &str) -> String {
        match self {
            H100Quota::Compute { limit, usage } => format!("{} (used {usage})", num(*limit)),
            H100Quota::CloudQuotas {
                limit,
                entry,
                usage,
            } => {
                let src = format!(
                    "{} (Cloud Quotas API, {H100_SPOT_QUOTA_ID}, {entry})",
                    num(*limit)
                );
                if *limit > 0.0 {
                    format!(
                        "{src}, used {usage} (H100 GPUs of this project's VMs in {region}: \
                         Cloud Quotas gives no usage)"
                    )
                } else {
                    src
                }
            }
            H100Quota::Unreadable(why) => format!("not readable (Cloud Quotas API: {why})"),
        }
    }

    pub fn line(&self, region: &str) -> String {
        match self {
            H100Quota::Unreadable(_) => format!(
                "H100 spot quota in {region} {}; Google refuses the create if there is none, \
                 nothing is billed",
                self.text(region)
            ),
            _ => format!("H100 spot quota in {region}: {}", self.text(region)),
        }
    }
}

fn last_segment(url: &str) -> &str {
    url.rsplit('/').next().unwrap_or(url)
}

fn api_fail(what: &str, e: ApiError) -> Failure {
    Failure::Api(format!("{what}: {e}"))
}

impl Gcp {
    pub fn new(api: Api, project: String, knobs: Knobs) -> Self {
        Gcp {
            api,
            project,
            knobs,
        }
    }

    fn compute(&self, path: &str) -> Url {
        self.api
            .ep
            .compute
            .join(&format!("compute/v1/projects/{}/{path}", self.project))
            .expect("validated names")
    }

    fn storage(&self, path: &str) -> Url {
        self.api
            .ep
            .storage
            .join(&format!("storage/v1/{path}"))
            .expect("validated names")
    }

    fn instance_url(&self, i: &Instance) -> Url {
        self.compute(&format!("zones/{}/instances/{}", i.zone, i.name))
    }

    /// Polls a zone or global operation until DONE; its error, if any, is returned.
    async fn wait_op(&self, op: &Value, timeout: Duration) -> Result<(), ApiError> {
        let Some(name) = op["name"].as_str() else {
            return Ok(());
        };
        if name.len() > 128
            || !name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        {
            return Err(ApiError {
                status: None,
                reason: "operation".into(),
                message: "operation name not understood".into(),
            });
        }
        let url = match op["zone"].as_str() {
            Some(z) => self.compute(&format!("zones/{}/operations/{name}", last_segment(z))),
            None => match op["region"].as_str() {
                Some(r) => self.compute(&format!("regions/{}/operations/{name}", last_segment(r))),
                None => self.compute(&format!("global/operations/{name}")),
            },
        };
        let deadline = Instant::now() + timeout;
        let mut cur = op.clone();
        loop {
            if cur["status"].as_str() == Some("DONE") {
                if let Some(errs) = cur["error"]["errors"].as_array() {
                    if let Some(e) = errs.first() {
                        return Err(ApiError {
                            status: cur["httpErrorStatusCode"].as_u64().map(|c| c as u16),
                            reason: crate::sanitize(e["code"].as_str().unwrap_or(""), 80),
                            message: crate::sanitize(
                                e["message"].as_str().unwrap_or("operation failed"),
                                400,
                            ),
                        });
                    }
                }
                return Ok(());
            }
            if Instant::now() > deadline {
                return Err(ApiError {
                    status: None,
                    reason: "timeout".into(),
                    message: format!("operation {name} not done after {} s", timeout.as_secs()),
                });
            }
            tokio::time::sleep(self.knobs.boot_poll.min(Duration::from_secs(2))).await;
            cur = self.api.get_json(url.clone()).await?;
        }
    }

    fn quota_info_url(&self, id: &str) -> Url {
        self.api
            .ep
            .quotas
            .join(&format!(
                "v1/projects/{}/locations/global/services/compute.googleapis.com/quotaInfos/{id}",
                self.project
            ))
            .expect("validated names")
    }

    /// The H100 spot quota of `region`: from its `regions.get` answer, else from Cloud
    /// Quotas (`info` caches that one read, the same for every region).
    async fn h100_quota(
        &self,
        region: &str,
        regional: &Value,
        info: &mut Option<Result<Value, ApiError>>,
    ) -> Result<H100Quota, Failure> {
        if let Some((limit, usage)) = region_quota(regional, H100_SPOT_QUOTA) {
            return Ok(H100Quota::Compute { limit, usage });
        }
        if info.is_none() {
            *info = Some(
                self.api
                    .get_json(self.quota_info_url(H100_SPOT_QUOTA_ID))
                    .await,
            );
        }
        let v = match info.as_ref().expect("just read") {
            Ok(v) => v,
            // The v1 docs name the project by number; this path carries the id (400 then).
            Err(e) if matches!(e.status, Some(400 | 403 | 404)) => {
                return Ok(H100Quota::Unreadable(e.to_string()))
            }
            Err(e) => {
                return Err(api_fail(
                    &format!("reading {H100_SPOT_QUOTA_ID} in Cloud Quotas"),
                    e.clone(),
                ))
            }
        };
        let (limit, entry) = quota_info_limit(v, region);
        let usage = if limit > 0.0 {
            self.aggregated("instances", "instances")
                .await
                .map_err(|e| api_fail("listing the project's VMs", e))?
                .iter()
                .filter(|(zone, _)| crate::region_of(zone) == region)
                .map(|(_, i)| h100_gpus_of(i))
                .sum::<u64>() as f64
        } else {
            0.0
        };
        Ok(H100Quota::CloudQuotas {
            limit,
            entry,
            usage,
        })
    }

    /// Every page of an aggregated list, flattened with its zone or region.
    async fn aggregated(&self, what: &str, key: &str) -> Result<Vec<(String, Value)>, ApiError> {
        let mut out = Vec::new();
        let mut token: Option<String> = None;
        for _ in 0..20 {
            let mut url = self.compute(&format!("aggregated/{what}"));
            {
                let mut q = url.query_pairs_mut();
                q.append_pair("maxResults", "500");
                q.append_pair("returnPartialSuccess", "true");
                if let Some(t) = &token {
                    q.append_pair("pageToken", t);
                }
            }
            let v = self.api.get_json(url).await?;
            if let Some(items) = v["items"].as_object() {
                for (scope, body) in items {
                    let loc = last_segment(scope).to_string();
                    for r in body[key].as_array().into_iter().flatten() {
                        out.push((loc.clone(), r.clone()));
                    }
                }
            }
            match v["nextPageToken"].as_str() {
                Some(t) if !t.is_empty() => token = Some(t.to_string()),
                _ => break,
            }
        }
        Ok(out)
    }

    async fn list_pages(&self, mut url: Url, key: &str) -> Result<Vec<Value>, ApiError> {
        let mut out = Vec::new();
        let base = url.clone();
        for _ in 0..20 {
            let v = self.api.get_json(url.clone()).await?;
            out.extend(v[key].as_array().into_iter().flatten().cloned());
            match v["nextPageToken"].as_str() {
                Some(t) if !t.is_empty() => {
                    url = base.clone();
                    url.query_pairs_mut().append_pair("pageToken", t);
                }
                _ => break,
            }
        }
        Ok(out)
    }

    async fn catalog_breakdown(&self, region: &str) -> Option<quote::Breakdown> {
        let mut skus = Vec::new();
        let mut token: Option<String> = None;
        for _ in 0..15 {
            let mut url = self
                .api
                .ep
                .billing
                .join(&format!("v1/services/{}/skus", quote::COMPUTE_SERVICE_ID))
                .ok()?;
            {
                let mut q = url.query_pairs_mut();
                q.append_pair("currencyCode", "USD");
                q.append_pair("pageSize", "5000");
                if let Some(t) = &token {
                    q.append_pair("pageToken", t);
                }
            }
            let v = self.api.get_json(url).await.ok()?;
            skus.extend(
                v["skus"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|s| quote::is_candidate(s))
                    .cloned(),
            );
            match v["nextPageToken"].as_str() {
                Some(t) if !t.is_empty() => token = Some(t.to_string()),
                _ => break,
            }
        }
        quote::breakdown_from_skus(&skus, region)
    }

    async fn delete_url(&self, url: Url, what: &str) -> Result<(), String> {
        match self.api.delete(url).await {
            Ok(op) => self
                .wait_op(&op, self.knobs.op_timeout)
                .await
                .map_err(|e| format!("{what}: {e}")),
            Err(e) if e.not_found() => Ok(()),
            Err(e) => Err(format!("{what}: {e}")),
        }
    }

    /// DELETE, then confirm absence. When the insert may still be landing,
    /// absence is re-checked after a pause and a reappearing VM is deleted again.
    async fn delete_instance(&self, i: &Instance, pending: bool) -> Result<(), String> {
        let url = self.instance_url(i);
        let needed = if pending { 3 } else { 1 };
        let deadline = Instant::now() + self.knobs.op_timeout * 2;
        loop {
            self.delete_url(url.clone(), &format!("VM {}", i.name))
                .await?;
            let mut absent = 0;
            loop {
                if Instant::now() > deadline {
                    return Err(format!("VM {}: still present after the delete", i.name));
                }
                match self.api.get_json(url.clone()).await {
                    Err(e) if e.not_found() => {
                        absent += 1;
                        if absent >= needed {
                            return Ok(());
                        }
                        tokio::time::sleep(self.knobs.settle).await;
                    }
                    Err(e) => return Err(format!("VM {}: {e}", i.name)),
                    Ok(_) => break,
                }
            }
        }
    }

    async fn delete_bucket(&self, bucket: &str) -> Result<(), String> {
        let mut list = self.storage(&format!("b/{bucket}/o"));
        list.query_pairs_mut().append_pair("versions", "true");
        let objects = match self.list_pages(list, "items").await {
            Ok(o) => o,
            Err(e) if e.not_found() => return Ok(()),
            Err(e) => return Err(format!("bucket {bucket}: {e}")),
        };
        for o in objects {
            let Some(name) = o["name"].as_str() else {
                continue;
            };
            let mut url = self.storage(&format!("b/{bucket}/o"));
            url.path_segments_mut()
                .map_err(|_| "bad storage URL".to_string())?
                .push(name);
            if let Some(g) = o["generation"].as_str() {
                url.query_pairs_mut().append_pair("generation", g);
            }
            match self.api.delete(url).await {
                Ok(_) => {}
                Err(e) if e.not_found() => {}
                Err(e) => return Err(format!("object {name} in {bucket}: {e}")),
            }
        }
        match self.api.delete(self.storage(&format!("b/{bucket}"))).await {
            Ok(_) => Ok(()),
            Err(e) if e.not_found() => Ok(()),
            Err(e) => Err(format!("bucket {bucket}: {e}")),
        }
    }

    async fn upload(&self, bucket: &str, object: &str, r: &ImportRequest) -> Result<(), String> {
        let mut start = self
            .api
            .ep
            .storage
            .join(&format!("upload/storage/v1/b/{bucket}/o"))
            .map_err(|e| e.to_string())?;
        start
            .query_pairs_mut()
            .append_pair("uploadType", "resumable")
            .append_pair("name", object);
        let resp = self
            .api
            .send(
                Method::POST,
                start,
                Body::Json(json!({"name": object, "contentType": "application/gzip"})),
                &[
                    ("x-upload-content-type", "application/gzip".into()),
                    ("x-upload-content-length", r.size.to_string()),
                ],
                true,
            )
            .await
            .map_err(|e| format!("starting the upload: {e}"))?;
        let session = resp
            .headers
            .get("location")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| Url::parse(s).ok())
            .filter(|u| self.api.ep.allows(u))
            .ok_or("the upload session URL is missing or not on the storage host")?;
        let mut file =
            std::fs::File::open(&r.file).map_err(|e| format!("{}: {e}", r.file.display()))?;
        let mut offset: u64 = 0;
        let mut buf = vec![0u8; UPLOAD_CHUNK];
        let mut rounds = 0;
        while offset < r.size {
            rounds += 1;
            if rounds > 100_000 {
                return Err("upload does not progress".into());
            }
            let n = (r.size - offset).min(UPLOAD_CHUNK as u64) as usize;
            file.seek(SeekFrom::Start(offset))
                .map_err(|e| e.to_string())?;
            file.read_exact(&mut buf[..n]).map_err(|e| e.to_string())?;
            let range = format!("bytes {}-{}/{}", offset, offset + n as u64 - 1, r.size);
            // The session URL authorises the upload; the token is not resent.
            let resp = self
                .api
                .send(
                    Method::PUT,
                    session.clone(),
                    Body::Bytes(buf[..n].to_vec(), "application/octet-stream"),
                    &[("content-range", range)],
                    false,
                )
                .await
                .map_err(|e| format!("uploading: {e}"))?;
            if resp.status.as_u16() == 308 {
                offset = resp
                    .headers
                    .get("range")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.rsplit('-').next())
                    .and_then(|e| e.parse::<u64>().ok())
                    .map_or(0, |e| e + 1);
            } else if offset + n as u64 == r.size {
                return Ok(());
            } else {
                return Err(format!("upload ended early (HTTP {})", resp.status));
            }
        }
        Ok(())
    }
}

impl Provider for Gcp {
    fn name(&self) -> &'static str {
        "gcp"
    }
    fn display_name(&self) -> &'static str {
        "Google Cloud"
    }
    fn guarantee(&self) -> Guarantee {
        Guarantee::TdxQuoteDirect
    }

    fn identity(&self) -> BoxFut<'_, Identity> {
        Box::pin(async move {
            let tok = self.api.raw_token()?;
            let url = self.api.ep.oauth2.join("tokeninfo").expect("constant path");
            // In the body, not the URL: URLs end up in logs.
            let resp = self
                .api
                .send(
                    Method::POST,
                    url,
                    Body::Form(vec![("access_token".into(), tok)]),
                    &[],
                    false,
                )
                .await
                .map_err(|e| {
                    Failure::Preflight(match e.status {
                        Some(_) => format!("Google refused the token ({e})"),
                        // No HTTP status: the request never got an answer.
                        None => format!(
                            "no answer from Google's tokeninfo ({e}): a network failure, \
                             not a verdict on the token"
                        ),
                    })
                })?;
            let v = resp
                .json()
                .map_err(|e| Failure::Preflight(format!("tokeninfo: {e}")))?;
            let scopes = v["scope"].as_str().unwrap_or("");
            if !scopes.split(' ').any(|s| SCOPES_OK.contains(&s)) {
                return Err(Failure::Preflight(
                    "the token has neither the cloud-platform nor the compute scope \
                     (gcloud auth print-access-token gives one that has)"
                        .into(),
                ));
            }
            let expires_in_secs = match &v["expires_in"] {
                Value::String(s) => s.parse().ok(),
                Value::Number(n) => n.as_u64(),
                _ => None,
            };
            Ok(Identity {
                account: v["email"].as_str().map(|s| crate::sanitize(s, 120)),
                expires_in_secs,
            })
        })
    }

    fn preflight<'a>(&'a self, r: &'a SessionRequest) -> BoxFut<'a, Preflight> {
        Box::pin(async move {
            let candidates: Vec<String> = match &r.zone {
                Some(z) => vec![z.clone()],
                None => crate::TDX_H100_ZONES
                    .iter()
                    .map(|z| z.to_string())
                    .collect(),
            };
            let mut chosen = None;
            let mut seen = Vec::new();
            let mut short_of_other = false;
            let mut info = None;
            for z in &candidates {
                let region = crate::region_of(z);
                let regional = self
                    .api
                    .get_json(self.compute(&format!("regions/{region}")))
                    .await
                    .map_err(|e| api_fail(&format!("reading the quotas of {region}"), e))?;
                let h100 = self.h100_quota(region, &regional, &mut info).await?;
                if h100.free().is_some_and(|f| f < 1.0) {
                    seen.push(format!("{region}: {}", h100.text(region)));
                    continue;
                }
                let mt = self
                    .api
                    .get_json(self.compute(&format!("zones/{z}/machineTypes/{}", r.machine_type)))
                    .await
                    .map_err(|e| {
                        if e.not_found() {
                            Failure::Preflight(format!("{} is not offered in {z}", r.machine_type))
                        } else {
                            api_fail("reading the machine type", e)
                        }
                    })?;
                let shape = Shape {
                    vcpus: mt["guestCpus"].as_u64().unwrap_or(0) as u32,
                    memory_gib: mt["memoryMb"].as_f64().unwrap_or(0.0) / 1024.0,
                    gpus: mt["accelerators"][0]["guestAcceleratorCount"]
                        .as_u64()
                        .unwrap_or(1) as u32,
                };
                let needs = [
                    (SPOT_CPUS_QUOTA, Some(f64::from(shape.vcpus))),
                    (SPOT_LOCAL_SSD_QUOTA, bundled_local_ssd_gb(&r.machine_type)),
                ];
                let short: Vec<String> = needs
                    .iter()
                    .filter_map(|(metric, need)| {
                        let need = (*need)?;
                        let (limit, usage) = region_quota(&regional, metric)?;
                        (limit - usage < need).then(|| {
                            format!(
                                "{metric} has {} free (limit {}, used {usage}), {} needs {need}",
                                num(limit - usage),
                                num(limit),
                                r.machine_type
                            )
                        })
                    })
                    .collect();
                if short.is_empty() {
                    chosen = Some((z.clone(), h100, shape));
                    break;
                }
                short_of_other = true;
                seen.push(format!("{region}: {}", short.join("; ")));
            }
            let Some((zone, h100, shape)) = chosen else {
                return Err(Failure::Preflight(if short_of_other {
                    format!(
                        "no zone has the spot quota {} needs: {}. Ask Google for it (IAM → Quotas)",
                        r.machine_type,
                        seen.join(", ")
                    )
                } else {
                    format!(
                        "no free H100 spot quota ({H100_SPOT_QUOTA}): {}. Ask Google for one \
                         (IAM → Quotas)",
                        seen.join(", ")
                    )
                }));
            };
            let quota = h100.line(crate::region_of(&zone));
            match self
                .api
                .get_json(self.compute("global/networks/default"))
                .await
            {
                Ok(_) => {}
                Err(e) if e.not_found() => {
                    return Err(Failure::Preflight(format!(
                        "project {} has no \"default\" VPC network",
                        self.project
                    )))
                }
                Err(e) => return Err(api_fail("reading the default network", e)),
            }
            let image = self.image(&r.image).await?;
            let (live, leftovers) = split_live(self.sweep().await?);
            Ok(Preflight {
                zone,
                quota,
                shape,
                image,
                live,
                leftovers,
            })
        })
    }

    fn quote<'a>(&'a self, r: &'a SessionRequest, p: &'a Preflight) -> BoxFut<'a, Quote> {
        Box::pin(async move {
            let region = crate::region_of(&p.zone).to_string();
            let read_at = crate::now_unix();
            let cc = quote::cc_surcharge(&r.machine_type).ok_or_else(|| {
                Failure::QuoteRefused(format!(
                    "no Confidential Computing price for {}: a quote without it is below the bill",
                    r.machine_type
                ))
            })?;
            let (vm, source, breakdown) = match self.catalog_breakdown(&region).await {
                Some(b) if p.shape.vcpus > 0 && p.shape.memory_gib > 0.0 => (
                    quote::vm_price(&b, &p.shape),
                    PriceSource::Catalog {
                        read_at_unix: read_at,
                    },
                    Some(b),
                ),
                _ => {
                    let p = quote::table_price(&r.machine_type, &region).ok_or_else(|| {
                        Failure::QuoteRefused(format!(
                            "no price for {} in {region}: the Cloud Billing catalog could not \
                             be read and the embedded table has no entry",
                            r.machine_type
                        ))
                    })?;
                    let source = PriceSource::Table {
                        date: quote::TABLE_DATE,
                        origin: quote::TABLE_ORIGIN,
                    };
                    (p, source, None)
                }
            };
            Ok(Quote {
                machine_type: r.machine_type.clone(),
                region,
                vm_usd_per_hour: vm,
                cc_usd_per_hour: cc,
                extras_usd_per_hour: quote::EXTRAS_USD_PER_HOUR,
                max_run_secs: r.max_run_secs,
                source,
                breakdown,
            })
        })
    }

    fn create<'a>(
        &'a self,
        r: &'a SessionRequest,
        i: &'a Instance,
        config: &'a [u8],
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            let config = std::str::from_utf8(config)
                .map_err(|_| Failure::RelayFailed("the relay's config is not UTF-8".into()))?;
            if r.source_ranges.is_empty() {
                return Err(Failure::Preflight(
                    "no source for the firewall rule; nothing was created".into(),
                ));
            }
            let op = self
                .api
                .post_json(
                    self.compute("global/firewalls"),
                    firewall_body(&i.firewall, &r.session_id, &r.source_ranges),
                )
                .await
                .map_err(|e| api_fail("creating the firewall rule", e))?;
            self.wait_op(&op, self.knobs.op_timeout)
                .await
                .map_err(|e| api_fail("creating the firewall rule", e))?;
            let body = instance_body(
                &i.project,
                &i.zone,
                &i.name,
                &r.session_id,
                &r.machine_type,
                &r.image,
                config,
                r.nras_key.as_ref().map(crate::nras_key::NrasKey::expose),
                r.max_run_secs,
            );
            // Google quotes a value it refuses: the key never comes back out.
            let hide = |f: Failure| match &r.nras_key {
                Some(k) => f.map_message(|m| k.redact(m)),
                None => f,
            };
            let op = self
                .api
                .post_json(self.compute(&format!("zones/{}/instances", i.zone)), body)
                .await
                .map_err(|e| hide(api_fail("creating the VM", e)))?;
            self.wait_op(&op, self.knobs.op_timeout)
                .await
                .map_err(|e| {
                    if e.reason.contains("RESOURCE_POOL_EXHAUSTED") || e.reason.contains("STOCKOUT")
                    {
                        Failure::Api(format!(
                            "no spot H100 capacity in {} right now ({e}); nothing is billed, try later",
                            i.zone
                        ))
                    } else {
                        api_fail("creating the VM", e)
                    }
                })
                .map_err(hide)
        })
    }

    fn instance_state<'a>(&'a self, i: &'a Instance) -> BoxFut<'a, VmState> {
        Box::pin(async move {
            let v = match self.api.get_json(self.instance_url(i)).await {
                Ok(v) => v,
                Err(e) if e.not_found() => return Ok(VmState::Gone),
                Err(e) => return Err(api_fail("reading the VM", e)),
            };
            let status = v["status"].as_str().unwrap_or("UNKNOWN").to_string();
            Ok(match status.as_str() {
                "RUNNING" => {
                    let ip = v["networkInterfaces"][0]["accessConfigs"][0]["natIP"]
                        .as_str()
                        .and_then(|s| s.parse::<IpAddr>().ok());
                    match ip {
                        Some(ip) => VmState::Running(ip),
                        None => VmState::Pending("RUNNING, no external IP yet".into()),
                    }
                }
                "PROVISIONING" | "STAGING" => VmState::Pending(status),
                _ => VmState::Stopped(crate::sanitize(&status, 40)),
            })
        })
    }

    fn serial_output<'a>(&'a self, i: &'a Instance) -> BoxFut<'a, String> {
        Box::pin(async move {
            let mut out = String::new();
            let mut start = "0".to_string();
            for _ in 0..16 {
                let mut url =
                    self.compute(&format!("zones/{}/instances/{}/serialPort", i.zone, i.name));
                url.query_pairs_mut()
                    .append_pair("port", "1")
                    .append_pair("start", &start);
                let v = self
                    .api
                    .get_json(url)
                    .await
                    .map_err(|e| api_fail("reading the serial console", e))?;
                let chunk = v["contents"].as_str().unwrap_or("");
                if chunk.is_empty() {
                    break;
                }
                let room = SERIAL_CAP.saturating_sub(out.len());
                out.extend(chunk.chars().take(room));
                if out.len() >= SERIAL_CAP {
                    break;
                }
                let next = match &v["next"] {
                    Value::String(s) => s.clone(),
                    Value::Number(n) => n.to_string(),
                    _ => break,
                };
                if next.is_empty() || next == start || !next.bytes().all(|c| c.is_ascii_digit()) {
                    break;
                }
                start = next;
            }
            Ok(out)
        })
    }

    fn serial_since<'a>(&'a self, i: &'a Instance, start: u64) -> BoxFut<'a, SerialChunk> {
        Box::pin(async move {
            let mut url =
                self.compute(&format!("zones/{}/instances/{}/serialPort", i.zone, i.name));
            url.query_pairs_mut()
                .append_pair("port", "1")
                .append_pair("start", &start.to_string());
            let v = self
                .api
                .get_json(url)
                .await
                .map_err(|e| api_fail("reading the serial console", e))?;
            // int64 fields come as strings; `start` moves on when older output was dropped.
            let position = |key: &str| match &v[key] {
                Value::String(s) => s.parse::<u64>().ok(),
                Value::Number(n) => n.as_u64(),
                _ => None,
            };
            let next = position("next").ok_or_else(|| {
                Failure::Api("reading the serial console: the answer has no `next` position".into())
            })?;
            Ok(SerialChunk {
                start: position("start").unwrap_or(start),
                next,
                contents: v["contents"].as_str().unwrap_or("").to_string(),
            })
        })
    }

    fn delete<'a>(&'a self, i: &'a Instance, pending: bool) -> BoxFut<'a, ()> {
        Box::pin(async move {
            let vm = self.delete_instance(i, pending).await;
            let fw = self
                .delete_url(
                    self.compute(&format!("global/firewalls/{}", i.firewall)),
                    &format!("firewall rule {}", i.firewall),
                )
                .await;
            match (vm, fw) {
                (Ok(()), Ok(())) => Ok(()),
                (a, b) => Err(Failure::DeleteUnconfirmed(
                    [a.err(), b.err()]
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>()
                        .join("; "),
                )),
            }
        })
    }

    fn sweep(&self) -> BoxFut<'_, Vec<Leftover>> {
        Box::pin(async move {
            let mut out = Vec::new();
            let fail = |what: &str| {
                let what = what.to_string();
                move |e: ApiError| api_fail(&format!("listing {what}"), e)
            };
            for (zone, v) in self
                .aggregated("instances", "instances")
                .await
                .map_err(fail("VMs"))?
            {
                let name = v["name"].as_str().unwrap_or("");
                if labelled_ours(name, &v["labels"]) {
                    out.push(Leftover {
                        kind: LeftoverKind::Instance,
                        name: name.into(),
                        location: zone,
                        detail: crate::sanitize(v["status"].as_str().unwrap_or(""), 40),
                    });
                }
            }
            for (zone, v) in self
                .aggregated("disks", "disks")
                .await
                .map_err(fail("disks"))?
            {
                let name = v["name"].as_str().unwrap_or("");
                if labelled_ours(name, &v["labels"]) {
                    out.push(Leftover {
                        kind: LeftoverKind::Disk,
                        name: name.into(),
                        location: zone,
                        detail: format!("{} GB", v["sizeGb"].as_str().unwrap_or("?")),
                    });
                }
            }
            let mut addresses = self
                .aggregated("addresses", "addresses")
                .await
                .map_err(fail("IP addresses"))?;
            addresses.extend(
                self.list_pages(self.compute("global/addresses"), "items")
                    .await
                    .map_err(fail("global IP addresses"))?
                    .into_iter()
                    .map(|v| ("global".to_string(), v)),
            );
            for (loc, v) in addresses {
                let name = v["name"].as_str().unwrap_or("");
                if labelled_ours(name, &v["labels"]) {
                    out.push(Leftover {
                        kind: LeftoverKind::Address,
                        name: name.into(),
                        location: loc,
                        detail: crate::sanitize(v["status"].as_str().unwrap_or(""), 40),
                    });
                }
            }
            for v in self
                .list_pages(self.compute("global/firewalls"), "items")
                .await
                .map_err(fail("firewall rules"))?
            {
                if firewall_ours(&v) {
                    out.push(Leftover {
                        kind: LeftoverKind::Firewall,
                        name: v["name"].as_str().unwrap_or("").into(),
                        location: "global".into(),
                        detail: "tcp:443".into(),
                    });
                }
            }
            let mut b = self.storage("b");
            b.query_pairs_mut()
                .append_pair("project", &self.project)
                .append_pair("prefix", NAME_PREFIX);
            for v in self.list_pages(b, "items").await.map_err(fail("buckets"))? {
                let name = v["name"].as_str().unwrap_or("");
                if labelled_ours(name, &v["labels"]) {
                    out.push(Leftover {
                        kind: LeftoverKind::Bucket,
                        name: name.into(),
                        location: crate::sanitize(v["location"].as_str().unwrap_or(""), 40),
                        detail: "import transit".into(),
                    });
                }
            }
            Ok(out)
        })
    }

    fn delete_leftover<'a>(&'a self, l: &'a Leftover) -> BoxFut<'a, ()> {
        Box::pin(async move {
            let ok_name = crate::valid_resource_name(&l.name)
                || (l.kind == LeftoverKind::Bucket
                    && l.name.len() <= 63
                    && l.name
                        .bytes()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-'));
            let ok_loc = l.location == "global"
                || l.kind == LeftoverKind::Bucket
                || l.location
                    .bytes()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-');
            if !ok_name || !ok_loc || !l.name.starts_with(NAME_PREFIX) {
                return Err(Failure::Usage(format!("refusing to delete {:?}", l.name)));
            }
            let res = match l.kind {
                LeftoverKind::Instance => {
                    self.delete_url(
                        self.compute(&format!("zones/{}/instances/{}", l.location, l.name)),
                        &format!("VM {}", l.name),
                    )
                    .await
                }
                LeftoverKind::Disk => {
                    self.delete_url(
                        self.compute(&format!("zones/{}/disks/{}", l.location, l.name)),
                        &format!("disk {}", l.name),
                    )
                    .await
                }
                LeftoverKind::Address if l.location == "global" => {
                    self.delete_url(
                        self.compute(&format!("global/addresses/{}", l.name)),
                        &format!("IP address {}", l.name),
                    )
                    .await
                }
                LeftoverKind::Address => {
                    self.delete_url(
                        self.compute(&format!("regions/{}/addresses/{}", l.location, l.name)),
                        &format!("IP address {}", l.name),
                    )
                    .await
                }
                LeftoverKind::Firewall => {
                    self.delete_url(
                        self.compute(&format!("global/firewalls/{}", l.name)),
                        &format!("firewall rule {}", l.name),
                    )
                    .await
                }
                LeftoverKind::Bucket => self.delete_bucket(&l.name).await,
            };
            res.map_err(Failure::DeleteUnconfirmed)
        })
    }

    fn image(&self, name: &str) -> BoxFut<'_, ImageRef> {
        let name = name.to_string();
        Box::pin(async move {
            match self
                .api
                .get_json(self.compute(&format!("global/images/{name}")))
                .await
            {
                Ok(v) => Ok(ImageRef {
                    project: self.project.clone(),
                    name,
                    status: crate::sanitize(v["status"].as_str().unwrap_or("UNKNOWN"), 40),
                    disk_sha256: image_disk_sha256(&v),
                    ours: image_ours(&v),
                }),
                Err(e) if e.not_found() => Ok(ImageRef {
                    project: self.project.clone(),
                    name,
                    status: "MISSING".into(),
                    disk_sha256: None,
                    ours: false,
                }),
                Err(e) => Err(api_fail("reading the image", e)),
            }
        })
    }

    fn import_image<'a>(&'a self, r: &'a ImportRequest) -> BoxFut<'a, ImageRef> {
        Box::pin(async move {
            let bucket = r.bucket.clone();
            let object = format!("{}.tar.gz", r.image);
            self.api
                .post_json(
                    {
                        let mut u = self.storage("b");
                        u.query_pairs_mut().append_pair("project", &self.project);
                        u
                    },
                    bucket_body(&bucket, &r.region),
                )
                .await
                .map_err(|e| api_fail("creating the transit bucket", e))?;
            let result = async {
                sayln!(
                    "  uploading {} to transit bucket {bucket} ...",
                    r.file.display()
                );
                self.upload(&bucket, &object, r)
                    .await
                    .map_err(Failure::Api)?;
                sayln!(
                    "  creating image {} (Google reads the upload; minutes) ...",
                    r.image
                );
                let op = self
                    .api
                    .post_json(
                        self.compute("global/images"),
                        image_body(&r.image, &bucket, &object, &r.sha256_hex),
                    )
                    .await
                    .map_err(|e| api_fail("creating the image", e))?;
                self.wait_op(&op, Duration::from_mins(45).min(self.knobs.op_timeout * 9))
                    .await
                    .map_err(|e| api_fail("creating the image", e))?;
                self.image(&r.image).await
            }
            .await;
            let cleanup = self.delete_bucket(&bucket).await;
            match (result, cleanup) {
                (Ok(img), Ok(())) => Ok(img),
                (Err(e), Ok(())) => Err(e),
                (Ok(_) | Err(_), Err(c)) => Err(Failure::DeleteUnconfirmed(format!(
                    "transit bucket not deleted ({c}): gcloud storage rm -r gs://{bucket} --project {}",
                    self.project
                ))),
            }
        })
    }

    fn image_users<'a>(&'a self, name: &'a str) -> BoxFut<'a, Vec<String>> {
        Box::pin(async move {
            let disks = self
                .aggregated("disks", "disks")
                .await
                .map_err(|e| api_fail("listing disks", e))?;
            let mut vms: Vec<String> = disks
                .iter()
                .filter(|(_, d)| disk_made_from(d, &self.project, name))
                .flat_map(|(_, d)| d["users"].as_array().cloned().unwrap_or_default())
                .filter_map(|u| u.as_str().map(|u| crate::sanitize(last_segment(u), 63)))
                .collect();
            vms.sort();
            vms.dedup();
            Ok(vms)
        })
    }

    fn delete_image<'a>(&'a self, name: &'a str) -> BoxFut<'a, ()> {
        Box::pin(async move {
            let fail = |why: String| {
                Failure::DeleteUnconfirmed(format!(
                    "{why}: gcloud compute images delete {name} --project {}",
                    self.project
                ))
            };
            if !crate::valid_resource_name(name) {
                return Err(Failure::Usage(format!("refusing to delete {name:?}")));
            }
            self.delete_url(
                self.compute(&format!("global/images/{name}")),
                &format!("image {name}"),
            )
            .await
            .map_err(fail)?;
            match self.image(name).await {
                Ok(img) if img.status == "MISSING" => Ok(()),
                Ok(img) => Err(fail(format!(
                    "image {name} still present ({}) after the delete",
                    img.status
                ))),
                Err(e) => Err(fail(e.message().to_string())),
            }
        })
    }

    fn manual_delete_commands(&self, i: &Instance) -> Vec<String> {
        vec![
            format!(
                "gcloud compute instances delete {} --zone {} --project {} --quiet",
                i.name, i.zone, i.project
            ),
            format!(
                "gcloud compute firewall-rules delete {} --project {} --quiet",
                i.firewall, i.project
            ),
        ]
    }
}

/// Bucket names are global to all of Cloud Storage: random, never derived.
pub fn import_bucket_name(random_hex: &str) -> String {
    format!("{IMPORT_BUCKET_PREFIX}{random_hex}")
}

/// VMs that still run (or are about to) block a second session.
pub fn split_live(all: Vec<Leftover>) -> (Vec<Leftover>, Vec<Leftover>) {
    all.into_iter().partition(|l| {
        l.kind == LeftoverKind::Instance
            && matches!(
                l.detail.as_str(),
                "PROVISIONING" | "STAGING" | "RUNNING" | "REPAIRING" | "STOPPING" | "SUSPENDING"
            )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_prefixed_and_labelled_resources_are_ours() {
        let l = json!({"x0ne-engine": "session"});
        assert!(labelled_ours("x0ne-engine-abc123", &l));
        assert!(!labelled_ours("x0ne-engine-abc123", &json!({})));
        assert!(!labelled_ours("x0ne-engine-abc123", &Value::Null));
        assert!(!labelled_ours("other-vm", &l));
        assert!(!labelled_ours("my-x0ne-engine-vm", &l));
    }

    #[test]
    fn a_firewall_rule_needs_prefix_mark_and_its_own_tag() {
        let ok = firewall_body("x0ne-engine-abc123", "abc123", &["192.0.2.0/24".into()]);
        assert_eq!(ok["sourceRanges"], json!(["192.0.2.0/24"]));
        assert!(firewall_ours(&ok));
        let mut other_tag = ok.clone();
        other_tag["targetTags"] = json!(["web"]);
        assert!(!firewall_ours(&other_tag));
        let mut no_mark = ok.clone();
        no_mark["description"] = json!("someone else's");
        assert!(!firewall_ours(&no_mark));
        let mut renamed = ok.clone();
        renamed["name"] = json!("allow-443");
        assert!(!firewall_ours(&renamed));
    }

    #[test]
    fn the_image_disk_sha256_is_read_back_only_from_our_description() {
        let sha = "ab".repeat(32);
        let ours = image_body("x0ne-cvm-0-2", "b", "o", &sha);
        assert_eq!(image_disk_sha256(&ours), Some(sha.clone()));
        for d in [
            json!("x0ne CVM image"),
            json!(format!(
                "x0ne CVM image; disk.tar.gz sha256 {}",
                "AB".repeat(32)
            )),
            json!(format!("x0ne CVM image; disk.tar.gz sha256 {sha} ")),
            json!(format!("x0ne CVM image; disk.tar.gz sha256 {}", &sha[2..])),
            Value::Null,
        ] {
            assert_eq!(image_disk_sha256(&json!({"description": d})), None, "{d}");
        }
    }

    #[test]
    fn running_vms_are_live_the_rest_are_leftovers() {
        let vm = |d: &str| Leftover {
            kind: LeftoverKind::Instance,
            name: "x0ne-engine-a".into(),
            location: "us-central1-a".into(),
            detail: d.into(),
        };
        let (live, rest) = split_live(vec![vm("RUNNING"), vm("TERMINATED"), vm("STAGING")]);
        assert_eq!(live.len(), 2);
        assert_eq!(rest.len(), 1);
    }

    #[test]
    fn the_cloud_quotas_limit_is_the_regions_entry_else_the_default_one() {
        let info = json!({"dimensionsInfos": [
            {"dimensions": {"region": "us-central1"}, "details": {"value": "1"}},
            {"dimensions": {"region": "us-east5"}, "details": {"value": "-1"}},
            {"details": {}}
        ]});
        assert_eq!(
            quota_info_limit(&info, "us-central1"),
            (1.0, "region entry")
        );
        assert_eq!(
            quota_info_limit(&info, "us-east5"),
            (f64::INFINITY, "region entry")
        );
        assert_eq!(
            quota_info_limit(&info, "europe-west4"),
            (0.0, "default entry")
        );
        let numeric = json!({"dimensionsInfos": [{"dimensions": {}, "details": {"value": 4}}]});
        assert_eq!(
            quota_info_limit(&numeric, "europe-west4"),
            (4.0, "default entry")
        );
        let zonal_only = json!({"dimensionsInfos": [{"dimensions": {"zone": "us-central1-a"}, "details": {"value": "2"}}]});
        assert_eq!(
            quota_info_limit(&zonal_only, "us-central1"),
            (0.0, "no entry")
        );
        assert_eq!(
            quota_info_limit(&json!({}), "us-central1"),
            (0.0, "no entry")
        );
    }

    #[test]
    fn h100_gpus_come_from_the_accelerators_else_from_the_a3_machine_type() {
        let z = "https://www.googleapis.com/compute/v1/projects/p/zones/us-central1-a";
        let acc = |t: &str, n: u64| json!({"guestAccelerators": [{"acceleratorType": format!("{z}/acceleratorTypes/{t}"), "acceleratorCount": n}]});
        let mt = |m: &str| json!({ "machineType": format!("{z}/machineTypes/{m}") });
        assert_eq!(h100_gpus_of(&acc("nvidia-h100-80gb", 2)), 2);
        assert_eq!(h100_gpus_of(&acc("nvidia-l4", 1)), 0);
        assert_eq!(h100_gpus_of(&mt("a3-highgpu-8g")), 8);
        assert_eq!(h100_gpus_of(&mt("a3-highgpu-1g")), 1);
        assert_eq!(h100_gpus_of(&mt("a3-ultragpu-8g")), 0, "H200");
        assert_eq!(h100_gpus_of(&mt("n2-standard-4")), 0);
    }

    #[test]
    fn the_quota_text_names_where_it_was_read() {
        let c = H100Quota::CloudQuotas {
            limit: 0.0,
            entry: "default entry",
            usage: 0.0,
        };
        assert_eq!(c.free(), Some(0.0));
        assert_eq!(
            c.text("us-east5"),
            "0 (Cloud Quotas API, PREEMPTIBLE-NVIDIA-H100-GPUS-per-project-region, default entry)"
        );
        let u = H100Quota::Unreadable("HTTP 403: denied".into());
        assert_eq!(u.free(), None);
        assert!(u.line("us-central1").starts_with(
            "H100 spot quota in us-central1 not readable (Cloud Quotas API: HTTP 403: denied); "
        ));
        assert_eq!(
            H100Quota::Compute {
                limit: 1.0,
                usage: 0.0
            }
            .line("us-central1"),
            "H100 spot quota in us-central1: 1 (used 0)"
        );
    }
}
