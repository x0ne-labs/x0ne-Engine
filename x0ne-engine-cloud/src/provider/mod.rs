//! The provider contract. GCP first; Azure is named, and refused, because its
//! proof is a different guarantee (see `azure`).

pub mod azure;
pub mod gcp;

use std::future::Future;
use std::net::IpAddr;
use std::path::PathBuf;
use std::pin::Pin;

use crate::quote::Quote;
use crate::Failure;

pub type BoxFut<'a, T> = Pin<Box<dyn Future<Output = Result<T, Failure>> + 'a>>;

/// What the hardware proof establishes, by name. A verifier exists for
/// `TdxQuoteDirect` only; the others are refused, never presented as it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Guarantee {
    /// Intel TDX quote made by the guest itself, checked to Intel's root;
    /// GPU through NVIDIA NRAS (GCP a3-highgpu with TDX).
    TdxQuoteDirect,
    /// AMD SEV-SNP report produced by Microsoft's paravisor (HCL); the guest's
    /// proof is rooted in the paravisor (Azure NCC H100 v5).
    SnpViaParavisor,
}

impl Guarantee {
    pub fn name(&self) -> &'static str {
        match self {
            Guarantee::TdxQuoteDirect => "TdxQuoteDirect",
            Guarantee::SnpViaParavisor => "SnpViaParavisor",
        }
    }
    pub fn verifiable(&self) -> bool {
        matches!(self, Guarantee::TdxQuoteDirect)
    }
}

#[derive(Debug, Clone)]
pub struct SessionRequest {
    pub project: String,
    /// None: the provider picks the first zone with quota.
    pub zone: Option<String>,
    pub machine_type: String,
    pub model: String,
    pub max_run_secs: u64,
    pub image: String,
    pub session_id: String,
    /// The firewall rule's sources (IPv4 CIDR). Empty: `create` refuses,
    /// never opens tcp:443 to everyone.
    pub source_ranges: Vec<String>,
    /// `--nras-key-file`: its own metadata item, never `x0ne-config`.
    pub nras_key: Option<crate::nras_key::NrasKey>,
}

impl SessionRequest {
    pub fn vm_name(&self) -> String {
        format!("{}{}", crate::NAME_PREFIX, self.session_id)
    }
}

#[derive(Debug, Clone)]
pub struct Identity {
    pub account: Option<String>,
    pub expires_in_secs: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Shape {
    pub vcpus: u32,
    pub memory_gib: f64,
    pub gpus: u32,
}

#[derive(Debug, Clone)]
pub struct Preflight {
    pub zone: String,
    /// `H100 spot quota in <region>: …`, with where it was read.
    pub quota: String,
    pub shape: Shape,
    pub image: ImageRef,
    /// x0ne-engine VMs alive in the project: a second session is refused.
    pub live: Vec<Leftover>,
    /// Other x0ne-engine leftovers, shown with how to remove them.
    pub leftovers: Vec<Leftover>,
}

/// What the lifecycle created, by name (deletion needs nothing else).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Instance {
    pub project: String,
    pub zone: String,
    pub name: String,
    pub firewall: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VmState {
    /// Being created or booting (provider status).
    Pending(String),
    Running(IpAddr),
    /// Stopped or terminated: preempted, or at the end of its run.
    Stopped(String),
    /// No such VM.
    Gone,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeftoverKind {
    Instance,
    Disk,
    Address,
    Firewall,
    Bucket,
}

impl LeftoverKind {
    pub fn label(&self) -> &'static str {
        match self {
            LeftoverKind::Instance => "VM",
            LeftoverKind::Disk => "disk",
            LeftoverKind::Address => "IP address",
            LeftoverKind::Firewall => "firewall rule",
            LeftoverKind::Bucket => "bucket",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Leftover {
    pub kind: LeftoverKind,
    pub name: String,
    /// Zone, region, or "global".
    pub location: String,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct ImageRef {
    pub project: String,
    pub name: String,
    /// READY, PENDING, FAILED, or MISSING.
    pub status: String,
    /// The disk.tar.gz sha256 `image import` wrote in the description.
    pub disk_sha256: Option<String>,
    /// It carries the label `image import` writes.
    pub ours: bool,
}

#[derive(Debug, Clone)]
pub struct ImportRequest {
    pub project: String,
    /// Chosen by the caller, so it can delete it if the import is interrupted.
    pub bucket: String,
    pub region: String,
    pub image: String,
    pub file: PathBuf,
    pub size: u64,
    pub sha256_hex: String,
}

/// A piece of the serial console. `start` may be past the position asked for
/// when the provider dropped older output; `next` is where the next read starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerialChunk {
    pub start: u64,
    pub next: u64,
    pub contents: String,
}

/// Every call takes the token from the provider's token source.
pub trait Provider {
    fn name(&self) -> &'static str;
    fn display_name(&self) -> &'static str;
    fn guarantee(&self) -> Guarantee;
    /// Who the token belongs to and how long it lasts.
    fn identity(&self) -> BoxFut<'_, Identity>;
    /// Quota, live sessions and leftovers, image, machine shape.
    fn preflight<'a>(&'a self, r: &'a SessionRequest) -> BoxFut<'a, Preflight>;
    /// Price per hour, its source and date, the maximum cost.
    fn quote<'a>(&'a self, r: &'a SessionRequest, p: &'a Preflight) -> BoxFut<'a, Quote>;
    /// Firewall rule and VM. Not cancelled midway by the caller: whatever
    /// it returns, `delete` is called with the same names afterwards.
    fn create<'a>(
        &'a self,
        r: &'a SessionRequest,
        i: &'a Instance,
        config: &'a [u8],
    ) -> BoxFut<'a, ()>;
    fn instance_state<'a>(&'a self, i: &'a Instance) -> BoxFut<'a, VmState>;
    /// Boot diagnostics (serial console). Untrusted text: saved, never printed.
    fn serial_output<'a>(&'a self, i: &'a Instance) -> BoxFut<'a, String>;
    /// The serial console from byte `start` on, as far as the provider holds
    /// it. Untrusted text: `td_console` parses it, `up` prints one cleaned line.
    fn serial_since<'a>(&'a self, i: &'a Instance, start: u64) -> BoxFut<'a, SerialChunk>;
    /// Idempotent; Ok only once the provider confirms both are gone.
    fn delete<'a>(&'a self, i: &'a Instance, insert_may_be_pending: bool) -> BoxFut<'a, ()>;
    /// x0ne-engine resources (name prefix AND label/marker), nothing else.
    fn sweep(&self) -> BoxFut<'_, Vec<Leftover>>;
    fn delete_leftover<'a>(&'a self, l: &'a Leftover) -> BoxFut<'a, ()>;
    fn image(&self, name: &str) -> BoxFut<'_, ImageRef>;
    fn import_image<'a>(&'a self, r: &'a ImportRequest) -> BoxFut<'a, ImageRef>;
    /// The VMs a disk made from image `name` is attached to.
    fn image_users<'a>(&'a self, name: &'a str) -> BoxFut<'a, Vec<String>>;
    /// Ok only once the provider confirms image `name` is gone.
    fn delete_image<'a>(&'a self, name: &'a str) -> BoxFut<'a, ()>;
    /// Commands that delete what `delete` could not, for the critical message.
    fn manual_delete_commands(&self, i: &Instance) -> Vec<String>;
}
