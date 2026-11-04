// SPDX-License-Identifier: Apache-2.0
//! The image reference the client pins (written by the image build,
//! x0ne-OS; read here). What each register must say, and how it is judged:
//!
//! - MRTD: a Google launch endorsement under the pinned root, with the
//!   `ram_gib` of the machine type (`gce_endorsement`);
//! - RTMR0, policy `report`: replayed, compared to the known values,
//!   reported, not blocking. Policy `gce-endorsed`: every event that extends
//!   it, in log order, is the reference's for the machine type, where the
//!   slot `firmware:cfv` is the configuration volume of the firmware binary
//!   Google endorsed (`tdvf`: its SHA-384 is the signed one and the MRTD it
//!   measures is the quote's); a missing, extra, reordered or other event
//!   refuses;
//! - RTMR1: the reference GPT event exactly once, before the first boot
//!   application; the reference boot applications, each once, in the
//!   reference's order; nothing else but benign content (separator, EFI
//!   action) carrying the hash of its own data;
//! - RTMR2: the reference sections, each once, plus separators of their
//!   own hash; nothing else;
//! - RTMR3: zero, and nothing in the log extends it.
//!
//! All four RTMRs must replay from the CCEL to the quote's values first.
//! The replay covers digests only: `event_type` and the data are not
//! authenticated by the quote, so the policy is a closed list on digests
//! (2026-10-12 attack: a skipped type with a foreign digest passed).
//! Schema v2 pins the GPT and orders the applications: under v1, a correctly
//! hashed foreign GPT, a log without the UKI and the UKI twice all passed
//! (2026-10-17). A v1 reference is refused by name.

use crate::ccel::{self, Event};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use x0ne_confidential::tdx_quote::TdxQuoteClaims;

pub const SCHEMA: &str = "x0ne-engine/reference/v2";
/// RTMR0 replayed and reported, not blocking.
pub const RTMR0_REPORT: &str = "report";
/// RTMR0 pinned: the endorsed firmware's CFV and the reference's host events.
pub const RTMR0_GCE_ENDORSED: &str = "gce-endorsed";
/// In `rtmr0.events_sha384`: the event of the endorsed firmware's CFV.
pub const FIRMWARE_CFV_SLOT: &str = "firmware:cfv";
/// The schema before the RTMR1 GPT pin: refused with what to use instead.
pub const SCHEMA_V1: &str = "x0ne-engine/reference/v1";
/// The oldest image a reference may name. Raise it when an image is retired
/// for a flaw: every build carrying the new floor then refuses a reference
/// for the old image, signed or not, so it cannot be handed back to the
/// relay. 0.1: nothing is retired yet; the images measured on GCE are 0.2
/// (2026-10-17) and 0.2.3 (2026-10-18), 0.1.0 is the name the test fixtures
/// carry.
pub const MIN_IMAGE_VERSION: &str = "0.1";
pub const DEFAULT_DIR: &str = "/usr/share/x0ne-engine/reference";
pub const MAX_BYTES: u64 = 1024 * 1024;
/// The relay's exit when the reference is not signed by the pinned archive
/// key: no signature, another key, other bytes, or no key in this build
/// (built before the signing ceremony).
pub const EXIT_SIGNATURE: i32 = 99;
/// The reference unreadable, not owned right, or not a v1 reference.
pub const EXIT_UNUSABLE: i32 = 89;
/// The reference reports RTMR0 instead of pinning it (`rtmr0.policy`
/// [`RTMR0_REPORT`]): the attested path, relay and SDK, admits
/// [`RTMR0_GCE_ENDORSED`] alone. The lab's replay
/// ([`check_image_rtmr0_reported`]) is the only reader of such a reference.
pub const EXIT_RTMR0_UNPINNED: i32 = 104;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub schema: String,
    pub provider: String,
    pub tee: String,
    pub image: Image,
    pub mrtd: Mrtd,
    pub rtmr0: Rtmr0,
    pub rtmr1: Rtmr1,
    pub rtmr2: Rtmr2,
    pub rtmr3: Rtmr3,
    pub td_attributes: TdAttributes,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Image {
    pub name: String,
    pub version: String,
    pub gcp_image: String,
    pub disk_tar_sha256: String,
    #[serde(default)]
    pub source: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mrtd {
    pub policy: String,
    /// SHA-256 of the SPKI of the Google root the verifier pins.
    pub root_spki_sha256: String,
    /// Machine type → GiB the endorsement's measurement must name.
    #[serde(default)]
    pub ram_gib: BTreeMap<String, u32>,
    /// Floor on the TDX SVN of the endorsement that names the MRTD: an
    /// older firmware Google once endorsed is refused. 0: no floor.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub min_svn: u32,
}

// serde's skip_serializing_if passes a reference.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_zero(v: &u32) -> bool {
    *v == 0
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rtmr0 {
    pub policy: String,
    /// `report`: machine type → the RTMR0 values seen, compared and reported.
    #[serde(default)]
    pub known: BTreeMap<String, Vec<String>>,
    /// `gce-endorsed`: machine type → every event that extends RTMR0, in log
    /// order: [`FIRMWARE_CFV_SLOT`] once, a SHA-384 for each of the host's.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub events_sha384: BTreeMap<String, Vec<String>>,
}

impl Rtmr0 {
    /// Each policy with its own field only; under `gce-endorsed`, every
    /// machine type's list places the firmware's CFV exactly once.
    fn validate(&self) -> Result<(), String> {
        match self.policy.as_str() {
            RTMR0_REPORT => {
                if !self.events_sha384.is_empty() {
                    return Err(format!(
                        "rtmr0.events_sha384 is for policy {RTMR0_GCE_ENDORSED:?}, not {RTMR0_REPORT:?}"
                    ));
                }
            }
            RTMR0_GCE_ENDORSED => {
                if self.known.values().any(|v| !v.is_empty()) {
                    return Err(format!(
                        "rtmr0.known is for policy {RTMR0_REPORT:?}: under {RTMR0_GCE_ENDORSED:?} \
                         RTMR0 is pinned by rtmr0.events_sha384"
                    ));
                }
                if self.events_sha384.is_empty() {
                    return Err(format!(
                        "rtmr0.events_sha384 is empty: policy {RTMR0_GCE_ENDORSED:?} pins RTMR0 \
                         per machine type"
                    ));
                }
                for (machine, list) in &self.events_sha384 {
                    let slots = list.iter().filter(|d| *d == FIRMWARE_CFV_SLOT).count();
                    if slots != 1 {
                        return Err(format!(
                            "rtmr0.events_sha384[{machine:?}] places {FIRMWARE_CFV_SLOT:?} {slots} \
                             time(s): the firmware's configuration volume is measured exactly once"
                        ));
                    }
                }
            }
            other => {
                return Err(format!(
                    "rtmr0.policy {other:?}: this verifier implements {RTMR0_REPORT:?} and \
                     {RTMR0_GCE_ENDORSED:?}"
                ))
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rtmr1 {
    pub policy: String,
    /// SHA-384 of the image disk's UEFI_GPT_DATA: the one `EV_EFI_GPT_EVENT`.
    pub gpt_event_sha384: String,
    /// Every boot application, in measurement order, each measured once.
    pub boot_applications_sha384: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rtmr2 {
    pub policy: String,
    pub events_sha384: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rtmr3 {
    pub policy: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TdAttributes {
    pub debug: bool,
}

/// A reference as loaded: the file is named in every verdict. Only
/// [`load`] makes one, after the signature and the transparency log
/// (`signed_by` and `logged` are not public).
// Not in x0ne-sdk (its build.rs sets `x0ne_sdk`): the caller hands it the
// reference, parsed by `parse`; no file, no archive key, no log here.
#[cfg(not(x0ne_sdk))]
#[derive(Debug, Clone)]
pub struct Loaded {
    pub reference: Reference,
    pub path: PathBuf,
    pub sha256_hex: String,
    /// `--reference FILE` rather than the installed default.
    pub explicit: bool,
    pub(crate) signed_by: crate::openpgp::Signer,
    pub(crate) logged: crate::transparency::Logged,
}

#[cfg(not(x0ne_sdk))]
impl Loaded {
    pub fn signed_by(&self) -> &crate::openpgp::Signer {
        &self.signed_by
    }

    /// Where the pinned transparency log shows this reference.
    pub fn logged(&self) -> &crate::transparency::Logged {
        &self.logged
    }
}

/// Why a reference was refused, with the relay's exit code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    Unusable(String),
    Signature(String),
    /// This build has no usable archive key: nothing can pass.
    NoKey(String),
    /// Not shown to be in the transparency log this build pins, or this
    /// build pins none: nothing can pass.
    NotLogged(String),
    /// Signed, logged and valid, but `rtmr0.policy` is not `gce-endorsed`:
    /// RTMR0 would be reported, not pinned ([`EXIT_RTMR0_UNPINNED`]).
    Rtmr0Unpinned(String),
}

impl LoadError {
    pub fn exit_code(&self) -> i32 {
        match self {
            LoadError::Unusable(_) => EXIT_UNUSABLE,
            LoadError::Signature(_) | LoadError::NoKey(_) => EXIT_SIGNATURE,
            LoadError::NotLogged(_) => crate::transparency::EXIT_NOT_LOGGED,
            LoadError::Rtmr0Unpinned(_) => EXIT_RTMR0_UNPINNED,
        }
    }
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Unusable(e) | LoadError::NoKey(e) => f.write_str(e),
            LoadError::Signature(e) => write!(f, "not signed by the x0ne archive key: {e}"),
            LoadError::NotLogged(e) => {
                write!(f, "not in the transparency log this build pins: {e}")
            }
            LoadError::Rtmr0Unpinned(e) => write!(f, "RTMR0 not pinned: {e}"),
        }
    }
}

/// The attested path pins RTMR0: a reference whose `rtmr0.policy` is not
/// [`RTMR0_GCE_ENDORSED`] is refused by both loaders ([`load`],
/// [`admit_offline`]) once signed, logged and parsed. `parse` still reads
/// it: the format did not change, the lab's replay still takes it.
fn require_rtmr0_pinned(reference: &Reference) -> Result<(), LoadError> {
    if reference.rtmr0.policy == RTMR0_GCE_ENDORSED {
        return Ok(());
    }
    Err(LoadError::Rtmr0Unpinned(format!(
        "{} {}: rtmr0.policy {:?} reports RTMR0 and pins nothing; the relay and the SDK \
         admit {RTMR0_GCE_ENDORSED:?} alone, which pins RTMR0 to the firmware Google \
         endorsed and to the host events of a measured boot (docs/pinning.md)",
        reference.image.name, reference.image.version, reference.rtmr0.policy
    )))
}

/// `<reference>.sig`: the detached signature, next to the file.
pub fn signature_path(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".sig");
    PathBuf::from(p)
}

/// What the image checks established, for the verdict.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageReport {
    pub rtmr0_hex: String,
    /// `None`: no known value for this machine type.
    pub rtmr0_known: Option<bool>,
    /// Policy `gce-endorsed`, passed: RTMR0 is the value the endorsed
    /// firmware and the reference give.
    pub rtmr0_pinned: bool,
    pub boot_applications: Vec<String>,
    pub sections: Vec<String>,
    pub warnings: Vec<String>,
}

fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

pub fn parse(bytes: &[u8]) -> Result<Reference, String> {
    // The schema first: a v1 file would otherwise fail as "missing field".
    #[derive(Deserialize)]
    struct Head {
        schema: Option<serde_json::Value>,
    }
    let head: Head =
        serde_json::from_slice(bytes).map_err(|e| format!("not an image reference: {e}"))?;
    match head.schema.as_ref().and_then(|s| s.as_str()) {
        Some(SCHEMA) => {}
        Some(SCHEMA_V1) => {
            return Err(format!(
                "schema {SCHEMA_V1}: written before RTMR1 pinned the disk's GPT and the order of \
                 the boot applications, refused by this version; use the {SCHEMA} reference.json \
                 built with the image"
            ))
        }
        Some(other) => return Err(format!("schema {other:?}, expected {SCHEMA:?}")),
        None => return Err(format!("no \"schema\" string, expected {SCHEMA:?}")),
    }
    let r: Reference =
        serde_json::from_slice(bytes).map_err(|e| format!("not an image reference: {e}"))?;
    r.validate()?;
    Ok(r)
}

impl Reference {
    /// The policies of schema v2, by name, and the digest shapes: a reference
    /// asking for another policy is refused rather than approximated.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != SCHEMA {
            return Err(format!("schema {:?}, expected {SCHEMA:?}", self.schema));
        }
        if self.provider != "gcp" || self.tee != "intel-tdx" {
            return Err(format!(
                "provider/tee {:?}/{:?}: this verifier implements gcp/intel-tdx",
                self.provider, self.tee
            ));
        }
        let floor =
            dotted_version(MIN_IMAGE_VERSION).expect("MIN_IMAGE_VERSION is digits and dots");
        match dotted_version(&self.image.version) {
            Some(v) if v >= floor => {}
            Some(_) => {
                return Err(format!(
                    "image.version {:?}: below {MIN_IMAGE_VERSION}, the oldest image this build accepts",
                    self.image.version
                ))
            }
            None => {
                return Err(format!(
                    "image.version {:?}: not a dotted version",
                    self.image.version
                ))
            }
        }
        let policies = [
            ("mrtd", self.mrtd.policy.as_str(), "gce-launch-endorsement"),
            ("rtmr1", self.rtmr1.policy.as_str(), "eventlog"),
            ("rtmr2", self.rtmr2.policy.as_str(), "eventlog"),
            ("rtmr3", self.rtmr3.policy.as_str(), "zero"),
        ];
        for (reg, have, want) in policies {
            if have != want {
                return Err(format!(
                    "{reg}.policy {have:?}: this verifier implements {want:?}"
                ));
            }
        }
        self.rtmr0.validate()?;
        if !is_hex(&self.mrtd.root_spki_sha256, 64) {
            return Err("mrtd.root_spki_sha256 is not 64 lowercase hex".into());
        }
        if self.rtmr1.boot_applications_sha384.is_empty() {
            return Err("rtmr1.boot_applications_sha384 is empty".into());
        }
        if self.rtmr2.events_sha384.is_empty() {
            return Err("rtmr2.events_sha384 is empty".into());
        }
        if self
            .rtmr1
            .boot_applications_sha384
            .contains(&self.rtmr1.gpt_event_sha384)
        {
            return Err("rtmr1.gpt_event_sha384 is also a boot application digest".into());
        }
        let digests = std::iter::once(&self.rtmr1.gpt_event_sha384)
            .chain(self.rtmr1.boot_applications_sha384.iter())
            .chain(self.rtmr2.events_sha384.values())
            .chain(self.rtmr0.known.values().flatten())
            .chain(
                self.rtmr0
                    .events_sha384
                    .values()
                    .flatten()
                    .filter(|d| *d != FIRMWARE_CFV_SLOT),
            );
        for d in digests {
            if !is_hex(d, 96) {
                return Err(format!("digest {d:?} is not 96 lowercase hex"));
            }
        }
        Ok(())
    }
}

/// Load a reference file, its signature `<file>.sig` and its proof of
/// inclusion `<file>.tlog-proof`. A build without a usable archive key or
/// transparency log pin refuses first, whatever the files. The signature is
/// checked against the pinned archive key, then the bytes are shown in the
/// pinned log (`transparency.rs`, consistent with the last checkpoint kept
/// under `state_dir`), before a byte is parsed; the installed default must
/// also belong to root and be writable by nobody else. An explicit
/// `--reference` is named, with its digest, in the verdict.
#[cfg(not(x0ne_sdk))]
pub fn load(
    path: &Path,
    explicit: bool,
    key: &crate::archive_key::Anchor,
    log: &crate::transparency::Anchor,
    state_dir: &Path,
    now_unix: u64,
) -> Result<Loaded, LoadError> {
    from_signed(
        load_signed(path, explicit, key, log, state_dir, now_unix)?,
        path,
        explicit,
    )
}

/// The bytes [`load_signed`] admitted, parsed as a TDX reference.
#[cfg(not(x0ne_sdk))]
pub(crate) fn from_signed(
    s: SignedBytes,
    path: &Path,
    explicit: bool,
) -> Result<Loaded, LoadError> {
    let reference =
        parse(&s.bytes).map_err(|e| LoadError::Unusable(format!("{}: {e}", path.display())))?;
    require_rtmr0_pinned(&reference)
        .map_err(|e| LoadError::Rtmr0Unpinned(format!("{}: {e}", path.display())))?;
    Ok(Loaded {
        reference,
        path: path.to_path_buf(),
        sha256_hex: s.sha256_hex,
        explicit,
        signed_by: s.signed_by,
        logged: s.logged,
    })
}

/// A reference file's bytes, as [`load`] admits them before parsing: the
/// signature under the pinned archive key, then the pinned log. Shared by
/// every reference schema (`snp_reference.rs` too), so none skips a step.
#[cfg(not(x0ne_sdk))]
pub(crate) struct SignedBytes {
    pub(crate) bytes: Vec<u8>,
    pub(crate) sha256_hex: String,
    pub(crate) signed_by: crate::openpgp::Signer,
    pub(crate) logged: crate::transparency::Logged,
}

#[cfg(not(x0ne_sdk))]
pub(crate) fn load_signed(
    path: &Path,
    explicit: bool,
    key: &crate::archive_key::Anchor,
    log: &crate::transparency::Anchor,
    state_dir: &Path,
    now_unix: u64,
) -> Result<SignedBytes, LoadError> {
    key.key().map_err(LoadError::NoKey)?;
    log.policy().map_err(LoadError::NotLogged)?;
    let bytes = read(path, explicit).map_err(LoadError::Unusable)?;
    let sig_path = signature_path(path);
    let sig = crate::fsread::read_capped(&sig_path, crate::openpgp::MAX_SIGNATURE_BYTES)
        .map_err(|e| LoadError::Signature(format!("{}: {e}", sig_path.display())))?;
    let (signed_by, logged) = admit_signed(&bytes, &sig, key, log, now_unix, |_| {
        crate::transparency::check_file(path, &bytes, log, state_dir, now_unix)
    })
    .map_err(|e| match e {
        LoadError::Signature(e) => LoadError::Signature(format!("{}: {e}", sig_path.display())),
        other => other,
    })?;
    Ok(SignedBytes {
        sha256_hex: crate::cloud_config::hex(&crate::cloud_config::sha256(&bytes)),
        bytes,
        signed_by,
        logged,
    })
}

/// What admits a reference's bytes, once read, wherever they came from: this
/// build's archive key and log pins, then the signature over the bytes under
/// the pinned key, then `logged`, the check of the pinned log. [`load_signed`]
/// runs it on a file and its neighbours, [`admit_offline`] (x0ne-sdk) on bytes.
pub(crate) fn admit_signed(
    bytes: &[u8],
    signature: &[u8],
    key: &crate::archive_key::Anchor,
    log: &crate::transparency::Anchor,
    now_unix: u64,
    logged: impl FnOnce(&crate::transparency::Policy) -> Result<crate::transparency::Logged, String>,
) -> Result<(crate::openpgp::Signer, crate::transparency::Logged), LoadError> {
    key.key().map_err(LoadError::NoKey)?;
    let policy = log.policy().map_err(LoadError::NotLogged)?;
    let signed_by = key
        .verify(bytes, signature, now_unix)
        .map_err(LoadError::Signature)?;
    let logged = logged(policy).map_err(LoadError::NotLogged)?;
    Ok((signed_by, logged))
}

/// A reference [`admit_offline`] admitted: its bytes' digest, who signed
/// them, where the log shows them.
#[derive(Debug, Clone)]
pub struct Admitted {
    pub reference: Reference,
    pub sha256_hex: String,
    pub signed_by: crate::openpgp::Signer,
    pub logged: crate::transparency::Logged,
}

/// A reference handed over as bytes, with its detached signature and its log
/// proof (the `.sig` and `.tlog-proof` beside a file), admitted by the steps
/// [`load`] runs (`admit_signed`), under the caps it reads them with, the
/// log checked on the proof alone (`transparency::check_offline`: no refresh,
/// no checkpoint kept here; x0ne-sdk then holds it to the caller's memory of
/// the log, `transparency::remember`), then parsed. x0ne-sdk's admission: it
/// reads no file.
pub fn admit_offline(
    bytes: &[u8],
    signature: &[u8],
    proof: &str,
    key: &crate::archive_key::Anchor,
    log: &crate::transparency::Anchor,
    now_unix: u64,
) -> Result<Admitted, LoadError> {
    if bytes.len() as u64 > MAX_BYTES {
        return Err(LoadError::Unusable(format!(
            "{} bytes, cap {MAX_BYTES}",
            bytes.len()
        )));
    }
    if signature.len() as u64 > crate::openpgp::MAX_SIGNATURE_BYTES {
        return Err(LoadError::Signature(format!(
            "signature of {} bytes, cap {}",
            signature.len(),
            crate::openpgp::MAX_SIGNATURE_BYTES
        )));
    }
    if proof.len() as u64 > crate::transparency::MAX_PROOF_BYTES {
        return Err(LoadError::NotLogged(format!(
            "proof of {} bytes, cap {}",
            proof.len(),
            crate::transparency::MAX_PROOF_BYTES
        )));
    }
    let (signed_by, logged) = admit_signed(bytes, signature, key, log, now_unix, |policy| {
        crate::transparency::check_offline(bytes, proof, policy, now_unix)
    })?;
    let reference = parse(bytes).map_err(LoadError::Unusable)?;
    require_rtmr0_pinned(&reference)?;
    use sha2::Digest as _;
    Ok(Admitted {
        reference,
        sha256_hex: ccel::hex(&sha2::Sha256::digest(bytes)),
        signed_by,
        logged,
    })
}

#[cfg(not(x0ne_sdk))]
fn read(path: &Path, explicit: bool) -> Result<Vec<u8>, String> {
    read_as(path, explicit, 0, || {})
}

/// [`read`] with the owner an installed file must have, and `checked` run
/// between the checks and the read: the race test swaps the file there.
#[cfg(not(x0ne_sdk))]
fn read_as(
    path: &Path,
    explicit: bool,
    owner: u32,
    checked: impl FnOnce(),
) -> Result<Vec<u8>, String> {
    #[cfg(not(unix))]
    let _ = owner;
    let (file, len) =
        crate::fsread::open_regular(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if len > MAX_BYTES {
        return Err(format!("{}: {len} bytes, cap {MAX_BYTES}", path.display()));
    }
    #[cfg(unix)]
    if !explicit {
        use std::os::unix::fs::MetadataExt;
        let meta = file
            .metadata()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if meta.uid() != owner {
            return Err(format!(
                "{}: installed reference must belong to root (uid {})",
                path.display(),
                meta.uid()
            ));
        }
        if meta.mode() & 0o022 != 0 {
            return Err(format!(
                "{}: installed reference is writable by group or others (mode {:o})",
                path.display(),
                meta.mode() & 0o777
            ));
        }
    }
    checked();
    crate::fsread::read_opened(file, len, MAX_BYTES, path)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// `"0.2.3"` as `[0, 2, 3]`: digits and dots only, no empty part. Compared
/// as vectors, `0.2` sits below `0.2.0` and `0.2.3`.
pub(crate) fn dotted_version(v: &str) -> Option<Vec<u64>> {
    v.split('.')
        .map(|p| {
            (!p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
                .then(|| p.parse().ok())
                .flatten()
        })
        .collect()
}

/// The version of `x0ne-cvm-<version>.json`, digits and dots only. Its
/// neighbours in x0ne-OS `cvm/` (`.manifest.json`, `.measurements-predicted.json`)
/// sorted above it before 2026-10-17 and were taken for the reference. Kept
/// byte for byte as the cloud CLI's copy (`image_import.rs`, held by its
/// source sweep), so not written through [`dotted_version`].
fn version_key(name: &str) -> Option<Vec<u64>> {
    name.strip_prefix("x0ne-cvm-")?
        .strip_suffix(".json")?
        .split('.')
        .map(|p| {
            p.bytes()
                .all(|b| b.is_ascii_digit())
                .then(|| p.parse().ok())
                .flatten()
        })
        .collect()
}

/// The newest `x0ne-cvm-<version>.json` in `dir`.
#[cfg(not(x0ne_sdk))]
pub fn default_path_in(dir: &Path) -> Result<PathBuf, String> {
    let newest = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter_map(|n| version_key(&n).map(|v| (v, n)))
        .max()
        .map(|(_, n)| n);
    let newest = newest.ok_or_else(|| {
        format!(
            "{}: no x0ne-cvm-<version>.json reference installed",
            dir.display()
        )
    })?;
    Ok(dir.join(newest))
}

#[cfg(not(x0ne_sdk))]
pub fn default_path() -> Result<PathBuf, String> {
    default_path_in(Path::new(DEFAULT_DIR))
}

/// MRTD against the endorsement: the measurement must name this MRTD with
/// the RAM size the reference gives for the machine type, and the
/// endorsement's TDX SVN must reach `mrtd.min_svn`.
pub fn check_mrtd(
    reference: &Reference,
    machine_type: &str,
    endorsement: &crate::gce_endorsement::Endorsement,
    mrtd: &[u8; 48],
) -> Result<u32, String> {
    let m = endorsement
        .tdx_measurement(mrtd)
        .ok_or_else(|| format!("MRTD {} is not in the Google endorsement", ccel::hex(mrtd)))?;
    let want = reference.mrtd.ram_gib.get(machine_type).ok_or_else(|| {
        format!("reference has no mrtd.ram_gib entry for machine type {machine_type:?}")
    })?;
    if m.ram_gib != *want {
        return Err(format!(
            "endorsement measurement is for {} GiB, the reference says {want} GiB for {machine_type}",
            m.ram_gib
        ));
    }
    if endorsement.tdx_svn < reference.mrtd.min_svn {
        return Err(format!(
            "the endorsed firmware has svn {}, below the reference's minimum {}",
            endorsement.tdx_svn, reference.mrtd.min_svn
        ));
    }
    Ok(m.ram_gib)
}

/// `EV_EFI_ACTION` strings the firmware measures into PCR4/PCR5, hence
/// RTMR1 (EDK2 MdePkg/Include/IndustryStandard/UefiTcgPlatform.h, read
/// 2026-10-13). "UEFI Debug Mode" is deliberately absent.
const EFI_ACTIONS: &[&[u8]] = &[
    b"Calling EFI Application from Boot Option",
    b"Returning from EFI Application from Boot Option",
    b"Exit Boot Services Invocation",
    b"Exit Boot Services Returned with Success",
    b"Exit Boot Services Returned with Failure",
];

fn type_label(event_type: u32) -> String {
    match event_type {
        ccel::EV_EFI_BOOT_SERVICES_APPLICATION => "EFI application".into(),
        ccel::EV_EFI_BOOT_SERVICES_DRIVER => "EFI driver".into(),
        ccel::EV_SEPARATOR => "separator".into(),
        ccel::EV_EFI_ACTION => "EFI action".into(),
        ccel::EV_EFI_GPT_EVENT => "GPT".into(),
        t => format!("type {t:#x}"),
    }
}

/// The events the firmware logs around our components, accepted for their
/// content: a closed list of types, a closed list of data, and the digest
/// must be the SHA-384 of that data. `event_type` alone proves nothing. The
/// GPT is not here: it is the reference's, pinned (schema v2).
fn benign_content(ev: &Event, rtmr: usize) -> Result<(), String> {
    let data = ev.data.as_slice();
    let listed = match (ev.event_type, rtmr) {
        (ccel::EV_SEPARATOR, _) => data == [0u8; 4] || data == [0xffu8; 4],
        (ccel::EV_EFI_ACTION, 1) => EFI_ACTIONS.contains(&data),
        _ => false,
    };
    if !listed {
        return Err(format!(
            "{} with {} bytes of data is not in the policy for RTMR{rtmr}",
            type_label(ev.event_type),
            data.len()
        ));
    }
    let digest = ev.sha384().ok_or("no SHA-384")?;
    if digest != ccel::sha384(data) {
        return Err("digest is not the SHA-384 of the event data".into());
    }
    Ok(())
}

/// What RTMR0 is checked against.
enum Rtmr0Check<'a> {
    /// The attested path: the binary Google endorsed, as this client measured it.
    Endorsed(&'a crate::tdvf::EndorsedFirmware),
    /// The lab's replay: reported with its warnings, pinned to nothing.
    Reported,
}

/// The lab's replay of a boot: RTMR1..3 as [`check_image_with`] checks
/// them, RTMR0 reported against `rtmr0.known` and pinned to nothing
/// (`run_fixtures.rs`, the tests). Never the attested path, whose loaders
/// refuse such a reference (104): a `gce-endorsed` reference refuses here,
/// it needs the endorsed binary.
pub fn check_image_rtmr0_reported(
    reference: &Reference,
    machine_type: &str,
    quote: &TdxQuoteClaims,
    events: &[Event],
) -> Result<ImageReport, String> {
    check_image_as(reference, machine_type, quote, events, Rtmr0Check::Reported)
}

/// The every-event list of RTMR0 for `rtmr0.events_sha384`, from a boot's
/// log: the endorsed firmware's CFV as [`FIRMWARE_CFV_SLOT`], which must be
/// there exactly once, every other event by its SHA-384.
pub fn rtmr0_template(events: &[Event], cfv_sha384: &[u8; 48]) -> Result<Vec<String>, String> {
    let mut list = Vec::new();
    let mut cfv_at: Option<usize> = None;
    for (n, d) in rtmr0_events(events)? {
        if d == *cfv_sha384 {
            if let Some(first) = cfv_at {
                return Err(format!(
                    "RTMR0 event {n}: the configuration volume {} twice (first: event {first})",
                    ccel::hex(cfv_sha384)
                ));
            }
            cfv_at = Some(n);
            list.push(FIRMWARE_CFV_SLOT.to_string());
        } else {
            list.push(ccel::hex(&d));
        }
    }
    if cfv_at.is_none() {
        return Err(format!(
            "the configuration volume {} of the endorsed firmware is not in the log's RTMR0",
            ccel::hex(cfv_sha384)
        ));
    }
    Ok(list)
}

/// The events that extend RTMR0, in log order, with their index in the log.
fn rtmr0_events(events: &[Event]) -> Result<Vec<(usize, [u8; 48])>, String> {
    events
        .iter()
        .enumerate()
        .filter(|(_, e)| e.rtmr() == Some(0) && e.event_type != ccel::EV_NO_ACTION)
        .map(|(n, e)| {
            e.sha384()
                .map(|d| (n, d))
                .ok_or_else(|| format!("RTMR0 event {n}: no SHA-384"))
        })
        .collect()
}

/// Policy `gce-endorsed`: every RTMR0 event, in order, is the reference's
/// for `machine_type`, the slot [`FIRMWARE_CFV_SLOT`] being the CFV of the
/// binary Google endorsed. The replay already matched the quote, so the
/// quote's RTMR0 is then the one these digests give, and nothing else.
fn check_rtmr0_pinned(
    reference: &Reference,
    machine_type: &str,
    events: &[Event],
    firmware: &crate::tdvf::EndorsedFirmware,
) -> Result<(), String> {
    let pins = reference
        .rtmr0
        .events_sha384
        .get(machine_type)
        .filter(|p| !p.is_empty())
        .ok_or_else(|| {
            format!("RTMR0: the reference pins no RTMR0 events for machine type {machine_type:?}")
        })?;
    let seen = rtmr0_events(events)?;
    for (k, (n, d)) in seen.iter().enumerate() {
        let Some(pin) = pins.get(k) else {
            return Err(format!(
                "RTMR0 event {n}: one event more than the {} the reference pins for {machine_type} ({})",
                pins.len(),
                ccel::hex(d)
            ));
        };
        if pin == FIRMWARE_CFV_SLOT {
            if *d != firmware.cfv_sha384 {
                return Err(format!(
                    "RTMR0 event {n}: the configuration volume measured ({}) is not the one of \
                     the firmware Google endorsed ({}, binary {})",
                    ccel::hex(d),
                    ccel::hex(&firmware.cfv_sha384),
                    firmware.sha384_hex
                ));
            }
        } else if ccel::hex(d) != *pin {
            return Err(format!(
                "RTMR0 event {n}: {} is not the reference's event {} of {} ({pin})",
                ccel::hex(d),
                k + 1,
                pins.len()
            ));
        }
    }
    if let Some(missing) = pins.get(seen.len()) {
        return Err(format!(
            "RTMR0: pinned event {} of {} ({missing}) was not measured",
            seen.len() + 1,
            pins.len()
        ));
    }
    Ok(())
}

/// RTMR0..3 against the reference, from the quote and its log, RTMR0
/// pinned to `firmware`, the binary Google endorsed as this client measured
/// it (`tdvf::verify_endorsed`): the attested path, relay and SDK. A
/// reference whose `rtmr0.policy` is not `gce-endorsed` is refused here
/// too, after both loaders refused it (104).
pub fn check_image_with(
    reference: &Reference,
    machine_type: &str,
    quote: &TdxQuoteClaims,
    events: &[Event],
    firmware: &crate::tdvf::EndorsedFirmware,
) -> Result<ImageReport, String> {
    check_image_as(
        reference,
        machine_type,
        quote,
        events,
        Rtmr0Check::Endorsed(firmware),
    )
}

fn check_image_as(
    reference: &Reference,
    machine_type: &str,
    quote: &TdxQuoteClaims,
    events: &[Event],
    rtmr0: Rtmr0Check<'_>,
) -> Result<ImageReport, String> {
    if matches!(rtmr0, Rtmr0Check::Endorsed(_)) && reference.rtmr0.policy != RTMR0_GCE_ENDORSED {
        return Err(format!(
            "RTMR0: the reference's rtmr0.policy {:?} reports RTMR0 and pins nothing; the \
             attested path takes {RTMR0_GCE_ENDORSED:?} alone",
            reference.rtmr0.policy
        ));
    }
    let mut report = ImageReport::default();
    // The log must explain the quote before it is believed about anything.
    let replayed = ccel::replay(events).map_err(|e| format!("CCEL: {e}"))?;
    for i in 0..4 {
        if replayed[i] != quote.rtmr[i] {
            return Err(format!(
                "CCEL replay of RTMR{i} ({}) differs from the quote ({})",
                ccel::hex(&replayed[i]),
                ccel::hex(&quote.rtmr[i])
            ));
        }
    }
    if quote.td_attributes[0] & 1 != 0 && !reference.td_attributes.debug {
        return Err("TD attributes: DEBUG is set".into());
    }
    // RTMR1 and RTMR2: a closed list on digests. EV_NO_ACTION does not
    // extend, which the replay above just proved against the quote.
    let mut expected: Vec<(&String, &String)> = reference.rtmr2.events_sha384.iter().collect();
    let apps = &reference.rtmr1.boot_applications_sha384;
    let gpt_pin = &reference.rtmr1.gpt_event_sha384;
    let mut gpt_at: Option<usize> = None;
    for (n, ev) in events.iter().enumerate() {
        let rtmr = match ev.rtmr() {
            Some(r @ (1 | 2)) if ev.event_type != ccel::EV_NO_ACTION => r,
            _ => continue,
        };
        let hex = ccel::hex(
            &ev.sha384()
                .ok_or_else(|| format!("RTMR{rtmr} event {n}: no SHA-384"))?,
        );
        if rtmr == 1 {
            if hex == *gpt_pin {
                if ev.event_type != ccel::EV_EFI_GPT_EVENT
                    || ccel::hex(&ccel::sha384(&ev.data)) != hex
                {
                    return Err(format!(
                        "RTMR1 event {n}: the reference GPT digest as a {} whose data does not hash to it",
                        type_label(ev.event_type)
                    ));
                }
                if let Some(first) = gpt_at {
                    return Err(format!(
                        "RTMR1 event {n}: the GPT is measured a second time (first: event {first})"
                    ));
                }
                // EDK2 DxeTpm2MeasureBootHandler: GPT once, before the first image loaded
                // from the disk. The GCE logs of 2026-10-14 and 2026-10-17 put it there.
                if !report.boot_applications.is_empty() {
                    return Err(format!(
                        "RTMR1 event {n}: the GPT after boot application {}; the firmware \
                         measures it before the first",
                        report.boot_applications.len()
                    ));
                }
                gpt_at = Some(n);
                continue;
            }
            if apps.contains(&hex) {
                let next = report.boot_applications.len();
                if apps.get(next) != Some(&hex) {
                    return Err(format!(
                        "RTMR1 event {n}: boot application {hex} repeated or out of order: the \
                         reference expects {}",
                        match apps.get(next) {
                            Some(d) =>
                                format!("application {} of {} ({d}) here", next + 1, apps.len()),
                            None => format!("no application after its {}", apps.len()),
                        }
                    ));
                }
                report.boot_applications.push(hex);
                continue;
            }
            if let Err(why) = benign_content(ev, 1) {
                return Err(format!(
                    "RTMR1 event {n}: {} {hex} is neither the reference GPT nor a reference boot \
                     application, and not benign content: {why}",
                    type_label(ev.event_type)
                ));
            }
            continue;
        }
        if let Some(pos) = expected.iter().position(|(_, d)| **d == hex) {
            let (name, _) = expected.swap_remove(pos);
            report.sections.push(name.clone());
            continue;
        }
        if let Err(why) = benign_content(ev, 2) {
            return Err(format!(
                "RTMR2 event {n} ({}, {}): digest {hex} is not in the reference, and not benign content: {why}",
                type_label(ev.event_type),
                ev.name().unwrap_or_else(|| "no name".into())
            ));
        }
    }
    if report.boot_applications.is_empty() {
        return Err("RTMR1: no EFI application measured, the log does not show our UKI".into());
    }
    if let Some(missing) = apps.get(report.boot_applications.len()) {
        return Err(format!(
            "RTMR1: boot application {} of {} ({missing}) was not measured",
            report.boot_applications.len() + 1,
            apps.len()
        ));
    }
    if gpt_at.is_none() {
        return Err(format!(
            "RTMR1: the reference GPT ({gpt_pin}) was not measured"
        ));
    }
    if let Some((name, d)) = expected.first() {
        return Err(format!(
            "RTMR2: reference section {name} ({d}) was not measured"
        ));
    }
    // RTMR3: nothing, ever.
    if quote.rtmr[3] != [0u8; 48] {
        return Err(format!(
            "RTMR3 is {}, expected zero",
            ccel::hex(&quote.rtmr[3])
        ));
    }
    if let Some((n, _)) = events
        .iter()
        .enumerate()
        .find(|(_, e)| e.rtmr() == Some(3) && e.event_type != ccel::EV_NO_ACTION)
    {
        return Err(format!("RTMR3: event {n} extends it"));
    }
    report.rtmr0_hex = ccel::hex(&quote.rtmr[0]);
    match rtmr0 {
        // RTMR0: pinned to the endorsed firmware and the reference's host events.
        Rtmr0Check::Endorsed(firmware) => {
            check_rtmr0_pinned(reference, machine_type, events, firmware)?;
            report.rtmr0_known = Some(true);
            report.rtmr0_pinned = true;
            return Ok(report);
        }
        Rtmr0Check::Reported if reference.rtmr0.policy == RTMR0_GCE_ENDORSED => {
            return Err(
                "RTMR0: the reference pins it to Google's endorsed firmware \
                        (gce-endorsed) and no endorsed firmware binary was checked"
                    .into(),
            );
        }
        Rtmr0Check::Reported => {}
    }
    // RTMR0: reported, the lab's replay.
    match reference.rtmr0.known.get(machine_type) {
        Some(list) if !list.is_empty() => {
            let known = list.contains(&report.rtmr0_hex);
            report.rtmr0_known = Some(known);
            if !known {
                report.warnings.push(format!(
                    "RTMR0 {} is not among the {} known value(s) for {machine_type} (host firmware tables; reported, not pinned)",
                    report.rtmr0_hex,
                    list.len()
                ));
            }
        }
        _ => report.warnings.push(format!(
            "RTMR0 {} has no known value for {machine_type} yet (reported, not pinned)",
            report.rtmr0_hex
        )),
    }
    Ok(report)
}

// Not when compiled into x0ne-sdk (its build.rs sets `x0ne_sdk`): these tests
// need this crate's fixtures and dev-dependencies.
#[cfg(all(test, not(x0ne_sdk)))]
pub(crate) mod tests {
    use super::*;
    use crate::ccel::{utf16le_nul, EV_EFI_BOOT_SERVICES_APPLICATION, EV_IPL, EV_SEPARATOR};

    /// SECURITY: a reference for an image older than the floor this build
    /// carries is refused, signature or not: an image retired for a flaw
    /// cannot be handed back to a newer relay (Trail of Bits, Meta Private
    /// Processing 2025, finding 21). A version that is not one is refused too.
    #[test]
    fn a_reference_for_an_image_below_the_version_floor_is_refused() {
        for old in ["0.0.1", "0.0.9", "0.0"] {
            let mut r = reference(&"0".repeat(64), MACHINE);
            r.image.version = old.into();
            let err = parse(&serde_json::to_vec(&r).unwrap()).unwrap_err();
            assert!(
                err.contains("image.version") && err.contains(old),
                "{old}: {err}"
            );
        }
        let mut r = reference(&"0".repeat(64), MACHINE);
        r.image.version = "latest".into();
        let err = parse(&serde_json::to_vec(&r).unwrap()).unwrap_err();
        assert!(err.contains("image.version"), "{err}");
    }

    /// The control: the images shipped so far, and newer ones, pass.
    #[test]
    fn the_shipped_image_versions_pass_the_floor() {
        for v in [MIN_IMAGE_VERSION, "0.1.0", "0.2", "0.2.3", "0.3", "1.0.0"] {
            let mut r = reference(&"0".repeat(64), MACHINE);
            r.image.version = v.into();
            parse(&serde_json::to_vec(&r).unwrap()).unwrap_or_else(|e| panic!("{v}: {e}"));
        }
    }

    pub(crate) const UKI: [u8; 48] = [0xaa; 48];
    pub(crate) const LINUX: [u8; 48] = [0xb1; 48];
    pub(crate) const INITRD: [u8; 48] = [0xb2; 48];
    pub(crate) const CMDLINE: [u8; 48] = [0xb3; 48];
    /// Stands for the image disk's UEFI_GPT_DATA.
    pub(crate) const GPT: &[u8] = b"EFI PART test header and its entries";

    /// The log our image produces, as the reference expects it.
    pub(crate) fn good_log() -> Vec<Event> {
        vec![
            Event::new_sha384(1, EV_SEPARATOR, ccel::sha384(&[0; 4]), vec![0; 4]),
            Event::new_sha384(2, EV_SEPARATOR, ccel::sha384(&[0; 4]), vec![0; 4]),
            Event::new_sha384(2, ccel::EV_EFI_GPT_EVENT, ccel::sha384(GPT), GPT.to_vec()),
            Event::new_sha384(
                2,
                EV_EFI_BOOT_SERVICES_APPLICATION,
                UKI,
                b"\\EFI\\BOOT\\BOOTX64.EFI".to_vec(),
            ),
            Event::new_sha384(3, EV_IPL, LINUX, utf16le_nul(".linux")),
            Event::new_sha384(3, EV_IPL, INITRD, utf16le_nul(".initrd")),
            Event::new_sha384(3, EV_IPL, CMDLINE, utf16le_nul(".cmdline")),
        ]
    }

    pub(crate) fn reference(root_spki_sha256: &str, machine: &str) -> Reference {
        let hex = ccel::hex;
        serde_json::from_value(serde_json::json!({
            "schema": SCHEMA, "provider": "gcp", "tee": "intel-tdx",
            "image": {"name": "x0ne-cvm", "version": "0.1.0", "gcp_image": "x0ne-cvm-0-1-0",
                      "disk_tar_sha256": "0".repeat(64), "source": {"x0ne_os": "test"}},
            "mrtd": {"policy": "gce-launch-endorsement", "root_spki_sha256": root_spki_sha256, "ram_gib": {machine: 0}},
            "rtmr0": {"policy": "report", "known": {machine: []}},
            "rtmr1": {"policy": "eventlog", "gpt_event_sha384": hex(&ccel::sha384(GPT)), "boot_applications_sha384": [hex(&UKI)]},
            "rtmr2": {"policy": "eventlog", "events_sha384": {".linux": hex(&LINUX), ".initrd": hex(&INITRD), ".cmdline": hex(&CMDLINE)}},
            "rtmr3": {"policy": "zero"},
            "td_attributes": {"debug": false}
        }))
        .unwrap()
    }

    pub(crate) fn claims_for(events: &[Event], mrtd: [u8; 48]) -> TdxQuoteClaims {
        TdxQuoteClaims {
            td_attributes: [0; 8],
            mr_td: mrtd,
            rtmr: ccel::replay(events).unwrap(),
            report_data: [0; 64],
            tee_type: 0x81,
        }
    }

    const MACHINE: &str = "a3-highgpu-1g";

    /// A `report` reference pins no RTMR0: the attested path refuses it
    /// whatever firmware was checked (the loaders refused it before, 104).
    /// The lab's replay, named as such, takes it and reports RTMR0 with a
    /// warning; it pins nothing, so a `gce-endorsed` reference refuses there.
    #[test]
    fn a_report_reference_is_refused_by_the_attested_path_and_replayed_by_the_lab_alone() {
        let r = reference(&"0".repeat(64), MACHINE);
        let log = good_log();
        let claims = claims_for(&log, [0x5; 48]);
        let err = check_image_with(&r, MACHINE, &claims, &log, &endorsed_firmware([0xcf; 48]))
            .unwrap_err();
        assert!(
            err.contains("RTMR0") && err.contains("\"report\"") && err.contains(RTMR0_GCE_ENDORSED),
            "{err}"
        );
        let report = check_image_rtmr0_reported(&r, MACHINE, &claims, &log).unwrap();
        assert_eq!(report.sections.len(), 3);
        assert_eq!(report.boot_applications, vec![ccel::hex(&UKI)]);
        assert_eq!((report.rtmr0_known, report.rtmr0_pinned), (None, false));
        assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
        let mut known = r.clone();
        known
            .rtmr0
            .known
            .insert(MACHINE.into(), vec![report.rtmr0_hex.clone()]);
        let again = check_image_rtmr0_reported(&known, MACHINE, &claims, &log).unwrap();
        assert_eq!((again.rtmr0_known, again.warnings.len()), (Some(true), 0));
        let endorsed = endorsed_reference(&"0".repeat(64), MACHINE, rtmr0_pins());
        let gce = gce_log([0xcf; 48]);
        let err =
            check_image_rtmr0_reported(&endorsed, MACHINE, &claims_for(&gce, [0x5; 48]), &gce)
                .unwrap_err();
        assert!(
            err.contains("no endorsed firmware binary was checked"),
            "{err}"
        );
    }

    /// The loaders refuse a reference that reports RTMR0 instead of pinning
    /// it, with 104, after its signature and log: `load` (the relay) and
    /// `admit_offline` (the SDK). The same file under `gce-endorsed` loads.
    #[test]
    fn a_reference_that_does_not_pin_rtmr0_is_refused_with_104_by_both_loaders() {
        let dir = tempfile::tempdir().unwrap();
        let p = written_as(dir.path(), &reference(&"0".repeat(64), MACHINE));
        sign_file(&p);
        let err = load(&p, true, &test_anchor(), NOW).map(|_| ()).unwrap_err();
        assert_eq!(err.exit_code(), EXIT_RTMR0_UNPINNED);
        assert!(matches!(err, LoadError::Rtmr0Unpinned(_)), "{err}");
        let text = err.to_string();
        assert!(
            text.contains("RTMR0")
                && text.contains("\"report\"")
                && text.contains(RTMR0_GCE_ENDORSED),
            "{text}"
        );
        let bytes = std::fs::read(&p).unwrap();
        let sig = std::fs::read(signature_path(&p)).unwrap();
        let proof = crate::transparency::test_log::proof_of_one(&bytes, NOW);
        let err = admit_offline(&bytes, &sig, &proof, &test_anchor(), &test_log(), NOW)
            .map(|_| ())
            .unwrap_err();
        assert_eq!(err.exit_code(), EXIT_RTMR0_UNPINNED, "{err}");
        // Pinned: both admit it.
        let endorsed = endorsed_reference(&"0".repeat(64), MACHINE, rtmr0_pins());
        let p = dir.path().join("x0ne-cvm-0.1.1.json");
        std::fs::write(&p, serde_json::to_vec(&endorsed).unwrap()).unwrap();
        sign_file(&p);
        let loaded = load(&p, true, &test_anchor(), NOW).expect("a gce-endorsed reference");
        assert_eq!(loaded.reference.rtmr0.policy, RTMR0_GCE_ENDORSED);
        let bytes = std::fs::read(&p).unwrap();
        let sig = std::fs::read(signature_path(&p)).unwrap();
        let proof = crate::transparency::test_log::proof_of_one(&bytes, NOW);
        admit_offline(&bytes, &sig, &proof, &test_anchor(), &test_log(), NOW)
            .expect("a gce-endorsed reference");
    }

    /// SECURITY: each register's policy, broken one at a time, refuses and
    /// names the register.
    #[test]
    fn every_register_policy_refuses_when_broken() {
        let r = reference(&"0".repeat(64), MACHINE);
        let log = good_log();
        // Replay ≠ quote.
        let mut claims = claims_for(&log, [0x5; 48]);
        claims.rtmr[1][0] ^= 1;
        assert!(check_image_rtmr0_reported(&r, MACHINE, &claims, &log)
            .unwrap_err()
            .contains("replay of RTMR1"));
        // RTMR1: another application.
        let mut other_app = log.clone();
        other_app[3] = Event::new_sha384(
            2,
            EV_EFI_BOOT_SERVICES_APPLICATION,
            [0xee; 48],
            b"shim".to_vec(),
        );
        let err =
            check_image_rtmr0_reported(&r, MACHINE, &claims_for(&other_app, [0x5; 48]), &other_app)
                .unwrap_err();
        assert!(
            err.contains("RTMR1 event 3") && err.contains("nor a reference boot application"),
            "{err}"
        );
        // RTMR1: a driver that is not listed.
        let mut driver = log.clone();
        driver.insert(
            3,
            Event::new_sha384(2, ccel::EV_EFI_BOOT_SERVICES_DRIVER, [0xdd; 48], vec![]),
        );
        assert!(
            check_image_rtmr0_reported(&r, MACHINE, &claims_for(&driver, [0x5; 48]), &driver)
                .unwrap_err()
                .contains("EFI driver")
        );
        // RTMR1: no application at all.
        let mut no_app = log.clone();
        no_app.remove(3);
        assert!(
            check_image_rtmr0_reported(&r, MACHINE, &claims_for(&no_app, [0x5; 48]), &no_app)
                .unwrap_err()
                .contains("no EFI application")
        );
        // RTMR2: a changed section.
        let mut cmdline = log.clone();
        cmdline[6] = Event::new_sha384(3, EV_IPL, [0xcc; 48], utf16le_nul(".cmdline"));
        let err =
            check_image_rtmr0_reported(&r, MACHINE, &claims_for(&cmdline, [0x5; 48]), &cmdline)
                .unwrap_err();
        assert!(
            err.contains("RTMR2 event 6") && err.contains(".cmdline"),
            "{err}"
        );
        // RTMR2: a missing section.
        let mut missing = log.clone();
        missing.remove(5);
        assert!(check_image_rtmr0_reported(
            &r,
            MACHINE,
            &claims_for(&missing, [0x5; 48]),
            &missing
        )
        .unwrap_err()
        .contains(".initrd"));
        // RTMR2: an extra event with a known digest is one event too many.
        let mut twice = log.clone();
        twice.push(Event::new_sha384(3, EV_IPL, LINUX, utf16le_nul(".linux")));
        assert!(
            check_image_rtmr0_reported(&r, MACHINE, &claims_for(&twice, [0x5; 48]), &twice)
                .unwrap_err()
                .contains("RTMR2 event 7")
        );
        // RTMR3 extended.
        let mut rtmr3 = log.clone();
        rtmr3.push(Event::new_sha384(4, EV_IPL, [0x33; 48], vec![]));
        assert!(
            check_image_rtmr0_reported(&r, MACHINE, &claims_for(&rtmr3, [0x5; 48]), &rtmr3)
                .unwrap_err()
                .contains("RTMR3")
        );
        // Debug TD.
        let mut debug = claims_for(&log, [0x5; 48]);
        debug.td_attributes[0] = 1;
        assert!(check_image_rtmr0_reported(&r, MACHINE, &debug, &log)
            .unwrap_err()
            .contains("DEBUG"));
    }

    use crate::ccel::sha384;

    /// SECURITY: `event_type` is not covered by the digest. An event that
    /// extends RTMR1 under a type the policy used to skip replays the quote
    /// and must still be refused: its digest is in no list.
    #[test]
    fn an_rtmr1_event_of_an_unlisted_type_with_a_foreign_digest_is_refused() {
        let r = reference(&"0".repeat(64), MACHINE);
        let mut log = good_log();
        log.insert(
            3,
            Event::new_sha384(2, ccel::EV_EVENT_TAG, [0xee; 48], b"tag".to_vec()),
        );
        let err = check_image_rtmr0_reported(&r, MACHINE, &claims_for(&log, [0x5; 48]), &log)
            .unwrap_err();
        assert!(err.contains("RTMR1 event 3"), "{err}");
    }

    /// SECURITY: a separator is accepted for what it hashes, not for its type.
    #[test]
    fn an_rtmr2_separator_with_a_foreign_digest_is_refused() {
        let r = reference(&"0".repeat(64), MACHINE);
        let mut log = good_log();
        log.push(Event::new_sha384(3, EV_SEPARATOR, [0xee; 48], vec![0; 4]));
        let err = check_image_rtmr0_reported(&r, MACHINE, &claims_for(&log, [0x5; 48]), &log)
            .unwrap_err();
        assert!(err.contains("RTMR2 event 7"), "{err}");
    }

    /// The benign events the firmware logs around our UKI (separators, EFI
    /// actions) pass only with the SHA-384 of their own data. A GPT is not
    /// benign: only the reference's, once.
    #[test]
    fn benign_content_is_accepted_only_with_the_hash_of_its_data() {
        let r = reference(&"0".repeat(64), MACHINE);
        let action = b"Calling EFI Application from Boot Option".to_vec();
        let gpt = b"\x45\x46\x49\x20PART fake gpt header and entries".to_vec();
        let mut log = good_log();
        log.insert(
            2,
            Event::new_sha384(2, ccel::EV_EFI_ACTION, sha384(&action), action.clone()),
        );
        log.push(Event::new_sha384(
            3,
            EV_SEPARATOR,
            sha384(&[0xff; 4]),
            vec![0xff; 4],
        ));
        let ok =
            check_image_rtmr0_reported(&r, MACHINE, &claims_for(&log, [0x5; 48]), &log).unwrap();
        assert_eq!(ok.boot_applications, vec![ccel::hex(&UKI)]);
        assert_eq!(ok.sections.len(), 3);
        // Another GPT, correctly hashed, in RTMR1.
        let mut bad = log.clone();
        bad.insert(
            3,
            Event::new_sha384(2, ccel::EV_EFI_GPT_EVENT, sha384(&gpt), gpt.clone()),
        );
        assert!(
            check_image_rtmr0_reported(&r, MACHINE, &claims_for(&bad, [0x5; 48]), &bad)
                .unwrap_err()
                .contains("RTMR1 event 3")
        );
        // Right string, foreign digest.
        let mut bad = log.clone();
        bad[2] = Event::new_sha384(2, ccel::EV_EFI_ACTION, [0xee; 48], action.clone());
        assert!(
            check_image_rtmr0_reported(&r, MACHINE, &claims_for(&bad, [0x5; 48]), &bad)
                .unwrap_err()
                .contains("RTMR1 event 2")
        );
        // A string the policy does not list, correctly hashed.
        let debug = b"UEFI Debug Mode".to_vec();
        let mut bad = log.clone();
        bad[2] = Event::new_sha384(2, ccel::EV_EFI_ACTION, sha384(&debug), debug);
        assert!(
            check_image_rtmr0_reported(&r, MACHINE, &claims_for(&bad, [0x5; 48]), &bad)
                .unwrap_err()
                .contains("RTMR1 event 2")
        );
        // A separator whose data is neither 0 nor ff, correctly hashed.
        let mut bad = log.clone();
        bad[1] = Event::new_sha384(2, EV_SEPARATOR, sha384(&[0, 0, 0, 1]), vec![0, 0, 0, 1]);
        assert!(
            check_image_rtmr0_reported(&r, MACHINE, &claims_for(&bad, [0x5; 48]), &bad)
                .unwrap_err()
                .contains("RTMR1 event 1")
        );
        // The GPT belongs to RTMR1: the same event in RTMR2 is refused.
        let mut bad = log.clone();
        bad.push(Event::new_sha384(
            3,
            ccel::EV_EFI_GPT_EVENT,
            sha384(&gpt),
            gpt,
        ));
        assert!(
            check_image_rtmr0_reported(&r, MACHINE, &claims_for(&bad, [0x5; 48]), &bad)
                .unwrap_err()
                .contains("RTMR2 event")
        );
    }

    #[test]
    fn mrtd_needs_the_endorsed_measurement_with_the_machines_ram() {
        let r = reference(&"0".repeat(64), MACHINE);
        let e = crate::gce_endorsement::verify(
            crate::gce_endorsement::tests::VENDORED,
            crate::gce_endorsement::ROOT_1_DER,
            crate::gce_endorsement::tests::NOW,
        )
        .unwrap();
        let mrtd = crate::gce_endorsement::tests::vendored_mrtd();
        assert_eq!(check_mrtd(&r, MACHINE, &e, &mrtd).unwrap(), 0);
        assert!(check_mrtd(&r, MACHINE, &e, &[0x7; 48])
            .unwrap_err()
            .contains("not in the Google endorsement"));
        assert!(check_mrtd(&r, "n2-standard-4", &e, &mrtd)
            .unwrap_err()
            .contains("no mrtd.ram_gib"));
        let mut other_ram = r.clone();
        other_ram.mrtd.ram_gib.insert(MACHINE.into(), 234);
        assert!(check_mrtd(&other_ram, MACHINE, &e, &mrtd)
            .unwrap_err()
            .contains("234 GiB"));
    }

    /// SECURITY: an older firmware Google once endorsed. `mrtd.min_svn` is a
    /// floor on the endorsement's TDX SVN, under either RTMR0 policy: the
    /// vendored endorsement (svn 2) passes a floor of 2 and is refused by 3.
    /// A reference without the field has no floor, and writes none.
    #[test]
    fn an_endorsement_below_the_minimum_firmware_svn_is_refused() {
        let e = crate::gce_endorsement::verify(
            crate::gce_endorsement::tests::VENDORED,
            crate::gce_endorsement::ROOT_1_DER,
            crate::gce_endorsement::tests::NOW,
        )
        .unwrap();
        assert_eq!(e.tdx_svn, 2);
        let mrtd = crate::gce_endorsement::tests::vendored_mrtd();
        let mut r = reference(&"0".repeat(64), MACHINE);
        assert_eq!(r.mrtd.min_svn, 0);
        assert!(!serde_json::to_string(&r).unwrap().contains("min_svn"));
        r.mrtd.min_svn = 2;
        check_mrtd(&r, MACHINE, &e, &mrtd).expect("svn 2 meets a floor of 2");
        r.mrtd.min_svn = 3;
        let err = check_mrtd(&r, MACHINE, &e, &mrtd).unwrap_err();
        assert!(err.contains("svn 2") && err.contains("minimum 3"), "{err}");
        // A reference file that carries the field loads it, and validates.
        let mut v = serde_json::to_value(reference(&"0".repeat(64), MACHINE)).unwrap();
        v["mrtd"]["min_svn"] = 3.into();
        let loaded: Reference = serde_json::from_value(v).unwrap();
        loaded.validate().unwrap();
        assert_eq!(loaded.mrtd.min_svn, 3);
        assert!(serde_json::to_string(&loaded)
            .unwrap()
            .contains("\"min_svn\":3"));
    }

    #[test]
    fn a_reference_asking_for_another_policy_or_shape_is_refused() {
        let good = serde_json::to_vec(&reference(&"0".repeat(64), MACHINE)).unwrap();
        assert!(parse(&good).is_ok());
        let mut r = reference(&"0".repeat(64), MACHINE);
        r.rtmr3.policy = "report".into();
        assert!(parse(&serde_json::to_vec(&r).unwrap())
            .unwrap_err()
            .contains("rtmr3.policy"));
        let mut r = reference(&"0".repeat(64), MACHINE);
        r.rtmr1.boot_applications_sha384 = vec!["abc".into()];
        assert!(parse(&serde_json::to_vec(&r).unwrap())
            .unwrap_err()
            .contains("96 lowercase hex"));
        let mut r = reference(&"0".repeat(64), MACHINE);
        r.provider = "azure".into();
        assert!(parse(&serde_json::to_vec(&r).unwrap())
            .unwrap_err()
            .contains("provider"));
        let mut r = reference(&"0".repeat(64), MACHINE);
        r.rtmr1.gpt_event_sha384 = "abc".into();
        assert!(parse(&serde_json::to_vec(&r).unwrap())
            .unwrap_err()
            .contains("96 lowercase hex"));
        let mut r = reference(&"0".repeat(64), MACHINE);
        r.rtmr1.gpt_event_sha384 = r.rtmr1.boot_applications_sha384[0].clone();
        assert!(parse(&serde_json::to_vec(&r).unwrap())
            .unwrap_err()
            .contains("also a boot application"));
        // The schema is read first: no schema, another one, or v2 without the pin.
        let mut v = serde_json::to_value(reference(&"0".repeat(64), MACHINE)).unwrap();
        v["schema"] = "x0ne-engine/reference/v3".into();
        assert!(parse(&serde_json::to_vec(&v).unwrap())
            .unwrap_err()
            .contains("expected \"x0ne-engine/reference/v2\""));
        v.as_object_mut().unwrap().remove("schema");
        assert!(parse(&serde_json::to_vec(&v).unwrap())
            .unwrap_err()
            .contains("no \"schema\""));
        let mut v = serde_json::to_value(reference(&"0".repeat(64), MACHINE)).unwrap();
        v["rtmr1"]
            .as_object_mut()
            .unwrap()
            .remove("gpt_event_sha384");
        assert!(parse(&serde_json::to_vec(&v).unwrap())
            .unwrap_err()
            .contains("gpt_event_sha384"));
    }

    /// SECURITY: the installed default must be root's and not writable by
    /// others; an explicit file is the user's responsibility.
    #[cfg(unix)]
    #[test]
    fn the_installed_default_must_belong_to_root() {
        let dir = tempfile::tempdir().unwrap();
        let p = written(dir.path());
        sign_file(&p);
        // Run as root (CI container), the file is root's: test the other gate.
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        if std::fs::metadata(&p).unwrap().uid() == 0 {
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o666)).unwrap();
        }
        let err = load(&p, false, &test_anchor(), NOW).unwrap_err();
        assert_eq!(err.exit_code(), EXIT_UNUSABLE);
        let err = err.to_string();
        assert!(
            err.contains("belong to root") || err.contains("writable"),
            "{err}"
        );
        let explicit = load(&p, true, &test_anchor(), NOW).unwrap();
        assert!(explicit.explicit && explicit.sha256_hex.len() == 64);
    }

    /// The checks hold for the bytes returned: a file renamed over the path
    /// after them, group and world writable, is not the one read.
    #[cfg(unix)]
    #[test]
    fn the_reference_read_is_the_file_that_was_checked() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x0ne-cvm-0.1.0.json");
        std::fs::write(&p, b"checked").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o644)).unwrap();
        let swap = dir.path().join("swap");
        std::fs::write(&swap, b"swapped").unwrap();
        std::fs::set_permissions(&swap, std::fs::Permissions::from_mode(0o666)).unwrap();
        let me = std::fs::metadata(&p).unwrap().uid();
        let got = read_as(&p, false, me, || std::fs::rename(&swap, &p).unwrap()).unwrap();
        assert_eq!(
            String::from_utf8_lossy(&got),
            "checked",
            "the bytes returned are not those of the file the checks passed"
        );
    }

    /// 2026-10-16T01:00:00Z: after the TEST and GnuPG fixture keys were made.
    const NOW: u64 = 1_792_112_400 + 60;

    fn test_anchor() -> crate::archive_key::Anchor {
        crate::archive_key::Anchor::of(crate::openpgp::Cert::parse(
            &crate::archive_key::test_key::keyring(),
        ))
    }

    /// Sign `p` with the TEST archive key and log it in the TEST
    /// transparency log next to it, cosigned at NOW.
    fn sign_file(p: &Path) {
        let bytes = std::fs::read(p).unwrap();
        let sig =
            crate::archive_key::test_key::sign(&bytes, crate::archive_key::test_key::CREATED + 1);
        std::fs::write(signature_path(p), sig).unwrap();
        crate::transparency::test_log::log_reference(p, NOW).unwrap();
    }

    /// [`super::load`] with the TEST transparency log, its state next to `p`.
    fn load(
        p: &Path,
        explicit: bool,
        key: &crate::archive_key::Anchor,
        now: u64,
    ) -> Result<Loaded, LoadError> {
        super::load(p, explicit, key, &test_log(), &state_of(p), now)
    }

    fn test_log() -> crate::transparency::Anchor {
        crate::transparency::Anchor::of(crate::transparency::test_log::policy())
    }

    fn state_of(p: &Path) -> PathBuf {
        p.parent().unwrap().join("state")
    }

    /// The test reference as a file the loaders admit: RTMR0 pinned.
    fn written(dir: &Path) -> PathBuf {
        written_as(
            dir,
            &endorsed_reference(&"0".repeat(64), MACHINE, rtmr0_pins()),
        )
    }

    fn written_as(dir: &Path, r: &Reference) -> PathBuf {
        let p = dir.join("x0ne-cvm-0.1.0.json");
        std::fs::write(&p, serde_json::to_vec(r).unwrap()).unwrap();
        p
    }

    #[test]
    fn a_signed_reference_loads_and_names_its_signer() {
        let dir = tempfile::tempdir().unwrap();
        let p = written(dir.path());
        sign_file(&p);
        let loaded = load(&p, true, &test_anchor(), NOW).unwrap();
        assert_eq!(loaded.reference.image.name, "x0ne-cvm");
        assert_eq!(
            loaded.signed_by().signed_at_unix,
            u64::from(crate::archive_key::test_key::CREATED) + 1
        );
    }

    /// The reference exactly as GnuPG 2.4.7 signed it (tests/archive_key/).
    /// Its body is v1 and no secret key is left to sign a v2 one: the
    /// signature passes, then the schema is refused (89, not 99).
    #[test]
    fn a_gnupg_signed_reference_passes_its_signature_under_its_key() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/archive_key");
        let gnupg = crate::archive_key::Anchor::of(crate::openpgp::Cert::parse(
            &std::fs::read(fixtures.join("test-archive-keyring.gpg")).unwrap(),
        ));
        let bytes = std::fs::read(fixtures.join("reference.json")).unwrap();
        let sig = std::fs::read(fixtures.join("reference.json.sig")).unwrap();
        let signer = gnupg.verify(&bytes, &sig, NOW).unwrap();
        assert_eq!(
            signer.subkey_fingerprint,
            "43B2C8EB81620E1C19492F2C86D2CD081726C748"
        );
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("reference.json");
        std::fs::write(&p, &bytes).unwrap();
        std::fs::write(signature_path(&p), &sig).unwrap();
        crate::transparency::test_log::log_reference(&p, NOW).unwrap();
        let err = load(&p, true, &gnupg, NOW).unwrap_err();
        assert_eq!(err.exit_code(), EXIT_UNUSABLE, "{err}");
        assert!(err.to_string().contains(SCHEMA_V1), "{err}");
        // The same file under the build's TEST key: another key, refused.
        let err = load(&p, true, &test_anchor(), NOW).unwrap_err();
        assert_eq!(err.exit_code(), EXIT_SIGNATURE, "{err}");
    }

    /// SECURITY: unsigned, signed by another key, changed after signing, or
    /// no key in the build: exit 99, and nothing of the file is parsed.
    #[test]
    fn an_unsigned_or_badly_signed_reference_is_refused_with_99() {
        let dir = tempfile::tempdir().unwrap();
        let p = written(dir.path());
        let code = |r: Result<Loaded, LoadError>| r.map(|_| 0).unwrap_or_else(|e| e.exit_code());
        // No signature file.
        let err = load(&p, true, &test_anchor(), NOW).unwrap_err();
        assert_eq!(err.exit_code(), EXIT_SIGNATURE);
        assert!(err.to_string().contains("x0ne-cvm-0.1.0.json.sig"), "{err}");
        // Signed, then one byte changed.
        sign_file(&p);
        let mut bytes = std::fs::read(&p).unwrap();
        let i = bytes.len() / 2;
        bytes[i] ^= 0x20;
        std::fs::write(&p, &bytes).unwrap();
        assert_eq!(code(load(&p, true, &test_anchor(), NOW)), EXIT_SIGNATURE);
        // A signature by another key (GnuPG's rogue TEST key, over the fixture).
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/archive_key");
        let p = written(dir.path());
        std::fs::copy(
            fixtures.join("reference.json.rogue.sig"),
            signature_path(&p),
        )
        .unwrap();
        assert_eq!(code(load(&p, true, &test_anchor(), NOW)), EXIT_SIGNATURE);
        // A good signature, but this build pins no key.
        sign_file(&p);
        let none = crate::archive_key::Anchor::none();
        let err = load(&p, true, &none, NOW).unwrap_err();
        assert_eq!(err.exit_code(), EXIT_SIGNATURE);
        assert!(
            err.to_string().contains("pins no x0ne archive key"),
            "{err}"
        );
        // Signed garbage is signed, then refused as unusable (89).
        std::fs::write(&p, b"not a reference").unwrap();
        sign_file(&p);
        assert_eq!(code(load(&p, true, &test_anchor(), NOW)), EXIT_UNUSABLE);
        // No reference at all: unusable.
        assert_eq!(
            code(load(
                &dir.path().join("none.json"),
                true,
                &test_anchor(),
                NOW
            )),
            EXIT_UNUSABLE
        );
    }

    /// SECURITY: a reference signed by the pinned archive key but not
    /// in the pinned transparency log is refused with 100, before a byte of
    /// it is parsed: no proof, a proof of another reference, or no log
    /// pinned in this build.
    #[test]
    fn a_signed_reference_absent_from_the_log_is_refused_with_100() {
        let dir = tempfile::tempdir().unwrap();
        let p = written(dir.path());
        sign_file(&p);
        let code = |r: Result<Loaded, LoadError>| r.map(|_| 0).unwrap_or_else(|e| e.exit_code());
        assert_eq!(code(load(&p, true, &test_anchor(), NOW)), 0, "logged");
        let proof = std::fs::read(crate::transparency::proof_path(&p)).unwrap();
        std::fs::remove_file(crate::transparency::proof_path(&p)).unwrap();
        let err = load(&p, true, &test_anchor(), NOW).unwrap_err();
        assert_eq!(
            err.exit_code(),
            crate::transparency::EXIT_NOT_LOGGED,
            "{err}"
        );
        assert!(err.to_string().contains("no proof of inclusion"), "{err}");
        // Another signed reference, carrying the first one's proof.
        let other = dir.path().join("x0ne-cvm-0.1.1.json");
        let mut v = serde_json::to_value(reference(&"0".repeat(64), MACHINE)).unwrap();
        v["image"]["version"] = "0.1.1".into();
        std::fs::write(&other, serde_json::to_vec(&v).unwrap()).unwrap();
        let sig = crate::archive_key::test_key::sign(
            &std::fs::read(&other).unwrap(),
            crate::archive_key::test_key::CREATED + 1,
        );
        std::fs::write(signature_path(&other), sig).unwrap();
        std::fs::write(crate::transparency::proof_path(&other), &proof).unwrap();
        let err = load(&other, true, &test_anchor(), NOW).unwrap_err();
        assert_eq!(
            err.exit_code(),
            crate::transparency::EXIT_NOT_LOGGED,
            "{err}"
        );
        assert!(err.to_string().contains("not shown in the log"), "{err}");
        // Signed garbage, not logged: 100, not 89 (the log comes before parse).
        let garbage = dir.path().join("x0ne-cvm-0.1.2.json");
        std::fs::write(&garbage, b"not a reference").unwrap();
        let sig = crate::archive_key::test_key::sign(
            b"not a reference",
            crate::archive_key::test_key::CREATED + 1,
        );
        std::fs::write(signature_path(&garbage), sig).unwrap();
        assert_eq!(
            code(load(&garbage, true, &test_anchor(), NOW)),
            crate::transparency::EXIT_NOT_LOGGED
        );
        // A build that pins no log: every reference, logged or not.
        std::fs::write(crate::transparency::proof_path(&p), &proof).unwrap();
        let none = crate::transparency::Anchor::none();
        let err = super::load(&p, true, &test_anchor(), &none, &state_of(&p), NOW).unwrap_err();
        assert_eq!(
            err.exit_code(),
            crate::transparency::EXIT_NOT_LOGGED,
            "{err}"
        );
        assert!(
            err.to_string().contains("pins no transparency log"),
            "{err}"
        );
        let loaded = load(&p, true, &test_anchor(), NOW).unwrap();
        assert_eq!(
            loaded.logged().origin,
            crate::transparency::test_log::ORIGIN
        );
    }

    /// A build made before the signing ceremony pins no key: every reference
    /// is refused with 99, with or without its .sig, and the reason names
    /// the build, not the reference.
    #[test]
    fn a_build_without_a_pin_refuses_every_reference_and_says_why() {
        let dir = tempfile::tempdir().unwrap();
        let p = written(dir.path());
        let none = crate::archive_key::Anchor::none();
        for case in ["without .sig", "signed by the TEST key"] {
            if case == "signed by the TEST key" {
                sign_file(&p);
            }
            let err = load(&p, true, &none, NOW).unwrap_err();
            assert_eq!(err.exit_code(), EXIT_SIGNATURE, "{case}: {err}");
            let err = err.to_string();
            assert!(
                err.contains("built before the signing ceremony"),
                "{case}: {err}"
            );
            assert!(
                !err.contains("not signed by the x0ne archive key"),
                "{case}: {err}"
            );
        }
    }

    /// SECURITY: signed by a primary alone (x0ne-OS gen-test-key.sh, the
    /// TEST key of `make cvm-sign`), even with that very key pinned: 99.
    #[test]
    fn a_reference_signed_by_a_primary_alone_is_refused_with_99() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/archive_key");
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x0ne-cvm-0.1.0.json");
        std::fs::copy(fixtures.join("reference.json"), &p).unwrap();
        std::fs::copy(
            fixtures.join("reference.json.primary-only.sig"),
            signature_path(&p),
        )
        .unwrap();
        let pinned = crate::archive_key::Anchor::of(crate::openpgp::Cert::parse(
            &std::fs::read(fixtures.join("primary-only-keyring.gpg")).unwrap(),
        ));
        let err = load(&p, true, &pinned, NOW).unwrap_err();
        assert_eq!(err.exit_code(), EXIT_SIGNATURE, "{err}");
        assert!(err.to_string().contains("never by the primary"), "{err}");
    }

    #[test]
    fn the_default_is_the_newest_version() {
        let dir = tempfile::tempdir().unwrap();
        for v in ["0.1.0", "0.10.0", "0.2.1", "notes.txt"] {
            std::fs::write(dir.path().join(format!("x0ne-cvm-{v}.json")), b"{}").unwrap();
        }
        assert!(default_path_in(dir.path())
            .unwrap()
            .ends_with("x0ne-cvm-0.10.0.json"));
        assert!(default_path_in(&dir.path().join("none")).is_err());
    }

    /// x0ne-OS publishes the reference in `cvm/` with its neighbours
    /// (since 2026-10-17): before, the manifest and the prediction of the same
    /// version sorted above `x0ne-cvm-<V>.json`, and the prediction won.
    #[test]
    fn the_default_is_the_reference_never_its_neighbours() {
        let dir = tempfile::tempdir().unwrap();
        let published = [
            ".json",
            ".json.sig",
            ".measurements-predicted.json",
            ".measurements-predicted.json.sig",
            ".manifest.json",
            ".tar.gz",
            ".efi",
            "-sources.tar.xz",
        ];
        for v in ["0.2", "0.2.1"] {
            for f in published {
                std::fs::write(dir.path().join(format!("x0ne-cvm-{v}{f}")), b"{}").unwrap();
            }
        }
        std::fs::write(dir.path().join("SHA256SUMS"), b"").unwrap();
        assert_eq!(
            default_path_in(dir.path()).unwrap(),
            dir.path().join("x0ne-cvm-0.2.1.json")
        );
        for v in ["0.2", "0.2.1"] {
            std::fs::remove_file(dir.path().join(format!("x0ne-cvm-{v}.json"))).unwrap();
        }
        let err = default_path_in(dir.path()).unwrap_err();
        assert!(
            err.contains("no x0ne-cvm-<version>.json reference"),
            "{err}"
        );
    }

    // The GCE a3-highgpu-1g TDX boot of x0ne-cvm 0.2 (tar 48180d78) on
    // 2026-10-17: its CCEL cut 256 bytes into the 0xFF padding, the TD report
    // body of its quote (quote bytes 48..632), the v1 reference published
    // with the image and the v2 one x0ne-OS writes for the same build.
    const LOG_1017: &[u8] = include_bytes!("../tests/ccel/gce-a3-tdx-2026-10-17.bin");
    const TDREPORT_1017: &[u8] = include_bytes!("../tests/ccel/tdreport-gce-a3-tdx-2026-10-17.bin");
    const REFERENCE_1017: &[u8] = include_bytes!("../tests/reference/x0ne-cvm-0.2.json");
    const REFERENCE_1017_V1: &[u8] = include_bytes!("../tests/reference/x0ne-cvm-0.2-v1.json");

    fn quote_1017() -> TdxQuoteClaims {
        let mut quote = vec![0u8; 48];
        quote[0] = 4;
        quote[4] = 0x81;
        quote.extend_from_slice(TDREPORT_1017);
        x0ne_confidential::tdx_quote::parse(&quote).expect("TD report body")
    }

    fn real_1017() -> (Reference, Vec<Event>) {
        (
            parse(REFERENCE_1017).expect("the v2 reference"),
            ccel::parse(LOG_1017).expect("the real log"),
        )
    }

    /// What a TD that measured `events` quotes: genuine hardware, its own log.
    fn quoted(events: &[Event]) -> TdxQuoteClaims {
        let mut q = quote_1017();
        q.rtmr = ccel::replay(events).unwrap();
        q
    }

    #[test]
    fn the_real_boot_of_2026_10_17_passes_its_v2_reference() {
        let (r, log) = real_1017();
        let quote = quote_1017();
        assert_eq!(&ccel::hex(&quote.rtmr[1])[..16], "5631fb8a4dfda234");
        let report = check_image_rtmr0_reported(&r, MACHINE, &quote, &log).expect("the real boot");
        assert_eq!(report.boot_applications, r.rtmr1.boot_applications_sha384);
        assert_eq!(report.sections.len(), r.rtmr2.events_sha384.len());
    }

    /// The relay's path for the image that ran on 2026-10-17: its v2 reference
    /// (d9eacb23), signed by a subkey of the pinned key (TEST here), is a
    /// `report` one, refused by the loader since 104; the same reference
    /// with RTMR0 pinned to the events of that very boot and Google's
    /// endorsed firmware loads as the relay loads it, then passes the boot.
    #[test]
    fn the_signed_v2_reference_of_2026_10_17_loads_pinned_and_passes_its_real_boot() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x0ne-cvm-0.2.json");
        std::fs::write(&p, REFERENCE_1017).unwrap();
        sign_file(&p);
        let err = load(&p, true, &test_anchor(), NOW).map(|_| ()).unwrap_err();
        assert_eq!(err.exit_code(), EXIT_RTMR0_UNPINNED, "{err}");
        let (r, log) = real_1017_pinned();
        let p = dir.path().join("x0ne-cvm-0.2.1.json");
        std::fs::write(&p, serde_json::to_vec(&r).unwrap()).unwrap();
        sign_file(&p);
        let loaded = load(&p, true, &test_anchor(), NOW).expect("signed v2 reference, pinned");
        let fw = firmware_53c558fe(quote_1017().mr_td);
        let report = check_image_with(&loaded.reference, MACHINE, &quote_1017(), &log, &fw)
            .expect("the real boot");
        assert!(report.rtmr0_pinned);
    }

    /// SECURITY: RTMR1 changes to the real log, each replaying to its own
    /// quote. 0.2.0-beta.5 accepted a correctly hashed foreign GPT, no UKI when
    /// the kernel PE was there, and the UKI twice (2026-10-17).
    #[test]
    fn rtmr1_changes_to_the_real_boot_are_refused() {
        let (r, log) = real_1017();
        let rtmr1 = |t: u32| -> Vec<usize> {
            (0..log.len())
                .filter(|&i| log[i].rtmr() == Some(1) && log[i].event_type == t)
                .collect()
        };
        let gpt = rtmr1(ccel::EV_EFI_GPT_EVENT);
        let apps = rtmr1(EV_EFI_BOOT_SERVICES_APPLICATION);
        assert_eq!((gpt.len(), apps.len()), (1, 2));
        let (gpt, uki, kernel) = (gpt[0], apps[0], apps[1]);
        let edit = |f: &dyn Fn(&mut Vec<Event>)| {
            let mut l = log.clone();
            f(&mut l);
            l
        };
        let mut other_disk = log[gpt].data.clone();
        *other_disk.last_mut().unwrap() ^= 1;
        let cases: Vec<(&str, Vec<Event>)> = vec![
            (
                "foreign GPT, correctly hashed",
                edit(&|l| {
                    l[gpt] = Event::new_sha384(
                        2,
                        ccel::EV_EFI_GPT_EVENT,
                        sha384(&other_disk),
                        other_disk.clone(),
                    )
                }),
            ),
            ("GPT twice", edit(&|l| l.insert(gpt + 1, l[gpt].clone()))),
            (
                "no GPT",
                edit(&|l| {
                    l.remove(gpt);
                }),
            ),
            (
                "UKI absent",
                edit(&|l| {
                    l.remove(uki);
                }),
            ),
            ("UKI twice", edit(&|l| l.insert(uki + 1, l[uki].clone()))),
            (
                "kernel PE absent",
                edit(&|l| {
                    l.remove(kernel);
                }),
            ),
            ("applications out of order", edit(&|l| l.swap(uki, kernel))),
            (
                "kernel PE added again",
                edit(&|l| l.insert(kernel + 1, l[kernel].clone())),
            ),
            (
                "foreign application added",
                edit(&|l| {
                    l.insert(
                        kernel + 1,
                        Event::new_sha384(
                            2,
                            EV_EFI_BOOT_SERVICES_APPLICATION,
                            [0xee; 48],
                            b"shim".to_vec(),
                        ),
                    )
                }),
            ),
        ];
        let mut accepted = Vec::new();
        for (name, events) in &cases {
            match check_image_rtmr0_reported(&r, MACHINE, &quoted(events), events) {
                Ok(_) => accepted.push(*name),
                Err(e) => assert!(e.contains("RTMR1"), "{name}: {e}"),
            }
        }
        assert!(accepted.is_empty(), "accepted: {accepted:?}");
    }

    /// SECURITY: the pinned digest counts only as an `EV_EFI_GPT_EVENT` whose
    /// data hashes to it, before the first boot application. A 0.2.0-beta.5 build accepted
    /// the GPT after the kernel PE.
    #[test]
    fn the_gpt_pin_counts_only_as_the_gpt_event_before_the_applications() {
        let (r, log) = real_1017();
        let rtmr1 = |t: u32| -> Vec<usize> {
            (0..log.len())
                .filter(|&i| log[i].rtmr() == Some(1) && log[i].event_type == t)
                .collect()
        };
        let (gpt, apps) = (
            rtmr1(ccel::EV_EFI_GPT_EVENT)[0],
            rtmr1(EV_EFI_BOOT_SERVICES_APPLICATION),
        );
        let (uki, kernel) = (apps[0], apps[1]);
        let pin = log[gpt].sha384().unwrap();
        let edit = |f: &dyn Fn(&mut Vec<Event>)| {
            let mut l = log.clone();
            f(&mut l);
            l
        };
        let move_gpt_to = |at: usize| {
            edit(&|l| {
                let ev = l.remove(gpt);
                l.insert(at, ev);
            })
        };
        let cases: Vec<(&str, Vec<Event>, &str)> = vec![
            (
                "pin as an EFI action, GPT data",
                edit(&|l| {
                    l[gpt] = Event::new_sha384(2, ccel::EV_EFI_ACTION, pin, l[gpt].data.clone())
                }),
                "the reference GPT digest as a EFI action",
            ),
            (
                "pin as a GPT event, other data",
                edit(&|l| {
                    l[gpt] = Event::new_sha384(2, ccel::EV_EFI_GPT_EVENT, pin, b"EFI PART".to_vec())
                }),
                "whose data does not hash to it",
            ),
            // After the removal, `uki` and `kernel` index the event that followed each.
            (
                "GPT after the UKI",
                move_gpt_to(uki),
                "after boot application 1",
            ),
            (
                "GPT after the kernel PE",
                move_gpt_to(kernel),
                "after boot application 2",
            ),
        ];
        for (name, events, want) in &cases {
            match check_image_rtmr0_reported(&r, MACHINE, &quoted(events), events) {
                Ok(_) => panic!("{name}: accepted"),
                Err(e) => assert!(e.contains("RTMR1") && e.contains(want), "{name}: {e}"),
            }
        }
    }

    /// The 2026-10-14 log against a reference made of its own digests: its
    /// RTMR1 order (GPT, UKI, kernel PE) is the one the policy requires.
    #[test]
    fn the_real_boot_of_2026_10_14_has_the_rtmr1_order_of_the_policy() {
        let log = ccel::parse(include_bytes!("../tests/ccel/gce-a3-tdx-2026-10-14.bin")).unwrap();
        let digest = |e: &Event| ccel::hex(&e.sha384().unwrap());
        let rtmr1 = |t: u32| -> Vec<String> {
            log.iter()
                .filter(|e| e.rtmr() == Some(1) && e.event_type == t)
                .map(digest)
                .collect()
        };
        let mut r = reference(&"0".repeat(64), MACHINE);
        let gpt = rtmr1(ccel::EV_EFI_GPT_EVENT);
        assert_eq!(gpt.len(), 1);
        r.rtmr1.gpt_event_sha384 = gpt[0].clone();
        r.rtmr1.boot_applications_sha384 = rtmr1(EV_EFI_BOOT_SERVICES_APPLICATION);
        r.rtmr2.events_sha384 = log
            .iter()
            .enumerate()
            .filter(|(_, e)| e.rtmr() == Some(2) && e.event_type != ccel::EV_NO_ACTION)
            .map(|(n, e)| (format!("event {n}"), digest(e)))
            .collect();
        let report = check_image_rtmr0_reported(&r, MACHINE, &quoted(&log), &log)
            .expect("the 2026-10-14 boot");
        assert_eq!(report.boot_applications.len(), 2);
    }

    /// A reference of the v1 schema is refused by name, saying what to use;
    /// it is never read as a v2 one.
    #[test]
    fn a_v1_reference_is_refused_with_a_clear_message() {
        let err = parse(REFERENCE_1017_V1).unwrap_err();
        assert!(
            err.contains("x0ne-engine/reference/v1") && err.contains(SCHEMA) && err.contains("GPT"),
            "{err}"
        );
    }

    /// `x0ne-engine cloud image import` and `up` read the schema this
    /// verifier reads: a reference it would refuse must not reach a VM.
    #[test]
    fn the_cloud_cli_reads_the_same_schema() {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../x0ne-engine-cloud/src/image_import.rs");
        let Ok(text) = std::fs::read_to_string(&src) else {
            return; // the crate alone, outside the workspace
        };
        assert!(
            text.contains(&format!("pub const REFERENCE_SCHEMA: &str = {SCHEMA:?};")),
            "{} does not read {SCHEMA}",
            src.display()
        );
    }

    // ── RTMR0 pinned to Google's endorsed firmware (`gce-endorsed`) ──────────

    pub(crate) const EV_PLATFORM_CONFIG_FLAGS: u32 = 0x0000_000a;
    pub(crate) const EV_EFI_VARIABLE_DRIVER_CONFIG: u32 = 0x8000_0001;
    pub(crate) const EV_EFI_VARIABLE_BOOT: u32 = 0x8000_0002;
    pub(crate) const EV_EFI_PLATFORM_FIRMWARE_BLOB2: u32 = 0x8000_000a;
    pub(crate) const EV_EFI_HANDOFF_TABLES2: u32 = 0x8000_000b;
    /// An ACPI table the host gives, as the stand-in log has it.
    pub(crate) const ACPI: [u8; 48] = [0x11; 48];

    /// RTMR0 of a GCE TDX boot, in the order the real logs have it: the TD
    /// HOB, the CFV, a Secure Boot variable, a separator, an ACPI table and
    /// a boot variable. Only the CFV comes from the firmware binary.
    pub(crate) fn rtmr0_log(cfv: [u8; 48]) -> Vec<Event> {
        let secure_boot = b"SecureBoot=0, as the GCE boots measured it".to_vec();
        vec![
            Event::new_sha384(
                1,
                EV_EFI_HANDOFF_TABLES2,
                [0x10; 48],
                b"\tTdxTable\0".to_vec(),
            ),
            Event::new_sha384(1, EV_EFI_PLATFORM_FIRMWARE_BLOB2, cfv, b"Fv(test)".to_vec()),
            Event::new_sha384(
                1,
                EV_EFI_VARIABLE_DRIVER_CONFIG,
                ccel::sha384(&secure_boot),
                secure_boot,
            ),
            Event::new_sha384(1, EV_SEPARATOR, ccel::sha384(&[0; 4]), vec![0; 4]),
            Event::new_sha384(1, EV_PLATFORM_CONFIG_FLAGS, ACPI, b"ACPI DATA".to_vec()),
            Event::new_sha384(1, EV_EFI_VARIABLE_BOOT, [0x12; 48], b"BootOrder".to_vec()),
        ]
    }

    /// The reference's RTMR0 list for [`rtmr0_log`].
    pub(crate) fn rtmr0_pins() -> Vec<String> {
        let mut pins: Vec<String> = rtmr0_log([0; 48])
            .iter()
            .map(|e| ccel::hex(&e.sha384().unwrap()))
            .collect();
        pins[1] = FIRMWARE_CFV_SLOT.into();
        pins
    }

    /// [`reference`], with RTMR0 pinned to `pins` for `machine`.
    pub(crate) fn endorsed_reference(
        root_spki_sha256: &str,
        machine: &str,
        pins: Vec<String>,
    ) -> Reference {
        let mut r = reference(root_spki_sha256, machine);
        r.rtmr0 = Rtmr0 {
            policy: RTMR0_GCE_ENDORSED.into(),
            known: BTreeMap::new(),
            events_sha384: BTreeMap::from([(machine.to_string(), pins)]),
        };
        r
    }

    pub(crate) fn endorsed_firmware(cfv: [u8; 48]) -> crate::tdvf::EndorsedFirmware {
        crate::tdvf::EndorsedFirmware {
            sha384_hex: "fe".repeat(48),
            mrtd: [0x5; 48],
            cfv_sha384: cfv,
            tdx_svn: 2,
        }
    }

    /// A whole boot under `gce-endorsed`: [`rtmr0_log`], then [`good_log`]
    /// without its RTMR0 separator (`rtmr0_log` has it, where GCE logs it).
    pub(crate) fn gce_log(cfv: [u8; 48]) -> Vec<Event> {
        let mut log = rtmr0_log(cfv);
        log.extend(good_log().into_iter().filter(|e| e.rtmr() != Some(0)));
        log
    }

    const CFV: [u8; 48] = [0xcf; 48];

    fn endorsed_log() -> Vec<Event> {
        gce_log(CFV)
    }

    #[test]
    fn the_rtmr0_list_of_a_boot_marks_the_firmware_cfv_once() {
        let log = endorsed_log();
        assert_eq!(rtmr0_template(&log, &CFV).unwrap(), rtmr0_pins());
        let err = rtmr0_template(&log, &[0xc0; 48]).unwrap_err();
        assert!(err.contains("not in the log"), "{err}");
        let mut twice = log.clone();
        twice.insert(2, twice[1].clone());
        let err = rtmr0_template(&twice, &CFV).unwrap_err();
        assert!(err.contains("twice"), "{err}");
    }

    #[test]
    fn under_gce_endorsed_the_pinned_rtmr0_passes() {
        let r = endorsed_reference(&"0".repeat(64), MACHINE, rtmr0_pins());
        let log = endorsed_log();
        let fw = endorsed_firmware(CFV);
        let report =
            check_image_with(&r, MACHINE, &claims_for(&log, [0x5; 48]), &log, &fw).unwrap();
        assert!(report.rtmr0_pinned);
        assert_eq!(report.rtmr0_known, Some(true));
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert_eq!(report.sections.len(), 3);
        // EV_NO_ACTION does not extend RTMR0: it is not one of its events.
        let mut noted = log.clone();
        noted.insert(
            0,
            Event::new_sha384(1, ccel::EV_NO_ACTION, [0x77; 48], vec![]),
        );
        check_image_with(&r, MACHINE, &claims_for(&noted, [0x5; 48]), &noted, &fw)
            .expect("EV_NO_ACTION is not an RTMR0 event");
        // Under `report`, the lab's replay of the same boot reports RTMR0 only.
        let report = check_image_rtmr0_reported(
            &reference(&"0".repeat(64), MACHINE),
            MACHINE,
            &claims_for(&log, [0x5; 48]),
            &log,
        )
        .unwrap();
        assert!(!report.rtmr0_pinned);
    }

    /// SECURITY: RTMR0 different. Each change replays to its own quote, so
    /// the policy refuses it, not the replay: a host table, the CFV, an event
    /// missing, one more, two swapped; then a firmware whose CFV is not the
    /// logged one, a quote that the log does not explain, no endorsed
    /// firmware at all, and a machine type the reference does not pin.
    #[test]
    fn every_rtmr0_drift_is_refused() {
        let r = endorsed_reference(&"0".repeat(64), MACHINE, rtmr0_pins());
        let fw = endorsed_firmware(CFV);
        let check = |log: &[Event], fw: &crate::tdvf::EndorsedFirmware| {
            check_image_with(&r, MACHINE, &claims_for(log, [0x5; 48]), log, fw)
        };
        type Edit = Box<dyn Fn(&mut Vec<Event>)>;
        let drifts: Vec<(&str, Edit, &str)> = vec![
            (
                "another ACPI table",
                Box::new(|l| {
                    l[4] = Event::new_sha384(
                        1,
                        EV_PLATFORM_CONFIG_FLAGS,
                        [0x99; 48],
                        b"ACPI DATA".to_vec(),
                    )
                }),
                "RTMR0 event 4",
            ),
            (
                "another CFV",
                Box::new(|l| {
                    l[1] = Event::new_sha384(
                        1,
                        EV_EFI_PLATFORM_FIRMWARE_BLOB2,
                        [0xc0; 48],
                        b"Fv(test)".to_vec(),
                    )
                }),
                "configuration volume",
            ),
            (
                "the TD HOB missing",
                Box::new(|l| {
                    l.remove(0);
                }),
                "RTMR0 event",
            ),
            (
                "one event more",
                Box::new(|l| {
                    l.insert(
                        6,
                        Event::new_sha384(
                            1,
                            EV_EFI_VARIABLE_BOOT,
                            [0x13; 48],
                            b"Boot0001".to_vec(),
                        ),
                    )
                }),
                "one event more",
            ),
            ("two swapped", Box::new(|l| l.swap(4, 5)), "RTMR0 event 4"),
            (
                "the last pinned event missing",
                Box::new(|l| {
                    l.remove(5);
                }),
                "was not measured",
            ),
        ];
        for (name, edit, want) in drifts {
            let mut log = endorsed_log();
            edit(&mut log);
            let err = check(&log, &fw)
                .err()
                .unwrap_or_else(|| panic!("{name}: accepted"));
            assert!(err.contains("RTMR0") && err.contains(want), "{name}: {err}");
        }
        let log = endorsed_log();
        let err = check(&log, &endorsed_firmware([0xc0; 48])).unwrap_err();
        assert!(err.contains("configuration volume"), "{err}");
        let mut claims = claims_for(&log, [0x5; 48]);
        claims.rtmr[0][0] ^= 1;
        assert!(check_image_with(&r, MACHINE, &claims, &log, &fw)
            .unwrap_err()
            .contains("replay of RTMR0"));
        let err = check_image_rtmr0_reported(&r, MACHINE, &claims_for(&log, [0x5; 48]), &log)
            .unwrap_err();
        assert!(err.contains("endorsed firmware"), "{err}");
        let err = check_image_with(&r, "c3-standard-4", &claims_for(&log, [0x5; 48]), &log, &fw)
            .unwrap_err();
        assert!(err.contains("c3-standard-4"), "{err}");
    }

    #[test]
    fn a_gce_endorsed_reference_must_place_the_firmware_once() {
        let ok = endorsed_reference(&"0".repeat(64), MACHINE, rtmr0_pins());
        assert!(parse(&serde_json::to_vec(&ok).unwrap()).is_ok());
        let edit = |f: &dyn Fn(&mut Reference)| {
            let mut r = ok.clone();
            f(&mut r);
            parse(&serde_json::to_vec(&r).unwrap()).unwrap_err()
        };
        let pins = |f: &dyn Fn(&mut Vec<String>)| {
            edit(&|r: &mut Reference| f(r.rtmr0.events_sha384.get_mut(MACHINE).unwrap()))
        };
        assert!(pins(&|p| p[1] = "ab".repeat(48)).contains("firmware:cfv"));
        assert!(pins(&|p| p[2] = FIRMWARE_CFV_SLOT.into()).contains("firmware:cfv"));
        assert!(pins(&|p| p[0] = "AB".repeat(48)).contains("96 lowercase hex"));
        assert!(edit(&|r| r.rtmr0.events_sha384.clear()).contains("rtmr0.events_sha384"));
        assert!(edit(&|r| {
            r.rtmr0.known.insert(MACHINE.into(), vec!["ab".repeat(48)]);
        })
        .contains("rtmr0.known"));
        assert!(edit(&|r| r.rtmr0.policy = "pin".into()).contains("rtmr0.policy"));
        let mut report = reference(&"0".repeat(64), MACHINE);
        report.rtmr0.events_sha384 = ok.rtmr0.events_sha384.clone();
        assert!(parse(&serde_json::to_vec(&report).unwrap())
            .unwrap_err()
            .contains("rtmr0.events_sha384"));
        // A `report` reference serializes as before: no new field appears.
        let v = serde_json::to_value(reference(&"0".repeat(64), MACHINE)).unwrap();
        assert!(v["rtmr0"].get("events_sha384").is_none());
    }

    /// SHA-384 of the CFV of Google's binary 53c558fe… (the one its signed
    /// endorsement of MRTD c1ee9c16… names), computed on 2026-10-22 by this
    /// crate and by `tests/tdvf/tdvf_reference.py`; `tests/gce_firmware_live.rs`
    /// recomputes it from the binary.
    pub(crate) const CFV_53C558FE: &str = "4e1221cf6f7815beec39a994bff99142ed305210e2735b1134fc8ef363923cbdcca91fa71e2e25ee8e689fb7d2c075bf";

    /// The real boot of 2026-10-17 passes once RTMR0 is pinned from its own
    /// log: 16 events, the CFV second. Then RTMR0 different, on the real log,
    /// each change replaying to its own quote: an ACPI table, the Secure Boot
    /// forbidden list (dbx), the TD HOB; and a firmware whose CFV is not the
    /// one logged.
    /// The v2 reference of 2026-10-17 with RTMR0 pinned to that boot's own
    /// events under Google's firmware 53c558fe, and the boot's log.
    pub(crate) fn real_1017_pinned() -> (Reference, Vec<Event>) {
        let (v2, log) = real_1017();
        let cfv = crate::tdvf::tests::unhex48(CFV_53C558FE);
        let pins = rtmr0_template(&log, &cfv).expect("the CFV is in the real log");
        assert_eq!(pins.len(), 16);
        assert_eq!(pins.iter().position(|p| p == FIRMWARE_CFV_SLOT), Some(1));
        let mut r = v2;
        r.rtmr0 = Rtmr0 {
            policy: RTMR0_GCE_ENDORSED.into(),
            known: BTreeMap::new(),
            events_sha384: BTreeMap::from([(MACHINE.to_string(), pins)]),
        };
        (
            parse(&serde_json::to_vec(&r).unwrap()).expect("a valid reference"),
            log,
        )
    }

    /// Google's firmware 53c558fe as `tdvf::verify_endorsed` measures it
    /// (`tests/gce_firmware_live.rs` on the binary), for `mrtd`.
    pub(crate) fn firmware_53c558fe(mrtd: [u8; 48]) -> crate::tdvf::EndorsedFirmware {
        crate::tdvf::EndorsedFirmware {
            sha384_hex: "53c558fe06714203d7428484c46ee7c0b9df748a9789e173421c28d097115b30efb8476304617532488ab40c3122db88".into(),
            mrtd,
            cfv_sha384: crate::tdvf::tests::unhex48(CFV_53C558FE),
            tdx_svn: 2,
        }
    }

    #[test]
    fn the_real_boot_of_2026_10_17_pins_rtmr0_to_googles_endorsed_firmware() {
        let (r, log) = real_1017_pinned();
        let quote = quote_1017();
        let fw = firmware_53c558fe(quote.mr_td);
        let report = check_image_with(&r, MACHINE, &quote, &log, &fw).expect("the real boot");
        assert!(report.rtmr0_pinned);
        assert_eq!(
            report.rtmr0_hex,
            "10944cd1f86a98bb39718a85718e6cf84855b38afce35cc84edff0c660998d0500dfedf998af8484e423949c23c40011"
        );
        let rtmr0: Vec<usize> = (0..log.len())
            .filter(|&i| log[i].rtmr() == Some(0) && log[i].event_type != ccel::EV_NO_ACTION)
            .collect();
        let by_type = |t: u32| {
            rtmr0
                .iter()
                .copied()
                .filter(|&i| log[i].event_type == t)
                .collect::<Vec<_>>()
        };
        let acpi = by_type(EV_PLATFORM_CONFIG_FLAGS);
        let vars = by_type(EV_EFI_VARIABLE_DRIVER_CONFIG);
        let hob = by_type(EV_EFI_HANDOFF_TABLES2);
        assert_eq!((acpi.len(), vars.len(), hob.len()), (3, 5, 1));
        for (name, at) in [
            ("ACPI table", acpi[2]),
            ("dbx", vars[4]),
            ("TD HOB", hob[0]),
        ] {
            let mut l = log.clone();
            let mut d = l[at].sha384().unwrap();
            d[0] ^= 1;
            l[at] = Event::new_sha384(1, l[at].event_type, d, l[at].data.clone());
            let err = check_image_with(&r, MACHINE, &quoted(&l), &l, &fw).unwrap_err();
            assert!(err.contains(&format!("RTMR0 event {at}")), "{name}: {err}");
        }
        let other = crate::tdvf::EndorsedFirmware {
            cfv_sha384: [0x4e; 48],
            ..fw
        };
        let err = check_image_with(&r, MACHINE, &quote, &log, &other).unwrap_err();
        assert!(err.contains("configuration volume"), "{err}");
    }

    // ── RTMR0 under `gce-endorsed`: what a host could try ───────────────────

    /// A boot whose log has no RTMR0 event at all, its quote replaying to
    /// zero, under a reference that pins RTMR0: refused.
    #[test]
    fn an_empty_rtmr0_under_pins_is_refused() {
        let r = endorsed_reference(&"0".repeat(64), MACHINE, rtmr0_pins());
        let fw = endorsed_firmware(CFV);
        let log: Vec<Event> = endorsed_log()
            .into_iter()
            .filter(|e| e.rtmr() != Some(0))
            .collect();
        let err =
            check_image_with(&r, MACHINE, &claims_for(&log, [0x5; 48]), &log, &fw).unwrap_err();
        assert!(err.contains("was not measured"), "{err}");
    }

    /// The endorsed CFV's digest logged where a host table belongs: refused.
    #[test]
    fn the_cfv_digest_cannot_stand_in_for_a_host_event() {
        let r = endorsed_reference(&"0".repeat(64), MACHINE, rtmr0_pins());
        let fw = endorsed_firmware(CFV);
        let mut log = endorsed_log();
        log[4] = Event::new_sha384(1, EV_PLATFORM_CONFIG_FLAGS, CFV, b"ACPI DATA".to_vec());
        let err =
            check_image_with(&r, MACHINE, &claims_for(&log, [0x5; 48]), &log, &fw).unwrap_err();
        assert!(err.contains("RTMR0 event 4"), "{err}");
    }

    /// A pinned host table's digest logged in the CFV's slot: refused.
    #[test]
    fn a_host_event_digest_cannot_stand_in_for_the_cfv() {
        let r = endorsed_reference(&"0".repeat(64), MACHINE, rtmr0_pins());
        let fw = endorsed_firmware(CFV);
        let mut log = endorsed_log();
        log[1] = Event::new_sha384(
            1,
            EV_EFI_PLATFORM_FIRMWARE_BLOB2,
            ACPI,
            b"Fv(test)".to_vec(),
        );
        let err =
            check_image_with(&r, MACHINE, &claims_for(&log, [0x5; 48]), &log, &fw).unwrap_err();
        assert!(err.contains("configuration volume"), "{err}");
    }

    /// An EV_NO_ACTION with a digest in RTMR0 is skipped by the pins and by
    /// the replay alike: the honest quote still explains the log, so nothing
    /// can be extended past the pins under that event type.
    #[test]
    fn no_action_events_change_neither_the_pins_nor_the_replay() {
        let r = endorsed_reference(&"0".repeat(64), MACHINE, rtmr0_pins());
        let fw = endorsed_firmware(CFV);
        let honest = endorsed_log();
        let claims = claims_for(&honest, [0x5; 48]);
        let mut log = honest.clone();
        log.insert(
            3,
            Event::new_sha384(1, ccel::EV_NO_ACTION, [0x42; 48], b"x".to_vec()),
        );
        let report = check_image_with(&r, MACHINE, &claims, &log, &fw).unwrap();
        assert!(report.rtmr0_pinned);
        assert_eq!(report.rtmr0_hex, ccel::hex(&claims.rtmr[0]));
    }

    /// An event logged for MR index 0 (MRTD) or past RTMR3: refused.
    #[test]
    fn an_event_outside_any_rtmr_is_refused() {
        let r = endorsed_reference(&"0".repeat(64), MACHINE, rtmr0_pins());
        let fw = endorsed_firmware(CFV);
        let honest = endorsed_log();
        let claims = claims_for(&honest, [0x5; 48]);
        for index in [0, 7] {
            let mut log = honest.clone();
            log.insert(
                3,
                Event::new_sha384(index, EV_PLATFORM_CONFIG_FLAGS, [0x42; 48], b"x".to_vec()),
            );
            assert!(
                check_image_with(&r, MACHINE, &claims, &log, &fw).is_err(),
                "MR index {index}"
            );
        }
    }

    /// A `gce-endorsed` reference must name the CFV by its slot: the digest
    /// written in its place is refused at load.
    #[test]
    fn a_gce_endorsed_reference_needs_the_slot_even_with_the_literal_cfv() {
        let mut pins = rtmr0_pins();
        pins[1] = ccel::hex(&CFV);
        let r = endorsed_reference(&"0".repeat(64), MACHINE, pins);
        let err = parse(&serde_json::to_vec(&r).unwrap()).unwrap_err();
        assert!(err.contains("firmware:cfv"), "{err}");
    }

    /// The CFV moved out of its slot, or measured twice: refused.
    #[test]
    fn the_endorsed_cfv_out_of_its_slot_or_twice_is_refused() {
        let r = endorsed_reference(&"0".repeat(64), MACHINE, rtmr0_pins());
        let fw = endorsed_firmware(CFV);
        let mut first = endorsed_log();
        first.swap(0, 1);
        let mut twice = endorsed_log();
        twice.insert(2, twice[1].clone());
        let mut elsewhere = endorsed_log();
        elsewhere[1] =
            Event::new_sha384(2, EV_EFI_PLATFORM_FIRMWARE_BLOB2, CFV, b"Fv(test)".to_vec());
        for (name, log) in [("first", first), ("twice", twice), ("in RTMR1", elsewhere)] {
            let err = check_image_with(&r, MACHINE, &claims_for(&log, [0x5; 48]), &log, &fw)
                .err()
                .unwrap_or_else(|| panic!("{name}: accepted"));
            assert!(err.contains("RTMR"), "{name}: {err}");
        }
    }

    /// A TD that booted another CFV and shows the endorsed boot's log: its
    /// quote's RTMR0 is the other firmware's, the replay of the log is not
    /// it, and nothing in the log is believed.
    #[test]
    fn a_quote_whose_rtmr0_is_another_firmware_is_refused_whatever_the_log_says() {
        let r = endorsed_reference(&"0".repeat(64), MACHINE, rtmr0_pins());
        let fw = endorsed_firmware(CFV);
        let shown = endorsed_log();
        let booted = gce_log([0xc0; 48]);
        let claims = claims_for(&booted, [0x5; 48]);
        assert_ne!(claims.rtmr[0], claims_for(&shown, [0x5; 48]).rtmr[0]);
        let err = check_image_with(&r, MACHINE, &claims, &shown, &fw).unwrap_err();
        assert!(err.contains("replay of RTMR0"), "{err}");
        let err = check_image_with(&r, MACHINE, &claims, &booted, &fw).unwrap_err();
        assert!(err.contains("configuration volume"), "{err}");
    }

    /// The three vendored GCE boots give one RTMR0 list and one RTMR0.
    #[test]
    fn the_three_real_logs_share_one_rtmr0_list() {
        let cfv = crate::tdvf::tests::unhex48(CFV_53C558FE);
        let mut lists = Vec::new();
        for day in ["14", "17", "18"] {
            let log = ccel::parse(
                &std::fs::read(format!(
                    "{}/tests/ccel/gce-a3-tdx-2026-10-{day}.bin",
                    env!("CARGO_MANIFEST_DIR")
                ))
                .unwrap(),
            )
            .unwrap();
            let rtmr = ccel::replay(&log).unwrap();
            lists.push((rtmr0_template(&log, &cfv).unwrap(), ccel::hex(&rtmr[0])));
        }
        assert_eq!(lists[0], lists[1]);
        assert_eq!(lists[1], lists[2]);
        assert_eq!(lists[0].0.len(), 16);
        assert_eq!(
            lists[0].1,
            "10944cd1f86a98bb39718a85718e6cf84855b38afce35cc84edff0c660998d0500dfedf998af8484e423949c23c40011"
        );
    }
}
