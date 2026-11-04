//! The model key after proof: a model owner's key, released by the owner's
//! own relay to one attested TD, and to nothing else.
//!
//! The owner seals the weights at home (`x0ne-infer seal`, `seal.rs`) and
//! keeps the 32-byte key there. Nobody else ever holds it in the clear: not
//! x0ne, not the cloud, not the lifecycle that starts the VM. The relay on
//! the owner's machine sends it, wrapped, once its own verdict and the
//! owner's policy both say yes:
//!
//! ```text
//! relay (the owner's machine)                              TD
//!   verdict on the document (remote_verify, steps 1-8)
//!   the owner's policy: image, TCB, GPU, expiry, model ── 102 if not
//!   release nonce N, 32 random bytes
//!   GET /v2/model-key/recipient?nonce=N ──────────────────▶ quote over SHA-512(CONTEXT ‖ N
//!                                                            ‖ kem_pk ‖ session key ‖ TLS key ‖ config)
//!   ◀──────────────────────────────────── {kem_pk, quote}
//!   same TD and platform as the document's quote; report_data
//!   binds N and kem_pk ────────────────────────────────── 88 if not
//!   HPKE to kem_pk: info = the same fields, aad = the grant
//!   POST /v2/model-key {N, enc, grant, ct} ───────────────▶ N issued here, used once; HPKE open
//!                                                            with kem_sk; the grant names this
//!                                                            config's sealed file and is live;
//!                                                            keys::release (same verdict, 300 s);
//!                                                            unseal; served digest = the grant's (77)
//! ```
//!
//! `kem_sk` is born in the TD at boot, lives in locked memory
//! (`harden::SecretBytes`) and is dropped once a key opened: a wrapped key
//! recorded on the way opens nowhere afterwards. The quote names `kem_pk`
//! directly, as Trustee's KBS protocol binds its `tee-pubkey`; the HPKE suite
//! is the one Azure's transparent KMS and Tinfoil's EHBP use.

use crate::binding::BindingV2;
use crate::cloud_config::{hex, unhex32, CloudConfig};
use crate::harden::SecretBytes;
use crate::hpke;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha512};
use zeroize::Zeroizing;

/// The domain of the recipient quote's `report_data`.
pub const CONTEXT: &[u8] = b"x0ne-infer/model-key/v1";
/// The domain of the HPKE `info`: never equal to [`CONTEXT`]'s input.
pub const INFO_DOMAIN: &[u8] = b"x0ne-infer/model-key/v1/hpke\0";
/// `GET /v2/model-key/recipient`.
pub const RECIPIENT_SCHEMA: &str = "x0ne-infer/model-key-recipient/v1";
/// `POST /v2/model-key`.
pub const WRAPPED_SCHEMA: &str = "x0ne-infer/model-key/v1";
/// The grant: what the key is released for, authenticated by the HPKE.
pub const GRANT_SCHEMA: &str = "x0ne-infer/model-key-grant/v1";
/// The owner's release policy file.
pub const POLICY_SCHEMA: &str = "x0ne-engine/model-key-policy/v1";
/// The key id `keys::release` gives the model key.
pub const KEY_ID: &str = "model-key";
/// The relay's exit when the OWNER's policy refuses a TD the verdict accepted.
/// Neither `x0ne-infer` nor the lifecycle uses it; 100 is the relay's refusal
/// of an unlogged reference (`transparency.rs`), 101 a Rust panic.
pub const EXIT_POLICY: i32 = 102;
/// What the sealed weights are called in the document's file list.
pub const SEALED_FILE: &str = "model.sealed";
/// A wrapped key, its grant included, is a few hundred bytes.
pub const MAX_BODY_BYTES: usize = 16 * 1024;
/// TDX platform statuses a policy may accept: the verifier refuses the others.
pub const TCB_VALUES: &[&str] = &["UpToDate", "SWHardeningNeeded"];

// ── The binding ──────────────────────────────────────────────────────────────

/// What the recipient quote's `report_data` covers, and the HPKE `info`.
/// Every field is 32 bytes: no framing to get wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecipientBinding {
    /// The relay's, fresh for this release.
    pub release_nonce: [u8; 32],
    /// The TD's X25519 key for this boot.
    pub kem_pk: [u8; 32],
    pub session_pubkey: [u8; 32],
    pub tls_spki_sha256: [u8; 32],
    pub config_sha256: [u8; 32],
}

impl RecipientBinding {
    /// The session's binding (the document's), for one release and one key.
    pub fn of(session: &BindingV2, release_nonce: [u8; 32], kem_pk: [u8; 32]) -> Self {
        Self {
            release_nonce,
            kem_pk,
            session_pubkey: session.session_pubkey,
            tls_spki_sha256: session.tls_spki_sha256,
            config_sha256: session.config_sha256,
        }
    }

    fn fields(&self) -> [&[u8; 32]; 5] {
        [
            &self.release_nonce,
            &self.kem_pk,
            &self.session_pubkey,
            &self.tls_spki_sha256,
            &self.config_sha256,
        ]
    }

    /// `SHA-512("x0ne-infer/model-key/v1" ‖ N ‖ kem_pk ‖ session key ‖ SHA-256(SPKI) ‖ SHA-256(config))`.
    pub fn tdx_report_data(&self) -> [u8; 64] {
        let mut h = Sha512::new();
        h.update(CONTEXT);
        for f in self.fields() {
            h.update(f);
        }
        h.finalize().into()
    }

    pub fn matches(&self, report_data: &[u8; 64]) -> bool {
        &self.tdx_report_data() == report_data
    }

    /// `INFO_DOMAIN ‖` the same fields.
    pub fn hpke_info(&self) -> Vec<u8> {
        let mut info = INFO_DOMAIN.to_vec();
        for f in self.fields() {
            info.extend_from_slice(f);
        }
        info
    }
}

// ── The wire ─────────────────────────────────────────────────────────────────

/// `GET /v2/model-key/recipient?nonce=<64 hex>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipientDoc {
    pub schema: String,
    pub suite: String,
    pub release_nonce_hex: String,
    pub kem_pk_hex: String,
    pub quote_b64: String,
}

/// `POST /v2/model-key`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Wrapped {
    pub schema: String,
    pub release_nonce_hex: String,
    pub enc_hex: String,
    /// The grant's exact bytes: the HPKE's `aad`.
    pub grant_b64: String,
    pub ct_hex: String,
}

/// A [`Wrapped`] body, parsed and shaped.
pub struct WrappedParts {
    pub release_nonce: [u8; 32],
    pub enc: [u8; 32],
    pub grant_bytes: Vec<u8>,
    pub ct: Vec<u8>,
}

fn unhex(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("not hex".into());
    }
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl Wrapped {
    /// The TD side: the parts, or why not.
    pub fn parse(body: &[u8]) -> Result<WrappedParts, String> {
        if body.len() > MAX_BODY_BYTES {
            return Err("wrapped key too large".into());
        }
        let w: Wrapped = serde_json::from_slice(body).map_err(|e| format!("wrapped key: {e}"))?;
        if w.schema != WRAPPED_SCHEMA {
            return Err(format!(
                "wrapped key schema {:?}, expected {WRAPPED_SCHEMA:?}",
                w.schema
            ));
        }
        let ct = unhex(&w.ct_hex).map_err(|e| format!("ct_hex: {e}"))?;
        Ok(WrappedParts {
            release_nonce: unhex32(&w.release_nonce_hex)
                .map_err(|e| format!("release_nonce_hex: {e}"))?,
            enc: unhex32(&w.enc_hex).map_err(|e| format!("enc_hex: {e}"))?,
            grant_bytes: crate::attest_v2::unb64(&w.grant_b64)
                .map_err(|e| format!("grant_b64: {e}"))?,
            ct,
        })
    }
}

/// What the key is released for. Its bytes are the HPKE `aad`: changed in
/// transit, the key does not open.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Grant {
    pub schema: String,
    /// The catalog model whose config.json and tokenizer.json the weights use.
    pub model: String,
    /// SHA-256 of the sealed file the TD holds.
    pub sealed_sha256: String,
    /// The served digest (v2) the unsealed model must have: every receipt carries it.
    pub digest_hex: String,
    /// The TD serves no request once its own wall clock, which the host
    /// sets, passes this instant.
    pub not_after_unix: u64,
    /// SHA-256 of the owner's policy file that released the key.
    pub policy_sha256: String,
}

impl Grant {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let g: Grant = serde_json::from_slice(bytes).map_err(|e| format!("grant: {e}"))?;
        if g.schema != GRANT_SCHEMA {
            return Err(format!(
                "grant schema {:?}, expected {GRANT_SCHEMA:?}",
                g.schema
            ));
        }
        Ok(g)
    }

    /// The TD's judgement: the grant is for THIS session's sealed file, and live.
    pub fn check(&self, config: &CloudConfig, now_unix: u64) -> Result<(), String> {
        if self.schema != GRANT_SCHEMA {
            return Err(format!(
                "grant schema {:?}, expected {GRANT_SCHEMA:?}",
                self.schema
            ));
        }
        let Some(sealed) = &config.sealed else {
            return Err(
                "this session's config names no sealed weights: no model key is taken".into(),
            );
        };
        if self.model != config.model {
            return Err(format!(
                "the grant is for base model {:?}, this session's is {:?}",
                self.model, config.model
            ));
        }
        if self.sealed_sha256 != sealed.sealed_sha256 {
            return Err(format!(
                "the grant is for sealed file {}, this session holds {}",
                self.sealed_sha256, sealed.sealed_sha256
            ));
        }
        if self.digest_hex != sealed.digest_hex {
            return Err(format!(
                "the grant's served digest {} is not this session's {}",
                self.digest_hex, sealed.digest_hex
            ));
        }
        if now_unix >= self.not_after_unix {
            return Err(format!(
                "the grant expired at {} (unix)",
                self.not_after_unix
            ));
        }
        Ok(())
    }
}

// ── The TD's half ────────────────────────────────────────────────────────────

/// The X25519 key the recipient quote names. The private half is in locked
/// memory and never leaves this value.
pub struct Recipient {
    sk: SecretBytes,
    pk: [u8; 32],
}

/// What a wrapped key opened to.
pub struct Opened {
    pub grant: Grant,
    pub key: Zeroizing<Vec<u8>>,
}

impl Recipient {
    /// `ikm`: 32 fresh random bytes (`/dev/urandom` in the TD).
    pub fn generate(ikm: &[u8; 32]) -> Result<Self, String> {
        let (sk, pk) = hpke::derive_key_pair(ikm);
        let sk =
            SecretBytes::from_slice(&sk[..]).map_err(|e| format!("recipient key memory: {e}"))?;
        Ok(Self { sk, pk })
    }

    pub fn public_key(&self) -> [u8; 32] {
        self.pk
    }

    /// Open a wrapped key sent for `binding` (the nonce this TD issued, its
    /// key, its session). Another TD's key, another release, another session
    /// or a changed grant: refused, never a wrong key.
    pub fn open(&self, binding: &RecipientBinding, w: &WrappedParts) -> Result<Opened, String> {
        if binding.kem_pk != self.pk {
            return Err("the binding names another recipient key".into());
        }
        if binding.release_nonce != w.release_nonce {
            return Err("the wrapped key is for another release".into());
        }
        let sk: &[u8; 32] = self
            .sk
            .expose()
            .try_into()
            .map_err(|_| "recipient key is not 32 bytes".to_string())?;
        let key = hpke::open(sk, &w.enc, &binding.hpke_info(), &w.grant_bytes, &w.ct)
            .map_err(|e| e.to_string())?;
        if key.len() != 32 {
            return Err(format!("the model key is {} bytes, expected 32", key.len()));
        }
        // Authenticated by the AEAD: parsed only once it opened.
        let grant = Grant::parse(&w.grant_bytes)?;
        Ok(Opened { grant, key })
    }
}

// ── The owner's half ─────────────────────────────────────────────────────────

/// What `rtmr0` of an [`ImagePin`] says to require the RTMR0 the verdict
/// pinned to Google's endorsed firmware (the reference's `gce-endorsed`).
pub const RTMR0_GCE_ENDORSED: &str = crate::reference::RTMR0_GCE_ENDORSED;

/// One image the key may go to: MRTD, RTMR1 and RTMR2, 96 hex each, and
/// RTMR0 if the owner pins it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImagePin {
    pub mrtd: String,
    pub rtmr1: String,
    pub rtmr2: String,
    /// Absent: RTMR0 is the reference's business (the relay admits a
    /// reference that pins it, `gce-endorsed`, and no other). 96 hex: this
    /// value exactly. [`RTMR0_GCE_ENDORSED`]: the verdict must have pinned
    /// RTMR0 to the firmware Google endorsed, whatever its value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rtmr0: Option<String>,
}

/// The owner's release policy (`x0ne-engine/model-key-policy/v1`). It pins
/// the image itself, beside the x0ne reference the verdict already checks:
/// whoever holds the x0ne archive key cannot have the model key released to
/// an image the owner did not name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleasePolicy {
    pub schema: String,
    /// The catalog model whose config.json and tokenizer.json the weights use.
    pub model: String,
    /// SHA-256 of the sealed file.
    pub sealed_sha256: String,
    /// SHA-256 of the plaintext `model.safetensors`.
    pub weights_sha256: String,
    /// The key goes to a TD running one of these, and no other.
    pub images: Vec<ImagePin>,
    /// The platform TCB statuses accepted (a subset of [`TCB_VALUES`]).
    pub tcb: Vec<String>,
    /// The key goes only to a TD whose GPU is attested in CC mode.
    pub require_gpu: bool,
    /// No release after this instant by the relay's clock; the TD stops
    /// serving once its own clock, which the host sets, passes it.
    pub not_after_unix: u64,
    /// Where the TD fetches the sealed file (`cloud_config::SealedSource`):
    /// copied into the session config, so it is in the quote. Absent: the
    /// file must reach the VM by other means (`--sealed-weights`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sealed_source: Option<crate::cloud_config::SealedSource>,
}

/// What the verdict established, as the policy reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observed {
    pub mrtd: [u8; 48],
    pub rtmr0: [u8; 48],
    /// The verdict pinned RTMR0 to Google's endorsed firmware
    /// (`Report::rtmr0_pinned`), not merely reported it.
    pub rtmr0_pinned: bool,
    pub rtmr1: [u8; 48],
    pub rtmr2: [u8; 48],
    pub tcb_platform: String,
    pub gpu_count: usize,
    pub model: String,
    pub sealed_sha256: Option<String>,
}

impl Observed {
    /// From a document that passed `remote_verify::verify_document` and its report.
    pub fn from_verified(
        doc: &crate::attest_v2::Document,
        report: &crate::remote_verify::Report,
        config: &CloudConfig,
    ) -> Result<Self, String> {
        let quote = crate::attest_v2::unb64(&doc.tdx.quote_b64)?;
        let claims =
            x0ne_confidential::tdx_quote::parse(&quote).map_err(|e| format!("TDX quote: {e:?}"))?;
        Ok(Self {
            mrtd: claims.mr_td,
            rtmr0: claims.rtmr[0],
            rtmr0_pinned: report.rtmr0_pinned,
            rtmr1: claims.rtmr[1],
            rtmr2: claims.rtmr[2],
            tcb_platform: report.tcb_platform.clone(),
            gpu_count: report.gpu_count,
            model: config.model.clone(),
            sealed_sha256: config.sealed.as_ref().map(|s| s.sealed_sha256.clone()),
        })
    }
}

impl ReleasePolicy {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_BODY_BYTES * 4 {
            return Err("policy file too large".into());
        }
        let p: ReleasePolicy =
            serde_json::from_slice(bytes).map_err(|e| format!("model key policy: {e}"))?;
        if p.schema != POLICY_SCHEMA {
            return Err(format!("schema {:?}, expected {POLICY_SCHEMA:?}", p.schema));
        }
        crate::registry::Catalog::embedded()
            .get(&p.model)
            .map_err(|e| format!("model: {e}"))?;
        for (name, v) in [
            ("sealed_sha256", &p.sealed_sha256),
            ("weights_sha256", &p.weights_sha256),
        ] {
            if !is_hex(v, 64) {
                return Err(format!("{name}: 64 lowercase hex characters expected"));
            }
        }
        if p.images.is_empty() {
            return Err("images: empty; the key would go to no image".into());
        }
        for (i, img) in p.images.iter().enumerate() {
            for (name, v) in [
                ("mrtd", &img.mrtd),
                ("rtmr1", &img.rtmr1),
                ("rtmr2", &img.rtmr2),
            ] {
                if !is_hex(v, 96) {
                    return Err(format!(
                        "images[{i}].{name}: 96 lowercase hex characters expected"
                    ));
                }
            }
            if let Some(r) = &img.rtmr0 {
                if r != RTMR0_GCE_ENDORSED && !is_hex(r, 96) {
                    return Err(format!(
                        "images[{i}].rtmr0: 96 lowercase hex characters or {RTMR0_GCE_ENDORSED:?} expected"
                    ));
                }
            }
        }
        if p.tcb.is_empty() {
            return Err("tcb: empty; the key would go to no platform".into());
        }
        if let Some(t) = p.tcb.iter().find(|t| !TCB_VALUES.contains(&t.as_str())) {
            return Err(format!(
                "tcb: {t:?} is not one of {TCB_VALUES:?}, the statuses the verdict accepts"
            ));
        }
        if p.not_after_unix == 0 {
            return Err("not_after_unix is 0: a release policy always ends".into());
        }
        if let Some(src) = &p.sealed_source {
            crate::registry::check_source_url(&src.url)
                .map_err(|e| format!("sealed_source.url: {e}"))?;
            if src.size == 0 || src.size > crate::fsread::WEIGHTS_CAP {
                return Err(format!(
                    "sealed_source.size {} outside 1..={}",
                    src.size,
                    crate::fsread::WEIGHTS_CAP
                ));
            }
        }
        Ok(p)
    }

    /// The served digest (v2) of the unsealed model: the catalog's
    /// config.json and tokenizer.json, and the plaintext weights.
    pub fn served_digest(&self) -> Result<String, String> {
        let catalog = crate::registry::Catalog::embedded();
        let entry = catalog.get(&self.model)?;
        let get = |name: &str| {
            entry
                .files
                .iter()
                .find(|f| f.path == name)
                .map(|f| f.sha256.as_str())
                .ok_or_else(|| format!("{}: no {name} in the catalog", entry.name))
        };
        crate::model_digest::from_hashes([
            ("config.json", get("config.json")?),
            ("tokenizer.json", get("tokenizer.json")?),
            (crate::shards::SINGLE_FILE, self.weights_sha256.as_str()),
        ])
    }

    /// The owner's yes or no, after the verdict's yes. Every no is [`EXIT_POLICY`].
    pub fn judge(
        &self,
        seen: &Observed,
        now_unix: u64,
    ) -> Result<(), crate::remote_verify::Refusal> {
        let no = |reason: String| crate::remote_verify::Refusal {
            code: EXIT_POLICY,
            reason: format!("the model owner's key policy: {reason}"),
        };
        if now_unix >= self.not_after_unix {
            return Err(no(format!("expired at {} (unix)", self.not_after_unix)));
        }
        if seen.model != self.model {
            return Err(no(format!(
                "the session's base model is {:?}, the policy's is {:?}",
                seen.model, self.model
            )));
        }
        if seen.sealed_sha256.as_deref() != Some(self.sealed_sha256.as_str()) {
            return Err(no(format!(
                "the session's sealed file is {:?}, the policy's is {}",
                seen.sealed_sha256, self.sealed_sha256
            )));
        }
        let (mrtd, rtmr1, rtmr2) = (hex(&seen.mrtd), hex(&seen.rtmr1), hex(&seen.rtmr2));
        let rtmr0 = hex(&seen.rtmr0);
        let rtmr0_as_pinned = |i: &ImagePin| match i.rtmr0.as_deref() {
            None => true,
            Some(RTMR0_GCE_ENDORSED) => seen.rtmr0_pinned,
            Some(value) => value == rtmr0,
        };
        if !self
            .images
            .iter()
            .any(|i| i.mrtd == mrtd && i.rtmr1 == rtmr1 && i.rtmr2 == rtmr2 && rtmr0_as_pinned(i))
        {
            return Err(no(format!(
                "the TD's image (MRTD {mrtd}, RTMR1 {rtmr1}, RTMR2 {rtmr2}, RTMR0 {rtmr0}{}) is \
                 not one the policy names",
                if seen.rtmr0_pinned {
                    " pinned to the endorsed firmware"
                } else {
                    " not pinned"
                }
            )));
        }
        if !self.tcb.contains(&seen.tcb_platform) {
            return Err(no(format!(
                "platform TCB {:?} is not one of {:?}",
                seen.tcb_platform, self.tcb
            )));
        }
        if self.require_gpu && seen.gpu_count == 0 {
            return Err(no(
                "no GPU attested in CC mode, and the policy requires one".into(),
            ));
        }
        Ok(())
    }
}

/// The model key and the policy it is released under, on the owner's machine.
pub struct OwnerKey {
    key: Zeroizing<[u8; 32]>,
    pub policy: ReleasePolicy,
    pub policy_sha256: [u8; 32],
}

impl std::fmt::Debug for OwnerKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnerKey")
            .field("key", &"<redacted>")
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

impl OwnerKey {
    /// `policy_bytes`: the file's exact bytes (their SHA-256 goes in the grant).
    pub fn new(key: Zeroizing<[u8; 32]>, policy_bytes: &[u8]) -> Result<Self, String> {
        let policy = ReleasePolicy::parse(policy_bytes)?;
        Ok(Self {
            key,
            policy,
            policy_sha256: crate::cloud_config::sha256(policy_bytes),
        })
    }

    /// The key file (64 hex characters, a final newline allowed, readable by
    /// its owner only) and the policy file.
    pub fn load(key_file: &std::path::Path, policy_file: &std::path::Path) -> Result<Self, String> {
        use std::io::Read as _;
        let shown = key_file.display();
        let (f, len) =
            crate::fsread::open_regular(key_file).map_err(|e| format!("{shown}: {e}"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = f
                .metadata()
                .map_err(|e| format!("{shown}: {e}"))?
                .permissions()
                .mode();
            if mode & 0o077 != 0 {
                return Err(format!(
                    "{shown}: mode {:04o}, others can read the model key; chmod 0600",
                    mode & 0o7777
                ));
            }
        }
        if len > 65 {
            return Err(format!("{shown}: a model key file is 64 hex characters"));
        }
        let mut text = Zeroizing::new(String::new());
        f.take(66)
            .read_to_string(&mut text)
            .map_err(|e| format!("{shown}: {e}"))?;
        let key = unhex32(text.strip_suffix('\n').unwrap_or(&text))
            .map(Zeroizing::new)
            .map_err(|e| format!("{shown}: a model key file is 64 hex characters ({e})"))?;
        let policy = crate::fsread::read_capped(policy_file, (MAX_BODY_BYTES * 4) as u64)
            .map_err(|e| format!("{}: {e}", policy_file.display()))?;
        Self::new(key, &policy).map_err(|e| format!("{}: {e}", policy_file.display()))
    }

    /// The grant this policy gives.
    pub fn grant(&self) -> Result<Grant, String> {
        Ok(Grant {
            schema: GRANT_SCHEMA.into(),
            model: self.policy.model.clone(),
            sealed_sha256: self.policy.sealed_sha256.clone(),
            digest_hex: self.policy.served_digest()?,
            not_after_unix: self.policy.not_after_unix,
            policy_sha256: hex(&self.policy_sha256),
        })
    }

    /// Wrap the key for the TD `binding` names; `ikm_e`: 32 fresh random bytes.
    pub fn wrap(
        &self,
        binding: &RecipientBinding,
        grant: &Grant,
        ikm_e: &[u8; 32],
    ) -> Result<Wrapped, String> {
        let grant_bytes = serde_json::to_vec(grant).map_err(|e| e.to_string())?;
        let (enc, ct) = hpke::seal(
            &binding.kem_pk,
            &binding.hpke_info(),
            &grant_bytes,
            &self.key[..],
            ikm_e,
        )
        .map_err(|e| e.to_string())?;
        Ok(Wrapped {
            schema: WRAPPED_SCHEMA.into(),
            release_nonce_hex: hex(&binding.release_nonce),
            enc_hex: hex(&enc),
            grant_b64: crate::attest_v2::b64(&grant_bytes),
            ct_hex: hex(&ct),
        })
    }
}

/// The document's file list for a sealed model: the catalog's files but the
/// weights, and the sealed file. Sorted by path.
pub fn sealed_files(entry: &crate::registry::Entry, sealed_sha256: &str) -> Vec<(String, String)> {
    let mut files: Vec<(String, String)> = entry
        .files
        .iter()
        .filter(|f| !is_weight_file(&f.path))
        .map(|f| (f.path.clone(), f.sha256.clone()))
        .collect();
    files.push((SEALED_FILE.into(), sealed_sha256.into()));
    files.sort();
    files
}

/// The TD's check of the sealed file it holds, before any key: an `X0NESEAL`
/// envelope whose SHA-256 is the one the config names.
pub fn check_sealed_file(
    bytes: &[u8],
    sealed: &crate::cloud_config::SealedWeights,
) -> Result<(), String> {
    if !crate::seal::is_sealed(bytes) {
        return Err("the sealed file is not an X0NESEAL envelope".into());
    }
    let got = hex(&crate::cloud_config::sha256(bytes));
    if got != sealed.sealed_sha256 {
        return Err(format!(
            "the sealed file's SHA-256 is {got}, the config names {}",
            sealed.sealed_sha256
        ));
    }
    Ok(())
}

/// The owner's sealed file from the config's source, after the attestation:
/// the catalog's transfer code (ranges, resumes, the hash of the whole), the
/// config's size and SHA-256 as the pin, kept under `<store_root>/sealed` for
/// a restart. 96 the source is refused (`registry::check_source_url`) or
/// failed, 77 other bytes than the config names or a file that is not an
/// envelope (nothing kept), 71 unreadable once fetched. A transfer cut short
/// keeps `partial/<sha256>.part` (and `.ranges`) for the next attempt: only
/// bytes the source sent, at their offsets; the key and the unsealed weights
/// never reach a file here (unsealed in memory after the release).
pub async fn fetch_sealed(
    store_root: &std::path::Path,
    src: &crate::cloud_config::SealedSource,
    sealed: &crate::cloud_config::SealedWeights,
    name: &str,
) -> Result<Vec<u8>, (i32, String)> {
    use crate::registry;
    registry::check_source_url(&src.url).map_err(|e| (96, e))?;
    let store = registry::Store::new(store_root.join("sealed"));
    let source = registry::Source::new(&src.url).map_err(|e| (96, e))?;
    let spec = registry::FileSpec {
        path: SEALED_FILE.into(),
        size: src.size,
        sha256: sealed.sealed_sha256.clone(),
    };
    let path = registry::fetch_file(&store, &source, name, &src.url, &spec, &mut registry::Quiet)
        .await
        .map_err(|e| match e {
            registry::PullError::Integrity(m) => (77, format!("sealed weights: {m}")),
            other => (96, format!("fetch of the sealed weights: {other}")),
        })?;
    let bytes = crate::fsread::read_capped(&path, crate::fsread::cap_for(SEALED_FILE))
        .map_err(|e| (71, format!("{}: {e}", path.display())))?;
    if let Err(e) = check_sealed_file(&bytes, sealed) {
        // The config pinned a file that is not an envelope: maybe plaintext weights.
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(store.partial_ranges(&sealed.sealed_sha256));
        return Err((77, format!("sealed weights: {e}; the file was deleted")));
    }
    Ok(bytes)
}

/// The key-release lease, ended at the owner's grant expiry if that comes
/// first. Both are judged on the TD's wall clock, which the host sets.
pub fn bound_lease(
    lease: &x0ne_confidential::lease::CapabilityLease,
    not_after_unix: u64,
) -> x0ne_confidential::lease::CapabilityLease {
    let mut bounded = lease.clone();
    bounded.expires_at_unix = bounded.expires_at_unix.min(not_after_unix);
    bounded
}

/// The base model without its weights: what a TD serving sealed weights
/// pulls from the catalog (config.json, tokenizer.json, the licence). No
/// digest pin: the served digest is the owner's.
pub fn base_entry(entry: &crate::registry::Entry) -> crate::registry::Entry {
    let mut base = entry.clone();
    base.files.retain(|f| !is_weight_file(&f.path));
    base.digest_sha256 = None;
    base
}

fn is_weight_file(path: &str) -> bool {
    path == crate::shards::SINGLE_FILE
        || path == crate::shards::INDEX_FILE
        || (path.starts_with("model-") && path.ends_with(".safetensors"))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::binding::SessionIdentity;

    pub(crate) const MODEL: &str = "qwen3:0.6b";

    fn session() -> BindingV2 {
        let id = SessionIdentity::generate([7u8; 32], [9u8; 32]);
        BindingV2::of(&id, [0x11; 32], [0x22; 32])
    }

    fn h(n: u8, len: usize) -> String {
        format!("{n:02x}").repeat(len / 2)
    }

    pub(crate) fn policy_json(model: &str, sealed: &str, weights: &str, not_after: u64) -> String {
        serde_json::json!({
            "schema": POLICY_SCHEMA,
            "model": model,
            "sealed_sha256": sealed,
            "weights_sha256": weights,
            "images": [{"mrtd": h(0xa1, 96), "rtmr1": h(0xb1, 96), "rtmr2": h(0xc1, 96)}],
            "tcb": ["UpToDate", "SWHardeningNeeded"],
            "require_gpu": true,
            "not_after_unix": not_after,
        })
        .to_string()
    }

    fn owner(not_after: u64) -> OwnerKey {
        OwnerKey::new(
            Zeroizing::new([0x5a; 32]),
            policy_json(MODEL, &h(0x33, 64), &h(0x44, 64), not_after).as_bytes(),
        )
        .unwrap()
    }

    fn observed() -> Observed {
        Observed {
            mrtd: [0xa1; 48],
            rtmr0: [0xd1; 48],
            rtmr0_pinned: true,
            rtmr1: [0xb1; 48],
            rtmr2: [0xc1; 48],
            tcb_platform: "UpToDate".into(),
            gpu_count: 1,
            model: MODEL.into(),
            sealed_sha256: Some(h(0x33, 64)),
        }
    }

    fn config(sealed: Option<&str>) -> CloudConfig {
        let mut c = CloudConfig::new(MODEL, [1; 32], [2; 32], 4_000_000_000, "cpu+gpu", 443, 64);
        c.sealed = sealed.map(|s| crate::cloud_config::SealedWeights {
            sealed_sha256: s.into(),
            digest_hex: owner(2_000).policy.served_digest().unwrap(),
            source: None,
        });
        c
    }

    fn parts(w: &Wrapped) -> WrappedParts {
        Wrapped::parse(serde_json::to_string(w).unwrap().as_bytes()).unwrap()
    }

    /// A policy may say where the TD fetches the sealed file; the URL is
    /// judged as a catalog endpoint is (https, or loopback http), the size
    /// as a weight file, and an unknown field inside is refused.
    #[test]
    fn a_policy_sealed_source_is_judged_when_the_policy_is_read() {
        let with = |source: serde_json::Value| {
            let mut v: serde_json::Value = serde_json::from_str(&policy_json(
                MODEL,
                &h(0x33, 64),
                &h(0x44, 64),
                4_000_000_000,
            ))
            .unwrap();
            v["sealed_source"] = source;
            ReleasePolicy::parse(v.to_string().as_bytes())
        };
        let ok = with(serde_json::json!({"url": "https://w.example/m/model.sealed", "size": 42}))
            .unwrap();
        assert_eq!(
            ok.sealed_source,
            Some(crate::cloud_config::SealedSource {
                url: "https://w.example/m/model.sealed".into(),
                size: 42
            })
        );
        assert!(
            with(serde_json::json!({"url": "http://127.0.0.1:1/model.sealed", "size": 42})).is_ok()
        );
        for (source, why) in [
            (
                serde_json::json!({"url": "http://w.example/model.sealed", "size": 42}),
                "https",
            ),
            (
                serde_json::json!({"url": "https://u:p@w.example/model.sealed", "size": 42}),
                "userinfo",
            ),
            (
                serde_json::json!({"url": "https://169.254.169.254/model.sealed", "size": 42}),
                "link-local",
            ),
            (
                serde_json::json!({"url": "https://instance-data/model.sealed", "size": 42}),
                "metadata",
            ),
            (
                serde_json::json!({"url": "https://w.example/model.sealed", "size": 0}),
                "size",
            ),
            (
                serde_json::json!({"url": "https://w.example/model.sealed", "size": 42, "token": "x"}),
                "unknown field",
            ),
        ] {
            let err = with(source.clone()).expect_err(&source.to_string());
            assert!(err.contains(why), "{source}: {err}");
            assert!(!err.contains("u:p@"), "{source}: {err}");
        }
        let o = OwnerKey::new(
            Zeroizing::new([0x5a; 32]),
            policy_json(MODEL, &h(0x33, 64), &h(0x44, 64), 4_000_000_000).as_bytes(),
        )
        .unwrap();
        assert_eq!(
            o.policy.sealed_source, None,
            "a policy without a source still reads"
        );
    }

    /// The recipient quote binds every field, and its domain is its own: no
    /// v2 or v3 `report_data` of the same session equals it.
    #[test]
    fn the_recipient_binding_binds_every_field_under_its_own_domain() {
        let s = session();
        let b = RecipientBinding::of(&s, [0x55; 32], [0x66; 32]);
        let rd = b.tdx_report_data();
        assert!(b.matches(&rd));
        assert_ne!(rd, s.tdx_report_data());
        assert_ne!(
            rd,
            crate::binding::BindingV3::of(&s, [0x55; 32]).tdx_report_data()
        );
        for i in 0..5 {
            let mut o = b;
            match i {
                0 => o.release_nonce[0] ^= 1,
                1 => o.kem_pk[0] ^= 1,
                2 => o.session_pubkey[0] ^= 1,
                3 => o.tls_spki_sha256[0] ^= 1,
                _ => o.config_sha256[0] ^= 1,
            }
            assert!(!o.matches(&rd), "field {i} substituted");
            assert_ne!(o.hpke_info(), b.hpke_info(), "field {i} in info");
        }
        assert!(b.hpke_info().starts_with(INFO_DOMAIN));
    }

    /// The key the owner wraps opens in the recipient it was wrapped for,
    /// with the grant the owner signed into the `aad`.
    #[test]
    fn a_wrapped_key_opens_in_its_recipient_with_its_grant() {
        let o = owner(4_000_000_000);
        let r = Recipient::generate(&[3u8; 32]).unwrap();
        let b = RecipientBinding::of(&session(), [0x55; 32], r.public_key());
        let grant = o.grant().unwrap();
        let w = o.wrap(&b, &grant, &[4u8; 32]).unwrap();
        assert_eq!(w.schema, WRAPPED_SCHEMA);
        let opened = r.open(&b, &parts(&w)).unwrap();
        assert_eq!(&opened.key[..], &[0x5a; 32]);
        assert_eq!(opened.grant, grant);
        assert_eq!(grant.digest_hex, o.policy.served_digest().unwrap());
        assert_eq!(grant.policy_sha256, hex(&o.policy_sha256));
    }

    /// SECURITY (key meant for another VM): the same wrapped key, presented
    /// to another TD, to the same TD under another release or session, or
    /// with a changed grant, does not open.
    #[test]
    fn a_wrapped_key_opens_nowhere_else() {
        let o = owner(4_000_000_000);
        let r = Recipient::generate(&[3u8; 32]).unwrap();
        let other = Recipient::generate(&[8u8; 32]).unwrap();
        let b = RecipientBinding::of(&session(), [0x55; 32], r.public_key());
        let w = o.wrap(&b, &o.grant().unwrap(), &[4u8; 32]).unwrap();
        let p = parts(&w);
        let other_b = RecipientBinding {
            kem_pk: other.public_key(),
            ..b
        };
        assert!(other.open(&other_b, &p).is_err(), "another TD");
        assert!(
            other.open(&b, &p).is_err(),
            "another TD, the first's binding"
        );
        for i in 0..4 {
            let mut o2 = b;
            match i {
                0 => o2.release_nonce[0] ^= 1,
                1 => o2.session_pubkey[0] ^= 1,
                2 => o2.tls_spki_sha256[0] ^= 1,
                _ => o2.config_sha256[0] ^= 1,
            }
            assert!(r.open(&o2, &p).is_err(), "binding field {i}");
        }
        let mut changed = o.grant().unwrap();
        changed.not_after_unix += 1;
        let mut p2 = parts(&w);
        p2.grant_bytes = serde_json::to_vec(&changed).unwrap();
        assert!(r.open(&b, &p2).is_err(), "grant changed in transit");
    }

    /// The TD takes a grant for its own sealed file only, and only while live.
    #[test]
    fn the_grant_must_name_this_sessions_sealed_file_and_be_live() {
        let g = owner(2_000).grant().unwrap();
        let sealed = h(0x33, 64);
        g.check(&config(Some(&sealed)), 1_000).unwrap();
        let err = |c: &CloudConfig, now| g.check(c, now).unwrap_err();
        assert!(err(&config(None), 1_000).contains("sealed"));
        assert!(err(&config(Some(&h(0x34, 64))), 1_000).contains("sealed"));
        assert!(err(&config(Some(&sealed)), 2_000).contains("expired"));
        let mut other_model = config(Some(&sealed));
        other_model.model = "qwen3:1.7b".into();
        assert!(err(&other_model, 1_000).contains("model"));
        let mut other_digest = config(Some(&sealed));
        other_digest.sealed.as_mut().unwrap().digest_hex = h(0x45, 64);
        assert!(err(&other_digest, 1_000).contains("digest"));
        let mut bad = g.clone();
        bad.schema = "x0ne-infer/model-key-grant/v0".into();
        assert!(bad.check(&config(Some(&sealed)), 1_000).is_err());
    }

    /// The policy says yes to the image, TCB, GPU and model it names, until
    /// its expiry; every other answer is [`EXIT_POLICY`], with what differed.
    #[test]
    fn the_policy_refuses_any_other_image_tcb_gpu_model_or_time() {
        let p = owner(2_000).policy;
        p.judge(&observed(), 1_000).unwrap();
        let cases: Vec<(&str, Observed, u64, &str)> = vec![
            (
                "mrtd",
                Observed {
                    mrtd: [0xa2; 48],
                    ..observed()
                },
                1_000,
                "image",
            ),
            (
                "rtmr1",
                Observed {
                    rtmr1: [0xb2; 48],
                    ..observed()
                },
                1_000,
                "image",
            ),
            (
                "rtmr2",
                Observed {
                    rtmr2: [0xc2; 48],
                    ..observed()
                },
                1_000,
                "image",
            ),
            (
                "tcb",
                Observed {
                    tcb_platform: "OutOfDate".into(),
                    ..observed()
                },
                1_000,
                "TCB",
            ),
            (
                "gpu",
                Observed {
                    gpu_count: 0,
                    ..observed()
                },
                1_000,
                "GPU",
            ),
            (
                "model",
                Observed {
                    model: "qwen3:1.7b".into(),
                    ..observed()
                },
                1_000,
                "model",
            ),
            (
                "sealed",
                Observed {
                    sealed_sha256: Some(h(0x34, 64)),
                    ..observed()
                },
                1_000,
                "sealed",
            ),
            (
                "public",
                Observed {
                    sealed_sha256: None,
                    ..observed()
                },
                1_000,
                "sealed",
            ),
            ("expired", observed(), 2_000, "expired"),
        ];
        for (name, seen, now, says) in cases {
            let r = p.judge(&seen, now).unwrap_err();
            assert_eq!(r.code, EXIT_POLICY, "{name}: {r}");
            assert!(r.reason.contains(says), "{name}: {r}");
        }
    }

    /// A policy file is read strictly: unknown fields, a bad hex, no image,
    /// a TCB status the verifier refuses anyway, a model the catalog does not
    /// serve in cloud mode, no expiry: refused by name.
    #[test]
    fn a_policy_file_is_read_strictly() {
        let good = policy_json(MODEL, &h(0x33, 64), &h(0x44, 64), 2_000);
        ReleasePolicy::parse(good.as_bytes()).unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&good).unwrap();
        let edits: Vec<(&str, Box<dyn Fn(&mut serde_json::Value)>)> = vec![
            (
                "schema",
                Box::new(|v| v["schema"] = "x0ne-engine/model-key-policy/v0".into()),
            ),
            ("extra", Box::new(|v| v["extra"] = 1.into())),
            (
                "sealed_sha256",
                Box::new(|v| v["sealed_sha256"] = "33".into()),
            ),
            (
                "weights_sha256",
                Box::new(|v| v["weights_sha256"] = h(0xab, 64).to_uppercase().into()),
            ),
            ("images", Box::new(|v| v["images"] = serde_json::json!([]))),
            (
                "rtmr2",
                Box::new(|v| v["images"][0]["rtmr2"] = h(0xc1, 94).into()),
            ),
            (
                "rtmr0",
                Box::new(|v| v["images"][0]["rtmr0"] = h(0xd1, 94).into()),
            ),
            (
                "rtmr0",
                Box::new(|v| v["images"][0]["rtmr0"] = "report".into()),
            ),
            (
                "tcb",
                Box::new(|v| v["tcb"] = serde_json::json!(["OutOfDate"])),
            ),
            ("tcb", Box::new(|v| v["tcb"] = serde_json::json!([]))),
            ("model", Box::new(|v| v["model"] = "nope:1b".into())),
            (
                "not_after_unix",
                Box::new(|v| v["not_after_unix"] = 0.into()),
            ),
        ];
        for (says, edit) in edits {
            let mut bad = v.clone();
            edit(&mut bad);
            let err = ReleasePolicy::parse(bad.to_string().as_bytes()).unwrap_err();
            assert!(err.contains(says) || says == "extra", "{says}: {err}");
        }
        v["require_gpu"] = false.into();
        ReleasePolicy::parse(v.to_string().as_bytes()).unwrap();
        // RTMR0 is optional: a value, or Google's endorsed firmware.
        v["images"][0]["rtmr0"] = h(0xd1, 96).into();
        ReleasePolicy::parse(v.to_string().as_bytes()).unwrap();
        v["images"][0]["rtmr0"] = "gce-endorsed".into();
        ReleasePolicy::parse(v.to_string().as_bytes()).unwrap();
    }

    /// The owner can pin RTMR0 too: to a value, or to "whatever the verdict
    /// pinned to Google's endorsed firmware" (`gce-endorsed`). Without it,
    /// RTMR0 is the reference's business alone.
    #[test]
    fn the_policy_can_pin_rtmr0_by_value_or_to_googles_endorsed_firmware() {
        let with = |rtmr0: Option<&str>| {
            let mut v: serde_json::Value =
                serde_json::from_str(&policy_json(MODEL, &h(0x33, 64), &h(0x44, 64), 2_000))
                    .unwrap();
            if let Some(r) = rtmr0 {
                v["images"][0]["rtmr0"] = r.into();
            }
            ReleasePolicy::parse(v.to_string().as_bytes()).unwrap()
        };
        let seen = |rtmr0: u8, pinned: bool| Observed {
            rtmr0: [rtmr0; 48],
            rtmr0_pinned: pinned,
            ..observed()
        };
        let refused = |p: &ReleasePolicy, o: &Observed| {
            let r = p.judge(o, 1_000).unwrap_err();
            assert_eq!(r.code, EXIT_POLICY);
            assert!(r.reason.contains("RTMR0"), "{}", r.reason);
        };
        // No rtmr0 in the policy: any RTMR0, pinned or not.
        let none = with(None);
        none.judge(&seen(0xd1, true), 1_000).unwrap();
        none.judge(&seen(0xd2, false), 1_000).unwrap();
        // A value: that value, whether or not the reference pinned it.
        let value = with(Some(&h(0xd1, 96)));
        value.judge(&seen(0xd1, true), 1_000).unwrap();
        value.judge(&seen(0xd1, false), 1_000).unwrap();
        refused(&value, &seen(0xd2, true));
        // gce-endorsed: the verdict must have pinned RTMR0 to the endorsed firmware.
        let endorsed = with(Some("gce-endorsed"));
        endorsed.judge(&seen(0xd1, true), 1_000).unwrap();
        endorsed.judge(&seen(0xd2, true), 1_000).unwrap();
        refused(&endorsed, &seen(0xd1, false));
        // The policy's own image match still holds with rtmr0 set.
        refused(
            &value,
            &Observed {
                rtmr1: [0xb2; 48],
                ..seen(0xd1, true)
            },
        );
    }

    /// The key file stays the owner's: 64 hex, readable by its owner only.
    #[cfg(unix)]
    #[test]
    fn the_key_file_is_64_hex_and_the_owners_alone() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let policy = dir.path().join("policy.json");
        std::fs::write(
            &policy,
            policy_json(MODEL, &h(0x33, 64), &h(0x44, 64), 2_000),
        )
        .unwrap();
        let key = dir.path().join("model.key");
        let write = |text: &str, mode: u32| {
            std::fs::write(&key, text).unwrap();
            std::fs::set_permissions(&key, std::fs::Permissions::from_mode(mode)).unwrap();
        };
        write(&format!("{}\n", h(0x5a, 64)), 0o600);
        let o = OwnerKey::load(&key, &policy).unwrap();
        assert_eq!(&o.key[..], &[0x5a; 32]);
        write(&h(0x5a, 64), 0o640);
        assert!(OwnerKey::load(&key, &policy).unwrap_err().contains("0600"));
        write(&h(0x5a, 62), 0o600);
        assert!(OwnerKey::load(&key, &policy)
            .unwrap_err()
            .contains("64 hex"));
    }

    /// The grant's expiry ends the serving lease when it comes first, and
    /// never extends it.
    #[test]
    fn the_grant_expiry_ends_the_serving_lease_never_extends_it() {
        let lease = crate::keys::test_released_key(vec![0; 32]).lease().clone();
        let sooner = lease.expires_at_unix - 100;
        let bounded = bound_lease(&lease, sooner);
        assert_eq!(bounded.expires_at_unix, sooner);
        assert!(!bounded.is_live(sooner));
        assert!(bounded.is_live(sooner - 1));
        assert_eq!(
            bound_lease(&lease, lease.expires_at_unix + 10_000).expires_at_unix,
            lease.expires_at_unix
        );
        assert_eq!(bounded.issued_at_unix, lease.issued_at_unix);
    }

    /// The behaviour as built, measured 2026-10-24: the TD ends the grant on
    /// its own wall clock, which the host sets, and the relay never compares
    /// that clock with its own. A TD clock held D behind takes a grant sent
    /// just before `not_after_unix` and serves D past it, within the lease's
    /// 24 h of monotonic time from the release. A fix changes this test.
    #[test]
    fn a_td_clock_held_back_serves_past_the_grant_end_by_as_much() {
        use std::time::{Duration, Instant};
        use x0ne_confidential::lease::{CapabilityLease, LeaseProvenance};
        let o = owner(2_000_000_000);
        let not_after = o.policy.not_after_unix;
        let sealed = h(0x33, 64);
        // The relay's clock is the true time here: it releases before the end.
        let release = not_after - 600;
        o.policy.judge(&observed(), release).unwrap();
        assert!(o.policy.judge(&observed(), not_after).is_err());
        let g = o.grant().unwrap();
        // Whether a TD whose clock reads t - d at the true time t serves at t.
        let served = |d: u64, t: u64| -> Result<(), String> {
            let td = |t: u64| t - d;
            g.check(&config(Some(&sealed)), td(release))?;
            let anchor = Instant::now();
            let lease = CapabilityLease::new(
                KEY_ID,
                LeaseProvenance::Verification,
                td(release),
                None,
                None,
            );
            let s = crate::serving_lease::ServingLease::new(
                bound_lease(&lease, g.not_after_unix),
                anchor,
            );
            s.check_at(td(t), anchor + Duration::from_secs(t - release))
                .map_err(|e| e.0)
        };
        served(0, not_after - 1).unwrap();
        served(0, not_after).unwrap_err();
        const D: u64 = 3_600;
        served(D, not_after + D - 1).unwrap();
        served(D, not_after + D).unwrap_err();
        const DAYS: u64 = 3 * 86_400;
        served(DAYS, release + 86_400 - 1).unwrap();
        served(DAYS, release + 86_400).unwrap_err();
    }

    /// A TD serving sealed weights pulls the base model's other files only.
    #[test]
    fn the_base_entry_has_no_weights_and_no_pin() {
        let catalog = crate::registry::Catalog::embedded();
        let entry = catalog.get(MODEL).unwrap();
        let base = base_entry(entry);
        assert_eq!(base.name, entry.name);
        assert_eq!(base.revision, entry.revision);
        assert!(base.weight_files().next().is_none());
        assert!(base.files.iter().any(|f| f.path == "config.json"));
        assert!(base.files.iter().any(|f| f.path == "tokenizer.json"));
        assert_eq!(base.digest_sha256, None);
        assert_eq!(
            base.files.len() + entry.weight_files().count(),
            entry.files.len()
        );
    }

    /// The TD takes only the sealed file the config names, and only sealed.
    #[test]
    fn the_td_takes_only_the_sealed_file_the_config_names() {
        let sealed = crate::seal::seal(b"weights", &[1; 32], &[2; 12]).unwrap();
        let cfg = crate::cloud_config::SealedWeights {
            sealed_sha256: hex(&crate::cloud_config::sha256(&sealed)),
            digest_hex: h(0x44, 64),
            source: None,
        };
        check_sealed_file(&sealed, &cfg).unwrap();
        let mut flipped = sealed.clone();
        flipped[20] ^= 1;
        assert!(check_sealed_file(&flipped, &cfg)
            .unwrap_err()
            .contains("SHA-256"));
        let plain = b"weights".to_vec();
        let cfg_plain = crate::cloud_config::SealedWeights {
            sealed_sha256: hex(&crate::cloud_config::sha256(&plain)),
            ..cfg
        };
        assert!(check_sealed_file(&plain, &cfg_plain)
            .unwrap_err()
            .contains("X0NESEAL"));
    }

    /// The document lists a sealed model's files as the catalog's, but the
    /// weights, plus the sealed file.
    #[test]
    fn a_sealed_models_files_are_the_catalogs_without_weights_plus_the_sealed_file() {
        let catalog = crate::registry::Catalog::embedded();
        let entry = catalog.get(MODEL).unwrap();
        let files = sealed_files(entry, &h(0x33, 64));
        assert!(files
            .iter()
            .any(|(p, s)| p == SEALED_FILE && *s == h(0x33, 64)));
        assert!(!files.iter().any(|(p, _)| is_weight_file(p)));
        assert!(files.iter().any(|(p, _)| p == "config.json"));
        let mut sorted = files.clone();
        sorted.sort();
        assert_eq!(files, sorted);
    }

    /// A wrapped key that opens to anything but 32 bytes is refused, never
    /// handed to the unsealing.
    #[test]
    fn a_wrapped_key_of_another_length_is_refused() {
        let o = owner(4_000_000_000);
        let r = Recipient::generate(&[3u8; 32]).unwrap();
        let b = RecipientBinding::of(&session(), [0x55; 32], r.public_key());
        let grant_bytes = serde_json::to_vec(&o.grant().unwrap()).unwrap();
        for len in [0, 16, 31, 33, 64] {
            let (enc, ct) = hpke::seal(
                &b.kem_pk,
                &b.hpke_info(),
                &grant_bytes,
                &vec![0x5a; len],
                &[4; 32],
            )
            .unwrap();
            let p = WrappedParts {
                release_nonce: b.release_nonce,
                enc,
                grant_bytes: grant_bytes.clone(),
                ct,
            };
            let err = r
                .open(&b, &p)
                .err()
                .unwrap_or_else(|| panic!("{len} bytes opened"));
            assert!(err.contains("expected 32"), "{len}: {err}");
        }
    }
}
