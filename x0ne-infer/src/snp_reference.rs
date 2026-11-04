//! The written SEV-SNP policy a client pins, `x0ne-engine/snp-reference/v1`:
//! the launch digests the image may have, the platform floor, the guest
//! policy and the signing keys accepted. Signed by the x0ne archive key and
//! shown in the pinned transparency log exactly as the TDX image reference
//! (`reference::load_signed`), parsed strictly (unknown fields refused,
//! every judgement field required).
//!
//! What it pins, and how it is judged:
//!
//! - `launch.measurements_sha384`: the report's `MEASUREMENT` must be one of
//!   them. They are predictions computed outside this crate (`sev-snp-measure`
//!   over the OVMF, kernel, initrd and command line, per vCPU count and
//!   type): a verifier only compares;
//! - `launch.host_data`: if present, `HOST_DATA` must equal it; if absent,
//!   it is not judged and the verdict says so;
//! - `products`: the CPU lines accepted; Milan only if listed;
//! - `tcb_minimum`: the floor of every TCB the report carries, component by
//!   component, raised over the bulletins' default (`snp::default_minimum`),
//!   never lowered;
//! - `guest_policy.smt`: `forbidden` or `allowed`;
//! - `signing_keys.vlek_csp_ids`: the cloud providers whose VLEK is
//!   accepted; empty, VCEK only.
//!
//! DEBUG, a migration agent and a VMPL other than 0 are refused by the
//! verifier whatever the reference says: no field can allow them.

use crate::snp::{SnpPolicy, SnpProduct, SnpSmt, SnpTcbMinimum, VerifiedSnp};
use serde::{Deserialize, Serialize};
#[cfg(not(x0ne_sdk))]
use std::path::{Path, PathBuf};

pub const SCHEMA: &str = "x0ne-engine/snp-reference/v1";
pub const TEE: &str = "amd-sev-snp";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnpReference {
    pub schema: String,
    /// Who runs the host, as the relay names the session's provider.
    pub provider: String,
    pub tee: String,
    pub image: Image,
    pub launch: Launch,
    pub products: Vec<String>,
    pub tcb_minimum: TcbMinimum,
    pub guest_policy: GuestPolicy,
    pub signing_keys: SigningKeys,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Image {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub source: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Launch {
    /// Lowercase hex SHA-384, 96 characters each, at least one.
    pub measurements_sha384: Vec<String>,
    /// Lowercase hex, 64 characters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_data: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TcbMinimum {
    pub boot_loader: u8,
    pub tee: u8,
    pub snp: u8,
    pub microcode: u8,
    /// Turin only; Milan and Genoa report 0 there.
    pub fmc: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuestPolicy {
    pub smt: SnpSmt,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SigningKeys {
    pub vlek_csp_ids: Vec<String>,
}

fn is_lower_hex(s: &str, n: usize) -> bool {
    s.len() == n
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

fn unhex<const N: usize>(s: &str) -> Option<[u8; N]> {
    if s.len() != 2 * N {
        return None;
    }
    let mut out = [0u8; N];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

pub fn parse(bytes: &[u8]) -> Result<SnpReference, String> {
    #[derive(Deserialize)]
    struct Head {
        schema: Option<serde_json::Value>,
    }
    let head: Head =
        serde_json::from_slice(bytes).map_err(|e| format!("not an SEV-SNP reference: {e}"))?;
    match head.schema.as_ref().and_then(|s| s.as_str()) {
        Some(SCHEMA) => {}
        Some(other) => return Err(format!("schema {other:?}, expected {SCHEMA:?}")),
        None => return Err(format!("no \"schema\" string, expected {SCHEMA:?}")),
    }
    let r: SnpReference =
        serde_json::from_slice(bytes).map_err(|e| format!("not an SEV-SNP reference: {e}"))?;
    r.validate()?;
    Ok(r)
}

impl SnpReference {
    /// Shapes and names: a value this verifier could not enforce is
    /// refused here rather than approximated later.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != SCHEMA {
            return Err(format!("schema {:?}, expected {SCHEMA:?}", self.schema));
        }
        if self.tee != TEE {
            return Err(format!(
                "tee {:?}: this reference schema is for {TEE:?}",
                self.tee
            ));
        }
        if self.provider.is_empty()
            || !self
                .provider
                .bytes()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
        {
            return Err(format!(
                "provider {:?}: lowercase letters, digits and dashes",
                self.provider
            ));
        }
        let floor = crate::reference::dotted_version(crate::reference::MIN_IMAGE_VERSION)
            .ok_or("MIN_IMAGE_VERSION is not a dotted version")?;
        match crate::reference::dotted_version(&self.image.version) {
            Some(v) if v >= floor => {}
            Some(_) => {
                return Err(format!(
                    "image version {} is below {}, the oldest this build accepts",
                    self.image.version,
                    crate::reference::MIN_IMAGE_VERSION
                ))
            }
            None => {
                return Err(format!(
                    "image version {:?}: digits and dots",
                    self.image.version
                ))
            }
        }
        let m = &self.launch.measurements_sha384;
        if m.is_empty() {
            return Err("launch.measurements_sha384 is empty: no image could match".into());
        }
        for (i, v) in m.iter().enumerate() {
            if !is_lower_hex(v, 96) {
                return Err(format!(
                    "launch.measurements_sha384[{i}]: 96 lowercase hex characters"
                ));
            }
            if m[..i].contains(v) {
                return Err(format!("launch.measurements_sha384[{i}] is listed twice"));
            }
        }
        if let Some(h) = &self.launch.host_data {
            if !is_lower_hex(h, 64) {
                return Err("launch.host_data: 64 lowercase hex characters".into());
            }
        }
        if self.products.is_empty() {
            return Err("products is empty: no CPU could match".into());
        }
        for (i, p) in self.products.iter().enumerate() {
            if SnpProduct::from_name(p).is_none() {
                return Err(format!(
                    "products[{i}] {p:?}: one of \"Milan\", \"Genoa\", \"Turin\""
                ));
            }
            if self.products[..i].contains(p) {
                return Err(format!("products[{i}] {p:?} is listed twice"));
            }
        }
        let ids = &self.signing_keys.vlek_csp_ids;
        for (i, id) in ids.iter().enumerate() {
            if id.is_empty() || !id.is_ascii() {
                return Err(format!(
                    "signing_keys.vlek_csp_ids[{i}]: a non-empty ASCII cspId"
                ));
            }
            if ids[..i].contains(id) {
                return Err(format!("signing_keys.vlek_csp_ids[{i}] is listed twice"));
            }
        }
        Ok(())
    }

    pub fn products(&self) -> Vec<SnpProduct> {
        self.products
            .iter()
            .filter_map(|p| SnpProduct::from_name(p))
            .collect()
    }

    /// The verifier's policy this reference writes: its products (Milan
    /// only if listed), its floor over the bulletins' default, its SMT
    /// rule and its providers.
    pub fn policy(&self) -> SnpPolicy {
        let products = self.products();
        let t = self.tcb_minimum;
        SnpPolicy {
            allow_milan: products.contains(&SnpProduct::Milan),
            products: Some(products),
            minimum: None,
            floor: Some(SnpTcbMinimum {
                fmc: t.fmc,
                boot_loader: t.boot_loader,
                tee: t.tee,
                snp: t.snp,
                microcode: t.microcode,
                require_alias_check: false,
            }),
            smt: self.guest_policy.smt,
            vlek_csp_ids: self.signing_keys.vlek_csp_ids.clone(),
            // The reference writes no mode: the verifier sets its own.
            tcb_mode: crate::tcb::TcbMode::default(),
        }
    }

    /// [`Self::policy`] under the verifier's TCB mode: which of AMD's
    /// bulletins the default minimum imposes.
    pub fn policy_with(&self, mode: crate::tcb::TcbMode) -> SnpPolicy {
        SnpPolicy {
            tcb_mode: mode,
            ..self.policy()
        }
    }

    /// The report's launch digest among the predictions, and its
    /// `HOST_DATA` the pinned one if any. Returns the lines the verdict
    /// says about the launch.
    pub fn check_launch(&self, v: &VerifiedSnp) -> Result<Vec<String>, String> {
        let got = crate::cloud_config::hex(&v.report.measurement);
        if !self
            .launch
            .measurements_sha384
            .iter()
            .any(|m| unhex::<48>(m) == Some(v.report.measurement))
        {
            return Err(format!(
                "launch digest {got} is not among the {} the reference predicts for {} {}: \
                 genuine SEV-SNP, but not this image",
                self.launch.measurements_sha384.len(),
                self.image.name,
                self.image.version
            ));
        }
        let mut said = vec![format!(
            "launch digest {}... is one the reference predicts",
            &got[..16]
        )];
        match &self.launch.host_data {
            Some(h) if unhex::<32>(h) == Some(v.report.host_data) => {
                said.push("HOST_DATA is the pinned one".into())
            }
            Some(h) => {
                return Err(format!(
                    "HOST_DATA {} is not the pinned {h}",
                    crate::cloud_config::hex(&v.report.host_data)
                ))
            }
            None => said.push("HOST_DATA is not pinned by the reference: not judged".into()),
        }
        Ok(said)
    }
}

/// A reference admitted by [`load`].
#[cfg(not(x0ne_sdk))]
#[derive(Debug, Clone)]
pub struct Loaded {
    pub reference: SnpReference,
    pub path: PathBuf,
    pub sha256_hex: String,
    pub explicit: bool,
    pub(crate) signed_by: crate::openpgp::Signer,
    pub(crate) logged: crate::transparency::Logged,
}

#[cfg(not(x0ne_sdk))]
impl Loaded {
    pub fn signed_by(&self) -> &crate::openpgp::Signer {
        &self.signed_by
    }

    pub fn logged(&self) -> &crate::transparency::Logged {
        &self.logged
    }
}

/// As `reference::load`: the signature under the pinned archive key, the
/// pinned transparency log, then the bytes parsed.
#[cfg(not(x0ne_sdk))]
pub fn load(
    path: &Path,
    explicit: bool,
    key: &crate::archive_key::Anchor,
    log: &crate::transparency::Anchor,
    state_dir: &Path,
    now_unix: u64,
) -> Result<Loaded, crate::reference::LoadError> {
    from_signed(
        crate::reference::load_signed(path, explicit, key, log, state_dir, now_unix)?,
        path,
        explicit,
    )
}

/// The bytes `reference::load_signed` admitted, parsed as an SEV-SNP reference.
#[cfg(not(x0ne_sdk))]
pub(crate) fn from_signed(
    s: crate::reference::SignedBytes,
    path: &Path,
    explicit: bool,
) -> Result<Loaded, crate::reference::LoadError> {
    let reference = parse(&s.bytes)
        .map_err(|e| crate::reference::LoadError::Unusable(format!("{}: {e}", path.display())))?;
    Ok(Loaded {
        reference,
        path: path.to_path_buf(),
        sha256_hex: s.sha256_hex,
        explicit,
        signed_by: s.signed_by,
        logged: s.logged,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::snp::tests as t;

    /// A reference for the TEST Genoa guest of `snp::tests`: its launch
    /// digest (0x4D…) among two predictions, its TCB as the floor.
    pub(crate) fn reference() -> SnpReference {
        SnpReference {
            schema: SCHEMA.into(),
            provider: "scaleway-em".into(),
            tee: TEE.into(),
            image: Image {
                name: "x0ne-cvm-snp".into(),
                version: "0.1.0".into(),
                source: serde_json::Value::Null,
            },
            launch: Launch {
                measurements_sha384: vec!["4c".repeat(48), "4d".repeat(48)],
                host_data: None,
            },
            products: vec!["Genoa".into(), "Turin".into()],
            tcb_minimum: TcbMinimum {
                boot_loader: 0x0A,
                tee: 0,
                snp: 0x1B,
                microcode: 0x58,
                fmc: 0,
            },
            guest_policy: GuestPolicy {
                smt: SnpSmt::Forbidden,
            },
            signing_keys: SigningKeys {
                vlek_csp_ids: vec![],
            },
        }
    }

    fn json(r: &SnpReference) -> Vec<u8> {
        serde_json::to_vec(r).unwrap()
    }

    fn verified(payload: &[u8], r: &SnpReference) -> Result<VerifiedSnp, crate::snp::SnpRejected> {
        crate::snp::verify_snp_report(
            payload,
            t::test_crl(payload),
            &t::test_anchors(),
            &r.policy(),
            t::NOW,
        )
    }

    #[test]
    fn a_written_reference_parses_and_writes_the_verifiers_policy() {
        let r = parse(&json(&reference())).expect("parses");
        let p = r.policy();
        assert_eq!(p.products, Some(vec![SnpProduct::Genoa, SnpProduct::Turin]));
        assert!(!p.allow_milan);
        assert_eq!(p.smt, SnpSmt::Forbidden);
        assert_eq!(
            p.minimum, None,
            "the bulletins' default stays under the floor"
        );
        assert_eq!(
            p.floor.map(|f| (f.boot_loader, f.microcode)),
            Some((0x0A, 0x58))
        );
        assert!(p.vlek_csp_ids.is_empty());
        let mut milan = reference();
        milan.products.push("Milan".into());
        assert!(milan.policy().allow_milan);
    }

    /// SECURITY: every judgement is written; nothing defaults to lenient,
    /// and a field this verifier does not read is refused, not ignored.
    #[test]
    fn a_reference_missing_a_judgement_or_carrying_an_unknown_field_is_refused() {
        let base = serde_json::to_value(reference()).unwrap();
        for (path, why) in [
            ("/tcb_minimum/tee", "missing field"),
            ("/guest_policy/smt", "missing field"),
            ("/signing_keys", "missing field"),
            ("/products", "missing field"),
            ("/launch/measurements_sha384", "missing field"),
        ] {
            let mut v = base.clone();
            let (parent, key) = path.rsplit_once('/').unwrap();
            let parent = if parent.is_empty() {
                &mut v
            } else {
                v.pointer_mut(parent).unwrap()
            };
            parent.as_object_mut().unwrap().remove(key).unwrap();
            let e = parse(&serde_json::to_vec(&v).unwrap()).unwrap_err();
            assert!(e.contains(why), "{path}: {e}");
        }
        for at in [
            "",
            "/launch",
            "/guest_policy",
            "/tcb_minimum",
            "/signing_keys",
        ] {
            let mut v = base.clone();
            let o = if at.is_empty() {
                &mut v
            } else {
                v.pointer_mut(at).unwrap()
            };
            o.as_object_mut()
                .unwrap()
                .insert("allow_debug".into(), serde_json::Value::Bool(true));
            let e = parse(&serde_json::to_vec(&v).unwrap()).unwrap_err();
            assert!(e.contains("unknown field"), "{at}: {e}");
        }
        let mut v = base.clone();
        v["guest_policy"]["smt"] = "sometimes".into();
        assert!(parse(&serde_json::to_vec(&v).unwrap()).is_err());
        let mut v = base;
        v["tcb_minimum"]["snp"] = 256.into();
        assert!(parse(&serde_json::to_vec(&v).unwrap()).is_err());
    }

    #[test]
    fn a_reference_this_verifier_could_not_enforce_is_refused() {
        let bad = |edit: fn(&mut SnpReference)| {
            let mut r = reference();
            edit(&mut r);
            parse(&json(&r)).unwrap_err()
        };
        assert!(bad(|r| r.schema = crate::reference::SCHEMA.into()).contains("schema"));
        assert!(bad(|r| r.tee = "intel-tdx".into()).contains("tee"));
        assert!(bad(|r| r.provider = "Scaleway EM".into()).contains("provider"));
        assert!(bad(|r| r.image.version = "0.0.9".into()).contains("below"));
        assert!(bad(|r| r.launch.measurements_sha384.clear()).contains("empty"));
        assert!(bad(|r| r.launch.measurements_sha384[0] = "4C".repeat(48)).contains("lowercase"));
        assert!(bad(|r| r.launch.measurements_sha384[0] = "4c".repeat(47)).contains("96"));
        assert!(bad(|r| r.launch.measurements_sha384[1] = "4c".repeat(48)).contains("twice"));
        assert!(bad(|r| r.launch.host_data = Some("00".repeat(31))).contains("host_data"));
        assert!(bad(|r| r.products.clear()).contains("empty"));
        assert!(bad(|r| r.products = vec!["Bergamo".into()]).contains("Bergamo"));
        assert!(bad(|r| r.products.push("Genoa".into())).contains("twice"));
        assert!(bad(|r| r.signing_keys.vlek_csp_ids = vec![String::new()]).contains("cspId"));
    }

    /// Required refusal: a launch digest that is not a predicted one.
    #[test]
    fn the_launch_digest_must_be_a_predicted_one_and_host_data_the_pinned_one() {
        let r = reference();
        let v = verified(&t::genoa_evidence(|_| {}), &r).expect("the TEST guest");
        let said = r.check_launch(&v).expect("0x4D... is predicted");
        assert!(said.iter().any(|l| l.contains("not judged")), "{said:?}");
        let other = verified(&t::genoa_evidence(|s| s.measurement = [0x4E; 48]), &r).unwrap();
        let e = r.check_launch(&other).unwrap_err();
        assert!(e.contains("not this image"), "{e}");
        let mut pinned = reference();
        pinned.launch.host_data = Some("48".repeat(32));
        let v = verified(&t::genoa_evidence(|s| s.host_data = [0x48; 32]), &pinned).unwrap();
        pinned.check_launch(&v).expect("pinned HOST_DATA");
        let v = verified(&t::genoa_evidence(|s| s.host_data = [0x49; 32]), &pinned).unwrap();
        assert!(pinned.check_launch(&v).unwrap_err().contains("HOST_DATA"));
    }

    /// The written floor and SMT rule reach the verifier: below the floor,
    /// or SMT on under `forbidden`, the TEST guest is refused.
    #[test]
    fn the_written_floor_and_smt_rule_are_the_verifiers() {
        let mut r = reference();
        r.tcb_minimum.tee = 1;
        let e = verified(&t::genoa_evidence(|_| {}), &r).unwrap_err();
        assert!(
            matches!(&e, crate::snp::SnpRejected::TcbTooLow(m) if m.contains("TEE")),
            "{e}"
        );
        let r = reference();
        let e = verified(&t::genoa_evidence(|s| s.platform_info |= 1), &r).unwrap_err();
        assert!(matches!(e, crate::snp::SnpRejected::Smt(_)), "{e}");
        let mut allowed = reference();
        allowed.guest_policy.smt = SnpSmt::Allowed;
        verified(&t::genoa_evidence(|s| s.platform_info |= 1), &allowed).expect("allowed");
        let mut turin_only = reference();
        turin_only.products = vec!["Turin".into()];
        let e = verified(&t::genoa_evidence(|_| {}), &turin_only).unwrap_err();
        assert!(
            matches!(e, crate::snp::SnpRejected::ProductRefused(_)),
            "{e}"
        );
    }

    /// Signed and logged like the TDX reference, by the same code.
    #[test]
    fn a_signed_and_logged_reference_loads_and_an_unsigned_one_is_refused() {
        const NOW: u64 = 1_792_112_400 + 60;
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x0ne-snp-0.1.0.json");
        std::fs::write(&p, json(&reference())).unwrap();
        let anchor = crate::archive_key::Anchor::of(crate::openpgp::Cert::parse(
            &crate::archive_key::test_key::keyring(),
        ));
        let log = crate::transparency::Anchor::of(crate::transparency::test_log::policy());
        let state = dir.path().join("state");
        let e = load(&p, true, &anchor, &log, &state, NOW).unwrap_err();
        assert_eq!(e.exit_code(), crate::reference::EXIT_SIGNATURE, "{e}");
        let bytes = std::fs::read(&p).unwrap();
        let sig =
            crate::archive_key::test_key::sign(&bytes, crate::archive_key::test_key::CREATED + 1);
        std::fs::write(crate::reference::signature_path(&p), sig).unwrap();
        let e = load(&p, true, &anchor, &log, &state, NOW).unwrap_err();
        assert_eq!(e.exit_code(), crate::transparency::EXIT_NOT_LOGGED, "{e}");
        crate::transparency::test_log::log_reference(&p, NOW).unwrap();
        let loaded = load(&p, true, &anchor, &log, &state, NOW).expect("signed and logged");
        assert_eq!(loaded.reference, reference());
        assert_eq!(loaded.sha256_hex.len(), 64);
        // A TDX reference is not an SNP one, signed or not.
        let tdx = dir.path().join("x0ne-cvm-0.1.0.json");
        std::fs::write(
            &tdx,
            serde_json::to_vec(&crate::reference::tests::reference(
                &"0".repeat(64),
                "c3-standard-4",
            ))
            .unwrap(),
        )
        .unwrap();
        let bytes = std::fs::read(&tdx).unwrap();
        let sig =
            crate::archive_key::test_key::sign(&bytes, crate::archive_key::test_key::CREATED + 1);
        std::fs::write(crate::reference::signature_path(&tdx), sig).unwrap();
        crate::transparency::test_log::log_reference(&tdx, NOW).unwrap();
        let e = load(&tdx, true, &anchor, &log, &state, NOW).unwrap_err();
        assert_eq!(e.exit_code(), crate::reference::EXIT_UNUSABLE, "{e}");
        assert!(e.to_string().contains("schema"), "{e}");
    }
}
