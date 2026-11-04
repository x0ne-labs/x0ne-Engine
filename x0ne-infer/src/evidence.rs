//! Where evidence comes from.
//!
//! Production: the TDX quote through configfs-tsm (`/sys/kernel/config/tsm/
//! report/<name>/{inblob,outblob}`, the kernel's standard interface since 6.7 —
//! the same path `x0ne cc` uses today) plus the NVIDIA GPU confidential-mode
//! evidence. There is no development substitute — see the note at the end
//! of this file and the test that keeps it out.
//!
//! The source reports what it saw; it decides nothing. Deciding is the
//! verifier's job (`verify`), opening the gate is the gate's (`gate`).

use x0ne_confidential::model::{AttestationEvidence, ExecutionEnvironment};

/// What an evidence source hands to the verifier: the environment it claims
/// and the evidence for that claim, collected at one instant with a nonce the
/// caller chose (replay protection: the quote must echo it).
#[derive(Debug, Clone)]
pub struct Collected {
    pub environment: ExecutionEnvironment,
    pub evidence: AttestationEvidence,
}

/// A source of attestation evidence.
pub trait EvidenceSource {
    /// Collect evidence that echoes `nonce`. `now_unix` stamps it.
    fn collect(&self, nonce: &[u8; 64], now_unix: u64) -> anyhow::Result<Collected>;
}

/// configfs-tsm `provider` of a TDX guest (Linux `drivers/virt/coco/tdx-guest`).
pub const PROVIDER_TDX: &str = "tdx_guest";
/// configfs-tsm `provider` of an SEV-SNP guest (Linux `drivers/virt/coco/sev-guest`).
pub const PROVIDER_SNP: &str = "sev_guest";
/// Where the image leaves the VCEK (or VLEK) DER when the host's auxblob
/// carries none. Untrusted like the auxblob: the verifier chains it to AMD.
pub const SNP_CERT_FILE: &str = "/run/x0ne/snp-vcek.der";
/// A VCEK or VLEK is about 1.4 KB; anything past this is not one.
const MAX_SNP_CERT: u64 = 16 * 1024;
/// Where the image leaves AMD's CRL for the chip's product (the host fetches
/// it from AMD's KDS). Untrusted: the verifier checks it against the pinned ARK.
pub const SNP_CRL_FILE: &str = "/run/x0ne/snp-crl.der";
/// The bound x0ne-cloud-config copies it under (fw_cfg `opt/x0ne/amd-crl`).
const MAX_SNP_CRL: u64 = 64 * 1024;

/// AMD's CRL as the image left it at `path`: present, not empty, within the
/// bound. Its signature and dates are the verifier's to judge.
pub fn read_snp_crl(path: &std::path::Path) -> anyhow::Result<Vec<u8>> {
    let crl = crate::fsread::read_capped(path, MAX_SNP_CRL)
        .map_err(|e| anyhow::anyhow!("AMD CRL {}: {e}", path.display()))?;
    if crl.is_empty() {
        anyhow::bail!("AMD CRL {}: empty", path.display());
    }
    Ok(crl)
}

/// GUIDs of the SEV-SNP certificate table (GHCB spec, extended guest
/// request), as virtee/snpguest names them.
pub const SNP_GUID_VCEK: &str = "63da758d-e664-4564-adc5-f4b93be8accd";
pub const SNP_GUID_VLEK: &str = "a8074bc2-a25a-483e-aae6-39c045a0b8a1";

/// configfs-tsm on the local kernel: the production source. The provider
/// names the evidence: a TDX quote (`tdx_guest`) or an SEV-SNP report
/// (`sev_guest`) followed by the certificate that signed it.
pub struct ConfigfsTsm {
    root: std::path::PathBuf,
    snp_cert_file: std::path::PathBuf,
}

impl ConfigfsTsm {
    /// The kernel's real interface (`/sys/kernel/config/tsm`). The same
    /// interface `x0ne cc` drives over SSH (`cc_runner.rs`), read in-process:
    ///
    /// ```text
    /// mkdir  <root>/report/<name>          # one report session
    /// read   <root>/report/<name>/provider # tdx_guest or sev_guest
    /// write  <root>/report/<name>/inblob   # 64 bytes: report_data (nonce/binding)
    /// read   <root>/report/<name>/outblob  # the TDX quote, or the SNP report
    /// read   <root>/report/<name>/auxblob  # SNP: the host's certificate table
    /// ```
    pub fn system() -> Self {
        Self {
            root: std::path::PathBuf::from("/sys/kernel/config/tsm"),
            snp_cert_file: std::path::PathBuf::from(SNP_CERT_FILE),
        }
    }
    /// An injected root — tests, and non-standard mounts.
    pub fn at(root: impl Into<std::path::PathBuf>) -> Self {
        Self {
            root: root.into(),
            snp_cert_file: std::path::PathBuf::from(SNP_CERT_FILE),
        }
    }
    /// Where the SNP certificate is read when the auxblob has none.
    #[must_use]
    pub fn with_snp_cert_file(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.snp_cert_file = path.into();
        self
    }

    /// The report, then the certificate of the key that signed it: the
    /// auxblob's entry for that key, else [`SNP_CERT_FILE`]. Checked for
    /// shape only; the verifier judges it.
    fn snp_evidence(&self, session: &std::path::Path, report: Vec<u8>) -> anyhow::Result<Vec<u8>> {
        use x0ne_confidential::snp_report;
        if report.len() != snp_report::REPORT_LEN {
            anyhow::bail!(
                "configfs-tsm: sev_guest outblob is {} bytes, an SEV-SNP report is {}",
                report.len(),
                snp_report::REPORT_LEN
            );
        }
        let parsed = snp_report::parse(&report)
            .map_err(|e| anyhow::anyhow!("configfs-tsm: the SEV-SNP report does not parse: {e}"))?;
        let (name, guid) = match parsed.signing_key() {
            snp_report::SIGNING_KEY_VCEK => ("VCEK", SNP_GUID_VCEK),
            snp_report::SIGNING_KEY_VLEK => ("VLEK", SNP_GUID_VLEK),
            other => anyhow::bail!(
                "configfs-tsm: the SEV-SNP report names signing key {other}, neither VCEK nor VLEK"
            ),
        };
        // auxblob is absent on kernels before the certificate support, and
        // empty when the host gives no certificates.
        let aux = std::fs::read(session.join("auxblob")).unwrap_or_default();
        if !aux.is_empty() {
            match snp_cert_from_table(&aux, guid) {
                Ok(Some(cert)) => return Ok(crate::snp::evidence_payload(&report, cert)),
                Ok(None) => {}
                Err(e) => anyhow::bail!("configfs-tsm: auxblob: {e}"),
            }
        }
        let cert = crate::fsread::read_capped(&self.snp_cert_file, MAX_SNP_CERT).map_err(|e| {
            anyhow::anyhow!(
                "configfs-tsm: no {name}: the host's certificate table has none and {} is \
                 unreadable ({e})",
                self.snp_cert_file.display()
            )
        })?;
        if cert.is_empty() {
            anyhow::bail!(
                "configfs-tsm: no {name}: {} is empty",
                self.snp_cert_file.display()
            );
        }
        Ok(crate::snp::evidence_payload(&report, &cert))
    }
}

/// The 16 bytes of a GUID written as text, in the order the text reads.
fn guid_bytes(text: &str) -> [u8; 16] {
    let hex: Vec<u8> = text.bytes().filter(|b| *b != b'-').collect();
    let mut out = [0u8; 16];
    for (i, pair) in hex.chunks(2).enumerate() {
        let s = std::str::from_utf8(pair).unwrap_or("00");
        out[i] = u8::from_str_radix(s, 16).unwrap_or(0);
    }
    out
}

/// The same GUID in the mixed-endian layout (first three fields reversed).
fn guid_bytes_mixed(text: &str) -> [u8; 16] {
    let mut b = guid_bytes(text);
    b[..4].reverse();
    b[4..6].reverse();
    b[6..8].reverse();
    b
}

/// The certificate `guid` names in an SEV-SNP certificate table: entries of
/// GUID (16), offset (u32 LE) and length (u32 LE) from the table's start,
/// ended by an all-zero entry. Either GUID byte order is accepted: the bytes
/// are verified against AMD's chain whatever the entry says.
pub fn snp_cert_from_table<'a>(table: &'a [u8], guid: &str) -> Result<Option<&'a [u8]>, String> {
    const ENTRY: usize = 24;
    let (plain, mixed) = (guid_bytes(guid), guid_bytes_mixed(guid));
    let mut at = 0usize;
    loop {
        let Some(entry) = table.get(at..at + ENTRY) else {
            return Err(format!(
                "certificate table of {} bytes has no terminator",
                table.len()
            ));
        };
        if entry.iter().all(|b| *b == 0) {
            return Ok(None);
        }
        let offset = u32::from_le_bytes([entry[16], entry[17], entry[18], entry[19]]) as usize;
        let length = u32::from_le_bytes([entry[20], entry[21], entry[22], entry[23]]) as usize;
        if entry[..16] == plain || entry[..16] == mixed {
            return match offset
                .checked_add(length)
                .and_then(|end| table.get(offset..end))
            {
                Some(cert) if !cert.is_empty() => Ok(Some(cert)),
                _ => Err(format!(
                    "entry {guid} at {offset}+{length} is outside the {}-byte table",
                    table.len()
                )),
            };
        }
        at += ENTRY;
    }
}

impl EvidenceSource for ConfigfsTsm {
    fn collect(&self, nonce: &[u8; 64], now_unix: u64) -> anyhow::Result<Collected> {
        use std::io::{Read, Write};
        use x0ne_confidential::model::{EvidenceFormat, TeeTechnology};
        let session = self.root.join("report").join("x0ne-infer");
        std::fs::create_dir_all(&session).map_err(|e| {
            anyhow::anyhow!(
                "configfs-tsm: cannot create report session {} ({e}); \
                 not a confidential guest, or not root",
                session.display()
            )
        })?;
        let provider = std::fs::read_to_string(session.join("provider"))
            .map_err(|e| anyhow::anyhow!("configfs-tsm: reading provider: {e}"))?;
        let provider = provider.trim();
        if provider != PROVIDER_TDX && provider != PROVIDER_SNP {
            anyhow::bail!(
                "configfs-tsm: provider {provider:?} is neither {PROVIDER_TDX} nor {PROVIDER_SNP}"
            );
        }
        std::fs::OpenOptions::new()
            .write(true)
            .open(session.join("inblob"))
            .and_then(|mut f| f.write_all(nonce))
            .map_err(|e| anyhow::anyhow!("configfs-tsm: writing inblob: {e}"))?;
        let mut quote = Vec::new();
        std::fs::File::open(session.join("outblob"))
            .and_then(|mut f| f.read_to_end(&mut quote))
            .map_err(|e| anyhow::anyhow!("configfs-tsm: reading outblob: {e}"))?;
        if quote.is_empty() {
            anyhow::bail!("configfs-tsm: outblob is empty — no quote was generated");
        }
        let (cpu, format, payload) = if provider == PROVIDER_SNP {
            let payload = self.snp_evidence(&session, quote)?;
            (
                TeeTechnology::AmdSevSnp,
                EvidenceFormat::SevSnpReport,
                payload,
            )
        } else {
            (TeeTechnology::IntelTdx, EvidenceFormat::TdxQuote, quote)
        };
        Ok(Collected {
            environment: ExecutionEnvironment::ConfidentialVm { cpu },
            evidence: AttestationEvidence {
                format,
                payload,
                collected_at_unix: now_unix,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole configfs path against a fake root: the nonce lands in
    /// inblob, the quote comes back from outblob, an empty outblob refuses.
    /// (The real mount differs only by the path and by who may write it.)
    #[test]
    fn configfs_collect_reads_the_quote_the_kernel_left() {
        let dir = tempfile::tempdir().unwrap();
        let session = dir.path().join("report").join("x0ne-infer");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(session.join("provider"), b"tdx_guest\n").unwrap();
        std::fs::write(session.join("inblob"), b"").unwrap();
        std::fs::write(session.join("outblob"), vec![7u8; 128]).unwrap();
        let src = ConfigfsTsm::at(dir.path());
        let nonce = [0x5au8; 64];
        let got = src.collect(&nonce, 1_000).expect("collects");
        assert_eq!(std::fs::read(session.join("inblob")).unwrap(), nonce);
        assert_eq!(got.evidence.payload, vec![7u8; 128]);
        assert!(got.environment.claims_confidentiality());

        std::fs::write(session.join("outblob"), b"").unwrap();
        assert!(src.collect(&nonce, 1_000).is_err(), "empty outblob refuses");
    }

    /// A configfs session as the kernel leaves it after a report request.
    fn session(provider: &str, outblob: &[u8], auxblob: Option<&[u8]>) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let s = dir.path().join("report").join("x0ne-infer");
        std::fs::create_dir_all(&s).unwrap();
        std::fs::write(s.join("provider"), provider).unwrap();
        std::fs::write(s.join("inblob"), b"").unwrap();
        std::fs::write(s.join("outblob"), outblob).unwrap();
        if let Some(aux) = auxblob {
            std::fs::write(s.join("auxblob"), aux).unwrap();
        }
        dir
    }

    /// A TEST Genoa report signed by the TEST VCEK, `report_data` echoed.
    fn snp_report(report_data: [u8; 64], key_info: Option<u32>) -> Vec<u8> {
        use crate::snp::tests::{sign, Spec, CURRENT};
        let mut spec = Spec::genoa();
        spec.report_data = report_data;
        if let Some(k) = key_info {
            spec.key_info = k;
        }
        sign(spec.bytes(), CURRENT)
    }

    /// A certificate table: `(guid bytes, certificate)` entries, then zeros.
    fn table(entries: &[([u8; 16], &[u8])]) -> Vec<u8> {
        let head = 24 * (entries.len() + 1);
        let mut out = Vec::new();
        let mut body = Vec::new();
        for (guid, cert) in entries {
            out.extend_from_slice(guid);
            out.extend_from_slice(&((head + body.len()) as u32).to_le_bytes());
            out.extend_from_slice(&(cert.len() as u32).to_le_bytes());
            body.extend_from_slice(cert);
        }
        out.extend_from_slice(&[0u8; 24]);
        out.extend_from_slice(&body);
        out
    }

    /// `sev_guest`: the report and the VCEK the host put in the auxblob come
    /// back as one `SevSnpReport`, and that payload verifies to the TEST AMD
    /// chain with the nonce in `REPORT_DATA`. Both GUID byte orders select it.
    #[test]
    fn sev_guest_collects_the_report_and_the_vcek_of_the_auxblob() {
        use crate::snp::tests::{test_anchors, NOW, VCEK_CURRENT, VCEK_OLD};
        use x0ne_confidential::model::{EvidenceFormat, TeeTechnology};
        let nonce = [0x3cu8; 64];
        let report = snp_report(nonce, None);
        for vcek_guid in [guid_bytes(SNP_GUID_VCEK), guid_bytes_mixed(SNP_GUID_VCEK)] {
            let aux = table(&[
                (guid_bytes(SNP_GUID_VLEK), VCEK_OLD),
                (vcek_guid, VCEK_CURRENT),
            ]);
            let dir = session("sev_guest\n", &report, Some(&aux));
            let got = ConfigfsTsm::at(dir.path())
                .with_snp_cert_file(dir.path().join("absent.der"))
                .collect(&nonce, NOW)
                .expect("collects");
            assert_eq!(got.evidence.format, EvidenceFormat::SevSnpReport);
            assert_eq!(
                got.environment,
                ExecutionEnvironment::ConfidentialVm {
                    cpu: TeeTechnology::AmdSevSnp
                }
            );
            assert_eq!(
                got.evidence.payload,
                crate::snp::evidence_payload(&report, VCEK_CURRENT)
            );
            let v = crate::snp::verify_snp_report(
                &got.evidence.payload,
                crate::snp::tests::TEST_GENOA_CRL,
                &test_anchors(),
                &crate::snp::SnpPolicy::default(),
                NOW,
            )
            .expect("the collected payload verifies");
            assert_eq!(v.report_data(), &nonce);
        }
    }

    /// No certificate in the auxblob (a host without certificate support):
    /// the image's file is read; without it the source refuses, naming it.
    #[test]
    fn sev_guest_falls_back_to_the_image_file_and_refuses_without_one() {
        use crate::snp::tests::{NOW, VCEK_CURRENT};
        let nonce = [0x11u8; 64];
        let report = snp_report(nonce, None);
        for aux in [
            None,
            Some(&[0u8; 24][..]),
            Some(&table(&[(guid_bytes(SNP_GUID_VLEK), b"x")])[..]),
        ] {
            let dir = session("sev_guest", &report, aux);
            let file = dir.path().join("vcek.der");
            let src = ConfigfsTsm::at(dir.path()).with_snp_cert_file(&file);
            let err = src
                .collect(&nonce, NOW)
                .expect_err("no certificate anywhere");
            assert!(format!("{err:#}").contains("no VCEK"), "{err:#}");
            std::fs::write(&file, VCEK_CURRENT).unwrap();
            let got = src.collect(&nonce, NOW).expect("the file is read");
            assert_eq!(
                got.evidence.payload,
                crate::snp::evidence_payload(&report, VCEK_CURRENT)
            );
            std::fs::write(&file, b"").unwrap();
            assert!(
                src.collect(&nonce, NOW).is_err(),
                "an empty file is no certificate"
            );
        }
    }

    /// A VLEK-signed report takes the VLEK entry, never the VCEK one.
    #[test]
    fn a_vlek_signed_report_takes_the_vlek_entry() {
        use crate::snp::tests::{NOW, VCEK_CURRENT, VCEK_OLD};
        let nonce = [0x22u8; 64];
        // KEY_INFO.SIGNING_KEY (bits 4:2) = 1, VLEK.
        let report = snp_report(nonce, Some(1 << 2));
        let aux = table(&[
            (guid_bytes(SNP_GUID_VCEK), VCEK_CURRENT),
            (guid_bytes(SNP_GUID_VLEK), VCEK_OLD),
        ]);
        let dir = session("sev_guest", &report, Some(&aux));
        let got = ConfigfsTsm::at(dir.path())
            .collect(&nonce, NOW)
            .expect("collects");
        assert_eq!(
            got.evidence.payload,
            crate::snp::evidence_payload(&report, VCEK_OLD)
        );
        let only_vcek = table(&[(guid_bytes(SNP_GUID_VCEK), VCEK_CURRENT)]);
        let dir = session("sev_guest", &report, Some(&only_vcek));
        let err = ConfigfsTsm::at(dir.path())
            .with_snp_cert_file(dir.path().join("absent.der"))
            .collect(&nonce, NOW)
            .expect_err("no VLEK");
        assert!(format!("{err:#}").contains("no VLEK"), "{err:#}");
    }

    /// Refusals before any verifier: an unknown or missing provider, an
    /// outblob that is not one report, a report that does not parse, a
    /// report signed by no key, a malformed certificate table.
    #[test]
    fn malformed_snp_evidence_and_unknown_providers_refuse() {
        use crate::snp::tests::{NOW, VCEK_CURRENT};
        let nonce = [0x33u8; 64];
        let report = snp_report(nonce, None);
        let refuses = |dir: &tempfile::TempDir, why: &str| {
            let err = ConfigfsTsm::at(dir.path())
                .with_snp_cert_file(dir.path().join("absent.der"))
                .collect(&nonce, NOW)
                .expect_err(why);
            format!("{err:#}")
        };
        let e = refuses(&session("arm_cca_guest", &report, None), "unknown provider");
        assert!(e.contains("arm_cca_guest"), "{e}");
        let dir = session("sev_guest", &report, None);
        std::fs::remove_file(dir.path().join("report/x0ne-infer/provider")).unwrap();
        assert!(refuses(&dir, "no provider").contains("provider"));
        let mut long = report.clone();
        long.push(0);
        assert!(refuses(&session("sev_guest", &long, None), "length").contains("1185 bytes"));
        assert!(
            refuses(&session("sev_guest", &report[..0x2A0], None), "short").contains("672 bytes")
        );
        let mut bad = report.clone();
        bad[0..4].copy_from_slice(&99u32.to_le_bytes());
        assert!(refuses(&session("sev_guest", &bad, None), "version").contains("does not parse"));
        let none = snp_report(nonce, Some(7 << 2));
        assert!(refuses(&session("sev_guest", &none, None), "no key").contains("signing key 7"));
        let mut cut = table(&[(guid_bytes(SNP_GUID_VCEK), VCEK_CURRENT)]);
        cut.truncate(30);
        assert!(refuses(&session("sev_guest", &report, Some(&cut)), "cut").contains("auxblob"));
        let mut out_of_range = table(&[(guid_bytes(SNP_GUID_VCEK), VCEK_CURRENT)]);
        out_of_range[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(
            refuses(&session("sev_guest", &report, Some(&out_of_range)), "range")
                .contains("outside")
        );
    }

    #[test]
    fn guid_text_and_byte_orders() {
        assert_eq!(
            guid_bytes(SNP_GUID_VCEK)[..4],
            [0x63, 0xda, 0x75, 0x8d],
            "the order the text reads"
        );
        assert_eq!(
            guid_bytes_mixed(SNP_GUID_VCEK)[..8],
            [0x8d, 0x75, 0xda, 0x63, 0x64, 0xe6, 0x64, 0x45]
        );
        assert_eq!(
            guid_bytes_mixed(SNP_GUID_VCEK)[8..],
            guid_bytes(SNP_GUID_VCEK)[8..]
        );
    }

    /// No build flag may substitute the evidence source.
    ///
    /// SECURITY: a `dev-unattested` feature once gated a `DevFake` here and
    /// the docs promised it served without a TEE — it did not (nothing
    /// referenced it), and the promise alone was a liability. A failure of
    /// this test means someone reintroduced a compile-time path around the
    /// only thing that produces evidence: that is a review event, not a
    /// convenience.
    #[test]
    fn no_compile_time_feature_can_substitute_the_evidence_source() {
        let src = include_str!("evidence.rs");
        let body = &src[..src.find("mod tests").unwrap_or(src.len())];
        assert!(
            !body.contains("cfg(feature"),
            "an evidence source is gated behind a build feature again"
        );
        let manifest = include_str!("../Cargo.toml");
        assert!(
            !manifest.contains("dev-unattested"),
            "the dev bypass feature is back in Cargo.toml"
        );
    }
}

// There is deliberately NO development bypass in this crate.
//
// A `dev-unattested` feature used to exist here with a `DevFake` evidence
// source, and README/SECURITY/Cargo.toml all promised it let the server run
// on a machine with no TEE. Measured 2026-08-21: `main.rs` never referenced
// it, so the binary refused identically with and without the feature (exit
// 64, both) — and even wired, `Gate::attest` refuses a `Standard`
// environment before reading any verdict. A documented flag that changes
// nothing is this repository's signature defect, and in a confidential
// engine it is worse than useless: it advertises a bypass that reviewers
// must then rule out.
//
// It is gone. Contributors develop against the library tests, which need no
// hardware; the only way to serve is a real, verified enclave. The test
// below keeps it that way.
