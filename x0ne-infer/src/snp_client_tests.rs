//! The client's verdict on an SEV-SNP guest, as the relay runs it
//! (`remote_verify::verify_snp_document`, `SessionCheck::snp`): TEST AMD
//! chains, a fake AMD KDS serving the TEST CRL with its clock, the fake
//! NVIDIA of `attest_v2::fixtures`, a reference signed and logged by the
//! TEST archive key and log.

use crate::attest_v2::fixtures::FakeNvidia;
use crate::attest_v2::{
    b64, BindingDoc, Document, FileDoc, GpuAttested, GpuDoc, ModelDoc, NrasDoc, ServerDoc,
    SessionDoc, SessionGpuDoc, SnpDoc, TdxDoc,
};
use crate::binding::BindingV2;
use crate::cloud_config::{hex, CloudConfig, Loaded};
use crate::remote_verify::{
    verify_snp_document, Anchors, Pinned, Refusal, Report, SessionCheck, SnpCrl, SnpExpectations,
};
use crate::snp::tests::{genoa_evidence, test_anchors, Spec, NOW};
use crate::snp_reference::SnpReference;
use std::sync::Arc;
use x0ne_confidential::snp_report::{POLICY_DEBUG, POLICY_MIGRATE_MA, SIGNED_LEN};

const MODEL: &str = "qwen3:0.6b";
const TLS: [u8; 32] = [0x03; 32];
/// When the TEST archive key and log sign and cosign the reference.
const REF_NOW: u64 = 1_792_112_400 + 60;

fn http() -> reqwest::Client {
    reqwest::Client::new()
}

/// A fake AMD KDS: the Genoa CRL at its path, `crl` as its body, `date` as
/// its clock (always set: hyper would write the real time otherwise).
async fn fake_kds(date: u64, crl: &'static [u8]) -> String {
    use axum::http::HeaderValue;
    let app = axum::Router::new().route(
        "/vcek/v1/Genoa/crl",
        axum::routing::get(move || async move {
            let mut resp = axum::response::Response::new(axum::body::Body::from(crl));
            resp.headers_mut().insert(
                axum::http::header::DATE,
                HeaderValue::from_str(&crate::remote_verify::http_date(date)).unwrap(),
            );
            resp
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

/// `snp_reference::tests::reference` edited, signed by the TEST archive
/// key, logged in the TEST log, loaded as the relay loads it.
fn signed_reference(
    dir: &std::path::Path,
    edit: impl FnOnce(&mut SnpReference),
) -> crate::snp_reference::Loaded {
    let mut r = crate::snp_reference::tests::reference();
    edit(&mut r);
    let p = own_dir(dir, &serde_json::to_vec(&r).unwrap()).join("x0ne-cvm-snp-0.1.0.json");
    std::fs::write(&p, serde_json::to_vec(&r).unwrap()).unwrap();
    sign_and_log(&p);
    match load(&p).expect("signed and logged") {
        Pinned::Snp(l) => *l,
        Pinned::Tdx(_) => panic!("an SEV-SNP reference loaded as TDX"),
    }
}

/// One directory per reference file: its own TEST log and log state.
fn own_dir(dir: &std::path::Path, bytes: &[u8]) -> std::path::PathBuf {
    let d = dir.join(&hex(&crate::cloud_config::sha256(bytes))[..16]);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn sign_and_log(p: &std::path::Path) {
    let bytes = std::fs::read(p).unwrap();
    let sig = crate::archive_key::test_key::sign(&bytes, crate::archive_key::test_key::CREATED + 1);
    std::fs::write(crate::reference::signature_path(p), sig).unwrap();
    crate::transparency::test_log::log_reference(p, REF_NOW).unwrap();
}

fn load(p: &std::path::Path) -> Result<Pinned, crate::reference::LoadError> {
    Pinned::load(
        p,
        true,
        &crate::archive_key::Anchor::of(crate::openpgp::Cert::parse(
            &crate::archive_key::test_key::keyring(),
        )),
        &crate::transparency::Anchor::of(crate::transparency::test_log::policy()),
        &p.parent().unwrap().join("state"),
        REF_NOW,
    )
}

/// One session: its config, its binding, its GPUs.
struct Guest {
    config: Loaded,
    binding: BindingV2,
    nvidia: Arc<FakeNvidia>,
}

impl Guest {
    fn new(attest: &str) -> Self {
        let config = CloudConfig::new(MODEL, [5; 32], [6; 32], NOW + 600, attest, 443, 64)
            .to_loaded()
            .unwrap();
        let session_pubkey = ed25519_dalek::SigningKey::from_bytes(&[0x51; 32])
            .verifying_key()
            .to_bytes();
        Self {
            binding: BindingV2 {
                session_nonce: [0x01; 32],
                session_pubkey,
                tls_spki_sha256: TLS,
                config_sha256: config.sha256,
            },
            config,
            nvidia: Arc::new(FakeNvidia::new()),
        }
    }

    /// The TEST Genoa guest's evidence over this session's v2 binding.
    fn evidence(&self, edit: impl FnOnce(&mut Spec)) -> Vec<u8> {
        let rd = self.binding.snp_report_data();
        genoa_evidence(|s| {
            s.report_data = rd;
            edit(s);
        })
    }

    fn gpu(&self) -> GpuAttested {
        self.nvidia.attest(&self.binding.gpu_nonce_hex(), NOW, None)
    }

    /// The document an SEV-SNP guest serves (the guest side's shape:
    /// empty TDX evidence, the report and its VCEK under `snp`).
    fn doc(&self, payload: &[u8], gpu: Option<&GpuAttested>) -> Document {
        let catalog = crate::registry::Catalog::embedded();
        let entry = catalog.get(MODEL).unwrap();
        let (gpu, nras) = match gpu {
            Some(g) => (
                g.evidence
                    .iter()
                    .enumerate()
                    .map(|(index, ev)| GpuDoc {
                        index,
                        nras_name: g.per_gpu[index].0.clone(),
                        certificate_b64: ev.certificate.clone(),
                        nras_token: g.per_gpu[index].1.clone(),
                    })
                    .collect(),
                NrasDoc {
                    overall_token: g.overall_token.clone(),
                    jwks: g.jwks.clone(),
                    ocsp_b64: Vec::new(),
                },
            ),
            None => (
                Vec::new(),
                NrasDoc {
                    overall_token: String::new(),
                    jwks: serde_json::json!({"keys": []}),
                    ocsp_b64: Vec::new(),
                },
            ),
        };
        Document {
            schema: crate::attest_v2::SCHEMA.into(),
            binding: BindingDoc {
                context: String::from_utf8_lossy(crate::binding::CONTEXT_V2).into_owned(),
                session_nonce_hex: hex(&self.binding.session_nonce),
                session_pubkey_hex: hex(&self.binding.session_pubkey),
                tls_spki_sha256_hex: hex(&TLS),
                config_sha256_hex: hex(&self.config.sha256),
            },
            config_b64: b64(&self.config.bytes),
            tdx: TdxDoc {
                quote_b64: String::new(),
                ccel_b64: String::new(),
            },
            snp: Some(SnpDoc {
                evidence_b64: b64(payload),
            }),
            gpu,
            nras,
            model: ModelDoc {
                name: MODEL.into(),
                digest_hex: entry.digest_sha256.clone().unwrap(),
                files: crate::attest_v2::catalog_files(entry)
                    .into_iter()
                    .map(|(path, sha256)| FileDoc { path, sha256 })
                    .collect(),
            },
            server: ServerDoc {
                version: "test".into(),
                attested_at_unix: NOW,
                verdict: serde_json::Value::Null,
                live_tcb: (String::new(), String::new()),
                tcb: None,
            },
        }
    }

    /// This connection's session attestation: its report over `BindingV3`
    /// of `exporter`, and with the GPU product NRAS tokens for the nonce
    /// over that report's signed bytes.
    fn session(&self, exporter: [u8; 32], gpu: bool) -> SessionDoc {
        let rd = crate::binding::BindingV3::of(&self.binding, exporter).snp_report_data();
        let payload = genoa_evidence(|s| s.report_data = rd);
        let nonce = crate::binding::session_gpu_nonce(&exporter, &payload[..SIGNED_LEN]);
        SessionDoc {
            schema: crate::attest_v2::SESSION_SCHEMA.into(),
            quote_b64: String::new(),
            snp: Some(SnpDoc {
                evidence_b64: b64(&payload),
            }),
            gpu: gpu.then(|| SessionGpuDoc::of(&self.nvidia.attest(&hex(&nonce), NOW, None))),
        }
    }

    fn anchors(&self, kds: &str) -> Anchors {
        let mut a = Anchors::production();
        a.snp = test_anchors();
        a.snp.kds_base_url = kds.into();
        a.nras_intermediate_pins = self.nvidia.intermediate_pins();
        a
    }
}

async fn verify(
    g: &Guest,
    doc: &Document,
    reference: &crate::snp_reference::Loaded,
    anchors: &Anchors,
) -> Result<(Report, SnpCrl), Refusal> {
    let catalog = crate::registry::Catalog::embedded();
    let exp = SnpExpectations {
        config: &g.config,
        tls_spki_sha256: TLS,
        reference,
        catalog_entry: catalog.get(MODEL).unwrap(),
    };
    verify_snp_document(doc, &exp, anchors, &http(), NOW).await
}

#[tokio::test]
async fn an_snp_guest_bound_to_this_session_is_verified_and_each_connection_checked() {
    let dir = tempfile::tempdir().unwrap();
    let reference = signed_reference(dir.path(), |_| {});
    let kds = fake_kds(NOW, crate::snp::tests::TEST_GENOA_CRL).await;
    for attest in ["cpu", "cpu+gpu"] {
        let g = Guest::new(attest);
        let with_gpu = attest == "cpu+gpu";
        let gpu = with_gpu.then(|| g.gpu());
        let doc = g.doc(&g.evidence(|_| {}), gpu.as_ref());
        let anchors = g.anchors(&kds);
        let (report, crl) = verify(&g, &doc, &reference, &anchors)
            .await
            .unwrap_or_else(|e| panic!("{attest}: {e}"));
        let snp = report.snp.as_ref().expect("an SEV-SNP view");
        assert_eq!(
            (snp.product.as_str(), snp.signing_key.as_str()),
            ("Genoa", "VCEK")
        );
        assert_eq!(snp.measurement_hex, "4d".repeat(48));
        assert!(
            snp.crl_url.ends_with("/vcek/v1/Genoa/crl"),
            "{}",
            snp.crl_url
        );
        assert!(
            report.environment.contains("AmdSevSnp"),
            "{}",
            report.environment
        );
        assert_eq!(report.environment.contains("WithGpu"), with_gpu);
        assert_eq!(report.gpu_count, usize::from(with_gpu));
        assert!(report.mrtd_hex.is_empty() && report.rtmr0_hex.is_empty());
        assert!(report.summary[0].starts_with("CPU   AMD SEV-SNP report genuine"));
        assert!(report.summary[1].contains("one the reference predicts"));
        let lim = report.limitations.join("\n");
        assert!(lim.contains("launch digests"), "{lim}");
        assert!(lim.contains("HOST_DATA is not pinned"), "{lim}");
        assert_eq!(lim.contains("CPU only"), !with_gpu, "{lim}");
        assert_eq!(lim.contains("NRAS"), with_gpu, "{lim}");
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["snp"]["product"], "Genoa");

        let exp_catalog = crate::registry::Catalog::embedded();
        let exp = SnpExpectations {
            config: &g.config,
            tls_spki_sha256: TLS,
            reference: &reference,
            catalog_entry: exp_catalog.get(MODEL).unwrap(),
        };
        let check = SessionCheck::snp(&doc, &exp, &anchors, &crl, Arc::new(|| NOW)).unwrap();
        let exporter = [9u8; 32];
        check
            .verify(&g.session(exporter, with_gpu), &exporter)
            .unwrap_or_else(|e| panic!("{attest}: {e}"));
        // Another connection's attestation: another exporter.
        let e = check
            .verify(&g.session([8; 32], with_gpu), &exporter)
            .unwrap_err();
        assert_eq!(e.code, 88, "{e}");
        if with_gpu {
            let mut no_gpu = g.session(exporter, true);
            no_gpu.gpu = None;
            assert_eq!(check.verify(&no_gpu, &exporter).unwrap_err().code, 75);
            // The document's GPU tokens replayed on a connection.
            let mut replayed = g.session(exporter, true);
            replayed.gpu = Some(SessionGpuDoc::of(gpu.as_ref().unwrap()));
            assert_eq!(check.verify(&replayed, &exporter).unwrap_err().code, 75);
        }
    }
}

/// The five required refusals, through the document as the relay sees it,
/// and the other fixed rules of the guest policy.
#[tokio::test]
async fn the_required_refusals_hold_at_the_document() {
    let dir = tempfile::tempdir().unwrap();
    let reference = signed_reference(dir.path(), |_| {});
    let floor = signed_reference(dir.path(), |r| r.tcb_minimum.snp = 0x1C);
    let kds = fake_kds(NOW, crate::snp::tests::TEST_GENOA_CRL).await;
    let g = Guest::new("cpu");
    let anchors = g.anchors(&kds);
    let mut other_root = g.anchors(&kds);
    other_root.snp = crate::snp::SnpAnchors::production();
    other_root.snp.kds_base_url = kds.clone();
    let mut other = g.binding;
    other.tls_spki_sha256[0] ^= 1;
    let cases: Vec<(
        &str,
        Vec<u8>,
        &crate::snp_reference::Loaded,
        &Anchors,
        i32,
        &str,
    )> = vec![
        (
            "debug",
            g.evidence(|s| s.policy |= POLICY_DEBUG),
            &reference,
            &anchors,
            65,
            "DEBUG",
        ),
        (
            "TCB under the written floor",
            g.evidence(|_| {}),
            &floor,
            &anchors,
            65,
            "TCB",
        ),
        (
            "another root",
            g.evidence(|_| {}),
            &reference,
            &other_root,
            65,
            "SEV-SNP report",
        ),
        (
            "launch digest not predicted",
            g.evidence(|s| s.measurement = [0x4E; 48]),
            &reference,
            &anchors,
            65,
            "not this image",
        ),
        (
            "REPORT_DATA of another binding",
            genoa_evidence(|s| s.report_data = other.snp_report_data()),
            &reference,
            &anchors,
            88,
            "REPORT_DATA",
        ),
        (
            "migration agent",
            g.evidence(|s| s.policy |= POLICY_MIGRATE_MA),
            &reference,
            &anchors,
            65,
            "MIGRATE_MA",
        ),
        (
            "VMPL 1",
            g.evidence(|s| s.vmpl = 1),
            &reference,
            &anchors,
            65,
            "VMPL",
        ),
        (
            "SMT on, forbidden by the reference",
            g.evidence(|s| s.platform_info |= 1),
            &reference,
            &anchors,
            65,
            "SMT",
        ),
    ];
    for (name, payload, r, a, code, reason) in cases {
        let e = verify(&g, &g.doc(&payload, None), r, a)
            .await
            .map(|_| ())
            .unwrap_err();
        assert_eq!(e.code, code, "{name}: {e}");
        assert!(e.reason.contains(reason), "{name}: {e}");
    }
}

/// No fallback: TDX evidence in the document, or none of SEV-SNP, is
/// refused before anything is fetched; another schema too.
#[tokio::test]
async fn a_document_that_is_not_snp_is_refused_under_an_snp_reference() {
    let dir = tempfile::tempdir().unwrap();
    let reference = signed_reference(dir.path(), |_| {});
    let g = Guest::new("cpu");
    // An unreachable KDS: each refusal comes before it is asked.
    let anchors = g.anchors("http://127.0.0.1:9");
    let good = g.doc(&g.evidence(|_| {}), None);
    let mut tdx = good.clone();
    tdx.tdx.quote_b64 = "AAAA".into();
    let mut log = good.clone();
    log.tdx.ccel_b64 = "AAAA".into();
    let mut none = good.clone();
    none.snp = None;
    let mut schema = good.clone();
    schema.schema = "x0ne-infer/attestation/v1".into();
    for (name, doc, why) in [
        ("TDX quote", tdx, "no fallback"),
        ("TDX event log", log, "no fallback"),
        ("no SEV-SNP evidence", none, "no SEV-SNP evidence"),
        ("schema", schema, "attestation schema"),
    ] {
        let e = verify(&g, &doc, &reference, &anchors)
            .await
            .map(|_| ())
            .unwrap_err();
        assert_eq!(e.code, 88, "{name}: {e}");
        assert!(e.reason.contains(why), "{name}: {e}");
    }
    let e = verify(&g, &good, &reference, &anchors)
        .await
        .map(|_| ())
        .unwrap_err();
    assert_eq!(e.code, 76, "control, then KDS unreachable: {e}");
    assert!(e.reason.contains("AMD KDS unreachable"), "{e}");
}

/// AMD's clock and AMD's CRL: a KDS off by more than 300 s is a wrong
/// clock (87); a CRL that is not the pinned ARK's for Genoa is refused (65).
#[tokio::test]
async fn the_kds_clock_and_crl_are_checked() {
    let dir = tempfile::tempdir().unwrap();
    let reference = signed_reference(dir.path(), |_| {});
    let g = Guest::new("cpu");
    let doc = g.doc(&g.evidence(|_| {}), None);
    let skewed = fake_kds(NOW + 1000, crate::snp::tests::TEST_GENOA_CRL).await;
    let e = verify(&g, &doc, &reference, &g.anchors(&skewed))
        .await
        .map(|_| ())
        .unwrap_err();
    assert_eq!(e.code, 87, "{e}");
    assert!(e.reason.contains("AMD KDS says"), "{e}");
    let turin = fake_kds(NOW, crate::snp::tests::TEST_TURIN_CRL).await;
    let e = verify(&g, &doc, &reference, &g.anchors(&turin))
        .await
        .map(|_| ())
        .unwrap_err();
    assert_eq!(e.code, 65, "{e}");
    assert!(e.reason.contains("CRL"), "{e}");
}

/// Config, binding, model and GPU: the same refusals as TDX.
#[tokio::test]
async fn config_binding_model_and_gpu_are_judged_as_for_tdx() {
    let dir = tempfile::tempdir().unwrap();
    let reference = signed_reference(dir.path(), |_| {});
    let kds = fake_kds(NOW, crate::snp::tests::TEST_GENOA_CRL).await;
    let g = Guest::new("cpu");
    let anchors = g.anchors(&kds);
    let good = g.doc(&g.evidence(|_| {}), None);
    verify(&g, &good, &reference, &anchors)
        .await
        .expect("control");
    let mut config = good.clone();
    config.config_b64 = b64(b"{}");
    let mut tls = good.clone();
    tls.binding.tls_spki_sha256_hex = hex(&[0x33; 32]);
    let mut model = good.clone();
    model.model.name = "qwen3:1.7b".into();
    let mut files = good.clone();
    files.model.files.pop();
    for (name, doc, code) in [
        ("config", config, 88),
        ("TLS key", tls, 88),
        ("model", model, 77),
        ("files", files, 77),
    ] {
        let e = verify(&g, &doc, &reference, &anchors)
            .await
            .map(|_| ())
            .unwrap_err();
        assert_eq!(e.code, code, "{name}: {e}");
    }
    let gg = Guest::new("cpu+gpu");
    let anchors = gg.anchors(&kds);
    let e = verify(
        &gg,
        &gg.doc(&gg.evidence(|_| {}), None),
        &reference,
        &anchors,
    )
    .await
    .map(|_| ())
    .unwrap_err();
    assert_eq!(e.code, 75, "no GPU: {e}");
    let other_nonce = gg.nvidia.attest(&hex(&[0x77; 32]), NOW, None);
    let e = verify(
        &gg,
        &gg.doc(&gg.evidence(|_| {}), Some(&other_nonce)),
        &reference,
        &anchors,
    )
    .await
    .map(|_| ())
    .unwrap_err();
    assert_eq!(e.code, 75, "another nonce: {e}");
}

/// The relay's loader: the schema picks the parser, after the signature
/// and the log; an unknown schema is refused (89), never guessed.
#[test]
fn the_loader_dispatches_on_the_schema_after_the_signature() {
    let dir = tempfile::tempdir().unwrap();
    let snp = signed_reference(dir.path(), |_| {});
    assert_eq!(snp.reference.tee, crate::snp_reference::TEE);
    let tdx_bytes = serde_json::to_vec(&crate::reference::tests::endorsed_reference(
        &"0".repeat(64),
        "c3-standard-4",
        crate::reference::tests::rtmr0_pins(),
    ))
    .unwrap();
    let tdx = own_dir(dir.path(), &tdx_bytes).join("x0ne-cvm-0.1.0.json");
    std::fs::write(&tdx, &tdx_bytes).unwrap();
    sign_and_log(&tdx);
    assert!(matches!(load(&tdx), Ok(Pinned::Tdx(_))));
    let mut v = serde_json::to_value(crate::snp_reference::tests::reference()).unwrap();
    v["schema"] = "x0ne-engine/snp-reference/v2".into();
    let unknown_bytes = serde_json::to_vec(&v).unwrap();
    let unknown = own_dir(dir.path(), &unknown_bytes).join("unknown.json");
    std::fs::write(&unknown, &unknown_bytes).unwrap();
    sign_and_log(&unknown);
    let e = load(&unknown).map(|_| ()).unwrap_err();
    assert_eq!(e.exit_code(), crate::reference::EXIT_UNUSABLE, "{e}");
    assert!(e.to_string().contains("schema"), "{e}");
    // Unsigned: refused before the schema is read.
    let unsigned = own_dir(dir.path(), b"unsigned").join("unsigned.json");
    std::fs::write(
        &unsigned,
        serde_json::to_vec(&crate::snp_reference::tests::reference()).unwrap(),
    )
    .unwrap();
    assert_eq!(
        load(&unsigned).map(|_| ()).unwrap_err().exit_code(),
        crate::reference::EXIT_SIGNATURE
    );
}
