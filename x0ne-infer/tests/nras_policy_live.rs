//! The per-GPU NRAS policy (`nras::gpu_policy`) accepts the claims of the
//! production tokens of the cloud runs of 2026-10-17 and 2026-10-18, as NRAS signed
//! them (provenance in VENDORED.md). The refusals are tested on signed fixtures in `nras.rs`.

const CLAIMS: [(&str, &str); 2] = [
    (
        "2026-10-17",
        include_str!("nras_live/gpu0_2026-10-17.claims.json"),
    ),
    (
        "2026-10-18",
        include_str!("nras_live/gpu0_2026-10-18.claims.json"),
    ),
];

#[test]
fn the_production_tokens_of_two_runs_meet_the_gpu_policy() {
    for (day, json) in CLAIMS {
        let c: serde_json::Value = serde_json::from_str(json).expect("JSON claims");
        assert_eq!(c["measres"], "success", "{day}");
        for chain in x0ne_infer::nras::CERT_CHAINS_PER_GPU {
            assert!(c[*chain].is_object(), "{day}: {chain} is {}", c[*chain]);
        }
        x0ne_infer::nras::gpu_policy(&c).unwrap_or_else(|e| panic!("{day}: {e}"));
        let mut revoked = c.clone();
        revoked["x-nvidia-gpu-vbios-rim-cert-chain"]["x-nvidia-cert-ocsp-status"] =
            serde_json::json!("revoked");
        assert!(x0ne_infer::nras::gpu_policy(&revoked).is_err(), "{day}");
    }
}
