//! Google's real TDX firmware binary against the real boots — ignored by
//! default: the 2 MiB binary is Google's and is not vendored. Run:
//!
//! ```bash
//! curl -fsSL -o /tmp/53c558fe.fd https://storage.googleapis.com/gce_tcb_integrity/ovmf_x64_csm/53c558fe06714203d7428484c46ee7c0b9df748a9789e173421c28d097115b30efb8476304617532488ab40c3122db88.fd
//! X0NE_GCE_FIRMWARE=/tmp/53c558fe.fd cargo test -p x0ne-infer --test gce_firmware_live -- --ignored
//! ```
//!
//! The binary Google's signed endorsement of MRTD c1ee9c16 names must
//! measure that MRTD, the one the GCE `a3-highgpu-1g` boots of 2026-10-17 and
//! -18 quoted, and its CFV must be the firmware event of RTMR0 in all three
//! vendored GCE logs. Passed on 2026-10-22 with the binary fetched that day.

use x0ne_infer::{ccel, gce_endorsement, tdvf};

const MRTD: &str = "c1ee9c16e3afc506cfe042c5b846a368528f3b37618eafb27469bc114cf914e9222c91618470e7f2b28ac360968270a5";
const CFV: &str = "4e1221cf6f7815beec39a994bff99142ed305210e2735b1134fc8ef363923cbdcca91fa71e2e25ee8e689fb7d2c075bf";
/// Inside the window of the endorsement's signer, `GCE-uefi-signer` serial 30
/// (2026-05-05 to 2031-05-05).
const AT: u64 = 1_792_000_000;

fn read(rel: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/{rel}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

#[test]
#[ignore = "needs X0NE_GCE_FIRMWARE, Google's binary 53c558fe... (2 MiB)"]
fn googles_binary_measures_the_quoted_mrtd_and_the_logged_cfv() {
    let path = std::env::var("X0NE_GCE_FIRMWARE").expect("X0NE_GCE_FIRMWARE");
    let firmware = std::fs::read(&path).unwrap();
    let endorsement = gce_endorsement::verify(
        &read("gce_endorsement_tdx_c1ee9c16.binarypb"),
        gce_endorsement::ROOT_1_DER,
        AT,
    )
    .expect("Google's endorsement verifies");
    let mrtd: [u8; 48] = x0ne_infer::cloud_config::unhex(MRTD).unwrap();
    let fw = tdvf::verify_endorsed(&firmware, &endorsement, &mrtd).expect("the endorsed binary");
    assert_eq!(fw.mrtd, mrtd);
    assert_eq!(ccel::hex(&fw.cfv_sha384), CFV);
    for day in ["14", "17", "18"] {
        let log = ccel::parse(&read(&format!("ccel/gce-a3-tdx-2026-10-{day}.bin"))).unwrap();
        let logged = log
            .iter()
            .filter(|e| e.rtmr() == Some(0))
            .filter_map(|e| e.sha384())
            .filter(|d| *d == fw.cfv_sha384)
            .count();
        assert_eq!(logged, 1, "2026-10-{day}: the CFV once in RTMR0");
    }
}
