//! Google's TDX firmware binary, measured offline, and the RTMR0 list of a
//! `gce-endorsed` image reference.
//!
//!   gce_firmware <binary.fd> [<ccel.bin>]
//!
//! Prints the binary's SHA-384, its TDVF sections, the MRTD it measures and
//! the SHA-384 of its configuration volume (CFV). With the event log of one
//! boot on that firmware (the attestation document's `tdx.ccel_b64`,
//! base64-decoded), prints the `rtmr0.events_sha384` list for the reference:
//! the CFV as `"firmware:cfv"`, every other RTMR0 event by its digest. Exit 1
//! if the binary is not a TDVF this verifier reads or the CFV is not in the
//! log once, 2 on a usage or I/O error. Nothing is fetched: the binary is the
//! one Google's endorsement names (`ovmf_x64_csm/<sha384>.fd`).

use sha2::Digest as _;
use x0ne_infer::{ccel, reference, tdvf};

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.len() > 2 || args[0].starts_with('-') {
        eprintln!("usage: gce_firmware <binary.fd> [<ccel.bin>]");
        return 2;
    }
    let firmware = match std::fs::read(&args[0]) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{}: {e}", args[0]);
            return 2;
        }
    };
    let image = match tdvf::Image::parse(&firmware) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{}: {e}", args[0]);
            return 1;
        }
    };
    println!("sha384 {}", ccel::hex(&sha2::Sha384::digest(&firmware)));
    for s in &image.sections {
        println!(
            "{:<8} data {:#x}+{:#x} memory {:#x}+{:#x} attributes {:#x}",
            format!("{:?}", s.kind),
            s.data_offset,
            s.raw_size,
            s.memory_base,
            s.memory_size,
            s.attributes
        );
    }
    println!("mrtd   {}", ccel::hex(&image.mrtd()));
    let cfv = image.cfv_sha384();
    println!("cfv    {}", ccel::hex(&cfv));
    let Some(log_path) = args.get(1) else {
        return 0;
    };
    let log = match std::fs::read(log_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{log_path}: {e}");
            return 2;
        }
    };
    let events = match ccel::parse(&log) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{log_path}: {e}");
            return 1;
        }
    };
    match reference::rtmr0_template(&events, &cfv) {
        Ok(list) => {
            println!(
                "rtmr0.events_sha384 {}",
                serde_json::to_string_pretty(&list).expect("strings")
            );
            0
        }
        Err(e) => {
            eprintln!("{log_path}: {e}");
            1
        }
    }
}
