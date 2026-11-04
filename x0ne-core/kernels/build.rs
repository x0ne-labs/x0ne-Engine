//! Compiles cuda/x0ne_kernels.cu to one cubin per target architecture with nvcc.
//! No PTX and no NVRTC: the SASS that runs is fixed here, and its set is hashed
//! into X0NE_CORE_KERNEL_SET_SHA256 (recipe in src/lib.rs, `kernel_set_sha256`).

use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Order matters: it is the order of the hash and of `CUBINS`.
const ARCHS: &[&str] = &["sm_90a", "sm_100a", "sm_120a"];
const UNIT: &str = "cuda/x0ne_kernels.cu";

fn nvcc() -> PathBuf {
    if let Some(p) = env::var_os("NVCC") {
        return p.into();
    }
    if let Some(home) = env::var_os("CUDA_HOME") {
        return Path::new(&home).join("bin").join("nvcc");
    }
    PathBuf::from("nvcc")
}

fn main() {
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    for entry in fs::read_dir("cuda").expect("crates/kernels/cuda must exist") {
        println!(
            "cargo:rerun-if-changed={}",
            entry.expect("readable cuda/ entry").path().display()
        );
    }

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let nvcc = nvcc();
    let version = Command::new(&nvcc)
        .arg("--version")
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "cannot run {} ({e}); kernels are built to cubin only, nvcc is required",
                nvcc.display()
            )
        });
    assert!(
        version.status.success(),
        "{} --version failed",
        nvcc.display()
    );
    let version = String::from_utf8_lossy(&version.stdout).into_owned();
    let release = version
        .lines()
        .find(|l| l.contains("release"))
        .unwrap_or("unknown")
        .trim()
        .to_owned();

    let mut set = Sha256::new();
    for arch in ARCHS {
        let dst = out.join(format!("x0ne_kernels.{arch}.cubin"));
        let status = Command::new(&nvcc)
            .arg("-cubin")
            .arg(format!("--gpu-architecture={arch}"))
            .args([
                "-O3",
                "-std=c++17",
                "-Werror",
                "all-warnings",
                "-Icuda",
                "-o",
            ])
            .arg(&dst)
            .arg(UNIT)
            .status()
            .expect("spawn nvcc");
        assert!(status.success(), "nvcc failed for {arch}");
        let bytes = fs::read(&dst).expect("read cubin");
        set.update(arch.as_bytes());
        set.update((bytes.len() as u64).to_le_bytes());
        set.update(&bytes);
    }
    println!(
        "cargo:rustc-env=X0NE_CORE_KERNEL_SET_SHA256={:x}",
        set.finalize()
    );
    println!("cargo:rustc-env=X0NE_CORE_NVCC_RELEASE={release}");
}
