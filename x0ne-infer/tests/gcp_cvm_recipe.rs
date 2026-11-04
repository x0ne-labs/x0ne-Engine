//! scripts/gcp-cvm.sh against a fake `gcloud` (tests/gcp_cvm/): no cloud, no
//! VM, no network. Each phase runs as far as a host without TDX or a GPU lets
//! it; what only a real VM does (packages, reboot, GPU) the fake refuses with
//! 255. The server is this build's x0ne-infer, put where `build` leaves it:
//! with the registers the operator pinned it stops with its own exit code (64
//! here, no TDX), never the 87 of a release build with nothing pinned, and
//! `run` says that code at once instead of 65 after two minutes.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A temporary operator machine (`home`, `tmp`, `path`) and VM (`vm`).
struct Rig {
    dir: tempfile::TempDir,
    port: u16,
}

impl Rig {
    fn new() -> Rig {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        for sub in [
            "home",
            "tmp",
            "path",
            "vm-bin",
            "vm/tmp",
            "vm/.cargo",
            "vm/model",
        ] {
            std::fs::create_dir_all(d.join(sub)).unwrap();
        }
        std::fs::write(d.join("vm/.cargo/env"), "").unwrap();
        // Copied with their mode set here: a checkout may not keep it.
        let fakes = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/gcp_cvm");
        let copy = |from: PathBuf, to: PathBuf| {
            std::fs::copy(&from, &to).unwrap();
            std::fs::set_permissions(&to, std::fs::Permissions::from_mode(0o755)).unwrap();
        };
        copy(fakes.join("gcloud"), d.join("path/gcloud"));
        for shim in ["sudo", "cargo", "curl"] {
            copy(fakes.join("vm-bin").join(shim), d.join("vm-bin").join(shim));
        }
        Rig {
            dir,
            port: free_port(),
        }
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.dir.path().join(rel)
    }

    /// `gcp-cvm.sh <phase>`: exit code, stdout and stderr, wall clock.
    fn phase(&self, phase: &str, vars: &[(&str, &str)]) -> (i32, String, Duration) {
        let mut cmd = Command::new("bash");
        cmd.arg(workspace().join("x0ne-infer/scripts/gcp-cvm.sh"))
            .arg(phase)
            .env_clear()
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.path("path").display()),
            )
            .env("HOME", self.path("home"))
            .env("TMPDIR", self.path("tmp"))
            .env("FAKE_VM_HOME", self.path("vm"))
            .env("FAKE_VM_BIN", self.path("vm-bin"))
            .env("FAKE_GCLOUD_LOG", self.path("gcloud.log"))
            .env("FAKE_INFER_BIN", env!("CARGO_BIN_EXE_x0ne-infer"))
            .env("X0NE_INFER_OUT", self.path("out"))
            .env("X0NE_INFER_PORT", self.port.to_string())
            .current_dir(self.path("home"));
        for (k, v) in vars {
            cmd.env(k, v);
        }
        let t = Instant::now();
        let out = cmd.output().expect("bash");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (out.status.code().unwrap_or(-1), text, t.elapsed())
    }

    fn gcloud_log(&self) -> String {
        std::fs::read_to_string(self.path("gcloud.log")).unwrap_or_default()
    }

    fn ssh_calls(&self) -> usize {
        self.gcloud_log()
            .lines()
            .filter(|l| l.starts_with("compute ssh "))
            .count()
    }
}

/// Two registers of the shape `measure` prints (values made up here).
fn pinned() -> String {
    format!("rtmr1={} rtmr2={}", "a1".repeat(48), "b2".repeat(48))
}

#[test]
fn every_phase_runs_until_only_a_real_vm_is_missing() {
    let rig = Rig::new();

    let (code, out, _) = rig.phase("create", &[]);
    assert_eq!(code, 0, "{out}");
    let create = rig.gcloud_log();
    for flag in [
        "compute instances create x0ne-infer-cvm ",
        "--confidential-compute-type=TDX",
        "--provisioning-model=SPOT",
        "--instance-termination-action=STOP",
        "--max-run-duration=120m",
        "--discard-local-ssds-at-termination-timestamp=true",
    ] {
        assert!(create.contains(flag), "create lacks {flag}: {create}");
    }

    let (code, out, _) = rig.phase("wait-ssh", &[]);
    assert_eq!((code, out.trim()), (0, "ssh OK (1)"), "{out}");

    // Packages, the GPU: a real VM only.
    for phase in ["driver", "checks", "arm"] {
        let (code, out, _) = rig.phase(phase, &[]);
        assert_eq!(code, 255, "{phase}: {out}");
        assert!(out.contains("needs a real VM"), "{phase}: {out}");
    }

    let (code, out, _) = rig.phase("build", &[]);
    assert_eq!(code, 0, "{out}");
    assert_eq!(
        std::fs::read_to_string(rig.path("vm/cargo.argv")).unwrap(),
        "build --release --locked -p x0ne-infer --features nvml\n"
    );
    assert!(rig.path("vm/x0ne-infer").is_file());
    let scp = rig
        .gcloud_log()
        .lines()
        .find(|l| l.starts_with("compute scp "))
        .expect("build copies the source")
        .to_string();
    let tmp = rig.path("tmp").display().to_string();
    assert!(
        scp.starts_with(&format!("compute scp {tmp}/")),
        "the source tarball is written under TMPDIR: {scp}"
    );
    assert_eq!(
        std::fs::read_dir(rig.path("tmp")).unwrap().count(),
        0,
        "the source tarball is removed once copied"
    );

    // No TDX here: the quote cannot be read.
    let (code, out, _) = rig.phase("measure", &[]);
    assert_eq!(code, 64, "{out}");
    assert!(out.contains("configfs-tsm"), "{out}");

    // Nothing pinned: refused before ssh, and told how to pin.
    let before = rig.ssh_calls();
    let (code, out, took) = rig.phase("run", &[]);
    assert_eq!(code, 2, "{out}");
    assert!(out.contains("X0NE_INFER_EXPECT"), "{out}");
    assert!(took < Duration::from_secs(5), "{took:?}");
    assert_eq!(rig.ssh_calls(), before, "run reached the VM: {out}");

    // Pinned: the server runs, finds no TDX and stops with 64, said at once.
    let (code, out, took) = rig.phase("run", &[("X0NE_INFER_EXPECT", &pinned())]);
    assert_eq!(code, 64, "{out}");
    assert!(out.contains("x0ne-infer exited 64"), "{out}");
    assert!(out.contains("no evidence, not serving"), "{out}");
    assert!(!out.contains("refused by a release build"), "{out}");
    assert!(took < Duration::from_secs(60), "{took:?}");

    let (code, out, _) = rig.phase("fetch", &[]);
    assert_eq!(code, 0, "{out}");
    let log = std::fs::read_to_string(rig.path("out/infer.log")).unwrap();
    assert!(log.contains("no evidence, not serving"), "{log}");

    let (code, out, _) = rig.phase("stop", &[]);
    assert_eq!(code, 0, "{out}");
    assert!(rig.gcloud_log().contains(
        "compute instances stop x0ne-infer-cvm --zone=us-central1-a --discard-local-ssd=true"
    ));
    let (code, out, _) = rig.phase("delete", &[]);
    assert_eq!(code, 0, "{out}");
    assert!(rig
        .gcloud_log()
        .contains("compute instances delete x0ne-infer-cvm --zone=us-central1-a --quiet"));
}

/// A register list that is not `<reg>=<96 hex>` words is refused with 2
/// before anything reaches the VM, as an empty one is.
#[test]
fn malformed_pins_are_refused_before_ssh() {
    let rig = Rig::new();
    let good = "c3".repeat(48);
    for bad in [
        "rtmr1=abc".to_string(),
        format!("rtmr9={good}"),
        format!("rtmr1={good};touch /tmp/x"),
        format!("rtmr1={good} rtmr2"),
        "  ".to_string(),
    ] {
        for phase in ["console", "run"] {
            let (code, out, _) = rig.phase(phase, &[("X0NE_INFER_EXPECT", &bad)]);
            assert_eq!(code, 2, "{phase} {bad:?}: {out}");
            assert!(out.contains("X0NE_INFER_EXPECT"), "{out}");
        }
    }
    assert_eq!(rig.ssh_calls(), 0, "{}", rig.gcloud_log());
    let pins = pinned();
    let (code, out, _) = rig.phase(
        "run",
        &[("X0NE_INFER_EXPECT", &pins), ("X0NE_INFER_PORT", "84 77")],
    );
    assert_eq!(code, 2, "{out}");
    assert!(out.contains("X0NE_INFER_PORT"), "{out}");
    assert_eq!(rig.ssh_calls(), 0, "{}", rig.gcloud_log());
}
