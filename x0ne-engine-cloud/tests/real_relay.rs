//! `up` with the real `x0ne-engine-proxy`, which `cargo test --workspace`
//! builds next to this crate's binary: each line the relay reports reaches
//! the terminal once. The fake relays print as it does unless `--quiet`.

mod common;

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use common::*;

/// The built relay next to the binary under test: linked, or copied 0755
/// when the build left it group- or other-writable (`up` refuses that).
fn real_relay(env: &Env) {
    let built = Path::new(BIN).with_file_name("x0ne-engine-proxy");
    let mode = fs::metadata(&built)
        .unwrap_or_else(|e| {
            panic!(
                "{}: {e}; `cargo test --workspace` builds it",
                built.display()
            )
        })
        .permissions()
        .mode();
    let dest = env.bin.with_file_name("x0ne-engine-proxy");
    let _g = fork_lock();
    if mode & 0o022 != 0 || fs::hard_link(&built, &dest).is_err() {
        fs::copy(&built, &dest).unwrap();
        fs::set_permissions(&dest, fs::Permissions::from_mode(0o755)).unwrap();
    }
}

/// A v2 reference the real relay loads, values fake, its `.sig` by the
/// TEST archive key and its `.tlog-proof` in the TEST transparency log a
/// debug `cargo test --workspace` relay pins.
/// Signed with `archive_key::test_key::sign` at CREATED + 3600 (2026-10-17),
/// logged with `transparency::test_log::log_reference` at 2026-10-22T00:00:00Z.
fn reference(env: &Env) -> PathBuf {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/real_relay");
    let p = env.path("x0ne-cvm-0.1.0.json");
    for suffix in ["", ".sig", ".tlog-proof"] {
        let name = format!("x0ne-cvm-0.1.0.json{suffix}");
        fs::copy(src.join(&name), env.path(&name)).unwrap();
    }
    p
}

fn cloud() -> Cloud {
    let mut c = Cloud::default();
    c.image_descriptions.insert(
        "x0ne-cvm-0-1-0".into(),
        format!("x0ne CVM image; disk.tar.gz sha256 {}", "0".repeat(64)),
    );
    c
}

/// The relay's `connecting` progress: printed by `up`, indented, and by
/// nothing else. The TD never answers; Ctrl-C deletes the VM.
#[test]
fn the_real_relay_progress_reaches_the_terminal_once() {
    let gcp = FakeGcp::start(cloud());
    let env = Env::new(None);
    real_relay(&env);
    let r = reference(&env);
    let mut cmd = env.command(&gcp.base, &up_args(&["--reference", r.to_str().unwrap()]));
    let mut up = spawn(&env, &mut cmd, Some(TOKEN), "up", None);
    // `up`'s copy: the relay writes its own, if any, before the event.
    up.wait_stderr(&format!("\n      waiting for the TD at {VM_IP}:443"));
    up.signal(libc::SIGINT);
    let d = up.finish();
    assert_eq!(d.code, 130, "{}", d.err);
    assert_eq!(
        d.err.matches("waiting for the TD at").count(),
        1,
        "{}",
        d.err
    );
    assert!(!d.err.contains("x0ne-engine-proxy: "), "{}", d.err);
    assert_cleaned(&gcp, &vm_name(&gcp));
}

/// A refusal before the config (an unsigned reference, 99): the relay's
/// reason once, with its code; `up` ends with 95 and nothing created.
#[test]
fn a_refusal_of_the_real_relay_reaches_the_terminal_once() {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(None);
    real_relay(&env);
    // `up` reads the image name from it; the relay refuses it unsigned (99).
    let r = env.path("x0ne-cvm-0.1.0.json");
    fs::write(
        &r,
        r#"{"schema":"x0ne-engine/reference/v2","provider":"gcp","tee":"intel-tdx",
            "image":{"name":"x0ne-cvm","version":"0.1.0","gcp_image":"x0ne-cvm-0-1-0"}}"#,
    )
    .unwrap();
    let mut cmd = env.command(&gcp.base, &up_args(&["--reference", r.to_str().unwrap()]));
    let d = spawn(&env, &mut cmd, Some(TOKEN), "up", None).finish();
    assert_eq!(d.code, 95, "{}", d.err);
    assert_eq!(d.err.matches("image reference: ").count(), 1, "{}", d.err);
    assert!(
        d.err
            .contains("x0ne-engine cloud: the relay refused (99): image reference: not signed by the x0ne archive key: "),
        "{}",
        d.err
    );
    assert!(!d.err.contains("x0ne-engine-proxy: "), "{}", d.err);
    no_creation(&gcp);
}
