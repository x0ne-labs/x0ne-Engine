//! PCK revocation: the documents say what the code does.
//!
//! Since 2026-10-18 every path that reads Intel's live TCB collateral also
//! reads Intel's PCK CRL (`tcb::verify_pck_crl`); since 2026-10-23 it reads
//! Intel's root CA CRL too (`tcb::verify_root_crl`: the PCK CA and the TCB
//! signing certificate). Until 2026-10-16 THREAT_MODEL.md said the server
//! checked PCK revocation while no code did. A failure here means a path
//! verifies the live TCB without a CRL, or a document still describes the
//! state before one.

use std::path::{Path, PathBuf};

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate sits in a workspace")
        .to_path_buf()
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(workspace().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// The published prose about the verifier. Run records are left out: they
/// are dated statements of what a past build printed.
fn documents() -> Vec<(PathBuf, String)> {
    let root = workspace();
    let mut paths = vec![
        root.join("README.md"),
        root.join("SECURITY.md"),
        root.join("x0ne-infer/THREAT_MODEL.md"),
    ];
    let mut docs: Vec<PathBuf> = std::fs::read_dir(root.join("docs"))
        .expect("docs/ is readable")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    docs.sort();
    paths.extend(docs);
    paths
        .into_iter()
        .map(|p| {
            let text =
                std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            (p, text)
        })
        .collect()
}

fn flat(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace("\\ ", "")
        .to_lowercase()
}

/// The body of `fn name` in `src`, to its closing brace at column 0.
fn fn_body<'a>(src: &'a str, name: &str) -> &'a str {
    let start = src
        .find(&format!("fn {name}("))
        .unwrap_or_else(|| panic!("no fn {name}"));
    let end = src[start..].find("\n}\n").expect("closing brace") + start;
    &src[start..end]
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).expect("readable") {
        let p = e.expect("entry").path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// Every caller of `tcb::judge_live` (the live TCB, since 2026-10-28; before,
/// `tcb::verify_live`) reads the PCK CRL in the same function; the TD's
/// paths go through `tcb_standing::judge_platform_live`, which does. A new
/// caller elsewhere fails until it is listed here.
#[test]
fn every_live_tcb_path_reads_the_pck_crl() {
    let paths = [
        ("x0ne-infer/src/remote_verify.rs", "verify_document"),
        ("x0ne-infer/src/tcb_standing.rs", "judge_platform_live"),
    ];
    for (file, f) in [
        ("x0ne-infer/src/attest_v2.rs", "attest_td"),
        ("x0ne-infer/src/main.rs", "run"),
        ("x0ne-infer/src/main.rs", "run_cloud"),
    ] {
        let src = read(file);
        assert!(
            fn_body(&src, f).contains("tcb_standing::judge_platform_live("),
            "{file} {f}: no live TCB through judge_platform_live"
        );
    }
    for (file, f) in paths {
        let src = read(file);
        let body = fn_body(&src, f);
        assert!(body.contains("judge_live("), "{file} {f}: no live TCB");
        for call in [
            "pck_crl_ca(",
            "pcs_pck_crl_url(",
            "verify_pck_crl(",
            "verify_root_crl(",
        ] {
            assert!(body.contains(call), "{file} {f}: no {call}");
        }
    }
    let root = workspace();
    let mut files = Vec::new();
    for krate in [
        "x0ne-infer",
        "x0ne-confidential",
        "x0ne-engine-cloud",
        "x0ne-sdk",
    ] {
        rust_files(&root.join(krate).join("src"), &mut files);
    }
    assert!(files.len() > 20, "the sweep found {} files", files.len());
    let known = [
        "tcb.rs",
        "remote_verify.rs",
        "attest_v2.rs",
        "main.rs",
        "tcb_standing.rs",
    ];
    for f in files {
        let text = std::fs::read_to_string(&f).expect("readable");
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let in_infer_src = f.starts_with(root.join("x0ne-infer/src"));
        let the_sdk = f == root.join("x0ne-sdk/src/attest.rs");
        if text.contains("verify_live(") || text.contains("judge_live(") {
            assert!(
                (in_infer_src && known.contains(&name.as_str())) || the_sdk,
                "{} verifies the live TCB: read the PCK CRL there and list it here",
                f.display()
            );
        }
    }
}

/// The SDK reads what the relay reads, in its order; its caller fetches the
/// CRLs, so no URL is built there. Its `pck_crl_ca` is held by behaviour:
/// `e2e_sdk_tests::a_pck_under_another_ca_of_the_root_is_refused_at_revocation`.
#[test]
fn the_sdk_reads_the_crls_the_relay_reads_in_its_order() {
    let src = read("x0ne-sdk/src/attest.rs");
    let body = fn_body(&src, "checked_quote");
    let mut last = 0;
    for call in [
        "verify_tdx_chain(",
        "judge_live(",
        "verify_pck_crl(",
        "verify_root_crl(",
    ] {
        let at = body
            .find(call)
            .unwrap_or_else(|| panic!("x0ne-sdk checked_quote: no {call}"));
        assert!(
            at > last,
            "x0ne-sdk checked_quote: {call} out of the relay's order"
        );
        last = at;
    }
}

#[test]
fn the_verdict_states_what_is_read_and_what_is_not() {
    let live = flat(&x0ne_infer::tcb::LIVE_LIMITATIONS.join(" "));
    assert!(live.contains("checked against intel's pck crl"), "{live}");
    assert!(live.contains("root ca crl"), "{live}");
    let client = flat(&x0ne_infer::remote_verify::LIMITATIONS.join(" "));
    assert!(
        client.contains("checked against intel's pck crl"),
        "{client}"
    );
    assert!(client.contains("root ca crl"), "{client}");
}

/// The sentences of the state before 2026-10-18, anywhere a reader looks.
#[test]
fn no_document_says_the_pck_crl_is_not_read() {
    let up = read("x0ne-engine-cloud/src/up.rs");
    let limits = &up[up.find("pub const LIMITS").expect("up.rs LIMITS")..];
    let mut texts = vec![
        (
            "up.rs LIMITS".to_string(),
            flat(&limits[..limits.find("];").unwrap()]),
        ),
        (
            "tcb::LIVE_LIMITATIONS".to_string(),
            flat(&x0ne_infer::tcb::LIVE_LIMITATIONS.join(" ")),
        ),
        (
            "remote_verify::LIMITATIONS".to_string(),
            flat(&x0ne_infer::remote_verify::LIMITATIONS.join(" ")),
        ),
    ];
    texts.extend(
        documents()
            .into_iter()
            .map(|(p, t)| (p.display().to_string(), flat(&t))),
    );
    for (name, text) in &texts {
        for stale in [
            "pck chain revocation not checked",
            "pck certificate revocation is not checked",
            "intel pck crls and",
            "no code reads intel's pck crl",
            "not checked: pck revocation",
            "**pck revocation**: not checked",
        ] {
            assert!(!text.contains(stale), "{name} still says {stale:?}");
        }
    }
    for rel in [
        "docs/attested-channel.md",
        "x0ne-infer/THREAT_MODEL.md",
        "docs/cloud.md",
    ] {
        assert!(
            flat(&read(rel)).contains("pck crl"),
            "{rel} does not say the PCK CRL is read"
        );
    }
}

/// The sentences of the state before 2026-10-23, when the root CA CRL was
/// not read, anywhere a reader looks; and the three documents say it is.
#[test]
fn no_document_says_the_root_ca_crl_is_not_read() {
    let up = read("x0ne-engine-cloud/src/up.rs");
    let limits = &up[up.find("pub const LIMITS").expect("up.rs LIMITS")..];
    let mut texts = vec![
        (
            "up.rs LIMITS".to_string(),
            flat(&limits[..limits.find("];").unwrap()]),
        ),
        (
            "tcb::LIVE_LIMITATIONS".to_string(),
            flat(&x0ne_infer::tcb::LIVE_LIMITATIONS.join(" ")),
        ),
        (
            "remote_verify::LIMITATIONS".to_string(),
            flat(&x0ne_infer::remote_verify::LIMITATIONS.join(" ")),
        ),
        (
            "tests/VENDORED.md".to_string(),
            flat(&read("x0ne-infer/tests/VENDORED.md")),
        ),
    ];
    texts.extend(
        documents()
            .into_iter()
            .map(|(p, t)| (p.display().to_string(), flat(&t))),
    );
    for (name, text) in &texts {
        for stale in [
            "root ca crl is not read",
            "root ca crl is not checked",
            "root ca crl and the nras chain's ocsp are not read",
            "root ca crl (the pck ca certificates) and the nras chain's ocsp are not read",
            "root ca crl and the ocsp of nvidia's chain are not checked",
            "root ca crl and the ocsp of the nras token's own chain are not read",
            "root ca crl, which would revoke a pck ca certificate, is not read",
            "not read: intel's root ca crl",
        ] {
            assert!(!text.contains(stale), "{name} still says {stale:?}");
        }
    }
    for rel in [
        "docs/attested-channel.md",
        "x0ne-infer/THREAT_MODEL.md",
        "docs/cloud.md",
    ] {
        assert!(
            flat(&read(rel)).contains("root ca crl"),
            "{rel} does not say the root CA CRL is read"
        );
    }
}
