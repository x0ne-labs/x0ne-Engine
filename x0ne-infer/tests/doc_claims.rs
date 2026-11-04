//! Three statements of THREAT_MODEL.md and docs/cloud.md, held to the code.
//!
//! - The TDX module identity (MRSIGNERSEAM) is checked on the live path
//!   (`tcb::verify_live` calls `tdx_module_status`): no document lists it as
//!   not checked. Until 2026-10-16 THREAT_MODEL.md did.
//! - The relay checks the image reference's signature before it parses it
//!   (`reference::load`). No text says it is unchecked; and while this tree
//!   pins no archive key (`archive_key::RELEASE` is `None`), docs/cloud.md,
//!   THREAT_MODEL.md and CHANGELOG.md say so (CHANGELOG.md in a released
//!   entry, kept as published; once pinned, Unreleased says the pin), and the relay and `status
//!   --limits` name the signing ceremony; once it pins one, they, README.md
//!   and docs/attested-channel.md name its primary fingerprint instead. Until 2026-10-17 up.rs promised a
//!   check no code made. docs/attested-channel.md puts that check at the
//!   relay's start, outside `verify_document`'s list.
//! - The session key reaches `memfd_secret` through ordinary copies (the
//!   relay's request body, the TD's request body). As long as the relay makes
//!   that copy, THREAT_MODEL.md must name it next to `memfd_secret`.
//!
//! A failure means a document and the code disagree: fix whichever is wrong,
//! and this test with it.

use std::path::{Path, PathBuf};

#[path = "support/pinned_tree.rs"]
mod pinned_tree;

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate sits in a workspace")
        .to_path_buf()
}

fn read(rel: &str) -> String {
    let p = workspace().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// Whitespace flattened, lower case: prose wraps differently from file to file.
/// README.md and the documents its sections moved to on 2026-10-24.
const README_AND_MOVED: &[&str] = &[
    "README.md",
    "docs/README.md",
    "docs/api.md",
    "docs/attested-channel.md",
    "docs/hardware-runs.md",
    "docs/limitations.md",
    "docs/pinning.md",
    "docs/related-work.md",
    "docs/running.md",
    "docs/testing.md",
    "docs/verification.md",
];

fn flat(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
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
            (p, flat(&text))
        })
        .collect()
}

/// The body of `fn <name>` in `src`, to the closing brace at its own indentation
/// (rustfmt layout).
fn fn_body<'a>(src: &'a str, name: &str) -> &'a str {
    let at = src
        .find(&format!("fn {name}("))
        .unwrap_or_else(|| panic!("fn {name} not found"));
    let line = src[..at].rfind('\n').map_or(0, |i| i + 1);
    let indent = &src[line..line + src[line..].len() - src[line..].trim_start().len()];
    let close = format!("\n{indent}}}\n");
    let end = src[at..]
        .find(&close)
        .unwrap_or_else(|| panic!("fn {name}: no closing brace"));
    &src[at..at + end]
}

/// The value of `const <name>: &str = "...";` in a Rust source, its line
/// continuations joined as rustc joins them (`\` at the end of a line eats
/// the newline and the next line's indent).
fn str_const(src: &str, name: &str) -> String {
    let decl = format!("const {name}: &str = \"");
    let at = src.find(&decl).unwrap_or_else(|| panic!("no `{decl}`"));
    let body = &src[at + decl.len()..];
    let body = &body[..body.find("\";").expect("end of the literal")];
    let mut parts = body.split("\\\n");
    let mut out = parts.next().unwrap_or("").to_string();
    for p in parts {
        out.push_str(p.trim_start());
    }
    out
}

#[test]
fn no_document_lists_the_tdx_module_identity_as_not_checked() {
    let tcb = read("x0ne-infer/src/tcb.rs");
    assert!(
        fn_body(&tcb, "verify_live").contains("rate(")
            && fn_body(&tcb, "judge_live").contains("rate(")
            && fn_body(&tcb, "rate").contains("tdx_module_status("),
        "verify_live and judge_live no longer check the TDX module: update THREAT_MODEL.md and this test"
    );
    let verdict = read("x0ne-infer/src/remote_verify.rs");
    assert!(verdict.contains("TDX module identity matched"));

    let mut claims = Vec::new();
    for (path, text) in documents() {
        // A "not checked" list runs to the end of its sentence or table cell.
        for (i, _) in text.match_indices("not checked") {
            let tail = &text[i..];
            let end = [". ", " | "]
                .iter()
                .filter_map(|stop| tail.find(stop))
                .min()
                .unwrap_or(tail.len());
            let list = &tail[..end];
            if list.contains("mrsignerseam") || list.contains("tdx module identity") {
                claims.push(format!("{}: {list}", path.display()));
            }
        }
    }
    assert!(
        claims.is_empty(),
        "these lists say the TDX module identity is not checked:\n{}",
        claims.join("\n")
    );
}

#[test]
fn the_reference_signature_is_described_as_the_code_checks_it() {
    let reference = read("x0ne-infer/src/reference.rs");
    assert!(
        fn_body(&reference, "load").contains("load_signed(")
            && fn_body(&reference, "load_signed").contains("admit_signed(&bytes")
            && fn_body(&reference, "admit_signed").contains(".verify(bytes, signature"),
        "reference::load no longer checks the signature: rewrite LIMITATIONS, \
         up.rs LIMITS, docs/cloud.md, THREAT_MODEL.md and this test"
    );
    let up = read("x0ne-engine-cloud/src/up.rs");
    let limits = &up[up.find("pub const LIMITS").expect("up.rs LIMITS")..];
    let limits = flat(&limits[..limits.find("];").unwrap()]);
    let mut texts = vec![
        (
            "remote_verify::LIMITATIONS".to_string(),
            flat(&x0ne_infer::remote_verify::LIMITATIONS.join(" ")),
        ),
        (
            "x0ne-engine-cloud/src/up.rs LIMITS".to_string(),
            limits.clone(),
        ),
    ];
    texts.extend(
        documents()
            .into_iter()
            .map(|(p, t)| (p.display().to_string(), t)),
    );
    for (name, text) in &texts {
        for stale in [
            "no signature is checked",
            "a test key today",
            "signed by the x0ne repository key",
            "the line on image references is wrong",
            "no code checks a signature on a reference",
            "unsigned yet",
            "no archive key exists yet",
            "it is not generated yet",
        ] {
            assert!(
                !text.contains(stale),
                "{name} still says {stale:?}, against reference::load"
            );
        }
    }

    // While no key is pinned, the user reads it where the relay is described.
    if let Err(e) = pinned_tree::archive_key_status(&read) {
        panic!("{e}");
    }
    let key = read("x0ne-infer/src/archive_key.rs");
    let unpinned = key.contains("pub const RELEASE: Option<Pin> = None;");
    // The signature is checked when the relay starts (proxy main), not in
    // verify_document: the attested channel's ordered list of its checks omits it.
    let remote_verify = read("x0ne-infer/src/remote_verify.rs");
    let verify_document = fn_body(&remote_verify, "verify_document");
    assert!(
        !verify_document.contains("archive_key") && !verify_document.contains("verify_detached"),
        "verify_document checks the reference's signature: docs/attested-channel.md lists it, fix this test"
    );
    let channel = read("docs/attested-channel.md");
    let list = &channel[channel
        .find("in this order (`remote_verify::verify_document`)")
        .expect("docs/attested-channel.md lists verify_document's checks")..];
    let list = flat(&list[..list.find("Only then does").expect("end of the list")]);
    assert!(
        !list.contains("archive key"),
        "docs/attested-channel.md lists the reference signature among verify_document's checks"
    );

    // NO_KEY is what the relay prints (proxy main, LoadError::NoKey); the
    // file's tests quote the phrase too, so the constant itself is read.
    let no_key = str_const(&key, "NO_KEY");
    assert!(
        no_key.contains("built before the signing ceremony"),
        "the relay's refusal (archive_key::NO_KEY) names the signing ceremony: {no_key}"
    );
    // status --limits: the ceremony while nothing is pinned, the key after it
    // (scripts/pin-archive-key.sh writes both states' texts).
    if unpinned {
        assert!(
            limits.contains("built before the signing ceremony"),
            "status --limits names the signing ceremony"
        );
        return;
    }
    let pin = &key[key.find("pub const RELEASE: Option<Pin> = Some(").unwrap()..];
    let fpr = pin
        .split("primary_fingerprint: \"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .expect("RELEASE names its primary fingerprint")
        .to_ascii_lowercase();
    assert!(
        fpr.len() == 40 && fpr.bytes().all(|b| b.is_ascii_hexdigit()),
        "RELEASE primary fingerprint {fpr:?}"
    );
    assert!(
        !limits.contains("built before the signing ceremony"),
        "status --limits says this build pins no key"
    );
    assert!(limits.contains(&fpr), "status --limits does not name {fpr}");
    for rel in [
        "docs/cloud.md",
        "docs/attested-channel.md",
        "x0ne-infer/THREAT_MODEL.md",
        "CHANGELOG.md",
        "README.md",
    ] {
        assert!(flat(&read(rel)).contains(&fpr), "{rel} does not name {fpr}");
    }
}

#[test]
fn the_ordinary_copies_of_the_session_key_are_named_next_to_memfd_secret() {
    let relay = read("x0ne-infer/src/remote_verify.rs");
    if !fn_body(&relay, "post_key").contains("body.to_string().into_bytes()") {
        return;
    }
    let tm = flat(&read("x0ne-infer/THREAT_MODEL.md"));
    let rows: Vec<&str> = tm
        .split(" | ")
        .filter(|cell| cell.contains("memfd_secret"))
        .collect();
    assert!(
        !rows.is_empty(),
        "THREAT_MODEL.md no longer names memfd_secret"
    );
    let text = rows.join(" ");
    assert!(
        text.contains("post_key") && text.contains("axum"),
        "THREAT_MODEL.md puts the session key in memfd_secret without naming \
         the ordinary copies on its way in (remote_verify post_key, the TD's axum body)"
    );
}

/// archive_key.rs has no compile_error! on `test-archive-key` (2026-10-17):
/// the release test build carries the feature and keeps `RELEASE`. Cargo.toml
/// still said archive_key.rs refused to compile without debug assertions.
#[test]
fn no_text_says_the_test_key_feature_refuses_to_compile() {
    assert!(!read("x0ne-infer/src/archive_key.rs").contains("compile_error!"));
    let toml: String = read("x0ne-infer/Cargo.toml")
        .lines()
        .map(|l| l.trim_start().trim_start_matches('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let mut texts = vec![("x0ne-infer/Cargo.toml", flat(&toml))];
    for rel in README_AND_MOVED.iter().copied().chain([
        "CHANGELOG.md",
        "docs/cloud.md",
        "x0ne-infer/THREAT_MODEL.md",
    ]) {
        texts.push((rel, flat(&read(rel))));
    }
    for (rel, text) in texts {
        for sentence in text.split(". ") {
            let about_the_key = ["test-archive-key", "archive_key.rs", "test archive key"]
                .iter()
                .any(|k| sentence.contains(k));
            assert!(
                !(about_the_key && sentence.contains("compile")),
                "{rel}: {sentence}"
            );
        }
    }
}

/// `cargo test --release` alone also builds x0ne-engine-cloud's tests, whose
/// `test-endpoints` refuses a release build (http.rs, measured 2026-10-17):
/// a document names the package make ci tests in release.
#[test]
fn the_release_test_build_is_named_with_its_package() {
    let http = read("x0ne-engine-cloud/src/http.rs");
    assert!(
        http.contains(
            "#[cfg(all(feature = \"test-endpoints\", not(debug_assertions)))]\ncompile_error!"
        ),
        "x0ne-engine-cloud builds its tests in release now: update this test"
    );
    for rel in README_AND_MOVED.iter().copied().chain([
        "CHANGELOG.md",
        "docs/cloud.md",
        "x0ne-infer/THREAT_MODEL.md",
        "x0ne-infer/Cargo.toml",
    ]) {
        let text = flat(&read(rel));
        for (i, _) in text.match_indices("cargo test --release") {
            assert!(
                text[i..].starts_with("cargo test --release -p x0ne-infer"),
                "{rel}: {}",
                &text[i..text.len().min(i + 80)]
            );
        }
    }
}

/// docs/pinning.md gives RTMR1 in the order the real GCE boots measured: the
/// logs of 2026-10-14 (attempt 3) and 2026-10-17 here; attempt 9 of 2026-10-14, kept
/// in x0ne-OS, has the same order. Until 2026-10-17 it put the separator first.
#[test]
fn the_pinning_doc_gives_rtmr1_in_the_order_of_the_real_boots() {
    use x0ne_infer::ccel::{
        self, EV_EFI_ACTION, EV_EFI_BOOT_SERVICES_APPLICATION, EV_EFI_GPT_EVENT, EV_SEPARATOR,
    };
    let want = [
        "calling efi application from boot option",
        "separator",
        "gpt",
        "uki",
        "kernel pe",
        "exit boot services invocation",
        "exit boot services returned with success",
    ];
    for log in ["gce-a3-tdx-2026-10-14.bin", "gce-a3-tdx-2026-10-17.bin"] {
        let bytes = std::fs::read(workspace().join("x0ne-infer/tests/ccel").join(log)).unwrap();
        let mut apps = 0;
        let seen: Vec<String> = ccel::parse(&bytes)
            .unwrap()
            .iter()
            .filter(|e| e.rtmr() == Some(1))
            .map(|e| match e.event_type {
                EV_EFI_ACTION => String::from_utf8_lossy(&e.data)
                    .trim_end_matches('\0')
                    .to_lowercase(),
                EV_SEPARATOR => "separator".into(),
                EV_EFI_GPT_EVENT => "gpt".into(),
                EV_EFI_BOOT_SERVICES_APPLICATION => {
                    apps += 1;
                    if apps == 1 { "uki" } else { "kernel pe" }.into()
                }
                t => format!("{t:#x}"),
            })
            .collect();
        assert_eq!(seen, want, "{log}");
    }
    let pinning = flat(&read("docs/pinning.md"));
    let at = pinning
        .find("rtmr1, in this order")
        .expect("docs/pinning.md gives RTMR1's order");
    let text = &pinning[at..];
    let text = &text[..text.find(" rtmr2 ").expect("then RTMR2")];
    let mut from = 0;
    for w in want {
        let i = text[from..].find(w).unwrap_or_else(|| {
            panic!("docs/pinning.md's RTMR1 order lacks or misplaces {w:?}: {text}")
        });
        from += i + w.len();
    }
}

/// Every real GCE boot whose event log is kept here is a dated entry of
/// "Proven on hardware" (docs/hardware-runs.md). Until 2026-10-17 README said
/// the cloud path had run once and never against Google Cloud, after the run
/// of that day.
#[test]
fn the_hardware_runs_list_every_real_boot_kept_here() {
    let runs = read("docs/hardware-runs.md");
    let proven = &runs[runs
        .find("## Proven on hardware")
        .expect("docs/hardware-runs.md has \"Proven on hardware\"")..];
    let proven = &proven[..proven[3..].find("\n## ").map_or(proven.len(), |i| i + 3)];
    let mut dates = Vec::new();
    for e in std::fs::read_dir(workspace().join("x0ne-infer/tests/ccel")).unwrap() {
        let name = e.unwrap().file_name().to_string_lossy().into_owned();
        if let Some(d) = name
            .strip_prefix("gce-a3-tdx-")
            .and_then(|n| n.strip_suffix(".bin"))
        {
            dates.push(d.to_string());
        }
    }
    assert!(dates.len() >= 2, "{dates:?}");
    for d in &dates {
        assert!(
            proven.contains(&format!("\n- {d}, GCP `a3-highgpu-1g`")),
            "\"Proven on hardware\" has no entry for the GCE boot of {d}"
        );
    }
    for rel in README_AND_MOVED {
        for stale in [
            "has not run against google cloud",
            "it has run once on real hardware",
        ] {
            assert!(!flat(&read(rel)).contains(stale), "{rel}: {stale}");
        }
    }
    for (rel, stale) in [
        ("docs/cloud.md", "not run against google cloud yet"),
        (
            "x0ne-infer/THREAT_MODEL.md",
            "one run on real hardware since",
        ),
    ] {
        assert!(!flat(&read(rel)).contains(stale), "{rel}: {stale}");
    }
}

/// The ueid binding met production NRAS tokens on the cloud runs of
/// 2026-10-14, -17 and -18 (verdicts "bound by ueid"). Until 2026-10-18 the
/// verdict of each of those runs said it never had.
#[test]
fn the_ueid_binding_is_not_described_as_never_exercised() {
    let nras = flat(&x0ne_infer::nras::LIMITATIONS.join(" "));
    assert!(!nras.contains("not yet exercised"), "{nras}");
    assert!(
        nras.contains("2026-10-14") && nras.contains("2026-10-18"),
        "{nras}"
    );
    let src = flat(&read("x0ne-infer/src/nras.rs"));
    assert!(!src.contains("it has not yet met a production token"));
    for rel in README_AND_MOVED {
        let stale = "it has met a production nras token once";
        assert!(!flat(&read(rel)).contains(stale), "{rel}: {stale}");
    }
    let stale = "has met a production token once";
    assert!(!flat(&read("x0ne-infer/THREAT_MODEL.md")).contains(stale));
}

/// Open WebUI 0.12.3, in its default Native function calling, sends `tools`
/// with a plain chat; the relay applies the server's rule and answers 400.
/// docs/api.md names the Legacy setting exactly while that refusal stands.
#[test]
fn the_api_doc_names_the_open_webui_setting_while_tools_are_refused() {
    let body = serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    });
    let refused = match x0ne_infer::server::parse_chat_request(body) {
        Err((status, why)) => {
            status.as_u16() == 400
                && why.starts_with("tools:")
                && why.contains("rather than ignored")
        }
        Ok(_) => false,
    };
    let relay = read("x0ne-infer/src/local_api.rs");
    assert!(
        relay.contains("crate::server::parse_chat_request(body.clone())"),
        "the relay no longer applies the server's rule to a chat: update docs/api.md and this test"
    );
    let api = read("docs/api.md");
    let at = api
        .find("\n### Open WebUI\n")
        .expect("docs/api.md has an Open WebUI section");
    let section = &api[at + 1..];
    let section = flat(&section[..section[4..].find("\n#").map_or(section.len(), |i| i + 4)]);
    assert!(
        section.contains("0.12.3"),
        "docs/api.md's Open WebUI section names no tested version"
    );
    let says_legacy = section.contains("`function_calling: \"legacy\"`");
    assert_eq!(
        says_legacy, refused,
        "docs/api.md's Open WebUI section must ask for function_calling legacy exactly while \
         `tools` is refused (refused: {refused}): {section}"
    );
    if refused {
        assert!(
            section.contains("refused rather than ignored"),
            "docs/api.md's Open WebUI section does not say why: {section}"
        );
    }
}
