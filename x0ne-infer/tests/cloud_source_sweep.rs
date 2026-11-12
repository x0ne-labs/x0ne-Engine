//! No flag, no variable and no fallback sets a trust anchor or skips a check
//! on the attested-channel path (after the Harness's
//! `production_sources()` sweep). The anchors have one constructor per side,
//! `production()`, built from pinned constants; the binaries call it and
//! nothing else; the verification modules read no environment at all.
//!
//! A failure means a new way to point the verifier at another root, service
//! or document appeared. Remove it, or make it test-only.

use std::path::Path;

fn src(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The shipped code: the source without its `#[cfg(test)]` items. Cutting at
/// the first `#[cfg(test)]` lost any code after a test module.
fn shipped(name: &str) -> String {
    without_test_items(&src(name))
}

/// What no shipped build compiles: tests; the tests of a file x0ne-sdk
/// compiles too; and a bench's NRAS pin, whose cfg only x0ne-sdk's build
/// script sets, for its `test-anchor` feature
/// (`the_test_anchor_cfg_is_set_by_the_sdks_test_anchor_feature_alone`).
const NOT_SHIPPED: &[&[u8]] = &[
    b"#[cfg(test)]",
    b"#[cfg(all(test, not(x0ne_sdk)))]",
    b"#[cfg(x0ne_sdk_test_anchor)]",
];

/// `source` minus each item under one of [`NOT_SHIPPED`], attribute
/// included. An item ends at its `;` or `,`, after its outermost `{…}`, or
/// before the `}` closing the block around it; only braces that are code
/// count. Unsure (a `,` in `<…>`, another `#[cfg(all(test, …))]`), it cuts
/// less, never more.
fn without_test_items(source: &str) -> String {
    let (b, code) = (source.as_bytes(), code_mask(source));
    let (mut out, mut kept, mut i) = (String::new(), 0, 0);
    while i < b.len() {
        match NOT_SHIPPED
            .iter()
            .find(|a| code[i] && b[i..].starts_with(a))
        {
            Some(attr) => {
                out.push_str(&source[kept..i]);
                i = item_end(b, &code, i + attr.len());
                kept = i;
            }
            None => i += 1,
        }
    }
    out + &source[kept..]
}

fn item_end(b: &[u8], code: &[bool], from: usize) -> usize {
    let mut depth = 0usize; // ( and [
    for i in from..b.len() {
        if !code[i] {
            continue;
        }
        match b[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' if depth > 0 => depth -= 1,
            b')' | b']' | b'}' if depth == 0 => return i,
            b';' | b',' if depth == 0 => return i + 1,
            b'{' if depth == 0 => {
                let mut braces = 0usize;
                for j in i..b.len() {
                    match (code[j], b[j]) {
                        (true, b'{') => braces += 1,
                        (true, b'}') => {
                            braces -= 1;
                            if braces == 0 {
                                return j + 1;
                            }
                        }
                        _ => {}
                    }
                }
                return b.len();
            }
            _ => {}
        }
    }
    b.len()
}

/// Per byte: true for code, false inside a comment, a string or a char
/// literal. A lifetime (`'a`) is code.
fn code_mask(s: &str) -> Vec<bool> {
    let b = s.as_bytes();
    let mut mask = vec![true; b.len()];
    let mut i = 0;
    while i < b.len() {
        let text_end = if b[i..].starts_with(b"//") {
            Some(
                b[i..]
                    .iter()
                    .position(|&c| c == b'\n')
                    .map_or(b.len(), |p| i + p),
            )
        } else if b[i..].starts_with(b"/*") {
            let (mut depth, mut j) = (0usize, i);
            while j < b.len() {
                if b[j..].starts_with(b"/*") {
                    (depth, j) = (depth + 1, j + 2);
                } else if b[j..].starts_with(b"*/") {
                    (depth, j) = (depth - 1, j + 2);
                    if depth == 0 {
                        break;
                    }
                } else {
                    j += 1;
                }
            }
            Some(j.min(b.len()))
        } else if let Some(end) = raw_string_end(b, i) {
            Some(end)
        } else if b[i] == b'"' {
            let mut j = i + 1;
            while j < b.len() && b[j] != b'"' {
                j += if b[j] == b'\\' { 2 } else { 1 };
            }
            Some((j + 1).min(b.len()))
        } else if b[i] == b'\'' && b.get(i + 1) == Some(&b'\\') {
            let j = (i + 3).min(b.len());
            Some(
                b[j..]
                    .iter()
                    .position(|&c| c == b'\'')
                    .map_or(b.len(), |p| j + p + 1),
            )
        } else if b[i] == b'\'' {
            let len = s[i + 1..].chars().next().map_or(0, char::len_utf8);
            (len > 0 && b.get(i + 1 + len) == Some(&b'\'')).then_some(i + 2 + len)
        } else {
            None
        };
        match text_end {
            Some(end) => {
                mask[i..end].fill(false);
                i = end;
            }
            None => i += 1,
        }
    }
    mask
}

/// End of a raw string (`r"…"`, `r#"…"#`, `br…`, `cr…`) starting at `i`.
fn raw_string_end(b: &[u8], i: usize) -> Option<usize> {
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let prefixed = |k: usize| matches!(b[k], b'b' | b'c') && (k == 0 || !ident(b[k - 1]));
    if b[i] != b'r' || (i > 0 && ident(b[i - 1]) && !prefixed(i - 1)) {
        return None;
    }
    let hashes = b[i + 1..].iter().take_while(|&&c| c == b'#').count();
    if b.get(i + 1 + hashes) != Some(&b'"') {
        return None;
    }
    let mut close = vec![b'#'; hashes + 1];
    close[0] = b'"';
    let body = i + 2 + hashes;
    Some(
        b[body..]
            .windows(close.len())
            .position(|w| w == close.as_slice())
            .map_or(b.len(), |p| body + p + close.len()),
    )
}

/// The cut keeps shipped code after a test module, and counts only braces
/// that are code. A failure means a sweep reads test code as shipped, or
/// misses shipped code: what the cut at the first `#[cfg(test)]` did.
#[test]
fn the_test_item_cut_keeps_what_follows_and_skips_braces_in_text() {
    let source = r##"pub fn a() {}
#[cfg(test)]
mod tests {
    // }
    const S: &str = "}\"}";
    const R: &str = r#"}"#;
    const C: [char; 3] = ['}', '\'', '\u{7d}'];
    fn f<'a>(x: &'a str) -> &'a str { x }
    /* } /* } */ } */
}
pub fn after() {}
#[cfg(test)]
use std::fmt;
pub struct P {
    #[cfg(test)]
    t: u8,
    pub kept: u8
}
"##;
    assert_eq!(
        without_test_items(source),
        "pub fn a() {}\n\npub fn after() {}\n\npub struct P {\n    \n    pub kept: u8\n}\n"
    );
}

/// The inherent `impl <ty> {` block, to its closing brace at column 0.
fn inherent_impl(body: &str, ty: &str) -> String {
    let start = body
        .find(&format!("impl {ty} {{"))
        .unwrap_or_else(|| panic!("no `impl {ty}`"));
    let rest = &body[start..];
    rest[..rest.find("\n}\n").expect("closing brace")].to_string()
}

/// The functions a block defines, in order.
fn fn_names(block: &str) -> Vec<String> {
    block
        .lines()
        .filter_map(|l| {
            let t = l.trim_start();
            let t = t.strip_prefix("pub ").unwrap_or(t);
            let t = t.strip_prefix("pub(crate) ").unwrap_or(t);
            let t = t.strip_prefix("async ").unwrap_or(t);
            t.strip_prefix("fn ")
                .map(|r| r.split(['(', '<']).next().unwrap_or("").to_string())
        })
        .collect()
}

/// `Self {` literals of a block (not the `-> Self {` of a signature).
fn self_literals(block: &str) -> usize {
    block.matches("Self {").count() - block.matches("-> Self {").count()
}

/// Struct literals of `ty`: `ty {` that is neither a definition nor an impl.
fn literals(body: &str, ty: &str) -> usize {
    let needle = format!("{ty} {{");
    body.match_indices(&needle)
        .filter(|(i, _)| {
            let before = &body[..*i];
            !(before.ends_with("struct ") || before.ends_with("impl ") || before.ends_with("for "))
        })
        .count()
}

const VERIFICATION_MODULES: &[&str] = &[
    "remote_verify.rs",
    "attest_v2.rs",
    "tls.rs",
    "gce_endorsement.rs",
    "tdvf.rs",
    "reference.rs",
    "ccel.rs",
    "cloud_serve.rs",
    "e2e_gateway.rs",
    "cloud_config.rs",
    "cloud_config_types.rs",
    "channel_key.rs",
    "local_api.rs",
    "relay.rs",
    "binding.rs",
    "archive_key.rs",
    "openpgp.rs",
    "tlog.rs",
    "transparency.rs",
    "snp.rs",
    "snp_reference.rs",
];

#[test]
fn the_verification_modules_read_no_environment_and_no_arguments() {
    for name in VERIFICATION_MODULES {
        let body = shipped(name);
        for forbidden in [
            "std::env",
            "env::var",
            "option_env!",
            "env::args",
            "args_os",
        ] {
            assert!(
                !body.contains(forbidden),
                "{name} contains {forbidden:?}: anchors and checks must not depend on the caller's environment"
            );
        }
    }
}

/// `term_color` is the lifecycle's `color.rs`, read here from that crate:
/// presentation only, it reads the four color variables and nothing else.
/// The relay passes its decision to `relay.rs` in `Env`, which reads none.
#[test]
fn terminal_colors_read_the_color_variables_only() {
    let lib = src("lib.rs");
    assert!(
        lib.contains("#[path = \"../../x0ne-engine-cloud/src/color.rs\"]\npub mod term_color;"),
        "term_color is no longer the lifecycle's color.rs"
    );
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../x0ne-engine-cloud/src/color.rs");
    let color = std::fs::read_to_string(&path).unwrap();
    let color = &color[..color.find("#[cfg(test)]").unwrap_or(color.len())];
    let reads: Vec<&str> = color
        .lines()
        .filter(|l| {
            [
                "std::env",
                "env::var",
                "option_env!",
                "env::args",
                "args_os",
            ]
            .iter()
            .any(|f| l.contains(f))
        })
        .map(str::trim)
        .collect();
    assert_eq!(
        reads,
        [r#""NO_COLOR" | "CLICOLOR_FORCE" | "CLICOLOR" | "TERM" => std::env::var_os(name),"#]
    );
}

/// `Anchors` (client) and `TdAnchors` (TD) are built by `production()`
/// alone, from the pinned constants, and every caller uses that.
#[test]
fn anchors_are_built_by_production_alone() {
    let client = shipped("remote_verify.rs");
    let block = inherent_impl(&client, "Anchors");
    assert_eq!(
        fn_names(&block),
        ["production", "gce_root_spki_sha256_hex"],
        "impl Anchors grew a constructor"
    );
    assert_eq!(self_literals(&block), 1, "one literal, in production()");
    assert_eq!(
        literals(&client, "Anchors"),
        0,
        "remote_verify.rs builds Anchors outside production()"
    );
    for pinned in [
        "TrustAnchor::intel_production()",
        "crate::nras::NVIDIA_NRAS_GPU_INTERMEDIATE_PINS",
        "crate::gce_endorsement::ROOT_1_DER",
        "crate::tcb::PCS_BASE_URL",
        "crate::gce_endorsement::BUCKET_BASE_URL",
    ] {
        assert!(
            block.contains(pinned),
            "Anchors::production() no longer uses {pinned}"
        );
    }

    let td = shipped("attest_v2.rs");
    let block = inherent_impl(&td, "TdAnchors");
    assert_eq!(
        fn_names(&block),
        ["production"],
        "impl TdAnchors grew a constructor"
    );
    assert_eq!(self_literals(&block), 1);
    assert_eq!(
        literals(&td, "TdAnchors"),
        0,
        "attest_v2.rs builds TdAnchors outside production()"
    );
    assert!(block.contains("TrustAnchor::intel_production()"));
    assert!(block.contains("crate::nras::NVIDIA_NRAS_GPU_INTERMEDIATE_PINS"));

    for name in [
        "main.rs",
        "bin/proxy.rs",
        "relay.rs",
        "cloud_serve.rs",
        "local_api.rs",
        "cloud_config.rs",
    ] {
        let body = shipped(name);
        assert_eq!(
            literals(&body, "Anchors") + literals(&body, "TdAnchors"),
            0,
            "{name} builds anchors by hand"
        );
        for (i, _) in body.match_indices("Anchors::") {
            assert!(
                body[i..].starts_with("Anchors::production()"),
                "{name} reaches anchors other than by production()"
            );
        }
    }
    assert!(shipped("bin/proxy.rs").contains("Anchors::production()"));
    assert!(shipped("main.rs").contains("TdAnchors::production()"));
}

/// `SnpAnchors` (AMD SEV-SNP) is built by `production()` alone, from the AMD
/// KDS certificates and SPKI pins compiled in; no shipped code elsewhere
/// writes one, so a caller can drop a root (tests do) but not add its own.
#[test]
fn snp_anchors_are_built_by_production_alone() {
    let snp = shipped("snp.rs");
    let block = inherent_impl(&snp, "SnpAnchors");
    // A constructor returns Self; anything else must read an existing value.
    for name in fn_names(&block) {
        let at = block.find(&format!("fn {name}(")).expect("fn");
        let sig = &block[at..at + block[at..].find(" {").expect("body")];
        if name == "production" {
            assert!(sig.ends_with("-> Self"), "{sig}");
        } else {
            assert!(
                sig.contains("(&self") && !sig.contains("Self") && !sig.contains("SnpAnchors"),
                "impl SnpAnchors grew a constructor: {sig}"
            );
        }
    }
    assert_eq!(self_literals(&block), 1, "one literal, in production()");
    assert_eq!(
        literals(&snp, "SnpAnchors"),
        0,
        "snp.rs builds SnpAnchors outside production()"
    );
    for pinned in [
        "AMD_ARK_MILAN_SPKI_SHA256_HEX",
        "AMD_ARK_GENOA_SPKI_SHA256_HEX",
        "AMD_ARK_TURIN_SPKI_SHA256_HEX",
        "AMD_ASK_GENOA_SPKI_SHA256_HEX",
        "AMD_ASK_TURIN_SPKI_SHA256_HEX",
        "AMD_ASVK_MILAN_SPKI_SHA256_HEX",
        "AMD_ASVK_GENOA_SPKI_SHA256_HEX",
        "AMD_ASVK_TURIN_SPKI_SHA256_HEX",
        "KDS_BASE_URL",
    ] {
        assert!(
            block.contains(pinned),
            "SnpAnchors::production() no longer uses {pinned}"
        );
    }
    let mut files = Vec::new();
    sources(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        "",
        &mut files,
    );
    for name in files.iter().filter(|n| n.as_str() != "snp.rs") {
        let body = shipped(name);
        assert_eq!(
            literals(&body, "SnpAnchors") + literals(&body, "SnpRoot"),
            0,
            "{name} builds SEV-SNP anchors by hand"
        );
        for (i, _) in body.match_indices("SnpAnchors::") {
            assert!(
                body[i..].starts_with("SnpAnchors::production()"),
                "{name} reaches SEV-SNP anchors other than by production()"
            );
        }
    }
}

/// Every `.rs` under `src/`, as a path relative to it.
fn sources(dir: &Path, rel: &str, out: &mut Vec<String>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let name = format!("{rel}{}", entry.file_name().to_string_lossy());
        if entry.file_type().unwrap().is_dir() {
            sources(&entry.path(), &format!("{name}/"), out);
        } else if Path::new(&name).extension().is_some_and(|e| e == "rs") {
            out.push(name);
        }
    }
}

/// NRAS pins: two literals in shipped code, Intermediates 004 and 005
/// in `nras.rs`. Throwaway test chains live under `#[cfg(test)]`. The fields
/// are private and no constructor or field write exists: shipped code
/// elsewhere can clone a pin, not build one nor rewrite its digests.
#[test]
fn the_nras_pins_are_built_in_nras_rs_alone() {
    let mut files = Vec::new();
    sources(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        "",
        &mut files,
    );
    files.sort();
    let found: Vec<(String, usize)> = files
        .iter()
        .map(|f| (f.clone(), literals(&shipped(f), "NrasIntermediatePin")))
        .filter(|(_, n)| *n > 0)
        .collect();
    assert_eq!(found, [("nras.rs".to_string(), 2)]);
    let nras = shipped("nras.rs");
    for pin in ["004", "005"] {
        for digest in ["CERT", "SPKI"] {
            assert!(nras.contains(&format!(
                "Cow::Borrowed(NVIDIA_NRAS_GPU_INTERMEDIATE_{pin}_{digest}_SHA256_HEX)"
            )));
        }
    }
    let head = "#[derive(Debug, Clone, PartialEq, Eq)]\npub struct NrasIntermediatePin {";
    let def = &nras[nras.find(head).expect("the pin type and its derives")..];
    let def = &def[..def.find("\n}\n").expect("its end")];
    assert!(
        !def.lines()
            .skip(2)
            .any(|l| l.trim_start().starts_with("pub")),
        "a pin field is public:\n{def}"
    );
    let block = inherent_impl(&nras, "NrasIntermediatePin");
    let fields = ["cert_sha256_hex", "spki_sha256_hex", "leaf_signature"];
    assert_eq!(fn_names(&block), fields, "impl NrasIntermediatePin grew");
    assert_eq!(self_literals(&block), 0);
    assert_eq!(nras.matches("impl NrasIntermediatePin").count(), 1);
    assert!(
        !nras.contains("for NrasIntermediatePin"),
        "a trait builds pins"
    );
    for field in fields {
        assert!(
            !nras.contains(&format!(".{field} = ")),
            "nras.rs edits {field}"
        );
    }
}

/// The cfg under which `nras.rs` builds a bench's pin (`throwaway_pin`) is
/// set by x0ne-sdk's build script, for its `test-anchor` feature, and by
/// nothing else: no other build script, no cargo config, no Makefile or CI
/// script passes it in `RUSTFLAGS`.
#[test]
fn the_test_anchor_cfg_is_set_by_the_sdks_test_anchor_feature_alone() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let read = |p: &str| std::fs::read_to_string(root.join(p)).unwrap_or_default();
    let build = read("x0ne-sdk/build.rs");
    let emit = "println!(\"cargo::rustc-cfg=x0ne_sdk_test_anchor\");";
    assert_eq!(build.matches("rustc-cfg=x0ne_sdk_test_anchor").count(), 1);
    let gate = "if std::env::var_os(\"CARGO_FEATURE_TEST_ANCHOR\").is_some() {\n        ";
    assert!(
        build.contains(&format!("{gate}{emit}")),
        "the emit lost its gate"
    );
    let manifest = read("x0ne-sdk/Cargo.toml");
    assert!(!manifest.contains("default = [\"test-anchor\"]"));
    let mut places: Vec<String> = [
        ".cargo/config.toml",
        ".cargo/config",
        "Makefile",
        "ci/ci.sh",
        "ci/inside.sh",
        "scripts/sdk-wasm.sh",
        "scripts/e2e-local.sh",
    ]
    .map(String::from)
    .to_vec();
    for crate_dir in std::fs::read_dir(&root).unwrap().flatten() {
        let name = crate_dir.file_name().to_string_lossy().to_string();
        if name != "x0ne-sdk" && crate_dir.path().join("build.rs").is_file() {
            places.push(format!("{name}/build.rs"));
        }
    }
    let elsewhere: Vec<&String> = places
        .iter()
        .filter(|p| read(p).contains("x0ne_sdk_test_anchor"))
        .collect();
    assert!(elsewhere.is_empty(), "set elsewhere: {elsewhere:?}");
    let nras = src("nras.rs");
    assert_eq!(nras.matches("#[cfg(x0ne_sdk_test_anchor)]").count(), 1);
    assert!(nras.contains("#[cfg(x0ne_sdk_test_anchor)]\npub fn throwaway_pin("));
}

/// The relay binary takes no flag that names a URL, a root, a service or a
/// skip, and reads only the three directory variables.
#[test]
fn the_relay_binary_offers_no_way_to_repoint_or_skip_the_verdict() {
    let body = shipped("bin/proxy.rs");
    let flags: Vec<&str> = body
        .lines()
        .flat_map(|l| l.split('"'))
        .filter(|w| w.starts_with("--"))
        .collect();
    assert!(flags.len() >= 9, "the flag list moved: {flags:?}");
    for flag in &flags {
        for forbidden in [
            "url", "pcs", "nras", "bucket", "anchor", "root", "insecure", "skip", "verify",
            "allow", "dev", "trust",
        ] {
            assert!(
                !flag.contains(forbidden),
                "flag {flag:?} looks like a way around the verdict"
            );
        }
    }
    for line in body.lines().filter(|l| l.contains("env::var")) {
        assert!(
            ["XDG_RUNTIME_DIR", "XDG_STATE_HOME", "\"HOME\""]
                .iter()
                .any(|v| line.contains(v)),
            "proxy.rs reads an environment variable that is not a directory: {line}"
        );
    }
    let relay = shipped("relay.rs").to_lowercase();
    for forbidden in [
        "dev_mode",
        "insecure",
        "skip_verif",
        "no_verify",
        "allow_unverified",
        "unattested",
    ] {
        assert!(!relay.contains(forbidden), "relay.rs mentions {forbidden}");
    }
}

/// The TD side: the cloud branch of `main.rs` uses the production anchors,
/// refuses the environment key and the operator pins, and serves only
/// through `cloud_serve::serve_tls` behind mutual TLS.
#[test]
fn the_td_side_has_no_development_fallback() {
    let main = shipped("main.rs");
    let cloud = &main[main.find("async fn run_cloud(").expect("run_cloud")..];
    let cloud = &cloud[..cloud.find("\n}\n").unwrap() + 3];
    assert!(cloud.contains("TdAnchors::production()"));
    assert!(
        cloud.contains("broker.0.is_some()"),
        "the environment key is refused in cloud mode"
    );
    assert!(cloud.contains("cloud_serve::serve_tls("));
    assert!(
        !cloud.contains("serve_with("),
        "cloud mode never falls back to the plain listener"
    );
    assert!(
        cloud.contains("tls::server_config("),
        "mutual TLS is mandatory"
    );
    for flag in [
        "expect_measurement",
        "measurements",
        "expected_model_digest",
    ] {
        assert!(
            cloud.contains(&format!("args.{flag}")),
            "cloud mode must refuse --{flag}"
        );
    }
    let serve = shipped("cloud_serve.rs");
    assert!(serve.contains(".route(\"/v2/attestation\", get(attestation))"));
    assert!(serve.contains(".route(\"/v2/key\", post(key))"));
    assert!(serve.contains(".route(\"/v2/ready\", get(ready))"));
    assert!(serve.contains(".route(\"/v1/chat/completions\", post(chat).layer(admitted))"));
    assert!(serve.contains(".route(\"/v2/attestation/session\", get(session_attestation))"));
    assert!(serve.contains(".route(\"/v2/model-key/recipient\", get(model_key_recipient))"));
    assert!(serve.contains(".route(\"/v2/model-key\", post(model_key))"));
    assert!(serve.contains(".route(\"/v2/e2e/key\", get(e2e_key))"));
    // SECURITY: the e2e chat takes a queue place, and reads at most the
    // largest request the SDK seals, a chunked one; a whole one is held to
    // the whole path's largest in the handler, before it is opened.
    let chat_route = &serve[serve.find("\"/v2/e2e/chat\",").expect("the e2e chat route")..];
    let chat_route: String = chat_route[..chat_route.find(".route(").unwrap()]
        .split_whitespace()
        .collect();
    assert!(chat_route.starts_with("\"/v2/e2e/chat\",post(e2e_chat)"));
    assert!(
        chat_route.contains(".layer(admitted.clone())"),
        "a queue place"
    );
    assert!(
        chat_route.contains("DefaultBodyLimit::max(x0ne_sdk::e2e::MAX_SEALED_STREAM_REQUEST"),
        "the SDK's largest sealed request, explicitly: {chat_route}"
    );
    let e2e_chat = &serve[serve.find("async fn e2e_chat(").expect("e2e_chat")..];
    let e2e_chat: String = e2e_chat[..e2e_chat.find("\n}\n").unwrap()]
        .split_whitespace()
        .collect();
    assert!(
        e2e_chat.contains("letlargest=ifwhole{x0ne_sdk::e2e::MAX_SEALED_REQUEST}")
            && e2e_chat.contains("Ok(b)if(whole||chunked)&&b.len()<=largest=>b,"),
        "a whole request past the whole path's largest is refused unopened"
    );
    assert_eq!(
        serve.matches(".route(").count(),
        9,
        "nine routes, nothing else before the key"
    );
    // SECURITY: the e2e key quote's report_data is this boot's OHTTP key, the
    // caller's nonce and this boot's binding (receipt key, boot nonce, TLS
    // key, config digest) under the e2e label, never chosen bytes (step 4 of
    // docs/sdk-e2e.md); on a GPU-attested session, over the GPUs' evidence too, read
    // for the e2e GPU nonce, and a document without it is never served
    // there. It passes the session quote's gate; neither e2e route
    // serves before the model is ready.
    let e2e_key = &serve[serve.find("async fn e2e_key(").expect("e2e_key")..];
    let e2e_key = &e2e_key[..e2e_key.find("\n}\n").unwrap()];
    assert!(e2e_key.contains("None => td.e2e.report_data(&nonce),"));
    assert!(
        !e2e_key.contains("receipt_key"),
        "the gateway's boot binding names it"
    );
    assert!(e2e_key.contains("let gpu = st.config.config.gpu_required();"));
    assert!(e2e_key.contains("let gpus = if gpu {"));
    assert!(e2e_key.contains(".gpu_evidence(&td.e2e.gpu_nonce(&nonce))"));
    assert!(e2e_key.contains(".report_data_gpu(&nonce, gpus)"));
    assert_eq!(e2e_key.matches(".gpu_evidence(").count(), 1);
    assert_eq!(e2e_key.matches(".tdx(").count(), 1);
    assert!(e2e_key.contains("source.tdx(&report_data, now).map(|c| (c, gpus))"));
    // SECURITY: NVML before the lock session quotes share, the quote under
    // it (review of 2026-10-27, finding 1).
    let at = |needle: &str| e2e_key.find(needle).expect(needle);
    assert!(at(".gpu_evidence(") < at("CHALLENGE_LOCK"));
    assert!(at(".report_data_gpu(") < at("CHALLENGE_LOCK"));
    assert!(at("CHALLENGE_LOCK") < at(".tdx("));
    assert!(e2e_key.contains("Some(gpus) => st.e2e.document_gpu(quote, &ccel, gpus),"));
    assert!(e2e_key.contains("None => st.e2e.document(quote, &ccel),"));
    assert_eq!(
        e2e_key
            .find("st.e2e_quotes.admit(")
            .map(|i| i < e2e_key.find(".tdx(").unwrap()),
        Some(true),
        "every e2e key quote passes its own quote gate first"
    );
    assert!(
        !e2e_key.contains("st.quotes."),
        "not the session quotes' bucket"
    );
    for route in [e2e_key, {
        let chat = &serve[serve.find("async fn e2e_chat(").expect("e2e_chat")..];
        &chat[..chat.find("\n}\n").unwrap()]
    }] {
        assert!(route.contains("st.serving_state()"), "ready first");
    }
    let gateway = shipped("e2e_gateway.rs");
    assert!(
        gateway.contains("e2e::key_binding(\n            &self.key_config(),\n            nonce,\n            &self.boot.session_pubkey,\n            &self.sdk_boot(),\n        )"),
        "the e2e report_data is the SDK's key binding, over this boot's binding"
    );
    assert!(
        gateway.contains("e2e::gpu_nonce(&self.key_config(), nonce, &self.boot.session_pubkey)")
    );
    assert!(gateway.contains("Ok(e2e::key_binding_gpu(\n            &self.key_config(),\n            nonce,\n            &self.boot.session_pubkey,\n            &self.sdk_boot(),\n            &views,\n        ))"));
    // The document carries the bytes the boot binding names, checked once.
    assert!(
        gateway.contains("if crate::cloud_config::sha256(session_config) != boot.config_sha256 {")
    );
    assert!(gateway.contains("session_config_b64: b64(&self.session_config),"));
    // SECURITY: the sealed chat is the attested chat handler, with its
    // padding, cadence and receipts, its panics contained: no other
    // answer path, no `chat_as` that unwinds.
    // SECURITY: the gateway's private scalar lives in `SecretBytes`, as the
    // model key's recipient key; its seed is read into memory wiped on drop.
    assert!(
        gateway.contains("Gateway::from_seed_held_by(seed, ")
            && gateway.contains("SecretBytes::from_slice("),
        "the OHTTP key in SecretBytes"
    );
    let main = shipped("main.rs");
    assert!(!main.contains("E2e::from_seed(&zeroize::Zeroizing::new(urandom32()))"));
    // SECURITY: the gateway names the binding the boot quote binds, and the
    // config bytes the TD serves under, never others.
    let at = |needle: &str| main.find(needle).expect(needle);
    assert!(
        at("let binding =\n        x0ne_infer::binding::BindingV2::of(&identity, tls_identity.spki_sha256(), loaded.sha256);")
            < at("x0ne_infer::e2e_gateway::E2e::from_seed(&seed, binding, &loaded.bytes)")
    );
    assert_eq!(main.matches("E2e::from_seed(").count(), 1);
    let answer = &gateway[gateway.find("pub async fn answer(").expect("answer")..];
    let answer = &answer[..answer.find("\n}\n").unwrap()];
    assert_eq!(
        answer.matches("crate::server::chat_contained_as(").count(),
        1
    );
    assert!(!answer.contains("chat_as("), "a panic would leave in clear");
    assert!(answer.contains("tokio::spawn("), "the answer's own task");
    // SECURITY: the recipient quote's report_data is the boot binding, the
    // boot's X25519 key and the relay's nonce under the model-key domain:
    // the nonce is never quoted as chosen bytes.
    let recipient = &serve[serve
        .find("async fn model_key_recipient(")
        .expect("model_key_recipient")..];
    let recipient = &recipient[..recipient.find("\n}\n").unwrap()];
    assert!(
        recipient.contains("crate::model_key::RecipientBinding::of(&st.binding, nonce, kem_pk)")
    );
    assert!(recipient.contains("let report_data = binding.tdx_report_data();"));
    assert!(
        recipient.contains("MAX_RECIPIENT_QUOTES"),
        "recipient quotes are bounded"
    );
    // A public session takes no model key, a sealed one no random key.
    let model_key = &serve[serve.find("async fn model_key(").expect("model_key")..];
    assert!(model_key.contains("if !st.config.config.is_sealed()"));
    let key = &serve[serve.find("async fn key(").expect("key")..];
    assert!(key[..key.find("\n}\n").unwrap()].contains("if st.config.config.is_sealed()"));
    // SECURITY: the session quote's report_data comes from the connection,
    // never from the request (an oracle would let a relay ask for its own).
    let session = &serve[serve
        .find("async fn session_attestation(")
        .expect("session_attestation")..];
    let signature = &session[..session.find(") -> Response").unwrap()];
    for extractor in ["Query", "Json", "Bytes", "Path", "body", "HeaderMap"] {
        assert!(
            !signature.contains(extractor),
            "the session route reads {extractor} from the request"
        );
    }
    let body = &session[..session.find("\n}\n").unwrap()];
    assert!(
        body.contains("BindingV3::of(&st.binding, exporter).tdx_report_data()"),
        "report_data is the boot binding and this connection's exporter, nothing else"
    );
    assert_eq!(
        body.find("st.quotes.admit(")
            .map(|i| i < body.find(".tdx(").unwrap()),
        Some(true),
        "every session quote passes the quote gate first"
    );
    assert!(
        serve.contains("Phase::Ready(s) => s.clone()"),
        "chat serves only in Ready"
    );
    let tls = shipped("tls.rs");
    assert!(tls.contains("TLS13"), "TLS 1.3 only");
    assert!(!tls.contains("TLS12"), "TLS 1.2 must not be offered");
    assert!(
        !tls.contains("allow_unauthenticated"),
        "a client certificate is mandatory"
    );
    // One connection = one full handshake: no resumption, no 0-RTT, both ends.
    for line in [
        "config.session_storage = Arc::new(rustls::server::NoServerSessionStorage {});",
        "config.send_tls13_tickets = 0;",
        "config.max_early_data_size = 0;",
        "config.resumption = rustls::client::Resumption::disabled();",
        "config.enable_early_data = false;",
    ] {
        assert!(tls.contains(line), "tls.rs lost `{line}`");
    }
}

/// Every refusal code the client verifier returns is in the documented list
/// below, and the relay binary documents the ones it adds.
#[test]
fn every_client_refusal_code_is_documented() {
    let body = shipped("remote_verify.rs");
    // 70: the TD refused to quote a model key recipient (`/v2/model-key/recipient`).
    let documented = [65, 66, 70, 75, 76, 77, 87, 88, 89];
    let mut seen = Vec::new();
    for (i, _) in body.match_indices("refuse(") {
        let rest = &body[i + 7..];
        let Some(end) = rest.find(',') else { continue };
        if let Ok(code) = rest[..end].trim().parse::<i32>() {
            assert!(
                documented.contains(&code),
                "refuse({code}) is not a documented code"
            );
            seen.push(code);
        }
    }
    for code in [65, 75, 76, 77, 87, 88, 89] {
        assert!(
            seen.contains(&code),
            "the verifier never refuses with {code}"
        );
    }
    let proxy = src("bin/proxy.rs");
    let header = &proxy[..proxy.find("use ").unwrap()];
    for code in ["87", "88", "89", "97", "98", "99", "100", "102"] {
        assert!(
            header.contains(code),
            "proxy.rs header must document exit {code}"
        );
    }
}

/// The transparency log is pinned as the archive key is: `RELEASE`, the
/// TEST log in debug builds with `test-archive-key` only, no other
/// constructor, and the test writer never in a build without the feature.
#[test]
fn the_transparency_log_is_pinned_and_the_test_log_never_ships() {
    let t = shipped("transparency.rs");
    let production = &t[t.find("pub fn production()").expect("production()")..];
    let production = &production[..production.find("\n    }\n").expect("end of production()")];
    let cfgs: Vec<&str> = production
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("#[cfg"))
        .collect();
    assert_eq!(
        cfgs,
        [
            "#[cfg(all(feature = \"test-archive-key\", debug_assertions))]",
            "#[cfg(not(all(feature = \"test-archive-key\", debug_assertions)))]",
        ],
        "production() picks the TEST log in debug builds with the feature only"
    );
    let test_branch = &production[production.find(cfgs[0]).unwrap()..];
    let test_branch = &test_branch[..test_branch.find(cfgs[1]).unwrap()];
    assert!(test_branch.contains("test_log::policy()"), "{test_branch}");
    let release_branch = &production[production.find(cfgs[1]).unwrap()..];
    assert!(
        release_branch.contains("&RELEASE") && !release_branch.contains("test_log"),
        "{release_branch}"
    );
    assert_eq!(
        fn_names(&inherent_impl(&t, "Anchor")),
        ["production", "policy"],
        "impl Anchor grew a constructor"
    );
    assert!(t.contains("pub const RELEASE: Option<Pin>"));
    assert!(
        t.contains("#[cfg(any(test, feature = \"test-archive-key\"))]\npub mod test_log"),
        "the TEST log writer is compiled without the feature"
    );
    // The reader of a pinned log follows no redirect and speaks plain http
    // to the loopback alone; the pin itself must be https (`pinned`).
    assert!(t.contains(".redirect(reqwest::redirect::Policy::none())"));
    assert!(t.contains(".https_only(!loopback)"));
    assert!(t.contains("!pin.tiles.starts_with(\"https://\")"));
}

/// Shipped Rust sources under `dir`, relative to the workspace, sorted.
fn rust_files(dir: &Path, root: &Path, out: &mut Vec<(String, String)>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            rust_files(&path, root, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            let text = std::fs::read_to_string(&path).unwrap();
            let shipped = text[..text.find("#[cfg(test)]").unwrap_or(text.len())].to_string();
            let rel = path.strip_prefix(root).unwrap().display().to_string();
            out.push((rel, shipped));
        }
    }
}

/// What a source does when it reaches a machine over SSH, or hands the
/// job to the cloud CLI (which terminates in the guest OS, outside the
/// attested channel).
fn reaches_ssh(body: &str) -> bool {
    body.contains("\"ssh\"")
        || body.contains("compute ssh")
        || body.contains("Command::new(\"ssh")
        || body.contains("\"scp\"")
        || body.contains("Command::new(\"gcloud")
}

/// SECURITY (downgrade to SSH): nothing on the attested path can reach the
/// TD over SSH. The SSH demonstration client left with the lab on 2026-11-04:
/// the list stays empty, and neither its name nor its old one comes back.
#[test]
fn no_ssh_on_the_shipped_path() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = crate_dir.parent().unwrap();
    let mut files = Vec::new();
    rust_files(&crate_dir.join("src"), root, &mut files);
    rust_files(
        &root.join("x0ne-engine-cloud").join("src"),
        root,
        &mut files,
    );
    assert!(
        files.len() > 40,
        "the sweep reads the sources: {}",
        files.len()
    );
    let ssh: Vec<&str> = files
        .iter()
        .filter(|(_, body)| reaches_ssh(body))
        .map(|(name, _)| name.as_str())
        .collect();
    assert!(ssh.is_empty(), "{ssh:?} reach SSH or the cloud CLI");
    for (name, body) in &files {
        assert!(
            !body.contains(&format!("\"{SSH_CLIENT_BIN}\"")) && !body.contains("\"bin/ssh_chat"),
            "{name} reaches the SSH demonstration client"
        );
    }
    let bins = workspace_bins(root);
    for gone in [SSH_CLIENT_BIN, "x0ne-infer-chat"] {
        assert!(!bins.contains_key(gone), "[[bin]] {gone} is back");
    }
}

/// The SSH demonstration client, removed with the lab on 2026-11-04.
const SSH_CLIENT_BIN: &str = "x0ne-infer-ssh-chat";

/// The client side of the attested path, which reaches the TD only through
/// the relay (`x0ne-engine cloud` starts the relay; the relay holds the
/// questions): what the client package (`ci/deb/x0ne-engine.ctl`), the .deb
/// build and the release archive ship. The TD's server has no use on a client
/// machine (2026-10-30); `x0ne-infer` ships in the CVM image only, built with
/// `cuda nvml` (x0ne-OS `scripts/pkg-infer-cuda.sh`, 2026-11-04).
const CLIENT_CTL: &str = "ci/deb/x0ne-engine.ctl";
const DEB_BUILD: &str = "ci/inside-deb.sh";
const CLIENT_BINS: [&str; 3] = ["x0ne-engine", "x0ne-engine-cloud", "x0ne-engine-proxy"];
const SERVER_BIN: &str = "x0ne-infer";

/// `[[bin]]` name → (`path`, `required-features`) across the workspace crates
/// that define binaries.
fn workspace_bins(root: &Path) -> std::collections::BTreeMap<String, (String, Vec<String>)> {
    let mut bins = std::collections::BTreeMap::new();
    for krate in ["x0ne-infer", "x0ne-engine-cloud"] {
        let manifest = root.join(krate).join("Cargo.toml");
        let text = std::fs::read_to_string(&manifest).unwrap();
        let table: toml::Table = text.parse().unwrap();
        for bin in table["bin"].as_array().unwrap() {
            let name = bin["name"].as_str().unwrap().to_string();
            let path = format!("{krate}/{}", bin["path"].as_str().unwrap());
            let features = bin
                .get("required-features")
                .and_then(|f| f.as_array())
                .map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect())
                .unwrap_or_default();
            assert!(bins.insert(name, (path, features)).is_none());
        }
    }
    bins
}

/// The binary names a packaging text lists: `out/<name>` on its `cp` lines,
/// `--bin <name>` on its build lines, `/out/bin/<name> /usr/bin/` in a
/// control file's `Files:`.
fn listed_binaries(text: &str) -> std::collections::BTreeSet<String> {
    let mut names = std::collections::BTreeSet::new();
    for line in text.lines() {
        let line = line.trim_start();
        if line.starts_with('#') {
            continue;
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            if line.starts_with("cp ") {
                // `out/<binary>`; the SBOMs (`out/*.cdx.json`) are files, not binaries.
                if let Some(name) = word.strip_prefix("out/") {
                    if !name.contains(['*', '.']) {
                        names.insert(name.trim_matches('"').to_string());
                    }
                }
            }
            if words.get(i + 1) == Some(&"/usr/bin/") {
                if let Some(name) = word.strip_prefix("/out/bin/") {
                    names.insert(name.to_string());
                }
            }
            if *word == "--bin" {
                if let Some(name) = words.get(i + 1) {
                    names.insert(name.to_string());
                }
            }
        }
    }
    names
}

/// SECURITY (downgrade to SSH): the release archive, its SBOM build, the .deb
/// build list and the client package (the `Files:` of its control file) ship
/// the client side of the attested path without the server, no binary whose
/// source reaches SSH or the cloud CLI, and no binary that needs a feature. A failure means a
/// packaging list grew a binary that bypasses the attested channel, or lost
/// one the path needs.
#[test]
fn the_release_ships_the_attested_path_and_nothing_that_reaches_ssh() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let bins = workspace_bins(root);
    let lists = [
        ".github/workflows/release.yml",
        "scripts/sbom.sh",
        DEB_BUILD,
        CLIENT_CTL,
    ];
    for list in lists {
        let text = std::fs::read_to_string(root.join(list))
            .unwrap_or_else(|e| panic!("{list} is missing: {e}"));
        let shipped = listed_binaries(&text);
        assert!(
            !shipped.contains(SSH_CLIENT_BIN),
            "{list} ships {SSH_CLIENT_BIN}"
        );
        for name in &shipped {
            let (path, features) = bins
                .get(name)
                .unwrap_or_else(|| panic!("{list} ships {name}, which no [[bin]] defines"));
            let body = std::fs::read_to_string(root.join(path)).unwrap();
            assert!(
                !reaches_ssh(&body),
                "{list} ships {name} ({path}), which reaches SSH"
            );
            assert!(
                features.is_empty(),
                "{list} ships {name}, which needs the features {features:?}"
            );
            assert!(
                !body.contains(&format!("\"{SSH_CLIENT_BIN}\""))
                    && !body.contains("\"bin/ssh_chat"),
                "{list} ships {name}, which reaches the SSH demonstration client"
            );
        }
        for name in CLIENT_BINS {
            assert!(shipped.contains(name), "{list} does not ship {name}");
        }
        assert!(
            !shipped.contains(SERVER_BIN),
            "{list} ships the TD's server {SERVER_BIN}: the CVM image ships it alone"
        );
    }
}

/// On the shipped client side, a question reaches the TD through the relay
/// alone: the relay binary drives `relay`, whose modules read no
/// environment (above), and neither the lifecycle nor the catalog binary
/// carries a question or a completion. A failure means a shipped client
/// grew a second way to the TD.
#[test]
fn the_shipped_client_data_path_is_the_attested_relay() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let bins = workspace_bins(root);
    assert!(shipped("bin/proxy.rs").contains("relay::"));
    for module in ["relay.rs", "local_api.rs", "tls.rs", "remote_verify.rs"] {
        assert!(
            VERIFICATION_MODULES.contains(&module),
            "{module} left the sweep"
        );
    }
    let mut files = Vec::new();
    rust_files(
        &root.join("x0ne-engine-cloud").join("src"),
        root,
        &mut files,
    );
    let engine = bins["x0ne-engine"].0.clone();
    files.push((engine.clone(), shipped("bin/engine.rs")));
    for (name, body) in &files {
        for marker in [
            "chat/completions",
            "request.json",
            "answer.json",
            "\"messages\"",
        ] {
            assert!(
                !body.contains(marker),
                "{name} carries a question ({marker:?})"
            );
        }
    }
}

/// The TD binary drops dumpability and core files before it reads a key,
/// starts a thread or parses an argument.
#[test]
fn the_td_binary_hardens_itself_first() {
    let main = shipped("main.rs");
    let body = &main[main.find("\nfn main() {").expect("fn main")..];
    let body = &body[..body.find("\n}\n").unwrap()];
    let harden = body
        .find("x0ne_infer::harden::process()")
        .expect("main calls harden::process()");
    for later in ["take_env_key(", "Runtime::new(", "async_main("] {
        assert!(
            harden < body.find(later).unwrap_or_else(|| panic!("{later}")),
            "harden::process() must run before {later}"
        );
    }
}

/// The section `[name]` of a manifest, to the next header.
fn manifest_section(manifest: &str, name: &str) -> String {
    let start = manifest
        .find(&format!("\n[{name}]\n"))
        .unwrap_or_else(|| panic!("Cargo.toml has no [{name}]"));
    let rest = &manifest[start + name.len() + 3..];
    rest[..rest.find("\n[").unwrap_or(rest.len())].to_string()
}

/// The archive key is pinned in the source: one constructor, no flag, no
/// variable. Only a debug build with `test-archive-key` (a self
/// dev-dependency) pins the TEST key; a default or a release build pins
/// `RELEASE`. The reference is verified before it is parsed, and only
/// `load` builds one.
#[test]
fn the_archive_key_is_pinned_and_the_test_key_never_ships() {
    let key = shipped("archive_key.rs");
    let production = &key[key.find("pub fn production()").expect("production()")..];
    let production = &production[..production.find("\n    }\n").expect("end of production()")];
    let cfgs: Vec<&str> = production
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("#[cfg"))
        .collect();
    assert_eq!(
        cfgs,
        [
            "#[cfg(all(feature = \"test-archive-key\", debug_assertions))]",
            "#[cfg(not(all(feature = \"test-archive-key\", debug_assertions)))]",
        ],
        "production() picks the TEST key in debug builds with the feature only"
    );
    let test_branch = &production[production.find(cfgs[0]).unwrap()..];
    let test_branch = &test_branch[..test_branch.find(cfgs[1]).unwrap()];
    assert!(test_branch.contains("test_key::keyring()"), "{test_branch}");
    assert!(
        !key.contains("compile_error!"),
        "a compile_error! on the feature breaks `cargo test --release` (2026-10-17)"
    );
    assert_eq!(
        fn_names(&inherent_impl(&key, "Anchor")),
        ["production", "key", "verify"],
        "impl Anchor grew a constructor"
    );
    assert!(key.contains("pub const RELEASE: Option<Pin>"));

    let manifest =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")).unwrap();
    let features = manifest_section(&manifest, "features");
    assert!(features.contains("test-archive-key = []"), "{features}");
    assert!(
        !features
            .lines()
            .any(|l| l.trim_start().starts_with("default")),
        "a default feature list: {features}"
    );
    let implied: Vec<&str> = features
        .lines()
        .filter(|l| !l.trim_start().starts_with('#') && l.contains("\"test-archive-key\""))
        .collect();
    assert!(
        implied.is_empty(),
        "a feature implies the TEST key: {implied:?}"
    );
    let me = manifest_section(&manifest, "dev-dependencies")
        .lines()
        .find(|l| l.trim_start().starts_with("x0ne-infer"))
        .expect("the crate is its own dev-dependency")
        .to_string();
    assert!(
        me.contains("path = \".\"")
            && me.contains("features = [\"test-archive-key\", \"candle-reference\"]"),
        "{me}"
    );
    assert!(
        !manifest_section(&manifest, "dependencies").contains("test-archive-key"),
        "[dependencies] enables the TEST key"
    );
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    // Every script the deb step of ci/ci.sh runs (`/ci/inside-deb*.sh` outside
    // a comment): a second one, as the lab's was until 2026-11-04, joins by its call.
    let ci = std::fs::read_to_string(root.join("ci/ci.sh")).unwrap();
    let mut debs: Vec<String> = ci
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .flat_map(str::split_whitespace)
        .filter_map(|w| w.strip_prefix("/ci/"))
        .filter(|n| n.starts_with("inside-deb"))
        .map(|n| format!("ci/{n}"))
        .collect();
    debs.dedup();
    assert!(debs.iter().any(|d| d == "ci/inside-deb.sh"), "{debs:?}");
    for release in debs.iter().map(String::as_str).chain([
        ".github/workflows/release.yml",
        "x0ne-engine-cloud/Cargo.toml",
    ]) {
        let body = std::fs::read_to_string(root.join(release)).unwrap();
        assert!(
            !body.contains("test-archive-key") && !body.contains("--all-features"),
            "{release} can build with the TEST archive key"
        );
    }
    // The release test build is where the TEST key must NOT be pinned
    // (tests/cloud_proxy_bin.rs): make ci runs it, the probe example with it.
    let inside = std::fs::read_to_string(root.join("ci/inside.sh")).unwrap();
    let steps = inside
        .lines()
        .find(|l| l.starts_with("STEPS="))
        .expect("ci/inside.sh STEPS");
    assert!(
        steps.split([' ', '}']).any(|w| w == "test-release"),
        "make ci does not run test-release: {steps}"
    );
    let step = inside
        .lines()
        .find(|l| l.trim_start().starts_with("test-release)"))
        .expect("ci/inside.sh has a test-release step");
    for want in [
        "cargo test",
        "--release",
        "-p x0ne-infer",
        "--all-targets",
        "--features probe",
    ] {
        assert!(step.contains(want), "test-release lacks {want}: {step}");
    }

    let proxy = shipped("bin/proxy.rs");
    assert!(proxy.contains("archive_key::Anchor::production()"));
    // The pin comes before the reference is looked for: a build made before
    // the signing ceremony answers 99 even with no reference installed.
    // Whitespace dropped: rustfmt splits method chains over lines.
    let main: String = proxy[proxy.find("fn main()").expect("proxy main")..]
        .split_whitespace()
        .collect();
    let pin = main
        .find("archive_key.key()")
        .expect("proxy main checks the pin");
    let default = main
        .find("default_path()")
        .expect("proxy main finds the installed reference");
    assert!(
        pin < default,
        "proxy main looks for the reference before it checks the pin"
    );
    for (i, _) in proxy.match_indices("Anchor::") {
        assert!(
            proxy[i..].starts_with("Anchor::production()"),
            "proxy.rs reaches the archive key other than by production()"
        );
    }
    assert!(proxy.contains("transparency::Anchor::production()"));
    let log_pin = main
        .find("tlog.policy()")
        .expect("proxy main checks the transparency log pin");
    assert!(
        log_pin < default,
        "proxy main looks for the reference before it checks the log pin"
    );

    let reference = shipped("reference.rs");
    let signed = &reference[reference
        .find("pub(crate) fn load_signed(")
        .expect("load_signed")..];
    let signed = &signed[..signed.find("\n}\n").unwrap()];
    // load_signed hands the bytes it read to admit_signed, the one body that
    // admits a reference (x0ne-sdk's admit_offline runs it too), with the
    // log check of the file it read.
    assert!(
        signed.contains("admit_signed(&bytes, &sig, key, log, now_unix,"),
        "load_signed does not admit the bytes it read through admit_signed"
    );
    assert!(
        signed.contains("transparency::check_file(path, &bytes, log,"),
        "load checks the transparency log"
    );
    let admit = &reference[reference
        .find("pub(crate) fn admit_signed(")
        .expect("admit_signed")..];
    let admit = &admit[..admit.find("\n}\n").unwrap()];
    let pinned = admit
        .find("key.key()")
        .expect("admit_signed checks the key pin");
    let verified = admit
        .find(".verify(bytes, signature, now_unix)")
        .expect("admit_signed verifies the bytes");
    // The transparency log, after the signature; nothing parses before both.
    let logged = admit
        .find("logged(policy)")
        .expect("admit_signed checks the transparency log");
    assert!(
        pinned < verified && verified < logged,
        "admit_signed must check the pins, the signature, then the log"
    );
    assert!(!signed.contains("parse("), "load_signed parses");
    assert!(!admit.contains("parse("), "admit_signed parses");
    let offline = &reference[reference
        .find("pub fn admit_offline(")
        .expect("admit_offline")..];
    let offline = &offline[..offline.find("\n}\n").unwrap()];
    let admitted = offline
        .find("admit_signed(bytes, signature, key, log, now_unix,")
        .expect("admit_offline admits through admit_signed");
    let parsed = offline.find("parse(bytes)").expect("admit_offline parses");
    assert!(admitted < parsed, "admit_offline parses before admitting");
    // Every loader admits the bytes through load_signed and parses them only
    // in its schema's from_signed, which takes what load_signed built.
    let snp_reference = shipped("snp_reference.rs");
    let remote_verify = shipped("remote_verify.rs");
    for (file, body, start) in [
        ("reference.rs", &reference, "pub fn load("),
        ("snp_reference.rs", &snp_reference, "pub fn load("),
        ("remote_verify.rs", &remote_verify, "    pub fn load("),
    ] {
        let load = &body[body.find(start).unwrap_or_else(|| panic!("{file}: load"))..];
        let load = &load[..load.find("\n}\n").unwrap()];
        assert!(
            load.contains("load_signed(path, explicit, key, log, state_dir, now_unix)?"),
            "{file}: load does not go through load_signed"
        );
        assert!(
            load.contains("from_signed("),
            "{file}: load skips from_signed"
        );
        assert!(
            !load.contains("parse("),
            "{file}: load parses outside from_signed"
        );
        assert!(
            !load.contains("fs::read"),
            "{file}: load reads the file itself"
        );
    }
    for (file, body) in [
        ("reference.rs", &reference),
        ("snp_reference.rs", &snp_reference),
    ] {
        let from = &body[body
            .find("pub(crate) fn from_signed(")
            .unwrap_or_else(|| panic!("{file}: from_signed"))..];
        let from = &from[..from.find("\n}\n").unwrap()];
        assert!(
            from.contains("SignedBytes,"),
            "{file}: from_signed takes other bytes"
        );
        assert!(
            from.contains("parse(&s.bytes)"),
            "{file}: from_signed parses other bytes"
        );
        assert_eq!(
            literals(body, "Loaded"),
            1,
            "{file}: one Loaded literal, in from_signed()"
        );
    }
    assert_eq!(
        literals(&reference, "SignedBytes"),
        1,
        "one SignedBytes literal, in load_signed()"
    );
    for name in VERIFICATION_MODULES
        .iter()
        .filter(|n| **n != "reference.rs")
    {
        assert_eq!(
            literals(&shipped(name), "SignedBytes"),
            0,
            "{name} builds SignedBytes"
        );
    }
    for name in VERIFICATION_MODULES
        .iter()
        .chain(&["bin/proxy.rs", "main.rs"])
    {
        assert!(
            !shipped(name).contains("reference::Loaded {"),
            "{name} builds a reference without its signature"
        );
    }
}

/// One recipe for what ships and what `make ci` tests, so the tests run the bytes shipped (2026-10-25):
/// `Cargo.toml` defines one profile, `release`, with the line tables `ci/strip-release.sh`
/// moves out, and beside it only the opt-level of dependencies in dev and test builds
/// (`[profile.dev.package."*"]`, 2026-10-30), which no shipped binary uses; every build of a
/// shipped binary and the step `test-release` use `--release`, never `--profile`; each
/// delivery script refuses `CARGO_PROFILE_*` and `RUSTFLAGS` before its first cargo command
/// (`ci/release-env.sh`); `make ci` (step `release-refusal`) runs `x0ne-infer` through
/// `ci/release-refusal.sh`. No .deb and no release archive ships `x0ne-infer` since
/// 2026-11-04: the CVM image does. A failure means a shipped binary may again be built by a
/// recipe no test runs.
#[test]
fn the_shipped_binaries_and_test_release_share_one_profile() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let read = |rel: &str| {
        std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
    };
    let code = |text: &str| -> Vec<String> {
        text.lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .map(str::to_string)
            .collect()
    };

    let cargo = read("Cargo.toml");
    let profiles: Vec<&str> = cargo
        .lines()
        .filter(|l| l.trim_start().starts_with("[profile."))
        .collect();
    assert_eq!(
        profiles,
        ["[profile.dev.package.\"*\"]", "[profile.release]"],
        "Cargo.toml profiles"
    );
    let table = |header: &str| -> String {
        let t = &cargo[cargo.find(header).unwrap()..];
        t[..t[1..].find("\n[").map_or(t.len(), |i| i + 1)].to_string()
    };
    let deps = table("[profile.dev.package.\"*\"]");
    let keys: Vec<String> = code(&deps)
        .iter()
        .skip(1)
        .filter_map(|l| l.split_once('=').map(|(k, _)| k.trim().to_string()))
        .collect();
    assert_eq!(keys, ["opt-level"], "{deps}");
    let release = table("[profile.release]");
    assert!(
        release.contains("debug = \"line-tables-only\"") && release.contains("strip = \"none\""),
        "{release}"
    );

    for script in ["ci/inside-deb.sh", "scripts/sbom.sh"] {
        let lines = code(&read(script));
        let builds: Vec<&String> = lines
            .iter()
            .filter(|l| l.contains("cargo build") || l.contains("cargo auditable build"))
            .collect();
        assert!(!builds.is_empty(), "{script} builds nothing");
        for b in builds {
            assert!(
                b.contains("--release") && !b.contains("--profile"),
                "{script}: {b}"
            );
        }
        let guard = lines
            .iter()
            .position(|l| l.contains("release-env.sh"))
            .unwrap_or_else(|| panic!("{script} does not run ci/release-env.sh"));
        let first_cargo = lines
            .iter()
            .position(|l| l.contains("cargo "))
            .unwrap_or_else(|| panic!("{script}: no cargo"));
        assert!(guard < first_cargo, "{script} runs cargo before the guard");
    }

    let inside = read("ci/inside.sh");
    let step = inside
        .lines()
        .find(|l| l.trim_start().starts_with("test-release)"))
        .expect("ci/inside.sh has a test-release step");
    assert!(
        step.contains("--release") && !step.contains("--profile"),
        "{step}"
    );
    for list in ["STEPS=", "LANE_C="] {
        let line = inside
            .lines()
            .find(|l| l.starts_with(list))
            .unwrap_or_else(|| panic!("ci/inside.sh: {list}"));
        assert!(
            line.split_whitespace()
                .any(|s| s.trim_matches('"') == "release-refusal"),
            "{list} lacks release-refusal"
        );
    }
    let refusal = &inside[inside
        .find("release_refusal() {")
        .expect("ci/inside.sh defines release_refusal")..];
    let refusal = &refusal[..refusal.find("\n}").unwrap()];
    assert!(
        refusal.contains("cargo build --release")
            && !refusal.contains("--profile")
            && !refusal.contains("--features")
            && refusal.contains("bash /src/ci/release-refusal.sh"),
        "{refusal}"
    );
}

/// `source` without its comments, strings and char literals.
fn code_only(source: &str) -> String {
    let code = code_mask(source);
    let kept: Vec<u8> = source
        .bytes()
        .zip(code)
        .filter_map(|(b, is_code)| is_code.then_some(b))
        .collect();
    String::from_utf8_lossy(&kept).into_owned()
}

/// SECURITY (time padding): every answer leaves on sizes of one tick,
/// `stream::TICK`, one second at least (`padding::MIN_TICKS`); the host reads
/// no shorter time. The unit tests of the crate that assert no time serve at
/// a shorter tick (`padding::Tick::SHORT`) so that they do not wait a second
/// a request. In shipped code `padding::Tick` has no field and one value,
/// its period is the constant, and what waits reads it: no flag, variable or
/// configuration field reaches it. (The cadence of a sealed stream is the
/// session's, from its signed config: `e2e_stream`, another matter.)
///
/// A failure means a tick other than the constant is reachable outside
/// `cfg(test)`: remove it, or put it back under `cfg(test)`.
#[test]
fn the_padding_tick_is_the_constant_outside_cfg_test() {
    let flat = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    let block = |body: &str, head: &str| {
        let start = body.find(head).unwrap_or_else(|| panic!("no `{head}`"));
        let rest = &body[start..];
        rest[..rest.find("\n}\n").expect("closing brace")].to_string()
    };
    let stream = shipped("stream.rs");
    let padding = shipped("padding.rs");
    assert!(
        stream.contains("\npub const TICK: Duration = Duration::from_millis(100);\n"),
        "stream::TICK is no longer 100 ms"
    );
    assert!(
        padding.contains("\npub const MIN_TICKS: u64 = 10;\n"),
        "padding::MIN_TICKS is no longer 10"
    );

    // The type: no field, one literal, a period that is the constant.
    assert!(
        flat(&padding).contains("pub struct Tick { }"),
        "padding::Tick has a field outside cfg(test)"
    );
    let tick = inherent_impl(&padding, "Tick");
    assert_eq!(tick.matches("fn ").count(), 2, "period and floor: {tick}");
    assert_eq!(self_literals(&tick), 1, "{tick}");
    assert!(
        flat(&tick).contains("pub const PRODUCTION: Self = Self { };"),
        "{tick}"
    );
    assert!(
        flat(&tick).contains("pub const fn period(self) -> Duration { crate::stream::TICK }"),
        "{tick}"
    );
    assert!(
        flat(&tick).contains("self.period() .saturating_mul("),
        "{tick}"
    );
    assert!(!tick.contains("Duration::from"), "{tick}");

    // What holds a tick hands out production's.
    let server = shipped("server.rs");
    assert!(
        !block(&server, "pub struct Serving {").contains("tick:"),
        "server::Serving carries a tick outside cfg(test)"
    );
    assert!(
        flat(&server).contains(
            "pub(crate) fn tick(&self) -> crate::padding::Tick { crate::padding::Tick::PRODUCTION }"
        ),
        "server::Serving::tick"
    );
    assert!(
        flat(&block(&padding, "impl Default for Turns {")).contains("tick: Tick::PRODUCTION,"),
        "padding::Turns::default"
    );
    assert!(
        flat(&stream).contains(
            "pub fn ticking(tick: crate::padding::Tick) -> Self { \
             Self::every(Instant::now(), tick.period()) }"
        ),
        "stream::Clock::ticking"
    );

    // What waits reads the tick, and nothing else gives it a period.
    for (file, f, wait) in [
        (
            "server.rs",
            "whole_refused",
            "sleep_until(asked + tick.floor())",
        ),
        (
            "e2e_gateway.rs",
            "at_floor",
            "sleep_until(asked + tick.floor())",
        ),
    ] {
        let body = shipped(file);
        let start = body
            .find(&format!("fn {f}("))
            .unwrap_or_else(|| panic!("{file}: no fn {f}"));
        let end = start + body[start..].find("\n}\n").expect("closing brace");
        assert!(body[start..end].contains(wait), "{file} {f}");
    }
    let lib = shipped("lib.rs");
    let mut files = Vec::new();
    sources(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        "",
        &mut files,
    );
    files.sort();
    let (mut swept, mut releases, mut clocks) = (0, 0, Vec::new());
    for file in &files {
        // A module `lib.rs` declares under `cfg(test)` is no shipped code.
        let module = file.strip_suffix(".rs").unwrap_or(file);
        let declared = module.contains('/')
            || ["lib", "main"].contains(&module)
            || lib.contains(&format!("mod {module};"));
        if !declared {
            continue;
        }
        swept += 1;
        let body = code_only(&shipped(file));
        for test_only in ["Tick::SHORT", ".ticking(", "Turns::every("] {
            assert!(!body.contains(test_only), "{file}: {test_only} is shipped");
        }
        if file != "padding.rs" {
            // `Tick {` after a path is a literal, unless a signature returns it.
            for (at, _) in body.match_indices("Tick {") {
                let path = body[..at]
                    .trim_end_matches(|c: char| c.is_alphanumeric() || c == '_' || c == ':');
                assert!(
                    ["-> ", "struct ", "impl ", "for "]
                        .iter()
                        .any(|before| path.ends_with(before)),
                    "{file} builds a Tick"
                );
            }
            assert!(!body.contains("MIN_TICKS") || file == "stream.rs", "{file}");
        }
        if !["padding.rs", "stream.rs"].contains(&file.as_str()) {
            assert!(
                !body.contains("stream::TICK"),
                "{file} reads stream::TICK and not a padding::Tick"
            );
        }
        for (at, _) in body.match_indices("release_after(") {
            if body[..at].ends_with("fn ") {
                continue;
            }
            let call = &body[at..at + body[at..].find(';').expect("a statement")];
            assert!(call.contains(".period())"), "{file}: {call}");
            releases += 1;
        }
        if body.contains("Clock::every(") {
            clocks.push(file.as_str());
        }
    }
    assert!(swept > 60, "{swept} shipped sources swept");
    assert_eq!(
        releases, 3,
        "release_after: a turn, a stream, a sealed panic"
    );
    // The sealed stream's, at its session's cadence.
    assert_eq!(clocks, ["e2e_gateway.rs"]);
    for file in ["padding.rs", "stream.rs", "serving_lease.rs"] {
        let body = code_only(&shipped(file));
        for read in ["env::", "var(", "var_os("] {
            assert!(!body.contains(read), "{file} reads {read}");
        }
    }
}
