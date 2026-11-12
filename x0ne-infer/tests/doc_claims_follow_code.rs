//! Statements of README.md, SECURITY.md and THREAT_MODEL.md held to the code:
//! each test reads the code fact first and binds the prose only while it holds.
//! A failure means a document and the code disagree: fix whichever is wrong.

use std::path::{Path, PathBuf};

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
fn flat(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// The published prose about the verifier: README, SECURITY, THREAT_MODEL
/// and docs/*.md. Run records are left out: they are dated statements of what
/// a past build printed.
fn documents() -> Vec<(String, String)> {
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
            (p.display().to_string(), flat(&text))
        })
        .collect()
}

fn none_says(stale: &[&str]) {
    for (name, text) in documents() {
        for s in stale {
            assert!(!text.contains(s), "{name} still says {s:?}");
        }
    }
}

/// Paragraphs, and the rows of a table one by one, flattened.
fn units(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for para in text.split("\n\n") {
        if para.trim_start().starts_with('|') {
            out.extend(para.lines().map(flat));
        } else {
            out.push(flat(para));
        }
    }
    out
}

/// The TD never reads the session expiry: `cloud_config` only refuses 0, the
/// relay stops sending past it, and the key's lease is the 24 h ceiling
/// counted from the release (`grant` passes no validity).
#[test]
fn the_td_lease_is_not_said_to_end_at_the_session_expiry() {
    if read("x0ne-infer/src/cloud_serve.rs").contains("expires_at_unix") {
        return;
    }
    assert!(
        read("x0ne-infer/src/local_api.rs").contains("expires_at_unix"),
        "the relay no longer reads the session expiry: re-read this test"
    );
    none_says(&["the lease ends at the session expiry"]);
}

/// `memfd_secret` holds the released key and the TLS key DER; the key's way in,
/// the loader's `ReleasedKey` and the receipt signing key are ordinary memory.
#[test]
fn the_memfd_secret_texts_name_the_ordinary_copies() {
    let keys = read("x0ne-infer/src/keys.rs");
    let binding = read("x0ne-infer/src/binding.rs");
    let relay = read("x0ne-infer/src/remote_verify.rs");
    let heap_released = keys.contains("material: Zeroizing<Vec<u8>>");
    let heap_signing = binding.contains("signing: SigningKey");
    let relay_copy = relay.contains("body.to_string().into_bytes()");
    for rel in ["docs/attested-channel.md", "x0ne-infer/THREAT_MODEL.md"] {
        let found: Vec<String> = units(&read(rel))
            .into_iter()
            .filter(|u| u.contains("memfd_secret"))
            .collect();
        assert!(!found.is_empty(), "{rel} no longer names memfd_secret");
        let text = found.join(" ");
        if heap_released {
            assert!(
                text.contains("releasedkey"),
                "{rel} puts the key in memfd_secret without the heap copy the loader gets (ReleasedKey)"
            );
        }
        if heap_signing {
            assert!(
                text.contains("signing key") && text.contains("ordinary heap"),
                "{rel} does not say the receipt signing key is ordinary heap"
            );
        }
        if relay_copy {
            assert!(
                text.contains("post_key") && text.contains("axum"),
                "{rel} does not name the copies on the key's way in"
            );
        }
    }
}

/// security.yml runs `cargo deny` on pushes and pull requests to `main` and
/// weekly, with no cargo-audit job; deny.toml ignores `paste`.
#[test]
fn the_audit_tools_are_described_as_ci_runs_them() {
    let sec = flat(&read("SECURITY.md"));
    if read(".github/workflows/security.yml").contains("No cargo-audit job") {
        assert!(
            !sec.contains("both run in ci on every push"),
            "SECURITY.md says cargo audit runs in CI"
        );
    }
    let paste_ignored = read("deny.toml").lines().any(|l| {
        l.split('#')
            .next()
            .unwrap_or("")
            .contains("RUSTSEC-2024-0436")
    });
    if paste_ignored {
        assert!(
            !sec.contains("not ignored — `cargo audit` reports"),
            "SECURITY.md says paste is not ignored, deny.toml ignores it"
        );
    }
}

/// `TcbStatus::accepted` admits `SWHardeningNeeded` beside `UpToDate`.
#[test]
fn the_accepted_tcb_statuses_are_the_ones_named() {
    let tcb = read("x0ne-infer/src/tcb.rs");
    let at = tcb
        .find("pub fn accepted(&self)")
        .expect("tcb.rs: TcbStatus::accepted");
    if !tcb[at..at + 200].contains("SwHardeningNeeded") {
        return;
    }
    none_says(&["the platform and the qe are up to date;"]);
}

/// The verifiers refuse every TDATTRIBUTES bit outside `TD_ATTRIBUTES_ALLOWED`,
/// not DEBUG alone: no text may still say the other bits go unjudged, and
/// docs/verification.md names MIGRATABLE among the refused.
#[test]
fn the_td_attributes_are_not_said_to_be_judged_on_debug_alone() {
    let code = read("x0ne-confidential/src/tdx_quote.rs");
    if !code.contains("pub const TD_ATTRIBUTES_ALLOWED") {
        return;
    }
    none_says(&[
        "the other attribute bits are not judged",
        "the other bits are not judged",
        "only the debug bit is checked",
    ]);
    assert!(flat(&read("docs/verification.md")).contains("migratable"));
    // Reserved bits 61:56 must be 0 (ABI 008): refused since 2026-10-24.
    if !code.contains("0xff << 56") {
        none_says(&[
            "other bits (63:56) are accepted",
            "a reserved bit with a negative or unknown impact",
        ]);
    }
}

/// Every variable `harden::LOADER_VARIABLES` and `RELEASE_LOADER_VARIABLES`
/// refuse is named in the threat model, which says what a launcher's
/// environment can no longer carry.
#[test]
fn the_refused_loader_variables_are_named_in_the_threat_model() {
    let code = read("x0ne-infer/src/harden.rs");
    let threat = read("x0ne-infer/THREAT_MODEL.md");
    for list in ["LOADER_VARIABLES", "RELEASE_LOADER_VARIABLES"] {
        let decl = format!("pub const {list}:");
        let at = code
            .find(&decl)
            .unwrap_or_else(|| panic!("harden.rs declares {list}"));
        let value = &code[at..];
        let value = &value[value.find('=').expect("=")..value.find("];").expect("];")];
        let names: Vec<&str> = value.split('"').skip(1).step_by(2).collect();
        assert!(!names.is_empty(), "no name in {list}: {value}");
        for name in names {
            assert!(
                threat.contains(&format!("`{name}`")),
                "THREAT_MODEL.md does not name {name}"
            );
        }
    }
}

/// Google's endorsement path checks the signer's validity window, not the
/// pinned root's.
#[test]
fn no_text_says_every_chain_has_every_window_checked() {
    let gce = read("x0ne-infer/src/gce_endorsement.rs");
    if gce.matches("is_valid_at").count() > 1 {
        return;
    }
    none_says(&[
        "every certificate in every chain is checked against its own validity window",
        "every certificate of every chain is checked against its own validity window",
    ]);
}

/// Cloud mode has an attested TLS channel and takes its key over it; the
/// cloud loader serves plaintext public weights and does not use that key.
#[test]
fn the_attested_channel_and_the_key_release_are_not_called_unbuilt() {
    let serve = read("x0ne-infer/src/cloud_serve.rs");
    if !serve.contains(".route(\"/v2/key\"") {
        return;
    }
    none_says(&[
        "an attested transport (plain http",
        "a key broker (v0 reads an environment variable)",
        "data in transit is out of scope today",
        "not a remote release",
    ]);
}

/// A measurements file that names registers is not described as naming none.
#[test]
fn published_measurements_are_described_as_they_are() {
    let dir = workspace().join("measurements");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("measurements/ is readable")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    assert!(!files.is_empty());
    for p in files {
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&p).expect("readable")).expect("json");
        let named = v["registers"].as_object().is_some_and(|m| !m.is_empty());
        let version = v["version"].as_str().expect("version");
        if named {
            none_says(&[&format!("the file of {version} names no register")]);
        }
    }
}

/// release.yml builds twice, refuses a difference and attests provenance.
#[test]
fn the_release_workflow_is_not_called_unbuilt() {
    let release = read(".github/workflows/release.yml");
    if !(release.contains("attest-build-provenance") && release.contains("cmp ")) {
        return;
    }
    none_says(&["planned (m8), nothing built"]);
}

/// A release build of this version exits 87 before collecting evidence; there
/// is no `lab` job (the lab left on 2026-11-04).
#[test]
fn the_exit_code_and_the_ci_jobs_named_exist() {
    let main = read("x0ne-infer/src/main.rs");
    let unpinned = main.find("measurements::EXIT_UNPINNED;");
    let no_evidence = main.find("return 64;");
    if let (Some(a), Some(b)) = (unpinned, no_evidence) {
        if a < b {
            none_says(&["it exits 64, whatever the features"]);
        }
    }
    let ci = read(".github/workflows/ci.yml");
    if !ci.lines().any(|l| l == "  lab:") {
        none_says(&["(job `lab`)", "ci `lab` and `image` jobs"]);
    }
}

/// A rustls version named in prose is the one Cargo.lock holds.
#[test]
fn the_rustls_version_named_is_the_locked_one() {
    let lock = read("Cargo.lock");
    let at = lock
        .find("name = \"rustls\"\n")
        .expect("Cargo.lock: rustls");
    let line = lock[at..].lines().nth(1).expect("version line");
    let version = line
        .trim_start_matches("version = \"")
        .trim_end_matches('"');
    for (name, text) in documents() {
        let mut rest = text.as_str();
        while let Some(i) = rest.find("rustls 0.") {
            let named: String = rest[i + "rustls ".len()..]
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            let named = named.trim_end_matches('.');
            assert_eq!(named, version, "{name} names rustls {named}");
            rest = &rest[i + 1..];
        }
    }
}

/// The PCK CRL and Intel's root CA CRL are read (`tcb::verify_pck_crl`,
/// `tcb::verify_root_crl`) and NRAS's OCSP `good` is required; the NRAS
/// chain's own OCSP is judged from stapled responses (below).
#[test]
fn revocations_are_not_said_to_be_unread() {
    if !read("x0ne-infer/src/tcb.rs").contains("fn verify_pck_crl") {
        return;
    }
    none_says(&[
        "revocations (intel crl, nvidia ocsp)",
        "revocations are not read",
    ]);
}

/// The NRAS signing chain's OCSP responses are judged when a document
/// staples them (`nras_ocsp::judge_chain`, 2026-10-31): no document still
/// says the chain's revocation is unchecked. What is left is said as it is:
/// no TD staples a response yet.
#[test]
fn the_nras_chain_ocsp_is_not_said_to_be_unchecked() {
    if !read("x0ne-infer/src/nras_ocsp.rs").contains("pub fn judge_chain") {
        return;
    }
    none_says(&[
        "the ocsp of nvidia's chain is not checked",
        "the nras chain's ocsp is not read",
        "revocation of that chain is not checked",
        "revocation of the nras signing chain is not checked",
        "nras chain revocation (nvidia publishes a crl and ocsp; neither is used)",
        "the ocsp of the nras token's own chain is not read",
    ]);
    for rel in [
        "README.md",
        "docs/limitations.md",
        "x0ne-infer/THREAT_MODEL.md",
    ] {
        let text = read(rel).to_lowercase();
        assert!(
            text.contains("no td staples") || text.contains("no td fetches"),
            "{rel} does not say that no TD staples a response yet"
        );
    }
}

/// The image reference is signed and checked; `measurements/*.json` carry
/// `"signature": null`.
#[test]
fn what_is_not_signed_is_named() {
    if !read("x0ne-infer/src/reference.rs").contains("openpgp") {
        return;
    }
    none_says(&["**the reference values are measured once and not signed**"]);
}

/// A request that names no token budget gets the server's default: the API
/// page says that number.
#[test]
fn the_default_token_budget_is_the_one_the_api_page_names() {
    let request = flat(&read("x0ne-infer/src/openai/request.rs"));
    let default = request
        .split("fn default_max_tokens() -> usize { ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .expect("request.rs: default_max_tokens");
    assert!(default.parse::<usize>().is_ok(), "{default}");
    let api = flat(&read("docs/api.md"));
    assert!(
        api.contains(&format!("neither: {default} tokens")),
        "docs/api.md does not say that a request without max_completion_tokens gets {default}"
    );
}

/// x0ne Core takes the trait's default `generate_turn`, which refuses a
/// conversation: while it does, the API page says so before it describes one.
#[test]
fn a_conversation_is_not_described_as_served_while_core_keeps_none() {
    let mut core = read("x0ne-infer/src/core_backend.rs");
    let dir = workspace().join("x0ne-infer/src/core_backend");
    for e in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let p = e.expect("entry").path();
        if p.extension().is_some_and(|x| x == "rs") {
            core.push_str(&std::fs::read_to_string(&p).expect("a source of core_backend"));
        }
    }
    if core.contains("fn generate_turn") {
        return;
    }
    let api = flat(&read("docs/api.md"));
    let section = api
        .split("### `x0ne.conversation`")
        .nth(1)
        .expect("docs/api.md: the x0ne.conversation section");
    let first = section.split_once(" between turns ").map(|(_, text)| text);
    assert!(
        first.is_some_and(|text| text.starts_with("**not served in this version.**")),
        "docs/api.md describes x0ne.conversation before saying x0ne Core refuses it"
    );
}

/// `up` cannot delete the VM when it is killed (`chaos.rs` kills it and finds
/// the VM): no title says every ending deletes it.
#[test]
fn a_killed_up_is_not_said_to_delete_its_vm() {
    if !read("x0ne-engine-cloud/tests/chaos.rs").contains("fn killed_while_serving()") {
        return;
    }
    none_says(&["every way it ends deletes the vm"]);
}

/// The members of the root Cargo.toml's `[workspace]`, in order.
fn workspace_members(cargo_toml: &str) -> Vec<String> {
    toml_list(cargo_toml, "members = [").expect("Cargo.toml: workspace members")
}

/// The crates the root Cargo.toml's `[workspace]` excludes, in order.
fn workspace_excluded(cargo_toml: &str) -> Vec<String> {
    toml_list(cargo_toml, "exclude = [").unwrap_or_default()
}

/// The strings of one `key = [...]` list of the root Cargo.toml, if it has one.
fn toml_list(cargo_toml: &str, key: &str) -> Option<Vec<String>> {
    let at = cargo_toml.find(key)?;
    let rest = &cargo_toml[at + key.len()..];
    let end = rest.find(']').expect("Cargo.toml: list closed");
    Some(
        rest[..end]
            .split(',')
            .map(|m| m.trim().trim_matches('"').to_string())
            .filter(|m| !m.is_empty())
            .collect(),
    )
}

/// The rows of README's table "The crates", header and rule left out: the
/// crate each names in `code` in its first cell, and the row itself.
fn crate_table(readme: &str) -> Vec<(String, String)> {
    let Some(at) = readme.find("\n## The crates\n") else {
        return Vec::new();
    };
    let section = &readme[at + 1..];
    let section = section.split_once("\n## ").map_or(section, |(s, _)| s);
    section
        .lines()
        .filter(|l| l.starts_with('|'))
        .skip(2)
        .filter_map(|l| {
            let cell = l.trim_start_matches('|').split('|').next()?;
            let (_, rest) = cell.split_once('`')?;
            let (name, _) = rest.split_once('`')?;
            Some((name.to_string(), l.to_string()))
        })
        .collect()
}

/// What README's table "The crates" says wrong about the workspace: every
/// member and every excluded crate has one row that names it in code, no
/// other crate has one, and an excluded crate's row says it is a workspace
/// of its own.
fn crate_table_problems(members: &[String], excluded: &[String], readme: &str) -> Vec<String> {
    let mut out = Vec::new();
    let rows = crate_table(readme);
    if rows.is_empty() {
        out.push("no table under \"## The crates\" names a crate in code".to_string());
        return out;
    }
    for c in members.iter().chain(excluded) {
        match rows.iter().filter(|(name, _)| name == c).count() {
            0 => out.push(format!("the crates table has no row for {c}")),
            1 => {}
            n => out.push(format!("the crates table has {n} rows for {c}")),
        }
    }
    for (name, _) in &rows {
        if !members.contains(name) && !excluded.contains(name) {
            out.push(format!(
                "the crates table names {name}, which Cargo.toml neither lists nor excludes"
            ));
        }
    }
    for c in excluded {
        let apart = rows.iter().any(|(name, row)| {
            let row = row.to_lowercase();
            name == c && (row.contains("its own workspace") || row.contains("workspace of its own"))
        });
        if rows.iter().any(|(name, _)| name == c) && !apart {
            out.push(format!(
                "the crates table does not say that {c} is a workspace of its own"
            ));
        }
    }
    out
}

/// README's table "The crates" names each crate of the workspace and the one
/// Cargo.toml excludes, said to be a workspace of its own, and no other; each
/// is a directory of the tree with a Cargo.toml.
#[test]
fn the_readme_counts_and_names_every_crate() {
    let toml = read("Cargo.toml");
    let members = workspace_members(&toml);
    let excluded = workspace_excluded(&toml);
    assert!(members.len() >= 2, "{members:?}");
    for c in members.iter().chain(&excluded) {
        assert!(
            workspace().join(c).join("Cargo.toml").is_file(),
            "Cargo.toml names {c}, which has no Cargo.toml of its own"
        );
    }
    let readme = read("README.md");
    let p = crate_table_problems(&members, &excluded, &readme);
    assert!(p.is_empty(), "README.md: {}", p.join("; "));
    assert_eq!(crate_table(&readme).len(), members.len() + excluded.len());
}

#[test]
fn a_crates_table_that_omits_a_crate_or_names_a_stranger_is_flagged() {
    let toml = "[workspace]\nmembers = [\"a\", \"b\"]\nexclude = [\"c\"]\nresolver = \"2\"\n";
    let members = workspace_members(toml);
    let excluded = workspace_excluded(toml);
    assert_eq!(members, ["a", "b"]);
    assert_eq!(excluded, ["c"]);
    assert!(workspace_excluded("[workspace]\nmembers = [\"a\"]\n").is_empty());
    let table = |rows: &str| {
        format!(
            "# T\n\nText.\n\n## The crates\n\n| Crate | What it does |\n|---|---|\n{rows}\n## Next\n\nMore.\n"
        )
    };
    let a = "| [`a`](a) | A. |\n";
    let b = "| [`b`](b) | B. |\n";
    let c = "| [`c`](c) | C, in its own workspace. |\n";
    let problems = |rows: &str| crate_table_problems(&members, &excluded, &table(rows));
    assert!(problems(&format!("{a}{b}{c}")).is_empty());
    assert_eq!(
        problems(&format!("{a}{c}")),
        ["the crates table has no row for b"]
    );
    assert_eq!(
        problems(&format!("{a}{b}{c}| [`d`](d) | D. |\n")),
        ["the crates table names d, which Cargo.toml neither lists nor excludes"]
    );
    assert_eq!(
        problems(&format!("{a}{b}| [`c`](c) | C. |\n")),
        ["the crates table does not say that c is a workspace of its own"]
    );
    assert_eq!(
        problems(&format!("{a}{a}{b}{c}")),
        ["the crates table has 2 rows for a"]
    );
    assert_eq!(
        crate_table_problems(&members, &excluded, "# T\n\nNo table.\n"),
        ["no table under \"## The crates\" names a crate in code"]
    );
}

/// Length padding (`padding.rs`): a stream's frame count and a whole answer's
/// bytes and time are Padme sizes above floors of 10 ticks and 16 KiB.
/// docs/api.md and the threat model name the rounding and both floors, and no
/// longer say that a stream's duration gives roughly the answer's length.
#[test]
fn the_length_padding_is_described_as_the_code_does_it() {
    let code = read("x0ne-infer/src/padding.rs");
    if !code.contains("pub fn padme(") {
        return;
    }
    assert!(
        code.contains("pub const MIN_TICKS: u64 = 10;")
            && code.contains("pub const MIN_WHOLE_BYTES: u64 = 16 * 1024;"),
        "a floor changed: say so in docs/api.md and THREAT_MODEL.md, then here"
    );
    none_says(&[
        "its duration, which gives roughly the answer's length",
        "its duration and roughly the answer's length",
        // The role frame goes with the 200 and the tenth frame 0.9 s later.
        "comes one second after the 200",
        // The turn holds the place: a full queue's 503s stop when it ends.
        "queue place is given back before that wait",
        "an error's body, which is not padded",
    ]);
    let capped = code.contains("pub const MAX_TICKS: u64 = 8_704;");
    let turns = code.contains("pub struct Turns(");
    for doc in ["docs/api.md", "x0ne-infer/THREAT_MODEL.md"] {
        let text = flat(&read(doc));
        for says in [
            "padme",
            "at least 10 frames",
            "at least 16 kib",
            "one second",
            "x0ne-pad",
        ] {
            assert!(text.contains(says), "{doc} does not say {says:?}");
        }
        assert!(
            !capped || (text.contains("8704") && text.contains("870.4 s")),
            "{doc} does not say that a stream stops at 8704 frames"
        );
        // One answer at a time, the next starting when the turn ends.
        assert!(
            !turns || text.contains("the model's turn"),
            "{doc} does not say that a chat runs in the model's turn"
        );
    }
}

/// The TD judges the owner's grant end on its own wall clock (`st.clock`, the
/// guest's, which the host sets) and the relay never compares it with its
/// own: no text says the TD stops at `not_after_unix` without that clock.
#[test]
fn the_grant_end_is_said_to_rest_on_the_tds_clock() {
    let serve = read("x0ne-infer/src/cloud_serve.rs");
    if !serve.contains("grant.check(&st.config.config, (st.clock)())") {
        return;
    }
    none_says(&[
        "stops serving at it",
        "serves at most until that expiry",
        "serves at most until the grant's expiry",
        "serves no request past",
    ]);
    let doc = flat(&read("docs/model-key.md"));
    for said in [
        "the td judges the grant's end on its own wall clock, and the host provides that clock",
        "never compares the td's clock with its own",
        "d more seconds of service past `not_after_unix`",
    ] {
        assert!(
            doc.contains(said),
            "docs/model-key.md no longer says {said:?}"
        );
    }
}

/// The first paragraph, or table row, of `text` that names the archive key:
/// the status line of README.md, docs/cloud.md and docs/attested-channel.md.
fn status_unit(rel: &str) -> String {
    units(&read(rel))
        .into_iter()
        .find(|u| u.contains("archive key"))
        .unwrap_or_else(|| panic!("{rel} names no archive key"))
}

/// While `transparency::RELEASE` is None, a release relay refuses every
/// reference with 100 even once the archive key is pinned (`bin/proxy.rs`
/// checks the key, then the log): each status line that says why the cloud
/// path does not start names the log and its 100 too.
#[test]
fn the_cloud_status_names_the_transparency_log_while_none_is_pinned() {
    let tlog = read("x0ne-infer/src/transparency.rs");
    if !tlog.contains("pub const RELEASE: Option<Pin> = None;") {
        return;
    }
    for rel in ["README.md", "docs/cloud.md", "docs/attested-channel.md"] {
        let status = status_unit(rel);
        assert!(
            status.contains("no transparency log") && status.contains("100"),
            "{rel}: the status line does not say that no transparency log is pinned (exit \
             100): {status}"
        );
    }
}

/// The client build docs/cloud.md gives ("## Building") is the one `make ci`
/// runs (step `build-cloud`, lane c) and the GitHub workflow runs, its
/// `cloud --help` included: a documented command that no CI builds rots.
#[test]
fn the_cloud_build_documented_is_the_one_ci_runs() {
    let doc = read("docs/cloud.md");
    let building = &doc[doc
        .find("\n## Building\n")
        .expect("docs/cloud.md: ## Building")..];
    let block = &building[building.find("```sh\n").expect("a sh block") + 6..];
    let mut lines = block.lines();
    let (build, help) = (lines.next().unwrap(), lines.next().unwrap());
    assert!(
        build.starts_with("cargo build --locked --release "),
        "{build}"
    );
    assert_eq!(help, "target/release/x0ne-engine cloud --help");

    let inside = read("ci/inside.sh");
    for list in ["STEPS=", "LANE_C="] {
        let line = inside
            .lines()
            .find(|l| l.starts_with(list))
            .unwrap_or_else(|| panic!("ci/inside.sh: {list}"));
        assert!(
            line.split(['"', ' ', '}']).any(|w| w == "build-cloud"),
            "{line}"
        );
    }
    let step = inside
        .lines()
        .find(|l| l.trim_start().starts_with("build-cloud)"))
        .expect("ci/inside.sh: build-cloud)");
    assert!(
        step.contains(build) && step.contains("release/x0ne-engine cloud --help"),
        "{step}"
    );
    let gh = read(".github/workflows/ci.yml");
    assert!(
        gh.contains(build) && gh.contains(help),
        ".github/workflows/ci.yml does not run the documented build"
    );
}
