//! The TCB policy (`tcb::TcbPolicy`): the code paths use the production
//! values, the endpoints ask for Intel's latest recovery, and the documents
//! state the floor the code enforces. A failure here means a path verifies
//! Intel's TCB under a policy written by hand, or a document no longer says
//! which recovery the verifier requires.

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

fn flat(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// SECURITY: a policy is built by `TcbPolicy::production()` or
/// `TcbPolicy::tightened()`, both in `tcb.rs`. A struct literal elsewhere in
/// a crate's sources is a hand-written floor; tests build theirs under
/// `tests/` or in `tcb.rs`'s own test module.
#[test]
fn no_source_outside_tcb_rs_writes_a_policy_by_hand() {
    let root = workspace();
    let mut files = Vec::new();
    for krate in ["x0ne-infer", "x0ne-confidential", "x0ne-engine-cloud"] {
        rust_files(&root.join(krate).join("src"), &mut files);
    }
    assert!(files.len() > 20, "the sweep found {} files", files.len());
    for f in files {
        if f.ends_with("x0ne-infer/src/tcb.rs") {
            continue;
        }
        let text = std::fs::read_to_string(&f).expect("readable");
        for pat in [
            "TcbPolicy {",
            "min_tcb_evaluation_data_number:",
            "max_collateral_age_secs:",
        ] {
            assert!(
                !text.contains(pat),
                "{} writes `{pat}`: build the policy with TcbPolicy::production() or tightened()",
                f.display()
            );
        }
    }
}

/// Both production anchor sets carry the production policy.
#[test]
fn the_production_anchors_carry_the_production_policy() {
    use x0ne_infer::tcb::TcbPolicy;
    assert_eq!(
        x0ne_infer::remote_verify::Anchors::production().tcb_policy,
        TcbPolicy::production()
    );
    assert_eq!(
        x0ne_infer::attest_v2::TdAnchors::production().tcb_policy,
        TcbPolicy::production()
    );
    let main = read("x0ne-infer/src/main.rs");
    assert_eq!(
        main.matches("TcbPolicy::production()").count(),
        1,
        "main.rs: one production policy, bound once, for the start-up verdict, the PCK CRL, \
         the platform's standing and its hourly re-check"
    );
    assert!(
        !main.contains("TcbPolicy::strict()") && !main.contains("with_mode("),
        "main.rs: the TD serves under the production mode; strict is the client's choice"
    );
}

/// The documents name the floor and the age limit the code enforces.
#[test]
fn the_documents_state_the_floor_and_the_age_limit() {
    use x0ne_infer::tcb::TcbPolicy;
    let min = TcbPolicy::MIN_TCB_EVALUATION_DATA_NUMBER;
    for doc in ["docs/attested-channel.md", "x0ne-infer/THREAT_MODEL.md"] {
        let text = flat(&read(doc));
        assert!(
            text.contains(&format!("tcb evaluation {min}")),
            "{doc} does not name TCB evaluation {min}"
        );
        assert!(text.contains("update=early"), "{doc}: update=early");
        assert!(
            text.contains("24 hours") || text.contains("a day"),
            "{doc}: the age limit"
        );
    }
}

/// `x0ne-engine cloud` checks the TCB policy options against the relay's
/// floor before any quote: its two constants are this build's.
#[test]
fn the_lifecycle_checks_the_options_against_this_builds_floor() {
    use x0ne_infer::tcb::TcbPolicy;
    let lib = read("x0ne-engine-cloud/src/lib.rs");
    for line in [
        format!(
            "pub const TCB_EVALUATION_FLOOR: u64 = {};",
            TcbPolicy::MIN_TCB_EVALUATION_DATA_NUMBER
        ),
        format!(
            "pub const MAX_COLLATERAL_AGE_SECS: u64 = {};",
            TcbPolicy::MAX_COLLATERAL_AGE_SECS
        ),
    ] {
        assert!(lib.contains(&line), "x0ne-engine-cloud/src/lib.rs: {line}");
    }
}
