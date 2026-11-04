//! The files that handle the text of a prompt or of an answer carry no log
//! macro, so nothing of a conversation can reach the journal from them,
//! whatever a line names. Trail of Bits found inference data logged by client
//! code in Meta's Private Processing (2025, finding 5). Each such file says
//! so in its header; this test holds the header to the file, and the list of
//! files to the paths that must make the claim.

use std::path::{Path, PathBuf};

/// The sentence a file's header carries.
const CLAIM: &str = "No log macro in this file";

/// The files that must carry it: the streamed body, its SSE encoding, the
/// local OpenAI-compatible API and the e2e gateway, which all hold the
/// answer's text (the gateway the question's too).
const MUST_CLAIM: &[&str] = &[
    "src/stream.rs",
    "src/sse.rs",
    "src/local_api.rs",
    "src/e2e_gateway.rs",
];

/// What a log line looks like: `tracing` and `log` macros, the standard
/// output ones, and the raw handles `io::stdout()` and `io::stderr()`, which
/// the unit's journal collects just the same.
const MACROS: &[&str] = &[
    "stdout()",
    "stderr()",
    "println!",
    "eprintln!",
    "print!",
    "eprint!",
    "dbg!",
    "trace!",
    "debug!",
    "info!",
    "warn!",
    "error!",
    "event!",
    "span!",
];

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("entry").path())
        .collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

fn read(rel: &str) -> String {
    let p = crate_root().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// SECURITY: a file that says it has no log macro has none, in code or in
/// its tests, comments aside. A failure names the file, the line and the macro.
#[test]
fn the_files_that_say_no_log_macro_have_none() {
    let mut files = Vec::new();
    rust_files(&crate_root().join("src"), &mut files);
    let mut claiming = Vec::new();
    for path in files {
        let text = std::fs::read_to_string(&path).unwrap();
        if !text.contains(CLAIM) {
            continue;
        }
        let rel = path
            .strip_prefix(crate_root())
            .unwrap()
            .display()
            .to_string();
        for (n, line) in text.lines().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            for m in MACROS {
                assert!(
                    !code.contains(m),
                    "{rel}:{}: {m} in a file whose header says {CLAIM:?}: {line}",
                    n + 1
                );
            }
        }
        claiming.push(rel);
    }
    for must in MUST_CLAIM {
        assert!(
            claiming.iter().any(|c| c == must),
            "{must} is not among the claiming files: {claiming:?}"
        );
    }
}

/// The claim cannot be dropped from a file that handles the text: the header
/// is what the test above keys on.
#[test]
fn the_answer_paths_make_the_claim() {
    for rel in MUST_CLAIM {
        assert!(read(rel).contains(CLAIM), "{rel} no longer says {CLAIM:?}");
    }
}
