//! Every exit code a binary can end with is listed where its user reads:
//! `x0ne-infer --help` and the header of `x0ne-engine-proxy`. A code missing
//! from the list reads as "unknown failure".

use std::path::Path;
use std::process::Command;

fn read(rel: &str) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// Numbers followed by a space and a word other than "as": "96 model ..."
/// counts, "95-98 as above" does not.
fn described(text: &str) -> Vec<u32> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out = Vec::new();
    for pair in words.windows(2) {
        let n = pair[0].trim_start_matches('(').trim_end_matches(',');
        let next = pair[1];
        if let Ok(code) = n.parse::<u32>() {
            if next != "as" && next.chars().next().is_some_and(char::is_alphabetic) {
                out.push(code);
            }
        }
    }
    out
}

/// Literal codes passed to `name(` in `src`: `exit(72)`, `refused(96, ...`.
fn literal_codes(src: &str, name: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for (at, _) in src.match_indices(name) {
        let digits: String = src[at + name.len()..]
            .trim_start()
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        if let Ok(code) = digits.parse() {
            out.push(code);
        }
    }
    out
}

#[test]
fn the_help_describes_each_cloud_mode_exit_code() {
    let out = Command::new(env!("CARGO_BIN_EXE_x0ne-infer"))
        // cargo sets LD_LIBRARY_PATH for tests; a release x0ne-infer refuses it (73).
        .env_remove("LD_LIBRARY_PATH")
        .arg("--help")
        .output()
        .expect("the binary runs");
    let help =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    let listed = described(&help);
    for code in (95..=98).chain([101, 103]) {
        assert!(
            listed.contains(&code),
            "--help does not describe exit {code}"
        );
    }
}

#[test]
fn the_proxy_header_lists_every_code_the_relay_refuses_with() {
    let proxy = read("src/bin/proxy.rs");
    let header: String = proxy
        .lines()
        .take_while(|l| l.starts_with("//!"))
        .collect::<Vec<_>>()
        .join(" ");
    let relay = read("src/relay.rs");
    let mut codes = literal_codes(&relay, "refused(");
    codes.extend(literal_codes(&proxy, "refuse_early("));
    assert!(
        !codes.is_empty(),
        "no literal refusal code found: re-read this test"
    );
    for code in codes {
        assert!(
            header.contains(&code.to_string()),
            "the relay refuses with {code}; proxy.rs's header does not list it"
        );
    }
}
