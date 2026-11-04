//! The relay's files compiled into this crate by `#[path]` (`lib.rs`): what
//! opens a file or a socket in them is left out under the `x0ne_sdk` cfg
//! (`build.rs`), so the browser's build reads no disk and fetches nothing of
//! its own. Until 2026-10-27 `reference::default_path_in` and `default_path`
//! (a directory listing) were compiled here.

const SHARED: &[(&str, &str)] = &[
    ("ccel.rs", include_str!("../../x0ne-infer/src/ccel.rs")),
    (
        "cloud_config_types.rs",
        include_str!("../../x0ne-infer/src/cloud_config_types.rs"),
    ),
    ("hex.rs", include_str!("../../x0ne-infer/src/hex.rs")),
    (
        "gce_endorsement.rs",
        include_str!("../../x0ne-infer/src/gce_endorsement.rs"),
    ),
    (
        "decoding.rs",
        include_str!("../../x0ne-infer/src/decoding.rs"),
    ),
    (
        "json_schema.rs",
        include_str!("../../x0ne-infer/src/json_schema.rs"),
    ),
    ("nras.rs", include_str!("../../x0ne-infer/src/nras.rs")),
    ("pcs.rs", include_str!("../../x0ne-infer/src/pcs.rs")),
    (
        "reference.rs",
        include_str!("../../x0ne-infer/src/reference.rs"),
    ),
    ("tcb.rs", include_str!("../../x0ne-infer/src/tcb.rs")),
    ("tdvf.rs", include_str!("../../x0ne-infer/src/tdvf.rs")),
];

/// What reads a disk or opens a socket.
const IO: &[&str] = &[
    "std::fs::",
    "fsread::",
    "read_dir(",
    "File::open",
    "std::net::",
    "TcpStream",
    "reqwest::",
    "ureq::",
    "http_get",
];

/// The attributes above the top-level item (a line at column 0 that opens a
/// `fn`, an `impl` or a `mod`) enclosing line `at`.
fn enclosing_attributes(lines: &[&str], at: usize) -> (String, Vec<String>) {
    let opens = |l: &str| {
        let l = l
            .trim_start_matches("pub(crate) ")
            .trim_start_matches("pub ");
        l.starts_with("fn ")
            || l.starts_with("async fn ")
            || l.starts_with("impl")
            || l.starts_with("mod ")
            || l.starts_with("const ")
            || l.starts_with("static ")
    };
    let header = (0..=at)
        .rev()
        .find(|&i| opens(lines[i]))
        .unwrap_or_else(|| panic!("line {}: no enclosing item", at + 1));
    let attributes = (0..header)
        .rev()
        .map(|i| lines[i])
        .take_while(|l| l.starts_with("#[") || l.starts_with("///") || l.starts_with("//"))
        .map(str::to_string)
        .collect();
    (lines[header].to_string(), attributes)
}

#[test]
fn what_reads_a_disk_or_a_socket_is_left_out_of_this_crate() {
    let mut found = 0;
    for (name, src) in SHARED {
        let lines: Vec<&str> = src.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            if !IO.iter().any(|io| code.contains(io)) {
                continue;
            }
            found += 1;
            let (header, attributes) = enclosing_attributes(&lines, i);
            assert!(
                attributes.iter().any(|a| a.contains("not(x0ne_sdk)")),
                "{name}:{}: {} is compiled into x0ne-sdk ({header})",
                i + 1,
                line.trim()
            );
        }
    }
    // The sweep reads something: the relay's fetches and file reads exist.
    assert!(found > 10, "{found} reads found");
}
