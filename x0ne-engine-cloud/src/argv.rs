//! Names an argument a command line does not know without printing its value:
//! a key typed on a command line must not reach a terminal or a log.
//! One file, two crates (x0ne-infer includes it by `#[path]`), as `color.rs`.

/// `"--name"` for a flag (an `=value` is cut and said so), else a placeholder.
pub fn shown(arg: &str) -> String {
    let flag = arg.strip_prefix("--").map(|rest| {
        rest.split_once('=')
            .map_or((rest, false), |(n, _)| (n, true))
    });
    match flag {
        Some((name, false)) if is_flag_name(name) => format!("\"--{name}\""),
        Some((name, true)) if is_flag_name(name) => format!("\"--{name}=\" (value not printed)"),
        _ => "(not printed: not a --flag, it may be a secret)".to_string(),
    }
}

/// The refusal of a value given to a known flag: the flag and what it takes,
/// never the value, which may be a secret typed in the wrong place.
pub fn refused(flag: &str, expected: &str) -> String {
    format!("{flag}: expected {expected} (the value given is not printed)")
}

/// 64 hex digits: the shape of a 32-byte key. Where a command line takes the
/// path of a key file, this is the key itself, and it is refused as such.
pub fn is_hex_key(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

/// A path for a message, unless it holds a run of 32 or more hex digits: a
/// key pasted where a file name goes. The path is then not printed.
pub fn path_shown(path: &str) -> String {
    let longest = path
        .split(|c: char| !c.is_ascii_hexdigit())
        .map(str::len)
        .max()
        .unwrap_or(0);
    if longest >= 32 {
        "(path not printed: it holds a key-like run of hex digits)".to_string()
    } else {
        format!("{path:?}")
    }
}

/// Lowercase ASCII, digits and `-`, a letter first, 32 bytes at most: a hex key is 64.
fn is_flag_name(name: &str) -> bool {
    (1..=32).contains(&name.len())
        && name.starts_with(|c: char| c.is_ascii_lowercase())
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

#[cfg(test)]
mod tests {
    use super::shown;

    #[test]
    fn a_flag_is_named_and_nothing_else_is_printed() {
        let key = "5ec7".repeat(16);
        assert_eq!(shown("--insecure"), "\"--insecure\"");
        for arg in [
            format!("--key-hex={key}"),
            key.clone(),
            format!("X0NE_INFER_KEY_HEX={key}"),
            format!("--{key}"),
            format!("-{key}"),
            format!("--Key={key}"),
        ] {
            let s = shown(&arg);
            assert!(!s.contains("5ec75ec7"), "{arg}: {s}");
        }
        assert_eq!(
            shown(&format!("--key-hex={key}")),
            "\"--key-hex=\" (value not printed)"
        );
    }

    #[test]
    fn a_refusal_names_the_flag_and_a_key_is_told_from_a_path() {
        assert_eq!(
            super::refused("--max-queue", "an integer"),
            "--max-queue: expected an integer (the value given is not printed)"
        );
        assert!(super::is_hex_key(&"5eC7".repeat(16)));
        assert_eq!(super::path_shown("/run/x0ne/key"), "\"/run/x0ne/key\"");
        let key = "5ec7".repeat(16);
        for p in [
            key.clone(),
            format!("/tmp/{key}"),
            format!("{}x", &key[..32]),
        ] {
            assert!(!super::path_shown(&p).contains("5ec75ec7"), "{p}");
        }
        for not_key in ["/run/key", "5ec7", &"5ec7".repeat(17), &"zz".repeat(32)] {
            assert!(!super::is_hex_key(not_key), "{not_key}");
        }
    }
}
