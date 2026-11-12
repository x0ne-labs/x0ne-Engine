//! The served-model digest, v2: one sha256 over a `sha256sum` manifest of the
//! files a generative model is served from, sorted by name. Definition and the
//! coreutils command that recomputes it: docs/cloud.md, "The served-model digest".

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

/// First line of the manifest; no `sha256sum` line can equal it.
pub const DOMAIN: &str = "x0ne-model-digest/2\n";

/// A manifest name: `[A-Za-z0-9._-]`, 1 to 255 bytes, not `.` or `..`.
/// No space and no newline, so name and line boundaries cannot shift.
pub fn is_manifest_name(name: &str) -> bool {
    (1..=255).contains(&name.len())
        && name != "."
        && name != ".."
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The digest from `(name, sha256 hex)` pairs, in any order. A duplicate name,
/// an unsafe name or a malformed sha256 is refused.
pub fn from_hashes<'a>(
    files: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<String, String> {
    let mut lines: BTreeMap<&str, &str> = BTreeMap::new();
    for (name, sha) in files {
        if !is_manifest_name(name) {
            return Err(format!("{name:?}: not a manifest file name"));
        }
        if !is_sha256_hex(sha) {
            return Err(format!("{name}: sha256 {sha:?} is not 64 lowercase hex"));
        }
        if lines.insert(name, sha).is_some() {
            return Err(format!("{name} listed twice"));
        }
    }
    if lines.is_empty() {
        return Err("no file to digest".into());
    }
    let mut h = Sha256::new();
    h.update(DOMAIN.as_bytes());
    for (name, sha) in lines {
        h.update(sha.as_bytes());
        h.update(b"  ");
        h.update(name.as_bytes());
        h.update(b"\n");
    }
    Ok(hex(&h.finalize()))
}

/// The digest of named contents, each file hashed on its own thread.
pub fn of_contents(files: &[(&str, &[u8])]) -> Result<String, String> {
    let hashes: Vec<String> = std::thread::scope(|s| {
        let jobs: Vec<_> = files
            .iter()
            .map(|(_, bytes)| s.spawn(move || hex(&Sha256::digest(bytes))))
            .collect();
        jobs.into_iter()
            .map(|j| j.join().expect("sha256 thread"))
            .collect()
    });
    for sha in &hashes {
        crate::hash_passes::note(sha);
    }
    from_hashes(
        files
            .iter()
            .zip(&hashes)
            .map(|((name, _), sha)| (*name, sha.as_str())),
    )
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(b: &[u8]) -> String {
        hex(&Sha256::digest(b))
    }

    /// The manifest as coreutils prints it, sorted by name, hashed by hand: the
    /// definition docs/cloud.md gives a third party.
    #[test]
    fn the_digest_is_the_sha256_of_the_sorted_sha256sum_manifest() {
        let files: [(&str, &[u8]); 4] = [
            ("tokenizer.json", b"tok"),
            ("model-00002-of-00002.safetensors", b"two"),
            ("config.json", b"cfg"),
            ("model.safetensors.index.json", b"idx"),
        ];
        let mut manifest = String::from(DOMAIN);
        for name in [
            "config.json",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
        ] {
            let bytes = files.iter().find(|(n, _)| *n == name).unwrap().1;
            manifest.push_str(&format!("{}  {name}\n", sha(bytes)));
        }
        assert_eq!(of_contents(&files).unwrap(), sha(manifest.as_bytes()));
    }

    /// A failure means the digest depends on the order files were listed or loaded.
    #[test]
    fn order_does_not_change_the_digest_and_content_does() {
        let a: [(&str, &[u8]); 3] = [
            ("config.json", b"c"),
            ("x.safetensors", b"1"),
            ("y.safetensors", b"2"),
        ];
        let b: [(&str, &[u8]); 3] = [
            ("y.safetensors", b"2"),
            ("config.json", b"c"),
            ("x.safetensors", b"1"),
        ];
        assert_eq!(of_contents(&a).unwrap(), of_contents(&b).unwrap());
        // The same bytes under swapped names: another manifest.
        let swapped: [(&str, &[u8]); 3] = [
            ("config.json", b"c"),
            ("x.safetensors", b"2"),
            ("y.safetensors", b"1"),
        ];
        assert_ne!(of_contents(&a).unwrap(), of_contents(&swapped).unwrap());
        let changed: [(&str, &[u8]); 3] = [
            ("config.json", b"c"),
            ("x.safetensors", b"1"),
            ("y.safetensors", b"3"),
        ];
        assert_ne!(of_contents(&a).unwrap(), of_contents(&changed).unwrap());
    }

    /// SECURITY: a name with a space or a newline could move a line boundary;
    /// a duplicate makes the manifest ambiguous.
    #[test]
    fn unsafe_names_duplicates_and_bad_hashes_are_refused() {
        let s = sha(b"x");
        for bad in ["", ".", "..", "a b", "a\nb", "sub/x", "x\\y", "é.json"] {
            assert!(from_hashes([(bad, s.as_str())]).is_err(), "{bad:?}");
        }
        assert!(from_hashes([("a", s.as_str()), ("a", s.as_str())])
            .unwrap_err()
            .contains("twice"));
        let upper = s.to_uppercase();
        assert!(from_hashes([("a", upper.as_str())]).is_err());
        assert!(from_hashes([("a", "00")]).is_err());
        assert!(from_hashes(std::iter::empty()).is_err());
    }

    /// One call hashes each file once, whatever their number: the half of "a load hashes each
    /// file once" that this function holds (a loader holds the other: one call per load). Bytes
    /// of this test's own, so that the tests beside it count apart.
    #[test]
    fn one_call_hashes_each_file_once() {
        let (a, b): (&[u8], &[u8]) = (b"model_digest one pass a", b"model_digest one pass b");
        of_contents(&[("a.gguf", a), ("b.gguf", b)]).unwrap();
        assert_eq!(crate::hash_passes::of(&sha(a)), 1);
        assert_eq!(crate::hash_passes::of(&sha(b)), 1);
        of_contents(&[("a.gguf", a)]).unwrap();
        assert_eq!(crate::hash_passes::of(&sha(a)), 2);
        assert_eq!(crate::hash_passes::of(&sha(b)), 1);
    }

    /// v1 hashed the concatenated bytes; v2 a manifest. The same files give two
    /// different values, so a v1 pin can never pass for v2 by accident.
    #[test]
    fn v1_and_v2_differ_on_the_same_files() {
        let (w, c, t): (&[u8], &[u8], &[u8]) = (b"weights", b"config", b"tokenizer");
        let v1 = sha(&[w, c, t].concat());
        let v2 = of_contents(&[
            ("model.safetensors", w),
            ("config.json", c),
            ("tokenizer.json", t),
        ])
        .unwrap();
        assert_ne!(v1, v2);
    }
}
