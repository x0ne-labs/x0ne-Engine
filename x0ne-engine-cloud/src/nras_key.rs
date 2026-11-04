//! `--nras-key-file`: the NVIDIA API key x0ne-infer sends to NRAS as
//! `Authorization: Bearer`. NVIDIA calls the key optional today and says it
//! will be required for production deployments. It reaches the VM in its own
//! metadata item, never in `x0ne-config`, whose bytes are hashed into the
//! quote and published in the attestation document.

/// A key file is one line; anything bigger is not one.
pub const FILE_CAP: u64 = 4096;
/// The longest key taken; an NGC key is about 70 characters.
pub const MAX_KEY_BYTES: usize = 1024;
/// The shortest key taken: [`NrasKey::redact`] replaces each copy of the key,
/// and a key of 1 to 3 characters masked ordinary text (review of 2026-10-30).
pub const MIN_KEY_BYTES: usize = 16;

/// The key. Its Debug says a key is there, never which.
#[derive(Clone, PartialEq)]
pub struct NrasKey(zeroize::Zeroizing<String>);

impl std::fmt::Debug for NrasKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NrasKey(not printed)")
    }
}

impl NrasKey {
    /// One token of [`MIN_KEY_BYTES`] to [`MAX_KEY_BYTES`] visible ASCII
    /// characters, one trailing newline allowed: the rule x0ne-infer applies
    /// again in the VM (`nras_http::NrasKey`).
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let token = bytes
            .strip_suffix(b"\r\n")
            .or_else(|| bytes.strip_suffix(b"\n"))
            .unwrap_or(bytes);
        if token.is_empty() {
            return Err("the key is empty".into());
        }
        if token.len() < MIN_KEY_BYTES {
            return Err(format!(
                "the key is shorter than {MIN_KEY_BYTES} characters"
            ));
        }
        if token.len() > MAX_KEY_BYTES {
            return Err(format!("the key is longer than {MAX_KEY_BYTES} bytes"));
        }
        if !token.iter().all(|b| (0x21..=0x7e).contains(b)) {
            return Err(
                "the key is not one token of visible ASCII (no space or control character, \
                 one final newline at most)"
                    .into(),
            );
        }
        let text = String::from_utf8(token.to_vec()).map_err(|_| "not ASCII".to_string())?;
        Ok(Self(zeroize::Zeroizing::new(text)))
    }

    /// A regular file, not a link, owned by this user, readable by no one
    /// else (mode 0600 or stricter), of at most [`FILE_CAP`] bytes. Errors
    /// name neither the path nor the content.
    pub fn read(path: &std::path::Path) -> Result<Self, String> {
        use std::io::Read as _;
        use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _};
        let not_regular = "not a regular file (links, directories, devices and FIFOs are refused)";
        let f = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(path)
            .map_err(|e| match e.raw_os_error() {
                Some(libc::ELOOP) => not_regular.to_string(),
                _ => format!("cannot be opened: {}", e.kind()),
            })?;
        let m = f
            .metadata()
            .map_err(|e| format!("cannot be read: {}", e.kind()))?;
        if !m.is_file() {
            return Err(not_regular.into());
        }
        // SAFETY: geteuid has no preconditions and cannot fail.
        if m.uid() != unsafe { libc::geteuid() } {
            return Err("not owned by this user".into());
        }
        if m.mode() & 0o077 != 0 {
            return Err(
                "readable or writable by others than its owner: chmod 0600 it, then make a new key \
                 if others may have read it"
                    .into(),
            );
        }
        let too_large = || format!("larger than {FILE_CAP} bytes: not a key file");
        if m.len() > FILE_CAP {
            return Err(too_large());
        }
        let mut buf = zeroize::Zeroizing::new(Vec::new());
        f.take(FILE_CAP + 1)
            .read_to_end(&mut buf)
            .map_err(|e| format!("cannot be read: {}", e.kind()))?;
        if buf.len() as u64 > FILE_CAP {
            return Err(too_large());
        }
        Self::parse(&buf)
    }

    /// The key itself, for the one place it goes: the VM's metadata.
    pub(crate) fn expose(&self) -> &str {
        &self.0
    }

    /// `text` with the key, wherever it appears, replaced.
    pub fn redact(&self, text: &str) -> String {
        text.replace(self.0.as_str(), "(key not printed)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    const KEY: &str = "nvapi-CANARY0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL_-xyz";

    #[test]
    fn a_key_is_one_token_of_visible_ascii_and_never_printed() {
        for good in [KEY.to_string(), format!("{KEY}\n"), format!("{KEY}\r\n")] {
            let k = NrasKey::parse(good.as_bytes()).unwrap();
            assert_eq!(k.expose(), KEY);
            assert_eq!(k.redact(&format!("a{KEY}b")), "a(key not printed)b");
            assert_eq!(format!("{k:?}"), "NrasKey(not printed)");
        }
        for bad in [
            String::new(),
            "\n".into(),
            "nvapi-CANARY a".into(),
            "nvapi-CANARY\tb".into(),
            "nvapi-CANARY\u{e9}".into(),
            format!("{KEY}\n\n"),
            format!("\n{KEY}"),
            "nvapi-CANARY\u{1b}[2J".into(),
            format!("nvapi-CANARY{}", "x".repeat(MAX_KEY_BYTES)),
        ] {
            let e = NrasKey::parse(bad.as_bytes()).unwrap_err();
            assert!(!e.contains("CANARY"), "{bad:?}: {e}");
        }
    }

    /// A key under 16 characters is refused: `redact` replaces each copy of
    /// the key in NRAS's words, and a key of 1 to 3 characters would mask
    /// ordinary text there. An NGC key is about 70 characters.
    #[test]
    fn a_key_shorter_than_16_characters_is_refused() {
        let (sixteen, fifteen) = (&KEY[..16], &KEY[..15]);
        assert!(NrasKey::parse(sixteen.as_bytes()).is_ok());
        assert!(NrasKey::parse(format!("{sixteen}\n").as_bytes()).is_ok());
        let shorts = [
            "e".to_string(),
            "abc".into(),
            fifteen.into(),
            format!("{fifteen}\n"),
        ];
        for short in shorts {
            let e = NrasKey::parse(short.as_bytes()).unwrap_err();
            assert!(e.contains("16"), "{short:?}: {e}");
            assert!(
                !e.contains("CANARY") && !e.contains("abc"),
                "{short:?}: {e}"
            );
        }
    }

    /// Only this user may read the file; refusals name neither it nor what it holds.
    #[test]
    fn a_key_file_is_private_regular_and_never_quoted() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, body: String, mode: u32| {
            let p = dir.path().join(name);
            std::fs::write(&p, body).unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
            p
        };
        let good = write("good-key", format!("{KEY}\n"), 0o600);
        assert_eq!(NrasKey::read(&good).unwrap().expose(), KEY);
        assert_eq!(
            NrasKey::read(&write("owner-ro-key", KEY.into(), 0o400))
                .unwrap()
                .expose(),
            KEY
        );
        let link = dir.path().join("link-key");
        std::os::unix::fs::symlink(&good, &link).unwrap();
        for (p, says) in [
            (write("group-key", KEY.into(), 0o640), "0600"),
            (write("world-key", KEY.into(), 0o604), "0600"),
            (link, "not a regular file"),
            (dir.path().to_path_buf(), "not a regular file"),
            (dir.path().join("absent-key"), "cannot be opened"),
            (
                write("big-key", "x".repeat(FILE_CAP as usize + 1), 0o600),
                "bytes",
            ),
            (write("two-key", format!("{KEY} {KEY}"), 0o600), "one token"),
        ] {
            let e = NrasKey::read(&p).unwrap_err();
            assert!(e.contains(says), "{}: {e}", p.display());
            assert!(
                !e.contains("CANARY") && !e.contains("-key") && !e.contains("tmp"),
                "{}: {e}",
                p.display()
            );
        }
    }
}
