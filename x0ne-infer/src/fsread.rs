//! Model and config files: regular files only, never through a link, never
//! past a per-file cap. `std::fs::read` follows a link to `/dev/zero` until
//! memory runs out and blocks forever on a FIFO (2026-10-11).

use std::io::Read;
use std::path::Path;

/// `config.json`, `calibrator.json`, the shard index, a registry record.
pub const CONFIG_CAP: u64 = 1 << 20;
/// `tokenizer.json`. Qwen3's is 11 MB (2026-09-19).
pub const TOKENIZER_CAP: u64 = 64 << 20;
/// One `model.safetensors` (or one shard), read or mapped whole.
pub const WEIGHTS_CAP: u64 = 128 << 30;

/// The cap for a file of a model directory, by name.
pub fn cap_for(file: &str) -> u64 {
    if file.ends_with(".safetensors") || file.ends_with(".sealed") {
        WEIGHTS_CAP
    } else if file == "tokenizer.json" {
        TOKENIZER_CAP
    } else {
        CONFIG_CAP
    }
}

fn not_regular(path: &Path) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!(
            "{}: not a regular file (links, devices and FIFOs are refused)",
            path.display()
        ),
    )
}

/// Open `path` for reading: refuses a link as the last component, does not
/// block on a FIFO, and checks the OPENED descriptor is a regular file.
pub fn open_regular(path: &Path) -> std::io::Result<(std::fs::File, u64)> {
    let mut open = std::fs::OpenOptions::new();
    open.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        open.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC);
    }
    #[cfg(not(unix))]
    if std::fs::symlink_metadata(path)?.file_type().is_symlink() {
        return Err(not_regular(path));
    }
    let f = open.open(path).map_err(|e| {
        #[cfg(unix)]
        if e.raw_os_error() == Some(libc::ELOOP) {
            return not_regular(path);
        }
        e
    })?;
    let meta = f.metadata()?;
    if !meta.is_file() {
        return Err(not_regular(path));
    }
    Ok((f, meta.len()))
}

/// Read a whole regular file of at most `cap` bytes. The size is checked
/// before any allocation and again while reading (a file that grows).
pub fn read_capped(path: &Path, cap: u64) -> std::io::Result<Vec<u8>> {
    let (f, len) = open_regular(path)?;
    read_opened(f, len, cap, path)
}

/// [`read_capped`] on a file [`open_regular`] gave: checks made on its
/// descriptor hold for the bytes returned, whatever the path names since.
pub fn read_opened(f: std::fs::File, len: u64, cap: u64, path: &Path) -> std::io::Result<Vec<u8>> {
    let too_large = || {
        std::io::Error::new(
            std::io::ErrorKind::FileTooLarge,
            format!("{}: larger than the {cap}-byte cap", path.display()),
        )
    };
    if len > cap {
        return Err(too_large());
    }
    let mut buf = Vec::new();
    buf.try_reserve_exact(usize::try_from(len).map_err(|_| too_large())?)
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::OutOfMemory))?;
    // One byte past the cap tells "exactly the cap" from "more".
    f.take(cap + 1).read_to_end(&mut buf)?;
    if buf.len() as u64 > cap {
        return Err(too_large());
    }
    Ok(buf)
}

/// [`read_capped`] with the cap [`cap_for`] gives `file`.
pub fn read_model_file(dir: &Path, file: &str) -> std::io::Result<Vec<u8>> {
    read_capped(&dir.join(file), cap_for(file))
}

/// Map a non-empty regular file of at most `cap` bytes, read-only: the weights
/// stay in the page cache (the TD's tmpfs) instead of a heap copy.
pub fn map_capped(path: &Path, cap: u64) -> std::io::Result<memmap2::Mmap> {
    let (f, len) = open_regular(path)?;
    if len > cap {
        return Err(std::io::Error::new(
            std::io::ErrorKind::FileTooLarge,
            format!("{}: larger than the {cap}-byte cap", path.display()),
        ));
    }
    if len == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: empty", path.display()),
        ));
    }
    // SAFETY: a read-only map of a regular file, opened without following a link.
    // A writer of the file changes the mapped bytes: the model store is the TCB's
    // (THREAT_MODEL.md, "Model files mapped, not copied").
    let map = unsafe { memmap2::Mmap::map(&f)? };
    if map.len() as u64 != len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: changed size while mapped", path.display()),
        ));
    }
    Ok(map)
}

/// [`map_capped`] with the cap [`cap_for`] gives `file`.
pub fn map_model_file(dir: &Path, file: &str) -> std::io::Result<memmap2::Mmap> {
    map_capped(&dir.join(file), cap_for(file))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Run `f` on a thread; a call that blocks (FIFO) or reads forever
    /// (`/dev/zero`) fails the test instead of hanging it.
    fn within<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(Duration::from_secs(5))
            .expect("the read blocked or ran away: a hostile file was not refused")
    }

    fn dir_with(file: &str, bytes: &[u8]) -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join(file), bytes).unwrap();
        d
    }

    #[cfg(unix)]
    #[test]
    fn a_config_linked_to_dev_zero_is_refused_not_read() {
        let d = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink("/dev/zero", d.path().join("config.json")).unwrap();
        let p = d.path().to_path_buf();
        let err = within(move || read_model_file(&p, "config.json")).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput, "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn dev_zero_itself_is_refused_not_read() {
        let err = within(|| read_capped(Path::new("/dev/zero"), CONFIG_CAP)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput, "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn weights_named_by_a_fifo_are_refused_without_blocking() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("model.safetensors");
        let c = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
        // SAFETY: `c` is a valid NUL-terminated path that outlives the call.
        assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
        let p = d.path().to_path_buf();
        let err = within(move || read_model_file(&p, "model.safetensors")).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput, "{err}");
    }

    #[test]
    fn a_directory_named_like_the_tokenizer_is_refused() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir(d.path().join("tokenizer.json")).unwrap();
        let err = read_model_file(d.path(), "tokenizer.json").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput, "{err}");
    }

    #[test]
    fn a_config_over_its_cap_is_refused_and_one_at_the_cap_is_read() {
        let d = dir_with("config.json", &vec![b' '; CONFIG_CAP as usize + 1]);
        let err = read_model_file(d.path(), "config.json").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::FileTooLarge, "{err}");
        let d = dir_with("config.json", &vec![b' '; CONFIG_CAP as usize]);
        assert_eq!(
            read_model_file(d.path(), "config.json").unwrap().len() as u64,
            CONFIG_CAP
        );
    }

    #[test]
    fn caps_follow_the_file_name() {
        assert_eq!(cap_for("config.json"), CONFIG_CAP);
        assert_eq!(cap_for("calibrator.json"), CONFIG_CAP);
        assert_eq!(cap_for("model.safetensors.index.json"), CONFIG_CAP);
        assert_eq!(cap_for("tokenizer.json"), TOKENIZER_CAP);
        assert_eq!(cap_for("model-00001-of-00002.safetensors"), WEIGHTS_CAP);
    }

    #[cfg(unix)]
    #[test]
    fn a_mapped_file_is_refused_like_a_read_one() {
        let d = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink("/dev/zero", d.path().join("model.safetensors")).unwrap();
        let p = d.path().to_path_buf();
        let err = within(move || map_model_file(&p, "model.safetensors")).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput, "{err}");
        let err = within(|| map_capped(Path::new("/dev/zero"), CONFIG_CAP)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput, "{err}");

        let d = dir_with("model-00001-of-00002.safetensors", b"");
        let err = map_model_file(d.path(), "model-00001-of-00002.safetensors").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{err}");
        let d = dir_with("config.json", &vec![b' '; CONFIG_CAP as usize + 1]);
        let err = map_model_file(d.path(), "config.json").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::FileTooLarge, "{err}");
        let d = dir_with("model.safetensors", b"weights");
        assert_eq!(
            &map_model_file(d.path(), "model.safetensors").unwrap()[..],
            b"weights"
        );
    }

    #[test]
    fn a_missing_file_keeps_its_not_found_kind() {
        let d = tempfile::tempdir().unwrap();
        let err = read_model_file(d.path(), "calibrator.json").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    /// A raw `std::fs::read` on a loading path reopens the /dev/zero and FIFO
    /// holes the tests above close (2026-10-11).
    #[test]
    fn model_loading_paths_read_only_through_this_module() {
        for (name, src) in [
            ("main.rs", include_str!("main.rs")),
            ("bin/engine.rs", include_str!("bin/engine.rs")),
            ("shards.rs", include_str!("shards.rs")),
            ("registry/mod.rs", include_str!("registry/mod.rs")),
            ("registry/store.rs", include_str!("registry/store.rs")),
        ] {
            let body = &src[..src.find("#[cfg(test)]").unwrap_or(src.len())];
            for raw in ["std::fs::read(", "fs::read_to_string("] {
                assert!(!body.contains(raw), "{name} calls {raw} outside its tests");
            }
        }
    }
}
