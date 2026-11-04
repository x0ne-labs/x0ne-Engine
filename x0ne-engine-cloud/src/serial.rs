//! Boot diagnostics: the VM's serial console, saved before the VM is deleted
//! on a boot timeout or a refusal of x0ne-infer read on it (`td_console`). The
//! text comes from the VM, so it is untrusted: it is written to a 0600 file and
//! not printed to the terminal, save the one cleaned line `td_console` returns.

use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;

use crate::provider::{Instance, Provider};
use crate::session::Paths;

/// Saves what the provider returns; the error says why nothing was saved.
pub async fn save_boot_log(
    provider: &dyn Provider,
    instance: &Instance,
    paths: &Paths,
) -> Result<PathBuf, String> {
    let text = provider
        .serial_output(instance)
        .await
        .map_err(|e| e.to_string())?;
    let dir = paths.state_dir().map_err(|e| e.to_string())?;
    let file = dir.join(format!(
        "serial-{}-{}.txt",
        instance.name,
        crate::stamp_utc(crate::now_unix())
    ));
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&file)
        .map_err(|e| format!("{}: {e}", file.display()))?;
    f.write_all(text.as_bytes())
        .map_err(|e| format!("{}: {e}", file.display()))?;
    Ok(file)
}
