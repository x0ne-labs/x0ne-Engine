//! Whether x0ne-infer could collect evidence on this host, for the tests of
//! its exit 64 ("no evidence, not serving"). The ubuntu-24.04 runner shows a
//! root-owned configfs-tsm to every user (2026-11-05): its mere presence skipped them.

/// A configfs-tsm report directory this process may write: a report session
/// could be opened, so the exit 64 is not certain. Otherwise none can be.
pub fn may_collect_evidence() -> bool {
    // SAFETY: access(2) on a NUL-terminated path; it changes nothing.
    unsafe { libc::access(c"/sys/kernel/config/tsm/report".as_ptr(), libc::W_OK) == 0 }
}
