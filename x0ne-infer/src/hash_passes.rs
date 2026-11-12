//! Whole-file hash passes, counted in tests only: the cloud intake hashes the
//! weights once before the attestation, and a test holds it to that. Keyed by
//! the sha256 found: tests run at once, each counts passes over its own bytes.
//! Outside tests `note` does nothing.

#[cfg(test)]
static SEEN: std::sync::Mutex<Option<std::collections::HashMap<String, usize>>> =
    std::sync::Mutex::new(None);

/// One pass over bytes whose sha256 is `sha256_hex` ended.
#[cfg(test)]
pub(crate) fn note(sha256_hex: &str) {
    let mut seen = SEEN.lock().unwrap();
    *seen
        .get_or_insert_with(Default::default)
        .entry(sha256_hex.to_string())
        .or_default() += 1;
}

#[cfg(not(test))]
#[inline]
pub(crate) fn note(_sha256_hex: &str) {}

/// The passes so far over bytes of that sha256.
#[cfg(test)]
pub(crate) fn of(sha256_hex: &str) -> usize {
    let seen = SEEN.lock().unwrap();
    seen.as_ref()
        .and_then(|m| m.get(sha256_hex).copied())
        .unwrap_or(0)
}
