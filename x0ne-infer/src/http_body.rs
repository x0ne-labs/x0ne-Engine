//! An HTTP body read whole, never past a cap. `bytes()`, `text()` and
//! `json()` buffer whatever a server sends before the caller can check a
//! length: a server that never ends the body holds the reader until timeout.

/// Why [`read_body`] failed.
#[derive(Debug)]
pub enum BodyError {
    /// Larger than the cap: the answer itself is wrong.
    Over(usize),
    /// The transfer broke: another request may pass.
    Transfer(reqwest::Error),
}

impl std::fmt::Display for BodyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BodyError::Over(cap) => write!(f, "larger than the {cap}-byte cap"),
            BodyError::Transfer(e) => e.fmt(f),
        }
    }
}

/// The body of `resp`, at most `cap` bytes: the announced length is checked
/// first, then each chunk as it arrives.
pub async fn read_body(mut resp: reqwest::Response, cap: usize) -> Result<Vec<u8>, BodyError> {
    if resp.content_length().is_some_and(|n| n > cap as u64) {
        return Err(BodyError::Over(cap));
    }
    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(BodyError::Transfer)? {
        if body.len() + chunk.len() > cap {
            return Err(BodyError::Over(cap));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// [`read_body`], its error as text.
pub async fn read_capped(resp: reqwest::Response, cap: usize) -> Result<Vec<u8>, String> {
    read_body(resp, cap).await.map_err(|e| e.to_string())
}
