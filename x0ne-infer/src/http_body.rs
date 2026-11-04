//! An HTTP body read whole, never past a cap. `bytes()`, `text()` and
//! `json()` buffer whatever a server sends before the caller can check a
//! length: a server that never ends the body holds the reader until timeout.

/// The body of `resp`, at most `cap` bytes: the announced length is checked
/// first, then each chunk as it arrives.
pub async fn read_capped(mut resp: reqwest::Response, cap: usize) -> Result<Vec<u8>, String> {
    let over = || format!("larger than the {cap}-byte cap");
    if resp.content_length().is_some_and(|n| n > cap as u64) {
        return Err(over());
    }
    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(|e| e.to_string())? {
        if body.len() + chunk.len() > cap {
            return Err(over());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}
