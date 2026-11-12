//! `pull` transfers: files at once, a large file in byte ranges at once, one connection
//! budget; each file is still hashed whole at the end (cloud mode's intake hashes
//! the weight files itself, on their mappings: `fetch_all`). Written ranges are listed
//! in `partial/<sha256>.ranges`. Precedents: aria2 `--split`, hf_xet (16 ranges per file).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinSet;

use super::PullError;
use super::{declared_total, local, short_source, shown_url, store, Entry, FileSpec, Progress};
use super::{Source, Store};

/// How `pull` spreads its requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transfer {
    /// Files downloaded at once.
    pub files: usize,
    /// HTTP requests in flight at once, all files together.
    pub connections: usize,
    /// Bytes per range request. A file of at most this size is one request.
    pub chunk: u64,
    /// Further attempts of one range after a transport error, a 5xx, a 429 or
    /// a short body. A plain (unranged) request is never retried.
    pub retries: u32,
    /// Wait before attempt `n` is `n` times this.
    pub retry_delay: Duration,
}

impl Default for Transfer {
    fn default() -> Self {
        Self {
            files: 4,
            connections: 16,
            chunk: 64 << 20,
            retries: 3,
            retry_delay: Duration::from_millis(500),
        }
    }
}

/// What every download task of one pull shares.
struct Shared {
    client: reqwest::Client,
    name: String,
    t: Transfer,
    conns: Arc<Semaphore>,
    /// Set on the first failure: no new request starts, those in flight end.
    stop: AtomicBool,
}

impl Shared {
    fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    /// A connection slot, or `Cancelled` once the pull is failing.
    async fn permit(&self) -> Result<OwnedSemaphorePermit, PullError> {
        let p = self
            .conns
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| cancelled())?;
        if self.stopped() {
            return Err(cancelled());
        }
        Ok(p)
    }
}

const CANCELLED: &str = "cancelled after another file failed";

/// Ends the tasks still queued after the first error; the first real error is reported.
fn cancelled() -> PullError {
    PullError::Network(CANCELLED.into())
}

fn is_cancel(e: Option<&PullError>) -> bool {
    matches!(e, Some(PullError::Network(m)) if m == CANCELLED)
}

fn declared_mismatch(name: &str, path: &str, size: u64, total: u64) -> PullError {
    let rel = if total > size { "more" } else { "fewer" };
    PullError::Integrity(format!(
        "{name} {path}: the source declares {total} bytes, {rel} than the catalog's {size}; \
         nothing was kept"
    ))
}

/// The whole file is not the one the catalog pins; its bytes are deleted by the caller.
pub(super) fn mismatch(name: &str, spec: &FileSpec, got: &str) -> PullError {
    PullError::Integrity(format!(
        "{name} {}: sha256 {got} does not match the catalog's {}; the download was deleted. \
         Either the source served other bytes or the catalog is wrong; nothing was installed",
        spec.path, spec.sha256
    ))
}

enum Event {
    At(usize, u64),
    Note(String),
}

/// Bring every file of `entry` to `partial/<sha256>.part`, verified. With
/// `defer_weights` the weight files are brought whole (size, ranges) and NOT
/// hashed: the caller hashes each on its mapping before it installs anything
/// (`verify_staged`). Returns the bytes transferred.
pub(super) async fn fetch_all(
    store: &Store,
    source: &Source,
    entry: &Entry,
    defer_weights: bool,
    progress: &mut dyn Progress,
) -> Result<u64, PullError> {
    let t = source.transfer;
    let shared = Arc::new(Shared {
        client: source.client.clone(),
        name: entry.name.clone(),
        t,
        conns: Arc::new(Semaphore::new(t.connections.max(1))),
        stop: AtomicBool::new(false),
    });
    let (tx, mut rx) = mpsc::unbounded_channel();
    let mut queue = entry.files.iter().enumerate();
    let mut jobs = JoinSet::new();
    let mut total = 0u64;
    let mut first: Option<PullError> = None;
    loop {
        while jobs.len() < t.files.max(1) && !shared.stopped() {
            let Some((index, f)) = queue.next() else {
                break;
            };
            let job = FileJob {
                shared: shared.clone(),
                index,
                url: source.url(entry, &f.path),
                spec: f.clone(),
                part: store.partial(&f.sha256),
                ranges: store.partial_ranges(&f.sha256),
                defer_hash: defer_weights && entry.weight_files().any(|w| w.path == f.path),
                tx: tx.clone(),
                shown: AtomicU64::new(u64::MAX),
            };
            jobs.spawn(job.run());
        }
        if jobs.is_empty() {
            break;
        }
        tokio::select! {
            Some(ev) = rx.recv() => show(progress, &entry.files, ev),
            Some(joined) = jobs.join_next() => {
                let r = joined.unwrap_or_else(|e| Err(PullError::Local(format!("download task: {e}"))));
                match r {
                    Ok(n) => total += n,
                    Err(e) => {
                        shared.stop.store(true, Ordering::SeqCst);
                        if first.is_none() || is_cancel(first.as_ref()) {
                            first = Some(e);
                        }
                    }
                }
            }
        }
    }
    while let Ok(ev) = rx.try_recv() {
        show(progress, &entry.files, ev);
    }
    first.map_or(Ok(total), Err)
}

/// One file, by its URL, to `partial/<sha256>.part`, verified: the job
/// `fetch_all` runs per catalog file, alone. Returns the bytes transferred.
pub(super) async fn fetch_one(
    store: &Store,
    source: &Source,
    name: &str,
    url: &str,
    spec: &FileSpec,
    progress: &mut dyn Progress,
) -> Result<u64, PullError> {
    let t = source.transfer;
    let shared = Arc::new(Shared {
        client: source.client.clone(),
        name: name.to_string(),
        t,
        conns: Arc::new(Semaphore::new(t.connections.max(1))),
        stop: AtomicBool::new(false),
    });
    let (tx, mut rx) = mpsc::unbounded_channel();
    let job = FileJob {
        shared,
        index: 0,
        url: url.to_string(),
        spec: spec.clone(),
        part: store.partial(&spec.sha256),
        ranges: store.partial_ranges(&spec.sha256),
        defer_hash: false,
        tx,
        shown: AtomicU64::new(u64::MAX),
    };
    let mut run = std::pin::pin!(job.run());
    let files = std::slice::from_ref(spec);
    let result = loop {
        tokio::select! {
            Some(ev) = rx.recv() => show(progress, files, ev),
            r = &mut run => break r,
        }
    };
    while let Ok(ev) = rx.try_recv() {
        show(progress, files, ev);
    }
    result
}

fn show(progress: &mut dyn Progress, files: &[FileSpec], ev: Event) {
    match ev {
        Event::At(i, done) => progress.file(&files[i].path, done, files[i].size),
        Event::Note(m) => progress.note(&m),
    }
}

struct FileJob {
    shared: Arc<Shared>,
    index: usize,
    url: String,
    spec: FileSpec,
    part: PathBuf,
    ranges: PathBuf,
    /// Leave the hash of the whole to the caller (`fetch_all`, `defer_weights`).
    defer_hash: bool,
    tx: mpsc::UnboundedSender<Event>,
    /// Last percent sent: one event per percent, not per body chunk.
    shown: AtomicU64,
}

fn remove_if_any(p: &Path) -> Result<(), PullError> {
    match std::fs::remove_file(p) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(local(p.display())(e)),
        _ => Ok(()),
    }
}

impl FileJob {
    fn at(&self, done: u64) {
        let pct = (done * 100).checked_div(self.spec.size).unwrap_or(100);
        if self.shown.swap(pct, Ordering::Relaxed) != pct {
            let _ = self.tx.send(Event::At(self.index, done));
        }
    }

    fn note(&self, msg: String) {
        let _ = self.tx.send(Event::Note(msg));
    }

    fn integrity(&self, what: String) -> PullError {
        PullError::Integrity(format!("{} {}: {what}", self.shared.name, self.spec.path))
    }

    /// Bring `part` to the full, verified file. Returns the bytes transferred.
    async fn run(self) -> Result<u64, PullError> {
        let size = self.spec.size;
        let ranged = cfg!(unix) && size > self.shared.t.chunk;
        let mut have = std::fs::symlink_metadata(&self.part)
            .map(|m| m.len())
            .unwrap_or(0);
        let has_ranges = std::fs::symlink_metadata(&self.ranges).is_ok();
        // A ranged partial read as one stream would trust its holes.
        if have > size || (has_ranges && !ranged) {
            remove_if_any(&self.part)?;
            remove_if_any(&self.ranges)?;
            have = 0;
        }
        let transferred = if ranged {
            match self.ranged(have, has_ranges).await {
                Err(e @ PullError::Integrity(_)) => {
                    let _ = std::fs::remove_file(&self.part);
                    let _ = std::fs::remove_file(&self.ranges);
                    return Err(e);
                }
                r => r?,
            }
        } else if have < size {
            self.single(have).await?
        } else {
            0
        };
        if self.defer_hash {
            // Whole, unverified: a later run finds it complete and hashes it too.
            remove_if_any(&self.ranges)?;
            return Ok(transferred);
        }
        let part = self.part.clone();
        let got = tokio::task::spawn_blocking(move || store::sha256_file(&part))
            .await
            .map_err(|e| PullError::Local(format!("hash task: {e}")))?
            .map_err(local(self.part.display()))?;
        if got != self.spec.sha256 {
            let _ = std::fs::remove_file(&self.part);
            let _ = std::fs::remove_file(&self.ranges);
            return Err(mismatch(&self.shared.name, &self.spec, &got));
        }
        remove_if_any(&self.ranges)?;
        Ok(transferred)
    }

    /// One request for the file, resuming at `have` when the source honours it.
    async fn single(&self, have: u64) -> Result<u64, PullError> {
        let _slot = self.shared.permit().await?;
        let url = &self.url;
        let mut req = self.shared.client.get(url);
        let shown = shown_url(url);
        if have > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let resp = req
            .send()
            .await
            .map_err(|e| PullError::Network(format!("{shown}: {}", send_failure(e))))?;
        let status = resp.status();
        let start = match status.as_u16() {
            206 if have > 0 => {
                let cr = resp
                    .headers()
                    .get(reqwest::header::CONTENT_RANGE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("");
                if !cr.starts_with(&format!("bytes {have}-")) {
                    return Err(PullError::Network(format!(
                        "{shown}: asked to resume at {have}, server answered Content-Range {cr:?}"
                    )));
                }
                have
            }
            200 => 0, // the server ignored the range: start over
            // Nothing at or past `have`: the source's file is shorter than the catalog's.
            416 if have > 0 => {
                let _ = std::fs::remove_file(&self.part);
                return Err(self.short(have));
            }
            _ => return Err(http_error(url, status)),
        };
        self.stream(resp, start).await
    }

    fn short(&self, got: u64) -> PullError {
        short_source(&self.shared.name, &self.spec.path, self.spec.size, got)
    }

    /// Append the body of `resp` to `part` from `start` (0 truncates).
    async fn stream(&self, mut resp: reqwest::Response, mut start: u64) -> Result<u64, PullError> {
        // The catalog URL in messages: the redirect target may carry a signature.
        let (url, size, part) = (shown_url(&self.url), self.spec.size, &self.part);
        if let Some(total) = declared_total(&resp, resp.status().as_u16()) {
            if total != size {
                let _ = std::fs::remove_file(part);
                return Err(declared_mismatch(
                    &self.shared.name,
                    &self.spec.path,
                    size,
                    total,
                ));
            }
        }
        let mut out = open_part(part, start == 0, start > 0)?;
        let mut received = 0u64;
        self.at(start);
        loop {
            let chunk = resp.chunk().await.map_err(|e| {
                PullError::Network(format!(
                    "{url}: {} (run pull again to resume)",
                    e.without_url()
                ))
            })?;
            let Some(chunk) = chunk else { break };
            if start + chunk.len() as u64 > size {
                drop(out);
                let _ = std::fs::remove_file(part);
                return Err(self.integrity(format!(
                    "the server sent more than the catalog's {size} bytes; deleted"
                )));
            }
            std::io::Write::write_all(&mut out, &chunk).map_err(local(part.display()))?;
            start += chunk.len() as u64;
            received += chunk.len() as u64;
            self.at(start);
        }
        out.sync_all().map_err(local(part.display()))?;
        if start < size {
            // A clean end of body: a cut connection is an error from `chunk()`.
            drop(out);
            let _ = std::fs::remove_file(part);
            return Err(self.short(start));
        }
        Ok(received)
    }

    /// The missing ranges of `part`, several at once. A partial without a
    /// ranges file is a prefix of `have` bytes; an unreadable list keeps nothing.
    async fn ranged(&self, have: u64, has_ranges: bool) -> Result<u64, PullError> {
        let s = &self.shared;
        let size = self.spec.size;
        let mut done = if has_ranges {
            // Bytes listed past the partial's end are not on disk (cut short, or gone):
            // trusted, they read as zeros and the end hash deleted the whole file.
            Ranges::load(&self.ranges, size)
                .unwrap_or_else(|| Ranges::prefix(size, 0))
                .within(have)
        } else {
            Ranges::prefix(size, have)
        };
        let todo = done.missing(s.t.chunk);
        let Some(&(s0, e0)) = todo.first() else {
            return Ok(0);
        };
        // Saved before any write: a partial that has a ranges file is never read as a prefix.
        done.save(&self.ranges)
            .map_err(local(self.ranges.display()))?;
        let file = Arc::new(open_part(&self.part, false, false)?);
        let urls = Arc::new(Urls {
            resolve: self.url.clone(),
            direct: Mutex::new(None),
        });
        self.at(done.bytes());
        // The first missing range alone: a 200 says the source ignores ranges.
        let first = Piece {
            shared: s.clone(),
            urls: urls.clone(),
            file: file.clone(),
            path: self.spec.path.clone(),
            size,
            start: s0,
            end: e0,
        };
        if let Some((resp, _slot)) = first.fetch(true).await? {
            drop(file);
            remove_if_any(&self.ranges)?;
            self.note(format!(
                "{}: the source ignores byte ranges, one stream from byte 0",
                self.spec.path
            ));
            return self.stream(resp, 0).await;
        }
        done.add(s0, e0);
        done.save(&self.ranges)
            .map_err(local(self.ranges.display()))?;
        self.at(done.bytes());
        let mut got = e0 - s0;
        let mut pieces = JoinSet::new();
        for &(start, end) in &todo[1..] {
            let p = Piece {
                shared: s.clone(),
                urls: urls.clone(),
                file: file.clone(),
                path: self.spec.path.clone(),
                size,
                start,
                end,
            };
            pieces.spawn(async move { p.fetch(false).await.map(|_| (start, end)) });
        }
        let mut failed: Option<PullError> = None;
        while let Some(joined) = pieces.join_next().await {
            let r = joined.unwrap_or_else(|e| Err(PullError::Local(format!("range task: {e}"))));
            let saved = r.and_then(|(start, end)| {
                done.add(start, end);
                done.save(&self.ranges)
                    .map_err(local(self.ranges.display()))?;
                Ok(end - start)
            });
            match saved {
                Ok(n) => {
                    got += n;
                    self.at(done.bytes());
                }
                Err(e) => {
                    s.stop.store(true, Ordering::SeqCst);
                    if failed.is_none() || is_cancel(failed.as_ref()) {
                        failed = Some(e);
                    }
                }
            }
        }
        if let Some(e) = failed {
            return Err(e);
        }
        file.sync_all().map_err(local(self.part.display()))?;
        Ok(got)
    }
}

fn http_error(url: &str, status: reqwest::StatusCode) -> PullError {
    PullError::Network(format!(
        "{}: HTTP {status}{}",
        shown_url(url),
        if status.as_u16() == 401 || status.as_u16() == 404 {
            " (gated, removed, or wrong revision in the catalog)"
        } else {
            ""
        }
    ))
}

/// `part` for writing, never through a link, created 0640.
fn open_part(part: &Path, truncate: bool, append: bool) -> Result<std::fs::File, PullError> {
    let mut open = std::fs::OpenOptions::new();
    open.create(true)
        .write(true)
        .truncate(truncate)
        .append(append);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        open.mode(store::FILE_MODE)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC);
    }
    let out = open.open(part).map_err(local(part.display()))?;
    if !out.metadata().map_err(local(part.display()))?.is_file() {
        return Err(PullError::Local(format!(
            "{}: not a regular file",
            part.display()
        )));
    }
    Ok(out)
}

#[cfg(unix)]
fn write_at(f: &std::fs::File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(f, buf, offset)
}

#[cfg(not(unix))]
fn write_at(_: &std::fs::File, _: &[u8], _: u64) -> std::io::Result<()> {
    Err(std::io::Error::other("ranged writes need unix"))
}

/// A request that failed, without its URL. reqwest shows a redirect policy's
/// refusal only through `source()`: "error following redirect" says no why.
fn send_failure(e: reqwest::Error) -> String {
    let e = e.without_url();
    match std::error::Error::source(&e).filter(|_| e.is_redirect()) {
        Some(why) => format!("{e}: {why}"),
        None => e.to_string(),
    }
}

/// The catalog URL, and where it redirected to (the Hub sends LFS files to a
/// CDN): ranges go there directly, one resolve per file and not per range.
struct Urls {
    resolve: String,
    direct: Mutex<Option<String>>,
}

impl Urls {
    fn direct(&self) -> Option<String> {
        self.direct.lock().map(|d| d.clone()).unwrap_or(None)
    }
    fn set_direct(&self, url: String) {
        if let Ok(mut d) = self.direct.lock() {
            *d = Some(url);
        }
    }
}

/// Bytes `start..end` of one file, written in place.
struct Piece {
    shared: Arc<Shared>,
    urls: Arc<Urls>,
    file: Arc<std::fs::File>,
    path: String,
    size: u64,
    start: u64,
    end: u64,
}

/// A failed attempt: `Retry` may be asked again, `Fatal` ends the pull.
enum Fail {
    Retry(String),
    Fatal(PullError),
}

impl Piece {
    /// The range, with retries. `whole_ok`: a 200 is handed back with its slot
    /// (the probe); otherwise a 200 is a failed attempt.
    async fn fetch(
        &self,
        whole_ok: bool,
    ) -> Result<Option<(reqwest::Response, OwnedSemaphorePermit)>, PullError> {
        let s = &self.shared;
        let mut last = String::new();
        for attempt in 0..=s.t.retries {
            if attempt > 0 {
                tokio::time::sleep(s.t.retry_delay * attempt).await;
            }
            let slot = s.permit().await?;
            // A retry resolves again: a signed CDN URL may have expired.
            let direct = if attempt == 0 {
                self.urls.direct()
            } else {
                None
            };
            let url = direct.clone().unwrap_or_else(|| self.urls.resolve.clone());
            match self.attempt(&url, direct.is_some()).await {
                Ok(Got::Written(final_url)) => {
                    if direct.is_none() {
                        self.urls.set_direct(final_url);
                    }
                    return Ok(None);
                }
                Ok(Got::Whole(resp)) if whole_ok => return Ok(Some((resp, slot))),
                Ok(Got::Whole(_)) => {
                    last = "the source answered a range with the whole file".into()
                }
                Err(Fail::Retry(why)) => last = why,
                Err(Fail::Fatal(e)) => {
                    s.stop.store(true, Ordering::SeqCst);
                    return Err(e);
                }
            }
        }
        s.stop.store(true, Ordering::SeqCst);
        Err(PullError::Network(format!(
            "{} {}: bytes {}-{}: {last}; {} attempts (run pull again to resume)",
            s.name,
            self.path,
            self.start,
            self.end - 1,
            s.t.retries + 1
        )))
    }

    async fn attempt(&self, url: &str, direct: bool) -> Result<Got, Fail> {
        let (start, end, size) = (self.start, self.end, self.size);
        let resp = self
            .shared
            .client
            .get(url)
            .header(reqwest::header::RANGE, format!("bytes={start}-{}", end - 1))
            .send()
            .await
            .map_err(|e| Fail::Retry(send_failure(e)))?;
        let status = resp.status();
        match status.as_u16() {
            206 => {}
            200 => return Ok(Got::Whole(resp)),
            416 => {
                return Err(Fail::Fatal(short_source(
                    &self.shared.name,
                    &self.path,
                    size,
                    start,
                )))
            }
            429 | 500..=599 => return Err(Fail::Retry(format!("HTTP {status}"))),
            // An expired or refused CDN URL: the retry goes through the catalog URL.
            _ if direct => {
                return Err(Fail::Retry(format!(
                    "HTTP {status} from the redirect target"
                )))
            }
            _ => return Err(Fail::Fatal(http_error(&self.urls.resolve, status))),
        }
        let cr = resp
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let (a, b, total) =
            parse_content_range(&cr).ok_or_else(|| Fail::Retry(format!("Content-Range {cr:?}")))?;
        if total != size {
            return Err(Fail::Fatal(declared_mismatch(
                &self.shared.name,
                &self.path,
                size,
                total,
            )));
        }
        if a != start || b + 1 != end {
            return Err(Fail::Retry(format!(
                "asked {start}-{}, server answered Content-Range {cr:?}",
                end - 1
            )));
        }
        let final_url = resp.url().to_string();
        let mut resp = resp;
        let mut at = start;
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| Fail::Retry(format!("{}", e.without_url())))?
        {
            if at + chunk.len() as u64 > end {
                return Err(Fail::Retry("the server sent more than the range".into()));
            }
            write_at(&self.file, &chunk, at)
                .map_err(|e| Fail::Fatal(PullError::Local(format!("{}: {e}", self.path))))?;
            at += chunk.len() as u64;
        }
        if at < end {
            return Err(Fail::Retry(format!(
                "the range ended after {} of {} bytes",
                at - start,
                end - start
            )));
        }
        Ok(Got::Written(final_url))
    }
}

enum Got {
    /// 206 for exactly the range asked, written; the URL that served it.
    Written(String),
    /// 200: the source sent the whole file.
    Whole(reqwest::Response),
}

/// `bytes a-b/N`.
fn parse_content_range(v: &str) -> Option<(u64, u64, u64)> {
    let (range, total) = v.strip_prefix("bytes ")?.split_once('/')?;
    let (a, b) = range.split_once('-')?;
    let (a, b, n) = (a.parse().ok()?, b.parse().ok()?, total.parse().ok()?);
    (a <= b).then_some((a, b, n))
}

/// Byte ranges of a partial file already written, half-open, sorted, merged.
/// The end hash still decides: this list only saves asking again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Ranges {
    size: u64,
    done: Vec<(u64, u64)>,
}

const RANGES_MAGIC: &str = "x0ne-ranges/1";

impl Ranges {
    /// The first `have` bytes, as a partial without a ranges file holds them.
    pub(super) fn prefix(size: u64, have: u64) -> Self {
        let have = have.min(size);
        Self {
            size,
            done: if have > 0 {
                vec![(0, have)]
            } else {
                Vec::new()
            },
        }
    }

    pub(super) fn add(&mut self, start: u64, end: u64) {
        if start >= end || end > self.size {
            return;
        }
        self.done.push((start, end));
        self.done.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(self.done.len());
        for &(a, b) in &self.done {
            match merged.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(b),
                _ => merged.push((a, b)),
            }
        }
        self.done = merged;
    }

    /// The ranges cut at `len`: what a partial of `len` bytes can hold.
    pub(super) fn within(mut self, len: u64) -> Self {
        self.done = self
            .done
            .into_iter()
            .filter(|&(a, _)| a < len)
            .map(|(a, b)| (a, b.min(len)))
            .collect();
        self
    }

    pub(super) fn bytes(&self) -> u64 {
        self.done.iter().map(|(a, b)| b - a).sum()
    }

    /// The gaps, cut in pieces of at most `chunk` bytes.
    pub(super) fn missing(&self, chunk: u64) -> Vec<(u64, u64)> {
        let chunk = chunk.max(1);
        let mut out = Vec::new();
        let mut at = 0;
        let ends = self.done.iter().copied().chain([(self.size, self.size)]);
        for (a, b) in ends {
            let mut s = at;
            while s < a {
                let e = (s + chunk).min(a);
                out.push((s, e));
                s = e;
            }
            at = at.max(b);
        }
        out
    }

    pub(super) fn encode(&self) -> String {
        let mut s = format!("{RANGES_MAGIC} {}\n", self.size);
        for (a, b) in &self.done {
            s += &format!("{a} {b}\n");
        }
        s
    }

    /// Strict: another size, a range out of bounds or any stray line is `None`.
    pub(super) fn parse(text: &str, size: u64) -> Option<Self> {
        let mut lines = text.lines();
        let head = lines.next()?;
        if head != format!("{RANGES_MAGIC} {size}") {
            return None;
        }
        let mut r = Self::prefix(size, 0);
        for l in lines {
            let (a, b) = l.split_once(' ')?;
            let (a, b): (u64, u64) = (a.parse().ok()?, b.parse().ok()?);
            if a >= b || b > size {
                return None;
            }
            r.add(a, b);
        }
        Some(r)
    }

    fn load(path: &Path, size: u64) -> Option<Self> {
        let bytes = crate::fsread::read_capped(path, crate::fsread::CONFIG_CAP).ok()?;
        Self::parse(std::str::from_utf8(&bytes).ok()?, size)
    }

    /// Written to a temporary file, then renamed over the old list.
    fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        let mut open = std::fs::OpenOptions::new();
        open.create(true).write(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            open.mode(store::FILE_MODE)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        }
        let mut f = open.open(&tmp)?;
        std::io::Write::write_all(&mut f, self.encode().as_bytes())?;
        drop(f);
        std::fs::rename(&tmp, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_ranges_are_the_gaps_cut_by_the_chunk() {
        let mut r = Ranges::prefix(1000, 0);
        assert_eq!(r.missing(400), vec![(0, 400), (400, 800), (800, 1000)]);
        r.add(400, 800);
        r.add(0, 100);
        assert_eq!(r.missing(250), vec![(100, 350), (350, 400), (800, 1000)]);
        assert_eq!(r.bytes(), 500);
        r.add(100, 400);
        r.add(800, 1000);
        assert_eq!(r.done, vec![(0, 1000)]);
        assert!(r.missing(1).is_empty());
    }

    #[test]
    fn a_prefix_is_one_range_capped_at_the_size() {
        assert_eq!(Ranges::prefix(10, 4).done, vec![(0, 4)]);
        assert_eq!(Ranges::prefix(10, 40).done, vec![(0, 10)]);
        assert!(Ranges::prefix(10, 0).done.is_empty());
    }

    #[test]
    fn ranges_are_cut_at_the_partial_length() {
        let mut r = Ranges::prefix(1000, 0);
        r.add(0, 300);
        r.add(500, 900);
        assert_eq!(r.clone().within(600).done, vec![(0, 300), (500, 600)]);
        assert_eq!(r.clone().within(500).done, vec![(0, 300)]);
        assert_eq!(r.clone().within(200).done, vec![(0, 200)]);
        assert!(r.clone().within(0).done.is_empty());
        assert_eq!(r.clone().within(900), r);
        assert_eq!(r.clone().within(5000), r);
    }

    #[test]
    fn the_ranges_file_round_trips_and_refuses_anything_else() {
        let mut r = Ranges::prefix(1000, 0);
        r.add(0, 100);
        r.add(500, 700);
        assert_eq!(Ranges::parse(&r.encode(), 1000), Some(r.clone()));
        for bad in [
            "x0ne-ranges/1 999\n0 100\n",
            "x0ne-ranges/2 1000\n",
            "x0ne-ranges/1 1000\n0 1001\n",
            "x0ne-ranges/1 1000\n5 5\n",
            "x0ne-ranges/1 1000\n0 -1\n",
            "x0ne-ranges/1 1000\nzero 10\n",
            "",
        ] {
            assert_eq!(Ranges::parse(bad, 1000), None, "{bad:?}");
        }
    }

    #[test]
    fn content_range_is_parsed_strictly() {
        assert_eq!(parse_content_range("bytes 0-99/1000"), Some((0, 99, 1000)));
        assert_eq!(parse_content_range("bytes 9-0/1000"), None);
        assert_eq!(parse_content_range("bytes 0-99/*"), None);
        assert_eq!(parse_content_range("0-99/1000"), None);
    }
}
