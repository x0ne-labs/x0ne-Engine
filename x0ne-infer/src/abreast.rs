//! Futures run abreast and read in a fixed order: the collateral of one
//! verdict is fetched together, and each answer is taken where the serial
//! code took it, so the first refusal is the one a serial run gives.
//!
//! Nothing here spawns a task: the futures run inside the caller's, and
//! dropping the set drops the ones not read yet, in flight or not.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::Poll;

type Runner<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

/// Futures started together. [`Abreast::read`] waits for one of them and
/// keeps the others moving meanwhile.
#[derive(Default)]
pub struct Abreast<'a> {
    running: Vec<Runner<'a>>,
}

/// The answer of one future given to [`Abreast::start`], to be read once.
pub struct Slot<T>(Arc<Mutex<Option<T>>>);

impl<'a> Abreast<'a> {
    pub fn new() -> Self {
        Self {
            running: Vec::new(),
        }
    }

    /// Add `fut`. It first runs at the next [`Abreast::read`], of any slot.
    pub fn start<T, F>(&mut self, fut: F) -> Slot<T>
    where
        T: Send + 'a,
        F: Future<Output = T> + Send + 'a,
    {
        let cell = Arc::new(Mutex::new(None));
        let filled = cell.clone();
        self.running.push(Box::pin(async move {
            let out = fut.await;
            *filled.lock().unwrap_or_else(|e| e.into_inner()) = Some(out);
        }));
        Slot(cell)
    }

    /// The answer of `slot`, as soon as it is there: the futures that are
    /// not done all run until then, and stay where they are afterwards.
    pub async fn read<T>(&mut self, slot: Slot<T>) -> T {
        std::future::poll_fn(|cx| {
            self.running
                .retain_mut(|f| f.as_mut().poll(cx).is_pending());
            match slot.0.lock().unwrap_or_else(|e| e.into_inner()).take() {
                Some(out) => Poll::Ready(out),
                None => Poll::Pending,
            }
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use std::time::Duration;

    /// Three waits of 40 ms end together, and are read in the order asked,
    /// whatever the order they end in.
    #[tokio::test(start_paused = true)]
    async fn futures_run_together_and_are_read_in_the_order_asked() {
        let started = tokio::time::Instant::now();
        let mut all = Abreast::new();
        let wait = |ms: u64, name: &'static str| async move {
            tokio::time::sleep(Duration::from_millis(ms)).await;
            name
        };
        let slow = all.start(wait(40, "slow"));
        let fast = all.start(wait(10, "fast"));
        let mid = all.start(wait(40, "mid"));
        assert_eq!(all.read(slow).await, "slow");
        assert_eq!(all.read(fast).await, "fast");
        assert_eq!(all.read(mid).await, "mid");
        assert_eq!(started.elapsed(), Duration::from_millis(40));
    }

    /// Reading one answer does not wait for a later one, and dropping the
    /// set drops what was not read: it never ends.
    #[tokio::test(start_paused = true)]
    async fn an_answer_is_read_without_waiting_for_the_others() {
        let ended = Arc::new(AtomicUsize::new(0));
        let started = tokio::time::Instant::now();
        let mut all = Abreast::new();
        let first = all.start(async { 1 });
        let e = ended.clone();
        let _late = all.start(async move {
            tokio::time::sleep(Duration::from_secs(60)).await;
            e.fetch_add(1, SeqCst);
        });
        assert_eq!(all.read(first).await, 1);
        assert_eq!(started.elapsed(), Duration::ZERO);
        drop(all);
        tokio::time::sleep(Duration::from_secs(120)).await;
        assert_eq!(ended.load(SeqCst), 0);
    }

    /// Nothing runs before the first read.
    #[tokio::test]
    async fn nothing_starts_before_the_first_read() {
        let ran = Arc::new(AtomicUsize::new(0));
        let mut all = Abreast::new();
        let r = ran.clone();
        let slot = all.start(async move { r.fetch_add(1, SeqCst) });
        tokio::task::yield_now().await;
        assert_eq!(ran.load(SeqCst), 0);
        all.read(slot).await;
        assert_eq!(ran.load(SeqCst), 1);
    }
}
