//! An answer's length, as the host of the VM can read it, rounded up to one
//! of few values: the bytes of a whole answer, the time it leaves, and the
//! number of frames of a stream.
//!
//! The rounding is Padme (Nikitin, Barman, Lueks, Underwood, Hubaux, Ford,
//! "Reducing Metadata Leakage from Encrypted Files and Communication with
//! PURBs", PETS 2019, arXiv:1806.03160): a length keeps only its top
//! `⌊log2 ⌊log2 L⌋⌋ + 1` bits, the rest is rounded up. The paper bounds what
//! the padded length leaks to O(log log M) bits for lengths up to M, at an
//! overhead "of at most 12%". The tests here measure that bound and the number
//! of sizes rather than recall it.
//!
//! A floor sits under each: below it every answer is one size. Above it,
//! answers of nearby lengths share a size; answers of very different lengths
//! do not, and that much is still read. A cap, [`crate::padding::MAX_TICKS`],
//! keeps a padded time within the 15 minutes the relay waits for a chat.
//!
//! No log macro in this file: it handles the answer's bytes.

use std::time::Duration;

/// Frames of a stream, the last included, and 100 ms ticks before a whole
/// answer leaves: never fewer. One second at [`crate::stream::TICK`].
pub const MIN_TICKS: u64 = 10;

/// Frames of a stream and ticks of a whole answer: never more once padded.
/// The largest size within the 9000 ticks of the relay's
/// [`crate::remote_verify::CHAT_TIMEOUT`]; the next is 9216, past it. The
/// 29.6 s left are for the queue, the prefill and the network. A stream
/// whose text is not out by then ends in error there; a whole answer that
/// ran longer leaves when done, its time not rounded.
pub const MAX_TICKS: u64 = 8_704;

/// Bytes of a whole answer's body: never fewer. The `x0ne` block of an H100
/// answer was 7322 bytes (2026-10-17): this holds it and about 8 KiB of text.
pub const MIN_WHOLE_BYTES: u64 = 16 * 1024;

/// `len` rounded up to the next Padme size: its low `E - S` bits cleared,
/// where `E = ⌊log2 len⌋` and `S = ⌊log2 E⌋ + 1`. 0 and 1 are their own
/// size; a length past the largest size of its range saturates.
pub fn padme(len: u64) -> u64 {
    if len < 2 {
        return len;
    }
    let e = u64::from(len.ilog2());
    let s = u64::from(e.ilog2()) + 1;
    let mask = (1u64 << (e - s)) - 1;
    len.checked_add(mask).map_or(u64::MAX, |l| l & !mask)
}

/// Whether `n` is a size an answer may end at: a Padme size, `floor` or more.
pub fn is_size(n: u64, floor: u64) -> bool {
    n >= floor && padme(n) == n
}

/// The smallest size an answer of `n` may end at, given `floor`.
pub fn size_for(n: u64, floor: u64) -> u64 {
    padme(n.max(floor))
}

/// A whole answer's JSON body, followed by spaces up to its size. JSON
/// allows whitespace after a value: the parsed answer is the same.
pub fn pad_json(mut body: Vec<u8>) -> Vec<u8> {
    let len = u64::try_from(body.len()).unwrap_or(u64::MAX);
    let size = usize::try_from(size_for(len, MIN_WHOLE_BYTES)).unwrap_or(body.len());
    body.resize(size.max(body.len()), b' ');
    body
}

/// The period of the time padding, as the code that pads holds it. Outside
/// this crate's own unit tests it has no field and no constructor but
/// [`Tick::PRODUCTION`]: its period is [`crate::stream::TICK`], which no
/// flag, variable or configuration changes. The unit tests of this crate
/// that assert nothing about time serve at `Tick::SHORT`, so that a table
/// of refusals does not wait one real second a row;
/// `tests/cloud_source_sweep.rs` holds every other value under `cfg(test)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tick {
    #[cfg(test)]
    period: Duration,
}

impl Tick {
    /// [`crate::stream::TICK`]: what every server built from this crate pads at.
    pub const PRODUCTION: Self = Self {
        #[cfg(test)]
        period: crate::stream::TICK,
    };

    /// A twentieth of [`crate::stream::TICK`], for unit tests that assert no
    /// time, no number of frames and no size of ticks.
    #[cfg(test)]
    pub(crate) const SHORT: Self = Self {
        period: Duration::from_millis(5),
    };

    /// One tick.
    #[cfg(not(test))]
    #[must_use]
    pub const fn period(self) -> Duration {
        crate::stream::TICK
    }

    /// One tick.
    #[cfg(test)]
    #[must_use]
    pub const fn period(self) -> Duration {
        self.period
    }

    /// [`MIN_TICKS`] ticks: when a refusal leaves, after its request.
    #[must_use]
    pub fn floor(self) -> Duration {
        self.period()
            .saturating_mul(u32::try_from(MIN_TICKS).unwrap_or(u32::MAX))
    }
}

impl Default for Tick {
    fn default() -> Self {
        Self::PRODUCTION
    }
}

/// How long after its request a whole answer that took `elapsed` leaves:
/// `elapsed` rounded up to whole ticks, then to a size of at least
/// [`MIN_TICKS`]; past [`MAX_TICKS`], the whole ticks only, so that the
/// relay still waits for it.
pub fn release_after(elapsed: Duration, tick: Duration) -> Duration {
    let tick_ns = tick.as_nanos().max(1);
    let ticks = u64::try_from(elapsed.as_nanos().div_ceil(tick_ns)).unwrap_or(u64::MAX);
    let at = if ticks > MAX_TICKS {
        ticks
    } else {
        size_for(ticks, MIN_TICKS)
    };
    tick.saturating_mul(u32::try_from(at).unwrap_or(u32::MAX))
}

/// The model's turns: one answer at a time, from its start until a size of
/// ticks after it. An answer queued behind another starts when that turn
/// ends, so its start tells the host the other's time to within a size, not
/// to the token, as it would if it started when the model freed up.
pub struct Turns {
    one: std::sync::Arc<tokio::sync::Semaphore>,
    tick: Tick,
}

impl Default for Turns {
    fn default() -> Self {
        Self {
            one: std::sync::Arc::new(tokio::sync::Semaphore::new(1)),
            tick: Tick::PRODUCTION,
        }
    }
}

impl Turns {
    /// Turns that end on sizes of `tick`.
    #[cfg(test)]
    pub(crate) fn every(tick: Tick) -> Self {
        Self {
            tick,
            ..Self::default()
        }
    }

    /// The tick its turns end on sizes of.
    pub fn tick(&self) -> Tick {
        self.tick
    }

    /// The next turn, once the one before has ended.
    pub async fn take(&self) -> Turn {
        let permit = self.one.clone().acquire_owned().await.ok();
        Turn {
            permit,
            held: Vec::new(),
            since: tokio::time::Instant::now(),
            ends: None,
            tick: self.tick,
        }
    }
}

/// One answer's turn. Dropped, on any path, it ends at [`Turn::ends`]: the
/// next turn waits until then, an early return, an error or a panic as well.
pub struct Turn {
    permit: Option<tokio::sync::OwnedSemaphorePermit>,
    /// Given back with the turn, not before: the answer's queue place.
    held: Vec<tokio::sync::OwnedSemaphorePermit>,
    since: tokio::time::Instant,
    ends: Option<tokio::time::Instant>,
    tick: Tick,
}

impl Turn {
    /// Keeps `place` until the turn ends. A full queue answers 503 at once:
    /// a place given back when the model frees up would tell the host,
    /// by when the 503s stop, how long the answer ran.
    pub fn hold(&mut self, place: Option<tokio::sync::OwnedSemaphorePermit>) {
        self.held.extend(place);
    }

    /// When the turn ends: a size of [`crate::stream::TICK`]s after it
    /// began, [`MIN_TICKS`] at least, fixed the first time it is asked, and
    /// asked at the latest when the turn is dropped.
    pub fn ends(&mut self) -> tokio::time::Instant {
        let (since, tick) = (self.since, self.tick);
        *self
            .ends
            .get_or_insert_with(|| since + release_after(since.elapsed(), tick.period()))
    }

    /// Ends the turn at `at`, unless its end was fixed already.
    pub fn ends_at(&mut self, at: tokio::time::Instant) -> tokio::time::Instant {
        *self.ends.get_or_insert(at)
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        let at = self.ends();
        let held = (self.permit.take(), std::mem::take(&mut self.held));
        // Past its end, given back now: before a whole answer's response leaves.
        if (held.0.is_none() && held.1.is_empty()) || tokio::time::Instant::now() >= at {
            return;
        }
        // Without a runtime there is no next turn to hold back.
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                tokio::time::sleep_until(at).await;
                drop(held);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TICK: Duration = Duration::from_millis(100);

    /// Hand-computed sizes, one per rule: 0 and 1 kept, below 8 exact, then
    /// low bits cleared. A failure is a rounding that differs from Padme's.
    #[test]
    fn padme_keeps_the_top_bits_of_a_length() {
        for (len, size) in [
            (0, 0),
            (1, 1),
            (2, 2),
            (7, 7),
            (8, 8),
            (9, 10),
            (17, 18),
            (33, 36),
            (65, 72),
            (129, 144),
            (1000, 1024),
            (16_384, 16_384),
            (16_385, 17_408),
            (u64::MAX - 1, u64::MAX),
        ] {
            assert_eq!(padme(len), size, "{len}");
        }
    }

    /// Every length up to 2^20: the size is at least the length, rounding a
    /// size gives itself, a longer length never gets a shorter size, and the
    /// overhead stays under 12 % (most: 15 bytes on 129). A failure is a
    /// length cut short, two roundings of one length, or an overhead past
    /// the paper's bound.
    #[test]
    fn a_size_is_never_shorter_stable_monotone_and_at_most_12_percent_more() {
        let mut worst = (0.0f64, 0u64);
        let mut last = 0;
        for len in 1..=(1u64 << 20) {
            let size = padme(len);
            assert!(size >= len, "{len}: {size}");
            assert_eq!(padme(size), size, "{len}: {size} is not a size");
            assert!(size >= last, "{len}: {size} after {last}");
            last = size;
            let over = (size - len) as f64 / len as f64;
            if over > worst.0 {
                worst = (over, len);
            }
        }
        assert_eq!(worst.1, 129, "{worst:?}");
        assert!(worst.0 < 0.12, "{worst:?}");
    }

    /// What the padded length leaves to read: the number of sizes. Up to
    /// 2^20 there are 300 of them (about 8.2 bits) where there were 2^20 lengths
    /// (20 bits). Above the floors: frames and bytes. A failure is a change
    /// of rounding that this count makes visible.
    #[test]
    fn few_sizes_remain_to_be_read() {
        let sizes = |from: u64, to: u64, floor: u64| {
            (from..=to)
                .map(|n| size_for(n, floor))
                .collect::<std::collections::BTreeSet<_>>()
        };
        assert_eq!(sizes(1, 1 << 20, 0).len(), 300);
        // Streams of 15 minutes, the relay's CHAT_TIMEOUT, at one frame per tick.
        let frames = sizes(1, 9_000, MIN_TICKS);
        assert_eq!(frames.len(), 118);
        assert_eq!(frames.first(), Some(&MIN_TICKS));
        // Whole bodies up to 1 MiB.
        assert_eq!(sizes(1, 1 << 20, MIN_WHOLE_BYTES).len(), 161);
    }

    /// Attack on the floors: a floor that is not a size would end answers on
    /// lengths no other answer takes. A failure is a floor changed to such a
    /// value.
    #[test]
    fn the_floors_are_sizes() {
        assert!(is_size(MIN_TICKS, MIN_TICKS));
        assert!(is_size(MIN_WHOLE_BYTES, MIN_WHOLE_BYTES));
        assert!(!is_size(MIN_TICKS - 2, MIN_TICKS), "below the floor");
        assert!(!is_size(MIN_TICKS + 1, MIN_TICKS), "11 is not a size");
    }

    /// Bodies of different lengths under the floor leave at one size, and
    /// parse to what they were. One byte past the floor takes the next size,
    /// never the floor: nothing is cut. An answer of escapes (six bytes a
    /// character) is sized after its serialization. A failure is a body whose
    /// length still follows its text, or a padded body that no longer parses.
    #[test]
    fn whole_bodies_are_padded_to_a_size_and_parse_the_same() {
        let body = |content: &str| {
            serde_json::to_vec(&serde_json::json!({
                "choices": [{"message": {"content": content}}],
            }))
            .unwrap()
        };
        let short = body("yes");
        let long = body(&"no, because ".repeat(400));
        let escapes = body(&"\u{1}".repeat(2000));
        assert!(short.len() < 100 && long.len() > 4_000 && escapes.len() > 12_000);
        for raw in [&short, &long, &escapes] {
            let padded = pad_json(raw.clone());
            assert_eq!(padded.len() as u64, MIN_WHOLE_BYTES);
            assert!(padded.starts_with(raw));
            assert!(padded[raw.len()..].iter().all(|&b| b == b' '));
            let back: serde_json::Value = serde_json::from_slice(&padded).unwrap();
            assert_eq!(
                back,
                serde_json::from_slice::<serde_json::Value>(raw).unwrap()
            );
        }
        let over = vec![b'1'; MIN_WHOLE_BYTES as usize + 1];
        assert_eq!(pad_json(over).len(), 17_408);
        let at = vec![b'1'; MIN_WHOLE_BYTES as usize];
        assert_eq!(pad_json(at).len(), MIN_WHOLE_BYTES as usize);
    }

    /// A whole answer leaves on a tick that is a size, one second at least:
    /// an instant answer and one of 950 ms leave together, one of 1.01 s at
    /// 1.2 s. Attack: an answer that took exactly a size leaves then, not a
    /// tick later; one that took longer than any size does not leave early.
    #[test]
    fn a_whole_answer_leaves_on_a_size_of_ticks() {
        let at = |ms: u64| release_after(Duration::from_millis(ms), TICK);
        assert_eq!(at(0), Duration::from_secs(1));
        assert_eq!(at(950), Duration::from_secs(1));
        assert_eq!(at(1_000), Duration::from_secs(1));
        assert_eq!(at(1_001), Duration::from_millis(1_200));
        assert_eq!(at(3_250), Duration::from_millis(3_600));
        assert_eq!(at(12_900), Duration::from_millis(14_400));
        for ms in (0..60_000).step_by(37) {
            let left = at(ms);
            assert!(left >= Duration::from_millis(ms), "{ms}: {left:?}");
            let ticks = (left.as_millis() / TICK.as_millis()) as u64;
            assert!(is_size(ticks, MIN_TICKS), "{ms}: {ticks}");
        }
        assert!(release_after(Duration::MAX, TICK) >= TICK * 1_000_000);
    }

    /// Length padding: an answer whose generation ends within the relay's 15 minutes
    /// ([`crate::remote_verify::CHAT_TIMEOUT`]) leaves within them. Rounded
    /// to a size, one that took 871 s would leave at 921.6 s, after the relay
    /// gave up on it and lost the session. A failure is a release time the
    /// relay no longer waits for.
    #[test]
    fn no_answer_is_held_past_the_relays_fifteen_minutes() {
        let limit = crate::remote_verify::CHAT_TIMEOUT;
        for ms in (0..limit.as_millis() as u64).step_by(97) {
            let left = release_after(Duration::from_millis(ms), TICK);
            assert!(left <= limit, "{ms} ms leaves at {left:?}");
        }
    }

    /// Length padding: [`MAX_TICKS`] is the largest size within the relay's 15 minutes;
    /// past it a whole answer leaves at its own tick, never later. A failure
    /// is a cap that is not a size, or one the relay does not wait for.
    #[test]
    fn the_cap_is_the_last_size_within_the_relays_fifteen_minutes() {
        let within = (crate::remote_verify::CHAT_TIMEOUT.as_millis() / TICK.as_millis()) as u64;
        assert!(is_size(MAX_TICKS, MIN_TICKS));
        assert!(MAX_TICKS <= within && size_for(MAX_TICKS + 1, MIN_TICKS) > within);
        let at = |ms: u64| release_after(Duration::from_millis(ms), TICK);
        assert_eq!(at(870_400), Duration::from_millis(870_400));
        assert_eq!(at(860_000), Duration::from_millis(870_400));
        assert_eq!(at(870_401), Duration::from_millis(870_500));
        assert_eq!(at(899_950), Duration::from_mins(15));
    }

    /// Length padding: a place held by a turn comes back when the turn ends, not when
    /// the turn is dropped: 1.4 s after a turn of 1.25 s, a whole second
    /// after one dropped at once. A failure is a place that comes back when
    /// the model frees up.
    #[tokio::test(start_paused = true)]
    async fn a_place_held_by_a_turn_comes_back_when_the_turn_ends() {
        for (ran, back) in [(1_250, 1_400), (0, 1_000)] {
            let places = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
            let turns = Turns::default();
            let t0 = tokio::time::Instant::now();
            let mut turn = turns.take().await;
            turn.hold(places.clone().try_acquire_owned().ok());
            tokio::time::sleep(Duration::from_millis(ran)).await;
            drop(turn);
            assert_eq!(places.available_permits(), 0, "{ran} ms: back at the drop");
            let _again = places.acquire().await.unwrap();
            assert_eq!(t0.elapsed(), Duration::from_millis(back), "{ran} ms");
        }
    }

    /// Length padding.
    /// A turn that ran 1.25 s ends at 1.4 s, one that ran 1.35 s too: the
    /// next turn starts at 1.4 s after either, never when the model freed
    /// up. Attack: a turn dropped at once, as on an error or a panic, still
    /// holds the next one back to the floor, one second. A failure is a next
    /// start that tells how long the turn before it ran.
    #[tokio::test(start_paused = true)]
    async fn the_next_turn_starts_when_the_one_before_is_a_size() {
        let next_start_after = |ran: Duration| async move {
            let turns = std::sync::Arc::new(Turns::default());
            let t0 = tokio::time::Instant::now();
            let first = turns.take().await;
            tokio::time::sleep(ran).await;
            drop(first);
            let _second = turns.take().await;
            t0.elapsed()
        };
        let a = next_start_after(Duration::from_millis(1_250)).await;
        let b = next_start_after(Duration::from_millis(1_350)).await;
        assert_eq!(a, Duration::from_millis(1_400));
        assert_eq!(b, a, "the second turn's start tells the first's length");
        let dropped_at_once = next_start_after(Duration::ZERO).await;
        assert_eq!(dropped_at_once, TICK * MIN_TICKS as u32);
    }

    /// `ends` is fixed once asked: an answer that waits for it and then
    /// drops its turn does not push the next turn one size further. A failure
    /// is a turn whose end moves with the time it is dropped.
    #[tokio::test(start_paused = true)]
    async fn a_turn_ends_where_it_was_first_asked_to() {
        let turns = Turns::default();
        let t0 = tokio::time::Instant::now();
        let mut turn = turns.take().await;
        tokio::time::sleep(Duration::from_millis(1_250)).await;
        let at = turn.ends();
        assert_eq!(at - t0, Duration::from_millis(1_400));
        tokio::time::sleep_until(at + Duration::from_millis(3)).await;
        assert_eq!(turn.ends(), at);
        drop(turn);
        let _next = turns.take().await;
        assert_eq!(t0.elapsed(), Duration::from_millis(1_403));
    }
}
