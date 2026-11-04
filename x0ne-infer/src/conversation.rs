//! The one conversation whose keys and values a model keeps between its
//! turns (docs/session-kv-cache.md). Asked per request (`crate::context`),
//! held under the model's lock by a backend that keeps one (`candle_backend`,
//! the tests' CPU reference; x0ne Core keeps none and refuses it with
//! `backend::ConversationUnsupported`): keyed by route and by a
//! handle the TD drew, spent by the turn that presents it, evicted by any
//! other request and after `KEEP_FOR` without a turn. Eviction zeroes the
//! cache buffers; what one forward derives from them is not zeroed.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::context::{AskedConversation, Handle};

/// How long a conversation stays resident without a turn.
pub const KEEP_FOR: Duration = Duration::from_secs(300);

/// Where a turn came in. A handle issued on one route continues nothing on
/// the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// `/v1/chat/completions`: the relay's route, and a local server's.
    Chat,
    /// `/v2/e2e/chat`: sealed by the caller, opened in the TD.
    Sealed,
}

/// One turn of a conversation, as the server admitted it.
#[derive(Debug)]
pub struct Turn {
    pub route: Route,
    /// `None` for `"new"`.
    pub presented: Option<Handle>,
    /// Drawn by the TD: the handle of the next turn.
    pub issued: Handle,
    /// Keep nothing after this turn.
    pub end: bool,
    reused: OnceLock<Reused>,
}

/// What a turn ran on: its number in the conversation and the prompt ids
/// whose keys and values an earlier turn computed. Zeroed when dropped.
#[derive(Debug, PartialEq, Eq)]
pub struct Reused {
    pub turn: u32,
    pub ids: Vec<u32>,
}

impl Drop for Reused {
    fn drop(&mut self) {
        zeroize::Zeroize::zeroize(&mut self.ids);
    }
}

impl Turn {
    /// The turn `asked` names on `route`, `None` without a conversation.
    /// `draw` gives the next turn's handle: 32 bytes of the salt source,
    /// `/dev/urandom` in production, drawn only for a conversation.
    pub fn admit(
        route: Route,
        asked: &AskedConversation,
        draw: impl FnOnce() -> [u8; 32],
    ) -> Option<Self> {
        let (presented, end) = match asked {
            AskedConversation::None => return None,
            AskedConversation::New { end } => (None, *end),
            AskedConversation::Continue { handle, end } => (Some(handle.clone()), *end),
        };
        Some(Self {
            route,
            presented,
            issued: Handle::from_bytes(draw()),
            end,
            reused: OnceLock::new(),
        })
    }

    /// Set by the backend once the turn ran; a second call changes nothing.
    pub fn record(&self, reused: Reused) {
        let _ = self.reused.set(reused);
    }

    /// What the backend recorded; `None` before it ran, or when it failed.
    pub fn reused(&self) -> Option<&Reused> {
        self.reused.get()
    }
}

/// The conversation whose keys and values are in a model's cache.
pub struct Resident {
    route: Route,
    handle: Handle,
    /// The ids whose keys and values the cache holds, position by position.
    ids: Vec<u32>,
    turn: u32,
    last: Instant,
}

impl Drop for Resident {
    fn drop(&mut self) {
        zeroize::Zeroize::zeroize(&mut self.ids);
    }
}

impl Resident {
    /// Kept after `turn`, its `number`th, for the turn that presents its
    /// issued handle on its route. `ids`: what the cache holds.
    pub fn new(turn: &Turn, number: u32, ids: Vec<u32>) -> Self {
        Self {
            route: turn.route,
            handle: turn.issued.clone(),
            ids,
            turn: number,
            last: Instant::now(),
        }
    }

    /// When it is evicted if no turn comes.
    pub fn deadline(&self, keep_for: Duration) -> Instant {
        self.last + keep_for
    }

    /// `(tokens reused, number of the turn)` when `turn` continues this
    /// conversation on its route; `None` otherwise. Tokens: the common prefix
    /// of the ids kept and `prompt`, one short of the prompt, whose last
    /// position is computed for its logits.
    pub fn continued_by(&self, turn: &Turn, prompt: &[u32]) -> Option<(usize, u32)> {
        let presented = turn.presented.as_ref()?;
        if turn.route != self.route || *presented != self.handle {
            return None;
        }
        let reuse = common_prefix(&self.ids, prompt).min(prompt.len().saturating_sub(1));
        Some((reuse, self.turn.saturating_add(1)))
    }
}

/// The length of the common prefix of `a` and `b`, every position of the
/// shorter compared wherever the first difference falls.
pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    let mut same = true;
    let mut n = 0;
    for (x, y) in a.iter().zip(b) {
        same &= x == y;
        n += usize::from(same);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(route: Route, presented: Option<u8>, issued: u8) -> Turn {
        let asked = match presented {
            None => AskedConversation::New { end: false },
            Some(b) => AskedConversation::Continue {
                handle: Handle::from_bytes([b; 32]),
                end: false,
            },
        };
        Turn::admit(route, &asked, || [issued; 32]).unwrap()
    }

    /// A conversation continues on its route, under the handle its last turn
    /// issued, by the prefix it shares, less the prompt's last token. A
    /// failure: a turn continuing another conversation, or past its prompt.
    #[test]
    fn a_resident_conversation_is_continued_by_its_handle_on_its_route_only() {
        let first = turn(Route::Sealed, None, 1);
        let kept = Resident::new(&first, 1, vec![5, 6, 7, 8]);
        let next = turn(Route::Sealed, Some(1), 2);
        assert_eq!(kept.continued_by(&next, &[5, 6, 7, 8, 9]), Some((4, 2)));
        assert_eq!(kept.continued_by(&next, &[5, 6, 0, 8, 9]), Some((2, 2)));
        assert_eq!(kept.continued_by(&next, &[0, 6, 7, 8, 9]), Some((0, 2)));
        assert_eq!(kept.continued_by(&next, &[5, 6, 7]), Some((2, 2)));
        assert_eq!(
            kept.continued_by(&turn(Route::Chat, Some(1), 2), &[5, 6]),
            None
        );
        assert_eq!(
            kept.continued_by(&turn(Route::Sealed, Some(9), 2), &[5, 6]),
            None
        );
        assert_eq!(
            kept.continued_by(&turn(Route::Sealed, None, 2), &[5, 6]),
            None
        );
        assert!(Turn::admit(Route::Chat, &AskedConversation::None, || panic!("drawn")).is_none());
    }

    #[test]
    fn the_common_prefix_runs_over_the_shorter_length() {
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 3, 4]), 3);
        assert_eq!(common_prefix(&[1, 9, 3], &[1, 2, 3]), 1);
        assert_eq!(common_prefix(&[], &[1]), 0);
    }
}
