//! The harness of the `hostile_bytes` suites: every hand-written parser of
//! bytes a third party controls is fed mutated copies of real fixtures, as
//! ordinary `cargo test`, offline, with no dependency.
//!
//! Deterministic: a campaign is seeded by a constant, so a finding names a
//! case anyone can replay. Two variables change what runs, neither read by
//! shipped code (this file is compiled into test binaries only):
//!
//! - `X0NE_FUZZ_ITERS=<n>` multiplies the number of mutated inputs of every
//!   campaign (default 1, the budget of `make ci`; 100 is a deep run);
//! - `X0NE_FUZZ_SEED=<n>` is mixed into every campaign's seed, to explore
//!   other inputs than the ones `make ci` replays.
//!
//! What a campaign holds every case to:
//!
//! - no panic (`catch_unwind`; debug builds check arithmetic overflow, so a
//!   wrapped length is a panic here);
//! - no hang: a case that runs longer than [`HANG`] aborts the test binary
//!   and names itself, one longer than [`SLOW`] is a finding;
//! - no allocation larger than the campaign's bound for the input's size
//!   (a counting allocator, per thread): a length field does not size memory;
//! - what the suite states about its verdict (a truncation is refused, a
//!   changed signed byte is refused, a round trip gives the same value).
//!
//! A campaign does not stop at its first finding: it lists one case per kind
//! of finding when it ends, so one run shows every way a parser fails.
//!
//! Shared by `#[path]` between the test targets of x0ne-sdk, x0ne-ohttp,
//! x0ne-engine-cloud and x0ne-infer (`hostile_bytes_tests.rs`). Coverage-guided
//! fuzzing on the same entry points: `fuzz/README.md`.

#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A case still running after this long is a hang: the binary aborts.
pub const HANG: Duration = Duration::from_secs(120);
/// A case that took this long is a finding. Far above any honest parse, so a
/// loaded CI host does not trip it.
pub const SLOW: Duration = Duration::from_secs(20);

// ── Allocation accounting ───────────────────────────────────────────────────

thread_local! {
    /// The largest single request of this thread since the last reset.
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}

/// The system allocator, noting the largest request of each thread: what a
/// parser asked for in one go while it read a case.
pub struct Counting;

fn note(size: usize) {
    // A thread being torn down has no cell left: nothing to note.
    let _ = LARGEST.try_with(|l| {
        if size > l.get() {
            l.set(size);
        }
    });
}

// The four methods forward to `System` under their own contracts.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn reset_largest() {
    let _ = LARGEST.try_with(|l| l.set(0));
}

fn largest() -> usize {
    LARGEST.try_with(Cell::get).unwrap_or(0)
}

/// The default bound on one allocation while a case of `input` bytes is
/// read: 64 times the input, and never less than 1 MiB (a parser's own
/// tables, a certificate, a hash state).
pub fn default_alloc_bound(input: usize) -> usize {
    (64 * input).max(1 << 20)
}

// ── Randomness ──────────────────────────────────────────────────────────────

/// SplitMix64 (Steele, Lea, Flood 2014): 64 bits of state, every seed good.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in `0..n`; `n` of 0 gives 0.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    pub fn byte(&mut self) -> u8 {
        self.next_u64() as u8
    }

    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.byte()).collect()
    }

    pub fn pick<'a, T>(&mut self, of: &'a [T]) -> &'a T {
        &of[self.below(of.len())]
    }
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse().ok()
}

/// `X0NE_FUZZ_ITERS`, at least 1.
pub fn depth() -> usize {
    env_u64("X0NE_FUZZ_ITERS").unwrap_or(1).max(1) as usize
}

/// `base` mutated inputs under `make ci`, `base × X0NE_FUZZ_ITERS` offline.
pub fn cases(base: usize) -> usize {
    base.saturating_mul(depth())
}

// ── Mutators of bytes ───────────────────────────────────────────────────────

/// The bytes a parser misreads first: zero, the largest positive and the
/// smallest negative signed byte, all ones.
pub const EDGE_BYTES: [u8; 4] = [0x00, 0x7f, 0x80, 0xff];

/// Text of more than one byte per character, a line and a paragraph
/// separator, a byte order mark, a NUL: what a parser that cuts a textual
/// field by byte offsets splits in the middle of.
pub const AWKWARD_TEXT: [&str; 10] = [
    "\u{e9}",
    "\u{4e2d}\u{6587}",
    "\u{1f600}",
    "\u{2014} ",
    "\u{feff}",
    "\u{2028}",
    "\u{0}",
    "\u{202e}",
    "e\u{301}",
    "\u{10ffff}",
];

/// Numbers at the edges of what the fields hold.
pub const HUGE_NUMBERS: [&str; 12] = [
    "0",
    "-1",
    "255",
    "256",
    "65536",
    "4294967295",
    "4294967296",
    "9223372036854775807",
    "18446744073709551615",
    "18446744073709551616",
    "1e308",
    "99999999999999999999999999999999999999999999999999",
];

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn span(rng: &mut Rng, len: usize, max: usize) -> (usize, usize) {
    if len == 0 {
        return (0, 0);
    }
    let at = rng.below(len);
    let n = 1 + rng.below(max.min(len - at));
    (at, n)
}

/// One mutation of `v`, in place, and what it was, precisely enough to redo
/// by hand. `donor` is another input of the same format, for splices.
pub fn mutate_once(rng: &mut Rng, v: &mut Vec<u8>, donor: &[u8]) -> String {
    let len = v.len();
    match rng.below(14) {
        0 if len > 0 => {
            let (at, bit) = (rng.below(len), rng.below(8));
            v[at] ^= 1 << bit;
            format!("flip bit {bit} of byte {at}")
        }
        1 if len > 0 => {
            let (at, b) = (rng.below(len), *rng.pick(&EDGE_BYTES));
            v[at] = b;
            format!("set byte {at} to {b:#04x}")
        }
        2 if len > 0 => {
            let to = rng.below(len);
            v.truncate(to);
            format!("truncate to {to} bytes")
        }
        3 => {
            let n = 1 + rng.below(64);
            let tail = match rng.below(3) {
                0 => vec![0u8; n],
                1 => vec![0xff; n],
                _ => rng.bytes(n),
            };
            v.extend_from_slice(&tail);
            format!("append {}", hex(&tail))
        }
        4 if len > 0 => {
            // A length field: 1, 2, 4 or 8 bytes, either endianness, set to
            // 0, all ones, one more, one less, or the bytes that are left.
            let width = *rng.pick(&[1usize, 2, 2, 4, 4, 8]);
            let at = rng.below(len);
            let width = width.min(len - at);
            let little = rng.below(2) == 0;
            let mut cur = 0u64;
            for k in 0..width {
                let b = u64::from(v[at + if little { width - 1 - k } else { k }]);
                cur = (cur << 8) | b;
            }
            let ones = if width == 8 {
                u64::MAX
            } else {
                (1u64 << (8 * width)) - 1
            };
            let new = match rng.below(6) {
                0 => 0,
                1 => ones,
                2 => cur.wrapping_add(1) & ones,
                3 => cur.wrapping_sub(1) & ones,
                4 => (len - at) as u64 & ones,
                _ => (ones >> 1) + 1,
            };
            for k in 0..width {
                let shift = 8 * if little { k } else { width - 1 - k };
                v[at + k] = (new >> shift) as u8;
            }
            format!(
                "set the {width}-byte {} integer at {at} from {cur} to {new}",
                if little { "LE" } else { "BE" }
            )
        }
        5 if len > 0 => {
            let (at, n) = span(rng, len, 256);
            let chunk = v[at..at + n].to_vec();
            let to = rng.below(len + 1);
            v.splice(to..to, chunk);
            format!("copy bytes {at}..{} to {to}", at + n)
        }
        6 if len > 1 => {
            let (a, n) = span(rng, len, 64);
            let b = rng.below(len - n + 1);
            for k in 0..n {
                v.swap(a + k, b + k);
            }
            format!("swap the {n} bytes at {a} and at {b}")
        }
        7 if !donor.is_empty() => {
            let (from, n) = span(rng, donor.len(), 512);
            let at = rng.below(len + 1);
            let end = (at + if rng.below(2) == 0 { n } else { 0 }).min(len);
            v.splice(at..end, donor[from..from + n].iter().copied());
            format!(
                "replace bytes {at}..{end} by the donor's {from}..{}",
                from + n
            )
        }
        8 if len > 0 => {
            let (at, n) = span(rng, len, 256);
            v.drain(at..at + n);
            format!("delete bytes {at}..{}", at + n)
        }
        9 if len > 0 => {
            let (at, n) = span(rng, len, 16);
            let new = rng.bytes(n);
            v[at..at + n].copy_from_slice(&new);
            format!("overwrite bytes at {at} with {}", hex(&new))
        }
        10 => {
            let s = *rng.pick(&AWKWARD_TEXT);
            let at = rng.below(len + 1);
            v.splice(at..at, s.bytes());
            format!("insert {} at {at}", hex(s.as_bytes()))
        }
        11 if len > 0 => {
            let at = rng.below(len);
            let d = if rng.below(2) == 0 { 1u8 } else { 0xff };
            v[at] = v[at].wrapping_add(d);
            format!("add {d:#04x} to byte {at}")
        }
        12 if !donor.is_empty() && len > 0 => {
            // The head of one, the tail of the other.
            let cut = rng.below(len);
            let from = rng.below(donor.len());
            v.truncate(cut);
            v.extend_from_slice(&donor[from..]);
            format!("keep {cut} bytes, then the donor from {from}")
        }
        _ => {
            let (at, b) = (rng.below(len + 1), rng.byte());
            v.insert(at, b);
            format!("insert byte {b:#04x} at {at}")
        }
    }
}

/// `seed` after one to three mutations, and what they were.
pub fn mutant(rng: &mut Rng, seed: &[u8], donor: &[u8]) -> (Vec<u8>, String) {
    let mut v = seed.to_vec();
    let n = 1 + rng.below(3);
    let what: Vec<String> = (0..n).map(|_| mutate_once(rng, &mut v, donor)).collect();
    (v, what.join("; "))
}

// ── Mutators of text ────────────────────────────────────────────────────────

fn boundary(s: &str, mut at: usize) -> usize {
    at = at.min(s.len());
    while !s.is_char_boundary(at) {
        at -= 1;
    }
    at
}

/// A run of ASCII digits of `s` starting at or after a random place.
fn digit_run(rng: &mut Rng, s: &str) -> Option<(usize, usize)> {
    let b = s.as_bytes();
    let from = rng.below(b.len().max(1));
    let start = (from..b.len())
        .chain(0..from)
        .find(|&i| b[i].is_ascii_digit() && (i == 0 || !b[i - 1].is_ascii_digit()))?;
    let end = (start..b.len())
        .find(|&i| !b[i].is_ascii_digit())
        .unwrap_or(b.len());
    Some((start, end))
}

/// One mutation of `s` that leaves it valid UTF-8, and what it was.
pub fn mutate_text_once(rng: &mut Rng, s: &mut String, donor: &str) -> String {
    let len = s.len();
    match rng.below(13) {
        0 => {
            let at = boundary(s, rng.below(len + 1));
            let t = *rng.pick(&AWKWARD_TEXT);
            s.insert_str(at, t);
            format!("insert {t:?} at {at}")
        }
        1 => match digit_run(rng, s) {
            Some((a, b)) => {
                let n = *rng.pick(&HUGE_NUMBERS);
                let old = s[a..b].to_string();
                s.replace_range(a..b, n);
                format!("replace the number {old} at {a} by {n}")
            }
            None => {
                s.push('0');
                "append 0".into()
            }
        },
        2 if len > 0 => {
            let to = boundary(s, rng.below(len));
            s.truncate(to);
            format!("truncate to {to} bytes")
        }
        3 if len > 0 => {
            let a = boundary(s, rng.below(len));
            let b = boundary(s, (a + 1 + rng.below(64)).min(len));
            s.replace_range(a..b, "");
            format!("delete bytes {a}..{b}")
        }
        4 if len > 0 => {
            // An ASCII character replaced by its neighbour, or its other case.
            let at = (0..len)
                .map(|k| (rng.below(len) + k) % len)
                .find(|&i| s.as_bytes()[i].is_ascii_graphic());
            match at {
                Some(at) => {
                    let old = s.as_bytes()[at];
                    let new = match rng.below(3) {
                        0 if old.is_ascii_alphabetic() => old ^ 0x20,
                        1 if old < b'~' => old + 1,
                        _ if old > b'!' => old - 1,
                        _ => b'~',
                    };
                    s.replace_range(at..at + 1, &char::from(new).to_string());
                    format!(
                        "replace {:?} at {at} by {:?}",
                        char::from(old),
                        char::from(new)
                    )
                }
                None => {
                    s.push('x');
                    "append x".into()
                }
            }
        }
        5 => {
            let lines: Vec<&str> = s.split_inclusive('\n').collect();
            if lines.is_empty() {
                s.push('\n');
                return "append a newline".into();
            }
            let k = rng.below(lines.len());
            let at: usize = lines[..k].iter().map(|l| l.len()).sum();
            let line = lines[k].to_string();
            s.insert_str(at, &line);
            format!("repeat line {k}")
        }
        6 => {
            let mut lines: Vec<String> = s.split_inclusive('\n').map(str::to_string).collect();
            if lines.len() < 2 {
                s.insert(0, '\n');
                return "prepend a newline".into();
            }
            let (a, b) = (rng.below(lines.len()), rng.below(lines.len()));
            lines.swap(a, b);
            *s = lines.concat();
            format!("swap lines {a} and {b}")
        }
        7 => {
            let mut lines: Vec<&str> = s.split_inclusive('\n').collect();
            if lines.is_empty() {
                return "nothing to delete".into();
            }
            let k = rng.below(lines.len());
            lines.remove(k);
            *s = lines.concat();
            format!("delete line {k}")
        }
        8 if !donor.is_empty() => {
            let a = boundary(donor, rng.below(donor.len()));
            let b = boundary(donor, (a + 1 + rng.below(256)).min(donor.len()));
            let at = boundary(s, rng.below(len + 1));
            s.insert_str(at, &donor[a..b]);
            format!("insert the donor's {a}..{b} at {at}")
        }
        9 => {
            let depth = *rng.pick(&[1usize, 8, 64, 127, 128, 129, 1000, 20_000]);
            let open = *rng.pick(&["[", "{\"a\":", "[{\"a\":[", "("]);
            let at = boundary(s, rng.below(len + 1));
            s.insert_str(at, &open.repeat(depth));
            format!("insert {open:?} {depth} times at {at}")
        }
        10 => {
            let t = *rng.pick(&[
                "\n", "\r\n", "\n\n", " ", "\t", "\"", "\\", "\\u0000", "\\ud800", ",", ":", "}",
                "]", "{", "[", "=", "==", "+", "-", ".",
            ]);
            let at = boundary(s, rng.below(len + 1));
            s.insert_str(at, t);
            format!("insert {t:?} at {at}")
        }
        11 if len > 0 => {
            let a = boundary(s, rng.below(len));
            let b = boundary(s, (a + 1 + rng.below(32)).min(len));
            let piece = s[a..b].to_string();
            let n = *rng.pick(&[2usize, 3, 100, 5000]);
            s.insert_str(a, &piece.repeat(n));
            format!("repeat bytes {a}..{b} {n} times")
        }
        _ => {
            let c = *rng.pick(&['\u{1}', '\u{7f}', '\r', '\u{b}', '\u{85}', '\u{a0}']);
            let at = boundary(s, rng.below(len + 1));
            s.insert(at, c);
            format!("insert {c:?} at {at}")
        }
    }
}

/// `seed` after one to three text mutations, and what they were.
pub fn text_mutant(rng: &mut Rng, seed: &str, donor: &str) -> (String, String) {
    let mut s = seed.to_string();
    let n = 1 + rng.below(3);
    let what: Vec<String> = (0..n)
        .map(|_| mutate_text_once(rng, &mut s, donor))
        .collect();
    (s, what.join("; "))
}

// ── Exhaustive small spaces ─────────────────────────────────────────────────

/// Every strict prefix of `seed`, shortest first.
pub fn prefixes(seed: &[u8]) -> impl Iterator<Item = &[u8]> + '_ {
    (0..seed.len()).map(move |n| &seed[..n])
}

/// The prefix lengths of an input of `len` bytes to try when one case is
/// costly (a signature is checked before the bytes are read): every `step`th
/// and the last `tail` under `make ci`, every one from `X0NE_FUZZ_ITERS=10`.
pub fn lengths(len: usize, step: usize, tail: usize) -> Vec<usize> {
    let all = depth() >= 10;
    (0..len)
        .filter(|n| all || n % step.max(1) == 0 || n + tail >= len)
        .collect()
}

/// `seed` with bit `bit` of byte `at` flipped.
pub fn flipped(seed: &[u8], at: usize, bit: usize) -> Vec<u8> {
    let mut v = seed.to_vec();
    v[at] ^= 1 << bit;
    v
}

/// Every single-bit flip of `seed[range]`: `(byte, bit, mutant)`.
pub fn bit_flips(
    seed: &[u8],
    range: std::ops::Range<usize>,
) -> impl Iterator<Item = (usize, usize, Vec<u8>)> + '_ {
    range.flat_map(move |at| (0..8).map(move |bit| (at, bit, flipped(seed, at, bit))))
}

// ── Campaigns ───────────────────────────────────────────────────────────────

struct Watch {
    /// The case being read: when it started, and what it is.
    current: Mutex<Option<(Instant, String)>>,
    done: AtomicBool,
}

/// One format under attack: runs cases, keeps one example of each kind of
/// finding, fails at the end if there is any.
pub struct Campaign {
    name: &'static str,
    pub rng: Rng,
    /// The largest single allocation a case of `n` input bytes may make.
    pub alloc_bound: fn(usize) -> usize,
    ran: u64,
    /// Mutated inputs a verifier accepted with the verdict of the intact
    /// input: bytes no signature covers. Printed, not a failure.
    same_verdict: u64,
    /// Kind of finding → (first case, how many).
    findings: BTreeMap<String, (String, u64)>,
    watch: Arc<Watch>,
    started: Instant,
}

impl Campaign {
    pub fn new(name: &'static str, seed: u64) -> Self {
        let watch = Arc::new(Watch {
            current: Mutex::new(None),
            done: AtomicBool::new(false),
        });
        let w = Arc::clone(&watch);
        // The watchdog: a parser that loops never returns to be judged, so
        // the judgement is here. It ends the whole binary, with the case.
        std::thread::spawn(move || {
            while !w.done.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(200));
                let stuck = w.current.lock().ok().and_then(|c| {
                    c.as_ref()
                        .filter(|(t, _)| t.elapsed() > HANG)
                        .map(|(_, what)| what.clone())
                });
                if let Some(what) = stuck {
                    eprintln!(
                        "hostile bytes, {name}: no answer after {} s on: {what}",
                        HANG.as_secs()
                    );
                    std::process::abort();
                }
            }
        });
        Self {
            name,
            rng: Rng::new(seed ^ env_u64("X0NE_FUZZ_SEED").unwrap_or(0)),
            alloc_bound: default_alloc_bound,
            ran: 0,
            same_verdict: 0,
            findings: BTreeMap::new(),
            watch,
            started: Instant::now(),
        }
    }

    /// `seed` mutated by this campaign's generator.
    pub fn mutant(&mut self, seed: &[u8], donor: &[u8]) -> (Vec<u8>, String) {
        mutant(&mut self.rng, seed, donor)
    }

    pub fn text_mutant(&mut self, seed: &str, donor: &str) -> (String, String) {
        text_mutant(&mut self.rng, seed, donor)
    }

    /// A finding of kind `kind` on the case `what`. The first case of each
    /// kind is kept.
    pub fn finding(&mut self, kind: impl Into<String>, what: &str) {
        let e = self
            .findings
            .entry(kind.into())
            .or_insert_with(|| (what.to_string(), 0));
        e.1 += 1;
    }

    /// `ok`, or a finding of kind `kind`.
    pub fn hold(&mut self, ok: bool, kind: &str, what: &str) {
        if !ok {
            self.finding(kind, what);
        }
    }

    /// A mutated input was accepted with the intact input's verdict.
    pub fn accepted_unchanged(&mut self) {
        self.same_verdict += 1;
    }

    /// Runs `f` on the case `what` of `input` bytes: its value, or `None`
    /// and a finding if it panicked. Too slow or too large an allocation is
    /// a finding too.
    pub fn run<T>(&mut self, what: &str, input: usize, f: impl FnOnce() -> T) -> Option<T> {
        self.ran += 1;
        let started = Instant::now();
        if let Ok(mut c) = self.watch.current.lock() {
            *c = Some((started, what.to_string()));
        }
        reset_largest();
        let out = catch_unwind(AssertUnwindSafe(f));
        let peak = largest();
        if let Ok(mut c) = self.watch.current.lock() {
            *c = None;
        }
        let took = started.elapsed();
        if took > SLOW {
            self.finding(format!("slow: more than {} s", SLOW.as_secs()), what);
        }
        let bound = (self.alloc_bound)(input);
        if peak > bound {
            self.finding(
                format!("one allocation above the bound of {bound} bytes for its input"),
                &format!("{what} ({input} bytes in, {peak} bytes asked)"),
            );
        }
        match out {
            Ok(v) => Some(v),
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "a panic that is not a string".into());
                // The kind is the message without its numbers: one line per
                // place that panics, not one per input.
                let kind: String = msg
                    .chars()
                    .map(|c| if c.is_ascii_digit() { '#' } else { c })
                    .collect();
                self.finding(format!("panic: {kind}"), &format!("{what} → {msg}"));
                None
            }
        }
    }

    /// Ends the campaign: fails with every kind of finding and its first case.
    pub fn finish(self) {
        self.watch.done.store(true, Ordering::Relaxed);
        eprintln!(
            "hostile bytes, {}: {} cases in {:.1} s, {} mutated inputs accepted with the intact verdict, {} kinds of finding",
            self.name,
            self.ran,
            self.started.elapsed().as_secs_f64(),
            self.same_verdict,
            self.findings.len()
        );
        if self.findings.is_empty() {
            return;
        }
        let mut report = format!(
            "hostile bytes, {}: {} kinds of finding in {} cases\n",
            self.name,
            self.findings.len(),
            self.ran
        );
        for (kind, (first, count)) in &self.findings {
            let first: String = first.chars().take(1500).collect();
            report.push_str(&format!("- {kind} ({count} cases); first: {first}\n"));
        }
        panic!("{report}");
    }
}

/// Lowercase hex of `b`, for a case small enough to print whole.
pub fn show(b: &[u8]) -> String {
    if b.len() <= 96 {
        hex(b)
    } else {
        format!("{}… ({} bytes)", hex(&b[..96]), b.len())
    }
}
