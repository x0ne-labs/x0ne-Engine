// A data-oblivious byte-pair-encoding tokenizer, and a page/cache-line
// side-channel simulator that measures the exposure of a data-DEPENDENT one.
//
// Why this exists. A confidential-computing guest still runs its tokenizer on
// the CPU, over secret input. A standard tokenizer resolves each piece through
// hash-map lookups whose memory-access sequence depends on the bytes looked up.
// A host that can observe the guest's accesses at page or cache-line
// granularity (controlled-channel page faults, physical-alias cache probes)
// learns which table slots were touched, and the public hash makes each slot a
// token. TDXRay (IEEE S&P 2026) reconstructs prompts from a single tokenization
// this way, at 91.5% (Llama 3.2) and 94.2% (Gemma 3) similarity; its remedy is a
// data-oblivious map. Spill the Beans (arXiv:2505.00817) is the de-tokenization
// analogue on the embedding table.
//
// This module gives two table backends behind one tokenizer:
//   * `HashTables`       — open addressing, the probe sequence depends on the
//                          key. Models the standard tokenizer, and is the
//                          subject measured by the simulator below.
//   * `ObliviousTables`  — a char lookup scans the single-char entries in slot
//                          order; a batch of pair lookups is answered by ONE
//                          pass over the merge table; every compare is a
//                          branch-free mask select. On top of it the merge
//                          stage runs a fixed schedule over fixed buffers
//                          (`Scratch`): the instructions executed and the
//                          addresses touched are a function of the piece's
//                          UTF-8 byte length alone, never of its content.
// Both return the same ids, so the oblivious tokenizer's token output is
// bit-for-bit the stock tokenizer's (tested against `tokenizers::…::bpe::BPE`).
//
// The simulator is a MEASUREMENT of our own tokenizer's leakage, on fixtures we
// build here. It models a side channel (slot -> byte offset -> page / cache
// line); it does not read any real hardware channel and reconstructs nothing
// outside these fixtures. The real binary's streams are measured separately,
// under valgrind (`scripts/obtok-trace.sh`, `examples/obtok_trace.rs`).

// The branch-free helpers are inline(always) on purpose (see `ct` below); clippy::inline_always
// judges speed only.
#![allow(clippy::inline_always)]

use std::collections::HashMap;

/// A merge rule's key: the two child token ids, left then right.
pub type Pair = (u32, u32);

/// Which lookup table an access landed in.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Table {
    Vocab,
    Merges,
}

/// One table slot touched during a lookup, in visit order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Access {
    pub table: Table,
    pub slot: usize,
}

/// The simulated memory layout of the two tables: each table is a contiguous
/// array of fixed-size entries, so slot `i` sits at `i * entry` bytes. A page is
/// 4 KiB and a cache line 64 B, as on x86-64. These are the knobs a host's
/// observation granularity reads through.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub vocab_entry: usize,
    pub merges_entry: usize,
    pub page: usize,
    pub line: usize,
}

impl Default for Layout {
    fn default() -> Self {
        // 64 B per vocab entry (one line each), 16 B per merge entry.
        Layout {
            vocab_entry: 64,
            merges_entry: 16,
            page: 4096,
            line: 64,
        }
    }
}

impl Layout {
    fn entry(&self, table: Table) -> usize {
        match table {
            Table::Vocab => self.vocab_entry,
            Table::Merges => self.merges_entry,
        }
    }
    /// The page index a slot of `table` falls on.
    pub fn page_of(&self, table: Table, slot: usize) -> usize {
        slot.saturating_mul(self.entry(table)) / self.page
    }
    /// The cache-line index a slot of `table` falls on.
    pub fn line_of(&self, table: Table, slot: usize) -> usize {
        slot.saturating_mul(self.entry(table)) / self.line
    }
}

/// The ordered sequence of slot accesses a tokenization performed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Trace {
    pub accesses: Vec<Access>,
    recording: bool,
}

impl Default for Trace {
    fn default() -> Self {
        Trace {
            accesses: Vec::new(),
            recording: true,
        }
    }
}

impl Trace {
    /// A trace that records nothing: for timing the lookups alone, and for
    /// the real binary, where a recording trace would allocate.
    pub fn off() -> Self {
        Trace {
            accesses: Vec::new(),
            recording: false,
        }
    }
    #[inline(always)]
    fn record(&mut self, table: Table, slot: usize) {
        if self.recording {
            self.accesses.push(Access { table, slot });
        }
    }
    /// The trace as the host would see it at page granularity: consecutive
    /// repeats on one page collapse, as a page-fault channel cannot count
    /// re-accesses within a resident page.
    pub fn page_view(&self, l: &Layout) -> Vec<(Table, usize)> {
        collapse(
            self.accesses
                .iter()
                .map(|a| (a.table, l.page_of(a.table, a.slot))),
        )
    }
    /// The trace at cache-line granularity.
    pub fn line_view(&self, l: &Layout) -> Vec<(Table, usize)> {
        collapse(
            self.accesses
                .iter()
                .map(|a| (a.table, l.line_of(a.table, a.slot))),
        )
    }
}

fn collapse<I: Iterator<Item = (Table, usize)>>(it: I) -> Vec<(Table, usize)> {
    let mut out: Vec<(Table, usize)> = Vec::new();
    for x in it {
        if out.last() != Some(&x) {
            out.push(x);
        }
    }
    out
}

/// The two tables a BPE tokenizer indexes, behind a lookup that records its
/// accesses. An implementation fixes the access pattern; the ids it returns are
/// the same whatever the pattern.
pub trait Tables {
    /// The id of a whole piece/char `key`, or `None` if absent.
    fn vocab_get(&self, key: &str, trace: &mut Trace) -> Option<u32>;
    /// The `(rank, new_id)` of a merge rule for `pair`, or `None`.
    fn merge_get(&self, pair: Pair, trace: &mut Trace) -> Option<(u32, u32)>;
    /// Resolve several pair queries at once, writing each answer in place.
    /// The default probes one query after the other; an oblivious backend
    /// answers the whole batch in a single pass over the merge table.
    fn merge_get_batch(&self, queries: &mut [(Pair, Option<(u32, u32)>)], trace: &mut Trace) {
        for q in queries.iter_mut() {
            q.1 = self.merge_get(q.0, trace);
        }
    }
    /// Number of vocab slots a single-char lookup touches: the whole vocab
    /// for a backend that cannot tell chars apart from longer entries.
    fn char_slots_len(&self) -> usize {
        self.vocab_slots_len()
    }
    /// The slot a vocab key occupies, for the simulator's reconstruction.
    fn vocab_slot(&self, key: &str) -> Option<usize>;
    /// Number of vocab slots (array length, including empties).
    fn vocab_slots_len(&self) -> usize;
    /// Number of merge slots.
    fn merge_slots_len(&self) -> usize;
    /// Whether this backend's access pattern is independent of the key.
    fn is_oblivious(&self) -> bool;
    /// The oblivious tables, when this backend is one: the tokenizer then runs
    /// the fixed-schedule merge stage on them.
    fn as_oblivious(&self) -> Option<&ObliviousTables> {
        None
    }
}

/// Open-addressing tables: a lookup probes from `hash(key) mod n` forward until
/// it meets the key or an empty slot, recording each probed slot. The probe
/// sequence is a function of the key, so the trace leaks it.
pub struct HashTables {
    vocab: Vec<Option<(String, u32)>>,
    merges: Vec<Option<(Pair, (u32, u32))>>,
    vocab_pos: HashMap<String, usize>,
}

/// Oblivious tables. A single-char lookup scans the char table (the vocab's
/// single-char entries, 256 for a byte-level vocabulary) in `0..c` order; a
/// batch of pair lookups is one pass over the merge table in `0..m` order,
/// comparing every rule against every query with mask selects and no
/// data-dependent branch. The trace is the same whatever the keys, so two
/// inputs of equal shape are indistinguishable.
pub struct ObliviousTables {
    /// `(code point, id)` of every single-char vocab entry, in vocab order.
    chars: Vec<(u32, u32)>,
    vocab: Vec<Option<(String, u32)>>,
    merges: Vec<(Pair, (u32, u32))>,
    vocab_pos: HashMap<String, usize>,
}

// Masks pass through `black_box`, so the compiler cannot turn their selects
// into branches. Best effort: valgrind on the release binary showed it held
// (2026-10-24, rust 1.98.1); another build must be measured again.

#[inline(always)]
fn ct(m: u32) -> u32 {
    std::hint::black_box(m)
}

/// All-ones when `a == b`, zero otherwise, without a branch.
#[inline(always)]
fn eq_mask(a: u32, b: u32) -> u32 {
    let d = a ^ b;
    // (d | -d) has its top bit set iff d != 0.
    ct((((d | d.wrapping_neg()) >> 31) ^ 1).wrapping_neg())
}

/// All-ones when `a < b` (unsigned), zero otherwise, without a branch.
#[inline(always)]
fn lt_mask(a: u32, b: u32) -> u32 {
    // The borrow of a 64-bit `a - b` is its top bit.
    ct(((u64::from(a).wrapping_sub(u64::from(b)) >> 63) as u32).wrapping_neg())
}

/// `a` where `m` is all-ones, `b` where it is zero.
#[inline(always)]
fn select(m: u32, a: u32, b: u32) -> u32 {
    b ^ ((a ^ b) & m)
}

/// FNV-1a over bytes; the public hash the simulator also knows.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn pair_bytes(p: Pair) -> [u8; 8] {
    let mut b = [0u8; 8];
    b[..4].copy_from_slice(&p.0.to_le_bytes());
    b[4..].copy_from_slice(&p.1.to_le_bytes());
    b
}

/// The id a BPE builds from `vocab` (string -> id) and `merges` (ordered child
/// string pairs); the rank of a merge is its index in the list.
pub struct Built {
    pub vocab: Vec<(String, u32)>,
    /// (left_id, right_id) -> (rank, new_id), one rule per pair.
    pub merges: Vec<(Pair, (u32, u32))>,
}

/// Build the id-level merge map from string merges, the way `tokenizers` does:
/// rank = position, new_id = id of the concatenation; for a pair named twice,
/// the last rule wins at its own rank, as in the crate's hash map.
pub fn build(vocab: Vec<(String, u32)>, merges: &[(String, String)]) -> Built {
    // u32::MAX is the oblivious tokenizer's "no slot" and sentinel value.
    assert!(
        vocab.iter().all(|(_, id)| *id < u32::MAX) && merges.len() < u32::MAX as usize,
        "token ids and merge ranks must stay below u32::MAX"
    );
    let idof: HashMap<&str, u32> = vocab.iter().map(|(s, i)| (s.as_str(), *i)).collect();
    let mut out = Vec::with_capacity(merges.len());
    for (rank, (a, b)) in merges.iter().enumerate() {
        let a_id = idof
            .get(a.as_str())
            .unwrap_or_else(|| panic!("merge child {a:?} not in vocab"));
        let b_id = idof
            .get(b.as_str())
            .unwrap_or_else(|| panic!("merge child {b:?} not in vocab"));
        let cat = format!("{a}{b}");
        let new_id = idof
            .get(cat.as_str())
            .unwrap_or_else(|| panic!("merge result {cat:?} not in vocab"));
        out.push(((*a_id, *b_id), (rank as u32, *new_id)));
    }
    let last: HashMap<Pair, usize> = out.iter().enumerate().map(|(i, (p, _))| (*p, i)).collect();
    let merges = out
        .into_iter()
        .enumerate()
        .filter(|(i, (p, _))| last[p] == *i)
        .map(|(_, rule)| rule)
        .collect();
    Built { vocab, merges }
}

fn next_pow2_at_least(n: usize) -> usize {
    let mut c = 8usize;
    while c < n {
        c <<= 1;
    }
    c
}

impl HashTables {
    pub fn new(b: &Built) -> Self {
        // Open addressing at load factor < 0.5 so probe runs stay short.
        let vcap = next_pow2_at_least(b.vocab.len() * 2);
        let mut vocab: Vec<Option<(String, u32)>> = vec![None; vcap];
        let mut vocab_pos = HashMap::with_capacity(b.vocab.len());
        for (s, id) in &b.vocab {
            let mut slot = (fnv1a(s.as_bytes()) as usize) & (vcap - 1);
            while vocab[slot].is_some() {
                slot = (slot + 1) & (vcap - 1);
            }
            vocab[slot] = Some((s.clone(), *id));
            vocab_pos.insert(s.clone(), slot);
        }
        let mcap = next_pow2_at_least(b.merges.len() * 2);
        let mut merges: Vec<Option<(Pair, (u32, u32))>> = vec![None; mcap];
        for (pair, val) in &b.merges {
            // A pair met again replaces its rule, as a map insert does.
            let mut slot = (fnv1a(&pair_bytes(*pair)) as usize) & (mcap - 1);
            while merges[slot].is_some_and(|(p, _)| p != *pair) {
                slot = (slot + 1) & (mcap - 1);
            }
            merges[slot] = Some((*pair, *val));
        }
        HashTables {
            vocab,
            merges,
            vocab_pos,
        }
    }

    /// Open-addressing capacity of the vocab table (its array length).
    pub fn vocab_cap(&self) -> usize {
        self.vocab.len()
    }

    /// The token stored at a vocab slot, if any — the simulator's slot->token map.
    pub fn vocab_entry_at(&self, slot: usize) -> Option<&str> {
        self.vocab
            .get(slot)
            .and_then(|o| o.as_ref())
            .map(|(s, _)| s.as_str())
    }
}

/// One answer of the oblivious merge pass: the rule's rank and new id, and an
/// all-ones `hit` when a rule matched (rank and id are 0 otherwise).
#[derive(Clone, Copy, Default, Debug)]
struct Hit {
    rank: u32,
    new_id: u32,
    hit: u32,
}

/// "No slot" in `Scratch::next`, and both halves of the sentinel pair that pads
/// a merge query when no merge fired: `build` keeps every id below it.
const NONE: u32 = u32::MAX;
const SENTINEL: Pair = (NONE, NONE);

/// Working buffers of the oblivious tokenizer, one slot per byte of the piece,
/// grown only for a longer piece. Reused across calls, they keep the allocator
/// out of the call: valgrind saw its addresses move between calls (2026-10-24).
#[derive(Default)]
pub struct Scratch {
    bytes: Vec<u8>,
    /// Token id of each live slot.
    tok: Vec<u32>,
    /// All-ones when the slot holds a live symbol.
    alive: Vec<u32>,
    /// The next live slot after this one, or `NONE`.
    next: Vec<u32>,
    /// The token of that next live slot.
    ntok: Vec<u32>,
    /// The rule of the pair (slot, next): rank, new id, all-ones when one exists.
    rank: Vec<u32>,
    new_id: Vec<u32>,
    valid: Vec<u32>,
    queries: Vec<Pair>,
    hits: Vec<Hit>,
    /// The ids of the last call, compacted to the front; `len` of them count.
    out: Vec<u32>,
    len: usize,
}

impl Scratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Buffers sized for pieces of up to `n` bytes.
    pub fn with_len(n: usize) -> Self {
        let mut s = Self::default();
        s.fit(n);
        s
    }

    fn fit(&mut self, n: usize) {
        let n = n.max(2);
        if self.tok.len() < n {
            self.bytes.resize(n + 3, 0);
            for v in [
                &mut self.tok,
                &mut self.alive,
                &mut self.next,
                &mut self.ntok,
                &mut self.rank,
                &mut self.new_id,
                &mut self.valid,
                &mut self.out,
            ] {
                v.resize(n, 0);
            }
            self.queries.resize(n, SENTINEL);
            self.hits.resize(n, Hit::default());
        }
    }

    /// The ids the last tokenization produced.
    pub fn ids(&self) -> &[u32] {
        &self.out[..self.len]
    }
}

impl ObliviousTables {
    pub fn new(b: &Built) -> Self {
        // Compact arrays: every slot holds an entry, scanned in order.
        let vocab: Vec<Option<(String, u32)>> =
            b.vocab.iter().map(|(s, i)| Some((s.clone(), *i))).collect();
        let chars: Vec<(u32, u32)> = b
            .vocab
            .iter()
            .filter_map(|(s, id)| {
                let mut it = s.chars();
                match (it.next(), it.next()) {
                    (Some(c), None) => Some((c as u32, *id)),
                    _ => None,
                }
            })
            .collect();
        let merges: Vec<(Pair, (u32, u32))> = b.merges.clone();
        let vocab_pos = b
            .vocab
            .iter()
            .enumerate()
            .map(|(i, (s, _))| (s.clone(), i))
            .collect();
        ObliviousTables {
            chars,
            vocab,
            merges,
            vocab_pos,
        }
    }

    /// One pass over the merge table answering every query through masks. A
    /// later rule for a pair replaces an earlier one, as in `tokenizers`; OR-ing
    /// the ranks named a rule that does not exist (ab at 2 and 4 gave 6).
    fn merge_pass(&self, queries: &[Pair], hits: &mut [Hit], trace: &mut Trace) {
        for h in hits.iter_mut() {
            *h = Hit::default();
        }
        for (slot, (p, (rank, new_id))) in self.merges.iter().enumerate() {
            trace.record(Table::Merges, slot);
            for (q, h) in queries.iter().zip(hits.iter_mut()) {
                let m = eq_mask(p.0, q.0) & eq_mask(p.1, q.1);
                h.rank = select(m, *rank, h.rank);
                h.new_id = select(m, *new_id, h.new_id);
                h.hit |= m;
            }
        }
    }

    /// The id of code point `cp` and an all-ones mask when it is in the char
    /// table: the whole table, in order, whatever `cp`.
    fn char_id(&self, cp: u32, trace: &mut Trace) -> (u32, u32) {
        let (mut id, mut hit) = (0u32, 0u32);
        for (slot, (k, v)) in self.chars.iter().enumerate() {
            trace.record(Table::Vocab, slot);
            let m = eq_mask(*k, cp);
            id = select(m, *v, id);
            hit |= m;
        }
        (id, hit)
    }

    /// Tokenize one piece into `s` and return its ids. Every loop bound and index
    /// is a function of the byte length `n`: `n - 1` masked merge rounds, one
    /// merge-table pass each (docs/oblivious-tokenizer.md, "The countermeasure").
    pub fn tokenize_into<'s>(
        &self,
        piece: &str,
        s: &'s mut Scratch,
        trace: &mut Trace,
    ) -> &'s [u32] {
        let n = piece.len();
        assert!(n < NONE as usize, "piece longer than u32::MAX bytes");
        s.fit(n);
        s.bytes[..n].copy_from_slice(piece.as_bytes());
        s.bytes[n..n + 3].fill(0);

        // Decode and look up every slot.
        for i in 0..n {
            let b0 = u32::from(s.bytes[i]);
            let b1 = u32::from(s.bytes[i + 1]) & 0x3f;
            let b2 = u32::from(s.bytes[i + 2]) & 0x3f;
            let b3 = u32::from(s.bytes[i + 3]) & 0x3f;
            let w1 = lt_mask(b0, 0x80);
            let w2 = eq_mask(b0 & 0xe0, 0xc0);
            let w3 = eq_mask(b0 & 0xf0, 0xe0);
            let w4 = eq_mask(b0 & 0xf8, 0xf0);
            let cp = (b0 & w1)
                | ((((b0 & 0x1f) << 6) | b1) & w2)
                | ((((b0 & 0x0f) << 12) | (b1 << 6) | b2) & w3)
                | ((((b0 & 0x07) << 18) | (b1 << 12) | (b2 << 6) | b3) & w4);
            let (id, hit) = self.char_id(cp, trace);
            // A char absent from the vocab is dropped, as `tokenizers` does
            // with no unknown token and no byte fallback.
            let live = (w1 | w2 | w3 | w4) & hit;
            s.tok[i] = id & live;
            s.alive[i] = live;
        }

        // Link every slot to the next live one.
        let (mut nxt, mut nxt_tok) = (NONE, 0u32);
        for i in (0..n).rev() {
            s.next[i] = nxt;
            s.ntok[i] = nxt_tok;
            let a = s.alive[i];
            nxt = select(a, i as u32, nxt);
            nxt_tok = select(a, s.tok[i], nxt_tok);
        }

        // The initial pairs, one query per slot, in one pass.
        for i in 0..n {
            let has = s.alive[i] & !eq_mask(s.next[i], NONE);
            s.queries[i] = (select(has, s.tok[i], NONE), select(has, s.ntok[i], NONE));
            s.valid[i] = has;
        }
        self.merge_pass(&s.queries[..n], &mut s.hits[..n], trace);
        for i in 0..n {
            let h = s.hits[i];
            s.rank[i] = h.rank;
            s.new_id[i] = h.new_id;
            s.valid[i] &= h.hit;
        }

        for _ in 1..n {
            // The lowest rank; a strict `<` in slot order keeps the left-most.
            let (mut best, mut p, mut new_id, mut fired) = (NONE, 0u32, 0u32, 0u32);
            for i in 0..n {
                let m = s.valid[i] & lt_mask(s.rank[i], best);
                best = select(m, s.rank[i], best);
                p = select(m, i as u32, p);
                new_id = select(m, s.new_id[i], new_id);
                fired |= s.valid[i];
            }
            // Its right partner r, and the live slot l before it.
            let (mut r, mut l, mut has_l, mut l_tok) = (NONE, 0u32, 0u32, 0u32);
            for i in 0..n {
                let at_p = eq_mask(i as u32, p);
                r = select(at_p, s.next[i], r);
                let before = s.alive[i] & lt_mask(i as u32, p);
                l = select(before, i as u32, l);
                has_l |= before;
                l_tok = select(before, s.tok[i], l_tok);
            }
            // The live slot after r, and its token.
            let (mut nr, mut nr_tok) = (NONE, 0u32);
            for i in 0..n {
                let at_r = eq_mask(i as u32, r);
                nr = select(at_r, s.next[i], nr);
                nr_tok = select(at_r, s.ntok[i], nr_tok);
            }
            let has_nr = !eq_mask(nr, NONE);
            // The pairs (l, p) and (p, nr) the merge creates, or sentinels.
            let ql = fired & has_l;
            let qr = fired & has_nr;
            s.queries[0] = (select(ql, l_tok, NONE), select(ql, new_id, NONE));
            s.queries[1] = (select(qr, new_id, NONE), select(qr, nr_tok, NONE));
            self.merge_pass(&s.queries[..2], &mut s.hits[..2], trace);
            let (hl, hr) = (s.hits[0], s.hits[1]);
            // Apply: p takes the new id and r's successor, r dies.
            for i in 0..n {
                let at_p = fired & eq_mask(i as u32, p);
                let at_r = fired & eq_mask(i as u32, r);
                let at_l = ql & eq_mask(i as u32, l);
                s.tok[i] = select(at_p, new_id, s.tok[i]);
                s.alive[i] &= !at_r;
                s.next[i] = select(at_p, nr, s.next[i]);
                s.ntok[i] = select(at_l, new_id, select(at_p, nr_tok, s.ntok[i]));
                s.rank[i] = select(at_l, hl.rank, select(at_p, hr.rank, s.rank[i]));
                s.new_id[i] = select(at_l, hl.new_id, select(at_p, hr.new_id, s.new_id[i]));
                s.valid[i] = select(at_l, hl.hit, select(at_p, hr.hit & qr, s.valid[i] & !at_r));
            }
        }

        // Compact the live ids to the front: slot i goes to position c.
        s.out[..n].fill(0);
        let mut c = 0u32;
        for i in 0..n {
            let a = s.alive[i];
            let t = s.tok[i] & a;
            for j in 0..n {
                s.out[j] |= t & a & eq_mask(j as u32, c);
            }
            c = c.wrapping_add(a & 1);
        }
        s.len = c as usize;
        s.ids()
    }
}

impl Tables for HashTables {
    fn vocab_get(&self, key: &str, trace: &mut Trace) -> Option<u32> {
        let cap = self.vocab.len();
        let mut slot = (fnv1a(key.as_bytes()) as usize) & (cap - 1);
        loop {
            trace.record(Table::Vocab, slot);
            match &self.vocab[slot] {
                Some((k, id)) if k == key => return Some(*id),
                Some(_) => slot = (slot + 1) & (cap - 1),
                None => return None,
            }
        }
    }
    fn merge_get(&self, pair: Pair, trace: &mut Trace) -> Option<(u32, u32)> {
        let cap = self.merges.len();
        let mut slot = (fnv1a(&pair_bytes(pair)) as usize) & (cap - 1);
        loop {
            trace.record(Table::Merges, slot);
            match &self.merges[slot] {
                Some((p, v)) if *p == pair => return Some(*v),
                Some(_) => slot = (slot + 1) & (cap - 1),
                None => return None,
            }
        }
    }
    fn vocab_slot(&self, key: &str) -> Option<usize> {
        self.vocab_pos.get(key).copied()
    }
    fn vocab_slots_len(&self) -> usize {
        self.vocab.len()
    }
    fn merge_slots_len(&self) -> usize {
        self.merges.len()
    }
    fn is_oblivious(&self) -> bool {
        false
    }
}

impl Tables for ObliviousTables {
    fn vocab_get(&self, key: &str, trace: &mut Trace) -> Option<u32> {
        let mut it = key.chars();
        let (Some(c), None) = (it.next(), it.next()) else {
            // A multi-char key: not on the tokenizer's path (it looks single
            // chars up); the whole table, in order, for completeness.
            let mut found = None;
            for (slot, entry) in self.vocab.iter().enumerate() {
                trace.record(Table::Vocab, slot);
                if let Some((k, id)) = entry {
                    if k == key {
                        found = Some(*id);
                    }
                }
            }
            return found;
        };
        let (id, hit) = self.char_id(c as u32, trace);
        (hit != 0).then_some(id)
    }
    fn merge_get(&self, pair: Pair, trace: &mut Trace) -> Option<(u32, u32)> {
        let mut q = [(pair, None)];
        self.merge_get_batch(&mut q, trace);
        q[0].1
    }
    fn merge_get_batch(&self, queries: &mut [(Pair, Option<(u32, u32)>)], trace: &mut Trace) {
        let keys: Vec<Pair> = queries.iter().map(|q| q.0).collect();
        let mut hits = vec![Hit::default(); keys.len()];
        self.merge_pass(&keys, &mut hits, trace);
        for (q, h) in queries.iter_mut().zip(hits) {
            q.1 = (h.hit != 0).then_some((h.rank, h.new_id));
        }
    }
    fn vocab_slot(&self, key: &str) -> Option<usize> {
        self.vocab_pos.get(key).copied()
    }
    fn vocab_slots_len(&self) -> usize {
        self.vocab.len()
    }
    fn char_slots_len(&self) -> usize {
        self.chars.len()
    }
    fn merge_slots_len(&self) -> usize {
        self.merges.len()
    }
    fn is_oblivious(&self) -> bool {
        true
    }
    fn as_oblivious(&self) -> Option<&ObliviousTables> {
        Some(self)
    }
}

/// A byte-pair-encoding tokenizer with no unknown token, no byte fallback and
/// empty affixes — the Qwen3/GPT-2 byte-level configuration, where every input
/// byte-level char is in the vocabulary. It mirrors `tokenizers`' `merge_word`
/// followed by `merge_all`, so its ids match that crate's for the same tables.
pub struct Tokenizer<T: Tables> {
    tables: T,
}

impl<T: Tables> Tokenizer<T> {
    pub fn new(tables: T) -> Self {
        Tokenizer { tables }
    }

    pub fn tables(&self) -> &T {
        &self.tables
    }

    /// Tokenize one piece, recording the table accesses; the ids are `tokenizers`'.
    /// An oblivious backend runs `tokenize_into` on buffers allocated here, and
    /// copies all `n` slots before cutting to the token count.
    pub fn tokenize(&self, piece: &str, trace: &mut Trace) -> Vec<u32> {
        if let Some(ob) = self.tables.as_oblivious() {
            let n = piece.len();
            let mut s = Scratch::with_len(n);
            let count = ob.tokenize_into(piece, &mut s, trace).len();
            let mut ids = s.out[..n].to_vec();
            ids.truncate(count);
            return ids;
        }
        // merge_word: one vocab lookup per char (no unk, no byte fallback).
        let mut ids: Vec<u32> = Vec::with_capacity(piece.len());
        let mut buf = [0u8; 4];
        for ch in piece.chars() {
            let s = ch.encode_utf8(&mut buf);
            if let Some(id) = self.tables.vocab_get(s, trace) {
                ids.push(id);
            }
        }
        self.merge_leaky(&mut ids, trace);
        ids
    }

    // Data-dependent: scan current pairs, apply the best, stop when none merges.
    fn merge_leaky(&self, ids: &mut Vec<u32>, trace: &mut Trace) {
        loop {
            let Some((pos, new_id)) = self.best_pair(ids, trace) else {
                break;
            };
            ids[pos] = new_id;
            ids.remove(pos + 1);
        }
    }

    // The lowest-rank, left-most mergeable adjacent pair, as (position, new_id).
    fn best_pair(&self, ids: &[u32], trace: &mut Trace) -> Option<(usize, u32)> {
        let mut best: Option<(u32, usize, u32)> = None;
        for i in 0..ids.len().saturating_sub(1) {
            let pair = (ids[i], ids[i + 1]);
            if let Some((rank, new_id)) = self.tables.merge_get(pair, trace) {
                if best.is_none_or(|(br, bp, _)| rank < br || (rank == br && i < bp)) {
                    best = Some((rank, i, new_id));
                }
            }
        }
        best.map(|(_, pos, new_id)| (pos, new_id))
    }
}

impl Tokenizer<ObliviousTables> {
    /// `ObliviousTables::tokenize_into` with the caller's buffers: no
    /// allocation once `s` fits the longest piece.
    pub fn tokenize_into<'s>(
        &self,
        piece: &str,
        s: &'s mut Scratch,
        trace: &mut Trace,
    ) -> &'s [u32] {
        self.tables.tokenize_into(piece, s, trace)
    }
}

/// The simulator's reconstruction of a piece from a HASH-backed trace: the host
/// knows the public hash and the slot->token layout, so the terminating slot of
/// each vocab lookup names the char it resolved. Returns the recovered string,
/// demonstrating the leak on OUR fixture.
pub fn reconstruct_from_line_view(tables: &HashTables, trace: &Trace, _layout: &Layout) -> String {
    // The vocab accesses, in order. Each lookup walks slots home, home+1, …
    // until it meets its key; that terminal slot names the char. A run of
    // consecutive (+1 mod cap) slots is one lookup; the token stored at the run's
    // last slot is what it resolved to.
    let cap = tables.vocab_cap();
    let slots: Vec<usize> = trace
        .accesses
        .iter()
        .filter(|a| a.table == Table::Vocab)
        .map(|a| a.slot)
        .collect();
    let mut out = String::new();
    let mut i = 0;
    while i < slots.len() {
        while i + 1 < slots.len() && slots[i + 1] == (slots[i] + 1) & (cap - 1) {
            i += 1;
        }
        if let Some(tok) = tables.vocab_entry_at(slots[i]) {
            out.push_str(tok);
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // A small byte-level-style BPE fixture: single-char tokens plus a few
    // merges, all present in the vocab, ranks by merge order.
    fn fixture() -> (Vec<(String, u32)>, Vec<(String, String)>) {
        let singles = ["a", "b", "c", "d", "e", "f", "g", "h"];
        let merged = ["ab", "abc", "cd", "ef", "gh", "bc", "cde", "abcd", "efgh"];
        let mut vocab = Vec::new();
        for (id, s) in singles.iter().chain(merged.iter()).enumerate() {
            vocab.push(((*s).to_string(), id as u32));
        }
        let merges = vec![
            ("a".to_string(), "b".to_string()),   // ab
            ("ab".to_string(), "c".to_string()),  // abc
            ("c".to_string(), "d".to_string()),   // cd
            ("e".to_string(), "f".to_string()),   // ef
            ("g".to_string(), "h".to_string()),   // gh
            ("b".to_string(), "c".to_string()),   // bc: competes with ab and cd
            ("cd".to_string(), "e".to_string()),  // cde
            ("ab".to_string(), "cd".to_string()), // abcd: competes with abc
            ("ef".to_string(), "gh".to_string()), // efgh
        ];
        (vocab, merges)
    }

    // The golden tokenizer: the real `tokenizers` BPE over the same tables.
    fn golden(vocab: &[(String, u32)], merges: &[(String, String)], piece: &str) -> Vec<u32> {
        use tokenizers::models::bpe::{Vocab, BPE};
        use tokenizers::tokenizer::Model;
        let v: Vocab = vocab.iter().cloned().collect();
        let bpe = BPE::new(v, merges.to_vec());
        bpe.tokenize(piece)
            .unwrap()
            .into_iter()
            .map(|t| t.id)
            .collect()
    }

    #[test]
    fn oblivious_tokens_equal_the_stock_tokenizer_bit_for_bit() {
        let (vocab, merges) = fixture();
        let b = build(vocab.clone(), &merges);
        let ob = Tokenizer::new(ObliviousTables::new(&b));
        let hh = Tokenizer::new(HashTables::new(&b));
        for piece in [
            "abcd", "abab", "efgh", "ghef", "abcabc", "dcba", "a", "hgfedcba",
        ] {
            let want = golden(&vocab, &merges, piece);
            let mut t = Trace::default();
            assert_eq!(
                ob.tokenize(piece, &mut t),
                want,
                "oblivious != stock on {piece:?}"
            );
            let mut t2 = Trace::default();
            assert_eq!(
                hh.tokenize(piece, &mut t2),
                want,
                "hash != stock on {piece:?}"
            );
        }
    }

    #[test]
    fn hash_backed_trace_differs_for_two_same_length_pieces() {
        // The leak: equal-length, equal-token-count pieces whose content differs
        // produce different cache-line traces.
        let (vocab, merges) = fixture();
        let b = build(vocab, &merges);
        let t = Tokenizer::new(HashTables::new(&b));
        let layout = Layout::default();
        let mut ta = Trace::default();
        let mut tb = Trace::default();
        let _ = t.tokenize("abab", &mut ta);
        let _ = t.tokenize("efef", &mut tb);
        assert_ne!(ta.line_view(&layout), tb.line_view(&layout));
    }

    #[test]
    fn oblivious_trace_identical_for_two_same_length_pieces() {
        // The fix: the oblivious backend's trace depends only on length.
        let (vocab, merges) = fixture();
        let b = build(vocab, &merges);
        let t = Tokenizer::new(ObliviousTables::new(&b));
        let layout = Layout::default();
        let mut ta = Trace::default();
        let mut tb = Trace::default();
        let _ = t.tokenize("abab", &mut ta);
        let _ = t.tokenize("efef", &mut tb);
        assert_eq!(ta, tb, "raw access sequences must match exactly");
        assert_eq!(ta.line_view(&layout), tb.line_view(&layout));
        assert_eq!(ta.page_view(&layout), tb.page_view(&layout));
    }

    #[test]
    fn oblivious_trace_identical_despite_different_merge_counts() {
        // A stronger attack on the length-only guarantee: two equal-length
        // pieces whose MERGE structure differs. "abcd" fires two merges
        // (ab, then abc), "dcba" fires none, yet every lookup — real or
        // sentinel-padded — scans the full table, so the recorded access
        // sequence must be byte-identical.
        let (vocab, merges) = fixture();
        let b = build(vocab, &merges);
        let t = Tokenizer::new(ObliviousTables::new(&b));
        let layout = Layout::default();
        let mut merges_fire = Trace::default();
        let mut merges_none = Trace::default();
        let ids_fire = t.tokenize("abcd", &mut merges_fire);
        let ids_none = t.tokenize("dcba", &mut merges_none);
        // The token outputs differ (proving the inputs truly diverge) …
        assert_ne!(ids_fire, ids_none);
        // … while the access traces do not, at every granularity.
        assert_eq!(
            merges_fire, merges_none,
            "raw access sequences must match exactly"
        );
        assert_eq!(
            merges_fire.line_view(&layout),
            merges_none.line_view(&layout)
        );
        assert_eq!(
            merges_fire.page_view(&layout),
            merges_none.page_view(&layout)
        );
    }

    #[test]
    fn simulator_reconstructs_a_piece_from_the_hash_trace() {
        // Exposure measured on our own fixture: the hash trace names the chars.
        let (vocab, merges) = fixture();
        let b = build(vocab, &merges);
        let t = Tokenizer::new(HashTables::new(&b));
        let layout = Layout::default();
        // A piece of single chars (no merges fire) so each vocab lookup is a
        // distinct leaked char.
        let mut tr = Trace::default();
        let _ = t.tokenize("hgfe", &mut tr);
        let got = reconstruct_from_line_view(t.tables(), &tr, &layout);
        assert_eq!(got, "hgfe");
    }

    #[test]
    fn oblivious_char_scan_is_the_whole_char_table_in_order() {
        let (vocab, merges) = fixture();
        let n = vocab.iter().filter(|(s, _)| s.chars().count() == 1).count();
        let b = build(vocab, &merges);
        let tables = ObliviousTables::new(&b);
        let mut tr = Trace::default();
        let _ = tables.vocab_get("c", &mut tr);
        let vocab_hits: Vec<usize> = tr
            .accesses
            .iter()
            .filter(|a| a.table == Table::Vocab)
            .map(|a| a.slot)
            .collect();
        assert_eq!(vocab_hits, (0..n).collect::<Vec<_>>());
    }

    // A deterministic pseudo-random corpus over the fixture alphabet: every
    // piece is a valid pre-tokenized input (each char is in the vocab), and the
    // competing merges (ab/bc/cd, abc/abcd) exercise the rank-then-position
    // selection order against the stock heap.
    fn corpus(n: usize, max_len: usize) -> Vec<String> {
        let alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        (0..n)
            .map(|_| {
                let len = 1 + (next() % max_len as u64) as usize;
                (0..len).map(|_| alphabet[(next() % 8) as usize]).collect()
            })
            .collect()
    }

    #[test]
    fn oblivious_tokens_equal_the_stock_tokenizer_on_a_random_corpus() {
        let (vocab, merges) = fixture();
        let b = build(vocab.clone(), &merges);
        let ob = Tokenizer::new(ObliviousTables::new(&b));
        let hh = Tokenizer::new(HashTables::new(&b));
        let mut merged_pieces = 0usize;
        for piece in corpus(400, 12) {
            let want = golden(&vocab, &merges, &piece);
            if want.len() < piece.chars().count() {
                merged_pieces += 1;
            }
            let mut t = Trace::default();
            assert_eq!(
                ob.tokenize(&piece, &mut t),
                want,
                "oblivious != stock on {piece:?}"
            );
            let mut t2 = Trace::default();
            assert_eq!(
                hh.tokenize(&piece, &mut t2),
                want,
                "hash != stock on {piece:?}"
            );
        }
        // The corpus must actually exercise merges, not only single chars: a
        // third of the pieces merge, so equality is checked on the merge path,
        // not just on single-char lookups.
        assert!(merged_pieces > 100, "only {merged_pieces} pieces merged");
    }

    #[test]
    fn oblivious_merge_table_scans_are_one_per_byte() {
        // The cost shape of the oblivious schedule: a piece of n bytes costs
        // exactly n full passes over the merge table (one for the initial
        // pairs, one per round for the two pairs a merge creates), whatever
        // its content. A per-pair scan would cost (n-1)^2 passes.
        let (vocab, merges) = fixture();
        let b = build(vocab, &merges);
        let t = Tokenizer::new(ObliviousTables::new(&b));
        let m = t.tables().merge_slots_len();
        for piece in ["a", "ab", "abcd", "dcba", "hgfedcba", "abcabcab"] {
            let mut tr = Trace::default();
            let _ = t.tokenize(piece, &mut tr);
            let merge_accesses = tr
                .accesses
                .iter()
                .filter(|a| a.table == Table::Merges)
                .count();
            assert_eq!(
                merge_accesses,
                m * piece.len(),
                "{piece:?}: {merge_accesses} merge-slot accesses, want {} passes of {m}",
                piece.len()
            );
        }
    }

    #[test]
    fn oblivious_char_lookup_scans_only_single_char_entries() {
        // merge_word looks single chars up, never longer entries: the
        // oblivious scan covers the single-char entries (256 for a byte-level
        // vocabulary) and not the whole table, a data-independent subset.
        let (vocab, merges) = fixture();
        let singles = vocab.iter().filter(|(s, _)| s.chars().count() == 1).count();
        let b = build(vocab, &merges);
        let tables = ObliviousTables::new(&b);
        assert_eq!(tables.char_slots_len(), singles);
        let mut tr = Trace::default();
        assert_eq!(tables.vocab_get("c", &mut tr), Some(2));
        let vocab_hits = tr
            .accesses
            .iter()
            .filter(|a| a.table == Table::Vocab)
            .count();
        assert_eq!(
            vocab_hits, singles,
            "a char lookup must scan the char table only"
        );
        // A char absent from the vocab costs the same scan and returns None.
        let mut tr2 = Trace::default();
        assert_eq!(tables.vocab_get("z", &mut tr2), None);
        assert_eq!(tr, tr2);
    }

    #[test]
    fn hash_backed_batch_lookup_still_leaks() {
        // The default batch is one probe per query, so its trace is the
        // concatenation of the per-key probe runs: content-dependent.
        let (vocab, merges) = fixture();
        let b = build(vocab, &merges);
        let tables = HashTables::new(&b);
        let mut ta = Trace::default();
        let mut tb = Trace::default();
        let mut qa = [((0u32, 1u32), None), ((2, 3), None)];
        let mut qb = [((4u32, 5u32), None), ((6, 7), None)];
        tables.merge_get_batch(&mut qa, &mut ta);
        tables.merge_get_batch(&mut qb, &mut tb);
        assert!(qa.iter().all(|q| q.1.is_some()) && qb.iter().all(|q| q.1.is_some()));
        assert_ne!(ta, tb);
    }

    // The module fixture plus `é` (two bytes in UTF-8) with a merge on it: one
    // byte class then mixes one- and two-byte chars. `z` stays out of the
    // vocab, so pieces holding it exercise the dropped-char path.
    fn fixture_utf8() -> (Vec<(String, u32)>, Vec<(String, String)>) {
        let (mut vocab, mut merges) = fixture();
        let n = vocab.len() as u32;
        vocab.push(("\u{e9}".to_string(), n));
        vocab.push(("\u{e9}a".to_string(), n + 1));
        merges.push(("\u{e9}".to_string(), "a".to_string()));
        (vocab, merges)
    }

    #[test]
    fn oblivious_tokens_equal_the_stock_tokenizer_with_multibyte_and_dropped_chars() {
        let (vocab, merges) = fixture_utf8();
        let b = build(vocab.clone(), &merges);
        let ob = Tokenizer::new(ObliviousTables::new(&b));
        let hh = Tokenizer::new(HashTables::new(&b));
        let alphabet = ['a', 'b', 'c', 'd', '\u{e9}', 'z'];
        let mut x: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut pieces: Vec<String> = [
            "",
            "z",
            "\u{e9}ab",
            "a\u{e9}a",
            "zab",
            "azb",
            "\u{e9}\u{e9}\u{e9}",
            "abzcd",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        for _ in 0..400 {
            let len = 1 + (next() % 10) as usize;
            pieces.push((0..len).map(|_| alphabet[(next() % 6) as usize]).collect());
        }
        // One Scratch for every piece, longest and shortest in any order.
        let mut s = Scratch::new();
        for piece in &pieces {
            let want = golden(&vocab, &merges, piece);
            assert_eq!(
                ob.tokenize(piece, &mut Trace::default()),
                want,
                "oblivious != stock on {piece:?}"
            );
            assert_eq!(
                hh.tokenize(piece, &mut Trace::default()),
                want,
                "hash != stock on {piece:?}"
            );
            assert_eq!(
                ob.tokenize_into(piece, &mut s, &mut Trace::off()),
                want.as_slice(),
                "reused scratch != stock on {piece:?}"
            );
        }
    }

    #[test]
    fn oblivious_trace_identical_across_one_and_two_byte_chars() {
        // The class is the UTF-8 byte length: "éab" (é is two bytes, one
        // merge) and "abcd" (two merges) are both four bytes.
        let (vocab, merges) = fixture_utf8();
        let b = build(vocab, &merges);
        let t = Tokenizer::new(ObliviousTables::new(&b));
        let mut ta = Trace::default();
        let mut tb = Trace::default();
        let ids_a = t.tokenize("\u{e9}ab", &mut ta);
        let ids_b = t.tokenize("abcd", &mut tb);
        assert_ne!(ids_a.len(), 0);
        assert_ne!(ids_a, ids_b);
        assert_eq!(ta, tb, "raw access sequences must match exactly");
    }

    #[test]
    fn a_duplicate_merge_rule_resolves_as_the_stock_tokenizer_does() {
        // The last rule of a pair wins in `tokenizers`. ab at 2 and 4 against bc
        // at 5: OR-ed ranks (6) let bc fire first. ab at 0 and 2 against bc at 1:
        // first-wins and last-wins disagree.
        let tokens = ["a", "b", "c", "d", "e", "f", "ef", "de", "ab", "cd", "bc"];
        let vocab: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .map(|(id, s)| ((*s).to_string(), id as u32))
            .collect();
        let m = |a: &str, b: &str| (a.to_string(), b.to_string());
        let lists = [
            vec![
                m("e", "f"),
                m("d", "e"),
                m("a", "b"),
                m("c", "d"),
                m("a", "b"),
                m("b", "c"),
            ],
            vec![m("a", "b"), m("b", "c"), m("a", "b")],
        ];
        for merges in lists {
            let b = build(vocab.clone(), &merges);
            let ob = Tokenizer::new(ObliviousTables::new(&b));
            let hh = Tokenizer::new(HashTables::new(&b));
            for piece in ["abc", "abcd", "cab", "aabbcc", "def", "abcdef"] {
                let want = golden(&vocab, &merges, piece);
                assert_eq!(
                    ob.tokenize(piece, &mut Trace::default()),
                    want,
                    "oblivious != stock on {piece:?} with {merges:?}"
                );
                assert_eq!(
                    hh.tokenize(piece, &mut Trace::default()),
                    want,
                    "hash != stock on {piece:?} with {merges:?}"
                );
            }
        }
        // The rule kept is the last one, at its own rank.
        let b = build(vocab.clone(), &[m("a", "b"), m("b", "c"), m("a", "b")]);
        assert_eq!(b.merges, vec![((1, 2), (1, 10)), ((0, 1), (2, 8))]);
    }

    #[test]
    fn mask_primitives_hold_at_the_edges() {
        let edges = [
            0u32,
            1,
            2,
            0x7fff_ffff,
            0x8000_0000,
            0x8000_0001,
            u32::MAX - 1,
            u32::MAX,
        ];
        for &a in &edges {
            for &b in &edges {
                assert_eq!(
                    eq_mask(a, b),
                    if a == b { u32::MAX } else { 0 },
                    "eq {a} {b}"
                );
                assert_eq!(
                    lt_mask(a, b),
                    if a < b { u32::MAX } else { 0 },
                    "lt {a} {b}"
                );
                assert_eq!(select(u32::MAX, a, b), a);
                assert_eq!(select(0, a, b), b);
            }
        }
    }

    // Reported, not asserted: the per-prompt cost of obliviousness on the
    // fixture. `cargo test … -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, printed not asserted"]
    fn obtok_overhead() {
        use std::time::Instant;
        let (vocab, merges) = fixture();
        let b = build(vocab, &merges);
        let leaky = Tokenizer::new(HashTables::new(&b));
        let obliv = Tokenizer::new(ObliviousTables::new(&b));
        let corpus = ["abcd", "abcabc", "hgfedcba", "efgh", "abab", "dcba"];
        let iters = 20_000u32;
        let bench = |t: &dyn Fn(&str)| {
            let start = Instant::now();
            for _ in 0..iters {
                for p in corpus {
                    t(p);
                }
            }
            start.elapsed().as_secs_f64() / f64::from(iters) / corpus.len() as f64 * 1e3
        };
        let ml = bench(&|p| {
            let mut tr = Trace::default();
            let _ = leaky.tokenize(p, &mut tr);
        });
        let mo = bench(&|p| {
            let mut tr = Trace::default();
            let _ = obliv.tokenize(p, &mut tr);
        });
        println!("obtok_overhead: leaky={ml:.6} ms/input oblivious={mo:.6} ms/input ratio={:.1}x (fixture vocab={})", mo / ml, b.vocab.len());
    }

    // Measured on the REAL tokenizer the server loads: point X0NE_OBTOK_VOCAB at a
    // `tokenizer.json` (e.g. Qwen3's). Reconstructs a prompt from the hash
    // trace against the full vocabulary, proves the oblivious backend matches
    // `tokenizers`' BPE token-for-token on it, and reports the per-prompt cost
    // of a linear-scan oblivious lookup over a production-size table. Reads no
    // hardware channel; the model file is never committed.
    //
    //   X0NE_OBTOK_VOCAB=/path/to/tokenizer.json \
    //     cargo test -p x0ne-infer --lib oblivious_tok::tests::obtok_real_vocab \
    //     -- --ignored --nocapture
    #[test]
    #[ignore = "measurement, needs X0NE_OBTOK_VOCAB=<tokenizer.json>"]
    fn obtok_real_vocab() {
        let path = match std::env::var("X0NE_OBTOK_VOCAB") {
            Ok(p) => p,
            Err(_) => {
                println!("obtok_real_vocab: X0NE_OBTOK_VOCAB unset, nothing to measure");
                return;
            }
        };
        let data = std::fs::read(&path).expect("read tokenizer.json");
        let v: serde_json::Value = serde_json::from_slice(&data).expect("parse tokenizer.json");
        let model = &v["model"];
        let vocab_obj = model["vocab"].as_object().expect("model.vocab object");
        let mut vocab: Vec<(String, u32)> = vocab_obj
            .iter()
            .map(|(k, val)| (k.clone(), val.as_u64().unwrap() as u32))
            .collect();
        vocab.sort_by_key(|(_, id)| *id);
        let in_vocab: std::collections::HashSet<&str> =
            vocab.iter().map(|(s, _)| s.as_str()).collect();
        let merges: Vec<(String, String)> = model["merges"]
            .as_array()
            .expect("model.merges array")
            .iter()
            .map(|m| match m {
                serde_json::Value::Array(a) => (
                    a[0].as_str().unwrap().to_string(),
                    a[1].as_str().unwrap().to_string(),
                ),
                serde_json::Value::String(s) => {
                    let mut it = s.splitn(2, ' ');
                    (
                        it.next().unwrap().to_string(),
                        it.next().unwrap().to_string(),
                    )
                }
                _ => panic!("merge entry neither array nor string"),
            })
            // A merge whose child or result is not a plain vocab entry (added
            // tokens live outside model.vocab) would make `build` panic; such
            // rows never fire on ASCII pieces, so dropping them changes no
            // output here while keeping the measurement robust.
            .filter(|(a, b)| {
                in_vocab.contains(a.as_str())
                    && in_vocab.contains(b.as_str())
                    && in_vocab.contains(format!("{a}{b}").as_str())
            })
            .collect();
        let n_v = vocab.len();
        let n_m = merges.len();
        let built = build(vocab.clone(), &merges);
        let leaky = Tokenizer::new(HashTables::new(&built));
        let obliv = Tokenizer::new(ObliviousTables::new(&built));
        let layout = Layout::default();

        // ASCII-letter pieces: every char is its own byte-level vocab token, so
        // these are valid pre-tokenized pieces and exercise real merges.
        let pieces = [
            "performance",
            "obliviousness",
            "confidential",
            "tokenization",
        ];
        for piece in pieces {
            let want = golden(&vocab, &merges, piece);
            let mut th = Trace::default();
            let got_hash = leaky.tokenize(piece, &mut th);
            let mut to = Trace::default();
            let got_obl = obliv.tokenize(piece, &mut to);
            assert_eq!(got_hash, want, "hash != stock on {piece:?}");
            assert_eq!(got_obl, want, "oblivious != stock on {piece:?}");
            let recovered = reconstruct_from_line_view(leaky.tables(), &th, &layout);
            let sim = similarity(&recovered, piece);
            println!(
                "obtok_real_vocab: piece={piece:?} ids={} recovered={recovered:?} similarity={sim:.1}%",
                got_hash.len()
            );
            // Open-addressing adjacency can occasionally fuse two lookups, so
            // (as in TDXRay's ~91.5% figure) reconstruction is near-exact, not
            // guaranteed bit-exact on a production table.
            assert!(
                sim >= 90.0,
                "reconstruction similarity {sim:.1}% too low on {piece:?}"
            );
        }

        // Obliviousness on the real table: two equal-length real pieces.
        let mut a = Trace::default();
        let mut b = Trace::default();
        let _ = obliv.tokenize("performance", &mut a);
        let _ = obliv.tokenize("engineering", &mut b); // both 11 chars
        assert_eq!(
            a, b,
            "oblivious trace must be identical for equal-length pieces"
        );

        // Whole prompts end to end: the crate's own pre-tokenizer (the regex
        // split, then the byte-level mapping) cuts the prompt into pieces, and
        // our tokenizer over those pieces must equal the crate's `encode` id
        // for id. The prompts are ASCII, so the NFC normalizer is identity.
        use tokenizers::tokenizer::{
            OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer,
        };
        let stock = tokenizers::Tokenizer::from_bytes(&data).expect("tokenizer.json");
        let pre = stock.get_pre_tokenizer().expect("pre-tokenizer");
        let prompts = [
            "the quick brown fox jumps over the lazy dog",
            "neither the provider nor the host can read your prompt by software",
            "a data oblivious tokenizer scans the table the same way for every input",
        ];
        let pieces_of = |prompt: &str| -> Vec<String> {
            let mut pts = PreTokenizedString::from(prompt);
            pre.pre_tokenize(&mut pts).expect("pre_tokenize");
            pts.get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, _, _)| s.to_string())
                .collect()
        };
        let mut by_len: Vec<(usize, Trace)> = Vec::new();
        for prompt in prompts {
            let want: Vec<u32> = stock
                .encode(prompt, false)
                .expect("encode")
                .get_ids()
                .to_vec();
            let pieces = pieces_of(prompt);
            let mut got_obl = Vec::new();
            let mut got_hash = Vec::new();
            for piece in &pieces {
                let mut to = Trace::default();
                got_obl.extend(obliv.tokenize(piece, &mut to));
                let mut th = Trace::default();
                got_hash.extend(leaky.tokenize(piece, &mut th));
                by_len.push((piece.len(), to));
            }
            assert_eq!(got_obl, want, "oblivious != stock encode on {prompt:?}");
            assert_eq!(got_hash, want, "hash != stock encode on {prompt:?}");
            println!(
                "obtok_real_vocab: input {} chars, {} pieces, {} ids: oblivious == stock encode",
                prompt.len(),
                pieces.len(),
                want.len()
            );
        }
        // Across all those real pieces, the oblivious trace is a function of
        // the byte length alone: equal length implies an identical trace.
        by_len.sort_by_key(|(n, _)| *n);
        for w in by_len.windows(2) {
            if w[0].0 == w[1].0 {
                assert_eq!(
                    w[0].1, w[1].1,
                    "two {}-byte pieces, different traces",
                    w[0].0
                );
            }
        }

        // Cost: the longest prompt, timed end to end per backend. The crate's
        // own `encode` (its hash maps and BPE cache) is the production baseline.
        let long = prompts[2];
        let long_pieces = pieces_of(long);
        fn bench<T: Tables>(t: &Tokenizer<T>, pieces: &[String], iters: u32) -> f64 {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                for p in pieces {
                    let mut tr = Trace::off();
                    let _ = t.tokenize(p, &mut tr);
                }
            }
            start.elapsed().as_secs_f64() / f64::from(iters) * 1e3
        }
        let iters = 20u32;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = stock.encode(long, false).unwrap();
        }
        let ms = start.elapsed().as_secs_f64() / f64::from(iters) * 1e3;
        let ml = bench(&leaky, &long_pieces, iters);
        let mo = bench(&obliv, &long_pieces, iters);
        println!(
            "obtok_real_vocab: vocab={n_v} merges={n_m} chars={} input={} pieces/{} chars: \
             stock encode={ms:.3} ms, hash model={ml:.3} ms, oblivious={mo:.1} ms \
             ({:.0}x stock encode, {:.0}x hash model)",
            obliv.tables().char_slots_len(),
            long_pieces.len(),
            long.chars().count(),
            mo / ms,
            mo / ml
        );
    }

    // Length-normalized similarity (1 - edit_distance/max_len), the TDXRay
    // metric, for the reconstruction report. Char-level Levenshtein.
    fn similarity(a: &str, b: &str) -> f64 {
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        let (n, m) = (a.len(), b.len());
        if n == 0 && m == 0 {
            return 100.0;
        }
        let mut prev: Vec<usize> = (0..=m).collect();
        let mut cur = vec![0usize; m + 1];
        for i in 1..=n {
            cur[0] = i;
            for j in 1..=m {
                let cost = usize::from(a[i - 1] != b[j - 1]);
                cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        let dist = prev[m];
        (1.0 - dist as f64 / n.max(m) as f64) * 100.0
    }
}
