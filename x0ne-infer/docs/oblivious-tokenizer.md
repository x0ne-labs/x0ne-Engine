# Tokenizer side-channel exposure, and a data-oblivious tokenizer

Status: R&D. Not wired into the serving path. This is a measurement of a known
leak and a tested countermeasure for it, behind `x0ne_infer::oblivious_tok`.

## The leak

An LLM server converts the prompt to token ids before any matrix multiply. A
standard BPE tokenizer does this with hash-map lookups: one lookup per input
character into the vocabulary, then a lookup per candidate pair into the merge
table. The sequence of memory slots those lookups touch is a function of the
bytes looked up.

A confidential-computing guest runs this on the CPU. Memory encryption (TDX/SEV)
keeps the guest's memory contents from the host's software, short of the
physical attacks THREAT_MODEL.md lists, but the host can observe *which* memory
the guest touches:

- page granularity, through controlled-channel page faults (the hypervisor
  revokes a guest page's translation and sees the fault when it is used);
- cache-line granularity, through physical-alias cache probes (Load+Probe,
  TSX-Probe).

Because the hash function is public and the table layout is stable, each touched
slot maps back to a token. TDXRay (IEEE S&P 2026) reconstructs prompts from a
single tokenization at 91.5% similarity for Llama 3.2 and 94.2% for Gemma 3,
tokenizing in only 62–346 ms, imperceptibly to the user. Spill the Beans
(arXiv:2505.00817) is the de-tokenization analogue: a Flush+Reload on the
embedding/token rows leaks output tokens (reported up to 80–90% of a
high-entropy key, ~40% of English text). Intel's advisory SA-01271
(2026-04-02) and AMD's SB-3044 accompany the TDXRay disclosure.

This is a confidentiality gap *inside* the attested boundary: the data stays
encrypted, but its access pattern does not, so "the host cannot read your
prompt by software" is not yet true for the tokenization phase.

## What `oblivious_tok` measures (the exposure)

`HashTables` models the standard tokenizer as an open-addressing map: a lookup
probes `hash(key) mod n`, `+1`, … until it meets the key, recording every
probed slot. A `Trace` is the ordered slot sequence; `Layout` maps a slot to its
page and cache-line index (fixed-size entries, 4 KiB pages, 64 B lines), so
`Trace::page_view` and `Trace::line_view` are what a host observes at each
granularity.

Measured on the in-repo fixture (`cargo test -p x0ne-infer --lib oblivious_tok`):

- two equal-length pieces with different content produce **different**
  cache-line traces (`hash_backed_trace_differs_for_two_same_length_pieces`);
- from a hash-backed trace, `reconstruct_from_line_view` recovers the exact
  piece, because each lookup's terminating slot names its character
  (`simulator_reconstructs_a_piece_from_the_hash_trace`).

The reconstruction runs only on fixtures built in this module, against our own
tokenizer, to size our own exposure. It reads no real hardware channel.

## The countermeasure (data-oblivious tokenization)

`ObliviousTables` stores the same vocabulary and merges but answers every lookup
by a fixed scan, never a key-dependent probe. A single-character lookup scans the
char table — the vocabulary's single-character entries, 256 for a byte-level
vocabulary — in slot order `0..c`, comparing each entry to the query with a
branch-free equality mask. A set of pair lookups is answered by **one** pass over
the merge table in slot order `0..m`, each rule compared to every query by the
same mask select. When two rules name the same pair, the later one wins, as in
`tokenizers`, which collects the rules into a hash map; `build` keeps only that
rule.

The merge stage runs on top with a fixed schedule over fixed buffers
(`Scratch`, one slot per byte of the piece, reused across calls):

- each slot decodes the UTF-8 sequence that starts at its byte with masks (a
  continuation byte gives a dead slot) and scans the char table; a char absent
  from the vocabulary is dropped, as `tokenizers` does with no unknown token;
- a backward scan links every slot to the next live one;
- the initial pairs, one query per slot, are resolved in one merge-table pass;
- then exactly `n - 1` rounds for `n` bytes: a masked scan picks the
  lowest-rank, left-most live pair; two masked scans find its right partner,
  the live slot before it and the slot after the partner; one merge-table pass
  resolves the two pairs the merge creates (sentinel pairs when no merge
  fired); a masked write to every slot applies the merge;
- a masked scan of every (slot, position) compacts the live ids to the front.

Every loop bound and every index is a function of `n`; nothing is removed from a
vector and no buffer is indexed by a secret value. The class of a piece is
therefore its UTF-8 **byte** length: `"éab"` and `"abcd"` (four bytes each) are
indistinguishable. Every mask passes through `std::hint::black_box`, so the
compiler cannot prove it is all-ones or zero and turn the selects back into
branches. That is best effort, as in any Rust code; the measurement below is
what shows it held for one build. The merge *selection* (lowest rank, left-most,
one per round) is `tokenizers`' heap order, so the token output should be the
standard tokenizer's; the tests below compare the two on a fixture vocabulary,
not on a production one.

`Tokenizer::tokenize` runs the same stage on buffers it allocates for the call,
sized from `n`, and returns a copy of all `n` slots cut to the token count.
`Tokenizer::tokenize_into` takes the caller's `Scratch`, so a call on a piece
that fits allocates nothing.

Tested (simulated trace, `cargo test -p x0ne-infer --lib oblivious_tok`):

- `oblivious_tokens_equal_the_stock_tokenizer_bit_for_bit`,
  `oblivious_tokens_equal_the_stock_tokenizer_on_a_random_corpus` and
  `oblivious_tokens_equal_the_stock_tokenizer_with_multibyte_and_dropped_chars`
  — the oblivious and the leaky backend both equal
  `tokenizers::models::bpe::BPE` (the crate the server uses today),
  token-for-token, on hand-picked pieces and on deterministic random corpora,
  one of them with a two-byte character, a character absent from the
  vocabulary, the empty piece, and one `Scratch` reused across every length;
- `a_duplicate_merge_rule_resolves_as_the_stock_tokenizer_does` — two rules for
  one pair: the ranks used to be OR-ed (ab at 2 and 4 gave 6, so bc at 5 fired
  first, `[a, bc]` where the crate gives `[ab, c]`); the last rule now wins, as
  in the crate, in both backends;
- `oblivious_trace_identical_for_two_same_length_pieces`,
  `oblivious_trace_identical_despite_different_merge_counts` and
  `oblivious_trace_identical_across_one_and_two_byte_chars` — two different
  pieces of one byte length (`"abcd"` fires two merges, `"dcba"` none, `"éab"`
  one) yield byte-identical raw, page and cache-line traces;
- `oblivious_merge_table_scans_are_one_per_byte` — a piece of `n` bytes costs
  exactly `n` full passes over the merge table, whatever its content;
- `oblivious_char_scan_is_the_whole_char_table_in_order` and
  `oblivious_char_lookup_scans_only_single_char_entries` — a char lookup touches
  every single-char slot, in order, and a char absent from the vocab costs the
  identical scan;
- `mask_primitives_hold_at_the_edges` — the equality, less-than and select masks
  on 0, 1, 2^31 - 1, 2^31, 2^31 + 1 and u32::MAX;
- `hash_backed_batch_lookup_still_leaks` — the leaky backend's batch is still one
  probe per query, so its trace stays content-dependent.

## Measured on the real binary

The tests above check the *simulated* trace: the table slots the code records.
The binary's own streams are measured by `scripts/obtok-trace.sh [rev]` (default
`HEAD`, results in `out/obtok-trace/<sha>/`). It extracts the revision with fresh
file times, so cargo never reuses another revision's build, builds
`examples/obtok_trace.rs` in release in the CI image with the crates `make ci`
vendored, no network and under the build lock, then runs it under valgrind
lackey (`ci/Containerfile.valgrind`, built on first use, the only networked
step), which logs every instruction fetch and every data access with its
address and size.

`obtok_trace run <mode> <piece>...` tokenizes every piece once to warm up, then
each again between two stores to a marker word, from one input buffer, so the
log cuts into one window per piece. `obtok_trace compare` groups the windows by
byte length and compares, within each class, the instruction addresses and the
data accesses (kind, address, size). Three modes: `obl`, `tokenize_into` with
one reused `Scratch`, must give identical windows; `hash`, the leaky model, must
not (the control: the measurement can see a leak); `obl-vec`, `tokenize`, is
reported and not judged. Exit 0 when both judged modes hold, 1 when one fails,
2 when the log does not hold one window per piece.

Measured on 2026-10-24 (rust 1.98.1, release, x86-64, valgrind 3.24.0):

| Revision | Oblivious tokenizer | Hash model (control) |
|---|---|---|
| the merge stage before this design (`Vec::remove` at the merged position, a branch on whether a merge fired) | **differ**: 4-byte pieces `abcd` 4 490 instructions, `dcba` 4 302, `efgh` 4 550, `éab` 3 372; 6-byte pieces 6 961, 6 694, 5 888, 4 474 | differ |
| this design, `tokenize_into` with one reused `Scratch`, 22 pieces | **identical** within every class: 990, 2 900, 8 094, 15 139 and 24 016 instructions for 1, 2, 4, 6 and 8 bytes, and the data accesses identical address for address | differ |

In both runs a piece repeated in its class gave the same stream, so a difference
is the input's, not noise. `tokenize` (mode `obl-vec`, which allocates in the
call) ran the same instructions within every class, but its data addresses
differed even between two calls on the same piece: the allocator returned other
chunks. Only `tokenize_into` with a reused `Scratch` is claimed.

## Overhead

A char lookup is O(c) per byte (c = 256 for a byte-level vocabulary), and the
merge stage O(n · m) for `n` bytes and `m` merges: `n` passes over the merge
table, each comparing every rule to one query per slot (the first pass) or to
two (the rounds), plus O(n²) for the masked scans of the slots. Measured in
release on 2026-10-24 by the ignored tests (`cargo test --release -p x0ne-infer
--lib oblivious_tok -- --ignored --nocapture`, the latter with
`X0NE_OBTOK_VOCAB=<tokenizer.json>`), one run each: 6.5× the leaky model on the
fixture; on the real Qwen3 vocabulary (151 643 tokens, 151 387 merges, no
duplicate rule), a 71-character input cut into 13 pieces takes 37.9 ms against
0.016 ms for the stock `tokenizers::encode`, about 2 400×. The previous design
measured 22.8 ms in release: the masks behind `black_box` cost about 1.7×. (An
earlier version of this document quoted 1.1 s, a debug build.) A linear scan is
too costly for production; an oblivious map (Oblivious Labs, TDXRay's remedy) or
an ORAM is the path to a production-grade overhead. This module proves output
equivalence and trace equality first.

## Scope and limits

What the measurement covers: the instructions the tokenizer executes and the
data addresses it touches, from the call to its return, on one build (rust
1.98.1, release profile, x86-64, the CI image) and on the fixture pieces of the
script. The argument for every input is the construction (every bound and index
a function of `n`); the measurement checks that the compiler kept it.

Not covered:

- **The class itself.** The byte length of each piece, and the number of pieces,
  stay visible by design. The pre-tokenizer that cuts the prompt into pieces and
  the NFC normalizer before it have their own data-dependent branches; this
  module does not address them.
- **The allocator.** `tokenize_into` with a fitting `Scratch` allocates nothing.
  `tokenize` allocates its buffers and its vector, sized from `n`; whether
  malloc then runs the same instructions depends on the heap's state, which this
  module does not control. Measured: `tokenize` ran the same instructions within each class, but its
  data addresses moved between two calls on the same piece.
- **The standard library at other addresses.** Copying the piece into the
  buffers (`copy_from_slice`, `fill`) runs `memcpy`/`memset`, whose path depends
  on the length and on the alignment of the source address, not on the bytes.
  The measurement holds the input address fixed; a caller whose pieces sit at
  differently aligned addresses changes that path.
- **Caches, timing and the rest of the microarchitecture.** lackey records the
  sequence of addresses, not cache state, port contention or frequency. Equal
  sequences from equal starting states give equal cache behaviour; the starting
  state, set by whatever ran before, is not controlled here.
- **Another compiler or profile.** The constant-time masks are best effort; a
  different rustc, profile or target must be measured again (the script takes a
  revision).
- **What the caller does with the ids.** The slice `tokenize_into` returns has
  the token count as its length, and the ids index the embedding table next:
  both leave the tokenizer. De-tokenization and the embedding lookup (Spill the
  Beans) are not covered.
- **Production tables.** The in-repo tests and the trace script use a small
  fixture so the constant scan is fast and needs no model files. Bit-exact
  equality against a real tokenizer (Qwen3), the reconstruction of a real
  prompt, and the per-prompt cost over a production-size vocabulary are an
  out-of-tree measurement — the ignored `obtok_real_vocab` test, given
  `X0NE_OBTOK_VOCAB=<tokenizer.json>`, since the model file is not committed.

## Sources

- TDXRay, IEEE S&P 2026 — <https://tdxray.cpusec.org/>
- Spill the Beans — <https://arxiv.org/abs/2505.00817>
- Intel SA-01271 (2026-04-02); AMD SB-3044
- Oblivious Labs oblivious map — <https://github.com/obliviouslabs/>
