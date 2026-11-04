# Keeping the KV cache between the turns of one conversation

Status: implemented on 2026-10-30 in the engine before x0ne Core, opt-in
(`x0ne.conversation`,
[api.md](api.md#x0neconversation-a-conversations-keys-and-values-kept-between-turns)).
A request that does not ask for it is served as before: its cache is
dropped at its end, on every exit path, zeroed first on Qwen3 and Ministral
3 (OLMo's cache is dropped, not zeroed), and its receipt signs
`kv_cache_shared_between_requests: false`. Nothing here changes the
README's statement of what the product is not.

The code this note names is that of the engine before x0ne Core, which
x0ne Core replaced on 2026-11-04 ([CHANGELOG](../CHANGELOG.md)), and of
families no longer served (Ministral 3, OLMo). That engine's backend,
`candle_backend.rs`, is now the CPU reference of the tests and is never in
a build that serves. x0ne Core keeps no conversation in this version: a
request that asks for one gets a 400 naming `x0ne.conversation`, on every
model, and each turn is computed from token 0
([limitations](limitations.md#conversations-kept-between-turns)).

A chat client sends the whole conversation at every turn, and the TD
prefilled it from token 0: the first token of a turn waited for the whole
history, not for the new message. Keeping the keys and values of one
conversation between its turns removes that work. It is also the surface
that the prompt-leak attacks on serving engines target; this note gives the
scope, the lifetime, what the receipt says and what stays observable.

## What the attacks teach

- **Shared prefix cache.** PromptPeek (Wu et al., NDSS 2025,
  <https://ndss-symposium.org/wp-content/uploads/2025-1772-paper.pdf>)
  rebuilds another user's prompt token by token from the serving order of a
  cache shared between users (SGLang's longest-prefix-match scheduling):
  95 % to 99 % success on Llama 2 13B. "The Early Bird Catches the Leak"
  (Song et al., arXiv 2409.20002 v5, 2025, <https://arxiv.org/abs/2409.20002>)
  detects a cache hit per token from timing at 99 % on SGLang 0.3.0; sharing
  by blocks of k tokens lowers recovery (91.5 % at k = 1, 49 % at k = 4)
  without stopping it. InputSnatch (Zheng et al., arXiv 2411.18191, 2024,
  <https://arxiv.org/abs/2411.18191>) times prefix and semantic caches.
- **Scope.** Gu et al. (ICML 2025, arXiv 2502.07776,
  <https://arxiv.org/abs/2502.07776>) detected a cache shared across users
  at 7 of 17 API providers; they class a cache scoped to one user as
  leaking nothing across users, and recommend that scope. OpenAI does not share caches across
  organizations (<https://developers.openai.com/api/docs/guides/prompt-caching>,
  5 to 10 minutes in memory, up to one hour).
- **A scope must hold on every route.** vLLM's `cache_salt` mixes a
  per-request salt into the hash of the first block
  (<https://docs.vllm.ai/en/latest/design/prefix_caching/>); CVE-2026-105752
  is that salt dropped on one route (THREAT_MODEL.md, "Known attacks").
- **Content as a key.** TDXRay (IEEE S&P 2026, <https://tdxray.cpusec.org/>)
  traces a TDX guest's memory accesses from the host and recovers a prompt
  during tokenization on the CPU. A table indexed by content in the guest
  is exposed in the same way; a table indexed by an opaque handle is not
  indexed by the text.
- **Remanence.** No published attack on freed KV memory was found. Zeroing
  an evicted cache guards against a later reuse of the buffer (the CUDA
  allocator's pool, a bug that reads past a length), not against timing.

## Before this change

- One model, one KV cache behind a `Mutex`, a forward of batch one
  (`candle_backend.rs`, `Loaded::model`). The cache was cleared after the
  generation loop on every exit path. Clearing dropped the tensors without
  overwriting them (`qwen3.rs`, `LayerCache::clear`); growing the cache
  copied it and dropped the old buffer as it was (`qwen3.rs`, `grown`).
- No request named a conversation: each carried the whole history.
- Receipts signed `Isolation` constants (`backend.rs`, `NONE_OF_IT`), which
  neither the relay nor the SDK read.

## Design

### Scope and key

- **Opt-in.** `"x0ne": {"conversation": "new"}` opens a conversation and
  `"x0ne": {"conversation": "<64 hex digits>"}` continues one
  (`openai/request.rs`, `context.rs`).
- **The TD draws the handle**: 32 bytes of the server's salt source,
  `/dev/urandom` in production (`server.rs`, `conversation_turn`), never a
  value the caller chose. It travels only in the answer
  (`x0ne.conversation.handle`, in a whole answer and in a stream's last
  chunk), which on the sealed path only the caller opens.
- **Single use.** Each answer carries the next handle; the one presented is
  spent when its turn is admitted, whatever happens next
  (`Slot::admit`). A handle copied from an older answer continues nothing.
- **Bound to the route.** The resident conversation records the route that
  issued its handle, `/v1/chat/completions` or `/v2/e2e/chat`
  (`conversation.rs`, `Route`). A handle shown on the other route continues
  nothing: the relay, which holds the TD's client key and may send on
  `/v1`, cannot spend a sealed-path handle even if one leaked to it.
- **Exact tokens, never a hash.** The TD keeps the token ids whose keys and
  values are resident. A continuing request is tokenized as any other and
  compared with them id by id; the longest common prefix is reused, at most
  the prompt less its last token, and the rest computed. No digest of a
  prefix is a key or a match criterion. The comparison runs over the
  shorter of the two lengths wherever the first difference falls
  (`common_prefix`), and the handle is compared in constant time.
- **Unknown, spent, expired or other-route handle**: the turn is answered
  from token 0 under a new handle, and its receipt says `turn: 1`, 0
  tokens reused.

### Lifetime, eviction and zeroing

- **One resident conversation per model**: the model's own cache is the
  slot (`candle_backend.rs`, `Slot`: the model, the resident conversation
  and its deadline under one lock). Any other request, another
  conversation or a request without `conversation`, evicts it before it
  runs.
- **Evicted** by that request; after 300 s without a turn (a thread named
  `x0ne-conversation` that takes the model lock at the deadline and holds
  only a weak reference to the model); on an error, a cancellation or a
  panic during a turn, and when the answer fails its checks after the
  generation (a text that is not prefix-stable, outside its schema, a
  client gone before the last delta); after a turn that asks
  `"end_conversation": true`; when the model is dropped.
- **Not evicted early**: when the serving lease ends, the cache stays until
  one of the above, 300 s at most; a process that ends or is killed does not
  zero it, and its memory goes back to the TD's kernel.
- **Zeroed**: eviction writes zeros over every position written in every
  cache buffer of Qwen3 and Ministral 3 (`Tensor::zero_set` on the written
  part of each layer's keys and values, which candle 0.11.0 implements as
  `const_set` over the tensor's own storage: on the CPU, measured, the
  tests read zeros back; on CUDA, a `const_set_*` kernel, read in candle's
  source and never run on a GPU: candle runs on the CPU only since
  2026-11-04) before dropping it; so does the end of
  every request without a conversation. OLMo's cache is dropped, not
  zeroed. Growing the cache zeroes the old
  buffer, and the copy made on the way, before dropping them. A turn that
  continues a conversation and reuses fewer positions than the cache holds
  leaves the positions past them as they are until the next append writes
  over them or the cache is zeroed; they are never read again. The token
  ids kept beside the cache, and the handles, are zeroed when dropped
  (`zeroize`).
- **Not zeroed**: the tensors one forward derives from the cache (keys and
  values repeated per query head, attention scores, activations), freed to
  the allocator as before; the copies of the text the tokenizer and the
  HTTP stack make. The statement covers the cache buffers, not every byte
  the process held.
- **Families.** Qwen3 and Ministral 3 write their cache in place
  (`qwen3.rs`, `LayerCache`) and keep conversations. OLMo 2 and 3 do not:
  their concatenating cache leaves a freed copy per step, so a conversation
  asked of them gets a 400 naming `x0ne.conversation` (`backend.rs`,
  `ConversationUnsupported`) rather than an answer whose receipt the
  caller's verifier would refuse. So did x0ne Core before it became the
  served engine: its model did not implement `generate_turn` (the trait's
  default refuses).

### Memory cap

- One conversation's cache, at most the context the model is served with,
  which `serving_limits` already reserves for one generation
  (`serving_peak_bytes`): keeping it between turns adds no device memory, it
  is memory not freed. A turn that outgrows the cache kept copies it into a
  buffer of that turn's reservation, its prompt and its token budget, which
  the served context bounds (`qwen3.rs`, `LayerCache::append`), one layer at
  a time, and zeroes the old one. Qwen3-0.6B in F32 on the CPU: 28 layers, 8 KV heads of
  128, keys and values in 4 bytes, 224 KiB a token, 3.5 GiB at 16 384 tokens.
- Several resident conversations would need a byte cap under the serving
  budget and an eviction order. Not in this design.

### What the receipt adds

- `isolation.kv_cache_shared_between_requests: true` on every turn of a
  conversation, the first included, since its keys and values stay for the
  next request. `prefix_cache_shared_between_sessions` stays `false`. A
  request without `conversation` signs `NONE_OF_IT`, as before, and its
  receipt is the same bytes as before.
- A signed `context` block on those turns, placed after `isolation`, read in
  the docs as "context reused in this session" (`context.rs`,
  `SignedContext`):
  - `scope`: `"session"`;
  - `turn`: 1 on the turn that opens the conversation or answers from token
    0, then one more per continued turn; a turn that comes back as 1 tells
    the caller its conversation is gone: evicted, expired, the TD restarted,
    or its handle spent by another request;
  - `reused_tokens`: the prompt tokens whose keys and values an earlier
    turn of this conversation computed and this one did not;
  - `reused_prefix_commitment_hex`: SHA-256(`x0ne-infer/commit/reused-prefix/v1\0`
    ‖ salt ‖ the reused token ids, u32 little-endian), under the receipt's
    salt;
  - `conversation_commitment_hex`: SHA-256(`x0ne-infer/commit/conversation/v1\0`
    ‖ salt ‖ `0x01` and the handle presented, or `0x00` and 32 zero bytes
    for `"new"` ‖ the handle issued). The handles themselves stay out of
    the receipt: they are bearer secrets. The caller can prove that two
    receipts are consecutive turns of one conversation, and two published
    receipts do not link without their openings.
- **Verifier** (relay `verify_answer`, SDK `check_answer_for`, both through
  `decoding::Asked::check_context`), from what the caller sent:
  - without `conversation`: refuse a receipt that signs
    `kv_cache_shared_between_requests: true`, a `context` block, or an
    answer that carries a handle;
  - with one: refuse a receipt without `context`, with
    `kv_cache_shared_between_requests` absent or `false`, with
    `prefix_cache_shared_between_sessions: true`, a `scope` other than
    `"session"`, turn 0, tokens reused on turn 1 or as many as the
    prompt has, `"new"` answered as a later turn, an answer without a
    handle, or a `conversation_commitment_hex` that does not open to the
    handle sent and the handle received.
  A relay or an SDK from before this change reads none of these fields and
  accepts such receipts: the three ship together. The reused-prefix
  commitment is not checked by either: the caller holds no token ids. It
  binds the TD to the ids it reused, for whoever later holds the salt and
  the tokenizer.

### TD restart

The cache lived in the memory of the TD that computed it, and a new boot
has a new receipt key. Every handle is unknown to it: the next turn is
computed from token 0, under a new handle, `turn: 1`, signed under the new
key. Nothing is written to disk.

### What timing still shows

Accepted on 2026-10-30, and stated in [limitations.md](limitations.md#conversations-kept-between-turns)
and THREAT_MODEL.md:

- **The host** sees when a request arrives and when its answer leaves. A
  turn is held to a Padme number of 100 ms ticks, ten at least
  (THREAT_MODEL.md, the rows on a whole answer and a stream). A continued
  turn lasts as long as its new tokens and its answer, a turn from token 0
  as long as the whole history: the host learns, to within a size, which
  turns continued a resident conversation, and from that when one was
  evicted. It already saw each request's size. Holding a continued turn to
  the time of a cold one would cancel the gain; this design does not.
- **Callers behind one relay.** With one resident conversation, a turn of
  conversation B evicts A, and A's next turn is computed from token 0: its
  caller learns that another request ran between two of its turns
  (activity, not content). No caller can have the TD compare its tokens with
  another conversation's: the lookup is by handle, never by content.
- **The holder of a handle.** `reused_tokens` counts the matching prefix
  exactly. Whoever holds the live handle of a conversation (its caller; the
  relay, on the relay route; anyone it leaks to) can test a guess about that
  conversation's history, once per turn: the handle is spent each time, and
  the caller's next turn then comes back as turn 1.
- **Inside the TD**, the lookup and the comparison run in guest memory,
  which TDXRay-style tracing observes as it observes the tokenizer.

## Decisions (2026-10-30)

1. A TD-drawn, single-use handle per conversation, bound to the route,
   rather than the TLS connection (lost at each reconnection, and shared by
   every caller behind the relay) or a caller-drawn identifier (two callers
   may pick the same value).
2. One resident conversation, evicted by any other request or after 300 s:
   accepts the activity signal between callers of one TD in exchange for no
   added memory. Slots per caller identity are not possible on the sealed
   route, whose callers the TD cannot tell apart.
3. `kv_cache_shared_between_requests: true` on every turn of a
   conversation, with the signed `context` block; the relay and the SDK
   refuse a receipt that hides it. Kept `false`, the flag would say that
   the next request cannot read this one's keys and values, which is false.
4. The host learns which turns continued a conversation; this is written
   down rather than padded away.
5. `reused_tokens` exact, rather than rounded: rounding would blur the
   count for its holder and not for the host, which reads time.

## Tests

Written first and seen failing (31 passed, 10 failed on 2026-10-30, the
code stubbed), then the code:

- `candle_backend.rs`, `tests::conversation`: two conversations with the
  same prompt share nothing (the second computes every position, the
  first one's keys read zeros before it runs; the same for a request
  without a conversation and for a handle shown on the other route); a
  continued turn computes only the positions past the reused prefix and
  answers, greedy, what the same prompt answers from token 0, over four
  turns; a prompt changed at position 0, 7 or 19 reuses exactly that many
  tokens and answers as from token 0; the cache reads zeros after its
  timer, after `end_conversation`, after a cancelled turn and after the
  model is dropped, and a spent handle continues nothing. OLMo refuses a
  conversation by name.
- `qwen3.rs`: clearing and growing the cache leave zeros behind; a turn
  that outgrows the cache grows it to its reservation, not to twice its
  size (red first: 74 positions for 40 reserved); a cache cut
  back to a prefix continues as a prefill from token 0 does (logits within
  the fork's tolerance, the same argmax). `ministral3.rs`: the in-place
  cache reads zeros once cleared.
- `server.rs`: the receipt of each turn, whole and streamed, signs the
  context and verifies; an edit of `reused_tokens` breaks the signature; a
  spent handle and a handle on the other route open a new conversation; the
  answer to a request without a conversation signs neither the flag nor a
  context; a model that keeps no conversation answers 400, and no refusal
  quotes a handle, well formed or not.
- `context.rs`, `local_api.rs` (the relay) and `x0ne-sdk/tests/receipt.rs`:
  a receipt that hides a reuse, invents one, or whose commitment opens to
  other handles is refused.
- `e2e_sdk_tests.rs`: two turns sealed by the SDK, through the carrier, to
  the TD and back; the SDK checks each context, the second turn reuses
  tokens, and no byte the carrier relayed holds the handle.

## Measurements

The time to a one-token answer on a conversation's second turn, on the CPU
(24 threads), Qwen3-0.6B in F32, greedy (`x0ne-infer/examples/session_ttft.rs`,
removed on 2026-11-04).
The first turn is one user message of about N tokens; the second carries it,
a fixed answer and a new question, 29 tokens more. Before: the same second
prompt on the tree without this change, from token 0. After: the second turn
continues the first, whose cache was kept. n = 3 per cell, median (min to
max), on 2026-10-30, before and after run in turn for each size, with other
jobs on the machine (load average 12 to 25). On an H100: the second table.

| Second turn | Before | After | Reused | Turn 1, after |
|---|---|---|---|---|
| 4 102 tokens | 79.4 s (71.4 to 88.0) | 1.85 s (1.56 to 2.36) | 4 073 | 77.6 s (66.7 to 86.3) |
| 8 198 tokens | 356.8 s (307.5 to 382.7) | 4.46 s (4.08 to 4.52) | 8 169 | 336.6 s (324.5 to 336.8) |
| 16 390 tokens | 1 722 s (1 716 to 1 952) | 12.3 s (9.9 to 16.0) | 16 361 | 2 313 s (2 068 to 2 638) |

- The tokens reused are the first turn's prompt less the 4 that close its
  generation prompt: Qwen3's template writes an empty thinking block after
  `assistant`, which a past answer in the history does not carry. The
  first turn's answer is not reused.
- Each second turn outgrows the cache kept by 25 positions: the times
  include the copy into a buffer of the turn's reservation and the zeroing
  of the old one. The 4 102 and 8 198 rows ran on the code before
  `LayerCache::append` sized that buffer to the reservation (it doubled
  it); the 16 390 row ran on the code as committed.
- Zeroing a kept cache: a 13-token request after a kept conversation, which
  zeroes it first, against one after another 13-token request, median of 3:
  0.37 s against 0.28 s at 4 096 tokens, 0.41 s against 0.27 s at 8 192,
  0.75 s against 0.35 s at 16 384 (3.5 GiB of keys and values in F32).
- A rerun of the 4 102 row on 2026-10-31 was stopped at a load average of
  61; its rows are not used.

On an H100 (GCP `a3-highgpu-1g`, Intel TDX, the GPU in CC mode, through the
relay and the attested channel; the run of 2026-10-31 in
[hardware-runs.md](hardware-runs.md)), `ministral3:14b` in BF16, served by
the Engine built from that day's sources. The first turn is one user message
of about N tokens, asked for 32 tokens; the second carries it, the answer and
a new question, 42 tokens more. Without: the same two prompts sent without a
conversation. With: the second turn continues the first. Streamed; the time
is to the first content event on the operator's machine, the relay and the
round trip to us-east5 included. n = 3 per cell, median (min to max). Every
receipt verified by the relay; each continued turn's signed `context` said
`turn` 2 and the tokens reused below, and its receipt
`kv_cache_shared_between_requests` true, false without a conversation. Rows:
[`2026-10-31-h100-tdx-cc/kv-cache.jsonl`](../bench/results/2026-10-31-h100-tdx-cc/kv-cache.jsonl);
`python3 bench/results/summarize.py` recomputes this table.

| Second turn | Without | With | Reused | Turn 1, with |
|---|---|---|---|---|
| 4 590 tokens | 1.114 s (1.112 to 1.114) | 0.321 s (0.310 to 0.329) | 4 579 | 1.119 s (1.117 to 1.128) |
| 8 590 tokens | 3.028 s (3.021 to 3.030) | 0.421 s (0.415 to 0.422) | 8 579 | 2.919 s (2.915 to 2.920) |
| 16 590 tokens | 8.872 s (8.861 to 8.902) | 0.472 s (0.468 to 0.482) | 16 579 | 8.871 s (8.854 to 8.887) |

- The tokens reused are the first turn's prompt (4 548, 8 548 and 16 548
  tokens) plus 31 of the tokens past it; the first turn's answer has 32.
- A first turn that opens a conversation took what the same prompt took
  without one: 1.119 s against 1.122 s, 2.919 against 2.929, 8.871 against
  8.837 (medians, n = 3).
- Each turn asked for 32 tokens and got them (finish `length`); the end of
  the continued turn came at 1.129, 1.426 and 1.682 s, against 1.917, 3.831
  and 10.685 s without a conversation.
