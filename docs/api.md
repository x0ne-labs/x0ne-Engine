# The OpenAI-compatible API

## What the API honours, and what it refuses

`POST /v1/chat/completions` reads `model` (the id
`GET /v1/models` lists, or none; any other gets a 404 `model_not_found`
that quotes nothing of it, and every answer names the id served), `messages` (each
message is `role` and `content`, a string; the content is text: a special
token's string typed in it, such as `<|im_end|>`, is tokenized as text and
never as that token, since 2026-10-26, `docs/models.md`), `max_completion_tokens` or its
deprecated alias `max_tokens` (two different values: 400), `temperature` (0 to
2; 0 or absent decodes greedily, where OpenAI's default is 1), `top_p` (0 to 1),
`seed` (drawn from `/dev/urandom` when sampling without one), `stop` (a string
or up to 4, each at most 256 bytes: the answer ends before the first one met,
`finish_reason: "stop"`), `response_format` (`text` or `json_schema`, below)
and `x0ne` (`conversation` and `end_conversation`,
[below](#x0neconversation-a-conversations-keys-and-values-kept-between-turns)). Greedy decoding
draws nothing, so without a `temperature` above 0 a `top_p` below 1 or any
`seed` gets a 400 that names it. `n` is accepted only at `1`. `stream: true`
answers OpenAI's `chat.completion.chunk` events, the receipt on the final
chunk only, and `stream_options` takes `include_usage` alone, with
`stream: true` only. A stream is padded, without an option to turn it off:
one frame every 100 ms, all of one size per stream (512 to 768 bytes, set by
the id served, the same for every stream of a server), and a
last frame of 16 KiB, success or error. The number of frames, the last
included, is rounded up to a Padme size (PURBs, Nikitin et al., PETS 2019)
of at least 10 frames and at most 8704: the last frame waits for it, after
the text has passed on, so the final chunk comes 0.9 s after the 200 at the
soonest (ten frames 100 ms apart, the first with the 200) and at most
11.6 % later than without the rounding. 8704 frames, 870.4 s, is the last
size within the 15 minutes a relay waits for a chat: a stream whose text is
not out by then ends there with an error event, and its generation is
cancelled. A whole answer is padded too: its body ends in spaces up to a
Padme size of at least 16 KiB, and it leaves the TD once a Padme number of
100 ms ticks has passed since its turn began, at least 10, so never within
one second. An error or a refusal of a whole answer leaves the same way:
`{"error": ...}` padded to the same size, no sooner than one second after
the request, its head of the same length whatever the status (an
`x0ne-pad` header makes up the reason phrase). One answer runs at a time,
in the model's turn, and the turn ends at that same size of ticks after it
began, counted when the generation ends, an error's included. The answer's
queue place comes back only when its turn ends: an answer asked while
another runs starts then, and a full queue's 503s stop then, not when the
model frees up, so neither tells how long the other ran. With
`--max-queue 0`, a request sent as a stream ends may get a 503 until that
stream's turn ends. None of this can be turned off. The aim is that the
VM's host sees neither the size nor the timing of the tokens, and the
length of an answer only to within a size. On the wire between the relay
and a simulated TD (loopback, 2026-10-22), every frame is one TLS record of
one size whatever its text, the gaps between records did not differ with
the text, and the last frame is a full record and a short one whatever the
size of the `x0ne` block. An error after the 200 ends with the same two
records, the chunked end included, and the connection stays open after
both and carries the next request. With the length padding (same bench,
2026-10-23): whole answers of 1 byte to 8 KiB of content, seven lengths
that left as seven record traces of 6194 to 14391 bytes within 3 ms, leave
as one trace 1.00 s after the request, a full record and a 170-byte one,
16581 bytes (2026-10-24, the `x0ne-pad` header included; 16539 without it);
16 KiB of content takes the next size (22725 bytes); a whole answer that
fails and one refused for an adapter leave the same records as one that
succeeds (2026-10-24). Streams of 1, 4 and 16
steps, three traces before, are one trace of nine frames and the final
pair, ending at 0.92 s instead of 0.21 to 0.41 s; a stream of 9.0 s ends
at 9.5 s and costs 4 % more bytes, one of 4.6 s one more frame.

What the host still reads: the frame cadence, one every 100 ms, the same
for every stream, which tells a stream from a whole answer; a stream's
total duration, from its 200 to its last frame, which is its number of
frames, so the answer's length to within a size (one of 117 counts from 10
to 8704); when a stream starts, its 200 coming once the generation starts,
after the turns before it; a whole answer's time and, past 16 KiB, its
size, each to within a size; a whole answer whose generation runs past
870.4 s, which leaves when done, to the tick; a stream refused before its
turn (a field the validator refuses, a model not served here, empty
messages, an adapter, a lease expired), padded as a whole answer's refusal
a second after the request, and one refused in its turn (a prompt past
the context, a lease expired at the model, a failure or a panic before its
first token), padded as one a size of ticks after its turn began: either
tells a refusal from a stream; a full queue or a full set of open streams
(a short 503 at once, its sentence set by the server); a foreign `Host` on
the TD served directly (421) and a panic of a stream before its generation
(500), each a short status at once; the request's
size; and a failure the relay detects (receipt, protocol), since the
session then stops and its connections close. No topic classifier has
been run on what is left.
On the caller-sealed path (`POST /v2/e2e/chat`, [sdk-e2e.md](sdk-e2e.md)),
since 2026-10-30, `stream: true` is sealed as a chunked OHTTP request
(`message/ohttp-chunked-req`, draft-ietf-ohai-chunked-ohttp-08) and
answered as a chunked response with the same `chat.completion.chunk`
events inside, at the cadence the session config sets (`e2e_stream`: L
bytes of plaintext every T ms, 256 and 100 by default, the first chunk T
after the request) and a last chunk of 16 KiB; the number of chunks is a
power of two, 16 at least, refusals included, which leave as a stream of
16 chunks holding the status and the error above. A request sealed whole
must not ask to stream, a chunked one must. Since 2026-10-31 the relay's
local endpoint carries a chunked request and its stream frame by frame, as
the TD writes it, and bounds it by the cadence's bytes, 30 s between frames
and 15 min in all; a cut breaks the body short of its end and is counted
under its code in `/x0ne/v1/status` (`sealed_streams`); a request over the
largest of its kind gets 413 and `sealed_request_too_large` before the TD.
Tool calling (`tools`,
`tool_choice`, `functions`, `function_call`, `parallel_tool_calls`, a `tool`
message or `tool_calls`) gets a 400 saying it is not supported yet. Any other
field (`logprobs`, penalties, `user`…) gets a 400 that names it. An ignored
parameter is an answer to a question the caller did not ask. The receipt
commits to the text returned, and names the sampling, its seed and the
digest of a `json_schema` ([below](#response_format-json-held-to-a-schema)).
The lists live in `HONOURED_FIELDS`, `PINNED_FIELDS` and
`TOOL_FIELDS` (`openai/request.rs`); field names as in openai-python
(`types/chat/completion_create_params.py`).

Every error before a 200 is OpenAI's shape on all three servers (the TD
served directly, the cloud TD and the relay):
`{"error": {"message", "type": "x0ne_engine", "param", "code"}}`, `param`
and `code` a string or null (`openai/error.rs`). On the TD, every
error of `/v1/chat/completions` that names part of the request is padded
as a whole answer's, its status and message kept: a whole answer's, a
stream's before its 200, and a body axum's JSON extractor rejects (400 for
a body that is not valid JSON, its message naming where the JSON broke,
413 past axum's 2 MiB body limit, 415 without a JSON content type). A full
queue, a full set of open streams and a foreign `Host` leave at once,
their sentences set by the server. An error quotes at most 64 bytes of a
field name or JSON path of the request, cut on a character, so that none
grows past 16 KiB. An error after a stream's 200
is one event, `{"error": {"message", "type": "x0ne_engine", "code"}}`,
without `param`.

`GET /v1/models` lists the one model a server serves, as OpenAI's `Model`
objects: `id` is the name its answers give when a request names none (the
TD served directly: the model digest; the relay: the session's model name),
`owned_by` is `"x0ne-engine"`, `created` is when serving began, and `x0ne`
holds `digest_hex` (with `attested: true` from the relay). The TD answers it
at once, unpadded and outside the queue: the list depends on no request.

### `response_format`: JSON held to a schema

`response_format` is read since 2026-10-30, on both servers and on the
sealed path (`/v2/e2e/chat` hands its request to the same handler, so the
same schema is held and the same refusals come back, OpenAI-shaped, inside
the sealed answer). Absent, `null` or `{"type": "text"}` asks nothing.
`{"type": "json_schema", "json_schema": {"name", "schema", "strict",
"description"}}` holds the answer to `schema`: at each step the tokens
whose bytes would make the answer something other than the start of a
document of the schema are masked before the token is chosen, greedy or
sampled, and the end-of-sequence token is offered only once the document
is complete. A document that can take no further byte (its object or array
closed) ends the answer there. `strict` is read and changes nothing: the
schema is always held. `{"type": "json_object"}` gets a 400: it names no
schema to hold. `stop` beside a `json_schema` gets a 400 naming `stop`: a
stop sequence would cut the document.

The schema is compiled (`json_schema.rs`) from this subset of JSON Schema:
`type` (`null`, `boolean`, `integer`, `number`, `string`, `object`,
`array`, or a list of them), `enum`, `const`, `properties`, `required`,
`additionalProperties` (absent or `false`: an object holds its declared
properties only, each at most once, in any order), `items`, `minItems` and
`maxItems`. `$schema`, `$comment`, `title`, `description`, `default` and
`examples` are accepted and change nothing. Any other keyword (`pattern`,
`format`, `minimum`, `minLength`, `anyOf`, `$ref`...) gets a 400 that
names its path in the schema, before the model runs: a keyword the mask
does not hold is refused rather than ignored. Bounds, each a 400 naming
it: 16 KiB of compact schema, 10 levels of nesting, 512 subschemas, 64
properties per object, 256 values per `enum`. The answer takes at most one
space between two tokens of an object or an array, and no other whitespace,
nor any before or after the document: Outlines' default, `[ ]?`, since
its PR 916 of 2024-05-24 (github.com/outlines-dev/outlines, issues 839 and
908: a small model given room repeats its spaces and newlines). Measured
2026-10-30 on Qwen3-0.6B, greedy, four prompts (`x0ne-infer-lab`, CPU):
without the space, two answers were degenerate
(`{"name":", ","skills":["","",""],"born":2000}`), since the model writes
a space after a colon; with it, none. A number has at most 18 digits
before its point and 2 in its exponent, so that every number written
parses (measured 2026-10-30: `504e+657853570652060901548415` is RFC 8259
JSON that serde_json refuses); an `integer` has neither fraction nor
exponent. An answer cut by its token budget is the start of a document,
not a document, with `finish_reason: "length"`.

The mask reads each token's bytes from the model's tokenizer, which works
only when the decoder writes a token's text whatever its neighbours: a
`ByteLevel` decoder (Qwen3's, measured) or `Fuse`. Another decoder gets a
400 naming it, before the model runs. The table of the vocabulary's
bytes is read once per loaded model, at the first request that asks: 79 to
104 ms for Qwen3-0.6B's 151,669 ids (release build, Ryzen 9 7900X under
other builds, 2026-10-30). x0ne Core reads it again for each such request.
A mask then took 144 µs on average and 343 µs at worst over 83 steps on the
same machine, on the CPU, between two tokens. The approach is the one of
llguidance (github.com/guidance-ai/llguidance) and XGrammar (arXiv
2411.15100), simplified: inside a string, a token without a quote, a
backslash or a control byte passes without a check; any other is fed to the
automaton byte by byte, past the first two bytes only when the automaton
takes them. Two effects known from that work hold here: the mask offers
tokenizations of a document that the tokenizer would not produce from its
text (arXiv 2403.06988), and sampling under a mask does not draw from the
distribution the model gives the documents of the schema (arXiv
2405.21047).

The receipt says what was asked of the decoding (since 2026-10-30): an
answer that samples signs `sampling` (`temperature`, `top_p`, `seed`: the
caller's, or the one the server drew, below 2^53 so that a JavaScript
client reads it exactly), and an answer under a schema signs
`json_schema_sha256_hex`, the SHA-256 of the schema serialized with sorted
keys and no whitespace (`json_schema::canonical`). A greedy answer without
a schema signs neither, and its receipt is the same bytes as before. The
server checks the text against the schema again before it signs: a
completion outside it gets a 500, or an error event in place of the final
chunk, and no receipt. The relay and x0ne-sdk (`Pending::open_verified`)
hold each receipt to the request they sent: the same sampling and seed,
the digest of the schema they sent, the text a document of it (or its
start, under `finish_reason: "length"`). Otherwise the relay stops the
session (65) and the SDK refuses the answer. `finish_reason`, `stop` and
`max_tokens` are not signed. A seed replays an answer on the same build
and device only.

### `x0ne.conversation`: a conversation's keys and values kept between turns

Read since 2026-10-30, on the TD's two routes (`/v1/chat/completions` and
the sealed `/v2/e2e/chat`), whole and streamed. Without it, a request is
served as before: its cache is zeroed at its end and the next request
computes its prompt from token 0. With it, the TD keeps the keys and values
of one conversation between its turns, and a turn computes only the part of
its prompt that the previous turn did not: the new message, as a rule.
In this version no model keeps one: x0ne Core, which serves every model of
the catalog, answers a request with `conversation` with the 400 below
([limitations.md](limitations.md#conversations-kept-between-turns)).

- `"x0ne": {"conversation": "new"}` opens a conversation. The answer
  carries `"x0ne": {"conversation": {"handle": "<64 hex digits>"}}`, in a
  whole answer and in a stream's last chunk: 32 bytes the TD drew from
  `/dev/urandom`.
- The next turn sends that handle, `"x0ne": {"conversation": "<handle>"}`,
  and its whole conversation as usual: every message, the previous answers
  included. The TD tokenizes the prompt as any other, compares its token
  ids with the ones it kept, one by one, and reuses the longest common
  prefix, at most the prompt less its last token. Nothing is looked up by
  content: a different first token reuses nothing.
- A handle serves one turn. Each answer carries the next; a handle already
  presented, one the TD no longer holds (evicted, expired, the TD restarted)
  or one issued on the other route opens a new conversation: the turn is
  computed from token 0 and signs `turn: 1`.
- Two requests that present the same handle: the first spends it and
  continues the conversation; the second opens a new conversation, which
  evicts the first's, and signs `turn: 1`.
- `"end_conversation": true` beside the handle (or beside `"new"`) answers
  that turn and zeroes the cache after it. Its answer still carries a
  handle, which names nothing.
- One conversation is resident at a time. Any other request (another
  conversation, or a request without `conversation`) zeroes it before it
  runs; so do 300 seconds without a turn and a turn that fails or is
  cancelled. A process that ends does not zero it: its memory goes back to
  the TD's kernel.
- A malformed handle gets a 400 naming `x0ne.conversation` that quotes
  nothing of it. A model that keeps no conversation answers a 400 naming
  `x0ne.conversation`, rather than an answer whose receipt would sign no
  context.

A turn of a conversation signs `kv_cache_shared_between_requests: true`,
the first turn included (its keys and values stay for the next request),
and a `context` block:

```json
"context": {
  "scope": "session",
  "turn": 2,
  "reused_tokens": 489,
  "reused_prefix_commitment_hex": "…",
  "conversation_commitment_hex": "…"
}
```

`turn` is 1 on a turn computed from token 0, then one more per continued
turn. `reused_tokens` is the exact number of prompt tokens whose keys and
values an earlier turn of this conversation computed.
`reused_prefix_commitment_hex` is SHA-256(`x0ne-infer/commit/reused-prefix/v1\0`
‖ salt ‖ those token ids, u32 little-endian) and
`conversation_commitment_hex` is SHA-256(`x0ne-infer/commit/conversation/v1\0`
‖ salt ‖ `0x01` and the handle sent, or `0x00` and 32 zero bytes for
`"new"` ‖ the handle issued), under the receipt's salt (`opening.salt_hex`).
The handles stay out of the signed bytes: they are bearer secrets. The
relay and x0ne-sdk hold the receipt to the request they sent: no flag and
no `context` without a conversation; with one, a `context` in scope
`session`, the flag `true`, 0 tokens reused on turn 1, `"new"` answered as
turn 1, fewer tokens reused than the prompt has, and a conversation
commitment that opens to the handle sent and the one received. Otherwise
the relay stops the session (65) and the SDK refuses the answer. What the
host still learns from the timing of such turns is in
[limitations.md](limitations.md#conversations-kept-between-turns).

### Open WebUI

Open WebUI needs its Function Calling parameter set to Legacy (Settings >
General > Advanced Parameters, saved as `function_calling: "legacy"`). In its
default mode, Native, it adds its built-in tools to a plain chat as `tools`,
and that chat gets the 400 above: no chat is answered. Tool definitions are
refused rather than ignored, because a model that never saw them would answer
another request than the one the client sent. In Legacy mode Open WebUI sends
no `tools`, and its chats are answered. Tested with Open WebUI 0.12.3 against
the relay of `x0ne-engine cloud` and a simulated TD (loopback, 2026-10-22).
The setting is per user: Open WebUI 0.12.3 reads it from the parameters of
the user's chats (`backend/open_webui/main.py`), and the test set it through
`POST /api/v1/users/user/settings/update`.

## The bill has a ceiling

`--max-completion-tokens` (default 4096) is the most one request can generate.
A client asking for more is clamped, not refused — `max_tokens` means "at
most" everywhere — and an answer that reached the ceiling carries
`finish_reason: "length"`, so nothing is hidden.

Without it, one POST asking for a billion tokens holds the GPU for as long as
it takes. That is not a hypothetical: it is the shape of every unbounded-spend
incident, and the number that costs money should be one the operator chose,
not one the caller typed.
