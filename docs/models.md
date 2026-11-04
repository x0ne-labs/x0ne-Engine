# Models

The engine serves Qwen3, dense, in BF16, on x0ne Core
(`x0ne-infer/src/core_backend.rs`, `x0ne-core/`). Since 2026-11-04 it serves
no other family: OLMo 2, OLMo 3 and Ministral 3, which the engine before
x0ne Core served, left the catalog ([CHANGELOG](../CHANGELOG.md)).

The family is read from `config.json` (`model_type`, else
`architectures[0]`). What x0ne Core does not implement is refused at load,
with a sentence naming it. Refusing is the point: a model built with the
wrong arithmetic answers fluent nonsense, not an error. Refused
(`core_backend.rs`, `core_config`; `x0ne-core/runtime/src/config.rs` and
`weights.rs`):

- a family other than Qwen3 dense;
- a `dtype` other than `bfloat16` in `config.json`, or a tensor stored in
  another type;
- `attention_bias`, `use_sliding_window` or `rope_scaling` set, or a RoPE
  type other than the default;
- an activation other than SiLU;
- a missing or extra tensor, a tensor of another shape, and a tied
  configuration whose `lm_head.weight` differs from the embedding.

Testable with `make core-gpu` (`scripts/core-gpu.sh`) on an NVIDIA GPU: a
conformance suite on Qwen3-0.6B and Qwen3-8B, the server's answers equal to
x0ne Core's own greedy decoding on five prompts, and compute-sanitizer over
the Qwen3-0.6B suite; no run of it is recorded in this repository. x0ne
Core's own tests, not this target, passed on an H100 in CC mode on
2026-11-03 ([hardware-runs.md](hardware-runs.md)).

**"Fully open"** here means weights, training code AND training data are
published under open licences: stricter than the OSI Open Source AI
Definition 1.0 (which accepts a description of unshareable data), close to
Class I of the Model Openness Framework. Qwen3 is open-weights (Apache-2.0
weights; training data and code not published), not fully open.

The level is looked up by the served digest (v2: config, tokenizer and every
weight file, shards with their index) in a table in
`x0ne-infer/src/openness.rs` — never by name, which any config can claim. A
digest not in the table is `unknown`, which means unchecked, not closed.

The chat template is Qwen3's, with an empty `<think>` block. Each marker it
writes as a control token (`<|im_start|>`, `<|im_end|>`) must be a special
added token of the tokenizer, or the model is refused at load (since
2026-10-28): as a plain word, a message could type it and write a turn. The
tokenizers the catalog pins pass (`tests/control_tokens.rs`).

Only the template's own markers are control tokens. A message is tokenized as
text: `<|im_end|>`, `<|im_start|>` or `<|endoftext|>` typed in it stays text
and opens no turn (`x0ne-infer/src/prompt.rs`, since 2026-10-26; before, the
whole rendered prompt was tokenized and a message could write a system turn).
A prompt whose messages hold no special token's string has the same token ids
as before.
