# Open models — manual test procedure

No GPU is needed; every step runs in a throwaway container. Each step says
what a pass and a failure mean.

Paths below: `$ENGINE` is this repository, `$MODELS` a directory for model
files (the author used `~/.cache/x0ne-build/models`), `$CACHE` a cargo cache.

## 0. Unit and fixture tests (any machine)

```sh
cargo test -p x0ne-infer --all-targets
cargo clippy -p x0ne-infer --all-targets -- -D warnings
```

Pass: all green. What they hold:

- `ministral3::tests`: Ministral 3 against transformers on tiny random models,
  one with the 14B's real RoPE on 16448 positions (Llama 4's query scale).
- `decoder::tests`: the decoder against Hugging Face transformers on tiny random
  Qwen3, OLMo 2 and OLMo 3 models (`x0ne-infer/tests/arch/`): logits, every
  attention map, every layer's output, greedy generation through the KV cache
  past the sliding window, loss, per-layer gradient norms and input saliency.
- `candle_backend::tests::qwen3_generation_is_exactly_candles_greedy_loop`:
  a Qwen3 model run by the fork `x0ne-infer/src/qwen3.rs` (the CPU reference of
  the tests, feature `candle-reference`) answers token for token what candle's
  own implementation produces in the v0 loop.
- `arch::tests`: unknown, missing and contradictory architectures are refused;
  what is not implemented (`clip_qkv`, other RoPE scalings, `mscale`) is refused.

## 1. Regenerate the reference fixtures (optional)

Only needed if the fixtures or the families change. Needs network once.

```sh
podman run --rm -v $CACHE/pyenv:/venv docker.io/library/python:3.12-slim sh -c \
  'python -m venv /venv && /venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch \
   && /venv/bin/pip install -q "transformers>=4.57" safetensors'
podman run --rm --network=none -v $CACHE/pyenv:/venv \
  -v $ENGINE/x0ne-infer/scripts:/s:ro -v $ENGINE/x0ne-infer/tests/arch:/out \
  -v $MODELS:/m:ro docker.io/library/python:3.12-slim /venv/bin/python \
  /s/make-arch-fixtures.py /out --olmo2-dir /m/OLMo-2-0425-1B-Instruct --olmo3-dir /m/Olmo-3-7B-Instruct \
  --ministral3-14b-dir /m/Ministral-3-14B-Instruct-2512-BF16 --ministral3-8b-dir /m/Ministral-3-8B-Instruct-2512-BF16
```

The `--olmo*-dir` and `--ministral3-*-dir` directories need only the
release's small files (`config.json`, the tokenizer files,
`chat_template.jinja`), no weights. They produce `chat_templates.json`, the
reference for the OLMo and Ministral 3 chat templates, and
`ministral3/rope.json`, the real releases' RoPE. `--families
ministral3,ministral3-long` regenerates only those. The Ministral 3 fixtures
were made with transformers 5.19.0 and torch 2.14.1 on the CPU (2026-10-28),
from the config, tokenizer and template files of the pinned commits.

`ministral3_mistral_common.json` holds the ids mistral_common gives 307
conversations (`scripts/make-mistral-common-ids.py`, run offline with
mistral_common 1.13.0, wheel sha256 `6304455b…611993d`, and the release's
`tekken.json`, sha256 `600bb279…312843`, equal for the 14B and the 8B). The
test that reads it needs the real tokenizer.json:

```sh
X0NE_INFER_TOKENIZER_MINISTRAL3=/m/Ministral-3-14B-Instruct-2512-BF16/tokenizer.json \
  cargo test --release -p x0ne-infer --test control_tokens -- --ignored --nocapture ministral3
```

## 2. Ministral 3 with its real weights, CPU

Not in CI: about 30 GB of RAM per side, and the weights. Never run the two
sides at once on a 14B.

```sh
x0ne-engine pull ministral3:14b    # or ministral3:8b; every file checked against the catalog
# transformers side: the release's small files (config.json, tokenizer.json,
# tokenizer_config.json, chat_template.jinja) in $TMPL, offline
python -I x0ne-infer/scripts/ministral3-real-reference.py $MODEL $TMPL ref-bf16.safetensors \
  --dtype bf16 --text-only --then-f16 ref-f16.safetensors --greedy 32 [--long 16448]
# engine side, same F16
X0NE_INFER_MINISTRAL3_DIR=$MODEL X0NE_INFER_MINISTRAL3_REFERENCE=ref-f16.safetensors \
X0NE_INFER_MINISTRAL3_DTYPE=f16 \
  cargo test --release -p x0ne-infer --lib real_weights -- --ignored --nocapture
```

The test checks that the pulled files give the catalog's served digest, that
the engine renders and tokenizes the reference's prompts to the same ids
(with the release's default system message), then compares the logits of the
last 64 positions and 32 greedy tokens. Pass: every greedy token equal and
the same top token at 95 % of the positions or more. Compare with a
reference in the engine's dtype: F16 against transformers' BF16 is up to
dtype noise. `X0NE_INFER_MINISTRAL3_EXTRA=<chats.jsonl>:<rows.jsonl>` also
checks the ids of the script's `--extra` rows.

Results, 2026-10-28, transformers 5.19.0, torch 2.14.1, 12 threads, F16:

| | 14B | 8B |
|---|---|---|
| served digest | the catalog's pin | the catalog's pin |
| prompt ids (3 chats, 1 raw text) | equal | equal |
| top token, last 120 positions | 120/120 | 120/120 |
| greedy tokens | 32/32 on each of 4 | 32/32 on each of 4 |
| max logit difference (max logit) | 0.60 (27) | 0.26 (24) |
| transformers BF16 vs F16, same prompts | 5.0, top token 118/120 | 0.57, top token 118/120 |
| peak memory (VmHWM) | 32.3 GB | 29.0-29.4 GB |

Past 16384 positions, on the 8B with `--long 16448`, only transformers
finished: BF16 and F16 give the same 32 greedy tokens and the same top token
at the last 64 positions (logits within 1.94). The engine's F16 run of that
prompt was stopped after 90 minutes (about 15 CPU hours): long context with
real weights is not compared; on tiny random models it is (section 0).
Two transformers runs in F16 of the same 8B differ by up to 0.16. F16
is the dtype of neither server: the CPU server loads F32, which for the 8B
needed more than the 32 GiB of the machine's build slice; the GPU loads BF16,
not compared here.
