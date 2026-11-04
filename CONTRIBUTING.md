# Contributing

This is a small, security-critical code base: a server that refuses to serve
unless it can prove where it runs. Contributions are welcome; the bar is that
a change leaves that refusal at least as strong as it found it.

## Before you open a pull request

```bash
cargo fmt -- --check
cargo test --locked --workspace --all-targets
cargo test --locked --workspace --all-targets --features x0ne-infer/nvml
cargo clippy --locked --workspace --all-targets --features x0ne-infer/nvml -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps
```

`--locked` everywhere: a change to `Cargo.lock` is a change to review, not a
side effect. If you touch a dependency, also run `cargo deny --locked
--workspace check` (policy in `deny.toml`) and `scripts/third-party-licenses.sh`
(regenerates `THIRD_PARTY_LICENSES.html`; CI fails on a stale one). The
compiler is pinned in `rust-toolchain.toml`; `tests/toolchain_pins.rs` holds
the Containerfile, CI and `rust-version` to it.

CI runs these (`.github/workflows/ci.yml`, jobs `fmt`, `clippy`, `test`,
`doc`, `msrv`, `stable`, `ci-scripts`). Its `clippy` job adds
`clippy::pedantic`, minus the lints its `PEDANTIC` variable allows (`make ci`
reads the same list), and also lints every feature but `cuda` at once.

A new model family comes with reference fixtures from Hugging Face transformers
(`x0ne-infer/scripts/make-arch-fixtures.py`, see `docs/learn-manual-test.md`):
the decoder is held to the reference implementation, not to itself.

Do **not** use `--all-features`: it enables `cuda`, whose build script needs
the CUDA toolkit. CI does not compile the `cuda` path; it is only built on
machines that have the toolkit.

Coverage is held per crate. `make coverage` (see `docs/ci-local.md`) measures
it in a container with the toolchain's own `llvm-tools`, and is red when a
crate falls more than 0.2 point under `ci/coverage-baseline.json` in lines or
regions. Code comes with its tests; `make coverage-update` then raises the
baseline, committed with the change. Lowering it
(`CI_COVERAGE_ALLOW_DROP=1 make coverage-update`) says why in the commit
message.

## What we ask of a change

- **Fail closed.** No evidence, an unreadable answer, a missing claim, an
  unknown value: refuse, with a distinct exit code and a sentence that says
  why. Never a default that serves.
- **Measured, not recalled.** A pin, a wire format, a claim name, a price:
  say where it was read and when. If it could not be checked against the real
  thing, say so in `LIMITATIONS` rather than leaving it implied.
- **A security test says what its failure would mean**, in its doc comment,
  and its name is a sentence stating the property. Check that it dies when
  the line it protects is removed.
- **A fixture never derives its size or shape from a constant the code under
  test also reads.** Pin the value, assert the constant equals it.
- Mark with `SECURITY:` any line whose modification would open a hole.
- Comments say *why*, with the evidence. A comment that restates the next
  line is noise.

## Reporting a vulnerability

Not in a public issue — see `SECURITY.md`.

## License and contributions

External pull requests are accepted once the Contributor License Agreement
is signed. It is not published here yet: until it is, no external pull
request is merged. Issues are welcome.

The code is under AGPL-3.0-only, and under Apache-2.0 for the crates an
application builds in: `LICENSING.md` maps them.
