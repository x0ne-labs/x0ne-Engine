#!/usr/bin/env bash
# Regenerates THIRD_PARTY_LICENSES.html: the license text of every crate, all
# features, from Cargo.lock. Needs cargo-about 0.9
# (`cargo install --locked --features cli cargo-about@0.9.2`). Deterministic:
# same lock, same file (two runs compared on 2026-10-05).
set -euo pipefail
cd "$(dirname "$0")/.."
cargo about generate --locked --workspace --all-features -o THIRD_PARTY_LICENSES.html about.hbs
