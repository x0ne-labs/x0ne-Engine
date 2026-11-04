#!/usr/bin/env bash
# Refuses a build environment that changes what the release profile compiles:
# any CARGO_PROFILE_* (CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true lifts the three release guards:
# x0ne-infer measurements.rs and archive_key.rs, x0ne-engine-cloud http.rs), RUSTFLAGS, and any
# CARGO_*RUSTFLAGS (CARGO_ENCODED_RUSTFLAGS, CARGO_BUILD_RUSTFLAGS, CARGO_TARGET_<triple>_RUSTFLAGS).
# Run as a child process by ci/inside-deb.sh, scripts/sbom.sh and the x0ne-OS packaging,
# before cargo: it sees the environment cargo would see. Names the variables, never
# their values. Exit 0 clean, 2 refused. Tested by ci/tests/release-guards.sh.
set -euo pipefail
bad=
# NUL-separated: a newline inside a value cannot pass for a variable.
while IFS= read -r -d '' kv; do
  name=${kv%%=*}
  case $name in
    CARGO_PROFILE_*|RUSTFLAGS|CARGO_*RUSTFLAGS) bad="$bad $name" ;;
  esac
done < <(env -0)
if [ -n "$bad" ]; then
  printf 'release-env: refused: the environment sets%s, which would change the shipped binaries (Cargo.toml [profile.release] is the only recipe). Unset it.\n' "$bad" >&2
  exit 2
fi
