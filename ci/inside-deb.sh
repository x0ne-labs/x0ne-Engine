#!/usr/bin/env bash
# Release build for the .deb, inside the CI image (ci/ci.sh deb), no network.
# Mounts: /src (the commit), /vendor (crates), /target, /out (bin/, debug/). Environment:
# SOURCE_DATE_EPOCH (commit date), X0NE_GIT_SHA, DEB_VERSION. The client side of the
# attested path only: x0ne-engine, x0ne-engine-cloud, x0ne-engine-proxy (package x0ne-engine,
# ci/deb/x0ne-engine.ctl), plus the .version and .debversion files x0ne-OS build.sh reads.
# x0ne-infer, the TD's server, ships in the CVM image only (x0ne-OS scripts/pkg-infer-cuda.sh,
# features cuda nvml): no .deb since 2026-11-04. tests/cloud_source_sweep.rs reads this list.
# Profile `release` (Cargo.toml, the only recipe: ci/release-env.sh refuses CARGO_PROFILE_* and
# RUSTFLAGS), then ci/strip-release.sh: bin/ holds the stripped binaries, the ones
# SHA256SUMS.engine names and the .deb ships; debug/.build-id/ their debug files.
# X0NE_DEB_SRC and X0NE_DEB_OUT replace /src and /out (ci/comment-invariance.sh).
set -euo pipefail
bash "$(dirname "$0")/release-env.sh"
export HOME=/tmp/home CARGO_HOME=/tmp/cargo-home CARGO_TARGET_DIR=/target CARGO_INCREMENTAL=0
export XDG_CONFIG_HOME=$HOME/.config XDG_DATA_HOME=$HOME/.local/share XDG_CACHE_HOME=$HOME/.cache
# Fixed mount paths (/src, /vendor, /target, /tmp/cargo-home) stand in for
# --remap-path-prefix: RUSTFLAGS would refingerprint the shared /target.
SRC=${X0NE_DEB_SRC:-/src} OUT=${X0NE_DEB_OUT:-/out}
BIN=$OUT/bin
mkdir -p "$HOME" "$CARGO_HOME" "$BIN"
cp /vendor/config.toml "$CARGO_HOME/config.toml"
if [ "${DEB_SCCACHE:-1}" = 1 ] && [ -x /usr/local/bin/sccache ] && [ -d /sccache ]; then
  export RUSTC_WRAPPER=/usr/local/bin/sccache SCCACHE_DIR=/sccache SCCACHE_CACHE_SIZE=${SCCACHE_CACHE_SIZE:-107374182400}
  sccache --start-server >/dev/null 2>&1 || true
fi
strip_release=$(cd "$(dirname "$0")" && pwd)/strip-release.sh
cd "$SRC"
ver=$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' Cargo.toml | head -1)
[ -n "$ver" ] || { echo "inside-deb: no workspace version in Cargo.toml" >&2; exit 2; }
echo "inside-deb: x0ne-Engine $ver, commit $X0NE_GIT_SHA, deb $DEB_VERSION, epoch $SOURCE_DATE_EPOCH, $(cargo --version), jobs ${CARGO_BUILD_JOBS:-default}, rustc wrapper ${RUSTC_WRAPPER:-none}"
cargo build --release --locked --offline -p x0ne-infer --bin x0ne-engine --bin x0ne-engine-proxy -p x0ne-engine-cloud --bin x0ne-engine-cloud
for b in x0ne-engine x0ne-engine-cloud x0ne-engine-proxy; do
  install -m 0755 "$CARGO_TARGET_DIR/release/$b" "$BIN/$b"
done
bash "$strip_release" "$OUT/debug" "$BIN"/x0ne-engine "$BIN"/x0ne-engine-cloud "$BIN"/x0ne-engine-proxy
printf 'x0ne-Engine %s, commit %s\n' "$ver" "$X0NE_GIT_SHA" > "$BIN/x0ne-engine.version"
printf '%s\n' "$DEB_VERSION" > "$BIN/x0ne-engine.debversion"
touch -d "@$SOURCE_DATE_EPOCH" "$BIN"/*
cd "$BIN" && sha256sum x0ne-engine x0ne-engine-cloud x0ne-engine-proxy > SHA256SUMS.engine && cat SHA256SUMS.engine
./x0ne-engine --version
[ -n "${RUSTC_WRAPPER:-}" ] && sccache --show-stats > "$OUT/sccache.log" 2>&1 || true
