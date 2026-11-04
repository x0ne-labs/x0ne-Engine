#!/usr/bin/env bash
# SBOM and auditable build for a release.
#
#   scripts/sbom.sh [out dir]            default: ./sbom
#
# Writes one CycloneDX 1.5 JSON per workspace crate (cargo-cyclonedx) and
# builds the client side of the attested path, the binaries of the .deb
# (`x0ne-engine`, `x0ne-engine-cloud`, `x0ne-engine-proxy`) with `cargo auditable`, which
# embeds the dependency list in the binary (`cargo audit bin <file>` reads
# it back). Needs: cargo-cyclonedx 0.5, cargo-auditable 0.7, binutils.
# Profile `release` and ci/strip-release.sh, as the .deb (ci/inside-deb.sh): the binaries
# are stripped, SHA256SUMS names the stripped ones, their debug files go to <out dir>/debug.
# ci/release-env.sh refuses CARGO_PROFILE_* and RUSTFLAGS in the environment first.
#
# Features are those of the .deb (none). `x0ne-infer`, the TD's server, is not
# built here: it serves through x0ne Core (`cuda`, whose build script needs nvcc)
# and ships in the CVM image only (x0ne-OS scripts/pkg-infer-cuda.sh, 2026-11-04).
set -euo pipefail
OUT=$(realpath -m "${1:-./sbom}")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
bash "$ROOT/ci/release-env.sh"
mkdir -p "$OUT"

for tool in cyclonedx auditable; do
  cargo "$tool" --version >/dev/null 2>&1 || { echo "cargo-$tool is not installed" >&2; exit 2; }
done

# One SBOM per crate, all dependencies, host target, spec 1.5.
cargo cyclonedx --format json --spec-version 1.5 --all \
  --manifest-path x0ne-infer/Cargo.toml -q
cargo cyclonedx --format json --spec-version 1.5 --all \
  --manifest-path x0ne-confidential/Cargo.toml -q
cargo cyclonedx --format json --spec-version 1.5 --all \
  --manifest-path x0ne-engine-cloud/Cargo.toml -q
mv x0ne-infer/x0ne-infer.cdx.json x0ne-confidential/x0ne-confidential.cdx.json \
  x0ne-engine-cloud/x0ne-engine-cloud.cdx.json "$OUT/"

# Release binaries with the embedded dependency list: the client side of the
# attested path and nothing that reaches SSH (x0ne-infer/tests/cloud_source_sweep.rs).
cargo auditable build --release --locked -p x0ne-infer -p x0ne-engine-cloud \
  --bin x0ne-engine --bin x0ne-engine-proxy --bin x0ne-engine-cloud
target_dir=${CARGO_TARGET_DIR:-target}
for bin in x0ne-engine x0ne-engine-cloud x0ne-engine-proxy; do
  cp "$target_dir/release/$bin" "$OUT/"
done
bash "$ROOT/ci/strip-release.sh" "$OUT/debug" "$OUT/x0ne-engine" "$OUT/x0ne-engine-cloud" "$OUT/x0ne-engine-proxy"

(cd "$OUT" && sha256sum x0ne-engine x0ne-engine-cloud x0ne-engine-proxy ./*.cdx.json > SHA256SUMS)
echo "wrote:"; ls -la "$OUT"
