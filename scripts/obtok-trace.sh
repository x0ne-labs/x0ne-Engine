#!/usr/bin/env bash
# scripts/obtok-trace.sh [rev]: the oblivious tokenizer under valgrind lackey, out/obtok-trace/<sha>/.
# Exit 0 green, 1 red, 2 setup; x0ne-infer/docs/oblivious-tokenizer.md, "Measured on the real binary".
# Settings as ci/ci.sh: CI_CACHE CI_LOCK CI_WAIT_MAX CI_CGROUP_PARENT CI_MEMORY CI_JOBS, or CI_ENV_FILE.
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
log() { printf 'obtok-trace: %s\n' "$*" >&2; }
die() { log "$*"; exit 2; }

CI_ENV_FILE=${CI_ENV_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/x0ne-engine/ci.env}
if [ -f "$CI_ENV_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in ''|'#'*) continue ;; esac
    key=${line%%=*}
    case $key in CI_CACHE|CI_LOCK|CI_WAIT_MAX|CI_CGROUP_PARENT|CI_MEMORY|CI_JOBS) ;; *) continue ;; esac
    [ -n "${!key+x}" ] || printf -v "$key" '%s' "${line#*=}"
  done < "$CI_ENV_FILE"
fi
CI_CACHE=${CI_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/x0ne-engine-ci}
LOCK=${CI_LOCK:-$CI_CACHE/build.lock}
CI_WAIT_MAX=${CI_WAIT_MAX:-3600}
CI_MEMORY=${CI_MEMORY:-16g}
CI_JOBS=${CI_JOBS:-$(nproc)}
RUN="podman run --rm --network=none${CI_CGROUP_PARENT:+ --cgroup-parent=$CI_CGROUP_PARENT}"

# Classes of 1, 2, 4, 6 and 8 bytes, each mixing pieces whose merges differ (abcd
# merges twice, dcba never, efgh three times), one- and two-byte chars (U+00E9, $E),
# and z, absent from the vocabulary. A piece repeated in its class is the same-input
# witness. $E is written as its UTF-8 bytes so that this file stays ASCII.
E=$'\xc3\xa9'
PIECES=(abcd dcba efgh "${E}ab" abcd abcdef fedcba "${E}abcd" "$E${E}ab" abcdef a z h "$E" ab ba
  zz "$E" abcdefgh hgfedcba "$E$E$E$E" abzzcdzz)

sha=$(git -C "$REPO" rev-parse --verify "${1:-HEAD}^{commit}") || die "no commit ${1:-HEAD}"
IMG=localhost/x0ne-engine-ci:$(sha256sum "$REPO/ci/Containerfile" | cut -c1-12)
VG_IMG=localhost/x0ne-engine-valgrind:$(sha256sum "$REPO/ci/Containerfile.valgrind" | cut -c1-12)
podman image exists "$IMG" || die "no CI image $IMG: make ci-image"
if ! podman image exists "$VG_IMG"; then
  log "building $VG_IMG"
  podman build --pull=never -f "$REPO/ci/Containerfile.valgrind" -t "$VG_IMG" "$REPO/ci" >&2
fi

SRC=$CI_CACHE/obtok-trace/src/$sha
TARGET=$CI_CACHE/obtok-trace/target
OUT=$REPO/out/obtok-trace/$sha
rm -rf "$SRC" "$OUT"; mkdir -p "$SRC" "$TARGET" "$OUT"
git -C "$REPO" archive --format=tar "$sha" | tar -x -m -C "$SRC"
VENDOR=$CI_CACHE/vendor/$(sha256sum "$SRC/Cargo.lock" | cut -c1-16)
[ -s "$VENDOR/config.toml" ] || die "no vendored crates for this Cargo.lock in $VENDOR: run make ci once"

log "release build of obtok_trace at ${sha:0:7} (lock $LOCK)"
flock -w "$CI_WAIT_MAX" "$LOCK" nice -n 10 $RUN --memory="$CI_MEMORY" --pids-limit=4096 --userns=keep-id \
  --name "engine-obtok-build-$$" \
  -v "$SRC":/src:ro -v "$VENDOR":/vendor:ro -v "$TARGET":/target -v "$OUT":/out \
  -e CARGO_BUILD_JOBS="$CI_JOBS" -w /src "$IMG" bash -c '
    set -e
    export HOME=/tmp/home CARGO_HOME=/tmp/cargo-home CARGO_TARGET_DIR=/target
    mkdir -p "$HOME" "$CARGO_HOME"; cp /vendor/config.toml "$CARGO_HOME/config.toml"
    cargo build --locked --offline --release -p x0ne-infer --example obtok_trace
    cp /target/release/examples/obtok_trace /out/obtok_trace' > "$OUT/build.log" 2>&1 \
  || { tail -n 30 "$OUT/build.log" >&2; die "build failed, $OUT/build.log"; }

rc=0
for run in obl:same obl-vec:report hash:differ; do
  mode=${run%%:*} expect=${run#*:}
  $RUN --memory=4g --name "engine-obtok-$mode-$$" -v "$OUT":/out "$VG_IMG" bash -c '
    valgrind --tool=lackey --trace-mem=yes --log-file=/out/lackey-$1.txt /out/obtok_trace run "$@" > /out/run-$1.txt
    /out/obtok_trace compare /out/lackey-$1.txt /out/run-$1.txt '"$expect" -- "$mode" "${PIECES[@]}" \
    > "$OUT/compare-$mode.txt" || { r=$?; if [ "$r" -gt "$rc" ]; then rc=$r; fi; }
  log "$mode (expected $expect): $(tail -n 1 "$OUT/compare-$mode.txt")"
done
log "$([ "$rc" = 0 ] && echo green || echo red) ${sha:0:7}; $OUT"
exit "$rc"
