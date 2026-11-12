#!/usr/bin/env bash
# Local CI of one commit in a throwaway podman container with no network.
# Usage: ci/ci.sh [run|image|refresh|status|deb|deb-repro|deb-check|deb-status|run-fixtures]
#        (make ci, ci-image, ci-refresh, ci-status, deb, deb-repro, deb-check, deb-status,
#        run-fixtures)
#
# run     export HEAD (git archive; CI_WORKTREE=1: the working tree's tracked and
#         untracked files), vendor the crates and fetch the RustSec db when stale
#         (the only networked steps, in their own container), take the build lock,
#         copy the tree into $CI_CACHE/v2/src by content (sync_src), then run
#         ci/inside.sh on it and $CI_CACHE/v2/target in --network=none, under
#         CI_CGROUP_PARENT when set. Writes out/ci/<sha>.json (<sha>+dirty.json for a worktree
#         with changes), logs the verdict to the journal, exits 0 (green) 1 (red) 2 (setup).
#         CI_VERDICT_NAME=<name> (lowercase letters, digits, dashes) names a run that is not
#         make ci: its verdict is out/ci/<sha>.<name>.json and latest.<name>.json, never the
#         <sha>.json and latest.json of make ci, which make deb and ci/ci.sh status read.
#         CI_STEPS naming only steps that build nothing (fmt ci-sh features deny gitleaks)
#         takes no build lock and mounts no target: the staged tree is the source, removed after.
#         A full run that is green leaves its verdict in $CI_CACHE/verdicts/<tree>.json; a later
#         full run of the same tree (the same files: a commit of what CI_WORKTREE=1 tested, a
#         merge that changed nothing) under the same image, within CI_REUSE_MAX_AGE, takes that
#         verdict without the lock or a container when CI_REUSE=1: "reused_from" names the
#         run, each step carries "reused": true. Never with CI_STEPS, CI_RANGE or
#         CI_ALL_FEATURES. Off by default: every make ci runs.
# image   build the CI image (ci/Containerfile); tagged by the file's hash.
# refresh force the vendor and advisory-db refresh.
# status  print the last result (CI_VERDICT_NAME=<name>: the last of that name).
# deb     release build of HEAD in the CI image (ci/inside-deb.sh), in a new and empty
#         target directory under DEB_TARGET, removed after, without sccache: nothing make
#         ci or run-fixtures left in their caches reaches the package. Then the
#         x0ne-engine .deb through the packaging of x0ne-OS
#         (X0NE_OS/scripts/build.sh, read-only, CHANNEL=edge). Version
#         <crate>+git<commit date>.<sha7>, SOURCE_DATE_EPOCH = commit date. The binaries
#         are stripped by ci/strip-release.sh; every one the .deb files ship must have the
#         sha256 SHA256SUMS.engine gives it (deb_verify), else red. No .deb
#         ships x0ne-infer: the CVM image does (x0ne-OS scripts/pkg-infer-cuda.sh, 2026-11-04).
#         Staged in DEB_STAGING/<sha>/ (.deb, SHA256SUMS, SHA256SUMS.engine, debug/,
#         deb.json); not published.
# deb-repro  deb, then a second build in a new container, its own new target
#         directory and no sccache: the .deb must match byte for byte, else diffoscope.
# deb-check  dpkg -i of the staged x0ne-engine .deb (the client, without x0ne-infer) in a
#         trixie container, no network; the --version of each of its three binaries must name
#         the commit.
# deb-status  print the last deb verdict.
# run-fixtures  the fixtures of one real boot from its run (examples/run_fixtures.rs):
#         RUN_ATTESTATION (the relay's GET /x0ne/v1/attestation, saved during the run),
#         RUN_REFERENCE (the image's v2 reference), RUN_DATE (YYYY-MM-DD), RUN_IMAGE
#         (image.name of the measurements file), RUN_MACHINE (a3-highgpu-1g). Writes
#         out/run-fixtures/<date>/ and compares it with the working tree: 0 identical,
#         1 different or refused. RUN_WRITE=1 copies a different set into the tree, but
#         replaces no file of it (measurements of a run, fixtures of a RUN_DATE already
#         taken) without RUN_OVERWRITE=1; a measurements file naming no image and no
#         register is a placeholder, filled. RUN_REFERENCE.sig is mounted when
#         present; RUN_TEST_KEY=1 accepts a reference the release archive key did not
#         sign, a TEST-key run, and the example says so.
#
# Environment: CI_CACHE (default ${XDG_CACHE_HOME:-~/.cache}/x0ne-engine-ci), CI_JOBS (nproc),
# CI_MIN_DISK_GB (150, free on / and on the filesystems of CI_CACHE, DEB_TARGET
# and DEB_STAGING), CI_WAIT_MAX (3600 s for the build lock), CI_STEPS (see ci/inside.sh),
# CI_RANGE (<from>..<to>, the merged commits scanned by gitleaks),
# CI_WORKTREE=1 (test the working tree instead of HEAD), CI_VERDICT_NAME (see run),
# CI_REUSE (1: a tree already green takes its verdict, see run; default 0: always run),
# CI_REUSE_MAX_AGE (86400 s, the age past which the advisory database is fetched again),
# CI_LOCK ($CI_CACHE/build.lock), CI_TARGET_MAX_GB (40: past it, v2/target
# is emptied), CI_DEBUGINFO (debug info of the dev and test profiles, line-tables-only),
# CI_MEMORY (--memory of every container: build, vendoring, advisory, packaging, diffoscope,
# install check; 16g), CI_LANES (0: the steps of
# ci/inside.sh one after the other instead of three lanes at once),
# CI_CGROUP_PARENT (cgroup of every container, none by default), CI_NOTIFY_CMD (run after
# each verdict as: <cmd> ci|deb <green|red> <short sha> <seconds> <detail>; unset: none),
# SCCACHE_BIN (sccache on PATH), SCCACHE_DISK ($CI_CACHE/sccache; sccache is used when both exist).
# deb: X0NE_OS (required: an x0ne OS checkout), DEB_STAGING ($CI_CACHE/staging),
# DEB_TARGET (where each release build makes its own new target directory, removed after;
# default $CI_CACHE/deb-target), DEB_SCCACHE=1 (the build shares the sccache of make ci:
# faster, proves less), DEB_REPRO_SCCACHE=1 (the same for the second build of deb-repro).
# CI_ENV_FILE (default ${XDG_CONFIG_HOME:-~/.config}/x0ne-engine/ci.env): KEY=value lines
# for the settings above that the environment leaves unset, one host's layout; read, never run.
set -euo pipefail
cmd=${1:-run}
REPO=$(git rev-parse --show-toplevel)
CI_DIR=$REPO/ci

log() { printf 'ci: %s\n' "$*" >&2; }
die() { log "$*"; exit 2; }

CI_ENV_FILE=${CI_ENV_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/x0ne-engine/ci.env}
if [ -f "$CI_ENV_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in ''|'#'*) continue ;; esac
    key=${line%%=*}
    [ "$key" != "$line" ] || die "$CI_ENV_FILE: not KEY=value: $key"
    if ! [[ $key =~ ^(CI_[A-Z0-9_]+|DEB_[A-Z0-9_]+|SCCACHE_BIN|SCCACHE_DISK|X0NE_OS)$ ]] || [ "$key" = CI_ENV_FILE ]; then
      die "$CI_ENV_FILE: $key is not a CI setting"
    fi
    [ -n "${!key+x}" ] || printf -v "$key" '%s' "${line#*=}"
  done < "$CI_ENV_FILE"
fi

CI_CACHE=${CI_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/x0ne-engine-ci}
CI_JOBS=${CI_JOBS:-$(nproc)}
CI_MIN_DISK_GB=${CI_MIN_DISK_GB:-150}
CI_WAIT_MAX=${CI_WAIT_MAX:-3600}
# Fixed, not derived from the checkout: three runs from a clone locked their own (2026-10-18).
LOCK=${CI_LOCK:-$CI_CACHE/build.lock}
# make ci builds only from CI_SRC into CI_TARGET, both touched under the lock (sync_src).
CI_SRC=$CI_CACHE/v2/src
CI_TARGET=$CI_CACHE/v2/target
CI_TARGET_MAX_GB=${CI_TARGET_MAX_GB:-40}
# Full debug info made each test binary ~390 MB, 11.8 GB per build (2026-10-18).
CI_DEBUGINFO=${CI_DEBUGINFO:-line-tables-only}
# Peak of one run (cgroup memory.peak, 2026-10-22): 9.1 GiB with the lanes, 7.2 GiB without.
# A cgroup parent caps the sum of the builds of a host, not one run.
CI_MEMORY=${CI_MEMORY:-16g}
CI_LANES=${CI_LANES:-1}
# CI_CGROUP_PARENT puts every container in one cgroup: one memory ceiling for all builds of a host.
PODMAN_RUN="podman run --rm${CI_CGROUP_PARENT:+ --cgroup-parent=$CI_CGROUP_PARENT}"
# The build containers (inside.sh, inside-deb.sh): the ceiling of one run.
BUILD_RUN="$PODMAN_RUN --memory=$CI_MEMORY --network=none --pids-limit=4096"
# The vendoring and advisory containers (refresh, the only ones with network): the same ceiling.
FETCH_RUN="$PODMAN_RUN --memory=$CI_MEMORY"
# The packaging, diffoscope and install-check containers (deb, deb-repro, deb-check): no network, the same ceiling.
TOOL_RUN="$PODMAN_RUN --memory=$CI_MEMORY --network=none"
NICE="nice -n 10"
IMG=localhost/x0ne-engine-ci:$(sha256sum "$CI_DIR/Containerfile" | cut -c1-12)
SCCACHE_BIN=${SCCACHE_BIN:-$(command -v sccache || true)}
SCCACHE_DISK=${SCCACHE_DISK:-$CI_CACHE/sccache}
OUT=$REPO/out/ci
# The verdict of a named run (CI_VERDICT_NAME, see run) beside make ci's, never in its place.
VNAME=${CI_VERDICT_NAME:-}
[ -z "$VNAME" ] || [[ $VNAME =~ ^[a-z0-9][a-z0-9-]*$ ]] || die "CI_VERDICT_NAME=$VNAME: lowercase letters, digits and dashes only"
VSUF=${VNAME:+.$VNAME}
LATEST=latest$VSUF.json
X0NE_OS=${X0NE_OS:-}
DEB_STAGING=${DEB_STAGING:-$CI_CACHE/staging}
# The release never builds in a target or sccache that make ci or run-fixtures write: a build.rs,
# a test or the run-fixtures example could leave there what the .deb ships (audit of 2026-10-15).
DEB_TARGET=${DEB_TARGET:-$CI_CACHE/deb-target}
mkdir -p "$OUT" "$CI_CACHE"/{target,vendor,cargo-home,logs,src,deb,v2/src,v2/target,v2/stage} "$(dirname "$LOCK")"

# disk_guard <path>...: the filesystem of each path keeps CI_MIN_DISK_GB free.
# Target and staging may sit on another filesystem than / (a data disk): checking / alone missed it.
disk_guard() {
  local p avail mnt
  for p in "$@"; do
    while [ ! -e "$p" ]; do p=$(dirname "$p"); done
    read -r avail mnt < <(df -BG --output=avail,target -- "$p" | tail -n 1) || true
    avail=${avail//[!0-9]/}
    [ -n "$avail" ] || die "df gave no free space for $p"
    [ "$avail" -ge "$CI_MIN_DISK_GB" ] || die "only ${avail} GB free on $mnt ($p), need $CI_MIN_DISK_GB"
  done
}

image() {
  if podman image exists "$IMG"; then log "image $IMG present"; return; fi
  log "building $IMG"
  podman build --pull=never -f "$CI_DIR/Containerfile" -t "$IMG" "$CI_DIR" >&2
}

# Vendor the crates of one exported source tree (keyed by the Cargo.lock hash)
# and refresh the RustSec database. Network allowed here, nowhere else.
refresh() {
  local src=$1 force=${2:-0}
  local key; key=$(sha256sum "$src/Cargo.lock" | cut -c1-16)
  VENDOR=$CI_CACHE/vendor/$key
  if [ "$force" = 1 ] || [ ! -s "$VENDOR/config.toml" ]; then
    log "vendoring crates for Cargo.lock $key"
    rm -rf "$VENDOR"; mkdir -p "$VENDOR"
    $NICE $FETCH_RUN --name "engine-ci-vendor-$$" \
      -v "$src":/src:ro -v "$VENDOR":/vendor -v "$CI_CACHE/cargo-home":/cargo-home \
      -e CARGO_HOME=/cargo-home -e CARGO_NET_OFFLINE=false -w /src "$IMG" \
      bash -c 'cargo vendor --locked --versioned-dirs /vendor/crates > /vendor/config.toml' >&2
    [ -s "$VENDOR/config.toml" ] || die "cargo vendor failed"
  fi
  DB=$CI_CACHE/cargo-home/advisory-dbs
  local age=999999 tried= fetched=
  [ -d "$DB" ] && age=$(( $(date +%s) - $(stat -c %Y "$DB") ))
  if [ "$force" = 1 ] || [ "$age" -gt "${CI_ADVISORY_MAX_AGE:-86400}" ]; then
    log "fetching the RustSec advisory database"
    tried=1
    if $NICE $FETCH_RUN --name "engine-ci-advisory-$$" \
      -v "$src":/src:ro -v "$CI_CACHE/cargo-home":/cargo-home \
      -e CARGO_HOME=/cargo-home -e CARGO_NET_OFFLINE=false -w /src "$IMG" \
      cargo deny fetch db >&2; then touch "$DB"; fetched=1; fi
  fi
  [ -d "$DB" ] || die "no advisory database"
  # The age the deny step judges against, in the verdict (advisory_db_age_s): a fetch that
  # failed, stale or forced, leaves the old database; "advisories ok" means "none known then".
  ADVISORY_AGE=$(( $(date +%s) - $(stat -c %Y "$DB") ))
  if [ -n "$tried" ] && [ -z "$fetched" ]; then
    log "advisory database: fetch failed, the deny step judges against a copy $ADVISORY_AGE s old"
  fi
}

# lock_build: hold the build lock (CI_LOCK, shared with other Rust builds) on fd 9 until the caller closes it.
# A cargo that takes no lock is not waited for: polling pgrep every 60 s starved make ci
# 43 min for 162 s of work. A CI_CGROUP_PARENT ceiling bounds such a cargo.
lock_build() {
  local t=$SECONDS
  exec 9>"$LOCK"
  if ! flock -n 9; then
    log "waiting for $LOCK, at most $CI_WAIT_MAX s"
    flock -w "$CI_WAIT_MAX" 9 || die "build lock $LOCK still held after $CI_WAIT_MAX s"
  fi
  LOCK_WAIT=$((SECONDS - t))
  log "lock: waited $LOCK_WAIT s"
}

# export_src <sha>: the commit in $CI_CACHE/src/<sha>, files dated at the commit (deb, refresh).
export_src() {
  SRC=$CI_CACHE/src/$1
  rm -rf "$SRC"; mkdir -p "$SRC"
  git -C "$REPO" archive --format=tar "$1" | tar -x -C "$SRC"
}

# stage_tree <sha>: the tree under test in STAGE: the commit, or with CI_WORKTREE=1 the
# working tree's tracked and untracked files that exist, ignored ones (out/, target/) left out.
stage_tree() {
  STAGE=$CI_CACHE/v2/stage/$$
  rm -rf "$STAGE"; mkdir -p "$STAGE"
  if [ "${CI_WORKTREE:-0}" = 1 ]; then
    (cd "$REPO" && git ls-files -z -c -o --exclude-standard --deduplicate \
      | while IFS= read -r -d '' f; do if [ -e "$f" ] || [ -L "$f" ]; then printf '%s\0' "$f"; fi; done \
      | tar --null --no-recursion -T - -cf -) | tar -x -C "$STAGE"
  else
    git -C "$REPO" archive --format=tar "$1" | tar -x -C "$STAGE"
  fi
}

# stage_is <sha>: whether STAGE holds exactly the tree of <sha> (paths, modes, links, contents),
# hashed through a throwaway index and object directory. git status misses a file marked
# --skip-worktree or --assume-unchanged, and untracked files under status.showUntrackedFiles=no.
stage_is() {
  local tmp gitdir common want got=
  tmp=$(mktemp -d) || return 1
  gitdir=$(git -C "$REPO" rev-parse --absolute-git-dir)
  common=$(cd "$REPO" && cd "$(git rev-parse --git-common-dir)" && pwd)
  want=$(git -C "$REPO" rev-parse "$1^{tree}")
  mkdir -p "$tmp/objects"
  got=$(cd "$STAGE" && GIT_DIR=$gitdir GIT_WORK_TREE=$STAGE GIT_INDEX_FILE=$tmp/index \
    GIT_OBJECT_DIRECTORY=$tmp/objects GIT_ALTERNATE_OBJECT_DIRECTORIES=$common/objects \
    sh -c 'git read-tree "$1" && git -c core.fsmonitor=false add -A . && git write-tree' sh "$1") || got=
  rm -rf "$tmp"
  [ -n "$got" ] && [ "$got" = "$want" ]
}

# sync_src <from> <to>: <to> becomes <from>; only a file whose content differs is rewritten.
# Cargo judges freshness by mtime: a tree edited before another tree's build ran that build's
# binaries. Here a changed file is always newer than the last build of <to>.
sync_src() {
  local from=$1 to=$2 rel
  mkdir -p "$to"
  while IFS= read -r -d '' rel; do
    if [ -L "$to/$rel" ] || [ -L "$from/$rel" ]; then
      [ -L "$to/$rel" ] && [ -L "$from/$rel" ] && [ "$(readlink "$to/$rel")" = "$(readlink "$from/$rel")" ] && continue
    elif [ -d "$to/$rel" ] && [ -d "$from/$rel" ]; then continue
    elif [ -f "$to/$rel" ] && [ -f "$from/$rel" ]; then continue
    fi
    rm -rf "${to:?}/$rel"
  done < <(cd "$to" && find . -mindepth 1 -depth -printf '%P\0')
  while IFS= read -r -d '' rel; do
    if [ -L "$from/$rel" ]; then [ -L "$to/$rel" ] || cp -P "$from/$rel" "$to/$rel"
    elif [ -d "$from/$rel" ]; then mkdir -p "$to/$rel"
    elif ! cmp -s "$from/$rel" "$to/$rel"; then rm -f "$to/$rel"; cp --preserve=mode "$from/$rel" "$to/$rel"
    elif [ "$(stat -c %a "$from/$rel")" != "$(stat -c %a "$to/$rel")" ]; then chmod --reference="$from/$rel" "$to/$rel"
    fi
  done < <(cd "$from" && find . -mindepth 1 -printf '%P\0')
}

# tree_id <dir>: one hash of the paths, modes, link targets and contents under <dir>.
tree_id() {
  (cd "$1" && { find . -mindepth 1 \( -type f -o -type l \) -printf '%m %y %p %l\n'
    find . -type f -print0 | xargs -0 -r sha256sum; } | LC_ALL=C sort | sha256sum | cut -c1-16)
}

# gc_target: past CI_TARGET_MAX_GB the target is emptied (under the lock); sccache refills it.
gc_target() {
  local mb; mb=$(du -sm "$CI_TARGET" 2>/dev/null | cut -f1)
  [ -n "$mb" ] || return 0
  if [ "$mb" -gt $((CI_TARGET_MAX_GB * 1024)) ]; then
    log "target over $CI_TARGET_MAX_GB GB ($mb MB): emptied"
    rm -rf "$CI_TARGET"; mkdir -p "$CI_TARGET"
  fi
}

json_result() {
  local status=$1 sha=$2 branch=$3 started=$4 ended=$5 dur=$6 steps=$7 rundir=$8 dirty=$9 tree=${10} wait=${11}
  jq -n --arg status "$status" --arg sha "$sha" --arg branch "$branch" --arg range "${CI_RANGE:-}" \
     --arg started "$started" --arg ended "$ended" --argjson duration "$dur" --arg image "$IMG" \
     --arg log "$rundir" --arg unit "${CI_UNIT:-}" --rawfile steps "$steps" \
     --arg dirty "$dirty" --arg tree "$tree" --argjson wait "$wait" --argjson advisory_db_age_s "${ADVISORY_AGE:--1}" \
     --arg source "$([ "${CI_WORKTREE:-0}" = 1 ] && echo worktree || echo commit)" '
    ($steps | split("\n") | map(select(length > 0) | split("\t") | {name: .[0], status: .[1], seconds: (.[2]|tonumber), exit: (.[3]|tonumber)})) as $s
    | {status: $status, sha: $sha, short: ($sha[0:7] + (if $dirty == "1" then "+dirty" else "" end)), dirty: ($dirty == "1"),
       source: $source, tree: $tree, branch: $branch, range: $range, started: $started, ended: $ended,
       duration_s: $duration, lock_wait_s: $wait, advisory_db_age_s: $advisory_db_age_s, image: $image, unit: $unit, log_dir: $log,
       failed: ($s | map(select(.status != "ok") | .name)), steps: $s}'
}

# full_run: no setting narrows or widens what the run checks.
full_run() { [ -z "${CI_STEPS:-}" ] && [ -z "${CI_RANGE:-}" ] && [ -z "${CI_ALL_FEATURES:-}" ]; }

# The steps of ci/inside.sh that build nothing: no target, so no build lock.
UNBUILT_STEPS="fmt ci-sh features deny gitleaks"
# unbuilt_only: CI_STEPS names only such steps.
unbuilt_only() {
  local s
  [ -n "${CI_STEPS:-}" ] || return 1
  for s in $CI_STEPS; do case " $UNBUILT_STEPS " in *" $s "*) ;; *) return 1 ;; esac; done
}

# reuse_verdict <tree> <sha> <branch> <dirty> <id>: when this tree has a green full verdict under
# this image, younger than CI_REUSE_MAX_AGE, writes it as the verdict of <id> and returns 0.
reuse_verdict() {
  local tree=$1 sha=$2 branch=$3 dirty=$4 id=$5 v=$CI_CACHE/verdicts/$1.json
  [ "${CI_REUSE:-0}" = 1 ] && full_run && [ -s "$v" ] || return 1
  jq -e --arg tree "$tree" --arg image "$IMG" --argjson now "$(date +%s)" --argjson max "${CI_REUSE_MAX_AGE:-86400}" '
    .status == "green" and .tree == $tree and .image == $image and (.failed | length) == 0
    and (.reused_from == null) and ($now - (.ended | sub("\\.[0-9]+"; "") | sub("(?<s>[+-][0-9]{2}):(?<m>[0-9]{2})$"; "\(.s)\(.m)") | strptime("%Y-%m-%dT%H:%M:%S%z") | mktime) <= $max)' "$v" > /dev/null 2>&1 || return 1
  jq --arg sha "$sha" --arg branch "$branch" --arg dirty "$dirty" --arg now "$(date -Is)" --arg file "$v" \
     --arg source "$([ "${CI_WORKTREE:-0}" = 1 ] && echo worktree || echo commit)" '
    {sha: .sha, short: .short, ended: .ended, log_dir: .log_dir, verdict: $file} as $from
    | . + {sha: $sha, short: ($sha[0:7] + (if $dirty == "1" then "+dirty" else "" end)), dirty: ($dirty == "1"),
           source: $source, branch: $branch, started: $now, ended: $now, duration_s: 0, lock_wait_s: 0, unit: "",
           reused_from: $from, steps: (.steps | map(. + {reused: true}))}' "$v" > "$OUT/$id.json.tmp" || return 1
  mv "$OUT/$id.json.tmp" "$OUT/$id.json"; ln -sfn "$id.json" "$OUT/$LATEST"
}

notify() {
  local status=$1 short=$2 dur=$3 failed=$4
  logger -t engine-ci "$status $short ${dur}s${failed:+ failed: $failed}" || true
  [ -n "${CI_NOTIFY_CMD:-}" ] || return 0
  $CI_NOTIFY_CMD ci "$status" "$short" "$dur" "$failed" || true
}

run() {
  local sha branch started t0 dirty=0
  sha=$(git -C "$REPO" rev-parse HEAD); branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
  disk_guard / "$CI_CACHE"
  image
  trap 'rm -rf "${STAGE:-}"' EXIT
  stage_tree "$sha"
  # A tree under test that is not its commit's: its verdict is <sha>+dirty, kept apart.
  if [ "${CI_WORKTREE:-0}" = 1 ] && ! stage_is "$sha"; then dirty=1; fi
  local id=$sha short=${sha:0:7}
  [ "$dirty" = 1 ] && { id=$sha+dirty; short=$short+dirty; }
  id=$id$VSUF
  local rundir=$CI_CACHE/logs/$short-$(date +%Y%m%d-%H%M%S)
  mkdir -p "$rundir"
  refresh "$STAGE" 0
  local tree; tree=$(tree_id "$STAGE")
  if [ -n "${CI_RANGE:-}" ]; then git -C "$REPO" log -p --format=fuller "$CI_RANGE" > "$rundir/range.patch" || die "bad CI_RANGE $CI_RANGE"; fi
  if reuse_verdict "$tree" "$sha" "$branch" "$dirty" "$id"; then
    rm -rf "$STAGE"; rmdir "$rundir" 2> /dev/null || true
    log "green $short reused: tree $tree was green as $(jq -r '.reused_from | .short + " at " + .ended' "$OUT/$id.json"), same image (without CI_REUSE=1 it runs again); $OUT/$id.json"
    notify green "$short" 0 ""
    return 0
  fi
  local sc=() src=$CI_SRC tgt
  if [ -x "$SCCACHE_BIN" ] && [ -d "$SCCACHE_DISK" ]; then sc=(-v "$(readlink -f "$SCCACHE_BIN")":/usr/local/bin/sccache:ro -v "$SCCACHE_DISK":/sccache); fi
  tgt=(-v "$CI_TARGET":/target)
  if unbuilt_only; then
    # Nothing is built: the staged copy is the source, no target is mounted, the lock is not
    # taken (CI_SRC and CI_TARGET are written under it only). STAGE is removed on exit.
    LOCK_WAIT=0; src=$STAGE; tgt=(); sc=()
    log "steps that build nothing ($CI_STEPS): no build lock, no shared target"
  else
    lock_build
    gc_target
    sync_src "$STAGE" "$CI_SRC"; rm -rf "$STAGE"
  fi
  started=$(date -Is); t0=$SECONDS
  log "run $short on $branch, tree $tree, image $IMG, jobs $CI_JOBS, logs $rundir"
  set +e
  # keep-id: tests/process_hardening.rs needs an ordinary user (root passes every check).
  $NICE $BUILD_RUN --userns=keep-id \
    --name "engine-ci-${sha:0:7}-$$" \
    -v "$src":/src:ro -v "$VENDOR":/vendor:ro -v "$DB":/advisory-db:ro \
    "${tgt[@]}" -v "$rundir":/out -v "$CI_DIR":/ci:ro "${sc[@]}" \
    -e CARGO_PROFILE_DEV_DEBUG="$CI_DEBUGINFO" \
    -e CARGO_BUILD_JOBS="$CI_JOBS" -e CI_STEPS="${CI_STEPS:-}" -e CI_ALL_FEATURES="${CI_ALL_FEATURES:-}" \
    -e CI_LANES="$CI_LANES" \
    "$IMG" bash /ci/inside.sh
  local rc=$?
  exec 9>&-
  set -e
  local dur=$((SECONDS - t0)) status=red
  [ $rc -eq 0 ] && status=green
  [ -s "$rundir/steps.tsv" ] || { printf 'container\tfail\t%d\t%d\n' "$dur" "$rc" > "$rundir/steps.tsv"; }
  json_result "$status" "$sha" "$branch" "$started" "$(date -Is)" "$dur" "$rundir/steps.tsv" "$rundir" "$dirty" "$tree" "$LOCK_WAIT" > "$OUT/$id.json.tmp"
  mv "$OUT/$id.json.tmp" "$OUT/$id.json"; ln -sfn "$id.json" "$OUT/$LATEST"
  # The verdict of this tree, for a later run of the same files (reuse_verdict).
  if [ "$status" = green ] && full_run; then
    mkdir -p "$CI_CACHE/verdicts"; cp "$OUT/$id.json" "$CI_CACHE/verdicts/$tree.json.tmp" && mv "$CI_CACHE/verdicts/$tree.json.tmp" "$CI_CACHE/verdicts/$tree.json"
  fi
  local failed; failed=$(jq -r '.failed | join(", ")' "$OUT/$id.json")
  log "$status $short in ${dur}s, lock wait ${LOCK_WAIT}s${failed:+ (failed: $failed)}; $OUT/$id.json"
  notify "$status" "$short" "$dur" "$failed"
  [ "$status" = green ]
}

# ---- deb: the installable package of one commit (make deb, deb-repro, deb-check).
os_image() {
  [ -n "$X0NE_OS" ] || die "make deb needs X0NE_OS, an x0ne OS checkout (x0ne OS is not public yet)"
  local cf=$X0NE_OS/containers/Containerfile.build
  [ -s "$cf" ] || die "no x0ne OS checkout at $X0NE_OS (X0NE_OS=...)"
  OS_IMG=localhost/x0ne-engine-deb:$(sha256sum "$cf" | cut -c1-12)
  if podman image exists "$OS_IMG"; then return; fi
  log "building $OS_IMG from $cf"
  podman build --pull=never -f "$cf" -t "$OS_IMG" "$X0NE_OS/containers" >&2
}

# Debian version of one commit: crate version (0.2.0-beta.2 -> 0.2.0~beta.2)
# + git<commit date, UTC>.<sha7>. 0.2.0~beta1 (x0ne-OS pkg-lib.sh) sorts below. The date carries the time:
# two commits of one day must still sort in commit order.
deb_version() {
  local src=$1 sha=$2
  local v; v=$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$src/Cargo.toml" | head -1 | sed -E 's/-(alpha|beta|rc)/~\1/')
  [ -n "$v" ] || die "no workspace version in $src/Cargo.toml"
  echo "$v+git$(TZ=UTC git -C "$REPO" show -s --format=%cd --date=format-local:%Y%m%dT%H%M%S "$sha").${sha:0:7}"
}

# deb_build <sha> <workdir> <log dir> <sccache 1|0>: binaries then .deb files in <workdir>/{bin,debug,debs},
# compiled in a new target under DEB_TARGET (mktemp: a name nothing could prepare), removed once the
# binaries are out. Sets DEB_SC to the sccache actually mounted (1 or 0).
deb_build() {
  local sha=$1 work=$2 logdir=$3 use_sc=$4 short=${1:0:7} target
  local epoch; epoch=$(git -C "$REPO" log -1 --format=%ct "$sha")
  rm -rf "$work"; mkdir -p "$work/bin" "$logdir" "$DEB_TARGET"
  target=$(mktemp -d "$DEB_TARGET/release.XXXXXXXX") || return 1
  DEB_TARGETS+=("$target")
  local sc=(); DEB_SC=0
  if [ "$use_sc" = 1 ] && [ -x "$SCCACHE_BIN" ] && [ -d "$SCCACHE_DISK" ]; then sc=(-v "$(readlink -f "$SCCACHE_BIN")":/usr/local/bin/sccache:ro -v "$SCCACHE_DISK":/sccache); DEB_SC=1; fi
  log "release build of $short, version $DEB_VERSION, epoch $epoch, target $target (new, empty), sccache $DEB_SC"
  exec 9>"$LOCK"; flock -w "$CI_WAIT_MAX" 9 || { log "build lock busy"; return 1; }
  $NICE $BUILD_RUN --name "engine-deb-$short-$$" \
    -v "$SRC":/src:ro -v "$VENDOR":/vendor:ro -v "$target":/target -v "$work":/out -v "$CI_DIR":/ci:ro "${sc[@]}" \
    -e CARGO_BUILD_JOBS="$CI_JOBS" -e SOURCE_DATE_EPOCH="$epoch" -e X0NE_GIT_SHA="$sha" -e DEB_VERSION="$DEB_VERSION" \
    -e DEB_SCCACHE="$DEB_SC" "$IMG" bash /ci/inside-deb.sh > "$logdir/inside-deb.log" 2>&1 \
    || { tail -n 30 "$logdir/inside-deb.log" >&2; exec 9>&-; return 1; }
  exec 9>&-
  rm -rf "$target"
  # x0ne-OS build.sh (equivs): CHANNEL=edge reads /out/edge/bin, writes /out/edge/debs. It copies
  # /key/x0ne-archive-keyring.gpg for the keyring package, which is not built here: an empty file.
  # Its sed misses the first entry of the "Files:" line (stays /out/bin/, 2026-10-13): the
  # binaries are mounted at /out/bin too, so that entry is this build, not another.
  # The x0ne-engine control file is this commit's (ci/deb/x0ne-engine.ctl), mounted over the
  # one of x0ne-OS.
  local key=$CI_CACHE/deb/key; mkdir -p "$key"; : > "$key/x0ne-archive-keyring.gpg"
  log "packaging with $X0NE_OS/scripts/build.sh in $OS_IMG"
  $NICE $TOOL_RUN --name "engine-deb-pkg-$short-$$" \
    -v "$X0NE_OS":/src:ro -v "$SRC/ci/deb/x0ne-engine.ctl":/src/packages/x0ne-engine.ctl:ro \
    -v "$work":/out/edge -v "$work/bin":/out/bin:ro -v "$key":/key:ro \
    -e CHANNEL=edge -e SOURCE_DATE_EPOCH="$epoch" "$OS_IMG" bash /src/scripts/build.sh x0ne-engine \
    > "$logdir/package.log" 2>&1 || { tail -n 30 "$logdir/package.log" >&2; return 1; }
  rm -rf "$work/pkgsrc"
  ls "$work"/debs/*.deb >/dev/null 2>&1 || { log "no .deb produced"; return 1; }
  deb_verify "$work" "$short" "$logdir"
}

# deb_verify <workdir> <short> <log dir>: ci/deb-verify.sh in the packaging image, no network.
# Each .deb ships exactly its binaries of SHA256SUMS.engine, by name and sha256, nothing else.
deb_verify() {
  local work=$1 short=$2 logdir=$3
  $NICE $TOOL_RUN --name "engine-deb-verify-$short-$$" -v "$work":/w:ro -v "$CI_DIR":/ci:ro "$OS_IMG" \
    bash /ci/deb-verify.sh /w > "$logdir/verify.log" 2>&1 \
    || { cat "$logdir/verify.log" >&2; log "the .deb files do not ship exactly the binaries of SHA256SUMS.engine"; return 1; }
  log "the .deb files ship the binaries of SHA256SUMS.engine: $(grep -c '^same ' "$logdir/verify.log")"
}

deb_json() {
  local status=$1 sha=$2 branch=$3 started=$4 dur=$5 stage=$6 repro=$7 logdir=$8 sc_build=$9 sc_repro=${10}
  local epoch; epoch=$(git -C "$REPO" log -1 --format=%ct "$sha")
  local sums="$stage/SHA256SUMS"; [ -s "$sums" ] || sums=/dev/null
  jq -n --arg status "$status" --arg sha "$sha" --arg branch "$branch" --arg version "$DEB_VERSION" \
     --argjson epoch "$epoch" --arg started "$started" --arg ended "$(date -Is)" --argjson duration "$dur" \
     --arg image "$IMG" --arg os_image "${OS_IMG:-}" --arg x0ne_os "$(git -C "$X0NE_OS" rev-parse HEAD 2>/dev/null || echo unknown)" \
     --arg repro "$repro" --arg sc_build "$sc_build" --arg sc_repro "$sc_repro" --arg ci "$([ -s "$OUT/$sha.json" ] && echo "$OUT/$sha.json")" --arg log "$logdir" --arg stage "$stage" --rawfile sums "$sums" '
    {status: $status, sha: $sha, short: $sha[0:7], branch: $branch, version: $version, source_date_epoch: $epoch,
     started: $started, ended: $ended, duration_s: $duration, image: $image, packaging_image: $os_image,
     x0ne_os_commit: $x0ne_os, reproducible: (if $repro == "" then null else ($repro == "yes") end),
     sccache: {build: (if $sc_build == "" then null else ($sc_build == "1") end),
               repro: (if $sc_repro == "" then null else ($sc_repro == "1") end)},
     ci_verdict: (if $ci == "" then null else $ci end),
     staging: $stage, log_dir: $log,
     debs: ($sums | split("\n") | map(select(length > 0) | split("  ") | {file: .[1], sha256: .[0]}))}'
}

deb_notify() {
  local status=$1 short=$2 dur=$3 detail=$4
  logger -t engine-deb "$status $short ${dur}s $detail" || true
  [ -n "${CI_NOTIFY_CMD:-}" ] || return 0
  $CI_NOTIFY_CMD deb "$status" "$short" "$dur" "$detail" || true
}

# deb [repro]: stage DEB_STAGING/<sha>/ (the x0ne-engine .deb, SHA256SUMS, debug/, deb.json).
deb() {
  local repro=${1:-0} sha branch started t0 status=red detail="" is_repro="" sc_build="" sc_repro=""
  # The release targets of this run, removed on any exit (a failed build leaves its own).
  DEB_TARGETS=()
  trap 'rm -rf "${DEB_TARGETS[@]}"' EXIT
  sha=$(git -C "$REPO" rev-parse HEAD); branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
  local short=${sha:0:7}
  local logdir=$CI_CACHE/logs/deb-$short-$(date +%Y%m%d-%H%M%S) stage=$DEB_STAGING/$sha
  disk_guard / "$CI_CACHE" "$DEB_TARGET" "$DEB_STAGING"
  image; os_image
  export_src "$sha"; refresh "$SRC" 0
  DEB_VERSION=$(deb_version "$SRC" "$sha")
  started=$(date -Is); t0=$SECONDS
  mkdir -p "$DEB_STAGING"
  local work=$CI_CACHE/deb/$short
  if deb_build "$sha" "$work" "$logdir" "${DEB_SCCACHE:-0}"; then
    sc_build=$DEB_SC
    rm -rf "$stage.tmp"; mkdir -p "$stage.tmp"
    cp "$work"/debs/*.deb "$stage.tmp/"; cp "$work/bin/SHA256SUMS.engine" "$stage.tmp/"
    # The debug files of the stripped binaries, by build-id: gdb -iex 'set debug-file-directory <stage>/debug'.
    if [ -d "$work/debug" ]; then cp -r "$work/debug" "$stage.tmp/debug"; fi
    (cd "$stage.tmp" && sha256sum ./*.deb | sed 's#  \./#  #' > SHA256SUMS)
    status=green; detail="$(cd "$stage.tmp" && ls ./*.deb | sed 's#^\./##' | tr '\n' ' ')"
    if [ "$repro" = 1 ]; then
      local work2=$CI_CACHE/deb/$short-repro
      log "second build in a new container, its own new target, sccache ${DEB_REPRO_SCCACHE:-0} (reproducibility)"
      if deb_build "$sha" "$work2" "$logdir/repro" "${DEB_REPRO_SCCACHE:-0}"; then
        sc_repro=$DEB_SC
        if (cd "$work2/debs" && sha256sum -c --quiet "$stage.tmp/SHA256SUMS"); then
          is_repro=yes; detail="reproducible, $detail"
        else
          is_repro=no; status=red; detail="NOT reproducible, see $logdir/diffoscope.html"
          diffoscope_run "$work/debs" "$work2/debs" "$logdir"
        fi
      else
        is_repro=no; status=red; detail="second build failed"
      fi
    fi
  else
    detail="build or packaging failed, see $logdir"
  fi
  local dur=$((SECONDS - t0))
  [ "$status" = green ] && { rm -rf "$stage"; mv "$stage.tmp" "$stage"; ln -sfn "$sha" "$DEB_STAGING/latest"; } || { mkdir -p "$stage"; rm -rf "$stage.tmp"; }
  deb_json "$status" "$sha" "$branch" "$started" "$dur" "$stage" "$is_repro" "$logdir" "$sc_build" "$sc_repro" > "$stage/deb.json.tmp"
  mv "$stage/deb.json.tmp" "$stage/deb.json"
  mkdir -p "$REPO/out/deb"; ln -sfn "$stage/deb.json" "$REPO/out/deb/latest.json"
  log "$status $short in ${dur}s: $detail; $stage/deb.json"
  deb_notify "$status" "$short" "$dur" "$detail"
  [ "$status" = green ]
}

# diffoscope_run <debs A> <debs B> <log dir>: one HTML report per differing .deb, in a pinned image.
diffoscope_run() {
  local a=$1 b=$2 logdir=$3
  local cf=$CI_DIR/Containerfile.diffoscope img
  img=localhost/x0ne-engine-diffoscope:$(sha256sum "$cf" | cut -c1-12)
  podman image exists "$img" || podman build --pull=never -f "$cf" -t "$img" "$CI_DIR" >&2
  local f
  for f in "$a"/*.deb; do
    local n; n=$(basename "$f")
    cmp -s "$f" "$b/$n" && continue
    log "diffoscope $n -> $logdir/diffoscope-$n.html"
    $NICE $TOOL_RUN --name "engine-diffoscope-$$" -v "$a":/a:ro -v "$b":/b:ro -v "$logdir":/out "$img" \
      diffoscope --html "/out/diffoscope-$n.html" --text "/out/diffoscope-$n.txt" "/a/$n" "/b/$n" >/dev/null 2>&1 || true
    head -n 60 "$logdir/diffoscope-$n.txt" >&2 || true
  done
}

# deb_check: dpkg -i of the staged x0ne-engine .deb in trixie (ci/Containerfile.check, no
# network at run time); the --version of each of the three client binaries names the commit.
deb_check() {
  local sha; sha=$(git -C "$REPO" rev-parse HEAD); local short=${sha:0:7} stage=$DEB_STAGING/$sha
  local deb; deb=$(ls "$stage"/x0ne-engine_*.deb 2>/dev/null | head -1)
  [ -n "$deb" ] || die "no staged x0ne-engine .deb in $stage (make deb first)"
  local cf=$CI_DIR/Containerfile.check img
  img=localhost/x0ne-engine-check:$(sha256sum "$cf" | cut -c1-12)
  podman image exists "$img" || podman build --pull=never -f "$cf" -t "$img" "$CI_DIR" >&2
  local out
  out=$($TOOL_RUN --name "engine-deb-check-$short-$$" -v "$stage":/stage:ro "$img" bash -c '
    set -e
    engine=$(ls /stage/x0ne-engine_*.deb)
    # The server of the TD ships in the CVM image only (2026-11-04): no package of its own.
    if ls /stage/x0ne-infer_*.deb 2>/dev/null; then echo "an x0ne-infer .deb is staged"; exit 5; fi
    # Contents and relations, from the archives themselves.
    if dpkg-deb -c "$engine" | grep -E "lab|ssh"; then echo "x0ne-engine ships a lab or SSH path"; exit 5; fi
    # The client ships no server: x0ne-infer, the server of the TD, is in the CVM image (2026-11-04).
    if dpkg-deb -c "$engine" | grep -E " \./usr/bin/x0ne-infer\$"; then echo "x0ne-engine ships x0ne-infer"; exit 5; fi
    dpkg -i "$engine" > /tmp/dpkg.log 2>&1 || { cat /tmp/dpkg.log; exit 3; }
    dpkg -s x0ne-engine | grep -E "^(Status|Version):"
    for b in x0ne-engine x0ne-engine-cloud x0ne-engine-proxy; do
      [ -x /usr/bin/$b ] || { echo "missing /usr/bin/$b"; exit 4; }
      echo "$b: $($b --version 2>&1 | head -1)"
    done
    if dpkg -L x0ne-engine | grep -E "lab|ssh|/usr/bin/x0ne-infer\$"; then echo "lab, SSH or server binary shipped"; exit 5; fi
    [ ! -e /usr/bin/x0ne-infer ] || { echo "x0ne-engine installs /usr/bin/x0ne-infer"; exit 5; }
    # run left with the lab server (2026-11-04): a named refusal that points to cloud run.
    msg=$(X0NE_ENGINE_HOME=/tmp/eh x0ne-engine run some-model 2>&1) && { echo "run served"; exit 6; }
    grep -q "x0ne-engine cloud run" <<<"$msg" || { echo "run message: $msg"; exit 6; }
    echo "run refused: $msg"
    cat /usr/share/x0ne-engine/x0ne-engine.version' 2>&1) || { printf '%s\n' "$out" >&2; die "install check failed"; }
  printf '%s\n' "$out" >&2
  local b
  for b in x0ne-engine x0ne-engine-cloud x0ne-engine-proxy; do
    grep -qE "^$b: $b [^ ]+ \(git $sha\)\$" <<<"$out" || die "$b --version does not name $sha"
  done
  [ "$(grep -c "^Status: install ok installed" <<<"$out")" = 1 ] || die "x0ne-engine is not installed ok"
  log "install check green for $short"
}

# A measurements file that names no image and no register: nothing measured to lose.
measurements_placeholder() {
  grep -Eq '"image": *null' "$1" && grep -Eq '"registers": *\{ *\}' "$1"
}

# The files of $1 that would replace a different file of the tree.
replaced_files() {
  local rel
  while IFS= read -r -d '' rel; do
    rel=${rel#./}
    [ -e "$REPO/$rel" ] || continue
    cmp -s "$1/$rel" "$REPO/$rel" && continue
    case $rel in measurements/*.json) measurements_placeholder "$REPO/$rel" && continue ;; esac
    printf '%s\n' "$rel"
  done < <(cd "$1" && find . -type f -print0 | sort -z)
}

run_fixtures() {
  local att=${RUN_ATTESTATION:-} ref=${RUN_REFERENCE:-} day=${RUN_DATE:-} name=${RUN_IMAGE:-}
  local machine=${RUN_MACHINE:-a3-highgpu-1g}
  [ -s "$att" ] && [ -s "$ref" ] && [ -n "$name" ] || die "run-fixtures: RUN_ATTESTATION, RUN_REFERENCE (files), RUN_DATE and RUN_IMAGE are required"
  case $day in [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]) ;; *) die "run-fixtures: RUN_DATE ${day:-unset}: expected YYYY-MM-DD" ;; esac
  local out=$REPO/out/run-fixtures/$day
  disk_guard / "$CI_CACHE"
  image
  refresh "$REPO" 0
  rm -rf "$out"; mkdir -p "$out"
  local sig=() tk=()
  [ -f "$ref.sig" ] && sig=(-v "$(readlink -f "$ref.sig")":/in/reference.json.sig:ro)
  [ "${RUN_TEST_KEY:-0}" = 1 ] && tk=(--test-key)
  set +e
  flock -w "$CI_WAIT_MAX" "$LOCK" $NICE $BUILD_RUN --userns=keep-id \
    --name "engine-run-fixtures-$$" \
    -v "$REPO":/src:ro -v "$VENDOR":/vendor:ro -v "$CI_CACHE/target":/target -v "$out":/out \
    -v "$(readlink -f "$att")":/in/attestation.json:ro -v "$(readlink -f "$ref")":/in/reference.json:ro "${sig[@]}" \
    "$IMG" bash -c 'export HOME=/tmp/home CARGO_HOME=/tmp/cargo-home CARGO_TARGET_DIR=/target
      mkdir -p "$CARGO_HOME" && cp /vendor/config.toml "$CARGO_HOME/config.toml" && cd /src &&
      cargo run --locked --offline -q -p x0ne-infer --features probe --example run_fixtures -- \
        --attestation /in/attestation.json --reference /in/reference.json \
        --date "$1" --image "$2" --machine "$3" --out /out --check /src "${@:4}"' run-fixtures "$day" "$name" "$machine" "${tk[@]}"
  local rc=$?
  set -e
  case $rc in
    0) log "run-fixtures: identical to the working tree ($out)" ;;
    1) if [ "${RUN_WRITE:-0}" != 1 ]; then log "run-fixtures: differs from the working tree ($out); RUN_WRITE=1 copies it"; return 1; fi
       [ -n "$(find "$out" -type f -print -quit)" ] || die "run-fixtures: $out is empty"
       local replaced; replaced=$(replaced_files "$out")
       if [ -n "$replaced" ] && [ "${RUN_OVERWRITE:-0}" != 1 ]; then
         log "run-fixtures: refused, nothing written: it would replace $(tr '\n' ' ' <<<"$replaced")in the working tree (the measurements of a run, or the fixtures of a RUN_DATE already taken); RUN_OVERWRITE=1 replaces them"
         return 1
       fi
       tar -C "$out" -cf - . | tar -xf - -C "$REPO"
       log "run-fixtures: copied into the working tree: $(cd "$out" && find . -type f | sort | tr '\n' ' ')" ;;
    3) log "run-fixtures: refused, nothing written"; return 1 ;;
    *) die "run-fixtures: exit $rc" ;;
  esac
}

# Sourced (ci/coverage.sh): its settings and helpers, no command.
if [ "${BASH_SOURCE[0]}" != "$0" ]; then return 0; fi
case $cmd in
  run)     run ;;
  image)   image ;;
  refresh) image; export_src "$(git -C "$REPO" rev-parse HEAD)"; refresh "$SRC" 1 ;;
  status)  [ -e "$OUT/$LATEST" ] && jq '{status, short, dirty, branch, duration_s, lock_wait_s, failed, ended}' "$OUT/$LATEST" || die "no result yet${VNAME:+ named $VNAME}" ;;
  deb)       deb 0 ;;
  deb-repro) deb 1 ;;
  deb-check) deb_check ;;
  run-fixtures) run_fixtures ;;
  deb-status) [ -e "$REPO/out/deb/latest.json" ] && jq '{status, short, version, reproducible, duration_s, staging, ended}' "$REPO/out/deb/latest.json" || die "no deb yet" ;;
  *)       die "usage: ci/ci.sh [run|image|refresh|status|deb|deb-repro|deb-check|deb-status|run-fixtures]" ;;
esac
