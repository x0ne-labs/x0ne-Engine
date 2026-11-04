#!/usr/bin/env bash
# Tests of ci/ci.sh against fake podman, df, pgrep, jq and logger: no container,
# no build, no network. Each case runs ci.sh in a throwaway git repository.
# Run by ci/inside.sh (step ci-sh), or by hand: bash ci/tests/ci-sh.sh
set -euo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
FAILS=0
pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s: %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }

# ---- fixture: repo/ (ci/, Cargo.toml, Cargo.lock), an x0ne OS checkout, data/ for the caches.
REPO=$T/repo BIN=$T/bin DATA=$T/data X0NE_OS=$T/x0ne-os
CI_CACHE=$DATA/engine-ci DEB_STAGING=$DATA/engine-staging
mkdir -p "$REPO" "$BIN" "$T/home" "$X0NE_OS/containers" "$DATA/sccache" "$DEB_STAGING"
cp -r "$SRC/ci" "$SRC/Cargo.toml" "$SRC/Cargo.lock" "$SRC/.gitignore" "$REPO/"
echo "FROM scratch" > "$X0NE_OS/containers/Containerfile.build"
git -C "$REPO" init -q -b main
git -C "$REPO" add -A
git -C "$REPO" -c user.name=t -c user.email=t@t commit -qm fixture
SHA=$(git -C "$REPO" rev-parse HEAD)
# Vendored crates and a fresh advisory database: refresh() needs no network.
KEY=$(sha256sum "$REPO/Cargo.lock" | cut -c1-16)
mkdir -p "$CI_CACHE/vendor/$KEY" "$CI_CACHE/cargo-home/advisory-dbs"
echo '[source]' > "$CI_CACHE/vendor/$KEY/config.toml"

# df: FAKE_DF_DATA GB under $FAKE_DATA, FAKE_DF_ROOT GB anywhere else.
cat > "$BIN/df" <<'EOF'
#!/usr/bin/env bash
p=${!#}
case $p in "$FAKE_DATA"|"$FAKE_DATA"/*) a=$FAKE_DF_DATA m=$FAKE_DATA ;; *) a=$FAKE_DF_ROOT m=/ ;; esac
case " $* " in *" --output=avail,target "*) printf 'Avail Mounted on\n %sG %s\n' "$a" "$m" ;; *) printf 'Avail\n %sG\n' "$a" ;; esac
EOF
# podman: one line per call in $FAKE_LOG; `run` writes what each CI script would and
# appends "<script> target=<host dir> files=<n> sccache=<yes|no> DEB_SCCACHE=<v>" to $FAKE_RUNS.
# inside.sh models cargo's freshness on /src/probe.txt: rebuilt only when newer than the last
# build, and red when the built "binary" says red; it appends "inside.sh src= target= built=".
cat > "$BIN/podman" <<'EOF'
#!/usr/bin/env bash
echo "podman $*" >> "$FAKE_LOG"
[ "${1:-}" = run ] || exit 0
shift
declare -A m=()
dsc=unset
while [ $# -gt 0 ]; do
  case $1 in
    -v) IFS=: read -r h c _ <<<"$2"; m[$c]=$h; shift 2 ;;
    -e) case $2 in DEB_SCCACHE=*) dsc=${2#DEB_SCCACHE=} ;; esac; shift 2 ;;
    --name|-w) shift 2 ;;
    localhost/*) shift; break ;;
    *) shift ;;
  esac
done
script=${2:-$1}
if [ -n "${m[/stage]:-}" ]; then
  # deb_check's container: one --version line per binary; FAKE_NOSHA omits the commit.
  sha=$(basename "${m[/stage]}")
  for b in x0ne-engine x0ne-engine-cloud x0ne-engine-proxy; do
    case " ${FAKE_NOSHA:-} " in *" $b "*) echo "$b: $b 0.2.0" ;; *) echo "$b: $b 0.2.0 (git $sha)" ;; esac
  done
  printf 'Status: install ok installed\n'
  exit 0
fi
if [ -n "${m[/in/attestation.json]:-}" ]; then
  # run-fixtures' container: one file at its repository path (FAKE_RF_FILE, body
  # FAKE_RF_BODY), exit FAKE_RF_RC.
  echo "run-fixtures attestation=${m[/in/attestation.json]} src=${m[/src]}" >> "$FAKE_RUNS"
  f=${m[/out]}/${FAKE_RF_FILE:-measurements/fake.json}
  body='{"version":"fake"}'; [ -n "${FAKE_RF_BODY:-}" ] && body=$FAKE_RF_BODY
  mkdir -p "${f%/*}"; echo "$body" > "$f"
  exit "${FAKE_RF_RC:-0}"
fi
if [ -n "${m[/w]:-}" ]; then
  # deb_verify's container: the binaries of the .deb against SHA256SUMS.engine; exit FAKE_VERIFY_RC.
  echo "deb-verify work=${m[/w]}" >> "$FAKE_RUNS"
  if [ "${FAKE_VERIFY_RC:-0}" = 0 ]; then echo "same x0ne-engine_0_amd64.deb 00  x0ne-engine"; else echo "NOT THE BINARIES OF SHA256SUMS.engine: x0ne-engine"; fi
  exit "${FAKE_VERIFY_RC:-0}"
fi
sc=no; [ -n "${m[/sccache]:-}" ] && sc=yes
case $script in
  /ci/inside.sh)
    src=${m[/src]} t=${m[/target]} built=no
    if [ ! -e "$t/.built" ] || [ "$src/probe.txt" -nt "$t/.built" ]; then
      cat "$src/probe.txt" > "$t/.binary" 2>/dev/null || : > "$t/.binary"; touch "$t/.built"; built=yes
    fi
    echo "inside.sh src=$src target=$t built=$built" >> "$FAKE_RUNS"
    if grep -q red "$t/.binary"; then printf 'test\tfail\t0\t101\n' > "${m[/out]}/steps.tsv"; exit 1; fi
    printf 'fmt\tok\t0\t0\n' > "${m[/out]}/steps.tsv" ;;
  /ci/inside-deb.sh)
    t=${m[/target]}
    echo "${script#/ci/} target=$t files=$(ls -A "$t" | wc -l) sccache=$sc DEB_SCCACHE=$dsc" >> "$FAKE_RUNS"
    # FAKE_DEB_BUILD_RC: the release build fails, after writing to its target as cargo would.
    if [ -n "${FAKE_DEB_BUILD_RC:-}" ]; then mkdir -p "$t/release"; : > "$t/release/x0ne-engine"; exit "$FAKE_DEB_BUILD_RC"; fi
    mkdir -p "${m[/out]}/bin"
    echo sums > "${m[/out]}/bin/SHA256SUMS.engine"
    mkdir -p "${m[/out]}/debug/.build-id/ab"; echo debug > "${m[/out]}/debug/.build-id/ab/cdef.debug" ;;
  /src/scripts/build.sh)
    # FAKE_DEB_DIFFER: the .deb names its work dir, so the two builds of deb-repro differ.
    body=deb; [ -n "${FAKE_DEB_DIFFER:-}" ] && body=${m[/out/edge]}
    mkdir -p "${m[/out/edge]}/debs"; echo "$body" > "${m[/out/edge]}/debs/x0ne-engine_0_amd64.deb" ;;
  deny)
    # The advisory fetch (cargo deny fetch db): exit FAKE_ADVISORY_RC.
    exit "${FAKE_ADVISORY_RC:-0}" ;;
  -c)
    # The vendoring (bash -c 'cargo vendor ...'): a config.toml only with FAKE_VENDOR=1.
    if [ "${FAKE_VENDOR:-0}" = 1 ] && [ -n "${m[/vendor]:-}" ]; then echo '[source]' > "${m[/vendor]}/config.toml"; fi ;;
esac
exit 0
EOF
# pgrep: no cargo or rustc on the host, unless FAKE_PGREP=0.
printf '#!/bin/sh\nexit ${FAKE_PGREP:-1}\n' > "$BIN/pgrep"
printf '#!/bin/sh\nexit 0\n' > "$BIN/logger"
printf '#!/bin/sh\nexit 0\n' > "$BIN/sccache"
# notify-hook stands for CI_NOTIFY_CMD: argument count, then arguments.
printf '#!/bin/sh\necho "$# $*" >> "%s"\n' "$T/notify.log" > "$BIN/notify-hook"
# jq: one line per call in $FAKE_LOG; '{}' for a document, '' for a -r query.
printf '#!/bin/sh\necho "jq $*" >> "$FAKE_LOG"\ncase "$*" in -r*) echo ;; *) echo "{}" ;; esac\n' > "$BIN/jq"
chmod +x "$BIN"/*

# ci <args...>: ci.sh in the fixture (or in $AT) with a clean environment; sets RC and OUT.
ci() {
  : > "$T/podman.log"; : > "$T/runs.log"
  set +e
  OUT=$(cd "${AT:-$REPO}" && env -i HOME="$T/home" PATH="$BIN:/usr/local/bin:/usr/bin:/bin" \
    CI_CACHE="$CI_CACHE" DEB_STAGING="$DEB_STAGING" SCCACHE_BIN="$BIN/sccache" SCCACHE_DISK="$DATA/sccache" \
    X0NE_OS="$X0NE_OS" FAKE_LOG="$T/podman.log" FAKE_RUNS="$T/runs.log" FAKE_DATA="$DATA" \
    FAKE_DF_ROOT="${DF_ROOT:-500}" FAKE_DF_DATA="${DF_DATA:-500}" ${EXTRA:-} bash ci/ci.sh "$@" 2>&1)
  RC=$?
  set -e
}
ran_podman_run() { grep -q '^podman run' "$T/podman.log"; }

# ---- the disk guard covers / and the filesystem of the target and staging ($DATA).
DF_ROOT=500 DF_DATA=149 ci run
if [ "$RC" = 2 ] && grep -q "only 149 GB free on $DATA" <<<"$OUT" && ! ran_podman_run; then pass "run stops when the data disk has 149 GB"
else fail "run stops when the data disk has 149 GB" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi

DF_ROOT=10 DF_DATA=500 ci run
if [ "$RC" = 2 ] && grep -q "only 10 GB free on /" <<<"$OUT" && ! ran_podman_run; then pass "run stops when / has 10 GB"
else fail "run stops when / has 10 GB" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi

DF_ROOT=150 DF_DATA=150 ci run
if [ "$RC" = 0 ] && grep -q 'bash /ci/inside.sh' "$T/podman.log"; then pass "run starts with 150 GB on both"
else fail "run starts with 150 GB on both" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
if ! grep -q -- '--cgroup-parent' "$T/podman.log"; then pass "run sets no cgroup parent unless CI_CGROUP_PARENT names one"
else fail "run sets no cgroup parent unless CI_CGROUP_PARENT names one" "$(grep -- '--cgroup-parent' "$T/podman.log" | head -n 1)"; fi
if grep 'bash /ci/inside.sh' "$T/podman.log" | grep -q -- '--memory=16g .*-e CI_LANES=1 '; then pass "run caps one container at CI_MEMORY and passes CI_LANES"
else fail "run caps one container at CI_MEMORY and passes CI_LANES" "$(grep inside.sh "$T/podman.log" | head -n 1)"; fi
EXTRA="CI_MEMORY=3g CI_LANES=0" ci run
if grep 'bash /ci/inside.sh' "$T/podman.log" | grep -q -- '--memory=3g .*-e CI_LANES=0 '; then pass "CI_MEMORY and CI_LANES reach the container"
else fail "CI_MEMORY and CI_LANES reach the container" "$(grep inside.sh "$T/podman.log" | head -n 1)"; fi
# The networked containers of refresh: the advisory fetch (a database older than -1 s), then
# the vendoring (ci refresh, which the fake podman leaves without config.toml: exit 2).
EXTRA="CI_MEMORY=3g CI_CGROUP_PARENT=t.slice CI_ADVISORY_MAX_AGE=-1" ci run
if grep 'engine-ci-advisory-' "$T/podman.log" | grep -q -- '--cgroup-parent=t.slice --memory=3g '; then pass "the advisory fetch runs under CI_MEMORY and CI_CGROUP_PARENT"
else fail "the advisory fetch runs under CI_MEMORY and CI_CGROUP_PARENT" "[$(grep -- '-advisory-' "$T/podman.log" | head -n 1)]"; fi
EXTRA="CI_MEMORY=3g CI_CGROUP_PARENT=t.slice" ci refresh
if [ "$RC" = 2 ] && grep 'engine-ci-vendor-' "$T/podman.log" | grep -q -- '--cgroup-parent=t.slice --memory=3g '; then pass "the vendoring runs under CI_MEMORY and CI_CGROUP_PARENT"
else fail "the vendoring runs under CI_MEMORY and CI_CGROUP_PARENT" "exit $RC, [$(grep -- '-vendor-' "$T/podman.log" | head -n 1)]"; fi
echo '[source]' > "$CI_CACHE/vendor/$KEY/config.toml"
# The verdict carries the advisory database's age (advisory_db_age_s). A failed fetch leaves
# the old database and says so with that age: the deny step then judges against a stale
# database, visibly, instead of a silent "advisories ok".
json_age() { sed -n 's/.*--argjson advisory_db_age_s \([0-9]*\) .*/\1/p' "$T/podman.log" | tail -n 1; }
touch -d '3 days ago' "$CI_CACHE/cargo-home/advisory-dbs"
EXTRA="FAKE_ADVISORY_RC=1" ci run
age=$(json_age)
if [ "$RC" = 0 ] && [ "${age:-0}" -ge 259200 ] && grep -q "advisory database: fetch failed, the deny step judges against a copy $age s old" <<<"$OUT"; then
  pass "a failed advisory fetch is said with the database's age, which the verdict carries"
else fail "a failed advisory fetch is said with the database's age, which the verdict carries" "exit $RC, age [$age], $(grep -i advisory <<<"$OUT" | head -n 2 | tr '\n' ' ')"; fi
ci run
age=$(json_age)
if [ "$RC" = 0 ] && [ "${age:-999}" -lt 60 ] && ! grep -q "fetch failed" <<<"$OUT"; then pass "a fetched advisory database gives the verdict an age under a minute"
else fail "a fetched advisory database gives the verdict an age under a minute" "exit $RC, age [$age], $(grep -i advisory <<<"$OUT" | head -n 2 | tr '\n' ' ')"; fi
# make ci-refresh fetches whatever the database's age: a fetch that fails there, on a fresh
# copy, is said too.
EXTRA="FAKE_VENDOR=1 FAKE_ADVISORY_RC=1" ci refresh
if [ "$RC" = 0 ] && grep -q "advisory database: fetch failed" <<<"$OUT"; then pass "a forced advisory fetch that fails is said, on a fresh database too"
else fail "a forced advisory fetch that fails is said, on a fresh database too" "exit $RC, $(grep -i advisory <<<"$OUT" | head -n 2 | tr '\n' ' ')"; fi
echo '[source]' > "$CI_CACHE/vendor/$KEY/config.toml"

DF_ROOT=500 DF_DATA=149 ci deb
if [ "$RC" = 2 ] && grep -q "only 149 GB free on $DATA" <<<"$OUT" && ! ran_podman_run; then pass "deb stops when the data disk has 149 GB"
else fail "deb stops when the data disk has 149 GB" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi

# ---- without X0NE_OS, deb names what it needs before any container.
EXTRA="X0NE_OS=" ci deb
if [ "$RC" = 2 ] && grep -q "make deb needs X0NE_OS" <<<"$OUT" && ! ran_podman_run; then pass "deb stops without X0NE_OS"
else fail "deb stops without X0NE_OS" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi

# ---- defaults: the caches go under ~/.cache, not a path of one host.
EXTRA="CI_CACHE=" ci status
if [ -d "$T/home/.cache/x0ne-engine-ci/v2/src" ]; then pass "CI_CACHE defaults to ~/.cache/x0ne-engine-ci"
else fail "CI_CACHE defaults to ~/.cache/x0ne-engine-ci" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi

# ---- host settings: KEY=value lines of ~/.config/x0ne-engine/ci.env, for keys the environment
# leaves unset; anything but a CI setting is refused.
ENVF=$T/home/.config/x0ne-engine/ci.env; mkdir -p "${ENVF%/*}"
printf '# this host\n\nCI_DEBUGINFO=1\n' > "$ENVF"
ci run
if [ "$RC" = 0 ] && grep -q -- '-e CARGO_PROFILE_DEV_DEBUG=1 ' "$T/podman.log"; then pass "ci.env sets a value the environment leaves unset"
else fail "ci.env sets a value the environment leaves unset" "exit $RC, $(grep inside.sh "$T/podman.log")"; fi
EXTRA="CI_DEBUGINFO=2" ci run
if [ "$RC" = 0 ] && grep -q -- '-e CARGO_PROFILE_DEV_DEBUG=2 ' "$T/podman.log"; then pass "the environment wins over ci.env"
else fail "the environment wins over ci.env" "exit $RC, $(grep inside.sh "$T/podman.log")"; fi
printf 'PATH=/nowhere\n' > "$ENVF"
ci run
if [ "$RC" = 2 ] && grep -q "PATH is not a CI setting" <<<"$OUT" && ! ran_podman_run; then pass "ci.env refuses a key that is not a CI setting"
else fail "ci.env refuses a key that is not a CI setting" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
rm -f "$ENVF"

# ---- notifications: CI_NOTIFY_CMD gets <ci|deb> <status> <short sha> <seconds> <detail>.
# Without it, the journal line only.
: > "$T/notify.log"
ci run
if [ "$RC" = 0 ] && [ ! -s "$T/notify.log" ]; then pass "run notifies nobody without CI_NOTIFY_CMD"
else fail "run notifies nobody without CI_NOTIFY_CMD" "hook [$(cat "$T/notify.log")]"; fi
HEAD7=$(git -C "$REPO" rev-parse --short=7 HEAD)
EXTRA="CI_NOTIFY_CMD=$BIN/notify-hook" ci run
if [ "$RC" = 0 ] && grep -Eq "^5 ci green $HEAD7 [0-9]+ ?\$" "$T/notify.log"; then pass "run calls CI_NOTIFY_CMD with ci, the verdict, the commit and the duration"
else fail "run calls CI_NOTIFY_CMD with ci, the verdict, the commit and the duration" "[$(cat "$T/notify.log")]"; fi
: > "$T/notify.log"
EXTRA="CI_NOTIFY_CMD=$BIN/notify-hook" ci deb
if [ "$RC" = 0 ] && grep -Eq "^5 deb green $HEAD7 [0-9]+ x0ne-engine_0_amd64.deb ?\$" "$T/notify.log"; then pass "deb calls CI_NOTIFY_CMD with deb, the verdict, the commit, the duration and the packages"
else fail "deb calls CI_NOTIFY_CMD with deb, the verdict, the commit, the duration and the packages" "exit $RC, [$(cat "$T/notify.log")]"; fi
EXTRA="CI_NOTIFY_CMD=false" ci run
if [ "$RC" = 0 ]; then pass "a failing CI_NOTIFY_CMD leaves the verdict alone"
else fail "a failing CI_NOTIFY_CMD leaves the verdict alone" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi

# ---- deb: the binaries the .deb files ship are those of SHA256SUMS.engine, and the debug files are staged.
ci deb
if [ "$RC" = 0 ] && grep -q "^deb-verify work=$CI_CACHE/deb/" "$T/runs.log" && [ -f "$DEB_STAGING/$SHA/debug/.build-id/ab/cdef.debug" ]; then
  pass "deb checks the binaries of the .deb and stages the debug files"
else fail "deb checks the binaries of the .deb and stages the debug files" "exit $RC, $(tr '\n' ';' < "$T/runs.log")"; fi
EXTRA="FAKE_VERIFY_RC=1" ci deb
if [ "$RC" = 1 ] && grep -q "the .deb files do not ship exactly the binaries of SHA256SUMS.engine$" <<<"$OUT" && grep -q "NOT THE BINARIES OF SHA256SUMS" <<<"$OUT"; then
  pass "deb is red when a .deb ships other bytes than SHA256SUMS.engine"
else fail "deb is red when a .deb ships other bytes than SHA256SUMS.engine" "exit $RC, $(tail -n 3 <<<"$OUT" | tr '\n' ' ')"; fi

# ---- deb-check requires the commit in the --version of each of the three client binaries.
mkdir -p "$DEB_STAGING/$SHA"; echo deb > "$DEB_STAGING/$SHA/x0ne-engine_0_amd64.deb"
ci deb-check
if [ "$RC" = 0 ] && grep -q "install check green" <<<"$OUT"; then pass "deb-check passes when the three binaries name the commit"
else fail "deb-check passes when the three binaries name the commit" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
for b in x0ne-engine x0ne-engine-cloud x0ne-engine-proxy; do
  EXTRA="FAKE_NOSHA=$b" ci deb-check
  if [ "$RC" = 2 ] && grep -q "$b --version does not name $SHA" <<<"$OUT"; then pass "deb-check fails when $b omits the commit"
  else fail "deb-check fails when $b omits the commit" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
done

# ---- deb builds outside the caches make ci and run-fixtures write (2026-10-30): a test or a build.rs
# of make ci, or the example of run-fixtures, could leave there what the .deb ships. Each release
# build gets a new, empty target under DEB_TARGET, removed after; no sccache unless DEB_SCCACHE=1.
# builds <n>: the target, file count and sccache of the n-th inside-deb.sh run.
builds() { grep '^inside-deb' "$T/runs.log" | sed -n "$1p"; }
target_of() { sed -n 's/^inside-deb\.sh target=\([^ ]*\) .*/\1/p' <<<"$1"; }
mkdir -p "$CI_CACHE/target/release" "$CI_CACHE/v2/target/release" "$CI_CACHE/deb-target"
echo planted > "$CI_CACHE/target/release/x0ne-infer"; echo planted > "$CI_CACHE/v2/target/release/x0ne-infer"
ci deb
one=$(builds 1); tgt=$(target_of "$one")
case $tgt in "$CI_CACHE/deb-target/"?*) under=1 ;; *) under= ;; esac
if [ "$RC" = 0 ] && [ -n "$under" ] && [ "$(grep -c " target=$tgt files=0 " <<<"$one")" = 1 ]; then
  pass "deb builds in a new, empty target under DEB_TARGET, not in the target of make ci or run-fixtures"
else fail "deb builds in a new, empty target under DEB_TARGET, not in the target of make ci or run-fixtures" "exit $RC, $(tr '\n' ';' <<<"$one")"; fi
if [ "$(grep -c 'sccache=no DEB_SCCACHE=0$' <<<"$one")" = 1 ]; then pass "deb builds without sccache by default"
else fail "deb builds without sccache by default" "$(tr '\n' ';' <<<"$one")"; fi
if [ -n "$tgt" ] && [ ! -e "$tgt" ] && [ -z "$(ls -A "$CI_CACHE/deb-target")" ]; then pass "deb removes its target after the build"
else fail "deb removes its target after the build" "$(ls -A "$CI_CACHE/deb-target" | tr '\n' ' ')"; fi
if grep -q "release build of .*, target $tgt (new, empty), sccache 0" <<<"$OUT"; then pass "deb logs its new target and sccache 0"
else fail "deb logs its new target and sccache 0" "$(grep 'release build' <<<"$OUT")"; fi
EXTRA="FAKE_DEB_BUILD_RC=101" ci deb
if [ "$RC" = 1 ] && [ -z "$(ls -A "$CI_CACHE/deb-target")" ]; then pass "a failed release build removes its target too"
else fail "a failed release build removes its target too" "exit $RC, $(ls -A "$CI_CACHE/deb-target" | tr '\n' ' ')"; fi
EXTRA="DEB_SCCACHE=1" ci deb
if [ "$(builds 1 | grep -c 'sccache=yes DEB_SCCACHE=1$')" = 1 ]; then pass "DEB_SCCACHE=1 shares sccache with make ci again"
else fail "DEB_SCCACHE=1 shares sccache with make ci again" "$(builds 1 | tr '\n' ';')"; fi
EXTRA="DEB_TARGET=$DATA/elsewhere" ci deb
case $(target_of "$(builds 1)") in "$DATA/elsewhere/"?*) where=1 ;; *) where= ;; esac
if [ "$RC" = 0 ] && [ -n "$where" ] && [ -z "$(ls -A "$DATA/elsewhere")" ]; then pass "DEB_TARGET names where the new target goes"
else fail "DEB_TARGET names where the new target goes" "exit $RC, $(builds 1 | tr '\n' ';')"; fi
if [ "$(cat "$CI_CACHE/target/release/x0ne-infer" "$CI_CACHE/v2/target/release/x0ne-infer")" = "$(printf 'planted\nplanted')" ]; then
  pass "deb leaves the targets of make ci and run-fixtures alone"
else fail "deb leaves the targets of make ci and run-fixtures alone" "$(ls -A "$CI_CACHE/target/release" "$CI_CACHE/v2/target/release" | tr '\n' ' ')"; fi

# ---- deb-repro: two builds, each in its own new and empty target, neither with sccache.
ci deb-repro
one=$(builds 1); two=$(builds 2); t1=$(target_of "$one"); t2=$(target_of "$two")
if [ "$RC" = 0 ] && grep -q "green .*reproducible" <<<"$OUT"; then pass "deb-repro is green with identical .deb"
else fail "deb-repro is green with identical .deb" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
if [ -n "$t1" ] && [ -n "$t2" ] && [ "$t1" != "$t2" ] && [ "$(printf '%s\n%s\n' "$one" "$two" | grep -c ' files=0 sccache=no DEB_SCCACHE=0$')" = 2 ]; then
  pass "deb-repro: two builds in two new, empty targets, without sccache"
else fail "deb-repro: two builds in two new, empty targets, without sccache" "$(printf '%s\n%s' "$one" "$two" | tr '\n' ';')"; fi
if [ -z "$(ls -A "$CI_CACHE/deb-target")" ]; then pass "deb-repro removes both targets"
else fail "deb-repro removes both targets" "$(ls -A "$CI_CACHE/deb-target" | tr '\n' ' ')"; fi
EXTRA="DEB_REPRO_SCCACHE=1" ci deb-repro
if [ "$(builds 1 | grep -c 'sccache=no DEB_SCCACHE=0$')" = 1 ] && [ "$(builds 2 | grep -c 'sccache=yes DEB_SCCACHE=1$')" = 1 ]; then
  pass "deb-repro: DEB_REPRO_SCCACHE=1 shares sccache in the second build only"
else fail "deb-repro: DEB_REPRO_SCCACHE=1 shares sccache in the second build only" "$(grep '^inside-deb' "$T/runs.log" | tr '\n' ';')"; fi

# ---- the packaging, diffoscope and install-check containers run under CI_MEMORY and
# CI_CGROUP_PARENT, without network, as the builds do.
capped() { grep "$1" "$T/podman.log" | grep -q -- '--cgroup-parent=t.slice --memory=3g --network=none '; }
EXTRA="CI_MEMORY=3g CI_CGROUP_PARENT=t.slice" ci deb
if [ "$RC" = 0 ] && capped 'engine-deb-pkg-'; then pass "the packaging runs under CI_MEMORY and CI_CGROUP_PARENT"
else fail "the packaging runs under CI_MEMORY and CI_CGROUP_PARENT" "exit $RC, [$(grep -- 'engine-deb-pkg-' "$T/podman.log" | head -n 1)]"; fi
EXTRA="CI_MEMORY=3g CI_CGROUP_PARENT=t.slice FAKE_DEB_DIFFER=1" ci deb-repro
if [ "$RC" = 1 ] && grep -q "NOT reproducible" <<<"$OUT" && capped 'engine-diffoscope-'; then pass "diffoscope runs under CI_MEMORY and CI_CGROUP_PARENT when the .deb differ"
else fail "diffoscope runs under CI_MEMORY and CI_CGROUP_PARENT when the .deb differ" "exit $RC, [$(grep -- 'engine-diffoscope-' "$T/podman.log" | head -n 1)]"; fi
EXTRA="CI_MEMORY=3g CI_CGROUP_PARENT=t.slice" ci deb-check
if [ "$RC" = 0 ] && capped 'engine-deb-check-'; then pass "the install check runs under CI_MEMORY and CI_CGROUP_PARENT"
else fail "the install check runs under CI_MEMORY and CI_CGROUP_PARENT" "exit $RC, [$(grep -- 'engine-deb-check-' "$T/podman.log" | head -n 1)]"; fi

# ---- run dates the exported files now, not at the commit. Cargo's freshness is by mtime:
# a commit older than the last build on the shared target ran stale test binaries (2026-10-16).
echo old > "$REPO/old.txt"; git -C "$REPO" add old.txt
GIT_COMMITTER_DATE=2001-01-01T00:00:00Z GIT_AUTHOR_DATE=2001-01-01T00:00:00Z \
  git -C "$REPO" -c user.name=t -c user.email=t@t commit -qm old
OLD=$(git -C "$REPO" rev-parse HEAD); t0=$(date +%s)
ci run
# The file as mounted at /src, wherever the run keeps its source.
src=$(sed -n 's/^inside.sh src=\([^ ]*\) .*/\1/p' "$T/runs.log")
m=$(stat -c %Y "${src:-/nonexistent}/old.txt" 2>/dev/null || echo 0)
if [ "$RC" = 0 ] && [ "$m" -ge "$t0" ] && grep -q "run ${OLD:0:7} " <<<"$OUT"; then pass "run dates the exported files now, not at the commit"
else fail "run dates the exported files now, not at the commit" "exit $RC, mtime $m, run started $t0"; fi

# ---- the lock alone orders the builds. A cargo outside it does not stall make ci,
# and the lock is waited for at most CI_WAIT_MAX s.
LOCKF=$CI_CACHE/build.lock; mkdir -p "${LOCKF%/*}"
EXTRA="FAKE_PGREP=0 CI_WAIT_MAX=0" ci run
if [ "$RC" = 0 ] && grep -q 'bash /ci/inside.sh' "$T/podman.log"; then pass "run does not wait for a cargo that holds no lock"
else fail "run does not wait for a cargo that holds no lock" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
hold() { flock "$LOCKF" sleep "$1" & holder=$!; sleep 0.5; }
hold 4; EXTRA="CI_WAIT_MAX=1" ci run; wait "$holder" || true
if [ "$RC" = 2 ] && grep -q "build lock" <<<"$OUT" && ! ran_podman_run; then pass "run gives up on a lock held past CI_WAIT_MAX"
else fail "run gives up on a lock held past CI_WAIT_MAX" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
hold 2; EXTRA="CI_WAIT_MAX=30" ci run; wait "$holder" || true
if [ "$RC" = 0 ] && grep -Eq "lock: waited [1-9][0-9]* s" <<<"$OUT"; then pass "run waits for the lock and says how long"
else fail "run waits for the lock and says how long" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
# A clone elsewhere takes the same lock, not one next to the clone (2026-10-18).
git clone -q "$REPO" "$T/elsewhere/clone"
hold 4; AT=$T/elsewhere/clone EXTRA="CI_WAIT_MAX=1" ci run; wait "$holder" || true
if [ "$RC" = 2 ] && [ -z "$(find "$T/elsewhere" -name '*build.lock' -print -quit)" ]; then pass "a clone elsewhere takes the shared lock"
else fail "a clone elsewhere takes the shared lock" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi

# ---- one verdict per tree, from any worktree. probe.txt stands for a source file: the fake
# inside.sh rebuilds when it is newer than the last build (cargo's rule) and is red when it says red.
echo green > "$REPO/probe.txt"; git -C "$REPO" add probe.txt
git -C "$REPO" -c user.name=t -c user.email=t@t commit -qm probe
PROBE=$(git -C "$REPO" rev-parse HEAD)
git -C "$REPO" worktree add -q --detach "$T/wt-b" HEAD
git -C "$REPO" worktree add -q --detach "$T/wt-c" HEAD
# Edited before the other worktree's build: older than that build, as in a past CI run.
echo red > "$T/wt-b/probe.txt"; echo new > "$T/wt-b/new.txt"; rm "$T/wt-b/old.txt"
built() { sed -n 's/^inside.sh .* built=\([a-z]*\)$/\1/p' "$T/runs.log"; }
srcdir() { sed -n 's/^inside.sh src=\([^ ]*\) .*/\1/p' "$T/runs.log"; }
ci run; ci run
if [ "$RC" = 0 ] && [ "$(built)" = no ]; then pass "run on an unchanged commit rebuilds nothing"
else fail "run on an unchanged commit rebuilds nothing" "exit $RC, built $(built)"; fi
AT=$T/wt-c ci run
if [ "$RC" = 0 ] && [ "$(built)" = no ]; then pass "the same commit from another worktree: same verdict, nothing rebuilt"
else fail "the same commit from another worktree: same verdict, nothing rebuilt" "exit $RC, built $(built)"; fi
EXTRA="CI_WORKTREE=1" ci run
if [ "$RC" = 0 ]; then pass "CI_WORKTREE=1 on the clean worktree is green"
else fail "CI_WORKTREE=1 on the clean worktree is green" "exit $RC, built $(built)"; fi
AT=$T/wt-b EXTRA="CI_WORKTREE=1" ci run
s=$(srcdir)
if [ "$RC" = 1 ] && [ "$(built)" = yes ]; then pass "a worktree edited before another tree's build is rebuilt, and red"
else fail "a worktree edited before another tree's build is rebuilt, and red" "exit $RC, built $(built)"; fi
if [ -s "$s/new.txt" ] && [ ! -e "$s/old.txt" ] && [ ! -e "$s/out" ]; then pass "CI_WORKTREE=1 tests the worktree's files: untracked in, deleted and ignored out"
else fail "CI_WORKTREE=1 tests the worktree's files: untracked in, deleted and ignored out" "src $s: $(ls -A "$s" | tr '\n' ' ')"; fi
if [ -e "$T/wt-b/out/ci/$PROBE+dirty.json" ] && [ ! -e "$T/wt-b/out/ci/$PROBE.json" ] && grep -q "${PROBE:0:7}+dirty" <<<"$OUT"; then
  pass "a worktree with changes gets a +dirty verdict, not the commit's"
else fail "a worktree with changes gets a +dirty verdict, not the commit's" "$(ls "$T/wt-b/out/ci" 2>/dev/null | tr '\n' ' ')"; fi
EXTRA="CI_WORKTREE=1" ci run
if [ "$RC" = 0 ] && [ "$(built)" = yes ]; then pass "back on the first worktree: rebuilt, green again"
else fail "back on the first worktree: rebuilt, green again" "exit $RC, built $(built)"; fi
# Changes git status does not show: a file marked --skip-worktree or --assume-unchanged,
# an untracked file under status.showUntrackedFiles=no. They are tested, so the verdict is +dirty.
verdict() { readlink "$1/out/ci/latest.json" || true; }
objects() { git -C "$REPO" count-objects -v | sed -n 's/^\(count\|in-pack\): //p' | tr '\n' ' '; }
before=$(objects)
for flag in skip-worktree assume-unchanged; do
  git -C "$T/wt-c" update-index "--$flag" probe.txt; echo red > "$T/wt-c/probe.txt"
  AT=$T/wt-c EXTRA="CI_WORKTREE=1" ci run
  if [ "$RC" = 1 ] && [ "$(verdict "$T/wt-c")" = "$PROBE+dirty.json" ]; then pass "a changed file marked --$flag gets a +dirty verdict"
  else fail "a changed file marked --$flag gets a +dirty verdict" "exit $RC, latest $(verdict "$T/wt-c"), status [$(git -C "$T/wt-c" status --porcelain)]"; fi
  git -C "$T/wt-c" update-index "--no-$flag" probe.txt; git -C "$T/wt-c" checkout -q probe.txt
done
git -C "$REPO" config status.showUntrackedFiles no; echo x > "$T/wt-c/forgotten.rs"
AT=$T/wt-c EXTRA="CI_WORKTREE=1" ci run
if [ "$(verdict "$T/wt-c")" = "$PROBE+dirty.json" ]; then pass "an untracked file hidden from git status gets a +dirty verdict"
else fail "an untracked file hidden from git status gets a +dirty verdict" "exit $RC, latest $(verdict "$T/wt-c")"; fi
if [ "$(objects)" = "$before" ]; then pass "comparing the tree with the commit writes no object to the repository"
else fail "comparing the tree with the commit writes no object to the repository" "before [$before], after [$(objects)]"; fi
git -C "$REPO" config --unset status.showUntrackedFiles; rm "$T/wt-c/forgotten.rs"
AT=$T/wt-c EXTRA="CI_WORKTREE=1" ci run
if [ "$RC" = 0 ] && [ "$(verdict "$T/wt-c")" = "$PROBE.json" ]; then pass "the same worktree back at its commit gets the commit's verdict"
else fail "the same worktree back at its commit gets the commit's verdict" "exit $RC, latest $(verdict "$T/wt-c")"; fi

# ---- the one target is emptied under the lock once past CI_TARGET_MAX_GB.
tgt=$(sed -n 's/^inside.sh .* target=\([^ ]*\) .*/\1/p' "$T/runs.log")
head -c 2097152 /dev/zero > "$tgt/bulk"
EXTRA="CI_TARGET_MAX_GB=0" ci run
if [ "$RC" = 0 ] && [ ! -e "$tgt/bulk" ] && grep -q "target over" <<<"$OUT"; then pass "a target past CI_TARGET_MAX_GB is emptied before the build"
else fail "a target past CI_TARGET_MAX_GB is emptied before the build" "exit $RC, $(ls -A "$tgt" | tr '\n' ' ')"; fi
ci run
if [ -e "$tgt/.built" ] && grep -q -- '-e CARGO_PROFILE_DEV_DEBUG=line-tables-only ' "$T/podman.log"; then pass "run builds with line-tables-only debug info"
else fail "run builds with line-tables-only debug info" "$(grep inside.sh "$T/podman.log")"; fi
EXTRA="CI_DEBUGINFO=2" ci run
if grep -q -- '-e CARGO_PROFILE_DEV_DEBUG=2 ' "$T/podman.log"; then pass "CI_DEBUGINFO=2 restores full debug info"
else fail "CI_DEBUGINFO=2 restores full debug info" "$(grep inside.sh "$T/podman.log")"; fi

# ---- run-fixtures: its inputs are required; a different set reaches the tree only with RUN_WRITE=1.
echo '{}' > "$T/att.json"; echo '{}' > "$T/ref.json"
RF="RUN_ATTESTATION=$T/att.json RUN_REFERENCE=$T/ref.json RUN_DATE=2026-10-17 RUN_IMAGE=x"
EXTRA="RUN_ATTESTATION=$T/att.json RUN_DATE=2026-10-17 RUN_IMAGE=x" ci run-fixtures
if [ "$RC" = 2 ] && grep -q "RUN_REFERENCE" <<<"$OUT" && ! ran_podman_run; then pass "run-fixtures stops without a reference"
else fail "run-fixtures stops without a reference" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
EXTRA="${RF/2026-10-17/17-10-2026}" ci run-fixtures
if [ "$RC" = 2 ] && grep -q "expected YYYY-MM-DD" <<<"$OUT" && ! ran_podman_run; then pass "run-fixtures stops on a date not YYYY-MM-DD"
else fail "run-fixtures stops on a date not YYYY-MM-DD" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
EXTRA="$RF CI_CGROUP_PARENT=test.slice" ci run-fixtures
if [ "$RC" = 0 ] && grep -q "identical to the working tree" <<<"$OUT" \
  && grep -q -- '--network=none' "$T/podman.log" && grep -q -- '--cgroup-parent=test.slice' "$T/podman.log" \
  && grep -q "src=$REPO$" "$T/runs.log"; then pass "run-fixtures runs the example offline, under CI_CGROUP_PARENT"
else fail "run-fixtures runs the example offline, under CI_CGROUP_PARENT" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
if ! grep -q -- '--test-key' "$T/podman.log" && ! grep -q '/in/reference.json.sig' "$T/podman.log"; then pass "run-fixtures accepts no TEST key and mounts no absent .sig"
else fail "run-fixtures accepts no TEST key and mounts no absent .sig" "$(cat "$T/podman.log")"; fi
echo sig > "$T/ref.json.sig"
EXTRA="$RF RUN_TEST_KEY=1" ci run-fixtures
if [ "$RC" = 0 ] && grep -q -- ' a3-highgpu-1g --test-key$' "$T/podman.log" \
  && grep -q -- "-v $T/ref.json.sig:/in/reference.json.sig:ro" "$T/podman.log"; then pass "run-fixtures passes --test-key with RUN_TEST_KEY=1 and mounts the .sig"
else fail "run-fixtures passes --test-key with RUN_TEST_KEY=1 and mounts the .sig" "exit $RC, $(cat "$T/podman.log")"; fi
rm -f "$T/ref.json.sig"
EXTRA="$RF FAKE_RF_RC=1" ci run-fixtures
if [ "$RC" = 1 ] && [ ! -e "$REPO/measurements/fake.json" ]; then pass "run-fixtures leaves the tree alone without RUN_WRITE"
else fail "run-fixtures leaves the tree alone without RUN_WRITE" "exit $RC"; fi
EXTRA="$RF FAKE_RF_RC=3 RUN_WRITE=1" ci run-fixtures
if [ "$RC" = 1 ] && grep -q "refused" <<<"$OUT" && [ ! -e "$REPO/measurements/fake.json" ]; then pass "run-fixtures writes nothing for a refused run"
else fail "run-fixtures writes nothing for a refused run" "exit $RC"; fi
EXTRA="$RF FAKE_RF_RC=1 RUN_WRITE=1" ci run-fixtures
if [ "$RC" = 0 ] && [ -s "$REPO/measurements/fake.json" ]; then pass "run-fixtures copies a different set with RUN_WRITE=1"
else fail "run-fixtures copies a different set with RUN_WRITE=1" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi

# ---- run-fixtures never replaces a file of the tree silently: the measurements of
# a run, or the fixtures of a RUN_DATE already taken, only with RUN_OVERWRITE=1. A
# measurements file that names no register and no image is a placeholder: it is filled.
EXTRA="$RF FAKE_RF_RC=1 RUN_WRITE=1 FAKE_RF_BODY={\"version\":\"other\"}" ci run-fixtures
if [ "$RC" = 1 ] && grep -q "measurements/fake.json" <<<"$OUT" && grep -q "RUN_OVERWRITE=1" <<<"$OUT" \
  && [ "$(cat "$REPO/measurements/fake.json")" = '{"version":"fake"}' ]; then pass "run-fixtures refuses to replace measurements"
else fail "run-fixtures refuses to replace measurements" "exit $RC, $(cat "$REPO/measurements/fake.json"), $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi
CCEL=x0ne-infer/tests/ccel/gce-a3-tdx-2026-10-17.bin
mkdir -p "$REPO/${CCEL%/*}"; echo taken > "$REPO/$CCEL"
EXTRA="$RF FAKE_RF_RC=1 RUN_WRITE=1 FAKE_RF_FILE=$CCEL FAKE_RF_BODY=other-boot" ci run-fixtures
if [ "$RC" = 1 ] && grep -q "$CCEL" <<<"$OUT" && [ "$(cat "$REPO/$CCEL")" = taken ]; then pass "run-fixtures refuses to replace the fixtures of a RUN_DATE already taken"
else fail "run-fixtures refuses to replace the fixtures of a RUN_DATE already taken" "exit $RC, $(cat "$REPO/$CCEL")"; fi
EXTRA="$RF FAKE_RF_RC=1 RUN_WRITE=1 RUN_OVERWRITE=1 FAKE_RF_FILE=$CCEL FAKE_RF_BODY=other-boot" ci run-fixtures
if [ "$RC" = 0 ] && [ "$(cat "$REPO/$CCEL")" = other-boot ]; then pass "run-fixtures replaces them with RUN_OVERWRITE=1"
else fail "run-fixtures replaces them with RUN_OVERWRITE=1" "exit $RC, $(cat "$REPO/$CCEL")"; fi
printf '{\n  "format": 1,\n  "image": null,\n  "registers": {},\n  "signature": null\n}\n' > "$REPO/measurements/fake.json"
EXTRA="$RF FAKE_RF_RC=1 RUN_WRITE=1 FAKE_RF_BODY={\"version\":\"measured\"}" ci run-fixtures
if [ "$RC" = 0 ] && [ "$(cat "$REPO/measurements/fake.json")" = '{"version":"measured"}' ]; then pass "run-fixtures fills a placeholder measurements file"
else fail "run-fixtures fills a placeholder measurements file" "exit $RC, $(tail -n 2 <<<"$OUT" | tr '\n' ' ')"; fi

# ---- post-merge: the slice and the cache come from the environment or ci.env, none by default.
printf '#!/bin/sh\necho "$*" > "%s"\n' "$T/systemd-run.log" > "$BIN/systemd-run"; chmod +x "$BIN/systemd-run"
hook() {
  : > "$T/systemd-run.log"
  (cd "$REPO" && env -i HOME="$T/home" PATH="$BIN:/usr/local/bin:/usr/bin:/bin" ${EXTRA:-} \
    bash ci/hooks/post-merge > /dev/null 2>&1) || true
}
hook
if grep -q -- '--unit=engine-ci-' "$T/systemd-run.log" && ! grep -q -- '--slice' "$T/systemd-run.log" \
  && ! grep -q 'CI_CACHE=' "$T/systemd-run.log"; then pass "post-merge on main starts the CI with no slice and no cache of its own"
else fail "post-merge on main starts the CI with no slice and no cache of its own" "[$(cat "$T/systemd-run.log")]"; fi
EXTRA="CI_SLICE=a.slice CI_CACHE=/c" hook
if grep -q -- '--slice=a.slice ' "$T/systemd-run.log" && grep -q -- '--setenv=CI_CACHE=/c ' "$T/systemd-run.log"; then pass "post-merge passes CI_SLICE and CI_CACHE from the environment"
else fail "post-merge passes CI_SLICE and CI_CACHE from the environment" "[$(cat "$T/systemd-run.log")]"; fi
printf 'CI_SLICE=b.slice\n' > "$ENVF"; hook; rm -f "$ENVF"
if grep -q -- '--slice=b.slice ' "$T/systemd-run.log"; then pass "post-merge reads CI_SLICE from ci.env"
else fail "post-merge reads CI_SLICE from ci.env" "[$(cat "$T/systemd-run.log")]"; fi
git -C "$REPO" switch -q -c topic; hook; git -C "$REPO" switch -q main
if [ ! -s "$T/systemd-run.log" ]; then pass "post-merge leaves a branch outside x0ne.ci.branches alone"
else fail "post-merge leaves a branch outside x0ne.ci.branches alone" "[$(cat "$T/systemd-run.log")]"; fi

# ---- clippy-pedantic: the lint list of the GitHub clippy job, which ci/inside.sh lints with.
CP=$SRC/ci/clippy-pedantic.sh
YML=$SRC/.github/workflows/ci.yml
out=$(bash "$CP" "$YML" 2>&1) || out="exit $?: $out"
if [ "$(sed -n 1,2p <<<"$out" | tr '\n' ' ')" = "-W clippy::pedantic " ] \
  && [ "$(grep -cx -- -A <<<"$out")" -gt 0 ] \
  && [ "$(grep -cx -- -A <<<"$out")" = "$(grep -cE '^ +-A clippy::[a-z0-9_]+$' "$YML")" ]; then
  pass "clippy-pedantic reads the PEDANTIC list of ci.yml ($(grep -cx -- -A <<<"$out") lints allowed)"
else fail "clippy-pedantic reads the PEDANTIC list of ci.yml" "[$(tr '\n' ' ' <<<"$out")]"; fi
PY=$T/pedantic.yml
printf '%b\n' 'jobs:\n  c:\n    env:\n      PEDANTIC: >-\n        -W clippy::pedantic\n\n        -A clippy::a_b\n    steps:\n      - run: x' > "$PY"
out=$(bash "$CP" "$PY" 2>&1 | tr '\n' ' ') || true
if [ "$out" = "-W clippy::pedantic -A clippy::a_b " ]; then pass "clippy-pedantic stops at the end of the block"
else fail "clippy-pedantic stops at the end of the block" "[$out]"; fi
# pedantic_refuses <case> <yaml, printf %b>: exit 2, nothing on stdout.
pedantic_refuses() {
  printf '%b\n' "$2" > "$PY"
  local rc=0 o
  o=$(bash "$CP" "$PY" 2>/dev/null) || rc=$?
  if [ "$rc" = 2 ] && [ -z "$o" ]; then pass "clippy-pedantic refuses $1"
  else fail "clippy-pedantic refuses $1" "exit $rc, [$o]"; fi
}
pedantic_refuses "a file with no PEDANTIC block" '    env:\n      OTHER: 1'
pedantic_refuses "two PEDANTIC blocks" '      PEDANTIC: >-\n        -W clippy::pedantic\n      PEDANTIC: >-\n        -W clippy::pedantic'
pedantic_refuses "a word that is no lint flag (--cap-lints allow)" '      PEDANTIC: >-\n        -W clippy::pedantic\n        --cap-lints allow'
pedantic_refuses "a list that does not open on -W clippy::pedantic" '      PEDANTIC: >-\n        -A clippy::pedantic'
pedantic_refuses "a flag with no lint" '      PEDANTIC: >-\n        -W clippy::pedantic -A'
pedantic_refuses "a lint name with a shell character" '      PEDANTIC: >-\n        -W clippy::pedantic\n        -A clippy::x;id'
rc=0; bash "$CP" "$T/no-such.yml" > /dev/null 2>&1 || rc=$?
if [ "$rc" = 2 ]; then pass "clippy-pedantic refuses a missing file"; else fail "clippy-pedantic refuses a missing file" "exit $rc"; fi

# ---- repro-compare: two build trees, byte for byte (its own cases).
if bash "$SRC/ci/tests/repro-compare.sh" > "$T/repro-compare.log" 2>&1; then pass "repro-compare: $(tail -n 1 "$T/repro-compare.log")"
else fail "repro-compare" "$(grep FAIL "$T/repro-compare.log" | tr '\n' ' ')"; fi

# ---- strip-release, deb-verify, release-guards: the strip of the shipped binaries, the check of the
# .deb, the build environment and the release refusal of x0ne-infer (their own cases).
for t in strip-release deb-verify release-guards; do
  if bash "$SRC/ci/tests/$t.sh" > "$T/$t.log" 2>&1; then pass "$t: $(tail -n 1 "$T/$t.log")"
  else fail "$t" "$(grep FAIL "$T/$t.log" | tr '\n' ' ')"; fi
done

[ "$FAILS" = 0 ] || { echo "ci-sh: $FAILS failed"; exit 1; }
echo "ci-sh: all passed"
