#!/usr/bin/env bash
# shellcheck disable=SC2016 # the scripts in single quotes run inside the containers
# install.sh in disposable containers (make install-check): ci/tests/installer.sh as an
# ordinary user, without network, in Debian 13, Ubuntu 24.04 LTS, Fedora 43, and Alpine 3,
# where the release binaries cannot start (no glibc loader) and only the refusal is tested.
# The Debian container also runs shellcheck on the installer and its tests.
#
#   ci/install-matrix.sh [debian13|ubuntu2404|fedora43|alpine3 ...]     default: all four
#
# ENGINE_ARCHIVE=<x0ne-engine-x86_64-unknown-linux-gnu.tar.gz>: the good release serves that
# archive, so the real binaries are installed and run. ENGINE_DEB=<.deb>: also dpkg -i it as
# root in Debian and Ubuntu, run its three binaries, dpkg -r it. CONTAINER: podman (default)
# or docker (GitHub Actions). CI_CGROUP_PARENT: with podman, the cgroup of every container.
# The client images (ci/Containerfile.install) are built once per base, with network.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
RUN=${CONTAINER:-podman}
# Index digests (multi-architecture), read on 2026-11-07.
declare -A BASE=(
  [debian13]=docker.io/library/debian@sha256:918311b7b6c4c6f68b232ba516584925f6c78ad82b6fd534b98979df6438e483
  [ubuntu2404]=docker.io/library/ubuntu@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
  [fedora43]=docker.io/library/fedora@sha256:f01112b9a1f1d0504ad962d9adfa401425033f8ab067e0a821904ebdde5b8ca5
  [alpine3]=docker.io/library/alpine@sha256:294b683cb724975bec92580e1e685676bd4b50bda910ddb8c51d4cabeaec77e6
)
ORDER=(debian13 ubuntu2404 fedora43 alpine3)
distros=("$@")
[ ${#distros[@]} -gt 0 ] || distros=("${ORDER[@]}")
for d in "${distros[@]}"; do
  [ -n "${BASE[$d]:-}" ] || { echo "install-matrix: unknown distribution $d (known: ${ORDER[*]})" >&2; exit 2; }
done

opts=(--rm --network=none)
if [ "$RUN" = podman ] && [ -n "${CI_CGROUP_PARENT:-}" ]; then opts+=(--cgroup-parent="$CI_CGROUP_PARENT"); fi
mounts=(-v "$SRC:/src:ro")
envs=()
if [ -n "${ENGINE_ARCHIVE:-}" ]; then
  mounts+=(-v "$(realpath "$ENGINE_ARCHIVE"):/release/archive.tar.gz:ro")
  envs+=(-e ENGINE_ARCHIVE=/release/archive.tar.gz)
fi
if [ -n "${ENGINE_DEB:-}" ]; then
  mounts+=(-v "$(realpath "$ENGINE_DEB"):/release/x0ne-engine.deb:ro")
fi

fails=0
# step <label> <command...>: prints the output indented, then one verdict line.
step() {
  local label=$1 out rc=0
  shift
  out=$("$@" 2>&1) || rc=$?
  printf '%s\n' "$out" | sed 's/^/  /'
  if [ "$rc" = 0 ]; then echo "PASS  $label"; else echo "FAIL  $label (exit $rc)"; fails=$((fails + 1)); fi
}

for d in "${distros[@]}"; do
  base=${BASE[$d]}
  key=$({ cat "$SRC/ci/Containerfile.install"; echo "$base"; } | sha256sum | cut -c1-12)
  img=localhost/x0ne-engine-install-$d:$key
  if ! "$RUN" image inspect "$img" >/dev/null 2>&1; then
    echo "install-matrix: building $img"
    "$RUN" build -q --build-arg BASE="$base" -t "$img" -f "$SRC/ci/Containerfile.install" "$SRC/ci" >/dev/null
  fi
  name=x0ne-engine-install-$d-$$
  step "$d: install.sh" "$RUN" run "${opts[@]}" --name "$name" "${mounts[@]}" "${envs[@]}" "$img" \
    sh /src/ci/tests/installer.sh
  if [ "$d" = debian13 ]; then
    step "$d: shellcheck" "$RUN" run "${opts[@]}" --name "$name-shellcheck" "${mounts[@]}" "$img" sh -euc '
      shellcheck --version | sed -n 2p
      shellcheck --shell=sh /src/install.sh /src/ci/tests/installer.sh
      shellcheck /src/ci/install-matrix.sh /src/ci/deb-pack.sh /src/ci/tests/deb-pack.sh
      echo "shellcheck: clean"'
  fi
  if [ -n "${ENGINE_DEB:-}" ] && { [ "$d" = debian13 ] || [ "$d" = ubuntu2404 ]; }; then
    step "$d: dpkg -i the .deb" "$RUN" run "${opts[@]}" --name "$name-deb" --user 0 "${mounts[@]}" "$img" sh -euc '
      dpkg -i /release/x0ne-engine.deb >/tmp/dpkg.log 2>&1 || { cat /tmp/dpkg.log; exit 3; }
      dpkg -s x0ne-engine | grep -E "^(Status|Version):"
      for b in x0ne-engine x0ne-engine-cloud x0ne-engine-proxy; do "/usr/bin/$b" --version; done
      cat /usr/share/x0ne-engine/x0ne-engine.version
      dpkg -r x0ne-engine >/dev/null
      if [ -e /usr/bin/x0ne-engine ]; then echo "dpkg -r left /usr/bin/x0ne-engine"; exit 4; fi
      echo "installed, ran, removed"'
  fi
done
[ "$fails" = 0 ] || { echo "install-matrix: $fails failed"; exit 1; }
echo "install-matrix: all passed (${distros[*]})"
