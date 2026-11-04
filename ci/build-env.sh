#!/usr/bin/env bash
# The environment one build ran in, as JSON on stdout, for the build-env.json
# that ci/repro-compare.sh reports next to each tree. Describes, proves nothing.
# Usage: ci/build-env.sh <build dir> [<image>...]   (images: name and ID each)
set -euo pipefail
dir=${1:?usage: ci/build-env.sh <build dir> [<image>...]}
shift
q() { printf '"%s"' "$(printf '%s' "$1" | tr -d '"\\' | tr '\n\t' '  ')"; }
os=$( (. /etc/os-release 2>/dev/null && printf '%s' "${PRETTY_NAME:-unknown}") || echo unknown)
cpu=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1)
fs=$(df -T -- "$dir" 2>/dev/null | awk 'NR == 2 {print $2}')
podman_v=$(podman --version 2>/dev/null || echo none)
images=""
for i in "$@"; do
  id=$(podman image inspect --format '{{.Id}}' "$i" 2>/dev/null || echo absent)
  images="$images${images:+, }{\"name\": $(q "$i"), \"id\": $(q "$id")}"
done
printf '{"host": %s, "kernel": %s, "os": %s, "cpu": %s, "nproc": %s, "uid": %s, "locale": %s, "tz": %s, "fs": %s, "podman": %s, "images": [%s]}\n' \
  "$(q "$(uname -n)")" "$(q "$(uname -r)")" "$(q "$os")" "$(q "$cpu")" "$(nproc)" "$(id -u)" \
  "$(q "${LC_ALL:-${LANG:-unset}}")" "$(q "$(date +%Z%z)")" "$(q "${fs:-unknown}")" "$(q "$podman_v")" "$images"
