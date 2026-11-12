#!/usr/bin/env bash
# cargo's runner for the test steps of ci/inside.sh (CARGO_TARGET_<host>_RUNNER): runs a test
# binary as cargo would, with libtest printing the wall clock of each test ("test <name> ... ok
# <s>"), then prints the CPU time of the binary and of what it started:
#   test-runner: user <s> s, sys <s> s, wall <s> s: <binary>
# Both stay in the step's log.
# Usage: bash ci/test-runner.sh <test binary> [libtest arguments...]; exits with the binary's code.
#
# --report-time is an unstable option of libtest: a stable toolchain takes it under
# RUSTC_BOOTSTRAP=1. It is set here, for the test binary and what it starts, and never for
# cargo, rustc or a build script, which would see a nightly compiler and could build other code
# than the one that ships. No test of the workspace starts cargo or rustc. Setting the runner
# rebuilds nothing: cargo uses it to run, not to build (checked on 2026-11-11).
TIMEFORMAT="test-runner: user %3U s, sys %3S s, wall %3R s: $1"
time RUSTC_BOOTSTRAP=1 "$@" -Z unstable-options --report-time
