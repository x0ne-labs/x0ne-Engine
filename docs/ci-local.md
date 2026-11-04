# Local CI: `make ci`

Runs the merge gate of CONTRIBUTING.md on the committed state of `HEAD`, in a
throwaway container with no network. Red means the merge is not finished.

```
make ci              # fmt, ci/ci.sh tests, FEATURES.toml and FEATURES.md, clippy pedantic as GitHub runs it (default
                     # and all features but cuda), test, doctests, doc -D warnings, cargo deny, gitleaks,
                     # comment invariance of the .deb binaries
make ci-status       # last verdict
make ci-image        # build ci/Containerfile (every input pinned)
make ci-refresh      # re-vendor the crates, refetch the RustSec database
make install-hooks   # post-merge hook: make ci && make deb in the background after a merge
make deb             # stripped release build + the x0ne-engine .deb (the client), staged (not published)
make deb-repro       # make deb, then a second build: the .deb must be identical
make deb-check       # dpkg -i of the staged .deb in a trixie container, --version names HEAD
make repro-compare A=<dir> B=<dir>  # two builds of one commit, from two environments, byte for byte
make deb-status      # last deb verdict
make coverage        # test coverage per crate, held to ci/coverage-baseline.json; not part of make ci
make coverage-update # raise the baseline to the measure of HEAD
```

## What runs where

- `ci/ci.sh` (host): exports `HEAD` with `git archive` (not the working tree:
  what is tested is what is committed; `CI_WORKTREE=1` takes the working
  tree's tracked and untracked files instead, ignored ones left out), vendors
  the crates of `Cargo.lock` into `$CI_CACHE/vendor/<lock hash>` and fetches
  the RustSec database when older than a day. The verdict carries the age of
  the database the `deny` step judged against (`advisory_db_age_s`); a fetch
  that fails leaves the old copy and is said with that age, so that
  "advisories ok" is never read as fresher than it is. These two steps are the only
  ones with network, in their own container, under the same `--memory=$CI_MEMORY`
  and `--cgroup-parent=$CI_CGROUP_PARENT` (when set) as the build. It then takes the lock shared
  with other Rust builds of the host (`CI_LOCK`, default `$CI_CACHE/build.lock`,
  waited for at most `CI_WAIT_MAX` s, the wait in the verdict) and nothing
  else: a `cargo` that holds no lock is not waited for.
- Under the lock it copies the tree into `$CI_CACHE/v2/src`, rewriting only
  the files whose content differs. Cargo judges freshness by modification
  time: a changed file is always newer than the last build, an unchanged one
  keeps its date. So one target (`$CI_CACHE/v2/target`) serves every worktree
  and clone, an unchanged tree rebuilds nothing, and two worktrees at one
  commit get the same verdict. Mounting each worktree on a shared target ran
  another tree's binaries when a tree was older than the last build.
- `ci/inside.sh` (container, `--network=none`, `--memory=$CI_MEMORY` (16g),
  `--cgroup-parent=$CI_CGROUP_PARENT` when set, `nice -n 10`): mounts `$CI_CACHE/v2/src`
  read-only, the vendored crates and the advisory database read-only, `$CI_CACHE/v2/target`
  and the sccache disk cache read-write (sccache is used when `SCCACHE_BIN`, `sccache` on
  `PATH` by default, and `SCCACHE_DISK`, `$CI_CACHE/sccache` by default, exist). One log per
  step under `$CI_CACHE/logs/`. The steps run in four lanes at once (`CI_LANES=0`: one
  after the other): `fmt ci-sh features deny gitleaks` (no build), `clippy test test-doc`,
  `clippy-all doc test-all` (in `target/all`) and `test-release build-cloud
  comment-invariance release-refusal`. `clippy` and `clippy-all` lint as the `clippy` job
  of `.github/workflows/ci.yml` does: `-D warnings`, then its `PEDANTIC` list
  (`clippy::pedantic` minus the lints it allows), read from the commit by
  `ci/clippy-pedantic.sh`, so the two CIs share one list; a list the script cannot read
  (no single `PEDANTIC: >-` block, a word outside `-W|-A|-D clippy::<lint>` pairs, a
  first pair other than `-W clippy::pedantic`) fails the step. `test-release` runs the
  tests in the profile that ships (`--release`, `target/release`).
  `build-cloud` is the client build of [cloud.md](cloud.md#building), with its
  `x0ne-engine cloud --help`. `comment-invariance` (`ci/comment-invariance.sh`)
  builds the binaries of the `.deb` packages as `make deb` does (profile
  `release`, `target/release`), from a copy of the tree, appends a comment line at the
  end of `x0ne-infer/src/server.rs`, `x0ne-engine-cloud/src/main.rs` and
  `x0ne-confidential/src/lib.rs`, builds again and requires the same sha256 for
  each binary; a difference is red, with the sections that moved. See
  [Stripped binaries](#stripped-binaries). `release-refusal` (`ci/release-refusal.sh`)
  builds `x0ne-infer` with the profile `release` and its default features and runs
  it on 22 cases, each with the exit code and the stderr text it requires: no
  argument (87, `refused by a release build`), the CPU asked to serve (90), each
  variable `x0ne-infer/src/harden.rs` refuses (73), `--device cuda:1` and `cpu:0`
  (81), `--attest gpu` and `CPU` (82), the lab server's flags (2), a session key or
  a pin in cloud mode (95): see [One recipe](#one-recipe). No `.deb` and no release
  archive ships `x0ne-infer`
  since 2026-11-04; the CVM image builds it with `cuda nvml`, which needs nvcc, and
  the release switch is the same code without them. Cargo serialises the builds of one profile on
  its own lock and releases it before the tests run, so one lane compiles while the other
  runs its tests; a step's seconds include the time it waited for the other lane's build,
  `duration_s` is the wall clock of the run. Same steps, same commands, same test binaries
  as the sequential run: lanes change when a step runs, not what it checks.
- Result: `out/ci/<sha>.json` (`status`, `failed`, one entry per step with its
  duration and exit code, `tree`: a hash of the files tested, `lock_wait_s`),
  `out/ci/<sha>+dirty.json` instead when `CI_WORKTREE=1` finds changes
  (`"dirty": true`), `out/ci/latest.json` and one journal line
  (`journalctl -t engine-ci`). `CI_NOTIFY_CMD`, when set, is run after each
  verdict as `<cmd> ci|deb <green|red> <short sha> <seconds> <detail>`, the
  detail being the failed steps (`ci`) or the packages (`deb`); its failure
  does not change the verdict. Nothing else notifies.
  Exit codes: 0 green, 1 red, 2 setup (disk, image, vendoring).

The settings of one host (cache disk, lock, cgroup, sccache, `X0NE_OS`,
`CI_NOTIFY_CMD`) can live in `CI_ENV_FILE`, by default
`${XDG_CONFIG_HOME:-~/.config}/x0ne-engine/ci.env`: one `KEY=value` per line,
`#` for comments, only the `CI_*`, `DEB_*`, `SCCACHE_BIN`, `SCCACHE_DISK` and
`X0NE_OS` keys. The file is read, not run, and the environment wins over it.

Steps and their commands are in `ci/inside.sh`; `CI_STEPS="fmt clippy"`
restricts a run (the lanes keep the requested steps in their order). `ci-sh` runs `ci/tests/ci-sh.sh`: `ci/ci.sh` against fake
`podman`, `df` and `jq` in a throwaway repository (no build, no network), then
`ci/tests/release-verify-tag.sh`: the tag verification step of `.github/workflows/release.yml`
against a fake `gh api` and the image's `jq` (by hand without `jq` it skips; under
`ci/inside.sh`, `X0NE_REQUIRE_JQ=1`, it fails).
Before any container, `ci/ci.sh` requires `CI_MIN_DISK_GB` (150) free on `/`
and on the filesystems of `CI_CACHE`, `DEB_TARGET` and `DEB_STAGING`
(all under `CI_CACHE` by default). The per-feature test matrix (nvml, ...) stays
in `.github/workflows/ci.yml`; here every feature but `cuda` is tested at
once, which is the gate, not the matrix. On GitHub, the `test`, `msrv` and
`stable` jobs build with `CARGO_PROFILE_DEV_DEBUG=line-tables-only` and first
free the runner's disk, as `image` does. The `test` job's eight steps,
replayed offline in a fresh `target` with that setting and
`CARGO_INCREMENTAL=0` (2026-11-03): 5.8 GiB after the first, 17.6 GiB after
the fifth, 27.0 GiB after the last. With full debug info, the first five
had filled 67 GB (measured the same day).

## Image

`ci/Containerfile`: `rust:1.98.1-slim-bookworm` by digest (the toolchain of
`rust-toolchain.toml`), cargo-deny 0.20.2 by release checksum, gitleaks
v8.30.1 copied from the image digest pinned in `.github/workflows/security.yml`.
The tag is the hash of the file, so a change to it rebuilds the image on the
next `make ci`. The apt packages (g++, git, curl, jq, pkg-config) are not snapshot
pinned.

## Hook

`ci/hooks/post-merge` starts `make ci && make deb` as a transient user unit
(`systemd-run --user`, in the slice `CI_SLICE` when set, from the environment
or `CI_ENV_FILE`), for the branches listed in
`git config x0ne.ci.branches` (default `main`), and passes
the merged range (`ORIG_HEAD..HEAD`) to gitleaks as patches; a red CI stops the chain, so no
package is staged for a commit that failed the gate (`git config x0ne.ci.deb
false` skips the package). `make install-hooks` copies it into the hooks directory and refuses to replace a
different hook already there.

## Package: `make deb`

`ci/ci.sh deb` builds the installable package of `HEAD` from what `make ci`
tests: `git archive` of the commit, the vendored crates, the CI image, no
network, one Rust build at a time (the build lock). It does not build in the
caches `make ci` and `make run-fixtures` write: each release build gets a new,
empty target directory under `DEB_TARGET` (`mktemp`, default
`$CI_CACHE/deb-target`), removed once the binaries are out, and no sccache.
A `build.rs`, a test or the `run-fixtures` example can write to `v2/target`,
`target/` and the sccache disk cache; none of that reaches a package
(`ci/tests/ci-sh.sh` plants a binary in both targets and requires the build's
target to be another, empty one). `DEB_SCCACHE=1` shares the sccache of `make ci`
again: faster, and `deb.json` then says `sccache.build: true`; not for a release.
The cost, measured on one commit (2026-10-30, 24 jobs on a loaded host):
102 s instead of 79 s with the shared target and sccache (the attested build
63 s instead of 35 s, the lab binary 13 s instead of 17 s), and the same three
`.deb` byte for byte.

- `ci/inside-deb.sh` (CI image, `--network=none`): `ci/release-env.sh`, then `cargo
  build --release --locked --offline` of the client side of the attested path:
  `x0ne-engine`, `x0ne-engine-cloud` and `x0ne-engine-proxy`, with
  `SOURCE_DATE_EPOCH` = the commit date and `X0NE_GIT_SHA` = the commit, which
  the `--version` of each of the three binaries prints (`tests/version_line.rs`
  in `x0ne-infer` and `x0ne-engine-cloud`). The mount paths are fixed (`/src`, `/vendor`,
  `/target`), which does what `--remap-path-prefix` would without changing
  `RUSTFLAGS` (that would refingerprint the shared target directory).
  `ci/strip-release.sh` then strips the three binaries; `SHA256SUMS.engine` names
  the stripped files, the ones the `.deb` files ship (see [Stripped binaries](#stripped-binaries)).
- Packaging: `scripts/build.sh` of the x0ne OS checkout at `X0NE_OS` (not public yet; required,
  mounted read-only), `CHANNEL=edge`, in an image built
  from its `containers/Containerfile.build` (tag = the file's hash).
  `x0ne-engine.ctl` is this commit's `ci/deb/x0ne-engine.ctl`, mounted over
  the x0ne-OS one: the client, `x0ne-engine`, `x0ne-engine-cloud` and
  `x0ne-engine-proxy`. It does not ship `x0ne-infer`
  (since 2026-10-30): the TD's server has no use on a client machine. Until
  then `x0ne-engine` shipped it too (16.6 MB of the 38.0 MB of binaries it
  installed, measured on 2026-10-30); from 2026-10-30 to 2026-11-04 it was a
  package of its own, from the x0ne-OS `x0ne-infer.ctl`. Since 2026-11-04
  `make deb` builds no server: it ships in the CVM image only, built with
  `cuda nvml` by x0ne-OS `scripts/pkg-infer-cuda.sh`.
  `tests/cloud_source_sweep.rs` reads `ci/inside-deb.sh` and the control
  file: a lab or SSH binary, a missing client binary, or `x0ne-infer` in
  either, is red.
- Version: `<crate version>+git<commit date, UTC, with time>.<sha7>`, for
  example `0.2.0~beta.2+git20261013T174143.4f8e2a1`. The time keeps two commits
  of one day in order for apt; `~beta.2` keeps the package below the `0.2.0`
  release and above `0.2.0~beta1`, the x0ne-OS `pkg-lib.sh` spelling.
- After the packaging, `ci/deb-verify.sh` (`deb_verify`, in the packaging image, no
  network) requires each `.deb` to ship exactly its binaries, by name and sha256:
  `x0ne-engine` the three lines of `SHA256SUMS.engine`; no package that
  ships `/usr/bin/x0ne-infer` (`THE SERVER IN A .deb`), no other package, no
  binary in a subdirectory of `usr/bin`, and no ELF file, executable or
  symbolic link anywhere else. Otherwise the verdict is red: the equivs packaging
  runs `dh_strip` again, and the hash published must be the one installed.
  `ci/tests/deb-verify.sh` plays it on packages it builds (run by `ci/tests/ci-sh.sh`).
- Staging: `$CI_CACHE/staging/<sha>/` (`DEB_STAGING`): the
  `.deb`, `SHA256SUMS` and `SHA256SUMS.engine` (the stripped binaries),
  `debug/` (their debug files, by build-id) and `deb.json`
  (status, version, epoch, images, x0ne-OS commit, reproducibility, digests);
  `latest` points at the last green one, `out/deb/latest.json` at its verdict.
  Nothing is published: no public apt repository exists.

`make deb-repro` builds a second time in a new container and its own new
target directory, without sccache, so both builds compile every crate
themselves (`DEB_REPRO_SCCACHE=1` shares the cache in the second build:
faster, but it then proves the packaging, not the compilation), and requires
the same `.deb` byte for byte; `deb.json` records `sccache.build` and
`sccache.repro`; otherwise `diffoscope` (image `ci/Containerfile.diffoscope`, built on
demand) writes `diffoscope-<pkg>.html` and `.txt` to the log directory and the
verdict is red. `make deb-check` installs the staged `x0ne-engine` package
with `dpkg -i` in `ci/Containerfile.check` (the pinned trixie-slim plus the
package's `Depends`, fetched at image build), no network, and requires
`Status: install ok installed`, the three client binaries in `/usr/bin` with a
`--version`, no path with `lab` or `ssh` and no `/usr/bin/x0ne-infer`
(`dpkg-deb -c`, `dpkg -L`); a staged `x0ne-infer` package is red. The
`--version` of each of the three binaries must name the commit.
The packaging, `diffoscope` and install-check containers run with
`--network=none`, under the same `--memory=$CI_MEMORY` and
`--cgroup-parent=$CI_CGROUP_PARENT` (when set) as the builds.

## Stripped binaries

Every binary that ships (those of the `.deb` packages, the three of the
release archive, `scripts/sbom.sh`, and those of the x0ne OS packages and image) is built
with the Cargo profile `release` (`Cargo.toml`: `debug = "line-tables-only"` and
`strip = "none"`), the profile `make ci` tests (`test-release`), then rewritten in
place by `ci/strip-release.sh`:

1. `objcopy --only-keep-debug --compress-debug-sections`: the symbols and line
   tables, in a debug file;
2. `strip --remove-section=.comment --remove-section=.note`: the flags
   `dh_strip` uses for an executable, so that its own pass in the packaging
   finds nothing to change (measured: same bytes after it);
3. the GNU build-id note is recomputed over the stripped bytes (the note's
   descriptor zeroed, SHA-256, first 20 bytes) and written into the binary and
   into the debug file, stored as `debug/.build-id/<2 hex>/<38 hex>.debug`.

A binary without a symbol table, or whose build-id is not one GNU note (a
descriptor of 8 to 32 bytes) alone in a single `.note.gnu.build-id` section, is
refused before it is touched (exit 2). `ci/tests/strip-release.sh` plays these
cases on binaries it compiles (run by `ci/tests/ci-sh.sh`).

No `.gnu_debuglink` is added. Why, measured on 2026-10-25 with a comment
appended at the end of three source files: the loaded sections of the release
binaries did not change, but about 900 local symbols of `x0ne-infer` did (ThinLTO
names them `<symbol>.llvm.<hash of the module>`, and the hash moves with any
source byte offset), and the linker hashes the whole file, symbols included, into
the build-id. A strip after the link that keeps the linker's build-id, and the
`.gnu_debuglink` of `dh_strip` (a CRC-32 of the debug file), both changed the
shipped bytes. `strip = "symbols"` at the link kept them, but leaves no debug
file; with `split-debuginfo = "packed"` it writes a `.dwp`, which gdb cannot use
once the binary has lost its skeleton units (gdb 16.3: `Function "main" not
defined`). A comment that adds a line before code is a change of the binary: panic
locations and log call sites carry line numbers.

To read a core of a packaged binary, take the debug files of its build
(`DEB_STAGING/<sha>/debug/`, or the release asset
`x0ne-engine-<target>-debug.tar.gz`): `readelf -n /usr/bin/x0ne-engine-cloud` (or
`eu-unstrip -n --core=<core>`) gives the build-id, and

```
gdb -iex 'set debug-file-directory <staging>/<sha>/debug' /usr/bin/x0ne-engine-cloud <core>
```

finds `.build-id/<2 hex>/<38 hex>.debug` by itself (measured with gdb 16.3 on a
core of `x0ne-engine-cloud` stopped in libc's `write`: `main () at
x0ne-engine-cloud/src/main.rs:47`; without the debug files, no frame of the
binary has a name). `-iex` sets the directory before gdb reads the binary; set
with `-ex`, it comes too late and gdb finds no debug file. Copied under
`/usr/lib/debug/.build-id/`, gdb's default debug directory on Debian, they need
no option. `x0ne-infer` and `x0ne-engine-cloud` set `PR_SET_DUMPABLE` and
`RLIMIT_CORE` to 0 at start (`x0ne-infer/src/harden.rs`,
`x0ne-engine-cloud/src/token.rs`), `x0ne-engine-proxy` the first: they leave no
core file. The core of that measurement was taken by gdb (`generate-core-file`)
of a process it started, with `CAP_SYS_PTRACE`.

### Built from source

`cargo build --release` builds the same profile, without the strip. Measured on
2026-11-04 in the CI image (Debian 12), without cuda: `x0ne-infer` 89.6 MB,
`x0ne-engine-proxy` 87.2 MB, `x0ne-engine` 78.3 MB, `x0ne-engine-cloud` 37.3 MB.
`ci/strip-release.sh <debug dir> <binary>...` (binutils) brings them to 9.9, 9.6, 5.1
and 5.5 MB, with 62.8 MB of debug files. On 2026-10-25, built that way in the CI image, with the `SOURCE_DATE_EPOCH` and
`X0NE_GIT_SHA` of the step `comment-invariance`, the four stripped files had its four sha256.

### One recipe

`[profile.release]` of `Cargo.toml` is the only recipe of what ships.
`ci/release-env.sh` runs before any `cargo` in `ci/inside-deb.sh`
and `scripts/sbom.sh` (and in the x0ne OS `scripts/pkg-engine.sh`
and `scripts/pkg-infer-cuda.sh`): an environment that sets a `CARGO_PROFILE_*`
variable, `RUSTFLAGS` or a `CARGO_*RUSTFLAGS` variable (`CARGO_ENCODED_RUSTFLAGS`,
`CARGO_BUILD_RUSTFLAGS`, `CARGO_TARGET_<triple>_RUSTFLAGS`) is refused with exit 2,
naming the variables and never their values, before anything is built. Measured on
2026-10-25: `CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true`,
`CARGO_PROFILE_RELEASE_STRIP=symbols`, `RUSTFLAGS`, `CARGO_ENCODED_RUSTFLAGS` and
`CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS` each stopped both scripts with
exit 2 and no output directory. The guard reads the environment only: `RUSTC`,
`RUSTC_WRAPPER` (the CI's sccache is one) and a `[profile]` or `rustflags` in a Cargo
configuration file are outside it.

A release build refuses a measurements file with TEST provenance, and a build with
`debug_assertions` does not (`x0ne-infer/src/measurements.rs`, exit 87). The built-in
file of this version has TEST provenance, so `ci/release-refusal.sh <x0ne-infer>...`
runs each binary on 22 cases (an empty directory, an empty environment but the case's
variable, stdin `/dev/null`, 30 s each) and requires each case's exit code and text on
stderr: with no argument, exit 87 and `refused by a release build`; the 21 others are
refusals decided before any evidence, listed at the head of the script (exits 90, 73,
81, 82, 2 and 95). It runs in `make ci`
(step `release-refusal`, on `x0ne-infer` built with the profile `release` and its
default features). No `.deb` and no release archive ships `x0ne-infer` since
2026-11-04: the CVM image does. `ci/tests/release-guards.sh` plays both scripts (run by
`ci/tests/ci-sh.sh`). Measured on 2026-10-25: the stripped `x0ne-infer` of a
plain `cargo build --release` exits 87; one built outside these scripts with
`CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true`, then stripped, printed `allowed because
this is a debug build`, exited 64, and the script exited 1. The probe needs a TEST
file: once the built-in file is a signed release one, it needs another.

## Which recipe reproduces which binary

Three pipelines build the Engine, from the same `Cargo.lock` and the same profile:

| Binaries | Recipe | Crates read from |
|---|---|---|
| `.deb` of `make deb` (`DEB_STAGING`): `x0ne-engine`, the client | `ci/inside-deb.sh` in the CI image | `/vendor/crates/` |
| x0ne OS packages and the `x0ne-infer` of the CVM image (`cuda nvml`) | x0ne OS `scripts/pkg-engine.sh`, `scripts/pkg-infer-cuda.sh` | `/out/cache/cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/` |
| release archive `x0ne-engine-<target>.tar.gz`, the client binaries (`cargo auditable`, no feature, as the `.deb`) | `.github/workflows/release.yml`, `scripts/sbom.sh` in `rust:1.98.1-bookworm` | `/usr/local/cargo/registry/` |

Panic messages carry the path of the crate that raised them: 519 strings under
`/vendor/crates/` in the `x0ne-infer` of the CI image, 523 under the x0ne OS cargo
home in the `x0ne-infer` of a CVM image (measured on 2026-10-25; the release
workflow path is the image's `CARGO_HOME`, not measured). The path of the workspace itself changes nothing (measured: the same
sha256 from `/src` and from `/tmp/comment-invariance/src`). So one commit gives three
sets of hashes: each pipeline reproduces its own binaries (`make deb-repro`, the
two builds of the release workflow, two `make pkg-infer-cuda` in x0ne OS), and no
pipeline reproduces another's. Rebuild with the pipeline that made the binary you
check. One registry path for the three pipelines is planned, not done.

## Two environments: `make repro-compare`

`make deb-repro` rebuilds in a second container on the same host, with the
same images. To rebuild elsewhere, build the same commit in another
environment (another machine, or a VM with another kernel, OS release, user,
locale and time zone; the build images loaded from the first, or rebuilt from
their Containerfiles), copy what each build made into one tree per build
(the staged `.deb` and `SHA256SUMS*`; for an image, the x0ne-OS output
directory: disk tar, EFI, manifest, reference, predicted measurements), and
write `ci/build-env.sh <tree> <image>...` to `<tree>/build-env.json` on the
machine that built it. Then

```
make repro-compare A=<tree A> B=<tree B> [JSON=<report>]
```

compares every regular file of both trees by SHA-256 and prints one line
each (`same`, `DIFFER`, `ONLY-A`, `ONLY-B`). Exit 0 only when every file is
in both trees with the same bytes; 1 when one differs or is missing; 2 when
there is nothing to compare (an empty or missing tree, a tree compared with
itself, a symbolic link, a file name outside `[A-Za-z0-9._+~/@:=,-]`). The
JSON report (default `out/repro/compare.json`) carries both
`build-env.json`, which are reported and never compared. A difference is
explained with `diffoscope` (`ci/Containerfile.diffoscope`) before anything
else. `ci/tests/repro-compare.sh` tests the script (run by `ci/tests/ci-sh.sh`).

What it shows: two environments under our control gave the same bytes. Both
still share the operator, the network sources and, for a VM, the physical
machine; only a rebuild by someone else, on their machine, removes those.

## Cache

`CI_CACHE` (default `${XDG_CACHE_HOME:-~/.cache}/x0ne-engine-ci`): `v2/src/` and `v2/target/`
(`make ci`, written only under the lock: never edit `v2/src` by hand), `v2/stage/`
(one export per run, removed after the copy), `target/` (`make run-fixtures`),
`deb-target/` (`DEB_TARGET`: one `release.XXXXXXXX` per release build of `make deb`,
removed after it), `vendor/<lock hash>/`, `cargo-home/`
(registry index and advisory database), `src/<sha>/` (`make deb`), `deb/<short>/`
(binaries, debug files and `.deb` of the last build), `logs/`. Profile `release` builds
in `target/release/` of each target directory. Delete by path; the next run
rebuilds. Past `CI_TARGET_MAX_GB` (40) `v2/target` is emptied before the build.
The dev and test profiles build with `CI_DEBUGINFO` (`line-tables-only`):
full debug info made each test binary ~390 MB and one build 11.8 GB.
Their dependencies build at `opt-level = 2` (`[profile.dev.package."*"]` in
`Cargo.toml`), the workspace crates at 0 with `debug_assertions`. Measured on
2026-10-30 (`cargo test --workspace --all-targets` in the CI image, 24 jobs,
no sccache): the cold build took 78 s instead of 59 s (one build each), the run
71 s instead of 126 to 152 s (3 runs each), and the 987 tests of the x0ne-infer lib 35 s
instead of 88 to 110 s. No shipped binary uses these profiles: the release
binaries keep their sha256 (step `comment-invariance`).

## Coverage: `make coverage`

```
make coverage          # coverage of HEAD per crate against ci/coverage-baseline.json (out/coverage/<sha>.json)
make coverage-update   # raise the baseline to the measure of HEAD, then commit ci/coverage-baseline.json
make coverage-status   # last coverage verdict
```

Not a step of `make ci`, which it would lengthen by a full instrumented build: run it
on a branch that changes code, before the merge.

- `ci/coverage.sh` (host) sources `ci/ci.sh` for its settings (`CI_ENV_FILE`) and its
  helpers: it exports `HEAD` the same way (`CI_WORKTREE=1`: the working tree), takes the
  same lock (`CI_LOCK`), and copies the tree into `$CI_COVERAGE_CACHE/src` (default
  `engine-coverage` beside `CI_CACHE`), to build in `$CI_COVERAGE_CACHE/target`: never in
  the source, target or sccache of `make ci`. It reads the crates `make ci` vendored for
  this `Cargo.lock` and fetches nothing; without them it stops and names `make ci-refresh`.
- `ci/inside-coverage.sh` (image `x0ne-engine-coverage:<hash>` from
  `ci/Containerfile.coverage`: the CI image plus the `llvm-tools` component of the pinned
  toolchain, no third-party tool; `--network=none`, the memory and cgroup of `make ci`):
  `cargo test --workspace --all-targets` with the features of `make ci` (`ALL_FEATURES` of
  `ci/inside.sh`, every one but `cuda`), the workspace members compiled with
  `-C instrument-coverage` through cargo's `RUSTC_WORKSPACE_WRAPPER`, the dependencies as
  usual: instrumented through `RUSTFLAGS` as well, they left the x0ne-infer lib tests at
  325 of 1109 after 23 min, where `make ci` runs them all in about 95 s (2026-10-30). The tests
  are built first (`--no-run --message-format=json`): that list gives the executables
  `llvm-cov` reads, the test binaries and the bins the tests start, not the build scripts
  or examples, whose profiles land apart. Then they run (`--no-fail-fast`,
  `LLVM_PROFILE_FILE=.../%8m.profraw`: at most eight profiles per binary, merged online),
  `llvm-profdata merge -sparse`, `llvm-cov export --summary-only`.
- Counted, per workspace member: the files under `<member>/src/`, but `tests.rs` and
  `*_tests.rs` (test modules in a file of their own). Not counted: `tests/`, `benches/`,
  `examples/`, `build.rs`, dependencies, the standard library, code generated under
  `target/`. A file one member compiles from another's `src/` through `#[path]` (x0ne-sdk
  takes 15 of x0ne-infer, 2026-10-30) counts once, under the member that holds it, as that
  member's own build and tests measure it; the copies are left out (`path_copies` in the
  report). Counted under the including crate, they made x0ne-sdk 72 % of lines where its
  own files are at 94 %. Lines and regions are held to the baseline; functions are recorded.
- Result: `out/coverage/<sha>.json` (`<sha>+dirty.json` for a modified working tree under
  `CI_WORKTREE=1`): `commit`, `dirty`, `source`, `duration_s` (the container),
  `lock_wait_s`, `total_s`, the duration of each step, toolchain, LLVM version, features,
  `lines`, `regions` and `functions` (covered, count, percent) per crate and in total, the
  counts per file, the comparison and `failed`. Exit 0 green, 1 red, 2 setup, or a
  baseline missing or unreadable.

### The baseline

`ci/coverage-baseline.json` holds, per crate, the percent of lines and of regions, and
`tolerance_pt` (0.2). `make coverage` is red when a test fails, when a crate is more than
`tolerance_pt` points under its baseline in lines or regions (judged in hundredths: a loss
of exactly the tolerance passes), and when a crate is on one side only. It never writes
the baseline.

`make coverage-update` writes it from the measure of `HEAD`: the report of `HEAD` when it
exists, is clean and its tests passed, else a new run. Each crate rises to the measure and
keeps its value when the measure is under it within the tolerance, so the baseline only
goes up. It refuses a modified tree, a failed test, a drop past the tolerance and a crate
gone. `CI_COVERAGE_ALLOW_DROP=1 make coverage-update` writes the measure as it is: for a
change that deletes tested code, moves to another toolchain or removes a crate, and the
diff of the baseline says so in review.

### Limits

- Doctests are not measured: their binaries are kept only by rustdoc's
  `--persist-doctests`, which 1.98.1 refuses outside nightly (measured 2026-10-30).
  `make ci` still runs them.
- One feature set. Code that set does not compile (`cuda`, or the `not(feature = ...)`
  side of one of its features) is neither covered nor uncovered: it is absent.
- An inline `#[cfg(test)] mod tests` counts as source: it runs, so it raises the percent.
- A process started with `env_clear()` loses `LLVM_PROFILE_FILE` and writes
  `default_*.profraw` in its working directory. `/src` is mounted as an overlay thrown away
  at exit (`:O`), so those writes succeed: on a read-only `/src` the runtime's
  `LLVM Profile Error` on stderr turned three golden-output tests red. The ones left under
  `/src` and `/tmp` are merged (`stray_profraw` in the report); one written in a directory
  a test removed is lost.
- `#[ignore]` tests do not run, as in `make ci`.
- Lines and regions move with the compiler: a new `rust-toolchain.toml` may need
  `CI_COVERAGE_ALLOW_DROP=1`.

Measured on 2026-10-30, 24 jobs, a host under load, the wait for the lock apart: 146 s
on an empty target (build 73 s, tests 71 s) and 74 s warm (tests 72 s). Two pairs of runs
of one code differed by 0.03 point at most (x0ne-engine-cloud, 5 regions of 12955): the
tolerance of 0.2 point is about 7 times that, four runs, not a law. First baseline, lines
and regions in percent:

| crate | lines | regions |
|---|---|---|
| x0ne-confidential | 96.62 | 96.43 |
| x0ne-engine-cloud | 89.01 | 89.29 |
| x0ne-infer | 91.32 | 90.72 |
| x0ne-ohttp | 93.33 | 92.47 |
| x0ne-sdk | 94.39 | 92.74 |
| workspace | 91.25 | 90.71 |

References: [rustc, instrumentation-based code coverage](https://doc.rust-lang.org/rustc/instrument-coverage.html)
(the commands; doctests nightly only), the defaults of
[cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov#readme) (the files it leaves
out, which this follows), Codecov's
[threshold](https://docs.codecov.com/docs/commit-status) (a tolerated drop against a base).
