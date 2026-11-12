# x0ne-Engine: every target runs in a container (ci/). `make help` lists them.
.DEFAULT_GOAL := help
.PHONY: help ci ci-image ci-refresh ci-status deb deb-repro deb-check deb-status install-check repro-compare run-fixtures sdk-wasm e2e-local core-gpu bench install-hooks

help: ## This list
	@grep -hE '^[a-z][a-z0-9-]*:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | column -t -s "$$(printf '\t')"

ci: ## fmt, ci.sh tests, clippy, test (debug and release), comment invariance of the .deb binaries, doc, the tests of x0ne-core that need no GPU, cargo-deny, gitleaks on HEAD, in a no-network container (out/ci/<sha>.json)
	ci/ci.sh run

ci-image: ## Build the pinned CI image (ci/Containerfile)
	ci/ci.sh image

ci-refresh: ## Re-vendor the crates and refetch the RustSec database (network)
	ci/ci.sh refresh

ci-status: ## Last CI verdict
	ci/ci.sh status

deb: ## Stripped release build of HEAD in a new, empty target, no sccache (debug files apart) + the x0ne-engine .deb (X0NE_OS: the x0ne OS packaging checkout), staged in DEB_STAGING/<sha>/
	ci/ci.sh deb

deb-repro: ## make deb, then a second build in a new container: the .deb must be identical (diffoscope otherwise)
	ci/ci.sh deb-repro

deb-check: ## dpkg -i the staged x0ne-engine .deb in a trixie container (no network); --version must name HEAD
	ci/ci.sh deb-check

deb-status: ## Last deb verdict
	ci/ci.sh deb-status

install-check: ## install.sh against fake releases on local HTTPS, in Debian 13, Ubuntu 24.04, Fedora 43 and Alpine containers (no network), + shellcheck (ci/install-matrix.sh: ENGINE_ARCHIVE, ENGINE_DEB, CI_CGROUP_PARENT)
	ci/install-matrix.sh

repro-compare: ## Two build trees of one commit, from two environments, byte for byte: A=<dir> B=<dir> [JSON=<report>, default out/repro/compare.json]
	@mkdir -p out/repro; ci/repro-compare.sh "$(A)" "$(B)" --json "$(or $(JSON),out/repro/compare.json)"

sdk-wasm: ## x0ne-sdk for wasm32 (with and without test-anchor), then its Node run against the simulated TD, no network (out/sdk-wasm/latest.json)
	scripts/sdk-wasm.sh

e2e-local: ## The caller-sealed path through the TD's routes and a recording carrier: Rust SDK, wasm32 in Node, Qwen3-0.6B on CPU with X0NE_E2E_QWEN3_DIR, no network (out/e2e-local/latest.json)
	scripts/e2e-local.sh

core-gpu: ## x0ne Core in x0ne-infer (--features cuda) on the local NVIDIA GPU, before a merge that touches it: build, CPU tests, GPU tests (conformance on Qwen3-0.6B and 8B, Core's greedy, device path, kernels and engine against the CPU reference), compute-sanitizer, perf ratchet; a step without a GPU is skipped, not passed; CORE_GPU_RUNNER=<launcher> runs the GPU steps as built binaries through a GPU microVM launcher where no container can have the card (scripts/core-gpu.sh: CORE_GPU_IMAGE, CORE_GPU_VENDOR, model directories; out/core-gpu/<sha>.json, gate: ci/core-gpu-check.sh)
	scripts/core-gpu.sh

bench: ## x0ne Engine measured on the local NVIDIA GPU, tier by tier (BENCH_STEPS: build core engine load soak baseline): Core alone per prompt length, the server on Core streamed and whole, 1 to 32 clients, a soak, llama.cpp on the same ids; held to ci/bench-baseline.json per GPU; BENCH_RUNNER=<launcher> as CORE_GPU_RUNNER (scripts/bench.sh, docs/benchmarks.md "The suite"; out/bench/<sha>.json and .md)
	scripts/bench.sh run

run-fixtures: ## Fixtures of a real boot from its run: RUN_ATTESTATION RUN_REFERENCE RUN_DATE RUN_IMAGE, out/run-fixtures/<date>/ compared with the tree (RUN_WRITE=1 copies, RUN_OVERWRITE=1 replaces, RUN_TEST_KEY=1 accepts a TEST-key reference)
	ci/ci.sh run-fixtures

.PHONY: coverage coverage-update coverage-status
coverage: ## Test coverage per crate (llvm-tools, the features of make ci) in a no-network container, held to ci/coverage-baseline.json; not part of make ci (out/coverage/<sha>.json)
	ci/coverage.sh run

coverage-update: ## Raise ci/coverage-baseline.json to the measure of HEAD (clean tree, tests green); CI_COVERAGE_ALLOW_DROP=1 lowers it
	ci/coverage.sh update

coverage-status: ## Last coverage verdict
	ci/coverage.sh status

# Copies ci/hooks/* into the repository's hooks directory. Never overwrites a
# hook that is already there and differs: remove it by hand first.
install-hooks: ## Install the post-merge hook (make ci then make deb, through systemd-run --user)
	@dir=$$(git rev-parse --git-path hooks); mkdir -p "$$dir"; \
	for h in ci/hooks/*; do n=$$(basename "$$h"); \
	  if [ -e "$$dir/$$n" ] && ! cmp -s "$$h" "$$dir/$$n"; then echo "install-hooks: $$dir/$$n exists and differs, not touched" >&2; exit 1; fi; \
	  install -m 0755 "$$h" "$$dir/$$n" && echo "installed $$dir/$$n"; done

# A workspace of the tree may add targets of its own: its ci/make.mk, included when present.
-include $(wildcard */ci/make.mk)
