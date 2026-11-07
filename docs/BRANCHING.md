# Branches, commits and releases

**Model: GitHub Flow.** `main` is the only long-lived branch and is always
releasable. Every change, the maintainer's included, is a short branch and a
pull request, squash-merged.

## Branches

- Name: `<type>/<optional issue number>-<short-description>`, lowercase
  letters, digits and hyphens, 50 characters at most.
  Checked by `scripts/check-branch-name.sh`
  (`^(feat|fix|docs|chore|security)/([0-9]+-)?[a-z0-9]+(-[a-z0-9]+)*$`).
- Types: `feat/` new capability, `fix/` bug fix, `docs/` documentation,
  `chore/` maintenance, CI, dependencies, refactoring, `security/` public
  hardening (an undisclosed vulnerability never goes in a public branch:
  see below).
- Valid: `fix/123-receipt-nonce`, `docs/install-steps`. Invalid:
  `Fix/123`, `feature/login`, `fix/my_change`, `fix/`.
- Branch from `main`, rebase on `main` to stay current, delete after merge.
  A branch older than 30 days is finished or closed.

## Pull requests and commits

- `main` accepts no direct push. A pull request merges when CI is green and
  the branch is up to date; history stays linear; no force-push.
- Squash-merge only. The pull request title becomes the commit on `main` and
  follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):
  `fix(receipt): reject a nonce of the wrong length`. Commits inside the
  branch are free.
- Required checks: every job of `ci.yml` (`fmt`, `clippy`, `test`, `doc`,
  `msrv`, `stable`, `image`, `ci-scripts`), of `security.yml` (`deny`,
  `gitleaks`, `dependency-review`, `workflow-lint`) and of `codeql.yml`
  (`codeql`), as each workflow's header says. No job runs `cargo audit`.
- Ordinary commits are not signed. The first public commit and every release
  tag are signed by the maintainer (SSH key); check with `git tag -v vX.Y.Z`.

## Versions and releases

[SemVer](https://semver.org/) as Cargo applies it: while at `0.y.z`, a new
`y` may break compatibility, a new `z` does not. One version for the whole
workspace; tag `vX.Y.Z` equals it. Pre-releases: `-alpha.N`, `-beta.N`, `-rc.N`.

1. Branch `chore/release-X.Y.Z` from `main`; `cargo release version X.Y.Z --execute`.
   Set `VERSION="X.Y.Z"` in `install.sh`, `Version: X.Y.Z` (Debian form:
   `0.2.0-beta.7` gives `0.2.0~beta.7`) in `ci/deb/x0ne-engine.ctl`, and the
   tag in the README's Install section: the release workflow refuses a tag
   that `install.sh` or the control file does not name.
2. Move `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` in `CHANGELOG.md`
   ([Keep a Changelog](https://keepachangelog.com/en/1.1.0/)); update `CITATION.cff` if present.
   Add `measurements/X.Y.Z.json` ([pinning.md](pinning.md)): the build
   embeds it and fails without it; `x0ne-infer/tests/measurements/fixture.json`
   names the version too.
3. Pull request `chore: release vX.Y.Z`; squash-merge on green CI.
4. On the updated `main`: `git tag -s vX.Y.Z -m vX.Y.Z && git tag -v vX.Y.Z`.
5. `git push origin refs/tags/vX.Y.Z`; the release workflow builds, checksums
   and attests the artifacts, installs the archive with `install.sh` and the
   `.deb` with `dpkg` in four distributions, then drafts the release.
6. Paste the `CHANGELOG.md` section into the GitHub release notes; check the
   artifacts download and verify.

## Vulnerabilities

Reported privately (`SECURITY.md`, answer within 72 hours). The fix is
prepared in the advisory's temporary private fork, merged with a test that
fails without it, released as a patch, and only then the advisory is
published.

## Later

When two release lines must be maintained, a `release/X.Y` branch is cut from
the tag `vX.Y.0`; fixes land on `main` first and are cherry-picked
(`git cherry-pick -x`). Contributor sign-off (DCO) starts with the first
external contribution.
