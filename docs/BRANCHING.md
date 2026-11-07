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
  The same key signs each release's `SHA256SUMS` (step 7 below).

## Versions and releases

[SemVer](https://semver.org/) as Cargo applies it: while at `0.y.z`, a new
`y` may break compatibility, a new `z` does not. One version for the whole
workspace; tag `vX.Y.Z` equals it. Pre-releases: `-alpha.N`, `-beta.N`, `-rc.N`.

1. Branch `chore/release-X-Y-Z` from `main` (`scripts/check-branch-name.sh`
   refuses dots: `chore/release-0-2-0-beta-7`); `cargo release version X.Y.Z --execute`.
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
   and attests the artifacts, installs the archive with `install.sh` in
   Debian 13, Ubuntu 24.04 and Fedora 43 (refused on Alpine 3) and the `.deb`
   with `dpkg` in Debian 13 and Ubuntu 24.04, then drafts the release.
6. The draft's notes are the `CHANGELOG.md` section, or a link to it when the
   section is over the 125,000 characters GitHub takes; edit them if needed
   (`gh release edit --notes-file`), and check the artifacts download and verify.
7. Sign the checksum list with the release key, the one that signs the tag
   (`install.sh` pins its public half), and upload the signature to the draft.
   `install.sh` and the README's `.deb` steps refuse a release without
   `SHA256SUMS.sig`. Whoever can change the draft's assets could put another
   list there, with the right first line: the signature must go only on a
   list the release workflow attested. So the block below, in one subshell
   that stops at the first command that fails:
   - downloads `SHA256SUMS` once, into a new directory;
   - verifies the build provenance of that local file with
     `gh attestation verify`: signed by `release.yml` of this repository,
     built from the tag's ref and commit, on a GitHub-hosted runner;
   - checks its first line, `# x0ne-Engine vX.Y.Z`;
   - signs that same file, never a second download, and uploads the
     signature (no `--clobber`: a `SHA256SUMS.sig` already in the draft
     stops it);
   - downloads both back into another new directory: the list must be the
     bytes signed (`cmp`), start with the same line, and verify against the
     key of `install.sh`, as the script does.

   It needs gh 2.102.0 or later, git, and OpenSSH 8.1 or later. gh has had
   `--source-ref` and `--source-digest` since 2.68.0, but before 2.97.0 it
   matched `--signer-workflow` as an unescaped regular expression
   ([GHSA-mm27-mwq9-fr5g](https://github.com/cli/cli/security/advisories/GHSA-mm27-mwq9-fr5g)),
   and before 2.102.0 as a prefix of the workflow's identity, and compared
   `--source-ref` case-insensitively
   ([GHSA-wjmr-j3rp-mh2g](https://github.com/cli/cli/security/advisories/GHSA-wjmr-j3rp-mh2g),
   [GHSA-4mq3-hpgx-9cx8](https://github.com/cli/cli/security/advisories/GHSA-4mq3-hpgx-9cx8)).
   On the maintainer's machine, in a checkout of the tag:

   ```sh
   (
     set -eu
     tag=vX.Y.Z
     repo=x0ne-labs/x0ne-Engine
     gh --version | awk 'NR == 1 { split($3, v, "."); ok = v[1] + 0 > 2 || (v[1] + 0 == 2 && v[2] + 0 >= 102) } END { exit !ok }' ||
       { echo "gh 2.102.0 or later is required" >&2; exit 1; }
     commit=$(git rev-parse --verify "$tag^{commit}")
     d=$(mktemp -d)
     gh release download "$tag" -R "$repo" --pattern SHA256SUMS --dir "$d"
     gh attestation verify "$d/SHA256SUMS" --repo "$repo" \
       --signer-workflow "$repo/.github/workflows/release.yml" \
       --source-ref "refs/tags/$tag" --source-digest "$commit" --deny-self-hosted-runners
     first=$(head -n 1 "$d/SHA256SUMS")
     test "$first" = "# x0ne-Engine $tag" || { echo "SHA256SUMS is not the list of $tag: $first" >&2; exit 1; }
     ssh-keygen -Y sign -f ~/.ssh/x0ne_signing -n x0ne-engine-release "$d/SHA256SUMS"
     gh release upload "$tag" -R "$repo" "$d/SHA256SUMS.sig"
     v=$(mktemp -d)
     gh release download "$tag" -R "$repo" --pattern SHA256SUMS --pattern SHA256SUMS.sig --dir "$v"
     cmp "$d/SHA256SUMS" "$v/SHA256SUMS"
     cmp "$d/SHA256SUMS.sig" "$v/SHA256SUMS.sig"
     first=$(head -n 1 "$v/SHA256SUMS")
     test "$first" = "# x0ne-Engine $tag" || { echo "SHA256SUMS is not the list of $tag: $first" >&2; exit 1; }
     sed -n "s/^ALLOWED_SIGNER='\(.*\)'\$/\1/p" install.sh > "$v/allowed_signers"
     test -s "$v/allowed_signers"
     ssh-keygen -Y verify -f "$v/allowed_signers" -I dev@x0ne.co -n x0ne-engine-release \
       -s "$v/SHA256SUMS.sig" < "$v/SHA256SUMS"
   )
   ```

   It ends with `Good "x0ne-engine-release" signature for dev@x0ne.co`.
   Any other end: do not publish the draft. `ci/tests/installer.sh` replays
   this block with a test key and a fake gh.
8. Publish the draft.

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
