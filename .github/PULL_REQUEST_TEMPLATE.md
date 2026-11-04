<!-- Title: Conventional Commits, e.g. "fix(receipt): reject a nonce of the wrong length".
     It becomes the squashed commit on main (docs/BRANCHING.md). -->

## What and why

## How it was checked

- [ ] The commands of `CONTRIBUTING.md` pass locally (fmt, tests, clippy, rustdoc)
- [ ] A changed `SECURITY:` line, verifier, pin, receipt format or exit code has a test that fails without the change
- [ ] `CHANGELOG.md` has a line under `## Unreleased` if users see the change
- [ ] A dependency change: `cargo deny check` and `scripts/third-party-licenses.sh` run

Not a vulnerability report: see `SECURITY.md`.
