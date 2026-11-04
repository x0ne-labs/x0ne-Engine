# Governance

A small project with one maintainer. This file says who decides and how a
change gets in, so that nothing depends on a conversation nobody can find.

## Roles

- **Maintainer**: the owner of the `x0ne-labs` GitHub organization. Decides
  scope, merges, releases, and acts as Community Moderator for
  `CODE_OF_CONDUCT.md` and as the contact of `SECURITY.md`. The maintainer
  is x0ne, the copyright holder named in `LICENSE`; there is no other
  maintainer today.
- **Contributor**: anyone who opens an issue or a pull request. External
  pull requests are accepted once the Contributor License Agreement is
  signed. It is not published here yet: until it is, no external pull
  request is merged. Issues are welcome (`CONTRIBUTING.md`).

## How a change gets in

1. Every change goes through a pull request, including the maintainer's own.
2. CI must be green: format, tests on every feature set, clippy `-D warnings`,
   rustdoc `-D warnings`, `cargo audit`, `cargo deny`.
3. A change that touches a `SECURITY:` line, a verifier, a pin, a receipt
   format or an exit code needs the test described in `CONTRIBUTING.md`
   ("a security test says what its failure would mean"), and a line in
   `CHANGELOG.md`.
4. The maintainer reviews contributors' pull requests; with a single
   maintainer, the maintainer's own pull requests are merged on green CI.
5. Dependency updates (Dependabot) are merged on green CI; a major version
   bump of a cryptographic or attestation crate is reviewed like a code change.

## Decisions

Product decisions (what the server promises, what the threat model covers,
the receipt format, hardware scope) are the maintainer's and are written down
in `README.md`, `docs/`, `x0ne-infer/THREAT_MODEL.md` or `CHANGELOG.md` when
taken.
Disagreement goes in an issue; the maintainer's answer closes it.

## Changing this file

By pull request, like everything else.
