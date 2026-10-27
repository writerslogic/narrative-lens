# Security Policy

narrative-lens is a stateless analysis library: it reads text passed to it and returns a
result. It makes no network calls, stores no data, and reads no files beyond what the
optional local ONNX model path is explicitly pointed at.

## Reporting a Vulnerability

Please report security issues privately rather than opening a public issue:

- Use [GitHub's private vulnerability reporting](https://github.com/writerslogic/narrative-lens/security/advisories/new), or
- Email security@writerslogic.com

Include what you found, how to reproduce it, and its potential impact. We aim to
acknowledge reports within 5 business days.

## Supported Versions

Only the latest published release receives security fixes.

## Automated checks

GitHub Actions runs Rust tests (including the Node and ONNX features), clippy,
Node binding tests, and TypeScript checks on pull requests and pushes to `main`.
CodeQL scans Rust, JavaScript/TypeScript, and workflow definitions. The daily
`cargo-fuzz` job also runs on pull requests and pushes, checking text analysis
and Unicode offset handling with AddressSanitizer. Failing inputs are retained
as workflow artifacts.

Workflow actions use immutable commit hashes. npm builds install the checked-in
dependency lockfile with `npm ci`. Dependabot groups CodeQL action updates so
initialization and analysis use the same version.

## Scorecard assessment (2026-10-27)

The 19 initial code scanning alerts were OpenSSF Scorecard supply-chain checks.
Alerts [2–14](https://github.com/writerslogic/narrative-lens/security/code-scanning?query=is%3Aopen+rule%3APinnedDependenciesID)
identified mutable workflow action references and npm installs that bypassed the
lockfile; these were replaced with commit hashes and `npm ci`. Alert
[16](https://github.com/writerslogic/narrative-lens/security/code-scanning/16)
identified missing fuzzing; the fuzz target and scheduled workflow address it.

The following historical metrics are accepted exceptions after required checks
and pull request review protection are enabled on `main`. They do not justify
bypassing checks or reviews on future changes:

- [15: SAST](https://github.com/writerslogic/narrative-lens/security/code-scanning/15)
  counted older commits created before scanning was consistently configured.
  CodeQL checks are required for future merges; old commits are not rewritten
  or given fabricated scan results.
- [17: Code Review](https://github.com/writerslogic/narrative-lens/security/code-scanning/17)
  counted changes already merged without approval. Future pull requests require
  approval; retrospective approvals do not establish a review before merge.
- [18: Maintained](https://github.com/writerslogic/narrative-lens/security/code-scanning/18)
  reports that the repository was created on 2026-09-15 and is less than 90 days
  old. Repository age cannot be repaired by a code change; the age condition
  expires in December 2026.

The [OpenSSF Best Practices entry](https://www.bestpractices.dev/en/projects/15147)
is a separate, external self-certification. Its passing status must be established
in that service; local metadata does not change the public badge.
