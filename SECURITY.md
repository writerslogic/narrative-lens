# Security Policy

narrative-lens analyzes caller-supplied text locally. Runtime analysis makes no
network calls. The CLI reads only the inputs you name; optional model adapters read
only explicitly configured model directories, with no ambient discovery. Native ONNX
Runtime telemetry is disabled. Builds can fetch Cargo packages and native runtime
binaries.

## Reporting a vulnerability

Report security issues privately rather than opening a public issue:

- [GitHub private vulnerability reporting](https://github.com/writerslogic/narrative-lens/security/advisories/new), or
- Email security@writerslogic.com

Include what you found, how to reproduce it, and its potential impact. We aim to
acknowledge reports within 5 business days.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.2.x | Yes |
| < 0.2 | No |

Only the latest published release receives security fixes.

## Data retention and models

- Callers own manuscript contexts and optional NER caches. They retain text until cleared, evicted within their budgets, or dropped; default NER calls keep no process-global memo.
- Cache budgets describe retained payload estimates, not process RSS.
- Verified profiles hash immutable model and tokenizer bytes, validate preprocessing contracts and declared labels, and record identity in NLI reports. Pins must come from a trusted source; they establish neither learned accuracy nor sandboxed graph execution.
- A failed verified replacement clears the prior model; replacement during an NLI pass discards that pass's model findings. Legacy setters remain explicitly unverified. Load models only from trusted sources.

## Automated checks

| Check | Where |
| --- | --- |
| Rust tests across feature sets, clippy, Node binding, TypeScript, and packed-package tests | `ci.yml` on pushes and pull requests to `main` |
| Minimum Rust versions (1.89 core, 1.92 native ONNX, 1.91 wasm) | `ci.yml` |
| Dependency advisories, licenses, and sources from `deny.toml` | `ci.yml` (`cargo deny --deny unsound`) |
| CodeQL for Rust, JavaScript/TypeScript, and workflow definitions | `codeql.yml` |
| `cargo-fuzz` text-analysis target with AddressSanitizer, failing inputs kept as artifacts | `fuzz.yml`, daily and on every push and pull request |
| OpenSSF Scorecard | `scorecard.yml` |
| Release gate: installed-package tests and native binary digest verification on every target and Node line | `platform-assurance.yml`, `release.yml` |

Workflow actions are pinned to commit hashes, npm installs use `npm ci` against the
checked-in lockfile, and Dependabot groups CodeQL action updates so initialization and
analysis share a version.

`deny.toml` accepts the maintenance advisory `RUSTSEC-2024-0436` because the current
`tokenizers` releases require the archived `paste` build-time macro and no patched
version exists. The exception does not cover vulnerability or soundness advisories;
reassess it when `tokenizers` changes.

## Scorecard history

The 19 initial code scanning alerts (2026-10-27) were OpenSSF Scorecard supply-chain
checks. Mutable action references and lockfile-bypassing npm installs were replaced
with pinned hashes and `npm ci`; the missing-fuzzing alert is addressed by the fuzz
target and workflow. No code scanning alerts were open as of 2026-10-28.

Three historical metrics are accepted exceptions and do not justify bypassing checks
or reviews on future changes:

- **SAST** counted commits made before scanning was consistently configured; old commits are not rewritten.
- **Code Review** counted changes merged before approval was required; retrospective approvals do not establish review before merge.
- **Maintained** reports a repository younger than 90 days (created 2026-09-15); the condition expires in December 2026.

The [OpenSSF Best Practices entry](https://www.bestpractices.dev/en/projects/15147)
is a separate external self-certification whose status is established in that service.
