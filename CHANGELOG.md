# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) +
[Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added
- The LitBank holdout split was scored once at 548a033 (heuristic and local-NER modes);
  the records are `docs/benchmarks/evidence/literary-holdout-{heuristic,ner}-v1.json` and
  `docs/EVALUATION.md` shows them beside the development figures.

### Changed
- `count_syllables` no longer allocates per call; results are unchanged.
- Heuristic person extraction segments each scene into sentences once and reuses the
  dialogue-attribution scan instead of repeating both; results are unchanged.
- The crate release workflow skips the upload when the version is already on crates.io.
- `docs/EVALUATION.md` names the manual WASI launcher used for the recorded tract parity run.

## [0.2.0] - 2026-10-28

### Added
- `manuscript` facade: coordinated findings with original-text evidence, author
  controls, coverage, chronology, incremental `AnalysisContext`, and revision
  comparison. Node exposes `analyzeManuscriptAsync`, `ManuscriptContext.analyzeAsync`,
  and `compareManuscriptReports` (`manuscript.d.ts`).
- `cli` feature and `narrative-lens` binary: text/Markdown adapter with terminal,
  JSON, and SARIF output, rule selection, baselines, `--fail-on`, and local model
  selection through `--model-root`/`--model-profile`.
- Author annotations the analyzers treat as ground truth: setup/resolution,
  truth/belief/knowledge, and communication events, validated against the source.
- Verified local model loading with pinned manifests (`loadVerifiedModel`), Ekman-7
  emotion classification, and SBERT/NER/NLI parity tests on wasm32-wasip1.
- `readability::analyze_smog` (McLaughlin 1969 block sampling) and the complete
  pinned Dale-Chall word list with offline provenance verification.
- Given-name gender table (`substrate::given_names`) and Unicode offset validation
  (`substrate::offsets`).
- Evaluation harness: synthetic finding fixtures, the LitBank development-split
  benchmark (`examples/evaluate_literary.rs`, `scripts/literary_benchmark.py`), and
  recorded Rust/Node performance measurements (`docs/EVALUATION.md`). Reported
  accuracy figures are development-split numbers used for tuning; the holdout split
  has not been scored.
- Release pipeline: `platform-assurance.yml` builds, installs, and smoke-tests the
  assembled tarballs on every declared target and Node 18-26 line; `release.yml`
  publishes only tarballs whose digests and smoke attestations match; crate
  publication requires the release tag to match `Cargo.toml`.
- Component and release-integrity test suites; wasm32 component tests in CI.

### Changed
- Coreference: `they`/`them` binds only to a character whose gender hint is
  `neutral`, a new accepted value alongside `male`/`female`/`unknown`. Without a
  neutral hint the pronoun stays unresolved.
- `build_character_gender_map` / `buildCharacterGenderMap` genders cast names with
  a gendered honorific or a listed given name without textual pronoun evidence;
  other names still need nearby pronouns. The map never emits `neutral`.
- Character entities prefer the fuller canonical (`Hester Prynne` over `Hester`).
  Entity IDs are unchanged, but persisted `EntityCorrection` bins keyed by an old
  bare canonical may land in a different bin.
- Narrative pronoun chains continue across quoted dialogue, and cast entries that
  prefix one another form one alias group, so `resolvedTo` strings for pronouns
  may differ from 0.1.0.
- Minimum supported Rust is 1.89 for the core crate (1.92 for native ONNX, 1.91 for
  wasm ONNX). `ort` is pinned to 2.0.0-rc.15 and `Cargo.lock` is committed.
- Unearned-payoff detection segments climactic sentences with the shared sentence
  splitter instead of splitting on periods.

### Deprecated
- `continuity::epistemic_engine::ReaderEpistemicState`, `manuscript::finding::adapt_explanation`,
  `manuscript::Scene::from_timeline`, `types::Options`, and `types::AnalysisResult` have no
  callers and will be removed in 0.3.0.

### Removed
- Uncalled public helpers: the atmosphere/sensory analysis in `substrate::text`
  (`analyze_atmosphere`, `detect_sensory_words`, `compute_sensory_balance`, `grounding_words`
  and their types), the numeric helpers and duplicate `Genre`/`split_sentences` in
  `substrate::utils`, `common_types::safe_substr`, `text::count_paragraphs`,
  `voice::build_voice_profile_py`/`compare_voices_py`, `pacing::get_action_verbs`/
  `enrich_with_info_density`, and `emotion::classify_text`/`classify_batch`.
- Audit evidence under `docs/audit-2026-10-27/`; the referenced measurement files moved to
  `docs/benchmarks/evidence/`, and the benchmark scripts now write new reports to
  `target/evaluation/`.
- `scripts/create_evaluation_v2.mjs`; the fixture it generated is frozen in the repo.

### Fixed
- Supply-chain hardening: pinned action SHAs, `cargo deny` with unsound advisories
  denied, provenance-checked fixtures and readability resources.
- `sha2` 0.11 for the `onnx` feature, which moves `block-buffer` past GHSA-qwgh-2vcv-g2f7.

## [0.1.0] - 2026-09-15

### Added
- Initial extraction of the craft, structure, and continuity analyzers from Emathy's
  meaning engine into a standalone crate.

[Unreleased]: https://github.com/writerslogic/narrative-lens/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/writerslogic/narrative-lens/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/writerslogic/narrative-lens/releases/tag/v0.1.0
