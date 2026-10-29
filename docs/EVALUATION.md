# Accuracy and performance evaluation

| Measurement | Data | What it does and does not show |
| --- | --- | --- |
| [Finding-level accuracy](#finding-level-evaluation) | 52 agent-authored fiction cases | Supported patterns, controls, and regressions; not editorial accuracy on published novels |
| [Analyzer-family baseline](#original-analyzer-family-baseline) | 56 older cases | Binary any-finding detection per family |
| [LitBank persons and coreference](#litbank-development-split-benchmark) | 100 public-domain novel excerpts, development split only | Person spans, cast inventory, and pronoun resolution against human annotations; the holdout split is unscored |
| [Pronunciation reference](#pronunciation-reference) | CMUdict development split | Syllable-count compatibility with a published dictionary |
| [Performance](#performance) | Two deterministic 160-scene workloads | Latency and retained memory on one machine; no real-time guarantee |
| [Model parity](#pinned-pretrained-model-parity) | One pinned MiniLM export, one pinned NER export | Native and WASI runtime fidelity against Python ONNX Runtime; not model accuracy |

The synthetic fixtures and labels are **agent-authored engineering cases with no human
editorial validation**. Original fiction was written for the repository; no published
passages were copied. Confidence values remain heuristic unless a caller supplies its
own calibration.

## Readability resources

- Dale-Chall uses the complete pinned 2,942-entry upstream [MIT word list](https://github.com/words/dale-chall/tree/cc171e5a515ef67818fa17308dfceff38a21efa4). Provenance, transformation, and hashes are in [readability-resources.json](readability-resources.json); `uv run --no-project scripts/verify_readability_resources.py` verifies them offline. It is not an independent reconstruction of the printed 1995 list, and inflections and proper names get no special expansion.
- SMOG follows [McLaughlin (1969), equation (d), page 643](https://ogg.osu.edu/media/documents/health_lit/WRRSMOG_Readability_Formula_G._Harry_McLaughlin__1969_.pdf) over three disjoint ten-sentence blocks at the beginning, center, and end. `readability::analyze_smog` / Node `analyzeSmog` return the selected byte ranges, polysyllable count, and a nullable grade; fewer than 30 complete word-bearing sentences is unavailable. Legacy `computeReadability` keeps numeric zero for that case and exposes `smogAvailable: false`.
- Syllables and sentence boundaries remain English heuristics; spoken numerals and abbreviations are not expanded. None of this validates reading comprehension on a human population.

## Finding-level evaluation

`tests/fixtures/evaluation-v2.json` (version `finding-level-fiction-v2.1`) holds 52
cases, 26 calibration and 26 held-out, with original UTF-8/UTF-16 spans, related
contradictory quotations, rule IDs, rationale, origin, and text hashes. Nonempty texts
are disjoint; empty input is an unscored boundary case in both splits. Cases cover
errors, clean prose, duplicate occurrences, Unicode, author dictionaries, deliberate
repetition, dialect, dense prose, long sentences, cyclical structure, supernatural
exceptions, transfers, and releases.

```sh
uv run --no-project scripts/verify_fixtures.py
cargo run --example evaluate_findings > measured.json
cargo test --test evaluation_findings
```

- The harness matches findings one-to-one by rule, scene, exact primary quotation, and original coordinates. Related evidence must cover the labeled conflicting text; missing one member of a duplicate pair is a miss. Aggregate annotations match explicit scope and beat identity. Every emitted quote is checked against the original manuscript.
- Metrics per rule and split: TP/FP/FN, precision/recall, clean cases flagged, unrated predictions, Brier score, and confidence bins. Calibration uses calibration labels only, with four fixed score bins per rule and Laplace precision `(correct + 1)/(count + 2)`. Brier measures correctness among rated emitted findings; recall accounts for misses separately, so missing output cannot improve the gate by disappearing.
- Flat-structure cases label only the missing climax and mark the other beat predictions **unrated** (three per held-out case). No NLI accuracy is measured; capitalization is an adapter ID without a detector; standalone families outside the facade are not covered.
- An author-controls ablation removes dictionary additions, corrections, intentions, and suppressions while keeping detector thresholds, separating supplied intent from the detector-only claim.

[Current measured data](evaluation-v2-current.json) records every bin and both
ablation modes. On the 26 held-out cases: 14 matched findings, zero rated FP/FN, and
three unrated structural predictions; the raw control ablation adds false positives in
six rule families. This small designed sample does not justify a perfect-accuracy claim.

| Held-out rule | TP | Clean cases | Raw Brier | Calibration-only fit applied to held-out Brier |
| --- | ---: | ---: | ---: | ---: |
| Repetition | 4 | 3 | 0.0400 | 0.0278 |
| Whitespace | 1 | 5 | 0.0400 | 0.1111 |
| Dictionary absence | 1 | 1 | 0.2500 | 0.1111 |
| Author lexicon | 2 | 1 | 0.0400 | 0.0625 |
| Readability | 1 | 2 | 0.0400 | 0.1111 |
| Long-sentence average | 1 | 2 | 0.0400 | 0.1111 |
| Missing structure beat | 1 | 2 | 0.0400 | 0.1111 |
| Locked location | 1 | 2 | 0.0400 | 0.1111 |
| Eye attribute | 1 | 2 | 0.0400 | 0.1111 |
| Object location | 1 | 1 | 0.2500 | 0.1111 |

Calibration worsens several scores on this sample, so it stays a diagnostic experiment
and is not installed into default library confidence.

The [frozen baseline](evaluation-v2-baseline.json) records pre-normalization behavior,
including the false conflict between grey and gray eyes; its input is retained as
`tests/fixtures/evaluation-v2-baseline.json`. Regression gates preserve each baseline
rule and split's true positives, disallow increased false positives, require zero
missed supported annotations and clean-case false positives, and allow at most 0.005
absolute Brier regression. Hashes protect annotations and baselines; changes require a
recorded reason and updated provenance.

## Original analyzer-family baseline

The 56-case `evaluation.json` and `evaluation-baseline.json` remain reproducible with
`cargo run --example evaluate -- --legacy`; the facade result is `evaluation-current.json`
(`cargo run --example evaluate` and `cargo test --test evaluation`). Each split has 28
cases. This harness measures binary any-finding detection per analyzer family with a
fixed 0.8 comparison score for legacy confidence, not individual finding accuracy. Its
broader genre coverage (fantasy, thriller, romance, literary, horror, historical) stays
useful regression input.

## LitBank development-split benchmark

[LitBank](https://github.com/dbamman/litbank) (Bamman et al., 2019 and 2020, CC-BY-4.0)
annotates entities, quotations, and coreference in 100 English novel excerpts.
[litbank-source.json](benchmarks/litbank-source.json) pins the repository revision,
file hashes, and an author-grouped split: a seeded hash assigns about one fifth of the
authors to `final_holdout` and the rest to `development`.

**All figures below are development-split numbers used while tuning the heuristics.
The holdout split has not been scored and its labels are never inspected during
development; no generalization claim is made.**

```sh
uv run --no-project scripts/fetch_literary_benchmark.py --download     # pinned corpus into target/external-validation
uv run --no-project scripts/literary_benchmark.py prepare --folder target/external-validation/litbank-prepared
cargo run --locked --release --example evaluate_literary -- target/external-validation/litbank-prepared/inputs.json > predictions.jsonl
cargo run --locked --release --features onnx --example evaluate_literary -- target/external-validation/litbank-prepared/inputs.json --ner-cache MODEL_DIR > predictions-ner.jsonl
uv run --no-project scripts/literary_benchmark.py score --folder target/external-validation/litbank-prepared --predictions predictions.jsonl --split development
```

`prepare` and `score` never modify the pinned labels or train anything. The example
runs the heuristic `continuity::ner` and `CoreferenceResolver` paths, or the verified
local NER model when `--ner-cache` points at a pinned export. The scored v6 runs are
kept as [benchmarks/evidence/literary-development-{heuristic,ner}-v6.json](benchmarks/evidence/);
new runs write to `target/evaluation/` by default.

| Metric (development split, v6, 2026-10-28) | Heuristic | With local NER model |
| --- | ---: | ---: |
| Proper person name spans, F1 | 0.354 | 0.598 |
| Named character inventory, F1 | 0.399 | 0.636 |
| Model person spans, exact, F1 | n/a | 0.816 |
| Coreference with supplied cast, precision / recall / F1 | 0.779 / 0.325 / 0.458 | 0.779 / 0.325 / 0.458 |
| Coreference with detected cast, precision / recall / F1 | 0.626 / 0.252 / 0.359 | 0.671 / 0.284 / 0.399 |
| Dialogue spans with supplied cast, F1 | 0.854 | 0.854 |
| Explicit speaker attribution with supplied cast, precision / recall | 0.982 / 0.128 | 0.982 / 0.128 |

Coreference abstains on ambiguity by design, which is why precision stays well above
recall. Gold pronouns outside any cast entry (about 22%) and plural `they` (about 12%)
cap heuristic coreference near F1 0.46-0.48 on this split; substantially higher recall
needs a neural coreference model. The LitBank text has one sentence per line and no
blank-line paragraphs, so paragraph-scoped heuristics see one paragraph per excerpt.

## Pronunciation reference

`scripts/pronunciation_benchmark.py` pins a [CMUdict](https://github.com/cmusphinx/cmudict)
revision ([pronunciation-reference.json](benchmarks/pronunciation-reference.json)) and
compares `substrate::text::count_syllables` with dictionary syllable counts:
`pronunciation_benchmark.py prepare` writes `inputs.json` under
`target/external-validation/pronunciation-prepared`, `cargo run --release --example
evaluate_pronunciation -- <that inputs.json>` emits predictions, and `score` compares them. On the development split (94,047 words, 80,644 with a compatible
variant) the recorded baseline reached 0.857 variant-compatible accuracy with a mean
minimum absolute error of 0.147 syllables. This measures compatibility with a published
reference, not human reading comprehension; the holdout split is unscored.

## Performance

Two deterministic novel-sized workloads exercise full and incremental analysis:

| Workload | Scenes | Content |
| --- | ---: | --- |
| `benchmark-novel-v1` | 160 | Four existing fiction passages cycled, each repeated 14 times (85,280 words, 474,880 UTF-8 bytes) |
| `benchmark-novel-v2-xorshift32-20261027` | 160 | Generated by `scripts/generate_benchmark.mjs` from a fixed seed: original sentence templates, Unicode names, varied vocabulary, seeded cross-scene findings |

```sh
mkdir -p target
node scripts/generate_benchmark.mjs > target/benchmark-novel-v2.json
cargo build --release --example benchmark_manuscript
# TARGET is the target_directory from `cargo metadata --no-deps --format-version 1`.
/usr/bin/time -l TARGET/release/examples/benchmark_manuscript                                  # macOS; Linux uses -v
/usr/bin/time -l TARGET/release/examples/benchmark_manuscript target/benchmark-novel-v2.json
npm run build:debug
NARRATIVE_LENS_ADDON_PROFILE='debug native addon; no ONNX models' node examples/benchmark-manuscript.mjs
NARRATIVE_LENS_ADDON_PROFILE='debug native addon; no ONNX models' node examples/benchmark-manuscript.mjs target/benchmark-novel-v2.json
```

Rust takes five samples each of cache-cleared full analysis, a single-scene edit, and
a reorder, resetting input before each sample; "full" does not reset process-wide regex
and dictionary initialization. Incremental output is asserted equal to a fresh result.
Node uses the same fixtures with a debug addon and reports input conversion, timer
ticks during async work, maximum timer gaps, latency, and process RSS. Retained cache
estimates differ from whole-process peak RSS.

### Recorded benchmark results

Recorded 2026-10-27 on an Apple M4 (arm64), 32 GiB RAM, macOS 27.0, Rust 1.96.0, and
Node 26.8.1, with no model and five samples per operation. Fixture recipes and hashes
are in [performance-fixtures.json](performance-fixtures.json).

| Rust release workload | Words / UTF-8 bytes | Full median | Scene edit median | Reorder median |
| --- | --- | ---: | ---: | ---: |
| Four-passage cycle | 85,280 / 474,880 | 406.59 ms | 22.43 ms | 1.47 ms |
| Varied templates | 104,455 / 571,260 | 58.26 ms | 12.53 ms | 0.42 ms |

Full analysis ranged 404.15-423.43 ms and 58.10-70.77 ms. Retained cache estimates were
835,102 and 840,528 bytes against a 16 MiB budget; peak RSS from `/usr/bin/time -l` was
68,091,904 and 73,842,688 bytes. Raw samples: [four-passage cycle](performance-rust.json),
[varied templates](performance-rust-varied.json).

| Node debug-addon workload | Full median | Scene edit median | Minimum timer ticks, full / edit | Maximum timer gap | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four-passage cycle | 3,409.28 ms | 197.37 ms | 1,320 / 73 | 12.89 ms | 133,760 KiB |
| Varied templates | 833.16 ms | 112.36 ms | 331 / 44 | 4.16 ms | 140,000 KiB |

Full analysis ranged 3,294.16-3,514.22 ms and 831.22-919.29 ms; median input conversion
took 0.31 and 0.93 ms. Raw samples: [four-passage cycle](performance-node.json),
[varied templates](performance-node-varied.json).

Timings apply to these workloads, profiles, and machine: word count alone does not
predict latency, the Node debug addon and Rust release build use different profiles,
and JS conversion still runs on the event loop. A release-blocker rerun of the varied
workload in the Rust release profile measured 58.27 ms full, 12.59 ms edit, and 0.45 ms
reorder; the release Node addon measured 59.97 ms full and 14.13 ms edit
([Rust samples](benchmarks/evidence/remediation-performance-rust.json),
[Node samples](benchmarks/evidence/remediation-performance-node.json)).

### Spelling and corpus-embedding components

Identical bounded workloads before and after the indexed spelling and sparse PPMI
changes, Rust 1.96 release builds on the same Apple M4. Medians exclude compilation;
MB means decimal megabytes requested through Rust's allocator, not process RSS.

| Workload | Before | After | Peak requested bytes before → after |
| --- | ---: | ---: | ---: |
| 20 warmed spelling queries | 11.91 ms | 0.181 ms | 13.0 KB → 4.5 KB |
| 2500 terms, 100 dimensions | 208.66 ms | 71.35 ms | 101.23 MB → 11.95 MB |
| 4096 terms, 100 dimensions | 632.16 ms | 120.35 ms | 270.45 MB → 19.54 MB |
| 1000 terms, mixed contexts | 40.01 ms | 39.91 ms | 16.50 MB → 5.46 MB |

The spelling trie adds about 1.09 MB of retained dictionary data after its first lookup
(2 MiB transient peak) and preserves the prior distance/frequency/lexical ranking
against a scan oracle. The 512-term embedding case is about 9% slower, the 128-term
exact path preserves vectors byte-for-byte, and the large-space solver alternates the
matrix and its transpose with QR reorthogonalization because smoothed PPMI is
nonsymmetric. Words and passages with zero learned vectors abstain from semantic
rankings; hybrid document search still retrieves exact terms lexically. Run
`cargo run --locked --release --example component_performance -- spelling 7` or
`-- embedding 2500 100 5` (`embedding-mixed` selects the denser corpus).
[Full comparisons](benchmarks/evidence/component-performance-remediation.json) retain all
eight workloads and raw samples.

### Pinned pretrained-model parity

`component_real_model` and `scripts/component_real_model_reference.py` compare the
strict embedding profile with an independently executed Python ONNX Runtime reference
on the pinned [all-MiniLM-L6-v2 export](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41).
Weights stay in an ignored cache (`--download` fetches them) and Python dependencies
are pinned in PEP 723 metadata run with `uv run --python 3.11`.

- The original tokenizer declares fixed padding and truncation, so both strict loaders reject it with `model_profile_tokenizer_settings`. Successful runs use a separately pinned derived tokenizer with those settings set to null; the manifest keeps the upstream 128-token limit.
- Native ONNX Runtime 1.31.0 and WASI tract 0.23.5 each produced 16 pooled 384-dimensional vectors across eight batches with maximum absolute errors of 2.3842×10⁻⁶ and 2.3544×10⁻⁶ against Python ONNX Runtime 1.27.0. Token IDs, masks, and type IDs matched exactly; 127- and 128-token inputs succeeded and 129 tokens failed without truncation.
- `scripts/prepare_ner_reference.py` and `scripts/ner_reference_parity.py` apply the same check to a pinned NER export ([ner-reference.json](benchmarks/ner-reference.json)), comparing verified native BIO spans with independent Python execution.

These checks verify preprocessing and runtime fidelity for these exports only, not the
full SentenceTransformer pipeline, other exports, editorial accuracy, or runtime speed.
[Commands, hashes, and limits](benchmarks/evidence/component-real-model-manifest.json) and the
[independent comparison](benchmarks/evidence/component-real-model-parent-comparison.json)
retain the evidence.

### World-state retention

`component_world_state` measures full snapshots, compact current state, and compact
state with replayable deltas. At 1,024 synthetic scenes and 2,048 final facts, five
release samples on the Apple M4 gave:

| Output mode | Median | Retained requested allocation | Peak additional requested allocation |
| --- | ---: | ---: | ---: |
| Full snapshots | 30.48 ms | 151.52 MB | 152.00 MB |
| Current state with event history | 2.82 ms | 0.43 MB | 0.61 MB |
| Current state, event history and net deltas | 3.06 ms | 0.68 MB | 0.83 MB |

Final facts, violations, knowledge, event histories, and scores have identical semantic
fingerprints in all three modes; compact output omits cumulative snapshots and, by
default, event histories. This is an output-retention choice, not a same-output
speedup. Run `cargo run --locked --release --example component_world_state -- 1024 5 current`
or select `full`, `deltas`, `minimal`, and the optional `activity` workload.
[Raw measurements](benchmarks/evidence/component-world-state-after.json) and the
[manifest](benchmarks/evidence/component-world-state-manifest.json) record the
oversized-name regression, static errors, and delta replay checks.
