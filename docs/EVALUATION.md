# Accuracy and performance evaluation

| Measurement | Data | What it does and does not show |
| --- | --- | --- |
| [Finding-level accuracy](#finding-level-evaluation) | 52 agent-authored fiction cases | Supported patterns, controls, and regressions; not editorial accuracy on published novels |
| [Analyzer-family baseline](#original-analyzer-family-baseline) | 56 older cases | Binary any-finding detection per family |
| [LitBank persons and coreference](#litbank-benchmark) | 100 public-domain novel excerpts: 76 development, 24 holdout scored once | Person spans, cast inventory, and pronoun resolution against human annotations |
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

## LitBank benchmark

[LitBank](https://github.com/dbamman/litbank) (Bamman et al., 2019 and 2020, CC-BY-4.0)
annotates entities, quotations, and coreference in 100 English novel excerpts.
[litbank-source.json](benchmarks/litbank-source.json) pins the repository revision,
file hashes, and an author-grouped split: a seeded hash assigns about one fifth of the
authors to `final_holdout` and the rest to `development`.

**The development split (76 excerpts, 57 authors) is the tuning surface. The holdout
split (24 excerpts, 18 authors) was scored once, on 2026-10-29 at commit 548a033
(v0.2.0 plus two performance-only commits); its labels were never inspected and it is
not re-scored while tuning.**

```sh
uv run --no-project scripts/fetch_literary_benchmark.py --download     # pinned corpus into target/external-validation
uv run --no-project scripts/literary_benchmark.py prepare --folder target/external-validation/litbank-prepared
cargo run --locked --release --example evaluate_literary -- target/external-validation/litbank-prepared/inputs.json > predictions.jsonl
cargo run --locked --release --features onnx --example evaluate_literary -- target/external-validation/litbank-prepared/inputs.json --ner-cache MODEL_DIR > predictions-ner.jsonl
uv run --no-project scripts/literary_benchmark.py score --folder target/external-validation/litbank-prepared --predictions predictions.jsonl --split development
```

`prepare` and `score` never modify the pinned labels or train anything. The example
runs the heuristic `continuity::ner` and `CoreferenceResolver` paths, or the verified
local NER model when `--ner-cache` points at a pinned export. The scored runs are kept
as [benchmarks/evidence/literary-development-{heuristic,ner}-v6.json](benchmarks/evidence/)
and [benchmarks/evidence/literary-holdout-{heuristic,ner}-v1.json](benchmarks/evidence/);
new runs write to `target/evaluation/` by default. Both the example and the script
refuse `final_holdout` without `--finalize`, so a holdout run is always a deliberate
one-off.

| Metric | Heuristic, dev | Heuristic, holdout | With NER model, dev | With NER model, holdout |
| --- | ---: | ---: | ---: | ---: |
| Proper person name spans, F1 | 0.354 | 0.380 | 0.598 | 0.556 |
| Named character inventory, F1 | 0.399 | 0.368 | 0.636 | 0.552 |
| Model person spans, exact, F1 | n/a | n/a | 0.816 | 0.843 |
| Coreference with supplied cast, P / R / F1 | 0.779 / 0.325 / 0.458 | 0.759 / 0.301 / 0.431 | 0.779 / 0.325 / 0.458 | 0.759 / 0.301 / 0.431 |
| Coreference with detected cast, P / R / F1 | 0.626 / 0.252 / 0.359 | 0.543 / 0.205 / 0.298 | 0.671 / 0.284 / 0.399 | 0.734 / 0.296 / 0.422 |
| Dialogue spans with supplied cast, F1 | 0.854 | 0.933 | 0.854 | 0.933 |
| Explicit speaker attribution with supplied cast, P / R | 0.982 / 0.128 | 0.870 / 0.112 | 0.982 / 0.128 | 0.870 / 0.112 |

Development: v6, 2026-10-28, 76 excerpts. Holdout: v1, 2026-10-29, 24 excerpts.
Holdout figures land within 0.09 of the development figures in both directions, except
explicit speaker attribution precision, which drops 0.11 on 23 decisions.

Coreference abstains on ambiguity by design, which is why precision stays well above
recall. Gold pronouns outside any cast entry (about 22%) and plural `they` (about 12%)
cap heuristic coreference near F1 0.46-0.48 on the development split; substantially
higher recall needs a neural coreference model. The LitBank text has one sentence per
line and no blank-line paragraphs, so paragraph-scoped heuristics see one paragraph per
excerpt.

### Literary fine-tune of the NER model (candidate v7)

The pinned export is `dslim/distilbert-NER`, trained on news. `scripts/ner_finetune.py`
fine-tunes it on the development split's proper-person spans and exports a drop-in
replacement (same tokenizer, same nine-label manifest profile, int8 dynamic
quantization, 65.6 MB). `cv` trains five author-disjoint folds so every number below
is measured on authors the model never saw; `export` trains on all 76 works.

```sh
uv run --no-project --with torch --with transformers --with onnx --with onnxruntime scripts/ner_finetune.py cv
uv run --no-project --with torch --with transformers --with onnx --with onnxruntime scripts/ner_finetune.py export
```

| Exact proper-person spans, held-out authors (pooled over 5 folds) | P | R | F1 |
| --- | ---: | ---: | ---: |
| Pinned news-trained export | 0.643 | 0.677 | 0.660 |
| Fine-tuned on LitBank development works | 0.843 | 0.886 | 0.864 |

Record: [benchmarks/evidence/ner-litbank-cv-v1.json](benchmarks/evidence/ner-litbank-cv-v1.json)
(4 epochs, learning rate 5e-5, seed 1). The full pipeline run on the development split
with the exported model is not a clean measurement, because those works are its training
data; for the record it moved model spans F1 0.816 to 0.934, proper person names 0.598 to
0.642, and the named inventory 0.636 to 0.663, with supplied-cast coreference unchanged.
The pinned model stays the default until the candidate is hosted and scored once on the
holdout split; the holdout labels were not used anywhere in training.

### Multi-task narrative model (candidate)

`scripts/narrative_corpus.py` builds one training corpus from LitBank's development
split (proper, nominal, and pronoun person mentions, coreference clusters, quotations
with speaker clusters) and from the [Project Dialogism Novel Corpus](benchmarks/pdnc-source.json)
(CC BY 4.0; 24 of its 28 novels, because the four whose LitBank excerpt is in the holdout
split are excluded from every use). `scripts/multitask_train.py` trains one DistilBERT
encoder with four heads and exports a single int8 ONNX graph:

| Head | Output | Trained on |
| --- | --- | --- |
| `entity` | 9-label BIO token tags, the pinned NER manifest's label set | LitBank proper person spans |
| `nominal` | O / B-NOM / I-NOM token tags for "her mother", "the old man" | LitBank nominal person spans |
| `coref` | antecedent score for every earlier mention in a 400-token window, plus a no-antecedent column | LitBank clusters; PDNC alias mentions |
| `attribution` | speaker score for every person mention near a quotation, plus a speaker-not-in-window column | LitBank and PDNC quotations |

```sh
uv run --no-project scripts/narrative_corpus.py litbank
uv run --no-project scripts/narrative_corpus.py pdnc        # downloads and verifies the pinned revision
uv run --no-project --with torch --with transformers --with onnx --with onnxruntime scripts/multitask_train.py cv --sources litbank
uv run --no-project --with torch --with transformers --with onnx --with onnxruntime scripts/multitask_train.py export --sources litbank pdnc
```

Five author-disjoint folds on LitBank alone, pooled
([benchmarks/evidence/narrative-multitask-cv-litbank-v1.json](benchmarks/evidence/narrative-multitask-cv-litbank-v1.json),
3 epochs, learning rate 3e-5, 400-token windows):

| Measurement on unseen authors | Value |
| --- | ---: |
| Proper person spans, exact F1 | 0.772 |
| Nominal person mentions, exact F1 | 0.514 |
| Antecedent accuracy, pronouns (11,604) | 0.841 |
| Antecedent accuracy, nominal mentions (4,593) | 0.754 |
| Antecedent accuracy, proper names (2,003) | 0.610 |
| Quotation speaker accuracy (1,037 quotes) | 0.312 |

Antecedent accuracy counts a mention as correct when its top-scored earlier mention in
the window is in the same gold cluster, or when it correctly scores no antecedent; it is
a within-window measure, not CoNLL coreference F1.

Adding PDNC (40 quotation-bearing windows per novel) and re-running the first fold with
its authors' novels held out as well
([benchmarks/evidence/narrative-multitask-cv-litbank-pdnc-fold0-v1.json](benchmarks/evidence/narrative-multitask-cv-litbank-pdnc-fold0-v1.json)):
speaker accuracy over 2,100 held-out quotations from both corpora was 0.426, against
0.225 on that fold's LitBank quotations alone, while the token heads lost about 0.04
(entity 0.718, nominal 0.474) because PDNC windows carry no span labels.

The crate consumes the export as `ModelKind::NarrativeMultitask` (profile
`narrative_multitask_v1`; the manifest lists the entity head under `labels` and the
nominal head under `nominalLabels`). `continuity::narrative::analyze` tokenizes once,
cuts 400-token windows, runs the graph twice per window (first for proper and nominal
spans, then with those spans plus the resolver's third-person pronouns as mention
candidates and the quotation ranges), decodes each mention's best earlier mention and
each quotation's best speaker greedily inside the window, and merges clusters whose
proper-name surfaces are identical across windows. `CoreferenceResolver::resolve_checked`
and `dialogue::extract_dialogue_checked` call it first; without a configured model, or
when it fails at runtime (`model_fallback:<code>` in diagnostics), the heuristic paths run
unchanged. Explicit speech tags still win over the model's speaker. Confidences are
softmax probabilities of the chosen link, not calibrated accuracies. The export keeps the
batch axis at 1 and drops intermediate shape annotations because tract (the wasm runtime)
cannot shape the encoder's Flatten over a symbolic batch × sequence; a native tract probe
loads and runs such a graph, while the LitBank NER fine-tune export above still carries
the symbolic batch axis and does not load in tract.

```sh
cargo run --locked --release --features onnx --example evaluate_literary -- target/external-validation/litbank-prepared-v4/inputs.json --ner-cache NER_DIR --narrative-model MODEL_DIR > predictions.jsonl
NARRATIVE_LENS_NARRATIVE_MODEL=MODEL_DIR cargo test --features onnx --test component_narrative_model
uv run scripts/narrative_model_reference.py MODEL_DIR   # onnxruntime decode of the test passage
```

`scripts/multitask_train.py` also runs a teacher-student program: `--teacher-base` trains a
larger encoder (DeBERTa-v3-large) on the same training documents first, the teacher then
labels every student window (PDNC carries no span labels of its own, so its 1.9M words get
entity and nominal distributions this way), and the student trains on gold plus the
teacher's distributions through its own tokenizer, aligned by character offsets. `--offsets`
adds shifted window grids, `--litbank-repeat` oversamples the span-labelled works, and
`--teacher-cache` keeps per-fold teachers so a second student reuses them. The export's
`windowTokens` records the training window the crate must use. DeBERTa itself is not a
deployable student: its ONNX export carries a data-dependent `If` that tract rejects and is
244 MB at int8; RoBERTa-base (126 MB) and BERT-base (110 MB) both load in tract.

Inference runs windows at half-window stride (a mention is decided in the window that
gives it the most left context, a quotation in the window it sits most centrally in), names
a cluster by its most frequent proper surface or, failing that, its most frequent nominal
surface, and ends with an alias pass: the antecedent head scores the list of distinct
proper surfaces ("Elizabeth . Miss Bennet . Mr. Darcy") and merges two surfaces' clusters
when it links them with probability 0.5 or more. `scripts/narrative_corpus.py pdnc` also
writes `pdnc_alias.jsonl`, shuffled alias lists from PDNC's `character_info.csv` clustered
by character, so the head is trained on that distribution. `scripts/gutenberg_corpus.py`
pins 100 public-domain novels ([benchmarks/gutenberg-source.json](benchmarks/gutenberg-source.json);
English prose fiction, lowest Gutenberg ids first, LitBank source novels excluded by id)
as unlabelled text: the teacher's token heads give them mention spans, paired quotation
marks give quotations, and the student learns from the teacher's distributions alone on
those windows (`--unlabeled-windows` caps how many join a run).

The distilled student is published as
[Writerslogic/narrative-multitask](https://huggingface.co/Writerslogic/narrative-multitask) (revision `v1`;
[benchmarks/narrative-model-source.json](benchmarks/narrative-model-source.json) pins the file hashes).
The crate still bundles and downloads nothing: fetch it with
`hf download Writerslogic/narrative-multitask --revision v1 --local-dir DIR` and pass `DIR` to
`evaluate_literary --narrative-model`, or load the files with `set_model_bytes_verified`.

Pipeline scores on the development split, each run loading the model beside the pinned
NER model (evidence `literary-development-narrative-v{7,8,9}.json` against
[literary-development-ner-v6.json](benchmarks/evidence/literary-development-ner-v6.json)).
Every model trained on these works, so the figures show the integration end to end, not
generalization; v7 is the DistilBERT candidate, v8 the distilled RoBERTa student with
disjoint windows, v9 the same student with overlapping windows, nominal cluster names, and
the alias pass:

| Development split, 76 works | pinned NER (v6) | v7 | v8 | v9 |
| --- | ---: | ---: | ---: | ---: |
| Supplied-cast pronoun resolution F1 (precision) | 0.458 (0.779) | 0.488 (0.730) | 0.625 (0.860) | 0.786 (0.797) |
| Automatic-cast pronoun resolution F1 (precision) | 0.399 (0.671) | 0.482 (0.721) | 0.615 (0.846) | 0.781 (0.792) |
| Explicit-mode speaker F1 (precision) | 0.226 (0.982) | 0.363 (0.519) | 0.534 (0.788) | 0.657 (0.716) |
| Heuristic-mode speaker F1 (precision) | 0.335 (0.599) | 0.409 (0.561) | 0.559 (0.763) | 0.692 (0.774) |
| Unresolved gold pronouns | | | 2,961 | 190 |

Person spans and the cast inventory are unchanged because the pinned NER model still
produces them. Confidences are softmax probabilities of the chosen link and each line
carries its probability for callers that prefer precision.

The unseen-author check for the distilled student is the first fold of the
author-disjoint cross-validation on LitBank plus PDNC
([benchmarks/evidence/narrative-multitask-cv-apex-roberta-fold0-v1.json](benchmarks/evidence/narrative-multitask-cv-apex-roberta-fold0-v1.json);
teacher DeBERTa-v3-large, 2 epochs, student RoBERTa-base, 3 epochs, 510-token windows,
offsets 0 and 255, all PDNC windows, LitBank repeated 4x, entity weight 3, nominal 2):

| Fold 0, unseen authors | DistilBERT (weight 3) | RoBERTa student | DeBERTa-large teacher |
| --- | ---: | ---: | ---: |
| Proper person spans, exact F1 | 0.811 | 0.812 | 0.829 |
| Nominal person mentions, exact F1 | 0.536 | 0.704 | 0.641 |
| Antecedent accuracy, pronouns | 0.834 | 0.908 | 0.923 |
| Quotation speaker accuracy (LitBank / PDNC quotes) | 0.419 | 0.814 (0.561 / 0.819) | 0.863 |

GUM was considered for coreference but deferred: its annotations are CC BY 4.0 while the
fiction, essay, letter, and podcast texts are CC BY-NC-SA and Reddit is non-commercial.

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
- Native ONNX Runtime 1.31.0 and WASI tract 0.23.5 each produced 16 pooled 384-dimensional vectors across eight batches with maximum absolute errors of 2.3842×10⁻⁶ and 2.3544×10⁻⁶ against Python ONNX Runtime 1.27.0. Token IDs, masks, and type IDs matched exactly; 127- and 128-token inputs succeeded and 129 tokens failed without truncation. The WASI side runs the `component_real_model` example built for `wasm32-wasip1` through `tests/component_real_model_wasi.cjs` (a manual Node WASI launcher that preopens the model cache; the CI runner cannot drive it because it needs the downloaded weights).
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
