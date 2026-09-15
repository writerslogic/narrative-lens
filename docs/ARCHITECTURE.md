# Architecture

## Analyzer categories

- **`craft`** -- sentence- and paragraph-level prose quality: readability, pacing, dialogue
  realism, voice consistency, subtext, syntax tension, lexical variety, grammar.
- **`structure`** -- manuscript-level narrative shape: foreshadowing and promise/payoff
  tracking, theme and thematic argument, narrative entropy, opening-page analysis, genre
  compliance.
- **`continuity`** -- cross-chapter consistency: timeline, world state, coreference and
  entity resolution, causal chains, epistemic state, gap analysis, anachronism detection.
- **`substrate`** -- shared primitives every category above depends on: text segmentation,
  embeddings, confidence scoring, and explanation generation.

## Design rule

Every public analyzer is a pure function: `analyze(text: &str, options: &Options) ->
Result<AnalysisResult>`. No analyzer touches the filesystem, the network, or shared mutable
state across calls. This is what makes the crate embeddable in anything -- an MCP server, a
CLI, a CI check on a manuscript repo -- without dragging along Emathy's project model,
storage, or versioning.

## Out of scope: Emathy's meaning graph

Emathy's `meaning_graph` (session-scoped agency/branch/expectation/intent/memory
tracking over a manuscript's revision history) was never ported here. It isn't a
standalone analyzer -- it's wired directly into Emathy's orchestration engine
(60+ call sites into `agency`, `branches`, `expectation`, `intent`, `memory`,
`orchestrator`, `research`, `tags`), which contradicts this crate's pure-function
design rule below. A narrative-lens equivalent would be a new design, not an
extraction, and isn't planned.

## Provenance

This crate began as an extraction from Emathy's `emathy-core`, a native macOS authoring
app. Emathy's own storage, branching, project, and export machinery is not part of
this crate and was never intended to be -- see the project README's "Why narrative-lens"
section for the reasoning.

## Local ONNX path

The `onnx` feature enables model-backed analysis: SBERT sentence embeddings
(`substrate::sbert`, used by `structure::{theme,narrative_entropy,genre}` for
semantic similarity), BERT NER (`continuity::ner`), NLI contradiction detection
(`substrate::nli`), and Ekman-7 emotion classification (`craft::emotion`).
Native (`not(target_arch = "wasm32")`) runs `ort` (ONNX Runtime); `wasm32` runs
`tract`, a pure-Rust CPU runtime.

Follows a "bring your own model, no download" contract, matching
`holographic-memory`'s local embedder: no model weights are bundled or fetched
by this crate. Point `set_model_directory` (per analyzer) at a directory
containing `model.onnx` + `tokenizer.json`, or leave it unset -- every analyzer
degrades to its non-model fallback (or empty output) when no model is found,
never panics. With the `onnx` feature disabled entirely, the same degrade
happens at compile time via a null backend, so no caller needs its own
`#[cfg]`.

Recommended checkpoints (not bundled -- fetch and export to ONNX yourself):
- SBERT: `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, 384-dim, ~90MB).
- NER: `dslim/bert-base-NER` (MIT) -- a maintained ONNX export exists as
  `onnx-community/bert-base-NER-ONNX`.
- NLI: `cross-encoder/nli-deberta-v3-xsmall` (22M params) rather than a
  `*-mnli-large` checkpoint -- comparable accuracy at a fraction of the size;
  an ONNX build exists as `Xenova/nli-deberta-v3-xsmall`.
- Emotion: `j-hartmann/emotion-english-distilroberta-base` (Ekman-7 labels,
  positionally aligned with `craft::emotion::EKMAN_LABELS`).
