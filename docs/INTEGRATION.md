# Integration

## As a Rust crate

```toml
[dependencies]
narrative-lens = "0.1"
```

Every analyzer is a plain function under `craft::`, `structure::`, `continuity::`, or
`substrate::` -- there is no facade type to construct first. Call the one you need:

```rust
use narrative_lens::craft::readability;

let scene = "The rain fell steadily against the window pane.";
let result = readability::analyze(scene);
```

Check each module's own doc comments for its exact signature; return types are
analyzer-specific structs, not a single unified `AnalysisResult` (see
[ARCHITECTURE.md](./ARCHITECTURE.md)).

### The `onnx` feature

Off by default -- a plain `Cargo.toml` dependency pulls in only `serde`, `regex`,
`log`, `aho-corasick`, `petgraph`, and `nalgebra`. Enable `onnx` for
model-backed SBERT/NER/NLI/emotion analysis:

```toml
narrative-lens = { version = "0.1", features = ["onnx"] }
```

```rust
narrative_lens::substrate::sbert::set_model_directory("/path/to/models");
// looks for /path/to/models/sbert-onnx/{model.onnx,tokenizer.json}
```

No model, or the feature disabled entirely: every model-backed function degrades to
empty output rather than erroring. See [ARCHITECTURE.md](./ARCHITECTURE.md) for the
per-analyzer model directory names and recommended checkpoints.

## As a Node.js package

```bash
npm install narrative-lens
```

```js
import { computeReadability, trackWorldState, buildTimeline } from 'narrative-lens';
```

The bindings (`src/napi_bindings/`, one file per analyzer category) expose most of the
`craft`/`structure`/`continuity`/`substrate` analyzer surface -- 59 functions as of this
writing. Two return conventions coexist:

- The original 3 bindings (`computeReadability`, `checkGrammar`, `listStructureTemplates`,
  in `napi_bindings/mod.rs`) hand-write a `*Js` DTO struct + `From<CoreType>` impl per
  function, giving precise generated TypeScript types.
- Everything else goes through a JSON bridge (`serde_json::to_value(result)`, return type
  `napi::Result<serde_json::Value>`) -- this scales to the many-field, deeply-nested
  return types the rest of the crate has, at the cost of `index.d.ts` typing those
  returns as `any` rather than a precise interface. Every struct/enum that crosses this
  bridge derives `Serialize` and `#[serde(rename_all = "camelCase")]`, so JS callers see
  the same field-naming convention either way.

Not exposed: `substrate::{text,utils,nlp,wordfreq,common_types,nli,sbert}` (implementation
substrate other analyzers call internally, not consumer-facing on their own), `continuity::ner`
and the onnx-only paths inside `structure::{theme,narrative_entropy,genre}` (model-backed,
see the `onnx` feature section above -- separate concern from this binding layer), a few
functions that mutate a caller-owned buffer in place (`pacing::enrich_with_info_density`,
`timeline::apply_overrides` -- no good JS shape for that), and `voice::{build_voice_profile_py,compare_voices_py}`
(redundant `_py`-suffixed duplicates). Add a new `#[napi]` function in the relevant
`napi_bindings/<category>.rs`, following the JSON-bridge pattern above, whenever a JS
consumer needs one of these.

To build locally: `npm run build:debug` (or `npm run build` for a release build) runs
`napi build --platform --features node-api`, which compiles the native addon and
regenerates `index.js` (the platform-detecting CommonJS loader) and `index.d.ts`
(TypeScript definitions, generated directly from the `#[napi]` attributes -- no
separate schema-generation step). `index.mjs` is a small hand-maintained ESM wrapper
around `index.js`, since a native addon loads via `require`; keep its named exports in
sync with `index.d.ts` when adding a binding. The platform-specific `.node` binary
itself is gitignored -- published per-platform via `napi artifacts`/`napi prepublish`,
not committed to source.

The `ts-rs`/`schemars` `bindings` feature is a separate, unrelated mechanism (JSON
Schema / pure-type generation for non-napi consumers) and is not part of this path.

## Embedding narrative-lens in a tool

The crate has no filesystem, network, or project-model dependency of its own (see the
design rule in [ARCHITECTURE.md](./ARCHITECTURE.md)), so it drops into anything that
can hand it manuscript text as `&str`:

- **An MCP server**: call the relevant analyzer per tool invocation; no state to
  manage between calls.
- **A CI check on a manuscript repo**: run continuity analyzers (`timeline`,
  `world_state`, `entity`) against a diff to catch consistency regressions before
  merge.
- **An editor plugin**: run `craft` analyzers on the active scene on save/idle for
  inline readability/pacing/voice feedback.

None of these need the `onnx` feature; it only matters if you want the
semantic-similarity-backed analyzers (`theme`, `narrative_entropy`, `genre`'s
SBERT path) or model-backed NER/NLI instead of the built-in heuristic fallbacks.
