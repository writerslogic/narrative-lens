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
import { computeReadability, checkGrammar, listStructureTemplates } from 'narrative-lens';
```

The current bindings (`src/napi_bindings.rs`) expose a small, deliberately incomplete
slice of the full analyzer set: `computeReadability`, `checkGrammar`, and
`listStructureTemplates`. It's a thin conversion layer over the pure-Rust analyzers,
not a 1:1 mirror of every function in `craft`/`structure`/`continuity`/`substrate` --
add a new `#[napi]` function there, following the existing pattern (a small `*Js`
struct + a `From<CoreType>` impl), whenever a JS consumer needs one that isn't
exposed yet.

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
