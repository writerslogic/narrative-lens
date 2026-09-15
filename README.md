<img src="https://raw.githubusercontent.com/writerslogic/narrative-lens/main/assets/logo-black.svg" alt="narrative-lens logo" width="120" align="left">

<h1>narrative-lens</h1>

<p><strong>Local, deterministic prose and narrative-craft analysis — readability, pacing, voice, foreshadowing, continuity — with no LLM and no network call. Extracted from Emathy's meaning engine, for Rust and Node.</strong></p>

<br clear="left">

[![CI](https://img.shields.io/github/actions/workflow/status/writerslogic/narrative-lens/ci.yml?style=flat-square&labelColor=20232a&branch=main&label=CI)](https://github.com/writerslogic/narrative-lens/actions/workflows/ci.yml) [![CodeQL](https://img.shields.io/github/actions/workflow/status/writerslogic/narrative-lens/codeql.yml?style=flat-square&labelColor=20232a&branch=main&label=CodeQL)](https://github.com/writerslogic/narrative-lens/actions/workflows/codeql.yml) [![OpenSSF Scorecard](https://img.shields.io/ossf-scorecard/github.com/writerslogic/narrative-lens?style=flat-square&labelColor=20232a&label=OpenSSF)](https://securityscorecards.dev/viewer/?uri=github.com/writerslogic/narrative-lens) [![License](https://img.shields.io/github/license/writerslogic/narrative-lens?style=flat-square&labelColor=20232a&color=007ec6&label=license)](https://github.com/writerslogic/narrative-lens/blob/main/LICENSE) [![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-Contributor%20Covenant%202.1-6a4c93?style=flat-square&labelColor=20232a)](https://github.com/writerslogic/narrative-lens/blob/main/CODE_OF_CONDUCT.md)

<a href="https://www.npmjs.com/package/narrative-lens">
    <img src="https://img.shields.io/npm/v/narrative-lens.svg?style=flat-square&labelColor=20232a&color=007ec6" alt="npm version"/>
  </a>
  <img src="https://img.shields.io/npm/dm/narrative-lens.svg?style=flat-square&labelColor=20232a&color=007ec6" alt="npm downloads"/>
  <a href="https://crates.io/crates/narrative-lens">
    <img src="https://img.shields.io/crates/v/narrative-lens.svg?style=flat-square&labelColor=20232a&color=007ec6" alt="crates.io version"/>
  </a>
  <a href="https://docs.rs/narrative-lens">
    <img src="https://img.shields.io/docsrs/narrative-lens?style=flat-square&labelColor=20232a&color=007ec6" alt="docs.rs"/>
  </a>
  <a href="https://github.com/writerslogic/narrative-lens">
    <img src="https://img.shields.io/github/stars/writerslogic/narrative-lens?style=flat-square&labelColor=20232a&color=6a4c93" alt="stars"/>
  </a>
</p>

<p align="center">
  <a href="#install">Install</a> &middot;
  <a href="#what-it-analyzes">What It Analyzes</a> &middot;
  <a href="#api">API</a> &middot;
  <a href="#guides">Guides</a> &middot;
  <a href="#contributing">Contributing</a>
</p>

---

narrative-lens reads a passage or a whole manuscript and returns structured, explainable judgments about it — a readability grade, a pacing score for a scene, whether a promise made in Chapter 2 pays off by Chapter 20, whether a character's established backstory contradicts what a later chapter states. No API key, no network call. Every result carries a confidence score and, where the analysis supports it, an explanation of what evidence produced it.

It's the analysis layer originally built inside Emathy, a native macOS authoring app, factored out into its own crate so any tool working with manuscript text — not just Emathy — can use the same local analyzers instead of reaching for an LLM call for judgments a deterministic model can make faster, cheaper, and more consistently.

> **You give it:** a scene from Chapter 12.
> **It gives you back:** Flesch-Kincaid grade 8.2, a pacing flag on four consecutive paragraphs of interior monologue with no dialogue or action, and a note that the scene's foreshadowing payoff rate is lower than the manuscript's average.

Works with Rust 1.75+ and Node.js 18+, on macOS, Linux, and Windows.

## Install

```bash
npm install narrative-lens
```

```toml
[dependencies]
narrative-lens = "0.1"
```

<details>
<summary><strong>Optional: local ONNX models for embeddings, NER, NLI, and emotion</strong></summary>

The default analyzers are pure Rust with no model to load. Enable the `onnx` feature for SBERT sentence embeddings (semantic theme/genre detection), BERT NER (entity resolution), NLI (contradiction detection), and Ekman-7 emotion classification instead of the built-in heuristics — "bring your own model, no download": point each analyzer at a local `model.onnx` + `tokenizer.json`, same contract as `holographic-memory`'s local embedder. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/INTEGRATION.md](docs/INTEGRATION.md).

</details>

## What It Analyzes

**Prose craft** — readability, pacing, prose quality, dialogue realism, voice consistency, subtext, sentence-level syntax tension, lexical variety, grammar.

**Narrative structure** — foreshadowing and promise/payoff tracking across the whole manuscript, thematic argument and theme presence, narrative-entropy (how predictable the next beat is), opening-page analysis, genre-convention compliance.

**Continuity and world state** — timeline consistency, world-state tracking (what a character knows/has/believes at a given point), coreference and named-entity resolution, causal-chain and epistemic-state checking (does a character act on knowledge they shouldn't have yet), anachronism detection, gap analysis (a setup with no payoff, or a payoff with no setup).

Every analyzer in these three groups is a pure function: manuscript text in, a structured, confidence-scored, explainable result out. None of them persist anything or require a project structure — that's the caller's job.

## API

<details>
<summary><strong>Rust</strong> -- analyzers, types, feature flags</summary>

| Module | What it does |
|--------|-------------|
| `craft::*` | Readability, pacing, prose quality, dialogue, voice, subtext, syntax tension, lexical, grammar |
| `structure::*` | Foreshadowing, promise/payoff, theme, thematic argument, narrative entropy, opening analysis, genre compliance |
| `continuity::*` | Timeline, world state, coreference, entity/NER, causal chains, epistemic state, mental models, gap analysis, anachronism |
| `substrate::*` | Shared primitives: text segmentation, embeddings, the meaning graph, confidence scoring, explanation generation |

Each analyzer module exposes an `analyze(text: &str, options: &Options) -> Result<AnalysisResult>` entry point returning a confidence-scored, explainable result specific to that analyzer.

</details>

<details>
<summary><strong>Node.js</strong> -- the same analyzers, generated bindings</summary>

Every Rust analyzer is exposed through the napi binding under the matching category (`craft`, `structure`, `continuity`). TypeScript definitions in `bindings/` are generated from the Rust types via `ts-rs` — treat them as the source of truth over this table.

</details>

## Guides

- **[Architecture](./docs/ARCHITECTURE.md)** -- the analyzer categories, the shared substrate, and what's reserved vs. shipped
- **[Integration](./docs/INTEGRATION.md)** -- using narrative-lens from Rust, Node, and as a local analysis backend for an MCP server
- **[Contributing](./CONTRIBUTING.md)** -- development setup and conventions

## Requirements

- **Rust 1.75+** or **Node.js 18+**
- No API key, no network access, no external service of any kind

## Development

```bash
git clone https://github.com/writerslogic/narrative-lens.git
cd narrative-lens
cargo test                          # all analyzer categories, Rust side
npm install && npm test             # node --test tests/node
npm run test:types                  # strict TS check against generated bindings
```

## Why narrative-lens

Most manuscript-analysis tooling reaches for an LLM call for judgments that don't need one — a Flesch-Kincaid grade, a pacing heuristic, whether a named entity in Chapter 20 was ever introduced. Those are deterministic questions with deterministic answers, and answering them with an API call adds cost, latency, and a dependency on a third party actually seeing your unpublished manuscript. narrative-lens exists so a consuming tool — an MCP server, an editor, a CI check on a manuscript repo — can get those answers locally, instantly, and for free, and reserve the LLM for judgments that actually need one: does this scene's prose sound like this author, would a human editor find this paragraph moving.

## Contributing

We welcome contributions of all sizes. Check the [issue tracker](https://github.com/writerslogic/narrative-lens/issues) for `good first issue` labels, or see [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup.

**Areas where help is especially welcome:**
- Additional language support for the craft and structure analyzers (currently English-tuned)
- Evaluation fixtures for the continuity analyzers against real manuscript excerpts
- Integration examples for MCP servers and other manuscript tooling beyond Emathy and scrivener-mcp

## Security

Found a vulnerability? Please report it privately — see [SECURITY.md](./SECURITY.md).

## License

Apache-2.0 &copy; [WritersLogic, Inc.](https://github.com/writerslogic)

<p align="center">
  <a href="https://github.com/writerslogic/narrative-lens">GitHub</a> &middot;
  <a href="https://www.npmjs.com/package/narrative-lens">npm</a> &middot;
  <a href="https://crates.io/crates/narrative-lens">crates.io</a> &middot;
  <a href="https://github.com/writerslogic/narrative-lens/issues">Issues</a> &middot;
  <a href="./CHANGELOG.md">Changelog</a>
</p>
