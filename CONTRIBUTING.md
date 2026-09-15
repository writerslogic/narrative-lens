# Contributing to narrative-lens

Thanks for considering a contribution. This project extracts Emathy's manuscript-analysis
engine into a standalone, dependency-light crate -- keep that scope in mind: analyzers here
are stateless (text in, a structured result out), with no project model, storage, or
versioning. That belongs in a consuming application, not here.

## Development setup

```bash
git clone https://github.com/writerslogic/narrative-lens.git
cd narrative-lens
cargo build
cargo test
```

For the Node bindings:

```bash
npm install
npm run build:debug
npm test
```

## Before opening a PR

- `cargo test` and `cargo clippy --all-targets -- -D warnings` pass.
- New analyzers include a test against at least one realistic passage, with the expected
  result authored independently of the analyzer's own implementation -- a fixture written by
  reading only the input text, not the code that will judge it.
- Public API changes update `bindings/` (regenerate via `cargo test --features bindings`,
  do not hand-edit) and `docs/ARCHITECTURE.md` if the analyzer categories change.

## Adding an analyzer

Pick the category it belongs to (`craft`, `structure`, `continuity`) based on what it
answers, not what techniques it uses -- a craft analyzer using an embedding model is still
a craft analyzer. Each analyzer is a pure function: no filesystem access, no network call,
no shared mutable state across calls beyond what's passed in via `Options`.

## Code of Conduct

This project follows the [Contributor Covenant](./CODE_OF_CONDUCT.md).
