## Summary

<!-- What does this change, and why? -->

## Checklist

- [ ] `cargo test` and `cargo clippy --all-targets -- -D warnings` pass
- [ ] `npm test` and `npm run test:types` pass (if the Node bindings are affected)
- [ ] New analyzers include a test fixture authored independently of the implementation
- [ ] `bindings/` regenerated if public types changed (never hand-edited)
