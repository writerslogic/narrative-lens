// Node.js usage example -- currently illustrative, not runnable.
//
// The native addon (built from src/napi_bindings.rs via `napi build --features
// node-api`) has no exports yet, so there is nothing to `require`/`import` and
// run here. See docs/INTEGRATION.md's "As a Node.js package" section, and
// tests/node/api.test.cjs, which skips for the same reason. Once
// napi_bindings.rs exports real functions and `npm run build` produces
// index.js, replace this comment with a working example against them, e.g.:
//
//   import { computeReadability } from '../index.js';
//
//   const passage = 'Maren stood at the edge of the dock...';
//   console.log(computeReadability(passage));
