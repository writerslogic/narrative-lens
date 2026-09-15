// Node.js binding smoke test, run via `npm test` (node --test).
//
// The native addon (`index.js`/`index.mjs`, built from `src/napi_bindings.rs`
// via `napi build --features node-api`) is not implemented yet -- see
// docs/INTEGRATION.md's "As a Node.js package" section. Until then this test
// skips cleanly instead of failing CI on a binding that was never meant to
// exist yet; once `napi_bindings.rs` exports real functions, replace the
// `t.skip(...)` below with assertions against them.

const test = require('node:test');
const assert = require('node:assert');

let addon;
try {
  addon = require('../../index.js');
} catch {
  addon = null;
}

test('narrative-lens native addon', (t) => {
  if (!addon || Object.keys(addon).length === 0) {
    t.skip('native addon not built / napi_bindings.rs has no exports yet');
    return;
  }
  // Once real bindings exist, assert their shape here, e.g.:
  // assert.strictEqual(typeof addon.computeReadability, 'function');
  assert.ok(addon);
});
