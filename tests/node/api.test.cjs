// Node.js binding smoke test, run via `npm test` (node --test).
// Requires the native addon to be built first: `npm run build:debug`.

const test = require('node:test');
const assert = require('node:assert');

let addon;
try {
  addon = require('../../index.js');
} catch {
  addon = null;
}

// Top-level tests, not nested t.test() subtests: node:test's subtests are
// async even with a synchronous callback, and a non-async parent that
// doesn't await them returns before they finish -- the runner then cancels
// them as orphaned ("test did not finish before its parent"). That raced
// clean locally but failed reliably in CI. `{ skip: !addon }` gets the same
// per-test skip behavior without the nesting hazard.

test('computeReadability returns real metrics for real prose', { skip: !addon && 'native addon not built -- run `npm run build:debug` first' }, () => {
  const result = addon.computeReadability(
    'Maren stood at the edge of the dock, watching the last ferry pull away without her.'
  );
  assert.strictEqual(typeof result.fkgl, 'number');
  assert.ok(result.wordCount > 0);
  assert.ok(result.sentenceCount > 0);
  assert.strictEqual(typeof result.gradeLevels.fkgl, 'string');
});

test('computeReadability handles empty text', { skip: !addon && 'native addon not built -- run `npm run build:debug` first' }, () => {
  const result = addon.computeReadability('');
  assert.strictEqual(result.wordCount, 0);
});

test('checkGrammar flags a doubled word', { skip: !addon && 'native addon not built -- run `npm run build:debug` first' }, () => {
  const findings = addon.checkGrammar('She was was tired.');
  assert.ok(findings.some((f) => f.kind === 'repetition'));
});

test('listStructureTemplates returns the built-in templates', { skip: !addon && 'native addon not built -- run `npm run build:debug` first' }, () => {
  const names = addon.listStructureTemplates();
  assert.ok(Array.isArray(names));
  assert.ok(names.includes('three_act'));
});
