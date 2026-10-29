// Node.js binding smoke test, run via `npm test` (node --test).
// Requires the native addon to be built first: `npm run build:debug`.

const test = require('node:test');
const assert = require('node:assert');

const addon = require('../../index.js');

// Top-level tests, not nested t.test() subtests: node:test's subtests are
// async even with a synchronous callback, and a non-async parent that
// doesn't await them returns before they finish -- the runner then cancels
// them as orphaned ("test did not finish before its parent"). That raced
// clean locally but failed reliably in CI.

test('computeReadability returns real metrics for real prose', () => {
  const result = addon.computeReadability(
    'Maren stood at the edge of the dock, watching the last ferry pull away without her.'
  );
  assert.strictEqual(typeof result.fkgl, 'number');
  assert.ok(result.wordCount > 0);
  assert.ok(result.sentenceCount > 0);
  assert.strictEqual(typeof result.gradeLevels.fkgl, 'string');
});

test('computeReadability handles empty text', () => {
  const result = addon.computeReadability('');
  assert.strictEqual(result.wordCount, 0);
});

test('checkGrammar flags a doubled word', () => {
  const findings = addon.checkGrammar('She was was tired.');
  assert.ok(findings.some((f) => f.kind === 'repetition'));
});

test('listStructureTemplates returns the built-in templates', () => {
  const names = addon.listStructureTemplates();
  assert.ok(Array.isArray(names));
  assert.ok(names.includes('three_act'));
});

test('standalone wire tags preserve facts and abstain from unobserved payoffs', () => {
  const result = addon.trackWorldState(['Alice was in the cellar.'], ['Alice']);
  assert.ok(result.snapshots[0].facts.some(fact => fact.characterAt?.location === 'cellar'));
  const ledger = addon.buildPromisePayoffLedger(['Who stole the key?', 'Alice stole the key.'], [5, 5], 'general', [['key', 0, true]], [], 'three_act', 2);
  assert.ok(ledger.promises.some(promise => promise.status === 'unknown'));
  const confirmed = ledger.promises.filter(promise => promise.status === 'confirmedFulfilled');
  assert.ok(confirmed.length > 0);
  for (const promise of confirmed) assert.equal(promise.payoffScene, null);
});
