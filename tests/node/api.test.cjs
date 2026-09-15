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

test('narrative-lens native addon', (t) => {
  if (!addon) {
    t.skip('native addon not built -- run `npm run build:debug` first');
    return;
  }

  t.test('computeReadability returns real metrics for real prose', () => {
    const result = addon.computeReadability(
      'Maren stood at the edge of the dock, watching the last ferry pull away without her.'
    );
    assert.strictEqual(typeof result.fkgl, 'number');
    assert.ok(result.wordCount > 0);
    assert.ok(result.sentenceCount > 0);
    assert.strictEqual(typeof result.gradeLevels.fkgl, 'string');
  });

  t.test('computeReadability handles empty text', () => {
    const result = addon.computeReadability('');
    assert.strictEqual(result.wordCount, 0);
  });

  t.test('checkGrammar flags a doubled word', () => {
    const findings = addon.checkGrammar('She was was tired.');
    assert.ok(findings.some((f) => f.kind === 'repetition'));
  });

  t.test('listStructureTemplates returns the built-in templates', () => {
    const names = addon.listStructureTemplates();
    assert.ok(Array.isArray(names));
    assert.ok(names.includes('three_act'));
  });
});
