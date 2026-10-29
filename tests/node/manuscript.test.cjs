const test = require('node:test');
const assert = require('node:assert/strict');
const { analyzeManuscript, analyzeManuscriptAsync, compareManuscriptReports, ManuscriptContext } = require('../../index.js');
const input = () => ({ documentId: 'novel', knownCharacters: ['Zoë'], scenes: [
  { id: 'cellar', text: '😀 Zoë was in the cellar. Someone locked the door.' },
  { id: 'garden', text: 'Zoë entered the garden. Zoë saw saw the bird.' },
], configuration: { rules: ['NL-CONTINUITY-LOCKED', 'NL-GRAMMAR-REPETITION'] } });

test('manuscript evidence, async results, and cache mutations match fresh results', async () => {
  const m = input();
  const context = new ManuscriptContext(1024 * 1024);
  const first = context.analyze(m);
  assert.deepEqual(await analyzeManuscriptAsync(m), first);
  assert.equal(first.findings.length, 2);
  for (const f of first.findings) for (const e of f.evidence) {
    const scene = m.scenes.find(s => s.id === e.sceneId);
    assert.equal(scene.text.slice(e.range.utf16Start, e.range.utf16End), e.quote);
    assert.equal(Buffer.from(scene.text).subarray(e.range.byteStart, e.range.byteEnd).toString(), e.quote);
  }
  assert.deepEqual(await context.analyzeAsync(m), first);
  assert.equal(context.stats().reusedScenes, 2);
  m.scenes[0].text = 'Zoë was in the cellar.';
  m.scenes.reverse();
  const changed = await context.analyzeAsync(m);
  assert.deepEqual(changed, analyzeManuscript(m));
  const diff = compareManuscriptReports(first, changed);
  assert.equal(diff.resolved.length, 1);
  assert.equal(diff.persistent.length, 1);
  context.clear();
  assert.equal(context.stats().entries, 0);
  assert.throws(() => context.analyze({ ...m, scenes: [m.scenes[0], m.scenes[0]] }), /unique/);
  assert.throws(() => analyzeManuscript({ ...m, configuration: { typo: true } }), /invalid manuscript/);
});

test('expensive asynchronous analysis leaves timers responsive', async () => {
  const passage = 'Maren stood at the edge of the dock, watching the last ferry pull away without her. The water slapped against the pilings, cold and indifferent. She had missed it on purpose, though she was not yet ready to admit that to herself. ';
  const m = { documentId: 'responsive-novel-v1', scenes: Array.from({ length: 160 }, (_, i) => ({ id: `scene-${i}`, text: passage.repeat(12) })), configuration: { rules: ['NL-READABILITY-COMPLEX', 'NL-PACING-LONG'] } };
  let ticks = 0;
  let settled = false;
  const timer = setInterval(() => { if (!settled) ticks++; }, 2);
  try {
    const report = await analyzeManuscriptAsync(m);
    settled = true;
    assert.equal(report.metrics.length, 160);
    assert.ok(ticks >= 3, `expected timer ticks during native computation; saw ${ticks}`);
  } finally { clearInterval(timer); }
});

test('context rejects simultaneous work without losing cached state', async () => {
  const context = new ManuscriptContext();
  const m = input();
  const results = await Promise.allSettled([context.analyzeAsync(m), context.analyzeAsync(m)]);
  assert.ok(results.some(r => r.status === 'fulfilled'));
  for (const r of results) if (r.status === 'rejected') assert.match(r.reason.message, /busy/);
  assert.deepEqual(context.analyze(m), analyzeManuscript(m));
});

test('defaults, catalog, report validation and selective invalidation are usable', () => {
  const api = require('../../index.js');
  const defaults = api.defaultManuscriptConfiguration();
  assert.equal(defaults.limits.maxFindings, 10000);
  assert.ok(api.listManuscriptRules().some(r => r.id === 'NL-CONTINUITY-NLI' && r.requiresModel));
  const m = input();
  const context = new api.ManuscriptContext();
  const before = context.analyze(m);
  assert.deepEqual(api.parseManuscriptReport(JSON.parse(JSON.stringify(before))), before);
  const malformed = structuredClone(before);
  malformed.findings[0].confidence.score = 2;
  assert.throws(() => api.parseManuscriptReport(malformed), /confidence/);
  m.configuration.thresholds = { maxGrade: 18 };
  assert.deepEqual(context.analyze(m), api.analyzeManuscript(m));
  assert.equal(context.stats().recomputedMetricsScenes, 0);
  assert.equal(context.stats().recomputedGrammarScenes, 0);
  assert.equal(context.stats().recomputedWorldScenes, 0);
});

test('empty revision deltas cannot hide unavailable selected work or missing coverage', () => {
  const report = analyzeManuscript({
    documentId: 'coverage',
    scenes: [{ id: 'first', text: 'Alice stayed in the cellar.' }, { id: 'second', text: 'Alice left the cellar.' }],
    configuration: { rules: ['NL-CONTINUITY-NLI'], nli: 'fallback' },
  });
  assert.equal(report.coverage.find(c => c.analyzer === 'nli').status, 'unavailable');
  const comparison = compareManuscriptReports(report, report);
  assert.equal(comparison.introduced.length, 0);
  assert.equal(comparison.resolved.length, 0);
  assert.equal(comparison.unassessed.length, 0);
  assert.equal(comparison.complete, false);

  const clean = analyzeManuscript({ documentId: 'coverage', scenes: [{ id: 'first', text: 'The rain fell.' }], configuration: { rules: ['NL-GRAMMAR-REPETITION'] } });
  assert.equal(compareManuscriptReports(clean, clean).complete, true);
  const missing = structuredClone(clean);
  missing.coverage = missing.coverage.filter(c => c.analyzer !== 'grammar');
  assert.equal(compareManuscriptReports(missing, clean).complete, false);
});
