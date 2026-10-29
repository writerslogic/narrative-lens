import assert from 'node:assert/strict';
import { analyzeManuscriptAsync, ManuscriptContext, compareManuscriptReports } from '../index.mjs';
const manuscript = { documentId: 'novel', scenes: [{ id: 'opening', text: 'She was was tired.' }] };
const before = await analyzeManuscriptAsync(manuscript);
assert(before.findings.some(f => f.ruleId === 'NL-GRAMMAR-REPETITION'));
const context = new ManuscriptContext();
manuscript.scenes[0].text = 'She was tired.';
const after = await context.analyzeAsync(manuscript);
assert.equal(compareManuscriptReports(before, after).resolved.length, 1);
