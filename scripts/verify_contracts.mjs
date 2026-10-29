import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import {
  analyzeManuscript, compareManuscriptReports,
  defaultManuscriptConfiguration, parseManuscriptReport,
} from '../index.mjs';

const configuration = defaultManuscriptConfiguration();
const manuscript = {
  documentId: 'contract-fixture', configuration, knownCharacters: ['Zoë'],
  scenes: [{ id: 'opening', text: '😀 Zoë saw saw the bird.' }],
};
const before = analyzeManuscript(manuscript);
manuscript.scenes[0].text = '😀 Zoë saw the bird.';
const report = analyzeManuscript(manuscript);
assert.deepEqual(parseManuscriptReport(report), report);
const comparison = compareManuscriptReports(before, report);
assert.equal(comparison.resolved.length, 1);
const result = spawnSync('uv', ['run', '--locked', 'scripts/validate_sarif.py', '--contracts'], {
  cwd: new URL('..', import.meta.url), encoding: 'utf8',
  input: JSON.stringify({ Manuscript: manuscript, Report: report, RevisionComparison: comparison, Configuration: configuration }),
});
assert.equal(result.status, 0, result.stderr || result.stdout || result.error?.message);
process.stdout.write(result.stdout);
