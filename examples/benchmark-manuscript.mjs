import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { ManuscriptContext } from '../index.mjs';
const seed = JSON.parse(readFileSync(new URL('../tests/fixtures/manuscript-passages.json', import.meta.url), 'utf8'));
let manuscript = process.argv[2] ? JSON.parse(readFileSync(process.argv[2], 'utf8')) : {
  documentId: 'benchmark-novel-v1', knownCharacters: ['Maren', 'Adrian'],
  scenes: Array.from({ length: 160 }, (_, i) => ({ id: `scene-${i.toString().padStart(3, '0')}`, text: seed.scenes[i % 4].repeat(14) })),
};
const context = new ManuscriptContext();
const original = structuredClone(manuscript);
const wordsAtStart = manuscript.scenes.reduce((n, s) => n + (s.text.match(/\S+/g)?.length ?? 0), 0);
const utf8BytesAtStart = manuscript.scenes.reduce((n, s) => n + Buffer.byteLength(s.text), 0);
async function measure() {
  let ticks = 0;
  let last = performance.now();
  let maxGapMs = 0;
  const timer = setInterval(() => { const now = performance.now(); maxGapMs = Math.max(maxGapMs, now - last); last = now; ticks++; }, 2);
  const start = performance.now();
  try {
    const promise = context.analyzeAsync(manuscript);
    const inputConversionMs = performance.now() - start;
    const report = await promise;
    maxGapMs = Math.max(maxGapMs, performance.now() - last);
    return { totalMs: performance.now() - start, inputConversionMs, timerTicks: ticks, maxTimerGapMs: maxGapMs, findings: report.findings.length };
  } finally { clearInterval(timer); }
}
const rssBeforeBytes = process.memoryUsage().rss;
const full = [], incremental = [];
for (let i = 0; i < 5; i++) {
  manuscript = structuredClone(original);
  context.clear();
  full.push(await measure());
  manuscript.scenes[80].text += ' Snow fell.';
  incremental.push(await measure());
}
console.log(JSON.stringify({ fixture: manuscript.documentId, iterations: 5, wordsAtStart, utf8BytesAtStart, profile: process.env.NARRATIVE_LENS_ADDON_PROFILE || 'unspecified native addon profile; no ONNX models', platform: process.platform, arch: process.arch, node: process.version, full, incremental, cache: context.stats(), rssBeforeBytes, rssAfterBytes: process.memoryUsage().rss, processPeakRssKiB: process.resourceUsage().maxRSS, memoryScope: 'OS whole-process RSS; includes Node, native code, inputs, outputs and temporary work', method: 'Identical input reset and context cleared before each full sample; process-wide native initialization and Node GC are not reset' }, null, 2));
