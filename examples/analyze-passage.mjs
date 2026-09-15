// Node.js usage example.
//
// Requires the native addon to be built first: `npm run build:debug`.
//
//   node examples/analyze-passage.mjs

import { computeReadability, checkGrammar, listStructureTemplates } from '../index.mjs';

const passage =
  'Maren stood at the edge of the dock, watching the last ferry pull away ' +
  'without her. The water slapped against the pilings, cold and indifferent.';

const readability = computeReadability(passage);
console.log(
  `readability: fkgl=${readability.fkgl.toFixed(1)} (${readability.wordCount} words, ${readability.sentenceCount} sentences)`
);

const findings = checkGrammar(passage);
console.log(`grammar: ${findings.length} finding(s)`);

console.log(`available structure templates: ${listStructureTemplates().join(', ')}`);
