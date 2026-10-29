const api = require('../../index.js');
const fs = require('node:fs');
const results = [];
function probe(id, fn) {
  try { results.push({id, result: fn()}); }
  catch (error) { results.push({id, error: error.message}); }
}
probe('entity-alias-bridge', () => api.resolveEntities([
  ['John Smith', []], ['Smith', []], ['Mary Smith', []],
], 'character'));
probe('entity-last-correction', () => api.resolveEntitiesWithCorrections([], 'character', [
  {surface: 'Ghost', action: {kind: 'assign', canonical: 'Alice'}},
  {surface: 'Ghost', action: {kind: 'reject'}},
]));
probe('voice-determinism', () => {
  const values = Array.from({length: 24}, () => api.buildVoiceProfile('A', ['cedar birch apple maple willow ash']).topWords);
  return {uniqueResults: new Set(values.map(JSON.stringify)).size, samples: values.slice(0, 4)};
});
probe('nonlinear-percentage', () => api.validateStructure('nonlinear', [0.1, 0.5, 0.9, 0.4], []));
probe('coreference-original-byte-offsets', () => {
  const text = 'İ Alice smiled. She waited.';
  return {expectedAliceByteStart: Buffer.from(text).indexOf('Alice'), actual: api.resolveCoreferences(text, ['Alice'])};
});
probe('world-explicit-acquisition', () => api.trackWorldState(['Rain fell.', 'Alice discovered that the key was gone.'], ['Alice']));
probe('communication-character-order', () => {
  const scenes = ['Bob told Alice that the gate was open.'];
  return [api.trackWorldState(scenes, ['Alice','Bob']).informationFlow, api.trackWorldState(scenes, ['Bob','Alice']).informationFlow];
});
probe('world-intrascene-location', () => api.trackWorldState(['Alice was in the cellar. Alice entered the garden.'], ['Alice']));
probe('causal-coherence-range', () => {
  const result = api.analyzeCausality(['Alice opened the gate. Alice crossed the bridge. Alice reached the tower. Alice lit the lamp.', 'Alice decided to leave.'], ['Alice']);
  return {plotCoherence: result.plotCoherence, events: result.events, links: result.links};
});
probe('standalone-typescript-runtime-drift', () => ({
  worldFactActualKeys: api.trackWorldState(['Alice was in the cellar.'], ['Alice']).snapshots[0].facts.map(Object.keys),
  worldFactDeclaredTag: 'CharacterAt',
  payoffActualStatuses: [...new Set(api.buildPromisePayoffLedger(['Who stole the silver key?', 'Alice stole the silver key.', 'Alice returned it.'], [5,5,3], 'general', [], [], 'three_act', 3).promises.map(p => JSON.stringify(p.status)))],
  payoffDeclaredLiterals: ['Outstanding', 'Broken'],
}));
probe('paragraph-line-endings', () => {
  const p = 'She waited in the garden.\n\nThen she went home.';
  return {lf: api.computeReadability(p), crlf: api.computeReadability(p.replaceAll('\n','\r\n'))};
});
probe('readability-trailing-sentence', () => api.computeReadability('She waited. Then she left'));
probe('answered-promise', () => api.buildPromisePayoffLedger(['Who stole the silver key?', 'Alice stole the silver key.', 'Alice returned it.'], [5,5,3], 'general', [], [], 'three_act', 3));
probe('model-required-coverage', () => api.analyzeManuscript({documentId:'d', scenes:[{id:'a',text:'Alice was here.'},{id:'b',text:'Alice was away.'}], configuration:{nli:'required',rules:['NL-CONTINUITY-NLI']}}));
fs.writeFileSync(__dirname + '/reproductions.json', JSON.stringify(results, null, 2) + '\n');
for (const r of results) console.log(r.id + ': ' + JSON.stringify(r.error || r.result).slice(0, 900));
