const test=require('node:test');
const assert=require('node:assert/strict');
const api=require('../../index.js');

test('integer inputs preserve JavaScript values until range validation',()=>{
  const calls=[
    value=>api.analyzePacing(value,0,1,1),
    value=>api.computeSceneConfidence('A sentence.',value),
    value=>api.computeReadabilityConfidence(value,3),
    value=>api.computeFindingConfidence(value,3,0.8),
    value=>api.wordFrequencies('garden garden',value),
    value=>api.wordFrequenciesFromWords(['garden'],value),
    value=>api.trainAndFindNearest([], 'garden',value),
    value=>api.reciprocalRankFusion([['a']],60,value),
    value=>api.explainPacing(20,value,false,0),
    value=>api.explainToneShift('hope','fear',value,'general'),
    value=>api.explainWhiteRoom(value,0,1000),
    value=>api.analyzeScene(['radio'],1200,value),
    value=>api.analyzeForeshadowing([],value),
  ];
  for(const value of [-1,0.5,Number.MIN_VALUE,2**32,Number.MAX_SAFE_INTEGER,NaN,Infinity,-Infinity]){
    for(const call of calls) assert.throws(()=>call(value),/unsigned|integer/i);
  }
  assert.equal(api.computeSceneConfidence('A sentence.',0).sceneId,0);
  assert.equal(api.computeSceneConfidence('A sentence.',2**32-1).sceneId,2**32-1);
  assert.equal(api.computeReadabilityConfidence(2**32-1,3).sampleAdequacy,1);
  for(const value of [0.5,2**31,-(2**31)-1,NaN,Infinity]) assert.throws(()=>api.analyzeScene(['radio'],value,0),/signed.*integer/);
});

test('finite metric populations and array alignment are enforced before computation',()=>{
  assert.throws(()=>api.analyzePacing(0,2,1,1),/subset/);
  const pacing=api.analyzePacing(0,0,10,2);
  for(const bad of [NaN,Infinity,-1,1+Number.EPSILON]){
    assert.throws(()=>api.aggregatePacing([{...pacing,dialogueRatio:bad}]),/finite/);
    assert.throws(()=>api.computeFindingConfidence(1,1,bad),/finite/);
    assert.throws(()=>api.getTensionMetrics([bad]),/finite/);
  }
  assert.throws(()=>api.aggregatePacing([{...pacing,sceneId:0.5}]),/unsigned/);
  assert.throws(()=>api.computeFindingConfidence(4,3,0.8),/subset/);
  assert.throws(()=>api.explainReadability(NaN,5,10,'general'),/finite/);
  assert.throws(()=>api.explainReadability(8,10,5,'general'),/inverted/);
  assert.throws(()=>api.explainTension(0.2,0.2,0,3,1,3,0.5),/range/);
  assert.equal(api.explainTension(0,0,0,0,0,0,0),null);
  assert.throws(()=>api.analyzeSubtext([],[],['Alice'],[],[],[],[],[],'general'),/equal lengths/);
  assert.throws(()=>api.buildPromisePayoffLedger(['Opening.'],[],'general',[],[],'three_act',1),/equal lengths/);
});

test('RRF rejects invalid damping and ignores duplicate votes within a source',()=>{
  for(const value of [-1,NaN,Infinity]) assert.throws(()=>api.reciprocalRankFusion([['a']],value,10),/finite/);
  assert.deepEqual(api.reciprocalRankFusion([['a','a','b'],['b','b','a']],0,10),[['a',1.5],['b',1.5]]);
});

test('sample and heuristic indices expose uncalibrated provenance and unknown populations',async()=>{
  const ascii=api.computeSceneConfidence('aaaa bbbb',0);
  const unicode=api.computeSceneConfidence('éééé 中文中文',0);
  assert.equal(ascii.dataQuality,unicode.dataQuality);
  assert.equal(unicode.confidenceBasis,'sample_adequacy');
  assert.equal(unicode.calibrated,false);
  assert.equal(api.measureFindingConfidence(0,0,0).value,null);
  assert.equal(api.measureFindingConfidence(0,1,0).value,0);
  assert.equal(api.explainReadability(20,5,10,'general').confidenceBasis,'heuristic_rule');
  assert.deepEqual((await import('../../index.mjs')).measureFindingConfidence(2,3,0.8),api.measureFindingConfidence(2,3,0.8));
});

test('bounded text entrypoints reject oversized sources before running analyzers',()=>{
  const oversized=' '.repeat(16*1024*1024+1);
  for(const analyze of [api.computeReadability,api.checkGrammar,api.analyzeSmog,api.analyzeSyntaxTension]){
    assert.throws(()=>analyze(oversized),/limit|MiB/);
  }
});

test('historical catalogue keeps ambiguous senses unassessed until explicitly annotated',async()=>{
  const manuscript={documentId:'history',scenes:[{id:'opening',text:'😀 A silver bell.'}]};
  const catalogue=[{id:'bell',term:'silver bell',earliestYear:1800,basis:'author_constraint',reference:'Synthetic story-bible constraint for this test.',context:'A fictional invention date supplied by the test author.',senseAmbiguous:true}];
  const unknown=api.analyzeHistoricalCatalogue(manuscript,catalogue,[],1700,0);
  assert.equal(unknown.candidates[0].dateConflict,null);
  const usage=[{id:'sense',entryId:'bell',evidence:{sceneId:'opening',utf16Start:5,utf16End:16,quote:'silver bell'}}];
  const confirmed=api.analyzeHistoricalCatalogue(manuscript,catalogue,usage,1700,0);
  assert.equal(confirmed.candidates[0].dateConflict,true);
  assert.equal(confirmed.candidates[0].authorUsageId,'sense');
  assert.deepEqual((await import('../../index.mjs')).analyzeHistoricalCatalogue(manuscript,catalogue,usage,1700,0),confirmed);
  assert.equal(api.analyzeSceneChecked(['cool'],1200,0).items.length,0);
  assert.deepEqual(api.extractNamedEntities([]).entities,[]);
});
