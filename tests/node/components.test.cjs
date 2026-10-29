const test = require('node:test');
const assert = require('node:assert/strict');
const api = require('../../index.js');

function fixture() {
  return {
    manuscript: { documentId: 'mystery', scenes: [{id:'question', text:'😀 Who stole the key?'},{id:'answer', text:'Alice stole the key.'}] },
    events: [
      {kind:'setup', id:'missing-key', description:'Identity of the thief', evidence:{sceneId:'question',utf16Start:3,utf16End:21,quote:'Who stole the key?'}},
      {kind:'resolution',id:'answer-key',setupId:'missing-key',outcome:'fulfilled',evidence:{sceneId:'answer',utf16Start:0,utf16End:20,quote:'Alice stole the key.'}},
    ],
  };
}

test('author events validate actual source evidence and preserve stable identities', async () => {
  const {manuscript,events} = fixture();
  assert.equal(api.analyzePromiseEvents(manuscript,events.slice(0,1)).unknownCount,1);
  const ledger = api.analyzePromiseEvents(manuscript,events);
  assert.equal(ledger.fulfilledCount,1);
  assert.equal(ledger.promises[0].resolution.evidence.sceneId,'answer');
  assert.equal(ledger.promises[0].setup.range.byteStart,5);
  for (const evidence of [ledger.promises[0].setup,ledger.promises[0].resolution.evidence]) {
    const text = manuscript.scenes.find(scene=>scene.id===evidence.sceneId).text;
    assert.equal(text.slice(evidence.range.utf16Start,evidence.range.utf16End),evidence.quote);
  }
  const esm = await import('../../index.mjs');
  assert.deepEqual(esm.analyzePromiseEvents({...manuscript,scenes:[...manuscript.scenes].reverse()},[...events].reverse()),ledger);
  manuscript.scenes[1].text='Later: Alice stole the key.';
  assert.throws(()=>api.analyzePromiseEvents(manuscript,events),/quote does not match/);
});

test('author annotations reject unresolved references, duplicate resolutions and unknown fields', () => {
  const {manuscript,events} = fixture();
  assert.throws(()=>api.analyzePromiseEvents(manuscript,events.slice(1)),/unknown setup/);
  assert.throws(()=>api.analyzePromiseEvents(manuscript,[...events,{...events[1],id:'duplicate-answer'}]),/one current author resolution/);
  assert.throws(()=>api.analyzePromiseEvents(manuscript,[{...events[0],invented:true}]),/invalid author promise events/);
});

test('SMOG exposes applicability and the actual thirty-sentence sample', async () => {
  assert.equal(api.analyzeSmog('Tomorrow.').grade,null);
  assert.equal(api.computeReadability('Tomorrow.').smogAvailable,false);
  const text='Tomorrow. '.repeat(30);
  const result=api.analyzeSmog(text);
  assert.equal(result.grade,8.84);
  assert.equal(result.polysyllableCount,30);
  assert.equal(result.sampledSentences.length,30);
  assert.equal(api.computeReadability(text).smogAvailable,true);
  for (const range of result.sampledSentences) assert.equal(Buffer.from(text).subarray(range.byteStart,range.byteEnd).toString(),'Tomorrow.');
  const esm = await import('../../index.mjs');
  assert.deepEqual(esm.analyzeSmog(text),result);
});

test('semantic neighbors abstain when vocabulary has no learned context', () => {
  assert.deepEqual(api.trainAndFindNearest(['quartz','quartz','velvet','velvet'],'quartz',5),[]);
  const neighbors=api.trainAndFindNearest(['king queen castle king queen castle','velvet','velvet'],'king',10);
  assert.ok(neighbors.length>0);
  assert.ok(neighbors.every(([word])=>word!=='velvet'));
});

test('embedding adapters propagate corpus limits instead of returning missing-signal results', () => {
  for (const size of [9999,10000]) {
    const texts=Array(size).fill('');
    assert.deepEqual(api.trainAndFindNearest(texts,'quartz',5),[]);
    assert.equal(api.trainAndSimilarity(texts,'quartz','velvet'),0);
  }
  const overflow=Array(10001).fill('');
  assert.throws(()=>api.trainAndFindNearest(overflow,'quartz',5),/corpus embedding exceeds option or input limits/);
  assert.throws(()=>api.trainAndSimilarity(overflow,'quartz','velvet'),/corpus embedding exceeds option or input limits/);
});
