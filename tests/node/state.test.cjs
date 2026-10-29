const test = require('node:test');
const assert = require('node:assert/strict');
const fixture = require('../fixtures/author-knowledge.json');
const api = require('../../index.js');

const character = name => ({kind:'character',characterId:name});
const reader = {kind:'reader'};
const run = (events=fixture.events,manuscript=fixture.manuscript) =>
  api.analyzeAuthorKnowledge(manuscript,fixture.propositions,events);
const state = (ledger,actor) => ledger.states.find(item =>
  item.actor.kind===actor.kind && item.actor.characterId===actor.characterId && item.propositionId==='hidden-key');

test('authored knowledge distinguishes lies, true beliefs, uncertainty and reader identity', async () => {
  const result=run();
  assert.equal(state(result,reader).classification,'knowledge');
  assert.equal(state(result,character('reader')).classification,'false_belief');
  assert.equal(state(result,character('Bob')).classification,'false_belief');
  assert.equal(state(result,character('Alice')).classification,'correct_belief');
  assert.equal(state(run(fixture.events.slice(0,4)),reader).classification,'uncertain');
  assert.equal(result.unassessedPairs,4);
  const transmitted=result.transitions.find(item=>item.eventId==='lie-to-bob');
  assert.equal(transmitted.action.value,false);
  assert.equal(transmitted.senderAssertion.kind,'knowledge');
  assert.equal(transmitted.senderAssertion.value,true);
  assert.equal(transmitted.senderEventId,'alice-knows');
  for (const transition of result.transitions) {
    const text=fixture.manuscript.scenes.find(scene=>scene.id===transition.evidence.sceneId).text;
    const range=transition.evidence.range;
    assert.equal(text.slice(range.utf16Start,range.utf16End),transition.evidence.quote);
    assert.equal(Buffer.from(text).subarray(range.byteStart,range.byteEnd).toString(),transition.evidence.quote);
  }
  assert.deepEqual(run([...fixture.events].reverse(),{...fixture.manuscript,scenes:[...fixture.manuscript.scenes].reverse()}),result);
  assert.deepEqual(run(),result);
  const esm=await import('../../index.mjs');
  assert.deepEqual(esm.analyzeAuthorKnowledge(fixture.manuscript,fixture.propositions,fixture.events),result);
});

test('authored knowledge rejects ambiguous replay, unknown actors and stale evidence atomically', () => {
  const events=structuredClone(fixture.events);
  events[1].sequence=events[0].sequence;
  assert.throws(()=>run(events),/sequence/);
  events[1].sequence=1.5;
  assert.throws(()=>run(events),/invalid author knowledge events/);
  events[1].sequence=2;
  events[1].action.actor.characterId='Bbo';
  assert.throws(()=>run(events),/character/);
  const ambiguous=structuredClone(fixture.events);
  ambiguous[4].action.actor={kind:'reader',characterId:'Alice'};
  assert.throws(()=>run(ambiguous),/invalid author knowledge events/);
  assert.throws(()=>api.analyzeAuthorKnowledge(fixture.manuscript,[{...fixture.propositions[0],truth:true}],fixture.events),/invalid author propositions/);
  const manuscript=structuredClone(fixture.manuscript);
  manuscript.scenes[0].text='Later, Alice hid the key.';
  assert.throws(()=>run(fixture.events,manuscript),/quote|range/);
  assert.deepEqual(run().documentId,fixture.manuscript.documentId);
});

test('current world-state output replays scene deltas and omits optional histories', async () => {
  const scenes=[
    'Alice was in the cellar. Bob was in the cellar. Alice picked up the key.',
    'Alice gave the key to Bob. Bob told Alice that the gate was open.',
    'Bob entered the garden. Alice locked the door. Alice unlocked the door.',
  ];
  const full=api.trackWorldState(scenes,['Alice','Bob']);
  const compact=api.trackWorldStateCurrent(scenes,['Alice','Bob']);
  assert.equal(compact.afterScene,2);
  assert.deepEqual(compact.facts,full.snapshots.at(-1).facts);
  assert.deepEqual(compact.violations,full.violations);
  assert.deepEqual(compact.deltas,[]);
  assert.deepEqual(compact.informationFlow,[]);
  assert.deepEqual(compact.locationTimeline,[]);
  const withDeltas=api.trackWorldStateCurrent(scenes,['Bob','Alice'],{includeDeltas:true,includeEventHistory:true});
  assert.deepEqual(withDeltas.informationFlow,full.informationFlow);
  assert.deepEqual(withDeltas.characterKnowledge,full.characterKnowledge);
  const facts=new Map();
  for (const [index,delta] of withDeltas.deltas.entries()) {
    assert.equal(delta.afterScene,index);
    for (const fact of delta.invalidatedFacts) facts.delete(JSON.stringify(fact));
    for (const fact of delta.newFacts) facts.set(JSON.stringify(fact),fact);
    assert.deepEqual([...facts.keys()].sort(),full.snapshots[index].facts.map(fact=>JSON.stringify(fact)).sort());
  }
  assert.throws(()=>api.trackWorldStateCurrent(scenes,['Alice'],{includeDeltas:'yes'}),/invalid world state options/);
  assert.throws(()=>api.trackWorldStateCurrent(scenes,['Alice'],{invented:true}),/invalid world state options/);
  assert.equal(api.trackWorldStateCurrent([],['Alice']).afterScene,null);
  const esm=await import('../../index.mjs');
  assert.deepEqual(esm.trackWorldStateCurrent(scenes,['Alice','Bob']),compact);
});

test('world-state adapters reject oversized cast and source before regex compilation', () => {
  for (const method of [api.trackWorldState,api.trackWorldStateCurrent]) {
    for (const length of [127,128]) {
      assert.ok(method(['A waited.'],['x'.repeat(length)]));
    }
    for (const length of [129,200000]) {
      assert.throws(()=>method(['A waited.'],['x'.repeat(length)]),/character names/);
    }
    for (const count of [9999,10000]) {
      assert.ok(method(Array(count).fill(''),['Alice']));
    }
    assert.throws(()=>method(Array(10001).fill(''),['Alice']),/scene|world.state/);
    for (const count of [255,256]) {
      assert.ok(method(['A waited.'],Array(count).fill('Alice')));
    }
    assert.throws(()=>method(['A waited.'],Array(257).fill('Alice')),/character|cast|world.state/);
  }
});
