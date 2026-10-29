const test=require('node:test');
const assert=require('node:assert/strict');
const api=require('../../index.js');
const knowledge=require('../fixtures/author-knowledge.json');

test('checked dialogue preserves source and separates speakers from addressees',async()=>{
  const text='İ'.repeat(35)+' Alice said to Bob, "Hello."';
  const result=api.extractDialogueChecked(text,7,['Bob','Alice']);
  assert.equal(result.lines.length,1);
  const located=result.lines[0];
  assert.equal(located.line.speaker,'Alice');
  assert.equal(located.kind,'direct');
  assert.equal(text.slice(located.range.utf16Start,located.range.utf16End),located.line.quote);
  assert.equal(Buffer.from(text).subarray(located.range.byteStart,located.range.byteEnd).toString(),located.line.quote);
  assert.equal(text.slice(located.quotationRange.utf16Start,located.quotationRange.utf16End),'"Hello."');
  assert.ok(located.attributionRange);
  assert.deepEqual((await import('../../index.mjs')).extractDialogueChecked(text,7,['Bob','Alice']),result);
  assert.equal(api.extractDialogue(text,7,['Bob','Alice'])[0].speaker,'Alice');
});

test('dialogue bounds and numeric indices fail explicitly',()=>{
  for(const value of [-1,0.5,4294967296,NaN,Infinity]){
    assert.throws(()=>api.extractDialogueChecked('"Hello."',value,[]),/unsigned/);
    assert.throws(()=>api.extractDialogue('"Hello."',value,[]),/unsigned/);
  }
  assert.throws(()=>api.extractDialogueChecked('"Hello."',0,[],{includeIndirect:'yes'}),/invalid dialogue/);
  assert.throws(()=>api.extractDialogueChecked('"Hello."',0,[],{extra:true}),/invalid dialogue/);
  assert.throws(()=>api.extractDialogue('"Hello."',0,['x'.repeat(129)]),/character/);
});

test('causal interventions preserve declared OR alternatives and reject cycles',async()=>{
  const manuscript=knowledge.manuscript;
  const evidence=knowledge.propositions[0].evidence;
  const nodes=[
    {id:'first',mechanism:{kind:'fixed',value:true},evidence},
    {id:'second',mechanism:{kind:'fixed',value:true},evidence},
    {id:'alarm',mechanism:{kind:'any',parents:['first','second']},evidence},
  ];
  const result=api.analyzeAuthorCausality(manuscript,nodes,[{nodeId:'first',value:false}]);
  assert.equal(result.nodes.find(node=>node.id==='alarm').intervened,true);
  assert.equal(result.nodes.find(node=>node.id==='first').intervened,false);
  const both=api.analyzeAuthorCausality(manuscript,nodes,[{nodeId:'first',value:false},{nodeId:'second',value:false}]);
  assert.equal(both.nodes.find(node=>node.id==='alarm').intervened,false);
  assert.throws(()=>api.analyzeAuthorCausality(manuscript,[{id:'cycle',mechanism:{kind:'not',parent:'cycle'},evidence}],[]),/cyc|self/);
  assert.deepEqual((await import('../../index.mjs')).analyzeAuthorCausality(manuscript,nodes,[{nodeId:'first',value:false}]),result);
});

test('reader projection distinguishes belief revision from world-truth updates',()=>{
  const events=[...knowledge.events,{id:'truth-correction',sequence:7,propositionId:'hidden-key',action:{kind:'truth',value:'false'},evidence:knowledge.propositions[0].evidence}];
  const model=api.buildAuthorMentalModel(knowledge.manuscript,knowledge.propositions,events);
  const change=model.readerChanges.find(change=>change.eventId==='truth-correction');
  assert.equal(change.kind,'truth_reclassification');
  assert.deepEqual(change.before,change.after);
  assert.equal(change.beforeClassification,'knowledge');
  assert.equal(change.afterClassification,'contradicted_knowledge');
  assert.equal(model.readerSummary.contradictedKnowledge,1);
  assert.equal(model.readerSummary.unassessedPropositions,1);
});

const sceneText='😀 Alice waited. She said, "I am fine."';
const authoredManuscript={documentId:'annotated',knownCharacters:['Alice'],scenes:[{id:'opening',text:sceneText}]};
function source(quote){
  const utf16Start=sceneText.indexOf(quote);
  assert.ok(utf16Start>=0);
  return {sceneId:'opening',utf16Start,utf16End:utf16Start+quote.length,quote};
}

test('authored identity links preserve exact endpoints and reject conflicting assignments',async()=>{
  const links=[{id:'she',characterId:'Alice',mention:source('She'),antecedent:source('Alice')}];
  const result=api.resolveAuthorCoreference(authoredManuscript,links);
  assert.equal(result.links[0].characterId,'Alice');
  const range=result.links[0].mention.range;
  assert.equal(sceneText.slice(range.utf16Start,range.utf16End),'She');
  assert.equal(Buffer.from(sceneText).subarray(range.byteStart,range.byteEnd).toString(),'She');
  assert.throws(()=>api.resolveAuthorCoreference(authoredManuscript,[{...links[0],characterId:'Bob'}]),/identit/);
  assert.throws(()=>api.resolveAuthorCoreference(authoredManuscript,[{...links[0],mention:{...source('She'),quote:'He'}}]),/source|quote/);
  assert.deepEqual((await import('../../index.mjs')).resolveAuthorCoreference(authoredManuscript,links),result);
});

test('annotation projections preserve author interpretations without inferring them',async()=>{
  const annotations=[{id:'denial',surface:source('I am fine.'),implied:'Alice is concealing distress.',subtextType:'EmotionalDenial',characters:['Alice']}];
  const subtext=api.analyzeSubtextAnnotations(authoredManuscript,annotations);
  assert.equal(subtext.instances[0].implied,annotations[0].implied);
  assert.deepEqual(subtext.unresolvedIds,['denial']);
  assert.equal(subtext.sceneCoverage[0].coveredUtf16,'I am fine.'.length);
  assert.throws(()=>api.analyzeSubtextAnnotations(authoredManuscript,[{...annotations[0],surface:{...source('I am fine.'),utf16Start:1}}]),/source|bound|surrogate|quote/);
  const theses=[{id:'honesty',statement:'Honesty preserves trust.',opposes:['concealment']},{id:'concealment',statement:'Concealment protects people.'}];
  const evidence=[{id:'claim',thesisId:'concealment',source:source('I am fine.'),valence:'Supporting',delivery:{Dialogue:{speaker:'Alice'}}}];
  const thematic=api.analyzeThematicAnnotations(authoredManuscript,theses,evidence);
  assert.equal(thematic.evidence[0].thesisId,'concealment');
  assert.equal(thematic.evidence[0].valence,'Supporting');
  assert.deepEqual(thematic.unassessedThesisIds,['honesty']);
  assert.throws(()=>api.analyzeThematicAnnotations(authoredManuscript,theses,[{...evidence[0],thesisId:'missing'}]),/thesis|reference/);
  assert.deepEqual((await import('../../index.mjs')).analyzeSubtextAnnotations(authoredManuscript,annotations),subtext);
  assert.deepEqual((await import('../../index.mjs')).analyzeThematicAnnotations(authoredManuscript,theses,evidence),thematic);
});

test('defined stylometry excludes empty turns and preserves turn-relative cue spans',()=>{
  const measured=api.buildVoiceProfileChecked('Alice',['Are you here?','...']);
  assert.equal(measured.sample.inputTurnCount,2);
  assert.equal(measured.sample.turnCount,1);
  assert.equal(measured.profile.questionRate,1);
  assert.equal(api.buildVoiceProfileChecked('Alice',[]).availability,'unavailable');
  const text='😀 As you know, we should leave.';
  const cues=api.analyzeDialogueRealism([9],['Alice'],[text],['Alice']);
  assert.equal(cues.realismAssessed,false);
  assert.equal(cues.cues[0].scene,9);
  for(const cue of cues.cues){
    assert.equal(text.slice(cue.range.utf16Start,cue.range.utf16End),cue.quote);
    assert.equal(Buffer.from(text).subarray(cue.range.byteStart,cue.range.byteEnd).toString(),cue.quote);
  }
  assert.throws(()=>api.analyzeDialogueRealism([0],[],['Hello'],['Alice']),/equal lengths/);
  assert.throws(()=>api.analyzeDialogueRealism([0.5],['Alice'],['Hello'],['Alice']),/unsigned/);
  assert.throws(()=>api.analyzeCharacterVoices(['Alice'],[]),/equal lengths/);
  assert.throws(()=>api.compareVoices([{...measured.profile,questionRate:NaN}]),/finite/);
  assert.throws(()=>api.measureVoiceDrift(measured.profile,{...measured.profile,character:'Bob'}),/same character/);
});

test('checked entity,coreference,gap and timeline adapters reject invalid data',()=>{
  assert.throws(()=>api.resolveEntities([['Alice',[1]],['Bob',[1,2]]],'character'),/dimensions/);
  assert.throws(()=>api.resolveEntities([['Alice',[Infinity]]],'character'),/finite/);
  const distinct=api.resolveEntities([['C++',[]],['C#',[]]],'symbol');
  assert.equal(new Set(distinct.map(entity=>entity.id)).size,2);
  assert.throws(()=>api.resolveCoreferences('Alice waited.',['Alice','ALICE']),/unique|duplicate/);
  assert.equal(api.resolveCoreferences('He left. John arrived.',['John'],{John:'male'}).mentions.find(m=>m.isPronoun).resolvedTo,null);
  for(const value of [-1,0.5,4294967296]) assert.throws(()=>api.analyzeGaps([],[],[],[],value,0),/unsigned/);
  assert.throws(()=>api.analyzeGaps([],[],[[1,0]],[],2,1),/range/);
  const documents=[['first','At 25:90, Alice waited.']];
  const patches=[{document_id:'first',label:'old'},{document_id:'first',label:'new',lane:'dream'}];
  const timeline=api.buildTimeline(documents,patches);
  assert.equal(timeline[0].label,'new');
  assert.equal(timeline[0].lane,'dream');
  assert.equal(timeline[0].timeHint,null);
  assert.throws(()=>api.buildTimeline(documents,[{document_id:'missing',label:'bad'}]),/unknown/);
});

test('genre evidence preserves unknowns and does not infer prose quality or intent',async()=>{
  const missing=api.checkGenreComplianceChecked('romance',{});
  const absent=api.checkGenreComplianceChecked('romance',{hasHea:false});
  assert.equal(missing.availability,'unavailable');
  assert.equal(missing.unknownCount,missing.conventions.length);
  assert.equal(absent.conventions.find(c=>c.name==='HEA/HFN').status,'NotMet');
  assert.equal(absent.qualityAssessed,false);
  const highStructure=api.checkGenreComplianceChecked('literary',{structureHealth:0.99});
  assert.equal(highStructure.conventions.find(c=>c.name==='Prose Quality').status,'Unknown');
  assert.throws(()=>api.checkGenreComplianceChecked('romance',{hasHea:true,conventionJudgments:{'HEA/HFN':'NotMet'}}),/conflicting/);
  assert.throws(()=>api.checkGenreComplianceChecked('romance',{wordCount:-1}),/invalid/);
  assert.throws(()=>api.checkGenreComplianceChecked('romance',{structureHealth:Infinity}),/invalid|finite|number/i);
  assert.deepEqual((await import('../../index.mjs')).checkGenreComplianceChecked('romance',{hasHea:false}),absent);
});

test('supplied Shannon uncertainty and measured vector novelty retain separate units',async()=>{
  const observations=[{questionId:'q',sceneId:'opening',outcomes:[{id:'a',probability:0.5},{id:'b',probability:0.5}]}];
  const uncertainty=api.measureNarrativeUncertainty(authoredManuscript,observations);
  assert.equal(uncertainty.measurements[0].entropyBits,1);
  assert.equal(uncertainty.measurements[0].effectiveOutcomes,2);
  assert.deepEqual(uncertainty.measurements[0].outcomes,observations[0].outcomes);
  assert.throws(()=>api.measureNarrativeUncertainty(authoredManuscript,[{...observations[0],outcomes:[{id:'a',probability:0.1}]}]),/sum/);
  const novelty=api.measureSemanticNovelty([[1,0],[0,1],[-1,0]],{window:1,includeNearest:true});
  assert.equal(novelty.points[0].novelty,null);
  assert.equal(novelty.points[1].novelty,1);
  assert.equal(novelty.points[2].nearestPriorScene,1);
  assert.throws(()=>api.measureSemanticNovelty([[NaN,0]]),/finite/);
  assert.throws(()=>api.measureSemanticNovelty([[1,0]],{window:0}),/window/);
  assert.deepEqual((await import('../../index.mjs')).measureSemanticNovelty([[1,0],[0,1]],{window:1}),api.measureSemanticNovelty([[1,0],[0,1]],{window:1}));
  assert.deepEqual((await import('../../index.mjs')).measureNarrativeUncertainty(authoredManuscript,observations),uncertainty);
});

test('epistemic checkpoint queries distinguish unassessed,uncertain,and conflicting actors',async()=>{
  const query={id:'q',propositionId:'hidden-key',observer:{kind:'reader'},subject:{kind:'character',characterId:'Bob'},atSequence:5};
  const result=api.analyzeAuthorEpistemics(knowledge.manuscript,knowledge.propositions,knowledge.events,[query,{...query,id:'earlier',atSequence:2}]);
  const current=result.answers.find(answer=>answer.queryId==='q');
  assert.equal(current.relation,'observer_advantage');
  assert.equal(current.assertionsConflict,true);
  assert.equal(current.observer.eventId,'reader-learns');
  const earlier=result.answers.find(answer=>answer.queryId==='earlier');
  assert.equal(earlier.relation,'unassessed');
  assert.equal(earlier.observer.classification,'unassessed');
  assert.throws(()=>api.analyzeAuthorEpistemics(knowledge.manuscript,knowledge.propositions,knowledge.events,[{...query,atSequence:-1}]),/invalid/);
  const literal=api.trackEpistemics(['Alice told Bob that the treasure was gone.'],['Alice','Bob'],['Alice']);
  assert.equal(literal.semanticMetricsAvailable,false);
  assert.deepEqual(literal.characterStates.Alice.knows,[]);
  assert.deepEqual((await import('../../index.mjs')).analyzeAuthorEpistemics(knowledge.manuscript,knowledge.propositions,knowledge.events,[query]),api.analyzeAuthorEpistemics(knowledge.manuscript,knowledge.propositions,knowledge.events,[query]));
});

test('typed opening voice comparison uses feature values and original source ranges',async()=>{
  const text='😀 I waited. Would she return?';
  const baseline=api.buildVoiceProfile('Narrator',[text]);
  const opening=api.analyzeOpeningChecked(text,'general',baseline);
  assert.equal(opening.qualityAssessed,false);
  assert.ok(opening.voiceComparison.overall<1e-12);
  for(const cue of opening.cues) assert.equal(text.slice(cue.range.utf16Start,cue.range.utf16End),cue.quote);
  assert.throws(()=>api.analyzeOpeningChecked(text,'general',{...baseline,questionRate:2}),/0\.\.=1|within|rate/);
  assert.equal(api.analyzeOpening(text,'general',[1,2,3]).voiceComparison,null);
  assert.deepEqual((await import('../../index.mjs')).analyzeOpeningChecked(text,'general',baseline),opening);
});
