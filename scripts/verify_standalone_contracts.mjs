import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import * as api from '../index.mjs';

const scenes = ['Alice was in the cellar. Alice discovered that the key was gone.', 'Alice entered the garden.'];
const profile = api.buildVoiceProfile('Alice', ['We should leave now.']);
const authorManuscript = {documentId:'author-contract',scenes:[{id:'question',text:'Who stole the key?'},{id:'answer',text:'Alice stole the key.'}]};
const authorEvents = [
  {kind:'setup',id:'key',description:'Identity of the thief',evidence:{sceneId:'question',utf16Start:0,utf16End:18,quote:'Who stole the key?'}},
  {kind:'resolution',id:'key-answer',setupId:'key',outcome:'fulfilled',evidence:{sceneId:'answer',utf16Start:0,utf16End:20,quote:'Alice stole the key.'}},
];
const knowledge=JSON.parse(readFileSync(new URL('../tests/fixtures/author-knowledge.json',import.meta.url),'utf8'));
const annotationText='😀 Alice waited. She said, "I am fine."';
const annotationManuscript={documentId:'annotations-contract',knownCharacters:['Alice'],scenes:[{id:'opening',text:annotationText}]};
const anchor=quote=>({sceneId:'opening',utf16Start:annotationText.indexOf(quote),utf16End:annotationText.indexOf(quote)+quote.length,quote});
const identityLinks=[{id:'she',characterId:'Alice',mention:anchor('She'),antecedent:anchor('Alice')}];
const subtextAnnotations=[{id:'denial',surface:anchor('I am fine.'),implied:'The author supplies a concealed meaning.',subtextType:'EmotionalDenial',characters:['Alice']}];
const thematicTheses=[{id:'honesty',statement:'Honesty preserves trust.',opposes:[]}];
const thematicAnnotations=[{id:'denial',thesisId:'honesty',source:anchor('I am fine.'),valence:'Contradicting',delivery:{Dialogue:{speaker:'Alice'}}}];
const causalNodes=[{id:'decision',mechanism:{kind:'fixed',value:true},evidence:knowledge.propositions[0].evidence}];
const interventions=[{nodeId:'decision',value:false}];
const epistemicQueries=[{id:'q',propositionId:'hidden-key',observer:{kind:'reader'},subject:{kind:'character',characterId:'Bob'},atSequence:5}];
const uncertainty=[{questionId:'q',sceneId:'opening',outcomes:[{id:'a',probability:0.5},{id:'b',probability:0.5}]}];
const samples = {
  AuthorPromiseEvents: authorEvents,
  AnalyzePromiseEventsResult: api.analyzePromiseEvents(authorManuscript,authorEvents),
  AnalyzeSmogResult: api.analyzeSmog('Tomorrow. '.repeat(30)),
  AuthorPropositions: knowledge.propositions,
  AuthorKnowledgeEvents: knowledge.events,
  AnalyzeAuthorKnowledgeResult: api.analyzeAuthorKnowledge(knowledge.manuscript,knowledge.propositions,knowledge.events),
  AuthorCausalNodes:causalNodes,
  AuthorInterventions:interventions,
  AnalyzeAuthorCausalityResult:api.analyzeAuthorCausality(knowledge.manuscript,causalNodes,interventions),
  BuildAuthorMentalModelResult:api.buildAuthorMentalModel(knowledge.manuscript,knowledge.propositions,knowledge.events),
  AuthorEpistemicQueries:epistemicQueries,
  AnalyzeAuthorEpistemicsResult:api.analyzeAuthorEpistemics(knowledge.manuscript,knowledge.propositions,knowledge.events,epistemicQueries),
  AuthorCoreferenceLinks:identityLinks,
  ResolveAuthorCoreferenceResult:api.resolveAuthorCoreference(annotationManuscript,identityLinks),
  AuthorSubtextAnnotations:subtextAnnotations,
  AnalyzeSubtextAnnotationsResult:api.analyzeSubtextAnnotations(annotationManuscript,subtextAnnotations),
  AuthorThematicTheses:thematicTheses,
  AuthorThematicAnnotations:thematicAnnotations,
  AnalyzeThematicAnnotationsResult:api.analyzeThematicAnnotations(annotationManuscript,thematicTheses,thematicAnnotations),
  DialogueExtractionOptionsContract:{attribution:'explicit_only',includeIndirect:false},
  ExtractDialogueCheckedResult:api.extractDialogueChecked(annotationText,0,['Alice']),
  BuildVoiceProfileCheckedResult:api.buildVoiceProfileChecked('Alice',['We wait.']),
  GenreEvidenceContract:{hasHea:false},
  SuppliedUncertaintyObservations:uncertainty,
  SemanticNoveltyOptionsContract:{window:1,includeNearest:true},
  MeasureNarrativeUncertaintyResult:api.measureNarrativeUncertainty(annotationManuscript,uncertainty),
  MeasureSemanticNoveltyResult:api.measureSemanticNovelty([[1,0],[0,1]],{window:1,includeNearest:true}),
  TimelineOverrides:[{document_id:'first',label:'Opening'}],
  WorldStateOptionsContract: {includeDeltas:true,includeEventHistory:true},
  TrackWorldStateCurrentResult: api.trackWorldStateCurrent(scenes,['Alice'],{includeDeltas:true,includeEventHistory:true}),
  AnalyzeSceneResult: api.analyzeScene(['radio'], 1200, 0),
  AnalyzeSceneCheckedResult:api.analyzeSceneChecked(['cool','radio'],1200,0),
  HistoricalEntries:[{id:'constraint',term:'Alice',earliestYear:1800,basis:'author_constraint',reference:'Synthetic author test constraint.',context:'Fictional date for testing arithmetic.',senseAmbiguous:true}],
  HistoricalUsages:[],
  AnalyzeHistoricalCatalogueResult:api.analyzeHistoricalCatalogue(annotationManuscript,[{id:'constraint',term:'Alice',earliestYear:1800,basis:'author_constraint',reference:'Synthetic author test constraint.',context:'Fictional date for testing arithmetic.',senseAmbiguous:true}],[],1700,0),
  ExtractNamedEntitiesResult:api.extractNamedEntities([]),
  ExtractPersonsResult:api.extractPersons(['Alice waited. Alice said, "Hello."']),
  MeasureFindingConfidenceResult:api.measureFindingConfidence(2,3,0.8),
  AnalyzeCausalityResult: api.analyzeCausality(scenes, ['Alice']),
  ResolveEntitiesResult: api.resolveEntities([['Alice', []]], 'character'),
  ResolveEntitiesWithCorrectionsResult: api.resolveEntitiesWithCorrections([['Alice', []]], 'character', []),
  BuildMentalModelResult: api.buildMentalModel(scenes, ['Alice'], ['Alice'], []),
  AnalyzeGapsResult: api.analyzeGaps([['Unpaid setup', 0]], [], [], [], 2, 1),
  TrackEpistemicsResult: api.trackEpistemics(scenes, ['Alice'], ['Alice','Alice']),
  BuildTimelineResult: api.buildTimeline([['first', 'Alice waited on Monday.']]),
  TrackWorldStateResult: api.trackWorldState(scenes, ['Alice']),
  ResolveCoreferencesResult: api.resolveCoreferences('İ Alice smiled. She waited.', ['Alice']),
  AnalyzePacingResult: api.analyzePacing(0, 0, 10, 2),
  AggregatePacingResult: api.aggregatePacing([api.analyzePacing(0, 0, 10, 2)]),
  ExtractDialogueResult: api.extractDialogue('"We should leave," Alice said.', 0, ['Alice']),
  AnalyzeDialoguePatternsResult: api.analyzeDialoguePatterns(api.extractDialogue('"We should leave," Alice said.', 0, ['Alice']), ['Alice']),
  AnalyzeDialogueRealismResult: api.analyzeDialogueRealism([0], ['Alice'], ['We should leave.'], ['Alice']),
  AnalyzeSyntaxTensionResult: api.analyzeSyntaxTension('Would she leave? She ran!'),
  GetTensionMetricsResult: api.getTensionMetrics([0.1, 0.8]),
  CollectProseExamplesResult: api.collectProseExamples(['Alice was very very tired.']),
  BuildVoiceProfileResult: profile,
  CompareVoicesResult: api.compareVoices([profile]),
  MeasureVoiceDriftResult: api.measureVoiceDrift(profile, profile),
  MeasureVoiceDriftFromTextResult: api.measureVoiceDriftFromText('Alice', ['Wait here.'], ['Wait here.']),
  AnalyzeCharacterVoicesResult: api.analyzeCharacterVoices(['Alice'], [['We should leave.']]),
  AnalyzeSubtextResult: api.analyzeSubtext(scenes, [], [], [], [], [], [], [], 'general'),
  GetStructureTemplateResult: api.getStructureTemplate('three_act'),
  ValidateStructureResult: api.validateStructure('nonlinear', [0.1, 0.5, 0.9, 0.4], []),
  AnalyzeForeshadowingResult: api.analyzeForeshadowing([['silver','key'], ['garden'], ['silver','key']], 3),
  BuildPromisePayoffLedgerResult: api.buildPromisePayoffLedger(['Who stole the silver key?', 'Alice stole the silver key.', 'Alice returned it.'], [5,5,3], 'general', [['silver key', 0, true]], [], 'three_act', 3),
  TrackThematicArgumentResult: api.trackThematicArgument(scenes, [], ['Alice'], [], 'general'),
  AnalyzeOpeningResult: api.analyzeOpening(scenes[0], 'general', []),
  CheckGenreComplianceResult: api.checkGenreCompliance('romance', false, false, false, true, false, false, 1000, 0.9, 'rising'),
  ComputeNarrativeEntropyResult: api.computeNarrativeEntropy(scenes, [], 'three_act', 'general', []),
  GenreFromLabelResult: api.genreFromLabel('general'),
  DetectGenreResult: api.detectGenre([], [0.1, 0.5], 0.2, 8),
  DetectGenreSmartResult: api.detectGenreSmart('Alice watched the garden.', [], [0.1, 0.5], 0.2, 8),
  EvaluateAgainstGenreResult: api.evaluateAgainstGenre('general', 8, 10, 0.2, 0.1, 0, 1000),
  GetGenreProfileResult: api.getGenreProfile('general'),
  DetectThemesSmartResult: api.detectThemesSmart(scenes, [['Alice','garden'], ['Alice','key']]),
  AnalyzeResolutionResult: api.analyzeResolution([{ hope: 0.8 }, { hope: 0.2 }]),
  ExplainReadabilityResult: api.explainReadability(20, 5, 10, 'general'),
  ExplainPacingResult: api.explainPacing(40, 0, false, 0),
  ExplainTensionResult: api.explainTension(0, 0.5, 1, 0, 0, 3, 1),
  ExplainToneShiftResult: api.explainToneShift('hopeful', 'grim', 1, 'general'),
  ExplainWhiteRoomResult: api.explainWhiteRoom(0, 0, 1000),
  ExplainShowDontTellResult: api.explainShowDontTell('filter_word', 'She noticed the garden.'),
  ComputeSceneConfidenceResult: api.computeSceneConfidence(scenes[0], 0),
  ComputeReadabilityConfidenceResult: api.computeReadabilityConfidence(100, 5),
  WordFrequenciesResult: api.wordFrequencies('garden key garden', 5),
  WordFrequenciesFromWordsResult: api.wordFrequenciesFromWords(['garden','key','garden'], 5),
};
assert.ok(samples.TrackWorldStateResult.snapshots.some(s => s.facts.some(f => 'characterAt' in f)));
assert.equal(samples.ValidateStructureResult.healthPct, 100);
const schemas=JSON.parse(readFileSync(new URL('../docs/standalone.schema.json',import.meta.url),'utf8'));
assert.deepEqual(Object.keys(schemas).filter(name=>!(name in samples)),[],'Each generated public contract needs an actual populated runtime/input sample');
const result = spawnSync('uv', ['run', '--locked', 'scripts/validate_sarif.py', '--standalone-contracts'], {
  cwd: new URL('..', import.meta.url), encoding: 'utf8', input: JSON.stringify(samples),
});
assert.equal(result.status, 0, result.stderr || result.stdout || result.error?.message);
process.stdout.write(result.stdout);
