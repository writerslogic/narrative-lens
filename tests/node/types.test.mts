// Type-only check for the generated TypeScript definitions (index.d.ts), run
// via `npm run test:types` (tsc --noEmit). This file has no runtime
// assertions -- a type error here is the test failure.

import {
  computeReadability,
  checkGrammar,
  listStructureTemplates,
  type ReadabilityResultJs,
  type GrammarFindingJs,
} from '../../index.js';

const readability: ReadabilityResultJs = computeReadability('Example passage.');
const fkgl: number = readability.fkgl;
const gradeLabel: string = readability.gradeLevels.fkgl;

const findings: GrammarFindingJs[] = checkGrammar('Example passage.');
const firstKind: string | undefined = findings[0]?.kind;

const templates: string[] = listStructureTemplates();

void fkgl;
void gradeLabel;
void firstKind;
void templates;

import { analyzeManuscript, analyzeManuscriptAsync, compareManuscriptReports, ManuscriptContext } from '../../index.js';
import type { Manuscript, Report, RuleId } from '../../manuscript.js';
const manuscriptInput: Manuscript = { documentId: 'typed', scenes: [{ id: 'first', text: 'She was was tired.' }], configuration: { rules: ['NL-GRAMMAR-REPETITION'] } };
const manuscriptReport: Report = analyzeManuscript(manuscriptInput);
const manuscriptPromise: Promise<Report> = analyzeManuscriptAsync(manuscriptInput);
const manuscriptContext = new ManuscriptContext(1024);
const contextReport: Report = manuscriptContext.analyze(manuscriptInput);
const contextPromise: Promise<Report> = manuscriptContext.analyzeAsync(manuscriptInput);
const introducedRule: RuleId | undefined = compareManuscriptReports(manuscriptReport, contextReport).introduced[0]?.ruleId;
// @ts-expect-error Unknown rule IDs are rejected statically.
const invalidRule: RuleId = 'unknown';
// @ts-expect-error Findings expose typed confidence provenance, not a numeric score alone.
const numericConfidence: number = manuscriptReport.findings[0]!.confidence;
void [manuscriptPromise, contextPromise, introducedRule, invalidRule, numericConfidence];

import { defaultManuscriptConfiguration, listManuscriptRules, parseManuscriptReport } from '../../index.js';
import type { Configuration, Scene } from '../../manuscript.js';
const authorSettings: Configuration = { limits: { maxFindings: 100 }, thresholds: { maxGrade: 16 }, suppressions: [{ ruleId: 'NL-PROSE-LEXICON', reason: 'Narrator diction' }] };
const chronologyHint: Scene = { id: 'unplaced', text: 'A clock stopped.', chronologyLane: 'dream' };
const requiredDefault: number = defaultManuscriptConfiguration().limits.nliPairs;
const knownCatalogId: RuleId = listManuscriptRules()[0]!.id;
const importedReport: Report = parseManuscriptReport({});
const countWarnings: number | undefined = manuscriptReport.summary.severityCounts.warning;
// @ts-expect-error Unknown limits are rejected by the generated contract.
const invalidLimits: Configuration = { limits: { typo: 4 } };
void [authorSettings, chronologyHint, requiredDefault, knownCatalogId, importedReport, countWarnings, invalidLimits];

import { trackWorldState, buildPromisePayoffLedger } from '../../index.js';
import type { WorldFact, PayoffStatus } from '../../standalone.generated.js';
const currentFact = trackWorldState(['Alice was in the cellar.'], ['Alice']).snapshots[0]!.facts[0]!;
if ('characterAt' in currentFact) {
  const location: string = currentFact.characterAt.location;
  void location;
}
const worldVariants: WorldFact[] = [
  { characterAt: { character: 'Alice', location: 'cellar' } },
  { characterPossesses: { character: 'Alice', object: 'key' } },
  { characterKnows: { character: 'Alice', information: 'the key is gone' } },
  { characterWitnessed: { character: 'Alice', event: 'the key vanished' } },
  { coLocated: { character_a: 'Alice', character_b: 'Bob' } },
  { communicationVector: { from: 'Alice', to: 'Bob', content: 'the key is gone' } },
  { objectAt: { object: 'key', location: 'cellar' } },
  { physicalConstraint: { character: 'Alice', constraint: 'locked in' } },
  { temporalOrdering: { before: 'arrival', after: 'departure' } },
];
const payoffVariants: PayoffStatus[] = ['unknown', 'confirmedFulfilled', 'outstanding',
  { fulfilled: { quality: 0.8 } }, { partiallyFulfilled: { completeness: 0.5 } },
  'broken', { subverted: { satisfaction: 0.8 } }, { overpaid: { excess: 0.2 } },
  { deferred: { new_deadline: null } }];
const payoff: PayoffStatus | undefined = buildPromisePayoffLedger(['Who stole the key?'], [5], 'general', [], [], 'three_act', 1).promises[0]?.status;
const comparisonComplete: boolean = compareManuscriptReports(manuscriptReport, contextReport).complete;
// @ts-expect-error serde emits camelCase variant tags, not Rust variant names.
const oldWorldTag: WorldFact = { CharacterAt: { character: 'Alice', location: 'cellar' } };
// @ts-expect-error Unit payoff variants are serialized as camelCase strings.
const oldPayoffTag: PayoffStatus = 'Outstanding';
void [worldVariants, payoffVariants, payoff, comparisonComplete, oldWorldTag, oldPayoffTag];

import { analyzePromiseEvents, analyzeSmog } from '../../index.js';
import type { AuthorPromiseEvent } from '../../standalone.generated.js';
const authored: AuthorPromiseEvent[] = [{kind:'setup',id:'question',description:'The missing key',evidence:{sceneId:'opening',utf16Start:0,utf16End:3,quote:'She'}}];
const authoredId: string | undefined = analyzePromiseEvents(manuscriptInput,authored).promises[0]?.id;
const smogGrade: number | null = analyzeSmog('Tomorrow.').grade;
const smogAvailable: boolean = computeReadability('Tomorrow.').smogAvailable;
// @ts-expect-error Author resolution references are required, not inferred from description.
const missingSetup: AuthorPromiseEvent = {kind:'resolution',id:'payoff',outcome:'fulfilled',evidence:{sceneId:'opening',utf16Start:0,utf16End:3,quote:'She'}};
void [authoredId,smogGrade,smogAvailable,missingSetup];

import { analyzeAuthorKnowledge, trackWorldStateCurrent } from '../../index.js';
import type { AuthorProposition, AuthorKnowledgeEvent, AuthorKnowledgeClassification } from '../../standalone.generated.js';
const proposition: AuthorProposition = {id:'key',description:'The key is hidden',truth:'unknown',evidence:{sceneId:'opening',utf16Start:0,utf16End:3,quote:'She'}};
const knowledgeEvent: AuthorKnowledgeEvent = {id:'reader',sequence:1,propositionId:'key',action:{kind:'uncertain',actor:{kind:'reader'}},evidence:proposition.evidence};
const classification: AuthorKnowledgeClassification | undefined = analyzeAuthorKnowledge(manuscriptInput,[proposition],[knowledgeEvent]).states[0]?.classification;
const lastWorldScene: number | null = trackWorldStateCurrent([],[],{includeDeltas:true}).afterScene;
// @ts-expect-error Actor tags separate reader from a named character.
const missingCharacterId: AuthorKnowledgeEvent = {...knowledgeEvent,action:{kind:'believe',actor:{kind:'character'},value:true}};
// @ts-expect-error Ground truth is tri-state, not an unqualified boolean.
const ambiguousTruth: AuthorProposition = {...proposition,truth:true};
// @ts-expect-error Compact world-state options retain boolean types.
trackWorldStateCurrent([],[],{includeDeltas:'yes'});
void [classification,lastWorldScene,missingCharacterId,ambiguousTruth];

import { extractDialogueChecked, analyzeAuthorCausality, buildAuthorMentalModel } from '../../index.js';
import type { AuthorCausalNode } from '../../standalone.generated.js';
const locatedQuote: string | undefined=extractDialogueChecked('"Hello."',0,[],{attribution:'explicit_only'}).lines[0]?.line.quote;
const causalNodes: AuthorCausalNode[]=[{id:'fixed',mechanism:{kind:'fixed',value:true},evidence:proposition.evidence}];
const changedNode: boolean | undefined=analyzeAuthorCausality(manuscriptInput,causalNodes,[{nodeId:'fixed',value:false}]).nodes[0]?.intervened;
const readerUnknown: number=buildAuthorMentalModel(manuscriptInput,[proposition],[knowledgeEvent]).readerSummary.unassessedPropositions;
// @ts-expect-error Declared Boolean mechanisms require actual Boolean values.
const invalidCausal: AuthorCausalNode={...causalNodes[0],mechanism:{kind:'fixed',value:'true'}};
void [locatedQuote,changedNode,readerUnknown,invalidCausal];

import {analyzeSubtextAnnotations,analyzeThematicAnnotations,resolveAuthorCoreference,buildVoiceProfileChecked,buildTimeline} from '../../index.js';
import type {AuthorSubtextAnnotation,AuthorThematicAnnotation,AuthorThematicThesis,AuthorCoreferenceLink} from '../../standalone.generated.js';
const subtextAnnotations:AuthorSubtextAnnotation[]=[{id:'subtext',surface:proposition.evidence,implied:'A supplied interpretation',subtextType:'EmotionalDenial',characters:[],resolution:null}];
const thematicTheses:AuthorThematicThesis[]=[{id:'thesis',statement:'A supplied position',opposes:[]}];
const thematicAnnotations:AuthorThematicAnnotation[]=[{id:'claim',thesisId:'thesis',source:proposition.evidence,valence:'Supporting',delivery:'Narration'}];
const coreferenceLinks:AuthorCoreferenceLink[]=[{id:'pronoun',characterId:'Alice',mention:proposition.evidence,antecedent:proposition.evidence}];
const surfaceSource:string|undefined=analyzeSubtextAnnotations(manuscriptInput,subtextAnnotations).instances[0]?.surface.quote;
const thematicEvidence:string|undefined=analyzeThematicAnnotations(manuscriptInput,thematicTheses,thematicAnnotations).evidence[0]?.thesisId;
const linkedSource:string|undefined=resolveAuthorCoreference(manuscriptInput,coreferenceLinks).links[0]?.characterId;
const eligibleTurns:number=buildVoiceProfileChecked('Alice',[]).sample.turnCount;
const timelineLabel:string|null|undefined=buildTimeline([['opening','It was Monday.']],[{document_id:'opening',chronology_key:null,lane:null,label:'Chapter1',time_hint:null,notes:null}])[0]?.label;
// @ts-expect-error Annotation valence follows the existing PascalCase wire enum.
const invalidValence:AuthorThematicAnnotation={...thematicAnnotations[0],valence:'supporting'};
void [surfaceSource,thematicEvidence,linkedSource,eligibleTurns,timelineLabel,invalidValence];

import {checkGenreComplianceChecked} from '../../index.js';
const genreCoverage:number=checkGenreComplianceChecked('romance',{hasHea:false,conventionJudgments:{'HEA/HFN':'NotMet'}}).evidenceCoverage;
// @ts-expect-error Caller convention statuses retain their existing PascalCase wire names.
checkGenreComplianceChecked('romance',{conventionJudgments:{'HEA/HFN':'not_met'}});
void genreCoverage;

import {measureNarrativeUncertainty,measureSemanticNovelty} from '../../index.js';
import type {SuppliedUncertainty} from '../../standalone.generated.js';
const uncertaintyObservations:SuppliedUncertainty[]=[{questionId:'q',sceneId:'opening',outcomes:[{id:'one',probability:1}]}];
const uncertaintyBits:number|undefined=measureNarrativeUncertainty(manuscriptInput,uncertaintyObservations).measurements[0]?.entropyBits;
const vectorNovelty:number|null|undefined=measureSemanticNovelty([[1,0]],{includeNearest:true}).points[0]?.novelty;
// @ts-expect-error Outcome probabilities are numeric rather than textual confidence labels.
const invalidProbability:SuppliedUncertainty={...uncertaintyObservations[0],outcomes:[{id:'one',probability:'likely'}]};
void [uncertaintyBits,vectorNovelty,invalidProbability];

import {analyzeAuthorEpistemics,analyzeOpeningChecked} from '../../index.js';
import type {AuthorEpistemicQuery} from '../../standalone.generated.js';
const epistemicQueries:AuthorEpistemicQuery[]=[{id:'query',propositionId:'p',observer:{kind:'reader'},subject:{kind:'character',characterId:'Alice'},atSequence:3}];
const queriedEvent:string|null|undefined=analyzeAuthorEpistemics(manuscriptInput,[proposition],[knowledgeEvent],epistemicQueries).answers[0]?.observer.eventId;
const openingDrift:number|undefined=analyzeOpeningChecked('An opening.','general').voiceComparison?.overall;
// @ts-expect-error An unlabelled numeric vector has no supported baseline voice contract.
analyzeOpeningChecked('An opening.','general',[1,2,3]);
void [queriedEvent,openingDrift];

import {analyzeSceneChecked,analyzeHistoricalCatalogue,extractNamedEntities,extractPersons,measureFindingConfidence} from '../../index.js';
import type {HistoricalEntry,HistoricalUsage} from '../../standalone.generated.js';
const historicalCatalogue:HistoricalEntry[]=[{id:'constraint',term:'silver bell',earliestYear:1800,basis:'author_constraint',reference:'Author-supplied test constraint',context:'Fictional date',senseAmbiguous:true}];
const historicalUsages:HistoricalUsage[]=[{id:'usage',entryId:'constraint',evidence:proposition.evidence}];
const declaredConflict:boolean|null|undefined=analyzeHistoricalCatalogue(manuscriptInput,historicalCatalogue,historicalUsages,1700,0).candidates[0]?.dateConflict;
const historicalAbstentions:string[]=analyzeSceneChecked([],1700,0).abstainedTerms;
const nerIdentity:string|undefined=extractNamedEntities([]).modelIdentity?.manifest?.modelSha256;
const modelEpoch:number=extractNamedEntities([]).modelGeneration;
const personEpoch:number=extractPersons([]).modelGeneration;
const agreementIndex:number|null=measureFindingConfidence(0,0,0).value;
void [declaredConflict,historicalAbstentions,nerIdentity,modelEpoch,personEpoch,agreementIndex];
