// ESM wrapper around the CommonJS native-addon loader (index.js, auto-generated
// by `napi build --platform`). A native addon loads via `require`, so this
// file is hand-maintained, not generated -- keep its exports in sync with
// index.d.ts when adding a new `#[napi]` binding.

import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const native = require('./index.js')

export const {
  aggregatePacing,
  analyzeCausality,
  analyzeCharacterVoices,
  analyzeDialoguePatterns,
  analyzeDialogueRealism,
  analyzeForeshadowing,
  analyzeGaps,
  analyzeOpening,
  analyzePacing,
  analyzeResolution,
  analyzeScene,
  analyzeSceneInfoDensity,
  analyzeSubtext,
  analyzeSyntaxTension,
  buildCharacterGenderMap,
  buildMentalModel,
  buildPromisePayoffLedger,
  buildTimeline,
  buildVoiceProfile,
  checkGenreCompliance,
  checkGrammar,
  collectProseExamples,
  compareVoices,
  computeFindingConfidence,
  computeNarrativeEntropy,
  computeReadability,
  computeReadabilityConfidence,
  computeSceneConfidence,
  detectGenre,
  detectGenreSmart,
  detectThemesSmart,
  evaluateAgainstGenre,
  explainPacing,
  explainReadability,
  explainShowDontTell,
  explainTension,
  explainToneShift,
  explainWhiteRoom,
  extractDialogue,
  genreFromLabel,
  getGenreProfile,
  getStructureTemplate,
  getTensionMetrics,
  listStructureTemplates,
  measureVoiceDrift,
  measureVoiceDriftFromText,
  reciprocalRankFusion,
  resolveCoreferences,
  resolveEntities,
  resolveEntitiesWithCorrections,
  suggestBestTemplate,
  trackEpistemics,
  trackThematicArgument,
  trackWorldState,
  trainAndFindNearest,
  trainAndSimilarity,
  validateStructure,
  wordFrequencies,
  wordFrequenciesFromWords,
} = native

export default native
