//! `craft` category napi bindings. See `napi_bindings/mod.rs` for the pattern.
//!
//! Return values go through the JSON bridge (`serde_json::to_value`) rather
//! than hand-written `*Js` DTOs, since this module covers ~25 functions with
//! many-field, nested return types. A handful of functions take a
//! previously-bound function's *output* struct as input (e.g. aggregating
//! per-scene metrics, comparing voice profiles); those get small
//! `#[napi(object)]` input DTOs, converted to the real `craft` type before
//! calling through.

use napi_derive::napi;
use serde::Serialize;

use crate::craft::{dialogue, dialogue_realism, lexical, pacing, prose_quality, subtext,
                    syntax_tension, voice};

// ---------------------------------------------------------------------------
// Input DTOs (for functions whose real signature takes a struct produced by
// another bound function's output)
// ---------------------------------------------------------------------------

#[napi(object)]
pub struct PacingMetricsInput {
    pub scene_id: u32,
    pub dialogue_ratio: f64,
    pub is_talking_head: bool,
    pub velocity: f64,
    pub word_count: u32,
    pub entity_density: f64,
    pub noun_density: f64,
    pub information_novelty: f64,
    pub action_density: f64,
}

fn to_pacing_metrics(m: PacingMetricsInput) -> pacing::PacingMetrics {
    pacing::PacingMetrics {
        scene_id: m.scene_id as usize,
        dialogue_ratio: m.dialogue_ratio,
        is_talking_head: m.is_talking_head,
        velocity: m.velocity,
        word_count: m.word_count as usize,
        entity_density: m.entity_density,
        noun_density: m.noun_density,
        information_novelty: m.information_novelty,
        action_density: m.action_density,
    }
}

#[napi(object)]
pub struct DialogueLineInput {
    pub scene_id: u32,
    pub quote: String,
    pub speaker: String,
    pub confidence: f64,
    pub method: String,
    pub tag_verb: Option<String>,
}

fn to_dialogue_line(l: DialogueLineInput) -> dialogue::DialogueLine {
    dialogue::DialogueLine {
        scene_id: l.scene_id as usize,
        quote: l.quote,
        speaker: l.speaker,
        confidence: l.confidence,
        method: l.method,
        tag_verb: l.tag_verb,
    }
}

#[napi(object)]
pub struct CharacterVoiceProfileInput {
    pub character: String,
    pub avg_sentence_length: f64,
    pub vocabulary_richness: f64,
    pub question_rate: f64,
    pub exclamation_rate: f64,
    pub contraction_rate: f64,
    pub formality_score: f64,
    pub avg_word_length: f64,
    pub unique_phrases: Vec<String>,
    pub top_words: Vec<String>,
}

fn to_character_voice_profile(p: CharacterVoiceProfileInput) -> voice::CharacterVoiceProfile {
    voice::CharacterVoiceProfile {
        character: p.character,
        avg_sentence_length: p.avg_sentence_length,
        vocabulary_richness: p.vocabulary_richness,
        question_rate: p.question_rate,
        exclamation_rate: p.exclamation_rate,
        contraction_rate: p.contraction_rate,
        formality_score: p.formality_score,
        avg_word_length: p.avg_word_length,
        unique_phrases: p.unique_phrases,
        top_words: p.top_words,
    }
}

// ---------------------------------------------------------------------------
// pacing.rs
// ---------------------------------------------------------------------------

/// Analyze a single scene and return its pacing metrics.
#[napi(
    js_name = "analyzePacing",
    ts_return_type = "{ sceneId: number; dialogueRatio: number; isTalkingHead: boolean; velocity: number; wordCount: number; entityDensity: number; nounDensity: number; informationNovelty: number; actionDensity: number }"
)]
pub fn analyze_pacing(
    scene_id: u32,
    dialogue_tokens: u32,
    total_tokens: u32,
    sentences_count: u32,
) -> napi::Result<serde_json::Value> {
    let result = pacing::analyze_scene(
        scene_id as usize,
        dialogue_tokens as usize,
        total_tokens as usize,
        sentences_count as usize,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SceneInfoDensityWithSeenNouns {
    #[serde(flatten)]
    info: pacing::SceneInfoDensity,
    seen_nouns: Vec<String>,
}

/// Analyze information density of a scene's text. `seen_nouns` carries nouns
/// seen in prior scenes across calls (JS has no mutable-reference channel, so
/// the updated set is returned alongside the density result instead of
/// mutated in place).
#[napi(
    js_name = "analyzeSceneInfoDensity",
    ts_return_type = "{ entityDensity: number; actionDensity: number; nounDensity: number; newNounCount: number; totalNounCount: number; seenNouns: Array<string> }"
)]
pub fn analyze_scene_info_density(
    scene_text: String,
    seen_nouns: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let mut seen: std::collections::HashSet<String> = seen_nouns.into_iter().collect();
    let info = pacing::analyze_scene_info_density(&scene_text, &mut seen);
    let mut seen_nouns: Vec<String> = seen.into_iter().collect();
    seen_nouns.sort();
    let combined = SceneInfoDensityWithSeenNouns { info, seen_nouns };
    serde_json::to_value(combined).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Aggregate pacing metrics across all scenes. Returns `null` when
/// `scene_metrics` is empty.
#[napi(
    js_name = "aggregatePacing",
    ts_return_type = "{ averageDialogueRatio: number; averageVelocity: number; perSceneVelocity: Array<number>; perSceneDialogueRatio: Array<number>; perSceneLabel: Array<string>; perSceneAssessment: Array<string>; talkingHeadScenes: Array<number>; thsCount: number; pacingTrend: string; velocityVariance: number; slowScenes: Array<number>; fastScenes: Array<number>; perSceneEntityDensity: Array<number>; perSceneActionDensity: Array<number>; perSceneNovelty: Array<number>; avgEntityDensity: number; avgActionDensity: number } | null"
)]
pub fn aggregate_pacing(scene_metrics: Vec<PacingMetricsInput>) -> napi::Result<serde_json::Value> {
    let metrics: Vec<pacing::PacingMetrics> =
        scene_metrics.into_iter().map(to_pacing_metrics).collect();
    let result = pacing::aggregate_pacing(&metrics);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

// ---------------------------------------------------------------------------
// dialogue.rs
// ---------------------------------------------------------------------------

/// Extract dialogue from a scene and attribute each quote to a speaker.
#[napi(
    js_name = "extractDialogue",
    ts_return_type = "Array<{ sceneId: number; quote: string; speaker: string; confidence: number; method: string; tagVerb?: string }>"
)]
pub fn extract_dialogue(
    scene_text: String,
    scene_id: u32,
    known_characters: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let result = dialogue::extract_dialogue(&scene_text, scene_id as usize, &known_characters);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Analyze dialogue patterns across all extracted dialogue lines.
#[napi(
    js_name = "analyzeDialoguePatterns",
    ts_return_type = "{ totalLines: number; attributedLines: number; attributionRate: number; perCharacter: Record<string, { lineCount: number; wordCount: number; avgLineLength: number; vocabularyRichness: number; questionRate: number; exclamationRate: number }>; dialogueDistribution: Record<string, number>; mostTalkative: string; mostVerbose: string; voiceProfiles: Record<string, { avgWordLength: number; vocabularyRichness: number; questionFrequency: number; avgLineLength: number; formalityScore: number; tagVarietyScore: number; speechVerbsUsed: Array<string> }>; sameSoundingPairs: Array<{ characterA: string; characterB: string; similarity: number }>; conversationFlow: { edges: Array<{ from: string; to: string; count: number }>; dominators: Array<string>; neverInitiators: Array<string>; monologues: Array<{ speaker: string; lineCount: number; sceneId: number }> } }"
)]
pub fn analyze_dialogue_patterns(
    all_dialogue: Vec<DialogueLineInput>,
    known_characters: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let lines: Vec<dialogue::DialogueLine> =
        all_dialogue.into_iter().map(to_dialogue_line).collect();
    let result = dialogue::analyze_dialogue_patterns(&lines, &known_characters);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

// ---------------------------------------------------------------------------
// dialogue_realism.rs
// ---------------------------------------------------------------------------

/// Analyze dialogue realism: voice fingerprints, info dumps, formality
/// issues, identical-voice pairs, and power dynamics between characters.
#[napi(
    js_name = "analyzeDialogueRealism",
    ts_return_type = "{ characterVoices: Array<{ character: string; avgTurnLength: number; vocabularyRichness: number; formalityLevel: number; contractionRate: number; questionRate: number; exclamationRate: number; frequentWords: Array<string> }>; voicePairs: Array<{ characterA: string; characterB: string; similarity: number }>; issues: Array<{ scene: number; issueType: string; description: string; severity: number; characters: Array<string> }>; powerDynamics: Array<{ characterA: string; characterB: string; dominant: string; indicators: Array<string> }>; overallRealism: number; voiceDistinctiveness: number; infoDumpCount: number }"
)]
pub fn analyze_dialogue_realism(
    turn_scenes: Vec<u32>,
    turn_speakers: Vec<String>,
    turn_texts: Vec<String>,
    characters: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let dialogue_turns: Vec<(usize, String, String)> = turn_scenes
        .into_iter()
        .zip(turn_speakers)
        .zip(turn_texts)
        .map(|((scene, speaker), text)| (scene as usize, speaker, text))
        .collect();
    let char_refs: Vec<&str> = characters.iter().map(String::as_str).collect();
    let result = dialogue_realism::analyze_dialogue_realism(&dialogue_turns, &char_refs);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

// ---------------------------------------------------------------------------
// syntax_tension.rs
// ---------------------------------------------------------------------------

/// Analyze syntax tension in scene text using the fallback (regex-only) path.
#[napi(
    js_name = "analyzeSyntaxTension",
    ts_return_type = "{ score: number; lengthVariation: number; burstCount: number; sentenceCount: number; avgSentenceLength: number; questionDensity: number; dialogueShiftScore: number; paragraphBurstScore: number; exclamationDensity: number; suspenseKeywordScore: number }"
)]
pub fn analyze_syntax_tension(scene_text: String) -> napi::Result<serde_json::Value> {
    let result = syntax_tension::analyze_scene_text(&scene_text);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Compute aggregate tension metrics from per-scene suspense scores.
#[napi(
    js_name = "getTensionMetrics",
    ts_return_type = "{ maxSuspenseScore: number; averageSuspenseScore: number; pacingStyle: string }"
)]
pub fn get_tension_metrics(scene_scores: Vec<f64>) -> napi::Result<serde_json::Value> {
    let result = syntax_tension::get_tension_metrics(&scene_scores);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

// ---------------------------------------------------------------------------
// lexical.rs
// ---------------------------------------------------------------------------

/// Combine ranked id lists (best first) with reciprocal rank fusion. Returns
/// the fused ranking, strongest first, truncated to `limit`, as
/// `[id, score]` pairs.
#[napi(
    js_name = "reciprocalRankFusion",
    ts_return_type = "Array<[string, number]>"
)]
pub fn reciprocal_rank_fusion(
    rankings: Vec<Vec<String>>,
    k: f64,
    limit: u32,
) -> napi::Result<serde_json::Value> {
    let result = lexical::reciprocal_rank_fusion(&rankings, k, limit as usize);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

// ---------------------------------------------------------------------------
// prose_quality.rs
// ---------------------------------------------------------------------------

/// Collect prose quality issues from scenes: overlong sentences, passive
/// voice, adverb clusters, repetitive openers, cliches, floating-head
/// dialogue, filter words, monotonous rhythm, and purple prose.
#[napi(
    js_name = "collectProseExamples",
    ts_return_type = "Array<{ sceneId: number; issueType: string; snippet: string; suggestion: string }>"
)]
pub fn collect_prose_examples(scenes: Vec<String>) -> napi::Result<serde_json::Value> {
    let result = prose_quality::collect_prose_examples(&scenes);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

// ---------------------------------------------------------------------------
// voice.rs
// ---------------------------------------------------------------------------

/// Build a character's voice profile from their dialogue lines.
#[napi(
    js_name = "buildVoiceProfile",
    ts_return_type = "{ character: string; avgSentenceLength: number; vocabularyRichness: number; questionRate: number; exclamationRate: number; contractionRate: number; formalityScore: number; avgWordLength: number; uniquePhrases: Array<string>; topWords: Array<string> }"
)]
pub fn build_voice_profile(
    character_name: String,
    dialogue_lines: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let result = voice::build_voice_profile(&character_name, &dialogue_lines);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Pairwise-compare voice profiles and flag characters whose voices are too
/// similar.
#[napi(
    js_name = "compareVoices",
    ts_return_type = "Array<{ charA: string; charB: string; similarity: number; mostSimilarDimension: string; suggestion: string }>"
)]
pub fn compare_voices(profiles: Vec<CharacterVoiceProfileInput>) -> napi::Result<serde_json::Value> {
    let profiles: Vec<voice::CharacterVoiceProfile> =
        profiles.into_iter().map(to_character_voice_profile).collect();
    let result = voice::compare_voices(&profiles);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Measure how far a `current` (revised) voice profile has drifted from a
/// `baseline` (established) profile across the features both expose.
#[napi(
    js_name = "measureVoiceDrift",
    ts_return_type = "{ overall: number; perFeature: Array<{ feature: string; delta: number; baseline: number; current: number }>; notes: Array<string>; drifted: boolean }"
)]
pub fn measure_voice_drift(
    baseline: CharacterVoiceProfileInput,
    current: CharacterVoiceProfileInput,
) -> napi::Result<serde_json::Value> {
    let baseline = to_character_voice_profile(baseline);
    let current = to_character_voice_profile(current);
    let result = voice::measure_voice_drift(&baseline, &current);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Convenience: build both profiles from baseline-text and current-text
/// lines (same speaker/narrator label) and measure drift between them.
#[napi(
    js_name = "measureVoiceDriftFromText",
    ts_return_type = "{ overall: number; perFeature: Array<{ feature: string; delta: number; baseline: number; current: number }>; notes: Array<string>; drifted: boolean }"
)]
pub fn measure_voice_drift_from_text(
    label: String,
    baseline_lines: Vec<String>,
    current_lines: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let result =
        voice::measure_voice_drift_from_text(&label, &baseline_lines, &current_lines);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Build voice profiles for every character and flag same-sounding pairs.
#[napi(
    js_name = "analyzeCharacterVoices",
    ts_return_type = "{ profiles: Array<{ character: string; avgSentenceLength: number; vocabularyRichness: number; questionRate: number; exclamationRate: number; contractionRate: number; formalityScore: number; avgWordLength: number; uniquePhrases: Array<string>; topWords: Array<string> }>; similarities: Array<{ charA: string; charB: string; similarity: number; mostSimilarDimension: string; suggestion: string }>; distinctVoices: boolean }"
)]
pub fn analyze_character_voices(
    character_names: Vec<String>,
    character_dialogue_lines: Vec<Vec<String>>,
) -> napi::Result<serde_json::Value> {
    let dialogue_data: Vec<(String, Vec<String>)> = character_names
        .into_iter()
        .zip(character_dialogue_lines)
        .collect();
    let result = voice::analyze_character_voices(dialogue_data);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

// ---------------------------------------------------------------------------
// subtext.rs
// ---------------------------------------------------------------------------

/// Analyze subtext across scenes (currently a reserved-for-reasoner stub:
/// always returns empty instances with zero density).
#[napi(
    js_name = "analyzeSubtext",
    ts_return_type = "{ instances: Array<{ scene: number; paragraph: number; surface: string; implied: string; subtextType: 'EmotionalDenial' | 'PowerPlay' | 'SelfDeception' | 'CodedCommunication' | 'DramaticIronySubtext' | 'Deflection' | 'VerbalIrony' | 'ActionContradiction'; density: number; evidence: Array<string>; characters: Array<string>; resolved: boolean; resolutionScene: number | null }>; sceneSummaries: Array<{ scene: number; averageDensity: number; instanceCount: number; dominantType: 'EmotionalDenial' | 'PowerPlay' | 'SelfDeception' | 'CodedCommunication' | 'DramaticIronySubtext' | 'Deflection' | 'VerbalIrony' | 'ActionContradiction' | null; appropriateness: number }>; globalDensity: number; genreCalibratedScore: number; onTheNose: Array<[number, string]>; unresolvedSubtext: Array<{ scene: number; paragraph: number; surface: string; implied: string; subtextType: 'EmotionalDenial' | 'PowerPlay' | 'SelfDeception' | 'CodedCommunication' | 'DramaticIronySubtext' | 'Deflection' | 'VerbalIrony' | 'ActionContradiction'; density: number; evidence: Array<string>; characters: Array<string>; resolved: boolean; resolutionScene: number | null }>; typeDistribution: Record<string, number>; characterPatterns: Record<string, [number, string]>; advice: Array<{ scene: number; issueType: string; quotedText: string; explanation: string; suggestion: string | null }>; craftNotes: Array<string> }"
)]
pub fn analyze_subtext(
    scenes: Vec<String>,
    dialogue_segment_scenes: Vec<u32>,
    dialogue_segment_speakers: Vec<String>,
    dialogue_segment_quotes: Vec<String>,
    dialogue_segment_contexts: Vec<String>,
    emotional_state_characters: Vec<String>,
    emotional_state_scenes: Vec<u32>,
    emotional_state_emotions: Vec<String>,
    genre: String,
) -> napi::Result<serde_json::Value> {
    let scene_refs: Vec<&str> = scenes.iter().map(String::as_str).collect();
    let dialogue_segments: Vec<(usize, String, String, String)> = dialogue_segment_scenes
        .into_iter()
        .zip(dialogue_segment_speakers)
        .zip(dialogue_segment_quotes)
        .zip(dialogue_segment_contexts)
        .map(|(((scene, speaker), quote), context)| (scene as usize, speaker, quote, context))
        .collect();
    let character_emotional_states: Vec<(String, usize, String)> = emotional_state_characters
        .into_iter()
        .zip(emotional_state_scenes)
        .zip(emotional_state_emotions)
        .map(|((character, scene), emotion)| (character, scene as usize, emotion))
        .collect();
    let result = subtext::analyze_subtext(
        &scene_refs,
        &dialogue_segments,
        &character_emotional_states,
        &genre,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}
