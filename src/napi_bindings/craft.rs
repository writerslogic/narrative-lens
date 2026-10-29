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

use crate::craft::{
    dialogue, dialogue_realism, lexical, pacing, prose_quality, readability, subtext,
    syntax_tension, voice,
};

#[napi(
    js_name = "analyzeSmog",
    ts_return_type = "import('./standalone.generated').AnalyzeSmogResult"
)]
pub fn analyze_smog(text: String) -> napi::Result<serde_json::Value> {
    super::bounded_texts([text.as_str()], 1)?;
    super::encode_result(readability::analyze_smog(&text))
}

// ---------------------------------------------------------------------------
// Input DTOs (for functions whose real signature takes a struct produced by
// another bound function's output)
// ---------------------------------------------------------------------------

#[napi(object)]
pub struct PacingMetricsInput {
    pub scene_id: f64,
    pub dialogue_ratio: f64,
    pub is_talking_head: bool,
    pub velocity: f64,
    pub word_count: f64,
    pub entity_density: f64,
    pub noun_density: f64,
    pub information_novelty: f64,
    pub action_density: f64,
}

fn to_pacing_metrics(m: PacingMetricsInput) -> napi::Result<pacing::PacingMetrics> {
    Ok(pacing::PacingMetrics {
        scene_id: super::unsigned_index(m.scene_id)?,
        dialogue_ratio: super::finite_range(m.dialogue_ratio, 0.0, 1.0)?,
        is_talking_head: m.is_talking_head,
        velocity: super::finite_range(m.velocity, 0.0, u32::MAX as f64)?,
        word_count: super::unsigned_index(m.word_count)?,
        entity_density: super::finite_range(m.entity_density, 0.0, 100.0)?,
        noun_density: super::finite_range(m.noun_density, 0.0, 100.0)?,
        information_novelty: super::finite_range(m.information_novelty, 0.0, 1.0)?,
        action_density: super::finite_range(m.action_density, 0.0, 100.0)?,
    })
}

#[napi(object)]
pub struct DialogueLineInput {
    pub scene_id: f64,
    pub quote: String,
    pub speaker: String,
    pub confidence: f64,
    pub method: String,
    pub tag_verb: Option<String>,
}

fn to_dialogue_line(l: DialogueLineInput) -> napi::Result<dialogue::DialogueLine> {
    Ok(dialogue::DialogueLine {
        scene_id: super::unsigned_index(l.scene_id)?,
        quote: l.quote,
        speaker: l.speaker,
        confidence: super::finite_range(l.confidence, 0.0, 1.0)?,
        method: l.method,
        tag_verb: l.tag_verb,
    })
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

pub(super) fn to_character_voice_profile(
    p: CharacterVoiceProfileInput,
) -> voice::CharacterVoiceProfile {
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
    ts_return_type = "import('./standalone.generated').AnalyzePacingResult"
)]
pub fn analyze_pacing(
    scene_id: f64,
    dialogue_tokens: f64,
    total_tokens: f64,
    sentences_count: f64,
) -> napi::Result<serde_json::Value> {
    if dialogue_tokens > total_tokens {
        return Err(napi::Error::from_reason(
            "dialogue tokens must be a subset of total tokens",
        ));
    }
    let result = pacing::analyze_scene(
        super::unsigned_index(scene_id)?,
        super::unsigned_index(dialogue_tokens)?,
        super::unsigned_index(total_tokens)?,
        super::unsigned_index(sentences_count)?,
    );
    super::encode_result(result)
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
    super::bounded_texts([scene_text.as_str()], 1)?;
    super::bounded_texts(seen_nouns.iter().map(String::as_str), 1_000_000)?;
    let mut seen: std::collections::HashSet<String> = seen_nouns.into_iter().collect();
    let info = pacing::analyze_scene_info_density(&scene_text, &mut seen);
    let mut seen_nouns: Vec<String> = seen.into_iter().collect();
    seen_nouns.sort();
    let combined = SceneInfoDensityWithSeenNouns { info, seen_nouns };
    super::encode_result(combined)
}

/// Aggregate pacing metrics across all scenes. Returns `null` when
/// `scene_metrics` is empty.
#[napi(
    js_name = "aggregatePacing",
    ts_return_type = "import('./standalone.generated').AggregatePacingResult"
)]
pub fn aggregate_pacing(scene_metrics: Vec<PacingMetricsInput>) -> napi::Result<serde_json::Value> {
    if scene_metrics.len() > 10_000 {
        return Err(napi::Error::from_reason(
            "pacing metrics exceed10000 scenes",
        ));
    }
    let metrics: Vec<pacing::PacingMetrics> = scene_metrics
        .into_iter()
        .map(to_pacing_metrics)
        .collect::<napi::Result<_>>()?;
    let result = pacing::aggregate_pacing(&metrics);
    super::encode_result(result)
}

// ---------------------------------------------------------------------------
// dialogue.rs
// ---------------------------------------------------------------------------

/// Extract dialogue from a scene and attribute each quote to a speaker.
#[napi(
    js_name = "extractDialogue",
    ts_return_type = "import('./standalone.generated').ExtractDialogueResult"
)]
pub fn extract_dialogue(
    scene_text: String,
    scene_id: f64,
    known_characters: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let result = dialogue::extract_dialogue_checked(
        &scene_text,
        super::unsigned_index(scene_id)?,
        &known_characters,
        &dialogue::DialogueExtractionOptions {
            attribution: dialogue::DialogueAttributionMode::ConservativeHeuristic,
            include_indirect: true,
        },
    )
    .map_err(super::reason)?;
    super::encode_result(
        result
            .lines
            .into_iter()
            .map(|located| located.line)
            .collect::<Vec<_>>(),
    )
}

#[napi(
    js_name = "extractDialogueChecked",
    ts_args_type = "sceneText: string, sceneId: number, knownCharacters: string[], options?: Partial<import('./standalone.generated').DialogueExtractionOptions>",
    ts_return_type = "import('./standalone.generated').ExtractDialogueCheckedResult"
)]
pub fn extract_dialogue_checked(
    scene_text: String,
    scene_id: f64,
    known_characters: Vec<String>,
    options: Option<serde_json::Value>,
) -> napi::Result<serde_json::Value> {
    let options: dialogue::DialogueExtractionOptions = super::decode_input(
        options.unwrap_or_else(|| serde_json::json!({})),
        "invalid dialogue extraction options",
    )?;
    let result = dialogue::extract_dialogue_checked(
        &scene_text,
        super::unsigned_index(scene_id)?,
        &known_characters,
        &options,
    )
    .map_err(super::reason)?;
    super::encode_result(result)
}

/// Analyze dialogue patterns across all extracted dialogue lines.
#[napi(
    js_name = "analyzeDialoguePatterns",
    ts_return_type = "import('./standalone.generated').AnalyzeDialoguePatternsResult"
)]
pub fn analyze_dialogue_patterns(
    all_dialogue: Vec<DialogueLineInput>,
    known_characters: Vec<String>,
) -> napi::Result<serde_json::Value> {
    crate::craft::measurements::validate_names(known_characters.iter().map(String::as_str))
        .map_err(super::reason)?;
    super::bounded_texts(all_dialogue.iter().map(|line| line.quote.as_str()), 10_000)?;
    let lines: Vec<dialogue::DialogueLine> = all_dialogue
        .into_iter()
        .map(to_dialogue_line)
        .collect::<napi::Result<_>>()?;
    let result = dialogue::analyze_dialogue_patterns(&lines, &known_characters);
    super::encode_result(result)
}

// ---------------------------------------------------------------------------
// dialogue_realism.rs
// ---------------------------------------------------------------------------

/// Measure dialogue stylometry and located lexical cues, with sample coverage.
#[napi(
    js_name = "analyzeDialogueRealism",
    ts_return_type = "import('./standalone.generated').AnalyzeDialogueRealismResult"
)]
pub fn analyze_dialogue_realism(
    turn_scenes: Vec<f64>,
    turn_speakers: Vec<String>,
    turn_texts: Vec<String>,
    characters: Vec<String>,
) -> napi::Result<serde_json::Value> {
    if turn_scenes.len() != turn_speakers.len() || turn_scenes.len() != turn_texts.len() {
        return Err(napi::Error::from_reason(
            "dialogue turn arrays must have equal lengths",
        ));
    }
    let dialogue_turns: Vec<(usize, String, String)> = turn_scenes
        .into_iter()
        .zip(turn_speakers)
        .zip(turn_texts)
        .map(|((scene, speaker), text)| Ok((super::unsigned_index(scene)?, speaker, text)))
        .collect::<napi::Result<_>>()?;
    let char_refs: Vec<&str> = characters.iter().map(String::as_str).collect();
    let result = dialogue_realism::analyze_dialogue_realism_checked(&dialogue_turns, &char_refs)
        .map_err(super::reason)?;
    super::encode_result(result)
}

// ---------------------------------------------------------------------------
// syntax_tension.rs
// ---------------------------------------------------------------------------

/// Analyze syntax tension in scene text using the fallback (regex-only) path.
#[napi(
    js_name = "analyzeSyntaxTension",
    ts_return_type = "import('./standalone.generated').AnalyzeSyntaxTensionResult"
)]
pub fn analyze_syntax_tension(scene_text: String) -> napi::Result<serde_json::Value> {
    super::bounded_texts([scene_text.as_str()], 1)?;
    let result = syntax_tension::analyze_scene_text(&scene_text);
    super::encode_result(result)
}

/// Compute aggregate tension metrics from per-scene suspense scores.
#[napi(
    js_name = "getTensionMetrics",
    ts_return_type = "import('./standalone.generated').GetTensionMetricsResult"
)]
pub fn get_tension_metrics(scene_scores: Vec<f64>) -> napi::Result<serde_json::Value> {
    if scene_scores.len() > 10_000 {
        return Err(napi::Error::from_reason(
            "tension scores exceed10000 scenes",
        ));
    }
    for value in &scene_scores {
        super::finite_range(*value, 0.0, 1.0)?;
    }
    let result = syntax_tension::get_tension_metrics(&scene_scores);
    super::encode_result(result)
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
    limit: f64,
) -> napi::Result<serde_json::Value> {
    super::finite_range(k, 0.0, f64::MAX)?;
    let result =
        lexical::reciprocal_rank_fusion_checked(&rankings, k, super::unsigned_index(limit)?)
            .map_err(super::reason)?;
    super::encode_result(result)
}

// ---------------------------------------------------------------------------
// prose_quality.rs
// ---------------------------------------------------------------------------

/// Collect prose quality issues from scenes: overlong sentences, passive
/// voice, adverb clusters, repetitive openers, cliches, floating-head
/// dialogue, filter words, monotonous rhythm, and purple prose.
#[napi(
    js_name = "collectProseExamples",
    ts_return_type = "import('./standalone.generated').CollectProseExamplesResult"
)]
pub fn collect_prose_examples(scenes: Vec<String>) -> napi::Result<serde_json::Value> {
    super::bounded_texts(scenes.iter().map(String::as_str), 10_000)?;
    let result = prose_quality::collect_prose_examples(&scenes);
    super::encode_result(result)
}

// ---------------------------------------------------------------------------
// voice.rs
// ---------------------------------------------------------------------------

/// Build a character's voice profile from their dialogue lines.
#[napi(
    js_name = "buildVoiceProfile",
    ts_return_type = "import('./standalone.generated').BuildVoiceProfileResult"
)]
pub fn build_voice_profile(
    character_name: String,
    dialogue_lines: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let result = voice::build_voice_profile_checked(&character_name, &dialogue_lines)
        .map_err(super::reason)?;
    super::encode_result(result.profile)
}

#[napi(
    js_name = "buildVoiceProfileChecked",
    ts_return_type = "import('./standalone.generated').BuildVoiceProfileCheckedResult"
)]
pub fn build_voice_profile_checked(
    character_name: String,
    dialogue_lines: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let result = voice::build_voice_profile_checked(&character_name, &dialogue_lines)
        .map_err(super::reason)?;
    super::encode_result(result)
}

/// Pairwise-compare voice profiles and flag characters whose voices are too
/// similar.
#[napi(
    js_name = "compareVoices",
    ts_return_type = "import('./standalone.generated').CompareVoicesResult"
)]
pub fn compare_voices(
    profiles: Vec<CharacterVoiceProfileInput>,
) -> napi::Result<serde_json::Value> {
    let profiles: Vec<voice::CharacterVoiceProfile> = profiles
        .into_iter()
        .map(to_character_voice_profile)
        .collect();
    let result = voice::compare_voices_checked(&profiles).map_err(super::reason)?;
    super::encode_result(result)
}

/// Measure how far a `current` (revised) voice profile has drifted from a
/// `baseline` (established) profile across the features both expose.
#[napi(
    js_name = "measureVoiceDrift",
    ts_return_type = "import('./standalone.generated').MeasureVoiceDriftResult"
)]
pub fn measure_voice_drift(
    baseline: CharacterVoiceProfileInput,
    current: CharacterVoiceProfileInput,
) -> napi::Result<serde_json::Value> {
    let baseline = to_character_voice_profile(baseline);
    let current = to_character_voice_profile(current);
    let result = voice::measure_voice_drift_checked(&baseline, &current).map_err(super::reason)?;
    super::encode_result(result)
}

/// Convenience: build both profiles from baseline-text and current-text
/// lines (same speaker/narrator label) and measure drift between them.
#[napi(
    js_name = "measureVoiceDriftFromText",
    ts_return_type = "import('./standalone.generated').MeasureVoiceDriftFromTextResult"
)]
pub fn measure_voice_drift_from_text(
    label: String,
    baseline_lines: Vec<String>,
    current_lines: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let result =
        voice::measure_voice_drift_from_text_checked(&label, &baseline_lines, &current_lines)
            .map_err(super::reason)?;
    super::encode_result(result)
}

/// Build voice profiles for every character and flag same-sounding pairs.
#[napi(
    js_name = "analyzeCharacterVoices",
    ts_return_type = "import('./standalone.generated').AnalyzeCharacterVoicesResult"
)]
pub fn analyze_character_voices(
    character_names: Vec<String>,
    character_dialogue_lines: Vec<Vec<String>>,
) -> napi::Result<serde_json::Value> {
    if character_names.len() != character_dialogue_lines.len() {
        return Err(napi::Error::from_reason(
            "character dialogue arrays must have equal lengths",
        ));
    }
    let dialogue_data: Vec<(String, Vec<String>)> = character_names
        .into_iter()
        .zip(character_dialogue_lines)
        .collect();
    let result = voice::analyze_character_voices_checked(&dialogue_data).map_err(super::reason)?;
    super::encode_result(result)
}

#[napi(
    js_name = "analyzeSubtextAnnotations",
    ts_args_type = "manuscript: import('./manuscript').Manuscript, annotations: import('./standalone.generated').AuthorSubtextAnnotation[]",
    ts_return_type = "import('./standalone.generated').AnalyzeSubtextAnnotationsResult"
)]
pub fn analyze_subtext_annotations(
    manuscript: serde_json::Value,
    annotations: serde_json::Value,
) -> napi::Result<serde_json::Value> {
    let manuscript = super::decode_input(manuscript, "invalid manuscript object")?;
    let annotations: Vec<subtext::AuthorSubtextAnnotation> =
        super::decode_input(annotations, "invalid author subtext annotations")?;
    let result =
        subtext::analyze_subtext_annotations(&manuscript, &annotations).map_err(super::reason)?;
    super::encode_result(result)
}

// ---------------------------------------------------------------------------
// subtext.rs
// ---------------------------------------------------------------------------

/// Analyze subtext across scenes (currently a reserved-for-reasoner stub:
/// always returns empty instances with zero density).
#[napi(
    js_name = "analyzeSubtext",
    ts_return_type = "import('./standalone.generated').AnalyzeSubtextResult"
)]
pub fn analyze_subtext(
    scenes: Vec<String>,
    dialogue_segment_scenes: Vec<f64>,
    dialogue_segment_speakers: Vec<String>,
    dialogue_segment_quotes: Vec<String>,
    dialogue_segment_contexts: Vec<String>,
    emotional_state_characters: Vec<String>,
    emotional_state_scenes: Vec<f64>,
    emotional_state_emotions: Vec<String>,
    genre: String,
) -> napi::Result<serde_json::Value> {
    super::matching_lengths(&[
        dialogue_segment_scenes.len(),
        dialogue_segment_speakers.len(),
        dialogue_segment_quotes.len(),
        dialogue_segment_contexts.len(),
    ])?;
    super::matching_lengths(&[
        emotional_state_characters.len(),
        emotional_state_scenes.len(),
        emotional_state_emotions.len(),
    ])?;
    let scene_refs: Vec<&str> = scenes.iter().map(String::as_str).collect();
    let dialogue_segments: Vec<(usize, String, String, String)> = dialogue_segment_scenes
        .into_iter()
        .zip(dialogue_segment_speakers)
        .zip(dialogue_segment_quotes)
        .zip(dialogue_segment_contexts)
        .map(|(((scene, speaker), quote), context)| {
            Ok((super::unsigned_index(scene)?, speaker, quote, context))
        })
        .collect::<napi::Result<_>>()?;
    let character_emotional_states: Vec<(String, usize, String)> = emotional_state_characters
        .into_iter()
        .zip(emotional_state_scenes)
        .zip(emotional_state_emotions)
        .map(|((character, scene), emotion)| {
            Ok((character, super::unsigned_index(scene)?, emotion))
        })
        .collect::<napi::Result<_>>()?;
    let result = subtext::analyze_subtext(
        &scene_refs,
        &dialogue_segments,
        &character_emotional_states,
        &genre,
    );
    super::encode_result(result)
}
