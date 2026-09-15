//! `continuity` category napi bindings. See `napi_bindings/mod.rs` for the pattern.

use std::collections::HashMap;

use napi_derive::napi;

use crate::continuity::{
    anachronism, causal_engine, coreference, entity, epistemic_engine, gap_analysis,
    mental_model, timeline, world_state,
};

/// Analyze a scene's tokens for anachronistic words given a target historical year.
/// Returns empty if target_year is 0 (no target set).
#[napi(
    js_name = "analyzeScene",
    ts_return_type = "Array<{ word: string; introducedYear: number; targetYear: number; sceneId: number }>"
)]
pub fn analyze_scene(
    tokens: Vec<String>,
    target_year: i32,
    scene_id: u32,
) -> napi::Result<serde_json::Value> {
    let result = anachronism::analyze_scene(&tokens, target_year, scene_id as usize);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Build a causal graph over scenes and analyze plot coherence, counterfactuals,
/// and plot holes.
#[napi(
    js_name = "analyzeCausality",
    ts_return_type = "{ events: Array<{ id: number; scene: number; description: string; agent: string | null; eventType: 'Action' | 'Reaction' | 'Consequence' | 'Discovery' | 'Decision' | 'External' }>; links: Array<{ cause: number; effect: number; strength: number; linkType: 'Necessary' | 'Enabling' | 'Triggering' | 'Coincidental' }>; counterfactuals: Array<{ removedEvent: number; question: string; consequences: Array<string>; plotNecessity: number }>; plotHoles: Array<{ scene: number; description: string; problem: string; causalGap: string; suggestion: string }>; causalDensity: number; longestChain: number; plotCoherence: number; structuralInsights: Array<string> }"
)]
pub fn analyze_causality(
    scenes: Vec<String>,
    characters: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let scenes: Vec<&str> = scenes.iter().map(String::as_str).collect();
    let characters: Vec<&str> = characters.iter().map(String::as_str).collect();
    let result = causal_engine::analyze_causality(&scenes, &characters);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

fn parse_entity_type(kind: &str) -> napi::Result<entity::EntityType> {
    match kind {
        "character" => Ok(entity::EntityType::Character),
        "setting" => Ok(entity::EntityType::Setting),
        "symbol" => Ok(entity::EntityType::Symbol),
        "other" => Ok(entity::EntityType::Other),
        _ => Err(napi::Error::from_reason(format!(
            "unknown entity kind: {kind}"
        ))),
    }
}

/// Resolve name mentions (surface, context-embedding pairs) into canonical
/// entities of the given `kind` ("character", "setting", "symbol", or "other").
#[napi(
    js_name = "resolveEntities",
    ts_return_type = "Array<{ id: string; canonical: string; kind: 'character' | 'setting' | 'symbol' | 'other'; aliases: Array<string>; mentionCount: number; confidence: number; provenance: 'ai_derived' | 'user_edited' }>"
)]
pub fn resolve_entities(
    mentions: Vec<(String, Vec<f64>)>,
    kind: String,
) -> napi::Result<serde_json::Value> {
    let kind = parse_entity_type(&kind)?;
    let mentions: Vec<entity::Mention> = mentions
        .into_iter()
        .map(|(surface, context)| entity::Mention::new(surface, context))
        .collect();
    let result = entity::resolve(&mentions, kind);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Resolve mentions, then apply the writer's persisted corrections (each a JSON
/// object shaped like `EntityCorrection`, e.g. `{"surface": "...", "action":
/// {"kind": "assign", "canonical": "..."}}` or `{"kind": "reject"}`) over the
/// automatic result.
#[napi(
    js_name = "resolveEntitiesWithCorrections",
    ts_return_type = "Array<{ id: string; canonical: string; kind: 'character' | 'setting' | 'symbol' | 'other'; aliases: Array<string>; mentionCount: number; confidence: number; provenance: 'ai_derived' | 'user_edited' }>"
)]
pub fn resolve_entities_with_corrections(
    mentions: Vec<(String, Vec<f64>)>,
    kind: String,
    #[napi(ts_arg_type = "Array<{ surface: string; action: { kind: 'assign'; canonical: string } | { kind: 'reject' } }>")]
    corrections: Vec<serde_json::Value>,
) -> napi::Result<serde_json::Value> {
    let kind = parse_entity_type(&kind)?;
    let mentions: Vec<entity::Mention> = mentions
        .into_iter()
        .map(|(surface, context)| entity::Mention::new(surface, context))
        .collect();
    let corrections: Vec<entity::EntityCorrection> = corrections
        .into_iter()
        .map(serde_json::from_value)
        .collect::<Result<_, _>>()
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
    let result = entity::resolve_with_corrections(&mentions, kind, &corrections);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Build the reader's evolving mental model (belief trajectory, dramatic
/// ironies, revelations/twists) across scenes.
#[napi(
    js_name = "buildMentalModel",
    ts_return_type = "{ trajectory: Array<{ afterScene: number; beliefs: Array<{ id: number; subject: string; proposition: string; confidence: number; establishedAt: number; lastReinforced: number; matchesTruth: boolean | null; category: 'Identity' | 'Motivation' | 'Allegiance' | 'EventCause' | 'Prediction' | 'WorldRule' | 'EmotionalState' | 'Capability'; salience: number; sourceReliability: number; inferenceChain: Array<number> }>; invalidated: Array<{ id: number; subject: string; proposition: string; confidence: number; establishedAt: number; lastReinforced: number; matchesTruth: boolean | null; category: 'Identity' | 'Motivation' | 'Allegiance' | 'EventCause' | 'Prediction' | 'WorldRule' | 'EmotionalState' | 'Capability'; salience: number; sourceReliability: number; inferenceChain: Array<number> }>; introduced: Array<{ id: number; subject: string; proposition: string; confidence: number; establishedAt: number; lastReinforced: number; matchesTruth: boolean | null; category: 'Identity' | 'Motivation' | 'Allegiance' | 'EventCause' | 'Prediction' | 'WorldRule' | 'EmotionalState' | 'Capability'; salience: number; sourceReliability: number; inferenceChain: Array<number> }>; beliefGap: number; gapDelta: number; activeIronies: Array<{ knowledge: string; ignorantCharacter: string; establishedAt: number; resolvedAt: number | null; tension: number }>; archetypeDivergence: number }>; revelations: Array<{ scene: number; magnitude: number; affectedBeliefs: Array<string>; eventType: 'Revelation' | 'Twist' | 'Confirmation' | 'Complication' | 'DramaticIronyEstablished' | 'DramaticIronyResolved'; preparationScore: number }>; twists: Array<{ scene: number; magnitude: number; affectedBeliefs: Array<string>; eventType: 'Revelation' | 'Twist' | 'Confirmation' | 'Complication' | 'DramaticIronyEstablished' | 'DramaticIronyResolved'; preparationScore: number }>; stasisZones: Array<[number, number]>; gapCurve: Array<number>; managementScore: number; dramaticIronies: Array<{ knowledge: string; ignorantCharacter: string; establishedAt: number; resolvedAt: number | null; tension: number }>; archetypeDivergencePeaks: Array<[number, number, string]>; narratorReliability: { overallScore: number; unreliabilityEvidence: Array<[number, string]>; intentional: boolean }; inferenceChains: Array<{ conclusionId: number; premiseIds: Array<number>; strength: number; description: string }>; readerExperience: Array<{ scene: number; noteType: string; description: string; craftImplication: string }>; engagementDiagnosis: string }"
)]
pub fn build_mental_model(
    scenes: Vec<String>,
    characters: Vec<String>,
    pov_characters: Vec<String>,
    theme_keywords: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let scenes: Vec<&str> = scenes.iter().map(String::as_str).collect();
    let characters: Vec<&str> = characters.iter().map(String::as_str).collect();
    let pov_characters: Vec<&str> = pov_characters.iter().map(String::as_str).collect();
    let theme_keywords: Vec<&str> = theme_keywords.iter().map(String::as_str).collect();
    let result = mental_model::build_mental_model(&scenes, &characters, &pov_characters, &theme_keywords);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Analyze narrative gaps based on signals from other analysis modules: broken
/// promises `(description, promiseScene)`, consequenceless decisions
/// `(character, scene)`, pacing stall zones `(startScene, endScene)`, and
/// unresolved ironies `(description, scene)`.
#[napi(
    js_name = "analyzeGaps",
    ts_return_type = "{ gaps: Array<{ gapType: 'MissingScene' | 'MissingBeat' | 'MissingEmotionalTransition' | 'MissingInformationReveal' | 'UnderdevelopedRelationship' | 'UnresolvedSubplot' | 'UnmotivatedAction'; description: string; location: { afterScene: number; beforeScene: number | null; affectedCharacters: Array<string> }; suggestion: string }>; totalGapCount: number }"
)]
pub fn analyze_gaps(
    broken_promises: Vec<(String, u32)>,
    consequenceless_decisions: Vec<(String, u32)>,
    stall_zones: Vec<(u32, u32)>,
    unresolved_ironies: Vec<(String, u32)>,
    total_scenes: u32,
    climax_scene: u32,
) -> napi::Result<serde_json::Value> {
    let broken_promises: Vec<(String, usize)> = broken_promises
        .into_iter()
        .map(|(d, s)| (d, s as usize))
        .collect();
    let consequenceless_decisions: Vec<(String, usize)> = consequenceless_decisions
        .into_iter()
        .map(|(c, s)| (c, s as usize))
        .collect();
    let stall_zones: Vec<(usize, usize)> = stall_zones
        .into_iter()
        .map(|(a, b)| (a as usize, b as usize))
        .collect();
    let unresolved_ironies: Vec<(String, usize)> = unresolved_ironies
        .into_iter()
        .map(|(d, s)| (d, s as usize))
        .collect();
    let result = gap_analysis::analyze_gaps(
        &broken_promises,
        &consequenceless_decisions,
        &stall_zones,
        &unresolved_ironies,
        total_scenes as usize,
        climax_scene as usize,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Track per-character and reader epistemic states across scenes, computing
/// asymmetries, revelations, and the suspense curve.
#[napi(
    js_name = "trackEpistemics",
    ts_return_type = "{ facts: Array<{ id: number; content: string; establishedScene: number; importance: number }>; characterStates: Record<string, { character: string; knows: Array<number>; believesFalse: Array<number>; uncertainAbout: Array<number> }>; asymmetries: Array<{ scene: number; readerAdvantage: number; characterAdvantage: number; totalHiddenBits: number; dramaticIronyIntensity: number }>; revelations: Array<{ scene: number; factId: number; revealedTo: string; bitsRevealed: number; dramaticEffect: string }>; suspenseCurve: Array<{ scene: number; suspenseLevel: number; source: string }>; peakSuspenseScene: number; informationDensity: number; manipulationScore: number; epistemicInsights: Array<string> }"
)]
pub fn track_epistemics(
    scenes: Vec<String>,
    characters: Vec<String>,
    pov_characters: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let scenes: Vec<&str> = scenes.iter().map(String::as_str).collect();
    let characters: Vec<&str> = characters.iter().map(String::as_str).collect();
    let pov_characters: Vec<&str> = pov_characters.iter().map(String::as_str).collect();
    let result = epistemic_engine::track_epistemics(&scenes, &characters, &pov_characters);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Build the timeline projection: the given `(documentId, text)` documents, in
/// the order supplied, each tagged with its first detected time hint.
#[napi(
    js_name = "buildTimeline",
    ts_return_type = "Array<{ documentId: string; order: number; timeHint: string | null; chronologyKey: string | null; lane: string | null; label: string | null; notes: string | null }>"
)]
pub fn build_timeline(documents: Vec<(String, String)>) -> napi::Result<serde_json::Value> {
    let result = timeline::build_timeline(&documents);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Analyze scenes and characters to produce a full world state analysis
/// (facts, continuity violations, character knowledge, information flow).
#[napi(
    js_name = "trackWorldState",
    ts_return_type = "{ snapshots: Array<{ afterScene: number; facts: Array<{ CharacterAt: { character: string; location: string } } | { CharacterPossesses: { character: string; object: string } } | { CharacterKnows: { character: string; information: string } } | { CharacterWitnessed: { character: string; event: string } } | { CoLocated: { characterA: string; characterB: string } } | { CommunicationVector: { from: string; to: string; content: string } } | { ObjectAt: { object: string; location: string } } | { PhysicalConstraint: { character: string; constraint: string } } | { TemporalOrdering: { before: string; after: string } }>; newFacts: Array<{ CharacterAt: { character: string; location: string } } | { CharacterPossesses: { character: string; object: string } } | { CharacterKnows: { character: string; information: string } } | { CharacterWitnessed: { character: string; event: string } } | { CoLocated: { characterA: string; characterB: string } } | { CommunicationVector: { from: string; to: string; content: string } } | { ObjectAt: { object: string; location: string } } | { PhysicalConstraint: { character: string; constraint: string } } | { TemporalOrdering: { before: string; after: string } }>; invalidatedFacts: Array<{ CharacterAt: { character: string; location: string } } | { CharacterPossesses: { character: string; object: string } } | { CharacterKnows: { character: string; information: string } } | { CharacterWitnessed: { character: string; event: string } } | { CoLocated: { characterA: string; characterB: string } } | { CommunicationVector: { from: string; to: string; content: string } } | { ObjectAt: { object: string; location: string } } | { PhysicalConstraint: { character: string; constraint: string } } | { TemporalOrdering: { before: string; after: string } }> }>; violations: Array<{ scene: number; violationType: 'ImpossibleKnowledge' | 'ImpossibleLocation' | 'ObjectContinuity' | 'TemporalParadox' | 'UnestablishedRelationship' | 'PhysicalImpossibility'; description: string; proof: string; severity: number; fixSuggestion: string }>; characterKnowledge: Array<{ character: string; knows: Array<string>; witnessed: Array<string>; toldBy: Array<[string, string]>; locationsVisited: Array<[number, string]>; currentLocation: string | null }>; locationTimeline: Array<[number, string, Array<string>]>; informationFlow: Array<[number, string, string, string]>; continuityScore: number; worldComplexity: number }"
)]
pub fn track_world_state(
    scenes: Vec<String>,
    characters: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let scenes: Vec<&str> = scenes.iter().map(String::as_str).collect();
    let characters: Vec<&str> = characters.iter().map(String::as_str).collect();
    let result = world_state::track_world_state(&scenes, &characters);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Resolve pronoun references to character names within a scene. `gender_map`
/// (character name -> "male"/"female"/"unknown") is inferred from `scene_text`
/// alone when omitted.
#[napi(
    js_name = "resolveCoreferences",
    ts_return_type = "{ mentions: Array<{ text: string; resolvedTo: string | null; start: number; isPronoun: boolean; confidence: number }>; characterMentions: Record<string, number>; pronounResolutionRate: number }"
)]
pub fn resolve_coreferences(
    scene_text: String,
    known_characters: Vec<String>,
    gender_map: Option<HashMap<String, String>>,
) -> napi::Result<serde_json::Value> {
    let known_characters: Vec<&str> = known_characters.iter().map(String::as_str).collect();
    let resolver = coreference::CoreferenceResolver::new();
    let result = resolver.resolve(&scene_text, &known_characters, gender_map.as_ref());
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Scan multiple scenes to infer a gender ("male"/"female"/"unknown") for each
/// named character, from pronoun proximity.
#[napi(js_name = "buildCharacterGenderMap", ts_return_type = "Record<string, string>")]
pub fn build_character_gender_map(
    scenes: Vec<String>,
    characters: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let scenes: Vec<&str> = scenes.iter().map(String::as_str).collect();
    let characters: Vec<&str> = characters.iter().map(String::as_str).collect();
    let resolver = coreference::CoreferenceResolver::new();
    let result = resolver.build_character_gender_map(&scenes, &characters);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}
