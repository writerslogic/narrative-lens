//! `continuity` category napi bindings. See `napi_bindings/mod.rs` for the pattern.

use std::collections::HashMap;

use napi_derive::napi;

use crate::continuity::{
    anachronism, causal_engine, coreference, entity, epistemic_engine, gap_analysis,
    mental_model, timeline, world_state,
};

/// Analyze a scene's tokens for anachronistic words given a target historical year.
/// Returns empty if target_year is 0 (no target set).
#[napi(js_name = "analyzeScene")]
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
#[napi(js_name = "analyzeCausality")]
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
#[napi(js_name = "resolveEntities")]
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
#[napi(js_name = "resolveEntitiesWithCorrections")]
pub fn resolve_entities_with_corrections(
    mentions: Vec<(String, Vec<f64>)>,
    kind: String,
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
#[napi(js_name = "buildMentalModel")]
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
#[napi(js_name = "analyzeGaps")]
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
#[napi(js_name = "trackEpistemics")]
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
#[napi(js_name = "buildTimeline")]
pub fn build_timeline(documents: Vec<(String, String)>) -> napi::Result<serde_json::Value> {
    let result = timeline::build_timeline(&documents);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Analyze scenes and characters to produce a full world state analysis
/// (facts, continuity violations, character knowledge, information flow).
#[napi(js_name = "trackWorldState")]
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
#[napi(js_name = "resolveCoreferences")]
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
#[napi(js_name = "buildCharacterGenderMap")]
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
