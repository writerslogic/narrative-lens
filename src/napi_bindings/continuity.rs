//! `continuity` category napi bindings. See `napi_bindings/mod.rs` for the pattern.

use std::collections::HashMap;

use napi_derive::napi;

use crate::continuity::{
    anachronism, author_knowledge, causal_engine, coreference, entity, epistemic_engine,
    gap_analysis, mental_model, ner, timeline, world_state,
};

#[napi(
    js_name = "analyzeAuthorKnowledge",
    ts_args_type = "manuscript: import('./manuscript').Manuscript, propositions: import('./standalone.generated').AuthorProposition[], events: import('./standalone.generated').AuthorKnowledgeEvent[]",
    ts_return_type = "import('./standalone.generated').AnalyzeAuthorKnowledgeResult"
)]
pub fn analyze_author_knowledge(
    manuscript: serde_json::Value,
    propositions: serde_json::Value,
    events: serde_json::Value,
) -> napi::Result<serde_json::Value> {
    let manuscript = serde_json::from_value(manuscript)
        .map_err(|_| napi::Error::from_reason("invalid manuscript object"))?;
    let propositions: Vec<author_knowledge::AuthorProposition> =
        serde_json::from_value(propositions)
            .map_err(|_| napi::Error::from_reason("invalid author propositions"))?;
    let events: Vec<author_knowledge::AuthorKnowledgeEvent> = serde_json::from_value(events)
        .map_err(|_| napi::Error::from_reason("invalid author knowledge events"))?;
    let ledger = author_knowledge::analyze_author_knowledge(&manuscript, &propositions, &events)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    serde_json::to_value(ledger).map_err(|error| napi::Error::from_reason(error.to_string()))
}

#[napi(
    js_name = "trackWorldStateCurrent",
    ts_args_type = "scenes: string[], characters: string[], options?: Partial<import('./standalone.generated').WorldStateOptions>",
    ts_return_type = "import('./standalone.generated').TrackWorldStateCurrentResult"
)]
pub fn track_world_state_current(
    scenes: Vec<String>,
    characters: Vec<String>,
    options: Option<serde_json::Value>,
) -> napi::Result<serde_json::Value> {
    let options: world_state::WorldStateOptions =
        serde_json::from_value(options.unwrap_or_else(|| serde_json::json!({})))
            .map_err(|_| napi::Error::from_reason("invalid world state options"))?;
    let scenes: Vec<&str> = scenes.iter().map(String::as_str).collect();
    let characters: Vec<&str> = characters.iter().map(String::as_str).collect();
    let result = world_state::track_world_state_current(&scenes, &characters, &options)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    serde_json::to_value(result).map_err(|error| napi::Error::from_reason(error.to_string()))
}

/// Analyze a scene's tokens for anachronistic words given a target historical year.
/// Returns empty if target_year is 0 (no target set).
#[napi(
    js_name = "analyzeScene",
    ts_return_type = "import('./standalone.generated').AnalyzeSceneResult"
)]
pub fn analyze_scene(
    tokens: Vec<String>,
    target_year: f64,
    scene_id: f64,
) -> napi::Result<serde_json::Value> {
    let result = anachronism::analyze_scene_checked(
        &tokens,
        super::signed_integer(target_year)?,
        super::unsigned_index(scene_id)?,
    )
    .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    super::encode_result(result.items)
}

#[napi(
    js_name = "analyzeSceneChecked",
    ts_return_type = "import('./standalone.generated').AnalyzeSceneCheckedResult"
)]
pub fn analyze_scene_checked(
    tokens: Vec<String>,
    target_year: f64,
    scene_id: f64,
) -> napi::Result<serde_json::Value> {
    let result = anachronism::analyze_scene_checked(
        &tokens,
        super::signed_integer(target_year)?,
        super::unsigned_index(scene_id)?,
    )
    .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    super::encode_result(result)
}

#[napi(
    js_name = "analyzeHistoricalCatalogue",
    ts_args_type = "manuscript: import('./manuscript').Manuscript, catalogue: import('./standalone.generated').HistoricalEntry[], usages: import('./standalone.generated').HistoricalUsage[], targetYear: number, toleranceYears: number",
    ts_return_type = "import('./standalone.generated').AnalyzeHistoricalCatalogueResult"
)]
pub fn analyze_historical_catalogue(
    manuscript: serde_json::Value,
    catalogue: serde_json::Value,
    usages: serde_json::Value,
    target_year: f64,
    tolerance_years: f64,
) -> napi::Result<serde_json::Value> {
    let manuscript = super::decode_input(manuscript, "invalid manuscript object")?;
    let catalogue: Vec<anachronism::HistoricalEntry> =
        super::decode_input(catalogue, "invalid historical catalogue")?;
    let usages: Vec<anachronism::HistoricalUsage> =
        super::decode_input(usages, "invalid historical usages")?;
    let result = anachronism::analyze_historical_catalogue(
        &manuscript,
        &catalogue,
        &usages,
        super::signed_integer(target_year)?,
        super::unsigned_index(tolerance_years)? as u32,
    )
    .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    super::encode_result(result)
}

#[napi(
    js_name = "extractNamedEntities",
    ts_return_type = "import('./standalone.generated').ExtractNamedEntitiesResult"
)]
pub fn extract_named_entities(texts: Vec<String>) -> napi::Result<serde_json::Value> {
    let texts: Vec<_> = texts.iter().map(String::as_str).collect();
    let result = ner::ner_extract_entities_batch_checked(&texts)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    super::encode_result(result)
}

#[napi(
    js_name = "extractPersons",
    ts_return_type = "import('./standalone.generated').ExtractPersonsResult"
)]
pub fn extract_persons(scenes: Vec<String>) -> napi::Result<serde_json::Value> {
    let result = ner::ner_extract_persons_checked(&scenes)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    super::encode_result(result)
}

/// Build lexical association candidates; causal mechanisms require author input.
#[napi(
    js_name = "analyzeCausality",
    ts_return_type = "import('./standalone.generated').AnalyzeCausalityResult"
)]
pub fn analyze_causality(
    scenes: Vec<String>,
    characters: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let scenes: Vec<&str> = scenes.iter().map(String::as_str).collect();
    let characters: Vec<&str> = characters.iter().map(String::as_str).collect();
    let result = causal_engine::analyze_causality_checked(&scenes, &characters)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

#[napi(
    js_name = "analyzeAuthorCausality",
    ts_args_type = "manuscript: import('./manuscript').Manuscript, nodes: import('./standalone.generated').AuthorCausalNode[], interventions: import('./standalone.generated').AuthorIntervention[]",
    ts_return_type = "import('./standalone.generated').AnalyzeAuthorCausalityResult"
)]
pub fn analyze_author_causality(
    manuscript: serde_json::Value,
    nodes: serde_json::Value,
    interventions: serde_json::Value,
) -> napi::Result<serde_json::Value> {
    let manuscript = super::decode_input(manuscript, "invalid manuscript object")?;
    let nodes: Vec<causal_engine::AuthorCausalNode> =
        super::decode_input(nodes, "invalid author causal nodes")?;
    let interventions: Vec<causal_engine::AuthorIntervention> =
        super::decode_input(interventions, "invalid author interventions")?;
    let result = causal_engine::analyze_author_causality(&manuscript, &nodes, &interventions)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    super::encode_result(result)
}

#[napi(
    js_name = "buildAuthorMentalModel",
    ts_args_type = "manuscript: import('./manuscript').Manuscript, propositions: import('./standalone.generated').AuthorProposition[], events: import('./standalone.generated').AuthorKnowledgeEvent[]",
    ts_return_type = "import('./standalone.generated').BuildAuthorMentalModelResult"
)]
pub fn build_author_mental_model(
    manuscript: serde_json::Value,
    propositions: serde_json::Value,
    events: serde_json::Value,
) -> napi::Result<serde_json::Value> {
    let manuscript = super::decode_input(manuscript, "invalid manuscript object")?;
    let propositions: Vec<author_knowledge::AuthorProposition> =
        super::decode_input(propositions, "invalid author propositions")?;
    let events: Vec<author_knowledge::AuthorKnowledgeEvent> =
        super::decode_input(events, "invalid author knowledge events")?;
    let result = mental_model::build_author_mental_model(&manuscript, &propositions, &events)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    super::encode_result(result)
}

fn parse_entity_type(kind: &str) -> napi::Result<entity::EntityType> {
    match kind {
        "character" => Ok(entity::EntityType::Character),
        "setting" => Ok(entity::EntityType::Setting),
        "symbol" => Ok(entity::EntityType::Symbol),
        "other" => Ok(entity::EntityType::Other),
        _ => Err(napi::Error::from_reason("unknown entity kind")),
    }
}

/// Resolve name mentions (surface, context-embedding pairs) into canonical
/// entities of the given `kind` ("character", "setting", "symbol", or "other").
#[napi(
    js_name = "resolveEntities",
    ts_return_type = "import('./standalone.generated').ResolveEntitiesResult"
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
    let result = entity::resolve_checked(&mentions, kind)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Resolve mentions, then apply the writer's persisted corrections (each a JSON
/// object shaped like `EntityCorrection`, e.g. `{"surface": "...", "action":
/// {"kind": "assign", "canonical": "..."}}` or `{"kind": "reject"}`) over the
/// automatic result.
#[napi(
    js_name = "resolveEntitiesWithCorrections",
    ts_return_type = "import('./standalone.generated').ResolveEntitiesWithCorrectionsResult"
)]
pub fn resolve_entities_with_corrections(
    mentions: Vec<(String, Vec<f64>)>,
    kind: String,
    #[napi(
        ts_arg_type = "Array<{ surface: string; action: { kind: 'assign'; canonical: string } | { kind: 'reject' } }>"
    )]
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
        .map_err(|_| napi::Error::from_reason("invalid entity corrections"))?;
    let result = entity::resolve_with_corrections_checked(&mentions, kind, &corrections)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Build the reader's evolving mental model (belief trajectory, dramatic
/// ironies, revelations/twists) across scenes.
#[napi(
    js_name = "buildMentalModel",
    ts_return_type = "import('./standalone.generated').BuildMentalModelResult"
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
    let result =
        mental_model::build_mental_model(&scenes, &characters, &pov_characters, &theme_keywords);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Analyze narrative gaps based on signals from other analysis modules: broken
/// promises `(description, promiseScene)`, consequenceless decisions
/// `(character, scene)`, pacing stall zones `(startScene, endScene)`, and
/// unresolved ironies `(description, scene)`.
#[napi(
    js_name = "analyzeGaps",
    ts_return_type = "import('./standalone.generated').AnalyzeGapsResult"
)]
pub fn analyze_gaps(
    broken_promises: Vec<(String, f64)>,
    consequenceless_decisions: Vec<(String, f64)>,
    stall_zones: Vec<(f64, f64)>,
    unresolved_ironies: Vec<(String, f64)>,
    total_scenes: f64,
    climax_scene: f64,
) -> napi::Result<serde_json::Value> {
    let broken_promises: Vec<(String, usize)> = broken_promises
        .into_iter()
        .map(|(d, s)| Ok((d, super::unsigned_index(s)?)))
        .collect::<napi::Result<_>>()?;
    let consequenceless_decisions: Vec<(String, usize)> = consequenceless_decisions
        .into_iter()
        .map(|(c, s)| Ok((c, super::unsigned_index(s)?)))
        .collect::<napi::Result<_>>()?;
    let stall_zones: Vec<(usize, usize)> = stall_zones
        .into_iter()
        .map(|(a, b)| Ok((super::unsigned_index(a)?, super::unsigned_index(b)?)))
        .collect::<napi::Result<_>>()?;
    let unresolved_ironies: Vec<(String, usize)> = unresolved_ironies
        .into_iter()
        .map(|(d, s)| Ok((d, super::unsigned_index(s)?)))
        .collect::<napi::Result<_>>()?;
    let result = gap_analysis::analyze_gaps_checked(
        &broken_promises,
        &consequenceless_decisions,
        &stall_zones,
        &unresolved_ironies,
        super::unsigned_index(total_scenes)?,
        super::unsigned_index(climax_scene)?,
    )
    .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Track per-character and reader epistemic states across scenes, computing
/// asymmetries, revelations, and the suspense curve.
#[napi(
    js_name = "trackEpistemics",
    ts_return_type = "import('./standalone.generated').TrackEpistemicsResult"
)]
pub fn track_epistemics(
    scenes: Vec<String>,
    characters: Vec<String>,
    pov_characters: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let scenes: Vec<&str> = scenes.iter().map(String::as_str).collect();
    let characters: Vec<&str> = characters.iter().map(String::as_str).collect();
    let pov_characters: Vec<&str> = pov_characters.iter().map(String::as_str).collect();
    let result = epistemic_engine::track_epistemics_checked(&scenes, &characters, &pov_characters)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

#[napi(
    js_name = "analyzeAuthorEpistemics",
    ts_args_type = "manuscript: import('./manuscript').Manuscript, propositions: import('./standalone.generated').AuthorProposition[], events: import('./standalone.generated').AuthorKnowledgeEvent[], queries: import('./standalone.generated').AuthorEpistemicQuery[]",
    ts_return_type = "import('./standalone.generated').AnalyzeAuthorEpistemicsResult"
)]
pub fn analyze_author_epistemics(
    manuscript: serde_json::Value,
    propositions: serde_json::Value,
    events: serde_json::Value,
    queries: serde_json::Value,
) -> napi::Result<serde_json::Value> {
    let manuscript = super::decode_input(manuscript, "invalid manuscript object")?;
    let propositions: Vec<author_knowledge::AuthorProposition> =
        super::decode_input(propositions, "invalid author propositions")?;
    let events: Vec<author_knowledge::AuthorKnowledgeEvent> =
        super::decode_input(events, "invalid author knowledge events")?;
    let queries: Vec<epistemic_engine::AuthorEpistemicQuery> =
        super::decode_input(queries, "invalid author epistemic queries")?;
    let result =
        epistemic_engine::analyze_author_epistemics(&manuscript, &propositions, &events, &queries)
            .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    super::encode_result(result)
}

/// Build the timeline projection: the given `(documentId, text)` documents, in
/// the order supplied, each tagged with its first detected time hint.
#[napi(
    js_name = "buildTimeline",
    ts_args_type = "documents: Array<[string, string]>, overrides?: import('./standalone.generated').TimelineOverride[]",
    ts_return_type = "import('./standalone.generated').BuildTimelineResult"
)]
pub fn build_timeline(
    documents: Vec<(String, String)>,
    overrides: Option<serde_json::Value>,
) -> napi::Result<serde_json::Value> {
    let mut result = timeline::build_timeline_checked(&documents)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    if let Some(overrides) = overrides {
        let overrides: Vec<timeline::TimelineOverride> =
            super::decode_input(overrides, "invalid timeline overrides")?;
        timeline::apply_overrides_checked(&mut result, &overrides)
            .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    }
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Analyze scenes and characters to produce a full world state analysis
/// (facts, continuity violations, character knowledge, information flow).
#[napi(
    js_name = "trackWorldState",
    ts_return_type = "import('./standalone.generated').TrackWorldStateResult"
)]
pub fn track_world_state(
    scenes: Vec<String>,
    characters: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let scenes: Vec<&str> = scenes.iter().map(String::as_str).collect();
    let characters: Vec<&str> = characters.iter().map(String::as_str).collect();
    let result = world_state::track_world_state_checked(&scenes, &characters)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Resolve pronoun references to character names within a scene. `gender_map`
/// (character name -> "male"/"female"/"neutral"/"unknown") is inferred from
/// `scene_text` alone when omitted; they/them binds only to a character hinted
/// "neutral". Cast entries that prefix one another ("Hester", "Hester Prynne")
/// form one alias group, and `resolvedTo` names the entry the antecedent
/// literally matched. Narrative pronoun chains continue across quoted dialogue.
#[napi(
    js_name = "resolveCoreferences",
    ts_return_type = "import('./standalone.generated').ResolveCoreferencesResult"
)]
pub fn resolve_coreferences(
    scene_text: String,
    known_characters: Vec<String>,
    gender_map: Option<HashMap<String, String>>,
) -> napi::Result<serde_json::Value> {
    let known_characters: Vec<&str> = known_characters.iter().map(String::as_str).collect();
    let resolver = coreference::CoreferenceResolver::new();
    let result = resolver
        .resolve_checked(&scene_text, &known_characters, gender_map.as_ref())
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Scan multiple scenes to infer a gender ("male"/"female"/"unknown") for each
/// named character. A gendered honorific ("Mrs.", "Sir") or a listed given name
/// decides without textual evidence; otherwise pronoun proximity decides, and
/// characters with too few nearby pronouns stay "unknown". Never emits "neutral".
#[napi(
    js_name = "buildCharacterGenderMap",
    ts_return_type = "Record<string, string>"
)]
pub fn build_character_gender_map(
    scenes: Vec<String>,
    characters: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let scenes: Vec<&str> = scenes.iter().map(String::as_str).collect();
    let characters: Vec<&str> = characters.iter().map(String::as_str).collect();
    let resolver = coreference::CoreferenceResolver::new();
    let result = resolver
        .build_character_gender_map_checked(&scenes, &characters)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

#[napi(
    js_name = "resolveAuthorCoreference",
    ts_args_type = "manuscript: import('./manuscript').Manuscript, links: import('./standalone.generated').AuthorCoreferenceLink[]",
    ts_return_type = "import('./standalone.generated').ResolveAuthorCoreferenceResult"
)]
pub fn resolve_author_coreference(
    manuscript: serde_json::Value,
    links: serde_json::Value,
) -> napi::Result<serde_json::Value> {
    let manuscript = super::decode_input(manuscript, "invalid manuscript object")?;
    let links: Vec<coreference::AuthorCoreferenceLink> =
        super::decode_input(links, "invalid author coreference links")?;
    let result = coreference::resolve_author_coreference(&manuscript, &links)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    super::encode_result(result)
}
