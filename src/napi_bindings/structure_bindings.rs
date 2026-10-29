//! `structure` category napi bindings. See `napi_bindings/mod.rs` for the pattern.

use napi_derive::napi;

use crate::structure::{
    author_events, foreshadow, genre, genre_compliance, narrative_entropy, opening_analysis,
    promise_payoff, structure, thematic_argument, theme,
};

#[napi(
    js_name = "analyzeThematicAnnotations",
    ts_args_type = "manuscript: import('./manuscript').Manuscript, theses: import('./standalone.generated').AuthorThematicThesis[], annotations: import('./standalone.generated').AuthorThematicAnnotation[]",
    ts_return_type = "import('./standalone.generated').AnalyzeThematicAnnotationsResult"
)]
pub fn analyze_thematic_annotations(
    manuscript: serde_json::Value,
    theses: serde_json::Value,
    annotations: serde_json::Value,
) -> napi::Result<serde_json::Value> {
    let manuscript = super::decode_input(manuscript, "invalid manuscript object")?;
    let theses: Vec<thematic_argument::AuthorThematicThesis> =
        super::decode_input(theses, "invalid author thematic theses")?;
    let annotations: Vec<thematic_argument::AuthorThematicAnnotation> =
        super::decode_input(annotations, "invalid author thematic annotations")?;
    let result =
        thematic_argument::analyze_thematic_annotations(&manuscript, &theses, &annotations)
            .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    super::encode_result(result)
}

#[napi(
    js_name = "analyzePromiseEvents",
    ts_args_type = "manuscript: import('./manuscript').Manuscript, events: import('./standalone.generated').AuthorPromiseEvent[]",
    ts_return_type = "import('./standalone.generated').AnalyzePromiseEventsResult"
)]
pub fn analyze_promise_events(
    manuscript: serde_json::Value,
    events: serde_json::Value,
) -> napi::Result<serde_json::Value> {
    let manuscript = serde_json::from_value(manuscript)
        .map_err(|_| napi::Error::from_reason("invalid manuscript object"))?;
    let events: Vec<author_events::AuthorPromiseEvent> = serde_json::from_value(events)
        .map_err(|_| napi::Error::from_reason("invalid author promise events"))?;
    let ledger = author_events::analyze_promise_events(&manuscript, &events)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    serde_json::to_value(ledger).map_err(|error| napi::Error::from_reason(error.to_string()))
}

/// Look up a built-in narrative structure template by name.
#[napi(
    js_name = "getStructureTemplate",
    ts_return_type = "import('./standalone.generated').GetStructureTemplateResult"
)]
pub fn get_structure_template(name: String) -> napi::Result<serde_json::Value> {
    let result = structure::get_structure_template(&name)
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Score how well a set of scene tensions/purposes matches a structure template.
#[napi(
    js_name = "validateStructure",
    ts_return_type = "import('./standalone.generated').ValidateStructureResult"
)]
pub fn validate_structure(
    template_name: String,
    scene_tensions: Vec<f64>,
    scene_purposes: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let result = structure::validate_structure(&template_name, scene_tensions, scene_purposes)
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Suggest the best-fitting structure template for a set of scene tensions/purposes.
#[napi(js_name = "suggestBestTemplate", ts_return_type = "string")]
pub fn suggest_best_template(
    scene_tensions: Vec<f64>,
    scene_purposes: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let result = structure::suggest_best_template(scene_tensions, scene_purposes);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Analyze foreshadowing setup-payoff arcs across scenes.
#[napi(
    js_name = "analyzeForeshadowing",
    ts_return_type = "import('./standalone.generated').AnalyzeForeshadowingResult"
)]
pub fn analyze_foreshadowing(
    scenes_tokens: Vec<Vec<String>>,
    total_scenes: f64,
) -> napi::Result<serde_json::Value> {
    let result =
        foreshadow::analyze_foreshadowing(scenes_tokens, super::unsigned_index(total_scenes)?);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Build the promise-payoff ledger from analysis data.
#[napi(
    js_name = "buildPromisePayoffLedger",
    ts_return_type = "import('./standalone.generated').BuildPromisePayoffLedgerResult"
)]
pub fn build_promise_payoff_ledger(
    scenes: Vec<String>,
    scene_word_counts: Vec<f64>,
    genre: String,
    foreshadowing_items: Vec<(String, f64, bool)>,
    character_first_appearances: Vec<(String, f64)>,
    structure_template: String,
    total_scenes: f64,
) -> napi::Result<serde_json::Value> {
    let scene_refs: Vec<&str> = scenes.iter().map(|s| s.as_str()).collect();
    super::matching_lengths(&[scenes.len(), scene_word_counts.len()])?;
    let word_counts: Vec<usize> = scene_word_counts
        .into_iter()
        .map(super::unsigned_index)
        .collect::<napi::Result<_>>()?;
    let foreshadowing: Vec<(String, usize, bool)> = foreshadowing_items
        .into_iter()
        .map(|(desc, scene, resolved)| Ok((desc, super::unsigned_index(scene)?, resolved)))
        .collect::<napi::Result<_>>()?;
    let characters: Vec<(String, usize)> = character_first_appearances
        .into_iter()
        .map(|(name, scene)| Ok((name, super::unsigned_index(scene)?)))
        .collect::<napi::Result<_>>()?;
    let result = promise_payoff::build_promise_payoff_ledger(
        &scene_refs,
        &word_counts,
        &genre,
        &foreshadowing,
        &characters,
        &structure_template,
        super::unsigned_index(total_scenes)?,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Track the thematic argument across the manuscript.
#[napi(
    js_name = "trackThematicArgument",
    ts_return_type = "import('./standalone.generated').TrackThematicArgumentResult"
)]
pub fn track_thematic_argument(
    scenes: Vec<String>,
    themes: Vec<(String, f64)>,
    characters: Vec<String>,
    character_decisions: Vec<(String, f64, String)>,
    genre: String,
) -> napi::Result<serde_json::Value> {
    let scene_refs: Vec<&str> = scenes.iter().map(|s| s.as_str()).collect();
    let character_refs: Vec<&str> = characters.iter().map(|s| s.as_str()).collect();
    let decisions: Vec<(String, usize, String)> = character_decisions
        .into_iter()
        .map(|(name, scene, decision)| Ok((name, super::unsigned_index(scene)?, decision)))
        .collect::<napi::Result<_>>()?;
    let result = thematic_argument::track_thematic_argument(
        &scene_refs,
        &themes,
        &character_refs,
        &decisions,
        &genre,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Analyze the opening pages of a manuscript for hook strength and agent appeal.
#[napi(
    js_name = "analyzeOpening",
    ts_return_type = "import('./standalone.generated').AnalyzeOpeningResult"
)]
pub fn analyze_opening(
    text: String,
    genre: String,
    full_manuscript_voice_fingerprint: Option<Vec<f64>>,
) -> napi::Result<serde_json::Value> {
    let mut result = opening_analysis::analyze_opening_checked(&text, &genre, None)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    if full_manuscript_voice_fingerprint.is_some_and(|values| !values.is_empty()) {
        result.diagnostics.push("legacy_voice_vector_has_no_declared_feature_contract;use_analyzeOpeningChecked_with_typed_baseline".into());
    }
    super::encode_result(result)
}

#[napi(
    js_name = "analyzeOpeningChecked",
    ts_args_type = "text: string, genre: string, baseline?: import('./standalone.generated').CharacterVoiceProfile",
    ts_return_type = "import('./standalone.generated').AnalyzeOpeningResult"
)]
pub fn analyze_opening_checked(
    text: String,
    genre: String,
    baseline: Option<super::craft::CharacterVoiceProfileInput>,
) -> napi::Result<serde_json::Value> {
    let baseline = baseline.map(super::craft::to_character_voice_profile);
    let result = opening_analysis::analyze_opening_checked(&text, &genre, baseline.as_ref())
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    super::encode_result(result)
}

/// Check a manuscript's compliance with genre conventions.
#[napi(
    js_name = "checkGenreCompliance",
    ts_return_type = "import('./standalone.generated').CheckGenreComplianceResult"
)]
pub fn check_genre_compliance(
    genre: String,
    has_hea: bool,
    has_solution: bool,
    has_ticking_clock: bool,
    protagonist_survives: bool,
    character_growth: bool,
    worldbuilding_present: bool,
    word_count: f64,
    structure_health: f64,
    tension_arc_shape: String,
) -> napi::Result<serde_json::Value> {
    let result = genre_compliance::check_genre_compliance_checked(
        &genre,
        &genre_compliance::GenreEvidence {
            has_hea: Some(has_hea),
            has_solution: Some(has_solution),
            has_ticking_clock: Some(has_ticking_clock),
            protagonist_survives: Some(protagonist_survives),
            character_growth: Some(character_growth),
            worldbuilding_present: Some(worldbuilding_present),
            word_count: Some(super::unsigned_index(word_count)?),
            structure_health: Some(structure_health),
            tension_arc_shape: Some(tension_arc_shape),
            ..Default::default()
        },
    )
    .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

#[napi(
    js_name = "checkGenreComplianceChecked",
    ts_args_type = "genre: string, evidence: Partial<import('./standalone.generated').GenreEvidence>",
    ts_return_type = "import('./standalone.generated').CheckGenreComplianceResult"
)]
pub fn check_genre_compliance_checked(
    genre: String,
    evidence: serde_json::Value,
) -> napi::Result<serde_json::Value> {
    let evidence = super::decode_input(evidence, "invalid genre evidence")?;
    let result = genre_compliance::check_genre_compliance_checked(&genre, &evidence)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    super::encode_result(result)
}

/// Compute narrative entropy (information-pacing) across the manuscript.
// Approximate: `EntropyDiagnosis::Mixed` is recursive (`Vec<Box<EntropyDiagnosis>>`);
// typed as `Array<unknown>` rather than inlining the self-referential union.
#[napi(
    js_name = "computeNarrativeEntropy",
    ts_return_type = "import('./standalone.generated').ComputeNarrativeEntropyResult"
)]
pub fn compute_narrative_entropy(
    scenes: Vec<String>,
    themes: Vec<(String, f64)>,
    structure_template: String,
    genre: String,
    character_arcs: Vec<(String, String)>,
) -> napi::Result<serde_json::Value> {
    let scene_refs: Vec<&str> = scenes.iter().map(|s| s.as_str()).collect();
    let result = narrative_entropy::compute_narrative_entropy_checked(
        &scene_refs,
        &themes,
        &structure_template,
        &genre,
        &character_arcs,
    )
    .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

#[napi(
    js_name = "measureNarrativeUncertainty",
    ts_args_type = "manuscript: import('./manuscript').Manuscript, observations: import('./standalone.generated').SuppliedUncertainty[]",
    ts_return_type = "import('./standalone.generated').MeasureNarrativeUncertaintyResult"
)]
pub fn measure_narrative_uncertainty(
    manuscript: serde_json::Value,
    observations: serde_json::Value,
) -> napi::Result<serde_json::Value> {
    let manuscript = super::decode_input(manuscript, "invalid manuscript object")?;
    let observations: Vec<narrative_entropy::SuppliedUncertainty> =
        super::decode_input(observations, "invalid uncertainty observations")?;
    let result = narrative_entropy::measure_narrative_uncertainty(&manuscript, &observations)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    super::encode_result(result)
}

#[napi(
    js_name = "measureSemanticNovelty",
    ts_args_type = "embeddings: number[][], options?: Partial<import('./standalone.generated').SemanticNoveltyOptions>",
    ts_return_type = "import('./standalone.generated').MeasureSemanticNoveltyResult"
)]
pub fn measure_semantic_novelty(
    embeddings: Vec<Vec<f64>>,
    options: Option<serde_json::Value>,
) -> napi::Result<serde_json::Value> {
    let options = super::decode_input(
        options.unwrap_or_else(|| serde_json::json!({})),
        "invalid semantic novelty options",
    )?;
    let result = narrative_entropy::measure_semantic_novelty(&embeddings, &options)
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    super::encode_result(result)
}

/// Map a writer-declared genre label to a canonical genre, if recognized.
#[napi(
    js_name = "genreFromLabel",
    ts_return_type = "import('./standalone.generated').GenreFromLabelResult"
)]
pub fn genre_from_label(label: String) -> napi::Result<serde_json::Value> {
    let result = genre::genre_from_label(&label);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Detect genre using SBERT cosine similarity against genre descriptions when the
/// model is available, degrading to the keyword/prose-stats heuristic otherwise.
#[napi(
    js_name = "detectGenre",
    ts_return_type = "import('./standalone.generated').DetectGenreResult"
)]
pub fn detect_genre(
    themes: Vec<String>,
    tension_pattern: Vec<f64>,
    dialogue_ratio: f64,
    fkgl: f64,
) -> napi::Result<serde_json::Value> {
    let result = genre::detect_genre(themes, tension_pattern, dialogue_ratio, fkgl);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Detect genre from a raw manuscript sample using SBERT cosine similarity when
/// available, falling back to the label/theme-driven `detectGenre` otherwise.
#[napi(
    js_name = "detectGenreSmart",
    ts_return_type = "import('./standalone.generated').DetectGenreSmartResult"
)]
pub fn detect_genre_smart(
    sample_text: String,
    themes: Vec<String>,
    tension_pattern: Vec<f64>,
    dialogue_ratio: f64,
    fkgl: f64,
) -> napi::Result<serde_json::Value> {
    let result =
        genre::detect_genre_smart(&sample_text, themes, tension_pattern, dialogue_ratio, fkgl);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Evaluate manuscript metrics against a genre's expected quantitative profile.
#[napi(
    js_name = "evaluateAgainstGenre",
    ts_return_type = "import('./standalone.generated').EvaluateAgainstGenreResult"
)]
pub fn evaluate_against_genre(
    genre_label: String,
    fkgl: f64,
    velocity: f64,
    dialogue_ratio: f64,
    tension_variance: f64,
    tonal_whiplash_count: f64,
    word_count: f64,
) -> napi::Result<serde_json::Value> {
    let genre = genre::genre_from_label(&genre_label).ok_or_else(|| {
        napi::Error::from_reason(format!("Unknown genre label: '{}'", genre_label))
    })?;
    let result = genre::evaluate_against_genre(
        genre,
        fkgl,
        velocity,
        dialogue_ratio,
        tension_variance,
        super::unsigned_index(tonal_whiplash_count)?,
        super::unsigned_index(word_count)?,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Get the expected quantitative profile for a genre, by label.
#[napi(
    js_name = "getGenreProfile",
    ts_return_type = "import('./standalone.generated').GetGenreProfileResult"
)]
pub fn get_genre_profile(genre_label: String) -> napi::Result<serde_json::Value> {
    let genre = genre::genre_from_label(&genre_label).ok_or_else(|| {
        napi::Error::from_reason(format!("Unknown genre label: '{}'", genre_label))
    })?;
    let result = genre::get_genre_profile(genre);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Detect themes using SBERT if available, falling back to keyword detection.
#[napi(
    js_name = "detectThemesSmart",
    ts_return_type = "import('./standalone.generated').DetectThemesSmartResult"
)]
pub fn detect_themes_smart(
    scenes: Vec<String>,
    scenes_tokens: Vec<Vec<String>>,
) -> napi::Result<serde_json::Value> {
    let result = theme::detect_themes_smart(&scenes, &scenes_tokens);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Analyze how detected themes evolve and resolve across the manuscript.
#[napi(
    js_name = "analyzeResolution",
    ts_return_type = "import('./standalone.generated').AnalyzeResolutionResult"
)]
pub fn analyze_resolution(
    scene_themes: Vec<std::collections::HashMap<String, f64>>,
) -> napi::Result<serde_json::Value> {
    let result = theme::analyze_resolution(&scene_themes);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}
