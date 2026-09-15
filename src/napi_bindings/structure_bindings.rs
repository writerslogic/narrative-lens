//! `structure` category napi bindings. See `napi_bindings/mod.rs` for the pattern.

use napi_derive::napi;

use crate::structure::{
    foreshadow, genre, genre_compliance, narrative_entropy, opening_analysis, promise_payoff,
    structure, thematic_argument, theme,
};

/// Look up a built-in narrative structure template by name.
#[napi(js_name = "getStructureTemplate")]
pub fn get_structure_template(name: String) -> napi::Result<serde_json::Value> {
    let result = structure::get_structure_template(&name)
        .map_err(|e| napi::Error::from_reason(e.to_string()))?;
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Score how well a set of scene tensions/purposes matches a structure template.
#[napi(js_name = "validateStructure")]
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
#[napi(js_name = "suggestBestTemplate")]
pub fn suggest_best_template(
    scene_tensions: Vec<f64>,
    scene_purposes: Vec<String>,
) -> napi::Result<serde_json::Value> {
    let result = structure::suggest_best_template(scene_tensions, scene_purposes);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Analyze foreshadowing setup-payoff arcs across scenes.
#[napi(js_name = "analyzeForeshadowing")]
pub fn analyze_foreshadowing(
    scenes_tokens: Vec<Vec<String>>,
    total_scenes: u32,
) -> napi::Result<serde_json::Value> {
    let result = foreshadow::analyze_foreshadowing(scenes_tokens, total_scenes as usize);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Build the promise-payoff ledger from analysis data.
#[napi(js_name = "buildPromisePayoffLedger")]
pub fn build_promise_payoff_ledger(
    scenes: Vec<String>,
    scene_word_counts: Vec<u32>,
    genre: String,
    foreshadowing_items: Vec<(String, u32, bool)>,
    character_first_appearances: Vec<(String, u32)>,
    structure_template: String,
    total_scenes: u32,
) -> napi::Result<serde_json::Value> {
    let scene_refs: Vec<&str> = scenes.iter().map(|s| s.as_str()).collect();
    let word_counts: Vec<usize> = scene_word_counts.iter().map(|&c| c as usize).collect();
    let foreshadowing: Vec<(String, usize, bool)> = foreshadowing_items
        .into_iter()
        .map(|(desc, scene, resolved)| (desc, scene as usize, resolved))
        .collect();
    let characters: Vec<(String, usize)> = character_first_appearances
        .into_iter()
        .map(|(name, scene)| (name, scene as usize))
        .collect();
    let result = promise_payoff::build_promise_payoff_ledger(
        &scene_refs,
        &word_counts,
        &genre,
        &foreshadowing,
        &characters,
        &structure_template,
        total_scenes as usize,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Track the thematic argument across the manuscript.
#[napi(js_name = "trackThematicArgument")]
pub fn track_thematic_argument(
    scenes: Vec<String>,
    themes: Vec<(String, f64)>,
    characters: Vec<String>,
    character_decisions: Vec<(String, u32, String)>,
    genre: String,
) -> napi::Result<serde_json::Value> {
    let scene_refs: Vec<&str> = scenes.iter().map(|s| s.as_str()).collect();
    let character_refs: Vec<&str> = characters.iter().map(|s| s.as_str()).collect();
    let decisions: Vec<(String, usize, String)> = character_decisions
        .into_iter()
        .map(|(name, scene, decision)| (name, scene as usize, decision))
        .collect();
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
#[napi(js_name = "analyzeOpening")]
pub fn analyze_opening(
    text: String,
    genre: String,
    full_manuscript_voice_fingerprint: Option<Vec<f64>>,
) -> napi::Result<serde_json::Value> {
    let result = opening_analysis::analyze_opening(
        &text,
        &genre,
        full_manuscript_voice_fingerprint.as_deref(),
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Check a manuscript's compliance with genre conventions.
#[napi(js_name = "checkGenreCompliance")]
pub fn check_genre_compliance(
    genre: String,
    has_hea: bool,
    has_solution: bool,
    has_ticking_clock: bool,
    protagonist_survives: bool,
    character_growth: bool,
    worldbuilding_present: bool,
    word_count: u32,
    structure_health: f64,
    tension_arc_shape: String,
) -> napi::Result<serde_json::Value> {
    let result = genre_compliance::check_genre_compliance(
        &genre,
        has_hea,
        has_solution,
        has_ticking_clock,
        protagonist_survives,
        character_growth,
        worldbuilding_present,
        word_count as usize,
        structure_health,
        &tension_arc_shape,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Compute narrative entropy (information-pacing) across the manuscript.
#[napi(js_name = "computeNarrativeEntropy")]
pub fn compute_narrative_entropy(
    scenes: Vec<String>,
    themes: Vec<(String, f64)>,
    structure_template: String,
    genre: String,
    character_arcs: Vec<(String, String)>,
) -> napi::Result<serde_json::Value> {
    let scene_refs: Vec<&str> = scenes.iter().map(|s| s.as_str()).collect();
    let result = narrative_entropy::compute_narrative_entropy(
        &scene_refs,
        &themes,
        &structure_template,
        &genre,
        &character_arcs,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Map a writer-declared genre label to a canonical genre, if recognized.
#[napi(js_name = "genreFromLabel")]
pub fn genre_from_label(label: String) -> napi::Result<serde_json::Value> {
    let result = genre::genre_from_label(&label);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Detect genre using SBERT cosine similarity against genre descriptions when the
/// model is available, degrading to the keyword/prose-stats heuristic otherwise.
#[napi(js_name = "detectGenre")]
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
#[napi(js_name = "detectGenreSmart")]
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
#[napi(js_name = "evaluateAgainstGenre")]
pub fn evaluate_against_genre(
    genre_label: String,
    fkgl: f64,
    velocity: f64,
    dialogue_ratio: f64,
    tension_variance: f64,
    tonal_whiplash_count: u32,
    word_count: u32,
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
        tonal_whiplash_count as usize,
        word_count as usize,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Get the expected quantitative profile for a genre, by label.
#[napi(js_name = "getGenreProfile")]
pub fn get_genre_profile(genre_label: String) -> napi::Result<serde_json::Value> {
    let genre = genre::genre_from_label(&genre_label).ok_or_else(|| {
        napi::Error::from_reason(format!("Unknown genre label: '{}'", genre_label))
    })?;
    let result = genre::get_genre_profile(genre);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Detect themes using SBERT if available, falling back to keyword detection.
#[napi(js_name = "detectThemesSmart")]
pub fn detect_themes_smart(
    scenes: Vec<String>,
    scenes_tokens: Vec<Vec<String>>,
) -> napi::Result<serde_json::Value> {
    let result = theme::detect_themes_smart(&scenes, &scenes_tokens);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Analyze how detected themes evolve and resolve across the manuscript.
#[napi(js_name = "analyzeResolution")]
pub fn analyze_resolution(
    scene_themes: Vec<std::collections::HashMap<String, f64>>,
) -> napi::Result<serde_json::Value> {
    let result = theme::analyze_resolution(&scene_themes);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}
