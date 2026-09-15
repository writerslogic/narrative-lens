//! `substrate` category napi bindings. See `napi_bindings/mod.rs` for the pattern.
//!
//! Unlike `napi_bindings/mod.rs`'s hand-written `*Js` DTOs, everything here
//! goes through a JSON bridge: the wrapped function's return type derives
//! `serde::Serialize` and crosses the FFI boundary as a `serde_json::Value`.
//! This scales to the number of small, varied return shapes under `substrate/`
//! without a parallel DTO per type.

use napi_derive::napi;

use crate::substrate::{confidence, embedding, explain, word_stats};

/// Explain a readability score against a genre's target grade-level range, if
/// it falls outside that range.
#[napi(js_name = "explainReadability")]
pub fn explain_readability(
    fkgl: f64,
    target_min: f64,
    target_max: f64,
    genre: String,
) -> napi::Result<serde_json::Value> {
    let result = explain::explain_readability(fkgl, target_min, target_max, &genre);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Explain a pacing issue (long sentences, or a talking-head scene) for one scene.
#[napi(js_name = "explainPacing")]
pub fn explain_pacing(
    velocity: f64,
    scene_id: u32,
    is_talking_head: bool,
    dialogue_ratio: f64,
) -> napi::Result<serde_json::Value> {
    let result = explain::explain_pacing(
        velocity,
        scene_id as usize,
        is_talking_head,
        dialogue_ratio,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Explain a tension issue: a flat run, a sharp drop, or a manuscript with no
/// significant tension peak.
#[napi(js_name = "explainTension")]
pub fn explain_tension(
    tension: f64,
    prev_tension: f64,
    scene_id: u32,
    flat_count: u32,
    flat_start: u32,
    total_scenes: u32,
    max_tension: f64,
) -> napi::Result<serde_json::Value> {
    let result = explain::explain_tension(
        tension,
        prev_tension,
        scene_id as usize,
        flat_count as usize,
        flat_start as usize,
        total_scenes as usize,
        max_tension,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Explain a tonal shift between two scenes, if `from_tone` and `to_tone` differ.
#[napi(js_name = "explainToneShift")]
pub fn explain_tone_shift(
    from_tone: String,
    to_tone: String,
    scene_id: u32,
    genre: String,
) -> napi::Result<serde_json::Value> {
    let result = explain::explain_tone_shift(&from_tone, &to_tone, scene_id as usize, &genre);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Explain a "white room" issue: a scene with too little sensory grounding
/// for its length.
#[napi(js_name = "explainWhiteRoom")]
pub fn explain_white_room(
    scene_id: u32,
    sensory_count: u32,
    word_count: u32,
) -> napi::Result<serde_json::Value> {
    let result =
        explain::explain_white_room(scene_id as usize, sensory_count as usize, word_count as usize);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Explain a show-don't-tell issue (`narrative_distance`, `filter_word`,
/// `diluted_action`, or any other issue type) found in `snippet`.
#[napi(js_name = "explainShowDontTell")]
pub fn explain_show_dont_tell(
    issue_type: String,
    snippet: String,
) -> napi::Result<serde_json::Value> {
    let result = explain::explain_show_dont_tell(&issue_type, &snippet);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Compute confidence for analysis of a scene based on its text content.
#[napi(js_name = "computeSceneConfidence")]
pub fn compute_scene_confidence(
    scene_text: String,
    scene_id: u32,
) -> napi::Result<serde_json::Value> {
    let result = confidence::compute_scene_confidence(&scene_text, scene_id as usize);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Compute confidence for a specific finding based on signal agreement and
/// scene confidence.
#[napi(js_name = "computeFindingConfidence")]
pub fn compute_finding_confidence(
    signal_count: u32,
    total_signals: u32,
    scene_confidence: f64,
) -> f64 {
    confidence::compute_finding_confidence(
        signal_count as usize,
        total_signals as usize,
        scene_confidence,
    )
}

/// Compute confidence for readability metrics based on sample size.
#[napi(js_name = "computeReadabilityConfidence")]
pub fn compute_readability_confidence(
    word_count: u32,
    sentence_count: u32,
) -> napi::Result<serde_json::Value> {
    let result =
        confidence::compute_readability_confidence(word_count as usize, sentence_count as usize);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Rank the most-frequent content words in `text`, most-frequent first,
/// keeping at most `limit`, using the crate's built-in stopword set.
#[napi(js_name = "wordFrequencies")]
pub fn word_frequencies(text: String, limit: u32) -> napi::Result<serde_json::Value> {
    let stop = crate::substrate::text::stopwords();
    let result = word_stats::word_frequencies(&text, limit as usize, |w| stop.contains(w));
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Count and rank already-tokenized `words`, applying the same content-word
/// filter, ordering, and `limit` as `wordFrequencies`, using the crate's
/// built-in stopword set.
#[napi(js_name = "wordFrequenciesFromWords")]
pub fn word_frequencies_from_words(
    words: Vec<String>,
    limit: u32,
) -> napi::Result<serde_json::Value> {
    let stop = crate::substrate::text::stopwords();
    let result =
        word_stats::word_frequencies_from_words(words, limit as usize, |w| stop.contains(w));
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Train a corpus embedding from `texts` (default build options) and return
/// the `k` vocabulary words most similar to `word`, strongest first.
///
/// Combined rather than exposing `CorpusEmbedding` as a stateful object: napi
/// cannot cheaply hold a live Rust struct across separate calls without
/// `#[napi]` on the struct itself (a bigger design decision than this binding
/// pass), so training and querying are fused into one call instead.
#[napi(js_name = "trainAndFindNearest")]
pub fn train_and_find_nearest(
    texts: Vec<String>,
    word: String,
    k: u32,
) -> napi::Result<serde_json::Value> {
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let corpus = embedding::CorpusEmbedding::train(&refs, &embedding::EmbeddingOptions::default());
    let result = corpus.nearest_words(&word, k as usize);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Train a corpus embedding from `texts` (default build options) and return
/// the cosine similarity between `word_a` and `word_b` in that space.
///
/// Same fused-call design as `trainAndFindNearest`, avoiding a stateful
/// `CorpusEmbedding` handle across the FFI boundary.
#[napi(js_name = "trainAndSimilarity")]
pub fn train_and_similarity(
    texts: Vec<String>,
    word_a: String,
    word_b: String,
) -> napi::Result<f64> {
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let corpus = embedding::CorpusEmbedding::train(&refs, &embedding::EmbeddingOptions::default());
    Ok(corpus.similarity(&word_a, &word_b))
}
