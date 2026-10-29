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
#[napi(
    js_name = "explainReadability",
    ts_return_type = "import('./standalone.generated').ExplainReadabilityResult"
)]
pub fn explain_readability(
    fkgl: f64,
    target_min: f64,
    target_max: f64,
    genre: String,
) -> napi::Result<serde_json::Value> {
    for value in [fkgl, target_min, target_max] {
        super::finite_range(value, -f64::MAX, f64::MAX)?;
    }
    if target_min > target_max {
        return Err(napi::Error::from_reason(
            "readability target range is inverted",
        ));
    }
    let result = explain::explain_readability(fkgl, target_min, target_max, &genre);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Explain a pacing issue (long sentences, or a talking-head scene) for one scene.
#[napi(
    js_name = "explainPacing",
    ts_return_type = "import('./standalone.generated').ExplainPacingResult"
)]
pub fn explain_pacing(
    velocity: f64,
    scene_id: f64,
    is_talking_head: bool,
    dialogue_ratio: f64,
) -> napi::Result<serde_json::Value> {
    super::finite_range(velocity, 0.0, u32::MAX as f64)?;
    super::finite_range(dialogue_ratio, 0.0, 1.0)?;
    let result = explain::explain_pacing(
        velocity,
        super::unsigned_index(scene_id)?,
        is_talking_head,
        dialogue_ratio,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Explain a tension issue: a flat run, a sharp drop, or a manuscript with no
/// significant tension peak.
#[napi(
    js_name = "explainTension",
    ts_return_type = "import('./standalone.generated').ExplainTensionResult"
)]
pub fn explain_tension(
    tension: f64,
    prev_tension: f64,
    scene_id: f64,
    flat_count: f64,
    flat_start: f64,
    total_scenes: f64,
    max_tension: f64,
) -> napi::Result<serde_json::Value> {
    for value in [tension, prev_tension, max_tension] {
        super::finite_range(value, 0.0, 1.0)?;
    }
    let flat_start = super::unsigned_index(flat_start)?;
    let flat_count = super::unsigned_index(flat_count)?;
    let total_scenes = super::unsigned_index(total_scenes)?;
    let scene_id = super::unsigned_index(scene_id)?;
    if total_scenes == 0 && scene_id == 0 && flat_start == 0 && flat_count == 0 {
        return Ok(serde_json::Value::Null);
    }
    if scene_id >= total_scenes
        || flat_start
            .checked_add(flat_count)
            .is_none_or(|end| end > total_scenes)
    {
        return Err(napi::Error::from_reason(
            "tension scene or flat-run range is outside the manuscript",
        ));
    }
    let result = explain::explain_tension(
        tension,
        prev_tension,
        scene_id,
        flat_count,
        flat_start,
        total_scenes,
        max_tension,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Explain a tonal shift between two scenes, if `from_tone` and `to_tone` differ.
#[napi(
    js_name = "explainToneShift",
    ts_return_type = "import('./standalone.generated').ExplainToneShiftResult"
)]
pub fn explain_tone_shift(
    from_tone: String,
    to_tone: String,
    scene_id: f64,
    genre: String,
) -> napi::Result<serde_json::Value> {
    let result = explain::explain_tone_shift(
        &from_tone,
        &to_tone,
        super::unsigned_index(scene_id)?,
        &genre,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Explain a "white room" issue: a scene with too little sensory grounding
/// for its length.
#[napi(
    js_name = "explainWhiteRoom",
    ts_return_type = "import('./standalone.generated').ExplainWhiteRoomResult"
)]
pub fn explain_white_room(
    scene_id: f64,
    sensory_count: f64,
    word_count: f64,
) -> napi::Result<serde_json::Value> {
    let result = explain::explain_white_room(
        super::unsigned_index(scene_id)?,
        super::unsigned_index(sensory_count)?,
        super::unsigned_index(word_count)?,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Explain a show-don't-tell issue (`narrative_distance`, `filter_word`,
/// `diluted_action`, or any other issue type) found in `snippet`.
#[napi(
    js_name = "explainShowDontTell",
    ts_return_type = "import('./standalone.generated').ExplainShowDontTellResult"
)]
pub fn explain_show_dont_tell(
    issue_type: String,
    snippet: String,
) -> napi::Result<serde_json::Value> {
    let result = explain::explain_show_dont_tell(&issue_type, &snippet);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Compute confidence for analysis of a scene based on its text content.
#[napi(
    js_name = "computeSceneConfidence",
    ts_return_type = "import('./standalone.generated').ComputeSceneConfidenceResult"
)]
pub fn compute_scene_confidence(
    scene_text: String,
    scene_id: f64,
) -> napi::Result<serde_json::Value> {
    super::bounded_texts([scene_text.as_str()], 1)?;
    let result =
        confidence::compute_scene_confidence(&scene_text, super::unsigned_index(scene_id)?);
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Compute confidence for a specific finding based on signal agreement and
/// scene confidence.
#[napi(js_name = "computeFindingConfidence")]
pub fn compute_finding_confidence(
    signal_count: f64,
    total_signals: f64,
    scene_confidence: f64,
) -> napi::Result<f64> {
    confidence::compute_finding_confidence_checked(
        super::unsigned_index(signal_count)?,
        super::unsigned_index(total_signals)?,
        scene_confidence,
    )
    .map_err(|error| napi::Error::from_reason(error.to_string()))
}

#[napi(
    js_name = "measureFindingConfidence",
    ts_return_type = "import('./standalone.generated').MeasureFindingConfidenceResult"
)]
pub fn measure_finding_confidence(
    signal_count: f64,
    total_signals: f64,
    scene_confidence: f64,
) -> napi::Result<serde_json::Value> {
    let result = confidence::measure_finding_confidence(
        super::unsigned_index(signal_count)?,
        super::unsigned_index(total_signals)?,
        scene_confidence,
    )
    .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    super::encode_result(result)
}

/// Compute confidence for readability metrics based on sample size.
#[napi(
    js_name = "computeReadabilityConfidence",
    ts_return_type = "import('./standalone.generated').ComputeReadabilityConfidenceResult"
)]
pub fn compute_readability_confidence(
    word_count: f64,
    sentence_count: f64,
) -> napi::Result<serde_json::Value> {
    let result = confidence::compute_readability_confidence(
        super::unsigned_index(word_count)?,
        super::unsigned_index(sentence_count)?,
    );
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Rank the most-frequent content words in `text`, most-frequent first,
/// keeping at most `limit`, using the crate's built-in stopword set.
#[napi(
    js_name = "wordFrequencies",
    ts_return_type = "import('./standalone.generated').WordFrequenciesResult"
)]
pub fn word_frequencies(text: String, limit: f64) -> napi::Result<serde_json::Value> {
    super::bounded_texts([text.as_str()], 1)?;
    let stop = crate::substrate::text::stopwords();
    let result =
        word_stats::word_frequencies(&text, super::unsigned_index(limit)?, |w| stop.contains(w));
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Count and rank already-tokenized `words`, applying the same content-word
/// filter, ordering, and `limit` as `wordFrequencies`, using the crate's
/// built-in stopword set.
#[napi(
    js_name = "wordFrequenciesFromWords",
    ts_return_type = "import('./standalone.generated').WordFrequenciesFromWordsResult"
)]
pub fn word_frequencies_from_words(
    words: Vec<String>,
    limit: f64,
) -> napi::Result<serde_json::Value> {
    super::bounded_texts(words.iter().map(String::as_str), 1_000_000)?;
    let stop = crate::substrate::text::stopwords();
    let result =
        word_stats::word_frequencies_from_words(words, super::unsigned_index(limit)?, |w| {
            stop.contains(w)
        });
    serde_json::to_value(result).map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// Train a corpus embedding from `texts` (default build options) and return
/// the `k` vocabulary words most similar to `word`, strongest first.
///
/// Combined rather than exposing `CorpusEmbedding` as a stateful object: napi
/// cannot cheaply hold a live Rust struct across separate calls without
/// `#[napi]` on the struct itself (a bigger design decision than this binding
/// pass), so training and querying are fused into one call instead.
#[napi(
    js_name = "trainAndFindNearest",
    ts_return_type = "Array<[string, number]>"
)]
pub fn train_and_find_nearest(
    texts: Vec<String>,
    word: String,
    k: f64,
) -> napi::Result<serde_json::Value> {
    let k = super::unsigned_index(k)?;
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let corpus =
        embedding::CorpusEmbedding::train_checked(&refs, &embedding::EmbeddingOptions::default())
            .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    let result = corpus.nearest_words(&word, k);
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
    let corpus =
        embedding::CorpusEmbedding::train_checked(&refs, &embedding::EmbeddingOptions::default())
            .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    Ok(corpus.similarity(&word_a, &word_b))
}
