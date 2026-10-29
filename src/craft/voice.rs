//! Defined lexical/style measurements over supplied attributed dialogue.
//! Similarity and drift are distances on fixed, documented feature scales;
//! they do not establish character identity or editorial quality.
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use super::measurements;
use crate::Result;
use crate::substrate::text::{split_sentences, tokenize_words, word_slices};
use crate::substrate::utils::{AnalysisAvailability, safe_div};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SIMILARITY_THRESHOLD: f64 = 0.85;

/// Function words only: `top_words` must keep common verbs and nouns so that
/// a voice profile reflects the character's vocabulary, which the broader
/// `text::stopwords` list would strip.
fn stopwords() -> &'static HashSet<&'static str> {
    static INST: OnceLock<HashSet<&str>> = OnceLock::new();
    INST.get_or_init(|| {
        [
            "the", "a", "an", "and", "but", "or", "nor", "for", "yet", "so", "is", "am", "are",
            "was", "were", "be", "been", "do", "does", "did", "has", "have", "had", "may", "might",
            "must", "shall", "should", "will", "would", "can", "could", "i", "me", "my", "we",
            "us", "our", "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
            "them", "their", "this", "that", "these", "those", "who", "whom", "whose", "which",
            "what", "when", "where", "why", "how", "all", "each", "both", "few", "more", "most",
            "some", "no", "not", "one", "two", "just", "if", "then", "than", "too", "as", "at",
            "by", "in", "of", "on", "to", "up", "with", "from", "out", "off", "own", "here",
            "there", "once", "about", "after", "again", "also", "before", "because", "between",
            "into", "very", "really", "much", "well", "back", "still", "even", "now",
        ]
        .into_iter()
        .collect()
    })
}

fn contractions() -> &'static HashSet<&'static str> {
    static INST: OnceLock<HashSet<&str>> = OnceLock::new();
    INST.get_or_init(|| {
        [
            "don't",
            "doesn't",
            "didn't",
            "won't",
            "can't",
            "couldn't",
            "shouldn't",
            "wouldn't",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "hasn't",
            "haven't",
            "hadn't",
            "i'm",
            "i'll",
            "i've",
            "i'd",
            "you're",
            "you'll",
            "you've",
            "you'd",
            "he's",
            "she's",
            "it's",
            "we're",
            "we'll",
            "we've",
            "we'd",
            "they're",
            "they'll",
            "they've",
            "they'd",
            "that's",
            "what's",
            "who's",
            "here's",
            "there's",
            "let's",
        ]
        .into_iter()
        .collect()
    })
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CharacterVoiceProfile {
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

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceSimilarity {
    pub char_a: String,
    pub char_b: String,
    pub similarity: f64,
    pub most_similar_dimension: String,
    pub suggestion: String,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceAnalysisResult {
    #[serde(default)]
    pub availability: AnalysisAvailability,
    #[serde(default)]
    pub diagnostics: Vec<String>,
    pub samples: Vec<VoiceSampleCounts>,
    pub profiles: Vec<CharacterVoiceProfile>,
    pub similarities: Vec<VoiceSimilarity>,
    pub distinct_voices: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract bigrams and trigrams from tokens.
fn extract_ngrams(tokens: &[String], n: usize) -> Vec<String> {
    if tokens.len() < n {
        return Vec::new();
    }
    tokens.windows(n).map(|w| w.join(" ")).collect()
}

// ---------------------------------------------------------------------------
// Profile building
// ---------------------------------------------------------------------------

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceSampleCounts {
    pub input_turn_count: usize,
    /// Turns containing at least one token; denominators exclude empty turns.
    pub turn_count: usize,
    pub word_count: usize,
    pub sentence_count: usize,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceProfileMeasurement {
    pub availability: AnalysisAvailability,
    pub diagnostics: Vec<String>,
    pub profile: CharacterVoiceProfile,
    pub sample: VoiceSampleCounts,
}

fn empty_profile(character_name: &str) -> CharacterVoiceProfile {
    CharacterVoiceProfile {
        character: character_name.into(),
        avg_sentence_length: 0.0,
        vocabulary_richness: 0.0,
        question_rate: 0.0,
        exclamation_rate: 0.0,
        contraction_rate: 0.0,
        formality_score: 0.5,
        avg_word_length: 0.0,
        unique_phrases: Vec::new(),
        top_words: Vec::new(),
    }
}

pub(crate) fn profile_measurement(
    character_name: &str,
    dialogue_lines: &[impl AsRef<str>],
) -> VoiceProfileMeasurement {
    let mut sample = VoiceSampleCounts {
        input_turn_count: dialogue_lines.len(),
        ..Default::default()
    };
    let mut total_chars = 0usize;
    let mut unique_words = HashSet::new();
    let mut content_word_freq = HashMap::<String, usize>::new();
    let mut phrase_freq = HashMap::<String, usize>::new();
    let mut contractions_count = 0usize;
    let mut questions = 0usize;
    let mut exclamations = 0usize;
    let mut sentence_words = 0usize;
    for line in dialogue_lines {
        let line = line.as_ref();
        let tokens = tokenize_words(line);
        if tokens.is_empty() {
            continue;
        }
        sample.turn_count += 1;
        sample.word_count += tokens.len();
        questions += usize::from(line.trim_end().ends_with('?'));
        exclamations += usize::from(line.contains('!'));
        for sentence in split_sentences(line) {
            let count = word_slices(sentence).count();
            if count > 0 {
                sample.sentence_count += 1;
                sentence_words += count;
            }
        }
        for token in &tokens {
            total_chars += token.chars().count();
            unique_words.insert(token.clone());
            contractions_count += usize::from(contractions().contains(token.as_str()));
            if !stopwords().contains(token.as_str()) && token.chars().count() > 1 {
                *content_word_freq.entry(token.clone()).or_default() += 1;
            }
        }
        for n in [2, 3] {
            for phrase in extract_ngrams(&tokens, n) {
                *phrase_freq.entry(phrase).or_default() += 1;
            }
        }
    }
    let mut profile = empty_profile(character_name);
    if sample.word_count > 0 {
        profile.avg_sentence_length = safe_div(sentence_words as f64, sample.sentence_count as f64);
        profile.vocabulary_richness = unique_words.len() as f64 / sample.word_count as f64;
        profile.question_rate = questions as f64 / sample.turn_count as f64;
        profile.exclamation_rate = exclamations as f64 / sample.turn_count as f64;
        profile.contraction_rate = contractions_count as f64 / sample.word_count as f64;
        profile.avg_word_length = total_chars as f64 / sample.word_count as f64;
        profile.formality_score = (1.0 - (profile.contraction_rate * 20.0).min(1.0)) * 0.4
            + ((profile.avg_word_length - 3.0) / 3.0).clamp(0.0, 1.0) * 0.3
            + ((profile.avg_sentence_length - 5.0) / 15.0).clamp(0.0, 1.0) * 0.3;
        let mut ranked: Vec<_> = content_word_freq.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        profile.top_words = ranked.into_iter().take(5).map(|(word, _)| word).collect();
        profile.unique_phrases = phrase_freq
            .into_iter()
            .filter(|(phrase, count)| {
                *count > 2
                    && !phrase
                        .split_whitespace()
                        .all(|word| stopwords().contains(word))
            })
            .map(|(phrase, _)| phrase)
            .collect();
        profile.unique_phrases.sort();
    }
    VoiceProfileMeasurement {
        availability: if sample.word_count == 0 {
            AnalysisAvailability::Unavailable
        } else {
            AnalysisAvailability::Complete
        },
        diagnostics: vec![
            "defined_stylometry;formality_and_similarity_are_uncalibrated_proxies;type_token_ratio_is_length_sensitive".into(),
        ],
        profile,
        sample,
    }
}

/// Measure token/sentence features on supplied turns, without joining turn boundaries.
/// Question rate counts eligible turns ending in `?`; exclamation rate counts
/// eligible turns containing `!`. Typographic apostrophes normalize for tokens only.
pub fn build_voice_profile_checked(
    character_name: &str,
    dialogue_lines: &[String],
) -> Result<VoiceProfileMeasurement> {
    crate::manuscript::config::validate_name(character_name)?;
    measurements::validate_texts(dialogue_lines.iter().map(String::as_str))?;
    Ok(profile_measurement(character_name, dialogue_lines))
}

pub fn build_voice_profile(
    character_name: &str,
    dialogue_lines: &[String],
) -> CharacterVoiceProfile {
    match build_voice_profile_checked(character_name, dialogue_lines) {
        Ok(measured) => measured.profile,
        Err(_) => {
            log::warn!("voice profile unavailable: invalid input; use checked API for details");
            empty_profile("[unavailable]")
        }
    }
}

// ---------------------------------------------------------------------------
// Voice comparison
// ---------------------------------------------------------------------------

const DIM_NAMES: [&str; 6] = [
    "sentence_length",
    "vocabulary",
    "question_rate",
    "exclamation_rate",
    "contraction_rate",
    "formality",
];

/// Map each profile to a fixed [0,1]^6 point using per-dimension scaling, so a
/// profile's vector is stable regardless of how many profiles are compared.
/// (Min-max across the set collapsed every dimension to 0/1 for any pair.)
fn normalize_vectors(profiles: &[&CharacterVoiceProfile]) -> Vec<[f64; 6]> {
    profiles
        .iter()
        .map(|p| {
            [
                (p.avg_sentence_length / 25.0).clamp(0.0, 1.0),
                p.vocabulary_richness.clamp(0.0, 1.0),
                p.question_rate.clamp(0.0, 1.0),
                p.exclamation_rate.clamp(0.0, 1.0),
                (p.contraction_rate * 4.0).clamp(0.0, 1.0),
                p.formality_score.clamp(0.0, 1.0),
            ]
        })
        .collect()
}

fn dimension_suggestion(dim: &str, a: &CharacterVoiceProfile, b: &CharacterVoiceProfile) -> String {
    format!(
        "{} and {} have close fixed-scale style measurements; the closest dimension is {}. Consider the samples and intended contrast before revising.",
        a.character, b.character, dim
    )
}

fn compare_valid_profiles(profiles: &[&CharacterVoiceProfile]) -> Vec<VoiceSimilarity> {
    let n = profiles.len();
    if n < 2 {
        return Vec::new();
    }
    let normed = normalize_vectors(profiles);
    let mut results = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = (1.0
                - crate::substrate::utils::euclidean_distance(&normed[i][..], &normed[j][..])
                    / 6f64.sqrt())
            .clamp(0.0, 1.0);

            // Find most similar dimension (smallest absolute difference)
            let mut min_diff = f64::MAX;
            let mut most_similar_dim = 0;
            for d in 0..6 {
                let diff = (normed[i][d] - normed[j][d]).abs();
                if diff < min_diff {
                    min_diff = diff;
                    most_similar_dim = d;
                }
            }

            let dim_name = DIM_NAMES[most_similar_dim];
            let suggestion = if sim > SIMILARITY_THRESHOLD {
                dimension_suggestion(dim_name, profiles[i], profiles[j])
            } else {
                String::new()
            };

            results.push(VoiceSimilarity {
                char_a: profiles[i].character.clone(),
                char_b: profiles[j].character.clone(),
                similarity: sim,
                most_similar_dimension: dim_name.to_string(),
                suggestion,
            });
        }
    }

    results
}

fn validate_profile(profile: &CharacterVoiceProfile) -> Result<()> {
    crate::manuscript::config::validate_name(&profile.character)?;
    for value in [profile.avg_sentence_length, profile.avg_word_length] {
        if !value.is_finite() || !(0.0..=measurements::MAX_TEXT_BYTES as f64).contains(&value) {
            return Err(measurements::invalid(
                "voice lengths must be finite within text limits",
            ));
        }
    }
    for value in [
        profile.vocabulary_richness,
        profile.question_rate,
        profile.exclamation_rate,
        profile.contraction_rate,
        profile.formality_score,
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(measurements::invalid(
                "voice rates must be finite within 0..=1",
            ));
        }
    }
    Ok(())
}

fn has_sample(profile: &CharacterVoiceProfile) -> bool {
    profile.avg_word_length > 0.0
        && profile.avg_sentence_length > 0.0
        && profile.vocabulary_richness > 0.0
}

/// Fixed-scale Euclidean similarity, not a validated identity/realism judgment.
/// Zero-word profiles are excluded rather than ranked as identical voices.
pub fn compare_voices_checked(profiles: &[CharacterVoiceProfile]) -> Result<Vec<VoiceSimilarity>> {
    measurements::validate_names(profiles.iter().map(|profile| profile.character.as_str()))?;
    for profile in profiles {
        validate_profile(profile)?;
    }
    let eligible: Vec<_> = profiles
        .iter()
        .filter(|profile| has_sample(profile))
        .collect();
    Ok(compare_valid_profiles(&eligible))
}

pub fn compare_voices(profiles: &[CharacterVoiceProfile]) -> Vec<VoiceSimilarity> {
    compare_voices_checked(profiles).unwrap_or_else(|_| {
        log::warn!("voice comparison unavailable: invalid input; use checked API for details");
        Vec::new()
    })
}

// ---------------------------------------------------------------------------
// Voice-drift detection (baseline vs. revision)
// ---------------------------------------------------------------------------
//
// `compare_voices` answers "do two *different* characters sound alike?".
// Drift answers a different question: "has a *single* established voice
// (narrative or per-character) moved away from itself across a revision?".
// We reuse `build_voice_profile` for both the baseline and the current draft,
// then measure the signed, normalized movement on each feature the profile
// already exposes. The per-feature deltas localize the drift; the overall
// score quantifies it. Identical profiles produce ~0 drift (no fabrication).

/// Feature dimensions measured for drift. Fixed order → deterministic output.
/// Superset of `DIM_NAMES`: drift also tracks `avg_word_length`, which the
/// inter-character vector omits but which is a meaningful stylistic shift on
/// revision (e.g. swapping plain words for ornate ones).
const DRIFT_DIM_NAMES: [&str; 7] = [
    "sentence_length",
    "vocabulary",
    "question_rate",
    "exclamation_rate",
    "contraction_rate",
    "formality",
    "word_length",
];

/// Map a profile's features onto a fixed [0,1]^7 point using the same
/// per-dimension scaling rationale as `normalize_vectors`, so a one-unit move
/// in any normalized dimension is comparable to a one-unit move in another and
/// the overall drift score is interpretable. Rates are already in [0,1];
/// sentence/word lengths and contraction rate are scaled by their typical
/// dynamic range before clamping.
fn drift_feature_vector(p: &CharacterVoiceProfile) -> [f64; 7] {
    [
        (p.avg_sentence_length / 25.0).clamp(0.0, 1.0),
        p.vocabulary_richness.clamp(0.0, 1.0),
        p.question_rate.clamp(0.0, 1.0),
        p.exclamation_rate.clamp(0.0, 1.0),
        (p.contraction_rate * 4.0).clamp(0.0, 1.0),
        p.formality_score.clamp(0.0, 1.0),
        // Word length typically spans ~3..9 chars; map that band to [0,1].
        ((p.avg_word_length - 3.0) / 6.0).clamp(0.0, 1.0),
    ]
}

/// How a feature moved from baseline to current. `delta` is signed in
/// normalized [-1,1] space (current minus baseline); `baseline`/`current` are
/// the raw feature values for human-readable reporting.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FeatureDrift {
    pub feature: String,
    pub delta: f64,
    pub baseline: f64,
    pub current: f64,
}

/// Result of measuring how far a revised voice has diverged from a baseline.
///
/// `overall` is the mean absolute normalized per-feature delta in [0,1]:
/// 0.0 = no drift (identical profiles), 1.0 = maximal movement on every
/// feature. `per_feature` lists every dimension's signed normalized delta,
/// sorted by magnitude descending (ties broken by the fixed dimension order)
/// so the largest contributors lead. `notes` summarizes the dominant shifts in
/// plain language. `drifted` is a convenience flag: overall exceeds the
/// notable-drift threshold.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceDrift {
    #[serde(default)]
    pub availability: AnalysisAvailability,
    #[serde(default)]
    pub diagnostics: Vec<String>,
    pub overall: f64,
    pub per_feature: Vec<FeatureDrift>,
    pub notes: Vec<String>,
    pub drifted: bool,
}

/// Overall drift at or above this is worth surfacing to the writer.
const DRIFT_NOTABLE_THRESHOLD: f64 = 0.15;
/// A single feature whose absolute normalized delta is at or above this gets a note.
const DRIFT_FEATURE_NOTE_THRESHOLD: f64 = 0.1;

fn raw_feature_value(dim: &str, p: &CharacterVoiceProfile) -> f64 {
    match dim {
        "sentence_length" => p.avg_sentence_length,
        "vocabulary" => p.vocabulary_richness,
        "question_rate" => p.question_rate,
        "exclamation_rate" => p.exclamation_rate,
        "contraction_rate" => p.contraction_rate,
        "formality" => p.formality_score,
        "word_length" => p.avg_word_length,
        _ => 0.0,
    }
}

fn drift_note(fd: &FeatureDrift) -> String {
    let direction = if fd.delta > 0.0 { "rose" } else { "fell" };
    let pct = (fd.delta.abs() * 100.0).round();
    match fd.feature.as_str() {
        "sentence_length" => format!(
            "Sentences {direction} (avg {:.1} → {:.1} words; {pct}% of range).",
            fd.baseline, fd.current
        ),
        "vocabulary" => format!(
            "Vocabulary richness {direction} ({:.2} → {:.2}; {pct}% of range).",
            fd.baseline, fd.current
        ),
        "question_rate" => format!(
            "Question rate {direction} ({:.0}% → {:.0}% of lines).",
            fd.baseline * 100.0,
            fd.current * 100.0
        ),
        "exclamation_rate" => format!(
            "Exclamation rate {direction} ({:.0}% → {:.0}% of lines).",
            fd.baseline * 100.0,
            fd.current * 100.0
        ),
        "contraction_rate" => format!(
            "Contraction use {direction} ({:.1}% → {:.1}% of words).",
            fd.baseline * 100.0,
            fd.current * 100.0
        ),
        "formality" => format!(
            "Formality {direction} ({:.2} → {:.2}; {pct}% of range).",
            fd.baseline, fd.current
        ),
        "word_length" => format!(
            "Word length {direction} (avg {:.1} → {:.1} chars; {pct}% of range).",
            fd.baseline, fd.current
        ),
        _ => format!("{} {direction} ({pct}% of range).", fd.feature),
    }
}

/// Measure how far a `current` (revised) voice profile has drifted from a
/// `baseline` (established) profile across the features both expose.
///
/// Deterministic: same inputs → same output, with `per_feature` in a stable
/// order. Identical profiles yield `overall == 0.0` and no notes.
fn measure_valid_voice_drift(
    baseline: &CharacterVoiceProfile,
    current: &CharacterVoiceProfile,
) -> VoiceDrift {
    let base_vec = drift_feature_vector(baseline);
    let cur_vec = drift_feature_vector(current);

    let mut per_feature: Vec<FeatureDrift> = (0..DRIFT_DIM_NAMES.len())
        .map(|d| {
            let dim = DRIFT_DIM_NAMES[d];
            FeatureDrift {
                feature: dim.to_string(),
                delta: cur_vec[d] - base_vec[d],
                baseline: raw_feature_value(dim, baseline),
                current: raw_feature_value(dim, current),
            }
        })
        .collect();

    // Overall = mean absolute normalized delta → interpretable [0,1] score.
    let overall = safe_div(
        per_feature.iter().map(|fd| fd.delta.abs()).sum::<f64>(),
        per_feature.len() as f64,
    )
    .clamp(0.0, 1.0);

    // Largest movers first; the fixed dimension order breaks ties so ordering
    // is fully deterministic.
    per_feature.sort_by(|a, b| {
        b.delta
            .abs()
            .partial_cmp(&a.delta.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let notes: Vec<String> = per_feature
        .iter()
        .filter(|fd| fd.delta.abs() >= DRIFT_FEATURE_NOTE_THRESHOLD)
        .map(drift_note)
        .collect();

    VoiceDrift {
        availability: AnalysisAvailability::Complete,
        diagnostics: vec!["fixed_scale_stylometric_drift;threshold_is_uncalibrated".into()],
        overall,
        per_feature,
        notes,
        drifted: overall >= DRIFT_NOTABLE_THRESHOLD,
    }
}

fn unavailable_drift(diagnostic: &str) -> VoiceDrift {
    VoiceDrift {
        availability: AnalysisAvailability::Unavailable,
        diagnostics: vec![diagnostic.into()],
        overall: 0.0,
        per_feature: Vec::new(),
        notes: Vec::new(),
        drifted: false,
    }
}

pub fn measure_voice_drift_checked(
    baseline: &CharacterVoiceProfile,
    current: &CharacterVoiceProfile,
) -> Result<VoiceDrift> {
    validate_profile(baseline)?;
    validate_profile(current)?;
    if baseline.character.to_lowercase() != current.character.to_lowercase() {
        return Err(measurements::invalid(
            "voice drift requires the same character label",
        ));
    }
    Ok(if has_sample(baseline) && has_sample(current) {
        measure_valid_voice_drift(baseline, current)
    } else {
        unavailable_drift("voice_drift_requires_nonempty_samples")
    })
}

pub fn measure_voice_drift(
    baseline: &CharacterVoiceProfile,
    current: &CharacterVoiceProfile,
) -> VoiceDrift {
    measure_voice_drift_checked(baseline, current)
        .unwrap_or_else(|_| unavailable_drift("invalid_voice_profiles;use_checked_api"))
}

pub fn measure_voice_drift_from_text_checked(
    label: &str,
    baseline_lines: &[String],
    current_lines: &[String],
) -> Result<VoiceDrift> {
    measurements::validate_texts(
        baseline_lines
            .iter()
            .chain(current_lines)
            .map(String::as_str),
    )?;
    let baseline = build_voice_profile_checked(label, baseline_lines)?;
    let current = build_voice_profile_checked(label, current_lines)?;
    measure_voice_drift_checked(&baseline.profile, &current.profile)
}

/// Convenience: build both profiles from baseline-text and current-text lines
/// (same speaker/narrator label) and measure drift between them. Reuses
/// `build_voice_profile`, so the drift is computed over exactly the features
/// the profile machinery already extracts.
pub fn measure_voice_drift_from_text(
    label: &str,
    baseline_lines: &[String],
    current_lines: &[String],
) -> VoiceDrift {
    let baseline = build_voice_profile(label, baseline_lines);
    let current = build_voice_profile(label, current_lines);
    measure_voice_drift(&baseline, &current)
}

// ---------------------------------------------------------------------------
// Top-level entry point
// ---------------------------------------------------------------------------

/// Refine unique_phrases: remove phrases that appear in other characters' dialogue.
fn refine_unique_phrases(profiles: &mut [CharacterVoiceProfile], data: &[(String, Vec<String>)]) {
    let mut owners = HashMap::<String, Option<usize>>::new();
    for (index, profile) in profiles.iter().enumerate() {
        for phrase in &profile.unique_phrases {
            owners
                .entry(phrase.clone())
                .and_modify(|owner| *owner = None)
                .or_insert(Some(index));
        }
    }
    for (index, (_, lines)) in data.iter().enumerate() {
        for line in lines {
            let tokens = tokenize_words(line);
            for n in [2, 3] {
                for phrase in extract_ngrams(&tokens, n) {
                    if let Some(owner) = owners.get_mut(&phrase)
                        && *owner != Some(index)
                    {
                        *owner = None;
                    }
                }
            }
        }
    }
    for (index, profile) in profiles.iter_mut().enumerate() {
        profile
            .unique_phrases
            .retain(|phrase| owners[phrase] == Some(index));
        profile.unique_phrases.truncate(10);
    }
}

pub fn analyze_character_voices_checked(
    dialogue_data: &[(String, Vec<String>)],
) -> Result<VoiceAnalysisResult> {
    measurements::validate_names(dialogue_data.iter().map(|(name, _)| name.as_str()))?;
    measurements::validate_texts(
        dialogue_data
            .iter()
            .flat_map(|(_, lines)| lines.iter().map(String::as_str)),
    )?;
    let measured: Vec<_> = dialogue_data
        .iter()
        .map(|(name, lines)| profile_measurement(name, lines))
        .collect();
    let samples: Vec<_> = measured
        .iter()
        .map(|measurement| measurement.sample.clone())
        .collect();
    let mut profiles: Vec<_> = measured
        .into_iter()
        .map(|measurement| measurement.profile)
        .collect();
    refine_unique_phrases(&mut profiles, dialogue_data);
    let similarities = compare_voices_checked(&profiles)?;
    let eligible = samples
        .iter()
        .filter(|sample| sample.word_count > 0)
        .count();
    let distinct_voices = eligible > 1
        && similarities
            .iter()
            .all(|s| s.similarity <= SIMILARITY_THRESHOLD);
    Ok(VoiceAnalysisResult {
        availability: if eligible == 0 {
            AnalysisAvailability::Unavailable
        } else if eligible < profiles.len() {
            AnalysisAvailability::Partial
        } else {
            AnalysisAvailability::Complete
        },
        diagnostics: vec![
            "defined_stylometry;distinct_voices_is_an_uncalibrated_distance_threshold".into(),
        ],
        samples,
        profiles,
        similarities,
        distinct_voices,
    })
}

pub fn analyze_character_voices(dialogue_data: Vec<(String, Vec<String>)>) -> VoiceAnalysisResult {
    analyze_character_voices_checked(&dialogue_data).unwrap_or_else(|_| VoiceAnalysisResult {
        availability: AnalysisAvailability::Unavailable,
        diagnostics: vec!["invalid_dialogue_input;use_checked_api".into()],
        samples: Vec::new(),
        profiles: Vec::new(),
        similarities: Vec::new(),
        distinct_voices: false,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn alice_lines() -> Vec<String> {
        vec![
            "I don't think that's a good idea, do you?".to_string(),
            "We shouldn't be here. It's too dangerous!".to_string(),
            "Can't we just go home?".to_string(),
            "I'm worried about what they'll say.".to_string(),
            "You're not listening to me, are you?".to_string(),
            "Let's try something different, okay?".to_string(),
            "I've been thinking about this for a while now.".to_string(),
            "Don't you think we should be more careful?".to_string(),
            "It's not like we have a choice.".to_string(),
            "I won't let them push us around anymore!".to_string(),
        ]
    }

    fn bob_lines() -> Vec<String> {
        vec![
            "The situation requires careful deliberation.".to_string(),
            "I do not believe that course of action is prudent.".to_string(),
            "We must consider all of the ramifications before proceeding.".to_string(),
            "It would be inadvisable to act without sufficient information.".to_string(),
            "One should always examine the evidence methodically.".to_string(),
            "The circumstances demand a measured and thoughtful response.".to_string(),
            "I have concluded that we must pursue an alternative approach.".to_string(),
            "It is imperative that we maintain our composure.".to_string(),
            "The probability of success remains quite uncertain.".to_string(),
            "We shall proceed with the utmost caution and discipline.".to_string(),
        ]
    }

    // Nearly identical to alice
    fn clone_lines() -> Vec<String> {
        vec![
            "I don't think we should do this, do you?".to_string(),
            "We can't stay here. It's way too risky!".to_string(),
            "Can't we just leave already?".to_string(),
            "I'm scared about what they'll think.".to_string(),
            "You're ignoring me, aren't you?".to_string(),
            "Let's do something else, alright?".to_string(),
            "I've thought about this a lot.".to_string(),
            "Don't you think we need to be careful?".to_string(),
            "It's not as if we have options.".to_string(),
            "I won't let them boss us around!".to_string(),
        ]
    }

    #[test]
    fn test_build_profile_basic() {
        let profile = build_voice_profile("Alice", &alice_lines());
        assert_eq!(profile.character, "Alice");
        assert!(profile.avg_sentence_length > 0.0);
        assert!(profile.vocabulary_richness > 0.0);
        assert!(profile.question_rate > 0.0, "Alice asks questions");
        assert!(profile.exclamation_rate > 0.0, "Alice uses exclamations");
        assert!(profile.contraction_rate > 0.0, "Alice uses contractions");
        assert!(profile.avg_word_length > 0.0);
    }

    #[test]
    fn test_build_profile_empty() {
        let profile = build_voice_profile("Nobody", &[]);
        assert_eq!(profile.avg_sentence_length, 0.0);
        assert_eq!(profile.vocabulary_richness, 0.0);
    }

    #[test]
    fn test_different_voices_low_similarity() {
        let alice = build_voice_profile("Alice", &alice_lines());
        let bob = build_voice_profile("Bob", &bob_lines());
        let sims = compare_voices(&[alice, bob]);
        assert_eq!(sims.len(), 1);
        // Alice (informal, questions) vs Bob (formal, declarative) should differ
        assert!(
            sims[0].similarity < SIMILARITY_THRESHOLD,
            "Alice and Bob should have distinct voices, got similarity {:.3}",
            sims[0].similarity
        );
    }

    #[test]
    fn test_similar_voices_high_similarity() {
        let alice = build_voice_profile("Alice", &alice_lines());
        let clone = build_voice_profile("AliceClone", &clone_lines());
        let sims = compare_voices(&[alice, clone]);
        assert_eq!(sims.len(), 1);
        assert!(
            sims[0].similarity > 0.8,
            "Alice and her clone should sound very similar, got {:.3}",
            sims[0].similarity
        );
    }

    #[test]
    fn test_analyze_character_voices_full() {
        let data = vec![
            ("Alice".to_string(), alice_lines()),
            ("Bob".to_string(), bob_lines()),
        ];
        let result = analyze_character_voices(data);
        assert_eq!(result.profiles.len(), 2);
        assert_eq!(result.similarities.len(), 1);
        assert!(
            result.distinct_voices,
            "Alice and Bob should be flagged as distinct"
        );
    }

    #[test]
    fn test_analyze_flags_indistinct() {
        let data = vec![
            ("Alice".to_string(), alice_lines()),
            ("AliceClone".to_string(), clone_lines()),
        ];
        let result = analyze_character_voices(data);
        // They are very similar; distinct_voices should be false
        // (depends on threshold; at least the similarity should be high)
        assert!(
            result.similarities[0].similarity > 0.8,
            "Clone pair should score high similarity"
        );
    }

    #[test]
    fn test_formality_contrast() {
        let alice = build_voice_profile("Alice", &alice_lines());
        let bob = build_voice_profile("Bob", &bob_lines());
        assert!(
            bob.formality_score > alice.formality_score,
            "Bob should be more formal ({:.2}) than Alice ({:.2})",
            bob.formality_score,
            alice.formality_score
        );
    }

    #[test]
    fn test_contraction_rate_contrast() {
        let alice = build_voice_profile("Alice", &alice_lines());
        let bob = build_voice_profile("Bob", &bob_lines());
        assert!(
            alice.contraction_rate > bob.contraction_rate,
            "Alice should use more contractions ({:.3}) than Bob ({:.3})",
            alice.contraction_rate,
            bob.contraction_rate
        );
    }

    #[test]
    fn test_question_rate() {
        let alice = build_voice_profile("Alice", &alice_lines());
        // Alice has ~4 questions out of 10 lines
        assert!(
            alice.question_rate >= 0.3,
            "Alice question rate should be >= 0.3, got {:.2}",
            alice.question_rate
        );
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = [1.0, 0.5, 0.3, 0.1, 0.2, 0.8];
        let sim = crate::substrate::utils::cosine_similarity(&a[..], &a[..]);
        assert!(
            (sim - 1.0).abs() < 1e-9,
            "identical vectors should have similarity 1.0"
        );
    }

    #[test]
    fn test_single_profile_no_comparison() {
        let sims = compare_voices(&[build_voice_profile("Solo", &alice_lines())]);
        assert!(sims.is_empty());
    }

    #[test]
    fn test_suggestion_generated_for_similar() {
        let alice = build_voice_profile("Alice", &alice_lines());
        let clone = build_voice_profile("AliceClone", &clone_lines());
        let sims = compare_voices(&[alice, clone]);
        if sims[0].similarity > SIMILARITY_THRESHOLD {
            assert!(
                !sims[0].suggestion.is_empty(),
                "Similar voices should have a suggestion"
            );
        }
    }

    #[test]
    fn test_drift_identical_is_zero() {
        let profile = build_voice_profile("Narrator", &alice_lines());
        let drift = measure_voice_drift(&profile, &profile);
        assert!(
            drift.overall < 1e-9,
            "identical profiles should show ~0 drift, got {:.4}",
            drift.overall
        );
        assert!(!drift.drifted, "identical profiles should not be flagged");
        assert!(drift.notes.is_empty(), "no drift → no notes");
        // Every feature delta is zero.
        assert!(drift.per_feature.iter().all(|fd| fd.delta.abs() < 1e-9));
        assert_eq!(drift.per_feature.len(), DRIFT_DIM_NAMES.len());
    }

    #[test]
    fn test_drift_revision_high_and_localized() {
        // Baseline: Alice's casual, contraction-heavy, question-asking voice.
        // Revision: rewritten in Bob's formal, long-sentence register.
        let baseline = build_voice_profile("Narrator", &alice_lines());
        let revised = build_voice_profile("Narrator", &bob_lines());
        let drift = measure_voice_drift(&baseline, &revised);

        assert!(
            drift.overall > DRIFT_NOTABLE_THRESHOLD,
            "a wholesale register change should register notable drift, got {:.4}",
            drift.overall
        );
        assert!(drift.drifted, "should be flagged as drifted");
        assert!(!drift.notes.is_empty(), "drift should produce notes");

        // Formality should be among the strongest movers, and it should rise
        // (Alice → Bob is more formal).
        let formality = drift
            .per_feature
            .iter()
            .find(|fd| fd.feature == "formality")
            .expect("formality dimension present");
        assert!(
            formality.delta > 0.0,
            "revision is more formal, delta should be positive, got {:.3}",
            formality.delta
        );

        // Contractions should fall (Alice uses many, Bob almost none).
        let contraction = drift
            .per_feature
            .iter()
            .find(|fd| fd.feature == "contraction_rate")
            .expect("contraction dimension present");
        assert!(
            contraction.delta < 0.0,
            "revision drops contractions, delta should be negative, got {:.3}",
            contraction.delta
        );
    }

    #[test]
    fn test_drift_per_feature_sorted_by_magnitude() {
        let baseline = build_voice_profile("Narrator", &alice_lines());
        let revised = build_voice_profile("Narrator", &bob_lines());
        let drift = measure_voice_drift(&baseline, &revised);
        for w in drift.per_feature.windows(2) {
            assert!(
                w[0].delta.abs() >= w[1].delta.abs(),
                "per_feature must be sorted by |delta| descending"
            );
        }
    }

    #[test]
    fn test_drift_minor_revision_low() {
        // Alice vs. her near-paraphrase: same voice, should barely drift.
        let drift = measure_voice_drift_from_text("Narrator", &alice_lines(), &clone_lines());
        assert!(
            drift.overall < DRIFT_NOTABLE_THRESHOLD,
            "a faithful paraphrase should stay below the notable threshold, got {:.4}",
            drift.overall
        );
    }

    #[test]
    fn test_drift_from_text_matches_profiles() {
        let from_text = measure_voice_drift_from_text("N", &alice_lines(), &bob_lines());
        let baseline = build_voice_profile("N", &alice_lines());
        let current = build_voice_profile("N", &bob_lines());
        let from_profiles = measure_voice_drift(&baseline, &current);
        assert!((from_text.overall - from_profiles.overall).abs() < 1e-12);
    }
}
