//! Source-grounded opening measurements and explicitly uncalibrated heuristic
//! indices; reader continuation, agent decisions and editorial quality are unassessed.
mod measurement;
use crate::craft::{
    lexicon::WordMatcher,
    measurements,
    voice::{CharacterVoiceProfile, VoiceDrift, VoiceSampleCounts},
};
use crate::manuscript::TextRange;
use crate::substrate::utils::AnalysisAvailability;
pub use measurement::{OpeningCue, OpeningCueKind, OpeningParagraph};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::OnceLock;

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HookType {
    Action,
    Question,
    Voice,
    Situation,
    Character,
    Image,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpeningAnalysisResult {
    #[serde(default)]
    pub availability: AnalysisAvailability,
    #[serde(default)]
    pub diagnostics: Vec<String>,
    #[serde(default)]
    pub quality_assessed: bool,
    #[serde(default)]
    pub first_line_range: Option<TextRange>,
    #[serde(default)]
    pub first_sentence_range: Option<TextRange>,
    #[serde(default)]
    pub paragraphs: Vec<OpeningParagraph>,
    #[serde(default)]
    pub cues: Vec<OpeningCue>,
    #[serde(default)]
    pub voice_comparison: Option<VoiceDrift>,
    #[serde(default)]
    pub voice_sample: VoiceSampleCounts,
    pub hook_type: Option<HookType>,
    pub hook_score: f64,
    pub first_line_quality: f64,
    pub first_paragraph_quality: f64,
    pub backstory_ratio: f64,
    pub voice_establishment: f64,
    pub engagement_trajectory: Vec<f64>,
    pub promises_made: Vec<String>,
    /// Legacy alias: capped capitalized-token candidates, not verified introductions.
    pub characters_introduced: usize,
    /// Legacy alias: setting/sensory wordlist threshold, not verified world clarity.
    pub world_established: bool,
    pub genre_alignment: f64,
    pub read_on_prediction: f64,
    /// Word count of the first line (structural signal).
    pub first_line_word_count: usize,
    /// Word counts per paragraph (structural signal: paragraph-length trajectory).
    pub paragraph_word_counts: Vec<usize>,
    /// Index of the first paragraph that contains dialogue (0-based), or None.
    pub dialogue_onset_paragraph: Option<usize>,
    pub issues: Vec<String>,
    pub strengths: Vec<String>,
    pub agent_perspective: String,
    pub first_line_feedback: String,
    pub revision_suggestion: String,
}

/// Backstory/exposition indicators.
const BACKSTORY_MARKERS: &[&str] = &[
    "had been",
    "had always",
    "used to",
    "years ago",
    "back when",
    "growing up",
    "as a child",
    "remembered when",
    "it all started",
    "the history of",
    "was born in",
    "for generations",
    "long before",
    "in the old days",
    "once upon a time",
    "ever since",
    "from the beginning",
    "originally",
];

/// Action hook indicators (used for hook classification only).
const ACTION_INDICATORS: &[&str] = &[
    "ran",
    "jumped",
    "crashed",
    "exploded",
    "screamed",
    "slammed",
    "fired",
    "dodged",
    "sprinted",
    "dove",
    "lunged",
    "shattered",
];

/// Question/mystery indicators.
const QUESTION_INDICATORS: &[&str] = &[
    "who",
    "why",
    "what if",
    "wondered",
    "mystery",
    "strange",
    "impossible",
    "shouldn't have",
    "shouldn’t have",
    "disappeared",
];

/// Voice/character indicators.
const VOICE_INDICATORS: &[&str] = &["i", "my", "me", "we", "our"];

/// Situation/setting indicators.
const SITUATION_INDICATORS: &[&str] = &[
    "the city",
    "the town",
    "the world",
    "the year was",
    "in the",
    "on the",
    "at the",
    "when the",
];

/// Sensory/image indicators (used for hook classification and world detection only).
const IMAGE_INDICATORS: &[&str] = &[
    "smell", "taste", "sound", "light", "dark", "cold", "hot", "wind", "rain", "blood", "silence",
    "shadow",
];

/// Genre-specific opening quality modifiers.
fn genre_opening_preferences(genre: &str) -> (Vec<&'static str>, f64) {
    match genre.to_lowercase().as_str() {
        "thriller" => (vec!["action", "question", "situation"], 0.9),
        "romance" => (vec!["character", "voice", "situation"], 0.8),
        "mystery" => (vec!["question", "situation", "image"], 0.9),
        "literary" | "literary fiction" => (vec!["voice", "image", "character"], 0.85),
        "fantasy" => (vec!["situation", "image", "action"], 0.8),
        "sci-fi" | "science fiction" => (vec!["situation", "question", "image"], 0.85),
        "horror" => (vec!["image", "situation", "question"], 0.9),
        "historical" => (vec!["situation", "image", "voice"], 0.8),
        "ya" | "young adult" => (vec!["voice", "character", "action"], 0.85),
        _ => (vec!["voice", "action", "question"], 0.75),
    }
}

fn action_matcher() -> &'static WordMatcher {
    static M: OnceLock<WordMatcher> = OnceLock::new();
    M.get_or_init(|| WordMatcher::new(ACTION_INDICATORS))
}
fn question_matcher() -> &'static WordMatcher {
    static M: OnceLock<WordMatcher> = OnceLock::new();
    M.get_or_init(|| WordMatcher::new(QUESTION_INDICATORS))
}
fn voice_matcher() -> &'static WordMatcher {
    static M: OnceLock<WordMatcher> = OnceLock::new();
    M.get_or_init(|| WordMatcher::new(VOICE_INDICATORS))
}
fn situation_matcher() -> &'static WordMatcher {
    static M: OnceLock<WordMatcher> = OnceLock::new();
    M.get_or_init(|| WordMatcher::new(SITUATION_INDICATORS))
}
fn image_matcher() -> &'static WordMatcher {
    static M: OnceLock<WordMatcher> = OnceLock::new();
    M.get_or_init(|| WordMatcher::new(IMAGE_INDICATORS))
}
fn backstory_matcher() -> &'static WordMatcher {
    static M: OnceLock<WordMatcher> = OnceLock::new();
    M.get_or_init(|| WordMatcher::new(BACKSTORY_MARKERS))
}
fn distinct_matches(matcher: &WordMatcher, text: &str) -> usize {
    matcher
        .find_iter(text)
        .map(|m| m.text.to_ascii_lowercase())
        .collect::<BTreeSet<_>>()
        .len()
}

fn classify_hook(first_line: &str, first_paragraph: &str) -> (Option<HookType>, f64) {
    let action = distinct_matches(action_matcher(), first_line);
    let question = distinct_matches(question_matcher(), first_paragraph)
        + usize::from(first_paragraph.contains('?'));
    let voice = distinct_matches(voice_matcher(), first_line);
    let situation = distinct_matches(situation_matcher(), first_paragraph);
    let image = distinct_matches(image_matcher(), first_paragraph);
    let character = count_characters_introduced(first_paragraph);
    let mut scores = Vec::new();
    if action > 0 {
        scores.push((HookType::Action, 0.5 + action as f64 * 0.15));
    }
    if question > 0 {
        scores.push((
            HookType::Question,
            0.4 + question as f64 * 0.1 + if first_line.contains('?') { 0.2 } else { 0.0 },
        ));
    }
    if voice > 0 {
        scores.push((HookType::Voice, 0.3 + voice as f64 * 0.15));
    }
    if situation > 0 {
        scores.push((HookType::Situation, 0.3 + situation as f64 * 0.1));
    }
    if image > 0 {
        scores.push((HookType::Image, 0.4 + image as f64 * 0.12));
    }
    if character > 1 {
        scores.push((
            HookType::Character,
            0.3 + (character as f64 * 0.08).min(0.4),
        ));
    }
    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    scores
        .into_iter()
        .next()
        .map_or((None, 0.0), |(kind, score)| (Some(kind), score.min(1.0)))
}

fn calculate_backstory_ratio(text: &str) -> f64 {
    let sentences = crate::substrate::text::split_sentences(text);
    crate::substrate::utils::safe_div(
        sentences
            .iter()
            .filter(|s| backstory_matcher().is_match(s))
            .count() as f64,
        sentences.len() as f64,
    )
}

fn count_characters_introduced(text: &str) -> usize {
    // Heuristic: count capitalized words that appear in dialogue attribution or as subjects
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut names: BTreeSet<String> = BTreeSet::new();

    for (i, word) in words.iter().enumerate() {
        // Skip first word of sentences
        if i == 0 {
            continue;
        }
        let prev = if i > 0 { words[i - 1] } else { "" };
        let is_sentence_start = prev.ends_with('.') || prev.ends_with('!') || prev.ends_with('?');
        if is_sentence_start {
            continue;
        }

        if word
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false)
            && word.len() > 1
            && word.chars().all(|c| c.is_alphabetic())
        {
            let name = word.to_string();
            names.insert(name);
        }
    }

    // Filter out common non-name capitalized words
    let common_words = [
        "The", "This", "That", "These", "Those", "His", "Her", "Its", "My", "Your", "Our", "Their",
        "He", "She", "It", "We", "They", "Mr", "Mrs", "Ms",
    ];
    names.retain(|n| !common_words.contains(&n.as_str()));

    names.len().min(10)
}

/// Returns per-paragraph engagement scores derived purely from structural signals:
/// dialogue presence, question marks, paragraph length, and backstory markers.
/// No IMAGE/ACTION wordlist hits are used here.
fn engagement_per_paragraph(paragraphs: &[&str], measured: &[OpeningParagraph]) -> Vec<f64> {
    paragraphs
        .iter()
        .zip(measured)
        .map(|(para, measured)| {
            let word_count = measured.word_count;
            let mut score = 0.5;

            // Dialogue boosts engagement (structural: punctuation-based)
            if measured.quotation_candidate_count > 0 {
                score += 0.2;
            }

            // Questions boost (structural: punctuation-based)
            let question_count = para.chars().filter(|c| *c == '?').count();
            score += (question_count as f64 * 0.1).min(0.2);

            // Short paragraphs signal action/urgency (structural: word count)
            if word_count < 20 {
                score += 0.1;
            }

            // Very long paragraphs reduce engagement (structural: word count)
            if word_count > 200 {
                score -= 0.1;
            }

            // Backstory penalty (structural: tense/phrase markers)
            let backstory = distinct_matches(backstory_matcher(), para);
            score -= backstory as f64 * 0.1;

            score.clamp(0.1, 1.0)
        })
        .collect()
}

pub fn analyze_opening_checked(
    text: &str,
    genre: &str,
    baseline: Option<&CharacterVoiceProfile>,
) -> crate::Result<OpeningAnalysisResult> {
    if genre.len() > 128 || genre.chars().any(char::is_control) {
        return Err(measurements::invalid(
            "opening genre label exceeds supported format",
        ));
    }
    if let Some(base) = baseline {
        crate::craft::voice::measure_voice_drift_checked(base, base)?;
    }
    let measured = measurement::measure(text)?;
    let paragraphs = crate::substrate::text::split_paragraphs(text);

    let first_line = text.lines().next().unwrap_or("").trim();
    let first_paragraph = paragraphs.first().copied().unwrap_or("");

    // Hook classification
    let (hook_type, hook_score) = classify_hook(first_line, first_paragraph);

    // --- Structural sub-signals ---

    // First-line word count
    let first_line_word_count = measurements::tokens(first_line).len();

    // Paragraph-length trajectory
    let para_word_counts: Vec<_> = measured.paragraphs.iter().map(|p| p.word_count).collect();

    // Dialogue onset
    let dialogue_onset_paragraph = measured
        .paragraphs
        .iter()
        .position(|p| p.quotation_candidate_count > 0);

    // First line quality: purely structural (word count range, backstory penalty)
    let first_line_quality = {
        let mut q: f64 = 0.5;
        // Ideal length: 5–25 words
        if (5..=25).contains(&first_line_word_count) {
            q += 0.3;
        } else if first_line_word_count > 40 {
            q -= 0.2;
        }
        // Penalty for starting with backstory markers
        if backstory_matcher().is_match(first_line) {
            q -= 0.3;
        }
        q.clamp(0.0, 1.0)
    };

    // First paragraph quality: structural (word count range, backstory ratio, dialogue)
    let first_paragraph_quality = {
        let mut q = 0.5;
        let para_words = measurements::tokens(first_paragraph).len();
        if (20..=150).contains(&para_words) {
            q += 0.15;
        }
        let backstory_ratio_para = calculate_backstory_ratio(first_paragraph);
        q -= backstory_ratio_para * 0.3;
        if measured
            .paragraphs
            .first()
            .is_some_and(|p| p.quotation_candidate_count > 0)
        {
            q += 0.1;
        }
        q.clamp(0.0, 1.0)
    };

    // Backstory ratio
    let backstory_ratio = calculate_backstory_ratio(text);

    // Voice establishment (structural: first-person pronouns, long-word density)
    let voice_establishment = {
        let mut score: f64 = 0.5;
        let lower = first_paragraph.to_lowercase();
        if lower.starts_with("i ") || lower.contains(" i ") {
            score += 0.3;
        }
        let words = measurements::tokens(first_paragraph);
        let long_words = words.iter().filter(|w| w.chars().count() > 8).count();
        if long_words > 2 {
            score += 0.1;
        }
        score.clamp(0.0, 1.0)
    };

    // Engagement trajectory (structural signals only)
    let engagement_trajectory = engagement_per_paragraph(&paragraphs, &measured.paragraphs);

    // Promises
    let promises_made = Vec::new();

    // Characters introduced
    let characters_introduced = count_characters_introduced(text);

    // World establishment
    let world_established = distinct_matches(situation_matcher(), text) >= 2
        || distinct_matches(image_matcher(), text) >= 3;

    // Genre alignment
    let (preferred_hooks, genre_weight) = genre_opening_preferences(genre);
    let hook_type_name = match &hook_type {
        Some(HookType::Action) => "action",
        Some(HookType::Question) => "question",
        Some(HookType::Voice) => "voice",
        Some(HookType::Situation) => "situation",
        Some(HookType::Character) => "character",
        Some(HookType::Image) => "image",
        None => "",
    };
    let genre_alignment = if preferred_hooks.contains(&hook_type_name) {
        genre_weight
    } else {
        genre_weight * 0.5
    };

    // KISS: read_on_prediction is built only from structural sub-signals
    // (word counts, dialogue/backstory markers, paragraph variety); IMAGE/ACTION
    // wordlist hits are deliberately excluded as measured-unreliable.
    let read_on_prediction = {
        let avg_engagement = if engagement_trajectory.is_empty() {
            0.5
        } else {
            engagement_trajectory.iter().sum::<f64>() / engagement_trajectory.len() as f64
        };

        // Dialogue onset bonus: dialogue in first 2 paragraphs = +0.1
        let dialogue_bonus = match dialogue_onset_paragraph {
            Some(idx) if idx < 2 => 0.1,
            Some(_) => 0.05,
            None => 0.0,
        };

        // Paragraph-length variety: coefficient of variation of word counts
        let para_variety = if para_word_counts.len() >= 2 {
            let mean =
                para_word_counts.iter().sum::<usize>() as f64 / para_word_counts.len() as f64;
            if mean > 0.0 {
                let variance = para_word_counts
                    .iter()
                    .map(|&c| {
                        let d = c as f64 - mean;
                        d * d
                    })
                    .sum::<f64>()
                    / para_word_counts.len() as f64;
                let cv = variance.sqrt() / mean;
                // CV > 0.5 = good variety; cap contribution at 0.1
                (cv * 0.1).min(0.1)
            } else {
                0.0
            }
        } else {
            0.0
        };

        (first_line_quality * 0.25
            + (1.0 - backstory_ratio) * 0.20
            + voice_establishment * 0.15
            + avg_engagement * 0.15
            + dialogue_bonus
            + para_variety
            + genre_alignment * 0.15)
            .clamp(0.0, 1.0)
    };

    let mut issues = Vec::new();
    let mut strengths = Vec::new();
    if backstory_ratio > 0.4 {
        issues.push(format!("Backstory-phrase cues occur in {:.0}% of segmented sentences; flashback content is unverified.",backstory_ratio*100.0));
    }
    if first_line_word_count > 40 {
        issues.push("The first physical line exceeds the heuristic 40-token band; a line may contain several sentences.".into());
    }
    if characters_introduced > 4 {
        issues.push(format!(
            "{} capitalized-token name candidates; character introductions are unverified.",
            characters_introduced
        ));
    }
    if hook_type.is_none() {
        issues.push(
            "No configured opening cue threshold was reached; hook quality is unassessed.".into(),
        );
    }
    if !world_established {
        issues.push(
            "Setting/sensory cue threshold not reached; setting clarity is unassessed.".into(),
        );
    }
    if first_line_quality > 0.7 {
        strengths
            .push("First-line length lies in the configured 5–25-token heuristic band.".into());
    }
    if backstory_ratio < 0.1 {
        strengths.push("Few configured backstory-phrase cues in segmented sentences.".into());
    }
    if dialogue_onset_paragraph.is_some_and(|i| i < 2) {
        strengths.push("Paired quotation candidates appear in the first two paragraphs.".into());
    }
    let agent_perspective="Agent interest and reader continuation are unassessed; numeric opening signals are uncalibrated heuristics.".into();
    let first_line_feedback = format!(
        "The first physical line contains {} tokens. Inspect its source range separately from the first segmented sentence.",
        first_line_word_count
    );
    let revision_suggestion="Use the source cues and paragraph trajectory to inspect choices against your intended opening; no particular hook or revision is mandatory.".into();
    let label = baseline.map_or("Opening", |p| p.character.as_str());
    let voice = crate::craft::voice::profile_measurement(label, &[text]);
    let voice_comparison = baseline
        .map(|base| crate::craft::voice::measure_voice_drift_checked(base, &voice.profile))
        .transpose()?;
    let mut diagnostics=vec!["opening_indices_are_uncalibrated_signals;editorial_quality_and_read_on_probability_unassessed".into(),
        "character_and_world_fields_are_wordlist_proxies;promises_and_author_intent_not_inferred".into(),
        "typed_voice_comparison_uses_entire_opening;comparable_baseline_sampling_is_caller_responsibility".into()];
    diagnostics.extend(measured.diagnostics);
    Ok(OpeningAnalysisResult {
        availability: if text.trim().is_empty() {
            AnalysisAvailability::Unavailable
        } else {
            AnalysisAvailability::Partial
        },
        diagnostics,
        quality_assessed: false,
        first_line_range: measured.first_line_range,
        first_sentence_range: measured.first_sentence_range,
        paragraphs: measured.paragraphs,
        cues: measured.cues,
        voice_comparison,
        voice_sample: voice.sample,

        hook_type,
        hook_score,
        first_line_quality,
        first_paragraph_quality,
        backstory_ratio,
        voice_establishment,
        engagement_trajectory,
        promises_made,
        characters_introduced,
        world_established,
        genre_alignment,
        read_on_prediction,
        first_line_word_count,
        paragraph_word_counts: para_word_counts,
        dialogue_onset_paragraph,
        issues,
        strengths,
        agent_perspective,
        first_line_feedback,
        revision_suggestion,
    })
}

pub fn analyze_opening(
    text: &str,
    genre: &str,
    legacy_fingerprint: Option<&[f64]>,
) -> OpeningAnalysisResult {
    let mut result = analyze_opening_checked(text, genre, None).unwrap_or_else(|_| {
        let mut unavailable = analyze_opening_checked("", "", None).expect("empty bounded opening");
        unavailable
            .diagnostics
            .push("invalid_opening_input;use_checked_api".into());
        unavailable
    });
    if legacy_fingerprint.is_some_and(|values| !values.is_empty()) {
        result
            .diagnostics
            .push("legacy_arbitrary_voice_fingerprint_unsupported;use_typed_baseline".into());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_opening() {
        let text = "The bullet shattered the window and she dove behind the counter. Glass rained across the tile floor. Another shot fired. She sprinted for the back door.";
        let result = analyze_opening(text, "thriller", None);
        assert!(matches!(result.hook_type, Some(HookType::Action)));
        assert!(result.hook_score > 0.5);
    }

    #[test]
    fn test_backstory_heavy_opening() {
        let text = "She had always been a quiet child. Growing up in the small town, she had been known for her love of books. Years ago, her mother used to read to her every night. Back when things were simpler, the family had been happy.";
        let result = analyze_opening(text, "literary", None);
        assert!(result.backstory_ratio > 0.3);
        assert!(
            result
                .issues
                .iter()
                .any(|i| i.to_ascii_lowercase().contains("backstory"))
        );
    }

    #[test]
    fn test_question_hook() {
        let text = "Who killed Marcus Webb? That was the question that had haunted Detective Sano for three months. The case file sat on her desk, growing thicker but no closer to an answer.";
        let result = analyze_opening(text, "mystery", None);
        assert!(matches!(result.hook_type, Some(HookType::Question)));
    }

    #[test]
    fn test_engagement_trajectory_non_empty() {
        let text = "First paragraph here.\n\nSecond paragraph with some action and running.\n\nThird paragraph with dialogue: \"Hello,\" she said.";
        let result = analyze_opening(text, "literary", None);
        assert_eq!(result.engagement_trajectory.len(), 3);
    }

    #[test]
    fn test_first_line_word_count_structural_signal() {
        let text = "She ran.\n\nThe rest of the story follows.";
        let result = analyze_opening(text, "thriller", None);
        // "She ran." = 2 words
        assert_eq!(result.first_line_word_count, 2);
    }

    #[test]
    fn test_paragraph_word_counts_trajectory() {
        let text = "Short line.\n\nThis is a somewhat longer paragraph with more words in it to test the trajectory signal.\n\nEnd.";
        let result = analyze_opening(text, "literary", None);
        assert_eq!(result.paragraph_word_counts.len(), 3);
        // Middle paragraph should be longest
        assert!(result.paragraph_word_counts[1] > result.paragraph_word_counts[0]);
        assert!(result.paragraph_word_counts[1] > result.paragraph_word_counts[2]);
    }

    #[test]
    fn test_dialogue_onset_detected() {
        let text = "The morning was cold.\n\n\"Get up,\" she said. \"We need to move.\"\n\nHe rolled over and stared at the ceiling.";
        let result = analyze_opening(text, "thriller", None);
        // Dialogue is in paragraph index 1
        assert_eq!(result.dialogue_onset_paragraph, Some(1));
    }

    #[test]
    fn test_dialogue_onset_none_when_absent() {
        let text =
            "The morning was cold. She walked to the window. Outside, the city was waking up.";
        let result = analyze_opening(text, "literary", None);
        assert_eq!(result.dialogue_onset_paragraph, None);
    }

    #[test]
    fn test_read_on_prediction_no_wordlist_inflation() {
        // A text with zero IMAGE/ACTION words but strong structural signals:
        // short first line, early dialogue, no backstory
        let text = "She stopped.\n\n\"Now,\" he said.\n\nThe door opened.";
        let result = analyze_opening(text, "thriller", None);
        // read_on_prediction should be > 0.5 due to structural signals alone
        assert!(
            result.read_on_prediction > 0.4,
            "read_on_prediction={} should be driven by structure",
            result.read_on_prediction
        );
        // Dialogue onset should be early
        assert_eq!(result.dialogue_onset_paragraph, Some(1));
    }

    #[test]
    fn test_backstory_heavy_lowers_read_on() {
        let text = "She had always been a quiet child. Growing up in the small town, she had been known for her love of books. Years ago, her mother used to read to her every night. Back when things were simpler, the family had been happy. It all started long before she was born, in the old days when the family had been different.";
        let result = analyze_opening(text, "literary", None);
        // Heavy backstory should suppress read_on_prediction
        assert!(
            result.read_on_prediction < 0.7,
            "read_on_prediction={} should be suppressed by backstory",
            result.read_on_prediction
        );
        assert!(result.backstory_ratio > 0.5);
    }
}
