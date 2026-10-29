//! Descriptive dialogue stylometry and source-grounded lexical cues.
//! Automatic realism, information-dump judgments and social power are unassessed.
use super::{lexicon::WordMatcher, measurements};
use crate::Result;
use crate::craft::voice::{VoiceSampleCounts, profile_measurement};
use crate::manuscript::TextRange;
use crate::substrate::utils::AnalysisAvailability;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CharacterVoice {
    pub character: String,
    pub avg_turn_length: f64,
    pub vocabulary_richness: f64,
    pub formality_level: f64,
    pub contraction_rate: f64,
    pub question_rate: f64,
    pub exclamation_rate: f64,
    /// Top words by normalized frequency (freq > 0.02) within this character's
    /// dialogue. These are the character's own most-used words; no
    /// cross-character distinctiveness filter is applied.
    pub frequent_words: Vec<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoicePair {
    pub character_a: String,
    pub character_b: String,
    pub similarity: f64,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DialogueIssue {
    pub scene: usize,
    pub issue_type: String,
    pub description: String,
    pub severity: f64,
    pub characters: Vec<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PowerDynamic {
    pub character_a: String,
    pub character_b: String,
    pub dominant: String,
    pub indicators: Vec<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DialogueRealismResult {
    #[serde(default)]
    pub availability: AnalysisAvailability,
    #[serde(default)]
    pub diagnostics: Vec<String>,
    #[serde(default)]
    pub realism_assessed: bool,
    #[serde(default)]
    pub sample: DialogueSampleCoverage,
    #[serde(default)]
    pub voice_samples: Vec<CharacterVoiceSample>,
    #[serde(default)]
    pub voice_comparison_availability: AnalysisAvailability,
    #[serde(default)]
    pub cues: Vec<DialogueCue>,
    #[serde(default)]
    pub exposition_cue_count: usize,
    pub character_voices: Vec<CharacterVoice>,
    pub voice_pairs: Vec<VoicePair>,
    pub issues: Vec<DialogueIssue>,
    pub power_dynamics: Vec<PowerDynamic>,
    /// Compatibility placeholder 0.5; automatic editorial realism is unassessed.
    pub overall_realism: f64,
    pub voice_distinctiveness: f64,
    /// Compatibility placeholder 0; cues do not prove information dumping.
    pub info_dump_count: usize,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DialogueSampleCoverage {
    pub input_turn_count: usize,
    /// Configured-speaker turns, including zero-word turns.
    pub matched_turn_count: usize,
    /// Configured-speaker turns with at least one word.
    pub eligible_turn_count: usize,
    /// Words in eligible configured-speaker turns only.
    pub word_count: usize,
    pub unmatched_turn_count: usize,
    /// Zero-word turns, across both configured and unconfigured speakers.
    pub empty_turn_count: usize,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CharacterVoiceSample {
    pub character: String,
    pub sample: VoiceSampleCounts,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DialogueCueKind {
    ExpositionMarker,
    FormalMarker,
    LongTurn,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DialogueCue {
    pub turn_index: usize,
    pub scene: usize,
    pub character: String,
    pub kind: DialogueCueKind,
    /// Offsets within the exact original supplied turn, not within a scene.
    pub range: TextRange,
    pub quote: String,
}

/// Expository dialogue patterns ("As you know, Bob").
const INFO_DUMP_PATTERNS: &[&str] = &[
    "as you know",
    "as i'm sure you're aware",
    "you already know",
    "let me explain",
    "as we discussed",
    "you may recall",
    "as you remember",
    "i don't need to tell you",
    "need i remind you",
    "for those who don't know",
    "the thing is",
    "what you need to understand is",
    "basically what happened was",
    "to put it simply",
    "in other words",
];

/// Formal language markers.
const FORMAL_MARKERS: &[&str] = &[
    "furthermore",
    "moreover",
    "nevertheless",
    "consequently",
    "subsequently",
    "notwithstanding",
    "henceforth",
    "whereas",
    "thereby",
    "wherein",
    "heretofore",
    "inasmuch",
    "insofar",
    "aforementioned",
];

fn exposition_matcher() -> &'static WordMatcher {
    static MATCHER: OnceLock<WordMatcher> = OnceLock::new();
    MATCHER.get_or_init(|| {
        let mut patterns: Vec<String> = INFO_DUMP_PATTERNS.iter().map(|p| (*p).into()).collect();
        patterns.extend(
            INFO_DUMP_PATTERNS
                .iter()
                .filter(|p| p.contains('\''))
                .map(|p| p.replace('\'', "’")),
        );
        WordMatcher::new(&patterns.iter().map(String::as_str).collect::<Vec<_>>())
    })
}

fn formal_matcher() -> &'static WordMatcher {
    static MATCHER: OnceLock<WordMatcher> = OnceLock::new();
    MATCHER.get_or_init(|| WordMatcher::new(FORMAL_MARKERS))
}

fn lexical_cosine(a: &BTreeMap<String, f64>, b: &BTreeMap<String, f64>) -> f64 {
    let norm_a = a.values().map(|value| value * value).sum::<f64>().sqrt();
    let norm_b = b.values().map(|value| value * value).sum::<f64>().sqrt();
    let dot = a
        .iter()
        .map(|(word, value)| value * b.get(word).copied().unwrap_or(0.0))
        .sum::<f64>();
    (dot / (norm_a * norm_b)).clamp(0.0, 1.0)
}

fn unavailable(diagnostic: &str) -> DialogueRealismResult {
    DialogueRealismResult {
        availability: AnalysisAvailability::Unavailable,
        diagnostics: vec![diagnostic.into()],
        realism_assessed: false,
        sample: DialogueSampleCoverage::default(),
        voice_samples: Vec::new(),
        voice_comparison_availability: AnalysisAvailability::Unavailable,
        cues: Vec::new(),
        exposition_cue_count: 0,
        character_voices: Vec::new(),
        voice_pairs: Vec::new(),
        issues: Vec::new(),
        power_dynamics: Vec::new(),
        overall_realism: 0.5,
        voice_distinctiveness: 0.5,
        info_dump_count: 0,
    }
}

/// Measure supplied attributed turns, preserving source-relative cue spans.
/// Word-frequency cosine is lexical overlap, not speaker identity. More than
/// 150 tokens is a long-turn cue, never a verdict that conversation is unrealistic.
pub fn analyze_dialogue_realism_checked(
    dialogue_turns: &[(usize, String, String)],
    characters: &[&str],
) -> Result<DialogueRealismResult> {
    measurements::validate_names(characters.iter().copied())?;
    measurements::validate_texts(dialogue_turns.iter().map(|(_, _, text)| text.as_str()))?;
    for (_, speaker, _) in dialogue_turns {
        crate::manuscript::config::validate_name(speaker)?;
    }
    let mut result =
        unavailable("automatic_realism_power_and_information_dump_judgments_unassessed");
    result.sample.input_turn_count = dialogue_turns.len();
    let cast: HashMap<_, _> = characters
        .iter()
        .map(|name| (name.to_lowercase(), *name))
        .collect();
    let mut groups = HashMap::<String, Vec<&str>>::new();
    let mut frequencies = HashMap::<String, BTreeMap<String, f64>>::new();
    for (index, (scene, speaker, text)) in dialogue_turns.iter().enumerate() {
        let key = speaker.to_lowercase();
        let canonical = cast.get(&key).copied();
        let tokens = measurements::tokens(text);
        if tokens.is_empty() {
            result.sample.empty_turn_count += 1;
        }
        if canonical.is_some() {
            result.sample.matched_turn_count += 1;
            if !tokens.is_empty() {
                result.sample.eligible_turn_count += 1;
                result.sample.word_count += tokens.len();
                let frequency = frequencies.entry(key.clone()).or_default();
                for token in &tokens {
                    *frequency.entry(token.clone()).or_default() += 1.0;
                }
            }
            groups.entry(key).or_default().push(text.as_str());
        } else {
            result.sample.unmatched_turn_count += 1;
        }
        let mut found: Vec<_> = exposition_matcher()
            .find_iter(text)
            .map(|m| (m.start, m.end, DialogueCueKind::ExpositionMarker))
            .chain(
                formal_matcher()
                    .find_iter(text)
                    .map(|m| (m.start, m.end, DialogueCueKind::FormalMarker)),
            )
            .take(20_001)
            .collect();
        if tokens.len() > 150 {
            found.push((0, text.len(), DialogueCueKind::LongTurn));
        }
        if result.cues.len().saturating_add(found.len()) > 20_000 {
            return Err(measurements::invalid("dialogue exceeds 20000 source cues"));
        }
        found.sort_by_key(|(start, end, _)| (*start, *end));
        let spans: Vec<_> = found.iter().map(|(start, end, _)| (*start, *end)).collect();
        for ((start, end, kind), range) in
            found.into_iter().zip(measurements::ranges(text, &spans)?)
        {
            result.exposition_cue_count += usize::from(kind == DialogueCueKind::ExpositionMarker);
            result.cues.push(DialogueCue {
                turn_index: index,
                scene: *scene,
                character: canonical.unwrap_or(speaker).into(),
                kind,
                range,
                quote: text[start..end].into(),
            });
        }
    }
    for character in characters {
        let key = character.to_lowercase();
        let lines = groups.get(&key).map(Vec::as_slice).unwrap_or(&[]);
        let measured = profile_measurement(character, lines);
        let profile = &measured.profile;
        if measured.sample.word_count > 0 {
            let frequency = frequencies
                .get_mut(&key)
                .expect("eligible voice frequencies");
            for value in frequency.values_mut() {
                *value /= measured.sample.word_count as f64;
            }
            let mut ranked: Vec<_> = frequency
                .iter()
                .filter(|(_, frequency)| **frequency > 0.02)
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(a.1).then_with(|| a.0.cmp(b.0)));
            result.character_voices.push(CharacterVoice {
                character: (*character).into(),
                avg_turn_length: measured.sample.word_count as f64
                    / measured.sample.turn_count as f64,
                vocabulary_richness: profile.vocabulary_richness,
                formality_level: profile.formality_score,
                contraction_rate: profile.contraction_rate,
                question_rate: profile.question_rate,
                exclamation_rate: profile.exclamation_rate,
                frequent_words: ranked
                    .into_iter()
                    .take(5)
                    .map(|(word, _)| word.clone())
                    .collect(),
            });
        }
        result.voice_samples.push(CharacterVoiceSample {
            character: (*character).into(),
            sample: measured.sample,
        });
    }
    for (i, a) in characters.iter().enumerate() {
        for b in characters.iter().skip(i + 1) {
            if let (Some(fa), Some(fb)) = (
                frequencies.get(&a.to_lowercase()),
                frequencies.get(&b.to_lowercase()),
            ) {
                result.voice_pairs.push(VoicePair {
                    character_a: (*a).into(),
                    character_b: (*b).into(),
                    similarity: lexical_cosine(fa, fb),
                });
            }
        }
    }
    if !result.voice_pairs.is_empty() {
        result.voice_comparison_availability = AnalysisAvailability::Complete;
        result.voice_distinctiveness = (1.0
            - result.voice_pairs.iter().map(|p| p.similarity).sum::<f64>()
                / result.voice_pairs.len() as f64)
            .clamp(0.0, 1.0);
    }
    if result.sample.eligible_turn_count > 0 {
        result.availability = AnalysisAvailability::Partial;
    }
    if result.sample.unmatched_turn_count > 0 {
        result
            .diagnostics
            .push("unconfigured_speakers_excluded_from_voice_profiles".into());
    }
    result.diagnostics.push(
        "formality_is_an_uncalibrated_style_index;source_cues_are_not_editorial_verdicts".into(),
    );
    Ok(result)
}

pub fn analyze_dialogue_realism(
    dialogue_turns: &[(usize, String, String)],
    characters: &[&str],
) -> DialogueRealismResult {
    analyze_dialogue_realism_checked(dialogue_turns, characters)
        .unwrap_or_else(|_| unavailable("invalid_dialogue_input;use_checked_api"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_dialogue_analysis() {
        let turns = vec![
            (
                0,
                "Alice".to_string(),
                "I don't think we should go there.".to_string(),
            ),
            (
                0,
                "Bob".to_string(),
                "Nevertheless, the circumstances require our immediate departure.".to_string(),
            ),
            (
                1,
                "Alice".to_string(),
                "Can't you see what's happening?".to_string(),
            ),
            (
                1,
                "Bob".to_string(),
                "Furthermore, I must inform you of the subsequent developments.".to_string(),
            ),
        ];
        let characters = &["Alice", "Bob"];
        let result = analyze_dialogue_realism(&turns, characters);

        assert_eq!(result.character_voices.len(), 2);
        assert_eq!(result.voice_pairs.len(), 1);

        // Bob should be more formal
        let alice = result
            .character_voices
            .iter()
            .find(|v| v.character == "Alice")
            .unwrap();
        let bob = result
            .character_voices
            .iter()
            .find(|v| v.character == "Bob")
            .unwrap();
        assert!(bob.formality_level > alice.formality_level);
    }

    #[test]
    fn test_info_dump_detection() {
        let turns = vec![
            (0, "Bob".to_string(), "As you know, the empire fell three hundred years ago when the great wizard Mordenkainen cast the spell of dissolution across the seven kingdoms, destroying the barrier between worlds.".to_string()),
        ];
        let result = analyze_dialogue_realism(&turns, &["Bob"]);
        assert!(result.exposition_cue_count > 0);
        assert_eq!(result.info_dump_count, 0);
        assert!(!result.realism_assessed);
    }

    #[test]
    fn test_identical_voices_flagged() {
        let turns: Vec<(usize, String, String)> = (0..20)
            .flat_map(|i| {
                vec![
                    (
                        i,
                        "Alice".to_string(),
                        "I think we should go to the store today.".to_string(),
                    ),
                    (
                        i,
                        "Bob".to_string(),
                        "I think we should go to the park today.".to_string(),
                    ),
                ]
            })
            .collect();
        let result = analyze_dialogue_realism(&turns, &["Alice", "Bob"]);
        assert!(result.voice_pairs[0].similarity > 0.7);
    }

    #[test]
    fn test_frequent_words_field_exists_and_is_vec() {
        let turns = vec![(
            0,
            "Alice".to_string(),
            "I really really really love this place so much.".to_string(),
        )];
        let result = analyze_dialogue_realism(&turns, &["Alice"]);
        let alice = result
            .character_voices
            .iter()
            .find(|v| v.character == "Alice")
            .unwrap();
        // frequent_words is a Vec<String> — just verify it compiles and is accessible
        let _: &Vec<String> = &alice.frequent_words;
    }
}
