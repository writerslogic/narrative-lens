/// Dialogue extraction and attribution: identifies who said what.
///
/// Ports the Python `DialogueAttributor` as standalone functions. Attributes
/// dialogue lines to characters via speech tags, proximity, continuation,
/// and alternation patterns.
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

mod extraction;
pub use extraction::{
    DialogueAttributionMode, DialogueExtractionOptions, DialogueExtractionResult, DialogueKind,
    LocatedDialogueLine, extract_dialogue_checked,
};
pub(crate) use extraction::{QuotationScopes, quotation_scopes};

// ---------------------------------------------------------------------------
// Constants & compiled patterns
// ---------------------------------------------------------------------------

fn speech_verbs() -> &'static HashSet<&'static str> {
    static INST: OnceLock<HashSet<&str>> = OnceLock::new();
    INST.get_or_init(|| {
        [
            "said",
            "asked",
            "replied",
            "whispered",
            "shouted",
            "muttered",
            "exclaimed",
            "answered",
            "cried",
            "called",
            "demanded",
            "insisted",
            "suggested",
            "explained",
            "admitted",
            "murmured",
            "yelled",
            "snapped",
            "pleaded",
            "urged",
            "sighed",
            "groaned",
            "stammered",
            "declared",
            "announced",
            "added",
            "continued",
            "interrupted",
            "protested",
            "agreed",
            "warned",
            "offered",
            "observed",
            "noted",
            "began",
        ]
        .into_iter()
        .collect()
    })
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DialogueLine {
    pub scene_id: usize,
    pub quote: String,
    pub speaker: String,
    pub confidence: f64,
    pub method: String,
    /// The speech verb used in the attribution tag (e.g. "whispered"), if
    /// the line was attributed via an explicit speech tag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag_verb: Option<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct DialogueCharacterStats {
    pub line_count: usize,
    pub word_count: usize,
    pub avg_line_length: f64,
    pub vocabulary_richness: f64,
    pub question_rate: f64,
    pub exclamation_rate: f64,
}

/// Voice profile for measuring how distinctive a character's speech patterns are.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct VoiceProfile {
    pub avg_word_length: f64,
    pub vocabulary_richness: f64,
    pub question_frequency: f64,
    pub avg_line_length: f64,
    pub formality_score: f64,
    pub tag_variety_score: f64,
    pub speech_verbs_used: Vec<String>,
}

/// A pair of characters whose voice profiles are too similar.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SameSoundingPair {
    pub character_a: String,
    pub character_b: String,
    pub similarity: f64,
}

/// Directed edge in the conversation flow graph.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationEdge {
    pub from: String,
    pub to: String,
    pub count: usize,
}

/// A detected monologue (5+ consecutive lines from one speaker).
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Monologue {
    pub speaker: String,
    pub line_count: usize,
    pub scene_id: usize,
}

/// Conversation flow analysis results.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ConversationFlow {
    pub edges: Vec<ConversationEdge>,
    pub dominators: Vec<String>,
    pub never_initiators: Vec<String>,
    pub monologues: Vec<Monologue>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct DialogueAnalysis {
    pub total_lines: usize,
    pub attributed_lines: usize,
    pub attribution_rate: f64,
    pub per_character: HashMap<String, DialogueCharacterStats>,
    pub dialogue_distribution: HashMap<String, f64>,
    pub most_talkative: String,
    pub most_verbose: String,
    pub voice_profiles: HashMap<String, VoiceProfile>,
    pub same_sounding_pairs: Vec<SameSoundingPair>,
    pub conversation_flow: ConversationFlow,
}

// ---------------------------------------------------------------------------
// Contractions list (used for formality scoring)
// ---------------------------------------------------------------------------

static CONTRACTIONS: &[&str] = &[
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "can't",
    "couldn't",
    "shouldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "haven't",
    "hasn't",
    "hadn't",
    "i'm",
    "i've",
    "i'll",
    "i'd",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "he's",
    "she's",
    "it's",
    "we're",
    "we've",
    "we'll",
    "we'd",
    "they're",
    "they've",
    "they'll",
    "they'd",
    "that's",
    "there's",
    "here's",
    "what's",
    "who's",
    "how's",
    "let's",
    "that'll",
    "there'll",
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Extract dialogue from a scene and attribute each quote to a speaker.
pub fn extract_dialogue(
    scene_text: &str,
    scene_id: usize,
    known_characters: &[String],
) -> Vec<DialogueLine> {
    let options = DialogueExtractionOptions {
        attribution: DialogueAttributionMode::ConservativeHeuristic,
        include_indirect: true,
    };
    match extract_dialogue_checked(scene_text, scene_id, known_characters, &options) {
        Ok(result) => result
            .lines
            .into_iter()
            .map(|located| located.line)
            .collect(),
        Err(_) => {
            log::warn!("dialogue extraction unavailable: invalid input or resource limit");
            Vec::new()
        }
    }
}

/// Analyze dialogue patterns across all extracted dialogue lines.
pub fn analyze_dialogue_patterns(
    all_dialogue: &[DialogueLine],
    known_characters: &[String],
) -> DialogueAnalysis {
    let total = all_dialogue.len();
    if total == 0 {
        return DialogueAnalysis::default();
    }

    let attributed = all_dialogue
        .iter()
        .filter(|d| d.speaker != "Unknown")
        .count();

    let mut per_character: HashMap<String, DialogueCharacterStats> = HashMap::new();

    for char_name in known_characters {
        let lines: Vec<&DialogueLine> = all_dialogue
            .iter()
            .filter(|d| d.speaker == *char_name)
            .collect();

        if lines.is_empty() {
            continue;
        }

        let n = lines.len();
        let words_per_line: Vec<usize> = lines
            .iter()
            .map(|q| q.quote.split_whitespace().count())
            .collect();

        let all_words: Vec<String> = lines
            .iter()
            .flat_map(|q| crate::substrate::text::tokenize_words(&q.quote))
            .collect();

        let total_words = all_words.len();
        let unique_words: HashSet<&str> = all_words.iter().map(|s| s.as_str()).collect();

        let questions = lines
            .iter()
            .filter(|q| q.quote.trim_end().ends_with('?'))
            .count();
        let exclamations = lines.iter().filter(|q| q.quote.contains('!')).count();

        let sum_wpl: usize = words_per_line.iter().sum();

        per_character.insert(
            char_name.clone(),
            DialogueCharacterStats {
                line_count: n,
                word_count: total_words,
                avg_line_length: sum_wpl as f64 / n as f64,
                vocabulary_richness: if total_words > 0 {
                    unique_words.len() as f64 / total_words as f64
                } else {
                    0.0
                },
                question_rate: questions as f64 / n as f64,
                exclamation_rate: exclamations as f64 / n as f64,
            },
        );
    }

    let attr_total: usize = per_character.values().map(|v| v.line_count).sum();
    let distribution: HashMap<String, f64> = if attr_total > 0 {
        per_character
            .iter()
            .map(|(c, v)| (c.clone(), v.line_count as f64 / attr_total as f64))
            .collect()
    } else {
        HashMap::new()
    };

    let most_talkative = per_character
        .iter()
        .max_by(|(name_a, a), (name_b, b)| {
            a.line_count
                .cmp(&b.line_count)
                .then_with(|| name_b.cmp(name_a))
        })
        .map(|(c, _)| c.clone())
        .unwrap_or_default();

    let most_verbose = per_character
        .iter()
        .max_by(|(name_a, a), (name_b, b)| {
            a.avg_line_length
                .total_cmp(&b.avg_line_length)
                .then_with(|| name_b.cmp(name_a))
        })
        .map(|(c, _)| c.clone())
        .unwrap_or_default();

    let voice_profiles = compute_voice_profiles(all_dialogue, known_characters);
    let same_sounding_pairs = find_same_sounding_pairs(&voice_profiles);
    let conversation_flow = analyze_conversation_flow(all_dialogue);

    DialogueAnalysis {
        total_lines: total,
        attributed_lines: attributed,
        attribution_rate: attributed as f64 / total as f64,
        per_character,
        dialogue_distribution: distribution,
        most_talkative,
        most_verbose,
        voice_profiles,
        same_sounding_pairs,
        conversation_flow,
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn compute_voice_profiles(
    all_dialogue: &[DialogueLine],
    known_characters: &[String],
) -> HashMap<String, VoiceProfile> {
    let verbs = speech_verbs();
    let mut profiles = HashMap::new();

    for char_name in known_characters {
        let lines: Vec<&DialogueLine> = all_dialogue
            .iter()
            .filter(|d| d.speaker == *char_name)
            .collect();
        if lines.is_empty() {
            continue;
        }

        let n = lines.len();
        let all_words: Vec<String> = lines
            .iter()
            .flat_map(|q| crate::substrate::text::tokenize_words(&q.quote))
            .collect();

        let total_words = all_words.len();
        let unique_words: HashSet<&str> = all_words.iter().map(|s| s.as_str()).collect();

        let avg_word_length = if total_words > 0 {
            all_words.iter().map(|w| w.len()).sum::<usize>() as f64 / total_words as f64
        } else {
            0.0
        };

        let vocabulary_richness = if total_words > 0 {
            unique_words.len() as f64 / total_words as f64
        } else {
            0.0
        };

        let question_frequency = lines
            .iter()
            .filter(|q| q.quote.trim_end().ends_with('?'))
            .count() as f64
            / n as f64;

        let avg_line_length = lines
            .iter()
            .map(|q| q.quote.split_whitespace().count())
            .sum::<usize>() as f64
            / n as f64;

        // Formality: 1.0 = no contractions (formal), lower = more informal
        let contraction_count = all_words
            .iter()
            .filter(|w| CONTRACTIONS.contains(&w.as_str()))
            .count();
        let formality_score = if total_words > 0 {
            1.0 - (contraction_count as f64 / total_words as f64).min(1.0)
        } else {
            1.0
        };

        // Build speech_verbs_used from the actual tag_verb stored on each
        // DialogueLine during attribution. This reflects the genuine speech
        // verbs used to attribute this character's lines, not every verb in
        // the surrounding narration.
        let mut char_verbs_used: HashSet<String> = HashSet::new();
        for line in &lines {
            if line.method == "explicit_tag"
                && let Some(ref verb) = line.tag_verb
            {
                // Only include verbs that are in our known speech-verb set
                let v_lower = verb.to_lowercase();
                if verbs.contains(v_lower.as_str()) {
                    char_verbs_used.insert(v_lower);
                }
            }
        }

        let tag_variety_score = extract_tag_variety(&lines, verbs);

        let mut speech_verbs_used: Vec<String> = char_verbs_used.into_iter().collect();
        speech_verbs_used.sort();

        profiles.insert(
            char_name.clone(),
            VoiceProfile {
                avg_word_length,
                vocabulary_richness,
                question_frequency,
                avg_line_length,
                formality_score,
                tag_variety_score,
                speech_verbs_used,
            },
        );
    }

    profiles
}

/// Count distinct speech verbs actually used to tag this character's lines.
/// Uses the `tag_verb` field stored on each `DialogueLine` during attribution.
fn extract_tag_variety(lines: &[&DialogueLine], verbs: &HashSet<&str>) -> f64 {
    let explicit_lines: Vec<&&DialogueLine> = lines
        .iter()
        .filter(|l| l.method == "explicit_tag")
        .collect();

    let explicit_count = explicit_lines.len();
    if explicit_count == 0 {
        return 0.0;
    }

    // Count distinct speech verbs from the tag_verb field
    let distinct_verbs: HashSet<String> = explicit_lines
        .iter()
        .filter_map(|l| l.tag_verb.as_ref())
        .map(|v| v.to_lowercase())
        .filter(|v| verbs.contains(v.as_str()))
        .collect();

    let unique_count = distinct_verbs.len();
    if unique_count == 0 {
        // No tag_verb data available; fall back to a modest default
        // (1 unique verb assumed for all explicit_tag lines)
        return (1.0_f64 / explicit_count as f64).min(1.0);
    }

    (unique_count as f64 / explicit_count as f64).min(1.0)
}

fn find_same_sounding_pairs(profiles: &HashMap<String, VoiceProfile>) -> Vec<SameSoundingPair> {
    let mut pairs = Vec::new();
    let chars: Vec<(&String, &VoiceProfile)> = profiles.iter().collect();

    for i in 0..chars.len() {
        for j in (i + 1)..chars.len() {
            let (name_a, profile_a) = chars[i];
            let (name_b, profile_b) = chars[j];
            let sim = voice_similarity(profile_a, profile_b);
            if sim > 0.85 {
                pairs.push(SameSoundingPair {
                    character_a: name_a.clone(),
                    character_b: name_b.clone(),
                    similarity: sim,
                });
            }
        }
    }

    pairs.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs
}

fn voice_similarity(a: &VoiceProfile, b: &VoiceProfile) -> f64 {
    // Normalize avg_line_length to a [0,1]-ish range before including in
    // the cosine computation. Using /20.0 means a 40-word line becomes 2.0,
    // which dominates the dot product and inflates similarity between
    // profiles that differ on every other dimension. Cap at 1.0 instead.
    let norm_line_len = 20.0_f64;
    let dims_a = [
        a.avg_word_length / 10.0,
        a.vocabulary_richness,
        a.question_frequency,
        (a.avg_line_length / norm_line_len).min(1.0),
        a.formality_score,
        a.tag_variety_score,
    ];
    let dims_b = [
        b.avg_word_length / 10.0,
        b.vocabulary_richness,
        b.question_frequency,
        (b.avg_line_length / norm_line_len).min(1.0),
        b.formality_score,
        b.tag_variety_score,
    ];

    let dot: f64 = dims_a.iter().zip(dims_b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = dims_a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = dims_b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if mag_a < 1e-9 || mag_b < 1e-9 {
        return 0.0;
    }

    (dot / (mag_a * mag_b)).clamp(0.0, 1.0)
}

fn analyze_conversation_flow(all_dialogue: &[DialogueLine]) -> ConversationFlow {
    if all_dialogue.is_empty() {
        return ConversationFlow::default();
    }

    let attributed: Vec<&DialogueLine> = all_dialogue
        .iter()
        .filter(|d| d.speaker != "Unknown")
        .collect();

    if attributed.is_empty() {
        return ConversationFlow::default();
    }

    // Build speaker-to-next-speaker transition counts
    let mut edge_counts: HashMap<(String, String), usize> = HashMap::new();
    // Track per-scene lines for domination detection
    let mut scene_lines: HashMap<usize, Vec<&str>> = HashMap::new();
    // Track who initiates (speaks first in a scene)
    let mut initiators: HashSet<String> = HashSet::new();
    let mut all_speakers: HashSet<String> = HashSet::new();

    for line in &attributed {
        all_speakers.insert(line.speaker.clone());
        scene_lines
            .entry(line.scene_id)
            .or_default()
            .push(&line.speaker);
    }

    // Build transitions and find initiators per scene
    let mut prev_speaker: Option<(&str, usize)> = None;
    for line in &attributed {
        if let Some((prev, prev_scene)) = prev_speaker {
            if prev_scene == line.scene_id && prev != line.speaker {
                *edge_counts
                    .entry((prev.to_string(), line.speaker.clone()))
                    .or_insert(0) += 1;
            }
        } else {
            initiators.insert(line.speaker.clone());
        }
        if prev_speaker.is_none_or(|(_, s)| s != line.scene_id) {
            initiators.insert(line.speaker.clone());
        }
        prev_speaker = Some((&line.speaker, line.scene_id));
    }

    let edges: Vec<ConversationEdge> = edge_counts
        .into_iter()
        .map(|((from, to), count)| ConversationEdge { from, to, count })
        .collect();

    // Dominators: characters with >50% of lines in any scene
    let mut dominators: HashSet<String> = HashSet::new();
    for speakers in scene_lines.values() {
        if speakers.len() < 2 {
            continue;
        }
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for s in speakers {
            *counts.entry(s).or_insert(0) += 1;
        }
        let total = speakers.len();
        for (name, count) in counts {
            if count as f64 / total as f64 > 0.5 {
                dominators.insert(name.to_string());
            }
        }
    }

    // Never-initiators: speakers who appear but never speak first in any scene
    let never_initiators: Vec<String> = all_speakers
        .iter()
        .filter(|s| !initiators.contains(*s))
        .cloned()
        .collect();

    // Monologues: 5+ consecutive lines from the same speaker within a scene
    let mut monologues = Vec::new();
    let mut run_speaker: Option<(&str, usize, usize)> = None; // (speaker, scene_id, count)
    for line in &attributed {
        match run_speaker {
            Some((prev, scene, count)) if prev == line.speaker && scene == line.scene_id => {
                run_speaker = Some((prev, scene, count + 1));
            }
            Some((prev, scene, count)) => {
                if count >= 5 {
                    monologues.push(Monologue {
                        speaker: prev.to_string(),
                        line_count: count,
                        scene_id: scene,
                    });
                }
                run_speaker = Some((&line.speaker, line.scene_id, 1));
            }
            None => {
                run_speaker = Some((&line.speaker, line.scene_id, 1));
            }
        }
    }
    if let Some((prev, scene, count)) = run_speaker
        && count >= 5
    {
        monologues.push(Monologue {
            speaker: prev.to_string(),
            line_count: count,
            scene_id: scene,
        });
    }

    let mut dominators_vec: Vec<String> = dominators.into_iter().collect();
    dominators_vec.sort();

    ConversationFlow {
        edges,
        dominators: dominators_vec,
        never_initiators,
        monologues,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_empty_input() {
        let result = extract_dialogue("", 0, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_basic_attribution() {
        let text = "\u{201c}Hello,\u{201d} Alice said.";
        let known = chars(&["Alice"]);
        let lines = extract_dialogue(text, 0, &known);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].speaker, "Alice");
        assert_eq!(lines[0].method, "explicit_tag");
        assert!((lines[0].confidence - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_alternation_pattern() {
        // First two lines have explicit tags, third relies on alternation
        let text = "\u{201c}Hello,\u{201d} Alice said.\n\
                     \u{201c}Hi there,\u{201d} Bob replied.\n\
                     \u{201c}How are you?\u{201d}";
        let known = chars(&["Alice", "Bob"]);
        let lines = extract_dialogue(text, 1, &known);

        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].speaker, "Alice");
        assert_eq!(lines[1].speaker, "Bob");
        // Third line should be attributed to Alice via alternation
        assert_eq!(lines[2].speaker, "Alice");
        assert_eq!(lines[2].method, "pattern");
        assert!((lines[2].confidence - 0.45).abs() < 0.01);
    }

    #[test]
    fn test_unknown_speaker() {
        let text = "\u{201c}Who said this?\u{201d}";
        let known = chars(&["Alice", "Bob"]);
        let lines = extract_dialogue(text, 0, &known);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].speaker, "Unknown");
    }

    #[test]
    fn test_analyze_dialogue_patterns() {
        let lines = vec![
            DialogueLine {
                scene_id: 1,
                quote: "Hello there friend".into(),
                speaker: "Alice".into(),
                confidence: 0.95,
                method: "explicit_tag".into(),
                tag_verb: Some("said".into()),
            },
            DialogueLine {
                scene_id: 1,
                quote: "Hi!".into(),
                speaker: "Bob".into(),
                confidence: 0.95,
                method: "explicit_tag".into(),
                tag_verb: Some("replied".into()),
            },
            DialogueLine {
                scene_id: 1,
                quote: "How are you?".into(),
                speaker: "Alice".into(),
                confidence: 0.45,
                method: "pattern".into(),
                tag_verb: None,
            },
        ];

        let known = chars(&["Alice", "Bob"]);
        let analysis = analyze_dialogue_patterns(&lines, &known);

        assert_eq!(analysis.total_lines, 3);
        assert_eq!(analysis.attributed_lines, 3);
        assert!((analysis.attribution_rate - 1.0).abs() < 0.001);
        assert_eq!(analysis.most_talkative, "Alice");
        assert_eq!(analysis.per_character["Alice"].line_count, 2);
        assert_eq!(analysis.per_character["Bob"].line_count, 1);
        assert!(analysis.per_character["Alice"].question_rate > 0.0);
        assert!(analysis.per_character["Bob"].exclamation_rate > 0.0);
        assert!(analysis.dialogue_distribution.contains_key("Alice"));
        assert!(analysis.dialogue_distribution.contains_key("Bob"));
    }

    #[test]
    fn test_straight_quotes() {
        let text = r#""Good morning," Alice said. "How are you?" Bob asked."#;
        let known = chars(&["Alice", "Bob"]);
        let lines = extract_dialogue(text, 0, &known);
        assert!(!lines.is_empty());
        assert!(lines.iter().any(|l| l.speaker == "Alice"));
    }

    #[test]
    fn test_indirect_speech_detection() {
        let text = "She told him that the door was locked. Alice warned them about the danger.";
        let known = chars(&["Alice", "Bob"]);
        let lines = extract_dialogue(text, 0, &known);

        // Only "Alice warned them" should match (She is not a known character)
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].speaker, "Alice");
        assert_eq!(lines[0].method, "indirect_speech");
        assert!((lines[0].confidence - 0.80).abs() < 0.01);
    }

    #[test]
    fn test_voice_profiles_computed() {
        let lines = vec![
            DialogueLine {
                scene_id: 1,
                quote: "I don't think we should go there, do you?".into(),
                speaker: "Alice".into(),
                confidence: 0.95,
                method: "explicit_tag".into(),
                tag_verb: Some("said".into()),
            },
            DialogueLine {
                scene_id: 1,
                quote: "Indeed, I believe we absolutely must proceed forthwith.".into(),
                speaker: "Bob".into(),
                confidence: 0.95,
                method: "explicit_tag".into(),
                tag_verb: Some("declared".into()),
            },
        ];
        let known = chars(&["Alice", "Bob"]);
        let analysis = analyze_dialogue_patterns(&lines, &known);

        assert!(analysis.voice_profiles.contains_key("Alice"));
        assert!(analysis.voice_profiles.contains_key("Bob"));

        let alice = &analysis.voice_profiles["Alice"];
        let bob = &analysis.voice_profiles["Bob"];
        // Alice uses a contraction, so lower formality
        assert!(alice.formality_score < bob.formality_score);
        // Alice asks a question
        assert!(alice.question_frequency > 0.0);
        assert_eq!(bob.question_frequency, 0.0);
    }

    #[test]
    fn test_same_sounding_pair_detection() {
        // Two characters with nearly identical speech patterns
        let lines: Vec<DialogueLine> = (0..10)
            .flat_map(|i| {
                vec![
                    DialogueLine {
                        scene_id: i,
                        quote: "Hello there my friend.".into(),
                        speaker: "Alice".into(),
                        confidence: 0.95,
                        method: "explicit_tag".into(),
                        tag_verb: Some("said".into()),
                    },
                    DialogueLine {
                        scene_id: i,
                        quote: "Hello there my buddy.".into(),
                        speaker: "Bob".into(),
                        confidence: 0.95,
                        method: "explicit_tag".into(),
                        tag_verb: Some("said".into()),
                    },
                ]
            })
            .collect();

        let known = chars(&["Alice", "Bob"]);
        let analysis = analyze_dialogue_patterns(&lines, &known);
        assert!(
            !analysis.same_sounding_pairs.is_empty(),
            "Alice and Bob with identical patterns should be flagged"
        );
        assert!(analysis.same_sounding_pairs[0].similarity > 0.8);
    }

    #[test]
    fn test_conversation_flow_edges() {
        let lines = vec![
            DialogueLine {
                scene_id: 1,
                quote: "Hello".into(),
                speaker: "Alice".into(),
                confidence: 0.95,
                method: "explicit_tag".into(),
                tag_verb: None,
            },
            DialogueLine {
                scene_id: 1,
                quote: "Hi".into(),
                speaker: "Bob".into(),
                confidence: 0.95,
                method: "explicit_tag".into(),
                tag_verb: None,
            },
            DialogueLine {
                scene_id: 1,
                quote: "How are you?".into(),
                speaker: "Alice".into(),
                confidence: 0.95,
                method: "explicit_tag".into(),
                tag_verb: None,
            },
        ];

        let known = chars(&["Alice", "Bob"]);
        let analysis = analyze_dialogue_patterns(&lines, &known);
        let flow = &analysis.conversation_flow;
        assert!(!flow.edges.is_empty());
        // Alice -> Bob and Bob -> Alice edges should exist
        assert!(
            flow.edges
                .iter()
                .any(|e| e.from == "Alice" && e.to == "Bob")
        );
        assert!(
            flow.edges
                .iter()
                .any(|e| e.from == "Bob" && e.to == "Alice")
        );
    }

    #[test]
    fn test_conversation_flow_dominator() {
        // Alice speaks 4 of 5 lines in a scene
        let mut lines = Vec::new();
        for i in 0..4 {
            lines.push(DialogueLine {
                scene_id: 1,
                quote: format!("Line {}", i),
                speaker: "Alice".into(),
                confidence: 0.95,
                method: "explicit_tag".into(),
                tag_verb: None,
            });
        }
        lines.push(DialogueLine {
            scene_id: 1,
            quote: "One line".into(),
            speaker: "Bob".into(),
            confidence: 0.95,
            method: "explicit_tag".into(),
            tag_verb: None,
        });

        let known = chars(&["Alice", "Bob"]);
        let analysis = analyze_dialogue_patterns(&lines, &known);
        assert!(
            analysis
                .conversation_flow
                .dominators
                .contains(&"Alice".to_string())
        );
    }

    #[test]
    fn test_monologue_detection() {
        // Alice speaks 6 consecutive lines
        let lines: Vec<DialogueLine> = (0..6)
            .map(|i| DialogueLine {
                scene_id: 1,
                quote: format!("Line {}", i),
                speaker: "Alice".into(),
                confidence: 0.95,
                method: "explicit_tag".into(),
                tag_verb: None,
            })
            .collect();

        let known = chars(&["Alice"]);
        let analysis = analyze_dialogue_patterns(&lines, &known);
        assert!(!analysis.conversation_flow.monologues.is_empty());
        assert_eq!(analysis.conversation_flow.monologues[0].speaker, "Alice");
        assert_eq!(analysis.conversation_flow.monologues[0].line_count, 6);
    }

    #[test]
    fn test_voice_similarity_identical() {
        let p = VoiceProfile {
            avg_word_length: 5.0,
            vocabulary_richness: 0.7,
            question_frequency: 0.2,
            avg_line_length: 10.0,
            formality_score: 0.8,
            tag_variety_score: 0.5,
            speech_verbs_used: vec![],
        };
        let sim = voice_similarity(&p, &p);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_voice_similarity_different() {
        let a = VoiceProfile {
            avg_word_length: 3.0,
            vocabulary_richness: 0.3,
            question_frequency: 0.8,
            avg_line_length: 5.0,
            formality_score: 0.2,
            tag_variety_score: 0.5,
            speech_verbs_used: vec![],
        };
        let b = VoiceProfile {
            avg_word_length: 8.0,
            vocabulary_richness: 0.9,
            question_frequency: 0.0,
            avg_line_length: 40.0,
            formality_score: 1.0,
            tag_variety_score: 0.5,
            speech_verbs_used: vec![],
        };
        let sim = voice_similarity(&a, &b);
        assert!(
            sim < 0.8,
            "Very different profiles should have low similarity: {}",
            sim
        );
    }

    /// A line like `"...," she whispered.` should yield `whispered` in
    /// `speech_verbs_used` for the attributed character.
    #[test]
    fn test_speech_verbs_used_from_tag() {
        let text = r#""I have a secret," she whispered."#;
        // Use a known character so attribution succeeds
        let known = chars(&["Alice"]);
        // Manually build a DialogueLine as attribution would produce
        let lines = vec![DialogueLine {
            scene_id: 0,
            quote: "I have a secret,".into(),
            speaker: "Alice".into(),
            confidence: 0.95,
            method: "explicit_tag".into(),
            tag_verb: Some("whispered".into()),
        }];
        let analysis = analyze_dialogue_patterns(&lines, &known);
        let profile = &analysis.voice_profiles["Alice"];
        assert!(
            profile.speech_verbs_used.contains(&"whispered".to_string()),
            "speech_verbs_used should contain 'whispered', got: {:?}",
            profile.speech_verbs_used
        );
        // Should NOT contain unrelated narration verbs
        assert!(
            !profile.speech_verbs_used.contains(&"said".to_string()),
            "speech_verbs_used should not contain 'said' when it wasn't used"
        );
        // Suppress unused variable warning
        let _ = text;
    }
}
