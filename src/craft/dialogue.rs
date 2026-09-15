/// Dialogue extraction and attribution: identifies who said what.
///
/// Ports the Python `DialogueAttributor` as standalone functions. Attributes
/// dialogue lines to characters via speech tags, proximity, continuation,
/// and alternation patterns.
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use log::debug;
use regex::Regex;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Constants & compiled patterns
// ---------------------------------------------------------------------------

const WINDOW: usize = 80;

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

fn verb_alternation() -> &'static str {
    static INST: OnceLock<String> = OnceLock::new();
    INST.get_or_init(|| {
        speech_verbs()
            .iter()
            .map(|v| regex::escape(v))
            .collect::<Vec<_>>()
            .join("|")
    })
}

/// Curly double, straight double, curly single quote patterns.
fn quote_patterns() -> &'static [Regex; 3] {
    static INST: OnceLock<[Regex; 3]> = OnceLock::new();
    INST.get_or_init(|| {
        [
            Regex::new(r"\u{201c}((?:[^\u{201d}]|\n)*?)\u{201d}").unwrap(),
            Regex::new(r#""((?:[^"]|\n)*?)""#).unwrap(),
            Regex::new(r"\u{2018}((?:[^\u{2019}]|\n)*?)\u{2019}").unwrap(),
        ]
    })
}

/// "verb NAME" pattern (generic fallback).
fn tag_before_re() -> &'static Regex {
    static INST: OnceLock<Regex> = OnceLock::new();
    INST.get_or_init(|| {
        let pat = format!(
            r"\b({})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            verb_alternation()
        );
        Regex::new(&pat).unwrap()
    })
}

/// "NAME verb" pattern (generic fallback).
fn tag_after_re() -> &'static Regex {
    static INST: OnceLock<Regex> = OnceLock::new();
    INST.get_or_init(|| {
        let pat = format!(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+({})\b",
            verb_alternation()
        );
        Regex::new(&pat).unwrap()
    })
}

/// Indirect speech pattern: "Name told/informed/... him/her/them that ..."
fn indirect_speech_re() -> &'static Regex {
    static INST: OnceLock<Regex> = OnceLock::new();
    INST.get_or_init(|| {
        Regex::new(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:told|informed|explained|mentioned|reminded|assured|convinced|warned)\s+(?:him|her|them|me|us)\b"
        ).unwrap()
    })
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
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
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SameSoundingPair {
    pub character_a: String,
    pub character_b: String,
    pub similarity: f64,
}

/// Directed edge in the conversation flow graph.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationEdge {
    pub from: String,
    pub to: String,
    pub count: usize,
}

/// A detected monologue (5+ consecutive lines from one speaker).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Monologue {
    pub speaker: String,
    pub line_count: usize,
    pub scene_id: usize,
}

/// Conversation flow analysis results.
#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ConversationFlow {
    pub edges: Vec<ConversationEdge>,
    pub dominators: Vec<String>,
    pub never_initiators: Vec<String>,
    pub monologues: Vec<Monologue>,
}

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
    "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't",
    "shouldn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
    "hadn't", "i'm", "i've", "i'll", "i'd", "you're", "you've", "you'll",
    "you'd", "he's", "she's", "it's", "we're", "we've", "we'll", "we'd",
    "they're", "they've", "they'll", "they'd", "that's", "there's", "here's",
    "what's", "who's", "how's", "let's", "that'll", "there'll",
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
    if scene_text.is_empty() {
        return Vec::new();
    }
    debug!(
        "[dialogue] extract_dialogue scene={} — text_len={}, known_chars={}",
        scene_id,
        scene_text.len(),
        known_characters.len()
    );

    // Detect indirect speech before processing quotes
    let mut indirect_lines = detect_indirect_speech(scene_text, scene_id, known_characters);

    let raw_quotes = find_quotes(scene_text);
    if raw_quotes.is_empty() && indirect_lines.is_empty() {
        return Vec::new();
    }

    let mut results = Vec::new();
    let mut prev_speaker: Option<String> = None;
    let mut scene_speakers: Vec<String> = Vec::new();

    for (quote_text, start, end) in &raw_quotes {
        let (speaker, confidence, method, tag_verb) = attribute_quote(
            scene_text,
            quote_text,
            *start,
            *end,
            known_characters,
            prev_speaker.as_deref(),
            &scene_speakers,
        );

        results.push(DialogueLine {
            scene_id,
            quote: quote_text.clone(),
            speaker: speaker.clone(),
            confidence,
            method,
            tag_verb,
        });

        if speaker != "Unknown" {
            prev_speaker = Some(speaker.clone());
            if !scene_speakers.contains(&speaker) {
                scene_speakers.push(speaker);
                if scene_speakers.len() > 3 {
                    scene_speakers.remove(0);
                }
            }
        }
    }

    results.append(&mut indirect_lines);
    results
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
            .flat_map(|q| q.quote.split_whitespace())
            .map(|w| {
                w.to_lowercase()
                    .trim_matches(&['.', ',', '!', '?', ';', ':', '"', '\''][..])
                    .to_string()
            })
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
        .max_by_key(|(_, v)| v.line_count)
        .map(|(c, _)| c.clone())
        .unwrap_or_default();

    let most_verbose = per_character
        .iter()
        .max_by(|(_, a), (_, b)| {
            a.avg_line_length
                .partial_cmp(&b.avg_line_length)
                .unwrap_or(std::cmp::Ordering::Equal)
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

/// Find all quoted text spans, deduplicated by position.
fn find_quotes(text: &str) -> Vec<(String, usize, usize)> {
    let mut found: HashMap<(usize, usize), (String, usize, usize)> = HashMap::new();

    for pat in quote_patterns() {
        for caps in pat.captures_iter(text) {
            if let Some(full) = caps.get(0) {
                let key = (full.start(), full.end());
                if !found.contains_key(&key)
                    && let Some(content_match) = caps.get(1) {
                        let content = content_match.as_str().trim();
                        if !content.is_empty() {
                            found.insert(key, (content.to_string(), full.start(), full.end()));
                        }
                    }
            }
        }
    }

    let mut result: Vec<(String, usize, usize)> = found.into_values().collect();
    result.sort_by_key(|(_, start, _)| *start);
    result
}

/// Extract the first speech verb found in a text window.
fn extract_verb_from_window(text: &str) -> Option<String> {
    let verb_re = speech_verb_re();
    verb_re
        .find(text)
        .map(|m| m.as_str().to_lowercase())
}

/// Attribute a single quote to a speaker.
/// Returns (speaker, confidence, method, tag_verb).
fn attribute_quote(
    scene_text: &str,
    _quote: &str,
    start: usize,
    end: usize,
    known_characters: &[String],
    prev_speaker: Option<&str>,
    prev_recent: &[String],
) -> (String, f64, String, Option<String>) {
    // 1. Explicit tag: limit to same paragraph
    let after_raw = crate::substrate::common_types::safe_substr(scene_text, end, end + WINDOW);
    let after_text = after_raw.split('\n').next().unwrap_or("");
    let before_start = start.saturating_sub(WINDOW);
    let before_raw = crate::substrate::common_types::safe_substr(scene_text, before_start, start);
    let before_text = before_raw.rsplit('\n').next().unwrap_or("");

    if let Some(speaker) = match_speech_tag(after_text, known_characters) {
        let verb = extract_verb_from_window(after_text);
        return (speaker, 0.95, "explicit_tag".into(), verb);
    }
    if let Some(speaker) = match_speech_tag(before_text, known_characters) {
        let verb = extract_verb_from_window(before_text);
        return (speaker, 0.90, "explicit_tag".into(), verb);
    }

    // 2. Proximity: character name in adjacent text
    let context = format!("{} {}", before_text, after_text);
    if let Some(speaker) = match_proximity(&context, known_characters) {
        return (speaker, 0.65, "proximity".into(), None);
    }

    // 3. Continuation: same speaker if line break before quote
    if let Some(prev) = prev_speaker
        && start > 0 {
            let pre_start = start.saturating_sub(3);
            let preceding = crate::substrate::common_types::safe_substr(scene_text, pre_start, start).trim();
            if preceding.ends_with('\n') || preceding.is_empty() {
                return (prev.to_string(), 0.50, "continuation".into(), None);
            }
        }

    // 4. Alternation pattern (uses up to 3 recent speakers)
    if let Some(prev) = prev_speaker
        && prev_recent.len() >= 2
            && let Some(other) = prev_recent.iter().rev().find(|s| s.as_str() != prev) {
                return (other.clone(), 0.45, "pattern".into(), None);
            }

    ("Unknown".into(), 0.0, "pattern".into(), None)
}

/// Detect indirect speech patterns and attribute them to speakers.
fn detect_indirect_speech(
    scene_text: &str,
    scene_id: usize,
    known_characters: &[String],
) -> Vec<DialogueLine> {
    let mut results = Vec::new();
    let re = indirect_speech_re();
    for caps in re.captures_iter(scene_text) {
        if let (Some(name_match), Some(full)) = (caps.get(1), caps.get(0)) {
            let name = name_match.as_str();
            if let Some(resolved) = resolve_name(name, known_characters) {
                results.push(DialogueLine {
                    scene_id,
                    quote: full.as_str().to_string(),
                    speaker: resolved,
                    confidence: 0.80,
                    method: "indirect_speech".into(),
                    tag_verb: None,
                });
            }
        }
    }
    results
}

fn speech_verb_re() -> &'static Regex {
    static INST: OnceLock<Regex> = OnceLock::new();
    INST.get_or_init(|| {
        let pat = format!(r"\b(?:{})\b", verb_alternation());
        Regex::new(&pat).unwrap()
    })
}

fn match_speech_tag(text: &str, known_characters: &[String]) -> Option<String> {
    let verb_re = speech_verb_re();

    // Check known character names near speech verbs using string matching
    // instead of compiling a regex per character per call.
    let text_lower = text.to_lowercase();
    for char_name in known_characters {
        let name_lower = char_name.to_lowercase();
        // Check if character name appears in the text (case-insensitive word boundary check)
        if let Some(name_pos) = text_lower.find(&name_lower) {
            let name_end = name_pos + name_lower.len();
            // Verify word boundaries
            let at_word_start =
                name_pos == 0 || !text.as_bytes()[name_pos - 1].is_ascii_alphanumeric();
            let at_word_end =
                name_end >= text.len() || !text.as_bytes()[name_end].is_ascii_alphanumeric();
            if at_word_start && at_word_end {
                // Check if a speech verb appears nearby in the text
                if verb_re.is_match(text) {
                    return Some(char_name.clone());
                }
            }
        }
    }

    // Fallback to generic regex
    for caps in tag_before_re().captures_iter(text) {
        if let Some(name) = caps.get(2)
            && let Some(resolved) = resolve_name(name.as_str(), known_characters) {
                return Some(resolved);
            }
    }
    for caps in tag_after_re().captures_iter(text) {
        if let Some(name) = caps.get(1)
            && let Some(resolved) = resolve_name(name.as_str(), known_characters) {
                return Some(resolved);
            }
    }

    None
}

fn match_proximity(text: &str, known_characters: &[String]) -> Option<String> {
    let text_lower = text.to_lowercase();
    for char_name in known_characters {
        let name_lower = char_name.to_lowercase();
        if let Some(pos) = text_lower.find(&name_lower) {
            let end = pos + name_lower.len();
            let at_start = pos == 0 || !text.as_bytes()[pos - 1].is_ascii_alphanumeric();
            let at_end = end >= text.len() || !text.as_bytes()[end].is_ascii_alphanumeric();
            if at_start && at_end {
                return Some(char_name.clone());
            }
        }
    }
    None
}

fn resolve_name(name: &str, known_characters: &[String]) -> Option<String> {
    let name_lower = name.to_lowercase();
    for char_name in known_characters {
        if char_name.to_lowercase() == name_lower {
            return Some(char_name.clone());
        }
        // Check if name matches first name of a known character
        let parts: Vec<&str> = char_name.split_whitespace().collect();
        if let Some(first) = parts.first()
            && first.to_lowercase() == name_lower
        {
            return Some(char_name.clone());
        }
    }
    None
}

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
            .flat_map(|q| q.quote.split_whitespace())
            .map(|w| {
                w.to_lowercase()
                    .trim_matches(&['.', ',', '!', '?', ';', ':', '"', '\''][..])
                    .to_string()
            })
            .filter(|w| !w.is_empty())
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
        && count >= 5 {
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
        assert!(flow
            .edges
            .iter()
            .any(|e| e.from == "Alice" && e.to == "Bob"));
        assert!(flow
            .edges
            .iter()
            .any(|e| e.from == "Bob" && e.to == "Alice"));
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
        assert!(analysis
            .conversation_flow
            .dominators
            .contains(&"Alice".to_string()));
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