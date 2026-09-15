use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::craft::lexicon::WordMatcher;

static MALE_PRONOUNS: &[&str] = &["he", "him", "his", "himself"];
static FEMALE_PRONOUNS: &[&str] = &["she", "her", "hers", "herself"];
static NEUTRAL_PRONOUNS: &[&str] = &["they", "them", "their", "theirs", "themselves"];

static POSSESSIVE_PRONOUNS: &[&str] = &["his", "her", "their"];
static REFLEXIVE_PRONOUNS: &[&str] = &["himself", "herself", "themselves"];

fn pronoun_matcher() -> &'static WordMatcher {
    static M: OnceLock<WordMatcher> = OnceLock::new();
    M.get_or_init(|| {
        let mut all: Vec<&str> = Vec::new();
        all.extend(MALE_PRONOUNS);
        all.extend(FEMALE_PRONOUNS);
        all.extend(NEUTRAL_PRONOUNS);
        WordMatcher::new(&all)
    })
}

fn capitalized_word_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b([A-Z][a-z]+)\b").expect("invalid capitalized word regex"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gender {
    Male,
    Female,
    Neutral,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PronounKind {
    Subject,              // he, she, they
    Object,               // him, her, them
    Possessive,           // his, her, their
    Reflexive,            // himself, herself, themselves
    PossessiveStandalone, // hers, theirs
}

fn pronoun_gender(token: &str) -> Gender {
    let low = token.to_lowercase();
    if MALE_PRONOUNS.contains(&low.as_str()) {
        Gender::Male
    } else if FEMALE_PRONOUNS.contains(&low.as_str()) {
        Gender::Female
    } else {
        Gender::Neutral
    }
}

fn pronoun_kind(token: &str) -> PronounKind {
    let low = token.to_lowercase();
    if REFLEXIVE_PRONOUNS.contains(&low.as_str()) {
        PronounKind::Reflexive
    } else if POSSESSIVE_PRONOUNS.contains(&low.as_str()) {
        PronounKind::Possessive
    } else if ["hers", "theirs"].contains(&low.as_str()) {
        PronounKind::PossessiveStandalone
    } else if ["him", "her", "them"].contains(&low.as_str()) {
        // "her" is ambiguous (object or possessive); if followed by a noun-like word
        // we treat as possessive, but that logic is handled in resolution
        PronounKind::Object
    } else {
        PronounKind::Subject
    }
}

fn gender_from_str(s: &str) -> Gender {
    match s {
        "male" => Gender::Male,
        "female" => Gender::Female,
        "neutral" => Gender::Neutral,
        _ => Gender::Unknown,
    }
}

/// Find the sentence containing the given byte position.
fn find_sentence_bounds(text: &str, pos: usize) -> (usize, usize) {
    let bytes = text.as_bytes();
    let mut start = pos;
    while start > 0 {
        if matches!(bytes[start - 1], b'.' | b'!' | b'?') {
            break;
        }
        start -= 1;
    }
    // Skip whitespace after sentence-ending punctuation
    while start < text.len() && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    let mut end = pos;
    while end < text.len() {
        if matches!(bytes[end], b'.' | b'!' | b'?') {
            end += 1;
            break;
        }
        end += 1;
    }
    (start, end)
}

/// For reflexive pronouns, find the likely subject of the sentence: the first
/// capitalized word that is not at the very start of the sentence (to skip
/// sentence-initial capitalization of common words).
fn find_sentence_subject<'a>(
    sentence: &str,
    known_characters: &[&'a str],
    char_lower_map: &HashMap<String, &'a str>,
) -> Option<&'a str> {
    let cap_re = capitalized_word_regex();
    for m in cap_re.find_iter(sentence) {
        let word = m.as_str();
        let lower = word.to_lowercase();
        if let Some(&name) = char_lower_map.get(&lower) {
            return Some(name);
        }
        // Check if any known character starts with this word
        for &ch in known_characters {
            if ch.eq_ignore_ascii_case(word) {
                return Some(ch);
            }
        }
    }
    None
}

/// Confidence tiers for resolution quality.
fn compute_confidence(
    kind: PronounKind,
    exact_gender: bool,
    same_paragraph: bool,
    distance: usize,
    is_cataphoric: bool,
) -> f64 {
    if is_cataphoric {
        return 0.30;
    }
    match kind {
        PronounKind::Possessive | PronounKind::Reflexive => 0.40,
        _ => {
            if exact_gender && same_paragraph && distance < 80 {
                0.85
            } else if exact_gender && distance < 200 {
                0.70
            } else if !exact_gender && same_paragraph {
                0.55
            } else if exact_gender {
                0.45
            } else {
                0.35
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mention {
    pub text: String,
    pub resolved_to: Option<String>,
    pub start: usize,
    pub is_pronoun: bool,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorefResult {
    pub mentions: Vec<Mention>,
    pub character_mentions: HashMap<String, usize>,
    pub pronoun_resolution_rate: f64,
}

pub struct CoreferenceResolver;

impl Default for CoreferenceResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl CoreferenceResolver {
    pub fn new() -> Self {
        Self
    }

    /// Scan multiple scenes to infer gender for each character.
    pub fn build_character_gender_map(
        &self,
        scenes: &[&str],
        characters: &[&str],
    ) -> HashMap<String, String> {
        let window: usize = 200;
        let pronoun_m = pronoun_matcher();

        let char_patterns: Vec<(&str, Regex)> = characters
            .iter()
            .map(|c| (*c, crate::substrate::text::word_boundary_regex(c)))
            .collect();

        let mut counts: HashMap<&str, (u32, u32)> =
            characters.iter().map(|c| (*c, (0, 0))).collect();

        for scene in scenes {
            for (char_name, pat) in &char_patterns {
                for m in pat.find_iter(scene) {
                    let idx = m.start();
                    // Window offsets are byte counts and can land inside a multibyte
                    // char on real manuscripts (smart punctuation, stray C1 bytes), so
                    // snap to char boundaries before slicing — slicing mid-char panics.
                    let ctx_start =
                        crate::substrate::text::floor_char_boundary(scene, idx.saturating_sub(window));
                    let ctx_end = crate::substrate::text::ceil_char_boundary(scene, idx + char_name.len() + window);
                    let ctx = &scene[ctx_start..ctx_end].to_lowercase();
                    for pm in pronoun_m.find_iter(ctx) {
                        let g = pronoun_gender(pm.text);
                        if let Some(entry) = counts.get_mut(char_name) {
                            match g {
                                Gender::Male => entry.0 += 1,
                                Gender::Female => entry.1 += 1,
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        let mut result = HashMap::new();
        for &c in characters {
            let (mc, fc) = counts[c];
            let total = mc + fc;
            let gender = if total < 3 {
                "unknown"
            } else if mc as f64 / total as f64 > 0.65 {
                "male"
            } else if fc as f64 / total as f64 > 0.65 {
                "female"
            } else {
                "unknown"
            };
            result.insert(c.to_string(), gender.to_string());
        }
        result
    }

    /// Resolve pronoun references to character names within a scene.
    pub fn resolve(
        &self,
        scene_text: &str,
        known_characters: &[&str],
        gender_map: Option<&HashMap<String, String>>,
    ) -> CorefResult {
        let owned_map;
        let gender_map = match gender_map {
            Some(m) => m,
            None => {
                owned_map = self.build_character_gender_map(&[scene_text], known_characters);
                &owned_map
            }
        };

        let text_lower = scene_text.to_lowercase();

        // Build a lowercase -> original name map for subject detection
        let char_lower_map: HashMap<String, &str> = known_characters
            .iter()
            .map(|&c| (c.to_lowercase(), c))
            .collect();

        // 1. Find all character name occurrences
        let mut name_mentions: Vec<Mention> = Vec::new();
        for &ch in known_characters {
            let pat = crate::substrate::text::word_boundary_regex(ch);
            for m in pat.find_iter(&text_lower) {
                name_mentions.push(Mention {
                    // `m` indexes `text_lower`; lowercasing can shift byte
                    // offsets (Turkish İ, German ß), so prefer the original
                    // text but fall back to the always-valid lowercase span.
                    text: scene_text
                        .get(m.start()..m.end())
                        .unwrap_or(&text_lower[m.start()..m.end()])
                        .to_string(),
                    resolved_to: Some(ch.to_string()),
                    start: m.start(),
                    is_pronoun: false,
                    confidence: 0.95,
                });
            }
        }
        name_mentions.sort_by_key(|m| m.start);

        // 2. Find all pronouns
        let pronoun_m = pronoun_matcher();
        let mut pronoun_mentions: Vec<Mention> = Vec::new();
        for m in pronoun_m.find_iter(scene_text) {
            pronoun_mentions.push(Mention {
                text: m.text.to_string(),
                resolved_to: None,
                start: m.start,
                is_pronoun: true,
                confidence: 0.0,
            });
        }

        // Paragraph boundary detection
        let para_re = Regex::new(r"\n\s*\n").expect("invalid para regex");
        let para_breaks: Vec<usize> = para_re.find_iter(scene_text).map(|m| m.start()).collect();

        let same_para = |a: usize, b: usize| -> bool {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            !para_breaks.iter().any(|&brk| lo < brk && brk < hi)
        };

        // 3. Resolve each pronoun
        for pm in &mut pronoun_mentions {
            let p_gender = pronoun_gender(&pm.text);
            let p_kind = pronoun_kind(&pm.text);
            let pos = pm.start;

            // they/them: only resolve with a single unknown-gender character
            if p_gender == Gender::Neutral {
                let unk: Vec<&&str> = known_characters
                    .iter()
                    .filter(|c| gender_map.get(**c).is_none_or(|g| g == "unknown"))
                    .collect();
                if unk.len() == 1 {
                    pm.resolved_to = Some(unk[0].to_string());
                    pm.confidence = 0.40;
                }
                continue;
            }

            // Reflexive: resolve to the subject of the current sentence
            if p_kind == PronounKind::Reflexive {
                let (sent_start, sent_end) = find_sentence_bounds(scene_text, pos);
                let sentence = &scene_text[sent_start..sent_end];
                if let Some(subj) =
                    find_sentence_subject(sentence, known_characters, &char_lower_map)
                {
                    let subj_gender = gender_from_str(
                        gender_map
                            .get(subj)
                            .map(|s| s.as_str())
                            .unwrap_or("unknown"),
                    );
                    if subj_gender == p_gender || subj_gender == Gender::Unknown {
                        pm.resolved_to = Some(subj.to_string());
                        pm.confidence = 0.40;
                        continue;
                    }
                }
                // Fall through to general resolution if reflexive heuristic fails
            }

            // General backward resolution
            let mut best: Option<usize> = None;
            let mut best_rank: u8 = 4;
            let mut best_dist: usize = usize::MAX;

            for (i, nm) in name_mentions.iter().enumerate() {
                if nm.start >= pos {
                    break;
                }
                let cg = gender_from_str(
                    gender_map
                        .get(nm.resolved_to.as_deref().unwrap_or(""))
                        .map(|s| s.as_str())
                        .unwrap_or("unknown"),
                );
                if cg != Gender::Unknown && cg != p_gender {
                    continue;
                }
                let dist = pos - nm.start;
                let sp = same_para(nm.start, pos);
                if dist > 300 && !sp {
                    continue;
                }
                let exact = cg == p_gender;
                let rank: u8 = if exact && sp {
                    0
                } else if exact {
                    1
                } else if sp {
                    2
                } else {
                    3
                };
                if rank < best_rank || (rank == best_rank && dist < best_dist) {
                    best = Some(i);
                    best_rank = rank;
                    best_dist = dist;
                }
            }

            if let Some(idx) = best {
                if let Some(resolved_name) = name_mentions[idx].resolved_to.clone() {
                    let exact_gender = gender_from_str(
                        gender_map
                            .get(&resolved_name)
                            .map(|s| s.as_str())
                            .unwrap_or("unknown"),
                    ) == p_gender;
                    let sp = same_para(name_mentions[idx].start, pos);
                    pm.resolved_to = Some(resolved_name);
                    pm.confidence = compute_confidence(p_kind, exact_gender, sp, best_dist, false);
                }
            } else {
                // Cataphoric fallback: look forward up to 100 chars for a name
                let forward_limit = (pos + 100).min(scene_text.len());
                let mut cat_best: Option<usize> = None;
                let mut cat_dist: usize = usize::MAX;

                for (i, nm) in name_mentions.iter().enumerate() {
                    if nm.start <= pos {
                        continue;
                    }
                    if nm.start > forward_limit {
                        break;
                    }
                    let cg = gender_from_str(
                        gender_map
                            .get(nm.resolved_to.as_deref().unwrap_or(""))
                            .map(|s| s.as_str())
                            .unwrap_or("unknown"),
                    );
                    if cg != Gender::Unknown && cg != p_gender {
                        continue;
                    }
                    let dist = nm.start - pos;
                    if dist < cat_dist {
                        cat_best = Some(i);
                        cat_dist = dist;
                    }
                }

                if let Some(idx) = cat_best
                    && let Some(resolved_name) = name_mentions[idx].resolved_to.clone()
                {
                    pm.resolved_to = Some(resolved_name);
                    pm.confidence = 0.30;
                }
            }
        }

        // Combine results
        let mut all_mentions: Vec<Mention> = Vec::new();
        all_mentions.extend(name_mentions);
        all_mentions.extend(pronoun_mentions.iter().cloned());
        all_mentions.sort_by_key(|m| m.start);

        let mut char_counts: HashMap<String, usize> = known_characters
            .iter()
            .map(|c| (c.to_string(), 0))
            .collect();
        for m in &all_mentions {
            if let Some(ref resolved) = m.resolved_to
                && let Some(count) = char_counts.get_mut(resolved) {
                    *count += 1;
                }
        }

        let total = pronoun_mentions.len();
        let resolved_count = pronoun_mentions
            .iter()
            .filter(|p| p.resolved_to.is_some())
            .count();
        let rate = if total == 0 {
            1.0
        } else {
            resolved_count as f64 / total as f64
        };

        CorefResult {
            mentions: all_mentions,
            character_mentions: char_counts,
            pronoun_resolution_rate: rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let resolver = CoreferenceResolver::new();
        let result = resolver.resolve("", &[], None);
        assert!(result.mentions.is_empty());
        assert_eq!(result.pronoun_resolution_rate, 1.0);
    }

    #[test]
    fn test_basic_he_resolution() {
        let resolver = CoreferenceResolver::new();
        let text = "John walked to the store. He bought some milk.";
        let mut gender_map = HashMap::new();
        gender_map.insert("John".to_string(), "male".to_string());
        let result = resolver.resolve(text, &["John"], Some(&gender_map));

        let he_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "he")
            .expect("should find 'He' pronoun");
        assert_eq!(he_mention.resolved_to.as_deref(), Some("John"));
        assert!(he_mention.confidence > 0.0);
        assert!(result.pronoun_resolution_rate > 0.0);
    }

    #[test]
    fn test_gender_inference() {
        let resolver = CoreferenceResolver::new();
        let scenes = vec![
            "John picked up his bag. He walked to the door. His coat was heavy.",
            "John said he would return. He promised his friend.",
        ];
        let map = resolver.build_character_gender_map(&scenes, &["John"]);
        assert_eq!(map.get("John").map(|s| s.as_str()), Some("male"));
    }

    #[test]
    fn test_paragraph_boundary() {
        let resolver = CoreferenceResolver::new();
        let text = "John entered the room.\n\nMary smiled. She waved.";
        let mut gender_map = HashMap::new();
        gender_map.insert("John".to_string(), "male".to_string());
        gender_map.insert("Mary".to_string(), "female".to_string());
        let result = resolver.resolve(text, &["John", "Mary"], Some(&gender_map));

        let she_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "she")
            .expect("should find 'She' pronoun");
        assert_eq!(she_mention.resolved_to.as_deref(), Some("Mary"));
    }

    #[test]
    fn test_they_them_neutral_resolution() {
        let resolver = CoreferenceResolver::new();
        let text = "Alex walked in. They sat down.";
        let mut gender_map = HashMap::new();
        gender_map.insert("Alex".to_string(), "unknown".to_string());
        let result = resolver.resolve(text, &["Alex"], Some(&gender_map));

        let they_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "they")
            .expect("should find 'They' pronoun");
        assert_eq!(they_mention.resolved_to.as_deref(), Some("Alex"));
        assert!((they_mention.confidence - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_possessive_resolution() {
        let resolver = CoreferenceResolver::new();
        let text = "Mary put down her book. John picked up his sword.";
        let mut gender_map = HashMap::new();
        gender_map.insert("Mary".to_string(), "female".to_string());
        gender_map.insert("John".to_string(), "male".to_string());
        let result = resolver.resolve(text, &["Mary", "John"], Some(&gender_map));

        let her_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "her")
            .expect("should find 'her' pronoun");
        assert_eq!(her_mention.resolved_to.as_deref(), Some("Mary"));
        assert!((her_mention.confidence - 0.40).abs() < f64::EPSILON);

        let his_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "his")
            .expect("should find 'his' pronoun");
        assert_eq!(his_mention.resolved_to.as_deref(), Some("John"));
        assert!((his_mention.confidence - 0.40).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reflexive_resolution() {
        let resolver = CoreferenceResolver::new();
        let text = "John looked at himself in the mirror.";
        let mut gender_map = HashMap::new();
        gender_map.insert("John".to_string(), "male".to_string());
        let result = resolver.resolve(text, &["John"], Some(&gender_map));

        let himself_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "himself")
            .expect("should find 'himself' pronoun");
        assert_eq!(himself_mention.resolved_to.as_deref(), Some("John"));
        assert!((himself_mention.confidence - 0.40).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cataphoric_resolution() {
        let resolver = CoreferenceResolver::new();
        let text = "When he arrived, John was tired.";
        let mut gender_map = HashMap::new();
        gender_map.insert("John".to_string(), "male".to_string());
        let result = resolver.resolve(text, &["John"], Some(&gender_map));

        let he_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "he")
            .expect("should find 'he' pronoun");
        assert_eq!(he_mention.resolved_to.as_deref(), Some("John"));
        assert!((he_mention.confidence - 0.30).abs() < f64::EPSILON);
    }

    #[test]
    fn test_confidence_tiers() {
        let resolver = CoreferenceResolver::new();
        // Same paragraph, close distance, exact gender match -> 0.85
        let text = "John smiled. He laughed.";
        let mut gender_map = HashMap::new();
        gender_map.insert("John".to_string(), "male".to_string());
        let result = resolver.resolve(text, &["John"], Some(&gender_map));

        let he_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "he")
            .expect("should find 'He' pronoun");
        assert_eq!(he_mention.resolved_to.as_deref(), Some("John"));
        assert!((he_mention.confidence - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_name_mention_confidence() {
        let resolver = CoreferenceResolver::new();
        let text = "John walked to the store.";
        let mut gender_map = HashMap::new();
        gender_map.insert("John".to_string(), "male".to_string());
        let result = resolver.resolve(text, &["John"], Some(&gender_map));

        let name_mention = result
            .mentions
            .iter()
            .find(|m| !m.is_pronoun && m.text == "John")
            .expect("should find 'John' name mention");
        assert!((name_mention.confidence - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cross_paragraph_confidence() {
        let resolver = CoreferenceResolver::new();
        let text = "John entered the room.\n\nHe sat down.";
        let mut gender_map = HashMap::new();
        gender_map.insert("John".to_string(), "male".to_string());
        let result = resolver.resolve(text, &["John"], Some(&gender_map));

        let he_mention = result
            .mentions
            .iter()
            .find(|m| m.is_pronoun && m.text.to_lowercase() == "he")
            .expect("should find 'He' pronoun");
        assert_eq!(he_mention.resolved_to.as_deref(), Some("John"));
        // Cross paragraph, exact gender, distance ~24 chars -> 0.70
        assert!((he_mention.confidence - 0.70).abs() < f64::EPSILON);
    }
}
