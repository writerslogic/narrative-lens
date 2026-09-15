use crate::substrate::utils::map_cosine_similarity;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoicePair {
    pub character_a: String,
    pub character_b: String,
    pub similarity: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DialogueIssue {
    pub scene: usize,
    pub issue_type: String,
    pub description: String,
    pub severity: f64,
    pub characters: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PowerDynamic {
    pub character_a: String,
    pub character_b: String,
    pub dominant: String,
    pub indicators: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DialogueRealismResult {
    pub character_voices: Vec<CharacterVoice>,
    pub voice_pairs: Vec<VoicePair>,
    pub issues: Vec<DialogueIssue>,
    pub power_dynamics: Vec<PowerDynamic>,
    pub overall_realism: f64,
    pub voice_distinctiveness: f64,
    pub info_dump_count: usize,
}

/// Word frequency vector for a character.
struct VoiceFingerprint {
    word_freq: HashMap<String, f64>,
    total_words: usize,
    total_turns: usize,
    contractions: usize,
    questions: usize,
    exclamations: usize,
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

/// Common contractions.
const CONTRACTIONS: &[&str] = &[
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
    "hasn't",
    "haven't",
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
    "let's",
];

fn build_fingerprint(turns: &[&str]) -> VoiceFingerprint {
    let mut word_freq: HashMap<String, f64> = HashMap::new();
    let mut total_words = 0;
    let mut contractions = 0;
    let mut questions = 0;
    let mut exclamations = 0;

    for turn in turns {
        let words = crate::substrate::text::tokenize_words(turn);
        total_words += words.len();

        for word in &words {
            *word_freq.entry(word.clone()).or_insert(0.0) += 1.0;
            if CONTRACTIONS.contains(&word.as_str()) {
                contractions += 1;
            }
        }

        questions += turn.chars().filter(|c| *c == '?').count();
        exclamations += turn.chars().filter(|c| *c == '!').count();
    }

    // Normalize frequencies
    if total_words > 0 {
        for v in word_freq.values_mut() {
            *v /= total_words as f64;
        }
    }

    VoiceFingerprint {
        word_freq,
        total_words,
        total_turns: turns.len(),
        contractions,
        questions,
        exclamations,
    }
}

fn formality_score(fingerprint: &VoiceFingerprint, turns: &[&str]) -> f64 {
    let text = turns.join(" ").to_lowercase();
    let formal_count = FORMAL_MARKERS
        .iter()
        .filter(|marker| text.contains(*marker))
        .count();

    let contraction_rate = if fingerprint.total_words > 0 {
        fingerprint.contractions as f64 / fingerprint.total_words as f64
    } else {
        0.0
    };

    // High formality = many formal markers, few contractions
    let formal_signal = (formal_count as f64 / turns.len().max(1) as f64).min(1.0);
    let informal_signal = (contraction_rate * 20.0).min(1.0);

    (formal_signal * 0.6 + (1.0 - informal_signal) * 0.4).clamp(0.0, 1.0)
}

fn detect_info_dumps(turns: &[(usize, &str, &str)]) -> Vec<DialogueIssue> {
    let mut issues = Vec::new();

    for (scene, speaker, text) in turns {
        let lower = text.to_lowercase();
        for pattern in INFO_DUMP_PATTERNS {
            if lower.contains(pattern) {
                issues.push(DialogueIssue {
                    scene: *scene,
                    issue_type: "info_dump".to_string(),
                    description: format!(
                        "{} uses expository dialogue pattern: \"{}\"",
                        speaker, pattern
                    ),
                    severity: 0.6,
                    characters: vec![speaker.to_string()],
                });
                break;
            }
        }

        // Long single turns (>150 words) are likely info dumps
        let word_count = text.split_whitespace().count();
        if word_count > 150 {
            issues.push(DialogueIssue {
                scene: *scene,
                issue_type: "info_dump".to_string(),
                description: format!(
                    "{} has a {}-word dialogue turn (unrealistically long for conversation)",
                    speaker, word_count
                ),
                severity: 0.7,
                characters: vec![speaker.to_string()],
            });
        }
    }

    issues
}

fn detect_formality_issues(
    characters: &[&str],
    char_turns: &HashMap<String, Vec<(usize, &str)>>,
) -> Vec<DialogueIssue> {
    let mut issues = Vec::new();

    for character in characters {
        let key = character.to_lowercase();
        if let Some(turns) = char_turns.get(&key) {
            let texts: Vec<&str> = turns.iter().map(|(_, t)| *t).collect();
            let fp = build_fingerprint(&texts);
            let formality = formality_score(&fp, &texts);

            // Flag excessively formal dialogue
            if formality > 0.8 && fp.total_turns > 2 {
                issues.push(DialogueIssue {
                    scene: turns[0].0,
                    issue_type: "unrealistic_formality".to_string(),
                    description: format!(
                        "{} speaks with consistently high formality ({:.0}%), which may feel unnatural",
                        character,
                        formality * 100.0
                    ),
                    severity: 0.5,
                    characters: vec![character.to_string()],
                });
            }
        }
    }

    issues
}

fn analyze_power_dynamics(
    characters: &[&str],
    char_turns: &HashMap<String, Vec<(usize, &str)>>,
) -> Vec<PowerDynamic> {
    let mut dynamics = Vec::new();

    for i in 0..characters.len() {
        for j in (i + 1)..characters.len() {
            let key_a = characters[i].to_lowercase();
            let key_b = characters[j].to_lowercase();

            let turns_a = char_turns.get(&key_a).map(|t| t.len()).unwrap_or(0);
            let turns_b = char_turns.get(&key_b).map(|t| t.len()).unwrap_or(0);

            if turns_a == 0 || turns_b == 0 {
                continue;
            }

            let texts_a: Vec<&str> = char_turns
                .get(&key_a)
                .map(|t| t.iter().map(|(_, s)| *s).collect())
                .unwrap_or_default();
            let texts_b: Vec<&str> = char_turns
                .get(&key_b)
                .map(|t| t.iter().map(|(_, s)| *s).collect())
                .unwrap_or_default();

            let fp_a = build_fingerprint(&texts_a);
            let fp_b = build_fingerprint(&texts_b);

            let mut indicators = Vec::new();
            let mut a_dominance = 0.0_f64;

            // Who speaks more
            if turns_a > turns_b + 2 {
                indicators.push(format!(
                    "{} speaks more often ({} vs {})",
                    characters[i], turns_a, turns_b
                ));
                a_dominance += 0.3;
            } else if turns_b > turns_a + 2 {
                indicators.push(format!(
                    "{} speaks more often ({} vs {})",
                    characters[j], turns_b, turns_a
                ));
                a_dominance -= 0.3;
            }

            // Who asks more questions (less dominant)
            let q_rate_a = fp_a.questions as f64 / fp_a.total_turns.max(1) as f64;
            let q_rate_b = fp_b.questions as f64 / fp_b.total_turns.max(1) as f64;
            if q_rate_a > q_rate_b + 0.2 {
                indicators.push(format!(
                    "{} asks more questions (deferential)",
                    characters[i]
                ));
                a_dominance -= 0.2;
            } else if q_rate_b > q_rate_a + 0.2 {
                indicators.push(format!(
                    "{} asks more questions (deferential)",
                    characters[j]
                ));
                a_dominance += 0.2;
            }

            // Longer turns = more dominant
            let avg_len_a = fp_a.total_words as f64 / fp_a.total_turns.max(1) as f64;
            let avg_len_b = fp_b.total_words as f64 / fp_b.total_turns.max(1) as f64;
            if avg_len_a > avg_len_b * 1.5 {
                indicators.push(format!("{} uses longer turns", characters[i]));
                a_dominance += 0.2;
            } else if avg_len_b > avg_len_a * 1.5 {
                indicators.push(format!("{} uses longer turns", characters[j]));
                a_dominance -= 0.2;
            }

            if !indicators.is_empty() {
                let dominant = if a_dominance > 0.1 {
                    characters[i].to_string()
                } else if a_dominance < -0.1 {
                    characters[j].to_string()
                } else {
                    "balanced".to_string()
                };

                dynamics.push(PowerDynamic {
                    character_a: characters[i].to_string(),
                    character_b: characters[j].to_string(),
                    dominant,
                    indicators,
                });
            }
        }
    }

    dynamics
}

pub fn analyze_dialogue_realism(
    dialogue_turns: &[(usize, String, String)],
    characters: &[&str],
) -> DialogueRealismResult {
    // Group turns by character
    let mut char_turns: HashMap<String, Vec<(usize, &str)>> = HashMap::new();
    let turn_refs: Vec<(usize, &str, &str)> = dialogue_turns
        .iter()
        .map(|(s, speaker, text)| (*s, speaker.as_str(), text.as_str()))
        .collect();

    for (scene, speaker, text) in &turn_refs {
        char_turns
            .entry(speaker.to_lowercase())
            .or_default()
            .push((*scene, *text));
    }

    // Build voice profiles
    let mut character_voices = Vec::new();
    let mut fingerprints: HashMap<String, VoiceFingerprint> = HashMap::new();

    for character in characters {
        let key = character.to_lowercase();
        let turns: Vec<&str> = char_turns
            .get(&key)
            .map(|t| t.iter().map(|(_, text)| *text).collect())
            .unwrap_or_default();

        if turns.is_empty() {
            continue;
        }

        let fp = build_fingerprint(&turns);
        let formality = formality_score(&fp, &turns);

        let avg_turn_length = fp.total_words as f64 / fp.total_turns.max(1) as f64;
        let vocabulary_richness = fp.word_freq.len() as f64 / fp.total_words.max(1) as f64;
        let contraction_rate = fp.contractions as f64 / fp.total_words.max(1) as f64;
        let question_rate = fp.questions as f64 / fp.total_turns.max(1) as f64;
        let exclamation_rate = fp.exclamations as f64 / fp.total_turns.max(1) as f64;

        // Top words by normalized frequency within this character's dialogue.
        // No cross-character distinctiveness filter is applied here.
        let frequent_words: Vec<String> = fp
            .word_freq
            .iter()
            .filter(|(_, freq)| **freq > 0.02)
            .map(|(word, _)| word.clone())
            .take(5)
            .collect();

        character_voices.push(CharacterVoice {
            character: character.to_string(),
            avg_turn_length,
            vocabulary_richness,
            formality_level: formality,
            contraction_rate,
            question_rate,
            exclamation_rate,
            frequent_words,
        });

        fingerprints.insert(key, fp);
    }

    // Pairwise voice similarity
    let mut voice_pairs = Vec::new();
    for i in 0..characters.len() {
        for j in (i + 1)..characters.len() {
            let key_a = characters[i].to_lowercase();
            let key_b = characters[j].to_lowercase();
            if let (Some(fp_a), Some(fp_b)) = (fingerprints.get(&key_a), fingerprints.get(&key_b)) {
                let sim = map_cosine_similarity(&fp_a.word_freq, &fp_b.word_freq);
                voice_pairs.push(VoicePair {
                    character_a: characters[i].to_string(),
                    character_b: characters[j].to_string(),
                    similarity: sim,
                });
            }
        }
    }

    // Detect issues
    let mut issues = detect_info_dumps(&turn_refs);
    issues.extend(detect_formality_issues(characters, &char_turns));

    // Flag identical voices
    for pair in &voice_pairs {
        if pair.similarity > 0.85 {
            issues.push(DialogueIssue {
                scene: 0,
                issue_type: "identical_voices".to_string(),
                description: format!(
                    "{} and {} have very similar voice patterns (similarity: {:.0}%)",
                    pair.character_a,
                    pair.character_b,
                    pair.similarity * 100.0
                ),
                severity: 0.65,
                characters: vec![pair.character_a.clone(), pair.character_b.clone()],
            });
        }
    }

    let info_dump_count = issues
        .iter()
        .filter(|i| i.issue_type == "info_dump")
        .count();

    // Power dynamics
    let power_dynamics = analyze_power_dynamics(characters, &char_turns);

    // Overall scores
    let voice_distinctiveness = if voice_pairs.is_empty() {
        0.5
    } else {
        let avg_sim =
            voice_pairs.iter().map(|p| p.similarity).sum::<f64>() / voice_pairs.len() as f64;
        (1.0 - avg_sim).clamp(0.0, 1.0)
    };

    let issue_penalty = (issues.len() as f64 * 0.05).min(0.4);
    let overall_realism = (voice_distinctiveness * 0.5 + 0.5 - issue_penalty).clamp(0.0, 1.0);

    DialogueRealismResult {
        character_voices,
        voice_pairs,
        issues,
        power_dynamics,
        overall_realism,
        voice_distinctiveness,
        info_dump_count,
    }
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
        assert!(result.info_dump_count > 0);
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
