use std::collections::HashSet;
use std::sync::OnceLock;

use log::debug;

use crate::substrate::text::{count_sentences, count_syllables};
use crate::substrate::utils::{round2, safe_div};

fn dale_chall_familiar() -> &'static HashSet<&'static str> {
    static INST: OnceLock<HashSet<&str>> = OnceLock::new();
    INST.get_or_init(|| include_str!("data/dale_chall_1995.txt").lines().collect())
}

// ---------------------------------------------------------------------------
// ReadabilityResult
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ReadabilityResult {
    pub fkgl: f64,
    pub gunning_fog: f64,
    pub smog: f64,
    pub smog_available: bool,
    pub ari: f64,
    pub coleman_liau: f64,
    pub flesch_ease: f64,
    pub dale_chall: f64,
    pub sentence_count: usize,
    pub word_count: usize,
    pub syllable_count: usize,
    pub char_count: usize,
    pub polysyllable_count: usize,
    pub grade_levels: GradeLevels,
    pub per_paragraph_fkgl: Vec<f64>,
    pub readability_variance: f64,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmogSample {
    pub byte_start: usize,
    pub byte_end: usize,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SmogAnalysis {
    pub availability: crate::substrate::utils::AnalysisAvailability,
    pub grade: Option<f64>,
    pub method: String,
    pub total_sentence_count: usize,
    pub eligible_sentence_count: usize,
    pub polysyllable_count: usize,
    pub sampled_sentences: Vec<SmogSample>,
    pub diagnostics: Vec<String>,
}

/// McLaughlin (1969), equation (d), over beginning/middle/end ten-sentence samples.
/// Sentence boundaries and spoken syllables are English heuristics.
pub fn analyze_smog(text: &str) -> SmogAnalysis {
    smog_from_spans(text, &crate::substrate::utils::sentence_spans(text))
}

fn smog_from_spans(text: &str, spans: &[(usize, usize)]) -> SmogAnalysis {
    let eligible: Vec<_> = spans
        .iter()
        .copied()
        .filter(|&(start, end)| {
            text[start..end].chars().any(char::is_alphabetic)
                && text[start..end]
                    .trim_end_matches(['"', '\'', '\u{2019}', '\u{201d}', ')', ']'])
                    .ends_with(['.', '?', '!'])
        })
        .collect();
    let mut analysis = SmogAnalysis {
        availability: crate::substrate::utils::AnalysisAvailability::Unavailable,
        grade: None,
        method: "mclaughlin_1969_equation_d_30_sentence_sample".into(),
        total_sentence_count: spans.len(),
        eligible_sentence_count: eligible.len(),
        polysyllable_count: 0,
        sampled_sentences: Vec::new(),
        diagnostics: vec!["English sentence and syllable heuristics; numerals and abbreviations are not expanded into spoken forms.".into()],
    };
    if eligible.len() < 30 {
        analysis
            .diagnostics
            .push("SMOG requires at least 30 complete word-bearing sentences.".into());
        return analysis;
    }
    let middle = (eligible.len() - 10) / 2;
    for &(start, end) in eligible[..10]
        .iter()
        .chain(&eligible[middle..middle + 10])
        .chain(&eligible[eligible.len() - 10..])
    {
        analysis.polysyllable_count += word_slices(&text[start..end])
            .iter()
            .filter(|word| count_syllables(word) >= 3)
            .count();
        analysis.sampled_sentences.push(SmogSample {
            byte_start: start,
            byte_end: end,
        });
    }
    analysis.grade = Some(round2(
        1.043 * (analysis.polysyllable_count as f64).sqrt() + 3.1291,
    ));
    analysis.availability = crate::substrate::utils::AnalysisAvailability::Complete;
    analysis
}

#[derive(Clone, Debug)]
pub struct GradeLevels {
    pub fkgl: &'static str,
    pub gunning_fog: &'static str,
    pub ari: &'static str,
    pub coleman_liau: &'static str,
    pub flesch_ease: &'static str,
    pub dale_chall: &'static str,
}

fn grade_level_label(score: f64) -> &'static str {
    if score < 1.0 {
        "Kindergarten"
    } else if score < 2.0 {
        "1st grade"
    } else if score < 3.0 {
        "2nd grade"
    } else if score < 4.0 {
        "3rd grade"
    } else if score < 5.0 {
        "4th grade"
    } else if score < 6.0 {
        "5th grade"
    } else if score < 7.0 {
        "6th grade"
    } else if score < 8.0 {
        "7th grade"
    } else if score < 9.0 {
        "8th grade"
    } else if score < 10.0 {
        "9th grade"
    } else if score < 11.0 {
        "10th grade"
    } else if score < 12.0 {
        "11th grade"
    } else if score < 13.0 {
        "12th grade"
    } else if score < 16.0 {
        "College level"
    } else {
        "Graduate level"
    }
}

fn flesch_ease_label(score: f64) -> &'static str {
    if score >= 90.0 {
        "5th grade (very easy)"
    } else if score >= 80.0 {
        "6th grade (easy)"
    } else if score >= 70.0 {
        "7th grade (fairly easy)"
    } else if score >= 60.0 {
        "8th-9th grade (standard)"
    } else if score >= 50.0 {
        "10th-12th grade (fairly difficult)"
    } else if score >= 30.0 {
        "College level (difficult)"
    } else {
        "Graduate level (very difficult)"
    }
}

fn dale_chall_label(score: f64) -> &'static str {
    if score <= 4.9 {
        "4th grade or below (easy)"
    } else if score <= 5.9 {
        "5th-6th grade"
    } else if score <= 6.9 {
        "7th-8th grade"
    } else if score <= 7.9 {
        "9th-10th grade"
    } else if score <= 8.9 {
        "11th-12th grade"
    } else if score <= 9.9 {
        "College level"
    } else {
        "Graduate level"
    }
}

/// Extract words from text (sequences of alphabetic + apostrophe characters).
fn word_slices(text: &str) -> Vec<&str> {
    text.split(|c: char| !c.is_alphabetic() && !matches!(c, '\'' | '‘' | '’'))
        .map(|word| word.trim_matches(['\'', '‘', '’']))
        .filter(|w| !w.is_empty() && w.chars().any(|c| c.is_alphabetic()))
        .collect()
}

pub fn compute_readability(text: &str) -> ReadabilityResult {
    debug!(
        "[readability] compute_readability — text_len={}",
        text.len()
    );
    let words: Vec<&str> = word_slices(text);
    let word_count = words.len();
    let sentence_spans = crate::substrate::utils::sentence_spans(text);
    let sentence_count = sentence_spans.len();
    debug!(
        "[readability] words={}, sentences={}",
        word_count, sentence_count
    );

    let mut syllable_count: usize = 0;
    let mut polysyllable_count: usize = 0;
    let mut char_count: usize = 0;

    for w in &words {
        let s = count_syllables(w);
        syllable_count += s;
        if s >= 3 {
            polysyllable_count += 1;
        }
        char_count += w.chars().filter(|c| c.is_alphabetic()).count();
    }

    let wf = word_count as f64;
    let sf = sentence_count as f64;
    let sylf = syllable_count as f64;
    let cf = char_count as f64;
    let pf = polysyllable_count as f64;

    let words_per_sentence = safe_div(wf, sf);
    let syllables_per_word = safe_div(sylf, wf);

    // FKGL
    let fkgl = if word_count == 0 || sentence_count == 0 {
        0.0
    } else {
        0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    };

    // Gunning Fog
    let gunning_fog = if word_count == 0 || sentence_count == 0 {
        0.0
    } else {
        0.4 * (words_per_sentence + 100.0 * safe_div(pf, wf))
    };

    let smog_analysis = smog_from_spans(text, &sentence_spans);
    let smog = smog_analysis.grade.unwrap_or(0.0);

    // ARI
    let ari = if word_count == 0 || sentence_count == 0 {
        0.0
    } else {
        4.71 * safe_div(cf, wf) + 0.5 * words_per_sentence - 21.43
    };

    // Coleman-Liau
    let coleman_liau = if word_count == 0 {
        0.0
    } else {
        let l = cf / wf * 100.0; // avg letters per 100 words
        let s = sf / wf * 100.0; // avg sentences per 100 words
        0.0588 * l - 0.296 * s - 15.8
    };

    // Flesch Reading Ease
    let flesch_ease = if word_count == 0 || sentence_count == 0 {
        0.0
    } else {
        206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    };

    // Dale-Chall
    let dale_chall = if word_count == 0 || sentence_count == 0 {
        0.0
    } else {
        let familiar = dale_chall_familiar();
        let difficult_count = words
            .iter()
            .filter(|w| !familiar.contains(crate::substrate::text::normalize_word(w).as_str()))
            .count();
        let pct_difficult = difficult_count as f64 / wf * 100.0;
        let raw = 0.1579 * pct_difficult + 0.0496 * words_per_sentence;
        if pct_difficult > 5.0 {
            raw + 3.6365
        } else {
            raw
        }
    };

    // Paragraph-level FKGL variation
    let paragraphs = crate::substrate::text::split_paragraphs(text);
    let per_paragraph_fkgl: Vec<f64> = if paragraphs.len() > 1 {
        paragraphs
            .iter()
            .map(|p| {
                let pw: Vec<&str> = word_slices(p);
                let pwc = pw.len();
                let psc = count_sentences(p);
                if pwc == 0 || psc == 0 {
                    return 0.0;
                }
                let psyl: usize = pw.iter().map(|w| count_syllables(w)).sum();
                let wps = pwc as f64 / psc as f64;
                let spw = psyl as f64 / pwc as f64;
                round2(0.39 * wps + 11.8 * spw - 15.59)
            })
            .collect()
    } else {
        vec![round2(fkgl)]
    };
    let readability_variance = if per_paragraph_fkgl.len() > 1 {
        let mean = per_paragraph_fkgl.iter().sum::<f64>() / per_paragraph_fkgl.len() as f64;
        let var = per_paragraph_fkgl
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / per_paragraph_fkgl.len() as f64;
        round2(var)
    } else {
        0.0
    };

    let fkgl = round2(fkgl);
    let gunning_fog = round2(gunning_fog);
    let ari = round2(ari);
    let coleman_liau = round2(coleman_liau);
    let flesch_ease = round2(flesch_ease);
    let dale_chall = round2(dale_chall);

    let grade_levels = GradeLevels {
        fkgl: grade_level_label(fkgl),
        gunning_fog: grade_level_label(gunning_fog),
        ari: grade_level_label(ari),
        coleman_liau: grade_level_label(coleman_liau),
        flesch_ease: flesch_ease_label(flesch_ease),
        dale_chall: dale_chall_label(dale_chall),
    };

    ReadabilityResult {
        fkgl,
        gunning_fog,
        smog: round2(smog),
        smog_available: smog_analysis.grade.is_some(),
        ari,
        coleman_liau,
        flesch_ease,
        dale_chall,
        sentence_count,
        word_count,
        syllable_count,
        char_count,
        polysyllable_count,
        grade_levels,
        per_paragraph_fkgl,
        readability_variance,
    }
}

// ---------------------------------------------------------------------------
// PacingResult
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PacingResult {
    pub scene_id: usize,
    pub word_count: usize,
    pub sentence_count: usize,
    pub velocity: f64,
    pub dialogue_ratio: f64,
    pub is_talking_head: bool,
    pub entity_density: f64,
    pub action_density: f64,
}

static ACTION_VERBS: OnceLock<HashSet<&'static str>> = OnceLock::new();

fn action_verb_set() -> &'static HashSet<&'static str> {
    ACTION_VERBS.get_or_init(|| {
        let verbs: &[&str] = &[
            // base forms and common inflections (~100 entries)
            "run",
            "ran",
            "runs",
            "running",
            "jump",
            "jumped",
            "jumps",
            "jumping",
            "hit",
            "hits",
            "hitting",
            "kick",
            "kicked",
            "kicks",
            "kicking",
            "punch",
            "punched",
            "punches",
            "punching",
            "throw",
            "threw",
            "throws",
            "throwing",
            "thrown",
            "grab",
            "grabbed",
            "grabs",
            "grabbing",
            "push",
            "pushed",
            "pushes",
            "pushing",
            "pull",
            "pulled",
            "pulls",
            "pulling",
            "chase",
            "chased",
            "chases",
            "chasing",
            "fight",
            "fought",
            "fights",
            "fighting",
            "strike",
            "struck",
            "strikes",
            "striking",
            "slash",
            "slashed",
            "slashes",
            "slashing",
            "stab",
            "stabbed",
            "stabs",
            "stabbing",
            "shoot",
            "shot",
            "shoots",
            "shooting",
            "fire",
            "fired",
            "fires",
            "firing",
            "dodge",
            "dodged",
            "dodges",
            "dodging",
            "sprint",
            "sprinted",
            "sprints",
            "sprinting",
            "leap",
            "leaped",
            "leapt",
            "leaps",
            "leaping",
            "climb",
            "climbed",
            "climbs",
            "climbing",
            "crash",
            "crashed",
            "crashes",
            "crashing",
            "smash",
            "smashed",
            "smashes",
            "smashing",
            "break",
            "broke",
            "breaks",
            "breaking",
            "broken",
            "tear",
            "tore",
            "tears",
            "tearing",
            "torn",
            "seize",
            "seized",
            "seizes",
            "seizing",
            "swing",
            "swung",
            "swings",
            "swinging",
            "charge",
            "charged",
            "charges",
            "charging",
            "rush",
            "rushed",
            "rushes",
            "rushing",
            "bolt",
            "bolted",
            "bolts",
            "bolting",
            "flee",
            "fled",
            "flees",
            "fleeing",
            "escape",
            "escaped",
            "escapes",
            "escaping",
            "attack",
            "attacked",
            "attacks",
            "attacking",
            "defend",
            "defended",
            "defends",
            "defending",
            "block",
            "blocked",
            "blocks",
            "blocking",
            "shove",
            "shoved",
            "shoves",
            "shoving",
            "tackle",
            "tackled",
            "tackles",
            "tackling",
            "wrestle",
            "wrestled",
            "wrestles",
            "wrestling",
            "drag",
            "dragged",
            "drags",
            "dragging",
            "hurl",
            "hurled",
            "hurls",
            "hurling",
            "lunge",
            "lunged",
            "lunges",
            "lunging",
            "duck",
            "ducked",
            "ducks",
            "ducking",
            "dive",
            "dived",
            "dove",
            "dives",
            "diving",
            "slam",
            "slammed",
            "slams",
            "slamming",
            "rip",
            "ripped",
            "rips",
            "ripping",
            "snatch",
            "snatched",
            "snatches",
            "snatching",
        ];
        verbs.iter().copied().collect()
    })
}

/// Check if a character is at a sentence start by looking backward from its
/// position for the nearest sentence-ending punctuation or start of text.
fn is_sentence_start(text: &str, word_start: usize) -> bool {
    // Walk backward through whitespace and find what precedes this word
    let bytes = text.as_bytes();
    let mut i = word_start;
    if i == 0 {
        return true;
    }
    i -= 1;
    // skip whitespace
    while i > 0 && bytes[i].is_ascii_whitespace() {
        i -= 1;
    }
    // Reached start of text (either whitespace or first character) means sentence start
    if i == 0 {
        return bytes[i].is_ascii_whitespace()
            || matches!(bytes[i], b'.' | b'!' | b'?' | b':' | b';');
    }
    matches!(bytes[i], b'.' | b'!' | b'?' | b':' | b';')
}

/// Count words inside dialogue markers. Delegates to text::estimate_dialogue_tokens
/// which handles straight, curly, and single quotes properly.
fn count_dialogue_words(text: &str) -> usize {
    crate::substrate::text::estimate_dialogue_tokens(text)
}

pub fn compute_pacing_metrics(scene_text: &str, scene_id: usize) -> PacingResult {
    let words: Vec<&str> = word_slices(scene_text);
    let word_count = words.len();
    let sentence_count = count_sentences(scene_text);
    let wf = word_count as f64;

    let velocity = round2(safe_div(wf, sentence_count as f64));

    // Dialogue ratio
    let dialogue_words = count_dialogue_words(scene_text);
    let dialogue_ratio = round2(safe_div(dialogue_words as f64, wf));
    let is_talking_head = dialogue_ratio > 0.6;

    // Entity density: capitalized words not at sentence starts
    let mut entity_count: usize = 0;
    let action_verbs = action_verb_set();
    let mut action_count: usize = 0;

    for w in &words {
        // Entity: capitalized word not at sentence start
        if w.chars().next().is_some_and(char::is_uppercase) {
            // Find this word's byte offset in the original text
            let word_ptr = w.as_ptr() as usize;
            let text_ptr = scene_text.as_ptr() as usize;
            let offset = word_ptr - text_ptr;
            if !is_sentence_start(scene_text, offset) {
                entity_count += 1;
            }
        }

        // Action verb check (lowercase comparison)
        let lower: String = w.to_lowercase();
        if action_verbs.contains(lower.as_str()) {
            action_count += 1;
        }
    }

    let entity_density = round2(safe_div(entity_count as f64, wf) * 100.0);
    let action_density = round2(safe_div(action_count as f64, wf) * 100.0);

    PacingResult {
        scene_id,
        word_count,
        sentence_count,
        velocity,
        dialogue_ratio,
        is_talking_head,
        entity_density,
        action_density,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ReadabilityResult tests --

    #[test]
    fn test_readability_empty() {
        let r = compute_readability("");
        assert_eq!(r.word_count, 0);
        assert_eq!(r.sentence_count, 0);
        assert_eq!(r.fkgl, 0.0);
        assert_eq!(r.gunning_fog, 0.0);
        assert_eq!(r.smog, 0.0);
        assert_eq!(r.ari, 0.0);
        assert_eq!(r.coleman_liau, 0.0);
        assert_eq!(r.flesch_ease, 0.0);
        assert_eq!(r.dale_chall, 0.0);
        assert_eq!(r.readability_variance, 0.0);
    }

    #[test]
    fn test_readability_single_word() {
        let r = compute_readability("Hello");
        assert_eq!(r.word_count, 1);
        assert_eq!(r.sentence_count, 1); // no punctuation, counts as 1
        assert_eq!(r.syllable_count, 2);
        assert_eq!(r.polysyllable_count, 0);
        // Formulas should produce some value without panicking
        assert!(r.fkgl.is_finite());
        assert!(r.flesch_ease.is_finite());
    }

    #[test]
    fn test_readability_simple_text() {
        let text = "The cat sat on the mat. The dog ran fast.";
        let r = compute_readability(text);
        assert_eq!(r.sentence_count, 2);
        assert_eq!(r.word_count, 10);
        assert!(r.fkgl.is_finite());
        assert!(
            r.flesch_ease > 0.0,
            "Simple text should have positive Flesch ease"
        );
    }

    #[test]
    fn test_readability_polysyllables() {
        let text = "The extraordinary communication demonstrated sophisticated understanding.";
        let r = compute_readability(text);
        assert!(r.polysyllable_count > 0);
        assert!(r.gunning_fog > 0.0);
    }

    #[test]
    fn test_readability_smog_needs_30_sentences() {
        // Fewer than 30 sentences: SMOG should be 0
        let text = "Hello. World. Test.";
        let r = compute_readability(text);
        assert_eq!(r.smog, 0.0);
    }

    #[test]
    fn test_readability_smog_with_30_sentences() {
        let mut sentences = Vec::new();
        for i in 0..30 {
            if i % 3 == 0 {
                sentences.push("The extraordinary communication demonstrated understanding.");
            } else {
                sentences.push("The cat sat.");
            }
        }
        let text = sentences.join(" ");
        let r = compute_readability(&text);
        assert!(r.smog > 0.0, "SMOG should be > 0 with 30+ sentences");
    }

    #[test]
    fn test_readability_no_division_by_zero() {
        // Text with no alphabetic characters
        let r = compute_readability("123 456 789");
        assert_eq!(r.word_count, 0);
        assert_eq!(r.fkgl, 0.0);
    }

    #[test]
    fn test_readability_values_rounded() {
        let text = "The quick brown fox jumps over the lazy dog. It was a simple sentence.";
        let r = compute_readability(text);
        // Check that values are rounded to 2 decimal places
        assert_eq!(r.fkgl, (r.fkgl * 100.0).round() / 100.0);
        assert_eq!(r.flesch_ease, (r.flesch_ease * 100.0).round() / 100.0);
    }

    // -- PacingResult tests --

    #[test]
    fn test_pacing_empty() {
        let p = compute_pacing_metrics("", 0);
        assert_eq!(p.word_count, 0);
        assert_eq!(p.sentence_count, 0);
        assert_eq!(p.velocity, 0.0);
        assert_eq!(p.dialogue_ratio, 0.0);
        assert!(!p.is_talking_head);
        assert_eq!(p.entity_density, 0.0);
        assert_eq!(p.action_density, 0.0);
    }

    #[test]
    fn test_pacing_scene_id() {
        let p = compute_pacing_metrics("Hello world.", 42);
        assert_eq!(p.scene_id, 42);
    }

    #[test]
    fn test_pacing_velocity() {
        let text = "One two three. Four five six.";
        let p = compute_pacing_metrics(text, 0);
        assert_eq!(p.sentence_count, 2);
        assert_eq!(p.word_count, 6);
        assert!((p.velocity - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_pacing_dialogue_ratio() {
        // 3 words in dialogue, 5 total (approximately)
        let text = "He said \"run away fast\" quickly.";
        let p = compute_pacing_metrics(text, 0);
        assert!(p.dialogue_ratio > 0.0);
        assert!(p.dialogue_ratio < 1.0);
    }

    #[test]
    fn test_pacing_talking_head() {
        // Mostly dialogue
        let text = "\"I think we should go. I really think we should leave now. Please let us go.\" He nodded.";
        let p = compute_pacing_metrics(text, 0);
        // Dialogue words dominate
        assert!(p.dialogue_ratio > 0.5);
    }

    #[test]
    fn test_pacing_entity_density() {
        // "Alice" and "Bob" mid-sentence are entities; "The" at start is not
        let text = "The hero met Alice and Bob at the park.";
        let p = compute_pacing_metrics(text, 0);
        assert!(
            p.entity_density > 0.0,
            "Should detect mid-sentence capitalized words"
        );
    }

    #[test]
    fn test_pacing_action_density() {
        let text = "She ran and jumped over the wall. He grabbed the rope.";
        let p = compute_pacing_metrics(text, 0);
        assert!(p.action_density > 0.0, "Should detect action verbs");
    }

    #[test]
    fn test_pacing_no_action_verbs() {
        let text = "The sky was blue. Clouds floated gently.";
        let p = compute_pacing_metrics(text, 0);
        // "floated" is not in our action verb list
        // This just verifies no panic and some density value
        assert!(p.action_density.is_finite());
    }

    #[test]
    fn test_pacing_single_word() {
        let p = compute_pacing_metrics("Hello", 0);
        assert_eq!(p.word_count, 1);
        assert_eq!(p.sentence_count, 1);
        assert!((p.velocity - 1.0).abs() < 0.01);
    }

    // -- Helper function tests --

    #[test]
    fn test_word_slices() {
        let words = word_slices("Hello, world! It's a test.");
        assert_eq!(words, vec!["Hello", "world", "It's", "a", "test"]);
    }

    #[test]
    fn test_word_slices_empty() {
        let words = word_slices("");
        assert!(words.is_empty());
    }

    #[test]
    fn test_count_dialogue_words_none() {
        assert_eq!(count_dialogue_words("No dialogue here."), 0);
    }

    #[test]
    fn test_count_dialogue_words_some() {
        let count = count_dialogue_words("He said \"hello world\" softly.");
        assert_eq!(count, 2);
    }

    #[test]
    fn test_safe_div() {
        assert_eq!(safe_div(10.0, 0.0), 0.0);
        assert_eq!(safe_div(10.0, 2.0), 5.0);
    }

    #[test]
    fn test_round2() {
        assert_eq!(round2(1.23456), 1.23);
        assert_eq!(round2(2.005), 2.01); // IEEE rounding
        assert_eq!(round2(0.0), 0.0);
    }

    // -- Dale-Chall tests --

    #[test]
    fn test_dale_chall_simple_text() {
        let text = "The cat sat on the mat. The dog ran fast.";
        let r = compute_readability(text);
        assert!(r.dale_chall.is_finite());
        assert!(
            r.dale_chall > 0.0,
            "Simple text should have positive Dale-Chall"
        );
    }

    #[test]
    fn test_dale_chall_difficult_text() {
        let simple = compute_readability("The cat sat on the mat. The dog ran home.");
        let hard = compute_readability(
            "The epistemological ramifications necessitate comprehensive \
             deliberation. Phenomenological considerations exacerbate \
             the hermeneutical predicament substantially.",
        );
        assert!(
            hard.dale_chall > simple.dale_chall,
            "Difficult text should score higher: {} vs {}",
            hard.dale_chall,
            simple.dale_chall
        );
    }

    #[test]
    fn test_dale_chall_familiar_list() {
        let familiar = dale_chall_familiar();
        assert!(familiar.contains("the"));
        assert!(familiar.contains("a"));
        assert!(familiar.contains("house"));
        assert!(familiar.contains("school"));
        assert!(
            familiar.len() >= 500,
            "Expected >= 500 familiar words, got {}",
            familiar.len()
        );
    }

    // -- Grade level tests --

    #[test]
    fn test_grade_levels_populated() {
        let text = "The cat sat on the mat. The dog ran fast.";
        let r = compute_readability(text);
        assert!(!r.grade_levels.fkgl.is_empty());
        assert!(!r.grade_levels.gunning_fog.is_empty());
        assert!(!r.grade_levels.ari.is_empty());
        assert!(!r.grade_levels.coleman_liau.is_empty());
        assert!(!r.grade_levels.flesch_ease.is_empty());
        assert!(!r.grade_levels.dale_chall.is_empty());
    }

    #[test]
    fn test_grade_level_label_values() {
        assert_eq!(grade_level_label(0.5), "Kindergarten");
        assert_eq!(grade_level_label(5.0), "5th grade");
        assert_eq!(grade_level_label(12.5), "12th grade");
        assert_eq!(grade_level_label(14.0), "College level");
        assert_eq!(grade_level_label(17.0), "Graduate level");
    }

    #[test]
    fn test_flesch_ease_label_values() {
        assert_eq!(flesch_ease_label(95.0), "5th grade (very easy)");
        assert_eq!(flesch_ease_label(65.0), "8th-9th grade (standard)");
        assert_eq!(flesch_ease_label(20.0), "Graduate level (very difficult)");
    }

    #[test]
    fn test_dale_chall_label_values() {
        assert_eq!(dale_chall_label(4.0), "4th grade or below (easy)");
        assert_eq!(dale_chall_label(7.5), "9th-10th grade");
        assert_eq!(dale_chall_label(10.5), "Graduate level");
    }

    // -- Paragraph-level variance tests --

    #[test]
    fn test_paragraph_variance_single_paragraph() {
        let text = "The cat sat on the mat. The dog ran fast.";
        let r = compute_readability(text);
        assert_eq!(r.readability_variance, 0.0);
        assert_eq!(r.per_paragraph_fkgl.len(), 1);
    }

    #[test]
    fn test_paragraph_variance_multiple() {
        let text = "The cat sat. The dog ran.\n\n\
                     The extraordinary communication demonstrated \
                     sophisticated understanding of epistemological matters.";
        let r = compute_readability(text);
        assert_eq!(r.per_paragraph_fkgl.len(), 2);
        assert!(
            r.readability_variance > 0.0,
            "Mixed difficulty paragraphs should have variance > 0"
        );
        // Simple paragraph should have lower FKGL than complex one
        assert!(
            r.per_paragraph_fkgl[0] < r.per_paragraph_fkgl[1],
            "Simple paragraph should score lower: {:?}",
            r.per_paragraph_fkgl
        );
    }

    #[test]
    fn test_paragraph_variance_uniform() {
        let text = "The cat sat on the mat.\n\n\
                     The dog ran to the door.\n\n\
                     The bird flew in the sky.";
        let r = compute_readability(text);
        assert_eq!(r.per_paragraph_fkgl.len(), 3);
        // All similar difficulty; variance should be low
        assert!(
            r.readability_variance < 5.0,
            "Uniform paragraphs should have low variance: {}",
            r.readability_variance
        );
    }
}
