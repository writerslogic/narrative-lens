use std::collections::HashSet;
use std::sync::OnceLock;

use crate::substrate::utils::{round1, round2, round3};

static SUSPENSE_WORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();

fn suspense_words() -> &'static HashSet<&'static str> {
    SUSPENSE_WORDS.get_or_init(|| {
        [
            "suddenly",
            "froze",
            "silence",
            "waiting",
            "heard",
            "shadow",
            "watched",
            "crept",
            "whispered",
            "darkness",
            "trembled",
            "shuddered",
            "lurked",
            "dread",
            "panic",
            "scream",
            "vanished",
            "motionless",
            "breathless",
            "echoed",
            "stalked",
            "hidden",
            "ominous",
            "looming",
            "creaked",
            "shiver",
            "eerie",
        ]
        .into_iter()
        .collect()
    })
}

/// Result of analyzing syntax tension in a single scene.
#[derive(Debug, Clone)]
pub struct SceneTensionResult {
    pub score: f64,
    pub length_variation: f64,
    pub burst_count: usize,
    pub sentence_count: usize,
    pub avg_sentence_length: f64,
    pub question_density: f64,
    pub dialogue_shift_score: f64,
    pub paragraph_burst_score: f64,
    pub exclamation_density: f64,
    pub suspense_keyword_score: f64,
}

/// Aggregate tension metrics across multiple scenes.
#[derive(Debug, Clone)]
pub struct TensionSummary {
    pub max_suspense_score: f64,
    pub average_suspense_score: f64,
    pub pacing_style: String,
}

/// Analyze syntax tension in scene text using the fallback (regex-only) path.
pub fn analyze_scene_text(scene_text: &str) -> SceneTensionResult {
    let trimmed = scene_text.trim();
    let empty = SceneTensionResult {
        score: 0.0,
        length_variation: 0.0,
        burst_count: 0,
        sentence_count: 0,
        avg_sentence_length: 0.0,
        question_density: 0.0,
        dialogue_shift_score: 0.0,
        paragraph_burst_score: 0.0,
        exclamation_density: 0.0,
        suspense_keyword_score: 0.0,
    };
    if trimmed.is_empty() {
        return empty;
    }

    let sents = crate::substrate::text::split_sentences(trimmed);
    if sents.is_empty() {
        return empty;
    }

    let lengths: Vec<usize> = sents.iter().map(|s| s.split_whitespace().count()).collect();
    let n = lengths.len();
    let mean_len = lengths.iter().sum::<usize>() as f64 / n as f64;

    // Length variation (coefficient of variation, capped at 1.0)
    let length_variation = if n > 1 {
        let variance = lengths
            .iter()
            .map(|&l| (l as f64 - mean_len).powi(2))
            .sum::<f64>()
            / n as f64;
        (variance.sqrt() / mean_len.max(1.0)).min(1.0)
    } else {
        0.0
    };

    // Short sentence bursts (3+ sentences under 8 words in a row)
    let mut burst_count: usize = 0;
    let mut streak: usize = 0;
    for &l in &lengths {
        if l <= 8 {
            streak += 1;
            if streak >= 3 {
                burst_count += 1;
            }
        } else {
            streak = 0;
        }
    }
    let burst_factor = (burst_count as f64 / (n as f64 / 10.0).max(1.0)).min(1.0);

    // Question density: fraction of sentences ending with '?'
    let question_count = sents.iter().filter(|s| s.ends_with('?')).count();
    let question_density = (question_count as f64 / n as f64).min(1.0);

    // Exclamation density: fraction of sentences ending with '!'
    let exclamation_count = sents.iter().filter(|s| s.ends_with('!')).count();
    let exclamation_density = (exclamation_count as f64 / n as f64).min(1.0);

    // Dialogue-to-narration shifts: detect rapid alternation
    let dialogue_shift_score = compute_dialogue_shift_score(trimmed);

    // Short paragraph bursts: multiple 1-2 sentence paragraphs in a row
    let paragraph_burst_score = compute_paragraph_burst_score(trimmed);

    // Suspense keywords: weighted count per 100 words
    let suspense_keyword_score = compute_suspense_keyword_score(trimmed);

    // Combined score with new weights
    // Original: avg_length_factor*2.0 + length_variation*4.0 + burst_factor*4.0 (max 10)
    // New total raw max: 2.0 + 4.0 + 4.0 + 1.5 + 1.0 + 1.5 + 0.5 + 1.5 = 16.0
    // Normalize to 0-10 scale
    let avg_length_factor = (mean_len / 30.0).min(1.0);
    let raw = avg_length_factor * 2.0
        + length_variation * 4.0
        + burst_factor * 4.0
        + question_density * 1.5
        + dialogue_shift_score * 1.0
        + paragraph_burst_score * 1.5
        + exclamation_density * 0.5
        + suspense_keyword_score * 1.5;
    let score = (raw * 10.0 / 16.0).clamp(0.0, 10.0);

    SceneTensionResult {
        score: round2(score),
        length_variation: round3(length_variation),
        burst_count,
        sentence_count: n,
        avg_sentence_length: round1(mean_len),
        question_density: round3(question_density),
        dialogue_shift_score: round3(dialogue_shift_score),
        paragraph_burst_score: round3(paragraph_burst_score),
        exclamation_density: round3(exclamation_density),
        suspense_keyword_score: round3(suspense_keyword_score),
    }
}

/// Detect rapid alternation between dialogue and narration lines.
/// Returns a 0.0-1.0 score where higher means more rapid shifts.
fn compute_dialogue_shift_score(text: &str) -> f64 {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.len() < 3 {
        return 0.0;
    }
    let is_dialogue: Vec<bool> = lines
        .iter()
        .map(|l| {
            let t = l.trim();
            t.contains('"')
                || t.contains('\u{201c}')
                || t.contains('\u{201d}')
                || t.starts_with('\u{2018}')
        })
        .collect();

    let mut shifts = 0usize;
    for w in is_dialogue.windows(2) {
        if w[0] != w[1] {
            shifts += 1;
        }
    }
    let shift_rate = shifts as f64 / (lines.len() - 1) as f64;
    // Rapid alternation (shift_rate > 0.5) scores high
    (shift_rate * 1.5).min(1.0)
}

/// Detect bursts of short paragraphs (1-2 sentences each).
/// Returns a 0.0-1.0 score.
fn compute_paragraph_burst_score(text: &str) -> f64 {
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .collect();
    if paragraphs.len() < 3 {
        return 0.0;
    }
    let sent_counts: Vec<usize> = paragraphs
        .iter()
        .map(|p| crate::substrate::text::count_sentences(p))
        .collect();

    // Count runs of 3+ consecutive short paragraphs (1-2 sentences)
    let mut burst_runs = 0usize;
    let mut run = 0usize;
    for &sc in &sent_counts {
        if sc <= 2 {
            run += 1;
            if run >= 3 {
                burst_runs += 1;
            }
        } else {
            run = 0;
        }
    }
    // Also count fraction of short paragraphs
    let short_frac =
        sent_counts.iter().filter(|&&c| c <= 2).count() as f64 / sent_counts.len() as f64;
    let burst_frac = burst_runs as f64 / (paragraphs.len() as f64 / 5.0).max(1.0);
    ((short_frac * 0.5 + burst_frac * 0.5) * 1.5).min(1.0)
}

/// Count suspense keywords per 100 words, normalized to 0.0-1.0.
fn compute_suspense_keyword_score(text: &str) -> f64 {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return 0.0;
    }
    let sw = suspense_words();
    let count = words
        .iter()
        .filter(|w| {
            let lower = w.to_lowercase();
            let cleaned = lower.trim_matches(|c: char| !c.is_alphabetic());
            sw.contains(cleaned)
        })
        .count();
    let per_100 = count as f64 / words.len() as f64 * 100.0;
    // 3+ suspense words per 100 words = max score
    (per_100 / 3.0).min(1.0)
}

/// Compute aggregate tension metrics from per-scene scores.
pub fn get_tension_metrics(scene_scores: &[f64]) -> TensionSummary {
    if scene_scores.is_empty() {
        return TensionSummary {
            max_suspense_score: 0.0,
            average_suspense_score: 0.0,
            pacing_style: "direct".to_string(),
        };
    }

    let max_s = scene_scores
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let avg_s = scene_scores.iter().sum::<f64>() / scene_scores.len() as f64;

    let pacing_style = if avg_s > 5.0 {
        "periodic"
    } else if avg_s > 3.0 {
        "balanced"
    } else {
        "direct"
    };

    TensionSummary {
        max_suspense_score: round2(max_s),
        average_suspense_score: round2(avg_s),
        pacing_style: pacing_style.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_text() {
        let r = analyze_scene_text("");
        assert_eq!(r.score, 0.0);
        assert_eq!(r.sentence_count, 0);
        assert_eq!(r.burst_count, 0);
    }

    #[test]
    fn test_whitespace_only() {
        let r = analyze_scene_text("   \n\t  ");
        assert_eq!(r.score, 0.0);
        assert_eq!(r.sentence_count, 0);
    }

    #[test]
    fn test_single_short_sentence() {
        let r = analyze_scene_text("He ran.");
        assert_eq!(r.sentence_count, 1);
        assert_eq!(r.length_variation, 0.0);
        assert_eq!(r.burst_count, 0);
        // avg_sentence_length = 2.0, factor = 2/30 ~ 0.067, score = 0.067*2 = 0.13
        assert!(r.score < 1.0, "single short sentence should score low");
    }

    #[test]
    fn test_action_scene_high_burst() {
        // Many short sentences in a row should trigger bursts
        let text = "He ran. She ducked. Glass shattered. \
                     He dove. She screamed. Shots rang. \
                     He rolled. She fired. Smoke rose. \
                     He gasped. She froze. Silence fell.";
        let r = analyze_scene_text(text);
        assert_eq!(r.sentence_count, 12);
        assert!(
            r.burst_count > 0,
            "expected burst_count > 0, got {}",
            r.burst_count
        );
        assert!(
            r.score > 2.0,
            "action scene should score above 2, got {}",
            r.score
        );
    }

    #[test]
    fn test_contemplative_scene_uniform_long() {
        // Long uniform sentences; low variation, no bursts
        let s = "The morning light filtered through the heavy curtains casting long pale shadows across the wooden floor of the old library. \
                  She sat quietly in the worn leather chair thinking about all the years that had slowly and inevitably slipped away from her grasp. \
                  The clock on the mantel ticked with a steady and almost hypnotic rhythm that seemed to measure out the silence in careful equal portions.";
        let r = analyze_scene_text(s);
        assert_eq!(r.sentence_count, 3);
        assert_eq!(r.burst_count, 0, "no short bursts expected");
        // High avg length, some variation, but no bursts
        assert!(r.avg_sentence_length > 15.0);
    }

    #[test]
    fn test_mixed_scene() {
        let text = "The old house stood silent on the hill overlooking the grey and restless sea below. \
                     He stopped. He listened. Nothing moved. \
                     Then the wind picked up and howled through the broken shutters of the second floor windows.";
        let r = analyze_scene_text(text);
        assert_eq!(r.sentence_count, 5);
        assert!(r.length_variation > 0.0);
        // 3 short sentences in a row should create a burst
        assert!(
            r.burst_count >= 1,
            "expected at least 1 burst, got {}",
            r.burst_count
        );
    }

    #[test]
    fn test_tension_metrics_empty() {
        let m = get_tension_metrics(&[]);
        assert_eq!(m.max_suspense_score, 0.0);
        assert_eq!(m.average_suspense_score, 0.0);
        assert_eq!(m.pacing_style, "direct");
    }

    #[test]
    fn test_tension_metrics_direct() {
        let m = get_tension_metrics(&[1.0, 2.0, 3.0]);
        assert_eq!(m.max_suspense_score, 3.0);
        assert_eq!(m.average_suspense_score, 2.0);
        assert_eq!(m.pacing_style, "direct");
    }

    #[test]
    fn test_tension_metrics_balanced() {
        let m = get_tension_metrics(&[3.5, 4.0, 3.5]);
        assert_eq!(m.pacing_style, "balanced");
    }

    #[test]
    fn test_tension_metrics_periodic() {
        let m = get_tension_metrics(&[6.0, 7.0, 8.0]);
        assert_eq!(m.max_suspense_score, 8.0);
        assert_eq!(m.pacing_style, "periodic");
    }

    #[test]
    fn test_question_density() {
        let text = "Where was he? What happened? Who did this? She looked around. Nothing.";
        let r = analyze_scene_text(text);
        assert!(
            r.question_density > 0.4,
            "expected high question density, got {}",
            r.question_density
        );
    }

    #[test]
    fn test_exclamation_density() {
        let text = "Stop! Run! Get down! The building exploded. Fire everywhere!";
        let r = analyze_scene_text(text);
        assert!(
            r.exclamation_density > 0.5,
            "expected high exclamation density, got {}",
            r.exclamation_density
        );
    }

    #[test]
    fn test_suspense_keywords() {
        let text = "She suddenly froze. The silence was deafening. \
                     A shadow crept across the wall. She heard whispered voices in the darkness.";
        let r = analyze_scene_text(text);
        assert!(
            r.suspense_keyword_score > 0.3,
            "expected suspense keyword score > 0.3, got {}",
            r.suspense_keyword_score
        );
    }

    #[test]
    fn test_no_suspense_in_calm_text() {
        let text = "The morning was pleasant. She ate breakfast. The garden looked beautiful. \
                     They walked along the path together.";
        let r = analyze_scene_text(text);
        assert!(
            r.suspense_keyword_score < 0.1,
            "calm text should have low suspense, got {}",
            r.suspense_keyword_score
        );
    }

    #[test]
    fn test_dialogue_shifts() {
        let text = "She stared at the door.\n\
                     \"Who's there?\" she called.\n\
                     No answer came.\n\
                     \"I said, who's there?\" she repeated.\n\
                     The handle turned slowly.\n\
                     \"Stay back!\" she warned.";
        let r = analyze_scene_text(text);
        assert!(
            r.dialogue_shift_score > 0.3,
            "expected dialogue shift score > 0.3, got {}",
            r.dialogue_shift_score
        );
    }

    #[test]
    fn test_paragraph_bursts() {
        let text = "He ran.\n\nShe screamed.\n\nGlass shattered.\n\nDarkness fell.\n\n\
                     The long hallway stretched before them, silent and empty, \
                     as they crept forward into the unknown depths of the ancient building.";
        let r = analyze_scene_text(text);
        assert!(
            r.paragraph_burst_score > 0.0,
            "expected paragraph burst score > 0, got {}",
            r.paragraph_burst_score
        );
    }

    #[test]
    fn test_high_tension_scene_scores_higher() {
        let calm = "The morning was quiet. She sat reading. The birds sang outside. \
                     Everything was peaceful and still.";
        let tense = "Where was he? She suddenly froze. \"Who's there?\" she whispered. \
                      Silence. A shadow crept past! She heard footsteps. Run!";
        let r_calm = analyze_scene_text(calm);
        let r_tense = analyze_scene_text(tense);
        assert!(
            r_tense.score > r_calm.score,
            "tense scene ({}) should score higher than calm scene ({})",
            r_tense.score,
            r_calm.score
        );
    }
}
