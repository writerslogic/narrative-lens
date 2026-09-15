use std::collections::HashSet;
use std::sync::OnceLock;

use log::debug;

use crate::substrate::utils::{round2, round3, round4};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const THS_THRESHOLD: f64 = 0.85;
const MIN_WORD_COUNT: usize = 500;

// ---------------------------------------------------------------------------
// Static action verbs set
// ---------------------------------------------------------------------------

static ACTION_VERBS: OnceLock<HashSet<&'static str>> = OnceLock::new();

fn action_verbs() -> &'static HashSet<&'static str> {
    ACTION_VERBS.get_or_init(|| {
        [
            "ran",
            "run",
            "running",
            "jumped",
            "jump",
            "jumping",
            "grabbed",
            "grab",
            "pulled",
            "pull",
            "pushed",
            "push",
            "threw",
            "throw",
            "caught",
            "catch",
            "hit",
            "kicked",
            "kick",
            "slammed",
            "slam",
            "rushed",
            "rush",
            "charged",
            "charge",
            "fled",
            "flee",
            "fought",
            "fight",
            "struck",
            "strike",
            "climbed",
            "climb",
            "fell",
            "fall",
            "crashed",
            "crash",
            "burst",
            "sprinted",
            "dodged",
            "dodge",
            "swung",
            "swing",
            "fired",
            "fire",
            "stabbed",
            "stab",
            "blocked",
            "block",
            "lunged",
            "lunge",
            "tackled",
            "tackle",
            "shoved",
            "shove",
            "ripped",
            "rip",
            "broke",
            "break",
            "smashed",
            "smash",
            "seized",
            "seize",
            "dragged",
            "drag",
            "hurled",
            "hurl",
            "leapt",
            "leap",
            "dove",
            "dive",
            "whispered",
            "crept",
            "crawled",
            "stumbled",
            "collapsed",
            "darted",
        ]
        .into_iter()
        .collect()
    })
}

// ---------------------------------------------------------------------------
// PacingMetrics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PacingMetrics {
    pub scene_id: usize,
    pub dialogue_ratio: f64,
    pub is_talking_head: bool,
    pub velocity: f64,
    pub word_count: usize,
    pub entity_density: f64,
    pub noun_density: f64,
    pub information_novelty: f64,
    pub action_density: f64,
}

// ---------------------------------------------------------------------------
// AggregatedPacing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AggregatedPacing {
    pub average_dialogue_ratio: f64,
    pub average_velocity: f64,
    pub per_scene_velocity: Vec<f64>,
    pub per_scene_dialogue_ratio: Vec<f64>,
    pub per_scene_label: Vec<String>,
    pub per_scene_assessment: Vec<String>,
    pub talking_head_scenes: Vec<usize>,
    pub ths_count: usize,
    pub pacing_trend: String,
    pub velocity_variance: f64,
    pub slow_scenes: Vec<usize>,
    pub fast_scenes: Vec<usize>,
    pub per_scene_entity_density: Vec<f64>,
    pub per_scene_action_density: Vec<f64>,
    pub per_scene_novelty: Vec<f64>,
    pub avg_entity_density: f64,
    pub avg_action_density: f64,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Returns a reference to the static action verbs set (exposed for reuse).
pub fn get_action_verbs() -> &'static HashSet<&'static str> {
    action_verbs()
}

/// Analyze a single scene and return its pacing metrics.
pub fn analyze_scene(
    scene_id: usize,
    dialogue_tokens: usize,
    total_tokens: usize,
    sentences_count: usize,
) -> PacingMetrics {
    let dialogue_ratio = if total_tokens > 0 {
        dialogue_tokens as f64 / total_tokens as f64
    } else {
        0.0
    };
    let is_talking_head = dialogue_ratio > THS_THRESHOLD && total_tokens > MIN_WORD_COUNT;
    let velocity = total_tokens as f64 / sentences_count.max(1) as f64;

    PacingMetrics {
        scene_id,
        dialogue_ratio,
        is_talking_head,
        velocity,
        word_count: total_tokens,
        entity_density: 0.0,
        noun_density: 0.0,
        information_novelty: 0.0,
        action_density: 0.0,
    }
}

/// Information density result for a single scene.
#[derive(Debug, Clone)]
pub struct SceneInfoDensity {
    pub entity_density: f64,
    pub action_density: f64,
    pub noun_density: f64,
    pub new_noun_count: usize,
    pub total_noun_count: usize,
}

/// Analyze information density of a scene's text.
///
/// `seen_nouns` tracks nouns from prior scenes; new nouns found here are added to it.
/// Returns entity density (capitalized non-sentence-start words per 100 words),
/// action verb density (per 100 words), and novelty (fraction of nouns not previously seen).
pub fn analyze_scene_info_density(
    scene_text: &str,
    seen_nouns: &mut HashSet<String>,
) -> SceneInfoDensity {
    let words: Vec<&str> = scene_text.split_whitespace().collect();
    let word_count = words.len();
    if word_count == 0 {
        return SceneInfoDensity {
            entity_density: 0.0,
            action_density: 0.0,
            noun_density: 0.0,
            new_noun_count: 0,
            total_noun_count: 0,
        };
    }

    let verbs = action_verbs();
    let mut entity_count = 0usize;
    let mut action_count = 0usize;
    let mut scene_nouns = Vec::new();
    let mut at_sentence_start = true;

    for &w in &words {
        let cleaned: String = w.chars().filter(|c| c.is_alphabetic()).collect();
        if cleaned.is_empty() {
            continue;
        }
        let lower = cleaned.to_lowercase();

        // Action verb detection
        if verbs.contains(lower.as_str()) {
            action_count += 1;
        }

        // Entity detection: capitalized words not at sentence start
        let Some(first_char) = cleaned.chars().next() else {
            continue;
        };
        if first_char.is_uppercase() && !at_sentence_start {
            entity_count += 1;
        }

        // Heuristic noun detection: words >= 4 chars, lowercase, not common function words
        if cleaned.len() >= 4 && first_char.is_lowercase() && !is_function_word(&lower) {
            scene_nouns.push(lower);
        }

        // Track sentence boundaries
        at_sentence_start = w.ends_with('.') || w.ends_with('!') || w.ends_with('?');
    }

    let total_noun_count = scene_nouns.len();
    let new_noun_count = scene_nouns
        .iter()
        .filter(|n| !seen_nouns.contains(n.as_str()))
        .count();

    // Add this scene's nouns to the running set
    for n in scene_nouns {
        seen_nouns.insert(n);
    }

    SceneInfoDensity {
        entity_density: round2(entity_count as f64 / word_count as f64 * 100.0),
        action_density: round2(action_count as f64 / word_count as f64 * 100.0),
        noun_density: round2(total_noun_count as f64 / word_count as f64 * 100.0),
        new_noun_count,
        total_noun_count,
    }
}

/// Enrich a `PacingMetrics` with info density values computed from scene text.
pub fn enrich_with_info_density(metrics: &mut PacingMetrics, info: &SceneInfoDensity) {
    metrics.entity_density = info.entity_density;
    metrics.action_density = info.action_density;
    metrics.noun_density = info.noun_density;
    metrics.information_novelty = if info.total_noun_count > 0 {
        info.new_noun_count as f64 / info.total_noun_count as f64
    } else {
        0.0
    };
}

fn is_function_word(w: &str) -> bool {
    matches!(
        w,
        "the"
            | "that"
            | "this"
            | "with"
            | "from"
            | "have"
            | "been"
            | "were"
            | "will"
            | "would"
            | "could"
            | "should"
            | "shall"
            | "they"
            | "them"
            | "their"
            | "there"
            | "then"
            | "than"
            | "what"
            | "when"
            | "where"
            | "which"
            | "while"
            | "about"
            | "into"
            | "over"
            | "after"
            | "before"
            | "between"
            | "through"
            | "under"
            | "again"
            | "also"
            | "just"
            | "only"
            | "very"
            | "some"
            | "more"
            | "most"
            | "much"
            | "many"
            | "such"
            | "each"
            | "every"
            | "other"
            | "another"
            | "both"
            | "same"
            | "like"
            | "even"
            | "still"
            | "already"
            | "back"
            | "well"
            | "here"
            | "being"
            | "does"
            | "done"
            | "going"
            | "come"
            | "came"
            | "went"
            | "said"
            | "told"
            | "made"
            | "knew"
            | "thought"
            | "looked"
            | "seemed"
            | "felt"
            | "asked"
    )
}

/// Aggregate pacing metrics across all scenes.
///
/// Returns `None` when the input slice is empty.
pub fn aggregate_pacing(scene_metrics: &[PacingMetrics]) -> Option<AggregatedPacing> {
    debug!(
        "[pacing] aggregate_pacing — {} scene metrics",
        scene_metrics.len()
    );
    if scene_metrics.is_empty() {
        return None;
    }

    let len = scene_metrics.len() as f64;
    let ratios: Vec<f64> = scene_metrics.iter().map(|m| m.dialogue_ratio).collect();
    let velocities: Vec<f64> = scene_metrics.iter().map(|m| m.velocity).collect();
    let ths_scenes: Vec<usize> = scene_metrics
        .iter()
        .filter(|m| m.is_talking_head)
        .map(|m| m.scene_id)
        .collect();

    let avg_vel = velocities.iter().sum::<f64>() / len;
    let variance = (velocities
        .iter()
        .map(|v| (v - avg_vel).powi(2))
        .sum::<f64>()
        / len)
        .sqrt();

    let mid = velocities.len() / 2;
    let pacing_trend = if mid > 0 {
        let first_half_avg = velocities[..mid].iter().sum::<f64>() / mid as f64;
        let second_half = &velocities[mid..];
        let second_half_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;
        let diff = second_half_avg - first_half_avg;
        if diff > avg_vel * 0.1 {
            "decelerating".to_string()
        } else if diff < -avg_vel * 0.1 {
            "accelerating".to_string()
        } else {
            "steady".to_string()
        }
    } else {
        "steady".to_string()
    };

    let slow_scenes: Vec<usize> = scene_metrics
        .iter()
        .filter(|m| m.velocity > 30.0)
        .map(|m| m.scene_id)
        .collect();
    let fast_scenes: Vec<usize> = scene_metrics
        .iter()
        .filter(|m| m.velocity < 12.0)
        .map(|m| m.scene_id)
        .collect();

    let mut labels = Vec::with_capacity(scene_metrics.len());
    let mut assessments = Vec::with_capacity(scene_metrics.len());
    for m in scene_metrics {
        let v = m.velocity;
        if v < 12.0 {
            labels.push("fast".to_string());
            assessments.push(format!(
                "Fast pace, avg {:.0} words/sentence; punchy, urgent feel",
                v
            ));
        } else if v <= 25.0 {
            labels.push("good".to_string());
            assessments.push(format!("Balanced pace, avg {:.0} words/sentence", v));
        } else if v <= 35.0 {
            labels.push("slow".to_string());
            assessments.push(format!(
                "Slow pace, avg {:.0} words/sentence; consider varying sentence length",
                v
            ));
        } else {
            labels.push("very_slow".to_string());
            assessments.push(format!(
                "Very slow, avg {:.0} words/sentence; readers may lose focus",
                v
            ));
        }
        if m.is_talking_head
            && let Some(last) = assessments.last_mut() {
                last.push_str("; dialogue-heavy with minimal action");
            }
    }

    Some(AggregatedPacing {
        average_dialogue_ratio: round4(ratios.iter().sum::<f64>() / len),
        average_velocity: round2(avg_vel),
        per_scene_velocity: velocities.iter().map(|v| round2(*v)).collect(),
        per_scene_dialogue_ratio: ratios.iter().map(|r| round4(*r)).collect(),
        per_scene_label: labels,
        per_scene_assessment: assessments,
        talking_head_scenes: ths_scenes.clone(),
        ths_count: ths_scenes.len(),
        pacing_trend,
        velocity_variance: round4(variance),
        slow_scenes,
        fast_scenes,
        per_scene_entity_density: scene_metrics
            .iter()
            .map(|m| round2(m.entity_density))
            .collect(),
        per_scene_action_density: scene_metrics
            .iter()
            .map(|m| round2(m.action_density))
            .collect(),
        per_scene_novelty: scene_metrics
            .iter()
            .map(|m| round3(m.information_novelty))
            .collect(),
        avg_entity_density: round2(
            scene_metrics.iter().map(|m| m.entity_density).sum::<f64>() / len,
        ),
        avg_action_density: round2(
            scene_metrics.iter().map(|m| m.action_density).sum::<f64>() / len,
        ),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        assert!(aggregate_pacing(&[]).is_none());
    }

    #[test]
    fn test_single_scene() {
        let m = analyze_scene(0, 100, 1000, 50);
        assert_eq!(m.scene_id, 0);
        assert!((m.dialogue_ratio - 0.1).abs() < 1e-9);
        assert!(!m.is_talking_head);
        assert!((m.velocity - 20.0).abs() < 1e-9);
        assert_eq!(m.word_count, 1000);

        let agg = aggregate_pacing(&[m]).unwrap();
        assert_eq!(agg.ths_count, 0);
        assert_eq!(agg.pacing_trend, "steady");
        assert_eq!(agg.per_scene_label.len(), 1);
        assert_eq!(agg.per_scene_label[0], "good");
    }

    #[test]
    fn test_ths_detection() {
        // High dialogue ratio + enough words => talking head
        let m = analyze_scene(1, 900, 1000, 50);
        assert!(m.dialogue_ratio > THS_THRESHOLD);
        assert!(m.is_talking_head);

        // High ratio but too few words => not talking head
        let m2 = analyze_scene(2, 90, 100, 5);
        assert!(m2.dialogue_ratio > THS_THRESHOLD);
        assert!(!m2.is_talking_head);
    }

    #[test]
    fn test_zero_tokens() {
        let m = analyze_scene(0, 0, 0, 0);
        assert!((m.dialogue_ratio).abs() < 1e-9);
        assert!((m.velocity).abs() < 1e-9);
    }

    #[test]
    fn test_pacing_trend_decelerating() {
        // First half fast (low velocity), second half slow (high velocity)
        let scenes: Vec<PacingMetrics> = vec![
            analyze_scene(0, 0, 100, 10), // velocity 10
            analyze_scene(1, 0, 100, 10), // velocity 10
            analyze_scene(2, 0, 400, 10), // velocity 40
            analyze_scene(3, 0, 400, 10), // velocity 40
        ];
        let agg = aggregate_pacing(&scenes).unwrap();
        assert_eq!(agg.pacing_trend, "decelerating");
    }

    #[test]
    fn test_pacing_trend_accelerating() {
        // First half slow, second half fast
        let scenes: Vec<PacingMetrics> = vec![
            analyze_scene(0, 0, 400, 10), // velocity 40
            analyze_scene(1, 0, 400, 10), // velocity 40
            analyze_scene(2, 0, 100, 10), // velocity 10
            analyze_scene(3, 0, 100, 10), // velocity 10
        ];
        let agg = aggregate_pacing(&scenes).unwrap();
        assert_eq!(agg.pacing_trend, "accelerating");
    }

    #[test]
    fn test_pacing_trend_steady() {
        let scenes: Vec<PacingMetrics> = vec![
            analyze_scene(0, 0, 200, 10), // velocity 20
            analyze_scene(1, 0, 200, 10), // velocity 20
            analyze_scene(2, 0, 200, 10), // velocity 20
            analyze_scene(3, 0, 200, 10), // velocity 20
        ];
        let agg = aggregate_pacing(&scenes).unwrap();
        assert_eq!(agg.pacing_trend, "steady");
    }

    #[test]
    fn test_slow_and_fast_scenes() {
        let scenes: Vec<PacingMetrics> = vec![
            analyze_scene(0, 0, 50, 5),   // velocity 10 => fast
            analyze_scene(1, 0, 200, 10), // velocity 20 => good
            analyze_scene(2, 0, 400, 10), // velocity 40 => very_slow, also slow_scenes (>30)
        ];
        let agg = aggregate_pacing(&scenes).unwrap();
        assert_eq!(agg.fast_scenes, vec![0]);
        assert_eq!(agg.slow_scenes, vec![2]);
        assert_eq!(agg.per_scene_label[0], "fast");
        assert_eq!(agg.per_scene_label[1], "good");
        assert_eq!(agg.per_scene_label[2], "very_slow");
        assert!(!agg.per_scene_assessment[0].is_empty());
    }

    #[test]
    fn test_talking_head_aggregation() {
        let scenes: Vec<PacingMetrics> = vec![
            analyze_scene(0, 900, 1000, 50), // THS
            analyze_scene(1, 100, 1000, 50), // not THS
            analyze_scene(2, 870, 1000, 50), // THS
        ];
        let agg = aggregate_pacing(&scenes).unwrap();
        assert_eq!(agg.talking_head_scenes, vec![0, 2]);
        assert_eq!(agg.ths_count, 2);
    }

    #[test]
    fn test_action_verbs_set() {
        let verbs = get_action_verbs();
        assert!(verbs.contains("ran"));
        assert!(verbs.contains("sprinted"));
        assert!(verbs.contains("darted"));
        assert!(!verbs.contains("walked"));
    }

    #[test]
    fn test_velocity_variance() {
        // All same velocity => variance 0
        let scenes: Vec<PacingMetrics> =
            vec![analyze_scene(0, 0, 200, 10), analyze_scene(1, 0, 200, 10)];
        let agg = aggregate_pacing(&scenes).unwrap();
        assert!((agg.velocity_variance).abs() < 1e-9);
    }

    #[test]
    fn test_scene_label_slow() {
        // velocity 30 => good (<=25 is good, <=35 is slow)
        let m = analyze_scene(0, 0, 300, 10); // velocity 30
        let agg = aggregate_pacing(&[m]).unwrap();
        assert_eq!(agg.per_scene_label[0], "slow");
    }

    #[test]
    fn test_info_density_empty() {
        let mut seen = HashSet::new();
        let info = analyze_scene_info_density("", &mut seen);
        assert!((info.entity_density).abs() < 1e-9);
        assert!((info.action_density).abs() < 1e-9);
        assert_eq!(info.new_noun_count, 0);
    }

    #[test]
    fn test_info_density_entities() {
        let mut seen = HashSet::new();
        // "John" and "Sarah" after sentence starts should be entities
        let text = "The door opened. John walked in. He saw Sarah near the window.";
        let info = analyze_scene_info_density(text, &mut seen);
        assert!(
            info.entity_density > 0.0,
            "expected entity density > 0, got {}",
            info.entity_density
        );
    }

    #[test]
    fn test_info_density_action_verbs() {
        let mut seen = HashSet::new();
        let text =
            "He ran down the hall. She jumped over the fence. They grabbed the rope and climbed.";
        let info = analyze_scene_info_density(text, &mut seen);
        assert!(
            info.action_density > 0.0,
            "expected action density > 0, got {}",
            info.action_density
        );
    }

    #[test]
    fn test_info_novelty_decreases() {
        let mut seen = HashSet::new();
        let scene1 = "The castle stood on a mountain. The knight carried a sword and shield.";
        let scene2 = "The castle stood on a mountain. The knight carried a sword and shield.";
        let info1 = analyze_scene_info_density(scene1, &mut seen);
        let info2 = analyze_scene_info_density(scene2, &mut seen);
        assert!(
            info1.new_noun_count > 0,
            "first scene should introduce new nouns"
        );
        // Second scene repeats same nouns, so novelty should be lower
        let novelty1 = if info1.total_noun_count > 0 {
            info1.new_noun_count as f64 / info1.total_noun_count as f64
        } else {
            0.0
        };
        let novelty2 = if info2.total_noun_count > 0 {
            info2.new_noun_count as f64 / info2.total_noun_count as f64
        } else {
            0.0
        };
        assert!(
            novelty2 <= novelty1,
            "repeated scene should have equal or lower novelty ({} vs {})",
            novelty2,
            novelty1
        );
    }

    #[test]
    fn test_enrich_with_info_density() {
        let mut m = analyze_scene(0, 50, 500, 25);
        assert!((m.entity_density).abs() < 1e-9);
        let info = SceneInfoDensity {
            entity_density: 3.5,
            action_density: 2.1,
            noun_density: 15.0,
            new_noun_count: 10,
            total_noun_count: 20,
        };
        enrich_with_info_density(&mut m, &info);
        assert!((m.entity_density - 3.5).abs() < 1e-9);
        assert!((m.action_density - 2.1).abs() < 1e-9);
        assert!((m.information_novelty - 0.5).abs() < 1e-9);
    }
}
