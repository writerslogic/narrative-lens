#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExplainedFinding {
    pub finding_type: String,
    pub severity: String,
    pub title: String,
    pub explanation: String,
    pub evidence: String,
    pub suggestion: String,
    pub confidence: f64,
}

pub fn explain_readability(
    fkgl: f64,
    target_min: f64,
    target_max: f64,
    genre: &str,
) -> Option<ExplainedFinding> {
    if fkgl > target_max {
        Some(ExplainedFinding {
            finding_type: "readability".into(),
            severity: "suggestion".into(),
            title: "Prose complexity above target range".into(),
            explanation: format!(
                "Your prose reads at a grade {:.1} level, which is above the typical range \
                 for {} ({:.0}-{:.0}). Readers of {} expect more accessible language. \
                 Consider shortening sentences and using simpler vocabulary in exposition.",
                fkgl, genre, target_min, target_max, genre
            ),
            evidence: format!(
                "FKGL score: {:.1} (target: {:.0}-{:.0})",
                fkgl, target_min, target_max
            ),
            suggestion: "Shorten sentences and replace multisyllabic words in exposition passages."
                .into(),
            confidence: clamp_confidence((fkgl - target_max) / 3.0 + 0.5),
        })
    } else if fkgl < target_min {
        Some(ExplainedFinding {
            finding_type: "readability".into(),
            severity: "info".into(),
            title: "Prose complexity below target range".into(),
            explanation: format!(
                "Your prose reads at a grade {:.1} level, which is below typical for {}. \
                 This may feel too simple for your audience.",
                fkgl, genre
            ),
            evidence: format!(
                "FKGL score: {:.1} (target: {:.0}-{:.0})",
                fkgl, target_min, target_max
            ),
            suggestion: "Consider varying sentence length and using more precise vocabulary."
                .into(),
            confidence: clamp_confidence((target_min - fkgl) / 3.0 + 0.5),
        })
    } else {
        None
    }
}

pub fn explain_pacing(
    velocity: f64,
    scene_id: usize,
    is_talking_head: bool,
    dialogue_ratio: f64,
) -> Option<ExplainedFinding> {
    if velocity > 30.0 {
        Some(ExplainedFinding {
            finding_type: "pacing".into(),
            severity: "suggestion".into(),
            title: format!("Scene {} has very long sentences", scene_id),
            explanation: format!(
                "Scene {} averages {:.0} words per sentence. Long sentences slow the reading \
                 pace and can lose the reader's attention. This works for contemplative scenes \
                 but not action sequences.",
                scene_id, velocity
            ),
            evidence: format!("Average sentence length: {:.1} words", velocity),
            suggestion:
                "Break long sentences into shorter ones, especially during high-tension moments."
                    .into(),
            confidence: clamp_confidence((velocity - 30.0) / 10.0 + 0.6),
        })
    } else if is_talking_head {
        let pct = (dialogue_ratio * 100.0).round() as u32;
        Some(ExplainedFinding {
            finding_type: "pacing".into(),
            severity: "suggestion".into(),
            title: format!("Scene {} is a talking-head scene", scene_id),
            explanation: format!(
                "Scene {} is {}% dialogue with minimal action. Consider grounding the \
                 conversation in physical setting and character movement.",
                scene_id, pct
            ),
            evidence: format!("Dialogue ratio: {}%", pct),
            suggestion: "Add beats of physical action, setting detail, or internal thought between dialogue lines.".into(),
            confidence: clamp_confidence(dialogue_ratio),
        })
    } else {
        None
    }
}

pub fn explain_tension(
    tension: f64,
    prev_tension: f64,
    scene_id: usize,
    flat_count: usize,
    flat_start: usize,
    total_scenes: usize,
    max_tension: f64,
) -> Option<ExplainedFinding> {
    if flat_count >= 3 {
        let flat_end = flat_start + flat_count - 1;
        Some(ExplainedFinding {
            finding_type: "tension".into(),
            severity: "warning".into(),
            title: format!("Flat tension for {} consecutive scenes", flat_count),
            explanation: format!(
                "Tension has been flat for {} consecutive scenes (scenes {}-{}). \
                 Readers need tension variation to stay engaged. Consider introducing \
                 a complication, revelation, or time pressure.",
                flat_count, flat_start, flat_end
            ),
            evidence: format!(
                "Tension level ~{:.2} across scenes {}-{}",
                tension, flat_start, flat_end
            ),
            suggestion: "Introduce a new obstacle, reveal information, or raise the stakes.".into(),
            confidence: clamp_confidence(0.5 + flat_count as f64 * 0.1),
        })
    } else if prev_tension > 0.0 && (prev_tension - tension) > 0.3 {
        Some(ExplainedFinding {
            finding_type: "tension".into(),
            severity: "info".into(),
            title: format!("Sharp tension drop at scene {}", scene_id),
            explanation: format!(
                "Tension drops from {:.2} to {:.2} at scene {}. If this is intentional \
                 (breather after climax), ensure the next scene rebuilds momentum.",
                prev_tension, tension, scene_id
            ),
            evidence: format!(
                "Tension: {:.2} -> {:.2} (drop of {:.2})",
                prev_tension, tension, prev_tension - tension
            ),
            suggestion: "If this drop is unintentional, add a subplot thread or lingering threat to maintain engagement.".into(),
            confidence: clamp_confidence(0.5 + (prev_tension - tension)),
        })
    } else if total_scenes > 0 && max_tension < 0.6 {
        let pct_75 = (total_scenes as f64 * 0.75).round() as usize;
        let pct_90 = (total_scenes as f64 * 0.90).round() as usize;
        Some(ExplainedFinding {
            finding_type: "tension".into(),
            severity: "concern".into(),
            title: "No significant tension peak".into(),
            explanation: format!(
                "The manuscript hasn't reached a significant tension peak. Most successful \
                 novels peak between 75-90% of the way through (scenes {}-{}).",
                pct_75, pct_90
            ),
            evidence: format!(
                "Maximum tension: {:.2} across {} scenes",
                max_tension, total_scenes
            ),
            suggestion:
                "Build toward a climactic confrontation or revelation in the final quarter.".into(),
            confidence: clamp_confidence(0.4 + (0.6 - max_tension)),
        })
    } else {
        None
    }
}

pub fn explain_tone_shift(
    from_tone: &str,
    to_tone: &str,
    scene_id: usize,
    genre: &str,
) -> Option<ExplainedFinding> {
    if from_tone == to_tone {
        return None;
    }

    let is_thriller = matches!(
        genre.to_lowercase().as_str(),
        "thriller" | "suspense" | "mystery" | "horror"
    );

    if is_thriller {
        Some(ExplainedFinding {
            finding_type: "tone_shift".into(),
            severity: "concern".into(),
            title: format!("Abrupt tonal shift at scene {}", scene_id),
            explanation: format!(
                "Sudden shift from {} to {} at scene {}. In thrillers, tonal shifts should \
                 be gradual or motivated by a plot event.",
                from_tone, to_tone, scene_id
            ),
            evidence: format!("Tone: {} -> {} at scene {}", from_tone, to_tone, scene_id),
            suggestion: "Bridge the tonal change with a transitional beat or motivating event."
                .into(),
            confidence: 0.7,
        })
    } else {
        Some(ExplainedFinding {
            finding_type: "tone_shift".into(),
            severity: "info".into(),
            title: format!("Tonal shift at scene {}", scene_id),
            explanation: format!(
                "Tonal shift from {} to {} at scene {}. This may be an effective contrast \
                 if intentional.",
                from_tone, to_tone, scene_id
            ),
            evidence: format!("Tone: {} -> {} at scene {}", from_tone, to_tone, scene_id),
            suggestion: "Verify the shift serves the narrative arc; unintentional shifts can jar the reader.".into(),
            confidence: 0.5,
        })
    }
}

pub fn explain_white_room(
    scene_id: usize,
    sensory_count: usize,
    word_count: usize,
) -> Option<ExplainedFinding> {
    if word_count == 0 {
        return None;
    }
    let density = sensory_count as f64 / word_count as f64 * 1000.0;
    if density < 3.0 {
        Some(ExplainedFinding {
            finding_type: "white_room".into(),
            severity: "suggestion".into(),
            title: format!("Scene {} lacks sensory grounding", scene_id),
            explanation: format!(
                "Scene {} has only {} sensory details in {} words. Readers need physical \
                 grounding to feel present. Add sight, sound, smell, touch, or taste details.",
                scene_id, sensory_count, word_count
            ),
            evidence: format!(
                "{} sensory words in {} words ({:.1} per 1000 words)",
                sensory_count, word_count, density
            ),
            suggestion:
                "Weave in at least two senses per scene; smell and touch are especially immersive."
                    .into(),
            confidence: clamp_confidence(0.8 - density / 10.0),
        })
    } else {
        None
    }
}

pub fn explain_show_dont_tell(issue_type: &str, snippet: &str) -> ExplainedFinding {
    match issue_type {
        "narrative_distance" => ExplainedFinding {
            finding_type: "show_dont_tell".into(),
            severity: "suggestion".into(),
            title: "Telling instead of showing emotion".into(),
            explanation: "The narrator is explaining what the character feels rather than \
                showing it through action, dialogue, or physical sensation. \
                'She felt angry' becomes 'Her fists clenched; her jaw tightened.'"
                .into(),
            evidence: snippet.into(),
            suggestion:
                "Replace the named emotion with a physical reaction or action that implies it."
                    .into(),
            confidence: 0.75,
        },
        "filter_word" => ExplainedFinding {
            finding_type: "show_dont_tell".into(),
            severity: "suggestion".into(),
            title: "Filter word creates narrative distance".into(),
            explanation: "'Could see' creates distance between the reader and the character. \
                 Remove the filter and describe directly: 'She could see the flames' \
                 becomes 'Flames licked the curtains.'"
                .to_string(),
            evidence: snippet.into(),
            suggestion: "Remove the filter word and state the observation directly.".into(),
            confidence: 0.8,
        },
        "diluted_action" => ExplainedFinding {
            finding_type: "show_dont_tell".into(),
            severity: "suggestion".into(),
            title: "Diluted action verb".into(),
            explanation: "'Started to run' dilutes the action. Use the direct verb: 'She ran.'"
                .into(),
            evidence: snippet.into(),
            suggestion: "Replace 'started to/began to' with the direct action verb.".into(),
            confidence: 0.85,
        },
        _ => ExplainedFinding {
            finding_type: "show_dont_tell".into(),
            severity: "info".into(),
            title: "Show-don't-tell issue".into(),
            explanation: format!(
                "A '{}' pattern was detected that may weaken the prose.",
                issue_type
            ),
            evidence: snippet.into(),
            suggestion: "Review this passage for opportunities to show rather than tell.".into(),
            confidence: 0.5,
        },
    }
}

fn clamp_confidence(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_readability_in_range() {
        assert!(explain_readability(8.0, 6.0, 10.0, "thriller").is_none());
    }

    #[test]
    fn test_readability_too_high() {
        let f = explain_readability(14.0, 6.0, 10.0, "thriller").unwrap();
        assert_eq!(f.finding_type, "readability");
        assert_eq!(f.severity, "suggestion");
        assert!(f.explanation.contains("grade 14.0"));
        assert!(f.explanation.contains("thriller"));
        assert!(f.confidence > 0.0 && f.confidence <= 1.0);
    }

    #[test]
    fn test_readability_too_low() {
        let f = explain_readability(3.0, 6.0, 10.0, "literary fiction").unwrap();
        assert_eq!(f.severity, "info");
        assert!(f.explanation.contains("below typical"));
    }

    #[test]
    fn test_readability_at_boundary() {
        assert!(explain_readability(6.0, 6.0, 10.0, "romance").is_none());
        assert!(explain_readability(10.0, 6.0, 10.0, "romance").is_none());
    }

    #[test]
    fn test_pacing_normal() {
        assert!(explain_pacing(18.0, 3, false, 0.3).is_none());
    }

    #[test]
    fn test_pacing_slow() {
        let f = explain_pacing(35.0, 5, false, 0.2).unwrap();
        assert_eq!(f.finding_type, "pacing");
        assert!(f.explanation.contains("35 words per sentence"));
    }

    #[test]
    fn test_pacing_talking_head() {
        let f = explain_pacing(15.0, 2, true, 0.75).unwrap();
        assert!(f.explanation.contains("75% dialogue"));
    }

    #[test]
    fn test_pacing_slow_overrides_talking_head() {
        let f = explain_pacing(32.0, 1, true, 0.8).unwrap();
        assert!(f.title.contains("long sentences"));
    }

    #[test]
    fn test_tension_flat() {
        let f = explain_tension(0.4, 0.4, 5, 4, 2, 20, 0.8).unwrap();
        assert_eq!(f.severity, "warning");
        assert!(f.explanation.contains("4 consecutive scenes"));
        assert!(f.explanation.contains("scenes 2-5"));
    }

    #[test]
    fn test_tension_sharp_drop() {
        let f = explain_tension(0.3, 0.8, 10, 0, 0, 20, 0.8).unwrap();
        assert_eq!(f.severity, "info");
        assert!(f.explanation.contains("drops from 0.80 to 0.30"));
    }

    #[test]
    fn test_tension_no_climax() {
        let f = explain_tension(0.4, 0.4, 15, 1, 14, 20, 0.4).unwrap();
        assert_eq!(f.severity, "concern");
        assert!(f.explanation.contains("tension peak"));
    }

    #[test]
    fn test_tension_normal() {
        assert!(explain_tension(0.6, 0.5, 10, 1, 9, 20, 0.8).is_none());
    }

    #[test]
    fn test_tone_shift_same() {
        assert!(explain_tone_shift("dark", "dark", 5, "thriller").is_none());
    }

    #[test]
    fn test_tone_shift_thriller() {
        let f = explain_tone_shift("dark", "comedic", 5, "thriller").unwrap();
        assert_eq!(f.severity, "concern");
        assert!(f.explanation.contains("thrillers"));
    }

    #[test]
    fn test_tone_shift_literary() {
        let f = explain_tone_shift("somber", "hopeful", 3, "literary fiction").unwrap();
        assert_eq!(f.severity, "info");
        assert!(f.explanation.contains("effective contrast"));
    }

    #[test]
    fn test_tone_shift_mystery() {
        let f = explain_tone_shift("tense", "lighthearted", 7, "mystery").unwrap();
        assert_eq!(f.severity, "concern");
    }

    #[test]
    fn test_white_room_sparse() {
        let f = explain_white_room(3, 1, 500).unwrap();
        assert!(f.explanation.contains("1 sensory details"));
        assert!(f.explanation.contains("500 words"));
    }

    #[test]
    fn test_white_room_adequate() {
        assert!(explain_white_room(3, 20, 500).is_none());
    }

    #[test]
    fn test_white_room_empty() {
        assert!(explain_white_room(1, 0, 0).is_none());
    }

    #[test]
    fn test_show_dont_tell_narrative_distance() {
        let f = explain_show_dont_tell("narrative_distance", "She felt angry.");
        assert!(f.explanation.contains("physical sensation"));
        assert_eq!(f.evidence, "She felt angry.");
    }

    #[test]
    fn test_show_dont_tell_filter_word() {
        let f = explain_show_dont_tell("filter_word", "He could see the door.");
        assert!(f.explanation.contains("distance"));
    }

    #[test]
    fn test_show_dont_tell_diluted_action() {
        let f = explain_show_dont_tell("diluted_action", "She started to run.");
        assert!(f.explanation.contains("direct verb"));
    }

    #[test]
    fn test_show_dont_tell_unknown() {
        let f = explain_show_dont_tell("unknown_type", "some text");
        assert_eq!(f.severity, "info");
        assert!(f.explanation.contains("unknown_type"));
    }

    #[test]
    fn test_clamp_confidence() {
        assert_eq!(clamp_confidence(0.5), 0.5);
        assert_eq!(clamp_confidence(-0.1), 0.0);
        assert_eq!(clamp_confidence(1.5), 1.0);
    }
}
