use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GapType {
    MissingScene,
    MissingBeat,
    MissingEmotionalTransition,
    MissingInformationReveal,
    UnderdevelopedRelationship,
    UnresolvedSubplot,
    UnmotivatedAction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NarrativeGap {
    pub gap_type: GapType,
    pub description: String,
    pub location: GapLocation,
    pub suggestion: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GapLocation {
    pub after_scene: usize,
    pub before_scene: Option<usize>,
    pub affected_characters: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GapAnalysisResult {
    pub gaps: Vec<NarrativeGap>,
    pub total_gap_count: usize,
}

/// Analyzes narrative gaps based on signals from other analysis modules.
///
/// Inputs:
/// - `broken_promises`: (description, promise_scene) from promise/payoff module
/// - `consequenceless_decisions`: (character, scene) from decision tree module
/// - `stall_zones`: (start_scene, end_scene) from pacing module
/// - `unresolved_ironies`: (description, scene) from expectation module
/// - `total_scenes`: total number of scenes in the manuscript
/// - `climax_scene`: the scene index of the climax
pub fn analyze_gaps(
    broken_promises: &[(String, usize)],
    consequenceless_decisions: &[(String, usize)],
    stall_zones: &[(usize, usize)],
    unresolved_ironies: &[(String, usize)],
    total_scenes: usize,
    climax_scene: usize,
) -> GapAnalysisResult {
    let mut gaps = Vec::new();

    // Broken promises -> MissingScene (need a payoff scene)
    for (description, promise_scene) in broken_promises {
        gaps.push(NarrativeGap {
            gap_type: GapType::MissingScene,
            description: format!("Broken promise: '{}' set up in scene {} has no payoff", description, promise_scene + 1),
            location: GapLocation {
                after_scene: *promise_scene,
                before_scene: if *promise_scene < climax_scene {
                    Some(climax_scene)
                } else {
                    None
                },
                affected_characters: Vec::new(),
            },
            suggestion: format!(
                "Add a payoff scene that resolves the promise '{}'. Place it before the climax for maximum impact.",
                description
            ),
        });
    }

    // Consequenceless decisions -> MissingScene (need consequence scene)
    for (character, scene) in consequenceless_decisions {
        gaps.push(NarrativeGap {
            gap_type: GapType::MissingScene,
            description: format!(
                "Decision by '{}' in scene {} has no visible consequences",
                character, scene + 1
            ),
            location: GapLocation {
                after_scene: *scene,
                before_scene: Some((*scene + 2).min(total_scenes.saturating_sub(1))),
                affected_characters: vec![character.clone()],
            },
            suggestion: format!(
                "Add a scene showing the consequences of {}'s decision. Consequences should complicate their situation, not merely confirm the choice was right.",
                character
            ),
        });
    }

    // Stall zones -> MissingBeat (need something to happen)
    for (start, end) in stall_zones {
        let zone_length = end.saturating_sub(*start);
        gaps.push(NarrativeGap {
            gap_type: GapType::MissingBeat,
            description: format!(
                "Narrative stall zone spanning scenes {}-{} ({} scenes with no significant change)",
                start + 1, end + 1, zone_length
            ),
            location: GapLocation {
                after_scene: *start,
                before_scene: Some(*end),
                affected_characters: Vec::new(),
            },
            suggestion: format!(
                "Insert a turning point or complication within scenes {}-{}. A reversal, revelation, or escalation will restore momentum.",
                start + 1, end + 1
            ),
        });
    }

    // Unresolved ironies -> MissingInformationReveal (character needs to learn)
    for (description, scene) in unresolved_ironies {
        gaps.push(NarrativeGap {
            gap_type: GapType::MissingInformationReveal,
            description: format!(
                "Dramatic irony '{}' established in scene {} is never resolved; the character never learns the truth",
                description, scene + 1
            ),
            location: GapLocation {
                after_scene: *scene,
                before_scene: Some(climax_scene),
                affected_characters: Vec::new(),
            },
            suggestion: format!(
                "Add a revelation scene where the character discovers the truth about '{}'. The later the reveal (closer to climax), the more devastating the impact.",
                description
            ),
        });
    }

    // Sort by scene position (structural document order)
    gaps.sort_by_key(|g| g.location.after_scene);

    let total_gap_count = gaps.len();

    GapAnalysisResult {
        gaps,
        total_gap_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_inputs() {
        let result = analyze_gaps(&[], &[], &[], &[], 20, 15);
        assert_eq!(result.total_gap_count, 0);
        assert!(result.gaps.is_empty());
    }

    #[test]
    fn test_broken_promises_generate_missing_scene() {
        let promises = vec![("the sword will be important".to_string(), 3)];
        let result = analyze_gaps(&promises, &[], &[], &[], 20, 15);
        assert_eq!(result.total_gap_count, 1);
        assert!(matches!(result.gaps[0].gap_type, GapType::MissingScene));
        assert!(result.gaps[0].description.contains("sword"));
    }

    #[test]
    fn test_consequenceless_decisions_generate_gaps() {
        let decisions = vec![("Alice".to_string(), 5)];
        let result = analyze_gaps(&[], &decisions, &[], &[], 20, 15);
        assert_eq!(result.total_gap_count, 1);
        assert!(matches!(result.gaps[0].gap_type, GapType::MissingScene));
        assert!(result.gaps[0]
            .location
            .affected_characters
            .contains(&"Alice".to_string()));
    }

    #[test]
    fn test_stall_zones_generate_missing_beat() {
        let stalls = vec![(8, 12)];
        let result = analyze_gaps(&[], &[], &stalls, &[], 20, 15);
        assert_eq!(result.total_gap_count, 1);
        assert!(matches!(result.gaps[0].gap_type, GapType::MissingBeat));
    }

    #[test]
    fn test_unresolved_ironies_generate_info_reveal() {
        let ironies = vec![("hero doesn't know the villain is his father".to_string(), 4)];
        let result = analyze_gaps(&[], &[], &[], &ironies, 20, 15);
        assert_eq!(result.total_gap_count, 1);
        assert!(matches!(
            result.gaps[0].gap_type,
            GapType::MissingInformationReveal
        ));
    }

    #[test]
    fn test_gaps_sorted_by_scene() {
        let promises = vec![
            ("late promise".to_string(), 14),
            ("early promise".to_string(), 1),
        ];
        let result = analyze_gaps(&promises, &[], &[], &[], 20, 15);
        assert!(result.gaps[0].location.after_scene <= result.gaps[1].location.after_scene);
    }

    #[test]
    fn test_gap_count_matches_inputs() {
        let promises: Vec<(String, usize)> =
            (0..10).map(|i| (format!("promise {}", i), i)).collect();
        let result = analyze_gaps(&promises, &[], &[], &[], 20, 15);
        assert_eq!(result.total_gap_count, 10);
        assert_eq!(result.gaps.len(), 10);
    }
}
