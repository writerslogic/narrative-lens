use serde::{Deserialize, Serialize};

use crate::{Error, Result, substrate::utils::AnalysisAvailability};

pub const MAX_GAP_SCENES: usize = 10_000;
pub const MAX_GAP_SIGNALS: usize = 10_000;
pub const MAX_GAP_METADATA_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_GAP_DESCRIPTION_BYTES: usize = 4096;

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GapType {
    MissingScene,
    MissingBeat,
    MissingEmotionalTransition,
    MissingInformationReveal,
    UnderdevelopedRelationship,
    UnresolvedSubplot,
    UnmotivatedAction,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NarrativeGap {
    pub gap_type: GapType,
    pub description: String,
    pub location: GapLocation,
    pub suggestion: String,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GapLocation {
    pub after_scene: usize,
    pub before_scene: Option<usize>,
    pub affected_characters: Vec<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GapAnalysisResult {
    #[serde(default)]
    pub availability: AnalysisAvailability,
    #[serde(default)]
    pub diagnostics: Vec<String>,
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
    analyze_gaps_checked(
        broken_promises,
        consequenceless_decisions,
        stall_zones,
        unresolved_ironies,
        total_scenes,
        climax_scene,
    )
    .unwrap_or_else(|error| GapAnalysisResult {
        availability: AnalysisAvailability::Unavailable,
        diagnostics: vec![error.to_string()],
        gaps: Vec::new(),
        total_gap_count: 0,
    })
}

/// Validates and projects supplied review signals. It does not discover gaps or
/// establish author intent, consequences, character knowledge, or plot quality.
pub fn analyze_gaps_checked(
    broken_promises: &[(String, usize)],
    consequenceless_decisions: &[(String, usize)],
    stall_zones: &[(usize, usize)],
    unresolved_ironies: &[(String, usize)],
    total_scenes: usize,
    climax_scene: usize,
) -> Result<GapAnalysisResult> {
    let count = [
        broken_promises.len(),
        consequenceless_decisions.len(),
        stall_zones.len(),
        unresolved_ironies.len(),
    ]
    .into_iter()
    .try_fold(0usize, |total, len| total.checked_add(len))
    .ok_or_else(|| Error::InvalidInput("gap signal count overflow".into()))?;
    if total_scenes > MAX_GAP_SCENES
        || count > MAX_GAP_SIGNALS
        || (total_scenes == 0 && (count != 0 || climax_scene != 0))
        || (total_scenes > 0 && climax_scene >= total_scenes)
    {
        return Err(Error::InvalidInput(
            "gap scene/signal count or climax is out of bounds".into(),
        ));
    }
    let mut metadata = 0usize;
    for (description, scene) in broken_promises.iter().chain(unresolved_ironies) {
        if *scene >= total_scenes
            || description.trim().is_empty()
            || description.len() > MAX_GAP_DESCRIPTION_BYTES
            || description.contains('\0')
        {
            return Err(Error::InvalidInput(
                "gap description or scene is invalid".into(),
            ));
        }
        metadata += description.len();
    }
    for (character, scene) in consequenceless_decisions {
        crate::manuscript::config::validate_name(character)?;
        if *scene >= total_scenes {
            return Err(Error::InvalidInput(
                "gap decision scene is out of bounds".into(),
            ));
        }
        metadata += character.len();
    }
    if metadata > MAX_GAP_METADATA_BYTES {
        return Err(Error::InvalidInput(
            "gap metadata exceeds byte limit".into(),
        ));
    }
    for (start, end) in stall_zones {
        if start >= end || *end >= total_scenes {
            return Err(Error::InvalidInput(
                "gap stall range must be increasing and within the manuscript".into(),
            ));
        }
    }
    let mut gaps = Vec::with_capacity(count);
    for (description, scene) in broken_promises {
        gaps.push(NarrativeGap {
            gap_type: GapType::MissingScene,
            description: format!("Supplied unresolved promise candidate '{}' from scene {}", description, scene + 1),
            location: GapLocation { after_scene: *scene,
                before_scene: (*scene < climax_scene).then_some(climax_scene),
                affected_characters: Vec::new() },
            suggestion: "Review whether the promise is intentional, already resolved, or needs a payoff; choose placement using the intended story order.".into(),
        });
    }
    for (character, scene) in consequenceless_decisions {
        let before = (scene + 2).min(total_scenes - 1);
        gaps.push(NarrativeGap {
            gap_type: GapType::MissingScene,
            description: format!("Supplied consequence-review candidate for '{}' in scene {}", character, scene + 1),
            location: GapLocation { after_scene: *scene,
                before_scene: (before > *scene).then_some(before),
                affected_characters: vec![character.clone()] },
            suggestion: "Review the decision's intended consequences, including delayed, implicit, or absent consequences, before adding a scene.".into(),
        });
    }
    for (start, end) in stall_zones {
        gaps.push(NarrativeGap {
            gap_type: GapType::MissingBeat,
            description: format!("Supplied pacing-review range: scenes {}–{} ({} inclusive scenes)", start + 1, end + 1, end - start + 1),
            location: GapLocation { after_scene: *start, before_scene: Some(*end), affected_characters: Vec::new() },
            suggestion: "Review changes within this range and its intended pace before adding a turning point or complication.".into(),
        });
    }
    for (description, scene) in unresolved_ironies {
        gaps.push(NarrativeGap {
            gap_type: GapType::MissingInformationReveal,
            description: format!("Supplied revelation-review candidate '{}' from scene {}", description, scene + 1),
            location: GapLocation { after_scene: *scene,
                before_scene: (*scene < climax_scene).then_some(climax_scene), affected_characters: Vec::new() },
            suggestion: "Check explicit reader/character assertions and intended ambiguity; this signal does not prove a character needs to learn the truth.".into(),
        });
    }
    gaps.sort_by(|a, b| {
        a.location
            .after_scene
            .cmp(&b.location.after_scene)
            .then_with(|| a.location.before_scene.cmp(&b.location.before_scene))
            .then_with(|| a.gap_type.cmp(&b.gap_type))
            .then_with(|| a.description.cmp(&b.description))
            .then_with(|| {
                a.location
                    .affected_characters
                    .cmp(&b.location.affected_characters)
            })
    });
    Ok(GapAnalysisResult {
        availability: AnalysisAvailability::Partial,
        diagnostics: vec![
            "supplied_review_signals_only; no_independent_gap_or_author_intent_inference".into(),
        ],
        total_gap_count: gaps.len(),
        gaps,
    })
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
        assert!(
            result.gaps[0]
                .location
                .affected_characters
                .contains(&"Alice".to_string())
        );
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
