//! Subtext detection is reserved for the reasoner; the keyword heuristic was
//! removed as measured-unreliable on real prose.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubtextType {
    EmotionalDenial,
    PowerPlay,
    SelfDeception,
    CodedCommunication,
    DramaticIronySubtext,
    Deflection,
    VerbalIrony,
    ActionContradiction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubtextInstance {
    pub scene: usize,
    pub paragraph: usize,
    pub surface: String,
    pub implied: String,
    pub subtext_type: SubtextType,
    /// 0.0 = transparent, 1.0 = fully oblique
    pub density: f64,
    pub evidence: Vec<String>,
    pub characters: Vec<String>,
    pub resolved: bool,
    pub resolution_scene: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneSubtext {
    pub scene: usize,
    pub average_density: f64,
    pub instance_count: usize,
    pub dominant_type: Option<SubtextType>,
    pub appropriateness: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubtextAdvice {
    pub scene: usize,
    pub issue_type: String,
    pub quoted_text: String,
    pub explanation: String,
    pub suggestion: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubtextResult {
    pub instances: Vec<SubtextInstance>,
    pub scene_summaries: Vec<SceneSubtext>,
    pub global_density: f64,
    pub genre_calibrated_score: f64,
    pub on_the_nose: Vec<(usize, String)>,
    pub unresolved_subtext: Vec<SubtextInstance>,
    pub type_distribution: HashMap<String, usize>,
    pub character_patterns: HashMap<String, (f64, String)>,
    pub advice: Vec<SubtextAdvice>,
    pub craft_notes: Vec<String>,
}

// ---------------------------------------------------------------------------
// Main analysis function
// ---------------------------------------------------------------------------

pub fn analyze_subtext(
    scenes: &[&str],
    _dialogue_segments: &[(usize, String, String, String)],
    _character_emotional_states: &[(String, usize, String)],
    _genre: &str,
) -> SubtextResult {
    let scene_summaries = scenes
        .iter()
        .enumerate()
        .map(|(i, _)| SceneSubtext {
            scene: i,
            average_density: 0.0,
            instance_count: 0,
            dominant_type: None,
            appropriateness: 1.0,
        })
        .collect();
    SubtextResult {
        instances: Vec::new(),
        scene_summaries,
        global_density: 0.0,
        genre_calibrated_score: 0.0,
        on_the_nose: Vec::new(),
        unresolved_subtext: Vec::new(),
        type_distribution: HashMap::new(),
        character_patterns: HashMap::new(),
        advice: Vec::new(),
        craft_notes: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_analysis_returns_empty() {
        let scenes = vec!["After the argument, she stood trembling. \"I'm fine,\" she said."];
        let dialogue = vec![(
            0_usize,
            "Sarah".to_string(),
            "I'm fine".to_string(),
            "After the argument, she stood trembling".to_string(),
        )];
        let emotions = vec![("Sarah".to_string(), 0_usize, "angry".to_string())];

        let result = analyze_subtext(&scenes, &dialogue, &emotions, "literary fiction");

        assert!(result.instances.is_empty());
        assert_eq!(result.global_density, 0.0);
    }

    #[test]
    fn test_full_analysis_empty() {
        let scenes: Vec<&str> = vec!["A quiet morning. Birds sang outside."];
        let dialogue: Vec<(usize, String, String, String)> = vec![];
        let emotions: Vec<(String, usize, String)> = vec![];

        let result = analyze_subtext(&scenes, &dialogue, &emotions, "literary fiction");

        assert!(result.instances.is_empty());
        assert_eq!(result.global_density, 0.0);
    }
}
