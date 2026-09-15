//! Thematic-argument extraction is reserved for the reasoner;
//! the keyword heuristic was removed as measured-unreliable.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThematicThesis {
    pub id: usize,
    pub statement: String,
    pub confidence: f64,
    pub is_primary: bool,
    pub antithesis: Option<String>,
    pub synthesis: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum EvidenceValence {
    Supporting,
    Contradicting,
    Complicating,
    Transcending,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EvidenceDelivery {
    CharacterAction { character: String, decision: String },
    PlotConsequence { cause: String },
    Dialogue { speaker: String },
    Narration,
    Symbolism { symbol: String },
    StructuralParallel { parallel_scene: usize },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThematicEvidence {
    pub scene: usize,
    pub thesis_id: usize,
    pub valence: EvidenceValence,
    pub delivery: EvidenceDelivery,
    pub content: String,
    pub weight: f64,
    pub character: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CharacterThematicRole {
    pub character: String,
    pub thesis_id: usize,
    pub position: EvidenceValence,
    pub articulacy: f64,
    pub fair_hearing: f64,
    /// (start_position, end_position)
    pub thematic_arc: Option<(String, String)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ThemeInteractionType {
    Reinforcing,
    Tensioning,
    Independent,
    Subsumes,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThemeInteraction {
    pub theme_a: usize,
    pub theme_b: usize,
    pub interaction_type: ThemeInteractionType,
    pub key_scenes: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThematicNote {
    pub note_type: String,
    pub description: String,
    pub suggestion: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThematicArgumentResult {
    pub theses: Vec<ThematicThesis>,
    pub evidence: Vec<ThematicEvidence>,
    pub character_roles: Vec<CharacterThematicRole>,
    pub theme_interactions: Vec<ThemeInteraction>,
    pub dialectical_score: f64,
    pub delivery_distribution: HashMap<String, f64>,
    pub embodied_ratio: f64,
    pub conclusion_earned: f64,
    pub evidence_balance: f64,
    pub argument_strength: f64,
    pub act_evolution: Vec<(u32, String)>,
    pub propaganda_risk: f64,
    pub thematic_notes: Vec<ThematicNote>,
}

// ---------------------------------------------------------------------------
// Main analysis function
// ---------------------------------------------------------------------------

/// Track the thematic argument across the manuscript.
///
/// Thematic-argument extraction (thesis/antithesis selection, evidence
/// scoring) is reserved for the reasoner. This entry point returns a
/// neutral empty result; the keyword heuristic was removed as
/// measured-unreliable.
pub fn track_thematic_argument(
    _scenes: &[&str],
    _themes: &[(String, f64)],
    _characters: &[&str],
    _character_decisions: &[(String, usize, String)],
    _genre: &str,
) -> ThematicArgumentResult {
    ThematicArgumentResult {
        theses: Vec::new(),
        evidence: Vec::new(),
        character_roles: Vec::new(),
        theme_interactions: Vec::new(),
        dialectical_score: 0.0,
        delivery_distribution: HashMap::new(),
        embodied_ratio: 0.0,
        conclusion_earned: 0.0,
        evidence_balance: 0.0,
        argument_strength: 0.0,
        act_evolution: Vec::new(),
        propaganda_risk: 0.0,
        thematic_notes: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let result = track_thematic_argument(&[], &[], &[], &[], "literary_fiction");
        assert_eq!(result.theses.len(), 0);
        assert_eq!(result.argument_strength, 0.0);
    }

    #[test]
    fn test_single_theme_basic() {
        let scenes = vec![
            "John loved Mary deeply and sacrificed everything for her.",
            "But love proved destructive when Mary betrayed him.",
        ];
        let themes = vec![("love".to_string(), 0.9)];
        let characters = vec!["John", "Mary"];
        let decisions: Vec<(String, usize, String)> = vec![];

        let result = track_thematic_argument(&scenes, &themes, &characters, &decisions, "romance");

        // Keyword heuristic removed; reasoner placeholder returns neutral result.
        assert_eq!(result.theses.len(), 0);
        assert_eq!(result.evidence.len(), 0);
        assert_eq!(result.argument_strength, 0.0);
    }

    #[test]
    fn test_propaganda_detection() {
        let scenes: Vec<&str> = (0..10)
            .map(|_| "Power enables the hero to save everyone. Power is good and just.")
            .collect();
        let themes = vec![("power".to_string(), 0.8)];
        let characters = vec!["Hero"];
        let decisions: Vec<(String, usize, String)> = vec![];

        let result = track_thematic_argument(&scenes, &themes, &characters, &decisions, "fantasy");

        // Keyword heuristic removed; propaganda_risk is neutral zero.
        assert_eq!(result.propaganda_risk, 0.0);
        assert_eq!(result.theses.len(), 0);
    }

    #[test]
    fn test_dialectical_arc() {
        let scenes = vec![
            "Freedom demands responsibility. The hero accepted this truth.",
            "The hero fought for freedom above all else.",
            "But freedom without responsibility led to chaos and failure.",
            "The villain showed that freedom is illusion when unchecked.",
            "However, the hero learned that both freedom and duty coexist.",
            "Ultimately, the hero realized freedom is neither absolute nor absent.",
        ];
        let themes = vec![("freedom".to_string(), 0.85)];
        let characters = vec!["Hero", "Villain"];
        let decisions = vec![
            ("Hero".to_string(), 0, "accepted responsibility".to_string()),
            (
                "Villain".to_string(),
                3,
                "rejected all constraints".to_string(),
            ),
        ];

        let result = track_thematic_argument(
            &scenes,
            &themes,
            &characters,
            &decisions,
            "literary_fiction",
        );

        // Keyword heuristic removed; returns neutral empty result.
        assert_eq!(result.theses.len(), 0);
        assert_eq!(result.dialectical_score, 0.0);
    }

    #[test]
    fn test_embodied_ratio() {
        let scenes = vec![
            "Alice decided to leave everything behind. She chose freedom.",
            "The result was isolation. Because of her choice, she lost everyone.",
            "\"I know I was right,\" Alice said to the mirror.",
        ];
        let themes = vec![("freedom".to_string(), 0.7)];
        let characters = vec!["Alice"];
        let decisions: Vec<(String, usize, String)> = vec![];

        let result = track_thematic_argument(
            &scenes,
            &themes,
            &characters,
            &decisions,
            "literary_fiction",
        );

        // Keyword heuristic removed; embodied_ratio is neutral zero.
        assert_eq!(result.embodied_ratio, 0.0);
        assert_eq!(result.evidence.len(), 0);
    }

    #[test]
    fn test_theme_interactions() {
        let scenes = vec![
            "Love and power intertwined as the queen sacrificed for her people.",
            "Power corrupts even love. The king chose the throne over family.",
            "In the end, love conquered even the corrupting influence of power.",
        ];
        let themes = vec![("love".to_string(), 0.8), ("power".to_string(), 0.7)];
        let characters = vec!["Queen", "King"];
        let decisions: Vec<(String, usize, String)> = vec![];

        let result = track_thematic_argument(&scenes, &themes, &characters, &decisions, "fantasy");

        // Keyword heuristic removed; theme_interactions is empty.
        assert_eq!(result.theme_interactions.len(), 0);
    }

    #[test]
    fn test_delivery_distribution_sums_to_one() {
        let scenes = vec![
            "The hero decided to fight. He chose war over peace.",
            "Because of the battle, the kingdom fell. The result was ruin.",
            "\"We must rebuild,\" said the elder. \"There is no other way.\"",
            "The shadow of the old tower loomed over the village.",
        ];
        let themes = vec![("power".to_string(), 0.8)];
        let characters = vec!["Hero", "Elder"];
        let decisions: Vec<(String, usize, String)> = vec![];

        let result = track_thematic_argument(&scenes, &themes, &characters, &decisions, "fantasy");

        // Keyword heuristic removed; delivery_distribution is empty (no evidence collected).
        assert!(result.delivery_distribution.is_empty());
    }
}
