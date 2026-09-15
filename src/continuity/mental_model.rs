//! Reader-belief / dramatic-irony modeling is reserved for future work; the
//! keyword heuristic was removed as measured-unreliable on real prose.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BeliefCategory {
    Identity,
    Motivation,
    Allegiance,
    EventCause,
    Prediction,
    WorldRule,
    EmotionalState,
    Capability,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReaderBelief {
    pub id: usize,
    pub subject: String,
    pub proposition: String,
    pub confidence: f64,
    pub established_at: usize,
    pub last_reinforced: usize,
    pub matches_truth: Option<bool>,
    pub category: BeliefCategory,
    pub salience: f64,
    pub source_reliability: f64,
    pub inference_chain: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DramaticIrony {
    pub knowledge: String,
    pub ignorant_character: String,
    pub established_at: usize,
    pub resolved_at: Option<usize>,
    pub tension: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BeliefState {
    pub after_scene: usize,
    pub beliefs: Vec<ReaderBelief>,
    pub invalidated: Vec<ReaderBelief>,
    pub introduced: Vec<ReaderBelief>,
    pub belief_gap: f64,
    pub gap_delta: f64,
    pub active_ironies: Vec<DramaticIrony>,
    pub archetype_divergence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceChain {
    pub conclusion_id: usize,
    pub premise_ids: Vec<usize>,
    pub strength: f64,
    pub description: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NarratorReliability {
    pub overall_score: f64,
    pub unreliability_evidence: Vec<(usize, String)>,
    pub intentional: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EventType {
    Revelation,
    Twist,
    Confirmation,
    Complication,
    DramaticIronyEstablished,
    DramaticIronyResolved,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NarrativeEvent {
    pub scene: usize,
    pub magnitude: f64,
    pub affected_beliefs: Vec<String>,
    pub event_type: EventType,
    pub preparation_score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReaderExperienceNote {
    pub scene: usize,
    pub note_type: String,
    pub description: String,
    pub craft_implication: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MentalModelResult {
    pub trajectory: Vec<BeliefState>,
    pub revelations: Vec<NarrativeEvent>,
    pub twists: Vec<NarrativeEvent>,
    pub stasis_zones: Vec<(usize, usize)>,
    pub gap_curve: Vec<f64>,
    pub management_score: f64,
    pub dramatic_ironies: Vec<DramaticIrony>,
    pub archetype_divergence_peaks: Vec<(usize, f64, String)>,
    pub narrator_reliability: NarratorReliability,
    pub inference_chains: Vec<InferenceChain>,
    pub reader_experience: Vec<ReaderExperienceNote>,
    pub engagement_diagnosis: String,
}

// ---------------------------------------------------------------------------
// Main analysis function
// ---------------------------------------------------------------------------

pub fn build_mental_model(
    _scenes: &[&str],
    _characters: &[&str],
    _pov_characters: &[&str],
    _theme_keywords: &[&str],
) -> MentalModelResult {
    MentalModelResult {
        trajectory: Vec::new(),
        revelations: Vec::new(),
        twists: Vec::new(),
        stasis_zones: Vec::new(),
        gap_curve: Vec::new(),
        management_score: 0.5,
        dramatic_ironies: Vec::new(),
        archetype_divergence_peaks: Vec::new(),
        narrator_reliability: NarratorReliability {
            overall_score: 1.0,
            unreliability_evidence: Vec::new(),
            intentional: false,
        },
        inference_chains: Vec::new(),
        reader_experience: Vec::new(),
        engagement_diagnosis: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let result = build_mental_model(&[], &[], &[], &[]);
        assert_eq!(result.trajectory.len(), 0);
        assert_eq!(result.management_score, 0.5);
    }

    #[test]
    fn test_single_scene_returns_neutral() {
        let scene = "Alice wanted revenge. She planned to betray Bob.";
        let result = build_mental_model(&[scene], &["Alice", "Bob"], &["Alice"], &["revenge"]);
        assert_eq!(result.trajectory.len(), 0);
        assert_eq!(result.management_score, 0.5);
    }

    #[test]
    fn test_narrator_reliability_is_neutral() {
        let scene = "Perhaps it seemed that one might think Alice was innocent. \
                     Apparently she supposedly had an alibi.";
        let result = build_mental_model(&[scene], &["Alice"], &[""], &[]);
        assert_eq!(result.narrator_reliability.overall_score, 1.0);
    }
}
