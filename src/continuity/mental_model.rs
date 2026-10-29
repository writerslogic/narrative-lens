//! Inferred reader psychology remains unsupported. Author annotations support
//! exact belief revisions and truth reclassifications without psychological scores.

use serde::{Deserialize, Serialize};

use super::author_knowledge::{
    self, AuthorAssertion, AuthorAudience, AuthorKnowledgeAction, AuthorKnowledgeClassification,
    AuthorKnowledgeEvent, AuthorKnowledgeLedger, AuthorProposition, AuthorUptake, TruthValue,
};
use crate::{
    Result,
    manuscript::{Evidence, Manuscript},
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
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

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DramaticIrony {
    pub knowledge: String,
    pub ignorant_character: String,
    pub established_at: usize,
    pub resolved_at: Option<usize>,
    pub tension: f64,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceChain {
    pub conclusion_id: usize,
    pub premise_ids: Vec<usize>,
    pub strength: f64,
    pub description: String,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NarratorReliability {
    pub overall_score: f64,
    pub unreliability_evidence: Vec<(usize, String)>,
    pub intentional: bool,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EventType {
    Revelation,
    Twist,
    Confirmation,
    Complication,
    DramaticIronyEstablished,
    DramaticIronyResolved,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NarrativeEvent {
    pub scene: usize,
    pub magnitude: f64,
    pub affected_beliefs: Vec<String>,
    pub event_type: EventType,
    pub preparation_score: f64,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReaderExperienceNote {
    pub scene: usize,
    pub note_type: String,
    pub description: String,
    pub craft_implication: String,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MentalModelResult {
    #[serde(default)]
    pub availability: crate::substrate::utils::AnalysisAvailability,
    #[serde(default)]
    pub diagnostics: Vec<String>,

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
        availability: crate::substrate::utils::AnalysisAvailability::Unavailable,
        diagnostics: vec!["reader_mental_model_not_implemented".into()],
        trajectory: Vec::new(),
        revelations: Vec::new(),
        twists: Vec::new(),
        stasis_zones: Vec::new(),
        gap_curve: Vec::new(),
        management_score: 0.5,
        dramatic_ironies: Vec::new(),
        archetype_divergence_peaks: Vec::new(),
        narrator_reliability: NarratorReliability {
            overall_score: 0.5,
            unreliability_evidence: Vec::new(),
            intentional: false,
        },
        inference_chains: Vec::new(),
        reader_experience: Vec::new(),
        engagement_diagnosis: "Reader mental-model analysis is unavailable.".into(),
    }
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorReaderChangeKind {
    AssertionChanged,
    RepeatedAssertion,
    TruthReclassification,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorReaderChange {
    pub event_id: String,
    pub sequence: u32,
    pub proposition_id: String,
    pub kind: AuthorReaderChangeKind,
    pub before: AuthorAssertion,
    pub after: AuthorAssertion,
    pub before_classification: AuthorKnowledgeClassification,
    pub after_classification: AuthorKnowledgeClassification,
    pub truth_before: TruthValue,
    pub truth_after: TruthValue,
    pub previous_event_id: Option<String>,
    pub evidence: Evidence,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorReaderSummary {
    pub annotated_propositions: usize,
    pub unassessed_propositions: usize,
    pub knowledge: usize,
    pub correct_beliefs: usize,
    pub false_beliefs: usize,
    pub unverified_beliefs: usize,
    pub unverified_knowledge: usize,
    pub contradicted_knowledge: usize,
    pub uncertain: usize,
    pub unknown: usize,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorMentalModelResult {
    pub knowledge: AuthorKnowledgeLedger,
    pub reader_changes: Vec<AuthorReaderChange>,
    pub reader_summary: AuthorReaderSummary,
}

/// Project supplied annotations in their one declared replay order. A truth
/// reclassification changes the assessment of a reader assertion, not what the
/// reader has learned. No narrator reliability, suspense, intent or engagement
/// estimate is inferred from these annotations.
pub fn build_author_mental_model(
    manuscript: &Manuscript,
    propositions: &[AuthorProposition],
    events: &[AuthorKnowledgeEvent],
) -> Result<AuthorMentalModelResult> {
    use std::collections::BTreeMap;
    let knowledge = author_knowledge::analyze_author_knowledge(manuscript, propositions, events)?;
    let mut truths: BTreeMap<_, _> = knowledge
        .propositions
        .iter()
        .map(|p| (p.id.as_str(), p.initial_truth))
        .collect();
    let mut reader: BTreeMap<&str, (AuthorAssertion, String)> = BTreeMap::new();
    let mut reader_changes = Vec::new();
    for transition in &knowledge.transitions {
        let id = transition.proposition_id.as_str();
        let truth_before = truths[id];
        let previous = reader.get(id);
        let before = previous.map_or(AuthorAssertion::Unassessed, |state| state.0);
        let previous_event_id = previous.map(|state| state.1.clone());
        let mut truth_after = truth_before;
        let mut is_truth_change = false;
        let after = match &transition.action {
            AuthorKnowledgeAction::Truth { value } => {
                truth_after = *value;
                truths.insert(id, *value);
                is_truth_change = true;
                before
            }
            AuthorKnowledgeAction::Know {
                actor: AuthorAudience::Reader,
                value,
            } => AuthorAssertion::Knowledge { value: *value },
            AuthorKnowledgeAction::Believe {
                actor: AuthorAudience::Reader,
                value,
            } => AuthorAssertion::Belief { value: *value },
            AuthorKnowledgeAction::Uncertain {
                actor: AuthorAudience::Reader,
            } => AuthorAssertion::Uncertain,
            AuthorKnowledgeAction::Forget {
                actor: AuthorAudience::Reader,
            } => AuthorAssertion::Unknown,
            AuthorKnowledgeAction::Communicate {
                recipient: AuthorAudience::Reader,
                value,
                uptake,
                ..
            } => match uptake {
                AuthorUptake::Unchanged => continue,
                AuthorUptake::Belief => AuthorAssertion::Belief { value: *value },
                AuthorUptake::Knowledge => AuthorAssertion::Knowledge { value: *value },
            },
            _ => continue,
        };
        let before_classification = author_knowledge::classify(before, truth_before);
        let after_classification = author_knowledge::classify(after, truth_after);
        let kind = if is_truth_change {
            if before == AuthorAssertion::Unassessed
                || before_classification == after_classification
            {
                continue;
            }
            AuthorReaderChangeKind::TruthReclassification
        } else {
            reader.insert(id, (after, transition.event_id.clone()));
            if before == after {
                AuthorReaderChangeKind::RepeatedAssertion
            } else {
                AuthorReaderChangeKind::AssertionChanged
            }
        };
        reader_changes.push(AuthorReaderChange {
            event_id: transition.event_id.clone(),
            sequence: transition.sequence,
            proposition_id: transition.proposition_id.clone(),
            kind,
            before,
            after,
            before_classification,
            after_classification,
            truth_before,
            truth_after,
            previous_event_id,
            evidence: transition.evidence.clone(),
        });
    }
    let mut reader_summary = AuthorReaderSummary::default();
    for state in knowledge
        .states
        .iter()
        .filter(|s| s.actor == AuthorAudience::Reader)
    {
        use AuthorKnowledgeClassification as C;
        reader_summary.annotated_propositions += 1;
        match state.classification {
            C::Knowledge => reader_summary.knowledge += 1,
            C::CorrectBelief => reader_summary.correct_beliefs += 1,
            C::FalseBelief => reader_summary.false_beliefs += 1,
            C::UnverifiedBelief => reader_summary.unverified_beliefs += 1,
            C::UnverifiedKnowledge => reader_summary.unverified_knowledge += 1,
            C::ContradictedKnowledge => reader_summary.contradicted_knowledge += 1,
            C::Uncertain => reader_summary.uncertain += 1,
            C::Unknown => reader_summary.unknown += 1,
            C::Unassessed => unreachable!("validated ledger stores only explicit actor assertions"),
        }
    }
    reader_summary.unassessed_propositions =
        propositions.len() - reader_summary.annotated_propositions;
    Ok(AuthorMentalModelResult {
        knowledge,
        reader_changes,
        reader_summary,
    })
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
        assert_eq!(result.narrator_reliability.overall_score, 0.5);
        assert_eq!(
            result.availability,
            crate::substrate::utils::AnalysisAvailability::Unavailable
        );
    }
}
