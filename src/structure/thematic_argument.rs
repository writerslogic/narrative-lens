//! Automatic thematic-argument extraction remains unavailable. Source-validated
//! author evidence maps are supported without inferred thematic truth or grades.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThematicThesis {
    pub id: usize,
    pub statement: String,
    pub confidence: f64,
    pub is_primary: bool,
    pub antithesis: Option<String>,
    pub synthesis: Option<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum EvidenceValence {
    Supporting,
    Contradicting,
    Complicating,
    Transcending,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EvidenceDelivery {
    CharacterAction { character: String, decision: String },
    PlotConsequence { cause: String },
    Dialogue { speaker: String },
    Narration,
    Symbolism { symbol: String },
    StructuralParallel { parallel_scene: usize },
    StructuralParallelById { scene_id: String },
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThematicEvidence {
    pub scene: usize,
    pub thesis_id: usize,
    pub valence: EvidenceValence,
    pub delivery: EvidenceDelivery,
    pub content: String,
    pub weight: f64,
    pub character: Option<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CharacterThematicRole {
    pub character: String,
    pub thesis_id: usize,
    pub position: EvidenceValence,
    pub articulacy: f64,
    pub fair_hearing: f64,
    /// (start_position, end_position)
    pub thematic_arc: Option<(String, String)>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ThemeInteractionType {
    Reinforcing,
    Tensioning,
    Independent,
    Subsumes,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThemeInteraction {
    pub theme_a: usize,
    pub theme_b: usize,
    pub interaction_type: ThemeInteractionType,
    pub key_scenes: Vec<usize>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThematicNote {
    pub note_type: String,
    pub description: String,
    pub suggestion: Option<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThematicArgumentResult {
    #[serde(default)]
    pub availability: crate::substrate::utils::AnalysisAvailability,
    #[serde(default)]
    pub diagnostics: Vec<String>,

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
        availability: crate::substrate::utils::AnalysisAvailability::Unavailable,
        diagnostics: vec!["thematic_argument_not_implemented".into()],
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

pub use crate::craft::annotations::AuthorAnnotationCoverage;
use crate::{
    manuscript::{Evidence, Manuscript, config::validate_id},
    structure::author_events::AuthorSource,
};
use std::collections::{BTreeMap, BTreeSet};

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AuthorThematicThesis {
    pub id: String,
    pub statement: String,
    #[serde(default)]
    pub opposes: Vec<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AuthorThematicAnnotation {
    pub id: String,
    pub thesis_id: String,
    pub source: AuthorSource,
    pub valence: EvidenceValence,
    pub delivery: EvidenceDelivery,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnnotatedThematicEvidence {
    pub id: String,
    pub thesis_id: String,
    pub source: Evidence,
    pub valence: EvidenceValence,
    pub delivery: EvidenceDelivery,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThematicThesisSummary {
    pub thesis_id: String,
    pub evidence_count: usize,
    pub valence_counts: BTreeMap<String, usize>,
    pub delivery_counts: BTreeMap<String, usize>,
    pub scene_ids: Vec<String>,
    pub character_counts: BTreeMap<String, usize>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThematicAnnotationResult {
    pub document_id: String,
    pub availability: crate::substrate::utils::AnalysisAvailability,
    pub diagnostics: Vec<String>,
    pub theses: Vec<AuthorThematicThesis>,
    pub evidence: Vec<AnnotatedThematicEvidence>,
    pub summaries: Vec<ThematicThesisSummary>,
    pub oppositions: Vec<(String, String)>,
    pub unassessed_thesis_ids: Vec<String>,
    pub scene_coverage: Vec<AuthorAnnotationCoverage>,
}

fn valence_label(value: &EvidenceValence) -> &'static str {
    match value {
        EvidenceValence::Supporting => "Supporting",
        EvidenceValence::Contradicting => "Contradicting",
        EvidenceValence::Complicating => "Complicating",
        EvidenceValence::Transcending => "Transcending",
    }
}
fn delivery_label(value: &EvidenceDelivery) -> &'static str {
    match value {
        EvidenceDelivery::CharacterAction { .. } => "CharacterAction",
        EvidenceDelivery::PlotConsequence { .. } => "PlotConsequence",
        EvidenceDelivery::Dialogue { .. } => "Dialogue",
        EvidenceDelivery::Narration => "Narration",
        EvidenceDelivery::Symbolism { .. } => "Symbolism",
        EvidenceDelivery::StructuralParallel { .. }
        | EvidenceDelivery::StructuralParallelById { .. } => "StructuralParallel",
    }
}

/// Aggregate author-labelled thematic evidence without judging thematic truth,
/// intent, earned conclusions or editorial quality. IDs remain stable on scene
/// reorder; evidence/scene lists project the current presentation order only.
/// At most 10,000 combined theses, evidence records and input opposition links
/// are accepted; duplicate links still consume the input budget.
pub fn analyze_thematic_annotations(
    manuscript: &Manuscript,
    theses: &[AuthorThematicThesis],
    annotations: &[AuthorThematicAnnotation],
) -> crate::Result<ThematicAnnotationResult> {
    use crate::craft::annotations as shared;
    let records = theses.len().saturating_add(annotations.len());
    let events = theses.iter().try_fold(records, |total, thesis| {
        total
            .checked_add(thesis.opposes.len())
            .filter(|count| *count <= crate::structure::author_events::MAX_AUTHOR_EVENTS)
            .ok_or_else(|| shared::invalid("thematic records and opposition links exceed 10000"))
    })?;
    let mut budget = shared::AnnotationBudget::new(manuscript, events)?;
    let order = shared::order(manuscript);
    let mut ids = BTreeSet::new();
    let mut thesis_map = BTreeMap::new();
    for thesis in theses {
        validate_id(&thesis.id)?;
        shared::validate_text(&thesis.statement)?;
        if !ids.insert(thesis.id.as_str()) {
            return Err(shared::invalid("annotation IDs must be unique"));
        }
        if thesis.opposes.len() > 10000 {
            return Err(shared::invalid(
                "thematic opposition link count exceeds 10000",
            ));
        }
        budget.add(&thesis.id)?;
        budget.add(&thesis.statement)?;
        let mut opposes = BTreeSet::new();
        for other in &thesis.opposes {
            validate_id(other)?;
            budget.add(other)?;
            opposes.insert(other.clone());
        }
        thesis_map.insert(
            thesis.id.clone(),
            AuthorThematicThesis {
                id: thesis.id.clone(),
                statement: thesis.statement.clone(),
                opposes: opposes.into_iter().collect(),
            },
        );
    }
    let mut oppositions = BTreeSet::new();
    for thesis in thesis_map.values() {
        for other in &thesis.opposes {
            if other == &thesis.id || !thesis_map.contains_key(other) {
                return Err(shared::invalid(
                    "thematic opposition requires a different known thesis",
                ));
            }
            let pair = if thesis.id < *other {
                (thesis.id.clone(), other.clone())
            } else {
                (other.clone(), thesis.id.clone())
            };
            oppositions.insert(pair);
        }
    }
    let mut normalized_delivery = Vec::new();
    let mut sources = Vec::new();
    for annotation in annotations {
        validate_id(&annotation.id)?;
        validate_id(&annotation.thesis_id)?;
        if !ids.insert(annotation.id.as_str()) {
            return Err(shared::invalid("annotation IDs must be unique"));
        }
        if !thesis_map.contains_key(&annotation.thesis_id) {
            return Err(shared::invalid(
                "thematic evidence references an unknown thesis",
            ));
        }
        budget.add(&annotation.id)?;
        budget.add(&annotation.thesis_id)?;
        budget.source(&annotation.source)?;
        let delivery = match &annotation.delivery {
            EvidenceDelivery::CharacterAction {
                character,
                decision,
            } => {
                budget.add(character)?;
                shared::validate_text(decision)?;
                budget.add(decision)?;
                EvidenceDelivery::CharacterAction {
                    character: shared::character(manuscript, character)?,
                    decision: decision.clone(),
                }
            }
            EvidenceDelivery::Dialogue { speaker } => {
                budget.add(speaker)?;
                EvidenceDelivery::Dialogue {
                    speaker: shared::character(manuscript, speaker)?,
                }
            }
            EvidenceDelivery::PlotConsequence { cause } => {
                shared::validate_text(cause)?;
                budget.add(cause)?;
                annotation.delivery.clone()
            }
            EvidenceDelivery::Symbolism { symbol } => {
                shared::validate_text(symbol)?;
                budget.add(symbol)?;
                annotation.delivery.clone()
            }
            EvidenceDelivery::StructuralParallelById { scene_id } => {
                validate_id(scene_id)?;
                budget.add(scene_id)?;
                if !order.contains_key(scene_id.as_str()) || scene_id == &annotation.source.scene_id
                {
                    return Err(shared::invalid(
                        "parallel evidence requires a different known scene ID",
                    ));
                }
                annotation.delivery.clone()
            }
            EvidenceDelivery::StructuralParallel { .. } => {
                return Err(shared::invalid(
                    "author thematic parallels require StructuralParallelById",
                ));
            }
            EvidenceDelivery::Narration => EvidenceDelivery::Narration,
        };
        normalized_delivery.push(delivery);
        sources.push(&annotation.source);
    }
    let validated = shared::validate(manuscript, &sources)?;
    let mut evidence: Vec<_> = annotations
        .iter()
        .zip(validated)
        .zip(normalized_delivery)
        .map(
            |((annotation, source), delivery)| AnnotatedThematicEvidence {
                id: annotation.id.clone(),
                thesis_id: annotation.thesis_id.clone(),
                source,
                valence: annotation.valence.clone(),
                delivery,
            },
        )
        .collect();
    evidence.sort_by(|a, b| {
        (
            order[a.source.scene_id.as_str()],
            a.source.range.byte_start,
            &a.id,
        )
            .cmp(&(
                order[b.source.scene_id.as_str()],
                b.source.range.byte_start,
                &b.id,
            ))
    });
    let mut summaries: BTreeMap<_, _> = thesis_map
        .keys()
        .map(|id| {
            (
                id.clone(),
                ThematicThesisSummary {
                    thesis_id: id.clone(),
                    evidence_count: 0,
                    valence_counts: BTreeMap::new(),
                    delivery_counts: BTreeMap::new(),
                    scene_ids: Vec::new(),
                    character_counts: BTreeMap::new(),
                },
            )
        })
        .collect();
    let mut seen_scenes = BTreeSet::new();
    for annotation in &evidence {
        let summary = summaries
            .get_mut(&annotation.thesis_id)
            .expect("validated thesis link");
        summary.evidence_count += 1;
        *summary
            .valence_counts
            .entry(valence_label(&annotation.valence).into())
            .or_insert(0) += 1;
        *summary
            .delivery_counts
            .entry(delivery_label(&annotation.delivery).into())
            .or_insert(0) += 1;
        if seen_scenes.insert((
            annotation.thesis_id.as_str(),
            annotation.source.scene_id.as_str(),
        )) {
            summary.scene_ids.push(annotation.source.scene_id.clone());
        }
        let actor = match &annotation.delivery {
            EvidenceDelivery::CharacterAction { character, .. } => Some(character),
            EvidenceDelivery::Dialogue { speaker } => Some(speaker),
            _ => None,
        };
        if let Some(actor) = actor {
            *summary.character_counts.entry(actor.clone()).or_insert(0) += 1;
        }
    }
    drop(seen_scenes);
    let unassessed_thesis_ids = summaries
        .values()
        .filter(|summary| summary.evidence_count == 0)
        .map(|summary| summary.thesis_id.clone())
        .collect();
    let scene_coverage = shared::coverage(
        manuscript,
        &evidence.iter().map(|item| &item.source).collect::<Vec<_>>(),
    );
    Ok(ThematicAnnotationResult {
        document_id: manuscript.document_id.clone(),
        availability: crate::substrate::utils::AnalysisAvailability::Complete,
        diagnostics: vec![
            "author_supplied_thematic_labels;automatic_truth_and_quality_unassessed".into(),
        ],
        theses: thesis_map.into_values().collect(),
        evidence,
        summaries: summaries.into_values().collect(),
        oppositions: oppositions.into_iter().collect(),
        unassessed_thesis_ids,
        scene_coverage,
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
