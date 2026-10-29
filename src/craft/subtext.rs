//! Automatic subtext detection remains unavailable. Source-validated author
//! interpretation maps expose declared meanings and resolution links.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
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

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
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

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneSubtext {
    pub scene: usize,
    pub average_density: f64,
    pub instance_count: usize,
    pub dominant_type: Option<SubtextType>,
    pub appropriateness: f64,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubtextAdvice {
    pub scene: usize,
    pub issue_type: String,
    pub quoted_text: String,
    pub explanation: String,
    pub suggestion: Option<String>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubtextResult {
    #[serde(default)]
    pub availability: crate::substrate::utils::AnalysisAvailability,
    #[serde(default)]
    pub diagnostics: Vec<String>,

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
            appropriateness: 0.5,
        })
        .collect();
    SubtextResult {
        availability: crate::substrate::utils::AnalysisAvailability::Unavailable,
        diagnostics: vec!["subtext_analysis_not_implemented".into()],
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

pub use crate::craft::annotations::AuthorAnnotationCoverage;
use crate::{
    manuscript::{Evidence, Manuscript, config::validate_id},
    structure::author_events::AuthorSource,
};
use std::collections::{BTreeMap, BTreeSet};

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AuthorSubtextAnnotation {
    pub id: String,
    pub surface: AuthorSource,
    pub implied: String,
    pub subtext_type: SubtextType,
    #[serde(default)]
    pub characters: Vec<String>,
    #[serde(default)]
    pub resolution: Option<AuthorSource>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnnotatedSubtext {
    pub id: String,
    pub surface: Evidence,
    pub implied: String,
    pub subtext_type: SubtextType,
    pub characters: Vec<String>,
    pub resolution: Option<Evidence>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubtextAnnotationResult {
    pub document_id: String,
    pub availability: crate::substrate::utils::AnalysisAvailability,
    pub diagnostics: Vec<String>,
    pub instances: Vec<AnnotatedSubtext>,
    /// Union of surface anchors only; resolution evidence is excluded.
    pub scene_coverage: Vec<AuthorAnnotationCoverage>,
    pub unresolved_ids: Vec<String>,
    pub type_counts: BTreeMap<String, usize>,
    pub character_counts: BTreeMap<String, usize>,
    pub resolved_count: usize,
}

/// Project author-declared interpretations onto exact original sources. No
/// implied meaning, author intent, density or editorial quality is inferred.
/// Resolution links may precede surfaces; ordering is presentation only.
pub fn analyze_subtext_annotations(
    manuscript: &Manuscript,
    annotations: &[AuthorSubtextAnnotation],
) -> crate::Result<SubtextAnnotationResult> {
    use crate::craft::annotations as shared;
    let mut budget = shared::AnnotationBudget::new(manuscript, annotations.len())?;
    let mut ids = BTreeSet::new();
    let mut sources = Vec::new();
    let mut source_indices = Vec::new();
    let mut actors = Vec::new();
    for annotation in annotations {
        validate_id(&annotation.id)?;
        if !ids.insert(&annotation.id) {
            return Err(shared::invalid("annotation IDs must be unique"));
        }
        shared::validate_text(&annotation.implied)?;
        if annotation.characters.len() > 256 {
            return Err(shared::invalid("annotation character count exceeds 256"));
        }
        budget.add(&annotation.id)?;
        budget.add(&annotation.implied)?;
        let mut characters = BTreeSet::new();
        for name in &annotation.characters {
            budget.add(name)?;
            characters.insert(shared::character(manuscript, name)?);
        }
        actors.push(characters.into_iter().collect::<Vec<_>>());
        budget.source(&annotation.surface)?;
        let surface = sources.len();
        sources.push(&annotation.surface);
        let resolution = if let Some(source) = &annotation.resolution {
            budget.source(source)?;
            let index = sources.len();
            sources.push(source);
            Some(index)
        } else {
            None
        };
        source_indices.push((surface, resolution));
    }
    let evidence = shared::validate(manuscript, &sources)?;
    let order = shared::order(manuscript);
    let mut instances = Vec::new();
    let mut unresolved_ids = Vec::new();
    let mut type_counts = BTreeMap::new();
    let mut character_counts = BTreeMap::new();
    let mut resolved_count = 0;
    for ((annotation, (surface, resolution)), characters) in
        annotations.iter().zip(source_indices).zip(actors)
    {
        if resolution.is_some() {
            resolved_count += 1;
        } else {
            unresolved_ids.push(annotation.id.clone());
        }
        let label = match annotation.subtext_type {
            SubtextType::EmotionalDenial => "EmotionalDenial",
            SubtextType::PowerPlay => "PowerPlay",
            SubtextType::SelfDeception => "SelfDeception",
            SubtextType::CodedCommunication => "CodedCommunication",
            SubtextType::DramaticIronySubtext => "DramaticIronySubtext",
            SubtextType::Deflection => "Deflection",
            SubtextType::VerbalIrony => "VerbalIrony",
            SubtextType::ActionContradiction => "ActionContradiction",
        }
        .to_string();
        *type_counts.entry(label).or_insert(0) += 1;
        for name in &characters {
            *character_counts.entry(name.clone()).or_insert(0) += 1;
        }
        instances.push(AnnotatedSubtext {
            id: annotation.id.clone(),
            surface: evidence[surface].clone(),
            implied: annotation.implied.clone(),
            subtext_type: annotation.subtext_type.clone(),
            characters,
            resolution: resolution.map(|index| evidence[index].clone()),
        });
    }
    instances.sort_by(|a, b| {
        (
            order[a.surface.scene_id.as_str()],
            a.surface.range.byte_start,
            &a.id,
        )
            .cmp(&(
                order[b.surface.scene_id.as_str()],
                b.surface.range.byte_start,
                &b.id,
            ))
    });
    unresolved_ids.sort();
    let scene_coverage = shared::coverage(
        manuscript,
        &instances
            .iter()
            .map(|instance| &instance.surface)
            .collect::<Vec<_>>(),
    );
    Ok(SubtextAnnotationResult {
        document_id: manuscript.document_id.clone(),
        availability: crate::substrate::utils::AnalysisAvailability::Complete,
        diagnostics: vec![
            "author_supplied_subtext;automatic_interpretation_and_quality_unassessed".into(),
        ],
        instances,
        scene_coverage,
        unresolved_ids,
        type_counts,
        character_counts,
        resolved_count,
    })
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
