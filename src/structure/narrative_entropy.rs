//! Explicitly separate mathematical uncertainty of supplied hypotheses from
//! cosine novelty of supplied/model embeddings. Neither establishes pacing quality.
mod measurement;
pub use measurement::*;
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NarrativeThread {
    MainPlot,
    Subplot(String),
    CharacterArc(String),
    Relationship(String, String),
    Mystery(String),
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PossibleOutcome {
    pub description: String,
    pub probability: f64,
    pub realized: bool,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NarrativeQuestion {
    pub id: usize,
    pub question: String,
    pub is_cdq: bool,
    pub thread: NarrativeThread,
    pub introduced_at: usize,
    pub resolved_at: Option<usize>,
    pub outcomes: Vec<PossibleOutcome>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntropyPoint {
    pub scene: usize,
    pub per_question_entropy: Vec<(usize, f64)>,
    pub composite_entropy: f64,
    pub entropy_rate: f64,
    pub entropy_acceleration: f64,
    pub information_gain: f64,
    pub conditional_entropies: Vec<(usize, usize, f64)>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EntropyDiagnosis {
    Unavailable,
    WellPaced,
    Predictable {
        plateau_scene: usize,
    },
    Chaotic {
        spike_scenes: Vec<usize>,
    },
    Rushed {
        drop_scene: usize,
        drop_magnitude: f64,
    },
    Unresolved {
        final_entropy: f64,
        unresolved_questions: Vec<String>,
    },
    Mixed(Vec<Box<EntropyDiagnosis>>),
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PacingNote {
    pub scene_range: (usize, usize),
    pub issue: String,
    pub reader_experience: String,
    pub fix: String,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NarrativeEntropyResult {
    #[serde(default)]
    pub availability: crate::substrate::utils::AnalysisAvailability,
    #[serde(default)]
    pub diagnostics: Vec<String>,

    #[serde(default)]
    pub uncertainty_assessed: bool,
    #[serde(default)]
    pub semantic_novelty: Option<SemanticNoveltyResult>,
    pub questions: Vec<NarrativeQuestion>,
    /// Neutral placeholders retained per scene for legacy shape compatibility.
    /// No Shannon uncertainty is inferred from prose.
    pub entropy_curve: Vec<EntropyPoint>,
    pub cdq_entropy_curve: Vec<f64>,
    pub entropy_rate_curve: Vec<f64>,
    pub redundant_scenes: Vec<(usize, String)>,
    pub subplot_integration: f64,
    pub pacing_score: f64,
    pub pacing_diagnosis: EntropyDiagnosis,
    pub complexity_curve: Vec<f64>,
    pub pacing_feedback: Vec<PacingNote>,
    pub overall_pacing_assessment: String,
}

/// Compute the weight for a narrative question based on its thread type.
pub fn question_weight(q: &NarrativeQuestion) -> f64 {
    if q.is_cdq {
        2.0
    } else {
        match &q.thread {
            NarrativeThread::MainPlot => 2.0,
            NarrativeThread::Subplot(_) => 1.0,
            NarrativeThread::CharacterArc(_) => 0.5,
            NarrativeThread::Relationship(_, _) => 0.5,
            NarrativeThread::Mystery(_) => 1.0,
        }
    }
}

fn unavailable_result(scene_count: usize, diagnostic: &str) -> NarrativeEntropyResult {
    NarrativeEntropyResult {
        availability: crate::substrate::utils::AnalysisAvailability::Unavailable,
        diagnostics: vec![diagnostic.into(), "legacy_entropy_fields_are_unassessed_compatibility_placeholders".into()],
        uncertainty_assessed: false, semantic_novelty: None, questions: Vec::new(),
        entropy_curve: (0..scene_count).map(|scene| EntropyPoint { scene, per_question_entropy: Vec::new(),
            composite_entropy: 0.5, entropy_rate: 0.0, entropy_acceleration: 0.0, information_gain: 0.0,
            conditional_entropies: Vec::new() }).collect(),
        cdq_entropy_curve: Vec::new(), entropy_rate_curve: vec![0.0; scene_count], redundant_scenes: Vec::new(),
        subplot_integration: 0.5, pacing_score: 0.5, pacing_diagnosis: EntropyDiagnosis::Unavailable,
        complexity_curve: Vec::new(), pacing_feedback: Vec::new(),
        overall_pacing_assessment: "Editorial pacing and narrative uncertainty are unassessed; use explicit supplied distributions or measured semantic novelty.".into(),
    }
}

/// Legacy prose entry: obtain checked model embeddings and expose cosine novelty
/// by name. Theme/genre/arc labels cannot create questions, outcomes or probabilities.
/// Mathematical uncertainty requires `measure_narrative_uncertainty` inputs.
pub fn compute_narrative_entropy_checked(
    scenes: &[&str],
    _themes: &[(String, f64)],
    _structure_template: &str,
    _genre: &str,
    _character_arcs: &[(String, String)],
) -> crate::Result<NarrativeEntropyResult> {
    crate::craft::measurements::validate_texts(scenes.iter().copied()).map_err(|_| {
        crate::Error::InvalidInput(
            "narrative scenes exceed 10000 scenes, 16 MiB per scene or 64 MiB total".into(),
        )
    })?;
    let mut result = unavailable_result(scenes.len(), "no_scenes");
    if scenes.is_empty() {
        return Ok(result);
    }
    let generation = crate::onnx::model_generation();
    let mut embeddings = Vec::new();
    let mut cells = 0usize;
    for chunk in scenes.chunks(64) {
        match crate::onnx::checked::embeddings(crate::onnx::backend(), chunk) {
            Ok(batch) => {
                cells = cells
                    .checked_add(batch.iter().map(Vec::len).sum::<usize>())
                    .filter(|count| *count <= 8 * 1024 * 1024)
                    .ok_or_else(|| {
                        crate::Error::InvalidInput(
                            "novelty model output exceeds 64 MiB vector budget".into(),
                        )
                    })?;
                embeddings.extend(batch);
            }
            Err(failure) => {
                result.diagnostics[0] =
                    format!("semantic_model_unavailable_or_failed:{}", failure.code);
                result.overall_pacing_assessment = "Semantic novelty unavailable; editorial pacing and narrative uncertainty are unassessed.".into();
                return Ok(result);
            }
        }
    }
    if crate::onnx::model_generation() != generation {
        result.diagnostics[0] = "semantic_model_changed_during_analysis".into();
        return Ok(result);
    }
    let novelty = measure_semantic_novelty(&embeddings, &SemanticNoveltyOptions::default())?;
    if novelty.availability != crate::substrate::utils::AnalysisAvailability::Unavailable {
        result.availability = crate::substrate::utils::AnalysisAvailability::Partial;
    }
    result.diagnostics[0] =
        "cosine_novelty_only;editorial_pacing_and_reader_uncertainty_unassessed".into();
    result.semantic_novelty = Some(novelty);
    Ok(result)
}

pub fn compute_narrative_entropy(
    scenes: &[&str],
    themes: &[(String, f64)],
    structure_template: &str,
    genre: &str,
    character_arcs: &[(String, String)],
) -> NarrativeEntropyResult {
    compute_narrative_entropy_checked(scenes, themes, structure_template, genre, character_arcs)
        .unwrap_or_else(|_| unavailable_result(0, "invalid_narrative_input;use_checked_api"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unavailable_prose_does_not_invent_reader_questions_or_probabilities() {
        let result = compute_narrative_entropy(
            &["They met at a cafe."],
            &[("love".into(), 0.9)],
            "three_act",
            "romance",
            &[("Alice".into(), "growth".into())],
        );
        assert!(result.questions.is_empty());
        assert!(!result.uncertainty_assessed);
        assert!(matches!(
            result.pacing_diagnosis,
            EntropyDiagnosis::Unavailable
        ));
        assert!(
            result
                .overall_pacing_assessment
                .to_lowercase()
                .contains("unavailable")
        );
        assert!(
            result
                .entropy_curve
                .iter()
                .all(|point| point.information_gain == 0.0)
        );
    }
    #[test]
    fn empty_prose_keeps_empty_compatibility_shape() {
        let result = compute_narrative_entropy(&[], &[], "", "", &[]);
        assert!(result.entropy_curve.is_empty());
        assert!(result.semantic_novelty.is_none());
        assert_eq!(result.pacing_score, 0.5);
    }
}
