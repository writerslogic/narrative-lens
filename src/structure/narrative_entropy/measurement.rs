use crate::manuscript::{Manuscript, config::validate_id};
use crate::substrate::utils::{AnalysisAvailability, cosine_similarity};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

fn invalid(message: &str) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SemanticNoveltyOptions {
    #[serde(default = "default_window")]
    pub window: usize,
    #[serde(default)]
    pub include_nearest: bool,
}
fn default_window() -> usize {
    5
}
impl Default for SemanticNoveltyOptions {
    fn default() -> Self {
        Self {
            window: default_window(),
            include_nearest: false,
        }
    }
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticNoveltyPoint {
    pub scene: usize,
    pub context_start: usize,
    /// Nonzero prior vectors in the requested preceding scene window.
    pub context_count: usize,
    /// 1 - cosine(query, mean(unit prior vectors)), in [0,2].
    pub novelty: Option<f64>,
    pub nearest_prior_scene: Option<usize>,
    pub nearest_prior_similarity: Option<f64>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticNoveltyResult {
    pub availability: AnalysisAvailability,
    pub diagnostics: Vec<String>,
    pub dimensions: usize,
    pub window: usize,
    pub points: Vec<SemanticNoveltyPoint>,
    pub mean_novelty: Option<f64>,
    pub novelty_std_dev: Option<f64>,
}

fn unit_vector(vector: &[f64]) -> Option<Vec<f64>> {
    let scale = vector.iter().map(|v| v.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return None;
    }
    let norm = vector
        .iter()
        .map(|v| (v / scale).powi(2))
        .sum::<f64>()
        .sqrt();
    Some(vector.iter().map(|v| (v / scale) / norm).collect())
}

/// Measure supplied embeddings, without estimating reader beliefs or pacing quality.
/// Scene indices are positions in this call. Scene zero has no prior context and
/// stays None; zero queries and cancelling/near-zero contexts also abstain.
/// Nearest comparisons use the same preceding scene window and exclude zero vectors.
pub fn measure_semantic_novelty(
    embeddings: &[Vec<f64>],
    options: &SemanticNoveltyOptions,
) -> Result<SemanticNoveltyResult> {
    let n = embeddings.len();
    let dimensions = embeddings.first().map_or(0, Vec::len);
    if !(1..=10_000).contains(&options.window)
        || n > 10_000
        || (n > 0 && !(1..=4096).contains(&dimensions))
        || n.checked_mul(dimensions)
            .is_none_or(|cells| cells > 8 * 1024 * 1024)
    {
        return Err(invalid(
            "novelty exceeds scene, dimension, window or 64 MiB vector budget",
        ));
    }
    let work = (0..n)
        .map(|i| i.min(options.window) as u64 * dimensions as u64)
        .sum::<u64>();
    if options.include_nearest && work > 100_000_000 {
        return Err(invalid(
            "nearest novelty search exceeds 100000000 vector cells",
        ));
    }
    if embeddings
        .iter()
        .any(|vector| vector.len() != dimensions || vector.iter().any(|value| !value.is_finite()))
    {
        return Err(invalid(
            "novelty vectors must have equal dimensions and finite values",
        ));
    }
    let mut points = Vec::with_capacity(n);
    let mut prior = VecDeque::<(usize, Option<Vec<f64>>)>::new();
    let mut context = vec![0.0; dimensions];
    let mut context_count = 0usize;
    for (scene, embedding) in embeddings.iter().enumerate() {
        let query = unit_vector(embedding);
        let context_norm = context
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let novelty = query
            .as_ref()
            .filter(|_| context_count > 0 && context_norm > 1e-12)
            .map(|query| (1.0 - cosine_similarity(query, &context)).clamp(0.0, 2.0));
        let mut nearest = None::<(usize, f64)>;
        if options.include_nearest
            && let Some(query) = &query
        {
            for (index, vector) in &prior {
                if let Some(vector) = vector {
                    let similarity = cosine_similarity(query, vector).clamp(-1.0, 1.0);
                    if nearest.is_none_or(|(_, previous)| similarity > previous) {
                        nearest = Some((*index, similarity));
                    }
                }
            }
        }
        points.push(SemanticNoveltyPoint {
            scene,
            context_start: scene.saturating_sub(options.window),
            context_count,
            novelty,
            nearest_prior_scene: nearest.map(|(index, _)| index),
            nearest_prior_similarity: nearest.map(|(_, similarity)| similarity),
        });
        if prior.len() == options.window
            && let Some((_, Some(expired))) = prior.pop_front()
        {
            for (sum, value) in context.iter_mut().zip(expired) {
                *sum -= value;
            }
            context_count -= 1;
        }
        if let Some(query) = &query {
            for (sum, value) in context.iter_mut().zip(query) {
                *sum += value;
            }
            context_count += 1;
        }
        prior.push_back((scene, query));
    }
    let measured: Vec<_> = points.iter().filter_map(|point| point.novelty).collect();
    let mean_novelty =
        (!measured.is_empty()).then(|| measured.iter().sum::<f64>() / measured.len() as f64);
    let novelty_std_dev = mean_novelty.map(|mean| {
        (measured
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / measured.len() as f64)
            .sqrt()
    });
    Ok(SemanticNoveltyResult {
        availability: if measured.is_empty() {
            AnalysisAvailability::Unavailable
        } else if measured.len() == n - 1 {
            AnalysisAvailability::Complete
        } else {
            AnalysisAvailability::Partial
        },
        diagnostics: vec![
            "supplied_embeddings;cosine_novelty_not_Shannon_entropy_or_editorial_quality".into(),
            "unit_vectors;cosine_distance_range_0_to_2;context_norm_at_most_1e-12_abstains".into(),
        ],
        dimensions,
        window: options.window,
        points,
        mean_novelty,
        novelty_std_dev,
    })
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HypothesisProbability {
    pub id: String,
    pub probability: f64,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SuppliedUncertainty {
    pub question_id: String,
    pub scene_id: String,
    pub outcomes: Vec<HypothesisProbability>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UncertaintyMeasurement {
    pub question_id: String,
    pub scene_id: String,
    /// Original probabilities, ordered by stable outcome ID; never inferred.
    pub outcomes: Vec<HypothesisProbability>,
    pub probability_sum: f64,
    pub entropy_bits: f64,
    /// H / log2(declared outcome count), including declared zero-probability outcomes.
    pub normalized_entropy: f64,
    pub effective_outcomes: f64,
    pub outcome_count: usize,
    pub positive_outcome_count: usize,
    /// Current H minus the previous supplied H for this question in presentation order.
    pub entropy_change_bits: Option<f64>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UncertaintyResult {
    pub document_id: String,
    pub availability: AnalysisAvailability,
    pub diagnostics: Vec<String>,
    pub measurements: Vec<UncertaintyMeasurement>,
}

/// Shannon uncertainty of caller-supplied discrete distributions, not estimated
/// narrative truth or reader beliefs. Sources and outcomes retain stable IDs;
/// ordering/deltas project manuscript presentation order, not chronology.
/// Accepted probability-sum roundoff (1e-9 plus floating-point summation error)
/// is normalized for math; original probabilities and their sum remain in output.
pub fn measure_narrative_uncertainty(
    manuscript: &Manuscript,
    observations: &[SuppliedUncertainty],
) -> Result<UncertaintyResult> {
    if observations.len() > 10_000 {
        return Err(invalid("uncertainty exceeds 10000 observations"));
    }
    let _cells = observations
        .iter()
        .try_fold(0usize, |count, observation| {
            count.checked_add(observation.outcomes.len())
        })
        .filter(|cells| *cells <= 100_000)
        .ok_or_else(|| invalid("uncertainty exceeds 100000 outcome cells"))?;
    manuscript.validate()?;
    let order: BTreeMap<_, _> = manuscript
        .scenes
        .iter()
        .enumerate()
        .map(|(i, scene)| (scene.id.as_str(), i))
        .collect();
    let mut seen = BTreeSet::new();
    let mut identities = BTreeMap::<&str, BTreeSet<&str>>::new();
    let mut bytes = 0usize;
    for observation in observations {
        validate_id(&observation.question_id)?;
        validate_id(&observation.scene_id)?;
        if !order.contains_key(observation.scene_id.as_str())
            || !seen.insert((&observation.question_id, &observation.scene_id))
        {
            return Err(invalid(
                "uncertainty requires known unique question/scene observations",
            ));
        }
        if !(1..=256).contains(&observation.outcomes.len()) {
            return Err(invalid("uncertainty requires 1..=256 declared outcomes"));
        }
        let mut ids = BTreeSet::new();
        bytes = bytes
            .checked_add(observation.question_id.len())
            .and_then(|bytes| bytes.checked_add(observation.scene_id.len()))
            .ok_or_else(|| invalid("uncertainty ID budget exceeded"))?;
        for outcome in &observation.outcomes {
            validate_id(&outcome.id)?;
            if !ids.insert(outcome.id.as_str())
                || !outcome.probability.is_finite()
                || !(0.0..=1.0).contains(&outcome.probability)
            {
                return Err(invalid(
                    "uncertainty outcome IDs must be unique and probabilities finite within 0..=1",
                ));
            }
            bytes = bytes
                .checked_add(outcome.id.len())
                .ok_or_else(|| invalid("uncertainty ID budget exceeded"))?;
        }
        if bytes > 4 * 1024 * 1024 {
            return Err(invalid("uncertainty exceeds 4 MiB supplied IDs"));
        }
        if identities
            .get(observation.question_id.as_str())
            .is_some_and(|previous| previous != &ids)
        {
            return Err(invalid(
                "a question must retain the same declared outcome identities",
            ));
        }
        identities.entry(&observation.question_id).or_insert(ids);
        let mut supplied: Vec<_> = observation.outcomes.iter().collect();
        supplied.sort_by(|a, b| a.id.cmp(&b.id));
        let sum = supplied
            .iter()
            .map(|outcome| outcome.probability)
            .sum::<f64>();
        if sum <= 0.0
            || (sum - 1.0).abs() > 1e-9 + f64::EPSILON * observation.outcomes.len() as f64 * 2.0
        {
            return Err(invalid(
                "uncertainty probabilities must sum to one within 1e-9 roundoff",
            ));
        }
    }
    let mut ordered: Vec<_> = observations.iter().collect();
    ordered.sort_by(|a, b| {
        (order[a.scene_id.as_str()], &a.question_id)
            .cmp(&(order[b.scene_id.as_str()], &b.question_id))
    });
    let mut previous = BTreeMap::<&str, f64>::new();
    let mut measurements = Vec::with_capacity(observations.len());
    for observation in ordered {
        let mut outcomes = observation.outcomes.clone();
        outcomes.sort_by(|a, b| a.id.cmp(&b.id));
        let probability_sum = outcomes
            .iter()
            .map(|outcome| outcome.probability)
            .sum::<f64>();
        let max_entropy = (outcomes.len() as f64).log2();
        let entropy_bits = outcomes
            .iter()
            .filter(|outcome| outcome.probability > 0.0)
            .map(|outcome| {
                let probability = outcome.probability / probability_sum;
                -probability * probability.log2()
            })
            .sum::<f64>()
            .clamp(0.0, max_entropy);
        measurements.push(UncertaintyMeasurement {
            question_id: observation.question_id.clone(),
            scene_id: observation.scene_id.clone(),
            probability_sum,
            entropy_bits,
            normalized_entropy: if max_entropy == 0.0 {
                0.0
            } else {
                (entropy_bits / max_entropy).clamp(0.0, 1.0)
            },
            effective_outcomes: entropy_bits.exp2(),
            outcome_count: outcomes.len(),
            positive_outcome_count: outcomes
                .iter()
                .filter(|outcome| outcome.probability > 0.0)
                .count(),
            entropy_change_bits: previous
                .insert(&observation.question_id, entropy_bits)
                .map(|prior| entropy_bits - prior),
            outcomes,
        });
    }
    Ok(UncertaintyResult { document_id: manuscript.document_id.clone(),
        availability: if observations.is_empty() { AnalysisAvailability::Unavailable } else { AnalysisAvailability::Complete },
        diagnostics: vec!["caller_supplied_probabilities;not_estimated_reader_beliefs_or_thematic_truth".into(),
            "Shannon_bits;zero_cells_contribute_zero;normalized_by_declared_outcome_count;presentation_order_deltas".into()],
        measurements,
    })
}
