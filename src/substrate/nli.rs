//! NLI (natural-language inference) via a sequence-PAIR classification ONNX
//! model.
//!
//! Given a `(premise, hypothesis)` pair, judges entailment / neutral /
//! contradiction -- the primitive a continuity checker gates on to flag
//! statements that contradict each other. Thin, signature-stable wrappers over
//! [`crate::onnx`] (the same runtime that serves SBERT, NER, and the emotion
//! classifier). When the model is unavailable (or the crate's `onnx` feature is
//! disabled) every function degrades to empty/`None`, never panics.

use crate::onnx;

/// NLI labels, POSITIONALLY ALIGNED with the model's output logits. Matches the
/// common `{0: contradiction, 1: neutral, 2: entailment}` id2label order used by
/// `*-mnli` checkpoints. If you swap the model, update this to its config.json
/// order.
pub const NLI_LABELS: &[&str] = &["contradiction", "neutral", "entailment"];

/// Index of the contradiction class in [`NLI_LABELS`] (and the model's logits).
const CONTRADICTION_IDX: usize = 0;

/// Set the directory searched for the NLI model (`nli-onnx/`). No-op on wasm.
pub fn set_model_directory(path: &str) {
    onnx::set_nli_model_directory(path);
}

/// Full 3-class distribution for one `(premise, hypothesis)` pair as `(label, prob)`.
/// Empty when the model is unavailable.
pub fn classify_pair(premise: &str, hypothesis: &str) -> Vec<(String, f64)> {
    classify_pairs(&[(premise, hypothesis)])
        .into_iter()
        .next()
        .unwrap_or_default()
}

/// Classify a batch of pairs: one `(label, prob)` list per pair over [`NLI_LABELS`].
/// A pair is empty when the model is unavailable, and -- defensively -- when the
/// class count does not match [`NLI_LABELS`] (so a mismatched model degrades,
/// never mislabels).
pub fn classify_pairs(pairs: &[(&str, &str)]) -> Vec<Vec<(String, f64)>> {
    onnx::backend()
        .classify_pairs(pairs)
        .into_iter()
        .map(|probs| {
            if probs.len() != NLI_LABELS.len() {
                if !probs.is_empty() {
                    log::warn!(
                        "[nli] model returned {} classes, expected {}; degrading",
                        probs.len(),
                        NLI_LABELS.len()
                    );
                }
                return Vec::new();
            }
            NLI_LABELS
                .iter()
                .zip(probs)
                .map(|(&label, p)| (label.to_string(), p))
                .collect()
        })
        .collect()
}

/// The contradiction probability for a pair, but only when contradiction is the
/// argmax class -- else `None`. `None` also when the model is unavailable or the
/// class count is unexpected (graceful degrade: the caller then finds no
/// contradiction).
pub fn contradiction_score(premise: &str, hypothesis: &str) -> Option<f64> {
    contradiction_scores(&[(premise, hypothesis)])
        .into_iter()
        .next()
        .flatten()
}

/// Batched [`contradiction_score`]: one model call over every pair instead of
/// one call per pair. Same per-pair semantics (argmax-gated, `None` on an
/// unavailable model or an unexpected class count), just run once over the
/// whole batch.
pub fn contradiction_scores(pairs: &[(&str, &str)]) -> Vec<Option<f64>> {
    onnx::backend()
        .classify_pairs(pairs)
        .into_iter()
        .map(|probs| {
            if probs.len() != NLI_LABELS.len() {
                return None;
            }
            let argmax = probs
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?
                .0;
            (argmax == CONTRADICTION_IDX).then_some(probs[CONTRADICTION_IDX])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_model_degrades_to_empty() {
        assert!(classify_pair("The bridge will hold.", "The bridge will fall.").is_empty());
        let batch = classify_pairs(&[("a", "b"), ("c", "d")]);
        assert_eq!(batch.len(), 2);
        assert!(batch.iter().all(|d| d.is_empty()));
        assert_eq!(contradiction_score("a", "b"), None);
        assert_eq!(
            contradiction_scores(&[("a", "b"), ("c", "d")]),
            vec![None, None]
        );
    }
}
