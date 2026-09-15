//! Ekman-7 emotion classification via a sequence-classification ONNX model.
//!
//! Thin, signature-stable wrappers over [`crate::onnx`] (the same runtime that
//! serves SBERT and NER). When the model is unavailable (or the crate's `onnx`
//! feature is disabled) every function degrades to empty output, so a caller
//! falls back to a keyword classifier rather than fabricating a distribution.

use crate::onnx;

/// Ekman-7 labels, POSITIONALLY ALIGNED with the model's output logits.
///
/// This order must match the exported model's `config.json` `id2label` (e.g.
/// `j-hartmann/emotion-english-distilroberta-base`). If you swap the model,
/// update this vector to the new model's label order.
pub const EKMAN_LABELS: &[&str] = &[
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise",
];

/// Set the directory searched for the emotion model (`emotion-onnx/`).
/// Must be called before the first classification. No-op on wasm (no filesystem).
pub fn set_model_directory(path: &str) {
    onnx::set_emotion_model_directory(path);
}

/// Classify one text into `(label, probability)` pairs over [`EKMAN_LABELS`].
/// Empty `Vec` when the model is unavailable.
pub fn classify_text(text: &str) -> Vec<(String, f64)> {
    classify_batch(&[text])
        .into_iter()
        .next()
        .unwrap_or_default()
}

/// Classify a batch of texts: one `(label, probability)` list per input, in
/// order. A text is empty when the model is unavailable, and -- defensively --
/// when the model's class count does not match [`EKMAN_LABELS`] (so a mismatched
/// model degrades rather than emitting mislabeled scores).
pub fn classify_batch(texts: &[&str]) -> Vec<Vec<(String, f64)>> {
    onnx::backend()
        .classify(texts)
        .into_iter()
        .map(|probs| {
            if probs.len() != EKMAN_LABELS.len() {
                if !probs.is_empty() {
                    log::warn!(
                        "[emotion] model returned {} classes, expected {}; degrading",
                        probs.len(),
                        EKMAN_LABELS.len()
                    );
                }
                return Vec::new();
            }
            EKMAN_LABELS
                .iter()
                .zip(probs)
                .map(|(&label, p)| (label.to_string(), p))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_model_degrades_to_empty() {
        assert!(classify_text("She was overjoyed.").is_empty());
        let batch = classify_batch(&["a", "b", "c"]);
        assert_eq!(batch.len(), 3);
        assert!(batch.iter().all(|d| d.is_empty()));
    }
}
