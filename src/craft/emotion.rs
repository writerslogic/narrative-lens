//! Ekman-7 emotion model configuration over [`crate::onnx`] (the same runtime
//! that serves SBERT and NER).

use crate::onnx;

/// Ekman-7 labels, POSITIONALLY ALIGNED with the model's output logits.
///
/// This order must match the exported model's `config.json` `id2label` (e.g.
/// `j-hartmann/emotion-english-distilroberta-base`). If you swap the model,
/// update this vector to the new model's label order.
pub const EKMAN_LABELS: &[&str] = &[
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise",
];

/// Set the explicit model root containing `emotion-onnx/`.
/// Replaces or retries the native model; a no-op on wasm.
pub fn set_model_directory(path: &str) {
    onnx::set_emotion_model_directory(path);
}
