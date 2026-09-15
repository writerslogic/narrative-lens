//! SBERT sentence embeddings.
//!
//! The model runtime lives behind [`crate::onnx::InferenceBackend`] (ONNX
//! Runtime natively, `tract` on wasm); these are thin, signature-stable wrappers
//! so callers are unaffected by the backend split. When the model is unavailable
//! (or the crate's `onnx` feature is disabled) every function degrades to an
//! empty embedding without panicking.

use crate::onnx;

/// Set a custom directory to search for SBERT model files.
/// Must be called before the first call to `encode_text`.
pub fn set_model_directory(path: &str) {
    onnx::set_sbert_model_directory(path);
}

/// Encode text into a sentence embedding. Returns an empty Vec when no model is
/// available (missing files, or the `onnx` feature is disabled).
pub fn encode_text(text: &str) -> Vec<f64> {
    onnx::backend()
        .embed(&[text])
        .into_iter()
        .next()
        .unwrap_or_default()
}

/// Encode multiple texts in one batched run. Returns one embedding per input
/// text. Entries for which encoding fails are returned as empty Vec.
pub fn encode_batch(texts: &[&str]) -> Vec<Vec<f64>> {
    onnx::backend().embed(texts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::utils::cosine_similarity;

    #[test]
    fn test_encode_text_no_model() {
        let result = encode_text("Hello world");
        assert!(result.is_empty());
    }

    #[test]
    fn test_encode_batch_no_model() {
        let result = encode_batch(&["Hello", "World"]);
        assert_eq!(result.len(), 2);
        assert!(result[0].is_empty());
        assert!(result[1].is_empty());
    }

    #[test]
    fn test_cosine_similarity_no_model() {
        let a = encode_text("The cat sat on the mat.");
        let b = encode_text("A dog lay on the rug.");
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }
}
