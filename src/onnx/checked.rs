//! Additive, diagnostic model calls. Legacy empty-output APIs stay compatible.
use super::InferenceBackend;
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Embedding,
    Ner,
    Emotion,
    Nli,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    Unavailable,
    Incompatible,
    Execution,
}
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelFailure {
    pub kind: FailureKind,
    pub code: String,
}
impl std::fmt::Display for ModelFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.code)
    }
}
impl std::error::Error for ModelFailure {}
impl ModelFailure {
    pub fn unavailable(code: &str) -> Self {
        Self {
            kind: FailureKind::Unavailable,
            code: code.into(),
        }
    }
    pub fn incompatible(code: &str) -> Self {
        Self {
            kind: FailureKind::Incompatible,
            code: code.into(),
        }
    }
    pub fn execution(code: &str) -> Self {
        Self {
            kind: FailureKind::Execution,
            code: code.into(),
        }
    }
}

pub fn nli(
    backend: &dyn InferenceBackend,
    pairs: &[(&str, &str)],
) -> Result<Vec<Vec<f64>>, ModelFailure> {
    if pairs.is_empty() {
        return Ok(Vec::new());
    }
    validate_texts(
        &pairs.iter().flat_map(|(a, b)| [*a, *b]).collect::<Vec<_>>(),
        512,
    )?;
    if pairs.len() > 256 {
        return Err(ModelFailure::execution("model_batch_limit"));
    }
    backend.check_model(ModelKind::Nli)?;
    let output = backend.classify_pairs_checked(pairs)?;
    if output.len() != pairs.len() {
        return Err(ModelFailure::execution("nli_batch_cardinality"));
    }
    for row in &output {
        if row.is_empty() {
            return Err(ModelFailure::execution("nli_empty_inference"));
        }
        if row.len() != 3 {
            return Err(ModelFailure::incompatible("nli_expected_three_classes"));
        }
        if row
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            || (row.iter().sum::<f64>() - 1.0).abs() > 1e-4
        {
            return Err(ModelFailure::incompatible("nli_invalid_distribution"));
        }
    }
    Ok(output)
}

pub fn embeddings(
    backend: &dyn InferenceBackend,
    texts: &[&str],
) -> Result<Vec<Vec<f64>>, ModelFailure> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    validate_texts(texts, 256)?;
    backend.check_model(ModelKind::Embedding)?;
    let output = backend.embed_checked(texts)?;
    if output.len() != texts.len() || output.iter().any(Vec::is_empty) {
        return Err(ModelFailure::execution(
            "embedding_empty_or_incomplete_inference",
        ));
    }
    let dim = output[0].len();
    if output
        .iter()
        .any(|r| r.len() != dim || r.iter().any(|x| !x.is_finite()))
    {
        return Err(ModelFailure::incompatible(
            "embedding_invalid_shape_or_values",
        ));
    }
    Ok(output)
}

pub fn emotion(
    backend: &dyn InferenceBackend,
    texts: &[&str],
) -> Result<Vec<Vec<f64>>, ModelFailure> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    validate_texts(texts, 256)?;
    backend.check_model(ModelKind::Emotion)?;
    let output = backend.classify_checked(texts)?;
    if output.len() != texts.len() {
        return Err(ModelFailure::execution("emotion_batch_cardinality"));
    }
    for row in &output {
        if row.len() != 7 {
            return Err(ModelFailure::incompatible("emotion_expected_seven_classes"));
        }
        if row
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || (row.iter().sum::<f64>() - 1.0).abs() > 1e-4
        {
            return Err(ModelFailure::incompatible("emotion_invalid_distribution"));
        }
    }
    Ok(output)
}
pub fn entities(
    backend: &dyn InferenceBackend,
    texts: &[&str],
) -> Result<Vec<super::NerSpans>, ModelFailure> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    validate_texts(texts, 256)?;
    backend.check_model(ModelKind::Ner)?;
    let output = backend.tag_batch_checked(texts)?;
    if output.len() != texts.len() {
        return Err(ModelFailure::execution("ner_batch_cardinality"));
    }
    for (text, row) in texts.iter().zip(&output) {
        for (quote, label, start, end) in row {
            if start >= end
                || text.get(*start..*end) != Some(quote.as_str())
                || !matches!(label.as_str(), "MISC" | "PER" | "ORG" | "LOC")
            {
                return Err(ModelFailure::incompatible("ner_invalid_span_or_label"));
            }
        }
    }
    Ok(output)
}
pub(super) fn validate_texts(texts: &[&str], max_batch: usize) -> Result<(), ModelFailure> {
    let mut bytes = 0usize;
    if texts.len() > max_batch {
        return Err(ModelFailure::execution("model_batch_limit"));
    }
    for text in texts {
        if text.len() > 64 * 1024 {
            return Err(ModelFailure::execution("model_text_limit"));
        }
        bytes = bytes
            .checked_add(text.len())
            .ok_or_else(|| ModelFailure::execution("model_text_limit"))?;
    }
    if bytes > 4 * 1024 * 1024 {
        return Err(ModelFailure::execution("model_batch_text_limit"));
    }
    Ok(())
}
#[cfg(any(feature = "onnx", test))]
pub(super) fn input_elements(batch: usize, tokens: usize) -> Result<usize, ModelFailure> {
    if batch > 256 || tokens == 0 || tokens > 4096 {
        return Err(ModelFailure::execution("model_token_or_batch_limit"));
    }
    batch
        .checked_mul(tokens)
        .filter(|v| *v <= 1024 * 1024)
        .ok_or_else(|| ModelFailure::execution("model_input_allocation_limit"))
}
#[cfg(any(feature = "onnx", test))]
pub(super) fn tensor_shape(
    shape: &[i64],
    expected: &[Option<usize>],
    data: &[f32],
) -> Result<(), ModelFailure> {
    if shape.len() != expected.len() {
        return Err(ModelFailure::incompatible("model_output_rank"));
    }
    let mut count = 1usize;
    for (dim, wanted) in shape.iter().zip(expected) {
        let dim = usize::try_from(*dim)
            .ok()
            .filter(|n| *n > 0)
            .ok_or_else(|| ModelFailure::incompatible("model_output_dimension"))?;
        if wanted.is_some_and(|n| n != dim) {
            return Err(ModelFailure::incompatible("model_output_shape"));
        }
        count = count
            .checked_mul(dim)
            .filter(|n| *n <= 16 * 1024 * 1024)
            .ok_or_else(|| ModelFailure::incompatible("model_output_allocation_limit"))?;
    }
    if count != data.len() || data.iter().any(|v| !v.is_finite()) {
        return Err(ModelFailure::incompatible("model_output_values"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tensor_and_token_boundaries_reject_overflow_and_nonfinite_outputs() {
        for tokens in [4095, 4096, 4097] {
            assert_eq!(input_elements(1, tokens).is_ok(), tokens <= 4096);
        }
        assert!(input_elements(256, 4096).is_ok());
        assert!(input_elements(257, 4096).is_err());
        assert!(input_elements(usize::MAX, usize::MAX).is_err());
        assert!(tensor_shape(&[1, 3], &[Some(1), Some(3)], &[1.0, 2.0, 3.0]).is_ok());
        for (shape, data) in [
            (vec![1, 2], vec![1.0, 2.0]),
            (vec![1, 3], vec![1.0, f32::NAN, 3.0]),
            (vec![-1, 3], vec![]),
            (vec![i64::MAX, i64::MAX], vec![]),
        ] {
            assert!(tensor_shape(&shape, &[Some(1), Some(3)], &data).is_err());
        }
    }
}
