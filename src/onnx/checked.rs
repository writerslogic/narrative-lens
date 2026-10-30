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
    /// One encoder with entity, nominal-mention, antecedent, and speaker heads
    /// (`scripts/multitask_train.py`).
    NarrativeMultitask,
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
/// Content tokens per narrative window when the manifest carries no `windowTokens`;
/// inference must use the training window so the heads stay inside their fitted
/// position range.
pub const NARRATIVE_WINDOW_TOKENS: usize = 400;
pub const NARRATIVE_TEXT_LIMIT: usize = 1024 * 1024;
pub const NARRATIVE_SPAN_LIMIT: usize = 10_000;
/// Mentions or quotations one window may carry; the pair matrices are quadratic in it.
pub const NARRATIVE_WINDOW_SPAN_LIMIT: usize = 512;

/// One narrative-model call: a text with the byte spans of the mention candidates
/// the antecedent head scores and the quotations the speaker head scores. Both
/// span lists must be sorted by `(start, end)`; the model scores antecedents in
/// that order.
#[derive(Debug, Clone, Copy)]
pub struct NarrativeRequest<'a> {
    pub text: &'a str,
    pub mentions: &'a [(usize, usize)],
    pub quotes: &'a [(usize, usize)],
}

/// One decoded window of a narrative-model call. Windows advance by half a
/// window, so consecutive windows overlap and a mention may appear in two.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NarrativeWindow {
    /// Byte range the window's tokens cover; starts are non-decreasing.
    pub start: usize,
    pub end: usize,
    pub entities: super::NerSpans,
    /// Nominal person-mention byte spans ("her mother").
    pub nominals: Vec<(usize, usize)>,
    /// Indices into the request's mentions whose tokens lie inside this window, ascending.
    pub mentions: Vec<usize>,
    /// Indices into the request's quotes whose tokens lie inside this window, ascending.
    pub quotes: Vec<usize>,
    /// `antecedents[i][0]` scores "no antecedent" (always 0); `antecedents[i][j + 1]`
    /// scores window mention `j` as the antecedent of window mention `i` (`j < i`).
    pub antecedents: Vec<Vec<f32>>,
    /// `speakers[q][0]` scores "speaker outside the window"; `speakers[q][j + 1]`
    /// scores window mention `j` as the speaker of window quote `q`.
    pub speakers: Vec<Vec<f32>>,
}

pub fn narrative(
    backend: &dyn InferenceBackend,
    request: &NarrativeRequest<'_>,
) -> Result<Vec<NarrativeWindow>, ModelFailure> {
    validate_narrative_request(request)?;
    backend.check_model(ModelKind::NarrativeMultitask)?;
    let windows = backend.narrative_checked(request)?;
    validate_narrative_windows(request, &windows)?;
    Ok(windows)
}

fn validate_spans(text: &str, spans: &[(usize, usize)], code: &str) -> Result<(), ModelFailure> {
    if spans.len() > NARRATIVE_SPAN_LIMIT {
        return Err(ModelFailure::execution("narrative_span_limit"));
    }
    let mut previous = (0usize, 0usize);
    for &(start, end) in spans {
        if start >= end
            || end > text.len()
            || !text.is_char_boundary(start)
            || !text.is_char_boundary(end)
            || (start, end) < previous
        {
            return Err(ModelFailure::execution(code));
        }
        previous = (start, end);
    }
    Ok(())
}

fn validate_narrative_request(request: &NarrativeRequest<'_>) -> Result<(), ModelFailure> {
    if request.text.len() > NARRATIVE_TEXT_LIMIT {
        return Err(ModelFailure::execution("model_text_limit"));
    }
    validate_spans(request.text, request.mentions, "narrative_mention_spans")?;
    validate_spans(request.text, request.quotes, "narrative_quote_spans")
}

fn validate_window_indices(
    indices: &[usize],
    spans: &[(usize, usize)],
    window: (usize, usize),
) -> bool {
    indices.len() <= NARRATIVE_WINDOW_SPAN_LIMIT
        && indices.windows(2).all(|pair| pair[0] < pair[1])
        && indices.iter().all(|&index| {
            spans
                .get(index)
                .is_some_and(|&(start, end)| start < window.1 && end > window.0)
        })
}

fn validate_narrative_windows(
    request: &NarrativeRequest<'_>,
    windows: &[NarrativeWindow],
) -> Result<(), ModelFailure> {
    let text = request.text;
    let mut previous_start = 0usize;
    for window in windows {
        if window.start < previous_start
            || window.start > window.end
            || window.end > text.len()
            || !text.is_char_boundary(window.start)
            || !text.is_char_boundary(window.end)
        {
            return Err(ModelFailure::incompatible("narrative_window_range"));
        }
        previous_start = window.start;
        for (quote, label, start, end) in &window.entities {
            if start >= end
                || text.get(*start..*end) != Some(quote.as_str())
                || !matches!(label.as_str(), "MISC" | "PER" | "ORG" | "LOC")
            {
                return Err(ModelFailure::incompatible("ner_invalid_span_or_label"));
            }
        }
        if window.nominals.iter().any(|&(start, end)| {
            start >= end
                || text
                    .get(start..end)
                    .is_none_or(|span| span.trim().is_empty())
        }) {
            return Err(ModelFailure::incompatible("narrative_invalid_nominal_span"));
        }
        let bounds = (window.start, window.end);
        if !validate_window_indices(&window.mentions, request.mentions, bounds)
            || !validate_window_indices(&window.quotes, request.quotes, bounds)
        {
            return Err(ModelFailure::incompatible("narrative_window_indices"));
        }
        let columns = window.mentions.len() + 1;
        let valid_matrix = |matrix: &[Vec<f32>], rows: usize| {
            matrix.len() == rows
                && matrix
                    .iter()
                    .all(|row| row.len() == columns && row.iter().all(|v| v.is_finite()))
        };
        if !valid_matrix(&window.antecedents, window.mentions.len())
            || !valid_matrix(&window.speakers, window.quotes.len())
        {
            return Err(ModelFailure::incompatible("narrative_matrix_shape"));
        }
    }
    Ok(())
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
