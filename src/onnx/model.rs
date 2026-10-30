//! Artifact pins and supported preprocessing/output contracts. Verification binds
//! bytes to a caller-trusted manifest; it does not establish model accuracy.
use super::checked::{ModelFailure, ModelKind};
use serde::{Deserialize, Serialize};

pub(super) const NER_LABEL_NAMES: [&str; 9] = [
    "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
];
pub(super) const NOMINAL_LABEL_NAMES: [&str; 3] = ["O", "B-NOM", "I-NOM"];
#[cfg(feature = "onnx")]
pub(super) const NARRATIVE_INPUT_NAMES: [&str; 6] = [
    "input_ids",
    "attention_mask",
    "mention_starts",
    "mention_ends",
    "quote_starts",
    "quote_ends",
];
#[cfg(feature = "onnx")]
pub(super) const NARRATIVE_OUTPUT_NAMES: [&str; 4] = [
    "entity_logits",
    "nominal_logits",
    "coref_scores",
    "attribution_scores",
];

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelProfile {
    HuggingFaceMaskedMeanV1,
    HuggingFaceBio9V1,
    HuggingFaceEkman7V1,
    HuggingFaceNli3V1,
    /// Six int64 inputs (`input_ids`, `attention_mask`, `mention_starts`,
    /// `mention_ends`, `quote_starts`, `quote_ends`) and four float outputs
    /// (`entity_logits`, `nominal_logits`, `coref_scores`, `attribution_scores`)
    /// over one 400-token window wrapped in `[CLS]`/`[SEP]`.
    NarrativeMultitaskV1,
}

impl ModelProfile {
    pub fn kind(self) -> ModelKind {
        match self {
            Self::HuggingFaceMaskedMeanV1 => ModelKind::Embedding,
            Self::HuggingFaceBio9V1 => ModelKind::Ner,
            Self::HuggingFaceEkman7V1 => ModelKind::Emotion,
            Self::HuggingFaceNli3V1 => ModelKind::Nli,
            Self::NarrativeMultitaskV1 => ModelKind::NarrativeMultitask,
        }
    }
}

/// V1 profiles encode with tokenizer special tokens, reject tokenizer-side
/// truncation/padding, and right-pad IDs with `padding_token_id` and masks/types
/// with zero. Embeddings use masked mean pooling without normalization; class
/// logits are softmaxed and remapped to the crate's canonical label order.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelManifest {
    pub schema_version: u32,
    pub kind: ModelKind,
    pub profile: ModelProfile,
    pub model_sha256: String,
    pub tokenizer_sha256: String,
    pub max_tokens: usize,
    pub padding_token_id: u32,
    /// Exact supported label names in source-logit order; empty for embeddings.
    /// The narrative model lists its entity head here.
    pub labels: Vec<String>,
    /// The narrative model's nominal-mention labels in source-logit order
    /// (`O`, `B-NOM`, `I-NOM`); empty for every other kind.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nominal_labels: Vec<String>,
    /// Content tokens per narrative window (the training window); defaults to 400.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_tokens: Option<usize>,
}

fn permutation_of(labels: &[String], canonical: &[&str]) -> bool {
    labels.len() == canonical.len()
        && labels.iter().all(|label| {
            label.len() <= 32
                && canonical.contains(&label.as_str())
                && labels.iter().filter(|other| *other == label).count() == 1
        })
}

impl ModelManifest {
    pub fn validate(&self, expected: ModelKind) -> Result<(), ModelFailure> {
        if self.schema_version != 1 {
            return Err(ModelFailure::incompatible("model_manifest_version"));
        }
        if self.kind != expected || self.profile.kind() != expected {
            return Err(ModelFailure::incompatible("model_profile_kind"));
        }
        if self.max_tokens == 0 || self.max_tokens > 4096 {
            return Err(ModelFailure::incompatible("model_profile_token_limit"));
        }
        for digest in [&self.model_sha256, &self.tokenizer_sha256] {
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(ModelFailure::incompatible("model_manifest_digest_format"));
            }
        }
        if !permutation_of(&self.labels, canonical_labels(expected))
            || !permutation_of(&self.nominal_labels, nominal_labels(expected))
        {
            return Err(ModelFailure::incompatible("model_profile_labels"));
        }
        if let Some(window) = self.window_tokens
            && (expected != ModelKind::NarrativeMultitask
                || window == 0
                || window + 2 > self.max_tokens)
        {
            return Err(ModelFailure::incompatible("model_profile_window"));
        }
        Ok(())
    }

    #[cfg(feature = "onnx")]
    pub fn from_json(bytes: &[u8]) -> Result<Self, ModelFailure> {
        if bytes.is_empty() || bytes.len() > 16 * 1024 {
            return Err(ModelFailure::incompatible("model_manifest_size"));
        }
        let manifest: Self = serde_json::from_slice(bytes)
            .map_err(|_| ModelFailure::incompatible("model_manifest_invalid"))?;
        manifest.validate(manifest.kind)?;
        Ok(manifest)
    }
}

fn canonical_labels(kind: ModelKind) -> &'static [&'static str] {
    match kind {
        ModelKind::Embedding => &[],
        ModelKind::Ner | ModelKind::NarrativeMultitask => &NER_LABEL_NAMES,
        ModelKind::Emotion => crate::craft::emotion::EKMAN_LABELS,
        ModelKind::Nli => crate::substrate::nli::NLI_LABELS,
    }
}

fn nominal_labels(kind: ModelKind) -> &'static [&'static str] {
    match kind {
        ModelKind::NarrativeMultitask => &NOMINAL_LABEL_NAMES,
        _ => &[],
    }
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelVerification {
    UnverifiedLegacy,
    VerifiedManifestV1,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelIdentity {
    pub kind: ModelKind,
    pub verification: ModelVerification,
    pub manifest: Option<ModelManifest>,
}

impl ModelIdentity {
    pub fn validate(&self, expected: ModelKind) -> Result<(), ModelFailure> {
        if self.kind != expected {
            return Err(ModelFailure::incompatible("model_identity_kind"));
        }
        match (self.verification, &self.manifest) {
            (ModelVerification::UnverifiedLegacy, None) => Ok(()),
            (ModelVerification::VerifiedManifestV1, Some(manifest)) => manifest.validate(expected),
            _ => Err(ModelFailure::incompatible("model_identity_verification")),
        }
    }

    pub fn unverified(kind: ModelKind) -> Self {
        Self {
            kind,
            verification: ModelVerification::UnverifiedLegacy,
            manifest: None,
        }
    }
}

#[cfg(feature = "onnx")]
pub(super) struct ModelMetadata {
    pub identity: ModelIdentity,
    canonical_to_source: Vec<usize>,
    nominal_to_source: Vec<usize>,
}

#[cfg(feature = "onnx")]
impl ModelMetadata {
    pub fn unverified(kind: ModelKind) -> Self {
        Self {
            identity: ModelIdentity::unverified(kind),
            canonical_to_source: Vec::new(),
            nominal_to_source: Vec::new(),
        }
    }

    pub fn verify(
        expected: ModelKind,
        manifest: &ModelManifest,
        model: &[u8],
        tokenizer: &[u8],
    ) -> Result<Self, ModelFailure> {
        manifest.validate(expected)?;
        validate_artifact_sizes(model, tokenizer)?;
        if sha256_hex(model) != manifest.model_sha256 {
            return Err(ModelFailure::incompatible("model_digest_mismatch"));
        }
        if sha256_hex(tokenizer) != manifest.tokenizer_sha256 {
            return Err(ModelFailure::incompatible("tokenizer_digest_mismatch"));
        }
        let order = |canonical: &[&str], source: &[String]| -> Vec<usize> {
            canonical
                .iter()
                .map(|label| source.iter().position(|v| v == label).unwrap())
                .collect()
        };
        Ok(Self {
            identity: ModelIdentity {
                kind: expected,
                verification: ModelVerification::VerifiedManifestV1,
                manifest: Some(manifest.clone()),
            },
            canonical_to_source: order(canonical_labels(expected), &manifest.labels),
            nominal_to_source: order(nominal_labels(expected), &manifest.nominal_labels),
        })
    }

    pub fn validate_tokenizer(
        &self,
        tokenizer: &tokenizers::Tokenizer,
    ) -> Result<(), ModelFailure> {
        if let Some(manifest) = &self.identity.manifest
            && (tokenizer.get_padding().is_some()
                || tokenizer.get_truncation().is_some()
                || tokenizer.id_to_token(manifest.padding_token_id).is_none())
        {
            return Err(ModelFailure::incompatible(
                "model_profile_tokenizer_settings",
            ));
        }
        Ok(())
    }

    pub fn validate_tokens(&self, tokens: usize) -> Result<(), ModelFailure> {
        super::checked::input_elements(1, tokens)?;
        if self
            .identity
            .manifest
            .as_ref()
            .is_some_and(|manifest| tokens > manifest.max_tokens)
        {
            return Err(ModelFailure::execution("model_profile_token_limit"));
        }
        Ok(())
    }

    /// Content tokens per narrative window: the manifest's `windowTokens`, else the
    /// default the first exports trained with.
    pub fn window_tokens(&self) -> usize {
        self.identity
            .manifest
            .as_ref()
            .and_then(|manifest| manifest.window_tokens)
            .unwrap_or(super::checked::NARRATIVE_WINDOW_TOKENS)
    }

    pub fn padding_id(&self) -> i64 {
        self.identity
            .manifest
            .as_ref()
            .map_or(0, |manifest| i64::from(manifest.padding_token_id))
    }

    pub fn probabilities(&self, logits: &[f32]) -> Result<Vec<f64>, ModelFailure> {
        let probabilities = super::decode::softmax(logits);
        if self.identity.manifest.is_none() {
            return Ok(probabilities);
        }
        if probabilities.len() != self.canonical_to_source.len() {
            return Err(ModelFailure::incompatible("model_profile_output_classes"));
        }
        Ok(self
            .canonical_to_source
            .iter()
            .map(|&source| probabilities[source])
            .collect())
    }

    pub fn ner_labels(
        &self,
        logits: &[f32],
        batch: usize,
        max_len: usize,
        num_labels: usize,
        seq_len: usize,
    ) -> Vec<usize> {
        let order = self
            .identity
            .manifest
            .is_some()
            .then_some(self.canonical_to_source.as_slice());
        super::decode::argmax_labels(logits, batch, max_len, num_labels, seq_len, order)
    }

    /// Nominal-head labels in canonical `O`, `B-NOM`, `I-NOM` order.
    pub fn nominal_labels(
        &self,
        logits: &[f32],
        batch: usize,
        max_len: usize,
        seq_len: usize,
    ) -> Vec<usize> {
        let order = self
            .identity
            .manifest
            .is_some()
            .then_some(self.nominal_to_source.as_slice());
        super::decode::argmax_labels(
            logits,
            batch,
            max_len,
            NOMINAL_LABEL_NAMES.len(),
            seq_len,
            order,
        )
    }

    pub fn validate_input_names<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), ModelFailure> {
        if self.identity.manifest.is_none() {
            return Ok(());
        }
        let names: Vec<_> = names.into_iter().take(8).collect();
        let unique = names
            .iter()
            .all(|name| names.iter().filter(|other| *other == name).count() == 1);
        let expected = match self.identity.kind {
            ModelKind::NarrativeMultitask => {
                names.len() == NARRATIVE_INPUT_NAMES.len()
                    && NARRATIVE_INPUT_NAMES
                        .iter()
                        .all(|name| names.contains(name))
            }
            _ => {
                (2..=3).contains(&names.len())
                    && names.contains(&"input_ids")
                    && names.contains(&"attention_mask")
                    && names.iter().all(|name| {
                        matches!(*name, "input_ids" | "attention_mask" | "token_type_ids")
                    })
            }
        };
        if !(unique && expected) {
            return Err(ModelFailure::incompatible("model_profile_inputs"));
        }
        Ok(())
    }

    /// Tensor rank a profile input must have: token tensors are `[batch, sequence]`,
    /// the narrative model's span-index inputs are one-dimensional.
    pub fn input_rank(&self, name: &str) -> usize {
        match name {
            "mention_starts" | "mention_ends" | "quote_starts" | "quote_ends" => 1,
            _ => 2,
        }
    }

    /// Expected output tensors in graph order; `None` dimensions are dynamic.
    pub fn output_shapes(&self) -> Vec<Vec<Option<usize>>> {
        match self.identity.kind {
            ModelKind::Embedding => vec![vec![None, None, None]],
            ModelKind::Ner => vec![vec![None, None, Some(9)]],
            ModelKind::Emotion => vec![vec![None, Some(7)]],
            ModelKind::Nli => vec![vec![None, Some(3)]],
            ModelKind::NarrativeMultitask => vec![
                vec![None, None, Some(NER_LABEL_NAMES.len())],
                vec![None, None, Some(NOMINAL_LABEL_NAMES.len())],
                vec![None, None],
                vec![None, None],
            ],
        }
    }
}

#[cfg(feature = "onnx")]
pub(super) fn validate_artifact_sizes(model: &[u8], tokenizer: &[u8]) -> Result<(), ModelFailure> {
    if model.is_empty()
        || model.len() > 1024 * 1024 * 1024
        || tokenizer.is_empty()
        || tokenizer.len() > 64 * 1024 * 1024
    {
        return Err(ModelFailure::incompatible("model_file_size_or_type"));
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}
