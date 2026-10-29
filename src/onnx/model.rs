//! Artifact pins and supported preprocessing/output contracts. Verification binds
//! bytes to a caller-trusted manifest; it does not establish model accuracy.
use super::checked::{ModelFailure, ModelKind};
use serde::{Deserialize, Serialize};

pub(super) const NER_LABEL_NAMES: [&str; 9] = [
    "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
];

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelProfile {
    HuggingFaceMaskedMeanV1,
    HuggingFaceBio9V1,
    HuggingFaceEkman7V1,
    HuggingFaceNli3V1,
}

impl ModelProfile {
    pub fn kind(self) -> ModelKind {
        match self {
            Self::HuggingFaceMaskedMeanV1 => ModelKind::Embedding,
            Self::HuggingFaceBio9V1 => ModelKind::Ner,
            Self::HuggingFaceEkman7V1 => ModelKind::Emotion,
            Self::HuggingFaceNli3V1 => ModelKind::Nli,
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
    pub labels: Vec<String>,
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
        let canonical = canonical_labels(expected);
        if self.labels.len() != canonical.len()
            || self.labels.iter().any(|label| {
                label.len() > 32
                    || !canonical.contains(&label.as_str())
                    || self.labels.iter().filter(|other| *other == label).count() != 1
            })
        {
            return Err(ModelFailure::incompatible("model_profile_labels"));
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
        ModelKind::Ner => &NER_LABEL_NAMES,
        ModelKind::Emotion => crate::craft::emotion::EKMAN_LABELS,
        ModelKind::Nli => crate::substrate::nli::NLI_LABELS,
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
}

#[cfg(feature = "onnx")]
impl ModelMetadata {
    pub fn unverified(kind: ModelKind) -> Self {
        Self {
            identity: ModelIdentity::unverified(kind),
            canonical_to_source: Vec::new(),
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
        let canonical = canonical_labels(expected);
        Ok(Self {
            identity: ModelIdentity {
                kind: expected,
                verification: ModelVerification::VerifiedManifestV1,
                manifest: Some(manifest.clone()),
            },
            canonical_to_source: canonical
                .iter()
                .map(|label| manifest.labels.iter().position(|v| v == label).unwrap())
                .collect(),
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

    pub fn validate_input_names<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), ModelFailure> {
        let names: Vec<_> = names.into_iter().take(4).collect();
        if self.identity.manifest.is_some()
            && (!(2..=3).contains(&names.len())
                || !names.contains(&"input_ids")
                || !names.contains(&"attention_mask")
                || names.iter().any(|name| {
                    !matches!(*name, "input_ids" | "attention_mask" | "token_type_ids")
                        || names.iter().filter(|other| *other == name).count() != 1
                }))
        {
            return Err(ModelFailure::incompatible("model_profile_inputs"));
        }
        Ok(())
    }

    pub fn output_shape(&self) -> Vec<Option<usize>> {
        match self.identity.kind {
            ModelKind::Embedding => vec![None, None, None],
            ModelKind::Ner => vec![None, None, Some(9)],
            ModelKind::Emotion => vec![None, Some(7)],
            ModelKind::Nli => vec![None, Some(3)],
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
