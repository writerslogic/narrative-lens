//! Model-runtime abstraction for the ONNX models narrative-lens can optionally
//! use: SBERT sentence embeddings, BERT NER, RoBERTa/DeBERTa NLI, and an Ekman-7
//! emotion classifier.
//!
//! The concrete inference engine is selected at compile time:
//! - native (`not(target_arch = "wasm32")`) uses [`ort`] (ONNX Runtime);
//! - `wasm32` uses [`tract`], a pure-Rust CPU ONNX runtime, because `ort`
//!   officially dropped the `wasm32` target.
//!
//! Both backends preserve the same graceful-degradation contract: when a model
//! (or, on wasm, the tokenizer) is unavailable, [`InferenceBackend::embed`]
//! returns one empty vector per input and [`InferenceBackend::tag`] returns an
//! empty list. Nothing panics on a missing model.
//!
//! This crate follows a "bring your own model, no download" contract (see
//! `docs/ARCHITECTURE.md`): no model weights are bundled or fetched. Without the
//! `onnx` feature enabled, [`backend`] returns a null backend with the same
//! empty-output contract as a feature-enabled build with no model files present
//! -- so callers (`substrate::sbert`, `continuity::ner`, `substrate::nli`,
//! `craft::emotion`) need no `#[cfg]` of their own.

/// A model runtime that can produce sentence embeddings and NER tags.
///
/// Return types mirror the free functions in [`crate::substrate::sbert`] and
/// [`crate::continuity::ner`] so callers are unaffected by the indirection:
/// - `embed` returns one `Vec<f64>` per input text (empty when unavailable),
///   matching `sbert::encode_batch`.
/// - `tag` returns `(entity_text, label, start_byte, end_byte)` tuples, matching
///   `ner::ner_extract_entities`.
pub mod checked;
pub mod model;

static MODEL_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Changes after a process-wide model setter; injected replacements use configuration.model_revision.
pub fn model_generation() -> u64 {
    MODEL_GENERATION.load(std::sync::atomic::Ordering::SeqCst)
}

pub trait InferenceBackend: Sync {
    fn model_identity(
        &self,
        kind: checked::ModelKind,
    ) -> Result<model::ModelIdentity, checked::ModelFailure> {
        self.check_model(kind)?;
        Ok(model::ModelIdentity::unverified(kind))
    }

    /// Backends opt into checked calls only when availability is known.
    fn check_model(
        &self,
        _kind: checked::ModelKind,
    ) -> std::result::Result<(), checked::ModelFailure> {
        Err(checked::ModelFailure::unavailable(
            "backend_does_not_advertise_capability",
        ))
    }
    fn classify_pairs_checked(
        &self,
        pairs: &[(&str, &str)],
    ) -> std::result::Result<Vec<Vec<f64>>, checked::ModelFailure> {
        Ok(self.classify_pairs(pairs))
    }

    fn embed_checked(
        &self,
        texts: &[&str],
    ) -> std::result::Result<Vec<Vec<f64>>, checked::ModelFailure> {
        Ok(self.embed(texts))
    }
    fn classify_checked(
        &self,
        texts: &[&str],
    ) -> std::result::Result<Vec<Vec<f64>>, checked::ModelFailure> {
        Ok(self.classify(texts))
    }
    fn tag_batch_checked(
        &self,
        _texts: &[&str],
    ) -> std::result::Result<Vec<NerSpans>, checked::ModelFailure> {
        Err(checked::ModelFailure::execution(
            "backend_does_not_provide_checked_ner",
        ))
    }
    /// Embed each text into a sentence vector. Entries for which embedding is
    /// unavailable come back as an empty `Vec`.
    fn embed(&self, texts: &[&str]) -> Vec<Vec<f64>>;

    /// Run token-classification NER over `text`, returning entity spans.
    fn tag(&self, text: &str) -> Vec<(String, String, usize, usize)>;

    /// Tag a batch of texts, returning one entity list per input (same order).
    /// The default tags each text separately; backends that can run the model
    /// once over a padded batch (the native ONNX path) override this for speed.
    fn tag_batch(&self, texts: &[&str]) -> Vec<Vec<(String, String, usize, usize)>> {
        texts.iter().map(|t| self.tag(t)).collect()
    }

    /// Classify each text into a probability distribution over the model's
    /// classes (one softmaxed `Vec<f64>` per text; empty when the model is
    /// unavailable). The label mapping lives in [`crate::craft::emotion`]. The
    /// default degrades to empty; the real backends override it.
    fn classify(&self, texts: &[&str]) -> Vec<Vec<f64>> {
        vec![Vec::new(); texts.len()]
    }

    /// Sequence-PAIR classification (NLI): each `(premise, hypothesis)` pair is
    /// encoded together (premise `[SEP]` hypothesis) and softmaxed over the
    /// model's classes (one `Vec<f64>` per pair; empty when unavailable). The
    /// label mapping lives in [`crate::substrate::nli`]. Default degrades to
    /// empty; real backends override.
    fn classify_pairs(&self, pairs: &[(&str, &str)]) -> Vec<Vec<f64>> {
        vec![Vec::new(); pairs.len()]
    }
}

/// Runtime-agnostic post-processing (masked pooling, BIO decode) shared by both
/// backends; inference kernels still require numerical parity checks.
#[cfg(feature = "onnx")]
mod decode;

#[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
mod ort_backend;
#[cfg(all(feature = "onnx", target_arch = "wasm32"))]
mod tract_backend;

#[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
use ort_backend as backend_impl;
#[cfg(all(feature = "onnx", target_arch = "wasm32"))]
use tract_backend as backend_impl;

/// A backend with no model runtime at all: every call degrades to empty output.
/// Active whenever the `onnx` feature is disabled, so consumers never need to
/// `#[cfg]` their own call sites -- a "bring your own model" build with the
/// feature off behaves exactly like one with the feature on and no model files
/// on disk.
#[cfg(not(feature = "onnx"))]
struct NullBackend;

#[cfg(not(feature = "onnx"))]
impl InferenceBackend for NullBackend {
    fn check_model(
        &self,
        _kind: checked::ModelKind,
    ) -> std::result::Result<(), checked::ModelFailure> {
        Err(checked::ModelFailure::unavailable("onnx_feature_disabled"))
    }

    fn embed(&self, texts: &[&str]) -> Vec<Vec<f64>> {
        vec![Vec::new(); texts.len()]
    }

    fn tag(&self, _text: &str) -> Vec<(String, String, usize, usize)> {
        Vec::new()
    }
}

/// The process-wide active inference backend.
pub fn backend() -> &'static dyn InferenceBackend {
    #[cfg(feature = "onnx")]
    {
        backend_impl::backend()
    }
    #[cfg(not(feature = "onnx"))]
    {
        static BACKEND: NullBackend = NullBackend;
        &BACKEND
    }
}

/// Load pinned model/tokenizer files from the legacy per-kind subdirectory.
/// An invalid replacement disables that model and invalidates retained results.
#[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
pub fn set_model_directory_verified(
    kind: checked::ModelKind,
    directory: &str,
    manifest: &model::ModelManifest,
) -> Result<model::ModelIdentity, checked::ModelFailure> {
    backend_impl::set_model_directory_verified(kind, directory, manifest)
}

/// Load exactly these pinned bytes on native or WASM. The manifest must come
/// from a caller-trusted source; artifact verification does not attest accuracy.
#[cfg(feature = "onnx")]
pub fn set_model_bytes_verified(
    kind: checked::ModelKind,
    model: &[u8],
    tokenizer: &[u8],
    manifest: &model::ModelManifest,
) -> Result<model::ModelIdentity, checked::ModelFailure> {
    backend_impl::set_model_bytes_verified(kind, model, tokenizer, manifest)
}

/// One NER tagging's output: `(entity_text, label, start_byte, end_byte)` spans.
pub type NerSpans = Vec<(String, String, usize, usize)>;

/// Caller-owned NER retention. The byte budget charges text, spans, and conservative
/// container overhead; it is not a process RSS limit.
pub struct NerCache {
    budget: usize,
    retained: usize,
    revision: String,
    generation: u64,
    entries: std::collections::BTreeMap<String, (NerSpans, usize)>,
}

impl NerCache {
    pub fn new(max_bytes: usize) -> crate::Result<Self> {
        if max_bytes > 256 * 1024 * 1024 {
            return Err(crate::Error::InvalidInput(
                "NER cache budget exceeds 256 MiB".into(),
            ));
        }
        Ok(Self {
            budget: max_bytes,
            retained: 0,
            revision: String::new(),
            generation: model_generation(),
            entries: Default::default(),
        })
    }
    pub fn clear(&mut self) {
        self.entries.clear();
        self.retained = 0;
    }
    pub fn retained_bytes(&self) -> usize {
        self.retained
    }

    /// Change `model_revision` when replacing an injected backend/model.
    pub fn tag_batch(
        &mut self,
        backend: &dyn InferenceBackend,
        model_revision: &str,
        texts: &[&str],
    ) -> Result<Vec<NerSpans>, checked::ModelFailure> {
        self.tag_batch_with_generation(backend, model_revision, texts, model_generation)
    }

    fn tag_batch_with_generation(
        &mut self,
        backend: &dyn InferenceBackend,
        model_revision: &str,
        texts: &[&str],
        generation_source: impl Fn() -> u64,
    ) -> Result<Vec<NerSpans>, checked::ModelFailure> {
        if model_revision.len() > 256 {
            return Err(checked::ModelFailure::incompatible("model_revision_limit"));
        }
        checked::validate_texts(texts, 256)?;
        let generation = generation_source();
        if self.revision != model_revision || self.generation != generation {
            self.clear();
            self.revision = model_revision.into();
            self.generation = generation;
        }
        if let Err(error) = backend.check_model(checked::ModelKind::Ner) {
            self.clear();
            return Err(error);
        }
        let mut output = vec![None; texts.len()];
        let mut misses = Vec::new();
        for (index, text) in texts.iter().enumerate() {
            if let Some((spans, _)) = self.entries.get(*text) {
                output[index] = Some(spans.clone());
            } else {
                misses.push(index);
            }
        }
        let inputs: Vec<_> = misses.iter().map(|&index| texts[index]).collect();
        let inferred = checked::entities(backend, &inputs);
        if generation != generation_source() {
            self.clear();
            return Err(checked::ModelFailure::execution(
                "model_changed_during_inference",
            ));
        }
        let spans = inferred?;
        for (index, spans) in misses.into_iter().zip(spans) {
            if generation == generation_source() {
                let charge = texts[index]
                    .len()
                    .saturating_add(1024)
                    .saturating_add(spans.capacity().saturating_mul(std::mem::size_of::<(
                        String,
                        String,
                        usize,
                        usize,
                    )>()))
                    .saturating_add(
                        spans
                            .iter()
                            .map(|(text, label, _, _)| {
                                text.capacity().saturating_add(label.capacity())
                            })
                            .fold(0usize, usize::saturating_add),
                    );
                if charge <= self.budget {
                    if let Some((_, old)) = self.entries.remove(texts[index]) {
                        self.retained -= old;
                    }
                    while self.retained.saturating_add(charge) > self.budget {
                        let Some((_, (_, removed))) = self.entries.pop_first() else {
                            break;
                        };
                        self.retained -= removed;
                    }
                    self.entries
                        .insert(texts[index].into(), (spans.clone(), charge));
                    self.retained += charge;
                }
            }
            output[index] = Some(spans);
        }
        if generation != generation_source() {
            self.clear();
            return Err(checked::ModelFailure::execution(
                "model_changed_during_inference",
            ));
        }
        Ok(output
            .into_iter()
            .map(|item| item.expect("NER output cardinality checked"))
            .collect())
    }
}

/// Legacy entry point; process-global manuscript retention has been removed.
/// Use a caller-owned [`NerCache`] to reuse windows across revisions.
pub fn tag_batch_cached(texts: &[&str]) -> Vec<NerSpans> {
    if backend().check_model(checked::ModelKind::Ner).is_err() {
        return vec![Vec::new(); texts.len()];
    }
    texts
        .chunks(256)
        .flat_map(|chunk| {
            checked::entities(backend(), chunk).unwrap_or_else(|_| vec![Vec::new(); chunk.len()])
        })
        .collect()
}

/// Set the directory searched for the SBERT model (`sbert-onnx/`).
/// May be called again to replace or retry the model; a no-op without `onnx`.
pub fn set_sbert_model_directory(path: &str) {
    #[cfg(feature = "onnx")]
    backend_impl::set_sbert_model_directory(path);
    #[cfg(not(feature = "onnx"))]
    let _ = path;
    #[cfg(any(not(feature = "onnx"), target_arch = "wasm32"))]
    MODEL_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// Set the directory searched for the NER model (`ner-onnx/`).
/// May be called again to replace or retry the model; a no-op without `onnx`.
pub fn set_ner_model_directory(path: &str) {
    #[cfg(feature = "onnx")]
    backend_impl::set_ner_model_directory(path);
    #[cfg(not(feature = "onnx"))]
    let _ = path;
    #[cfg(any(not(feature = "onnx"), target_arch = "wasm32"))]
    MODEL_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// Set the directory searched for the emotion model (`emotion-onnx/`).
/// May be called again to replace or retry the model; a no-op without `onnx`.
pub fn set_emotion_model_directory(path: &str) {
    #[cfg(feature = "onnx")]
    backend_impl::set_emotion_model_directory(path);
    #[cfg(not(feature = "onnx"))]
    let _ = path;
    #[cfg(any(not(feature = "onnx"), target_arch = "wasm32"))]
    MODEL_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// Set the directory searched for the NLI model (`nli-onnx/`).
/// May be called again to replace or retry the model; a no-op without `onnx`.
pub fn set_nli_model_directory(path: &str) {
    #[cfg(feature = "onnx")]
    backend_impl::set_nli_model_directory(path);
    #[cfg(not(feature = "onnx"))]
    let _ = path;
    #[cfg(any(not(feature = "onnx"), target_arch = "wasm32"))]
    MODEL_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// Provide the SBERT model + tokenizer as in-memory bytes (wasm only).
///
/// wasm has no filesystem, so the directory search the native backend uses
/// never resolves; the host (which *can* fetch over the network) hands the
/// model and tokenizer bytes in instead. Must be called before the first
/// embedding; setters also replace or retry previous models. `model` is raw ONNX,
/// `tokenizer` is the HuggingFace `tokenizer.json` bytes.
#[cfg(all(feature = "onnx", target_arch = "wasm32"))]
pub fn set_sbert_model_bytes(model: &[u8], tokenizer: &[u8]) {
    backend_impl::set_sbert_model_bytes(model, tokenizer);
    MODEL_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// Provide the NER model + tokenizer as in-memory bytes (wasm only).
/// See [`set_sbert_model_bytes`]: the host supplies the bytes because wasm has
/// no filesystem. May be called again to replace or retry a model.
#[cfg(all(feature = "onnx", target_arch = "wasm32"))]
pub fn set_ner_model_bytes(model: &[u8], tokenizer: &[u8]) {
    backend_impl::set_ner_model_bytes(model, tokenizer);
    MODEL_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// Provide the emotion model + tokenizer as in-memory bytes (wasm only).
/// See [`set_sbert_model_bytes`]: the host supplies the bytes because wasm has
/// no filesystem. May be called again to replace or retry a model.
#[cfg(all(feature = "onnx", target_arch = "wasm32"))]
pub fn set_emotion_model_bytes(model: &[u8], tokenizer: &[u8]) {
    backend_impl::set_emotion_model_bytes(model, tokenizer);
    MODEL_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// Provide the NLI model + tokenizer as in-memory bytes (wasm only).
/// See [`set_sbert_model_bytes`]: the host supplies the bytes because wasm has
/// no filesystem. May be called again to replace or retry a model.
#[cfg(all(feature = "onnx", target_arch = "wasm32"))]
pub fn set_nli_model_bytes(model: &[u8], tokenizer: &[u8]) {
    backend_impl::set_nli_model_bytes(model, tokenizer);
    MODEL_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Available;
    impl InferenceBackend for Available {
        fn check_model(&self, _: checked::ModelKind) -> Result<(), checked::ModelFailure> {
            Ok(())
        }
        fn embed(&self, _: &[&str]) -> Vec<Vec<f64>> {
            Vec::new()
        }
        fn tag(&self, _: &str) -> NerSpans {
            Vec::new()
        }
        fn tag_batch_checked(
            &self,
            texts: &[&str],
        ) -> Result<Vec<NerSpans>, checked::ModelFailure> {
            Ok(vec![Vec::new(); texts.len()])
        }
    }
    struct Unavailable;
    impl InferenceBackend for Unavailable {
        fn embed(&self, _: &[&str]) -> Vec<Vec<f64>> {
            Vec::new()
        }
        fn tag(&self, _: &str) -> NerSpans {
            Vec::new()
        }
    }
    #[test]
    fn ner_retention_is_bounded_clearable_and_isolated() {
        let mut cache = NerCache::new(2048).unwrap();
        for index in 0..100 {
            cache
                .tag_batch(
                    &Available,
                    "model-a",
                    &[&format!("private revision {index}")],
                )
                .unwrap();
            assert!(cache.retained_bytes() <= 2048);
        }
        assert!(cache.retained_bytes() > 0);
        let other = NerCache::new(2048).unwrap();
        assert_eq!(other.retained_bytes(), 0);
        cache.clear();
        assert_eq!(cache.retained_bytes(), 0);
        cache
            .tag_batch(&Available, "model-a", &["private"])
            .unwrap();
        assert!(
            cache
                .tag_batch(&Unavailable, "model-a", &["private"])
                .is_err()
        );
        assert_eq!(cache.retained_bytes(), 0);
    }
    #[test]
    fn ner_retention_budget_boundaries_and_model_revision() {
        for budget in [1030, 1031, 1032] {
            let mut cache = NerCache::new(budget).unwrap();
            cache.tag_batch(&Available, "a", &["private"]).unwrap();
            assert_eq!(cache.entries.contains_key("private"), budget >= 1031);
            cache.tag_batch(&Available, "b", &[]).unwrap();
            assert_eq!(cache.retained_bytes(), 0);
        }
        assert!(NerCache::new(256 * 1024 * 1024 + 1).is_err());
    }

    #[test]
    fn ner_batch_rejects_a_model_change_between_cached_hits_and_inference() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        struct Changing<'a> {
            generation: &'a AtomicU64,
            replace: AtomicBool,
        }
        impl InferenceBackend for Changing<'_> {
            fn check_model(&self, _: checked::ModelKind) -> Result<(), checked::ModelFailure> {
                Ok(())
            }
            fn embed(&self, _: &[&str]) -> Vec<Vec<f64>> {
                Vec::new()
            }
            fn tag(&self, _: &str) -> NerSpans {
                Vec::new()
            }
            fn tag_batch_checked(
                &self,
                texts: &[&str],
            ) -> Result<Vec<NerSpans>, checked::ModelFailure> {
                if self.replace.swap(false, Ordering::SeqCst) {
                    self.generation.fetch_add(1, Ordering::SeqCst);
                }
                Ok(texts
                    .iter()
                    .map(|text| vec![(text.to_string(), "PER".into(), 0, text.len())])
                    .collect())
            }
        }
        let generation = AtomicU64::new(0);
        let backend = Changing {
            generation: &generation,
            replace: AtomicBool::new(false),
        };
        let source = || generation.load(Ordering::SeqCst);
        let mut cache = NerCache::new(4096).unwrap();
        cache
            .tag_batch_with_generation(&backend, "model", &["Alice"], source)
            .unwrap();
        assert!(cache.entries.contains_key("Alice"));
        backend.replace.store(true, Ordering::SeqCst);
        let error = cache
            .tag_batch_with_generation(&backend, "model", &["Alice", "Bob"], source)
            .unwrap_err();
        assert_eq!(error.code, "model_changed_during_inference");
        assert_eq!(cache.retained_bytes(), 0);
        assert!(cache.entries.is_empty());
        let output = cache
            .tag_batch_with_generation(&backend, "model", &["Alice", "Bob"], source)
            .unwrap();
        assert_eq!(
            output
                .iter()
                .map(|spans| spans[0].0.as_str())
                .collect::<Vec<_>>(),
            ["Alice", "Bob"]
        );
    }
}
