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
/// - `tag` returns `(entity_text, label, start_char, end_char)` tuples, matching
///   `ner::ner_extract_entities`.
pub trait InferenceBackend: Sync {
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
/// backends so a wasm result is bit-identical to the native one.
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

/// One NER tagging's output: `(entity_text, label, start_char, end_char)` spans.
type NerSpans = Vec<(String, String, usize, usize)>;

/// Process-wide memo for NER window tagging, keyed by a stable content hash.
///
/// NER over a window is a pure function of that window's bytes -- the tagger
/// sees one window at a time with no cross-window context (see
/// `crate::continuity::ner`) -- so a window that reappears byte-for-byte (the
/// overwhelming case after a localized edit: only the touched windows change)
/// can reuse its prior spans instead of paying another model inference. The
/// window text is stored alongside the spans so a 64-bit hash collision
/// recomputes rather than returning wrong spans.
fn ner_memo() -> &'static std::sync::RwLock<std::collections::HashMap<u64, (String, NerSpans)>> {
    static MEMO: std::sync::OnceLock<
        std::sync::RwLock<std::collections::HashMap<u64, (String, NerSpans)>>,
    > = std::sync::OnceLock::new();
    MEMO.get_or_init(|| std::sync::RwLock::new(std::collections::HashMap::new()))
}

/// FNV-1a (64-bit) over `s`'s bytes. A stable, dependency-free content hash so a
/// persisted NER memo keeps the same keys across runs. Locked by the canonical
/// vectors in the tests below.
fn fnv1a64(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Tag a batch of windows through the active backend, memoized per window by a
/// stable content hash (see [`ner_memo`]). Returns one span list per input, in
/// order -- byte-identical to [`InferenceBackend::tag_batch`], just skipping the
/// model for windows already tagged. Only the cache-miss windows hit the model,
/// so a localized edit re-tags only the windows whose text changed.
pub fn tag_batch_cached(texts: &[&str]) -> Vec<NerSpans> {
    let keys: Vec<u64> = texts.iter().map(|t| fnv1a64(t)).collect();

    let mut out: Vec<Option<NerSpans>> = vec![None; texts.len()];
    let mut miss_idx: Vec<usize> = Vec::new();
    {
        let memo = ner_memo().read().unwrap_or_else(|e| e.into_inner());
        for (i, (&key, &text)) in keys.iter().zip(texts).enumerate() {
            match memo.get(&key) {
                Some((stored, spans)) if stored == text => out[i] = Some(spans.clone()),
                _ => miss_idx.push(i),
            }
        }
    }
    if miss_idx.is_empty() {
        return out.into_iter().map(Option::unwrap_or_default).collect();
    }

    let miss_texts: Vec<&str> = miss_idx.iter().map(|&i| texts[i]).collect();
    let tagged = backend().tag_batch(&miss_texts);
    {
        let mut memo = ner_memo().write().unwrap_or_else(|e| e.into_inner());
        for (&i, spans) in miss_idx.iter().zip(tagged) {
            memo.insert(keys[i], (texts[i].to_string(), spans.clone()));
            out[i] = Some(spans);
        }
    }
    out.into_iter().map(Option::unwrap_or_default).collect()
}

/// Set the directory searched for the SBERT model (`sbert-onnx/`).
/// Must be called before the first embedding. No-op once a model load has been
/// attempted, and a no-op entirely without the `onnx` feature.
pub fn set_sbert_model_directory(path: &str) {
    #[cfg(feature = "onnx")]
    backend_impl::set_sbert_model_directory(path);
    #[cfg(not(feature = "onnx"))]
    let _ = path;
}

/// Set the directory searched for the NER model (`ner-onnx/`).
/// Must be called before the first NER call. No-op once a model load has been
/// attempted, and a no-op entirely without the `onnx` feature.
pub fn set_ner_model_directory(path: &str) {
    #[cfg(feature = "onnx")]
    backend_impl::set_ner_model_directory(path);
    #[cfg(not(feature = "onnx"))]
    let _ = path;
}

/// Set the directory searched for the emotion model (`emotion-onnx/`).
/// Must be called before the first classification. No-op once a model load has
/// been attempted, and a no-op entirely without the `onnx` feature.
pub fn set_emotion_model_directory(path: &str) {
    #[cfg(feature = "onnx")]
    backend_impl::set_emotion_model_directory(path);
    #[cfg(not(feature = "onnx"))]
    let _ = path;
}

/// Set the directory searched for the NLI model (`nli-onnx/`).
/// Must be called before the first pair classification. No-op once a model load
/// has been attempted, and a no-op entirely without the `onnx` feature.
pub fn set_nli_model_directory(path: &str) {
    #[cfg(feature = "onnx")]
    backend_impl::set_nli_model_directory(path);
    #[cfg(not(feature = "onnx"))]
    let _ = path;
}

/// Provide the SBERT model + tokenizer as in-memory bytes (wasm only).
///
/// wasm has no filesystem, so the directory search the native backend uses
/// never resolves; the host (which *can* fetch over the network) hands the
/// model and tokenizer bytes in instead. Must be called before the first
/// embedding; no-op once a load has been attempted. `model` is raw ONNX,
/// `tokenizer` is the HuggingFace `tokenizer.json` bytes.
#[cfg(all(feature = "onnx", target_arch = "wasm32"))]
pub fn set_sbert_model_bytes(model: &[u8], tokenizer: &[u8]) {
    backend_impl::set_sbert_model_bytes(model, tokenizer);
}

/// Provide the NER model + tokenizer as in-memory bytes (wasm only).
/// See [`set_sbert_model_bytes`]: the host supplies the bytes because wasm has
/// no filesystem. Must be called before the first NER call; no-op once a load
/// has been attempted.
#[cfg(all(feature = "onnx", target_arch = "wasm32"))]
pub fn set_ner_model_bytes(model: &[u8], tokenizer: &[u8]) {
    backend_impl::set_ner_model_bytes(model, tokenizer);
}

/// Provide the emotion model + tokenizer as in-memory bytes (wasm only).
/// See [`set_sbert_model_bytes`]: the host supplies the bytes because wasm has
/// no filesystem. Must be called before the first classification; no-op once a
/// load has been attempted.
#[cfg(all(feature = "onnx", target_arch = "wasm32"))]
pub fn set_emotion_model_bytes(model: &[u8], tokenizer: &[u8]) {
    backend_impl::set_emotion_model_bytes(model, tokenizer);
}

/// Provide the NLI model + tokenizer as in-memory bytes (wasm only).
/// See [`set_sbert_model_bytes`]: the host supplies the bytes because wasm has
/// no filesystem. Must be called before the first pair classification; no-op
/// once a load has been attempted.
#[cfg(all(feature = "onnx", target_arch = "wasm32"))]
pub fn set_nli_model_bytes(model: &[u8], tokenizer: &[u8]) {
    backend_impl::set_nli_model_bytes(model, tokenizer);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hash is the persisted memo's key, so its values must never drift.
    /// These are the canonical FNV-1a-64 vectors.
    #[test]
    fn fnv1a64_matches_canonical_vectors() {
        assert_eq!(fnv1a64(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64("a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64("foobar"), 0x85944171f73967e8);
    }

    /// The memo preserves the backend contract: one span list per input, in
    /// order, and a repeated call returns identical output (served from the
    /// memo). With no model loaded the backend degrades to empty spans, which
    /// still exercises the hit path on the second call.
    #[test]
    fn tag_batch_cached_is_shape_preserving_and_deterministic() {
        let texts = [
            "tag_batch_cached test sentinel alpha",
            "tag_batch_cached test sentinel beta",
            "tag_batch_cached test sentinel alpha",
        ];
        let first = tag_batch_cached(&texts);
        assert_eq!(first.len(), texts.len());
        assert_eq!(first[0], first[2]);
        assert_eq!(tag_batch_cached(&texts), first);
    }

    /// A stored entry whose text differs from the query (a hash collision) must
    /// be recomputed, never returned as-is. Poison the memo at a known key with
    /// the wrong text and confirm the lookup falls through to a fresh tag.
    #[test]
    fn tag_batch_cached_recomputes_on_hash_collision() {
        let query = "collision sentinel — the real window text";
        let key = fnv1a64(query);
        let poison = vec![("WRONG".to_string(), "PER".to_string(), 0usize, 5usize)];
        ner_memo()
            .write()
            .unwrap()
            .insert(key, ("a different window".to_string(), poison.clone()));
        let out = tag_batch_cached(&[query]);
        assert_eq!(out.len(), 1);
        assert_ne!(out[0], poison, "collided entry must not be returned");
    }
}
