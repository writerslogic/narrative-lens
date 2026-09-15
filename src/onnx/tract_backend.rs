//! Pure-Rust ([`tract`]) backend for [`InferenceBackend`], used on `wasm32`.
//!
//! `ort` (ONNX Runtime) officially dropped the `wasm32` target, so the wasm
//! build runs inference through `tract-onnx`, a pure-Rust CPU runtime, and
//! tokenizes with the `unstable_wasm` build of HuggingFace `tokenizers` (the
//! native `onig` C regex does not compile for wasm; that feature swaps in a
//! pure-Rust regex).
//!
//! `wasm32-unknown-unknown` has no filesystem, so the model-search paths the
//! native backend uses never resolve. Instead the host (which *can* fetch over
//! the network) supplies the ONNX model bytes and the `tokenizer.json` bytes via
//! [`set_sbert_model_bytes`] / [`set_ner_model_bytes`] before the first call.
//! Until then — or if a load fails — the backend degrades to empty output,
//! exactly the historical missing-model contract; nothing ever panics.
//!
//! Everything after the raw tensor (masked mean-pooling, BIO span decoding) is
//! the runtime-agnostic [`super::decode`] code the native backend also uses, so
//! a wasm embedding/tag is bit-for-bit identical to the native one.

use std::sync::OnceLock;

use tokenizers::Tokenizer;
use tract_onnx::prelude::*;

use super::decode::{argmax_labels, decode_entities, masked_mean_pool, softmax};
use super::InferenceBackend;

type RunnableModel = TypedRunnableModel<TypedModel>;

/// A loaded model: the optimized tract plan, its declared input names (so inputs
/// are fed in the model's own order, not a guessed one), and the tokenizer.
struct TractModel {
    model: RunnableModel,
    input_names: Vec<String>,
    tokenizer: Tokenizer,
}

struct TractBackend;

static BACKEND: TractBackend = TractBackend;

/// Host-supplied `(model_onnx, tokenizer_json)` bytes, set before first use.
static SBERT_BYTES: OnceLock<(Vec<u8>, Vec<u8>)> = OnceLock::new();
static NER_BYTES: OnceLock<(Vec<u8>, Vec<u8>)> = OnceLock::new();
static EMOTION_BYTES: OnceLock<(Vec<u8>, Vec<u8>)> = OnceLock::new();
static NLI_BYTES: OnceLock<(Vec<u8>, Vec<u8>)> = OnceLock::new();

pub fn backend() -> &'static dyn InferenceBackend {
    &BACKEND
}

/// Setting a model directory is a no-op on wasm: there is no filesystem to read
/// model files from. Retained for signature parity with the native backend; the
/// host provides bytes through [`set_sbert_model_bytes`] instead.
pub fn set_sbert_model_directory(_path: &str) {}

/// See [`set_sbert_model_directory`]: no-op on wasm.
pub fn set_ner_model_directory(_path: &str) {}

/// See [`set_sbert_model_directory`]: no-op on wasm.
pub fn set_emotion_model_directory(_path: &str) {}

/// See [`set_sbert_model_directory`]: no-op on wasm.
pub fn set_nli_model_directory(_path: &str) {}

/// Provide the SBERT model + tokenizer bytes. Must be called before the first
/// embedding; no-op once a load has been attempted (the `OnceLock` is set).
pub fn set_sbert_model_bytes(model: &[u8], tokenizer: &[u8]) {
    let _ = SBERT_BYTES.set((model.to_vec(), tokenizer.to_vec()));
}

/// Provide the NER model + tokenizer bytes. Must be called before the first NER
/// call; no-op once a load has been attempted.
pub fn set_ner_model_bytes(model: &[u8], tokenizer: &[u8]) {
    let _ = NER_BYTES.set((model.to_vec(), tokenizer.to_vec()));
}

/// Provide the emotion model + tokenizer bytes. Must be called before the first
/// classification; no-op once a load has been attempted.
pub fn set_emotion_model_bytes(model: &[u8], tokenizer: &[u8]) {
    let _ = EMOTION_BYTES.set((model.to_vec(), tokenizer.to_vec()));
}

/// Provide the NLI model + tokenizer bytes. Must be called before the first pair
/// classification; no-op once a load has been attempted.
pub fn set_nli_model_bytes(model: &[u8], tokenizer: &[u8]) {
    let _ = NLI_BYTES.set((model.to_vec(), tokenizer.to_vec()));
}

/// Parse and optimize an ONNX model from raw bytes with the pure-Rust `tract`
/// runtime, pairing it with the tokenizer parsed from `tokenizer_bytes`. Returns
/// `None` on any parse/optimize/tokenizer failure (graceful degrade).
fn load_model(model_bytes: &[u8], tokenizer_bytes: &[u8]) -> Option<TractModel> {
    let mut reader = model_bytes;
    let typed = tract_onnx::onnx()
        .model_for_read(&mut reader)
        .and_then(|m| m.into_optimized())
        .ok()?;
    // Capture the model's declared input names so we feed tensors in its order.
    let input_names: Vec<String> = typed
        .inputs
        .iter()
        .map(|outlet| typed.node(outlet.node).name.clone())
        .collect();
    let model = typed.into_runnable().ok()?;
    let tokenizer = Tokenizer::from_bytes(tokenizer_bytes).ok()?;
    Some(TractModel {
        model,
        input_names,
        tokenizer,
    })
}

fn sbert_model() -> Option<&'static TractModel> {
    static MODEL: OnceLock<Option<TractModel>> = OnceLock::new();
    MODEL
        .get_or_init(|| SBERT_BYTES.get().and_then(|(m, t)| load_model(m, t)))
        .as_ref()
}

fn ner_model() -> Option<&'static TractModel> {
    static MODEL: OnceLock<Option<TractModel>> = OnceLock::new();
    MODEL
        .get_or_init(|| NER_BYTES.get().and_then(|(m, t)| load_model(m, t)))
        .as_ref()
}

fn emotion_model() -> Option<&'static TractModel> {
    static MODEL: OnceLock<Option<TractModel>> = OnceLock::new();
    MODEL
        .get_or_init(|| EMOTION_BYTES.get().and_then(|(m, t)| load_model(m, t)))
        .as_ref()
}

fn nli_model() -> Option<&'static TractModel> {
    static MODEL: OnceLock<Option<TractModel>> = OnceLock::new();
    MODEL
        .get_or_init(|| NLI_BYTES.get().and_then(|(m, t)| load_model(m, t)))
        .as_ref()
}

/// A row's tokenization: BERT input ids / attention mask / token-type ids, plus
/// the char offsets NER needs to map tokens back to byte spans.
struct Row {
    ids: Vec<i64>,
    mask: Vec<i64>,
    type_ids: Vec<i64>,
    offsets: Vec<(usize, usize)>,
}

/// Tokenize every text, returning the per-row tensors and the batch's max token
/// length. `None` on any tokenizer error (graceful degrade, like the native
/// path). `want_offsets` is false for embedding (offsets are NER-only).
fn tokenize(model: &TractModel, texts: &[&str], want_offsets: bool) -> Option<(Vec<Row>, usize)> {
    let mut rows: Vec<Row> = Vec::with_capacity(texts.len());
    let mut max_len = 0usize;
    for text in texts {
        let encoding = model.tokenizer.encode(*text, true).ok()?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();
        let offsets = if want_offsets {
            encoding.get_offsets().to_vec()
        } else {
            Vec::new()
        };
        max_len = max_len.max(ids.len());
        rows.push(Row {
            ids,
            mask,
            type_ids,
            offsets,
        });
    }
    Some((rows, max_len))
}

/// Flatten rows into padded `[batch, max_len]` row-major buffers (pad value 0;
/// the zero attention mask makes padding inert), then build the tract input
/// tensors in the model's declared input order. `None` on any shape error.
fn build_inputs(model: &TractModel, rows: &[Row], max_len: usize) -> Option<TVec<TValue>> {
    let batch = rows.len();
    let mut ids_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut mask_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut type_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    for row in rows {
        let len = row.ids.len();
        ids_flat.extend_from_slice(&row.ids);
        mask_flat.extend_from_slice(&row.mask);
        type_flat.extend_from_slice(&row.type_ids);
        for _ in len..max_len {
            ids_flat.push(0);
            mask_flat.push(0);
            type_flat.push(0);
        }
    }

    let shape = [batch, max_len];
    let ids_t = Tensor::from_shape(&shape, &ids_flat).ok()?;
    let mask_t = Tensor::from_shape(&shape, &mask_flat).ok()?;
    let type_t = Tensor::from_shape(&shape, &type_flat).ok()?;

    // Feed inputs in the model's own order, matched by HF's standard names; a
    // model declaring only input_ids + attention_mask never sees token_type_ids.
    let mut inputs: TVec<TValue> = tvec!();
    for name in &model.input_names {
        let t = match name.as_str() {
            "attention_mask" => mask_t.clone(),
            "token_type_ids" => type_t.clone(),
            _ => ids_t.clone(),
        };
        inputs.push(t.into());
    }
    Some(inputs)
}

/// Run the model and return its first output as `(last_dim, flat_f32)`,
/// validating a 3-D `[batch, max_len, last_dim]` shape. `None` on any error.
fn run_3d(model: &TractModel, inputs: TVec<TValue>) -> Option<(usize, Vec<f32>)> {
    let outputs = model.model.run(inputs).ok()?;
    let view = outputs.first()?.to_array_view::<f32>().ok()?;
    if view.shape().len() != 3 {
        return None;
    }
    let last_dim = view.shape()[2];
    let data: Vec<f32> = view.iter().copied().collect();
    Some((last_dim, data))
}

/// Run the model and return its first output as `(num_classes, flat_f32)`,
/// validating a 2-D `[batch, num_classes]` shape (sequence classification).
/// `None` on any error.
fn run_2d(model: &TractModel, inputs: TVec<TValue>) -> Option<(usize, Vec<f32>)> {
    let outputs = model.model.run(inputs).ok()?;
    let view = outputs.first()?.to_array_view::<f32>().ok()?;
    if view.shape().len() != 2 {
        return None;
    }
    let last_dim = view.shape()[1];
    let data: Vec<f32> = view.iter().copied().collect();
    Some((last_dim, data))
}

fn embed_inner(texts: &[&str]) -> Vec<Vec<f64>> {
    if texts.is_empty() {
        return Vec::new();
    }
    let empty = || vec![Vec::new(); texts.len()];

    let model = match sbert_model() {
        Some(m) => m,
        None => return empty(),
    };
    let (rows, max_len) = match tokenize(model, texts, false) {
        Some(r) => r,
        None => return empty(),
    };
    // Per-row attention masks padded to max_len, for masked pooling.
    let row_masks: Vec<Vec<i64>> = rows
        .iter()
        .map(|r| {
            let mut m = r.mask.clone();
            m.resize(max_len, 0);
            m
        })
        .collect();

    let inputs = match build_inputs(model, &rows, max_len) {
        Some(i) => i,
        None => return empty(),
    };
    let (hidden_dim, data) = match run_3d(model, inputs) {
        Some(o) => o,
        None => return empty(),
    };
    masked_mean_pool(&data, &row_masks, max_len, hidden_dim)
}

fn tag_inner(texts: &[&str]) -> Vec<Vec<(String, String, usize, usize)>> {
    if texts.is_empty() {
        return Vec::new();
    }
    let empty = || vec![Vec::new(); texts.len()];

    let model = match ner_model() {
        Some(m) => m,
        None => return empty(),
    };
    let (rows, max_len) = match tokenize(model, texts, true) {
        Some(r) => r,
        None => return empty(),
    };
    let inputs = match build_inputs(model, &rows, max_len) {
        Some(i) => i,
        None => return empty(),
    };
    let (num_labels, logits) = match run_3d(model, inputs) {
        Some(o) => o,
        None => return empty(),
    };

    let mut out = Vec::with_capacity(rows.len());
    for (b, (row, text)) in rows.iter().zip(texts.iter()).enumerate() {
        let predicted = argmax_labels(&logits, b, max_len, num_labels, row.offsets.len());
        out.push(decode_entities(&predicted, &row.offsets, text));
    }
    out
}

/// Batched sequence classification (emotion): tokenize, pad, one run, then
/// softmax each `[num_classes]` row. Mirrors [`embed_inner`] but consumes the
/// 2-D classification head via [`run_2d`]. Graceful degrade to empty on any
/// missing-model / tokenizer / shape error.
fn classify_inner(texts: &[&str]) -> Vec<Vec<f64>> {
    if texts.is_empty() {
        return Vec::new();
    }
    let empty = || vec![Vec::new(); texts.len()];

    let model = match emotion_model() {
        Some(m) => m,
        None => return empty(),
    };
    let (rows, max_len) = match tokenize(model, texts, false) {
        Some(r) => r,
        None => return empty(),
    };
    let inputs = match build_inputs(model, &rows, max_len) {
        Some(i) => i,
        None => return empty(),
    };
    let (num_classes, logits) = match run_2d(model, inputs) {
        Some(o) => o,
        None => return empty(),
    };
    (0..rows.len())
        .map(|b| softmax(&logits[b * num_classes..(b + 1) * num_classes]))
        .collect()
}

/// Batched sequence-PAIR classification (NLI): pair-encode each `(premise,
/// hypothesis)`, one run, softmax each `[num_classes]` row. Mirrors
/// [`classify_inner`] but builds its rows from pair encodings and consumes the 2-D
/// head via [`run_2d`]. Graceful degrade to empty on any missing-model / tokenizer
/// / shape error.
fn classify_pairs_inner(pairs: &[(&str, &str)]) -> Vec<Vec<f64>> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let empty = || vec![Vec::new(); pairs.len()];

    let model = match nli_model() {
        Some(m) => m,
        None => return empty(),
    };
    let mut rows: Vec<Row> = Vec::with_capacity(pairs.len());
    let mut max_len = 0usize;
    for &(premise, hypothesis) in pairs {
        let encoding = match model.tokenizer.encode((premise, hypothesis), true) {
            Ok(e) => e,
            Err(_) => return empty(),
        };
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();
        max_len = max_len.max(ids.len());
        rows.push(Row {
            ids,
            mask,
            type_ids,
            offsets: Vec::new(),
        });
    }
    let inputs = match build_inputs(model, &rows, max_len) {
        Some(i) => i,
        None => return empty(),
    };
    let (num_classes, logits) = match run_2d(model, inputs) {
        Some(o) => o,
        None => return empty(),
    };
    (0..rows.len())
        .map(|b| softmax(&logits[b * num_classes..(b + 1) * num_classes]))
        .collect()
}

impl InferenceBackend for TractBackend {
    fn embed(&self, texts: &[&str]) -> Vec<Vec<f64>> {
        embed_inner(texts)
    }

    fn tag(&self, text: &str) -> Vec<(String, String, usize, usize)> {
        tag_inner(&[text]).into_iter().next().unwrap_or_default()
    }

    fn tag_batch(&self, texts: &[&str]) -> Vec<Vec<(String, String, usize, usize)>> {
        tag_inner(texts)
    }

    fn classify(&self, texts: &[&str]) -> Vec<Vec<f64>> {
        classify_inner(texts)
    }

    fn classify_pairs(&self, pairs: &[(&str, &str)]) -> Vec<Vec<f64>> {
        classify_pairs_inner(pairs)
    }
}
