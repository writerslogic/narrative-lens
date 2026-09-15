//! Native ONNX Runtime ([`ort`]) backend for [`InferenceBackend`].
//!
//! This is the runtime emathy-core has always used; the inference logic here is
//! the code that previously lived inline in `sbert.rs` and `ner.rs`, now behind
//! the [`InferenceBackend`] trait. Behavior is byte-for-byte identical, including
//! the silent degrade-to-empty path when a model or tokenizer is absent.

use ort::session::Session;
use ort::value::Tensor;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use tokenizers::Tokenizer;

use super::decode::{argmax_labels, decode_entities, masked_mean_pool, softmax};
use super::InferenceBackend;

struct OnnxModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

// `Session` is not `Sync`, but every access is serialized through the `Mutex`
// guarding it, so sharing the model across threads is sound.
unsafe impl Sync for OnnxModel {}

struct OrtBackend;

static BACKEND: OrtBackend = OrtBackend;

static SBERT_MODEL: OnceLock<Option<OnnxModel>> = OnceLock::new();
static SBERT_DIR: OnceLock<Option<String>> = OnceLock::new();
static NER_MODEL: OnceLock<Option<OnnxModel>> = OnceLock::new();
static NER_DIR: OnceLock<Option<String>> = OnceLock::new();
static EMOTION_MODEL: OnceLock<Option<OnnxModel>> = OnceLock::new();
static EMOTION_DIR: OnceLock<Option<String>> = OnceLock::new();
static NLI_MODEL: OnceLock<Option<OnnxModel>> = OnceLock::new();
static NLI_DIR: OnceLock<Option<String>> = OnceLock::new();

pub fn backend() -> &'static dyn InferenceBackend {
    &BACKEND
}

pub fn set_sbert_model_directory(path: &str) {
    let _ = SBERT_DIR.set(Some(path.to_string()));
}

pub fn set_ner_model_directory(path: &str) {
    let _ = NER_DIR.set(Some(path.to_string()));
}

pub fn set_emotion_model_directory(path: &str) {
    let _ = EMOTION_DIR.set(Some(path.to_string()));
}

pub fn set_nli_model_directory(path: &str) {
    let _ = NLI_DIR.set(Some(path.to_string()));
}

/// Load a model + tokenizer, searching: custom dir (if set) → `models/` →
/// `../models/`. Returns `None` (graceful degrade) if none load.
fn load_model(dir: &OnceLock<Option<String>>, subdir: &str) -> Option<OnnxModel> {
    let mut model_paths: Vec<PathBuf> = Vec::new();
    let mut tokenizer_paths: Vec<PathBuf> = Vec::new();

    if let Some(Some(d)) = dir.get() {
        model_paths.push(PathBuf::from(d).join(subdir).join("model.onnx"));
        tokenizer_paths.push(PathBuf::from(d).join(subdir).join("tokenizer.json"));
    }

    model_paths.push(format!("models/{subdir}/model.onnx").into());
    model_paths.push(format!("../models/{subdir}/model.onnx").into());
    tokenizer_paths.push(format!("models/{subdir}/tokenizer.json").into());
    tokenizer_paths.push(format!("../models/{subdir}/tokenizer.json").into());

    for (mp, tp) in model_paths.iter().zip(tokenizer_paths.iter()) {
        let session = match Session::builder().and_then(|mut b| b.commit_from_file(mp)) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let tokenizer = match Tokenizer::from_file(tp) {
            Ok(t) => t,
            Err(_) => continue,
        };
        return Some(OnnxModel {
            session: Mutex::new(session),
            tokenizer,
        });
    }
    None
}

fn sbert_model() -> Option<&'static OnnxModel> {
    SBERT_MODEL
        .get_or_init(|| load_model(&SBERT_DIR, "sbert-onnx"))
        .as_ref()
}

fn ner_model() -> Option<&'static OnnxModel> {
    NER_MODEL
        .get_or_init(|| load_model(&NER_DIR, "ner-onnx"))
        .as_ref()
}

fn emotion_model() -> Option<&'static OnnxModel> {
    EMOTION_MODEL
        .get_or_init(|| load_model(&EMOTION_DIR, "emotion-onnx"))
        .as_ref()
}

fn nli_model() -> Option<&'static OnnxModel> {
    NLI_MODEL
        .get_or_init(|| load_model(&NLI_DIR, "nli-onnx"))
        .as_ref()
}

/// Build the three standard BERT input tensors of shape `[batch, seq_len]`.
/// Returns `None` if any tensor fails to construct (graceful degrade).
type TensorTriple = (Tensor<i64>, Tensor<i64>, Tensor<i64>);
fn build_inputs(
    batch: usize,
    seq_len: usize,
    ids: Vec<i64>,
    mask: Vec<i64>,
    type_ids: Vec<i64>,
) -> Option<TensorTriple> {
    let shape = [batch as i64, seq_len as i64];
    let input_ids = Tensor::from_array((shape, ids)).ok()?;
    let attention_mask = Tensor::from_array((shape, mask)).ok()?;
    let token_type_ids = Tensor::from_array((shape, type_ids)).ok()?;
    Some((input_ids, attention_mask, token_type_ids))
}

/// Batched SBERT embedding: tokenize every input, pad each sequence to the
/// batch's max sequence length, run the ONNX session ONCE on `[N, max_len]`
/// tensors, then masked-mean-pool each row using that row's attention mask.
///
/// CORRECTNESS INVARIANT: pooling is masked, so padding tokens (input_ids = 0,
/// attention_mask = 0, token_type_ids = 0) contribute zero weight and zero
/// value to the pooled vector. Each batched row's pooled result is therefore
/// bit-for-bit identical to embedding that text alone — this is a pure speedup,
/// not a behavior change. Padding choices match a no-pad single run: the real
/// tokens occupy the same positions and the mask zeroes everything after them.
///
/// Graceful degrade is preserved: no model → one empty Vec per text; any
/// tokenizer/session/shape/lock error → all entries empty (same as the old
/// per-text path returned empty for the failing call). Empty input → empty Vec.
fn embed_batch_inner(texts: &[&str]) -> Vec<Vec<f64>> {
    if texts.is_empty() {
        return Vec::new();
    }

    let empty = || vec![Vec::new(); texts.len()];

    let model = match sbert_model() {
        Some(m) => m,
        None => return empty(),
    };

    // Tokenize every text; keep per-row ids/mask/type_ids and track max length.
    struct Row {
        ids: Vec<i64>,
        mask: Vec<i64>,
        type_ids: Vec<i64>,
    }
    let mut rows: Vec<Row> = Vec::with_capacity(texts.len());
    let mut max_len = 0usize;
    for text in texts {
        let encoding = match model.tokenizer.encode(*text, true) {
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
        });
    }

    let batch = rows.len();

    // Flatten into padded `[batch, max_len]` row-major buffers. Pad value is 0
    // for all three inputs; the zero attention_mask is what makes padding inert.
    let mut ids_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut mask_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut type_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    // Per-row attention masks, kept for masked pooling after inference.
    let mut row_masks: Vec<Vec<i64>> = Vec::with_capacity(batch);
    for row in &rows {
        let len = row.ids.len();
        ids_flat.extend_from_slice(&row.ids);
        mask_flat.extend_from_slice(&row.mask);
        type_flat.extend_from_slice(&row.type_ids);
        for _ in len..max_len {
            ids_flat.push(0);
            mask_flat.push(0);
            type_flat.push(0);
        }
        let mut padded_mask = row.mask.clone();
        padded_mask.resize(max_len, 0);
        row_masks.push(padded_mask);
    }

    let (input_ids, attention_mask, token_type_ids) =
        match build_inputs(batch, max_len, ids_flat, mask_flat, type_flat) {
            Some(t) => t,
            None => return empty(),
        };

    // Extract token embeddings: shape [batch, max_len, hidden_dim]. The session
    // lock + outputs are scoped to this block so both release before pooling.
    let (hidden_dim, data) = {
        let mut session = match model.session.lock() {
            Ok(s) => s,
            Err(_) => return empty(),
        };
        // This SBERT export (MiniLM-L6-v2) declares only input_ids +
        // attention_mask; passing token_type_ids errors with "Invalid input
        // name". Include it only when the model actually declares that input.
        let wants_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");
        let run = if wants_type_ids {
            session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
            ])
        } else {
            session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
            ])
        };
        let outputs = match run {
            Ok(o) => o,
            Err(_) => return empty(),
        };
        let first_output = match outputs.values().next() {
            Some(v) => v,
            None => return empty(),
        };
        let (out_shape, data) = match first_output.try_extract_tensor::<f32>() {
            Ok(pair) => pair,
            Err(_) => return empty(),
        };
        if out_shape.len() != 3 {
            return empty();
        }
        let hidden_dim = out_shape[2] as usize;
        (hidden_dim, data.to_vec())
    };

    // Masked mean-pool each row independently over the `[batch, max_len,
    // hidden_dim]` token-embedding tensor (shared with the wasm backend).
    masked_mean_pool(&data, &row_masks, max_len, hidden_dim)
}

fn tag_text(text: &str) -> Vec<(String, String, usize, usize)> {
    tag_batch_inner(&[text])
        .into_iter()
        .next()
        .unwrap_or_default()
}

/// Batched NER: tag every text in `texts` with a single padded ONNX run instead
/// of one run per text. Mirrors [`embed_batch_inner`] — pad to the batch's max
/// token length, the zero attention mask makes padding inert — then decode each
/// row independently over its own real tokens. Same graceful-degrade contract:
/// missing model / tokenizer / shape error → one empty Vec per text; empty
/// input → empty Vec.
fn tag_batch_inner(texts: &[&str]) -> Vec<Vec<(String, String, usize, usize)>> {
    if texts.is_empty() {
        return Vec::new();
    }
    let empty = || vec![Vec::new(); texts.len()];

    let model = match ner_model() {
        Some(m) => m,
        None => return empty(),
    };

    struct Row {
        ids: Vec<i64>,
        mask: Vec<i64>,
        type_ids: Vec<i64>,
        offsets: Vec<(usize, usize)>,
    }
    let mut rows: Vec<Row> = Vec::with_capacity(texts.len());
    let mut max_len = 0usize;
    for text in texts {
        let encoding = match model.tokenizer.encode(*text, true) {
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
        let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();
        max_len = max_len.max(ids.len());
        rows.push(Row {
            ids,
            mask,
            type_ids,
            offsets,
        });
    }

    let batch = rows.len();

    // Flatten into padded `[batch, max_len]` buffers (pad value 0; the zero
    // attention mask is what makes the padding inert), matching the embedder.
    let mut ids_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut mask_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut type_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    for row in &rows {
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

    let (input_ids, attention_mask, token_type_ids) =
        match build_inputs(batch, max_len, ids_flat, mask_flat, type_flat) {
            Some(t) => t,
            None => return empty(),
        };

    // Session lock + outputs scoped to this block so both release before decode.
    let (num_labels, logits) = {
        let mut session = match model.session.lock() {
            Ok(s) => s,
            Err(_) => return empty(),
        };
        // Include token_type_ids only when the model declares it (the NER BERT
        // model does; sentence-transformer exports do not — passing an
        // undeclared input errors with "Invalid input name").
        let wants_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");
        let run = if wants_type_ids {
            session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
            ])
        } else {
            session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
            ])
        };
        let outputs = match run {
            Ok(o) => o,
            Err(_) => return empty(),
        };
        let first_output = match outputs.values().next() {
            Some(v) => v,
            None => return empty(),
        };
        let (out_shape, logits_data) = match first_output.try_extract_tensor::<f32>() {
            Ok(pair) => pair,
            Err(_) => return empty(),
        };
        if out_shape.len() != 3 {
            return empty();
        }
        (out_shape[2] as usize, logits_data.to_vec())
    };

    // Decode each row independently over its real tokens. Logits are row-major
    // `[batch, max_len, num_labels]`; row `b`'s token `i` starts at
    // `(b * max_len + i) * num_labels`.
    let mut out = Vec::with_capacity(batch);
    for (b, (row, text)) in rows.iter().zip(texts.iter()).enumerate() {
        let predicted = argmax_labels(&logits, b, max_len, num_labels, row.offsets.len());
        out.push(decode_entities(&predicted, &row.offsets, text));
    }
    out
}

/// Batched sequence classification (emotion). Mirrors [`embed_batch_inner`] —
/// tokenize, pad to the batch's max token length, one padded ONNX run — but the
/// classification head emits a 2-D `[batch, num_classes]` logit tensor (no
/// per-token output, no pooling), so each row is softmaxed into a class
/// distribution. Same graceful degrade: missing model / tokenizer / shape error
/// → one empty Vec per text; empty input → empty Vec.
fn classify_batch_inner(texts: &[&str]) -> Vec<Vec<f64>> {
    if texts.is_empty() {
        return Vec::new();
    }
    let empty = || vec![Vec::new(); texts.len()];

    let model = match emotion_model() {
        Some(m) => m,
        None => return empty(),
    };

    struct Row {
        ids: Vec<i64>,
        mask: Vec<i64>,
        type_ids: Vec<i64>,
    }
    let mut rows: Vec<Row> = Vec::with_capacity(texts.len());
    let mut max_len = 0usize;
    for text in texts {
        let encoding = match model.tokenizer.encode(*text, true) {
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
        });
    }

    let batch = rows.len();

    let mut ids_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut mask_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut type_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    for row in &rows {
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

    let (input_ids, attention_mask, token_type_ids) =
        match build_inputs(batch, max_len, ids_flat, mask_flat, type_flat) {
            Some(t) => t,
            None => return empty(),
        };

    // Sequence-classification head: the output tensor is `[batch, num_classes]`.
    let (num_classes, logits) = {
        let mut session = match model.session.lock() {
            Ok(s) => s,
            Err(_) => return empty(),
        };
        // RoBERTa-family classifiers declare input_ids + attention_mask only;
        // pass token_type_ids only when the model actually declares it.
        let wants_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");
        let run = if wants_type_ids {
            session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
            ])
        } else {
            session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
            ])
        };
        let outputs = match run {
            Ok(o) => o,
            Err(_) => return empty(),
        };
        let first_output = match outputs.values().next() {
            Some(v) => v,
            None => return empty(),
        };
        let (out_shape, data) = match first_output.try_extract_tensor::<f32>() {
            Ok(pair) => pair,
            Err(_) => return empty(),
        };
        if out_shape.len() != 2 {
            return empty();
        }
        (out_shape[1] as usize, data.to_vec())
    };

    (0..batch)
        .map(|b| softmax(&logits[b * num_classes..(b + 1) * num_classes]))
        .collect()
}

/// Batched sequence-PAIR classification (NLI). Identical to [`classify_batch_inner`]
/// except each row is a `(premise, hypothesis)` pair encoded together (premise
/// `[SEP]` hypothesis) via the tokenizer's pair mode, and it reads the NLI model.
/// Output is a 2-D `[batch, num_classes]` logit tensor, softmaxed per row. Same
/// graceful degrade: missing model / tokenizer / shape error → one empty Vec per
/// pair; empty input → empty Vec.
fn classify_pairs_inner(pairs: &[(&str, &str)]) -> Vec<Vec<f64>> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let empty = || vec![Vec::new(); pairs.len()];

    let model = match nli_model() {
        Some(m) => m,
        None => return empty(),
    };

    struct Row {
        ids: Vec<i64>,
        mask: Vec<i64>,
        type_ids: Vec<i64>,
    }
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
        });
    }

    let batch = rows.len();

    let mut ids_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut mask_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut type_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    for row in &rows {
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

    let (input_ids, attention_mask, token_type_ids) =
        match build_inputs(batch, max_len, ids_flat, mask_flat, type_flat) {
            Some(t) => t,
            None => return empty(),
        };

    let (num_classes, logits) = {
        let mut session = match model.session.lock() {
            Ok(s) => s,
            Err(_) => return empty(),
        };
        let wants_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");
        let run = if wants_type_ids {
            session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
            ])
        } else {
            session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
            ])
        };
        let outputs = match run {
            Ok(o) => o,
            Err(_) => return empty(),
        };
        let first_output = match outputs.values().next() {
            Some(v) => v,
            None => return empty(),
        };
        let (out_shape, data) = match first_output.try_extract_tensor::<f32>() {
            Ok(pair) => pair,
            Err(_) => return empty(),
        };
        if out_shape.len() != 2 {
            return empty();
        }
        (out_shape[1] as usize, data.to_vec())
    };

    (0..batch)
        .map(|b| softmax(&logits[b * num_classes..(b + 1) * num_classes]))
        .collect()
}

impl InferenceBackend for OrtBackend {
    fn embed(&self, texts: &[&str]) -> Vec<Vec<f64>> {
        embed_batch_inner(texts)
    }

    fn tag(&self, text: &str) -> Vec<(String, String, usize, usize)> {
        tag_text(text)
    }

    fn tag_batch(&self, texts: &[&str]) -> Vec<Vec<(String, String, usize, usize)>> {
        tag_batch_inner(texts)
    }

    fn classify(&self, texts: &[&str]) -> Vec<Vec<f64>> {
        classify_batch_inner(texts)
    }

    fn classify_pairs(&self, pairs: &[(&str, &str)]) -> Vec<Vec<f64>> {
        classify_pairs_inner(pairs)
    }
}
