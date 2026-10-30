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
//! [`set_sbert_model_bytes`] / [`set_ner_model_bytes`]; setters replace or retry models.
//! Until then — or if a load fails — the backend degrades to empty output,
//! exactly the historical missing-model contract; nothing ever panics.
//!
//! Everything after the raw tensor (masked mean-pooling, BIO span decoding) is
//! the runtime-agnostic [`super::decode`] code the native backend also uses, so
//! backend numerical parity remains a tolerance-based runtime check.

use super::InferenceBackend;
use super::checked::{ModelFailure, ModelKind, NarrativeRequest, NarrativeWindow};
use super::decode::{decode_entities, masked_mean_pool};
use super::model::{ModelIdentity, ModelManifest, ModelMetadata, NARRATIVE_OUTPUT_NAMES};
use super::narrative;
use std::sync::{Arc, Mutex, OnceLock};
use tokenizers::Tokenizer;
use tract_onnx::prelude::*;

type RunnableModel = TypedRunnableModel;
struct TractModel {
    model: Arc<RunnableModel>,
    input_names: Vec<String>,
    output_names: Vec<String>,
    tokenizer: Tokenizer,
    metadata: ModelMetadata,
}
#[derive(Default)]
struct ModelSlot {
    bytes: Option<Arc<(Vec<u8>, Vec<u8>)>>,
    generation: u64,
    loaded: Option<Result<Arc<TractModel>, ModelFailure>>,
}
struct TractBackend;
static BACKEND: TractBackend = TractBackend;
static SBERT_MODEL: OnceLock<Mutex<ModelSlot>> = OnceLock::new();
static NER_MODEL: OnceLock<Mutex<ModelSlot>> = OnceLock::new();
static EMOTION_MODEL: OnceLock<Mutex<ModelSlot>> = OnceLock::new();
static NLI_MODEL: OnceLock<Mutex<ModelSlot>> = OnceLock::new();
static NARRATIVE_MODEL: OnceLock<Mutex<ModelSlot>> = OnceLock::new();
pub fn backend() -> &'static dyn InferenceBackend {
    &BACKEND
}
pub fn set_sbert_model_directory(_path: &str) {}
pub fn set_ner_model_directory(_path: &str) {}
pub fn set_emotion_model_directory(_path: &str) {}
pub fn set_nli_model_directory(_path: &str) {}

fn set_bytes(slot: &OnceLock<Mutex<ModelSlot>>, model: &[u8], tokenizer: &[u8]) {
    let bytes = if model.len() > 1024 * 1024 * 1024 || tokenizer.len() > 64 * 1024 * 1024 {
        Err(ModelFailure::incompatible("model_file_size_or_type"))
    } else {
        Ok(Arc::new((model.to_vec(), tokenizer.to_vec())))
    };
    let previous = {
        let mut state = slot
            .get_or_init(Default::default)
            .lock()
            .expect("model configuration lock poisoned");
        state.generation = state.generation.wrapping_add(1);
        let previous = state.loaded.take();
        match bytes {
            Ok(bytes) => state.bytes = Some(bytes),
            Err(error) => {
                state.bytes = None;
                state.loaded = Some(Err(error));
            }
        }
        previous
    };
    drop(previous);
}
pub fn set_sbert_model_bytes(model: &[u8], tokenizer: &[u8]) {
    set_bytes(&SBERT_MODEL, model, tokenizer);
}
pub fn set_ner_model_bytes(model: &[u8], tokenizer: &[u8]) {
    set_bytes(&NER_MODEL, model, tokenizer);
}
pub fn set_emotion_model_bytes(model: &[u8], tokenizer: &[u8]) {
    set_bytes(&EMOTION_MODEL, model, tokenizer);
}
pub fn set_nli_model_bytes(model: &[u8], tokenizer: &[u8]) {
    set_bytes(&NLI_MODEL, model, tokenizer);
}

fn validate_model(model: &TypedModel, metadata: &ModelMetadata) -> Result<(), ModelFailure> {
    metadata.validate_input_names(
        model
            .inputs
            .iter()
            .map(|outlet| model.node(outlet.node).name.as_str()),
    )?;
    for (i, outlet) in model.inputs.iter().enumerate() {
        let rank = metadata.input_rank(&model.node(outlet.node).name);
        let fact = model
            .input_fact(i)
            .map_err(|_| ModelFailure::incompatible("model_profile_input_tensor"))?;
        if fact.datum_type != DatumType::I64
            || fact.shape.rank() != rank
            || fact
                .shape
                .iter()
                .any(|dim| dim.as_i64().is_some_and(|n| n <= 0))
        {
            return Err(ModelFailure::incompatible("model_profile_input_tensor"));
        }
    }
    let expected = metadata.output_shapes();
    if model.outputs.len() != expected.len() {
        return Err(ModelFailure::incompatible("model_profile_outputs"));
    }
    for (i, expected) in expected.into_iter().enumerate() {
        let fact = model
            .output_fact(i)
            .map_err(|_| ModelFailure::incompatible("model_profile_output_tensor"))?;
        if fact.datum_type != DatumType::F32 {
            return Err(ModelFailure::incompatible("model_profile_output_tensor"));
        }
        if fact.shape.rank() != expected.len()
            || fact.shape.iter().zip(expected).any(|(actual, expected)| {
                actual.as_i64().is_some_and(|n| n <= 0)
                    || expected.is_some_and(|n| actual.as_i64() != Some(n as i64))
            })
        {
            return Err(ModelFailure::incompatible("model_profile_output_shape"));
        }
    }
    Ok(())
}

fn load_model(
    model_bytes: &[u8],
    tokenizer_bytes: &[u8],
    metadata: ModelMetadata,
) -> Result<Arc<TractModel>, ModelFailure> {
    let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
        .map_err(|_| ModelFailure::incompatible("tokenizer_invalid"))?;
    metadata.validate_tokenizer(&tokenizer)?;
    let mut reader = model_bytes;
    let typed = tract_onnx::onnx()
        .model_for_read(&mut reader)
        .and_then(|model| model.into_optimized())
        .map_err(|_| ModelFailure::incompatible("model_load_failed"))?;
    if metadata.identity.manifest.is_some() {
        validate_model(&typed, &metadata)?;
    }
    let input_names = typed
        .inputs
        .iter()
        .map(|outlet| typed.node(outlet.node).name.clone())
        .collect();
    let output_names = typed
        .outputs
        .iter()
        .map(|outlet| {
            typed
                .outlet_label(*outlet)
                .map_or_else(|| typed.node(outlet.node).name.clone(), str::to_string)
        })
        .collect();
    let model = typed
        .into_runnable()
        .map_err(|_| ModelFailure::incompatible("model_load_failed"))?;
    Ok(Arc::new(TractModel {
        model,
        input_names,
        output_names,
        tokenizer,
        metadata,
    }))
}
fn configured_model(
    slot: &OnceLock<Mutex<ModelSlot>>,
    kind: ModelKind,
) -> Result<Arc<TractModel>, ModelFailure> {
    let slot = slot.get_or_init(Default::default);
    loop {
        let (bytes, generation) = {
            let state = slot
                .lock()
                .map_err(|_| ModelFailure::execution("model_configuration_lock_failed"))?;
            if let Some(loaded) = &state.loaded {
                return loaded.clone();
            }
            (
                state
                    .bytes
                    .clone()
                    .ok_or_else(|| ModelFailure::unavailable("model_bytes_missing"))?,
                state.generation,
            )
        };
        let loaded = load_model(&bytes.0, &bytes.1, ModelMetadata::unverified(kind));
        let mut state = slot
            .lock()
            .map_err(|_| ModelFailure::execution("model_configuration_lock_failed"))?;
        if state.generation != generation {
            drop(state);
            drop(loaded);
            continue;
        }
        if let Some(existing) = &state.loaded {
            let existing = existing.clone();
            drop(state);
            drop(loaded);
            return existing;
        }
        state.loaded = Some(loaded.clone());
        drop(state);
        return loaded;
    }
}
fn sbert_model() -> Result<Arc<TractModel>, ModelFailure> {
    configured_model(&SBERT_MODEL, ModelKind::Embedding)
}
fn ner_model() -> Result<Arc<TractModel>, ModelFailure> {
    configured_model(&NER_MODEL, ModelKind::Ner)
}
fn emotion_model() -> Result<Arc<TractModel>, ModelFailure> {
    configured_model(&EMOTION_MODEL, ModelKind::Emotion)
}
fn nli_model() -> Result<Arc<TractModel>, ModelFailure> {
    configured_model(&NLI_MODEL, ModelKind::Nli)
}
fn narrative_model() -> Result<Arc<TractModel>, ModelFailure> {
    configured_model(&NARRATIVE_MODEL, ModelKind::NarrativeMultitask)
}

fn slot_for_kind(kind: ModelKind) -> &'static OnceLock<Mutex<ModelSlot>> {
    match kind {
        ModelKind::Embedding => &SBERT_MODEL,
        ModelKind::Ner => &NER_MODEL,
        ModelKind::Emotion => &EMOTION_MODEL,
        ModelKind::Nli => &NLI_MODEL,
        ModelKind::NarrativeMultitask => &NARRATIVE_MODEL,
    }
}

/// One run per 400-token window, inputs in the graph's declared order and
/// outputs located by their ONNX names (falling back to graph order).
fn narrative_inner_checked(
    request: &NarrativeRequest<'_>,
) -> std::result::Result<Vec<NarrativeWindow>, ModelFailure> {
    let model = narrative_model()?;
    let tokens = narrative::tokenize(&model.tokenizer, request.text)?;
    let plans = narrative::plan_windows(&tokens, request, model.metadata.window_tokens())?;
    let mut windows = Vec::with_capacity(plans.len());
    for plan in &plans {
        let inputs = narrative::window_inputs(&tokens, plan);
        let length = inputs.ids.len();
        model.metadata.validate_tokens(length)?;
        let (m, q) = (inputs.mention_starts.len(), inputs.quote_starts.len());
        let tensor = |shape: &[usize], data: &[i64]| {
            Tensor::from_shape(shape, data)
                .map_err(|_| ModelFailure::execution("onnx_execution_failed"))
        };
        let mut values: TVec<TValue> = tvec!();
        for name in &model.input_names {
            let value = match name.as_str() {
                "input_ids" => tensor(&[1, length], &inputs.ids)?,
                "attention_mask" => tensor(&[1, length], &vec![1i64; length])?,
                "mention_starts" => tensor(&[m], &inputs.mention_starts)?,
                "mention_ends" => tensor(&[m], &inputs.mention_ends)?,
                "quote_starts" => tensor(&[q], &inputs.quote_starts)?,
                "quote_ends" => tensor(&[q], &inputs.quote_ends)?,
                _ => return Err(ModelFailure::incompatible("model_profile_inputs")),
            };
            values.push(value.into());
        }
        let outputs = model
            .model
            .run(values)
            .map_err(|_| ModelFailure::execution("onnx_execution_failed"))?;
        if outputs.len() != NARRATIVE_OUTPUT_NAMES.len() {
            return Err(ModelFailure::incompatible("model_profile_outputs"));
        }
        let named = NARRATIVE_OUTPUT_NAMES
            .iter()
            .all(|name| model.output_names.iter().any(|actual| actual == name));
        let extract = |name: &str,
                       expected: &[Option<usize>]|
         -> std::result::Result<Vec<f32>, ModelFailure> {
            let index = if named {
                model
                    .output_names
                    .iter()
                    .position(|actual| actual == name)
                    .expect("every narrative output name is present")
            } else {
                NARRATIVE_OUTPUT_NAMES
                    .iter()
                    .position(|candidate| *candidate == name)
                    .expect("known narrative output")
            };
            let view = outputs[index]
                .to_plain_array_view::<f32>()
                .map_err(|_| ModelFailure::incompatible("model_output_not_float32"))?;
            let shape = view.shape().iter().map(|n| *n as i64).collect::<Vec<_>>();
            if view.len() > 16 * 1024 * 1024 {
                return Err(ModelFailure::incompatible("model_output_allocation_limit"));
            }
            let data = view.iter().copied().collect::<Vec<_>>();
            super::checked::tensor_shape(&shape, expected, &data)?;
            Ok(data)
        };
        let entity = extract("entity_logits", &[Some(1), Some(length), Some(9)])?;
        let nominal = extract("nominal_logits", &[Some(1), Some(length), Some(3)])?;
        let antecedent = extract("coref_scores", &[Some(m), Some(m)])?;
        let speaker = extract("attribution_scores", &[Some(q), Some(m + 1)])?;
        windows.push(narrative::decode_window(
            request.text,
            &tokens,
            plan,
            &model.metadata,
            narrative::WindowOutputs {
                entity: &entity,
                nominal: &nominal,
                antecedent: &antecedent,
                speaker: &speaker,
            },
        ));
    }
    Ok(windows)
}

pub fn set_model_bytes_verified(
    kind: ModelKind,
    model: &[u8],
    tokenizer: &[u8],
    manifest: &ModelManifest,
) -> Result<ModelIdentity, ModelFailure> {
    let loaded = ModelMetadata::verify(kind, manifest, model, tokenizer)
        .and_then(|metadata| load_model(model, tokenizer, metadata));
    let identity = loaded
        .as_ref()
        .map(|m| m.metadata.identity.clone())
        .map_err(Clone::clone);
    let previous = {
        let mut state = slot_for_kind(kind)
            .get_or_init(Default::default)
            .lock()
            .map_err(|_| ModelFailure::execution("model_configuration_lock_failed"))?;
        state.generation = state.generation.wrapping_add(1);
        let bytes = state.bytes.take();
        let previous = state.loaded.replace(loaded);
        super::MODEL_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        (bytes, previous)
    };
    drop(previous);
    identity
}

/// A row's tokenization: BERT input ids / attention mask / token-type ids, plus
/// the char offsets NER needs to map tokens back to byte spans.
struct Row {
    ids: Vec<i64>,
    mask: Vec<i64>,
    type_ids: Vec<i64>,
    offsets: Vec<(usize, usize)>,
    words: Vec<Option<u32>>,
}

/// Tokenize every text, returning the per-row tensors and the batch's max token
/// length. `None` on any tokenizer error (graceful degrade, like the native
/// path). `want_offsets` is false for embedding (offsets are NER-only).
fn tokenize(
    model: &TractModel,
    texts: &[&str],
    want_offsets: bool,
) -> Result<(Vec<Row>, usize), ModelFailure> {
    let mut rows: Vec<Row> = Vec::with_capacity(texts.len());
    let mut max_len = 0usize;
    for text in texts {
        let encoding = model
            .tokenizer
            .encode(*text, true)
            .map_err(|_| ModelFailure::execution("tokenization_failed"))?;
        model.metadata.validate_tokens(encoding.len())?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();
        let (offsets, words) = if want_offsets {
            (
                encoding.get_offsets().to_vec(),
                encoding.get_word_ids().to_vec(),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        max_len = max_len.max(ids.len());
        rows.push(Row {
            ids,
            mask,
            type_ids,
            offsets,
            words,
        });
    }
    Ok((rows, max_len))
}

/// Flatten rows into padded `[batch, max_len]` row-major buffers (pad value 0;
/// the zero attention mask makes padding inert), then build the tract input
/// tensors in the model's declared input order. `None` on any shape error.
fn build_inputs(model: &TractModel, rows: &[Row], max_len: usize) -> Option<TVec<TValue>> {
    let batch = rows.len();
    super::checked::input_elements(batch, max_len).ok()?;
    let mut ids_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut mask_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut type_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    for row in rows {
        let len = row.ids.len();
        ids_flat.extend_from_slice(&row.ids);
        mask_flat.extend_from_slice(&row.mask);
        type_flat.extend_from_slice(&row.type_ids);
        for _ in len..max_len {
            ids_flat.push(model.metadata.padding_id());
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
            "input_ids" => ids_t.clone(),
            _ => return None,
        };
        inputs.push(t.into());
    }
    Some(inputs)
}

/// Run the model and return its first output as `(last_dim, flat_f32)`,
/// validating a 3-D `[batch, max_len, last_dim]` shape. `None` on any error.
fn run_3d(
    model: &TractModel,
    inputs: TVec<TValue>,
    batch: usize,
    tokens: usize,
    channels: Option<usize>,
) -> std::result::Result<(usize, Vec<f32>), ModelFailure> {
    let outputs = model
        .model
        .run(inputs)
        .map_err(|_| ModelFailure::execution("onnx_execution_failed"))?;
    let view = outputs
        .first()
        .ok_or_else(|| ModelFailure::incompatible("model_has_no_outputs"))?
        .to_plain_array_view::<f32>()
        .map_err(|_| ModelFailure::incompatible("model_output_not_float32"))?;
    let shape = view.shape().iter().map(|n| *n as i64).collect::<Vec<_>>();
    if view.len() > 16 * 1024 * 1024 {
        return Err(ModelFailure::incompatible("model_output_allocation_limit"));
    }
    let data = view.iter().copied().collect::<Vec<_>>();
    super::checked::tensor_shape(&shape, &[Some(batch), Some(tokens), channels], &data)?;
    Ok((shape[2] as usize, data))
}
fn run_2d(
    model: &TractModel,
    inputs: TVec<TValue>,
    batch: usize,
) -> std::result::Result<(usize, Vec<f32>), ModelFailure> {
    let outputs = model
        .model
        .run(inputs)
        .map_err(|_| ModelFailure::execution("onnx_execution_failed"))?;
    let view = outputs
        .first()
        .ok_or_else(|| ModelFailure::incompatible("model_has_no_outputs"))?
        .to_plain_array_view::<f32>()
        .map_err(|_| ModelFailure::incompatible("model_output_not_float32"))?;
    let shape = view.shape().iter().map(|n| *n as i64).collect::<Vec<_>>();
    if view.len() > 16 * 1024 * 1024 {
        return Err(ModelFailure::incompatible("model_output_allocation_limit"));
    }
    let data = view.iter().copied().collect::<Vec<_>>();
    super::checked::tensor_shape(&shape, &[Some(batch), None], &data)?;
    Ok((shape[1] as usize, data))
}

fn embed_inner(texts: &[&str]) -> Vec<Vec<f64>> {
    texts
        .chunks(256)
        .flat_map(|chunk| {
            embed_inner_checked(chunk).unwrap_or_else(|_| vec![Vec::new(); chunk.len()])
        })
        .collect()
}

fn embed_inner_checked(texts: &[&str]) -> std::result::Result<Vec<Vec<f64>>, ModelFailure> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let model = sbert_model()?;
    let (rows, max_len) = tokenize(&model, texts, false)?;
    // Per-row attention masks padded to max_len, for masked pooling.
    let row_masks: Vec<Vec<i64>> = rows
        .iter()
        .map(|r| {
            let mut m = r.mask.clone();
            m.resize(max_len, 0);
            m
        })
        .collect();

    let inputs = match build_inputs(&model, &rows, max_len) {
        Some(i) => i,
        None => return Err(ModelFailure::execution("model_tokenization_or_input_limit")),
    };
    let (hidden_dim, data) = run_3d(&model, inputs, rows.len(), max_len, None)?;
    Ok(masked_mean_pool(&data, &row_masks, max_len, hidden_dim))
}

fn tag_inner(texts: &[&str]) -> Vec<super::NerSpans> {
    texts
        .chunks(256)
        .flat_map(|chunk| {
            tag_inner_checked(chunk).unwrap_or_else(|_| vec![Vec::new(); chunk.len()])
        })
        .collect()
}

fn tag_inner_checked(texts: &[&str]) -> std::result::Result<Vec<super::NerSpans>, ModelFailure> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let model = ner_model()?;
    let (rows, max_len) = tokenize(&model, texts, true)?;
    let inputs = match build_inputs(&model, &rows, max_len) {
        Some(i) => i,
        None => return Err(ModelFailure::execution("model_tokenization_or_input_limit")),
    };
    let (num_labels, logits) = run_3d(&model, inputs, rows.len(), max_len, Some(9))?;

    let mut out = Vec::with_capacity(rows.len());
    for (b, (row, text)) in rows.iter().zip(texts.iter()).enumerate() {
        if row
            .offsets
            .iter()
            .any(|(a, b)| a > b || text.get(*a..*b).is_none())
        {
            return Err(ModelFailure::incompatible("tokenizer_invalid_offsets"));
        }
        let predicted =
            model
                .metadata
                .ner_labels(&logits, b, max_len, num_labels, row.offsets.len());
        out.push(decode_entities(&predicted, &row.offsets, &row.words, text));
    }
    Ok(out)
}

/// Batched sequence classification (emotion): tokenize, pad, one run, then
/// softmax each `[num_classes]` row. Mirrors [`embed_inner`] but consumes the
/// 2-D classification head via [`run_2d`]. Graceful degrade to empty on any
/// missing-model / tokenizer / shape error.
fn classify_inner(texts: &[&str]) -> Vec<Vec<f64>> {
    texts
        .chunks(256)
        .flat_map(|chunk| {
            classify_inner_checked(chunk).unwrap_or_else(|_| vec![Vec::new(); chunk.len()])
        })
        .collect()
}

fn classify_inner_checked(texts: &[&str]) -> std::result::Result<Vec<Vec<f64>>, ModelFailure> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let model = emotion_model()?;
    let (rows, max_len) = tokenize(&model, texts, false)?;
    let inputs = match build_inputs(&model, &rows, max_len) {
        Some(i) => i,
        None => return Err(ModelFailure::execution("model_tokenization_or_input_limit")),
    };
    let (num_classes, logits) = run_2d(&model, inputs, rows.len())?;
    (0..rows.len())
        .map(|b| {
            model
                .metadata
                .probabilities(&logits[b * num_classes..(b + 1) * num_classes])
        })
        .collect()
}

/// Batched sequence-PAIR classification (NLI): pair-encode each `(premise,
/// hypothesis)`, one run, softmax each `[num_classes]` row. Mirrors
/// [`classify_inner`] but builds its rows from pair encodings and consumes the 2-D
/// head via [`run_2d`]. Graceful degrade to empty on any missing-model / tokenizer
/// / shape error.
fn classify_pairs_inner(pairs: &[(&str, &str)]) -> Vec<Vec<f64>> {
    pairs
        .chunks(256)
        .flat_map(|chunk| {
            classify_pairs_inner_checked(chunk).unwrap_or_else(|_| vec![Vec::new(); chunk.len()])
        })
        .collect()
}

fn classify_pairs_inner_checked(
    pairs: &[(&str, &str)],
) -> std::result::Result<Vec<Vec<f64>>, ModelFailure> {
    if pairs.is_empty() {
        return Ok(Vec::new());
    }

    let model = nli_model()?;
    let mut rows: Vec<Row> = Vec::with_capacity(pairs.len());
    let mut max_len = 0usize;
    for &(premise, hypothesis) in pairs {
        let encoding = match model.tokenizer.encode((premise, hypothesis), true) {
            Ok(e) => e,
            Err(_) => return Err(ModelFailure::execution("model_tokenization_or_input_limit")),
        };
        model.metadata.validate_tokens(encoding.len())?;
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
            words: Vec::new(),
        });
    }
    let inputs = match build_inputs(&model, &rows, max_len) {
        Some(i) => i,
        None => return Err(ModelFailure::execution("model_tokenization_or_input_limit")),
    };
    let (num_classes, logits) = run_2d(&model, inputs, rows.len())?;
    (0..rows.len())
        .map(|b| {
            model
                .metadata
                .probabilities(&logits[b * num_classes..(b + 1) * num_classes])
        })
        .collect()
}

impl InferenceBackend for TractBackend {
    fn model_identity(&self, kind: ModelKind) -> Result<ModelIdentity, ModelFailure> {
        Ok(configured_model(slot_for_kind(kind), kind)?
            .metadata
            .identity
            .clone())
    }

    fn embed_checked(&self, texts: &[&str]) -> std::result::Result<Vec<Vec<f64>>, ModelFailure> {
        super::checked::validate_texts(texts, 256)?;
        embed_inner_checked(texts)
    }
    fn classify_checked(&self, texts: &[&str]) -> std::result::Result<Vec<Vec<f64>>, ModelFailure> {
        super::checked::validate_texts(texts, 256)?;
        classify_inner_checked(texts)
    }
    fn tag_batch_checked(
        &self,
        texts: &[&str],
    ) -> std::result::Result<Vec<super::NerSpans>, ModelFailure> {
        super::checked::validate_texts(texts, 256)?;
        tag_inner_checked(texts)
    }
    fn classify_pairs_checked(
        &self,
        pairs: &[(&str, &str)],
    ) -> std::result::Result<Vec<Vec<f64>>, ModelFailure> {
        super::checked::validate_texts(
            &pairs.iter().flat_map(|(a, b)| [*a, *b]).collect::<Vec<_>>(),
            512,
        )?;
        classify_pairs_inner_checked(pairs)
    }
    fn narrative_checked(
        &self,
        request: &NarrativeRequest<'_>,
    ) -> std::result::Result<Vec<NarrativeWindow>, ModelFailure> {
        narrative_inner_checked(request)
    }

    fn check_model(
        &self,
        kind: super::checked::ModelKind,
    ) -> std::result::Result<(), super::checked::ModelFailure> {
        match kind {
            ModelKind::Embedding => sbert_model(),
            ModelKind::Ner => ner_model(),
            ModelKind::Emotion => emotion_model(),
            ModelKind::Nli => nli_model(),
            ModelKind::NarrativeMultitask => narrative_model(),
        }
        .map(|_| ())
    }

    fn embed(&self, texts: &[&str]) -> Vec<Vec<f64>> {
        embed_inner(texts)
    }

    fn tag(&self, text: &str) -> Vec<(String, String, usize, usize)> {
        tag_inner(&[text]).into_iter().next().unwrap_or_default()
    }

    fn tag_batch(&self, texts: &[&str]) -> Vec<super::NerSpans> {
        tag_inner(texts)
    }

    fn classify(&self, texts: &[&str]) -> Vec<Vec<f64>> {
        classify_inner(texts)
    }

    fn classify_pairs(&self, pairs: &[(&str, &str)]) -> Vec<Vec<f64>> {
        classify_pairs_inner(pairs)
    }
}
