//! Native ONNX Runtime ([`ort`]) backend for [`InferenceBackend`].
//!
//! This is the runtime emathy-core has always used; the inference logic here is
//! the code that previously lived inline in `sbert.rs` and `ner.rs`, now behind
//! the [`InferenceBackend`] trait. Legacy methods retain degrade-to-empty behavior;
//! checked methods preserve load, compatibility, and execution failures.

use ort::session::Session;
use ort::value::Tensor;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use tokenizers::Tokenizer;

use super::InferenceBackend;
use super::checked::{ModelFailure, ModelKind, NarrativeRequest, NarrativeWindow};
use super::decode::{decode_entities, masked_mean_pool};
use super::model::{ModelIdentity, ModelManifest, ModelMetadata, NARRATIVE_OUTPUT_NAMES};
use super::narrative;

struct OnnxModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    metadata: ModelMetadata,
}

#[derive(Default)]
struct ModelSlot {
    directory: Option<PathBuf>,
    generation: u64,
    loaded: Option<Result<Arc<OnnxModel>, ModelFailure>>,
}

struct OrtBackend;
static BACKEND: OrtBackend = OrtBackend;
static SBERT_MODEL: OnceLock<Mutex<ModelSlot>> = OnceLock::new();
static NER_MODEL: OnceLock<Mutex<ModelSlot>> = OnceLock::new();
static EMOTION_MODEL: OnceLock<Mutex<ModelSlot>> = OnceLock::new();
static NLI_MODEL: OnceLock<Mutex<ModelSlot>> = OnceLock::new();
static NARRATIVE_MODEL: OnceLock<Mutex<ModelSlot>> = OnceLock::new();
static ENVIRONMENT: OnceLock<Result<ort::environment::Environment, ModelFailure>> = OnceLock::new();

pub fn backend() -> &'static dyn InferenceBackend {
    &BACKEND
}

fn set_directory(
    slot: &OnceLock<Mutex<ModelSlot>>,
    path: &str,
    cache_generation: &std::sync::atomic::AtomicU64,
) {
    let directory = if path.trim().is_empty() || path.len() > 4096 || path.contains('\0') {
        Err(ModelFailure::incompatible("model_directory_invalid"))
    } else {
        let path = PathBuf::from(path);
        if path.is_absolute() {
            Ok(path)
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(path))
                .map_err(|_| ModelFailure::execution("model_directory_unresolved"))
        }
    };
    let previous = {
        let mut state = slot
            .get_or_init(Default::default)
            .lock()
            .expect("model configuration lock poisoned");
        state.generation = state.generation.wrapping_add(1);
        let previous = state.loaded.take();
        match directory {
            Ok(directory) => state.directory = Some(directory),
            Err(error) => {
                state.directory = None;
                state.loaded = Some(Err(error));
            }
        }
        cache_generation.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        previous
    };
    drop(previous);
}

pub fn set_sbert_model_directory(path: &str) {
    set_directory(&SBERT_MODEL, path, &super::MODEL_GENERATION);
}
pub fn set_ner_model_directory(path: &str) {
    set_directory(&NER_MODEL, path, &super::MODEL_GENERATION);
}
pub fn set_emotion_model_directory(path: &str) {
    set_directory(&EMOTION_MODEL, path, &super::MODEL_GENERATION);
}
pub fn set_nli_model_directory(path: &str) {
    set_directory(&NLI_MODEL, path, &super::MODEL_GENERATION);
}

fn load_model(directory: &std::path::Path, subdir: &str) -> Result<Arc<OnnxModel>, ModelFailure> {
    let model = directory.join(subdir).join("model.onnx");
    let tokenizer = directory.join(subdir).join("tokenizer.json");
    for (path, maximum) in [
        (&model, 1024 * 1024 * 1024u64),
        (&tokenizer, 64 * 1024 * 1024u64),
    ] {
        let metadata = std::fs::metadata(path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                ModelFailure::unavailable("model_files_missing")
            } else {
                ModelFailure::execution("model_files_unreadable")
            }
        })?;
        if !metadata.is_file() || metadata.len() > maximum {
            return Err(ModelFailure::incompatible("model_file_size_or_type"));
        }
    }
    let tokenizer = Tokenizer::from_file(tokenizer)
        .map_err(|_| ModelFailure::incompatible("tokenizer_invalid"))?;
    let environment = ENVIRONMENT
        .get_or_init(|| {
            ort::init()
                .with_telemetry(false)
                .build()
                .map_err(|_| ModelFailure::execution("onnx_environment_failed"))
        })
        .as_ref()
        .map_err(Clone::clone)?;
    let session = Session::builder(environment)
        .and_then(|mut builder| builder.commit_from_file(model))
        .map_err(|_| ModelFailure::incompatible("model_load_failed"))?;
    Ok(Arc::new(OnnxModel {
        session: Mutex::new(session),
        tokenizer,
        metadata: ModelMetadata::unverified(kind_for_subdir(subdir)),
    }))
}

fn kind_for_subdir(subdir: &str) -> ModelKind {
    match subdir {
        "sbert-onnx" => ModelKind::Embedding,
        "ner-onnx" => ModelKind::Ner,
        "emotion-onnx" => ModelKind::Emotion,
        "nli-onnx" => ModelKind::Nli,
        "narrative-multitask-onnx" => ModelKind::NarrativeMultitask,
        _ => unreachable!("model subdirectory is internal"),
    }
}

fn slot_and_subdir(kind: ModelKind) -> (&'static OnceLock<Mutex<ModelSlot>>, &'static str) {
    match kind {
        ModelKind::Embedding => (&SBERT_MODEL, "sbert-onnx"),
        ModelKind::Ner => (&NER_MODEL, "ner-onnx"),
        ModelKind::Emotion => (&EMOTION_MODEL, "emotion-onnx"),
        ModelKind::Nli => (&NLI_MODEL, "nli-onnx"),
        ModelKind::NarrativeMultitask => (&NARRATIVE_MODEL, "narrative-multitask-onnx"),
    }
}

fn read_bounded(path: &std::path::Path, maximum: u64) -> Result<Vec<u8>, ModelFailure> {
    use std::io::Read;
    let file = std::fs::File::open(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ModelFailure::unavailable("model_files_missing")
        } else {
            ModelFailure::execution("model_files_unreadable")
        }
    })?;
    let metadata = file
        .metadata()
        .map_err(|_| ModelFailure::execution("model_files_unreadable"))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > maximum {
        return Err(ModelFailure::incompatible("model_file_size_or_type"));
    }
    let mut bytes = Vec::new();
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| ModelFailure::execution("model_files_unreadable"))?;
    if bytes.len() as u64 > maximum {
        return Err(ModelFailure::incompatible("model_file_size_or_type"));
    }
    Ok(bytes)
}

fn validate_session(session: &Session, metadata: &ModelMetadata) -> Result<(), ModelFailure> {
    use ort::value::{TensorElementType, ValueType};
    metadata.validate_input_names(session.inputs().iter().map(|input| input.name()))?;
    for input in session.inputs() {
        let rank = metadata.input_rank(input.name());
        if !matches!(input.dtype(), ValueType::Tensor { ty: TensorElementType::Int64, shape, .. } if shape.len() == rank && shape.iter().all(|d| *d == -1 || *d > 0))
        {
            return Err(ModelFailure::incompatible("model_profile_input_tensor"));
        }
    }
    let expected = metadata.output_shapes();
    if session.outputs().len() != expected.len()
        || (metadata.identity.kind == ModelKind::NarrativeMultitask
            && session
                .outputs()
                .iter()
                .map(|output| output.name())
                .ne(NARRATIVE_OUTPUT_NAMES))
    {
        return Err(ModelFailure::incompatible("model_profile_outputs"));
    }
    for (output, expected) in session.outputs().iter().zip(expected) {
        let ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape,
            ..
        } = output.dtype()
        else {
            return Err(ModelFailure::incompatible("model_profile_output_tensor"));
        };
        if shape.len() != expected.len()
            || shape.iter().zip(expected).any(|(actual, expected)| {
                (*actual != -1 && *actual <= 0) || expected.is_some_and(|n| *actual != n as i64)
            })
        {
            return Err(ModelFailure::incompatible("model_profile_output_shape"));
        }
    }
    Ok(())
}

fn load_verified(
    kind: ModelKind,
    model: &[u8],
    tokenizer: &[u8],
    manifest: &ModelManifest,
) -> Result<Arc<OnnxModel>, ModelFailure> {
    let metadata = ModelMetadata::verify(kind, manifest, model, tokenizer)?;
    let tokenizer = Tokenizer::from_bytes(tokenizer)
        .map_err(|_| ModelFailure::incompatible("tokenizer_invalid"))?;
    metadata.validate_tokenizer(&tokenizer)?;
    let environment = ENVIRONMENT
        .get_or_init(|| {
            ort::init()
                .with_telemetry(false)
                .build()
                .map_err(|_| ModelFailure::execution("onnx_environment_failed"))
        })
        .as_ref()
        .map_err(Clone::clone)?;
    let session = Session::builder(environment)
        .and_then(|mut builder| builder.commit_from_memory(model))
        .map_err(|_| ModelFailure::incompatible("model_load_failed"))?;
    validate_session(&session, &metadata)?;
    Ok(Arc::new(OnnxModel {
        session: Mutex::new(session),
        tokenizer,
        metadata,
    }))
}

fn publish_verified(
    kind: ModelKind,
    result: Result<Arc<OnnxModel>, ModelFailure>,
) -> Result<ModelIdentity, ModelFailure> {
    let identity = result
        .as_ref()
        .map(|m| m.metadata.identity.clone())
        .map_err(Clone::clone);
    let previous = {
        let mut state = slot_and_subdir(kind)
            .0
            .get_or_init(Default::default)
            .lock()
            .map_err(|_| ModelFailure::execution("model_configuration_lock_failed"))?;
        state.generation = state.generation.wrapping_add(1);
        state.directory = None;
        let previous = state.loaded.replace(result);
        super::MODEL_GENERATION.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        previous
    };
    drop(previous);
    identity
}

pub fn set_model_bytes_verified(
    kind: ModelKind,
    model: &[u8],
    tokenizer: &[u8],
    manifest: &ModelManifest,
) -> Result<ModelIdentity, ModelFailure> {
    publish_verified(kind, load_verified(kind, model, tokenizer, manifest))
}

pub fn set_model_directory_verified(
    kind: ModelKind,
    directory: &str,
    manifest: &ModelManifest,
) -> Result<ModelIdentity, ModelFailure> {
    let loaded = (|| {
        manifest.validate(kind)?;
        if directory.trim().is_empty() || directory.len() > 4096 || directory.contains('\0') {
            return Err(ModelFailure::incompatible("model_directory_invalid"));
        }
        let path = PathBuf::from(directory).join(slot_and_subdir(kind).1);
        let model = read_bounded(&path.join("model.onnx"), 1024 * 1024 * 1024)?;
        let tokenizer = read_bounded(&path.join("tokenizer.json"), 64 * 1024 * 1024)?;
        load_verified(kind, &model, &tokenizer, manifest)
    })();
    publish_verified(kind, loaded)
}

fn configured_model(
    slot: &OnceLock<Mutex<ModelSlot>>,
    subdir: &str,
) -> Result<Arc<OnnxModel>, ModelFailure> {
    let slot = slot.get_or_init(Default::default);
    loop {
        let (directory, generation) = {
            let state = slot
                .lock()
                .map_err(|_| ModelFailure::execution("model_configuration_lock_failed"))?;
            if let Some(loaded) = &state.loaded {
                return loaded.clone();
            }
            (
                state
                    .directory
                    .clone()
                    .ok_or_else(|| ModelFailure::unavailable("model_directory_not_configured"))?,
                state.generation,
            )
        };
        let loaded = load_model(&directory, subdir);
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

fn sbert_model() -> Result<Arc<OnnxModel>, ModelFailure> {
    configured_model(&SBERT_MODEL, "sbert-onnx")
}
fn ner_model() -> Result<Arc<OnnxModel>, ModelFailure> {
    configured_model(&NER_MODEL, "ner-onnx")
}
fn emotion_model() -> Result<Arc<OnnxModel>, ModelFailure> {
    configured_model(&EMOTION_MODEL, "emotion-onnx")
}
fn nli_model() -> Result<Arc<OnnxModel>, ModelFailure> {
    configured_model(&NLI_MODEL, "nli-onnx")
}
fn narrative_model() -> Result<Arc<OnnxModel>, ModelFailure> {
    configured_model(&NARRATIVE_MODEL, "narrative-multitask-onnx")
}

/// One session run per 400-token window: the span-index inputs address the
/// window's own hidden states, so windows cannot share a batch.
fn narrative_inner_checked(
    request: &NarrativeRequest<'_>,
) -> Result<Vec<NarrativeWindow>, ModelFailure> {
    let model = narrative_model()?;
    let tokens = narrative::tokenize(&model.tokenizer, request.text)?;
    let plans = narrative::plan_windows(&tokens, request, model.metadata.window_tokens())?;
    let mut windows = Vec::with_capacity(plans.len());
    for plan in &plans {
        let inputs = narrative::window_inputs(&tokens, plan);
        let length = inputs.ids.len();
        model.metadata.validate_tokens(length)?;
        let (m, q) = (inputs.mention_starts.len(), inputs.quote_starts.len());
        let tensor = |shape: Vec<i64>, data: Vec<i64>| {
            Tensor::from_array((shape, data))
                .map_err(|_| ModelFailure::execution("onnx_execution_failed"))
        };
        let wide = vec![1, length as i64];
        let input_ids = tensor(wide.clone(), inputs.ids)?;
        let attention_mask = tensor(wide, vec![1; length])?;
        let mention_starts = tensor(vec![m as i64], inputs.mention_starts)?;
        let mention_ends = tensor(vec![m as i64], inputs.mention_ends)?;
        let quote_starts = tensor(vec![q as i64], inputs.quote_starts)?;
        let quote_ends = tensor(vec![q as i64], inputs.quote_ends)?;
        let (entity, nominal, antecedent, speaker) = {
            let mut session = model
                .session
                .lock()
                .map_err(|_| ModelFailure::execution("onnx_execution_failed"))?;
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => input_ids,
                    "attention_mask" => attention_mask,
                    "mention_starts" => mention_starts,
                    "mention_ends" => mention_ends,
                    "quote_starts" => quote_starts,
                    "quote_ends" => quote_ends,
                ])
                .map_err(|_| ModelFailure::execution("onnx_execution_failed"))?;
            let extract =
                |name: &str, expected: &[Option<usize>]| -> Result<Vec<f32>, ModelFailure> {
                    let (shape, data) = outputs
                        .get(name)
                        .ok_or_else(|| ModelFailure::incompatible("model_profile_outputs"))?
                        .try_extract_tensor::<f32>()
                        .map_err(|_| ModelFailure::execution("onnx_execution_failed"))?;
                    super::checked::tensor_shape(shape, expected, data)?;
                    Ok(data.to_vec())
                };
            (
                extract("entity_logits", &[Some(1), Some(length), Some(9)])?,
                extract("nominal_logits", &[Some(1), Some(length), Some(3)])?,
                extract("coref_scores", &[Some(m), Some(m)])?,
                extract("attribution_scores", &[Some(q), Some(m + 1)])?,
            )
        };
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
/// value to the pooled vector. Runtime kernels may introduce small numerical
/// differences between batch shapes; callers should compare with a tolerance,
/// not a behavior change. Padding choices match a no-pad single run: the real
/// tokens occupy the same positions and the mask zeroes everything after them.
///
/// Graceful degrade is preserved: no model → one empty Vec per text; any
/// tokenizer/session/shape/lock error → all entries empty (same as the old
/// per-text path returned empty for the failing call). Empty input → empty Vec.
fn embed_batch_inner(texts: &[&str]) -> Vec<Vec<f64>> {
    texts
        .chunks(256)
        .flat_map(|chunk| {
            embed_batch_inner_checked(chunk).unwrap_or_else(|_| vec![Vec::new(); chunk.len()])
        })
        .collect()
}

fn embed_batch_inner_checked(texts: &[&str]) -> Result<Vec<Vec<f64>>, ModelFailure> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let model = sbert_model()?;

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
            Err(_) => return Err(ModelFailure::execution("tokenization_failed")),
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
        });
    }

    let batch = rows.len();
    super::checked::input_elements(batch, max_len)?;

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
            ids_flat.push(model.metadata.padding_id());
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
            None => return Err(ModelFailure::execution("onnx_execution_failed")),
        };

    // Extract token embeddings: shape [batch, max_len, hidden_dim]. The session
    // lock + outputs are scoped to this block so both release before pooling.
    let (hidden_dim, data) = {
        let mut session = match model.session.lock() {
            Ok(s) => s,
            Err(_) => return Err(ModelFailure::execution("onnx_execution_failed")),
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
            Err(_) => return Err(ModelFailure::execution("onnx_execution_failed")),
        };
        let first_output = match outputs.values().next() {
            Some(v) => v,
            None => return Err(ModelFailure::execution("onnx_execution_failed")),
        };
        let (out_shape, data) = match first_output.try_extract_tensor::<f32>() {
            Ok(pair) => pair,
            Err(_) => return Err(ModelFailure::execution("onnx_execution_failed")),
        };
        super::checked::tensor_shape(out_shape, &[Some(batch), Some(max_len), None], data)?;
        let hidden_dim = out_shape[2] as usize;
        (hidden_dim, data.to_vec())
    };

    // Masked mean-pool each row independently over the `[batch, max_len,
    // hidden_dim]` token-embedding tensor (shared with the wasm backend).
    Ok(masked_mean_pool(&data, &row_masks, max_len, hidden_dim))
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
fn tag_batch_inner(texts: &[&str]) -> Vec<super::NerSpans> {
    texts
        .chunks(256)
        .flat_map(|chunk| {
            tag_batch_inner_checked(chunk).unwrap_or_else(|_| vec![Vec::new(); chunk.len()])
        })
        .collect()
}

fn tag_batch_inner_checked(texts: &[&str]) -> Result<Vec<super::NerSpans>, ModelFailure> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let model = ner_model()?;

    struct Row {
        ids: Vec<i64>,
        mask: Vec<i64>,
        type_ids: Vec<i64>,
        offsets: Vec<(usize, usize)>,
        words: Vec<Option<u32>>,
    }
    let mut rows: Vec<Row> = Vec::with_capacity(texts.len());
    let mut max_len = 0usize;
    for text in texts {
        let encoding = match model.tokenizer.encode(*text, true) {
            Ok(e) => e,
            Err(_) => return Err(ModelFailure::execution("tokenization_failed")),
        };
        model.metadata.validate_tokens(encoding.len())?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();
        let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();
        if offsets.iter().any(|(a, b)| {
            a > b || !text.is_char_boundary(*a) || !text.is_char_boundary(*b) || *b > text.len()
        }) {
            return Err(ModelFailure::incompatible("tokenizer_invalid_offsets"));
        }
        let words = encoding.get_word_ids().to_vec();
        max_len = max_len.max(ids.len());
        rows.push(Row {
            ids,
            mask,
            type_ids,
            offsets,
            words,
        });
    }

    let batch = rows.len();
    super::checked::input_elements(batch, max_len)?;

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
            ids_flat.push(model.metadata.padding_id());
            mask_flat.push(0);
            type_flat.push(0);
        }
    }

    let (input_ids, attention_mask, token_type_ids) =
        match build_inputs(batch, max_len, ids_flat, mask_flat, type_flat) {
            Some(t) => t,
            None => return Err(ModelFailure::execution("onnx_execution_failed")),
        };

    // Session lock + outputs scoped to this block so both release before decode.
    let (num_labels, logits) = {
        let mut session = match model.session.lock() {
            Ok(s) => s,
            Err(_) => return Err(ModelFailure::execution("onnx_execution_failed")),
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
            Err(_) => return Err(ModelFailure::execution("onnx_execution_failed")),
        };
        let first_output = match outputs.values().next() {
            Some(v) => v,
            None => return Err(ModelFailure::execution("onnx_execution_failed")),
        };
        let (out_shape, logits_data) = match first_output.try_extract_tensor::<f32>() {
            Ok(pair) => pair,
            Err(_) => return Err(ModelFailure::execution("onnx_execution_failed")),
        };
        super::checked::tensor_shape(
            out_shape,
            &[Some(batch), Some(max_len), Some(9)],
            logits_data,
        )?;
        (out_shape[2] as usize, logits_data.to_vec())
    };

    // Decode each row independently over its real tokens. Logits are row-major
    // `[batch, max_len, num_labels]`; row `b`'s token `i` starts at
    // `(b * max_len + i) * num_labels`.
    let mut out = Vec::with_capacity(batch);
    for (b, (row, text)) in rows.iter().zip(texts.iter()).enumerate() {
        let predicted =
            model
                .metadata
                .ner_labels(&logits, b, max_len, num_labels, row.offsets.len());
        out.push(decode_entities(&predicted, &row.offsets, &row.words, text));
    }
    Ok(out)
}

/// Batched sequence classification (emotion). Mirrors [`embed_batch_inner`] —
/// tokenize, pad to the batch's max token length, one padded ONNX run — but the
/// classification head emits a 2-D `[batch, num_classes]` logit tensor (no
/// per-token output, no pooling), so each row is softmaxed into a class
/// distribution. Same graceful degrade: missing model / tokenizer / shape error
/// → one empty Vec per text; empty input → empty Vec.
fn classify_batch_inner(texts: &[&str]) -> Vec<Vec<f64>> {
    texts
        .chunks(256)
        .flat_map(|chunk| {
            classify_batch_inner_checked(chunk).unwrap_or_else(|_| vec![Vec::new(); chunk.len()])
        })
        .collect()
}

fn classify_batch_inner_checked(texts: &[&str]) -> Result<Vec<Vec<f64>>, ModelFailure> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let model = emotion_model()?;

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
            Err(_) => return Err(ModelFailure::execution("tokenization_failed")),
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
        });
    }

    let batch = rows.len();
    super::checked::input_elements(batch, max_len)?;

    let mut ids_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut mask_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut type_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    for row in &rows {
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

    let (input_ids, attention_mask, token_type_ids) =
        match build_inputs(batch, max_len, ids_flat, mask_flat, type_flat) {
            Some(t) => t,
            None => return Err(ModelFailure::execution("onnx_execution_failed")),
        };

    // Sequence-classification head: the output tensor is `[batch, num_classes]`.
    let (num_classes, logits) = {
        let mut session = match model.session.lock() {
            Ok(s) => s,
            Err(_) => return Err(ModelFailure::execution("onnx_execution_failed")),
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
            Err(_) => return Err(ModelFailure::execution("onnx_execution_failed")),
        };
        let first_output = match outputs.values().next() {
            Some(v) => v,
            None => return Err(ModelFailure::execution("onnx_execution_failed")),
        };
        let (out_shape, data) = match first_output.try_extract_tensor::<f32>() {
            Ok(pair) => pair,
            Err(_) => return Err(ModelFailure::execution("onnx_execution_failed")),
        };
        super::checked::tensor_shape(out_shape, &[Some(batch), None], data)?;
        (out_shape[1] as usize, data.to_vec())
    };

    (0..batch)
        .map(|b| {
            model
                .metadata
                .probabilities(&logits[b * num_classes..(b + 1) * num_classes])
        })
        .collect()
}

/// Batched sequence-PAIR classification (NLI). Identical to [`classify_batch_inner`]
/// except each row is a `(premise, hypothesis)` pair encoded together (premise
/// `[SEP]` hypothesis) via the tokenizer's pair mode, and it reads the NLI model.
/// Output is a 2-D `[batch, num_classes]` logit tensor, softmaxed per row. Same
/// graceful degrade: missing model / tokenizer / shape error → one empty Vec per
/// pair; empty input → empty Vec.
fn classify_pairs_inner(pairs: &[(&str, &str)]) -> Vec<Vec<f64>> {
    pairs
        .chunks(256)
        .flat_map(|chunk| {
            classify_pairs_inner_checked(chunk).unwrap_or_else(|_| vec![Vec::new(); chunk.len()])
        })
        .collect()
}

fn classify_pairs_inner_checked(pairs: &[(&str, &str)]) -> Result<Vec<Vec<f64>>, ModelFailure> {
    if pairs.is_empty() {
        return Ok(Vec::new());
    }

    let model = nli_model()?;

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
            Err(_) => return Err(ModelFailure::execution("tokenization_failed")),
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
        });
    }

    let batch = rows.len();
    super::checked::input_elements(batch, max_len)?;

    let mut ids_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut mask_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    let mut type_flat: Vec<i64> = Vec::with_capacity(batch * max_len);
    for row in &rows {
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

    let (input_ids, attention_mask, token_type_ids) =
        match build_inputs(batch, max_len, ids_flat, mask_flat, type_flat) {
            Some(t) => t,
            None => return Err(ModelFailure::execution("onnx_execution_failed")),
        };

    let (num_classes, logits) = {
        let mut session = match model.session.lock() {
            Ok(s) => s,
            Err(_) => return Err(ModelFailure::execution("onnx_execution_failed")),
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
            Err(_) => return Err(ModelFailure::execution("onnx_execution_failed")),
        };
        let first_output = match outputs.values().next() {
            Some(v) => v,
            None => return Err(ModelFailure::execution("onnx_execution_failed")),
        };
        let (out_shape, data) = match first_output.try_extract_tensor::<f32>() {
            Ok(pair) => pair,
            Err(_) => return Err(ModelFailure::execution("onnx_execution_failed")),
        };
        super::checked::tensor_shape(out_shape, &[Some(batch), None], data)?;
        (out_shape[1] as usize, data.to_vec())
    };

    (0..batch)
        .map(|b| {
            model
                .metadata
                .probabilities(&logits[b * num_classes..(b + 1) * num_classes])
        })
        .collect()
}

impl InferenceBackend for OrtBackend {
    fn model_identity(&self, kind: ModelKind) -> Result<ModelIdentity, ModelFailure> {
        let (slot, subdir) = slot_and_subdir(kind);
        Ok(configured_model(slot, subdir)?.metadata.identity.clone())
    }

    fn embed_checked(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>, ModelFailure> {
        super::checked::validate_texts(texts, 256)?;
        embed_batch_inner_checked(texts)
    }
    fn tag_batch_checked(&self, texts: &[&str]) -> Result<Vec<super::NerSpans>, ModelFailure> {
        super::checked::validate_texts(texts, 256)?;
        tag_batch_inner_checked(texts)
    }
    fn classify_checked(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>, ModelFailure> {
        super::checked::validate_texts(texts, 256)?;
        classify_batch_inner_checked(texts)
    }
    fn classify_pairs_checked(
        &self,
        pairs: &[(&str, &str)],
    ) -> Result<Vec<Vec<f64>>, ModelFailure> {
        super::checked::validate_texts(
            &pairs.iter().flat_map(|(a, b)| [*a, *b]).collect::<Vec<_>>(),
            512,
        )?;
        classify_pairs_inner_checked(pairs)
    }
    fn narrative_checked(
        &self,
        request: &NarrativeRequest<'_>,
    ) -> Result<Vec<NarrativeWindow>, ModelFailure> {
        narrative_inner_checked(request)
    }

    fn check_model(&self, kind: ModelKind) -> Result<(), ModelFailure> {
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
        embed_batch_inner(texts)
    }

    fn tag(&self, text: &str) -> Vec<(String, String, usize, usize)> {
        tag_text(text)
    }

    fn tag_batch(&self, texts: &[&str]) -> Vec<super::NerSpans> {
        tag_batch_inner(texts)
    }

    fn classify(&self, texts: &[&str]) -> Vec<Vec<f64>> {
        classify_batch_inner(texts)
    }

    fn classify_pairs(&self, pairs: &[(&str, &str)]) -> Vec<Vec<f64>> {
        classify_pairs_inner(pairs)
    }
}

#[cfg(test)]
mod configuration_tests {
    use super::*;

    #[test]
    fn configuration_can_recover_and_replace_a_loaded_model() {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "narrative-model-lifecycle-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let model_dir = root.join("nli-onnx");
        let slot = OnceLock::new();
        let cache_generation = std::sync::atomic::AtomicU64::new(0);
        assert!(matches!(
            configured_model(&slot, "nli-onnx"),
            Err(ModelFailure {
                kind: super::super::checked::FailureKind::Unavailable,
                ..
            })
        ));
        set_directory(&slot, root.to_str().unwrap(), &cache_generation);
        assert_eq!(
            cache_generation.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert!(configured_model(&slot, "nli-onnx").is_err());
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::copy(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/models/valid-nli.onnx"),
            model_dir.join("model.onnx"),
        )
        .unwrap();
        let vocabulary = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab([("[UNK]".into(), 0)].into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        Tokenizer::new(vocabulary)
            .save(model_dir.join("tokenizer.json"), false)
            .unwrap();
        set_directory(&slot, root.to_str().unwrap(), &cache_generation);
        let loaded = configured_model(&slot, "nli-onnx");
        assert!(loaded.is_ok(), "{:?}", loaded.err());
        set_directory(
            &slot,
            root.join("missing").to_str().unwrap(),
            &cache_generation,
        );
        assert!(configured_model(&slot, "nli-onnx").is_err());
        set_directory(&slot, root.to_str().unwrap(), &cache_generation);
        let loaded = configured_model(&slot, "nli-onnx");
        assert!(loaded.is_ok(), "{:?}", loaded.err());
        set_directory(&slot, "", &cache_generation);
        assert!(matches!(
            configured_model(&slot, "nli-onnx"),
            Err(ModelFailure {
                kind: super::super::checked::FailureKind::Incompatible,
                ..
            })
        ));
        std::fs::remove_dir_all(root).unwrap();
    }
}
