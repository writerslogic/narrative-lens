//! Optional pinned-model parity probe. Weights and emitted vectors stay in an
//! ignored target cache; the Python companion records compact comparison evidence.
use narrative_lens::onnx::{self, checked, model::ModelManifest};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::Read;
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Cases {
    texts: Vec<String>,
    batches: Vec<Vec<usize>>,
    over_limit_text: String,
}

fn read_bounded(path: &Path, maximum: u64) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(maximum + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum {
        return Err(std::io::Error::other("probe input exceeds size limit"));
    }
    Ok(bytes)
}

fn versions(name: &str) -> Vec<String> {
    include_str!("../Cargo.lock")
        .split("[[package]]")
        .filter(|part| {
            part.lines()
                .any(|line| line == format!("name = \"{name}\""))
        })
        .filter_map(|part| {
            part.lines().find_map(|line| {
                line.strip_prefix("version = \"")
                    .map(|v| v.trim_end_matches('"').to_string())
            })
        })
        .collect()
}

fn runtime() -> Value {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let build = ort::info();
        let version = unsafe {
            let base = ort::sys::OrtGetApiBase();
            assert!(!base.is_null());
            let text = ((*base).GetVersionString)();
            assert!(!text.is_null());
            std::ffi::CStr::from_ptr(text)
                .to_string_lossy()
                .into_owned()
        };
        json!({"runtime":"native ONNX Runtime", "buildInfo":build, "runtimeVersion":version,"ortCrateVersions":versions("ort"),"tokenizersLockVersions":versions("tokenizers")})
    }
    #[cfg(target_arch = "wasm32")]
    {
        json!({"runtime":"actual wasm32-wasip1 tract under Node WASI", "tractOnnxVersions":versions("tract-onnx"),"tokenizersLockVersions":versions("tokenizers")})
    }
}

#[cfg(target_arch = "wasm32")]
fn tract_load_diagnostic(bytes: &[u8]) -> Option<String> {
    use tract_onnx::prelude::*;
    let mut cursor = std::io::Cursor::new(bytes);
    tract_onnx::onnx()
        .model_for_read(&mut cursor)
        .and_then(|model| model.into_optimized())
        .and_then(|model| model.into_runnable())
        .err()
        .map(|error| format!("{error:#}"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 || !["upstream", "adapted"].contains(&args[2].as_str()) {
        return Err(
            "usage: component_real_model <ignored-target-cache-dir> upstream|adapted".into(),
        );
    }
    let cache = Path::new(&args[1]);
    let variant = &args[2];
    let model = read_bounded(&cache.join("model.onnx"), 120 * 1024 * 1024)?;
    let tokenizer_bytes = read_bounded(
        &cache.join(format!("tokenizer.{variant}.json")),
        4 * 1024 * 1024,
    )?;
    let manifest: ModelManifest = serde_json::from_slice(&read_bounded(
        &cache.join(format!("manifest.{variant}.json")),
        16 * 1024,
    )?)?;
    if manifest.model_sha256 != "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452" {
        return Err("probe only supports the pinned official MiniLM export".into());
    }
    let cases: Cases =
        serde_json::from_slice(&read_bounded(&cache.join("cases.json"), 2 * 1024 * 1024)?)?;
    let started = Instant::now();
    let loaded = onnx::set_model_bytes_verified(
        checked::ModelKind::Embedding,
        &model,
        &tokenizer_bytes,
        &manifest,
    );
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut result = runtime();
    result["variant"] = json!(variant);
    result["loadMs"] = json!(load_ms);
    let identity = match loaded {
        Ok(identity) => identity,
        Err(failure) => {
            result["status"] = json!("load_failed");
            #[cfg(target_arch = "wasm32")]
            if failure.code == "model_load_failed" {
                result["diagnostic"] = json!(tract_load_diagnostic(&model));
            }
            result["failure"] = json!(failure);
            println!("{}", serde_json::to_string(&result)?);
            return Ok(());
        }
    };
    result["identity"] = json!(identity);
    let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes).map_err(|error| error.to_string())?;
    let mut batches = Vec::new();
    for indices in cases.batches {
        let texts: Vec<&str> = indices
            .iter()
            .map(|&index| {
                cases
                    .texts
                    .get(index)
                    .map(String::as_str)
                    .ok_or("invalid probe text index")
            })
            .collect::<Result<_, _>>()?;
        let encodings = texts
            .iter()
            .map(|text| tokenizer.encode(*text, true))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        let length = encodings
            .iter()
            .map(|e| e.len())
            .max()
            .ok_or("empty probe batch")?;
        let pad = |values: &[u32], padding: u32| {
            let mut output = values.to_vec();
            output.resize(length, padding);
            output
        };
        let ids: Vec<_> = encodings
            .iter()
            .map(|e| pad(e.get_ids(), manifest.padding_token_id))
            .collect();
        let masks: Vec<_> = encodings
            .iter()
            .map(|e| pad(e.get_attention_mask(), 0))
            .collect();
        let types: Vec<_> = encodings.iter().map(|e| pad(e.get_type_ids(), 0)).collect();
        let started = Instant::now();
        match checked::embeddings(onnx::backend(), &texts) {
            Ok(vectors) => batches.push(json!({"indices":indices,"ids":ids,"masks":masks,"types":types,"vectors":vectors,"runtimeMs":started.elapsed().as_secs_f64()*1000.0})),
            Err(failure) => {
                result["status"] = json!("inference_failed");
                result["failure"] = json!(failure);
                result["failedIndices"] = json!(indices);
                result["completedBatchCount"] = json!(batches.len());
                println!("{}", serde_json::to_string(&result)?);
                return Ok(());
            }
        }
    }
    result["overLimitFailure"] =
        json!(checked::embeddings(onnx::backend(), &[&cases.over_limit_text]).err());
    result["status"] = json!("ok");
    result["batches"] = json!(batches);
    result["inputObservationScope"] = json!(
        "IDs/masks/types reconstructed by the same crate tokenizer; production checked embeddings tokenize independently, and pooled-vector parity checks the end-to-end path."
    );
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
