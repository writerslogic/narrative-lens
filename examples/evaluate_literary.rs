use narrative_lens::{
    continuity::{coreference::CoreferenceResolver, ner},
    craft::dialogue,
};
use serde::Deserialize;
use serde_json::json;
use std::{fs::File, io::Read, path::Path};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Case {
    id: String,
    split: String,
    entity_text: String,
    quotation_text: String,
    supplied_cast: Vec<String>,
}

fn bounded_file(path: &Path, limit: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err("benchmark file exceeds its byte limit".into());
    }
    Ok(bytes)
}

#[cfg(feature = "onnx")]
fn load_ner_cache(path: &Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    use narrative_lens::onnx::{self, checked::ModelKind, model::ModelManifest};
    let manifest =
        ModelManifest::from_json(&bounded_file(&path.join("manifest.json"), 16 * 1024)?)?;
    let model = bounded_file(&path.join("model.onnx"), 70 * 1024 * 1024)?;
    let tokenizer = bounded_file(&path.join("tokenizer.adapted.json"), 2 * 1024 * 1024)?;
    let identity = onnx::set_model_bytes_verified(ModelKind::Ner, &model, &tokenizer, &manifest)?;
    Ok(serde_json::to_value(identity)?)
}

#[cfg(not(feature = "onnx"))]
fn load_ner_cache(_: &Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    Err("--ner-cache requires the onnx feature".into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: evaluate_literary <prepared-input.json> [--ner-cache PATH] [--split development|final_holdout] [--finalize]")?;
    let mut cache = None;
    let mut split = "development".to_string();
    let mut finalize = false;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--ner-cache" => cache = Some(args.next().ok_or("missing NER cache path")?),
            "--split" => split = args.next().ok_or("missing split")?,
            "--finalize" => finalize = true,
            _ => return Err("unknown benchmark argument".into()),
        }
    }
    if !matches!(split.as_str(), "development" | "final_holdout")
        || split == "final_holdout" && !finalize
    {
        return Err("invalid split or missing finalization authorization flag".into());
    }
    let bytes = bounded_file(Path::new(&path), 16 * 1024 * 1024)?;
    let cases: Vec<Case> = serde_json::from_slice(&bytes)?;
    if cases.len() > 1000 {
        return Err("benchmark exceeds1000 works".into());
    }
    let identity = cache
        .as_deref()
        .map(Path::new)
        .map(load_ner_cache)
        .transpose()?;
    for case in cases.into_iter().filter(|case| case.split == split) {
        if case.entity_text.len() > 512 * 1024
            || case.quotation_text.len() > 512 * 1024
            || case.supplied_cast.len() > 256
        {
            return Err("benchmark case exceeds text/cast limits".into());
        }
        let started = std::time::Instant::now();
        let prediction = std::panic::catch_unwind(
            || -> narrative_lens::Result<serde_json::Value> {
                let extracted =
                    ner::ner_extract_persons_checked(std::slice::from_ref(&case.entity_text))?;
                if identity.is_some()
                    && let Some(failure) = &extracted.model_failure
                {
                    return Err(narrative_lens::Error::InvalidInput(format!(
                        "requested NER model failed:{}",
                        failure.code
                    )));
                }
                let persons = extracted.persons;
                let spans = extracted
                    .model_entities
                    .into_iter()
                    .next()
                    .unwrap_or_default();
                let detected: Vec<String> = persons
                    .iter()
                    .take(256)
                    .map(|(name, _)| name.clone())
                    .collect();
                let supplied: Vec<_> = case.supplied_cast.iter().map(String::as_str).collect();
                // The automatic coreference cast is every surface the extractor
                // observed, not only canonical forms; the resolver groups the
                // aliases of one character itself. Surfaces the resolver rejects
                // as names fall back to the canonical inventory.
                let mut surfaces = detected.clone();
                let mut seen: std::collections::HashSet<String> =
                    surfaces.iter().map(|name| name.to_lowercase()).collect();
                for mention in &extracted.mentions {
                    let surface = mention
                        .surface
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ");
                    if surfaces.len() < 256 && seen.insert(surface.to_lowercase()) {
                        surfaces.push(surface);
                    }
                }
                let discovered: Vec<_> = surfaces.iter().map(String::as_str).collect();
                let canonical: Vec<_> = detected.iter().map(String::as_str).collect();
                let resolver = CoreferenceResolver::new();
                let detected_coreference = resolver
                    .resolve_checked(&case.entity_text, &discovered, None)
                    .or_else(|_| resolver.resolve_checked(&case.entity_text, &canonical, None))?;
                Ok(json!({
                    "personNames":persons,
                    "nerSpans":spans,
                    "personExtraction": {"availability":extracted.availability,"diagnostics":extracted.diagnostics,"modelIdentity":extracted.model_identity,"modelFailure":extracted.model_failure,"modelGeneration":extracted.model_generation,"nerWindows":extracted.ner_windows,"unassessedMentions":extracted.unassessed_mentions},
                    "coreferenceSuppliedCast":resolver.resolve_checked(&case.entity_text,&supplied,None)?,
                    "coreferenceDetectedCast":detected_coreference,
                    "dialogueSuppliedCast":dialogue::extract_dialogue(&case.quotation_text,0,&case.supplied_cast),
                    "dialogueDetectedCast":dialogue::extract_dialogue(&case.quotation_text,0,&detected),
                    "dialogueExplicitSuppliedCast":dialogue::extract_dialogue_checked(&case.quotation_text,0,&case.supplied_cast,&dialogue::DialogueExtractionOptions::default())?,
                    "dialogueExplicitDetectedCast":dialogue::extract_dialogue_checked(&case.quotation_text,0,&detected,&dialogue::DialogueExtractionOptions::default())?,
                    "detectedCastTruncated":persons.len()>256,
                }))
            },
        );
        let mut output = match prediction {
            Ok(Ok(value)) => value,
            Ok(Err(error)) => {
                json!({"failure":"checked_analysis_failed","diagnostic":error.to_string()})
            }
            Err(_) => json!({"failure":"analysis_panicked"}),
        };
        output["id"] = json!(case.id);
        output["split"] = json!(case.split);
        output["runtimeMs"] = json!(started.elapsed().as_secs_f64() * 1000.0);
        output["modelMode"] = json!(if identity.is_some() {
            "verified local NER model plus heuristic person inventory"
        } else {
            "no supplied trained models; heuristic person inventory"
        });
        output["modelIdentity"] = json!(identity);
        println!("{}", serde_json::to_string(&output)?);
    }
    Ok(())
}
