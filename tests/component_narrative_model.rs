//! End-to-end check of the narrative multi-task model on a locally exported
//! artifact. Model weights are never in the repository, so the test only runs
//! when `NARRATIVE_LENS_NARRATIVE_MODEL` names a directory written by
//! `scripts/multitask_train.py export` (`manifest.json`, `model.onnx`,
//! `tokenizer.adapted.json`).
#![cfg(all(feature = "onnx", not(target_arch = "wasm32")))]

use narrative_lens::{
    continuity::{
        coreference::{CoreferenceBasis, CoreferenceResolver},
        narrative::{self, MentionKind},
    },
    craft::dialogue,
    onnx::{self, checked::ModelKind, model::ModelManifest},
};

const TEXT: &str = "Alice Bennet walked into the parlour and sat by the fire. \"Come in, Jane,\" said Alice to her sister. Jane Bennet looked up from her book. \"I will,\" she replied, closing it.";

fn span(needle: &str) -> (usize, usize) {
    let start = TEXT.find(needle).expect("needle present");
    (start, start + needle.len())
}

#[test]
fn local_narrative_model_drives_mentions_coreference_and_speakers() {
    let Some(directory) = std::env::var_os("NARRATIVE_LENS_NARRATIVE_MODEL") else {
        eprintln!("skipping: NARRATIVE_LENS_NARRATIVE_MODEL is not set");
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    let manifest =
        ModelManifest::from_json(&std::fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    let model = std::fs::read(directory.join("model.onnx")).unwrap();
    let tokenizer = std::fs::read(directory.join("tokenizer.adapted.json")).unwrap();
    let identity = onnx::set_model_bytes_verified(
        ModelKind::NarrativeMultitask,
        &model,
        &tokenizer,
        &manifest,
    )
    .unwrap();
    assert!(narrative::available());

    let quotes = [span("\"Come in, Jane,\""), span("\"I will,\"")];
    let analysis = narrative::analyze(TEXT, &quotes).unwrap();
    assert_eq!(analysis.identity, identity);
    assert_eq!(analysis.windows, 1);
    for entity in &analysis.entities {
        assert_eq!(&TEXT[entity.2..entity.3], entity.0);
    }
    let proper: Vec<_> = analysis
        .mentions
        .iter()
        .filter(|mention| mention.kind == MentionKind::Proper)
        .map(|mention| &TEXT[mention.start..mention.end])
        .collect();
    assert!(proper.contains(&"Alice Bennet"), "{proper:?}");
    assert!(
        proper.iter().any(|name| name.starts_with("Jane")),
        "{proper:?}"
    );
    let pronouns: Vec<_> = analysis
        .mentions
        .iter()
        .filter(|mention| mention.kind == MentionKind::Pronoun)
        .collect();
    assert_eq!(pronouns.len(), 3, "her, her, she");
    assert!(
        pronouns
            .iter()
            .all(|mention| (0.0..=1.0).contains(&mention.confidence))
    );
    for (index, mention) in analysis.mentions.iter().enumerate() {
        eprintln!(
            "mention {index}: {:?} {:?} cluster {} antecedent {:?} p {:.3} name {:?}",
            &TEXT[mention.start..mention.end],
            mention.kind,
            mention.cluster,
            mention.antecedent,
            mention.confidence,
            analysis.mention_name(index)
        );
    }
    for (index, quote) in analysis.quotes.iter().enumerate() {
        eprintln!(
            "quote {index}: speaker {:?} p {:.3} name {:?}",
            quote.speaker,
            quote.confidence,
            analysis.speaker_name(index)
        );
    }
    // Which antecedent or speaker the model picks on a toy passage changes
    // between exports (scripts/narrative_model_reference.py prints the
    // onnxruntime decode of this passage for comparison); the contract checked
    // here is that every decision is well-formed and traceable to the model.
    for (index, mention) in analysis.mentions.iter().enumerate() {
        assert!(mention.cluster < analysis.cluster_names.len());
        assert!(
            mention
                .antecedent
                .is_none_or(|antecedent| antecedent < index)
        );
        if mention.kind == MentionKind::Proper {
            assert!(analysis.mention_name(index).is_some());
        }
    }
    assert_eq!(analysis.quotes.len(), 2);
    for (quote, expected) in analysis.quotes.iter().zip(quotes) {
        assert_eq!((quote.start, quote.end), expected);
        assert!((0.0..=1.0).contains(&quote.confidence));
        assert!(
            quote
                .speaker
                .is_none_or(|speaker| speaker < analysis.mentions.len())
        );
    }

    let cast = ["Alice Bennet", "Jane Bennet"];
    let coreference = CoreferenceResolver::new()
        .resolve_checked(TEXT, &cast, None)
        .unwrap();
    assert_eq!(coreference.model_identity.as_ref(), Some(&identity));
    assert_eq!(coreference.pronoun_count, 3);
    assert!(
        coreference
            .mentions
            .iter()
            .all(|mention| mention.range.is_some())
    );
    for mention in coreference
        .mentions
        .iter()
        .filter(|mention| mention.is_pronoun)
    {
        assert!((0.0..=1.0).contains(&mention.confidence));
        if let Some(resolved) = &mention.resolved_to {
            assert_eq!(
                mention.resolution_basis,
                Some(CoreferenceBasis::NarrativeModel)
            );
            assert!(
                cast.contains(&resolved.as_str()) || proper.contains(&resolved.as_str()),
                "{resolved}"
            );
        }
    }
    assert!(
        coreference
            .mentions
            .iter()
            .any(|mention| !mention.is_pronoun)
    );

    let cast: Vec<String> = cast.iter().map(|name| (*name).into()).collect();
    let result = dialogue::extract_dialogue_checked(TEXT, 0, &cast, &Default::default()).unwrap();
    assert_eq!(result.model_identity.as_ref(), Some(&identity));
    assert_eq!(result.lines.len(), 2);
    assert_eq!(
        result.lines[0].line.speaker, "Alice Bennet",
        "explicit tag wins"
    );
    assert_eq!(result.lines[0].line.method, "explicit_tag");
    let second = &result.lines[1].line;
    assert!(
        matches!(second.method.as_str(), "narrative_model" | "unknown"),
        "{second:?}"
    );
    if second.method == "narrative_model" {
        assert!(!second.speaker.is_empty() && second.speaker != "Unknown");
        assert!((0.0..=1.0).contains(&second.confidence));
    }
}
