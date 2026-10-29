use narrative_lens::{
    continuity::ner::*,
    manuscript::TextRange,
    onnx::{
        self, InferenceBackend, NerSpans,
        checked::{ModelFailure, ModelKind},
    },
    substrate::utils::AnalysisAvailability,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Names {
    calls: AtomicUsize,
}
impl InferenceBackend for Names {
    fn embed(&self, texts: &[&str]) -> Vec<Vec<f64>> {
        vec![Vec::new(); texts.len()]
    }
    fn check_model(&self, _: ModelKind) -> Result<(), ModelFailure> {
        Ok(())
    }
    fn tag(&self, text: &str) -> NerSpans {
        ["Alice Smith", "Alice", "McDonald", "Élodie", "Al"]
            .iter()
            .flat_map(|name| {
                text.match_indices(name)
                    .map(|(s, _)| (name.to_string(), "PER".into(), s, s + name.len()))
            })
            .collect()
    }
    fn tag_batch_checked(&self, texts: &[&str]) -> Result<Vec<NerSpans>, ModelFailure> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.tag_batch(texts))
    }
}
fn backend() -> Names {
    Names {
        calls: AtomicUsize::new(0),
    }
}

#[test]
fn person_counts_are_located_occurrences_not_correlated_signal_votes() {
    let model = backend();
    let text = "\u{feff}😀 Alice said, \"Hello.\" Alice waited. Alice said, \"Goodbye.\"";
    let result = ner_extract_persons_with_backend_checked(&model, &[text.into()]).unwrap();
    assert_eq!(result.persons, [("Alice".into(), 3)]);
    assert_eq!(result.mentions.len(), 3);
    assert_eq!(model.calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.model_entities.len(), 1);
    assert_eq!(
        result.model_entities[0]
            .iter()
            .filter(|(name, _, _, _)| name == "Alice")
            .count(),
        3
    );
    assert_eq!(result.availability, AnalysisAvailability::Partial);
    assert!(result.model_identity.is_some() && result.model_failure.is_none());
    for mention in result.mentions {
        assert_eq!(mention.character, "Alice");
        assert_eq!(mention.surface, "Alice");
        assert_eq!(
            &text[mention.range.byte_start..mention.range.byte_end],
            mention.surface
        );
        assert_eq!(
            TextRange::from_utf16(text, mention.range.utf16_start, mention.range.utf16_end)
                .unwrap(),
            mention.range
        );
    }
}

#[test]
fn longest_compound_aliases_do_not_double_count_and_source_casing_survives() {
    let text = "Alice Smith said, \"Hello.\" Alice Smith waited. Alice said, \"Goodbye.\" Alice rested. McDonald said, \"Yes.\" McDonald waited. Élodie spoke. Élodie said, \"Good day.\" Al spoke. Al waited.";
    let result = ner_extract_persons_with_backend_checked(&backend(), &[text.into()]).unwrap();
    let alice: Vec<_> = result
        .mentions
        .iter()
        .filter(|m| m.surface.starts_with("Alice"))
        .collect();
    assert_eq!(alice.len(), 4);
    assert_eq!(
        alice.iter().filter(|m| m.surface == "Alice Smith").count(),
        2
    );
    assert!(alice.iter().all(|m| m.character == alice[0].character));
    assert!(
        result
            .persons
            .iter()
            .any(|(n, c)| n == "McDonald" && *c == 2),
        "{:?}",
        result.persons
    );
    assert!(result.persons.iter().any(|(n, c)| n == "Élodie" && *c == 2));
    assert!(result.persons.iter().any(|(n, c)| n == "Al" && *c == 2));
    let total: usize = result.persons.iter().map(|(_, n)| *n).sum();
    assert_eq!(total, result.mentions.len());
}

#[test]
fn model_failure_is_explicit_while_bounded_heuristic_candidates_remain_useful() {
    struct Failed;
    impl InferenceBackend for Failed {
        fn embed(&self, texts: &[&str]) -> Vec<Vec<f64>> {
            vec![Vec::new(); texts.len()]
        }
        fn tag(&self, _: &str) -> NerSpans {
            Vec::new()
        }
        fn check_model(&self, _: ModelKind) -> Result<(), ModelFailure> {
            Err(ModelFailure::execution("fixture_execution_failure"))
        }
    }
    let text = "Alice said, \"Hello.\" Alice smiled. Alice said, \"Goodbye.\"".to_string();
    let result =
        ner_extract_persons_with_backend_checked(&Failed, std::slice::from_ref(&text)).unwrap();
    assert_eq!(
        result.model_failure.unwrap().code,
        "fixture_execution_failure"
    );
    assert!(result.model_identity.is_none());
    assert!(result.model_entities[0].is_empty());
    assert_eq!(result.persons, [("Alice".into(), 3)]);
    #[cfg(not(feature = "onnx"))]
    assert_eq!(
        ner_extract_persons_checked(&[text])
            .unwrap()
            .model_failure
            .unwrap()
            .code,
        "onnx_feature_disabled"
    );
    let empty = ner_extract_persons_with_backend_checked(onnx::backend(), &[]).unwrap();
    assert!(empty.persons.is_empty() && empty.model_failure.is_none());
}

#[test]
fn person_resource_and_name_boundaries_reject_without_panics_or_truncation() {
    for count in [9999, 10_000, 10_001] {
        let scenes = vec![String::new(); count];
        assert_eq!(
            ner_extract_persons_with_backend_checked(&backend(), &scenes).is_ok(),
            count <= MAX_NER_TEXTS
        );
    }
    for count in [
        MAX_PERSON_TOKENS - 1,
        MAX_PERSON_TOKENS,
        MAX_PERSON_TOKENS + 1,
    ] {
        let text = "x ".repeat(count);
        assert_eq!(
            ner_extract_persons_with_backend_checked(&backend(), &[text]).is_ok(),
            count <= MAX_PERSON_TOKENS
        );
    }
    for count in [
        MAX_PERSON_MARKERS - 1,
        MAX_PERSON_MARKERS,
        MAX_PERSON_MARKERS + 1,
    ] {
        let text = ".".repeat(count);
        assert_eq!(
            ner_extract_persons_with_backend_checked(&backend(), &[text]).is_ok(),
            count <= MAX_PERSON_MARKERS
        );
    }
    for bytes in [127, 128, 129, 200_000] {
        let name = format!("A{}", "a".repeat(bytes - 1));
        let text = format!("She saw {name}. She met {name}.");
        assert_eq!(
            ner_extract_persons_with_backend_checked(&backend(), &[text]).is_ok(),
            bytes <= 128
        );
    }
    for count in [
        MAX_PERSON_CANDIDATES - 1,
        MAX_PERSON_CANDIDATES,
        MAX_PERSON_CANDIDATES + 1,
    ] {
        let mut text = String::new();
        for i in 0..count {
            let mut name = String::from("N");
            for divisor in [26usize.pow(3), 26usize.pow(2), 26, 1] {
                name.push(char::from(b'a' + ((i / divisor) % 26) as u8));
            }
            text.push_str(&format!("She met {name}. "));
        }
        assert_eq!(
            ner_extract_persons_with_backend_checked(&backend(), &[text]).is_ok(),
            count <= MAX_PERSON_CANDIDATES
        );
    }
}
