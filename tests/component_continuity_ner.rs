use narrative_lens::{
    continuity::ner::*,
    onnx::{
        self, InferenceBackend, NerSpans,
        checked::{ModelFailure, ModelKind},
        model::ModelVerification,
    },
};

#[cfg(feature = "onnx")]
#[path = "support/digest.rs"]
mod digest;

static MODEL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Names;
impl InferenceBackend for Names {
    fn embed(&self, texts: &[&str]) -> Vec<Vec<f64>> {
        vec![Vec::new(); texts.len()]
    }
    fn check_model(&self, _: ModelKind) -> Result<(), ModelFailure> {
        Ok(())
    }
    fn tag_batch_checked(&self, texts: &[&str]) -> Result<Vec<NerSpans>, ModelFailure> {
        Ok(self.tag_batch(texts))
    }
    fn tag(&self, text: &str) -> NerSpans {
        text.match_indices("Alice")
            .map(|(s, _)| ("Alice".into(), "PER".into(), s, s + 5))
            .collect()
    }
}

#[test]
fn checked_ner_preserves_full_source_coordinates_and_declares_identity() {
    let _guard = MODEL_TEST_LOCK.lock().unwrap();
    let text = format!(
        "{}\u{3000}Alice {} Alice",
        "雪".repeat(497),
        "x ".repeat(1500)
    );
    let result =
        ner_extract_entities_batch_with_backend_checked(&Names, &["", &text, "Alice"]).unwrap();
    assert_eq!(result.entities.len(), 3);
    assert!(result.entities[0].is_empty());
    assert_eq!(result.entities[1].len(), 2);
    assert_eq!(result.entities[2], [("Alice".into(), "PER".into(), 0, 5)]);
    for (_, _, start, end) in &result.entities[1] {
        assert_eq!(text.get(*start..*end), Some("Alice"));
    }
    assert!(result.windows > 2);
    assert_eq!(
        result.model_identity.unwrap().verification,
        ModelVerification::UnverifiedLegacy
    );
    let empty =
        ner_extract_entities_batch_with_backend_checked(onnx::backend(), &["", " \u{3000}"])
            .unwrap();
    assert!(empty.model_identity.is_none());
    assert_eq!(empty.windows, 0);
}

#[test]
fn checked_ner_distinguishes_execution_cardinality_and_invalid_source_failures() {
    let _guard = MODEL_TEST_LOCK.lock().unwrap();
    struct Broken(u8);
    impl InferenceBackend for Broken {
        fn embed(&self, texts: &[&str]) -> Vec<Vec<f64>> {
            vec![Vec::new(); texts.len()]
        }
        fn check_model(&self, _: ModelKind) -> Result<(), ModelFailure> {
            Ok(())
        }
        fn tag(&self, _: &str) -> NerSpans {
            Vec::new()
        }
        fn tag_batch_checked(&self, texts: &[&str]) -> Result<Vec<NerSpans>, ModelFailure> {
            match self.0 {
                0 => Err(ModelFailure::execution("model_profile_token_limit")),
                1 => Ok(Vec::new()),
                2 => Ok(vec![
                    vec![("wrong".into(), "PER".into(), 0, 5)];
                    texts.len()
                ]),
                _ => Ok(vec![
                    vec![("Alice".into(), "PERSON".into(), 0, 5)];
                    texts.len()
                ]),
            }
        }
    }
    for (kind, code) in [
        (0, "model_profile_token_limit"),
        (1, "ner_batch_cardinality"),
        (2, "ner_invalid_span_or_label"),
        (3, "ner_invalid_span_or_label"),
    ] {
        assert_eq!(
            ner_extract_entities_batch_with_backend_checked(&Broken(kind), &["Alice"])
                .unwrap_err()
                .code,
            code
        );
    }
    #[cfg(not(feature = "onnx"))]
    assert_eq!(
        ner_extract_entities_batch_checked(&["Alice"])
            .unwrap_err()
            .code,
        "onnx_feature_disabled"
    );
}

#[test]
fn checked_ner_bounds_inputs_and_observed_entities() {
    let _guard = MODEL_TEST_LOCK.lock().unwrap();
    for count in [9999, 10_000, 10_001] {
        let texts = vec![""; count];
        assert_eq!(
            ner_extract_entities_batch_with_backend_checked(&Names, &texts).is_ok(),
            count <= MAX_NER_TEXTS
        );
        let text = "Alice ".repeat(count);
        let result = ner_extract_entities_batch_with_backend_checked(&Names, &[&text]);
        assert_eq!(result.is_ok(), count <= MAX_NER_ENTITIES);
        if let Ok(result) = result {
            assert_eq!(result.entities[0].len(), count);
        }
    }
    for bytes in [
        MAX_NER_TEXT_BYTES - 1,
        MAX_NER_TEXT_BYTES,
        MAX_NER_TEXT_BYTES + 1,
    ] {
        let text = " ".repeat(bytes);
        assert_eq!(
            ner_extract_entities_batch_with_backend_checked(&Names, &[&text]).is_ok(),
            bytes <= MAX_NER_TEXT_BYTES
        );
    }
    for bytes in [
        MAX_NER_TOTAL_BYTES - 1,
        MAX_NER_TOTAL_BYTES,
        MAX_NER_TOTAL_BYTES + 1,
    ] {
        let mut remaining = bytes;
        let mut texts = Vec::new();
        while remaining > 0 {
            let len = remaining.min(MAX_NER_TEXT_BYTES);
            texts.push(" ".repeat(len));
            remaining -= len;
        }
        let refs: Vec<_> = texts.iter().map(String::as_str).collect();
        assert_eq!(
            ner_extract_entities_batch_with_backend_checked(&Names, &refs).is_ok(),
            bytes <= MAX_NER_TOTAL_BYTES
        );
    }
}

#[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
#[test]
fn native_verified_ner_reports_actual_token_budget_failure() {
    let _guard = MODEL_TEST_LOCK.lock().unwrap();
    use onnx::model::{ModelManifest, ModelProfile};
    let bytes = include_bytes!("fixtures/models/valid-ner.onnx");
    let vocabulary = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(
            [("[UNK]".into(), 0), ("Alice".into(), 1)]
                .into_iter()
                .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(vocabulary);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    let tokenizer = tokenizer.to_string(false).unwrap().into_bytes();
    let profile = ModelManifest {
        schema_version: 1,
        kind: ModelKind::Ner,
        profile: ModelProfile::HuggingFaceBio9V1,
        model_sha256: digest::sha256_hex(bytes),
        tokenizer_sha256: digest::sha256_hex(&tokenizer),
        max_tokens: 1,
        padding_token_id: 0,
        labels: [
            "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    };
    onnx::set_model_bytes_verified(ModelKind::Ner, bytes, &tokenizer, &profile).unwrap();
    for tokens in [0, 1, 2] {
        let text = vec!["Alice"; tokens].join(" ");
        let result = ner_extract_entities_batch_checked(&[&text]);
        if tokens == 0 {
            assert!(result.unwrap().model_identity.is_none());
        } else if tokens == 1 {
            assert_eq!(
                result.unwrap().model_identity.unwrap().verification,
                ModelVerification::VerifiedManifestV1
            );
        } else {
            assert_eq!(result.unwrap_err().code, "model_profile_token_limit");
        }
    }
}
