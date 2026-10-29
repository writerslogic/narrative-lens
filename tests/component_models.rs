use narrative_lens::onnx::{
    checked::ModelKind,
    model::{ModelIdentity, ModelManifest, ModelProfile, ModelVerification},
};

fn manifest(kind: ModelKind) -> ModelManifest {
    let (profile, labels): (_, &[&str]) = match kind {
        ModelKind::Embedding => (ModelProfile::HuggingFaceMaskedMeanV1, &[]),
        ModelKind::Ner => (
            ModelProfile::HuggingFaceBio9V1,
            &[
                "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
            ],
        ),
        ModelKind::Emotion => (
            ModelProfile::HuggingFaceEkman7V1,
            narrative_lens::craft::emotion::EKMAN_LABELS,
        ),
        ModelKind::Nli => (
            ModelProfile::HuggingFaceNli3V1,
            narrative_lens::substrate::nli::NLI_LABELS,
        ),
    };
    ModelManifest {
        schema_version: 1,
        kind,
        profile,
        model_sha256: "0".repeat(64),
        tokenizer_sha256: "0".repeat(64),
        max_tokens: 4096,
        padding_token_id: 0,
        labels: labels.iter().map(|label| (*label).into()).collect(),
    }
}

#[test]
fn profiles_require_supported_kinds_versions_digests_and_unique_labels() {
    for kind in [
        ModelKind::Embedding,
        ModelKind::Ner,
        ModelKind::Emotion,
        ModelKind::Nli,
    ] {
        let valid = manifest(kind);
        valid.validate(kind).unwrap();
        let mut reordered = valid.clone();
        reordered.labels.reverse();
        reordered.validate(kind).unwrap();
        for version in [0, 2] {
            let mut invalid = valid.clone();
            invalid.schema_version = version;
            assert_eq!(
                invalid.validate(kind).unwrap_err().code,
                "model_manifest_version"
            );
        }
        let other = if kind == ModelKind::Nli {
            ModelKind::Emotion
        } else {
            ModelKind::Nli
        };
        assert_eq!(
            valid.validate(other).unwrap_err().code,
            "model_profile_kind"
        );
        let mut wrong_profile = valid.clone();
        wrong_profile.profile = manifest(other).profile;
        assert_eq!(
            wrong_profile.validate(kind).unwrap_err().code,
            "model_profile_kind"
        );
        for limit in [0, 4095, 4096, 4097] {
            let mut bounded = valid.clone();
            bounded.max_tokens = limit;
            assert_eq!(bounded.validate(kind).is_ok(), (1..=4096).contains(&limit));
        }
        for digest in [
            "f".repeat(63),
            "f".repeat(65),
            "F".repeat(64),
            "z".repeat(64),
        ] {
            let mut invalid = valid.clone();
            invalid.model_sha256 = digest;
            assert_eq!(
                invalid.validate(kind).unwrap_err().code,
                "model_manifest_digest_format"
            );
        }
        let mut unknown = valid.clone();
        if unknown.labels.is_empty() {
            unknown.labels.push("unknown".into());
        } else {
            unknown.labels[0] = "unknown".into();
        }
        assert_eq!(
            unknown.validate(kind).unwrap_err().code,
            "model_profile_labels"
        );
        if valid.labels.len() > 1 {
            let mut duplicate = valid.clone();
            duplicate.labels[0] = duplicate.labels[1].clone();
            assert_eq!(
                duplicate.validate(kind).unwrap_err().code,
                "model_profile_labels"
            );
            let mut missing = valid;
            missing.labels.pop();
            assert_eq!(
                missing.validate(kind).unwrap_err().code,
                "model_profile_labels"
            );
        }
    }
}

#[test]
fn identity_validation_rejects_inconsistent_verification_claims() {
    let mut identity = ModelIdentity::unverified(ModelKind::Nli);
    identity.validate(ModelKind::Nli).unwrap();
    assert_eq!(
        identity.validate(ModelKind::Emotion).unwrap_err().code,
        "model_identity_kind"
    );
    identity.manifest = Some(manifest(ModelKind::Nli));
    assert_eq!(
        identity.validate(ModelKind::Nli).unwrap_err().code,
        "model_identity_verification"
    );
    identity.verification = ModelVerification::VerifiedManifestV1;
    identity.validate(ModelKind::Nli).unwrap();
    identity.manifest.as_mut().unwrap().kind = ModelKind::Emotion;
    assert_eq!(
        identity.validate(ModelKind::Nli).unwrap_err().code,
        "model_profile_kind"
    );
    identity.manifest = None;
    assert_eq!(
        identity.validate(ModelKind::Nli).unwrap_err().code,
        "model_identity_verification"
    );
}

#[cfg(feature = "onnx")]
#[test]
fn json_manifests_are_bounded_and_reject_unknown_fields() {
    let valid = manifest(ModelKind::Nli);
    let json = serde_json::to_vec(&valid).unwrap();
    for size in [16 * 1024 - 1, 16 * 1024, 16 * 1024 + 1] {
        let mut padded = json.clone();
        padded.resize(size, b' ');
        assert_eq!(ModelManifest::from_json(&padded).is_ok(), size <= 16 * 1024);
    }
    assert!(ModelManifest::from_json(&[]).is_err());
    let mut unknown = serde_json::to_value(valid).unwrap();
    unknown["normalization"] = "maybe".into();
    assert_eq!(
        ModelManifest::from_json(&serde_json::to_vec(&unknown).unwrap())
            .unwrap_err()
            .code,
        "model_manifest_invalid"
    );
}

#[cfg(all(feature = "onnx", any(not(target_arch = "wasm32"), target_os = "wasi")))]
mod runtime {
    use super::*;
    use narrative_lens::onnx::{self, NerCache, checked};
    use sha2::{Digest, Sha256};

    static MODEL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/models")
                .join(name),
        )
        .unwrap()
    }

    fn tokenizer() -> Vec<u8> {
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
        tokenizer.to_string(false).unwrap().into_bytes()
    }

    fn pinned(kind: ModelKind, model: &[u8], tokenizer: &[u8]) -> ModelManifest {
        let mut profile = manifest(kind);
        profile.model_sha256 = format!("{:x}", Sha256::digest(model));
        profile.tokenizer_sha256 = format!("{:x}", Sha256::digest(tokenizer));
        profile
    }

    fn ner_with_logits(logits: [f32; 9]) -> Vec<u8> {
        let mut bytes = fixture("valid-ner.onnx");
        let mut raw_data = vec![0x4a, 36];
        raw_data.extend(3.0f32.to_le_bytes());
        raw_data.resize(38, 0);
        let positions: Vec<_> = bytes
            .windows(raw_data.len())
            .enumerate()
            .filter_map(|(i, candidate)| (candidate == raw_data).then_some(i))
            .collect();
        assert_eq!(
            positions.len(),
            1,
            "fixture has one known float32 raw-data field"
        );
        let start = positions[0] + 2;
        for (i, logit) in logits.iter().enumerate() {
            bytes[start + i * 4..start + (i + 1) * 4].copy_from_slice(&logit.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn strict_ner_ties_are_invariant_to_source_label_permutations() {
        let _guard = MODEL_TEST_LOCK.lock().unwrap();
        let tokenizer = tokenizer();
        let canonical = manifest(ModelKind::Ner).labels;
        let mut two_maxima = [-1.0; 9];
        two_maxima[3] = 2.0;
        two_maxima[5] = 2.0;
        for (semantic_logits, expected) in [
            ([0.0; 9], vec![]),
            (two_maxima, vec![("Alice".into(), "PER".into(), 0, 5)]),
        ] {
            for reverse in [false, true] {
                for rotation in 0..9 {
                    let mut order: Vec<_> = (0..9).collect();
                    if reverse {
                        order.reverse();
                    }
                    order.rotate_left(rotation);
                    let source_logits =
                        std::array::from_fn(|source| semantic_logits[order[source]]);
                    let bytes = ner_with_logits(source_logits);
                    let mut profile = pinned(ModelKind::Ner, &bytes, &tokenizer);
                    profile.labels = order
                        .iter()
                        .map(|&source| canonical[source].clone())
                        .collect();
                    onnx::set_model_bytes_verified(ModelKind::Ner, &bytes, &tokenizer, &profile)
                        .unwrap();
                    assert_eq!(
                        checked::entities(onnx::backend(), &["Alice"]).unwrap(),
                        vec![expected.clone()],
                        "reverse={reverse}, rotation={rotation}"
                    );
                }
            }
        }
    }

    fn failed_replacement(
        kind: ModelKind,
        model: &[u8],
        tokenizer: &[u8],
        profile: &ModelManifest,
        code: &str,
    ) {
        let generation = onnx::model_generation();
        let failure = onnx::set_model_bytes_verified(kind, model, tokenizer, profile).unwrap_err();
        assert_eq!(failure.code, code);
        assert_eq!(onnx::model_generation(), generation + 1);
        assert_eq!(onnx::backend().check_model(kind).unwrap_err(), failure);
        assert_eq!(onnx::backend().model_identity(kind).unwrap_err(), failure);
    }

    #[test]
    fn verified_artifacts_labels_and_model_replacements_are_enforced() {
        let _guard = MODEL_TEST_LOCK.lock().unwrap();
        let tokenizer = tokenizer();
        for (kind, file) in [
            (ModelKind::Embedding, "valid-embedding.onnx"),
            (ModelKind::Ner, "valid-ner.onnx"),
            (ModelKind::Emotion, "valid-emotion.onnx"),
            (ModelKind::Nli, "valid-nli.onnx"),
        ] {
            let bytes = fixture(file);
            let profile = pinned(kind, &bytes, &tokenizer);
            let generation = onnx::model_generation();
            let identity =
                onnx::set_model_bytes_verified(kind, &bytes, &tokenizer, &profile).unwrap();
            assert_eq!(identity.verification, ModelVerification::VerifiedManifestV1);
            assert_eq!(identity.manifest.as_ref(), Some(&profile));
            assert_eq!(onnx::backend().model_identity(kind).unwrap(), identity);
            assert_eq!(onnx::model_generation(), generation + 1);
            let mut changed = bytes.clone();
            *changed.last_mut().unwrap() ^= 1;
            failed_replacement(
                kind,
                &changed,
                &tokenizer,
                &profile,
                "model_digest_mismatch",
            );
            let mut changed_tokenizer = tokenizer.clone();
            changed_tokenizer.push(b' ');
            failed_replacement(
                kind,
                &bytes,
                &changed_tokenizer,
                &profile,
                "tokenizer_digest_mismatch",
            );
            let mut wrong = profile.clone();
            wrong.schema_version = 2;
            failed_replacement(kind, &bytes, &tokenizer, &wrong, "model_manifest_version");
            let mut wrong_kind = profile.clone();
            wrong_kind.kind = if kind == ModelKind::Nli {
                ModelKind::Emotion
            } else {
                ModelKind::Nli
            };
            failed_replacement(kind, &bytes, &tokenizer, &wrong_kind, "model_profile_kind");
            let mut wrong_labels = profile.clone();
            if wrong_labels.labels.len() > 1 {
                wrong_labels.labels[0] = wrong_labels.labels[1].clone();
            } else {
                wrong_labels.labels.push("unknown".into());
            }
            failed_replacement(
                kind,
                &bytes,
                &tokenizer,
                &wrong_labels,
                "model_profile_labels",
            );
            onnx::set_model_bytes_verified(kind, &bytes, &tokenizer, &profile).unwrap();
            match kind {
                ModelKind::Embedding => assert_eq!(
                    checked::embeddings(onnx::backend(), &["Alice"]).unwrap(),
                    vec![vec![3.0, 0.0]]
                ),
                ModelKind::Nli => assert!(
                    checked::nli(onnx::backend(), &[("Alice", "Alice")]).unwrap()[0][0] > 0.9
                ),
                ModelKind::Emotion => {
                    assert!(checked::emotion(onnx::backend(), &["Alice"]).unwrap()[0][0] > 0.7)
                }
                ModelKind::Ner => {
                    assert!(checked::entities(onnx::backend(), &["Alice"]).unwrap()[0].is_empty())
                }
            }
        }

        let nli = fixture("valid-nli.onnx");
        let mut profile = pinned(ModelKind::Nli, &nli, &tokenizer);
        profile.labels.rotate_right(1);
        onnx::set_model_bytes_verified(ModelKind::Nli, &nli, &tokenizer, &profile).unwrap();
        let probabilities = checked::nli(onnx::backend(), &[("Alice", "Alice")]).unwrap();
        assert!(probabilities[0][2] > 0.9 && probabilities[0][0] < 0.1);
        profile.max_tokens = 2;
        onnx::set_model_bytes_verified(ModelKind::Nli, &nli, &tokenizer, &profile).unwrap();
        for (hypothesis, tokens) in [("", 1), ("Alice", 2), ("Alice Alice", 3)] {
            let result = checked::nli(onnx::backend(), &[("Alice", hypothesis)]);
            if tokens <= 2 {
                result.unwrap();
            } else {
                assert_eq!(result.unwrap_err().code, "model_profile_token_limit");
            }
        }
        let emotion = fixture("valid-emotion.onnx");
        let mut profile = pinned(ModelKind::Emotion, &emotion, &tokenizer);
        profile.labels.swap(0, 3);
        onnx::set_model_bytes_verified(ModelKind::Emotion, &emotion, &tokenizer, &profile).unwrap();
        assert!(checked::emotion(onnx::backend(), &["Alice"]).unwrap()[0][3] > 0.7);

        let ner = fixture("valid-ner.onnx");
        let mut profile = pinned(ModelKind::Ner, &ner, &tokenizer);
        let mut cache = NerCache::new(1024 * 1024).unwrap();
        onnx::set_model_bytes_verified(ModelKind::Ner, &ner, &tokenizer, &profile).unwrap();
        assert!(
            cache
                .tag_batch(onnx::backend(), "same", &["Alice"])
                .unwrap()[0]
                .is_empty()
        );
        profile.labels.swap(0, 3);
        onnx::set_model_bytes_verified(ModelKind::Ner, &ner, &tokenizer, &profile).unwrap();
        let spans = cache
            .tag_batch(onnx::backend(), "same", &["Alice"])
            .unwrap();
        assert_eq!(spans, vec![vec![("Alice".into(), "PER".into(), 0, 5)]]);
        profile.model_sha256 = "0".repeat(64);
        failed_replacement(
            ModelKind::Ner,
            &ner,
            &tokenizer,
            &profile,
            "model_digest_mismatch",
        );
        assert!(
            cache
                .tag_batch(onnx::backend(), "same", &["Alice"])
                .is_err()
        );
        assert_eq!(cache.retained_bytes(), 0);

        for (file, error) in [
            ("incompatible-nli.onnx", "model_profile_output_shape"),
            ("failing-nli.onnx", "model_profile_inputs"),
        ] {
            let bytes = fixture(file);
            let profile = pinned(ModelKind::Nli, &bytes, &tokenizer);
            failed_replacement(ModelKind::Nli, &bytes, &tokenizer, &profile, error);
        }
        let mut configured = tokenizers::Tokenizer::from_bytes(&tokenizer).unwrap();
        configured.with_padding(Some(tokenizers::PaddingParams::default()));
        let padded = configured.to_string(false).unwrap().into_bytes();
        let profile = pinned(ModelKind::Nli, &nli, &padded);
        failed_replacement(
            ModelKind::Nli,
            &nli,
            &padded,
            &profile,
            "model_profile_tokenizer_settings",
        );
        let mut profile = pinned(ModelKind::Nli, &nli, &tokenizer);
        profile.padding_token_id = 999;
        failed_replacement(
            ModelKind::Nli,
            &nli,
            &tokenizer,
            &profile,
            "model_profile_tokenizer_settings",
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            let root = std::env::temp_dir()
                .join(format!("narrative-pinned-models-{}", std::process::id()));
            let directory = root.join("nli-onnx");
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(directory.join("model.onnx"), &nli).unwrap();
            std::fs::write(directory.join("tokenizer.json"), &tokenizer).unwrap();
            let profile = pinned(ModelKind::Nli, &nli, &tokenizer);
            let identity = onnx::set_model_directory_verified(
                ModelKind::Nli,
                root.to_str().unwrap(),
                &profile,
            )
            .unwrap();
            std::fs::write(directory.join("model.onnx"), b"changed after loading").unwrap();
            std::fs::write(directory.join("tokenizer.json"), b"changed after loading").unwrap();
            assert_eq!(
                onnx::backend().model_identity(ModelKind::Nli).unwrap(),
                identity
            );
            assert!(checked::nli(onnx::backend(), &[("Alice", "Alice")]).unwrap()[0][0] > 0.9);
            assert_eq!(
                onnx::set_model_directory_verified(
                    ModelKind::Nli,
                    root.to_str().unwrap(),
                    &profile
                )
                .unwrap_err()
                .code,
                "model_digest_mismatch"
            );
            assert!(onnx::backend().check_model(ModelKind::Nli).is_err());
            std::fs::write(directory.join("model.onnx"), nli).unwrap();
            std::fs::write(directory.join("tokenizer.json"), tokenizer).unwrap();
            onnx::set_nli_model_directory(root.to_str().unwrap());
            let legacy = onnx::backend().model_identity(ModelKind::Nli).unwrap();
            assert_eq!(legacy.verification, ModelVerification::UnverifiedLegacy);
            assert_eq!(legacy.manifest, None);
            std::fs::remove_dir_all(root).unwrap();
        }
    }
}
