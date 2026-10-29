#![cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
use narrative_lens::onnx::{
    self,
    checked::{self, FailureKind},
};
#[test]
fn native_runtime_contracts_use_small_local_models() {
    if let Ok(case) = std::env::var("NARRATIVE_LENS_MODEL_CASE") {
        let (capability, status) = case.split_once(':').unwrap();
        let temp =
            std::env::temp_dir().join(format!("narrative-lens-runtime-{}", std::process::id()));
        let folder = match capability {
            "embedding" => "sbert-onnx",
            "ner" => "ner-onnx",
            "emotion" => "emotion-onnx",
            "nli" => "nli-onnx",
            _ => panic!("unknown test capability"),
        };
        let model = temp.join(folder);
        std::fs::create_dir_all(&model).unwrap();
        std::fs::copy(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/models")
                .join(format!("{status}-{capability}.onnx")),
            model.join("model.onnx"),
        )
        .unwrap();
        let wordlevel = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab([("[UNK]".into(), 0)].into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        tokenizers::Tokenizer::new(wordlevel)
            .save(model.join("tokenizer.json"), false)
            .unwrap();
        let work = temp.join("work");
        std::fs::create_dir(&work).unwrap();
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(&work).unwrap();
        match capability {
            "embedding" => onnx::set_sbert_model_directory(temp.to_str().unwrap()),
            "ner" => onnx::set_ner_model_directory(temp.to_str().unwrap()),
            "emotion" => onnx::set_emotion_model_directory(temp.to_str().unwrap()),
            _ => onnx::set_nli_model_directory(temp.to_str().unwrap()),
        }
        let result = match capability {
            "embedding" => checked::embeddings(onnx::backend(), &["Alice stayed."]).map(|v| {
                assert_eq!(v, vec![vec![3.0, 0.0]]);
            }),
            "ner" => checked::entities(onnx::backend(), &["Alice stayed."]).map(|v| {
                assert_eq!(v, vec![vec![]]);
            }),
            "emotion" => checked::emotion(onnx::backend(), &["Alice stayed."]).map(|v| {
                assert_eq!(v.len(), 1);
                assert_eq!(v[0].len(), 7);
            }),
            _ => checked::nli(onnx::backend(), &[("Alice stayed.", "Alice left.")]).map(|v| {
                assert_eq!(v.len(), 1);
                assert!(v[0][0] > 0.9);
            }),
        };
        match status {
            "valid" => result.unwrap(),
            "incompatible" => assert_eq!(result.unwrap_err().kind, FailureKind::Incompatible),
            "failing" => {
                let error = result.unwrap_err();
                assert_eq!(error.kind, FailureKind::Execution);
                assert_eq!(error.code, "onnx_execution_failed");
            }
            _ => unreachable!(),
        }
        std::env::set_current_dir(original).unwrap();
        std::fs::remove_dir_all(temp).unwrap();
        return;
    }
    for capability in ["embedding", "ner", "emotion", "nli"] {
        for status in ["valid", "incompatible", "failing"] {
            let case = format!("{capability}:{status}");
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "native_runtime_contracts_use_small_local_models",
                    "--nocapture",
                ])
                .env("NARRATIVE_LENS_MODEL_CASE", &case)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{case}: {} {}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }
}
