#![cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
#[test]
fn invalid_local_model_fixture_has_a_useful_diagnostic() {
    use narrative_lens::onnx::{
        self,
        checked::{self, FailureKind},
    };
    let dir = std::env::temp_dir().join(format!(
        "narrative-lens-invalid-model-{}",
        std::process::id()
    ));
    let model = dir.join("nli-onnx");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("model.onnx"), b"invalid local fixture").unwrap();
    std::fs::write(model.join("tokenizer.json"), b"{}").unwrap();
    let previous = std::env::current_dir().unwrap();
    let work = dir.join("work");
    std::fs::create_dir(&work).unwrap();
    std::env::set_current_dir(&work).unwrap();
    onnx::set_nli_model_directory(dir.to_str().unwrap());
    let failure = checked::nli(onnx::backend(), &[("a", "b")]).unwrap_err();
    std::env::set_current_dir(previous).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    assert_eq!(failure.kind, FailureKind::Incompatible);
    assert_eq!(failure.code, "tokenizer_invalid");
}
