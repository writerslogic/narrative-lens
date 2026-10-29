#[test]
fn missing_model_is_never_reported_as_a_clean_inference() {
    use narrative_lens::onnx::{
        self,
        checked::{self, FailureKind},
    };
    let dir = std::env::temp_dir().join(format!(
        "narrative-lens-missing-model-{}",
        std::process::id()
    ));
    let work = dir.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let previous = std::env::current_dir().unwrap();
    std::env::set_current_dir(&work).unwrap();
    onnx::set_nli_model_directory(dir.to_str().unwrap());
    let error = checked::nli(onnx::backend(), &[("a", "b")]).unwrap_err();
    assert_eq!(error.kind, FailureKind::Unavailable);
    assert!(narrative_lens::substrate::nli::classify_pair("a", "b").is_empty());
    std::env::set_current_dir(previous).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}
