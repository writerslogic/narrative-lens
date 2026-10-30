use narrative_lens::{
    manuscript::*,
    onnx::{
        self, InferenceBackend,
        checked::{ModelFailure, ModelKind},
        model::{ModelIdentity, ModelVerification},
    },
};

#[cfg(feature = "onnx")]
#[path = "support/digest.rs"]
mod digest;

struct Backend {
    kind: ModelKind,
    replace_during_inference: bool,
}
impl InferenceBackend for Backend {
    fn check_model(&self, kind: ModelKind) -> Result<(), ModelFailure> {
        if kind == ModelKind::Nli {
            Ok(())
        } else {
            Err(ModelFailure::unavailable("test_capability"))
        }
    }
    fn model_identity(&self, _: ModelKind) -> Result<ModelIdentity, ModelFailure> {
        Ok(ModelIdentity::unverified(self.kind))
    }
    fn embed(&self, texts: &[&str]) -> Vec<Vec<f64>> {
        vec![vec![]; texts.len()]
    }
    fn tag(&self, _: &str) -> onnx::NerSpans {
        vec![]
    }
    fn classify_pairs_checked(
        &self,
        pairs: &[(&str, &str)],
    ) -> Result<Vec<Vec<f64>>, ModelFailure> {
        if self.replace_during_inference {
            onnx::set_nli_model_directory("/narrative-lens-test-generation-change");
        }
        Ok(vec![vec![0.95, 0.025, 0.025]; pairs.len()])
    }
}
fn input() -> Manuscript {
    Manuscript {
        document_id: "provenance".into(),
        scenes: vec![
            Scene::new("before", "Alice stayed."),
            Scene::new("after", "Alice left."),
        ],
        known_characters: vec![],
        configuration: Configuration {
            rules: Some(["NL-CONTINUITY-NLI".into()].into()),
            nli: ModelPolicy::Fallback,
            ..Default::default()
        },
    }
}

#[test]
fn reports_preserve_backend_identity_without_claiming_unverified_pins() {
    let report = analyze_with_backend(
        &input(),
        &Backend {
            kind: ModelKind::Nli,
            replace_during_inference: false,
        },
    )
    .unwrap();
    assert_eq!(report.findings.len(), 1);
    assert_eq!(
        report.model_identities,
        [ModelIdentity::unverified(ModelKind::Nli)]
    );
    let value = serde_json::to_value(&report).unwrap();
    assert_eq!(
        value["modelIdentities"][0]["verification"],
        "unverified_legacy"
    );
    let mut old = value;
    old.as_object_mut().unwrap().remove("modelIdentities");
    let imported: Report = serde_json::from_value(old).unwrap();
    imported.validate().unwrap();
    assert!(imported.model_identities.is_empty());
    let mut invalid = report.clone();
    invalid.model_identities[0].verification = ModelVerification::VerifiedManifestV1;
    assert!(invalid.validate().is_err());
    let mut disabled = report;
    disabled.configuration.as_mut().unwrap().nli = ModelPolicy::Disabled;
    assert!(disabled.validate().is_err());
}

#[test]
fn incompatible_identity_abstains_and_required_mode_fails() {
    let backend = Backend {
        kind: ModelKind::Emotion,
        replace_during_inference: false,
    };
    let mut input = input();
    let report = analyze_with_backend(&input, &backend).unwrap();
    assert!(report.findings.is_empty());
    assert!(report.model_identities.is_empty());
    let coverage = report
        .coverage
        .iter()
        .find(|c| c.analyzer == "nli")
        .unwrap();
    assert_eq!(coverage.status, Status::ExecutionFailure);
    assert!(
        coverage
            .diagnostics
            .iter()
            .any(|d| d == "model_identity_kind")
    );
    input.configuration.nli = ModelPolicy::Required;
    assert!(analyze_with_backend(&input, &backend).is_err());
}

#[test]
fn replacement_during_inference_cannot_publish_stale_provenance_or_findings() {
    if std::env::var_os("NARRATIVE_LENS_PROVENANCE_RACE").is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "replacement_during_inference_cannot_publish_stale_provenance_or_findings",
                "--nocapture",
            ])
            .env("NARRATIVE_LENS_PROVENANCE_RACE", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let backend = Backend {
        kind: ModelKind::Nli,
        replace_during_inference: true,
    };
    let mut input = input();
    let report = analyze_with_backend(&input, &backend).unwrap();
    assert!(report.findings.is_empty());
    assert!(report.model_identities.is_empty());
    let coverage = report
        .coverage
        .iter()
        .find(|c| c.analyzer == "nli")
        .unwrap();
    assert_eq!(coverage.status, Status::ExecutionFailure);
    assert!(
        coverage
            .diagnostics
            .iter()
            .any(|d| d == "model_changed_during_inference")
    );
    input.configuration.nli = ModelPolicy::Required;
    assert!(analyze_with_backend(&input, &backend).is_err());
}

#[test]
#[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
fn actual_verified_model_identity_is_attached_to_nli_report() {
    if std::env::var_os("NARRATIVE_LENS_VERIFIED_REPORT").is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "actual_verified_model_identity_is_attached_to_nli_report",
                "--nocapture",
            ])
            .env("NARRATIVE_LENS_VERIFIED_REPORT", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    use onnx::model::{ModelManifest, ModelProfile};
    let model = include_bytes!("fixtures/models/valid-nli.onnx");
    let vocab = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab([("[UNK]".into(), 0)].into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let tokenizer = tokenizers::Tokenizer::new(vocab)
        .to_string(false)
        .unwrap()
        .into_bytes();
    let manifest = ModelManifest {
        schema_version: 1,
        kind: ModelKind::Nli,
        profile: ModelProfile::HuggingFaceNli3V1,
        model_sha256: digest::sha256_hex(model),
        tokenizer_sha256: digest::sha256_hex(&tokenizer),
        max_tokens: 512,
        padding_token_id: 0,
        labels: ["contradiction", "neutral", "entailment"]
            .map(String::from)
            .into(),
        nominal_labels: Vec::new(),
        window_tokens: None,
    };
    let identity =
        onnx::set_model_bytes_verified(ModelKind::Nli, model, &tokenizer, &manifest).unwrap();
    let report = analyze_manuscript(&input()).unwrap();
    assert_eq!(report.findings.len(), 1);
    assert_eq!(report.model_identities, [identity]);
    assert_eq!(
        report.model_identities[0].verification,
        ModelVerification::VerifiedManifestV1
    );
    report.validate().unwrap();
}
