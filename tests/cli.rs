#![cfg(feature = "cli")]
use serde_json::Value;
use std::{
    fs,
    process::{Command, Output},
};
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_narrative-lens"))
        .args(args)
        .output()
        .unwrap()
}
struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "narrative-lens-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn file(&self, name: &str, text: &str) -> String {
        let p = self.0.join(name);
        fs::write(&p, text).unwrap();
        p.to_str().unwrap().into()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn validate_sarif(paths: &[&str]) {
    let required = std::env::var("NARRATIVE_LENS_REQUIRE_SARIF_VALIDATION").is_ok_and(|v| v == "1");
    let validation = match Command::new("uv")
        .args(["run", "--locked", "scripts/validate_sarif.py"])
        .args(paths)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
    {
        Ok(validation) => validation,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !required => {
            eprintln!(
                "skipping official SARIF schema validation: uv is not installed; set NARRATIVE_LENS_REQUIRE_SARIF_VALIDATION=1 to require it"
            );
            return;
        }
        Err(error) => panic!("uv is required for official SARIF schema validation: {error}"),
    };
    assert!(
        validation.status.success(),
        "{}",
        String::from_utf8_lossy(&validation.stderr)
    );
}

#[test]
fn new_only_requires_the_entire_population_even_when_retained_findings_persist() {
    let f = Fixture::new();
    for limit in [1, 2, 3] {
        let file = f.file("revision.txt", "She saw saw the bird.");
        let config = f.file(
            "limit.json",
            &format!(r#"{{"rules":["NL-GRAMMAR-REPETITION"],"limits":{{"maxFindings":{limit}}}}}"#),
        );
        let before = run(&["--format", "json", "--config", &config, &file]);
        let baseline = f.file("before.json", std::str::from_utf8(&before.stdout).unwrap());
        fs::write(&file, "She saw saw the bird. He ran ran through town.").unwrap();
        let output = run(&[
            "--format",
            "json",
            "--config",
            &config,
            "--baseline",
            &baseline,
            "--new-only",
            "--fail-on",
            "warning",
            &file,
        ]);
        assert_eq!(output.status.code(), Some(if limit == 1 { 2 } else { 1 }));
        let json: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            json["comparison"]["persistent"].as_array().unwrap().len(),
            1
        );
        assert_eq!(json["comparison"]["complete"], limit >= 2);
        if limit == 1 {
            assert_eq!(json["report"]["summary"]["omittedFindings"], 1);
        }
    }
}

#[test]
fn sarif_bom_offsets_and_historical_locations_preserve_text_semantics() {
    let f = Fixture::new();
    let mut regions = Vec::new();
    let mut raw_starts = Vec::new();
    for bom in ["", "\u{feff}"] {
        let file = f.file("bom.txt", &format!("{bom}😀 She saw saw the bird.\r\n"));
        let args = [
            "--rules",
            "NL-GRAMMAR-REPETITION",
            "--format",
            "json",
            &file,
        ];
        let before = run(&args);
        let json: Value = serde_json::from_slice(&before.stdout).unwrap();
        raw_starts.push(
            json["report"]["findings"][0]["evidence"][0]["range"]["utf16Start"]
                .as_u64()
                .unwrap(),
        );
        assert_eq!(
            json["sourceBomUtf16"]
                .as_object()
                .unwrap()
                .values()
                .next()
                .unwrap(),
            if bom.is_empty() { 0 } else { 1 }
        );
        let baseline = f.file(
            "baseline.json",
            std::str::from_utf8(&before.stdout).unwrap(),
        );
        let current = run(&[
            "--rules",
            "NL-GRAMMAR-REPETITION",
            "--format",
            "sarif",
            &file,
        ]);
        let sarif: Value = serde_json::from_slice(&current.stdout).unwrap();
        let region =
            sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["region"].clone();
        regions.push(region.clone());
        fs::write(&file, "The bird flew away.").unwrap();
        let resolved = run(&[
            "--rules",
            "NL-GRAMMAR-REPETITION",
            "--format",
            "sarif",
            "--baseline",
            &baseline,
            &file,
        ]);
        let historical: Value = serde_json::from_slice(&resolved.stdout).unwrap();
        let old = &historical["runs"][0]["results"][0];
        assert_eq!(old["baselineState"], "absent");
        assert_eq!(
            old["locations"][0]["physicalLocation"]["region"]["charOffset"],
            region["charOffset"]
        );
        assert_eq!(
            old["locations"][0]["physicalLocation"]["region"]["charLength"],
            region["charLength"]
        );
        let mut legacy = json;
        legacy.as_object_mut().unwrap().remove("sourceBomUtf16");
        fs::write(&baseline, serde_json::to_vec(&legacy).unwrap()).unwrap();
        let old_without_metadata = run(&[
            "--rules",
            "NL-GRAMMAR-REPETITION",
            "--format",
            "sarif",
            "--baseline",
            &baseline,
            &file,
        ]);
        let old: Value = serde_json::from_slice(&old_without_metadata.stdout).unwrap();
        assert!(old["runs"][0]["results"][0].get("locations").is_none());
        assert!(
            !old["runs"][0]["results"][0]["properties"]["originalEvidence"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
    assert_eq!(regions[0], regions[1]);
    assert_eq!(raw_starts[1], raw_starts[0] + 1);
}
#[test]
fn cli_reports_filters_baselines_errors_and_official_sarif_schema() {
    let f = Fixture::new();
    let file = f.file("snow scene.md", "# Snow\r\n😀 Zoë saw saw the gate.\r\n");
    let rule = "NL-GRAMMAR-REPETITION";
    let output = run(&[
        "--format",
        "json",
        "--rules",
        rule,
        "--fail-on",
        "warning",
        &file,
    ]);
    assert_eq!(output.status.code(), Some(1));
    let json: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["report"]["findings"].as_array().unwrap().len(), 1);
    let baseline = f.file(
        "baseline.json",
        std::str::from_utf8(&output.stdout).unwrap(),
    );
    assert_eq!(
        run(&[
            "--baseline",
            &baseline,
            "--new-only",
            "--rules",
            rule,
            "--fail-on",
            "warning",
            &file
        ])
        .status
        .code(),
        Some(0)
    );
    assert_eq!(
        run(&[
            "--rules",
            rule,
            "--suppress",
            rule,
            "--fail-on",
            "info",
            &file
        ])
        .status
        .code(),
        Some(0)
    );
    assert_eq!(
        run(&["--rules", rule, "--minimum-severity", "error", &file])
            .status
            .code(),
        Some(0)
    );
    assert_eq!(run(&["--rules", "bad", &file]).status.code(), Some(2));
    assert_eq!(run(&["--format", "xml", &file]).status.code(), Some(2));
    assert_eq!(
        run(&[
            "--rules",
            rule,
            "--fail-on",
            "warning",
            "--max-findings",
            "1",
            &file
        ])
        .status
        .code(),
        Some(0)
    );
    assert_eq!(run(&["missing.txt"]).status.code(), Some(2));
    let config = f.file(
        "config.json",
        r#"{"rules":["NL-GRAMMAR-REPETITION"],"thresholds":{"minSceneWords":0}}"#,
    );
    assert_eq!(run(&["--config", &config, &file]).status.code(), Some(2));
    let sarif = run(&["--format", "sarif", "--rules", rule, &file]);
    assert!(sarif.status.success());
    let value: Value = serde_json::from_slice(&sarif.stdout).unwrap();
    let region = &value["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["region"];
    assert_eq!(region["startLine"], 2);
    assert_eq!(region["startColumn"], 12);
    assert_eq!(region["charLength"], 3);
    let sarif_file = f.file("result.sarif", std::str::from_utf8(&sarif.stdout).unwrap());
    let first = f.file(
        "cellar.txt",
        "Zoë was in the cellar. Someone locked the door.",
    );
    let second = f.file("garden.txt", "Zoë entered the garden.");
    let conflict = run(&[
        "--format",
        "sarif",
        "--characters",
        "Zoë",
        "--rules",
        "NL-CONTINUITY-LOCKED",
        &first,
        &second,
    ]);
    assert!(conflict.status.success());
    let conflict_value: Value = serde_json::from_slice(&conflict.stdout).unwrap();
    let related =
        &conflict_value["runs"][0]["results"][0]["relatedLocations"][0]["physicalLocation"];
    assert!(
        related["artifactLocation"]["uri"]
            .as_str()
            .unwrap()
            .ends_with("cellar.txt")
    );
    assert_eq!(
        related["region"]["snippet"]["text"],
        "Zoë was in the cellar."
    );
    assert_eq!(
        conflict_value["runs"][0]["results"][0]["relatedLocations"][1]["physicalLocation"]["region"]
            ["snippet"]["text"],
        "Someone locked the door."
    );
    let conflict_file = f.file(
        "conflict.sarif",
        std::str::from_utf8(&conflict.stdout).unwrap(),
    );
    let strict = f.file(
        "strict.json",
        r#"{"nli":"required","rules":["NL-CONTINUITY-NLI"]}"#,
    );
    assert_eq!(
        run(&["--config", &strict, &first, &second]).status.code(),
        Some(2)
    );
    validate_sarif(&[&sarif_file, &conflict_file]);
    let terminal = run(&["--rules", rule, &file]);
    assert!(
        String::from_utf8(terminal.stdout)
            .unwrap()
            .contains(":2:12:")
    );
}

#[test]
fn manifests_preserve_ids_overlay_controls_and_export_resolved_sarif() {
    let f = Fixture::new();
    f.file(
        "cellar Zoë.txt",
        "Zoë was in the cellar. Someone locked the door.",
    );
    let garden = f.file("garden.txt", "Zoë entered the garden.");
    let manifest=f.file("manuscript.json",r#"{"documentId":"quality-novel","knownCharacters":["Zoë"],"configuration":{"chronology":true,"rules":["NL-CONTINUITY-LOCKED"],"thresholds":{"maxGrade":18}},"scenes":[{"id":"garden","path":"garden.txt","chronology":{"key":"02","lane":"main"}},{"id":"cellar","path":"cellar Zoë.txt","chronology":{"key":"01","lane":"main"}}]}"#);
    let config = f.file("overlay.json", r#"{"thresholds":{"minSceneWords":1}}"#);
    let before = run(&[
        "--manifest",
        &manifest,
        "--config",
        &config,
        "--format=json",
    ]);
    assert!(
        before.status.success(),
        "{}",
        String::from_utf8_lossy(&before.stderr)
    );
    let json: Value = serde_json::from_slice(&before.stdout).unwrap();
    assert_eq!(
        json["report"]["sceneIds"],
        serde_json::json!(["garden", "cellar"])
    );
    assert_eq!(
        json["report"]["configuration"]["thresholds"]["maxGrade"].as_f64(),
        Some(18.0)
    );
    assert_eq!(json["report"]["findings"][0]["scope"]["sceneId"], "garden");
    assert!(
        json["sources"]["cellar"]
            .as_str()
            .unwrap()
            .contains("%C3%AB")
    );
    let baseline = f.file(
        "baseline-manifest.json",
        std::str::from_utf8(&before.stdout).unwrap(),
    );
    fs::write(
        &garden,
        "Zoë escaped from the cellar. Zoë entered the garden.",
    )
    .unwrap();
    let output = run(&[
        "--manifest",
        &manifest,
        "--baseline",
        &baseline,
        "--format",
        "sarif",
        "--fail-on",
        "warning",
    ]);
    assert!(output.status.success());
    let sarif: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(sarif["runs"][0]["results"][0]["baselineState"], "absent");
    assert_eq!(
        sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["region"]["charOffset"],
        0
    );
    let path = f.file(
        "resolved.sarif",
        std::str::from_utf8(&output.stdout).unwrap(),
    );
    validate_sarif(&[&path]);
}
#[test]
fn thresholds_include_omitted_results_and_unassessed_baselines_are_not_clean() {
    let f = Fixture::new();
    let file = f.file(
        "many.txt",
        "Zoë saw saw a bird. Zoë went went home. Zoë came came back.",
    );
    let config = f.file(
        "limits.json",
        r#"{"rules":["NL-GRAMMAR-REPETITION"],"limits":{"maxFindings":1}}"#,
    );
    let capped = run(&[
        "--config",
        &config,
        "--format",
        "json",
        "--fail-on",
        "warning",
        "--max-findings",
        "2",
        &file,
    ]);
    assert_eq!(capped.status.code(), Some(1));
    let value: Value = serde_json::from_slice(&capped.stdout).unwrap();
    assert_eq!(value["report"]["summary"]["severityCounts"]["warning"], 3);
    assert_eq!(value["report"]["findings"].as_array().unwrap().len(), 1);
    let baseline = f.file("capped.json", std::str::from_utf8(&capped.stdout).unwrap());
    let changed = run(&[
        "--baseline",
        &baseline,
        "--new-only",
        "--format",
        "json",
        "--rules",
        "NL-GRAMMAR-REPETITION",
        &file,
    ]);
    assert_eq!(changed.status.code(), Some(2));
    let diff: Value = serde_json::from_slice(&changed.stdout).unwrap();
    assert_eq!(
        diff["comparison"]["unassessed"].as_array().unwrap().len(),
        2
    );
    assert!(run(&["--version"]).status.success());
    assert!(
        String::from_utf8(run(&["--list-rules"]).stdout)
            .unwrap()
            .contains("Repeated word")
    );
    assert_eq!(run(&["--unknown"]).status.code(), Some(2));
}

#[test]
fn suppressed_and_unassessed_sarif_preserve_status_and_validate() {
    let f = Fixture::new();
    let file = f.file("opening.md", "😀 Zoë saw saw the bird.");
    let unavailable = run(&[
        "--rules",
        "NL-GRAMMAR-CAPITALIZATION",
        "--require-complete",
        &file,
    ]);
    assert_eq!(unavailable.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&unavailable.stderr).contains("adapter-only"));
    let selected = run(&[
        "--rules",
        "NL-GRAMMAR-REPETITION",
        "--require-complete",
        &file,
    ]);
    assert_eq!(
        selected.status.code(),
        Some(0),
        "unselected structure must not fail strict coverage: {}",
        String::from_utf8_lossy(&selected.stderr)
    );
    let baseline_output = run(&[
        "--format",
        "json",
        "--rules",
        "NL-GRAMMAR-REPETITION",
        &file,
    ]);
    assert!(baseline_output.status.success());
    let baseline = f.file(
        "baseline.json",
        std::str::from_utf8(&baseline_output.stdout).unwrap(),
    );
    let suppressed = run(&[
        "--format",
        "sarif",
        "--rules",
        "NL-GRAMMAR-REPETITION",
        "--suppress",
        "NL-GRAMMAR-REPETITION",
        &file,
    ]);
    assert!(suppressed.status.success());
    let json: Value = serde_json::from_slice(&suppressed.stdout).unwrap();
    assert_eq!(
        json["runs"][0]["results"][0]["suppressions"][0]["status"],
        "accepted"
    );
    let suppressed_file = f.file(
        "suppressed.sarif",
        std::str::from_utf8(&suppressed.stdout).unwrap(),
    );
    let disabled = f.file("disabled.json", r#"{"rules":[]}"#);
    let uncertain = run(&[
        "--config",
        &disabled,
        "--baseline",
        &baseline,
        "--new-only",
        "--format",
        "sarif",
        &file,
    ]);
    assert_eq!(uncertain.status.code(), Some(2));
    let json: Value = serde_json::from_slice(&uncertain.stdout).unwrap();
    assert_eq!(
        json["runs"][0]["invocations"][0]["executionSuccessful"],
        false
    );
    assert_eq!(json["runs"][0]["results"][0]["kind"], "review");
    assert!(json["runs"][0]["results"][0].get("baselineState").is_none());
    let uncertain_file = f.file(
        "unassessed.sarif",
        std::str::from_utf8(&uncertain.stdout).unwrap(),
    );
    validate_sarif(&[&suppressed_file, &uncertain_file]);
}

#[test]
fn legacy_report_population_imports_but_new_only_requires_known_counts() {
    let f = Fixture::new();
    let file = f.file("old-report.txt", "She saw saw the bird.");
    let before = run(&[
        "--format",
        "json",
        "--rules",
        "NL-GRAMMAR-REPETITION",
        &file,
    ]);
    let mut legacy: Value = serde_json::from_slice(&before.stdout).unwrap();
    legacy["report"]["schemaVersion"] = "1".into();
    legacy["report"].as_object_mut().unwrap().remove("summary");
    let baseline = f.file("schema1.json", &serde_json::to_string(&legacy).unwrap());
    fs::write(&file, "She saw saw the bird. He ran ran through town.").unwrap();
    let args = [
        "--format",
        "json",
        "--rules",
        "NL-GRAMMAR-REPETITION",
        "--baseline",
        &baseline,
        &file,
    ];
    let ordinary = run(&args);
    assert_eq!(ordinary.status.code(), Some(0));
    let imported: Value = serde_json::from_slice(&ordinary.stdout).unwrap();
    assert_eq!(
        imported["comparison"]["persistent"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        imported["comparison"]["introduced"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(imported["comparison"]["complete"], false);
    let mut strict = args.to_vec();
    strict.push("--new-only");
    let gated = run(&strict);
    assert_eq!(gated.status.code(), Some(2));
    let gated: Value = serde_json::from_slice(&gated.stdout).unwrap();
    assert_eq!(gated["comparison"], imported["comparison"]);
}

#[test]
fn selected_unavailable_model_cannot_pass_new_only_or_required_coverage() {
    let f = Fixture::new();
    let first = f.file("first.txt", "Alice stayed in the cellar.");
    let second = f.file("second.txt", "Alice left the cellar.");
    let config = f.file(
        "nli.json",
        r#"{"rules":["NL-CONTINUITY-NLI"],"nli":"fallback"}"#,
    );
    let common = ["--format", "json", "--config", &config, &first, &second];
    let output = run(&common);
    assert_eq!(output.status.code(), Some(0));
    let baseline = f.file(
        "baseline.json",
        std::str::from_utf8(&output.stdout).unwrap(),
    );
    let mut new_only = common.to_vec();
    new_only.extend(["--baseline", &baseline, "--new-only"]);
    let gated = run(&new_only);
    assert_eq!(gated.status.code(), Some(2));
    let value: Value = serde_json::from_slice(&gated.stdout).unwrap();
    assert!(
        value["comparison"]["unassessed"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(value["comparison"]["complete"], false);
    let mut strict = common.to_vec();
    strict.push("--require-complete");
    assert_eq!(run(&strict).status.code(), Some(2));
    let required = f.file(
        "required.json",
        r#"{"rules":["NL-CONTINUITY-NLI"],"nli":"required"}"#,
    );
    assert_eq!(
        run(&["--config", &required, &first, &second]).status.code(),
        Some(2)
    );
}

#[test]
fn model_options_require_a_pair_and_explicit_nli_selection() {
    let f = Fixture::new();
    let file = f.file("source.txt", "Alice stayed.");
    let missing = run(&["--model-root", "models", &file]);
    assert_eq!(missing.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&missing.stderr).contains("supplied together"));
    let disabled = run(&[
        "--model-root",
        "models",
        "--model-profile",
        "pins.json",
        &file,
    ]);
    assert_eq!(disabled.status.code(), Some(2));
    let message = String::from_utf8_lossy(&disabled.stderr);
    assert!(message.contains(if cfg!(feature = "onnx") {
        "require selected NLI"
    } else {
        "onnx feature"
    }));
}

#[test]
#[cfg(feature = "onnx")]
fn cli_loads_pinned_nli_and_reports_identity_then_rejects_tampering() {
    use narrative_lens::onnx::{
        checked::ModelKind,
        model::{ModelManifest, ModelProfile},
    };
    use sha2::{Digest, Sha256};
    let f = Fixture::new();
    let first = f.file("first.txt", "Alice stayed in the cellar.");
    let second = f.file("second.txt", "Alice left the cellar.");
    let config = f.file(
        "config.json",
        r#"{"rules":["NL-CONTINUITY-NLI"],"nli":"required"}"#,
    );
    let root = f.0.join("models");
    let directory = root.join("nli-onnx");
    fs::create_dir_all(&directory).unwrap();
    let model = include_bytes!("fixtures/models/valid-nli.onnx");
    fs::write(directory.join("model.onnx"), model).unwrap();
    let vocab = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab([("[UNK]".into(), 0)].into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let tokenizer = tokenizers::Tokenizer::new(vocab).to_string(false).unwrap();
    fs::write(directory.join("tokenizer.json"), &tokenizer).unwrap();
    let manifest = ModelManifest {
        schema_version: 1,
        kind: ModelKind::Nli,
        profile: ModelProfile::HuggingFaceNli3V1,
        model_sha256: format!("{:x}", Sha256::digest(model)),
        tokenizer_sha256: format!("{:x}", Sha256::digest(tokenizer.as_bytes())),
        max_tokens: 512,
        padding_token_id: 0,
        labels: ["contradiction", "neutral", "entailment"]
            .map(String::from)
            .into(),
    };
    let profile = f.file("pins.json", &serde_json::to_string(&manifest).unwrap());
    let root = root.to_str().unwrap();
    let args = [
        "--config",
        &config,
        "--format",
        "json",
        "--model-root",
        root,
        "--model-profile",
        &profile,
        &first,
        &second,
    ];
    let output = run(&args);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        value["report"]["modelIdentities"][0]["verification"],
        "verified_manifest_v1"
    );
    assert_eq!(
        value["report"]["modelIdentities"][0]["manifest"]["modelSha256"],
        manifest.model_sha256
    );
    assert_eq!(value["report"]["findings"].as_array().unwrap().len(), 1);
    fs::write(directory.join("model.onnx"), b"tampered").unwrap();
    let rejected = run(&args);
    assert_eq!(rejected.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("model_digest_mismatch"));
}
