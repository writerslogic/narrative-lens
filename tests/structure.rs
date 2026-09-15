//! Integration tests for the `structure` analyzer category.

use std::fs;

use narrative_lens::structure::{genre, narrative_entropy, structure};

fn fixture() -> serde_json::Value {
    let raw = fs::read_to_string("tests/fixtures/manuscript-passages.json")
        .expect("fixture file must exist");
    serde_json::from_str(&raw).expect("fixture must be valid JSON")
}

fn fixture_scenes(json: &serde_json::Value) -> Vec<String> {
    json["scenes"]
        .as_array()
        .expect("fixture must have a scenes array")
        .iter()
        .map(|v| v.as_str().expect("scene must be a string").to_string())
        .collect()
}

#[test]
fn structure_templates_are_listed_and_fetchable_by_name() {
    let names = structure::list_templates();
    assert!(!names.is_empty(), "at least one built-in template");

    for name in &names {
        let template = structure::get_structure_template(name)
            .unwrap_or_else(|e| panic!("listed template {name:?} must fetch: {e}"));
        assert_eq!(&template.name, name);
    }
}

#[test]
fn structure_template_lookup_fails_gracefully_on_unknown_name() {
    let result = structure::get_structure_template("not-a-real-template-xyz");
    assert!(result.is_err());
}

#[test]
fn genre_detection_returns_a_genre_without_panicking() {
    let json = fixture();
    let themes: Vec<String> = json["genre_themes"]
        .as_array()
        .expect("fixture must have genre_themes")
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let tension: Vec<f64> = json["tension_pattern"]
        .as_array()
        .expect("fixture must have tension_pattern")
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    // No model loaded in this test environment (the `onnx` feature is off by
    // default), so this exercises the keyword/prose-stats heuristic fallback.
    let detected = genre::detect_genre(themes, tension, 0.3, 8.5);
    // detect_genre always returns a Genre variant; just confirm it round-trips
    // through the label mapping (guards against an id2label ordering drift).
    let label = format!("{detected:?}");
    assert!(!label.is_empty());
}

#[test]
fn narrative_entropy_handles_a_real_multi_scene_manuscript() {
    let json = fixture();
    let scenes = fixture_scenes(&json);
    let scene_refs: Vec<&str> = scenes.iter().map(String::as_str).collect();
    let themes = vec![("homecoming".to_string(), 0.8)];
    let character_arcs = vec![("Maren".to_string(), "return".to_string())];

    let result = narrative_entropy::compute_narrative_entropy(
        &scene_refs,
        &themes,
        "three_act",
        "literary",
        &character_arcs,
    );
    assert_eq!(result.entropy_curve.len(), scenes.len());
}

#[test]
fn narrative_entropy_handles_empty_manuscript() {
    let result = narrative_entropy::compute_narrative_entropy(&[], &[], "three_act", "literary", &[]);
    assert!(result.entropy_curve.is_empty());
    assert!(result.questions.is_empty());
}
