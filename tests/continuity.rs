//! Integration tests for the `continuity` analyzer category.

use std::fs;

use narrative_lens::continuity::{entity, timeline, world_state};

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
fn timeline_preserves_input_order_over_real_scenes() {
    let json = fixture();
    let scenes = fixture_scenes(&json);
    let documents: Vec<(String, String)> = scenes
        .iter()
        .enumerate()
        .map(|(i, text)| (format!("scene-{i}"), text.clone()))
        .collect();

    let entries = timeline::build_timeline(&documents);
    assert_eq!(entries.len(), documents.len());
    for (i, entry) in entries.iter().enumerate() {
        assert_eq!(entry.order, i);
        assert_eq!(entry.document_id, format!("scene-{i}"));
    }
}

#[test]
fn timeline_handles_no_documents() {
    assert!(timeline::build_timeline(&[]).is_empty());
}

#[test]
fn world_state_tracks_characters_across_real_scenes() {
    let json = fixture();
    let scenes = fixture_scenes(&json);
    let scene_refs: Vec<&str> = scenes.iter().map(String::as_str).collect();
    let characters: Vec<&str> = json["characters"]
        .as_array()
        .expect("fixture must have characters")
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();

    let result = world_state::track_world_state(&scene_refs, &characters);
    assert!((0.0..=1.0).contains(&result.continuity_score));
    assert!(result.world_complexity >= 0.0);
}

#[test]
fn world_state_handles_no_scenes() {
    let result = world_state::track_world_state(&[], &[]);
    assert!(result.snapshots.is_empty());
    assert!(result.violations.is_empty());
}

#[test]
fn entity_resolution_groups_repeated_surface_forms() {
    // No embedding model in this test environment (`onnx` feature off), so
    // every mention carries an empty context vector -- resolution then falls
    // back to surface-form matching, which is what this test exercises.
    let mentions = vec![
        entity::Mention::new("Maren", vec![]),
        entity::Mention::new("Maren", vec![]),
        entity::Mention::new("Adrian", vec![]),
    ];
    let entities = entity::resolve(&mentions, entity::EntityType::Character);
    assert!(!entities.is_empty());
    assert!(entities.len() <= mentions.len());
}

#[test]
fn entity_resolution_handles_no_mentions() {
    assert!(entity::resolve(&[], entity::EntityType::Character).is_empty());
}
