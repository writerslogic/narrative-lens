//! Integration tests for the `craft` analyzer category, exercised against the
//! shared manuscript fixture rather than synthetic one-liners.

use std::fs;

use narrative_lens::craft::{grammar, prose_quality, readability, syntax_tension};

fn fixture_scenes() -> Vec<String> {
    let raw = fs::read_to_string("tests/fixtures/manuscript-passages.json")
        .expect("fixture file must exist");
    let json: serde_json::Value = serde_json::from_str(&raw).expect("fixture must be valid JSON");
    json["scenes"]
        .as_array()
        .expect("fixture must have a scenes array")
        .iter()
        .map(|v| v.as_str().expect("scene must be a string").to_string())
        .collect()
}

#[test]
fn readability_scores_a_real_passage_in_sane_ranges() {
    let scenes = fixture_scenes();
    let result = readability::compute_readability(&scenes[0]);

    assert!(result.word_count > 0);
    assert!(result.sentence_count > 0);
    assert!(result.syllable_count >= result.word_count);
    // A grade-school-reading-level heuristic on ordinary adult prose should
    // land somewhere plausible, not at either extreme -- this guards against
    // an off-by-one in the syllable/sentence counting silently zeroing FKGL.
    assert!(
        result.fkgl > 0.0 && result.fkgl < 20.0,
        "fkgl {} out of sane range",
        result.fkgl
    );
}

#[test]
fn readability_handles_empty_text_without_panicking() {
    let result = readability::compute_readability("");
    assert_eq!(result.word_count, 0);
    assert_eq!(result.sentence_count, 0);
}

#[test]
fn grammar_flags_a_duplicated_word() {
    let findings = grammar::check("She was was tired.");
    assert!(
        !findings.is_empty(),
        "a doubled word must produce at least one finding"
    );
}

#[test]
fn grammar_is_quiet_on_clean_prose() {
    let scenes = fixture_scenes();
    // Not asserting zero findings (grammar checking is heuristic), just that
    // it runs to completion over real prose without panicking.
    let _ = grammar::check(&scenes[2]);
}

#[test]
fn syntax_tension_analyzes_a_real_scene() {
    let scenes = fixture_scenes();
    let result = syntax_tension::analyze_scene_text(&scenes[1]);
    assert!(result.sentence_count > 0);
    assert!(result.score.is_finite() && result.score >= 0.0);
}

#[test]
fn prose_quality_collects_examples_across_scenes() {
    let scenes = fixture_scenes();
    // Must not panic across a small multi-scene manuscript; the analyzer is
    // heuristic, so we only assert it terminates and returns a well-formed Vec.
    let examples = prose_quality::collect_prose_examples(&scenes);
    assert!(examples.len() <= scenes.len() * 10);
}
