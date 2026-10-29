use narrative_lens::craft::dialogue_realism::analyze_dialogue_realism;
use narrative_lens::craft::voice::{analyze_character_voices, build_voice_profile};

#[test]
fn recurring_phrases_never_cross_turn_boundaries() {
    let profile = build_voice_profile("Alice", &vec!["quartz".into(); 4]);
    assert!(
        profile.unique_phrases.is_empty(),
        "{:?}",
        profile.unique_phrases
    );
}

#[test]
fn zero_word_turns_do_not_dilute_measured_question_rates() {
    let bare = build_voice_profile("Alice", &["Ready?".into()]);
    let padded = build_voice_profile("Alice", &["Ready?".into(), " ".into(), "...".into()]);
    assert_eq!(bare.question_rate, padded.question_rate);
}

#[test]
fn typographic_and_straight_apostrophes_have_equivalent_contraction_features() {
    let straight = build_voice_profile("Alice", &["I'm sure we don't know.".into()]);
    let curly = build_voice_profile("Alice", &["I’m sure we don’t know.".into()]);
    assert_eq!(straight.contraction_rate, curly.contraction_rate);
    assert_eq!(straight.vocabulary_richness, curly.vocabulary_richness);
}

#[test]
fn phrases_seen_once_by_another_speaker_are_not_unique() {
    let result = analyze_character_voices(vec![
        ("Alice".into(), vec!["purple clocks ring".into(); 3]),
        ("Bob".into(), vec!["purple clocks ring".into()]),
    ]);
    assert!(result.profiles[0].unique_phrases.is_empty());
}

#[test]
fn no_sample_does_not_report_three_quarter_realism() {
    let result = analyze_dialogue_realism(&[], &[]);
    assert_eq!(result.overall_realism, 0.5);
}

use narrative_lens::craft::dialogue_realism::{DialogueCueKind, analyze_dialogue_realism_checked};
use narrative_lens::craft::voice::{
    analyze_character_voices_checked, build_voice_profile_checked, compare_voices_checked,
    measure_voice_drift, measure_voice_drift_checked, measure_voice_drift_from_text_checked,
};
use narrative_lens::substrate::utils::AnalysisAvailability;

#[test]
fn coverage_distinguishes_no_words_from_valid_measured_zero() {
    let no_words = build_voice_profile_checked("Alice", &["...".into(), " ".into()]).unwrap();
    assert_eq!(no_words.availability, AnalysisAvailability::Unavailable);
    assert_eq!(no_words.sample.input_turn_count, 2);
    assert_eq!(no_words.sample.turn_count, 0);
    let zero_questions = build_voice_profile_checked("Alice", &["We wait.".into()]).unwrap();
    assert_eq!(zero_questions.availability, AnalysisAvailability::Complete);
    assert_eq!(zero_questions.sample.word_count, 2);
    assert_eq!(zero_questions.sample.sentence_count, 1);
    assert_eq!(zero_questions.profile.question_rate, 0.0);
    assert!(
        compare_voices_checked(&[no_words.profile.clone(), build_voice_profile("Bob", &[])])
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        measure_voice_drift_checked(&no_words.profile, &no_words.profile)
            .unwrap()
            .availability,
        AnalysisAvailability::Unavailable
    );
    assert!(
        measure_voice_drift_from_text_checked("Alice", &["We wait.".into()], &[])
            .unwrap()
            .per_feature
            .is_empty()
    );
    let analysis = analyze_character_voices_checked(&[("Alice".into(), vec![])]).unwrap();
    assert_eq!(analysis.availability, AnalysisAvailability::Unavailable);
    assert!(!analysis.distinct_voices);
}

#[test]
fn lexical_cues_preserve_original_turn_coordinates_and_do_not_assert_realism() {
    let text = "🜂İ As I’m sure you’re aware, we wait. Furthermore, as you knowledge.";
    let turns = vec![
        (7, "ALICE".into(), text.into()),
        (12, "Bob".into(), "...".into()),
        (17, "Charlie".into(), "I wait.".into()),
    ];
    let result = analyze_dialogue_realism_checked(&turns, &["Alice", "Bob"]).unwrap();
    assert_eq!(result.sample.input_turn_count, 3);
    assert_eq!(result.sample.matched_turn_count, 2);
    assert_eq!(result.sample.eligible_turn_count, 1);
    assert_eq!(result.sample.empty_turn_count, 1);
    assert_eq!(result.sample.unmatched_turn_count, 1);
    assert_eq!(
        result.sample.word_count,
        result.voice_samples[0].sample.word_count
    );
    assert_eq!(result.cues.len(), 2);
    assert_eq!(result.exposition_cue_count, 1);
    for cue in &result.cues {
        assert_eq!(cue.turn_index, 0);
        assert_eq!(cue.scene, 7);
        assert_eq!(cue.character, "Alice");
        assert_eq!(&text[cue.range.byte_start..cue.range.byte_end], cue.quote);
        assert_eq!(
            text[..cue.range.byte_start].encode_utf16().count() as u32,
            cue.range.utf16_start
        );
        assert_eq!(
            text[..cue.range.byte_end].encode_utf16().count() as u32,
            cue.range.utf16_end
        );
    }
    assert_eq!(result.cues[0].kind, DialogueCueKind::ExpositionMarker);
    assert_eq!(result.cues[1].kind, DialogueCueKind::FormalMarker);
    assert_eq!(result.overall_realism, 0.5);
    assert!(!result.realism_assessed);
    assert!(result.power_dynamics.is_empty());
    assert!(result.issues.is_empty());
    assert_eq!(result.info_dump_count, 0);
    assert_eq!(
        result.voice_comparison_availability,
        AnalysisAvailability::Unavailable
    );
}

#[test]
fn exact_lexical_overlap_has_deterministic_cosine_without_identity_claims() {
    let turns = vec![
        (4, "Alice".into(), "red quartz, blue clocks. red!".into()),
        (15, "Bob".into(), "red quartz, blue clocks. red!".into()),
        (33, "Charlie".into(), "velvet mountains snow.".into()),
    ];
    let baseline = analyze_dialogue_realism_checked(&turns, &["Alice", "Bob", "Charlie"]).unwrap();
    assert!((baseline.voice_pairs[0].similarity - 1.0).abs() < 1e-12);
    assert_eq!(baseline.voice_pairs[1].similarity, 0.0);
    assert!(baseline.issues.is_empty());
    assert_eq!(
        baseline.voice_comparison_availability,
        AnalysisAvailability::Complete
    );
    let expected = serde_json::to_value(baseline).unwrap();
    for _ in 0..20 {
        assert_eq!(
            expected,
            serde_json::to_value(
                analyze_dialogue_realism_checked(&turns, &["Alice", "Bob", "Charlie"]).unwrap()
            )
            .unwrap()
        );
    }
    let reordered = analyze_dialogue_realism_checked(
        &[turns[2].clone(), turns[1].clone(), turns[0].clone()],
        &["Charlie", "Bob", "Alice"],
    )
    .unwrap();
    assert!(
        (reordered.voice_distinctiveness - expected["voiceDistinctiveness"].as_f64().unwrap())
            .abs()
            < 1e-15
    );
}

#[test]
fn invalid_numeric_profiles_fail_without_leaking_nonfinite_serialized_values() {
    let a = build_voice_profile("Alice", &["We wait.".into()]);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
        let mut b = a.clone();
        b.question_rate = value;
        assert!(compare_voices_checked(&[b.clone()]).is_err());
        assert!(measure_voice_drift_checked(&a, &b).is_err());
        let legacy = measure_voice_drift(&a, &b);
        assert_eq!(legacy.availability, AnalysisAvailability::Unavailable);
        assert!(serde_json::to_string(&legacy).is_ok());
    }
    for value in [0.0, 1.0] {
        let mut b = a.clone();
        b.question_rate = value;
        assert!(measure_voice_drift_checked(&a, &b).is_ok());
    }
    let b = build_voice_profile("Bob", &["We wait.".into()]);
    assert!(measure_voice_drift_checked(&a, &b).is_err());
    assert!(compare_voices_checked(&[a.clone(), a]).is_err());
}

#[test]
fn text_name_turn_and_cast_limits_cover_minus_equal_plus_one() {
    for bytes in [127, 128, 129] {
        let result = build_voice_profile_checked(&"a".repeat(bytes), &[]);
        assert_eq!(result.is_ok(), bytes <= 128);
    }
    for count in [255, 256, 257] {
        let data: Vec<_> = (0..count).map(|i| (format!("p{i}"), vec![])).collect();
        assert_eq!(
            analyze_character_voices_checked(&data).is_ok(),
            count <= 256
        );
    }
    for count in [9999, 10000, 10001] {
        assert_eq!(
            build_voice_profile_checked("Alice", &vec!["...".into(); count]).is_ok(),
            count <= 10000
        );
    }
    for bytes in [16 * 1024 * 1024 - 1, 16 * 1024 * 1024, 16 * 1024 * 1024 + 1] {
        assert_eq!(
            build_voice_profile_checked("Alice", &[" ".repeat(bytes)]).is_ok(),
            bytes <= 16 * 1024 * 1024
        );
    }
    for delta in [-1isize, 0, 1] {
        let text = " ".repeat(16 * 1024 * 1024);
        let lines = vec![
            text.clone(),
            text.clone(),
            text.clone(),
            " ".repeat(8 * 1024 * 1024),
            " ".repeat((8 * 1024 * 1024isize + delta) as usize),
        ];
        assert_eq!(
            build_voice_profile_checked("Alice", &lines).is_ok(),
            delta <= 0
        );
    }
    let error = build_voice_profile_checked(&"secret".repeat(20000), &[])
        .unwrap_err()
        .to_string();
    assert!(!error.contains("secret"));
}

#[test]
fn long_turn_and_cue_budget_boundaries_are_explicit() {
    for words in [149, 150, 151] {
        let text = vec!["quartz"; words].join(" ");
        let result =
            analyze_dialogue_realism_checked(&[(4, "Alice".into(), text)], &["Alice"]).unwrap();
        assert_eq!(
            result
                .cues
                .iter()
                .any(|cue| cue.kind == DialogueCueKind::LongTurn),
            words > 150
        );
    }
    for markers in [19998, 19999, 20000] {
        let text = "as you know; ".repeat(markers);
        let result = analyze_dialogue_realism_checked(&[(4, "Alice".into(), text)], &["Alice"]);
        assert_eq!(result.is_ok(), markers < 20000);
        if let Ok(result) = result {
            assert_eq!(result.cues.len(), markers + 1);
        }
    }
}
