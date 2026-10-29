use narrative_lens::{
    craft::voice::build_voice_profile, structure::opening_analysis::*,
    substrate::utils::AnalysisAvailability,
};

#[test]
fn substring_collisions_do_not_create_action_backstory_or_promises() {
    let result = analyze_opening_checked("A stranger waited quietly.", "literary", None).unwrap();
    assert!(result.hook_type.is_none());
    assert!(
        !result
            .cues
            .iter()
            .any(|c| c.kind == OpeningCueKind::ActionWord)
    );
    let result = analyze_opening_checked(
        "An unoriginally named stranger waited. His glove was red.",
        "literary",
        None,
    )
    .unwrap();
    assert_eq!(result.backstory_ratio, 0.0);
    assert!(result.promises_made.is_empty());
    let topics =
        analyze_opening_checked("A secret journey ended in love and death.", "romance", None)
            .unwrap();
    assert!(topics.promises_made.is_empty());
    assert!(
        topics
            .cues
            .iter()
            .any(|c| c.kind == OpeningCueKind::TopicWord)
    );
    assert!(!topics.quality_assessed);
    assert!(topics.agent_perspective.contains("unassessed"));
}

#[test]
fn original_unicode_crlf_line_sentence_and_paragraph_ranges_are_distinct() {
    let text = "  🜂İ Dr. Vale waited. Another sentence.  \r\n\r\n  ‘I’m ready,’ Alice said.  ";
    let result = analyze_opening_checked(text, "literary", None).unwrap();
    let line = result.first_line_range.as_ref().unwrap();
    let sentence = result.first_sentence_range.as_ref().unwrap();
    assert_eq!(
        &text[line.byte_start..line.byte_end],
        "🜂İ Dr. Vale waited. Another sentence."
    );
    assert_eq!(
        &text[sentence.byte_start..sentence.byte_end],
        "🜂İ Dr. Vale waited."
    );
    assert_eq!(result.paragraphs.len(), 2);
    assert_eq!(result.paragraphs[0].sentence_count, 2);
    assert_eq!(result.paragraphs[1].quotation_candidate_count, 1);
    assert_eq!(result.dialogue_onset_paragraph, Some(1));
    for cue in &result.cues {
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
    let lf = analyze_opening_checked(&text.replace("\r\n", "\n"), "literary", None).unwrap();
    assert_eq!(result.paragraph_word_counts, lf.paragraph_word_counts);
}

#[test]
fn abbreviation_aware_sentences_and_paired_quotes_define_measurements() {
    let result =
        analyze_opening_checked("Dr. Vale had been waiting. She left.", "literary", None).unwrap();
    assert_eq!(result.paragraphs[0].sentence_count, 2);
    assert_eq!(result.paragraphs[0].backstory_sentence_count, 1);
    assert_eq!(result.backstory_ratio, 0.5);
    let quoted = analyze_opening_checked("‘Hello,’ Alice said.", "literary", None).unwrap();
    assert_eq!(quoted.dialogue_onset_paragraph, Some(0));
    let unpaired = analyze_opening_checked("“Unclosed words.", "literary", None).unwrap();
    assert!(unpaired.dialogue_onset_paragraph.is_none());
    assert!(
        unpaired
            .diagnostics
            .iter()
            .any(|d| d.contains("unbalanced"))
    );
}

#[test]
fn typed_voice_baseline_uses_actual_values_and_arbitrary_vectors_stay_unsupported() {
    let text = "We wait quietly. Will you stay?";
    let identical = build_voice_profile("Narrator", &[text.into()]);
    let other = build_voice_profile("Narrator", &["I don't! Don't! Don't!".into()]);
    let a = analyze_opening_checked(text, "literary", Some(&identical)).unwrap();
    let b = analyze_opening_checked(text, "literary", Some(&other)).unwrap();
    assert_eq!(a.voice_comparison.unwrap().overall, 0.0);
    assert!(b.voice_comparison.unwrap().overall > 0.0);
    assert!(b.voice_sample.word_count > 0);
    let mut invalid = other;
    invalid.formality_score = f64::NAN;
    assert!(analyze_opening_checked(text, "literary", Some(&invalid)).is_err());
    let legacy = analyze_opening(text, "literary", Some(&[1.0]));
    assert!(legacy.voice_comparison.is_none());
    assert!(legacy.diagnostics.iter().any(|d| d.contains("unsupported")));
    assert_eq!(
        legacy.voice_establishment,
        analyze_opening(text, "literary", None).voice_establishment
    );
    let encoded =
        serde_json::to_value(analyze_opening_checked(text, "literary", Some(&identical)).unwrap())
            .unwrap();
    assert!(serde_json::from_value::<OpeningAnalysisResult>(encoded).is_ok());
}

#[test]
fn empty_input_and_resource_limits_abstain_or_fail_without_successful_judgments() {
    let empty = analyze_opening_checked("", "literary", None).unwrap();
    assert_eq!(empty.availability, AnalysisAvailability::Unavailable);
    assert!(empty.first_sentence_range.is_none());
    for bytes in [16 * 1024 * 1024 - 1, 16 * 1024 * 1024, 16 * 1024 * 1024 + 1] {
        assert_eq!(
            analyze_opening_checked(&" ".repeat(bytes), "literary", None).is_ok(),
            bytes <= 16 * 1024 * 1024
        );
    }
    for count in [9999, 10000, 10001] {
        assert_eq!(
            analyze_opening_checked(&vec!["x"; count].join("\n\n"), "literary", None).is_ok(),
            count <= 10000
        );
    }
    for count in [19999, 20000, 20001] {
        assert_eq!(
            analyze_opening_checked(&"cold ".repeat(count), "literary", None).is_ok(),
            count <= 20000
        );
    }
    let failed = analyze_opening(&"x".repeat(16 * 1024 * 1024 + 1), "literary", None);
    assert_eq!(failed.availability, AnalysisAvailability::Unavailable);
    assert!(
        failed
            .diagnostics
            .iter()
            .any(|d| d.contains("invalid_opening_input"))
    );
}
