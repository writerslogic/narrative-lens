use narrative_lens::{
    craft::readability::{analyze_smog, compute_readability},
    substrate::utils::AnalysisAvailability,
};

#[test]
fn smog_uses_the_published_regression_and_thirty_distinct_sentences() {
    let text = "Tomorrow. ".repeat(30);
    let result = analyze_smog(&text);
    assert_eq!(result.availability, AnalysisAvailability::Complete);
    assert_eq!(result.polysyllable_count, 30);
    assert_eq!(result.grade, Some(8.84));
    assert_eq!(compute_readability(&text).smog, 8.84);
    assert!(compute_readability(&text).smog_available);
    assert_eq!(result.sampled_sentences.len(), 30);
    let mut previous_end = 0;
    for sample in result.sampled_sentences {
        assert_eq!(&text[sample.byte_start..sample.byte_end], "Tomorrow.");
        assert!(sample.byte_start >= previous_end);
        previous_end = sample.byte_end;
    }
}

#[test]
fn smog_short_and_unterminated_samples_are_unavailable() {
    for sentences in [29, 30, 31] {
        let text = "Tomorrow. ".repeat(sentences);
        let result = analyze_smog(&text);
        assert_eq!(result.grade.is_some(), sentences >= 30);
        assert_eq!(
            result.sampled_sentences.len(),
            if sentences >= 30 { 30 } else { 0 }
        );
    }
    let text = format!("{}Tomorrow", "Tomorrow. ".repeat(29));
    let result = analyze_smog(&text);
    assert_eq!(result.total_sentence_count, 30);
    assert_eq!(result.eligible_sentence_count, 29);
    assert_eq!(result.grade, None);
    let legacy = compute_readability(&text);
    assert_eq!(legacy.smog, 0.0);
    assert!(!legacy.smog_available);
    for text in [". ".repeat(30), "123. ".repeat(30), "“.” ".repeat(30)] {
        assert_eq!(analyze_smog(&text).grade, None);
        assert!(!compute_readability(&text).smog_available);
    }
}

#[test]
fn smog_samples_the_beginning_middle_and_end_without_whole_text_scaling() {
    let mut sentences = vec!["The cat sat."; 60];
    for sentence in &mut sentences[10..20] {
        *sentence = "Tomorrow tomorrow tomorrow.";
    }
    for sentence in &mut sentences[40..50] {
        *sentence = "Tomorrow tomorrow tomorrow.";
    }
    let text = sentences.join(" ");
    let result = analyze_smog(&text);
    assert_eq!(result.polysyllable_count, 0);
    assert_eq!(result.grade, Some(3.13));
    let selected: Vec<_> = result
        .sampled_sentences
        .iter()
        .map(|range| &text[range.byte_start..range.byte_end])
        .collect();
    assert_eq!(selected, vec!["The cat sat."; 30]);
    let spans = narrative_lens::substrate::utils::sentence_spans(&text);
    assert_eq!(result.sampled_sentences[10].byte_start, spans[25].0);
    assert_eq!(result.sampled_sentences[20].byte_start, spans[50].0);
    let unicode = "“Tomorrow.”\r\n".repeat(30);
    assert_eq!(analyze_smog(&unicode).grade, Some(8.84));
    for range in analyze_smog(&unicode).sampled_sentences {
        assert_eq!(&unicode[range.byte_start..range.byte_end], "“Tomorrow.”");
    }
}

#[test]
fn dale_chall_uses_the_pinned_full_resource_and_correction_boundary() {
    let resource = include_str!("../src/craft/data/dale_chall_1995.txt");
    let words: Vec<_> = resource.lines().collect();
    assert_eq!(words.len(), 2942);
    assert!(words.windows(2).all(|pair| pair[0] < pair[1]));
    for word in [
        "aboard",
        "absent",
        "airplane",
        "banana",
        "butterfly",
        "calendar",
        "cellar",
    ] {
        assert!(words.binary_search(&word).is_ok());
        assert_eq!(compute_readability(&format!("{word}.")).dale_chall, 0.05);
    }
    for difficult in [0, 1, 2] {
        let text = format!(
            "{}{}.",
            "the ".repeat(20 - difficult),
            "quorvex ".repeat(difficult)
        );
        let score = compute_readability(&text).dale_chall;
        let expected = [0.99, 1.78, 6.21][difficult];
        assert_eq!(score, expected, "difficult words={difficult}");
    }
}
