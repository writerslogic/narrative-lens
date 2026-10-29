use narrative_lens::craft::dialogue::{DialogueExtractionOptions, extract_dialogue_checked};
fn quotes(text: &str) -> Vec<String> {
    extract_dialogue_checked(
        text,
        0,
        &["Alice".into(), "Bob".into()],
        &DialogueExtractionOptions::default(),
    )
    .unwrap()
    .lines
    .into_iter()
    .map(|line| line.line.quote)
    .collect()
}

#[test]
fn token_separated_curly_contraction_suffixes_preserve_complete_quotation() {
    for content in [
        "How brave they ’ll think me !",
        "I ’ve fallen by now .",
        "I ’m certain we ’re early .",
        "Dinah ’s clock is gone .",
        "I ’d say they ’ve gone .",
        "I would n’t say that .",
    ] {
        assert_eq!(quotes(&format!("‘ {content} ’ Alice said .")), [content]);
    }
}

#[test]
fn fully_separated_apostrophe_tokens_stay_inside_supported_clitics() {
    for content in [
        "I ' m waiting .",
        "We ' re waiting .",
        "They ' ll wait .",
        "Mira ' s clock .",
        "I can ' t wait .",
    ] {
        assert_eq!(quotes(&format!("' {content} ' Alice said .")), [content]);
    }
}

#[test]
fn leading_elisions_do_not_open_phantom_quotes() {
    let text = "'Tis evening. \"Bring 'em here,\" Alice said. \"Wait,\" Bob said.";
    assert_eq!(quotes(text), ["Bring 'em here,", "Wait,"]);
}

#[test]
fn closing_straight_quote_at_line_start_is_not_paragraph_continuation() {
    let text = "\"First line.\n\" Alice said.\n\"Second,\" Bob said.";
    assert_eq!(quotes(text), ["First line.", "Second,"]);
}

#[test]
fn malformed_inner_quote_does_not_hide_balanced_outer_and_later_quotes() {
    let text = "\"Read 'manual,\" Alice said. \"Wait,\" Bob said.";
    let result = extract_dialogue_checked(
        text,
        0,
        &["Alice".into(), "Bob".into()],
        &DialogueExtractionOptions::default(),
    )
    .unwrap();
    assert_eq!(
        result
            .lines
            .iter()
            .map(|line| line.line.quote.as_str())
            .collect::<Vec<_>>(),
        ["Read 'manual,", "Wait,"]
    );
    assert!(
        result
            .diagnostics
            .contains(&"unbalanced_quote_delimiters".into())
    );
}

#[test]
fn orphan_opening_does_not_swallow_balanced_other_delimiter_pairs() {
    let text = "An orphan ' marker. \"First,\" Alice said. \"Second,\" Bob said.";
    assert_eq!(quotes(text), ["First,", "Second,"]);
}

#[test]
fn recovered_and_spaced_quotations_preserve_original_utf16_source() {
    let text =
        "🜂İ 'Tis late. ‘ We ’re here and I ’ll wait . ’ Alice said . \"Bring 'em,\" Bob said .";
    let result = extract_dialogue_checked(
        text,
        11,
        &["Alice".into(), "Bob".into()],
        &DialogueExtractionOptions::default(),
    )
    .unwrap();
    assert_eq!(result.lines.len(), 2);
    for line in result.lines {
        assert_eq!(line.line.scene_id, 11);
        assert_eq!(
            &text[line.range.byte_start..line.range.byte_end],
            line.line.quote
        );
        assert_eq!(
            text[..line.range.byte_start].encode_utf16().count() as u32,
            line.range.utf16_start
        );
        assert_eq!(
            text[..line.range.byte_end].encode_utf16().count() as u32,
            line.range.utf16_end
        );
        assert!(line.quotation_range.is_some());
    }
}

#[test]
fn completed_nested_pair_budget_is_checked_before_returning_large_outer_quote() {
    for pairs in [319999, 320000, 320001] {
        let text = format!("‘ {} ’ Alice said.", "\"x\" ".repeat(pairs - 1));
        let result = extract_dialogue_checked(
            &text,
            0,
            &["Alice".into()],
            &DialogueExtractionOptions::default(),
        );
        assert_eq!(result.is_ok(), pairs <= 320000);
        if let Ok(result) = result {
            assert_eq!(result.lines.len(), 1);
        }
    }
}
