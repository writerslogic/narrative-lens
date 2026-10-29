use narrative_lens::craft::dialogue::*;

fn cast(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).into()).collect()
}

#[test]
fn explicit_actor_is_not_the_addressee_and_cast_order_is_irrelevant() {
    let text = r#""Hello," Bob said to Alice. "Leave," Alice replied."#;
    for names in [cast(&["Alice", "Bob"]), cast(&["Bob", "Alice"])] {
        let lines = extract_dialogue(text, 0, &names);
        assert_eq!(
            lines
                .iter()
                .map(|line| line.speaker.as_str())
                .collect::<Vec<_>>(),
            ["Bob", "Alice"]
        );
    }
}

#[test]
fn another_quotes_tag_does_not_attribute_an_untagged_quote() {
    let text = r#""Hello." Alice left. Bob said, "Goodbye.""#;
    let lines = extract_dialogue(text, 0, &cast(&["Alice", "Bob"]));
    assert_ne!(lines[0].method, "explicit_tag");
    assert_eq!(lines[1].speaker, "Bob");
}

#[test]
fn ambiguous_short_aliases_remain_unknown() {
    let lines = extract_dialogue(
        "John said, ‘Hello.’",
        0,
        &cast(&["John Brown", "John Smith"]),
    );
    assert_eq!(lines[0].speaker, "Unknown");
}

#[test]
fn nested_quotes_and_word_apostrophes_do_not_create_extra_turns() {
    let text = "“He called it ‘the devil’s bargain’ before he left,” Alice said.";
    let lines = extract_dialogue(text, 0, &cast(&["Alice"]));
    assert_eq!(lines.len(), 1);
    assert_eq!(
        lines[0].quote,
        "He called it ‘the devil’s bargain’ before he left,"
    );
}

#[test]
fn straight_single_quotes_support_internal_contractions() {
    let lines = extract_dialogue("'I can't leave,' Alice said.", 0, &cast(&["Alice"]));
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].quote, "I can't leave,");
    assert_eq!(lines[0].speaker, "Alice");
}

#[test]
fn unicode_case_expansion_does_not_reuse_transformed_offsets() {
    let text = format!("{} Alice said, \"Hello.\"", "İ".repeat(35));
    let lines = extract_dialogue(&text, 0, &cast(&["Alice"]));
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].speaker, "Alice");
}

fn checked(text: &str, names: &[&str]) -> DialogueExtractionResult {
    extract_dialogue_checked(
        text,
        42,
        &cast(names),
        &DialogueExtractionOptions::default(),
    )
    .unwrap()
}

#[test]
fn located_quotes_and_tags_round_trip_original_unicode_and_crlf() {
    for newline in ["\n", "\r\n"] {
        let text = format!("🜂 İpek said, “  Café can’t wait!  ”{newline}'好的,' 李明 replied.");
        let result = checked(&text, &["İpek", "李明"]);
        assert_eq!(result.lines.len(), 2);
        assert_eq!(result.lines[0].line.speaker, "İpek");
        assert_eq!(result.lines[1].line.speaker, "李明");
        for located in &result.lines {
            let r = &located.range;
            assert_eq!(&text[r.byte_start..r.byte_end], located.line.quote);
            assert_eq!(
                r.utf16_start as usize,
                text[..r.byte_start].encode_utf16().count()
            );
            assert_eq!(
                r.utf16_end as usize,
                text[..r.byte_end].encode_utf16().count()
            );
            let q = located.quotation_range.as_ref().unwrap();
            assert!(q.byte_start < r.byte_start && q.byte_end > r.byte_end);
            assert!(located.attribution_range.is_some());
        }
        let serialized = serde_json::to_value(&result).unwrap();
        let decoded: DialogueExtractionResult = serde_json::from_value(serialized.clone()).unwrap();
        assert_eq!(serialized, serde_json::to_value(decoded).unwrap());
    }
}

#[test]
fn defaults_abstain_and_heuristics_are_explicitly_partial() {
    let text = "“One,” Alice said.\n“Two,” Bob said.\n“Three.”";
    let result = checked(text, &["Bob", "Alice"]);
    assert_eq!(result.lines[2].line.speaker, "Unknown");
    let inferred = extract_dialogue_checked(
        text,
        0,
        &cast(&["Alice", "Bob"]),
        &DialogueExtractionOptions {
            attribution: DialogueAttributionMode::ConservativeHeuristic,
            include_indirect: false,
        },
    )
    .unwrap();
    assert_eq!(inferred.lines[2].line.speaker, "Alice");
    assert!(
        inferred
            .diagnostics
            .contains(&"heuristic_speakers_unverified".into())
    );
    assert_eq!(
        inferred.availability,
        narrative_lens::substrate::utils::AnalysisAvailability::Partial
    );
}

#[test]
fn previous_tag_with_complex_comma_clause_does_not_leak() {
    let text = r#""First," Bob said to Alice, who sat down. "Second.""#;
    let result = checked(text, &["Bob", "Alice"]);
    assert_eq!(result.lines[0].line.speaker, "Bob");
    assert_eq!(result.lines[1].line.speaker, "Unknown");
}

#[test]
fn split_quotation_and_explicit_receivers_keep_speaker_and_actual_verb() {
    for text in [
        r#""First," Alice said, "second.""#,
        r#""First," Alice said to Bob, "second.""#,
    ] {
        let result = checked(text, &["Alice", "Bob"]);
        assert_eq!(result.lines.len(), 2);
        assert!(result.lines.iter().all(|line| line.line.speaker == "Alice"));
        assert!(
            result
                .lines
                .iter()
                .all(|line| line.line.tag_verb.as_deref() == Some("said"))
        );
    }
    let result = checked(r#"Alice said to Bob: "Stay.""#, &["Alice", "Bob"]);
    assert_eq!(result.lines[0].line.speaker, "Alice");
}

#[test]
fn full_and_short_aliases_are_stable_and_prefixes_do_not_match_other_names() {
    let text = "“Hi,” John Smith said. “Bye,” Mary said. “Wait,” Anna said.";
    let a = checked(text, &["Mary Jones", "John Smith", "Ann", "Anna"]);
    let b = checked(text, &["Anna", "John Smith", "Ann", "Mary Jones"]);
    assert_eq!(
        serde_json::to_value(&a).unwrap(),
        serde_json::to_value(&b).unwrap()
    );
    assert_eq!(
        a.lines
            .iter()
            .map(|line| line.line.speaker.as_str())
            .collect::<Vec<_>>(),
        ["John Smith", "Mary Jones", "Anna"]
    );
}

#[test]
fn paragraph_continuation_escapes_and_malformed_delimiters_are_explicit() {
    let result = checked(
        "“First paragraph.\n\n“Second paragraph,” Alice said.",
        &["Alice"],
    );
    assert_eq!(result.lines.len(), 1);
    assert!(result.lines[0].line.quote.contains("\n\n“Second"));
    let result = checked(r#""She said \"leave\" and left," Alice said."#, &["Alice"]);
    assert_eq!(result.lines.len(), 1);
    let result = checked("“No closing quote", &["Alice"]);
    assert!(result.lines.is_empty());
    assert!(
        result
            .diagnostics
            .contains(&"unbalanced_quote_delimiters".into())
    );
}

#[test]
fn indirect_mode_is_source_ordered_and_separate_from_direct_quotes() {
    let text = r#"Alice told him that the road was closed. "Wait," Bob said."#;
    assert_eq!(checked(text, &["Alice", "Bob"]).lines.len(), 1);
    let result = extract_dialogue_checked(
        text,
        0,
        &cast(&["Alice", "Bob"]),
        &DialogueExtractionOptions {
            include_indirect: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.lines.len(), 2);
    assert_eq!(result.lines[0].kind, DialogueKind::Indirect);
    assert_eq!(
        result.lines[0].line.quote,
        "Alice told him that the road was closed."
    );
    assert!(result.lines[0].quotation_range.is_none());
    assert_eq!(result.lines[1].kind, DialogueKind::Direct);
}

#[test]
fn checked_limits_reject_without_partial_success() {
    assert_eq!(
        serde_json::from_str::<DialogueExtractionOptions>("{}").unwrap(),
        DialogueExtractionOptions::default()
    );
    assert!(serde_json::from_str::<DialogueExtractionOptions>(r#"{"unknown":true}"#).is_err());
    for length in [127, 128, 129] {
        let name = "x".repeat(length);
        assert_eq!(
            extract_dialogue_checked("", 0, &[name], &Default::default()).is_ok(),
            length <= 128
        );
    }
    for depth in [31, 32, 33] {
        let opening: String = (0..depth)
            .map(|i| if i % 2 == 0 { '“' } else { '‘' })
            .collect();
        let closing: String = (0..depth)
            .rev()
            .map(|i| if i % 2 == 0 { '”' } else { '’' })
            .collect();
        assert_eq!(
            extract_dialogue_checked(
                &format!("{opening}word{closing}"),
                0,
                &[],
                &Default::default()
            )
            .is_ok(),
            depth <= 32
        );
    }
    for count in [9999, 10000, 10001] {
        let text = "\"x\"\n".repeat(count);
        assert_eq!(
            extract_dialogue_checked(&text, 0, &[], &Default::default()).is_ok(),
            count <= 10000
        );
    }
}

#[test]
fn quote_style_substitution_and_cast_permutations_preserve_explicit_speakers() {
    for (open, close) in [('"', '"'), ('\'', '\''), ('“', '”'), ('‘', '’')] {
        let text = format!(
            "{open}Good morning,{close} 李明 said to Zoë. Zoë replied, {open}Good evening.{close}"
        );
        let a = checked(&text, &["Zoë", "李明"]);
        let b = checked(&text, &["李明", "Zoë"]);
        assert_eq!(
            a.lines
                .iter()
                .map(|line| line.line.speaker.as_str())
                .collect::<Vec<_>>(),
            ["李明", "Zoë"]
        );
        assert_eq!(
            serde_json::to_value(a).unwrap(),
            serde_json::to_value(b).unwrap()
        );
    }
}

#[test]
fn conflicting_or_unknown_explicit_speakers_never_fall_back_to_proximity() {
    for text in [
        r#"Alice said, "Hello," Bob replied."#,
        r#"Alice watched. "Hello," Charlie said."#,
    ] {
        let result = extract_dialogue_checked(
            text,
            0,
            &cast(&["Alice", "Bob"]),
            &DialogueExtractionOptions {
                attribution: DialogueAttributionMode::ConservativeHeuristic,
                include_indirect: false,
            },
        )
        .unwrap();
        assert_eq!(result.lines[0].line.speaker, "Unknown");
        assert_eq!(result.lines[0].line.confidence, 0.0);
    }
}

#[test]
fn text_and_cast_resource_boundaries_are_checked_before_analysis() {
    for count in [255, 256, 257] {
        let names = vec!["Alice".to_string(); count];
        assert_eq!(
            extract_dialogue_checked("", 0, &names, &Default::default()).is_ok(),
            count <= 256
        );
    }
    let limit = 16 * 1024 * 1024;
    let source = " ".repeat(limit + 1);
    for bytes in [limit - 1, limit, limit + 1] {
        assert_eq!(
            extract_dialogue_checked(&source[..bytes], 0, &[], &Default::default()).is_ok(),
            bytes <= limit
        );
    }
    assert!(extract_dialogue_checked("", 0, &["Alice\n".into()], &Default::default()).is_err());
    assert!(extract_dialogue_checked("", 0, &["x".repeat(200000)], &Default::default()).is_err());
}
