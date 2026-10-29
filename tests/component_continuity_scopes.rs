use narrative_lens::continuity::coreference::*;
use std::collections::HashMap;

fn resolve(text: &str) -> CorefResult {
    let hints: HashMap<_, _> = [("Alice", "female"), ("Bob", "male"), ("James", "male")]
        .into_iter()
        .map(|(name, gender)| (name.into(), gender.into()))
        .collect();
    CoreferenceResolver::new()
        .resolve_checked(text, &["Alice", "Bob", "James"], Some(&hints))
        .unwrap()
}
fn pronouns(result: &CorefResult) -> Vec<&Mention> {
    result.mentions.iter().filter(|m| m.is_pronoun).collect()
}

#[test]
fn lexical_apostrophes_elisions_and_escaped_marks_do_not_hide_narrative_antecedents() {
    for text in [
        "Alice waited. 'Tis quiet. She rested.",
        "Alice waited. James 's coat moved. She rested.",
        "Alice waited. James ’s coat moved. She rested.",
        r#"Alice waited. A \" mark was visible. She rested."#,
    ] {
        let result = resolve(text);
        assert_eq!(
            pronouns(&result).last().unwrap().resolved_to.as_deref(),
            Some("Alice"),
            "{text}"
        );
        assert!(
            !result
                .diagnostics
                .iter()
                .any(|d| d == "unbalanced_quote_delimiters")
        );
    }
}

#[test]
fn nested_structural_boundaries_keep_inner_and_outer_reference_contexts_separate() {
    let inner = resolve("“Alice waited. ‘She rested,’ Bob said.”");
    assert!(pronouns(&inner)[0].resolved_to.is_none());
    let explicit = resolve("“Alice waited. ‘Bob paused. He rested.’ He left.”");
    let mentions = pronouns(&explicit);
    assert_eq!(mentions[0].resolved_to.as_deref(), Some("Bob"));
    assert_eq!(
        mentions[0].resolution_basis,
        Some(CoreferenceBasis::QuoteLocalNamedCandidate)
    );
    assert!(mentions[1].resolved_to.is_none());
    let returned = resolve("“Alice waited. ‘No one spoke.’ She rested.”");
    assert!(pronouns(&returned)[0].resolved_to.is_none());
}

#[test]
fn malformed_outer_scopes_preserve_barriers_and_shared_diagnostics() {
    let result =
        resolve("Alice waited. ‘ unclosed outer with \"balanced inner\" words. She rested.");
    assert!(pronouns(&result)[0].resolved_to.is_none());
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d == "unbalanced_quote_delimiters")
    );
    let no_speaker = resolve("Alice said, ‘She waited.’");
    assert!(pronouns(&no_speaker)[0].resolved_to.is_none());
    let text = "\u{feff}😀 “Alice paused. She rested.”";
    let exact = resolve(text);
    let mention = pronouns(&exact)[0];
    let range = mention.range.as_ref().unwrap();
    assert_eq!(&text[range.byte_start..range.byte_end], "She");
    assert_eq!(mention.antecedent_start, text.find("Alice"));
}
