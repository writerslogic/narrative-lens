use narrative_lens::craft::dialogue::{DialogueExtractionOptions, extract_dialogue_checked};
fn speaker(text: &str, cast: &[&str]) -> String {
    let cast: Vec<_> = cast.iter().map(|name| (*name).into()).collect();
    let result =
        extract_dialogue_checked(text, 0, &cast, &DialogueExtractionOptions::default()).unwrap();
    assert_eq!(result.lines.len(), 1);
    result.lines[0].line.speaker.clone()
}
#[test]
fn articles_are_not_generated_identity_aliases_but_explicit_names_remain_authoritative() {
    assert_eq!(
        speaker("\"Wait,\" said the mother.", &["the Carthusians"]),
        "Unknown"
    );
    assert_eq!(speaker("\"Wait,\" said The.", &["The"]), "The");
    assert_eq!(
        speaker("\"Wait,\" said Carthusians.", &["the Carthusians"]),
        "the Carthusians"
    );
}
#[test]
fn declared_names_ending_ly_are_not_discarded_as_adverbs() {
    for name in ["Sally", "Emily", "Kimberly"] {
        for text in [
            format!("\"Wait,\" said {name}."),
            format!("{name} said, \"Wait.\""),
            format!("Some prose preceded it. Said {name}: \"Wait.\""),
        ] {
            assert_eq!(speaker(&text, &[name]), name, "{text}");
        }
    }
}
#[test]
fn generated_surname_does_not_override_an_incompatible_explicit_given_name() {
    assert_eq!(
        speaker("William Brown said, \"Wait.\"", &["John Brown"]),
        "Unknown"
    );
    assert_eq!(
        speaker("\"Wait,\" said Brown Green.", &["John Brown"]),
        "Unknown"
    );
    assert_eq!(
        speaker("\"Wait,\" said John Green.", &["John Brown"]),
        "Unknown"
    );
}
#[test]
fn full_registered_names_win_and_genuine_adverbs_still_work() {
    assert_eq!(
        speaker(
            "William Brown said, \"Wait.\"",
            &["John Brown", "William Brown"]
        ),
        "William Brown"
    );
    assert_eq!(
        speaker("\"Wait,\" said Alice quietly.", &["Alice"]),
        "Alice"
    );
    assert_eq!(speaker("Alice softly said, \"Wait.\"", &["Alice"]), "Alice");
    assert_eq!(
        speaker("\"Wait,\" said quietly Sally.", &["Sally"]),
        "Sally"
    );
    assert_eq!(
        speaker("Mr. Brown said, \"Wait.\"", &["John Brown"]),
        "John Brown"
    );
}

#[test]
fn sentence_connectives_and_speech_verbs_are_not_incompatible_name_qualifiers() {
    for text in [
        "Then John said, \"Wait.\"",
        "Finally John said, \"Wait.\"",
        "Said John: \"Wait.\"",
    ] {
        assert_eq!(speaker(text, &["John Brown"]), "John Brown");
    }
    assert_eq!(
        speaker("\"Wait,\" Bob Sally said.", &["Bob", "Sally"]),
        "Unknown"
    );
}
