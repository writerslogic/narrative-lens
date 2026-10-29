use narrative_lens::{
    continuity::{author_knowledge::AuthorSource, coreference::*},
    manuscript::{Configuration, Manuscript, Scene},
    substrate::utils::AnalysisAvailability,
};
use std::collections::HashMap;

fn hints(values: &[(&str, &str)]) -> HashMap<String, String> {
    values
        .iter()
        .map(|(name, value)| ((*name).into(), (*value).into()))
        .collect()
}
fn pronouns(result: &CorefResult) -> Vec<&Mention> {
    result
        .mentions
        .iter()
        .filter(|mention| mention.is_pronoun)
        .collect()
}

#[test]
fn automatic_coreference_requires_visible_unambiguous_entities_not_inventory_or_proximity() {
    let resolver = CoreferenceResolver::new();
    let male = hints(&[("Alex", "male")]);
    for text in [
        "They waited. Alex arrived.",
        "He left. Alex arrived.",
        "They waited alone.",
    ] {
        let result = resolver
            .resolve_checked(text, &["Alex"], Some(&male))
            .unwrap();
        assert!(
            pronouns(&result)
                .iter()
                .all(|mention| mention.resolved_to.is_none())
        );
    }
    let result = resolver
        .resolve_checked("Alex waited. They left.", &["Alex"], Some(&male))
        .unwrap();
    assert!(pronouns(&result)[0].resolved_to.is_none());
    let result = resolver
        .resolve_checked(
            "Alex waited. They left.",
            &["Alex"],
            Some(&hints(&[("Alex", "neutral")])),
        )
        .unwrap();
    assert_eq!(pronouns(&result)[0].resolved_to.as_deref(), Some("Alex"));
    let result = resolver
        .resolve_checked(
            "John greeted James. He left.",
            &["John", "James"],
            Some(&hints(&[("John", "male"), ("James", "male")])),
        )
        .unwrap();
    assert!(pronouns(&result)[0].resolved_to.is_none());
    let result = resolver
        .resolve_checked("Alice and Bob waited. They left.", &["Alice", "Bob"], None)
        .unwrap();
    assert!(pronouns(&result)[0].resolved_to.is_none());
}

#[test]
fn quoted_pronouns_remain_unattributed_and_cataphora_has_real_clause_bounds() {
    let resolver = CoreferenceResolver::new();
    for text in [
        "Alice said, \"She is ready.\"",
        "Alice said, ‘She's ready.’",
        "Alice said, 'She's ready.'",
    ] {
        let result = resolver
            .resolve_checked(text, &["Alice"], Some(&hints(&[("Alice", "female")])))
            .unwrap();
        assert_eq!(pronouns(&result).len(), 1);
        assert!(pronouns(&result)[0].resolved_to.is_none());
    }
    let male = hints(&[("John", "male")]);
    let bounded = resolver
        .resolve_checked("When he arrived, John rested.", &["John"], Some(&male))
        .unwrap();
    assert_eq!(
        pronouns(&bounded)[0].resolution_basis,
        Some(CoreferenceBasis::BoundedCataphora)
    );
    let far = format!("When he {}arrived, John rested.", "quietly ".repeat(30));
    assert!(
        pronouns(
            &resolver
                .resolve_checked(&far, &["John"], Some(&male))
                .unwrap()
        )[0]
        .resolved_to
        .is_none()
    );
    let mixed = resolver
        .resolve_checked(
            "When he arrived, John greeted James.",
            &["John", "James"],
            Some(&hints(&[("John", "male"), ("James", "male")])),
        )
        .unwrap();
    assert!(pronouns(&mixed)[0].resolved_to.is_none());
}

#[test]
fn literal_mentions_use_maximal_names_and_preserve_unicode_byte_and_utf16_spans() {
    let resolver = CoreferenceResolver::new();
    let text = "\u{feff}İ😀 Élodie greeted Ann Lee. She left.";
    let result = resolver
        .resolve_checked(
            text,
            &["Ann", "Ann Lee", "Élodie"],
            Some(&hints(&[
                ("Élodie", "female"),
                ("Ann", "male"),
                ("Ann Lee", "male"),
            ])),
        )
        .unwrap();
    assert_eq!(result.character_mentions["Ann"], 0);
    assert_eq!(result.character_mentions["Ann Lee"], 1);
    assert_eq!(result.character_mentions["Élodie"], 2);
    let utf16: Vec<_> = text.encode_utf16().collect();
    for mention in &result.mentions {
        let range = mention.range.as_ref().unwrap();
        assert_eq!(range.byte_start, mention.start);
        assert_eq!(&text[range.byte_start..range.byte_end], mention.text);
        assert_eq!(
            String::from_utf16(&utf16[range.utf16_start as usize..range.utf16_end as usize])
                .unwrap(),
            mention.text
        );
    }
    let mut cast = vec!["Ann", "Ann Lee", "Élodie"];
    cast.reverse();
    let repeated = resolver
        .resolve_checked(
            text,
            &cast,
            Some(&hints(&[
                ("Élodie", "female"),
                ("Ann", "male"),
                ("Ann Lee", "male"),
            ])),
        )
        .unwrap();
    assert_eq!(
        serde_json::to_string(&result).unwrap(),
        serde_json::to_string(&repeated).unwrap()
    );
    let ambiguous = resolver
        .resolve_checked("Σ paused. He left.", &["Σ", "ς"], None)
        .unwrap();
    assert!(
        ambiguous
            .mentions
            .iter()
            .all(|mention| mention.resolved_to.is_none())
    );
}

#[test]
fn pronoun_hints_do_not_assign_every_nearby_pronoun_to_every_character() {
    let resolver = CoreferenceResolver::new();
    // "Bobo" is not in the given-name lexicon, so only the text can gender it,
    // and a window naming two characters attributes nothing.
    let scene = "Alice met Bobo. He paused. His bag was heavy. He left.\n\nAlice rested. She smiled. Her bag was light. She left.";
    let result = resolver
        .build_character_gender_map_checked(&[scene], &["Alice", "Bobo"])
        .unwrap();
    assert_eq!(result["Bobo"], "unknown");
    assert_eq!(result["Alice"], "female");
    assert!(
        resolver
            .resolve_checked(
                "Alice waited.",
                &["Alice"],
                Some(&hints(&[("Other", "male")]))
            )
            .is_err()
    );
    assert!(
        resolver
            .resolve_checked(
                "Alice waited.",
                &["Alice"],
                Some(&hints(&[("Alice", "invalid")]))
            )
            .is_err()
    );
    assert!(
        resolver
            .resolve_checked("Alice waited.", &["Alice", "ALICE"], None)
            .is_err()
    );
    let invalid = resolver.resolve("Alice waited.", &[&"x".repeat(200_000)], None);
    assert_eq!(invalid.availability, AnalysisAvailability::Unavailable);
    assert!(
        resolver
            .build_character_gender_map_checked(&["x"], &[&"x".repeat(200_000)])
            .is_err()
    );
}

#[test]
fn automatic_context_and_mention_budgets_are_observable_at_boundaries() {
    let resolver = CoreferenceResolver::new();
    let gender = hints(&[("John", "male")]);
    for distance in [299, 300, 301] {
        let text = format!("John{}he waited.", " ".repeat(distance - 4));
        let result = resolver
            .resolve_checked(&text, &["John"], Some(&gender))
            .unwrap();
        assert_eq!(pronouns(&result)[0].resolved_to.is_some(), distance <= 300);
    }
    for count in [
        MAX_COREFERENCE_MENTIONS - 1,
        MAX_COREFERENCE_MENTIONS,
        MAX_COREFERENCE_MENTIONS + 1,
    ] {
        let result = resolver.resolve_checked(&"John ".repeat(count), &["John"], Some(&gender));
        assert_eq!(result.is_ok(), count <= MAX_COREFERENCE_MENTIONS);
    }
    let empty = resolver.resolve_checked("", &[], None).unwrap();
    assert_eq!((empty.pronoun_count, empty.resolved_pronoun_count), (0, 0));
    assert!(empty.resolution_coverage.is_none());
}

fn manuscript() -> Manuscript {
    Manuscript {
        document_id: "links".into(),
        scenes: vec![
            Scene::new("s", "\u{feff}😀 Alice met Bob. Her brother met him."),
            Scene::new("t", "Alice waited."),
        ],
        known_characters: vec!["Alice".into(), "Bob".into()],
        configuration: Configuration::default(),
    }
}
fn source(input: &Manuscript, scene: &str, quote: &str) -> AuthorSource {
    let text = &input.scenes.iter().find(|s| s.id == scene).unwrap().text;
    let byte = text.find(quote).unwrap();
    let start = text[..byte].encode_utf16().count() as u32;
    AuthorSource {
        scene_id: scene.into(),
        utf16_start: start,
        utf16_end: start + quote.encode_utf16().count() as u32,
        quote: quote.into(),
    }
}
fn links(input: &Manuscript) -> Vec<AuthorCoreferenceLink> {
    vec![
        AuthorCoreferenceLink {
            id: "her".into(),
            character_id: "Alice".into(),
            mention: source(input, "s", "Her"),
            antecedent: source(input, "s", "Alice"),
        },
        AuthorCoreferenceLink {
            id: "him".into(),
            character_id: "Bob".into(),
            mention: source(input, "s", "him"),
            antecedent: source(input, "s", "Bob"),
        },
    ]
}

#[test]
fn explicit_coreference_retains_stable_sources_and_rejects_conflicting_span_identities() {
    let mut input = manuscript();
    let mut links = links(&input);
    let expected = resolve_author_coreference(&input, &links).unwrap();
    assert_eq!(expected.links.len(), 2);
    assert_eq!(expected.links[0].character_id, "Alice");
    links.reverse();
    input.scenes.reverse();
    assert_eq!(
        resolve_author_coreference(&input, &links).unwrap(),
        expected
    );
    let mut bad = links.clone();
    bad.push(AuthorCoreferenceLink {
        id: "conflict".into(),
        character_id: "Bob".into(),
        mention: source(&input, "t", "Alice"),
        antecedent: source(&input, "s", "Alice"),
    });
    assert!(resolve_author_coreference(&input, &bad).is_err());
    let mut bad = links.clone();
    bad[0].antecedent = bad[0].mention.clone();
    assert!(resolve_author_coreference(&input, &bad).is_err());
    let mut bad = links.clone();
    bad[0].character_id = "Unknown".into();
    assert!(resolve_author_coreference(&input, &bad).is_err());
    let mut bad = links.clone();
    bad[0].mention.utf16_start = 2;
    assert!(resolve_author_coreference(&input, &bad).is_err());
    let scene = input.scenes.iter_mut().find(|s| s.id == "s").unwrap();
    scene.text.insert_str(0, "Later: ");
    assert!(resolve_author_coreference(&input, &links).is_err());
}

#[test]
fn explicit_coreference_supports_nested_mentions_and_declared_cataphora() {
    let mut input = manuscript();
    let mut values = links(&input);
    values.push(AuthorCoreferenceLink {
        id: "nested".into(),
        character_id: "Bob".into(),
        mention: source(&input, "s", "Her brother"),
        antecedent: source(&input, "s", "Bob"),
    });
    let result = resolve_author_coreference(&input, &values).unwrap();
    assert_eq!(result.links.len(), 3);
    input.scenes[0].text = "He arrived before Bob.".into();
    let result = resolve_author_coreference(
        &input,
        &[AuthorCoreferenceLink {
            id: "forward".into(),
            character_id: "Bob".into(),
            mention: source(&input, "s", "He"),
            antecedent: source(&input, "s", "Bob"),
        }],
    )
    .unwrap();
    assert!(result.links[0].mention.range.byte_start < result.links[0].antecedent.range.byte_start);
}

#[test]
fn author_link_count_and_metadata_limits_are_exact_without_quadratic_history() {
    let mut input = manuscript();
    input.scenes = vec![
        Scene::new("s", "x".repeat(65_600)),
        Scene::new("t", "x".repeat(65_536)),
    ];
    let make = |id: String, start: u32, length: usize, antecedent: usize| AuthorCoreferenceLink {
        id,
        character_id: "Alice".into(),
        mention: AuthorSource {
            scene_id: "s".into(),
            utf16_start: start,
            utf16_end: start + length as u32,
            quote: "x".repeat(length),
        },
        antecedent: AuthorSource {
            scene_id: "t".into(),
            utf16_start: 0,
            utf16_end: antecedent as u32,
            quote: "x".repeat(antecedent),
        },
    };
    let many: Vec<_> = (0..=MAX_COREFERENCE_MENTIONS)
        .map(|i| make(format!("l{i}"), i as u32, 1, 1))
        .collect();
    for count in [MAX_COREFERENCE_MENTIONS - 1, MAX_COREFERENCE_MENTIONS] {
        assert_eq!(
            resolve_author_coreference(&input, &many[..count])
                .unwrap()
                .links
                .len(),
            count
        );
    }
    assert!(resolve_author_coreference(&input, &many).is_err());
    for count in [255, 256, 257] {
        assert_eq!(
            resolve_author_coreference(&input, &[make("i".repeat(count), 0, 1, 1)]).is_ok(),
            count <= 256
        );
    }
    for count in [65_535, 65_536, 65_537] {
        assert_eq!(
            resolve_author_coreference(&input, &[make("i".into(), 0, count, 1)]).is_ok(),
            count <= 65_536
        );
    }
    for target in [
        MAX_AUTHOR_COREFERENCE_BYTES - 1,
        MAX_AUTHOR_COREFERENCE_BYTES,
        MAX_AUTHOR_COREFERENCE_BYTES + 1,
    ] {
        let mut values = Vec::new();
        let mut remaining = target;
        while remaining > 0 {
            let id = format!("l{}", values.len());
            let overhead = id.len() + 7;
            let mut quote = (remaining - overhead).min(131_072);
            if (1..64).contains(&(remaining - overhead - quote)) {
                quote -= 64;
            }
            let mention = 65_536.min(quote - 1);
            let antecedent = quote - mention;
            values.push(make(id, values.len() as u32, mention, antecedent));
            remaining -= overhead + quote;
        }
        assert_eq!(
            resolve_author_coreference(&input, &values).is_ok(),
            target <= MAX_AUTHOR_COREFERENCE_BYTES
        );
    }
}
