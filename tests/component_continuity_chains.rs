use narrative_lens::continuity::coreference::*;
use std::collections::HashMap;

const CAST: &[&str] = &["Alice", "Bob", "John", "James", "Mary"];

fn hints() -> HashMap<String, String> {
    [
        ("Alice", "female"),
        ("Bob", "male"),
        ("John", "male"),
        ("James", "male"),
        ("Mary", "female"),
    ]
    .into_iter()
    .map(|(n, g)| (n.into(), g.into()))
    .collect()
}
fn chain_prefix() -> String {
    format!("Alice{}She waited.", " ".repeat(284))
}
fn pronouns(result: &CorefResult) -> Vec<&Mention> {
    result.mentions.iter().filter(|m| m.is_pronoun).collect()
}

#[test]
fn bounded_chains_extend_visible_antecedents_with_inspectable_original_links() {
    let mut text = chain_prefix();
    for _ in 0..MAX_COREFERENCE_CHAIN_DEPTH + 1 {
        text.push_str(&format!("{}She waited.", " ".repeat(160)));
    }
    let result = CoreferenceResolver::new()
        .resolve_checked(&text, CAST, Some(&hints()))
        .unwrap();
    let mentions = pronouns(&result);
    assert_eq!(mentions.len(), usize::from(MAX_COREFERENCE_CHAIN_DEPTH) + 2);
    assert_eq!(
        mentions[0].resolution_basis,
        Some(CoreferenceBasis::UniquePrecedingCandidate)
    );
    assert_eq!(mentions[0].antecedent_start, Some(0));
    for hop in 1..=usize::from(MAX_COREFERENCE_CHAIN_DEPTH) {
        assert_eq!(mentions[hop].resolved_to.as_deref(), Some("Alice"));
        assert_eq!(
            mentions[hop].resolution_basis,
            Some(CoreferenceBasis::PronounChain)
        );
        assert_eq!(
            mentions[hop].antecedent_start,
            Some(mentions[hop - 1].start)
        );
        let range = mentions[hop].range.as_ref().unwrap();
        assert_eq!(&text[range.byte_start..range.byte_end], "She");
    }
    assert!(mentions.last().unwrap().resolved_to.is_none());
    assert!(mentions.last().unwrap().antecedent_start.is_none());
}

#[test]
fn per_link_distance_limits_are_exact_and_dont_create_first_antecedents() {
    let resolver = CoreferenceResolver::new();
    for distance in [299, 300, 301] {
        let text = format!(
            "{}{}She rested.",
            chain_prefix(),
            " ".repeat(distance - "She waited.".len())
        );
        let result = resolver
            .resolve_checked(&text, CAST, Some(&hints()))
            .unwrap();
        let mentions = pronouns(&result);
        assert_eq!(mentions[1].start - mentions[0].start, distance);
        assert_eq!(mentions[1].resolved_to.is_some(), distance <= 300);
    }
    for text in [
        "She waited. She left.",
        "She waited. Alice arrived. He left.",
    ] {
        let result = resolver
            .resolve_checked(text, CAST, Some(&hints()))
            .unwrap();
        assert!(pronouns(&result).iter().all(|m| m.resolved_to.is_none()));
    }
}

#[test]
fn chains_stop_at_scope_changes_competing_names_and_unresolved_mentions() {
    let resolver = CoreferenceResolver::new();
    // A paragraph break, or a gender-compatible name the pronoun cannot see as
    // a candidate (here inside a quotation), severs the chain.
    for between in ["\n\n", " \"Mary, come!\" "] {
        let text = format!(
            "{}{}{}She rested.",
            chain_prefix(),
            between,
            " ".repeat(160)
        );
        let result = resolver
            .resolve_checked(&text, CAST, Some(&hints()))
            .unwrap();
        assert_eq!(pronouns(&result)[0].resolved_to.as_deref(), Some("Alice"));
        assert!(pronouns(&result)[1].resolved_to.is_none());
    }
    // A visible compatible name is simply the new unique candidate.
    let text = format!(
        "{} Mary waited.{}She rested.",
        chain_prefix(),
        " ".repeat(160)
    );
    let result = resolver
        .resolve_checked(&text, CAST, Some(&hints()))
        .unwrap();
    assert_eq!(pronouns(&result)[1].resolved_to.as_deref(), Some("Mary"));
    assert_eq!(
        pronouns(&result)[1].resolution_basis,
        Some(CoreferenceBasis::UniquePrecedingCandidate)
    );
    // An incompatible name or an intervening quotation does not sever the
    // chain; a pronoun inside the quotation stays outside the narrative chain.
    for between in [
        " Bob waited. ",
        " \"Bob, come!\" ",
        " \"No one moved.\" ",
        " \"She moved.\" ",
    ] {
        let text = format!(
            "{}{}{}She rested.",
            chain_prefix(),
            between,
            " ".repeat(160)
        );
        let result = resolver
            .resolve_checked(&text, CAST, Some(&hints()))
            .unwrap();
        let mentions = pronouns(&result);
        assert_eq!(mentions[0].resolved_to.as_deref(), Some("Alice"));
        let last = mentions.last().unwrap();
        assert_eq!(last.resolved_to.as_deref(), Some("Alice"));
        assert_eq!(last.resolution_basis, Some(CoreferenceBasis::PronounChain));
        assert_eq!(last.antecedent_start, Some(mentions[0].start));
        if mentions.len() == 3 {
            assert!(mentions[1].resolved_to.is_none());
        }
    }
    let text = format!(
        "John waited. He rested. James met John. He paused.{}He left.",
        " ".repeat(310)
    );
    let result = resolver
        .resolve_checked(&text, CAST, Some(&hints()))
        .unwrap();
    assert_eq!(pronouns(&result)[0].resolved_to.as_deref(), Some("John"));
    assert!(pronouns(&result)[1].resolved_to.is_none());
    assert!(pronouns(&result)[2].resolved_to.is_none());
}

#[test]
fn quote_local_references_use_only_names_and_chains_in_the_same_quote() {
    let resolver = CoreferenceResolver::new();
    let text = format!(
        "Bob said, \"{}{}She rested.\" She left.",
        chain_prefix(),
        " ".repeat(160)
    );
    let result = resolver
        .resolve_checked(&text, CAST, Some(&hints()))
        .unwrap();
    let mentions = pronouns(&result);
    assert_eq!(mentions[0].resolved_to.as_deref(), Some("Alice"));
    assert_eq!(
        mentions[0].resolution_basis,
        Some(CoreferenceBasis::QuoteLocalNamedCandidate)
    );
    assert_eq!(
        mentions[1].resolution_basis,
        Some(CoreferenceBasis::PronounChain)
    );
    assert!(mentions[2].resolved_to.is_none());
    let ambiguous = resolver
        .resolve_checked("\"John greeted James. He left.\"", CAST, Some(&hints()))
        .unwrap();
    assert!(pronouns(&ambiguous)[0].resolved_to.is_none());
    for text in ["Alice said, \"She left.\"", "\"She left.\" Alice arrived."] {
        assert!(
            pronouns(
                &resolver
                    .resolve_checked(text, CAST, Some(&hints()))
                    .unwrap()
            )[0]
            .resolved_to
            .is_none()
        );
    }
}
