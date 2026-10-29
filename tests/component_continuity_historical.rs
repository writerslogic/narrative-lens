use narrative_lens::{
    continuity::{anachronism::*, author_knowledge::AuthorSource},
    manuscript::{Configuration, Manuscript, Scene, TextRange},
    substrate::utils::AnalysisAvailability,
};

fn input(text: &str) -> Manuscript {
    Manuscript {
        document_id: "history".into(),
        scenes: vec![Scene::new("s", text)],
        known_characters: Vec::new(),
        configuration: Configuration::default(),
    }
}
fn entry(id: &str, term: &str, ambiguous: bool) -> HistoricalEntry {
    HistoricalEntry {
        id: id.into(),
        term: term.into(),
        earliest_year: 1900,
        basis: HistoricalBasis::AuthorConstraint,
        reference: "Synthetic author constraint; not a historical citation".into(),
        context: "Synthetic fixture context".into(),
        sense_ambiguous: ambiguous,
    }
}
fn usage(m: &Manuscript, id: &str, entry_id: &str, quote: &str) -> HistoricalUsage {
    let text = &m.scenes[0].text;
    let start = text.find(quote).unwrap();
    let range = TextRange::new(text, start, start + quote.len()).unwrap();
    HistoricalUsage {
        id: id.into(),
        entry_id: entry_id.into(),
        evidence: AuthorSource {
            scene_id: "s".into(),
            utf16_start: range.utf16_start,
            utf16_end: range.utf16_end,
            quote: quote.into(),
        },
    }
}

#[test]
fn catalogue_matches_original_unicode_spans_overlaps_and_whitespace_without_substrings() {
    let m = input("\u{feff}😀 police cream; ice\u{2003}\tcream; C++; XC++; ÉTHER.");
    let catalogue = [
        entry("ice", "ice", false),
        entry("phrase", "ice cream", false),
        entry("cream", "cream", false),
        entry("cpp", "C++", false),
        entry("ether", "Éther", false),
    ];
    let result = analyze_historical_catalogue(&m, &catalogue, &[], 1800, 0).unwrap();
    assert_eq!(result.candidates.len(), 6);
    assert_eq!(
        result
            .candidates
            .iter()
            .filter(|c| c.entry_id == "ice")
            .count(),
        1
    );
    assert_eq!(
        result
            .candidates
            .iter()
            .filter(|c| c.entry_id == "cpp")
            .count(),
        1
    );
    assert_eq!(
        result
            .candidates
            .iter()
            .find(|c| c.entry_id == "phrase")
            .unwrap()
            .quote,
        "ice\u{2003}\tcream"
    );
    for candidate in &result.candidates {
        assert_eq!(candidate.date_conflict, Some(true));
        assert!(candidate.author_usage_id.is_none());
        assert_eq!(
            TextRange::from_utf16(
                &m.scenes[0].text,
                candidate.range.utf16_start,
                candidate.range.utf16_end
            )
            .unwrap(),
            candidate.range
        );
        assert_eq!(
            &m.scenes[0].text[candidate.range.byte_start..candidate.range.byte_end],
            candidate.quote
        );
    }
    let mut reordered = catalogue.to_vec();
    reordered.reverse();
    assert_eq!(
        result,
        analyze_historical_catalogue(&m, &reordered, &[], 1800, 0).unwrap()
    );
    assert_eq!(result.availability, AnalysisAvailability::Partial);
}

#[test]
fn ambiguous_senses_are_unassessed_until_exact_usage_is_confirmed() {
    let m = input("\u{feff}😀 cool breeze. That is cool.");
    let mut e = entry("cool", "cool", true);
    e.basis = HistoricalBasis::FirstAttestation;
    e.reference = "Synthetic attestation record for engine testing; not actual etymology".into();
    let original = analyze_historical_catalogue(&m, &[e.clone()], &[], 1800, 0).unwrap();
    assert!(
        original
            .candidates
            .iter()
            .all(|c| c.date_conflict.is_none())
    );
    let annotation = usage(&m, "u", "cool", "cool");
    let confirmed = analyze_historical_catalogue(&m, &[e], &[annotation], 1800, 0).unwrap();
    assert_eq!(confirmed.candidates[0].date_conflict, Some(true));
    assert_eq!(
        confirmed.candidates[0].author_usage_id.as_deref(),
        Some("u")
    );
    assert_eq!(confirmed.candidates[1].date_conflict, None);
    assert!(
        confirmed
            .diagnostics
            .iter()
            .any(|d| d.contains("first_attestation_is_not_first_use"))
    );
}

#[test]
fn declared_date_math_handles_tolerance_and_extreme_years_without_inventing_calendar_facts() {
    let m = input("moonwire");
    let e = entry("m", "moonwire", false);
    for (target, tolerance, expected) in [
        (1899, 0, true),
        (1900, 0, false),
        (1901, 0, false),
        (1879, 20, true),
        (1880, 20, false),
        (1881, 20, false),
        (i32::MIN, 0, true),
        (i32::MAX, u32::MAX, false),
    ] {
        let result =
            analyze_historical_catalogue(&m, std::slice::from_ref(&e), &[], target, tolerance)
                .unwrap();
        assert_eq!(result.candidates[0].date_conflict, Some(expected));
    }
}

#[test]
fn usages_require_valid_exact_sources_and_atomic_unambiguous_references() {
    let m = input("\u{feff}😀 moonwire");
    let catalogue = [entry("m", "moonwire", true)];
    let u = usage(&m, "u", "m", "moonwire");
    assert!(
        analyze_historical_catalogue(&m, &catalogue, &[u.clone(), u.clone()], 1800, 0).is_err()
    );
    for bad in [
        HistoricalUsage {
            entry_id: "typo".into(),
            ..u.clone()
        },
        HistoricalUsage {
            evidence: AuthorSource {
                quote: "other".into(),
                ..u.evidence.clone()
            },
            ..u.clone()
        },
        HistoricalUsage {
            evidence: AuthorSource {
                utf16_start: 2,
                ..u.evidence.clone()
            },
            ..u.clone()
        },
    ] {
        assert!(analyze_historical_catalogue(&m, &catalogue, &[bad], 1800, 0).is_err());
    }
    let edited = input("\u{feff}😀 otherxxx");
    assert!(
        analyze_historical_catalogue(&edited, &catalogue, std::slice::from_ref(&u), 1800, 0)
            .is_err()
    );
    let inside = input("amoonwire");
    let substring = usage(&inside, "u", "m", "moonwire");
    assert!(analyze_historical_catalogue(&inside, &catalogue, &[substring], 1800, 0).is_err());
    let mut bad = catalogue[0].clone();
    bad.reference.clear();
    assert!(analyze_historical_catalogue(&m, &[bad], &[], 1800, 0).is_err());
    assert!(
        analyze_historical_catalogue(
            &m,
            &[catalogue[0].clone(), catalogue[0].clone()],
            &[],
            1800,
            0
        )
        .is_err()
    );
    assert!(serde_json::from_value::<HistoricalEntry>(serde_json::json!({"id":"m","term":"moonwire","earliestYear":1900,"basis":"author_constraint","reference":"r","context":"c","senseAmbiguous":false,"invented":true})).is_err());
}

#[test]
fn catalogue_entry_term_token_and_metadata_limits_cover_exact_boundaries() {
    let m = input("");
    for count in [9999, 10_000, 10_001] {
        let catalogue: Vec<_> = (0..count)
            .map(|i| entry(&format!("e{i}"), "x", false))
            .collect();
        assert_eq!(
            analyze_historical_catalogue(&m, &catalogue, &[], 1800, 0).is_ok(),
            count <= MAX_HISTORICAL_ENTRIES
        );
    }
    for bytes in [127, 128, 129] {
        assert_eq!(
            analyze_historical_catalogue(
                &m,
                &[entry("e", &"x".repeat(bytes), false)],
                &[],
                1800,
                0
            )
            .is_ok(),
            bytes <= MAX_HISTORICAL_TERM_BYTES
        );
    }
    for tokens in [31, 32, 33] {
        let term = format!("a{}", "+".repeat(tokens - 1));
        assert_eq!(
            analyze_historical_catalogue(&m, &[entry("e", &term, false)], &[], 1800, 0).is_ok(),
            tokens <= MAX_HISTORICAL_TERM_TOKENS
        );
    }
    for delta in [-1, 0, 1] {
        let mut catalogue: Vec<_> = (0..1024)
            .map(|i| HistoricalEntry {
                id: format!("e{i:04}"),
                term: "t".into(),
                earliest_year: 1,
                basis: HistoricalBasis::AuthorConstraint,
                reference: String::new(),
                context: "x".into(),
                sense_ambiguous: false,
            })
            .collect();
        let mut remaining = (MAX_HISTORICAL_METADATA_BYTES as isize + delta) as usize - 7 * 1024;
        for i in 0..1024 {
            let length = 4096.min(remaining - (1023 - i));
            catalogue[i].reference = "r".repeat(length);
            remaining -= length;
        }
        assert_eq!(remaining, 0);
        assert_eq!(
            analyze_historical_catalogue(&m, &catalogue, &[], 0, 0).is_ok(),
            delta <= 0
        );
    }
}

#[test]
fn candidate_usage_and_source_token_budgets_are_exact() {
    for count in [9999, 10_000, 10_001] {
        let m = input(&"t ".repeat(count));
        let catalogue = [entry("e", "t", true)];
        let usages: Vec<_> = (0..count)
            .map(|i| HistoricalUsage {
                id: format!("u{i}"),
                entry_id: "e".into(),
                evidence: AuthorSource {
                    scene_id: "s".into(),
                    utf16_start: (2 * i) as u32,
                    utf16_end: (2 * i + 1) as u32,
                    quote: "t".into(),
                },
            })
            .collect();
        let candidates = analyze_historical_catalogue(&m, &catalogue, &[], 0, 0);
        assert_eq!(candidates.is_ok(), count <= MAX_HISTORICAL_CANDIDATES);
        let annotated = analyze_historical_catalogue(&m, &catalogue, &usages, 0, 0);
        assert_eq!(annotated.is_ok(), count <= MAX_HISTORICAL_USAGES);
        if let Ok(result) = annotated {
            assert!(
                result
                    .candidates
                    .iter()
                    .all(|c| c.author_usage_id.is_some())
            );
        }
    }
    for count in [
        MAX_HISTORICAL_TOKENS - 1,
        MAX_HISTORICAL_TOKENS,
        MAX_HISTORICAL_TOKENS + 1,
    ] {
        let m = input(&"a ".repeat(count));
        let catalogue = [entry("e", "b", false)];
        assert_eq!(
            analyze_historical_catalogue(&m, &catalogue, &[], 0, 0).is_ok(),
            count <= MAX_HISTORICAL_TOKENS
        );
    }
}

#[test]
fn legacy_lexicon_checks_whole_phrases_and_abstains_on_ambiguous_senses() {
    let tokens = [
        "police",
        "cream",
        "cool",
        "awesome",
        "computer",
        "education",
        "republic",
    ]
    .map(str::to_string);
    let result = analyze_scene_checked(&tokens, 1200, 0).unwrap();
    assert!(result.items.is_empty());
    assert!(result.abstained_terms.contains(&"cool".into()));
    assert!(result.abstained_terms.contains(&"education".into()));
    let phrase = ["ice", "cream"].map(str::to_string);
    assert!(
        analyze_scene_checked(&phrase, 1200, 0)
            .unwrap()
            .items
            .iter()
            .any(|item| item.word == "ice cream")
    );
    assert!(analyze_scene_checked(&phrase, 1200, usize::MAX).is_err());
    for count in [
        MAX_HISTORICAL_TOKENS - 1,
        MAX_HISTORICAL_TOKENS,
        MAX_HISTORICAL_TOKENS + 1,
    ] {
        let tokens = vec!["x".into(); count];
        assert_eq!(
            analyze_scene_checked(&tokens, 1200, 0).is_ok(),
            count <= MAX_HISTORICAL_TOKENS
        );
    }
}
