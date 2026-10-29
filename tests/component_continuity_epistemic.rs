use narrative_lens::{
    continuity::{author_knowledge::*, epistemic_engine::*},
    manuscript::{Configuration, Manuscript, Scene, TextRange},
    substrate::utils::AnalysisAvailability,
};

fn input() -> Manuscript {
    Manuscript {
        document_id: "book".into(),
        scenes: vec![Scene::new("s", "\u{feff}😀 Alice told reader a lie.")],
        known_characters: vec!["Alice".into(), "reader".into(), "x".into()],
        configuration: Configuration::default(),
    }
}
fn source(m: &Manuscript) -> AuthorSource {
    let text = &m.scenes[0].text;
    let start = text.find("Alice").unwrap();
    let start = text[..start].encode_utf16().count() as u32;
    AuthorSource {
        scene_id: "s".into(),
        utf16_start: start,
        utf16_end: start + 5,
        quote: "Alice".into(),
    }
}
fn character(id: &str) -> AuthorAudience {
    AuthorAudience::Character {
        character_id: id.into(),
    }
}
fn proposition(m: &Manuscript, id: &str, truth: TruthValue) -> AuthorProposition {
    AuthorProposition {
        id: id.into(),
        description: "author declaration".into(),
        truth,
        evidence: source(m),
    }
}
fn event(m: &Manuscript, sequence: u32, action: AuthorKnowledgeAction) -> AuthorKnowledgeEvent {
    AuthorKnowledgeEvent {
        id: format!("e{sequence}"),
        sequence,
        proposition_id: "p".into(),
        action,
        evidence: source(m),
    }
}
fn query(id: &str, sequence: Option<u32>) -> AuthorEpistemicQuery {
    AuthorEpistemicQuery {
        id: id.into(),
        proposition_id: "p".into(),
        observer: AuthorAudience::Reader,
        subject: character("Alice"),
        at_sequence: sequence,
    }
}

#[test]
fn checkpoints_reclassify_truth_without_inventing_learning_or_ignorance() {
    use AuthorEpistemicRelation as R;
    use AuthorKnowledgeAction as A;
    let m = input();
    let p = [proposition(&m, "p", TruthValue::True)];
    let events = [
        event(
            &m,
            10,
            A::Believe {
                actor: AuthorAudience::Reader,
                value: true,
            },
        ),
        event(
            &m,
            20,
            A::Forget {
                actor: character("Alice"),
            },
        ),
        event(
            &m,
            30,
            A::Communicate {
                sender: AuthorAudience::Reader,
                recipient: character("Alice"),
                value: false,
                uptake: AuthorUptake::Belief,
            },
        ),
        event(
            &m,
            40,
            A::Know {
                actor: character("Alice"),
                value: true,
            },
        ),
        event(
            &m,
            50,
            A::Truth {
                value: TruthValue::False,
            },
        ),
        event(
            &m,
            60,
            A::Believe {
                actor: character("Alice"),
                value: false,
            },
        ),
        event(
            &m,
            70,
            A::Truth {
                value: TruthValue::Unknown,
            },
        ),
        event(
            &m,
            80,
            A::Uncertain {
                actor: AuthorAudience::Reader,
            },
        ),
        event(
            &m,
            90,
            A::Forget {
                actor: character("Alice"),
            },
        ),
        event(
            &m,
            100,
            A::Communicate {
                sender: AuthorAudience::Reader,
                recipient: character("Alice"),
                value: true,
                uptake: AuthorUptake::Unchanged,
            },
        ),
    ];
    let expected = [
        (0, R::Unassessed),
        (10, R::Unassessed),
        (20, R::ObserverAdvantage),
        (30, R::ObserverAdvantage),
        (40, R::SharedCorrect),
        (50, R::SharedIncorrect),
        (60, R::SubjectAdvantage),
        (70, R::ConflictingAssertions),
        (80, R::Undetermined),
        (90, R::SharedUncertainty),
        (100, R::SharedUncertainty),
    ];
    let queries: Vec<_> = expected
        .iter()
        .map(|(s, _)| query(&format!("q{s:03}"), Some(*s)))
        .collect();
    let result = analyze_author_epistemics(&m, &p, &events, &queries).unwrap();
    for (answer, (sequence, relation)) in result.answers.iter().zip(expected) {
        assert_eq!(answer.at_sequence, Some(sequence));
        assert_eq!(answer.relation, relation);
    }
    assert_eq!(result.assessed_queries, 9);
    assert_eq!(result.unassessed_queries, 2);
    assert_eq!(
        result.answers[1].subject.assertion,
        AuthorAssertion::Unassessed
    );
    assert_eq!(
        result.answers[2].subject.assertion,
        AuthorAssertion::Unknown
    );
    assert!(result.answers[3].assertions_conflict);
    assert_eq!(
        result.answers[5].subject.classification,
        AuthorKnowledgeClassification::ContradictedKnowledge
    );
    assert_eq!(result.answers[5].subject.event_id.as_deref(), Some("e40"));
    assert_eq!(result.answers[5].truth_event_id.as_deref(), Some("e50"));
    assert_eq!(result.answers[10].subject.event_id.as_deref(), Some("e90"));
    let lie = result
        .knowledge
        .transitions
        .iter()
        .find(|t| t.event_id == "e30")
        .unwrap();
    assert_eq!(
        lie.sender_assertion,
        Some(AuthorAssertion::Belief { value: true })
    );
    assert_eq!(lie.sender_event_id.as_deref(), Some("e10"));
    assert_eq!(lie.evidence.quote, "Alice");
    assert_eq!(lie.evidence.range.utf16_start, 4);
}

#[test]
fn final_and_max_sequence_queries_preserve_typed_reader_and_replay_order() {
    let m = input();
    let p = [proposition(&m, "p", TruthValue::False)];
    let events = [
        event(
            &m,
            u32::MAX,
            AuthorKnowledgeAction::Believe {
                actor: AuthorAudience::Reader,
                value: false,
            },
        ),
        event(
            &m,
            1,
            AuthorKnowledgeAction::Believe {
                actor: character("reader"),
                value: true,
            },
        ),
    ];
    let mut queries = [
        query("at-max", Some(u32::MAX)),
        query("before-max", Some(u32::MAX - 1)),
        query("final", None),
    ];
    for query in &mut queries {
        query.subject = character("reader");
    }
    let a = analyze_author_epistemics(&m, &p, &events, &queries).unwrap();
    assert_eq!(
        a.answers[0].observer.classification,
        AuthorKnowledgeClassification::CorrectBelief
    );
    assert_eq!(
        a.answers[0].subject.classification,
        AuthorKnowledgeClassification::FalseBelief
    );
    assert_eq!(a.answers[1].observer.assertion, AuthorAssertion::Unassessed);
    assert_eq!(
        a.answers[2].observer.event_id.as_deref(),
        Some("e4294967295")
    );
    queries.reverse();
    let mut reversed = events.to_vec();
    reversed.reverse();
    assert_eq!(
        a,
        analyze_author_epistemics(&m, &p, &reversed, &queries).unwrap()
    );
    assert_eq!(
        a,
        analyze_author_epistemics(&m, &p, &events, &queries).unwrap()
    );
}

#[test]
fn query_validation_is_atomic_and_source_validation_cannot_be_bypassed() {
    let m = input();
    let p = [proposition(&m, "p", TruthValue::True)];
    let valid = query("q", None);
    for invalid in [
        AuthorEpistemicQuery {
            id: "".into(),
            ..valid.clone()
        },
        AuthorEpistemicQuery {
            proposition_id: "typo".into(),
            ..valid.clone()
        },
        AuthorEpistemicQuery {
            subject: character("typo"),
            ..valid.clone()
        },
    ] {
        assert!(analyze_author_epistemics(&m, &p, &[], &[valid.clone(), invalid]).is_err());
    }
    assert!(analyze_author_epistemics(&m, &p, &[], &[valid.clone(), valid.clone()]).is_err());
    let mut bad = p.clone();
    bad[0].evidence.utf16_start = 2;
    assert!(analyze_author_epistemics(&m, &bad, &[], std::slice::from_ref(&valid)).is_err());
    let mut edited = m.clone();
    edited.scenes[0].text = edited.scenes[0].text.replace("Alice", "other");
    assert!(analyze_author_epistemics(&edited, &p, &[], &[valid]).is_err());
    let invalid_event = event(
        &m,
        1,
        AuthorKnowledgeAction::Know {
            actor: AuthorAudience::Reader,
            value: false,
        },
    );
    assert!(analyze_author_epistemics(&m, &p, &[invalid_event], &[]).is_err());
}

#[test]
fn query_count_id_and_metadata_limits_include_exact_boundaries() {
    let m = input();
    let p = [proposition(&m, "p", TruthValue::Unknown)];
    for count in [9999, 10_000, 10_001] {
        let queries: Vec<_> = (0..count).map(|i| query(&format!("q{i}"), None)).collect();
        assert_eq!(
            analyze_author_epistemics(&m, &p, &[], &queries).is_ok(),
            count <= MAX_AUTHOR_EPISTEMIC_QUERIES
        );
    }
    for bytes in [255, 256, 257] {
        assert_eq!(
            analyze_author_epistemics(&m, &p, &[], &[query(&"q".repeat(bytes), None)]).is_ok(),
            bytes <= 256
        );
    }
    let p = [proposition(&m, &"p".repeat(256), TruthValue::Unknown)];
    for delta in [-1, 0, 1] {
        let mut queries: Vec<_> = (0..8192)
            .map(|i| AuthorEpistemicQuery {
                id: format!("q{i:0255}"),
                proposition_id: p[0].id.clone(),
                observer: AuthorAudience::Reader,
                subject: AuthorAudience::Reader,
                at_sequence: None,
            })
            .collect();
        if delta == -1 {
            queries[0].id.pop();
        }
        if delta == 1 {
            queries[0].subject = character("x");
        }
        let bytes: usize = queries
            .iter()
            .map(|q| q.id.len() + q.proposition_id.len() + usize::from(q.subject == character("x")))
            .sum();
        assert_eq!(
            bytes as isize,
            MAX_AUTHOR_EPISTEMIC_QUERY_BYTES as isize + delta
        );
        assert_eq!(
            analyze_author_epistemics(&m, &p, &[], &queries).is_ok(),
            delta <= 0
        );
    }
}

#[test]
fn lexical_candidates_preserve_unicode_sources_and_never_invent_knowledge_or_suspense() {
    let text = "\u{feff}😀 Élodie discovered a hidden wooden door. \"The dragon is real,\" Alice lied to Bob.";
    let result = track_epistemics_checked(
        &[text, text],
        &["Élodie", "Alice", "Bob"],
        &["Alice", "Bob"],
    )
    .unwrap();
    assert_eq!(result.availability, AnalysisAvailability::Partial);
    assert_eq!(result.facts.len(), 2);
    assert_eq!(result.candidates.len(), 4);
    assert_eq!(result.analyzed_sentences, 4);
    assert_eq!(result.candidate_scene_coverage, Some(1.0));
    assert!(!result.semantic_metrics_available);
    assert!(
        result.asymmetries.is_empty()
            && result.revelations.is_empty()
            && result.suspense_curve.is_empty()
    );
    assert_eq!(result.manipulation_score, 0.0);
    for state in result.character_states.values() {
        assert!(
            state.knows.is_empty()
                && state.believes_false.is_empty()
                && state.uncertain_about.is_empty()
        );
    }
    for candidate in &result.candidates {
        let quote = &text[candidate.range.byte_start..candidate.range.byte_end];
        assert_eq!(
            TextRange::from_utf16(text, candidate.range.utf16_start, candidate.range.utf16_end)
                .unwrap(),
            candidate.range
        );
        assert_eq!(quote, result.facts[candidate.fact_id].content);
    }
    let a = track_epistemics_checked(&[text], &["Alice", "Élodie", "Bob"], &[]).unwrap();
    let b = track_epistemics_checked(&[text], &["Bob", "Alice", "Élodie"], &[]).unwrap();
    assert_eq!(
        serde_json::to_value(a.candidates).unwrap(),
        serde_json::to_value(b.candidates).unwrap()
    );
}

#[test]
fn lexical_boundaries_do_not_promote_substrings_or_quoted_questions() {
    let none = track_epistemics_checked(&["Sally walked around the valley quietly."], &["Al"], &[])
        .unwrap();
    assert!(none.candidates.is_empty());
    let question =
        track_epistemics_checked(&["\"Alice knew the secret was false?\""], &["Alice"], &[])
            .unwrap();
    assert!(question.candidates.is_empty());
    assert_eq!(
        track_epistemics_checked(&[], &[], &[])
            .unwrap()
            .candidate_scene_coverage,
        None
    );
    assert!(track_epistemics_checked(&["x"], &["Alice", "alice"], &[]).is_err());
    assert!(track_epistemics_checked(&["x"], &["Alice"], &["typo"]).is_err());
    assert!(track_epistemics_checked(&["x", "y"], &["Alice"], &["Alice"]).is_err());
    assert_eq!(
        track_epistemics(&["x"], &[&"x".repeat(200_000)], &[]).availability,
        AnalysisAvailability::Unavailable
    );
}

#[test]
fn lexical_sentence_marker_scene_and_cast_budgets_are_enforced() {
    for count in [9999, 10_000, 10_001] {
        let text = "Alice found a small parcel. ".repeat(count);
        assert_eq!(
            track_epistemics_checked(&[&text], &["Alice"], &[]).is_ok(),
            count <= MAX_EPISTEMIC_SENTENCES
        );
        let scenes = vec![""; count];
        assert_eq!(
            track_epistemics_checked(&scenes, &[], &[]).is_ok(),
            count <= MAX_EPISTEMIC_SCENES
        );
    }
    for count in [
        MAX_EPISTEMIC_MARKERS - 1,
        MAX_EPISTEMIC_MARKERS,
        MAX_EPISTEMIC_MARKERS + 1,
    ] {
        let text = format!("Alice found a small parcel{}", ".".repeat(count));
        assert_eq!(
            track_epistemics_checked(&[&text], &["Alice"], &[]).is_ok(),
            count <= MAX_EPISTEMIC_MARKERS
        );
    }
    for count in [255, 256, 257] {
        let names: Vec<_> = (0..count).map(|i| format!("actor{i}")).collect();
        let cast: Vec<_> = names.iter().map(String::as_str).collect();
        assert_eq!(
            track_epistemics_checked(&[], &cast, &[]).is_ok(),
            count <= MAX_EPISTEMIC_CHARACTERS
        );
    }
}
