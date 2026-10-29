use narrative_lens::{
    continuity::author_knowledge::*,
    manuscript::{Configuration, Manuscript, Scene},
};

fn manuscript() -> Manuscript {
    Manuscript {
        document_id: "knowledge".into(),
        scenes: vec![
            Scene::new("fact", "\u{feff}😀 The key is silver."),
            Scene::new("speech", "Alice told Bob the key is gold."),
            Scene::new("change", "Alice forgot. Bob learned the key is silver."),
        ],
        known_characters: vec!["Alice".into(), "Bob".into(), "reader".into()],
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

fn proposition(input: &Manuscript, truth: TruthValue) -> AuthorProposition {
    AuthorProposition {
        id: "key".into(),
        description: "The key is silver".into(),
        truth,
        evidence: source(input, "fact", "The key is silver."),
    }
}

fn actor(name: &str) -> AuthorAudience {
    AuthorAudience::Character {
        character_id: name.into(),
    }
}

fn event(input: &Manuscript, sequence: u32, action: AuthorKnowledgeAction) -> AuthorKnowledgeEvent {
    AuthorKnowledgeEvent {
        id: format!("e{sequence}"),
        sequence,
        proposition_id: "key".into(),
        action,
        evidence: source(input, "change", "Bob learned the key is silver."),
    }
}

fn state<'a>(
    ledger: &'a AuthorKnowledgeLedger,
    actor: &AuthorAudience,
) -> &'a AuthorKnowledgeState {
    ledger
        .states
        .iter()
        .find(|s| &s.actor == actor && s.proposition_id == "key")
        .unwrap()
}

#[test]
fn author_truth_does_not_teach_the_reader_and_reader_is_not_a_character_name() {
    let input = manuscript();
    let propositions = [proposition(&input, TruthValue::True)];
    let baseline = analyze_author_knowledge(&input, &propositions, &[]).unwrap();
    assert!(baseline.states.is_empty() && baseline.asymmetries.is_empty());
    assert_eq!(baseline.unassessed_pairs, 4);
    let evidence = &baseline.propositions[0].establishment;
    assert_eq!(
        (evidence.range.byte_start, evidence.range.utf16_start),
        (8, 4)
    );
    let events = [
        event(
            &input,
            10,
            AuthorKnowledgeAction::Believe {
                actor: AuthorAudience::Reader,
                value: false,
            },
        ),
        event(
            &input,
            20,
            AuthorKnowledgeAction::Know {
                actor: actor("reader"),
                value: true,
            },
        ),
    ];
    let ledger = analyze_author_knowledge(&input, &propositions, &events).unwrap();
    assert_eq!(
        state(&ledger, &AuthorAudience::Reader).classification,
        AuthorKnowledgeClassification::FalseBelief
    );
    assert_eq!(
        state(&ledger, &actor("reader")).classification,
        AuthorKnowledgeClassification::Knowledge
    );
    assert_eq!(ledger.asymmetries.len(), 1);
    assert_eq!(ledger.asymmetries[0].character_id, "reader");
    assert_eq!(ledger.asymmetries[0].proposition_id, "key");
    assert_eq!(
        ledger.asymmetries[0].reader_event_id.as_deref(),
        Some("e10")
    );
    assert_eq!(ledger.unassessed_pairs, 2);
}

#[test]
fn unknown_truth_uncertainty_forgetting_and_unassessed_are_distinct() {
    let input = manuscript();
    let propositions = [proposition(&input, TruthValue::Unknown)];
    let mut events = vec![
        event(
            &input,
            1,
            AuthorKnowledgeAction::Believe {
                actor: actor("Alice"),
                value: true,
            },
        ),
        event(
            &input,
            2,
            AuthorKnowledgeAction::Uncertain {
                actor: actor("Bob"),
            },
        ),
        event(
            &input,
            3,
            AuthorKnowledgeAction::Forget {
                actor: AuthorAudience::Reader,
            },
        ),
    ];
    let unknown = analyze_author_knowledge(&input, &propositions, &events).unwrap();
    assert_eq!(
        state(&unknown, &actor("Alice")).classification,
        AuthorKnowledgeClassification::UnverifiedBelief
    );
    assert_eq!(
        state(&unknown, &actor("Bob")).classification,
        AuthorKnowledgeClassification::Uncertain
    );
    assert_eq!(
        state(&unknown, &AuthorAudience::Reader).classification,
        AuthorKnowledgeClassification::Unknown
    );
    assert_eq!(unknown.unassessed_pairs, 1);
    events.push(event(
        &input,
        4,
        AuthorKnowledgeAction::Truth {
            value: TruthValue::False,
        },
    ));
    let false_truth = analyze_author_knowledge(&input, &propositions, &events).unwrap();
    assert_eq!(
        state(&false_truth, &actor("Alice")).classification,
        AuthorKnowledgeClassification::FalseBelief
    );
    assert_eq!(
        state(&false_truth, &AuthorAudience::Reader)
            .last_event_id
            .as_deref(),
        Some("e3")
    );
    events[3].action = AuthorKnowledgeAction::Truth {
        value: TruthValue::True,
    };
    let true_truth = analyze_author_knowledge(&input, &propositions, &events).unwrap();
    assert_eq!(
        state(&true_truth, &actor("Alice")).classification,
        AuthorKnowledgeClassification::CorrectBelief
    );
    assert_ne!(
        state(&true_truth, &actor("Alice")).classification,
        AuthorKnowledgeClassification::Knowledge
    );
    let only_character = analyze_author_knowledge(&input, &propositions, &events[..1]).unwrap();
    assert_eq!(
        only_character.asymmetries[0].reader_classification,
        AuthorKnowledgeClassification::Unassessed
    );
}

#[test]
fn truth_correction_reclassifies_without_automatic_learning_and_revelations_replace_state() {
    let input = manuscript();
    let propositions = [proposition(&input, TruthValue::True)];
    let mut events = vec![
        event(
            &input,
            1,
            AuthorKnowledgeAction::Know {
                actor: actor("Alice"),
                value: true,
            },
        ),
        event(
            &input,
            2,
            AuthorKnowledgeAction::Know {
                actor: AuthorAudience::Reader,
                value: true,
            },
        ),
        event(
            &input,
            3,
            AuthorKnowledgeAction::Truth {
                value: TruthValue::Unknown,
            },
        ),
    ];
    let unknown = analyze_author_knowledge(&input, &propositions, &events).unwrap();
    assert_eq!(
        state(&unknown, &actor("Alice")).classification,
        AuthorKnowledgeClassification::UnverifiedKnowledge
    );
    events.push(event(
        &input,
        4,
        AuthorKnowledgeAction::Truth {
            value: TruthValue::False,
        },
    ));
    let contradicted = analyze_author_knowledge(&input, &propositions, &events).unwrap();
    assert_eq!(
        state(&contradicted, &AuthorAudience::Reader).classification,
        AuthorKnowledgeClassification::ContradictedKnowledge
    );
    assert_eq!(
        state(&contradicted, &AuthorAudience::Reader)
            .last_event_id
            .as_deref(),
        Some("e2")
    );
    events.push(event(
        &input,
        5,
        AuthorKnowledgeAction::Know {
            actor: actor("Alice"),
            value: false,
        },
    ));
    events.push(event(
        &input,
        6,
        AuthorKnowledgeAction::Forget {
            actor: actor("Alice"),
        },
    ));
    let forgotten = analyze_author_knowledge(&input, &propositions, &events).unwrap();
    assert_eq!(
        state(&forgotten, &actor("Alice")).assertion,
        AuthorAssertion::Unknown
    );
    events.push(event(
        &input,
        7,
        AuthorKnowledgeAction::Know {
            actor: actor("Alice"),
            value: false,
        },
    ));
    events.push(event(
        &input,
        8,
        AuthorKnowledgeAction::Know {
            actor: actor("Alice"),
            value: false,
        },
    ));
    let corrected = analyze_author_knowledge(&input, &propositions, &events).unwrap();
    assert_eq!(corrected.states.len(), 2);
    assert_eq!(
        state(&corrected, &actor("Alice")).assertion,
        AuthorAssertion::Knowledge { value: false }
    );
    assert_eq!(
        state(&corrected, &actor("Alice")).classification,
        AuthorKnowledgeClassification::Knowledge
    );
    assert_eq!(
        corrected.transitions[4].previous_event_id.as_deref(),
        Some("e1")
    );
    assert_eq!(
        corrected.transitions[4].previous_assertion,
        Some(AuthorAssertion::Knowledge { value: true })
    );
    assert_eq!(
        corrected.transitions[7].previous_event_id.as_deref(),
        Some("e7")
    );
    assert_eq!(
        corrected.propositions[0].last_event_id.as_deref(),
        Some("e4")
    );
    assert_eq!(corrected.propositions[0].initial_truth, TruthValue::True);
}

#[test]
fn communication_preserves_said_values_roles_and_transmission_provenance() {
    let input = manuscript();
    let propositions = [proposition(&input, TruthValue::True)];
    let communicate = |sender, recipient, value, uptake| AuthorKnowledgeAction::Communicate {
        sender,
        recipient,
        value,
        uptake,
    };
    let mut events = vec![
        event(
            &input,
            1,
            AuthorKnowledgeAction::Know {
                actor: actor("Alice"),
                value: true,
            },
        ),
        event(
            &input,
            2,
            communicate(actor("Alice"), actor("Bob"), false, AuthorUptake::Belief),
        ),
        event(
            &input,
            3,
            communicate(
                actor("Bob"),
                AuthorAudience::Reader,
                false,
                AuthorUptake::Belief,
            ),
        ),
        event(
            &input,
            4,
            communicate(actor("Bob"), actor("Alice"), true, AuthorUptake::Unchanged),
        ),
    ];
    let ledger = analyze_author_knowledge(&input, &propositions, &events).unwrap();
    assert_eq!(
        state(&ledger, &actor("Alice")).last_event_id.as_deref(),
        Some("e1")
    );
    assert_eq!(
        state(&ledger, &actor("Bob")).classification,
        AuthorKnowledgeClassification::FalseBelief
    );
    assert_eq!(
        state(&ledger, &AuthorAudience::Reader).classification,
        AuthorKnowledgeClassification::FalseBelief
    );
    assert_eq!(
        ledger.transitions[1].sender_assertion,
        Some(AuthorAssertion::Knowledge { value: true })
    );
    assert_eq!(ledger.transitions[1].sender_event_id.as_deref(), Some("e1"));
    assert_eq!(ledger.transitions[2].sender_event_id.as_deref(), Some("e2"));
    assert_eq!(
        ledger.transitions[3].previous_event_id.as_deref(),
        Some("e1")
    );
    assert_eq!(
        ledger.transitions[3].sender_assertion,
        Some(AuthorAssertion::Belief { value: false })
    );
    assert!(matches!(
        ledger.transitions[1].action,
        AuthorKnowledgeAction::Communicate { value: false, .. }
    ));
    events.push(event(
        &input,
        5,
        communicate(actor("reader"), actor("Bob"), true, AuthorUptake::Knowledge),
    ));
    let corrected = analyze_author_knowledge(&input, &propositions, &events).unwrap();
    assert_eq!(
        state(&corrected, &actor("Bob")).classification,
        AuthorKnowledgeClassification::Knowledge
    );
    assert_eq!(
        corrected.transitions[4].sender_assertion,
        Some(AuthorAssertion::Unassessed)
    );
    assert!(corrected.transitions[4].sender_event_id.is_none());
    assert!(!corrected.states.iter().any(|s| s.actor == actor("reader")));
    events[4].action = communicate(actor("Alice"), actor("Bob"), false, AuthorUptake::Knowledge);
    assert!(analyze_author_knowledge(&input, &propositions, &events).is_err());
}

#[test]
fn explicit_sequence_and_stable_ids_control_replay_reorder_and_replacement() {
    let mut input = manuscript();
    let mut propositions = vec![proposition(&input, TruthValue::True)];
    let mut extra = propositions[0].clone();
    extra.id = "other".into();
    propositions.push(extra);
    let mut events = vec![
        event(
            &input,
            20,
            AuthorKnowledgeAction::Believe {
                actor: actor("Bob"),
                value: false,
            },
        ),
        event(
            &input,
            10,
            AuthorKnowledgeAction::Truth {
                value: TruthValue::False,
            },
        ),
        event(
            &input,
            u32::MAX,
            AuthorKnowledgeAction::Know {
                actor: AuthorAudience::Reader,
                value: false,
            },
        ),
    ];
    let before = analyze_author_knowledge(&input, &propositions, &events).unwrap();
    assert_eq!(
        state(&before, &actor("Bob")).classification,
        AuthorKnowledgeClassification::CorrectBelief
    );
    assert_eq!(
        before
            .transitions
            .iter()
            .map(|t| t.sequence)
            .collect::<Vec<_>>(),
        [10, 20, u32::MAX]
    );
    input.scenes.reverse();
    propositions.reverse();
    events.reverse();
    assert_eq!(
        analyze_author_knowledge(&input, &propositions, &events).unwrap(),
        before
    );
    assert_eq!(
        analyze_author_knowledge(&input, &propositions, &events).unwrap(),
        before
    );
    events
        .iter_mut()
        .find(|event| event.id == "e20")
        .unwrap()
        .action = AuthorKnowledgeAction::Forget {
        actor: actor("Bob"),
    };
    let after = analyze_author_knowledge(&input, &propositions, &events).unwrap();
    assert_eq!(
        state(&after, &actor("Bob")).assertion,
        AuthorAssertion::Unknown
    );
    assert_eq!(after.transitions.len(), before.transitions.len());
    assert_eq!(
        after.transitions[1].event_id,
        before.transitions[1].event_id
    );
}

#[test]
fn shared_author_fixture_preserves_the_adapter_contract() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/author-knowledge.json")).unwrap();
    let manuscript: Manuscript = serde_json::from_value(fixture["manuscript"].clone()).unwrap();
    let propositions: Vec<AuthorProposition> =
        serde_json::from_value(fixture["propositions"].clone()).unwrap();
    let events: Vec<AuthorKnowledgeEvent> =
        serde_json::from_value(fixture["events"].clone()).unwrap();
    let ledger = analyze_author_knowledge(&manuscript, &propositions, &events).unwrap();
    assert_eq!(ledger.propositions.len(), 2);
    assert_eq!(ledger.states.len(), 4);
    assert_eq!(ledger.unassessed_pairs, 4);
    let classification = |actor| {
        ledger
            .states
            .iter()
            .find(|s| s.actor == actor && s.proposition_id == "hidden-key")
            .unwrap()
            .classification
    };
    assert_eq!(
        classification(AuthorAudience::Reader),
        AuthorKnowledgeClassification::Knowledge
    );
    assert_eq!(
        classification(actor("reader")),
        AuthorKnowledgeClassification::FalseBelief
    );
    assert_eq!(
        classification(actor("Bob")),
        AuthorKnowledgeClassification::FalseBelief
    );
    assert_eq!(
        classification(actor("Alice")),
        AuthorKnowledgeClassification::CorrectBelief
    );
    let communication = ledger
        .transitions
        .iter()
        .find(|t| t.event_id == "lie-to-bob")
        .unwrap();
    assert_eq!(
        communication.sender_event_id.as_deref(),
        Some("alice-knows")
    );
    assert_eq!(
        communication.sender_assertion,
        Some(AuthorAssertion::Knowledge { value: true })
    );
}

#[test]
fn invalid_annotations_fail_atomically_including_late_sequence_knowledge() {
    let input = manuscript();
    let propositions = vec![proposition(&input, TruthValue::True)];
    let valid = vec![event(
        &input,
        1,
        AuthorKnowledgeAction::Believe {
            actor: actor("Alice"),
            value: false,
        },
    )];
    let before = analyze_author_knowledge(&input, &propositions, &valid).unwrap();
    let mut invalid = valid.clone();
    invalid.push(event(
        &input,
        2,
        AuthorKnowledgeAction::Know {
            actor: actor("Alice"),
            value: false,
        },
    ));
    let snapshot = serde_json::to_value(&invalid).unwrap();
    assert!(analyze_author_knowledge(&input, &propositions, &invalid).is_err());
    assert_eq!(serde_json::to_value(invalid).unwrap(), snapshot);
    for action in [
        AuthorKnowledgeAction::Know {
            actor: actor("alice"),
            value: true,
        },
        AuthorKnowledgeAction::Believe {
            actor: actor("absent"),
            value: false,
        },
    ] {
        assert!(
            analyze_author_knowledge(&input, &propositions, &[event(&input, 1, action)]).is_err()
        );
    }
    let mut duplicate = valid.clone();
    let mut other = valid[0].clone();
    other.id = "another".into();
    duplicate.push(other);
    assert!(analyze_author_knowledge(&input, &propositions, &duplicate).is_err());
    duplicate[1].sequence = 2;
    duplicate[1].id = duplicate[0].id.clone();
    assert!(analyze_author_knowledge(&input, &propositions, &duplicate).is_err());
    duplicate = valid.clone();
    duplicate[0].proposition_id = "missing".into();
    assert!(analyze_author_knowledge(&input, &propositions, &duplicate).is_err());
    assert!(
        analyze_author_knowledge(
            &input,
            &[propositions[0].clone(), propositions[0].clone()],
            &valid
        )
        .is_err()
    );
    let unknown = [proposition(&input, TruthValue::Unknown)];
    assert!(
        analyze_author_knowledge(
            &input,
            &unknown,
            &[event(
                &input,
                1,
                AuthorKnowledgeAction::Know {
                    actor: actor("Alice"),
                    value: false
                }
            )]
        )
        .is_err()
    );
    let mut json = serde_json::to_value(&valid[0]).unwrap();
    for bad in [
        serde_json::json!(-1),
        serde_json::json!(4_294_967_296u64),
        serde_json::json!(1.5),
    ] {
        json["sequence"] = bad;
        assert!(serde_json::from_value::<AuthorKnowledgeEvent>(json.clone()).is_err());
    }
    json = serde_json::to_value(&valid[0]).unwrap();
    json["unexpected"] = true.into();
    assert!(serde_json::from_value::<AuthorKnowledgeEvent>(json).is_err());
    assert_eq!(
        analyze_author_knowledge(&input, &propositions, &valid).unwrap(),
        before
    );
}

#[test]
fn tagged_actor_and_action_inputs_reject_role_smuggling_and_wrong_types() {
    for kind in ["unassessed", "unknown", "uncertain"] {
        assert!(
            serde_json::from_value::<AuthorAssertion>(
                serde_json::json!({"kind":kind,"value":true})
            )
            .is_err()
        );
    }
    for value in [
        serde_json::json!({"kind":"reader","characterId":"Alice"}),
        serde_json::json!({"kind":"character","characterId":"Alice","extra":true}),
        serde_json::json!({"kind":"character","characterId":false}),
        serde_json::json!({"kind":"Reader"}),
    ] {
        assert!(serde_json::from_value::<AuthorAudience>(value).is_err());
    }
    for value in [
        serde_json::json!({"kind":"know","actor":{"kind":"reader"},"value":"true"}),
        serde_json::json!({"kind":"forget","actor":{"kind":"reader"},"value":true}),
        serde_json::json!({"kind":"communicate","sender":{"kind":"reader"},"recipient":null,"value":true,"uptake":"knowledge"}),
    ] {
        assert!(serde_json::from_value::<AuthorKnowledgeAction>(value).is_err());
    }
}

#[test]
fn source_unicode_boundaries_and_stale_edits_require_exact_reanchoring() {
    let mut input = manuscript();
    let mut propositions = [proposition(&input, TruthValue::False)];
    let events = [event(
        &input,
        1,
        AuthorKnowledgeAction::Believe {
            actor: actor("Alice"),
            value: true,
        },
    )];
    let ledger = analyze_author_knowledge(&input, &propositions, &events).unwrap();
    assert_eq!(
        state(&ledger, &actor("Alice")).classification,
        AuthorKnowledgeClassification::FalseBelief
    );
    for evidence in ledger
        .propositions
        .iter()
        .map(|p| &p.establishment)
        .chain(ledger.transitions.iter().map(|t| &t.evidence))
    {
        let text = &input
            .scenes
            .iter()
            .find(|s| s.id == evidence.scene_id)
            .unwrap()
            .text;
        assert_eq!(
            &text[evidence.range.byte_start..evidence.range.byte_end],
            evidence.quote
        );
        let utf16: Vec<_> = text.encode_utf16().collect();
        assert_eq!(
            String::from_utf16(
                &utf16[evidence.range.utf16_start as usize..evidence.range.utf16_end as usize]
            )
            .unwrap(),
            evidence.quote
        );
    }
    for start in [2, u32::MAX] {
        let mut bad = propositions.clone();
        bad[0].evidence.utf16_start = start;
        assert!(analyze_author_knowledge(&input, &bad, &events).is_err());
    }
    let mut bad_event = events.clone();
    bad_event[0].evidence.scene_id = "absent".into();
    assert!(analyze_author_knowledge(&input, &propositions, &bad_event).is_err());
    input.scenes[0].text.insert_str(0, "Later: ");
    assert!(analyze_author_knowledge(&input, &propositions, &events).is_err());
    propositions[0].evidence = source(&input, "fact", "The key is silver.");
    let refreshed = analyze_author_knowledge(&input, &propositions, &events).unwrap();
    assert_eq!(refreshed.propositions[0].id, "key");
    input.scenes[0].text = input.scenes[0].text.replace("silver", "gold");
    assert!(analyze_author_knowledge(&input, &propositions, &events).is_err());
}

#[test]
fn count_and_field_limits_bound_sparse_work_at_limit_minus_equal_plus_one() {
    let mut input = manuscript();
    input.scenes = vec![Scene::new("s", "x".repeat(65_537))];
    let make = |id: String, description: String, length: usize| AuthorProposition {
        id,
        description,
        truth: TruthValue::Unknown,
        evidence: source(&input, "s", &"x".repeat(length)),
    };
    for size in [255, 256, 257] {
        assert_eq!(
            analyze_author_knowledge(&input, &[make("p".repeat(size), "d".into(), 1)], &[]).is_ok(),
            size <= 256
        );
    }
    for size in [4095, 4096, 4097] {
        assert_eq!(
            analyze_author_knowledge(&input, &[make("p".into(), "d".repeat(size), 1)], &[]).is_ok(),
            size <= 4096
        );
    }
    for size in [65_535, 65_536, 65_537] {
        assert_eq!(
            analyze_author_knowledge(&input, &[make("p".into(), "d".into(), size)], &[]).is_ok(),
            size <= 65_536
        );
    }
    let many: Vec<_> = (0..=MAX_AUTHOR_PROPOSITIONS)
        .map(|i| make(format!("p{i}"), "d".into(), 1))
        .collect();
    for count in [MAX_AUTHOR_PROPOSITIONS - 1, MAX_AUTHOR_PROPOSITIONS] {
        let ledger = analyze_author_knowledge(&input, &many[..count], &[]).unwrap();
        assert_eq!(ledger.propositions.len(), count);
        assert!(ledger.states.is_empty() && ledger.transitions.is_empty());
        assert_eq!(ledger.unassessed_pairs, count * 4);
    }
    assert!(analyze_author_knowledge(&input, &many, &[]).is_err());
    let one = [make("key".into(), "d".into(), 1)];
    let many_events: Vec<_> = (0..=MAX_AUTHOR_KNOWLEDGE_EVENTS)
        .map(|i| AuthorKnowledgeEvent {
            id: format!("e{i}"),
            sequence: i as u32,
            proposition_id: "key".into(),
            action: AuthorKnowledgeAction::Truth {
                value: TruthValue::False,
            },
            evidence: source(&input, "s", "x"),
        })
        .collect();
    for count in [MAX_AUTHOR_KNOWLEDGE_EVENTS - 1, MAX_AUTHOR_KNOWLEDGE_EVENTS] {
        let ledger = analyze_author_knowledge(&input, &one, &many_events[..count]).unwrap();
        assert_eq!(ledger.transitions.len(), count);
        assert!(ledger.states.is_empty());
    }
    assert!(analyze_author_knowledge(&input, &one, &many_events).is_err());
}

#[test]
fn annotation_byte_budget_is_checked_exactly_before_source_expansion() {
    let mut input = manuscript();
    input.scenes = vec![Scene::new("s", "x".repeat(65_536))];
    for target in [
        MAX_AUTHOR_KNOWLEDGE_BYTES - 1,
        MAX_AUTHOR_KNOWLEDGE_BYTES,
        MAX_AUTHOR_KNOWLEDGE_BYTES + 1,
    ] {
        let mut remaining = target;
        let mut propositions = Vec::new();
        while remaining > 0 {
            let id = format!("p{}", propositions.len());
            let overhead = id.len() + 2;
            let mut length = (remaining - overhead).min(65_536);
            if (1..32).contains(&(remaining - overhead - length)) {
                length -= 32;
            }
            propositions.push(AuthorProposition {
                id,
                description: "d".into(),
                truth: TruthValue::Unknown,
                evidence: source(&input, "s", &"x".repeat(length)),
            });
            remaining -= overhead + length;
        }
        assert_eq!(
            analyze_author_knowledge(&input, &propositions, &[]).is_ok(),
            target <= MAX_AUTHOR_KNOWLEDGE_BYTES
        );
    }
}
