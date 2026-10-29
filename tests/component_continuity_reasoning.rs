use narrative_lens::{
    continuity::{author_knowledge::*, causal_engine::*, mental_model::*},
    manuscript::{Configuration, Manuscript, Scene},
    substrate::utils::AnalysisAvailability,
};

fn manuscript() -> Manuscript {
    Manuscript {
        document_id: "model".into(),
        scenes: vec![Scene::new("s", "\u{feff}😀 x")],
        known_characters: vec!["Alice".into(), "reader".into()],
        configuration: Configuration::default(),
    }
}
fn source(input: &Manuscript, quote: &str) -> AuthorSource {
    let text = &input.scenes[0].text;
    let byte = text.find(quote).unwrap();
    let start = text[..byte].encode_utf16().count() as u32;
    AuthorSource {
        scene_id: "s".into(),
        utf16_start: start,
        utf16_end: start + quote.encode_utf16().count() as u32,
        quote: quote.into(),
    }
}
fn node(input: &Manuscript, id: &str, mechanism: AuthorCausalMechanism) -> AuthorCausalNode {
    AuthorCausalNode {
        id: id.into(),
        mechanism,
        evidence: source(input, "x"),
    }
}
fn model(input: &Manuscript) -> Vec<AuthorCausalNode> {
    vec![
        node(input, "a", AuthorCausalMechanism::Fixed { value: true }),
        node(input, "b", AuthorCausalMechanism::Fixed { value: true }),
        node(
            input,
            "any",
            AuthorCausalMechanism::Any {
                parents: vec!["b".into(), "a".into()],
            },
        ),
        node(
            input,
            "all",
            AuthorCausalMechanism::All {
                parents: vec!["a".into(), "b".into()],
            },
        ),
        node(
            input,
            "not",
            AuthorCausalMechanism::Not { parent: "a".into() },
        ),
        node(
            input,
            "end",
            AuthorCausalMechanism::All {
                parents: vec!["any".into(), "not".into()],
            },
        ),
    ]
}
fn value<'a>(result: &'a AuthorCausalResult, id: &str) -> &'a AuthorCausalValue {
    result.nodes.iter().find(|node| node.id == id).unwrap()
}

#[test]
fn interventions_preserve_alternative_causes_and_evaluate_negative_and_joint_mechanisms() {
    let input = manuscript();
    let nodes = model(&input);
    let intervention = [AuthorIntervention {
        node_id: "a".into(),
        value: false,
    }];
    let result = analyze_author_causality(&input, &nodes, &intervention).unwrap();
    assert!(value(&result, "any").baseline && value(&result, "any").intervened);
    assert!(value(&result, "all").baseline && !value(&result, "all").intervened);
    assert!(!value(&result, "not").baseline && value(&result, "not").intervened);
    assert!(!value(&result, "end").baseline && value(&result, "end").intervened);
    assert_eq!(
        result
            .changes
            .iter()
            .map(|change| change.node_id.as_str())
            .collect::<Vec<_>>(),
        ["a", "all", "end", "not"]
    );
    assert!(result.changes[0].direct_intervention);
    assert!(
        result.changes[1..]
            .iter()
            .all(|change| !change.direct_intervention)
    );
    assert_eq!(
        (
            result.nodes[0].evidence.range.byte_start,
            result.nodes[0].evidence.range.utf16_start
        ),
        (8, 4)
    );
    let joint = analyze_author_causality(
        &input,
        &nodes,
        &[
            intervention[0].clone(),
            AuthorIntervention {
                node_id: "not".into(),
                value: false,
            },
        ],
    )
    .unwrap();
    assert!(!value(&joint, "not").intervened && !value(&joint, "end").intervened);
    assert!(!joint.changes.iter().any(|change| change.node_id == "not"));
    assert_eq!(joint.interventions.len(), 2);
}

#[test]
fn causal_models_replay_canonically_and_validate_all_references_atomically() {
    let mut input = manuscript();
    let mut nodes = model(&input);
    let mut interventions = vec![
        AuthorIntervention {
            node_id: "a".into(),
            value: false,
        },
        AuthorIntervention {
            node_id: "b".into(),
            value: false,
        },
    ];
    let expected = analyze_author_causality(&input, &nodes, &interventions).unwrap();
    nodes.reverse();
    interventions.reverse();
    input.scenes.reverse();
    for node in &mut nodes {
        if let AuthorCausalMechanism::All { parents } | AuthorCausalMechanism::Any { parents } =
            &mut node.mechanism
        {
            parents.reverse();
        }
    }
    assert_eq!(
        analyze_author_causality(&input, &nodes, &interventions).unwrap(),
        expected
    );
    let before = serde_json::to_value(&nodes).unwrap();
    let mut bad = nodes.clone();
    bad.push(nodes[0].clone());
    assert!(analyze_author_causality(&input, &bad, &[]).is_err());
    for mechanism in [
        AuthorCausalMechanism::Not {
            parent: "missing".into(),
        },
        AuthorCausalMechanism::All { parents: vec![] },
        AuthorCausalMechanism::Any {
            parents: vec!["a".into(), "a".into()],
        },
    ] {
        let bad = vec![
            node(&input, "a", AuthorCausalMechanism::Fixed { value: true }),
            node(&input, "c", mechanism),
        ];
        assert!(analyze_author_causality(&input, &bad, &[]).is_err());
    }
    let cycle = [
        node(
            &input,
            "a",
            AuthorCausalMechanism::Not { parent: "b".into() },
        ),
        node(
            &input,
            "b",
            AuthorCausalMechanism::Not { parent: "a".into() },
        ),
    ];
    assert!(analyze_author_causality(&input, &cycle, &[]).is_err());
    assert!(
        analyze_author_causality(
            &input,
            &nodes,
            &[AuthorIntervention {
                node_id: "missing".into(),
                value: false
            }]
        )
        .is_err()
    );
    assert!(
        analyze_author_causality(
            &input,
            &nodes,
            &[interventions[0].clone(), interventions[0].clone()]
        )
        .is_err()
    );
    assert_eq!(serde_json::to_value(&nodes).unwrap(), before);
    input.scenes[0].text.insert_str(0, "edit");
    assert!(analyze_author_causality(&input, &nodes, &interventions).is_err());
}

#[test]
fn boolean_node_parent_and_edge_limits_include_deep_nonrecursive_graphs() {
    let input = manuscript();
    let nodes: Vec<_> = (0..=MAX_AUTHOR_CAUSAL_NODES)
        .map(|i| {
            node(
                &input,
                &format!("n{i:05}"),
                if i == 0 {
                    AuthorCausalMechanism::Fixed { value: true }
                } else {
                    AuthorCausalMechanism::Not {
                        parent: format!("n{:05}", i - 1),
                    }
                },
            )
        })
        .collect();
    for count in [MAX_AUTHOR_CAUSAL_NODES - 1, MAX_AUTHOR_CAUSAL_NODES] {
        let result = analyze_author_causality(&input, &nodes[..count], &[]).unwrap();
        assert_eq!(result.topological_order.len(), count);
        assert_eq!(result.nodes.last().unwrap().baseline, count % 2 == 1);
        assert!(result.changes.is_empty());
    }
    assert!(analyze_author_causality(&input, &nodes, &[]).is_err());
    for count in [31, 32, 33] {
        let mut nodes: Vec<_> = (0..count)
            .map(|i| {
                node(
                    &input,
                    &format!("r{i}"),
                    AuthorCausalMechanism::Fixed { value: true },
                )
            })
            .collect();
        nodes.push(node(
            &input,
            "out",
            AuthorCausalMechanism::All {
                parents: (0..count).map(|i| format!("r{i}")).collect(),
            },
        ));
        assert_eq!(
            analyze_author_causality(&input, &nodes, &[]).is_ok(),
            count <= 32
        );
    }
    for count in [
        MAX_AUTHOR_CAUSAL_EDGES - 1,
        MAX_AUTHOR_CAUSAL_EDGES,
        MAX_AUTHOR_CAUSAL_EDGES + 1,
    ] {
        let mut nodes: Vec<_> = (0..32)
            .map(|i| {
                node(
                    &input,
                    &format!("r{i}"),
                    AuthorCausalMechanism::Fixed { value: true },
                )
            })
            .collect();
        let mut remaining = count;
        while remaining > 0 {
            let parents = remaining.min(32);
            nodes.push(node(
                &input,
                &format!("out{}", nodes.len()),
                AuthorCausalMechanism::All {
                    parents: (0..parents).map(|i| format!("r{i}")).collect(),
                },
            ));
            remaining -= parents;
        }
        let result = analyze_author_causality(&input, &nodes, &[]);
        assert_eq!(result.is_ok(), count <= MAX_AUTHOR_CAUSAL_EDGES);
        if let Ok(result) = result {
            assert_eq!(result.edges.len(), count);
        }
    }
}

#[test]
fn causal_field_and_total_metadata_boundaries_are_exact() {
    let mut input = manuscript();
    input.scenes[0].text = "x".repeat(65_537);
    for count in [255, 256, 257] {
        let nodes = [node(
            &input,
            &"n".repeat(count),
            AuthorCausalMechanism::Fixed { value: true },
        )];
        assert_eq!(
            analyze_author_causality(&input, &nodes, &[]).is_ok(),
            count <= 256
        );
    }
    for count in [65_535, 65_536, 65_537] {
        let mut nodes = [node(
            &input,
            "n",
            AuthorCausalMechanism::Fixed { value: true },
        )];
        nodes[0].evidence = source(&input, &"x".repeat(count));
        assert_eq!(
            analyze_author_causality(&input, &nodes, &[]).is_ok(),
            count <= 65_536
        );
    }
    for target in [
        MAX_AUTHOR_CAUSAL_BYTES - 1,
        MAX_AUTHOR_CAUSAL_BYTES,
        MAX_AUTHOR_CAUSAL_BYTES + 1,
    ] {
        let mut nodes = Vec::new();
        let mut remaining = target;
        while remaining > 0 {
            let id = format!("n{}", nodes.len());
            let overhead = id.len() + 1;
            let mut quote = (remaining - overhead).min(65_536);
            if (1..32).contains(&(remaining - overhead - quote)) {
                quote -= 32;
            }
            nodes.push(AuthorCausalNode {
                id,
                mechanism: AuthorCausalMechanism::Fixed { value: true },
                evidence: source(&input, &"x".repeat(quote)),
            });
            remaining -= overhead + quote;
        }
        assert_eq!(
            analyze_author_causality(&input, &nodes, &[]).is_ok(),
            target <= MAX_AUTHOR_CAUSAL_BYTES
        );
    }
}

#[test]
fn lexical_associations_do_not_assert_counterfactual_necessity_or_plot_holes() {
    let result = analyze_causality_checked(&["The king declared the devastating war. Because of the devastating war, thousands fled. Therefore the economy completely collapsed."], &["king"]).unwrap();
    assert_eq!(result.availability, AnalysisAvailability::Partial);
    assert!(!result.links.is_empty());
    assert!(
        result
            .links
            .iter()
            .all(|link| link.link_type == CausalLinkType::Association)
    );
    assert!(result.counterfactuals.is_empty() && result.plot_holes.is_empty());
    assert!(
        result
            .structural_insights
            .iter()
            .all(|s| !s.contains("would eliminate") && !s.contains("load-bearing"))
    );
    assert!(analyze_causality_checked(&["Alice waited."], &[&"x".repeat(200_000)]).is_err());
    assert_eq!(
        analyze_causality(&["Alice waited."], &[&"x".repeat(200_000)]).availability,
        AnalysisAvailability::Unavailable
    );
}

fn proposition(input: &Manuscript) -> AuthorProposition {
    AuthorProposition {
        id: "p".into(),
        description: "Declared proposition".into(),
        truth: TruthValue::True,
        evidence: source(input, "x"),
    }
}
fn event(input: &Manuscript, sequence: u32, action: AuthorKnowledgeAction) -> AuthorKnowledgeEvent {
    AuthorKnowledgeEvent {
        id: format!("e{sequence}"),
        sequence,
        proposition_id: "p".into(),
        action,
        evidence: source(input, "x"),
    }
}

#[test]
fn reader_projection_distinguishes_truth_reclassification_from_reader_learning() {
    let input = manuscript();
    let propositions = [proposition(&input)];
    let events = [
        event(
            &input,
            1,
            AuthorKnowledgeAction::Believe {
                actor: AuthorAudience::Reader,
                value: false,
            },
        ),
        event(
            &input,
            2,
            AuthorKnowledgeAction::Truth {
                value: TruthValue::Unknown,
            },
        ),
        event(
            &input,
            3,
            AuthorKnowledgeAction::Truth {
                value: TruthValue::False,
            },
        ),
        event(
            &input,
            4,
            AuthorKnowledgeAction::Know {
                actor: AuthorAudience::Reader,
                value: false,
            },
        ),
        event(
            &input,
            5,
            AuthorKnowledgeAction::Truth {
                value: TruthValue::True,
            },
        ),
    ];
    let result = build_author_mental_model(&input, &propositions, &events).unwrap();
    assert_eq!(result.reader_changes.len(), 5);
    let expected = [
        AuthorKnowledgeClassification::FalseBelief,
        AuthorKnowledgeClassification::UnverifiedBelief,
        AuthorKnowledgeClassification::CorrectBelief,
        AuthorKnowledgeClassification::Knowledge,
        AuthorKnowledgeClassification::ContradictedKnowledge,
    ];
    assert_eq!(
        result
            .reader_changes
            .iter()
            .map(|change| change.after_classification)
            .collect::<Vec<_>>(),
        expected
    );
    for index in [1, 2, 4] {
        assert_eq!(
            result.reader_changes[index].kind,
            AuthorReaderChangeKind::TruthReclassification
        );
        assert_eq!(
            result.reader_changes[index].before,
            result.reader_changes[index].after
        );
    }
    assert_eq!(
        result.knowledge.states[0].last_event_id.as_deref(),
        Some("e4")
    );
    assert_eq!(result.reader_summary.contradicted_knowledge, 1);
    assert_eq!(result.reader_summary.annotated_propositions, 1);
    assert_eq!(result.reader_summary.unassessed_propositions, 0);
}

#[test]
fn reader_projection_preserves_unassessed_and_explicit_uptake_and_repeated_assertions() {
    let mut input = manuscript();
    let propositions = [proposition(&input)];
    let baseline = build_author_mental_model(
        &input,
        &propositions,
        &[event(
            &input,
            1,
            AuthorKnowledgeAction::Truth {
                value: TruthValue::False,
            },
        )],
    )
    .unwrap();
    assert!(baseline.reader_changes.is_empty());
    assert_eq!(baseline.reader_summary.unassessed_propositions, 1);
    let mut events = vec![
        event(
            &input,
            10,
            AuthorKnowledgeAction::Know {
                actor: AuthorAudience::Character {
                    character_id: "reader".into(),
                },
                value: true,
            },
        ),
        event(
            &input,
            20,
            AuthorKnowledgeAction::Uncertain {
                actor: AuthorAudience::Reader,
            },
        ),
        event(
            &input,
            30,
            AuthorKnowledgeAction::Uncertain {
                actor: AuthorAudience::Reader,
            },
        ),
        event(
            &input,
            40,
            AuthorKnowledgeAction::Communicate {
                sender: AuthorAudience::Character {
                    character_id: "Alice".into(),
                },
                recipient: AuthorAudience::Reader,
                value: false,
                uptake: AuthorUptake::Unchanged,
            },
        ),
        event(
            &input,
            50,
            AuthorKnowledgeAction::Forget {
                actor: AuthorAudience::Reader,
            },
        ),
    ];
    let result = build_author_mental_model(&input, &propositions, &events).unwrap();
    assert_eq!(
        result
            .reader_changes
            .iter()
            .map(|c| c.sequence)
            .collect::<Vec<_>>(),
        [20, 30, 50]
    );
    assert_eq!(
        result.reader_changes[1].kind,
        AuthorReaderChangeKind::RepeatedAssertion
    );
    assert_eq!(result.reader_summary.unknown, 1);
    assert_eq!(result.reader_summary.knowledge, 0);
    events.reverse();
    input.scenes.reverse();
    assert_eq!(
        build_author_mental_model(&input, &propositions, &events).unwrap(),
        result
    );
    assert_eq!(
        build_mental_model(
            &["Alice thought about the lost key."],
            &["Alice"],
            &["Alice"],
            &[]
        )
        .availability,
        AnalysisAvailability::Unavailable
    );
}
