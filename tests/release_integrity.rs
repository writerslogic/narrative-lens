use narrative_lens::{
    continuity::{anachronism, entity::*, ner, world_state::*},
    manuscript::*,
    structure::foreshadow,
};

#[test]
fn public_text_analysis_handles_unicode_and_extreme_counts() {
    for whitespace in ['\u{0085}', '\u{00a0}', '\u{2003}', '\u{3000}'] {
        for end in [1493usize, 1499, 1500, 1501] {
            let text = format!(
                "{}{}{}",
                "a".repeat(end - whitespace.len_utf8()),
                whitespace,
                "b".repeat(20)
            );
            let _ = ner::ner_extract_entities(&text);
        }
    }
    for actual in 0..=3 {
        for declared in [0, 2, 3, 4, usize::MAX] {
            let found =
                foreshadow::analyze_foreshadowing(vec![vec!["key".into()]; actual], declared);
            if actual < 3 || declared < 3 || actual != declared {
                assert!(found.is_empty());
            }
        }
    }
    for year in [i32::MIN, i32::MIN + 1, i32::MAX - 1, i32::MAX] {
        let found = anachronism::analyze_scene(&["radio".into()], year, 0);
        if year > 2000 {
            assert!(found.is_empty());
        } else {
            assert!(found.iter().any(|f| f.word == "radio"));
        }
    }
}

fn correction(surface: &str, canonical: Option<&str>) -> EntityCorrection {
    EntityCorrection {
        surface: surface.into(),
        action: canonical.map_or(CorrectionAction::Reject, |canonical| {
            CorrectionAction::Assign {
                canonical: canonical.into(),
            }
        }),
    }
}

#[test]
fn correction_history_is_last_wins_for_present_and_absent_mentions() {
    for count in [0, 1, 2] {
        let mentions: Vec<_> = (0..count).map(|_| Mention::new("Ghost", vec![])).collect();
        let rejected = [
            correction("Ghost", Some("Alice")),
            correction("Ghost", None),
        ];
        assert!(resolve_with_corrections(&mentions, EntityType::Character, &rejected).is_empty());
        let reassigned = [
            correction("Ghost", Some("Alice")),
            correction("Ghost", Some("Bob")),
        ];
        let result = resolve_with_corrections(&mentions, EntityType::Character, &reassigned);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].canonical, "Bob");
        assert_eq!(result[0].mention_count, count.max(1));
        let mut duplicate_history = reassigned.to_vec();
        duplicate_history.extend(reassigned.clone());
        assert_eq!(
            result,
            resolve_with_corrections(&mentions, EntityType::Character, &duplicate_history)
        );
    }
}

#[test]
fn reassignment_preserves_observed_surface_counts() {
    let mentions: Vec<_> = std::iter::repeat_n("John Smith", 7)
        .chain(std::iter::repeat_n("John", 3))
        .map(|surface| Mention::new(surface, vec![]))
        .collect();
    let result = resolve_with_corrections(
        &mentions,
        EntityType::Character,
        &[correction("John", Some("Other John"))],
    );
    assert_eq!(
        result
            .iter()
            .find(|e| e.canonical == "John Smith")
            .unwrap()
            .mention_count,
        7
    );
    assert_eq!(
        result
            .iter()
            .find(|e| e.canonical == "Other John")
            .unwrap()
            .mention_count,
        3
    );
}

#[test]
fn ambiguous_surname_cannot_bridge_distinct_people_in_any_order() {
    let names = ["John Smith", "Smith", "Mary Smith"];
    let mut expected = None;
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let mentions = order.map(|i| Mention::new(names[i], vec![]));
        let result = resolve(&mentions, EntityType::Character);
        assert_eq!(result.len(), 3, "{result:?}");
        let counts = order.iter().map(|&i| (names[i].to_string(), 3)).collect();
        let merged = narrative_lens::substrate::nlp::merge_character_aliases(counts).unwrap();
        assert_eq!(
            merged,
            vec![
                ("John Smith".into(), 3),
                ("Mary Smith".into(), 3),
                ("Smith".into(), 3)
            ]
        );
        if let Some(expected) = &expected {
            assert_eq!(expected, &result);
        } else {
            expected = Some(result);
        }
    }
}

#[test]
fn world_events_preserve_roles_and_are_independent_of_cast_order() {
    let scenes = [
        "Bob discovered that the gate was open.",
        "Bob told Alice that the gate was open. Bob carried the key. Bob gave the key to Alice.",
    ];
    let a = track_world_state(&scenes, &["Alice", "Bob"]);
    let b = track_world_state(&scenes, &["Bob", "Alice"]);
    assert_eq!(
        a.information_flow,
        vec![(1, "bob".into(), "alice".into(), "the gate was open".into())]
    );
    assert_eq!(
        serde_json::to_value(&a).unwrap(),
        serde_json::to_value(&b).unwrap()
    );
    assert!(
        a.snapshots
            .last()
            .unwrap()
            .facts
            .contains(&WorldFact::CharacterPossesses {
                character: "alice".into(),
                object: "key".into()
            })
    );
    assert!(
        !a.snapshots
            .last()
            .unwrap()
            .facts
            .contains(&WorldFact::CharacterPossesses {
                character: "bob".into(),
                object: "key".into()
            })
    );
}

#[test]
fn explicit_acquisitions_are_evidence_of_learning() {
    for action in [
        "discovered that the key was gone",
        "read the letter",
        "overheard the plan",
        "realized that the gate was open",
        "witnessed the bridge collapse",
    ] {
        let scene = format!("Alice {action}.");
        let result = track_world_state(&["Rain fell.", &scene], &["Alice"]);
        assert!(!result.character_knowledge[0].knows.is_empty(), "{action}");
        assert!(
            !result
                .violations
                .iter()
                .any(|v| v.violation_type == ViolationType::ImpossibleKnowledge),
            "{action}"
        );
    }
}

#[test]
fn source_order_controls_live_location_and_colocation() {
    for (text, expected) in [
        (
            "Alice was in the cellar. Alice entered the garden.",
            "garden",
        ),
        (
            "Alice entered the garden. Alice was in the cellar.",
            "cellar",
        ),
    ] {
        let result = track_world_state(&[text], &["Alice"]);
        let locations: Vec<_> = result.snapshots[0]
            .facts
            .iter()
            .filter_map(|f| match f {
                WorldFact::CharacterAt { location, .. } => Some(location.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(locations, vec![expected]);
        assert_eq!(
            result.character_knowledge[0].current_location.as_deref(),
            Some(expected)
        );
        assert_eq!(result.snapshots[0].invalidated_facts.len(), 1);
    }
    let result = track_world_state(
        &[
            "Alice was in the cellar. Bob was in the cellar.",
            "Bob entered the garden.",
        ],
        &["Bob", "Alice"],
    );
    assert!(
        result.snapshots[0]
            .facts
            .iter()
            .any(|f| matches!(f, WorldFact::CoLocated { .. }))
    );
    assert!(
        !result.snapshots[1]
            .facts
            .iter()
            .any(|f| matches!(f, WorldFact::CoLocated { .. }))
    );
    assert!(
        result.snapshots[1]
            .invalidated_facts
            .iter()
            .any(|f| matches!(f, WorldFact::CoLocated { .. }))
    );
}

#[test]
fn transfers_remove_old_owners_within_and_between_scenes() {
    let result = track_world_state(
        &[
            "Bob carried the key. Bob gave the key to Alice.",
            "Alice handed the key to Clara.",
        ],
        &["Clara", "Alice", "Bob"],
    );
    for (snapshot, expected) in result.snapshots.iter().zip(["alice", "clara"]) {
        let owners: Vec<_> = snapshot
            .facts
            .iter()
            .filter_map(|f| match f {
                WorldFact::CharacterPossesses { character, object } if object == "key" => {
                    Some(character.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(owners, vec![expected]);
    }
}

#[test]
fn constraint_release_is_scoped_and_respects_event_order() {
    let locked = "Alice was in the cellar. Someone locked the door.";
    let blocked = track_world_state(&[locked, "Alice entered the garden."], &["Alice"]);
    assert!(
        blocked
            .violations
            .iter()
            .any(|v| v.violation_type == ViolationType::ImpossibleLocation)
    );
    for release in [
        "Someone unlocked the door.",
        "Alice was freed.",
        "Alice escaped.",
    ] {
        let result = track_world_state(&[locked, release, "Alice entered the garden."], &["Alice"]);
        assert!(
            result.violations.is_empty(),
            "{release}: {:?}",
            result.violations
        );
        assert!(
            !result
                .snapshots
                .last()
                .unwrap()
                .facts
                .iter()
                .any(|f| matches!(f, WorldFact::PhysicalConstraint { .. }))
        );
    }
    let result = track_world_state(
        &[
            "Alice was in the cellar. Someone unlocked the door.",
            "Alice entered the garden.",
        ],
        &["Alice"],
    );
    assert!(result.violations.is_empty());
    let reversed = track_world_state(
        &[
            locked,
            "Alice entered the garden. Someone unlocked the door.",
        ],
        &["Alice"],
    );
    assert_eq!(reversed.violations.len(), 1);
    let ambiguous = track_world_state(
        &["Alice was in the cellar. Bob was in the garden. Someone locked the door."],
        &["Alice", "Bob"],
    );
    assert!(
        !ambiguous.snapshots[0]
            .facts
            .iter()
            .any(|f| matches!(f, WorldFact::PhysicalConstraint { .. }))
    );
}

fn manuscript(text: &str, limit: usize, candidates: usize) -> Manuscript {
    let configuration = Configuration {
        rules: Some(["NL-GRAMMAR-REPETITION".into()].into()),
        limits: AnalysisLimits {
            max_findings: limit,
            scene_candidates: candidates,
            ..Default::default()
        },
        ..Default::default()
    };
    Manuscript {
        document_id: "limits".into(),
        scenes: vec![Scene::new("s", text)],
        known_characters: vec![],
        configuration,
    }
}

#[test]
fn persistent_results_cannot_hide_an_incomplete_comparison_population() {
    for limit in [1, 2, 3] {
        let before = analyze_manuscript(&manuscript("She saw saw the bird.", limit, 10)).unwrap();
        let after = analyze_manuscript(&manuscript(
            "She saw saw the bird. He ran ran through town.",
            limit,
            10,
        ))
        .unwrap();
        let comparison = try_compare_reports(&before, &after).unwrap();
        assert_eq!(comparison.persistent.len(), 1);
        assert_eq!(comparison.is_complete(&before, &after), limit >= 2);
        assert_eq!(comparison.complete, limit >= 2);
    }
    let before = analyze_manuscript(&manuscript("She saw saw the bird.", 10, 1)).unwrap();
    let after = analyze_manuscript(&manuscript(
        "She saw saw the bird. He ran ran through town.",
        10,
        1,
    ))
    .unwrap();
    let comparison = try_compare_reports(&before, &after).unwrap();
    assert_eq!(comparison.persistent.len(), 1);
    assert!(!comparison.is_complete(&before, &after));
    assert!(!comparison.complete);
    let mut forged = comparison;
    forged.complete = true;
    assert!(!forged.is_complete(&before, &after));
}

#[test]
fn foreshadowing_ties_have_stable_results() {
    let scenes = vec![
        vec![
            "compass".into(),
            "sword".into(),
            "compass".into(),
            "sword".into(),
        ],
        vec![],
        vec![],
        vec![],
        vec![],
        vec!["compass".into(), "sword".into()],
    ];
    let first = foreshadow::analyze_foreshadowing(scenes.clone(), 6);
    assert!(first.iter().any(|f| f.setup_term == "compass"));
    assert!(first.iter().any(|f| f.setup_term == "sword"));
    let serialized = serde_json::to_value(first).unwrap();
    for _ in 0..32 {
        assert_eq!(
            serialized,
            serde_json::to_value(foreshadow::analyze_foreshadowing(scenes.clone(), 6)).unwrap()
        );
    }
}

#[test]
fn older_comparisons_do_not_claim_complete_coverage() {
    let comparison: RevisionComparison =
        serde_json::from_str(r#"{"introduced":[],"persistent":[],"resolved":[],"unassessed":[]}"#)
            .unwrap();
    assert!(!comparison.complete);
}

fn nli_manuscript() -> Manuscript {
    let mut input = manuscript("Alice stayed in the cellar.", 10, 10);
    input
        .scenes
        .push(Scene::new("other", "Alice left the cellar."));
    input.configuration.rules = Some(["NL-CONTINUITY-NLI".into()].into());
    input.configuration.nli = ModelPolicy::Fallback;
    input
}

#[test]
fn selected_analysis_without_output_cannot_prove_completeness() {
    let mut input = nli_manuscript();
    let unavailable = analyze_manuscript(&input).unwrap();
    assert!(unavailable.findings.is_empty());
    let nli = unavailable
        .coverage
        .iter()
        .find(|c| c.analyzer == "nli")
        .unwrap();
    assert_eq!(nli.status, Status::Unavailable);
    assert_eq!(nli.total_units, 1);
    assert!(!nli.is_intentionally_disabled());
    for status in [
        Status::Unavailable,
        Status::ExecutionFailure,
        Status::InsufficientInput,
    ] {
        let mut report = unavailable.clone();
        let nli = report
            .coverage
            .iter_mut()
            .find(|c| c.analyzer == "nli")
            .unwrap();
        nli.status = status;
        let comparison = try_compare_reports(&report, &report).unwrap();
        assert!(comparison.unassessed.is_empty());
        assert!(!comparison.complete, "{status:?}");
    }
    input.configuration.nli = ModelPolicy::Required;
    assert!(analyze_manuscript(&input).is_err());
    input.configuration.nli = ModelPolicy::Fallback;
    input.configuration.limits.model_input_bytes = 1;
    let oversized = analyze_manuscript(&input).unwrap();
    assert_eq!(
        oversized
            .coverage
            .iter()
            .find(|c| c.analyzer == "nli")
            .unwrap()
            .status,
        Status::InsufficientInput
    );
    assert!(
        !try_compare_reports(&oversized, &oversized)
            .unwrap()
            .complete
    );
}

#[test]
fn selected_analysis_requires_unique_coverage_and_configuration() {
    let input = manuscript("The bird flew home.", 10, 10);
    let report = analyze_manuscript(&input).unwrap();
    assert!(try_compare_reports(&report, &report).unwrap().complete);
    let mut missing = report.clone();
    missing.coverage.retain(|c| c.analyzer != "grammar");
    missing.validate().unwrap();
    assert!(!try_compare_reports(&missing, &missing).unwrap().complete);
    let mut duplicate = report.clone();
    duplicate.coverage.push(
        report
            .coverage
            .iter()
            .find(|c| c.analyzer == "grammar")
            .unwrap()
            .clone(),
    );
    duplicate.validate().unwrap();
    assert!(
        !try_compare_reports(&duplicate, &duplicate)
            .unwrap()
            .complete
    );
    let mut unknown_selection = report;
    unknown_selection.configuration = None;
    unknown_selection.validate().unwrap();
    assert!(
        !try_compare_reports(&unknown_selection, &unknown_selection)
            .unwrap()
            .complete
    );

    let before = analyze_manuscript(&manuscript("She saw saw the bird.", 10, 10)).unwrap();
    let comparison = try_compare_reports(&before, &missing).unwrap();
    assert!(comparison.resolved.is_empty());
    assert_eq!(comparison.unassessed[0].reason, "coverage_missing");
}

#[test]
fn deterministic_fallback_and_no_eligible_units_remain_comparable() {
    let report = analyze_manuscript(&manuscript("The bird flew home.", 10, 10)).unwrap();
    let mut fallback = report.clone();
    let grammar = fallback
        .coverage
        .iter_mut()
        .find(|c| c.analyzer == "grammar")
        .unwrap();
    grammar.status = Status::HeuristicFallback;
    grammar.diagnostics = vec!["deterministic_local_heuristic".into()];
    assert!(try_compare_reports(&fallback, &fallback).unwrap().complete);

    let mut nli = nli_manuscript();
    nli.scenes.truncate(1);
    let no_pairs = analyze_manuscript(&nli).unwrap();
    assert!(try_compare_reports(&no_pairs, &no_pairs).unwrap().complete);

    let mut default_input = manuscript("The bird flew home.", 10, 10);
    default_input.configuration = Configuration::default();
    let short = analyze_manuscript(&default_input).unwrap();
    assert!(try_compare_reports(&short, &short).unwrap().complete);

    let mut sampling = manuscript("The bird flew home.", 10, 10);
    sampling
        .scenes
        .push(Scene::new("adequate", "The bird flew home. ".repeat(30)));
    sampling.configuration.rules =
        Some(["NL-READABILITY-COMPLEX".into(), "NL-PACING-LONG".into()].into());
    let sampled = analyze_manuscript(&sampling).unwrap();
    assert!(
        sampled
            .coverage
            .iter()
            .any(|c| c.status == Status::HeuristicFallback)
    );
    assert!(try_compare_reports(&sampled, &sampled).unwrap().complete);

    let before = analyze_manuscript(&manuscript("She saw saw the bird.", 10, 10)).unwrap();
    let mut removed = manuscript("", 10, 10);
    removed.scenes.clear();
    let after = analyze_manuscript(&removed).unwrap();
    let comparison = try_compare_reports(&before, &after).unwrap();
    assert_eq!(comparison.resolved.len(), 1);
    assert!(comparison.complete);
}

#[test]
fn disabled_analysis_and_author_intent_exemptions_are_explicit() {
    let mut input = manuscript("The bird flew home.", 10, 10);
    input.configuration.rules =
        Some(["NL-CONTINUITY-NLI".into(), "NL-PROSE-LEXICON".into()].into());
    let disabled = analyze_manuscript(&input).unwrap();
    assert!(try_compare_reports(&disabled, &disabled).unwrap().complete);
    for reason in [
        "rule_not_selected",
        "rules_not_selected",
        "lexicon_empty",
        "analysis_disabled",
        "model_analysis_disabled_or_rule_not_selected",
        "author_intends_nonlinear_structure",
    ] {
        let coverage = Coverage {
            analyzer: "structure".into(),
            status: Status::Unavailable,
            total_units: 1,
            analyzed_units: 0,
            unit: "scenes".into(),
            diagnostics: vec![reason.into()],
        };
        assert!(coverage.is_intentionally_disabled(), "{reason}");
    }
    let unavailable = Coverage {
        analyzer: "nli".into(),
        status: Status::Unavailable,
        total_units: 1,
        analyzed_units: 0,
        unit: "candidate_pairs".into(),
        diagnostics: vec!["onnx_feature_disabled".into()],
    };
    assert!(!unavailable.is_intentionally_disabled());
    let mut nonlinear = manuscript("The bird flew home.", 10, 10);
    nonlinear.configuration.rules = Some(["NL-STRUCTURE-MISSING-BEAT".into()].into());
    nonlinear.scenes[0].intentions = vec![Intention::Nonlinear];
    let intended = analyze_manuscript(&nonlinear).unwrap();
    assert!(try_compare_reports(&intended, &intended).unwrap().complete);
}

#[test]
fn legacy_report_population_without_summary_stays_unknown() {
    for text in ["She saw saw the bird.", "The bird flew home."] {
        let current = analyze_manuscript(&manuscript(text, 10, 10)).unwrap();
        let mut old = serde_json::to_value(&current).unwrap();
        old["schemaVersion"] = "1".into();
        old.as_object_mut().unwrap().remove("summary");
        let legacy: Report = serde_json::from_value(old).unwrap();
        legacy.validate().unwrap();
        assert_eq!(legacy.summary, ReportSummary::default());
        for (before, after) in [(&legacy, &current), (&current, &legacy), (&legacy, &legacy)] {
            let comparison = try_compare_reports(before, after).unwrap();
            assert_eq!(comparison.persistent.len(), current.findings.len());
            assert!(comparison.introduced.is_empty() && comparison.resolved.is_empty());
            assert!(!comparison.complete);
            assert!(!comparison.is_complete(before, after));
        }
        let mut invalid = current.clone();
        invalid.summary.returned_findings += 1;
        assert!(!compare_reports(&invalid, &current).complete);
    }
}
