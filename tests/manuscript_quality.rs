use narrative_lens::manuscript::*;
fn book(texts: &[&str], rules: &[&str]) -> Manuscript {
    Manuscript {
        document_id: "quality".into(),
        scenes: texts
            .iter()
            .enumerate()
            .map(|(i, t)| Scene::new(format!("scene-{i}"), *t))
            .collect(),
        known_characters: vec!["Zoë".into()],
        configuration: Configuration {
            rules: Some(rules.iter().map(|r| (*r).into()).collect()),
            ..Configuration::default()
        },
    }
}
#[test]
fn unselected_findings_cannot_starve_selected_rules_or_coverage() {
    let noise = "go ".repeat(4100);
    let mut m = book(
        &[&noise, &noise, &noise, "A suddenly opened gate."],
        &["NL-PROSE-LEXICON"],
    );
    m.configuration.lexicon = vec!["suddenly".into()];
    let r = analyze_manuscript(&m).unwrap();
    assert_eq!(r.findings.len(), 1);
    assert_eq!(r.findings[0].evidence[0].scene_id, "scene-3");
    assert!(
        !r.diagnostics
            .iter()
            .any(|d| d.code == "scene_candidate_limit")
    );
}
#[test]
fn inactive_grammar_cannot_hide_cross_scene_contradictions() {
    let noise = "go ".repeat(4100);
    let m = book(
        &[
            &noise,
            &noise,
            &noise,
            "Zoë had blue eyes.",
            "Zoë had brown eyes.",
        ],
        &["NL-CONTINUITY-ATTRIBUTE"],
    );
    assert_eq!(analyze_manuscript(&m).unwrap().findings.len(), 1);
}
#[test]
fn no_selected_rules_produces_no_partial_analysis_diagnostic() {
    let noise = "go ".repeat(4100);
    let r = analyze_manuscript(&book(&[&noise], &[])).unwrap();
    assert!(r.findings.is_empty());
    assert!(r.diagnostics.is_empty());
    assert!(r.coverage.iter().all(|c| c.status == Status::Unavailable));
}
#[test]
fn exact_candidate_limit_is_not_misreported_as_truncation() {
    let text = "go ".repeat(4097);
    let r = analyze_manuscript(&book(&[&text], &["NL-GRAMMAR-REPETITION"])).unwrap();
    assert_eq!(r.findings.len(), 4096);
    assert!(
        !r.diagnostics
            .iter()
            .any(|d| d.code == "scene_candidate_limit")
    );
}
#[test]
fn revision_matching_distinguishes_same_word_in_different_passages() {
    let before = analyze_manuscript(&book(
        &["Mara saw saw the bell. Zoë saw saw the sea."],
        &["NL-GRAMMAR-REPETITION"],
    ))
    .unwrap();
    let after =
        analyze_manuscript(&book(&["Zoë saw saw the sea."], &["NL-GRAMMAR-REPETITION"])).unwrap();
    let d = compare_reports(&before, &after);
    assert_eq!(d.persistent.len(), 1);
    assert_eq!(d.resolved.len(), 1);
    assert_eq!(
        d.persistent[0].before.evidence[0].range,
        before.findings[1].evidence[0].range
    );
    assert_eq!(
        d.resolved[0].evidence[0].range,
        before.findings[0].evidence[0].range
    );
}
#[test]
fn tied_chronology_does_not_disable_unambiguous_later_segments() {
    let mut m = book(
        &[
            "The bell rang.",
            "The tide turned.",
            "Zoë had blue eyes.",
            "Zoë had brown eyes.",
        ],
        &["NL-CONTINUITY-ATTRIBUTE"],
    );
    m.configuration.chronology = true;
    for (s, key) in m.scenes.iter_mut().zip(["01", "01", "02", "03"]) {
        s.chronology = Some(Chronology {
            key: key.into(),
            lane: "main".into(),
        });
    }
    let r = analyze_manuscript(&m).unwrap();
    assert_eq!(r.findings.len(), 1);
    assert_eq!(r.chronology.ambiguous_scene_ids.len(), 2);
}
#[test]
fn quoted_dialogue_with_a_unicode_prefix_is_not_a_world_assertion() {
    let m = book(
        &["Zoë had blue eyes.", "😀 “Zoë had brown eyes.”"],
        &["NL-CONTINUITY-ATTRIBUTE"],
    );
    assert!(analyze_manuscript(&m).unwrap().findings.is_empty());
}
#[test]
fn changing_thresholds_and_model_identity_preserves_unaffected_scene_work() {
    let mut m = book(
        &["Zoë saw the gate.", "Zoë saw a bird."],
        &["NL-GRAMMAR-REPETITION"],
    );
    let mut c = AnalysisContext::default();
    c.analyze(&m).unwrap();
    m.configuration.thresholds.max_grade = Some(18.0);
    assert_eq!(c.analyze(&m).unwrap(), analyze_manuscript(&m).unwrap());
    assert_eq!(c.stats().recomputed_scenes, 0);
    m.configuration.model_revision = "new-model".into();
    assert_eq!(c.analyze(&m).unwrap(), analyze_manuscript(&m).unwrap());
    assert_eq!(c.stats().recomputed_scenes, 0);
}
#[test]
fn suppression_precedes_output_limit_and_preserves_total_counts() {
    let mut m = book(
        &["Zoë saw saw a very small bird."],
        &["NL-GRAMMAR-REPETITION", "NL-PROSE-LEXICON"],
    );
    m.configuration.lexicon = vec!["very".into()];
    m.configuration.limits.max_findings = 1;
    m.configuration.suppressions.push(Suppression {
        rule_id: Some("NL-GRAMMAR-REPETITION".into()),
        scene_id: None,
        fingerprint: None,
        reason: "Deliberate repetition".into(),
    });
    let r = analyze_manuscript(&m).unwrap();
    assert_eq!(r.findings.len(), 1);
    assert_eq!(r.findings[0].rule_id, "NL-PROSE-LEXICON");
    assert_eq!(r.summary.total_findings, 2);
    assert_eq!(r.summary.suppressed_findings, 1);
    assert_eq!(r.summary.omitted_findings, 0);
    r.validate_against(&m).unwrap();
}
#[test]
fn output_limit_counts_omitted_severities_and_never_fabricates_resolution() {
    let mut m = book(
        &["Zoë saw saw a bird. Zoë went went home. Zoë came came back."],
        &["NL-GRAMMAR-REPETITION"],
    );
    let before = analyze_manuscript(&m).unwrap();
    m.configuration.limits.max_findings = 1;
    let after = analyze_manuscript(&m).unwrap();
    assert_eq!(after.summary.severity_counts[&Severity::Warning], 3);
    assert_eq!(after.summary.omitted_findings, 2);
    let diff = try_compare_reports(&before, &after).unwrap();
    assert!(diff.resolved.is_empty());
    assert_eq!(diff.unassessed.len(), 2);
}
#[test]
fn removing_a_rule_is_not_reported_as_repairing_its_findings() {
    let mut m = book(&["Zoë saw saw a bird."], &["NL-GRAMMAR-REPETITION"]);
    let before = analyze_manuscript(&m).unwrap();
    m.configuration.rules = Some(Default::default());
    let after = analyze_manuscript(&m).unwrap();
    let diff = try_compare_reports(&before, &after).unwrap();
    assert!(diff.resolved.is_empty());
    assert_eq!(diff.unassessed[0].reason, "rule_not_evaluated");
}
#[test]
fn unplaced_scene_only_blocks_its_declared_lane() {
    let mut m = book(
        &[
            "The tide turned.",
            "Zoë had blue eyes.",
            "Zoë had brown eyes.",
        ],
        &["NL-CONTINUITY-ATTRIBUTE"],
    );
    m.configuration.chronology = true;
    m.scenes[0].chronology_lane = Some("dream".into());
    for (i, s) in m.scenes.iter_mut().enumerate().skip(1) {
        s.chronology = Some(Chronology {
            key: i.to_string(),
            lane: "main".into(),
        });
    }
    let r = analyze_manuscript(&m).unwrap();
    assert_eq!(r.findings.len(), 1);
    assert_eq!(r.chronology.ambiguous_scene_ids, vec!["scene-0"]);
}
#[test]
fn imported_reports_reject_confidence_and_coordinate_corruption() {
    let m = book(&["😀 Zoë saw saw a bird."], &["NL-GRAMMAR-REPETITION"]);
    let r = analyze_manuscript(&m).unwrap();
    let mut corrupt = r.clone();
    corrupt.findings[0].confidence.score = f64::NAN;
    assert!(corrupt.validate().is_err());
    let mut corrupt = r.clone();
    corrupt.findings[0].evidence[0].range.utf16_start += 1;
    assert!(corrupt.validate().is_err());
    let mut corrupt = r.clone();
    corrupt.findings[0].evidence[0].range.byte_start += 1;
    corrupt.findings[0].evidence[0].range.byte_end += 1;
    assert!(corrupt.validate_against(&m).is_err());
    let mut corrupt = r.clone();
    corrupt.findings[0].evidence[0].document_id = "foreign".into();
    assert!(try_compare_reports(&r, &corrupt).is_err());
}
#[test]
fn dictionaries_filter_before_candidate_limits_and_only_invalidate_grammar() {
    let mut m = book(&["A wyrmlet saw a qwick bird."], &["NL-GRAMMAR-SPELLING"]);
    m.configuration.limits.scene_candidates = 1;
    m.configuration.dictionary = vec!["wyrmlet".into()];
    let mut c = AnalysisContext::default();
    let r = c.analyze(&m).unwrap();
    assert_eq!(r.findings.len(), 1);
    assert_eq!(r.findings[0].evidence[0].quote, "qwick");
    m.configuration.dictionary.push("qwick".into());
    assert!(c.analyze(&m).unwrap().findings.is_empty());
    assert_eq!(c.stats().recomputed_metrics_scenes, 0);
    assert_eq!(c.stats().recomputed_grammar_scenes, 1);
}
#[test]
fn unrelated_text_between_a_location_and_its_constraint_does_not_change_identity() {
    let mut m = book(
        &[
            "Zoë was in the cellar. Rain fell on the roof. Someone locked the door.",
            "Zoë entered the garden.",
        ],
        &["NL-CONTINUITY-LOCKED"],
    );
    let before = analyze_manuscript(&m).unwrap();
    assert_eq!(before.findings[0].evidence.len(), 3);
    m.scenes[0].text="Zoë was in the cellar. Snow lay on the empty road beyond the house. Someone locked the door.".into();
    let after = analyze_manuscript(&m).unwrap();
    let diff = compare_reports(&before, &after);
    assert_eq!(diff.persistent.len(), 1);
    assert!(diff.introduced.is_empty());
    assert!(diff.resolved.is_empty());
}
#[test]
fn ambiguous_constraints_and_equivalent_color_spellings_do_not_create_errors() {
    let mut m = book(
        &["Zoë had grey eyes.", "Zoë had gray eyes."],
        &["NL-CONTINUITY-ATTRIBUTE"],
    );
    assert!(analyze_manuscript(&m).unwrap().findings.is_empty());
    m = book(
        &[
            "Zoë was in the cellar. Mara was in the garden. Someone locked the door.",
            "Zoë entered the kitchen.",
        ],
        &["NL-CONTINUITY-LOCKED"],
    );
    m.known_characters.push("Mara".into());
    let report = analyze_manuscript(&m).unwrap();
    assert!(report.findings.is_empty());
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code == "ambiguous_constraint")
    );
}

#[test]
fn duplicate_lexicon_entries_do_not_consume_the_candidate_budget() {
    let mut m = book(&["A suddenly suddenly opened gate."], &["NL-PROSE-LEXICON"]);
    m.configuration.lexicon = vec!["suddenly".into(), "SUDDENLY".into()];
    m.configuration.limits.scene_candidates = 2;
    let report = analyze_manuscript(&m).unwrap();
    assert_eq!(report.findings.len(), 2);
    assert!(
        !report
            .diagnostics
            .iter()
            .any(|d| d.code == "lexicon_candidate_limit")
    );
}

#[test]
fn imported_reader_evidence_and_configuration_are_validated() {
    let mut m = book(
        &["Zoë learned that the gate was open."],
        &["NL-CONTINUITY-ATTRIBUTE"],
    );
    m.configuration.chronology = true;
    m.scenes[0].chronology = Some(Chronology {
        key: "01".into(),
        lane: "main".into(),
    });
    let report = analyze_manuscript(&m).unwrap();
    let mut corrupt = report.clone();
    corrupt.chronology.reader_knowledge[0].introduced[0]
        .range
        .byte_start += 1;
    assert!(corrupt.validate().is_err());
    let mut corrupt = report.clone();
    corrupt.chronology.reader_knowledge[0].introduced[0].scene_id = "foreign".into();
    assert!(corrupt.validate().is_err());
    let mut corrupt = report.clone();
    corrupt
        .chronology
        .lanes
        .get_mut("main")
        .unwrap()
        .push("foreign".into());
    assert!(corrupt.validate().is_err());
    let mut corrupt = report;
    corrupt.configuration.as_mut().unwrap().thresholds.max_grade = Some(f64::NAN);
    assert!(corrupt.validate().is_err());
}

#[test]
fn standalone_adapters_have_identity_and_batch_assignment_preserves_duplicates() {
    let text = "😀 Zoë saw saw the bird. Zoë saw saw the sea.";
    let mut findings = narrative_lens::craft::grammar::check(text)
        .iter()
        .filter(|f| f.kind == narrative_lens::craft::grammar::FindingKind::Repetition)
        .map(|f| finding::adapt_grammar("quality", "opening", text, f).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(findings.len(), 2);
    assert!(findings.iter().all(|f| f.fingerprint.starts_with("nl-v2-")));
    assign_fingerprints("quality", &mut findings).unwrap();
    assert_ne!(findings[0].fingerprint, findings[1].fingerprint);
    findings[1].document_id = "foreign".into();
    let unchanged = findings.clone();
    assert!(assign_fingerprints("quality", &mut findings).is_err());
    assert_eq!(unchanged, findings);
    let mut c = Configuration::default();
    for value in [0, 1, 2, 99_999, 100_000, 100_001] {
        c.limits.max_findings = value;
        assert_eq!(c.validate().is_ok(), (1..=100_000).contains(&value));
    }
    c = Configuration::default();
    for value in [0, 1, 2, 65_535, 65_536, 65_537] {
        c.limits.scene_candidates = value;
        assert_eq!(c.validate().is_ok(), (1..=65_536).contains(&value));
    }
}
