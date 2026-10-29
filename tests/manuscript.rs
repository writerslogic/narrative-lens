use narrative_lens::{
    manuscript::*,
    onnx::{
        InferenceBackend,
        checked::{ModelFailure, ModelKind},
    },
};
use std::collections::BTreeSet;
fn manuscript(texts: &[&str], rules: &[&str]) -> Manuscript {
    Manuscript {
        document_id: "book".into(),
        scenes: texts
            .iter()
            .enumerate()
            .map(|(i, t)| Scene::new(format!("s{i}"), *t))
            .collect(),
        known_characters: vec!["Zoë".into()],
        configuration: Configuration {
            rules: Some(rules.iter().map(|s| (*s).into()).collect()),
            ..Configuration::default()
        },
    }
}
#[test]
fn unicode_ranges_preserve_original_text_and_reject_surrogate_splits() {
    let text = "😀 e\u{301} 雪 Zoë saw saw the gate.";
    let m = manuscript(&[text], &["NL-GRAMMAR-REPETITION"]);
    let r = analyze_manuscript(&m).unwrap();
    let e = &r.findings[0].evidence[0];
    assert_eq!(&text[e.range.byte_start..e.range.byte_end], e.quote);
    let utf16: Vec<_> = text.encode_utf16().collect();
    assert_eq!(
        String::from_utf16(&utf16[e.range.utf16_start as usize..e.range.utf16_end as usize])
            .unwrap(),
        "saw"
    );
    assert!(TextRange::from_utf16(text, 1, 2).is_err());
    for offset in [text.len() - 1, text.len(), text.len() + 1] {
        assert_eq!(
            TextRange::new(text, 0, offset).is_ok(),
            offset <= text.len() && text.is_char_boundary(offset)
        );
    }
}
#[test]
fn contradictions_have_both_original_passages_and_release_is_respected() {
    let m = manuscript(
        &[
            "😀 Zoë was in the cellar. Someone locked the door.",
            "Zoë entered the garden.",
        ],
        &["NL-CONTINUITY-LOCKED"],
    );
    let r = analyze_manuscript(&m).unwrap();
    assert_eq!(r.findings.len(), 1);
    let f = &r.findings[0];
    assert_eq!(f.evidence.len(), 3);
    assert_eq!(f.evidence[0].scene_id, "s1");
    assert!(f.evidence[1].quote.contains("Zoë was in the cellar."));
    assert_eq!(f.evidence[2].quote, "Someone locked the door.");
    let mut released = m.clone();
    released.scenes[1].text = "Zoë escaped from the cellar. Zoë entered the garden.".into();
    assert!(analyze_manuscript(&released).unwrap().findings.is_empty());
}
#[test]
fn eye_color_and_object_conflicts_are_evidenced_and_intentions_clear_state() {
    let mut m = manuscript(
        &[
            "Zoë had blue eyes. The key was on the desk.",
            "Zoë had brown eyes. The key was on the shelf.",
        ],
        &["NL-CONTINUITY-ATTRIBUTE", "NL-CONTINUITY-OBJECT"],
    );
    assert_eq!(analyze_manuscript(&m).unwrap().findings.len(), 2);
    m.scenes[0].intentions.push(Intention::Supernatural);
    assert!(analyze_manuscript(&m).unwrap().findings.is_empty());
}
#[test]
fn controls_affect_findings_and_roundtrip() {
    let mut m = manuscript(
        &["Zoë saw saw a qwick bird. Very cold."],
        &[
            "NL-GRAMMAR-REPETITION",
            "NL-GRAMMAR-SPELLING",
            "NL-PROSE-LEXICON",
        ],
    );
    m.configuration.dictionary.push("qwick".into());
    m.configuration.lexicon.push("very".into());
    m.scenes[0].intentions.push(Intention::Repetition);
    let r = analyze_manuscript(&m).unwrap();
    assert_eq!(r.findings.len(), 1);
    assert_eq!(r.findings[0].evidence[0].quote, "Very");
    m.configuration.suppressions.push(Suppression {
        rule_id: None,
        scene_id: Some("s0".into()),
        fingerprint: Some(r.findings[0].fingerprint.clone()),
        reason: "Intentional emphasis".into(),
    });
    let r = analyze_manuscript(&m).unwrap();
    assert!(r.findings.is_empty());
    assert_eq!(r.suppressed.len(), 1);
    let json = serde_json::to_string(&m).unwrap();
    assert_eq!(serde_json::from_str::<Manuscript>(&json).unwrap(), m);
    m.configuration.rules = Some(BTreeSet::new());
    assert!(analyze_manuscript(&m).unwrap().findings.is_empty());
    m.configuration.rules = Some(["bogus".into()].into());
    assert!(m.validate().is_err());
}
#[test]
fn configuration_boundaries_are_validated_and_used() {
    for n in [0, 1, 2, 99_999, 100_000, 100_001] {
        let mut c = Configuration::default();
        c.thresholds.min_scene_words = Some(n);
        assert_eq!(c.validate().is_ok(), (1..=100_000).contains(&n));
    }
    for grade in [-0.1, 0.0, 0.1, 29.9, 30.0, 30.1, f64::NAN] {
        let mut c = Configuration::default();
        c.thresholds.max_grade = Some(grade);
        assert_eq!(
            c.validate().is_ok(),
            grade.is_finite() && (0.0..=30.0).contains(&grade)
        );
    }
    let mut m = manuscript(
        &[
            "The institutionalization of interplanetary communication necessitated constitutional reconsideration.",
        ],
        &["NL-READABILITY-COMPLEX"],
    );
    m.configuration.thresholds.min_scene_words = Some(1);
    m.configuration.thresholds.max_grade = Some(0.0);
    assert_eq!(analyze_manuscript(&m).unwrap().findings.len(), 1);
    m.scenes[0].intentions.push(Intention::ComplexProse);
    assert!(analyze_manuscript(&m).unwrap().findings.is_empty());
    for length in [127, 128, 129] {
        m.configuration.dictionary = vec!["x".repeat(length)];
        assert_eq!(m.validate().is_ok(), length <= 128);
    }
}
#[test]
fn chronology_is_opt_in_and_presentation_order_and_reader_knowledge_are_preserved() {
    let mut m = manuscript(
        &[
            "Zoë entered the garden. Zoë learned that the key was lost.",
            "Zoë was in the cellar. Someone locked the door.",
        ],
        &["NL-CONTINUITY-LOCKED"],
    );
    m.configuration.chronology = true;
    m.scenes[0].chronology = Some(Chronology {
        key: "02".into(),
        lane: "main".into(),
    });
    m.scenes[1].chronology = Some(Chronology {
        key: "01".into(),
        lane: "main".into(),
    });
    let r = analyze_manuscript(&m).unwrap();
    assert_eq!(r.scene_ids, vec!["s0", "s1"]);
    assert_eq!(r.chronology.lanes["main"], vec!["s1", "s0"]);
    assert_eq!(r.findings.len(), 1);
    assert_eq!(
        r.chronology.reader_knowledge[0].introduced[0].scene_id,
        "s0"
    );
    m.scenes[0].chronology.as_mut().unwrap().lane = "parallel".into();
    assert!(analyze_manuscript(&m).unwrap().findings.is_empty());
    m.scenes[0].chronology.as_mut().unwrap().lane = "main".into();
    m.scenes[0].chronology.as_mut().unwrap().key = "01".into();
    let r = analyze_manuscript(&m).unwrap();
    assert!(r.findings.is_empty());
    assert_eq!(r.chronology.ambiguous_scene_ids.len(), 2);
    m.scenes[0].chronology = None;
    assert_eq!(
        analyze_manuscript(&m).unwrap().diagnostics[0].code,
        "chronology_ambiguous"
    );
}
#[test]
fn incremental_matches_fresh_for_all_mutations_and_cross_scene_dependencies() {
    let mut m = manuscript(
        &[
            "Zoë was in the cellar. Someone locked the door.",
            "Zoë entered the garden.",
            "Zoë saw saw a bird.",
        ],
        &["NL-GRAMMAR-REPETITION", "NL-CONTINUITY-LOCKED"],
    );
    let mut context = AnalysisContext::new(1024 * 1024).unwrap();
    let check = |c: &mut AnalysisContext, m: &Manuscript| {
        assert_eq!(c.analyze(m).unwrap(), analyze_manuscript(m).unwrap());
        assert!(c.stats().retained_bytes <= c.stats().budget_bytes);
    };
    check(&mut context, &m);
    check(&mut context, &m);
    assert_eq!(context.stats().reused_scenes, 3);
    m.scenes[0].text = "Zoë was in the cellar.".into();
    check(&mut context, &m);
    assert_eq!(context.stats().recomputed_scenes, 1);
    assert!(
        context
            .analyze(&m)
            .unwrap()
            .findings
            .iter()
            .all(|f| f.rule_id != "NL-CONTINUITY-LOCKED")
    );
    m.scenes
        .insert(1, Scene::new("inserted", "Zoë had green eyes."));
    check(&mut context, &m);
    m.scenes.remove(0);
    check(&mut context, &m);
    m.scenes.swap(0, 2);
    check(&mut context, &m);
    assert_eq!(context.stats().recomputed_scenes, 0);
    m.scenes[0].intentions.push(Intention::Repetition);
    check(&mut context, &m);
    m.configuration.chronology = true;
    check(&mut context, &m);
    m.configuration.model_revision = "replacement".into();
    check(&mut context, &m);
    assert_eq!(context.stats().recomputed_scenes, 0);
    m.known_characters.push("Mara".into());
    check(&mut context, &m);
    m.document_id = "new-document".into();
    check(&mut context, &m);
    for budget in [0, 1, 1024, 4096] {
        let mut c = AnalysisContext::new(budget).unwrap();
        check(&mut c, &m);
    }
}
struct TestBackend {
    failure: Option<ModelFailure>,
    output: Vec<f64>,
}
impl InferenceBackend for TestBackend {
    fn embed(&self, texts: &[&str]) -> Vec<Vec<f64>> {
        vec![vec![]; texts.len()]
    }
    fn tag(&self, _: &str) -> Vec<(String, String, usize, usize)> {
        vec![]
    }
    fn check_model(&self, _: ModelKind) -> Result<(), ModelFailure> {
        self.failure.clone().map_or(Ok(()), Err)
    }
    fn classify_pairs(&self, pairs: &[(&str, &str)]) -> Vec<Vec<f64>> {
        vec![self.output.clone(); pairs.len()]
    }
}
#[test]
fn model_missing_incompatible_and_execution_failures_are_distinct_and_strict() {
    let mut m = manuscript(
        &["Zoë had blue eyes.", "Zoë had brown eyes."],
        &["NL-CONTINUITY-NLI"],
    );
    m.configuration.nli = ModelPolicy::Fallback;
    for (failure, output, status, code) in [
        (
            Some(ModelFailure::unavailable("missing_local_fixture")),
            vec![],
            Status::Unavailable,
            "missing_local_fixture",
        ),
        (
            None,
            vec![0.5, 0.5],
            Status::ExecutionFailure,
            "nli_expected_three_classes",
        ),
        (
            None,
            vec![],
            Status::ExecutionFailure,
            "nli_empty_inference",
        ),
        (
            Some(ModelFailure::execution("test_backend_failed")),
            vec![],
            Status::ExecutionFailure,
            "test_backend_failed",
        ),
    ] {
        let backend = TestBackend { failure, output };
        let r = analyze_with_backend(&m, &backend).unwrap();
        let c = r.coverage.iter().find(|c| c.analyzer == "nli").unwrap();
        assert_eq!(c.status, status);
        assert_eq!(c.diagnostics[0], code);
        m.configuration.nli = ModelPolicy::Required;
        assert!(analyze_with_backend(&m, &backend).is_err());
        m.configuration.nli = ModelPolicy::Fallback;
    }
    let mut context = AnalysisContext::default();
    let first = TestBackend {
        failure: None,
        output: vec![0.95, 0.03, 0.02],
    };
    assert_eq!(
        context
            .analyze_with_backend(&m, &first)
            .unwrap()
            .findings
            .len(),
        1
    );
    let replacement = TestBackend {
        failure: None,
        output: vec![0.02, 0.03, 0.95],
    };
    assert!(
        context
            .analyze_with_backend(&m, &replacement)
            .unwrap()
            .findings
            .is_empty()
    );
    assert_eq!(context.stats().reused_scenes, 2);
}
#[test]
fn revisions_tolerate_offset_shifts_reordering_and_preserve_duplicate_multiplicity() {
    let m = manuscript(
        &[
            "Zoë saw saw a bird. Zoë saw saw a bird.",
            "Zoë saw saw a bird.",
        ],
        &["NL-GRAMMAR-REPETITION"],
    );
    let before = analyze_manuscript(&m).unwrap();
    assert_eq!(before.findings.len(), 3);
    assert_eq!(
        before
            .findings
            .iter()
            .map(|f| &f.fingerprint)
            .collect::<BTreeSet<_>>()
            .len(),
        3
    );
    let mut changed = m.clone();
    changed.scenes[0].text.insert_str(0, "😀 Snow fell. ");
    changed.scenes.swap(0, 1);
    let after = analyze_manuscript(&changed).unwrap();
    let diff = compare_reports(&before, &after);
    assert_eq!(diff.persistent.len(), 3);
    assert!(diff.introduced.is_empty());
    assert!(diff.resolved.is_empty());
    changed
        .scenes
        .iter_mut()
        .find(|s| s.id == "s0")
        .unwrap()
        .text = "Zoë saw saw a bird.".into();
    let diff = compare_reports(&before, &analyze_manuscript(&changed).unwrap());
    assert_eq!(diff.persistent.len(), 2);
    assert_eq!(diff.resolved.len(), 1);
    changed
        .scenes
        .push(Scene::new("new", "Zoë saw saw a bird."));
    assert_eq!(
        compare_reports(&before, &analyze_manuscript(&changed).unwrap())
            .introduced
            .len(),
        1
    );
}
#[test]
fn explicit_object_transfers_unlocks_and_negations_do_not_create_false_conflicts() {
    for bridge in ["Zoë carried the key to the garden.", "Zoë moved the key."] {
        let m = manuscript(
            &[
                "The key was on the desk.",
                bridge,
                "The key was on the shelf.",
            ],
            &["NL-CONTINUITY-OBJECT"],
        );
        assert!(analyze_manuscript(&m).unwrap().findings.is_empty());
    }
    for lock in [
        "Someone did not lock the door.",
        "Someone had not locked the door.",
        "Someone unlocked the door.",
    ] {
        let text = format!("Zoë was in the cellar. {lock}");
        let m = manuscript(
            &[&text, "Zoë entered the garden."],
            &["NL-CONTINUITY-LOCKED"],
        );
        assert!(analyze_manuscript(&m).unwrap().findings.is_empty());
    }
    let m = manuscript(
        &[
            "Zoë was in the cellar. Someone locked the door.",
            "Someone unlocked the door. Zoë entered the garden.",
        ],
        &["NL-CONTINUITY-LOCKED"],
    );
    assert!(analyze_manuscript(&m).unwrap().findings.is_empty());
}
#[test]
fn bounded_candidates_are_reported_as_partial_instead_of_clean() {
    let text = "go ".repeat(4100);
    let m = manuscript(&[&text], &["NL-GRAMMAR-REPETITION"]);
    let r = analyze_manuscript(&m).unwrap();
    assert_eq!(r.findings.len(), 4096);
    assert!(
        r.diagnostics
            .iter()
            .any(|d| d.code == "grammar_candidate_limit")
    );
    assert_eq!(
        r.coverage
            .iter()
            .find(|c| c.analyzer == "grammar")
            .unwrap()
            .status,
        Status::HeuristicFallback
    );
}
#[test]
fn malformed_model_distributions_and_partial_coverage_fail_strict_requirements() {
    let mut m = manuscript(
        &["Zoë had blue eyes.", "Zoë had brown eyes."],
        &["NL-CONTINUITY-NLI"],
    );
    m.configuration.nli = ModelPolicy::Required;
    for output in [
        vec![f64::NAN, 0.0, 1.0],
        vec![1.1, -0.1, 0.0],
        vec![0.5, 0.5, 0.5],
    ] {
        let b = TestBackend {
            failure: None,
            output,
        };
        assert!(analyze_with_backend(&m, &b).is_err());
    }
    m.scenes[0].text = "Snow fell. ".repeat(9);
    let b = TestBackend {
        failure: None,
        output: vec![0.01, 0.98, 0.01],
    };
    assert!(analyze_with_backend(&m, &b).is_err());
    m.configuration.nli = ModelPolicy::Fallback;
    let r = analyze_with_backend(&m, &b).unwrap();
    let c = r.coverage.iter().find(|c| c.analyzer == "nli").unwrap();
    assert_eq!(c.total_units, 9);
    assert_eq!(c.analyzed_units, 8);
    assert_eq!(c.status, Status::HeuristicFallback);
}
#[test]
fn cached_metadata_changes_recompute_chronology_and_structure_dependencies() {
    let mut m = manuscript(
        &[
            "Zoë was in the cellar. Someone locked the door.",
            "Zoë entered the garden.",
            "Zoë saw a bird.",
        ],
        &["NL-CONTINUITY-LOCKED", "NL-STRUCTURE-MISSING-BEAT"],
    );
    m.configuration.chronology = true;
    for (i, s) in m.scenes.iter_mut().enumerate() {
        s.chronology = Some(Chronology {
            key: format!("{i:02}"),
            lane: "main".into(),
        });
    }
    let mut c = AnalysisContext::default();
    assert_eq!(c.analyze(&m).unwrap(), analyze_manuscript(&m).unwrap());
    m.scenes[1].chronology.as_mut().unwrap().lane = "parallel".into();
    assert_eq!(c.analyze(&m).unwrap(), analyze_manuscript(&m).unwrap());
    assert_eq!(c.stats().reused_scenes, 3);
    m.scenes[0].purpose = "setup inciting incident".into();
    m.scenes[1].purpose = "confrontation midpoint".into();
    m.scenes[2].purpose = "climax resolution".into();
    assert_eq!(c.analyze(&m).unwrap(), analyze_manuscript(&m).unwrap());
    assert_eq!(c.stats().reused_scenes, 3);
    m.scenes[0].chronology.as_mut().unwrap().key = "99".into();
    assert_eq!(c.analyze(&m).unwrap(), analyze_manuscript(&m).unwrap());
}
#[test]
fn unicode_case_folding_and_regex_characters_preserve_subject_identity() {
    for (canonical, surface) in [("Σ", "ς"), ("A+B", "A+B")] {
        let mut m = manuscript(
            &[
                &format!("{surface} had blue eyes."),
                &format!("{surface} had brown eyes."),
            ],
            &["NL-CONTINUITY-ATTRIBUTE"],
        );
        m.known_characters = vec![canonical.into()];
        let r = analyze_manuscript(&m).unwrap();
        assert_eq!(r.findings.len(), 1);
        assert!(r.findings[0].evidence[0].quote.starts_with(surface));
    }
}
