#[path = "../examples/support/evaluation.rs"]
mod evaluation;
#[test]
fn held_out_accuracy_does_not_regress_against_frozen_legacy_baseline() {
    let corpus: evaluation::Corpus =
        serde_json::from_str(include_str!("fixtures/evaluation.json")).unwrap();
    assert_eq!(corpus.version, "synthetic-fiction-v1");
    assert!(corpus.annotation_origin.contains("agent-authored"));
    let baseline: serde_json::Value =
        serde_json::from_str(include_str!("../docs/evaluation-baseline.json")).unwrap();
    let measured = evaluation::evaluate(&corpus, evaluation::facade);
    for (key, m) in measured {
        let b = &baseline["metrics"][&key];
        assert!(
            m.true_positives >= b["truePositives"].as_u64().unwrap() as usize,
            "{key}: recall regressed"
        );
        assert!(
            m.false_positives <= b["falsePositives"].as_u64().unwrap() as usize,
            "{key}: false positives regressed"
        );
        assert_eq!(m.false_negatives, 0, "{key}: lost a supported seeded error");
        assert_eq!(
            m.false_positives, 0,
            "{key}: flagged a clean passage or author exception"
        );
        assert!(
            m.brier_score <= 0.06,
            "{key}: case confidence reliability regressed"
        );
    }
    // Mutating held-out outcomes cannot affect fitted calibration.
    let fitted = evaluation::fit_calibration(&corpus);
    let mut changed = corpus;
    for c in &mut changed.cases {
        if c.split == "held_out" {
            c.expected = !c.expected;
        }
    }
    assert_eq!(fitted, evaluation::fit_calibration(&changed));
    assert_eq!(evaluation::legacy(&changed.cases[0]), 0.8);
}
