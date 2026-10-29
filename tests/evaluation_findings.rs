#[path = "../examples/support/finding_evaluation.rs"]
mod evaluation;

#[test]
fn individual_rule_evidence_and_false_positives_do_not_regress() {
    let corpus: evaluation::Corpus =
        serde_json::from_str(include_str!("fixtures/evaluation-v2.json")).unwrap();
    assert_eq!(corpus.version, "finding-level-fiction-v2.1");
    assert!(corpus.annotation_origin.contains("agent-authored"));
    let baseline: serde_json::Value =
        serde_json::from_str(include_str!("../docs/evaluation-v2-baseline.json")).unwrap();
    let controlled = evaluation::evaluate(&corpus, true);
    let raw = evaluation::evaluate(&corpus, false);
    for (key, metric) in &controlled.metrics {
        let prior = &baseline["controlled"]["metrics"][key];
        assert!(
            metric.true_positives >= prior["truePositives"].as_u64().unwrap() as usize,
            "{key}: seeded findings lost"
        );
        assert!(
            metric.false_positives <= prior["falsePositives"].as_u64().unwrap() as usize,
            "{key}: false positives increased"
        );
        assert_eq!(
            metric.false_negatives, 0,
            "{key}: expected evidence not found"
        );
        assert_eq!(
            metric.false_positives, 0,
            "{key}: clean passage or intention flagged"
        );
        if let Some(brier) = metric.confidence_brier {
            assert!(
                brier <= prior["confidenceBrier"].as_f64().unwrap() + 0.005,
                "{key}: confidence reliability regressed"
            );
        }
    }
    for split in ["calibration", "held_out"] {
        for rule in [
            "NL-GRAMMAR-REPETITION",
            "NL-GRAMMAR-SPELLING",
            "NL-READABILITY-COMPLEX",
            "NL-PACING-LONG",
            "NL-STRUCTURE-MISSING-BEAT",
            "NL-CONTINUITY-LOCKED",
        ] {
            let key = format!("{split}/{rule}");
            assert!(
                raw.metrics[&key].false_positives > controlled.metrics[&key].false_positives,
                "{key}: author controls had no measured effect"
            );
        }
    }
    let calibration = serde_json::to_value(&controlled.calibration).unwrap();
    let mut changed = corpus;
    for case in &mut changed.cases {
        if case.split == "held_out" {
            case.expected.clear();
        }
    }
    assert_eq!(
        calibration,
        serde_json::to_value(evaluation::evaluate(&changed, true).calibration).unwrap(),
        "held-out labels leaked into calibration"
    );
}
