use narrative_lens::manuscript::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Corpus {
    pub version: String,
    pub annotation_origin: String,
    pub cases: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Case {
    pub id: String,
    pub split: String,
    pub family: String,
    pub annotation_origin: String,
    pub manuscript: Manuscript,
    pub expected: Vec<Annotation>,
    pub exhaustive: bool,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Annotation {
    pub rule_id: String,
    pub scene_id: Option<String>,
    pub quote: Option<String>,
    pub range: Option<TextRange>,
    #[serde(default)]
    pub related: Vec<Related>,
    pub message_contains: Option<String>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Related {
    pub scene_id: String,
    pub quote: String,
}
#[derive(Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReliabilityBin {
    pub count: usize,
    pub mean_confidence: f64,
    pub observed_precision: f64,
}
#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Metrics {
    pub cases: usize,
    pub expected_findings: usize,
    pub predicted_findings: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub unrated_predictions: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub confidence_brier: Option<f64>,
    pub calibrated_brier: Option<f64>,
    pub reliability: Vec<ReliabilityBin>,
    pub calibrated_reliability: Vec<ReliabilityBin>,
    pub clean_cases: usize,
    pub clean_cases_flagged: usize,
}
#[derive(Default)]
struct Accumulator {
    metrics: Metrics,
    scores: Vec<(f64, bool)>,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Evaluation {
    pub corpus_version: String,
    pub annotation_origin: String,
    pub mode: String,
    pub metrics: BTreeMap<String, Metrics>,
    pub calibration: BTreeMap<String, [Option<f64>; 4]>,
}
fn bin(score: f64) -> usize {
    ((score * 4.0) as usize).min(3)
}
fn scope_scene(f: &Finding) -> Option<&str> {
    match &f.scope {
        Scope::Passage { scene_id } | Scope::Scene { scene_id } => Some(scene_id),
        Scope::Manuscript { .. } => None,
    }
}
fn matches(annotation: &Annotation, finding: &Finding, manuscript: &Manuscript) -> bool {
    if annotation.rule_id != finding.rule_id
        || annotation.scene_id.as_deref() != scope_scene(finding)
    {
        return false;
    }
    if annotation
        .message_contains
        .as_ref()
        .is_some_and(|s| !finding.message.to_lowercase().contains(&s.to_lowercase()))
    {
        return false;
    }
    if let Some(quote) = &annotation.quote
        && !finding.evidence.iter().any(|e| {
            e.role == EvidenceRole::Primary
                && &e.quote == quote
                && annotation.range.as_ref() == Some(&e.range)
        })
    {
        return false;
    }
    annotation.related.iter().all(|related| {
        let Some(scene) = manuscript.scenes.iter().find(|s| s.id == related.scene_id) else {
            return false;
        };
        scene
            .text
            .match_indices(&related.quote)
            .any(|(start, quote)| {
                let end = start + quote.len();
                let evidence = finding
                    .evidence
                    .iter()
                    .filter(|e| e.scene_id == related.scene_id && e.role != EvidenceRole::Primary)
                    .collect::<Vec<_>>();
                evidence.iter().any(|e| {
                    e.role == EvidenceRole::Conflict
                        && e.range.byte_start < end
                        && e.range.byte_end > start
                }) && quote
                    .char_indices()
                    .filter(|(_, ch)| !ch.is_whitespace())
                    .all(|(offset, ch)| {
                        evidence.iter().any(|e| {
                            e.range.byte_start <= start + offset
                                && e.range.byte_end >= start + offset + ch.len_utf8()
                        })
                    })
            })
    })
}
fn report(case: &Case, controls: bool) -> Report {
    let mut manuscript = case.manuscript.clone();
    if !controls {
        manuscript.configuration.dictionary.clear();
        manuscript.configuration.entity_corrections.clear();
        manuscript.configuration.suppressions.clear();
        for s in &mut manuscript.scenes {
            s.intentions.clear();
        }
    }
    let report = analyze_manuscript(&manuscript).unwrap();
    report.validate_against(&manuscript).unwrap();
    report
}
fn rated(case: &Case, report: &Report) -> (Vec<(usize, bool)>, Vec<bool>) {
    let mut expected = vec![false; case.expected.len()];
    let mut results = Vec::new();
    for (i, f) in report.findings.iter().enumerate() {
        if let Some(j) = case
            .expected
            .iter()
            .enumerate()
            .find_map(|(j, a)| (!expected[j] && matches(a, f, &case.manuscript)).then_some(j))
        {
            expected[j] = true;
            results.push((i, true));
        } else if case.exhaustive {
            results.push((i, false));
        }
    }
    (results, expected)
}
pub fn evaluate(corpus: &Corpus, controls: bool) -> Evaluation {
    let mut calibration_counts = BTreeMap::<String, [(usize, usize); 4]>::new();
    for case in corpus.cases.iter().filter(|c| c.split == "calibration") {
        let report = report(case, controls);
        let (results, _) = rated(case, &report);
        for (i, correct) in results {
            let f = &report.findings[i];
            let row = calibration_counts.entry(f.rule_id.clone()).or_default();
            row[bin(f.confidence.score)].0 += usize::from(correct);
            row[bin(f.confidence.score)].1 += 1;
        }
    }
    let calibration = calibration_counts
        .into_iter()
        .map(|(rule, counts)| {
            (
                rule,
                counts.map(|(positive, total)| {
                    (total > 0).then_some((positive + 1) as f64 / (total + 2) as f64)
                }),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut groups = BTreeMap::<String, Accumulator>::new();
    for case in &corpus.cases {
        assert!(
            !case.id.is_empty()
                && !case.family.is_empty()
                && case.annotation_origin.contains("agent-authored")
        );
        let report = report(case, controls);
        let (results, matched) = rated(case, &report);
        let by_index = results.iter().copied().collect::<BTreeMap<_, _>>();
        for rule in case.manuscript.configuration.rules.as_ref().unwrap() {
            let group = groups.entry(format!("{}/{rule}", case.split)).or_default();
            let metrics = &mut group.metrics;
            metrics.cases += 1;
            let annotated = case.expected.iter().filter(|a| &a.rule_id == rule).count();
            metrics.expected_findings += annotated;
            if annotated == 0 && case.exhaustive {
                metrics.clean_cases += 1;
                if report.findings.iter().any(|f| &f.rule_id == rule) {
                    metrics.clean_cases_flagged += 1;
                }
            }
        }
        for (j, annotation) in case.expected.iter().enumerate() {
            if !matched[j] {
                groups
                    .entry(format!("{}/{}", case.split, annotation.rule_id))
                    .or_default()
                    .metrics
                    .false_negatives += 1;
            }
        }
        for (i, finding) in report.findings.iter().enumerate() {
            let group = groups
                .entry(format!("{}/{}", case.split, finding.rule_id))
                .or_default();
            group.metrics.predicted_findings += 1;
            if let Some(correct) = by_index.get(&i) {
                if *correct {
                    group.metrics.true_positives += 1;
                } else {
                    group.metrics.false_positives += 1;
                }
                group.scores.push((finding.confidence.score, *correct));
            } else {
                group.metrics.unrated_predictions += 1;
            }
        }
    }
    let ratio = |a: usize, b: usize| (b > 0).then_some(a as f64 / b as f64);
    let metrics = groups
        .into_iter()
        .map(|(key, mut g)| {
            let m = &mut g.metrics;
            m.precision = ratio(m.true_positives, m.true_positives + m.false_positives);
            m.recall = ratio(m.true_positives, m.expected_findings);
            m.reliability = vec![ReliabilityBin::default(); 4];
            m.calibrated_reliability = vec![ReliabilityBin::default(); 4];
            let mut brier = 0.0;
            let mut calibrated = 0.0;
            let mut calibrated_count = 0;
            let rule = key.split_once('/').unwrap().1;
            for (score, correct) in &g.scores {
                let target = f64::from(*correct);
                brier += (score - target).powi(2);
                let b = &mut m.reliability[bin(*score)];
                b.count += 1;
                b.mean_confidence += score;
                b.observed_precision += target;
                if let Some(score) = calibration.get(rule).and_then(|row| row[bin(*score)]) {
                    calibrated += (score - target).powi(2);
                    calibrated_count += 1;
                    let b = &mut m.calibrated_reliability[bin(score)];
                    b.count += 1;
                    b.mean_confidence += score;
                    b.observed_precision += target;
                }
            }
            for b in m
                .reliability
                .iter_mut()
                .chain(&mut m.calibrated_reliability)
            {
                if b.count > 0 {
                    b.mean_confidence /= b.count as f64;
                    b.observed_precision /= b.count as f64;
                }
            }
            m.confidence_brier = (!g.scores.is_empty()).then_some(brier / g.scores.len() as f64);
            m.calibrated_brier =
                (calibrated_count > 0).then_some(calibrated / calibrated_count as f64);
            (key, g.metrics)
        })
        .collect();
    Evaluation {
        corpus_version: corpus.version.clone(),
        annotation_origin: corpus.annotation_origin.clone(),
        mode: if controls { "controlled" } else { "raw" }.into(),
        metrics,
        calibration,
    }
}
