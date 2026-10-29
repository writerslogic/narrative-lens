use narrative_lens::{
    continuity::world_state,
    craft::{grammar, readability},
    structure::{genre, structure},
};
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
    pub analyzer: String,
    pub genre: String,
    pub scenes: Vec<String>,
    pub characters: Vec<String>,
    pub purposes: Vec<String>,
    pub intentions: Vec<String>,
    #[serde(default)]
    pub dictionary: Vec<String>,
    pub expected: bool,
    pub rationale: String,
}
#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Metrics {
    pub cases: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub true_negatives: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub false_positive_rate: Option<f64>,
    pub brier_score: f64,
    pub reliability: Vec<Bin>,
}
#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Bin {
    pub count: usize,
    pub mean_confidence: f64,
    pub observed_positive_rate: f64,
}

pub fn legacy(case: &Case) -> f64 {
    let profile = genre::get_genre_profile(genre::genre_from_label(&case.genre).unwrap());
    let found = match case.analyzer.as_str() {
        "grammar" => case.scenes.iter().any(|s| !grammar::check(s).is_empty()),
        "readability" => case.scenes.iter().any(|s| {
            let r = readability::compute_readability(s);
            r.word_count >= 100 && r.fkgl > profile.target_fkgl_max
        }),
        "pacing" => case.scenes.iter().any(|s| {
            let r = readability::compute_pacing_metrics(s, 0);
            r.word_count >= 100 && r.velocity > profile.target_velocity_max
        }),
        "structure" => !structure::validate_structure(
            "three_act",
            vec![0.2; case.scenes.len()],
            case.purposes.clone(),
        )
        .unwrap()
        .missing_beats
        .is_empty(),
        "continuity" => !world_state::track_world_state(
            &case.scenes.iter().map(String::as_str).collect::<Vec<_>>(),
            &case
                .characters
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        )
        .violations
        .is_empty(),
        _ => panic!("unknown fixture analyzer"),
    };
    // A fixed engineering score makes legacy reliability reproducible; it is
    // not an editorial probability or a calibration of analyzer-native scores.
    if found { 0.8 } else { 0.0 }
}

pub fn evaluate(corpus: &Corpus, predict: impl Fn(&Case) -> f64) -> BTreeMap<String, Metrics> {
    let mut groups: BTreeMap<String, Vec<(bool, f64)>> = BTreeMap::new();
    for c in &corpus.cases {
        assert!(!c.id.is_empty() && !c.rationale.is_empty());
        assert!(matches!(c.split.as_str(), "calibration" | "held_out"));
        let score = predict(c);
        assert!(score.is_finite() && (0.0..=1.0).contains(&score));
        groups
            .entry(format!("{}/{}", c.split, c.analyzer))
            .or_default()
            .push((c.expected, score));
    }
    groups
        .into_iter()
        .map(|(name, samples)| {
            let mut m = Metrics {
                cases: samples.len(),
                reliability: (0..4).map(|_| Bin::default()).collect(),
                ..Metrics::default()
            };
            for (expected, score) in &samples {
                match (*expected, *score > 0.0) {
                    (true, true) => m.true_positives += 1,
                    (false, true) => m.false_positives += 1,
                    (true, false) => m.false_negatives += 1,
                    (false, false) => m.true_negatives += 1,
                }
                m.brier_score += (score - f64::from(*expected)).powi(2);
                let bin = &mut m.reliability[((*score * 4.0) as usize).min(3)];
                bin.count += 1;
                bin.mean_confidence += score;
                bin.observed_positive_rate += f64::from(*expected);
            }
            let ratio = |a: usize, b: usize| (b > 0).then(|| a as f64 / b as f64);
            m.precision = ratio(m.true_positives, m.true_positives + m.false_positives);
            m.recall = ratio(m.true_positives, m.true_positives + m.false_negatives);
            m.false_positive_rate = ratio(m.false_positives, m.false_positives + m.true_negatives);
            m.brier_score /= m.cases as f64;
            for b in &mut m.reliability {
                if b.count > 0 {
                    b.mean_confidence /= b.count as f64;
                    b.observed_positive_rate /= b.count as f64;
                }
            }
            (name, m)
        })
        .collect()
}

pub fn facade(case: &Case) -> f64 {
    use narrative_lens::manuscript::*;
    let scenes = case
        .scenes
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let mut s = Scene::new(format!("scene-{i}"), text);
            s.purpose = case.purposes.get(i).cloned().unwrap_or_default();
            s.intentions = case
                .intentions
                .iter()
                .map(|label| {
                    serde_json::from_value(serde_json::Value::String(label.clone())).unwrap()
                })
                .collect();
            s
        })
        .collect();
    let prefix = format!("NL-{}-", case.analyzer.to_uppercase());
    let rules = rule_definitions()
        .into_iter()
        .filter(|rule| rule.facade_available && rule.id.starts_with(&prefix))
        .map(|rule| rule.id.into())
        .collect();
    let manuscript = Manuscript {
        document_id: case.id.clone(),
        scenes,
        known_characters: case.characters.clone(),
        configuration: Configuration {
            genre: genre::genre_from_label(&case.genre).unwrap(),
            dictionary: case.dictionary.clone(),
            rules: Some(rules),
            ..Configuration::default()
        },
    };
    analyze_manuscript(&manuscript)
        .unwrap()
        .findings
        .iter()
        .map(|f| f.confidence.score)
        .fold(0.0, f64::max)
}

/// Laplace-smoothed emitted-case accuracy fitted using calibration labels only.
/// This calibrates the case-level "any finding" score, not individual findings.
pub fn fit_calibration(corpus: &Corpus) -> BTreeMap<String, f64> {
    let mut counts = BTreeMap::<String, (usize, usize)>::new();
    for c in corpus.cases.iter().filter(|c| c.split == "calibration") {
        if facade(c) > 0.0 {
            let e = counts.entry(c.analyzer.clone()).or_default();
            e.0 += usize::from(c.expected);
            e.1 += 1;
        }
    }
    counts
        .into_iter()
        .map(|(k, (positive, total))| (k, (positive + 1) as f64 / (total + 2) as f64))
        .collect()
}
