#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StructureBeat {
    pub name: String,
    pub target_pct: f64,
    pub tolerance: f64,
    pub required: bool,
    pub description: String,
}

impl StructureBeat {
    pub fn new(
        name: String,
        target_pct: f64,
        tolerance: f64,
        required: bool,
        description: String,
    ) -> Self {
        Self {
            name,
            target_pct,
            tolerance,
            required,
            description,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StructureTemplate {
    pub name: String,
    pub description: String,
    pub beats: Vec<StructureBeat>,
    pub flexibility: f64,
}

impl StructureTemplate {
    pub fn new(
        name: String,
        description: String,
        beats: Vec<StructureBeat>,
        flexibility: f64,
    ) -> Self {
        Self {
            name,
            description,
            beats,
            flexibility,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BeatMatch {
    pub beat_name: String,
    pub expected_pct: f64,
    pub actual_pct: f64,
    pub matched: bool,
    pub confidence: f64,
}

#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StructureReport {
    pub template: String,
    pub health_pct: f64,
    pub matched_beats: Vec<String>,
    pub missing_beats: Vec<String>,
    pub beat_details: Vec<BeatMatch>,
    pub suggested_template: String,
}

fn beat(
    name: &str,
    target_pct: f64,
    tolerance: f64,
    required: bool,
    description: &str,
) -> StructureBeat {
    StructureBeat {
        name: name.into(),
        target_pct,
        tolerance,
        required,
        description: description.into(),
    }
}

fn three_act() -> StructureTemplate {
    StructureTemplate {
        name: "three_act".into(),
        description: "Classic three-act structure with setup, confrontation, and resolution".into(),
        flexibility: 0.7,
        beats: vec![
            beat(
                "Opening",
                0.0,
                0.05,
                true,
                "Establishes world and protagonist",
            ),
            beat(
                "Inciting Incident",
                0.12,
                0.05,
                true,
                "Event that disrupts the status quo",
            ),
            beat(
                "First Plot Point",
                0.25,
                0.08,
                true,
                "Protagonist commits to the journey",
            ),
            beat(
                "Midpoint",
                0.50,
                0.08,
                true,
                "Stakes raised; reversal or revelation",
            ),
            beat(
                "Second Plot Point",
                0.75,
                0.08,
                true,
                "Lowest point or final push",
            ),
            beat(
                "Climax",
                0.88,
                0.08,
                true,
                "Highest tension; decisive confrontation",
            ),
            beat("Resolution", 0.95, 0.05, false, "Aftermath and new normal"),
        ],
    }
}

fn heros_journey() -> StructureTemplate {
    StructureTemplate {
        name: "heros_journey".into(),
        description: "Campbell's monomyth with twelve stages of transformation".into(),
        flexibility: 0.6,
        beats: vec![
            beat("Ordinary World", 0.0, 0.05, true, "Ordinary World"),
            beat("Call to Adventure", 0.10, 0.05, true, "Call to Adventure"),
            beat(
                "Refusal of the Call",
                0.15,
                0.05,
                false,
                "Refusal of the Call",
            ),
            beat(
                "Meeting the Mentor",
                0.18,
                0.05,
                false,
                "Meeting the Mentor",
            ),
            beat(
                "Crossing the Threshold",
                0.25,
                0.05,
                true,
                "Crossing the Threshold",
            ),
            beat(
                "Tests, Allies, Enemies",
                0.40,
                0.10,
                true,
                "Tests, Allies, Enemies",
            ),
            beat(
                "Approach to Inmost Cave",
                0.50,
                0.08,
                true,
                "Approach to Inmost Cave",
            ),
            beat("Ordeal", 0.60, 0.08, true, "Ordeal"),
            beat("Reward", 0.70, 0.08, true, "Reward"),
            beat("The Road Back", 0.80, 0.08, true, "The Road Back"),
            beat("Resurrection", 0.90, 0.08, true, "Resurrection"),
            beat(
                "Return with Elixir",
                0.95,
                0.05,
                false,
                "Return with Elixir",
            ),
        ],
    }
}

fn save_the_cat() -> StructureTemplate {
    StructureTemplate {
        name: "save_the_cat".into(),
        description: "Blake Snyder's fifteen-beat screenplay structure".into(),
        flexibility: 0.7,
        beats: vec![
            beat("Opening Image", 0.0, 0.03, true, "Opening Image"),
            beat("Theme Stated", 0.05, 0.03, false, "Theme Stated"),
            beat("Setup", 0.10, 0.05, true, "Setup"),
            beat("Catalyst", 0.12, 0.03, true, "Catalyst"),
            beat("Debate", 0.18, 0.05, false, "Debate"),
            beat("Break into Two", 0.25, 0.05, true, "Break into Two"),
            beat("B Story", 0.30, 0.05, false, "B Story"),
            beat("Fun and Games", 0.38, 0.08, true, "Fun and Games"),
            beat("Midpoint", 0.50, 0.05, true, "Midpoint"),
            beat("Bad Guys Close In", 0.58, 0.08, true, "Bad Guys Close In"),
            beat("All Is Lost", 0.75, 0.05, true, "All Is Lost"),
            beat(
                "Dark Night of the Soul",
                0.80,
                0.05,
                false,
                "Dark Night of the Soul",
            ),
            beat("Break into Three", 0.83, 0.05, true, "Break into Three"),
            beat("Finale", 0.90, 0.08, true, "Finale"),
            beat("Final Image", 0.98, 0.03, false, "Final Image"),
        ],
    }
}

fn kishotenketsu() -> StructureTemplate {
    StructureTemplate {
        name: "kishotenketsu".into(),
        description: "Japanese four-act structure emphasizing twist over conflict".into(),
        flexibility: 0.8,
        beats: vec![
            beat("Ki (Introduction)", 0.0, 0.10, true, "Ki (Introduction)"),
            beat("Sho (Development)", 0.25, 0.10, true, "Sho (Development)"),
            beat(
                "Ten (Twist/Complication)",
                0.55,
                0.10,
                true,
                "Ten (Twist/Complication)",
            ),
            beat("Ketsu (Conclusion)", 0.85, 0.10, true, "Ketsu (Conclusion)"),
        ],
    }
}

fn nonlinear() -> StructureTemplate {
    StructureTemplate {
        name: "nonlinear".into(),
        description:
            "Freeform structure; only checks that tension arc has shape and reaches a climax".into(),
        flexibility: 1.0,
        beats: vec![],
    }
}

fn all_templates() -> Vec<StructureTemplate> {
    vec![
        three_act(),
        heros_journey(),
        save_the_cat(),
        kishotenketsu(),
        nonlinear(),
    ]
}

pub fn get_structure_template(name: &str) -> Result<StructureTemplate, crate::error::Error> {
    for t in all_templates() {
        if t.name == name {
            return Ok(t);
        }
    }
    Err(crate::error::Error::InvalidInput(format!(
        "Unknown template: '{}'. Use list_templates() for available names.",
        name
    )))
}

pub fn list_templates() -> Vec<String> {
    all_templates().iter().map(|t| t.name.clone()).collect()
}

/// Score how well a set of scene tensions matches a single beat.
/// Returns (best_scene_index, confidence) where confidence is 0.0-1.0.
fn match_beat(
    beat_def: &StructureBeat,
    scene_tensions: &[f64],
    scene_purposes: &[String],
) -> (usize, f64) {
    let n = scene_tensions.len();
    if n == 0 {
        return (0, 0.0);
    }

    let mut best_idx = 0;
    let mut best_conf = 0.0_f64;

    for i in 0..n {
        let scene_pct = i as f64 / (n.max(2) - 1) as f64;
        let pct_diff = (scene_pct - beat_def.target_pct).abs();
        let pct_score = (1.0 - pct_diff / (beat_def.tolerance * 2.0)).clamp(0.0, 1.0);

        // Purpose match: check if scene purpose contains beat name (case-insensitive)
        let purpose_score = if scene_purposes.get(i).is_some_and(|p| {
            let p_lower = p.to_lowercase();
            let beat_lower = beat_def.name.to_lowercase();
            p_lower.contains(&beat_lower)
                || beat_lower.split_whitespace().any(|w| p_lower.contains(w))
        }) {
            1.0
        } else {
            0.0
        };

        // Tension contribution: higher tension scenes match "climax"-like beats better
        let tension_score = scene_tensions[i].clamp(0.0, 1.0);

        // Weight: position matters most, then purpose, then tension
        let conf = pct_score * 0.5 + purpose_score * 0.3 + tension_score * 0.2;
        if conf > best_conf {
            best_conf = conf;
            best_idx = i;
        }
    }

    (best_idx, best_conf)
}

/// Score the nonlinear template: check that tensions have an arc shape with a clear peak.
fn score_nonlinear(scene_tensions: &[f64]) -> f64 {
    if scene_tensions.is_empty() {
        return 0.0;
    }
    let n = scene_tensions.len();
    if n == 1 {
        return 0.5;
    }

    // Find peak tension
    let max_tension = scene_tensions.iter().cloned().fold(0.0_f64, f64::max);
    if max_tension < 0.01 {
        return 0.1; // flat line, minimal credit
    }

    let peak_idx = scene_tensions
        .iter()
        .position(|&t| (t - max_tension).abs() < 1e-10)
        .unwrap_or(0);

    // Credit for having a peak somewhere (not at very start)
    let peak_pct = peak_idx as f64 / (n - 1) as f64;
    let has_shape = if peak_pct > 0.1 { 0.5 } else { 0.2 };

    // Credit for variance (not flat)
    let mean = scene_tensions.iter().sum::<f64>() / n as f64;
    let variance = scene_tensions
        .iter()
        .map(|t| (t - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let variance_score = (variance.sqrt() * 3.0).clamp(0.0, 0.5);

    has_shape + variance_score
}

fn score_template(
    template: &StructureTemplate,
    scene_tensions: &[f64],
    scene_purposes: &[String],
) -> (f64, Vec<BeatMatch>, Vec<String>, Vec<String>) {
    // Special case for nonlinear
    if template.beats.is_empty() {
        let health = score_nonlinear(scene_tensions);
        return (health, vec![], vec![], vec![]);
    }

    let mut beat_details = Vec::new();
    let mut matched_beats = Vec::new();
    let mut missing_beats = Vec::new();
    let mut total_weight = 0.0;
    let mut earned_weight = 0.0;
    let n = scene_tensions.len();

    let match_threshold = 0.3 * template.flexibility + 0.1; // more flexible = lower threshold

    for b in &template.beats {
        let weight = if b.required { 2.0 } else { 1.0 };
        total_weight += weight;

        let (best_idx, confidence) = match_beat(b, scene_tensions, scene_purposes);
        let actual_pct = if n > 1 {
            best_idx as f64 / (n - 1) as f64
        } else {
            0.0
        };

        let matched = confidence >= match_threshold;
        if matched {
            matched_beats.push(b.name.clone());
            earned_weight += weight * confidence;
        } else {
            missing_beats.push(b.name.clone());
        }

        beat_details.push(BeatMatch {
            beat_name: b.name.clone(),
            expected_pct: b.target_pct,
            actual_pct,
            matched,
            confidence,
        });
    }

    let health = if total_weight > 0.0 {
        (earned_weight / total_weight * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };

    (health, beat_details, matched_beats, missing_beats)
}

pub fn validate_structure(
    template_name: &str,
    scene_tensions: Vec<f64>,
    scene_purposes: Vec<String>,
) -> Result<StructureReport, crate::error::Error> {
    let template = all_templates()
        .into_iter()
        .find(|t| t.name == template_name)
        .ok_or_else(|| {
            crate::error::Error::InvalidInput(format!(
                "Unknown template: '{}'. Use list_templates() for available names.",
                template_name
            ))
        })?;

    let (health, beat_details, matched_beats, missing_beats) =
        score_template(&template, &scene_tensions, &scene_purposes);

    let suggested = suggest_best_impl(&scene_tensions, &scene_purposes);

    Ok(StructureReport {
        template: template_name.into(),
        health_pct: health,
        matched_beats,
        missing_beats,
        beat_details,
        suggested_template: suggested,
    })
}

fn suggest_best_impl(scene_tensions: &[f64], scene_purposes: &[String]) -> String {
    let mut best_name = String::from("nonlinear");
    let mut best_score = f64::NEG_INFINITY;

    for t in all_templates() {
        let (health, _, _, _) = score_template(&t, scene_tensions, scene_purposes);
        if health > best_score {
            best_score = health;
            best_name = t.name;
        }
    }

    best_name
}

pub fn suggest_best_template(scene_tensions: Vec<f64>, scene_purposes: Vec<String>) -> String {
    suggest_best_impl(&scene_tensions, &scene_purposes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_templates() {
        let names = list_templates();
        assert!(names.contains(&"three_act".into()));
        assert!(names.contains(&"heros_journey".into()));
        assert!(names.contains(&"save_the_cat".into()));
        assert!(names.contains(&"kishotenketsu".into()));
        assert!(names.contains(&"nonlinear".into()));
        assert_eq!(names.len(), 5);
    }

    #[test]
    fn test_get_template_valid() {
        let t = get_structure_template("three_act").unwrap();
        assert_eq!(t.name, "three_act");
        assert_eq!(t.beats.len(), 7);
        assert!((t.flexibility - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_get_template_invalid() {
        assert!(get_structure_template("doesnt_exist").is_err());
    }

    #[test]
    fn test_validate_three_act_aligned() {
        // Simulate a well-structured 20-scene manuscript with tension peaks at expected beats
        let n = 20;
        let mut tensions = vec![0.3; n];
        // Opening scene low tension
        tensions[0] = 0.2;
        // Inciting incident around scene 2 (10%)
        tensions[2] = 0.6;
        // First plot point around scene 5 (25%)
        tensions[5] = 0.7;
        // Midpoint around scene 10 (50%)
        tensions[10] = 0.8;
        // Second plot point around scene 15 (75%)
        tensions[15] = 0.7;
        // Climax around scene 17-18 (85-90%)
        tensions[17] = 0.95;
        // Resolution
        tensions[19] = 0.3;

        let purposes: Vec<String> = (0..n).map(|_| "scene".into()).collect();

        let report = validate_structure("three_act", tensions, purposes).unwrap();
        assert_eq!(report.template, "three_act");
        assert!(report.health_pct > 0.0, "health should be positive");
    }

    #[test]
    fn test_validate_nonlinear() {
        let tensions = vec![0.2, 0.4, 0.3, 0.8, 0.6, 0.3];
        let purposes: Vec<String> = (0..6).map(|_| "scene".into()).collect();

        let report = validate_structure("nonlinear", tensions, purposes).unwrap();
        assert_eq!(report.template, "nonlinear");
        assert!(
            report.health_pct > 0.0,
            "nonlinear with arc should score > 0"
        );
    }

    #[test]
    fn test_validate_nonlinear_flat() {
        let tensions = vec![0.0; 10];
        let purposes: Vec<String> = (0..10).map(|_| "scene".into()).collect();

        let report = validate_structure("nonlinear", tensions, purposes).unwrap();
        assert!(report.health_pct < 20.0, "flat tension should score low");
    }

    #[test]
    fn test_suggest_best_returns_valid_name() {
        let tensions = vec![0.2, 0.5, 0.3, 0.7, 0.9, 0.4];
        let purposes: Vec<String> = (0..6).map(|_| "scene".into()).collect();

        let suggested = suggest_best_template(tensions, purposes);
        let valid = list_templates();
        assert!(
            valid.contains(&suggested),
            "suggested '{}' not in templates",
            suggested
        );
    }

    #[test]
    fn test_kishotenketsu_beats() {
        let t = get_structure_template("kishotenketsu").unwrap();
        assert_eq!(t.beats.len(), 4);
        assert!((t.flexibility - 0.8).abs() < 1e-10);
        assert!(t.beats.iter().all(|b| b.required));
    }

    #[test]
    fn test_match_beat_empty_scenes() {
        let b = beat("Test", 0.5, 0.1, true, "test");
        let (idx, conf) = match_beat(&b, &[], &[]);
        assert_eq!(idx, 0);
        assert!((conf - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_beat_match_fields() {
        let tensions = vec![0.2, 0.5, 0.8, 0.4];
        let purposes: Vec<String> = vec![
            "setup".into(),
            "rising".into(),
            "climax".into(),
            "end".into(),
        ];

        let report = validate_structure("three_act", tensions, purposes).unwrap();
        for detail in &report.beat_details {
            assert!(detail.confidence >= 0.0 && detail.confidence <= 1.0);
            assert!(detail.expected_pct >= 0.0 && detail.expected_pct <= 1.0);
        }
    }
}
