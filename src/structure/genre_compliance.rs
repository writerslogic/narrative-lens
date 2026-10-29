//! Caller-supplied genre convention checklists; descriptive structure inputs
//! never establish prose quality, thematic truth, social intent or genre blends.
use crate::substrate::utils::AnalysisAvailability;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenreConvention {
    pub name: String,
    pub description: String,
    pub status: ConventionStatus,
    pub evidence: String,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConventionStatus {
    Unknown,
    Met,
    PartiallyMet,
    NotMet,
    Subverted,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenreComplianceResult {
    #[serde(default)]
    pub availability: crate::substrate::utils::AnalysisAvailability,
    #[serde(default)]
    pub diagnostics: Vec<String>,

    #[serde(default)]
    pub quality_assessed: bool,
    #[serde(default)]
    pub judgment_provenance: String,
    #[serde(default)]
    pub blend_assessed: bool,
    #[serde(default)]
    pub structural_signals: GenreStructuralSignals,
    #[serde(default)]
    pub assessed_count: usize,
    #[serde(default)]
    pub unknown_count: usize,
    /// Fraction of checklist entries with supplied judgments, not a quality score.
    #[serde(default)]
    pub evidence_coverage: f64,
    pub genre: String,
    pub conventions: Vec<GenreConvention>,
    /// Number of conventions observed as present (Met or Subverted).
    pub present_count: usize,
    /// Number of conventions observed as absent (NotMet).
    pub absent_count: usize,
    pub is_genre_blend: bool,
    pub secondary_genres: Vec<String>,
    pub deviations: Vec<String>,
    pub strengths: Vec<String>,
}

struct ConventionDef {
    name: &'static str,
    description: &'static str,
}

fn romance_conventions() -> Vec<ConventionDef> {
    vec![
        ConventionDef {
            name: "HEA/HFN",
            description: "Happily ever after or happy for now ending",
        },
        ConventionDef {
            name: "Central Love Story",
            description: "Romance is the main plot, not subplot",
        },
        ConventionDef {
            name: "Emotional Satisfaction",
            description: "Reader feels emotionally fulfilled by resolution",
        },
        ConventionDef {
            name: "Character Growth",
            description: "Both leads grow through the relationship",
        },
        ConventionDef {
            name: "Conflict Between Leads",
            description: "Meaningful obstacles between the couple",
        },
        ConventionDef {
            name: "Dual POV or Deep POV",
            description: "Intimate access to character interiority",
        },
        ConventionDef {
            name: "Meet-Cute or Inciting Event",
            description: "Clear moment that brings leads together",
        },
    ]
}

fn mystery_conventions() -> Vec<ConventionDef> {
    vec![
        ConventionDef {
            name: "Fair Play Clues",
            description: "Reader has access to clues needed to solve the mystery",
        },
        ConventionDef {
            name: "Solution Revealed",
            description: "Mystery is solved by the end",
        },
        ConventionDef {
            name: "Red Herrings",
            description: "False leads that misdirect without cheating",
        },
        ConventionDef {
            name: "Rising Suspicion",
            description: "Tension builds as suspects are eliminated",
        },
        ConventionDef {
            name: "Protagonist Agency",
            description: "Detective/protagonist drives the investigation",
        },
        ConventionDef {
            name: "Stakes Escalation",
            description: "Consequences increase as story progresses",
        },
        ConventionDef {
            name: "Satisfying Resolution",
            description: "Solution feels earned and logical",
        },
    ]
}

fn thriller_conventions() -> Vec<ConventionDef> {
    vec![
        ConventionDef {
            name: "Ticking Clock",
            description: "Time pressure drives urgency",
        },
        ConventionDef {
            name: "High Stakes",
            description: "Consequences of failure are severe",
        },
        ConventionDef {
            name: "Protagonist in Danger",
            description: "Main character faces mortal or existential threat",
        },
        ConventionDef {
            name: "Fast Pacing",
            description: "Scenes move quickly with minimal downtime",
        },
        ConventionDef {
            name: "Plot Twists",
            description: "Unexpected reversals that recontextualize events",
        },
        ConventionDef {
            name: "Antagonist Presence",
            description: "Clear, active opposition driving conflict",
        },
        ConventionDef {
            name: "Climactic Confrontation",
            description: "Direct showdown between protagonist and antagonist",
        },
        ConventionDef {
            name: "Protagonist Survives",
            description: "Hero survives (usually)",
        },
    ]
}

fn literary_conventions() -> Vec<ConventionDef> {
    vec![
        ConventionDef {
            name: "Prose Quality",
            description: "Language is crafted with attention to style and rhythm",
        },
        ConventionDef {
            name: "Character Interiority",
            description: "Deep psychological exploration of characters",
        },
        ConventionDef {
            name: "Thematic Depth",
            description: "Explores meaningful themes beyond plot",
        },
        ConventionDef {
            name: "Ambiguity Tolerance",
            description: "Comfortable with unresolved questions",
        },
        ConventionDef {
            name: "Voice Distinctiveness",
            description: "Unique authorial or narrative voice",
        },
        ConventionDef {
            name: "Subtext-Rich Dialogue",
            description: "Characters communicate beneath surface meaning",
        },
        ConventionDef {
            name: "Resonant Ending",
            description: "Ending illuminates theme rather than merely resolving plot",
        },
    ]
}

fn fantasy_conventions() -> Vec<ConventionDef> {
    vec![
        ConventionDef {
            name: "Worldbuilding",
            description: "Coherent secondary world or magic system",
        },
        ConventionDef {
            name: "Magic System Rules",
            description: "Magic has consistent internal logic",
        },
        ConventionDef {
            name: "Quest or Journey",
            description: "Characters pursue a goal through the world",
        },
        ConventionDef {
            name: "Good vs Evil",
            description: "Clear moral conflict (or nuanced subversion)",
        },
        ConventionDef {
            name: "Character Growth",
            description: "Protagonist transforms through trials",
        },
        ConventionDef {
            name: "Sense of Wonder",
            description: "Moments of awe at the fantastic",
        },
        ConventionDef {
            name: "Consistent Rules",
            description: "World operates by established internal logic",
        },
    ]
}

fn scifi_conventions() -> Vec<ConventionDef> {
    vec![
        ConventionDef {
            name: "Speculative Premise",
            description: "Central 'what if' question drives narrative",
        },
        ConventionDef {
            name: "Internal Consistency",
            description: "Technology and world follow consistent rules",
        },
        ConventionDef {
            name: "Thematic Exploration",
            description: "Premise used to explore human condition",
        },
        ConventionDef {
            name: "Worldbuilding",
            description: "Setting feels plausible and realized",
        },
        ConventionDef {
            name: "Stakes Beyond Personal",
            description: "Implications extend beyond individual characters",
        },
        ConventionDef {
            name: "Scientific Plausibility",
            description: "Technology grounded in extrapolation (soft or hard)",
        },
    ]
}

fn horror_conventions() -> Vec<ConventionDef> {
    vec![
        ConventionDef {
            name: "Dread Building",
            description: "Atmosphere creates sustained unease",
        },
        ConventionDef {
            name: "Threat Escalation",
            description: "Danger increases throughout",
        },
        ConventionDef {
            name: "Vulnerability",
            description: "Characters are genuinely at risk",
        },
        ConventionDef {
            name: "The Unknown",
            description: "Mystery or incomprehensibility of the threat",
        },
        ConventionDef {
            name: "Isolation",
            description: "Characters cut off from help",
        },
        ConventionDef {
            name: "Body/Mind Horror",
            description: "Visceral or psychological violation",
        },
        ConventionDef {
            name: "Thematic Subtext",
            description: "Horror as metaphor for real fears",
        },
    ]
}

fn historical_conventions() -> Vec<ConventionDef> {
    vec![
        ConventionDef {
            name: "Period Accuracy",
            description: "Setting details are historically plausible",
        },
        ConventionDef {
            name: "Authentic Voice",
            description: "Language and attitudes fit the era",
        },
        ConventionDef {
            name: "Historical Context",
            description: "Real events shape character lives",
        },
        ConventionDef {
            name: "Sensory Immersion",
            description: "Reader experiences the period through senses",
        },
        ConventionDef {
            name: "Character Agency",
            description: "Characters act within period constraints",
        },
        ConventionDef {
            name: "Thematic Relevance",
            description: "Historical setting illuminates universal themes",
        },
    ]
}

fn ya_conventions() -> Vec<ConventionDef> {
    vec![
        ConventionDef {
            name: "Teen Protagonist",
            description: "Main character is 14-18 years old",
        },
        ConventionDef {
            name: "Coming of Age",
            description: "Character faces identity-defining challenges",
        },
        ConventionDef {
            name: "Voice Authenticity",
            description: "Narrative voice sounds genuinely young",
        },
        ConventionDef {
            name: "Fast Pacing",
            description: "Story moves quickly with short chapters",
        },
        ConventionDef {
            name: "High Stakes (Personal)",
            description: "Consequences feel world-ending to the protagonist",
        },
        ConventionDef {
            name: "Hope",
            description: "Even dark stories offer hope or agency",
        },
        ConventionDef {
            name: "First Experience",
            description: "Character encounters things for the first time",
        },
    ]
}

fn womens_fiction_conventions() -> Vec<ConventionDef> {
    vec![
        ConventionDef {
            name: "Female Protagonist",
            description: "Story centers a woman's journey",
        },
        ConventionDef {
            name: "Personal Transformation",
            description: "Protagonist undergoes meaningful internal change",
        },
        ConventionDef {
            name: "Relationships Central",
            description: "Family, friendship, or romantic relationships drive plot",
        },
        ConventionDef {
            name: "Emotional Depth",
            description: "Rich interiority and emotional truth",
        },
        ConventionDef {
            name: "Life Transition",
            description: "Story triggered by major life change",
        },
        ConventionDef {
            name: "Satisfying Resolution",
            description: "Protagonist finds peace or new direction",
        },
    ]
}

fn upmarket_conventions() -> Vec<ConventionDef> {
    vec![
        ConventionDef {
            name: "Literary Prose",
            description: "Writing quality above commercial average",
        },
        ConventionDef {
            name: "Compelling Plot",
            description: "Strong narrative drive despite literary quality",
        },
        ConventionDef {
            name: "Thematic Substance",
            description: "Explores meaningful ideas",
        },
        ConventionDef {
            name: "Character Depth",
            description: "Complex, fully realized characters",
        },
        ConventionDef {
            name: "Accessible Voice",
            description: "Literary but not alienating",
        },
        ConventionDef {
            name: "Emotional Resonance",
            description: "Leaves lasting emotional impact",
        },
    ]
}

fn middle_grade_conventions() -> Vec<ConventionDef> {
    vec![
        ConventionDef {
            name: "Age-Appropriate Protagonist",
            description: "Main character is 8-12 years old",
        },
        ConventionDef {
            name: "Adventure/Discovery",
            description: "Story involves exploration or quest",
        },
        ConventionDef {
            name: "Humor",
            description: "Moments of genuine fun and levity",
        },
        ConventionDef {
            name: "Clear Moral Framework",
            description: "Right and wrong are distinguishable",
        },
        ConventionDef {
            name: "Friendship Themes",
            description: "Peer relationships are central",
        },
        ConventionDef {
            name: "Adult Absence",
            description: "Kids solve problems without adult intervention",
        },
        ConventionDef {
            name: "Hopeful Resolution",
            description: "Ending is optimistic",
        },
    ]
}

fn canonical_genre(genre: &str) -> Result<&'static str> {
    if genre.len() > 128 {
        return Err(Error::InvalidInput("unsupported genre label".into()));
    }
    Ok(match genre.trim().to_lowercase().as_str() {
        "romance" => "romance",
        "mystery" => "mystery",
        "thriller" => "thriller",
        "literary" | "literary fiction" => "literary",
        "fantasy" => "fantasy",
        "sci-fi" | "science fiction" => "sci-fi",
        "horror" => "horror",
        "historical" | "historical fiction" => "historical",
        "ya" | "young adult" => "ya",
        "women's fiction" | "womens fiction" => "women's fiction",
        "upmarket" | "upmarket fiction" => "upmarket",
        "middle-grade" | "middle grade" => "middle-grade",
        _ => return Err(Error::InvalidInput("unsupported genre label".into())),
    })
}

fn get_conventions(genre: &str) -> Vec<ConventionDef> {
    match genre {
        "romance" => romance_conventions(),
        "mystery" => mystery_conventions(),
        "thriller" => thriller_conventions(),
        "literary" => literary_conventions(),
        "fantasy" => fantasy_conventions(),
        "sci-fi" => scifi_conventions(),
        "horror" => horror_conventions(),
        "historical" => historical_conventions(),
        "ya" => ya_conventions(),
        "women's fiction" => womens_fiction_conventions(),
        "upmarket" => upmarket_conventions(),
        "middle-grade" => middle_grade_conventions(),
        _ => Vec::new(),
    }
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GenreEvidence {
    #[serde(default)]
    pub has_hea: Option<bool>,
    #[serde(default)]
    pub has_solution: Option<bool>,
    #[serde(default)]
    pub has_ticking_clock: Option<bool>,
    #[serde(default)]
    pub protagonist_survives: Option<bool>,
    #[serde(default)]
    pub character_growth: Option<bool>,
    #[serde(default)]
    pub worldbuilding_present: Option<bool>,
    #[serde(default)]
    pub word_count: Option<usize>,
    #[serde(default)]
    pub structure_health: Option<f64>,
    #[serde(default)]
    pub tension_arc_shape: Option<String>,
    /// Exact names from the selected checklist. These are supplied judgments.
    #[serde(default)]
    pub convention_judgments: BTreeMap<String, ConventionStatus>,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenreStructuralSignals {
    pub word_count: Option<usize>,
    pub structure_health: Option<f64>,
    pub tension_arc_shape: Option<String>,
}

fn direct_judgment(genre: &str, name: &str, evidence: &GenreEvidence) -> Option<ConventionStatus> {
    let value = match (genre, name) {
        ("romance", "HEA/HFN") => evidence.has_hea,
        ("mystery", "Solution Revealed") => evidence.has_solution,
        ("thriller", "Ticking Clock") => evidence.has_ticking_clock,
        ("thriller", "Protagonist Survives") => evidence.protagonist_survives,
        ("fantasy" | "sci-fi", "Worldbuilding") => evidence.worldbuilding_present,
        (_, "Character Growth" | "Personal Transformation") => evidence.character_growth,
        _ => None,
    };
    value.map(|present| {
        if present {
            ConventionStatus::Met
        } else {
            ConventionStatus::NotMet
        }
    })
}

/// Project supplied judgments onto a known, non-normative convention checklist.
/// Absent evidence remains Unknown. Explicit PartiallyMet/Subverted judgments are
/// accepted as caller labels; the engine never infers author intent from absence.
pub fn check_genre_compliance_checked(
    genre: &str,
    evidence: &GenreEvidence,
) -> Result<GenreComplianceResult> {
    let genre = canonical_genre(genre)?;
    let definitions = get_conventions(genre);
    if evidence.convention_judgments.len() > definitions.len()
        || evidence
            .convention_judgments
            .keys()
            .any(|name| !definitions.iter().any(|def| def.name == name))
    {
        return Err(Error::InvalidInput(
            "judgment does not name a convention in the selected genre".into(),
        ));
    }
    if evidence
        .word_count
        .is_some_and(|count| count > 64 * 1024 * 1024)
    {
        return Err(Error::InvalidInput(
            "word count exceeds manuscript byte-derived bound".into(),
        ));
    }
    if evidence
        .structure_health
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(Error::InvalidInput(
            "structure health must be finite within 0..=1".into(),
        ));
    }
    if evidence.tension_arc_shape.as_deref().is_some_and(|shape| {
        !matches!(
            shape,
            "rising" | "falling" | "flat" | "chaotic" | "varied" | "unknown"
        )
    }) {
        return Err(Error::InvalidInput("unsupported tension arc shape".into()));
    }
    let mut conventions = Vec::new();
    for def in definitions {
        let direct = direct_judgment(genre, def.name, evidence);
        let explicit = evidence.convention_judgments.get(def.name);
        if let (Some(direct), Some(explicit)) = (&direct, explicit)
            && direct != explicit
        {
            return Err(Error::InvalidInput(
                "conflicting supplied convention judgments".into(),
            ));
        }
        let status = explicit
            .cloned()
            .or(direct)
            .unwrap_or(ConventionStatus::Unknown);
        let source = if status == ConventionStatus::Unknown {
            "Unassessed: no supplied judgment"
        } else {
            "Caller-supplied judgment; not independently detected"
        };
        conventions.push(GenreConvention {
            name: def.name.into(),
            description: def.description.into(),
            status,
            evidence: source.into(),
        });
    }
    let present_count = conventions
        .iter()
        .filter(|c| {
            matches!(
                c.status,
                ConventionStatus::Met | ConventionStatus::Subverted
            )
        })
        .count();
    let absent_count = conventions
        .iter()
        .filter(|c| c.status == ConventionStatus::NotMet)
        .count();
    let unknown_count = conventions
        .iter()
        .filter(|c| c.status == ConventionStatus::Unknown)
        .count();
    let assessed_count = conventions.len() - unknown_count;
    let deviations = conventions
        .iter()
        .filter(|c| c.status == ConventionStatus::NotMet)
        .map(|c| format!("Supplied as absent: {}", c.name))
        .collect();
    let strengths = conventions
        .iter()
        .filter(|c| c.status == ConventionStatus::Met)
        .map(|c| format!("Supplied as present: {}", c.name))
        .collect();
    Ok(GenreComplianceResult {
        availability: if assessed_count == 0 { AnalysisAvailability::Unavailable } else if unknown_count > 0 { AnalysisAvailability::Partial } else { AnalysisAvailability::Complete },
        diagnostics: vec!["caller_supplied_checklist;quality_blend_and_intent_not_inferred;conventions_are_not_normative_requirements".into()],
        quality_assessed: false, judgment_provenance: "caller_supplied".into(), blend_assessed: false,
        structural_signals: GenreStructuralSignals { word_count: evidence.word_count, structure_health: evidence.structure_health, tension_arc_shape: evidence.tension_arc_shape.clone() },
        evidence_coverage: assessed_count as f64 / conventions.len() as f64,
        assessed_count, unknown_count, genre: genre.into(), conventions, present_count, absent_count,
        is_genre_blend: false, secondary_genres: Vec::new(), deviations, strengths,
    })
}

pub fn check_genre_compliance(
    genre: &str,
    has_hea: bool,
    has_solution: bool,
    has_ticking_clock: bool,
    protagonist_survives: bool,
    character_growth: bool,
    worldbuilding_present: bool,
    word_count: usize,
    structure_health: f64,
    tension_arc_shape: &str,
) -> GenreComplianceResult {
    check_genre_compliance_checked(
        genre,
        &GenreEvidence {
            has_hea: Some(has_hea),
            has_solution: Some(has_solution),
            has_ticking_clock: Some(has_ticking_clock),
            protagonist_survives: Some(protagonist_survives),
            character_growth: Some(character_growth),
            worldbuilding_present: Some(worldbuilding_present),
            word_count: Some(word_count),
            structure_health: Some(structure_health),
            tension_arc_shape: Some(tension_arc_shape.into()),
            convention_judgments: BTreeMap::new(),
        },
    )
    .unwrap_or_else(|_| GenreComplianceResult {
        availability: AnalysisAvailability::Unavailable,
        diagnostics: vec!["invalid_genre_evidence;use_checked_api".into()],
        quality_assessed: false,
        judgment_provenance: "unavailable".into(),
        blend_assessed: false,
        structural_signals: GenreStructuralSignals::default(),
        assessed_count: 0,
        unknown_count: 0,
        evidence_coverage: 0.0,
        genre: String::new(),
        conventions: Vec::new(),
        present_count: 0,
        absent_count: 0,
        is_genre_blend: false,
        secondary_genres: Vec::new(),
        deviations: Vec::new(),
        strengths: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_romance_hea_required() {
        let result = check_genre_compliance(
            "romance", false, false, false, true, true, false, 80000, 0.7, "rising",
        );
        let hea = result
            .conventions
            .iter()
            .find(|c| c.name == "HEA/HFN")
            .unwrap();
        assert!(matches!(hea.status, ConventionStatus::NotMet));
        assert!(result.absent_count >= 1);
    }

    #[test]
    fn test_mystery_needs_solution() {
        let result = check_genre_compliance(
            "mystery", false, true, false, true, true, false, 90000, 0.7, "rising",
        );
        let sol = result
            .conventions
            .iter()
            .find(|c| c.name == "Solution Revealed")
            .unwrap();
        assert!(matches!(sol.status, ConventionStatus::Met));
        assert!(result.present_count >= 1);
    }

    #[test]
    fn test_thriller_ticking_clock() {
        let result = check_genre_compliance(
            "thriller", false, false, true, true, true, false, 85000, 0.75, "rising",
        );
        let clock = result
            .conventions
            .iter()
            .find(|c| c.name == "Ticking Clock")
            .unwrap();
        assert!(matches!(clock.status, ConventionStatus::Met));
    }

    #[test]
    fn test_coarse_evidence_does_not_prove_genre_blend() {
        let result = check_genre_compliance(
            "mystery", true, true, true, true, true, false, 80000, 0.7, "rising",
        );
        assert!(!result.is_genre_blend);
        assert!(!result.blend_assessed);
    }

    #[test]
    fn test_present_absent_counts() {
        let result = check_genre_compliance(
            "thriller", false, false, true, true, true, false, 85000, 0.75, "rising",
        );
        let total = result.conventions.len();
        // present + absent + partially_met = total
        assert!(result.present_count + result.absent_count <= total);
        // ticking clock is met, so present_count >= 1
        assert!(result.present_count >= 1);
    }
}
