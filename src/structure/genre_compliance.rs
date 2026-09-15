use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenreConvention {
    pub name: String,
    pub description: String,
    pub status: ConventionStatus,
    pub evidence: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConventionStatus {
    Met,
    PartiallyMet,
    NotMet,
    Subverted,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenreComplianceResult {
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

fn get_conventions(genre: &str) -> Vec<ConventionDef> {
    match genre.to_lowercase().as_str() {
        "romance" => romance_conventions(),
        "mystery" => mystery_conventions(),
        "thriller" => thriller_conventions(),
        "literary" | "literary fiction" => literary_conventions(),
        "fantasy" => fantasy_conventions(),
        "sci-fi" | "science fiction" => scifi_conventions(),
        "horror" => horror_conventions(),
        "historical" | "historical fiction" => historical_conventions(),
        "ya" | "young adult" => ya_conventions(),
        "women's fiction" | "womens fiction" => womens_fiction_conventions(),
        "upmarket" | "upmarket fiction" => upmarket_conventions(),
        "middle-grade" | "middle grade" => middle_grade_conventions(),
        _ => literary_conventions(),
    }
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
    let convention_defs = get_conventions(genre);
    let genre_lower = genre.to_lowercase();

    let mut conventions = Vec::new();

    for def in &convention_defs {
        let status = evaluate_convention(
            &genre_lower,
            def.name,
            has_hea,
            has_solution,
            has_ticking_clock,
            protagonist_survives,
            character_growth,
            worldbuilding_present,
            word_count,
            structure_health,
            tension_arc_shape,
        );

        let evidence = evidence_for(&genre_lower, def.name, &status);

        conventions.push(GenreConvention {
            name: def.name.to_string(),
            description: def.description.to_string(),
            status,
            evidence,
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
        .filter(|c| matches!(c.status, ConventionStatus::NotMet))
        .count();

    // Detect secondary genre signals
    let mut secondary_genres = Vec::new();
    if genre_lower != "romance" && has_hea && character_growth {
        secondary_genres.push("romance".to_string());
    }
    if genre_lower != "thriller" && has_ticking_clock && tension_arc_shape == "rising" {
        secondary_genres.push("thriller".to_string());
    }
    if genre_lower != "mystery" && has_solution {
        secondary_genres.push("mystery".to_string());
    }
    if genre_lower != "literary" && structure_health > 0.8 {
        secondary_genres.push("literary".to_string());
    }

    // A blend requires at least one strong secondary signal.
    let is_genre_blend =
        (genre_lower != "thriller" && has_ticking_clock && tension_arc_shape == "rising")
            || (genre_lower != "romance" && has_hea && character_growth);

    let deviations: Vec<String> = conventions
        .iter()
        .filter(|c| matches!(c.status, ConventionStatus::NotMet))
        .map(|c| format!("Missing: {}", c.name))
        .collect();

    let strengths: Vec<String> = conventions
        .iter()
        .filter(|c| matches!(c.status, ConventionStatus::Met))
        .map(|c| format!("Present: {}", c.name))
        .collect();

    GenreComplianceResult {
        genre: genre.to_string(),
        conventions,
        present_count,
        absent_count,
        is_genre_blend,
        secondary_genres,
        deviations,
        strengths,
    }
}

fn evaluate_convention(
    genre: &str,
    convention_name: &str,
    has_hea: bool,
    has_solution: bool,
    has_ticking_clock: bool,
    protagonist_survives: bool,
    character_growth: bool,
    worldbuilding_present: bool,
    word_count: usize,
    structure_health: f64,
    tension_arc_shape: &str,
) -> ConventionStatus {
    match (genre, convention_name) {
        // Romance
        ("romance", "HEA/HFN") => {
            if has_hea {
                ConventionStatus::Met
            } else {
                ConventionStatus::NotMet
            }
        }
        ("romance", "Character Growth") => {
            if character_growth {
                ConventionStatus::Met
            } else {
                ConventionStatus::PartiallyMet
            }
        }

        // Mystery
        ("mystery", "Solution Revealed") => {
            if has_solution {
                ConventionStatus::Met
            } else {
                ConventionStatus::NotMet
            }
        }
        ("mystery", "Rising Suspicion") => match tension_arc_shape {
            "rising" => ConventionStatus::Met,
            "flat" => ConventionStatus::NotMet,
            _ => ConventionStatus::PartiallyMet,
        },

        // Thriller
        ("thriller", "Ticking Clock") => {
            if has_ticking_clock {
                ConventionStatus::Met
            } else {
                ConventionStatus::NotMet
            }
        }
        ("thriller", "Protagonist Survives") => {
            if protagonist_survives {
                ConventionStatus::Met
            } else {
                ConventionStatus::Subverted
            }
        }
        ("thriller", "Fast Pacing") => match tension_arc_shape {
            "rising" => ConventionStatus::Met,
            "chaotic" => ConventionStatus::PartiallyMet,
            _ => ConventionStatus::NotMet,
        },
        ("thriller", "High Stakes") => {
            if structure_health > 0.6 {
                ConventionStatus::Met
            } else {
                ConventionStatus::PartiallyMet
            }
        }

        // Fantasy / Sci-fi
        ("fantasy", "Worldbuilding") | ("sci-fi" | "science fiction", "Worldbuilding") => {
            if worldbuilding_present {
                ConventionStatus::Met
            } else {
                ConventionStatus::NotMet
            }
        }
        ("fantasy", "Character Growth") => {
            if character_growth {
                ConventionStatus::Met
            } else {
                ConventionStatus::PartiallyMet
            }
        }

        // Literary
        ("literary" | "literary fiction", "Prose Quality") => {
            if structure_health > 0.7 {
                ConventionStatus::Met
            } else {
                ConventionStatus::PartiallyMet
            }
        }
        ("literary" | "literary fiction", "Character Interiority") => {
            if structure_health > 0.6 {
                ConventionStatus::Met
            } else {
                ConventionStatus::PartiallyMet
            }
        }

        // Horror
        ("horror", "Threat Escalation") => match tension_arc_shape {
            "rising" => ConventionStatus::Met,
            "flat" => ConventionStatus::NotMet,
            _ => ConventionStatus::PartiallyMet,
        },

        // YA
        ("ya" | "young adult", "Fast Pacing") => {
            if word_count < 90_000 && tension_arc_shape == "rising" {
                ConventionStatus::Met
            } else if word_count < 100_000 {
                ConventionStatus::PartiallyMet
            } else {
                ConventionStatus::NotMet
            }
        }

        // Middle grade
        ("middle-grade" | "middle grade", "Hopeful Resolution") => {
            if has_hea || protagonist_survives {
                ConventionStatus::Met
            } else {
                ConventionStatus::NotMet
            }
        }

        // Generic: character growth applies broadly
        (_, "Character Growth" | "Personal Transformation") => {
            if character_growth {
                ConventionStatus::Met
            } else {
                ConventionStatus::PartiallyMet
            }
        }

        // Default: use structure health as proxy
        _ => {
            if structure_health > 0.7 {
                ConventionStatus::Met
            } else if structure_health > 0.4 {
                ConventionStatus::PartiallyMet
            } else {
                ConventionStatus::NotMet
            }
        }
    }
}

fn evidence_for(_genre: &str, convention: &str, status: &ConventionStatus) -> String {
    let status_word = match status {
        ConventionStatus::Met => "Present",
        ConventionStatus::PartiallyMet => "Partially present",
        ConventionStatus::NotMet => "Not detected",
        ConventionStatus::Subverted => "Intentionally subverted",
    };
    format!("{}: {}", convention, status_word)
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
    fn test_genre_blend_detection() {
        let result = check_genre_compliance(
            "mystery", true, true, true, true, true, false, 80000, 0.7, "rising",
        );
        assert!(result.is_genre_blend);
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
