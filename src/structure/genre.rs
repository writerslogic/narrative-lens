use serde::Serialize;

use crate::craft::lexicon::WordMatcher;
use crate::substrate::utils::cosine_similarity;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum Genre {
    Literary,
    Thriller,
    Mystery,
    Romance,
    Fantasy,
    SciFi,
    Horror,
    HistoricalFiction,
    YoungAdult,
    ChildrensBook,
    General,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenreProfile {
    pub genre: Genre,
    pub name: String,
    pub target_fkgl_min: f64,
    pub target_fkgl_max: f64,
    pub target_velocity_min: f64,
    pub target_velocity_max: f64,
    pub dialogue_ratio_min: f64,
    pub dialogue_ratio_max: f64,
    pub tension_variance_min: f64,
    pub tonal_whiplash_tolerance: usize,
    pub accepts_nonlinear: bool,
    pub requires_resolution: bool,
    pub word_count_min: usize,
    pub word_count_max: usize,
    pub expected_scenes_per_80k: (usize, usize),
    pub expected_avg_scene_length: (usize, usize),
    pub max_acceptable_adverb_density: f64,
    pub expected_character_count: (usize, usize),
    pub expected_attribution_rate: (f64, f64),
    pub min_protagonist_agency_growth: f64,
    pub tension_weight: f64,
    pub expected_tension_arc: String,
    pub climax_position_range: (f64, f64),
    pub expected_vocab_sophistication: (f64, f64),
    pub exposition_tolerance: f64,
}

impl GenreProfile {
    fn __repr__(&self) -> String {
        format!("GenreProfile(genre={:?}, name='{}')", self.genre, self.name)
    }
}

pub fn get_genre_profile(genre: Genre) -> GenreProfile {
    let base = match genre {
        Genre::Literary => (
            "Literary Fiction",
            8.0,
            16.0,
            15.0,
            30.0,
            0.15,
            0.45,
            0.3,
            5,
            true,
            false,
            60_000,
            120_000,
            (30, 60),
            (1500, 3500),
            0.025,
            (3, 12),
            (0.3, 0.7),
            0.05,
            0.20,
            "slow_burn",
            (0.75, 0.95),
            (0.55, 0.85),
            0.15,
        ),
        Genre::Thriller => (
            "Thriller",
            5.0,
            10.0,
            10.0,
            20.0,
            0.25,
            0.50,
            0.5,
            2,
            false,
            true,
            70_000,
            100_000,
            (50, 80),
            (800, 2000),
            0.015,
            (4, 15),
            (0.5, 0.85),
            0.15,
            0.40,
            "escalating",
            (0.80, 0.95),
            (0.30, 0.55),
            0.08,
        ),
        Genre::Mystery => (
            "Mystery",
            6.0,
            12.0,
            12.0,
            22.0,
            0.30,
            0.55,
            0.3,
            2,
            false,
            true,
            70_000,
            90_000,
            (40, 70),
            (1000, 2500),
            0.020,
            (5, 15),
            (0.5, 0.80),
            0.10,
            0.35,
            "escalating",
            (0.80, 0.95),
            (0.35, 0.60),
            0.10,
        ),
        Genre::Romance => (
            "Romance",
            5.0,
            10.0,
            12.0,
            22.0,
            0.35,
            0.60,
            0.25,
            3,
            false,
            true,
            50_000,
            90_000,
            (40, 65),
            (1000, 2500),
            0.025,
            (3, 10),
            (0.5, 0.85),
            0.10,
            0.25,
            "escalating",
            (0.75, 0.90),
            (0.25, 0.50),
            0.10,
        ),
        Genre::Fantasy => (
            "Fantasy",
            7.0,
            14.0,
            15.0,
            28.0,
            0.20,
            0.45,
            0.35,
            3,
            true,
            true,
            80_000,
            150_000,
            (35, 60),
            (1200, 3000),
            0.020,
            (5, 20),
            (0.4, 0.75),
            0.15,
            0.30,
            "escalating",
            (0.80, 0.95),
            (0.45, 0.75),
            0.12,
        ),
        Genre::SciFi => (
            "Science Fiction",
            8.0,
            14.0,
            15.0,
            28.0,
            0.20,
            0.45,
            0.35,
            3,
            true,
            true,
            70_000,
            120_000,
            (35, 60),
            (1200, 3000),
            0.020,
            (4, 18),
            (0.4, 0.75),
            0.10,
            0.30,
            "escalating",
            (0.80, 0.95),
            (0.50, 0.80),
            0.15,
        ),
        Genre::Horror => (
            "Horror",
            5.0,
            12.0,
            10.0,
            22.0,
            0.20,
            0.45,
            0.5,
            2,
            false,
            false,
            60_000,
            100_000,
            (45, 75),
            (800, 2200),
            0.020,
            (3, 12),
            (0.4, 0.75),
            0.05,
            0.40,
            "escalating",
            (0.80, 0.95),
            (0.30, 0.60),
            0.08,
        ),
        Genre::HistoricalFiction => (
            "Historical Fiction",
            8.0,
            14.0,
            15.0,
            28.0,
            0.25,
            0.50,
            0.25,
            3,
            true,
            true,
            80_000,
            120_000,
            (35, 55),
            (1500, 3500),
            0.022,
            (5, 18),
            (0.4, 0.75),
            0.10,
            0.25,
            "slow_burn",
            (0.75, 0.92),
            (0.50, 0.80),
            0.15,
        ),
        Genre::YoungAdult => (
            "Young Adult",
            4.0,
            8.0,
            10.0,
            18.0,
            0.35,
            0.60,
            0.25,
            3,
            false,
            true,
            50_000,
            80_000,
            (45, 70),
            (800, 2000),
            0.025,
            (3, 12),
            (0.5, 0.85),
            0.15,
            0.30,
            "escalating",
            (0.78, 0.92),
            (0.20, 0.45),
            0.10,
        ),
        Genre::ChildrensBook => (
            "Children's Book",
            2.0,
            5.0,
            8.0,
            15.0,
            0.40,
            0.65,
            0.1,
            1,
            false,
            true,
            20_000,
            50_000,
            (50, 80),
            (500, 1500),
            0.030,
            (2, 8),
            (0.6, 0.90),
            0.10,
            0.15,
            "escalating",
            (0.70, 0.90),
            (0.10, 0.30),
            0.08,
        ),
        Genre::General => (
            "General Fiction",
            5.0,
            14.0,
            12.0,
            25.0,
            0.10,
            0.70,
            0.15,
            3,
            true,
            false,
            50_000,
            120_000,
            (35, 70),
            (1000, 3000),
            0.025,
            (3, 15),
            (0.4, 0.80),
            0.05,
            0.25,
            "varied",
            (0.75, 0.95),
            (0.30, 0.70),
            0.12,
        ),
    };
    GenreProfile {
        genre,
        name: base.0.into(),
        target_fkgl_min: base.1,
        target_fkgl_max: base.2,
        target_velocity_min: base.3,
        target_velocity_max: base.4,
        dialogue_ratio_min: base.5,
        dialogue_ratio_max: base.6,
        tension_variance_min: base.7,
        tonal_whiplash_tolerance: base.8,
        accepts_nonlinear: base.9,
        requires_resolution: base.10,
        word_count_min: base.11,
        word_count_max: base.12,
        expected_scenes_per_80k: base.13,
        expected_avg_scene_length: base.14,
        max_acceptable_adverb_density: base.15,
        expected_character_count: base.16,
        expected_attribution_rate: base.17,
        min_protagonist_agency_growth: base.18,
        tension_weight: base.19,
        expected_tension_arc: base.20.into(),
        climax_position_range: base.21,
        expected_vocab_sophistication: base.22,
        exposition_tolerance: base.23,
    }
}

#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenreFinding {
    pub metric: String,
    pub value: f64,
    pub expected_min: f64,
    pub expected_max: f64,
    pub severity: String,
    pub message: String,
}

impl GenreFinding {
    fn __repr__(&self) -> String {
        format!(
            "GenreFinding(metric='{}', severity='{}', message='{}')",
            self.metric, self.severity, self.message
        )
    }
}

/// Short semantic descriptions for each genre, used for SBERT cosine similarity.
const GENRE_DESCRIPTIONS: &[(Genre, &str)] = &[
    (Genre::Literary, "literary fiction with complex prose, introspective characters, ambiguous morality, slow-burn narrative, and thematic depth"),
    (Genre::Thriller, "thriller with high stakes, danger, murder, conspiracy, chase sequences, escalating tension, and a ticking clock"),
    (Genre::Mystery, "mystery with a detective, crime, clues, suspects, investigation, alibi, and a solution revealed at the end"),
    (Genre::Romance, "romance with love, passion, desire, attraction, relationships, emotional connection, and a happy ending"),
    (Genre::Fantasy, "fantasy with magic, kingdoms, quests, dragons, prophecy, sorcery, and an epic battle between good and evil"),
    (Genre::SciFi, "science fiction with technology, space travel, aliens, robots, artificial intelligence, future worlds, and scientific speculation"),
    (Genre::Horror, "horror with fear, dread, monsters, supernatural evil, blood, darkness, nightmares, and psychological terror"),
    (Genre::HistoricalFiction, "historical fiction set in a specific past era with period-accurate details, historical events, and authentic atmosphere"),
    (Genre::YoungAdult, "young adult fiction with teenage protagonists, coming of age, school, identity, first love, and family conflict"),
    (Genre::ChildrensBook, "children's book with simple language, young characters, adventure, friendship, moral lessons, and wonder"),
    (Genre::General, "general fiction with everyday characters, realistic settings, and universal human experiences"),
];

/// Attempt SBERT-based genre detection. Returns `None` if the model is unavailable.
fn detect_genre_sbert(manuscript_sample: &str) -> Option<Genre> {
    // Encode genre descriptions; if all empty, model is unavailable
    let genre_embeddings: Vec<(Genre, Vec<f64>)> = GENRE_DESCRIPTIONS
        .iter()
        .map(|(genre, desc)| (*genre, crate::substrate::sbert::encode_text(desc)))
        .collect();

    if genre_embeddings.iter().all(|(_, emb)| emb.is_empty()) {
        return None;
    }

    let sample_emb = crate::substrate::sbert::encode_text(manuscript_sample);
    if sample_emb.is_empty() {
        return None;
    }

    // Find the genre description with highest cosine similarity to the sample
    let best = genre_embeddings
        .iter()
        .filter(|(_, emb)| !emb.is_empty())
        .map(|(genre, emb)| (*genre, cosine_similarity(&sample_emb, emb)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    best.map(|(genre, _)| genre)
}

/// Theme keyword lists used for the heuristic fallback genre detection.
const THRILLER_THEMES: &[&str] = &[
    "death",
    "murder",
    "kill",
    "chase",
    "conspiracy",
    "danger",
    "escape",
    "threat",
    "spy",
    "hunt",
    "weapon",
    "hostage",
    "bomb",
    "terror",
];

const HORROR_THEMES: &[&str] = &[
    "fear",
    "dread",
    "horror",
    "dark",
    "nightmare",
    "monster",
    "haunt",
    "blood",
    "scream",
    "evil",
    "shadow",
    "curse",
    "demon",
    "ghost",
];

const ROMANCE_THEMES: &[&str] = &[
    "love",
    "heart",
    "passion",
    "desire",
    "kiss",
    "romance",
    "attraction",
    "relationship",
    "wedding",
    "longing",
    "intimacy",
    "devotion",
];

const FANTASY_THEMES: &[&str] = &[
    "power", "magic", "kingdom", "quest", "dragon", "sword", "prophecy", "betrayal", "freedom",
    "throne", "realm", "sorcery", "enchant", "fate",
];

const SCIFI_THEMES: &[&str] = &[
    "technology",
    "space",
    "alien",
    "future",
    "robot",
    "AI",
    "colony",
    "planet",
    "starship",
    "cybernetic",
    "quantum",
    "galaxy",
    "android",
];

const MYSTERY_THEMES: &[&str] = &[
    "clue",
    "detective",
    "suspect",
    "alibi",
    "witness",
    "evidence",
    "investigate",
    "crime",
    "victim",
    "motive",
    "puzzle",
];

const YA_THEMES: &[&str] = &[
    "school", "friend", "bully", "identity", "family", "growing", "first", "teen", "parent",
    "belong", "fitting",
];

fn count_theme_matches(themes: &[String], keywords: &[&str]) -> usize {
    // Match a genre keyword as a word stem within the theme: "friend" counts
    // for the theme "friendship", but "teen" no longer fires inside "canteen"
    // the way the previous str::contains did.
    let matcher = WordMatcher::new(keywords);
    themes
        .iter()
        .filter(|t| matcher.is_prefix_match(t.as_str()))
        .count()
}

fn tension_variance(pattern: &[f64]) -> f64 {
    if pattern.len() < 2 {
        return 0.0;
    }
    let mean = pattern.iter().sum::<f64>() / pattern.len() as f64;
    let var = pattern.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / pattern.len() as f64;
    var.sqrt()
}

/// Map a writer-declared genre label to a [`Genre`]. Case-insensitive, alias-aware;
/// returns `None` for unrecognised labels so the caller can fall back to detection.
pub fn genre_from_label(label: &str) -> Option<Genre> {
    match label.trim().to_lowercase().as_str() {
        "literary" | "literary fiction" => Some(Genre::Literary),
        "thriller" | "suspense" => Some(Genre::Thriller),
        "mystery" | "crime" | "detective" => Some(Genre::Mystery),
        "romance" => Some(Genre::Romance),
        "fantasy" => Some(Genre::Fantasy),
        "scifi" | "sci-fi" | "science fiction" => Some(Genre::SciFi),
        "horror" => Some(Genre::Horror),
        "historical" | "historical fiction" => Some(Genre::HistoricalFiction),
        "ya" | "young adult" => Some(Genre::YoungAdult),
        "childrens" | "children's" | "childrens book" | "picture book" => {
            Some(Genre::ChildrensBook)
        }
        "general" => Some(Genre::General),
        _ => None,
    }
}

fn first_words(text: &str, n: usize) -> String {
    let mut count = 0;
    let mut end = text.len();
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            count += 1;
            if count >= n {
                end = i;
                break;
            }
        }
    }
    text[..end].to_string()
}

/// Detect genre from a raw manuscript sample using SBERT cosine similarity
/// (via [`detect_genre_sbert`]) when a model is available, falling back to
/// the label/theme-driven [`detect_genre`] otherwise.
pub fn detect_genre_smart(
    sample_text: &str,
    themes: Vec<String>,
    tension_pattern: Vec<f64>,
    dialogue_ratio: f64,
    fkgl: f64,
) -> Genre {
    let sample = first_words(sample_text, 500);
    if let Some(genre) = detect_genre_sbert(&sample) {
        return genre;
    }
    detect_genre(themes, tension_pattern, dialogue_ratio, fkgl)
}

/// Heuristic fallback genre detection using theme keywords and prose statistics.
fn detect_genre_heuristic(
    themes: &[String],
    tension_pattern: &[f64],
    dialogue_ratio: f64,
    fkgl: f64,
) -> Genre {
    let tv = tension_variance(tension_pattern);

    let thriller_t = count_theme_matches(themes, THRILLER_THEMES) as f64;
    let horror_t = count_theme_matches(themes, HORROR_THEMES) as f64;
    let romance_t = count_theme_matches(themes, ROMANCE_THEMES) as f64;
    let fantasy_t = count_theme_matches(themes, FANTASY_THEMES) as f64;
    let scifi_t = count_theme_matches(themes, SCIFI_THEMES) as f64;
    let mystery_t = count_theme_matches(themes, MYSTERY_THEMES) as f64;
    let ya_t = count_theme_matches(themes, YA_THEMES) as f64;

    // Score each genre
    let thriller_score = {
        let mut s = thriller_t * 2.0;
        if tv > 0.4 {
            s += 3.0;
        }
        if dialogue_ratio < 0.45 {
            s += 1.0;
        }
        if fkgl <= 10.0 {
            s += 1.0;
        }
        s
    };
    let horror_score = {
        let mut s = horror_t * 2.0;
        if tv > 0.4 {
            s += 3.0;
        }
        if dialogue_ratio < 0.45 {
            s += 1.0;
        }
        s
    };
    let romance_score = {
        let mut s = romance_t * 2.5;
        if dialogue_ratio > 0.35 {
            s += 2.0;
        }
        if fkgl <= 10.0 {
            s += 1.0;
        }
        s
    };
    let mystery_score = {
        let mut s = mystery_t * 2.5;
        if dialogue_ratio > 0.30 {
            s += 1.5;
        }
        if tv > 0.2 && tv < 0.5 {
            s += 1.0;
        }
        s
    };
    let fantasy_score = {
        let mut s = fantasy_t * 2.0;
        if tv > 0.3 {
            s += 1.5;
        }
        if fkgl >= 7.0 {
            s += 1.0;
        }
        s
    };
    let scifi_score = {
        let mut s = scifi_t * 2.5;
        if fkgl >= 8.0 {
            s += 1.0;
        }
        s
    };
    let ya_score = {
        let mut s = ya_t * 2.0;
        if fkgl < 8.0 {
            s += 2.0;
        }
        if dialogue_ratio > 0.35 {
            s += 1.5;
        }
        if tv < 0.35 {
            s += 1.0;
        }
        s
    };
    let childrens_score = {
        let mut s = 0.0_f64;
        if fkgl < 5.0 {
            s += 4.0;
        }
        if dialogue_ratio > 0.40 {
            s += 2.0;
        }
        if tv < 0.15 {
            s += 2.0;
        }
        s
    };
    let literary_score = {
        let mut s = 0.0_f64;
        if fkgl >= 10.0 {
            s += 3.0;
        }
        if tv < 0.25 {
            s += 2.0;
        }
        if dialogue_ratio < 0.35 {
            s += 1.5;
        }
        s
    };
    let historical_score = {
        let mut s = 0.0_f64;
        if fkgl >= 8.0 {
            s += 1.5;
        }
        if tv > 0.2 && tv < 0.45 {
            s += 1.0;
        }
        s
    };

    let scores: &[(Genre, f64)] = &[
        (Genre::Thriller, thriller_score),
        (Genre::Horror, horror_score),
        (Genre::Romance, romance_score),
        (Genre::Mystery, mystery_score),
        (Genre::Fantasy, fantasy_score),
        (Genre::SciFi, scifi_score),
        (Genre::YoungAdult, ya_score),
        (Genre::ChildrensBook, childrens_score),
        (Genre::Literary, literary_score),
        (Genre::HistoricalFiction, historical_score),
    ];

    let best = scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    match best {
        Some(&(genre, score)) if score >= 3.0 => genre,
        _ => Genre::General,
    }
}

/// Detect genre using SBERT cosine similarity against genre descriptions when the
/// model is available, degrading to the keyword/prose-stats heuristic otherwise.
pub fn detect_genre(
    themes: Vec<String>,
    tension_pattern: Vec<f64>,
    dialogue_ratio: f64,
    fkgl: f64,
) -> Genre {
    // Build a representative sample from the theme names joined with prose stats
    // context. This gives SBERT something meaningful to embed even when the full
    // manuscript text is not passed in.
    let sample = if themes.is_empty() {
        format!(
            "prose with fkgl {:.1} dialogue ratio {:.2}",
            fkgl, dialogue_ratio
        )
    } else {
        format!(
            "{} fkgl {:.1} dialogue ratio {:.2}",
            themes.join(" "),
            fkgl,
            dialogue_ratio
        )
    };

    if let Some(genre) = detect_genre_sbert(&sample) {
        return genre;
    }

    // Fallback: heuristic
    detect_genre_heuristic(&themes, &tension_pattern, dialogue_ratio, fkgl)
}

pub fn evaluate_against_genre(
    genre: Genre,
    fkgl: f64,
    velocity: f64,
    dialogue_ratio: f64,
    tension_variance: f64,
    tonal_whiplash_count: usize,
    word_count: usize,
) -> Vec<GenreFinding> {
    let profile = get_genre_profile(genre);
    let mut findings = Vec::new();

    // Helper: how far outside the range a value falls, as fraction of range width
    let deviation_severity = |val: f64, lo: f64, hi: f64| -> &'static str {
        let range = hi - lo;
        if range <= 0.0 {
            return "info";
        }
        let dist = if val < lo { lo - val } else { val - hi };
        let ratio = dist / range;
        if ratio > 0.5 {
            "concern"
        } else if ratio > 0.2 {
            "warning"
        } else {
            "info"
        }
    };

    // FKGL
    if fkgl < profile.target_fkgl_min || fkgl > profile.target_fkgl_max {
        let sev = deviation_severity(fkgl, profile.target_fkgl_min, profile.target_fkgl_max);
        let msg = if fkgl < profile.target_fkgl_min {
            format!(
                "Readability (FKGL {:.1}) is below typical {} range ({:.0}-{:.0}); prose may be too simple for the genre.",
                fkgl, profile.name, profile.target_fkgl_min, profile.target_fkgl_max
            )
        } else {
            format!(
                "Readability (FKGL {:.1}) is above typical {} range ({:.0}-{:.0}); prose may be too dense for the genre.",
                fkgl, profile.name, profile.target_fkgl_min, profile.target_fkgl_max
            )
        };
        findings.push(GenreFinding {
            metric: "fkgl".into(),
            value: fkgl,
            expected_min: profile.target_fkgl_min,
            expected_max: profile.target_fkgl_max,
            severity: sev.into(),
            message: msg,
        });
    }

    // Velocity
    if velocity < profile.target_velocity_min || velocity > profile.target_velocity_max {
        let sev = deviation_severity(
            velocity,
            profile.target_velocity_min,
            profile.target_velocity_max,
        );
        let msg = if velocity < profile.target_velocity_min {
            format!(
                "Pacing velocity ({:.1}) is slower than typical for {}; scenes may drag.",
                velocity, profile.name
            )
        } else {
            format!(
                "Pacing velocity ({:.1}) is faster than typical for {}; scenes may feel rushed.",
                velocity, profile.name
            )
        };
        findings.push(GenreFinding {
            metric: "velocity".into(),
            value: velocity,
            expected_min: profile.target_velocity_min,
            expected_max: profile.target_velocity_max,
            severity: sev.into(),
            message: msg,
        });
    }

    // Dialogue ratio
    if dialogue_ratio < profile.dialogue_ratio_min || dialogue_ratio > profile.dialogue_ratio_max {
        let sev = deviation_severity(
            dialogue_ratio,
            profile.dialogue_ratio_min,
            profile.dialogue_ratio_max,
        );
        let msg = if dialogue_ratio < profile.dialogue_ratio_min {
            format!(
                "Dialogue ratio ({:.0}%) is below {} expectations ({:.0}%-{:.0}%); consider adding more dialogue.",
                dialogue_ratio * 100.0, profile.name,
                profile.dialogue_ratio_min * 100.0, profile.dialogue_ratio_max * 100.0
            )
        } else {
            format!(
                "Dialogue ratio ({:.0}%) is above {} expectations ({:.0}%-{:.0}%); narrative may be dialogue-heavy.",
                dialogue_ratio * 100.0, profile.name,
                profile.dialogue_ratio_min * 100.0, profile.dialogue_ratio_max * 100.0
            )
        };
        findings.push(GenreFinding {
            metric: "dialogue_ratio".into(),
            value: dialogue_ratio,
            expected_min: profile.dialogue_ratio_min,
            expected_max: profile.dialogue_ratio_max,
            severity: sev.into(),
            message: msg,
        });
    }

    // Tension variance
    if tension_variance < profile.tension_variance_min {
        let sev = if tension_variance < profile.tension_variance_min * 0.5 {
            "concern"
        } else {
            "warning"
        };
        findings.push(GenreFinding {
            metric: "tension_variance".into(),
            value: tension_variance,
            expected_min: profile.tension_variance_min,
            expected_max: 1.0,
            severity: sev.into(),
            message: format!(
                "Tension variance ({:.2}) is low for {}; the narrative may lack dynamic range.",
                tension_variance, profile.name
            ),
        });
    }

    // Tonal whiplash
    if tonal_whiplash_count > profile.tonal_whiplash_tolerance {
        let over = tonal_whiplash_count - profile.tonal_whiplash_tolerance;
        let sev = if over > profile.tonal_whiplash_tolerance {
            "concern"
        } else {
            "warning"
        };
        findings.push(GenreFinding {
            metric: "tonal_whiplash".into(),
            value: tonal_whiplash_count as f64,
            expected_min: 0.0,
            expected_max: profile.tonal_whiplash_tolerance as f64,
            severity: sev.into(),
            message: format!(
                "{} tonal shifts detected (tolerance for {} is {}); abrupt tone changes may disorient readers.",
                tonal_whiplash_count, profile.name, profile.tonal_whiplash_tolerance
            ),
        });
    }

    // Word count
    if word_count < profile.word_count_min || word_count > profile.word_count_max {
        let sev =
            if word_count < profile.word_count_min / 2 || word_count > profile.word_count_max * 2 {
                "concern"
            } else {
                "warning"
            };
        let msg = if word_count < profile.word_count_min {
            format!(
                "Word count ({}) is below typical {} range ({}K-{}K).",
                word_count,
                profile.name,
                profile.word_count_min / 1000,
                profile.word_count_max / 1000
            )
        } else {
            format!(
                "Word count ({}) is above typical {} range ({}K-{}K).",
                word_count,
                profile.name,
                profile.word_count_min / 1000,
                profile.word_count_max / 1000
            )
        };
        findings.push(GenreFinding {
            metric: "word_count".into(),
            value: word_count as f64,
            expected_min: profile.word_count_min as f64,
            expected_max: profile.word_count_max as f64,
            severity: sev.into(),
            message: msg,
        });
    }

    findings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_genre_profile_thriller() {
        let p = get_genre_profile(Genre::Thriller);
        assert_eq!(p.name, "Thriller");
        assert!(p.tension_variance_min >= 0.5);
        assert!(p.requires_resolution);
        assert!(!p.accepts_nonlinear);
    }

    #[test]
    fn test_get_genre_profile_literary() {
        let p = get_genre_profile(Genre::Literary);
        assert_eq!(p.target_fkgl_min, 8.0);
        assert_eq!(p.target_fkgl_max, 16.0);
        assert!(p.accepts_nonlinear);
        assert!(!p.requires_resolution);
    }

    // --- SBERT path tests ---

    /// When no model is present (test environment), detect_genre_sbert returns None.
    #[test]
    fn test_detect_genre_sbert_absent_returns_none() {
        let result = detect_genre_sbert("love passion romance desire kiss");
        // In test env without model files, all embeddings are empty → None
        assert!(
            result.is_none(),
            "Expected None when SBERT model is absent, got {:?}",
            result
        );
    }

    /// detect_genre falls back to heuristic when SBERT is unavailable.
    #[test]
    fn test_detect_genre_uses_heuristic_fallback_when_no_model() {
        // With no model, detect_genre must still return a sensible result via heuristic.
        let themes = vec!["love".into(), "passion".into(), "desire".into()];
        let tension = vec![0.3, 0.35, 0.32, 0.28, 0.33];
        let g = detect_genre(themes, tension, 0.50, 7.0);
        // Heuristic should pick Romance for these inputs
        assert_eq!(g, Genre::Romance);
    }

    // --- Heuristic fallback tests (exercise detect_genre_heuristic directly) ---

    #[test]
    fn test_detect_genre_romance() {
        let themes = vec!["love".into(), "passion".into(), "desire".into()];
        let tension = vec![0.3, 0.35, 0.32, 0.28, 0.33];
        let g = detect_genre_heuristic(&themes, &tension, 0.50, 7.0);
        assert_eq!(g, Genre::Romance);
    }

    #[test]
    fn test_detect_genre_thriller() {
        let themes = vec![
            "death".into(),
            "murder".into(),
            "chase".into(),
            "danger".into(),
        ];
        let tension = vec![0.2, 0.8, 0.3, 0.9, 0.1, 0.95];
        let g = detect_genre_heuristic(&themes, &tension, 0.30, 7.0);
        assert_eq!(g, Genre::Thriller);
    }

    #[test]
    fn test_detect_genre_literary() {
        let themes = vec!["memory".into(), "solitude".into(), "time".into()];
        let tension = vec![0.4, 0.42, 0.38, 0.41, 0.39];
        let g = detect_genre_heuristic(&themes, &tension, 0.20, 12.0);
        assert_eq!(g, Genre::Literary);
    }

    #[test]
    fn test_detect_genre_ya() {
        let themes = vec![
            "school".into(),
            "friend".into(),
            "identity".into(),
            "belonging".into(),
        ];
        let tension = vec![0.3, 0.35, 0.4, 0.35, 0.3];
        let g = detect_genre_heuristic(&themes, &tension, 0.45, 5.5);
        assert_eq!(g, Genre::YoungAdult);
    }

    #[test]
    fn test_detect_genre_fallback_general() {
        let themes: Vec<String> = vec![];
        let tension = vec![0.5, 0.5, 0.5];
        let g = detect_genre_heuristic(&themes, &tension, 0.30, 9.0);
        assert!(
            matches!(g, Genre::General | Genre::Literary),
            "Expected General or Literary for ambiguous inputs, got {:?}",
            g
        );
    }

    #[test]
    fn test_detect_genre_smart_falls_back_gracefully() {
        // No SBERT model in test env; must fall back to heuristic without panic.
        let themes = vec!["love".to_string(), "romance".to_string()];
        let tension = vec![0.2, 0.3, 0.4];
        let g = detect_genre_smart("She loved him.", themes.clone(), tension.clone(), 0.45, 7.0);
        // Result is a valid Genre variant (not a panic).
        let _ = get_genre_profile(g);
    }

    #[test]
    fn test_detect_genre_smart_matches_fallback_when_no_model() {
        let themes = vec!["murder".to_string(), "detective".to_string()];
        let tension = vec![0.5, 0.7, 0.6];
        let smart = detect_genre_smart(
            "The detective examined the clues.",
            themes.clone(),
            tension.clone(),
            0.35,
            8.0,
        );
        let heuristic = detect_genre(themes, tension, 0.35, 8.0);
        assert_eq!(
            smart, heuristic,
            "Without a model, smart must match the heuristic"
        );
    }

    #[test]
    fn theme_match_is_word_stem_not_midword() {
        // "teen" must not fire inside "canteen"; it does count for "teenager".
        assert_eq!(count_theme_matches(&["canteen".to_string()], &["teen"]), 0);
        assert_eq!(count_theme_matches(&["teenager".to_string()], &["teen"]), 1);
    }

    #[test]
    fn test_genre_from_label() {
        // Canonical labels and aliases, case-insensitive and whitespace-tolerant.
        assert_eq!(genre_from_label("Thriller"), Some(Genre::Thriller));
        assert_eq!(genre_from_label("suspense"), Some(Genre::Thriller));
        assert_eq!(genre_from_label("  Sci-Fi  "), Some(Genre::SciFi));
        assert_eq!(genre_from_label("science fiction"), Some(Genre::SciFi));
        assert_eq!(genre_from_label("LITERARY FICTION"), Some(Genre::Literary));
        assert_eq!(genre_from_label("young adult"), Some(Genre::YoungAdult));
        assert_eq!(genre_from_label("children's"), Some(Genre::ChildrensBook));
        // Unknown labels fall through to None so the caller can detect_genre.
        assert_eq!(genre_from_label("western"), None);
        assert_eq!(genre_from_label(""), None);
    }

    #[test]
    fn test_evaluate_in_range_no_findings() {
        let findings = evaluate_against_genre(Genre::Thriller, 7.0, 15.0, 0.35, 0.6, 1, 85_000);
        assert!(
            findings.is_empty(),
            "Expected no findings for in-range values: {:?}",
            findings
        );
    }

    #[test]
    fn test_evaluate_fkgl_too_high_for_ya() {
        let findings = evaluate_against_genre(Genre::YoungAdult, 12.0, 14.0, 0.40, 0.30, 2, 65_000);
        let fkgl_finding = findings.iter().find(|f| f.metric == "fkgl");
        assert!(fkgl_finding.is_some(), "Should flag high FKGL for YA");
        assert_eq!(fkgl_finding.unwrap().severity, "concern");
    }

    #[test]
    fn test_evaluate_low_tension_thriller() {
        let findings = evaluate_against_genre(Genre::Thriller, 7.0, 15.0, 0.35, 0.1, 1, 85_000);
        let tension_finding = findings.iter().find(|f| f.metric == "tension_variance");
        assert!(
            tension_finding.is_some(),
            "Should flag low tension for Thriller"
        );
    }

    #[test]
    fn test_evaluate_word_count_too_short() {
        let findings = evaluate_against_genre(Genre::Fantasy, 10.0, 20.0, 0.30, 0.4, 2, 40_000);
        let wc_finding = findings.iter().find(|f| f.metric == "word_count");
        assert!(
            wc_finding.is_some(),
            "Should flag short word count for Fantasy"
        );
        let sev = &wc_finding.unwrap().severity;
        assert!(
            sev == "concern" || sev == "warning",
            "unexpected severity: {}",
            sev
        );
    }

    #[test]
    fn test_evaluate_whiplash_over_tolerance() {
        let findings =
            evaluate_against_genre(Genre::ChildrensBook, 3.0, 10.0, 0.50, 0.1, 5, 30_000);
        let wh = findings.iter().find(|f| f.metric == "tonal_whiplash");
        assert!(
            wh.is_some(),
            "Should flag excessive whiplash for Children's Book"
        );
    }

    #[test]
    fn test_tension_variance_calculation() {
        let tv = tension_variance(&[0.0, 1.0]);
        assert!((tv - 0.5).abs() < 0.001);

        assert_eq!(tension_variance(&[]), 0.0);
        assert_eq!(tension_variance(&[0.5]), 0.0);
    }
}
