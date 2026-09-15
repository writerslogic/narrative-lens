use std::collections::HashSet;
use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::craft::lexicon::WordMatcher;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NarrativePromise {
    pub id: usize,
    pub description: String,
    pub scene: usize,
    pub promise_type: PromiseType,
    pub explicitness: f64,
    pub status: PayoffStatus,
    pub payoff_scene: Option<usize>,
    pub payoff_quality: Option<f64>,
    pub investment: f64,
    pub urgency: f64,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub weight: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PromiseType {
    OpeningHook,
    GenreContract(String),
    CharacterIntroduction(String),
    ChekhovsGun(String),
    Foreshadowing(String),
    RelationshipSetup(String, String),
    ThematicQuestion(String),
    StructuralBeat(String),
    ExplicitQuestion(String),
    TimedThreat(String),
    RecurringMotif(String),
    CharacterGoal(String, String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PayoffStatus {
    Outstanding,
    Fulfilled { quality: f64 },
    PartiallyFulfilled { completeness: f64 },
    Broken,
    Subverted { satisfaction: f64 },
    Overpaid { excess: f64 },
    Deferred { new_deadline: Option<usize> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UnearnedPayoff {
    pub description: String,
    pub scene: usize,
    pub magnitude: f64,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CascadePayoff {
    pub scene: usize,
    pub promises_resolved: Vec<usize>,
    pub elegance: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GratificationDelay {
    pub promise_id: usize,
    pub delay_scenes: usize,
    pub delay_words: usize,
    pub within_genre_norm: bool,
    pub tension_maintained: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActLedger {
    pub act: u32,
    pub promises_made: usize,
    pub promises_fulfilled: usize,
    pub net_balance: i32,
    pub outstanding_weight: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContractNote {
    pub promise_id: usize,
    pub note_type: String,
    pub description: String,
    pub urgency: f64,
    pub suggestion: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromisePayoffResult {
    pub promises: Vec<NarrativePromise>,
    pub unearned_payoffs: Vec<UnearnedPayoff>,
    pub cascade_payoffs: Vec<CascadePayoff>,
    pub fulfillment_ratio: f64,
    pub broken_count: usize,
    pub unearned_count: usize,
    pub weighted_health: f64,
    pub act_breakdown: Vec<ActLedger>,
    pub investment_curve: Vec<f64>,
    pub max_nesting_depth: usize,
    pub gratification_delays: Vec<GratificationDelay>,
    pub overpaid_moments: Vec<(usize, f64)>,
    pub contract_notes: Vec<ContractNote>,
    pub reader_trust_assessment: String,
}

// ---------------------------------------------------------------------------
// Genre conventions
// ---------------------------------------------------------------------------

/// Returns (convention_description, weight) pairs for the given genre.
fn genre_conventions(genre: &str) -> Vec<(String, f64)> {
    let g = genre.to_lowercase();
    match g.as_str() {
        "romance" => vec![
            ("Happily-ever-after or happy-for-now ending".into(), 1.0),
            ("Meet-cute or initial encounter between leads".into(), 0.9),
            ("Romantic tension and obstacles".into(), 0.8),
            ("Declaration of love".into(), 0.85),
            ("Black moment / breakup before resolution".into(), 0.7),
        ],
        "mystery" => vec![
            ("Solution to the central mystery".into(), 1.0),
            ("Introduction of clues".into(), 0.9),
            ("Red herring misdirection".into(), 0.7),
            ("Reveal of the culprit".into(), 0.95),
            ("Detective reasoning / deduction scene".into(), 0.8),
        ],
        "thriller" => vec![
            ("Resolution of the central threat".into(), 1.0),
            ("Ticking clock or deadline pressure".into(), 0.9),
            ("Escalating stakes".into(), 0.85),
            ("Protagonist faces antagonist directly".into(), 0.9),
            ("Twist or revelation".into(), 0.75),
        ],
        "fantasy" => vec![
            ("Resolution of the main quest or conflict".into(), 1.0),
            ("Magic system demonstration".into(), 0.8),
            ("World-building payoff".into(), 0.7),
            ("Mentor figure guidance".into(), 0.6),
            ("Final confrontation with dark force".into(), 0.9),
        ],
        "scifi" | "sci-fi" | "science fiction" => vec![
            ("Resolution of speculative premise".into(), 1.0),
            ("Consequences of technology explored".into(), 0.85),
            ("World-building payoff".into(), 0.7),
            ("Thematic question answered".into(), 0.8),
        ],
        "horror" => vec![
            ("Confrontation with the source of horror".into(), 1.0),
            ("Escalating dread and tension".into(), 0.9),
            ("Explanation or revelation of the threat".into(), 0.8),
            ("Survival or doom resolution".into(), 0.85),
        ],
        "literary" | "literary fiction" => vec![
            ("Thematic resolution or illumination".into(), 0.9),
            ("Character transformation arc".into(), 1.0),
            ("Emotional catharsis".into(), 0.85),
        ],
        "historical" | "historical fiction" => vec![
            ("Resolution of the personal story".into(), 1.0),
            ("Historical event payoff".into(), 0.8),
            ("Period-specific conflict resolved".into(), 0.75),
        ],
        _ => vec![
            ("Resolution of central conflict".into(), 1.0),
            ("Character arc completion".into(), 0.85),
            ("Thematic coherence".into(), 0.7),
        ],
    }
}

// ---------------------------------------------------------------------------
// Detection helpers
// ---------------------------------------------------------------------------

/// Detect objects described at unusual length (potential Chekhov's guns).
/// Returns (description_snippet, approximate_word_position_in_scene).
fn detect_chekhov_guns(text: &str) -> Vec<(String, usize)> {
    let mut results = Vec::new();

    // Pattern: a noun phrase followed by an extended description clause.
    // We look for "the [adj]* [noun]" followed by a relative clause or long appositive.
    let re = Regex::new(
        r"(?i)\b(the\s+(?:\w+\s+){0,3}\w+)\s*(?:,\s*(?:which|that|whose)[^.]{20,}[.]|—[^—]{20,}—)",
    )
    .unwrap();

    for caps in re.captures_iter(text) {
        let Some(full) = caps.get(0) else {
            continue;
        };
        let noun_phrase = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let word_pos = text[..full.start()].split_whitespace().count();
        // Only count if the full match is lengthy (indicating unusual detail)
        if full.as_str().split_whitespace().count() >= 15 {
            results.push((noun_phrase.to_string(), word_pos));
        }
    }

    // Also detect objects with conspicuous emphasis verbs nearby
    let emphasis_re = Regex::new(
        r"(?i)(?:noticed|studied|examined|fixated on|couldn't (?:stop|help) (?:looking at|staring at))\s+(the\s+\w+(?:\s+\w+){0,2})"
    ).unwrap();

    for caps in emphasis_re.captures_iter(text) {
        let obj = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let Some(full) = caps.get(0) else {
            continue;
        };
        let word_pos = text[..full.start()].split_whitespace().count();
        results.push((obj.to_string(), word_pos));
    }

    results
}

/// Detect questions in non-dialogue narration that promise answers.
fn narrator_question_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"([^""\u{201C}\u{201D}]{10,}\?)"#).expect("narrator question regex")
    })
}

fn detect_narrator_questions(text: &str) -> Vec<String> {
    let mut results = Vec::new();

    // Split into sentences, skip dialogue lines
    for line in text.lines() {
        let trimmed = line.trim();
        // Skip dialogue (starts with quote or contains quote attribution)
        if trimmed.starts_with('"')
            || trimmed.starts_with('\u{201C}')
            || trimmed.starts_with('\u{2018}')
        {
            continue;
        }

        // Find question marks in non-dialogue text
        for caps in narrator_question_re().captures_iter(trimmed) {
            let q = caps.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            // Filter out very short or rhetorical questions
            if q.split_whitespace().count() >= 5 {
                results.push(q.to_string());
            }
        }
    }

    results
}

/// Compute the act breakdown for promises across three acts.
fn compute_act_breakdown(promises: &[NarrativePromise], total_scenes: usize) -> Vec<ActLedger> {
    if total_scenes == 0 {
        return vec![];
    }

    // Three-act structure: 25% / 50% / 25%
    let act1_end = (total_scenes as f64 * 0.25).ceil() as usize;
    let act2_end = (total_scenes as f64 * 0.75).ceil() as usize;

    let mut ledgers = vec![
        ActLedger {
            act: 1,
            promises_made: 0,
            promises_fulfilled: 0,
            net_balance: 0,
            outstanding_weight: 0.0,
        },
        ActLedger {
            act: 2,
            promises_made: 0,
            promises_fulfilled: 0,
            net_balance: 0,
            outstanding_weight: 0.0,
        },
        ActLedger {
            act: 3,
            promises_made: 0,
            promises_fulfilled: 0,
            net_balance: 0,
            outstanding_weight: 0.0,
        },
    ];

    for p in promises {
        let act_idx = if p.scene < act1_end {
            0
        } else if p.scene < act2_end {
            1
        } else {
            2
        };

        ledgers[act_idx].promises_made += 1;

        if let Some(payoff_scene) = p.payoff_scene {
            let payoff_act = if payoff_scene < act1_end {
                0
            } else if payoff_scene < act2_end {
                1
            } else {
                2
            };
            ledgers[payoff_act].promises_fulfilled += 1;
        }

        // Track outstanding weight
        match &p.status {
            PayoffStatus::Outstanding | PayoffStatus::Deferred { .. } => {
                ledgers[act_idx].outstanding_weight += p.weight;
            }
            _ => {}
        }
    }

    for ledger in &mut ledgers {
        ledger.net_balance = ledger.promises_made as i32 - ledger.promises_fulfilled as i32;
    }

    ledgers
}

/// Determine genre-appropriate delay norms (in scene-fraction of total).
fn genre_delay_norm(genre: &str) -> (f64, f64) {
    let g = genre.to_lowercase();
    match g.as_str() {
        "thriller" => (0.05, 0.30),
        "mystery" => (0.10, 0.80),
        "romance" => (0.10, 0.90),
        "horror" => (0.05, 0.50),
        "literary" | "literary fiction" => (0.15, 0.95),
        _ => (0.10, 0.70),
    }
}

/// Detect climactic / high-tension sentences that might represent unearned payoffs.
fn detect_climactic_moments(text: &str) -> Vec<(usize, String)> {
    let mut results = Vec::new();

    let climax_markers = [
        "finally",
        "at last",
        "the truth",
        "revealed",
        "gasped",
        "screamed",
        "everything changed",
        "it was over",
        "triumphant",
        "collapsed",
        "shattered",
        "the answer",
        "in that moment",
        "suddenly understood",
        "epiphany",
    ];

    let text_lower = text.to_lowercase();
    for (i, sentence) in text.split('.').enumerate() {
        let sent_lower = sentence.to_lowercase();
        let marker_count = climax_markers
            .iter()
            .filter(|m| sent_lower.contains(*m))
            .count();
        if marker_count >= 2 {
            let snippet = sentence.trim();
            if snippet.len() > 10 {
                let word_pos = text_lower[..text_lower
                    .find(snippet.to_lowercase().as_str())
                    .unwrap_or(0)]
                    .split_whitespace()
                    .count();
                let _ = word_pos;
                results.push((i, snippet.chars().take(80).collect()));
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Main algorithm
// ---------------------------------------------------------------------------

/// Build the promise-payoff ledger from analysis data.
///
/// Parameters:
/// - scenes: the text of each scene
/// - scene_word_counts: word count per scene (for investment computation)
/// - genre: detected genre name
/// - foreshadowing_items: items from foreshadow module (description, setup_scene, resolved bool)
/// - character_first_appearances: (character_name, first_scene_index)
/// - structure_template: which structure template ("three_act", "hero_journey", etc.)
/// - total_scenes: total number of scenes
pub fn build_promise_payoff_ledger(
    scenes: &[&str],
    scene_word_counts: &[usize],
    genre: &str,
    foreshadowing_items: &[(String, usize, bool)],
    character_first_appearances: &[(String, usize)],
    structure_template: &str,
    total_scenes: usize,
) -> PromisePayoffResult {
    let mut promises: Vec<NarrativePromise> = Vec::new();
    let mut next_id: usize = 0;

    // --- 1. Seed genre contracts ---
    let conventions = genre_conventions(genre);
    for (desc, weight) in &conventions {
        promises.push(NarrativePromise {
            id: next_id,
            description: desc.clone(),
            scene: 0,
            promise_type: PromiseType::GenreContract(genre.to_string()),
            explicitness: 0.3, // implicit genre contract
            status: PayoffStatus::Outstanding,
            payoff_scene: None,
            payoff_quality: None,
            investment: 0.0,
            urgency: 0.5,
            parent: None,
            children: Vec::new(),
            weight: *weight,
        });
        next_id += 1;
    }

    // --- 2. Seed from foreshadowing items ---
    for (desc, setup_scene, resolved) in foreshadowing_items {
        let status = if *resolved {
            PayoffStatus::Fulfilled { quality: 1.0 }
        } else {
            PayoffStatus::Outstanding
        };
        promises.push(NarrativePromise {
            id: next_id,
            description: desc.clone(),
            scene: *setup_scene,
            promise_type: PromiseType::Foreshadowing(desc.clone()),
            explicitness: 0.7,
            status,
            payoff_scene: if *resolved {
                Some(total_scenes.saturating_sub(1))
            } else {
                None
            },
            payoff_quality: if *resolved { Some(1.0) } else { None },
            investment: 0.0,
            urgency: 0.6,
            parent: None,
            children: Vec::new(),
            weight: 0.8,
        });
        next_id += 1;
    }

    // --- 3. Seed from character introductions ---
    for (name, first_scene) in character_first_appearances {
        promises.push(NarrativePromise {
            id: next_id,
            description: format!(
                "Character '{}' introduced; promises narrative relevance",
                name
            ),
            scene: *first_scene,
            promise_type: PromiseType::CharacterIntroduction(name.clone()),
            explicitness: 0.5,
            status: PayoffStatus::Outstanding,
            payoff_scene: None,
            payoff_quality: None,
            investment: 0.0,
            urgency: 0.4,
            parent: None,
            children: Vec::new(),
            weight: 0.6,
        });
        next_id += 1;
    }

    // --- 4. Scan for Chekhov's guns ---
    for (scene_idx, scene_text) in scenes.iter().enumerate() {
        let guns = detect_chekhov_guns(scene_text);
        for (obj_desc, _word_pos) in guns {
            promises.push(NarrativePromise {
                id: next_id,
                description: format!(
                    "Object '{}' described in detail; promises later significance",
                    obj_desc
                ),
                scene: scene_idx,
                promise_type: PromiseType::ChekhovsGun(obj_desc),
                explicitness: 0.6,
                status: PayoffStatus::Outstanding,
                payoff_scene: None,
                payoff_quality: None,
                investment: 0.0,
                urgency: 0.5,
                parent: None,
                children: Vec::new(),
                weight: 0.75,
            });
            next_id += 1;
        }
    }

    // --- 5. Scan for explicit questions ---
    for (scene_idx, scene_text) in scenes.iter().enumerate() {
        let questions = detect_narrator_questions(scene_text);
        for q in questions {
            promises.push(NarrativePromise {
                id: next_id,
                description: format!("Narrative question: {}", q),
                scene: scene_idx,
                promise_type: PromiseType::ExplicitQuestion(q),
                explicitness: 0.9,
                status: PayoffStatus::Outstanding,
                payoff_scene: None,
                payoff_quality: None,
                investment: 0.0,
                urgency: 0.7,
                parent: None,
                children: Vec::new(),
                weight: 0.85,
            });
            next_id += 1;
        }
    }

    // --- 6. Seed opening hook ---
    if !scenes.is_empty() {
        promises.push(NarrativePromise {
            id: next_id,
            description: "Opening hook promises an engaging story worth reading".into(),
            scene: 0,
            promise_type: PromiseType::OpeningHook,
            explicitness: 0.4,
            status: PayoffStatus::Outstanding,
            payoff_scene: None,
            payoff_quality: None,
            investment: 0.0,
            urgency: 0.8,
            parent: None,
            children: Vec::new(),
            weight: 0.9,
        });
        next_id += 1;
    }

    // --- 7. Seed structural beats ---
    let beat_names: Vec<&str> = match structure_template {
        "hero_journey" => vec![
            "Call to Adventure",
            "Crossing the Threshold",
            "Ordeal",
            "Return with Elixir",
        ],
        "save_the_cat" => vec![
            "Opening Image",
            "Catalyst",
            "Midpoint",
            "All Is Lost",
            "Final Image",
        ],
        _ => vec![
            "Inciting Incident",
            "Midpoint Reversal",
            "Climax",
            "Resolution",
        ],
    };
    for beat_name in &beat_names {
        promises.push(NarrativePromise {
            id: next_id,
            description: format!("Structural beat '{}' expected", beat_name),
            scene: 0,
            promise_type: PromiseType::StructuralBeat(beat_name.to_string()),
            explicitness: 0.2,
            status: PayoffStatus::Outstanding,
            payoff_scene: None,
            payoff_quality: None,
            investment: 0.0,
            urgency: 0.5,
            parent: None,
            children: Vec::new(),
            weight: 0.65,
        });
        next_id += 1;
    }
    let _ = next_id; // suppress unused warning

    // --- 9. Detect cascade payoffs ---
    let mut cascade_payoffs: Vec<CascadePayoff> = Vec::new();
    {
        let mut scene_resolutions: Vec<Vec<usize>> = vec![Vec::new(); total_scenes];
        for p in &promises {
            if let Some(ps) = p.payoff_scene
                && ps < total_scenes {
                    scene_resolutions[ps].push(p.id);
                }
        }
        for (scene_idx, resolved_ids) in scene_resolutions.iter().enumerate() {
            if resolved_ids.len() >= 2 {
                let elegance = (resolved_ids.len() as f64 / 5.0).min(1.0);
                cascade_payoffs.push(CascadePayoff {
                    scene: scene_idx,
                    promises_resolved: resolved_ids.clone(),
                    elegance,
                });
            }
        }
    }

    // --- 10. Detect unearned payoffs ---
    let mut unearned_payoffs: Vec<UnearnedPayoff> = Vec::new();
    for (scene_idx, scene_text) in scenes.iter().enumerate() {
        let climactic = detect_climactic_moments(scene_text);
        if !climactic.is_empty() {
            // Check if any promise set up this moment
            let has_setup = promises
                .iter()
                .any(|p| p.payoff_scene == Some(scene_idx) && p.scene < scene_idx);
            if !has_setup {
                for (_sent_idx, snippet) in &climactic {
                    unearned_payoffs.push(UnearnedPayoff {
                        description: snippet.clone(),
                        scene: scene_idx,
                        magnitude: 0.7,
                        reason: "Climactic moment without prior narrative setup".into(),
                    });
                }
            }
        }
    }

    // --- 11. Compute investment curve ---
    let mut investment_curve: Vec<f64> = Vec::with_capacity(total_scenes);
    {
        let mut cumulative_investment = 0.0;
        for scene_idx in 0..total_scenes {
            // Outstanding promises accumulate investment each scene
            for p in &promises {
                if p.scene <= scene_idx {
                    let is_outstanding = matches!(p.status, PayoffStatus::Outstanding)
                        || matches!(
                            p.payoff_scene,
                            Some(ps) if ps > scene_idx
                        );
                    if is_outstanding {
                        cumulative_investment += p.weight * 0.1;
                    }
                }
            }
            // Fulfilled promises release investment
            for p in &promises {
                if p.payoff_scene == Some(scene_idx) {
                    cumulative_investment -= p.weight * 0.5;
                }
            }
            cumulative_investment = cumulative_investment.max(0.0);
            investment_curve.push(cumulative_investment);
        }
    }

    // Update investment field on each promise
    for p in &mut promises {
        let delay = p
            .payoff_scene
            .unwrap_or(total_scenes.saturating_sub(1))
            .saturating_sub(p.scene);
        let word_delay: usize = scene_word_counts
            .get(p.scene..p.payoff_scene.unwrap_or(total_scenes))
            .map(|s| s.iter().sum())
            .unwrap_or(0);
        p.investment = p.weight * delay as f64 * (1.0 + word_delay as f64 / 10000.0);
    }

    // --- 12. Compute gratification delays ---
    let (norm_min, norm_max) = genre_delay_norm(genre);
    let mut gratification_delays: Vec<GratificationDelay> = Vec::new();
    for p in &promises {
        if let Some(payoff_scene) = p.payoff_scene {
            let delay_scenes = payoff_scene.saturating_sub(p.scene);
            let delay_words: usize = scene_word_counts
                .get(p.scene..payoff_scene)
                .map(|s| s.iter().sum())
                .unwrap_or(0);
            let delay_frac = if total_scenes > 0 {
                delay_scenes as f64 / total_scenes as f64
            } else {
                0.0
            };
            let within_norm = delay_frac >= norm_min && delay_frac <= norm_max;

            // Tension maintained = proportion of intervening scenes that reference promise keywords
            let desc_lower = p.description.to_lowercase();
            let keywords: Vec<&str> = desc_lower
                .split_whitespace()
                .filter(|w| w.len() > 4)
                .collect();
            // One stem matcher for the promise keywords, reused across scenes:
            // inflection-tolerant ("uncover" -> "uncovered") without firing a
            // keyword mid-word as str::contains did.
            let key_matcher = WordMatcher::new(&keywords);
            let mut tension_hits = 0usize;
            let intervening = p.scene + 1..payoff_scene;
            let intervening_count = intervening.len().max(1);
            for si in intervening {
                if si < scenes.len() {
                    let sl = scenes[si].to_lowercase();
                    if key_matcher.is_prefix_match(&sl) {
                        tension_hits += 1;
                    }
                }
            }
            let tension_maintained = tension_hits as f64 / intervening_count as f64;

            gratification_delays.push(GratificationDelay {
                promise_id: p.id,
                delay_scenes,
                delay_words,
                within_genre_norm: within_norm,
                tension_maintained,
            });
        }
    }

    // --- 13. Detect overpaid moments ---
    let mut overpaid_moments: Vec<(usize, f64)> = Vec::new();
    for p in &promises {
        if let PayoffStatus::Overpaid { excess } = &p.status
            && let Some(ps) = p.payoff_scene {
                overpaid_moments.push((ps, *excess));
            }
    }
    // Also flag promises with disproportionately large payoff scenes relative to setup
    for p in &promises {
        if let (Some(ps), PayoffStatus::Fulfilled { quality }) = (p.payoff_scene, &p.status)
            && *quality > 0.9 && p.weight < 0.4 {
                overpaid_moments.push((ps, quality - p.weight));
            }
    }

    // --- 14. Mark broken promises ---
    // Promises still outstanding past 90% of manuscript are likely broken
    let cutoff = (total_scenes as f64 * 0.9).ceil() as usize;
    for p in &mut promises {
        if matches!(p.status, PayoffStatus::Outstanding) && p.scene < cutoff && p.weight >= 0.7 {
            p.status = PayoffStatus::Broken;
        }
    }

    // --- 15. Compute summary statistics ---
    let total_promises = promises.len();
    let fulfilled = promises
        .iter()
        .filter(|p| {
            matches!(
                p.status,
                PayoffStatus::Fulfilled { .. }
                    | PayoffStatus::PartiallyFulfilled { .. }
                    | PayoffStatus::Subverted { .. }
            )
        })
        .count();
    let broken_count = promises
        .iter()
        .filter(|p| matches!(p.status, PayoffStatus::Broken))
        .count();
    let fulfillment_ratio = if total_promises > 0 {
        fulfilled as f64 / total_promises as f64
    } else {
        1.0
    };

    // Weighted health: accounts for promise weight in fulfillment
    let total_weight: f64 = promises.iter().map(|p| p.weight).sum();
    let fulfilled_weight: f64 = promises
        .iter()
        .filter(|p| {
            matches!(
                p.status,
                PayoffStatus::Fulfilled { .. }
                    | PayoffStatus::PartiallyFulfilled { .. }
                    | PayoffStatus::Subverted { .. }
            )
        })
        .map(|p| p.weight)
        .sum();
    let weighted_health = if total_weight > 0.0 {
        fulfilled_weight / total_weight
    } else {
        1.0
    };

    // Max nesting depth (parent-child chains)
    let max_nesting_depth = compute_max_nesting(&promises);

    let act_breakdown = compute_act_breakdown(&promises, total_scenes);
    let unearned_count = unearned_payoffs.len();

    // --- 16. Generate contract notes ---
    let mut contract_notes: Vec<ContractNote> = Vec::new();

    // Long-outstanding promises (>15 scenes unpaid)
    let genre_patience: usize = match genre.to_lowercase().as_str() {
        "thriller" => 10,
        "mystery" => 25,
        "romance" => 20,
        "literary" | "literary fiction" => 30,
        _ => 15,
    };
    for p in &promises {
        if matches!(
            p.status,
            PayoffStatus::Outstanding | PayoffStatus::Deferred { .. }
        ) {
            let delay = total_scenes.saturating_sub(p.scene);
            if delay > 15 {
                let urgency = (delay as f64 / genre_patience as f64).min(1.0);
                contract_notes.push(ContractNote {
                    promise_id: p.id,
                    note_type: "approaching_deadline".to_string(),
                    description: format!(
                        "You promised '{}' in scene {} and it's been {} scenes. The reader's patience has a limit; for {}, typically {} scenes.",
                        p.description, p.scene, delay, genre, genre_patience
                    ),
                    urgency,
                    suggestion: Some("Either pay this off soon, or reinforce it (remind the reader it matters, raise the stakes of the answer).".to_string()),
                });
            }
        }
    }

    // Broken promises
    for p in &promises {
        if matches!(p.status, PayoffStatus::Broken) {
            contract_notes.push(ContractNote {
                promise_id: p.id,
                note_type: "broken_trust".to_string(),
                description: format!(
                    "The reader was promised '{}' and never received it. This breaks trust.",
                    p.description
                ),
                urgency: 0.9,
                suggestion: Some("Either: fulfill it (even late), explicitly subvert it (character acknowledges it won't happen), or retroactively show it was never really promised (tricky, requires rewrite).".to_string()),
            });
        }
    }

    // Cascade payoffs (elegance > 0.7)
    for cp in &cascade_payoffs {
        if cp.elegance > 0.7 {
            contract_notes.push(ContractNote {
                promise_id: cp.promises_resolved.first().copied().unwrap_or(0),
                note_type: "well_managed".to_string(),
                description: format!(
                    "Scene {} simultaneously pays off {} promises. This is MASTERFUL plotting; the reader feels everything click into place. Protect this scene in revision.",
                    cp.scene, cp.promises_resolved.len()
                ),
                urgency: 0.0,
                suggestion: None,
            });
        }
    }

    // Overpaid moments
    for &(scene_idx, excess) in &overpaid_moments {
        contract_notes.push(ContractNote {
            promise_id: promises.iter().find(|p| p.payoff_scene == Some(scene_idx)).map(|p| p.id).unwrap_or(0),
            note_type: "overpaid".to_string(),
            description: format!(
                "The payoff in scene {} exceeds what the setup warranted (excess: {:.2}). A minor mystery resolved with a dramatic revelation feels disproportionate.",
                scene_idx, excess
            ),
            urgency: 0.4,
            suggestion: Some("Either increase the earlier promise's weight (more setup, more stakes) or reduce the payoff's drama.".to_string()),
        });
    }

    // --- 17. Reader trust assessment ---
    let trust_level = if weighted_health > 0.7 {
        "high"
    } else if weighted_health > 0.4 {
        "moderate"
    } else {
        "low"
    };
    let feeling = if weighted_health > 0.7 {
        "will feel the story delivers on its promises"
    } else if weighted_health > 0.4 {
        "may feel some threads were left dangling"
    } else {
        "will feel cheated by unresolved setups"
    };
    let reader_trust_assessment = format!(
        "The reader's contract trust is {}. {} promises fulfilled, {} broken. The reader {}.",
        trust_level, fulfilled, broken_count, feeling
    );

    PromisePayoffResult {
        promises,
        unearned_payoffs,
        cascade_payoffs,
        fulfillment_ratio,
        broken_count,
        unearned_count,
        weighted_health,
        act_breakdown,
        investment_curve,
        max_nesting_depth,
        gratification_delays,
        overpaid_moments,
        contract_notes,
        reader_trust_assessment,
    }
}

/// Compute the maximum depth of parent-child nesting among promises.
fn compute_max_nesting(promises: &[NarrativePromise]) -> usize {
    let mut max_depth = 0usize;
    for p in promises {
        let mut depth = 0usize;
        let mut current = p.parent;
        let mut visited = HashSet::new();
        while let Some(pid) = current {
            if visited.contains(&pid) {
                break;
            }
            visited.insert(pid);
            depth += 1;
            current = promises
                .iter()
                .find(|pp| pp.id == pid)
                .and_then(|pp| pp.parent);
        }
        max_depth = max_depth.max(depth);
    }
    max_depth
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genre_conventions_returns_entries() {
        let convs = genre_conventions("romance");
        assert!(!convs.is_empty());
        assert!(convs[0].1 > 0.0);
    }

    #[test]
    fn test_genre_conventions_unknown_genre() {
        let convs = genre_conventions("cyberpunk");
        assert!(!convs.is_empty()); // falls back to general
    }

    #[test]
    fn test_detect_chekhov_guns_empty_input() {
        // Empty and keyword-free input must yield no matches without panicking
        // on the capture-group access in the iteration.
        assert!(detect_chekhov_guns("").is_empty());
        assert!(detect_chekhov_guns("A short plain sentence.").is_empty());
    }

    #[test]
    fn test_detect_narrator_questions() {
        let text = "He walked into the room. What could possibly have driven her to leave so suddenly? The door was open.";
        let qs = detect_narrator_questions(text);
        assert_eq!(qs.len(), 1);
        assert!(qs[0].contains("driven her to leave"));
    }

    #[test]
    fn test_detect_narrator_questions_skips_dialogue() {
        let text = "\"What do you think you're doing here?\" she asked.";
        let qs = detect_narrator_questions(text);
        assert!(qs.is_empty());
    }

    #[test]
    fn test_build_ledger_basic() {
        let scenes = vec![
            "The old silver key sat on the mantle, conspicuous and strange.",
            "She went about her day, thinking of nothing in particular.",
            "Finally she grabbed the silver key and unlocked the hidden door, revealed at last.",
        ];
        let word_counts: Vec<usize> = scenes
            .iter()
            .map(|s| s.split_whitespace().count())
            .collect();
        let foreshadowing = vec![("silver key on mantle".into(), 0usize, false)];
        let characters = vec![("Sarah".into(), 0usize)];

        let result = build_promise_payoff_ledger(
            &scenes,
            &word_counts,
            "mystery",
            &foreshadowing,
            &characters,
            "three_act",
            3,
        );

        assert!(!result.promises.is_empty());
        assert!(result.fulfillment_ratio >= 0.0);
        assert_eq!(result.act_breakdown.len(), 3);
        assert_eq!(result.investment_curve.len(), 3);
    }

    #[test]
    fn test_build_ledger_empty() {
        let result = build_promise_payoff_ledger(&[], &[], "", &[], &[], "three_act", 0);
        assert_eq!(result.broken_count, 0);
        assert_eq!(result.unearned_count, 0);
    }

    #[test]
    fn test_act_breakdown_distribution() {
        let promises = vec![NarrativePromise {
            id: 0,
            description: "test".into(),
            scene: 0,
            promise_type: PromiseType::OpeningHook,
            explicitness: 0.5,
            status: PayoffStatus::Fulfilled { quality: 0.8 },
            payoff_scene: Some(9),
            payoff_quality: Some(0.8),
            investment: 1.0,
            urgency: 0.5,
            parent: None,
            children: Vec::new(),
            weight: 0.9,
        }];
        let breakdown = compute_act_breakdown(&promises, 10);
        assert_eq!(breakdown.len(), 3);
        assert_eq!(breakdown[0].promises_made, 1); // scene 0 is in act 1
        assert_eq!(breakdown[2].promises_fulfilled, 1); // scene 9 is in act 3
    }

    #[test]
    fn test_detect_chekhov_guns_emphasis() {
        let text = "She noticed the old revolver sitting in the drawer, its barrel gleaming.";
        let guns = detect_chekhov_guns(text);
        assert!(!guns.is_empty());
    }
}
