use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NarrativeThread {
    MainPlot,
    Subplot(String),
    CharacterArc(String),
    Relationship(String, String),
    Mystery(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PossibleOutcome {
    pub description: String,
    pub probability: f64,
    pub realized: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NarrativeQuestion {
    pub id: usize,
    pub question: String,
    pub is_cdq: bool,
    pub thread: NarrativeThread,
    pub introduced_at: usize,
    pub resolved_at: Option<usize>,
    pub outcomes: Vec<PossibleOutcome>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntropyPoint {
    pub scene: usize,
    pub per_question_entropy: Vec<(usize, f64)>,
    pub composite_entropy: f64,
    pub entropy_rate: f64,
    pub entropy_acceleration: f64,
    pub information_gain: f64,
    pub conditional_entropies: Vec<(usize, usize, f64)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EntropyDiagnosis {
    WellPaced,
    Predictable {
        plateau_scene: usize,
    },
    Chaotic {
        spike_scenes: Vec<usize>,
    },
    Rushed {
        drop_scene: usize,
        drop_magnitude: f64,
    },
    Unresolved {
        final_entropy: f64,
        unresolved_questions: Vec<String>,
    },
    Mixed(Vec<Box<EntropyDiagnosis>>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PacingNote {
    pub scene_range: (usize, usize),
    pub issue: String,
    pub reader_experience: String,
    pub fix: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NarrativeEntropyResult {
    pub questions: Vec<NarrativeQuestion>,
    pub entropy_curve: Vec<EntropyPoint>,
    pub cdq_entropy_curve: Vec<f64>,
    pub entropy_rate_curve: Vec<f64>,
    pub redundant_scenes: Vec<(usize, String)>,
    pub subplot_integration: f64,
    pub pacing_score: f64,
    pub pacing_diagnosis: EntropyDiagnosis,
    pub complexity_curve: Vec<f64>,
    pub pacing_feedback: Vec<PacingNote>,
    pub overall_pacing_assessment: String,
}

/// Window of preceding scenes used as the novelty baseline. Windowed rather than
/// whole-book so later scenes are not washed out by an ever-growing mean.
const NOVELTY_WINDOW: usize = 5;

/// Per-scene novelty used when SBERT embeddings are unavailable, and for scene 0
/// before back-fill. The neutral midpoint of the [0, 1] novelty range.
const NEUTRAL_NOVELTY: f64 = 0.5;

/// A scene whose semantic novelty falls below this floor carries little new
/// information relative to its recent predecessors and is flagged as redundant.
const REDUNDANT_NOVELTY: f64 = 0.15;

/// Per-scene semantic novelty: `1 - cosine(scene, mean(prior window))`, clamped to
/// [0, 1]. High means the scene introduces new material; low means it retreads
/// recent content. Scene 0 is back-filled to the mean of the rest so the curve has
/// no spurious endpoint. Assumes every embedding is non-empty (caller guards).
fn compute_novelty_curve(embeddings: &[Vec<f64>]) -> Vec<f64> {
    let n = embeddings.len();
    let mut novelty = vec![NEUTRAL_NOVELTY; n];
    for i in 1..n {
        // A zero-norm (degenerate) scene vector yields cosine 0 -> novelty 1.0, a
        // fabricated "maximally novel" reading; leave it at the neutral default.
        if embeddings[i].iter().all(|&x| x == 0.0) {
            continue;
        }
        let dim = embeddings[i].len();
        let start = i.saturating_sub(NOVELTY_WINDOW);
        let mut context = vec![0.0_f64; dim];
        let mut count = 0usize;
        for prior in &embeddings[start..i] {
            if prior.len() == dim {
                for (c, v) in context.iter_mut().zip(prior) {
                    *c += *v;
                }
                count += 1;
            }
        }
        if count == 0 {
            continue;
        }
        for c in context.iter_mut() {
            *c /= count as f64;
        }
        let sim = crate::substrate::utils::cosine_similarity(&embeddings[i], &context);
        novelty[i] = (1.0 - sim).clamp(0.0, 1.0);
    }
    if n > 1 {
        let mean = novelty[1..].iter().sum::<f64>() / (n - 1) as f64;
        novelty[0] = mean;
    }
    novelty
}

/// Determine the CDQ based on genre, structure, and dominant theme.
fn derive_cdq(
    genre: &str,
    _structure_template: &str,
    dominant_theme: Option<&str>,
) -> (String, Vec<PossibleOutcome>) {
    let genre_lower = genre.to_lowercase();
    let (question, outcomes) =
        if genre_lower.contains("mystery") || genre_lower.contains("detective") {
            (
                "Will the truth be discovered?".to_string(),
                vec![
                    PossibleOutcome {
                        description: "Truth is fully revealed".to_string(),
                        probability: 0.5,
                        realized: false,
                    },
                    PossibleOutcome {
                        description: "Truth remains hidden".to_string(),
                        probability: 0.3,
                        realized: false,
                    },
                    PossibleOutcome {
                        description: "Partial truth emerges".to_string(),
                        probability: 0.2,
                        realized: false,
                    },
                ],
            )
        } else if genre_lower.contains("romance") {
            (
                "Will they end up together?".to_string(),
                vec![
                    PossibleOutcome {
                        description: "They end up together".to_string(),
                        probability: 0.7,
                        realized: false,
                    },
                    PossibleOutcome {
                        description: "They part ways".to_string(),
                        probability: 0.2,
                        realized: false,
                    },
                    PossibleOutcome {
                        description: "Ambiguous ending".to_string(),
                        probability: 0.1,
                        realized: false,
                    },
                ],
            )
        } else if genre_lower.contains("thriller")
            || genre_lower.contains("suspense")
            || genre_lower.contains("action")
        {
            (
                "Will the protagonist succeed?".to_string(),
                vec![
                    PossibleOutcome {
                        description: "Protagonist succeeds".to_string(),
                        probability: 0.5,
                        realized: false,
                    },
                    PossibleOutcome {
                        description: "Protagonist fails".to_string(),
                        probability: 0.3,
                        realized: false,
                    },
                    PossibleOutcome {
                        description: "Pyrrhic victory".to_string(),
                        probability: 0.2,
                        realized: false,
                    },
                ],
            )
        } else {
            // Literary/general: derive from dominant theme
            let theme_str = dominant_theme.unwrap_or("growth");
            (
                format!("Will the character achieve {}?", theme_str),
                vec![
                    PossibleOutcome {
                        description: format!("Character achieves {}", theme_str),
                        probability: 0.5,
                        realized: false,
                    },
                    PossibleOutcome {
                        description: format!("Character fails to achieve {}", theme_str),
                        probability: 0.3,
                        realized: false,
                    },
                    PossibleOutcome {
                        description: "Transformation occurs unexpectedly".to_string(),
                        probability: 0.2,
                        realized: false,
                    },
                ],
            )
        };
    (question, outcomes)
}

/// Compute the weight for a narrative question based on its thread type.
pub fn question_weight(q: &NarrativeQuestion) -> f64 {
    if q.is_cdq {
        2.0
    } else {
        match &q.thread {
            NarrativeThread::MainPlot => 2.0,
            NarrativeThread::Subplot(_) => 1.0,
            NarrativeThread::CharacterArc(_) => 0.5,
            NarrativeThread::Relationship(_, _) => 0.5,
            NarrativeThread::Mystery(_) => 1.0,
        }
    }
}

/// Compute narrative entropy across the manuscript.
///
/// Parameters:
/// - scenes: text of each scene
/// - themes: detected themes (name, relevance_score)
/// - structure_template: "three_act", etc.
/// - genre: detected genre
/// - character_arcs: (character, arc_type) detected character arcs
pub fn compute_narrative_entropy(
    scenes: &[&str],
    themes: &[(String, f64)],
    structure_template: &str,
    genre: &str,
    character_arcs: &[(String, String)],
) -> NarrativeEntropyResult {
    if scenes.is_empty() {
        return NarrativeEntropyResult {
            questions: vec![],
            entropy_curve: vec![],
            cdq_entropy_curve: vec![],
            entropy_rate_curve: vec![],
            redundant_scenes: vec![],
            subplot_integration: 0.0,
            pacing_score: 1.0,
            pacing_diagnosis: EntropyDiagnosis::WellPaced,
            complexity_curve: vec![],
            pacing_feedback: vec![],
            overall_pacing_assessment: String::new(),
        };
    }

    // Step 1: Identify Narrative Questions
    let dominant_theme = themes
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let dominant_theme_name = dominant_theme.map(|(name, _)| name.as_str());

    let mut questions: Vec<NarrativeQuestion> = Vec::new();
    let mut next_id = 0;

    // CDQ
    let (cdq_text, cdq_outcomes) = derive_cdq(genre, structure_template, dominant_theme_name);
    questions.push(NarrativeQuestion {
        id: next_id,
        question: cdq_text,
        is_cdq: true,
        thread: NarrativeThread::MainPlot,
        introduced_at: 0,
        resolved_at: None,
        outcomes: cdq_outcomes,
    });
    next_id += 1;

    // Subplot questions from secondary themes
    for (i, (theme_name, _score)) in themes.iter().enumerate().skip(1).take(3) {
        questions.push(NarrativeQuestion {
            id: next_id,
            question: format!("How will the theme of '{}' resolve?", theme_name),
            is_cdq: false,
            thread: NarrativeThread::Subplot(theme_name.clone()),
            introduced_at: (i * scenes.len()) / (themes.len().max(1)),
            resolved_at: None,
            outcomes: vec![
                PossibleOutcome {
                    description: format!("{} is affirmed", theme_name),
                    probability: 0.5,
                    realized: false,
                },
                PossibleOutcome {
                    description: format!("{} is subverted", theme_name),
                    probability: 0.5,
                    realized: false,
                },
            ],
        });
        next_id += 1;
    }

    // Character arc questions
    for (character, arc_type) in character_arcs.iter().take(4) {
        questions.push(NarrativeQuestion {
            id: next_id,
            question: format!("Will {} complete their {} arc?", character, arc_type),
            is_cdq: false,
            thread: NarrativeThread::CharacterArc(character.clone()),
            introduced_at: 0,
            resolved_at: None,
            outcomes: vec![
                PossibleOutcome {
                    description: format!("{} completes arc", character),
                    probability: 0.5,
                    realized: false,
                },
                PossibleOutcome {
                    description: format!("{} fails or regresses", character),
                    probability: 0.3,
                    realized: false,
                },
                PossibleOutcome {
                    description: format!("{} transforms differently", character),
                    probability: 0.2,
                    realized: false,
                },
            ],
        });
        next_id += 1;
    }

    // Relationship questions from character pairs
    if character_arcs.len() >= 2 {
        for pair in character_arcs.windows(2).take(2) {
            let (char_a, _) = &pair[0];
            let (char_b, _) = &pair[1];
            questions.push(NarrativeQuestion {
                id: next_id,
                question: format!(
                    "How will the relationship between {} and {} evolve?",
                    char_a, char_b
                ),
                is_cdq: false,
                thread: NarrativeThread::Relationship(char_a.clone(), char_b.clone()),
                introduced_at: 0,
                resolved_at: None,
                outcomes: vec![
                    PossibleOutcome {
                        description: "Relationship strengthens".to_string(),
                        probability: 0.4,
                        realized: false,
                    },
                    PossibleOutcome {
                        description: "Relationship deteriorates".to_string(),
                        probability: 0.3,
                        realized: false,
                    },
                    PossibleOutcome {
                        description: "Relationship transforms".to_string(),
                        probability: 0.3,
                        realized: false,
                    },
                ],
            });
            next_id += 1;
        }
    }
    let _ = next_id; // suppress unused warning

    // ---- Information-pacing curve, re-sourced from semantic novelty ----
    // Each scene's novelty is how much new semantic content it carries versus its
    // recent predecessors, measured from SBERT embeddings. This replaces the
    // former keyword-matched question-probability tracking, whose curve was
    // near-flat noise on real prose. The dramatic-question list above is retained
    // as a descriptive map of open threads, not a scored signal.
    let n = scenes.len();
    let embeddings = crate::substrate::sbert::encode_batch(scenes);
    let have_embeddings = embeddings.len() == n && embeddings.iter().all(|e| !e.is_empty());
    let novelty = if have_embeddings {
        compute_novelty_curve(&embeddings)
    } else {
        // SBERT unavailable: emit a flat neutral curve instead of fabricating a
        // novelty signal. The pacing verdict below is also neutralized so a
        // model-less run is not mistaken for a genuinely well-paced manuscript.
        log::warn!("[narrative_entropy] SBERT embeddings unavailable; pacing analysis degraded");
        vec![NEUTRAL_NOVELTY; n]
    };

    let mut entropy_curve: Vec<EntropyPoint> = Vec::with_capacity(n);
    let mut cdq_entropy_curve: Vec<f64> = Vec::with_capacity(n);
    let mut entropy_rate_curve: Vec<f64> = Vec::with_capacity(n);
    let mut complexity_curve: Vec<f64> = Vec::with_capacity(n);
    let mut redundant_scenes: Vec<(usize, String)> = Vec::new();
    let mut cumulative_info = 0.0;
    let mut prev_rate = 0.0;

    for (i, &nov) in novelty.iter().enumerate() {
        let rate = if i == 0 { 0.0 } else { nov - novelty[i - 1] };
        let acceleration = if i == 0 { 0.0 } else { rate - prev_rate };
        cumulative_info += nov;

        if have_embeddings && i > 0 && nov < REDUNDANT_NOVELTY {
            redundant_scenes.push((i, "Little new information vs. preceding scenes".to_string()));
        }

        entropy_curve.push(EntropyPoint {
            scene: i,
            per_question_entropy: Vec::new(),
            composite_entropy: nov,
            entropy_rate: rate,
            entropy_acceleration: acceleration,
            information_gain: nov,
            conditional_entropies: Vec::new(),
        });
        cdq_entropy_curve.push(nov);
        entropy_rate_curve.push(rate);
        complexity_curve.push(cumulative_info);
        prev_rate = rate;
    }

    // Per-thread integration tracking is gone with the per-question curves (its
    // former Pearson-correlation proxy had no real basis). Report neutral.
    let subplot_integration = 0.5;

    // Diagnose pacing from the novelty curve.
    let pacing_diagnosis = diagnose_pacing(&entropy_curve, n);

    let mut pacing_score = match &pacing_diagnosis {
        EntropyDiagnosis::WellPaced => 1.0,
        EntropyDiagnosis::Predictable { .. } => 0.4,
        EntropyDiagnosis::Chaotic { .. } => 0.4,
        EntropyDiagnosis::Rushed { drop_magnitude, .. } => {
            if *drop_magnitude > 0.4 {
                0.4
            } else {
                0.7
            }
        }
        EntropyDiagnosis::Unresolved { .. } => 0.2,
        EntropyDiagnosis::Mixed(issues) => {
            let base = 1.0 - (issues.len() as f64 * 0.2);
            base.clamp(0.2, 0.7)
        }
    };

    // Generate pacing feedback notes
    let pacing_feedback = generate_pacing_feedback(
        &entropy_curve,
        &redundant_scenes,
        &pacing_diagnosis,
        scenes.len(),
    );

    let mut overall_pacing_assessment = generate_pacing_assessment(
        &pacing_diagnosis,
        pacing_score,
        &redundant_scenes,
        &entropy_curve,
        scenes.len(),
    );

    if !have_embeddings {
        // No embeddings means no measurement: report a neutral score and say so,
        // rather than surfacing the default verdict's confident positive assessment.
        pacing_score = 0.5;
        overall_pacing_assessment =
            "Pacing analysis unavailable: the semantic model is not loaded.".to_string();
    }

    NarrativeEntropyResult {
        questions,
        entropy_curve,
        cdq_entropy_curve,
        entropy_rate_curve,
        redundant_scenes,
        subplot_integration,
        pacing_score,
        pacing_diagnosis,
        complexity_curve,
        pacing_feedback,
        overall_pacing_assessment,
    }
}

/// Diagnose pacing from the novelty curve.
fn diagnose_pacing(entropy_curve: &[EntropyPoint], num_scenes: usize) -> EntropyDiagnosis {
    if entropy_curve.is_empty() {
        return EntropyDiagnosis::WellPaced;
    }

    let mut issues: Vec<Box<EntropyDiagnosis>> = Vec::new();

    let composites: Vec<f64> = entropy_curve
        .iter()
        .map(|ep| ep.composite_entropy)
        .collect();

    // Check for Predictable: novelty stalls low (< 0.15) before 30% through.
    let thirty_pct = (num_scenes as f64 * 0.3) as usize;
    for (i, &c) in composites.iter().enumerate() {
        if i > 0 && i < thirty_pct && c < 0.15 {
            issues.push(Box::new(EntropyDiagnosis::Predictable { plateau_scene: i }));
            break;
        }
    }

    // Check for Chaotic: high novelty variance (std dev > 0.25)
    if composites.len() > 2 {
        let mean: f64 = composites.iter().sum::<f64>() / composites.len() as f64;
        let variance: f64 =
            composites.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / composites.len() as f64;
        let std_dev = variance.sqrt();
        if std_dev > 0.25 {
            let spike_scenes: Vec<usize> = composites
                .iter()
                .enumerate()
                .filter(|(_, c)| (**c - mean).abs() > std_dev)
                .map(|(i, _)| i)
                .collect();
            issues.push(Box::new(EntropyDiagnosis::Chaotic { spike_scenes }));
        }
    }

    // Check for Rushed: novelty drops sharply (> 0.3) in the final 10%.
    let ninety_pct = (num_scenes as f64 * 0.9) as usize;
    if composites.len() > ninety_pct && ninety_pct > 0 {
        let entropy_at_90 = composites[ninety_pct];
        if let Some(&final_entropy) = composites.last() {
            let drop = entropy_at_90 - final_entropy;
            if drop > 0.3 {
                issues.push(Box::new(EntropyDiagnosis::Rushed {
                    drop_scene: ninety_pct,
                    drop_magnitude: drop,
                }));
            }
        }
    }

    // No "Unresolved" check: it required per-question resolution tracking, which the
    // novelty reframe removed (resolved_at is now always None). A high final-scene
    // novelty does not imply unresolved threads, so scoring it here produced frequent
    // false positives (worst-case pacing_score plus a misleading open-questions dump).

    match issues.len() {
        0 => EntropyDiagnosis::WellPaced,
        1 => *issues.remove(0),
        _ => EntropyDiagnosis::Mixed(issues),
    }
}

/// Generate pacing feedback notes from entropy analysis.
fn generate_pacing_feedback(
    entropy_curve: &[EntropyPoint],
    redundant_scenes: &[(usize, String)],
    diagnosis: &EntropyDiagnosis,
    num_scenes: usize,
) -> Vec<PacingNote> {
    let mut notes = Vec::new();

    // Redundant scenes: low semantic novelty vs. recent scenes
    if !redundant_scenes.is_empty() {
        // Group consecutive redundant scenes into ranges
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for &(scene, _) in redundant_scenes {
            if let Some(last) = ranges.last_mut()
                && scene == last.1 + 1 {
                    last.1 = scene;
                    continue;
                }
            ranges.push((scene, scene));
        }
        for (start, end) in ranges {
            notes.push(PacingNote {
                scene_range: (start, end),
                issue: "Redundant scenes".to_string(),
                reader_experience: format!(
                    "Scenes {}\u{2013}{} carry little new information relative to the surrounding scenes. \
                     The reader is treading water.",
                    start, end
                ),
                fix: "Either cut these scenes, or use them to advance a subplot or deepen character \u{2014} \
                    but they MUST change something the reader cares about.".to_string(),
            });
        }
    }

    // Rushed resolution: entropy drops >1.5 bits in final scenes
    if let EntropyDiagnosis::Rushed {
        drop_scene,
        drop_magnitude,
    } = diagnosis
    {
        notes.push(PacingNote {
            scene_range: (*drop_scene, num_scenes.saturating_sub(1)),
            issue: "Rushed resolution".to_string(),
            reader_experience: format!(
                "The resolution feels rushed \u{2014} the story stops introducing new material and races to \
                 wrap up (semantic novelty dropped {:.2} in the final stretch). The reader needs time to FEEL each revelation.",
                drop_magnitude
            ),
            fix: "Consider spacing out the final answers with breathing room between each. \
                Let the reader absorb one revelation before delivering the next.".to_string(),
        });
    }

    // Check for stalled momentum: novelty drops below the redundancy floor early.
    let forty_pct = (num_scenes as f64 * 0.4) as usize;
    for ep in entropy_curve.iter() {
        if ep.scene > 0 && ep.scene < forty_pct && ep.composite_entropy < 0.15 {
            notes.push(PacingNote {
                scene_range: (ep.scene, ep.scene),
                issue: "Stalled momentum".to_string(),
                reader_experience: format!(
                    "By scene {}, the prose is retreading familiar ground rather than introducing new \
                     material. The reader is treading water early.",
                    ep.scene
                ),
                fix: "Introduce a new complication, setting, or revelation here, or deepen an existing \
                    thread \u{2014} the scene needs to change something the reader cares about.".to_string(),
            });
            break;
        }
    }

    // Chaotic threads: high variance in entropy rate
    if let EntropyDiagnosis::Chaotic { spike_scenes } = diagnosis {
        let start = spike_scenes.first().copied().unwrap_or(0);
        let end = spike_scenes
            .last()
            .copied()
            .unwrap_or(num_scenes.saturating_sub(1));
        notes.push(PacingNote {
            scene_range: (start, end),
            issue: "Chaotic narrative threads".to_string(),
            reader_experience: "The reader is pulled in too many directions \u{2014} scene-to-scene novelty \
                swings sharply, with no steady build."
                .to_string(),
            fix:
                "The fix isn't to simplify, but to SEQUENCE: focus each act on a primary question, \
                letting secondary questions orbit."
                    .to_string(),
        });
    }

    // Handle Mixed diagnosis
    if let EntropyDiagnosis::Mixed(issues) = diagnosis {
        for issue in issues {
            if let EntropyDiagnosis::Rushed {
                drop_scene,
                drop_magnitude,
            } = issue.as_ref()
            {
                notes.push(PacingNote {
                    scene_range: (*drop_scene, num_scenes.saturating_sub(1)),
                    issue: "Rushed resolution".to_string(),
                    reader_experience: format!(
                        "The resolution feels rushed \u{2014} semantic novelty dropped {:.2} in the final stretch.",
                        drop_magnitude
                    ),
                    fix: "Space out final revelations with breathing room between each.".to_string(),
                });
            }
            if let EntropyDiagnosis::Chaotic { spike_scenes } = issue.as_ref() {
                let start = spike_scenes.first().copied().unwrap_or(0);
                let end = spike_scenes
                    .last()
                    .copied()
                    .unwrap_or(num_scenes.saturating_sub(1));
                notes.push(PacingNote {
                    scene_range: (start, end),
                    issue: "Chaotic narrative threads".to_string(),
                    reader_experience: "Scene-to-scene novelty swings unpredictably."
                        .to_string(),
                    fix: "Sequence your questions: focus each act on a primary question."
                        .to_string(),
                });
            }
        }
    }

    notes
}

/// Generate overall pacing assessment string.
fn generate_pacing_assessment(
    diagnosis: &EntropyDiagnosis,
    pacing_score: f64,
    redundant_scenes: &[(usize, String)],
    entropy_curve: &[EntropyPoint],
    num_scenes: usize,
) -> String {
    if entropy_curve.is_empty() {
        return String::new();
    }

    let pace_desc = if pacing_score > 0.8 {
        "well-controlled and builds appropriately"
    } else if pacing_score > 0.6 {
        "adequate but with notable weak spots"
    } else if pacing_score > 0.4 {
        "uneven, with significant pacing issues"
    } else {
        "problematic, undermining reader engagement"
    };

    let feel_desc = match diagnosis {
        EntropyDiagnosis::WellPaced => {
            "feel engaged throughout, with questions answered at a satisfying pace"
        }
        EntropyDiagnosis::Predictable { .. } => {
            "lose interest early because the outcome becomes obvious too soon"
        }
        EntropyDiagnosis::Chaotic { .. } => "feel disoriented by the lack of narrative focus",
        EntropyDiagnosis::Rushed { .. } => {
            "feel cheated by a resolution that doesn't give revelations room to breathe"
        }
        EntropyDiagnosis::Unresolved { .. } => {
            "feel frustrated by narrative questions left hanging without resolution"
        }
        EntropyDiagnosis::Mixed(_) => {
            "experience inconsistent engagement due to multiple pacing issues"
        }
    };

    let redundant_pct = if num_scenes > 0 {
        (redundant_scenes.len() as f64 / num_scenes as f64) * 100.0
    } else {
        0.0
    };

    let manuscript_desc = if redundant_pct > 30.0 {
        format!(
            "The manuscript has {:.0}% redundant scenes that could be cut or repurposed.",
            redundant_pct
        )
    } else if pacing_score > 0.8 {
        "The manuscript handles information delivery effectively.".to_string()
    } else {
        "The manuscript needs pacing adjustments to maintain reader investment.".to_string()
    };

    format!(
        "The reader's experience of pace is {}. They will {}. {}",
        pace_desc, feel_desc, manuscript_desc
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_scenes() {
        let result = compute_narrative_entropy(&[], &[], "three_act", "literary", &[]);
        assert!(result.entropy_curve.is_empty());
        assert_eq!(result.pacing_score, 1.0);
    }

    #[test]
    fn test_basic_entropy_computation() {
        let scenes = vec![
            "The detective found a clue at the crime scene. Blood on the floor.",
            "She interviewed the suspect who denied everything. Lies were evident.",
            "New evidence pointed to the butler. The truth emerged slowly.",
            "In the final confrontation, the truth was fully revealed.",
        ];
        let themes = vec![("justice".to_string(), 0.9), ("deception".to_string(), 0.7)];
        let arcs = vec![("Detective".to_string(), "growth".to_string())];

        let result = compute_narrative_entropy(&scenes, &themes, "three_act", "mystery", &arcs);

        assert!(!result.entropy_curve.is_empty());
        assert_eq!(result.entropy_curve.len(), 4);
        assert!(!result.questions.is_empty());
        assert!(result.questions[0].is_cdq);
        assert_eq!(result.complexity_curve.len(), 4);
        // Complexity should be non-decreasing
        for i in 1..result.complexity_curve.len() {
            assert!(result.complexity_curve[i] >= result.complexity_curve[i - 1]);
        }
    }

    #[test]
    fn test_romance_genre_cdq() {
        let scenes = vec!["They met at the cafe. Love was in the air."];
        let themes = vec![("love".to_string(), 0.9)];
        let result = compute_narrative_entropy(&scenes, &themes, "three_act", "romance", &[]);
        assert!(result.questions[0].question.contains("together"));
    }

    #[test]
    fn test_compute_novelty_curve_flags_repeats() {
        // Scenes 0-2 are identical; scene 3 is orthogonal. The repeats must read as
        // ~0 novelty, the divergent scene as high.
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let embeddings = vec![a.clone(), a.clone(), a.clone(), b.clone()];
        let novelty = compute_novelty_curve(&embeddings);
        assert_eq!(novelty.len(), 4);
        assert!(novelty[2] < 0.1, "repeated scene should read low: {}", novelty[2]);
        assert!(
            novelty[3] > 0.5,
            "divergent scene should read high: {}",
            novelty[3]
        );
    }

    #[test]
    fn test_compute_novelty_curve_backfills_scene_zero() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]];
        let novelty = compute_novelty_curve(&embeddings);
        let mean_rest = (novelty[1] + novelty[2]) / 2.0;
        assert!((novelty[0] - mean_rest).abs() < 1e-9);
    }

    #[test]
    fn test_degraded_path_without_model() {
        // No SBERT model in the test environment: embeddings are empty, so the curve
        // degrades to a flat neutral signal and the verdict is neutralized (score 0.5,
        // explicit "unavailable" assessment) rather than a fabricated positive one.
        let scenes = vec!["First scene.", "Second scene.", "Third scene."];
        let result = compute_narrative_entropy(&scenes, &[], "three_act", "literary", &[]);
        assert_eq!(result.entropy_curve.len(), 3);
        assert!(
            result
                .entropy_curve
                .iter()
                .all(|ep| ep.composite_entropy == NEUTRAL_NOVELTY)
        );
        assert!(result.redundant_scenes.is_empty());
        assert_eq!(result.pacing_score, 0.5);
        assert!(
            result.overall_pacing_assessment.contains("unavailable"),
            "degraded assessment should signal unavailability: {:?}",
            result.overall_pacing_assessment
        );
    }

    fn curve_from(vals: &[f64]) -> Vec<EntropyPoint> {
        vals.iter()
            .enumerate()
            .map(|(i, &c)| EntropyPoint {
                scene: i,
                per_question_entropy: Vec::new(),
                composite_entropy: c,
                entropy_rate: 0.0,
                entropy_acceleration: 0.0,
                information_gain: c,
                conditional_entropies: Vec::new(),
            })
            .collect()
    }

    #[test]
    fn test_diagnose_chaotic_on_high_variance() {
        // A novelty curve swinging between extremes has std dev > 0.25.
        let diagnosis = diagnose_pacing(&curve_from(&[0.05, 0.95, 0.05, 0.95, 0.05]), 5);
        assert!(matches!(
            diagnosis,
            EntropyDiagnosis::Chaotic { .. } | EntropyDiagnosis::Mixed(_)
        ));
    }

    #[test]
    fn test_diagnose_predictable_on_early_stall() {
        // Novelty drops below the redundancy floor early (scene 1 of 10).
        let mut vals = vec![0.5; 10];
        vals[1] = 0.05;
        let diagnosis = diagnose_pacing(&curve_from(&vals), 10);
        assert!(matches!(
            diagnosis,
            EntropyDiagnosis::Predictable { .. } | EntropyDiagnosis::Mixed(_)
        ));
    }

    #[test]
    fn test_diagnose_rushed_on_late_drop() {
        // Novelty falls > 0.3 across the final 10% (scene 18 -> 19 of 20).
        let mut vals = vec![0.5; 20];
        vals[18] = 0.9;
        vals[19] = 0.1;
        let diagnosis = diagnose_pacing(&curve_from(&vals), 20);
        assert!(matches!(
            diagnosis,
            EntropyDiagnosis::Rushed { .. } | EntropyDiagnosis::Mixed(_)
        ));
    }
}
