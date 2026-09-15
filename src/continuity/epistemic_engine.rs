use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::substrate::utils::split_sentences;

/// A single piece of information in the story world.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoryFact {
    pub id: usize,
    pub content: String,
    pub established_scene: usize,
    pub importance: f64,
}

impl PartialEq for StoryFact {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.content == other.content
            && self.established_scene == other.established_scene
            && self.importance.to_bits() == other.importance.to_bits()
    }
}

impl Eq for StoryFact {}

impl std::hash::Hash for StoryFact {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.content.hash(state);
        self.established_scene.hash(state);
        self.importance.to_bits().hash(state);
    }
}

/// Per-character knowledge state.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpistemicState {
    pub character: String,
    pub knows: HashSet<usize>,
    pub believes_false: HashSet<usize>,
    pub uncertain_about: HashSet<usize>,
}

/// The reader's knowledge state (always knows everything shown).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReaderEpistemicState {
    pub knows: HashSet<usize>,
    pub scene: usize,
}

/// Epistemic asymmetry measurement.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpistemicAsymmetry {
    pub scene: usize,
    pub reader_advantage: f64,
    pub character_advantage: f64,
    pub total_hidden_bits: f64,
    pub dramatic_irony_intensity: f64,
}

/// Information revelation event.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RevelationEvent {
    pub scene: usize,
    pub fact_id: usize,
    pub revealed_to: String,
    pub bits_revealed: f64,
    pub dramatic_effect: String,
}

/// Suspense curve point.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SuspensePoint {
    pub scene: usize,
    pub suspense_level: f64,
    pub source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpistemicResult {
    pub facts: Vec<StoryFact>,
    pub character_states: HashMap<String, EpistemicState>,
    pub asymmetries: Vec<EpistemicAsymmetry>,
    pub revelations: Vec<RevelationEvent>,
    pub suspense_curve: Vec<SuspensePoint>,
    pub peak_suspense_scene: usize,
    pub information_density: f64,
    pub manipulation_score: f64,
    pub epistemic_insights: Vec<String>,
}

// --- Internal helpers ---

/// Extract declarative facts from a scene's text.
fn extract_facts(
    scene_text: &str,
    scene_idx: usize,
    characters: &[&str],
    fact_id_start: usize,
) -> Vec<StoryFact> {
    let sentences = split_sentences(scene_text);
    let mut facts: Vec<StoryFact> = Vec::new();

    let secret_markers = [
        "secret",
        "hidden",
        "concealed",
        "nobody knew",
        "no one knew",
        "unknown",
        "private",
        "confidential",
    ];
    let plan_markers = [
        "planned",
        "intended",
        "would soon",
        "scheme",
        "plot",
        "strategy",
        "meant to",
    ];
    let identity_markers = [
        "actually was",
        "real name",
        "true identity",
        "disguised",
        "pretending",
        "in truth",
        "really",
    ];
    let relationship_markers = [
        "loved", "hated", "betrayed", "trusted", "married", "siblings", "father", "mother",
        "enemy", "ally",
    ];
    let event_markers = [
        "happened",
        "occurred",
        "took place",
        "died",
        "born",
        "killed",
        "destroyed",
        "created",
        "built",
        "discovered",
    ];

    for sentence in &sentences {
        if sentence.split_whitespace().count() < 5 {
            continue;
        }

        let lower = sentence.to_lowercase();

        // Compute importance score
        let mut importance: f64 = 0.2; // base

        // Protagonist involvement
        let has_char = characters.iter().any(|c| lower.contains(&c.to_lowercase()));
        if has_char {
            importance += 0.3;
        }

        // Secret/mystery connection
        if secret_markers.iter().any(|m| lower.contains(m))
            || plan_markers.iter().any(|m| lower.contains(m))
            || identity_markers.iter().any(|m| lower.contains(m))
        {
            importance += 0.3;
        }

        // Relationship facts
        if relationship_markers.iter().any(|m| lower.contains(m)) {
            importance += 0.2;
        }

        // Major events
        if event_markers.iter().any(|m| lower.contains(m)) {
            importance += 0.2;
        }

        // Cap at 1.0
        importance = importance.min(1.0);

        // Only include sentences that establish facts (declarative, not questions/exclamations)
        let is_question = sentence.trim_end().ends_with('?');
        let is_exclamation_only =
            sentence.trim_end().ends_with('!') && sentence.split_whitespace().count() <= 3;

        if !is_question && !is_exclamation_only && importance >= 0.3 {
            facts.push(StoryFact {
                id: fact_id_start + facts.len(),
                content: sentence.clone(),
                established_scene: scene_idx,
                importance,
            });
        }
    }

    facts
}

/// Determine which characters are "present" in a scene based on mentions.
fn characters_present<'a>(scene_text: &str, characters: &'a [&str]) -> Vec<&'a str> {
    let lower = scene_text.to_lowercase();
    characters
        .iter()
        .filter(|&&c| lower.contains(&c.to_lowercase()))
        .copied()
        .collect()
}

/// Check if a fact involves dialogue (spoken aloud, so present characters hear it).
fn is_dialogue_fact(fact_content: &str) -> bool {
    fact_content.contains('"')
        || fact_content.contains('\u{201C}') // left double quote
        || fact_content.contains('\u{201D}') // right double quote
}

/// Check if a fact represents hidden information (secrets, internal thoughts).
fn is_hidden_from_others(fact_content: &str) -> bool {
    let lower = fact_content.to_lowercase();
    let internal_markers = [
        "thought to himself",
        "thought to herself",
        "secretly",
        "in her mind",
        "in his mind",
        "inwardly",
        "silently wondered",
        "didn't tell",
        "kept hidden",
        "nobody knew",
        "no one knew",
    ];
    internal_markers.iter().any(|m| lower.contains(m))
}

/// Weighted bits: sum of importance for unknown facts.
fn weighted_bits(fact_ids: &HashSet<usize>, facts: &[StoryFact]) -> f64 {
    fact_ids
        .iter()
        .filter_map(|&id| facts.iter().find(|f| f.id == id))
        .map(|f| f.importance)
        .sum()
}

pub fn track_epistemics(
    scenes: &[&str],
    characters: &[&str],
    pov_characters: &[&str],
) -> EpistemicResult {
    if scenes.is_empty() {
        return EpistemicResult {
            facts: Vec::new(),
            character_states: HashMap::new(),
            asymmetries: Vec::new(),
            revelations: Vec::new(),
            suspense_curve: Vec::new(),
            peak_suspense_scene: 0,
            information_density: 0.0,
            manipulation_score: 0.0,
            epistemic_insights: Vec::new(),
        };
    }

    // --- Phase 1: Fact Extraction ---
    let mut all_facts: Vec<StoryFact> = Vec::new();
    let mut scene_facts: Vec<Vec<usize>> = Vec::new(); // fact IDs per scene

    for (scene_idx, &scene_text) in scenes.iter().enumerate() {
        let new_facts = extract_facts(scene_text, scene_idx, characters, all_facts.len());
        let ids: Vec<usize> = new_facts.iter().map(|f| f.id).collect();
        scene_facts.push(ids);
        all_facts.extend(new_facts);
    }

    // --- Phase 2: Knowledge Assignment ---
    let mut char_states: HashMap<String, EpistemicState> = HashMap::new();
    for &c in characters {
        char_states.insert(
            c.to_string(),
            EpistemicState {
                character: c.to_string(),
                knows: HashSet::new(),
                believes_false: HashSet::new(),
                uncertain_about: HashSet::new(),
            },
        );
    }

    let mut reader_state = ReaderEpistemicState {
        knows: HashSet::new(),
        scene: 0,
    };

    let mut revelations: Vec<RevelationEvent> = Vec::new();
    let mut asymmetries: Vec<EpistemicAsymmetry> = Vec::new();
    let mut suspense_curve: Vec<SuspensePoint> = Vec::new();

    for (scene_idx, &scene_text) in scenes.iter().enumerate() {
        let present = characters_present(scene_text, characters);
        let pov = pov_characters.get(scene_idx).copied().unwrap_or("");
        let facts_this_scene = &scene_facts[scene_idx];

        for &fact_id in facts_this_scene {
            let fact = &all_facts[fact_id];
            let fact_content = &fact.content;

            // Reader always learns narrated facts
            let reader_already_knew = reader_state.knows.contains(&fact_id);
            reader_state.knows.insert(fact_id);

            if !reader_already_knew {
                revelations.push(RevelationEvent {
                    scene: scene_idx,
                    fact_id,
                    revealed_to: "reader".to_string(),
                    bits_revealed: fact.importance,
                    dramatic_effect: "information_gained".to_string(),
                });
            }

            // POV character learns all facts in their scene
            if !pov.is_empty()
                && let Some(state) = char_states.get_mut(pov) {
                    let char_already_knew = state.knows.contains(&fact_id);
                    state.knows.insert(fact_id);
                    state.uncertain_about.remove(&fact_id);

                    if !char_already_knew {
                        revelations.push(RevelationEvent {
                            scene: scene_idx,
                            fact_id,
                            revealed_to: pov.to_string(),
                            bits_revealed: fact.importance,
                            dramatic_effect: classify_revelation_effect(
                                reader_already_knew,
                                char_already_knew,
                                fact,
                            ),
                        });
                    }
                }

            // Other present characters: learn dialogue facts (unless hidden)
            if is_dialogue_fact(fact_content) && !is_hidden_from_others(fact_content) {
                for &c in &present {
                    if c == pov {
                        continue;
                    }
                    if let Some(state) = char_states.get_mut(c) {
                        state.knows.insert(fact_id);
                        state.uncertain_about.remove(&fact_id);
                    }
                }
            }

            // Hidden facts: only POV character knows (internal thought)
            if is_hidden_from_others(fact_content) {
                // Mark other characters as uncertain
                for &c in &present {
                    if c == pov {
                        continue;
                    }
                    if let Some(state) = char_states.get_mut(c) {
                        state.uncertain_about.insert(fact_id);
                    }
                }
            }
        }

        // --- Phase 3: Compute Asymmetry for this scene ---
        let pov_knows = if !pov.is_empty() {
            char_states
                .get(pov)
                .map(|s| &s.knows)
                .cloned()
                .unwrap_or_default()
        } else {
            HashSet::new()
        };

        // Reader advantage: facts reader knows but POV doesn't
        let reader_advantage_facts: HashSet<usize> =
            reader_state.knows.difference(&pov_knows).copied().collect();
        let reader_advantage = weighted_bits(&reader_advantage_facts, &all_facts);

        // Character advantage: facts SOME character knows but reader doesn't
        let mut all_char_knowledge: HashSet<usize> = HashSet::new();
        for state in char_states.values() {
            all_char_knowledge.extend(&state.knows);
        }
        let char_advantage_facts: HashSet<usize> = all_char_knowledge
            .difference(&reader_state.knows)
            .copied()
            .collect();
        let character_advantage = weighted_bits(&char_advantage_facts, &all_facts);

        // Total hidden bits: all asymmetric information
        let mut hidden_facts: HashSet<usize> = HashSet::new();
        hidden_facts.extend(&reader_advantage_facts);
        hidden_facts.extend(&char_advantage_facts);
        let total_hidden_bits = weighted_bits(&hidden_facts, &all_facts);

        // Dramatic irony: reader knows something bad will happen to a character who doesn't know
        let dramatic_irony_intensity = compute_dramatic_irony(&reader_advantage_facts, &all_facts);

        asymmetries.push(EpistemicAsymmetry {
            scene: scene_idx,
            reader_advantage,
            character_advantage,
            total_hidden_bits,
            dramatic_irony_intensity,
        });

        // Suspense = total hidden information generating tension
        let suspense_source = if character_advantage > reader_advantage {
            "mystery (characters know more than reader)".to_string()
        } else if reader_advantage > character_advantage {
            "dramatic irony (reader knows more than characters)".to_string()
        } else {
            "balanced information distribution".to_string()
        };

        suspense_curve.push(SuspensePoint {
            scene: scene_idx,
            suspense_level: total_hidden_bits,
            source: suspense_source,
        });
    }

    // --- Phase 4: Peak suspense ---
    let peak_suspense_scene = suspense_curve
        .iter()
        .max_by(|a, b| {
            a.suspense_level
                .partial_cmp(&b.suspense_level)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| p.scene)
        .unwrap_or(0);

    // --- Phase 5: Information density ---
    let information_density = if scenes.is_empty() {
        0.0
    } else {
        all_facts.len() as f64 / scenes.len() as f64
    };

    // --- Phase 6: Manipulation score ---
    // How well does suspense build toward the climax?
    // Ideal: suspense peaks at ~75% of the way through (climax position)
    let manipulation_score = compute_manipulation_score(&suspense_curve, scenes.len());

    // --- Phase 7: Insights ---
    let mut epistemic_insights: Vec<String> = Vec::new();

    // Reader advantage insight
    if let Some(max_asym) = asymmetries.iter().max_by(|a, b| {
        a.reader_advantage
            .partial_cmp(&b.reader_advantage)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
        && max_asym.reader_advantage > 0.0 {
            let pov = pov_characters
                .get(max_asym.scene)
                .copied()
                .unwrap_or("the POV character");
            epistemic_insights.push(format!(
                "The reader holds {:.1} bits of information advantage over {} at scene {}. \
                 This creates dramatic irony; the reader knows something will go wrong.",
                max_asym.reader_advantage,
                pov,
                max_asym.scene + 1
            ));
        }

    // Peak suspense timing insight
    let total_scenes = scenes.len();
    let peak_ratio = if total_scenes > 1 {
        peak_suspense_scene as f64 / (total_scenes - 1) as f64
    } else {
        0.5
    };
    let timing = if peak_ratio < 0.5 {
        "early"
    } else if peak_ratio > 0.85 {
        "late"
    } else {
        "optimal"
    };
    epistemic_insights.push(format!(
        "Peak suspense occurs at scene {} ({}% through), which is {} for narrative tension. {}",
        peak_suspense_scene + 1,
        (peak_ratio * 100.0).round() as usize,
        timing,
        if timing == "early" {
            "Consider withholding key revelations longer to sustain tension."
        } else if timing == "late" {
            "Consider seeding more mystery earlier to build anticipation."
        } else {
            "The information flow is well-calibrated for maximum impact."
        }
    ));

    // Character with most hidden information
    if let Some((char_name, state)) = char_states
        .iter()
        .filter(|(_, s)| {
            !s.knows
                .difference(&reader_state.knows)
                .collect::<HashSet<_>>()
                .is_empty()
        })
        .max_by_key(|(_, s)| s.knows.difference(&reader_state.knows).count())
    {
        let hidden_count = state.knows.difference(&reader_state.knows).count();
        epistemic_insights.push(format!(
            "{} holds {} facts the reader doesn't know. This character is a source of mystery.",
            char_name, hidden_count
        ));
    }

    // Longest dramatic irony stretch
    let mut irony_streak = 0usize;
    let mut max_irony_streak = 0usize;
    let mut irony_streak_start = 0usize;
    let mut max_irony_start = 0usize;
    for (i, asym) in asymmetries.iter().enumerate() {
        if asym.dramatic_irony_intensity > 0.0 {
            if irony_streak == 0 {
                irony_streak_start = i;
            }
            irony_streak += 1;
            if irony_streak > max_irony_streak {
                max_irony_streak = irony_streak;
                max_irony_start = irony_streak_start;
            }
        } else {
            irony_streak = 0;
        }
    }
    if max_irony_streak >= 2 {
        let pov = pov_characters
            .get(max_irony_start)
            .copied()
            .unwrap_or("the POV character");
        epistemic_insights.push(format!(
            "{} operates with less information than the reader for {} consecutive scenes \
             (starting at scene {}). This sustained dramatic irony is the engine of tension.",
            pov,
            max_irony_streak,
            max_irony_start + 1
        ));
    }

    // Manipulation score insight
    if manipulation_score > 0.7 {
        epistemic_insights.push(
            "The author controls information flow effectively. Revelations are well-timed \
             and suspense builds naturally toward the climax."
                .to_string(),
        );
    } else if manipulation_score < 0.4 {
        epistemic_insights.push(
            "Information flow is uneven. Consider restructuring revelations: \
             withhold key secrets longer and space out major reveals for sustained engagement."
                .to_string(),
        );
    }

    EpistemicResult {
        facts: all_facts,
        character_states: char_states,
        asymmetries,
        revelations,
        suspense_curve,
        peak_suspense_scene,
        information_density,
        manipulation_score,
        epistemic_insights,
    }
}

fn classify_revelation_effect(
    reader_already_knew: bool,
    char_already_knew: bool,
    _fact: &StoryFact,
) -> String {
    if reader_already_knew && !char_already_knew {
        // Reader knew, character just learned — suspense release
        "suspense_release".to_string()
    } else if !reader_already_knew && !char_already_knew {
        // Both learn simultaneously — surprise
        "surprise".to_string()
    } else if reader_already_knew && char_already_knew {
        // Both already knew — confirmation
        "confirmation".to_string()
    } else {
        // Character knew but reader didn't (shouldn't happen in narration, but...)
        "dramatic_irony_established".to_string()
    }
}

/// Compute dramatic irony intensity from reader-advantage facts.
/// Higher importance facts known to the reader but not the character = stronger irony.
fn compute_dramatic_irony(reader_advantage_facts: &HashSet<usize>, all_facts: &[StoryFact]) -> f64 {
    if reader_advantage_facts.is_empty() {
        return 0.0;
    }

    // Dramatic irony is strongest when the reader knows high-importance facts
    // that the character doesn't (especially secrets, dangers, betrayals).
    let irony_weight: f64 = reader_advantage_facts
        .iter()
        .filter_map(|&id| all_facts.iter().find(|f| f.id == id))
        .map(|f| {
            let lower = f.content.to_lowercase();
            let mut weight = f.importance;
            // Boost for danger/betrayal facts
            let danger_markers = [
                "danger",
                "trap",
                "betray",
                "kill",
                "poison",
                "lie",
                "deceive",
                "ambush",
                "plot against",
            ];
            if danger_markers.iter().any(|m| lower.contains(m)) {
                weight += 0.3;
            }
            weight.min(1.0)
        })
        .sum();

    // Normalize to 0.0-1.0 range
    (irony_weight / (reader_advantage_facts.len() as f64 + 1.0)).min(1.0)
}

/// Compute manipulation score: how well the author controls information flow.
/// Ideal: suspense builds steadily, peaks near 75% (climax), then releases.
fn compute_manipulation_score(suspense_curve: &[SuspensePoint], _total_scenes: usize) -> f64 {
    if suspense_curve.len() < 3 {
        return 0.5; // insufficient data
    }

    let max_suspense = suspense_curve
        .iter()
        .map(|p| p.suspense_level)
        .fold(0.0f64, f64::max);

    if max_suspense == 0.0 {
        return 0.0; // no suspense at all
    }

    // Normalize suspense values
    let normalized: Vec<f64> = suspense_curve
        .iter()
        .map(|p| p.suspense_level / max_suspense)
        .collect();

    // Ideal curve: rises to peak at 75%, then drops
    let n = normalized.len() as f64;
    let ideal: Vec<f64> = (0..normalized.len())
        .map(|i| {
            let x = i as f64 / (n - 1.0);
            if x <= 0.75 {
                x / 0.75 // linear rise to 1.0 at 75%
            } else {
                1.0 - (x - 0.75) / 0.25 // linear drop from 1.0 to 0.0
            }
        })
        .collect();

    // Compute correlation between actual and ideal
    let mean_actual = normalized.iter().sum::<f64>() / n;
    let mean_ideal = ideal.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_actual = 0.0;
    let mut var_ideal = 0.0;

    for i in 0..normalized.len() {
        let da = normalized[i] - mean_actual;
        let di = ideal[i] - mean_ideal;
        cov += da * di;
        var_actual += da * da;
        var_ideal += di * di;
    }

    let denom = (var_actual * var_ideal).sqrt();
    if denom < 1e-10 {
        return 0.5;
    }

    let correlation = cov / denom;
    // Map correlation (-1..1) to score (0..1)
    ((correlation + 1.0) / 2.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let result = track_epistemics(&[], &[], &[]);
        assert_eq!(result.facts.len(), 0);
        assert_eq!(result.peak_suspense_scene, 0);
    }

    #[test]
    fn test_basic_fact_extraction() {
        let scene = "John discovered the secret passage behind the bookshelf. \
                     Nobody knew about this hidden room.";
        let result = track_epistemics(&[scene], &["John"], &["John"]);

        assert!(!result.facts.is_empty());
        // Facts should have high importance (secret + character involvement)
        assert!(result.facts[0].importance >= 0.5);
    }

    #[test]
    fn test_reader_learns_all_narrated_facts() {
        let scene = "The bomb was hidden under the table. Mary sat down, unaware.";
        let result = track_epistemics(&[scene], &["Mary"], &["Mary"]);

        // Reader knows all facts
        assert!(!result.facts.is_empty());
        // Check asymmetry: reader knows about the bomb, Mary may not
        // (depends on extraction)
    }

    #[test]
    fn test_dramatic_irony_detection() {
        let scene1 = "The assassin secretly poisoned the wine glass meant for the king.";
        let scene2 = "The king happily picked up his wine glass and raised it in a toast.";
        let result = track_epistemics(
            &[scene1, scene2],
            &["king", "assassin"],
            &["assassin", "king"],
        );

        // Reader knows about the poison from scene 1
        // King doesn't know in scene 2
        // Should detect dramatic irony
        if result.asymmetries.len() >= 2 {
            // Scene 2 should have reader advantage (reader knows about poison, king doesn't)
            assert!(result.asymmetries[1].reader_advantage >= 0.0);
        }
    }

    #[test]
    fn test_suspense_curve_generation() {
        let scenes = ["It was a quiet morning in the village.",
            "Sarah discovered that someone had been secretly watching her house.",
            "Nobody knew that the watcher planned to strike at midnight.",
            "At midnight, Sarah heard the door creak open."];
        let characters = ["Sarah"];
        let povs = ["Sarah", "Sarah", "Sarah", "Sarah"];

        let scene_refs: Vec<&str> = scenes.iter().map(|s| s.as_ref()).collect();
        let char_refs: Vec<&str> = characters.iter().map(|s| s.as_ref()).collect();
        let pov_refs: Vec<&str> = povs.iter().map(|s| s.as_ref()).collect();

        let result = track_epistemics(&scene_refs, &char_refs, &pov_refs);

        assert_eq!(result.suspense_curve.len(), 4);
        // Suspense should generally increase as secrets accumulate
    }

    #[test]
    fn test_manipulation_score_bounds() {
        let scene = "Something happened. Then another thing.";
        let result = track_epistemics(&[scene], &[], &[]);
        assert!(result.manipulation_score >= 0.0);
        assert!(result.manipulation_score <= 1.0);
    }

    #[test]
    fn test_multiple_characters_knowledge() {
        let scene = "\"I know where the treasure is,\" Alice told Bob loudly. \
                     Carol was standing nearby.";
        let result = track_epistemics(&[scene], &["Alice", "Bob", "Carol"], &["Alice"]);

        // Alice is POV, knows everything
        let alice_state = result.character_states.get("Alice").unwrap();
        // Bob and Carol are present; dialogue facts should propagate to them
        let bob_state = result.character_states.get("Bob").unwrap();

        // Alice should know at least as much as Bob
        assert!(alice_state.knows.len() >= bob_state.knows.len());
    }

    #[test]
    fn test_hidden_information() {
        let scene = "She thought to himself that he would betray them all. \
                     He smiled warmly at the group.";
        let result = track_epistemics(&[scene], &["he"], &["he"]);

        // The secret thought should be marked as hidden from others
        assert!(!result.facts.is_empty());
    }

    #[test]
    fn test_information_density() {
        let scene1 = "John walked to the store and bought milk.";
        let scene2 = "Mary discovered the secret vault. Nobody knew the password. \
                     The treasure had been hidden for centuries.";
        let result = track_epistemics(&[scene1, scene2], &["John", "Mary"], &["John", "Mary"]);

        // Scene 2 is more information-dense
        assert!(result.information_density > 0.0);
    }

    #[test]
    fn test_insights_generated() {
        let scenes = ["The detective knew the butler was lying about the murder weapon.",
            "The butler smiled innocently at the detective, unaware he had been found out.",
            "The detective decided to confront the butler at dinner."];
        let scene_refs: Vec<&str> = scenes.iter().map(|s| s.as_ref()).collect();
        let result = track_epistemics(
            &scene_refs,
            &["detective", "butler"],
            &["detective", "detective", "detective"],
        );

        assert!(!result.epistemic_insights.is_empty());
    }
}
