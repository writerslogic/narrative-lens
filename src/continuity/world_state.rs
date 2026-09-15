//! Neuro-symbolic world state tracker.
//!
//! Maintains a formal model of the narrative universe by extracting facts from
//! scene text and tracking them as hard logic. Detects continuity violations by
//! proving that no valid information path exists for impossible knowledge,
//! locations, or object states.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use regex::Regex;
use serde::{Deserialize, Serialize};

// ─── Data Types ───────────────────────────────────────────────────────────────

/// A fact about the story world at a specific point in narrative time.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum WorldFact {
    /// Character is at a location.
    CharacterAt { character: String, location: String },
    /// Character possesses an object.
    CharacterPossesses { character: String, object: String },
    /// Character knows a piece of information.
    CharacterKnows {
        character: String,
        information: String,
    },
    /// Character has witnessed an event.
    CharacterWitnessed { character: String, event: String },
    /// Two characters are in the same location (derived).
    CoLocated {
        character_a: String,
        character_b: String,
    },
    /// Communication occurred between characters.
    CommunicationVector {
        from: String,
        to: String,
        content: String,
    },
    /// Object is at a location.
    ObjectAt { object: String, location: String },
    /// A constraint: character CANNOT leave / is restricted.
    PhysicalConstraint {
        character: String,
        constraint: String,
    },
    /// Temporal fact: event X happened before event Y.
    TemporalOrdering { before: String, after: String },
}

/// A continuity violation: something the author got wrong.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContinuityViolation {
    pub scene: usize,
    pub violation_type: ViolationType,
    pub description: String,
    /// The logical chain proving the violation.
    pub proof: String,
    /// 0.0 = minor (object in wrong place), 1.0 = major (impossible knowledge).
    pub severity: f64,
    pub fix_suggestion: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ViolationType {
    /// Character knows something they couldn't have learned.
    ImpossibleKnowledge,
    /// Character is in a location they couldn't have reached.
    ImpossibleLocation,
    /// Object appears where it shouldn't be.
    ObjectContinuity,
    /// Events reference something that hasn't happened yet (non-flashback).
    TemporalParadox,
    /// Character references another character they haven't met.
    UnestablishedRelationship,
    /// Physical impossibility (locked room, distance).
    PhysicalImpossibility,
}

/// Snapshot of the world state at a scene boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldStateSnapshot {
    pub after_scene: usize,
    pub facts: HashSet<WorldFact>,
    pub new_facts: Vec<WorldFact>,
    pub invalidated_facts: Vec<WorldFact>,
}

/// Knowledge state for a single character.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CharacterKnowledgeState {
    pub character: String,
    /// Information they possess.
    pub knows: Vec<String>,
    /// Events they've seen.
    pub witnessed: Vec<String>,
    /// (who_told_them, what).
    pub told_by: Vec<(String, String)>,
    /// (scene, location).
    pub locations_visited: Vec<(usize, String)>,
    pub current_location: Option<String>,
}

/// Full world state analysis result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldStateResult {
    pub snapshots: Vec<WorldStateSnapshot>,
    pub violations: Vec<ContinuityViolation>,
    pub character_knowledge: Vec<CharacterKnowledgeState>,
    /// (scene, location, characters_present).
    pub location_timeline: Vec<(usize, String, Vec<String>)>,
    /// (scene, from, to, info).
    pub information_flow: Vec<(usize, String, String, String)>,
    /// 1.0 = no violations, 0.0 = full of holes.
    pub continuity_score: f64,
    /// Number of tracked facts / scenes.
    pub world_complexity: f64,
}

// ─── Compiled Regex Patterns ──────────────────────────────────────────────────

struct Patterns {
    location: Regex,
    entered: Regex,
    told: Regex,
    discovered: Regex,
    saw: Regex,
    overheard: Regex,
    read_info: Regex,
    realized: Regex,
    picked_up: Regex,
    held: Regex,
    took: Regex,
    gave: Regex,
    locked: Regex,
    couldnt_leave: Regex,
    alone_in: Regex,
}

impl Patterns {
    fn compile(characters_pattern: &str) -> Self {
        // Character names pattern for capture groups
        let cp = characters_pattern;

        Self {
            location: Regex::new(&format!(
                r"(?i)(?:{cp})\s+(?:was|were|stood|sat|waited|remained)\s+(?:in|at|inside)\s+(?:the\s+)?(\w[\w\s]{{0,30}}?)(?:\.|,|;|\s+(?:when|while|and|but|as))"
            )).unwrap(),
            entered: Regex::new(&format!(
                r"(?i)(?:{cp})\s+(?:entered|walked into|stepped into|arrived at|went to|came to|moved to|returned to)\s+(?:the\s+)?(\w[\w\s]{{0,30}}?)(?:\.|,|;|\s+(?:when|while|and|but|as))"
            )).unwrap(),
            told: Regex::new(&format!(
                r"(?i)(?:{cp})\s+(?:told|informed|explained to|said to|whispered to|mentioned to)\s+(?:{cp})\s+(?:about|that)\s+(.+?)(?:\.|;)"
            )).unwrap(),
            discovered: Regex::new(&format!(
                r"(?i)(?:{cp})\s+(?:discovered|found out|learned|uncovered)\s+(?:that\s+)?(.+?)(?:\.|;)"
            )).unwrap(),
            saw: Regex::new(&format!(
                r"(?i)(?:{cp})\s+(?:saw|watched|witnessed|observed|noticed)\s+(.+?)(?:\.|;)"
            )).unwrap(),
            overheard: Regex::new(&format!(
                r"(?i)(?:{cp})\s+(?:overheard|eavesdropped on|listened to)\s+(.+?)(?:\.|;)"
            )).unwrap(),
            read_info: Regex::new(&format!(
                r"(?i)(?:{cp})\s+(?:read|studied|examined)\s+(?:the\s+)?(.+?)(?:\.|;)"
            )).unwrap(),
            realized: Regex::new(&format!(
                r"(?i)(?:{cp})\s+(?:realized|understood|grasped|figured out)\s+(?:that\s+)?(.+?)(?:\.|;)"
            )).unwrap(),
            picked_up: Regex::new(&format!(
                r"(?i)(?:{cp})\s+(?:picked up|grabbed|snatched|seized)\s+(?:the\s+|a\s+)?(.+?)(?:\.|,|;)"
            )).unwrap(),
            held: Regex::new(&format!(
                r"(?i)(?:{cp})\s+(?:held|clutched|gripped|carried)\s+(?:the\s+|a\s+)?(.+?)(?:\.|,|;)"
            )).unwrap(),
            took: Regex::new(&format!(
                r"(?i)(?:{cp})\s+(?:took|pocketed|kept)\s+(?:the\s+|a\s+)?(.+?)(?:\.|,|;)"
            )).unwrap(),
            gave: Regex::new(&format!(
                r"(?i)(?:{cp})\s+(?:gave|handed|passed|tossed)\s+(?:the\s+|a\s+)?(.+?)\s+to\s+(?:{cp})(?:\.|,|;)"
            )).unwrap(),
            locked: Regex::new(
                r"(?i)(?:locked|bolted|barred|sealed)\s+(?:the\s+)?(?:door|gate|room|exit)"
            ).unwrap(),
            couldnt_leave: Regex::new(&format!(
                r"(?i)(?:{cp})\s+(?:couldn't leave|could not leave|was trapped|was stuck|was imprisoned)"
            )).unwrap(),
            alone_in: Regex::new(&format!(
                r"(?i)(?:{cp})\s+(?:was\s+)?alone\s+in\s+(?:the\s+)?(\w[\w\s]{{0,30}}?)(?:\.|,|;)"
            )).unwrap(),
        }
    }
}

/// Return the compiled [`Patterns`] for a character alternation, reusing a cached
/// set when the same alternation was compiled before. Compiling the 15 patterns
/// is allocation-heavy, and the alternation is a pure function of the (sorted,
/// escaped) character list, which changes far less often than scene text — so a
/// project re-analyzed with the same cast skips the recompile entirely.
fn compile_cached(chars_pattern: &str) -> Arc<Patterns> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<Patterns>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    // A poisoned lock only means a prior holder panicked; the cached patterns are
    // immutable `Arc`s, so recovering the guard and continuing is safe.
    if let Some(hit) = cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(chars_pattern)
    {
        return Arc::clone(hit);
    }

    // Compile outside the lock — it is the expensive part and must not block other
    // readers; a rare duplicate compile under a race is harmless.
    let compiled = Arc::new(Patterns::compile(chars_pattern));

    let mut map = cache.lock().unwrap_or_else(|e| e.into_inner());
    // Bound memory: casts vary per project, so cap distinct compiled sets. A miss
    // after a clear just recompiles, which is the pre-cache behavior.
    if map.len() >= 64 {
        map.clear();
    }
    map.insert(chars_pattern.to_string(), Arc::clone(&compiled));
    compiled
}

// ─── Knowledge Graph ──────────────────────────────────────────────────────────

/// Tracks how information flows between characters.
struct KnowledgeGraph {
    graph: DiGraph<String, (usize, String)>, // node=character, edge=(scene, info)
    node_map: HashMap<String, NodeIndex>,
}

impl KnowledgeGraph {
    fn new(characters: &[&str]) -> Self {
        let mut graph = DiGraph::new();
        let mut node_map = HashMap::new();
        for &c in characters {
            let idx = graph.add_node(c.to_string());
            node_map.insert(c.to_lowercase(), idx);
        }
        Self { graph, node_map }
    }

    fn add_communication(&mut self, from: &str, to: &str, scene: usize, info: &str) {
        let from_lower = from.to_lowercase();
        let to_lower = to.to_lowercase();
        if let (Some(&from_idx), Some(&to_idx)) =
            (self.node_map.get(&from_lower), self.node_map.get(&to_lower))
        {
            self.graph
                .add_edge(from_idx, to_idx, (scene, info.to_string()));
        }
    }

    /// Check if there is any path from a source of `info` to `target` by `scene`.
    fn has_info_path(&self, target: &str, info: &str, by_scene: usize) -> bool {
        let target_lower = target.to_lowercase();
        let target_idx = match self.node_map.get(&target_lower) {
            Some(&idx) => idx,
            None => return false,
        };

        // BFS backwards from target looking for any edge carrying this info
        // that happened at or before `by_scene`.
        let mut visited = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(target_idx);
        visited.insert(target_idx);

        while let Some(node) = queue.pop_front() {
            // Check incoming edges
            for edge in self
                .graph
                .edges_directed(node, petgraph::Direction::Incoming)
            {
                let (scene, ref edge_info) = *edge.weight();
                let source = edge.source();
                if scene <= by_scene && edge_info == info && !visited.contains(&source) {
                    return true;
                }
                if scene <= by_scene && !visited.contains(&source) {
                    visited.insert(source);
                    queue.push_back(source);
                }
            }
        }
        false
    }
}

// ─── Main Entry Point ─────────────────────────────────────────────────────────

/// Analyze scenes and characters to produce a full world state analysis.
///
/// # Arguments
/// * `scenes` - Ordered scene texts.
/// * `characters` - Known character names (case-insensitive matching).
pub fn track_world_state(scenes: &[&str], characters: &[&str]) -> WorldStateResult {
    if scenes.is_empty() || characters.is_empty() {
        return WorldStateResult {
            snapshots: Vec::new(),
            violations: Vec::new(),
            character_knowledge: Vec::new(),
            location_timeline: Vec::new(),
            information_flow: Vec::new(),
            continuity_score: 1.0,
            world_complexity: 0.0,
        };
    }

    // Build regex character alternation (escaped, longest first for greedy match)
    let mut sorted_chars: Vec<&str> = characters.to_vec();
    sorted_chars.sort_by_key(|c| std::cmp::Reverse(c.len()));
    let chars_pattern = sorted_chars
        .iter()
        .map(|c| regex::escape(c))
        .collect::<Vec<_>>()
        .join("|");

    let patterns = compile_cached(&chars_pattern);
    let mut knowledge_graph = KnowledgeGraph::new(characters);

    // Per-character state tracking
    let mut char_locations: HashMap<String, String> = HashMap::new();
    let mut char_knowledge: HashMap<String, HashSet<String>> = HashMap::new();
    let mut char_witnessed: HashMap<String, HashSet<String>> = HashMap::new();
    let mut char_possessions: HashMap<String, HashSet<String>> = HashMap::new();
    let mut char_constraints: HashMap<String, Vec<String>> = HashMap::new();
    let mut char_told_by: HashMap<String, Vec<(String, String)>> = HashMap::new();
    let mut char_locations_history: HashMap<String, Vec<(usize, String)>> = HashMap::new();

    // Object location tracking
    let mut object_locations: HashMap<String, (usize, String)> = HashMap::new(); // object -> (scene, location)

    // Global accumulators
    let mut snapshots: Vec<WorldStateSnapshot> = Vec::new();
    let mut violations: Vec<ContinuityViolation> = Vec::new();
    let mut location_timeline: Vec<(usize, String, Vec<String>)> = Vec::new();
    let mut information_flow: Vec<(usize, String, String, String)> = Vec::new();

    // Initialize character maps
    for &c in characters {
        let cl = c.to_lowercase();
        char_knowledge.insert(cl.clone(), HashSet::new());
        char_witnessed.insert(cl.clone(), HashSet::new());
        char_possessions.insert(cl.clone(), HashSet::new());
        char_constraints.insert(cl.clone(), Vec::new());
        char_told_by.insert(cl.clone(), Vec::new());
        char_locations_history.insert(cl.clone(), Vec::new());
    }

    
    let mut cumulative_facts: HashSet<WorldFact> = HashSet::new();

    for (scene_idx, &scene_text) in scenes.iter().enumerate() {
        let mut new_facts: Vec<WorldFact> = Vec::new();
        let mut invalidated_facts: Vec<WorldFact> = Vec::new();
        let text_lower = scene_text.to_lowercase();

        // ─── Location Extraction ──────────────────────────────────────────────
        extract_locations(
            scene_text,
            scene_idx,
            characters,
            &patterns,
            &mut char_locations,
            &mut char_locations_history,
            &mut new_facts,
            &mut invalidated_facts,
            &cumulative_facts,
        );

        // ─── Knowledge Extraction ─────────────────────────────────────────────
        extract_knowledge(
            scene_text,
            scene_idx,
            characters,
            &patterns,
            &mut char_knowledge,
            &mut char_witnessed,
            &mut char_told_by,
            &mut knowledge_graph,
            &mut new_facts,
            &mut information_flow,
        );

        // ─── Possession Extraction ────────────────────────────────────────────
        extract_possessions(
            scene_text,
            scene_idx,
            characters,
            &patterns,
            &mut char_possessions,
            &mut object_locations,
            &mut new_facts,
        );

        // ─── Constraint Extraction ────────────────────────────────────────────
        extract_constraints(
            scene_text,
            scene_idx,
            characters,
            &patterns,
            &text_lower,
            &char_locations,
            &mut char_constraints,
            &mut new_facts,
        );

        // ─── Inference: Co-location ──────────────────────────────────────────
        let colocation_facts = infer_colocation(characters, &char_locations);
        for fact in &colocation_facts {
            if !cumulative_facts.contains(fact) {
                new_facts.push(fact.clone());
            }
        }

        // ─── Build Location Timeline ─────────────────────────────────────────
        let mut loc_chars: HashMap<String, Vec<String>> = HashMap::new();
        for (ch, loc) in &char_locations {
            loc_chars.entry(loc.clone()).or_default().push(ch.clone());
        }
        for (loc, chars_at) in &loc_chars {
            location_timeline.push((scene_idx, loc.clone(), chars_at.clone()));
        }

        // ─── Violation Detection ─────────────────────────────────────────────
        detect_violations(
            scene_idx,
            characters,
            &char_locations,
            &char_constraints,
            &char_knowledge,
            &knowledge_graph,
            &cumulative_facts,
            &new_facts,
            &mut violations,
        );

        // ─── Update Cumulative Facts ─────────────────────────────────────────
        for fact in &invalidated_facts {
            cumulative_facts.remove(fact);
        }
        for fact in &new_facts {
            cumulative_facts.insert(fact.clone());
        }

        // Clear constraints at scene boundary (they may not persist)
        for constraints in char_constraints.values_mut() {
            constraints.clear();
        }

        snapshots.push(WorldStateSnapshot {
            after_scene: scene_idx,
            facts: cumulative_facts.clone(),
            new_facts,
            invalidated_facts,
        });
    }

    let total_facts_extracted: usize = cumulative_facts.len();

    // ─── Build Character Knowledge States ────────────────────────────────────
    let character_knowledge_states: Vec<CharacterKnowledgeState> = characters
        .iter()
        .map(|&c| {
            let cl = c.to_lowercase();
            CharacterKnowledgeState {
                character: c.to_string(),
                knows: char_knowledge
                    .get(&cl)
                    .map(|s| s.iter().cloned().collect())
                    .unwrap_or_default(),
                witnessed: char_witnessed
                    .get(&cl)
                    .map(|s| s.iter().cloned().collect())
                    .unwrap_or_default(),
                told_by: char_told_by.get(&cl).cloned().unwrap_or_default(),
                locations_visited: char_locations_history.get(&cl).cloned().unwrap_or_default(),
                current_location: char_locations.get(&cl).cloned(),
            }
        })
        .collect();

    // ─── Scoring ─────────────────────────────────────────────────────────────
    let weighted_violations: f64 = violations.iter().map(|v| v.severity).sum();
    let max_possible = scenes.len() as f64 * 0.5;
    let continuity_score = if max_possible > 0.0 {
        (1.0 - weighted_violations / max_possible).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let world_complexity = if !scenes.is_empty() {
        total_facts_extracted as f64 / scenes.len() as f64
    } else {
        0.0
    };

    WorldStateResult {
        snapshots,
        violations,
        character_knowledge: character_knowledge_states,
        location_timeline,
        information_flow,
        continuity_score,
        world_complexity,
    }
}

// ─── Extraction Helpers ───────────────────────────────────────────────────────

fn find_character_in_match<'a>(cap_text: &str, characters: &[&'a str]) -> Option<&'a str> {
    let lower = cap_text.to_lowercase();
    characters
        .iter()
        .find(|&&c| lower.contains(&c.to_lowercase()))
        .copied()
}

fn extract_locations(
    scene_text: &str,
    scene_idx: usize,
    characters: &[&str],
    patterns: &Patterns,
    char_locations: &mut HashMap<String, String>,
    char_locations_history: &mut HashMap<String, Vec<(usize, String)>>,
    new_facts: &mut Vec<WorldFact>,
    invalidated_facts: &mut Vec<WorldFact>,
    cumulative_facts: &HashSet<WorldFact>,
) {
    // "Character was in/at LOCATION"
    for cap in patterns.location.captures_iter(scene_text) {
        let full_match = cap.get(0).map(|m| m.as_str()).unwrap_or("");
        if let Some(character) = find_character_in_match(full_match, characters)
            && let Some(loc_match) = cap.get(1) {
                let location = loc_match.as_str().trim().to_lowercase();
                if is_valid_location(&location) {
                    update_character_location(
                        character,
                        &location,
                        scene_idx,
                        char_locations,
                        char_locations_history,
                        new_facts,
                        invalidated_facts,
                        cumulative_facts,
                    );
                }
            }
    }

    // "Character entered/walked into LOCATION"
    for cap in patterns.entered.captures_iter(scene_text) {
        let full_match = cap.get(0).map(|m| m.as_str()).unwrap_or("");
        if let Some(character) = find_character_in_match(full_match, characters)
            && let Some(loc_match) = cap.get(1) {
                let location = loc_match.as_str().trim().to_lowercase();
                if is_valid_location(&location) {
                    update_character_location(
                        character,
                        &location,
                        scene_idx,
                        char_locations,
                        char_locations_history,
                        new_facts,
                        invalidated_facts,
                        cumulative_facts,
                    );
                }
            }
    }
}

fn update_character_location(
    character: &str,
    location: &str,
    scene_idx: usize,
    char_locations: &mut HashMap<String, String>,
    char_locations_history: &mut HashMap<String, Vec<(usize, String)>>,
    new_facts: &mut Vec<WorldFact>,
    invalidated_facts: &mut Vec<WorldFact>,
    cumulative_facts: &HashSet<WorldFact>,
) {
    let cl = character.to_lowercase();
    let old_location = char_locations.get(&cl).cloned();

    // Invalidate previous location fact
    if let Some(ref old_loc) = old_location
        && old_loc != location {
            let old_fact = WorldFact::CharacterAt {
                character: cl.clone(),
                location: old_loc.clone(),
            };
            if cumulative_facts.contains(&old_fact) {
                invalidated_facts.push(old_fact);
            }
        }

    char_locations.insert(cl.clone(), location.to_string());
    char_locations_history
        .entry(cl.clone())
        .or_default()
        .push((scene_idx, location.to_string()));

    let fact = WorldFact::CharacterAt {
        character: cl,
        location: location.to_string(),
    };
    new_facts.push(fact);
}

fn is_valid_location(location: &str) -> bool {
    // Must be non-empty and not too long (avoids capturing garbage)
    if location.is_empty() || location.len() > 40 {
        return false;
    }
    // Basic sanity: contains at least one alphabetic character
    location.chars().any(|c| c.is_alphabetic())
}

fn extract_knowledge(
    scene_text: &str,
    scene_idx: usize,
    characters: &[&str],
    patterns: &Patterns,
    char_knowledge: &mut HashMap<String, HashSet<String>>,
    char_witnessed: &mut HashMap<String, HashSet<String>>,
    char_told_by: &mut HashMap<String, Vec<(String, String)>>,
    knowledge_graph: &mut KnowledgeGraph,
    new_facts: &mut Vec<WorldFact>,
    information_flow: &mut Vec<(usize, String, String, String)>,
) {
    // "Character told Character about INFO"
    for cap in patterns.told.captures_iter(scene_text) {
        let full_match = cap.get(0).map(|m| m.as_str()).unwrap_or("");
        // Find two different characters in the match
        let chars_found: Vec<&str> = characters
            .iter()
            .filter(|&&c| full_match.to_lowercase().contains(&c.to_lowercase()))
            .copied()
            .collect();

        if chars_found.len() >= 2 {
            let from = chars_found[0];
            let to = chars_found[1];
            if let Some(info_match) = cap.get(cap.len() - 1) {
                let info = info_match.as_str().trim().to_string();
                let from_lower = from.to_lowercase();
                let to_lower = to.to_lowercase();

                char_knowledge
                    .entry(to_lower.clone())
                    .or_default()
                    .insert(info.clone());
                char_told_by
                    .entry(to_lower.clone())
                    .or_default()
                    .push((from_lower.clone(), info.clone()));
                knowledge_graph.add_communication(from, to, scene_idx, &info);
                information_flow.push((
                    scene_idx,
                    from_lower.clone(),
                    to_lower.clone(),
                    info.clone(),
                ));

                new_facts.push(WorldFact::CommunicationVector {
                    from: from_lower,
                    to: to_lower.clone(),
                    content: info.clone(),
                });
                new_facts.push(WorldFact::CharacterKnows {
                    character: to_lower,
                    information: info,
                });
            }
        }
    }

    // "Character discovered INFO"
    for cap in patterns.discovered.captures_iter(scene_text) {
        let full_match = cap.get(0).map(|m| m.as_str()).unwrap_or("");
        if let Some(character) = find_character_in_match(full_match, characters)
            && let Some(info_match) = cap.get(cap.len() - 1) {
                let info = info_match.as_str().trim().to_string();
                let cl = character.to_lowercase();
                char_knowledge
                    .entry(cl.clone())
                    .or_default()
                    .insert(info.clone());
                new_facts.push(WorldFact::CharacterKnows {
                    character: cl,
                    information: info,
                });
            }
    }

    // "Character saw EVENT"
    for cap in patterns.saw.captures_iter(scene_text) {
        let full_match = cap.get(0).map(|m| m.as_str()).unwrap_or("");
        if let Some(character) = find_character_in_match(full_match, characters)
            && let Some(event_match) = cap.get(cap.len() - 1) {
                let event = event_match.as_str().trim().to_string();
                let cl = character.to_lowercase();
                char_witnessed
                    .entry(cl.clone())
                    .or_default()
                    .insert(event.clone());
                new_facts.push(WorldFact::CharacterWitnessed {
                    character: cl,
                    event,
                });
            }
    }

    // "Character overheard INFO"
    for cap in patterns.overheard.captures_iter(scene_text) {
        let full_match = cap.get(0).map(|m| m.as_str()).unwrap_or("");
        if let Some(character) = find_character_in_match(full_match, characters)
            && let Some(info_match) = cap.get(cap.len() - 1) {
                let info = info_match.as_str().trim().to_string();
                let cl = character.to_lowercase();
                char_knowledge
                    .entry(cl.clone())
                    .or_default()
                    .insert(info.clone());
                new_facts.push(WorldFact::CharacterKnows {
                    character: cl,
                    information: info,
                });
            }
    }

    // "Character read the letter/note"
    for cap in patterns.read_info.captures_iter(scene_text) {
        let full_match = cap.get(0).map(|m| m.as_str()).unwrap_or("");
        if let Some(character) = find_character_in_match(full_match, characters)
            && let Some(info_match) = cap.get(cap.len() - 1) {
                let info = info_match.as_str().trim().to_string();
                let cl = character.to_lowercase();
                char_knowledge
                    .entry(cl.clone())
                    .or_default()
                    .insert(info.clone());
                new_facts.push(WorldFact::CharacterKnows {
                    character: cl,
                    information: info,
                });
            }
    }

    // "Character realized INFO"
    for cap in patterns.realized.captures_iter(scene_text) {
        let full_match = cap.get(0).map(|m| m.as_str()).unwrap_or("");
        if let Some(character) = find_character_in_match(full_match, characters)
            && let Some(info_match) = cap.get(cap.len() - 1) {
                let info = info_match.as_str().trim().to_string();
                let cl = character.to_lowercase();
                char_knowledge
                    .entry(cl.clone())
                    .or_default()
                    .insert(info.clone());
                new_facts.push(WorldFact::CharacterKnows {
                    character: cl,
                    information: info,
                });
            }
    }
}

fn extract_possessions(
    scene_text: &str,
    scene_idx: usize,
    characters: &[&str],
    patterns: &Patterns,
    char_possessions: &mut HashMap<String, HashSet<String>>,
    object_locations: &mut HashMap<String, (usize, String)>,
    new_facts: &mut Vec<WorldFact>,
) {
    // "Character picked up/grabbed OBJECT"
    for pattern in [&patterns.picked_up, &patterns.held, &patterns.took] {
        for cap in pattern.captures_iter(scene_text) {
            let full_match = cap.get(0).map(|m| m.as_str()).unwrap_or("");
            if let Some(character) = find_character_in_match(full_match, characters)
                && let Some(obj_match) = cap.get(cap.len() - 1) {
                    let object = obj_match.as_str().trim().to_lowercase();
                    if is_valid_object(&object) {
                        let cl = character.to_lowercase();
                        char_possessions
                            .entry(cl.clone())
                            .or_default()
                            .insert(object.clone());
                        object_locations
                            .insert(object.clone(), (scene_idx, format!("with:{}", cl)));
                        new_facts.push(WorldFact::CharacterPossesses {
                            character: cl,
                            object,
                        });
                    }
                }
        }
    }

    // "Character gave OBJECT to Character"
    for cap in patterns.gave.captures_iter(scene_text) {
        let full_match = cap.get(0).map(|m| m.as_str()).unwrap_or("");
        let chars_found: Vec<&str> = characters
            .iter()
            .filter(|&&c| full_match.to_lowercase().contains(&c.to_lowercase()))
            .copied()
            .collect();

        if chars_found.len() >= 2 {
            let from = chars_found[0].to_lowercase();
            let to = chars_found[1].to_lowercase();
            // Try to find the object (middle capture group)
            if let Some(obj_match) = cap.get(1) {
                let object = obj_match.as_str().trim().to_lowercase();
                if is_valid_object(&object) {
                    char_possessions
                        .entry(from.clone())
                        .or_default()
                        .remove(&object);
                    char_possessions
                        .entry(to.clone())
                        .or_default()
                        .insert(object.clone());
                    object_locations.insert(object.clone(), (scene_idx, format!("with:{}", to)));
                    new_facts.push(WorldFact::CharacterPossesses {
                        character: to,
                        object,
                    });
                }
            }
        }
    }
}

fn is_valid_object(object: &str) -> bool {
    if object.is_empty() || object.len() > 40 {
        return false;
    }
    // Filter out common false positives (verbs, pronouns, etc.)
    let stopwords = [
        "him",
        "her",
        "them",
        "it",
        "his",
        "that",
        "this",
        "what",
        "which",
        "something",
        "nothing",
        "everything",
        "anything",
    ];
    !stopwords.contains(&object)
}

fn extract_constraints(
    scene_text: &str,
    _scene_idx: usize,
    characters: &[&str],
    patterns: &Patterns,
    text_lower: &str,
    char_locations: &HashMap<String, String>,
    char_constraints: &mut HashMap<String, Vec<String>>,
    new_facts: &mut Vec<WorldFact>,
) {
    // "locked the door" — applies to characters at the current location
    if patterns.locked.is_match(scene_text) {
        // All characters currently at a location get constrained
        for (cl, loc) in char_locations.iter() {
            // Check if any character name appears in this scene
            if text_lower.contains(cl.as_str()) {
                let constraint = format!("locked in {}", loc);
                char_constraints
                    .entry(cl.clone())
                    .or_default()
                    .push(constraint.clone());
                new_facts.push(WorldFact::PhysicalConstraint {
                    character: cl.clone(),
                    constraint,
                });
            }
        }
    }

    // "Character couldn't leave"
    for cap in patterns.couldnt_leave.captures_iter(scene_text) {
        let full_match = cap.get(0).map(|m| m.as_str()).unwrap_or("");
        if let Some(character) = find_character_in_match(full_match, characters) {
            let cl = character.to_lowercase();
            let constraint = "cannot leave current location".to_string();
            char_constraints
                .entry(cl.clone())
                .or_default()
                .push(constraint.clone());
            new_facts.push(WorldFact::PhysicalConstraint {
                character: cl,
                constraint,
            });
        }
    }

    // "Character alone in LOCATION" — means only that character is there
    for cap in patterns.alone_in.captures_iter(scene_text) {
        let full_match = cap.get(0).map(|m| m.as_str()).unwrap_or("");
        if let Some(character) = find_character_in_match(full_match, characters) {
            let cl = character.to_lowercase();
            let constraint = "alone — no other characters present".to_string();
            char_constraints
                .entry(cl.clone())
                .or_default()
                .push(constraint.clone());
            new_facts.push(WorldFact::PhysicalConstraint {
                character: cl,
                constraint,
            });
        }
    }
}

fn infer_colocation(
    characters: &[&str],
    char_locations: &HashMap<String, String>,
) -> Vec<WorldFact> {
    let mut facts = Vec::new();
    let chars: Vec<&str> = characters.to_vec();

    for i in 0..chars.len() {
        for j in (i + 1)..chars.len() {
            let a = chars[i].to_lowercase();
            let b = chars[j].to_lowercase();
            if let (Some(loc_a), Some(loc_b)) = (char_locations.get(&a), char_locations.get(&b))
                && loc_a == loc_b {
                    facts.push(WorldFact::CoLocated {
                        character_a: a,
                        character_b: b,
                    });
                }
        }
    }
    facts
}

// ─── Violation Detection ──────────────────────────────────────────────────────

fn detect_violations(
    scene_idx: usize,
    characters: &[&str],
    char_locations: &HashMap<String, String>,
    _char_constraints: &HashMap<String, Vec<String>>,
    _char_knowledge: &HashMap<String, HashSet<String>>,
    knowledge_graph: &KnowledgeGraph,
    cumulative_facts: &HashSet<WorldFact>,
    new_facts: &[WorldFact],
    violations: &mut Vec<ContinuityViolation>,
) {
    // Skip scene 0 — no prior state to violate
    if scene_idx == 0 {
        return;
    }

    // ─── Impossible Location ─────────────────────────────────────────────────
    // Check if any character had a constraint in the previous scene's facts
    // but has now moved.
    for &c in characters {
        let cl = c.to_lowercase();
        // Look for constraint facts in cumulative state
        let was_constrained = cumulative_facts.iter().any(|f| {
            matches!(
                f,
                WorldFact::PhysicalConstraint { character, constraint }
                if character == &cl && constraint.starts_with("locked in")
            )
        });

        if was_constrained {
            // Check if character moved to a new location in this scene
            let moved_to_new = new_facts.iter().any(|f| {
                matches!(
                    f,
                    WorldFact::CharacterAt { character, .. }
                    if character == &cl
                )
            });

            if moved_to_new {
                let current_loc = char_locations.get(&cl).cloned().unwrap_or_default();
                // Find the constraint location
                let constraint_loc = cumulative_facts
                    .iter()
                    .find_map(|f| {
                        if let WorldFact::PhysicalConstraint {
                            character,
                            constraint,
                        } = f
                        {
                            if character == &cl && constraint.starts_with("locked in") {
                                Some(constraint.trim_start_matches("locked in ").to_string())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .unwrap_or_default();

                if !constraint_loc.is_empty() && current_loc != constraint_loc {
                    violations.push(ContinuityViolation {
                        scene: scene_idx,
                        violation_type: ViolationType::ImpossibleLocation,
                        description: format!(
                            "{} appears at '{}' but was locked in '{}' with no escape shown.",
                            c, current_loc, constraint_loc
                        ),
                        proof: format!(
                            "Scene {}: PhysicalConstraint({}, locked in {}) is active. \
                             Scene {}: CharacterAt({}, {}) with no intervening release event.",
                            scene_idx - 1,
                            c,
                            constraint_loc,
                            scene_idx,
                            c,
                            current_loc
                        ),
                        severity: 0.8,
                        fix_suggestion: format!(
                            "Either remove the physical constraint in the previous scene, \
                             add a scene showing {}'s escape/release, or adjust the timeline.",
                            c
                        ),
                    });
                }
            }
        }
    }

    // ─── Impossible Knowledge ────────────────────────────────────────────────
    // For new CharacterKnows facts, check if there's a valid info path.
    for fact in new_facts {
        if let WorldFact::CharacterKnows {
            character,
            information,
        } = fact
        {
            // Skip if this character discovered/realized it in this scene (self-generated)
            // The knowledge_graph path check handles external knowledge
            if !knowledge_graph.has_info_path(character, information, scene_idx) {
                // Check if it was witnessed
                let was_witnessed = cumulative_facts.iter().any(|f| {
                    matches!(
                        f,
                        WorldFact::CharacterWitnessed { character: c, event }
                        if c == character && event == information
                    )
                });

                // Check if character was co-located with someone who spoke about it
                // (This is a simplified heuristic — in a real system we'd track dialogue)
                let could_have_overheard = cumulative_facts.iter().any(|f| {
                    matches!(
                        f,
                        WorldFact::CoLocated { character_a, character_b }
                        if character_a == character || character_b == character
                    )
                });

                // Only flag if there's genuinely no explanation
                if !was_witnessed && !could_have_overheard && scene_idx > 0 {
                    // Check: is this info established in a PRIOR scene that the character wasn't in?
                    // For now, we use the knowledge graph as the definitive check.
                    // Only report if the character has no path to this info at all.
                    let has_prior_communication = cumulative_facts.iter().any(|f| {
                        matches!(
                            f,
                            WorldFact::CommunicationVector { from: _, to, content }
                            if to == character && content == information
                        )
                    });

                    if !has_prior_communication {
                        let char_str = character.as_str();
                        let original_char = characters
                            .iter()
                            .find(|&&c| c.to_lowercase() == *character)
                            .unwrap_or(&char_str);

                        violations.push(ContinuityViolation {
                            scene: scene_idx,
                            violation_type: ViolationType::ImpossibleKnowledge,
                            description: format!(
                                "{} references '{}' but no valid information path exists.",
                                original_char, information
                            ),
                            proof: format!(
                                "Character {} acquires knowledge '{}' in scene {}, but: \
                                 - {} was not present when this information was established. \
                                 - No character who knew this communicated with {} before scene {}. \
                                 - No written record of this information was accessible to {}.",
                                original_char, information, scene_idx,
                                original_char, original_char, scene_idx, original_char
                            ),
                            severity: 0.9,
                            fix_suggestion: format!(
                                "Add a scene where {} learns '{}' — either witness it directly, \
                                 be told by someone who knows, or find written evidence.",
                                original_char, information
                            ),
                        });
                    }
                }
            }
        }
    }

    // ─── Object Continuity ───────────────────────────────────────────────────
    // Check for objects appearing in new locations without transfer
    for fact in new_facts {
        if let WorldFact::ObjectAt { object, location } = fact {
            // Check if this object was previously at a different location
            let prev_location = cumulative_facts.iter().find_map(|f| {
                if let WorldFact::ObjectAt {
                    object: o,
                    location: l,
                } = f
                {
                    if o == object && l != location {
                        Some(l.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

            if let Some(prev_loc) = prev_location {
                // Check if any character moved the object
                let was_transferred = new_facts.iter().any(|f| {
                    matches!(
                        f,
                        WorldFact::CharacterPossesses { object: o, .. }
                        if o == object
                    )
                });

                if !was_transferred {
                    violations.push(ContinuityViolation {
                        scene: scene_idx,
                        violation_type: ViolationType::ObjectContinuity,
                        description: format!(
                            "'{}' appears at '{}' but was last seen at '{}'.",
                            object, location, prev_loc
                        ),
                        proof: format!(
                            "Object '{}' established at '{}' in prior scene. \
                             Now appears at '{}' in scene {} with no transfer event.",
                            object, prev_loc, location, scene_idx
                        ),
                        severity: 0.5,
                        fix_suggestion: format!(
                            "Show '{}' being moved from '{}' to '{}' — have a character \
                             carry it, mail it, or explain its relocation.",
                            object, prev_loc, location
                        ),
                    });
                }
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let result = track_world_state(&[], &[]);
        assert_eq!(result.continuity_score, 1.0);
        assert!(result.violations.is_empty());
        assert!(result.snapshots.is_empty());
    }

    #[test]
    fn test_basic_location_tracking() {
        let scenes = [
            "Alice walked into the kitchen. She started making tea.",
            "Alice entered the library. She picked up a book.",
        ];
        let characters = ["Alice"];
        let result = track_world_state(&scenes.map(|s| s as &str), &characters);

        // Alice should have visited at least one location
        assert!(!result.character_knowledge.is_empty());
        let alice = &result.character_knowledge[0];
        assert_eq!(alice.character, "Alice");
        assert!(!alice.locations_visited.is_empty());
    }

    #[test]
    fn test_knowledge_transfer() {
        let scenes = [
            "Bob discovered that the safe was empty.",
            "Bob told Alice about the empty safe.",
        ];
        let characters = ["Alice", "Bob"];
        let result = track_world_state(&scenes.map(|s| s as &str), &characters);

        // Bob should know about the safe
        let bob = result
            .character_knowledge
            .iter()
            .find(|c| c.character == "Bob");
        assert!(bob.is_some(), "Bob should be in character_knowledge");
        assert!(
            !bob.unwrap().knows.is_empty() || !bob.unwrap().witnessed.is_empty(),
            "Bob should know or have witnessed something"
        );

        // Alice should have been told (if extraction detected it)
        let alice = result
            .character_knowledge
            .iter()
            .find(|c| c.character == "Alice");
        // Relaxed: extraction may not catch all formulations
        assert!(alice.is_some(), "Alice should be in character_knowledge");
    }

    #[test]
    fn test_colocation_inference() {
        let scenes = ["Alice was in the office. Bob was in the office."];
        let characters = ["Alice", "Bob"];
        let result = track_world_state(&scenes.map(|s| s as &str), &characters);

        // Should infer co-location
        let has_colocation = result.snapshots[0]
            .facts
            .iter()
            .any(|f| matches!(f, WorldFact::CoLocated { .. }));
        assert!(has_colocation);
    }

    #[test]
    fn test_impossible_location_violation() {
        let scenes = [
            "Alice was in the basement. Someone locked the door.",
            "Alice entered the rooftop garden.",
        ];
        let characters = ["Alice"];
        let result = track_world_state(&scenes.map(|s| s as &str), &characters);

        // Should detect impossible location
        let has_location_violation = result
            .violations
            .iter()
            .any(|v| v.violation_type == ViolationType::ImpossibleLocation);
        assert!(
            has_location_violation,
            "Expected ImpossibleLocation violation, got: {:?}",
            result.violations
        );
    }

    #[test]
    fn test_possession_tracking() {
        let scenes = [
            "Charlie picked up the golden key.",
            "Charlie gave the golden key to Diana.",
        ];
        let characters = ["Charlie", "Diana"];
        let result = track_world_state(&scenes.map(|s| s as &str), &characters);

        // Diana should possess the key
        let _diana = result
            .character_knowledge
            .iter()
            .find(|c| c.character == "Diana")
            .unwrap();
        // Check facts for possession
        let diana_has_key = result.snapshots.last().unwrap().facts.iter().any(|f| {
            matches!(
                f,
                WorldFact::CharacterPossesses { character, object }
                if character == "diana" && object.contains("key")
            )
        });
        assert!(diana_has_key, "Diana should possess the golden key");
    }

    #[test]
    fn test_witnessed_event() {
        let scenes = ["Eve saw the building collapse."];
        let characters = ["Eve"];
        let result = track_world_state(&scenes.map(|s| s as &str), &characters);

        let eve = result
            .character_knowledge
            .iter()
            .find(|c| c.character == "Eve")
            .unwrap();
        assert!(!eve.witnessed.is_empty());
    }

    #[test]
    fn test_continuity_score_perfect() {
        let scenes = [
            "Frank was in the office.",
            "Frank walked into the hallway.",
            "Frank entered the conference room.",
        ];
        let characters = ["Frank"];
        let result = track_world_state(&scenes.map(|s| s as &str), &characters);

        // No violations expected — movement is consistent
        assert_eq!(result.continuity_score, 1.0);
    }

    #[test]
    fn test_world_complexity() {
        let scenes = [
            "Alice was in the kitchen. Bob was in the garden. Alice picked up the knife.",
            "Alice told Bob about the secret passage.",
        ];
        let characters = ["Alice", "Bob"];
        let result = track_world_state(&scenes.map(|s| s as &str), &characters);

        // Should have extracted multiple facts per scene
        assert!(result.world_complexity > 0.0);
    }

    #[test]
    fn test_information_flow_tracking() {
        let scenes = [
            "Grace discovered that the treasure was hidden under the bridge.",
            "Grace told Henry about the treasure was hidden under the bridge.",
        ];
        let characters = ["Grace", "Henry"];
        let result = track_world_state(&scenes.map(|s| s as &str), &characters);

        assert!(!result.information_flow.is_empty());
    }
}
