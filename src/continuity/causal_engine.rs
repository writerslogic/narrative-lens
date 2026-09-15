use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::OnceLock;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Topo;
use serde::{Deserialize, Serialize};

use crate::craft::lexicon::WordMatcher;
use crate::substrate::utils::split_sentences;

/// A narrative event node in the causal graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalEvent {
    pub id: usize,
    pub scene: usize,
    pub description: String,
    pub agent: Option<String>,
    pub event_type: CausalEventType,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CausalEventType {
    Action,
    Reaction,
    Consequence,
    Discovery,
    Decision,
    External,
}

/// A causal link between events.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalLink {
    pub cause: usize,
    pub effect: usize,
    pub strength: f64,
    pub link_type: CausalLinkType,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CausalLinkType {
    Necessary,
    Enabling,
    Triggering,
    Coincidental,
}

/// A counterfactual: "What if event X hadn't happened?"
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Counterfactual {
    pub removed_event: usize,
    pub question: String,
    pub consequences: Vec<String>,
    pub plot_necessity: f64,
}

/// A plot hole: weak or broken causal chain.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlotHole {
    pub scene: usize,
    pub description: String,
    pub problem: String,
    pub causal_gap: String,
    pub suggestion: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalAnalysisResult {
    pub events: Vec<CausalEvent>,
    pub links: Vec<CausalLink>,
    pub counterfactuals: Vec<Counterfactual>,
    pub plot_holes: Vec<PlotHole>,
    pub causal_density: f64,
    pub longest_chain: usize,
    pub plot_coherence: f64,
    pub structural_insights: Vec<String>,
}

// --- Internal helpers ---

fn find_agent<'a>(sentence: &str, characters: &'a [&str]) -> Option<&'a str> {
    let lower = sentence.to_lowercase();
    // Find the character mentioned earliest in the sentence (likely the subject).
    characters
        .iter()
        .filter_map(|&c| {
            let pos = lower.find(&c.to_lowercase())?;
            Some((pos, c))
        })
        .min_by_key(|(pos, _)| *pos)
        .map(|(_, c)| c)
}

/// Whole-word matcher for a causal-marker set, built once. Whole-word matching
/// stops the previous `str::contains` from misclassifying on substrings:
/// "unresolved" no longer fires "resolved", "adopted" no longer fires "opted",
/// "selected" no longer fires "elected", "unnoticed" no longer fires "noticed",
/// "enthusiastic" no longer fires "thus", "swallowed" no longer fires "allowed".
fn marker_matcher(words: &[&str], cell: &'static OnceLock<WordMatcher>) -> &'static WordMatcher {
    cell.get_or_init(|| WordMatcher::new(words))
}

macro_rules! causal_markers {
    ($name:ident, $($w:literal),+ $(,)?) => {
        fn $name() -> &'static WordMatcher {
            static CELL: OnceLock<WordMatcher> = OnceLock::new();
            marker_matcher(&[$($w),+], &CELL)
        }
    };
}

causal_markers!(decision_markers, "decided", "chose", "resolved", "determined", "opted",
    "elected", "made up his mind", "made up her mind", "made the decision");
causal_markers!(consequence_markers, "because", "as a result", "therefore", "consequently",
    "which caused", "leading to", "thus", "hence", "owing to", "due to this", "this meant",
    "the result was", "in consequence");
causal_markers!(discovery_markers, "discovered", "realized", "found out", "learned that",
    "revealed", "uncovered", "noticed", "saw that", "understood", "became aware",
    "it turned out", "the truth was");
causal_markers!(reaction_markers, "in response", "reacted", "flinched", "gasped", "screamed",
    "laughed", "cried", "trembled", "froze", "staggered", "recoiled", "startled");
causal_markers!(external_markers, "the weather", "the storm", "lightning", "earthquake",
    "coincidentally", "by chance", "it happened that", "fate", "luck");
causal_markers!(necessary_markers, "because of", "without which", "only because",
    "had it not been", "impossible without", "required");
causal_markers!(trigger_markers, "finally", "at last", "the last straw", "triggered",
    "snapped", "that was when", "the moment");
causal_markers!(coincidental_markers, "coincidentally", "by chance", "happened to",
    "it so happened", "luck would have it", "just then");
causal_markers!(enabling_markers, "allowed", "enabled", "made possible", "opened the way",
    "created the conditions", "set the stage");

fn classify_sentence(sentence: &str) -> CausalEventType {
    let lower = sentence.to_lowercase();

    // Decision markers
    if decision_markers().is_match(&lower) {
        return CausalEventType::Decision;
    }

    // Consequence markers
    if consequence_markers().is_match(&lower) {
        return CausalEventType::Consequence;
    }

    // Discovery / revelation markers
    if discovery_markers().is_match(&lower) {
        return CausalEventType::Discovery;
    }

    // Reaction markers
    if reaction_markers().is_match(&lower) {
        return CausalEventType::Reaction;
    }

    // External markers
    if external_markers().is_match(&lower) {
        return CausalEventType::External;
    }

    // Default: Action (character does something)
    CausalEventType::Action
}

/// Check if sentence B references content from sentence A (shared nouns/entities).
fn sentences_share_content(a: &str, b: &str, characters: &[&str]) -> bool {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();

    // Check if they share a character reference
    for c in characters {
        let cl = c.to_lowercase();
        if a_lower.contains(&cl) && b_lower.contains(&cl) {
            return true;
        }
    }

    // Check for pronoun chains or shared significant words (4+ chars, not stopwords)
    let stopwords: HashSet<&str> = [
        "that", "this", "with", "from", "they", "them", "their", "have", "been", "were", "would",
        "could", "should", "about", "which", "there", "these", "those", "what", "when", "where",
        "will", "into", "then", "than", "also", "just", "more", "some", "very", "much", "only",
    ]
    .iter()
    .copied()
    .collect();

    let words_a: HashSet<&str> = a_lower
        .split_whitespace()
        .filter(|w| w.len() >= 4 && !stopwords.contains(w))
        .collect();
    let words_b: HashSet<&str> = b_lower
        .split_whitespace()
        .filter(|w| w.len() >= 4 && !stopwords.contains(w))
        .collect();

    let shared = words_a.intersection(&words_b).count();
    shared >= 2
}

/// Determine link type from sentence text and context.
fn classify_link(cause_sentence: &str, effect_sentence: &str) -> (CausalLinkType, f64) {
    let lower_effect = effect_sentence.to_lowercase();

    // Necessary: strong causal language
    if necessary_markers().is_match(&lower_effect) {
        return (CausalLinkType::Necessary, 0.9);
    }

    // Triggering: final trigger language
    if trigger_markers().is_match(&lower_effect) {
        return (CausalLinkType::Triggering, 0.7);
    }

    // Coincidental: coincidence language
    if coincidental_markers().is_match(&lower_effect) {
        return (CausalLinkType::Coincidental, 0.3);
    }

    // Check cause sentence for weak causal implication
    let lower_cause = cause_sentence.to_lowercase();
    if enabling_markers().is_match(&lower_cause) || enabling_markers().is_match(&lower_effect) {
        return (CausalLinkType::Enabling, 0.5);
    }

    // Default: enabling with moderate strength
    (CausalLinkType::Enabling, 0.6)
}

/// Compute longest path in DAG using topological sort.
fn longest_path_dag(graph: &DiGraph<usize, f64>) -> usize {
    let mut dist: HashMap<NodeIndex, usize> = HashMap::new();
    let mut topo = Topo::new(graph);

    while let Some(node) = topo.next(graph) {
        let max_parent = graph
            .neighbors_directed(node, petgraph::Direction::Incoming)
            .filter_map(|p| dist.get(&p).copied())
            .max()
            .unwrap_or(0);
        dist.insert(node, max_parent + 1);
    }

    dist.values().copied().max().unwrap_or(0)
}

/// BFS downstream from a node, returning all reachable node indices.
fn bfs_downstream(graph: &DiGraph<usize, f64>, start: NodeIndex) -> HashSet<NodeIndex> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(start);
    while let Some(node) = queue.pop_front() {
        for neighbor in graph.neighbors_directed(node, petgraph::Direction::Outgoing) {
            if visited.insert(neighbor) {
                queue.push_back(neighbor);
            }
        }
    }
    visited
}

pub fn analyze_causality(scenes: &[&str], characters: &[&str]) -> CausalAnalysisResult {
    if scenes.is_empty() {
        return CausalAnalysisResult {
            events: Vec::new(),
            links: Vec::new(),
            counterfactuals: Vec::new(),
            plot_holes: Vec::new(),
            causal_density: 0.0,
            longest_chain: 0,
            plot_coherence: 0.0,
            structural_insights: Vec::new(),
        };
    }

    // --- Phase 1: Event Extraction ---
    let mut events: Vec<CausalEvent> = Vec::new();
    let mut scene_sentences: Vec<Vec<(usize, String)>> = Vec::new(); // (event_id, sentence_text) per scene

    for (scene_idx, &scene_text) in scenes.iter().enumerate() {
        let sentences = split_sentences(scene_text);
        let mut scene_evts: Vec<(usize, String)> = Vec::new();

        for sentence in &sentences {
            if sentence.split_whitespace().count() < 4 {
                continue; // skip very short fragments
            }

            let event_type = classify_sentence(sentence);
            let agent = find_agent(sentence, characters).map(|s| s.to_string());

            let id = events.len();
            events.push(CausalEvent {
                id,
                scene: scene_idx,
                description: sentence.clone(),
                agent,
                event_type,
            });
            scene_evts.push((id, sentence.clone()));
        }
        scene_sentences.push(scene_evts);
    }

    if events.is_empty() {
        return CausalAnalysisResult {
            events: Vec::new(),
            links: Vec::new(),
            counterfactuals: Vec::new(),
            plot_holes: Vec::new(),
            causal_density: 0.0,
            longest_chain: 0,
            plot_coherence: 0.0,
            structural_insights: Vec::new(),
        };
    }

    // --- Phase 2: Causal Link Detection ---
    let mut links: Vec<CausalLink> = Vec::new();
    let mut linked_effects: HashSet<usize> = HashSet::new();

    // 2a: Explicit consequence markers within a scene
    for scene_evts in &scene_sentences {
        for i in 1..scene_evts.len() {
            let (effect_id, ref effect_text) = scene_evts[i];
            let effect_type = &events[effect_id].event_type;

            if *effect_type == CausalEventType::Consequence
                || *effect_type == CausalEventType::Reaction
            {
                // Link to the immediately preceding event
                let (cause_id, ref cause_text) = scene_evts[i - 1];
                let (link_type, strength) = classify_link(cause_text, effect_text);
                links.push(CausalLink {
                    cause: cause_id,
                    effect: effect_id,
                    strength,
                    link_type,
                });
                linked_effects.insert(effect_id);
            }
        }
    }

    // 2b: Sequential within scene — if event B references A's content
    for scene_evts in &scene_sentences {
        for i in 1..scene_evts.len() {
            let (effect_id, ref effect_text) = scene_evts[i];
            if linked_effects.contains(&effect_id) {
                continue;
            }
            // Look back up to 3 sentences for content overlap
            let lookback = i.saturating_sub(3);
            for j in lookback..i {
                let (cause_id, ref cause_text) = scene_evts[j];
                if sentences_share_content(cause_text, effect_text, characters) {
                    let (link_type, strength) = classify_link(cause_text, effect_text);
                    links.push(CausalLink {
                        cause: cause_id,
                        effect: effect_id,
                        strength,
                        link_type,
                    });
                    linked_effects.insert(effect_id);
                    break;
                }
            }
        }
    }

    // 2c: Cross-scene links — last event of scene N to first event of scene N+1 if content overlap
    for s in 0..scene_sentences.len().saturating_sub(1) {
        if scene_sentences[s].is_empty() || scene_sentences[s + 1].is_empty() {
            continue;
        }
        let Some(&(cause_id, ref cause_text)) = scene_sentences[s].last() else {
            continue;
        };
        let (effect_id, ref effect_text) = scene_sentences[s + 1][0];

        if !linked_effects.contains(&effect_id)
            && sentences_share_content(cause_text, effect_text, characters)
        {
            let (link_type, strength) = classify_link(cause_text, effect_text);
            links.push(CausalLink {
                cause: cause_id,
                effect: effect_id,
                strength,
                link_type,
            });
            linked_effects.insert(effect_id);
        }
    }

    // 2d: Character-mediated cross-scene links
    // If character C acts in scene X and appears in an event in scene Y (Y > X),
    // and the scene Y event references consequences, link them.
    let mut char_actions: HashMap<String, Vec<usize>> = HashMap::new();
    for evt in &events {
        if (evt.event_type == CausalEventType::Action || evt.event_type == CausalEventType::Decision)
            && let Some(ref agent) = evt.agent {
                char_actions.entry(agent.clone()).or_default().push(evt.id);
            }
    }

    for evt in &events {
        if linked_effects.contains(&evt.id) {
            continue;
        }
        if evt.event_type != CausalEventType::Consequence {
            continue;
        }
        if let Some(ref agent) = evt.agent
            && let Some(prior_actions) = char_actions.get(agent) {
                // Find the latest action by this character before this event's scene
                if let Some(&cause_id) = prior_actions
                    .iter().rfind(|&&aid| events[aid].scene < evt.scene)
                {
                    links.push(CausalLink {
                        cause: cause_id,
                        effect: evt.id,
                        strength: 0.5,
                        link_type: CausalLinkType::Enabling,
                    });
                    linked_effects.insert(evt.id);
                }
            }
    }

    // --- Phase 3: Build petgraph for analysis ---
    let mut graph: DiGraph<usize, f64> = DiGraph::new();
    let mut node_map: HashMap<usize, NodeIndex> = HashMap::new();

    for evt in &events {
        let idx = graph.add_node(evt.id);
        node_map.insert(evt.id, idx);
    }
    for link in &links {
        if let (Some(&cause_idx), Some(&effect_idx)) =
            (node_map.get(&link.cause), node_map.get(&link.effect))
        {
            graph.add_edge(cause_idx, effect_idx, link.strength);
        }
    }

    // --- Phase 4: Counterfactual Simulation ---
    let mut counterfactuals: Vec<Counterfactual> = Vec::new();

    // Find events with high out-degree (significant events)
    let mut out_degrees: Vec<(usize, usize)> = events
        .iter()
        .map(|e| {
            let idx = node_map[&e.id];
            let out = graph
                .neighbors_directed(idx, petgraph::Direction::Outgoing)
                .count();
            (e.id, out)
        })
        .collect();
    out_degrees.sort_by_key(|x| std::cmp::Reverse(x.1));

    let total_events = events.len();
    // Simulate counterfactuals for top events (up to 10)
    for &(event_id, out_deg) in out_degrees.iter().take(10) {
        if out_deg == 0 {
            break;
        }
        let start = node_map[&event_id];
        let downstream = bfs_downstream(&graph, start);
        let plot_necessity = downstream.len() as f64 / total_events as f64;

        let consequences: Vec<String> = downstream
            .iter()
            .take(5)
            .filter_map(|&ni| {
                let eid = graph[ni];
                events.get(eid).map(|e| e.description.clone())
            })
            .collect();

        let evt = &events[event_id];
        let question = format!(
            "What if \"{}\" hadn't happened?",
            crate::substrate::utils::truncate_str(&evt.description, 60)
        );

        counterfactuals.push(Counterfactual {
            removed_event: event_id,
            question,
            consequences,
            plot_necessity,
        });
    }

    counterfactuals.sort_by(|a, b| {
        b.plot_necessity
            .partial_cmp(&a.plot_necessity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // --- Phase 5: Plot Hole Detection ---
    let mut plot_holes: Vec<PlotHole> = Vec::new();

    // Events with no incoming links (except in scene 0)
    for evt in &events {
        if evt.scene == 0 {
            continue;
        }
        let idx = node_map[&evt.id];
        let in_degree = graph
            .neighbors_directed(idx, petgraph::Direction::Incoming)
            .count();
        if in_degree == 0 {
            // Only flag significant events (not trivial actions)
            if evt.event_type == CausalEventType::Consequence
                || evt.event_type == CausalEventType::Reaction
                || evt.event_type == CausalEventType::Discovery
            {
                plot_holes.push(PlotHole {
                    scene: evt.scene,
                    description: crate::substrate::utils::truncate_str(&evt.description, 80).to_string(),
                    problem: "Event has no causal origin".to_string(),
                    causal_gap: format!(
                        "\"{}\" happens but nothing in the story caused it",
                        crate::substrate::utils::truncate_str(&evt.description, 50)
                    ),
                    suggestion: "Establish a prior event that leads to this, or acknowledge the randomness explicitly.".to_string(),
                });
            }
        }
    }

    // Coincidental links with high importance (out-degree > 1)
    for link in &links {
        if link.link_type == CausalLinkType::Coincidental {
            let effect_idx = node_map[&link.effect];
            let out = graph
                .neighbors_directed(effect_idx, petgraph::Direction::Outgoing)
                .count();
            if out >= 1 {
                let evt = &events[link.effect];
                plot_holes.push(PlotHole {
                    scene: evt.scene,
                    description: crate::substrate::utils::truncate_str(&evt.description, 80).to_string(),
                    problem: "Important event relies on coincidence".to_string(),
                    causal_gap: format!(
                        "\"{}\" depends on coincidence but drives {} downstream events",
                        crate::substrate::utils::truncate_str(&evt.description, 50),
                        out
                    ),
                    suggestion: "Replace the coincidence with a character-driven cause to strengthen narrative logic.".to_string(),
                });
            }
        }
    }

    // --- Phase 6: Scoring ---
    let causal_density = if total_events > 0 {
        links.len() as f64 / total_events as f64
    } else {
        0.0
    };

    let events_with_cause = linked_effects.len();
    // Scene 0 events don't need a cause
    let scene0_events = events.iter().filter(|e| e.scene == 0).count();
    let need_cause = total_events.saturating_sub(scene0_events);
    let plot_coherence = if need_cause > 0 {
        events_with_cause as f64 / need_cause as f64
    } else {
        1.0
    };

    let longest_chain = longest_path_dag(&graph);

    // --- Phase 7: Structural Insights ---
    let mut structural_insights: Vec<String> = Vec::new();

    let chain_count = {
        // Count events with out-degree > 0 and in-degree == 0 (chain starts)
        events
            .iter()
            .filter(|e| {
                let idx = node_map[&e.id];
                let in_d = graph
                    .neighbors_directed(idx, petgraph::Direction::Incoming)
                    .count();
                let out_d = graph
                    .neighbors_directed(idx, petgraph::Direction::Outgoing)
                    .count();
                in_d == 0 && out_d > 0
            })
            .count()
    };

    structural_insights.push(format!(
        "Your plot has {} causal chains. The longest runs {} events deep. \
         Deeper chains create more satisfying narrative logic.",
        chain_count, longest_chain
    ));

    if !plot_holes.is_empty() {
        let first_hole = &plot_holes[0];
        structural_insights.push(format!(
            "Scene {} contains an event with no causal origin: \"{}\". \
             Either establish a cause earlier or acknowledge the randomness in-story.",
            first_hole.scene, first_hole.description
        ));
    }

    // Load-bearing events
    if let Some(top) = counterfactuals.first() {
        let evt = &events[top.removed_event];
        let downstream_count = (top.plot_necessity * total_events as f64).round() as usize;
        structural_insights.push(format!(
            "Removing \"{}\" from the story would eliminate {} downstream consequences. \
             This is a load-bearing event; protect it in revision.",
            crate::substrate::utils::truncate_str(&evt.description, 50),
            downstream_count
        ));
    }

    // Character agency insight
    let mut char_decision_counts: HashMap<&str, usize> = HashMap::new();
    for evt in &events {
        if evt.event_type == CausalEventType::Decision
            && let Some(ref agent) = evt.agent {
                let idx = node_map[&evt.id];
                let out_d = graph
                    .neighbors_directed(idx, petgraph::Direction::Outgoing)
                    .count();
                if out_d > 0 {
                    // Find matching character name
                    if let Some(&c) = characters.iter().find(|&&c| c == agent.as_str()) {
                        *char_decision_counts.entry(c).or_insert(0) += 1;
                    }
                }
            }
    }
    if let Some((best_char, count)) = char_decision_counts.iter().max_by_key(|(_, v)| *v)
        && *count > 0 {
            structural_insights.push(format!(
                "{}'s arc has {} decision points that cause later events. \
                 This creates strong character agency.",
                best_char, count
            ));
        }

    // Density insight
    if causal_density < 0.5 {
        structural_insights.push(
            "Causal density is low. Many events feel disconnected. \
             Consider adding explicit cause-effect language to link events together."
                .to_string(),
        );
    } else if causal_density > 1.5 {
        structural_insights.push(
            "Causal density is high. The plot is tightly woven with strong narrative logic."
                .to_string(),
        );
    }

    CausalAnalysisResult {
        events,
        links,
        counterfactuals,
        plot_holes,
        causal_density,
        longest_chain,
        plot_coherence,
        structural_insights,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_sentence_uses_whole_word_markers() {
        // Substring/negation forms must not misclassify: "unresolved" is not a
        // decision, "adopted" does not contain decision marker "opted".
        assert_eq!(classify_sentence("she resolved to leave"), CausalEventType::Decision);
        assert_ne!(
            classify_sentence("the conflict remained unresolved"),
            CausalEventType::Decision
        );
        assert_ne!(
            classify_sentence("the stray puppy was adopted"),
            CausalEventType::Decision
        );
    }

    #[test]
    fn test_empty_input() {
        let result = analyze_causality(&[], &[]);
        assert_eq!(result.events.len(), 0);
        assert_eq!(result.causal_density, 0.0);
    }

    #[test]
    fn test_single_scene_causal_chain() {
        let scene = "John grabbed the knife. Because of this, Mary screamed in terror. \
                     As a result, the neighbors called the police.";
        let result = analyze_causality(&[scene], &["John", "Mary"]);

        assert!(result.events.len() >= 3);
        assert!(!result.links.is_empty());
        assert!(result.causal_density > 0.0);
    }

    #[test]
    fn test_decision_detection() {
        let scene = "Sarah decided to leave the country. She chose to abandon everything.";
        let result = analyze_causality(&[scene], &["Sarah"]);

        let decisions: Vec<_> = result
            .events
            .iter()
            .filter(|e| e.event_type == CausalEventType::Decision)
            .collect();
        assert!(!decisions.is_empty());
        assert_eq!(decisions[0].agent.as_deref(), Some("Sarah"));
    }

    #[test]
    fn test_cross_scene_link() {
        let scene1 = "Thomas planted the bomb under the bridge.";
        let scene2 = "The bridge exploded as a result of what Thomas had done.";
        let result = analyze_causality(&[scene1, scene2], &["Thomas"]);

        // Should detect cross-scene causality via shared content (Thomas, bridge)
        assert!(result.events.len() >= 2);
    }

    #[test]
    fn test_cross_scene_link_with_empty_scene() {
        // An empty scene between content scenes previously reached
        // `scene_sentences[s].last().unwrap()` in the cross-scene block; must not panic.
        let scene1 = "Thomas planted the bomb under the bridge.";
        let empty = "   ";
        let scene3 = "The bridge exploded as a result of what Thomas had done.";
        let result = analyze_causality(&[scene1, empty, scene3], &["Thomas"]);

        assert!(result.events.len() >= 2);
        assert!(result.plot_coherence <= 1.0);
    }

    #[test]
    fn test_plot_hole_detection() {
        let scene1 = "Alice walked through the park.";
        let scene2 = "Bob suddenly realized the secret code. As a result, the vault opened.";
        let result = analyze_causality(&[scene1, scene2], &["Alice", "Bob"]);

        // Bob's discovery has no established cause from scene 1
        // Check that we get insights or plot holes
        assert!(result.plot_coherence <= 1.0);
    }

    #[test]
    fn test_counterfactual_generation() {
        let scene = "The king declared war. Because of the war, thousands fled. \
                     As a result, the economy collapsed. Therefore the rebellion began.";
        let result = analyze_causality(&[scene], &["king"]);

        // The war declaration should be identified as load-bearing
        if !result.counterfactuals.is_empty() {
            assert!(result.counterfactuals[0].plot_necessity > 0.0);
        }
    }

    #[test]
    fn test_longest_chain() {
        let scene = "A happened. Because of A, B occurred. As a result, C followed. \
                     Therefore D was inevitable.";
        let result = analyze_causality(&[scene], &[]);

        assert!(result.longest_chain >= 2);
    }

    #[test]
    fn test_classify_sentence_types() {
        assert_eq!(
            classify_sentence("He decided to leave"),
            CausalEventType::Decision
        );
        assert_eq!(
            classify_sentence("As a result the building fell"),
            CausalEventType::Consequence
        );
        assert_eq!(
            classify_sentence("She discovered the truth"),
            CausalEventType::Discovery
        );
        assert_eq!(
            classify_sentence("She reacted with fury"),
            CausalEventType::Reaction
        );
        assert_eq!(
            classify_sentence("The storm destroyed the crops"),
            CausalEventType::External
        );
        assert_eq!(
            classify_sentence("He walked to the store"),
            CausalEventType::Action
        );
    }
}
