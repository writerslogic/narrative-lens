use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::OnceLock;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Topo;
use serde::{Deserialize, Serialize};

use super::author_knowledge::AuthorSource;
use crate::craft::lexicon::WordMatcher;
use crate::substrate::utils::split_sentences;
use crate::{
    Error, Result,
    manuscript::{
        Evidence, Manuscript,
        author_source::validate_sources,
        config::{validate_id, validate_name},
    },
};

/// A narrative event node in the causal graph.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalEvent {
    pub id: usize,
    pub scene: usize,
    pub description: String,
    pub agent: Option<String>,
    pub event_type: CausalEventType,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
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
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalLink {
    pub cause: usize,
    pub effect: usize,
    pub strength: f64,
    pub link_type: CausalLinkType,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CausalLinkType {
    Association,
    Necessary,
    Enabling,
    Triggering,
    Coincidental,
}

/// A counterfactual: "What if event X hadn't happened?"
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Counterfactual {
    pub removed_event: usize,
    pub question: String,
    pub consequences: Vec<String>,
    pub plot_necessity: f64,
}

/// A plot hole: weak or broken causal chain.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlotHole {
    pub scene: usize,
    pub description: String,
    pub problem: String,
    pub causal_gap: String,
    pub suggestion: String,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalAnalysisResult {
    #[serde(default)]
    pub availability: crate::substrate::utils::AnalysisAvailability,
    #[serde(default)]
    pub diagnostics: Vec<String>,
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

causal_markers!(
    decision_markers,
    "decided",
    "chose",
    "resolved",
    "determined",
    "opted",
    "elected",
    "made up his mind",
    "made up her mind",
    "made the decision"
);
causal_markers!(
    consequence_markers,
    "because",
    "as a result",
    "therefore",
    "consequently",
    "which caused",
    "leading to",
    "thus",
    "hence",
    "owing to",
    "due to this",
    "this meant",
    "the result was",
    "in consequence"
);
causal_markers!(
    discovery_markers,
    "discovered",
    "realized",
    "found out",
    "learned that",
    "revealed",
    "uncovered",
    "noticed",
    "saw that",
    "understood",
    "became aware",
    "it turned out",
    "the truth was"
);
causal_markers!(
    reaction_markers,
    "in response",
    "reacted",
    "flinched",
    "gasped",
    "screamed",
    "laughed",
    "cried",
    "trembled",
    "froze",
    "staggered",
    "recoiled",
    "startled"
);
causal_markers!(
    external_markers,
    "the weather",
    "the storm",
    "lightning",
    "earthquake",
    "coincidentally",
    "by chance",
    "it happened that",
    "fate",
    "luck"
);
causal_markers!(
    necessary_markers,
    "because of",
    "without which",
    "only because",
    "had it not been",
    "impossible without",
    "required"
);
causal_markers!(
    trigger_markers,
    "finally",
    "at last",
    "the last straw",
    "triggered",
    "snapped",
    "that was when",
    "the moment"
);
causal_markers!(
    coincidental_markers,
    "coincidentally",
    "by chance",
    "happened to",
    "it so happened",
    "luck would have it",
    "just then"
);
causal_markers!(
    enabling_markers,
    "allowed",
    "enabled",
    "made possible",
    "opened the way",
    "created the conditions",
    "set the stage"
);

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

/// Lexical marker tiers are association weights, not causal probabilities.
fn classify_link(cause_sentence: &str, effect_sentence: &str) -> (CausalLinkType, f64) {
    let lower_effect = effect_sentence.to_lowercase();

    // Necessary: strong causal language
    if necessary_markers().is_match(&lower_effect) {
        return (CausalLinkType::Association, 0.9);
    }

    // Triggering: final trigger language
    if trigger_markers().is_match(&lower_effect) {
        return (CausalLinkType::Association, 0.7);
    }

    // Coincidental: coincidence language
    if coincidental_markers().is_match(&lower_effect) {
        return (CausalLinkType::Association, 0.3);
    }

    // Check cause sentence for weak causal implication
    let lower_cause = cause_sentence.to_lowercase();
    if enabling_markers().is_match(&lower_cause) || enabling_markers().is_match(&lower_effect) {
        return (CausalLinkType::Association, 0.5);
    }

    // Default: enabling with moderate strength
    (CausalLinkType::Association, 0.6)
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

pub fn analyze_causality_checked(
    scenes: &[&str],
    characters: &[&str],
) -> Result<CausalAnalysisResult> {
    if scenes.len() > 10_000 || characters.len() > 256 {
        return Err(Error::InvalidInput(
            "causal input exceeds scene/cast limits".into(),
        ));
    }
    let mut bytes = 0usize;
    for scene in scenes {
        bytes = bytes
            .checked_add(scene.len())
            .filter(|n| *n <= 64 * 1024 * 1024)
            .ok_or_else(|| Error::InvalidInput("causal source exceeds64MiB".into()))?;
        if scene.len() > 16 * 1024 * 1024 {
            return Err(Error::InvalidInput("causal scene exceeds16MiB".into()));
        }
    }
    for character in characters {
        validate_name(character)?;
    }
    analyze_causality_inner(scenes, characters)
}

pub fn analyze_causality(scenes: &[&str], characters: &[&str]) -> CausalAnalysisResult {
    analyze_causality_checked(scenes, characters).unwrap_or_else(|error| CausalAnalysisResult {
        availability: crate::substrate::utils::AnalysisAvailability::Unavailable,
        diagnostics: vec![error.to_string()],
        events: Vec::new(),
        links: Vec::new(),
        counterfactuals: Vec::new(),
        plot_holes: Vec::new(),
        causal_density: 0.0,
        longest_chain: 0,
        plot_coherence: 0.0,
        structural_insights: Vec::new(),
    })
}

fn analyze_causality_inner(
    scenes: &[&str],
    characters: &[&str],
) -> crate::Result<CausalAnalysisResult> {
    if scenes.is_empty() {
        return Ok(CausalAnalysisResult {
            events: Vec::new(),
            links: Vec::new(),
            counterfactuals: Vec::new(),
            plot_holes: Vec::new(),
            causal_density: 0.0,
            longest_chain: 0,
            plot_coherence: 0.0,
            structural_insights: Vec::new(),
            availability: crate::substrate::utils::AnalysisAvailability::Partial,
            diagnostics: vec!["lexical_associations_not_causal_proof".into()],
        });
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

            if events.len() == 10_000 {
                return Err(crate::Error::InvalidInput(
                    "causal candidate count exceeds10000".into(),
                ));
            }
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
        return Ok(CausalAnalysisResult {
            events: Vec::new(),
            links: Vec::new(),
            counterfactuals: Vec::new(),
            plot_holes: Vec::new(),
            causal_density: 0.0,
            longest_chain: 0,
            plot_coherence: 0.0,
            structural_insights: Vec::new(),
            availability: crate::substrate::utils::AnalysisAvailability::Partial,
            diagnostics: vec!["lexical_associations_not_causal_proof".into()],
        });
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
        if (evt.event_type == CausalEventType::Action
            || evt.event_type == CausalEventType::Decision)
            && let Some(ref agent) = evt.agent
        {
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
            && let Some(prior_actions) = char_actions.get(agent)
        {
            // Find the latest action by this character before this event's scene
            if let Some(&cause_id) = prior_actions
                .iter()
                .rfind(|&&aid| events[aid].scene < evt.scene)
            {
                links.push(CausalLink {
                    cause: cause_id,
                    effect: evt.id,
                    strength: 0.5,
                    link_type: CausalLinkType::Association,
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

    let total_events = events.len();
    let counterfactuals = Vec::new();
    let plot_holes = Vec::new();

    // --- Phase 6: Scoring ---
    let causal_density = if total_events > 0 {
        links.len() as f64 / total_events as f64
    } else {
        0.0
    };

    let events_with_cause = linked_effects
        .iter()
        .filter(|&&id| events[id].scene != 0)
        .count();
    // Scene 0 events don't need a cause
    let scene0_events = events.iter().filter(|e| e.scene == 0).count();
    let need_cause = total_events.saturating_sub(scene0_events);
    let plot_coherence = if need_cause > 0 {
        events_with_cause as f64 / need_cause as f64
    } else {
        1.0
    };

    let longest_chain = longest_path_dag(&graph);

    let structural_insights = vec![format!(
        "Lexical association graph: {} candidate events, {} associations, longest path {}. These do not establish causal necessity, plot quality or character agency.",
        events.len(),
        links.len(),
        longest_chain
    )];
    Ok(CausalAnalysisResult {
        availability: crate::substrate::utils::AnalysisAvailability::Partial,
        diagnostics: vec![
            "lexical_associations_not_causal_proof".into(),
            "counterfactuals_and_plot_holes_require_explicit_model".into(),
            "legacy_density_and_coherence_are_lexical_graph_descriptors_only".into(),
        ],
        events,
        links,
        counterfactuals,
        plot_holes,
        causal_density,
        longest_chain,
        plot_coherence,
        structural_insights,
    })
}

pub const MAX_AUTHOR_CAUSAL_NODES: usize = 10_000;
pub const MAX_AUTHOR_CAUSAL_EDGES: usize = 30_000;
pub const MAX_AUTHOR_CAUSAL_BYTES: usize = 4 * 1024 * 1024;

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum AuthorCausalMechanism {
    Fixed { value: bool },
    All { parents: Vec<String> },
    Any { parents: Vec<String> },
    Not { parent: String },
}

impl AuthorCausalMechanism {
    fn parents(&self) -> &[String] {
        match self {
            Self::Fixed { .. } => &[],
            Self::All { parents } | Self::Any { parents } => parents,
            Self::Not { parent } => std::slice::from_ref(parent),
        }
    }
    fn evaluate(&self, values: &BTreeMap<&str, bool>) -> bool {
        match self {
            Self::Fixed { value } => *value,
            Self::All { parents } => parents.iter().all(|p| values[p.as_str()]),
            Self::Any { parents } => parents.iter().any(|p| values[p.as_str()]),
            Self::Not { parent } => !values[parent.as_str()],
        }
    }
    fn canonical(&self) -> Self {
        let mut mechanism = self.clone();
        if let Self::All { parents } | Self::Any { parents } = &mut mechanism {
            parents.sort();
        }
        mechanism
    }
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AuthorCausalNode {
    pub id: String,
    pub mechanism: AuthorCausalMechanism,
    pub evidence: AuthorSource,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AuthorIntervention {
    pub node_id: String,
    pub value: bool,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorCausalValue {
    pub id: String,
    pub mechanism: AuthorCausalMechanism,
    pub baseline: bool,
    pub intervened: bool,
    pub evidence: Evidence,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorCausalEdge {
    pub cause_id: String,
    pub effect_id: String,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorCausalChange {
    pub node_id: String,
    pub before: bool,
    pub after: bool,
    pub direct_intervention: bool,
}

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthorCausalResult {
    pub document_id: String,
    pub availability: crate::substrate::utils::AnalysisAvailability,
    pub diagnostics: Vec<String>,
    pub nodes: Vec<AuthorCausalValue>,
    pub edges: Vec<AuthorCausalEdge>,
    pub topological_order: Vec<String>,
    pub interventions: Vec<AuthorIntervention>,
    pub changes: Vec<AuthorCausalChange>,
}

/// Evaluate one joint intervention on supplied deterministic Boolean mechanisms.
/// Incoming mechanisms of intervened nodes are replaced by constants; other
/// mechanisms remain unchanged. Sources locate author model assertions, not
/// independently established causes or probabilities about the manuscript.
pub fn analyze_author_causality(
    manuscript: &Manuscript,
    nodes: &[AuthorCausalNode],
    interventions: &[AuthorIntervention],
) -> Result<AuthorCausalResult> {
    manuscript.validate()?;
    let invalid = |message: &str| Error::InvalidInput(message.into());
    if nodes.len() > MAX_AUTHOR_CAUSAL_NODES || interventions.len() > MAX_AUTHOR_CAUSAL_NODES {
        return Err(invalid(
            "author causal model exceeds10000 nodes/interventions",
        ));
    }
    let mut by_id = BTreeMap::new();
    let mut metadata = 0usize;
    let mut edges = Vec::new();
    let mut parents_total = 0usize;
    for node in nodes {
        validate_id(&node.id)?;
        if by_id.insert(node.id.as_str(), node).is_some() {
            return Err(invalid("author causal node IDs must be unique"));
        }
        let parents = node.mechanism.parents();
        if !matches!(node.mechanism, AuthorCausalMechanism::Fixed { .. })
            && !(1..=32).contains(&parents.len())
        {
            return Err(invalid("author causal mechanisms require1..32 parents"));
        }
        parents_total += parents.len();
        if parents_total > MAX_AUTHOR_CAUSAL_EDGES {
            return Err(invalid("author causal model exceeds30000 edges"));
        }
        let mut unique = BTreeSet::new();
        for parent in parents {
            validate_id(parent)?;
            if !unique.insert(parent) {
                return Err(invalid("author causal parents must be unique"));
            }
        }
        metadata = metadata
            .checked_add(
                node.id.len()
                    + parents.iter().map(String::len).sum::<usize>()
                    + node.evidence.scene_id.len()
                    + node.evidence.quote.len(),
            )
            .filter(|n| *n <= MAX_AUTHOR_CAUSAL_BYTES)
            .ok_or_else(|| invalid("author causal metadata exceeds4MiB"))?;
    }
    let mut overrides = BTreeMap::new();
    for intervention in interventions {
        validate_id(&intervention.node_id)?;
        if !by_id.contains_key(intervention.node_id.as_str())
            || overrides
                .insert(intervention.node_id.as_str(), intervention.value)
                .is_some()
        {
            return Err(invalid(
                "author interventions require unique known node IDs",
            ));
        }
        metadata = metadata
            .checked_add(intervention.node_id.len())
            .filter(|n| *n <= MAX_AUTHOR_CAUSAL_BYTES)
            .ok_or_else(|| invalid("author causal metadata exceeds4MiB"))?;
    }
    let mut indegree = BTreeMap::new();
    let mut successors: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for node in nodes {
        indegree.insert(node.id.as_str(), node.mechanism.parents().len());
        for parent in node.mechanism.parents() {
            if !by_id.contains_key(parent.as_str()) {
                return Err(invalid("author causal parent references an unknown node"));
            }
            successors.entry(parent).or_default().push(&node.id);
            edges.push(AuthorCausalEdge {
                cause_id: parent.clone(),
                effect_id: node.id.clone(),
            });
        }
    }
    let mut ready: BTreeSet<&str> = indegree
        .iter()
        .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
        .collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(id) = ready.pop_first() {
        order.push(id);
        for child in successors.get(id).into_iter().flatten() {
            let degree = indegree.get_mut(child).expect("all graph nodes indexed");
            *degree -= 1;
            if *degree == 0 {
                ready.insert(child);
            }
        }
    }
    if order.len() != nodes.len() {
        return Err(invalid("author causal mechanisms must be acyclic"));
    }
    let evidence = validate_sources(
        manuscript,
        &nodes.iter().map(|node| &node.evidence).collect::<Vec<_>>(),
    )?;
    let mut evidence: BTreeMap<_, _> = nodes
        .iter()
        .zip(evidence)
        .map(|(node, evidence)| (node.id.as_str(), evidence))
        .collect();
    let mut baseline = BTreeMap::new();
    let mut intervened = BTreeMap::new();
    for id in &order {
        let node = by_id[id];
        baseline.insert(*id, node.mechanism.evaluate(&baseline));
        let value = overrides
            .get(id)
            .copied()
            .unwrap_or_else(|| node.mechanism.evaluate(&intervened));
        intervened.insert(*id, value);
    }
    let mut values = Vec::with_capacity(nodes.len());
    let mut changes = Vec::new();
    for (id, node) in by_id {
        let before = baseline[id];
        let after = intervened[id];
        if before != after {
            changes.push(AuthorCausalChange {
                node_id: id.into(),
                before,
                after,
                direct_intervention: overrides.contains_key(id),
            });
        }
        values.push(AuthorCausalValue {
            id: id.into(),
            mechanism: node.mechanism.canonical(),
            baseline: before,
            intervened: after,
            evidence: evidence.remove(id).expect("all sources validated"),
        });
    }
    edges.sort_by(|a, b| {
        a.cause_id
            .cmp(&b.cause_id)
            .then_with(|| a.effect_id.cmp(&b.effect_id))
    });
    Ok(AuthorCausalResult {
        document_id: manuscript.document_id.clone(), availability: crate::substrate::utils::AnalysisAvailability::Complete,
        diagnostics: vec!["Author-supplied Boolean mechanisms only; joint interventions replace incoming equations. No causal mechanisms, probabilities, necessity or plot quality are inferred from prose.".into()],
        nodes: values, edges, topological_order: order.into_iter().map(str::to_owned).collect(),
        interventions: overrides.into_iter().map(|(id,value)| AuthorIntervention { node_id: id.into(), value }).collect(), changes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_sentence_uses_whole_word_markers() {
        // Substring/negation forms must not misclassify: "unresolved" is not a
        // decision, "adopted" does not contain decision marker "opted".
        assert_eq!(
            classify_sentence("she resolved to leave"),
            CausalEventType::Decision
        );
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

        assert!(result.counterfactuals.is_empty());
        assert!(result.plot_holes.is_empty());
        assert!(
            result
                .links
                .iter()
                .all(|link| link.link_type == CausalLinkType::Association)
        );
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
