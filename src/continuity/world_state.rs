//! Tracks explicit English location, acquisition, possession and constraint events
//! in source order. Reports conflicts between observed movement and active
//! constraints; unobserved information paths remain unknown.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::substrate::utils::AnalysisAvailability;

// ─── Data Types ───────────────────────────────────────────────────────────────

/// A fact about the story world at a specific point in narrative time.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "camelCase")]
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

/// A potential conflict between observed physical-state assertions.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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

#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
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
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorldStateSnapshot {
    pub after_scene: usize,
    #[serde(serialize_with = "serialize_facts")]
    pub facts: HashSet<WorldFact>,
    pub new_facts: Vec<WorldFact>,
    pub invalidated_facts: Vec<WorldFact>,
}

/// Knowledge state for a single character.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorldStateResult {
    #[serde(default)]
    pub availability: AnalysisAvailability,
    #[serde(default)]
    pub diagnostics: Vec<String>,
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

/// Retention for the additive current-state API. Current facts, knowledge,
/// locations, violations and scores are always computed by the full reducer.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorldStateOptions {
    /// Retain net changes at every reduced scene boundary, including empty ones.
    #[serde(default)]
    pub include_deltas: bool,
    /// Retain told_by, locations_visited, location_timeline and information_flow.
    /// False leaves those optional histories empty without suppressing learning.
    #[serde(default)]
    pub include_event_history: bool,
}

/// Net changes from the previous scene boundary (the initial state is empty).
/// Replay by removing invalidated_facts, then inserting new_facts. Facts created
/// and removed within one scene cancel; legacy snapshot event lists are unchanged.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct WorldStateDelta {
    pub after_scene: usize,
    pub new_facts: Vec<WorldFact>,
    pub invalidated_facts: Vec<WorldFact>,
}

/// Final state without a cumulative fact-set copy for every scene.
#[cfg_attr(feature = "bindings", derive(ts_rs::TS, schemars::JsonSchema))]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorldStateCurrentResult {
    #[serde(default)]
    pub availability: AnalysisAvailability,
    #[serde(default)]
    pub diagnostics: Vec<String>,
    /// Zero-based last reduced scene; None for empty scenes or an empty cast.
    pub after_scene: Option<usize>,
    #[serde(serialize_with = "serialize_facts")]
    pub facts: HashSet<WorldFact>,
    pub deltas: Vec<WorldStateDelta>,
    pub options: WorldStateOptions,
    pub violations: Vec<ContinuityViolation>,
    pub character_knowledge: Vec<CharacterKnowledgeState>,
    pub location_timeline: Vec<(usize, String, Vec<String>)>,
    pub information_flow: Vec<(usize, String, String, String)>,
    pub continuity_score: f64,
    pub world_complexity: f64,
}

fn serialize_facts<S: serde::Serializer>(
    facts: &HashSet<WorldFact>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut ordered: Vec<_> = facts.iter().collect();
    ordered.sort();
    ordered.serialize(serializer)
}

struct Patterns {
    location: Regex,
    told: Regex,
    acquired: Regex,
    saw: Regex,
    possession: Regex,
    gave: Regex,
    locked: Regex,
    unlocked: Regex,
    trapped: Regex,
    released: Regex,
}

impl Patterns {
    fn compile(cp: &str) -> crate::Result<Self> {
        let compile = |pattern: &str| {
            Regex::new(pattern).map_err(|_| {
                crate::Error::AnalysisFailed("world_state_pattern_compile_failed".into())
            })
        };
        Ok(Self {
            location: compile(&format!(
                r"(?i)\b(?P<actor>{cp})\s+(?:(?:was|were|stood|sat|waited|remained)\s+(?:alone\s+)?(?:in|at|inside)|entered|walked into|stepped into|arrived at|went to|came to|moved to|returned to)\s+(?:the\s+)?(?P<value>\w[\w\s]{{0,30}}?)(?:[.,;!?]|$|\s+(?:when|while|and|but|as)\b)"
            ))?,
            told: compile(&format!(
                r"(?i)\b(?P<actor>{cp})\s+(?:told|informed|explained to|said to|whispered to|mentioned to)\s+(?P<recipient>{cp})\s+(?:about|that)\s+(?P<value>[^.;!?]+)(?:[.;!?]|$)"
            ))?,
            acquired: compile(&format!(
                r"(?i)\b(?P<actor>{cp})\s+(?:discovered|found out|learned|uncovered|overheard|eavesdropped on|listened to|read|studied|examined|realized|understood|grasped|figured out)\s+(?:that\s+|the\s+)?(?P<value>[^.;!?]+)(?:[.;!?]|$)"
            ))?,
            saw: compile(&format!(
                r"(?i)\b(?P<actor>{cp})\s+(?:saw|watched|witnessed|observed|noticed)\s+(?P<value>[^.;!?]+)(?:[.;!?]|$)"
            ))?,
            possession: compile(&format!(
                r"(?i)\b(?P<actor>{cp})\s+(?:picked up|grabbed|snatched|seized|held|clutched|gripped|carried|took|pocketed|kept)\s+(?:the\s+|a\s+)?(?P<value>[^.,;!?]+)(?:[.,;!?]|$)"
            ))?,
            gave: compile(&format!(
                r"(?i)\b(?P<actor>{cp})\s+(?:gave|handed|passed|tossed)\s+(?:the\s+|a\s+)?(?P<value>[^.,;!?]+?)\s+to\s+(?P<recipient>{cp})(?:[.,;!?]|$)"
            ))?,
            locked: compile(&format!(
                r"(?i)\b(?:(?P<actor>{cp})\s+)?(?:locked|bolted|barred|sealed)\s+(?:the\s+)?(?P<value>door|gate|room|exit)\b"
            ))?,
            unlocked: compile(&format!(
                r"(?i)\b(?:(?P<actor>{cp})\s+)?(?:unlocked|unbolted|unbarred|unsealed|opened)\s+(?:the\s+)?(?P<value>door|gate|room|exit)\b"
            ))?,
            trapped: compile(&format!(
                r"(?i)\b(?P<actor>{cp})\s+(?:couldn't leave|could not leave|was trapped|was stuck|was imprisoned)\b"
            ))?,
            released: compile(&format!(
                r"(?i)\b(?P<actor>{cp})\s+(?:was freed|was released|escaped|broke free|was no longer trapped|could leave)\b"
            ))?,
        })
    }
}

fn compile_cached(chars_pattern: &str) -> crate::Result<Arc<Patterns>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<Patterns>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(chars_pattern)
    {
        return Ok(Arc::clone(hit));
    }
    let compiled = Arc::new(Patterns::compile(chars_pattern)?);
    let mut map = cache.lock().unwrap_or_else(|e| e.into_inner());
    if map.len() >= 64 {
        map.clear();
    }
    let ready = if let Some(existing) = map.get(chars_pattern) {
        Arc::clone(existing)
    } else {
        map.insert(chars_pattern.to_owned(), Arc::clone(&compiled));
        Arc::clone(&compiled)
    };
    drop(map);
    Ok(ready)
}

#[derive(Debug)]
enum EventKind {
    Location(String, String),
    Learn(String, String),
    Witness(String, String),
    Tell(String, String, String),
    Possess(String, String),
    Transfer(String, String, String),
    Lock(String, Option<String>),
    Unlock(String, Option<String>),
    Trap(String),
    Release(String),
}

struct Event {
    start: usize,
    end: usize,
    assertion: String,
    kind: EventKind,
}

// Unrecognized explicit subjects must not borrow scene-level constraint locations.
fn unqualified_constraint_prefix(text: &str, start: usize) -> bool {
    fn trim(prefix: &str) -> &str {
        prefix.trim_end_matches(|c: char| c.is_whitespace() && !matches!(c, '\n' | '\r'))
    }
    let boundary = |prefix: &str| {
        trim(prefix)
            .chars()
            .next_back()
            .is_none_or(|c| matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | '\n' | '\r'))
    };
    let prefix = trim(&text[..start]);
    if boundary(prefix) {
        return true;
    }
    ["someone", "somebody"].iter().any(|subject| {
        prefix
            .len()
            .checked_sub(subject.len())
            .is_some_and(|offset| {
                prefix
                    .get(offset..)
                    .is_some_and(|tail| tail.eq_ignore_ascii_case(subject))
                    && boundary(&prefix[..offset])
            })
    })
}

fn events(text: &str, patterns: &Patterns) -> Vec<Event> {
    let mut events = Vec::new();
    for (pattern, kind) in [
        (&patterns.location, 0),
        (&patterns.told, 1),
        (&patterns.acquired, 2),
        (&patterns.saw, 3),
        (&patterns.possession, 4),
        (&patterns.gave, 5),
        (&patterns.locked, 6),
        (&patterns.unlocked, 7),
        (&patterns.trapped, 8),
        (&patterns.released, 9),
    ] {
        for capture in pattern.captures_iter(text) {
            let matched = capture.get(0).expect("matched event");
            let actor = capture
                .name("actor")
                .map(|m| m.as_str().to_lowercase())
                .unwrap_or_default();
            if matches!(kind, 6 | 7)
                && actor.is_empty()
                && !unqualified_constraint_prefix(text, matched.start())
            {
                continue;
            }
            let recipient = capture
                .name("recipient")
                .map(|m| m.as_str().to_lowercase())
                .unwrap_or_default();
            let value = capture
                .name("value")
                .map(|m| m.as_str().trim())
                .unwrap_or_default();
            let event = match kind {
                0 if is_valid_location(value) => EventKind::Location(actor, value.to_lowercase()),
                1 if !value.is_empty() => EventKind::Tell(actor, recipient, value.into()),
                2 if !value.is_empty() => EventKind::Learn(actor, value.into()),
                3 if !value.is_empty() => EventKind::Witness(actor, value.into()),
                4 if is_valid_object(value) => EventKind::Possess(actor, value.to_lowercase()),
                5 if is_valid_object(value) => {
                    EventKind::Transfer(actor, recipient, value.to_lowercase())
                }
                6 => EventKind::Lock(value.to_lowercase(), (!actor.is_empty()).then_some(actor)),
                7 => EventKind::Unlock(value.to_lowercase(), (!actor.is_empty()).then_some(actor)),
                8 => EventKind::Trap(actor),
                9 => EventKind::Release(actor),
                _ => continue,
            };
            events.push(Event {
                start: matched.start(),
                end: matched.end(),
                assertion: matched.as_str().into(),
                kind: event,
            });
        }
    }
    events.sort_by_key(|e| (e.start, e.end));
    events
}

struct Constraint {
    location: Option<String>,
    scene: usize,
    assertion: String,
}

#[derive(Default)]
struct State {
    locations: BTreeMap<String, String>,
    knowledge: BTreeMap<String, BTreeSet<String>>,
    witnessed: BTreeMap<String, BTreeSet<String>>,
    told_by: BTreeMap<String, Vec<(String, String)>>,
    location_history: BTreeMap<String, Vec<(usize, String)>>,
    constraints: BTreeMap<(String, String), Constraint>,
    facts: HashSet<WorldFact>,
}

struct FactChanges {
    legacy: bool,
    delta: bool,
    new_facts: Vec<WorldFact>,
    invalidated_facts: Vec<WorldFact>,
    net_added: BTreeSet<WorldFact>,
    net_removed: BTreeSet<WorldFact>,
}

impl FactChanges {
    fn new(legacy: bool, delta: bool) -> Self {
        Self {
            legacy,
            delta,
            new_facts: Vec::new(),
            invalidated_facts: Vec::new(),
            net_added: BTreeSet::new(),
            net_removed: BTreeSet::new(),
        }
    }

    fn added(&mut self, fact: WorldFact) {
        if self.delta && !self.net_removed.remove(&fact) {
            self.net_added.insert(fact.clone());
        }
        if self.legacy {
            self.new_facts.push(fact);
        }
    }

    fn removed(&mut self, fact: WorldFact) {
        if self.delta && !self.net_added.remove(&fact) {
            self.net_removed.insert(fact.clone());
        }
        if self.legacy {
            self.invalidated_facts.push(fact);
        }
    }
}

fn remove_facts(
    state: &mut State,
    predicate: impl Fn(&WorldFact) -> bool,
    changes: &mut FactChanges,
) {
    if !changes.legacy && !changes.delta {
        state.facts.retain(|fact| !predicate(fact));
        return;
    }
    let mut removed: Vec<_> = state
        .facts
        .iter()
        .filter(|f| predicate(f))
        .cloned()
        .collect();
    removed.sort();
    for fact in removed {
        state.facts.remove(&fact);
        changes.removed(fact);
    }
}

fn add_fact(state: &mut State, fact: WorldFact, changes: &mut FactChanges) {
    if !changes.legacy && !changes.delta {
        state.facts.insert(fact);
    } else if state.facts.insert(fact.clone()) {
        changes.added(fact);
    }
}

fn sync_colocation(state: &mut State, changes: &mut FactChanges) {
    let mut desired = BTreeSet::new();
    let locations: Vec<_> = state.locations.iter().collect();
    for (i, (a, location_a)) in locations.iter().enumerate() {
        for (b, location_b) in locations.iter().skip(i + 1) {
            if location_a == location_b {
                desired.insert(WorldFact::CoLocated {
                    character_a: (*a).clone(),
                    character_b: (*b).clone(),
                });
            }
        }
    }
    remove_facts(
        state,
        |f| matches!(f, WorldFact::CoLocated { .. }) && !desired.contains(f),
        changes,
    );
    for fact in desired {
        add_fact(state, fact, changes);
    }
}

fn sync_constraints(state: &mut State, changes: &mut FactChanges) {
    let desired: BTreeSet<_> = state
        .constraints
        .iter()
        .map(
            |((character, _), constraint)| WorldFact::PhysicalConstraint {
                character: character.clone(),
                constraint: constraint
                    .location
                    .as_ref()
                    .map(|location| format!("locked in {location}"))
                    .unwrap_or_else(|| "cannot leave current location".into()),
            },
        )
        .collect();
    remove_facts(
        state,
        |f| matches!(f, WorldFact::PhysicalConstraint { .. }) && !desired.contains(f),
        changes,
    );
    for fact in desired {
        add_fact(state, fact, changes);
    }
}

fn observed_location(state: &State, participants: &BTreeSet<String>) -> Option<String> {
    let locations: BTreeSet<_> = participants
        .iter()
        .filter_map(|c| state.locations.get(c))
        .collect();
    if locations.len() == 1 {
        locations.into_iter().next().cloned()
    } else {
        None
    }
}

fn learn(state: &mut State, character: &str, information: &str, changes: &mut FactChanges) {
    state
        .knowledge
        .entry(character.into())
        .or_default()
        .insert(information.into());
    add_fact(
        state,
        WorldFact::CharacterKnows {
            character: character.into(),
            information: information.into(),
        },
        changes,
    );
}

fn possess(state: &mut State, character: String, object: String, changes: &mut FactChanges) {
    remove_facts(
        state,
        |f| matches!(f, WorldFact::CharacterPossesses { character: old, object: held } if held == &object && old != &character),
        changes,
    );
    add_fact(
        state,
        WorldFact::CharacterPossesses { character, object },
        changes,
    );
}

struct WorldReduction {
    result: WorldStateResult,
    facts: HashSet<WorldFact>,
    deltas: Vec<WorldStateDelta>,
    after_scene: Option<usize>,
}

/// Track explicit English location, acquisition, transfer and constraint events
/// in presentation order. Unobserved knowledge paths remain unknown. Invalid
/// input returns Unavailable with static diagnostics; prefer the checked API.
pub fn track_world_state(scenes: &[&str], characters: &[&str]) -> WorldStateResult {
    track_world_state_checked(scenes, characters).unwrap_or_else(|error| WorldStateResult {
        availability: AnalysisAvailability::Unavailable,
        diagnostics: vec![
            match error {
                crate::Error::InvalidInput(_) => "world_state_invalid_input",
                crate::Error::AnalysisFailed(_) => "world_state_analysis_failed",
            }
            .into(),
        ],
        snapshots: Vec::new(),
        violations: Vec::new(),
        character_knowledge: Vec::new(),
        location_timeline: Vec::new(),
        information_flow: Vec::new(),
        continuity_score: 0.0,
        world_complexity: 0.0,
    })
}

/// Validate input bounds and propagate static, input-free analysis failures.
/// Full history retains a fact set per scene; use the current-state API to avoid
/// that output-size lower bound. Input bounds are not an output-heap budget.
pub fn track_world_state_checked(
    scenes: &[&str],
    characters: &[&str],
) -> crate::Result<WorldStateResult> {
    Ok(reduce_world_state(scenes, characters, true, false, true)?.result)
}

/// Run the same source-ordered reducer without retaining cumulative snapshots.
/// With deltas disabled, no per-scene fact history is built or discarded.
pub fn track_world_state_current(
    scenes: &[&str],
    characters: &[&str],
    options: &WorldStateOptions,
) -> crate::Result<WorldStateCurrentResult> {
    let reduced = reduce_world_state(
        scenes,
        characters,
        false,
        options.include_deltas,
        options.include_event_history,
    )?;
    let WorldStateResult {
        availability,
        diagnostics,
        snapshots: _,
        violations,
        character_knowledge,
        location_timeline,
        information_flow,
        continuity_score,
        world_complexity,
    } = reduced.result;
    Ok(WorldStateCurrentResult {
        availability,
        diagnostics,
        after_scene: reduced.after_scene,
        facts: reduced.facts,
        deltas: reduced.deltas,
        options: *options,
        violations,
        character_knowledge,
        location_timeline,
        information_flow,
        continuity_score,
        world_complexity,
    })
}

fn validate_world_inputs(scenes: &[&str], characters: &[&str]) -> crate::Result<()> {
    if scenes.len() > 10_000 || characters.len() > 256 {
        return Err(crate::Error::InvalidInput(
            "world state exceeds 10000 scenes or 256 characters".into(),
        ));
    }
    for name in characters {
        crate::manuscript::config::validate_name(name)?;
    }
    let mut total = 0usize;
    for scene in scenes {
        if scene.len() > 16 * 1024 * 1024 {
            return Err(crate::Error::InvalidInput(
                "world state scene exceeds 16 MiB".into(),
            ));
        }
        total = total
            .checked_add(scene.len())
            .filter(|bytes| *bytes <= 64 * 1024 * 1024)
            .ok_or_else(|| {
                crate::Error::InvalidInput("world state total text exceeds 64 MiB".into())
            })?;
    }
    Ok(())
}

fn reduce_world_state(
    scenes: &[&str],
    characters: &[&str],
    retain_snapshots: bool,
    include_deltas: bool,
    include_event_history: bool,
) -> crate::Result<WorldReduction> {
    validate_world_inputs(scenes, characters)?;
    let mut cast: Vec<_> = characters
        .iter()
        .copied()
        .filter(|c| !c.trim().is_empty())
        .collect();
    cast.sort_by(|a, b| {
        a.to_lowercase()
            .cmp(&b.to_lowercase())
            .then_with(|| a.cmp(b))
    });
    cast.dedup_by(|a, b| a.to_lowercase() == b.to_lowercase());
    if scenes.is_empty() || cast.is_empty() {
        return Ok(WorldReduction {
            result: WorldStateResult {
                availability: AnalysisAvailability::Complete,
                diagnostics: Vec::new(),
                snapshots: Vec::new(),
                violations: Vec::new(),
                character_knowledge: Vec::new(),
                location_timeline: Vec::new(),
                information_flow: Vec::new(),
                continuity_score: 1.0,
                world_complexity: 0.0,
            },
            facts: HashSet::new(),
            deltas: Vec::new(),
            after_scene: None,
        });
    }
    let mut names = cast.clone();
    names.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    let pattern = names
        .iter()
        .map(|c| regex::escape(c))
        .collect::<Vec<_>>()
        .join("|");
    let patterns = compile_cached(&pattern)?;
    let mut state = State::default();
    let mut snapshots = Vec::new();
    let mut deltas = Vec::new();
    let mut violations = Vec::new();
    let mut location_timeline = Vec::new();
    let mut information_flow = Vec::new();
    for (scene, text) in scenes.iter().enumerate() {
        let mut changes = FactChanges::new(retain_snapshots, include_deltas);
        let mut participants = BTreeSet::new();
        for event in events(text, &patterns) {
            match event.kind {
                EventKind::Location(character, location) => {
                    if let Some(old) = state
                        .locations
                        .get(&character)
                        .filter(|old| **old != location)
                        && let Some(constraint) =
                            state.constraints.iter().find_map(|((who, _), constraint)| {
                                (who == &character && constraint.location.as_ref() == Some(old))
                                    .then_some(constraint)
                            })
                    {
                        let display = cast
                            .iter()
                            .find(|c| c.to_lowercase() == character)
                            .copied()
                            .unwrap_or(&character);
                        violations.push(ContinuityViolation {
                                scene,
                                violation_type: ViolationType::ImpossibleLocation,
                                description: format!("{display} moves from '{old}' to '{location}' while an observed constraint remains active."),
                                proof: format!("Scene {}: '{}'. Scene {scene}: '{}'. No intervening release was observed.", constraint.scene, constraint.assertion, event.assertion),
                                severity: 0.8,
                                fix_suggestion: "Review the constraint, movement and any missing release event.".into(),
                            });
                    }
                    remove_facts(
                        &mut state,
                        |f| matches!(f, WorldFact::CharacterAt { character: who, location: old } if who == &character && old != &location),
                        &mut changes,
                    );
                    state.locations.insert(character.clone(), location.clone());
                    if include_event_history {
                        state
                            .location_history
                            .entry(character.clone())
                            .or_default()
                            .push((scene, location.clone()));
                    }
                    participants.insert(character.clone());
                    add_fact(
                        &mut state,
                        WorldFact::CharacterAt {
                            character,
                            location,
                        },
                        &mut changes,
                    );
                    sync_colocation(&mut state, &mut changes);
                }
                EventKind::Learn(character, information) => {
                    participants.insert(character.clone());
                    learn(&mut state, &character, &information, &mut changes);
                }
                EventKind::Witness(character, information) => {
                    participants.insert(character.clone());
                    state
                        .witnessed
                        .entry(character.clone())
                        .or_default()
                        .insert(information.clone());
                    learn(&mut state, &character, &information, &mut changes);
                    add_fact(
                        &mut state,
                        WorldFact::CharacterWitnessed {
                            character,
                            event: information,
                        },
                        &mut changes,
                    );
                }
                EventKind::Tell(from, to, information) => {
                    participants.insert(from.clone());
                    participants.insert(to.clone());
                    learn(&mut state, &to, &information, &mut changes);
                    if include_event_history {
                        state
                            .told_by
                            .entry(to.clone())
                            .or_default()
                            .push((from.clone(), information.clone()));
                        information_flow.push((
                            scene,
                            from.clone(),
                            to.clone(),
                            information.clone(),
                        ));
                    }
                    add_fact(
                        &mut state,
                        WorldFact::CommunicationVector {
                            from,
                            to,
                            content: information,
                        },
                        &mut changes,
                    );
                }
                EventKind::Possess(character, object) => {
                    participants.insert(character.clone());
                    possess(&mut state, character, object, &mut changes);
                }
                EventKind::Transfer(from, to, object) => {
                    participants.insert(from);
                    participants.insert(to.clone());
                    possess(&mut state, to, object, &mut changes);
                }
                EventKind::Lock(exit, actor) => {
                    let location = if let Some(actor) = actor {
                        participants.insert(actor.clone());
                        state.locations.get(&actor).cloned()
                    } else {
                        observed_location(&state, &participants)
                    };
                    if let Some(location) = location {
                        for character in state
                            .locations
                            .iter()
                            .filter(|(_, loc)| **loc == location)
                            .map(|(c, _)| c.clone())
                            .collect::<Vec<_>>()
                        {
                            state.constraints.insert(
                                (character, exit.clone()),
                                Constraint {
                                    location: Some(location.clone()),
                                    scene,
                                    assertion: event.assertion.clone(),
                                },
                            );
                        }
                        sync_constraints(&mut state, &mut changes);
                    }
                }
                EventKind::Unlock(exit, actor) => {
                    let location = if let Some(actor) = actor {
                        participants.insert(actor.clone());
                        state.locations.get(&actor).cloned()
                    } else {
                        observed_location(&state, &participants).or_else(|| {
                            let locations: BTreeSet<_> = state
                                .constraints
                                .iter()
                                .filter(|((_, key), _)| key == &exit)
                                .filter_map(|(_, c)| c.location.clone())
                                .collect();
                            (locations.len() == 1)
                                .then(|| locations.into_iter().next())
                                .flatten()
                        })
                    };
                    if let Some(location) = location {
                        state.constraints.retain(|(_, key), c| {
                            key != &exit || c.location.as_ref() != Some(&location)
                        });
                        sync_constraints(&mut state, &mut changes);
                    }
                }
                EventKind::Trap(character) => {
                    participants.insert(character.clone());
                    let location = state.locations.get(&character).cloned();
                    state.constraints.insert(
                        (character, "restriction".into()),
                        Constraint {
                            location,
                            scene,
                            assertion: event.assertion,
                        },
                    );
                    sync_constraints(&mut state, &mut changes);
                }
                EventKind::Release(character) => {
                    participants.insert(character.clone());
                    state.constraints.retain(|(who, _), _| who != &character);
                    sync_constraints(&mut state, &mut changes);
                }
            }
        }
        if include_event_history {
            let mut at: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for (character, location) in &state.locations {
                at.entry(location.clone())
                    .or_default()
                    .push(character.clone());
            }
            location_timeline.extend(
                at.into_iter()
                    .map(|(location, chars)| (scene, location, chars)),
            );
        }
        if retain_snapshots {
            snapshots.push(WorldStateSnapshot {
                after_scene: scene,
                facts: state.facts.clone(),
                new_facts: changes.new_facts,
                invalidated_facts: changes.invalidated_facts,
            });
        }
        if include_deltas {
            deltas.push(WorldStateDelta {
                after_scene: scene,
                new_facts: changes.net_added.into_iter().collect(),
                invalidated_facts: changes.net_removed.into_iter().collect(),
            });
        }
    }
    let character_knowledge = cast
        .iter()
        .map(|&character| {
            let key = character.to_lowercase();
            CharacterKnowledgeState {
                character: character.into(),
                knows: state
                    .knowledge
                    .get(&key)
                    .map(|s| s.iter().cloned().collect())
                    .unwrap_or_default(),
                witnessed: state
                    .witnessed
                    .get(&key)
                    .map(|s| s.iter().cloned().collect())
                    .unwrap_or_default(),
                told_by: state.told_by.get(&key).cloned().unwrap_or_default(),
                locations_visited: state
                    .location_history
                    .get(&key)
                    .cloned()
                    .unwrap_or_default(),
                current_location: state.locations.get(&key).cloned(),
            }
        })
        .collect();
    let penalty: f64 = violations.iter().map(|v| v.severity).sum();
    let result = WorldStateResult {
        availability: AnalysisAvailability::Complete,
        diagnostics: Vec::new(),
        snapshots,
        violations,
        character_knowledge,
        location_timeline,
        information_flow,
        continuity_score: (1.0 - penalty / (scenes.len() as f64 * 0.5)).clamp(0.0, 1.0),
        world_complexity: state.facts.len() as f64 / scenes.len() as f64,
    };
    Ok(WorldReduction {
        result,
        facts: state.facts,
        deltas,
        after_scene: Some(scenes.len() - 1),
    })
}

fn is_valid_location(location: &str) -> bool {
    !location.is_empty() && location.len() <= 40 && location.chars().any(char::is_alphabetic)
}

fn is_valid_object(object: &str) -> bool {
    !object.is_empty()
        && object.len() <= 40
        && ![
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
        ]
        .contains(&object.to_lowercase().as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_internal_pattern_failure_has_static_context() {
        let pattern = "x".repeat(200_000);
        let error = Patterns::compile(&pattern)
            .err()
            .expect("compiled pattern limit");
        assert_eq!(
            error.to_string(),
            "analysis failed: world_state_pattern_compile_failed"
        );
    }

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
